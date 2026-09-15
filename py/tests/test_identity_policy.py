""" ============================================================================
py/tests/test_identity_policy.py — 许可分级、清单择优、run 配置指纹的契约测试
----------------------------------------------------------------------------
盯三件 V3 P0 出口条件里可机械判定的事：
  1) license_policy：**混合许可必须取更严的一档**，认不出来的落 unknown 且不可分发
     （宁可漏报成「不能发布」，也不能误报成「可以发布」）。
  2) splits.resolve_current：**过期快照绝不能当默认**。
     实测踩过：v2 是 32,036 部时期的快照，v1 已涨到 192,823 部，
     「v2 存在就优先用 v2」会静默丢掉 16 万部数据。
  3) run_identity：同一配置+同一数据指纹稳定；数据换版则身份必变
     （否则两条不同数据上的结果在仓库里长得一样，无法区分）。

运行：
  python -m unittest py.tests.test_identity_policy
============================================================================ """
from __future__ import annotations
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
if PY not in sys.path:
    sys.path.insert(0, PY)

import splits  # noqa: E402
from license_policy import (DISTRIBUTABLE, assert_distributable, classify,  # noqa: E402
                            summarise)
from run_identity import data_identity, run_identity  # noqa: E402


def row(sid, h, sp='train', bucket='b', lic=None, **extra):
    return dict({'story_id': sid, 'hash': h, 'split': sp, 'bucket': bucket,
                 'title': sid, 'license': lic, 'chars': 10}, **extra)


class TestLicensePolicy(unittest.TestCase):
    def test_混合许可取更严的一档(self):
        # 「公版古籍 + CC BY-NC-SA」不能因为出现"公版"就判成可自由分发
        self.assertEqual(classify('仓库注明来源（公版古籍，CC BY-NC-SA4.0 仓库）'), 'noncommercial')
        # 「CC BY-SA / Public Domain」的署名与相同方式共享义务不能被抹掉
        self.assertEqual(classify('CC BY-SA 4.0 / Public Domain (中文维基文库)'),
                         'attribution_sharealike')
        # 权利不清优先于 MIT/Apache 标签（标签不覆盖原作者权利）
        self.assertEqual(classify('HF 仓库 MIT；抓取内容作者权利不清；本地学习用'), 'local_only')
        self.assertEqual(classify('Apache-2.0（标签不覆盖原著作者权利；本地学习用）'), 'local_only')

    def test_干净许可正确分档(self):
        self.assertEqual(classify('Public Domain (Project Gutenberg)'), 'public_domain')
        self.assertEqual(classify('CDLA-Sharing-1.0（合成数据）'), 'synthetic')
        self.assertEqual(classify('CC BY 4.0（署名：某某）'), 'attribution')

    def test_认不出来落unknown且不可分发(self):
        for s in ('按仓库说明', '按仓库说明（未明确标注）', '', None):
            self.assertEqual(classify(s), 'unknown', s)
        self.assertNotIn('unknown', DISTRIBUTABLE)
        self.assertNotIn('local_only', DISTRIBUTABLE)
        self.assertNotIn('noncommercial', DISTRIBUTABLE)

    def test_闸门只放过全部可分发(self):
        ok = [row('a', 'h1', lic='Public Domain'), row('b', 'h2', lic='CDLA-Sharing-1.0')]
        self.assertEqual(summarise(ok)['public_domain']['works'], 1)
        assert_distributable(ok)                       # 不抛
        bad = ok + [row('c', 'h3', lic='权利不清；本地学习用')]
        with self.assertRaises(ValueError) as ctx:
            assert_distributable(bad, what='导出权重')
        msg = str(ctx.exception)
        self.assertIn('导出权重', msg)
        self.assertIn('local_only', msg)


class _ManifestFixture(unittest.TestCase):
    """在临时目录里搭一套 manifest/report，并把 splits 模块的路径常量指过去。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='ident_')
        self._saved = (splits.DEFAULT_MANIFEST, splits.DEFAULT_REPORT,
                       splits.DEDUP_CANDIDATES, splits._resolved, splits._warned)
        splits.DEFAULT_MANIFEST = os.path.join(self.tmp, 'manifest.jsonl')
        splits.DEFAULT_REPORT = os.path.join(self.tmp, 'split_report.json')
        # 候选表换成临时目录里的一代去重产物（v2 命名只是沿袭仓库习惯）
        self.v2m = os.path.join(self.tmp, 'manifest_v2.jsonl')
        self.v2r = os.path.join(self.tmp, 'split_report_v2.json')
        splits.DEDUP_CANDIDATES = [(self.v2m, self.v2r)]
        splits._resolved = None
        splits._warned = True            # 测试里不要往 stderr 刷告警

    def tearDown(self):
        (splits.DEFAULT_MANIFEST, splits.DEFAULT_REPORT, splits.DEDUP_CANDIDATES,
         splits._resolved, splits._warned) = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, path, rows):
        with open(path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    def write_report(self, path, rows):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'split_fingerprint': splits.split_fingerprint(rows)}, f)


class TestResolveCurrent(_ManifestFixture):
    def test_显式路径优先(self):
        m, r = splits.resolve_current('data/manifest_v2.jsonl', 'x/report.json')
        self.assertEqual((m, r), ('data/manifest_v2.jsonl', 'x/report.json'))

    def test_新鲜且已合并的v2才被选为默认(self):
        v1 = [row('b_1', 'h1'), row('b_2', 'h2', 'test')]
        v2 = [row('b_1', 'h1', dedup_cluster='c1'), row('b_2', 'h2', 'test', dedup_cluster='c1')]
        self.write(splits.DEFAULT_MANIFEST, v1)
        self.write(self.v2m, v2)
        self.write_report(splits.DEFAULT_REPORT, v1)
        self.write_report(self.v2r, v2)
        m, r = splits.resolve_current()
        self.assertEqual((m, r), (self.v2m, self.v2r))

    def test_过期快照绝不当默认(self):
        # v1 已涨到 3 部，v2 只有 2 部（真实情况：192,823 vs 32,036）
        v1 = [row('b_1', 'h1'), row('b_2', 'h2', 'test'), row('b_3', 'h3')]
        v2 = [row('b_1', 'h1'), row('b_2', 'h2', 'test')]
        self.write(splits.DEFAULT_MANIFEST, v1)
        self.write(self.v2m, v2)
        self.write_report(splits.DEFAULT_REPORT, v1)
        self.write_report(self.v2r, v2)
        m, _ = splits.resolve_current()
        self.assertEqual(m, splits.DEFAULT_MANIFEST)

    def test_没有去重候选也正常回退(self):
        splits.DEDUP_CANDIDATES = []
        v1 = [row('b_1', 'h1'), row('b_2', 'h2', 'test')]
        self.write(splits.DEFAULT_MANIFEST, v1)
        self.write_report(splits.DEFAULT_REPORT, v1)
        m, r = splits.resolve_current()
        self.assertEqual((m, r), (splits.DEFAULT_MANIFEST, splits.DEFAULT_REPORT))


class TestRunIdentity(_ManifestFixture):
    def _setup(self, rows, cluster=False):
        rows = [dict(r, dedup_cluster='c1') for r in rows] if cluster else rows
        self.write(splits.DEFAULT_MANIFEST, rows)
        self.write_report(splits.DEFAULT_REPORT, rows)
        return rows

    def test_数据身份带出指纹与合并状态(self):
        self._setup([row('b_1', 'h1', lic='Public Domain'),
                     row('b_2', 'h2', 'test', lic='权利不清；本地学习用')])
        did = data_identity()
        self.assertEqual(did['works'], 2)
        self.assertEqual(did['near_dup_merged'], False)
        self.assertEqual(did['license']['local_only']['works'], 1)
        self.assertEqual(did['split_fingerprint'], splits.split_fingerprint(
            splits.read_manifest(splits.DEFAULT_MANIFEST)))

    def test_合并过的清单要如实标出(self):
        self._setup([row('b_1', 'h1'), row('b_2', 'h2', 'test')], cluster=True)
        self.assertEqual(data_identity()['near_dup_merged'], True)

    def test_身份稳定且随数据换版而变(self):
        rows = self._setup([row('b_1', 'h1'), row('b_2', 'h2', 'test')])
        cfg, seed, steps, batch, lr = dict(d=8, layers=2), 42, 100, 8, 0.004
        d1 = data_identity()
        i1 = run_identity(cfg, seed, steps, batch, lr, d1)
        self.assertEqual(i1, run_identity(cfg, seed, steps, batch, lr, data_identity()))
        # 同一批作品、换了 split 归属 → 指纹必须变
        moved = [dict(r) for r in rows]
        moved[1]['split'] = 'train'
        self.write(splits.DEFAULT_MANIFEST, moved)
        self.write_report(splits.DEFAULT_REPORT, moved)
        splits._resolved = None
        i2 = run_identity(cfg, seed, steps, batch, lr, data_identity())
        self.assertNotEqual(i1, i2, '数据换版后 run 身份没变 → 结果无法区分是哪版数据')


if __name__ == '__main__':
    unittest.main(verbosity=2)
