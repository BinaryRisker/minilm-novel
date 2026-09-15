""" ============================================================================
py/tests/test_locked_test.py — 锁定测试族的契约测试（V3 P0 第 4 条）
----------------------------------------------------------------------------
锁定测试族的价值全在「断言它还是当年冻的那一批」上，所以测试盯三件事：
  1) 冻结是**确定**的：同一批数据重冻，指纹逐位相同（否则"锁定"无意义）；
     且换用 story_id 重排序也不变——身份按内容哈希，不按编号。
  2) **漂移必须被抓到**：锁定作品的 split 归属变了 / 作品消失了 / 锁定文件被手改，
     三种都要非零失败。清单换代（指纹变了）本身不算失败，但必须显式注记。
  3) 兼容 py/splits.py 的择优：显式传入的清单路径优先，不被 resolve_current 干扰。

运行：
  python -m unittest py.tests.test_locked_test
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

import locked_test  # noqa: E402
import splits  # noqa: E402


def row(sid, h, sp='test', bucket='b', chars=100):
    return {'story_id': sid, 'hash': h, 'split': sp, 'bucket': bucket,
            'title': sid, 'license': 'Public Domain', 'chars': chars}


class _Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='lock_')
        self.man = os.path.join(self.tmp, 'manifest.jsonl')
        self.rep = os.path.join(self.tmp, 'split_report.json')
        self.lock = os.path.join(self.tmp, 'locked_test.json')
        self._saved = (splits.DEFAULT_MANIFEST, splits.DEFAULT_REPORT,
                       splits.DEDUP_CANDIDATES, splits._resolved, splits._warned)
        splits.DEFAULT_MANIFEST = self.man
        splits.DEFAULT_REPORT = self.rep
        splits.DEDUP_CANDIDATES = []
        splits._resolved = None
        splits._warned = True
        self.rows = [row('b_1', 'h1'), row('b_2', 'h2'),
                     row('b_3', 'h3', 'train'), row('b_4', 'h4', 'val')]
        self.write_manifest(self.rows)
        self.write_report(self.rows)

    def tearDown(self):
        (splits.DEFAULT_MANIFEST, splits.DEFAULT_REPORT, splits.DEDUP_CANDIDATES,
         splits._resolved, splits._warned) = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_manifest(self, rows):
        with open(self.man, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    def write_report(self, rows):
        with open(self.rep, 'w', encoding='utf-8') as f:
            json.dump({'split_fingerprint': splits.split_fingerprint(rows)}, f)


class TestFreeze(_Fixture):
    def test_只从test取且确定可复现(self):
        d1 = locked_test.freeze(2, out_path=self.lock)
        self.assertEqual([w['hash'] for w in d1['works']], ['h1', 'h2'])  # train/val 不入锁
        d2 = locked_test.freeze(2, out_path=os.path.join(self.tmp, 'again.json'))
        self.assertEqual(d1['lock_fingerprint'], d2['lock_fingerprint'])

    def test_清单重排不改指纹(self):
        locked_test.freeze(2, out_path=self.lock)
        fp1 = locked_test.load_lock(self.lock)['lock_fingerprint']
        reordered = list(reversed(self.rows))
        self.write_manifest(reordered)          # 行序变了（mtime 也变了）
        self.write_report(reordered)
        splits._resolved = None
        d2 = locked_test.freeze(2, out_path=self.lock)
        self.assertEqual(fp1, d2['lock_fingerprint'])

    def test_per_bucket为0拒绝冻结(self):
        with self.assertRaises(ValueError):
            locked_test.freeze(0, out_path=self.lock)


class TestVerify(_Fixture):
    def setUp(self):
        super().setUp()
        locked_test.freeze(2, out_path=self.lock)

    def test_未动过则通过(self):
        r = locked_test.verify(self.lock)
        self.assertEqual(r['works'], 2)
        self.assertFalse(r['manifest_changed_since_freeze'])

    def test_锁定作品被挪出test必须失败(self):
        self.rows[0]['split'] = 'train'          # h1: test → train
        self.write_manifest(self.rows)
        self.write_report(self.rows)
        splits._resolved = None
        with self.assertRaises(locked_test.LockDriftError) as ctx:
            locked_test.verify(self.lock)
        self.assertIn('归属变了', str(ctx.exception))

    def test_锁定作品消失必须失败(self):
        kept = [r for r in self.rows if r['hash'] != 'h1']
        self.write_manifest(kept)
        self.write_report(kept)
        splits._resolved = None
        with self.assertRaises(locked_test.LockDriftError) as ctx:
            locked_test.verify(self.lock)
        self.assertIn('已不存在', str(ctx.exception))

    def test_锁定文件被手改必须失败(self):
        doc = locked_test.load_lock(self.lock)
        doc['works'][0]['chars'] = 99999          # 偷改内容
        with open(self.lock, 'w', encoding='utf-8') as f:
            json.dump(doc, f)
        with self.assertRaises(locked_test.LockDriftError) as ctx:
            locked_test.verify(self.lock)
        self.assertIn('被改动', str(ctx.exception))

    def test_清单换代但锁定族未受影响_注记不失败(self):
        extra = self.rows + [row('b_9', 'h9', 'train')]
        self.write_manifest(extra)                # 清单变了（指纹变），锁定族原样
        self.write_report(extra)
        splits._resolved = None
        r = locked_test.verify(self.lock)
        self.assertTrue(r['manifest_changed_since_freeze'])   # 注记，但不失败


if __name__ == '__main__':
    unittest.main(verbosity=2)
