""" ============================================================================
py/tests/test_pack_remote.py — 远程数据打包的契约测试（AutoDL 集成 §2/§6）
----------------------------------------------------------------------------
测四个会真出事故的点：
  1) 许可闸门 fail-closed：含「权利不清/仅本地学习」作品的桶默认**拒绝打包**；
  2) 包内身份**自洽**：manifest 是全量的子集，指纹必然 ≠ 全量——包内 report 必须按
     **子集**重算指纹，远端 splits.verify(包内manifest, 包内report) 才能直接通过；
     （这是实测踩过的坑：第一版把全量指纹塞进包里，远端永远对不上）
  3) 远端消费路径可用：解包后 bucket_texts() 只给清单路径就能读（报告跟随同目录）；
  4) 来源指纹记进 PACK_INFO（谱系用），与子集指纹分开、不混用。

运行：
  python -m unittest py.tests.test_pack_remote
============================================================================ """
from __future__ import annotations
import json
import os
import shutil
import sys
import tarfile
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
if PY not in sys.path:
    sys.path.insert(0, PY)

import splits  # noqa: E402
from data import pack_remote  # noqa: E402
from license_policy import classify  # noqa: E402


def _row(sid, h, sp, bucket, lic, text):
    return {'story_id': sid, 'hash': h, 'split': sp, 'bucket': bucket,
            'title': sid, 'license': lic, 'chars': len(text)}


class _Fixture(unittest.TestCase):
    """临时目录里造 2 桶：clean（公版，2 部）与 dirty（权利不清，1 部）。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='pack_')
        self.man = os.path.join(self.tmp, 'manifest.jsonl')
        self.rep = os.path.join(self.tmp, 'split_report.json')
        self.proc = os.path.join(self.tmp, 'processed')
        self._saved = (splits.DEFAULT_MANIFEST, splits.DEFAULT_REPORT,
                       splits.DEDUP_CANDIDATES, splits._resolved, splits._warned)
        splits.DEFAULT_MANIFEST = self.man
        splits.DEFAULT_REPORT = self.rep
        splits.DEDUP_CANDIDATES = []
        splits._resolved = None
        splits._warned = True

        self.texts = {
            ('clean', 'a_1'): '第一段落。第二句话。第三句话。第四句话。',
            ('clean', 'a_2'): '另一部作品的开头。它的第二句。',
            ('dirty', 'b_1'): '权利不清作品的正文。只有一句话。',
        }
        rows = [_row('a_1', 'h1', 'train', 'clean', 'Public Domain', self.texts[('clean', 'a_1')]),
                _row('a_2', 'h2', 'test', 'clean', 'Public Domain', self.texts[('clean', 'a_2')]),
                _row('b_1', 'h3', 'train', 'dirty', '抓取内容权利不清；本地学习用',
                     self.texts[('dirty', 'b_1')])]
        with open(self.man, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        with open(self.rep, 'w', encoding='utf-8') as f:
            json.dump({'split_fingerprint': splits.split_fingerprint(rows)}, f)
        for (b, _sid), t in self.texts.items():
            d = os.path.join(self.proc, b)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, splits.slug(_sid) + '.txt'), 'w', encoding='utf-8') as f:
                f.write(t)

    def tearDown(self):
        (splits.DEFAULT_MANIFEST, splits.DEFAULT_REPORT, splits.DEDUP_CANDIDATES,
         splits._resolved, splits._warned) = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestGate(_Fixture):
    def test_不可分发桶默认拒绝(self):
        out = os.path.join(self.tmp, 'bad.tar')
        with self.assertRaises(SystemExit) as ctx:
            pack_remote.build(['clean', 'dirty'], out, self.man, self.rep, False, self.proc)
        self.assertIn('拒绝打包', str(ctx.exception))
        self.assertFalse(os.path.exists(out))

    def test_显式允许后打包并留痕(self):
        out = os.path.join(self.tmp, 'ok.tar')
        pack_remote.build(['clean', 'dirty'], out, self.man, self.rep, True, self.proc)
        with tarfile.open(out) as t:
            info = json.load(t.extractfile('data/PACK_INFO.json'))
        self.assertTrue(info['allow_non_distributable'])
        self.assertEqual(info['license_summary'].get('local_only'), 1)


class TestPackIdentity(_Fixture):
    def test_包内身份自洽且与来源指纹分开(self):
        out = os.path.join(self.tmp, 'clean.tar')
        r = pack_remote.build(['clean'], out, self.man, self.rep, False, self.proc)
        with tarfile.open(out) as t:
            man = [json.loads(l) for l in
                   t.extractfile('data/manifest.jsonl').read().decode('utf-8').splitlines() if l.strip()]
            rep = json.load(t.extractfile('data/split_report.json'))
            info = json.load(t.extractfile('data/PACK_INFO.json'))
        # 子集指纹 = 子集清单现算指纹（远端 verify 用）
        self.assertEqual(rep['split_fingerprint'], splits.split_fingerprint(man))
        self.assertEqual(info['subset_split_fingerprint'], rep['split_fingerprint'])
        # 来源指纹 ≠ 子集指纹（全量含 dirty 桶），但必须被记录（谱系用）
        self.assertNotEqual(info['source_split_fingerprint'], rep['split_fingerprint'])
        self.assertEqual(info['source_split_fingerprint'], r['source_fingerprint'])
        self.assertEqual(info['license_summary'], {'public_domain': 2})
        self.assertEqual(len(man), 2)          # 只含所选桶

    def test_远端消费路径_解包即可读(self):
        out = os.path.join(self.tmp, 'clean.tar')
        pack_remote.build(['clean'], out, self.man, self.rep, False, self.proc)
        remote = os.path.join(self.tmp, 'remote')
        with tarfile.open(out) as t:
            names = t.getnames()
            t.extractall(remote)
        # 包内布局必须镜像远端目录树：解到工作目录根，data/ 一次落位
        self.assertIn('data/manifest.jsonl', names)
        self.assertIn('data/processed/clean/a_2.txt', names)
        data_dir = os.path.join(remote, 'data')
        bt = splits.bucket_texts('clean', manifest_path=os.path.join(data_dir, 'manifest.jsonl'),
                                 processed_dir=os.path.join(data_dir, 'processed'))
        self.assertEqual({sp: len(bt[sp]) for sp in ('train', 'val', 'test')},
                         {'train': 1, 'val': 0, 'test': 1})
        self.assertEqual(bt['test'][0][2], self.texts[('clean', 'a_2')])


if __name__ == '__main__':
    unittest.main(verbosity=2)
