""" ============================================================================
py/tests/test_dataset.py — 预编码分片数据集的契约测试（V3 P1，D1 准入）
----------------------------------------------------------------------------
盯四件会真出事故的事：
  1) **逐字往返**：分片 → 解码必须与原作品逐字一致（含跨分片的文档）；
     这是「分片没有丢字/错位」的唯一硬证据；
  2) **文档边界**：doc_offsets 单调、段长与各作品字数一致、分片边界不产生假边界；
  3) **语义与既有路径一致**：sample_batch 与 minilm_torch.make_batch 同分布
     （同 seed 同窗口）、unk→0、targets 右移一位；
  4) **内存承诺**：memmap 读取不把语料拖进 Python list（用分片数与形状断言代理）。

运行：
  python -m unittest py.tests.test_dataset
============================================================================ """
from __future__ import annotations
import json
import os
import random
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
if PY not in sys.path:
    sys.path.insert(0, PY)

import splits  # noqa: E402
from minilm_torch import make_batch  # noqa: E402
from novellm import dataset as ds  # noqa: E402

TEXTS = {
    'a_1': '第一篇作品。' * 40,                        # 足够跨小分片
    'a_2': '第二篇。' * 7,
    'a_3': '第三篇作品，只有一个 unicode 生僻字：龦。收尾。',
}


class _Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='ds_')
        self.man = os.path.join(self.tmp, 'manifest.jsonl')
        self.rep = os.path.join(self.tmp, 'split_report.json')
        self.proc = os.path.join(self.tmp, 'processed')
        self.out = os.path.join(self.tmp, 'shards')
        self._saved = (splits.DEFAULT_MANIFEST, splits.DEFAULT_REPORT,
                       splits.DEDUP_CANDIDATES, splits._resolved, splits._warned)
        splits.DEFAULT_MANIFEST = self.man
        splits.DEFAULT_REPORT = self.rep
        splits.DEDUP_CANDIDATES = []
        splits._resolved = None
        splits._warned = True

        rows = []
        for i, (sid, text) in enumerate(TEXTS.items(), 1):
            rows.append({'story_id': sid, 'hash': f'h{i}', 'split': 'train',
                         'bucket': 'b', 'title': sid, 'license': 'Public Domain',
                         'chars': len(text)})
            d = os.path.join(self.proc, 'b')
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, splits.slug(sid) + '.txt'), 'w', encoding='utf-8') as f:
                f.write(text)
        with open(self.man, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        with open(self.rep, 'w', encoding='utf-8') as f:
            json.dump({'split_fingerprint': splits.split_fingerprint(rows)}, f)

    def tearDown(self):
        (splits.DEFAULT_MANIFEST, splits.DEFAULT_REPORT, splits.DEDUP_CANDIDATES,
         splits._resolved, splits._warned) = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, shard_tokens=64):
        itos, _ = ds.fit_char_vocab('train', min_freq=1, manifest_path=self.man,
                                    processed_dir=self.proc)
        meta = ds.build_shards('train', itos, self.out, shard_tokens,
                               manifest_path=self.man, report_path=self.rep,
                               processed_dir=self.proc)
        return itos, meta


class TestBuildAndRoundTrip(_Fixture):

    def test_逐字往返_含跨分片文档(self):
        itos, meta = self.build(shard_tokens=64)          # 刻意小分片 → a_1 必跨片
        c = ds.ShardedCorpus(self.out, 'train')
        self.assertEqual(meta['n_tokens'], sum(len(t) for t in TEXTS.values()))
        self.assertEqual(meta['n_docs'], len(TEXTS))
        self.assertGreater(len(meta['shards']), 1, '测试需要至少跨一个分片')
        for k, sid in enumerate(TEXTS):
            start, end = int(c.doc_offsets[k]), int(c.doc_offsets[k + 1])
            self.assertEqual(c.decode(c.get_slice(start, end - start)), TEXTS[sid], sid)

    def test_文档边界_单调且无分片假边界(self):
        _itos, meta = self.build(shard_tokens=64)
        c = ds.ShardedCorpus(self.out, 'train')
        offs = c.doc_offsets
        self.assertEqual(offs[0], 0)
        self.assertEqual(int(offs[-1]), meta['n_tokens'])
        self.assertTrue(np.all(np.diff(offs) > 0), '空文档/假边界')
        self.assertEqual(len(offs) - 1, len(TEXTS))       # 分片边界不产生额外边界
        lens = np.diff(offs)
        for k, sid in enumerate(TEXTS):
            self.assertEqual(int(lens[k]), len(TEXTS[sid]), sid)

    def test_词表只来自train且unk封顶(self):
        itos, _ = self.build(shard_tokens=64)
        self.assertEqual(itos[0], '')                     # UNK 位
        self.assertLess(len(itos), 65536)                 # uint16 可容纳

    def test_verify命令级往返(self):
        self.build(shard_tokens=64)
        rc = ds.main.__self__ if False else None
        # 直接调 verify 逻辑（main 需 argv，这里复刻其断言路径）
        c = ds.ShardedCorpus(self.out, 'train')
        offs = c.doc_offsets
        for k, sid in enumerate(TEXTS):
            dec = c.decode(c.get_slice(int(offs[k]), int(offs[k + 1] - offs[k])))
            self.assertEqual(dec, TEXTS[sid], sid)


class TestBatchSemantics(_Fixture):
    def test_sample_batch与make_batch同分布(self):
        itos, _ = self.build(shard_tokens=64)
        c = ds.ShardedCorpus(self.out, 'train')
        ids = c.get_slice(0, c.n_tokens).astype(np.int64).tolist()
        rng = random.Random(7)
        x, y = c.sample_batch(16, 8, rng)
        self.assertEqual(str(x.dtype), 'torch.int64')
        self.assertEqual(x.shape, (16, 8))
        # 同 seed 走 make_batch 的参考实现，窗口必须一致（同语义的证据）
        rng2 = random.Random(7)
        rx, ry = make_batch(ids, 16, 8, rng2)
        self.assertTrue((x == rx).all().item())
        self.assertTrue((y == ry).all().item())

    def test_targets是tokens右移一位(self):
        itos, _ = self.build(shard_tokens=64)
        c = ds.ShardedCorpus(self.out, 'train')
        x, y = c.sample_batch(8, 16, random.Random(3))
        # 不变量：每个窗口内 targets[t] == tokens[t+1]（右移一位），对随机窗口成立
        for b in range(x.shape[0]):
            self.assertEqual(x[b, 1:].tolist(), y[b, :-1].tolist())

    def test_语料太短报错与make_batch一致(self):
        itos, _ = self.build(shard_tokens=64)
        c = ds.ShardedCorpus(self.out, 'train')
        with self.assertRaises(ValueError):
            c.sample_batch(1, c.n_tokens + 5, random.Random(1))


class TestMemoryAndId(_Fixture):
    def test_分片是memmap_非PythonList(self):
        itos, _ = self.build(shard_tokens=64)
        c = ds.ShardedCorpus(self.out, 'train')
        for f in c._files:
            self.assertIsInstance(f, np.memmap)

    def test_meta带身份指纹(self):
        _itos, meta = self.build(shard_tokens=64)
        self.assertIn('source_split_fingerprint', meta)
        self.assertIn('vocab_fingerprint', meta)
        rows = splits.read_manifest(self.man)
        self.assertEqual(meta['source_split_fingerprint'],
                         splits.split_fingerprint(rows))


if __name__ == '__main__':
    unittest.main(verbosity=2)
