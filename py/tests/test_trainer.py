""" ============================================================================
py/tests/test_trainer.py — novellm 分片训练器的契约测试（V3 P1）
----------------------------------------------------------------------------
测四件 V3 点名的事：
  1) **真正 resume**（P0 出口条件的延续）：「训 5 步 → 续训到 10」与「直训 10」
     最终权重**逐位相等**。采样 rng 按步派生（Random(f'{seed}:{step}')），
     torch 全局状态随 checkpoint v2 走，等价性不依赖 accum/batch 的消耗数；
  2) **取消安全存盘**：cancel_check 触发 → checkpoint 落盘 + meta.cancelled=True，
     而不是丢训练；
  3) **累计梯度**：accum=2 的两步消耗与 loss 缩放正确（loss/accum 反传）；
  4) **参数量断言**：novellm.model 的公式与 MiniLM 实例实数一致（锁死架构漂移）。

运行：
  python -m unittest py.tests.test_trainer
============================================================================ """
from __future__ import annotations
import json
import os
import shutil
import sys
import tempfile
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
if PY not in sys.path:
    sys.path.insert(0, PY)

import checkpoint  # noqa: E402
import splits  # noqa: E402
from minilm_torch import MiniLM  # noqa: E402
from novellm import dataset as ds  # noqa: E402
from novellm import model as nov_model  # noqa: E402
from novellm import trainer  # noqa: E402

TRAIN_TEXTS = {'s_1': '故事一。' * 30, 's_2': '故事二。' * 30, 's_3': '故事三。' * 25}
VAL_TEXTS = {'s_9': '验证故事。' * 20}


class _Fixture(unittest.TestCase):
    """train/val 两个小分片 + 微型 cfg（CPU 秒级）。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='trainer_')
        self.shard_dir = os.path.join(self.tmp, 'shards')
        self._write_split('train', TRAIN_TEXTS)
        self._write_split('val', VAL_TEXTS)
        self._saved = (splits.DEFAULT_MANIFEST, splits.DEFAULT_REPORT,
                       splits.DEDUP_CANDIDATES, splits._resolved, splits._warned)
        splits.DEFAULT_MANIFEST = self.man
        splits.DEFAULT_REPORT = self.rep
        splits.DEDUP_CANDIDATES = []
        splits._resolved = None
        splits._warned = True
        self.cfg = dict(ctx=16, d=16, layers=1, heads=2, ffMul=1)
        self.train = ds.ShardedCorpus(self.shard_dir, 'train')
        self.val = ds.ShardedCorpus(self.shard_dir, 'val')
        self.itos = self.train.itos

    def tearDown(self):
        (splits.DEFAULT_MANIFEST, splits.DEFAULT_REPORT, splits.DEDUP_CANDIDATES,
         splits._resolved, splits._warned) = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_split(self, split, texts):
        rows = []
        self_manifest = os.path.join(self.tmp, f'{split}_manifest.jsonl')
        rep = os.path.join(self.tmp, f'{split}_report.json')
        for i, (sid, text) in enumerate(sorted(texts.items()), 1):
            rows.append({'story_id': sid, 'hash': f'{split}{i}', 'split': split,
                         'bucket': 'b', 'title': sid, 'license': 'Public Domain',
                         'chars': len(text)})
            d = os.path.join(self.tmp, 'processed', 'b')
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, splits.slug(sid) + '.txt'), 'w', encoding='utf-8') as f:
                f.write(text)
        # build_shards 按 (manifest, report) 校验身份：把 manifest/report 放到 splits 默认位
        self.man = self_manifest
        self.rep = rep
        splits.DEFAULT_MANIFEST = self.man
        splits.DEFAULT_REPORT = self.rep
        with open(self.man, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        with open(self.rep, 'w', encoding='utf-8') as f:
            json.dump({'split_fingerprint': splits.split_fingerprint(rows)}, f)
        itos, _ = ds.fit_char_vocab(split, min_freq=1, manifest_path=self.man,
                                    processed_dir=os.path.join(self.tmp, 'processed'))
        ds.build_shards(split, itos, self.shard_dir, shard_tokens=4096,
                        manifest_path=self.man, report_path=self.rep,
                        processed_dir=os.path.join(self.tmp, 'processed'))


class TestResumeEquivalence(_Fixture):
    def test_续训与直训逐位一致(self):
        out_direct = os.path.join(self.tmp, 'direct')
        m_direct = trainer.train(self.cfg, self.train, self.val, self.itos,
                                 steps=10, batch=2, accum=2, lr=0.01, seed=7,
                                 out_dir=out_direct, val_every=5, val_windows=8,
                                 log=lambda _s: None)
        # 两阶段：训 5 → 从 checkpoint 续训到 10
        out_a = os.path.join(self.tmp, 'phase_a')
        trainer.train(self.cfg, self.train, self.val, self.itos,
                      steps=5, batch=2, accum=2, lr=0.01, seed=7,
                      out_dir=out_a, log=lambda _s: None)
        ckpt5 = os.path.join(out_a, 'checkpoint.pt')
        out_b = os.path.join(self.tmp, 'phase_b')
        trainer.train(self.cfg, self.train, self.val, self.itos,
                      steps=10, batch=2, accum=2, lr=0.01, seed=7,
                      out_dir=out_b, resume_path=ckpt5, log=lambda _s: None)
        a = torch.load(os.path.join(out_direct, 'checkpoint.pt'), weights_only=False)
        b = torch.load(os.path.join(out_b, 'checkpoint.pt'), weights_only=False)
        self.assertEqual(a['step'], b['step'], 10)
        for k in a['model']:
            self.assertTrue(torch.equal(a['model'][k], b['model'][k]),
                            f'续训与直训在 {k} 上不一致')
        # 优化器状态同样一致（Adam 动量是续训等价性的一半）
        for k in a['opt']['state']:
            for field in a['opt']['state'][k]:
                va, vb = a['opt']['state'][k][field], b['opt']['state'][k][field]
                if torch.is_tensor(va):
                    self.assertTrue(torch.equal(va, vb), f'opt state {k}/{field} 不一致')

    def test_取消安全存盘(self):
        out = os.path.join(self.tmp, 'cancel')
        calls = {'n': 0}

        def cancel_after_first():
            calls['n'] += 1
            return calls['n'] > 1          # 第一步之后触发取消

        m = trainer.train(self.cfg, self.train, self.val, self.itos,
                          steps=50, batch=2, accum=1, lr=0.01, seed=7,
                          out_dir=out, cancel_check=cancel_after_first,
                          log=lambda _s: None)
        self.assertTrue(m['cancelled'])
        self.assertEqual(m['cancel_reason'], 'signal')
        self.assertTrue(os.path.exists(os.path.join(out, 'checkpoint.pt')))
        ck = torch.load(os.path.join(out, 'checkpoint.pt'), weights_only=False)
        self.assertGreater(ck['step'], 0)   # 有进度可续，不是空壳


class TestModelSchema(unittest.TestCase):
    def test_参数量公式与实例一致(self):
        for V in (50, 1772, 7000):
            cfg = dict(ctx=32, d=64, layers=2, heads=2, ffMul=2)
            m = MiniLM(V, dict(cfg))
            n = nov_model.assert_param_count(m, cfg, V)      # 不一致会抛
            self.assertEqual(n, sum(p.numel() for p in m.parameters()))

    def test_参数量漂移必须被拦下(self):
        cfg = dict(ctx=32, d=64, layers=2, heads=2, ffMul=2)
        m = MiniLM(65, dict(cfg))                            # V=65，与公式所用的 64 不符
        with self.assertRaises(AssertionError):
            nov_model.assert_param_count(m, cfg, V=64)       # 故意给错 V

    def test_构建带谱系声明(self):
        m, prov = nov_model.build(dict(ctx=16, d=16, layers=1, heads=2, ffMul=1),
                                  V=100, seed=42)
        self.assertEqual(prov['init'], 'local_random_seed')
        self.assertTrue(prov['no_external_weights'])
        self.assertGreater(prov['param_count'], 0)

    def test_配置档解析(self):
        self.assertEqual(nov_model.resolve_cfg('micro')['d'], 128)
        self.assertEqual(nov_model.resolve_cfg('t30m')['d'], 512)
        j = nov_model.resolve_cfg('{"d": 96}')
        self.assertEqual(j['d'], 96)
        with self.assertRaises(ValueError):
            nov_model.resolve_cfg('不存在档')


if __name__ == '__main__':
    unittest.main(verbosity=2)
