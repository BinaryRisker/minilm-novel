""" ============================================================================
py/tests/test_baselines.py — NGram 稀疏实现与「每上下文一个 V 维数组」的
                             参考稠密实现做逐点差分，保证语义完全一致
----------------------------------------------------------------------------
背景：把 NGram 的存储从稠密改成稀疏是为了解决百万字语料的几十 GB 内存爆炸，
但基线能不能改，取决于「改了之后数字是否一模一样」。这里用固定的随机语料
逐点比对 log_prob / dist / used_order。

运行：
  python -m unittest py.tests.test_baselines
  python py/tests/test_baselines.py          # 直接跑，附带规模基准
============================================================================ """
from __future__ import annotations
import math
import os
import random
import sys
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
if PY not in sys.path:
    sys.path.insert(0, PY)

from baselines import NGram, Acc, for_each_position, count_positions  # noqa


class RefNGram:
    """参考实现：与原版一致的稠密存储（只用于测试，不用于大语料）。"""

    def __init__(self, train_ids, order=3, alpha=0.1, V=0):
        self.order = max(1, int(order))
        self.alpha = float(alpha)
        self.V = int(V)
        ids = list(train_ids)
        self.N = len(ids)
        self.uni = np.zeros(self.V, dtype=np.float64)
        for t in ids:
            self.uni[t] += 1
        self.ctxTotal = {}
        self.ctxCount = {}
        for n in range(2, self.order + 1):
            tot, cnt = {}, {}
            for j in range(len(ids) - n + 1):
                key = tuple(ids[j:j + n - 1])
                tgt = ids[j + n - 1]
                if key not in tot:
                    tot[key] = 0
                    cnt[key] = np.zeros(self.V, dtype=np.float64)
                tot[key] += 1
                cnt[key][tgt] += 1
            self.ctxTotal[n] = tot
            self.ctxCount[n] = cnt

    def log_prob(self, hist, target):
        n = self.order
        while n > 1:
            need = n - 1
            if len(hist) >= need:
                key = tuple(hist[-need:])
                tot = self.ctxTotal[n].get(key)
                if tot is not None and tot > 0:
                    c = self.ctxCount[n][key][target]
                    return math.log((c + self.alpha) / (tot + self.alpha * self.V))
            n -= 1
        return math.log((self.uni[target] + self.alpha) / (self.N + self.alpha * self.V))

    def dist(self, hist):
        out = np.zeros(self.V, dtype=np.float64)
        for v in range(self.V):
            out[v] = self.log_prob(hist, v)
        return out

    def used_order(self, hist):
        n = self.order
        while n > 1:
            need = n - 1
            if len(hist) >= need:
                key = tuple(hist[-need:])
                tot = self.ctxTotal[n].get(key)
                if tot is not None and tot > 0:
                    return n
            n -= 1
        return 1


def make_data(n=3000, V=40, seed=7):
    rng = random.Random(seed)
    # 掺入重复片段，让高阶上下文真的出现，避免全部回退到 unigram
    base = [rng.randrange(V) for _ in range(V * 3)]
    ids = []
    while len(ids) < n:
        ids += base if rng.random() < 0.5 else [rng.randrange(V) for _ in range(rng.randrange(5, 40))]
    return ids[:n], V


class TestNGramParity(unittest.TestCase):

    def setUp(self):
        self.ids, self.V = make_data()
        self.a = NGram(self.ids, order=3, alpha=0.1, V=self.V)
        self.b = RefNGram(self.ids, order=3, alpha=0.1, V=self.V)

    def test_used_order_matches(self):
        for i in range(2, 400):
            hist = self.ids[max(0, i - 5):i]
            self.assertEqual(self.a.used_order(hist), self.b.used_order(hist),
                             f'used_order 不一致, i={i}')

    def test_log_prob_matches(self):
        for i in range(2, 400):
            hist = self.ids[max(0, i - 5):i]
            tgt = self.ids[i]
            self.assertAlmostEqual(self.a.log_prob(hist, tgt),
                                   self.b.log_prob(hist, tgt), places=12)

    def test_dist_matches(self):
        for i in (5, 37, 120, 399):
            hist = self.ids[max(0, i - 5):i]
            da = self.a.dist(hist)
            db = self.b.dist(hist)
            self.assertEqual(da.shape, (self.V,))
            np.testing.assert_allclose(da, db, atol=1e-12, rtol=0)

    def test_acc_topk_matches(self):
        acc_a, acc_b = Acc(), Acc()
        for hist, tgt in for_each_position(self.ids[len(self.ids) // 2:], 8):
            acc_a.add(self.a.dist(hist), tgt)
            acc_b.add(self.b.dist(hist), tgt)
        sa, sb = acc_a.summary(), acc_b.summary()
        self.assertEqual(sa['n'], sb['n'])
        self.assertAlmostEqual(sa['top1'], sb['top1'], places=9)
        self.assertAlmostEqual(sa['top5'], sb['top5'], places=9)
        self.assertAlmostEqual(sa['nll'], sb['nll'], places=9)

    def test_backoff_when_context_unseen(self):
        # 训练语料只含 0/1/2 的循环；hist=[2,3] 时：
        #   三元上下文 (2,3) 没出现 → 回退；二元上下文 (3,) 也没出现 → 再回退
        # 因此应落到 unigram（used_order == 1）。
        ids = [0, 1, 2] * 200
        a = NGram(ids, order=3, alpha=0.1, V=8)
        b = RefNGram(ids, order=3, alpha=0.1, V=8)
        hist = [2, 3]
        self.assertEqual(a.used_order(hist), 1)
        for tgt in range(8):
            self.assertAlmostEqual(a.log_prob(hist, tgt), b.log_prob(hist, tgt),
                                   places=12)

    def test_backoff_to_bigram_when_trigram_missing(self):
        # [1,2,0,1,2] 里三元上下文 (1,2) 出现过、四元 (0,1,2) 不一定；
        # 用整段训练保证 (2,0) 存在而 (2,2) 不存在 → 应回退到 bigram
        ids = [0, 1, 2] * 200
        a = NGram(ids, order=3, alpha=0.1, V=8)
        b = RefNGram(ids, order=3, alpha=0.1, V=8)
        hist = [3, 2]          # (3,2) 没见过 → 回退；trigram 路径不看 hist[0]
        self.assertEqual(a.used_order(hist), b.used_order(hist))
        self.assertAlmostEqual(a.log_prob(hist, 0), b.log_prob(hist, 0), places=12)

    def test_order1_equals_unigram(self):
        m = NGram(self.ids, order=1, alpha=0.1, V=self.V)
        r = RefNGram(self.ids, order=1, alpha=0.1, V=self.V)
        for tgt in range(self.V):
            self.assertAlmostEqual(m.log_prob([], tgt), r.log_prob([], tgt), places=12)


class TestWindowProtocol(unittest.TestCase):

    def test_stride_is_ctx_not_one(self):
        ids = list(range(100))
        pos = list(for_each_position(ids, 16))
        self.assertEqual(len(pos), count_positions(ids, 16))
        # 每 16 步一窗：窗数 = floor((100-1)/16) = 6，共 96 个位置
        self.assertEqual(len(pos), 96)

    def test_history_lengths(self):
        ids = list(range(100))
        pos = list(for_each_position(ids, 4))
        for i in range(4):
            hist, tgt = pos[i]
            self.assertEqual(len(hist), i + 1)
            self.assertEqual(tgt, i + 1)


def scale_benchmark():
    """在真实语料上测内存与耗时，确认百万字规模可用。"""
    import json
    corpus_path = os.path.join(PY, 'data', 'corpus.json')
    if not os.path.exists(corpus_path):
        print('（跳过规模基准：找不到 py/data/corpus.json）')
        return
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    from minilm_torch import build_vocab, encode
    for key in ('scifi', 'wuxia'):
        if key not in corpus or not corpus[key].get('text'):
            continue
        text = corpus[key]['text']
        itos, stoi = build_vocab(text, min_freq=3)
        ids = encode(text, stoi)
        cut = int(len(ids) * 0.9)
        train, test = ids[:cut], ids[cut:]
        t0 = time.time()
        m = NGram(train, order=3, alpha=0.1, V=len(itos))
        t_fit = time.time() - t0
        t0 = time.time()
        acc = Acc()
        for hist, tgt in for_each_position(test, 32):
            acc.add(m.dist(hist), tgt)
        t_eval = time.time() - t0
        s = acc.summary()
        ctx_all = sum(len(m.ctxCount[n]) for n in m.ctxCount)
        print(f'[{key}] 训练 {len(train)} 字 / 留出 {len(test)} 字 / V={len(itos)} | '
              f'上下文数={ctx_all} | 建表 {t_fit:.1f}s 评测 {t_eval:.1f}s | '
              f'n={s["n"]} top1={s["top1"]:.2f}% nll={s["nll"]:.3f}')


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
    scale_benchmark()
