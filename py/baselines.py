""" ============================================================================
py/baselines.py — Python 复刻 eval/baselines.js 的 NGram + α 选择
----------------------------------------------------------------------------
为什么需要这个：sweep 实验要在「小训练集」上跑 trigram 基线，
而基线 JS 用 Map+Float64Array，Python 必须严格等价（同样的训练 / 留出切分、
同样的 α 网格、同样的回退规则、同样的 NLL/Top-1 公式）。

接口约定（与 JS 一一对应）：
    ng = NGram(train_ids, order=3, alpha=0.1, V=vocab_size)
    ng.log_prob(hist, target) -> float          # 单点 logP（自然对数）
    ng.dist(hist)            -> np.ndarray[V]   # 完整 logP 分布
    ng.used_order(hist)      -> int             # 1/2/3，回退到几阶
    select_alpha(train_ids, order, alphas, V, ctx) -> {alpha, curve}

评估位与 JS 完全一致：32 字滑窗 × 32 位置（ctx 决定窗长）。

用法：
    from baselines import NGram, select_alpha, run_baselines
============================================================================ """
from __future__ import annotations
import math
from typing import Iterable, List, Dict, Tuple

import numpy as np

DEFAULT_ALPHAS: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0)


# ---------------------------- n-gram 模型 ----------------------------

class NGram:
    """加 α 平滑、带回退的字符 n-gram 语言模型。

    P(t | c) = (count(c,t) + α) / (count(c) + α·V)   当 count(c) > 0
    否则回退到低一阶；一阶（unigram）永远有定义。

    存储为**稀疏计数**：上下文用整型编码当键（key = Σ tok·V^k），值只记实际出现过的
    后继（dict: tgt → count）。早期版本给每个上下文分配一个 V 维 float64 数组，
    在 200 万字语料上会膨胀到几十 GB（约 10^6 个三元上下文 × 4500 维 × 8B ≈ 36 GB），
    实测直接把数据量扫描挂死。现在按需在 dist() 里临时构造稠密数组（36 KB/次）。
    """

    def __init__(self, train_ids: Iterable[int], order: int = 3,
                 alpha: float = 0.1, V: int = 0):
        self.order = max(1, int(order))
        self.alpha = float(alpha)
        self.V = max(1, int(V))
        ids = list(train_ids)
        self.N = len(ids)

        # unigram：全词表计数
        uni = [0] * self.V
        for t in ids:
            uni[t] += 1
        self.uni = np.asarray(uni, dtype=np.float64)

        # n >= 2：ctxTotal[n][key] = 上下文出现次数；ctxCount[n][key] = {tgt: count}
        self.ctxTotal: Dict[int, Dict[int, int]] = {}
        self.ctxCount: Dict[int, Dict[int, Dict[int, int]]] = {}
        for n in range(2, self.order + 1):
            need = n - 1
            tot: Dict[int, int] = {}
            cnt: Dict[int, Dict[int, int]] = {}
            for j in range(len(ids) - n + 1):
                key = 0
                for k in range(need):
                    key = key * self.V + ids[j + k]
                tgt = ids[j + n - 1]
                d = cnt.get(key)
                if d is None:
                    cnt[key] = {tgt: 1}
                    tot[key] = 1
                else:
                    d[tgt] = d.get(tgt, 0) + 1
                    tot[key] += 1
            self.ctxTotal[n] = tot
            self.ctxCount[n] = cnt

    def _key(self, hist: List[int], need: int) -> int:
        key = 0
        for tok in hist[-need:]:
            key = key * self.V + tok
        return key

    def _backoff(self, hist: List[int]):
        """按回退规则挑出最合适的计数来源。
        返回 (counts, total, used_order)；counts 为 self.uni（numpy）或 {tgt: count}。"""
        n = self.order
        while n > 1:
            need = n - 1
            if len(hist) >= need:
                key = self._key(hist, need)
                tot = self.ctxTotal[n].get(key)
                if tot:
                    return self.ctxCount[n][key], tot, n
            n -= 1
        return self.uni, self.N, 1

    def log_prob(self, hist: List[int], target: int) -> float:
        """带回退的单点 logP；hist 是从早到晚的上文 token 序列。"""
        counts, tot, _ = self._backoff(hist)
        if counts is self.uni:
            c = float(self.uni[target])
        else:
            c = float(counts.get(target, 0))
        return math.log((c + self.alpha) / (tot + self.alpha * self.V))

    def dist(self, hist: List[int]) -> np.ndarray:
        """完整对数概率分布，shape=[V]（按需构造稠密数组，不常驻）"""
        counts, tot, _ = self._backoff(hist)
        if counts is self.uni:
            dense = self.uni
        else:
            dense = np.zeros(self.V, dtype=np.float64)
            for t, c in counts.items():
                dense[t] = c
        return np.log((dense + self.alpha) / (tot + self.alpha * self.V))

    def used_order(self, hist: List[int]) -> int:
        return self._backoff(hist)[2]


# ---------------------------- 评测滑窗 ----------------------------

def for_each_position(test_ids: List[int], ctx: int):
    """生成器：(hist, target)，每次返回一个 (上下文 token 序列, 下一字 id)。
    与 eval/protocol.js 的 forEachPosition 完全一致：起点 s=0, **每 ctx 步一窗**，
    直到 s+ctx+1 > len(test_ids) 停止。每个窗口内逐位置给 (hist, target)。
    注意：必须与 minilm_torch.holdout_eval 的步长一致，否则模型与基线不在同一批
    评测位上比较（曾因步长写成 1 导致基线被评测 32 倍的位置、耗时爆炸）。"""
    n = len(test_ids)
    s = 0
    while s + ctx + 1 <= n:
        window = test_ids[s:s + ctx + 1]
        for i in range(ctx):
            yield window[:i + 1], window[i + 1]
        s += ctx


def count_positions(test_ids: List[int], ctx: int) -> int:
    n = len(test_ids)
    c = 0
    s = 0
    while s + ctx + 1 <= n:
        c += ctx
        s += ctx
    return c


# ---------------------------- 评估累加 ----------------------------

class Acc:
    """与 eval/protocol.js 的 Acc 完全对应：累加 NLL、Top-1、Top-5。"""

    def __init__(self):
        self.nll_sum = 0.0
        self.top1 = 0
        self.top5 = 0
        self.total = 0

    def add(self, log_dist: np.ndarray, target: int):
        # log_dist[v] = logP(v|hist)
        d = log_dist
        # NLL = -logP(target)
        self.nll_sum += -float(d[target])
        # top-k：取前 5
        top5_idx = np.argpartition(-d, 5)[:5] if d.size > 5 else np.argsort(-d)
        if target in top5_idx:
            self.top5 += 1
        if top5_idx[np.argmax(d[top5_idx])] == target:
            self.top1 += 1
        self.total += 1

    def summary(self) -> Dict[str, float]:
        n = max(1, self.total)
        nll = self.nll_sum / n
        return {
            'nll': nll,
            'ppl': math.exp(nll),
            'top1': 100.0 * self.top1 / n,
            'top5': 100.0 * self.top5 / n,
            'n': self.total,
        }


# ---------------------------- α 选择 ----------------------------

def select_alpha(train_ids: List[int], order: int, alphas: Iterable[float],
                 V: int, ctx: int) -> Dict:
    """训练集尾部 10% 当开发集挑 α，避免直接在测试集上挑。
    返回 {alpha, curve: [{alpha, nll, top1}]}。"""
    sp = split_ids(train_ids, 0.9)
    curve = []
    best = {'alpha': alphas[0], 'nll': float('inf')}
    for a in alphas:
        if len(sp['test']) < ctx + 1:
            break
        m = NGram(sp['train'], order, a, V)
        acc = Acc()
        for hist, tgt in for_each_position(sp['test'], ctx):
            acc.add(m.dist(hist), tgt)
        s = acc.summary()
        curve.append({'alpha': a, 'nll': s['nll'], 'top1': s['top1']})
        if s['nll'] < best['nll']:
            best = {'alpha': a, 'nll': s['nll']}
    return {'alpha': best['alpha'], 'curve': curve}


def split_ids(ids: List[int], train_frac: float = 0.9) -> Dict[str, List[int]]:
    """前 train_frac 当训练，后 1-train_frac 当留出。
    与 minilm.js/PyTorch 复刻的 math.floor(len*0.9) 完全一致。"""
    cut = int(math.floor(len(ids) * train_frac))
    return {'train': ids[:cut], 'test': ids[cut:]}


# ---------------------------- 一次性跑全部基线 ----------------------------

def run_baselines(train_ids: List[int], test_ids: List[int], ctx: int = 32,
                  V: int = 0, alphas: Iterable[float] = DEFAULT_ALPHAS,
                  orders: Iterable[int] = (1, 2, 3),
                  select_on_dev: bool = True) -> List[Dict]:
    """对所有 (order, alpha) 组合跑 NGram，返回结构与 JS 的 runBaselines 一致。
    默认会在训练集尾部 10% 开发集上挑 α；select_on_dev=False 则只用 alphas[0]。"""
    out = []
    for order in orders:
        if select_on_dev and len(alphas) > 1:
            sel = select_alpha(train_ids, order, alphas, V, ctx)
            alpha = sel['alpha']
            curve = sel['curve']
        else:
            alpha = alphas[0]
            curve = None
        m = NGram(train_ids, order, alpha, V)
        acc = Acc()
        backoff = {1: 0, 2: 0, 3: 0}
        n = 0
        for hist, tgt in for_each_position(test_ids, ctx):
            acc.add(m.dist(hist), tgt)
            backoff[m.used_order(hist)] += 1
            n += 1
        s = acc.summary()
        out.append({
            'order': order,
            'name': {1: 'unigram（字频）', 2: 'bigram（+α平滑）',
                     3: 'trigram（+α平滑）'}[order],
            'alpha': alpha,
            'n': n,
            'top1': s['top1'],
            'top5': s['top5'],
            'nll': s['nll'],
            'ppl': s['ppl'],
            'backoff': backoff,
            'curve': curve,
        })
    return out