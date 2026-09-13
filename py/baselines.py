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
    P(t | c) = (count(c,t) + α) / (count(c) + α·V)  当 count(c) > 0
    否则回退到低一阶；一阶（unigram）永远有定义。
    """

    def __init__(self, train_ids: Iterable[int], order: int = 3,
                 alpha: float = 0.1, V: int = 0):
        self.order = max(1, int(order))
        self.alpha = float(alpha)
        self.V = int(V)
        ids = list(train_ids)
        self.N = len(ids)

        # unigram：全词表计数
        self.uni = np.zeros(self.V, dtype=np.float64)
        for t in ids:
            self.uni[t] += 1

        # n>=2：ctxTotal[n][key]、ctxCount[n][key][tgt]
        self.ctxTotal: Dict[int, Dict[Tuple[int, ...], int]] = {}
        self.ctxCount: Dict[int, Dict[Tuple[int, ...], np.ndarray]] = {}
        for n in range(2, self.order + 1):
            tot: Dict[Tuple[int, ...], int] = {}
            cnt: Dict[Tuple[int, ...], np.ndarray] = {}
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

    def log_prob(self, hist: List[int], target: int) -> float:
        """带回退的单点 logP；hist 是从早到晚的上文 token 序列。"""
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

    def dist(self, hist: List[int]) -> np.ndarray:
        """完整对数概率分布，shape=[V]"""
        out = np.zeros(self.V, dtype=np.float64)
        for v in range(self.V):
            out[v] = self.log_prob(hist, v)
        return out

    def used_order(self, hist: List[int]) -> int:
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


# ---------------------------- 评测滑窗 ----------------------------

def for_each_position(test_ids: List[int], ctx: int):
    """生成器：(hist, target)，每次返回一个 (上下文 token 序列, 下一字 id)。
    与 eval/protocol.js 的 forEachPosition 完全一致：起点 s=0, 每 ctx 步一窗，
    直到 s+ctx+1 > len(test_ids) 停止。每个窗口内逐位置给 (hist, target)。"""
    for s in range(0, len(test_ids) - ctx):
        window = test_ids[s:s + ctx + 1]
        for i in range(ctx):
            yield window[:i + 1], window[i + 1]


def count_positions(test_ids: List[int], ctx: int) -> int:
    return max(0, len(test_ids) - ctx) * ctx


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