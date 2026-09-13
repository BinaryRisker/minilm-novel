# ============================================================================
# minilm_torch.py — minilm.js 的 PyTorch 复刻（L0 平移验证）
# ----------------------------------------------------------------------------
# 目标不是"写一个更好的模型"，而是"同一个模型换一种实现"。
# 因此这里逐项对齐 minilm.js 的细节，任何一处偏离都会让对齐验证失去意义：
#
#   minilm.js                          本文件
#   ---------------------------------------------------------------
#   randn(n, 0.02)                     normal_(0, 0.02)
#   Wo/W2 用 resStd=0.02/sqrt(2L)      同（残差路径小初始化）
#   LayerNorm eps=1e-5                 nn.LayerNorm(eps=1e-5)
#   GELU tanh 近似                     F.gelu(approximate='tanh')
#   logits = x @ emb^T（权重共享）      同
#   Adam b1=.9 b2=.999 eps=1e-8        torch.optim.Adam（默认值相同）
#   全局范数裁剪 1.0（+1e-6）           clip_grad_norm_(1.0)（公式相同）
#   线性 warmup + 余弦衰减             同一公式
#   makeBatch 随机窗口                 同一采样区间
#   留出集评测：32 字滑窗 × 32 位置     同一分窗方式
# ============================================================================
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CFG = dict(ctx=32, d=40, layers=2, heads=4, ffMul=4)


# ---------------------------- 分词器 ----------------------------

def build_vocab(text, min_freq=1, max_vocab=None):
    # minilm.js: itos = ['\x00'] + Array.from(new Set(text)).sort()
    # Python 对单字符按 Unicode 码点排序，与 JS 默认排序在 BMP 上一致。
    # min_freq<=1 且 max_vocab 为空时行为与 minilm.js 完全一致（逐位对齐依赖此路径）；
    # 语料放大到百万字后，可选按词频裁掉长尾生僻字（映射到 0 号 UNK）。
    if min_freq <= 1 and not max_vocab:
        chars = sorted(set(text))
    else:
        from collections import Counter
        cnt = Counter(text)
        keep = {c for c, n in cnt.items() if n >= min_freq}
        if max_vocab and len(keep) > max_vocab:
            keep = {c for c, _ in cnt.most_common(max_vocab)}
        chars = sorted(keep)
    itos = ['\x00'] + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return itos, stoi


def encode(text, stoi):
    return [stoi.get(ch, 0) for ch in text]


def decode(ids, itos):
    return ''.join('' if i == 0 else itos[i] for i in ids)


# ---------------------------- 模型本体 ----------------------------

class Block(nn.Module):
    def __init__(self, d, heads, ff_mul, n_layers):
        super().__init__()
        self.heads = heads
        self.dh = d // heads
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.Wq = nn.Linear(d, d, bias=True)
        self.Wk = nn.Linear(d, d, bias=True)
        self.Wv = nn.Linear(d, d, bias=True)
        self.Wo = nn.Linear(d, d, bias=True)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.W1 = nn.Linear(d, ff_mul * d, bias=True)
        self.W2 = nn.Linear(ff_mul * d, d, bias=True)
        self.apply(self._init_weights)
        # 残差输出路径用缩小初始化（minilm.js 的 resStd = 0.02/sqrt(2L)）
        res_std = 0.02 / math.sqrt(2 * n_layers)
        for lin in (self.Wo, self.W2):
            nn.init.normal_(lin.weight, std=res_std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)
        q = self.Wq(h).view(B, T, self.heads, self.dh).transpose(1, 2)
        k = self.Wk(h).view(B, T, self.heads, self.dh).transpose(1, 2)
        v = self.Wv(h).view(B, T, self.heads, self.dh).transpose(1, 2)
        # causal mask：只准看过去（minilm.js 里就是 s <= t 那个循环）
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.Wo(a)
        h = self.ln2(x)
        h = self.W2(F.gelu(self.W1(h), approximate='tanh'))
        return x + h


class MiniLM(nn.Module):
    def __init__(self, vocab_size, cfg=None):
        super().__init__()
        c = dict(DEFAULT_CFG)
        if cfg:
            c.update(cfg)
        self.cfg = c
        D, L = c['d'], c['layers']
        self.emb = nn.Embedding(vocab_size, D)
        self.pos = nn.Embedding(c['ctx'], D)
        self.blocks = nn.ModuleList(Block(D, c['heads'], c['ffMul'], L) for _ in range(L))
        self.lnf = nn.LayerNorm(D, eps=1e-5)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, tokens, targets=None):
        # tokens/targets: [B, T]，targets 为 tokens 右移一位
        T = tokens.shape[1]
        pos = torch.arange(T, device=tokens.device)
        x = self.emb(tokens) + self.pos(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.lnf(x)
        logits = x @ self.emb.weight.T          # 权重共享（Weight Tying）
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss


# ---------------------------- 训练批次 ----------------------------

def make_batch(ids, B, T, rng):
    # minilm.js makeBatch：起点 ∈ [0, len - T - 1) 的均匀整数
    max_start = len(ids) - T - 1
    if max_start <= 0:
        raise ValueError('语料太短，请增加文本长度或减小上下文窗口')
    tokens, targets = [], []
    for _ in range(B):
        s = rng.randrange(max_start)
        tokens.append(ids[s:s + T])
        targets.append(ids[s + 1:s + 1 + T])
    return (torch.tensor(tokens, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long))


def lr_at(step, total, base_lr, warmup):
    # 与 test_novel.js 同一条调度曲线
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


# ---------------------------- 训练 + 评测 ----------------------------

def train(text, steps=400, batch=8, lr=0.004, seed=1234, cfg=None,
          snapshot_every=None, log=None, min_freq=1, max_vocab=None):
    """复刻 test_novel.js 的 trainCorpus 训练段。返回 (model, vocab, snapshots)。"""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    itos, stoi = build_vocab(text, min_freq=min_freq, max_vocab=max_vocab)
    ids = encode(text, stoi)

    # 90% 训练 / 10% 留出（词表建立在全文上，与 JS 相同）
    split = math.floor(len(ids) * 0.9)
    train_ids = ids[:split]

    model = MiniLM(len(itos), cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = max(10, math.floor(steps * 0.15))
    snap_every = snapshot_every or max(1, steps // 4)

    model.train()
    snapshots = []
    seg_loss, seg_n = 0.0, 0
    for step in range(steps):
        tokens, targets = make_batch(train_ids, batch, model.cfg['ctx'], rng)
        opt.zero_grad()
        _, loss = model(tokens, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps, lr, warmup)
        opt.step()
        seg_loss += loss.item()
        seg_n += 1
        if (step + 1) % snap_every == 0 or step == 0:
            snapshots.append({'step': step + 1, 'loss': seg_loss / seg_n})
            if log:
                log(f'    step {step + 1}/{steps}  loss={seg_loss / seg_n:.4f}')
            seg_loss, seg_n = 0.0, 0
    return model, (itos, stoi), snapshots


@torch.no_grad()
def holdout_eval(model, text, seed=1234, min_freq=1, max_vocab=None):
    """复刻 test_novel.js 的留出集评测：后 10% 分成 32 字窗口逐位置测 top-k。"""
    torch.manual_seed(seed)
    itos, stoi = build_vocab(text, min_freq=min_freq, max_vocab=max_vocab)
    ids = encode(text, stoi)
    split = math.floor(len(ids) * 0.9)
    test_ids = ids[split:]
    ctx = model.cfg['ctx']
    model.eval()
    top1 = top5 = total = 0
    s = 0
    while s + ctx + 1 <= len(test_ids):
        chunk = torch.tensor([test_ids[s:s + ctx]], dtype=torch.long)
        tgt = test_ids[s + 1:s + 1 + ctx]
        logits, _ = model(chunk)
        topv, topi = logits[0].topk(5)
        for i in range(ctx):
            pred = topi[i].tolist()
            total += 1
            if pred[0] == tgt[i]:
                top1 += 1
            if tgt[i] in pred:
                top5 += 1
        s += ctx
    return {'top1': 100 * top1 / total, 'top5': 100 * top5 / total, 'n': total,
            'final_loss': None}
