""" ============================================================================
py/ablation_ctx.py — 两个「拿掉一个东西会怎样」的对照实验
----------------------------------------------------------------------------
mode=ctx   ：上下文长度 8 / 16 / 32 / 64，在**同一批固定评测位置**上比较。
             为什么不能直接用 sweep 的曲线：`for_each_position` 是「每 ctx 步一窗、
             窗内 history 长度 1..ctx」，不同 ctx 评的是**不同的 history 长度分布**，
             跨 ctx 比大小是被混淆的。这里改成：固定一组位置 p（要求 p 之前至少
             有 64 个字），每个模型都只看自己 ctx 能容纳的最后 ctx 个字。
mode=mask  ：把因果掩码拿掉（cfg.causal=False），看训练损失与留出成绩如何分道扬镳。
             —— 顺带回答一个协议层面的问题：本项目的留出评测是「窗口内逐位置预测下一字」，
                它之所以安全，**恰恰是因为有因果掩码**；掩码一拿掉，同一份评测立刻变成泄漏。

用法：
  python py/ablation_ctx.py --mode ctx  --bucket scifi --size 20000 --ctxs 8,16,32,64
  python py/ablation_ctx.py --mode mask --bucket scifi --size 20000
输出：py/runs/abl_ctx_<bucket>_t<size>.json / abl_mask_<bucket>_t<size>.json + 同名 _report.md
============================================================================ """
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from minilm_torch import MiniLM, build_vocab, encode, make_batch, lr_at   # noqa
from baselines import NGram, Acc                                          # noqa
from sweep_worksplit import read_manifest, load_bucket, build_slice        # noqa

CFG_BASE = dict(d=40, layers=2, heads=4, ffMul=4)
HIST_MIN = 64        # 固定评测位置要求的前置历史长度（覆盖最大 ctx）


def fixed_positions(ids: List[int], stride: int, cap: int) -> List[int]:
    """留出序列里，前置历史至少 HIST_MIN 的位置；等距抽样后封顶 cap 个。"""
    pos = [p for p in range(HIST_MIN, len(ids) - 1) if (p - HIST_MIN) % stride == 0]
    return pos[:cap]


@torch.no_grad()
def eval_fixed(model, ids: List[int], positions: List[int], ctx: int) -> Dict:
    """每个位置用 ids[p-ctx:p] 预测 ids[p] —— 所有 ctx 档评的是同一批位置。"""
    model.eval()
    top1 = top5 = 0
    nll = 0.0
    B = 256
    for i in range(0, len(positions), B):
        ps = positions[i:i + B]
        x = torch.tensor([ids[p - ctx:p] for p in ps], dtype=torch.long)
        y = torch.tensor([ids[p] for p in ps], dtype=torch.long)
        logits, _ = model(x)
        lp = torch.log_softmax(logits[:, -1, :], dim=-1)      # 末位 → 预测下一个字
        pred = lp.argmax(dim=-1)
        top1 += int((pred == y).sum())
        top5 += int((y.unsqueeze(1) == lp.topk(5, dim=-1).indices).any(dim=-1).sum())
        nll += float(-lp[torch.arange(len(ps)), y].sum())
    n = len(positions)
    return {'top1': 100.0 * top1 / n, 'top5': 100.0 * top5 / n,
            'nll': nll / n, 'ppl': math.exp(min(nll / n, 30)), 'n': n}


def trigram_fixed(train_ids: List[int], ids: List[int], positions: List[int], V: int,
                  alpha: float = 0.1) -> Dict:
    m = NGram(train_ids, order=3, alpha=alpha, V=V)
    acc = Acc()
    for p in positions:
        acc.add(m.dist(ids[p - 2:p]), ids[p])
    return acc.summary()


def run_once(text_train: List[int], ids: List[int], positions: List[int], V: int,
             cfg: dict, steps: int, batch: int, lr: float, seed: int) -> Dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    ctx = cfg['ctx']
    model = MiniLM(V, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = max(10, math.floor(steps * 0.15))
    params = sum(p.numel() for p in model.parameters())

    model.train()
    seg, cnt, final_loss = 0.0, 0, float('nan')
    for step in range(steps):
        tokens, targets = make_batch(text_train, batch, ctx, rng)
        opt.zero_grad()
        _, loss = model(tokens, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps, lr, warmup)
        opt.step()
        seg += loss.item()
        cnt += 1
    final_loss = seg / max(1, cnt)

    ev = eval_fixed(model, ids, positions, ctx)
    ev.update({'train_loss': final_loss, 'steps': steps, 'params': params})
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='ctx', choices=['ctx', 'mask'])
    ap.add_argument('--bucket', default='scifi')
    ap.add_argument('--size', type=int, default=20000, help='训练用的训练作品前缀字数')
    ap.add_argument('--test_chars', type=int, default=20000)
    ap.add_argument('--ctxs', default='8,16,32,64')
    ap.add_argument('--seeds', default='1234,42,2024')
    ap.add_argument('--epochs', type=float, default=3.0)
    ap.add_argument('--min_steps', type=int, default=200)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.004)
    ap.add_argument('--min_freq', type=int, default=3)
    ap.add_argument('--eval_cap', type=int, default=8000)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'runs'))
    args = ap.parse_args()

    manifest = read_manifest(os.path.join(ROOT, 'data', 'manifest.jsonl'))
    b = load_bucket(args.bucket, manifest, os.path.join(ROOT, 'data', 'processed'),
                    os.path.join(ROOT, 'data', 'splits'))
    itos, stoi = build_vocab(b['train_text'], min_freq=args.min_freq)
    V = len(itos)
    train_all = encode(b['train_text'], stoi)
    test_slice = build_slice(b['test_texts'], args.test_chars)
    test_ids = encode(test_slice, stoi)
    train_ids = train_all[:min(args.size, len(train_all))]

    positions = fixed_positions(test_ids, args.stride, args.eval_cap)
    seeds = [int(s) for s in args.seeds.split(',')]
    print(f'[data] {args.bucket} V={V} 训练 {len(train_ids)} 字 / 留出 {len(test_ids)} 字 '
          f'/ 固定评测位置 {len(positions)} 个', flush=True)

    t0 = time.time()
    tri = trigram_fixed(train_ids, test_ids, positions, V)
    print(f'[trigram] Top-1={tri["top1"]:.1f}% NLL={tri["nll"]:.3f} ({time.time()-t0:.1f}s)',
          flush=True)

    variants = []
    if args.mode == 'ctx':
        for c in [int(x) for x in args.ctxs.split(',')]:
            variants.append((f'ctx{c}', dict(CFG_BASE, ctx=c)))
    else:
        for cm, name in ((True, 'causal'), (False, 'no_mask')):
            variants.append((name, dict(CFG_BASE, ctx=32, causal=cm)))

    runs = []
    for label, cfg in variants:
        rows = []
        for seed in seeds:
            steps = max(args.min_steps,
                        int(round(args.epochs * len(train_ids) / (args.batch * cfg['ctx']))))
            t1 = time.time()
            r = run_once(train_ids, test_ids, positions, V, cfg, steps,
                         args.batch, args.lr, seed)
            r.update({'seed': seed, 'ctx': cfg['ctx'],
                      'causal': cfg.get('causal', True)})
            rows.append(r)
            print(f'  {label} seed={seed} Top-1={r["top1"]:.2f}% NLL={r["nll"]:.3f} '
                  f'train_loss={r["train_loss"]:.3f} ({time.time()-t1:.1f}s)', flush=True)
        runs.append({
            'label': label, 'cfg': cfg, 'n_seeds': len(rows),
            'top1_mean': float(np.mean([r['top1'] for r in rows])),
            'top1_std': float(np.std([r['top1'] for r in rows])),
            'top5_mean': float(np.mean([r['top5'] for r in rows])),
            'nll_mean': float(np.mean([r['nll'] for r in rows])),
            'nll_std': float(np.std([r['nll'] for r in rows])),
            'train_loss_mean': float(np.mean([r['train_loss'] for r in rows])),
            'params': rows[0]['params'], 'steps': rows[0]['steps'],
            'per_seed': rows,
        })

    out = {
        'meta': {'mode': args.mode, 'bucket': args.bucket, 'V': V,
                 'train_chars': len(train_ids), 'test_chars': len(test_ids),
                 'n_positions': len(positions), 'seeds': seeds,
                 'epochs': args.epochs, 'batch': args.batch, 'lr': args.lr,
                 'min_freq': args.min_freq, 'hist_min': HIST_MIN,
                 'protocol': '固定评测位置集：用 ids[p-ctx:p] 预测 ids[p]，'
                             '所有档位评同一批位置（与 sweep 的「每 ctx 步一窗」不同，'
                             '后者跨 ctx 的 history 长度分布不可比）',
                 'trigram': {'top1': tri['top1'], 'top5': tri['top5'], 'nll': tri['nll']}},
        'runs': runs,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    stem = f'abl_{args.mode}_{args.bucket}_t{args.size}'
    with open(os.path.join(args.out_dir, stem + '.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    L = [f'# 消融：{"上下文长度" if args.mode == "ctx" else "因果掩码"}（{args.bucket}）', '',
         f'> 固定评测位置 {len(positions)} 个（前置历史 ≥{HIST_MIN} 字），'
         f'所有档位评同一批位置；训练 {len(train_ids)} 字，留出 {len(test_ids)} 字（{args.mode} 模式）。',
         '',
         f'**trigram 基线：Top-1 {tri["top1"]:.1f}% · NLL {tri["nll"]:.3f}**',
         '',
         '| 配置 | Top-1 | std | Top-5 | NLL | std | 训练 loss | 步数 | 参数 |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        L.append(f'| {r["label"]} | {r["top1_mean"]:.2f}% | {r["top1_std"]:.2f} | '
                 f'{r["top5_mean"]:.1f}% | {r["nll_mean"]:.3f} | {r["nll_std"]:.3f} | '
                 f'{r["train_loss_mean"]:.3f} | {r["steps"]} | {r["params"]:,} |')
    L += ['', f'> 种子 {seeds}；`总参数 = 40,800 + V×40`，pos 表随 ctx 变化（ctx×40）。', '']
    with open(os.path.join(args.out_dir, stem + '_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'✓ 写入 py/runs/{stem}.json 与 {stem}_report.md')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
