""" ============================================================================
py/verify_seed.py — 种子可复现性验证（含「旧的错误顺序」失败样本）
----------------------------------------------------------------------------
验证三件事：
  T1 正确顺序（set_global_seed → 建模型 → 训练）：同 seed 两次，初始权重与
     末步权重应逐位相同（max|Δ| = 0），loss 轨迹相同。
  T2 错误顺序（建模型 → set_global_seed → 训练）：同 seed 两次，初始权重**不同**。
     这是修复前的 py/train.py 真实行为（模型在 manual_seed 之前构造），
     用来证明「seed 设了但没管事」不是猜想。
  T3 不同 seed：末步权重应显著不同 —— 反证 T1 的「相同」不是因为训练没起作用。

用法：
  python py/verify_seed.py                     # 默认 story 桶 4 万字切片 / 30 步
  python py/verify_seed.py --steps 60 --chars 60000 --out dev/verify/verify_seed.json
============================================================================ """
from __future__ import annotations
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from minilm_torch import MiniLM, build_vocab, encode, make_batch, lr_at  # noqa
from repro import set_global_seed, random_seed_bundle  # noqa
import splits as splits_mod  # noqa

CFG = dict(ctx=32, d=40, layers=2, heads=4, ffMul=4)


def flat_diff(sd_a: dict, sd_b: dict) -> float:
    """两组 state_dict 的最大绝对差。"""
    m = 0.0
    for k in sd_a:
        d = (sd_a[k].float() - sd_b[k].float()).abs().max().item()
        m = max(m, d)
    return m


def run(seed: int, text: str, steps: int, batch: int, lr: float, seed_before_model: bool):
    """跑一段训练。seed_before_model=False 复刻修复前的错误顺序。"""
    if seed_before_model:
        set_global_seed(seed)
        itos, stoi = build_vocab(text)
        model = MiniLM(len(itos), CFG)
        initial = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        itos, stoi = build_vocab(text)
        model = MiniLM(len(itos), CFG)                      # ← 先建模型
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        set_global_seed(seed)                               # ← 后设种子（管不到初始权重）

    ids = encode(text, stoi)
    rng = random_seed_bundle(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = max(10, int(steps * 0.15))
    losses = []
    model.train()
    for step in range(steps):
        tokens, targets = make_batch(ids, batch, CFG['ctx'], rng)
        opt.zero_grad()
        _, loss = model(tokens, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps, lr, warmup)
        opt.step()
        losses.append(round(loss.item(), 6))
    return initial, {k: v.clone() for k, v in model.state_dict().items()}, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bucket', default='story')
    ap.add_argument('--chars', type=int, default=40000)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.004)
    ap.add_argument('--out', default=os.path.join(ROOT, 'dev', 'verify', 'verify_seed.json'))
    args = ap.parse_args()

    text = splits_mod.bucket_joined(args.bucket)['train'][:args.chars]
    print(f'[*] 语料 {args.bucket} 前 {len(text)} 字 / {args.steps} 步 / batch={args.batch}')

    rep = {'protocol': {'bucket': args.bucket, 'chars': len(text), 'steps': args.steps,
                        'batch': args.batch, 'lr': args.lr, 'cfg': CFG}}

    # T1 正确顺序：同 seed 两次
    i1, f1, l1 = run(1234, text, args.steps, args.batch, args.lr, True)
    i2, f2, l2 = run(1234, text, args.steps, args.batch, args.lr, True)
    rep['T1_seed_before_model'] = {
        'init_max_abs_diff': flat_diff(i1, i2),
        'final_max_abs_diff': flat_diff(f1, f2),
        'loss_trajectory_identical': l1 == l2,
        'loss_head': l1[:3], 'loss_tail': l1[-3:],
    }

    # T2 错误顺序（修复前的 train.py 行为）
    i3, f3, l3 = run(1234, text, args.steps, args.batch, args.lr, False)
    i4, f4, l4 = run(1234, text, args.steps, args.batch, args.lr, False)
    rep['T2_seed_after_model_old_behavior'] = {
        'init_max_abs_diff': flat_diff(i3, i4),
        'final_max_abs_diff': flat_diff(f3, f4),
        'loss_trajectory_identical': l3 == l4,
        'loss_head': l3[:3], 'loss_tail': l3[-3:],
    }

    # T3 不同 seed
    _, f5, l5 = run(4321, text, args.steps, args.batch, args.lr, True)
    rep['T3_different_seed'] = {
        'final_max_abs_diff_vs_seed1234': flat_diff(f1, f5),
        'loss_tail': l5[-3:],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print('\n== 结果 ==')
    t1, t2, t3 = rep['T1_seed_before_model'], rep['T2_seed_after_model_old_behavior'], rep['T3_different_seed']
    print(f"T1 正确顺序  同 seed 两次：init max|Δ|={t1['init_max_abs_diff']:.3e}  "
          f"final max|Δ|={t1['final_max_abs_diff']:.3e}  loss 逐位相同={t1['loss_trajectory_identical']}")
    print(f"T2 旧顺序    同 seed 两次：init max|Δ|={t2['init_max_abs_diff']:.3e}  "
          f"final max|Δ|={t2['final_max_abs_diff']:.3e}  loss 逐位相同={t2['loss_trajectory_identical']}")
    print(f"T3 不同 seed 末步 max|Δ|={t3['final_max_abs_diff_vs_seed1234']:.3e}")
    print(f'[*] 报告 -> {args.out}')

    ok = (t1['init_max_abs_diff'] == 0.0 and t1['final_max_abs_diff'] == 0.0
          and t2['init_max_abs_diff'] > 0 and t3['final_max_abs_diff_vs_seed1234'] > 0)
    print(f'[*] 判定：{"通过（修复生效，旧顺序确实不可复现）" if ok else "★失败★"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
