""" ============================================================================
py/ablation.py — 结构消融：注意力头数 与 位置嵌入 的留外出影响
----------------------------------------------------------------------------
目的：给《模型拆解课》两篇草稿补上真实数字（文章16 head 分工、文章21 位置编码）。
固定：ctx=32 d=40 layers=2 ffMul=4、batch=8、lr=0.004、优化器/调度/裁剪同 sweep。
唯一变量：
  - heads ∈ {1,2,4,8}（d 固定 40，每头维度 40/heads，参数规模不变）
  - pos ∈ {学习式绝对位置嵌入, 全部置零并冻结}
训练集取全文前 train_size 字，测试集取全文最后 10%（与 holdout_eval / sweep 同口径，不重叠）。
输出：py/runs/ablation_<corpus>_t<size>_s<steps>.json + ablation_report.md
用法：
  python py/ablation.py --corpus scifi --train_size 20000 --steps 800 --seeds 42,7,1234
============================================================================ """
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from minilm_torch import MiniLM, build_vocab, encode, make_batch, lr_at  # noqa
from sweep_data_scale import trigram_eval  # noqa

DEFAULT_CFG = dict(ctx=32, d=40, layers=2, heads=4, ffMul=4)


def param_count(model, include_frozen=True):
    n = 0
    for p in model.parameters():
        if include_frozen or p.requires_grad:
            n += p.numel()
    return n


def train_eval(text, train_size, cfg, steps, batch, lr, seed, zero_pos=False):
    import random as _r
    torch.manual_seed(seed)
    rng_py = _r.Random(seed)

    itos, stoi = build_vocab(text, min_freq=1)
    V = len(itos)
    full_ids = encode(text, stoi)
    split = math.floor(len(full_ids) * 0.9)
    train_ids = full_ids[:train_size]
    test_ids = full_ids[split:]
    # 安全过滤越界（理论上按全文词表不会越界）
    train_ids = [t for t in train_ids if t < V]
    test_ids = [t for t in test_ids if t < V]
    if len(train_ids) < cfg['ctx'] + 2 or len(test_ids) < cfg['ctx'] + 1:
        return None

    model = MiniLM(V, cfg)
    if zero_pos:
        model.pos.weight.data.zero_()
        model.pos.weight.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = max(10, math.floor(steps * 0.15))
    model.train()
    seg_loss, seg_n = 0.0, 0
    for step in range(steps):
        tokens, targets = make_batch(train_ids, batch, cfg['ctx'], rng_py)
        opt.zero_grad()
        _, loss = model(tokens, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps, lr, warmup)
        opt.step()
        seg_loss += loss.item()
        seg_n += 1
    final_loss = seg_loss / seg_n

    model.eval()
    top1 = top5 = total = 0
    ctx = cfg['ctx']
    s = 0
    while s + ctx + 1 <= len(test_ids):
        chunk = torch.tensor([test_ids[s:s + ctx]], dtype=torch.long)
        tgt = test_ids[s + 1:s + 1 + ctx]
        with torch.no_grad():
            logits, _ = model(chunk)
        topi = logits[0].topk(5)[1].tolist()
        for i in range(ctx):
            total += 1
            if topi[i][0] == tgt[i]:
                top1 += 1
            if tgt[i] in topi[i]:
                top5 += 1
        s += ctx

    return {
        'top1': 100.0 * top1 / max(1, total),
        'top5': 100.0 * top5 / max(1, total),
        'final_loss': final_loss,
        'n': total,
        'params': param_count(model),
        'V': V,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='scifi')
    ap.add_argument('--corpus_file', default=os.path.join('py', 'data', 'corpus.json'),
                    help='语料文件（可指向 py/data/corpus.orig.json 做原始短篇的 in-distribution 对照）')
    ap.add_argument('--train_size', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.004)
    ap.add_argument('--seeds', default='42,7,1234')
    ap.add_argument('--heads', default='1,2,4,8')
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'runs'))
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(',')]
    heads = [int(x) for x in args.heads.split(',')]

    with open(os.path.join(ROOT, args.corpus_file) if not os.path.isabs(args.corpus_file)
              else args.corpus_file, 'r', encoding='utf-8') as f:
        corpora = json.load(f)
    if args.corpus not in corpora or not corpora[args.corpus].get('text'):
        sys.exit(f'语料 {args.corpus} 不存在')
    text = corpora[args.corpus]['text']
    full_len = len(text)
    split = math.floor(full_len * 0.9)
    if args.train_size > split:   # 训练集不能越过留出集起点，否则泄漏
        print(f'    注意：train_size={args.train_size} > 留出起点 {split}，已截断')
        args.train_size = split

    ctx = DEFAULT_CFG['ctx']
    itos_full, stoi_full = build_vocab(text, min_freq=1)
    ids_full = encode(text, stoi_full)
    t0 = time.time()
    trig = trigram_eval(ids_full[:args.train_size],
                        ids_full[math.floor(full_len * 0.9):],
                        V=len(itos_full), ctx=ctx, alpha=0.1)
    print(f'[trigram] top1={trig["top1"]:.1f}% nll={trig["nll"]:.3f} '
          f'({time.time() - t0:.1f}s)', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    runs = []
    for h in heads:
        for zero in (False, True):
            if h != 4 and zero:   # 位置消融只在 heads=4 基线做
                continue
            cfg = dict(DEFAULT_CFG); cfg['heads'] = h
            label = f'h{h}{"_nopos" if zero else ""}'
            row_seeds = []
            for seed in seeds:
                print(f'>>> {label} seed={seed}', flush=True)
                t0 = time.time()
                r = train_eval(text, args.train_size, cfg, args.steps,
                               args.batch, args.lr, seed, zero_pos=zero)
                if r is None:
                    print('    skip (数据不足)', flush=True)
                    continue
                r = dict(r); r['head'] = h; r['zero_pos'] = zero; r['seed'] = seed
                r['seconds'] = round(time.time() - t0, 1)
                row_seeds.append(r)
                print(f'    top1={r["top1"]:.1f}% top5={r["top5"]:.1f}% '
                      f'loss={r["final_loss"]:.3f} params={r["params"]} {r["seconds"]}s', flush=True)
            if row_seeds:
                top1s = [x['top1'] for x in row_seeds]
                top5s = [x['top5'] for x in row_seeds]
                runs.append({
                    'label': label, 'head': h, 'zero_pos': zero, 'seeds': len(row_seeds),
                    'top1_mean': float(np.mean(top1s)), 'top1_std': float(np.std(top1s)),
                    'top5_mean': float(np.mean(top5s)), 'top5_std': float(np.std(top5s)),
                    'params': row_seeds[0]['params'],
                    'per_seed': row_seeds,
                })

    out = {
        'meta': {'corpus': args.corpus, 'corpus_chars': full_len, 'train_size': args.train_size,
                 'steps': args.steps, 'batch': args.batch, 'lr': args.lr, 'seeds': seeds,
                 'cfg': DEFAULT_CFG, 'trigram_top1': trig['top1'], 'trigram_top5': trig['top5'],
                 'trigram_nll': trig['nll']},
        'runs': runs,
    }
    fn = os.path.join(args.out_dir, f'ablation_{args.corpus}_t{args.train_size}_s{args.steps}.json')
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n✓ 写入 {fn}')

    # markdown
    L = []
    L.append(f'# 结构消融：head 数 × 位置嵌入（{args.corpus}）')
    L.append('')
    L.append(f'> 训练集前 {args.train_size} 字，测试集全文最后 10%（{out["meta"]["corpus_chars"]} 字全文）。')
    L.append(f'> 固定 ctx=32 d=40 layers=2 ffMul=4、steps={args.steps}、batch={args.batch}、lr={args.lr}、'
             f'种子 {seeds}。')
    L.append('')
    L.append(f'**trigram 基线 Top-1 = {trig["top1"]:.1f}%**（α=0.1，固定不挑）。')
    L.append('')
    L.append('| 配置 | 种子数 | Top-1 均值 | Top-1 std | Top-5 均值 | 参数 | 对比 trigram |')
    L.append('|---|---:|---:|---:|---:|---:|---|')
    for r in runs:
        vs = '胜' if r['top1_mean'] > trig['top1'] else '负'
        L.append(f'| {r["label"]} | {r["seeds"]} | {r["top1_mean"]:.1f}% | {r["top1_std"]:.1f} | '
                 f'{r["top5_mean"]:.1f}% | {r["params"]:,} | {vs} |')
    L.append('')
    L.append('> 注意：heads 1/2/4/8 的**参数规模完全相同**——head 数只改变注意力内部如何切分，不改变总容量。')
    L.append('> 因此 head 数的差异纯粹是「表达方式」的差异，应理解为结构而非预算。')
    md = os.path.join(args.out_dir, f'ablation_{args.corpus}_t{args.train_size}_s{args.steps}_report.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'✓ 写入 {md}')


if __name__ == '__main__':
    main()
