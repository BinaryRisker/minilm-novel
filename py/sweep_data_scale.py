""" ============================================================================
py/sweep_data_scale.py — 回答一个核心问题：
    "模型到底需要多少训练字才能超过 trigram？"
----------------------------------------------------------------------------
设计原则（PRD P2 核心）：
- 控制变量：固定模型配置（ctx=32 d=40 layers=2 heads=4 ffMul=4，5.5万参数）、
            固定优化器（Adam lr=0.004 + warmup+cosine）、固定步数 400、
            固定 seed=1234、只改训练集大小。
- 评测位一致：测试集始终是「全文最后 10%」，不论训练集大小。
            这样所有 sweep 配置都在同一份留出集上比较。
- 基线一致：trigram 用同一份训练字同一份 α 选择规则，
            与 eval/baselines.js 的 JS 实现严格等价（见 py/baselines.py）。

输出：
  py/runs/sweep_<corpus>_<size>_s<seed>.json   每次跑的单条结果
  py/runs/sweep_summary.json                   汇总（4 尺寸 × 3 语料）
  py/runs/sweep_report.md                      表格 + 结论
  docs/img/sweep_data_scale.png                折线图（Model vs Trigram，3 语料）

用法：
  python py/sweep_data_scale.py
  python py/sweep_data_scale.py --sizes 500,1000,2000 --steps 200 --seed 42
============================================================================ """
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from typing import List, Dict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from minilm_torch import MiniLM, build_vocab, encode, train, holdout_eval  # noqa
from baselines import NGram, Acc, run_baselines, for_each_position  # noqa

DEFAULT_CFG = dict(ctx=32, d=40, layers=2, heads=4, ffMul=4)
DEFAULT_SIZES = (500, 1000, 2000)         # 'all' 单独处理为 -1
CORPORA = ('scifi', 'wuxia', 'urban')


def load_corpus() -> Dict[str, str]:
    with open(os.path.join(HERE, 'data', 'corpus.json'), 'r', encoding='utf-8') as f:
        c = json.load(f)
    return {k: v['text'] for k, v in c.items() if v.get('text')}


def trigram_eval(train_ids: List[int], test_ids: List[int], V: int, ctx: int = 32,
                 alpha: float = 0.1) -> Dict:
    """固定 α=0.1（不挑）跑 trigram baseline。返回 {top1, top5, nll, ppl, backoff}。
    这里不挑 α 是因为 sweep 的目的是看「数据量对模型/基线的影响」，
    挑 α 会在每个尺寸上重新搜一遍、引入额外随机性，让对比更难解释。"""
    m = NGram(train_ids, order=3, alpha=alpha, V=V)
    acc = Acc()
    backoff = {1: 0, 2: 0, 3: 0}
    n = 0
    for hist, tgt in for_each_position(test_ids, ctx):
        acc.add(m.dist(hist), tgt)
        backoff[m.used_order(hist)] += 1
        n += 1
    s = acc.summary()
    return {**s, 'backoff': backoff}


def model_eval(text: str, train_ids: List[int], test_ids: List[int],
               vocab_chars: List[str], cfg: dict, steps: int,
               batch: int, lr: float, seed: int) -> Dict:
    """用同样的 train/test split 训练模型并评测。返回 {top1, top5, final_loss}。
    关键约束：模型词表必须在全文上建（与 JS/复刻一致），不能只看 train，
    否则 test_ids 会出现「vocab 越界」越界访问。"""
    import random as _r
    rng_py = _r.Random(seed)
    torch.manual_seed(seed)

    # 词表 = 全文，与 JS 的 kit.buildVocab(text) 一致
    itos_full, stoi_full = build_vocab(text)
    V = len(itos_full)
    # train/test ids 已经是按全文 vocab 编出的，直接复用即可
    train_ids_v = [t for t in train_ids if t < V]   # 安全过滤
    test_ids_v = [t for t in test_ids if t < V]

    if len(train_ids_v) < cfg['ctx'] + 2:
        return {'top1': 0.0, 'top5': 0.0, 'final_loss': float('nan'), 'n': 0}

    model = MiniLM(V, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = max(10, math.floor(steps * 0.15))

    from minilm_torch import make_batch, lr_at
    model.train()
    seg_loss, seg_n = 0.0, 0
    for step in range(steps):
        tokens, targets = make_batch(train_ids_v, batch, cfg['ctx'], rng_py)
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

    # 留出集评测（基于 test_ids_v，与 sweep 用的 test 切分一致）
    model.eval()
    top1 = top5 = total = 0
    s = 0
    ctx = cfg['ctx']
    while s + ctx + 1 <= len(test_ids_v):
        chunk = torch.tensor([test_ids_v[s:s + ctx]], dtype=torch.long)
        tgt = test_ids_v[s + 1:s + 1 + ctx]
        with torch.no_grad():
            logits, _ = model(chunk)
        topv, topi = logits[0].topk(5)
        pred = topi.tolist()
        for i in range(ctx):
            if pred[i][0] == tgt[i]:
                top1 += 1
            if tgt[i] in pred[i]:
                top5 += 1
            total += 1
        s += ctx
    return {
        'top1': 100.0 * top1 / max(1, total),
        'top5': 100.0 * top5 / max(1, total),
        'final_loss': final_loss,
        'n': total,
    }


def sweep_one(corpus_key: str, full_text: str, size: int,
              cfg: dict, steps: int, batch: int, lr: float, seed: int,
              ctx: int) -> Dict:
    """单个 (corpus, size) 组合。返回一条 run dict。"""
    itos, stoi = build_vocab(full_text)
    V = len(itos)
    full_ids = encode(full_text, stoi)
    # 测试集：始终是全文最后 10%（与 JS/复刻一致）
    test_cut = int(math.floor(len(full_ids) * 0.9))
    test_ids = full_ids[test_cut:]
    # 训练集：'all' → floor(0.9*N)（与 JS/复刻一致）；其他 size → 前 size 个字（必须 ≤ test_cut 以保证不重叠）
    if size < 0:
        train_ids = full_ids[:test_cut]
        actual_size = len(train_ids)
        size_label = 'all'
    else:
        if size > test_cut:
            # size 超过 90% 切点会被截断到 test_cut，避免与测试重叠；提示用户
            print(f'    注意：size={size} > test_cut={test_cut}，已截断到 {test_cut}')
            size = test_cut
        train_ids = full_ids[:size]
        actual_size = len(train_ids)
        size_label = str(size)

    # 确保 train_ids 与 test_ids 完全不重叠
    assert test_cut >= actual_size, f'训练集 {actual_size} 越界测试起点 {test_cut}'

    vocab_chars = itos[1:]   # 去掉 0 号未登录位（仅供 model_eval 兼容，实际内部重建全文词表）

    t0 = time.time()
    tri = trigram_eval(train_ids, test_ids, V=V, ctx=ctx, alpha=0.1)
    t_trigram = time.time() - t0

    t0 = time.time()
    mod = model_eval(full_text, train_ids, test_ids, vocab_chars,
                     cfg=cfg, steps=steps, batch=batch, lr=lr, seed=seed)
    t_model = time.time() - t0

    return {
        'corpus': corpus_key,
        'size': actual_size,
        'size_label': size_label,
        'seed': seed,
        'steps': steps,
        'batch': batch,
        'lr': lr,
        'cfg': cfg,
        'V': V,
        'train_chars': len(train_ids),
        'test_chars': len(test_ids),
        'n_positions': mod['n'],
        'model': {
            'top1': mod['top1'],
            'top5': mod['top5'],
            'final_loss': mod['final_loss'],
        },
        'trigram': {
            'top1': tri['top1'],
            'top5': tri['top5'],
            'nll': tri['nll'],
            'ppl': tri['ppl'],
            'backoff': tri['backoff'],
        },
        'wallclock': {'model': t_model, 'trigram': t_trigram},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default=','.join(str(s) for s in DEFAULT_SIZES),
                    help='comma-separated training sizes; add "all" for the full text')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.004)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--ctx', type=int, default=DEFAULT_CFG['ctx'])
    ap.add_argument('--corpus', default='all', help='all 或单个语料 id')
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--no_plot', action='store_true', help='跳过画图')
    args = ap.parse_args()

    sizes_arg = [s.strip() for s in args.sizes.split(',')]
    sizes = [-1 if s == 'all' else int(s) for s in sizes_arg]
    cfg = dict(DEFAULT_CFG)
    cfg['ctx'] = args.ctx

    corpora = load_corpus()
    keys = [args.corpus] if args.corpus != 'all' else CORPORA
    keys = [k for k in keys if k in corpora]
    if not keys:
        sys.exit('没有可用的语料')

    os.makedirs(args.out_dir, exist_ok=True)

    all_runs = []
    for ck in keys:
        text = corpora[ck]
        for sz in sizes:
            label = 'all' if sz < 0 else str(sz)
            print(f'>>> {ck} size={label} ...', flush=True)
            try:
                r = sweep_one(ck, text, sz, cfg=cfg, steps=args.steps,
                              batch=args.batch, lr=args.lr, seed=args.seed,
                              ctx=args.ctx)
                all_runs.append(r)
                print(f'    model Top-1={r["model"]["top1"]:.1f}%  '
                      f'trigram Top-1={r["trigram"]["top1"]:.1f}%  '
                      f'wallclock={r["wallclock"]["model"]:.1f}s',
                      flush=True)
                # 单条 JSON
                fn = f'sweep_{ck}_{label}_s{args.seed}.json'
                with open(os.path.join(args.out_dir, fn), 'w', encoding='utf-8') as f:
                    json.dump(r, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f'    FAILED: {e}', flush=True)

    summary_path = os.path.join(args.out_dir, 'sweep_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'meta': {
                'cfg': cfg,
                'steps': args.steps,
                'batch': args.batch,
                'lr': args.lr,
                'seed': args.seed,
                'sizes': sizes,
                'corpora': keys,
            },
            'runs': all_runs,
        }, f, ensure_ascii=False, indent=2)
    print(f'\n✓ 汇总写入 {summary_path}（{len(all_runs)} 条）')

    # Markdown 报告
    write_report(all_runs, args)

    # 画图
    if not args.no_plot and all_runs:
        try:
            plot_sweep(all_runs, cfg, args)
        except Exception as e:
            print(f'画图失败（不影响主结果）：{e}')


def write_report(runs: List[Dict], args):
    """生成 py/runs/sweep_report.md：按语料分组，每个语料一张表。"""
    # 去重 (corpus, actual_size) — 当 'all' 与某个 size 截断到同一大小时，保留 size_label='all' 那一条
    best_by_key = {}
    for r in runs:
        k = (r['corpus'], r['size'])
        prev = best_by_key.get(k)
        if prev is None or (r['size_label'] == 'all' and prev['size_label'] != 'all'):
            best_by_key[k] = r
    runs = list(best_by_key.values())

    by_corpus = {}
    for r in runs:
        by_corpus.setdefault(r['corpus'], []).append(r)
    L = []
    L.append('# 数据量扫描：模型 vs Trigram')
    L.append('')
    L.append('> 回答「模型到底需要多少训练字才能超过 trigram？」')
    L.append(f'> 固定 seed={args.seed}、steps={args.steps}、batch={args.batch}、lr={args.lr}、'
             f'ctx={args.ctx}；只改训练集大小。测试集是全文最后 10%，所有配置在同一份留出集上比较。')
    L.append('')
    for ck, rows in by_corpus.items():
        rows.sort(key=lambda r: r['size'])
        L.append(f'## {ck}')
        L.append('')
        L.append('| 训练字数 | 留出集字数 | Model Top-1 | Model Top-5 | Trigram Top-1 | Trigram NLL | 胜负 |')
        L.append('|---|---|---|---|---|---|---|')
        for r in rows:
            win = '✓ 模型胜' if r['model']['top1'] > r['trigram']['top1'] else '✗ 模型负'
            L.append(f'| {r["size"]} | {r["test_chars"]} | '
                     f'{r["model"]["top1"]:.1f}% | {r["model"]["top5"]:.1f}% | '
                     f'{r["trigram"]["top1"]:.1f}% | {r["trigram"]["nll"]:.3f} | {win} |')
        L.append('')
    out = os.path.join(HERE, 'runs', 'sweep_report.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'✓ 报告写入 {out}')


def plot_sweep(runs: List[Dict], cfg: dict, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 中文字体
    for cand in ('Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC'):
        try:
            matplotlib.rcParams['font.sans-serif'] = [cand]
            matplotlib.rcParams['axes.unicode_minus'] = False
            break
        except Exception:
            continue

    # 去重：保留 size_label='all' 的那条
    best_by_key = {}
    for r in runs:
        k = (r['corpus'], r['size'])
        prev = best_by_key.get(k)
        if prev is None or (r['size_label'] == 'all' and prev['size_label'] != 'all'):
            best_by_key[k] = r
    runs = list(best_by_key.values())

    by_corpus = {}
    for r in runs:
        by_corpus.setdefault(r['corpus'], []).append(r)
    for arr in by_corpus.values():
        arr.sort(key=lambda r: r['size'])

    fig, axes = plt.subplots(1, len(by_corpus), figsize=(5 * len(by_corpus), 4),
                             sharey=True)
    if len(by_corpus) == 1:
        axes = [axes]
    palette = {'scifi': '#d62728', 'wuxia': '#1f77b4', 'urban': '#2ca02c'}

    for ax, (ck, rows) in zip(axes, by_corpus.items()):
        xs = [r['size'] for r in rows]
        m_t1 = [r['model']['top1'] for r in rows]
        t_t1 = [r['trigram']['top1'] for r in rows]
        ax.plot(xs, m_t1, 'o-', color=palette.get(ck, 'gray'),
                label='Model (MiniLM)', linewidth=2, markersize=8)
        ax.plot(xs, t_t1, 's--', color=palette.get(ck, 'gray'), alpha=0.6,
                label='Trigram (α=0.1)', linewidth=1.5, markersize=7)
        ax.set_title(f'{ck}', fontsize=13)
        ax.set_xlabel('训练字数')
        ax.set_xticks(xs)
        ax.set_xticklabels([r['size_label'] for r in rows], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    axes[0].set_ylabel('留出集 Top-1 (%)')
    fig.suptitle('数据量扫描 · Model vs Trigram (固定 steps=400 seed=1234)',
                 fontsize=14)
    fig.tight_layout()

    out_img = os.path.join(HERE, '..', 'docs', 'img', 'sweep_data_scale.png')
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    fig.savefig(out_img, dpi=120, bbox_inches='tight')
    print(f'✓ 图写入 {out_img}')


if __name__ == '__main__':
    main()