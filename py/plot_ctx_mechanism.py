#!/usr/bin/env python3
"""
py/plot_ctx_mechanism.py — 画「机制对照」图：把各臂的曲线按**训练步数**对齐。

背景：扫 ctx 时 `steps = epochs × 语料量 / (batch × ctx)`，所以 ctx 一变、步数就变。
主对照（同 token 预算）下 ctx=64 看起来"更差"，但那可能只是**更新次数减半**。
本图把不同 ctx / 不同 epoch 的臂画在同一根「步数」轴上：
  若同一 ctx 的曲线随步数下移（而不是因 ctx 更大而整体更高），
  就说明起作用的是步数，不是上下文长度。

x 轴 = 训练步数；y 轴 = 留出集 NLL（上排）与 Top-1（下排）；
每个子图一个桶；虚线 = trigram（≈常数基准）。
"""
import argparse
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 臂定义：(显示名, tag, 颜色, 线型)
ARMS = [
    ('ctx32 · 3ep（586 步）', '', '#1f77b4', '-'),
    ('ctx64 · 3ep（293 步）', 'ctx64', '#d62728', '-'),
    ('ctx32 · 1.5ep（293 步）', 'ctx32e15', '#1f77b4', '--'),
    ('ctx128 · 3ep（146 步）', 'ctx128', '#2ca02c', '-'),
    ('ctx64 · 3ep · lr×2（293 步）', 'ctx64lr8', '#9467bd', '-'),
    ('ctx64 · 6ep（586 步）', 'ctx64e6', '#d62728', '--'),
    ('ctx128 · 12ep（586 步）', 'ctx128e12', '#2ca02c', '--'),
]
BUCKETS = ('gudian', 'wuxia', 'urban')


def _cn_font(matplotlib):
    for cand in ('Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC'):
        try:
            matplotlib.rcParams['font.sans-serif'] = [cand]
            matplotlib.rcParams['axes.unicode_minus'] = False
            break
        except Exception:
            continue


def load_arm(tag, runs_dir, size_label='200000'):
    out = {}
    for fp in sorted(glob.glob(os.path.join(runs_dir, 'wsplit_*.json'))):
        base = os.path.basename(fp)
        if base.startswith('wsplit_summary') or base.startswith('wsplit_tri_'):
            continue
        with open(fp, 'r', encoding='utf-8') as f:
            d = json.load(f)
        if 'curve' not in d or d.get('tag', '') != tag or d.get('size_label') != size_label:
            continue
        out.setdefault(d['bucket'], []).append(d)
    return out


def mean_curve(rs, key):
    """按 step 对齐后取多种子均值：(steps, mean, n)"""
    by = {}
    for r in rs:
        for p in r['curve']:
            by.setdefault(p['step'], []).append(p[key])
    xs = sorted(by)
    return xs, [statistics.mean(by[x]) for x in xs], len(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'img', 'ctx_mechanism.png'))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _cn_font(matplotlib)

    data = {tag: load_arm(tag, args.runs_dir) for _, tag, _, _ in ARMS}

    fig, axes = plt.subplots(2, len(BUCKETS), figsize=(5.0 * len(BUCKETS), 8.4), sharex='col')
    for j, bk in enumerate(BUCKETS):
        tri_nll = data[''][bk][0]['trigram']['nll']
        for i, key in enumerate(('nll', 'top1')):
            ax = axes[i][j]
            for name, tag, color, ls in ARMS:
                rs = data[tag].get(bk)
                if not rs:
                    continue
                xs, ys, n = mean_curve(rs, key)
                ax.plot(xs, ys, ls, color=color, marker='o', ms=3.2, lw=1.5,
                        label=f'{name}' + (f' · n={n}' if n else ''))
            if key == 'nll':
                ax.axhline(tri_nll, color='#888888', ls=':', lw=1.3,
                           label='trigram（α=0.1）')
            ax.set_title(f'{bk} · {key.upper()}', fontsize=11)
            ax.set_xlabel('训练步数')
            ax.grid(alpha=0.25)
            if key == 'nll':
                ax.set_ylabel('留出集 NLL（越低越好）')
            else:
                ax.set_ylabel('留出集 Top-1 %（越高越好）')
    axes[0][0].legend(fontsize=7.4, loc='upper right', framealpha=0.9)
    fig.suptitle('机制对照：按「训练步数」对齐后，ctx 的差异基本消失，剩下的是步数效应', fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f'✓ 图写入 {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
