#!/usr/bin/env python3
"""
py/compare_budget.py — 「预算给足」对照：固定 ctx64 / 20万字档，只改训练步数。

回答的问题：上轮机制归因显示 586 步时 Top-1 边距已接近转正（−0.09），
那么缺口是否随**优化预算**单调消失？还是 20 万字档早已饱和 / 过拟合？

口径：
  固定 ctx=64、sizes=200000、batch=32、cfg d40/L2/H4、A 级 3 桶、3 种子；
  步数 = epochs × 200000 / (batch×ctx)：293（3ep）→ 586（6ep）→ 1172（12ep）→ 2344（24ep）。
  lr 调度（15% warmup + 余弦衰减）按各自总步数归一 —— 每条臂都是「完整调度」，
  这正是「预算给足」与上轮「同 token 不同步数」的差别。
  基线 trigram 与步数无关，边距 = 模型 − trigram 直接可比。

用法：
  python py/compare_budget.py                # 表 + 图（docs/img/budget_steps.png）
  python py/compare_budget.py --no_plot      # 只出表
"""
import argparse
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 步数曲线臂：(显示名, tag, 步数, 颜色)
ARMS = [
    ('ctx64 · 3ep',  'ctx64',    293,  '#d62728'),
    ('ctx64 · 6ep',  'ctx64e6',  586,  '#ff7f0e'),
    ('ctx64 · 12ep', 'ctx64e12', 1172, '#2ca02c'),
    ('ctx64 · 24ep', 'ctx64e24', 2344, '#1f77b4'),
]
# 参考臂（不进曲线，只在表里做参照）：ctx32 · 3ep = 586 步（上轮主网格基准）
REF_ARMS = [('ctx32 · 3ep（参照）', '', 586, '#1f77b4')]
BUCKETS = ('gudian', 'wuxia', 'urban')
SIZE_LABEL = '200000'


def _cn_font(matplotlib):
    for cand in ('Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC'):
        try:
            matplotlib.rcParams['font.sans-serif'] = [cand]
            matplotlib.rcParams['axes.unicode_minus'] = False
            break
        except Exception:
            continue


def load_arm(tag, runs_dir, size_label=SIZE_LABEL):
    out = {}
    for fp in sorted(glob.glob(os.path.join(runs_dir, 'wsplit_*.json'))):
        base = os.path.basename(fp)
        if base.startswith('wsplit_summary') or base.startswith('wsplit_tri_'):
            continue
        with open(fp, 'r', encoding='utf-8') as f:
            d = json.load(f)
        if 'final' not in d or d.get('tag', '') != tag or d.get('size_label') != size_label:
            continue
        if d['bucket'] not in BUCKETS:
            continue
        out.setdefault(d['bucket'], []).append(d)
    return out


def margin(rs, key):
    """多种子均值边距（模型 − trigram）；nll 为负 = 模型赢。"""
    tri = rs[0]['trigram']
    return statistics.mean([r['final'][key] for r in rs]) - tri[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'img', 'budget_steps.png'))
    ap.add_argument('--no_plot', action='store_true')
    args = ap.parse_args()

    data = {}
    for name, tag, steps, _ in ARMS + REF_ARMS:
        rs = load_arm(tag, args.runs_dir)
        data[tag] = rs
        missing = [bk for bk in BUCKETS if not rs.get(bk)]
        if missing:
            print(f'⚠️  臂 {name!r}（tag={tag!r}）缺桶：{missing} —— 相关行将留空')

    # ① 核对：trigram 与步数无关（各臂应逐桶一致）
    print('① trigram 基线核对（同桶各臂应一致，与步数无关）')
    for bk in BUCKETS:
        vals = []
        for _, tag, _, _ in ARMS + REF_ARMS:
            rs = data[tag].get(bk)
            vals.append(f"{rs[0]['trigram']['top1']:.2f}/{rs[0]['trigram']['nll']:.3f}" if rs else '—')
        print(f"  {bk:>7}: " + '  '.join(vals))
    print()

    # ② 边距 × 步数表
    METRICS = (('top1', 'Top-1', '{:+7.2f}', '正 = 模型赢'),
               ('top5', 'Top-5', '{:+7.2f}', '正 = 模型赢'),
               ('nll', 'NLL', '{:+8.3f}', '负 = 模型赢'))
    col_head = [n for n, _, _, _ in ARMS + REF_ARMS]
    for mkey, mname, fmt, better in METRICS:
        print(f'② 边距（模型 − trigram）· {mname}（{better}）')
        h = f"{'bucket':>7} {'步数':>6} | " + ' '.join(f'{n.split(" ·")[0] + "@" + str(s):>13}'
                                                      for (_, _, s, _), n in zip(ARMS + REF_ARMS, col_head))
        print(h)
        print('-' * len(h))
        for bk in BUCKETS:
            cells = []
            for _, tag, _, _ in ARMS + REF_ARMS:
                rs = data[tag].get(bk)
                cells.append(fmt.format(margin(rs, mkey)) if rs else '      —')
            print(f'{bk:>7} {"":>6} | ' + ' '.join(f'{c:>13}' for c in cells))
        print()

    # ③ 模型绝对值 + 训练时长
    print('③ 模型绝对值（Top-1% / NLL）与单条训练时长')
    for bk in BUCKETS:
        cells = []
        for _, tag, _, _ in ARMS:
            rs = data[tag].get(bk)
            if not rs:
                cells.append('—')
                continue
            t = statistics.mean(r['final']['top1'] for r in rs)
            n_ = statistics.mean(r['final']['nll'] for r in rs)
            w = statistics.mean(r['wallclock']['train'] for r in rs)
            cells.append(f'{t:.2f}/{n_:.3f} ({w/60:.1f}m)')
        print(f'  {bk:>7}: ' + '  '.join(f'{c:>24}' for c in cells))
    print()

    # ④ 汇总：边距随步数是否单调改善 / 是否转正
    for mkey, mname, _, _ in METRICS:
        sign = -1 if mkey == 'nll' else 1
        mono = 0
        for bk in BUCKETS:
            ms = [margin(data[tag][bk], mkey) for _, tag, _, _ in ARMS
                  if data[tag].get(bk)]
            if len(ms) == len(ARMS) and all(sign * (ms[i + 1] - ms[i]) >= 0 for i in range(len(ms) - 1)):
                mono += 1
        print(f'④ {mname} 边距随步数单调改善：{mono}/{len(BUCKETS)} 桶')

    if args.no_plot:
        return

    # ⑤ 图：边距 × 步数（2 行 × 3 桶），零线 = 与 trigram 打平
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _cn_font(matplotlib)

    fig, axes = plt.subplots(2, len(BUCKETS), figsize=(4.8 * len(BUCKETS), 7.6))
    for j, bk in enumerate(BUCKETS):
        for i, (mkey, mname, _, better) in enumerate((METRICS[0], METRICS[2])):
            ax = axes[i][j]
            for name, tag, steps, color in ARMS:
                rs = data[tag].get(bk)
                if not rs:
                    continue
                ax.plot(steps, margin(rs, mkey), 'o', color=color, ms=7)
                ax.annotate(f'{margin(rs, mkey):+.2f}' if mkey != 'nll' else f'{margin(rs, mkey):+.2f}',
                            (steps, margin(rs, mkey)), textcoords='offset points',
                            xytext=(6, 4), fontsize=8, color=color)
            ax.axhline(0, color='#888888', ls='--', lw=1.2)
            ax.set_xscale('log', base=2)
            ax.set_xticks([a[2] for a in ARMS])
            ax.set_xticklabels([str(a[2]) for a in ARMS])
            ax.set_title(f'{bk} · {mname} 边距', fontsize=11)
            ax.set_xlabel('训练步数（ctx64 · 20万字档）')
            ax.grid(alpha=0.25)
            if j == 0:
                ax.set_ylabel(f'{mname} 边距（{better}）')
    fig.suptitle('预算给足：ctx64 固定，步数 293→2344 —— Top-1 边距是否随预算转正', fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f'✓ 图写入 {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
