#!/usr/bin/env python3
"""
py/compare_datascale.py — 数据量扩展对照：tag=ctx64 家族内跨档位（200k→5M）的边距走势。

回答的问题：数据量加大后，模型与 trigram 的 Top-1 缺口是继续变宽（容量轮的担忧），
还是在「步数给足」（本档 2930~7325 步，远超 200k 档实测阈值 1172）下收敛？

口径：
  固定 tag='ctx64'（ctx=64、batch=32、lr=0.004、cfg d40/L2/H4、3 epoch、3 种子）；
  步数随档位 = 3×size/2048：200k=293 → 1M=1465 → 2M=2930 → 4M=5860 → 5M=7325。
  trigram 基线随档位重算（训练语料变多 → 基线也变强），所以看边距 = 模型 − trigram；
  同时打印两者绝对值的增量分解（谁涨得快）。

用法：
  python py/compare_datascale.py               # 表 + 图（docs/img/datascale_margins.png）
  python py/compare_datascale.py --tag ctx64   # 换 tag
"""
import argparse
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SIZES = ['200000', '1000000', '2000000', '4000000', '5000000']
BUCKETS = ('gudian', 'wuxia', 'urban')
COLORS = {'gudian': '#d62728', 'wuxia': '#2ca02c', 'urban': '#1f77b4'}


def _cn_font(matplotlib):
    for cand in ('Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC'):
        try:
            matplotlib.rcParams['font.sans-serif'] = [cand]
            matplotlib.rcParams['axes.unicode_minus'] = False
            break
        except Exception:
            continue


def load(tag, runs_dir):
    """{(bucket, size_label): [run, ...]}，同格多种子。"""
    out = {}
    for fp in sorted(glob.glob(os.path.join(runs_dir, 'wsplit_*.json'))):
        base = os.path.basename(fp)
        if base.startswith('wsplit_summary') or base.startswith('wsplit_tri_'):
            continue
        with open(fp, 'r', encoding='utf-8') as f:
            d = json.load(f)
        if 'final' not in d or d.get('tag', '') != tag:
            continue
        if d['bucket'] not in BUCKETS or d['size_label'] not in SIZES:
            continue
        out.setdefault((d['bucket'], d['size_label']), []).append(d)
    return out


def margin(rs, key):
    tri = rs[0]['trigram']
    return statistics.mean([r['final'][key] for r in rs]) - tri[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='ctx64')
    ap.add_argument('--runs_dir', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'img', 'datascale_margins.png'))
    ap.add_argument('--no_plot', action='store_true')
    args = ap.parse_args()

    data = load(args.tag, args.runs_dir)

    # ① 主表：边距 × 档位
    METRICS = (('top1', 'Top-1', '{:+7.2f}'), ('top5', 'Top-5', '{:+7.2f}'), ('nll', 'NLL', '{:+8.3f}'))
    for mkey, mname, fmt in METRICS:
        sign = -1 if mkey == 'nll' else 1
        print(f'① 边距（模型 − trigram）· {mname}' + ('（负 = 模型赢）' if mkey == 'nll' else '（正 = 模型赢）'))
        h = f"{'bucket':>7} | " + ' '.join(f'{s:>9}' for s in SIZES)
        print(h)
        print('-' * len(h))
        for bk in BUCKETS:
            cells = []
            for sz in SIZES:
                rs = data.get((bk, sz))
                cells.append(fmt.format(margin(rs, mkey)) if rs else '      —')
            print(f'{bk:>7} | ' + ' '.join(f'{c:>9}' for c in cells))
        print()

    # ② 绝对值与增量分解：trigram vs 模型，从最小档到最大档各涨多少
    print('② 绝对值增量分解（最小档 → 最大档，pt / nat）')
    for bk in BUCKETS:
        have = [(sz, data[(bk, sz)]) for sz in SIZES if (bk, sz) in data]
        if len(have) < 2:
            continue
        s0, r0 = have[0]
        s1, r1 = have[-1]
        tri_d = r1[0]['trigram']['top1'] - r0[0]['trigram']['top1']
        mod_d = statistics.mean(x['final']['top1'] for x in r1) - statistics.mean(x['final']['top1'] for x in r0)
        tri_n = r1[0]['trigram']['nll'] - r0[0]['trigram']['nll']
        mod_n = statistics.mean(x['final']['nll'] for x in r1) - statistics.mean(x['final']['nll'] for x in r0)
        print(f'  {bk:>7} {s0}→{s1}: Top-1 trigram {tri_d:+.2f} vs 模型 {mod_d:+.2f} | '
              f'NLL trigram {tri_n:+.3f} vs 模型 {mod_n:+.3f}')
    print()

    # ③ 逐种子稳定性（最大档）
    biggest = {}
    for bk in BUCKETS:
        have = [sz for sz in SIZES if (bk, sz) in data]
        if have:
            biggest[bk] = have[-1]
    print('③ 最大档逐种子 Top-1 边距')
    for bk, sz in biggest.items():
        ms = sorted((r['seed'], round(r['final']['top1'] - r['trigram']['top1'], 2))
                    for r in data[(bk, sz)])
        print(f'  {bk:>7}@{sz}: {ms}')

    if args.no_plot:
        return

    # ④ 图：上排 Top-1 边距、下排 NLL 边距，x = 档位
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _cn_font(matplotlib)

    fig, axes = plt.subplots(2, len(BUCKETS), figsize=(4.6 * len(BUCKETS), 7.4))
    xs = list(range(len(SIZES)))
    for j, bk in enumerate(BUCKETS):
        for i, (mkey, mname, _) in enumerate((METRICS[0], METRICS[2])):
            ax = axes[i][j]
            pts = [(k, margin(data[(bk, sz)], mkey)) for k, sz in enumerate(SIZES)
                   if (bk, sz) in data]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-',
                        color=COLORS[bk], ms=6, lw=1.6)
                for k, v in pts:
                    ax.annotate(f'{v:+.2f}', (k, v), textcoords='offset points',
                                xytext=(5, 5), fontsize=8, color=COLORS[bk])
            ax.axhline(0, color='#888888', ls='--', lw=1.2)
            ax.set_xticks(xs)
            ax.set_xticklabels([f'{int(s)/10000:.0f}万' if s != '1000000' else '100万'
                                for s in SIZES], fontsize=8)
            ax.set_title(f'{bk} · {mname} 边距', fontsize=11)
            ax.grid(alpha=0.25)
            if j == 0:
                ax.set_ylabel(f'{mname} 边距（模型 − trigram）')
    fig.suptitle(f'数据量扩展（tag={args.tag} · ctx64 · 3ep token 对齐）：缺口随数据加大是收敛还是变宽', fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f'✓ 图写入 {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
