""" ============================================================================
py/sweep_report.py — 汇总数据量扫描结果：表格 + 对比折线图
----------------------------------------------------------------------------
读取 py/runs/sweep_*.json（由 py/sweep_data_scale.py 产出），把两种步数模式
（固定步数 / 步数随数据量放大）放在同一张图里对比，并叠加 trigram 基线。

为什么需要单独一个报告脚本：sweep_data_scale.py 每次运行只写自己那一批结果，
会把上一次的报告覆盖掉。这里把所有批次合起来看，才能同时呈现两种步数模式。

用法：
  python py/sweep_report.py
  python py/sweep_report.py --runs py/runs --img ../docs/img/sweep_data_scale.png
============================================================================ """
from __future__ import annotations
import argparse
import glob
import json
import os
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))

ORDER = ['scifi', 'wuxia', 'urban', 'gudian']
LABEL = {'scifi': '科幻 · 清末科学小说', 'wuxia': '武侠 · 侠义公案',
         'urban': '都市 · 世情谴责', 'gudian': '古典 · 神魔笔记'}


def load_runs(runs_dir: str) -> List[Dict]:
    out = []
    for fp in sorted(glob.glob(os.path.join(runs_dir, 'sweep_*.json'))):
        if os.path.basename(fp) == 'sweep_summary.json':
            continue
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                d = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        if 'runs' in d:          # summary 文件
            out += d['runs']
        elif 'corpus' in d:
            out.append(d)
    return out


def mode_of(r: Dict) -> str:
    return 'prop' if r.get('proportional_steps') else 'fix'


def dedupe(runs: List[Dict]) -> List[Dict]:
    """同一 (corpus, mode, size) 只留一条；'all' 与截断后同尺寸时保留 'all'。"""
    best = {}
    for r in runs:
        k = (r['corpus'], mode_of(r), r['size'])
        prev = best.get(k)
        if prev is None or (r['size_label'] == 'all' and prev['size_label'] != 'all'):
            best[k] = r
    return list(best.values())


def write_report(runs: List[Dict], out_md: str):
    by = {}
    for r in runs:
        by.setdefault(r['corpus'], {}).setdefault(mode_of(r), {})[r['size']] = r

    L = ['# 数据量扫描：模型 vs Trigram（公版语料 613 万字）', '',
         '> ⚠️ **本报告是「桶内连续 90/10」口径的历史存档，已被取代。**',
         '> 它有两个已知问题：① 留出集与训练集来自同一批作品（**同作品泄漏**）；',
         '> ② 只记 Top-1、没记留出集 NLL，且只跑了单种子。',
         '> 现行结论见 `docs/experiments/2026-09-14-worksplit-nll-converge.md`',
         '> （按作品切分 + NLL 同口径 + 3 种子 + 收敛曲线），脚本为 `py/sweep_worksplit.py`。',
         '> 另注意本文档里 **scifi 一列对应的是已被重建的旧语料**（13.2 万字，现为 179 万字）。', '',
         '> 回答「模型到底需要多少训练字才能超过 trigram？」',
         '> 模型配置固定（ctx=32 d=40 layers=2 heads=4 ffMul=4）、Adam lr=0.004、'
         'batch=8、seed=1234、词频下限 min_freq=3；测试集始终是全文最后 10%。', '',
         '两种步数模式：',
         '- `fix`：固定 400 步。模型 token 预算恒定，数据变大并不会多看 token。',
         '- `prop`：步数随训练字数同比放大（参考点 20 万字=400 步，上限 4000 步），'
         '等价固定 epoch，才能真正看出「更多数据」的收益。', '']

    for ck in ORDER:
        if ck not in by:
            continue
        modes = by[ck]
        sizes = sorted(set().union(*[set(d) for d in modes.values()]))
        L += [f'## {ck} — {LABEL.get(ck, ck)}', '',
              '| 训练字数 | 步数(fix) | 步数(prop) | Model Top-1 fix | Model Top-1 prop | '
              'Model Top-5 prop | Trigram Top-1 | 胜负(prop) |',
              '|---|---|---|---|---|---|---|---|']
        for sz in sizes:
            rf = modes.get('fix', {}).get(sz)
            rp = modes.get('prop', {}).get(sz)
            ref = rp or rf
            tri = ref['trigram']['top1']
            mf = f"{rf['model']['top1']:.1f}%" if rf else '—'
            mp = f"{rp['model']['top1']:.1f}%" if rp else '—'
            m5 = f"{rp['model']['top5']:.1f}%" if rp else '—'
            sf = str(rf.get('steps', 400)) if rf else '—'
            sp = str(rp.get('steps', 400)) if rp else '—'
            win = '—'
            if rp:
                win = '模型胜' if rp['model']['top1'] > tri else '模型负'
            L.append(f'| {sz} | {sf} | {sp} | {mf} | {mp} | {m5} | {tri:.1f}% | {win} |')
        L.append('')

    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'报告 -> {out_md}')


def plot(runs: List[Dict], out_img: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for cand in ('Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC'):
        try:
            matplotlib.rcParams['font.sans-serif'] = [cand]
            matplotlib.rcParams['axes.unicode_minus'] = False
            break
        except Exception:  # noqa: BLE001
            continue

    by = {}
    for r in runs:
        by.setdefault(r['corpus'], {}).setdefault(mode_of(r), {})[r['size']] = r
    keys = [k for k in ORDER if k in by] or list(by)

    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 4.2), sharey=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, ck in zip(axes, keys):
        modes = by[ck]
        sizes = sorted(set().union(*[set(d) for d in modes.values()]))
        xs = [s for s in sizes if s > 0]
        tri = [modes[[m for m in modes][0]][s]['trigram']['top1'] for s in sizes]
        ax.plot(sizes, tri, 's--', color='#7f7f7f', alpha=0.85, markersize=7,
                label='Trigram (α=0.1)')
        for mode, color, mk, style, name in (
                ('fix', '#1f77b4', 'o', '-', 'Model · 固定 400 步'),
                ('prop', '#d62728', '^', '-', 'Model · 步数随数据放大')):
            d = modes.get(mode)
            if not d:
                continue
            ys = [d[s]['model']['top1'] for s in sizes]
            ax.plot(sizes, ys, marker=mk, color=color, linestyle=style,
                    linewidth=2, markersize=7, label=name)
        ax.set_title(f'{ck}', fontsize=12)
        ax.set_xlabel('训练字数（对数轴）')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    axes[0].set_ylabel('留出集 Top-1 (%)')
    fig.suptitle('数据量扫描 · 公版中文小说语料（613 万字，90 部作品）', fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    fig.savefig(out_img, dpi=120, bbox_inches='tight')
    print(f'图 -> {out_img}')


def write_summary(runs: List[Dict], out_json: str):
    """把**全部批次合并后**的结果写成一份汇总 json。
    为什么需要：sweep_data_scale.py 每跑一批就覆盖一次 runs/sweep_summary.json，
    于是它永远只含"最后那一批"（曾经只剩 prop 模式），照着它读会得出片面结论。"""
    runs = sorted(runs, key=lambda r: (ORDER.index(r['corpus'])
                                       if r['corpus'] in ORDER else 99,
                                       mode_of(r), r['size']))
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'meta': {
                'note': '由 py/sweep_report.py 合并全部 sweep_*.json 生成（含 fix 与 prop 两种模式）',
                'n_runs': len(runs),
                'corpora': sorted({r['corpus'] for r in runs}),
                'modes': sorted({mode_of(r) for r in runs}),
                'seed': sorted({r['seed'] for r in runs}),
                'min_freq': sorted({r.get('min_freq') for r in runs}, key=lambda x: (x is None, x)),
            },
            'runs': runs,
        }, f, ensure_ascii=False, indent=2)
    print(f'汇总 -> {out_json}（{len(runs)} 条，含 fix+prop）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--md', default=os.path.join(HERE, 'runs', 'sweep_report.md'))
    ap.add_argument('--img', default=os.path.join(HERE, '..', 'docs', 'img',
                                                  'sweep_data_scale.png'))
    ap.add_argument('--summary', default=os.path.join(HERE, 'runs', 'sweep_summary.json'))
    ap.add_argument('--no_plot', action='store_true')
    args = ap.parse_args()

    runs = dedupe(load_runs(args.runs))
    if not runs:
        raise SystemExit('没有找到 sweep_*.json')
    print(f'载入 {len(runs)} 条结果')
    write_report(runs, args.md)
    write_summary(runs, args.summary)
    if not args.no_plot:
        plot(runs, args.img)


if __name__ == '__main__':
    main()
