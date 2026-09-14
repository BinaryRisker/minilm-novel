#!/usr/bin/env python3
"""
py/compare_cfg.py — 对照两组同协议、仅模型主体不同的结果（小主体 tag='' vs 大主体 tag='big'）。

回答的问题：留出集 Top-1 的差距是「容量不够」还是「结构（字符级 + 32 上下文 + 基线本身很强）」？
做法：**训练数据/步数/种子/评测协议完全不变**，只换 cfg 里的 d / layers / heads。

用法：
  python py/compare_cfg.py                       # 缺省对比 tag='' 与 tag='big'
  python py/compare_cfg.py --a '' --b big --sizes 50000,200000,1000000
  python py/compare_cfg.py --a big --b huge      # 任意两个 tag

输出：
  - 每组 (bucket, size) 的 mean±σ（Top-1 pt / NLL nat）
  - Δ = B − A（Top-1 正数=变大主体更好；NLL 负数=变大主体更好）
  - 参数量分账（词表嵌入 与 主体），说明「5 万参数」只在极小词表下成立
"""
import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def body_params(cfg: dict) -> int:
    """Transformer 主体参数量（不含词表嵌入），与权重绑定无关。"""
    d, L, ff = cfg['d'], cfg['layers'], cfg.get('ffMul', 4)
    per_layer = (3 * d * d + 3 * d) + (d * d + d) \
        + (d * ff * d + ff * d) + (ff * d * d + d) + (4 * d)
    return L * per_layer + cfg['ctx'] * d + 2 * d


def load(tag: str, runs_dir: str):
    out = []
    for fp in sorted(glob.glob(os.path.join(runs_dir, 'wsplit_*.json'))):
        base = os.path.basename(fp)
        if base.startswith('wsplit_summary') or base.startswith('wsplit_tri_'):
            continue
        with open(fp, 'r', encoding='utf-8') as f:
            d = json.load(f)
        if 'bucket' in d and 'curve' in d and d.get('tag', '') == tag:
            out.append(d)
    return out


def agg(rs, key):
    v = [r['final'][key] for r in rs]
    return statistics.mean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default='', help="基准组 tag（缺省 '' = 小主体）")
    ap.add_argument('--b', default='big', help="对照组 tag（缺省 'big'）")
    ap.add_argument('--runs_dir', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--buckets', default='', help='逗号分隔；缺省全部')
    ap.add_argument('--sizes', default='', help='逗号分隔的 size_label；缺省全部')
    args = ap.parse_args()

    A, B = load(args.a, args.runs_dir), load(args.b, args.runs_dir)
    if not A:
        sys.exit(f"没有 tag='{args.a}' 的结果")
    if not B:
        sys.exit(f"没有 tag='{args.b}' 的结果（先跑 --tag {args.b}）")

    def key(r):
        return (r['bucket'], r['size'])

    ga, gb = {}, {}
    for r in A:
        ga.setdefault(key(r), []).append(r)
    for r in B:
        gb.setdefault(key(r), []).append(r)

    bk_f = set(args.buckets.split(',')) if args.buckets else None
    sz_f = set(args.sizes.split(',')) if args.sizes else None

    keys = sorted(set(ga) & set(gb), key=lambda k: (k[0], k[1]))
    keys = [k for k in keys if (bk_f is None or k[0] in bk_f)
            and (sz_f is None or gb[k][0]['size_label'] in sz_f)]
    if not keys:
        sys.exit('两组没有可比的 (bucket, size) 交集')

    cfgA, cfgB = A[0]['cfg'], B[0]['cfg']
    pA, pB = body_params(cfgA), body_params(cfgB)
    print(f'A: tag={args.a!r} cfg={cfgA} 主体 {pA:,} 参数')
    print(f'B: tag={args.b!r} cfg={cfgB} 主体 {pB:,} 参数'
          f'（{pB / pA:.2f}× 主体，训练步数与数据完全相同）')
    print()
    hdr = (f"{'bucket':>7} {'size':>8} {'n':>3} | "
           f"{'Top-1 A':>8} {'Top-1 B':>8} {'ΔTop-1':>8} | "
           f"{'NLL A':>7} {'NLL B':>7} {'ΔNLL':>7} | {'判读':>10}")
    print(hdr)
    print('-' * len(hdr))
    wins_top = wins_nll = 0
    for k in keys:
        ra, rb = ga[k], gb[k]
        ta, sa = agg(ra, 'top1')
        tb, sb = agg(rb, 'top1')
        na, _ = agg(ra, 'nll')
        nb, _ = agg(rb, 'nll')
        dtop, dnll = tb - ta, nb - na
        if dtop > 0:
            wins_top += 1
        if dnll < 0:
            wins_nll += 1
        verdict = []
        if dtop > max(1.0, 2 * max(sa, sb)):
            verdict.append('Top-1↑')
        if dnll < -0.05:
            verdict.append('NLL↓')
        print(f"{k[0]:>7} {rb[0]['size_label']:>8} {min(len(ra), len(rb)):>3} | "
              f"{ta:>8.2f} {tb:>8.2f} {dtop:>+8.2f} | "
              f"{na:>7.3f} {nb:>7.3f} {dnll:>+7.3f} | "
              f"{' '.join(verdict) or '—':>10}")
    print()
    print(f'合计：变大主体后 Top-1 变好 {wins_top}/{len(keys)} 格，NLL 变好 {wins_nll}/{len(keys)} 格')

    # 参数量分账：说明「5.5 万参数」只在极小词表成立
    print('\n参数量分账（词表嵌入 V×d 与主体分开列）：')
    for tag, runs, cfg in ((args.a or '小', A, cfgA), (args.b, B, cfgB)):
        Vs = sorted({r['V'] for r in runs})
        lo, hi = min(Vs), max(Vs)
        bp = body_params(cfg)
        emb_lo, emb_hi = lo * cfg['d'], hi * cfg['d']
        tot_lo, tot_hi = emb_lo + bp, emb_hi + bp
        print(f"  tag={tag!r:<6} V∈[{lo},{hi}]  嵌入 {emb_lo:,}~{emb_hi:,}"
              f"（占总参数 {100 * emb_lo / tot_lo:.1f}%~{100 * emb_hi / tot_hi:.1f}%）"
              f"  主体 {bp:,} 常值")


if __name__ == '__main__':
    main()
