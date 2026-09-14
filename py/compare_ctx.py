#!/usr/bin/env python3
"""
py/compare_ctx.py — 对照多个上下文长度（ctx）的同协议结果。

回答的问题：`Top-1` 的缺口是不是「上下文太短、模型看不到长程依赖」造成的？

为什么不能直接比绝对值：
  ctx 同时改变**模型能用的历史**和**评测窗口的切分**（窗口长 = ctx、步长 = ctx），
  因此 trigram 的分数也会随 ctx 轻微漂移（窗口起点处的历史被截断，ctx 越大这种位置越少）。
  所以要看的是 **边距 = 模型 − trigram**，它把「基线也一起受益」的部分扣掉。

对照口径（compute-matched）：
  `steps = max(min_steps, 3 × 训练字数 / (batch × ctx))`，即 ctx 只改变「每步塞多少 token」，
  不改变总 token 预算。但小档位会被 `min_steps` 兜住 → 必须把 `min_steps` 按 1/ctx 等比缩小
  （ctx=32→200、64→100、128→50），否则「ctx 越大训练步数越多」，对比是混淆的。
  脚本会把每格的 `有效 epoch` 打出来，逐格核对是否真的对齐。

用法：
  python py/compare_ctx.py                                  # 缺省 base(+ctx64+ctx128)
  python py/compare_ctx.py --tags base=,ctx64,ctx128
  python py/compare_ctx.py --tags base=,ctx64 --sizes 200000,1000000 --buckets gudian,wuxia

输出：
  ① trigram 自身随 ctx 的漂移（证明基线没有跟着变）
  ②a/b/c 边距（模型 − trigram）逐 ctx + Δ，分 Top-1 / Top-5 / NLL ——
    Top-1 是「峰值」指标，Top-5 与 NLL 是「排序质量」指标，三者一起看才能区分
    「分布整体更准」与「峰值更尖」
  ③ compute-matched 核对（每格的有效 epoch）+ 模型绝对值
  ④ 汇总：边距从 ctx=32 到最大 ctx 变好的格数
"""
import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

SIZE_ORDER = ['2000', '10000', '50000', '200000', '1000000', '2000000', '5000000', 'all']


def parse_tags(spec: str):
    """'base=,ctx64,ctx128' -> [('base',''), ('ctx64','ctx64'), ('ctx128','ctx128')]
    没有 '=' 的项：名字=标签本身；'base=' 表示标签为空串（小主体/基准）。"""
    out = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' in item:
            name, tag = item.split('=', 1)
            out.append((name.strip(), tag.strip()))
        else:
            out.append((item, item))
    return out


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
    ap.add_argument('--tags', default='base=,ctx64,ctx128',
                    help="逗号分隔；'名字=标签'，'base=' 表示空标签（缺省基准）")
    ap.add_argument('--runs_dir', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--buckets', default='', help='逗号分隔；缺省全部')
    ap.add_argument('--sizes', default='', help='逗号分隔的 size_label；缺省全部')
    args = ap.parse_args()

    tags = parse_tags(args.tags)
    groups = []
    for name, tag in tags:
        rs = load(tag, args.runs_dir)
        if not rs:
            print(f'⚠️  tag={tag!r}（{name}）没有结果，跳过')
            continue
        groups.append((name, tag, rs))
    if len(groups) < 2:
        sys.exit('至少需要两组结果才能对照（先跑 --tag ... 生成另一组）')

    bk_f = set(args.buckets.split(',')) if args.buckets else None
    sz_f = set(args.sizes.split(',')) if args.sizes else None

    def idx(rs):
        g = {}
        for r in rs:
            g.setdefault((r['bucket'], r['size_label']), []).append(r)
        return g

    idxs = [(name, tag, idx(rs), rs) for name, tag, rs in groups]

    # 只保留在所有组里都出现的 (bucket, size) —— 逐格对照要求同格可比
    common = set(idxs[0][2])
    for _, _, ix, _ in idxs[1:]:
        common &= set(ix)
    keys = sorted(common, key=lambda k: (k[0], SIZE_ORDER.index(k[1]) if k[1] in SIZE_ORDER else 99, k[1]))
    keys = [k for k in keys if (bk_f is None or k[0] in bk_f) and (sz_f is None or k[1] in sz_f)]
    if not keys:
        print('没有共同可比格。各组覆盖的 (bucket,size)：')
        for name, _, ix, _ in idxs:
            print(f'  {name}: {sorted(ix)[:12]}{" ..." if len(ix) > 12 else ""}')
        sys.exit(1)

    names = [n for n, _, _, _ in idxs]
    print('ctx 对照（每格 = 多种子均值）')
    for name, tag, ix, rs in idxs:
        cfgs = {json.dumps(r['cfg'], sort_keys=True) for r in rs}
        cfg = json.loads(sorted(cfgs)[0])
        tot = sum(r['wallclock']['train'] for r in rs)
        print(f'  {name:<8} tag={tag!r:<9} cfg={cfg}  {len(rs)} 条 训练合计 {tot / 3600:.2f} h')
    print(f'  共同可比格：{len(keys)} 个（{len(names)} 组）')
    print()

    # ① 基线漂移：trigram 逐 ctx
    print('① trigram 自身随 ctx 的漂移（同格、与种子的模型无关）')
    h = f"{'bucket':>7} {'size':>9} | " + ' '.join(f"{n + ' Top-1':>13}" for n in names) \
        + ' | ' + ' '.join(f"{n + ' NLL':>12}" for n in names)
    print(h)
    print('-' * len(h))
    for bk, sz in keys:
        cells_t, cells_n = [], []
        for _, _, ix, _ in idxs:
            t = ix[(bk, sz)][0]['trigram']
            cells_t.append(f"{t['top1']:>13.2f}")
            cells_n.append(f"{t['nll']:>12.3f}")
        print(f"{bk:>7} {sz:>9} | " + ' '.join(cells_t) + ' | ' + ' '.join(cells_n))
    print()

    # ② 边距（模型 − trigram）。Top-1 是「峰值」指标，Top-5 与 NLL 是「排序质量」指标
    # —— 三者一起看才能区分「分布更准」与「峰值更尖」。
    METRICS = (
        ('top1', 'Top-1', '{v:+10.2f}', '正 = 模型赢'),
        ('top5', 'Top-5', '{v:+10.2f}', '正 = 模型赢'),
        ('nll', 'NLL', '{v:+10.3f}', '负 = 模型赢'),
    )

    def margin(metric, rs):
        tri = rs[0]['trigram']
        return statistics.mean([r['final'][metric] for r in rs]) - tri[metric]

    for mi, (mkey, mname, fmt, better) in enumerate(METRICS):
        print(f'②{"abc"[mi]} 边距（模型 − trigram）· {mname}（{better}）')
        h = f"{'bucket':>7} {'size':>9} | " + ' '.join(f"{n:>10}" for n in names) \
            + ' | ' + ' '.join(f"{'Δ ' + n2:>10}" for n2 in names[1:])
        print(h)
        print('-' * len(h))
        for bk, sz in keys:
            ms = [margin(mkey, ix[(bk, sz)]) for _, _, ix, _ in idxs]
            cells = [fmt.format(v=m) for m in ms]
            deltas = [fmt.format(v=ms[i] - ms[0]) for i in range(1, len(ms))]
            print(f"{bk:>7} {sz:>9} | " + ' '.join(f"{c:>10}" for c in cells)
                  + ' | ' + ' '.join(f"{d:>10}" for d in deltas))
        print()

    # ③ compute-matched 核对 + 模型绝对值
    print('③ compute-matched 核对（每格取第一种子的 cfg/步数；有效 epoch = steps×batch×ctx/训练字数）')
    h = f"{'bucket':>7} {'size':>9} | " + ' '.join(f"{n:>16}" for n in names)
    print(h)
    print('-' * len(h))
    for bk, sz in keys:
        cells = []
        for _, _, ix, _ in idxs:
            r = ix[(bk, sz)][0]
            cells.append(f"ctx{r['cfg']['ctx']}/s{r['steps']}/e{r['eff_epochs']:.0f}")
        print(f"{bk:>7} {sz:>9} | " + ' '.join(f"{c:>16}" for c in cells))
    print()
    print('模型绝对值：')
    h = f"{'bucket':>7} {'size':>9} | " + ' '.join(f"{n + ' Top-1':>14}" for n in names) \
        + ' | ' + ' '.join(f"{n + ' NLL':>12}" for n in names)
    print(h)
    print('-' * len(h))
    for bk, sz in keys:
        ct, cn = [], []
        for _, _, ix, _ in idxs:
            rs = ix[(bk, sz)]
            t, _ = agg(rs, 'top1')
            n_, _ = agg(rs, 'nll')
            ct.append(f"{t:>14.2f}")
            cn.append(f"{n_:>12.3f}")
        print(f"{bk:>7} {sz:>9} | " + ' '.join(ct) + ' | ' + ' '.join(cn))

    # 汇总：边距在 ctx 上是否单调改善
    print()
    for mkey, mname, _, _ in METRICS:
        sign = -1 if mkey == 'nll' else 1
        wins = 0
        for bk, sz in keys:
            ms = [margin(mkey, ix[(bk, sz)]) for _, _, ix, _ in idxs]
            if sign * (ms[-1] - ms[0]) > 0:
                wins += 1
        print(f'{mname} 边距：从 {names[0]} 到 {names[-1]} 变好 {wins}/{len(keys)} 格')


if __name__ == '__main__':
    main()
