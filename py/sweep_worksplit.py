""" ============================================================================
py/sweep_worksplit.py — 在「按作品切分」的语料上做数据量扫描 + 收敛曲线 + 多种子误差棒
----------------------------------------------------------------------------
为什么另起一个脚本（而不是改 sweep_data_scale.py）：
  sweep_data_scale.py 的 train/test 是「整桶拼接文本上按 len*0.9 连续切一刀」。
  整桶文本是同一批作品的拼接，于是同一部作品的头进 train、尾巴进 test —— **同作品泄漏**，
  留出指标被高估（见 py/data/build_splits.py 头部说明）。
  本脚本改吃 py/data/build_splits.py 产出的**按作品整篇切分**：
      data/splits/{train,val,test}.txt          （全部作品，用于 --buckets ALL）
      data/manifest.jsonl + data/processed/     （按桶重组，用于 --buckets wuxia 等）
  留出集里的作品在训练集里**一部都不出现**，指标才代表「对没见过的作品」的泛化。

回答三个问题：
  ① 模型留出集 NLL/PPL 是多少？—— 与 trigram 的 NLL 同口径（都对完整 V 维归一化、
     同一批留出位置平均），这样「峰值精度(Top-1)」和「整分布质量(NLL)」两个维度都能正面比。
  ② 训到收敛还差多少？—— 固定 epoch 数（步数 ∝ 训练字数），按 epoch 记录评测曲线，
     看曲线是否走平；cosine 调度在总步数处衰减到 0，所以末点即「本次预算下的收敛点」。
  ③ 差距是不是种子噪声？—— 多种子重跑同一配置，报告 mean±std 与 min~max 区间。

评测位协议（与 py/baselines.py 严格一致，不可随意改）：
  32 字滑窗，**每 ctx 步一个窗口**，窗口内逐位置计 top-1/top-5/NLL。
  模型与 trigram 必须用同一批位置，否则不可比。

选型/验收分离（2026-09-15 起，docs/EVAL_PROTOCOL_V1.md）：
  收敛**曲线在 val 上取**（调参/看走平，可反复读）；
  **test 只在训练结束评一次**（结果 JSON 的 `final`），曲线点全部来自 val（`curve_split: "val"`）。
  旧结果文件没有 `curve_split` 字段 = 曲线来自 test = **exploratory**，不得作验收依据。
  trigram 基线仍在 test 上一次性评（与 `final` 同口径可比）。

留出切片（--test_chars，默认 50000）：
  从该桶的留出作品里**轮流取**前若干字，凑够预算。这样留出集覆盖所有留出作品，
  而不是被某一部长篇独占。取的是原始前缀，稳定可复现。val 切片同规则（--val_chars）。
  50,000 个评测位在 Top-1≈15% 处标准误约 0.16pt，够分辨 1pt 级差异；
  要更细可调大，但评测成本与之成正比（每条配置 eval_points 次）。

词表：
  在该桶的**全部训练作品**上建（min_freq 默认 3，低于该频次的字映射到 UNK=0）。
  注意：词表**不随 size 变化**——扫描的目的是隔离「数据量」这一个变量，
  若 size=2k 时另建一个小词表，数据量与词表两个变量会纠缠在一起。
  测试切片上用该词表编码，UNK 占比会记进结果（oov）。

输出（带 --tag X 时下列文件名统一插入 _X，用来并列小主体与大主体两套结果）：
  py/runs/wsplit_<bucket>_<label>_s<seed>.json     单条结果（含 curve）
  py/runs/wsplit_tri_<bucket>_<label>_t<n>.json    trigram 缓存（与 seed/cfg 无关，可跨 --tag 复用）
  py/runs/wsplit_summary.json                      合并汇总（--report_only 写）
  py/runs/wsplit_report.md                         表格（逐桶：模型 mean±std vs trigram）
  docs/img/worksplit_scaling.png                   数据量曲线（Top-1 与 NLL 两行）
  docs/img/worksplit_converge.png                  收敛曲线（NLL vs epoch，逐 seed）

用法：
  # 数据量扫描（默认 3 个种子、3 个 epoch、5 个桶）
  python py/sweep_worksplit.py
  # 只要一个桶、最快跑通
  python py/sweep_worksplit.py --buckets scifi --sizes 2000,10000,50000 --seeds 1234 --epochs 2
  # 大模型对照（看「是容量不够还是训练不够」）
  python py/sweep_worksplit.py --cfg_d 64 --cfg_layers 4 --cfg_heads 8 --tag big
============================================================================ """
from __future__ import annotations
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time
from typing import Dict, List

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from minilm_torch import MiniLM, build_vocab, encode, make_batch, lr_at  # noqa
from baselines import NGram, Acc, for_each_position  # noqa

DEFAULT_CFG = dict(ctx=32, d=40, layers=2, heads=4, ffMul=4)
DEFAULT_SIZES = (2000, 10000, 50000, 200000, 1000000)
BUCKETS = ('scifi', 'wuxia', 'urban', 'gudian', 'story', 'xiandai')
TRI_ALPHAS = (0.01, 0.05, 0.1, 0.5, 1.0)


# ---------------------------- 语料装配 ----------------------------

def slug(s: str) -> str:
    """与 fetch_corpus.py / build_splits.py 保持一致（作品正文落盘用的文件名）"""
    s = re.sub(r'[\\/:*?"<>|\s]+', '_', s.strip())
    return s[:80] or 'untitled'


def read_manifest(path: str) -> List[dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def work_texts(manifest: List[dict], processed: str, bucket: str, split: str) -> List[str]:
    """取某桶某 split 的全部作品正文（整篇），按 story_id 排序保证可复现。"""
    rows = [r for r in manifest if r['bucket'] == bucket and r['split'] == split]
    rows.sort(key=lambda r: r['story_id'])
    out = []
    for r in rows:
        p = os.path.join(processed, bucket, slug(r['title']) + '.txt')
        if not os.path.exists(p):
            raise FileNotFoundError(f'作品正文缺失：{p}（{r["story_id"]} {r["title"]}）')
        with open(p, 'r', encoding='utf-8') as f:
            out.append(f.read())
    return out


def read_split_file(split_dir: str, name: str) -> str:
    p = os.path.join(split_dir, name + '.txt')
    if not os.path.exists(p):
        raise FileNotFoundError(f'{p} 不存在（先跑 python py/data/build_splits.py）')
    with open(p, 'r', encoding='utf-8') as f:
        return f.read()


def build_slice(texts: List[str], budget: int) -> str:
    """从多部作品里**轮流取前缀**凑够 budget 字，保证留出集覆盖所有留出作品。
    单一长篇独占整个留出集会让指标变成「对这一部作品的表现」，方差很大。"""
    if not texts:
        return ''
    total = sum(len(t) for t in texts)
    if total <= budget:
        return '\n\n'.join(texts)
    per = max(1, budget // len(texts))
    out, used = [], 0
    # 先每部取 per 字
    for t in texts:
        take = min(per, len(t))
        out.append(t[:take])
        used += take
    # 还有预算就从头逐部补齐
    i = 0
    while used < budget and i < len(texts):
        rem = budget - used
        extra = min(rem, len(texts[i]) - len(out[i]))
        if extra > 0:
            out[i] = texts[i][:len(out[i]) + extra]
            used += extra
        i += 1
    return '\n\n'.join(out)


def load_bucket(bucket: str, manifest: List[dict], processed: str,
                split_dir: str) -> Dict:
    """返回 {train_text, val_texts, test_texts, n_train_works, n_val_works, n_test_works}。
    **不做截断** —— 词表要在该桶全部训练作品上建；截断只在编码阶段做（见 main）。
    bucket == 'ALL' 时直接读 data/splits/*.txt（与审计产物逐字一致）。"""
    if bucket == 'ALL':
        tr = read_split_file(split_dir, 'train')
        va = read_split_file(split_dir, 'val')
        te = read_split_file(split_dir, 'test')
        n = {sp: sum(1 for r in manifest if r['split'] == sp) for sp in ('train', 'val', 'test')}
        return {'train_text': tr, 'val_texts': [va], 'test_texts': [te],
                'n_train_works': n['train'], 'n_val_works': n['val'], 'n_test_works': n['test']}
    tr_texts = work_texts(manifest, processed, bucket, 'train')
    va_texts = work_texts(manifest, processed, bucket, 'val')
    te_texts = work_texts(manifest, processed, bucket, 'test')
    if not tr_texts:
        raise SystemExit(f'桶 {bucket} 没有训练作品')
    if not va_texts:
        raise SystemExit(f'桶 {bucket} 没有验证作品（val 曲线没有载体，无法做合规评测）')
    if not te_texts:
        raise SystemExit(f'桶 {bucket} 没有留出作品（该桶留出作品为空，无法做无泄漏评测）')
    return {'train_text': '\n\n'.join(tr_texts), 'val_texts': va_texts, 'test_texts': te_texts,
            'n_train_works': len(tr_texts), 'n_val_works': len(va_texts),
            'n_test_works': len(te_texts)}


# ---------------------------- 训练与评测 ----------------------------

@torch.no_grad()
def eval_model(model, ids: List[int], ctx: int) -> Dict:
    """32 字滑窗、每 ctx 步一窗、窗口内逐位置。返回 top-1/top-5/NLL/PPL。
    NLL = -log P(target)，对完整 V 维归一化，与 baselines.Acc 同口径。"""
    model.eval()
    top1 = top5 = total = 0
    nll_sum = 0.0
    s = 0
    n_ids = len(ids)
    while s + ctx + 1 <= n_ids:
        chunk = torch.tensor([ids[s:s + ctx]], dtype=torch.long)
        tgt = ids[s + 1:s + 1 + ctx]
        logits, _ = model(chunk)
        lg = logits[0]                              # [ctx, V]
        topv, topi = lg.topk(5)
        logp = torch.log_softmax(lg.float(), dim=-1)
        tgt_t = torch.tensor(tgt, dtype=torch.long)
        nll_sum += float(-logp[torch.arange(ctx), tgt_t].sum())
        pred = topi.tolist()
        for i in range(ctx):
            if pred[i][0] == tgt[i]:
                top1 += 1
            if tgt[i] in pred[i]:
                top5 += 1
            total += 1
        s += ctx
    n = max(1, total)
    nll = nll_sum / n
    return {'top1': 100.0 * top1 / n, 'top5': 100.0 * top5 / n,
            'nll': nll, 'ppl': math.exp(nll) if nll < 700 else float('inf'),
            'n': total}


def train_curve(train_ids: List[int], val_ids: List[int], test_ids: List[int], V: int,
                cfg: dict, steps: int, batch: int, lr: float, seed: int,
                eval_marks: set) -> Dict:
    """训练 steps 步。**曲线全部评在 val 上**（选型口径，可反复读）；
    test 只在训练结束评一次（验收口径）。返回 curve(val) / val_final / test_final。"""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = MiniLM(V, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = max(10, math.floor(steps * 0.15))

    curve = []
    seg_loss, seg_n = 0.0, 0
    t0 = time.time()
    model.train()
    for step in range(steps):
        tokens, targets = make_batch(train_ids, batch, cfg['ctx'], rng)
        opt.zero_grad()
        _, loss = model(tokens, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps, lr, warmup)
        opt.step()
        seg_loss += loss.item()
        seg_n += 1
        if (step + 1) in eval_marks:
            ev = eval_model(model, val_ids, cfg['ctx'])
            curve.append({'step': step + 1, 'train_loss': seg_loss / seg_n, **ev})
            seg_loss, seg_n = 0.0, 0
            model.train()
    # test 一次性验收：训练全部结束后、权重不再变化时评一次
    test_final = eval_model(model, test_ids, cfg['ctx'])
    return {'curve': curve, 'val_final': {k: curve[-1][k] for k in ('top1', 'top5', 'nll', 'ppl')} if curve else None,
            'test_final': {k: test_final[k] for k in ('top1', 'top5', 'nll', 'ppl')},
            'wallclock': time.time() - t0, 'model': model}


def trigram_eval_uncached(train_ids: List[int], test_ids: List[int], V: int,
                          ctx: int, alpha: float = 0.1) -> Dict:
    m = NGram(train_ids, order=3, alpha=alpha, V=V)
    acc = Acc()
    backoff = {1: 0, 2: 0, 3: 0}
    for hist, tgt in for_each_position(test_ids, ctx):
        acc.add(m.dist(hist), tgt)
        backoff[m.used_order(hist)] += 1
    s = acc.summary()
    return {**s, 'backoff': backoff, 'alpha': alpha}


def trigram_cached(cache_path: str, train_ids: List[int], test_ids: List[int],
                   V: int, ctx: int, alpha_probe: bool) -> Dict:
    """trigram 与 seed 无关 → 缓存复用（多进程同时算同一配置时用原子写避免半截文件）。"""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass  # 缓存损坏就当没有，重算
    base = trigram_eval_uncached(train_ids, test_ids, V, ctx, alpha=0.1)
    out = {'alpha_fixed': base}
    if alpha_probe:
        probe = []
        for a in TRI_ALPHAS:
            r = trigram_eval_uncached(train_ids, test_ids, V, ctx, alpha=a)
            probe.append({'alpha': a, 'nll': r['nll'], 'top1': r['top1']})
        best = min(probe, key=lambda r: r['nll'])
        out['alpha_probe'] = probe
        out['alpha_best_on_test'] = best      # 注意：这是在留出集上挑的，属于上界参考
    tmp = cache_path + '.tmp'
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or '.', exist_ok=True)
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cache_path)
    return out


# ---------------------------- 主流程 ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--buckets', default='all',
                    help="all（5 个桶各自一遍）| POOLED（全部作品合并成一份）| 逗号分隔的桶名")
    ap.add_argument('--report_only', action='store_true',
                    help='不训练，只把 out_dir 里已有的 wsplit_*.json 合并成报告与图')
    ap.add_argument('--no_report', action='store_true',
                    help='不写报告与图（多进程并行跑各种子时用，最后再单独合并一次）')
    ap.add_argument('--sizes', default=','.join(str(s) for s in DEFAULT_SIZES),
                    help='训练字数，逗号分隔；all 表示该桶全部训练作品')
    ap.add_argument('--seeds', default='1234,42,2024', help='逗号分隔的随机种子')
    ap.add_argument('--epochs', type=float, default=3.0,
                    help='训练轮数（步数 = epochs × 训练字数 / (batch×ctx)）')
    ap.add_argument('--min_steps', type=int, default=200,
                    help='小数据量的步数下限（否则 2000 字只有 6 步）')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.004)
    ap.add_argument('--ctx', type=int, default=DEFAULT_CFG['ctx'])
    ap.add_argument('--min_freq', type=int, default=3)
    ap.add_argument('--test_chars', type=int, default=50000,
                    help='留出切片字数上限（从留出作品轮流取前缀）。留出越短越省，但指标越抖')
    ap.add_argument('--val_chars', type=int, default=50000,
                    help='val 切片字数上限（曲线评在 val 上，可反复读；test 只在结束评一次）')
    ap.add_argument('--eval_points', type=int, default=4,
                    help='每条配置的评测点数（等距分布，末点=训练结束）。评测成本∝点数×留出切片')
    ap.add_argument('--cfg_d', type=int, default=DEFAULT_CFG['d'])
    ap.add_argument('--cfg_layers', type=int, default=DEFAULT_CFG['layers'])
    ap.add_argument('--cfg_heads', type=int, default=DEFAULT_CFG['heads'])
    ap.add_argument('--cfg_ffmul', type=int, default=DEFAULT_CFG['ffMul'])
    ap.add_argument('--tag', default='', help='结果文件名后缀（区分不同模型配置）')
    ap.add_argument('--manifest', default=os.path.join(ROOT, 'data', 'manifest.jsonl'))
    ap.add_argument('--processed', default=os.path.join(ROOT, 'data', 'processed'))
    ap.add_argument('--split_dir', default=os.path.join(ROOT, 'data', 'splits'))
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--no_alpha_probe', action='store_true')
    ap.add_argument('--leak_probe', action='store_true',
                    help='受控 A/B：同一训练、同一步数，额外在「训练区尾部连续切片」（旧泄漏口径）上'
                         '评测，用来单独量化泄漏对模型/基线各自的影响')
    ap.add_argument('--no_plot', action='store_true', help='跳过画图')
    ap.add_argument('--force', action='store_true', help='重算已存在的单条结果')
    args = ap.parse_args()

    cfg = dict(ctx=args.ctx, d=args.cfg_d, layers=args.cfg_layers,
               heads=args.cfg_heads, ffMul=args.cfg_ffmul)
    sizes = [-1 if s.strip() == 'all' else int(s) for s in args.sizes.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    if args.buckets.strip().lower() == 'all':
        buckets = list(BUCKETS)
    elif args.buckets.strip() == 'POOLED':
        buckets = ['ALL']
    else:
        buckets = [b.strip() for b in args.buckets.split(',')]

    os.makedirs(args.out_dir, exist_ok=True)
    tag = ('_' + args.tag) if args.tag else ''

    # --report_only：把 out_dir 里已有的单条结果合并成报告+图（供多进程并行跑完各种子后收口）
    if args.report_only:
        runs = []
        for fp in sorted(glob.glob(os.path.join(args.out_dir, 'wsplit_*.json'))):
            base = os.path.basename(fp)
            if base.startswith('wsplit_summary') or base.startswith('wsplit_tri_'):
                continue          # 汇总与 trigram 缓存不是单条结果
            with open(fp, 'r', encoding='utf-8') as f:
                d = json.load(f)
            # 只收本轮 tag 的单条结果：否则同目录里小主体（tag=''）和大主体（tag='big'）
            # 会被混进同一张表/同一张图，cfg 不同却看不出（踩过）。
            if 'bucket' in d and 'curve' in d and d.get('tag', '') == args.tag:
                runs.append(d)
        if not runs:
            sys.exit(f"没有找到 tag='{args.tag}' 的 wsplit_*_s*.json")
        print(f'[*] 合并 {len(runs)} 条结果')
        # 合并版汇总（权威），另写一份 wsplit_summary.json 供下游消费
        seeds = sorted({r['seed'] for r in runs})
        merged = {
            'meta': {'batch': runs[0]['batch'], 'lr': runs[0]['lr'],
                     'epochs': max(r['epochs'] for r in runs),
                     'seeds': seeds,
                     'buckets': sorted({r['bucket'] for r in runs}),
                     'min_freq': runs[0]['min_freq'],
                     'test_chars': max(r['test_chars'] for r in runs),
                     'cfg': runs[0]['cfg'], 'tag': args.tag,
                     'note': '按作品切分（无同作品跨 train/test 泄漏）；词表在该桶全部训练作品上建、'
                             '不随 size 变化；NLL 对完整 V 维归一化，与 trigram 同口径'},
            'runs': runs}
        # 合并汇总也带 tag：否则 `--report_only --tag big` 会用大主体的结果
        # 覆盖掉 py/runs/wsplit_summary.json（那份 75 条小主体的权威汇总，踩过）。
        # tag='' 时文件名仍是 wsplit_summary.json，下游消费不受影响。
        merged_fn = f'wsplit_summary{tag}.json'
        merged_path = os.path.join(args.out_dir, merged_fn)
        with open(merged_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f'✓ 合并汇总写入 {merged_path}')
        # 表头/图标题的 cfg、epochs 一律取结果里的真实值，不取 CLI 默认值
        # （--report_only 时 CLI 往往没带 --cfg_*，默认值会把大主体的表头写错）
        cfg = runs[0]['cfg']
        args.epochs = max(r['epochs'] for r in runs)
        write_report(runs, cfg, args, tag)
        if not args.no_plot:
            try:
                plot_scaling(runs, cfg, args, tag)
                plot_converge(runs, cfg, args, tag)
            except Exception as e:
                print(f'画图失败（不影响主结果）：{e}')
        return

    manifest = read_manifest(args.manifest)
    max_size = max([s for s in sizes if s > 0], default=0)

    print(f'[*] 配置 cfg={cfg} batch={args.batch} lr={args.lr} epochs={args.epochs} '
          f'seeds={seeds} buckets={buckets} sizes={sizes}')
    print(f'[*] 留出切片={args.test_chars} 字 · 词表在该桶全部训练作品上建(min_freq={args.min_freq})')

    all_runs = []
    for bk in buckets:
        print(f'\n=== {bk} ===', flush=True)
        try:
            data = load_bucket(bk, manifest, args.processed, args.split_dir)
        except (FileNotFoundError, SystemExit) as e:
            print(f'  跳过：{e}', flush=True)
            continue

        itos, stoi = build_vocab(data['train_text'], min_freq=args.min_freq)
        V = len(itos)
        test_text = build_slice(data['test_texts'], args.test_chars)
        oov = sum(1 for ch in test_text if ch not in stoi)
        test_ids = encode(test_text, stoi)
        # val 曲线切片：同一轮流取前缀规则，独立预算
        val_text = build_slice(data['val_texts'], args.val_chars)
        val_ids = encode(val_text, stoi)
        # 只编码需要的那段前缀（大池子全编码纯浪费内存与时间）
        cap = len(data['train_text']) if -1 in sizes else min(max_size, len(data['train_text']))
        ids_all = encode(data['train_text'][:cap], stoi)
        ctx = cfg['ctx']
        print(f'  训练作品 {data["n_train_works"]} 部 / 池 {len(data["train_text"]):,} 字'
              f'（本次编码至 {cap:,}）→ V={V} · val 作品 {data["n_val_works"]} 部 / 切片 '
              f'{len(val_text):,} 字 · 留出作品 {data["n_test_works"]} 部 / '
              f'切片 {len(test_text):,} 字（UNK {oov}，{100.0 * oov / max(1, len(test_text)):.3f}%）',
              flush=True)

        for sz in sizes:
            label = 'all' if sz < 0 else str(sz)
            train_ids = ids_all if sz < 0 else ids_all[:sz]
            # 「all」只在该 size 真的覆盖了整个训练池时才用 —— 不能拿 len(ids_all) 比，
            # 因为 ids_all 本身被截到 max_size，会把这个上限误标成 all（踩过）。
            if sz > 0 and len(train_ids) >= len(data['train_text']):
                label = 'all'
            if len(train_ids) < ctx + 2:
                print(f'  [{label}] 训练字太少（{len(train_ids)}），跳过', flush=True)
                continue
            steps = max(args.min_steps, int(round(args.epochs * len(train_ids) / (args.batch * ctx))))
            # 评测点：等距取 eval_points 个（末点=steps）。不用「每 N 步一次」——
            # 小数据量时 min_steps 远大于 steps_per_epoch，会退化成几十个评测点，
            # 评测成本（固定 ∝ 留出切片长度）直接压过训练成本。
            npts = max(2, int(args.eval_points))
            marks = set()
            for k in range(1, npts + 1):
                marks.add(max(1, int(round(steps * k / npts))))
            marks.add(steps)
            tri_cache = os.path.join(
                args.out_dir,
                f'wsplit_tri_{bk}_{label}{tag}_t{len(test_ids)}.json')
            # α 网格只在最大数据量上探（6× 评测成本），用来给基线一个「上界参考」
            want_probe = (not args.no_alpha_probe) and (sz < 0 or sz >= max_size)

            # --leak_probe：受控 A/B。旧口径的留出集 = 训练区最后 10% 起的一段**连续**文本，
            # 与训练集取自同一批作品（同作品泄漏）。这里在**同一份训练、同一步数**下多评一次，
            # 把「泄漏」从「数据量 / 训练预算」里单独隔离出来 —— 否则两种口径的对比是混淆的。
            leak_ids, tri_leak = None, None
            if args.leak_probe and sz > 0:
                lc = int(len(train_ids) * 0.9)
                seg = ids_all[lc:lc + len(test_ids)]
                if len(seg) >= ctx + 1:
                    leak_ids = seg
            if leak_ids is not None:
                tri_leak = trigram_eval_uncached(train_ids, leak_ids, V, ctx, alpha=0.1)

            t0 = time.time()
            tri = trigram_cached(tri_cache, train_ids, test_ids, V, ctx,
                                 alpha_probe=want_probe)
            t_tri = time.time() - t0
            tri_f = tri['alpha_fixed']
            eff_epochs = steps * args.batch * ctx / max(1, len(train_ids))
            print(f'  [{label}] 训练 {len(train_ids):,} 字 · 步数 {steps}'
                  f'（≈{eff_epochs:.2g} epoch）· 评测点 {len(marks)} · '
                  f'trigram NLL={tri_f["nll"]:.3f} Top-1={tri_f["top1"]:.1f}%'
                  f'（{t_tri:.1f}s）', flush=True)

            for seed in seeds:
                fn = f'wsplit_{bk}_{label}{tag}_s{seed}.json'
                path = os.path.join(args.out_dir, fn)
                if os.path.exists(path) and not args.force:
                    with open(path, 'r', encoding='utf-8') as f:
                        r = json.load(f)
                    all_runs.append(r)
                    last = r['curve'][-1]
                    if not r.get('curve_split'):
                        print(f'    seed={seed} 已存在，复用（★旧协议：曲线来自 test，'
                              f'exploratory，勿作验收依据）：'
                              f'NLL={last["nll"]:.3f} Top-1={last["top1"]:.1f}%', flush=True)
                    else:
                        print(f'    seed={seed} 已存在，复用（曲线=val）：'
                              f'val NLL={last["nll"]:.3f} Top-1={last["top1"]:.1f}% · '
                              f'test NLL={r["final"]["nll"]:.3f} Top-1={r["final"]["top1"]:.1f}%',
                              flush=True)
                    continue
                tr = train_curve(train_ids, val_ids, test_ids, V, cfg, steps, args.batch,
                                 args.lr, seed, marks)
                last = tr['curve'][-1]
                leak_final = (eval_model(tr['model'], leak_ids, ctx)
                              if leak_ids is not None else None)
                r = {
                    'bucket': bk, 'size': len(train_ids), 'size_label': label,
                    'seed': seed, 'steps': steps, 'epochs': args.epochs,
                    'eff_epochs': eff_epochs, 'eval_points': len(marks),
                    'batch': args.batch, 'lr': args.lr, 'cfg': cfg, 'V': V,
                    'min_freq': args.min_freq, 'tag': args.tag,
                    'protocol': 'val_select_test_once_v1',
                    'curve_split': 'val',
                    'n_train_works': data['n_train_works'],
                    'n_val_works': data['n_val_works'],
                    'n_test_works': data['n_test_works'],
                    'pool_chars': len(data['train_text']),
                    'train_chars': len(train_ids), 'val_chars': len(val_text),
                    'test_chars': len(test_text),
                    'oov_chars': oov, 'n_positions': last['n'],
                    'curve': tr['curve'],
                    # 「final」= test 一次性验收（训练结束后权重不变时评的那一次）；
                    # 曲线末点是 val，别把它当 test 读
                    'final': tr['test_final'],
                    'val_final': tr['val_final'],
                    'test_eval_points': 1,
                    'trigram': tri_f,
                    'trigram_alpha_probe': tri.get('alpha_probe'),
                    'trigram_alpha_best_on_test': tri.get('alpha_best_on_test'),
                    'leaky_test_chars': len(leak_ids) if leak_ids is not None else 0,
                    'leaky_final': leak_final,
                    'trigram_leaky': tri_leak,
                    'wallclock': {'train': tr['wallclock'], 'trigram': t_tri},
                }
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(r, f, ensure_ascii=False, indent=2)
                all_runs.append(r)
                msg = (f'    seed={seed} → val NLL={last["nll"]:.3f} '
                       f'Top-1={last["top1"]:.1f}% · '
                       f'test(一次性) NLL={r["final"]["nll"]:.3f} '
                       f'Top-1={r["final"]["top1"]:.1f}% Top-5={r["final"]["top5"]:.1f}% '
                       f'({tr["wallclock"]:.0f}s)')
                if leak_final is not None:
                    msg += (f'\n        泄漏留出：模型 NLL={leak_final["nll"]:.3f} '
                            f'Top-1={leak_final["top1"]:.1f}% vs '
                            f'trigram NLL={tri_leak["nll"]:.3f} Top-1={tri_leak["top1"]:.1f}%')
                print(msg, flush=True)

    if not all_runs:
        sys.exit('没有产生任何结果')

    write_summary(all_runs, args, tag)
    if not args.no_report:
        write_report(all_runs, cfg, args, tag)
        if not args.no_plot:
            try:
                plot_scaling(all_runs, cfg, args, tag)
                plot_converge(all_runs, cfg, args, tag)
            except Exception as e:
                print(f'画图失败（不影响主结果）：{e}')
    print(f'\n✓ 完成 {len(all_runs)} 条')


def write_summary(all_runs: List[Dict], args, tag: str):
    """单批汇总放进带桶名的文件，避免多个桶进程互相覆盖；
    全部批次的合并版由 --report_only 写进 wsplit_summary.json。"""
    seeds = '-'.join(str(s) for s in sorted({r['seed'] for r in all_runs}))
    bks = '-'.join(sorted({r['bucket'] for r in all_runs}))
    fn = f'wsplit_summary_{bks}{tag}_s{seeds}.json'
    meta = {'batch': args.batch, 'lr': args.lr, 'epochs': args.epochs,
            'seeds': sorted({r['seed'] for r in all_runs}),
            'buckets': sorted({r['bucket'] for r in all_runs}),
            'min_freq': args.min_freq, 'test_chars': args.test_chars,
            'eval_points': args.eval_points, 'tag': args.tag,
            'note': '按作品切分（无同作品跨 train/test 泄漏）；词表在该桶全部训练作品上建、'
                    '不随 size 变化；NLL 对完整 V 维归一化，与 trigram 同口径'}
    path = os.path.join(args.out_dir, fn)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'runs': all_runs}, f, ensure_ascii=False, indent=2)
    print(f'✓ 汇总写入 {path}（{len(all_runs)} 条）')


def _agg(rs: List[Dict], key: str):
    v = np.asarray([r['final'][key] for r in rs], dtype=float)
    return float(v.mean()), float(v.std()), float(v.min()), float(v.max())


def write_report(runs: List[Dict], cfg: dict, args, tag: str):
    groups: Dict = {}
    for r in runs:
        groups.setdefault(r['bucket'], {}).setdefault(r['size'], []).append(r)

    L = []
    tc = max(r['test_chars'] for r in runs)
    ep = max(r['epochs'] for r in runs)
    L.append('# 数据量扫描 + 收敛对照（按作品切分 · 无泄漏）')
    L.append('')
    L.append(f'> 语料：`data/splits/`（按作品整篇切分，留出作品不出现在训练集）。'
             f'留出切片 {tc:,} 字，从留出作品轮流取前缀。')
    L.append(f'> 模型 cfg={cfg}｜batch={args.batch}｜lr={args.lr}｜'
             f'epochs={ep}（步数 = epochs × 训练字数 / (batch×ctx)，下限 {args.min_steps} 步）｜'
             f'min_freq={args.min_freq}')
    L.append(f'> NLL/PPL 均对完整 V 维归一化，模型与 trigram 在**同一批留出位置**上计算。')
    L.append('')
    for bk in sorted(groups):
        rows_by_size = groups[bk]
        L.append(f'## {bk}')
        L.append('')
        r0 = rows_by_size[max(rows_by_size)][0]
        L.append(f'训练作品 {r0["n_train_works"]} 部 · 留出作品 {r0["n_test_works"]} 部 · '
                 f'留出切片 {r0["test_chars"]:,} 字（UNK {r0["oov_chars"]}）· V={r0["V"]}')
        L.append('')
        L.append('| 训练字数 | 步数 | 种子数 | 模型 Top-1 | 模型 Top-5 | 模型 NLL | 模型 PPL | '
                 'trigram Top-1 | trigram NLL | trigram PPL | Top-1 | NLL |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for sz in sorted(rows_by_size):
            rs = rows_by_size[sz]
            m1, s1, lo1, hi1 = _agg(rs, 'top1')
            m5, _, _, _ = _agg(rs, 'top5')
            mn, sn, lon, hin = _agg(rs, 'nll')
            mp, _, _, _ = _agg(rs, 'ppl')
            tri = rs[0]['trigram']
            d1 = m1 - tri['top1']
            dn = mn - tri['nll']
            L.append(f'| {rs[0]["train_chars"]:,} | {rs[0]["steps"]} | {len(rs)} | '
                     f'{m1:.1f}%{" ±%.1f" % s1 if len(rs) > 1 else ""} | {m5:.1f}% | '
                     f'{mn:.3f}{" ±%.3f" % sn if len(rs) > 1 else ""} | {mp:.1f} | '
                     f'{tri["top1"]:.1f}% | {tri["nll"]:.3f} | {tri["ppl"]:.1f} | '
                     f'{"胜" if d1 > 0 else "负"} {d1:+.1f} | '
                     f'{"胜" if dn < 0 else "负"} {dn:+.3f} |')
        L.append('')
        # 收敛情况：末点相对中点还有多少提升
        big = max(rows_by_size)
        rs = rows_by_size[big]
        c = rs[0]['curve']
        if len(c) >= 2:
            L.append(f'收敛检查（{rs[0]["size_label"]}，{len(c)} 个评测点，曲线=val）：'
                     f'第 1 点 NLL={c[0]["nll"]:.3f} → 末点 NLL={c[-1]["nll"]:.3f}；'
                     f'倒数第 2 点 → 末点 {c[-2]["nll"]:.3f} → {c[-1]["nll"]:.3f}。')
            L.append('')
        probe = rs[0].get('trigram_alpha_best_on_test')
        if probe:
            L.append(f'trigram α 网格（在留出集上取最优，**属上界参考**）：'
                     f'α={probe["alpha"]} 时 NLL={probe["nll"]:.3f} Top-1={probe["top1"]:.1f}%'
                     f'（固定 α=0.1 为 NLL={rs[0]["trigram"]["nll"]:.3f}）。')
            L.append('')
    out = os.path.join(HERE, 'runs', f'wsplit_report{tag}.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'✓ 报告写入 {out}')


def _cn_font(matplotlib):
    for cand in ('Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC'):
        try:
            matplotlib.rcParams['font.sans-serif'] = [cand]
            matplotlib.rcParams['axes.unicode_minus'] = False
            break
        except Exception:
            continue


def plot_scaling(runs: List[Dict], cfg: dict, args, tag: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _cn_font(matplotlib)

    groups: Dict = {}
    for r in runs:
        groups.setdefault(r['bucket'], {}).setdefault(r['size'], []).append(r)
    bks = sorted(groups)
    palette = {'ALL': '#333333', 'scifi': '#d62728', 'wuxia': '#1f77b4',
               'urban': '#2ca02c', 'gudian': '#9467bd', 'story': '#ff7f0e'}

    fig, axes = plt.subplots(2, len(bks), figsize=(4.2 * len(bks), 7.4), sharex='col')
    if len(bks) == 1:
        axes = axes.reshape(2, 1)

    for j, bk in enumerate(bks):
        sizes = sorted(groups[bk])
        xs = sizes
        col = palette.get(bk, 'gray')
        for row, key in ((0, 'top1'), (1, 'nll')):
            ax = axes[row][j]
            mm = [np.mean([r['final'][key] for r in groups[bk][s]]) for s in sizes]
            lo = [np.min([r['final'][key] for r in groups[bk][s]]) for s in sizes]
            hi = [np.max([r['final'][key] for r in groups[bk][s]]) for s in sizes]
            ax.plot(xs, mm, 'o-', color=col, linewidth=2, markersize=7, label='Model')
            if any(h > l for h, l in zip(hi, lo)):
                ax.fill_between(xs, lo, hi, color=col, alpha=0.22,
                                label='Model 种子区间')
            tk = 'top1' if key == 'top1' else 'nll'
            tv = [groups[bk][s][0]['trigram'][tk] for s in sizes]
            ax.plot(xs, tv, 's--', color='#555555', alpha=0.75, linewidth=1.5,
                    markersize=6, label='Trigram(α=0.1)')
            ax.set_xscale('log')
            ax.set_xticks(xs)
            ax.set_xticklabels([groups[bk][s][0]['size_label'] for s in sizes],
                               fontsize=8, rotation=45)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(bk, fontsize=12)
                ax.set_ylabel('留出集 Top-1 (%)')
            else:
                ax.set_ylabel('留出集 NLL（越低越好）')
                ax.set_xlabel('训练字数（对数轴）')
            if j == 0:
                ax.legend(fontsize=8, loc='best')
    fig.suptitle(f'按作品切分的数据量扫描 · Model vs Trigram（cfg={cfg}，'
                 f'epochs={args.epochs}，{len({r["seed"] for r in runs})} 种子）', fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, '..', 'docs', 'img', f'worksplit_scaling{tag}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'✓ 图写入 {out}')


def plot_converge(runs: List[Dict], cfg: dict, args, tag: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _cn_font(matplotlib)

    groups: Dict = {}
    for r in runs:
        groups.setdefault(r['bucket'], {}).setdefault(r['size'], []).append(r)
    bks = sorted(groups)
    palette = {'ALL': '#333333', 'scifi': '#d62728', 'wuxia': '#1f77b4',
               'urban': '#2ca02c', 'gudian': '#9467bd', 'story': '#ff7f0e'}

    fig, axes = plt.subplots(1, len(bks), figsize=(4.2 * len(bks), 3.9), sharey=False)
    if len(bks) == 1:
        axes = [axes]
    for ax, bk in zip(axes, bks):
        big = max(groups[bk])
        rs = groups[bk][big]
        col = palette.get(bk, 'gray')
        for r in rs:
            c = r['curve']
            ax.plot([p['step'] for p in c], [p['nll'] for p in c], 'o-',
                    color=col, alpha=0.75, linewidth=1.4, markersize=5,
                    label=f'seed={r["seed"]}')
        ax.axhline(rs[0]['trigram']['nll'], color='#555555', linestyle='--',
                   linewidth=1.5, label='Trigram(α=0.1)')
        ax.set_title(f'{bk} · {rs[0]["size_label"]} 字', fontsize=12)
        ax.set_xlabel('训练步数')
        ax.set_ylabel('val 集 NLL（选型口径，越低越好）')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle(f'收敛曲线·val（最大数据量配置，逐种子；test 只在末点一次性验收）'
                 f'· cfg={cfg} epochs={args.epochs}',
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, '..', 'docs', 'img', f'worksplit_converge{tag}.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'✓ 图写入 {out}')


if __name__ == '__main__':
    main()
