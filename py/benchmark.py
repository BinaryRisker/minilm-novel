""" ============================================================================
py/benchmark.py — 训练吞吐与资源占用实测
----------------------------------------------------------------------------
职责：
  - 在指定语料上跑一段训练，测量：tokens/s、chars/s、单步耗时、峰值内存、
    验证/留出集评测开销。
  - 输出机器可读的 runs/benchmark_<corpus>_<device>.json，
    以及 Markdown 汇总 runs/benchmark_report.md。
  - 先预热再正式计时，避免首次 import/编译的噪声。

用法：
  python py/benchmark.py --corpus scifi --steps 400 --device cpu
============================================================================ """
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import tracemalloc

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from minilm_torch import MiniLM, build_vocab, encode, holdout_eval  # noqa


def benchmark(corpus_name: str, steps: int, batch: int, lr: float,
              seed: int, device: str, cfg: dict) -> dict:
    corpus_path = os.path.join(HERE, 'data', 'corpus.json')
    with open(corpus_path, 'r', encoding='utf-8') as f:
        c = json.load(f)
    text = c[corpus_name]['text']
    itos, stoi = build_vocab(text)
    ids = encode(text, stoi)
    split = math.floor(len(ids) * 0.9)
    train_ids = ids[:split]

    import random
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = MiniLM(len(itos), cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = max(10, math.floor(steps * 0.15))
    from minilm_torch import make_batch, lr_at

    # 预热 30 步，不计时
    model.train()
    for _ in range(30):
        tokens, targets = make_batch(train_ids, batch, cfg['ctx'], rng)
        tokens, targets = tokens.to(device), targets.to(device)
        opt.zero_grad()
        _, loss = model(tokens, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    if device.startswith('cuda'):
        # GPU 峰值显存从预热后开始算（建模块/搬卡不计入）
        torch.cuda.reset_peak_memory_stats(device)

    # 正式计时
    torch.manual_seed(seed)   # 复现预热前状态
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    tracemalloc.start()
    t0 = time.time()
    seg_loss = 0.0
    seg_n = 0
    for step in range(steps):
        tokens, targets = make_batch(train_ids, batch, cfg['ctx'], rng)
        tokens, targets = tokens.to(device), targets.to(device)
        opt.zero_grad()
        _, loss = model(tokens, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps, lr, warmup)
        opt.step()
        seg_loss += loss.item()
        seg_n += 1
    train_time = time.time() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 验证开销
    model_cpu = model.to('cpu')
    t1 = time.time()
    eval_res = holdout_eval(model_cpu, text, seed=seed)
    eval_time = time.time() - t1

    total_tokens = steps * batch * cfg['ctx']
    avg_loss = seg_loss / seg_n
    res = {
        'corpus': corpus_name,
        'cfg': cfg,
        'steps': steps,
        'batch': batch,
        'lr': lr,
        'seed': seed,
        'device': device,
        'train_time_sec': train_time,
        'eval_time_sec': eval_time,
        'avg_loss': avg_loss,
        'tokens_total': total_tokens,
        'tokens_per_sec': total_tokens / max(1e-9, train_time),
        'chars_per_sec': (steps * batch) / max(1e-9, train_time),
        'ms_per_step': 1000 * train_time / steps,
        'peak_memory_mb': peak / (1024 * 1024),
        'current_memory_mb': current / (1024 * 1024),
        'holdout_top1': eval_res['top1'],
        'holdout_top5': eval_res['top5'],
        'holdout_n': eval_res['n'],
    }
    if device.startswith('cuda') and torch.cuda.is_available():
        # 训练侧峰值显存：预算表的核心列（V3 P1：显存/RSS、OOM 与恢复）
        res['peak_vram_mb'] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
    return res


def run_sweep(corpus_name: str, combos, steps: int, lr: float, seed: int,
              device: str, cfg_base: dict, estimate_tokens: int = None,
              price_per_hour: float = None) -> list:
    """(ctx, batch) 组合扫描：每个组合独立建模训 steps 步。
    OOM 不是失败——记 `oom: true` 后清缓存继续，产出的就是「哪些组合可行」的表；
    给了 estimate_tokens/price 时按各组实测 tokens/s 换算小时数与费用。"""
    results = []
    for ctx, batch in combos:
        cfg = dict(cfg_base, ctx=ctx)
        tag = {'ctx': ctx, 'batch': batch, 'cfg': cfg}
        try:
            r = benchmark(corpus_name, steps, batch, lr, seed, device, cfg)
            r.update(tag)
            r['oom'] = False
            if estimate_tokens:
                hours = estimate_tokens / r['tokens_per_sec'] / 3600
                r['estimate'] = {'tokens': estimate_tokens,
                                 'hours': round(hours, 3),
                                 'cost_yuan': round(hours * price_per_hour, 2) if price_per_hour else None}
        except RuntimeError as e:  # noqa: BLE001 —— OOM 是扫描的**正常产出**，不是崩溃
            if 'out of memory' not in str(e).lower():
                raise
            r = dict(tag, oom=True, error=str(e)[:160])
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
        results.append(r)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='scifi')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.004)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--cfg', default=None)
    ap.add_argument('--cfg-file', default=None,
                    help='与 --cfg 二选一；从 JSON 文件读 cfg（避免 ssh heredoc 把 {...} 拆 token）')
    ap.add_argument('--sweep_ctx', default=None,
                    help='逗号分隔的 ctx 列表；与 --sweep_batch 同给时进入组合扫描模式'
                         '（每个组合独立建模，OOM 记为 oom:true 不崩溃）')
    ap.add_argument('--sweep_batch', default=None)
    ap.add_argument('--estimate_tokens', type=int, default=None,
                    help='按各组实测 tokens/s 估算「训这么多 token 要几小时」')
    ap.add_argument('--price_per_hour', type=float, default=None,
                    help='实例时价（元/小时），配合 --estimate_tokens 换算费用')
    ap.add_argument('--out', default=None, help='JSON 输出路径（默认按 corpus/device 命名）')
    args = ap.parse_args()

    cfg = dict(ctx=32, d=40, layers=2, heads=4, ffMul=4)
    if args.cfg:
        cfg.update(json.loads(args.cfg))
    # ssh heredoc 会把 "{...}" 拆成多 token；支持 --cfg-file 旁路
    if getattr(args, 'cfg_file', None):
        cfg.update(json.load(open(args.cfg_file, encoding='utf-8')))

    out_dir = os.path.join(HERE, 'runs')
    os.makedirs(out_dir, exist_ok=True)

    if args.sweep_ctx or args.sweep_batch:
        ctxs = [int(x) for x in args.sweep_ctx.split(',')] if args.sweep_ctx else [cfg['ctx']]
        bats = [int(x) for x in args.sweep_batch.split(',')] if args.sweep_batch else [args.batch]
        combos = [(c, b) for c in ctxs for b in bats]
        res = run_sweep(args.corpus, combos, args.steps, args.lr, args.seed,
                        args.device, cfg, args.estimate_tokens, args.price_per_hour)
        out_path = args.out or os.path.join(out_dir, f'benchmark_sweep_{args.corpus}_{args.device}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'protocol': 'sweep_v1', 'device': args.device, 'steps': args.steps,
                       'seed': args.seed, 'results': res},
                      f, ensure_ascii=False, indent=2)
        ok = [r for r in res if not r.get('oom')]
        print(f'[*] 组合扫描：{len(res)} 组，可行 {len(ok)} 组，OOM {len(res) - len(ok)} 组')
        for r in res:
            if r.get('oom'):
                line = f"  ctx={r['ctx']:>4} batch={r['batch']:>3}  OOM"
            else:
                line = (f"  ctx={r['ctx']:>4} batch={r['batch']:>3}  "
                        f"tokens/s={r['tokens_per_sec']:.0f}  "
                        f"vram={r.get('peak_vram_mb', '-')}MB")
            est = r.get('estimate')
            if est:
                line += f"  → {est['hours']}h / ¥{est['cost_yuan']}"
            print(line)
        print(f'Benchmark saved to {out_path}')
        return

    res = benchmark(args.corpus, args.steps, args.batch, args.lr,
                    args.seed, args.device, cfg)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    out_path = args.out or os.path.join(out_dir, f'benchmark_{args.corpus}_{args.device}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f'\nBenchmark saved to {out_path}')

    # 简易 markdown 追加
    md_path = os.path.join(out_dir, 'benchmark_report.md')
    lines = [
        f'## {args.corpus} / {args.device}',
        '',
        f'- cfg: `{json.dumps(cfg, ensure_ascii=False)}`',
        f'- steps={args.steps} batch={args.batch} lr={args.lr} seed={args.seed}',
        f'- train_time={res["train_time_sec"]:.2f}s, eval_time={res["eval_time_sec"]:.3f}s',
        f'- tokens/s={res["tokens_per_sec"]:.1f}, chars/s={res["chars_per_sec"]:.1f}, ms/step={res["ms_per_step"]:.2f}',
        f'- peak_memory={res["peak_memory_mb"]:.2f} MB',
        f'- holdout Top-1={res["holdout_top1"]:.2f}%, Top-5={res["holdout_top5"]:.2f}%',
        '',
    ]
    with open(md_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'Report appended to {md_path}')


if __name__ == '__main__':
    main()