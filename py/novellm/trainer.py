""" ============================================================================
py/novellm/trainer.py — 分片数据集上的训练器（V3 P1）
----------------------------------------------------------------------------
与 py/train.py（旧路径）的分工：
  train.py 吃「整段文本 + 连续编码」，是 L0 对齐与 D0（≤千万字级桶）的路径；
  本模块吃 **py/novellm/dataset.py 的预编码分片**（memmap、O(1) 常驻内存），
  是 D1（全量含 localugc，18.8 亿字）的唯一可行喂入方式。

能力（V3 P1 点名项）：
  - 预编码分片喂数（ShardedCorpus.sample_batch，与 make_batch 同语义）；
  - **累计梯度**（--accum：大 ctx 小显存也能等效大 batch）；
  - **验证**：val 分片上定期评测（val 选型口径，评测协议 V1）；
  - **取消/超限安全存盘**：SIGINT / max_hours 触发 → 保存 checkpoint 再退出，
    宁可少训不可丢训练；
  - **复用已验证状态契约**：checkpoint v2（py/checkpoint.py，与旧路径同一份实现）；
    直训 vs 续训的等价性由同一契约保证（P0 已验证）。

用法：
  python -X utf8 -m py.novellm.trainer --split train \
      --cfg t30m --steps 100000 --accum 4 --out py/runs/d1_t30m
============================================================================ """
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
import random
import signal
import sys
import time
from typing import Callable

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
if PY not in sys.path:
    sys.path.insert(0, PY)

import numpy as np  # noqa: E402

import checkpoint  # noqa: E402
from minilm_torch import lr_at  # noqa: E402
from novellm import model as nov_model  # noqa: E402
from novellm.dataset import ShardedCorpus  # noqa: E402
from repro import set_global_seed  # noqa: E402
import run_identity  # noqa: E402


@torch.no_grad()
def val_eval(model, corpus: ShardedCorpus, ctx: int, max_windows: int = 256) -> dict:
    """val 分片上的轻量评测（前 max_windows 个 ctx 窗，选模型用别太贵）。
    窗口含目标位（ctx+1 长），逐位置 top-1/top-5/NLL，与 holdout_eval 同口径。"""
    import math
    model.eval()
    device = next(model.parameters()).device
    top1 = tot5 = total = 0
    nll_sum = 0.0
    for w in corpus.iter_windows(ctx, max_windows=max_windows):
        chunk = torch.from_numpy(w[:ctx].astype(np.int64, copy=False)).unsqueeze(0).to(device)
        tgt = w[1:ctx + 1].astype(np.int64)
        logits, _ = model(chunk)
        lg = logits[0].float()
        logp = torch.log_softmax(lg, dim=-1)
        topv, topi = lg.topk(5)
        for i in range(ctx):
            pred = topi[i].tolist()
            total += 1
            if pred[0] == tgt[i]:
                top1 += 1
            if tgt[i] in pred:
                tot5 += 1
            nll_sum += float(-logp[i, tgt[i]])
    n = max(1, total)
    nll = nll_sum / n
    return {'top1': 100.0 * top1 / n, 'top5': 100.0 * tot5 / n, 'nll': nll,
            'ppl': math.exp(nll) if nll < 700 else float('inf'), 'n': total,
            'windows': max_windows}


def train(cfg: dict, train: ShardedCorpus, val: ShardedCorpus, itos: list,
          steps: int, batch: int, accum: int, lr: float, seed: int,
          out_dir: str, ctx: int = None, val_every: int = 0,
          val_windows: int = 256, snapshot_every: int = 0,
          max_hours: float = None, resume_path: str = None,
          cancel_check: Callable[[], bool] = None, device: str = 'cpu',
          log: Callable[[str], None] = print) -> dict:
    """分片训练主循环。返回 meta（同时落 out_dir/meta.json）。
    取消/超限 → 保存 checkpoint 后返回 meta['cancelled']=True，退出码由 CLI 决定。"""
    os.makedirs(out_dir, exist_ok=True)
    ctx = ctx or cfg['ctx']
    V = len(itos)

    model, prov = nov_model.build(cfg, V, seed, device=device)
    start_step = 0
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if resume_path:
        ckpt = checkpoint.load_ckpt_raw(resume_path, device=device)
        start_step = checkpoint.load_state(model, opt, ckpt, device=device)
        seed = ckpt['meta'].get('seed', seed)
        log(f'[*] 从 {resume_path} 续训：step {start_step} → {steps}')
    else:
        set_global_seed(seed)                    # 数据采样流（模型初始化已在上面的 seed 里）
        rng_state_note = None

    # 采样 rng **按步派生**（Random(f'{seed}:{step}')），不从上一步连续消费——
    # 这让「续训到 N」与「直训 N」在数据采样上天然逐位一致（P0 真正 resume 的要求），
    # 且与 accum/batch 的每次消耗数无关。

    # 数据身份：分片 meta 原样入库（含来源清单指纹与词表指纹）
    data_id = {
        'source_split_fingerprint': train.meta.get('source_split_fingerprint'),
        'vocab_fingerprint': train.meta.get('vocab_fingerprint'),
        'n_tokens': train.n_tokens, 'n_docs': train.meta.get('n_docs'),
        'split': train.split, 'shard_dir': os.path.basename(os.path.abspath(train.shard_dir)),
    }
    rid = run_identity.run_identity(cfg, seed, steps, batch, lr, {
        'split_fingerprint': data_id['source_split_fingerprint'],
        'manifest': f'shards:{data_id["shard_dir"]}',
        'near_dup_merged': None,
    })

    meta = {
        'run_id': out_dir, 'cfg': cfg, 'V': V, 'seed': seed,
        'steps': steps, 'batch': batch, 'accum': accum, 'lr': lr,
        'ctx': ctx, 'step': start_step, 'device': device,
        'train_path': 'novellm.trainer',
        'protocol': 'val_select_test_once_v1',
        'data': data_id, 'provenance': prov,
        'run_identity': rid, 'val_curve': [], 'cancelled': False,
    }
    if resume_path:
        with open(resume_path, 'rb') as f:
            pass
        old_meta = checkpoint.load_ckpt_raw(resume_path, device=device)['meta']
        meta['val_curve'] = old_meta.get('val_curve', [])
        meta['resumed_from'] = resume_path

    deadline = time.time() + max_hours * 3600 if max_hours else None
    stop = {'flag': False}

    def _sig(_f, _b):
        stop['flag'] = True
    try:
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)
    except ValueError:                           # 非 main 线程（测试里）跳过信号注册
        pass

    def save(step):
        p = checkpoint.save_checkpoint(os.path.join(out_dir, 'checkpoint.pt'),
                                       model, opt, itos, meta, step)
        with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        log(f'  [ckpt] step={step} -> {p}')

    warmup = max(10, steps // 15)
    model.train()
    opt.zero_grad()
    accum_count = 0
    t0 = time.time()
    seg_loss = seg_n = 0.0

    for step in range(start_step, steps):
        if cancel_check is not None and cancel_check():
            stop['flag'] = True
        if stop['flag'] or (deadline and time.time() > deadline):
            meta['cancelled'] = True
            meta['cancel_reason'] = ('signal' if stop['flag'] else 'max_hours')
            log(f'[!] {"收到取消信号" if stop["flag"] else "到达 max_hours 上限"}'
                f'，安全存盘后退出（step {step}）')
            save(step)
            return meta

        # ---- 累计梯度：accum 个 micro-batch 才 opt.step 一次 ----
        micro_loss = 0.0
        step_rng = random.Random(f'{seed}:{step}')
        for _ in range(accum):
            tokens, targets = train.sample_batch(batch, ctx, step_rng)
            tokens, targets = tokens.to(device), targets.to(device)
            _, loss = model(tokens, targets)
            (loss / accum).backward()
            micro_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps, lr, warmup)
        opt.step()
        opt.zero_grad()
        accum_count += 1
        seg_loss += micro_loss / accum
        seg_n += 1
        meta['step'] = step + 1

        if val_every and (step + 1) % val_every == 0 and val is not None:
            v = val_eval(model, val, ctx, max_windows=val_windows)
            meta['val_curve'].append({'step': step + 1, **v})
            model.train()
            log(f'  step {step + 1}/{steps} loss={seg_loss / max(1, seg_n):.4f} '
                f'val: top1={v["top1"]:.2f}% nll={v["nll"]:.3f}')
            seg_loss, seg_n = 0.0, 0

        if snapshot_every and (step + 1) % snapshot_every == 0:
            save(step + 1)

    save(steps)
    meta['wallclock_sec'] = round(time.time() - t0, 1)
    if val is not None and meta['val_curve']:
        meta['val_final'] = meta['val_curve'][-1]
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f'[*] 训练完成：steps={steps} run_identity={rid} '
        f'wallclock={meta["wallclock_sec"]}s')
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--shard-dir', default=os.path.join(PY, 'runs', 'shards'))
    ap.add_argument('--cfg', default='micro', help='micro / t30m / JSON 覆盖')
    ap.add_argument('--steps', type=int, required=True)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--accum', type=int, default=1, help='累计梯度步数（等效 batch=batch×accum）')
    ap.add_argument('--lr', type=float, default=0.004)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--val-every', type=int, default=0, help='每 N 步评一次 val（0=不评）')
    ap.add_argument('--val-windows', type=int, default=256)
    ap.add_argument('--val-split', default='val')
    ap.add_argument('--snapshot-every', type=int, default=0, help='每 N 步存一次 checkpoint（0=只存末点）')
    ap.add_argument('--max-hours', type=float, default=None)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='训练设备；默认有 CUDA 时用 cuda，否则 cpu')
    args = ap.parse_args()

    cfg = nov_model.resolve_cfg(args.cfg)
    train_c = ShardedCorpus(args.shard_dir, args.split)
    itos = train_c.itos
    val_c = None
    try:
        val_c = ShardedCorpus(args.shard_dir, args.val_split)
    except Exception:  # noqa: BLE001 —— val 分片不存在时允许无 val 训练（冒烟）
        print(f'[!] 无 {args.val_split} 分片，本次训练不带 val 曲线')

    meta = train(cfg, train_c, val_c, itos, args.steps, args.batch, args.accum,
                 args.lr, args.seed, args.out, ctx=cfg['ctx'], device=args.device,
                 val_every=args.val_every, val_windows=args.val_windows,
                 snapshot_every=args.snapshot_every, max_hours=args.max_hours,
                 resume_path=args.resume)
    return 1 if meta.get('cancelled') else 0


if __name__ == '__main__':
    sys.exit(main())
