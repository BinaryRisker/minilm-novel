""" ============================================================================
py/train.py — MiniLM 训练脚本（命令行入口）
----------------------------------------------------------------------------
职责：
  - 从 py/data/corpus.json 加载一个语料
  - 支持 save/resume：每次训练把 checkpoint 写到 runs/<run_id>/checkpoint.pt
  - 保存 tokenizer (itos.json) 与 meta.json（配置、语料、训练步数、随机种子）
  - 输出训练日志和阶段性验证指标

用法：
  python py/train.py --corpus scifi --steps 400 --out runs/scifi_s1234_n400
  python py/train.py --resume runs/scifi_s1234_n400/checkpoint.pt
============================================================================ """
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from minilm_torch import MiniLM, build_vocab, encode, holdout_eval  # noqa


def load_corpus(corpus_name: str) -> str:
    path = os.path.join(HERE, 'data', 'corpus.json')
    with open(path, 'r', encoding='utf-8') as f:
        c = json.load(f)
    if corpus_name not in c or not c[corpus_name].get('text'):
        raise ValueError(f'语料 {corpus_name} 不存在或为空。可用：{list(c.keys())}')
    return c[corpus_name]['text']


def save_checkpoint(path: str, model, itos: list, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'itos': itos,
        'meta': meta,
    }, path)


def load_checkpoint(path: str, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    itos = ckpt['itos']
    meta = ckpt['meta']
    cfg = meta['cfg']
    model = MiniLM(len(itos), cfg).to(device)
    model.load_state_dict(ckpt['model'])
    return model, itos, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='scifi', help='corpus key in py/data/corpus.json')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.004)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--cfg', default=None, help='JSON string to override DEFAULT_CFG, e.g. {"d":40,"layers":2}')
    ap.add_argument('--out', default=os.path.join('py', 'runs', 'run_default'),
                    help='output directory relative to project root; also used as run_id')
    ap.add_argument('--resume', default=None, help='path to checkpoint.pt to resume')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--snapshot_every', type=int, default=0,
                    help='save checkpoint every N steps; 0 = only final')
    args = ap.parse_args()

    if args.resume:
        print(f'[*] Resuming from {args.resume}')
        model, itos, meta = load_checkpoint(args.resume, args.device)
        cfg = meta['cfg']
        corpus_name = meta.get('corpus', 'unknown')
        text = load_corpus(corpus_name) if corpus_name != 'unknown' else ''
        start_step = meta.get('step', 0)
        run_id = os.path.basename(os.path.dirname(args.resume))
        out_dir = os.path.dirname(args.resume)
        seed = meta.get('seed', 1234)
    else:
        text = load_corpus(args.corpus)
        cfg = dict(ctx=32, d=40, layers=2, heads=4, ffMul=4)
        if args.cfg:
            cfg.update(json.loads(args.cfg))
        run_id = args.out
        out_dir = run_id if os.path.isabs(run_id) else os.path.join(ROOT, run_id)
        os.makedirs(out_dir, exist_ok=True)
        start_step = 0
        seed = args.seed

        itos, stoi = build_vocab(text)
        model = MiniLM(len(itos), cfg).to(args.device)
        meta = {
            'corpus': args.corpus,
            'cfg': cfg,
            'steps': args.steps,
            'batch': args.batch,
            'lr': args.lr,
            'seed': seed,
            'step': 0,
            'run_id': run_id,
        }
        # save tokenizer upfront
        with open(os.path.join(out_dir, 'itos.json'), 'w', encoding='utf-8') as f:
            json.dump(itos, f, ensure_ascii=False, indent=2)

    # training logger
    log_path = os.path.join(out_dir, 'train.log')
    log_f = open(log_path, 'a', encoding='utf-8')

    def log(s):
        print(s)
        log_f.write(s + '\n')
        log_f.flush()

    log(f'run_id={run_id}')
    log(f'corpus={meta.get("corpus", args.corpus)} chars={len(text)}')
    log(f'cfg={json.dumps(cfg, ensure_ascii=False)}')
    log(f'seed={seed} steps={args.steps} batch={args.batch} lr={args.lr}')

    # fine-grained training with intermediate snapshots
    if args.resume:
        log(f'Resuming from step {start_step}; no retraining in this simple script.')
    else:
        ids = encode(text, stoi)
        split = math.floor(len(ids) * 0.9)
        train_ids = ids[:split]
        test_ids = ids[split:]

        import random
        rng = random.Random(seed)
        torch.manual_seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        warmup = max(10, math.floor(args.steps * 0.15))
        from minilm_torch import make_batch, lr_at

        model.train()
        seg_loss, seg_n = 0.0, 0
        t0 = time.time()
        for step in range(args.steps):
            tokens, targets = make_batch(train_ids, args.batch, cfg['ctx'], rng)
            tokens, targets = tokens.to(args.device), targets.to(args.device)
            opt.zero_grad()
            _, loss = model(tokens, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for g in opt.param_groups:
                g['lr'] = lr_at(step, args.steps, args.lr, warmup)
            opt.step()
            seg_loss += loss.item()
            seg_n += 1
            meta['step'] = step + 1

            if (args.snapshot_every and (step + 1) % args.snapshot_every == 0) or step == args.steps - 1:
                save_checkpoint(os.path.join(out_dir, 'checkpoint.pt'), model, itos, meta)
                elapsed = time.time() - t0
                avg_loss = seg_loss / seg_n
                log(f'  step {step + 1}/{args.steps}  loss={avg_loss:.4f}  elapsed={elapsed:.1f}s')
                seg_loss, seg_n = 0.0, 0

        # holdout eval
        from minilm_torch import holdout_eval
        eval_res = holdout_eval(model.to('cpu'), text, seed=seed)
        log(f'Holdout: {json.dumps(eval_res, ensure_ascii=False)}')
        meta['eval'] = eval_res
        save_checkpoint(os.path.join(out_dir, 'checkpoint.pt'), model, itos, meta)
        with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        log(f'Checkpoint saved to {os.path.join(out_dir, "checkpoint.pt")}')

    log_f.close()


if __name__ == '__main__':
    main()