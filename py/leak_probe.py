""" ============================================================================
py/leak_probe.py — 泄漏值多少分？（同源泄漏对留出指标的虚高量化）
----------------------------------------------------------------------------
背景：py/audit_splits.py 在 gudian 桶发现，按作品切分后留出作品仍有 12%~14.7% 的
40 字片段能在训练集里逐字找到（异名同书/选本重复）。py/dedup_works.py 把同源作品
合并进同一 split 修掉了它（v2 清单：文本级命中 0/150）。

本脚本回答「修这个值不值」：
  用同一份训练数据（v1 train）训一个模型，然后**逐部作品**在留出集上评测，
  把作品分成两组：
    · 被泄漏污染的作品（其同源兄弟在训练集里）—— v1 把它们放在 test/val
    · 干净作品（训练集里没有任何同源）
  看两组的 Top-1/NLL 差多少。若污染组显著更高，说明「报平均分」时这部分是白送的。

用法：
  python py/leak_probe.py --bucket gudian --chars 100000 --steps 500
  python py/leak_probe.py --bucket gudian --model-manifest data/manifest.jsonl \
                          --eval-manifests data/manifest.jsonl,data/manifest_v2.jsonl
============================================================================ """
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from minilm_torch import MiniLM, build_vocab, encode, eval_ids, make_batch, lr_at  # noqa
from repro import set_global_seed, random_seed_bundle  # noqa
import splits as S  # noqa

CFG = dict(ctx=32, d=40, layers=2, heads=4, ffMul=4)


def train_model(text: str, steps: int, batch: int, lr: float, seed: int):
    set_global_seed(seed)
    itos, stoi = build_vocab(text)
    model = MiniLM(len(itos), CFG)
    ids = encode(text, stoi)
    rng = random_seed_bundle(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = max(10, int(steps * 0.15))
    model.train()
    t0 = time.time()
    for step in range(steps):
        tokens, targets = make_batch(ids, batch, CFG['ctx'], rng)
        opt.zero_grad()
        _, loss = model(tokens, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps, lr, warmup)
        opt.step()
    return model, itos, stoi, len(ids), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bucket', default='gudian')
    ap.add_argument('--chars', type=int, default=100000)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.004)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--model-manifest', default=S.DEFAULT_MANIFEST,
                    help='训练数据来自哪份清单（默认 v1 = 有同源泄漏的那份）')
    ap.add_argument('--report', default=None,
                    help='与 --manifest 配套的切分报告（v2 清单要传 data/split_report_v2.json）')
    ap.add_argument('--eval-manifests', default=f'{S.DEFAULT_MANIFEST},data/manifest_v2.jsonl')
    ap.add_argument('--out', default=os.path.join(ROOT, 'dev', 'verify', 'leak_probe.json'))
    args = ap.parse_args()

    mp = args.model_manifest if os.path.isabs(args.model_manifest) else os.path.join(ROOT, args.model_manifest)
    kw = {'manifest_path': mp}
    if args.report:
        kw['report_path'] = args.report
    train_text = S.bucket_joined(args.bucket, **kw)['train'][:args.chars]
    print(f'[*] 训练：bucket={args.bucket} manifest={os.path.basename(mp)} '
          f'chars={len(train_text):,} steps={args.steps}')
    model, itos, stoi, n_ids, secs = train_model(train_text, args.steps, args.batch, args.lr, args.seed)
    print(f'    完成 {secs:.1f}s，训练 token={n_ids:,}，V={len(itos)}')

    rep = {'protocol': {'bucket': args.bucket, 'model_manifest': os.path.basename(mp),
                        'chars': len(train_text), 'steps': args.steps, 'seed': args.seed,
                        'cfg': CFG, 'train_seconds': round(secs, 1)},
           'manifests': {}}

    for mf in args.eval_manifests.split(','):
        mf = mf.strip()
        mfp = mf if os.path.isabs(mf) else os.path.join(ROOT, mf)
        if not os.path.exists(mfp):
            print(f'[!] 跳过不存在的清单 {mf}')
            continue
        kw2 = {'manifest_path': mfp}
        base = os.path.basename(mfp)
        if 'v2' in base:
            kw2['report_path'] = os.path.join(ROOT, 'data', 'split_report_v2.json')
        elif args.report:
            kw2['report_path'] = args.report
        t = S.bucket_texts(args.bucket, **kw2)
        per_work = []
        for sp in ('val', 'test'):
            for sid, title, txt in t[sp]:
                ids = encode(txt[:20000], stoi)
                oov = sum(1 for ch in txt[:20000] if ch not in itos)
                ev = eval_ids(model, ids, ctx=CFG['ctx'])
                per_work.append({'story_id': sid, 'title': title, 'split': sp,
                                 'chars': min(len(txt), 20000), 'oov': oov,
                                 'top1': round(ev['top1'], 2), 'nll': round(ev['nll'], 3)})
        rep['manifests'][mf] = {'per_work': per_work}
        print(f'\n== 评测清单 {os.path.basename(mfp)} ==')
        print(f'   {"作品":<14}{"split":>6}{"字数":>8}{"Top-1":>8}{"NLL":>8}')
        for w in per_work:
            print(f'   {w["title"][:12]:<14}{w["split"]:>6}{w["chars"]:>8}{w["top1"]:>8.2f}{w["nll"]:>8.3f}')

    # 污染组 vs 干净组（用 v2 清单判定：v1 测试集里被 v2 移走的作品 = 被污染作品）
    v2p = os.path.join(ROOT, 'data', 'manifest_v2.jsonl')
    v1_first = list(rep['manifests'].values())[0]['per_work']
    if os.path.exists(v2p):
        v2split = {r['story_id']: r['split'] for r in S.read_manifest(v2p)}
        moved = {sid for sid, sp in v2split.items() if sp == 'train'}
        leaked = [w for w in v1_first if w['story_id'] in moved and w['split'] in ('val', 'test')]
        clean = [w for w in v1_first if w['story_id'] not in moved]

        def mean(xs, k):
            return sum(x[k] for x in xs) / max(1, len(xs))

        rep['group_compare'] = {
            'leaked_works': [w['title'] for w in leaked],
            'clean_works': [w['title'] for w in clean],
            'leaked_mean_top1': round(mean(leaked, 'top1'), 2),
            'clean_mean_top1': round(mean(clean, 'top1'), 2),
            'leaked_mean_nll': round(mean(leaked, 'nll'), 3),
            'clean_mean_nll': round(mean(clean, 'nll'), 3),
            'top1_gap_pt': round(mean(leaked, 'top1') - mean(clean, 'top1'), 2),
            'nll_gap': round(mean(leaked, 'nll') - mean(clean, 'nll'), 3),
        }
        g = rep['group_compare']
        print('\n== 污染组 vs 干净组（同一模型、同一份训练数据）==')
        print(f'   被污染（同源兄弟在训练集里）{len(leaked)} 部：'
              f'{"、".join(g["leaked_works"])}')
        print(f'     平均 Top-1={g["leaked_mean_top1"]:.2f}%  NLL={g["leaked_mean_nll"]:.3f}')
        print(f'   干净 {len(clean)} 部：{"、".join(g["clean_works"])}')
        print(f'     平均 Top-1={g["clean_mean_top1"]:.2f}%  NLL={g["clean_mean_nll"]:.3f}')
        print(f'   ★ 差值：Top-1 {g["top1_gap_pt"]:+.2f}pt，NLL {g["nll_gap"]:+.3f}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f'\n[*] 报告 -> {args.out}')


if __name__ == '__main__':
    main()
