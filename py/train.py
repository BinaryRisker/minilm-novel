""" ============================================================================
py/train.py — MiniLM 训练脚本（命令行入口）
----------------------------------------------------------------------------
职责：
  - 读语料：`--bucket <桶>` 走 py/splits.py 的**按作品切分**（词表只在 train 上建，
    val 用于选模型，test 只在最后评一次）；不带 --bucket 时退回 corpus.json 的
    连续 90/10（JS 对齐口径，**有同作品家族泄漏，只许用于对齐验证**）
  - 随机种子：在**建模型之前**统一设置（py/repro.py），保证初始权重可复现
  - checkpoint v2：完整保存 model + optimizer + 全局步数 + lr 调度位置 + 两套 RNG 状态，
    因此「直训 N 步」与「训 N/2 再续训到 N」逐位等价（py/verify_resume.py 验证）
  - 阶段快照：--snapshot_every 步存一次，日志同时记 val 指标

用法：
  python py/train.py --bucket story --steps 600 --out py/runs/loop_s1234
  python py/train.py --resume py/runs/loop_s1234/checkpoint.pt --steps 900
  python py/train.py --corpus scifi --steps 400 --out py/runs/scifi_s1234_n400   # 旧口径（泄漏，见上）

checkpoint 字段（v2）：
  version / model / opt / itos / meta / rng / step
  旧版本（只有 model/itos/meta）仍可读入，但会告警：optimizer 状态与 RNG 已丢失，
  续训结果不会与直训等价。
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
from minilm_torch import MiniLM, build_vocab, encode, eval_ids, make_batch, lr_at  # noqa
from repro import set_global_seed, rng_state, set_rng_state  # noqa
import splits as splits_mod  # noqa
import run_identity  # noqa

CKPT_VERSION = 2


def load_corpus(corpus_name: str) -> str:
    path = os.path.join(HERE, 'data', 'corpus.json')
    with open(path, 'r', encoding='utf-8') as f:
        c = json.load(f)
    if corpus_name not in c or not c[corpus_name].get('text'):
        raise ValueError(f'语料 {corpus_name} 不存在或为空。可用：{list(c.keys())}')
    return c[corpus_name]['text']


def save_checkpoint(path: str, model, opt, itos: list, meta: dict, step: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'version': CKPT_VERSION,
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'itos': itos,
        'meta': meta,
        'step': step,
        'rng': rng_state(),
    }, path)


def load_checkpoint(path: str, device='cpu'):
    """返回 (model, itos, meta, ckpt)。ckpt 里带不带 opt/rng 由版本决定。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    itos = ckpt['itos']
    meta = ckpt['meta']
    cfg = meta['cfg']
    model = MiniLM(len(itos), cfg).to(device)
    model.load_state_dict(ckpt['model'])
    if 'opt' not in ckpt:
        print('[!] 旧格式 checkpoint：没有 optimizer 状态与 RNG 状态，'
              '续训不会与直训等价（Adam 动量从零重启）。')
    return model, itos, meta, ckpt


def bucket_joined(bucket: str, manifest: str = None, report: str = None) -> dict:
    """按桶取 train/val/test；manifest 给定时用那份清单的 split 归属
    （py/dedup_works.py 产出的 manifest_v2.jsonl 已消除同源跨集合）。"""
    kw = {'manifest_path': manifest} if manifest else {}
    if report:
        kw['report_path'] = report
    return splits_mod.bucket_joined(bucket, **kw)


@torch.no_grad()
def quick_eval(model, ids, ctx, max_points: int = 512):
    """在给定 id 序列上取前 max_points 个位置做轻量评测（选模型用，别太贵）。"""
    if not ids or len(ids) < ctx + 2:
        return None
    sub = ids[:max_points + ctx + 1]
    model.eval()
    r = eval_ids(model, sub, ctx=ctx)
    model.train()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='scifi', help='corpus key in py/data/corpus.json（不带 --bucket 时生效）')
    ap.add_argument('--bucket', default=None,
                    help='按作品切分的桶名（gudian/wuxia/urban/story/scifi/xiandai/ALL）。'
                         '给定时 train/val/test 用 py/splits.py 的按作品切分，词表只在 train 上建')
    ap.add_argument('--manifest', default=None,
                    help='作品清单（默认 data/manifest.jsonl；用 py/dedup_works.py 产出的 '
                         'manifest_v2.jsonl 可得到「同源作品不跨集合」的干净切分）')
    ap.add_argument('--report', default=None,
                    help='与 --manifest 配套的切分报告（用 manifest_v2.jsonl 时传 data/split_report_v2.json）')
    ap.add_argument('--splits', default=None,
                    help='兼容旧用法：data/splits 目录（只有 --bucket 缺失时才读全桶合并文件）')
    ap.add_argument('--steps', type=int, default=400,
                    help='**计划**总步数（lr 调度按它归一；续训时是「续到」的绝对步数）')
    ap.add_argument('--stop_at', type=int, default=0,
                    help='本阶段停在第几步（0=停到 --steps）。用于「先训一半存盘再续训」：'
                         '两阶段都用同一个 --steps，调度形状才一致 —— '
                         '否则第一阶段的 warmup/余弦是按更小的总步数归一，续训必然不等价')
    ap.add_argument('--chars', type=int, default=0,
                    help='只用语料前 N 字（0=全量）。续训时 0 表示沿用 checkpoint 里记录的规模 —— '
                         '否则「前半段用 2 万字、续训用全量」会在数据层面就不同，等价性无从谈起（踩过）')
    ap.add_argument('--val_chars', type=int, default=20000, help='val 评测用的字数上限（0=全量）')
    ap.add_argument('--test_chars', type=int, default=50000, help='test 评测用的字数上限（0=全量）')
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
    ap.add_argument('--allow_leaky', action='store_true',
                    help='显式允许旧口径（corpus.json 连续 90/10，同作品家族跨 train/test）。'
                         '不带此开关时，--corpus 路径会打印泄漏警告')
    args = ap.parse_args()

    # ---------------- 载入 / 续训 ----------------
    ckpt = None
    if args.resume:
        print(f'[*] Resuming from {args.resume}')
        model, itos, meta, ckpt = load_checkpoint(args.resume, args.device)
        cfg = meta['cfg']
        out_dir = os.path.dirname(os.path.abspath(args.resume))
        run_id = meta.get('run_id', os.path.basename(out_dir))
        start_step = int(ckpt.get('step', meta.get('step', 0)))
        seed = int(meta.get('seed', args.seed))
        bucket = meta.get('bucket')
        manifest = meta.get('manifest')
        report = meta.get('split_report')
        # 续训必须能重建同一份训练数据
        if bucket:
            sp = bucket_joined(bucket, manifest, report)
            text, val_text, test_text = sp['train'], sp['val'], sp['test']
        else:
            text = load_corpus(meta.get('corpus', args.corpus))
            val_text = test_text = None
        # 训练切片规模必须与前半段一致：--chars 未给时沿用 checkpoint 记录
        chars = args.chars or int(meta.get('chars', 0) or 0)
        if chars:
            text = text[:chars]
        if chars != int(meta.get('chars', 0) or 0):
            print(f'[!] 训练切片规模与 checkpoint 不同（{meta.get("chars")} → {chars}）：'
                  '续训用的是不同数据，结果不会与直训等价')
        stoi = {ch: i for i, ch in enumerate(itos)}
        # 恢复 RNG：必须在建 optimizer / 下一步采样之前
        rng_ok = set_rng_state(ckpt.get('rng'))
        if not rng_ok:
            # 没有 RNG 状态（旧格式 checkpoint）：只能按 seed 重新开始采样，
            # batch 顺序与直训不同 —— 这是「续训不等价」的一个独立成因，
            # py/verify_resume.py 的 C1/C3 就是这一格。重设 seed 让它至少可复现。
            set_global_seed(seed)
            print('[!] checkpoint 无 RNG 状态：批量采样顺序将从 seed 重新开始，'
                  '续训结果与直训**不等价**（不要用它做等价性结论）')
    else:
        bucket = args.bucket
        manifest = args.manifest
        report = args.report
        val_text = test_text = None
        if bucket:
            sp = bucket_joined(bucket, manifest, report)
            text, val_text, test_text = sp['train'], sp['val'], sp['test']
        elif args.splits:
            split_dir = args.splits if os.path.isabs(args.splits) else os.path.join(ROOT, args.splits)
            with open(os.path.join(split_dir, 'train.txt'), encoding='utf-8') as f:
                text = f.read()
            with open(os.path.join(split_dir, 'test.txt'), encoding='utf-8') as f:
                test_text = f.read()
        else:
            text = load_corpus(args.corpus)
            if not args.allow_leaky:
                print('[!] 旧口径警告：corpus.json 是「按作品截断后拼接」的整桶文本，'
                      '连续 90/10 会让同一部作品的头进 train、尾进 test（同作品家族泄漏）。'
                      '质量结论请用 --bucket；本路径只用于 JS 逐位对齐验证。')
        if args.chars:
            text = text[:args.chars]
        cfg = dict(ctx=32, d=40, layers=2, heads=4, ffMul=4)
        if args.cfg:
            cfg.update(json.loads(args.cfg))
        run_id = args.out
        out_dir = run_id if os.path.isabs(run_id) else os.path.join(ROOT, run_id)
        os.makedirs(out_dir, exist_ok=True)
        start_step = 0
        seed = int(args.seed)
        # 关键顺序：先设种子，再建词表/模型（否则初始权重不受 seed 控制）
        set_global_seed(seed)
        itos, stoi = build_vocab(text)
        model = MiniLM(len(itos), cfg).to(args.device)
        meta = {
            'corpus': args.corpus if not bucket else None,
            'bucket': bucket,
            'manifest': args.manifest,
            'split_report': args.report,
            'splits': args.splits,
            'cfg': cfg,
            'steps': args.steps,
            'batch': args.batch,
            'lr': args.lr,
            'seed': seed,
            'step': 0,
            'chars': args.chars,
            'run_id': run_id,
            'vocab_source': 'train_only' if (bucket or args.splits) else 'full_text(JS对齐)',
            'leaky_protocol': bool(not bucket and not args.splits),
        }
        with open(os.path.join(out_dir, 'itos.json'), 'w', encoding='utf-8') as f:
            json.dump(itos, f, ensure_ascii=False, indent=2)

    stop_at = args.stop_at or args.steps
    if start_step >= stop_at:
        print(f'[*] checkpoint 已在 step {start_step}，本阶段目标 {stop_at}：无事可做')
        return
    if args.resume and int(ckpt.get('meta', {}).get('steps', args.steps)) != args.steps:
        print(f'[!] 计划总步数变了（checkpoint {ckpt["meta"].get("steps")} → 本次 {args.steps}）：'
              'lr 调度形状与原计划不同，不要拿它做「续训等价性」结论')

    log_path = os.path.join(out_dir, 'train.log')
    log_f = open(log_path, 'a', encoding='utf-8')

    def log(s):
        print(s)
        log_f.write(s + '\n')
        log_f.flush()

    log(f'--- train.py run_id={run_id} resume={bool(args.resume)} ---')
    log(f'bucket={bucket} corpus={meta.get("corpus")} chars={len(text)} vocab={len(itos)} '
        f'vocab_source={meta.get("vocab_source")}')
    log(f'cfg={json.dumps(cfg, ensure_ascii=False)}')
    log(f'seed={seed} start_step={start_step} stop_at={stop_at} plan_steps={args.steps} '
        f'batch={args.batch} lr={args.lr}')

    # ---------------- 训练 ----------------
    ids = encode(text, stoi)
    train_ids = ids
    val_ids = encode(val_text, stoi) if val_text else None
    test_ids = encode(test_text, stoi) if test_text else None

    # 批量采样直接使用 python 全局 RNG：set_global_seed 已把它设成 seed，
    # 而 rng_state() 存的也正是它。若这里另起一个 Random(seed) 实例，
    # 存下来的全局状态与真正消费的那条流就不是同一条，续训必然不等价（踩过）。
    import random as _random
    rng = _random
    # 续训的 RNG 恢复已在载入分支里由 repro.set_rng_state 完成（在建 optimizer 之前）
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.resume and ckpt.get('opt'):
        opt.load_state_dict(ckpt['opt'])
    if args.resume:
        # 续训时优化器动量已恢复；Adam 的 step 计数随 state_dict 一起回来
        pass
    warmup = max(10, math.floor(args.steps * 0.15))

    model.train()
    seg_loss, seg_n = 0.0, 0
    t0 = time.time()
    for step in range(start_step, stop_at):
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

        if (args.snapshot_every and (step + 1) % args.snapshot_every == 0) or step == stop_at - 1:
            save_checkpoint(os.path.join(out_dir, 'checkpoint.pt'), model, opt, itos, meta, step + 1)
            msg = (f'  step {step + 1}/{stop_at}  loss={seg_loss / max(1, seg_n):.4f}  '
                   f'elapsed={time.time() - t0:.1f}s')
            if val_ids is not None:
                v = quick_eval(model, val_ids[:args.val_chars] if args.val_chars else val_ids, cfg['ctx'])
                if v:
                    meta['val'] = v
                    msg += f'  val: top1={v["top1"]:.2f}% nll={v["nll"]:.3f}'
            log(msg)
            seg_loss, seg_n = 0.0, 0

    # ---------------- 评测（val 选模型 / test 只报一次） ----------------
    # 只有跑到计划终点才报 test：半程存盘的阶段不该“提前看答案”
    if stop_at < args.steps:
        log(f'（阶段存盘 step {stop_at} < 计划 {args.steps}：本阶段不评 test）')
    elif test_ids is not None:
        vt = args.test_chars
        ev = eval_ids(model.to('cpu'), test_ids[:vt] if vt else test_ids)
        log(f'Holdout test（按作品切分, 词表仅 train, {ev["n"]} 个位置）: '
            f'{json.dumps(ev, ensure_ascii=False)}')
        meta['test'] = ev
    elif val_text is None:
        # 旧口径：整桶连续 90/10（JS 对齐路径）
        from minilm_torch import holdout_eval
        ev = holdout_eval(model.to('cpu'), text, seed=seed)
        log(f'Holdout（连续 90/10, JS 对齐口径）: {json.dumps(ev, ensure_ascii=False)}')
        meta['test'] = ev

    meta['train_chars'] = len(train_ids)
    meta['stop_at'] = stop_at
    meta['eff_epochs'] = stop_at * args.batch * cfg['ctx'] / max(1, len(train_ids))
    if args.bucket:
        # 数据身份 + 配置指纹：没有它，跑在「有同源泄漏的清单」和「已合并清单」上的
        # 结果在仓库里长得一模一样，事后无法区分（见 py/run_identity.py）。
        # 算不出来就别存盘 —— 直接让它抛，这次 run 不该被当成可追溯的。
        run_identity.attach(meta, args.manifest, args.report)
        di, ds = meta['data_identity'], meta['distributable_share']
        log(f'run_identity={meta["run_identity"]} manifest={di["manifest"]}'
            f' split_fp={di["split_fingerprint"]} near_dup_merged={di["near_dup_merged"]}'
            f' 可分发占比=部{ds["works_pct"]}%/字{ds["chars_pct"]}%')
    save_checkpoint(os.path.join(out_dir, 'checkpoint.pt'), model, opt, itos, meta, stop_at)
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f'Checkpoint v{CKPT_VERSION} saved to {os.path.join(out_dir, "checkpoint.pt")}')
    log_f.close()


if __name__ == '__main__':
    main()
