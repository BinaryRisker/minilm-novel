""" ============================================================================
py/verify_resume.py — 断点续训等价性验证（含三组失败样本）
----------------------------------------------------------------------------
对比矩阵（同一个 seed / 同一份语料 / 同一计划步数）：
  A  直训 N 步（基准）
  B  训 N/2 → 存盘 → 从 checkpoint 续训到 N        ← 期望与 A 逐位相同
  C1 同 B，但 checkpoint 里**删掉 RNG 状态**（只留权重+优化器）→ 采样顺序不同
  C2 同 B，但 checkpoint 里**删掉优化器状态**（只留权重+RNG）→ Adam 动量从零重启
  C3 同 B，但**两者都删**（= 修复前 train.py 的旧格式 checkpoint）

C1/C2/C3 是失败样本：修复前的脚本只能存「权重 + itos + meta」，也就是 C3 那一行。
判据：
  末步权重 max|Δ|（对 A）、固定窗口探测 loss 偏差（对 A）。
  探测 loss 用固定的一批窗口在 test 切片上算 CE，确定性、对参数差异敏感。

用法：
  python py/verify_resume.py
  python py/verify_resume.py --steps 160 --chars 24000 --out dev/verify/verify_resume.json
============================================================================ """
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import splits as splits_mod  # noqa
from minilm_torch import MiniLM, encode  # noqa


def run_train(args: list, tag: str) -> float:
    t0 = time.time()
    cmd = [PY, os.path.join(HERE, 'train.py')] + args
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if p.returncode != 0:
        print(p.stdout[-3000:]); print(p.stderr[-3000:])
        raise SystemExit(f'训练失败：{tag}')
    dt = time.time() - t0
    tail = [l for l in p.stdout.strip().splitlines() if l.strip()][-1:]
    print(f'  [{tag}] {dt:.1f}s  {tail[0][:110] if tail else ""}')
    return dt


def load_ckpt(path: str):
    return torch.load(path, map_location='cpu', weights_only=False)


def param_diff(a: dict, b: dict) -> float:
    m = 0.0
    for k in a['model']:
        m = max(m, (a['model'][k].float() - b['model'][k].float()).abs().max().item())
    return m


@torch.no_grad()
def probe_loss(ckpt: dict, text: str, n_win: int = 64, ctx: int = 32) -> float:
    """固定窗口上的平均 CE（确定性，不用任何 RNG）。"""
    itos = ckpt['itos']
    stoi = {ch: i for i, ch in enumerate(itos)}
    ids = encode(text, stoi)
    cfg = ckpt['meta']['cfg']
    model = MiniLM(len(itos), cfg)
    model.load_state_dict(ckpt['model'])
    model.eval()
    stride = max(1, (len(ids) - ctx - 1) // n_win)
    tot, n = 0.0, 0
    for i in range(0, max(1, len(ids) - ctx - 1), stride):
        if n >= n_win:
            break
        chunk = torch.tensor([ids[i:i + ctx]], dtype=torch.long)
        tgt = torch.tensor([ids[i + 1:i + 1 + ctx]], dtype=torch.long)
        _, loss = model(chunk, tgt)
        tot += float(loss.item()); n += 1
    return tot / max(1, n)


def clean_dir(path: str):
    """占位：本环境有 safe-delete 钩子（会异步 trash 且可能误删），
    因此本脚本改为**每次用唯一目录**，不删除任何历史结果。"""
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bucket', default='story')
    ap.add_argument('--chars', type=int, default=24000)
    ap.add_argument('--steps', type=int, default=160)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--out', default=os.path.join(ROOT, 'dev', 'verify', 'verify_resume.json'))
    args = ap.parse_args()

    # 唯一目录：本环境的 safe-delete 钩子会拦截删除（且可能异步误删新文件），
    # 所以不复用固定目录名、不删除任何东西。
    stamp = time.strftime('%m%d_%H%M%S')
    work_name = f'resume_{stamp}'
    work = os.path.join(ROOT, 'dev', 'loop', work_name)
    os.makedirs(work, exist_ok=True)
    rel = f'dev/loop/{work_name}'
    half = args.steps // 2
    base = ['--bucket', args.bucket, '--chars', str(args.chars), '--batch', str(args.batch),
            '--seed', str(args.seed)]
    print(f'[*] 语料 {args.bucket} 前 {args.chars} 字 / 计划 {args.steps} 步 / 中途点 {half}')

    print('[*] 运行 A（直训）')
    tA = run_train(base + ['--steps', str(args.steps), '--out', f'{rel}/A'], 'A 直训')

    print('[*] 运行 B（训一半 → 续训，完整状态）')
    tB1 = run_train(base + ['--steps', str(args.steps), '--stop_at', str(half),
                            '--out', f'{rel}/B'], 'B 前半')
    ck_half = os.path.join(work, 'B', 'checkpoint.pt')
    # 必须在续训**之前**备份中途 checkpoint：续训会把 B/checkpoint.pt 覆盖成末步版本，
    # 若 C 组从被覆盖后的文件出发，就会「从终点续训」而变成空操作（踩过）。
    half_backup = os.path.join(work, '_half.pt')
    shutil.copyfile(ck_half, half_backup)
    tB2 = run_train(['--resume', ck_half, '--steps', str(args.steps)], 'B 续训')

    # 三组失败样本：从同一个中途 checkpoint 出发，按不同「丢失程度」续训
    variants = {'C1_丢RNG': ('rng',), 'C2_丢优化器': ('opt',), 'C3_旧格式(全丢)': ('rng', 'opt')}
    times = {'A': tA, 'B': tB1 + tB2}
    for name, drop in variants.items():
        d = os.path.join(work, name)
        os.makedirs(d, exist_ok=True)
        ck = load_ckpt(half_backup)
        for k in drop:
            ck.pop(k, None)
        torch.save(ck, os.path.join(d, 'checkpoint.pt'))
        times[name] = run_train(['--resume', os.path.join(d, 'checkpoint.pt'),
                                 '--steps', str(args.steps)], name)

    # ---- 对比 ----
    test_text = splits_mod.bucket_joined(args.bucket)['test'][:20000]
    A = load_ckpt(os.path.join(work, 'A', 'checkpoint.pt'))
    print('[*] 对比（探测窗口 %d 个，test 切片 %d 字）' % (64, len(test_text)))
    probe_A = probe_loss(A, test_text)
    rows = []
    for tag, path in [('B_完整状态', os.path.join(work, 'B', 'checkpoint.pt')),
                      ('C1_丢RNG', os.path.join(work, 'C1_丢RNG', 'checkpoint.pt')),
                      ('C2_丢优化器', os.path.join(work, 'C2_丢优化器', 'checkpoint.pt')),
                      ('C3_旧格式(全丢)', os.path.join(work, 'C3_旧格式(全丢)', 'checkpoint.pt'))]:
        ck = load_ckpt(path)
        pr = probe_loss(ck, test_text)
        rows.append({
            'tag': tag,
            'step': int(ck.get('step', -1)),
            'param_max_abs_diff_vs_A': param_diff(A, ck),
            'probe_loss': pr,
            'probe_loss_delta_vs_A': pr - probe_A,
            'wallclock_sec': round(times.get(tag.split('_')[0], 0.0), 1),
        })

    rep = {
        'protocol': {'bucket': args.bucket, 'chars': args.chars, 'steps': args.steps,
                     'half': half, 'batch': args.batch, 'seed': args.seed,
                     'note': 'A=直训；B/C*=训一半后从 checkpoint 续训；'
                             'C 组按「丢失哪些状态」分解续训偏差'},
        'baseline_A': {'probe_loss': probe_A,
                       'test_top1': A['meta'].get('test', {}).get('top1'),
                       'test_nll': A['meta'].get('test', {}).get('nll'),
                       'wallclock_sec': round(tA, 1)},
        'rows': rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print('\n== 对比（基准 A 探测 loss=%.4f）==' % probe_A)
    print(f'{"变体":<16}{"step":>6}{"参数max|Δ|":>14}{"探测loss":>11}{"Δloss":>10}')
    for r in rows:
        print(f'{r["tag"]:<16}{r["step"]:>6}{r["param_max_abs_diff_vs_A"]:>14.3e}'
              f'{r["probe_loss"]:>11.4f}{r["probe_loss_delta_vs_A"]:>+10.4f}')
    print(f'[*] 报告 -> {args.out}')

    b = rows[0]
    ok = (b['param_max_abs_diff_vs_A'] == 0.0 and abs(b['probe_loss_delta_vs_A']) < 1e-12
          and all(r['param_max_abs_diff_vs_A'] > 0 for r in rows[1:]))
    print(f'[*] 判定：{"通过（完整状态续训与直训逐位一致；缺状态必然偏离）" if ok else "★失败★"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
