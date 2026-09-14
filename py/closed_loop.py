""" ============================================================================
py/closed_loop.py — 随机初始化最小闭环（四类修复的端到端串演）
----------------------------------------------------------------------------
把「种子 → 训练 → 中途存盘 → 断点续训 → 留出评测 → 生成故事」串成一条可复核的链路，
每一步的产物都落在同一个时间戳目录里，任何人可以照报告重跑：

  stage 1  随机初始化 + 训练到中途（--stop_at），存 checkpoint v2（含优化器/RNG/步数）
  stage 2  从 checkpoint 续训到计划终点（验证续训在真实流程里可用）
  stage 3  留出评测：val 看模型有没有在学（早停/选点用），test 只在最后评一次
  stage 4  故事生成评测：提示词取自**留出作品**，模型 vs trigram 基线，同一 seed

设计取巧之处（有意为之，写进报告）：
  · 全程用 --bucket story + 按作品切分的留出集（train 出词表，val 选点，test 只报一次）；
  · 语料用 --chars 截断到能在几分钟内跑完的规模 —— 这是「最小闭环」，不是刷分实验；
  · 目录用时间戳，不删任何历史结果（本环境的 safe-delete 钩子会拦截删除）。

用法：
  python py/closed_loop.py --bucket story --chars 60000 --steps 600 --stop_at 300
============================================================================ """
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import splits as S  # noqa


def run(cmd: list, tag: str, log_path: str) -> float:
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    dt = time.time() - t0
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f'===== {tag} (rc={p.returncode}, {dt:.1f}s) =====\n')
        f.write('$ ' + ' '.join(cmd) + '\n')
        f.write(p.stdout + '\n' + p.stderr + '\n')
    if p.returncode != 0:
        print(f'  [{tag}] ★失败 rc={p.returncode}（日志见 {log_path}）')
        print(p.stdout[-1500:])
        print(p.stderr[-1500:])
        raise SystemExit(1)
    print(f'  [{tag}] ok {dt:.1f}s')
    return dt


def tail_log(path: str, n: int = 3) -> list:
    if not os.path.exists(path):
        return []
    lines = [l for l in open(path, encoding='utf-8', errors='replace').read().splitlines() if l.strip()]
    return lines[-n:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bucket', default='story')
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--report', default=None)
    ap.add_argument('--chars', type=int, default=60000)
    ap.add_argument('--steps', type=int, default=600, help='计划总步数')
    ap.add_argument('--stop_at', type=int, default=300, help='第一阶段停在第几步')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.004)
    ap.add_argument('--n_prompts', type=int, default=6)
    ap.add_argument('--max_new', type=int, default=120)
    args = ap.parse_args()

    stamp = time.strftime('%m%d_%H%M%S')
    work_rel = f'dev/loop/closed_{stamp}'
    work = os.path.join(ROOT, work_rel)
    os.makedirs(work, exist_ok=True)
    log_path = os.path.join(work, 'commands.log')
    print(f'[*] 闭环目录 {work_rel}')

    mf = ['--manifest', args.manifest] if args.manifest else []
    rp = ['--report', args.report] if args.report else []
    base = ['--bucket', args.bucket, '--chars', str(args.chars), '--batch', str(args.batch),
            '--lr', str(args.lr), '--seed', str(args.seed)] + mf + rp

    summary = {'protocol': {'bucket': args.bucket, 'chars': args.chars, 'steps': args.steps,
                            'stop_at': args.stop_at, 'seed': args.seed, 'batch': args.batch,
                            'lr': args.lr, 'manifest': args.manifest, 'work_dir': work_rel},
               'stages': {}}

    # ---- stage 1：随机初始化 + 训练到中途 ----
    print('[*] stage 1 随机初始化 + 训练到中途')
    t1 = run([PY, 'py/train.py'] + base + ['--steps', str(args.steps), '--stop_at', str(args.stop_at),
                                           '--out', f'{work_rel}/run'],
             'stage1 训练到中途', log_path)
    ck = os.path.join(work, 'run', 'checkpoint.pt')
    ck1 = json.load(open(os.path.join(work, 'run', 'checkpoint.pt'), 'rb').read()[:0] or '{}') if False else None
    import torch
    c1 = torch.load(ck, map_location='cpu', weights_only=False)
    summary['stages']['stage1'] = {
        'seconds': round(t1, 1), 'step': c1['step'], 'ckpt_version': c1.get('version'),
        'has_opt': 'opt' in c1, 'has_rng': 'rng' in c1,
        'meta_keys': sorted(k for k in c1['meta'] if k in
                            ('bucket', 'chars', 'seed', 'steps', 'vocab_source', 'leaky_protocol')),
        'vocab_source': c1['meta'].get('vocab_source'),
        'train_chars': c1['meta'].get('train_chars'),
        'val': c1['meta'].get('val'),
        'log_tail': tail_log(os.path.join(work, 'run', 'train.log')),
    }

    # ---- stage 2：断点续训到终点 ----
    print('[*] stage 2 断点续训到终点')
    t2 = run([PY, 'py/train.py', '--resume', ck, '--steps', str(args.steps)],
             'stage2 续训到终点', log_path)
    c2 = torch.load(ck, map_location='cpu', weights_only=False)
    summary['stages']['stage2'] = {
        'seconds': round(t2, 1), 'step': c2['step'], 'val': c2['meta'].get('val'),
        'test': c2['meta'].get('test'), 'eff_epochs': c2['meta'].get('eff_epochs'),
        'log_tail': tail_log(os.path.join(work, 'run', 'train.log')),
    }

    # ---- stage 3：留出评测（val 看趋势 / test 只报一次）----
    print('[*] stage 3 留出评测')
    summary['stages']['stage3'] = {'val': c2['meta'].get('val'), 'test': c2['meta'].get('test')}

    # ---- stage 4：故事生成评测（留出作品出题 + trigram 基线）----
    print('[*] stage 4 故事生成评测')
    out_eval = f'{work_rel}/story_eval.json'
    t4 = run([PY, 'py/eval_story.py', '--checkpoint', ck, '--bucket', args.bucket,
              '--n_prompts', str(args.n_prompts), '--max_new', str(args.max_new),
              '--seed', str(args.seed), '--out', out_eval] + mf + rp,
             'stage4 故事评测', log_path)
    ev = json.load(open(os.path.join(ROOT, out_eval), encoding='utf-8'))
    summary['stages']['stage4'] = {'seconds': round(t4, 1), 'summary': ev['summary'],
                                   'prompt_source': ev['meta']['prompt_source'],
                                   'samples': [{'prompt': r['prompt'], 'model': r['model']['text'],
                                                'trigram': r.get('baseline_trigram', {}).get('text')}
                                               for r in ev['results'][:3]]}

    # ---- 报告 ----
    with open(os.path.join(work, 'loop_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    L = [f'# 随机初始化最小闭环报告（{stamp}）', '',
         f'- 桶：`{args.bucket}`，语料截断 {args.chars:,} 字，计划 {args.steps} 步，'
         f'中途点 {args.stop_at} 步，seed={args.seed}，batch={args.batch}，lr={args.lr}',
         f'- 词表来源：`{summary["stages"]["stage1"]["vocab_source"]}`'
         f'（train 上建，val/test 未参与）',
         f'- checkpoint 版本：v{summary["stages"]["stage1"]["ckpt_version"]}，'
         f'含 optimizer={summary["stages"]["stage1"]["has_opt"]}，RNG={summary["stages"]["stage1"]["has_rng"]}',
         '', '## stage 1 训练到中途', '',
         f'- step={summary["stages"]["stage1"]["step"]}，val={summary["stages"]["stage1"]["val"]}',
         f'- 耗时 {summary["stages"]["stage1"]["seconds"]}s', '',
         '## stage 2 断点续训', '',
         f'- 续训到 step={summary["stages"]["stage2"]["step"]}，'
         f'val={summary["stages"]["stage2"]["val"]}',
         f'- 有效 epoch={summary["stages"]["stage2"]["eff_epochs"]:.2f}，'
         f'耗时 {summary["stages"]["stage2"]["seconds"]}s', '',
         '## stage 3 留出评测（test 只在最后评一次）', '',
         f'- val：{summary["stages"]["stage3"]["val"]}',
         f'- test：`{json.dumps(summary["stages"]["stage3"]["test"], ensure_ascii=False)}`', '',
         '## stage 4 故事生成评测（指标只在生成段上算）', '']
    for k, v in summary['stages']['stage4']['summary'].items():
        if isinstance(v, dict):
            L.append(f'- {k}: ' + ' '.join(f'{kk}={vv}' for kk, vv in v.items()))
        else:
            L.append(f'- {k}: {v}')
    L += ['', '### 样本（前 3 条，提示词来自留出作品）', '']
    for s in summary['stages']['stage4']['samples']:
        L += [f'**提示词**：{s["prompt"]}', '', f'**模型**：{s["model"]}', '',
              f'**trigram 基线**：{s["trigram"]}', '']
    md = os.path.join(work, 'loop_report.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))

    print(f'\n[*] 报告 -> {os.path.relpath(md, ROOT)}')
    print(f'[*] 汇总 -> {os.path.relpath(os.path.join(work, "loop_summary.json"), ROOT)}')
    print(f'[*] 命令日志 -> {os.path.relpath(log_path, ROOT)}')


if __name__ == '__main__':
    main()
