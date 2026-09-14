"""dev/verify/run_leak_seeds.py — 用 3 个种子量化「同源泄漏」的效应量。

产出 dev/verify/leak_probe_3seeds.json：三种子下「污染作品组」与「干净作品组」的
Top-1/NLL 均值、差值及其离散度（用来判断泄漏效应是否淹没在种子波动里）。
"""
import json
import os
import statistics
import subprocess
import sys

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEDS = (1234, 42, 2024)
res = []
for seed in SEEDS:
    out = f'dev/verify/leak_probe_s{seed}.json'
    cmd = [PY, 'py/leak_probe.py', '--bucket', 'gudian', '--chars', '100000',
           '--steps', '500', '--seed', str(seed), '--out', out,
           '--model-manifest', 'dev/verify/manifest_snapshot.jsonl',
           '--report', 'dev/verify/split_report_snapshot.json',
           '--eval-manifests', 'dev/verify/manifest_snapshot.jsonl,data/manifest_v2.jsonl']
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if p.returncode != 0:
        print(p.stdout[-1500:])
        print(p.stderr[-1500:])
        raise SystemExit('leak_probe 失败 seed=%s' % seed)
    g = json.load(open(os.path.join(ROOT, out), encoding='utf-8'))['group_compare']
    res.append(g)
    print('seed=%s: 污染 Top-1=%.2f  干净=%.2f  Δ=%+.2fpt  ΔNLL=%+.3f'
          % (seed, g['leaked_mean_top1'], g['clean_mean_top1'], g['top1_gap_pt'], g['nll_gap']))

lp = [r['leaked_mean_top1'] for r in res]
cp = [r['clean_mean_top1'] for r in res]
d1 = [r['top1_gap_pt'] for r in res]
dn = [r['nll_gap'] for r in res]
print()
print('污染组 Top-1 3 种子: %s  均值 %.2f ± %.2f' % (lp, statistics.mean(lp), statistics.pstdev(lp)))
print('干净组 Top-1 3 种子: %s  均值 %.2f ± %.2f' % (cp, statistics.mean(cp), statistics.pstdev(cp)))
print('★ ΔTop-1: %s  均值 %+.2fpt ± %.2f' % (d1, statistics.mean(d1), statistics.pstdev(d1)))
print('★ ΔNLL  : %s  均值 %+.3f ± %.3f'
      % ([round(x, 3) for x in dn], statistics.mean(dn), statistics.pstdev(dn)))
json.dump(res, open(os.path.join(ROOT, 'dev/verify/leak_probe_3seeds.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=2)
print('[*] -> dev/verify/leak_probe_3seeds.json')
