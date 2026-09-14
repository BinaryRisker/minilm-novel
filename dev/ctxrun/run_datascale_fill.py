#!/usr/bin/env python3
"""dev/ctxrun/run_datascale_fill.py — 补跑 2M 臂缺失种子（42,2024 × 3 桶）。
arm1 首轮在 seed=1234 后三进程同时静默退出（rc=1、无 traceback，疑似系统级外因），
arm2 完整。本脚本只补 2M 的 6 条。"""
import os
import subprocess

ROOT = r"D:\work\outputs\minilm-lab"
PY = r"C:\Python314\python.exe"
LOGS = os.path.join(ROOT, "dev", "ctxrun")
CFG = ["--cfg_d", "40", "--cfg_layers", "2", "--cfg_heads", "4"]

procs = []
for bk in ("gudian", "wuxia", "urban"):
    log = open(os.path.join(LOGS, f"dsfill_{bk}_2000000.log"), "w", encoding="utf-8")
    cmd = [PY, "py/sweep_worksplit.py", "--buckets", bk, "--sizes", "2000000",
           "--seeds", "42,2024", "--tag", "ctx64", "--epochs", "3",
           "--ctx", "64", "--min_steps", "100",
           "--no_report", "--no_plot", "--no_alpha_probe"] + CFG
    print(f"[launch] {bk} 2000000 seeds=42,2024", flush=True)
    procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
for p in procs:
    p.wait()
    print(f"[done] rc={p.returncode}", flush=True)
print("FILL_DONE", flush=True)
