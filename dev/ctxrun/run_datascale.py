#!/usr/bin/env python3
"""dev/ctxrun/run_datascale.py — 数据量扩展启动器（绕开 bash shim，纯 Python 进程管理）。

arm1: 2M × {gudian, wuxia, urban}
arm2: 5M × {gudian, wuxia} + 4M × urban（urban 池 423 万 < 5M，不请求 5M）
tag=ctx64 并入主网格；3 epoch token 对齐；ctx64 → 2930/5860/7325 步。
"""
import os
import subprocess
import sys

ROOT = r"D:\work\outputs\minilm-lab"
PY = r"C:\Python314\python.exe"
LOGS = os.path.join(ROOT, "dev", "ctxrun")
os.makedirs(LOGS, exist_ok=True)

CFG = ["--cfg_d", "40", "--cfg_layers", "2", "--cfg_heads", "4"]


def run_one(bucket, size, epochs):
    log = open(os.path.join(LOGS, f"ds_{bucket}_{size}.log"), "w", encoding="utf-8")
    cmd = [PY, "py/sweep_worksplit.py", "--buckets", bucket, "--sizes", str(size),
           "--seeds", "1234,42,2024", "--tag", "ctx64", "--epochs", str(epochs),
           "--ctx", "64", "--min_steps", "100",
           "--no_report", "--no_plot", "--no_alpha_probe"] + CFG
    print(f"[launch] {bucket} {size} ep={epochs}", flush=True)
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def run_arm(jobs):
    procs = [run_one(*j) for j in jobs]
    for p in procs:
        p.wait()
        print(f"[done] rc={p.returncode}", flush=True)


print("=== arm1: 2M x 3 buckets ===", flush=True)
run_arm([("gudian", 2000000, 3), ("wuxia", 2000000, 3), ("urban", 2000000, 3)])
print("=== arm2: 5M x {gudian,wuxia} + 4M x urban ===", flush=True)
run_arm([("gudian", 5000000, 3), ("wuxia", 5000000, 3), ("urban", 4000000, 3)])
print("ALL_DATASCALE_DONE", flush=True)
