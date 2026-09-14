#!/usr/bin/env bash
# 数据量扩展：2M（3 桶）/ 5M（gudian,wuxia）/ 4M（urban，池 423 万）
# 预算口径：3 epoch（token 对齐主网格），ctx64 → 2M=2930 步、4M=5860 步、5M=7325 步
# 均远超 200k 档实测「够用」阈值 1172 步；tag=ctx64 直接并入主网格家族
# ⚠️ 不给 urban 请求 5M：池 423 万 < 5M 会被夹断成同源数据（xiandai 轮的坑）
set -u
cd /d/work/outputs/minilm-lab || exit 1
PY=/c/Python314/python.exe
CFG="--cfg_d 40 --cfg_layers 2 --cfg_heads 4"
SEEDS=1234,42,2024
mkdir -p dev/ctxrun

run_one () {  # $1=bucket  $2=size  $3=epochs
  $PY py/sweep_worksplit.py --buckets "$1" --sizes "$2" --seeds $SEEDS \
      --tag ctx64 --epochs "$3" --ctx 64 --min_steps 100 \
      --no_report --no_plot --no_alpha_probe $CFG \
      > "dev/ctxrun/ds_$1_$2.log" 2>&1 &
}

echo "=== arm1: 2M x {gudian,wuxia,urban} ==="
run_one gudian 2000000 3
run_one wuxia  2000000 3
run_one urban  2000000 3
wait

echo "=== arm2: 5M x {gudian,wuxia} + 4M x urban ==="
run_one gudian 5000000 3
run_one wuxia  5000000 3
run_one urban  4000000 3
wait

echo ALL_DATASCALE_DONE
