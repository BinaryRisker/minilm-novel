#!/usr/bin/env bash
# 预算给足对照：20万字档 ctx64，步数 1172（12ep）/ 2344（24ep）
# 目的：验证 Top-1 缺口是否随优化预算单调消失（上轮 586 步已 +0.04）
# lr 调度内置（15% warmup + 余弦衰减，按各自总步数归一），无需额外参数
set -u
cd /d/work/outputs/minilm-lab || exit 1
PY=/c/Python314/python.exe
CFG="--cfg_d 40 --cfg_layers 2 --cfg_heads 4"
SZ=200000
SEEDS=1234,42,2024
mkdir -p dev/ctxrun

run_arm () {  # $1=tag  $2...=额外参数
  local tag="$1"; shift
  echo "=== arm $tag : $* ==="
  for bk in gudian wuxia urban; do
    $PY py/sweep_worksplit.py --buckets "$bk" --sizes $SZ --seeds $SEEDS \
        --tag "$tag" --no_report --no_plot --no_alpha_probe $CFG "$@" \
        > "dev/ctxrun/${tag}_${bk}.log" 2>&1 &
  done
  wait
}

# 臂A：ctx64 + 12 epoch = 1172 步（上轮 586 步的 2 倍）
run_arm ctx64e12 --ctx 64 --epochs 12 --min_steps 100

# 臂B：ctx64 + 24 epoch = 2344 步（4 倍；同时观察 200k 字档是否过拟合）
run_arm ctx64e24 --ctx 64 --epochs 24 --min_steps 100

echo ALL_BUDGET_DONE
