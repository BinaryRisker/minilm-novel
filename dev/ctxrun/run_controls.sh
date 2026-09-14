#!/usr/bin/env bash
# 机制对照 4 臂：分离「上下文长度」与「更新次数 / lr 预算」
# 全部固定 sizes=200000、3 个 A 级桶、3 个种子、batch=32、cfg d40/L2/H4
# 与主对照（base=ctx32/3ep=586 步；ctx64/3ep=293 步）拼成归因表。
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

# 臂1：ctx32 + 293 步（1.5 epoch）—— 与 ctx64/3ep 更新次数相同、token 减半
run_arm ctx32e15 --ctx 32 --epochs 1.5 --min_steps 200

# 臂2：ctx128 + 146 步（3 epoch）—— 上下文再翻倍，更新次数再减半
run_arm ctx128 --ctx 128 --epochs 3 --min_steps 50

# 臂3：ctx64 + lr×2（3 epoch）—— 同 token、同更新次数，只把每步步长加倍
run_arm ctx64lr8 --ctx 64 --epochs 3 --min_steps 100 --lr 0.008

# 臂4：ctx64 + 6 epoch —— 586 步，与 ctx32/3ep 更新次数相同、token 加倍
run_arm ctx64e6 --ctx 64 --epochs 6 --min_steps 100

echo ALL_CONTROLS_DONE
for t in ctx32e15 ctx128 ctx64lr8 ctx64e6; do
  $PY py/sweep_worksplit.py --report_only --tag "$t" 2>&1 | tail -2
done
