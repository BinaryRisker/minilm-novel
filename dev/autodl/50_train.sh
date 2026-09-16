#!/usr/bin/env bash
# 远端：训练 + 计费安全阀 + 跑完自动关机。在**本地**触发，参数原样传给 py/train.py：
#   bash dev/autodl/50_train.sh --bucket story --steps 20000 --seed 1234 \
#        --out py/runs/gpu30m_story_s1234
# 安全阀：AUTODL_MAX_HOURS（默认 instance.local.sh 里）到点强制收尾并关机；
#         训练走 checkpoint v2，中断后可用 --resume 续（先 60_fetch 再 50 --resume 亦可）。
set -euo pipefail
cd "$(dirname "$0")/../.."
source dev/autodl/instance.local.sh

ssh $AUTODL_SSH "bash -s" -- "$@" <<'REMOTE'
set -euo pipefail
cd "$AUTODL_DIR"
MAX_SEC=$(( AUTODL_MAX_HOURS * 3600 ))
echo "[*] 训练开始（硬上限 ${AUTODL_MAX_HOURS}h；结束/超限后自动关机）"
# timeout 到点发 SIGTERM：train.py 的 checkpoint 每 N 步保存（--snapshot_every），
# 最后一次保存的 checkpoint 就是续训起点——宁可少训，不可烧钱空转。
timeout --signal=TERM --kill-after=60 "$MAX_SEC" \
  python -X utf8 py/train.py "$@" || rc=$?
rc=${rc:-0}
if [ "$rc" -eq 124 ]; then
  echo "[!] 触达 AUTODL_MAX_HOURS 上限，已收尾（checkpoint 保留，可用 --resume 续训）"
fi
echo "[*] 训练结束（rc=$rc），60 秒后关机。先回传结果再关机的话，现在 Ctrl+C 并跑 60_fetch_results.sh"
sleep 60
shutdown
REMOTE
