#!/usr/bin/env bash
# 远端：训练 + 计费安全阀。在**本地**触发，参数原样传给 py/train.py：
#   bash dev/autodl/50_train.sh --bucket story --steps 20000 --seed 1234 \
#        --out py/runs/gpu30m_story_s1234
# 安全阀：AUTODL_MAX_HOURS（默认 instance.local.sh 里）到点强制收尾；
#         训练走 checkpoint v2，中断后可用 --resume 续（先 60_fetch 再 50 --resume 亦可）。
# 默认**不**自动关机——交给 60_fetch_results + 你手点「关机」。如要自动关机：
#   AUTODL_AUTO_SHUTDOWN=1 bash dev/autodl/50_train.sh ...
set -euo pipefail
cd "$(dirname "$0")/../.."
source dev/autodl/instance.local.sh

ssh $AUTODL_SSH "bash -s" -- "$AUTODL_DIR" "$AUTODL_MAX_HOURS" "${AUTODL_AUTO_SHUTDOWN:-0}" "$@" <<'REMOTE'
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
AUTODL_DIR="$1"; shift
AUTODL_MAX_HOURS="$1"; shift
AUTODL_AUTO_SHUTDOWN="$1"; shift
cd "$AUTODL_DIR"
MAX_SEC=$(( AUTODL_MAX_HOURS * 3600 ))
echo "[*] 训练开始（硬上限 ${AUTODL_MAX_HOURS}h）"
# timeout 到点发 SIGTERM：train.py 的 checkpoint 每 N 步保存（--snapshot_every），
# 最后一次保存的 checkpoint 就是续训起点——宁可少训，不可烧钱空转。
timeout --signal=TERM --kill-after=60 "$MAX_SEC" \
  python -X utf8 py/train.py "$@" || rc=$?
rc=${rc:-0}
if [ "$rc" -eq 124 ]; then
  echo "[!] 触达 AUTODL_MAX_HOURS 上限，已收尾（checkpoint 保留，可用 --resume 续训）"
fi
echo "[*] 训练结束（rc=$rc）"
if [ "$AUTODL_AUTO_SHUTDOWN" = "1" ]; then
  echo "[*] 60 秒后自动关机"
  sleep 60
  shutdown
else
  echo "[*] 不自动关机——请先跑 60_fetch_results.sh 回传结果，再去控制台手点「关机」"
fi
REMOTE
