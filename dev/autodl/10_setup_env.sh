#!/usr/bin/env bash
# 远端环境准备：装缺的包 + 校验目录。在**本地**执行：bash dev/autodl/10_setup_env.sh
# 只装运行必需的小件（psutil 给 doctor 的内存项）；torch 用实例自带镜像，不重装。
set -euo pipefail
cd "$(dirname "$0")/../.."
source dev/autodl/instance.local.sh

ssh $AUTODL_SSH "bash -s" <<REMOTE
set -euo pipefail
mkdir -p "$AUTODL_DIR"
cd "$AUTODL_DIR"
python -X utf8 -m pip install -q psutil 2>/dev/null || pip install -q psutil || echo "[!] psutil 装不上：doctor 的内存项会留空（不猜数字），其余功能不受影响"
echo "[*] 远端就绪：$AUTODL_DIR"
REMOTE
