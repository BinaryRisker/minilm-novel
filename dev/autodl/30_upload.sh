#!/usr/bin/env bash
# 本地：打包 + 上传（代码 + 数据包）。数据打包走 py/data/pack_remote.py（含许可闸门）。
# 用法：bash dev/autodl/30_upload.sh "gudian,wuxia,urban,xiandai,scifi,story"
set -euo pipefail
cd "$(dirname "$0")/../.."
source dev/autodl/instance.local.sh
BUCKETS="${1:-gudian,wuxia,urban,xiandai,scifi,story}"

# 解析 host / port（兼容 Git Bash：scp 需 -P 大写；rsync / ssh 走 -p 小写）
AUTODL_HOST=$(echo "$AUTODL_SSH" | sed -n 's/.*@//p')
AUTODL_PORT=$(echo "$AUTODL_SSH" | sed -n 's/.*-p[[:space:]]*\([0-9][0-9]*\).*/\1/p')
: "${AUTODL_PORT:=22}"

# 1) 数据包（许可闸门在 pack_remote.py 里：local_only/unknown 直接拒绝）
python py/data/pack_remote.py --buckets "$BUCKETS" --out dist/remote/data.tar

# 2) 代码（tar + ssh 解压；Git Bash 没 rsync，scp 不支持 exclude）
tar --exclude='.git' --exclude='__pycache__' --exclude='node_modules' \
    --exclude='data/raw' --exclude='data/processed' --exclude='data/splits' \
    --exclude='data/manifest*.jsonl' --exclude='data/sources.json' \
    --exclude='data/corpus.orig.json' \
    --exclude='py/runs' --exclude='runs' --exclude='dist' \
    --exclude='dev/autodl/instance.local.sh' \
    -czf - ./ | ssh $AUTODL_SSH "tar -xzf - -C '$AUTODL_DIR'"

# 3) 数据包解到工作目录根（包内布局镜像 data/ 树，一次落位；远端 verify 自洽）
scp -P "$AUTODL_PORT" -q "dist/remote/data.tar" "$AUTODL_SSH:$AUTODL_DIR/data_pack.tar"
ssh $AUTODL_SSH "cd '$AUTODL_DIR' && tar -xf data_pack.tar && rm data_pack.tar && ls data data/processed"
echo "[*] 上传完成。下一步：bash dev/autodl/40_benchmark.sh"

