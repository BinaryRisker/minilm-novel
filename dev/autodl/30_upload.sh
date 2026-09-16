#!/usr/bin/env bash
# 本地：打包 + 上传（代码 + 数据包）。数据打包走 py/data/pack_remote.py（含许可闸门）。
# 用法：bash dev/autodl/30_upload.sh "gudian,wuxia,urban,xiandai,scifi,story"
set -euo pipefail
cd "$(dirname "$0")/../.."
source dev/autodl/instance.local.sh
BUCKETS="${1:-gudian,wuxia,urban,xiandai,scifi,story}"

# 1) 数据包（许可闸门在 pack_remote.py 里：local_only/unknown 直接拒绝）
python py/data/pack_remote.py --buckets "$BUCKETS" --out dist/remote/data.tar

# 2) 代码（rsync 排除数据/产物/本地私件；不用 git clone——凭据不上租用机）
rsync -az --delete \
  --exclude '.git' --exclude 'node_modules' --exclude '__pycache__' \
  --exclude 'data/raw' --exclude 'data/processed' --exclude 'data/splits' \
  --exclude 'data/manifest*.jsonl' --exclude 'data/sources.json' --exclude 'data/corpus*' \
  --exclude 'py/runs' --exclude 'runs' --exclude 'dist' --exclude 'dev/autodl/instance.local.sh' \
  ./ "$AUTODL_SSH:$AUTODL_DIR/"

# 3) 数据包解到工作目录根（包内布局镜像 data/ 树，一次落位；远端 verify 自洽）
scp -q "dist/remote/data.tar" "$AUTODL_SSH:$AUTODL_DIR/data_pack.tar"
ssh $AUTODL_SSH "cd '$AUTODL_DIR' && tar -xf data_pack.tar && rm data_pack.tar && ls data data/processed"
echo "[*] 上传完成。下一步：bash dev/autodl/40_benchmark.sh"
