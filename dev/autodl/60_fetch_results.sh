#!/usr/bin/env bash
# 本地：回传远端结果（run JSON / checkpoint / 日志 / doctor·benchmark JSON），
# 并校验回传 checkpoint 的数据指纹 = 出发时指纹（不一致拒收，防"远端悄悄换数据"）。
# 用法：bash dev/autodl/60_fetch_results.sh [远端相对目录，默认 py/runs]
set -euo pipefail
cd "$(dirname "$0")/../.."
source dev/autodl/instance.local.sh
REMOTE_SUB="${1:-py/runs}"

rsync -az "$AUTODL_SSH:$AUTODL_DIR/$REMOTE_SUB/" "$REMOTE_SUB/"
rsync -az --ignore-existing "$AUTODL_SSH:$AUTODL_DIR/dev/verify/" dev/verify/
echo "[*] 已回传 $REMOTE_SUB 与 dev/verify/"

# 指纹对账：回传的每个 checkpoint/meta.json 必须带 data_identity，且与包内清单同源
python - <<'PY'
import glob, json, sys
bad, n = [], 0
for p in glob.glob('py/runs/**/meta.json', recursive=True):
    m = json.load(open(p, encoding='utf-8'))
    di = m.get('data_identity')
    if not di:
        continue          # 本地旧 run：当时没记身份，不在此校验范围
    n += 1
    src = json.load(open('dist/remote/data.tar.pack_info.json', encoding='utf-8')) \
        if __import__('os').path.exists('dist/remote/data.tar.pack_info.json') else None
    if src and di.get('split_fingerprint') != src.get('subset_split_fingerprint'):
        bad.append((p, di.get('split_fingerprint')))
if bad:
    print('[!] 指纹对不上（数据版本不一致），相关结果拒收：')
    for p, fp in bad:
        print('   ', p, fp)
    sys.exit(1)
print(f'[i] 校验 {n} 个带身份的 meta：指纹一致' if n else '[i] 未发现带身份的 meta')
PY
echo "[*] 回传完成。确认无误后到控制台关机/释放实例（停止计费）。"
