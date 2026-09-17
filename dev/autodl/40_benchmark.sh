#!/usr/bin/env bash
# 远端：doctor + benchmark 组合扫描 → budget_table.json。在**本地**触发：
#   bash dev/autodl/40_benchmark.sh [额外传给 benchmark 的参数]
# 例：bash dev/autodl/40_benchmark.sh --cfg '{"d":512,"layers":6,"heads":8}' --steps 200
set -euo pipefail
cd "$(dirname "$0")/../.."
source dev/autodl/instance.local.sh

ssh $AUTODL_SSH "bash -s" -- "$AUTODL_DIR" "$AUTODL_PRICE_PER_HOUR" "$@" <<'REMOTE'
set -euo pipefail
# AutoDL 基础镜像把 python 装在 /root/miniconda3，但不在 PATH。补上再跑。
export PATH=/root/miniconda3/bin:$PATH
AUTODL_DIR="$1"; shift
AUTODL_PRICE_PER_HOUR="$1"; shift
cd "$AUTODL_DIR"
# 把 cfg 写成文件，避免 ssh heredoc 把 "{...}" 拆 token（实测踩过）
BENCH_CFG_FILE="$AUTODL_DIR/dev/verify/_bench_cfg.json"
python -c 'import json,sys; json.dump({"d":512,"layers":8,"heads":8,"ffMul":4}, open(sys.argv[1],"w"))' "$BENCH_CFG_FILE"
echo "== doctor =="
python -X utf8 -m py.novellm.doctor --out dev/verify/doctor_remote.json
echo "== benchmark 组合扫描 =="
python -X utf8 py/benchmark.py --corpus scifi --device cuda \
  --cfg-file "$BENCH_CFG_FILE" --steps 200 \
  --sweep_ctx 32,64,128,256,512,1024 --sweep_batch 8,16,32,64 \
  --estimate_tokens 92000000 --price_per_hour "${AUTODL_PRICE_PER_HOUR:-2.0}" \
  --out dev/verify/budget_table.json "$@"
echo "== 摘要 =="
python -X utf8 - <<'PY'
import json
d = json.load(open('dev/verify/budget_table.json', encoding='utf-8'))
ok = [r for r in d['results'] if not r.get('oom')]
best = max(ok, key=lambda r: r['tokens_per_sec']) if ok else None
print(f"可行 {len(ok)}/{len(d['results'])} 组；最快 {best['tokens_per_sec']:.0f} tokens/s"
      f"（ctx={best['ctx']} batch={best['batch']}）" if best else "无可行组合")
if best and best.get('estimate'):
    e = best['estimate']
    print(f"估算：92M tokens（30M 候选 × 6 证据桶 × ~3 epoch）≈ {e['hours']}h / ¥{e['cost_yuan']}")
PY
REMOTE
echo "[*] benchmark 完成。结果在实例 dev/verify/，用 60_fetch_results.sh 回传。"
