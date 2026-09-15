""" ============================================================================
py/run_identity.py — 一次 run 的身份 = 配置指纹 + 数据指纹 + 许可分级
----------------------------------------------------------------------------
要解决的问题（2026-09-15 实测）：
  `py/runs/*/meta.json` 只记 `corpus/cfg/steps/seed`，`splits: null`，
  **没有任何数据身份**。而现在同时存在两份清单（data/manifest.jsonl 与
  py/dedup_works.py 产的 data/manifest_v2.jsonl），同一套 bucket+seed+steps
  可以指向两份不同的数据集，结果文件里却看不出区别 —— 于是一条"漏的数据"上跑出的
  数字和"干净数据"上的数字在仓库里长得一模一样。

本模块给出可核对的身份，字段全部可回溯：
  split_fingerprint   来自 py/splits.py（与 data/split_report*.json 同源，单一算法）
  manifest            实际用的清单文件名（区分 v1 / v2）
  near_dup_merged     这份清单是否做过**近重复/同源合并**（看有没有 dedup_cluster 字段）
  license             按 py/license_policy.py 的用途分级统计（部数/字数）
  以上 + cfg/seed/steps/batch/lr 一起哈希成 `identity`

注意诚实边界：**指纹只保证可追溯，不保证数据是干净的。**
没做近重复审计的清单，指纹照样能算 —— 所以这里把 `near_dup_merged` 显式带出来，
免得"有身份"被误读成"数据没问题"。

用法：
  python py/run_identity.py --meta py/runs/xxx/meta.json      # 校验/补算一个既有 run 的身份
  python py/run_identity.py --bucket story --show             # 打印当前默认清单的身份摘要
  from run_identity import data_identity, run_identity, attach
============================================================================ """
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import splits  # noqa: E402
from license_policy import DISTRIBUTABLE, classify  # noqa: E402


def license_summary(rows: List[dict]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for r in rows:
        cls = classify(r.get('license'))
        e = out.setdefault(cls, {'works': 0, 'chars': 0})
        e['works'] += 1
        e['chars'] += r.get('chars') or 0
    return out


def data_identity(manifest_path: Optional[str] = None,
                  report_path: Optional[str] = None) -> Dict[str, object]:
    """从清单 + 切分报告算数据身份。不传路径时走 py/splits.py 的当前默认（含新鲜度判定）。"""
    man, rep = splits.resolve_current(manifest_path, report_path)
    rows = splits.read_manifest(man)
    v = splits.verify(man, rep)          # 不合规直接抛 SplitIdentityError，不静默
    return {
        'manifest': os.path.basename(man),
        'split_report': os.path.basename(rep),
        'split_fingerprint': v['fingerprint'],
        'works': len(rows),
        'chars': sum(r.get('chars') or 0 for r in rows),
        'buckets': _bucket_counts(rows),
        'near_dup_merged': bool(rows) and 'dedup_cluster' in rows[0],
        'license': license_summary(rows),
    }


def _bucket_counts(rows: List[dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        b = r.get('bucket') or 'unknown'
        out[b] = out.get(b, 0) + 1
    return out


def distributable_share(did: Dict[str, object]) -> Dict[str, float]:
    """按部数/字数给出「许可明确、可进入可分发产物」的占比（见 license_policy）。"""
    lic = did['license']  # type: ignore[index]
    tot_w = sum(v['works'] for v in lic.values()) or 1
    tot_c = sum(v['chars'] for v in lic.values()) or 1
    ok_w = sum(v['works'] for k, v in lic.items() if k in DISTRIBUTABLE)
    ok_c = sum(v['chars'] for k, v in lic.items() if k in DISTRIBUTABLE)
    return {'works_pct': round(100.0 * ok_w / tot_w, 2),
            'chars_pct': round(100.0 * ok_c / tot_c, 2),
            'works': ok_w, 'chars': ok_c}


def run_identity(cfg: dict, seed: int, steps: int, batch: int, lr: float,
                 did: Dict[str, object]) -> str:
    """配置 + 数据的稳定指纹（16 hex）。键排序后序列化，保证跨次运行一致。"""
    payload = {
        'cfg': cfg, 'seed': seed, 'steps': steps, 'batch': batch, 'lr': lr,
        'split_fingerprint': did['split_fingerprint'],
        'manifest': did['manifest'],
        'near_dup_merged': did['near_dup_merged'],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def attach(meta: dict, manifest_path: Optional[str] = None,
           report_path: Optional[str] = None) -> dict:
    """给 train/评测脚本用：把数据身份与配置指纹写进 meta（原地并返回）。"""
    did = data_identity(manifest_path, report_path)
    meta['data_identity'] = did
    meta['run_identity'] = run_identity(meta.get('cfg', {}), meta.get('seed', -1),
                                        meta.get('steps', meta.get('plan_steps', -1)),
                                        meta.get('batch', -1), meta.get('lr', -1.0), did)
    meta['distributable_share'] = distributable_share(did)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', default=None, help='要校验/补算身份的 meta.json 或 runs/*.json')
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--report', default=None)
    ap.add_argument('--show', action='store_true', help='只打印当前默认清单的数据身份')
    args = ap.parse_args()

    did = data_identity(args.manifest, args.report)
    share = distributable_share(did)
    print(f'[*] 数据身份：manifest={did["manifest"]} report={did["split_report"]}')
    print(f'    split_fingerprint={did["split_fingerprint"]}  works={did["works"]}  chars={did["chars"]}')
    merged = '已做' if did['near_dup_merged'] else '★未做★（留出集可能仍有同源泄漏）'
    print(f'    近重复/同源合并：{merged}')
    print(f'    许可可分发占比：按部数 {share["works_pct"]}% / 按字数 {share["chars_pct"]}%'
          f'（{share["works"]} 部 / {share["chars"]} 字）')
    for k, v in sorted(did['license'].items(), key=lambda kv: -kv[1]['works']):
        mark = '可分发' if k in DISTRIBUTABLE else '★不可分发★'
        print(f'      {k:<24}{v["works"]:>7} 部 {v["chars"]:>11} 字  {mark}')

    if args.meta:
        with open(args.meta, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        attach(meta, args.manifest, args.report)
        print(f'\n[*] {args.meta}')
        print(f'    run_identity = {meta["run_identity"]}')
        if meta.get('corpus'):
            print(f'    corpus={meta["corpus"]} seed={meta.get("seed")} steps={meta.get("steps")}')
        print('    注：既有 run 的 data_identity 是**按当前清单**补算的，只用于对照，'
              '不能反推它当时跑在哪版数据上（当时没记）。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
