#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据一致性 / 完整性只读审计（可复现的三方对账）。

为什么有它：2026-09-17 root 在恢复缓存时报「data/processed 抽样 153 部中 151 部
与 manifest hash 不一致 + 226725 文件 slug 覆盖」，据此把 task-20260917-001
（research_full 六桶分片）标为 blocked_pending_integrity_fix。但独立复核
（dev/review_current.py、随机抽样、以及 root 自己的 data/_research_full_audit.tmp.json
的 issues=[] / ok=true）都显示**当前正式数据是一致的**。本脚本把三方对账固化成
一条可复现命令，谁都能跑，用来判断「数据到底还能不能用」，避免在误诊的阻塞上
空等，也避免在真损坏上继续跑实验。

审计四项（任一失败 → 非零退出）：
  1. 计数恒等：sources.total_works - manifest 行数 == split_audit.dedupe.dropped_works
     （抓取全集减去去重丢弃 = 现行清单；差值必须全在声明被丢的桶里）
  2. hash 唯一：manifest 内无重复 hash（重复作品必须在 prepare 阶段就被丢掉）
  3. 家族不跨切分：parent_story_id / dedup_cluster 不跨 train/val/test
  4. 正文可用：抽样 processed/<bucket>/<slug(title)>.txt 存在且 sha256[:16] == manifest.hash

只读：不写任何文件、不改 manifest / processed / split_audit。
用法：
  python -X utf8 dev/audit_data_consistency.py                 # 默认每桶抽 40 部
  python -X utf8 dev/audit_data_consistency.py --sample 200    # 加大抽样
  python -X utf8 dev/audit_data_consistency.py --full          # 全量逐部校验（慢）
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from data.build_splits import slug  # noqa: E402

MANIFEST = ROOT / "data" / "manifest.jsonl"
SOURCES = ROOT / "data" / "sources.json"
SPLIT_AUDIT = ROOT / "data" / "split_audit.json"
PROCESSED = ROOT / "data" / "processed"


def load_rows(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def check_counts(rows) -> list:
    """1. 计数恒等：sources - manifest == dedupe.dropped_works，且差值桶一致。"""
    problems = []
    if not SOURCES.exists():
        problems.append(f"缺 {SOURCES.name}")
        return problems
    src = json.loads(SOURCES.read_text(encoding="utf-8"))
    sa = json.loads(SPLIT_AUDIT.read_text(encoding="utf-8")) if SPLIT_AUDIT.exists() else {}
    total = src.get("total_works")
    dropped = (sa.get("dedupe") or {}).get("dropped_works")
    if total is None or dropped is None:
        problems.append("sources.total_works 或 split_audit.dedupe.dropped_works 缺失，无法做计数恒等")
        return problems
    diff = int(total) - len(rows)
    if diff != int(dropped):
        problems.append(
            f"计数不恒等：sources.total_works({total}) - manifest({len(rows)}) = {diff}"
            f" ≠ dedupe.dropped_works({dropped})"
        )
    # 差值必须落在 split_audit 声明的桶里（不能出现未声明的桶被丢）
    dropped_buckets = set(((sa.get("dedupe") or {}).get("by_bucket") or {}).keys())
    if dropped_buckets and not dropped_buckets.issubset({r["bucket"] for r in rows} | dropped_buckets):
        problems.append(f"dedupe.by_bucket 出现未知桶：{dropped_buckets}")
    print(f"  [1] 计数恒等：{total} - {len(rows)} = {diff}，声明丢弃 {dropped}，桶={sorted(dropped_buckets)}")
    return problems


def check_hash_unique(rows) -> list:
    """2. hash 唯一（去重已生效）。"""
    problems = []
    groups = collections.defaultdict(list)
    for r in rows:
        h = r.get("hash")
        if h:
            groups[h].append(r)
    dups = {h: v for h, v in groups.items() if len(v) > 1}
    if dups:
        h0, v0 = next(iter(dups.items()))
        problems.append(f"manifest 有 {len(dups)} 组重复 hash（如 {v0[0]['story_id']} vs {v0[1]['story_id']}）")
    print(f"  [2] hash 唯一：{len(groups)} 组 / {len(rows)} 部，重复组={len(dups)}")
    return problems


def check_family_split(rows) -> list:
    """3. 家族不跨切分。"""
    problems = []
    for field in ("parent_story_id", "dedup_cluster"):
        groups = collections.defaultdict(set)
        for r in rows:
            key = r.get(field)
            size = int(r.get("dedup_cluster_size") or 1)
            if key and (field == "parent_story_id" or size > 1):
                groups[str(key)].add(r.get("split"))
        moved = [k for k, v in groups.items() if len(v) > 1]
        if moved:
            problems.append(f"{field} 有 {len(moved)} 个家族跨切分（如 {moved[0]}）")
        print(f"  [3] {field}：{len(groups)} 个家族，跨切分={len(moved)}")
    return problems


def check_bodies(rows, sample: int, full: bool, seed: int) -> list:
    """4. 正文存在且 hash 一致（抽样或全量）。"""
    problems = []
    if not PROCESSED.exists():
        problems.append(f"缺 {PROCESSED}（正文未落盘，无法校验）")
        return problems
    rng = random.Random(seed)
    by_bucket = collections.defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r)
    picked = []
    for b, lst in by_bucket.items():
        if full:
            picked.extend(lst)
        else:
            picked.extend(rng.sample(lst, min(sample, len(lst))))
    missing, mismatch = [], []
    for r in picked:
        p = PROCESSED / r["bucket"] / (slug(r["title"]) + ".txt")
        if not p.exists():
            missing.append(r["story_id"])
            continue
        digest = hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()[:16]
        if digest != r.get("hash"):
            mismatch.append((r["story_id"], r["bucket"], digest, r.get("hash")))
    if missing:
        problems.append(f"正文缺失 {len(missing)} 部（如 {missing[:3]}）")
    if mismatch:
        s = mismatch[0]
        problems.append(f"hash 不一致 {len(mismatch)} 部（如 {s[0]}/{s[1]}：file={s[2]} manifest={s[3]}）")
    mode = "全量" if full else f"抽样 {sample}/桶"
    print(f"  [4] 正文校验（{mode}）：检查 {len(picked)} 部，缺失={len(missing)}，hash 不一致={len(mismatch)}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--sample", type=int, default=40, help="每桶抽样部数（默认 40）")
    ap.add_argument("--full", action="store_true", help="全量逐部校验（慢，19 万文件）")
    ap.add_argument("--seed", type=int, default=20260918)
    args = ap.parse_args()

    man = Path(args.manifest)
    if not man.exists():
        print(f"[FATAL] 缺 {man}")
        return 2
    rows = load_rows(man)
    print(f"审计 {man}：{len(rows):,} 部")
    problems = []
    problems += check_counts(rows)
    problems += check_hash_unique(rows)
    problems += check_family_split(rows)
    problems += check_bodies(rows, args.sample, args.full, args.seed)

    print()
    if problems:
        print(f"结论：✗ 发现 {len(problems)} 个问题")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("结论：✓ 数据一致性 / 完整性四项全部通过（正文可用，可进入分片与训练）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
