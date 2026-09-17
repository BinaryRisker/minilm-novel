#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两条训练线的清单对账（只读）。

按 owner 18:55 复核结论：
- 研究线：含 localugc，仅本地研究 / 内部评测；候选 = data/manifest.jsonl
  或 data/manifest_v2.jsonl（两者内容相同，225,756 部）
- 发布线：bucket≠localugc 的可分发数据；当前仓库无现成清单，
  需从 manifest 过滤得到（约 1,144 部）

只读，不写任何文件；不动同事工作线（manifest / shards / dataset.py）。
输出到 stdout，便于同事 review 后决定是否纳入。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = ROOT / "py" / "data"

# 两条线的正式命名（owner 19:10 修订；不再依赖 v2/v3 这种易误导的文件名）
LINE_RESEARCH = "research_full"
LINE_RELEASE_CANDIDATE = "release_candidate"
SHARDS_DIR_RESEARCH = "py/runs/shards/research_full"
SHARDS_DIR_RELEASE = "py/runs/shards/release_candidate"


def scan_manifest(path: Path) -> dict:
    """扫一份 manifest，返回桶分布 / 字符数 / 许可统计。"""
    bucket_n = collections.Counter()
    bucket_chars = collections.Counter()
    lic_n = collections.Counter()
    n = 0
    total_chars = 0
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            bucket = r.get("bucket", "?")
            n += 1
            chars = int(r.get("chars") or r.get("n_chars") or 0)
            bucket_n[bucket] += 1
            bucket_chars[bucket] += chars
            lic_n[(r.get("license") or "?")[:60]] += 1
            total_chars += chars
    return {
        "path": str(path.relative_to(ROOT)),
        "n": n,
        "total_chars": total_chars,
        "buckets_n": dict(bucket_n),
        "buckets_chars": dict(bucket_chars),
        "license_top": dict(lic_n.most_common(8)),
    }


def filter_release_candidate(stats: dict) -> dict:
    """发布线候选：bucket≠localugc 的工程过滤。

    注意：bucket 过滤 ≠ 许可合规。每条来源仍要逐项核 license。
    """
    keep_n = {b: c for b, c in stats["buckets_n"].items() if b != "localugc"}
    keep_chars = {b: c for b, c in stats["buckets_chars"].items() if b != "localugc"}
    return {
        "n": sum(keep_n.values()),
        "chars": sum(keep_chars.values()),
        "buckets_n": keep_n,
        "buckets_chars": keep_chars,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--manifest",
        action="append",
        default=None,
        help="要扫的 manifest 路径（相对仓库根），可重复；默认扫 manifest / manifest_v2 / manifest_v3 三份",
    )
    args = ap.parse_args()

    if args.manifest:
        paths = [ROOT / p for p in args.manifest]
    else:
        paths = [DATA_DIR / "manifest.jsonl", DATA_DIR / "manifest_v2.jsonl", DATA_DIR / "manifest_v3.jsonl"]

    print("=" * 72)
    print(f"两线对账：{LINE_RESEARCH} / {LINE_RELEASE_CANDIDATE}（只读；不动同事工作线）")
    print("=" * 72)

    # 默认推荐映射：研究线=manifest（或 manifest_v2，两者内容相同），
    # 发布候选=从 manifest_v3 过滤 bucket≠localugc。
    for p in paths:
        if not p.exists():
            print(f"\n[skip] {p.relative_to(ROOT)} 不存在")
            continue
        stats = scan_manifest(p)
        rc = filter_release_candidate(stats)
        print(f"\n## {stats['path']}（全量）")
        print(f"  部数={stats['n']:,}  字符={stats['total_chars']:,}")
        print(f"  桶分布：")
        for b in sorted(stats["buckets_n"], key=lambda k: -stats["buckets_n"][k]):
            print(f"    {b:>10s}: {stats['buckets_n'][b]:>7,} 部  {stats['buckets_chars'][b]:>14,} 字")
        print(f"  许可（前 8）：")
        for k, v in stats["license_top"].items():
            print(f"    {v:>7,}  {k}")

        print(f"\n  ## {LINE_RELEASE_CANDIDATE}（filter bucket≠localugc，仅工程过滤）")
        print(f"    部数={rc['n']:,}  字符={rc['chars']:,}")
        for b in sorted(rc["buckets_n"], key=lambda k: -rc["buckets_n"][k]):
            print(f"      {b:>10s}: {rc['buckets_n'][b]:>7,} 部  {rc['buckets_chars'][b]:>14,} 字")

    print("\n" + "=" * 72)
    print("两线正式定义 + 推荐映射")
    print("=" * 72)
    print(f"""
{LINE_RESEARCH}:
  来源：data/manifest.jsonl（含 localugc，仅本地研究 / 内部评测）
  候选分片目录：{SHARDS_DIR_RESEARCH}/

{LINE_RELEASE_CANDIDATE}:
  来源：从 data/manifest_v3.jsonl 过滤 bucket != localugc（工程过滤）
  ⚠️ bucket 过滤 ≠ 许可合规；每条来源仍要逐项核 license
  候选分片目录：{SHARDS_DIR_RELEASE}/

两套分片目录必须严格分开：
  - train / val / test
  - tokenizer
  - shard metadata
  - source split fingerprint
  - vocab fingerprint
绝不共用 py/runs/shards/ 顶层；绝不在两线之间复用 ckpt。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
