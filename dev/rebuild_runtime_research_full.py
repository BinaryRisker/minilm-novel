"""Rebuild a collision-free research_full runtime corpus from local caches.

This script deliberately writes only under data/runtime_research_full/ and
data/_runtime_research_full*. It never changes the tracked manifest or the
normal data/processed directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "py"))
from data import fetch_corpus  # noqa: E402


OUT_ROOT = os.path.join(ROOT, "data", "runtime_research_full")
MANIFEST_OUT = os.path.join(ROOT, "data", "_runtime_research_full_manifest.jsonl")
REPORT_OUT = os.path.join(ROOT, "data", "_runtime_research_full_split_report.json")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    print("[*] loading local cached sources")
    stories = []
    stories += fetch_corpus.fetch_gutenberg(None)
    stories += fetch_corpus.fetch_hcc()
    stories += fetch_corpus.fetch_wikisource(None)
    stories += fetch_corpus.fetch_huggingface(None)
    stories += fetch_corpus.fetch_tinystories(1000)
    stories += fetch_corpus.fetch_nutrans()
    stories += fetch_corpus.fetch_localugc(per_source_cap=2500)
    stories = fetch_corpus.dedupe(stories)
    print(f"[*] deduped works={len(stories)}")

    # Stable IDs are assigned after dedupe. Files use these IDs, so title
    # collisions cannot overwrite another work.
    counters = defaultdict(int)
    rows = []
    for st in stories:
        bucket = st.get("bucket") or "unknown"
        counters[bucket] += 1
        story_id = f"{bucket}_{counters[bucket]:06d}"
        text = st.get("text") or ""
        if not text:
            continue
        path = os.path.join(OUT_ROOT, bucket, story_id + ".txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        rows.append({
            "story_id": story_id,
            "parent_story_id": None,
            "split": "unsplit",
            "bucket": bucket,
            "title": st.get("title") or story_id,
            "source": st.get("source"),
            "all_sources": st.get("all_sources"),
            "license": st.get("license"),
            "ref": st.get("ref"),
            "hash": sha(text),
            "chars": len(text),
            "vocab_size": len(set(text)),
            "processed_file": os.path.relpath(path, ROOT).replace("\\", "/"),
        })

    # Deterministic 80/10/10 split within each bucket.
    by_bucket = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket"]].append(row)
    for bucket, entries in by_bucket.items():
        entries.sort(key=lambda r: (r["hash"], r["story_id"]))
        n = len(entries)
        n_train = max(1, math.floor(n * 0.8))
        n_val = max(1, math.floor(n * 0.1)) if n >= 10 else (1 if n >= 8 else 0)
        for i, row in enumerate(entries):
            row["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

    rows.sort(key=lambda r: r["story_id"])
    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {sp: {"works": 0, "chars": 0} for sp in ("train", "val", "test")}
    for row in rows:
        stats[row["split"]]["works"] += 1
        stats[row["split"]]["chars"] += row["chars"]
    source = "\n".join(f"{r['story_id']}|{r['hash']}|{r['split']}" for r in rows)
    fp = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    report = {
        "generated_by": "dev/rebuild_runtime_research_full.py",
        "manifest": os.path.relpath(MANIFEST_OUT, ROOT).replace("\\", "/"),
        "processed_dir": os.path.relpath(OUT_ROOT, ROOT).replace("\\", "/"),
        "split_fingerprint": fp,
        "full_manifest_works": len(rows),
        "total_chars": sum(r["chars"] for r in rows),
        "splits": stats,
        "buckets": {b: len(v) for b, v in sorted(by_bucket.items())},
        "ok": True,
    }
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
