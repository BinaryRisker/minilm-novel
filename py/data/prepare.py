""" ============================================================================
py/data/prepare.py — 数据管线：作品清单 → manifest.jsonl + split 审计
----------------------------------------------------------------------------
职责：
  - 读取语料来源审计 data/sources.json（由 py/data/fetch_corpus.py 生成），
    为**每一部作品**生成一条 manifest 记录
  - 记录 story_id / parent_story_id / split / source / license / hash / 字数 / 独立字符数
  - 按**内容哈希**去重（不同 id、逐字相同的正文只留一部），再切分
  - 输出 data/manifest.jsonl（审计线，train/val/test 不跨作品）
  - 输出 data/split_audit.json，说明每个 split 的字数与作品数、去重情况、是否有哈希冲突

为什么按作品而不是按桶：语料扩充到 90+ 部作品后，按桶切成一条记录会让
train/val/test 的边界无法追溯；按作品记录才能保证「同一部作品不跨集合」。
注意：当前 minilm_torch.train 仍用「按字符连续切 90/10」的旧方案，
本 manifest 的 split 字段是按作品切分的**推荐方案与审计依据**，两者暂未打通。

回退：若 data/sources.json 不存在，则退回按 py/data/corpus.json 的桶级记录。

字段定义：
  story_id:        全局唯一短 ID，形如 "wuxia_007"
  parent_story_id: 若从某父作品改写/节选而来则填；原作为空
  split:           "train" / "val" / "test" / "unsplit"
  source:          gutenberg / hcc / wikisource
  license:         许可说明
  ref:             来源链接
  hash:            sha256(text) 前 16 字节
  chars:           文本字数（含标点）
  vocab_size:      该作品的独立字符数

用法：
  python py/data/prepare.py                      # 只列清单，不切分
  python py/data/prepare.py --split              # 按作品 80/10/10 切分
  python py/data/prepare.py --out data/manifest.jsonl --audit data/split_audit.json
============================================================================ """
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_SOURCES = os.path.join(ROOT, 'data', 'sources.json')
FALLBACK_CORPUS = os.path.join(HERE, 'corpus.json')


def sha256_prefix(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:n]


def vocab_size_of(text: str) -> int:
    return len(set(text))


def build_manifest_from_sources(sources_path: str, bucket_of_title: Dict[str, str]) -> List[Dict]:
    """从 data/sources.json 逐作品建 manifest。"""
    with open(sources_path, 'r', encoding='utf-8') as f:
        audit = json.load(f)

    manifest: List[Dict] = []
    counters: Dict[str, int] = {}
    for w in audit.get('works', []):
        bucket = w.get('bucket') or bucket_of_title.get(w['title'], 'unknown')
        counters[bucket] = counters.get(bucket, 0) + 1
        manifest.append({
            'story_id': f"{bucket}_{counters[bucket]:03d}",
            'parent_story_id': None,
            'split': 'unsplit',
            'bucket': bucket,
            'title': w.get('title'),
            'source': w.get('source'),
            'all_sources': w.get('all_sources'),
            'license': w.get('license'),
            'ref': w.get('ref'),
            'hash': w.get('sha256') or '',
            # 注意：sources.json 里 chars 是**清洗后全文**字数；
            # 实际入桶字数是 corpus.json 里 stories[].chars（可能被均分截断）
            'chars': w.get('chars', 0),
            'vocab_size': None,
        })
    return manifest


def build_manifest_from_corpus(corpus_path: str) -> List[Dict]:
    """回退路径：按桶建记录（旧行为）。"""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    manifest: List[Dict] = []
    for idx, (key, item) in enumerate(corpus.items(), start=1):
        text = item.get('text', '')
        if not text:
            continue
        manifest.append({
            'story_id': f'{key}_{idx:03d}',
            'parent_story_id': None,
            'split': 'train' if key != 'custom' else 'unsplit',
            'bucket': key,
            'title': item.get('name', key),
            'source': item.get('source'),
            'all_sources': None,
            'license': item.get('license'),
            'ref': None,
            'hash': sha256_prefix(text),
            'chars': len(text),
            'vocab_size': vocab_size_of(text),
        })
    return manifest


def dedupe_by_content(manifest: List[Dict]) -> "tuple[List[Dict], Dict]":
    """内容逐字相同的作品只保留一部（按 story_id 稳定顺序保留第一部），其余丢弃并记录。

    为什么必须在 manifest 层做：`fetch_corpus.py` 的跨源去重按**标题**，而 localugc
    这类切片源里不同 id 可能对应**逐字相同**的正文（本次实测 292 组 / 550 个多余副本 /
    149.6 万字，标题互不相同故抓不到）。不去重有两个后果：
      1) 虚增数据量，数据规模实验的「字数」不是真实信息量；
      2) 同一正文的两个副本被切进不同 split —— 那是货真价实的 train/test 泄漏（本次 1 组）。
    去重放在 split 分配**之前**，保证留下的那一部只属于一个集合。
    """
    seen: Dict[str, str] = {}
    kept: List[Dict] = []
    dropped_by_bucket: Dict[str, int] = {}
    dropped_chars = 0
    for e in sorted(manifest, key=lambda x: x['story_id']):
        h = e.get('hash')
        if h and h in seen:
            b = e.get('bucket') or 'unknown'
            dropped_by_bucket[b] = dropped_by_bucket.get(b, 0) + 1
            dropped_chars += e.get('chars') or 0
            continue
        if h:
            seen[h] = e['story_id']
        kept.append(e)
    info = {'dropped_works': sum(dropped_by_bucket.values()),
            'dropped_chars': dropped_chars,
            'by_bucket': dropped_by_bucket}
    return kept, info


def assign_splits(manifest: List[Dict], train_frac: float = 0.8,
                  val_frac: float = 0.1) -> List[Dict]:
    """按作品切分 train/val/test：同一桶内按 hash 稳定排序后按比例切，
    保证「同一部作品的各章节不跨集合」（本管线里每部作品整体属于一个 split）。"""
    by_bucket: Dict[str, List[Dict]] = {}
    for e in manifest:
        if e['split'] == 'unsplit':
            by_bucket.setdefault(e['bucket'], []).append(e)

    for bucket, entries in by_bucket.items():
        entries.sort(key=lambda e: e['hash'] or e['title'])
        n = len(entries)
        if n < 3:
            for e in entries:
                e['split'] = 'train'
            continue
        n_train = max(1, int(math.floor(n * train_frac)))
        n_val = max(1, int(math.floor(n * val_frac))) if n >= 10 else (1 if n >= 8 else 0)
        for i, e in enumerate(entries):
            if i < n_train:
                e['split'] = 'train'
            elif i < n_train + n_val:
                e['split'] = 'val'
            else:
                e['split'] = 'test'
    return manifest


def audit(manifest: List[Dict]) -> Dict:
    """split 审计：字数 / 作品数 / 哈希冲突 / 跨 split 的 parent。"""
    stats: Dict[str, Dict] = {}
    by_bucket: Dict[str, Dict[str, int]] = {}
    hashes: Dict[str, str] = {}
    issues: List[str] = []

    for e in manifest:
        sp = e['split']
        stats.setdefault(sp, {'works': 0, 'chars': 0})
        stats[sp]['works'] += 1
        stats[sp]['chars'] += e.get('chars') or 0

        b = e.get('bucket', 'unknown')
        by_bucket.setdefault(b, {})
        by_bucket[b][sp] = by_bucket[b].get(sp, 0) + 1

        h = e.get('hash')
        if h:
            if h in hashes:
                issues.append(f"hash collision: {e['story_id']} vs {hashes[h]}")
            hashes[h] = e['story_id']

    parent_to_split = {e['story_id']: e['split'] for e in manifest}
    for e in manifest:
        pid = e.get('parent_story_id')
        if pid and pid in parent_to_split and parent_to_split[pid] != e['split']:
            issues.append(f"cross-split parent: {e['story_id']}({e['split']}) "
                          f"-> {pid}({parent_to_split[pid]})")

    return {
        'total_works': len(manifest),
        'total_chars': sum((e.get('chars') or 0) for e in manifest),
        'splits': stats,
        'by_bucket': by_bucket,
        'unique_hashes': len(hashes),
        'issues': issues,
        'ok': len(issues) == 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', default=DEFAULT_SOURCES,
                    help='来源审计（优先）；不存在则回退 --corpus')
    ap.add_argument('--corpus', default=FALLBACK_CORPUS)
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'manifest.jsonl'))
    ap.add_argument('--audit', default=os.path.join(ROOT, 'data', 'split_audit.json'))
    ap.add_argument('--split', action='store_true', help='按作品执行 train/val/test 切分')
    ap.add_argument('--train_frac', type=float, default=0.8)
    ap.add_argument('--val_frac', type=float, default=0.1)
    args = ap.parse_args()

    bucket_of_title: Dict[str, str] = {}
    if os.path.exists(args.corpus):
        with open(args.corpus, 'r', encoding='utf-8') as f:
            for k, v in json.load(f).items():
                for s in (v.get('stories') or []):
                    bucket_of_title[s['title']] = k

    if os.path.exists(args.sources):
        manifest = build_manifest_from_sources(args.sources, bucket_of_title)
        print(f'[*] 来源：{args.sources}（逐作品）')
    else:
        manifest = build_manifest_from_corpus(args.corpus)
        print(f'[!] 未找到 {args.sources}，回退到桶级记录 {args.corpus}')

    manifest, dedup = dedupe_by_content(manifest)
    if dedup['dropped_works']:
        print(f'[*] 内容去重：丢弃 {dedup["dropped_works"]} 部逐字重复作品 / '
              f'{dedup["dropped_chars"]} 字（{dedup["by_bucket"]}）')

    if args.split:
        manifest = assign_splits(manifest, args.train_frac, args.val_frac)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for e in manifest:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(f'[*] manifest -> {args.out} ({len(manifest)} 条作品记录)')

    res = audit(manifest)
    res['dedupe'] = dedup
    with open(args.audit, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f'[*] audit -> {args.audit}')
    print(f'    ok={res["ok"]} issues={len(res["issues"])} '
          f'total={res["total_works"]} works / {res["total_chars"]} chars')
    for sp, st in sorted(res['splits'].items()):
        print(f'    {sp:<8}: {st["works"]:>3} 部 / {st["chars"]:>9} 字')
    for b, d in sorted(res['by_bucket'].items()):
        print(f'    [{b}] ' + ', '.join(f'{k}={v}' for k, v in sorted(d.items())))
    for i in res['issues']:
        print('    [!]', i)


if __name__ == '__main__':
    main()
