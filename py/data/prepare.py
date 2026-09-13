""" ============================================================================
py/data/prepare.py — 数据管线：原始故事 → manifest.jsonl + split 审计
----------------------------------------------------------------------------
职责：
  - 读取 py/data/corpus.json（或用户指定的原始目录）中的 story 记录
  - 为每个 story 分配 story_id，记录 parent_story_id、split、source、哈希、字数
  - 输出 data/manifest.jsonl（审计线，train/val/test 不可交叉）
  - 输出一份 split_audit.json，说明每个 split 包含多少字、多少 story、是否有泄漏

字段定义：
  story_id:      全局唯一短 ID，形如 "scifi_001"
  parent_story_id: 若本文是从某父故事改写/节选而来则填；原创为空
  split:         "train" / "val" / "test" / "unsplit"（未分配）
  source:        来源描述（如 "original by author" / "public domain" / URL）
  hash:          sha256(text) 前 16 字节，用于检测重复或篡改
  chars:         文本字数（含标点）
  vocab_size:    该 story 的独立字符数

用法：
  python py/data/prepare.py
  python py/data/prepare.py --out data/manifest.jsonl --audit data/split_audit.json
============================================================================ """
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def sha256_prefix(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:n]


def build_manifest(corpus_path: str, default_split: str = 'train') -> List[Dict]:
    """把 corpus.json 中的每个故事转成 manifest 条目。"""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    manifest: List[Dict] = []
    for idx, (key, item) in enumerate(corpus.items(), start=1):
        text = item.get('text', '')
        if not text:
            continue
        story_id = f'{key}_{idx:03d}'
        chars = set(text)
        entry = {
            'story_id': story_id,
            'parent_story_id': None,
            'split': default_split if key != 'custom' else 'unsplit',
            'source': item.get('source', 'project original'),
            'title': item.get('name', key),
            'desc': item.get('desc', ''),
            'hash': sha256_prefix(text),
            'chars': len(text),
            'vocab_size': len(chars),
            'license': item.get('license', 'unknown'),
            'note': item.get('note', ''),
        }
        manifest.append(entry)
    return manifest


def split_manifest(manifest: List[Dict], train_frac: float = 0.8,
                   val_frac: float = 0.1) -> List[Dict]:
    """按 story 切分：80% train / 10% val / 10% test（字符级语料少时，先整 story 切分）。
    注意：当前每个 corpus 只有 1 个 story，所以这一版会整体作为 train，需要更多 story 后
    再用时间或哈希重分。此函数主要是给未来多 story 场景用。"""
    out = []
    for i, e in enumerate(manifest):
        if e['split'] != 'train':
            out.append(e)
            continue
        r = i % 10
        if r < int(train_frac * 10):
            e['split'] = 'train'
        elif r < int((train_frac + val_frac) * 10):
            e['split'] = 'val'
        else:
            e['split'] = 'test'
        out.append(e)
    return out


def audit(manifest: List[Dict]) -> Dict:
    """生成 split 审计信息：字数、story 数、检查是否有 hash 冲突或跨 split 的 parent。"""
    stats = {}
    hashes = {}
    issues = []
    for e in manifest:
        sp = e['split']
        stats.setdefault(sp, {'stories': 0, 'chars': 0})
        stats[sp]['stories'] += 1
        stats[sp]['chars'] += e['chars']

        h = e['hash']
        if h in hashes:
            issues.append(f"hash collision: {e['story_id']} vs {hashes[h]}")
        hashes[h] = e['story_id']

    # parent 检查：父故事必须在同 split 或不被允许跨 split
    parent_to_split = {e['story_id']: e['split'] for e in manifest}
    for e in manifest:
        pid = e.get('parent_story_id')
        if pid and pid in parent_to_split:
            if parent_to_split[pid] != e['split']:
                issues.append(f"cross-split parent: {e['story_id']}({e['split']}) -> {pid}({parent_to_split[pid]})")

    return {
        'total_stories': len(manifest),
        'splits': stats,
        'hash_count': len(hashes),
        'issues': issues,
        'ok': len(issues) == 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default=os.path.join(HERE, 'corpus.json'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'manifest.jsonl'))
    ap.add_argument('--audit', default=os.path.join(ROOT, 'data', 'split_audit.json'))
    ap.add_argument('--split', action='store_true', help='是否执行 train/val/test 切分')
    ap.add_argument('--train_frac', type=float, default=0.8)
    ap.add_argument('--val_frac', type=float, default=0.1)
    args = ap.parse_args()

    manifest = build_manifest(args.corpus, default_split='train')
    if args.split:
        manifest = split_manifest(manifest, args.train_frac, args.val_frac)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for e in manifest:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(f'[*] manifest -> {args.out} ({len(manifest)} stories)')

    audit_res = audit(manifest)
    with open(args.audit, 'w', encoding='utf-8') as f:
        json.dump(audit_res, f, ensure_ascii=False, indent=2)
    print(f'[*] audit -> {args.audit}')
    print(f'    ok={audit_res["ok"]} issues={len(audit_res["issues"])}')
    for sp, st in audit_res['splits'].items():
        print(f'    {sp}: {st["stories"]} stories / {st["chars"]} chars')


if __name__ == '__main__':
    main()