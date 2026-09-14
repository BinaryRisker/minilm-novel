""" ============================================================================
py/splits.py — 按作品切分的统一读取入口（train / eval_story / 审计共用）
----------------------------------------------------------------------------
为什么要单独抽一层（本次自检发现的问题）：
  train.py 的 `--splits` 直接读 `data/splits/train.txt`，那是**全部桶合并**的文件；
  想「只在一个桶上训练」只能自己手工预处理，实验脚本各写一套过滤，容易写歪。
  eval_story.py 则完全不看 split，提示词从哪来都说不清 —— 生成评测可能直接
  在训练过的作品上出题（自问自答）。

本模块提供唯一口径：
  bucket_texts(bucket) -> {'train': [...], 'val': [...], 'test': [...]}
      每项是 [(story_id, title, text), ...]，按 story_id 排序（可复现）。
      正文取 data/processed/<bucket>/<slug(title)>.txt（整篇，非截断拼接）。
  bucket_joined(bucket) -> {'train': str, 'val': str, 'test': str}
      按固定顺序 \n\n 拼接，供训练使用。

约定：
  - 词表只在 train 上建（调用方负责），val 用于选模型，test 只在最后评一次。
  - 找不到作品正文直接抛错，不静默跳过（静默跳过会让字数对不上还看起来跑通了）。

强制校验（V3 P0「不合规数据和不匹配缓存非零退出」）：
  bucket_texts() 每次先 verify()，两件事不满足就直接抛错、非零退出：
    1) **split 版本**：manifest 现算指纹 == data/split_report.json 记录的指纹。
       实测踩过：抓取/去重重跑后 manifest 变了，data/splits/*.txt 还是旧版，
       实验在旧切分上跑出无法复现的数字。
    2) **家族/内容去重**：manifest 内不得有逐字重复作品（重复会虚增数据量，
       且副本落进不同 split 就是 train/test 泄漏）。
  指纹算法只在这里定义一份，py/data/build_splits.py 反向 import 它，避免两处各写一遍算歪。
============================================================================ """
from __future__ import annotations
import hashlib
import os
import re
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_MANIFEST = os.path.join(ROOT, 'data', 'manifest.jsonl')
DEFAULT_PROCESSED = os.path.join(ROOT, 'data', 'processed')
DEFAULT_REPORT = os.path.join(ROOT, 'data', 'split_report.json')

SPLITS = ('train', 'val', 'test')
Work = Tuple[str, str, str]          # (story_id, title, text)


class SplitIdentityError(RuntimeError):
    """切分身份不成立（版本不匹配 / 未去重）。调用方应当让它直接非零退出。"""


def split_fingerprint(rows: List[dict]) -> str:
    """切分版本指纹：对 (story_id, 内容哈希, split) 三元组按 story_id 稳定排序后哈希。"""
    src = '\n'.join(f'{r["story_id"]}|{r.get("hash") or ""}|{r["split"]}'
                    for r in sorted(rows, key=lambda x: x['story_id']))
    return hashlib.sha256(src.encode('utf-8')).hexdigest()[:16]


def duplicate_hashes(rows: List[dict]) -> List[Tuple[str, str]]:
    """返回逐字重复的作品对 [(重复者 story_id, 首个 story_id), ...]。"""
    first: Dict[str, str] = {}
    out: List[Tuple[str, str]] = []
    for r in sorted(rows, key=lambda x: x['story_id']):
        h = r.get('hash')
        if not h:
            continue
        if h in first:
            out.append((r['story_id'], first[h]))
        else:
            first[h] = r['story_id']
    return out


def verify(manifest_path: str = DEFAULT_MANIFEST,
           report_path: str = DEFAULT_REPORT) -> Dict[str, object]:
    """校验切分身份，不成立抛 SplitIdentityError；返回 {'fingerprint', 'works'}。"""
    import json
    if not os.path.exists(manifest_path):
        raise SplitIdentityError(f'缺 {manifest_path}（先跑 python py/data/prepare.py --split）')
    rows = read_manifest(manifest_path)
    problems: List[str] = []

    dups = duplicate_hashes(rows)
    if dups:
        problems.append(f'manifest 有 {len(dups)} 个逐字重复作品（如 {dups[0][0]} vs {dups[0][1]}）'
                        f'—— 先跑 python py/data/prepare.py --split 完成内容去重')

    fp = split_fingerprint(rows)
    if not os.path.exists(report_path):
        problems.append(f'缺 {report_path} —— 先跑 python py/data/build_splits.py')
    else:
        with open(report_path, 'r', encoding='utf-8') as f:
            rep = json.load(f)
        got = rep.get('split_fingerprint')
        if got != fp:
            problems.append(
                f'splits 与 manifest 不是同一版：报告指纹 {got!r} ≠ 当前 manifest 指纹 {fp!r}'
                f' —— 重跑 python py/data/build_splits.py（旧切分上跑出的结果不可复现，勿沿用）')

    if problems:
        raise SplitIdentityError('切分身份校验失败：\n  - ' + '\n  - '.join(problems))
    return {'fingerprint': fp, 'works': len(rows)}


def slug(s: str) -> str:
    """与 py/data/fetch_corpus.py / build_splits.py 保持一致的落盘文件名规则。"""
    s = re.sub(r'[\\/:*?"<>|\s]+', '_', s.strip())
    return s[:80] or 'untitled'


def read_manifest(path: str = DEFAULT_MANIFEST) -> List[dict]:
    import json
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bucket_texts(bucket: str, manifest_path: str = DEFAULT_MANIFEST,
                 processed_dir: str = DEFAULT_PROCESSED,
                 report_path: str = None) -> Dict[str, List[Work]]:
    """取某桶 train/val/test 的作品正文（整篇）。加载前强制校验切分身份。

    report_path：与 manifest_path 配套的切分报告（默认 data/split_report.json）。
    用别的清单（如 py/dedup_works.py 产出的 manifest_v2.jsonl）时，要传它自己的
    报告文件，否则指纹对不上会被判「不是同一版」。
    """
    if processed_dir == DEFAULT_PROCESSED:
        verify(manifest_path, report_path or DEFAULT_REPORT)   # 版本不匹配 / 未去重 → 非零退出
    rows = [r for r in read_manifest(manifest_path) if r['bucket'] == bucket]
    if not rows:
        raise ValueError(f'桶 {bucket} 在 manifest 里没有任何作品')
    out: Dict[str, List[Work]] = {sp: [] for sp in SPLITS}
    for r in sorted(rows, key=lambda x: x['story_id']):
        sp = r['split']
        if sp not in out:
            continue
        p = os.path.join(processed_dir, bucket, slug(r['title']) + '.txt')
        if not os.path.exists(p):
            raise FileNotFoundError(f'作品正文缺失：{p}（{r["story_id"]} {r["title"]}）')
        with open(p, 'r', encoding='utf-8') as f:
            out[sp].append((r['story_id'], r['title'], f.read()))
    return out


def bucket_joined(bucket: str, **kw) -> Dict[str, str]:
    """返回 {'train': text, 'val': text, 'test': text}（作品按 story_id 排序，\\n\\n 拼接）。"""
    t = bucket_texts(bucket, **kw)
    return {sp: '\n\n'.join(x[2] for x in t[sp]) for sp in SPLITS}


def bucket_stats(bucket: str, **kw) -> Dict[str, dict]:
    t = bucket_texts(bucket, **kw)
    return {sp: {'works': len(t[sp]), 'chars': sum(len(x[2]) for x in t[sp])} for sp in SPLITS}
