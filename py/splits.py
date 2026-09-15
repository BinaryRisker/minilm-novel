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
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_MANIFEST = os.path.join(ROOT, 'data', 'manifest.jsonl')
DEFAULT_PROCESSED = os.path.join(ROOT, 'data', 'processed')
DEFAULT_REPORT = os.path.join(ROOT, 'data', 'split_report.json')

SPLITS = ('train', 'val', 'test')
Work = Tuple[str, str, str]          # (story_id, title, text)

# 「近重复/同源已合并」的候选清单（可选、非破坏性），越靠前越优先。
# 每一代去重产物的完整文件若落在 data/ 下，就自动进入候选；没落盘（如 v3 因 105 MB
# 超 GitHub 单文件上限只入了增量 moves）就不会被选中 —— 候选必须有**可加载的完整文件**。
DEDUP_CANDIDATES: List[Tuple[str, str]] = [
    (os.path.join(ROOT, 'data', 'manifest_v3.jsonl'), os.path.join(ROOT, 'data', 'split_report_v3.json')),
    (os.path.join(ROOT, 'data', 'manifest_v2.jsonl'), os.path.join(ROOT, 'data', 'split_report_v2.json')),
]

_warned = False
_resolved: "Optional[Tuple[str, str]]" = None


class SplitIdentityError(RuntimeError):
    """切分身份不成立（版本不匹配 / 未去重）。调用方应当让它直接非零退出。"""


def _same_corpus(a: str, b: str) -> bool:
    """两份清单是否描述同一批作品（按内容哈希集合判，不看 story_id 编号）。"""
    ha = {r.get('hash') for r in read_manifest(a) if r.get('hash')}
    hb = {r.get('hash') for r in read_manifest(b) if r.get('hash')}
    return ha == hb


def resolve_current(manifest_path: str = None, report_path: str = None,
                    verbose: bool = True) -> Tuple[str, str]:
    """决定「当前该用哪份清单」，返回 (manifest, report)。

    规则（显式优先，其次择优，绝不静默降级）：
      1. 调用方给了路径 → 用它（配套报告默认 data/split_report.json）。
      2. 否则按 DEDUP_CANDIDATES 从新到旧找第一份**存在且与当前 data/manifest.jsonl
         是同一批作品**（按内容哈希集合判）的去重清单 —— 它做过近重复/同源合并。
      3. 找不到就退回 data/manifest.jsonl，并**明确告警**：这份清单未做近重复合并，
         留出集可能仍有同源泄漏（py/audit_splits.py 的 B 检查会给数字）。

    为什么不能「去重清单存在就无条件用」：实测 v2 是 32,036 部时期的快照，而当前 v1 已涨到
    192,823 部 —— 无条件优先会**静默丢掉 16 万部数据**。乐观偏好必须带新鲜度校验。
    """
    global _warned, _resolved
    if manifest_path:
        return manifest_path, (report_path or DEFAULT_REPORT)
    if _resolved is not None:
        return _resolved          # 进程内只判定一次（清单 19 万行，重复判定很贵）
    for dedup_man, dedup_rep in DEDUP_CANDIDATES:
        if not (os.path.exists(dedup_man) and os.path.exists(dedup_rep)):
            continue
        try:
            if os.path.exists(DEFAULT_MANIFEST) and _same_corpus(dedup_man, DEFAULT_MANIFEST):
                _resolved = (dedup_man, dedup_rep)
                return _resolved
            if verbose and not _warned:
                _warned = True
                import sys as _sys
                print(f'[!] {os.path.basename(dedup_man)} 与当前 '
                      f'{os.path.basename(DEFAULT_MANIFEST)} 不是同一批作品（过期快照），'
                      f'本轮回退到未合并近重复的默认清单。重跑近重复合并：'
                      f'python py/dedup_works.py', file=_sys.stderr)
        except Exception:       # noqa: BLE001 —— 对齐判定失败时按「不可信」处理，安全侧退回
            pass
    if verbose and not _warned:
        _warned = True
        import sys as _sys
        print('[!] 当前清单未做近重复/同源合并（无可用且新鲜的 data/manifest_v*.jsonl）；'
              '留出集可能仍有同源泄漏，泄漏量化见 python py/audit_splits.py', file=_sys.stderr)
    _resolved = (DEFAULT_MANIFEST, DEFAULT_REPORT)
    return _resolved


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


def verify(manifest_path: str = None,
           report_path: str = None,
           rows: List[dict] = None) -> Dict[str, object]:
    """校验切分身份，不成立抛 SplitIdentityError；返回 {'fingerprint', 'works', 'manifest', 'report'}。
    不传路径时按 resolve_current() 择优（干净且新鲜的清单优先）。
    rows 可传入已解析的清单，省一次 19 万行的重复解析。"""
    import json
    manifest_path, report_path = resolve_current(manifest_path, report_path)
    if rows is None:
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
    return {'fingerprint': fp, 'works': len(rows),
            'manifest': manifest_path, 'report': report_path}


def slug(s: str) -> str:
    """与 py/data/fetch_corpus.py / build_splits.py 保持一致的落盘文件名规则。"""
    s = re.sub(r'[\\/:*?"<>|\s]+', '_', s.strip())
    return s[:80] or 'untitled'


_manifest_cache: Dict[Tuple[str, float], List[dict]] = {}


def read_manifest(path: str = DEFAULT_MANIFEST) -> List[dict]:
    """解析清单。按 (路径, mtime) 缓存，避免同一次实验里反复解析 19 万行；
    返回浅拷贝，调用方可以安全地对自己的 list 排序/增删（不共享 list 对象）。"""
    import json
    try:
        key: Optional[Tuple[str, float]] = (os.path.abspath(path), os.path.getmtime(path))
    except OSError:
        key = None
    if key is not None:
        hit = _manifest_cache.get(key)
        if hit is not None:
            return list(hit)
    rows: List[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if key is not None:
        if len(_manifest_cache) > 4:
            _manifest_cache.clear()
        _manifest_cache[key] = rows
    return list(rows)


def bucket_texts(bucket: str, manifest_path: str = None,
                 processed_dir: str = DEFAULT_PROCESSED,
                 report_path: str = None) -> Dict[str, List[Work]]:
    """取某桶 train/val/test 的作品正文（整篇）。加载前强制校验切分身份。

    manifest_path / report_path：不传则按 resolve_current() 择优（干净且新鲜的清单优先）。
    用别的清单（如 py/dedup_works.py 产出的 manifest_v2.jsonl）时要连同它自己的报告一起传，
    否则指纹对不上会被判「不是同一版」。
    """
    manifest_path, report_path = resolve_current(manifest_path, report_path)
    all_rows = read_manifest(manifest_path)
    if processed_dir == DEFAULT_PROCESSED:
        verify(manifest_path, report_path, rows=all_rows)   # 版本不匹配 / 未去重 → 非零退出
    rows = [r for r in all_rows if r['bucket'] == bucket]
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
