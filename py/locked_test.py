""" ============================================================================
py/locked_test.py — 锁定测试族：一次性冻结、漂移告警、（将来）一次性使用
----------------------------------------------------------------------------
要解决的问题（V3 P0：「所有调参改用 val；历史 test 标 exploratory，新锁定测试族」）：
  实测 `py/sweep_worksplit.py` 在**每个 eval 步号**都读 test 画收敛曲线，
  预算阈值 / 收敛点这些结论实际是在 test 上反复看过之后才定下的 —— 按 V3 的定义，
  这类数字只能算 **exploratory（探索性）**，不能当验收依据。
  `train.py` 本身已经合规：训练中只看 val（用于选模型），test 只在最后评一次。

本模块提供「锁定测试族」：
  freeze  从当前清单的 test split 里按**内容哈希**稳定抽取每桶 N 部，连同
          当时的清单指纹一起冻结到 data/locked_test.json。
          用哈希不用 story_id —— story_id 会随去重/重跑而重新编号，哈希不会。
  verify  校验冻结仍然成立：这批作品的哈希还存在吗？split 归属变了吗？
          清单换代 / 近重复合并导致归属变化 → 报告 drift、非零退出。
          这是有意为之：锁定族**必须**跟着数据版本断言，否则"锁定"形同虚设。

用法：
  python py/locked_test.py --freeze --per-bucket 20     # 冻结（幂等：同数据同结果）
  python py/locked_test.py --verify                     # 校验（漂移 → 退出码 1）
  python py/locked_test.py --show                       # 打印锁定族摘要
协议（docs/TODO_NOVEL_MODEL_LEARNING_V3.md P0 第 4 条）：
  调参 / 选模型 / 定超参 → 只看 val；
  锁定测试族 → 冻结之后**只在最终验收时评一次**；
  历史 test 反复读的数字（wsplit / datascale / budget / ctx 各轮 sweep）→ exploratory。
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

LOCK_PATH = os.path.join(ROOT, 'data', 'locked_test.json')
PROTOCOL_VERSION = 1


class LockDriftError(RuntimeError):
    """锁定测试族与当前清单不再对应（数据换版 / 归属变化）。必须重新冻结。"""


def _lock_fingerprint(works: List[dict]) -> str:
    src = '\n'.join(f'{w["hash"]}|{w["bucket"]}|{w["split"]}|{w["chars"]}'
                    for w in sorted(works, key=lambda x: (x['bucket'], x['hash'])))
    return hashlib.sha256(src.encode('utf-8')).hexdigest()[:16]


def freeze(per_bucket: int = 20, buckets: Optional[List[str]] = None,
           manifest_path: str = None, report_path: str = None,
           out_path: str = LOCK_PATH) -> dict:
    """从当前（择优后的）清单 test split 冻结锁定族。同数据重复执行 → 同一文件（幂等）。"""
    man, rep = splits.resolve_current(manifest_path, report_path)
    rows = [r for r in splits.read_manifest(man) if r['split'] == 'test' and r.get('hash')]
    by_bucket: Dict[str, List[dict]] = {}
    for r in rows:
        by_bucket.setdefault(r['bucket'] or 'unknown', []).append(r)
    works: List[dict] = []
    for b in sorted(by_bucket):
        if buckets and b not in buckets:
            continue
        cand = sorted(by_bucket[b], key=lambda r: r['hash'])   # 稳定：按内容哈希取前 N
        for r in cand[:max(0, per_bucket)]:
            works.append({'hash': r['hash'], 'bucket': b, 'title': r.get('title'),
                          'split': r['split'], 'chars': r.get('chars') or 0})
    if not works:
        raise ValueError('test split 为空（或 --buckets 过滤后为空），无可锁定的作品')
    doc = {
        'protocol_version': PROTOCOL_VERSION,
        'generated_by': 'py/locked_test.py --freeze',
        'note': '锁定测试族：调参只看 val；本族只在最终验收时评一次。'
                '身份按内容哈希（story_id 会随去重/重跑重新编号，不可靠）。',
        'manifest_fingerprint_at_freeze': splits.verify(man, rep)['fingerprint'],
        'manifest': os.path.basename(man),
        'per_bucket': per_bucket,
        'lock_fingerprint': _lock_fingerprint(works),
        'works': works,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    return doc


def load_lock(path: str = LOCK_PATH) -> dict:
    if not os.path.exists(path):
        raise LockDriftError(f'缺 {path}（先跑 python py/locked_test.py --freeze）')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def verify(path: str = LOCK_PATH, manifest_path: str = None,
           report_path: str = None) -> dict:
    """校验锁定族仍与当前清单对应；任何漂移抛 LockDriftError（非零退出）。"""
    doc = load_lock(path)
    works = doc.get('works') or []
    problems: List[str] = []

    if doc.get('protocol_version') != PROTOCOL_VERSION:
        problems.append(f'协议版本不符：锁定文件 v{doc.get("protocol_version")} ≠ 当前 v{PROTOCOL_VERSION}')
    if doc.get('lock_fingerprint') != _lock_fingerprint(works):
        problems.append('锁定文件被改动过：lock_fingerprint 与 works 内容不符')

    man, rep = splits.resolve_current(manifest_path, report_path)
    rows = splits.read_manifest(man)
    cur_fp = splits.verify(man, rep)['fingerprint']
    notes: List[str] = []
    if cur_fp != doc.get('manifest_fingerprint_at_freeze'):
        # 清单换代**不必然**作废锁定（锁定族按哈希逐部追踪），但必须显式说明：
        # 失败只看下面的逐部核对（作品消失 / 归属变化），换代本身只作注记。
        notes.append(f'清单已换代：冻结时指纹 {doc.get("manifest_fingerprint_at_freeze")} '
                     f'≠ 当前 {cur_fp}；以下按哈希逐部核对归属。')

    by_hash: Dict[str, dict] = {r['hash']: r for r in rows if r.get('hash')}
    moved, gone = [], []
    for w in works:
        r = by_hash.get(w['hash'])
        if r is None:
            gone.append(w)
        elif r['split'] != w['split']:
            moved.append((w, r['split']))
    if gone:
        problems.append(f'{len(gone)} 部锁定作品在当前清单中已不存在'
                        f'（如 {gone[0]["bucket"]}/{gone[0]["title"]}）')
    if moved:
        w, now = moved[0]
        problems.append(f'{len(moved)} 部锁定作品的 split 归属变了'
                        f'（如 {w["bucket"]}/{w["title"]}: {w["split"]} → {now}）')

    if problems:
        raise LockDriftError('锁定测试族校验失败（需重新冻结，且旧锁定族上跑过的数字全部作废）：\n  - '
                             + '\n  - '.join(problems))
    if notes:
        print('[i] ' + ' '.join(notes), file=sys.stderr)
    return {'works': len(works), 'lock_fingerprint': doc['lock_fingerprint'],
            'manifest': os.path.basename(man), 'chars': sum(w['chars'] for w in works),
            'manifest_changed_since_freeze': bool(notes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--per-bucket', type=int, default=20)
    ap.add_argument('--buckets', default=None, help='逗号分隔；默认全部桶')
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--report', default=None)
    ap.add_argument('--out', default=LOCK_PATH)
    args = ap.parse_args()

    buckets = [b.strip() for b in args.buckets.split(',')] if args.buckets else None
    if args.freeze:
        doc = freeze(args.per_bucket, buckets, args.manifest, args.report, args.out)
        by: Dict[str, int] = {}
        for w in doc['works']:
            by[w['bucket']] = by.get(w['bucket'], 0) + 1
        print(f'[*] 锁定测试族已冻结 -> {args.out}')
        print(f'    {len(doc["works"])} 部（' + ', '.join(f'{k}={v}' for k, v in sorted(by.items())) + '）')
        print(f'    lock_fingerprint={doc["lock_fingerprint"]}  '
              f'清单指纹（冻结时）={doc["manifest_fingerprint_at_freeze"]}')
        return 0
    if args.verify:
        r = verify(args.out, args.manifest, args.report)
        print(f'[*] 锁定测试族校验通过：{r["works"]} 部 / {r["chars"]} 字，'
              f'lock_fingerprint={r["lock_fingerprint"]}，清单={r["manifest"]}')
        return 0
    if args.show:
        doc = load_lock(args.out)
        by: Dict[str, int] = {}
        for w in doc['works']:
            by[w['bucket']] = by.get(w['bucket'], 0) + 1
        print(f'[*] 锁定测试族（{len(doc["works"])} 部）：'
              + ', '.join(f'{k}={v}' for k, v in sorted(by.items())))
        print(f'    lock_fingerprint={doc["lock_fingerprint"]}  protocol=v{doc["protocol_version"]}')
        return 0
    ap.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
