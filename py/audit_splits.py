""" ============================================================================
py/audit_splits.py — 数据隔离审计（可执行门禁，非零退出码）
----------------------------------------------------------------------------
回答一个问题：**「留出集」到底隔没隔干净？** 五组检查，全部给数字：

  A 清单级：story_id 唯一 / split 合法 / 同一内容(hash)不跨集合 / 同名不跨集合 /
            母子作品(parent_story_id)不跨集合
  B 文本级：从 val、test 各抽 N 个 40 字片段，逐段在 train 全文里精确查找
            （字面重合 = 最硬的泄漏证据）
  C 词表级：只用 train 建词表时，val/test 的 OOV(映射到 UNK) 占比
  D 训练集内重复：同内容作品占训练作品比例（不等于泄漏，但会虚高有效数据量）
  E 旧口径对照：把 corpus.json（连续 90/10）拉来做同一套检查，量化它到底漏在哪 ——
            包括「有作品同时出现在两段」「test 段只含几部最短作品」「作品被截断」

判定：A/B 出现违规 → 退出码 1（门禁失败）；C/D/E 只报告不判负。

用法：
  python py/audit_splits.py                      # 审计 gudian/wuxia/urban/story
  python py/audit_splits.py --buckets gudian --probe 300
  python py/audit_splits.py --out dev/verify/audit_splits.json
============================================================================ """
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import splits as S  # noqa
from minilm_torch import build_vocab  # noqa

A_BUCKETS = ('gudian', 'wuxia', 'urban')
ALL_BUCKETS = ('gudian', 'wuxia', 'urban', 'story', 'scifi', 'xiandai')


# ---------------------------- A 清单级 ----------------------------

def check_manifest(rows: list) -> dict:
    ids = [r['story_id'] for r in rows]
    viol = []
    if len(ids) != len(set(ids)):
        viol.append('story_id 不唯一')
    for r in rows:
        if r['split'] not in S.SPLITS:
            viol.append(f'{r["story_id"]} split={r["split"]} 非法')

    by_hash, by_title, by_parent = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        by_hash[r.get('hash') or ''].append(r)
        by_title[r['title']].append(r)
        if r.get('parent_story_id'):
            idx = {x['story_id']: x for x in rows}
            p = idx.get(r['parent_story_id'])
            if p:
                by_parent[r['parent_story_id']].append(r)

    def cross(groups, label):
        out = []
        for k, v in groups.items():
            if not k:
                continue
            sp = {x['split'] for x in v}
            if len(sp) > 1:
                out.append({'key': k, 'splits': sorted(sp),
                            'works': [(x['story_id'], x['split']) for x in v][:6]})
        if out:
            viol.append(f'{label} 跨集合：{len(out)} 组')
        return out

    hash_cross = cross(by_hash, '同内容(hash)')
    title_cross = cross(by_title, '同名')
    parent_cross = cross(by_parent, '母子')
    return {'violations': viol, 'hash_cross': hash_cross, 'title_cross': title_cross,
            'parent_cross': parent_cross, 'works': len(rows)}


# ---------------------------- B 文本级 ----------------------------

def probe_leak(train_text: str, test_text: str, n: int, seed: int = 7, span: int = 40) -> dict:
    rng = random.Random(seed)
    if len(test_text) <= span + 1 or not train_text:
        return {'probes': 0, 'hits': 0, 'ratio': 0.0, 'examples': []}
    hits, ex = 0, []
    n = min(n, len(test_text) - span)
    for _ in range(n):
        i = rng.randrange(0, len(test_text) - span)
        s = test_text[i:i + span]
        if s in train_text:
            hits += 1
            if len(ex) < 3:
                j = train_text.find(s)
                ex.append({'test_off': i, 'train_off': j, 'snippet': s[:24]})
    return {'probes': n, 'hits': hits, 'ratio': hits / max(1, n), 'span': span,
            'seed': seed, 'examples': ex}


# ---------------------------- C 词表级 ----------------------------

def oov_report(train_text: str, texts: dict) -> dict:
    itos, stoi = build_vocab(train_text)
    out = {'vocab': len(itos)}
    for sp, t in texts.items():
        oov = sum(1 for ch in t if ch not in stoi)
        out[sp] = {'chars': len(t), 'oov': oov, 'ratio': oov / max(1, len(t))}
    return out


# ---------------------------- D 训练集内重复 ----------------------------

def train_dup(rows: list, bucket: str) -> dict:
    tr = [r for r in rows if r['bucket'] == bucket and r['split'] == 'train']
    c = defaultdict(list)
    for r in tr:
        c[r.get('hash') or r['story_id']].append(r['story_id'])
    dup_groups = {k: v for k, v in c.items() if len(v) > 1}
    return {'works': len(tr), 'unique': len(c), 'dup_groups': len(dup_groups),
            'dup_works': sum(len(v) for v in dup_groups.values()),
            'examples': [v[:5] for v in list(dup_groups.values())[:3]]}


# ---------------------------- E 旧口径对照 ----------------------------

def legacy_report(key: str, rows: list, corpus_path: str, span: int = 200, stride: int = 2000) -> dict:
    """量化 corpus.json 连续 90/10 的缺陷：作品跨段 / test 段作品数 / 被截断作品数。"""
    import math
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = json.load(f)[key]['text']
    cut = int(len(text) * 0.9)
    head, tail = text[:cut], text[cut:]
    works = S.bucket_texts(key)['train']
    both, only_tail, only_head, missing = [], [], [], 0
    for sid, title, txt in works:
        chunks = [txt[i:i + span] for i in range(0, min(len(txt), 80000), stride)
                  if len(txt[i:i + span]) == span]
        ih = any(ch in head for ch in chunks)
        it = any(ch in tail for ch in chunks)
        if ih and it:
            both.append({'story_id': sid, 'title': title})
        elif it:
            only_tail.append({'story_id': sid, 'title': title})
        elif ih:
            only_head.append({'story_id': sid, 'title': title})
        else:
            missing += 1
    kept_chars = len(text)
    full_chars = sum(len(t) for _, _, t in works)
    return {'corpus_chars': kept_chars, 'train_seg': len(head), 'test_seg': len(tail),
            'works_total': len(works), 'works_in_both_segments': both,
            'works_only_test': only_tail, 'works_only_train': len(only_head),
            'works_absent': missing,
            'truncation_loss': 1 - kept_chars / max(1, full_chars),
            'full_works_chars': full_chars}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--buckets', default=','.join(A_BUCKETS))
    ap.add_argument('--probe', type=int, default=200, help='每个 split 的片段探针数')
    ap.add_argument('--corpus', default=os.path.join(HERE, 'data', 'corpus.json'))
    ap.add_argument('--manifest', default=S.DEFAULT_MANIFEST,
                    help='要审计的清单（py/dedup_works.py 产出的 data/manifest_v2.jsonl 可直接审计）')
    ap.add_argument('--report', default=None,
                    help='与 --manifest 配套的切分报告（用 manifest_v2.jsonl 时传 data/split_report_v2.json）')
    ap.add_argument('--out', default=os.path.join(ROOT, 'dev', 'verify', 'audit_splits.json'))
    args = ap.parse_args()
    buckets = [b.strip() for b in args.buckets.split(',') if b.strip()]

    rows = S.read_manifest(args.manifest)
    rep = {'manifest_path': args.manifest, 'manifest': check_manifest(rows),
           'buckets': {}, 'legacy': {}}

    print('== A 清单级 ==')
    print(f"   作品 {rep['manifest']['works']} 部；违规 {len(rep['manifest']['violations'])} 条"
          f"{'：' + str(rep['manifest']['violations']) if rep['manifest']['violations'] else '（无）'}")
    print(f"   同内容跨集合 {len(rep['manifest']['hash_cross'])} 组 / "
          f"同名跨集合 {len(rep['manifest']['title_cross'])} 组 / "
          f"母子跨集合 {len(rep['manifest']['parent_cross'])} 组")

    for bk in buckets:
        t = S.bucket_texts(bk, manifest_path=args.manifest, report_path=args.report)
        entry = {'stats': {sp: {'works': len(t[sp]), 'chars': sum(len(x[2]) for x in t[sp])}
                           for sp in S.SPLITS}}
        train_text = '\n\n'.join(x[2] for x in t['train'])
        print(f'\n== 桶 {bk} ==  train {entry["stats"]["train"]["works"]} 部/'
              f'{entry["stats"]["train"]["chars"]:,} 字 · val {entry["stats"]["val"]["works"]} 部 · '
              f'test {entry["stats"]["test"]["works"]} 部')
        for sp in ('val', 'test'):
            txt = '\n\n'.join(x[2] for x in t[sp])
            r = probe_leak(train_text, txt, args.probe)
            entry[f'leak_{sp}'] = r
            print(f'   B 文本级 {sp}: {r["hits"]}/{r["probes"]} 个 {r["span"]} 字片段在 train 中出现 '
                  f'({100 * r["ratio"]:.2f}%)')
            if r['examples']:
                print(f'      例: {r["examples"][0]["snippet"]}…')
        texts = {sp: '\n\n'.join(x[2] for x in t[sp]) for sp in S.SPLITS}
        entry['oov'] = oov_report(train_text, {'val': texts['val'], 'test': texts['test']})
        print(f'   C 词表级 V={entry["oov"]["vocab"]}  '
              f'val OOV={100 * entry["oov"]["val"]["ratio"]:.3f}%  '
              f'test OOV={100 * entry["oov"]["test"]["ratio"]:.3f}%')
        entry['train_dup'] = train_dup(rows, bk)
        d = entry['train_dup']
        print(f'   D 训练集内重复: {d["dup_groups"]} 组 / 重叠 {d["dup_works"]} 部 '
              f'(唯一 {d["unique"]}/{d["works"]})')
        rep['buckets'][bk] = entry

    print('\n== E 旧口径（corpus.json 连续 90/10）==')
    for bk in buckets:
        if bk == 'story':
            continue
        try:
            lr = legacy_report(bk, rows, args.corpus)
        except Exception as e:
            print(f'   {bk}: 跳过（{type(e).__name__}: {e}）')
            continue
        rep['legacy'][bk] = lr
        print(f'   {bk}: 作品 {lr["works_total"]} 部 → 两段都有 {len(lr["works_in_both_segments"])} 部 '
              f'({[w["story_id"] for w in lr["works_in_both_segments"]]})，'
              f'test 段只有 {len(lr["works_only_test"])} 部、train 段 {lr["works_only_train"]} 部；'
              f'拼接后字数 {lr["corpus_chars"]:,} / 全池 {lr["full_works_chars"]:,} '
              f'→ 丢弃 {100 * lr["truncation_loss"]:.1f}%')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    fails = list(rep['manifest']['violations'])
    for bk, e in rep['buckets'].items():
        for sp in ('val', 'test'):
            if e[f'leak_{sp}']['hits'] > 0:
                fails.append(f'{bk}/{sp} 文本片段在 train 中出现 {e[f"leak_{sp}"]["hits"]} 次')
    print(f'\n[*] 报告 -> {args.out}')
    if fails:
        print('[*] 判定：★门禁失败★')
        for f_ in fails:
            print(f'    - {f_}')
        sys.exit(1)
    print('[*] 判定：通过（按作品切分的 val/test 与 train 无作品级/文本级重合）')


if __name__ == '__main__':
    main()
