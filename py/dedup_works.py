""" ============================================================================
py/dedup_works.py — 近重复作品检测 + 合并重切分（修「异名/异本/选本」泄漏）
----------------------------------------------------------------------------
背景（py/audit_splits.py 实测发现）：
  按作品切分**不等于**文本级隔离。gudian 桶里，留出作品的 40 字片段有
  12%~14.7% 能在训练集里原文找到，来源全是三类「同源不同名」：
    · 异名同书：《新齐谐》=《子不语》、《三国演义》=《三国志演义》
    · 选本重复：《今古奇观》是「三言二拍」的选本 → 《警世通言》《二刻拍案惊奇》
      的大段正文与它逐字相同
    · 同系列共用篇目：《初刻拍案惊奇》↔《二刻拍案惊奇》
  作品 id 不同、hash 不同、标题不同，只有**正文**能看出来。

做法：
  1. 每部作品取 40 字 shingle（步长 stride，最多 cap 字）→ 指纹集合
  2. 两两算重叠系数 overlap = |A∩B| / min(|A|,|B|)，≥ 阈值即判「同源」
  3. union-find 合并成簇：**一个簇必须整体落在同一个 split**
     （簇内含任何 train 作品 → 整簇进 train；否则按簇内最小 story_id 落到原 split）
  4. 产出新 manifest（data/manifest_v2.jsonl，不覆盖原文件）+ 簇报告

用法：
  python py/dedup_works.py --buckets gudian,wuxia,urban
  python py/dedup_works.py --buckets gudian --thresh 0.05 --stride 10 --cap 300000
============================================================================ """
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import splits as S  # noqa


def anchors(text: str, k: int = 20, w: int = 20, cap: int = 0) -> set:
    """内容决定的锚点集合（winnowing / 滑动窗口最小哈希）。

    为什么不用「固定步长抽样」（踩过两次）：
      · 两边各自 stride 抽样 → 同一段文字在两侧的起始位置对不齐，交集被稀释 1/stride
        （实测 38% 被压成 3.8%，直接漏检）；
      · 改成「对另一部全文精确查找」→ 准，但每对 ~1.25s，gudian 1485 对要半小时。
    winnowing 的锚点只由**内容**决定：同一段文字在任何文本里都产生同一批锚点，
    因此对齐无关；且用 crc32 做哈希，跨进程可复现。
    """
    t = text[:cap] if cap else text
    n = len(t)
    if n < k:
        return set()
    import zlib
    from collections import deque
    out, dq = set(), deque()
    prev = None
    for i in range(n - k + 1):
        h = zlib.crc32(t[i:i + k].encode('utf-8'))
        while dq and dq[-1][1] >= h:
            dq.pop()
        dq.append((i, h))
        if dq[0][0] <= i - w:
            dq.popleft()
        if i >= w - 1:
            out.add(dq[0][1])
    return out


def containment(a_anchors: set, b_anchors: set) -> float:
    """锚点级包含度：A 有多少比例的锚点也出现在 B 里。"""
    if not a_anchors:
        return 0.0
    return len(a_anchors & b_anchors) / len(a_anchors)


def pair_overlap(a_anchors: set, b_anchors: set) -> tuple:
    """返回 (max_containment, a→b, b→a)。选本场景只有单向高（选本 ⊃ 被选篇目），
    所以判同源取两个方向的最大值。"""
    ab = containment(a_anchors, b_anchors)
    ba = containment(b_anchors, a_anchors)
    return max(ab, ba), ab, ba


def exact_containment(a_text: str, b_text: str, n_probe: int = 2000, span: int = 40) -> float:
    """精确复核：对另一部全文逐字查找（慢，只用来确认候选对，不用于全量两两比对）。"""
    if len(a_text) <= span or not b_text:
        return 0.0
    step = max(1, (len(a_text) - span) // n_probe)
    hits = tot = 0
    for i in range(0, len(a_text) - span, step):
        s = a_text[i:i + span]
        tot += 1
        if s in b_text:
            hits += 1
    return hits / max(1, tot)


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def groups(self):
        g = defaultdict(list)
        for x in self.p:
            g[self.find(x)].append(x)
        return g


def resplit(rows: list, groups: dict) -> tuple:
    """把「同源簇」整体压到同一个 split，返回 (新清单, 移动作品数)。

    groups: {root: [member, ...]}（只含 ≥2 部的簇；单部作品保持原 split）。

    规则（两条都是刻意的）：
      · 簇内有 train → **整簇进 train**：保守优先，绝不把可能被训练见过的东西留在留出集；
      · 否则按簇内**最小 story_id** 的原 split：纯确定性，避免两次运行结果漂移。
    字段（**只在有信息量时写**，见下方注释）：
      orig_split（**首次**原始 split，只在它 ≠ 最终 split 时写；链式重跑 v2→v3 时不覆盖）、
      dedup_cluster / dedup_cluster_size（只在作品真在 ≥2 部的簇里时写）。

    抽成独立函数的原因：py/audit_neardup.py（行级倒排，可扩展到 19 万部）也要产出清单。
    规则若两处各写一遍，就会「自己校验自己」——同 py/splits.py 指纹那条教训。
    """
    root_of, size_of = {}, {}
    for root, members in groups.items():
        for m in members:
            root_of[m] = root
            size_of[m] = len(members)
    split_of = {r['story_id']: r['split'] for r in rows}
    changed, out_rows = 0, []
    for r in rows:
        sid = r['story_id']
        if sid in root_of:
            root = root_of[sid]
            members = groups[root]
            sps = [split_of[m] for m in members if m in split_of]
            new_split = 'train' if 'train' in sps else split_of.get(sorted(members)[0], 'train')
        else:
            root, new_split = sid, r['split']
        if new_split != r['split']:
            changed += 1
        nr = dict(r)
        nr['split'] = new_split
        size = size_of.get(sid, 1)
        # 只写**有信息量**的字段。192,823 部里只有约 2,500 部真的在簇里、427 部真的换了 split；
        # 无条件补三个字段会让清单凭空胖 15.3 MiB（实测 v3 104.79 MB，其中 96.4% 是这三个字段，
        # 且 99% 冗余——非簇成员的 dedup_cluster 恒等于自身 story_id、dedup_cluster_size 恒为 1）。
        first_split = r.get('orig_split') or r['split']
        if first_split != new_split:
            nr['orig_split'] = first_split          # 只有换过 split 的作品才需要记首次归属
        if size > 1:
            nr['dedup_cluster'] = root
            nr['dedup_cluster_size'] = size
        out_rows.append(nr)
    return out_rows, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--buckets', default='gudian,wuxia,urban')
    ap.add_argument('--manifest', default=S.DEFAULT_MANIFEST)
    ap.add_argument('--processed', default=S.DEFAULT_PROCESSED)
    ap.add_argument('--out-manifest', default=os.path.join(ROOT, 'data', 'manifest_v2.jsonl'))
    ap.add_argument('--report', default=os.path.join(ROOT, 'dev', 'verify', 'dedup_report.json'))
    ap.add_argument('--thresh', type=float, default=0.05,
                    help='重叠系数阈值：≥ 该值判为同源（0.05 = 较短作品有 5%% 的 40 字片段逐字相同）')
    ap.add_argument('--span', type=int, default=40)
    ap.add_argument('--k', type=int, default=20, help='锚点用的 k-gram 长度')
    ap.add_argument('--w', type=int, default=20, help='winnowing 窗口宽度（越大锚点越稀疏）')
    ap.add_argument('--verify', action='store_true', default=True,
                    help='对候选对再做「对全文逐字查找」精确复核（默认开）')
    ap.add_argument('--probe', type=int, default=2000,
                    help='精确复核时每对的等距探针数')
    ap.add_argument('--cap', type=int, default=0, help='每部作品参与比对的字数上限（0=全文）')
    args = ap.parse_args()

    buckets = [b.strip() for b in args.buckets.split(',') if b.strip()]
    rows = S.read_manifest(args.manifest)
    print(f'[*] manifest {len(rows)} 部；检查桶 {buckets}；阈值 {args.thresh}；'
          f'锚点 k={args.k}/w={args.w}（对齐无关，crc32 可复现）')

    report = {'protocol': {'thresh': args.thresh, 'span': args.span, 'k': args.k, 'w': args.w,
                           'probe': args.probe, 'cap': args.cap, 'buckets': buckets,
                           'method': 'winnowing 锚点包含度 + 候选对全文逐字复核'},
              'buckets': {}}
    uf = UF()

    for bk in buckets:
        raw = [(sid, title, txt if not args.cap else txt[:args.cap])
               for sp in S.SPLITS for sid, title, txt in S.bucket_texts(bk)[sp]]
        titles = {sid: t for sid, t, _x in raw}
        import time
        t0 = time.time()
        anch = {sid: anchors(txt, args.k, args.w, args.cap) for sid, _t, txt in raw}
        n = len(raw)
        cand = []
        for i in range(n):
            si = raw[i][0]
            for j in range(i + 1, n):
                sj = raw[j][0]
                best, ab, ba = pair_overlap(anch[si], anch[sj])
                if best >= args.thresh:
                    cand.append({'overlap': round(best, 4), 'a': si, 'b': sj,
                                 'a_to_b': round(ab, 4), 'b_to_a': round(ba, 4),
                                 'a_title': titles[si], 'b_title': titles[sj]})
                    uf.union(si, sj)
        # 候选对用「对全文逐字查找」精确复核（winnowing 是估计值，复核才有说服力）
        if args.verify:
            txts = {sid: t for sid, _t, t in raw}
            for c in cand:
                a_ex = exact_containment(txts[c['a']], txts[c['b']], args.probe, args.span)
                b_ex = exact_containment(txts[c['b']], txts[c['a']], args.probe, args.span)
                c['exact_a_to_b'] = round(a_ex, 4)
                c['exact_b_to_a'] = round(b_ex, 4)
                c['exact_max'] = round(max(a_ex, b_ex), 4)
        groups = {r: sorted(v) for r, v in uf.groups().items() if len(v) > 1}
        clusters = []
        for root, members in groups.items():
            if not all(m in titles for m in members):
                continue
            ms = set(members)
            pair = max([c for c in cand if c['a'] in ms and c['b'] in ms] or [{'overlap': 0}],
                       key=lambda c: c['overlap'])
            clusters.append({'members': members,
                             'titles': [titles[m] for m in members],
                             'splits': sorted({next(r['split'] for r in rows if r['story_id'] == m)
                                               for m in members}),
                             'max_overlap': pair['overlap'],
                             'top_pair': [pair.get('a'), pair.get('b')]})
        clusters.sort(key=lambda c: -c['max_overlap'])
        report['buckets'][bk] = {'works': len(raw), 'candidate_pairs': len(cand),
                                 'clusters': clusters,
                                 'top_pairs': sorted(cand, key=lambda c: -c['overlap'])[:12],
                                 'seconds': round(time.time() - t0, 1)}
        print(f'\n== {bk} == {len(raw)} 部，同源对 {len(cand)} 组，合并为 {len(clusters)} 个簇'
              f'（{report["buckets"][bk]["seconds"]}s）')
        for c in clusters[:8]:
            print(f'   重叠 {c["max_overlap"]:.3f} | {len(c["members"])} 部 | '
                  f'{"、".join(t[:10] for t in c["titles"][:5])} | 原 split={c["splits"]}')
        for c in report['buckets'][bk]['top_pairs'][:6]:
            ex = f'  精确复核 {c["exact_max"]:.3f}' if 'exact_max' in c else ''
            print(f'   对 {c["a_title"][:10]}→{c["b_title"][:10]}  锚点 {c["overlap"]:.3f}{ex}')

    # ---------------- 重新分配 split（簇必须整体一致） ----------------
    groups = {r: sorted(v) for r, v in uf.groups().items()}
    out_rows, changed = resplit(rows, groups)

    os.makedirs(os.path.dirname(args.out_manifest), exist_ok=True)
    with open(args.out_manifest, 'w', encoding='utf-8') as f:
        for nr in out_rows:
            f.write(json.dumps(nr, ensure_ascii=False) + '\n')

    # 配套切分报告：py/splits.py 的身份校验要求「manifest 指纹 == 报告里的指纹」，
    # 否则任何读取都会非零退出。这里给 v2 清单写一份自己的报告（不动原报告文件）。
    report_path = args.out_manifest.replace('manifest', 'split_report').replace('.jsonl', '.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_by': 'py/dedup_works.py',
            'note': '近重复（异名/异本/选本）合并后的切分：同一簇作品整体落在同一 split。'
                    '指纹算法与 py/splits.py 共用一份实现。',
            'split_fingerprint': S.split_fingerprint(out_rows),
            'total_works': len(out_rows),
            'manifest': args.out_manifest,
            'cluster_moves': changed,
        }, f, ensure_ascii=False, indent=2)

    # 新 manifest 的分桶统计
    stats = defaultdict(lambda: defaultdict(int))
    for nr in out_rows:
        stats[nr['bucket']][nr['split']] += 1
    report['result'] = {'changed_works': changed,
                        'out_manifest': args.out_manifest,
                        'stats': {b: dict(v) for b, v in stats.items()}}
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'\n[*] 移动作品 {changed} 部 → {args.out_manifest}')
    for bk in buckets:
        st = stats.get(bk, {})
        print(f'    {bk}: ' + ' '.join(f'{k}={st.get(k, 0)}' for k in S.SPLITS))
    print(f'[*] 报告 -> {args.report}')


if __name__ == '__main__':
    main()
