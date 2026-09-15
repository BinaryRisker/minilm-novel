""" ============================================================================
py/audit_neardup.py — 可扩展的「近重复 / 同源」检测（行级倒排索引）
----------------------------------------------------------------------------
为什么需要它（原方法在数据涨 6 倍后失效）：
  py/dedup_works.py 走的是**全量两两比对**：每桶 n 部 → C(n,2) 对。
  gudian 55 部 = 1,485 对没问题；但 localugc 涨到 **191,679 部** 就是
  **1.83e10 对**，任何实现都跑不完。数据规模变了，算法必须换。

做法（倒排索引 + 候选对 + 双守卫）：
  1. 每部作品按行切分（行是论坛帖的天然单位：一楼/一条回复），
     只保留长度 ≥ --minline 的行，行内容用 blake2b(8B) 作指纹；
  2. 建**倒排索引** `行指纹 -> [story_id, ...]`，O(总行数) 而非 O(n²)；
  3. **双守卫**过滤假阳性（这是本脚本相对「朴素共享行计数」的全部价值）：
     · df 守卫：出现在太多作品里的行是套话/模板 → 丢弃。
       规则取两者之并：df > --max-df-abs **或** df > --max-df-frac × 作品数。
       （只用绝对阈值会在小桶上失效：xiandai 只有 35 部，wikisource 的
        版权声明模板行 df≈20 也过不了 500 的门槛，于是被当成泄漏。）
     · 最少共享数守卫：共享行数 < --min-shared 不判同源。
       （只用比例会在小作品上失效：17 行的《孔乙己》共享 3 行模板 = 0.176。）
  4. containment = 共享行数 / min(行数a, 行数b) ≥ --thresh 判同源；
  5. 并查集合并成簇，报告「跨 split 的簇」（= 真泄漏）与「桶内同源」。

与 py/splits.py 的关系：slug() 与 DEFAULT_PROCESSED 一律 import，**不重写**。
（踩过：探针里手抄 slug 规则，漏掉空格等字符 → 一批作品被静默跳过，
  数字偏低还看不出来。规则只能有一份实现。）

用法：
  python py/audit_neardup.py --buckets gudian,xiandai          # 快
  python py/audit_neardup.py --buckets localugc --max-df-abs 200
  python py/audit_neardup.py --buckets localugc --no-verify     # 清单正在被重写时
============================================================================ """
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import splits as S  # noqa

ALL_BUCKETS = ('gudian', 'wuxia', 'urban', 'xiandai', 'scifi', 'story', 'localugc')


def unit_hash(s: str) -> int:
    """行指纹（64bit int；比 bytes 省内存，dict 键也更省）。"""
    return int.from_bytes(hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest(), 'big')


def dump_path(args, bucket: str) -> str:
    """--dump-pairs 的每桶实际落点（多桶时不能互相覆盖）。"""
    root, ext = os.path.splitext(args.dump_pairs)
    return f'{root}.{bucket}{ext or ".jsonl"}'


def units_of(text: str, minline: int) -> dict:
    """返回 {行指纹: 行文本}（同一部内重复行只算一次）。"""
    out = {}
    for ln in text.split('\n'):
        ln = ln.strip()
        if len(ln) < minline:
            continue
        out.setdefault(unit_hash(ln), ln)
    return out


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:               # 路径压缩
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def groups(self):
        g = collections.defaultdict(list)
        for x in self.p:
            g[self.find(x)].append(x)
        return g


def analyze(bucket: str, rows: list, args) -> dict:
    works = [r for r in rows if r['bucket'] == bucket]
    if not works:
        raise ValueError(f'桶 {bucket} 在清单里没有作品')
    n = len(works)
    idx = collections.defaultdict(list)      # 行指纹 -> [story_id]
    lens = {}                                # 行指纹 -> 行长（只在 df≥2 时才记，省内存）
    nline = {}                               # story_id -> 该作的去重行数
    nchar = {}                               # story_id -> 该作去重行总字数
    missing = 0
    t0 = time.time()
    for k, r in enumerate(works, 1):
        p = os.path.join(args.processed, bucket, S.slug(r['title']) + '.txt')
        try:
            with open(p, 'r', encoding='utf-8') as f:
                txt = f.read()
        except FileNotFoundError:
            missing += 1                      # 只计数，不静默吞掉（报告里体现）
            continue
        us = units_of(txt, args.minline)
        nline[r['story_id']] = len(us)
        nchar[r['story_id']] = sum(len(l) for l in us.values())
        for h, l in us.items():
            if h in idx and h not in lens:    # 第二次见到 → 说明 df≥2，此时才记长度
                lens[h] = len(l)
            idx[h].append(r['story_id'])
        if args.progress and k % args.progress == 0:
            print(f'  ..{bucket} {k}/{n} 行指纹={len(idx):,} {time.time()-t0:.0f}s', flush=True)

    # ---------------- 双守卫 ----------------
    df_abs = args.max_df_abs
    df_frac = max(2, int(args.max_df_frac * n))
    skipped_abs = skipped_frac = 0
    usable = []
    for h, ids in idx.items():
        d = len(ids)
        if d < 2:
            continue
        if d > df_abs:
            skipped_abs += 1
            continue
        if d > df_frac:
            skipped_frac += 1
            continue
        usable.append((h, ids))

    # ---------------- 候选对：累计「共享行数」与「共享字数」 ----------------
    split_of = {r['story_id']: r['split'] for r in works}
    pair = {}
    for h, ids in usable:
        L = lens.get(h, 0)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                key = (a, b) if a < b else (b, a)
                v = pair.get(key)
                if v is None:
                    pair[key] = [1, L]
                else:
                    v[0] += 1
                    v[1] += L

    recs = []
    uf = UF()
    drop_guard_lines = drop_guard_chars = drop_thresh = 0
    for (a, b), (sh, sch) in pair.items():
        if sh < args.min_shared:
            drop_guard_lines += 1
            continue
        if sch < args.min_shared_chars:
            drop_guard_chars += 1
            continue
        na, nb = nline.get(a, 0), nline.get(b, 0)
        ca, cb = nchar.get(a, 0), nchar.get(b, 0)
        m, cm = min(na, nb), min(ca, cb)
        if not m or not cm:
            continue
        cont = sch / cm                     # 守卫 3 后的口径：**按字数**，不是按行数
        if cont < args.thresh:
            drop_thresh += 1
            continue
        recs.append({'a': a, 'b': b, 'shared': sh, 'shared_chars': sch,
                     'containment': round(cont, 4), 'cont_lines': round(sh / m, 4),
                     'lines_min': m, 'chars_min': cm,
                     'a_split': split_of.get(a), 'b_split': split_of.get(b)})
        uf.union(a, b)
    recs.sort(key=lambda x: (-x['containment'], -x['shared_chars']))

    # 可选：把**全部**过阈值对写成 JSONL。并查集只留下「簇」这一个聚合结果，
    # 簇会被传递性放大（A~B、B~C 但 A 与 C 只共享分隔线），要诊断这种链式放大
    # 就必须能看到逐条边。只在这里 dump，不进报告（报告会爆）。
    if args.dump_pairs:
        with open(dump_path(args, bucket), 'w', encoding='utf-8') as f:
            for x in recs:
                f.write(json.dumps({'bucket': bucket, **x}, ensure_ascii=False) + '\n')
        print(f'   [dump] {len(recs):,} 条过阈值对 -> {dump_path(args, bucket)}')

    clusters = []
    fulls = []                               # 与 clusters 同序的完整成员表（报告里只留前 40 个展示）
    for root, members in uf.groups().items():
        if len(members) < 2:
            continue
        fm = sorted(members)
        ms = set(fm)
        sub = [x for x in recs if x['a'] in ms and x['b'] in ms]
        sps = sorted({split_of.get(m) for m in fm})
        clusters.append({'members': fm[:40], 'size': len(fm),
                         'splits': sps, 'cross_split': len(sps) > 1,
                         'max_containment': max(x['containment'] for x in sub) if sub else 0})
        fulls.append(fm)
    order = sorted(range(len(clusters)), key=lambda i: (not clusters[i]['cross_split'],
                                                        -clusters[i]['size'],
                                                        -clusters[i]['max_containment']))
    clusters = [clusters[i] for i in order]
    fulls = [fulls[i] for i in order]
    # 跨 split 簇要**全量**留档：它是「必须移动的作品」清单，截断前 10 个就丢掉了大部分真泄漏。
    # 桶内同源簇只留展示用的前 10 个（它们不影响切分正确性，只影响训练集冗余）。
    cross_clusters = [{'size': clusters[i]['size'], 'splits': clusters[i]['splits'],
                       'members': fulls[i][:500], 'members_truncated': clusters[i]['size'] > 500,
                       'max_containment': clusters[i]['max_containment']}
                      for i in range(len(clusters)) if clusters[i]['cross_split']]

    cross = [x for x in recs if x['a_split'] != x['b_split']]
    out = {'bucket': bucket, 'algo': 'lines(倒排索引)', 'works': n, 'works_read': n - missing,
           'files_missing': missing,
           'distinct_units': len(idx), 'total_units': sum(nline.values()),
           'skipped_by_abs_df': skipped_abs, 'skipped_by_frac_df': skipped_frac,
           'df_abs': df_abs, 'df_frac': df_frac,
           'pairs_sharing_a_line': len(pair), 'pairs_pass': len(recs),
           'dropped_guard_min_lines': drop_guard_lines,
           'dropped_guard_min_chars': drop_guard_chars,
           'dropped_guard_thresh': drop_thresh,
           'pairs_cross_split': len(cross),
           'clusters': len(clusters), 'clusters_cross_split': sum(1 for c in clusters if c['cross_split']),
           'top_pairs': recs[:20], 'top_clusters': clusters[:10],
           'cross_split_clusters': cross_clusters,
           'seconds': round(time.time() - t0, 1)}
    _print_bucket(out)
    groups = {root: sorted(v) for root, v in uf.groups().items() if len(v) > 1}
    return out, groups


def analyze_anchors(bucket: str, rows: list, args) -> dict:
    """对齐无关的锚点法（O(n²)，只在作品数少的桶上可行）。

    为什么不自己再写一遍 winnowing：py/dedup_works.py 已经有一份，
    两处各写一遍就会「自己校验自己永远通过」。这里一律 import。
    """
    from dedup_works import anchors as _anchors, pair_overlap, exact_containment, UF as _UF
    works = [r for r in rows if r['bucket'] == bucket]
    if not works:
        raise ValueError(f'桶 {bucket} 在清单里没有作品')
    t0 = time.time()
    texts, missing = {}, 0
    for r in works:
        p = os.path.join(args.processed, bucket, S.slug(r['title']) + '.txt')
        try:
            with open(p, 'r', encoding='utf-8') as f:
                texts[r['story_id']] = f.read()
        except FileNotFoundError:
            missing += 1
    split_of = {r['story_id']: r['split'] for r in works}
    ids = sorted(texts)
    anch = {i: _anchors(texts[i], args.k, args.w, args.cap) for i in ids}

    # 守卫 1（锚点版）：df 过高的锚点是模板 / 套话，先从集合里剔掉。
    # 实测必要性：xiandai 的 wikisource 版权声明模板会让 11 部作品「锚点重叠 0.125、
    # 精确复核 0.143」——不加这道守卫，模板会被报成泄漏。
    df = collections.Counter()
    for i in ids:
        for h in anch[i]:
            df[h] += 1
    df_abs = args.max_df_abs
    df_frac = max(2, int(args.max_df_frac * len(ids)))
    dropped_anchor_df = 0
    for i in ids:
        before = len(anch[i])
        anch[i] = {h for h in anch[i] if df[h] <= df_abs and df[h] <= df_frac}
        dropped_anchor_df += before - len(anch[i])

    uf = _UF()
    cand = []
    dropped_exact = 0
    overlap_pass = 0                       # 过了锚点阈值、进入精确复核的对数
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            best, ab, ba = pair_overlap(anch[a], anch[b])
            if best < args.thresh:
                continue
            overlap_pass += 1
            ea = exact_containment(texts[a], texts[b], args.probe, args.exact_span)
            eb = exact_containment(texts[b], texts[a], args.probe, args.exact_span)
            ex = max(ea, eb)
            # 守卫 2（锚点版）：锚点重叠高但**逐字 40 字片段一个都对不上** → 短作品的
            # 锚点集太小，重叠是统计噪声。实测必要性：story 桶 10 个「簇」的精确复核
            # 全部为 0.0（作品中位 253 字），照单全收就会凭空造出 4 个跨 split 泄漏。
            if ex < args.min_exact:
                dropped_exact += 1
                continue
            cand.append({'a': a, 'b': b, 'containment': round(best, 4),
                         'a_to_b': round(ab, 4), 'b_to_a': round(ba, 4),
                         'exact_max': round(ex, 4),
                         'a_split': split_of.get(a), 'b_split': split_of.get(b)})
            uf.union(a, b)
    cand.sort(key=lambda x: (-x['containment'], -x['exact_max']))
    if args.dump_pairs:
        with open(dump_path(args, bucket), 'w', encoding='utf-8') as f:
            for x in cand:
                f.write(json.dumps({'bucket': bucket, **x}, ensure_ascii=False) + '\n')
        print(f'   [dump] {len(cand):,} 条过阈值对 -> {dump_path(args, bucket)}')
    clusters = []
    fulls = []
    for root, members in uf.groups().items():
        if len(members) < 2:
            continue
        fm = sorted(members)
        ms = set(fm)
        sub = [x for x in cand if x['a'] in ms and x['b'] in ms]
        sps = sorted({split_of.get(m) for m in fm})
        clusters.append({'members': fm[:40], 'size': len(fm), 'splits': sps,
                         'cross_split': len(sps) > 1,
                         'max_containment': max(x['containment'] for x in sub) if sub else 0,
                         'max_exact': max(x['exact_max'] for x in sub) if sub else 0})
        fulls.append(fm)
    order = sorted(range(len(clusters)), key=lambda i: (not clusters[i]['cross_split'],
                                                        -clusters[i]['size'],
                                                        -clusters[i]['max_containment']))
    clusters = [clusters[i] for i in order]
    fulls = [fulls[i] for i in order]
    cross_clusters = [{'size': clusters[i]['size'], 'splits': clusters[i]['splits'],
                       'members': fulls[i][:500], 'members_truncated': clusters[i]['size'] > 500,
                       'max_containment': clusters[i]['max_containment'],
                       'max_exact': clusters[i]['max_exact']}
                      for i in range(len(clusters)) if clusters[i]['cross_split']]
    cross = [x for x in cand if x['a_split'] != x['b_split']]
    out = {'bucket': bucket, 'algo': 'anchors(winnowing)', 'works': len(works),
           'works_read': len(texts), 'files_missing': missing,
           'anchors_total': sum(len(v) for v in anch.values()), 'anchors_dropped_df': dropped_anchor_df,
           'df_abs': df_abs, 'df_frac': df_frac,
           'pairs_overlap_pass': overlap_pass, 'dropped_by_min_exact': dropped_exact,
           'candidate_pairs': len(cand), 'pairs_pass': len(cand),
           'k': args.k, 'w': args.w, 'probe': args.probe, 'exact_span': args.exact_span,
           'min_exact': args.min_exact,
           'pairs_cross_split': len(cross), 'clusters': len(clusters),
           'clusters_cross_split': sum(1 for c in clusters if c['cross_split']),
           'top_pairs': cand[:20], 'top_clusters': clusters[:10],
           'cross_split_clusters': cross_clusters,
           'seconds': round(time.time() - t0, 1)}
    print(f"\n== {bucket} == [锚点法/对齐无关] 作品 {len(works):,}（读到 {len(texts):,}，缺 {missing}）"
          f" 锚点 {out['anchors_total']:,}（df 守卫剔 {dropped_anchor_df:,}，abs={df_abs}/frac={df_frac}）")
    print(f"   锚点重叠过阈 {overlap_pass:,} 对 → 精确复核({args.exact_span} 字片段×{args.probe}) "
          f"不足 {args.min_exact} 剔 {dropped_exact:,} → 剩 {len(cand):,}"
          f"（其中跨 split {len(cross):,}）→ 簇 {len(clusters)}"
          f"（跨 split {out['clusters_cross_split']}）  [{out['seconds']}s]")
    for cl in clusters[:8]:
        print(f"   簇 {cl['size']:>4} 部  跨split={cl['cross_split']}  splits={cl['splits']}  "
              f"锚点={cl['max_containment']} 精确={cl['max_exact']}  {cl['members'][:6]}")
    groups = {root: sorted(v) for root, v in uf.groups().items() if len(v) > 1}
    return out, groups


def _print_bucket(o: dict):
    print(f"\n== {o['bucket']} == 作品 {o['works']:,}（读到 {o['works_read']:,}，缺文件 {o['files_missing']}）"
          f" 去重行 {o['distinct_units']:,} / 总行 {o['total_units']:,}")
    print(f"   df 守卫丢弃：绝对阈值 {o['skipped_by_abs_df']:,} 行 / 占比阈值 {o['skipped_by_frac_df']:,} 行"
          f"（abs={o['df_abs']}，frac={o['df_frac']}）")
    print(f"   过守卫后仍共享 ≥1 行的对 {o['pairs_sharing_a_line']:,}："
          f"行数不足丢 {o['dropped_guard_min_lines']:,} / 字数不足丢 {o['dropped_guard_min_chars']:,} / "
          f"未达阈值丢 {o['dropped_guard_thresh']:,} → 剩 {o['pairs_pass']:,}"
          f"（其中跨 split {o['pairs_cross_split']:,}）→ 簇 {o['clusters']:,}"
          f"（跨 split 簇 {o['clusters_cross_split']:,}）  [{o['seconds']}s]")
    for c in o['top_clusters'][:6]:
        print(f"   簇 {c['size']:>5} 部  跨split={c['cross_split']}  splits={c['splits']}  "
              f"maxCont={c['max_containment']}  例 {c['members'][:3]}")
    for x in o['top_pairs'][:6]:
        print(f"   对 {x['a']} [{x['a_split']}] ↔ {x['b']} [{x['b_split']}]  共享 {x['shared']} 行/"
              f"{x['shared_chars']} 字  cont(字)={x['containment']} cont(行)={x['cont_lines']}")
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--buckets', default=','.join(ALL_BUCKETS))
    ap.add_argument('--manifest', default=S.DEFAULT_MANIFEST)
    ap.add_argument('--report', default=None, help='与 manifest 配套的切分报告（默认 data/split_report.json）')
    ap.add_argument('--processed', default=S.DEFAULT_PROCESSED)
    ap.add_argument('--minline', type=int, default=12, help='参与比对的最短行长度')
    ap.add_argument('--thresh', type=float, default=0.05, help='containment 判定阈值')
    ap.add_argument('--min-shared', type=int, default=5, help='最少共享行数（守卫 2）')
    ap.add_argument('--min-shared-chars', type=int, default=200,
                    help='最少共享字数（守卫 3）：短行（分隔线/空格式套话）撑不起比例')
    ap.add_argument('--max-df-abs', type=int, default=200, help='行 df 绝对上限（守卫 1a）')
    ap.add_argument('--max-df-frac', type=float, default=0.20, help='行 df 占作品数比例上限（守卫 1b）')
    ap.add_argument('--out', default=os.path.join(ROOT, 'dev', 'verify', 'audit_neardup.json'))
    ap.add_argument('--out-manifest', default=None,
                    help='可选：把同源簇整体重切分后写到这里（如 data/manifest_v3.jsonl），'
                         '并配套写 data/split_report_v3.json；规则复用 dedup_works.resplit')
    ap.add_argument('--out-moves', default=None,
                    help='可选：只写重切分的**增量**（移动了哪些作品，如 data/manifest_v3_moves.json）。'
                         '「源清单 + 增量」可确定性重建完整清单，体积小约 1,300 倍（64 KB vs 89 MB），'
                         '是推荐入库的形态；完整清单只在本地留档即可。')
    ap.add_argument('--no-verify', action='store_true', help='跳过切分身份校验（清单正在被重写时用）')
    ap.add_argument('--dump-pairs', default=None,
                    help='可选：把全部过阈值对写成 JSONL（路径会自动追加 .<桶> 后缀）。'
                         '诊断「并查集链式放大」必须看逐条边：A~B、B~C 不等于 A~C。')
    ap.add_argument('--progress', type=int, default=0, help='每 N 部打一次进度（0=不打）')
    ap.add_argument('--algo', choices=['auto', 'lines', 'anchors'], default='auto',
                    help='auto：作品数 ≤ --anchor-max 用锚点法（对齐无关，能抓同书异本），'
                         '否则用行级倒排（可扩展，但漏「断行不同」的同书异本）')
    ap.add_argument('--anchor-max', type=int, default=3000, help='auto 模式下改用行级法的作品数门槛')
    ap.add_argument('--k', type=int, default=20, help='锚点 k-gram 长度（锚点法）')
    ap.add_argument('--w', type=int, default=20, help='winnowing 窗口（锚点法）')
    ap.add_argument('--cap', type=int, default=0, help='每部参与比对字数上限（0=全文，锚点法）')
    ap.add_argument('--probe', type=int, default=2000, help='精确复核探针数（锚点法；默认与 dedup_works.py 一致）')
    ap.add_argument('--exact-span', type=int, default=40, help='精确复核片段长度（锚点法）')
    ap.add_argument('--min-exact', type=float, default=0.02,
                    help='锚点法的精确复核门槛（守卫 2）：锚点重叠够、但逐字片段复核低于此值 → 判噪声。'
                         '短作品的锚点集太小，重叠可以是统计噪声（story 桶 10 个簇精确复核全 0.0）。')
    args = ap.parse_args()

    src_fp = None
    if not args.no_verify and args.processed == S.DEFAULT_PROCESSED:
        try:
            v = S.verify(args.manifest, args.report or S.DEFAULT_REPORT)
            src_fp = v['fingerprint']
            print(f"[*] 切分身份校验通过：works={v['works']:,} 指纹={v['fingerprint']}")
        except S.SplitIdentityError as e:
            print(f'[*] 切分身份校验失败：\n{e}')
            sys.exit(1)

    rows = S.read_manifest(args.manifest)
    buckets = [b.strip() for b in args.buckets.split(',') if b.strip()]
    rep = {'protocol': {'minline': args.minline, 'thresh': args.thresh, 'min_shared': args.min_shared,
                        'min_shared_chars': args.min_shared_chars,
                        'max_df_abs': args.max_df_abs, 'max_df_frac': args.max_df_frac,
                        'algo': args.algo, 'anchor_max': args.anchor_max,
                        'anchors': {'k': args.k, 'w': args.w, 'cap': args.cap,
                                    'probe': args.probe, 'exact_span': args.exact_span,
                                    'min_exact': args.min_exact},
                        'manifest': args.manifest, 'buckets': buckets,
                        'source_manifest_fingerprint': src_fp,
                        'method': 'auto=小桶锚点法(对齐无关) / 大桶行级倒排(可扩展)；'
                                  '行级含三道守卫：df(绝对+占比) / 最少共享行数 / 最少共享字数；'
                                  '锚点法含两道守卫：锚点 df 过滤 / 精确复核门槛'},
           'buckets': {}}
    all_groups = {}
    for bk in buckets:
        n_bk = sum(1 for r in rows if r['bucket'] == bk)
        algo = args.algo
        if algo == 'auto':
            algo = 'anchors' if n_bk <= args.anchor_max else 'lines'
        print(f'\n[*] 桶 {bk}：作品 {n_bk:,} → {algo} 法'
              f'{"（注意：行级法会漏「同书异本/断行不同」）" if algo == "lines" else ""}')
        out_bk, groups_bk = (analyze(bk, rows, args) if algo == 'lines'
                             else analyze_anchors(bk, rows, args))
        rep['buckets'][bk] = out_bk
        all_groups.update(groups_bk)

    # ---------------- 可选：把同源簇整体重切分，产出新清单 ----------------
    if args.out_manifest or args.out_moves:
        from dedup_works import resplit
        out_rows, changed = resplit(rows, all_groups)
        fp_out = S.split_fingerprint(out_rows)
        stats = collections.defaultdict(lambda: collections.Counter())
        for nr in out_rows:
            stats[nr['bucket']][nr['split']] += 1

        # 增量清单：**只记移动的那些作品**。(源清单 + 本文件) 可确定性重建完整清单，
        # 信息量等价，而体积小约 1,300 倍（64 KB vs 89 MB）——不必往仓库里再塞一份
        # 与 v1 近似重复的 89 MB 副本。这也是「修复不要只落在旁支文件里」那条教训的落地。
        if args.out_moves:
            before = {r['story_id']: (r.get('orig_split') or r['split']) for r in rows}
            mv = [{'story_id': nr['story_id'], 'bucket': nr['bucket'],
                   'from': before.get(nr['story_id']), 'to': nr['split'],
                   'cluster_size': nr.get('dedup_cluster_size')}
                  for nr in out_rows if before.get(nr['story_id']) != nr['split']]
            os.makedirs(os.path.dirname(args.out_moves), exist_ok=True)
            with open(args.out_moves, 'w', encoding='utf-8') as f:
                json.dump({'generated_by': 'py/audit_neardup.py',
                           'note': '重切分**增量**（哪些作品换了 split）。'
                                   '配合 base_manifest 可确定性重建完整新清单：'
                                   '把每个 story_id 的 split 改成 to。',
                           'base_manifest': args.manifest,
                           'base_fingerprint': src_fp,
                           'target_fingerprint': fp_out,
                           'cluster_count': len(all_groups), 'moved': len(mv),
                           'moves': mv}, f, ensure_ascii=False, indent=2)
            print(f'[*] 移动清单 {args.out_moves}（{len(mv)} 部）')

        if args.out_manifest:
            os.makedirs(os.path.dirname(args.out_manifest), exist_ok=True)
            with open(args.out_manifest, 'w', encoding='utf-8') as f:
                for nr in out_rows:
                    f.write(json.dumps(nr, ensure_ascii=False) + '\n')
            rep2 = args.out_manifest.replace('manifest', 'split_report').replace('.jsonl', '.json')
            with open(rep2, 'w', encoding='utf-8') as f:
                json.dump({'generated_by': 'py/audit_neardup.py',
                           'note': '近重复（转发/同源）簇整体落在同一 split 后的切分；'
                                   '重切分规则与 py/dedup_works.py 共用一份实现（resplit）。',
                           'split_fingerprint': fp_out,
                           'total_works': len(out_rows), 'manifest': args.out_manifest,
                           'cluster_count': len(all_groups), 'cluster_moves': changed,
                           'protocol': rep['protocol']}, f, ensure_ascii=False, indent=2)
            print(f"[*] 新清单 {args.out_manifest}\n[*] 新报告 {rep2}")

        rep['resplit'] = {'out_manifest': args.out_manifest, 'moves': args.out_moves,
                          'cluster_count': len(all_groups), 'changed_works': changed,
                          'stats': {b: dict(v) for b, v in stats.items()},
                          'fingerprint': fp_out}
        print(f"\n[*] 重切分：{len(all_groups):,} 个簇 → 移动 {changed:,} 部")
        for b in buckets:
            st = stats.get(b, {})
            print(f"    {b}: " + ' '.join(f'{k}={st.get(k, 0)}' for k in S.SPLITS))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    tot_cross = sum(b['clusters_cross_split'] for b in rep['buckets'].values())
    print('\n== 汇总 ==')
    for bk, b in rep['buckets'].items():
        print(f"   {bk:<10} 作品 {b['works']:>7,}  跨split对 {b['pairs_cross_split']:>6,}  "
              f"跨split簇 {b['clusters_cross_split']:>4,}")
    print(f'[*] 报告 -> {args.out}')
    print(f'[*] 跨 split 同源簇合计 {tot_cross} 个'
          f'{"（★ 存在真泄漏，需处理）" if tot_cross else "（未发现）"}')


if __name__ == '__main__':
    main()
