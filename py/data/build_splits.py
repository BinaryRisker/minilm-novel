""" ============================================================================
py/data/build_splits.py — 按作品切分训练/验证/测试语料，消除 train↔test 泄漏
----------------------------------------------------------------------------
背景（本次自检发现的问题）：
  py/train.py、py/sweep_data_scale.py、py/baselines.py、py/minilm_torch.py 此前
  都从 py/data/corpus.json 的「整桶拼接文本」上按 `len*0.9` **连续切一刀** 当留出。
  整桶文本是「按字数从多到少把各部作品截断后 \\n\\n 拼接」，于是同一部作品的头进 train、
  尾巴进 test —— 留出集和训练集来自同一批原文，指标被高估。这正是 PRD 反复强调
  「train/val/test 不跨作品」要防的事。

本脚本把泄漏堵住，产出**按作品切好**的语料，供真实实验使用：
  - 读 data/manifest.jsonl（story_id / bucket / title / split，由 prepare.py --split 生成）
  - 读 data/processed/<bucket>/<slug(title)>.txt（dump_works 落的每部作品全文，非截断）
  - 按 split 把每部作品**整篇**拼进 data/splits/{train,val,test}.txt
  - 写 data/split_report.json 做审计（每 split 的部数/字数 + 唯一性 + 无跨集合断言）

与旧口径的分工：
  - `minilm_torch.train()` 的连续 90/10 切分**保留不动**，它是 L0 与 minilm.js 的
    逐位对齐路径，切分语义必须和 JS 一模一样；对齐结论只依赖那段。
  - 本脚本产出的按作品切分语料，是 L3 之后**做质量结论**时用的：sweep / eval / 正式训练
    改吃 data/splits/*.txt，指标才代表「对没见过的作品」的泛化。

用法：
  python py/data/build_splits.py                 # 用默认路径，含自检断言
  python py/data/build_splits.py --out-dir data/splits
============================================================================ """
from __future__ import annotations
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_MANIFEST = os.path.join(ROOT, 'data', 'manifest.jsonl')
DEFAULT_PROCESSED = os.path.join(ROOT, 'data', 'processed')
DEFAULT_OUT = os.path.join(ROOT, 'data', 'splits')
DEFAULT_REPORT = os.path.join(ROOT, 'data', 'split_report.json')

# 指纹算法只在 py/splits.py 定义一份：切分方与消费方必须用同一个函数，
# 各写一遍就会「自己校验自己永远通过」。
if os.path.join(ROOT, 'py') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'py'))
from splits import split_fingerprint, duplicate_hashes, read_manifest, slug  # noqa: E402

SPLITS = ('train', 'val', 'test')


def assert_family_dedupe(rows: list) -> None:
    """家族/内容去重必须已生效，否则拒绝切分：重复正文会虚增数据量，
    且两个副本一旦落进不同 split 就是 train/test 泄漏。"""
    dup = duplicate_hashes(rows)
    if dup:
        raise AssertionError(
            f'manifest 存在 {len(dup)} 个逐字重复作品（如 {dup[0][0]} vs {dup[0][1]}）；'
            f'请先跑 python py/data/prepare.py --split 完成内容去重，再切分')


def resolve_text(work: dict, processed_dir: str) -> str:
    """按 (bucket, slug(title)) 找回作品全文。找不到就报错，不静默跳过——
    静默跳过会让审计的字数对不上却看起来「跑通了」。"""
    cand = os.path.join(processed_dir, work['bucket'], slug(work['title']) + '.txt')
    if not os.path.exists(cand):
        raise FileNotFoundError(f'作品正文缺失：{cand}（{work["story_id"]} {work["title"]}）')
    with open(cand, 'r', encoding='utf-8') as f:
        return f.read()


def build(manifest_path: str, processed_dir: str, out_dir: str, report_path: str) -> dict:
    rows = read_manifest(manifest_path)

    ids = [r['story_id'] for r in rows]
    assert len(ids) == len(set(ids)), 'story_id 不唯一'
    for r in rows:
        assert r['split'] in SPLITS, f'{r["story_id"]} split={r["split"]} 非法（应为 train/val/test）'
    assert_family_dedupe(rows)   # 含重复正文的 manifest 一律拒绝切分

    buckets: dict = {sp: [] for sp in SPLITS}
    per_work = []
    for r in sorted(rows, key=lambda x: x['story_id']):  # 稳定顺序，可复现
        text = resolve_text(r, processed_dir)
        buckets[r['split']].append(text)
        per_work.append({'story_id': r['story_id'], 'bucket': r['bucket'],
                         'title': r['title'], 'split': r['split'], 'chars': len(text)})

    os.makedirs(out_dir, exist_ok=True)
    report_splits = {}
    for sp in SPLITS:
        # 每 split 的「字数」用逐作品 raw 之和（不含分隔符），才是可审计的作品字数；
        # 落盘文件因 \n\n 拼接会多出 (works-1)*2，另记 file_chars 供核对。
        raw_chars = sum(len(t) for t in buckets[sp])
        joined = '\n\n'.join(buckets[sp])
        with open(os.path.join(out_dir, f'{sp}.txt'), 'w', encoding='utf-8') as f:
            f.write(joined)
        report_splits[sp] = {'works': len(buckets[sp]), 'chars': raw_chars,
                             'file_chars': len(joined)}

    report = {
        'generated_by': 'py/data/build_splits.py',
        'note': '按作品整篇切分（train/val/test 不跨作品）。正文取 data/processed/ 每部全文，'
                '非 corpus.json 的截断拼接。chars 为逐作品原文字数之和，'
                'file_chars 含作品间 \\n\\n 分隔符。供 L3 之后的质量实验使用；'
                'minilm_torch.train 的连续 90/10 切分是 JS 对齐路径，与此无关。',
        'splits': report_splits,
        'split_fingerprint': split_fingerprint(rows),
        'total_works': len(rows),
        'total_chars': sum(v['chars'] for v in report_splits.values()),
        'work_chars_sum': sum(w['chars'] for w in per_work),
        'ok': sum(v['chars'] for v in report_splits.values()) == sum(w['chars'] for w in per_work),
    }
    assert sum(v['works'] for v in report_splits.values()) == len(rows), '作品数不守恒'

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=DEFAULT_MANIFEST)
    ap.add_argument('--processed', default=DEFAULT_PROCESSED)
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    ap.add_argument('--report', default=DEFAULT_REPORT)
    args = ap.parse_args()

    rep = build(args.manifest, args.processed, args.out_dir, args.report)
    print(f'[*] 按作品切分语料 -> {args.out_dir}/{{train,val,test}}.txt')
    for sp in SPLITS:
        v = rep['splits'][sp]
        print(f'    {sp:<5}: {v["works"]:>5} 部 / {v["chars"]:>10} 字')
    print(f'    合计  : {rep["total_works"]:>5} 部 / {rep["total_chars"]:>10} 字')
    print(f'[*] 审计 -> {args.report}  ok={rep["ok"]}（无跨作品泄漏，字数守恒）')


if __name__ == '__main__':
    main()
