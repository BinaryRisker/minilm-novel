""" ============================================================================
py/license_policy.py — 数据用途分级与「可分发」闸门（V3 P0 的 allowlist / P4 的模型卡）
----------------------------------------------------------------------------
为什么不做成「加载时校验」：
  当前清单 192,823 部里 localugc 占 **191,679 部（99.4%）**，许可原文是自由文本，
  实际含义分三档：公版 / 明确开放许可 / **权利不清·仅本地学习用**。
  若把它塞进训练入口做门禁，只有两种结局：要么 99.4% 的数据进不了训练
  （V3 的学习曲线正要用它），要么放行（门禁等于没有）。
  它是**权限分类**，不是完整性校验 —— 真正的落点是 V3 P4 的导出与模型卡。

设计要点：**fail-closed**。识别不了的许可一律 `unknown`，unknown 一律不可分发。
宁可漏报成「不能发布」，也不能误报成「可以发布」。

分类（按优先级，先命中先赢；顺序即安全边界）：
  local_only          权利不清 / 仅本地学习用
  noncommercial       含 NC（如 CC BY-NC 4.0、CC BY-NC-SA4.0）
  attribution_sharealike  CC BY-SA（含「CC BY-SA / Public Domain」这类混合声明，取严）
  public_domain       公版 / Public Domain
  synthetic           CDLA 标注的合成数据（TinyStories 中文译文）
  attribution         CC BY / MIT / Apache（有明确署名条件）
  unknown             以上都不匹配（如「按仓库说明」）→ 不可分发

用法：
  python py/license_policy.py                    # 打印当前清单的分级统计
  python py/license_policy.py --bucket gudian    # 只看某桶
  from license_policy import summarise, assert_distributable
============================================================================ """
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, Iterable, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_MANIFEST = os.path.join(ROOT, 'data', 'manifest.jsonl')

# 可进入**可分发产物**（导出权重 / 公开发布的数字）的分类
DISTRIBUTABLE = frozenset({'public_domain', 'attribution', 'attribution_sharealike', 'synthetic'})
NOT_DISTRIBUTABLE = frozenset({'local_only', 'noncommercial', 'unknown'})

# 顺序敏感，且一律「取更严的那一档」：
#  · 「公版古籍，CC BY-NC-SA4.0 仓库」既含公版又含 NC → 判 noncommercial，不判 public_domain
#  · 「CC BY-SA 4.0 / Public Domain（维基文库）」同时声明两种 → 判 attribution_sharealike，
#    因为署名/相同方式共享的义务不能因为句子里还有个"Public Domain"就被抹掉
_RULES: List[tuple] = [
    ('local_only', ('权利不清', '本地学习', '本地文件', '不覆盖原著')),
    ('noncommercial', ('by-nc', 'nc 4.0', 'noncommercial', '非商业', '不得商用')),
    ('attribution_sharealike', ('by-sa', 'share-alike')),
    ('public_domain', ('public domain', '公版', '版权已过期')),
    ('synthetic', ('cdla', '合成数据')),
    ('attribution', ('cc by', 'mit', 'apache', '署名')),
]
# 说明：'按仓库说明' / '未明确标注' 这类**没有授权含义**的措辞不进 local_only，
# 而是落到 unknown —— 两者都不可分发，但 unknown 更诚实（"我不知道它是什么"≠"它仅限本地"）。
# 相应地，'CC BY-NC 4.0（按仓库说明）' 会命中 noncommercial（措辞里有明确 NC 授权）。


def classify(license_text: str) -> str:
    """把一段自由文本许可映射到用途分类；匹配不到 → unknown（fail-closed）。"""
    t = (license_text or '').lower()
    for cls, keys in _RULES:
        if any(k in t for k in keys):
            return cls
    return 'unknown'


def summarise(rows: Iterable[dict]) -> Dict[str, dict]:
    """按用途分类统计（部数 / 字数），并给出是否可分发。"""
    out: Dict[str, dict] = {}
    for r in rows:
        cls = classify(r.get('license'))
        e = out.setdefault(cls, {'works': 0, 'chars': 0, 'distributable': cls in DISTRIBUTABLE,
                                 'buckets': {}})
        e['works'] += 1
        e['chars'] += r.get('chars') or 0
        b = r.get('bucket') or 'unknown'
        e['buckets'][b] = e['buckets'].get(b, 0) + 1
    return out


def assert_distributable(rows: List[dict], what: str = '该产物') -> Dict[str, dict]:
    """闸门：只允许全部作品都落在 DISTRIBUTABLE 时通过，否则抛错并列出越界分类。
    供 P4 的 export / 模型卡与任何「可发布数字」的生成路径调用。"""
    s = summarise(rows)
    bad = {c: v for c, v in s.items() if c in NOT_DISTRIBUTABLE}
    if bad:
        detail = '; '.join(f'{c}={v["works"]}部/{v["chars"]}字' for c, v in sorted(bad.items()))
        raise ValueError(
            f'{what} 含**不可分发**的语料：{detail}。'
            f' 许可为「权利不清 / 仅本地学习 / unknown」的数据只能用于个人本地训练；'
            f' 导出权重或公开发布须剔除这些桶（见 docs/PRD_NOVEL_MODEL_LEARNING_V3.md §1 与本文件）。')
    return s


def load_rows(path: str) -> List[dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=DEFAULT_MANIFEST)
    ap.add_argument('--bucket', default=None)
    args = ap.parse_args()
    rows = load_rows(args.manifest)
    if args.bucket:
        rows = [r for r in rows if r.get('bucket') == args.bucket]
    s = summarise(rows)
    total = len(rows)
    print(f'[*] 清单 {args.manifest}{" 桶=" + args.bucket if args.bucket else ""} 共 {total} 部')
    for cls, e in sorted(s.items(), key=lambda kv: -kv[1]['works']):
        flag = '可分发' if e['distributable'] else '★不可分发★'
        print(f'    {cls:<24}{e["works"]:>7} 部 {e["chars"]:>11} 字  {e["works"]/total*100:5.1f}%  {flag}')
        print(f'        桶：' + ', '.join(f'{k}={v}' for k, v in sorted(e['buckets'].items())))
    ok = all(e['distributable'] for e in s.values())
    print(f'[*] 闸门判定：{"全部可分发" if ok else "存在不可分发分类 → 导出/发布须剔除"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
