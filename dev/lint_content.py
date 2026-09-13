""" ============================================================================
dev/lint_content.py — 内容稿结构与口径检查（《文章质量审计与后续计划 V1》§三 / 计划 §四）
----------------------------------------------------------------------------
检查项：
  1) 系列编号唯一且连续（1..N）
  2) 每篇有「实验身份：」行
  3) 每篇有 复现 / 原始结果 / 本篇没有证明 / 下一步 四行
  4) 短文要素（计划 §四）：有表格或输出块、有边界/限制、有下一步
  5) 汉字数（计划 §四 目标：短文 800~1200，长文 2000~4000）
  6) 禁止口径：把随机猜测当及格线（出现「随机猜测」时必须同时出现「基线」）
用法：python dev/lint_content.py
============================================================================ """
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(os.path.dirname(HERE), 'docs', 'content')

REQUIRED = ['复现：', '原始结果：', '本篇没有证明：', '下一步：']


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CONTENT, '文章*.md')))
             if re.match(r'文章\d\d', os.path.basename(f))]
    problems, rows = [], []
    seen = {}
    for fn in files:
        name = os.path.basename(fn)
        t = open(fn, encoding='utf-8').read()
        han = len(re.findall(r'[\u4e00-\u9fff]', t))
        ser = re.search(r'第\s*(\d+)\s*篇', t)
        n = int(ser.group(1)) if ser else None
        if n is not None:
            if n in seen:
                problems.append(f'{name}: 系列编号 {n} 与 {seen[n]} 重复')
            seen[n] = name
        if '实验身份：' not in t:
            problems.append(f'{name}: 缺「实验身份」行')
        miss = [k for k in REQUIRED if k not in t]
        if miss:
            problems.append(f'{name}: 缺 footer 行 {"".join(miss)}')
        has_table = t.count('|---') >= 1 or '```' in t
        has_limit = bool(re.search('边界|限制|没有证明|不证明|异常|失败', t))
        has_next = '下一步' in t
        if '随机猜测' in t and '基线' not in t:
            problems.append(f'{name}: 出现「随机猜测」但未提「基线」口径')
        rows.append((name[:4], n, han, has_table, has_limit, has_next))

    print(f'{"篇":4s} {"系列":>4s} {"汉字":>5s}  表格/代码 边界 下一步')
    for r in rows:
        print(f'{r[0]:4s} {str(r[1]):>4s} {r[2]:>5d}    {"Y" if r[3] else "-":^3s} '
              f'{"Y" if r[4] else "-":^3s} {"Y" if r[5] else "-":^3s}')

    nums = sorted(k for k in seen if k is not None)
    exp = list(range(1, len(nums) + 1))
    if nums != exp:
        problems.append(f'系列编号不连续：{nums}')

    print()
    if problems:
        print(f'✗ {len(problems)} 个问题：')
        for p in problems:
            print('  -', p)
        return 1
    print(f'✓ {len(rows)} 篇通过结构检查（编号连续 1..{len(nums)}，身份块与四行 footer 齐全）')
    short = [r[0] for r in rows if r[2] < 800]
    if short:
        print(f'注：{len(short)} 篇当前 <800 汉字（计划短文目标 800~1200）：{short}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
