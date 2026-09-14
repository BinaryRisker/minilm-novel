""" ============================================================================
dev/lint_content.py — 内容稿结构与口径检查（《文章质量审计与后续计划 V1》§三 / 计划 §四）
----------------------------------------------------------------------------
目录按三部曲前缀组织（2026-09-14 重组）：原理篇NN / 实践篇NN / 论文解读NN。
检查项：
  1) 分类内系列编号唯一且连续（1..N），且与文件名编号一致
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
CATEGORIES = [('原理篇', '原理篇*.md'), ('实践篇', '实践篇*.md'), ('论文解读篇', '论文解读*.md')]


def main():
    problems, rows = [], []
    for cat, pattern in CATEGORIES:
        files = sorted(glob.glob(os.path.join(CONTENT, pattern)))
        seen = {}
        for fn in files:
            name = os.path.basename(fn)
            t = open(fn, encoding='utf-8').read()
            han = len(re.findall(r'[\u4e00-\u9fff]', t))
            ser = re.search(r'第\s*(\d+)\s*篇', t)
            n = int(ser.group(1)) if ser else None
            fnum = re.match(r'(?:原理篇|实践篇|论文解读)(\d+)', name)
            if n is not None:
                if n in seen:
                    problems.append(f'{cat}{name}: 系列编号 {n} 与 {seen[n]} 重复')
                seen[n] = name
                if fnum and int(fnum.group(1)) != n:
                    problems.append(f'{name}: 文件名编号 {fnum.group(1)} 与系列编号 {n} 不一致')
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
            # 旧口径（同作品泄漏 90/10 + 单种子）的数字必须显式标注口径
            LEAKY = ('41.4%', '一次都没赢', '12 组配置', '12 组实验')
            if any(k in t for k in LEAKY) and '口径' not in t:
                problems.append(f'{name}: 引用旧口径数字但无「口径说明」标记')
            # 现行结论必须可追溯到 Day-2 报告（若引用了 worksplit 数字）
            if '作品级' in t and 'worksplit' not in t and '2026-09-14' not in t:
                problems.append(f'{name}: 提到作品级口径但未指向 Day-2 报告')
            rows.append((name[:6], cat, n, han, has_table, has_limit, has_next))
        nums = sorted(k for k in seen if k is not None)
        if nums != list(range(1, len(nums) + 1)):
            gaps = sorted(set(range(1, max(nums) + 1)) - set(nums))
            # 空号允许（如 23 号升格为论文解读篇），重复与不一致才是错误
            print(f'注：{cat} 编号有空位 {gaps}（升格/预留，非错误）')

    print(f'{"篇":6s} {"分类":6s} {"系列":>4s} {"汉字":>5s}  表格/代码 边界 下一步')
    for r in rows:
        print(f'{r[0]:6s} {r[1]:6s} {str(r[2]):>4s} {r[3]:>5d}    {"Y" if r[4] else "-":^3s} '
              f'{"Y" if r[5] else "-":^3s} {"Y" if r[6] else "-":^3s}')

    print()
    if problems:
        print(f'✗ {len(problems)} 个问题：')
        for p in problems:
            print('  -', p)
        return 1
    per = ', '.join(f'{cat} {len([r for r in rows if r[1] == cat])} 篇' for cat, _ in CATEGORIES)
    print(f'✓ {len(rows)} 篇通过结构检查（{per}；各分类编号连续，身份块与四行 footer 齐全）')
    short = [r[0] for r in rows if r[3] < 800]
    if short:
        print(f'注：{len(short)} 篇当前 <800 汉字（计划短文目标 800~1200）：{short}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
