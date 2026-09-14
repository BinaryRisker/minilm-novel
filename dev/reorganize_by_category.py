""" ============================================================================
dev/reorganize_by_category.py — 全库按三部曲前缀重组（2026-09-14，用户要求）
----------------------------------------------------------------------------
  文章31/32/33        → 原理篇01/02/03        （原理篇从 1 起编号，内容随后单独重写）
  文章01~22, 24~30    → 实践篇01~30           （编号不变，历史引用「第 N 篇」≡ 实践篇第 N 篇）
  文章23              → 论文解读01            （论文解读篇从 1 起编号）
同时：
  - 系列行加分类标记：第 N 篇（实践篇·栏目）/ 第 1 篇（论文解读篇·…）
  - 「前置阅读」行的 N 代号换算为原理篇编号（N0→1 … N10→11）
  - 正文交叉引用「第 N 篇」→「实践篇第 N 篇」（跳过系列行/前置阅读行）
幂等：已改过的行/文件自动跳过，可重跑复核。
用法：python dev/reorganize_by_category.py
============================================================================ """
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(os.path.dirname(HERE), 'docs', 'content')

RENAME = {f'文章{i:02d}' : f'实践篇{i:02d}' for i in range(1, 31)}
RENAME['文章23'] = '论文解读01'
RENAME.update({'文章31': '原理篇01', '文章32': '原理篇02', '文章33': '原理篇03'})

REF = re.compile(r'第 (\d+(?:[、，]\d+)*) 篇')
PREREQ_CODE = re.compile(r'原理篇 N(\d+)（')


def fix_body(line: str) -> str:
    if line.startswith('**系列：') or line.startswith('**前置阅读：'):
        return line
    return REF.sub(lambda m: f'实践篇第 {m.group(1)} 篇', line)


def main():
    renamed, edited = [], []
    for fn in sorted(glob.glob(os.path.join(CONTENT, '文章*.md'))):
        base = os.path.basename(fn)
        m = re.match(r'文章(\d\d)', base)
        if not m or ('文章' + m.group(1)) not in RENAME:
            continue
        new_base = RENAME['文章' + m.group(1)] + base[4:]
        new_fn = os.path.join(CONTENT, new_base)
        if not os.path.exists(new_fn):
            os.rename(fn, new_fn)
            renamed.append(f'{base} -> {new_base}')
        fn, name = new_fn, new_base
        if name.startswith('原理篇'):
            continue  # 原理篇内容随后整体重写，不在本脚本处理
        t = open(fn, encoding='utf-8').read()
        lines = t.split('\n')
        for i, ln in enumerate(lines):
            orig = ln
            if ln.startswith('**系列：'):
                ln = re.sub(r'｜第 (\d+) 篇（论文解读篇·',
                            r'｜第 \1 篇（论文解读篇·', ln)
                ln = re.sub(r'｜第 (\d+) 篇（(?!实践篇|论文解读篇)',
                            r'｜第 \1 篇（实践篇·', ln)
                ln = re.sub(r'｜第 (\d+) 篇\*\*$', r'｜第 \1 篇（实践篇）**', ln)
                if '论文解读' in name and '论文解读篇' not in ln:
                    ln = re.sub(r'｜第 (\d+) 篇（', r'｜第 1 篇（论文解读篇·', ln)
            elif ln.startswith('**前置阅读：'):
                ln = PREREQ_CODE.sub(lambda mm: f'原理篇 {int(mm.group(1)) + 1}（', ln)
            else:
                ln = fix_body(ln)
            if ln != orig:
                lines[i] = ln
                edited.append(f'{name}: {orig[:36]} -> {ln[:36]}')
        open(fn, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))

    print(f'改名 {len(renamed)} 个文件，改行 {len(edited)} 处')
    for r in renamed:
        print(' mv', r)
    for e in edited:
        print(' ~', e)


if __name__ == '__main__':
    main()
