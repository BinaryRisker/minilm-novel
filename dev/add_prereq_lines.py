""" ============================================================================
dev/add_prereq_lines.py — 按《内容重组_从0到1阅读路径_V1.md》§三 依赖表，
为 28 篇旧稿在「系列行」之后插入「前置阅读：」行（幂等：已有则跳过）。
【已被取代】2026-09-14 分类重组后，前置阅读行改用「原理篇 编号」格式，
见 dev/reorganize_by_category.py；本脚本保留仅作历史记录，重跑为 no-op。
用法：python dev/add_prereq_lines.py
============================================================================ """
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(os.path.dirname(HERE), 'docs', 'content')

N = {
    'N0': '原理篇 N0（全景：一个小说模型有哪些零件）',
    'N1': '原理篇 N1（字怎么变成数字：词表、OOV 与 UNK）',
    'N2': '原理篇 N2（字的向量：embedding）',
    'N3': '原理篇 N3（顺序从哪来：位置编码）',
    'N4': '原理篇 N4（读上下文：注意力与因果掩码）',
    'N5': '原理篇 N5（分数变概率：logits 与 softmax）',
    'N6': '原理篇 N6（训练信号：loss 与交叉熵）',
    'N7': '原理篇 N7（改参数：反向传播与优化器）',
    'N8': '原理篇 N8（写字：解码、采样与温度）',
    'N9': '原理篇 N9（一次训练的账本：epoch、种子与 checkpoint）',
    'N10': '原理篇 N10（评测与基线：留出集、Top-1、NLL 与 trigram）',
}

# 文章号 -> 前置章节（按《内容重组_从0到1阅读路径_V1.md》§二 P 表）
MAP = {
    '01': ['N1'],
    '02': ['N10'],
    '03': ['N10'],
    '04': ['N0'],
    '05': ['N6', 'N10'],
    '06': ['N8'],
    '07': ['N6'],
    '08': ['N9'],
    '09': ['N10'],
    '10': ['N9', 'N10'],
    '11': ['N8', 'N10'],
    '12': ['N0'],
    '13': ['N7'],
    '14': ['N0'],
    '15': ['N4'],
    '16': ['N4'],
    '17': ['N0'],
    '18': ['N8'],
    '19': ['N10'],
    '20': ['N10', 'N9'],
    '21': ['N3'],
    '22': ['N6'],
    '23': ['N10'],
    '24': ['N10'],
    '25': ['N10'],
    '26': ['N1', 'N2'],
    '27': ['N1'],
    '28': ['N9'],
    '29': ['N4', 'N10'],
    '30': ['N4'],
}

changed, skipped = [], []
for fn in sorted(glob.glob(os.path.join(CONTENT, '文章*.md'))):
    name = os.path.basename(fn)
    m = re.match(r'文章(\d\d)', name)
    if not m or m.group(1) not in MAP:
        continue
    num = m.group(1)
    t = open(fn, encoding='utf-8').read()
    if '前置阅读：' in t:
        skipped.append(name)
        continue
    lines = t.split('\n')
    idx = next((i for i, ln in enumerate(lines) if ln.startswith('**系列：')), None)
    if idx is None:
        print(f'!! {name}: 找不到系列行，跳过')
        continue
    prereq = '、'.join(N[k] for k in MAP[num])
    lines.insert(idx + 1, f'**前置阅读：{prereq}**')
    open(fn, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
    changed.append(f'{name} -> {MAP[num]}')

print(f'插入 {len(changed)} 篇，跳过（已有）{len(skipped)} 篇')
for c in changed:
    print(' +', c)
