""" ============================================================================
dev/protocol_note.py — Day-2（按作品切分）口径迁移：
  1) 给引用「旧口径」（桶内连续 90/10 + 单种子）数字的稿子加【口径说明】；
  2) 文章03 加后续指针（其结论未被推翻，只是已被延伸）；
  3) 身份行的 params=55,400 补注：主体恒 40,800，差额全是词表。
幂等（按 marker 跳过）。用法：python dev/protocol_note.py
============================================================================ """
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(os.path.dirname(HERE), 'docs', 'content')

NEW_DOC = 'docs/experiments/2026-09-14-worksplit-nll-converge.md'

NOTE = (
    '> **口径说明（2026-09-14 追加）**：本篇的数字来自**旧口径**——桶内连续 90/10 切分'
    '（同一部作品的头进训练、尾巴进留出，**同作品泄漏**）+ 单种子。后来做了受控 A/B：同一份训练、'
    '同一步数、同一种子，**只换留出集**，结果显示这个口径把 trigram 的 Top-1 平均抬高 **'
    '+33.5pt**、只把模型抬高 **+6.1pt**（**5.5 倍不对称**）——也就是说它**系统性偏向记忆型基线**，'
    '会把结论推向「模型打不过基线」。现行结论见 `{new}`：按作品切分后，'
    '**模型的留出集 NLL 从 1 万字起在 5/5 个桶全部胜过 trigram**，Top-1 则是混合的'
    '（`wuxia` 在 1 万~100 万字四档全胜）。本篇保留作历史对照，不代表当前能力判断。'
).format(new=NEW_DOC)

POINTER = (
    '> **后续（2026-09-14）**：本篇预告的万字级实验已经跑完，见 `{new}`。'
    '换成**按作品切分**的留出集后，交叉点落在 **2,000~10,000 字之间**：'
    '模型的留出集 NLL 自 1 万字起在 5/5 个桶全部胜过 trigram，Top-1 混合。'
    '本篇「500~2400 字还不够」这句话仍然成立，只是它已经不再是终点。'
).format(new=NEW_DOC)

# 需要加【口径说明】的稿子（正文引用了旧口径的对照数字）
STALE = ['文章02', '文章10', '文章11', '文章15', '文章19', '文章20', '文章22']
# 只需加后续指针
POINTERED = ['文章03']

PARAMS_OLD = '｜params=55,400｜'
PARAMS_NEW = '｜params=55,400（原始手写小语料；换成真实语料后总参数 11~30 万，Transformer 主体恒 40,800）｜'


def each(prefixes, block, marker):
    done = []
    for pre in prefixes:
        fn = next((f for f in os.listdir(CONTENT) if f.startswith(pre)), None)
        if not fn:
            print('✗ 找不到', pre)
            continue
        p = os.path.join(CONTENT, fn)
        t = open(p, encoding='utf-8').read()
        if marker in t:
            print('= 跳过', pre)
            continue
        import re as _re
        m = _re.search(r'实验身份：[^\n]*\n', t)
        if not m:
            print('✗ 无身份行', pre)
            continue
        i = m.end()
        t = t[:i] + '\n' + block + '\n' + t[i:]
        open(p, 'w', encoding='utf-8').write(t)
        done.append(pre)
    return done


def main():
    a = each(STALE, NOTE, '口径说明（2026-09-14')
    b = each(POINTERED, POINTER, '后续（2026-09-14）')

    n = 0
    for fn in os.listdir(CONTENT):
        if not fn.endswith('.md'):
            continue
        p = os.path.join(CONTENT, fn)
        t = open(p, encoding='utf-8').read()
        if PARAMS_OLD in t:
            open(p, 'w', encoding='utf-8').write(t.replace(PARAMS_OLD, PARAMS_NEW))
            n += 1
    print(f'✓ 口径说明 {len(a)} 篇：{a}')
    print(f'✓ 后续指针 {len(b)} 篇：{b}')
    print(f'✓ params 补注 {n} 篇')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
