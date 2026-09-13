""" dev/finalize_content.py — 收尾一致性修订（幂等）。"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(os.path.dirname(HERE), 'docs', 'content')


def sub(prefix, old, new, marker=None):
    fn = next(f for f in os.listdir(CONTENT) if f.startswith(prefix) and f.endswith('.md'))
    p = os.path.join(CONTENT, fn)
    t = open(p, encoding='utf-8').read()
    if marker and marker in t:
        print('= 跳过', prefix)
        return
    if old not in t:
        print('✗ 未命中', prefix)
        return
    open(p, 'w', encoding='utf-8').write(t.replace(old, new, 1))
    print('✓', prefix)


# 1) 发布说明里“待跑”已过期
sub('文章发布说明',
    '附一个待跑的「pos 置零」对照；',
    '附已跑完的「pos 置零」对照（结论：收益不显著）；',
    marker='已跑完的「pos 置零」')

# 2) 文章09 补一句四舍五入说明，消除与 34.0% 的表面冲突
sub('文章09',
    '## 协议变了哪里',
    '''**一个容易困惑的地方**：本系列其它篇里出现的「科幻 34.0%」，就是这个 33.98% 的四舍五入。它们不是两次实验，是同一个数。

## 协议变了哪里''',
    marker='就是这个 33.98% 的四舍五入')

# 3) 审计文档：可复现性从「待补」改为已补
sub('文章质量审计与后续计划_V1',
    '| 可复现性 | 待补 | 多数文章有入口，但缺统一的“版本 / seed / 命令 / 输出路径”信息块。 |',
    '| 可复现性 | 已补（2026-09-13） | 23 篇均加「实验身份」行 + 四行 footer（复现 / 原始结果 / 本篇没有证明 / 下一步）；全文 800~1200 汉字、篇篇含表格或输出块，由 `dev/lint_content.py` 校验。 |',
    marker='dev/lint_content.py` 校验')

# 4) 审计文档追加本轮修订记录
p = os.path.join(CONTENT, '文章质量审计与后续计划_V1.md')
t = open(p, encoding='utf-8').read()
if '## 七、本轮修订记录' not in t:
    t = t.rstrip() + '''

## 七、本轮修订记录（2026-09-13）

按 §三 规范对 23 篇统一施工，全部由脚本完成，可重跑复核：

- `dev/apply_content_meta.py`：为 23 篇插入「实验身份」行与四行 footer；
- `dev/expand_content.py` / `expand_content2.py` / `expand_content3.py`：补真实数据块（表格/样本），把偏短的稿子补进 800~1200 汉字区间；
- `dev/lint_content.py`：结构校验（编号连续、身份块、四行 footer、表格或输出块、字数区间），当前 23/23 通过；
- 逐条落实 §二 点名的缺口：文章 13 补有限差分步长（`eps=1e-5`）、容差公式与配置；文章 21/16 换成本轮消融的实测数字；文章 01/02/03/09/10 补口径说明。

仍未完成、需要人工的部分：署名、平台摘要、真实截图、commit 绑定、完整失败样本的人工逐句批注。
'''
    open(p, 'w', encoding='utf-8').write(t)
    print('✓ 审计文档追加修订记录')
print('done')
