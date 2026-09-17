""" dev/scope_seed.py — Day-2 发现二（种子 σ 随数据量档剧变）回填到 08/16/21。幂等。"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(os.path.dirname(HERE), 'docs', 'content')

ABL_ANCHOR = '完整数据与边界见 `docs/experiments/2026-09-13-ablation-heads-pos.md`，脚本 `py/ablation.py`。'

ABL_QUAL = '''**一个必须加的限定（2026-09-14）**：本对照做在 2,401 字与 20,000 字两档。Day-2 的多种子数据显示，**种子噪声随数据量档剧变**——2,000 字档 Top-1 的 σ 是 ±0.2~1.5pt，而 1,000,000 字档只有 **±0.1pt**（1pt 差异 ≈ 10σ）。

也就是说：「差异不超过种子波动」这句话**只在小数据档成立**。到了大数据档，1pt 级的头数差异就是真实的，本实验没有在那个档重跑。

'''

JOBS = [
    ('文章08', '## 一个更重要的反转', '种子噪声不是常数（2026-09-14 修正）', '''## 但种子噪声不是常数

这条判断后来被更大数据量修正过一次。同一批实验（按作品切分、3 epoch、3 种子）在不同数据量档上的种子标准差是这样的：

| 训练字数档 | Top-1 种子 σ | NLL 种子 σ |
|---|---|---|
| 2,000 字 | ±0.2 ~ ±1.5pt | ±0.08 ~ ±0.30 |
| 1,000,000 字 | **±0.1pt** | ±0.008 ~ ±0.033 |

**结论要分档读，不能一句「单次不可信」走完天下：**

- 在**小数据档**，1pt 级的胜负确实分不清是差距还是噪声——本篇武侠 24.0% vs 均值 19.8% 就是这一档的事故；
- 在**百万字档**，1pt 大约等于 10σ。这时候还用「反正种子会波动」去回避 0.5pt、0.8pt 的差异，就是另一种不负责任了。

所以我现在写结论时会同时给三样东西：**数据量档、σ、以及这个差值相当于几倍 σ**。少了任何一样，「显著」两个字都没有意义。

'''),
    ('文章16', ABL_ANCHOR, '种子噪声不是常数', ABL_QUAL),
    ('文章21', ABL_ANCHOR, '种子噪声不是常数', ABL_QUAL),
]


def main():
    for prefix, anchor, marker, block in JOBS:
        fn = next((f for f in os.listdir(CONTENT) if f.startswith(prefix)), None)
        p = os.path.join(CONTENT, fn)
        t = open(p, encoding='utf-8').read()
        if marker in t:
            print('= 跳过', prefix)
            continue
        if anchor not in t:
            print('✗ 锚点未命中', prefix)
            continue
        open(p, 'w', encoding='utf-8').write(t.replace(anchor, block + anchor, 1))
        print('✓', prefix)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
