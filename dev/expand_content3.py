""" dev/expand_content3.py — 第三轮微调：把最后 5 篇补到短文下限。真实数据，幂等。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(os.path.dirname(HERE), 'docs', 'content')

JOBS = [
    ('文章01', '## 这件事为什么重要', '一个词的代价',
     '''顺带说清一个容易被忽略的代价：**词表越小，陌生字越多。** 三段原创语料的唯一字符数分别是 364 / 304 / 323，也就是说词表只有三百多。任何一段没见过的现代生活用语——冰箱、电脑、警察——都会立刻撞在词表边界上。这不是 bug，是字符级方案在这个语料规模下的固有属性。

'''),
    ('文章02', '## 这是不是说明 Transformer 没用', '两个指标讲的是两件事',
     '''还有一条经验值得单独写出来：这次比较用的是**同一份留出位置、同一份可见上文、同一个词表**。只要其中一项不一致，比如基线多看一个字，结论就直接作废——而且它不会报错，只会安静地给出一个错的差值。所以评测协议本身要先于评测结果被信任。

'''),
    ('文章03', '**配图建议**', '为什么不能只看字数',
     '''还有一个陷阱值得提醒：**trigram 不是固定基线。** 很多人把基线当成一把不动的尺子，实际上它也在吃同一批训练字——数据越多，它记住的三元组合也越多。所以「模型随数据提升」这句话本身没有意义，有意义的说法是「模型的提升有没有跑赢基线的提升」。这次扫描里，没有跑赢。

'''),
    ('文章09', '## 结论边界', '一个可执行的习惯',
     '''所以从这次之后，我给自己定了一条很机械的习惯：**任何评测结果落盘时，同时记下协议实现的文件路径与函数名**（例如 `eval/protocol.js::forEachPosition`）。不写版本的协议，等于没有协议。

'''),
    ('文章10', '## 这不等于「Transformer 没用」', '下一步该看什么',
     '''判断到底是哪一种，有个很省事的分法：**看训练 loss 与留出 NLL 的裂口有多大。** 裂口大（本例 1.3~1.6 对 3.5~5.7）说明数据不够、模型在背；裂口小但两个都高，说明模型容量不够、连训练集都没吃下。这两种情况的药方相反——前者加数据，后者加容量。把它们混成一句「过拟合」，就会在错误的方向上调参。

'''),
]


def main():
    for prefix, anchor, marker, block in JOBS:
        fn = next(f for f in os.listdir(CONTENT) if f.startswith(prefix) and f.endswith('.md'))
        path = os.path.join(CONTENT, fn)
        t = open(path, encoding='utf-8').read()
        if marker in t:
            print(f'= 跳过 {prefix}')
            continue
        if anchor not in t:
            print(f'✗ 锚点未命中 {prefix}')
            continue
        t = t.replace(anchor, block.strip() + '\n\n' + anchor, 1)
        open(path, 'w', encoding='utf-8').write(t)
        print('✓', prefix)
    return 0


if __name__ == '__main__':
    sys.exit(main())
