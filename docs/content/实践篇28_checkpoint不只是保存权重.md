# checkpoint 不只是保存权重

**系列：从零训练中文小说模型｜第 28 篇（实践篇·工程与性能）**
**前置阅读：原理篇 10（一次训练的账本：epoch、种子与 checkpoint）**

实验身份：脚本=`py/train.py`｜配置 ctx=32 d=40 layers=2 heads=4｜优化器 Adam lr=0.004 + warmup(15%)/cosine + clip 1.0｜JS 侧对照 `minilm.js`

「训练中断了能接着跑吗？」——我一度以为这个问题已经解决了，因为 `train.py` 有 `--resume`，也有 `checkpoint.pt`。

去看它到底存了什么之后，答案是：**没有。**

## 这个 checkpoint 里到底有什么

```python
torch.save({
    'model': model.state_dict(),   # 权重
    'itos':  itos,                 # 词表
    'meta':  meta,                 # 配置、seed、steps、当前 step、评测结果
}, path)
```

就这三样。缺的东西才是重点：

| 续训真正需要 | 是否保存 | 不保存的后果 |
|---|---|---|
| 权重 | ✅ | — |
| 词表 | ✅ | — |
| **Adam 的两个状态向量** | ❌ | 动量 m 与二阶矩 v 清零，重启后前几十步相当于换了优化器 |
| **PyTorch RNG 状态** | ❌ | 初始化/dropout 类随机性无法续上 |
| **批次采样的随机流状态** | ❌（用的是 `random.Random(seed)`） | 续跑会重放同一批数据顺序，而不是接着往下走 |
| **学习率调度的位置** | 半 ✅（meta 里有 step） | 但调度按 `lr_at(step, steps…)` 重算，恢复后要正确传起点，否则 warmup 会重来一遍 |

所以 `--resume` 现在的行为，日志里其实写得很诚实：

```text
Resuming from step N; no retraining in this simple script.
```

**它加载权重供评测，并不真的继续训练。** 危险之处在于：一个叫 `checkpoint.pt` 的文件、一个 `--resume` 参数，看起来太像「能续训」了。

## 这笔账有多大

以古典桶为例，词表 6,379 → 总参数 **295,960**。float32 下：

| 内容 | 体积 |
|---|---:|
| 权重 | ≈ 1.13 MB |
| 梯度 | ≈ 1.13 MB |
| Adam 的 m + v | ≈ **2.25 MB** |

也就是说，Adam 优化器状态几乎是权重的**两倍**。当前 checkpoint 只存了训练中常驻显存的约**三分之一**。**「保存了模型」和「保存了训练状态」差的就是这两份向量。**

## 浏览器版反而做对了一半

有意思的是，JS 教学版在「可复现」这件事上走得更前。`minilm.js` 把随机性拆成三条独立流（init / batch / sample），并导出纯数字状态：

```js
MiniLM.prototype.rngState    = function () { … }   // 三条流各自的状态
MiniLM.prototype.setRngState = function (st) { … }
```

`test_rng.js` 11 项断言里最关键的一条是：**训练中途插入 3 次生成之后，批次序列与损失序列仍与不生成时逐位相同。** 这正是续训能成立的前提——随机流可序列化、可隔离。

但 JS 侧也没有把优化器状态写进任何持久化格式。两边各缺了一半：**PyTorch 缺随机流与优化器状态，JS 缺持久化。**

## 一个能验收的清单

我把「真正的 checkpoint」定义成能回答这个问题：**用它恢复之后，接下来 10 步的损失序列是否和中断前继续跑一致？**

按这个标准，至少要存：

1. `model.state_dict()`；
2. `optimizer.state_dict()`（两份状态向量）；
3. `torch.get_rng_state()` + `numpy.random.get_state()` + `random.getstate()`；
4. 数据游标（下一个 batch 从哪里取，而不是靠 seed 重放）；
5. 调度位置（step、steps、当前 lr）；
6. 配置与词表（已有）。

第 4 条最容易被漏：**「重放随机流」和「续上批次」不是一回事**——前者会让模型重复看同一批片段，后者才是接着往下读。

## 边界

这不是 bug，是 `train.py` 明确标注过的简化实现（教学/短任务够用，400 步 12~64 秒，本来就很少中断）。**它之所以值得写一篇，是因为这个简化会被误用**：一旦语料上到千万字、训练跑到过夜，「我存了 checkpoint」就会变成「我以为我能续训」。

**可复现入口**：`py/train.py`（`save_checkpoint` / `--resume`）、`test_rng.js`（JS 三条流的隔离证明）。

---

复现：`python py/train.py --corpus scifi --steps 400 --out py/runs/ckpt_demo && python py/train.py --resume py/runs/ckpt_demo/checkpoint.pt`
原始结果：`py/train.py`（save_checkpoint 仅含 model/itos/meta）；`test_rng.js` 11/11
本篇没有证明：加了这些字段就能续训成功——恢复等价性还没被任何实验验证过
下一步：补齐 6 项字段，做「中断-恢复 vs 连续跑」的损失序列逐位对照

> 发布前补充（【人工项】）：若把这条实现掉，本文要改写成「修好了 + 逐位对照数据」，并附恢复前后的 loss 曲线。
