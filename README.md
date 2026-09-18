# MiniLM Lab：从零看懂并动手训练语言模型

这个项目用一个可在浏览器训练的微型字符级 Transformer，展示文字怎样变成预测、误差怎样更新参数；再用 Python/PyTorch 学习数据处理、训练、存盘与评测。浏览器演示零安装；Python 路径需要依赖和数据。

**目标是学会原理并验证自己的小模型，不承诺训练几分钟就能写好小说。** 小模型与大规模自回归语言模型共享基础机制，规模扩大还需要数据、算力和工程验证。

## 新手从这里开始

1. [阅读地图：11 篇原理与按问题选读的实践](docs/content/README.md)。
2. [第一次训练：浏览器体验 → CPU 训练 → 保存恢复 → 阅读生成](docs/content/第一次训练_从浏览器到Python.md)。
3. [长篇入门指南](docs/NOVEL_MODEL_FROM_ZERO_BEGINNER_GUIDE.md)：数据、硬件、训练阶段与写作系统。
4. [当前实现与历史结论](docs/content/当前实现与历史结论.md)：避免把旧问题或未来方案当作现行功能。

## 立即体验

双击 `index.html`。选择语料，先生成，再训练，然后比较完整输出、loss 与 token 映射。无需联网，也不会自动读取磁盘上的小说。

从源码重建页面需要 Node.js：

```powershell
node build.js
node smoke.js
node uidom.js
node test_rng.js
```

改界面用 `ui-src.html`，改数学实现用 `minilm.js`；不要手改生成的 `index.html`。

## 截至 2026-09-17 的能力边界

| 路径 | 作用 | 状态 |
|---|---|---|
| 浏览器/JS | 看清前向、梯度、采样与注意力 | 教学实验台 |
| `py/train.py` | 小规模字符模型和恢复练习 | checkpoint v2；旧入口最终会读 test，练习只作探索 |
| `py/novellm/` | 分片数据、micro/t30m 候选、累积梯度、设备训练 | 已有实现；正式全量训练仍需数据与设备验收 |
| 故事能力 | 人物、因果、完整结尾 | 自动指标不足，需完整样本与人工盲评 |

t30m 是候选档名，不是固定参数量；当前仍复用 MiniLM 的绝对位置、LayerNorm、GELU。旧设计里的 RoPE 等结构尚未成为主实现。GPU 短训练记录见[交接文档](docs/HANDOFF_AUTODL_T30M_GPU_SMOKE_20260917.md)，本轮没有重跑 GPU。

训练耗时应由目标配置的实测有效 token/s 估计，再加验证与存盘。旧版“浏览器 17 年、GPU 3 小时、费用 4 元”的数字不能作为当前承诺，已从入口撤下。

## 资料与代码

| 入口 | 用途 |
|---|---|
| [现行训练手册](docs/FORMAL_30M_SCRATCH_TRAINING_RUNBOOK.md) | 用同一训练词表构建 train/val，分片短测与 GPU 前提 |
| [从零与微调比较](docs/SCRATCH_VS_FINETUNING_NOVEL_GUIDE.md) | 区分权重起点、数据和训练任务 |
| [评测协议](docs/EVAL_PROTOCOL_V1.md) / [人评协议](docs/NOVEL_HUMAN_EVAL_V1.md) | 验证集选型、最终测试与故事阅读 |
| [PRD V3](docs/PRD_NOVEL_MODEL_LEARNING_V3.md) / [V3.1 审查](docs/PRD_NOVEL_MODEL_LEARNING_V3_1_REVIEW.md) | 历史需求与设计，不等于已完成功能 |
| `py/minilm_torch.py` / `py/checkpoint.py` | 模型数学与统一存档契约 |
| `py/novellm/model.py` / `dataset.py` / `trainer.py` | 候选配置、分片与训练 |
| `docs/experiments/` / `py/runs/` / `LOG.md` | 保留日期和协议的实验记录 |
| [文章审查报告](docs/content/文章审查_2026-09-17.md) | 全库文档清点、逐篇处置与本轮验证 |

历史记录里的数据总量与切分不能当作今天的默认值。复现需固定代码、正文、清单、词表与协议；若默认清单回退或指纹漂移，先解决身份问题，不把结果当最终能力证据。
