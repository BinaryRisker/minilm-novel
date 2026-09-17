# 30M 随机初始化模型训练手册

本手册是 V3.1 的第一次正式训练方案。它不使用任何外部模型权重、LoRA、蒸馏权重或外部 tokenizer。当前仓库尚未完成 `py/novellm/` 训练器；下面的命令在实现 P1 后才可执行，现有 `py/train.py` 只能用于旧教学模型。

## 训练目标

第一阶段不是“直接写长篇”，而是训练一个能可靠学习中文、完成短故事并保存完整谱系的因果语言模型。首个可验收目标是 300–800 字有结尾的短故事；只有它通过人工 G1，才进入规划和多场景训练。

## 模型配置

建议首个候选：

```yaml
vocab_size: 8192          # 只用 train 拟合的 tokenizer
d_model: 512
n_layers: 8
n_heads: 8
ffn_hidden: 1408
context_length: 1024
norm: rmsnorm
activation: swiglu
position: rope
tie_embeddings: true
bias: false
init: random               # root checkpoint 必须记录 seed 和 step-0 hash
```

在实现固定后由代码断言参数量为 29,893,120；不能只看配置文件中的估算。先不同时引入 GQA、稀疏 MoE、Muon、超长上下文或 RL，避免无法归因。

## 数据准备

1. 从 `manifest_v3.jsonl` 和配套 `split_report_v3.json` 建立不可变数据版本。正文按实际 SHA-256 校验，作品家族、改写和近重复只能出现在一个 split。
2. tokenizer 只在 train 拟合；val/test 不参与词表和模板构造。记录字符/token 比、特殊 token 往返还原、未知率和每个作品 token 数。
3. 每个文档末尾加入 EOS；BOS、EOS、PAD、UNK 使用不同 ID。训练 batch 不能跨文档拼接，除非 packing 明确记录边界和 loss mask。
4. 先建三个分域验证集：通用中文、完整故事、未见题材组合。正式 test 由 `final_eval` 单独加载，训练和选型入口不能读取。
5. 目标故事比例做小网格：25%、50%、75%。每格固定总 token 预算和相同作品池，记录每个作品及近重复簇的曝光次数。不要把重复读取当作新数据。

## 首次运行顺序

```powershell
python -X utf8 -m py.novellm.doctor --out runs/doctor_target.json
python -X utf8 -m py.novellm.prepare_data --manifest data/manifest_v3.jsonl --report data/split_report_v3.json --out runs/data_v1
python -X utf8 -m py.novellm.fit_tokenizer --data runs/data_v1 --vocab-size 8192 --out runs/tok_v1
python -X utf8 -m py.novellm.smoke --config configs/novellm_30m.yaml --tokenizer runs/tok_v1 --out runs/smoke_30m
```

`smoke` 必须在临时目录中完成：随机初始化、前向、反向、保存、杀进程后恢复、生成一段文本、离线加载。它通过前不能正式训练。

然后先跑 200 步设备基准：

```powershell
python -X utf8 -m py.novellm.benchmark --config configs/novellm_30m.yaml --tokenizer runs/tok_v1 --contexts 512,1024 --micro-batch 1 --grad-accum 8 --steps 200 --out runs/bench_30m
```

报告有效 token/s、峰值显存、CPU RSS、验证 loss、NaN/Inf、每次 checkpoint 时间。`--grad-accum` 增加有效 batch，不改变单步显存；显存不足时先降 context，再降模型宽度，不通过外部量化权重解决。

## 正式阶段

### L0：语言学习

```powershell
python -X utf8 -m py.novellm.train --config configs/novellm_30m.yaml --tokenizer runs/tok_v1 --data runs/data_v1 --stage lm --tokens 100000000 --device cuda:0 --micro-batch 1 --grad-accum 8 --eval-every 2000 --save-every 2000 --out runs/scratch30m_lm_100m
```

检查点必须保存模型、optimizer、scheduler、所有 RNG、数据游标、累计 token、配置/代码/数据/tokenizer 指纹。训练中只看 train/dev；test 不读。

### L1：完整短故事课程

从 L0 最佳 checkpoint 继续，不重新初始化：先混合 300–800 字故事，再加入 800–1500 字场景。每个文档带 EOS；生成评测固定 30 个开发题，保存全部样本和失败原因。

```powershell
python -X utf8 -m py.novellm.train --resume runs/scratch30m_lm_100m/checkpoints/best.pt --config configs/novellm_30m.yaml --stage story --tokens 200000000 --out runs/scratch30m_story_200m
```

这里的 `tokens` 是本阶段新增曝光量；报告 unique tokens、总曝光量和重复分布，不能写成新增语料量。

### L2：层级任务

在自己的 L1 checkpoint 上训练题目→大纲、状态→场景、正文→状态摘要。计划和状态是有证据的监督目标；写正文时只对正文 token 计算 loss。正确计划/正确状态的上限实验与自主规划实验分开。

### L3：多场景短篇

先生成 3 个相连场景，再扩大到 6–10 个场景、6000–10000 字。每个场景提交 `state_delta` 前必须带正文证据；修订最多两次并记录修复数、新增错误数和总 token。direct、planner、state、revision 四组使用相同题目和预算。

## 8GB 显存策略

先按 micro-batch=1、context=512/1024、gradient accumulation=8 测量。若完整 BF16 训练不稳定，使用 BF16 激活加 FP32 optimizer/master 权重；启用 activation checkpointing。不要把理论权重大小当峰值显存。每次改变 context、精度、累计步数都建立新 run identity。

30M 模型的参数、梯度和 Adam 状态本身通常不是 8GB 的主要压力，激活、临时 logits、数据缓存和框架分配才是；具体容量只以目标机 benchmark 为准。4090 显存容量必须由 doctor 实测确认，Mac mini M4 用于 CPU/MPS 兼容性和推理回归，不能把两台设备拼成一个训练显存池。

## 何时停止或扩大

- 200 步 benchmark OOM：先降 context 512；仍 OOM 才降宽度或改 grad accumulation。
- train loss 降、故事 dev 不升：优先检查数据重复、文档边界和故事比例，不直接加层数。
- 故事 dev 有提升但 30 个完整样本仍无结尾：增加完整故事课程或降低题目复杂度，不进入长篇 pipeline。
- L1 通过且 30M 曲线仍稳定改善：再评估 300M 曝光；只有新增有效数据和人评都支持时才训练 100M 候选。
- 任何阶段发现身份、恢复或测试污染：作废该 run，保留原始文件，修复后新建数据/协议版本。

## 训练后的交付

导出模型、tokenizer、配置、锁定依赖、根初始化信息、数据用途/许可、全部 checkpoint 指纹、30 个开发样本、失败样本、自动指标和人评导入模板。没有完整短篇真人记录时，模型卡状态写“待人评”，不能称为小说模型已完成。
