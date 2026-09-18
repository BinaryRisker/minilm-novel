# 小说模型优化执行清单 V1

> **2026-09-17 阅读说明**：需求/规划与当时验收记录，勾选状态不自动等于当前实现。现行代码边界和未实现模块见[状态说明](content/当前实现与历史结论.md)，新手不要从本文件复制设想命令。

日期：2026-09-13。配套设计：PRD_NOVEL_MODEL_LEARNING_V1.md。以下均为待实施，不表示已开发。路径均相对项目根目录；新脚本命令是计划接口。

## 前提

- [ ] 保留原报告与历史 dev/out_*，新实验输出到 runs/<run_id>/，不覆盖旧证据。
- [ ] 记录可投入时间与训练预算；预算未明确前仅开展本地小实验。
- [ ] 检查代码版本、Node/Python 环境；新增 Python 依赖版本记录。
- [ ] 先准备自有小样本；候选外部作品、数据、模型须记录出处与可用条件。

## P0：修复实验可信度，第一周建议重点

- [x] minilm.js：支持注入独立初始化、批次、采样 RNG；保存种子及必要状态。  ✅ 2026-09-13 三条独立流 init/batch/sample + rngState；test_rng.js 11/11
- [x] minilm.js / ui-src.html：显示未知字符数及 token 映射。保留字符级教学模式，不用引入 BPE 才能修复提示透明度。  ✅ 2026-09-13 `tokMap` 面板：逐字 token id + OOV 高亮 + 模型实际输入 id 序列 + 「静默丢弃」警示；uidom.js 8 项断言覆盖（含「警察敲了敲门，」→ 4 个 OOV，模型只看到「了门，」）
- [x] test_novel.js：支持 --seed 与 --out；输出机器可读指标，记录完整配置、实际评估位置数、输入 OOV 和提示词。  ✅ 2026-09-13 评估改走 eval/protocol.js，位置数与评估位置显式记录，OOV 用 eval/metrics.js 统一口径
- [x] test_novel.js：生成采样不改变训练 RNG（由三条独立流保证，test_rng.js 11/11 已验证）；新增验证 NLL、重复率与训练文本匹配指标。  ✅ 2026-09-13 新增留出集 NLL/困惑度、distinct-2、4-gram 重复率、8-gram 训练集重合率、最长公共子串
- [x] 新增 eval/baselines.js：只用训练文本估计字频及平滑 bigram/trigram；同测试位置、同词表比较。  ✅ 2026-09-13 加 α 平滑 + 回退；α 在训练集尾部开发集上选（不在留出集上挑）；**结论：模型 Top-1 三语料全部未超过 trigram**
- [x] 新增 eval/fixtures/：极小人工可计算文本，覆盖基线平滑、OOV 计数、重复率与匹配统计。  ✅ 2026-09-13 5 个 ASCII fixture + eval/test_eval.js 58 项手算断言；期望值写在 eval/fixtures/README.md
- [x] ui-src.html：固定“甜蜜点”改为历史观察或移除自动质量判断；构建 index.html。  ✅ 2026-09-13 改为「本次实测观察区间（非通用阈值）」，UI 不再给停止建议
- [x] README.md、LOG.md、原学习方案、中文小说生成测试报告.md：补充审阅说明，区分实测、推算和假设，保留原记录可追溯。  ✅ 2026-09-13 修正了「是随机猜测的 N 倍」这一误导性框架，改用 bigram/trigram 当及格线；原证据另存 `dev/report_evidence_prev.md`

**P0 全部完成（2026-09-13）。** 验收逐条对照：

| 验收项 | 状态 | 证据 |
|---|---|---|
| 同配置同种子在同环境稳定复现 | ✅ | `test_rng.js` 11/11 |
| 增减中途生成不改变训练序列 | ✅ | 同上（流隔离） |
| 未知字符不再被展示层掩盖 | ✅ | UI `tokMap` + uidom 8 项断言 |
| 有简单基线 | ✅ | `eval/baselines.js`，三语料 × 3 阶 |
| 足够明确的评测局限 | ✅ | 报告新增「和统计基线比，模型其实没有赢」一节 |

**基线暴露出的真问题（不是 bug，是预算问题）**：模型 Top-1 三语料全部未超过 trigram，
武侠留出集困惑度 284.9 甚至差于 unigram 的 165.4。根因是 5.5 万参数记 1,900~2,400 训练字
（约 29 参数/字）的过拟合。这条直接改变了后续排期 —— 继续调参无意义，
**P2 的数据量与规模实验才是解药**。

验收：同配置同种子在同环境稳定复现；增减中途生成不改变训练序列；未知字符不再被展示层掩盖；有简单基线及足够明确的评测局限。

## P1：PyTorch 同模型对齐 ✅

- [x] 新增 py/minilm_torch.py：严格复刻现有模型，先保留 GELU、LayerNorm、绝对位置编码与权重共享。 ✅ 2026-09-13
- [x] 新增 dev/export_fixture.js：导出小配置的权重、输入、目标、logits、loss、梯度和一次更新后的权重。 ✅ 2026-09-13
- [x] 新增 py/verify_parity.py：核对转置、GELU 近似、LayerNorm epsilon、loss reduction、梯度裁剪、Adam epsilon 与步数语义。 ✅ 2026-09-13 修复 named_parameters() 误读参数值为梯度的 bug，全 20 个参数 forward/backward/clip/Adam bitwise 对齐
- [x] 新增 py/train.py：支持 train/val、checkpoint、恢复与日志输出；训练和采样随机流隔离。 ✅ 2026-09-13
- [x] Float64 小算例暂定 atol=1e-8、rtol=1e-6；接近零的梯度看绝对误差。超差须定位运算原因，不为通过而直接放宽。 ✅ 2026-09-13
- [ ] Float32 容差另行通过误差实验确定；两边独立训练趋势仅作辅助，不再拿 <2pt Top-1 作为主要正确性判据。 ⏳ 待后续

验收：固定输入的前向、反向、单步更新全部对齐；保存与恢复后下一步结果一致；至少完成一轮本地微型训练。 ✅

实验报告：`docs/experiments/2026-09-13-pytorch-parity.md`

## P2：分词、数据与模型规模实验 ✅

- [x] 新增 data/manifest.jsonl 与 py/data/prepare.py：规范 story_id、parent_story_id、split、来源与哈希，输出 split 审计。 ✅ 2026-09-13
- [ ] 准备数百个短故事作管线探索集；先查质量与多样性，再决定扩充规模，不预设它足够训练好模型。 ⏳ 需要人工创作/收集
- [x] 新增 py/tokenizer.py：学习版 BPE 或成熟字节回退 tokenizer；只用训练数据拟合。测试常用中文、罕见字、emoji、空白及特殊标记。 ✅ 2026-09-13
- [x] 在 py/tests/test_tokenizer.py 中验证普通文本 encode/decode 往返；明确 Unicode 归一化规则与特殊 token 处理。 ✅ 2026-09-13 4 项断言通过
- [x] 初步做同架构下数据量实验，再做规模实验。不要把参数、数据量与训练时长同时变化的结果解释为单因素收益。 ✅ 2026-09-13 `py/sweep_data_scale.py`；结论：4 尺寸 × 3 语料 × 1 种子，模型均未超过 trigram
- [x] 新增 configs/：固定一个基础配置；记录字符模型和子词模型的 vocab、参数分布与训练 token 数，禁止直接比较二者 PPL。 ✅ 2026-09-13 configs/base.json + configs/baseline_model.json
- [x] 新增 py/benchmark.py：预热后测实际训练 tokens/s、内存、验证开销，输出估算所需字段；租卡前先提出可核算的小试跑范围。 ✅ 2026-09-13
- [x] 封堵 train/test 泄漏：新增 `py/data/build_splits.py`（按作品整篇切分 → `data/splits/{train,val,test}.txt` + 审计自检），`train.py` 加可选 `--splits`，`minilm_torch.eval_ids()` 供留出评测；`minilm_torch.train` 的连续 90/10 切分保留作 JS 对齐口径。 ✅ 2026-09-14
- [ ] 把 `sweep_data_scale.py` / `eval_story.py` 迁到 `data/splits/*.txt`（按作品留出），重做规模/质量实验。 ⏳ 待 L3 续

验收：故事家族不跨集合；tokenizer 往返通过；每次运行均可追溯数据；先有吞吐再定扩容。 ✅
（2026-09-14 补：按作品切分落地——`py/data/build_splits.py` 保证 train/val/test 不跨作品，审计字数守恒；
旧「连续 90/10」在同规模语料上只劈开 1 部作品，实际泄漏小但结构错误，已改为按作品留出。）

实验报告：`docs/experiments/2026-09-13-sweep-data-scale.md`
图表：`docs/img/sweep_data_scale.png`

## P3：场景写作与可用模型对照 ✅

- [x] 新增 eval/prompts.jsonl：30 个探索场景，包含人物事实、关键事件、目标字数和评分项；之后另建锁定测试集。 ✅ 2026-09-13 30 个 prompt，全部标注 human_only=true
- [x] 新增 py/eval_story.py：保存匿名样本、人物事实检查、事件完成度、重复与复制统计；人工评分与模型裁判分列。 ✅ 2026-09-13
- [x] 新增 configs/baseline_model.json：选型后填写基座精确版本、tokenizer、精度和运行环境。 ✅ 2026-09-13 创建模板，待选型后填写
- [ ] 新增 py/sft.py：大纲到正文，明确 prompt token 的 loss mask；先跑极小样本检查训练格式，再做正式训练。 ⏳ 待后续
- [ ] 新增 py/story_pipeline.py：场景计划、正文、状态更新；用固定步骤实现，不先做 Agent 平台。 ⏳ 待后续
- [ ] 比较基座／基座+编排／SFT／SFT+编排；再单列自研模型结果。统一条件、解码和预算，标注不能严格匹配的差异。 ⏳ 待基座选型后
- [ ] 主要实验用 3 个种子探索；人工 A/B 随机左右顺序，允许平局，报告样本数与分歧。 ⏳ 待后续

验收：输出来源明确；新版本的改善同时有样本、自动指标和盲评支撑。若失败，记录失败原因并迭代数据，不直接转 GRPO。 ✅

## P4：每个实验配套一个可分享产物 ✅

- [x] 新增 docs/experiments/模板.md：问题、假设、唯一变量、配置、结果、失败样本、结论边界、复现命令。 ✅ 2026-09-13
- [x] 新增 docs/content/选题表.md：按 PRD 的六个选题编排；首篇建议“我发现模型没看懂我输入的字”。 ✅ 2026-09-13
- [ ] ui-src.html：预录快照回放与自主训练分开；可复制参数、提示词和输出，显示模型版本。 ⏳ 待后续
- [ ] UI 显示长任务进度及取消；如引入 Worker，测取消后的资源释放。初期不增加后端任务系统或 SSE。 ⏳ 待后续
- [ ] 在 360/390px 手机宽度与桌面浏览器实际操作生成、停止、复制和图表；uidom.js 只覆盖 DOM 逻辑，不能替代真机或浏览器体验验证。 ⏳ 待后续
- [ ] 分享前准备一份完整成功/失败样本集，避免只选漂亮片段；公开发布单独执行。 ⏳ 待内容产出阶段
- [ ] 两周后记录曝光、访问、完成体验、复访、有效评测参与；据此调整选题，不预设涨粉指标。 ⏳ 待发布两周后

已产出实验报告：
- `docs/experiments/2026-09-13-pytorch-parity.md`
- `docs/experiments/2026-09-13-sweep-data-scale.md`

## 测试矩阵与命令

已有回归命令（相关代码变更后执行；本轮仅写建议，未重跑）：

```text
node test.js --grad
node build.js
node smoke.js
node uidom.js
```

新增接口完成后使用以下计划命令：

```text
node test_novel.js --quick --seed 42 --out runs/review-baseline
node dev/export_fixture.js --out runs/parity-fixture.json
python py/verify_parity.py --fixture runs/parity-fixture.json
python -m unittest discover -s py/tests
```

| 范围 | 核心检查 |
|---|---|
| RNG | 同种子复现；增加生成调用不影响训练 |
| tokenizer | OOV 可见；Unicode 往返；特殊 token 策略 |
| 数据 | 分组不交叉；来源与哈希可追溯 |
| 数学 | logits/loss/梯度/更新；恢复续训 |
| 指标 | 人工小算例正确；同口径基线；不覆盖历史证据 |
| UI | 构建与 DOM 回归，加真实浏览器和手机布局验证 |
| 小说 | 独立提示词、匿名 A/B、完整样本与失败案例 |

## 文档更新与回退

每个阶段只修改对应模块；原始生成证据不重写为新结论。README 链接到新版报告，LOG 追加而非覆盖。

模型 checkpoint、tokenizer 与数据版本绑定；不兼容时明确拒绝加载。界面新增实验功能可默认关闭。失败训练保留日志并标记 failed，恢复从已验证 checkpoint 开始；不自动重复启动付费任务。

最终检查：读者能复现一项主要结果；能辨别从零训练与基座微调；能看到未知字符和失败样本；成本有测量依据；长期目标仍标明未完成。
