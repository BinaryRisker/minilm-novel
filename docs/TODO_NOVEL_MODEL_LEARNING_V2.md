# 小说模型下一阶段执行清单 V2

> **2026-09-17 阅读说明**：需求/规划与当时验收记录，勾选状态不自动等于当前实现。现行代码边界和未实现模块见[状态说明](content/当前实现与历史结论.md)，新手不要从本文件复制设想命令。

日期：2026-09-14。配套 [审计与设计](PRD_NOVEL_MODEL_LEARNING_V2.md)。本文件全部为待办；本轮新增审阅文档与复核探针，不代表下面功能已经实现。

## 前置条件

- [ ] 保留当前未提交修改与历史结果，冻结当前代码/数据身份；为新实验另建 runs 目录。
- [ ] 优先定义 M1 为 300–800 字完整场景，自研模型与外部基座结果分别验收。
- [ ] 先用 CPU 微型配置验收工程；候选规模的吞吐/内存测完后再决定付费算力。
- [ ] 保留现有 parity 路径；新增生产训练路径不得静默改变历史 fixture 语义。

## P0：修复阻断问题，建立可信最小闭环

- [ ] `eval/prompts.jsonl`：修复17/29行，校验30条记录schema、唯一ID、字段类型；新增 `py/tests/test_story_eval.py`。
- [ ] `py/eval_story.py`：返回 prompt/effective_prompt/continuation；只对正文计长度、重复与事件；支持 seed、停止原因、OOV与截断信息。
- [ ] 将 fact_check 重命名或替换；名字出现与事实一致性分开，未提及不当作正确；“红衣通过蓝衣”反例必须失败。
- [ ] 独立 UNK/BOS/EOS/PAD；非法温度、空 prompt、过长 prompt、短输出有明确行为，max_new=0正文为空。
- [ ] `py/train.py`：构造模型前设seed，批次/生成用独立流；训练初始化和恢复走统一配置解析。
- [ ] 新增 `py/checkpoint.py`：保存模型、optimizer、调度、RNG、数据/tokenizer/config身份、全局step/token数；原子保存latest/best。
- [ ] 实现真正resume，固定总调度长度；恢复时校验数据身份、配置、tokenizer；旧权重文件支持显式 warm_start，不能伪装 exact resume。
- [ ] 新增 `py/tests/test_training_state.py`：连续N步与第K步恢复至N步，比较下一批数据、学习率、optimizer与最终权重；同seed两次CLI结果一致。
- [ ] `py/data/prepare.py`、`build_splits.py`：重复分组、稳定ID、固定split版本；审计失败则非零退出，训练入口拒绝使用。
- [ ] 建立用途allowlist，release/evaluation不默认包含项目约定隔离的localugc；learning允许显式启用。
- [ ] 新增 `py/run_identity.py`：完整数据/配置/协议/代码指纹，扫描缓存严格校验；report_only拒绝混合不兼容身份。
- [ ] 新增 `py/tests/test_data_contract.py`：重复内容、同父故事、ID稳定、增量数据不改变既有split、用途隔离、缓存失效均有反例测试。

验收：一份合法数据+一个配置可以训练、保存、断点续训、生成、评测，产物身份一致；默认30场景可解析；空输出不能因prompt获人物/事件分。P0未通过不开始大规模新扫描。

## P1：重建评测协议并修订已有内容

- [ ] `train.py` / `sweep_worksplit.py` 接val；早停、alpha、超参只用val；现有test标记exploratory，另锁定未用过的测试作品家族。
- [ ] 新建固定目标位置协议，跨ctx对比使用同一批目标，显示有效历史长度；按文档边界评测，避免拼接制造跨作品目标。
- [ ] `--leak_probe` 改成训练内/同作品未训练/异作品三臂，断言后两臂与训练的目标区间不重叠。
- [ ] 训练采样记录作品曝光，采用固定混合比例的嵌套子集；新增数据量×预算小网格，避免一次跑满所有桶。
- [ ] 模型与n-gram输出相同目标集合上的NLL/Top-1/Top-5、OOV与已知字子集指标；按作品报告均值/离散度，避免token独立性假设。
- [ ] 修订实践篇24/31、对应实验报告、README和发布说明；旧数字保留版本与更正原因，不覆盖原始JSON。
- [ ] 修正V1 TODO总状态与当前事实不符处，或将其明确标记历史记录并链接V2。

验收：读者能区分训练内评估、开发集探索、最终盲测；文章每个主要数字有run_id和数据版本；复现命令能解析并指向正确配置。

## P2：从“下一字实验”进入完整短故事训练

- [ ] 新增 `py/data/prepare_stories.py`：完整故事、场景、人物/事件标签、parent/group_id、来源、数据用途；先审阅小样质量再扩量。
- [ ] 新增 `py/tokenizer_adapter.py`：统一fit/encode/decode/save/load，字符baseline与成熟tokenizer分别版本化；记录无损性/OOV/压缩率。
- [ ] 新增 `py/dataset.py`：预编码分片、文档边界、EOS、训练packing与跨文档策略；评测不跨文档。
- [ ] 重构train.py接配置文件；训练器支持梯度累计、设备精度、验证周期、早停、取消、错误快照与JSONL日志。
- [ ] 修复benchmark.py：字符吞吐、RSS/CUDA峰值、计时边界、预热权重语义；记录软件、CPU/GPU与线程配置。
- [ ] 1–3M候选配置先跑工程和语料小试；再依据吞吐、val、完整生成质量决定10–30M及更长ctx。保存参数分账和预算，不承诺必然成功。
- [ ] 新增 `py/eval_language.py` 与场景报告：NLL、重复、长匹配片段、EOS结束率、完整样本；禁止只展示挑选出的好句。

验收：达到PRD M1或明确记录未达标原因；随机初始化模型输出与外部生成数据/外部基座输出身份分开。未达标先修改数据/语言训练，不跳到长篇。

## P3：大纲条件写作与单场景验收

- [ ] 新增 `py/data/prepare_sft.py`：设定/人物/状态/大纲/正文配对；父故事及改写保持同split；评测场景不使用从答案反抽的大纲。
- [ ] 新增 `py/sft.py`：prompt loss mask、next-token shift、EOS、padding位置处理；训练前打印一条可读mask样例。
- [ ] 新增 `py/model_adapter.py`；填实 configs/baseline_model.json 的精确revision、模型卡、tokenizer、运行环境。
- [ ] 比较自研预训练、自研SFT，以及外部基座各组；采样预算和信息量尽量一致，不可一致处明确披露。
- [ ] 匿名A/B随机左右顺序，至少两名读者，允许平局；记录完整性、因果、人物动机、事件执行、重大矛盾，不把LLM裁判替代人评。
- [ ] 在锁定集报告样本数、失败数、区间与完整样本；达到M2再继续扩场景。

## P4：多场景到完整小说

- [ ] 新增 `py/story_pipeline.py`：大纲→场景→正文→审核→状态更新→全文，模型接口可替换。
- [ ] 新增 `py/story_state.py`：人物属性、目标、位置、时间线、伏笔、事件状态，每条带正文证据位置与版本。
- [ ] 运行保存project_id/scene_id/status/attempt/input_hash；重试幂等、限制修订次数、取消和失败可恢复；已accepted场景不因重试重复追加。
- [ ] 状态更新只接受已发生且有证据的事实；生成自相矛盾时保留原稿与修订，不抹除失败。
- [ ] 新增 `py/tests/test_story_pipeline.py`：假模型验证状态/幂等/恢复/取消；另用真实模型样本验证质量，两种验收分开。
- [ ] 先完成3–5场景短篇，审查冲突收束和跨场景事实；通过M3后才做章节级与数万字实验。
- [ ] 导出正文、人物表、大纲、状态轨迹与运行成本，便于读者核验。

## P5：将实验产物转为分享文章

- [ ] 新增 `py/export_experiment.py`：由run目录生成证据卡、表格、成功/失败样本索引、结论边界占位。
- [ ] 新增 `docs/content/catalog.json`：文章状态、claim_id、run_id、code/data版本、作者审校、发布链接。
- [ ] `dev/lint_content.py` 保留结构检查，增加命令参数/本地链接/数字来源/元数据检查；它不能自动批准科学结论。
- [ ] 下一批优先：评测空输出反例、真正续训、独立故事数据、写完一场戏、听懂大纲、跨场景记忆。
- [ ] 每篇人工检查叙事经历真实性、因果归因、完整失败样本与图片；先打磨4–6篇，避免重复原理文抢占实验时间。
- [ ] `ui-src.html` 增成果回放/完整样本/证据入口，显示预录和实时来源；相关代码变更后重建index.html。
- [ ] 真浏览器验收桌面与360/390px：生成、停止、复制、全文阅读、图表滚动、空状态；长任务可取消，不锁住主界面。
- [ ] 文章人工审阅完成后再执行实际发布；已发布稿保留更正入口，记录有效问题和评测参与反馈。

## 测试矩阵与现有可运行命令

在仓库根目录运行；Windows使用UTF-8输出。

```powershell
python -X utf8 -m unittest discover -s py/tests -t . -v
python -X utf8 py/verify_parity.py
node smoke.js
node uidom.js
node test_rng.js
node eval/test_eval.js
python -X utf8 dev/lint_content.py
python -X utf8 dev/review_current.py
```

review_current.py是本轮诊断快照工具，输出runs/review_20260914/probes.json；修复后探针应表现为同种子equal=true、resume步数前进、空生成为空、JSONL无错误。它不代替新增的断点等价/数据/场景测试。

| 范围 | 需要新增的关键验证 |
|---|---|
| 训练 | 同seed、连续/恢复等价、生成不扰乱batch、数据不符拒绝恢复 |
| 数据 | 完全/家族重复、用途隔离、固定split、缓存指纹变更 |
| 评测 | JSONL schema、零生成、事实矛盾、EOS、上下文截断、目标区间不重叠 |
| SFT | mask/shift/padding、训练和测试家族隔离、实际场景A/B |
| 管线 | 场景幂等、取消/恢复、事实证据、有限重试、完整故事人工验收 |
| 内容 | 数字来源、配置/数据身份、命令、结论边界与作者审校 |
| UI | 原有DOM回归和真实桌面/手机布局分别通过 |

## 回退与完成标准

新训练器/tokenizer/protocol使用显式版本，不改写旧checkpoint和run。无法精确恢复的旧文件仅允许warm_start；新数据去重后创建新manifest版本。模型、数据、文章证据不兼容时阻止混用，失败实验保留并标failed。

本阶段不建设通用Agent平台、复杂数据库/任务中心、不先做DPO/GRPO、不以浏览器大训练为目标。最终交付至少是一条可靠的“训练→场景→完整短篇→质量报告→文章证据”闭环；若模型质量未达门槛，应如实停在相应阶段，而非勾选整个路线完成。
