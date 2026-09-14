# 随机初始化小说模型执行清单 V3

配套：[设计](PRD_NOVEL_MODEL_LEARNING_V3.md)、[人评协议](NOVEL_HUMAN_EVAL_V1.md)。全部为待办。现有实现问题继续以 V2 证据为准；正式训练不采用外部基座。

## P0：可信实验底座（阻断后续大训练）

- [ ] 按 V2 P0 修复 `py/train.py` seed/resume、`py/eval_story.py` prompt/continuation、`eval/prompts.jsonl` 两条非法 JSON；添加反例测试。
  - 进度（2026-09-15 定时自检核对）：`prompts.jsonl` 17/29 两条非法 JSON **已修**，现 30/30 可 `json.loads`；
    `train.py` seed 与 resume **已修并有反例证据**（`py/verify_seed.py`：同 seed Δ=0、旧顺序不可复现；
    `py/verify_resume.py`：完整状态续训与直训逐位一致 Δ=0，丢 RNG/丢优化器/旧格式分别 Δ=3.1e-2/2.5e-2/3.7e-2）。
    ⏳ `py/eval_story.py` 的 prompt/continuation 重复计算本轮未核，故本条不勾选。
- [ ] `py/checkpoint.py` 与 `py/run_identity.py` 保存恢复完整状态；验证连续训练与中断恢复下一批数据、学习率和最终权重等价。
- [ ] 数据准备及 sweep 入口强制验证家族去重、split 版本、用途 allowlist、配置指纹；不复用旧污染缓存。
  - ✅ 2026-09-15 已做：**家族/内容去重**与 **split 版本指纹**两项由 `py/splits.py:verify()` 强制执行
    （`bucket_texts()` 每次加载先校验，不成立抛 `SplitIdentityError` → 实测非零退出码 1）；
    指纹算法只在 `py/splits.py` 定义一份，`py/data/build_splits.py` 反向 import，避免「自己校验自己永远通过」。
    去重动作本身在 `py/data/prepare.py:dedupe_by_content()`（分配 split **之前**）。契约测试 `py/tests/test_splits.py` 11 项。
  - ⏳ 仍缺：**用途 allowlist**（license/用途未进校验）、**配置指纹**（run 配置→指纹未落盘），
    以及 `sweep_*` 侧独立入口校验（目前靠 `py/splits.py` 统一拦截）。
- [ ] 所有调参改用 val；历史 test 标 exploratory，新锁定测试族。
- [ ] 修订实践篇 24/31 的泄漏归因、预算推断与错误命令；保留原始数字及勘误原因。

出口：空生成不获故事分；同 seed 可重复；真正 resume；不合规数据和不匹配缓存非零退出。

## P1：本地训练资源与随机初始化证明

- [ ] 新增 `py/novellm/doctor.py`：CUDA/MPS/CPU、设备实际名称/总量/可用内存、torch版本、精度支持；不读取或输出凭证。
- [ ] 新增 `model.py`、`tokenizer.py` 和配置 schema：先实现微型与 30M 配置，词表由训练集拟合；断言参数量、特殊 token 与根初始化来源。
- [ ] 新增 `dataset.py/trainer.py/checkpoint.py`：预编码分片、文档边界、累计梯度、验证、取消、安全存盘；复用已验证状态契约。
- [ ] 新增 `benchmark.py`：预热后 200 步有效 token/s、显存/RSS、OOM 和恢复；对 1024/2048 ctx 的可行组合输出估算，不自动租卡。
- [ ] 新增 `tests/test_scratch_lineage.py`、`test_model.py`、`test_resume.py`：根权重可由 seed 重建，禁止外部权重入口混入；因果遮罩、padding/loss mask、KV cache 与非 cache logits 对齐。
- [ ] CUDA/MPS 各执行微型前向、反向、保存加载、生成；跨设备比较设数值容差，不宣称跨平台 bitwise 等价。

出口：CPU 微型端到端通过；GPU 实测后写预算表。100M/325M 只保留候选配置，不默认训练。

## P2：中文完整故事数据与第一条学习曲线

- [ ] 新增 `py/data/prepare_stories.py`：来源、许可、家族 ID、用途、正文、难度标签、证据位置；小批审阅不合格则停止扩量。
- [ ] 新增 tokenizer 统计：字符/token比、OOV、往返还原、唯一 token 与曝光量，tokenizer 不读取 val/test。
- [ ] 建立词汇/人物/冲突/结局组合的开发题与独立保留组合，近重复检查覆盖所有派生数据。
- [ ] 先微型可过拟合检查，再执行 30M 候选的阶段训练；每次预算扩展依据 val 和完整输出，不只看 train loss。
- [ ] `evaluate.py` 输出 30 个预定题全部正文、EOS/长度/重复/抄录长片段及真实失败，完成人工 G1 检查。

出口：达到 G1 才进入自主规划；否则写失败报告并调整数据或任务难度。

## P3：自模型规划与多场景

- [ ] `py/data/prepare_sft.py` 构建计划、正文、状态、修订多任务记录；任务级 loss mask 和父家族同 split。
- [ ] `py/novellm/planner.py/state.py/pipeline.py` 接自己的检查点；世界事实与人物认知分开，计划与已发生事件分开。
- [ ] 候选状态、正文证据、场景原子提交、取消恢复、最多 2 次修订；保留每次候选和费用/时间。
- [ ] 新增契约测试：未来事实泄漏、无依据状态写入、误解误判矛盾、重复提交、超预算、中断恢复。
- [ ] 固定开发集分别比较直接写作、加计划、加状态、加修订；报告总生成 token、失败率及人工质量，不能只增加计算后宣称架构收益。

出口：达到 G2；推理时没有外部模型完成写作、规划或润色。

## P4：离线交付与真人验收

- [ ] `export.py/generate.py` 导出权重、tokenizer、配置、身份和最小锁定依赖；新进程离线加载，不依赖作者机器绝对路径。
- [ ] 新增 `dev/smoke_novel_scratch.py`：临时目录随机初始化→短训→中断恢复→导出→离线加载→完整产物，不把微型垃圾文本标文学通过。
- [ ] 新增 `py/novellm/human_eval.py`：匿名包、映射分离、评分导入校验、题目聚类区间、失败列表；不生成或补齐真人评分。
- [ ] 按人工协议先校准，再冻结 20 题执行；招募未完成时明确标待人评，不能勾选 G3。
- [ ] 功能可靠性 30 次任务单独报告；导出模型单次运行和多场景流程分别验收。
- [ ] 模型卡记录初始化谱系、数据用途/许可、目标范围、硬件、训练曝光 token、全部人评统计及局限。

## 测试矩阵与交付命令

现有基线可执行：`python -X utf8 -m unittest discover -s py/tests`；审计探针：`python -X utf8 dev/review_current.py`。后者创建微型训练及审计输出，不替代小说质量验收。

以下是实现后才成立的接口约定，当前不能当作已可执行功能：

| 入口 | 验证目标 |
|---|---|
| `python -X utf8 -m py.novellm.doctor` | 明确真实设备及后端 |
| `python -X utf8 dev/smoke_novel_scratch.py` | 新环境训练/恢复/导出/推理 |
| `python -X utf8 -m unittest discover -s py/novellm/tests` | 数据、谱系、数值、状态与评分导入契约 |

实现时在 `py/README.md` 写入实际训练、恢复、生成、评审导入命令并逐一试跑，不仅检查 --help。每条命令提供微型配置和输出位置；付费训练命令必须另列实测估算。

## 文章、界面与回滚

- [ ] 新增实验 manifest→文章引用检查，所有数字可追到配置和原始结果；失败案例同行展示。
- [ ] 每阶段一篇文章，区分“方法已实现”“工程已验证”“质量经人评”；V1/V2 加索引指向 V3，保留审计记录。
- [ ] 本阶段无新增网页流程，保持原页面回归；未来 UI 接入时再验 375px/桌面长正文、取消、恢复、下载及错误提示，不能提前勾通过。
- [ ] 新训练路径显式选择；回滚到上一个已验证 checkpoint/config，禁止覆盖旧模型和已发布实验；发现数据污染则作废对应评测版本。

执行顺序：P0 → P1 → P2 → P3 → P4。P0 和设备检测是下一次实施的首批工作；本次文档完成不等于其中任务完成。
