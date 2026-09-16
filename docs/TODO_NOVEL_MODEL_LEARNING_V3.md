# 随机初始化小说模型执行清单 V3

配套：[设计](PRD_NOVEL_MODEL_LEARNING_V3.md)、[人评协议](NOVEL_HUMAN_EVAL_V1.md)。全部为待办。现有实现问题继续以 V2 证据为准；正式训练不采用外部基座。

## P0：可信实验底座（阻断后续大训练）

- [x] 按 V2 P0 修复 `py/train.py` seed/resume、`py/eval_story.py` prompt/continuation、`eval/prompts.jsonl` 两条非法 JSON；添加反例测试。 ✅ 2026-09-15
  - 证据：`prompts.jsonl` 17/29 两条非法 JSON 已修，现 30/30 可 `json.loads`；
    `train.py` seed 与 resume 已修并有反例证据（`py/verify_seed.py`：同 seed Δ=0、旧顺序不可复现；
    `py/verify_resume.py`：完整状态续训与直训逐位一致 Δ=0，丢 RNG/丢优化器/旧格式分别 Δ=3.1e-2/2.5e-2/3.7e-2）；
    `py/eval_story.py` 已重写为「只用 continuation 评分 + 并排展示旧口径送分幅度」，
    并有 `py/verify_story_eval.py` T1/T2/T3 三项实证（出题来自留出作品、生成可复现、旧口径送分幅度）。
- [x] `py/checkpoint.py` 与 `py/run_identity.py` 保存恢复完整状态；验证连续训练与中断恢复下一批数据、学习率和最终权重等价。 ✅ 2026-09-16
  - `py/run_identity.py` 已建（数据身份 + 配置指纹，见下条）；`checkpoint.py` **未单独抽模块**——
    完整状态保存内联在 `py/train.py`（checkpoint v2：model/opt/itos/meta/step/rng）。
    等价性已验证：`py/verify_resume.py` 完整状态续训 vs 直训**逐位一致（Δ=0）**，
    丢 RNG/丢优化器/旧格式三种失败样本分别偏 3.1e-2/2.5e-2/3.7e-2；
    `dev/review_current.py` 探针亦复证同 seed 两次训练 Δ=0、resume step 1→2 正常。
  - ⏳ 形式项：把 checkpoint 逻辑从 train.py 抽成 `py/checkpoint.py`（纯重构，无行为变化，待办）。
- [ ] 数据准备及 sweep 入口强制验证家族去重、split 版本、用途 allowlist、配置指纹；不复用旧污染缓存。
  - ✅ 2026-09-15 已做：**家族/内容去重**与 **split 版本指纹**由 `py/splits.py:verify()` 强制执行
    （`bucket_texts()` 每次加载先校验，不成立抛 `SplitIdentityError` → 实测非零退出码 1）；
    指纹算法只在 `py/splits.py` 定义一份，`py/data/build_splits.py` 反向 import，避免「自己校验自己永远通过」。
    去重动作在 `py/data/prepare.py:dedupe_by_content()`（分配 split **之前**）。契约测试 `py/tests/test_splits.py` 11 项。
  - ✅ 2026-09-15（续二）补齐：**配置指纹** = `py/run_identity.py`（数据身份：清单名 / split 指纹 /
    `near_dup_merged` / 许可分级，+ cfg/seed/steps/batch/lr → 16 位 `run_identity`），由 `train.py`
    写进 `meta.json`，算不出身份就不存盘；**用途 allowlist** = `py/license_policy.py`（fail-closed
    七档分级 + `assert_distributable()` 闸门；实测全库仅 **1,126 部 / 0.58% 部数（11.5% 字数）**许可
    明确可分发）。清单择优 `py/splits.py:resolve_current()`：去重候选必须**经新鲜度校验**
    （内容哈希集合与默认清单一致）才可作默认——v2 是 32,036 部旧快照、当前 v1 已 192,823 部，
    无条件优先会静默丢 16 万部。测试 `py/tests/test_identity_policy.py` 11 项。
  - ⏳ 仍缺：`sweep_*` 侧独立入口校验（目前靠 `py/splits.py` 统一拦截）；
    `assert_distributable()` 的调用方（`export.py`，属 P4）；
    ~~默认清单的近重复/同源泄漏未清零~~ → ✅ 2026-09-15（续五）：`dedup_works` 对 6 个证据桶
    重生成净化清单（移动 10 部），`audit_splits` 文本级 6 桶全部 0/200、门禁**通过**，
    默认清单已自动切到 `manifest_v2.jsonl`（`near_dup_merged=True`）。
    剩 `localugc`（224,612 部）未审计——不进 splits/探针，其上数字单独标注。
  - ✅ 2026-09-15（续六）：`py/sweep_worksplit.py` 已 val 化（曲线=val、test 一次性验收，
    `protocol='val_select_test_once_v1'` 标记，旧文件复用时提示 exploratory）；
    附带发现 **scifi 桶没有 val 作品**（5 部：4/1/0），合规 sweep 在该桶被拒绝——需补数据或定豁免。
- [ ] 所有调参改用 val；历史 test 标 exploratory，新锁定测试族。
  - ✅ 2026-09-15 已立：协议见 [`docs/EVAL_PROTOCOL_V1.md`](EVAL_PROTOCOL_V1.md)。
    `py/train.py` 本就只看 val（test 最后评一次）；锁定测试族由 `py/locked_test.py` 冻结
    （`data/locked_test.json`：58 部，`lock_fingerprint=6f60d9363678bf5b`，按**内容哈希**取身份，
    story_id 会随去重重新编号不可靠）；`--verify` 对「作品消失 / split 归属变化 / 文件被手改」
    非零退出，清单换代只注记。契约测试 `py/tests/test_locked_test.py` 8 项。
  - ⏳ 仍缺：历史数字的 exploratory 标注只落在协议文档里，`docs/experiments/` 各报告与
    `LOG.md` 既有条目尚未逐篇补标；`py/sweep_worksplit.py` 每个 eval 步号读 test 的用法未改
    （属并发工作线在途文件）；实践篇 24/31 的修订（下一条）也依赖同一批勘误。
- [ ] 修订实践篇 24/31 的泄漏归因、预算推断与错误命令；保留原始数字及勘误原因。

出口：空生成不获故事分；同 seed 可重复；真正 resume；不合规数据和不匹配缓存非零退出。

**✅ 2026-09-16 P0 出口条件全部达成**（实证 `python -X utf8 dev/review_current.py`，落盘
`runs/review_20260914/probes.json`）：

| 出口条件 | 实证 |
|---|---|
| 空生成不获故事分 | 30 题空生成事实/事件命中 **0/153**（旧协议把提示词拼进去可白拿 **67/153**——送分实证）；`zero_generation` 返回空串 |
| 同 seed 可重复 | 两次独立训练全参数**逐位相等（Δ=0.0）** |
| 真正 resume | `--resume` 退出码 0，step 1→2，checkpoint v2 七字段齐全；另有 verify_resume Δ=0 |
| 不合规数据/不匹配缓存非零退出 | `py/splits.py:verify()`（实测退出码 1）+ `py/audit_splits.py` 门禁（当前 6 桶 0/200 **通过**）；数据审计 0 缺文件 / 0 哈希不符 / 0 跨 split 哈希组 |

P0 余留（不阻断 P1）：实践篇 24/31 的泄漏归因修订（下条）；`docs/experiments/` 逐篇 exploratory 补标；
`localugc` 未做同源审计；checkpoint 抽模块（纯形式）。

## P1：本地训练资源与随机初始化证明

- [x] 新增 `py/novellm/doctor.py`：CUDA/MPS/CPU、设备实际名称/总量/可用内存、torch版本、精度支持；不读取或输出凭证。 ✅ 2026-09-16
  - `python -X utf8 -m py.novellm.doctor`（可 `--out` 存 JSON）。fail-closed：没有的设备如实
    `available: false`（不缺字段、不猜数字）；psutil 缺席时内存留空并注明，绝不估。
    本机（N100 级 CPU 盒）实测：torch 2.14.0+cpu、无 CUDA/MPS、4 逻辑核、CPU bf16/fp16 可用
    （`dev/verify/doctor.json`）。**⏳ 待在 4090/8GB 目标机上跑一次**，才能写真正的预算表。
    契约测试 `py/tests/test_doctor.py` 4 项。
- [ ] AutoDL 租卡集成（方案：[`docs/AUTODL_INTEGRATION_V1.md`](AUTODL_INTEGRATION_V1.md)，2026-09-16 起草，**待 §9 三个决策点拍板后实施**）。
  - 要点：AutoDL = 可插拔远程 Linux GPU 运行时，训练代码零改动；`dev/autodl/` 六个薄脚本
    （setup/pack/upload/benchmark/train/fetch）；付费任务永不自动启动、跑完自动关机、
    `AUTODL_MAX_HOURS` 硬上限；数据最小化上传——默认只上 6 个证据桶（≈3,073 万字，公版/开放许可），
    **localugc 默认不出本地**（待用户拍板）；锁定测试族不上云；回传校验 split 指纹防"远端悄悄换数据"。
  - ⏳ 实施 = ①扩展 `py/benchmark.py`（GPU 峰值显存 / (ctx,batch) 组合扫描 + OOM 捕获 / 预算估算，
    本地 CPU 先跑通）②写 6 个脚本 ③阶段 B 探测（无卡开机 → doctor → 有卡 benchmark）。
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
