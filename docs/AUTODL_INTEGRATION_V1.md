# AutoDL 远程训练集成方案 V1

日期：2026-09-16。状态：**方案，待用户对 §9 三个决策点拍板后实施**。
配套：[`docs/TODO_NOVEL_MODEL_LEARNING_V3.md`](TODO_NOVEL_MODEL_LEARNING_V3.md)（P1/P2）、
[`docs/EVAL_PROTOCOL_V1.md`](EVAL_PROTOCOL_V1.md)、`py/license_policy.py`、`py/run_identity.py`。

## 1. 定位与五条硬规矩

AutoDL 只是**一台按小时计费的远程 Linux GPU 机**，不是新技术栈：训练代码零改动，
本地与远端跑同一套入口。集成 = 一层薄薄的「打包 → 上传 → 跑 → 回传 → 关机」脚本 +
若干安全阀。

1. **付费任务永不自动启动。** 所有花钱的命令显式执行；跑前打印估算；跑完自动关机
   （V3 既有规则：不自动重复启动付费任务）。
2. **先 doctor → 再 benchmark → 才训练。** 预算表驱动决策，不拍脑袋。
3. **数据最小化上传。** 只传训练所需桶；默认**「权利不清/仅本地学习」数据不出本地**
   （见 §2，待用户确认）。
4. **结果必须带身份回传。** 远端 run 与本地 run 同库：`run_identity` + 数据指纹 +
   doctor 环境快照，缺一不入库。
5. **跨硬件不做 bitwise 宣称。** CPU↔GPU 续训按数值容差验证（V3 P1 原文）。

## 2. 许可红线（2026-09-16 用户已拍板：localugc 上云获批）

`py/license_policy.py` 已把全库分成七档。当前事实（2026-09-16）：

| 档 | 部数 | 处置 |
|---|---|---|
| public_domain / attribution / attribution_sharealike / synthetic | 1,126 部 / 2,793 万字 | 可上云训练 |
| noncommercial（gudian 的 18 部 NiuTrans，CC BY-NC-SA） | 18 部 | 个人学习可上云；产物不可商用，模型卡标注 |
| local_only / unknown（localugc 224,612 部等） | 224,612 部 / 18.9 亿字 | **用户 2026-09-16 批准上云训练** |

**批准的后果（必须如实执行，不可回退删除）**：
- 打包走 `pack_remote.py --allow_non_distributable`，`PACK_INFO.json` 留痕；
- **在该数据上训出的检查点（及蒸馏/合并产物）永久不可对外分发或商用**，模型卡谱系
  必须写明「含权利不清数据训练，仅限个人学习使用」；
- 与净化清单（manifest_v2，无 localugc）上训出的检查点**分开存放、分开命名**，不混用。

因此数据包分两档：`A 包` = 6 证据桶（≈3,073 万字）；`B 包` = 全量含 localugc
（≈24.5 亿字，`--gzip` 后约 2 GB）。谱系跟随包走。

## 3. 实例与镜像（2026-09-16 用户选定：RTX 3080 Ti 12GB）

**结论：该配置满足 30M 候选与全量数据训练的要求**，逐项核对：

| 项 | 配置 | 评估 |
|---|---|---|
| GPU | RTX 3080 Ti 12GB | 30M 模型（Adam fp32 状态 ≈0.5GB + 激活）余量充足；ctx 1024 / batch 32 可跑。速度约为 4090 的 6~7 成，够用 |
| 数据盘 | 50GB SSD | 全量打包（B 包 gzip ≈2GB）+ 解包 7.5GB + checkpoint（30M×fp32 ≈360MB/个）绰绰有余 |
| 内存 | 90GB | 充足；但见 §7 的编码瓶颈——瓶颈在 Python 对象不在物理内存 |
| CPU | 12 核 | 数据预处理/打包足够 |
| 镜像 | PyTorch 2.12.1 / Py3.12 / CUDA 13.0 | 可用；仓库代码不挑 torch 小版本（2.12 vs 本地 2.14），`--resume` 跨版本加载在冒烟阶段（§7-C）验证 |
| 系统盘 | 30GB | 装镜像+环境后偏紧；**代码、数据、checkpoint 全放数据盘**（autodl-tmp），系统盘只留环境 |

价格以控制台实时价为准；3080 Ti 通常低于 4090，对本项目是性价比更优的选择。

## 4. 脚本层（实施时落在 `dev/autodl/`，全部薄封装、无新依赖）

| 脚本 | 跑在哪 | 职责 |
|---|---|---|
| `10_setup_env.sh` | 实例 | 装缺的包（psutil/zstd）；rsync 代码；跑 `doctor --out` |
| `20_pack_data.sh` | 本地 | 按桶打包 `data/processed/<bucket>` + 过滤后的 manifest → `tar`；打印体积 + 许可分级摘要；**未获批准的桶直接拒绝**（复用 license 分类） |
| `30_upload.sh` | 本地 | rsync/scp 代码包 + 数据包到实例数据盘 |
| `40_benchmark.sh` | 实例 | `doctor` + 扩展版 `py/benchmark.py --sweep`（见 §5）→ `budget_table.json` |
| `50_train.sh` | 实例 | 训练入口：checkpoint 每 N 步、日志落盘、结束时 `shutdown`；受 `AUTODL_MAX_HOURS` 硬上限保护 |
| `60_fetch_results.sh` | 本地 | 回传 `runs/`（结果 JSON + checkpoint + 日志）；本地登记 remote 元数据 |

实例连接信息只放**本地** `dev/autodl/instance.local.sh`（gitignore），仓库里只留
`instance.example.sh` 模板——不在仓库落任何主机/端口/密钥。

## 5. `py/benchmark.py` 需要的扩展（P1 既有项，本地也能测）

现有脚本已支持 `--device`、tokens/s、RSS。缺三样，正是预算表需要的：

1. **GPU 显存**：`torch.cuda.reset_peak_memory_stats()` + `max_memory_allocated()`；
2. **(ctx, batch) 组合扫描 + OOM 捕获**：`--sweep` 模式逐组合试跑 200 步，OOM 记为
   `oom: true` 而非崩溃——产出「哪些组合可行」的表；
3. **估算输出**：按实测 tokens/s 推算「30M 候选 × 语料档位 × epochs → 小时数/费用」。

本地 CPU 先把扫描跑通，上云只是换 `--device cuda`。

## 6. 数据打包细则

- 打包**按桶**：`data/processed/<bucket>/*.txt` + 从 `manifest_v2.jsonl` 过滤出这些桶的行
  （保留 `hash`/`split`）+ 与子集清单自洽的 `split_report.json`（指纹按子集重算）+
  `PACK_INFO.json`（来源指纹/许可摘要留痕）。
- 全量含 localugc 的 **B 包用 `--gzip`**（中文文本压缩 ~4×，实测 story 单桶 3.0MB→374KB），
  24.5 亿字全量压后约 2GB，家宽上传可承受。
- 远端跑任何实验前先过 `splits.verify()`——包内清单/报告自洽，同一套指纹校验在两边生效。
- **锁定测试族（`data/locked_test.json`）不上云**：test 族只留在本地做一次性验收，
  远端只允许 val 选型（评测协议 V1）。
- 回传时校验：checkpoint 的 `meta.data_identity.split_fingerprint` 必须等于出发时的指纹，
  不一致就拒收（防止"远端悄悄换了数据"）。

## 7. 分阶段计划（每阶段有独立自检与退出条件）

**新增硬前置（2026-09-16，随"全量含 localugc 训练"的决定而来）**：
`py/novellm/dataset.py` 的**预编码分片**必须先做。原因：当前 `train.py` 全量编码会把
18.8 亿字变成 Python list（指针 14GB + int 对象 ≈49GB），内存直接爆掉；array/numpy 分片
只需 ~7GB 且喂得快。这本来就是 V3 P1 的既定项，现在从"可后置"升级为 **D 阶段（全量训练）
的准入条件**——D0（6 证据桶，3,073 万字）不受影响，现行 `train.py` 可直接跑。

| 阶段 | 内容 | 估算成本* | 自检/退出 |
|---|---|---|---|
| A 准备 | 本地：6 个脚本 + benchmark 扩展 + pack_remote + 本地 CPU 试跑 | ¥0 | ✅ 2026-09-16 完成（62/62 测试；story 包 374KB gzip 实测） |
| B 探测 | 无卡开机 → `10_setup_env` → `doctor`；有卡 → `40_benchmark`（3080 Ti 实测 tokens/s + 显存） | ≈1 GPU 时 | `budget_table.json` 产出；(ctx,batch) 可行组合非空 |
| C 冒烟 | 微型训练端到端 + **跨 torch 版本 resume**（本地 2.14 存 → 实例 2.12 续）+ 数值容差对比 | ≈0.5 GPU 时 | 续训 loss 连续；checkpoint 身份字段齐全 |
| D0 首训 | 30M 候选 × 6 证据桶（3,073 万字）第一段学习曲线（现行 train.py 可跑） | 按预算表 | val 曲线走平或预算触顶；test 只在结束评一次 |
| D1 全量 | **前置：dataset.py 预编码分片**（localugc 18.8 亿字）→ 全量含 localugc 训练 | 按 B 实测 | 分片往返一致；内存峰值实测 <90GB；训练吞吐达标 |
| E 收口 | 回传 + 本地登记 + 关机销毁 | ¥0 | `run_identity` 入库；旧 run 无覆盖 |

**D1 的谱系警示（重申）**：全量含 localugc 训出的检查点**永久不可对外分发或商用**；
若未来要发布模型，须回到 D0 的净化谱系单独训练一个发布版。

\* 4090 时价以控制台为准；D 阶段的费用**由 B 阶段实测决定**，本表不预填。

## 8. 与 V3 路线的对齐

- 本方案只涉及 **P1**（doctor ✅、benchmark 扩展、跨设备冒烟）与 **P2** 的算力侧；
  不触碰 P3/P4。
- 远端训练**不加载任何外部权重**（V3 硬约束）；镜像自带的 torch/库不算权重。
- 模型尺寸推进顺序不变：微型可过拟合 → 30M 候选；100M/325M 只留配置。

## 9. 决策记录（2026-09-16 用户已拍板，原"待拍板"三条关闭）

1. **localugc 上云：✅ 批准**（用户：「我需要把所有数据都进行训练，包括localugc」）。
   按上文 §2 执行：`--allow_non_distributable` 打包留痕，产物不可分发/商用，谱系如实记录。
2. **预算上限：沿用脚本默认**（单次 ≤¥20 / 累计 ≤¥100，`AUTODL_MAX_HOURS=10`），
   环境变量可随时调高。
3. **实例规格：✅ RTX 3080 Ti 12GB**（用户已选定，见 §3 核对表）。

## 10. 明确不做的事

- 不做 AutoDL API 自动开机/选卡自动化（付费动作保持人工触发）；
- 不做多机分布式（30M 用不上）；
- **锁定测试族 / `data/raw` 缓存不上云**（localugc 已获批，但锁定族仍只在本地）；
- 不在仓库落任何实例地址与凭证；
- localugc 训出的检查点**不做对外分发**（许可后果，见 §2）。
