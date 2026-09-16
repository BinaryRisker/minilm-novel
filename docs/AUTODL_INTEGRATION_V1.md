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

## 2. 许可红线（默认值，待 §9 拍板）

`py/license_policy.py` 已把全库分成七档。当前事实（2026-09-16）：

| 档 | 部数 | 处置 |
|---|---|---|
| public_domain / attribution / attribution_sharealike / synthetic | 1,126 部 / 2,793 万字 | **可上云训练**（gudian/wuxia/urban/xiandai/scifi/story 六个证据桶） |
| noncommercial（gudian 的 18 部 NiuTrans，CC BY-NC-SA） | 18 部 | 个人学习可上云；**产物不可商用**，模型卡标注 |
| local_only / unknown（localugc 224,612 部等） | 224,612 部 / 18.9 亿字 | **默认不出本地**：上云=数据落第三方磁盘，与「仅本地学习」的许可判断冲突 |

因此默认上云语料 = **6 个证据桶 ≈ 3,073 万字**（打包后约 15–20 MB，上传几分钟）。
localugc 若要上云需用户显式批准并记录（§9-①）。这条红线同时决定了**模型卡的谱系**：
在净化清单上训出的检查点才谈得上对外可分发。

## 3. 实例与镜像（建议起步配置，价格以控制台实时价为准）

- **规格**：RTX 4090（24 GB）按量计费起步。V3 按 8 GB 可用显存设计是**下限约束**，
  24 GB 只是把安全余量变大，不改变「微型 → 30M 候选」的推进顺序；100M/325M 仍只作候选。
- **镜像**：官方 PyTorch 2.x + CUDA 12.x 镜像（免自己装驱动）；`pip install psutil`（doctor 的
  内存项）+ `zstd`（数据包）。
- **数据准备/调试用「无卡模式」**（约 ¥0.1/小时量级）：解包数据、跑 `doctor --device cpu`、
  改代码都不该烧 GPU 钱。
- **学术加速**：仅当需要从 HF 拉东西时开；本项目默认不需要外部下载。
- 计费按小时；**跑完即关机**是成本第一杠杆。

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
  （保留 `hash`/`split`，远端 `splits.verify()` 才能继续当门禁用）+ `split_report_v2.json`。
- 远端跑任何实验前先过 `splits.verify()`——同一套指纹校验在两边生效，杜绝"远端用了
  对不上的数据"。
- **锁定测试族（`data/locked_test.json`）不上云**：test 族只留在本地做一次性验收，
  远端只允许 val 选型（评测协议 V1）。
- 回传时校验：checkpoint 的 `meta.data_identity.split_fingerprint` 必须等于出发时的指纹，
  不一致就拒收（防止"远端悄悄换了数据"）。

## 7. 分阶段计划（每阶段有独立自检与退出条件）

| 阶段 | 内容 | 估算成本* | 自检/退出 |
|---|---|---|---|
| A 准备 | 本地：写 6 个脚本 + benchmark 扩展 + 本地 CPU 试跑 | ¥0 | 本地全量回归绿；打包体积与许可摘要打印 |
| B 探测 | 无卡开机 → `10_setup_env` → `doctor`（CPU 项）；有卡 10 分钟 → `40_benchmark` | ≈1 GPU 时 | `budget_table.json` 产出；(ctx,batch) 可行组合非空 |
| C 冒烟 | 微型训练端到端：seed 固定训 200 步 → 存 checkpoint → 关机重开续训 1 步 → 数值容差对比 | ≈0.5 GPU 时 | 续训前后 loss 曲线连续；checkpoint 身份字段齐全 |
| D 首训 | 30M 候选 × 6 证据桶（3,073 万字）第一段学习曲线 | 按预算表 | val 曲线走平或预算触顶；test 只在结束评一次 |
| E 收口 | 回传 + 本地登记 + 关机销毁 | ¥0 | `run_identity` 入库；旧 run 无覆盖 |

\* 4090 时价以控制台为准；D 阶段的费用**由 B 阶段实测决定**，本表不预填。

## 8. 与 V3 路线的对齐

- 本方案只涉及 **P1**（doctor ✅、benchmark 扩展、跨设备冒烟）与 **P2** 的算力侧；
  不触碰 P3/P4。
- 远端训练**不加载任何外部权重**（V3 硬约束）；镜像自带的 torch/库不算权重。
- 模型尺寸推进顺序不变：微型可过拟合 → 30M 候选；100M/325M 只留配置。

## 9. 待用户拍板（实施前需要明确的三件事）

1. **localugc 是否允许上云？** 默认否（许可"仅本地学习"，上云=数据落第三方磁盘）。
   若批准，需写明范围与用途并记录在模型卡谱系里。
2. **单次/累计预算上限**（例如单次 ≤¥20、累计 ≤¥100，超限脚本拒绝启动）。
   `AUTODL_MAX_HOURS` 的默认值据此定。
3. **实例规格**：4090-24G 起步是否可以？（预算按 8GB 可用显存设计不受影响，
   只是余量更大；更便宜的卡如 3090 同样能跑 30M 候选。）

## 10. 明确不做的事

- 不做 AutoDL API 自动开机/选卡自动化（付费动作保持人工触发）；
- 不做多机分布式（30M 用不上）；
- 不上传 `localugc` / 锁定测试族 / `data/raw` 缓存；
- 不在仓库落任何实例地址与凭证。
