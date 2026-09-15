# 近重复检测推到全 7 桶（含 localugc）· 内部索引

> 对应内容稿：`docs/content/实践篇36_数据涨了六倍我的去重脚本就再也跑不完了.md`
> 脚本：`py/audit_neardup.py`（新，可扩展近重复检测）、`py/dedup_works.py`（抽出 `resplit()` 供复用）

## 1 为什么换算法

`py/dedup_works.py` 走全量两两比对 C(n,2)：

| 桶 | 作品数 | C(n,2) 对 | 可行性 |
|---|---|---|---|
| gudian（第 34 篇时的规模） | 55 | 1,485 | 22 秒 |
| **localugc（本轮）** | **191,679** | **≈1.8×10¹⁰** | 按 1.25 s/对 ≈ 700 年 |

换 **行级倒排索引**：论坛帖的天然单位是「行」。
localugc 实测 **3,382,869** 个去重行指纹 / 3,477,547 总行 → 单位数少 **5,400 倍**，**156 秒**跑完。

## 2 分层：小桶锚点法 / 大桶行级法

行级法要求整行逐字相同，**漏「同书异本」**（实测漏掉《三国演义》↔《三国志演义》，锚点 0.5516 / 精确 0.3693）。
故 `--algo auto`：作品数 ≤ `--anchor-max`(3000) 走锚点法（`analyze_anchors`，import `dedup_works` 的 winnowing），否则走行级倒排（`analyze`）。

| 桶 | 作品数 | 算法 | 跨 split 对 | 跨 split 簇 |
|---|---|---|---|---|
| gudian | 55 | anchors | 3 | 3 |
| wuxia | 23 | anchors | 0 | 0 |
| urban | 26 | anchors | 0 | 0 |
| xiandai | 35 | anchors | 0 | 0 |
| scifi | 5 | anchors | 0 | 0 |
| story | 1,000 | anchors | 0 | 0 |
| localugc | 191,679 | lines | 1,309 | 295 |
| **合计** | **192,823** | | **1,312** | **298** |

## 3 五道守卫（每一道都由一次实测假阳性逼出来）

| # | 守卫 | 参数 | 被它挡掉的假阳性（实测） |
|---|---|---|---|
| 1a | 行 df 绝对上限 | `--max-df-abs 200` | xiandai 的 wikisource 版权声明模板（df≈20 在 35 部队桶里过不了 500 的旧门槛） |
| 1b | 行 df 占比上限 | `--max-df-frac 0.20` | 小桶上绝对阈值失效（xiandai frac=7 才有效） |
| 2 | 最少共享行数 | `--min-shared 5` | xiandai 首次运行报 63 对跨 split，取证为三行版权模板 |
| 3 | 最少共享字数 + 按字数算 containment | `--min-shared-chars 200` | `localugc` 巨型长帖仅共享 `-----` 分隔线，行数比例被分母放大 |
| 4 | 锚点 df 过滤 | 同 1a/1b | xiandai 11 部「锚点 0.125 / 精确 0.143」的模板簇（剔 388 个锚点） |
| 5 | 锚点精确复核门槛 | `--min-exact` **0.02 → 0.005**（下一轮扫描后改口） | story 桶 16 对「簇」精确复核**全为 0.0000**（作品中位 253 字，锚点集太小）。**旧默认 0.02 落在真同源内部**（最小真值 0.0159），已扫阈值修正，见 `2026-09-15-min-exact-sensitivity.md` |

守卫 3/5 补齐后六个小桶跨 split 簇**全部归零**，gudian 保留 3 个真簇。

## 4 修复前后（B 组片段级检查，150 探针/split）

manifest 快照指纹 `ba915011fd4ce482`（192,823 部 / 242,870,927 字）

| 桶 | 检查 | v1（修复前） | v3（修复后） |
|---|---|---|---|
| gudian | val 40 字片段在 train 中 | **18/150（12.00%）** | **0/150（0.00%）** |
| gudian | test 40 字片段在 train 中 | **22/150（14.67%）** | **3/150（2.00%）** ← 门禁仍失败 |
| xiandai | val / test | 0/150、0/150 | 0/150、0/150 |
| story | val / test | 0/150、0/150 | 0/150、0/150 |

逐部作品归因（300 探针/部，`_residual.py`）：

| 留出作品 | v1 命中 | 归属 | v3 |
|---|---|---|---|
| 新齐谐 | **127/300（42.33%）** | 子不语 | 移入 train |
| 三国演义 | **100/300（33.33%）** | 三国志演义 | 移入 train |
| 警世通言 | 20/300（6.67%） | 今古奇观 | 移入 train |
| 二刻拍案惊奇 | 7/300（2.33%） | 初刻拍案惊奇(4) + 今古奇观(3) | **仍留在 test，命中不变** |

切分变化与移动：**665 个簇 → 427 部移动**（localugc 424、gudian 3），新指纹 `901a6a299d899fb6`。

| 桶 | v1 (train/val/test) | v3 (train/val/test) |
|---|---|---|
| gudian | 44/5/6 | 47/4/4 |
| localugc | 153,343/19,167/19,169 | 153,761/18,964/18,954 |
| 其余 5 桶 | — | 不变 |

## 5 残余 2.00% 的根因（**已修**：`--min-exact` 0.02 → 0.005）

gudian/test 残余来自《二刻拍案惊奇》。**根因是本轮新加的守卫 5**：

| 对 | 锚点 | 精确 | 结果 |
|---|---|---|---|
| 今古奇观(train) ↔ 警世通言(test) | 0.1351 | 0.0584 | 进簇（真泄漏） |
| 今古奇观(train) ↔ 二刻拍案惊奇(test) | **0.0519** | **0.0159** | **被守卫 5 剔掉** |
| 初刻拍案惊奇(train) ↔ 二刻拍案惊奇(test) | 0.0102 | 0.0080 | 未达锚点阈值 |

守卫 5 的门槛（0.02）压在决策边界上：真实重合约 1%~2%，而 story 噪声的精确复核是 **0.0**——门槛从 0 挪到 0.02 挡住了 0.0，也挡住了 0.0159。
`exact_containment` 是**确定性**的（等差 stride 采样，无 RNG），所以这不是抖动。**默认值未改**：0.005~0.019 区间的假阳性/假阴性比例只有两个数据点，需要敏感性扫描。

> **【已修 · 2026-09-15 补】扫描做完了，默认值改成 0.005。**
> 完整扫描见 `2026-09-15-min-exact-sensitivity.md`。要点：
> - 六个小桶的候选对**全量只有 24 对**（gudian 8 + story 16），升序排是**双峰**的：
>   `0.0000 ×16`，然后是 `0.0159 0.0220 0.0254 0.0409 0.0442 0.0584 0.3693 0.4214`。
>   噪声**恰好为 0**，不是「偏小」——所以 0.02 **不是落在空档里，而是落在真同源内部**。
> - ★ **`跨 split 簇数` 对这次回退是盲的**：0.005 是 3、0.02 是 3、0.05 还是 3。
>   差别只在**簇成员**：`≤0.0159` 时 `gudian_012`《二刻拍案惊奇》在簇里（→移入 train），
>   `≥0.016` 时它掉出簇、留在 test 继续泄漏。**只看聚合指标会以为「和上轮一致、没退化」。**
> - 端到端：gudian 44/5/6 → **48/4/3**，val **18/150(12.00%)→0/150(0.00%)**、
>   test **22/150(14.67%)→0/150(0.00%)**，**门禁通过**；story 的 16 对噪声全剔。
> - **但完整清单还没重生成**：语料已从 192,823 部涨到 **225,756 部**（localugc 正文
>   2.19 亿字 → **18.9 亿字**），localugc 那部分必须重跑，当前受内存限制（见新文档 §7）。

## 6 本轮未修掉的：并查集链式放大

最大 localugc 簇 **232 部**，但成员间没有直接边：

```
localugc_001 与 localugc_002 之间无直接边
路径：001 --(0.3684)-- localugc_3543 --(0.0806)-- localugc_002
localugc_002 的直接邻居只有 1 个：localugc_3543（0.0806）
localugc_001 的直接邻居 61 个（最大 cont 0.8799）
```

3,865 条边的重合度分布：p10=0.076 / p25=0.172 / **p50=0.644** / p75=0.911；
`cont ≥ 0.3` 有 2,552 条，`≥ 0.999` 有 440 条。
662 个簇中 **111 个连一条强边（≥0.3）都没有**；232 部那个簇 930 条边只有 **122 条**强边。

判定：**大多数簇是真密集团**，链式放大集中在少数枢纽作品（`localugc_001` 是 324 万字巨型聚合帖）。
放大代价小（移动 427 部 = 全库 0.22%，localugc 留出集缩 1.09%），且方向安全（只缩留出集），故保留当前行为。

## 7 顺手发现：上一轮修复回退了

v1 gudian 又是 **44/5/6**、泄漏 12.00%/14.67%——与第 34 篇修复前逐字相同。
原因：修复只落在旁支 `data/manifest_v2.jsonl`，主清单被数据管线重新生成后 v2 未回流。

## 8 产物

| 文件 | 内容 |
|---|---|
| `py/audit_neardup.py` | 可扩展近重复检测（分层 + 五道守卫 + 簇级重切分 + `--dump-pairs`） |
| `docs/content/实践篇36_*.md` | 内容稿（含 7 个失败样本） |
| `data/manifest_v3.jsonl` | 簇级重切分后的**完整清单**（88,904,420 字节 = **84.79 MiB**，比 v1 只大 141 KB / +0.16%）。入库后 `splits.resolve_current()` 自动选中它 |
| `data/manifest_v3_moves.json` | 重切分**增量**（427 部移动，64 KB），保留作可审计差异；「v1 + 本文件」可确定性重建 |
| `data/split_report_v3.json` | 与 v3 配套的切分报告（指纹 `901a6a299d899fb6`） |
| `dev/verify/audit_neardup.json` | 检测报告（含全部 298 个跨 split 簇的成员表） |
| `dev/verify/audit_splits_v1.json` / `_v3.json` / `_v3mx0.json` | 修复前 / 默认 / `min-exact 0` 三份门禁对照 |
| `dev/verify/manifest_small.jsonl` | 六小桶清单（1,144 部，指纹 `9cb09686a8f0e86f`），从当前 manifest 抽出，供 `--min-exact` 扫描用 |
| `dev/verify/manifest_small_dedup.jsonl` + `split_report_small_dedup.json` | 同一批小桶去重后（新默认 0.005）清单与报告，gudian 48/4/3、门禁通过 |
| `dev/verify/audit_neardup_small.json` | 六小桶检测报告（新默认下） |
| `docs/experiments/2026-09-15-min-exact-sensitivity.md` | **`--min-exact` 敏感性扫描**：双峰分布、0.02 落在真同源内部、聚合指标盲区 |

## 9 复现

```bash
python py/audit_neardup.py --manifest <v1 快照> --report data/split_report.json \
    --out dev/verify/audit_neardup.json \
    --out-manifest data/manifest_v3.jsonl \
    --out-moves data/manifest_v3_moves.json                                        # ≈3.5 min
python py/audit_splits.py --probe 150                                              # 旧切分应报门禁失败（1）
python py/audit_splits.py --manifest data/manifest_v3.jsonl \
    --report data/split_report_v3.json --probe 150                                 # gudian/test 仍有 3/150（1）
python py/audit_neardup.py --buckets gudian --algo anchors --min-exact 0 \
    --manifest <v1 快照> --report data/split_report.json \
    --out-manifest /tmp/v3mx0.jsonl                       # gudian 48/4/3，门禁通过（0）
```

重建校验：把 `data/manifest_v3_moves.json` 里每个 `story_id` 的 `split` 改成 `to`，
指纹应为 `901a6a299d899fb6`（192,823 部）。

## 10 下一步

1. 用**枢纽法**替换并查集传递闭包（每个簇至少一条强边）——需先量化 111 个全弱边簇里有多少是真簇（`今古奇观` 簇的边全在 0.058~0.135，是**真**选本，说明「全弱边 = 假」不成立）。
2. ~~给 `--min-exact` 做敏感性扫描~~ **✅ 已做**：`2026-09-15-min-exact-sensitivity.md`，默认值改为 0.005，gudian 门禁由失败转**通过**。
3. 给 localugc 设计可负担的片段级探针（当前只有作品级证据，口径与 gudian 不一致）。
4. 把「修复回流」做成管线的一步：v3 之后 `data/manifest.jsonl` 应直接采用 v3，否则下一次重新生成切分又会回退。
5. **★ 语料换代导致 v3 作废（2026-09-15 新增）**：`data/manifest.jsonl` 已从 192,823 部涨到
   **225,756 部**（localugc 正文 2.19 亿 → 18.9 亿字），`data/manifest_v3.jsonl` 成了**过期快照**，
   `resolve_current()` 的新鲜度校验会拒绝它并每次打告警。→ 必须在**新语料上重跑** localugc 的行级检测
   （当前被内存挡住：可用 RAM 1.6 GB vs 正文 5.3 GB），再产出新一代清单。
6. `py/splits.py` 的 `DEDUP_CANDIDATES` 是**硬编码 `[v3, v2]`**——每换一代都要手改，且写错就静默回退。
   建议改成按 `data/manifest_v*.jsonl` **通配 + 版本号降序**，让「落盘即被选中」自动成立。
7. 其余阈值（锚点 `--thresh 0.05` / 最少共享 200 字 / df 200 / `--min-shared 5`）仍是拍定的，
   本轮只扫了 `--min-exact`。
