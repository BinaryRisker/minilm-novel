# py/ — Python + PyTorch 训练链路（L0 起）

这是路线图里 **L0 · 平移验证** 的产物，也是今后所有真实训练（L3 数据工程 → L5 对齐）的起点。

## 先说清楚：这里有**两条**语料线

这一节是踩过坑之后补的。早期 `data/corpus.json` 是从 `corpus.js` 导出的，
所以"JS 侧是唯一真源"成立；**加入公版语料抓取管线后这句话不再成立**：

| | 浏览器实验台 | Python 训练链路 |
|---|---|---|
| 语料文件 | `corpus.js`（项目自写 3 篇，**7360 字**） | `py/data/corpus.json`（公版抓取，**807 万字**）＋ `data/splits/`（按作品切分，**2682 万字**） |
| 谁生成 | 人工撰写 | `py/data/fetch_corpus.py` → `py/data/build_splits.py` |
| 为何分开 | 单文件 HTML 要能双击即开，不宜内联几兆文本 | 训练需要量级更大的真实语料 |

Python 链路内部还有**两个层级**，别混：

- `py/data/corpus.json` —— 按桶**均分截断**（每桶 ≤200 万字）的桶级拼接文本。
  桶内按 `len*0.9` 连续切留出，**会把同一部作品劈进 train 和 test**。
  早期扫描（`sweep_data_scale.py`）走这条。
- `data/splits/{train,val,test}.txt` —— **按作品整篇切分**，train/val/test 不跨作品，
  正文取 `data/processed/` 里的逐作品全文（不截断）。**做质量结论请用这条**
  （`py/sweep_worksplit.py`、`py/train.py --splits`）。

两条线**互不覆盖**。唯一需要小心的是逐位对齐验证：它的 fixture 由
`dev/export_fixture.js` 从 `corpus.js` 导出（小配置 V=9），因此对齐结论只依赖
`corpus.js` 与 `minilm_torch.py` 的数学，与公版语料无关。

### ⚠️ 分桶比较前先看这张分级表

`wsplit_*` 的结果按桶列出，但**5 个桶的可比性不一样**（2026-09-14 审计）：

| 分级 | 桶 | 为什么 |
|---|---|---|
| **A 可用作证据** | `gudian` / `wuxia` / `urban` | 公版自然文本、多部作品、按作品留出 |
| **B 只作工程样本** | `story` | 1000 部全是 `TinyStoriesChinese`（**GPT 生成 + 机翻**合成短篇，中位 253 字，留出是 100 个碎片）→ trigram 异常强，与自然文本桶并列排名无意义 |
| **C 只作跨域对照** | `scifi` | 池里 **92.6%** 是**一部现代网文**（HF 上传、许可存疑），唯一留出作品《貓城記》是 **1932 年**作品 → 测的是跨域迁移；且单部作品**无法既训练又留出**，不具备独立成桶的资格 |

**不要只删数字，要把「这一列为什么不能读」写进文档**——完整依据见
`docs/experiments/2026-09-14-worksplit-nll-converge.md` §9.1–9.3 与 `docs/DATA_CANDIDATES.md` §1.2。

### ⚠️ Top-1 落后不是「模型太小」

同日做了**只换主体**的容量对照（40,800 → 202,112 参数，**4.95×**，数据/步数/种子/评测全不变）：
≥5 万字档 Top-1 边距一致改善 **+0.3 ~ +3.4pt**，但 **A 级桶最大档仍 −0.2 ~ −3.5pt，缺口没被填平**；
而 **NLL 优势继续扩大**。代价 **2.2× 耗时**。结论是**指标口径**问题（trigram 的 Top-1 随语料增长涨得更快）。
见 `docs/experiments/2026-09-14-bigbody-capacity-control.md`，逐格对照用 `py/compare_cfg.py`。

### ⚠️ Top-1 落后也不是「上下文太短」

同日又做了**只换 ctx**的对照（32 → 64，**token 预算对齐**）：≥1 万字的 **20 格 Top-1 没有一格变好**；
只有 2,000 字档（≈102 epoch 的记忆区间）变好，而那属于"记忆容量"不是"泛化"。

决定性的是**步数对齐**的 7 臂归因（20 万字档 × 3 桶 × 3 种子）：

| 步数（更新次数） | ctx=32 | ctx=64 | ctx=128 |
|---|---|---|---|
| 293 | −1.81 | **−1.83** | — |
| 586 | −0.33 | −0.09 | **+0.04** |

（表内 = Top-1 边距「模型 − trigram」）**同为 293 步时 ctx32 与 ctx64 逐格相同**——
差距来自**更新次数**，不是上下文长度。步数 586→293→146 时边距 −0.33 → −1.83 → −3.35。
**在 586 步下，模型终于在 Top-1 上与 trigram 打平。**

> ⚠️ **扫超参时的陷阱**：`steps = epochs × 语料量 / (batch × ctx)`。
> 「固定 epochs」= 固定 **token**，但**步数 ∝ 1/ctx** —— 直接比 ctx 会把
> "优化预算被砍半"误读成"长上下文有害"。扫 ctx 前先决定对齐 **token** 还是**步数**；
> 若沿用 `min_steps` 下限，还要按 1/ctx 缩（否则小档位的预算被下限静默改变）。

见 `docs/experiments/2026-09-14-ctx-length-control.md`，
逐格对照用 `py/compare_ctx.py`，机制图用 `py/plot_ctx_mechanism.py`。

**预算给足**（同日第六轮）：固定 ctx=64 / 20 万字档、只把步数 293→2344 ——
Top-1 边距 3 桶全部转正（gudian −3.19→+0.62），缺口确系优化预算；但 NLL 边距
在 2344 步回升（Top-1 晚于 NLL 过拟合）。见 `docs/experiments/2026-09-14-budget-steps.md`，
对照与图用 `py/compare_budget.py`。

**数据量扩展**（同日第七轮）：20 万→500 万字（3 epoch token 对齐，步数 6× 阈值）——
Top-1 缺口**总体变宽**（gudian 5M −6.60 全程最差），trigram Top-1 随数据涨得比模型快；
NLL 边距继续扩大。**≥1M 档缺口是数据效应，预算杠杆失效；NLL 才是模型的可持续优势**。
见 `docs/experiments/2026-09-14-datascale-to-5m.md`，对照与图用 `py/compare_datascale.py`。

## 这一站回答一个问题

> **手写的 `minilm.js` 和 PyTorch 训出来的是不是同一个模型？**

PyTorch 复刻版（`minilm_torch.py`，约 150 行）逐项对齐 JS 版的全部细节：
初始化尺度（含残差路径的 `0.02/√(2L)`）、LayerNorm eps、GELU tanh 近似、权重共享、
Adam 超参、全局梯度裁剪、warmup+余弦调度、批次采样区间、留出集评测分窗——
任何一处偏离都会让对齐验证失去意义。

## 用法

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu

# ① 逐位对齐（前向 logits/loss、原始梯度、裁剪、Adam 一步；float64）
node dev/export_fixture.js        # 从 corpus.js 导出 fixture
python py/verify_parity.py

# ② 分布对分布对齐（多种子，判据是"实现间差距 < 换种子波动"）
python align.py                   # 5 种子 × 3 语料全量验证，生成 align_report.md
python align.py --corpus scifi    # 只跑一段（结果缓存在 runs/）
python align.py --seeds 42 7      # 指定种子

# ③ 语料管线（公版抓取 → 清洗 → 去重 → 分桶 → 审计 → 按作品切分）
python py/data/fetch_corpus.py --list-sources        # 只看源清单，不下载
python py/data/fetch_corpus.py --max-chars 2000000   # 抓取 + 清洗 + 分桶
python py/data/prepare.py --split                    # 逐作品 manifest + split 审计
python py/data/build_splits.py                       # 按作品整篇切 → data/splits/{train,val,test}.txt

# ④ 数据量扫描 + 收敛 + 多种子（**推荐口径：按作品切分，无泄漏**）
python py/sweep_worksplit.py                         # 5 桶 × 5 尺寸 × 3 种子 × 3 epoch
python py/sweep_worksplit.py --buckets scifi --sizes 2000,10000,50000 --seeds 1234 --epochs 2
python py/sweep_worksplit.py --report_only           # 只把**本 tag** 的结果合并成报告 + 两张图
#    大模型对照（容量 vs 结构）：训练数据/步数/种子/评测协议全不变，只换模型主体。
#    --tag 会插进所有输出文件名（含图与报告），所以两套结果可并列、互不覆盖；
#    注意 --report_only 必须带上同一个 --tag，否则合并的是小主体那一套。
python py/sweep_worksplit.py --cfg_d 64 --cfg_layers 4 --cfg_heads 8 --tag big --no_report
python py/sweep_worksplit.py --report_only --tag big  # → wsplit_report_big.md + worksplit_*_big.png
python py/compare_cfg.py                              # 大/小主体逐格对照 ΔTop-1 / ΔNLL + 参数量分账
#    上下文长度对照（ctx 32 vs 64 vs 128）：**必须先决定对齐 token 还是步数**。
#    steps = epochs × 语料量 / (batch × ctx)，所以「固定 epochs」= 固定 token、步数 ∝ 1/ctx。
#    还要把 min_steps 按 1/ctx 缩（32→200 / 64→100 / 128→50），否则下限会静默改变预算。
for bk in scifi wuxia urban gudian story; do
  python py/sweep_worksplit.py --buckets $bk --sizes 2000,10000,50000,200000,1000000 \
         --seeds 1234,42,2024 --epochs 3 --ctx 64 --min_steps 100 --tag ctx64 --no_report --no_plot &
done; wait
python py/sweep_worksplit.py --report_only --tag ctx64
python py/compare_ctx.py --tags 'base=,ctx64'        # 逐格 Top-1/Top-5/NLL 边距 + 基线漂移 + 预算核对
python py/plot_ctx_mechanism.py                      # 按「训练步数」对齐的机制图 → docs/img/ctx_mechanism.png
python py/compare_budget.py                          # 预算给足：边距 × 步数表 + 曲线 → docs/img/budget_steps.png
python py/compare_datascale.py                       # 数据量扩展：跨档位边距走势 + 增量分解 → docs/img/datascale_margins.png
#    泄漏受控 A/B（同一训练、只换留出集，量化泄漏对模型/基线各自的影响）：
python py/sweep_worksplit.py --buckets wuxia --sizes 200000 --leak_probe \
       --no_report --no_alpha_probe --out_dir py/runs/leakprobe

# ④′ 早期口径（桶内连续 90/10，**含同作品泄漏**，仅作历史对照）
python py/sweep_data_scale.py --sizes 2000,10000,50000,200000,1000000,all --steps 400 --min_freq 3
python py/sweep_report.py                                # 合并全部批次 → 报告 + 折线图

# ⑤ 测试
python -m unittest discover -s py/tests -t . -v
```

## 验收标准（M0）

1. **损失曲线**与 JS 版同量级重合；
2. **留出集 Top-1**：JS 参考值落在 PyTorch 多种子分布的均值 ±2σ 内。

第 2 条是关键设计：单次训练 400 步、留出集只有 256 个评测位置，Top-1 的种子间波动
本身就有几个百分点。**实现间差距必须小于"同一个实现换种子"的波动**，对比才有意义——
固定 "±2pt" 这种阈值反而是在假装噪声不存在。

## 文件

| 文件 | 作用 |
|---|---|
| `minilm_torch.py` | PyTorch 复刻：模型 / 分词器 / 批次 / 调度 / 训练 / 留出集评测。`build_vocab` 支持 `min_freq`/`max_vocab` 裁剪词表，**默认 `min_freq=1` 走与 `minilm.js` 逐位一致的路径**（对齐验证依赖此路径） |
| `verify_parity.py` | 逐位对齐：forward logits/loss、原始梯度、裁剪后梯度、Adam 一步（float64，20 参数逐项） |
| `align.py` | 多种子对齐验证驱动，产出 `align_report.md` |
| `baselines.py` | 统计基线：unigram / bigram / trigram（加 α 平滑 + 回退），**稀疏计数**存储；`for_each_position` 定义共享评测滑窗（每 `ctx` 步一窗，与模型同口径） |
| `sweep_worksplit.py` | **数据量扫描 + 收敛曲线 + 多种子误差棒**（现行口径）。吃 `data/splits/`，留出作品不参与训练；模型与 trigram 同口径记 NLL/PPL；`--tag` 并列多套模型主体；`--report_only` 只合并**同 tag** 的结果 |
| `compare_cfg.py` | 两组**同协议、仅模型主体不同**的结果对照：逐格 ΔTop-1 / ΔNLL + 参数量分账（词表嵌入 vs 主体），回答「差距是容量还是结构」 |
| `compare_ctx.py` | 多组**同协议、仅上下文长度不同**的结果对照：逐格 Top-1/Top-5/NLL **边距**（模型 − trigram）+ **基线自身漂移** + compute-matched 核对（打印每格的 `ctx/步数/有效 epoch`），回答「差距是不是上下文太短」 |
| `compare_budget.py` | **预算给足**对照：固定 ctx/档位、只改步数（epochs），输出边距 × 步数表 + 曲线图（`docs/img/budget_steps.png`），并逐种子核对稳定性，回答「差距是不是优化预算不够 / 预算给足后会不会过拟合」 |
| `compare_datascale.py` | **数据量扩展**对照：同一 tag 家族内跨档位（200k→5M）的边距走势 + **基线/模型增量分解**（谁随数据涨得快）+ 最大档逐种子核对，回答「缺口随数据加大是收敛还是变宽」 |
| `plot_ctx_mechanism.py` | 把各实验臂按**训练步数**对齐画在一张图里（`docs/img/ctx_mechanism.png`）：同 ctx 的曲线随步数下移，而不同 ctx 在相同步数处重合 → 一眼区分「上下文效应」与「步数效应」 |
| `sweep_data_scale.py` | 早期数据量扫描（桶内**连续** 90/10，含同作品泄漏）。**已停止维护**，仅作历史口径对照 |
| `sweep_report.py` | 合并 `runs/sweep_*.json` 全部批次 → `runs/sweep_report.md` + `docs/img/sweep_data_scale.png` |
| `ablation.py` | 结构消融：注意力头数 ∈ {1,2,4,8} × 位置嵌入（学习式 / 置零冻结） |
| `data/fetch_corpus.py` | **公版语料抓取管线**：Gutenberg / GitHub 古典文库 / 维基文库 → 清洗 → 跨源去重 → 分桶均分截断 |
| `data/prepare.py` | 逐作品 manifest（train/val/test 不跨作品）+ split 审计 |
| `data/build_splits.py` | 按作品**整篇**切分 → `data/splits/{train,val,test}.txt`（含无跨集合 + 字数守恒自检） |
| `data/corpus.json` | 桶级训练语料（公版，807 万字，按桶均分截断），**由 `fetch_corpus.py` 生成**，不再从 `corpus.js` 导出 |
| `tokenizer.py` | 极简 BPE tokenizer（学习用，无外部依赖） |
| `train.py` | 训练 CLI：**种子前置**、checkpoint v2（含优化器/步数/RNG）、真·断点续训（`--resume` + `--stop_at`）、`--bucket` 按桶读按作品切分、val 选点/test 只评一次 |
| `eval_story.py` | 场景写作评测（重写）：提示词取自**留出作品**（记录 provenance）、`--seed` 可复现、指标只算生成段（旧口径另存作对照）、`--baseline trigram` 同参数对照 |
| `splits.py` | 按作品切分的统一读取入口（`bucket_texts`/`bucket_joined`，词表只在 train 上建）；`verify()` 切分身份校验（指纹不符 / 未去重一律非零退出） |
| `repro.py` | 可复现性原语：`set_global_seed`（random/numpy/torch）+ `rng_state`/`set_rng_state`（把全局 RNG 与 torch RNG 一起打包，断点续训用） |
| `audit_splits.py` | 数据隔离审计（门禁）：清单级 / 片段级精确查找 / 词表 OOV / 训练集内重复 / 旧口径对照 |
| `audit_neardup.py` | **可扩展近重复检测**（19 万部量级）：`auto` 分层——小桶走锚点法（对齐无关，能抓「同书异本」）、大桶走行级倒排；**五道守卫**（行级：df 绝对+占比 / 最少共享行 / 最少共享字数；锚点：锚点 df 过滤 / 精确复核门槛 `--min-exact`）；`--out-manifest` 产出重切分清单、`--out-moves` 只产出增量（完整清单 105 MB 超 GitHub 100 MB 上限，不入库）、`--dump-pairs` 导出逐条边用于诊断并查集链式放大 |
| `dedup_works.py` | 近重复合并（锚点法）：异名同书 / 选本重复合并成簇整簇同 split → `data/manifest_v2.jsonl`；`resplit()` 抽成函数被 `audit_neardup.py` 复用 |
| `leak_probe.py` | 同源泄漏值多少分：同一模型在「污染测试集」与「干净测试集」上逐作品对比 |
| `verify_seed.py` / `verify_resume.py` / `verify_story_eval.py` | 三份可复核判定：种子逐位可复现 / 续训与直训逐位等价 / 出题来自留出集 + 生成可复现 + 旧口径不送分 |
| `closed_loop.py` | 随机初始化最小闭环：训练 → 中途存盘 → 续训 → 留出评测 → 生成故事（报告含失败样本） |
| `benchmark.py` | 训练吞吐与资源实测（tokens/s、单步耗时、峰值内存） |
| `tests/` | 单测：`test_baselines.py`（稀疏 NGram 对稠密参考逐点差分 + 滑窗步长协议）、`test_tokenizer.py` |
| `runs/` | 每个 语料×种子 的结果缓存（JSON）；扫描结果 `sweep_*.json` 亦在此 |
| `align_report.md` | 对齐报告（分布对分布的数字与结论） |
