# py/ — Python + PyTorch 训练链路（L0 起）

这是路线图里 **L0 · 平移验证** 的产物，也是今后所有真实训练（L3 数据工程 → L5 对齐）的起点。

## 先说清楚：这里有**两条**语料线

这一节是踩过坑之后补的。早期 `data/corpus.json` 是从 `corpus.js` 导出的，
所以"JS 侧是唯一真源"成立；**加入公版语料抓取管线后这句话不再成立**：

| | 浏览器实验台 | Python 训练链路 |
|---|---|---|
| 语料文件 | `corpus.js`（项目自写 3 篇，**7360 字**） | `py/data/corpus.json`（公版抓取，**613 万字 / 90 部**） |
| 谁生成 | 人工撰写 | `py/data/fetch_corpus.py` |
| 为何分开 | 单文件 HTML 要能双击即开，不宜内联几兆文本 | 训练需要量级更大的真实语料 |

两条线**互不覆盖**。唯一需要小心的是逐位对齐验证：它的 fixture 由
`dev/export_fixture.js` 从 `corpus.js` 导出（小配置 V=9），因此对齐结论只依赖
`corpus.js` 与 `minilm_torch.py` 的数学，与公版语料无关。

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

# ③ 语料管线（公版抓取 → 清洗 → 去重 → 分桶 → 审计）
python py/data/fetch_corpus.py --list-sources        # 只看源清单，不下载
python py/data/fetch_corpus.py --max-chars 2000000   # 抓取 + 清洗 + 分桶
python py/data/prepare.py --split                    # 逐作品 manifest + split 审计

# ④ 数据量扫描（模型 vs trigram，"到底要多少字"）
python py/sweep_data_scale.py --sizes 2000,10000,50000,200000,1000000,all --steps 400 --min_freq 3
python py/sweep_data_scale.py ... --proportional_steps   # 步数随数据量放大（≈固定 epoch）
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
| `sweep_data_scale.py` | 数据量扫描：固定模型配置，只改训练集大小 |
| `sweep_report.py` | 合并 `runs/sweep_*.json` 全部批次 → `runs/sweep_report.md` + `docs/img/sweep_data_scale.png` |
| `ablation.py` | 结构消融：注意力头数 ∈ {1,2,4,8} × 位置嵌入（学习式 / 置零冻结） |
| `data/fetch_corpus.py` | **公版语料抓取管线**：Gutenberg / GitHub 古典文库 / 维基文库 → 清洗 → 跨源去重 → 分桶均分截断 |
| `data/prepare.py` | 逐作品 manifest（train/val/test 不跨作品）+ split 审计 |
| `data/corpus.json` | 训练语料（公版，613 万字 / 90 部），**由 `fetch_corpus.py` 生成**，不再从 `corpus.js` 导出 |
| `tokenizer.py` | 极简 BPE tokenizer（学习用，无外部依赖） |
| `train.py` | 训练 CLI：save/resume checkpoint、导出 `itos.json` 与 `meta.json` |
| `eval_story.py` | 场景写作评测：按 `eval/prompts.jsonl` 续写 + 自动指标 |
| `benchmark.py` | 训练吞吐与资源实测（tokens/s、单步耗时、峰值内存） |
| `tests/` | 单测：`test_baselines.py`（稀疏 NGram 对稠密参考逐点差分 + 滑窗步长协议）、`test_tokenizer.py` |
| `runs/` | 每个 语料×种子 的结果缓存（JSON）；扫描结果 `sweep_*.json` 亦在此 |
| `align_report.md` | 对齐报告（分布对分布的数字与结论） |
