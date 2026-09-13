# py/ — Python + PyTorch 训练链路（L0 起）

这是路线图里 **L0 · 平移验证** 的产物，也是今后所有真实训练（L3 数据工程 → L5 对齐）的起点。

## 这一站回答一个问题

> **手写的 `minilm.js` 和 PyTorch 训出来的是不是同一个模型？**

PyTorch 复刻版（`minilm_torch.py`，约 150 行）逐项对齐 JS 版的全部细节：
初始化尺度（含残差路径的 `0.02/√(2L)`）、LayerNorm eps、GELU tanh 近似、权重共享、
Adam 超参、全局梯度裁剪、warmup+余弦调度、批次采样区间、留出集评测分窗——
任何一处偏离都会让对齐验证失去意义。

## 用法

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu

python align.py                  # 5 种子 × 3 语料全量验证，生成 align_report.md
python align.py --corpus scifi   # 只跑一段（结果缓存在 runs/）
python align.py --seeds 42 7     # 指定种子
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
| `minilm_torch.py` | PyTorch 复刻：模型 / 分词器 / 批次 / 调度 / 训练 / 留出集评测 |
| `align.py` | 多种子对齐验证驱动，产出 `align_report.md` |
| `data/corpus.json` | 语料（由 `corpus.js` 一次性导出，保持 JS 侧为唯一真源） |
| `runs/` | 每个 语料×种子 的结果缓存（JSON） |
| `align_report.md` | 验证报告（对齐数字与结论） |
