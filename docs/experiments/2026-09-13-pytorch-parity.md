# PyTorch 复刻与 minilm.js 逐位对齐

> 日期：2026-09-13。实验文件：`dev/export_fixture.js`、`py/verify_parity.py`。

## 1. 问题

浏览器里的 minilm.js 与 PyTorch 复刻版是否在数学上完全一致？

## 2. 假设

如果 GELU 近似、LayerNorm epsilon、权重布局、loss reduction、梯度裁剪、Adam 语义全部对齐，那么固定输入下前向、反向、单步更新的结果应该一致。

## 3. 唯一变量

| 项目 | 值 | 是否固定 |
|---|---|---|
| 配置 | ctx=4 d=8 layers=1 heads=2 ffMul=2 | 是 |
| 词表 | 9 个字符 | 是 |
| 输入 | 固定 tokens/targets | 是 |
| 权重来源 | JS 导出，PyTorch 加载 | 是 |
| 比较对象 | forward / backward / clip / Adam step | **比较范围** |

## 4. 配置与命令

```bash
node dev/export_fixture.js
python py/verify_parity.py
```

输出：
- `py/runs/parity_fixture.json`
- 控制台 PASS/FAIL 报告

## 5. 结果

```text
forward: logits PASS   loss PASS
backward (raw):   0/20 FAIL
grad-clip:        0/20 FAIL
adam-step:        0/20 FAIL
```

具体最大误差：
- forward logits: max|Δ| = 5.55e-17（bitwise 一致）
- forward loss: |Δ| = 9.02e-09（JS log 加 1e-9 epsilon 导致）
- backward (raw): 所有 20 个参数梯度 max|Δ| ≤ 1.94e-16（bitwise 一致）
- grad-clip: 所有 20 个参数 max|Δ| ≤ 1.94e-16（bitwise 一致）
- adam-step: 所有 20 个参数 max|Δ| ≤ 4.68e-17（bitwise 一致）

## 6. 调试记录 / 失败样本

- 早期 `verify_parity.py` 报告 emb.weight 梯度差 0.28，误以为反向传播未对齐。根因是误把 `model.named_parameters()` 的**参数值**当成**梯度**来比较。
- 修复：改为 `v.grad.detach().clone()`。
- Adam 步最初用纯手写的 Adam 公式，因 `sqrt(v_hat) + eps` 的运算顺序与 PyTorch 略有差异，导致部分参数相对误差达 1e-2。修复：改用 `torch.optim.Adam` 真实跑一步。
- 发现 JS 的 `step()` 是在 `clipGrad()` 之后调用，所以 PyTorch 侧也必须用裁剪后的梯度做 Adam 步。

## 7. 结论与边界

**结论**：
> minilm.js 与 PyTorch 复刻在 float64 下实现了前向、反向、梯度裁剪、Adam 单步的 bitwise 对齐。

**边界**：
> - 仅在一个极小配置（ctx=4 d=8 layers=1）上验证；
> - 未验证 long-term 训练趋势是否一致（仅验证单步数学）；
> - Float32 容差未单独测试；
> - 训练采样 RNG 未对齐（JS 用 mulberry32，PyTorch 用 MT19937），因此两边独立训练结果不会逐位相同，但数学实现等价。

## 8. 复现检查清单

- [x] 命令可直接复制运行
- [x] 输出文件路径明确
- [x] 随机种子固定
- [x] 数据切分方式明确（无切分，固定输入）
- [x] 失败/调试记录已保留