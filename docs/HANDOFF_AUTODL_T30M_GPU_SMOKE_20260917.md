# AutoDL t30m GPU 冒烟修复与交接记录

日期：2026-09-17
目的：承接 AutoDL 第一轮运行结果，修正“实际跑成 micro 却按 30M 解读”的问题，为下一位同事准备可复现的 t30m GPU 冒烟入口。

## 结论

本轮修改已完成，本地验证通过。根据 31932 端口实例的新记录，真正的 t30m GPU 冒烟已经跑通；可以继续租用 AutoDL，但下一轮仍应先完成多桶分片和严格恢复等价测试，再决定全量训练。上一轮的 GPU 吞吐和 ¥0.03 估算不能作为 t30m 预算依据。

## 上一轮证据核对

- `dev/autodl/40_benchmark.sh` 未传 `--cfg`；`py/benchmark.py` 默认是 `d=40, layers=2, heads=4, ctx=32` 的 micro 配置。
- `py/runs/gpu30m_story_s1234/meta.json` 明确记录 `d=40, layers=2, heads=4`，D0 也是 micro。
- D0 数据是 `story` 桶的 1000 个 synthetic works、274460 chars，不是本地全部小说。
- 上一轮确实验证了 AutoDL SSH、CUDA、上传/回传、checkpoint 和指纹对账，但没有验证新 `py/novellm.trainer` 的 t30m GPU 运行。
- 31932 实例补充证明：t30m 为 26,651,648 参数，`ctx=1024,batch=2,accum=4` 训练 500 步用时 84.3 秒；验证集 nll 从 7.668 升到 8.232，说明在 804 部、约 22 万 token 的 story 分片上已出现明显过拟合。

## 本次修改

### `py/novellm/trainer.py`

- `train()` 增加 `device='cpu'` 参数，并将模型放到该设备。
- CLI 增加 `--device`；有 CUDA 时默认 `cuda`，否则 `cpu`。
- 训练 batch 的 `tokens` 和 `targets` 移到指定设备。
- `val_eval()` 根据模型参数所在设备创建验证输入，避免 GPU 模型与 CPU 输入不匹配。
- `meta.json` 增加 `device` 字段。
- 验证窗口改用 `torch.from_numpy(...).unsqueeze(0)`，避免逐元素 tensor 警告。

### `dev/autodl/40_benchmark.sh`

- benchmark 明确传入 `--cfg '{"d":512,"layers":8,"heads":8,"ffMul":4}'`。
- 后续预算表代表旧 MiniLM 数学路径下的 t30m 配置，不再默认为 micro。

### `dev/autodl/50_train.sh`

- 旧训练入口默认传 `--device cuda`，可用 `AUTODL_DEVICE=cpu` 覆盖。
- 若命令行显式提供其他 device，应以最后一个参数为准；交接时请明确写出 device，避免误判。

## 本地验证

已执行：

```powershell
python -m compileall -q py/novellm py/train.py py/benchmark.py
python -m unittest py.tests.test_trainer
git diff --check
```

结果：compileall 通过；trainer 6 个测试全部通过；diff check 无错误。31932 实例已完成远端 t30m 运行；本地本次补齐了 resume 加载时的 `device=device`。远端记录的 500→750 延续不等于“500 步直训与 250→500 续训”的严格位等价证明，仍建议补做后者。

## 下一位同事的执行顺序

### 1. 重新创建或启动实例并核对设备

```bash
bash dev/autodl/10_setup_env.sh
bash dev/autodl/40_benchmark.sh
```

检查 `dev/verify/budget_table.json` 的 `cfg` 是否为 `d=512,layers=8,heads=8,ffMul=4`，并记录 `peak_vram_mb`、`tokens_per_sec`。如果输出仍显示 micro，立即停止，不要估算正式费用。

### 2. 准备新训练器所需的预编码分片

`py/novellm.trainer` 依赖 `py/novellm/dataset.py` 生成的 `train`/`val` 分片，不能只上传 `corpus.json`。确认远端存在 shard 目录、词表和分片 metadata，并核对 source split fingerprint 与本地清单一致。

### 3. 跑真正的 t30m GPU 冒烟

建议先用 100–500 步、`ctx=1024`、`batch=1` 或 2、`--accum 4`，显式指定：

```bash
python -X utf8 -m py.novellm.trainer \
  --split train --val-split val \
  --cfg t30m --device cuda \
  --steps 500 --batch 2 --accum 4 \
  --val-every 100 --snapshot-every 100 \
  --out py/runs/gpu_t30m_smoke_s1234
```

记录实际显存、每步耗时、loss/val 曲线、checkpoint 大小和 `meta.json` 中的 `provenance.param_count`。如果 OOM，从 `batch=1` 或 `ctx=512` 开始；不要把换成 micro 当成解决 t30m 问题。

### 4. 验证恢复

先跑到 200 步，回传 checkpoint，再使用 `--resume` 续到 500 步。比较续训日志、step、配置、seed、数据指纹和最终 checkpoint。恢复成功后才进入较长训练。

### 5. 再决定 D1 全量训练

全量训练前必须重新盘点：清洗后文件数、独立作品数、token 总数、train/val/test 作品隔离、重复合并情况、分片总大小、50GB 数据盘余量。上一轮的 1000 synthetic story 只能作为冒烟数据。

## 不要据此作出的结论

- 不要把 micro benchmark 的 958771 tokens/s 外推成 t30m 速度。
- 不要把 92M token 的 ¥0.03 当作正式报价或质量成本。
- 不要把 `gpu30m_story_s1234` 目录名当成 30M 证据；以 `meta.json` 的 `cfg` 和 `provenance` 为准。
- 不要把 loss、top1 或 perplexity 单独当作小说质量验收；正式训练后仍需未见题目的完整短故事盲评。

## 文件与协作边界

本次主要改动：`py/novellm/trainer.py`、`dev/autodl/40_benchmark.sh`、`dev/autodl/50_train.sh`，以及本交接文档。仓库中其他同事正在修改的 manifest、sources、split audit、corpus 和实验产物不要覆盖或回滚。提交前请用 `git diff --name-only` 再确认暂存范围。

关联文档：[从零训练与基座微调比较](SCRATCH_VS_FINETUNING_NOVEL_GUIDE.md)、[AutoDL 集成说明](AUTODL_INTEGRATION_V1.md)、[从零训练入门指南](NOVEL_MODEL_FROM_ZERO_BEGINNER_GUIDE.md)。
