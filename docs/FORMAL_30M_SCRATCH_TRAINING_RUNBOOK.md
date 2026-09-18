# 随机初始化模型训练手册：当前可运行路径

更新：2026-09-17。先完成[CPU 入门练习](content/第一次训练_从浏览器到Python.md)。本手册说明现行分片入口，不代表数据、GPU 恢复和故事质量已全部验收。

## 现行模型与旧设计不同

`py/novellm/model.py` 的 t30m 复用 MiniLM，ctx=1024、d=512、8 层、8 头、ffMul=4，使用学习式绝对位置、LayerNorm、GELU、共享嵌入输出，参数随机初始化。实际数量由代码核对。

旧手册的 29,893,120 参数、RoPE/RMSNorm/SwiGLU、独立 BPE 管线是设计方案，尚非这个实现。`py.novellm.prepare_data`、`fit_tokenizer`、`smoke`、`benchmark`、`train` 这些旧示意模块不存在，不能执行。

## 固定数据与环境

记录 commit、依赖版本、设备、清单、切分报告和正文指纹；先完成作品家族隔离与用途核对。数据构建期间不能改写同一份正文和清单。解析器回退到未审计清单时，不进入正式质量训练。

```powershell
python -X utf8 -m py.novellm.doctor --out runs/doctor_target.json
python -X utf8 -m py.novellm.trainer --help
```

本轮核验本地 CPU 与入口，未重跑 GPU。历史短训练见[交接记录](HANDOFF_AUTODL_T30M_GPU_SMOKE_20260917.md)，不能沿用 micro 的测速预算 t30m。

## 同一份训练词表编码 train 和 val

**不能分别调用 `dataset build --split train` 和 `dataset build --split val` 后假定编号相同。** 当前 CLI 会在指定 split 上重新拟合词表。下方用现有 API，只在 train 拟合一次，再共享 itos。

将代码保存为项目根目录 `prepare_lesson_shards.py`，再用同一 Python 执行；先只做 story 工程样本。输出目录必须是新目录。

```python
from pathlib import Path
from py.novellm.dataset import fit_char_vocab, build_shards

out = Path('runs/my_story_shards')
if out.exists():
    raise SystemExit('请换一个新输出目录，避免覆盖已有分片')
itos, _ = fit_char_vocab('train', min_freq=3, buckets=['story'])
for split in ('train', 'val'):
    build_shards(split, itos, str(out), shard_tokens=1000000, buckets=['story'])
```

同一轮构建默认清单和正文要保持不变。正式数据应通过 API 的 manifest/report/processed 参数固定版本；构建通过不等于近重复审计通过。

```powershell
python -X utf8 -m py.novellm.dataset info --split train --dir runs/my_story_shards
python -X utf8 -m py.novellm.dataset info --split val --dir runs/my_story_shards
python -X utf8 -m py.novellm.dataset verify --split train --dir runs/my_story_shards
python -X utf8 -m py.novellm.dataset verify --split val --dir runs/my_story_shards
```

核对 train/val metadata 的词表指纹和源切分指纹一致，以及文档数、token 数与 UNK。当前路径是字符编码，尚非未来设计的 BPE、EOS/PAD 与专用 loss mask 链路。

## 先短测，再扩大

```powershell
python -X utf8 -m py.novellm.trainer --shard-dir runs/my_story_shards --split train --val-split val --cfg micro --device cpu --steps 10 --batch 1 --accum 1 --lr 0.001 --val-every 5 --val-windows 4 --snapshot-every 5 --out runs/my_micro_smoke
```

这个 micro 也大于最早 d=40 的教学模型，速度不可直接类比。确认加载、更新、验证和存盘后，才考虑 GPU。下方仅在 CUDA 可用、数据检查通过后运行，会消耗租卡时间；学习率只是起步试验值。

```powershell
python -X utf8 -m py.novellm.trainer --shard-dir runs/my_story_shards --split train --val-split val --cfg t30m --device cuda --steps 100 --batch 1 --accum 4 --lr 0.001 --val-every 20 --val-windows 32 --snapshot-every 20 --out runs/my_t30m_smoke
```

记录实际参数、配置、时间、有效 token/s、显存、验证曲线与存档。小 story 分片仅用于工程短测，多次曝光容易过拟合。本训练器只读指定 train/val，不要用 test 选型；当前完整独立 final_eval 发布链路还需实现。

## 恢复和交付

恢复时显式保持 cfg、shard-dir、batch、accum、lr、seed 和计划总步数，再提供 `--resume`。精确等价必须在同一计划下中断，不能先计划 100 步、再改成 500 步后要求学习率轨迹不变。

Ctrl+C 或 max-hours 可触发安全存盘，但外部强杀、磁盘故障没有保存保证。周期快照写同一路径，不会自动保留所有历史点或最佳模型；要独立备份，并在目标设备比较直训与恢复。

时间用计划 token 除以实测有效 token/s，再加验证、存盘、预处理；费用用控制台当期单价。显存要算权重、梯度、优化器、激活与临时张量，不只算权重文件。

交付包括完整样本、失败分析与人评记录。没有故事质量证据时，写“工程运行通过，故事能力待验收”。
