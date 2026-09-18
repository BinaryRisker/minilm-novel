# 第一次训练：先看见变化，再学会保存和恢复

更新：2026-09-17。目标是亲手走完一次小规模训练，不要求先掌握微积分，也不需要租 GPU。命令在项目根目录运行；Python 部分用 Windows PowerShell 写法。

## 第一步：不用安装，观察模型变化

双击根目录的 `index.html`，选语料，记下配置和 seed。先用一句开头生成并保存完整输出，再训练，观察 loss，最后用相同开头生成。

记录三件事：标点是否更合理、是否连续重复、人物和事件是否连起来。曲线下降说明更适应练习题，不代表学会故事。此页面用内置语料，不会自动读取磁盘上的全部小说；它和 Python 正文数据不是同一份教材。

## 第二步：准备 Python

已有能导入 torch 和 numpy 的环境就直接使用，否则在根目录逐行执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy
```

下文 `python` 指安装依赖的同一个解释器。使用虚拟环境时换成 `.\.venv\Scripts\python.exe`，不必修改 PowerShell 执行策略。其他系统或 GPU 安装按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择。

```powershell
python -X utf8 -c "import sys,torch,numpy; print(sys.executable); print(torch.__version__); print('CUDA:',torch.cuda.is_available())"
```

CPU 练习中 `CUDA: False` 正常。`-X utf8` 避免 Windows 中文编码问题。

## 第三步：确认教材，再训练

需要已有的 `data/manifest.jsonl`、对应切分报告及 `data/processed/story/` 正文。只克隆代码未必包含大文件。先检查：

```powershell
Test-Path data/manifest.jsonl
Test-Path data/processed/story
python -X utf8 py/train.py --help
```

正文缺失时，先准备数据或停留在浏览器练习，不要创建空文件绕过错误。story 是合成故事工程样本，不代表自然小说全体。每次使用新的输出目录；下例 checkpoint 会被续训更新。

```powershell
python -X utf8 py/train.py --bucket story --chars 4000 --steps 80 --stop_at 40 --batch 2 --seed 1234 --device cpu --val_chars 512 --snapshot_every 20 --out runs/my_first_lesson
```

| 参数 | 这次是什么意思 |
|---|---|
| `--bucket story` | 从作品清单读取该桶的 train/val/test |
| `--chars 4000` | 截取训练文本前 4000 字做教学短跑 |
| `--steps 80` | 计划总共更新 80 步 |
| `--stop_at 40` | 这阶段在第 40 步存盘停下 |
| `--batch 2` | 每步取两段训练窗口 |
| `--seed 1234` | 固定初始化与采样起点 |
| `--val_chars 512` | 用一小段验证文本检查流程 |

应看到 step、loss、val 和存档路径。`val nll` 是未用于梯度更新的数据上的预测损失；只有 40 步不应期待通顺故事。若看到“过期快照/回退”，只能当教学运行，不能当数据隔离通过。

## 第四步：从 40 接到 80 步

先备份中途状态，再继续到原定终点：

```powershell
Copy-Item -LiteralPath runs/my_first_lesson/checkpoint.pt -Destination runs/my_first_lesson/checkpoint_step40.pt
python -X utf8 py/train.py --resume runs/my_first_lesson/checkpoint.pt --steps 80 --device cpu --val_chars 512 --test_chars 512 --snapshot_every 20
```

80 是“续到第 80 步”，不是“再增加 80 步”。日志应从 40 开始到 80。这只说明流程能运行；证明与连续训练等价，还需同起点对照并比较参数，见实践篇 33。

此旧入口到终点会读 test，因此这套练习归为探索。后续不可把同一测试成绩当未经查看的最终考试。正式选型走只读 train/val 的流程，见[训练手册](../FORMAL_30M_SCRATCH_TRAINING_RUNBOOK.md)。

## 第五步：生成并阅读完整样本

```powershell
python -X utf8 py/eval_story.py --checkpoint runs/my_first_lesson/checkpoint.pt --bucket story --n_prompts 2 --max_new 60 --seed 1234 --baseline none --out runs/my_first_lesson/story_eval.json
```

这里关闭基线，只检查生成链路；比较 trigram 时需另核两者训练池与评测口径。默认出题访问留出 test，仍只作教学探索。

打开 JSON，找到完整提示词和模型输出，查看陌生字、重复、语句和事件，不只看分数。关键词出现不等于事件完成；人工评分为空不能写成“通过人评”。

## 最后检查产物

| 文件 | 用途 |
|---|---|
| `train.log` | 阶段起止、loss、警告 |
| `meta.json` | 配置、数据身份、实际步数 |
| `itos.json` | 字符与编号 |
| `checkpoint_step40.pt` | 中途备份 |
| `checkpoint.pt` | 最终训练状态 |
| `story_eval.json` | 完整生成和自动指标 |

`No module named torch` 常是装依赖和运行用了不同 Python；“语料太短”要增加文本或缩短窗口；身份校验失败应查版本，不跳过校验。loss 出现 NaN/Inf 时先查数据和数值，不用延长训练掩盖问题。

本轮实际命令核验见[审查报告](文章审查_2026-09-17.md)。理解每个产物的用途，比第一次成绩多少更重要。
