# 随机初始化小说模型：研究核查与可借鉴范围

核查日期：2026-09-14。约束：正式模型从随机初始化训练。论文结果不等于本项目已复现；公开仓库不等于训练数据、权重、评测全部开放。公开文件元数据另见 [核查记录](2026-09-14-open-artifacts.json)，没有下载或运行这些权重。

## 最接近本项目的证据

| 工作与一手来源 | 核查结果 | 本项目采用的方法与适用边界 |
|---|---|---|
| [TinyStories](https://arxiv.org/abs/2305.07759) | 受限英语故事让很小的模型学会连贯短文 | 先限制词汇、人物和情节复杂度；不能据此保证中文小说质量 |
| [SimpleStories](https://arxiv.org/abs/2504.09184)、[训练代码](https://github.com/simple-stories/simple_stories_train)、[11M 模型](https://huggingface.co/SimpleStories/SimpleStories-V2-11M) | 英语、日语各约 200 万样本；参数化生成提升数据多样性，公开小模型系列与训练入口 | 最适合借鉴随机初始化的受限故事课程；需要自己建立中文数据及许可、质量审查，不能直接搬用英语结论 |
| [ChineseBabyLM v2](https://arxiv.org/abs/2607.10745) | 2026 年 8 月修订版描述 102M 中文词的数据；主要是中文理解、认知及汉字任务 | 支持研究有限中文数据的学习效率；数据单位是词，不是 token；获胜的双向编码器不是小说生成架构 |
| [SmolLM3 官方技术说明](https://huggingface.co/blog/smollm3) | 3B 随机初始化模型使用 11.2T token、384 H100、约 24 天；主要支持语言不含中文 | 借鉴数据配比、分阶段训练及公开配方；其投入不能作为消费显卡能训练同等模型的依据 |

## 长篇写作成果：借方法，不借初始化权重

| 工作与一手来源 | 开放情况、训练方式 | 可借鉴内容 / 局限 |
|---|---|---|
| [LongWriter](https://arxiv.org/abs/2408.07055)、[仓库](https://github.com/THUDM/LongWriter) | 基于已有 8B/9B 基座；训练代码、长输出数据和模型入口可核查 | 长输出需要足量长目标训练；规划再写作。项目中的模型裁判不替代本项目人工验收 |
| [LongWriter-Zero](https://arxiv.org/abs/2506.18841)、[模型](https://huggingface.co/THU-KEG/LongWriter-Zero-32B) | 基于预训练 Qwen2.5-32B；Zero 指强化学习没有 SFT 热身，不是随机初始化 | 后期奖励设计的参考；不在尚无语言能力的小模型上直接照搬 RL |
| [StoryWriter](https://arxiv.org/abs/2506.16445)、[仓库](https://github.com/THU-KEG/StoryWriter) | 论文约 6k 故事，仓库描述约 5k；公开部分脚本，README 数据链接仍为占位文本；未核实完整权重与数据发布 | 事件关系大纲、章节规划、动态历史压缩；按部分开放研究看待，不能标记为完整复现基线 |
| [PageStorm / LongPage](https://arxiv.org/html/2605.17064v2)、[模型](https://huggingface.co/Pageshift-Entertainment/pagestorm-research-preview-14b-full-book)、[推理仓库](https://github.com/Pageshift-ai/pagestorm) | 2026 年 6 月修订；从 Ministral 3 14B Base 继续训练。模型卡记录 150B token、256K 序列；公开全书权重与推理入口，未核实完整训练配方代码 | 从人类正文反向生成层级计划，让最终写作目标保留人类文本。论文评测包含模型抽取的叙事诊断和首章质量，不能证明整本小说已通过独立人类检验。模型卡还限制单轮使用 |
| [LongPage 数据](https://huggingface.co/datasets/Pageshift-Entertainment/LongPage) | 6067 条英语书籍记录；正文公版、生成脚手架 CC BY 4.0；仅 train。卡片明确缺少原作者、原书名、年份与 Gutenberg ID 等溯源字段 | 借数据结构；本项目必须补足来源和证据位置，不直接把该数据集当中文训练、最终盲测两用数据 |
| [中文 WebNovel Writer 27B](https://huggingface.co/wcn123/Qwen3.5-27B-WebNovel-Writer-zh) | 基于 Qwen3.5；公开权重，卡片自述 SFT 20320 条、ORPO 8000 条；训练数据未公开，许可有研究/个人用途限制 | 可作外部研究对照。缺少完整数据和独立人评，不能等同可复现中文小说训练方案，不能用作本项目初始化 |

## 评测与修订

- [WebNovelBench](https://arxiv.org/html/2505.14818v1) 提供中文网文场景下的评测思路。借鉴梗概条件及叙事维度；模型裁判分数、特定网文分布上的排名不等于读者验收。
- [LongWeave](https://arxiv.org/html/2510.24345v1) 将长输出要求拆为可验证约束。用于长度、事件顺序等硬条件；文学质量另行人工评判。
- [CraftAlign](https://arxiv.org/html/2608.01377v1) 将叙事特征变成具体修订指导。人评仅为经过筛选的 20 组、8 名评审，研究的人类相似性不等于完整小说质量。本项目借鉴有正文证据的修订，不优化 AI 检测器分数。
- [Chinchilla](https://arxiv.org/abs/2203.15556) 支持联合考虑模型规模与训练数据。不能把某个 token/参数比例当作达到小说能力的充分条件；用本项目受控实验决定规模。

## 设计结论（本项目推导，尚待实验）

最可行的起点是“受限中文完整故事数据 + 约 30M 自训练因果模型 + 可靠的续训和盲测”，随后才加入层级规划。能输出数万字、能保持故事一致、能让人愿意读完，是三个不同目标，必须分开报告。

现有证据没有证明：用当前约 5500 万中文字符和 8GB 显存，就能训练出通用、稳定的长篇小说模型。计划应提供失败后如何缩小任务、补数据或扩算力的决策依据，而不是承诺某个参数量必然成功。
