# 训练数据候选清单

**核对日期**：2026-09-14
**核对方式**：直接查询 HuggingFace / GitHub API 与数据集卡片，实测记录 id、体积、许可、字段结构；**未下载全量数据**（除注明外，体积取自仓库文件清单合计）。
**为何需要这份文档**：PRD 前提要求「候选外部作品、数据、模型须记录出处与可用条件」，且「先准备自有小样本；先查质量与多样性，再决定扩充规模」。

> ⚠️ **2026-09-14 更正（判定标准一致性）**：本文原先对「上传者自贴许可」采用了**双标**——
> 对第二梯队的 `wdndev/webnovel-chinese`、`lainka0o0/chinese-novel-nonH-collect` 判为「标签无效」，
> 却把性质完全相同的 `asd567557275/chinese_novel`（《空間魔導書與少年魔法師》，**现代网文**）
> 列入「第一梯队 / 许可干净」并排在接入建议第 1 位。现统一判定：**上传者非权利人的自贴标签一律无效**，
> 只可本地学习实验，且**不入评测桶**。详见 §1.2 与 §2。

---

## 一、第一梯队：现在就能用

### 1. `adam89/TinyStoriesChinese` ★ 最贴合当前阶段

- 链接：https://huggingface.co/datasets/adam89/TinyStoriesChinese
- 体积：**2.6 GB**（单个 `TinyStories_all_data_zh.tar.gz`）
- 许可：**CDLA-Sharing-1.0**（允许使用与再分发，衍生数据须沿用同一许可；上游为 GPT 生成 + 机器翻译，商用前需自行确认）
- 结构：每行一个**完整短故事**，字段 `story`（英文原文）/ `instruction`（写作提示）/ `story_zh`（中文译文）
- 上游：[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 的中文译文（只译了 story 字段）
- **为什么适合**：
  1. 每个样本就是一篇独立故事 → `story_id` 天然成立，PRD「故事家族不跨集合」可直接落实；
  2. 词汇窄、语法简单、长度短 → 正对应方案第一节 TinyStories 实证（"又窄又浅的模型也能语法正确"）与 TODO P2「数百个短故事作管线探索集」；
  3. 来源是合成数据，不涉及原著版权。
- 局限：故事极简，风格与中文网文/文学差得远；只能用来**跑通管线、验证流程**，不能指望它产出好小说。

### 2. `asd567557275/chinese_novel`（《空間魔導書與少年魔法師》）⚠️ **已降级，见下方许可更正**

- 链接：https://huggingface.co/datasets/asd567557275/chinese_novel
- 体积：**47 MB**，315 个文件，**繁体中文**，单部作品
- 许可：卡片自贴 **CC BY 4.0**（署名「睡半夜怎麼三更」）—— ⚠️ **更正：与第二梯队同一性质**。
  这是一部**现代网文**，上传者 `asd567557275` 是搬运方**而非权利人**，自贴标签**不覆盖原著版权**。
  按本文第 51 / 56 行对 `wdndev/webnovel-chinese`、`lainka0o0/chinese-novel-nonH-collect` 的同一判定标准，
  它应同样记为「仅限本地学习实验，**不可再分发 / 商用**」——原文把它列入「第一梯队 / 许可干净」并排在
  接入建议第 1 位，属**判定标准不一致**（2026-09-14 更正）。
- 结构（4 个 jsonl，字段已实测）：
  - `chapters.jsonl`：章元数据 —— `chapter / part_no / part / chapter_in_part / title / pid_start / pid_end / n_paras / n_chars / file`
  - `novel_by_chapter.jsonl`：`chapter_id / title / text` ← **正文按章切好**
  - `pretrain.jsonl`：`id / part / title / text` ← 预训练友好格式
  - `space_grimoire.jsonl`：实测内容疑似只有标题行，价值存疑
  - 另有 12 个分「篇」markdown + 298 个逐章文件 + 全书单文件
- **为什么适合**：章节边界清晰，`chapters.jsonl` 自带 `n_chars`/`part` → 直接可映射成 PRD 要的 `story_id` / `parent_story_id`，是"结构化语料管线"的现成样板。
- ⚠️ **更正方案文档的两处错误**（都会影响 L5 数据计划）：
  1. 方案称该数据集含「原文 + SFT 情节接龙问答对 + 角色扮演 + 人工校验大纲」**四件套** —— 实测 HF 仓库**只有原文**，没有 SFT/大纲/角色扮演数据；
  2. 方案标注为 **CC0 公共领域** —— 实为 **CC BY 4.0（需署名）**，且正文是繁体。
- ⚠️ **实测后果：它污染了 `scifi` 桶，并使其失去作证据的资格**（2026-09-14 实测）
  - 该作 **1,661,311 字**，占 `scifi` 桶池的 **92.6%**、训练区的 **96.4%**；
  - 而同桶**唯一**的留出作品《貓城記》是 **1932 年**早期白话作品 → `scifi` 实际测的是
    「**现代网文 → 1932 年作品**」的**跨域迁移**，且训练分布被单部作品垄断；
  - 又因为它是**单部作品**，按「按作品留出」（`py/data/build_splits.py`）的口径
    **无法既训练又留出**，**不具备独立成桶的资格**；
  - → 结论：`scifi` 一列在 `sweep_worksplit` 的全部结果中**只作跨域对照，不作容量 / 结构证据**。
  - 处置决定：**保留数据、只做标注**（不删除），修正本文的判定标准至上表口径。

### 3. 自有《零号计划》

- 授权链最干净（完全自有），PRD 明确要求「先准备自有小样本」。
- 注意：方案文档引用了它，但仓库内没有全文与授权说明；引入前应先在 `manifest` 里登记来源与授权状态。

---

## 二、第二梯队：量大，但有版权与清洗风险

| 数据集 | 体积/规模 | 标签许可 | 结构 | 风险 |
|---|---|---|---|---|
| [`wdndev/webnovel-chinese`](https://huggingface.co/datasets/wdndev/webnovel-chinese) | **39 GB**（10 shard，每个约 4.2 GB），约 9000 本，约 9B tokens | Apache-2.0（**上传者自贴**） | jsonl：`title / chapter / text` | **内容是抓取的商业网文**（卡片示例即《斗破苍穹》）；上传者的许可标签**不覆盖原著版权** → 仅限本地学习实验，不可再分发/商用 |
| [`zxbsmk/webnovel_cn`](https://huggingface.co/datasets/zxbsmk/webnovel_cn) | 仓库内 50k 子集；完整版 21.7M 条指令样本在百度网盘 | MIT（同上问题） | 512-token 指令对（输入/输出） | 同上；其**指令数据格式**值得参考 |
| [`LooksJuicy/Chinese-Roleplay-Novel`](https://huggingface.co/datasets/LooksJuicy/Chinese-Roleplay-Novel) | 约 260 条多轮对话（基于 4500 本小说，GPT-4o 构造） | Apache-2.0 | `world` + 逐轮状态（时间/角色状态/任务进度） | 合成本身较干净；对 **L6 状态机与结构化生成**有直接参考价值 |
| [`liwu/MNBVC`](https://huggingface.co/datasets/liwu/MNBVC) | 合集（项目目标 60 TB） | MIT（仓库） | 混杂 | 规模最大但需重度清洗，不适合作为第一批 |

**另需规避**：[`lainka0o0/chinese-novel-nonH-collect`](https://huggingface.co/datasets/lainka0o0/chinese-novel-nonH-collect) 标 CC0、2.9k 个 txt，但实测内容多为**外国科幻的中文译本**（特德·蒋、克拉克、弗诺·文奇等）——CC0 标签**不能覆盖原作者与译者的权利**，法律风险高，不建议使用。

> ⚠️ **更正方案文档**：方案写的 `steven-fei/webnovel-chinese` 在 HuggingFace 上**不存在**；正确条目是 `wdndev/webnovel-chinese`。

**普遍规律**：中文小说数据集里，**大多数没有许可**（只能内部实验）；少数带许可标签的，要么是译文/抓取内容（标签无效），要么体量极小。

---

## 三、公版古典小说：许可最干净，但风格差异大

- **[中文维基文库](https://zh.wikisource.org)** —— 实测其版权页确认：贡献内容以 **CC BY-SA 4.0 + GFDL** 发布，公版作品可自由使用。收录四大名著、三言二拍、聊斋志异等，是**零版权风险**的古典小说来源。
  - 注意：现代**点校/注释本**的校勘与注释部分可能仍受保护，应取公版原文。
- GitHub 上的古籍仓库（如 `hanzhaodeng/chinese-ancient-text`、`PoeticFlower/EBook_ChineseClassicalNovels`）：**许可不明**（无 / NOASSERTION），只能内部实验。
- 用途定位：文言/近代白话与目标现代文风差异大 → 适合做**风格对照实验**（"同架构换语料，风格怎么变"），不适合当主力语料。

---

## 四、与仓库现有数据层的关系（重要）

核对时发现，本仓库**已有可用的数据层**，不需要另起一套：

- `py/data/fetch_corpus.py` → `data/sources.json`：已抓取 **86 部公版作品、24,720,684 字**
  （来源：Project Gutenberg 80 部 + 中国古典文学原文 6 部；繁转简 opencc t2s；跨源按标题去重）
- 分桶现状：**古典 37 部 / 1122 万字**（红楼梦、西游记、三国演义、聊斋志异…）、
  **武侠 23 部 / 808 万字**（水浒传、三侠五义、施公案…）、
  **都市 26 部 / 542 万字**（官场现形记、二十年目睹之怪现状、孽海花…）、**科幻 0 部**
- `py/data/prepare.py` → `data/manifest.jsonl`（`story_id / parent_story_id / split / source / hash / chars / vocab_size / license`）
  + `data/split_audit.json`（分组不交叉审计，当前 3 条记录 = 项目自有语料）
- `py/tokenizer.py` / `py/train.py` / `py/sweep_data_scale.py` 均已存在

**所以"没有训练数据"不是当前的真问题。** 现有 2472 万字已经解决了「规模」，
但它的成分全是**明清白话与晚清小说**，而且**科幻桶是 0**：

> 用这批语料训出来的模型，会写"话说……"和"看官听说"，不会写现代小说。

于是真正的缺口浮出来了：

| 需求 | 现有公版语料 | 本清单第一梯队 | 本清单第二梯队 |
|---|---|---|---|
| 规模 | ✅ 2472 万字 | ⚠️ 2.6GB 译文 | ✅ 39GB |
| 许可干净 | ✅ 公版 | ⚠️ CDLA 干净 /「CC BY」系上传者自贴、不覆盖原著 | ❌ 标签无效 |
| 现代文风 | ❌ 明清/晚清 | ⚠️ 儿童化简单句 | ✅ 网文风 |
| 故事级边界 | ⚠️ 按书/回 | ✅ 每行一篇 | ✅ 按章 |

**"现代文风 + 许可干净"这一格目前是空的**——外部数据里找不到交集，
这正是方案 L3 里"用大模型合成扩充 + 过滤"和《零号计划》自有语料要补的位置。

---

## 五、现代白话公版补充（2026-09-14）

已接入 `py/data/fetch_corpus.py` 的 `xiandai` 桶：中文维基文库公版现代白话，共 **35 部 / 110 万字**，含鲁迅《狂人日记》《孔乙己》《药》《故乡》《阿Q正传》《伤逝》《补天》《铸剑》等单篇，以及老舍《骆驼祥子》《四世同堂》、郁达夫《沉沦》、萧红《呼兰河传》《生死场》。作者卒年均满足中国大陆公版线；每篇保留 `ref`、`license`、清洗后 hash，按作品切分。

知乎、天涯、猫扑：**默认不收**。它们是 UGC 平台，著作权属于发帖作者；平台 ToS 也不授权批量抓取。网上搬运包或 HuggingFace 自贴 `CC BY`/`MIT` 标签不覆盖原作者权利，和 `wdndev/webnovel-chinese`、`asd567557275/chinese_novel` 属同一风险等级。

**2026-09-14 追加：本地学习桶 `localugc`（明确授权下启用）**。项目主在**仅个人本地学习**的前提下，要求把这类数据也接进来。于是新增 `bucket='localugc'`，与四个证据桶完全隔离：

- **边界**：只作本地实验；**不可再分发 / 不可商用 / 不进评测桶**；不出现在 `sweep_worksplit.py` 的 `BUCKETS` 里；`data/sources.json` 逐条记录 `source` / `license` / `ref`，以便事后审计和按权利人要求删除。
- **实际接入**（HF 上可核验的）：
  - `liyucheng/zhihu_rlhf_3k`（`zhihu_3k_rlfh.tsv`，GBK）、`liyucheng/zhihu_26k`（`zhihu_26k.tsv`，GBK）
  - `zirui3/zhihu_qa`（`zhihu_qa.jsonl.gz`，需 gunzip）、`next-social/zhihu_beautiful_super_alpaca`（`data.json`）
  - `wdndev/webnovel-chinese` **首片前 30 MB 流式切片**（上游 10 片共 ~39 GB；Range 请求，不整库下载）
- **天涯**：HF 上没有任何可核验的数据集，但 GitHub 上有两份真实抓取，均已实测：
  - `lnlzhp126/TianYa-Zatan` —— 60 个按日 CSV，**只有帖子标题 + URL + 回复数，没有正文**，对训练无用，**未接入**；
  - `zhaoqixu/tianya-forum-scrapper` 的 `LXZ_chi(B).xml`（莲蓬鬼话板）—— **接入为 `tianya_*`**：
    18,550 帖 / 326 万字的 XML 多轮对话（`<s>` 是帖、`<utt>` 是发言），取 ≥200 字的前 3,000 帖。
    ⚠️ **它是论坛辩论 / 聊天体，不是小说叙事**；发言前缀 `@用户名：时间戳`（时间戳常不带空格）已在清洗时剥除。
- **猫扑**：GitHub / HF 双向检索**都没有可核验的叙事数据集**，故未接入。
- **通用本地导入位**：任何自有 dump 存成 `data/raw/_cache/localugc/<标题>.txt`（UTF-8，文件名即作品名），
  重跑 `fetch_corpus.py` 即自动作为一部作品入库，走同一套清洗与按作品切分。

## 六、建议路线（对齐 PRD P2）

1. **先接入现有管线**，不要另写抓取脚本：新来源登记进 `data/sources.json`，
   语料规格走 `py/data/prepare.py` → `data/manifest.jsonl` + `split_audit.json`（分组不交叉），
   tokenizer 只用训练切分拟合。
2. **现代文风的第一批**，按许可干净程度排序：
   1. `adam89/TinyStoriesChinese`（CDLA，简单现代汉语，故事级切分天然成立）
      —— 用来跑通管线、验证"故事级 split"与 tokenizer，别指望它产出好小说；
   2. ~~`asd567557275/chinese_novel`~~ —— **已从"第一批"降级**：上传者非权利人、标签不覆盖原著，
      按 §1.2 / §2 的判定只能**本地学习**，且已确认会污染 `scifi` 桶（单部作品占 96.4% 训练区），**不入评测桶**；
   3. **自有 / 合成语料**（《零号计划》 + 大模型合成后过滤）—— 唯一能同时满足"现代文风 + 许可干净"的路径，
      PRD P2 的"数百个短故事探索集"建议就走这条。
3. **网文切片放最后**：`wdndev/webnovel-chinese` 只取小切片做本地实验，
   manifest 里必须标注「抓取内容，不可再分发/商用」；不要整库下载 39 GB。
4. **科幻桶补数据**：现有桶里科幻为 0，而项目原语料（记忆之海）与《零号计划》都是科幻——
   公版领域几乎没有中文科幻可抓，这一格只能靠自有/合成补齐。

**下一步可做的具体工作**：把上表第一梯队两个来源登记进 `data/sources.json` 并跑 `py/data/prepare.py`，
产出带 `story_id` 的 manifest 与切分审计；随后用 `py/tokenizer.py` 在训练切分上拟合一个 BPE，
对比"字符级 vs 子词"在同一批故事上的表现。

> ✅ **已完成（2026-09-13）**：两个来源均已接入 `py/data/fetch_corpus.py`（新增 `hf` 源与 `story` 桶），
> 全量重跑后 1091 部作品 / 2682 万字，`prepare.py --split` 产出 871/107/113 的故事级切分（0 冲突）。
> BPE 对照结果见 `py/runs/bpe_probe_*.json` 与 LOG.md 续六。

