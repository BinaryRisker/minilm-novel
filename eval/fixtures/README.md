# eval/fixtures — 手算可验证的最小样本

这些文件的存在只为一件事：**让指标自己被验证**。

一个算错的指标比没有指标更危险 —— 它会安静地给出漂亮的错数字。
所以这里放的全是**能用手算出答案**的极小样本，期望值写在下面，
`test_eval.js` 逐条断言。改了指标实现而期望值没变，测试就应该红。

所有 fixture 必须**不含换行、不含 BOM、纯 ASCII**（避免 Windows 编码问题）。
`test_eval.js` 会断言字节长度，长度不对就是 fixture 被污染了。

---

## 1. `tiny_train.txt` = `abababab`（8 字节）—— n-gram 概率

词表（由这 8 个字建）：`itos = ['\0', 'a', 'b']`，`V = 3`，`a=1, b=2, \0=0`
编码：`[1,2,1,2,1,2,1,2]`

**unigram**（α=1，N=8）
- count(a)=4, count(b)=4, count(\0)=0
- P(a) = (4+1)/(8 + 1×3) = **5/11**
- P(\0) = (0+1)/11 = **1/11**

**bigram**（α=1）—— 相邻对共 7 个：ab ba ab ba ab ba ab
- ctx `a`：total=4，count(a→b)=4 → P(b|a) = (4+1)/(4+3) = **5/7**；P(a|a)=P(\0|a)=**1/7**
- ctx `b`：total=3，count(b→a)=3 → P(a|b) = (3+1)/(3+3) = **2/3**；P(b|b)=P(\0|b)=**1/6**

**trigram**（α=1）—— 三元组共 6 个：aba bab aba bab aba bab
- ctx `ab`：total=3，→a×3 → P(a|ab) = (3+1)/(3+3) = **2/3**；其余各 **1/6**
- ctx `ba`：total=3，→b×3 → P(b|ba) = **2/3**；其余各 **1/6**

**回退**
- hist=`[b]`（只有 1 个上文）要 trigram → 上文不够 2 个 → 退 bigram → P(b|b)=**1/6**
- hist=`[a,a]`（上下文 `aa` 训练里没出现）→ ctx total 不存在 → 退 bigram → P(b|a)=**5/7**

> 注：加 α 平滑后 **argmax 与 α 无关**（分母对同一上下文是常数，分子统一 +α）。
> 所以 Top-1 / Top-5 不受 α 影响，只有 NLL 受影响。这意味着挑 α 不会"刷高" Top-1。

---

## 2. `repeat.txt` = `abcabcabc`（9 字节）—— 重复率

- 4-gram 共 6 个：abca, bcab, cabc, abca, bcab, cabc
- 去重后 3 种 → **distinct-4 = 3/6 = 0.5**
- 后 3 个都是前面出现过的 → **4-gram 重复率 = 3/6 = 0.5**
- distinct-1 = 3 种字符 / 9 = **1/3**
- distinct-2：8 个二元组，3 种 → **3/8 = 0.375**

---

## 3. `overlap_gen.txt` = `abcabdabc` 对照训练集 `repeat.txt` = `abcabcabc` —— 重合度

生成文本的 3-gram 共 7 个：
`abc, bca, cab, abd, bda, dab, abc`

训练集的 3-gram 集合：`{abc, bca, cab}`

- 命中 4 个（abc, bca, cab, abc）→ **3-gram 重合率 = 4/7**
- 8-gram 重合率 = **0**（两个 8-gram 在训练集里都找不到）
- **最长公共子串 = `abcab`，长度 5**（长度 6 时两边就不一样了：abcabc vs abcabd）

---

## 4. `oov_vocab.txt` = `abcabc` + `oov_probe.txt` = `abXzc` —— 未登录字

- 词表：`itos = ['\0','a','b','c']`，`V = 4`
- probe `abXzc` 共 5 字，其中 `X`、`z` 不在词表 → **oov = 2**
- 模型实际看到的是 **`abc`**
- 编码结果：`[1, 2, 0, 0, 3]`（未登录字统一映射到 0，然后被静默丢弃）

---

## 5. 协议本身（`protocol.js`）

`testIds` 长度 8、`ctx = 2`：
- 窗口起点 s 需满足 `s + 2 + 1 <= 8` → s = 0, 2, 4 共 **3 个窗口**
- 每窗口 2 个位置 → **6 个位置**
- 位置 (s=0, i=0)：上文 `[test[0]]`，目标 `test[1]`
- 位置 (s=0, i=1)：上文 `[test[0], test[1]]`，目标 `test[2]`

`Acc` 喂 `[ln0.1, ln0.7, ln0.2]`、目标 = 1：
- Top-1 命中（argmax=1）→ 100%
- Top-5 命中（总共才 3 项）→ 100%
- NLL = −ln(0.7) = **0.356675**；困惑度 = e^NLL = 1/0.7 = **1.428571**
