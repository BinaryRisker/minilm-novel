/* ============================================================================
 * test_novel.js — 中文小说生成能力测试（产出报告所需的全部证据）
 * ----------------------------------------------------------------------------
 * 用法：node test_novel.js [--corpus scifi] [--steps 400] [--quick]
 *
 * 对每段语料做四件事：
 *   1) 用与网页工具完全相同的默认配置训练，记录 [100/200/300/400] 步的损失
 *      与生成快照 —— 看它怎么从乱码长成句子
 *   2) 训练后用 5 个"语料内提示词" × 3 个温度生成，看风格保真度
 *   3) 用 2 个"语料外提示词"生成 —— 测它到底是"会写"还是"会背"
 *   4) 留出集下一字准确率：语料后 10% 不参与正文训练（词表由全文建立，属同篇尾段续写），
 *      统计 top-1 / top-5 命中率 —— 这是唯一不看人眼的量化指标
 *
 * 输出：report_evidence.md（机器生成的原始证据，供人工报告引用）
 * ============================================================================ */

var fs = require('fs');
var path = require('path');

var kit = require('./minilm.js');
var CORPUS = require('./corpus.js');

// 评估协议、统计基线、生成质量指标 —— 三处共用同一份定义，避免模型和基线各算一套
var PROTO = require('./eval/protocol.js');
var BASE = require('./eval/baselines.js');
var M = require('./eval/metrics.js');

var argv = process.argv.slice(2);
function argVal(name, dft) {
  var i = argv.indexOf('--' + name);
  return i >= 0 && argv[i + 1] && argv[i + 1][0] !== '-' ? argv[i + 1] : dft;
}
var QUICK = argv.indexOf('--quick') >= 0;
var ONLY = argVal('corpus', null);
var STEPS = parseInt(argVal('steps', QUICK ? 150 : 400), 10);
var SEED = argVal('seed', null) != null ? parseInt(argVal('seed', null), 10) : null;
var OUT = argVal('out', null);

// 与网页工具 index.html 的默认配置严格一致
var CFG = { ctx: 32, d: 40, layers: 2, heads: 4, ffMul: 4 };
var BATCH = QUICK ? 8 : 8;
var LR = 0.004;

var IN_PROMPTS = {
  scifi: ['新长安的雨下了', '恒站在窗前，', '苏念说，我', '他低下头，看着', '她的手指很'],
  wuxia: ['沈砚站在', '女子说，你', '雨落在他的', '剑雨楼的灯', '沈砚说，我'],
  urban: ['陈老板说，', '巷子口有一家', '她低头吃面', '他说，我今天', '门上的铃响了']
};
// 语料里一定不存在的提示词，用来测泛化
var OOD_PROMPTS = {
  scifi: ['今天的天气真', '他打开冰箱，', '警察敲了敲门，'],
  wuxia: ['汽车停在楼', '她打开电脑，', '今天是星期五，'],
  urban: ['外星飞船降落在', '他举起长剑，', '考卷发下来了，']
};

var out = [];
function w(s) { out.push(s); console.log(s); }

/** 带符号、固定一位小数 */
function sgn(x) { return (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(1); }

/** 百分点差值的判读。±2pt 以内视为打平 —— 因为不同随机种子本身就会抖动几个点 */
function judgePt(d) {
  if (d > 2) return '明显超过，模型确实用上了比「相邻字频次」更多的信息';
  if (d > -2) return '**基本打平** —— 模型没明显超出这个纯统计基线';
  return '**低于** —— 模型不如这个纯统计基线';
}

/* ------------------------------------------------------------------ */

function trainCorpus(key) {
  var corpus = CORPUS[key];
  var text = corpus.text;
  var vocab = kit.buildVocab(text);
  var ids = kit.encode(text, vocab.stoi);

  // 90% 训练 / 10% 留出
  var split = Math.floor(ids.length * 0.9);
  var trainIds = ids.slice(0, split);
  var testIds = ids.slice(split);

  var model = new kit.MiniLM(vocab.size, CFG, SEED == null ? {} : { seed: SEED });
  var warmup = Math.max(10, Math.floor(STEPS * 0.15));

  w('\n## ' + corpus.name);
  w('');
  w('> ' + corpus.desc);
  w('');
  w('| 项目 | 值 |');
  w('|---|---|');
  w('| 语料字数 | ' + text.length.toLocaleString() + ' |');
  w('| 训练集 / 留出集 | ' + trainIds.length.toLocaleString() + ' / ' + testIds.length.toLocaleString() + ' 字 |');
  w('| 唯一字符 | ' + (vocab.size - 1) + ' |');
  w('| 参数量 | ' + model.paramCount().toLocaleString() + ' |');
  w('| 配置 | ctx=' + CFG.ctx + ' d=' + CFG.d + ' layers=' + CFG.layers +
    ' heads=' + CFG.heads + ' batch=' + BATCH + ' lr=' + LR + ' steps=' + STEPS + ' |');
  w('| 随机种子 | ' + model.seed + (SEED == null ? '（自动取值，本次运行不可复现）' : '（显式指定，可复现）') + ' |');
  w('');

  var t0 = Date.now();
  var snapEvery = Math.max(1, Math.floor(STEPS / 4));
  var snapPrompt = IN_PROMPTS[key][0].slice(0, 6);
  var snapIds = kit.encode(snapPrompt, vocab.stoi);
  var rows = [];
  var lossAcc = 0, lossN = 0;

  for (var step = 0; step < STEPS; step++) {
    var batch = kit.makeBatch(trainIds, BATCH, CFG.ctx, model.rng.batch);
    model.zeroGrad();
    var r = model.forward(batch.tokens, batch.targets);
    model.backward();
    model.clipGrad(1.0);
    var curLr = step < warmup ? LR * (step + 1) / warmup
      : LR * 0.5 * (1 + Math.cos(Math.PI * (step - warmup) / Math.max(1, STEPS - warmup)));
    model.step(curLr, step + 1);
    lossAcc += r.loss; lossN++;

    if ((step + 1) % snapEvery === 0 || step === 0) {
      var avg = lossAcc / lossN;
      var g = model.generate(snapIds, { maxNew: 46, temp: 0.8, topk: 8, repPenalty: 1.15 });
      rows.push({ step: step + 1, loss: avg, text: kit.decode(g.ids, vocab.itos).replace(/\n/g, ' ') });
      lossAcc = 0; lossN = 0;
      process.stdout.write('    step ' + (step + 1) + '/' + STEPS + '  loss=' + avg.toFixed(4) + '\r');
    }
  }
  var elapsed = (Date.now() - t0) / 1000;
  console.log('    训练完成 ' + elapsed.toFixed(1) + 's（' + (STEPS / elapsed).toFixed(1) + ' step/s）                ');

  w('**训练过程**：' + elapsed.toFixed(1) + ' 秒，' + (STEPS / elapsed).toFixed(1) + ' step/s（Node 单线程）');
  w('');
  w('### 损失与生成快照（同一提示词「' + snapPrompt + '」）');
  w('');
  w('| 步数 | 交叉熵损失 | 模型输出 |');
  w('|---|---|---|');
  rows.forEach(function (r) {
    w('| ' + r.step + ' | ' + r.loss.toFixed(3) + ' | ' + r.text + ' |');
  });
  w('');

  /* ---- 留出集评估（模型与统计基线逐位对齐）---- */
  // 位置切分、可见上文长度全部走 eval/protocol.js，和基线共用，不可能对不上。
  // 模型侧按窗口 forward（一次算 32 个位置），再逐位算 log-softmax 交给累计器。
  var acc = new PROTO.Acc();
  var lpBuf = new Float64Array(vocab.size);
  PROTO.forEachWindow(testIds, CFG.ctx, function (chunk, tgt) {
    model.forward(chunk, tgt, CFG.ctx);
    var logits = model.cache.logits;
    for (var i = 0; i < CFG.ctx; i++) {
      var off = i * vocab.size, v;
      var mx = -Infinity;
      for (v = 0; v < vocab.size; v++) if (logits[off + v] > mx) mx = logits[off + v];
      var sum = 0;
      for (v = 0; v < vocab.size; v++) { var e = Math.exp(logits[off + v] - mx); lpBuf[v] = e; sum += e; }
      var lsum = Math.log(sum);
      for (v = 0; v < vocab.size; v++) lpBuf[v] = logits[off + v] - mx - lsum;   // log-softmax
      acc.add(lpBuf, tgt[i]);
    }
  });
  var ms = acc.summary();
  var modelRow = { top1: ms.top1, top5: ms.top5, nll: ms.nll, ppl: ms.ppl };

  // 统计基线：只在 trainIds 上拟合；评估位置 / 可见上文 / 词表与模型逐位一致
  var baseRes = BASE.runBaselines(trainIds, testIds, { ctx: CFG.ctx, V: vocab.size });

  var rand1 = 100 / vocab.size;
  w('### 留出集下一字预测准确率');
  w('');
  w('> 语料后 10%（' + testIds.length + ' 字）**从未参加正文训练**。注意两点局限：'
    + '词表由全文建立（含留出文本的字符集合），且评测是同篇尾段续写——'
    + '因此这是「局部字序泛化」，不等于跨故事泛化。');
  w('>');
  w('> **这张表是本报告最重要的一张。** 「比随机猜测高 124 倍」看着唬人，但随机猜测');
  w('> 只是 1/V 的数学下限，不是及格线。真正的及格线是 bigram / trigram —— 两个');
  w('> **完全不懂语法、只会数相邻字出现多少次**的统计模型。模型赢不了它们，5.5 万');
  w('> 参数和 400 步训练就没有换来结构知识，只是把「数频次」这件事干得更贵。');
  w('');
  w(BASE.mdTable(baseRes, modelRow));
  w('');
  w('共评估 ' + ms.n + ' 个位置。NLL 用自然对数，与训练 loss 同量纲；困惑度 = e^NLL。');
  w('随机猜测 Top-1 = ' + rand1.toFixed(2) + '%（1/V，V=' + vocab.size + '）。');
  w('');

  var bBig = baseRes[1], bTri = baseRes[2];
  var dBig = ms.top1 - bBig.top1, dTri = ms.top1 - bTri.top1, dNll = ms.nll - bTri.nll;
  w('**判定**：');
  w('');
  w('| 对比 | 差值 | 读法 |');
  w('|---|---|---|');
  w('| 模型 − unigram | ' + sgn(ms.top1 - baseRes[0].top1) + ' pt | 远超字频基线，说明模型确实用上了上下文 |');
  w('| 模型 − bigram | ' + sgn(dBig) + ' pt | ' + judgePt(dBig) + ' |');
  w('| 模型 − trigram | ' + sgn(dTri) + ' pt | ' + judgePt(dTri) + ' |');
  w('| 模型 NLL − trigram NLL | ' + sgn(dNll) + ' | ' +
    (dNll < -0.05 ? '概率估计更准（越低越好）'
      : dNll < 0.05 ? '概率估计基本相当'
        : '**概率估计更差**（越低越好）') + ' |');
  w('');
  w(BASE.mdBackoff(baseRes));
  w('');

  /* ---- 语料内提示词生成 ---- */
  w('### 语料内提示词生成（top-k=8，重复惩罚 1.15，长度 80 字）');
  w('');
  [0.5, 0.8, 1.2].forEach(function (temp) {
    w('**温度 ' + temp.toFixed(1) + '**');
    w('');
    IN_PROMPTS[key].forEach(function (p) {
      var pid = kit.encode(p, vocab.stoi);
      var g = model.generate(pid, { maxNew: 80, temp: temp, topk: 8, repPenalty: 1.15 });
      w('- `' + p + '` → ' + kit.decode(g.ids, vocab.itos).replace(/\n/g, ' '));
    });
    w('');
  });

  /* ---- 语料外提示词（泛化测试）---- */
  w('### 语料外提示词生成（这些短语在语料里根本不存在）');
  w('');
  w('> 这是区分"会写"和"会背"的关键测试。如果模型只是记住了语料，' +
    '遇到没见过的开头就会崩成随机字或强行拐回训练集里的句子。');
  w('');
  OOD_PROMPTS[key].forEach(function (p) {
    // 统计有多少字不在词表内——字符级模型会把它们静默丢弃，必须显式披露
    var oi2 = M.oovInfo(p, vocab.stoi);
    var pid = kit.encode(p, vocab.stoi);
    var g = model.generate(pid, { maxNew: 60, temp: 0.8, topk: 8, repPenalty: 1.15 });
    var full = kit.decode(g.ids, vocab.itos).replace(/\n/g, ' ');
    w('- `' + p + '` → ' + full);
    if (oi2.oov > 0) {
      w('  - **注意**：这 ' + oi2.total + ' 个字里有 **' + oi2.oov + ' 个不在词表内**（' +
        oi2.oovUnique.map(function (c) { return '`' + c + '`'; }).join(' ') + '），' +
        '模型实际只看到了 `' + oi2.kept + '`。所以下面这段输出**不能**当作完整语义的泛化证据。');
    }
    var trace = g.trace[0].candidates.slice(0, 3).map(function (c) {
      return "'" + (vocab.itos[c.id] || '?') + "'" + (c.p * 100).toFixed(0) + '%';
    }).join(' ');
    w('  - 第一个字的候选：' + trace);
  });
  w('');

  /* ---- 生成文本质量指标 ---- */
  // Top-1 只衡量"猜下一个字准不准"，衡量不了"生成的东西能不能读"。分开测：
  //   重复率  → 自己跟自己重复到什么程度（复读机检测）
  //   重合度  → 有多少是从训练集原样搬来的（抄袭检测）
  var trainText = kit.decode(trainIds, vocab.itos);
  var gRows = [];
  IN_PROMPTS[key].forEach(function (p) {
    var gg = model.generate(kit.encode(p, vocab.stoi), { maxNew: 80, temp: 0.8, topk: 8, repPenalty: 1.15 });
    var f = kit.decode(gg.ids, vocab.itos).replace(/\n/g, ' ');
    var body = f.indexOf(p) === 0 ? f.slice(p.length) : f;
    if (!body.length) return;
    gRows.push({
      p: p, body: body,
      d1: M.distinctN(body, 1), d2: M.distinctN(body, 2),
      rep4: M.repeatRate(body, 4),
      ov8: M.overlapRate(body, trainText, 8),
      lcs: M.longestCommonSubstring(body, trainText)
    });
  });
  function mean(a) { return a.reduce(function (x, y) { return x + y; }, 0) / (a.length || 1); }
  var mD2 = mean(gRows.map(function (r) { return r.d2; }));
  var mRep = mean(gRows.map(function (r) { return r.rep4; }));
  var mOv = mean(gRows.map(function (r) { return r.ov8; }));
  var maxLcs = gRows.reduce(function (m, r) { return r.lcs.len > m.len ? r.lcs : m; }, { len: 0, str: '' });

  w('### 生成文本质量指标（温度 0.8，语料内 5 个提示词）');
  w('');
  w('> 这两项和「猜得准不准」是两回事，必须分开看：');
  w('>');
  w('> - **4-gram 重复率** ↓：生成文本自己跟自己重复的比例，越高越复读机');
  w('> - **distinct-2** ↑：不重复二元组占比，越低说明用词越单调');
  w('> - **8-gram 训练集重合率**：生成内容里能在训练集原样找到的比例，越高越像在背');
  w('> - **最长公共子串**：与训练集一字不差的最长片段 —— 这是「回放」最硬的证据');
  w('');
  w('| 提示词 | 生成正文（节选） | distinct-2 ↑ | 4-gram 重复率 ↓ | 8-gram 重合率 | 最长公共子串 |');
  w('|---|---|---|---|---|---|');
  gRows.forEach(function (r) {
    w('| `' + r.p + '` | ' + r.body.slice(0, 24) + (r.body.length > 24 ? '…' : '') + ' | ' +
      r.d2.toFixed(2) + ' | ' + (r.rep4 * 100).toFixed(1) + '% | ' + (r.ov8 * 100).toFixed(1) + '% | ' +
      r.lcs.len + ' 字 |');
  });
  w('| **均值** | — | **' + mD2.toFixed(2) + '** | **' + (mRep * 100).toFixed(1) + '%** | **' +
    (mOv * 100).toFixed(1) + '%** | 最长 **' + maxLcs.len + ' 字** |');
  w('');
  if (maxLcs.len >= 8) {
    w('**最长公共子串原文**：`' + maxLcs.str + '` —— ' + maxLcs.len +
      ' 个字与训练集一字不差。这个长度直接说明它是在回放还是在改写。');
    w('');
  }
  w('读法：**高重合 + 低重复** = 正常借用，在用学到的句式重组；' +
    '**高重合 + 高重复** = 复读训练集片段；**低重合 + 高重复** = 崩成乱码绕圈。');
  w('');

  /* ---- 候选分布 ---- */
  var probe = IN_PROMPTS[key][2] || IN_PROMPTS[key][0];
  var g2 = model.generate(kit.encode(probe, vocab.stoi), { maxNew: 6, temp: 1.0, topk: 6, repPenalty: 1.0 });
  w('### 下一步候选分布（提示词「' + probe + '」）');
  w('');
  g2.trace.forEach(function (tr) {
    w('| 第 ' + (tr.step + 1) + ' 步 | ' + tr.candidates.map(function (c) {
      return '`' + (vocab.itos[c.id] === '\n' ? '⏎' : vocab.itos[c.id]) + '` ' + (c.p * 100).toFixed(1) + '%';
    }).join(' · ') + ' |');
  });
  w('');

  return {
    top1: ms.top1, top5: ms.top5, nll: ms.nll, ppl: ms.ppl,
    loss: rows[rows.length - 1].loss, rows: rows,
    baselines: baseRes,
    gen: { d2: mD2, rep4: mRep, ov8: mOv, lcs: maxLcs.len }
  };
}

/* ------------------------------------------------------------------ */

var HEAD = [
  '# 中文小说生成测试 · 原始证据',
  '',
  '> 本文件由 `test_novel.js` 自动生成，是人工分析报告的数据来源。',
  '> 生成时间：' + new Date().toISOString().replace('T', ' ').slice(0, 19) + ' (UTC)',
  '> 运行模式：' + (QUICK ? '快速' : '标准') + ' · 每段语料 ' + STEPS + ' 步',
  '> 随机种子：' + (SEED == null ? '未指定 —— 每次运行自动取新值，**不可复现**；各段表格记录了实际值' : SEED),
  '',
  '> **读这份证据前请注意**：本文件是**单次运行**的结果。留出集 Top-1 在不同随机种子下',
  '> 会波动几个百分点，单次数字不能用来比较实现差异或宣称能力。',
  '> **量化结论请以 `py/align_report.md` 为准**（多种子、分布对分布、带 σ）。',
  '> 本文件的价值在于定性证据：训练快照、温度对比、语料外提示词、候选分布。',
  '',
  '配置与网页工具 `index.html` 的默认档完全一致：' +
  '`ctx=32 d=40 layers=2 heads=4 ffMul=4 batch=' + BATCH + ' lr=' + LR + '`，' +
  '优化器为 Adam + 梯度裁剪(1.0) + 线性 warmup 与余弦衰减，采样为 top-k + 温度 + 重复惩罚。',
  '',
  '---'
];

var keys = ONLY ? [ONLY] : ['scifi', 'wuxia', 'urban'];
w(HEAD.join('\n'));

var summary = [];
keys.forEach(function (k) {
  var r = trainCorpus(k);
  summary.push({ key: k, name: CORPUS[k].name, r: r });
  w('\n---');
});

// 汇总表 —— 基线列是重点：模型 vs bigram / trigram，而不是 vs 随机猜测
var sumLines = ['', '## 汇总', '',
  '| 语料 | 最终损失 | 模型 Top-1 | bigram | trigram | 模型−trigram | 模型 NLL | trigram NLL |',
  '|---|---|---|---|---|---|---|---|'];
summary.forEach(function (s) {
  var b = s.r.baselines || [];
  var bi = b[1], tr = b[2];
  sumLines.push('| ' + s.name + ' | ' + s.r.loss.toFixed(3) + ' | **' + s.r.top1.toFixed(1) + '%** | ' +
    (bi ? bi.top1.toFixed(1) + '%' : '—') + ' | ' + (tr ? tr.top1.toFixed(1) + '%' : '—') + ' | ' +
    (tr ? sgn(s.r.top1 - tr.top1) + ' pt' : '—') + ' | ' + s.r.nll.toFixed(3) + ' | ' +
    (tr ? tr.nll.toFixed(3) : '—') + ' |');
});
sumLines.push('');
sumLines.push('| 语料 | distinct-2 ↑ | 4-gram 重复率 ↓ | 8-gram 训练集重合率 | 最长公共子串 |');
sumLines.push('|---|---|---|---|---|');
summary.forEach(function (s) {
  var g = s.r.gen;
  sumLines.push('| ' + s.name + ' | ' + g.d2.toFixed(2) + ' | ' + (g.rep4 * 100).toFixed(1) + '% | ' +
    (g.ov8 * 100).toFixed(1) + '% | ' + g.lcs + ' 字 |');
});
sumLines.push('');
sumLines.push('> **怎么读**：「模型−trigram」为负，说明模型**输给了一个只看前两个字的');
sumLines.push('> 统计模型**。这不是实现 bug，是这个训练预算下的真实水平 —— 语料只有两三千字、');
sumLines.push('> 400 步，Transformer 的长处（长程依赖）根本没机会发挥。');
sumLines.push('');
w(sumLines.join('\n'));

var outFile = OUT
  ? (/\.md$/i.test(OUT) ? path.resolve(OUT) : path.join(path.resolve(OUT), 'report_evidence.md'))
  : path.join(__dirname, 'report_evidence.md');
fs.mkdirSync(path.dirname(outFile), { recursive: true });
fs.writeFileSync(outFile, out.join('\n'), 'utf8');
console.log('\n✓ 已写入 ' + outFile);
