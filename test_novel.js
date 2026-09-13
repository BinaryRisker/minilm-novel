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
 *   4) 留出集下一字准确率：把语料后 10% 作为从未参加训练的测试集，
 *      统计 top-1 / top-5 命中率 —— 这是唯一不看人眼的量化指标
 *
 * 输出：report_evidence.md（机器生成的原始证据，供人工报告引用）
 * ============================================================================ */

var fs = require('fs');
var path = require('path');

var kit = require('./minilm.js');
var CORPUS = require('./corpus.js');

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

  /* ---- 留出集准确率 ---- */
  var top1 = 0, top5 = 0, total = 0;
  for (var s = 0; s + CFG.ctx + 1 <= testIds.length; s += CFG.ctx) {
    var chunk = testIds.slice(s, s + CFG.ctx);
    var tgt = testIds.slice(s + 1, s + 1 + CFG.ctx);
    model.forward(chunk, tgt, CFG.ctx);
    var logits = model.cache.logits;
    for (var i = 0; i < CFG.ctx; i++) {
      var off = i * vocab.size, best = -Infinity, bi = 0;
      var arr = [];
      for (var v = 0; v < vocab.size; v++) {
        var lg = logits[off + v];
        arr.push({ v: v, l: lg });
        if (lg > best) { best = lg; bi = v; }
      }
      total++;
      if (bi === tgt[i]) top1++;
      arr.sort(function (a, b) { return b.l - a.l; });
      for (var k = 0; k < 5; k++) if (arr[k].v === tgt[i]) { top5++; break; }
    }
  }

  // 随机基线
  var rand1 = 100 / vocab.size;
  w('### 留出集下一字预测准确率');
  w('');
  w('> 语料后 10%（' + testIds.length + ' 字）**从未参加训练**，用它对模型做纯盲测。');
  w('');
  w('| 指标 | 模型 | 随机猜测基线 | 提升倍数 |');
  w('|---|---|---|---|');
  w('| Top-1 准确率 | **' + (100 * top1 / total).toFixed(1) + '%** | ' + rand1.toFixed(2) + '% | ' +
    (top1 / total * vocab.size).toFixed(1) + '× |');
  w('| Top-5 准确率 | **' + (100 * top5 / total).toFixed(1) + '%** | ' + (5 * rand1).toFixed(2) + '% | ' +
    (top5 / total / (5 / vocab.size)).toFixed(1) + '× |');
  w('');
  w('（共评估 ' + total + ' 个位置）');
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
    var oov = 0;
    for (var oi = 0; oi < p.length; oi++) if (!(p[oi] in vocab.stoi)) oov++;
    var pid = kit.encode(p, vocab.stoi);
    var g = model.generate(pid, { maxNew: 60, temp: 0.8, topk: 8, repPenalty: 1.15 });
    var full = kit.decode(g.ids, vocab.itos).replace(/\n/g, ' ');
    w('- `' + p + '` → ' + full);
    if (oov > 0) {
      w('  - **注意**：这 ' + p.length + ' 个字里有 **' + oov + ' 个不在词表内**，' +
        '模型实际只看到了 `' + p.split('').filter(function (ch) { return ch in vocab.stoi; }).join('') +
        '`。所以下面这段输出**不能**当作完整语义的泛化证据。');
    }
    var trace = g.trace[0].candidates.slice(0, 3).map(function (c) {
      return "'" + (vocab.itos[c.id] || '?') + "'" + (c.p * 100).toFixed(0) + '%';
    }).join(' ');
    w('  - 第一个字的候选：' + trace);
  });
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

  return { top1: 100 * top1 / total, top5: 100 * top5 / total, loss: rows[rows.length - 1].loss, rows: rows };
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

// 汇总表
var sumLines = ['', '## 汇总', '',
  '| 语料 | 最终损失 | 留出集 Top-1 | 留出集 Top-5 | 随机基线 Top-1 |', '|---|---|---|---|---|'];
summary.forEach(function (s) {
  sumLines.push('| ' + s.name + ' | ' + s.r.loss.toFixed(3) + ' | ' +
    s.r.top1.toFixed(1) + '% | ' + s.r.top5.toFixed(1) + '% | ' +
    (100 / (CORPUS[s.key].text ? new Set(CORPUS[s.key].text.split('')).size + 1 : 1)).toFixed(2) + '% |');
});
sumLines.push('');
w(sumLines.join('\n'));

var outFile = OUT
  ? (/\.md$/i.test(OUT) ? path.resolve(OUT) : path.join(path.resolve(OUT), 'report_evidence.md'))
  : path.join(__dirname, 'report_evidence.md');
fs.mkdirSync(path.dirname(outFile), { recursive: true });
fs.writeFileSync(outFile, out.join('\n'), 'utf8');
console.log('\n✓ 已写入 ' + outFile);
