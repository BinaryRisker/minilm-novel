/* ============================================================================
 * test.js — 在 Node 中验证模型正确性并实测中文小说生成
 * ----------------------------------------------------------------------------
 * 用法：node test.js [--quick]
 *   1) 梯度检查：用有限差分对比解析梯度，验证手写反向传播是否正确
 *   2) 训练：在中文小说语料上真实训练
 *   3) 生成：多种提示词 + 多温度，观察中文小说生成质量
 * ============================================================================ */

var GRAD_ONLY = process.argv.indexOf('--grad') >= 0;
var QUICK = process.argv.indexOf('--quick') >= 0;

// minilm.js 在模块加载时就把 REAL 定型为 Float32/Float64，
// 所以这里加载两份模块实例：kit64 专供梯度检查，kit 供训练（float32，更快）。
// 梯度检查必须用 float64 —— float32 的量化噪声（~1e-6）会淹没小梯度分量，
// 导致"看起来掉"的伪失败（曾经在 --quick 下踩过这个坑）。
globalThis.MINILM_USE_F64 = true;
var kit64 = require('./minilm.js');
globalThis.MINILM_USE_F64 = false;
delete require.cache[require.resolve('./minilm.js')];
var kit = require('./minilm.js');
var CORPUS = require('./corpus.js');

var F64_ONLY = GRAD_ONLY;

/* ------------------------------------------------------------------ */
/* 第一部分：梯度检查                                                   */
/* ------------------------------------------------------------------ */

function gradCheck(K) {
  console.log('='.repeat(64));
  console.log('第一部分 · 梯度检查（有限差分 vs 解析梯度，float64）');
  console.log('='.repeat(64));

  var text = '从前有座山，山里有座庙，庙里有个老和尚，老和尚对小和尚说。';
  var vocab = K.buildVocab(text);
  var cfg = { ctx: 8, d: 8, layers: 1, heads: 2, ffMul: 2 };
  var m = new K.MiniLM(vocab.size, cfg);
  var ids = K.encode(text, vocab.stoi);

  var T = 8, B = 2;
  var tokens = new Int32Array(B * T), targets = new Int32Array(B * T);
  for (var i = 0; i < B * T; i++) {
    tokens[i] = ids[i % ids.length];
    targets[i] = ids[(i + 1) % ids.length];
  }

  m.zeroGrad();
  var r = m.forward(tokens, targets);
  m.backward();

  console.log('语料长度 ' + ids.length + ' | 词表 ' + vocab.size +
              ' | 参数 ' + m.paramCount() + ' | loss ' + r.loss.toFixed(6));

  var paramsToCheck = ['emb', 'pos', 'Wq0', 'Wo0', 'Wv0', 'W1_0', 'W2_0',
                       'ln1g0', 'ln2b0', 'bq0', 'lnfg', 'b2_0'];
  var eps = 1e-5;
  var maxRel = 0, worst = '', totalChecked = 0, totalPassed = 0;
  var sigChecked = 0, sigPassed = 0;

  paramsToCheck.forEach(function (name) {
    if (!m.P[name]) return;
    var P = m.P[name], G = m.G[name], n = P.length;
    for (var idx = 0; idx < n; idx++) {
      var num, ana;
      var orig = P[idx];
      P[idx] = orig + eps;
      var lp = m.forward(tokens, targets).loss;
      P[idx] = orig - eps;
      var lm = m.forward(tokens, targets).loss;
      P[idx] = orig;
      num = (lp - lm) / (2 * eps);
      ana = G[idx];

      // 判据：绝对容差 + 相对容差（CS231n 的 gradient check 标准做法）。
      // 纯相对误差在梯度接近 0 的地方会失真 —— |g|~1e-7 时任何微小噪声
      // 都能算出百分之几的"相对误差"，那是噪声不是 bug。
      var diff = Math.abs(num - ana);
      var scale = Math.max(Math.abs(num), Math.abs(ana));
      if (scale < 1e-7) continue;                       // 两边都是 0，无意义
      totalChecked++;
      var passedThis = diff <= 1e-5 + 1e-4 * scale;
      if (passedThis) totalPassed++;

      // 只在"量级显著"的梯度上统计相对误差，这样报出来的数字才有意义
      if (scale > 1e-4) {
        sigChecked++;
        if (passedThis) sigPassed++;
        var rel = diff / scale;
        if (rel > maxRel) { maxRel = rel; worst = name + '[' + idx + ']'; }
      }
    }
  });

  console.log('\n检查了 ' + totalChecked + ' 个参数分量，通过 ' + totalPassed +
              ' 个 (' + (100 * totalPassed / Math.max(totalChecked, 1)).toFixed(1) + '%)');
  console.log('其中梯度量级显著（|g| > 1e-4）的 ' + sigChecked + ' 个分量通过 ' + sigPassed + ' 个');
  console.log('显著分量最大相对误差: ' + maxRel.toExponential(3) + '  (最差点: ' + worst + ')');
  if (totalPassed === totalChecked && sigPassed === sigChecked) {
    console.log('结论: ✓ 反向传播实现正确（误差在数值精度范围内）');
  } else if (totalPassed / Math.max(totalChecked, 1) > 0.95) {
    console.log('结论: ~ 基本正确，但存在偏大误差，建议排查');
  } else {
    console.log('结论: ✗ 反向传播有误，相对误差过大');
  }
  console.log('');
  return totalPassed === totalChecked && sigPassed === sigChecked;
}

/* ------------------------------------------------------------------ */
/* 第二部分 + 第三部分：训练与生成                                       */
/* ------------------------------------------------------------------ */

function trainAndGenerate(corpusKey, overrides) {
  var corpus = CORPUS[corpusKey];
  var text = corpus.text;
  var vocab = kit.buildVocab(text);
  var ids = kit.encode(text, vocab.stoi);

  var cfg = Object.assign({ ctx: 32, d: 40, layers: 2, heads: 4 }, overrides || {});
  var m = new kit.MiniLM(vocab.size, cfg);

  var B = QUICK ? 8 : 12;
  var steps = QUICK ? 200 : 700;
  var lr = 0.004;
  var warmup = 60;

  console.log('='.repeat(64));
  console.log('语料: ' + corpus.name);
  console.log('='.repeat(64));
  console.log('字符数 ' + ids.length + ' | 唯一字符 ' + vocab.size +
              ' | 参数量 ' + m.paramCount().toLocaleString());
  console.log('配置 ctx=' + cfg.ctx + ' d=' + cfg.d + ' layers=' + cfg.layers +
              ' heads=' + cfg.heads + ' | batch=' + B + ' steps=' + steps);
  console.log('');

  var t0 = Date.now();
  var lossAcc = 0, lossN = 0;
  var snapshots = [];

  for (var step = 0; step < steps; step++) {
    var batch = kit.makeBatch(ids, B, cfg.ctx, m.rng.batch);
    m.zeroGrad();
    var r = m.forward(batch.tokens, batch.targets);
    m.backward();
    m.clipGrad(1.0);

    // 线性 warmup + 余弦衰减
    var curLr;
    if (step < warmup) curLr = lr * (step + 1) / warmup;
    else curLr = lr * 0.5 * (1 + Math.cos(Math.PI * (step - warmup) / (steps - warmup)));

    m.step(curLr, step + 1);
    lossAcc += r.loss; lossN++;

    if ((step + 1) % 100 === 0 || step === 0) {
      var avg = lossAcc / lossN;
      var el = (Date.now() - t0) / 1000;
      console.log('  step ' + String(step + 1).padStart(4) + '/' + steps +
                  '  loss=' + avg.toFixed(4) +
                  '  lr=' + curLr.toFixed(5) +
                  '  ' + el.toFixed(1) + 's' +
                  '  (' + ((step + 1) / el).toFixed(1) + ' step/s)');
      // 每 200 步抓一次生成快照，观察"从乱码到成句"的过程
      if ((step + 1) % 200 === 0 || step === 0) {
        var p0 = kit.encode(text.slice(0, 6), vocab.stoi);
        var g0 = m.generate(p0, { maxNew: 40, temp: 0.8, topk: 8, repPenalty: 1.15 });
        snapshots.push({ step: step + 1, text: kit.decode(g0.ids, vocab.itos) });
      }
      lossAcc = 0; lossN = 0;
    }
  }

  var elapsed = (Date.now() - t0) / 1000;
  console.log('\n训练完成，耗时 ' + elapsed.toFixed(1) + ' 秒\n');

  console.log('-'.repeat(64));
  console.log('训练过程快照（同一个提示词，不同训练步数）');
  console.log('-'.repeat(64));
  snapshots.forEach(function (s) {
    console.log('[step ' + s.step + '] ' + s.text.replace(/\n/g, ' '));
  });
  console.log('');

  var prompts = [
    '新长安的雨下了',
    '恒站在窗前，',
    '苏念说，我',
    '他低下头，看着'
  ];
  if (corpusKey === 'wuxia') prompts = ['沈砚站在', '女子说，你', '雨落在他的', '剑雨楼的灯'];
  if (corpusKey === 'urban') prompts = ['陈老板说，', '巷子口有一家', '她低头吃面', '他说，我今天'];

  console.log('-'.repeat(64));
  console.log('生成测试（top-k=8, 重复惩罚 1.15, 长度 80 字）');
  console.log('-'.repeat(64));
  [0.6, 0.8, 1.0].forEach(function (temp) {
    console.log('\n### 温度 = ' + temp);
    prompts.forEach(function (p) {
      var pids = kit.encode(p, vocab.stoi);
      var out = m.generate(pids, { maxNew: 80, temp: temp, topk: 8, repPenalty: 1.15 });
      console.log('\n> ' + p);
      console.log(kit.decode(out.ids, vocab.itos).replace(/\n/g, ' '));
    });
  });

  // 展示"模型在想什么"：下一步的候选分布
  console.log('\n' + '-'.repeat(64));
  console.log('下一步候选分布（体现模型的概率本质）');
  console.log('-'.repeat(64));
  var probe = kit.encode('苏念说，我', vocab.stoi);
  var g = m.generate(probe, { maxNew: 6, temp: 1.0, topk: 6, repPenalty: 1.0 });
  g.trace.forEach(function (tr) {
    var top = tr.candidates.map(function (c) {
      return "'" + (vocab.itos[c.id] || '?') + "' " + (c.p * 100).toFixed(1) + '%';
    }).join('  ');
    console.log('  第 ' + (tr.step + 1) + ' 步 → ' + top);
  });

  console.log('');
  return m;
}

/* ------------------------------------------------------------------ */

var ok = gradCheck(kit64);

if (F64_ONLY) {
  console.log('（仅梯度检查模式，跳过训练）');
  process.exit(ok ? 0 : 1);
}

if (QUICK) {
  trainAndGenerate('scifi');
} else {
  trainAndGenerate('scifi');
  console.log('\n\n');
  trainAndGenerate('wuxia');
  console.log('\n\n');
  trainAndGenerate('urban');
}

console.log('\n' + '='.repeat(64));
console.log('全部测试完成');
console.log('='.repeat(64));
