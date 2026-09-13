/* 标定梯度检查的 eps 与判定阈值：多次随机初始化，看最大相对误差的分布 */
var kit = require('./minilm.js');

function run(useF64, eps, d, layers, seedInfo) {
  globalThis.MINILM_USE_F64 = useF64;
  delete require.cache[require.resolve('./minilm.js')];
  var K = require('./minilm.js');

  var text = '从前有座山，山里有座庙，庙里有个老和尚，老和尚对小和尚说。今天天气不错，我们一起去山上看看吧。';
  var vocab = K.buildVocab(text);
  var m = new K.MiniLM(vocab.size, { ctx: 8, d: d, layers: layers, heads: 2, ffMul: 2 });
  var ids = K.encode(text, vocab.stoi);
  var T = 8, B = 2;
  var tk = new Int32Array(B * T), tg = new Int32Array(B * T);
  for (var i = 0; i < B * T; i++) { tk[i] = ids[i % ids.length]; tg[i] = ids[(i + 1) % ids.length]; }
  m.zeroGrad();
  m.forward(tk, tg);
  m.backward();

  var maxRel = 0, maxRelSmall = 0, checked = 0, over1e4 = 0, over1e3 = 0, worst = '';
  var maxAbsOverSmall = 0;
  m.paramList.forEach(function (p) {
    var P = m.P[p.name], G = m.G[p.name], n = P.length;
    for (var idx = 0; idx < n; idx++) {
      var orig = P[idx];
      P[idx] = orig + eps; var lp = m.forward(tk, tg).loss;
      P[idx] = orig - eps; var lm = m.forward(tk, tg).loss;
      P[idx] = orig;
      var num = (lp - lm) / (2 * eps), ana = G[idx];
      if (Math.abs(num) < 1e-7 && Math.abs(ana) < 1e-7) continue;
      checked++;
      var den = Math.max(Math.abs(num), Math.abs(ana));
      var rel = Math.abs(num - ana) / Math.max(den, 1e-12);
      if (rel > maxRel) { maxRel = rel; worst = p.name + '[' + idx + '] num=' + num.toExponential(2) + ' ana=' + ana.toExponential(2) + ' |num|=' + Math.abs(num).toExponential(2); }
      if (rel > 1e-4) over1e4++;
      if (rel > 1e-3) over1e3++;
    }
  });
  return { maxRel: maxRel, checked: checked, over1e4: over1e4, over1e3: over1e3, worst: worst };
}

console.log('=== float64, eps=1e-5, 5 次随机初始化 ===');
for (var t = 0; t < 5; t++) {
  var r = run(true, 1e-5, 8, 1);
  console.log('  maxRel=' + r.maxRel.toExponential(2) + '  checked=' + r.checked +
    '  >1e-4: ' + r.over1e4 + '  >1e-3: ' + r.over1e3);
  if (t === 0) console.log('    worst: ' + r.worst);
}

console.log('\n=== float64, eps=1e-6, 5 次随机初始化 ===');
for (var t2 = 0; t2 < 5; t2++) {
  var r2 = run(true, 1e-6, 8, 1);
  console.log('  maxRel=' + r2.maxRel.toExponential(2) + '  checked=' + r2.checked +
    '  >1e-4: ' + r2.over1e4 + '  >1e-3: ' + r2.over1e3);
  if (t2 === 0) console.log('    worst: ' + r2.worst);
}

console.log('\n=== float64, eps=1e-5, d=16 layers=2（大一点的模型）===');
for (var t3 = 0; t3 < 3; t3++) {
  var r3 = run(true, 1e-5, 16, 2);
  console.log('  maxRel=' + r3.maxRel.toExponential(2) + '  checked=' + r3.checked +
    '  >1e-4: ' + r3.over1e4 + '  >1e-3: ' + r3.over1e3);
  if (t3 === 0) console.log('    worst: ' + r3.worst);
}

console.log('\n=== float32, eps=1e-3（对照：不换精度会怎样）===');
for (var t4 = 0; t4 < 3; t4++) {
  var r4 = run(false, 1e-3, 8, 1);
  console.log('  maxRel=' + r4.maxRel.toExponential(2) + '  checked=' + r4.checked +
    '  >1e-4: ' + r4.over1e4 + '  >1e-3: ' + r4.over1e3);
}
