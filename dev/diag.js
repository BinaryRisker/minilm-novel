/* 梯度诊断：逐参数检查解析梯度 vs 数值梯度
 * 使用 float64 精度，避免 float32 量化噪声淹没小梯度分量。 */
globalThis.MINILM_USE_F64 = true;
var kit = require('./minilm.js');

var text = '从前有座山，山里有座庙，庙里有个老和尚，老和尚对小和尚说。';
var vocab = kit.buildVocab(text);
var cfg = { ctx: 8, d: 8, layers: 1, heads: 2, ffMul: 2 };
var m = new kit.MiniLM(vocab.size, cfg);
var ids = kit.encode(text, vocab.stoi);

var T = 8, B = 2;
var tokens = new Int32Array(B * T), targets = new Int32Array(B * T);
for (var i = 0; i < B * T; i++) {
  tokens[i] = ids[i % ids.length];
  targets[i] = ids[(i + 1) % ids.length];
}

m.zeroGrad();
var r0 = m.forward(tokens, targets);
m.backward();
console.log('loss =', r0.loss.toFixed(8));
console.log('');

var EPS = 1e-5;

function diag(name) {
  var P = m.P[name], G = m.G[name];
  var worst = 0, wi = -1, wn = 0, wa = 0;
  var strong = 0, strongOk = 0;
  var sumNum = 0, sumAna = 0, dot = 0, nA = 0, nB = 0;

  for (var idx = 0; idx < P.length; idx++) {
    var orig = P[idx];
    P[idx] = orig + EPS; var lp = m.forward(tokens, targets).loss;
    P[idx] = orig - EPS; var lm = m.forward(tokens, targets).loss;
    P[idx] = orig;

    var num = (lp - lm) / (2 * EPS);
    var ana = G[idx];

    // 只对"显著"分量做严格比较，避免噪声放大
    if (Math.abs(num) > 1e-6 || Math.abs(ana) > 1e-6) {
      strong++;
      var denom = Math.max(Math.abs(num), Math.abs(ana));
      var rel = Math.abs(num - ana) / denom;
      if (rel < 1e-4) strongOk++;
      if (rel > worst) { worst = rel; wi = idx; wn = num; wa = ana; }
    }
    sumNum += num * num; sumAna += ana * ana; dot += num * ana; nA++;
  }

  var cos = dot / (Math.sqrt(sumNum) * Math.sqrt(sumAna) + 1e-12);
  console.log(name.padEnd(8) +
    ' dim=' + String(P.length).padStart(3) +
    '  显著分量 ' + String(strong).padStart(3) + ' (通过 ' + strongOk + ')' +
    '  最差rel=' + worst.toFixed(4) +
    '  余弦相似度=' + cos.toFixed(6));
  if (worst > 1e-4) {
    console.log('        └ 最差点 idx=' + wi + '  num=' + wn.toExponential(4) + '  ana=' + wa.toExponential(4));
  }
}

console.log('参数名    维度      显著分量/通过      最差相对误差      整体方向一致性');
console.log('-'.repeat(78));
['emb', 'pos', 'ln1g0', 'ln1b0', 'Wq0', 'bq0', 'Wk0', 'Wv0', 'Wo0',
 'ln2g0', 'ln2b0', 'W1_0', 'b1_0', 'W2_0', 'b2_0', 'lnfg', 'lnfb'
].forEach(function (n) { if (m.P[n]) diag(n); });
