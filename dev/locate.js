/* 精确定位：B=1 时 G.pos 应逐元素等于输入梯度 dx */
globalThis.MINILM_USE_F64 = true;
var kit = require('./minilm.js');

var text = '从前有座山，山里有座庙，庙里有个老和尚，老和尚对小和尚说。';
var vocab = kit.buildVocab(text);
var cfg = { ctx: 8, d: 8, layers: 1, heads: 2, ffMul: 2 };
var m = new kit.MiniLM(vocab.size, cfg);
var ids = kit.encode(text, vocab.stoi);

var T = 8, B = 1;                       // B=1：pos[t] 的梯度 = dx[t]
var tokens = new Int32Array(B * T), targets = new Int32Array(B * T);
for (var i = 0; i < B * T; i++) {
  tokens[i] = ids[i % ids.length];
  targets[i] = ids[(i + 1) % ids.length];
}

m.zeroGrad();
var r0 = m.forward(tokens, targets);
m.backward();

function nrm(a) { var s = 0; for (var q = 0; q < a.length; q++) s += a[q] * a[q]; return Math.sqrt(s); }

console.log('loss =', r0.loss.toFixed(10));
console.log('|G.pos| (解析) =', nrm(m.G.pos).toExponential(6));
console.log('');

// 数值梯度
var EPS = 1e-6;
var numPos = [];
for (var idx = 0; idx < m.P.pos.length; idx++) {
  var o = m.P.pos[idx];
  m.P.pos[idx] = o + EPS;
  var lp = m.forward(tokens, targets).loss;
  m.P.pos[idx] = o - EPS;
  var lm = m.forward(tokens, targets).loss;
  m.P.pos[idx] = o;
  numPos[idx] = (lp - lm) / (2 * EPS);
}
console.log('|num.pos| (数值) =', nrm(numPos).toExponential(6));
console.log('');

console.log('逐元素对比 pos（前 16 个）:');
console.log('idx   解析           数值           比值(解析/数值)');
for (var k = 0; k < 16; k++) {
  var a = m.G.pos[k], n = numPos[k];
  var ratio = Math.abs(n) > 1e-12 ? (a / n).toFixed(4) : 'n/a';
  console.log(String(k).padStart(3) + '  ' + a.toExponential(6) + '  ' + n.toExponential(6) + '   ' + ratio);
}

// 同样对比 pos 与 emb 的"输入贡献"关系：
// emb 的梯度 = 输出层贡献 + dx ；pos 的梯度 = dx
// 所以 G.emb - G.pos 应该等于"输出层贡献"，且其结构应与 xnf 相关
console.log('');
console.log('检查关键中间量范数:');
console.log('  |logits| =', nrm(m.cache.logits).toExponential(4));
console.log('  |xnf|    =', nrm(m.cache.xnf).toExponential(4));
console.log('  |xn1|    =', nrm(m.cache.layers[0].xn1).toExponential(4));
console.log('  |ao|     =', nrm(m.cache.layers[0].ao).toExponential(4));
console.log('  |attnStore| =', nrm(m.cache.layers[0].attnStore).toExponential(4));
console.log('  attnStore 行和（应为 1）=',
  (function () {
    var s = 0;
    for (var t = 0; t < T; t++) {
      var b0 = (0 * cfg.heads + 0) * T * T + t * T;
      var r = 0;
      for (var q = 0; q <= t; q++) r += m.cache.layers[0].attnStore[b0 + q];
      s += r;
    }
    return (s / T).toFixed(6);
  })());
