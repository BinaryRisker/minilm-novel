/* 二分定位：先验证输出层反向（xnf -> logits -> loss），再验证最终 LayerNorm 反向 */
globalThis.MINILM_USE_F64 = true;
var kit = require('./minilm.js');

var text = '从前有座山，山里有座庙，庙里有个老和尚，老和尚对小和尚说。';
var vocab = kit.buildVocab(text);
var cfg = { ctx: 8, d: 8, layers: 1, heads: 2, ffMul: 2 };
var m = new kit.MiniLM(vocab.size, cfg);
var ids = kit.encode(text, vocab.stoi);

var T = 8, B = 1, BT = B * T, D = cfg.d, V = vocab.size;
var tokens = new Int32Array(BT), targets = new Int32Array(BT);
for (var i = 0; i < BT; i++) {
  tokens[i] = ids[i % ids.length];
  targets[i] = (i + 1) % ids.length < ids.length ? ids[(i + 1) % ids.length] : ids[0];
}
targets[BT - 1] = ids[0];

m.DEBUG = true;
m.zeroGrad();
var r0 = m.forward(tokens, targets);
m.backward();

var xnf = m.cache.xnf, emb = m.P.emb, dxnf = m._dxnf, dx = m._dx;

function nrm(a) { var s = 0; for (var q = 0; q < a.length; q++) s += a[q] * a[q]; return Math.sqrt(s); }

/* ---------- 复刻"从 xnf 到 loss"的后半段 ---------- */
function lossFromXnf(x) {
  var logits = new Float64Array(BT * V);
  for (var b = 0; b < BT; b++) {
    for (var j = 0; j < V; j++) {
      var s = 0;
      for (var d = 0; d < D; d++) s += x[b * D + d] * emb[j * D + d];
      logits[b * V + j] = s;
    }
  }
  var loss = 0;
  for (var b2 = 0; b2 < BT; b2++) {
    var off = b2 * V, mx = -Infinity, j2;
    for (j2 = 0; j2 < V; j2++) if (logits[off + j2] > mx) mx = logits[off + j2];
    var sum = 0, probs = new Float64Array(V);
    for (j2 = 0; j2 < V; j2++) { var e = Math.exp(logits[off + j2] - mx); probs[j2] = e; sum += e; }
    loss -= Math.log(probs[targets[b2]] / sum + 1e-12);
  }
  return loss / BT;
}

/* ---------- 检查 1：dxnf 是否正确 ---------- */
console.log('=== 检查 1：输出层反向 (xnf -> loss) ===');
console.log('复刻 loss =', lossFromXnf(xnf).toFixed(10), ' 模型 loss =', r0.loss.toFixed(10));
console.log('');

var EPS = 1e-7;
var worst = 0, wi = -1, wn = 0, wa = 0;
for (var idx = 0; idx < xnf.length; idx++) {
  var o = xnf[idx];
  var x1 = Float64Array.from(xnf); x1[idx] = o + EPS;
  var x2 = Float64Array.from(xnf); x2[idx] = o - EPS;
  var num = (lossFromXnf(x1) - lossFromXnf(x2)) / (2 * EPS);
  var ana = dxnf[idx];
  if (Math.abs(num) > 1e-7 || Math.abs(ana) > 1e-7) {
    var rel = Math.abs(num - ana) / Math.max(Math.abs(num), Math.abs(ana));
    if (rel > worst) { worst = rel; wi = idx; wn = num; wa = ana; }
  }
}
console.log('dxnf 最大相对误差 =', worst.toExponential(3), ' (idx=' + wi + ' num=' + wn.toExponential(3) + ' ana=' + wa.toExponential(3) + ')');
console.log(worst < 1e-4 ? '  => 输出层反向【正确】' : '  => 输出层反向【错误】');
console.log('');

/* ---------- 检查 2：最终 LayerNorm 反向 ---------- */
console.log('=== 检查 2：最终 LayerNorm 反向 (cur -> xnf) ===');
console.log('|dxnf| =', nrm(dxnf).toExponential(4));
console.log('|dx|   =', nrm(dx).toExponential(4));
console.log('|cur|  =', nrm(m.cache.lnfc.x).toExponential(4));
console.log('rstd[0] =', m.cache.lnfc.rstd[0].toFixed(6), ' mean[0] =', m.cache.lnfc.mean[0].toFixed(6));
console.log('gamma 范数 =', nrm(m.P.lnfg).toExponential(4), ' (应为 sqrt(8)=2.83，若偏差大说明未初始化为 1)');
console.log('');

// 用独立实现验证：d(cur) = 通过 xnf 的链式
function xnfFromCur(cur) {
  return kit.layernorm(cur, m.P.lnfg, m.P.lnfb, BT, D, 1e-5, {});
}
var worst2 = 0, wi2 = -1;
for (var k = 0; k < m.cache.lnfc.x.length; k++) {
  var o2 = m.cache.lnfc.x[k];
  var c1 = Float64Array.from(m.cache.lnfc.x); c1[k] = o2 + EPS;
  var c2 = Float64Array.from(m.cache.lnfc.x); c2[k] = o2 - EPS;
  // loss 对 cur 的数值梯度 = <dxnf_true, d(xnf)/d(cur)>
  // 直接用 lossFromXnf 组合
  var num2 = (lossFromXnf(xnfFromCur(c1)) - lossFromXnf(xnfFromCur(c2))) / (2 * EPS);
  var ana2 = dx[k];
  if (Math.abs(num2) > 1e-7 || Math.abs(ana2) > 1e-7) {
    var rel2 = Math.abs(num2 - ana2) / Math.max(Math.abs(num2), Math.abs(ana2));
    if (rel2 > worst2) { worst2 = rel2; wi2 = k; }
  }
}
console.log('d(cur) 最大相对误差 =', worst2.toExponential(3), ' (idx=' + wi2 + ')');
console.log(worst2 < 1e-4 ? '  => 最终 LayerNorm 反向【正确】' : '  => 最终 LayerNorm 反向【错误】');
