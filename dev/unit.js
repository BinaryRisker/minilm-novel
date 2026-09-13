/* attention 与 layernorm 的独立单元测试：用有限差分验证反向传播 */
var kit = require('./minilm.js');

function randn(n) {
  var a = new Float32Array(n);
  for (var i = 0; i < n; i++) {
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    a[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  return a;
}
function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

var EPS = 1e-3;

/* ---------------- 测试 1：attention ---------------- */
function testAttention(B, T, D, H) {
  var q = randn(B * T * D), k = randn(B * T * D), v = randn(B * T * D);
  var attnStore = new Float32Array(B * H * T * T);
  var out = kit.attentionForward(q, k, v, B, T, D, H, attnStore);

  // 损失 = <out, W>，于是 dout = W
  var W = randn(B * T * D);
  var loss0 = dot(out, W);
  if (!isFinite(loss0)) { console.log('  loss 非有限值'); return; }

  var dq = new Float32Array(B * T * D);
  var dk = new Float32Array(B * T * D);
  var dv = new Float32Array(B * T * D);
  kit.attentionBackward(W, q, k, v, attnStore, B, T, D, H, dq, dk, dv);

  // 数值梯度
  function numGrad(arr, darr, name) {
    var worst = 0, wi = -1, wn = 0, wa = 0, nStrong = 0, nOk = 0, sn = 0, sa = 0, dt = 0;
    for (var i = 0; i < arr.length; i++) {
      var o = arr[i];
      arr[i] = o + EPS;
      var at = new Float32Array(B * H * T * T);
      var ot = kit.attentionForward(q, k, v, B, T, D, H, at);
      var lp = dot(ot, W);

      arr[i] = o - EPS;
      var at2 = new Float32Array(B * H * T * T);
      var ot2 = kit.attentionForward(q, k, v, B, T, D, H, at2);
      var lm = dot(ot2, W);
      arr[i] = o;

      var num = (lp - lm) / (2 * EPS);
      var ana = darr[i];
      sn += num * num; sa += ana * ana; dt += num * ana;
      if (Math.abs(num) > 1e-3 || Math.abs(ana) > 1e-3) {
        nStrong++;
        var rel = Math.abs(num - ana) / Math.max(Math.abs(num), Math.abs(ana));
        if (rel < 0.02) nOk++;
        if (rel > worst) { worst = rel; wi = i; wn = num; wa = ana; }
      }
    }
    var cos = dt / (Math.sqrt(sn) * Math.sqrt(sa) + 1e-12);
    console.log('  d' + name + ': 显著=' + nStrong + ' 通过=' + nOk +
      ' 最差rel=' + worst.toFixed(4) + ' 余弦=' + cos.toFixed(6) +
      (worst > 0.02 ? '  (idx=' + wi + ' num=' + wn.toExponential(3) + ' ana=' + wa.toExponential(3) + ')' : ''));
  }

  console.log('attention  B=' + B + ' T=' + T + ' D=' + D + ' H=' + H);
  numGrad(q, dq, 'q');
  numGrad(k, dk, 'k');
  numGrad(v, dv, 'v');
}

/* ---------------- 测试 2：layernorm ---------------- */
function testLN(M, D) {
  var x = randn(M * D);
  var g = new Float32Array(D), b = new Float32Array(D);
  for (var i = 0; i < D; i++) { g[i] = 1 + 0.1 * (Math.random() - 0.5); b[i] = 0.1 * (Math.random() - 0.5); }

  var cache = {};
  var y = kit.layernorm(x, g, b, M, D, 1e-5, cache);

  var W = randn(M * D);
  var dx = new Float32Array(M * D), dg = new Float32Array(D), db = new Float32Array(D);
  kit.layernormBack(W, M, D, cache, dx, dg, db);

  function check(name, arr, darr) {
    var worst = 0, wi = -1, wn = 0, wa = 0, nStrong = 0, nOk = 0, sn = 0, sa = 0, dt = 0;
    for (var i = 0; i < arr.length; i++) {
      var o = arr[i];
      arr[i] = o + EPS;
      var c1 = {};
      var y1 = kit.layernorm(x, g, b, M, D, 1e-5, c1);
      var lp = dot(y1, W);
      arr[i] = o - EPS;
      var c2 = {};
      var y2 = kit.layernorm(x, g, b, M, D, 1e-5, c2);
      var lm = dot(y2, W);
      arr[i] = o;

      var num = (lp - lm) / (2 * EPS);
      var ana = darr[i];
      sn += num * num; sa += ana * ana; dt += num * ana;
      if (Math.abs(num) > 1e-3 || Math.abs(ana) > 1e-3) {
        nStrong++;
        var rel = Math.abs(num - ana) / Math.max(Math.abs(num), Math.abs(ana));
        if (rel < 0.02) nOk++;
        if (rel > worst) { worst = rel; wi = i; wn = num; wa = ana; }
      }
    }
    var cos = dt / (Math.sqrt(sn) * Math.sqrt(sa) + 1e-12);
    console.log('  d' + name + ': 显著=' + nStrong + ' 通过=' + nOk +
      ' 最差rel=' + worst.toFixed(4) + ' 余弦=' + cos.toFixed(6) +
      (worst > 0.02 ? '  (idx=' + wi + ' num=' + wn.toExponential(3) + ' ana=' + wa.toExponential(3) + ')' : ''));
  }

  console.log('layernorm  M=' + M + ' D=' + D);
  check('x', x, dx);
  check('gamma', g, dg);
  check('beta', b, db);
}

console.log('='.repeat(70));
console.log('单元测试：反向传播算子级验证');
console.log('='.repeat(70));
testLN(6, 8);
testAttention(2, 4, 8, 2);
testAttention(2, 6, 12, 3);
console.log('='.repeat(70));
