/* 双精度、多配置的 attention 反向验证（含模型真实维度 T=8） */
globalThis.MINILM_USE_F64 = true;
var kit = require('./minilm.js');

function randn64(n) {
  var a = new Float64Array(n);
  for (var i = 0; i < n; i++) {
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    a[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  return a;
}
function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

var EPS = 1e-6;

function test(B, T, D, H, correlated) {
  var q, k, v;
  if (correlated) {
    // 模拟模型：q,k,v 都来自同一个 xn1（线性投影）
    var X = randn64(B * T * D);
    var Wq = randn64(D * D), Wk = randn64(D * D), Wv = randn64(D * D);
    q = new Float64Array(B * T * D); k = new Float64Array(B * T * D); v = new Float64Array(B * T * D);
    for (var i = 0; i < B * T; i++) {
      for (var j = 0; j < D; j++) {
        var s1 = 0, s2 = 0, s3 = 0;
        for (var m = 0; m < D; m++) {
          var xv = X[i * D + m];
          s1 += xv * Wq[m * D + j]; s2 += xv * Wk[m * D + j]; s3 += xv * Wv[m * D + j];
        }
        q[i * D + j] = s1; k[i * D + j] = s2; v[i * D + j] = s3;
      }
    }
  } else {
    q = randn64(B * T * D); k = randn64(B * T * D); v = randn64(B * T * D);
  }

  var attnStore = new Float64Array(B * H * T * T);
  var out = kit.attentionForward(q, k, v, B, T, D, H, attnStore);
  var W = randn64(B * T * D);

  var dq = new Float64Array(B * T * D), dk = new Float64Array(B * T * D), dv = new Float64Array(B * T * D);
  kit.attentionBackward(W, q, k, v, attnStore, B, T, D, H, dq, dk, dv);

  function gradCheck(arr, darr, label) {
    var worst = 0, wi = -1, wn = 0, wa = 0, sn = 0, sa = 0, dt = 0;
    for (var i = 0; i < arr.length; i++) {
      var o = arr[i];
      arr[i] = o + EPS;
      var at = new Float64Array(B * H * T * T);
      var ot = kit.attentionForward(q, k, v, B, T, D, H, at);
      var lp = dot(ot, W);
      arr[i] = o - EPS;
      var at2 = new Float64Array(B * H * T * T);
      var ot2 = kit.attentionForward(q, k, v, B, T, D, H, at2);
      var lm = dot(ot2, W);
      arr[i] = o;
      var num = (lp - lm) / (2 * EPS), ana = darr[i];
      sn += num * num; sa += ana * ana; dt += num * ana;
      var rel = Math.abs(num - ana) / Math.max(Math.abs(num), Math.abs(ana), 1e-12);
      if (rel > worst) { worst = rel; wi = i; wn = num; wa = ana; }
    }
    var cos = dt / (Math.sqrt(sn) * Math.sqrt(sa) + 1e-12);
    return { label: label, worst: worst, wi: wi, wn: wn, wa: wa, cos: cos };
  }

  var res = [gradCheck(q, dq, 'dq'), gradCheck(k, dk, 'dk'), gradCheck(v, dv, 'dv')];
  var tag = 'B=' + B + ' T=' + T + ' D=' + D + ' H=' + H + (correlated ? ' [相关输入]' : ' [独立随机]');
  var ok = res.every(function (r) { return r.cos > 0.999; });
  console.log((ok ? '✓ ' : '✗ ') + tag);
  res.forEach(function (r) {
    console.log('    ' + r.label + ' 余弦=' + r.cos.toFixed(8) + ' 最差rel=' + r.worst.toExponential(2) +
      (r.worst > 1e-4 ? '  (idx=' + r.wi + ' num=' + r.wn.toExponential(3) + ' ana=' + r.wa.toExponential(3) + ')' : ''));
  });
}

console.log('=== attention 反向传播验证（float64）===');
console.log('');
test(2, 4, 8, 2, false);
test(2, 8, 8, 2, false);
test(2, 8, 8, 2, true);
test(1, 8, 8, 1, true);
test(2, 8, 8, 4, true);
test(1, 32, 40, 4, true);
