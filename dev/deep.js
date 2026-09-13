/* 深入 layernormBack：校验缓存一致性 + 手工复刻对比 */
globalThis.MINILM_USE_F64 = true;
var kit = require('./minilm.js');

var text = '从前有座山，山里有座庙，庙里有个老和尚，老和尚对小和尚说。';
var vocab = kit.buildVocab(text);
var cfg = { ctx: 8, d: 8, layers: 1, heads: 2, ffMul: 2 };
var m = new kit.MiniLM(vocab.size, cfg);
var ids = kit.encode(text, vocab.stoi);

var T = 8, B = 1, BT = B * T, D = cfg.d, V = vocab.size;
var tokens = new Int32Array(BT), targets = new Int32Array(BT);
for (var i = 0; i < BT; i++) { tokens[i] = ids[i % ids.length]; targets[i] = ids[(i + 1) % ids.length]; }

m.DEBUG = true;
m.zeroGrad();
m.forward(tokens, targets);
m.backward();

var cache = m.cache.lnfc;
var dxnf = m._dxnf, dx = m._dx, xnf = m.cache.xnf;

console.log('=== 校验 1：用 cache 重算 xnf，应与缓存的 xnf 完全一致 ===');
var recomputed = kit.layernorm(cache.x, m.P.lnfg, m.P.lnfb, BT, D, 1e-5, {});
var maxdiff = 0;
for (var z = 0; z < xnf.length; z++) maxdiff = Math.max(maxdiff, Math.abs(recomputed[z] - xnf[z]));
console.log('最大差异 =', maxdiff.toExponential(3), maxdiff < 1e-14 ? '  => 缓存一致' : '  => 缓存不一致！');
console.log('');

console.log('=== 校验 2：cache.x 与 cache.mean / cache.rstd 是否自洽 ===');
// 用 cache.x 独立算 mean 和 rstd，看是否等于 cache 里存的
var bad = 0;
for (var a = 0; a < BT; a++) {
  var off = a * D, s = 0;
  for (var d = 0; d < D; d++) s += cache.x[off + d];
  var mu = s / D;
  var vv = 0;
  for (var d2 = 0; d2 < D; d2++) { var t = cache.x[off + d2] - mu; vv += t * t; }
  vv /= D;
  var rs = 1 / Math.sqrt(vv + 1e-5);
  if (Math.abs(mu - cache.mean[a]) > 1e-12 || Math.abs(rs - cache.rstd[a]) > 1e-9) {
    bad++;
    console.log('  行 ' + a + ': 重算 mu=' + mu.toFixed(10) + ' rstd=' + rs.toFixed(6) +
                ' | 缓存 mu=' + cache.mean[a].toFixed(10) + ' rstd=' + cache.rstd[a].toFixed(6));
  }
}
console.log(bad === 0 ? '  所有行自洽 => 缓存正确' : '  有 ' + bad + ' 行不自洽！');
console.log('');

console.log('=== 校验 3：手工复刻 layernormBack，与实现的输出对比 ===');
var dxh = new Float64Array(BT * D);
for (var a2 = 0; a2 < BT; a2++) {
  var off2 = a2 * D, mu2 = cache.mean[a2], rs2 = cache.rstd[a2];
  var dvar = 0, dmu = 0;
  for (var b = 0; b < D; b++) {
    var xh = (cache.x[off2 + b] - mu2) * rs2;
    var dh = dxnf[off2 + b] * cache.g[b];
    dxh[off2 + b] += dh * rs2;
    dvar += dh * (cache.x[off2 + b] - mu2) * (-0.5) * rs2 * rs2 * rs2;
    dmu += dh * (-rs2);
  }
  for (var c = 0; c < D; c++) {
    dxh[off2 + c] += dvar * 2 * (cache.x[off2 + c] - mu2) / D + dmu / D;
  }
}
var maxdiff2 = 0;
for (var e = 0; e < dx.length; e++) maxdiff2 = Math.max(maxdiff2, Math.abs(dxh[e] - dx[e]));
console.log('手工复刻 vs 实现 最大差异 =', maxdiff2.toExponential(3),
  maxdiff2 < 1e-14 ? '  => 实现与公式一致' : '  => 实现与公式不一致！');
console.log('');

console.log('=== 校验 4：用"标准公式"的另一种写法验证 ===');
// 标准形式：dx_i = rs*g_i*dy_i - (rs/N)*sum_j(g_j*dy_j) - (rs/N)*xhat_i*sum_j(g_j*dy_j*xhat_j)
var dxs = new Float64Array(BT * D);
for (var a3 = 0; a3 < BT; a3++) {
  var off3 = a3 * D, mu3 = cache.mean[a3], rs3 = cache.rstd[a3];
  var s1 = 0, s2 = 0;
  for (var b3 = 0; b3 < D; b3++) {
    var xh3 = (cache.x[off3 + b3] - mu3) * rs3;
    var gd = cache.g[b3] * dxnf[off3 + b3];
    s1 += gd;
    s2 += gd * xh3;
  }
  for (var c3 = 0; c3 < D; c3++) {
    var xh4 = (cache.x[off3 + c3] - mu3) * rs3;
    dxs[off3 + c3] = rs3 * cache.g[c3] * dxnf[off3 + c3] - (rs3 / D) * s1 - (rs3 / D) * xh4 * s2;
  }
}
var maxdiff3 = 0;
for (var f = 0; f < dx.length; f++) maxdiff3 = Math.max(maxdiff3, Math.abs(dxs[f] - dx[f]));
console.log('标准公式 vs 实现 最大差异 =', maxdiff3.toExponential(3));
console.log('');

console.log('=== 校验 5：逐元素对比 实现 vs 手工（前 8 个）===');
for (var k = 0; k < 8; k++) {
  console.log('  idx=' + k + '  实现=' + dx[k].toExponential(8) + '  手工=' + dxh[k].toExponential(8) + '  标准=' + dxs[k].toExponential(8));
}
