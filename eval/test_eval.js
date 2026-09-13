/* ============================================================================
 * eval/test_eval.js — 指标自测（用 eval/fixtures/README.md 里手算的期望值）
 * ----------------------------------------------------------------------------
 * 为什么单测要用手算值而不是"上一次的输出"：
 *   用上次输出当期望值，等于把 bug 固化成规范。这里每个数字都能在 README 里
 *   用纸笔推出来，改了实现而期望值没变 → 测试红 → 要么实现错了要么 README 该改。
 *
 * 用法：node eval/test_eval.js
 * ============================================================================ */

var fs = require('fs');
var path = require('path');

var kit = require('../minilm.js');
var PROTO = require('./protocol.js');
var BASE = require('./baselines.js');
var M = require('./metrics.js');

var FIX = path.join(__dirname, 'fixtures');
function fix(name, expectLen) {
  var raw = fs.readFileSync(path.join(FIX, name), 'utf8');
  if (expectLen != null && raw.length !== expectLen) {
    throw new Error('fixture 被污染：' + name + ' 应为 ' + expectLen +
      ' 字节，实际 ' + raw.length + '（可能多了换行或 BOM）。' +
      '请重新写入，不要改动期望值。');
  }
  return raw;
}

var pass = 0, fail = 0, logs = [];
function ok(name, cond, detail) {
  if (cond) { pass++; logs.push('  ✓ ' + name); }
  else { fail++; logs.push('  ✗ ' + name + (detail ? '  →  ' + detail : '')); }
}
function near(name, got, want, eps) {
  var d = Math.abs(got - want);
  ok(name, d <= (eps == null ? 1e-9 : eps),
    'got=' + got + ' want=' + want + ' Δ=' + d);
}
function section(t) { logs.push(''); logs.push(t); }

/* ========================================================================== */
section('【1】n-gram 概率（fixture: tiny_train.txt = "abababab"）');
/* ========================================================================== */

var tiny = fix('tiny_train.txt', 8);
var vTiny = kit.buildVocab(tiny);
var idTiny = kit.encode(tiny, vTiny.stoi);

ok('词表大小 V = 3（\\0 + a + b）', vTiny.size === 3, 'V=' + vTiny.size);
ok('编码 = [1,2,1,2,1,2,1,2]',
  JSON.stringify(idTiny) === JSON.stringify([1, 2, 1, 2, 1, 2, 1, 2]),
  JSON.stringify(idTiny));

// 约定：A=1(a)  B=2(b)  O=0(\0)
var A = 1, B = 2, O = 0;

var uni = new BASE.NGram(idTiny, 1, 1, 3);
near('unigram  P(a)   = 5/11', Math.exp(uni.logProb([], A)), 5 / 11);
near('unigram  P(\\0) = 1/11', Math.exp(uni.logProb([], O)), 1 / 11);

var bi = new BASE.NGram(idTiny, 2, 1, 3);
near('bigram   P(b|a) = 5/7', Math.exp(bi.logProb([A], B)), 5 / 7);
near('bigram   P(a|a) = 1/7', Math.exp(bi.logProb([A], A)), 1 / 7);
near('bigram   P(a|b) = 2/3', Math.exp(bi.logProb([B], A)), 2 / 3);
near('bigram   P(b|b) = 1/6', Math.exp(bi.logProb([B], B)), 1 / 6);

var tri = new BASE.NGram(idTiny, 3, 1, 3);
near('trigram  P(a|ab) = 2/3', Math.exp(tri.logProb([A, B], A)), 2 / 3);
near('trigram  P(b|ab) = 1/6', Math.exp(tri.logProb([A, B], B)), 1 / 6);
near('trigram  P(b|ba) = 2/3', Math.exp(tri.logProb([B, A], B)), 2 / 3);

// 回退：上文不够长 / 上下文没见过
near('回退 上文只有 1 字 → 退 bigram  P(b|b) = 1/6',
  Math.exp(tri.logProb([B], B)), 1 / 6);
near('回退 上下文 "aa" 未见过 → 退 bigram  P(b|a) = 5/7',
  Math.exp(tri.logProb([A, A], B)), 5 / 7);
ok('usedOrder([A,B]) = 3', tri.usedOrder([A, B]) === 3, String(tri.usedOrder([A, B])));
ok('usedOrder([B]) = 2（上文不够，降到 bigram）', tri.usedOrder([B]) === 2, String(tri.usedOrder([B])));
ok('usedOrder([A,A]) = 2（上下文没见过，降到 bigram）', tri.usedOrder([A, A]) === 2, String(tri.usedOrder([A, A])));

// 归一化：任意上下文下概率和为 1
[[], [A], [B], [A, B], [B, A]].forEach(function (h) {
  var d = tri.dist(h), s = 0;
  for (var v = 0; v < 3; v++) s += Math.exp(d[v]);
  near('归一化  hist=[' + h.join(',') + ']', s, 1, 1e-12);
});

// α 只影响 NLL，不影响 argmax（README 里那条注记的实证）
var biA = new BASE.NGram(idTiny, 2, 0.01, 3);
near('α=0.01  P(b|a) = 4.01/4.03', Math.exp(biA.logProb([A], B)), 4.01 / 4.03, 1e-12);
var d1 = new BASE.NGram(idTiny, 2, 0.01, 3).dist([A]);
var d2 = new BASE.NGram(idTiny, 2, 1.0, 3).dist([A]);
var am1 = 0, am2 = 0;
for (var v = 0; v < 3; v++) { if (d1[v] > d1[am1]) am1 = v; if (d2[v] > d2[am2]) am2 = v; }
ok('argmax 与 α 无关（α=0.01 与 α=1 取到同一个字）', am1 === am2, am1 + ' vs ' + am2);

/* ========================================================================== */
section('【2】重复率（fixture: repeat.txt = "abcabcabc"）');
/* ========================================================================== */

var rep = fix('repeat.txt', 9);
near('distinct-1 = 1/3', M.distinctN(rep, 1), 1 / 3, 1e-12);
near('distinct-2 = 3/8', M.distinctN(rep, 2), 3 / 8, 1e-12);
near('distinct-4 = 3/6', M.distinctN(rep, 4), 3 / 6, 1e-12);
near('4-gram 重复率 = 3/6', M.repeatRate(rep, 4), 3 / 6, 1e-12);
near('distinct-N 在文本短于 N 时 = 0', M.distinctN('ab', 5), 0);
near('完全不重复的文本重复率 = 0', M.repeatRate('abcdefgh', 4), 0);

/* ========================================================================== */
section('【3】重合度 / 最长公共子串（gen="abcabdabc" vs train="abcabcabc"）');
/* ========================================================================== */

var gen = fix('overlap_gen.txt', 9);
var train = fix('repeat.txt', 9);
near('3-gram 重合率 = 4/7', M.overlapRate(gen, train, 3), 4 / 7, 1e-12);
near('8-gram 重合率 = 0', M.overlapRate(gen, train, 8), 0);
near('8-gram 与自身重合率 = 1', M.overlapRate(gen, gen, 8), 1, 1e-12);
var lcs = M.longestCommonSubstring(gen, train);
ok('最长公共子串 = "abcab"', lcs.str === 'abcab', JSON.stringify(lcs.str));
ok('最长公共子串长度 = 5', lcs.len === 5, String(lcs.len));
ok('空串参与时长度 = 0', M.longestCommonSubstring('', 'abc').len === 0);
ok('完全相同时 LCS = 全长', M.longestCommonSubstring('abcd', 'abcd').len === 4);

/* ========================================================================== */
section('【4】未登录字（词表 abc，探针 "abXzc"）');
/* ========================================================================== */

var vTxt = fix('oov_vocab.txt', 6);
var pTxt = fix('oov_probe.txt', 5);
var vOov = kit.buildVocab(vTxt);
ok('词表大小 V = 4（\\0 + a + b + c）', vOov.size === 4, 'V=' + vOov.size);

var oi = M.oovInfo(pTxt, vOov.stoi);
ok('未登录字数 = 2', oi.oov === 2, String(oi.oov));
ok('未登录字符 = [X, z]', JSON.stringify(oi.oovUnique) === JSON.stringify(['X', 'z']),
  JSON.stringify(oi.oovUnique));
ok('模型实际看到 = "abc"', oi.kept === 'abc', JSON.stringify(oi.kept));
ok('总数 = 5', oi.total === 5, String(oi.total));
ok('编码 = [1,2,0,0,3]',
  JSON.stringify(kit.encode(pTxt, vOov.stoi)) === JSON.stringify([1, 2, 0, 0, 3]),
  JSON.stringify(kit.encode(pTxt, vOov.stoi)));
ok('词表内文本 oov = 0', M.oovInfo('abcabc', vOov.stoi).oov === 0);
ok('oovInfo 兼容普通对象词表（不只 Map）',
  M.oovInfo('abXzc', { a: 1, b: 2, c: 3 }).oov === 2);

/* ========================================================================== */
section('【5】评估协议本身');
/* ========================================================================== */

near('countPositions(len=8, ctx=2) = 6', PROTO.countPositions([0, 0, 0, 0, 0, 0, 0, 0], 2), 6);
near('countPositions 太短 → 0', PROTO.countPositions([0, 0, 0], 32), 0);

var test8 = [10, 11, 12, 13, 14, 15, 16, 17];
var seen = [];
var npos = PROTO.forEachPosition(test8, 2, function (hist, target, s, i) {
  seen.push({ s: s, i: i, hist: hist.join(''), target: target });
});
ok('遍历到 6 个位置', npos === 6, String(npos));
ok('位置 (0,0)：上文=[10]，目标=11',
  seen[0].hist === '10' && seen[0].target === 11, JSON.stringify(seen[0]));
ok('位置 (0,1)：上文=[10,11]，目标=12',
  seen[1].hist === '1011' && seen[1].target === 12, JSON.stringify(seen[1]));
ok('位置 (4,1)：上文=[14,15]，目标=16',
  seen[5].hist === '1415' && seen[5].target === 16, JSON.stringify(seen[5]));
ok('最后一个窗口起点 s=4（s=6 时 6+2+1>8 被排除）',
  seen[seen.length - 1].s === 4, String(seen[seen.length - 1].s));

// 窗口遍历与位置遍历必须同源
var wins = [];
PROTO.forEachWindow(test8, 2, function (chunk, tgt, s) {
  wins.push(s + ':' + chunk.join('') + '>' + tgt.join(''));
});
ok('窗口数 = 3', wins.length === 3, JSON.stringify(wins));
ok('窗口 0 = chunk[10,11] → tgt[11,12]', wins[0] === '0:1011>1112', wins[0]);

var sp = PROTO.splitIds([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 0.9);
ok('splitIds(10, 0.9) → train=9, test=1',
  sp.train.length === 9 && sp.test.length === 1,
  sp.train.length + '/' + sp.test.length);

/* ========================================================================== */
section('【6】指标累计器 Acc');
/* ========================================================================== */

var acc = new PROTO.Acc();
acc.add([Math.log(0.1), Math.log(0.7), Math.log(0.2)], 1);
var s1 = acc.summary();
ok('Top-1 命中 → 100%', Math.abs(s1.top1 - 100) < 1e-9, String(s1.top1));
ok('Top-5 命中（总项数 < 5）→ 100%', Math.abs(s1.top5 - 100) < 1e-9, String(s1.top5));
near('NLL = −ln(0.7)', s1.nll, -Math.log(0.7), 1e-12);
near('困惑度 = 1/0.7', s1.ppl, 1 / 0.7, 1e-12);
ok('位置数 = 1', s1.n === 1, String(s1.n));

// 目标排第 6 → 不在 Top-5
var acc2 = new PROTO.Acc();
acc2.add([0.9, 0.8, 0.7, 0.6, 0.5, 0.4].map(Math.log), 5);
var s2 = acc2.summary();
ok('目标排第 6 → Top-5 不命中', Math.abs(s2.top5) < 1e-9, String(s2.top5));
ok('目标排第 6 → Top-1 不命中', Math.abs(s2.top1) < 1e-9, String(s2.top1));

// 平局取 id 最小者（与 minilm.js 的严格大于一致）
var acc3 = new PROTO.Acc();
acc3.add([Math.log(0.5), Math.log(0.5), Math.log(0.1)], 0);
var s3 = acc3.summary();
ok('平局时取 id 最小者 → id 0 命中', Math.abs(s3.top1 - 100) < 1e-9, String(s3.top1));

// 概率为 0 不污染均值，但要记一笔
var acc4 = new PROTO.Acc();
acc4.add([Math.log(0.5), -Infinity], 1);
var s4 = acc4.summary();
ok('目标概率为 0 时记 nInf = 1', s4.nInf === 1, String(s4.nInf));
ok('目标概率为 0 时 NLL 不变成 Infinity', isFinite(s4.nll), String(s4.nll));

/* ========================================================================== */
section('【7】端到端：协议 + 基线串起来能跑');
/* ========================================================================== */

// 用 tiny fixture 自身当"语料"，确认 runBaselines 不炸且三种阶数单调不恶化
var ids = kit.encode(tiny, vTiny.stoi);
var sp2 = PROTO.splitIds(ids, 0.5);            // 4 训练 / 4 测试
var res = BASE.runBaselines(sp2.train, sp2.test, {
  ctx: 2, V: vTiny.size, alphas: [0.1]         // 单 α → 跳过开发集挑选，结果确定
});
ok('runBaselines 返回 3 条', res.length === 3, String(res.length));
ok('三种阶数都有 NLL', res.every(function (r) { return isFinite(r.nll); }),
  JSON.stringify(res.map(function (r) { return r.nll; })));
ok('Top-1 都在 [0,100]',
  res.every(function (r) { return r.top1 >= 0 && r.top1 <= 100; }),
  JSON.stringify(res.map(function (r) { return r.top1; })));
ok('回退占比之和 = 位置数',
  res[2].backoff[1] + res[2].backoff[2] + res[2].backoff[3] === res[2].n,
  JSON.stringify(res[2].backoff) + ' n=' + res[2].n);

/* ========================================================================== */

console.log(logs.join('\n'));
console.log('');
console.log((fail === 0 ? '✓ 全部通过' : '✗ 有失败') + '：' + pass + ' 通过 / ' + fail + ' 失败');
process.exit(fail === 0 ? 0 : 1);
