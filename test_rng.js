/* ============================================================================
 * test_rng.js — 随机流隔离与可复现性验证
 * ----------------------------------------------------------------------------
 * 针对 PRD P0 第一条：「初始化、批次和采样共用 Math.random，
 * 增加一次演示采样也会改变后续训练批次，无法公平做实验。」
 *
 * 验证四件事：
 *   1. 同种子 → 训练轨迹逐位相同（可复现）
 *   2. 不同种子 → 结果确实不同（种子真的起作用）
 *   3. 中途插入任意次生成 → 训练批次序列完全不变（流隔离）
 *   4. rngState 保存/恢复 → 随机流可续跑
 *
 * 用法：node test_rng.js
 * ============================================================================ */
'use strict';

var kit = require('./minilm.js');
var CORPUS = require('./corpus.js');

var pass = 0, fail = 0;
function ok(cond, name, extra) {
  if (cond) { pass++; console.log('  \u2713 ' + name + (extra ? '  \u2014 ' + extra : '')); }
  else { fail++; console.log('  \u2717 ' + name + (extra ? '  \u2014 ' + extra : '')); }
}

var CFG = { ctx: 32, d: 40, layers: 2, heads: 4, ffMul: 4 };

// 跑一小段训练。genAt：在第几步后插入 3 次生成（null = 不插入）
function run(key, seed, steps, genAt) {
  var text = CORPUS[key].text;
  var vocab = kit.buildVocab(text);
  var ids = kit.encode(text, vocab.stoi);
  var m = new kit.MiniLM(vocab.size, CFG, { seed: seed });
  var batchSig = [], losses = [];

  for (var s = 0; s < steps; s++) {
    var b = kit.makeBatch(ids, 8, CFG.ctx, m.rng.batch);
    // 用批次前 16 个 token 做指纹
    batchSig.push(Array.prototype.slice.call(b.tokens, 0, 16).join(','));
    m.zeroGrad();
    var r = m.forward(b.tokens, b.targets);
    m.backward();
    m.clipGrad(1.0);
    m.step(0.004, s + 1);
    losses.push(r.loss);

    if (genAt != null && s === genAt) {
      // 中途插三次生成，模拟界面上的"训练快照"
      m.generate(ids.slice(0, 8), { maxNew: 12, temp: 0.8 });
      m.generate(ids.slice(0, 8), { maxNew: 12, temp: 0.9 });
      m.generate(ids.slice(0, 8), { maxNew: 12, temp: 1.0 });
    }
  }
  return { batchSig: batchSig, losses: losses, seed: m.seed, state: m.rngState(), model: m };
}

console.log('='.repeat(70));
console.log('随机流隔离与可复现性验证');
console.log('='.repeat(70));
console.log();

/* ---- 1. 同种子完全可复现 ---- */
console.log('1. 可复现性');
var a = run('scifi', 42, 12, null);
var b = run('scifi', 42, 12, null);
ok(a.seed === 42, '显式种子被记录', 'seed=' + a.seed);
ok(a.losses.join('|') === b.losses.join('|'),
   '同种子两次训练，损失序列逐位相同',
   '末位 loss=' + a.losses[a.losses.length - 1].toFixed(6));
ok(a.batchSig.join('|') === b.batchSig.join('|'),
   '同种子两次训练，批次序列完全相同');

/* ---- 2. 不同种子确实不同 ---- */
console.log('\n2. 种子有效性');
var c = run('scifi', 7, 12, null);
ok(c.losses.join('|') !== a.losses.join('|'),
   '换种子后训练轨迹不同',
   'seed42=' + a.losses[11].toFixed(4) + '  seed7=' + c.losses[11].toFixed(4));

/* ---- 3. 生成不影响训练批次（PRD 的核心诉求）---- */
console.log('\n3. 流隔离：中途生成不影响训练批次');
var d = run('scifi', 42, 12, 4);   // 第 4 步后插 3 次生成
ok(d.batchSig.join('|') === a.batchSig.join('|'),
   '插入 3 次生成后，批次序列仍与不生成时完全一致');
ok(d.losses.join('|') === a.losses.join('|'),
   '插入 3 次生成后，损失序列完全一致',
   '末位 loss=' + d.losses[11].toFixed(6));

/* ---- 4. 三条流互不干扰 ---- */
console.log('\n4. 三条流互相独立');
var m1 = new kit.MiniLM(100, CFG, { seed: 99 });
var m2 = new kit.MiniLM(100, CFG, { seed: 99 });
for (var i = 0; i < 500; i++) m1.rng.sample();      // 狂消耗采样流
for (var j = 0; j < 500; j++) m1.rng.init();        // 狂消耗初始化流
var seqA = [m1.rng.batch(), m1.rng.batch(), m1.rng.batch()];
var seqB = [m2.rng.batch(), m2.rng.batch(), m2.rng.batch()];
ok(seqA.join('|') === seqB.join('|'),
   '消耗 1000 次采样/初始化后，批次流数值不变',
   seqA.map(function (x) { return x.toFixed(6); }).join(', '));

/* ---- 5. 状态保存与恢复 ---- */
console.log('\n5. 状态保存与恢复');
var m3 = new kit.MiniLM(100, CFG, { seed: 2026 });
var st = m3.rngState();
var after1 = [m3.rng.batch(), m3.rng.batch()];
var m4 = new kit.MiniLM(100, CFG, { seed: 2026 });
m4.setRngState(st);
var after2 = [m4.rng.batch(), m4.rng.batch()];
ok(after1.join('|') === after2.join('|'),
   'rngState 恢复后批次流接续一致',
   after1.map(function (x) { return x.toFixed(6); }).join(', '));
ok(typeof st.seed === 'number' && typeof st.batch === 'number',
   'rngState 可序列化（纯数字字段）',
   JSON.stringify(st));

/* ---- 6. 旧调用方式仍可用（向后兼容）---- */
console.log('\n6. 向后兼容');
var mOld = new kit.MiniLM(100, CFG);              // 不传 seed
ok(typeof mOld.seed === 'number' && mOld.seed > 0,
   '不传种子时自动取随机种子并记录', 'seed=' + mOld.seed);
var ids2 = kit.encode(CORPUS.scifi.text, kit.buildVocab(CORPUS.scifi.text).stoi);
ok(kit.makeBatch(ids2, 4, CFG.ctx) !== null,
   'makeBatch 不传 rng 时仍可用（回落到 Math.random）');

console.log('\n' + '='.repeat(70));
console.log(fail === 0
  ? '全部通过：' + pass + ' / ' + (pass + fail)
  : '有 ' + fail + ' 项失败：' + pass + ' / ' + (pass + fail));
console.log('='.repeat(70));
process.exit(fail === 0 ? 0 : 1);
