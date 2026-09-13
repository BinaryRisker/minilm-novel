/* ============================================================================
 * js_seed_run.js — 给 minilm.js 侧做"多种子"运行（L0 对齐验证的另一半）
 * ----------------------------------------------------------------------------
 * 背景：早期 minilm.js 用 Math.random（不可播种），JS 参考值只是 JS 分布里的
 * 一个抽样。现在 minilm.js 原生支持种子（三条独立流 init / batch / sample），
 * 本脚本改用官方 API：new MiniLM(V, CFG, { seed }) + makeBatch(..., m.rng.batch)。
 *
 * 【协议变更 2026-09-13 17:55】此前用 "Math.random = mulberry32(seed)" 猴子补丁
 * 跑出的 js_*.json 与新协议不等价（init 与 batch 的派生路径不同），需按新协议重跑。
 *
 * 训练与评测逻辑逐行复制自 test_novel.js 的 trainCorpus（仅保留损失快照
 * 与留出集评测，砍掉生成部分），保证协议完全一致。
 *
 * 用法：node py/js_seed_run.js <scifi|wuxia|urban> <seed>
 * 输出：py/runs/js_<key>_s<seed>_n400.json
 * ============================================================================ */
'use strict';
const fs = require('fs');
const path = require('path');

// 种子由 minilm.js 原生支持（init / batch / sample 三条独立流），
// 不再替换全局 Math.random —— 那样会让三条流共用同一条，破坏隔离性。

const kit = require('../minilm.js');
const CORPUS = require('../corpus.js');

const key = process.argv[2];
const seed = parseInt(process.argv[3], 10);
const CFG = { ctx: 32, d: 40, layers: 2, heads: 4, ffMul: 4 };
const BATCH = 8, LR = 0.004, STEPS = 400;

(function main() {
  const corpus = CORPUS[key];
  const vocab = kit.buildVocab(corpus.text);
  const ids = kit.encode(corpus.text, vocab.stoi);
  const split = Math.floor(ids.length * 0.9);
  const trainIds = ids.slice(0, split);
  const testIds = ids.slice(split);

  const model = new kit.MiniLM(vocab.size, CFG, { seed: seed });
  const warmup = Math.max(10, Math.floor(STEPS * 0.15));

  const t0 = Date.now();
  const snapEvery = Math.max(1, Math.floor(STEPS / 4));
  const rows = [];
  let lossAcc = 0, lossN = 0;

  for (let step = 0; step < STEPS; step++) {
    const batch = kit.makeBatch(trainIds, BATCH, CFG.ctx, model.rng.batch);
    model.zeroGrad();
    const r = model.forward(batch.tokens, batch.targets);
    model.backward();
    model.clipGrad(1.0);
    const curLr = step < warmup ? LR * (step + 1) / warmup
      : LR * 0.5 * (1 + Math.cos(Math.PI * (step - warmup) / Math.max(1, STEPS - warmup)));
    model.step(curLr, step + 1);
    lossAcc += r.loss; lossN++;
    if ((step + 1) % snapEvery === 0 || step === 0) {
      rows.push({ step: step + 1, loss: lossAcc / lossN });
      lossAcc = 0; lossN = 0;
    }
  }

  // 留出集评测——与 test_novel.js 逐行一致
  let top1 = 0, top5 = 0, total = 0;
  for (let s = 0; s + CFG.ctx + 1 <= testIds.length; s += CFG.ctx) {
    const chunk = testIds.slice(s, s + CFG.ctx);
    const tgt = testIds.slice(s + 1, s + 1 + CFG.ctx);
    model.forward(chunk, tgt, CFG.ctx);
    const logits = model.cache.logits;
    for (let i = 0; i < CFG.ctx; i++) {
      const off = i * vocab.size;
      const arr = [];
      let best = -Infinity, bi = 0;
      for (let v = 0; v < vocab.size; v++) {
        const lg = logits[off + v];
        arr.push({ v, l: lg });
        if (lg > best) { best = lg; bi = v; }
      }
      total++;
      if (bi === tgt[i]) top1++;
      arr.sort((a, b) => b.l - a.l);
      for (let k = 0; k < 5; k++) if (arr[k].v === tgt[i]) { top5++; break; }
    }
  }

  const out = {
    key, seed, side: 'js',
    losses: rows.map(r => +r.loss.toFixed(4)),
    steps_idx: rows.map(r => r.step),
    top1: +(100 * top1 / total).toFixed(2),
    top5: +(100 * top5 / total).toFixed(2),
    n: total,
    final_loss: rows[rows.length - 1].loss,
    secs: +((Date.now() - t0) / 1000).toFixed(1),
  };
  const outDir = path.join(__dirname, 'runs');
  fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `js_${key}_s${seed}_n400.json`);
  fs.writeFileSync(outFile, JSON.stringify(out, null, 1));
  console.log(`${key} seed=${seed} top1=${out.top1}% top5=${out.top5}% loss=${out.final_loss} (${out.secs}s)`);
})();
