/* 基准测试：测 JS 手写 Transformer 的真实算力利用率
   目的：用硬数字回答"浏览器 JS 到底能不能训练大模型" */
var kit = require('../minilm.js');
var CORPUS = require('../corpus.js');

function benchOne(label, text, vocabSize, cfg, B, steps) {
  var vocab = kit.buildVocab(text);
  var m = new kit.MiniLM(vocab.size, cfg);
  var ids = kit.encode(text, vocab.stoi);
  var batch = kit.makeBatch(ids, B, cfg.ctx);   // 返回扁平 {tokens, targets}，一次前向处理整个 batch

  var params = 0;
  m.paramList.forEach(function (p) { params += p.n; });

  // 预热
  for (var w = 0; w < 3; w++) { m.forward(batch.tokens, batch.targets); m.backward(); }

  var t0 = Date.now();
  for (var s = 0; s < steps; s++) {
    m.forward(batch.tokens, batch.targets);
    m.backward();
  }
  var dt = (Date.now() - t0) / 1000;

  var tokensPerStep = B * cfg.ctx;
  var flops = 6 * params * tokensPerStep * steps;   // 前向 2N + 反向 4N
  var gf = flops / dt / 1e9;

  console.log('  ' + label);
  console.log('    参数 ' + params.toLocaleString() + ' | 词表 ' + vocab.size +
              ' | ctx ' + cfg.ctx + ' d ' + cfg.d + ' L ' + cfg.layers);
  console.log('    batch ' + B + ' = ' + tokensPerStep + ' token/step');
  console.log('    ' + steps + ' 步耗时 ' + dt.toFixed(2) + 's  →  ' +
              (dt * 1000 / steps).toFixed(1) + ' ms/step');
  console.log('    折算算力 ~' + gf.toFixed(2) + ' GFLOPS  |  吞吐 ~' +
              (tokensPerStep * steps / dt).toFixed(0) + ' tok/s');
  console.log('');
  return { params: params, gf: gf, ms: dt * 1000 / steps, tokps: tokensPerStep * steps / dt };
}

console.log('='.repeat(72));
console.log('JS 手写 Transformer 基准（Node V8，单线程，纯 JS 无 SIMD）');
console.log('='.repeat(72));
console.log('');

var r1 = benchOne('① 当前默认（教学规模，网页默认）',
  CORPUS.scifi.text, 4000, { ctx: 32, d: 40, layers: 2, heads: 4, ffMul: 4 }, 8, 40);

var r2 = benchOne('② 放大到 d=192 / 4 层（约 150 万参数）',
  CORPUS.scifi.text, 4000, { ctx: 64, d: 192, layers: 4, heads: 6, ffMul: 4 }, 4, 8);

console.log('='.repeat(72));
console.log('外推：这套 JS 代码能不能训大模型');
console.log('='.repeat(72));
console.log('');
var bestGf = Math.max(r1.gf, r2.gf);
console.log('取实测最好利用率 ' + bestGf.toFixed(2) + ' GFLOPS，外推到 1B 参数模型：');
console.log('');
var target = 1e9;
[['小规模预训练', 2e9], ['可用级预训练', 2e10]].forEach(function (pair) {
  var label = pair[0], tokens = pair[1];
  var flops = 6 * target * tokens;
  var secJs = flops / (bestGf * 1e9);
  var secGpu = flops / (100e12);
  console.log('  【' + label + '】10 亿参数 + ' + (tokens / 1e9) + ' 亿 token');
  console.log('    总计算量 ≈ ' + (flops / 1e18).toFixed(1) + ' ×10^18 FLOPs');
  console.log('    浏览器 JS   ≈ ' + (secJs / 86400).toFixed(0) + ' 天  (' +
              (secJs / 86400 / 365).toFixed(1) + ' 年)');
  console.log('    RTX 3090    ≈ ' + (secGpu / 3600).toFixed(1) + ' 小时');
  console.log('    差距        ≈ ' + Math.round(secJs / secGpu).toLocaleString() + ' 倍');
  console.log('');
});

// 显存/内存账
console.log('='.repeat(72));
console.log('显存账：1B 参数训练到底要多少内存');
console.log('='.repeat(72));
console.log('');
console.log('  权重 fp16            ' + (1e9 * 2 / 1e9).toFixed(1) + ' GB');
console.log('  梯度 fp16            ' + (1e9 * 2 / 1e9).toFixed(1) + ' GB');
console.log('  Adam 一阶动量 fp32   ' + (1e9 * 4 / 1e9).toFixed(1) + ' GB');
console.log('  Adam 二阶动量 fp32   ' + (1e9 * 4 / 1e9).toFixed(1) + ' GB');
console.log('  ---------------------------------');
console.log('  小计（不含激活）      ' + (1e9 * 12 / 1e9).toFixed(1) + ' GB');
console.log('  + 激活 / 中间张量     ≈ 2~4 GB（视 batch 与 ctx）');
console.log('  = 至少 14~16 GB 显存');
console.log('');
console.log('  浏览器现状：单标签页可用 GPU 显存通常 2~6 GB，内存上限约 4 GB。');
console.log('  结论：连放都放不下，更谈不上训练。');
console.log('');
