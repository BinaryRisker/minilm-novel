/* ============================================================================
 * smoke.js — 对构建产物 index.html 做端到端冒烟测试
 * ----------------------------------------------------------------------------
 * 做四件事：
 *   1) 确认 index.html 里没有任何外部依赖（script src / link href）
 *   2) 把内联的库脚本抽出来，在 Node vm 里执行，确认 __makeMiniLM 工厂可用
 *   3) 语法检查 UI 脚本（有语法错误的话整个页面就是白屏）
 *   4) 用 UI 默认配置真跑一遍：训练 → 生成 → 注意力矩阵 → 梯度检查
 *
 *   UI 里用到的每一个 MiniLMKit API 都在这里被调用一次，
 *   这样"页面能打开但一点就报错"这类问题能在 Node 里提前抓到。
 *
 * 用法：node smoke.js
 * ============================================================================ */

var fs = require('fs');
var path = require('path');
var vm = require('vm');

var dir = __dirname;
var html = fs.readFileSync(path.join(dir, 'index.html'), 'utf8');
var fails = 0, passes = 0;

function ok(cond, label, extra) {
  if (cond) { passes++; console.log('  ✓ ' + label + (extra ? '  ' + extra : '')); }
  else { fails++; console.log('  ✗ ' + label + (extra ? '  ' + extra : '')); }
}
function head(t) { console.log('\n' + t); console.log('-'.repeat(60)); }

/* ------------------------------------------------------------------ */
head('1 · 依赖自检（单文件必须零外部依赖）');

var extScripts = html.match(/<script[^>]+src=/g) || [];
var extLinks = html.match(/<link[^>]+href=/g) || [];
ok(extScripts.length === 0, '没有外链 <script src>', '(' + extScripts.length + ')');
ok(extLinks.length === 0, '没有外链 <link href>', '(' + extLinks.length + ')');
ok(/__makeMiniLM/.test(html), '内联了 minilm 工厂函数');
ok(/global\.CORPUS = CORPUS|window\.CORPUS/.test(html) || /var CORPUS = \{\}/.test(html), '内联了 corpus.js');
ok(/MiniLMKitF64/.test(html), '同时提供 float32 与 float64 两份实例入口');

/* ------------------------------------------------------------------ */
head('2 · 抽出内联脚本并在沙箱中执行');

var blocks = [];
var re = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g;
var m;
while ((m = re.exec(html)) !== null) blocks.push(m[1]);
ok(blocks.length >= 2, '找到内联脚本块', '(' + blocks.length + ' 块)');

var sandbox = { console: console, Math: Math, Date: Date, performance: { now: Date.now } };
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

var libOK = true, err = null;
try {
  vm.runInContext(blocks[0], sandbox, { filename: 'inline-lib.js', timeout: 20000 });
} catch (e) { libOK = false; err = e; }
ok(libOK, '库脚本执行无异常', err ? String(err.message) : '');

var K = sandbox.window.MiniLMKit;
var K64 = sandbox.window.MiniLMKitF64;
var CORPUS = sandbox.window.CORPUS;

ok(!!K, 'window.MiniLMKit 存在');
ok(!!K64, 'window.MiniLMKitF64 存在');
ok(!!CORPUS && !!CORPUS.scifi && !!CORPUS.wuxia && !!CORPUS.urban, 'window.CORPUS 含三段语料');
ok(K !== K64, '两份实例是不同对象（精度开关生效）');

/* ------------------------------------------------------------------ */
head('3 · UI 脚本语法检查');

// 最后一块内联脚本就是 UI 逻辑
var uiSrc = blocks[blocks.length - 1];
var uiOK = true, uiErr = null;
try { new vm.Script(uiSrc, { filename: 'inline-ui.js' }); }
catch (e) { uiOK = false; uiErr = e; }
ok(uiOK, 'UI 脚本语法正确', uiErr ? String(uiErr.message) : '');

// UI 里引用的每个 DOM id 是否真实存在于 HTML 中
var ids = [];
var idRe = /\$\('([A-Za-z0-9_]+)'\)/g;
while ((m = idRe.exec(uiSrc)) !== null) ids.push(m[1]);
var uniqIds = Array.from(new Set(ids));
var missing = uniqIds.filter(function (id) { return html.indexOf('id="' + id + '"') < 0; });
ok(missing.length === 0, 'UI 引用的 ' + uniqIds.length + ' 个 DOM id 全部存在',
   missing.length ? '缺失: ' + missing.join(', ') : '');

/* ------------------------------------------------------------------ */
head('4 · 实跑：训练 → 生成 → 注意力 → 梯度检查（UI 默认配置）');

var text = CORPUS.scifi.text;
var vocab = K.buildVocab(text);
var ids2 = K.encode(text, vocab.stoi);
var cfg = { ctx: 32, d: 40, layers: 2, heads: 4, ffMul: 4 };
var model = new K.MiniLM(vocab.size, cfg);

ok(vocab.size === 365, '词表大小符合预期', '= ' + vocab.size);
ok(model.paramCount() === 55400, '参数量符合预期', '= ' + model.paramCount().toLocaleString());

var t0 = Date.now();
var B = 8, STEPS = 30, lr = 0.004, warmup = Math.max(10, Math.floor(STEPS * 0.15));
var firstLoss = null, lastLoss = null;
for (var s = 0; s < STEPS; s++) {
  var batch = K.makeBatch(ids2, B, cfg.ctx);
  model.zeroGrad();
  var r = model.forward(batch.tokens, batch.targets);
  model.backward();
  model.clipGrad(1.0);
  var curLr = s < warmup ? lr * (s + 1) / warmup : lr * 0.5 * (1 + Math.cos(Math.PI * (s - warmup) / (STEPS - warmup)));
  model.step(curLr, s + 1);
  if (s === 0) firstLoss = r.loss;
  lastLoss = r.loss;
}
var dt = (Date.now() - t0) / 1000;
ok(lastLoss < firstLoss, '30 步后损失下降', firstLoss.toFixed(3) + ' → ' + lastLoss.toFixed(3));
console.log('    ' + (STEPS / dt).toFixed(1) + ' step/s（Node 单线程，浏览器同引擎）');

// 生成（UI 用 generate 的返回结构：ids / trace[].candidates）
var g = model.generate(K.encode('新长安的雨下了', vocab.stoi),
  { maxNew: 40, temp: 0.8, topk: 8, repPenalty: 1.15 });
var genText = K.decode(g.ids, vocab.itos);
ok(g.ids.length === 7 + 40, '生成长度正确', '= ' + g.ids.length);
ok(g.trace.length === 40 && g.trace[0].candidates.length > 0, 'trace 候选分布可用',
   '首个候选概率 ' + (g.trace[0].candidates[0].p * 100).toFixed(1) + '%');
ok(genText.indexOf('新长安的雨下了') === 0, '生成文本以提示词开头');
console.log('    样例: ' + genText.replace(/\n/g, ' ').slice(0, 60));

// 注意力矩阵（UI 热力图用 getAttention(layer, head)）
var probe = K.encode('新长安的雨下了，他说', vocab.stoi);
model.forward(probe, null, probe.length);
var T = model.cache.T;
ok(T === probe.length, 'forward 的 TOverride 生效', 'T=' + T);
var badDim = null, sumBad = null;
for (var l = 0; l < cfg.layers; l++) {
  for (var h = 0; h < cfg.heads; h++) {
    var A = model.getAttention(l, h);
    if (!A || A.length !== T * T) { badDim = 'L' + l + 'H' + h; break; }
    // 因果性 + 每行和为 1
    for (var i = 0; i < T; i++) {
      var rowSum = 0;
      for (var j = 0; j < T; j++) {
        var v = A[i * T + j];
        if (j > i && v !== 0) { sumBad = 'L' + l + 'H' + h + ' 上三角非零'; }
        rowSum += v;
      }
      if (Math.abs(rowSum - 1) > 1e-3) sumBad = 'L' + l + 'H' + h + ' 第' + i + '行和=' + rowSum.toFixed(4);
    }
  }
}
ok(!badDim, '全部 ' + (cfg.layers * cfg.heads) + ' 组注意力矩阵尺寸正确', badDim || '');
ok(!sumBad, '注意力满足因果掩码且每行归一化', sumBad || '');

// 梯度检查（UI 的"梯度检查"按钮走这条路径，用 K64）
var gtext = '从前有座山，山里有座庙，庙里有个老和尚，老和尚对小和尚说。';
var gv = K64.buildVocab(gtext);
var gm = new K64.MiniLM(gv.size, { ctx: 8, d: 8, layers: 1, heads: 2, ffMul: 2 });
var gids = K64.encode(gtext, gv.stoi);
var GT = 8, GB = 2;
var tk = new Int32Array(GB * GT), tg = new Int32Array(GB * GT);
for (var q = 0; q < GB * GT; q++) { tk[q] = gids[q % gids.length]; tg[q] = gids[(q + 1) % gids.length]; }
gm.zeroGrad();
gm.forward(tk, tg);
gm.backward();

var names = ['emb', 'pos', 'Wq0', 'Wo0', 'Wv0', 'W1_0', 'W2_0', 'ln1g0', 'ln2b0', 'bq0', 'lnfg', 'b2_0'];
var eps = 1e-5, maxRel = 0, worst = '', checked = 0, passed = 0;
var sigChecked = 0, sigPassed = 0;
names.forEach(function (name) {
  if (!gm.P[name]) return;
  var P = gm.P[name], G = gm.G[name], n = P.length;
  for (var idx = 0; idx < n; idx++) {
    var orig = P[idx];
    P[idx] = orig + eps; var lp = gm.forward(tk, tg).loss;
    P[idx] = orig - eps; var lm = gm.forward(tk, tg).loss;
    P[idx] = orig;
    var num = (lp - lm) / (2 * eps), ana = G[idx];
    var diff = Math.abs(num - ana);
    var scale = Math.max(Math.abs(num), Math.abs(ana));
    if (scale < 1e-7) continue;
    checked++;
    var okThis = diff <= 1e-5 + 1e-4 * scale;
    if (okThis) passed++;
    if (scale > 1e-4) {
      sigChecked++;
      if (okThis) sigPassed++;
      var rel = diff / scale;
      if (rel > maxRel) { maxRel = rel; worst = name + '[' + idx + ']'; }
    }
  }
});
ok(passed === checked && sigPassed === sigChecked && checked > 0, '梯度检查全部通过',
   passed + '/' + checked + '，显著分量 ' + sigPassed + '/' + sigChecked +
   '，最大相对误差 ' + maxRel.toExponential(2) + '（' + worst + '）');

/* ------------------------------------------------------------------ */
console.log('\n' + '='.repeat(60));
console.log(fails === 0
  ? '✓ 冒烟测试全部通过（' + passes + ' 项）—— index.html 可以正常打开使用'
  : '✗ 有 ' + fails + ' 项失败 / ' + passes + ' 项通过');
console.log('='.repeat(60));
process.exit(fails === 0 ? 0 : 1);
