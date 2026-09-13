/* ============================================================================
 * dev/export_fixture.js — 把 minilm.js 在一个小配置上的一次
 *         训练步（前向 + 反向 + Adam 步）整体落到 JSON，
 *         给 py/verify_parity.py 做逐位对齐用。
 * ----------------------------------------------------------------------------
 * 为什么是「导出 JS 的初始权重、再让 PyTorch 加载」而不是「同种子同时初始化」：
 *   两个实现的 PRNG 不同（mulberry32 vs PyTorch MT19937），强对齐它们的
 *   初始权重没意义 —— 我们要验证的是「前向/反向/Adam 步的数学」对不对。
 *   权重从 JS 一次性导出，PyTorch 拿到后原样加载，再各自跑一步：
 *   这就只比对 forward / backward / step 的数学，不掺杂 RNG 差异。
 *
 * 体积控制：ctx=4 d=8 layers=1 heads=2 ffMul=2 + V=9 → 约 720 参数，JSON 数十 KB
 * ============================================================================ */
global.MINILM_USE_F64 = true;   // 切到 float64，与 PyTorch 默认对齐

var fs = require('fs');
var path = require('path');
var kit = require('../minilm.js');

var CFG = { ctx: 4, d: 8, layers: 1, heads: 2, ffMul: 2 };
var TEXT = 'abc12345';   // 8 个唯一字符（ASCII，跨平台无编码问题）
var SEED = 1234;          // 任何固定种子都行，权重会被整体导出
var LR = 0.001;
var T_STEP = 1;           // Adam 的 t 从 1 开始（与 minilm.js step(lr, t) 一致）

// 固定的输入：vocab('abc12345') = ['', 1,2,3,4,5,a,b,c]（'0' < '1' < ... < 'a'）
//              encode('abc12345abc12345') = [6,7,8,1,2,3,4,5, 2,3,4,5,6,7,8,1]
//              取前 8 个做 B=2 T=4：rows = [[6,7,8,1],[2,3,4,5]]
//              targets 右移一位：      rows = [[7,8,1,2],[3,4,5,6]]
var vocab = kit.buildVocab(TEXT);
var ids = kit.encode('abc12345abc12345', vocab.stoi);
var TOKENS = ids.slice(0, 8);
var TARGETS = ids.slice(1, 9);

var model = new kit.MiniLM(vocab.size, CFG, { seed: SEED });

function dumpWeights(m) {
  var out = {};
  for (var i = 0; i < m.paramList.length; i++) {
    var p = m.paramList[i];
    out[p.name] = Array.prototype.slice.call(m.P[p.name]);
  }
  return out;
}
function dumpGrads(m) {
  var out = {};
  for (var i = 0; i < m.paramList.length; i++) {
    var p = m.paramList[i];
    out[p.name] = Array.prototype.slice.call(m.G[p.name]);
  }
  return out;
}

var t0 = Date.now();

// ---- 1. 初始权重 ----
var initWeights = dumpWeights(model);

// ---- 2. 前向（同时得到 loss / logits / probs）----
var fr = model.forward(TOKENS, TARGETS, CFG.ctx);
var logits = Array.prototype.slice.call(model.cache.logits);
var probs = Array.prototype.slice.call(model.cache.probs);
var loss = fr.loss;

// ---- 3. 反向（得到原始梯度，裁剪前）----
model.backward();
var gradsRaw = dumpGrads(model);

// ---- 4. 梯度裁剪（与 PyTorch 的 clip_grad_norm_(1.0) 同语义）----
model.clipGrad(1.0);
var gradsClipped = dumpGrads(model);

// ---- 5. Adam 步 ----
model.step(LR, T_STEP);
var stepWeights = dumpWeights(model);

var fixture = {
  meta: {
    js: 'minilm.js + export_fixture.js',
    note: '初始权重从 JS 导出，PyTorch 原样加载后跑同一前向/反向/步。' +
          '目的是比对数学，不比对 RNG。',
    cfg: CFG,
    vocab_chars: TEXT,
    seed: SEED,
    lr: LR,
    t: T_STEP,
    B: 2, T: 4,
    batch_tokens: TOKENS,
    batch_targets: TARGETS,
    param_count: model.paramCount()
  },
  init_weights: initWeights,
  forward: { logits: logits, probs: probs, loss: loss },
  backward: { raw: gradsRaw, clipped: gradsClipped, max_norm: 1.0 },
  step_weights: stepWeights
};

var outPath = path.join(__dirname, '..', 'py', 'runs', 'parity_fixture.json');
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(fixture));
console.log('wrote ' + outPath + '  (' + (JSON.stringify(fixture).length) + ' bytes,  ' +
  (Date.now() - t0) + 'ms)');
console.log('param count: ' + model.paramCount() +
  '   loss: ' + loss.toFixed(10));
