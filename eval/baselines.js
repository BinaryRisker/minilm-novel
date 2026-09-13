/* ============================================================================
 * eval/baselines.js — 统计基线（字频 / 平滑 bigram / 平滑 trigram）
 * ----------------------------------------------------------------------------
 * 为什么必须有这个：
 *   「留出集 Top-1 = 34%」这个数字本身没有意义。随机猜是 1/V ≈ 0.2%，
 *   34% 是它的 170 倍 —— 听起来很厉害。但一个**只会数相邻字出现多少次**的
 *   bigram 模型可能也有 30%。如果模型打不过 bigram，那 5.5 万参数、
 *   400 步训练、两层 Transformer 就白搭了，它只是把「数频次的活」干得更贵。
 *
 *   所以基线不是陪跑，是**及格线**：模型必须显著超过 bigram / trigram，
 *   「模型学到了结构」这句话才成立。
 *
 * 三条硬约束（做不到就是无效对比）：
 *   1) 只在 trainIds 上拟合；留出集一个字都不许看（除了当上文，见下）
 *   2) 评估位置、可见上文长度、词表 —— 全部走 eval/protocol.js，与模型逐位对齐
 *   3) 加 α 平滑，保证任何字符概率 > 0，NLL 不会崩成 Infinity
 *
 * 用法：
 *   node eval/baselines.js [--corpus scifi] [--ctx 32] [--out eval/baselines_report.md]
 * ============================================================================ */

var PROTO = require('./protocol.js');

/* ------------------------------------------------------------------ */
/* n-gram 模型                                                         */
/* ------------------------------------------------------------------ */

/**
 * 加 α 平滑、带回退的 n-gram 字符语言模型。
 *
 * P(t | c) = (count(c,t) + α) / (count(c) + α·V)      当 count(c) > 0
 * 否则回退到低一阶；一阶（unigram）永远有定义。
 *
 * 注意回退不只是「上下文没见过」时才发生：窗口里前几个位置上文本来就不够长
 * （位置 i 只有 i+1 个上文），此时三角gram 自然退化成 bigram / unigram。
 * 这正好和因果注意力的模型一一对应 —— 位置 0 大家都是瞎猜。
 */
function NGram(trainIds, order, alpha, V) {
  this.order = Math.max(1, order | 0);
  this.alpha = alpha;
  this.V = V;
  this.N = trainIds.length;

  // 一阶：全词表计数（含 0 号未登录位，计数恒为 0，但占一个平滑名额）
  this.uni = new Float64Array(V);
  for (var i = 0; i < trainIds.length; i++) this.uni[trainIds[i]]++;

  // 二阶及以上：context key → { total, counts }
  this.ctxTotal = {};   // order → Map
  this.ctxCount = {};   // order → Map(key → Map(target → count))
  for (var n = 2; n <= this.order; n++) {
    var tot = new Map(), cnt = new Map();
    for (var j = 0; j + n <= trainIds.length; j++) {
      var key = trainIds.slice(j, j + n - 1).join(',');
      var tgt = trainIds[j + n - 1];
      var m = cnt.get(key);
      if (!m) { m = new Map(); cnt.set(key, m); tot.set(key, 0); }
      m.set(tgt, (m.get(tgt) || 0) + 1);
      tot.set(key, tot.get(key) + 1);
    }
    this.ctxCount[n] = cnt;
    this.ctxTotal[n] = tot;
  }
}

/** 单个目标的对数概率（自然对数），带回退 */
NGram.prototype.logProb = function (hist, target) {
  var n = this.order;
  while (n > 1) {
    var need = n - 1;
    if (hist.length >= need) {
      var key = hist.slice(hist.length - need).join(',');
      var tot = this.ctxTotal[n].get(key);
      if (tot !== undefined && tot > 0) {
        var c = this.ctxCount[n].get(key).get(target) || 0;
        return Math.log((c + this.alpha) / (tot + this.alpha * this.V));
      }
    }
    n--;   // 上下文没见过 / 上文不够长 → 退回低一阶
  }
  return Math.log((this.uni[target] + this.alpha) / (this.N + this.alpha * this.V));
};

/** 完整对数概率分布（Float64Array，长度 V） */
NGram.prototype.dist = function (hist) {
  var out = new Float64Array(this.V);
  for (var v = 0; v < this.V; v++) out[v] = this.logProb(hist, v);
  return out;
};

/** 回退到几阶（1=unigram, 2=bigram, 3=trigram）。用于诊断，看基线到底用了多少信息 */
NGram.prototype.usedOrder = function (hist) {
  var n = this.order;
  while (n > 1) {
    var need = n - 1;
    if (hist.length >= need) {
      var tot = this.ctxTotal[n].get(hist.slice(hist.length - need).join(','));
      if (tot !== undefined && tot > 0) return n;
    }
    n--;
  }
  return 1;
};

/* ------------------------------------------------------------------ */
/* α 选择                                                              */
/* ------------------------------------------------------------------ */

/**
 * 用训练集尾部 10% 当开发集挑 α，避免直接在测试集上挑（那是偷看答案）。
 * 返回 { alpha, curve: [{alpha, nll}] }
 */
function selectAlpha(trainIds, order, alphas, V, ctx) {
  var dev = PROTO.splitIds(trainIds, 0.9);
  var curve = [];
  var best = { alpha: alphas[0], nll: Infinity };

  for (var a = 0; a < alphas.length; a++) {
    if (dev.test.length < ctx + 1) break;      // 开发集太短，跳过
    var m = new NGram(dev.train, order, alphas[a], V);
    var acc = new PROTO.Acc();
    PROTO.forEachPosition(dev.test, ctx, function (hist, target) {
      acc.add(m.dist(hist), target);
    });
    var s = acc.summary();
    curve.push({ alpha: alphas[a], nll: s.nll, top1: s.top1 });
    if (s.nll < best.nll) best = { alpha: alphas[a], nll: s.nll };
  }
  return { alpha: best.alpha, curve: curve };
}

/* ------------------------------------------------------------------ */
/* 跑一遍全部基线                                                      */
/* ------------------------------------------------------------------ */

var DEFAULT_ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0];

/**
 * opts: { ctx, V, alphas, orders, selectOnDev }
 * 返回 [{ order, name, alpha, n, top1, top5, nll, ppl, backoff: {...}, curve }]
 */
function runBaselines(trainIds, testIds, opts) {
  opts = opts || {};
  var ctx = opts.ctx || 32;
  var V = opts.V;
  var alphas = opts.alphas || DEFAULT_ALPHAS;
  var orders = opts.orders || [1, 2, 3];
  var out = [];

  for (var oi = 0; oi < orders.length; oi++) {
    var order = orders[oi];

    var alpha, curve = null;
    if (alphas.length === 1) {
      alpha = alphas[0];
    } else {
      var sel = selectAlpha(trainIds, order, alphas, V, ctx);
      alpha = sel.alpha;
      curve = sel.curve;
    }

    var m = new NGram(trainIds, order, alpha, V);
    var acc = new PROTO.Acc();
    var backoff = { 1: 0, 2: 0, 3: 0 };
    var n = PROTO.forEachPosition(testIds, ctx, function (hist, target) {
      acc.add(m.dist(hist), target);
      backoff[m.usedOrder(hist)]++;
    });
    var s = acc.summary();

    out.push({
      order: order,
      name: order === 1 ? 'unigram（字频）' : (order === 2 ? 'bigram（+α平滑）' : 'trigram（+α平滑）'),
      alpha: alpha,
      n: n,
      top1: s.top1,
      top5: s.top5,
      nll: s.nll,
      ppl: s.ppl,
      backoff: backoff,
      curve: curve
    });
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Markdown 表格                                                       */
/* ------------------------------------------------------------------ */

function mdTable(res, modelRow) {
  var L = [];
  L.push('| 方法 | 平滑 α | 留出集 Top-1 | 留出集 Top-5 | 留出集 NLL | 困惑度 |');
  L.push('|---|---|---|---|---|---|');
  if (modelRow) {
    L.push('| **MiniLM（本模型）** | — | **' + modelRow.top1.toFixed(1) + '%** | **' +
      modelRow.top5.toFixed(1) + '%** | **' + modelRow.nll.toFixed(3) + '** | **' +
      modelRow.ppl.toFixed(1) + '** |');
  }
  res.forEach(function (r) {
    L.push('| ' + r.name + ' | ' + r.alpha + ' | ' + r.top1.toFixed(1) + '% | ' +
      r.top5.toFixed(1) + '% | ' + r.nll.toFixed(3) + ' | ' + r.ppl.toFixed(1) + ' |');
  });
  return L.join('\n');
}

/** 回退分布：告诉读者基线在每个位置实际用到了几阶信息 */
function mdBackoff(res) {
  var L = [];
  L.push('| 基线 | 只用字频 | 用到 bigram | 用到 trigram |');
  L.push('|---|---|---|---|');
  res.forEach(function (r) {
    if (r.order === 1) return;
    var n = r.n || 1;
    L.push('| ' + r.name + ' | ' + (100 * r.backoff[1] / n).toFixed(0) + '% | ' +
      (100 * r.backoff[2] / n).toFixed(0) + '% | ' + (100 * r.backoff[3] / n).toFixed(0) + '% |');
  });
  return L.join('\n');
}

function mdAlphaCurve(res) {
  var L = [];
  res.forEach(function (r) {
    if (!r.curve || !r.curve.length) return;
    L.push('**' + r.name + '** α 扫描（在训练集尾部 10% 开发集上选，未在留出集上挑）');
    L.push('');
    L.push('| α | 开发集 NLL | 开发集 Top-1 |');
    L.push('|---|---|---|');
    r.curve.forEach(function (c) {
      L.push('| ' + c.alpha + (c.alpha === r.alpha ? ' **← 选中**' : '') + ' | ' +
        c.nll.toFixed(4) + ' | ' + c.top1.toFixed(1) + '% |');
    });
    L.push('');
  });
  return L.join('\n');
}

/* ------------------------------------------------------------------ */
/* CLI                                                                 */
/* ------------------------------------------------------------------ */

function main() {
  var argv = process.argv.slice(2);
  function argVal(name, dft) {
    var i = argv.indexOf('--' + name);
    return i >= 0 && argv[i + 1] && argv[i + 1][0] !== '-' ? argv[i + 1] : dft;
  }
  var ONLY = argVal('corpus', null);
  var CTX = parseInt(argVal('ctx', '32'), 10);
  var OUT = argVal('out', null);

  var kit = require('../minilm.js');
  var CORPUS = require('../corpus.js');
  var fs = require('fs');
  var path = require('path');

  var keys = ONLY ? [ONLY] : ['scifi', 'wuxia', 'urban'];
  var out = [];
  function w(s) { out.push(s); console.log(s); }

  var head = [
    '# 统计基线对比 · 原始证据',
    '',
    '> 由 `eval/baselines.js` 自动生成。回答一个问题：**模型那个 Top-1 数字，',
    '> 有没有超过「只会数相邻字频次」的统计模型？**',
    '> 生成时间：' + new Date().toISOString().replace('T', ' ').slice(0, 19) + ' (UTC)',
    '',
    '基线只在训练集（前 90%）上拟合；评估位置、每个位置可见的上文长度、词表大小',
    '与模型**逐位对齐**（共用 `eval/protocol.js`）。平滑系数 α 在训练集尾部 10% 的',
    '开发集上挑选，没有在留出集上挑。',
    '',
    '---'
  ].join('\n');
  console.log(head);
  out.push(head);

  keys.forEach(function (k) {
    var text = CORPUS[k].text;
    var vocab = kit.buildVocab(text);
    var ids = kit.encode(text, vocab.stoi);
    var sp = PROTO.splitIds(ids, 0.9);

    var res = runBaselines(sp.train, sp.test, { ctx: CTX, V: vocab.size });

    w('\n## ' + CORPUS[k].name);
    w('');
    w('> ' + CORPUS[k].desc);
    w('');
    w('| 项目 | 值 |');
    w('|---|---|');
    w('| 语料字数 | ' + text.length.toLocaleString() + ' |');
    w('| 训练集 / 留出集 | ' + sp.train.length.toLocaleString() + ' / ' + sp.test.length.toLocaleString() + ' 字 |');
    w('| 词表大小 V | ' + vocab.size + ' |');
    w('| 随机猜测 Top-1 | ' + (100 / vocab.size).toFixed(2) + '% |');
    w('| 评估位置数 | ' + PROTO.countPositions(sp.test, CTX).toLocaleString() + ' |');
    w('');
    w(mdTable(res, null));
    w('');
    w('**基线在每个位置实际用到几阶信息**（受上文长度限制会自动降级）：');
    w('');
    w(mdBackoff(res));
    w('');
    w(mdAlphaCurve(res));
    w('\n---');
  });

  var outFile = OUT ? path.resolve(__dirname, OUT)
    : path.join(__dirname, 'baselines_report.md');
  fs.writeFileSync(outFile, out.join('\n'), 'utf8');
  console.log('\n✓ 已写入 ' + outFile);
}

if (require.main === module) main();

module.exports = {
  NGram: NGram,
  selectAlpha: selectAlpha,
  runBaselines: runBaselines,
  mdTable: mdTable,
  mdBackoff: mdBackoff,
  mdAlphaCurve: mdAlphaCurve,
  DEFAULT_ALPHAS: DEFAULT_ALPHAS
};
