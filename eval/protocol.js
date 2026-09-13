/* ============================================================================
 * eval/protocol.js — 评估协议的唯一定义处
 * ----------------------------------------------------------------------------
 * 为什么要有这个文件：
 *   原来「模型」和「基线」各写一遍切分和遍历，只要有一边多看到上文、或多算一个
 *   位置，两边的数字就不可比 —— 而这类偏差非常安静，不会报错，只会让结论悄悄错。
 *   所以这里把「切分 / 评估哪些位置 / 每个位置能看到多少上文 / 怎么累计指标」
 *   定义成一份，模型和基线都必须调用它。
 *
 * 协议（与 test_novel.js 的历史行为严格一致，不改变任何已产出数字）：
 *   1) 切分   split = floor(len * 0.9)；train = ids[:split]，test = ids[split:]
 *   2) 窗口   for (s = 0; s + ctx + 1 <= test.length; s += ctx)
 *   3) 位置   位置 (s, i) 上：
 *             上文 = test[s .. s+i]（因果注意力，看不到 i 之后，也看不到窗口之前）
 *             目标 = test[s+1+i]
 *   4) 指标   Top-1 / Top-5 / NLL（自然对数，与模型交叉熵同量纲）
 *
 * 第 3 条是整套基线的关键约束：n-gram 也必须只拿到这些字符，不许往前多借一个字。
 * 否则三角gram在窗口第一个位置就能偷看窗口外的真实上文，等于开了天眼。
 * ============================================================================ */
(function (root) {
  'use strict';

  /* ------------------------------------------------------------------ */
  /* 切分                                                                */
  /* ------------------------------------------------------------------ */

  function splitIds(ids, trainRatio) {
    var r = trainRatio == null ? 0.9 : trainRatio;
    var split = Math.floor(ids.length * r);
    return { train: ids.slice(0, split), test: ids.slice(split), split: split };
  }

  /* ------------------------------------------------------------------ */
  /* 位置遍历                                                            */
  /* ------------------------------------------------------------------ */

  /**
   * 遍历每个评估窗口。fn(chunk, tgt, s)。
   * 模型一次 forward 就能算完整个窗口，所以给模型用这个（省 32 倍 forward）。
   * forEachPosition 建在它之上 —— 两者同源，位置集合不可能对不上。
   */
  function forEachWindow(testIds, ctx, fn) {
    var n = 0;
    for (var s = 0; s + ctx + 1 <= testIds.length; s += ctx) {
      fn(testIds.slice(s, s + ctx), testIds.slice(s + 1, s + 1 + ctx), s);
      n++;
    }
    return n;
  }

  /**
   * 遍历模型真正被评估的每一个位置。
   * fn(hist, target, s, i)：hist 是该位置可见上文（数组），target 是待预测 id。
   * 返回位置总数。
   */
  function forEachPosition(testIds, ctx, fn) {
    var n = 0;
    forEachWindow(testIds, ctx, function (chunk, tgt, s) {
      for (var i = 0; i < ctx; i++) {
        fn(chunk.slice(0, i + 1), tgt[i], s, i);
        n++;
      }
    });
    return n;
  }

  /** 位置总数（不遍历，用于预估 / 断言） */
  function countPositions(testIds, ctx) {
    if (testIds.length < ctx + 1) return 0;
    var windows = Math.floor((testIds.length - ctx - 1) / ctx) + 1;
    return windows * ctx;
  }

  /* ------------------------------------------------------------------ */
  /* 指标累计                                                            */
  /* ------------------------------------------------------------------ */

  /**
   * 累计器。喂进来的是**对数概率**（自然对数），避免下溢，也和 loss 同量纲。
   * Top-1 / Top-5 的平局处理与 minilm.js 一致：严格大于才更新，即平局取 id 最小者。
   */
  function Acc() {
    this.n = 0;
    this.top1 = 0;
    this.top5 = 0;
    this.nll = 0;        // 累计 -log p(target)，自然对数
    this.nInf = 0;       // 目标概率为 0（-Infinity）的位置数，正常应为 0
  }

  Acc.prototype.add = function (logprobs, target) {
    var V = logprobs.length;
    var best = -Infinity, bi = 0;
    var top = [];   // 最多 5 个，按 lp 降序

    for (var v = 0; v < V; v++) {
      var lp = logprobs[v];
      if (lp > best) { best = lp; bi = v; }
      if (top.length < 5) {
        top.push({ v: v, lp: lp });
        top.sort(cmpLp);
      } else if (lp > top[4].lp) {
        top[4] = { v: v, lp: lp };
        top.sort(cmpLp);
      }
    }

    this.n++;
    if (bi === target) this.top1++;
    for (var k = 0; k < top.length; k++) {
      if (top[k].v === target) { this.top5++; break; }
    }

    var lpT = logprobs[target];
    if (lpT === -Infinity || lpT !== lpT) {
      this.nInf++;              // NaN 或 -Inf：记一笔，不污染 NLL 均值
    } else {
      this.nll += -lpT;
    }
    return this;
  };

  function cmpLp(a, b) { return b.lp - a.lp; }   // 稳定排序 ⇒ 平局时 id 小的在前

  Acc.prototype.summary = function () {
    var n = this.n || 1;
    var meanNll = this.nll / (this.n - this.nInf || 1);
    return {
      n: this.n,
      top1: 100 * this.top1 / n,
      top5: 100 * this.top5 / n,
      nll: meanNll,
      ppl: Math.exp(meanNll),
      nInf: this.nInf
    };
  };

  /** 把普通概率数组转成对数概率（就地返回新数组） */
  function toLogProbs(probs) {
    var out = new Float64Array(probs.length);
    for (var i = 0; i < probs.length; i++) out[i] = Math.log(probs[i]);
    return out;
  }

  var api = {
    splitIds: splitIds,
    forEachWindow: forEachWindow,
    forEachPosition: forEachPosition,
    countPositions: countPositions,
    Acc: Acc,
    toLogProbs: toLogProbs
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.EvalProtocol = api;

})(typeof globalThis !== 'undefined' ? globalThis : this);
