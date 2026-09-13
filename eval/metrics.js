/* ============================================================================
 * eval/metrics.js — 生成质量指标（重复率 / 训练集重合度 / 未登录字）
 * ----------------------------------------------------------------------------
 * 为什么需要这些：
 *   Top-1 只衡量「猜下一个字猜得准不准」，它完全衡量不了「生成的东西能不能读」。
 *   一个把训练集背下来的模型 Top-1 可以很高，但那是记忆不是写作。
 *   所以要分开测两件事：
 *     · 重复率   —— 生成文本自己跟自己重复到什么程度（复读机检测）
 *     · 重合度   —— 生成文本有多少是直接从训练集里搬的（抄袭检测）
 *
 * 所有函数都设计成「输入短字符串能手算出答案」，配套 eval/fixtures/ 做断言。
 * ============================================================================ */
(function (root) {
  'use strict';

  /* ------------------------------------------------------------------ */
  /* 未登录字（字符级模型的静默数据丢失）                                  */
  /* ------------------------------------------------------------------ */

  /**
   * 统计有多少字不在词表里。字符级模型遇到不在词表的字会静默丢掉，
   * 用户输入被悄悄改写了却不自知 —— 这个数字必须显式披露。
   * stoi 是 Map（minilm.js 的 buildVocab 返回的确实是 Map，不是普通对象）。
   */
  function oovInfo(text, stoi) {
    var oovChars = [], kept = '';
    for (var i = 0; i < text.length; i++) {
      var ch = text[i];
      // 兼容 Map 与普通对象两种词表
      var known = (stoi && typeof stoi.has === 'function') ? stoi.has(ch)
        : Object.prototype.hasOwnProperty.call(stoi, ch);
      if (known) kept += ch;
      else oovChars.push(ch);   // 按出现次数计，重复出现的也算多次
    }
    return {
      total: text.length,
      oov: oovChars.length,
      oovUnique: Array.from(new Set(oovChars)),
      kept: kept,
      ratio: text.length ? oovChars.length / text.length : 0
    };
  }

  /* ------------------------------------------------------------------ */
  /* n-gram 工具                                                         */
  /* ------------------------------------------------------------------ */

  function ngrams(text, n) {
    var out = [];
    if (n <= 0 || text.length < n) return out;
    for (var i = 0; i + n <= text.length; i++) out.push(text.slice(i, i + n));
    return out;
  }

  /** distinct-N：不重复 n-gram 数 / 总 n-gram 数。越低越复读机 */
  function distinctN(text, n) {
    var g = ngrams(text, n);
    if (!g.length) return 0;
    var set = new Set(g);
    return set.size / g.length;
  }

  /**
   * 重复率：出现过一次以上的 n-gram 占比（按出现次数算，不是按种类算）。
   * 例：abcabcabc 的 4-gram 为 abca,bcab,cabc,abca,bcab → 5 个中 2 个是重复 → 0.4
   */
  function repeatRate(text, n) {
    var g = ngrams(text, n);
    if (g.length <= 1) return 0;
    var seen = new Set(), dup = 0;
    for (var i = 0; i < g.length; i++) {
      if (seen.has(g[i])) dup++;
      else seen.add(g[i]);
    }
    return dup / g.length;
  }

  /**
   * 训练集重合率：生成文本里有多少 n-gram 能在训练文本里原样找到。
   * 这是「会写」和「会背」的分水岭 —— 高重合 + 低重复 = 在合理借用；
   * 高重合 + 高重复 = 在复读训练集片段。
   */
  function overlapRate(genText, trainText, n) {
    var g = ngrams(genText, n);
    if (!g.length) return 0;
    var pool = new Set(ngrams(trainText, n));
    var hit = 0;
    for (var i = 0; i < g.length; i++) if (pool.has(g[i])) hit++;
    return hit / g.length;
  }

  /* ------------------------------------------------------------------ */
  /* 最长公共子串                                                        */
  /* ------------------------------------------------------------------ */

  /**
   * 最长公共子串（连续，不是子序列）。返回 { len, str }。
   * 这是「抄袭」最直观的证据：如果生成文本里有一段 30 字和训练集一字不差，
   * 那它不是在创作，是在回放。
   */
  function longestCommonSubstring(a, b) {
    if (!a.length || !b.length) return { len: 0, str: '' };
    // prev/cur 滚动数组，空间 O(len(b))
    var prev = new Uint32Array(b.length + 1);
    var cur = new Uint32Array(b.length + 1);
    var bestLen = 0, bestEndB = 0;

    for (var i = 1; i <= a.length; i++) {
      for (var j = 1; j <= b.length; j++) {
        if (a.charCodeAt(i - 1) === b.charCodeAt(j - 1)) {
          cur[j] = prev[j - 1] + 1;
          if (cur[j] > bestLen) { bestLen = cur[j]; bestEndB = j; }
        } else {
          cur[j] = 0;
        }
      }
      var t = prev; prev = cur; cur = t;
      cur.fill(0);
    }
    return { len: bestLen, str: bestLen ? b.slice(bestEndB - bestLen, bestEndB) : '' };
  }

  var api = {
    oovInfo: oovInfo,
    ngrams: ngrams,
    distinctN: distinctN,
    repeatRate: repeatRate,
    overlapRate: overlapRate,
    longestCommonSubstring: longestCommonSubstring
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.EvalMetrics = api;

})(typeof globalThis !== 'undefined' ? globalThis : this);
