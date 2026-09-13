/* ============================================================================
 * minilm.js — 从零手写的微型 Transformer 语言模型
 * ----------------------------------------------------------------------------
 * 纯 JavaScript，零依赖，可在浏览器与 Node 中运行。
 *
 * 包含内容（全部手写，不调用任何深度学习库）：
 *   - 字符级分词器
 *   - Token Embedding + 可学习位置编码
 *   - 因果多头自注意力（Causal Multi-Head Self-Attention）
 *   - 前馈网络（GELU）+ Pre-LayerNorm 残差结构
 *   - 权重共享（Weight Tying）：输出层复用 Embedding 矩阵
 *   - 手写反向传播（反向推导到每一个矩阵乘法）
 *   - Adam 优化器
 *   - Top-k / 温度 / 重复惩罚 采样
 *   - 注意力矩阵导出（用于可视化）
 * ============================================================================ */

(function (global) {
  'use strict';

  // 数值精度开关：默认 float32（快，生产用）；
  // 设 globalThis.MINILM_USE_F64 = true 可切到 float64，
  // 用于梯度检查——float32 的量化噪声会淹没小梯度分量。
  var REAL = (global.MINILM_USE_F64 && typeof Float64Array !== 'undefined')
    ? Float64Array : Float32Array;

  var SQRT2PI = Math.sqrt(2 / Math.PI);

  /* ---------------------------- 基础工具 ---------------------------- */

  function zeros(n) { return new REAL(n); }

  // 标准正态分布初始化（Box-Muller）
  function randn(n, scale) {
    var a = new REAL(n);
    for (var i = 0; i < n; i++) {
      var u = 0, v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      a[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * scale;
    }
    return a;
  }

  // C[M,N] = A[M,K] * B[K,N]
  function mm(A, B, M, K, N, C) {
    for (var i = 0; i < M; i++) {
      var ai = i * K, ci = i * N;
      for (var j = 0; j < N; j++) C[ci + j] = 0;
      for (var k = 0; k < K; k++) {
        var a = A[ai + k];
        if (a === 0) continue;
        var bk = k * N;
        for (var j2 = 0; j2 < N; j2++) C[ci + j2] += a * B[bk + j2];
      }
    }
  }

  // 反向：dA += dC * B^T ; dB += A^T * dC
  function mmBack(dC, A, B, M, K, N, dA, dB) {
    var i, k, j, s;
    for (i = 0; i < M; i++) {
      var ci = i * N, ai = i * K;
      for (k = 0; k < K; k++) {
        s = 0;
        var bk = k * N;
        for (j = 0; j < N; j++) s += dC[ci + j] * B[bk + j];
        dA[ai + k] += s;
      }
    }
    for (k = 0; k < K; k++) {
      var bk2 = k * N;
      for (j = 0; j < N; j++) {
        s = 0;
        for (i = 0; i < M; i++) s += A[i * K + k] * dC[i * N + j];
        dB[bk2 + j] += s;
      }
    }
  }

  function addBias(y, b, M, N) {
    for (var i = 0; i < M; i++) {
      var off = i * N;
      for (var j = 0; j < N; j++) y[off + j] += b[j];
    }
  }

  function addBiasBack(dy, db, M, N) {
    for (var i = 0; i < M; i++) {
      var off = i * N;
      for (var j = 0; j < N; j++) db[j] += dy[off + j];
    }
  }

  /* ---------------------------- 激活函数 ---------------------------- */

  // GELU (tanh 近似，与 GPT 系列一致)
  function gelu(x) {
    var u = SQRT2PI * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1 + Math.tanh(u));
  }
  function dgelu(x) {
    var u = SQRT2PI * (x + 0.044715 * x * x * x);
    var t = Math.tanh(u);
    var du = SQRT2PI * (1 + 3 * 0.044715 * x * x);
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du;
  }

  /* --------------------------- LayerNorm --------------------------- */

  // y = (x - mean) / sqrt(var + eps) * gamma + beta
  function layernorm(x, g, b, M, D, eps, cache) {
    var y = zeros(M * D);
    var mean = zeros(M), rstd = zeros(M);
    for (var i = 0; i < M; i++) {
      var off = i * D, s = 0, d;
      for (d = 0; d < D; d++) s += x[off + d];
      var mu = s / D;
      var v = 0;
      for (d = 0; d < D; d++) { var t = x[off + d] - mu; v += t * t; }
      v /= D;
      var rs = 1 / Math.sqrt(v + eps);
      mean[i] = mu; rstd[i] = rs;
      for (d = 0; d < D; d++) y[off + d] = (x[off + d] - mu) * rs * g[d] + b[d];
    }
    if (cache) { cache.mean = mean; cache.rstd = rstd; cache.x = x; cache.g = g; }
    return y;
  }

  function layernormBack(dy, M, D, cache, dx, dg, db) {
    var mean = cache.mean, rstd = cache.rstd, x = cache.x, g = cache.g;
    for (var i = 0; i < M; i++) {
      var off = i * D, mu = mean[i], rs = rstd[i], d;
      var dvar = 0, dmu = 0;
      for (d = 0; d < D; d++) {
        var xhat = (x[off + d] - mu) * rs;
        var dhat = dy[off + d] * g[d];
        dg[d] += dy[off + d] * xhat;
        db[d] += dy[off + d];
        dx[off + d] += dhat * rs;
        dvar += dhat * (x[off + d] - mu) * (-0.5) * rs * rs * rs;
        dmu += dhat * (-rs);
      }
      for (d = 0; d < D; d++) {
        dx[off + d] += dvar * 2 * (x[off + d] - mu) / D + dmu / D;
      }
    }
  }

  /* ------------------------- 因果多头自注意力 ------------------------- */

  // q,k,v: [B*T, D]  →  out: [B*T, D]，attnStore: [B*H*T*T]
  function attentionForward(q, k, v, B, T, D, H, attnStore) {
    var DH = D / H;
    var scale = 1 / Math.sqrt(DH);
    var out = zeros(B * T * D);
    var scores = new REAL(T);
    for (var b = 0; b < B; b++) {
      for (var h = 0; h < H; h++) {
        var hoff = h * DH;
        for (var t = 0; t < T; t++) {
          var qOff = (b * T + t) * D + hoff;
          var mx = -Infinity, s, d;
          // 只计算 s <= t —— 这就是"因果掩码"，保证模型不能偷看未来
          for (s = 0; s <= t; s++) {
            var kOff = (b * T + s) * D + hoff;
            var dot = 0;
            for (d = 0; d < DH; d++) dot += q[qOff + d] * k[kOff + d];
            dot *= scale;
            scores[s] = dot;
            if (dot > mx) mx = dot;
          }
          var sum = 0;
          for (s = 0; s <= t; s++) { var e = Math.exp(scores[s] - mx); scores[s] = e; sum += e; }
          var inv = 1 / sum;
          for (s = 0; s <= t; s++) scores[s] *= inv;

          var base = ((b * H + h) * T + t) * T;
          for (s = 0; s <= t; s++) attnStore[base + s] = scores[s];

          var oOff = (b * T + t) * D + hoff;
          for (d = 0; d < DH; d++) {
            var acc = 0;
            for (s = 0; s <= t; s++) acc += scores[s] * v[(b * T + s) * D + hoff + d];
            out[oOff + d] = acc;
          }
        }
      }
    }
    return out;
  }

  function attentionBackward(dout, q, k, v, attnStore, B, T, D, H, dq, dk, dv) {
    var DH = D / H;
    var scale = 1 / Math.sqrt(DH);
    var dattn = new REAL(T);
    for (var b = 0; b < B; b++) {
      for (var h = 0; h < H; h++) {
        var hoff = h * DH;
        for (var t = 0; t < T; t++) {
          var base = ((b * H + h) * T + t) * T;
          var oOff = (b * T + t) * D + hoff;
          var DOT = 0, s, d;
          for (s = 0; s <= t; s++) {
            var kOff = (b * T + s) * D + hoff;
            var dot = 0;
            for (d = 0; d < DH; d++) dot += dout[oOff + d] * v[kOff + d];
            dattn[s] = dot;
            DOT += dot * attnStore[base + s];
          }
          for (s = 0; s <= t; s++) {
            var p = attnStore[base + s];
            var ds = p * (dattn[s] - DOT);   // softmax 的反向传播
            var vOff = (b * T + s) * D + hoff;
            var qOff = (b * T + t) * D + hoff;
            var kOff2 = (b * T + s) * D + hoff;
            for (d = 0; d < DH; d++) {
              dv[vOff + d] += p * dout[oOff + d];
              dq[qOff + d] += ds * k[kOff2 + d] * scale;
              dk[kOff2 + d] += ds * q[qOff + d] * scale;
            }
          }
        }
      }
    }
  }

  /* --------------------------- 模型本体 --------------------------- */

  function MiniLM(vocabSize, cfg) {
    cfg = cfg || {};
    this.cfg = {
      ctx: cfg.ctx || 32,
      d: cfg.d || 40,
      layers: cfg.layers || 2,
      heads: cfg.heads || 4,
      ffMul: cfg.ffMul || 4
    };
    this.V = vocabSize;
    this.P = {}; this.G = {}; this.M = {}; this.Vv = {};
    this.paramList = [];
    this.cache = null;
    this._init();
  }

  MiniLM.prototype._init = function () {
    var c = this.cfg;
    var D = c.d, L = c.layers, T = c.ctx, F = c.ffMul * D, V = this.V;
    var P = this.P, G = this.G, M = this.M, A = this.Vv;
    var list = this.paramList;

    var mk = function (name, n, scale) {
      P[name] = randn(n, scale); G[name] = zeros(n);
      M[name] = zeros(n); A[name] = zeros(n);
      list.push({ name: name, n: n });
    };

    var std = 0.02;
    var resStd = 0.02 / Math.sqrt(2 * L);

    // gamma 必须是全 1、beta 必须是全 0 —— 这是 LayerNorm 的标准初始化，
    // 若给 gamma 用随机值会让归一化层一开始就把信号扭曲掉。
    var mkOnes = function (name, n) {
      P[name] = new REAL(n); P[name].fill(1);
      G[name] = zeros(n); M[name] = zeros(n); A[name] = zeros(n);
      list.push({ name: name, n: n });
    };

    mk('emb', V * D, std);
    mk('pos', T * D, std);

    for (var l = 0; l < L; l++) {
      mkOnes('ln1g' + l, D); mk('ln1b' + l, D, 0);
      mk('Wq' + l, D * D, std); mk('bq' + l, D, 0);
      mk('Wk' + l, D * D, std); mk('bk' + l, D, 0);
      mk('Wv' + l, D * D, std); mk('bv' + l, D, 0);
      mk('Wo' + l, D * D, resStd); mk('bo' + l, D, 0);
      mkOnes('ln2g' + l, D); mk('ln2b' + l, D, 0);
      mk('W1_' + l, D * F, std); mk('b1_' + l, F, 0);
      mk('W2_' + l, F * D, resStd); mk('b2_' + l, D, 0);
    }
    mkOnes('lnfg', D); mk('lnfb', D, 0);
  };

  MiniLM.prototype.paramCount = function () {
    var n = 0;
    for (var i = 0; i < this.paramList.length; i++) n += this.paramList[i].n;
    return n;
  };

  /* ------------------------------ 前向 ------------------------------ */

  // tokens: Int32Array/Array，长度必须是 T 的整数倍（B*T）
  // targets: 同长度，训练时提供；生成时传 null
  MiniLM.prototype.forward = function (tokens, targets, TOverride) {
    var c = this.cfg;
    var D = c.d, L = c.layers, H = c.heads, F = c.ffMul * D;
    var T = TOverride || c.ctx;
    var V = this.V, P = this.P;
    var BT = tokens.length;
    var B = BT / T;

    var cache = this.cache = {
      tokens: tokens, targets: targets, B: B, T: T, BT: BT
    };
    var i, d, t, j;

    // ---- 1. 词嵌入 + 位置嵌入 ----
    var x = zeros(BT * D);
    for (i = 0; i < BT; i++) {
      var tok = tokens[i], tp = (i % T) * D;
      for (d = 0; d < D; d++) x[i * D + d] = P.emb[tok * D + d] + P.pos[tp + d];
    }
    var cur = x;

    // ---- 2. N 层 Transformer Block ----
    var layerCaches = [];
    for (var l = 0; l < L; l++) {
      var lc = {};

      // Pre-LayerNorm + 多头注意力
      lc.xin1 = cur;
      var ln1c = {};
      var xn1 = layernorm(cur, P['ln1g' + l], P['ln1b' + l], BT, D, 1e-5, ln1c);
      lc.ln1 = ln1c; lc.xn1 = xn1;

      var q = zeros(BT * D), k = zeros(BT * D), v = zeros(BT * D);
      mm(xn1, P['Wq' + l], BT, D, D, q);
      mm(xn1, P['Wk' + l], BT, D, D, k);
      mm(xn1, P['Wv' + l], BT, D, D, v);
      addBias(q, P['bq' + l], BT, D);
      addBias(k, P['bk' + l], BT, D);
      addBias(v, P['bv' + l], BT, D);
      lc.q = q; lc.k = k; lc.v = v;

      var attnStore = zeros(B * H * T * T);
      var ao = attentionForward(q, k, v, B, T, D, H, attnStore);
      lc.attnStore = attnStore; lc.ao = ao;

      var attnOut = zeros(BT * D);
      mm(ao, P['Wo' + l], BT, D, D, attnOut);
      addBias(attnOut, P['bo' + l], BT, D);

      // 残差连接 1
      var xmid = zeros(BT * D);
      for (i = 0; i < BT * D; i++) xmid[i] = cur[i] + attnOut[i];
      lc.xmid = xmid;

      // Pre-LayerNorm + 前馈网络
      var ln2c = {};
      var xn2 = layernorm(xmid, P['ln2g' + l], P['ln2b' + l], BT, D, 1e-5, ln2c);
      lc.ln2 = ln2c; lc.xn2 = xn2;

      var hp = zeros(BT * F);
      mm(xn2, P['W1_' + l], BT, D, F, hp);
      addBias(hp, P['b1_' + l], BT, F);
      var hg = new REAL(BT * F);
      for (i = 0; i < BT * F; i++) hg[i] = gelu(hp[i]);
      lc.hp = hp; lc.hg = hg;

      var fo = zeros(BT * D);
      mm(hg, P['W2_' + l], BT, F, D, fo);
      addBias(fo, P['b2_' + l], BT, D);

      // 残差连接 2
      var xout = zeros(BT * D);
      for (i = 0; i < BT * D; i++) xout[i] = xmid[i] + fo[i];
      lc.xout = xout;
      cur = xout;

      layerCaches.push(lc);
    }

    // ---- 3. 最终 LayerNorm ----
    var lnfc = {};
    var xnf = layernorm(cur, P.lnfg, P.lnfb, BT, D, 1e-5, lnfc);
    cache.lnfc = lnfc; cache.xnf = xnf;

    // ---- 4. 输出层（权重共享：复用 emb 的转置）----
    var logits = zeros(BT * V);
    for (i = 0; i < BT; i++) {
      var xo = i * D;
      for (j = 0; j < V; j++) {
        var eo = j * D, s = 0;
        for (d = 0; d < D; d++) s += xnf[xo + d] * P.emb[eo + d];
        logits[i * V + j] = s;
      }
    }
    cache.logits = logits;

    // ---- 5. Softmax + 交叉熵损失 ----
    var loss = 0;
    var probs = zeros(BT * V);
    if (targets) {
      for (i = 0; i < BT; i++) {
        var off = i * V, mx = -Infinity;
        for (j = 0; j < V; j++) if (logits[off + j] > mx) mx = logits[off + j];
        var sum = 0;
        for (j = 0; j < V; j++) {
          var e = Math.exp(logits[off + j] - mx);
          probs[off + j] = e; sum += e;
        }
        var inv = 1 / sum;
        for (j = 0; j < V; j++) probs[off + j] *= inv;
        loss -= Math.log(probs[off + targets[i]] + 1e-9);
      }
      loss /= BT;
    }
    cache.probs = probs; cache.loss = loss; cache.layers = layerCaches;

    return { loss: loss, logits: logits, probs: probs, cache: cache };
  };

  /* ------------------------------ 反向 ------------------------------ */

  MiniLM.prototype.backward = function () {
    var c = this.cache;
    if (!c || !c.targets) throw new Error('backward() 需要先带 targets 前向');
    var cfg = this.cfg;
    var D = cfg.d, L = cfg.layers, H = cfg.heads, F = cfg.ffMul * D;
    var V = this.V, P = this.P, G = this.G;
    var BT = c.BT, B = c.B, T = c.T;
    var i, d, j, l;

    // ---- 交叉熵 + Softmax 的反向：dlogits = probs - onehot ----
    var dlogits = zeros(BT * V);
    for (i = 0; i < BT; i++) {
      var off = i * V, dl;
      for (j = 0; j < V; j++) dlogits[off + j] = c.probs[off + j] / BT;
      dlogits[off + c.targets[i]] -= 1 / BT;
    }

    // ---- 反向输出层（权重共享：梯度回流到 emb）----
    var dxnf = zeros(BT * D);
    for (j = 0; j < V; j++) {
      var eo = j * D;
      for (i = 0; i < BT; i++) {
        var dl2 = dlogits[i * V + j];
        if (dl2 === 0) continue;
        var xo = i * D;
        for (d = 0; d < D; d++) G.emb[eo + d] += c.xnf[xo + d] * dl2;
      }
    }
    for (i = 0; i < BT; i++) {
      var xo2 = i * D;
      for (j = 0; j < V; j++) {
        var dl3 = dlogits[i * V + j];
        if (dl3 === 0) continue;
        var eo2 = j * D;
        for (d = 0; d < D; d++) dxnf[xo2 + d] += dl3 * P.emb[eo2 + d];
      }
    }

    // ---- 反向最终 LayerNorm ----
    var dx = zeros(BT * D);
    layernormBack(dxnf, BT, D, c.lnfc, dx, G.lnfg, G.lnfb);
    if (this.DEBUG) { this._dxnf = dxnf; this._dx = dx; }

    // ---- 逐层倒序反向 ----
    for (l = L - 1; l >= 0; l--) {
      var lc = c.layers[l];

      // 残差 2：xout = xmid + fo
      var dfo = dx;
      var dhg = zeros(BT * F);
      mmBack(dfo, lc.hg, P['W2_' + l], BT, F, D, dhg, G['W2_' + l]);
      addBiasBack(dfo, G['b2_' + l], BT, D);

      // GELU 反向
      var dhp = new REAL(BT * F);
      for (i = 0; i < BT * F; i++) dhp[i] = dhg[i] * dgelu(lc.hp[i]);

      var dxn2 = zeros(BT * D);
      mmBack(dhp, lc.xn2, P['W1_' + l], BT, D, F, dxn2, G['W1_' + l]);
      addBiasBack(dhp, G['b1_' + l], BT, F);

      // LayerNorm 2 反向
      var dxmid_ln2 = zeros(BT * D);
      layernormBack(dxn2, BT, D, lc.ln2, dxmid_ln2, G['ln2g' + l], G['ln2b' + l]);

      // xmid 影响输出有两条路：直通残差 2 + 经过 LayerNorm2/FFN，
      // 所以 dL/d(xmid) 必须把两者相加，不能只用 ln2 那一条。
      var dxmid = zeros(BT * D);
      for (i = 0; i < BT * D; i++) dxmid[i] = dfo[i] + dxmid_ln2[i];

      // 残差 1：xmid = xin1 + attnOut  → 梯度分两路
      var dao = zeros(BT * D);
      mmBack(dxmid, lc.ao, P['Wo' + l], BT, D, D, dao, G['Wo' + l]);
      addBiasBack(dxmid, G['bo' + l], BT, D);

      // 注意力反向
      var dq = zeros(BT * D), dk = zeros(BT * D), dv = zeros(BT * D);
      attentionBackward(dao, lc.q, lc.k, lc.v, lc.attnStore, B, T, D, H, dq, dk, dv);

      var dxn1 = zeros(BT * D);
      mmBack(dq, lc.xn1, P['Wq' + l], BT, D, D, dxn1, G['Wq' + l]);
      mmBack(dk, lc.xn1, P['Wk' + l], BT, D, D, dxn1, G['Wk' + l]);
      mmBack(dv, lc.xn1, P['Wv' + l], BT, D, D, dxn1, G['Wv' + l]);
      addBiasBack(dq, G['bq' + l], BT, D);
      addBiasBack(dk, G['bk' + l], BT, D);
      addBiasBack(dv, G['bv' + l], BT, D);

      // LayerNorm 1 反向
      var dxin1 = zeros(BT * D);
      layernormBack(dxn1, BT, D, lc.ln1, dxin1, G['ln1g' + l], G['ln1b' + l]);

      // dL/d(xin1) = dL/d(xmid)  ← 残差 1 直通
      //            + dL/d(xin1)|attn ← 经 LayerNorm1 / Attention
      // dxmid 此刻已是 dL/d(xmid)（含残差 2 直通），故这里是赋值而非累加。
      for (i = 0; i < BT * D; i++) dx[i] = dxmid[i] + dxin1[i];
    }

    // ---- 反向词嵌入 / 位置嵌入 ----
    for (i = 0; i < BT; i++) {
      var tok = c.tokens[i], tp = (i % T) * D, xo3 = i * D;
      for (d = 0; d < D; d++) {
        G.emb[tok * D + d] += dx[xo3 + d];
        G.pos[tp + d] += dx[xo3 + d];
      }
    }
  };

  /* ---------------------------- 优化器 ---------------------------- */

  MiniLM.prototype.zeroGrad = function () {
    for (var i = 0; i < this.paramList.length; i++) this.G[this.paramList[i].name].fill(0);
  };

  MiniLM.prototype.clipGrad = function (maxNorm) {
    var total = 0, i, k;
    for (i = 0; i < this.paramList.length; i++) {
      var g = this.G[this.paramList[i].name];
      for (k = 0; k < g.length; k++) total += g[k] * g[k];
    }
    var norm = Math.sqrt(total);
    if (norm > maxNorm) {
      var s = maxNorm / (norm + 1e-6);
      for (i = 0; i < this.paramList.length; i++) {
        var g2 = this.G[this.paramList[i].name];
        for (k = 0; k < g2.length; k++) g2[k] *= s;
      }
    }
    return norm;
  };

  MiniLM.prototype.step = function (lr, t) {
    var b1 = 0.9, b2 = 0.999, eps = 1e-8;
    var bc1 = 1 - Math.pow(b1, t), bc2 = 1 - Math.pow(b2, t);
    for (var i = 0; i < this.paramList.length; i++) {
      var name = this.paramList[i].name, n = this.paramList[i].n;
      var P = this.P[name], G = this.G[name], M = this.M[name], A = this.Vv[name];
      for (var k = 0; k < n; k++) {
        var g = G[k];
        M[k] = b1 * M[k] + (1 - b1) * g;
        A[k] = b2 * A[k] + (1 - b2) * g * g;
        P[k] -= lr * (M[k] / bc1) / (Math.sqrt(A[k] / bc2) + eps);
      }
    }
  };

  /* ---------------------------- 文本生成 ---------------------------- */

  // ids: 提示词的 token id 数组；返回 { ids, steps } —— steps 记录每步候选分布
  MiniLM.prototype.generate = function (ids, opts) {
    opts = opts || {};
    var maxNew = opts.maxNew || 60;
    var temp = opts.temp == null ? 0.8 : opts.temp;
    var topk = opts.topk || 8;
    var repPen = opts.repPenalty == null ? 1.15 : opts.repPenalty;
    var repWin = opts.repWindow || 24;

    var V = this.V, out = ids.slice();
    var trace = [];
    var self = this;

    for (var s = 0; s < maxNew; s++) {
      var ctx = out.slice(Math.max(0, out.length - this.cfg.ctx));
      if (ctx.length === 0) break;
      var n = ctx.length;

      this.forward(ctx, null, n);

      var logits = this.cache.logits;
      var last = (n - 1) * V;
      var arr = new REAL(V);
      var j;
      for (j = 0; j < V; j++) arr[j] = logits[last + j] / temp;

      // 重复惩罚：压掉最近出现过的 token，缓解小模型"复读机"问题
      if (repPen !== 1) {
        var from = Math.max(0, out.length - repWin);
        for (var k = from; k < out.length; k++) {
          var id = out[k];
          arr[id] = arr[id] > 0 ? arr[id] / repPen : arr[id] * repPen;
        }
      }

      // top-k 截断
      var sorted = Array.prototype.slice.call(arr).sort(function (a, b) { return b - a; });
      var kth = sorted[Math.min(topk, V - 1) - 1];
      var mx = -Infinity;
      for (j = 0; j < V; j++) {
        if (arr[j] < kth) arr[j] = -Infinity;
        else if (arr[j] > mx) mx = arr[j];
      }

      var probs = new REAL(V), sum = 0;
      for (j = 0; j < V; j++) {
        var e = arr[j] === -Infinity ? 0 : Math.exp(arr[j] - mx);
        probs[j] = e; sum += e;
      }
      var inv = 1 / (sum || 1);

      var r = Math.random(), acc = 0, pick = 0;
      for (j = 0; j < V; j++) {
        acc += probs[j] * inv;
        if (r <= acc) { pick = j; break; }
      }

      // 记录这一步的高概率候选（用于界面展示"模型在想什么"）
      var cands = [];
      for (j = 0; j < V; j++) {
        if (probs[j] > 0) cands.push({ id: j, p: probs[j] * inv });
      }
      cands.sort(function (a, b) { return b.p - a.p; });
      trace.push({ step: s, chosen: pick, candidates: cands.slice(0, 6) });

      out.push(pick);
    }
    return { ids: out, trace: trace };
  };

  // 返回某一层某个注意力头的 [T, T] 注意力矩阵（用于热力图）
  MiniLM.prototype.getAttention = function (layer, head) {
    if (!this.cache || !this.cache.layers) return null;
    var lc = this.cache.layers[layer];
    if (!lc) return null;
    var T = this.cache.T, H = this.cfg.heads;
    var base = head * T * T;
    var m = new REAL(T * T);
    for (var i = 0; i < T * T; i++) m[i] = lc.attnStore[base + i];
    return m;
  };

  /* ---------------------------- 分词器 ---------------------------- */

  function buildVocab(text) {
    var set = new Set();
    for (var i = 0; i < text.length; i++) set.add(text[i]);
    var chars = Array.from(set).sort();
    var itos = ['\u0000'].concat(chars);   // 0 号保留作未知字符
    var stoi = new Map();
    for (var k = 0; k < itos.length; k++) stoi.set(itos[k], k);
    return { itos: itos, stoi: stoi, size: itos.length };
  }

  function encode(text, stoi) {
    var ids = [];
    for (var i = 0; i < text.length; i++) {
      var id = stoi.get(text[i]);
      ids.push(id === undefined ? 0 : id);
    }
    return ids;
  }

  function decode(ids, itos) {
    var s = '';
    for (var i = 0; i < ids.length; i++) {
      if (ids[i] === 0) continue;
      s += itos[ids[i]] || '';
    }
    return s;
  }

  /* ---------------------------- 训练批次 ---------------------------- */

  // 从 id 序列里随机采样 B 个长度为 T 的连续片段
  function makeBatch(ids, B, T) {
    var tokens = new Int32Array(B * T);
    var targets = new Int32Array(B * T);
    var maxStart = ids.length - T - 1;
    if (maxStart <= 0) throw new Error('语料太短，请增加文本长度或减小上下文窗口');
    for (var b = 0; b < B; b++) {
      var start = Math.floor(Math.random() * maxStart);
      for (var t = 0; t < T; t++) {
        tokens[b * T + t] = ids[start + t];
        targets[b * T + t] = ids[start + t + 1];
      }
    }
    return { tokens: tokens, targets: targets };
  }

  /* ---------------------------- 导出 ---------------------------- */

  var api = {
    MiniLM: MiniLM,
    buildVocab: buildVocab,
    encode: encode,
    decode: decode,
    makeBatch: makeBatch,
    gelu: gelu,
    dgelu: dgelu,
    // 以下导出仅供单元测试使用
    attentionForward: attentionForward,
    attentionBackward: attentionBackward,
    layernorm: layernorm,
    layernormBack: layernormBack
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.MiniLMKit = api;

})(typeof globalThis !== 'undefined' ? globalThis : this);
