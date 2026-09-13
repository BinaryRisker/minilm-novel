/* ============================================================================
 * uidom.js — 用最小 DOM 桩把 index.html 的 UI 真正跑起来
 * ----------------------------------------------------------------------------
 * 为什么不用真浏览器：agent-browser 需要下载 ~500MB Chromium，成本太高。
 * 而 UI 里真正容易出错的东西 —— init() 的 DOM 装配、事件绑定、训练循环、
 * canvas 绘制 —— 都可以用一个手写的最小 DOM 桩喂进去跑，抓到异常即可。
 *
 * 覆盖范围：
 *   - 从 HTML 里解析每个 id 的初始 value / min / max，让表单状态是真实的
 *   - 用"虚拟时钟"代替 performance.now()，让训练循环可确定地推进
 *   - 手动 drain requestAnimationFrame 队列，完整跑完一次训练
 *   - 训练后调用 生成 / 三温度对比 / 注意力 / 梯度检查，
 *     任何一个环节抛异常都会让测试失败
 *
 * 用法：node uidom.js
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
/* 1. 从 HTML 里解析表单初值，让桩环境尽量接近真实                        */
/* ------------------------------------------------------------------ */

var seed = {};
var tagRe = /<(input|select|textarea)\b([^>]*)>/g, tm;
while ((tm = tagRe.exec(html)) !== null) {
  var attrs = tm[2];
  var idm = /\bid="([^"]+)"/.exec(attrs);
  if (!idm) continue;
  var id = idm[1];
  var val = /(?:\bvalue)="([^"]*)"/.exec(attrs);
  var mn = /\bmin="([^"]*)"/.exec(attrs);
  var mx = /\bmax="([^"]*)"/.exec(attrs);
  var ty = /\btype="([^"]*)"/.exec(attrs);
  seed[id] = {
    value: val ? val[1] : '',
    min: mn ? mn[1] : undefined,
    max: mx ? mx[1] : undefined,
    type: ty ? ty[1] : tm[1]
  };
}

/* ------------------------------------------------------------------ */
/* 2. 最小 DOM 实现                                                    */
/* ------------------------------------------------------------------ */

var elById = {};
var created = 0;

function mkStyle() { return {}; }

function Context2D() {
  var noop = function () { return undefined; };
  this.clearRect = noop; this.setTransform = noop; this.beginPath = noop;
  this.moveTo = noop; this.lineTo = noop; this.stroke = noop; this.fill = noop;
  this.fillRect = noop; this.fillText = noop; this.arc = noop;
  this.setLineDash = noop; this.save = noop; this.restore = noop;
  this.createLinearGradient = function () { return { addColorStop: noop }; };
  this.createRadialGradient = function () { return { addColorStop: noop }; };
}

function El(tag, id) {
  this.tagName = String(tag || 'div').toUpperCase();
  this.id = id || '';
  this.children = [];
  this._text = '';
  this._html = '';
  this.className = '';
  this.style = mkStyle();
  this.dataset = {};
  this.handlers = {};
  this.classList = {
    _s: {},
    add: function (c) { this._s[c] = 1; },
    remove: function (c) { delete this._s[c]; },
    toggle: function (c, on) { if (on === undefined) on = !this._s[c]; if (on) this._s[c] = 1; else delete this._s[c]; },
    contains: function (c) { return !!this._s[c]; }
  };
  this._qs = {};
  var self = this;
  this.getContext = function () { return self._ctx || (self._ctx = new Context2D()); };
  this.getAttribute = function (n) { return self._attrs && self._attrs[n] !== undefined ? self._attrs[n] : null; };
}
Object.defineProperty(El.prototype, 'textContent', {
  get: function () { return this._text || this._html.replace(/<[^>]*>/g, ''); },
  set: function (v) { this._text = String(v); this._html = String(v); this.children = []; }
});
// 真实 DOM 里设置 innerHTML 会清空子节点，这里必须一致，
// 否则 renderSnapshots() 的"先清空再重填"会漏掉旧节点。
Object.defineProperty(El.prototype, 'innerHTML', {
  get: function () { return this._html; },
  set: function (v) { this._html = String(v); this._text = ''; this.children = []; }
});
Object.defineProperty(El.prototype, 'clientWidth', {
  get: function () { return this._cw || 420; }
});
Object.defineProperty(El.prototype, 'clientHeight', {
  get: function () {
    if (this._ch) return this._ch;
    var h = parseInt(this.getAttribute('height'), 10);
    return isNaN(h) ? 200 : h;
  }
});
Object.defineProperty(El.prototype, 'scrollHeight', { get: function () { return 100; } });
Object.defineProperty(El.prototype, 'scrollTop', { get: function () { return 0; }, set: function () {} });
El.prototype.appendChild = function (c) { this.children.push(c); return c; };
El.prototype.querySelector = function (sel) {
  if (!this._qs[sel]) this._qs[sel] = new El('div', '');
  return this._qs[sel];
};
El.prototype.querySelectorAll = function () { return []; };
El.prototype.addEventListener = function (ev, fn) {
  (this.handlers[ev] = this.handlers[ev] || []).push(fn);
};
El.prototype.fire = function (ev) {
  var hs = this.handlers[ev] || [];
  var self = this;
  hs.forEach(function (h) { h.call(self, { target: self }); });
  return hs.length;
};

function getEl(id) {
  if (!elById[id]) {
    var e = new El('div', id);
    var s = seed[id];
    if (s) {
      e.value = s.value;
      e._attrs = {};
      if (s.min !== undefined) e._attrs.min = s.min;
      if (s.max !== undefined) e._attrs.max = s.max;
      if (s.type) e._attrs.type = s.type;
    }
    elById[id] = e;
  }
  return elById[id];
}

var rafQueue = [];
var alerts = [];
var clock = 0;

var sandbox = {
  console: console,
  Math: Math, Date: Date, JSON: JSON, Set: Set, Map: Map, Array: Array, Object: Object,
  Int32Array: Int32Array, Float32Array: Float32Array, Float64Array: Float64Array,
  isNaN: isNaN, parseInt: parseInt, parseFloat: parseFloat,
  performance: { now: function () { return clock++; } },   // 虚拟时钟：每次调用 +1
  requestAnimationFrame: function (fn) { rafQueue.push(fn); return rafQueue.length; },
  setTimeout: function (fn) { fn(); return 0; },            // 同步执行，便于测试
  clearTimeout: function () {},
  alert: function (m) { alerts.push(String(m)); },
  devicePixelRatio: 1
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
sandbox.document = {
  body: new El('body', ''),
  getElementById: getEl,
  createElement: function (t) { created++; return new El(t, ''); },
  querySelectorAll: function () { return []; },
  querySelector: function () { return new El('div', ''); }
};
sandbox.window.addEventListener = function () {};
vm.createContext(sandbox);

/* ------------------------------------------------------------------ */
/* 3. 执行 index.html 里的两块内联脚本                                 */
/* ------------------------------------------------------------------ */

head('1 · 加载构建产物');

var blocks = [];
var re = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g, m;
while ((m = re.exec(html)) !== null) blocks.push(m[1]);
ok(blocks.length === 2, '内联脚本块数量正确', '(' + blocks.length + ')');

var loadErr = null;
try {
  vm.runInContext(blocks[0], sandbox, { filename: 'lib.js', timeout: 20000 });
  vm.runInContext(blocks[1], sandbox, { filename: 'ui.js', timeout: 30000 });
} catch (e) { loadErr = e; }
ok(!loadErr, '两块脚本执行无异常', loadErr ? String(loadErr.stack).split('\n').slice(0, 3).join(' | ') : '');
if (loadErr) { console.log('\n致命错误，终止。'); process.exit(1); }

var LAB = sandbox.window.__MiniLMLab;
ok(!!LAB, 'UI 暴露了 __MiniLMLab 测试出口');
ok(!!sandbox.window.__MiniLMLab.state, '内部状态可访问');

/* ------------------------------------------------------------------ */
head('2 · 初始状态（init() 是否正确装配）');

var S = LAB.state;
ok(/[\d,]{4,}/.test(sandbox.document.getElementById('sParam').innerHTML),
   '参数量已显示', '= ' + sandbox.document.getElementById('sParam').innerHTML.replace(/<[^>]*>/g, ''));
ok(sandbox.document.getElementById('sCorpus').textContent === '2,668',
   '语料字符数已显示', '= ' + sandbox.document.getElementById('sCorpus').textContent);
ok(sandbox.document.getElementById('structTable').innerHTML.length > 100,
   '模型结构表已渲染', '(' + sandbox.document.getElementById('structTable').innerHTML.length + ' 字符)');
ok(sandbox.document.getElementById('promptChips').children.length === 6,
   '提示词预设按钮已生成', '(' + sandbox.document.getElementById('promptChips').children.length + ' 个)');
ok(sandbox.document.getElementById('fCorpus').children.length === 4,
   '语料下拉框有 4 个选项（含自定义）');

/* ------------------------------------------------------------------ */
head('3 · 完整跑一遍训练（虚拟时钟驱动）');

LAB.preset('fast');                       // 更小的模型，测试更快
sandbox.document.getElementById('fSteps').value = '120';
sandbox.document.getElementById('pTotal').value = '120';

var startErr = null;
try { LAB.start(); } catch (e) { startErr = e; }
ok(!startErr, '点击「开始训练」无异常', startErr ? String(startErr.stack).split('\n')[1] : '');
ok(S.running === true, '训练进入运行态');

var guard = 0;
var tickErr = null;
try {
  while (!S.done && guard < 5000) {
    if (rafQueue.length === 0) break;
    var fn = rafQueue.shift();
    fn();
    guard++;
  }
} catch (e) { tickErr = e; }
ok(!tickErr, '训练循环无异常', tickErr ? String(tickErr.stack).split('\n').slice(0, 2).join(' | ') : '');
ok(S.done === true, '训练正常结束', '用了 ' + guard + ' 帧');
ok(S.step === 120, '步数达到设定值', '= ' + S.step);
ok(S.losses.length === 120, '每步损失都被记录', '= ' + S.losses.length);

var first = S.losses.slice(0, 10).reduce(function (a, b) { return a + b; }, 0) / 10;
var last = S.losses.slice(-10).reduce(function (a, b) { return a + b; }, 0) / 10;
ok(last < first, '损失确实下降', first.toFixed(3) + ' → ' + last.toFixed(3));

ok(S.snapshots.length >= 4, '训练快照已生成', '(' + S.snapshots.length + ' 张)');
ok(sandbox.document.getElementById('snapBox').children.length === S.snapshots.length,
   '快照已渲染到 DOM', '(' + sandbox.document.getElementById('snapBox').children.length + ' 个节点)');
ok(sandbox.document.getElementById('progBar').style.width === '100%',
   '进度条到达 100%', '= ' + sandbox.document.getElementById('progBar').style.width);
ok(sandbox.document.getElementById('bGen').disabled === false, '生成按钮已启用');

/* ------------------------------------------------------------------ */
head('4 · 训练后各功能按钮');

var genErr = null;
try { LAB.generate(); } catch (e) { genErr = e; }
ok(!genErr, '「生成」无异常', genErr ? String(genErr.stack).split('\n')[1] : '');
var genText = sandbox.document.getElementById('genOut').querySelector('.gen').textContent;
ok(genText.length > 50, '生成文本已写入输出区', '(' + genText.length + ' 字)');
console.log('    样例: ' + genText.slice(0, 56).replace(/\n/g, ' '));

var candN = sandbox.document.getElementById('candBox').children.length;
ok(candN > 0, '候选分布已渲染', '(' + candN + ' 组)');

var g3Err = null;
try { LAB.generate3(); } catch (e) { g3Err = e; }
ok(!g3Err, '「三种温度对比」无异常', g3Err ? String(g3Err.stack).split('\n')[1] : '');
ok(sandbox.document.getElementById('genOut').innerHTML.indexOf('温度 0.5') >= 0,
   '三温度结果已写入');

var atErr = null;
try { LAB.attention(); } catch (e) { atErr = e; }
ok(!atErr, '「刷新注意力」无异常', atErr ? String(atErr.stack).split('\n')[1] : '');
var heatMeta = sandbox.document.getElementById('heatMeta').innerHTML;
ok(/序列长度 \d+/.test(heatMeta), '注意力元信息已更新', heatMeta.slice(0, 42));

// 切层/切头（会触发 refreshAttention，路径与滑块回调一致）
var slErr = null;
try {
  sandbox.document.getElementById('aLayer').value = '1';
  sandbox.document.getElementById('aHead').value = '3';
  sandbox.document.getElementById('aLayer').fire('input');
  sandbox.document.getElementById('aHead').fire('input');
} catch (e) { slErr = e; }
ok(!slErr, '切换层/头的滑块回调无异常', slErr ? String(slErr.stack).split('\n')[1] : '');

/* ---- 提示词 → token 映射 / 未登录字披露 ---- */
var tmBox = sandbox.document.getElementById('tokMap');
var tmErr = null;
try {
  sandbox.document.getElementById('gPrompt').value = '苏念说，我';
  sandbox.document.getElementById('gPrompt').fire('input');
} catch (e) { tmErr = e; }
ok(!tmErr, '输入提示词触发 token 映射无异常', tmErr ? String(tmErr.stack).split('\n')[1] : '');
var tmHtml = tmBox.innerHTML;
ok(tmHtml.indexOf('class="tok"') >= 0, 'token 映射已渲染逐字格子');
ok(tmHtml.indexOf('全部在词表内') >= 0, '语料内提示词 → 判定为全部在词表内');
ok(tmHtml.indexOf('模型实际输入 id') >= 0, '显示了模型实际拿到的 id 序列');

// 故意塞进词表外的字：字符级模型会静默丢弃，这里必须显式告警
var oovErr = null;
try {
  sandbox.document.getElementById('gPrompt').value = '警察敲了敲门，';
  sandbox.document.getElementById('gPrompt').fire('input');
} catch (e) { oovErr = e; }
ok(!oovErr, '含未登录字的提示词无异常', oovErr ? String(oovErr.stack).split('\n')[1] : '');
var oovHtml = tmBox.innerHTML;
var oovCnt = (oovHtml.match(/class="tok oov"/g) || []).length;
ok(oovCnt === 4, '「警察敲了敲门，」标出 4 个未登录字（警/察/敲/敲）', '(' + oovCnt + ' 个)');
ok(oovHtml.indexOf('静默丢弃') >= 0, '未登录字给出了「静默丢弃」警示');
ok(oovHtml.indexOf('了门，') >= 0, '警示里列出了模型实际看到的字（了门，）');

// 语料切换
var swErr = null;
try {
  sandbox.document.getElementById('fCorpus').value = 'wuxia';
  sandbox.document.getElementById('fCorpus').fire('change');
  sandbox.document.getElementById('fCorpus').value = 'custom';
  sandbox.document.getElementById('fCorpus').fire('change');
  sandbox.document.getElementById('fCustom').value = '很短的文本';
  sandbox.document.getElementById('fCustom').fire('input');
} catch (e) { swErr = e; }
ok(!swErr, '切换语料 / 自定义语料输入无异常', swErr ? String(swErr.stack).split('\n')[1] : '');

// 梯度检查按钮
var gcErr = null;
try { LAB.gradCheck(); } catch (e) { gcErr = e; }
ok(!gcErr, '「梯度检查」无异常', gcErr ? String(gcErr.stack).split('\n')[1] : '');
var notice = sandbox.document.getElementById('gradNotice').innerHTML;
ok(/反向传播实现正确|存在偏差/.test(notice), '梯度检查给出结论',
   notice.replace(/<[^>]*>/g, '').slice(0, 96));

// 配置预设
var psErr = null;
try { ['std', 'deep', 'big', 'fast'].forEach(function (p) { LAB.preset(p); }); }
catch (e) { psErr = e; }
ok(!psErr, '四档预设切换无异常', psErr ? String(psErr.stack).split('\n')[1] : '');

ok(alerts.length === 0, '全程没有触发 alert（没有意外错误路径）',
   alerts.length ? alerts.join(' / ') : '');

/* ------------------------------------------------------------------ */
console.log('\n' + '='.repeat(60));
console.log(fails === 0
  ? '✓ UI 端到端测试全部通过（' + passes + ' 项）'
  : '✗ 有 ' + fails + ' 项失败 / ' + passes + ' 项通过');
console.log('='.repeat(60));
process.exit(fails === 0 ? 0 : 1);
