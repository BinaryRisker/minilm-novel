/* ============================================================================
 * build.js — 把多文件版本打包成单文件 index.html
 * ----------------------------------------------------------------------------
 * 为什么需要它：
 *   把 minilm.js / corpus.js 内联进 HTML 之后，index.html 就是一个
 *   完全自包含的文件 —— 双击即开、拷到手机上也能跑、不需要任何服务器。
 *   这正好呼应了"能在手机/嵌入式设备上运行"这个原始目标。
 *
 * 用法：node build.js
 *   输入：ui-src.html（外壳）+ minilm.js + corpus.js
 *   输出：index.html（单文件，可直接分发）
 *
 * 关键技术点：
 *   minilm.js 在加载时就把 REAL 定型为 Float32/Float64，所以要拿到
 *   float64 实例（梯度检查用）和 float32 实例（训练用），正常情况下
 *   需要重复加载模块两次。这里把模块体包成一个工厂函数 __makeMiniLM(useF64)，
 *   一次内联、按需实例化两份，避免源码重复。
 * ============================================================================ */

var fs = require('fs');
var path = require('path');

var dir = __dirname;
var read = function (f) { return fs.readFileSync(path.join(dir, f), 'utf8'); };

var shell = read('ui-src.html');
var minilm = read('minilm.js');
var corpus = read('corpus.js');

/* --- 1. 把 minilm.js 改造成工厂函数 -------------------------------------- */

// 精度开关从全局变量改成工厂入参
var before = minilm;
minilm = minilm.replace(
  /var REAL = \(global\.MINILM_USE_F64[\s\S]*?Float32Array;/,
  'var REAL = __USE_F64__ ? Float64Array : Float32Array;'
);
if (minilm === before) {
  throw new Error('未能改写 REAL 定义 —— minilm.js 的精度开关代码变了，请同步更新 build.js');
}

// IIFE 的挂载目标从 globalThis 换成工厂内部的私有容器
var tail = '})(typeof globalThis !== \'undefined\' ? globalThis : this);';
if (minilm.indexOf(tail) < 0) {
  throw new Error('未能找到 minilm.js 的 IIFE 结尾 —— 文件结构变了，请同步更新 build.js');
}
minilm = minilm.replace(tail, '})(__g);');

var factory =
  '/* ---- minilm.js（内联，包成工厂：一次代码，两份实例）---- */\n' +
  'function __makeMiniLM(__USE_F64__) {\n' +
  '  var __g = {};\n' +
  minilm +
  '\n  return __g.MiniLMKit;\n' +
  '}\n' +
  'window.MiniLMKitF64 = __makeMiniLM(true);   // 梯度检查用\n' +
  'window.MiniLMKit    = __makeMiniLM(false);  // 训练用\n';

/* --- 2. 替换外壳里的外部 script 标签 ------------------------------------ */

var marker =
  '<!-- 先加载两份 minilm 实例：\n' +
  '     第一份 float64（专供梯度检查——float32 的量化噪声会淹没小梯度分量），\n' +
  '     第二份 float32（训练用，更快）。两份都必须在主脚本之前拿到引用。 -->\n' +
  '<script>window.MINILM_USE_F64 = true;</script>\n' +
  '<script src="minilm.js"></script>\n' +
  '<script>window.MiniLMKitF64 = window.MiniLMKit; window.MINILM_USE_F64 = false;</script>\n' +
  '<script src="minilm.js"></script>\n' +
  '<script src="corpus.js"></script>';

if (shell.indexOf(marker) < 0) {
  throw new Error('未能在 ui-src.html 中找到外链 script 标记块 —— 外壳结构变了，请同步更新 build.js');
}

var inlined =
  '<!-- ===== 以下三块由 build.js 内联生成，请勿手工编辑 ===== -->\n' +
  '<script>\n' + factory + '\n' +
  '/* ---- corpus.js（内联）---- */\n' +
  corpus + '\n' +
  '</script>';

var out = shell.replace(marker, inlined);
out = out.replace('<title>MiniLM Lab · 微型中文小说模型实验台</title>',
  '<title>MiniLM Lab · 微型中文小说模型实验台</title>\n' +
  '<meta name="description" content="零依赖手写 Transformer，在浏览器里从零训练一个中文小说字符级小模型">');

fs.writeFileSync(path.join(dir, 'index.html'), out);
console.log('✓ 已生成 index.html（单文件自包含）');
console.log('  大小 ' + (out.length / 1024).toFixed(1) + ' KB');
console.log('  行数 ' + out.split('\n').length);
