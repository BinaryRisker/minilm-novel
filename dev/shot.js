// dev/shot.js —— 无头浏览器截图工具：加载实验台 → 自动训练 → 全页截图
// 零依赖（Node 22+ 自带 fetch / WebSocket）。
// 用法：
//   node dev/shot.js [输出.png] [最长训练等待毫秒]
//   node dev/shot.js docs/img/training.png 260000
//
// 原理：Edge headless + CDP。页面加载后调用 window.__MiniLMLab.start()，
// 轮询 state.step 直到训练结束（state.running === false），再整页截图。

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const EDGE =
  process.env.EDGE_PATH ||
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const PORT = 9333;
const PAGE_URL =
  'file:///' + path.resolve(__dirname, '..', 'index.html').replace(/\\/g, '/');
const OUT =
  process.argv[2] || path.resolve(__dirname, '..', 'docs', 'img', 'training.png');
const BUDGET_MS = parseInt(process.argv[3] || '260000', 10);

function cdp(ws) {
  let seq = 0;
  const pending = new Map();
  ws.addEventListener('message', (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const done = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? done.reject(new Error(JSON.stringify(msg.error))) : done.resolve(msg.result);
    }
  });
  return (method, params = {}) =>
    new Promise((resolve, reject) => {
      const id = ++seq;
      pending.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params }));
    });
}

async function waitForTargetWs() {
  for (let i = 0; i < 40; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
      if (page) return page.webSocketDebuggerUrl;
    } catch (_) {
      /* 浏览器还没起来，继续等 */
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error('无法连接无头浏览器调试端口');
}

(async () => {
  const browser = spawn(
    EDGE,
    [
      '--headless=new',
      '--disable-gpu',
      '--hide-scrollbars',
      `--remote-debugging-port=${PORT}`,
      '--user-data-dir=' + path.join(process.env.TEMP || '.', 'minilm-shot-profile'),
      'about:blank',
    ],
    { stdio: 'ignore' }
  );

  try {
    const ws = new WebSocket(await waitForTargetWs());
    await new Promise((res, rej) => {
      ws.addEventListener('open', res);
      ws.addEventListener('error', rej);
    });
    const send = cdp(ws);
    await send('Page.enable');
    await send('Runtime.enable');

    await send('Page.navigate', { url: PAGE_URL });
    await new Promise((r) => setTimeout(r, 3000)); // 等页面脚本装配完

    const hasHook = await send('Runtime.evaluate', {
      expression:
        'typeof window.__MiniLMLab === "object" && typeof window.__MiniLMLab.start === "function"',
      returnByValue: true,
    });
    if (!hasHook.result.value) throw new Error('页面没有 __MiniLMLab 钩子');

    await send('Runtime.evaluate', {
      expression:
        'window.__MiniLMLab.start();' +
        // 兜底驱动：无头模式下 rAF 可能被节流，用 setTimeout 跑同一个 tick
        '(function drive(){ if(window.__MiniLMLab.state.running){' +
        ' window.__MiniLMLab.tick(); setTimeout(drive, 10);} })()',
    });
    console.log('训练已启动，轮询进度…');

    const t0 = Date.now();
    while (Date.now() - t0 < BUDGET_MS) {
      await new Promise((r) => setTimeout(r, 5000));
      const st = await send('Runtime.evaluate', {
        expression:
          '(s => s ? JSON.stringify({step:s.step, steps:s.steps, running:s.running}) )(window.__MiniLMLab && window.__MiniLMLab.state)',
        returnByValue: true,
      });
      let info = null;
      try { info = JSON.parse(st.result.value); } catch (_) {}
      if (info) {
        console.log(`step ${info.step}/${info.steps} running=${info.running}`);
        if (!info.running && info.step >= info.steps) break;
      }
    }

    // 给最后一帧渲染留时间
    await new Promise((r) => setTimeout(r, 1500));

    // 整页截图：用 metrics override 把视口撑到页面全高
    const height = await send('Runtime.evaluate', {
      expression: 'document.body.scrollHeight',
      returnByValue: true,
    });
    const H = Math.min(Math.max(height.result.value || 3200, 1600), 8000);
    await send('Emulation.setDeviceMetricsOverride', {
      width: 1440,
      height: H,
      deviceScaleFactor: 1.5,
      mobile: false,
    });
    await new Promise((r) => setTimeout(r, 1200));
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    fs.mkdirSync(path.dirname(OUT), { recursive: true });
    fs.writeFileSync(OUT, Buffer.from(shot.data, 'base64'));
    console.log(`已保存 ${OUT}（页面高度 ${H}px）`);
  } finally {
    browser.kill();
  }
})().catch((e) => {
  console.error('截图失败：', e.message);
  process.exit(1);
});
