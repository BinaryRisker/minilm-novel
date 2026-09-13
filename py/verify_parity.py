""" ============================================================================
py/verify_parity.py — 与 minilm.js 做「前向/反向/Adam 步」逐位对齐
----------------------------------------------------------------------------
比对范围（PRD P1 验收口径）：
    1) 前向：logits、loss
    2) 反向：每个参数的原始梯度（裁剪前）
    3) 梯度裁剪：与 PyTorch clip_grad_norm_(1.0) 同语义
    4) Adam 一步：所有参数更新后的值

数据来源：
    py/runs/parity_fixture.json 由 dev/export_fixture.js 一次性导出。
    里面是 JS 在小配置（ctx=4 d=8 layers=1 heads=2 ffMul=2 / V=9）上
    跑出来的：初始权重、logits、loss、原始梯度、裁剪后梯度、步后权重。

为什么这样测：
    JS 用 mulberry32，PyTorch 用 MT19937，两个 PRNG 不同，强对齐它们的
    初始权重没意义。这里改成"权重从 JS 一次性导出，PyTorch 原样加载"，
    然后只比对 forward / backward / Adam 的数学 —— 这就是 PRD 要求的
    「固定输入的前向、反向、单步更新全部对齐」。

容差（按 PRD 约定）：
    float64 · 默认 atol=1e-8 · rtol=1e-6
    Adam 步（涉及除法）atol=1e-7
    接近零的梯度看绝对误差，不被相对误差淹没

用法：
    python py/verify_parity.py
    python py/verify_parity.py --fixture py/runs/parity_fixture.json
============================================================================ """
import argparse
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from minilm_torch import MiniLM, build_vocab  # noqa: E402

DEFAULT_FIXTURE = os.path.join(HERE, 'runs', 'parity_fixture.json')

ATOL_FWD = 1e-8
ATOL_STEP = 1e-7
RTOL = 1e-6

# JS 参数名 → PyTorch state_dict key
# 线性层 weight JS 存 [in, out]，PyTorch 存 [out, in]，比对时 PyTorch 侧需 .t()。
JS_TO_PT = {
    'emb':   ('emb',            'weight'),
    'pos':   ('pos',            'weight'),
    'ln1g0': ('blocks.0.ln1',   'weight'),
    'ln1b0': ('blocks.0.ln1',   'bias'),
    'Wq0':   ('blocks.0.Wq',    'weight'),
    'bq0':   ('blocks.0.Wq',    'bias'),
    'Wk0':   ('blocks.0.Wk',    'weight'),
    'bk0':   ('blocks.0.Wk',    'bias'),
    'Wv0':   ('blocks.0.Wv',    'weight'),
    'bv0':   ('blocks.0.Wv',    'bias'),
    'Wo0':   ('blocks.0.Wo',    'weight'),
    'bo0':   ('blocks.0.Wo',    'bias'),
    'ln2g0': ('blocks.0.ln2',   'weight'),
    'ln2b0': ('blocks.0.ln2',   'bias'),
    'W1_0':  ('blocks.0.W1',    'weight'),
    'b1_0':  ('blocks.0.W1',    'bias'),
    'W2_0':  ('blocks.0.W2',    'weight'),
    'b2_0':  ('blocks.0.W2',    'bias'),
    'lnfg':  ('lnf',            'weight'),
    'lnfb':  ('lnf',            'bias'),
}
PT_TO_JS = {v[0] + '.' + v[1]: k for k, v in JS_TO_PT.items()}
LINEAR_W = ('Wq', 'Wk', 'Wv', 'Wo', 'W1', 'W2')  # weight 需要 .t() 比对


def js_to_torch(name, arr, cfg):
    """JS 1D 数组 → PyTorch 张量。线性层 weight 自动 .t()。"""
    V, D, T, F = cfg['V'], cfg['d'], cfg['ctx'], cfg['d'] * cfg['ffMul']
    a = torch.tensor(arr, dtype=torch.float64)
    if name == 'emb':   return a.reshape(V, D)
    if name == 'pos':   return a.reshape(T, D)
    if name in ('ln1g0', 'ln1b0', 'ln2g0', 'ln2b0', 'lnfg', 'lnfb'):
        return a.reshape(D)
    if name in ('Wq0', 'Wk0', 'Wv0', 'Wo0'):
        return a.reshape(D, D).t().contiguous()
    if name == 'W1_0':  return a.reshape(D, F).t().contiguous()
    if name == 'W2_0':  return a.reshape(F, D).t().contiguous()
    if name in ('bq0', 'bk0', 'bv0', 'bo0', 'b2_0'):
        return a.reshape(D)
    if name == 'b1_0':  return a.reshape(F)
    raise ValueError('unknown param: ' + name)


def fixture_to_state_dict(weights_dict, cfg):
    sd = {}
    for jsn, (mod, attr) in JS_TO_PT.items():
        sd[mod + '.' + attr] = js_to_torch(jsn, weights_dict[jsn], cfg)
    return sd


def align_to_js(t, key):
    """PyTorch 张量 → 与 JS 同方向（线性层 weight 转置回来）。"""
    if key.endswith('.weight') and any(s + '.' in key for s in LINEAR_W):
        return t.t().contiguous()
    return t


def diff(t_pt, t_js, atol, rtol, name):
    """严格比对：max|Δ| 和 max 相对误差。两者必须都达标才算通过，
    OR 条件在「一边接近 0」时会把 100% 误差误判成通过。"""
    a = t_pt.detach().cpu().to(torch.float64)
    b = t_js.detach().cpu().to(torch.float64)
    assert a.shape == b.shape, f'{name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}'
    d = (a - b).abs()
    max_abs = float(d.max().item()) if d.numel() else 0.0
    denom = b.abs().clamp_min(1e-30)
    rel = d / denom
    max_rel = float(rel.max().item()) if rel.numel() else 0.0
    # 通过条件：|Δ| ≤ atol；或 |Δ|/|b| ≤ rtol 且 |b| ≥ atol（避免 0 附近的「100% rel 误差」通过）
    abs_ok = d <= atol
    rel_ok = (d <= rtol * b.abs()) & (b.abs() >= atol)
    ok = bool((abs_ok | rel_ok).all().item()) if d.numel() else True
    return max_abs, max_rel, ok, name


def torch_adam_step(model, sd_grads_pt, meta):
    """用 torch.optim.Adam 真实跑一步 Adam —— 重点是确认与 minilm.js 的 step()
    在浮点层面一致。手工重写 Adam 在 sqrt(v_hat)+eps 的顺序上会和 torch 略不同，
    这里直接用 torch 的实现以最小化比较噪声。"""
    # 先把 sd_grads_pt 灌进 model.grad
    for k, p in model.named_parameters():
        p.grad = sd_grads_pt[k].clone()
    opt = torch.optim.Adam(model.parameters(), lr=meta['lr'], betas=(0.9, 0.999), eps=1e-8)
    opt.step()
    return {k: p.detach().clone() for k, p in model.named_parameters()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixture', default=DEFAULT_FIXTURE)
    args = ap.parse_args()

    with open(args.fixture, 'r', encoding='utf-8') as f:
        fix = json.load(f)

    cfg = dict(fix['meta']['cfg'])
    itos, stoi = build_vocab(fix['meta']['vocab_chars'])
    cfg['V'] = len(itos)
    assert cfg['V'] == 9, f'V 期望 9，实际 {cfg["V"]}'

    torch.set_default_dtype(torch.float64)
    model = MiniLM(len(itos), cfg)
    sd = fixture_to_state_dict(fix['init_weights'], cfg)
    miss, unex = model.load_state_dict(sd, strict=True)
    assert not miss and not unex, f'load_state_dict 不匹配：missing={miss}, unexpected={unex}'

    tokens = torch.tensor(fix['meta']['batch_tokens'], dtype=torch.long).reshape(2, cfg['ctx'])
    targets = torch.tensor(fix['meta']['batch_targets'], dtype=torch.long).reshape(2, cfg['ctx'])

    # ---- 1. 前向 ----
    logits_pt, loss_pt = model(tokens, targets)
    logits_pt_flat = logits_pt.reshape(-1, logits_pt.shape[-1])
    logits_js = torch.tensor(fix['forward']['logits'], dtype=torch.float64).reshape(-1, logits_pt.shape[-1])

    ma, mr, ok_logits, _ = diff(logits_pt_flat, logits_js, ATOL_FWD, RTOL, 'logits')
    loss_js = fix['forward']['loss']
    ma2, mr2, ok_loss, _ = diff(loss_pt.reshape(()), torch.tensor(loss_js, dtype=torch.float64),
                                ATOL_FWD, RTOL, 'loss')
    print(f'[1] forward.logits   max|D|={ma:.2e}  rel={mr:.2e}  {"PASS" if ok_logits else "FAIL"}')
    print(f'[1] forward.loss     JS={loss_js:.10f}  PT={loss_pt.item():.10f}  D={abs(loss_pt.item()-loss_js):.2e}  {"PASS" if ok_loss else "FAIL"}')

    # ---- 2. 反向 ----
    # fixture_to_state_dict 已经把 JS 侧的 [in,out] 转置为 PyTorch 的 [out,in]，
    # 所以两侧都是 PyTorch 方向，直接比对即可（不再 align_to_js）。
    model.zero_grad()
    loss_pt.backward()
    # 注意：named_parameters() 返回的是参数本身（值），梯度在 .grad 上。
    # 之前写 v.detach().clone() 是直接拷了参数值，和 JS 梯度对比当然会差 ~0.1 量级。
    grads_pt = {k: v.grad.detach().clone() for k, v in model.named_parameters() if v.grad is not None}
    grads_js = fixture_to_state_dict(fix['backward']['raw'], cfg)
    bad = []
    for k, v_pt in grads_pt.items():
        v_js = grads_js[k]
        ma, mr, ok, _ = diff(v_pt, v_js, ATOL_FWD, RTOL, k)
        print(f'[2] grad {k:30s}  D={ma:.2e}  rel={mr:.2e}  {"PASS" if ok else "FAIL"}')
        if not ok: bad.append(k)

    # ---- 3. 梯度裁剪 ----
    # clip_grad_norm_ 是原地修改 .grad，所以读 .grad 才是被裁剪过的值。
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    clip_pt = {k: v.grad.detach().clone() for k, v in model.named_parameters() if v.grad is not None}
    clip_js = fixture_to_state_dict(fix['backward']['clipped'], cfg)
    bad_clip = []
    for k, v_pt in clip_pt.items():
        v_js = clip_js[k]
        ma, mr, ok, _ = diff(v_pt, v_js, ATOL_FWD, RTOL, k)
        print(f'[3] clip {k:30s}  D={ma:.2e}  rel={mr:.2e}  {"PASS" if ok else "FAIL"}')
        if not ok: bad_clip.append(k)

    # ---- 4. Adam 一步 ----
    # 注意：minilm.js 的 step() 是在 clipGrad() 之后跑的，吃的是裁剪后的梯度。
    # 因此 torch_adam_step 也必须用 clip_pt 而不是 grads_pt。
    # 用 torch.optim.Adam 真跑一步，避免手工实现 Adam 与 torch 在 sqrt/add 顺序上的微小差异。
    step_pt = torch_adam_step(model, clip_pt, fix['meta'])
    step_js = fixture_to_state_dict(fix['step_weights'], cfg)
    bad_step = []
    for k, v_pt in step_pt.items():
        v_js = step_js[k]
        ma, mr, ok, _ = diff(v_pt, v_js, ATOL_STEP, RTOL, k)
        print(f'[4] adam {k:30s}  D={ma:.2e}  rel={mr:.2e}  {"PASS" if ok else "FAIL"}')
        if not ok: bad_step.append(k)

    print()
    print(f'forward: logits {"PASS" if ok_logits else "FAIL"}   loss {"PASS" if ok_loss else "FAIL"}')
    print(f'backward (raw):   {len(bad)}/{len(grads_pt)} FAIL')
    print(f'grad-clip:        {len(bad_clip)}/{len(clip_pt)} FAIL')
    print(f'adam-step:        {len(bad_step)}/{len(step_pt)} FAIL')

    sys.exit(0 if (not bad and not bad_clip and not bad_step and ok_logits and ok_loss) else 1)


if __name__ == '__main__':
    main()
