"""诊断脚本：打印 JS 与 PT 的关键参数和梯度样本，看方向/量级。"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from minilm_torch import MiniLM, build_vocab
from verify_parity import fixture_to_state_dict  # type: ignore

torch.set_default_dtype(torch.float64)

with open(os.path.join(HERE, 'runs', 'parity_fixture.json'), 'r', encoding='utf-8') as f:
    fix = json.load(f)
cfg = dict(fix['meta']['cfg'])
cfg['V'] = 9
model = MiniLM(9, cfg)
model.load_state_dict(fixture_to_state_dict(fix['init_weights'], cfg))

tokens = torch.tensor(fix['meta']['batch_tokens'], dtype=torch.long).reshape(2, cfg['ctx'])
targets = torch.tensor(fix['meta']['batch_targets'], dtype=torch.long).reshape(2, cfg['ctx'])
logits_pt, loss_pt = model(tokens, targets)
print(f'PT loss = {loss_pt.item():.12f}   JS loss = {fix["forward"]["loss"]:.12f}')

# 1) 看 dL/dlogits 是否对得上
# PyTorch 通过 autograd 内部计算
# 我们手工算一份做参考
B, T, V = 2, cfg['ctx'], cfg['V']
import math
logits = logits_pt.reshape(-1, V).detach()
maxs = logits.max(dim=-1, keepdim=True).values
exps = (logits - maxs).exp()
probs = exps / exps.sum(dim=-1, keepdim=True)
targets_flat = targets.reshape(-1)
js_logits = torch.tensor(fix['forward']['logits'], dtype=torch.float64).reshape(-1, V)
js_maxs = js_logits.max(dim=-1, keepdim=True).values
js_exps = (js_logits - js_maxs).exp()
js_probs = js_exps / js_exps.sum(dim=-1, keepdim=True)
dL_pt = probs.clone()
dL_pt[range(B*T), targets_flat] -= 1
dL_pt /= (B*T)
dL_js = js_probs.clone()
dL_js[range(B*T), targets_flat] -= 1
dL_js /= (B*T)
print(f'dL/dlogits  max|Δ|={(dL_pt-dL_js).abs().max().item():.2e}')

# 2) emb.weight grad: 简单看几个值
model.zero_grad()
loss_pt.backward()
g_emb_pt = model.emb.weight.grad.detach().clone()  # [V, D]
g_emb_js = fixture_to_state_dict(fix['backward']['raw'], cfg)['emb.weight']  # [V, D] (转置过)
print(f'\nemb.weight.grad (shape {tuple(g_emb_pt.shape)}):')
print(' token  d_pt        d_js        Δ         rel')
nz = 0
for i in range(V):
    for j in range(8):
        a = g_emb_pt[i, j].item()
        b = g_emb_js[i, j].item()
        if abs(a) > 1e-3 or abs(b) > 1e-3 or abs(a-b) > 1e-3:
            print(f'  [{i},{j}]  {a:+.4e}  {b:+.4e}  {a-b:+.4e}')
            nz += 1
            if nz > 12: break
    if nz > 12: break
print(f'  RMS PT = {g_emb_pt.pow(2).mean().sqrt().item():.4e}   RMS JS = {g_emb_js.pow(2).mean().sqrt().item():.4e}')

# 3) 看一个 weight: Wq
g_wq_pt = model.blocks[0].Wq.weight.grad.detach().clone()
g_wq_js = fixture_to_state_dict(fix['backward']['raw'], cfg)['blocks.0.Wq.weight']
print(f'\nWq.weight.grad shape {tuple(g_wq_pt.shape)} (PyTorch) vs {tuple(g_wq_js.shape)} (JS→PT)')
print(f'  RMS PT = {g_wq_pt.pow(2).mean().sqrt().item():.4e}   RMS JS = {g_wq_js.pow(2).mean().sqrt().item():.4e}')
print(f'  Wq.weight (PT):  first row[:4] = {model.blocks[0].Wq.weight[0, :4].tolist()}')
print(f'  Wq.weight (JS):  first row[:4] = {g_wq_js[0, :4].tolist()}  (already .t()\'d to PT orientation)')
print(f'  Σ PT = {g_wq_pt.sum().item():+.4e}   Σ JS = {g_wq_js.sum().item():+.4e}')
print(f'  PT[0:2,0:4] =\n{g_wq_pt[:2, :4]}')
print(f'  JS[0:2,0:4] =\n{g_wq_js[:2, :4]}')
