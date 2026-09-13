"""诊断：找出 emb.weight.grad 的最大差值点。"""
import json
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from minilm_torch import MiniLM, build_vocab
from verify_parity import fixture_to_state_dict

torch.set_default_dtype(torch.float64)

with open(os.path.join(HERE, 'runs', 'parity_fixture.json'), 'r', encoding='utf-8') as f:
    fix = json.load(f)
cfg = dict(fix['meta']['cfg'])
cfg['V'] = 9
model = MiniLM(9, cfg)
model.load_state_dict(fixture_to_state_dict(fix['init_weights'], cfg))

tokens = torch.tensor(fix['meta']['batch_tokens'], dtype=torch.long).reshape(2, cfg['ctx'])
targets = torch.tensor(fix['meta']['batch_targets'], dtype=torch.long).reshape(2, cfg['ctx'])
logits, loss = model(tokens, targets)
model.zero_grad()
loss.backward()
g_pt = model.emb.weight.grad.detach().clone()
g_js = fixture_to_state_dict(fix['backward']['raw'], cfg)['emb.weight']

d = (g_pt - g_js).abs()
print(f'shape: PT {g_pt.shape}  JS {g_js.shape}')
print(f'PT:  min={g_pt.min().item():+.4e}  max={g_pt.max().item():+.4e}  RMS={g_pt.pow(2).mean().sqrt().item():.4e}')
print(f'JS:  min={g_js.min().item():+.4e}  max={g_js.max().item():+.4e}  RMS={g_js.pow(2).mean().sqrt().item():.4e}')
print(f'|Δ|: min={d.min().item():+.4e}  max={d.max().item():+.4e}  mean={d.mean().item():+.4e}')

# 找 max Δ 所在位置
flat_idx = d.flatten().argmax().item()
i, j = divmod(flat_idx, 8)
print(f'\nmax |Δ| at [{i},{j}]:')
print(f'  PT = {g_pt[i,j].item():+.10e}')
print(f'  JS = {g_js[i,j].item():+.10e}')
print(f'  Δ  = {d[i,j].item():+.4e}')

# 也看 Wq
g_wq_pt = model.blocks[0].Wq.weight.grad.detach().clone()
g_wq_js = fixture_to_state_dict(fix['backward']['raw'], cfg)['blocks.0.Wq.weight']
d2 = (g_wq_pt - g_wq_js).abs()
print(f'\nWq.weight.grad  shape: PT {g_wq_pt.shape}  JS {g_wq_js.shape}')
print(f'  max |Δ| = {d2.max().item():.4e}  mean = {d2.mean().item():.4e}')
flat_idx = d2.flatten().argmax().item()
i, j = divmod(flat_idx, 8)
print(f'  max at [{i},{j}]: PT={g_wq_pt[i,j].item():+.4e}  JS={g_wq_js[i,j].item():+.4e}')
