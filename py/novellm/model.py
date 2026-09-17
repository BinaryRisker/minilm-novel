""" ============================================================================
py/novellm/model.py — novellm 训练路径的模型构建 + 配置 schema（V3 P1）
----------------------------------------------------------------------------
职责边界：
  **数学不另写一份。** Transformer 本体直接复用 py/minilm_torch.MiniLM（L0 已对齐、
  梯度已验证的手写路径——V3：「保留手写数学路径作为教学与回归基准」）。
  本模块只做三件训练路径需要、而 MiniLM 不管的事：
    1) **配置 schema 与候选档**：micro / t30m 两个命名档位 + 参数量公式；
    2) **参数量断言**：公式 vs 实例化实数，不一致即抛——防止架构静默漂移
       （一个参数量对不上号的「30M」比没有更危险）；
    3) **根初始化谱系**：torch.manual_seed(seed) 后从随机初始化构建，
       provenance 里显式写「无外部权重」（V3 硬约束），供 checkpoint/meta 与模型卡引用。

参数量公式（与 MiniLM 结构一一对应，test_model.py 用实数对账锁死）：
  emb V·d ＋ pos ctx·d ＋ 每层 (4+2·ffMul)·d² + (9+ffMul)·d ＋ 末层 ln 2d
  （每层明细：attn 4d²+4d、两个 LN 4d、mlp 2d²f+d(f+1)；logits = x @ embᵀ 权重共享）

用法：
  from novellm.model import CANDIDATES, param_count, assert_param_count, build
  model, prov = build(CANDIDATES['micro'], V=1772, seed=42)
============================================================================ """
from __future__ import annotations
import os
import sys
from typing import Dict, Tuple

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
if PY not in sys.path:
    sys.path.insert(0, PY)

from minilm_torch import MiniLM  # noqa: E402

# 命名候选档（V3：微型 → 30M 候选推进；100M/325M 只留想法不落默认档）
CANDIDATES: Dict[str, Dict[str, int]] = {
    # 微型：本机 CPU 可过拟合检查、快速冒烟
    'micro': dict(ctx=256, d=128, layers=4, heads=4, ffMul=4),
    # 30M 候选：GPU 阶段训练的主力档（实际参数量随词表 ≈28-30M，build 后以断言为准）
    't30m': dict(ctx=1024, d=512, layers=8, heads=8, ffMul=4),
}


def param_count(cfg: Dict[str, int], V: int) -> int:
    """按 MiniLM 结构的解析参数量公式。"""
    d, L, f = cfg['d'], cfg['layers'], cfg['ffMul']
    return (V * d                      # token embedding（与输出共享）
            + cfg['ctx'] * d           # 绝对位置
            + L * ((4 + 2 * f) * d * d + (9 + f) * d)  # 每层：attn 4d²+4d · LN×2 4d · mlp 2d²f+d(f+1)
            + 2 * d)                   # 末层 LN


def assert_param_count(model, cfg: Dict[str, int], V: int) -> int:
    """公式 vs 实例化实数对账。不一致 = MiniLM 结构变了而公式没跟上
    （或反过来）——任何一种都必须停下来，而不是拿错参数量去填预算表。"""
    actual = sum(p.numel() for p in model.parameters())
    expect = param_count(cfg, V)
    if actual != expect:
        raise AssertionError(f'参数量对账失败：公式 {expect:,} ≠ 实际 {actual:,}'
                             f'（cfg={cfg}, V={V}）——架构或公式有一方漂移了')
    return actual


def build(cfg: Dict[str, int], V: int, seed: int,
          device: str = None) -> Tuple[torch.nn.Module, Dict]:
    """从随机初始化构建模型（根初始化谱系：seed → manual_seed → MiniLM）。
    **绝不加载任何外部权重**——provenance 里显式声明，供 checkpoint/meta/模型卡引用。"""
    torch.manual_seed(seed)                      # 根初始化：seed 是唯一来源
    model = MiniLM(V, dict(cfg))
    n = assert_param_count(model, cfg, V)
    prov = {
        'init': 'local_random_seed',
        'seed': seed,
        'no_external_weights': True,
        'param_count': n,
        'param_count_million': round(n / 1e6, 2),
        'cfg': dict(cfg),
        'V': V,
        'reference_impl': 'py/minilm_torch.MiniLM（L0 对齐路径）',
    }
    if device:
        model = model.to(device)
    return model, prov


def resolve_cfg(name_or_json: str) -> Dict[str, int]:
    """'micro' / 't30m' 命名档，或 JSON 覆盖（基于 micro 起 stub 再 update）。"""
    import json
    s = name_or_json.strip()
    if s.startswith('{'):
        cfg = dict(CANDIDATES['micro'])
        cfg.update(json.loads(s))
        return cfg
    if s in CANDIDATES:
        return dict(CANDIDATES[s])
    raise ValueError(f'未知配置档：{s}（可选 {sorted(CANDIDATES)} 或 JSON 覆盖）')
