""" ============================================================================
py/checkpoint.py — checkpoint v2 契约（从 py/train.py 抽出，供两条训练路径共用）
----------------------------------------------------------------------------
契约（与 P0 已验证的语义逐位一致，勿改字段）：
  {'version': 2, 'model': state_dict, 'opt': opt.state_dict, 'itos': [...],
   'meta': {...}, 'step': int, 'rng': 全局 RNG 状态（repro.rng_state）}

为什么抽成独立模块：
  py/train.py（旧路径）与 py/novellm/trainer.py（分片训练路径）必须共用**同一份**
  保存/恢复实现——两条路径各写一份，迟早有一份悄悄偏离契约（V3 P1：复用已验证状态契约）。
  load 不绑定具体模型类：模型由调用方用 meta['cfg'] 重建后 load_state_dict。

警告（旧格式）：缺 'opt'/'rng' 的 checkpoint 是修复前的旧格式——能读，但续训
**不会**与直训等价（Adam 动量从零重启、采样顺序不同），加载时会打印告警。
============================================================================ """
from __future__ import annotations
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from repro import rng_state  # noqa: E402

CKPT_VERSION = 2


def save_checkpoint(path: str, model, opt, itos: list, meta: dict, step: int) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        'version': CKPT_VERSION,
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'itos': itos,
        'meta': meta,
        'step': step,
        'rng': rng_state(),
    }, path)
    return path


def load_ckpt_raw(path: str, device: str = 'cpu') -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def load_state(model, opt, ckpt: dict, device: str = 'cpu'):
    """把 checkpoint 灌进已构建的 model/opt（模型由调用方按 meta['cfg'] 重建）。
    兼容旧格式：缺 opt/rng 时打告警并跳过。"""
    model.load_state_dict(ckpt['model'])
    model.to(device)
    if opt is not None and 'opt' in ckpt:
        opt.load_state_dict(ckpt['opt'])
    if 'opt' not in ckpt or 'rng' not in ckpt:
        print('[!] 旧格式 checkpoint：没有 optimizer 状态与 RNG 状态，'
              '续训不会与直训等价（Adam 动量从零重启）。')
    if 'rng' in ckpt:
        from repro import set_rng_state
        set_rng_state(ckpt['rng'])
    return ckpt.get('step', 0)
