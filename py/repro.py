""" ============================================================================
py/repro.py — 可复现性基础设施：统一种子 + RNG 状态存取
----------------------------------------------------------------------------
背景（本次自检发现的问题）：
  1. py/train.py 在 `torch.manual_seed(seed)` **之前**就构造了 MiniLM ——
     模型的初始权重来自 torch 默认 RNG（按进程顺序变化），seed 管不到它。
     表现：同 seed 同参数跑两次，初始权重不同 → 结论无法逐位复核。
  2. 训练里同时存在两套 RNG：`random.Random(seed)`（决定 make_batch 的窗口起点）
     与 torch 的全局 RNG（决定 dropout/采样等）。checkpoint 里一个都没存，
     所以续训不可能与直训等价。
  3. py/eval_story.py 用 `torch.multinomial` 采样解码但不设种子 →
     同一 checkpoint 两次生成的故事不同，评测结果不可复核。

本模块把这三件事收敛成两个原语：
  set_global_seed(seed)  —— 一次性把所有涉及的 RNG 设成同一 seed（可选 numpy）
  rng_state()/set_rng_state(s) —— 把 python-random 与 torch 的 RNG 状态打包存取，
                                 供 checkpoint 完整续训使用

用法：
  from repro import set_global_seed, rng_state, set_rng_state
  set_global_seed(1234)                 # 必须在建模型/建 optimizer 之前调用
  ckpt['rng'] = rng_state()             # 存
  set_rng_state(ckpt['rng'])            # 续训时恢复
============================================================================ """
from __future__ import annotations
import random
from typing import Any, Dict, Optional

import torch

__all__ = ['set_global_seed', 'rng_state', 'set_rng_state', 'random_seed_bundle', 'SEED_KEYS']


def set_global_seed(seed: int, deterministic_torch: bool = False) -> Dict[str, Any]:
    """把所有 RNG 统一到同一 seed。

    调用时机很关键：必须早于任何 nn.Module 构造（即早于模型初始化），
    否则初始权重不受控 —— 这是本模块存在的主要理由。

    deterministic_torch=True 时还会打开 cudnn 确定性开关（本项目的实验都在 CPU
    上跑，默认关闭以免拖慢）。
    """
    seed = int(seed)
    # 1) python 标准库 random（make_batch 的窗口起点走这条）
    random.seed(seed)
    # 2) numpy（可选依赖：训练本身不依赖 numpy，但绘图/统计脚本会用）
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    # 3) torch（模型初始化、采样解码、dropout 等）
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    return {'seed': seed, 'torch': torch.initial_seed()}


def random_seed_bundle(seed: int) -> random.Random:
    """给 make_batch 用的独立 Random。注意它与 torch 全局 RNG 是**两条独立流**：
    只设一个不等于两条都对齐，checkpoint 必须两条都存。"""
    return random.Random(int(seed))


SEED_KEYS = ('python_random', 'torch')


def rng_state() -> Dict[str, Any]:
    """打包当前 RNG 状态（python-random 的 getstate 与 torch 的 get_rng_state）。"""
    st: Dict[str, Any] = {
        'python_random': random.getstate(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st['torch_cuda'] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(state: Optional[Dict[str, Any]]) -> bool:
    """恢复 RNG 状态；state 为 None（旧 checkpoint）时返回 False，由调用方决定是否告警。

    兼容性：旧 checkpoint 没有 'rng' 字段，旧格式里的 python_random 是 tuple
    （torch.save 直接存 tuple 也能读回），这里原样接受。
    """
    if not state:
        return False
    try:
        if state.get('python_random') is not None:
            random.setstate(state['python_random'])
        if state.get('torch') is not None:
            torch_state = state['torch']
            # pickle/unpickle 后 ByteTensor 可能变成普通 Tensor；强制恢复 ByteTensor 类型
            if not isinstance(torch_state, torch.ByteTensor):
                torch_state = torch.from_numpy(
                    torch_state.detach().cpu().numpy()
                    if isinstance(torch_state, torch.Tensor) else torch_state
                ).byte()
            torch.set_rng_state(torch_state)
        if state.get('torch_cuda') and torch.cuda.is_available():
            cuda_states = state['torch_cuda']
            fixed = []
            for cs in cuda_states:
                if not isinstance(cs, torch.ByteTensor):
                    cs = torch.from_numpy(
                        cs.detach().cpu().numpy() if isinstance(cs, torch.Tensor) else cs
                    ).byte()
                fixed.append(cs)
            torch.cuda.set_rng_state_all(fixed)
    except Exception as e:  # 状态格式不兼容时不要把训练整挂掉
        print(f'[!] 恢复 RNG 状态失败（{type(e).__name__}: {e}），续训结果预计与直训不等价')
        return False
    return True
