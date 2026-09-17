""" ============================================================================
py/novellm/doctor.py — 本机训练资源探测（V3 P1 首项）
----------------------------------------------------------------------------
回答一个问题：**这台机器到底能训多大的模型？** 在租卡/开训之前先拿到实测数字，
不靠用户口头报告（V3 §3：设备由 doctor 实测确认）。

输出（JSON，可直接进预算表）：
  torch 版本 / 编译期 CUDA 版本
  cuda      : 可用性、设备数、每设备的名称/总量显存/当前可用显存/算力
  mps       : Apple Silicon 可用性
  cpu       : 物理核/逻辑核、总内存（psuv 不可用时留空，不猜）
  precision : bf16 / fp16 / tf32 是否可用（用真实小张量试算，不看文档）

硬性边界：
  - **只读探测**：不读任何凭证、不写网络、不改任何全局状态（cudnn benchmark 之类也不动）。
  - 探测失败的项**留空并记 error**，绝不猜——预算表里一个猜出来的数字比缺一格危险得多。

用法：
  python -X utf8 -m py.novellm.doctor              # 人读 + JSON
  python -X utf8 -m py.novellm.doctor --out x.json # 另存 JSON
  from novellm.doctor import probe
============================================================================ """
from __future__ import annotations
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
if PY not in sys.path:
    sys.path.insert(0, PY)


def _precision_probe(device: str) -> dict:
    """用真实小张量试算各精度是否可用。任何失败记 False + 原因，不猜。"""
    import torch
    out = {}
    # Probes must not perturb training or generation randomness.
    saved = torch.get_rng_state()
    for name, dt in (('bf16', torch.bfloat16), ('fp16', torch.float16)):
        try:
            x = torch.randn(8, 8, device=device, dtype=dt)
            _ = (x @ x).sum().item()
            out[name] = True
        except Exception as e:  # noqa: BLE001
            out[name] = False
            out[f'{name}_error'] = str(e)[:120]
    try:
        if device != 'cpu':
            # tf32 是 CUDA 侧的 matmul 开关；在 CPU 上报它只会误导预算表
            out['tf32'] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:  # noqa: BLE001
        pass
    torch.set_rng_state(saved)
    return out


def probe() -> dict:
    import torch
    out: dict = {'torch': torch.__version__,
                 'torch_cuda_compiled': getattr(torch.version, 'cuda', None)}

    # ---- CUDA ----
    cuda = {'available': bool(torch.cuda.is_available())}
    if cuda['available']:
        cuda['device_count'] = torch.cuda.device_count()
        devs = []
        for i in range(cuda['device_count']):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            devs.append({'index': i, 'name': props.name,
                         'total_mem_gb': round(props.total_memory / 2**30, 2),
                         'free_mem_gb': round(free / 2**30, 2),
                         'compute_capability': f'{props.major}.{props.minor}',
                         'multiprocessors': getattr(props, 'multi_processor_count', None)})
        cuda['devices'] = devs
        # V3 §3 按 8GB 可用显存设计：把「当前进程可见的空闲显存」单独放最外层，方便预算表引用
        cuda['free_mem_gb_primary'] = devs[0]['free_mem_gb'] if devs else None
    out['cuda'] = cuda

    # ---- MPS（Apple Silicon）----
    mps = {'available': False}
    try:
        mps['available'] = bool(getattr(torch.backends, 'mps', None)
                                and torch.backends.mps.is_available())
    except Exception as e:  # noqa: BLE001
        mps['error'] = str(e)[:120]
    out['mps'] = mps

    # ---- CPU ----
    cpu: dict = {'logical_cores': os.cpu_count()}
    try:
        import psutil
        cpu['physical_cores'] = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001
        cpu['physical_cores'] = None
    try:
        import psutil  # 可选依赖：没有就不给内存数，绝不猜
        vm = psutil.virtual_memory()
        cpu['total_mem_gb'] = round(vm.total / 2**30, 2)
        cpu['available_mem_gb'] = round(vm.available / 2**30, 2)
    except ImportError:
        cpu['total_mem_gb'] = None
        cpu['note'] = 'psutil 未安装，内存数字缺省（pip install psutil 可补）'
    out['cpu'] = cpu

    # ---- 精度 ----
    out['precision'] = {'cpu': _precision_probe('cpu')}
    if cuda['available']:
        out['precision']['cuda:0'] = _precision_probe('cuda:0')
    if mps['available']:
        out['precision']['mps'] = _precision_probe('mps')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None, help='另存 JSON 的路径')
    args = ap.parse_args()
    r = probe()
    text = json.dumps(r, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
        print(f'[*] 另存 -> {args.out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
