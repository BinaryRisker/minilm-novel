"""Python/NumPy 侧矩阵乘法基准：走 BLAS（SIMD + 多线程 + 缓存分块）"""
import time
import numpy as np

print("numpy", np.__version__)
print("--- NumPy 矩阵乘法（同一台机器，走 BLAS）---")


def bench(M, K, N, iters):
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = A @ B  # 预热，触发 BLAS 初始化

    t0 = time.time()
    for _ in range(iters):
        C = A @ B
    dt = time.time() - t0

    flops = 2 * M * K * N * iters
    gf = flops / dt / 1e9
    print(f"NumPy {M}x{K} @ {K}x{N}  x{iters}  →  {dt:.2f}s  |  {gf:.2f} GFLOPS")
    return gf


a = bench(256, 192, 768, 3000)
b = bench(1024, 1024, 1024, 300)
print(f"平均: {(a + b) / 2:.2f} GFLOPS")
print()

# 标注 BLAS 后端
try:
    cfg = np.show_config(mode="dicts")
    blas = cfg.get("Build Dependencies", {}).get("blas", {})
    print("BLAS 后端:", blas.get("name", "unknown"), blas.get("version", ""))
except Exception:
    pass
