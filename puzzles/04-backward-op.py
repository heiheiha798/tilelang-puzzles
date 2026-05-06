"""
Puzzle 04: Backward Op
==============
这个 puzzle 会实现一个 backward operator，用来更好地理解 TileLang
如何处理自定义的计算需求。

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import test_puzzle

"""
考虑前一个 puzzle 中 fused vector multiplication + ReLU 的例子。
现在我们把第一个输入 `A` 扩展成 2D tensor，而 `B` 会像是被 "broadcast"
到这个 2D shape 上。

04-1: Fused multiplication ReLU with broadcasting.

输入:
    A: Tensor([N, M], float16)  # 输入 tensor
    B: Tensor([M,], float16)  # 输入 tensor
    N: int   # tensor 的大小，1 <= N <= 8192
    M: int   # tensor 的大小，1 <= M <= 8192

输出:
    C: Tensor([N, M], float16)  # 输出 tensor

定义:
    for i in range(N):
        for j in range(M):
            C[i, j] = max(0, A[i, j] * B[j])
"""


def ref_mul_relu_bcast(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 2
    assert len(B.shape) == 1
    assert A.shape[1] == B.shape[0]  # M
    assert A.dtype == B.dtype == torch.float16

    # torch.mul 会自动把 B broadcast 到 A 的 shape
    return (A * B).relu_()


@tilelang.jit
def tl_mul_relu_bcast(A, B, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    dtype = T.float16
    A: T.Tensor((N, M), dtype)
    B: T.Tensor((M,), dtype)
    C = T.empty((N, M), dtype)

    with T.Kernel(N // BLOCK_N, M // BLOCK_M, threads=256) as (pid_n, pid_m):
        n_idx = pid_n * BLOCK_N
        m_idx = pid_m * BLOCK_M

        for i, j in T.Parallel(BLOCK_N, BLOCK_M):
            product = A[n_idx + i, m_idx + j] * B[m_idx + j]
            C[n_idx + i, m_idx + j] = T.if_then_else(product > 0, product, 0)

    return C


def run_mul_relu_bcast():
    print("\n=== Fused Multiplication ReLU with Broadcasting ===\n")
    N = 8192
    M = 4096
    BLOCK_N = 64
    BLOCK_M = 64
    test_puzzle(
        tl_mul_relu_bcast,
        ref_mul_relu_bcast,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )


"""
现在来考虑上面这个 operation 的 backward。
我们要计算 loss 对 `A` 的 gradient，也就是在给定 `dC` 的情况下，
求出 `dA`。根据 chain rule，这个计算任务可以形式化写成：

04-2: Backward of fused multiplication ReLU with broadcasting.

输入:
    A: Tensor([N, M], float16)  # 输入 tensor
    B: Tensor([M,], float16)  # 输入 tensor
    dC: Tensor([N, M], float16)  # 对 C 的导数，derivative w.r.t. C
    N: int   # tensor 的大小，1 <= N <= 8192
    M: int   # tensor 的大小，1 <= M <= 8192

输出:
    dA: Tensor([N, M], float16)  # 对 A 的导数，derivative w.r.t. A

定义:
    for i in range(N):
        for j in range(M):
            dA[i, j] = dC[i, j] * B[j] * (A[i, j] * B[j] > 0)
"""


def ref_mul_relu_bwd(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor):
    assert len(A.shape) == 2
    assert len(B.shape) == 1
    assert A.shape[0] == dC.shape[0]  # N
    assert A.shape[1] == B.shape[0] == dC.shape[1]  # M
    assert len(dC.shape) == 2
    assert A.dtype == B.dtype == dC.dtype == torch.float16

    A = A.clone()
    B = B.clone()
    A.requires_grad_(True)
    B.requires_grad_(True)
    C = torch.relu(A * B)
    C.backward(dC)
    return A.grad


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_mul_relu_bwd(A, B, dC, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    dtype = T.float16
    A: T.Tensor((N, M), dtype)
    B: T.Tensor((M,), dtype)
    dC: T.Tensor((N, M), dtype)
    dA = T.empty((N, M), dtype)

    with T.Kernel(N // BLOCK_N, M // BLOCK_M, threads=256) as (pid_n, pid_m):
        n_idx = pid_n * BLOCK_N
        m_idx = pid_m * BLOCK_M

        for i, j in T.Parallel(BLOCK_N, BLOCK_M):
            product = A[n_idx + i, m_idx + j] * B[m_idx + j]
            grad = dC[n_idx + i, m_idx + j] * B[m_idx + j]
            dA[n_idx + i, m_idx + j] = T.if_then_else(product > 0, grad, 0)

    return dA


def run_mul_relu_bwd():
    print("\n=== Fused Multiplication ReLU with Broadcasting, Backward ===\n")
    N = 8192
    M = 4096
    BLOCK_N = 64
    BLOCK_M = 64
    # kernel = tl_mul_relu_bwd(N, M, dtype, BLOCK_N, BLOCK_M)
    # kernel.print_source_code()
    test_puzzle(
        tl_mul_relu_bwd,
        ref_mul_relu_bwd,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )


if __name__ == "__main__":
    run_mul_relu_bcast()
    run_mul_relu_bwd()
