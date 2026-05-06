"""
Puzzle 03: Outer Vector Add
==============
这个 puzzle 里，我们正式进入 2D world。

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import test_puzzle

"""
考虑一个 outer vector addition operation。它的结果是一个 matrix，其中
每个元素 `(i, j)` 都等于 `A[i] + B[j]`。

和前一个 puzzle 相比，最大的区别是现在 `C` 变成了 2D tensor，
而且 `A` 和 `B` 这两个 buffer 的迭代方式也不一样，所以 dataflow 也会略有变化。

但要记住，任何 N 维 tensor 在 memory 里本质上都可以看成 1D tensor。
因此关键只是把 indexing 处理正确。

03-1: Outer vector addition.

输入:
    A: Tensor([N,], float16)  # 输入 tensor
    B: Tensor([M,], float16)  # 输入 tensor
    N: int   # tensor 的大小，1 <= N <= 8192
    M: int   # tensor 的大小，1 <= M <= 8192

输出:
    C: [N, M]  # 输出 tensor

定义:
    for i in range(N):
        for j in range(M):
            C[i, j] = A[i] + B[j]
"""


def ref_outer_add(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 1
    assert len(B.shape) == 1
    assert A.dtype == B.dtype == torch.float16
    return torch.add(input=A[:, None], other=B[None, :])


@tilelang.jit
def tl_outer_add(A, B, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    dtype = T.float16
    A: T.Tensor((N,), dtype)
    B: T.Tensor((M,), dtype)
    C = T.empty((N, M), dtype)

    with T.Kernel(N // BLOCK_N, M // BLOCK_M, threads=256) as (pid_n, pid_m):
        n_idx = pid_n * BLOCK_N
        m_idx = pid_m * BLOCK_M

        for i, j in T.Parallel(BLOCK_N, BLOCK_M):
            C[n_idx + i, m_idx + j] = A[n_idx + i] + B[m_idx + j]

    return C


def run_outer_add():
    print("\n=== Outer Vector Add ===\n")
    N = 8192
    M = 4096
    BLOCK_N = 1024
    BLOCK_M = 1024
    test_puzzle(
        tl_outer_add,
        ref_outer_add,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )


if __name__ == "__main__":
    run_outer_add()
