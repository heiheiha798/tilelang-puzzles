"""
Puzzle 05: Reduce Sum
==============
这个 puzzle 会带你学习如何在 TileLang 里做 reduce。

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
前面的例子里我们已经做过 broadcasting。现在来看 reduction 应该怎么写。幸运的是，
我们不需要自己实现所有 reduction 细节，因为 TileLang 已经提供了内建 TileOp。
在这之前，我们见到的 TileOp 基本只有 `T.copy`，但你应该已经体会到，仅靠
`T.copy` 和 `T.Parallel`，其实就能搭出很多东西。

提示:
1. 对 reduction 来说，可以使用 `T.reduce` 和 `T.reduce_xxx`，其中 `xxx`
代表具体的 reduction operation，比如 `T.reduce_sum`。为了效率，最好把这些
TileOp 放在 fragment buffer 上执行，而不是直接在 global memory 上做。
2. 这道题可能需要 serial loop，可以用 `T.Serial` 来创建。
3. 为了 numerical stability，这里暂时把数据类型提升到 `float32`。

05-1: Reduce sum.

输入:
    A: Tensor([N, M], float32)  # 输入 tensor
    B: Tensor([N,], float32)  # 输入 tensor
    N: int   # tensor 的大小，1 <= N <= 4096
    M: int   # tensor 的大小，1 <= M <= 16384

输出:
    B: Tensor([N,], float32)  # 输出 tensor

定义:
    for i in range(N):
        B[i] = 0
        for j in range(M):
            B[i] += A[i, j]
"""


def ref_reduce_sum(A: torch.Tensor):
    assert len(A.shape) == 2
    assert A.dtype == torch.float32
    return torch.sum(A, dim=1)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_reduce_sum(A, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    dtype = T.float32
    A: T.Tensor((N, M), dtype)
    B = T.empty((N,), dtype)

    with T.Kernel(N // BLOCK_N, threads=256) as pid_n:
        A_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        B_local = T.alloc_fragment((BLOCK_N,), dtype)

        T.clear(B_local)

        for m_blk_id in T.Serial(M // BLOCK_M):
            T.copy(A[pid_n * BLOCK_N, m_blk_id * BLOCK_M], A_local)
            T.reduce_sum(A_local, B_local, dim=1, clear=False)

        T.copy(B_local, B[pid_n * BLOCK_N])

    return B


def run_reduce_sum():
    print("\n=== Reduce Sum ===\n")
    N = 4096
    M = 16384
    BLOCK_N = 16
    BLOCK_M = 128
    test_puzzle(
        tl_reduce_sum,
        ref_reduce_sum,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )
    bench_puzzle(
        tl_reduce_sum,
        ref_reduce_sum,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
        bench_torch=True,
    )


if __name__ == "__main__":
    run_reduce_sum()
