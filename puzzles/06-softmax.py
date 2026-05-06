"""
Puzzle 06: Softmax
==============
Softmax 是这套教程里我们学习到的第一个 fundamental NN operator。

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

r"""
Softmax operator 比 reduce sum 更进一步。除了做求和之外，我们还需要使用 serial loop
来累积 summation，同时对每个元素执行 element-wise 的 exp operation。

注意，softmax 需要像 Python 里常见实现那样，以 numerically stable 的形式来计算。
具体做法是：在应用 exponential function 之前，先用每一行的最大值减掉该行所有元素。

提示:
1. 用 `T.fill` 来设置 buffer 的初始值。`T.clear` 默认会把所有元素清成零，
这不一定是你想要的行为。

3. 我们更推荐不用 `T.exp`，而是改用 `T.exp2`。你需要用到下面这个恒等式：

.. math::
    \exp(x) = 2^{\log_2(e) x}

常量 `log2_e` 已经提供好了。

进阶：尝试用 "Online Softmax" algorithm 来实现优化版 softmax。
这也是 FlashAttention algorithm 的核心思想之一。使用它之后，softmax 可以只用
两轮 pass / loop 来完成。

06-1: Softmax.

输入:
    A: Tensor([N, M], float32)  # 输入 tensor
    N: int   # tensor 的大小，1 <= N <= 4096
    M: int   # tensor 的大小，1 <= M <= 16384

输出:
    B: Tensor([N, M], float16)  # 输出 tensor

中间量:
    MAX: float32  # 每一行的最大值
    SUM: float32  # 每一行的求和值

定义:
    for i in range(N):
        SUM = 0
        MAX = -inf
        for j in range(M):
            MAX = max(A[i, j], MAX)
        for j in range(M):
            B[i, j] = exp(A[i, j] - MAX)
            SUM += B[i, j]
        for j in range(M):
            B[i, j] /= SUM
"""


def ref_softmax(A: torch.Tensor):
    assert len(A.shape) == 2
    assert A.dtype == torch.float32
    return torch.softmax(A, dim=1)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_softmax(A, BLOCK_N: int, BLOCK_M: int):
    log2_e = 1.44269504
    N, M = T.const("N, M")
    dtype = T.float32
    A: T.Tensor((N, M), dtype)
    B = T.empty((N, M), dtype)

    # TODO: Implement this function

    return B


def run_softmax():
    print("\n=== Softmax ===\n")
    N = 4096
    M = 16384
    BLOCK_N = 16
    BLOCK_M = 256
    test_puzzle(
        tl_softmax,
        ref_softmax,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )
    bench_puzzle(
        tl_softmax,
        ref_softmax,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
        bench_torch=True,
    )


if __name__ == "__main__":
    run_softmax()
