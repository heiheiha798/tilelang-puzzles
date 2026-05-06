"""
Puzzle 09: Convolution
==============
Convolution 是 deep learning operator 里另一种非常基础的 computation pattern。

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
Convolution 本质上是通过 sliding window 的方式，在输入 tensor 上滑动并计算。
它最重要的特点是数据复用很强，因此通常需要非常小心地处理 memory access optimization。
不过在 TileLang 里，我们可以先忽略绝大多数底层细节，把注意力放在计算逻辑上。

在这个 puzzle 里，我们先去掉 "channel (C)" 维度，简化整个问题。
我们会先看 1D convolution，再进一步扩展到 2D。与此同时，这一章也会接触到 GPU 的
shared memory 应该怎么用。

09-1: 1D Convolution.

输入:
    X: Tensor([N, L], float16)  # 输入 tensor
    K: Tensor([KL,], float16)  # kernel tensor
    N: int   # batch size 维度，1 <= N <= 64
    H: int   # length 维度，1 <= H <= 1024
    KL: int  # kernel 长度，1 <= KH <= 32

输出:
    O: Tensor([N, L], float16)  # 输出 tensor

中间量:
    ACC: float32  # 累加器，accumulator

定义:
    for i in range(N):
        for j in range(L):
            ACC = 0
            for k in range(KL):
                if j + k < L:  # 边界检查，boundary check
                    ACC += X[i, j + k] * K[k]
            O[i, j] = ACC
"""


"""
我们可以先考虑一个 naive implementation。外层遍历 `N` 和 `L` 的 loop 可以通过
`T.Kernel` 分配到不同 block 上。至于遍历 `BLOCK_L` 的 loop，当前先用 serial
实现即可。要特别留意 convolution 里的 data dependency。
"""


def ref_conv1d(X: torch.Tensor, K: torch.Tensor):
    assert len(X.shape) == 2
    assert len(K.shape) == 1
    assert X.dtype == K.dtype == torch.float16

    # for i in range(N):
    #     for j in range(L):
    #         O[i, j] = 0
    #         for k in range(KL):
    #             if j + k < L:  # 边界检查，boundary check
    #                 O[i, j] += X[i, j + k] * K[k]

    N, L = X.shape
    KL = K.shape[0]

    padding_size = KL - 1
    X_padded = torch.nn.functional.pad(X.view(N, 1, L), (0, padding_size))

    return torch.conv1d(
        input=X_padded,
        weight=K.view(1, 1, KL),
    ).view(N, L)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_conv1d_naive(X, K, BLOCK_N: int, BLOCK_L: int):
    N, L, KL = T.const("N, L, KL")
    dtype = T.float16
    accum_dtype = T.float32
    X: T.Tensor((N, L), dtype)
    K: T.Tensor((KL,), dtype)
    O = T.empty((N, L), dtype)

    # TODO: Implement this function

    return O


def run_conv1d_naive():
    print("\n=== Convolution 1D Naive ===\n")
    N = 128
    L = 128
    BLOCK_N = 16
    BLOCK_L = 32
    KL = 32
    test_puzzle(
        tl_conv1d_naive,
        ref_conv1d,
        {"N": N, "L": L, "KL": KL, "BLOCK_N": BLOCK_N, "BLOCK_L": BLOCK_L},
    )


"""
naive 的 Conv1D 实现能工作，但效率不高。还记得上一题提到过 Tensor Core 和 `T.gemm`
吗？实际上，我们也可以通过一个叫 `im2col` 的变换，把 convolution 问题转成 GEMM。
核心想法是：把 convolution 改写成 matrix multiplication，其中输入 matrix 的每一行
对应输入 tensor 的一个局部 patch，而 kernel 则被 reshape 成另一个 matrix。
这样就能直接复用高度优化的 GEMM 实现。

为了避免这个 GEMM 退化成 GEMV，我们需要引入一个输出 channel 维度 `F`。

09-2: 1D Convolution with multiple output channels.

输入:
    X: Tensor([N, L], float16)  # 输入 tensor
    K: Tensor([KL, F], float16)  # kernel tensor
    N: int   # batch size 维度，1 <= N <= 64
    H: int   # length 维度，1 <= H <= 1024
    KL: int  # kernel 长度，1 <= KH <= 32
    F: int   # filter channel 数，32 <= F <= 128

输出:
    O: Tensor([N, L, F], float16)  # 输出 tensor

中间量:
    ACC: float32  # 累加器，accumulator

定义:
    for i in range(N):
        for j in range(L):
            for f in range(F):
                ACC = 0
                for k in range(KL):
                    if j + k < L:  # 边界检查，boundary check
                        ACC += X[i, j + k] * K[k, f]
                O[i, j, f] = ACC
"""


def ref_conv1d_multi_outchannel(X: torch.Tensor, K: torch.Tensor):
    assert len(X.shape) == 2
    assert len(K.shape) == 2
    assert X.dtype == K.dtype == torch.float16

    # for i in range(N):
    #     for j in range(L):
    #         for f in range(F):
    #             O[i, j, f] = 0
    #             for k in range(KL):
    #                 if j + k < L:  # 边界检查，boundary check
    #                     O[i, j, f] += X[i, j + k] * K[k, f]

    N, L = X.shape
    KL, F = K.shape

    padding_size = KL - 1
    X_padded = torch.nn.functional.pad(X.view(N, 1, L), (0, padding_size))

    return (
        torch.conv1d(
            input=X_padded,
            weight=K.permute(1, 0).view(F, 1, KL),
        )
        .permute(0, 2, 1)
        .contiguous()
    )


"""
先从最直接的版本开始，把上面 `F=1` 的 conv1d 扩展成多输出 channel 版本。
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_conv1d_multi_outchannel(X, K, BLOCK_N: int, BLOCK_L: int):
    N, L, KL, F = T.const("N, L, KL, F")
    dtype = T.float16
    accum_dtype = T.float32
    X: T.Tensor((N, L), dtype)
    K: T.Tensor((KL, F), dtype)
    O = T.empty((N, L, F), dtype)

    # TODO: Implement this function

    return O


"""
接着再试试 `im2col`，并用 `T.gemm` 来加速这个计算。
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_conv1d_im2col(X, K, BLOCK_N: int, BLOCK_L: int):
    N, L, KL, F = T.const("N, L, KL, F")
    dtype = T.float16
    accum_dtype = T.float32
    X: T.Tensor((N, L), dtype)
    K: T.Tensor((KL, F), dtype)
    O = T.empty((N, L, F), dtype)

    # TODO: Implement this function

    return O


def run_conv1d_im2col():
    print("\n=== Convolution 1D im2col ===\n")
    N = 128
    L = 128
    BLOCK_N = 16
    BLOCK_L = 32
    KL = 32
    F = 32
    args_dict = {
        "N": N,
        "L": L,
        "KL": KL,
        "F": F,
        "BLOCK_N": BLOCK_N,
        "BLOCK_L": BLOCK_L,
    }
    test_puzzle(tl_conv1d_multi_outchannel, ref_conv1d_multi_outchannel, args_dict)
    test_puzzle(tl_conv1d_im2col, ref_conv1d_multi_outchannel, args_dict)
    bench_puzzle(
        tl_conv1d_multi_outchannel,
        ref_conv1d_multi_outchannel,
        args_dict,
        bench_torch=True,
        bench_name="Conv1D Multi OutChannel Naive",
    )
    bench_puzzle(
        tl_conv1d_im2col,
        ref_conv1d_multi_outchannel,
        args_dict,
        bench_torch=False,
        bench_name="Conv1D im2col",
    )


if __name__ == "__main__":
    run_conv1d_naive()
    run_conv1d_im2col()
