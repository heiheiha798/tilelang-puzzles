"""
Puzzle 08: Matrix Computation
==============
现在我们开始处理 deep learning 里最基础的一类 workload：matrix computation。

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
这一章包含两个 puzzle：(1) matrix-vector multiplication（GEMV），以及
(2) matrix-matrix multiplication（GEMM）。其中 GEMV 可以看作是前面
"reduce sum" 例子的自然延伸。

注意：现代 AI workload 通常会把 `float16` 作为默认 data type。
因此在这一题里，我们会使用 `float16` 作为输入/输出 dtype，同时配合一个更高精度的
accumulator dtype，比如 `float32`。

08-1: Matrix-Vector Multiplication.

输入:
    A: Tensor([M, K], float16)  # 输入 matrix
    B: Tensor([K,], float16)  # 输入 vector
    N: int   # tensor 的大小，1 <= N <= 8192
    K: int   # tensor 的大小，1 <= K <= 8192

输出:
    C: Tensor([M,], float16)  # 输出 tensor

中间量:
    ACC: float32  # 累加器，accumulator

定义:
    for i in range(M):
        ACC = 0
        for k in range(K):
            ACC += A[i, k] * B[k]
        C[i] = ACC
"""


def ref_gemv(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 2
    assert len(B.shape) == 1
    assert A.shape[1] == B.shape[0]  # K
    assert A.dtype == B.dtype == torch.float16
    return torch.matmul(input=A, other=B)


@tilelang.jit
def tl_gemv(A, B, BLOCK_M: int, BLOCK_K: int):
    M, K = T.const("M, K")
    dtype = T.float16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K,), dtype)
    C = T.empty((M,), dtype)

    # TODO: Implement this function
    
    return C


def run_gemv():
    print("\n=== Matrix-Vector Multiplication ===\n")

    M = 4096
    K = 4096
    BLOCK_M = 128
    BLOCK_K = 32

    test_puzzle(tl_gemv, ref_gemv, {"M": M, "K": K, "BLOCK_M": BLOCK_M, "BLOCK_K": BLOCK_K})
    bench_puzzle(
        tl_gemv,
        ref_gemv,
        {"M": M, "K": K, "BLOCK_M": BLOCK_M, "BLOCK_K": BLOCK_K},
        bench_torch=True,
    )


"""
从 GEMV 走到 GEMM，问题的实际复杂度会明显上升。如果你想写出一个能接近 cuBLAS
性能的高性能 matmul kernel，需要理解很多优化手段，比如 pipelining、swizzling、
tiling 等等。但在 TileLang 里，我们可以先把注意力集中在 dataflow 和 tiling computation 上。

在现代 GPU 上，比如 NVIDIA Hopper architecture，会有专门用于 matrix multiplication
的硬件单元，叫 Tensor Cores。它们可以执行类似 `16x16x16` 的 FP16 tensor core
operation，这种底层操作通常叫 MMA instruction。前面大多数例子里的计算主要跑在
CUDA Cores 上，它们更适合 scalar/vector operation；而 Tensor Cores 则专门为 matrix
operation 做了优化，在大矩阵场景下可以提供高得多的 throughput。

TileLang 把这些复杂 instruction 以及相关的 memory loading pattern 包装成了一个简单的
`T.gemm` operator，用它就能生成高性能 matrix multiplication kernel。`T.gemm`
和前面见过的其他 TileOp 一样，接收两个 Buffer 作为输入、一个 Buffer 作为输出。
剩下的工作，核心就是把整个 matrix 做好 tiling。

08-2: Matmul (Matrix-Matrix Multiplication)

输入:
    A: Tensor([M, K], float16)  # 输入 tensor
    B: Tensor([K, N], float16)  # 输入 tensor
    N: int   # tensor 的大小，1 <= N <= 8192
    M: int   # tensor 的大小，1 <= M <= 8192
    K: int   # tensor 的大小，1 <= K <= 8192

中间量:
    ACC: float32  # 累加器，accumulator

输出:
    C: [M, N]  # 输出 tensor

定义:
    for i in range(M):
        for j in range(N):
            ACC = 0
            for k in range(K):
                ACC += A[i, k] * B[k, j]
            C[i, j] = ACC
"""


def ref_matmul(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 2
    assert len(B.shape) == 2
    assert A.shape[1] == B.shape[0]  # K
    assert A.dtype == B.dtype == torch.float16
    return torch.matmul(input=A, other=B)


@tilelang.jit
def tl_matmul_naive(A, B, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    # TODO: Implement this function

    return C


def run_matmul_naive():
    print("\n=== Matrix Multiplication Naive ===\n")

    M = 4096
    N = 4096
    K = 4096
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    test_puzzle(
        tl_matmul_naive,
        ref_matmul,
        {
            "M": M,
            "N": N,
            "K": K,
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
        },
    )
    bench_puzzle(
        tl_matmul_naive,
        ref_matmul,
        {
            "M": M,
            "N": N,
            "K": K,
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
        },
        bench_torch=True,
    )


"""
前一个实现可以工作，但性能还不够好。这里我们只通过少量代码改动，引入两个优化点。

1. Shared Memory Optimization。前面的 puzzle 里，我们一直在使用 fragment 作为
intermediate buffer，但没有展开讲太多，是为了让教程更简单。现在要回忆一下：
fragment 本质上是一个 block 内所有 thread 的 register 统一抽象。如果把 `A`、`B`、`C`
的 tile 全部塞进 registers，register 很快就会不够用，进而发生 register spilling。
所以这里需要把 `A` 和 `B` 的 tile 放进 shared memory。`T.gemm` 会高效地从 shared memory
读取数据，因此我们可以直接用 `T.alloc_shared` 给这些 tile 分配 shared memory。

2. Software Pipeline。从 NVIDIA Ampere architecture 开始，software pipeline 就成了一个
非常重要的优化技术，用来重叠 computation 和 memory access。在这里，我们可以用 software
pipeline 来把 `A` 和 `B` tile 的加载，与 GEMM operation 的计算重叠起来。实现方式是用
`T.Pipeline` 替换 `T.Serial`，并设置一个合适的 stage 数，比如 `num_stage=3`。

改完之后，你可以直接看生成出来的 CUDA code，并比较性能提升。
"""


@tilelang.jit
def tl_matmul_opt(A, B, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    # TODO: Implement this function

    return C


def run_matmul_opt():
    print("\n=== Matrix Multiplication ===\n")

    M = 4096
    N = 4096
    K = 4096
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    args_dict = {
        "M": M,
        "N": N,
        "K": K,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
    }

    print("Naive Matmul Implementation: ")
    naive_matmul_kernel = tl_matmul_naive.compile(**args_dict)
    naive_matmul_kernel.print_source_code()

    print("OPT Matmul Implementation: ")
    opt_matmul_kernel = tl_matmul_opt.compile(**args_dict)
    opt_matmul_kernel.print_source_code()

    bench_puzzle(tl_matmul_naive, ref_matmul, args_dict, bench_torch=True)
    bench_puzzle(tl_matmul_opt, ref_matmul, args_dict, bench_torch=True)


if __name__ == "__main__":
    run_gemv()
    run_matmul_naive()
    run_matmul_opt()
