"""
Puzzle 02: Vector Add
==============
这个 puzzle 要你实现一个 vector addition operation。

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
vector addition 是我们正式进入计算类 kernel 的第一步。TileLang 提供了基础 arithmetic
operation，比如 add、sub、mul、div 等。但这些 operation 都是 element-wise 的
（它们不像 `T.copy` 那样属于 TileOp）。所以我们需要借助 loop abstraction 来遍历
tensor 中的元素，并在 loop body 里写下自己需要的计算逻辑。

02-1: 1-D vector addition.

输入:
    A: Tensor([N,], float16)  # 输入 tensor
    B: Tensor([N,], float16)  # 输入 tensor
    N: int   # tensor 的大小，1 <= N <= 1024*1024

输出:
    C: Tensor([N,], T.float16)  # 输出 tensor

定义:
    for i in range(N):
        C[i] = A[i] + B[i]
"""


def ref_add_1d(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 1
    assert len(B.shape) == 1
    assert A.shape[0] == B.shape[0]
    assert A.dtype == B.dtype == torch.float16
    return A + B


@tilelang.jit
def tl_add_1d(A, B, BLOCK_N: int):
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B: T.Tensor((N,), T.float16)
    C = T.empty((N,), T.float16)

    with T.Kernel(N // BLOCK_N, threads=256) as bx:
        base_idx = bx * BLOCK_N
        for i in T.Parallel(BLOCK_N):
            C[base_idx + i] = A[base_idx + i] + B[base_idx + i]

    return C


def run_add_1d():
    print("\n=== Vector Add 1D ===\n")
    N = 1024 * 256
    BLOCK_N = 1024
    test_puzzle(tl_add_1d, ref_add_1d, {"N": N, "BLOCK_N": BLOCK_N})


"""
我们还可以把更多 element-wise operation 融合进同一个 kernel。
现在试着做一个 element-wise multiplication，并接一个 ReLU activation。

提示：可以用 `T.if_then_else(cond, true_value, false_value)` 来实现 conditional logic。

02-2: 1-D vector multiplication with ReLU activation

输入:
    A: Tensor([N,], float16)  # 输入 tensor
    B: Tensor([N,], float16)  # 输入 tensor
    N: int   # tensor 的大小，1 <= N <= 1024*1024

输出:
    C: Tensor([N,], T.float16)  # 输出 tensor

输出:
    C: [N,]  # 输出 tensor

定义:
    for i in range(N):
        C[i] = max(0, A[i] * B[i])
"""


def ref_mul_relu_1d(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 1
    assert len(B.shape) == 1
    assert A.shape[0] == B.shape[0]
    assert A.dtype == B.dtype == torch.float16
    return (A * B).relu_()


@tilelang.jit
def tl_mul_relu_1d(A, B, BLOCK_N: int):
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B: T.Tensor((N,), T.float16)
    C = T.empty((N,), T.float16)

    # TODO: Implement this function

    return C


def run_mul_relu_1d():
    print("\n=== Vector Multiplication with ReLU 1D ===\n")
    N = 1024 * 256
    BLOCK_N = 1024
    test_puzzle(tl_mul_relu_1d, ref_mul_relu_1d, {"N": N, "BLOCK_N": BLOCK_N})


"""
注意：这一节需要你对 GPU memory hierarchy 和基础 CUDA 编程有一点了解。

我们可以继续优化上一个例子。这里会引入一种 kernel programming 中非常常见的优化思路。
如果你写过 CUDA 或其他 GPU 编程框架，应该已经知道 GPU 上存在分层的 memory hierarchy。

通常主要有三层 memory：global memory (DRAM)、shared memory 和 registers。
其中 registers 最快，但容量也最小。在 CUDA 中，你在 kernel 内声明 local variable 时，
往往就会对应到 registers 的使用。

前面的实现是直接从 `A`、`B` 读取数据，再把结果写回 `C`，而 `A`、`B`、`C`
本质上都是 global memory pointer。这样做不够高效，因为每个元素都要单独访问
global memory。你可以用 `print_source_code()` 来查看生成出来的 CUDA code。

这里我们考虑用 registers 来优化 kernel。核心想法是：一次性在 registers 和
global memory 之间搬运多个数据元素。比如 CUDA 里常见的 `ldg128`，会一次从
global memory 读取 128 bits 到 registers，从理论上说可以把 memory access 次数
减少到原来的四分之一。

在这个 fused kernel 例子里，`A * B` 的 intermediate result 也可以先放在 registers 里。
这样在应用 ReLU 时，就可以直接从 registers 读取，而不必再次访问 global memory。
当然在真实编译中，这种优化有时不需要你手动写，NVCC 也可能通过 common subexpression
elimination（CSE）自动做掉。
"""

"""
TileLang 会把这些 memory level 显式暴露给用户。你可以使用 `T.alloc_fragment`
来分配一块 fragment register 空间。要注意，在原生 CUDA 里，register 是 thread-local 的，
所以你通常需要自己处理 mapping，确保每个 thread 只加载自己负责的那一部分数据。
但在 TileLang 里，不需要你手动做这些映射。
这里的 fragment 可以理解成一个 block 内所有 thread 的 register 抽象，我们可以像操作
`T.Buffer` 一样统一地操作这个 fragment。
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_mul_relu_1d_mem(A, B, BLOCK_N: int):
    N = T.const("N")
    dtype = T.float16
    A: T.Tensor((N,), dtype)
    B: T.Tensor((N,), dtype)
    C = T.empty((N,), dtype)

    # TODO: Implement this function

    return C


def run_mul_relu_1d_mem():
    print("\n=== Vector Multiplication with ReLU 1D (Memory Optimized) ===\n")
    N = 1024 * 4096
    BLOCK_N = 1024

    print("Naive TL Implementation: ")
    tl_mul_relu_kernel = tl_mul_relu_1d.compile(N=N, BLOCK_N=BLOCK_N)
    tl_mul_relu_kernel.print_source_code()

    print("Optimized Version")
    tl_mul_relu_kernel_opt = tl_mul_relu_1d_mem.compile(N=N, BLOCK_N=BLOCK_N)
    tl_mul_relu_kernel_opt.print_source_code()

    test_puzzle(tl_mul_relu_1d_mem, ref_mul_relu_1d, {"N": N, "BLOCK_N": BLOCK_N})
    bench_puzzle(
        tl_mul_relu_1d,
        ref_mul_relu_1d,
        {"N": N, "BLOCK_N": BLOCK_N},
        bench_name="TL Naive",
        bench_torch=True,
    )
    bench_puzzle(
        tl_mul_relu_1d_mem,
        ref_mul_relu_1d,
        {"N": N, "BLOCK_N": BLOCK_N},
        bench_name="TL OPT",
        bench_torch=False,
    )


if __name__ == "__main__":
    run_add_1d()
    run_mul_relu_1d()
    run_mul_relu_1d_mem()
