"""
Puzzle 01: Copy
==============
这个 puzzle 要你实现一个 copy operation，把数据从一个 tensor
复制到另一个 tensor。

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
开始之前，我们先给一个可以直接运行的 TileLang copy 示例。
下面这段代码展示了如何用 TileLang 定义一个 1-D copy kernel。这里假设
所有 tensor 一开始都存放在 GPU 的 global memory (DRAM) 中。

01-1: 1-D copy kernel.

输入:
    A: Tensor([N,], float16)  # 输入 tensor
    N: int   # tensor 的大小，1 <= N <= 1024*1024

输出:
    B: Tensor([N,], float16)  # copy 之后得到的 tensor

定义:
    for i in range(N):
        B[i] = A[i]
"""


def ref_copy_1d(A: torch.Tensor):
    assert len(A.shape) == 1
    assert A.dtype == torch.float16
    return A.clone()


"""
这里我们使用 TileLang 的 EagerJIT kernel 编程风格，这样写起来更直观。

在 TileLang 里，一个 kernel 通常定义成带有 `@tilelang.jit` 装饰器的 Python 函数。
这个 decorator 会启用 kernel 的 JIT compilation。函数参数表示 kernel 的输入/输出 tensor
（这里默认是 fully compacted 的 torch Tensor）以及其他 hyperparameter。
在这个例子里，输入 tensor `A` 作为参数传入，输出 tensor `B` 作为返回值给出。

函数声明之后，host code 部分会定义常量、tensor 的 shape 和 dtype。

接下来我们需要指定 kernel launch configuration。在 TileLang 中，我们用 `T.Kernel`
来启动一个 kernel。它接收 block 数量，以及一个 `threads` 整数来表示每个 block
里有多少个 thread。最终启动的总 thread 数可以理解为 `blocks * threads`。

第一步里，我们先写一个最简单的 serial copy kernel，只启动一个 thread。
"""


@tilelang.jit
def tl_copy_1d_serial(A):
    # 这是 TileLang script 的 host/declaration 部分。
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B = T.empty((N,), T.float16)

    # 下面是 kernel function 的主体，用 TileLang DSL 来写。
    # 这里通过 T.Kernel 启动一个 kernel。
    with T.Kernel(1, threads=1) as _:
        # 这里的 T.copy 是 TileLang 内建的 TileOp。
        # 它会自动利用当前 block 中可用的 thread 做高效 memory copy，
        # 其中包括自动 parallelism 和 vectorization。
        # 因为这里我们只启动了一个 thread，所以它最终会被 lower 成 serial loop copy，
        # 同时仍可能带有一定的位宽 vectorization，比如一次 copy 128 bits。
        T.copy(A, B)

    return B


def run_copy_1d_serial():
    print("\n=== Copy 1D Serial ===\n")
    N = 1024
    test_puzzle(tl_copy_1d_serial, ref_copy_1d, {"N": N})


"""
上面的实现只启动了一个 thread，所以效率不会高。
现在我们希望在一个 kernel 里启动多个 thread，并行完成数据 copy。

因为 `T.copy` 本身就会在一个 block 内自动做并行 copy，所以这里其实不需要改很多地方。

你可以尝试把每个 block 的 thread 数改成 128 或 256，然后比较一下 speedup。
"""


@tilelang.jit
def tl_copy_1d_multi_threads(A):
    # 这是 TileLang script 的 host/declaration 部分。
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B = T.empty((N,), T.float16)

    # TODO: 实现这个函数

    return B


def run_copy_1d_multi_threads():
    print("\n=== Copy 1D Multi-threads ===\n")
    N = 1024 * 256

    test_puzzle(tl_copy_1d_multi_threads, ref_copy_1d, {"N": N})

    # 因为 N 比较大，这里的 benchmark 可能会花一点时间
    bench_puzzle(
        tl_copy_1d_serial,
        ref_copy_1d,
        {"N": N},
        bench_name="TL Serial",
        bench_torch=True,
    )
    bench_puzzle(
        tl_copy_1d_multi_threads,
        ref_copy_1d,
        {"N": N},
        bench_name="TL Multi-threads",
        bench_torch=False,
    )


"""
最后，我们希望把 copy operation 进一步扩展到多个 block 上并行执行。
这里用 `BLOCK_N` 表示每个 block 负责 copy 的元素数量。
剩下的实现思路和前一个版本类似。这里假设 `N` 可以被 `BLOCK_N` 整除。

注意：你需要自己处理不同 block 对应的 memory access 区间。好在这里我们可以拿到
`bx`（也就是 block index），所以可以据此计算每个 block 的起始和结束位置。
"""


@tilelang.jit
def tl_copy_1d_parallel(A, BLOCK_N: int):
    # 这是 TileLang script 的 host/declaration 部分。
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B = T.empty((N,), T.float16)

    # TODO: 实现这个函数

    return B


def run_copy_1d_parallel():
    print("\n=== Copy 1D Parallel ===\n")
    N = 1024 * 256
    BLOCK_N = 1024
    test_puzzle(tl_copy_1d_parallel, ref_copy_1d, {"N": N, "BLOCK_N": BLOCK_N})
    bench_puzzle(
        tl_copy_1d_parallel,
        ref_copy_1d,
        {"N": N, "BLOCK_N": BLOCK_N},
        bench_name="TL Parallel",
        bench_torch=True,
    )


if __name__ == "__main__":
    run_copy_1d_serial()
    run_copy_1d_multi_threads()
    run_copy_1d_parallel()
