"""
Puzzle 07: Scalar FlashAttention
==============
从 softmax 走到 FlashAttention，本质上只差一些额外计算。

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
既然你已经掌握了 softmax / online softmax，现在就可以开始实现 LLM 里最重要的
operator 之一：FlashAttention。

为了让学习曲线更平滑，这里我们先实现一个 scalar 版本的 FlashAttention。
同时也去掉 multi-head attention 部分。这样整个问题只剩两个维度：batch size `B`
和 sequence length `S`，它们和上一题里的 `N`、`M` 基本是对应的。
做完这些简化之后，你会发现自己离真正的 FlashAttention algorithm 其实已经不远了。
而在 TileLang 里，从这个版本扩展到完整 FlashAttention 也会更自然。

07-1: Simplified Scalar Flash Attention.

输入:
    Q: Tensor([B, S], float32)  # 输入 tensor
    K: Tensor([B, S], float32)  # 输入 tensor
    V: Tensor([B, S], float32)  # 输入 tensor
    B: int   # batch size 维度，1 <= B <= 256
    S: int   # sequence length 维度，1 <= S <= 16384

输出:
    O: Tensor([B, S], float32)  # 输出 tensor

中间量:
    MAX: float32  # 每一行的最大值
    SUM: float32  # 每一行的求和值
    QK: Tensor([B, S], float32)  # `q * k` 的结果
    P:  Tensor([B, S], float32)  # `softmax(q * k)` 的中间结果（尚未除以 summation）

定义:
    for i in range(B):
        SUM = 0
        MAX = -inf
        for j in range(S):
            QK[i, j] = Q[i, j] * K[i, j]
            MAX = max(QK[i, j], MAX)
        for j in range(S):
            P[i, j] = exp(QK[i, j] - MAX)
            SUM += P[i, j]
        for j in range(M):
            O[i, j] = P[i, j] / SUM * V[i, j]
"""


def ref_scalar_flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    assert len(Q.shape) == 2
    assert len(K.shape) == 2
    assert len(V.shape) == 2
    assert Q.shape[0] == K.shape[0] == V.shape[0]  # B
    assert Q.shape[1] == K.shape[1] == V.shape[1]  # S
    assert Q.dtype == K.dtype == V.dtype == torch.float32
    return torch.softmax(Q * K, dim=1).mul_(V)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_scalar_flash_attn(Q, K, V, BLOCK_B: int, BLOCK_S: int):
    log2_e = 1.44269504
    B, S = T.const("B, S")
    dtype = T.float32
    Q: T.Tensor((B, S), dtype)
    K: T.Tensor((B, S), dtype)
    V: T.Tensor((B, S), dtype)
    O = T.empty((B, S), dtype)

    with T.Kernel(B // BLOCK_B, threads=256) as pid_b:
        Q_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        K_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        V_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        O_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)

        cur_QK = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        cur_exp_QK = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        cur_max_QK = T.alloc_fragment((BLOCK_B,), dtype)
        cur_sum_exp_QK = T.alloc_fragment((BLOCK_B,), dtype)
        lse = T.alloc_fragment((BLOCK_B,), dtype)

        T.fill(lse, -T.infinity(dtype))

        for s_blk_id in T.Serial(S // BLOCK_S):
            T.copy(Q[pid_b * BLOCK_B, s_blk_id * BLOCK_S], Q_local)
            T.copy(K[pid_b * BLOCK_B, s_blk_id * BLOCK_S], K_local)

            for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                cur_QK[i, j] = Q_local[i, j] * K_local[i, j]

            T.reduce_max(cur_QK, cur_max_QK, dim=1, clear=True)

            for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                cur_exp_QK[i, j] = T.exp2(cur_QK[i, j] * log2_e - cur_max_QK[i] * log2_e)

            T.reduce_sum(cur_exp_QK, cur_sum_exp_QK, dim=1, clear=True)

            for i in T.Parallel(BLOCK_B):
                lse[i] = cur_max_QK[i] * log2_e + T.log2(
                    T.exp2(lse[i] - cur_max_QK[i] * log2_e) + cur_sum_exp_QK[i]
                )

        for s_blk_id in T.Serial(S // BLOCK_S):
            T.copy(Q[pid_b * BLOCK_B, s_blk_id * BLOCK_S], Q_local)
            T.copy(K[pid_b * BLOCK_B, s_blk_id * BLOCK_S], K_local)
            T.copy(V[pid_b * BLOCK_B, s_blk_id * BLOCK_S], V_local)

            for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                O_local[i, j] = (
                    T.exp2(Q_local[i, j] * K_local[i, j] * log2_e - lse[i]) * V_local[i, j]
                )

            T.copy(O_local, O[pid_b * BLOCK_B, s_blk_id * BLOCK_S])

    return O


# 上面同一个 scalar FlashAttention 算法的优化版本。
#
# 官方示例把 BLOCK_B 行合并到一个 CTA 里。默认参数 B=256、BLOCK_B=16
# 时，总共只会启动 16 个 CTA；虽然每行还要串行扫很长的 S loop，但 GPU
# 的并行度不够。这个版本改成一行一个 CTA，把 launch grid 提高到 B 个 CTA。
# 算法本身仍然只使用 scalar multiply/exp/reduce，没有走 GEMM path，也没有用
# tensor core。
#
# 第二个优化是减少每个 S tile 内的中间 fragment 和 math 指令。这个版本不再
# 先把 Q/K/V/O 用 T.copy 搬到 fragment，而是在 T.Parallel 里直接 global
# load/store；因为每个元素在当前 tile 里只使用一次，额外的 Q_local/K_local/
# V_local/O_local fragment 不划算。O 在第一遍会临时保存 Q*K*log2_e，第二遍
# 复用这个 scratch，避免重新 global load Q/K 和重算一次 Q*K。
#
# 第三个优化是把官方示例里的 LSE update 换成 running row_max/row_sum。
# 官方写法每个 S tile 都会做一次 scalar log2；这里维护 log2 scale 下的
# row_max，以及 softmax denominator row_sum。合并 tile 时只需要 rescale
# row_sum，避免了循环里的 log2。
#
# 第四个优化是把 Q*K 提前转换到 log2 scale。cur_QK、row_max 和 O scratch
# 都保存 Q*K*log2_e，后续 T.exp2 里就不需要对每个元素重复乘 log2_e。
#
# 第五个优化是 benchmark 里给 opt kernel 使用 BLOCK_S_OPT=256。相比官方示例的
# BLOCK_S=128，serial S tile 次数从 128 次降到 64 次；实测 BLOCK_S=512 会因为
# reduction tile 过大而变慢。
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_scalar_flash_attn_opt(Q, K, V, BLOCK_S: int):
    log2_e = 1.44269504
    B, S = T.const("B, S")
    dtype = T.float32
    Q: T.Tensor((B, S), dtype)
    K: T.Tensor((B, S), dtype)
    V: T.Tensor((B, S), dtype)
    O = T.empty((B, S), dtype)

    # 这里和官方示例的 B // BLOCK_B 个 CTA 不同，改成每行启动一个 CTA。
    # threads=128 在 BLOCK_S_OPT=256 下通常比 256 threads 更稳，寄存器/调度压力更低。
    with T.Kernel(B, threads=128) as pid_b:
        # 这里只保留 reduction 必须的 cur_QK fragment。Q/K/V 直接 global load，
        # O 直接 global store，并在第一遍临时作为 Q*K*log2_e scratch。
        # 这里复用 O 是安全的，因为 O 是 kernel 内新分配的 output，第一遍结束前
        # 外部还看不到它的值，第二遍会把 scratch 全部覆盖成最终输出。
        cur_QK = T.alloc_fragment((1, BLOCK_S), dtype)
        cur_max_QK = T.alloc_fragment((1,), dtype)
        cur_sum_exp_QK = T.alloc_fragment((1,), dtype)
        row_max = T.alloc_fragment((1,), dtype)
        row_sum = T.alloc_fragment((1,), dtype)
        new_row_max = T.alloc_fragment((1,), dtype)

        T.fill(row_max, -T.infinity(dtype))
        T.fill(row_sum, 0)

        for s_blk_id in T.Serial(S // BLOCK_S):
            for i, j in T.Parallel(1, BLOCK_S):
                # cur_QK 和 O scratch 保存 log2 scale 下的 Q*K，后续 exp2
                # 直接用差值即可，不再在每个 exp2 前重复乘 log2_e。
                cur_QK[i, j] = (
                    Q[pid_b, s_blk_id * BLOCK_S + j] * K[pid_b, s_blk_id * BLOCK_S + j]
                    * log2_e
                )
                O[pid_b, s_blk_id * BLOCK_S + j] = cur_QK[i, j]

            T.reduce_max(cur_QK, cur_max_QK, dim=1, clear=True)

            for i in T.Parallel(1):
                # 用 running row_max/row_sum 合并当前 tile，避免官方 LSE update
                # 里每个 tile 一次的 log2。
                new_row_max[i] = T.max(row_max[i], cur_max_QK[i])

            for i, j in T.Parallel(1, BLOCK_S):
                # cur_QK 在 reduce_max 后覆写成 exp(QK - new_row_max)，
                # 这样 reduce_sum 的结果可以直接并入 row_sum。
                cur_QK[i, j] = T.exp2(cur_QK[i, j] - new_row_max[i])

            T.reduce_sum(cur_QK, cur_sum_exp_QK, dim=1, clear=True)

            for i in T.Parallel(1):
                row_sum[i] = (
                    row_sum[i] * T.exp2(row_max[i] - new_row_max[i])
                    + cur_sum_exp_QK[i]
                )
                row_max[i] = new_row_max[i]

        for s_blk_id in T.Serial(S // BLOCK_S):
            for i, j in T.Parallel(1, BLOCK_S):
                # 第二遍也直接 global load/store。这里用 row_max/row_sum
                # 还原最终 softmax，而不是依赖官方示例里的 lse。
                O[pid_b, s_blk_id * BLOCK_S + j] = (
                    T.exp2(O[pid_b, s_blk_id * BLOCK_S + j] - row_max[i])
                    * V[pid_b, s_blk_id * BLOCK_S + j]
                    / row_sum[i]
                )

    return O


def run_scalar_flash_attn():
    print("\n=== Scalar Flash Attention ===\n")
    B = 256
    S = 16384
    BLOCK_B = 16
    BLOCK_S = 128
    # 原 kernel 保持官方示例的 BLOCK_S；opt kernel 单独调大 tile，
    # 减少 S 方向 serial loop 次数，避免改变 original baseline。
    BLOCK_S_OPT = 256

    print("Original scalar kernel:")
    test_puzzle(
        tl_scalar_flash_attn,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
    )

    print("Optimized scalar kernel:")
    test_puzzle(
        tl_scalar_flash_attn_opt,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_S": BLOCK_S_OPT},
    )

    bench_puzzle(
        tl_scalar_flash_attn,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
        bench_name="Tilelang original",
        bench_torch=True,
    )
    bench_puzzle(
        tl_scalar_flash_attn_opt,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_S": BLOCK_S_OPT},
        bench_name="Tilelang scalar opt",
        bench_torch=False,
    )


if __name__ == "__main__":
    run_scalar_flash_attn()
