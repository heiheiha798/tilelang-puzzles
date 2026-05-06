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


def run_scalar_flash_attn():
    print("\n=== Scalar Flash Attention ===\n")
    B = 256
    S = 16384
    BLOCK_B = 16
    BLOCK_S = 128
    test_puzzle(
        tl_scalar_flash_attn,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
    )
    bench_puzzle(
        tl_scalar_flash_attn,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
        bench_torch=True,
    )


if __name__ == "__main__":
    run_scalar_flash_attn()
