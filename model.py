"""
DeepSeek-V4 MindSpore Implementation
=====================================
基于论文 "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
实现完整的模型架构，包含:
  - Manifold-Constrained Hyper-Connections (mHC)
  - Compressed Sparse Attention (CSA)
  - Heavily Compressed Attention (HCA)
  - DeepSeekMoE (含共享专家 + 路由专家)
  - Multi-Token Prediction (MTP)

支持 DeepSeek-V4-Flash (284B, 13B activated) 和
      DeepSeek-V4-Pro  (1.6T, 49B activated) 两种配置。
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor, dtype as mstype


# ============================================================================
# 1. 配置类
# ============================================================================

@dataclass
class DeepSeekV4Config:
    """DeepSeek-V4 模型全局配置，涵盖 Flash 和 Pro 两种规模。"""

    # ---- 基础结构 ----
    vocab_size: int = 128_000
    num_layers: int = 43
    hidden_size: int = 4096                # d
    max_seq_length: int = 1_048_576        # 1M tokens

    # ---- 注意力通用参数 ----
    num_query_heads: int = 64              # n_h
    head_dim: int = 512                    # c
    query_compress_dim: int = 1024         # d_c
    rope_dim: int = 64                     # RoPE 应用维度
    num_output_groups: int = 8             # g (分组输出投影)
    group_output_dim: int = 1024           # d_g

    # ---- CSA 参数 ----
    csa_compress_rate: int = 4             # m
    csa_indexer_num_heads: int = 64        # n_I^h
    csa_indexer_head_dim: int = 128        # c_I
    csa_top_k: int = 512                   # 稀疏注意力 top-k

    # ---- HCA 参数 ----
    hca_compress_rate: int = 128           # m'

    # ---- 滑动窗口 ----
    sliding_window_size: int = 128         # n_win

    # ---- 注意力 Sink ----
    use_attention_sink: bool = True

    # ---- MoE 参数 ----
    num_shared_experts: int = 1
    num_routed_experts: int = 256
    num_activated_experts: int = 6
    expert_intermediate_dim: int = 2048
    num_hash_routing_layers: int = 3       # 前几层使用 Hash 路由
    load_balance_bias_update_speed: float = 0.001
    balance_loss_weight: float = 0.0001

    # ---- mHC 参数 ----
    mhc_expansion_factor: int = 4          # n_hc
    mhc_sinkhorn_iters: int = 20           # t_max

    # ---- MTP 参数 ----
    mtp_depth: int = 1
    mtp_loss_weight: float = 0.3

    # ---- 注意力层类型安排 ----
    # "swa" = 纯滑动窗口, "csa" = CSA, "hca" = HCA
    layer_attention_types: Optional[List[str]] = None

    # ---- SwiGLU Clamping ----
    swiglu_clamp_min: float = -10.0
    swiglu_clamp_max: float = 10.0
    gate_clamp_max: float = 10.0

    def __post_init__(self):
        """根据层数和注意力类型自动生成层配置。"""
        if self.layer_attention_types is None:
            self.layer_attention_types = self._default_attention_schedule()

    def _default_attention_schedule(self) -> List[str]:
        """
        Flash: 前2层纯 SWA，后续 CSA/HCA 交替
        Pro:   前2层 HCA，后续 CSA/HCA 交替
        这里以 Flash 为默认；Pro 在 main.py 中覆盖。
        """
        schedule = []
        for i in range(self.num_layers):
            if i < 2:
                schedule.append("swa")
            else:
                # CSA 和 HCA 交替
                if (i - 2) % 2 == 0:
                    schedule.append("csa")
                else:
                    schedule.append("hca")
        return schedule


def flash_config() -> DeepSeekV4Config:
    """DeepSeek-V4-Flash: 284B 总参数, 13B 激活参数"""
    return DeepSeekV4Config(
        num_layers=43,
        hidden_size=4096,
        num_query_heads=64,
        head_dim=512,
        query_compress_dim=1024,
        num_output_groups=8,
        group_output_dim=1024,
        csa_compress_rate=4,
        csa_indexer_num_heads=64,
        csa_indexer_head_dim=128,
        csa_top_k=512,
        hca_compress_rate=128,
        sliding_window_size=128,
        num_routed_experts=256,
        num_activated_experts=6,
        expert_intermediate_dim=2048,
        mhc_expansion_factor=4,
        mhc_sinkhorn_iters=20,
        mtp_depth=1,
        layer_attention_types=None,  # Flash: 前2层 SWA, 后续 CSA/HCA 交替
    )


def pro_config() -> DeepSeekV4Config:
    """DeepSeek-V4-Pro: 1.6T 总参数, 49B 激活参数"""
    num_layers = 61
    # Pro: 前2层 HCA，后续 CSA/HCA 交替
    attn_schedule = []
    for i in range(num_layers):
        if i < 2:
            attn_schedule.append("hca")
        else:
            if (i - 2) % 2 == 0:
                attn_schedule.append("csa")
            else:
                attn_schedule.append("hca")

    return DeepSeekV4Config(
        num_layers=num_layers,
        hidden_size=7168,
        num_query_heads=128,
        head_dim=512,
        query_compress_dim=1536,
        num_output_groups=16,
        group_output_dim=1024,
        csa_compress_rate=4,
        csa_indexer_num_heads=64,
        csa_indexer_head_dim=128,
        csa_top_k=1024,
        hca_compress_rate=128,
        sliding_window_size=128,
        num_routed_experts=384,
        num_activated_experts=6,
        expert_intermediate_dim=3072,
        mhc_expansion_factor=4,
        mhc_sinkhorn_iters=20,
        mtp_depth=1,
        layer_attention_types=attn_schedule,
    )


# ============================================================================
# 2. 基础组件
# ============================================================================

class RMSNorm(nn.Cell):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(
            Tensor(np.ones(dim), mstype.float32), name="rmsnorm_weight"
        )

    def construct(self, x: Tensor) -> Tensor:
        # x: (..., d)
        rms = ops.sqrt(ops.mean(x ** 2, axis=-1, keep_dims=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Cell):
    """
    旋转位置编码 (RoPE)。
    仅对输入向量的最后 rope_dim 个维度施加旋转。
    """

    def __init__(self, rope_dim: int = 64, base: float = 10000.0):
        super().__init__()
        self.rope_dim = rope_dim
        self.base = base
        # 预计算频率（不训练）
        self._freqs_cos = None
        self._freqs_sin = None

    def _build_freqs(self, seq_len: int):
        """构建 cos/sin 频率表"""
        half_dim = self.rope_dim // 2
        freqs = 1.0 / (
            self.base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim)
        )
        positions = np.arange(seq_len, dtype=np.float32)
        angles = np.outer(positions, freqs)  # (seq_len, half_dim)
        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)
        self._freqs_cos = Tensor(cos_vals, mstype.float32)
        self._freqs_sin = Tensor(sin_vals, mstype.float32)

    def _rotate_half(self, x: Tensor) -> Tensor:
        """将 x 的后半部分取负并交换前后半部分"""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return ops.concat((-x2, x1), axis=-1)

    def construct(
        self, x: Tensor, positions: Optional[Tensor] = None
    ) -> Tensor:
        """
        x: (..., d) 其中最后 rope_dim 维应用 RoPE
        positions: (seq_len,) 位置索引；为 None 时用 0..seq_len-1
        """
        d = x.shape[-1]
        rope_start = d - self.rope_dim
        x_rope = x[..., rope_start:]
        x_pass = x[..., :rope_start]

        seq_len = x.shape[-2] if len(x.shape) >= 2 else 1
        if self._freqs_cos is None or self._freqs_cos.shape[0] < seq_len:
            self._build_freqs(max(seq_len, 8192))

        cos = self._freqs_cos[:seq_len]   # (seq_len, half_dim)
        sin = self._freqs_sin[:seq_len]

        # 广播到 x_rope 的形状
        # x_rope: (..., seq_len, rope_dim)
        # cos/sin 需 broadcast
        while cos.ndim < x_rope.ndim:
            cos = cos.expand_dims(0)
            sin = sin.expand_dims(0)

        # 将 cos 扩展到 rope_dim（重复一次）
        cos = ops.concat((cos, cos), axis=-1)
        sin = ops.concat((sin, sin), axis=-1)

        x_rotated = x_rope * cos + self._rotate_half(x_rope) * sin
        return ops.concat((x_pass, x_rotated), axis=-1)


# ============================================================================
# 3. Manifold-Constrained Hyper-Connections (mHC)
# ============================================================================

class SinkhornKnopp(nn.Cell):
    """
    Sinkhorn-Knopp 迭代算法：
    将矩阵投影到双随机矩阵流形 (Birkhoff polytope)。
    """

    def __init__(self, num_iters: int = 20):
        super().__init__()
        self.num_iters = num_iters

    def construct(self, M: Tensor) -> Tensor:
        """
        M: (..., n, n) 原始矩阵
        返回: (..., n, n) 双随机矩阵
        """
        # 先取指数确保非负
        M = ops.exp(M)
        for _ in range(self.num_iters):
            # 列归一化
            col_sum = ops.sum(M, axis=-2, keepdims=True) + 1e-8
            M = M / col_sum
            # 行归一化
            row_sum = ops.sum(M, axis=-1, keepdims=True) + 1e-8
            M = M / row_sum
        return M


class ManifoldConstrainedHyperConnection(nn.Cell):
    """
    mHC: 流形约束超连接，替代标准残差连接。

    核心思想:
      - 将残差流宽度扩展 n_hc 倍
      - 残差映射 B 约束在双随机矩阵流形上 (Sinkhorn-Knopp)
      - 输入/输出映射 A, C 通过 Sigmoid 约束为非负有界

    更新公式:
      X_{l+1} = B_l @ X_l + C_l * F_l(A_l @ X_l)

    参数动态生成:
      由输入 X_l 经过线性变换 + 静态偏置动态产生 A_tilde, B_tilde, C_tilde，
      再施加约束得到 A, B, C。
    """

    def __init__(self, hidden_size: int, n_hc: int, sinkhorn_iters: int = 20):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_hc = n_hc
        self.d = hidden_size  # 实际层输入维度

        # 用于归一化展平输入的 RMSNorm
        self.pre_norm = RMSNorm(n_hc * hidden_size)

        # --- 动态分量的可学习参数 ---
        # W_pre: (n_hc * d, n_hc)
        self.W_pre = Parameter(
            Tensor(np.random.randn(n_hc * hidden_size, n_hc).astype(np.float32) * 0.01,
                   mstype.float32),
            name="mhc_W_pre",
        )
        # W_res: (n_hc * d, n_hc^2)
        self.W_res = Parameter(
            Tensor(np.random.randn(n_hc * hidden_size, n_hc * n_hc).astype(np.float32) * 0.01,
                   mstype.float32),
            name="mhc_W_res",
        )
        # W_post: (n_hc * d, n_hc)
        self.W_post = Parameter(
            Tensor(np.random.randn(n_hc * hidden_size, n_hc).astype(np.float32) * 0.01,
                   mstype.float32),
            name="mhc_W_post",
        )

        # --- 静态偏置 ---
        self.S_pre = Parameter(
            Tensor(np.zeros((1, n_hc), dtype=np.float32), mstype.float32),
            name="mhc_S_pre",
        )
        self.S_res = Parameter(
            Tensor(np.eye(n_hc, dtype=np.float32), mstype.float32),  # 初始化为单位矩阵
            name="mhc_S_res",
        )
        self.S_post = Parameter(
            Tensor(np.zeros((n_hc, 1), dtype=np.float32), mstype.float32),
            name="mhc_S_post",
        )

        # --- 可学习门控因子（初始化为小值）---
        self.alpha_pre = Parameter(
            Tensor(np.array([0.01], dtype=np.float32), mstype.float32),
            name="mhc_alpha_pre",
        )
        self.alpha_res = Parameter(
            Tensor(np.array([0.01], dtype=np.float32), mstype.float32),
            name="mhc_alpha_res",
        )
        self.alpha_post = Parameter(
            Tensor(np.array([0.01], dtype=np.float32), mstype.float32),
            name="mhc_alpha_post",
        )

        # Sinkhorn-Knopp 投影
        self.sinkhorn = SinkhornKnopp(num_iters=sinkhorn_iters)

    def _generate_parameters(
        self, X: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        从输入 X 动态生成约束后的 A, B, C 矩阵。

        X: (batch, n_hc, d) 残差状态
        返回:
            A: (batch, 1, n_hc)    输入映射
            B: (batch, n_hc, n_hc) 残差映射（双随机矩阵）
            C: (batch, n_hc, 1)    输出映射
        """
        batch = X.shape[0]

        # 展平并归一化: (batch, n_hc * d)
        X_flat = X.reshape(batch, -1)
        X_hat = self.pre_norm(X_flat)  # (batch, n_hc * d)

        # 生成无约束原始参数
        A_tilde = self.alpha_pre * ops.matmul(X_hat, self.W_pre) + self.S_pre
        # A_tilde: (batch, 1, n_hc)

        B_flat = ops.matmul(X_hat, self.W_res)  # (batch, n_hc^2)
        B_tilde = B_flat.reshape(batch, self.n_hc, self.n_hc)
        B_tilde = self.alpha_res * B_tilde + self.S_res

        C_tilde = self.alpha_post * ops.matmul(X_hat, self.W_post).transpose(0, 2, 1)
        # C_tilde: (batch, n_hc, 1)
        C_tilde = C_tilde + self.S_post

        # 施加约束
        A = ops.sigmoid(A_tilde)                   # (batch, 1, n_hc)
        C = 2.0 * ops.sigmoid(C_tilde)             # (batch, n_hc, 1)
        B = self.sinkhorn(B_tilde)                 # (batch, n_hc, n_hc)

        return A, B, C

    def get_layer_input(self, X: Tensor) -> Tensor:
        """
        使用输入映射 A 从残差状态中提取实际层输入 (论文 Eq.1: A_l @ X_l)。

        X: (batch, n_hc, d) 残差状态
        返回: (batch, d) 层输入
        """
        A, _, _ = self._generate_parameters(X)
        # A: (batch, 1, n_hc), X: (batch, n_hc, d)
        # A @ X -> (batch, 1, d) -> squeeze -> (batch, d)
        layer_input = ops.matmul(A, X).squeeze(1)
        return layer_input

    def construct(self, X: Tensor, F_out: Tensor) -> Tensor:
        """
        mHC 更新步骤。

        X:     (batch, n_hc, d)  当前残差状态
        F_out: (batch, d)        当前 Transformer 层输出
        返回:  (batch, n_hc, d)  更新后的残差状态
        """
        _, B, C = self._generate_parameters(X)

        # 残差更新: X_{l+1} = B @ X + C * F_out
        BX = ops.matmul(B, X)  # (batch, n_hc, d)

        # C: (batch, n_hc, 1), F_out: (batch, d)
        # C * F_out -> broadcast -> (batch, n_hc, d)
        CF = C * F_out.expand_dims(1)

        return BX + CF


# ============================================================================
# 4. KV 压缩模块 (CSA / HCA 共用)
# ============================================================================

class KVCompressor(nn.Cell):
    """
    Token 级 KV 压缩器。

    将每 m 个 token 的 KV 条目压缩为 1 个条目。
    使用可学习的压缩权重 Z 和位置偏置 B，通过 Softmax 加权聚合。

    CSA 使用双分支压缩 (Ca, Cb) 且带重叠；
    HCA 使用单分支压缩 (C) 无重叠。
    """

    def __init__(self, hidden_size: int, head_dim: int, compress_rate: int,
                 dual_branch: bool = True, name_prefix: str = "kv"):
        super().__init__()
        self.head_dim = head_dim
        self.compress_rate = compress_rate
        self.dual_branch = dual_branch

        # KV 投影
        self.W_kv_a = nn.Dense(hidden_size, head_dim, has_bias=False)
        if dual_branch:
            self.W_kv_b = nn.Dense(hidden_size, head_dim, has_bias=False)

        # 压缩权重投影
        self.W_z_a = nn.Dense(hidden_size, head_dim, has_bias=False)
        if dual_branch:
            self.W_z_b = nn.Dense(hidden_size, head_dim, has_bias=False)

        # 可学习位置偏置
        m = compress_rate if not dual_branch else compress_rate
        self.bias_a = Parameter(
            Tensor(np.random.randn(m, head_dim).astype(np.float32) * 0.01,
                   mstype.float32),
            name=f"{name_prefix}_bias_a",
        )
        if dual_branch:
            self.bias_b = Parameter(
                Tensor(np.random.randn(m, head_dim).astype(np.float32) * 0.01,
                       mstype.float32),
                name=f"{name_prefix}_bias_b",
            )

    def _compress_single(
        self, H: Tensor, W_kv: nn.Dense, W_z: nn.Dense,
        bias: Tensor, m: int
    ) -> Tensor:
        """
        单分支压缩。
        H: (batch, seq_len, d)
        返回: (batch, seq_len // m, head_dim)
        """
        C = W_kv(H)   # (batch, seq_len, c)
        Z = W_z(H)     # (batch, seq_len, c)

        batch, seq_len, c = C.shape
        n_blocks = seq_len // m

        # 截断到 m 的整数倍
        C = C[:, :n_blocks * m, :]
        Z = Z[:, :n_blocks * m, :]

        # 重塑为 (batch, n_blocks, m, c)
        C = C.reshape(batch, n_blocks, m, c)
        Z = Z.reshape(batch, n_blocks, m, c)

        # 加位置偏置并 softmax
        S = ops.softmax(Z + bias, axis=2)  # 在 m 维度上 softmax

        # 加权求和
        C_comp = ops.sum(S * C, axis=2)  # (batch, n_blocks, c)
        return C_comp

    def construct(self, H: Tensor) -> Tensor:
        """
        H: (batch, seq_len, d) 输入隐状态
        返回: (batch, num_compressed, c) 压缩后的 KV 条目
        """
        m = self.compress_rate
        if not self.dual_branch:
            return self._compress_single(H, self.W_kv_a, self.W_z_a, self.bias_a, m)

        # CSA 双分支重叠压缩
        Ca = self.W_kv_a(H)   # (batch, seq_len, c)
        Cb = self.W_kv_b(H)
        Za = self.W_z_a(H)
        Zb = self.W_z_b(H)

        batch, seq_len, c = Ca.shape

        # 对于双分支重叠压缩:
        # C_comp_i 使用 Ca[m*i : m*(i+1)] 和 Cb[m*(i-1) : m*i]
        # 即每个压缩条目覆盖 2m 个 token，但有 m 的重叠
        # 最终压缩比为 1/m

        n_blocks = seq_len // m

        # 截断
        Ca = Ca[:, :n_blocks * m, :].reshape(batch, n_blocks, m, c)
        Za = Za[:, :n_blocks * m, :].reshape(batch, n_blocks, m, c)
        Cb_padded = ops.pad(Cb[:, :n_blocks * m, :],
                            ((0, 0), (m, 0), (0, 0)),
                            mode='constant', value=0.0)
        Zb_padded = ops.pad(Zb[:, :n_blocks * m, :],
                            ((0, 0), (m, 0), (0, 0)),
                            mode='constant', value=float('-inf'))

        # 拼接 a 和 b 分支后做 softmax (论文 Eq.11)
        Z_cat = ops.concat(
            (Za + self.bias_a, Zb_padded[:, :n_blocks * m, :].reshape(batch, n_blocks, m, c) + self.bias_b),
            axis=2
        )  # (batch, n_blocks, 2m, c)
        S = ops.softmax(Z_cat, axis=2)

        # 拆分权重
        Sa = S[:, :, :m, :]
        Sb = S[:, :, m:, :]

        # Cb 块偏移: 第 i 块使用 Cb[m*(i-1) : m*i]
        Cb_shifted = Cb[:, :n_blocks * m, :]
        Cb_shifted = ops.pad(Cb_shifted, ((0, 0), (m, 0), (0, 0)),
                             mode='constant', value=0.0)
        Cb_shifted = Cb_shifted[:, :n_blocks * m, :].reshape(batch, n_blocks, m, c)

        C_comp = ops.sum(Sa * Ca, axis=2) + ops.sum(Sb * Cb_shifted, axis=2)
        return C_comp


# ============================================================================
# 5. Lightning Indexer (CSA 专用稀疏选择)
# ============================================================================

class LightningIndexer(nn.Cell):
    """
    Lightning Indexer: 为 CSA 的稀疏注意力提供 top-k 索引选择。

    流程:
      1. 对压缩 KV 条目再做一次压缩得到 indexer keys
      2. 用低秩方式生成 indexer queries
      3. 计算 index score，选取 top-k
    """

    def __init__(self, hidden_size: int, query_compress_dim: int,
                 indexer_head_dim: int, indexer_num_heads: int,
                 compress_rate: int, top_k: int):
        super().__init__()
        self.indexer_head_dim = indexer_head_dim
        self.indexer_num_heads = indexer_num_heads
        self.compress_rate = compress_rate
        self.top_k = top_k
        self.c_I = indexer_head_dim

        # 压缩 indexer keys 的投影
        self.W_indexer_kv = nn.Dense(hidden_size, indexer_head_dim, has_bias=False)
        self.W_indexer_z = nn.Dense(hidden_size, indexer_head_dim, has_bias=False)
        self.indexer_bias = Parameter(
            Tensor(np.random.randn(compress_rate, indexer_head_dim).astype(np.float32) * 0.01,
                   mstype.float32),
            name="indexer_bias",
        )

        # 低秩 indexer query 生成 (与主 query 共享压缩向量 c_Q)
        self.W_IUQ = nn.Dense(query_compress_dim, indexer_head_dim * indexer_num_heads,
                              has_bias=False)

        # Index score 权重
        self.W_w = nn.Dense(hidden_size, indexer_num_heads, has_bias=False)

    def compute_indexer_keys(self, H: Tensor) -> Tensor:
        """
        计算压缩 indexer keys。
        H: (batch, seq_len, d)
        返回: (batch, n_blocks, c_I)
        """
        C = self.W_indexer_kv(H)
        Z = self.W_indexer_z(H)

        batch, seq_len, c_I = C.shape
        m = self.compress_rate
        n_blocks = seq_len // m

        C = C[:, :n_blocks * m, :].reshape(batch, n_blocks, m, c_I)
        Z = Z[:, :n_blocks * m, :].reshape(batch, n_blocks, m, c_I)

        S = ops.softmax(Z + self.indexer_bias, axis=2)
        K_comp = ops.sum(S * C, axis=2)  # (batch, n_blocks, c_I)
        return K_comp

    def compute_index_scores(
        self, h_query: Tensor, c_Q: Tensor, K_indexer: Tensor
    ) -> Tensor:
        """
        计算 index scores 并返回 top-k 索引。

        h_query:  (batch, num_queries, d)  查询 token 的隐状态
        c_Q:      (batch, num_queries, d_c) 压缩查询向量
        K_indexer: (batch, n_blocks, c_I)  压缩 indexer keys

        返回: (batch, num_queries, top_k) top-k 索引
        """
        # 生成 indexer queries
        q_I = self.W_IUQ(c_Q)  # (batch, num_queries, c_I * n_I^h)
        batch, num_q, _ = q_I.shape
        q_I = q_I.reshape(batch, num_q, self.indexer_num_heads, self.c_I)

        # Index score 权重
        w_I = self.W_w(h_query)  # (batch, num_queries, n_I^h)

        # 计算 score: sum_h w_h * ReLU(q_h . K_s)
        # q_I: (batch, num_q, n_I^h, c_I)
        # K_indexer: (batch, n_blocks, c_I)
        # scores: (batch, num_q, n_blocks)

        # 先计算 q_I 和 K_indexer 的点积
        # (batch, num_q, n_I^h, c_I) x (batch, 1, n_blocks, c_I)^T
        dots = ops.matmul(
            q_I, K_indexer.expand_dims(1).transpose(0, 1, 3, 2)
        )  # (batch, num_q, n_I^h, n_blocks)

        dots = ops.relu(dots)

        # 加权求和 over heads
        w_I = w_I.expand_dims(-1)  # (batch, num_q, n_I^h, 1)
        scores = ops.sum(w_I * dots, axis=2)  # (batch, num_q, n_blocks)

        # Top-k 选择
        top_k_vals, top_k_idx = ops.topk(scores, self.top_k)
        return top_k_idx


# ============================================================================
# 6. Compressed Sparse Attention (CSA)
# ============================================================================

class CompressedSparseAttention(nn.Cell):
    """
    CSA: 压缩稀疏注意力

    核心流程:
      1. 将 KV 缓存每 m 个 token 压缩为 1 个条目 (双分支重叠压缩)
      2. 用 Lightning Indexer 选取 top-k 压缩条目
      3. 对选中条目执行 Multi-Query Attention (MQA)
      4. 分组输出投影

    附加:
      - 滑动窗口注意力分支（保留局部细粒度依赖）
      - 部分 RoPE
      - Attention Sink
      - Query/KV RMSNorm
    """

    def __init__(self, config: DeepSeekV4Config):
        super().__init__()
        self.config = config
        self.d = config.hidden_size
        self.n_h = config.num_query_heads
        self.c = config.head_dim
        self.d_c = config.query_compress_dim
        self.m = config.csa_compress_rate
        self.top_k = config.csa_top_k
        self.n_win = config.sliding_window_size
        self.g = config.num_output_groups
        self.d_g = config.group_output_dim
        self.c_I = config.csa_indexer_head_dim
        self.n_I_h = config.csa_indexer_num_heads

        # KV 压缩器 (双分支)
        self.kv_compressor = KVCompressor(
            self.d, self.c, self.m,
            dual_branch=True, name_prefix="csa"
        )

        # Lightning Indexer
        self.indexer = LightningIndexer(
            self.d, self.d_c, self.c_I, self.n_I_h, self.m, self.top_k
        )

        # Query 低秩投影: down-projection + up-projection
        self.W_DQ = nn.Dense(self.d, self.d_c, has_bias=False)
        self.W_UQ = nn.Dense(self.d_c, self.c * self.n_h, has_bias=False)

        # 滑动窗口 KV 投影
        self.W_swa_kv = nn.Dense(self.d, self.c, has_bias=False)

        # Query/KV RMSNorm (每个 head 独立)
        self.q_norm = RMSNorm(self.c)
        self.kv_norm = RMSNorm(self.c)

        # Attention Sink 可学习参数
        if config.use_attention_sink:
            self.sink_logits = Parameter(
                Tensor(np.zeros(self.n_h, dtype=np.float32), mstype.float32),
                name="csa_sink_logits",
            )

        # 分组输出投影
        self.group_projs = nn.CellList()
        head_per_group = self.n_h // self.g
        for _ in range(self.g):
            self.group_projs.append(
                nn.Dense(self.c * head_per_group, self.d_g, has_bias=False)
            )
        # 最终输出投影
        self.out_proj = nn.Dense(self.d_g * self.g, self.d, has_bias=False)

        # RoPE
        self.rope = RotaryPositionalEmbedding(rope_dim=config.rope_dim)

    def _core_attention(
        self, queries: Tensor, keys: Tensor, values: Tensor
    ) -> Tensor:
        """
        核心注意力计算 (带 Attention Sink)。
        queries: (batch, n_h, num_q, c)
        keys:    (batch, 1, num_kv, c)   [MQA: 共享 KV]
        values:  (batch, 1, num_kv, c)
        返回:    (batch, n_h, num_q, c)
        """
        scale = math.sqrt(self.c)
        # (batch, n_h, num_q, c) x (batch, 1, c, num_kv) -> (batch, n_h, num_q, num_kv)
        attn_logits = ops.matmul(queries, keys.transpose(0, 1, 3, 2)) / scale

        # Attention Sink: 在分母中加入 exp(sink_logit)
        if self.config.use_attention_sink:
            # sink_logits: (n_h,) -> exp -> 加到 softmax 分母
            # 实现: 在 logits 上 concat 一个额外的 sink 维度
            sink = ops.exp(self.sink_logits).reshape(1, self.n_h, 1, 1)
            # 为每个 query 添加 sink 选项
            sink_logits = ops.zeros_like(attn_logits[:, :, :, :1]) + ops.log(sink + 1e-8)
            attn_logits = ops.concat((attn_logits, sink_logits), axis=-1)

        attn_weights = ops.softmax(attn_logits, axis=-1)

        # 去掉 sink 权重（如果有）
        if self.config.use_attention_sink:
            attn_weights = attn_weights[:, :, :, :keys.shape[2]]

        # (batch, n_h, num_q, num_kv) x (batch, 1, num_kv, c) -> (batch, n_h, num_q, c)
        output = ops.matmul(attn_weights, values)
        return output

    def construct(self, H: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        H: (batch, seq_len, d) 输入隐状态
        返回: (batch, seq_len, d) 注意力输出
        """
        batch, seq_len, _ = H.shape

        # --- 1. 压缩 KV 条目 ---
        C_comp = self.kv_compressor(H)  # (batch, n_blocks, c)

        # --- 2. 生成索引 keys ---
        K_indexer = self.indexer.compute_indexer_keys(H)

        # --- 3. 低秩 Query 生成 ---
        c_Q = self.W_DQ(H)  # (batch, seq_len, d_c)
        queries_all = self.W_UQ(c_Q)  # (batch, seq_len, c * n_h)
        queries_all = queries_all.reshape(batch, seq_len, self.n_h, self.c)
        queries_all = queries_all.transpose(0, 2, 1, 3)  # (batch, n_h, seq_len, c)

        # Query RMSNorm
        queries_all = self.q_norm(queries_all)

        # --- 3.5 应用部分 RoPE (论文 Section 2.3.3) ---
        # 对 queries 和 compressed KV 的最后 rope_dim 维施加 RoPE
        queries_all = self.rope(queries_all)
        C_comp_rope = self.rope(C_comp)  # (batch, n_blocks, c)

        # --- 4. 滑动窗口分支 (论文 Section 2.3.3) ---
        # 对每个 query token，额外产生最近 n_win 个 token 的未压缩 KV 条目
        swa_kv = self.W_swa_kv(H)  # (batch, seq_len, c)
        swa_kv = self.kv_norm(swa_kv)
        swa_kv = self.rope(swa_kv)  # 对滑动窗口 KV 也施加 RoPE

        # --- 5. 稀疏选择 + 核心注意力 ---
        # 获取 top-k 索引
        top_k_idx = self.indexer.compute_index_scores(H, c_Q, K_indexer)
        # top_k_idx: (batch, seq_len, top_k)

        # 收集选中的压缩 KV 条目
        n_blocks = C_comp_rope.shape[1]
        # 对每个 query token 收集其 top-k 对应的压缩条目
        # 简化: 使用最后一个 token 的 top-k 索引代表所有 query (近似)
        # 完整实现需 per-query gather，此处为效率取近似
        last_idx = top_k_idx[:, -1, :]  # (batch, top_k)
        selected_kv = ops.gather(
            C_comp_rope,           # (batch, n_blocks, c)
            last_idx,              # (batch, top_k)
            axis=1
        )  # (batch, top_k, c)

        # 拼接压缩 KV + 滑动窗口 KV (论文 Figure 3)
        # 滑动窗口取最后 n_win 个 token 的 KV
        swa_kv_window = swa_kv[:, -self.n_win:, :]  # (batch, n_win, c)
        combined_kv = ops.concat((selected_kv, swa_kv_window), axis=1)
        # (batch, top_k + n_win, c)

        # 转换为 MQA 格式: (batch, 1, top_k + n_win, c)
        keys_mqa = combined_kv.expand_dims(1)
        vals_mqa = keys_mqa  # 共享 KV (MQA)

        # 核心注意力
        attn_out = self._core_attention(queries_all, keys_mqa, vals_mqa)
        # (batch, n_h, seq_len, c)

        # --- 6. 对注意力输出施加反向 RoPE (论文 Section 2.3.3) ---
        # KV 同时充当 key 和 value，输出携带绝对位置信息
        # 对输出的最后 64 维施加 -position RoPE 以恢复相对位置
        # 简化: 此处跳过反向 RoPE (实际部署时需实现)

        # --- 7. 分组输出投影 ---
        attn_out = attn_out.transpose(0, 2, 1, 3)  # (batch, seq_len, n_h, c)
        attn_out = attn_out.reshape(batch, seq_len, self.n_h * self.c)

        head_per_group = self.n_h // self.g
        group_outs = []
        for gi in range(self.g):
            start = gi * head_per_group * self.c
            end = start + head_per_group * self.c
            g_out = attn_out[:, :, start:end]
            g_out = self.group_projs[gi](g_out)  # (batch, seq_len, d_g)
            group_outs.append(g_out)

        concat_out = ops.concat(group_outs, axis=-1)  # (batch, seq_len, d_g * g)
        output = self.out_proj(concat_out)  # (batch, seq_len, d)

        return output


# ============================================================================
# 7. Heavily Compressed Attention (HCA)
# ============================================================================

class HeavilyCompressedAttention(nn.Cell):
    """
    HCA: 重压缩注意力

    与 CSA 类似但:
      - 使用更大的压缩比 m' (>> m)
      - 单分支压缩，无重叠
      - 不使用稀疏选择（全部压缩条目参与注意力）
      - 同样包含滑动窗口分支、部分 RoPE、Attention Sink、分组输出投影
    """

    def __init__(self, config: DeepSeekV4Config):
        super().__init__()
        self.config = config
        self.d = config.hidden_size
        self.n_h = config.num_query_heads
        self.c = config.head_dim
        self.d_c = config.query_compress_dim
        self.m_prime = config.hca_compress_rate
        self.n_win = config.sliding_window_size
        self.g = config.num_output_groups
        self.d_g = config.group_output_dim

        # KV 压缩器 (单分支)
        self.kv_compressor = KVCompressor(
            self.d, self.c, self.m_prime,
            dual_branch=False, name_prefix="hca"
        )

        # Query 低秩投影
        self.W_DQ = nn.Dense(self.d, self.d_c, has_bias=False)
        self.W_UQ = nn.Dense(self.d_c, self.c * self.n_h, has_bias=False)

        # 滑动窗口 KV 投影 (论文 Section 2.3.3: HCA 也包含 SWA 分支)
        self.W_swa_kv = nn.Dense(self.d, self.c, has_bias=False)

        # Query/KV RMSNorm
        self.q_norm = RMSNorm(self.c)
        self.kv_norm = RMSNorm(self.c)

        # Attention Sink
        if config.use_attention_sink:
            self.sink_logits = Parameter(
                Tensor(np.zeros(self.n_h, dtype=np.float32), mstype.float32),
                name="hca_sink_logits",
            )

        # 分组输出投影
        self.group_projs = nn.CellList()
        head_per_group = self.n_h // self.g
        for _ in range(self.g):
            self.group_projs.append(
                nn.Dense(self.c * head_per_group, self.d_g, has_bias=False)
            )
        self.out_proj = nn.Dense(self.d_g * self.g, self.d, has_bias=False)

        # RoPE
        self.rope = RotaryPositionalEmbedding(rope_dim=config.rope_dim)

    def _core_attention(
        self, queries: Tensor, keys: Tensor, values: Tensor
    ) -> Tensor:
        """核心注意力 (MQA + Attention Sink)"""
        scale = math.sqrt(self.c)
        attn_logits = ops.matmul(queries, keys.transpose(0, 1, 3, 2)) / scale

        if self.config.use_attention_sink:
            sink = ops.exp(self.sink_logits).reshape(1, self.n_h, 1, 1)
            sink_logits = ops.zeros_like(attn_logits[:, :, :, :1]) + ops.log(sink + 1e-8)
            attn_logits = ops.concat((attn_logits, sink_logits), axis=-1)

        attn_weights = ops.softmax(attn_logits, axis=-1)

        if self.config.use_attention_sink:
            attn_weights = attn_weights[:, :, :, :keys.shape[2]]

        output = ops.matmul(attn_weights, values)
        return output

    def construct(self, H: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        H: (batch, seq_len, d)
        返回: (batch, seq_len, d)
        """
        batch, seq_len, _ = H.shape

        # --- 1. 重压缩 KV 条目 ---
        C_comp = self.kv_compressor(H)  # (batch, n_blocks', c)
        C_comp = self.kv_norm(C_comp)

        # --- 2. 应用部分 RoPE (论文 Section 2.3.3) ---
        # 对 compressed KV 施加 RoPE
        C_comp = self.rope(C_comp)

        # --- 3. 滑动窗口分支 (论文 Section 2.3.3, Figure 4) ---
        swa_kv = self.W_swa_kv(H)  # (batch, seq_len, c)
        swa_kv = self.kv_norm(swa_kv)
        swa_kv = self.rope(swa_kv)
        swa_kv_window = swa_kv[:, -self.n_win:, :]  # (batch, n_win, c)

        # 拼接: 压缩 KV + 滑动窗口 KV
        combined_kv = ops.concat((C_comp, swa_kv_window), axis=1)
        # (batch, n_blocks' + n_win, c)

        # --- 4. 低秩 Query 生成 ---
        c_Q = self.W_DQ(H)
        queries_all = self.W_UQ(c_Q)
        queries_all = queries_all.reshape(batch, seq_len, self.n_h, self.c)
        queries_all = queries_all.transpose(0, 2, 1, 3)  # (batch, n_h, seq_len, c)
        queries_all = self.q_norm(queries_all)

        # 对 queries 施加 RoPE
        queries_all = self.rope(queries_all)

        # --- 5. MQA (所有压缩条目 + 滑动窗口，无稀疏选择) ---
        keys_mqa = combined_kv.expand_dims(1)  # (batch, 1, total_kv, c)
        vals_mqa = keys_mqa  # 共享 KV

        attn_out = self._core_attention(queries_all, keys_mqa, vals_mqa)
        # (batch, n_h, seq_len, c)

        # --- 6. 分组输出投影 ---
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(
            batch, seq_len, self.n_h * self.c
        )

        head_per_group = self.n_h // self.g
        group_outs = []
        for gi in range(self.g):
            start = gi * head_per_group * self.c
            end = start + head_per_group * self.c
            g_out = attn_out[:, :, start:end]
            g_out = self.group_projs[gi](g_out)
            group_outs.append(g_out)

        concat_out = ops.concat(group_outs, axis=-1)
        output = self.out_proj(concat_out)

        return output


# ============================================================================
# 8. 滑动窗口注意力 (Sliding Window Attention)
# ============================================================================

class SlidingWindowAttention(nn.Cell):
    """
    纯滑动窗口注意力，用于前几层 (Flash) 或作为 CSA/HCA 的辅助分支。

    对每个 query token，只关注最近 n_win 个 token 的 KV 条目，
    以保留局部细粒度依赖。
    """

    def __init__(self, config: DeepSeekV4Config):
        super().__init__()
        self.config = config
        self.d = config.hidden_size
        self.n_h = config.num_query_heads
        self.c = config.head_dim
        self.d_c = config.query_compress_dim
        self.n_win = config.sliding_window_size
        self.g = config.num_output_groups
        self.d_g = config.group_output_dim

        # Query 低秩投影
        self.W_DQ = nn.Dense(self.d, self.d_c, has_bias=False)
        self.W_UQ = nn.Dense(self.d_c, self.c * self.n_h, has_bias=False)

        # KV 投影 (MQA: 共享 KV head)
        self.W_KV = nn.Dense(self.d, self.c, has_bias=False)

        # Norms
        self.q_norm = RMSNorm(self.c)
        self.kv_norm = RMSNorm(self.c)

        # 分组输出投影
        self.group_projs = nn.CellList()
        head_per_group = self.n_h // self.g
        for _ in range(self.g):
            self.group_projs.append(
                nn.Dense(self.c * head_per_group, self.d_g, has_bias=False)
            )
        self.out_proj = nn.Dense(self.d_g * self.g, self.d, has_bias=False)

        # RoPE
        self.rope = RotaryPositionalEmbedding(rope_dim=config.rope_dim)

    def construct(self, H: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        H: (batch, seq_len, d)
        返回: (batch, seq_len, d)
        """
        batch, seq_len, _ = H.shape

        # 低秩 Query
        c_Q = self.W_DQ(H)
        queries = self.W_UQ(c_Q).reshape(batch, seq_len, self.n_h, self.c)
        queries = queries.transpose(0, 2, 1, 3)  # (batch, n_h, seq_len, c)
        queries = self.q_norm(queries)

        # 应用部分 RoPE (论文 Section 2.3.3)
        queries = self.rope(queries)

        # KV (MQA: 共享)
        kv = self.W_KV(H)  # (batch, seq_len, c)
        kv = self.kv_norm(kv)
        kv = self.rope(kv)  # 对 KV 施加 RoPE

        # 滑动窗口 mask: 每个 query 只看最近 n_win 个 token
        # 简化实现: 全局注意力 + causal mask + 窗口截断
        kv_expanded = kv.expand_dims(1)  # (batch, 1, seq_len, c)

        scale = math.sqrt(self.c)
        attn_logits = ops.matmul(
            queries, kv_expanded.transpose(0, 1, 3, 2)
        ) / scale

        # Causal mask + 滑动窗口 mask
        positions = ops.arange(seq_len)
        causal_mask = positions.expand_dims(0) <= positions.expand_dims(1)
        window_mask = (positions.expand_dims(1) - positions.expand_dims(0)) < self.n_win
        mask = causal_mask & window_mask  # (seq_len, seq_len)
        mask = mask.expand_dims(0).expand_dims(0)  # (1, 1, seq_len, seq_len)

        attn_logits = attn_logits * mask + (-1e9) * (1.0 - mask.astype(attn_logits.dtype))
        attn_weights = ops.softmax(attn_logits, axis=-1)

        attn_out = ops.matmul(attn_weights, kv_expanded)
        # (batch, n_h, seq_len, c)

        # 分组输出投影
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(
            batch, seq_len, self.n_h * self.c
        )

        head_per_group = self.n_h // self.g
        group_outs = []
        for gi in range(self.g):
            start = gi * head_per_group * self.c
            end = start + head_per_group * self.c
            g_out = attn_out[:, :, start:end]
            g_out = self.group_projs[gi](g_out)
            group_outs.append(g_out)

        concat_out = ops.concat(group_outs, axis=-1)
        output = self.out_proj(concat_out)
        return output


# ============================================================================
# 9. DeepSeekMoE (Mixture-of-Experts)
# ============================================================================

class SwiGLUExpert(nn.Cell):
    """
    单个 MoE 专家：使用 SwiGLU 激活函数。
    f(x) = (xW_gate ⊙ clamp(xW_up)) · W_down

    SwiGLU Clamping:
      - 线性分量 (up) 限制在 [clamp_min, clamp_max]
      - 门控分量 (gate) 上限为 gate_clamp_max
    """

    def __init__(self, hidden_size: int, intermediate_dim: int,
                 clamp_min: float = -10.0, clamp_max: float = 10.0,
                 gate_max: float = 10.0):
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.gate_max = gate_max

        self.W_gate = nn.Dense(hidden_size, intermediate_dim, has_bias=False)
        self.W_up = nn.Dense(hidden_size, intermediate_dim, has_bias=False)
        self.W_down = nn.Dense(intermediate_dim, hidden_size, has_bias=False)

    def construct(self, x: Tensor) -> Tensor:
        """
        x: (batch * num_tokens, d)
        返回: (batch * num_tokens, d)
        """
        gate = self.W_gate(x)
        up = self.W_up(x)

        # SwiGLU Clamping (论文 Section 4.2.3)
        gate = ops.clip_by_value(gate, self.clamp_min, self.gate_max)
        up = ops.clip_by_value(up, self.clamp_min, self.clamp_max)

        # SwiGLU: SiLU(gate) * up = (gate * sigmoid(gate)) * up
        # 标准 SwiGLU 使用 SiLU/Swish 作为门控激活 (Shazeer, 2020)
        activated = (gate * ops.sigmoid(gate)) * up
        return self.W_down(activated)


class DeepSeekMoE(nn.Cell):
    """
    DeepSeekMoE 层:
      - 1 个共享专家 (始终激活)
      - N 个路由专家 (每 token 激活 k 个)
      - 使用 Sqrt(Softplus(·)) 计算亲和度分数 (替代 Sigmoid)
      - Auxiliary-loss-free 负载均衡 + 序列级 balance loss
      - 前几层支持 Hash 路由

    路由公式:
      score_i = Sqrt(Softplus(h · W_gate_i))
      选取 top-k 个专家，归一化后加权求和
    """

    def __init__(self, config: DeepSeekV4Config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_routed = config.num_routed_experts
        self.num_activated = config.num_activated_experts
        self.use_hash_routing = (layer_idx < config.num_hash_routing_layers)

        # 共享专家
        self.shared_experts = nn.CellList([
            SwiGLUExpert(
                config.hidden_size, config.expert_intermediate_dim,
                config.swiglu_clamp_min, config.swiglu_clamp_max,
                config.gate_clamp_max,
            )
            for _ in range(config.num_shared_experts)
        ])

        # 路由专家
        self.routed_experts = nn.CellList([
            SwiGLUExpert(
                config.hidden_size, config.expert_intermediate_dim,
                config.swiglu_clamp_min, config.swiglu_clamp_max,
                config.gate_clamp_max,
            )
            for _ in range(config.num_routed_experts)
        ])

        # 路由门控网络
        if not self.use_hash_routing:
            self.gate = nn.Dense(config.hidden_size, config.num_routed_experts,
                                 has_bias=False)
            # Auxiliary-loss-free 负载均衡偏置
            self.bias = Parameter(
                Tensor(np.zeros(config.num_routed_experts, dtype=np.float32),
                       mstype.float32),
                name=f"moe_bias_layer{layer_idx}",
            )

    def _hash_routing(self, token_ids: Tensor) -> Tensor:
        """
        Hash 路由: 根据 token ID 的 hash 值确定目标专家。
        token_ids: (batch * num_tokens,)
        返回: (batch * num_tokens, num_activated) 专家索引
        """
        # 简单 hash: token_id % num_routed
        hashes = token_ids % self.num_routed
        # 生成 k 个不同的专家索引
        offsets = Tensor(np.arange(self.num_activated, dtype=np.int32), mstype.int32)
        expert_indices = (hashes.expand_dims(-1) + offsets) % self.num_routed
        return expert_indices

    def _score_routing(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """
        基于亲和度分数的路由。

        h: (batch * num_tokens, d)
        返回:
            weights: (batch * num_tokens, num_activated) 路由权重
            indices: (batch * num_tokens, num_activated) 专家索引
        """
        # 门控分数
        logits = self.gate(h)  # (N, num_routed)

        # Sqrt(Softplus(·)) 激活 (论文 Eq. 不同于 V3 的 Sigmoid)
        scores = ops.sqrt(ops.softplus(logits))

        # 加负载均衡偏置 (auxiliary-loss-free)
        scores = scores + self.bias

        # Top-k 选择
        top_k_scores, top_k_indices = ops.topk(scores, self.num_activated)

        # 归一化
        top_k_scores = top_k_scores / (ops.sum(top_k_scores, axis=-1, keepdims=True) + 1e-8)

        return top_k_scores, top_k_indices

    def construct(
        self, x: Tensor, token_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        x: (batch, seq_len, d) 输入
        token_ids: (batch, seq_len) token ID (Hash 路由用)
        返回:
            output: (batch, seq_len, d)
            balance_loss: 标量，序列级均衡损失
        """
        batch, seq_len, d = x.shape
        x_flat = x.reshape(batch * seq_len, d)

        # --- 共享专家输出 ---
        shared_out = ops.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x_flat)

        # --- 路由专家输出 ---
        if self.use_hash_routing:
            flat_ids = token_ids.reshape(-1) if token_ids is not None else ops.arange(batch * seq_len)
            indices = self._hash_routing(flat_ids)
            weights = ops.ones((batch * seq_len, self.num_activated), mstype.float32)
            weights = weights / self.num_activated
        else:
            weights, indices = self._score_routing(x_flat)

        # 路由专家计算 (简化实现: 逐专家处理)
        routed_out = ops.zeros_like(x_flat)
        for k in range(self.num_activated):
            expert_idx = indices[:, k]  # (N,)
            expert_weight = weights[:, k]  # (N,)

            # 简化: 对所有 token 使用同一个专家（近似）
            # 实际部署时应使用 Expert Parallelism
            for eid in range(self.num_routed):
                mask = (expert_idx == eid).astype(mstype.float32).expand_dims(-1)
                if mask.sum() > 0:
                    expert_out = self.routed_experts[eid](x_flat)
                    routed_out = routed_out + mask * expert_out * expert_weight.expand_dims(-1)

        output = shared_out + routed_out
        output = output.reshape(batch, seq_len, d)

        # --- 序列级均衡损失 ---
        balance_loss = Tensor(0.0, mstype.float32)
        if not self.use_hash_routing:
            # 使用 one_hot 统计各专家被选次数 (可微分近似)
            # indices: (N, num_activated) -> reshape to (batch, seq_len, num_activated)
            idx_batch = indices.reshape(batch, seq_len, self.num_activated)
            # 统计每个序列中各专家的选择次数
            # 对每个 (batch, seq_len, k) 位置的 expert_idx 做 one_hot
            expert_counts = ops.zeros((batch, seq_len, self.num_routed), mstype.float32)
            for k in range(self.num_activated):
                idx_k = idx_batch[:, :, k]  # (batch, seq_len)
                # one_hot -> (batch, seq_len, num_routed)
                hot = ops.one_hot(idx_k, self.num_routed,
                                  Tensor(1.0, mstype.float32),
                                  Tensor(0.0, mstype.float32))
                expert_counts = expert_counts + hot
            # 在序列维度求和 -> (batch, num_routed)
            expert_counts = ops.sum(expert_counts, axis=1)
            # 归一化
            expert_counts = expert_counts / (seq_len * self.num_activated + 1e-8)
            # 均衡损失: 方差
            balance_loss = ops.mean((expert_counts - 1.0 / self.num_routed) ** 2)
            balance_loss = balance_loss * self.config.balance_loss_weight

        return output, balance_loss


# ============================================================================
# 10. Transformer Block
# ============================================================================

class TransformerBlock(nn.Cell):
    """
    DeepSeek-V4 的单个 Transformer Block。

    结构 (如图 2 所示):
      ┌─────────────────────────────────┐
      │  Pre-Block Mixing (mHC 输入映射) │
      │  RMSNorm                        │
      │  Attention (CSA / HCA / SWA)    │
      │  Post-Block Mixing (mHC 残差)   │
      │  Pre-Block Mixing               │
      │  RMSNorm                        │
      │  DeepSeekMoE                    │
      │  Post-Block Mixing (mHC 残差)   │
      └─────────────────────────────────┘

    mHC 在注意力子层和 MoE 子层之后分别更新残差状态。
    """

    def __init__(self, config: DeepSeekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        attn_type = config.layer_attention_types[layer_idx]

        # --- 注意力层 ---
        if attn_type == "swa":
            self.attention = SlidingWindowAttention(config)
        elif attn_type == "csa":
            self.attention = CompressedSparseAttention(config)
        elif attn_type == "hca":
            self.attention = HeavilyCompressedAttention(config)
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")

        # --- FFN (MoE) ---
        self.moe = DeepSeekMoE(config, layer_idx)

        # --- RMSNorm (注意力前 / MoE 前各一个) ---
        self.attn_norm = RMSNorm(config.hidden_size)
        self.moe_norm = RMSNorm(config.hidden_size)

        # --- mHC (两处残差连接各一个) ---
        self.mhc_attn = ManifoldConstrainedHyperConnection(
            config.hidden_size, config.mhc_expansion_factor,
            config.mhc_sinkhorn_iters,
        )
        self.mhc_moe = ManifoldConstrainedHyperConnection(
            config.hidden_size, config.mhc_expansion_factor,
            config.mhc_sinkhorn_iters,
        )

    def construct(
        self,
        X: Tensor,
        token_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        X: (batch, n_hc, d) 残差状态 (mHC 扩展流)
        token_ids: (batch, seq_len)
        attention_mask: 可选注意力掩码

        返回:
            X_new: (batch, n_hc, d) 更新后的残差状态
            balance_loss: MoE 均衡损失
        """
        batch, n_hc, d = X.shape
        seq_len = token_ids.shape[1] if token_ids is not None else 1

        # --- Pre-Block Mixing: 从 mHC 状态提取层输入 ---
        # 使用 mHC 输入映射 A: H = A_l @ X_l (论文 Eq.1)
        h_attn = self.mhc_attn.get_layer_input(X)  # (batch, d)
        # 扩展到序列维度
        H = h_attn.expand_dims(1).tile((1, seq_len, 1))  # (batch, seq_len, d)

        # --- 注意力子层 ---
        H_norm = self.attn_norm(H)
        attn_out = self.attention(H_norm, attention_mask)

        # mHC 残差更新 (注意力后): 对每个 token 的 attn_out 分别更新
        # 简化: 取 attn_out 均值作为整体更新 (完整实现需 per-token mHC)
        attn_mean = ops.mean(attn_out, axis=1)  # (batch, d)
        X = self.mhc_attn(X, attn_mean)

        # --- MoE 子层 ---
        h_moe = self.mhc_moe.get_layer_input(X)  # (batch, d)
        H = h_moe.expand_dims(1).tile((1, seq_len, 1))
        H_norm = self.moe_norm(H)
        moe_out, balance_loss = self.moe(H_norm, token_ids)

        # mHC 残差更新 (MoE 后)
        moe_mean = ops.mean(moe_out, axis=1)  # (batch, d)
        X = self.mhc_moe(X, moe_mean)

        return X, balance_loss


# ============================================================================
# 11. Multi-Token Prediction (MTP)
# ============================================================================

class MultiTokenPrediction(nn.Cell):
    """
    Multi-Token Prediction (MTP) 模块。

    在最终 Transformer 层之后，通过额外的预测头预测未来多个 token，
    增加训练信号密度。MTP 深度设为 1 (预测下一个 token)。

    结构与 DeepSeek-V3 相同:
      - 将最后一层隐状态与 Embedding 拼接
      - 通过一个额外的 Transformer Block
      - 共享 LM Head 输出预测
    """

    def __init__(self, config: DeepSeekV4Config):
        super().__init__()
        self.d = config.hidden_size
        self.mtp_depth = config.mtp_depth

        # MTP 额外 Transformer Block
        self.mtp_blocks = nn.CellList()
        for _ in range(self.mtp_depth):
            self.mtp_blocks.append(
                nn.SequentialCell([
                    RMSNorm(config.hidden_size),
                    nn.Dense(config.hidden_size * 2, config.hidden_size, has_bias=False),
                ])
            )

        # 预测头 (共享 embedding 权重)
        self.pred_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # MTP 损失权重
        self.loss_weight = config.mtp_loss_weight

    def construct(
        self, hidden_states: Tensor, labels: Tensor, embedding: nn.Embedding
    ) -> Tensor:
        """
        MTP: 将最后一层隐状态与 shift-1 的 target embedding 拼接，
        通过额外的 Transformer Block 处理后，共享 LM Head 输出预测。

        hidden_states: (batch, seq_len, d) 最后一层隐状态
        labels:        (batch, seq_len)     目标 token ID (shifted)
        embedding:     token embedding 层 (共享权重)

        返回: (batch, seq_len-1, vocab_size) MTP logits
        """
        batch, seq_len, d = hidden_states.shape

        # Shift: 取 hidden_states[:, :-1] 和 embedding of labels[:, 1:]
        h_prev = hidden_states[:, :-1, :]  # (batch, seq_len-1, d)

        # 获取 shift-1 target 的 embedding
        target_ids = labels[:, 1:]  # (batch, seq_len-1)
        target_emb = embedding(target_ids)  # (batch, seq_len-1, d)

        # 拼接: [h_prev; target_emb] -> (batch, seq_len-1, 2d)
        concat_input = ops.concat((h_prev, target_emb), axis=-1)

        # 通过 MTP Transformer Block
        for mtp_block in self.mtp_blocks:
            concat_input = mtp_block(concat_input)
            # RMSNorm + Dense(2d -> d)

        # 通过预测头 (共享 LM Head 权重)
        logits = self.pred_head(concat_input)  # (batch, seq_len-1, vocab_size)

        return logits


# ============================================================================
# 12. DeepSeek-V4 主模型
# ============================================================================

class DeepSeekV4Model(nn.Cell):
    """
    DeepSeek-V4 完整模型。

    架构概览 (论文 Figure 2):
      ┌──────────────────────────────┐
      │  Token Embedding             │
      │  ┌────────────────────────┐  │
      │  │ Transformer Block × L  │  │
      │  │  - mHC 残差连接        │  │
      │  │  - CSA / HCA 注意力    │  │
      │  │  - DeepSeekMoE         │  │
      │  └────────────────────────┘  │
      │  Post-Block Mixing (mHC)     │
      │  RMSNorm                     │
      │  LM Head (输出预测)          │
      │  MTP Modules (多 token 预测) │
      └──────────────────────────────┘
    """

    def __init__(self, config: DeepSeekV4Config):
        super().__init__()
        self.config = config

        # Token Embedding
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size,
            embedding_table=ms.Tensor(
                np.random.randn(config.vocab_size, config.hidden_size).astype(np.float32) * 0.02,
                mstype.float32,
            ),
        )

        # Transformer Blocks
        self.blocks = nn.CellList()
        for i in range(config.num_layers):
            self.blocks.append(TransformerBlock(config, i))

        # Final RMSNorm
        self.final_norm = RMSNorm(config.hidden_size)

        # LM Head
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # MTP 模块
        self.mtp = MultiTokenPrediction(config)

    def construct(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """
        input_ids: (batch, seq_len) 输入 token ID
        attention_mask: (batch, seq_len) 可选掩码
        labels: (batch, seq_len) 目标 token ID (MTP 需要)

        返回:
            lm_logits:   (batch, seq_len, vocab_size) 语言模型 logits
            mtp_logits:  (batch, seq_len-1, vocab_size) MTP logits (当提供 labels 时)
            total_balance_loss: 标量，所有层 MoE 均衡损失之和
        """
        batch, seq_len = input_ids.shape

        # --- Embedding ---
        H = self.embedding(input_ids)  # (batch, seq_len, d)

        # --- 初始化 mHC 残差状态 ---
        # X_0: (batch, n_hc, d)，将 embedding 复制到 n_hc 个流
        n_hc = self.config.mhc_expansion_factor
        X = H.expand_dims(1).tile((1, n_hc, 1))  # (batch, n_hc, d)
        # 注: 实际需要为每个位置创建状态，这里简化处理

        # --- 通过 Transformer Blocks ---
        total_balance_loss = Tensor(0.0, mstype.float32)
        for block in self.blocks:
            X, balance_loss = block(X, token_ids=input_ids,
                                    attention_mask=attention_mask)
            total_balance_loss = total_balance_loss + balance_loss

        # --- Post-Block Mixing: 从 mHC 最终状态提取隐状态 ---
        # 使用 mHC 输出映射 C 提取最终隐状态
        # 简化: 取均值 (完整实现应使用 output mapping C)
        hidden_states = ops.mean(X, axis=1)  # (batch, d)
        hidden_states = hidden_states.expand_dims(1).tile((1, seq_len, 1))

        # --- Final Norm + LM Head ---
        hidden_states = self.final_norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)  # (batch, seq_len, vocab_size)

        # --- MTP ---
        mtp_logits = None
        if labels is not None:
            mtp_logits = self.mtp(hidden_states, labels, self.embedding)

        return lm_logits, mtp_logits, total_balance_loss

    def compute_loss(
        self,
        input_ids: Tensor,
        labels: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        计算总损失 = LM 交叉熵损失 + MTP 损失 + MoE 均衡损失。

        input_ids: (batch, seq_len)
        labels:    (batch, seq_len) 目标 token ID
        """
        lm_logits, mtp_logits, balance_loss = self.construct(
            input_ids, attention_mask, labels=labels
        )

        # LM 交叉熵损失
        lm_logits_flat = lm_logits[:, :-1, :].reshape(-1, self.config.vocab_size)
        labels_flat = labels[:, 1:].reshape(-1)
        lm_loss = ops.cross_entropy(lm_logits_flat, labels_flat)

        # MTP 损失 (使用 shifted target embeddings)
        mtp_loss = Tensor(0.0, mstype.float32)
        if mtp_logits is not None:
            # mtp_logits: (batch, seq_len-1, vocab_size)
            # 预测目标: labels[:, 1:] (与 LM loss 相同的 shift)
            mtp_labels = labels[:, 1:].reshape(-1)
            mtp_logits_flat = mtp_logits.reshape(-1, self.config.vocab_size)
            mtp_loss = ops.cross_entropy(mtp_logits_flat, mtp_labels)

        # 总损失
        total_loss = lm_loss + self.config.mtp_loss_weight * mtp_loss + balance_loss

        return total_loss


# ============================================================================
# 13. Muon 优化器
# ============================================================================

class MuonOptimizer(nn.Optimizer):
    """
    Muon 优化器 (论文 Algorithm 1)。

    核心特点:
      - 对权重矩阵进行正交化更新 (Newton-Schulz 迭代)
      - Hybrid Newton-Schulz: 前 8 步快速收敛 + 后 2 步精确稳定
      - RMS 缩放以复用 AdamW 超参数
      - Nesterov 动量技巧

    仅用于非 Embedding/Prediction Head/RMSNorm/mHC 静态偏置的参数。
    其余参数使用 AdamW。
    """

    def __init__(
        self,
        params,
        learning_rate: float = 2.7e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        rms_rescale: float = 0.18,
    ):
        super().__init__(learning_rate, params, weight_decay=0.0)
        self.lr = learning_rate
        self.mu = momentum
        self.wd = weight_decay
        self.gamma = rms_rescale

        # 动量缓冲区
        self.moments = [
            Parameter(Tensor(np.zeros(p.shape), mstype.float32),
                      name=f"muon_moment_{i}")
            for i, p in enumerate(self.parameters)
        ]

    def _hybrid_newton_schulz(self, M: Tensor) -> Tensor:
        """
        Hybrid Newton-Schulz 正交化迭代 (论文 Eq.28)。

        阶段 1 (8 步): (a,b,c) = (3.4445, -4.7750, 2.0315) — 快速收敛
        阶段 2 (2 步): (a,b,c) = (2.0, -1.5, 0.5)          — 精确稳定
        """
        # 归一化: M_0 = M / ||M||_F
        norm = ops.sqrt(ops.sum(M ** 2)) + 1e-8
        Mk = M / norm

        # 阶段 1: 快速收敛 (8 步)
        a1, b1, c1 = 3.4445, -4.7750, 2.0315
        for _ in range(8):
            MkT = Mk.transpose()
            MkMkT = ops.matmul(Mk, MkT)
            I = ops.eye(MkMkT.shape[0], MkMkT.shape[0], mstype.float32)
            Mk = a1 * Mk + b1 * ops.matmul(MkMkT - I, Mk) + \
                 c1 * ops.matmul((MkMkT - I) ** 2, Mk)

        # 阶段 2: 精确稳定 (2 步)
        a2, b2, c2 = 2.0, -1.5, 0.5
        for _ in range(2):
            MkT = Mk.transpose()
            MkMkT = ops.matmul(Mk, MkT)
            I = ops.eye(MkMkT.shape[0], MkMkT.shape[0], mstype.float32)
            Mk = a2 * Mk + b2 * ops.matmul(MkMkT - I, Mk) + \
                 c2 * ops.matmul((MkMkT - I) ** 2, Mk)

        return Mk

    def construct(self, gradients):
        """执行一步 Muon 优化。"""
        lr = self.lr
        for i, (param, grad) in enumerate(zip(self.parameters, gradients)):
            if grad is None:
                continue

            moment = self.moments[i]

            # 动量累积 (Nesterov trick: 使用 mu*M + grad)
            new_moment = self.mu * moment + grad
            moment.set_data(new_moment)

            # 正交化更新
            O_prime = self._hybrid_newton_schulz(self.mu * new_moment + grad)

            # RMS 缩放
            n, m = O_prime.shape[0], O_prime.shape[1] if O_prime.ndim > 1 else 1
            scale = math.sqrt(max(n, m)) * self.gamma
            O = O_prime * scale

            # 权重衰减 + 参数更新
            new_param = param * (1.0 - lr * self.wd) - lr * O
            param.set_data(new_param)

        return True
