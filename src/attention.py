"""
注意力机制模块
==============
实现 DeepSeek-V4 的三种注意力机制:
  1. CompressedSparseAttention (CSA) — 压缩 + 稀疏注意力
  2. HeavilyCompressedAttention (HCA) — 重压缩 + 稠密注意力
  3. SlidingWindowAttention (SWA) — 纯滑动窗口注意力

所有注意力均包含:
  - 部分 RoPE (仅对最后 rope_dim 维施加)
  - Attention Sink 机制
  - Query/KV RMSNorm
  - 滑动窗口辅助分支
  - 分组输出投影 (Grouped Output Projection)
"""

import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor, dtype as mstype
from typing import Optional

from .config import DeepSeekV4Config
from .normalization import RMSNorm, RotaryPositionalEmbedding


# ============================================================================
# KV 压缩器 (CSA / HCA 共用)
# ============================================================================

class KVCompressor(nn.Cell):
    """
    Token 级 KV 压缩器。

    将每 m 个 token 的 KV 条目压缩为 1 个条目。
    使用可学习的压缩权重 Z 和位置偏置 B，通过 Softmax 加权聚合。

    CSA 使用双分支压缩 (Ca, Cb) 且带重叠 (论文 Eq.9-12)；
    HCA 使用单分支压缩 (C) 无重叠 (论文 Eq.20-23)。
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
        m = compress_rate
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
        单分支压缩 (用于 HCA)。

        H: (batch, seq_len, d)
        返回: (batch, seq_len // m, head_dim)

        论文 Eq.22-23:
            S = Softmax_row(Z + B)
            C_comp_i = sum_j S_j * C_j
        """
        C = W_kv(H)
        Z = W_z(H)

        batch, seq_len, c = C.shape
        n_blocks = seq_len // m

        C = C[:, :n_blocks * m, :]
        Z = Z[:, :n_blocks * m, :]

        C = C.reshape(batch, n_blocks, m, c)
        Z = Z.reshape(batch, n_blocks, m, c)

        S = ops.softmax(Z + bias, axis=2)
        C_comp = ops.sum(S * C, axis=2)
        return C_comp

    def construct(self, H: Tensor) -> Tensor:
        """
        H: (batch, seq_len, d) 输入隐状态
        返回: (batch, num_compressed, c) 压缩后的 KV 条目
        """
        m = self.compress_rate

        if not self.dual_branch:
            return self._compress_single(H, self.W_kv_a, self.W_z_a, self.bias_a, m)

        # CSA 双分支重叠压缩 (论文 Eq.9-12)
        Ca = self.W_kv_a(H)
        Cb = self.W_kv_b(H)
        Za = self.W_z_a(H)
        Zb = self.W_z_b(H)

        batch, seq_len, c = Ca.shape
        n_blocks = seq_len // m

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
            (Za + self.bias_a,
             Zb_padded[:, :n_blocks * m, :].reshape(batch, n_blocks, m, c) + self.bias_b),
            axis=2
        )
        S = ops.softmax(Z_cat, axis=2)

        Sa = S[:, :, :m, :]
        Sb = S[:, :, m:, :]

        Cb_shifted = Cb[:, :n_blocks * m, :]
        Cb_shifted = ops.pad(Cb_shifted, ((0, 0), (m, 0), (0, 0)),
                             mode='constant', value=0.0)
        Cb_shifted = Cb_shifted[:, :n_blocks * m, :].reshape(batch, n_blocks, m, c)

        C_comp = ops.sum(Sa * Ca, axis=2) + ops.sum(Sb * Cb_shifted, axis=2)
        return C_comp


# ============================================================================
# Lightning Indexer (CSA 专用稀疏选择)
# ============================================================================

class LightningIndexer(nn.Cell):
    """
    Lightning Indexer: 为 CSA 的稀疏注意力提供 top-k 索引选择。

    流程 (论文 Section 2.3.1):
      1. 对压缩 KV 条目再做一次压缩得到 indexer keys K^I_comp
      2. 用低秩方式生成 indexer queries: q^I = c_Q @ W_IUQ
      3. 计算 index score: I_{t,s} = sum_h w^I_{t,h} * ReLU(q^I_{t,h} . K^I_s)
      4. 选取 top-k 个压缩条目
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

        # Index score 权重 (论文 Eq.15)
        self.W_w = nn.Dense(hidden_size, indexer_num_heads, has_bias=False)

    def compute_indexer_keys(self, H: Tensor) -> Tensor:
        """
        计算压缩 indexer keys (与 KV 压缩相同的方式)。

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
        K_comp = ops.sum(S * C, axis=2)
        return K_comp

    def compute_index_scores(
        self, h_query: Tensor, c_Q: Tensor, K_indexer: Tensor
    ) -> Tensor:
        """
        计算 index scores 并返回 top-k 索引。

        h_query:   (batch, num_queries, d)      查询 token 的隐状态
        c_Q:       (batch, num_queries, d_c)    压缩查询向量
        K_indexer: (batch, n_blocks, c_I)       压缩 indexer keys

        返回: (batch, num_queries, top_k) top-k 索引
        """
        # 生成 indexer queries (论文 Eq.14)
        q_I = self.W_IUQ(c_Q)
        batch, num_q, _ = q_I.shape
        q_I = q_I.reshape(batch, num_q, self.indexer_num_heads, self.c_I)

        # Index score 权重 (论文 Eq.15)
        w_I = self.W_w(h_query)

        # 计算 score: sum_h w_h * ReLU(q_h . K_s) (论文 Eq.16)
        dots = ops.matmul(
            q_I, K_indexer.expand_dims(1).transpose(0, 1, 3, 2)
        )
        dots = ops.relu(dots)

        w_I = w_I.expand_dims(-1)
        scores = ops.sum(w_I * dots, axis=2)

        # Top-k 选择 (论文 Eq.17)
        _, top_k_idx = ops.topk(scores, self.top_k)
        return top_k_idx


# ============================================================================
# Compressed Sparse Attention (CSA)
# ============================================================================

class CompressedSparseAttention(nn.Cell):
    """
    CSA: 压缩稀疏注意力 (论文 Section 2.3.1, Figure 3)

    核心流程:
      1. 将 KV 缓存每 m 个 token 压缩为 1 个条目 (双分支重叠压缩)
      2. 用 Lightning Indexer 选取 top-k 压缩条目
      3. 对选中条目 + 滑动窗口 KV 执行 Multi-Query Attention (MQA)
      4. 分组输出投影 (Grouped Output Projection)

    附加机制:
      - 部分 RoPE (仅对最后 64 维施加)
      - Attention Sink
      - Query/KV RMSNorm
      - 滑动窗口辅助分支 (保留局部细粒度依赖)
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

        # Attention Sink 可学习参数 (论文 Eq.27)
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
        self.out_proj = nn.Dense(self.d_g * self.g, self.d, has_bias=False)

        # RoPE (部分旋转位置编码)
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
        attn_logits = ops.matmul(queries, keys.transpose(0, 1, 3, 2)) / scale

        # Attention Sink: 在 softmax 分母中加入 exp(z'_h) (论文 Eq.27)
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
        H: (batch, seq_len, d) 输入隐状态
        返回: (batch, seq_len, d) 注意力输出
        """
        batch, seq_len, _ = H.shape

        # --- 1. 压缩 KV 条目 ---
        C_comp = self.kv_compressor(H)

        # --- 2. 生成索引 keys ---
        K_indexer = self.indexer.compute_indexer_keys(H)

        # --- 3. 低秩 Query 生成 ---
        c_Q = self.W_DQ(H)
        queries_all = self.W_UQ(c_Q)
        queries_all = queries_all.reshape(batch, seq_len, self.n_h, self.c)
        queries_all = queries_all.transpose(0, 2, 1, 3)

        # Query RMSNorm
        queries_all = self.q_norm(queries_all)

        # --- 3.5 应用部分 RoPE (论文 Section 2.3.3) ---
        queries_all = self.rope(queries_all)
        C_comp_rope = self.rope(C_comp)

        # --- 4. 滑动窗口分支 (论文 Section 2.3.3) ---
        swa_kv = self.W_swa_kv(H)
        swa_kv = self.kv_norm(swa_kv)
        swa_kv = self.rope(swa_kv)

        # --- 5. 稀疏选择 + 核心注意力 ---
        top_k_idx = self.indexer.compute_index_scores(H, c_Q, K_indexer)

        # 收集选中的压缩 KV 条目 (使用最后一个 token 的索引作为近似)
        last_idx = top_k_idx[:, -1, :]
        selected_kv = ops.gather(C_comp_rope, last_idx, axis=1)

        # 拼接压缩 KV + 滑动窗口 KV
        swa_kv_window = swa_kv[:, -self.n_win:, :]
        combined_kv = ops.concat((selected_kv, swa_kv_window), axis=1)

        # MQA 格式
        keys_mqa = combined_kv.expand_dims(1)
        vals_mqa = keys_mqa

        attn_out = self._core_attention(queries_all, keys_mqa, vals_mqa)

        # --- 6. 分组输出投影 ---
        attn_out = attn_out.transpose(0, 2, 1, 3)
        attn_out = attn_out.reshape(batch, seq_len, self.n_h * self.c)

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
# Heavily Compressed Attention (HCA)
# ============================================================================

class HeavilyCompressedAttention(nn.Cell):
    """
    HCA: 重压缩注意力 (论文 Section 2.3.2, Figure 4)

    与 CSA 类似但:
      - 使用更大的压缩比 m' (>> m)，论文中 m'=128
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

        # 滑动窗口 KV 投影
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
        C_comp = self.kv_compressor(H)
        C_comp = self.kv_norm(C_comp)

        # --- 2. 应用部分 RoPE ---
        C_comp = self.rope(C_comp)

        # --- 3. 滑动窗口分支 ---
        swa_kv = self.W_swa_kv(H)
        swa_kv = self.kv_norm(swa_kv)
        swa_kv = self.rope(swa_kv)
        swa_kv_window = swa_kv[:, -self.n_win:, :]

        # 拼接: 压缩 KV + 滑动窗口 KV
        combined_kv = ops.concat((C_comp, swa_kv_window), axis=1)

        # --- 4. 低秩 Query 生成 ---
        c_Q = self.W_DQ(H)
        queries_all = self.W_UQ(c_Q)
        queries_all = queries_all.reshape(batch, seq_len, self.n_h, self.c)
        queries_all = queries_all.transpose(0, 2, 1, 3)
        queries_all = self.q_norm(queries_all)
        queries_all = self.rope(queries_all)

        # --- 5. MQA (所有压缩条目 + 滑动窗口，无稀疏选择) ---
        keys_mqa = combined_kv.expand_dims(1)
        vals_mqa = keys_mqa

        attn_out = self._core_attention(queries_all, keys_mqa, vals_mqa)

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
# 滑动窗口注意力 (Sliding Window Attention)
# ============================================================================

class SlidingWindowAttention(nn.Cell):
    """
    纯滑动窗口注意力，用于前几层 (Flash 前2层 / Pro 前2层)。

    对每个 query token，只关注最近 n_win 个 token 的 KV 条目，
    以保留局部细粒度依赖。使用低秩 Query 投影 + MQA + 分组输出投影。
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
        queries = queries.transpose(0, 2, 1, 3)
        queries = self.q_norm(queries)
        queries = self.rope(queries)

        # KV (MQA: 共享)
        kv = self.W_KV(H)
        kv = self.kv_norm(kv)
        kv = self.rope(kv)

        kv_expanded = kv.expand_dims(1)

        scale = math.sqrt(self.c)
        attn_logits = ops.matmul(
            queries, kv_expanded.transpose(0, 1, 3, 2)
        ) / scale

        # Causal mask + 滑动窗口 mask
        positions = ops.arange(seq_len)
        causal_mask = positions.expand_dims(0) <= positions.expand_dims(1)
        window_mask = (positions.expand_dims(1) - positions.expand_dims(0)) < self.n_win
        mask = causal_mask & window_mask
        mask = mask.expand_dims(0).expand_dims(0)

        attn_logits = attn_logits * mask + (-1e9) * (1.0 - mask.astype(attn_logits.dtype))
        attn_weights = ops.softmax(attn_logits, axis=-1)

        attn_out = ops.matmul(attn_weights, kv_expanded)

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
