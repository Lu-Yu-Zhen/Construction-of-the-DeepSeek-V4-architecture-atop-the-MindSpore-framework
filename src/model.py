"""
DeepSeek-V4 主模型
===================
包含 TransformerBlock 和 DeepSeekV4Model 完整模型。

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

from typing import Optional, Tuple

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, dtype as mstype

from .config import DeepSeekV4Config
from .normalization import RMSNorm
from .mhc import ManifoldConstrainedHyperConnection
from .attention import (
    CompressedSparseAttention,
    HeavilyCompressedAttention,
    SlidingWindowAttention,
)
from .moe import DeepSeekMoE
from .mtp import MultiTokenPrediction


# ============================================================================
# Transformer Block
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
# DeepSeek-V4 主模型
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
                np.random.randn(config.vocab_size, config.hidden_size)
                .astype(np.float32) * 0.02,
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
        self.lm_head = nn.Dense(
            config.hidden_size, config.vocab_size, has_bias=False
        )

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
            mtp_logits:  (batch, seq_len-1, vocab_size) MTP logits
                         (当提供 labels 时)
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
            X, balance_loss = block(
                X, token_ids=input_ids, attention_mask=attention_mask
            )
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
