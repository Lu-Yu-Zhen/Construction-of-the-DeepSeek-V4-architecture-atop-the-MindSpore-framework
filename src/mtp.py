"""
Multi-Token Prediction (MTP) 模块
==================================
在最终 Transformer 层之后，通过额外的预测头预测未来多个 token，
增加训练信号密度。MTP 深度设为 1 (预测下一个 token)。

结构与 DeepSeek-V3 相同:
  - 将最后一层隐状态与 shift-1 target embedding 拼接
  - 通过一个额外的 Transformer Block
  - 共享 LM Head 输出预测
"""

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, dtype as mstype

from .config import DeepSeekV4Config
from .normalization import RMSNorm


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
                    nn.Dense(config.hidden_size * 2, config.hidden_size,
                             has_bias=False),
                ])
            )

        # 预测头 (共享 embedding 权重)
        self.pred_head = nn.Dense(
            config.hidden_size, config.vocab_size, has_bias=False
        )

        # MTP 损失权重
        self.loss_weight = config.mtp_loss_weight

    def construct(
        self,
        hidden_states: Tensor,
        labels: Tensor,
        embedding: nn.Embedding,
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
