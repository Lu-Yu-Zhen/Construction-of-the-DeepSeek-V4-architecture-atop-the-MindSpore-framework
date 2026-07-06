"""
DeepSeekMoE (Mixture-of-Experts)
=================================
实现 DeepSeek-V4 的 MoE 层:
  - 1 个共享专家 (始终激活)
  - N 个路由专家 (每 token 激活 k 个)
  - Sqrt(Softplus(·)) 计算亲和度分数 (替代 V3 的 Sigmoid)
  - Auxiliary-loss-free 负载均衡 + 序列级 balance loss
  - 前几层使用 Hash 路由
  - SwiGLU 激活 + Clamping 稳定训练
"""

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor, dtype as mstype
from typing import Optional, Tuple

from .config import DeepSeekV4Config


class SwiGLUExpert(nn.Cell):
    """
    单个 MoE 专家：使用 SwiGLU 激活函数。

    f(x) = (SiLU(xW_gate) * clamp(xW_up)) · W_down
         = (xW_gate * sigmoid(xW_gate) * clamp(xW_up)) · W_down

    SwiGLU Clamping (论文 Section 4.2.3):
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

        # SwiGLU Clamping
        gate = ops.clip_by_value(gate, self.clamp_min, self.gate_max)
        up = ops.clip_by_value(up, self.clamp_min, self.clamp_max)

        # SwiGLU: SiLU(gate) * up = gate * sigmoid(gate) * up
        activated = (gate * ops.sigmoid(gate)) * up
        return self.W_down(activated)


class DeepSeekMoE(nn.Cell):
    """
    DeepSeekMoE 层:
      - 1 个共享专家 (始终激活)
      - N 个路由专家 (每 token 激活 k 个)
      - 使用 Sqrt(Softplus(·)) 计算亲和度分数 (替代 V3 的 Sigmoid)
      - Auxiliary-loss-free 负载均衡 + 序列级 balance loss
      - 前几层使用 Hash 路由 (根据 token ID 的 hash 值确定目标专家)

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
        hashes = token_ids % self.num_routed
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
        logits = self.gate(h)

        # Sqrt(Softplus(·)) 激活 (替代 V3 的 Sigmoid)
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

        # 路由专家计算
        routed_out = ops.zeros_like(x_flat)
        for k in range(self.num_activated):
            expert_idx = indices[:, k]
            expert_weight = weights[:, k]

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
            idx_batch = indices.reshape(batch, seq_len, self.num_activated)
            expert_counts = ops.zeros((batch, seq_len, self.num_routed), mstype.float32)
            for k in range(self.num_activated):
                idx_k = idx_batch[:, :, k]
                hot = ops.one_hot(idx_k, self.num_routed,
                                  Tensor(1.0, mstype.float32),
                                  Tensor(0.0, mstype.float32))
                expert_counts = expert_counts + hot
            expert_counts = ops.sum(expert_counts, axis=1)
            expert_counts = expert_counts / (seq_len * self.num_activated + 1e-8)
            balance_loss = ops.mean((expert_counts - 1.0 / self.num_routed) ** 2)
            balance_loss = balance_loss * self.config.balance_loss_weight

        return output, balance_loss
