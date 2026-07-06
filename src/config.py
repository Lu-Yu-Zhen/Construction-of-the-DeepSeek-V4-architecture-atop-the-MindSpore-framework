"""
DeepSeek-V4 模型配置类
=====================
定义 Flash (284B) 和 Pro (1.6T) 两种模型配置。
"""

from dataclasses import dataclass, field
from typing import Optional, List


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
        这里以 Flash 为默认；Pro 在 pro_config() 中覆盖。
        """
        schedule = []
        for i in range(self.num_layers):
            if i < 2:
                schedule.append("swa")
            else:
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
        layer_attention_types=None,
    )


def pro_config() -> DeepSeekV4Config:
    """DeepSeek-V4-Pro: 1.6T 总参数, 49B 激活参数"""
    num_layers = 61
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
