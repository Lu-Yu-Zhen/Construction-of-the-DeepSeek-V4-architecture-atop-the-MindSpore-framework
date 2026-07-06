"""
DeepSeek-V4 MindSpore Implementation
=====================================
基于论文 "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
"""

from .config import DeepSeekV4Config, flash_config, pro_config
from .normalization import RMSNorm, RotaryPositionalEmbedding
from .mhc import SinkhornKnopp, ManifoldConstrainedHyperConnection
from .attention import (
    KVCompressor,
    LightningIndexer,
    CompressedSparseAttention,
    HeavilyCompressedAttention,
    SlidingWindowAttention,
)
from .moe import SwiGLUExpert, DeepSeekMoE
from .mtp import MultiTokenPrediction
from .model import TransformerBlock, DeepSeekV4Model
from .optimizer import MuonOptimizer

__all__ = [
    # Config
    "DeepSeekV4Config",
    "flash_config",
    "pro_config",
    # Normalization
    "RMSNorm",
    "RotaryPositionalEmbedding",
    # mHC
    "SinkhornKnopp",
    "ManifoldConstrainedHyperConnection",
    # Attention
    "KVCompressor",
    "LightningIndexer",
    "CompressedSparseAttention",
    "HeavilyCompressedAttention",
    "SlidingWindowAttention",
    # MoE
    "SwiGLUExpert",
    "DeepSeekMoE",
    # MTP
    "MultiTokenPrediction",
    # Model
    "TransformerBlock",
    "DeepSeekV4Model",
    # Optimizer
    "MuonOptimizer",
]
