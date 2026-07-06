"""
基础组件
========
  - RMSNorm: Root Mean Square Layer Normalization
  - RotaryPositionalEmbedding: 旋转位置编码 (RoPE)，仅对最后 rope_dim 维施加
"""

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor, dtype as mstype
from typing import Optional


class RMSNorm(nn.Cell):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(
            Tensor(np.ones(dim), mstype.float32), name="rmsnorm_weight"
        )

    def construct(self, x: Tensor) -> Tensor:
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
        self._freqs_cos = None
        self._freqs_sin = None

    def _build_freqs(self, seq_len: int):
        """构建 cos/sin 频率表"""
        half_dim = self.rope_dim // 2
        freqs = 1.0 / (
            self.base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim)
        )
        positions = np.arange(seq_len, dtype=np.float32)
        angles = np.outer(positions, freqs)
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

        cos = self._freqs_cos[:seq_len]
        sin = self._freqs_sin[:seq_len]

        while cos.ndim < x_rope.ndim:
            cos = cos.expand_dims(0)
            sin = sin.expand_dims(0)

        cos = ops.concat((cos, cos), axis=-1)
        sin = ops.concat((sin, sin), axis=-1)

        x_rotated = x_rope * cos + self._rotate_half(x_rope) * sin
        return ops.concat((x_pass, x_rotated), axis=-1)
