"""
Muon 优化器
============
论文 Algorithm 1 实现。

核心特点:
  - 对权重矩阵进行正交化更新 (Newton-Schulz 迭代)
  - Hybrid Newton-Schulz: 前 8 步快速收敛 + 后 2 步精确稳定
  - RMS 缩放以复用 AdamW 超参数
  - Nesterov 动量技巧

仅用于非 Embedding/Prediction Head/RMSNorm/mHC 静态偏置的参数。
其余参数使用 AdamW。
"""

import math

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor, dtype as mstype


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
            Parameter(
                Tensor(np.zeros(p.shape), mstype.float32),
                name=f"muon_moment_{i}",
            )
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
            Mk = (a1 * Mk
                  + b1 * ops.matmul(MkMkT - I, Mk)
                  + c1 * ops.matmul((MkMkT - I) ** 2, Mk))

        # 阶段 2: 精确稳定 (2 步)
        a2, b2, c2 = 2.0, -1.5, 0.5
        for _ in range(2):
            MkT = Mk.transpose()
            MkMkT = ops.matmul(Mk, MkT)
            I = ops.eye(MkMkT.shape[0], MkMkT.shape[0], mstype.float32)
            Mk = (a2 * Mk
                  + b2 * ops.matmul(MkMkT - I, Mk)
                  + c2 * ops.matmul((MkMkT - I) ** 2, Mk))

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
            n = O_prime.shape[0]
            m = O_prime.shape[1] if O_prime.ndim > 1 else 1
            scale = math.sqrt(max(n, m)) * self.gamma
            O = O_prime * scale

            # 权重衰减 + 参数更新
            new_param = param * (1.0 - lr * self.wd) - lr * O
            param.set_data(new_param)

        return True
