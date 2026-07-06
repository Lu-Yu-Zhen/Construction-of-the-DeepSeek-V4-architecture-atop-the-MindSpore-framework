"""
Manifold-Constrained Hyper-Connections (mHC)
=============================================
替代标准残差连接，将残差流宽度扩展 n_hc 倍。

核心公式 (论文 Eq.1):
    X_{l+1} = B_l @ X_l + C_l * F_l(A_l @ X_l)

其中:
  - A_l: 输入映射 (Sigmoid 约束，非负有界)
  - B_l: 残差映射 (约束在双随机矩阵流形上，通过 Sinkhorn-Knopp 算法)
  - C_l: 输出映射 (2 * Sigmoid 约束)

参数动态生成:
  由输入 X_l 经 RMSNorm + 线性变换 + 静态偏置动态产生。
"""

import numpy as np
import mindspore as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor, dtype as mstype
from typing import Tuple

from .normalization import RMSNorm


class SinkhornKnopp(nn.Cell):
    """
    Sinkhorn-Knopp 迭代算法：
    将矩阵投影到双随机矩阵流形 (Birkhoff polytope)。

    论文 Eq.8:
        M^(0) = exp(B_tilde)
        M^(t) = T_r(T_c(M^(t-1)))
    其中 T_r, T_c 分别为行归一化和列归一化。
    """

    def __init__(self, num_iters: int = 20):
        super().__init__()
        self.num_iters = num_iters

    def construct(self, M: Tensor) -> Tensor:
        """
        M: (..., n, n) 原始矩阵
        返回: (..., n, n) 双随机矩阵
        """
        M = ops.exp(M)
        for _ in range(self.num_iters):
            col_sum = ops.sum(M, axis=-2, keepdims=True) + 1e-8
            M = M / col_sum
            row_sum = ops.sum(M, axis=-1, keepdims=True) + 1e-8
            M = M / row_sum
        return M


class ManifoldConstrainedHyperConnection(nn.Cell):
    """
    mHC: 流形约束超连接，替代标准残差连接。

    更新公式 (论文 Eq.1):
        X_{l+1} = B_l @ X_l + C_l * F_l(A_l @ X_l)

    动态参数生成 (论文 Eq.3-5):
        A_tilde = alpha_pre * (X_hat @ W_pre) + S_pre
        B_tilde = alpha_res * Mat(X_hat @ W_res) + S_res
        C_tilde = alpha_post * (X_hat @ W_post)^T + S_post

    约束施加 (论文 Eq.6-8):
        A = Sigmoid(A_tilde)
        C = 2 * Sigmoid(C_tilde)
        B = SinkhornKnopp(B_tilde)
    """

    def __init__(self, hidden_size: int, n_hc: int, sinkhorn_iters: int = 20):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_hc = n_hc
        self.d = hidden_size

        # 归一化展平输入的 RMSNorm
        self.pre_norm = RMSNorm(n_hc * hidden_size)

        # 动态分量的可学习参数
        self.W_pre = Parameter(
            Tensor(np.random.randn(n_hc * hidden_size, n_hc).astype(np.float32) * 0.01,
                   mstype.float32),
            name="mhc_W_pre",
        )
        self.W_res = Parameter(
            Tensor(np.random.randn(n_hc * hidden_size, n_hc * n_hc).astype(np.float32) * 0.01,
                   mstype.float32),
            name="mhc_W_res",
        )
        self.W_post = Parameter(
            Tensor(np.random.randn(n_hc * hidden_size, n_hc).astype(np.float32) * 0.01,
                   mstype.float32),
            name="mhc_W_post",
        )

        # 静态偏置
        self.S_pre = Parameter(
            Tensor(np.zeros((1, n_hc), dtype=np.float32), mstype.float32),
            name="mhc_S_pre",
        )
        self.S_res = Parameter(
            Tensor(np.eye(n_hc, dtype=np.float32), mstype.float32),
            name="mhc_S_res",
        )
        self.S_post = Parameter(
            Tensor(np.zeros((n_hc, 1), dtype=np.float32), mstype.float32),
            name="mhc_S_post",
        )

        # 可学习门控因子 (初始化为小值)
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
        X_hat = self.pre_norm(X_flat)

        # 生成无约束原始参数 (论文 Eq.3-5)
        A_tilde = self.alpha_pre * ops.matmul(X_hat, self.W_pre) + self.S_pre
        B_flat = ops.matmul(X_hat, self.W_res)
        B_tilde = B_flat.reshape(batch, self.n_hc, self.n_hc)
        B_tilde = self.alpha_res * B_tilde + self.S_res
        C_tilde = self.alpha_post * ops.matmul(X_hat, self.W_post).transpose(0, 2, 1)
        C_tilde = C_tilde + self.S_post

        # 施加约束 (论文 Eq.6-8)
        A = ops.sigmoid(A_tilde)
        C = 2.0 * ops.sigmoid(C_tilde)
        B = self.sinkhorn(B_tilde)

        return A, B, C

    def get_layer_input(self, X: Tensor) -> Tensor:
        """
        使用输入映射 A 从残差状态中提取实际层输入 (论文 Eq.1: A_l @ X_l)。

        X: (batch, n_hc, d) 残差状态
        返回: (batch, d) 层输入
        """
        A, _, _ = self._generate_parameters(X)
        layer_input = ops.matmul(A, X).squeeze(1)
        return layer_input

    def construct(self, X: Tensor, F_out: Tensor) -> Tensor:
        """
        mHC 更新步骤。

        X:     (batch, n_hc, d)  当前残差状态
        F_out: (batch, d)        当前子层输出 (注意力或 MoE)
        返回:  (batch, n_hc, d)  更新后的残差状态
        """
        _, B, C = self._generate_parameters(X)

        # X_{l+1} = B @ X + C * F_out
        BX = ops.matmul(B, X)
        CF = C * F_out.expand_dims(1)

        return BX + CF
