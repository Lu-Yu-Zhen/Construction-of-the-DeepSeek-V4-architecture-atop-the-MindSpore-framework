"""
DeepSeek-V4 模型单元测试
=========================
测试各模块的基本功能，确保模型结构正确。

运行:
  python -m pytest tests/test_model.py -v
"""

import sys
import os
import unittest

import numpy as np

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mindspore as ms
from mindspore import Tensor, dtype as mstype
import mindspore.ops as ops

from src.config import DeepSeekV4Config, flash_config, pro_config
from src.normalization import RMSNorm, RotaryPositionalEmbedding
from src.mhc import SinkhornKnopp, ManifoldConstrainedHyperConnection
from src.attention import (
    KVCompressor,
    LightningIndexer,
    SlidingWindowAttention,
)
from src.moe import SwiGLUExpert, DeepSeekMoE
from src.mtp import MultiTokenPrediction
from src.model import TransformerBlock, DeepSeekV4Model


class TestConfig(unittest.TestCase):
    """测试配置类。"""

    def test_flash_config(self):
        config = flash_config()
        self.assertEqual(config.num_layers, 43)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_query_heads, 64)
        self.assertEqual(config.head_dim, 512)
        self.assertEqual(config.csa_compress_rate, 4)
        self.assertEqual(config.hca_compress_rate, 128)
        self.assertEqual(config.num_routed_experts, 256)
        self.assertEqual(config.mhc_expansion_factor, 4)
        self.assertEqual(len(config.layer_attention_types), 43)
        self.assertEqual(config.layer_attention_types[0], "swa")
        self.assertEqual(config.layer_attention_types[1], "swa")
        self.assertEqual(config.layer_attention_types[2], "csa")
        self.assertEqual(config.layer_attention_types[3], "hca")

    def test_pro_config(self):
        config = pro_config()
        self.assertEqual(config.num_layers, 61)
        self.assertEqual(config.hidden_size, 7168)
        self.assertEqual(config.num_query_heads, 128)
        self.assertEqual(config.num_routed_experts, 384)
        self.assertEqual(len(config.layer_attention_types), 61)
        self.assertEqual(config.layer_attention_types[0], "hca")
        self.assertEqual(config.layer_attention_types[1], "hca")
        self.assertEqual(config.layer_attention_types[2], "csa")


class TestRMSNorm(unittest.TestCase):
    """测试 RMSNorm。"""

    def test_output_shape(self):
        norm = RMSNorm(64)
        x = Tensor(np.random.randn(2, 10, 64).astype(np.float32), mstype.float32)
        out = norm(x)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_normalization_effect(self):
        norm = RMSNorm(32)
        x = Tensor(np.random.randn(4, 32).astype(np.float32) * 10, mstype.float32)
        out = norm(x)
        # RMSNorm 后 RMS 应接近 1 (乘以 weight=1)
        rms = np.sqrt(np.mean(out.asnumpy() ** 2, axis=-1))
        np.testing.assert_allclose(rms, np.ones_like(rms), atol=0.1)


class TestRoPE(unittest.TestCase):
    """测试旋转位置编码。"""

    def test_output_shape(self):
        rope = RotaryPositionalEmbedding(rope_dim=64)
        x = Tensor(np.random.randn(2, 10, 128).astype(np.float32), mstype.float32)
        out = rope(x)
        self.assertEqual(out.shape, (2, 10, 128))

    def test_rope_only_applied_to_last_dims(self):
        rope = RotaryPositionalEmbedding(rope_dim=32)
        x = Tensor(np.random.randn(1, 5, 64).astype(np.float32), mstype.float32)
        out = rope(x)
        # 前 32 维 (64-32) 应保持不变
        np.testing.assert_array_equal(
            x.asnumpy()[..., :32], out.asnumpy()[..., :32]
        )


class TestSinkhornKnopp(unittest.TestCase):
    """测试 Sinkhorn-Knopp 算法。"""

    def test_doubly_stochastic(self):
        sk = SinkhornKnopp(num_iters=20)
        M = Tensor(np.random.randn(4, 4).astype(np.float32), mstype.float32)
        result = sk(M).asnumpy()
        # 行和应接近 1
        row_sums = np.sum(result, axis=-1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-3)
        # 列和应接近 1
        col_sums = np.sum(result, axis=-2)
        np.testing.assert_allclose(col_sums, np.ones_like(col_sums), atol=1e-3)
        # 所有元素应为非负
        self.assertTrue(np.all(result >= 0))


class TestMHC(unittest.TestCase):
    """测试 mHC 模块。"""

    def test_output_shape(self):
        mhc = ManifoldConstrainedHyperConnection(
            hidden_size=64, n_hc=4, sinkhorn_iters=10
        )
        X = Tensor(np.random.randn(2, 4, 64).astype(np.float32), mstype.float32)
        F_out = Tensor(np.random.randn(2, 64).astype(np.float32), mstype.float32)
        out = mhc(X, F_out)
        self.assertEqual(out.shape, (2, 4, 64))

    def test_get_layer_input(self):
        mhc = ManifoldConstrainedHyperConnection(
            hidden_size=64, n_hc=4, sinkhorn_iters=10
        )
        X = Tensor(np.random.randn(2, 4, 64).astype(np.float32), mstype.float32)
        layer_input = mhc.get_layer_input(X)
        self.assertEqual(layer_input.shape, (2, 64))


class TestKVCompressor(unittest.TestCase):
    """测试 KV 压缩器。"""

    def test_single_branch(self):
        comp = KVCompressor(
            hidden_size=64, head_dim=32, compress_rate=4,
            dual_branch=False,
        )
        H = Tensor(np.random.randn(2, 16, 64).astype(np.float32), mstype.float32)
        out = comp(H)
        self.assertEqual(out.shape, (2, 4, 32))  # 16/4 = 4

    def test_dual_branch(self):
        comp = KVCompressor(
            hidden_size=64, head_dim=32, compress_rate=4,
            dual_branch=True,
        )
        H = Tensor(np.random.randn(2, 16, 64).astype(np.float32), mstype.float32)
        out = comp(H)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[2], 32)


class TestSwiGLUExpert(unittest.TestCase):
    """测试 SwiGLU Expert。"""

    def test_output_shape(self):
        expert = SwiGLUExpert(
            hidden_size=64, intermediate_dim=128,
            clamp_min=-10.0, clamp_max=10.0, gate_clamp_max=10.0,
        )
        x = Tensor(np.random.randn(4, 64).astype(np.float32), mstype.float32)
        out = expert(x)
        self.assertEqual(out.shape, (4, 64))


class TestMoE(unittest.TestCase):
    """测试 DeepSeekMoE。"""

    def test_output_shape(self):
        config = DeepSeekV4Config(
            hidden_size=64,
            num_shared_experts=1,
            num_routed_experts=8,
            num_activated_experts=2,
            expert_intermediate_dim=128,
            num_hash_routing_layers=3,
        )
        moe = DeepSeekMoE(config, layer_idx=0)
        x = Tensor(np.random.randn(2, 10, 64).astype(np.float32), mstype.float32)
        token_ids = Tensor(
            np.random.randint(0, 1000, (2, 10)).astype(np.int32), mstype.int32
        )
        out, balance_loss = moe(x, token_ids)
        self.assertEqual(out.shape, (2, 10, 64))
        self.assertEqual(balance_loss.shape, ())


class TestDeepSeekV4Model(unittest.TestCase):
    """测试完整模型 (使用极小配置)。"""

    def _make_tiny_config(self):
        """创建一个极小的配置用于测试。"""
        attn_schedule = ["swa", "csa", "hca"]
        return DeepSeekV4Config(
            vocab_size=100,
            num_layers=3,
            hidden_size=64,
            max_seq_length=128,
            num_query_heads=4,
            head_dim=32,
            query_compress_dim=32,
            rope_dim=16,
            num_output_groups=2,
            group_output_dim=32,
            csa_compress_rate=4,
            csa_indexer_num_heads=4,
            csa_indexer_head_dim=16,
            csa_top_k=4,
            hca_compress_rate=4,
            sliding_window_size=8,
            num_shared_experts=1,
            num_routed_experts=4,
            num_activated_experts=2,
            expert_intermediate_dim=64,
            num_hash_routing_layers=2,
            mhc_expansion_factor=2,
            mhc_sinkhorn_iters=5,
            mtp_depth=1,
            layer_attention_types=attn_schedule,
        )

    def test_forward(self):
        config = self._make_tiny_config()
        model = DeepSeekV4Model(config)

        batch, seq_len = 2, 16
        input_ids = Tensor(
            np.random.randint(0, 100, (batch, seq_len)).astype(np.int32),
            mstype.int32,
        )
        labels = Tensor(
            np.random.randint(0, 100, (batch, seq_len)).astype(np.int32),
            mstype.int32,
        )

        lm_logits, mtp_logits, balance_loss = model(input_ids, labels=labels)

        self.assertEqual(lm_logits.shape, (batch, seq_len, 100))
        self.assertEqual(mtp_logits.shape, (batch, seq_len - 1, 100))
        self.assertEqual(balance_loss.shape, ())

    def test_compute_loss(self):
        config = self._make_tiny_config()
        model = DeepSeekV4Model(config)

        batch, seq_len = 2, 16
        input_ids = Tensor(
            np.random.randint(0, 100, (batch, seq_len)).astype(np.int32),
            mstype.int32,
        )
        labels = Tensor(
            np.random.randint(0, 100, (batch, seq_len)).astype(np.int32),
            mstype.int32,
        )

        loss = model.compute_loss(input_ids, labels)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.asnumpy() > 0)


if __name__ == "__main__":
    unittest.main()
