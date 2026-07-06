"""
DeepSeek-V4 MindSpore 实现 — 入口脚本
======================================

功能:
  1. 实例化 DeepSeek-V4-Flash (284B) 或 DeepSeek-V4-Pro (1.6T) 模型
  2. 打印模型结构与参数统计
  3. 演示前向传播
  4. 演示训练循环 (含 Muon + AdamW 混合优化)

用法:
  python main.py --model flash          # 使用 Flash 配置
  python main.py --model pro            # 使用 Pro 配置
  python main.py --model flash --train  # 启动训练演示
"""

import argparse
import sys
import time
from typing import Dict

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, dtype as mstype
from mindspore import context

# 导入模型定义 (从 src/ 模块化包)
from src import (
    DeepSeekV4Config,
    DeepSeekV4Model,
    MuonOptimizer,
    flash_config,
    pro_config,
)


# ============================================================================
# 环境配置
# ============================================================================

def setup_context(use_ascend: bool = True):
    """
    配置 MindSpore 运行上下文。
    优先使用昇腾 NPU，回退到 GPU，最后回退到 CPU。
    """
    if use_ascend:
        try:
            context.set_context(
                mode=context.GRAPH_MODE,
                device_target="Ascend",
                device_id=0,
            )
            print("[Context] 使用昇腾 Ascend NPU (Graph Mode)")
            return
        except Exception:
            pass

    try:
        context.set_context(
            mode=context.PYNATIVE_MODE,
            device_target="GPU",
            device_id=0,
        )
        print("[Context] 使用 GPU (Pynative Mode)")
    except Exception:
        context.set_context(
            mode=context.PYNATIVE_MODE,
            device_target="CPU",
        )
        print("[Context] 使用 CPU (Pynative Mode)")


# ============================================================================
# 模型参数统计
# ============================================================================

def count_parameters(model: nn.Cell) -> Dict[str, int]:
    """统计模型参数: 总参数数、可训练参数数、各模块参数数。"""
    total = 0
    trainable = 0
    module_counts: Dict[str, int] = {}

    for name, param in model.parameters_and_names():
        numel = param.size
        total += numel
        if param.requires_grad:
            trainable += numel

        # 按模块分组
        parts = name.split(".")
        module_key = parts[0] if parts else "other"
        module_counts[module_key] = module_counts.get(module_key, 0) + numel

    return {
        "total": total,
        "trainable": trainable,
        "modules": module_counts,
    }


def print_model_summary(config: DeepSeekV4Config, model: DeepSeekV4Model):
    """打印模型配置和参数摘要。"""
    print("=" * 72)
    print("  DeepSeek-V4 模型配置摘要")
    print("=" * 72)

    print(f"\n[配置]")
    print(f"  模型类型:        {'Flash (284B)' if config.num_layers == 43 else 'Pro (1.6T)'}")
    print(f"  Transformer 层数: {config.num_layers}")
    print(f"  隐层维度 d:      {config.hidden_size}")
    print(f"  注意力头数 n_h:   {config.num_query_heads}")
    print(f"  头维度 c:         {config.head_dim}")
    print(f"  查询压缩维度 d_c: {config.query_compress_dim}")

    print(f"\n[注意力架构]")
    # 统计各注意力类型层数
    attn_counts = {}
    for at in config.layer_attention_types:
        attn_counts[at] = attn_counts.get(at, 0) + 1
    for at, cnt in attn_counts.items():
        type_name = {"swa": "滑动窗口 (SWA)", "csa": "压缩稀疏 (CSA)",
                     "hca": "重压缩 (HCA)"}[at]
        print(f"  {type_name}:  {cnt} 层")

    print(f"  CSA 压缩比 m:      {config.csa_compress_rate}")
    print(f"  CSA top-k:         {config.csa_top_k}")
    print(f"  HCA 压缩比 m':     {config.hca_compress_rate}")
    print(f"  滑动窗口 n_win:    {config.sliding_window_size}")
    print(f"  输出分组数 g:      {config.num_output_groups}")

    print(f"\n[MoE]")
    print(f"  共享专家:          {config.num_shared_experts}")
    print(f"  路由专家:          {config.num_routed_experts}")
    print(f"  激活专家:          {config.num_activated_experts}")
    print(f"  专家中间维度:      {config.expert_intermediate_dim}")
    print(f"  Hash 路由层数:     {config.num_hash_routing_layers}")
    print(f"  激活函数:          Sqrt(Softplus(·)) (替代 V3 的 Sigmoid)")

    print(f"\n[mHC]")
    print(f"  扩展因子 n_hc:     {config.mhc_expansion_factor}")
    print(f"  Sinkhorn 迭代:     {config.mhc_sinkhorn_iters}")
    print(f"  约束流形:          双随机矩阵 (Birkhoff polytope)")

    print(f"\n[MTP]")
    print(f"  预测深度:          {config.mtp_depth}")
    print(f"  损失权重:          {config.mtp_loss_weight}")

    # 参数统计
    stats = count_parameters(model)
    print(f"\n[参数统计]")
    print(f"  总参数量:          {stats['total']:,}")
    print(f"  可训练参数量:      {stats['trainable']:,}")

    # 估算激活参数量 (每 token)
    activated_per_token = (
        config.hidden_size  # embedding
        + config.num_layers * (
            config.num_shared_experts * config.expert_intermediate_dim * 3  # 共享专家
            + config.num_activated_experts * config.expert_intermediate_dim * 3  # 路由专家
        )
        + config.hidden_size * config.vocab_size  # lm_head
    )
    print(f"  每 token 激活参数: ~{activated_per_token:,} (估算)")

    print(f"\n[模块参数分布]")
    for mod, cnt in sorted(stats["modules"].items(), key=lambda x: -x[1]):
        pct = cnt / stats["total"] * 100
        print(f"  {mod:<30s} {cnt:>12,}  ({pct:.1f}%)")

    print("=" * 72)


# ============================================================================
# 前向传播演示
# ============================================================================

def demo_forward(model: DeepSeekV4Model, config: DeepSeekV4Config):
    """演示模型前向传播。"""
    print("\n[前向传播演示]")

    batch_size = 2
    seq_len = 64  # 短序列用于演示

    # 随机生成输入
    input_ids = Tensor(
        np.random.randint(0, config.vocab_size, (batch_size, seq_len)),
        mstype.int32,
    )
    labels = Tensor(
        np.random.randint(0, config.vocab_size, (batch_size, seq_len)),
        mstype.int32,
    )

    print(f"  输入形状:  input_ids = {input_ids.shape}")
    print(f"  开始推理...")

    t0 = time.time()
    lm_logits, mtp_logits, balance_loss = model(input_ids, labels=labels)
    elapsed = time.time() - t0

    print(f"  输出形状:  lm_logits = {lm_logits.shape}")
    print(f"             mtp_logits = {mtp_logits.shape if mtp_logits is not None else 'None'}")
    print(f"  MoE 均衡损失: {balance_loss.asnumpy():.6f}")
    print(f"  推理耗时:  {elapsed:.2f}s")
    print(f"  前向传播成功!")


# ============================================================================
# 训练循环演示
# ============================================================================

def demo_training(model: DeepSeekV4Model, config: DeepSeekV4Config):
    """
    演示训练循环。

    混合优化策略 (论文 Section 2.4):
      - Muon: 大部分参数 (Transformer blocks)
      - AdamW: Embedding, LM Head, RMSNorm, mHC 静态偏置/门控
    """
    print("\n[训练循环演示]")
    print("  (使用缩小序列长度进行演示)")

    # 定义损失函数
    class DeepSeekV4Loss(nn.Cell):
        def __init__(self, model: DeepSeekV4Model):
            super().__init__()
            self.model = model

        def construct(self, input_ids: Tensor, labels: Tensor):
            return self.model.compute_loss(input_ids, labels)

    loss_fn = DeepSeekV4Loss(model)

    # 参数分组: Muon vs AdamW
    muon_params = []
    adamw_params = []
    for name, param in model.parameters_and_names():
        # Embedding, LM Head, RMSNorm, mHC 静态参数用 AdamW
        if any(key in name for key in [
            "embedding", "lm_head", "rmsnorm", "mhc_S_", "mhc_alpha_"
        ]):
            adamw_params.append(param)
        else:
            muon_params.append(param)

    print(f"  Muon 参数数:  {len(muon_params)}")
    print(f"  AdamW 参数数: {len(adamw_params)}")

    # 优化器
    # 注: 实际部署时分别创建两个优化器
    # 这里简化使用 AdamW 做演示
    optimizer = nn.Adam(
        model.trainable_params(),
        learning_rate=2.7e-4,
        beta1=0.9,
        beta2=0.95,
        eps=1e-20,
        weight_decay=0.1,
    )

    # 训练网络
    train_net = nn.TrainOneStepCell(loss_fn, optimizer)
    train_net.set_train(True)

    # 训练参数 (论文 Section 4.2.2)
    batch_size = 1
    seq_len = 32  # 演示用短序列
    num_steps = 5

    print(f"\n  训练超参:")
    print(f"    batch_size = {batch_size}")
    print(f"    seq_len = {seq_len} (演示)")
    print(f"    学习率 = 2.7e-4 (峰值)")
    print(f"    优化器 = AdamW (演示; 实际为 Muon + AdamW 混合)")
    print()

    for step in range(num_steps):
        input_ids = Tensor(
            np.random.randint(0, config.vocab_size, (batch_size, seq_len)),
            mstype.int32,
        )
        labels = Tensor(
            np.random.randint(0, config.vocab_size, (batch_size, seq_len)),
            mstype.int32,
        )

        loss = train_net(input_ids, labels)
        print(f"  Step {step + 1}/{num_steps}  |  loss = {loss.asnumpy():.4f}")

    print(f"\n  训练演示完成!")


# ============================================================================
# 推理效率分析
# ============================================================================

def analyze_efficiency(config: DeepSeekV4Config):
    """
    分析 DeepSeek-V4 在长上下文下的推理效率 (论文 Section 2.3.4)。

    对比 V3.2 (BF16 GQA 8, head_dim=128) 基线:
      - KV Cache 大小
      - 单 token FLOPs
    """
    print("\n[推理效率分析]")
    print("=" * 56)

    seq_lengths = [4096, 16384, 65536, 262144, 1_048_576]

    d = config.hidden_size
    n_h = config.num_query_heads
    c = config.head_dim
    d_c = config.query_compress_dim
    m = config.csa_compress_rate
    m_prime = config.hca_compress_rate
    top_k = config.csa_top_k
    n_win = config.sliding_window_size
    L = config.num_layers

    # CSA 层数和 HCA 层数
    csa_layers = sum(1 for t in config.layer_attention_types if t == "csa")
    hca_layers = sum(1 for t in config.layer_attention_types if t == "hca")

    print(f"\n  模型: {'Flash' if config.num_layers == 43 else 'Pro'}")
    print(f"  CSA 层数: {csa_layers}, HCA 层数: {hca_layers}")
    print(f"  压缩比: CSA m={m}, HCA m'={m_prime}")
    print(f"\n  {'序列长度':>12s} | {'CSA KV/token':>12s} | {'HCA KV/token':>12s} | {'总 KV/token':>12s}")
    print(f"  {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12}")

    for seq_len in seq_lengths:
        # CSA: 每 m token 压缩为 1, 存储 top_k 个压缩条目 + n_win 滑动窗口
        csa_entries_per_layer = (seq_len // m) + n_win
        # HCA: 每 m' token 压缩为 1, 存储所有压缩条目 + n_win
        hca_entries_per_layer = (seq_len // m_prime) + n_win

        total_csa = csa_entries_per_layer * csa_layers * c  # 每个条目 c 维
        total_hca = hca_entries_per_layer * hca_layers * c

        total_kv = total_csa + total_hca  # bytes (FP32, 4 bytes each)
        total_kv_mb = total_kv * 4 / (1024 * 1024)  # MB

        seq_str = f"{seq_len:>12,}"
        csa_str = f"{total_csa:>12,}"
        hca_str = f"{total_hca:>12,}"
        kv_str = f"{total_kv_mb:>10.1f} MB"

        print(f"  {seq_str} | {csa_str} | {hca_str} | {kv_str}")

    print(f"\n  论文报告 (1M token 上下文):")
    print(f"    DeepSeek-V4-Pro:   单 token FLOPs 为 V3.2 的 27%, KV Cache 为 10%")
    print(f"    DeepSeek-V4-Flash: 单 token FLOPs 为 V3.2 的 10%, KV Cache 为 7%")
    print("=" * 56)


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 MindSpore 实现")
    parser.add_argument(
        "--model",
        type=str,
        default="flash",
        choices=["flash", "pro"],
        help="模型配置: flash (284B) 或 pro (1.6T)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="是否运行训练演示",
    )
    parser.add_argument(
        "--no-ascend",
        action="store_true",
        help="禁用昇腾 NPU (回退到 GPU/CPU)",
    )
    args = parser.parse_args()

    # 设置上下文
    setup_context(use_ascend=not args.no_ascend)

    # 选择配置
    if args.model == "flash":
        config = flash_config()
        print("\n>>> 加载 DeepSeek-V4-Flash 配置 (284B total, 13B activated)")
    else:
        config = pro_config()
        print("\n>>> 加载 DeepSeek-V4-Pro 配置 (1.6T total, 49B activated)")

    # 实例化模型
    print(">>> 构建模型...")
    model = DeepSeekV4Model(config)
    print(">>> 模型构建完成!")

    # 打印摘要
    print_model_summary(config, model)

    # 效率分析
    analyze_efficiency(config)

    # 前向传播演示
    demo_forward(model, config)

    # 训练演示 (可选)
    if args.train:
        demo_training(model, config)

    print("\n>>> 全部完成! DeepSeek-V4 架构已成功在 MindSpore 上实现。")


if __name__ == "__main__":
    main()
