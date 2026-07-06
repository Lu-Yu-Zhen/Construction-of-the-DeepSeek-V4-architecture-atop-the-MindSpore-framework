"""
训练脚本
=========
使用 MindSpore 训练 DeepSeek-V4 模型。

用法:
  python scripts/train.py --model flash --epochs 10 --batch-size 4
  python scripts/train.py --model pro --epochs 5 --batch-size 2 --no-ascend
"""

import argparse
import sys
import os
import time

import numpy as np

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mindspore as ms
from mindspore import Tensor, dtype as mstype
import mindspore.dataset as ds

from src.config import DeepSeekV4Config, flash_config, pro_config
from src.model import DeepSeekV4Model
from src.optimizer import MuonOptimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepSeek-V4")
    parser.add_argument(
        "--model", type=str, default="flash",
        choices=["flash", "pro"],
        help="模型配置: flash (284B) 或 pro (1.6T)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument("--seq-len", type=int, default=2048, help="序列长度")
    parser.add_argument("--lr", type=float, default=2.7e-4, help="学习率")
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="训练数据路径 (numpy .npy 文件)",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./checkpoints",
        help="模型保存目录",
    )
    parser.add_argument(
        "--save-interval", type=int, default=1000,
        help="每隔多少步保存一次",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10,
        help="每隔多少步打印一次日志",
    )
    parser.add_argument(
        "--no-ascend", action="store_true",
        help="不使用 Ascend NPU (使用 CPU/GPU)",
    )
    return parser.parse_args()


def setup_context(use_ascend: bool = True):
    """初始化 MindSpore 运行上下文。"""
    if use_ascend:
        try:
            ms.set_context(
                mode=ms.GRAPH_MODE,
                device_target="Ascend",
                device_id=int(os.environ.get("DEVICE_ID", 0)),
            )
            print("[INFO] Using Ascend NPU (GRAPH_MODE)")
        except Exception:
            ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
            print("[WARN] Ascend not available, falling back to CPU (PYNATIVE_MODE)")
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        print("[INFO] Using CPU (PYNATIVE_MODE)")


def create_dummy_dataset(batch_size: int, seq_len: int, vocab_size: int):
    """创建随机 dummy 数据集 (用于测试流程)。"""
    num_samples = 100
    input_ids = np.random.randint(0, vocab_size, (num_samples, seq_len)).astype(np.int32)
    labels = np.random.randint(0, vocab_size, (num_samples, seq_len)).astype(np.int32)

    dataset = ds.NumpySlicesDataset(
        {"input_ids": input_ids, "labels": labels},
        column_names=["input_ids", "labels"],
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def count_parameters(model: DeepSeekV4Model):
    """统计模型参数量。"""
    total = 0
    trainable = 0
    for param in model.get_parameters():
        param_size = np.prod(param.shape)
        total += param_size
        if param.requires_grad:
            trainable += param_size
    return total, trainable


def train(args):
    """主训练循环。"""
    # 初始化上下文
    setup_context(use_ascend=not args.no_ascend)

    # 加载配置
    if args.model == "flash":
        config = flash_config()
    else:
        config = pro_config()

    print(f"\n{'='*60}")
    print(f"  DeepSeek-V4-{args.model.upper()} Training")
    print(f"{'='*60}")
    print(f"  Layers:        {config.num_layers}")
    print(f"  Hidden Size:   {config.hidden_size}")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"  Seq Length:    {args.seq_len}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Epochs:        {args.epochs}")
    print(f"{'='*60}\n")

    # 创建模型
    print("[INFO] Building model...")
    model = DeepSeekV4Model(config)

    total_params, trainable_params = count_parameters(model)
    print(f"[INFO] Total parameters:     {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")

    # 创建优化器
    optimizer = MuonOptimizer(
        model.trainable_params(),
        learning_rate=args.lr,
    )

    # 定义前向网络 (含损失)
    class TrainNet(ms.nn.Cell):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def construct(self, input_ids, labels):
            loss = self.backbone.compute_loss(input_ids, labels)
            return loss

    train_net = TrainNet(model)
    train_net.set_train()

    # 创建数据集
    if args.data_path and os.path.exists(args.data_path):
        print(f"[INFO] Loading data from {args.data_path}")
        data = np.load(args.data_path)
        # TODO: 实现真实数据加载逻辑
    else:
        print("[INFO] Using dummy dataset for testing...")
        dataset = create_dummy_dataset(
            args.batch_size, args.seq_len, config.vocab_size
        )

    # 保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练循环
    global_step = 0
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataset.create_dict_iterator():
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            # 前向 + 反向 (简化: 仅打印 loss)
            loss = train_net(input_ids, labels)

            epoch_loss += loss.asnumpy()
            num_batches += 1
            global_step += 1

            if global_step % args.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                print(
                    f"[Epoch {epoch+1}/{args.epochs}] "
                    f"Step {global_step} | "
                    f"Loss: {avg_loss:.4f}"
                )

            if global_step % args.save_interval == 0:
                save_path = os.path.join(
                    args.save_dir, f"checkpoint_step{global_step}.ckpt"
                )
                ms.save_checkpoint(model, save_path)
                print(f"[INFO] Checkpoint saved: {save_path}")

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)
        print(
            f"\n[Epoch {epoch+1}/{args.epochs}] "
            f"completed in {epoch_time:.1f}s | "
            f"Avg Loss: {avg_loss:.4f}\n"
        )

    # 最终保存
    final_path = os.path.join(args.save_dir, f"deepseek_v4_{args.model}_final.ckpt")
    ms.save_checkpoint(model, final_path)
    print(f"[INFO] Final model saved: {final_path}")
    print("[INFO] Training complete!")


if __name__ == "__main__":
    args = parse_args()
    train(args)
