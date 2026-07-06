"""
推理脚本
=========
使用 MindSpore 加载 DeepSeek-V4 模型进行文本生成。

用法:
  python scripts/inference.py --model flash --checkpoint ./checkpoints/model.ckpt
  python scripts/inference.py --model pro --prompt "Hello" --max-len 256
"""

import argparse
import sys
import os

import numpy as np

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mindspore as ms
from mindspore import Tensor, dtype as mstype
import mindspore.ops as ops

from src.config import DeepSeekV4Config, flash_config, pro_config
from src.model import DeepSeekV4Model


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 Inference")
    parser.add_argument(
        "--model", type=str, default="flash",
        choices=["flash", "pro"],
        help="模型配置: flash (284B) 或 pro (1.6T)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="模型检查点路径 (.ckpt 文件)",
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello, I am",
        help="输入 prompt",
    )
    parser.add_argument(
        "--max-len", type=int, default=128,
        help="最大生成长度",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="采样温度",
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k 采样",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Top-p (nucleus) 采样",
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


class SimpleTokenizer:
    """
    简单 Tokenizer 占位实现。
    实际使用时应替换为 SentencePiece / BPE tokenizer。
    """

    def __init__(self, vocab_size: int = 128000):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list:
        """将文本编码为 token ID 列表 (占位实现)。"""
        # 占位: 将每个字符的 ASCII 码作为 token ID
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, token_ids: list) -> str:
        """将 token ID 列表解码为文本 (占位实现)。"""
        # 占位: 将 token ID 映射回字符
        return "".join(chr(tid % 128) for tid in token_ids)


def top_k_top_p_sampling(logits: np.ndarray, temperature: float,
                         top_k: int, top_p: float) -> int:
    """
    Top-k + Top-p 采样。

    logits: (vocab_size,) 原始 logits
    返回: 采样的 token ID
    """
    # 温度缩放
    logits = logits / max(temperature, 1e-8)

    # Top-k 过滤
    if top_k > 0:
        indices_to_remove = logits < np.sort(logits)[-top_k]
        logits[indices_to_remove] = float("-inf")

    # Top-p (nucleus) 过滤
    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        cumulative_probs = np.cumsum(
            np.exp(sorted_logits) / np.sum(np.exp(sorted_logits))
        )
        # 移除累积概率超过 top_p 的 token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    # 转为概率分布并采样
    probs = np.exp(logits) / np.sum(np.exp(logits))
    next_token = np.random.choice(len(probs), p=probs)
    return int(next_token)


def generate(model: DeepSeekV4Model, tokenizer: SimpleTokenizer,
             prompt: str, max_len: int, temperature: float,
             top_k: int, top_p: float) -> str:
    """
    自回归文本生成。

    model: 已加载的 DeepSeekV4Model
    tokenizer: tokenizer 实例
    prompt: 输入文本
    max_len: 最大生成长度
    temperature: 采样温度
    top_k: Top-k 采样
    top_p: Top-p 采样
    """
    # 编码 prompt
    token_ids = tokenizer.encode(prompt)
    generated = list(token_ids)

    print(f"\n[Prompt] {prompt}")
    print("[Generating]", end=" ", flush=True)

    for step in range(max_len):
        # 构建输入
        seq_len = len(generated)
        input_ids = Tensor(
            np.array([generated], dtype=np.int32), mstype.int32
        )

        # 前向传播
        lm_logits, _, _ = model(input_ids)

        # 取最后一个 token 的 logits
        next_logits = lm_logits[0, -1, :].asnumpy()

        # 采样
        next_token = top_k_top_p_sampling(
            next_logits, temperature, top_k, top_p
        )
        generated.append(next_token)

        # 打印生成的 token
        decoded = tokenizer.decode([next_token])
        print(decoded, end="", flush=True)

        # 终止条件 (简化: 遇到句号或达到最大长度)
        if decoded in (".", "。", "\n") and step > 10:
            break

    print("\n")
    return tokenizer.decode(generated)


def main():
    args = parse_args()

    # 初始化上下文
    setup_context(use_ascend=not args.no_ascend)

    # 加载配置
    if args.model == "flash":
        config = flash_config()
    else:
        config = pro_config()

    print(f"\n{'='*60}")
    print(f"  DeepSeek-V4-{args.model.upper()} Inference")
    print(f"{'='*60}")
    print(f"  Layers:      {config.num_layers}")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Max Length:  {args.max_len}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k:       {args.top_k}")
    print(f"  Top-p:       {args.top_p}")
    print(f"{'='*60}\n")

    # 创建模型
    print("[INFO] Building model...")
    model = DeepSeekV4Model(config)
    model.set_train(False)

    # 加载检查点
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        param_dict = ms.load_checkpoint(args.checkpoint)
        ms.load_param_into_net(model, param_dict)
        print("[INFO] Checkpoint loaded successfully.")
    else:
        print("[WARN] No checkpoint provided, using random weights.")

    # 创建 Tokenizer
    tokenizer = SimpleTokenizer(config.vocab_size)

    # 生成文本
    result = generate(
        model, tokenizer,
        prompt=args.prompt,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    print(f"\n[Output]\n{result}")


if __name__ == "__main__":
    main()
