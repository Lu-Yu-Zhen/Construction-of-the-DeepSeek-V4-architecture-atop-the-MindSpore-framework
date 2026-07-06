# DeepSeek-V4

English | [中文](README_zh.md)

MindSpore implementation of **DeepSeek-V4** based on the paper
*"DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"*.

This repository provides a clean, modular implementation of the complete DeepSeek-V4
architecture, supporting both **Flash** (284B total / 13B activated) and
**Pro** (1.6T total / 49B activated) configurations.

## About MindSpore

[MindSpore](https://www.mindspore.cn/) is an open-source deep learning framework developed by [Huawei](https://www.huawei.com/), designed for both cloud and edge AI scenarios. It provides native support for Huawei's Ascend AI processors while also running on CPU and GPU.

Key features of MindSpore include:

- **Automatic Differentiation** — A functional automatic differentiation framework that supports both static and dynamic graphs, enabling flexible model development.
- **Ascend NPU Native Support** — Optimized for Huawei Ascend series (Ascend 910, 310, etc.) with hardware-software co-design for maximum throughput.
- **Unified Cloud-Edge Framework** — Same model code can be deployed across cloud data centers, edge devices, and mobile terminals without modification.
- **Graph-Mode Performance** — The `GRAPH_MODE` compiles the entire network into a static graph for optimized execution, while `PYNATIVE_MODE` offers dynamic debugging for development.
- **Rich Ecosystem** — Includes MindFormers for large model training, MindX for industrial applications, and MindInsight for model debugging and profiling.

This implementation uses MindSpore's `nn.Cell` as the base class for all model components (equivalent to PyTorch's `nn.Module`), with `construct()` replacing `forward()` as the computation entry point.

## About DeepSeek

[DeepSeek](https://www.deepseek.com/) is an AI research lab that has released a series of influential open-source large language models, known for their strong performance and innovative architectures. The DeepSeek model family includes:

- **DeepSeek-V1** — An early MoE-based language model demonstrating the feasibility of sparse expert architectures at scale.
- **DeepSeek-V2** — Introduced Multi-head Latent Attention (MLA) and DeepSeekMoE with fine-grained experts, significantly reducing inference costs.
- **DeepSeek-V3** — A 671B-parameter MoE model with 37B activated parameters per token, featuring Multi-Token Prediction (MTP) and auxiliary-loss-free load balancing. Trained on 14.8T tokens with remarkable efficiency.
- **DeepSeek-R1** — A reasoning-focused model trained via reinforcement learning, achieving performance comparable to OpenAI o1 on math and coding benchmarks.
- **DeepSeek-V4** — The latest iteration, introducing mHC residual connections, CSA/HCA hybrid attention, and Muon optimizer for million-token context intelligence.

DeepSeek's open-source philosophy has made it one of the most influential players in the open LLM ecosystem, with models widely adopted by researchers and developers worldwide.

## Architecture Overview

DeepSeek-V4 introduces several novel components for efficient million-token context modeling:

- **mHC (Manifold-Constrained Hyper-Connections)** — Replaces standard residual connections with a doubly stochastic matrix constraint via Sinkhorn-Knopp iteration. The residual state is expanded `n_hc` times and updated as `X_{l+1} = B_l @ X_l + C_l * F_l(A_l @ X_l)`.

- **CSA (Compressed Sparse Attention)** — Dual-branch overlapping KV compression (rate m=4) with a Lightning Indexer for top-k sparse selection, combined with multi-query attention (MQA) and a sliding window branch.

- **HCA (Heavily Compressed Attention)** — Single-branch heavy KV compression (rate m'=128) with MQA and sliding window, designed for very long contexts.

- **DeepSeekMoE** — Mixture-of-Experts with shared + routed experts, Sqrt(Softplus) routing scores, Hash routing for early layers, and auxiliary-loss-free load balancing.

- **MTP (Multi-Token Prediction)** — Shifted embedding concatenation with an extra Transformer block, sharing the LM head for improved training signal density.

- **Muon Optimizer** — Hybrid Newton-Schulz orthogonalization (8+2 steps) with Nesterov momentum and RMS rescaling for weight matrix updates.

## Project Structure

```
DeepSeek-V4/
├── src/                        # Core model implementation
│   ├── __init__.py
│   ├── config.py               # Model configurations (Flash & Pro)
│   ├── normalization.py        # RMSNorm, RotaryPositionalEmbedding
│   ├── mhc.py                  # Sinkhorn-Knopp, mHC residual connections
│   ├── attention.py            # CSA, HCA, SWA, KV Compressor, Lightning Indexer
│   ├── moe.py                  # SwiGLU Expert, DeepSeekMoE
│   ├── mtp.py                  # Multi-Token Prediction
│   ├── model.py                # TransformerBlock, DeepSeekV4Model
│   └── optimizer.py            # Muon Optimizer
├── configs/                    # YAML configuration files
│   ├── flash_config.yaml
│   └── pro_config.yaml
├── scripts/                    # Training and inference scripts
│   ├── train.py
│   └── inference.py
├── tests/                      # Unit tests
│   └── test_model.py
├── docs/                       # Documentation
├── data/                       # Data directory
├── checkpoints/                # Model checkpoints
├── main.py                     # Entry point with demos
├── model.py                    # Original monolithic implementation (reference)
├── requirements.txt
├── setup.py
└── README.md
```

## Requirements

- Python >= 3.9
- MindSpore >= 2.8.0
- NumPy >= 1.20

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/DeepSeek-V4.git
cd DeepSeek-V4

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Usage

### Quick Start

```python
from src.config import flash_config
from src.model import DeepSeekV4Model

# Create Flash configuration
config = flash_config()
model = DeepSeekV4Model(config)

# Forward pass
import mindspore as ms
from mindspore import Tensor, dtype as mstype
import numpy as np

input_ids = Tensor(np.random.randint(0, 128000, (1, 128)).astype(np.int32), mstype.int32)
lm_logits, mtp_logits, balance_loss = model(input_ids)
```

### Training

```bash
# Train Flash model
python scripts/train.py --model flash --epochs 10 --batch-size 4 --lr 2.7e-4

# Train Pro model (requires significant compute)
python scripts/train.py --model pro --epochs 5 --batch-size 2
```

### Inference

```bash
# Generate text
python scripts/inference.py --model flash --prompt "Hello, I am" --max-len 256

# With checkpoint
python scripts/inference.py --model flash --checkpoint ./checkpoints/model.ckpt
```

### Run Tests

```bash
python -m pytest tests/test_model.py -v
```

## Model Configurations

| Parameter | Flash | Pro |
|---|---|---|
| Total Parameters | 284B | 1.6T |
| Activated Parameters | 13B | 49B |
| Layers | 43 | 61 |
| Hidden Size | 4096 | 7168 |
| Query Heads | 64 | 128 |
| Head Dim | 512 | 512 |
| Routed Experts | 256 | 384 |
| Activated Experts | 6 | 6 |
| CSA Compress Rate | 4 | 4 |
| HCA Compress Rate | 128 | 128 |
| CSA Top-k | 512 | 1024 |
| mHC Expansion | 4 | 4 |
| Max Seq Length | 1M | 1M |

## Key Design Decisions

1. **Hybrid Attention**: Layers alternate between CSA (for medium-range dependencies) and HCA (for ultra-long context), with the first 2 layers using SWA (Flash) or HCA (Pro).

2. **mHC Residual Streams**: Instead of simple `x + F(x)`, the residual state is expanded `n_hc=4` times and updated through constrained matrix operations, enabling richer information flow.

3. **KV Compression**: CSA achieves 4x compression via dual-branch overlapping windows; HCA achieves 128x compression for extreme context lengths.

4. **Auxiliary-Loss-Free Balancing**: Load balancing is achieved through a learned bias term added to routing scores, avoiding the need for auxiliary losses that can interfere with model training.

## References

- Paper: *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*
- Framework: [MindSpore](https://www.mindspore.cn/)
- Related: DeepSeek-V3, DeepSeekMoE

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Disclaimer

This is a research implementation based on the published paper. It is intended for academic and educational purposes.
