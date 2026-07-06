# Installation Guide

This guide covers the installation of the DeepSeek-V4 MindSpore implementation.

## Prerequisites

- **Python >= 3.9**
- **MindSpore >= 2.8.0**
- **CUDA-compatible GPU** (recommended) or Ascend NPU

## Step 1: Install MindSpore

### Option A: CUDA (NVIDIA GPU)

```bash
# CUDA 12.1
pip install mindspore==2.8.0

# Verify installation
python -c "import mindspore; print(mindspore.__version__)"
```

### Option B: Ascend NPU

```bash
# Install MindSpore for Ascend
pip install mindspore-ascend==2.8.0
```

### Option C: CPU Only

```bash
pip install mindspore==2.8.0
```

> **Note:** CPU-only mode is suitable for development and testing but not for training or inference at scale.

## Step 2: Install the Package

### From Source

```bash
# Clone the repository
git clone https://github.com/Lu-Yu-Zhen/Construction-of-the-DeepSeek-V4-architecture-atop-the-MindSpore-framework.git
cd Construction-of-the-DeepSeek-V4-architecture-atop-the-MindSpore-framework

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Using pip (future)

```bash
pip install deepseek-v4-mindspore
```

## Step 3: Verify Installation

```python
import mindspore as ms
from src.config import flash_config
from src.model import DeepSeekV4Model

# Create a small config for testing
config = flash_config()
model = DeepSeekV4Model(config)
print(f"Model created successfully: {type(model).__name__}")
print(f"Total parameters: {sum(p.numel() for p in model.trainable_params()) / 1e9:.2f}B")
```

## Hardware Requirements

### Minimum Requirements (Inference Only)

| Component | Flash (284B) | Pro (1.6T) |
|-----------|-------------|------------|
| GPU Memory | 24GB+ (FP8) | 80GB+ (FP8) |
| Recommended GPU | RTX 4090 / 5090 | H100 / A100 |
| System RAM | 64GB+ | 256GB+ |
| Storage | 300GB+ | 2TB+ |

### Training Requirements

| Component | Flash (284B) | Pro (1.6T) |
|-----------|-------------|------------|
| GPU Memory | 80GB+ per device | 80GB+ per device |
| GPUs | 8+ (H100/A100) | 64+ (H100/A100) |
| System RAM | 512GB+ | 2TB+ |
| Interconnect | NVLink / InfiniBand | InfiniBand |

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'mindspore'` | Install MindSpore first (see Step 1) |
| Out of Memory | Reduce batch size, use FP8, or enable gradient checkpointing |
| `ASCEND_HOME_PATH` not set | Source the Ascend environment: `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| CUDA version mismatch | Check compatibility: `python -c "import mindspore; print(mindspore.run_check())"` |

### Getting Help

- [MindSpore Documentation](https://www.mindspore.cn/docs)
- [MindSpore GitHub Issues](https://github.com/mindspore-ai/mindspore/issues)
- [DeepSeek-V4 Paper](https://arxiv.org/abs/2504.09286)