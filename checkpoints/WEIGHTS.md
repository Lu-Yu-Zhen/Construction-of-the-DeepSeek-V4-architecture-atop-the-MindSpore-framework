# Model Weights

This directory is for storing downloaded model checkpoint files.

The official **DeepSeek-V4** pretrained weights can be downloaded from Hugging Face or ModelScope.

## Hugging Face

### Collection Page

DeepSeek-V4 full model collection on Hugging Face:

[https://huggingface.co/collections/deepseek-ai/deepseek-v4](https://huggingface.co/collections/deepseek-ai/deepseek-v4) [$TRAE_REF](https://huggingface.co/collections/deepseek-ai/deepseek-v4)

### Available Models

| Model | Size | Type | Hugging Face Link |
|-------|------|------|------------------|
| **DeepSeek-V4-Flash-Base** | 292B | Base (pretrained) | [deepseek-ai/DeepSeek-V4-Flash-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base) |
| **DeepSeek-V4-Flash** | 158B | Chat (instruction-tuned) | [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) |
| **DeepSeek-V4-Flash-DSpark** | 165B | Chat (DSpark accelerated) | [deepseek-ai/DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) |
| **DeepSeek-V4-Pro-Base** | 1.6T | Base (pretrained) | [deepseek-ai/DeepSeek-V4-Pro-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base) |
| **DeepSeek-V4-Pro** | 862B | Chat (instruction-tuned) | [deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) |
| **DeepSeek-V4-Pro-DSpark** | 889B | Chat (DSpark accelerated) | [deepseek-ai/DeepSeek-V4-Pro-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark) |

> **Note:** DSpark variants use the same model weights with accelerated inference kernels.

## ModelScope (China - Faster Downloads)

For users in China, ModelScope provides faster download speeds:

[https://modelscope.cn/collections/deepseek-ai/DeepSeek-V4](https://modelscope.cn/collections/deepseek-ai/DeepSeek-V4) [$TRAE_REF](https://modelscope.cn/collections/deepseek-ai/DeepSeek-V4)

| Model | ModelScope Link |
|-------|----------------|
| DeepSeek-V4-Flash | [deepseek-ai/DeepSeek-V4-Flash](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash) |
| DeepSeek-V4-Pro | [deepseek-ai/DeepSeek-V4-Pro](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro) |

## Download Instructions

### Using Hugging Face CLI

```bash
# Install huggingface-cli
pip install huggingface_hub

# Download DeepSeek-V4-Flash (recommended for most users)
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir ./checkpoints/DeepSeek-V4-Flash

# Download DeepSeek-V4-Pro
huggingface-cli download deepseek-ai/DeepSeek-V4-Pro --local-dir ./checkpoints/DeepSeek-V4-Pro
```

### Using ModelScope CLI (China, Faster)

```bash
# Install modelscope
pip install modelscope

# Download DeepSeek-V4-Flash
modelscope download --model deepseek-ai/DeepSeek-V4-Flash --local-dir ./checkpoints/DeepSeek-V4-Flash

# Download DeepSeek-V4-Pro
modelscope download --model deepseek-ai/DeepSeek-V4-Pro --local-dir ./checkpoints/DeepSeek-V4-Pro
```

### Using Python

```python
# Hugging Face
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    local_dir="./checkpoints/DeepSeek-V4-Flash"
)

# ModelScope
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    model_id="deepseek-ai/DeepSeek-V4-Flash",
    cache_dir="./checkpoints/DeepSeek-V4-Flash"
)
```

## File Structure After Download

After downloading, the expected directory structure:

```
checkpoints/
├── WEIGHTS.md                       # This file
├── DeepSeek-V4-Flash/               # Flash chat model
│   ├── config.json
│   ├── tokenizer.json
│   ├── model-00001-of-000NN.safetensors
│   └── ...
├── DeepSeek-V4-Pro/                 # Pro chat model (if downloaded)
│   ├── config.json
│   ├── tokenizer.json
│   ├── model-00001-of-000NN.safetensors
│   └── ...
└── model.ckpt                       # Converted MindSpore checkpoint (after conversion)
```

## Weight Conversion (MindSpore)

The original PyTorch weights from Hugging Face need to be converted to MindSpore format before use:

```python
import mindspore as ms
from safetensors import safe_open

# Example: convert safetensors to MindSpore checkpoint
# (Actual conversion script may vary based on model structure)
ms_parameters = []
with safe_open("./checkpoints/DeepSeek-V4-Flash/model-00001-of-000NN.safetensors", framework="np") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        ms_param = ms.Parameter(ms.Tensor(tensor), name=key)
        ms_parameters.append(ms_param)

ms.save_checkpoint(ms_parameters, "./checkpoints/model.ckpt")
print("Checkpoint converted successfully!")
```

## Hardware Requirements

| Model | Storage | Recommended GPU |
|-------|---------|-----------------|
| DeepSeek-V4-Flash | ~300GB | 24GB+ (RTX 4090/5090) with FP8 |
| DeepSeek-V4-Pro | ~2TB | 80GB+ (H100/A100) |