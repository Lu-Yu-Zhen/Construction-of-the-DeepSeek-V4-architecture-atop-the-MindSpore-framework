# 模型权重

本目录用于存放下载的模型检查点文件。

官方 **DeepSeek-V4** 预训练权重可从 Hugging Face 或 ModelScope 下载。

## Hugging Face

### 合集页面

DeepSeek-V4 在 Hugging Face 上的完整模型合集：

[https://huggingface.co/collections/deepseek-ai/deepseek-v4](https://huggingface.co/collections/deepseek-ai/deepseek-v4) [$TRAE_REF](https://huggingface.co/collections/deepseek-ai/deepseek-v4)

### 可用模型

| 模型 | 大小 | 类型 | Hugging Face 链接 |
|-------|------|------|------------------|
| **DeepSeek-V4-Flash-Base** | 292B | 基座（预训练） | [deepseek-ai/DeepSeek-V4-Flash-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base) |
| **DeepSeek-V4-Flash** | 158B | 对话（指令微调） | [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) |
| **DeepSeek-V4-Flash-DSpark** | 165B | 对话（DSpark 加速） | [deepseek-ai/DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) |
| **DeepSeek-V4-Pro-Base** | 1.6T | 基座（预训练） | [deepseek-ai/DeepSeek-V4-Pro-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base) |
| **DeepSeek-V4-Pro** | 862B | 对话（指令微调） | [deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) |
| **DeepSeek-V4-Pro-DSpark** | 889B | 对话（DSpark 加速） | [deepseek-ai/DeepSeek-V4-Pro-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark) |

> **注意：** DSpark 变体使用相同的模型权重，但配备了加速推理内核。

## ModelScope（国内用户 · 下载更快）

国内用户可通过 ModelScope 获得更快的下载速度：

[https://modelscope.cn/collections/deepseek-ai/DeepSeek-V4](https://modelscope.cn/collections/deepseek-ai/DeepSeek-V4) [$TRAE_REF](https://modelscope.cn/collections/deepseek-ai/DeepSeek-V4)

| 模型 | ModelScope 链接 |
|-------|----------------|
| DeepSeek-V4-Flash | [deepseek-ai/DeepSeek-V4-Flash](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash) |
| DeepSeek-V4-Pro | [deepseek-ai/DeepSeek-V4-Pro](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro) |

## 下载方式

### 使用 Hugging Face CLI

```bash
# 安装 huggingface-cli
pip install huggingface_hub

# 下载 DeepSeek-V4-Flash（推荐大多数用户使用）
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir ./checkpoints/DeepSeek-V4-Flash

# 下载 DeepSeek-V4-Pro
huggingface-cli download deepseek-ai/DeepSeek-V4-Pro --local-dir ./checkpoints/DeepSeek-V4-Pro
```

### 使用 ModelScope CLI（国内更快）

```bash
# 安装 modelscope
pip install modelscope

# 下载 DeepSeek-V4-Flash
modelscope download --model deepseek-ai/DeepSeek-V4-Flash --local-dir ./checkpoints/DeepSeek-V4-Flash

# 下载 DeepSeek-V4-Pro
modelscope download --model deepseek-ai/DeepSeek-V4-Pro --local-dir ./checkpoints/DeepSeek-V4-Pro
```

### 使用 Python

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

## 下载后的文件结构

下载完成后，预期的目录结构如下：

```
checkpoints/
├── WEIGHTS.md                       # 本文件
├── WEIGHTS_zh.md                    # 本文件（中文版）
├── DeepSeek-V4-Flash/               # Flash 对话模型
│   ├── config.json
│   ├── tokenizer.json
│   ├── model-00001-of-000NN.safetensors
│   └── ...
├── DeepSeek-V4-Pro/                 # Pro 对话模型（如已下载）
│   ├── config.json
│   ├── tokenizer.json
│   ├── model-00001-of-000NN.safetensors
│   └── ...
└── model.ckpt                       # 转换后的 MindSpore 检查点
```

## 权重转换（MindSpore）

Hugging Face 的 PyTorch 权重需要转换为 MindSpore 格式后才能使用：

```python
import mindspore as ms
from safetensors import safe_open

# 示例：将 safetensors 转换为 MindSpore 检查点
# （实际转换脚本可能因模型结构而异）
ms_parameters = []
with safe_open("./checkpoints/DeepSeek-V4-Flash/model-00001-of-000NN.safetensors", framework="np") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        ms_param = ms.Parameter(ms.Tensor(tensor), name=key)
        ms_parameters.append(ms_param)

ms.save_checkpoint(ms_parameters, "./checkpoints/model.ckpt")
print("检查点转换成功！")
```

## 硬件要求

| 模型 | 存储空间 | 推荐 GPU |
|-------|---------|-----------------|
| DeepSeek-V4-Flash | ~300GB | 24GB+ (RTX 4090/5090) 配合 FP8 |
| DeepSeek-V4-Pro | ~2TB | 80GB+ (H100/A100) |