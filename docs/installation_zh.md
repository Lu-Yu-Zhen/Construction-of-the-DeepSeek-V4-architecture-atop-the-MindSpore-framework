# 安装指南

本文档介绍 DeepSeek-V4 MindSpore 实现的安装步骤。

## 前置要求

- **Python >= 3.9**
- **MindSpore >= 2.8.0**
- **CUDA 兼容 GPU**（推荐）或昇腾 Ascend NPU

## 第一步：安装 MindSpore

### 选项 A：CUDA（NVIDIA GPU）

```bash
# CUDA 12.1
pip install mindspore==2.8.0

# 验证安装
python -c "import mindspore; print(mindspore.__version__)"
```

### 选项 B：昇腾 Ascend NPU

```bash
# 安装 Ascend 版 MindSpore
pip install mindspore-ascend==2.8.0
```

### 选项 C：仅 CPU

```bash
pip install mindspore==2.8.0
```

> **注意：** 仅 CPU 模式适用于开发和测试，不适合大规模训练或推理。

## 第二步：安装本包

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/Lu-Yu-Zhen/Construction-of-the-DeepSeek-V4-architecture-atop-the-MindSpore-framework.git
cd Construction-of-the-DeepSeek-V4-architecture-atop-the-MindSpore-framework

# 安装依赖
pip install -r requirements.txt

# 以开发模式安装
pip install -e .
```

### 使用 pip 安装（未来）

```bash
pip install deepseek-v4-mindspore
```

## 第三步：验证安装

```python
import mindspore as ms
from src.config import flash_config
from src.model import DeepSeekV4Model

# 创建测试用的小配置
config = flash_config()
model = DeepSeekV4Model(config)
print(f"模型创建成功：{type(model).__name__}")
print(f"总参数量：{sum(p.numel() for p in model.trainable_params()) / 1e9:.2f}B")
```

## 硬件要求

### 最低要求（仅推理）

| 组件 | Flash (284B) | Pro (1.6T) |
|-----------|-------------|------------|
| GPU 显存 | 24GB+ (FP8) | 80GB+ (FP8) |
| 推荐 GPU | RTX 4090 / 5090 | H100 / A100 |
| 系统内存 | 64GB+ | 256GB+ |
| 存储空间 | 300GB+ | 2TB+ |

### 训练要求

| 组件 | Flash (284B) | Pro (1.6T) |
|-----------|-------------|------------|
| GPU 显存 | 每设备 80GB+ | 每设备 80GB+ |
| GPU 数量 | 8+ (H100/A100) | 64+ (H100/A100) |
| 系统内存 | 512GB+ | 2TB+ |
| 互联方式 | NVLink / InfiniBand | InfiniBand |

## 常见问题

| 问题 | 解决方法 |
|-------|----------|
| `ModuleNotFoundError: No module named 'mindspore'` | 先安装 MindSpore（参见第一步） |
| 显存不足 | 减小 batch size、使用 FP8 或启用梯度检查点 |
| `ASCEND_HOME_PATH` 未设置 | 加载昇腾环境：`source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| CUDA 版本不匹配 | 检查兼容性：`python -c "import mindspore; print(mindspore.run_check())"` |

### 获取帮助

- [MindSpore 文档](https://www.mindspore.cn/docs)
- [MindSpore GitHub Issues](https://github.com/mindspore-ai/mindspore/issues)
- [DeepSeek-V4 论文](https://arxiv.org/abs/2504.09286)