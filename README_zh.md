# DeepSeek-V4

基于论文 *"DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"* 的 **MindSpore** 实现。

本仓库提供了 DeepSeek-V4 完整架构的清晰模块化实现，支持 **Flash**（284B 总参数 / 13B 激活参数）和 **Pro**（1.6T 总参数 / 49B 激活参数）两种配置。

[English](README.md) | 中文

## 关于 MindSpore

[MindSpore](https://www.mindspore.cn/) 是由 [华为](https://www.huawei.com/) 开发的开源深度学习框架，面向云端和边缘 AI 场景设计。它对华为昇腾 AI 处理器提供原生支持，同时也可在 CPU 和 GPU 上运行。

MindSpore 的核心特性包括：

- **自动微分** — 函数式自动微分框架，同时支持静态图和动态图，提供灵活的模型开发体验。
- **昇腾 NPU 原生支持** — 针对华为昇腾系列（Ascend 910、310 等）深度优化，通过软硬件协同设计实现最大吞吐量。
- **云边端统一框架** — 同一套模型代码可无缝部署在云端数据中心、边缘设备和移动端，无需修改。
- **图模式高性能** — `GRAPH_MODE` 将整个网络编译为静态图以获得优化执行性能；`PYNATIVE_MODE` 提供动态调试能力，便于开发阶段使用。
- **丰富生态** — 包含 MindFormers（大模型训练）、MindX（行业应用）、MindInsight（模型调试与性能分析）等组件。

本实现使用 MindSpore 的 `nn.Cell` 作为所有模型组件的基类（对应 PyTorch 中的 `nn.Module`），使用 `construct()` 替代 `forward()` 作为计算入口。

## 关于 DeepSeek

[DeepSeek（深度求索）](https://www.deepseek.com/) 是一家 AI 研究实验室，已发布了一系列具有广泛影响力的开源大语言模型，以强劲的性能和创新的架构设计著称。DeepSeek 模型家族包括：

- **DeepSeek-V1** — 早期的 MoE 语言模型，验证了稀疏专家架构在大规模场景下的可行性。
- **DeepSeek-V2** — 引入了多头潜在注意力（MLA）和细粒度 DeepSeekMoE，显著降低了推理成本。
- **DeepSeek-V3** — 671B 参数的 MoE 模型，每个 token 激活 37B 参数，引入多 Token 预测（MTP）和无辅助损失负载均衡，在 14.8T token 上完成训练，效率极高。
- **DeepSeek-R1** — 专注推理能力的模型，通过强化学习训练，在数学和编程基准上达到了与 OpenAI o1 相当的水平。
- **DeepSeek-V4** — 最新一代模型，引入 mHC 残差连接、CSA/HCA 混合注意力和 Muon 优化器，实现百万级 token 上下文的高效智能。

DeepSeek 的开源理念使其成为开源大模型生态中最具影响力的力量之一，其模型被全球研究者和开发者广泛采用。

## 架构概览

DeepSeek-V4 引入了多个创新组件，用于高效的百万级 token 上下文建模：

- **mHC（流形约束超连接）** — 通过 Sinkhorn-Knopp 迭代引入双随机矩阵约束，替代标准残差连接。残差状态被扩展 `n_hc` 次，更新公式为 `X_{l+1} = B_l @ X_l + C_l * F_l(A_l @ X_l)`。

- **CSA（压缩稀疏注意力）** — 双分支重叠 KV 压缩（压缩率 m=4），配合 Lightning Indexer 进行 top-k 稀疏选择，结合多查询注意力（MQA）和滑动窗口分支。

- **HCA（重度压缩注意力）** — 单分支重度 KV 压缩（压缩率 m'=128），配合 MQA 和滑动窗口，专为超长上下文设计。

- **DeepSeekMoE** — 共享专家 + 路由专家的混合专家架构，采用 Sqrt(Softplus) 路由分数、前几层的 Hash 路由，以及无辅助损失的负载均衡机制。

- **MTP（多 Token 预测）** — 将移位后的 embedding 与隐状态拼接，通过额外的 Transformer Block 处理，共享 LM Head 输出，提升训练信号密度。

- **Muon 优化器** — 混合 Newton-Schulz 正交化（8+2 步），配合 Nesterov 动量和 RMS 缩放，用于权重矩阵更新。

## 项目结构

```
DeepSeek-V4/
├── src/                        # 核心模型实现
│   ├── __init__.py
│   ├── config.py               # 模型配置（Flash & Pro）
│   ├── normalization.py        # RMSNorm, 旋转位置编码
│   ├── mhc.py                  # Sinkhorn-Knopp, mHC 残差连接
│   ├── attention.py            # CSA, HCA, SWA, KV 压缩器, Lightning Indexer
│   ├── moe.py                  # SwiGLU Expert, DeepSeekMoE
│   ├── mtp.py                  # 多 Token 预测
│   ├── model.py                # TransformerBlock, DeepSeekV4Model
│   └── optimizer.py            # Muon 优化器
├── configs/                    # YAML 配置文件
│   ├── flash_config.yaml
│   └── pro_config.yaml
├── scripts/                    # 训练与推理脚本
│   ├── train.py
│   └── inference.py
├── tests/                      # 单元测试
│   └── test_model.py
├── docs/                       # 文档
├── data/                       # 数据目录
├── checkpoints/                # 模型检查点
├── main.py                     # 入口脚本（含演示）
├── model.py                    # 原始单体实现（参考保留）
├── requirements.txt
├── setup.py
└── README.md
```

## 环境要求

- Python >= 3.9
- MindSpore >= 2.8.0
- NumPy >= 1.20

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/DeepSeek-V4.git
cd DeepSeek-V4

# 安装依赖
pip install -r requirements.txt

# 以开发模式安装
pip install -e .
```

## 使用方法

### 快速开始

```python
from src.config import flash_config
from src.model import DeepSeekV4Model

# 创建 Flash 配置
config = flash_config()
model = DeepSeekV4Model(config)

# 前向传播
import mindspore as ms
from mindspore import Tensor, dtype as mstype
import numpy as np

input_ids = Tensor(np.random.randint(0, 128000, (1, 128)).astype(np.int32), mstype.int32)
lm_logits, mtp_logits, balance_loss = model(input_ids)
```

### 训练

```bash
# 训练 Flash 模型
python scripts/train.py --model flash --epochs 10 --batch-size 4 --lr 2.7e-4

# 训练 Pro 模型（需要大量计算资源）
python scripts/train.py --model pro --epochs 5 --batch-size 2
```

### 推理

```bash
# 生成文本
python scripts/inference.py --model flash --prompt "你好，我是" --max-len 256

# 加载检查点推理
python scripts/inference.py --model flash --checkpoint ./checkpoints/model.ckpt
```

### 运行测试

```bash
python -m pytest tests/test_model.py -v
```

## 模型配置

| 参数 | Flash | Pro |
|---|---|---|
| 总参数量 | 284B | 1.6T |
| 激活参数量 | 13B | 49B |
| Transformer 层数 | 43 | 61 |
| 隐层维度 | 4096 | 7168 |
| 查询头数 | 64 | 128 |
| 头维度 | 512 | 512 |
| 路由专家数 | 256 | 384 |
| 激活专家数 | 6 | 6 |
| CSA 压缩率 | 4 | 4 |
| HCA 压缩率 | 128 | 128 |
| CSA Top-k | 512 | 1024 |
| mHC 扩展因子 | 4 | 4 |
| 最大序列长度 | 1M | 1M |

## 关键设计决策

1. **混合注意力**：各层在 CSA（处理中等距离依赖）和 HCA（处理超长上下文）之间交替，前 2 层使用 SWA（Flash）或 HCA（Pro）。

2. **mHC 残差流**：不同于简单的 `x + F(x)`，残差状态被扩展 `n_hc=4` 次，通过约束矩阵运算进行更新，实现更丰富的信息流动。

3. **KV 压缩**：CSA 通过双分支重叠窗口实现 4 倍压缩；HCA 实现 128 倍压缩，适用于极端长度的上下文。

4. **无辅助损失负载均衡**：通过在路由分数上添加可学习的偏置项来实现负载均衡，避免了可能干扰模型训练的辅助损失。

## 参考文献

- 论文：*DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*
- 框架：[MindSpore](https://www.mindspore.cn/)
- 相关工作：DeepSeek-V3, DeepSeekMoE

## 许可证

本项目基于 MIT 许可证开源 — 详见 [LICENSE](LICENSE) 文件。

## 免责声明

本实现为基于已发表论文的研究性代码，仅供学术和教育用途。
