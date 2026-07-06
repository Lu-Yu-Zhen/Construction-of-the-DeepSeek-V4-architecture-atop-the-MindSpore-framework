# 架构概述

DeepSeek-V4 是一个混合专家（MoE）大语言模型，引入了多个创新组件以实现高效的百万级 Token 上下文建模。本文档详细介绍了本仓库实现的模型架构。

## 目录

- [整体架构](#整体架构)
- [mHC：流形约束超连接](#mhc流形约束超连接)
- [CSA：压缩稀疏注意力](#csa压缩稀疏注意力)
- [HCA：高度压缩注意力](#hca高度压缩注意力)
- [DeepSeekMoE](#deepseekmoe)
- [MTP：多 Token 预测](#mtp多-token-预测)
- [Muon 优化器](#muon-优化器)
- [配置变体](#配置变体)

## 整体架构

DeepSeek-V4 采用仅解码器（Decoder-only）Transformer 架构，核心创新如下：

```
输入 Token
    |
    v
[嵌入层]
    |
    v
[TransformerBlock] x N  (Flash: 43 层, Pro: 61 层)
    |-- mHC 残差连接
    |-- CSA / HCA 混合注意力（交替层）
    |-- DeepSeekMoE FFN
    |
    v
[MTP 模块]（可选，训练时）
    |
    v
[LM Head] -> 输出 Logits
```

前 2 层根据配置使用滑动窗口注意力（SWA）或 HCA 代替 CSA。

## mHC：流形约束超连接

**文件：** `src/mhc.py`

mHC 用更具表达力的公式替代了标准残差连接（`x + F(x)`）。残差状态被扩展 `n_hc` 倍，并通过约束矩阵运算更新。

### 关键组件

- **Sinkhorn-Knopp 迭代：** 将矩阵归一化为双随机矩阵（行和列之和均为 1），确保满足约束 `B_l @ 1 = 1` 和 `1^T @ A_l = 1^T`。
- **扩展残差流：** 不再使用单一残差路径，而是将状态投影到 `n_hc` 条并行流中，实现层间更丰富的信息流动。

### 数学公式

```
X_{l+1} = B_l @ X_l + C_l * F_l(A_l @ X_l)
```

其中：
- `A_l`、`B_l`、`C_l` 是学习到的线性投影
- `A_l` 和 `B_l` 是双随机矩阵（通过 Sinkhorn-Knopp 归一化）
- `F_l` 是第 l 层的注意力 + FFN 计算
- `X_l` 是第 l 层的扩展残差状态
- 每层扩展数 `n_hc = 4`

### 实现示例

```python
from src.mhc import SinkhornKnopp, mHC

# Sinkhorn-Knopp 归一化
sk = SinkhornKnopp(n_iters=5, tau=0.1)
normalized_matrix = sk(matrix)

# mHC 残差连接
hc = mHC(hidden_size=4096, n_hc=4)
output = hc(residual_state, layer_output)
```

## CSA：压缩稀疏注意力

**文件：** `src/attention.py`

CSA 是一种双分支注意力机制，结合了重叠 KV 压缩与稀疏选择，实现高效的长上下文处理。

### 组件

| 组件 | 说明 |
|-----------|-------------|
| **KV 压缩器** | 通过重叠窗口以压缩率（m=4）对 KV 对进行降采样 |
| **Lightning Indexer** | 选择最相关的 top-k 压缩 KV 位置进行稀疏注意力 |
| **多查询注意力（MQA）** | 所有查询头共享单个 KV 头以提高效率 |
| **滑动窗口注意力（SWA）** | 用于细粒度上下文的局部注意力窗口 |

### 架构

```
Query
 |---[SWA 分支]-----> 滑动窗口注意力
 |---[CSA 分支]-----> KV 压缩（压缩率 m=4）
                         -> Lightning Indexer（top-k 选择）
                         -> 与选中的 KV 进行稀疏注意力
```

## HCA：高度压缩注意力

**文件：** `src/attention.py`

HCA 是为超长上下文设计的单分支注意力机制。与 CSA 相比，它使用更高的压缩率（m'=128）。

### 关键特性

- 单分支重度 KV 压缩
- 多查询注意力（MQA）提升效率
- 滑动窗口处理局部上下文
- 专为超过 100K Token 的上下文设计

### 注意力层分布

| 配置 | CSA 层数 | HCA 层数 | 前 2 层 |
|--------|-----------|-----------|-------------------|
| Flash | 20 | 21 | SWA |
| Pro | 29 | 30 | HCA |

## DeepSeekMoE

**文件：** `src/moe.py`

DeepSeekMoE 用混合专家层替代标准 FFN 层，每个 Token 仅激活部分专家，以提高计算效率。

### 关键组件

| 组件 | 说明 |
|-----------|-------------|
| **共享专家** | 处理所有 Token（通用知识） |
| **路由专家** | 按 Token 激活的专用专家 |
| **路由器** | 基于 Sqrt(Softplus) 得分的门控机制 |
| **哈希路由** | 前 2 层的确定性路由 |
| **无辅助损失均衡** | 用于负载均衡的学习偏置 |

### 专家配置

| 参数 | Flash | Pro |
|-----------|-------|-----|
| 路由专家数 | 256 | 384 |
| 激活专家数 | 6 | 6 |
| 共享专家数 | 2 | 2 |
| 激活函数 | SwiGLU | SwiGLU |

### 路由得分

```python
# Sqrt(Softplus) 路由
scores = sqrt(softplus(router_logits))
```

## MTP：多 Token 预测

**文件：** `src/mtp.py`

MTP 通过同时预测多个未来 Token 来提高训练信号密度。该模块使用偏移嵌入拼接和额外的 Transformer 块。

### 架构

```
输入嵌入
    |
    [偏移 + 拼接] <-- 前一个 Token 预测
    |
    v
[MTP Transformer 块]
    |
    v
[共享 LM Head] -> 预测下一个 Token
```

### 训练优势

- 提高训练信号密度（每个位置预测 N 个 Token）
- 提升样本效率
- 共享 LM Head 减少参数开销
- 兼容标准自回归推理

## Muon 优化器

**文件：** `src/optimizer.py`

Muon 优化器专为大规模神经网络训练设计，结合了正交化与动量。

### 关键特性

- **Newton-Schulz 正交化：** 8 次主迭代 + 2 次精炼迭代
- **Nesterov 动量：** 前瞻梯度更新，加速收敛
- **RMS 重新缩放：** 每个参数组的自适应学习率
- **权重矩阵更新：** 仅应用于 2D 权重矩阵（不作用于偏置、归一化层或嵌入层）

### 使用示例

```python
from src.optimizer import MuonOptimizer

# 将参数分为权重矩阵和其他
muon_params = [p for name, p in model.parameters_and_names() if 'weight' in name and p.ndim == 2]
optimizer = MuonOptimizer(
    params=[
        {'params': muon_params, 'lr': 2.7e-4, 'muon': True},
        {'params': model.trainable_params(), 'lr': 2.7e-4, 'muon': False}
    ],
    learning_rate=2.7e-4,
    weight_decay=0.1
)
```

## 配置变体

`configs/` 目录下提供两个配置文件：

| 配置文件 | 说明 |
|-------------|-------------|
| `configs/flash_config.yaml` | 284B 总参数 / 13B 激活参数 |
| `configs/pro_config.yaml` | 1.6T 总参数 / 49B 激活参数 |

### 参数对比

| 参数 | Flash | Pro |
|-----------|-------|-----|
| 总参数量 | 284B | 1.6T |
| 激活参数量 | 13B | 49B |
| 层数 | 43 | 61 |
| 隐藏层大小 | 4096 | 7168 |
| 查询头数 | 64 | 128 |
| 头维度 | 512 | 512 |
| 路由专家数 | 256 | 384 |
| 激活专家数 | 6 | 6 |
| CSA 压缩率 | 4 | 4 |
| HCA 压缩率 | 128 | 128 |
| CSA Top-k | 512 | 1024 |
| mHC 扩展数 | 4 | 4 |
| 最大序列长度 | 1M | 1M |

## 参考

- 论文：*DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*
- 框架：[MindSpore](https://www.mindspore.cn/)
- 源代码：本仓库中的 `src/` 目录