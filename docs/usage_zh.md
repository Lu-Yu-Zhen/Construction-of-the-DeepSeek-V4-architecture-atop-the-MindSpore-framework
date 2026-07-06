# 使用指南

本文档介绍如何使用 DeepSeek-V4 MindSpore 实现进行推理、训练和测试。

## 快速开始

### 创建模型

```python
from src.config import flash_config, pro_config
from src.model import DeepSeekV4Model

# Flash 配置（284B 总参数 / 13B 激活参数）
config = flash_config()
model = DeepSeekV4Model(config)

# Pro 配置（1.6T 总参数 / 49B 激活参数）
# pro_cfg = pro_config()
# pro_model = DeepSeekV4Model(pro_cfg)
```

### 前向传播

```python
import mindspore as ms
from mindspore import Tensor, dtype as mstype
import numpy as np

# 创建模拟输入
batch_size, seq_len = 1, 128
input_ids = Tensor(
    np.random.randint(0, 128000, (batch_size, seq_len)).astype(np.int32),
    mstype.int32
)

# 前向传播
lm_logits, mtp_logits, balance_loss = model(input_ids)
print(f"LM Logits 形状: {lm_logits.shape}")
print(f"均衡损失: {balance_loss}")
```

## 推理

### 使用推理脚本

```bash
# 基本推理
python scripts/inference.py --model flash --prompt "你好，我是" --max-len 256

# 加载检查点
python scripts/inference.py --model flash --checkpoint ./checkpoints/model.ckpt

# 自定义温度参数
python scripts/inference.py --model flash --prompt "解释一下量子计算" --temperature 0.7 --max-len 512
```

### 编程式推理

```python
from src.config import flash_config
from src.model import DeepSeekV4Model
import mindspore as ms
from mindspore import Tensor, ops
import numpy as np

class DeepSeekV4Inference:
    def __init__(self, config_name="flash"):
        self.config = flash_config() if config_name == "flash" else pro_config()
        self.model = DeepSeekV4Model(self.config)
        self.model.set_train(False)

    def generate(self, prompt_ids, max_new_tokens=256, temperature=0.7):
        """根据提示 Token ID 生成文本。"""
        input_ids = Tensor(prompt_ids, ms.int32)
        for _ in range(max_new_tokens):
            logits, _, _ = self.model(input_ids)
            next_logits = logits[:, -1, :] / temperature
            next_token = ops.argmax(next_logits, axis=-1)
            input_ids = ops.concat([input_ids, next_token.unsqueeze(0)], axis=1)
        return input_ids.asnumpy()

# 使用示例
inferencer = DeepSeekV4Inference("flash")
prompt = np.array([[1, 2, 3]])  # 已分词的提示
output = inferencer.generate(prompt, max_new_tokens=100)
```

## 训练

### 使用训练脚本

```bash
# 训练 Flash 模型
python scripts/train.py --model flash --epochs 10 --batch-size 4 --lr 2.7e-4

# 训练 Pro 模型（需要大量算力）
python scripts/train.py --model pro --epochs 5 --batch-size 2 --lr 1.5e-4

# 启用梯度检查点
python scripts/train.py --model flash --epochs 10 --gradient-checkpointing

# 混合精度训练
python scripts/train.py --model flash --epochs 10 --amp-level O2
```

### 编程式训练

```python
from src.config import flash_config
from src.model import DeepSeekV4Model
from src.optimizer import MuonOptimizer
import mindspore as ms
from mindspore import nn

# 设置
config = flash_config()
model = DeepSeekV4Model(config)

# 分离 Muon 参数（2D 权重）和其他参数
muon_params = []
other_params = []
for name, param in model.parameters_and_names():
    if 'weight' in name and param.ndim == 2:
        muon_params.append(param)
    else:
        other_params.append(param)

# 优化器
optimizer = MuonOptimizer(
    params=[
        {'params': muon_params, 'lr': 2.7e-4, 'muon': True},
        {'params': other_params, 'lr': 2.7e-4, 'muon': False}
    ],
    learning_rate=2.7e-4,
    weight_decay=0.1
)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(10):
    for batch in dataloader:
        loss = train_step(model, batch, loss_fn, optimizer)
        print(f"Epoch {epoch}, Loss: {loss}")
```

## 运行测试

```bash
# 运行所有测试
python -m pytest tests/test_model.py -v

# 运行特定测试
python -m pytest tests/test_model.py::test_forward_pass -v

# 带覆盖率运行
python -m pytest tests/ --cov=src -v
```

## 配置管理

### 加载 YAML 配置

```python
import yaml

# 从 YAML 加载
with open("configs/flash_config.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

# 覆盖特定参数
config_dict["model"]["num_layers"] = 20  # 测试时减少层数

# 创建配置对象
from src.config import FlashConfig
config = FlashConfig(**config_dict["model"])
```

### 自定义配置

```python
from src.config import FlashConfig

# 为测试创建自定义配置
custom_config = FlashConfig(
    vocab_size=128000,
    hidden_size=2048,       # 测试时缩小
    num_layers=6,           # 测试时减少
    num_heads=32,
    num_routed_experts=64,
    num_activated_experts=4,
    max_seq_len=8192,
    n_hc=4
)
```

## 模型导出

### 保存检查点

```python
import mindspore as ms

# 保存模型权重
ms.save_checkpoint(model, "checkpoints/deepseek_v4_flash.ckpt")

# 保存优化器状态
ms.save_checkpoint(optimizer, "checkpoints/optimizer_state.ckpt")
```

### 加载检查点

```python
# 加载模型权重
param_dict = ms.load_checkpoint("checkpoints/deepseek_v4_flash.ckpt")
ms.load_param_into_net(model, param_dict)
```

## 性能优化建议

1. **使用 FP8/FP16：** 启用混合精度以加速训练和推理
2. **梯度检查点：** 在长序列中用计算换显存
3. **批量大小：** 从小批量开始，根据可用显存逐步增加
4. **序列长度：** Flash 配置支持最长 1M Token，但测试时建议从短序列（4K-8K）开始
5. **并行训练：** 使用 MindSpore 的自动并行进行分布式训练