# Usage Guide

This guide covers how to use the DeepSeek-V4 MindSpore implementation for inference, training, and testing.

## Quick Start

### Creating a Model

```python
from src.config import flash_config, pro_config
from src.model import DeepSeekV4Model

# Flash configuration (284B total / 13B activated)
config = flash_config()
model = DeepSeekV4Model(config)

# Pro configuration (1.6T total / 49B activated)
# pro_cfg = pro_config()
# pro_model = DeepSeekV4Model(pro_cfg)
```

### Forward Pass

```python
import mindspore as ms
from mindspore import Tensor, dtype as mstype
import numpy as np

# Create dummy input
batch_size, seq_len = 1, 128
input_ids = Tensor(
    np.random.randint(0, 128000, (batch_size, seq_len)).astype(np.int32),
    mstype.int32
)

# Forward pass
lm_logits, mtp_logits, balance_loss = model(input_ids)
print(f"LM Logits shape: {lm_logits.shape}")
print(f"Balance Loss: {balance_loss}")
```

## Inference

### Using the Inference Script

```bash
# Basic inference
python scripts/inference.py --model flash --prompt "Hello, I am" --max-len 256

# With checkpoint
python scripts/inference.py --model flash --checkpoint ./checkpoints/model.ckpt

# With custom temperature
python scripts/inference.py --model flash --prompt "Explain quantum computing" --temperature 0.7 --max-len 512
```

### Programmatic Inference

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
        """Generate text from prompt token IDs."""
        input_ids = Tensor(prompt_ids, ms.int32)
        for _ in range(max_new_tokens):
            logits, _, _ = self.model(input_ids)
            next_logits = logits[:, -1, :] / temperature
            next_token = ops.argmax(next_logits, axis=-1)
            input_ids = ops.concat([input_ids, next_token.unsqueeze(0)], axis=1)
        return input_ids.asnumpy()

# Usage
inferencer = DeepSeekV4Inference("flash")
prompt = np.array([[1, 2, 3]])  # Your tokenized prompt
output = inferencer.generate(prompt, max_new_tokens=100)
```

## Training

### Using the Training Script

```bash
# Train Flash model
python scripts/train.py --model flash --epochs 10 --batch-size 4 --lr 2.7e-4

# Train Pro model (requires significant compute)
python scripts/train.py --model pro --epochs 5 --batch-size 2 --lr 1.5e-4

# With gradient checkpointing
python scripts/train.py --model flash --epochs 10 --gradient-checkpointing

# Mixed precision training
python scripts/train.py --model flash --epochs 10 --amp-level O2
```

### Programmatic Training

```python
from src.config import flash_config
from src.model import DeepSeekV4Model
from src.optimizer import MuonOptimizer
import mindspore as ms
from mindspore import nn

# Setup
config = flash_config()
model = DeepSeekV4Model(config)

# Separate muon parameters (2D weights) from others
muon_params = []
other_params = []
for name, param in model.parameters_and_names():
    if 'weight' in name and param.ndim == 2:
        muon_params.append(param)
    else:
        other_params.append(param)

# Optimizer
optimizer = MuonOptimizer(
    params=[
        {'params': muon_params, 'lr': 2.7e-4, 'muon': True},
        {'params': other_params, 'lr': 2.7e-4, 'muon': False}
    ],
    learning_rate=2.7e-4,
    weight_decay=0.1
)

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    for batch in dataloader:
        loss = train_step(model, batch, loss_fn, optimizer)
        print(f"Epoch {epoch}, Loss: {loss}")
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/test_model.py -v

# Run specific test
python -m pytest tests/test_model.py::test_forward_pass -v

# Run with coverage
python -m pytest tests/ --cov=src -v
```

## Configuration

### Loading YAML Configs

```python
import yaml

# Load from YAML
with open("configs/flash_config.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

# Override specific parameters
config_dict["model"]["num_layers"] = 20  # Reduce layers for testing

# Create config object
from src.config import FlashConfig
config = FlashConfig(**config_dict["model"])
```

### Custom Configuration

```python
from src.config import FlashConfig

# Create a custom config for testing
custom_config = FlashConfig(
    vocab_size=128000,
    hidden_size=2048,       # Reduced for testing
    num_layers=6,           # Reduced for testing
    num_heads=32,
    num_routed_experts=64,
    num_activated_experts=4,
    max_seq_len=8192,
    n_hc=4
)
```

## Model Export

### Saving Checkpoints

```python
import mindspore as ms

# Save model weights
ms.save_checkpoint(model, "checkpoints/deepseek_v4_flash.ckpt")

# Save optimizer state
ms.save_checkpoint(optimizer, "checkpoints/optimizer_state.ckpt")
```

### Loading Checkpoints

```python
# Load model weights
param_dict = ms.load_checkpoint("checkpoints/deepseek_v4_flash.ckpt")
ms.load_param_into_net(model, param_dict)
```

## Performance Tips

1. **Use FP8/FP16:** Enable mixed precision for faster training and inference
2. **Gradient Checkpointing:** Trade compute for memory in long sequences
3. **Batch Size:** Start with small batch sizes and increase based on available memory
4. **Sequence Length:** The Flash config supports up to 1M tokens, but start with shorter sequences (4K-8K) for testing
5. **Parallelism:** Use MindSpore's automatic parallel for distributed training