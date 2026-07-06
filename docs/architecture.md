# Architecture Overview

DeepSeek-V4 is a Mixture-of-Experts (MoE) large language model that introduces several novel components for efficient million-token context modeling. This document provides a detailed walkthrough of the architecture implemented in this repository.

## Table of Contents

- [Overall Architecture](#overall-architecture)
- [mHC: Manifold-Constrained Hyper-Connections](#mhc-manifold-constrained-hyper-connections)
- [CSA: Compressed Sparse Attention](#csa-compressed-sparse-attention)
- [HCA: Heavily Compressed Attention](#hca-heavily-compressed-attention)
- [DeepSeekMoE](#deepseekmoe)
- [MTP: Multi-Token Prediction](#mtp-multi-token-prediction)
- [Muon Optimizer](#muon-optimizer)
- [Configuration Variants](#configuration-variants)

## Overall Architecture

DeepSeek-V4 follows a decoder-only Transformer architecture with the following key innovations:

```
Input Tokens
    |
    v
[Embedding Layer]
    |
    v
[TransformerBlock] x N  (43 for Flash, 61 for Pro)
    |-- mHC Residual Connection
    |-- CSA / HCA Hybrid Attention (alternating layers)
    |-- DeepSeekMoE FFN
    |
    v
[MTP Module] (optional, during training)
    |
    v
[LM Head] -> Output Logits
```

The first 2 layers use Sliding Window Attention (SWA) or HCA instead of CSA, depending on the configuration.

## mHC: Manifold-Constrained Hyper-Connections

**File:** `src/mhc.py`

mHC replaces standard residual connections (`x + F(x)`) with a more expressive formulation. The residual state is expanded `n_hc` times and updated through constrained matrix operations.

### Key Components

- **Sinkhorn-Knopp Iteration:** Normalizes a matrix to be doubly stochastic (rows and columns sum to 1), ensuring the constraint `B_l @ 1 = 1` and `1^T @ A_l = 1^T`.
- **Expanded Residual Stream:** Instead of a single residual path, the state is projected into `n_hc` parallel streams, allowing richer information flow between layers.

### Mathematical Formulation

```
X_{l+1} = B_l @ X_l + C_l * F_l(A_l @ X_l)
```

Where:
- `A_l`, `B_l`, `C_l` are learned linear projections
- `A_l` and `B_l` are doubly stochastic (via Sinkhorn-Knopp)
- `F_l` is the attention + FFN computation at layer l
- `X_l` is the expanded residual state at layer l
- `n_hc = 4` expansions per layer

### Implementation Details

```python
from src.mhc import SinkhornKnopp, mHC

# Sinkhorn-Knopp normalization
sk = SinkhornKnopp(n_iters=5, tau=0.1)
normalized_matrix = sk(matrix)

# mHC residual connection
hc = mHC(hidden_size=4096, n_hc=4)
output = hc(residual_state, layer_output)
```

## CSA: Compressed Sparse Attention

**File:** `src/attention.py`

CSA is a dual-branch attention mechanism that combines overlapping KV compression with sparse selection, achieving efficient long-context processing.

### Components

| Component | Description |
|-----------|-------------|
| **KV Compressor** | Downsamples KV pairs by a compression rate (m=4) using overlapping windows |
| **Lightning Indexer** | Selects top-k most relevant compressed KV positions for sparse attention |
| **Multi-Query Attention (MQA)** | Single KV head shared across all query heads for efficiency |
| **Sliding Window Attention (SWA)** | Local attention window for fine-grained context |

### Architecture

```
Query
 |---[SWA Branch]-----> Sliding Window Attention
 |---[CSA Branch]-----> KV Compression (rate m=4)
                         -> Lightning Indexer (top-k selection)
                         -> Sparse Attention with selected KV
```

## HCA: Heavily Compressed Attention

**File:** `src/attention.py`

HCA is a single-branch attention mechanism designed for ultra-long contexts. It uses a much higher compression rate (m'=128) compared to CSA.

### Key Features

- Single-branch heavy KV compression
- Multi-Query Attention (MQA) for efficiency
- Sliding window for local context
- Designed for contexts exceeding 100K tokens

### Attention Layer Distribution

| Config | CSA Layers | HCA Layers | SWA/HCA (first 2) |
|--------|-----------|-----------|-------------------|
| Flash | 20 | 21 | SWA |
| Pro | 29 | 30 | HCA |

## DeepSeekMoE

**File:** `src/moe.py`

DeepSeekMoE replaces standard FFN layers with a mixture of specialized experts, activating only a subset of experts per token for computational efficiency.

### Key Components

| Component | Description |
|-----------|-------------|
| **Shared Expert** | Processes all tokens (common knowledge) |
| **Routed Experts** | Specialized experts activated per token |
| **Router** | Gating mechanism with Sqrt(Softplus) scores |
| **Hash Router** | Deterministic routing for early layers (first 2) |
| **Auxiliary-Loss-Free Balancing** | Learned bias for load balancing |

### Expert Configuration

| Parameter | Flash | Pro |
|-----------|-------|-----|
| Routed Experts | 256 | 384 |
| Activated Experts | 6 | 6 |
| Shared Experts | 2 | 2 |
| Activation | SwiGLU | SwiGLU |

### Routing Score

```python
# Sqrt(Softplus) routing
scores = sqrt(softplus(router_logits))
```

## MTP: Multi-Token Prediction

**File:** `src/mtp.py`

MTP improves training signal density by predicting multiple future tokens simultaneously. The module uses shifted embedding concatenation with an extra Transformer block.

### Architecture

```
Input Embeddings
    |
    [Shift + Concat] <-- previous token prediction
    |
    v
[MTP Transformer Block]
    |
    v
[Shared LM Head] -> Predict next token
```

### Training Benefits

- Increased training signal density (N tokens predicted per position)
- Improved sample efficiency
- Shared LM head reduces parameter overhead
- Compatible with standard autoregressive inference

## Muon Optimizer

**File:** `src/optimizer.py`

The Muon optimizer is designed for large-scale neural network training, combining orthogonalization with momentum.

### Key Features

- **Newton-Schulz Orthogonalization:** 8 iterations for main computation + 2 for refinement
- **Nesterov Momentum:** Look-ahead gradient updates for faster convergence
- **RMS Rescaling:** Adaptive learning rate per parameter group
- **Weight Matrix Updates:** Applied only to 2D weight matrices (not biases, norms, or embeddings)

### Usage

```python
from src.optimizer import MuonOptimizer

# Separate parameters into weight matrices and others
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

## Configuration Variants

Two configuration files are provided in the `configs/` directory:

| Config File | Description |
|-------------|-------------|
| `configs/flash_config.yaml` | 284B total / 13B activated parameters |
| `configs/pro_config.yaml` | 1.6T total / 49B activated parameters |

### Parameter Comparison

| Parameter | Flash | Pro |
|-----------|-------|-----|
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

## References

- Paper: *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*
- Framework: [MindSpore](https://www.mindspore.cn/)
- Source Code: `src/` directory in this repository