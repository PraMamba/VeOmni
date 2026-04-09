---
name: add-model
description: Guide for adding a new HuggingFace model to VeOmni. Use when user wants to support a new model architecture for training.
---

# Add Model to VeOmni

Add support for a new HuggingFace model architecture in VeOmni.

## When to Use

- User asks "how do I add a model to VeOmni?"
- User wants to support a new model family (e.g., Gemma, Mistral)
- User mentions adding model registration or model patches

## Prerequisites

- Target model available on HuggingFace with `config.json`
- Model uses a transformer architecture (decoder-only or encoder-decoder)
- Know the HuggingFace `model_type` string

## Step-by-Step Guide

### Step 1: Analyze the Target Model

Read the HuggingFace model source to identify:
- `model_type` string (from `config.json`)
- Architecture details (attention, FFN, normalization)
- Special features (MoE, sliding window, multi-latent attention)
- RoPE variant

### Step 2: Create Model Directory

```
veomni/models/transformers/<model_name>/
├── __init__.py           # Registration + imports
├── modeling_<name>.py    # Custom modeling (if needed)
├── gpu_patch.py          # GPU-specific optimizations
├── npu_patch.py          # NPU-specific adaptations
└── parallel_plan.py      # MoE EP plan (if MoE model)
```

### Step 3: Register the Model

In `veomni/models/transformers/<model_name>/__init__.py`:

```python
from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY

# If using custom modeling:
from .modeling_<name> import <Name>ForCausalLM
MODELING_REGISTRY.register("<model_type>", <Name>ForCausalLM)

# If using custom config:
from .configuration_<name> import <Name>Config
MODEL_CONFIG_REGISTRY.register("<model_type>", <Name>Config)
```

### Step 4: Import in Parent `__init__.py`

Add to `veomni/models/transformers/__init__.py`:

```python
from . import <model_name>
```

### Step 5: Create GPU/NPU Patches

**GPU Patch** (`gpu_patch.py`):
- Flash Attention integration
- Fused operations (cross-entropy, RMSNorm)
- SP-aware attention variants

**NPU Patch** (`npu_patch.py`):
- Ascend-compatible attention
- NPU-specific optimizations

### Step 6: Add Parallel Plan (MoE only)

If model has MoE layers, create `parallel_plan.py`:

```python
def get_parallel_plan(model_config, parallel_state):
    """Define EP module mapping for FSDP."""
    ...
```

### Step 7: Create Training Config

Add YAML config in `configs/text/<model_name>.yml` (or `multimodal/`):

```yaml
model:
  model_path: "org/model-name"
  ops_implementation:
    attn_implementation: "flash_attention_2"
data:
  data_type: "conversation"
  max_seq_len: 4096
train:
  ...
```

### Step 8: Add Tests

Create `tests/models/test_<model_name>.py`:
- Test model registration
- Test config loading
- Test forward pass (small model)
- Test GPU/NPU patch application

### Step 9: Verify

```bash
# Lint
make commit

# Run model tests
pytest tests/models/test_<model_name>.py -v

# Verify registration
python -c "from veomni.models.loader import MODELING_REGISTRY; print(MODELING_REGISTRY.valid_keys())"

# Check patchgen (if generated patches)
python -m veomni.patchgen.codegen --check
```
