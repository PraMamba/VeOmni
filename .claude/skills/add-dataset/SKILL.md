---
name: add-dataset
description: Guide for adding a new dataset type to VeOmni. Use when user wants to support a new data source or format.
---

# Add Dataset to VeOmni

Add support for a new dataset type or data transform in VeOmni.

## When to Use

- User asks "how do I add a dataset?"
- User wants to support a new data format
- User needs a new data transform function

## Step-by-Step Guide

### Option A: Add a New Dataset Type

Location: `veomni/data/dataset.py`

```python
@DATASET_REGISTRY.register("my_dataset_type")
def build_my_dataset(
    dataset_config: DataArguments,
    data_transform: Callable,
    split: str = "train",
    **kwargs,
) -> Dataset:
    """Build dataset from my custom source."""
    # Load raw data
    raw_data = ...

    # Apply transform
    dataset = raw_data.map(data_transform, ...)

    return dataset
```

### Option B: Add a New Data Transform

Location: `veomni/data/data_transform.py`

```python
@DATA_TRANSFORM_REGISTRY.register("my_transform")
def process_my_format(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Transform raw example into model-ready tensors."""
    # Tokenize
    input_ids = tokenizer.encode(...)

    # Create labels
    labels = ...

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
```

### Option C: Add a Multimodal Preprocessor

Location: `veomni/data/multimodal/preprocess.py`

```python
@PREPROCESSOR_REGISTRY.register("my_preprocessor")
def my_preprocessor(
    example: Dict,
    processor: ProcessorMixin,
    **kwargs,
) -> Dict:
    """Preprocess multimodal data (images, audio, video)."""
    ...
```

### Configuration

In training config YAML:

```yaml
data:
  data_type: "my_dataset_type"      # or existing type
  data_transform: "my_transform"    # transform to apply
  dataset_path: "path/to/data"
  max_seq_len: 4096
```

### Testing

Create `tests/data/test_my_dataset.py`:

```python
def test_my_dataset_builds():
    """Test dataset construction."""
    ...

def test_my_transform_tokenizes_correctly():
    """Test transform produces correct tensors."""
    ...
```

### Verification

```bash
# Lint
make commit

# Run data tests
pytest tests/data/ -v

# Verify registration
python -c "from veomni.data.dataset import DATASET_REGISTRY; print(DATASET_REGISTRY.valid_keys())"
python -c "from veomni.data.data_transform import DATA_TRANSFORM_REGISTRY; print(DATA_TRANSFORM_REGISTRY.valid_keys())"
```
