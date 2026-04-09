---
paths:
  - veomni/arguments/**
---

# API & Config Rules

## Dataclass Conventions

```python
@dataclass
class XxxConfig:
    """One-line description.

    Attributes:
        field_name: Description with default explained.
    """
    # Required fields first (no default)
    required_field: str

    # Optional fields with defaults
    optional_field: int = 32

    # Internal fields last (underscore prefix)
    _internal: str = field(default="", repr=False)
```

## Field Ordering

1. Required fields (no default)
2. Common optional fields
3. Advanced/rare optional fields
4. Internal fields (`_prefix`)

## Config Hierarchy

```
VeOmniArguments
  ├── model: ModelArguments
  │     └── ops_implementation: OpsImplementationConfig
  ├── data: DataArguments
  │     └── dataloader: DataloaderConfig
  └── train: TrainingArguments
        ├── optimizer: OptimizerConfig
        ├── wandb: WandbConfig
        ├── profile: ProfileConfig
        ├── gradient_checkpointing: GradientCheckpointingConfig
        ├── accelerator: AcceleratorConfig
        │     ├── fsdp_config: FSDPConfig
        │     └── offload_config: OffloadConfig
        └── checkpoint: CheckpointConfig
```

## Validation

- Use `__post_init__` for validation
- Raise `ValueError` with clear message:
  ```python
  def __post_init__(self):
      if self.batch_size <= 0:
          raise ValueError(f"batch_size must be positive, got {self.batch_size}")
  ```
- Avoid side effects in `__post_init__` beyond validation

## CLI Integration

- Parse via `veomni.arguments.parse_args(VeOmniArguments)`
- Supports YAML config + CLI overrides
- Nested fields accessed via dot notation: `--train.optimizer.lr 1e-4`
- Fields exposed to CLI must have clear descriptions

## Backward Compatibility

- **Adding fields**: Add with default value (safe)
- **Removing fields**: Deprecate first, remove in next major version
- **Renaming fields**: Add new field, keep old with deprecation warning
- **Changing types**: Avoid; use Union if necessary

## Config File Format

Location: `configs/` (YAML)

```yaml
# configs/text/qwen3.yml
model:
  model_path: "path/to/model"
  ops_implementation:
    attn_implementation: "flash_attention_2"
data:
  data_type: "conversation"
  max_seq_len: 4096
train:
  num_train_epochs: 1
  global_batch_size: 32
  micro_batch_size: 4
  optimizer:
    type: "adamw"
    lr: 1e-5
```
