# Code Style Rules

Rules beyond pre-commit (ruff format/lint).

## Design Patterns

- **Registry-based extensibility**: Use `Registry` from `veomni/utils/registry.py`
  for all plugin points (models, datasets, transforms, etc.)
- **Composition over inheritance**: Trainers compose `BaseTrainer` via
  `__new__`, don't subclass it
- **Device abstraction**: Never hardcode `torch.cuda.*` — use `veomni/utils/device.py`
  - Good: `get_device_type()`, `get_torch_device()`, `synchronize()`
  - Avoid: `torch.cuda.synchronize()`, `torch.cuda.current_device()`

## Logging

- Use `veomni.utils.logging.get_logger(__name__)`, NOT `print` or stdlib `logging`
- For rank-0 only: `logger.info_rank0(msg)`, `logger.warning_rank0(msg)`
- For one-time messages: `logger.info_once(msg)`
- Log levels:
  - DEBUG: Detailed tracing (avoid in hot paths)
  - INFO: Milestones (training start, checkpoint saved)
  - WARNING: Recoverable issues
  - ERROR: Failures requiring attention
- Verbosity controlled via `VEOMNI_VERBOSITY` env var

## Performance Patterns

- **Avoid GPU-CPU sync**: `.item()`, `.tolist()`, `print(tensor)` cause sync
- **Prefer batch operations**: Avoid Python loops over tensor elements
- **In-place ops**: Use when safe, but careful with autograd (`.add_()` vs `+`)

## Naming Conventions

| Type              | Pattern              | Example                                  |
| ----------------- | -------------------- | ---------------------------------------- |
| Config dataclass  | `XxxConfig`          | `FSDPConfig`, `OptimizerConfig`          |
| Trainer class     | `XxxTrainer`         | `TextTrainer`, `VLMTrainer`              |
| Registry instance | `XXX_REGISTRY`       | `MODELING_REGISTRY`, `DATASET_REGISTRY`  |
| Build function    | `build_xxx()`        | `build_dataset()`, `build_optimizer()`   |
| Callback class    | `XxxCallback`        | `CheckpointerCallback`, `TqdmCallback`  |
| Patch function    | `apply_xxx_patch()`  | `apply_veomni_fused_moe_patch()`         |

## Tensor Conventions

- Shape convention: `[batch, seq_len, hidden]` or document clearly
- Use `torch.Size` assertions for shape validation in debug
- Prefer explicit dtype/device over implicit conversion

## Import Style

- Group: stdlib, third-party, veomni (ruff handles order)
- Avoid `from x import *` (except in `__init__.py`)
- Prefer explicit imports over module-level imports for large modules
- Heavy optional deps inside functions (e.g., `flash_attn`, `triton`)
