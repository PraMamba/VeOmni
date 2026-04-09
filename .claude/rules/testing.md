---
paths:
  - '**/tests/**'
  - '*_test.py'
  - test_*.py
---

# Testing Rules

## Pytest Markers

| Marker                                  | When to Use          |
| --------------------------------------- | -------------------- |
| `@pytest.mark.benchmark`               | Benchmark tests       |
| `@pytest.mark.qwen3_5_ulysses`         | Qwen3.5 Ulysses SP   |
| `@pytest.mark.skipif(cond, reason=...)` | Conditional skip      |
| `@pytest.mark.parametrize(...)`         | Parameterized tests   |

## Test Structure

```python
def test_<what>_<condition>_<expected>():
    """Test that <what> does <expected> when <condition>."""
    # Arrange
    ...
    # Act
    ...
    # Assert
    ...
```

## GPU/NPU Test Constraints

- **Always skip gracefully** when accelerator unavailable:
  ```python
  import torch
  CUDA_AVAILABLE = torch.cuda.is_available()

  @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
  def test_gpu_feature():
      ...
  ```
- Clean up GPU memory: `torch.cuda.empty_cache()` in fixtures
- Use smallest possible model/batch for unit tests

## Distributed Test Patterns

- Use `torch.distributed.fake_pg` for unit tests when possible
- Multi-GPU tests: use `torchrun` in CI, skip locally when insufficient GPUs
- Mock `dist.get_rank()` and `dist.get_world_size()` explicitly
- Don't mock internals of FSDP/DTensor — use integration tests

## Fixtures

- Prefer `tmp_path` over manual temp directories
- Use `monkeypatch` for environment variables
- Scope expensive fixtures appropriately (`session` > `module` > `function`)

## Assertions

- Use `torch.testing.assert_close()` for tensor comparison
- Specify `rtol`/`atol` explicitly for numerical tests
- Avoid bare `assert tensor.equal()` — no useful error message

## Test Locations

```
tests/
├── models/         # Model registry, patch tests
├── parallel/       # Distributed/SP tests
│   ├── ulysses/    # Ulysses SP correctness
│   └── encoder_data_balance/
├── e2e/            # End-to-end training
├── checkpoints/    # DCP save/load
├── data/           # Data pipeline
├── utils/          # Utility functions
├── ops/            # Kernel operations
├── toy_config/     # Minimal test configs
└── testdata/       # Test fixtures
```

## CI Workflows

- `gpu_unit_tests.yml` — GPU unit tests (L20x8)
- `npu_unit_tests.yml` — NPU unit tests (Ascend)
- `gpu_e2e_test.yml` — End-to-end GPU tests
- `check_patchgen.yml` — Validates code generation consistency
- `device_api_check.yml` — No hardcoded "cuda" usage
