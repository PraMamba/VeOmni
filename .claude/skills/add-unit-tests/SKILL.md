---
name: add-unit-tests
description: Guide for adding unit tests to VeOmni. Use when user wants to add tests for models, parallel code, data pipeline, or utilities.
---

# Add Unit Tests to VeOmni

Add comprehensive unit tests following VeOmni testing conventions.

## When to Use

- User asks "how do I add tests?"
- After implementing a new feature
- When improving test coverage

## Test Directory Structure

```
tests/
├── models/         # Model registration, patching
├── parallel/       # Distributed/SP/EP tests
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

## Test Template

```python
import pytest
import torch

# Skip if GPU unavailable
CUDA_AVAILABLE = torch.cuda.is_available()


class TestMyFeature:
    """Tests for my feature."""

    def test_basic_functionality(self):
        """Test basic case works correctly."""
        # Arrange
        input_data = ...

        # Act
        result = my_function(input_data)

        # Assert
        assert result.shape == expected_shape
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_functionality(self):
        """Test GPU-specific behavior."""
        ...

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_multiple_dtypes(self, dtype):
        """Test across data types."""
        ...
```

## Distributed Test Template

```python
import torch.distributed as dist

def run_test_distributed(rank, world_size):
    """Test function run on each rank."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Test logic here
    ...

    dist.destroy_process_group()

@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Need at least 2 GPUs"
)
def test_distributed_feature():
    """Test distributed feature with multiple GPUs."""
    world_size = 2
    torch.multiprocessing.spawn(
        run_test_distributed,
        args=(world_size,),
        nprocs=world_size,
    )
```

## Best Practices

1. **Naming**: `test_<what>_<condition>_<expected>()`
2. **Assertions**: Use `torch.testing.assert_close()` with explicit rtol/atol
3. **Fixtures**: Use `tmp_path` for temp dirs, `monkeypatch` for env vars
4. **GPU tests**: Always `@pytest.mark.skipif` when GPU needed
5. **Smallest model**: Use tiny configs from `tests/toy_config/`
6. **Clean up**: `torch.cuda.empty_cache()` in teardown for GPU tests

## Running Tests

```bash
source .venv/bin/activate

# All tests
pytest tests/ -v

# Specific module
pytest tests/models/ -v
pytest tests/parallel/ulysses/ -v

# With markers
pytest tests/ -m "not benchmark" -v

# Single test
pytest tests/models/test_model_registry.py::test_specific -v
```

## CI Integration

Tests are run in CI via `.github/workflows/`:
- `gpu_unit_tests.yml`: Runs on L20x8
- `npu_unit_tests.yml`: Runs on Ascend
- Conditional execution based on changed paths

To add a test to CI, ensure it's in the correct `tests/` subdirectory
and CI will pick it up automatically.
