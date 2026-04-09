---
paths:
  - veomni/distributed/**
  - veomni/trainer/**
  - tasks/**
---

# Distributed Code Rules

## Device Abstraction (Critical)

- **Never** use `torch.cuda.*` directly — use `veomni/utils/device.py`
- Use `get_device_type()` for device strings ("cuda", "npu", "cpu")
- Use `get_dist_comm_backend()` for backend ("nccl", "hccl")
- Use `synchronize()` instead of `torch.cuda.synchronize()`

## Parallel State Management

- Access via `get_parallel_state()` from `veomni/distributed/parallel_state.py`
- ParallelState is frozen — never mutate after initialization
- Device mesh dimensions conditionally included (only when d > 1 or name == "dp_shard")

## Process Group Management

- **Never create global process group** in module-level code
- Always pass `process_group` explicitly, don't rely on default
- Use `dist.get_rank(group)` not `dist.get_rank()` when group matters

## FSDP Rules

### FSDP1
- Standard `FullyShardedDataParallel` wrapping
- Gradient clipping via `veomni/distributed/fsdp/clip_grad_norm.py`
- Supports CPU and GPU init

### FSDP2
- Composable `fully_shard` API
- **Requires** `meta` device initialization
- Gradient clipping via `veomni/distributed/fsdp2/clip_grad_norm.py`
- Uses rank0 broadcast for weight loading (more efficient)

## Sequence Parallel Rules

- **Sync mode**: Operates on 3D tensors BEFORE reshape to 4D
- **Async mode**: Reshape to 4D FIRST, then all-to-all
- `reduce_sequence_parallel_loss` takes 2 args (both `torch.Tensor`)
- Async NPU: Supports RMSNorm but NOT LayerNorm

## MoE Expert Parallel Rules

- EP modules must use EP-specific FSDP mesh (from parallel_plan)
- Token alignment: group_gemm requires tokens aligned to 8/16/32
- Load balancing loss must be included in total loss

## Communication Patterns

- **All-reduce**: Must be called by all ranks in the group
- **Broadcast**: Specify `src` rank explicitly
- **Barrier**: Avoid unless necessary (debugging only)
- **All-to-all**: Verify consistent token counts across ranks

## Common Pitfalls

| Issue         | Cause                            | Fix                            |
| ------------- | -------------------------------- | ------------------------------ |
| Hang          | Mismatched collective calls      | Ensure all ranks call same op  |
| Wrong results | Incorrect reduction op           | Check `ReduceOp` (SUM vs MEAN)|
| OOM           | Unsharded tensor on wrong device | Verify sharding and mesh       |
| Wrong loss    | Missing SP loss reduction        | Use `reduce_sequence_parallel_loss` |

## Debugging

- Set `TORCH_DISTRIBUTED_DEBUG=DETAIL` for verbose logging
- Set `NCCL_DEBUG=INFO` for NCCL-level issues
- Set `HCCL_LOG_LEVEL=INFO` for Ascend NPU issues
