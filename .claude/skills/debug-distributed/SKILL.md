---
name: debug-distributed
description: Comprehensive guide for debugging distributed training issues in VeOmni. Use when encountering hangs, crashes, wrong results, or OOM in multi-GPU/NPU training.
---

# Debug Distributed Training

Systematic approach to debugging distributed training issues in VeOmni.

## When to Use

- Training hangs or deadlocks
- Wrong training results (loss divergence, NaN)
- OOM errors in distributed settings
- Communication failures (NCCL/HCCL errors)

## Step 1: Classify the Issue

| Symptom            | Likely Category       | Start Here                |
| ------------------ | --------------------- | ------------------------- |
| Training hangs     | Collective mismatch   | Check process groups      |
| NCCL error         | Communication failure | Check environment vars    |
| Wrong loss         | Reduction error       | Check SP loss reduction   |
| OOM                | Sharding error        | Check FSDP configuration  |
| NaN loss           | Numerical instability | Check dtype/precision     |
| Crash on init      | Mesh setup error      | Check ParallelState       |

## Step 2: Enable Debugging

```bash
# Verbose distributed logging
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# NCCL debugging (GPU)
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# HCCL debugging (NPU)
export HCCL_LOG_LEVEL=INFO

# VeOmni verbose logging
export VEOMNI_VERBOSITY=DEBUG
```

## Step 3: Common Issues & Fixes

### Hang / Deadlock

**Symptom**: Training stops, all GPUs at 100% or 0%

**Checklist**:
1. All ranks calling same collective operation?
2. All ranks have same tensor shapes for collective?
3. Process group membership matches across ranks?
4. No conditional code that only some ranks execute?

**Debug**:
```python
import torch.distributed as dist
print(f"Rank {dist.get_rank()}: about to call all_reduce")
dist.all_reduce(tensor, group=group)
print(f"Rank {dist.get_rank()}: completed all_reduce")
```

### Wrong Loss Values

**Symptom**: Loss doesn't match single-GPU baseline

**Checklist**:
1. Using `reduce_sequence_parallel_loss` when SP enabled?
2. Using `mean_global_loss` for final loss aggregation?
3. Gradient accumulation step count correct?
4. Loss reduction mode (SUM vs MEAN) consistent?

### OOM Errors

**Symptom**: CUDA/NPU out of memory

**Checklist**:
1. FSDP sharding strategy correct? (`FULL_SHARD` for large models)
2. Activation checkpointing enabled?
3. Micro batch size appropriate for GPU memory?
4. Gradient accumulation instead of larger batch?

### FSDP-Specific Issues

**Symptom**: Errors during FSDP wrapping or training

**Checklist**:
1. FSDP2 requires `meta` device init (check config)
2. Gradient clipping function matches `dp_mode`
3. Weight loading after FSDP wrapping (not before for FSDP2)
4. Mixed precision reduce_dtype matches training dtype

### Sequence Parallel Issues

**Symptom**: Wrong results with SP enabled

**Checklist**:
1. Sync mode: operating on 3D tensors?
2. Async mode: operating on 4D tensors?
3. Loss reduction using `reduce_sequence_parallel_loss`?
4. Collator padding to SP-divisible length?
5. NPU async: using RMSNorm (NOT LayerNorm)?

### MoE Expert Parallel Issues

**Symptom**: Errors in MoE forward/backward

**Checklist**:
1. Token count aligned to group_gemm requirement (8/16/32)?
2. EP mesh from parallel_plan matches expectations?
3. Load balancing loss coefficient non-zero?
4. All-to-all dispatch/combine token counts consistent?

## Step 4: Reproduce with Minimal Config

Use toy configs from `tests/toy_config/` to reproduce with small models:
```bash
torchrun --nproc_per_node=2 tasks/train_text.py \
    --config tests/toy_config/tiny_model.yml \
    --train.train_steps 10
```

## Step 5: Escalation

If the issue is in VeOmni core distributed code, file an issue at:
https://github.com/ByteDance-Seed/VeOmni/issues
