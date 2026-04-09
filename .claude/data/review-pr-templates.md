# PR Review: Task Templates Reference

Referenced by `.claude/commands/review-pr.md`

---

## FSDP Tasks [Opus]

**Task: FSDP/FSDP2 Parallelization Correctness**

```
Checklist:
- Shard/reshard operation timing and correctness
- FSDP1 vs FSDP2 API usage (no mixing)
- Mixed precision (param_dtype vs reduce_dtype) consistency
- Device mesh dimension matching with ParallelState
- Gradient clipping using correct dp_mode-specific function
- Rank0 broadcast weight loading (FSDP2 only)
```

**Task: FSDP State Management [Sonnet]**

```
Checklist:
- state_dict save/load sharded vs full mode
- DCP checkpointer integration correctness
- Checkpoint resumption logic
- Optimizer state handling
```

## Sequence Parallel Tasks [Opus]

**Task: Ulysses SP Correctness**

```
Checklist:
- Sync mode: 3D tensor operations BEFORE 4D reshape
- Async mode: 4D reshape FIRST, then all-to-all
- reduce_sequence_parallel_loss takes exactly 2 tensor args
- SequenceParallelCollator padding and slicing correctness
- Flash attention kwargs (cu_seq_lens) pre-computed in collator
- NPU async mode: only RMSNorm (NOT LayerNorm)
```

## MoE Tasks [Opus]

**Task: Expert Parallel Correctness**

```
Checklist:
- EP mesh configuration via parallel_plan.py
- Token alignment for group_gemm (8/16/32)
- All-to-all dispatch/combine consistency
- Load balancing loss inclusion in total loss
- MoE implementation mode (eager/fused/fused_quack) correctness
```

## Trainer Tasks [Opus]

**Task: Trainer Core Logic**

```
Checklist:
- BaseTrainer.__new__ initialization completeness
- All required attributes set before train()
- Callback dispatch in all 6 on_* methods
- Loss computation with mean_global_loss
- Gradient accumulation step counting
- DataLoader exhaustion handling (StopIteration)
```

## Model Tasks [Sonnet]

**Task: Model Registration and Loading**

```
Checklist:
- MODELING_REGISTRY registration with correct model_type key
- MODEL_CONFIG_REGISTRY if custom config needed
- MODEL_PROCESSOR_REGISTRY if custom processor needed
- __init__.py imports trigger registration
- GPU/NPU patch files present and applied
- Patchgen config if generated patches used
```

---

## General Review Templates

### Logic and Boundary Conditions [Opus]

```
Applicable: Any non-doc/config changes
Checklist:
- Conditional logic errors (boundary conditions, short-circuit)
- Loop errors (off-by-one, infinite loops, early exit)
- Missing null/None/empty handling
- Exception handling (swallowed exceptions, wrong type)
- Return value errors (missing return, inconsistent paths)
```

### Tensor Shape and Data Type [Opus]

```
Applicable: TENSOR_OPS type detected
Checklist:
- Tensor shape mismatch (broadcast errors, wrong dimensions)
- Batch dimension handling (missing dim, wrong order)
- Sequence length and padding handling
- dtype mismatch (fp16/fp32/bf16 mixing)
- Device placement errors (CPU/GPU mixed)
- Gradient issues (missing detach, missing no_grad)
- view/reshape contiguity requirements
```

### Communication and Synchronization [Sonnet]

```
Applicable: DISTRIBUTED_COMM type detected
Checklist:
- Process group usage correctness
- Device mesh configuration
- Unnecessary GPU-CPU sync (.item(), .tolist())
- Collective communication order dependencies
- Error state synchronization across ranks
```

### Configuration Validation [Sonnet]

```
Applicable: API_CONFIG type detected
Checklist:
- New fields have defaults (backward compatible)
- __post_init__ validates constraints
- No side effects beyond validation in __post_init__
- CLI integration (dot-notation access works)
- YAML config compatibility
```

### Data Pipeline [Sonnet]

```
Applicable: DATA_PIPELINE type detected
Checklist:
- Registry registration correctness
- Transform function signature matches expected
- Collator DataCollateInfo per-key config
- Dynamic batching token counting
- Multimodal preprocessing error handling
```

### Performance [Sonnet]

```
Applicable: Any non-doc changes
Checklist:
- Unnecessary GPU-CPU sync (.item(), .tolist(), print(tensor))
- Memory allocation pattern changes (potential OOM)
- Communication volume increase
- torch.compile compatibility
- Unnecessary tensor copies
```

### Device Abstraction [Haiku]

```
Applicable: Any Python file changes
Checklist:
- No hardcoded torch.cuda.* (use veomni/utils/device.py)
- Communication backend via get_dist_comm_backend()
- Device type via get_device_type()
```

### Import and Dependencies [Haiku]

```
Applicable: Any Python file changes
Checklist:
- No wildcard imports (from x import *)
- Correct import grouping (stdlib/third-party/veomni)
- Heavy optional deps inside functions
- Circular import risks
```
