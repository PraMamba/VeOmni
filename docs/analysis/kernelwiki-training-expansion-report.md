# KernelWiki Training Library Analysis Report: VeOmni

**Framework**: VeOmni (ByteDance Seed Team)  
**Source Path**: `/root/VeOmni/.worktrees/source_code_analysis`  
**Analysis Date**: 2026-05-28  
**Library Type**: **Training-Orchestration Framework**  
**Dimension Emphasis Rationale**: Zero CUDA kernels detected in Dimension 1 → classified as orchestration framework → Dimensions 3 (Parallelism) and 4 (Memory) received deep treatment; Dimension 1 was replaced with dependency graph analysis.

---

## Dimension 1: Compute Kernels

**Classification: VeOmni is an orchestration framework with ZERO CUDA kernels (.cu/.cuh/.ptx), ZERO C++ extension files (.cpp), and 11 Triton kernel files containing 34 distinct `@triton.jit` decorated functions.**

### Kernel File Census

| File Type | Count | Location |
|-----------|-------|----------|
| `.cu` / `.cuh` / `.ptx` (CUDA) | 0 | N/A |
| `.cpp` (C++ extension) | 0 | N/A |
| `.py` with `@triton.jit` (in `veomni/ops/`) | 10 | `veomni/ops/` subdirectories |
| `.py` with `@triton.jit` (in `scripts/`) | 1 | `scripts/deepseek_v3/kernel.py` (utility, not runtime) |
| Total `@triton.jit` functions | 34 | Across all 11 files |

### Training-Specific Kernels

| Kernel | File Path | Proposed Tag | Description |
|--------|-----------|--------------|-------------|
| `_lb_loss_fwd_kernel` | `ops/kernels/load_balancing_loss/triton.py:L36` | `moe-lb-loss-fused` | Fused softmax + top-k selection + expert count/prob accumulation for MoE auxiliary loss |
| `_lb_loss_bwd_kernel` | `ops/kernels/load_balancing_loss/triton.py:L110` | `moe-lb-loss-fused` | Backward: recomputes softmax, computes d(loss)/d(gate_logits) |
| `rotary_interleaved_kernel` | `ops/kernels/rotary/triton_wan.py:L28` | `rope-interleaved` | Interleaved RoPE with backward via conjugate trick (autograd Function) |
| `group_gemm_same_nk_kernel` | `ops/kernels/moe/_kernels/kernel/group_gemm.py:L65` | `moe-group-gemm` | Grouped GEMM with fused activation (GELU/SiLU), used in MoE fwd/bwd |
| `group_gemm_same_mn_kernel` | `ops/kernels/moe/_kernels/kernel/group_gemm.py:L251` | `moe-group-gemm` | Grouped GEMM for wgrad path in MoE |

### Inference-Shared Kernels (with training behavior differences)

| Kernel | File Path | Training Behavior Difference |
|--------|-----------|------------------------------|
| `matmul_kernel_persistent` | `ops/batch_invariant_ops/batch_invariant_ops.py:L41` | Persistent GEMM ensuring batch-invariant determinism (DeepSeek V3) |
| `_rms_norm_kernel` | `ops/kernels/rms_norm/triton_batch_invariant.py:L26` | Forward in Triton, backward falls back to PyTorch eager |
| `_bmm_kernel` | `ops/kernels/rotary/triton_deterministic.py:L26` | Batched GEMM with int64 offset support for deterministic RoPE |
| `_moe_*_kernel` (5 kernels) | `ops/kernels/moe/_kernels/kernel/moe.py` | histogram, gather, add_gather, scatter, index_compute for MoE token routing |

### Fused Kernels

| Kernel | Operations Fused | Estimated Memory Savings |
|--------|-----------------|------------------------|
| `_lb_loss_fwd_kernel` | softmax + top-k selection + expert count accumulation | Eliminates `[N, top_k, num_experts]` one-hot tensor |
| `group_gemm_same_nk_kernel` | GEMM + optional activation (GELU/SiLU) + optional save-activation | Saves one read/write pass for activation materialization |
| `matmul_kernel_persistent` | GEMM + optional bias add | Eliminates separate bias-add pass |
| `_moe_add_gather_kernel` | Element-wise add + gather (top-k reduction) | Eliminates intermediate `[M*topk, N]` sum tensor |

### Kernel Dependency Graph

VeOmni's actual GPU compute is overwhelmingly delegated to upstream kernel libraries.

| Provider Library | Kernel Types Provided | Import Evidence |
|-----------------|----------------------|-----------------|
| **Flash Attention 2** (`flash_attn`) | Self-attention (fwd+bwd), varlen attention | `ops/kernels/attention/__init__.py:L61` |
| **Flash Attention 3** (`flash_attn_interface`) | Self-attention (fwd+bwd), Hopper optimized | `ops/kernels/attention/__init__.py:L69` |
| **Flash Attention 4** (`flash_attn.cute`) | Self-attention (fwd+bwd), CuTe-based | `ops/kernels/attention/__init__.py:L77` |
| **Liger Kernel** (`liger_kernel`) | Fused linear cross-entropy, RMSNorm, RoPE, SwiGLU MLP | `ops/liger/__init__.py`, `ops/kernels/cross_entropy/liger.py` |
| **Quack GEMM** (`quack`) | CUTLASS/CuTe grouped GEMM (SM90+), for MoE fwd/dgrad/wgrad | `ops/kernels/moe/quack_gemm.py:L28` |
| **Flash Linear Attention** (`fla`) | Gated delta rule (chunk), causal conv1d, fused RMSNorm+gated | `ops/kernels/gated_delta_rule/__init__.py:L83,L107,L128` |
| **FlashQLA** (`flash_qla`) | Chunk gated delta rule (SM90-only alternative to FLA) | `ops/kernels/gated_delta_rule/__init__.py:L154` |
| **PyTorch SDPA** | Scaled dot-product attention fallback (via HuggingFace) | `ops/kernels/attention/__init__.py:L19` |
| **torch_npu** (Ascend NPU) | NPU-specific RMSNorm, RoPE, group GEMM, conv1d | 12 files in `ops/` subdirectories |

### Kernel Dispatch Architecture

VeOmni uses a `KERNEL_REGISTRY` (`ops/kernel_registry.py`) with `OpSlot` dispatch. Supported backends per operation: `"eager"`, `"liger_kernel"`, `"triton"`, `"quack"`, `"fla"`, `"flash_qla"`, `"npu"`, `"chunk_loss"`. Hardware requirements (device_type, compute_capability) are enforced at resolve time via `OpsImplementationConfig`.

### Proposed New kernel_types

| Tag | Representative File | Description |
|-----|---------------------|-------------|
| `batch-invariant-gemm` | `batch_invariant_ops/batch_invariant_ops.py` | Persistent matmul ensuring identical results regardless of batch composition |
| `moe-lb-loss-fused` | `load_balancing_loss/triton.py` | Fused softmax + top-k + accumulation for MoE load-balancing auxiliary loss |
| `moe-token-routing` | `moe/_kernels/kernel/moe.py` | Expert histogram, scatter, gather, and index computation for MoE token routing |
| `moe-group-gemm` | `moe/_kernels/kernel/group_gemm.py` | Grouped GEMM with per-expert batching, fused activation, and pretuned heuristics |
| `rope-interleaved` | `rotary/triton_wan.py` | Interleaved rotary positional embedding with varlen support |
| `rms-norm-batch-invariant` | `rms_norm/triton_batch_invariant.py` | Deterministic RMSNorm ensuring batch-invariant outputs |

---

## Dimension 2: Communication Kernels and Strategies

### Executive Summary

VeOmni contains zero custom CUDA/NCCL communication kernels. All communication is delegated to `torch.distributed` collective APIs (NCCL on GPU, HCCL on NPU). VeOmni's value lies in how it composes three orthogonal communication dimensions -- FSDP2 (all-gather/reduce-scatter), Ulysses SP (all-to-all), and Expert Parallel (all-to-all) -- with a unified DeviceMesh, and in its async communication-compute overlap strategy.

### Collective Operations

| Operation | PyTorch API | Purpose | Communication Dimension | File |
|-----------|------------|---------|------------------------|------|
| all-to-all (chunked) | `dist.all_to_all()` | SP: scatter heads, gather sequence (sync Ulysses) | Ulysses SP group | `distributed/sequence_parallel/ulysses.py:75` |
| all-to-all-single | `dist.all_to_all_single()` | SP: fused single-buffer variant | Ulysses SP group | `distributed/sequence_parallel/ulysses.py:107` |
| all-to-all-single (MoE) | `dist.all_to_all_single()` | EP: permute tokens across expert ranks | EP group | `distributed/moe/comm.py:38` |
| all-to-all-single (MoE async) | `dist.all_to_all_single(async_op=True)` | EP: async token dispatch with handle | EP group | `distributed/moe/comm.py:75` |
| all-to-all (VLM images) | `dist.all_to_all()` | SP: rebalance image embeddings across SP ranks | Ulysses SP group | `distributed/sequence_parallel/ulysses.py:309,315` |
| all-gather (variable-size) | `dist.all_gather()` | SP: gather variable-length tensors | Ulysses SP group | `distributed/sequence_parallel/ulysses.py:44,46` |
| all-gather-into-tensor | `dist.all_gather_into_tensor()` | SP: gather fixed-size sliced tensor backward | Ulysses SP group | `distributed/sequence_parallel/ulysses.py:60` |
| all-gather-into-tensor | `dist.all_gather_into_tensor()` | EP: gather per-expert token counts | EP group | `distributed/moe/moe_layer.py:51` |
| all-reduce (SUM) | `dist.all_reduce()` | SP loss: aggregate loss and valid-token counts | Unified SP group | `distributed/sequence_parallel/loss.py:35-36` |
| all-reduce (SUM/MAX) | `dist.all_reduce()` | Grad norm: sum/max across FSDP/EP groups | FSDP/EP groups | `distributed/fsdp2/clip_grad_norm.py:230-238` |
| reduce-scatter | (delegated to FSDP2 runtime) | FSDP2: gradient accumulation across dp_shard | dp_shard mesh | via `fully_shard()` |
| all-gather | (delegated to FSDP2 runtime) | FSDP2: unshard parameters before forward/backward | dp_shard mesh | via `fully_shard()` |
| broadcast | `dist.broadcast()` | Weight loading: rank-0 broadcasts shards | World group | `models/module_utils.py:435,635` |
| P2P isend/irecv | `dist.batch_isend_irecv()` | SP: distributed `torch.roll` for shift operations | Ulysses SP group | `distributed/sequence_parallel/ulysses.py:354` |

### Communication-Compute Overlap Patterns

| Pattern | Mechanism | Evidence |
|---------|-----------|----------|
| **Async Ulysses QKV Projection (forward)** | Launch all-to-all for Q asynchronously, compute K projection during Q comm, launch K comm, compute V projection during K comm, launch V comm, collect Q, normalize Q/K during V comm, collect K, collect V | `async_ulysses.py:97-179` |
| **Async Ulysses QKV Projection (backward)** | Launch V grad all-to-all, compute Q/K norm backward during V comm, collect V grad, launch K grad comm, compute V projection grad during K comm, collect K grad, launch Q grad comm, compute K projection grad during Q comm, collect Q grad | `async_ulysses.py:258-384` |
| **Async Ulysses Output Projection (backward)** | Launch output grad all-to-all, compute weight grad during output comm, collect output grad | `async_ulysses.py:483-493` |
| **FSDP2 Forward Prefetch** | Current decoder block prefetches next block's FSDP modules (all-gather); order: attn, gate, experts | `torch_parallelize.py:350-357` |
| **FSDP2 Backward Prefetch** | Current decoder block prefetches previous block's FSDP modules (all-gather for recomputation) | `torch_parallelize.py:359-365` |
| **MoE all-to-all async variant** | `_AllToAll_Async` returns `(output, async_handle)` for explicit wait-later pattern | `distributed/moe/comm.py:57-83` |
| **Checkpoint all-gather avoidance** | Custom `CheckpointFunction` patches FSDP1's backward to skip redundant all-gather | `distributed/checkpoint.py:60-68,81-86` |

### Advanced Communication Features Checklist

- [ ] Symmetric memory support (NCCL 2.27+): **No** -- no references found
- [ ] Device API support (NCCL 2.28+): LSA **No**, Multimem **No**, GIN **No**
- [ ] Copy Engine zero-SM collectives (NCCL 2.28+): **No**
- [ ] NCCL Inspector integration: **No**
- [ ] PyTorch SymmetricMemory: **No**
- [ ] Alternative backend support (MSCCL++): **No**
- [x] Gloo CPU groups for SP metadata: **Yes** -- `sequence_parallel/comm.py:288`
- [x] NCCL/HCCL backend abstraction: **Yes** -- `get_dist_comm_backend()` returns "nccl" (GPU) or "hccl" (NPU)
- [x] HCCL PREMUL_SUM workaround: **Yes** -- `ops/platform/npu/hccl_premul_sum.py:30-35`

### Communication Topology: How FSDP2 + SP + EP Compose

VeOmni constructs a single unified DeviceMesh with dimensions `[pp, dp_replicate, dp_shard, ulysses, cp, tp]` (only dimensions with size > 1 are materialized, except `dp_shard` which is always present). Flattened sub-meshes:

- **`dp`** = `[dp_replicate, dp_shard]` -- for data loading
- **`dp_shard_sp`** = `[dp_shard, ulysses, cp]` -- for FSDP2 parameter sharding (SP ranks share parameters)
- **`dp_sp`** = `[dp_replicate, dp_shard, ulysses, cp]` -- for loss all-reduce
- **`sp`** = `[ulysses, cp]` -- for sequence-parallel communications

EP uses a separate 2D DeviceMesh `[ep, ep_fsdp]` with an independent rank layout.

The communication flow for a single MoE decoder block:
1. FSDP2 all-gathers attention parameters on `dp_shard_sp` mesh
2. SP all-to-all exchanges Q/K/V across `ulysses` group (async if enabled)
3. Flash attention computes locally
4. SP all-to-all gathers output, scatters heads across `ulysses` group
5. FSDP2 all-gathers MoE gate parameters
6. MoE router computes token-to-expert assignment locally
7. EP all-gather collects per-expert token counts across `ep` group
8. EP all-to-all dispatches tokens to expert-owning ranks
9. Group GEMM computes expert FFN
10. EP all-to-all returns expert outputs to originating ranks
11. FSDP2 reduce-scatter accumulates gradients (in backward) on respective meshes

### Proposed New Communication kernel_types

| Tag | File | Description |
|-----|------|-------------|
| `sp-all-to-all-sync` | `distributed/sequence_parallel/ulysses.py` | Synchronous Ulysses all-to-all: scatter heads, gather sequence |
| `sp-all-to-all-async` | `distributed/sequence_parallel/async_ulysses.py` | Async Ulysses all-to-all with compute overlap |
| `sp-all-reduce-loss` | `distributed/sequence_parallel/loss.py` | All-reduce of per-rank loss and valid-token counts for SP loss normalization |
| `sp-p2p-roll` | `distributed/sequence_parallel/ulysses.py` | Distributed `torch.roll` using P2P isend/irecv |
| `ep-all-to-all-sync` | `distributed/moe/comm.py` | Synchronous MoE token dispatch/combine via all_to_all_single |
| `ep-all-to-all-async` | `distributed/moe/comm.py` | Async MoE token dispatch returning handle for later wait |
| `fsdp2-grad-norm-allreduce` | `distributed/fsdp2/clip_grad_norm.py` | Multi-group all-reduce for EP-aware gradient norm computation |

### Proposed New Communication techniques

| Tag | Evidence | Description |
|-----|----------|-------------|
| `async-qkv-comm-compute-overlap` | `async_ulysses.py:97-179` | 3-stage overlap: launch Q comm, compute K projection, launch K comm, compute V projection, etc. |
| `fsdp2-manual-prefetch-for-moe` | `torch_parallelize.py:346-365` | Manual forward/backward prefetch ordering when ExtraParallel is present |
| `ep-token-permute-all2all-unpermute` | `distributed/moe/moe_layer.py:72-137` | Three-phase MoE: local permute → all-to-all dispatch → expert GEMMs → reverse pipeline |
| `multi-group-grad-norm` | `distributed/fsdp2/clip_grad_norm.py:74-153` | Cascaded all-reduce across FSDP, ep_fsdp, and ep groups for unified gradient clipping |
| `distributed-roll-via-p2p` | `distributed/sequence_parallel/ulysses.py:327-401` | Distributed `torch.roll` using P2P instead of full all-to-all |
| `root-auto-no-reshard` | `torch_parallelize.py:334-344` | Root FSDP2 module omits explicit reshard to keep lm_head unsharded for fused-linear backward |

---

## Dimension 3: Parallelism Strategies

### Supported Parallelism Dimensions

| Dimension | Config Field | Supported | Status | Implementation File | Communication Pattern |
|-----------|-------------|-----------|--------|--------------------|-----------------------|
| Data Parallel -- Shard (FSDP2) | `dp_shard_size` | **Yes** | Production | `distributed/torch_parallelize.py`, `distributed/fsdp2/` | AllGather (fwd), ReduceScatter (bwd) |
| Data Parallel -- Replicate (HSDP) | `dp_replicate_size` | **Yes** | Production | `distributed/parallel_state.py` | AllReduce over replicate group |
| Ulysses Sequence Parallel | `ulysses_size` | **Yes** | Production (sync + async) | `distributed/sequence_parallel/ulysses.py`, `async_ulysses.py` | All-to-all (scatter heads, gather seq) |
| Context Parallel (Ring Attention) | `cp_size` | Declared | `NotImplementedError` | `distributed/sequence_parallel/comm.py` (group plumbing exists) | Would be P2P ring send/recv |
| Expert Parallel (EP) | `ep_size` | **Yes** | Production | `distributed/parallel_plan.py`, `distributed/moe/` | All-to-all (token dispatch + combine) |
| Extra Parallel (generic) | `extra_parallel_sizes/names` | **Yes** | Production | `distributed/parallel_plan.py` | All-to-all + custom FSDP mesh |
| Tensor Parallel (TP) | `tp_size` | Declared | `assert tp_size == 1` | Mesh dim + `parallelize_module()` call exist; models have `_tp_plan` | Would be column/row parallel |
| Pipeline Parallel (PP) | `pp_size` | Declared | `assert pp_size == 1` | Mesh dim declared but blocked | Would be micro-batch scheduling |
| DDP (legacy) | `fsdp_mode="ddp"` | **Yes** | Fallback | `torch_parallelize.py:478` | AllReduce |

### DeviceMesh Topology

Base mesh construction order (outermost to innermost):
```
[pp, dp_replicate, dp_shard, ulysses, cp, tp]
```

Dimensions with size==1 are **elided**, **except** `dp_shard` which is always present (`parallel_state.py:542`).

Flattened composite meshes:

| Composite Mesh | Component Dims | Purpose |
|----------------|----------------|---------|
| `dp` | `(dp_replicate, dp_shard)` | Data loading / batch distribution |
| `dp_shard_sp` | `(dp_shard, ulysses, cp)` | FSDP parameter sharding (includes SP ranks) |
| `dp_sp` | `(dp_replicate, dp_shard, ulysses, cp)` | Loss all-reduce / FSDP grad reduce |
| `sp` | `(ulysses, cp)` | Sequence parallel communication |

**Separate EP mesh**: Expert Parallel uses an independent 2D `DeviceMesh` with dims `(ep, ep_fsdp)`, NOT embedded in the main mesh. Layout depends on `ep_outside`: interleaved (default) or contiguous.

### Pipeline Scheduling Strategies

| Schedule | Status | Evidence |
|----------|--------|----------|
| 1F1B | NOT IMPLEMENTED | No code found |
| Interleaved 1F1B | NOT IMPLEMENTED | No code found |
| Zero Bubble | NOT IMPLEMENTED | No code found |

Pipeline parallelism is fully gated: `assert acc.pp_size == 1, "Pipeline parallel size not supported yet."` (`arguments_types.py:562`)

### Parallelization Flow (Config → Wrapped Model)

Entry point: `build_parallelize_model()` in `torch_parallelize.py:419`.

1. **Gradient Checkpointing** (line 445): `model.gradient_checkpointing_enable()` with reentrant/non-reentrant option
2. **Tensor Parallel** (line 458): If `tp_enabled`, calls `parallelize_module()` — currently blocked by assertion
3. **Data Parallel dispatch** (line 465):
   - `dp_mode == "fsdp2"` → `parallelize_model_fsdp2()`
   - `dp_mode == "ddp"` → `DistributedDataParallel`

Inside `parallelize_model_fsdp2()` (line 76):
1. **ExtraParallel Application** (line 129): DTensor `redistribute()` slices expert/embed weights along dim-0
2. **FSDP kwargs preparation** (line 196): Builds `MixedPrecisionPolicy`, sets `reshard_after_forward`, optional `CPUOffloadPolicy`
3. **EP-specific FSDP mesh** (line 234): Each ExtraParallel gets `{para}_fsdp` sub-mesh with `Shard(1)` (hidden dim) by default, or `Shard(0)` if `muon_expert_zero_comm=True`
4. **Bottom-up `fully_shard()` wrapping** (line 289): EP modules → MP-ignored modules → layer itself → root
5. **Root shard** (line 343): WITHOUT explicit `reshard_after_forward` (auto-no-reshard for fused-linear kernels)
6. **Manual prefetching** (line 347): Forward and backward prefetch chains across decoder layers
7. **Weight loading** (line 371): `rank0_load_and_broadcast_weights()` or per-rank loading
8. **Grad norm clipping** (line 412): Registers EP-aware `clip_grad_norm`

### FSDP2 Configuration Details

| Field | Default | Description |
|-------|---------|-------------|
| `fsdp_mode` | `"fsdp2"` | Mode selector (ddp or fsdp2) |
| `reshard_after_forward` | `True` | Re-shard parameters after forward (ZeRO-3 style) |
| `reshard_after_backward` | `True` | Re-shard after backward |
| `forward_prefetch` | `True` | Enable forward prefetch of next layer's params |
| `offload` | `False` | CPU offload for params/grads/optimizer states |
| `max_load_broadcast_size` | `20.0` GB | Chunking threshold for rank0 broadcast loading |

Key design decisions:
- Uses PyTorch composable `fully_shard()` API (not legacy wrapper)
- SP dimensions included inside FSDP mesh (`include_sp_in_fsdp=True`; decoupled SP raises `NotImplementedError`)
- Root module auto-no-reshard for fused-linear backward kernels
- ExtraParallel modules use separate 2D FSDP mesh with different shard dimensions

### Sequence Parallel Details (Ulysses SP)

| Mode | File | Tensor Shape | Overlap Strategy | NPU Support |
|------|------|-------------|------------------|-------------|
| **Sync** | `ulysses.py` | 3D: `[B, seq, heads*head_dim]` → scatter heads, gather seq on 3D BEFORE reshape to 4D | None (blocking) | Full |
| **Async** | `async_ulysses.py` | 4D: `[B, seq, heads, head_dim]` → reshape FIRST, then async all-to-all | Q/K/V projection overlapped with communication | RMSNorm only (LayerNorm NOT supported on NPU) |

Composition with FSDP2: SP dimensions are flattened INTO the FSDP mesh as `dp_shard_sp = (dp_shard, ulysses, cp)`.

### Expert Parallel Details

- EP implemented via generic "ExtraParallel" abstraction (supports EP, Embed Parallel, and future extensions)
- Token dispatch flow: `preprocess()` → `token_pre_all2all()` → expert GEMMs → `tokens_post_all2all()`
- Two GEMM strategies: `EPGroupGemm` (separate gate/up) and `EPMergedFc1GroupGemm` (merged fc1_1_2)
- `muon_expert_zero_comm`: switches to `Shard(0)` so each FSDP rank holds complete experts, enabling Muon optimizer without cross-rank communication

### Composable Parallelism

| Composition | Status |
|-------------|--------|
| FSDP + Ulysses SP | Production |
| HSDP + Ulysses SP | Production |
| FSDP + EP | Production |
| FSDP + EP + Ulysses SP | Production |
| FSDP + EP + Embed Parallel | Production (tested) |
| FSDP + TP | Not Implemented |
| Any + PP | Not Implemented |

### Proposed New Parallelism techniques

| Tag | Evidence | Description |
|-----|----------|-------------|
| `composable-fsdp2-sp-ep` | `distributed/parallel_state.py`, `torch_parallelize.py` | Orthogonal composition of FSDP2 + Ulysses SP + EP via separate DeviceMeshes |
| `muon-zero-comm-ep` | `torch_parallelize.py` (muon_expert_zero_comm) | Shard(0) expert placement for zero-communication Muon optimizer |
| `extra-parallel-abstraction` | `distributed/parallel_plan.py` | Generic ExtraParallel framework: pluggable parallelism dimensions beyond EP |
| `async-ulysses-overlap` | `distributed/sequence_parallel/async_ulysses.py` | 3-stage Q/K/V all-to-all compute-communication overlap |
| `sp-inside-fsdp` | `distributed/parallel_state.py` | SP ranks treated as additional FSDP shard ranks via flattened dp_shard_sp mesh |

---

## Dimension 4: Memory Management

### Memory Component Analysis

| Component | Storage Format | Sharding Strategy | Communication Triggered |
|-----------|---------------|-------------------|------------------------|
| Model Parameters | bf16 (param_dtype) | FSDP2 `fully_shard` per `_no_split_modules` | AllGather (fwd), ReduceScatter (bwd) |
| Gradients | fp32 (reduce_dtype) | Reduce-scattered across `dp_shard` mesh | ReduceScatter via FSDP2 |
| Optimizer States (AdamW) | fp32 (exp_avg + exp_avg_sq) | Sharded same as params via FSDP2 DTensor | None (local update) |
| Optimizer States (AnyPrecisionAdamW) | bf16 momentum + bf16 variance + optional bf16 compensation | Sharded via FSDP2 DTensor | None (local update) |
| Optimizer States (Muon) | 1x param momentum (no variance) | Sharded via FSDP2 DTensor | AllGather for Newton-Schulz |
| Activations | bf16 (autocast) | Local per-rank; optionally offloaded to CPU | None (local); PCIe H2D if offloaded |
| EP Expert Params | bf16 | Sliced dim-0 across EP, then FSDP2 | AllGather within ep_fsdp, AllToAll for tokens |

### Activation Checkpointing Strategies

| Strategy | Description | Memory-Compute Tradeoff | Evidence |
|----------|-------------|------------------------|----------|
| Full Layer Checkpointing (default) | HF `gradient_checkpointing_enable()` applied to all `_no_split_modules` | ~60-70% activation memory saved; 33% compute overhead | `torch_parallelize.py:451` |
| Non-reentrant (default) | `use_reentrant=False`, compatible with FSDP2 | Better hook compatibility | `arguments_types.py:234` |
| Selective Recomputation | `context_fn` parameter plumbed through (default `noop_context_fn`) | Configurable fine-grained tradeoff | `torch_parallelize.py:454` |
| Activation Offloading to CPU | `custom_save_on_cpu` with `saved_tensors_hooks`, GPU budget via `activation_gpu_limit` | Near-zero GPU activation memory; PCIe overhead | `offloading.py:32-70` |
| GC + Offload Dual Context | Forward uses `gpu_limit=0`, backward uses configured limit | Maximum savings with controlled GPU budget | `offloading.py:82-85` |

### Gradient Accumulation

- Formula: `gradient_accumulation_steps = global_batch_size / (micro_batch_size * dp_size)` (`arguments_types.py:616`)
- No explicit `no_sync()` -- FSDP2 handles synchronization internally
- `reshard_after_backward` toggled: disabled for intermediate micro-steps, re-enabled on last (`base.py:566-577`)
- HSDP all-reduce deferred to last micro-step via `set_requires_all_reduce(False/True)` (`base.py:579-589`)
- Effective batch size: `micro_batch_size × dp_size × gradient_accumulation_steps = global_batch_size`

### CPU Offload

**A. FSDP2 CPU Offload** (Parameters + Gradients + Optimizer States):
- Config: `train.accelerator.fsdp_config.offload: true`
- Passes `CPUOffloadPolicy()` to `fully_shard()` (`torch_parallelize.py:207-211`)
- Custom `_cpu_offload_fsdp2_clip_grad_norm` for gradient clipping on CPU tensors

**B. Activation Offload** (Intermediate Activations):
- Config: `train.accelerator.offload_config.enable_activation: true`, `activation_gpu_limit: <GB>`
- Uses PyTorch's `saved_tensors_hooks` (`offloading.py:32-89`)
- Heuristic filtering: skips transposition tensors and tensors ≤ 1024 bytes

**C. Model/Optimizer Offload for RL** (Manual):
- `offload_model_to_cpu()` / `load_model_to_gpu()` (`offloading.py:124-156`)
- `offload_optimizer()` / `load_optimizer()` (`offloading.py:168-199`)
- Used in RL training loops where actor model alternates with vLLM rollouts

### Dynamic Batching (Token-Level Batching)

- `dyn_bsz: True` (default): token budget per micro-batch = `micro_batch_size × max_seq_len`
- `TextBatchingStrategy` packs variable-length samples greedily into fixed token budget
- Batch-size warmup: linearly ramps from `bsz_warmup_init_mbtoken` to full budget over configured steps
- `pad_to_length` option for deterministic memory allocation

### Memory Profiling

| Capability | Implementation | Metrics |
|------------|---------------|---------|
| Per-step GPU memory | `EnvironMeter.step()` (`helper.py:255-277`) | `max_memory_allocated(GB)`, `max_memory_reserved(GB)`, `num_alloc_retries` |
| Per-step CPU memory | `psutil.virtual_memory()` | `cpu_used_memory(GB)`, `cpu_available_memory(GB)` |
| Periodic cache clearing | `empty_cache_steps` config (default 500) | `device.empty_cache()` every N steps |
| Periodic GC | `gc_steps` config (default 500); GC disabled between intervals | Reduces GC pauses by batching collection |
| PyTorch Profiler | `ProfileTraceCallback` with `profile_memory=True` | Full trace + memory snapshot `.pkl` |
| WandB logging | `WandbTraceCallback` | All EnvironMeter metrics per step |

### Memory Estimation (Qwen3-72B, 8× L20, FSDP2, bf16, grad_ckpt=True)

| Component | Per-GPU (dp_shard_size=8) |
|-----------|--------------------------|
| Parameters (sharded) | ~18 GB |
| Gradients (sharded, fp32 reduce) | ~18 GB |
| Optimizer states (AdamW, sharded) | ~9 GB |
| Activations (with grad ckpt) | ~2-4 GB |
| Temporary AllGather buffers | ~varies by layer |
| **Total estimate** | **~30-35 GB per GPU** |

With `AnyPrecisionAdamW` (bf16 states): optimizer drops to ~4.5 GB, total ~25-30 GB.

---

## Dimension 5: Precision Management

### FP8 Scaling Strategies Found

| Strategy | Class/Function | Granularity | Data Formats | GPU Support | Evidence |
|----------|---------------|-------------|--------------|-------------|----------|
| Dynamic Symmetric Per-Head | `symmetric_quantize()` | Per-head | `float8_e4m3fn` | Hopper (FA3) | `models/transformers/wan/modeling_wan.py:75-101` |
| Block-wise FP8 Act Quant (script) | `act_quant()` | Per-block (128) | `float8_e4m3fn` | CUDA (Triton) | `scripts/deepseek_v3/kernel.py:22-29` |
| Block-wise FP8 Weight Dequant (script) | `weight_dequant()` | Per-block (128) | `float8_e4m3fn` → bf16 | CUDA (Triton) | `scripts/deepseek_v3/kernel.py:47-54` |
| Block-wise FP8 GEMM (script) | `fp8_gemm()` | Per-block tiles | FP8 inputs, FP32 accum | CUDA (Triton) | `scripts/deepseek_v3/kernel.py:110-119` |

**Key Finding**: VeOmni does NOT implement FP8 training-time precision management as a framework feature. There are **zero** occurrences of `DelayedScaling`, `Float8CurrentScaling`, `Float8BlockScaling`, `MXFPBlockScaling`, `torchao`, `transformer_engine`, `Float8Linear`, or `enable_fsdp_float8_all_gather`.

### Precision per Training Component

| Component | Forward Pass | Backward Pass | Optimizer Step | Evidence |
|-----------|-------------|---------------|----------------|----------|
| Parameters (storage) | FP32 (master weights) | — | FP32 | `model.float()` when MP enabled (`torch_parallelize.py:442`) |
| Parameters (compute) | BF16 (FSDP2 param_dtype) | BF16 | FP32 | `MixedPrecisionConfig.param_dtype` default `"bfloat16"` |
| Gradient Reduce-Scatter | — | FP32 | — | `reduce_dtype` default `"float32"` |
| Forward Inputs | BF16 (cast_forward_inputs) | — | — | `cast_forward_inputs=True` default |
| AdamW States | — | — | FP32 | `fused=True` (`base.py:414`) |
| AnyPrecisionAdamW States | — | — | BF16 (momentum, variance, compensation) | `optimizer.py:50-52` |
| Matmul Accumulation | FP32 (TF32 disabled) | FP32 | — | `allow_tf32 = False` (`helper.py:399`) |
| Cross-Entropy Loss | FP32 (upcasted) | FP32 | — | `chunk_logprobs.py:169` |
| MoE Router | FP32 (autocast disabled) | FP32 | — | `torch.autocast(enabled=False)` |
| Batch-Invariant Ops | FP32 accumulation in Triton | — | — | `batch_invariant_ops.py:92` |

### FP8 Communication Integration

- [ ] FP8 AllGather in FSDP2: **Not implemented** — no torchao/TE references
- [ ] FP8 ReduceScatter: **Not implemented** — reduce_dtype is FP32
- [ ] NVLink-SHARP FP8: **Not implemented** — no SHARP references
- [ ] Estimated volume reduction: **N/A** — all communication is FP32 (reduce) or BF16 (all-gather)

### Mixed Precision Policy

| Field | Default | PyTorch Mapping |
|-------|---------|-----------------|
| `enable` | `True` | Controls entire MP pipeline |
| `param_dtype` | `"bfloat16"` | `MixedPrecisionPolicy(param_dtype=torch.bfloat16)` |
| `reduce_dtype` | `"float32"` | `MixedPrecisionPolicy(reduce_dtype=torch.float32)` |
| `output_dtype` | `None` | No extra cast |
| `cast_forward_inputs` | `True` | Auto-cast inputs to param_dtype |

Modules excluded from MP: Models can define `get_ignore_modules_in_mixed_precision()` — these modules get FP32 compute and `reshard_after_forward=False`.

### Loss Scaling

**No loss scaling used.** VeOmni uses BF16 (same dynamic range as FP32, 8-bit exponent), eliminating gradient underflow/overflow issues. Training loop is simply `loss.backward()` with no GradScaler.

### Precision Architecture Summary

```
Storage:     FP32 master weights (model.float())
                    ↓
FSDP2 Shard: FP32 shards on each rank
                    ↓
All-Gather:  BF16 (param_dtype=bfloat16) — 2 bytes/param communicated
                    ↓
Forward:     BF16 compute (FSDP2 MixedPrecisionPolicy autocast)
             FP32 accumulators in all Triton kernels
             FP32 for MoE router, loss, critical paths
                    ↓
Backward:    BF16 compute
                    ↓
Reduce-Scatter: FP32 (reduce_dtype=float32) — 4 bytes/grad communicated
                    ↓
Optimizer:   FP32 (AdamW) or BF16 (AnyPrecisionAdamW with Kahan compensation)
```

### Proposed New Precision techniques

| Tag | Evidence | Description |
|-----|----------|-------------|
| `bf16-mixed-precision-fsdp2` | `torch_parallelize.py:195-206` | Conservative BF16 MP with FP32 master weights, FP32 gradient reduction, and FP32 accumulation |
| `fp32-moe-router` | `deepseek_v3_gpu.py:197-198` | Explicit FP32 upcasting + autocast disabling for MoE router numerical stability |
| `anyprecision-adamw` | `optim/optimizer.py:50-52` | BF16 optimizer states with Kahan summation for 50% optimizer memory reduction |

### Precision Gaps (potential expansion areas)

| Gap | Description |
|-----|-------------|
| FP8 Linear with FSDP2 AllGather | torchao `Float8Linear` + `enable_fsdp_float8_all_gather` would halve all-gather volume |
| FP8 Training Recipes | DelayedScaling / CurrentScaling for E4M3/E5M2 GEMMs on Hopper/Blackwell |
| MXFP8 Microscaling (Blackwell) | Hardware-native 32-element E8M0 scaling |
| Selective TF32 | Current `allow_tf32=False` blanket-disables; per-module TF32 could recover throughput |
| BF16 Gradient Reduction | Default FP32 reduce-scatter doubles comm volume vs BF16 |

---

## Dimension 6: Profiling and Observability

### Built-in Profiling Capabilities

| Feature | Supported | Integration Method | Evidence |
|---------|-----------|-------------------|----------|
| PyTorch Profiler | Yes | `ProfileTraceCallback` wraps `torch.profiler.profile` with schedule | `trace_callback.py:161-184`, `helper.py:647-760` |
| NPU Profiler | Yes | Auto-detects NPU, uses `torch_npu.profiler` with AiCMetrics | `helper.py:720-730` |
| Chrome Trace Export | Yes | `export_chrome_trace()` to `.pt.trace.json.gz` | `helper.py:693` |
| GPU Memory Snapshot | Yes | `memory._dump_snapshot()` to `.pkl` | `helper.py:620-644, 696-697` |
| Multi-Rank Trace Merge | Yes | Script merges per-rank Chrome traces into single timeline | `scripts/profile/merge_chrome_trace.py` |
| NVTX / record_function | No | No annotations found | — |
| NCCL Inspector | No | No integration | — |
| Flight Recorder | No | No references | — |
| MFU Computation | Yes | `VeomniFlopsCounter` with 15 model-specific estimators | `utils/count_flops.py` |
| MoE Expert Load Monitor | Yes | `MoERouterMonitor` with heatmap, violation metrics, WandB | `utils/moe_monitor.py` |
| WandB Integration | Yes | `WandbTraceCallback` logs all metrics per step | `trace_callback.py:136-158` |
| Tqdm Progress Bar | Yes | `TqdmCallback` with per-step loss/metric postfix | `trace_callback.py:235-252` |

### Performance Metrics Collected

| Metric | Computation | Where Logged |
|--------|------------|--------------|
| `flops_achieved(T)` | Model-specific FLOPs / delta_time (TFLOPS) | EnvironMeter → WandB |
| `flops_promised(T)` | Hardware peak BF16 TFLOPS from device lookup (H100=989T, A100=312T, L20=119.5T, B200=2250T) | `count_flops.py:25-53` |
| `mfu` | `flops_achieved / flops_promised` (0-1) | EnvironMeter → WandB |
| `tokens_per_second(M)` | `batch_tokens / delta_time / 1e6` (M tokens/sec, all-reduced) | EnvironMeter → WandB |
| `consume_tokens(B)` | Cumulative token count (billions) | EnvironMeter → WandB |
| `max_memory_allocated(GB)` | `max_memory_allocated()`, all-reduced max | EnvironMeter → WandB |
| `max_memory_reserved(GB)` | `max_memory_reserved()`, all-reduced max | EnvironMeter → WandB |
| `num_alloc_retries` | Memory stats, all-reduced max | EnvironMeter → WandB |
| `training/total_loss` | Per-step loss, all-reduced | EnvironMeter → WandB |
| `training/grad_norm` | Per-step gradient norm | EnvironMeter → WandB |
| `training/lr` | From lr_scheduler | EnvironMeter → WandB |

### MoE Monitoring Details

- **Supported routers**: Qwen3MoeTopKRouter, Qwen3VLMoeTopKRouter, Qwen3OmniMoeTopKRouter, DeepseekV3TopkRouter
- **Monitored**: Per-layer per-expert token counts, normalized load matrix `[num_moe_layers, num_experts]`
- **Violation metrics**: `moe/{max,min,avg}_vio/layer_i` — deviation from uniform `1/num_experts`
- **Visualization**: PIL heatmap (YlOrRd colormap) logged as `wandb.Image`
- **Collective safety**: All-reduce across DP+SP group (excludes EP to avoid inflating counts)
- **Gating**: `moe_load_balance_monitor_interval` (default 0 = disabled)
- **RL support**: Pause/resume for rollout phases; MoE router replay for gradient consistency

### Recommended Profiling Dimensions for KernelWiki

| Dimension | What to Measure | Tools Needed | Key Metrics |
|-----------|----------------|-------------|-------------|
| Communication-Compute Overlap | SP async all-to-all overlap with Q/K/V compute | nsys timeline + NVTX annotations (currently absent) | overlap ratio, comm-exposed time |
| FSDP AllGather/ReduceScatter | Per-layer sharding communication cost | nsys NCCL trace, torch.profiler with_modules | time per collective, data volume |
| MoE All-to-All | Expert parallel token dispatch efficiency | nsys, NCCL Inspector | all-to-all latency, token imbalance |
| Pipeline Bubble Rate | (future) PP idle time fraction | nsys multi-GPU timeline | bubble_time / total_time |
| Memory Pressure | Fragmentation, OOM proximity | `num_alloc_retries`, memory snapshot | peak/reserved ratio |
| MoE Load Balance | Expert utilization uniformity | Existing MoERouterMonitor | max_vio trend, heatmap entropy |

### External Profiling Resources (Gaps)

| Tool | Status | Recommendation |
|------|--------|---------------|
| NVTX Range Annotations | Not present | Add `record_function` to key code paths |
| NCCL Inspector | Not integrated | Would enable per-collective latency analysis |
| PyTorch Flight Recorder | Not integrated | `torch.distributed.flight_recorder` for hang debugging |
| Structured JSON Logging | Not present | Enable automated metric parsing |
| nsys Recipes | Not integrated | Standardized capture scripts for distributed training |

---

## Synthesis: Expansion Decision Summary

### S.1 Library Classification

| Property | Value |
|----------|-------|
| Library | VeOmni |
| GitHub URL | ByteDance/VeOmni (internal, Seed Team) |
| Type | **training-orchestration** |
| Contains CUDA Kernels | No (11 Triton kernel files, zero CUDA) |
| Primary Knowledge Dimensions | Dim 3 (Parallelism), Dim 4 (Memory), Dim 2 (Communication) |
| Recommended KernelWiki Priority | **P1** (important orchestration framework; does not provide novel kernels but demonstrates advanced composition of FSDP2 + SP + EP) |

### S.2 Proposed Tags (for controlled vocabulary YAML)

```yaml
kernel_types:
  # New from VeOmni
  - batch-invariant-gemm         # Persistent matmul ensuring identical results regardless of batch composition
  - moe-lb-loss-fused            # Fused softmax + top-k + accumulation for MoE load-balancing loss
  - moe-token-routing            # Expert histogram, scatter, gather, index computation for MoE token routing
  - moe-group-gemm               # Grouped GEMM with per-expert batching and fused activation
  - rope-interleaved             # Interleaved rotary positional embedding with varlen support
  - rms-norm-batch-invariant     # Deterministic RMSNorm ensuring batch-invariant outputs
  - sp-all-to-all                # Ulysses sequence-parallel all-to-all (scatter heads / gather seq)
  - ep-all-to-all                # Expert-parallel all-to-all for MoE token dispatch/combine

techniques:
  # New from VeOmni
  - async-qkv-comm-overlap       # 3-stage Q/K/V all-to-all compute-communication overlap
  - composable-fsdp2-sp-ep       # Orthogonal composition of FSDP2 + Ulysses SP + EP via separate DeviceMeshes
  - muon-zero-comm-ep            # Shard(0) expert placement for zero-communication Muon optimizer
  - extra-parallel-abstraction   # Generic ExtraParallel framework for pluggable parallelism dimensions
  - sp-inside-fsdp               # SP ranks as FSDP shard ranks via flattened dp_shard_sp mesh
  - root-auto-no-reshard         # Root FSDP2 omits reshard for fused-linear backward compatibility
  - fsdp2-manual-prefetch-moe    # Manual forward/backward prefetch ordering for ExtraParallel
  - ep-token-permute-dispatch    # Three-phase MoE: permute → all-to-all → expert GEMM → reverse
  - multi-group-grad-norm        # Cascaded all-reduce for EP-aware gradient clipping
  - distributed-roll-p2p         # Distributed torch.roll via P2P isend/irecv
  - bf16-mixed-precision-fsdp2   # Conservative BF16 MP with FP32 master weights and FP32 reduction
  - dynamic-token-batching       # Token-level dynamic batching with warmup for memory optimization

hardware_features:
  # No new hardware features discovered (VeOmni is hardware-agnostic via device abstraction)

source_categories:
  - training-orchestration       # Framework that orchestrates upstream kernel libraries for distributed training
```

### S.3 Wiki Page Topics

| # | Wiki Subdirectory | Proposed Page ID | Title | Source Evidence | Related Existing Pages |
|---|-------------------|------------------|-------|----------------|----------------------|
| 1 | training/ | training-fsdp2-composition | FSDP2 Composable Parallelism: Multi-Dimensional Sharding with DeviceMesh | `distributed/torch_parallelize.py`, `distributed/parallel_state.py` | — |
| 2 | training/ | training-async-ulysses | Async Ulysses Sequence Parallel: Compute-Communication Overlap | `distributed/sequence_parallel/async_ulysses.py` | — |
| 3 | training/ | training-moe-expert-parallel | Expert Parallel: Token Dispatch, Group GEMM, and FSDP2 Integration | `distributed/moe/`, `ops/kernels/moe/` | — |
| 4 | training/ | training-batch-invariant-ops | Batch-Invariant Operations for Deterministic Training (DeepSeek V3) | `ops/batch_invariant_ops/` | — |
| 5 | training/ | training-kernel-orchestration | Kernel Dispatch Architecture: Registry-Based Backend Selection | `ops/kernel_registry.py`, `ops/dispatch.py` | — |
| 6 | training/ | training-activation-offload | Activation Offloading: CPU Offload with GPU Budget Control | `distributed/offloading.py` | — |
| 7 | training/ | training-dynamic-batching | Token-Level Dynamic Batching with Warmup | `data/dynamic_batching.py` | — |
| 8 | training/ | training-moe-monitoring | MoE Router Load Monitoring and Expert Balance Visualization | `utils/moe_monitor.py` | — |
| 9 | parallelism/ | parallel-extra-parallel | ExtraParallel: A Generic Abstraction for Pluggable Parallelism Dimensions | `distributed/parallel_plan.py` | — |
| 10 | parallelism/ | parallel-muon-zero-comm | Zero-Communication Expert Parallel for Muon Optimizer | `torch_parallelize.py` (muon_expert_zero_comm) | — |

### S.4 Repository Mappings (slug → org/repo)

```python
# For the PR candidate search script
"veomni": "ByteDance/VeOmni",

# For the PR page generation script
"veomni": "ByteDance/VeOmni",
```

### S.5 Keyword-to-Tag Mappings (for automated PR tagger)

```python
# keyword -> kernel_type tag
"batch_invariant": "batch-invariant-gemm",
"moe_lb_loss": "moe-lb-loss-fused",
"load_balancing_loss": "moe-lb-loss-fused",
"moe_scatter": "moe-token-routing",
"moe_gather": "moe-token-routing",
"moe_index": "moe-token-routing",
"expert_histogram": "moe-token-routing",
"group_gemm": "moe-group-gemm",
"rotary_interleaved": "rope-interleaved",
"rms_norm_batch": "rms-norm-batch-invariant",
"batch_invariant_rms": "rms-norm-batch-invariant",
"ulysses": "sp-all-to-all",
"seq_all_to_all": "sp-all-to-all",
"gather_seq_scatter_heads": "sp-all-to-all",
"ep_all_to_all": "ep-all-to-all",
"token_dispatch": "ep-all-to-all",
"token_combine": "ep-all-to-all",

# keyword -> technique tag
"async_ulysses": "async-qkv-comm-overlap",
"AsyncUlyssesQKV": "async-qkv-comm-overlap",
"fully_shard": "composable-fsdp2-sp-ep",
"dp_shard_sp": "sp-inside-fsdp",
"muon_expert_zero_comm": "muon-zero-comm-ep",
"ExtraParallel": "extra-parallel-abstraction",
"extra_parallel": "extra-parallel-abstraction",
"auto_no_reshard": "root-auto-no-reshard",
"set_modules_to_forward_prefetch": "fsdp2-manual-prefetch-moe",
"set_modules_to_backward_prefetch": "fsdp2-manual-prefetch-moe",
"token_pre_all2all": "ep-token-permute-dispatch",
"tokens_post_all2all": "ep-token-permute-dispatch",
"clip_grad_norm": "multi-group-grad-norm",
"roll_with_sequence_parallel": "distributed-roll-p2p",
"MixedPrecisionPolicy": "bf16-mixed-precision-fsdp2",
"dyn_bsz": "dynamic-token-batching",
"TextBatchingStrategy": "dynamic-token-batching",
```

### S.6 PR Search Keywords

```yaml
keywords_used:
  - fsdp2
  - fully_shard
  - sequence_parallel
  - ulysses
  - expert_parallel
  - moe
  - group_gemm
  - batch_invariant
  - load_balancing_loss
  - triton.jit
  - flash_attn
  - liger_kernel
  - quack
  - DeviceMesh
  - dp_shard
  - reshard
  - activation_checkpoint
  - gradient_checkpointing
  - cpu_offload
  - dynamic_batching
  - dyn_bsz
  - mixed_precision
  - MixedPrecisionPolicy
  - muon
  - AdamW
  - profiler
  - MFU
  - moe_monitor
  - async_ulysses
  - all_to_all
  - token_dispatch
  - extra_parallel
```

### S.7 Inclusion Policy Lane

```yaml
training-orchestration:
  description: |
    Capture PRs touching distributed training orchestration,
    parallelism composition, communication patterns, kernel dispatch,
    memory management, precision configuration, and profiling infrastructure.
    Skip pure documentation, CI configuration, and example config changes.
  capture_criteria:
    - changed_paths_match:
        - "veomni/distributed/**"
        - "veomni/ops/**"
        - "veomni/trainer/**"
        - "veomni/checkpoint/**"
        - "veomni/optim/**"
        - "veomni/data/dynamic_batching.py"
        - "veomni/utils/count_flops.py"
        - "veomni/utils/moe_monitor.py"
        - "veomni/utils/helper.py"
    - title_contains_any:
        - fsdp
        - fsdp2
        - parallel
        - ulysses
        - sequence_parallel
        - expert_parallel
        - moe
        - group_gemm
        - triton
        - flash_attn
        - liger
        - quack
        - checkpoint
        - gradient
        - offload
        - mixed_precision
        - fp8
        - bf16
        - profiler
        - mfu
        - dynamic_batch
        - muon
        - device_mesh
        - reshard
        - all_to_all
        - allgather
        - reduce_scatter
        - overlap
        - prefetch
  skip_criteria:
    - changed_paths_match_only:
        - "docs/**"
        - "*.md"
        - ".github/**"
        - "docker/**"
        - "configs/**"
    - pure_config_only: true
```

### S.8 Schema Extensions

New optional frontmatter fields for Wiki pages from VeOmni:

```yaml
# Proposed extensions
scope: training                                    # vs inference (existing)
parallelism_dimensions:
  - fsdp2                                          # which parallelism dimensions are involved
  - sequence_parallel
  - expert_parallel
communication_pattern: collective | p2p | mixed    # type of communication
kernel_provider: upstream | in-house | hybrid       # whether kernels are from upstream libraries
orchestration_level: framework | library | kernel   # abstraction level
```

### S.9 Hardware Features Relevant to VeOmni's Training Workloads

| Hardware Feature | Inference Relevance | Training Relevance | Specific Impact on VeOmni |
|-----------------|--------------------|--------------------|--------------------------|
| NVLink 5 (1.8 TB/s) | Partial | Core | Doubles FSDP2 all-gather/reduce-scatter bandwidth; critical for Ulysses SP all-to-all |
| NVSwitch 4 (NVL72, 130 TB/s) | Partial | Core | Enables efficient EP all-to-all at scale with NVL72 topology |
| NVLink-SHARP FP8 | No | Core | Would reduce FSDP2 gradient reduce-scatter bandwidth by 4× (currently not integrated) |
| Symmetric Memory (9× latency reduction) | Partial | Core | Would benefit small-message SP all-to-all and MoE token dispatch (not integrated) |
| Copy Engine (zero-SM transfer) | Partial | Core | Would free SMs for compute during FSDP2 all-gather (not integrated) |
| MXFP8 hardware (Blackwell) | Yes | Core | VeOmni has zero FP8 training integration; MXFP8 would provide 50-60% BF16 speedup |
| 192 MB L2 Cache (3.8× vs H100) | Beneficial | Beneficial | Improves activation recomputation performance (VeOmni uses gradient checkpointing extensively) |
| 192 GB HBM3e @ 8 TB/s | Beneficial | Core | Larger models fit per GPU; faster memory access for MoE expert weights |
| GPU Direct RDMA | Important | Important | Eliminates CPU copy for multi-node FSDP2 gradient sync |

### S.10 Upstream/Downstream Dependencies to Also Track

| Slug | GitHub URL | Relationship | Justification |
|------|-----------|-------------|---------------|
| `flash-attn` | Dao-AILab/flash-attention | kernel-provider | Primary attention kernel for all model families (FA 2/3/4) |
| `liger-kernel` | linkedin/liger-kernel | kernel-provider | Fused cross-entropy, RMSNorm, RoPE, SwiGLU for memory optimization |
| `quack` | (internal/unreleased) | kernel-provider | CUTLASS/CuTe grouped GEMM for MoE expert computation (SM90+) |
| `fla` | fla-org/flash-linear-attention | kernel-provider | Gated delta rule, causal conv1d for linear attention models |
| `flash-qla` | (unreleased) | kernel-provider | SM90-only alternative to FLA for gated delta rule |
| `pytorch` | pytorch/pytorch | runtime-dependency | FSDP2, DeviceMesh, DTensor, distributed collectives, torch.profiler |
| `transformers` | huggingface/transformers | runtime-dependency | Model architectures, tokenizers, generation utilities |
| `torch-npu` | Ascend/pytorch | runtime-dependency | NPU backend for Huawei Ascend accelerators |

---

## Appendix: Dimension Emphasis Rationale

VeOmni is a **training-orchestration framework** (zero CUDA kernels, 11 Triton kernels primarily for MoE operations). Per the KernelWiki Training Library Analysis skill's Library Type Adaptation table:

| Library Type | Examples | Primary Dimensions | Light/Skip Dimensions |
|--------------|---------|-------------------|-----------------------|
| **Orchestration framework** | torchtitan, **VeOmni** | Dim 3 (parallelism), Dim 4 (memory) | Dim 1 (compute — focus on dependency graph) |

Adjustments made:
1. **Dimension 1**: Replaced kernel-by-kernel analysis with dependency graph analysis showing upstream kernel providers (Flash Attention 2/3/4, Liger Kernel, Quack GEMM, FLA/FlashQLA)
2. **Dimension 2**: Focused on communication orchestration patterns (how FSDP2 + SP + EP compose) rather than NCCL algorithm internals
3. **Dimension 3**: Deep analysis — VeOmni's primary value is in composable parallelism with DeviceMesh
4. **Dimension 4**: Deep analysis — training memory optimization with dynamic batching, activation offloading, gradient accumulation
5. **Dimension 5**: Moderate — conservative BF16 strategy with significant FP8 gaps identified
6. **Dimension 6**: Standard — comprehensive profiling with MoE monitoring as standout feature
