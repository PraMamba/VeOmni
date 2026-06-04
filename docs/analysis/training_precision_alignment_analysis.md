# VeOmni 训练精度对齐体系源码与 Commit History 深度分析

## 0. Executive Summary

| 项目名称 | VeOmni (ByteDance Seed Team) |
|---------|------|
| **总体判断** | **较完整** — 在关键路径（mixed precision、FSDP gradient reduction、loss normalization、operator-level bitwise parity）上已形成闭环，但在系统性 golden regression、NaN guard、GPU RNG checkpoint 方面存在空白 |
| **最强能力** | 1) Batch-invariant ops (DeepSeek V3 确定性运算); 2) Bitwise logits parity test (HF vs VeOmni); 3) FSDP2 equivalence test; 4) MoE router replay for RL parity |
| **最大短板** | 1) 无 golden loss baseline regression; 2) 无 NaN/Inf training loop guard; 3) GPU RNG state 未写入 checkpoint; 4) 无 activation/gradient dump 调试工具 |
| **最值得借鉴的源码模块** | `veomni/ops/batch_invariant_ops/`, `tests/distributed/test_fsdp_equivalence.py`, `tests/models/test_models_logits_equal_v5.py`, `veomni/utils/moe_router_replay.py` |
| **最值得研究的 commits/PR** | `4011a06` (bitwise logit test + bf16 MoE dtype fix), `aa1578c` (optimizer state regression test), `3290fe3`+`b6efb6e` (cross-EP tolerance 探索与回退), `ac22e25` (MoE router replay), `30af894` (FSDP equivalence test) |
| **是否适合作为训练精度对齐基础设施参考** | **是** — 尤其在 FSDP2 + MoE + SP 场景下，是 PyTorch-native 路线的优秀参考；但需补齐 regression golden 和 dump 工具层 |

---

## 1. 项目训练流程与精度相关架构总览

### 1.1 训练主入口

| Entry Point | 文件 | Trainer | 
|-------------|------|---------|
| Text SFT | `tasks/train_text.py` | `TextTrainer` |
| VLM | `tasks/train_vlm.py` | `VLMTrainer` |
| DiT | `tasks/train_dit.py` | `DiTTrainer` |
| DPO | `tasks/train_text_dpo.py` | `TextDPOTrainer` |
| RL | `tasks/train_text_rl.py` | `BaseRLTrainer` |

所有入口调用模式统一：`parse_args() → Trainer() → trainer.train()`

### 1.2 训练 step 主循环（精度关键路径）

```
BaseTrainer.train() [base.py:640]
  └── train_step() [base.py:591]
        ├── next(data_iterator)          # 数据确定性依赖 seed
        ├── for micro_batch in micro_batches:
        │     └── forward_backward_step() [base.py:547]
        │           ├── model_fwd_context (batch_invariant_mode)  [base.py:552]
        │           ├── model.forward()   # BF16 forward via FSDP2 MP
        │           ├── postforward() → mean_global_loss()  [base.py:537]
        │           ├── model_bwd_context (batch_invariant_mode)  [base.py:560]
        │           └── loss.backward()   # FP32 reduce-scatter
        ├── veomni_clip_grad_norm()       # FP32 norm computation
        ├── optimizer.step()             # FP32 optimizer states
        ├── lr_scheduler.step()
        └── optimizer.zero_grad()
```

### 1.3 精度控制链

```
TrainingArguments
  ├── seed: int = 42
  ├── enable_full_determinism: bool = False
  ├── enable_batch_invariant_mode: bool = False
  └── accelerator.fsdp_config.mixed_precision:
        ├── enable: bool = True
        ├── param_dtype: "bfloat16"      # forward in BF16
        ├── reduce_dtype: "float32"      # gradient reduction in FP32
        └── output_dtype: None
```

**关键初始化序列**（`BaseTrainer._setup()`，`base.py:237-248`）：
1. `helper.set_seed(seed, enable_full_determinism)` — 全局 RNG 设置
2. `helper.enable_high_precision_for_bf16()` — 禁用 TF32 + BF16 reduced precision reduction
3. `save_args(self.args, ...)` — 配置快照

### 1.4 Mixed Precision 架构

**唯一机制：FSDP2 MixedPrecisionPolicy**

```python
# torch_parallelize.py:198-205
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,      # 参数 BF16
    reduce_dtype=torch.float32,      # 梯度 reduce FP32
    output_dtype=None,               # 不额外 cast
    cast_forward_inputs=True,
)
```

**无 GradScaler、无手动 autocast**。这是正确的 FSDP2-native 设计。

### 1.5 Distributed Parallel 架构

| 并行维度 | 实现文件 | 精度相关 |
|---------|---------|---------|
| FSDP2 (dp_shard) | `distributed/fsdp2/` | reduce_dtype=FP32 |
| HSDP (dp_replicate) | `distributed/fsdp2/` | gradient all-reduce delayed in accum |
| Ulysses SP | `distributed/sequence_parallel/` | loss reduction via ReduceLoss autograd fn |
| Expert Parallel | `distributed/moe/` | gradient_divide_factor = world_size |
| Embed Parallel | `distributed/parallel_plan.py` | 与 FSDP2 共用 MP policy |

---

## 2. 精度对齐能力矩阵

| 能力项 | 是否具备 | 源码证据 | Commit/PR 证据 | 成熟度 | 备注 |
|--------|---------|---------|---------------|--------|------|
| 配置一致性扫描 | 部分 | `save_args()` at startup | — | 1 | 有配置快照但无 resume diff 验证 |
| 随机种子/RNG 控制 | **是** | `helper.set_seed()`, `enable_full_determinism()` | `8447a8d` | 3 | CPU+CUDA+numpy+python，缺分布式 per-rank tracker |
| 数据加载顺序确定性 | **是** | `SeedSequence([seed, epoch, dp_rank, worker_id])` | — | 3 | 确定性 seeded per-rank per-epoch |
| 初始权重一致性 | **是** | `rank0_load_and_broadcast_weights()` | `083873c` | 3 | Rank0 加载后 broadcast 保证一致 |
| 单步 forward loss 对齐 | 间接 | `test_fsdp_equivalence.py` (single-GPU vs FSDP2) | `30af894` | 2 | 验证 grad_norm，非直接 loss compare |
| Activation dump/compare | **未发现** | — | — | 0 | 无 activation dump 工具 |
| Gradient dump/compare | **未发现** | — | — | 0 | 无 gradient dump 工具 |
| Optimizer state 对齐 | **是** | `test_trainer_saveload.py`, `assert_close(rtol=0, atol=0)` | `aa1578c` | 3 | Bitwise round-trip 验证 |
| Scheduler/LR curve 对齐 | 部分 | LR scheduler state_dict save/load | — | 2 | 有 save/load 但无独立测试 |
| Loss curve golden regression | **未发现** | — | — | 0 | 无 golden baseline loss 对比 |
| Mixed precision 对齐 | **是** | `MixedPrecisionPolicy`, `enable_high_precision_for_bf16()` | `2b21f29`, `b1d6bf5` | 3 | BF16 forward + FP32 reduce + TF32 disabled |
| FP16/BF16/FP8 数值稳定性 | 部分 | BF16 完整, FP8 仅推理, FP16 无 scaler | `946ccca` | 2 | FP8 训练不支持 |
| TF32 控制 | **是** | `allow_tf32 = False` 全局 | `b1d6bf5` | 4 | 所有 trainer 强制关闭 |
| NaN/Inf/overflow 检测 | 零散 | model-level `torch.isnan` only | — | 1 | 训练主循环无 guard |
| Checkpoint resume 一致性 | **是** | `verify_dcp_to_hf_conversion()`, `assert_close(rtol=0, atol=0)` | `fe5d65d`, `3bf42c3` | 3 | 权重 bitwise 验证，optimizer round-trip |
| Data parallel correctness | **是** | `test_fsdp_equivalence.py` | `30af894` | 3 | Single-GPU vs FSDP2 grad_norm |
| Tensor parallel correctness | **未发现** | — | — | 0 | TP 未实现 (NotImplementedError) |
| Pipeline parallel correctness | **未发现** | — | — | 0 | PP 未实现 (NotImplementedError) |
| Sequence parallel correctness | **是** | `test_ulysses.py`, `test_e2e_parallel.py` | `5fb669f` | 3 | SP attention parity + e2e loss curve |
| Expert parallel / MoE correctness | **是** | `test_e2e_parallel.py` (EP grid), MoE router replay | `3290fe3`, `ac22e25` | 3 | EP loss/grad_norm + router replay for RL |
| Collective communication correctness | 部分 | `hccl_premul_sum.py` (NPU fix) | `b0e38dc` | 2 | 主要修复 NPU，GPU 隐式依赖 PyTorch |
| CI 精度回归测试 | **是** | `gpu_unit_tests.yml`, `gpu_e2e_test.yml` | 多个 | 3 | 每 PR 跑 logit parity + FSDP equiv + e2e |
| 自动化二分定位能力 | **未发现** | — | — | 0 | 无自动 bisect 工具 |
| 跨硬件/跨后端对齐能力 | 部分 | GPU + NPU dual CI | NPU workflows | 2 | 有双端 CI 但无跨端数值对比 |

---

## 3. 源码证据地图

### 3.1 确定性控制

| 功能 | 文件 | 函数/类 | 行号 |
|------|------|--------|------|
| 全面确定性设置 | `veomni/utils/helper.py` | `enable_full_determinism()` | 407-431 |
| BF16 高精度模式 | `veomni/utils/helper.py` | `enable_high_precision_for_bf16()` | 394-404 |
| Seed 设置 | `veomni/utils/helper.py` | `set_seed()` | 438 |
| Batch invariant ops context | `veomni/ops/batch_invariant_ops/__init__.py` | `set_batch_invariant_mode()` | — |
| Batch invariant mm kernel | `veomni/ops/batch_invariant_ops/batch_invariant_ops.py` | `mm_batch_invariant` | FP32 accumulator |
| Batch invariant RMSNorm | `veomni/ops/kernels/rms_norm/triton_batch_invariant.py` | `batch_invariant_rms_norm` | — |
| Deterministic RoPE (Triton) | `veomni/ops/kernels/rotary/triton_deterministic.py` | `triton_bmm` | — |
| DeepSeek-V3 deterministic RoPE patch | `veomni/models/transformers/deepseek_v3/device_patch.py` | `_make_deterministic_rope_forward` | 20-50 |
| Gradient checkpointing RNG save/restore | `veomni/distributed/checkpoint.py` | `CheckpointFunction` | 29-39, 97-115 |
| Flash Attention determinism env var | `veomni/models/transformers/wan/modeling_wan.py` | 检查 `FLASH_ATTENTION_DETERMINISTIC` | 278-281 |
| Load balancing loss determinism | `veomni/ops/kernels/load_balancing_loss/triton.py` | "ensuring deterministic results" | 21 |
| MoE LB loss determinism test | `tests/ops/test_fused_load_balancing_loss.py` | `test_determinism` | 222, 238, 313 |
| Dataset seed per-rank | `veomni/data/dataset.py` | `SeedSequence` | 274, 453-454 |
| Patchgen determinism CI | `veomni/patchgen/check_patchgen.py` | — | 16 |

### 3.2 Mixed Precision 关键路径

| 功能 | 文件 | 函数/类 | 行号 |
|------|------|--------|------|
| MP config dataclass | `veomni/arguments/arguments_types.py` | `MixedPrecisionConfig` | 241-305 |
| FSDP2 MP policy 应用 | `veomni/distributed/torch_parallelize.py` | `parallelize_model_fsdp2()` | 198-222 |
| 忽略 MP 模块列表 | `veomni/distributed/torch_parallelize.py` | `get_ignore_modules_in_mixed_precision` hook | 213-228 |
| FP32 upcast before optimizer | `veomni/distributed/torch_parallelize.py` | `build_parallelize_model()` | 442 |
| Cross entropy FP32 upcast | `veomni/ops/kernels/cross_entropy/eager.py` | `logits = F.linear(...).float()` | 34 |
| Cross entropy chunk FP32 | `veomni/ops/kernels/cross_entropy/chunk_loss.py` | `logits = F.linear(...).float()` | 115 |
| RoPE autocast disabled | 多个 patched_modeling_*.py | `maybe_autocast(enabled=False)` | — |
| DeepSeek-V3 router FP32 | patched_modeling_deepseek_v3_gpu.py | `torch.autocast(enabled=False)` | 197-198 |
| `_keep_in_fp32_modules_strict` | patched_modeling_deepseek_v3_gpu.py | — | 657 |
| AnyPrecisionAdamW | `veomni/optim/optimizer.py` | `AnyPrecisionAdamW` | 41-143 |

### 3.3 分布式精度关键路径

| 功能 | 文件 | 函数/类 | 行号 |
|------|------|--------|------|
| Global loss normalization | `veomni/utils/loss_utils.py` | `mean_global_loss()` | 54-90 |
| SP loss reduction | `veomni/distributed/sequence_parallel/loss.py` | `reduce_sequence_parallel_loss()` | 63 |
| SP loss zero-division guard | 同上 | `ReduceLoss` autograd fn | 30, 44 (clamp_min(1)) |
| EP gradient divide factor | `veomni/distributed/torch_parallelize.py` | `set_gradient_divide_factor()` | 306-313 |
| HSDP all-reduce delay | `veomni/trainer/base.py` | `_configure_hsdp_allreduce()` | 579-589 |
| FP32 grad norm computation | `veomni/distributed/fsdp2/clip_grad_norm.py` | `_local_pth_sum()` | 175-193 |
| EP+FSDP2 clip grad norm | `veomni/distributed/fsdp2/clip_grad_norm.py` | `extra_parallel_fsdp2_clip_grad_norm()` | — |
| NPU premul_sum patch | `veomni/ops/platform/npu/hccl_premul_sum.py` | Patches `all_reduce`, `reduce_scatter` | — |
| Rank0 weight broadcast | `veomni/models/module_utils.py` | `rank0_load_and_broadcast_weights()` | 384 |
| MoE router replay | `veomni/utils/moe_router_replay.py` | RECORD/REPLAY modes | — |

### 3.4 Checkpoint 精度路径

| 功能 | 文件 | 函数/类 | 行号 |
|------|------|--------|------|
| DCP checkpointer | `veomni/checkpoint/dcp_checkpointer.py` | `DistributedCheckpointer` | — |
| Checkpoint callback | `veomni/trainer/callbacks/checkpoint_callback.py` | `CheckpointerCallback` | — |
| Extra state (含 RNG) | 同上 | `_save_checkpoint()` | 111-122 |
| CPU RNG save | 同上 | `torch.get_rng_state()` | 121 |
| GPU RNG **未保存** | 同上 | — | **缺失** |
| EP-aware checkpoint | `veomni/checkpoint/dcp_checkpointer.py` | `_apply_extra_parallel_dim()` | — |
| Checkpoint verification | `tests/checkpoints/checkpoint_verification_utils.py` | `verify_hf_checkpoint_weights()` | 221-290 |
| DCP→HF bitwise exact | 同上 | `verify_dcp_to_hf_conversion()` | 319 (assert_close rtol=0, atol=0) |
| Dtype unchanged assertion | `tests/checkpoints/test_trainer_saveload.py` | `assert_param_dtypes_unchanged()` | 85-93 |

### 3.5 测试工具

| 工具 | 文件 | 功能 |
|------|------|------|
| TensorComparator | `tests/tools/comparison_utils.py` | 配置化 rtol/atol tensor 对比 |
| check_metric | `tests/e2e/utils.py` | 训练 log 中 loss/grad_norm series 对比 |
| compare_log | `tests/e2e/utils.py` | 双 series 同时对比 (loss + grad_norm) |
| verify_hf_checkpoint_weights | `tests/checkpoints/checkpoint_verification_utils.py` | Checkpoint 权重 bitwise 验证 |
| CUDA sync gate | `tests/models/test_model_forward_no_implicit_sync.py` | 检测隐式 GPU-CPU 同步 |

---

## 4. Commit / PR / Issue 历史演进时间线

### 4.1 精度对齐能力演进总览

```
2025-09 ─── 项目初创，GradientCheckpointingLayer RNG 修复
2025-10 ─── FSDP2 gradient checkpoint 修复
2025-11 ─── FSDP2 grad norm clipping 重构
2025-12 ─── full determinism 支持 / SP loss 修复 / 统一 grad norm
2026-01 ─── Checkpoint verification / DCP→HF 转换
2026-02 ─── enable_high_precision_for_bf16 全 trainer / padding batch exclusion
2026-03 ─── SP zero-division guard / ReduceLoss
2026-04 ─── Bitwise HF vs VeOmni logits test / FSDP equivalence test / mixed_precision config
2026-05 ─── Cross-EP tolerance 探索与回退 / MoE router replay / optimizer state regression test
2026-05 ─── DCP multi-node broadcast race fix / full determinism in save/load test
```

### 4.2 高价值 Commit 详细分析

---

#### Commit 1: Bitwise HF vs VeOmni Logits Parity Test

| 字段 | 内容 |
|------|------|
| **Commit** | `4011a06` |
| **PR** | #670 |
| **时间** | 2026-04-17 |
| **涉及文件** | `tests/models/test_models_logits_equal.py` (+217), `veomni/models/transformers/deepseek_v3/gpu_patch.py`, `veomni/models/transformers/qwen3_moe/modeling_qwen3_moe.py` |
| **问题背景** | VeOmni 对 HuggingFace 模型进行了 monkey-patching（flash attention、fused ops），需验证 patch 后模型输出与原始 HF 模型 bitwise 一致 |
| **修改内容** | 1) 新增 `test_logits_bitwise_equal` 使用 `torch.equal` 验证 bitwise 精确; 2) 修复了 DeepSeek-V3 和 Qwen3-MoE 中因 router weight 在 bf16 下 dtype 不一致导致的 logits drift |
| **新增测试** | `test_models_logits_equal.py` — 所有支持模型的 bitwise logits 回归测试 |
| **影响范围** | 模型 patch 正确性的核心保障 |
| **进入 CI** | 是 (`gpu_unit_tests.yml`) |
| **对精度对齐体系的启发** | **模型 patch 后必须有 bitwise parity gate** — 不是"大致相同"，是"完全一致"。此 commit 同时修复了发现的 dtype 问题，体现了"test → discover → fix → CI" 闭环 |

---

#### Commit 2: FSDP Equivalence Test

| 字段 | 内容 |
|------|------|
| **Commit** | `30af894` |
| **PR** | #620 |
| **时间** | 2026-04-01 |
| **涉及文件** | `tests/distributed/test_fsdp_equivalence.py` (+251), `tests/tools/comparison_utils.py` (+117), `tests/distributed/distributed_test_helpers.py` (+188) |
| **问题背景** | FSDP2 wrapping 可能引入数值差异（gradient reduction、mixed precision casting、sharding 顺序） |
| **修改内容** | 新增 single-GPU vs FSDP2 (2 GPU) 训练对比，使用 grad_norm 作为 proxy metric，覆盖 llama3.1, qwen3_5, qwen3_5_moe, deepseek_v3 |
| **新增测试** | `test_fsdp_equivalence.py` + `TensorComparator` 工具 |
| **进入 CI** | 是 (`gpu_e2e_test.yml`) |
| **对精度对齐体系的启发** | **Single-GPU baseline 是 distributed correctness 的 ground truth** — 如果 1GPU 和 FSDP 结果不一致，一定是 FSDP 引入了问题。grad_norm 是好的 aggregate proxy |

---

#### Commit 3: Cross-EP Grad Norm Tolerance 探索与回退

| 字段 | 内容 |
|------|------|
| **Commit** | `3290fe3` (relax) → `b6efb6e` (revert to strict) |
| **时间** | 2026-05-05 (同一天) |
| **问题背景** | EP=1 vs EP=2 在 MoE 模型上 grad_norm 存在差异，因 BF16 噪声翻转了 router top-k 决策 |
| **修改内容** | 第一版放宽到 rtol=0.5, atol=1.0；后经 H100 100次重复验证确认 L20 是硬件特异性 flake，回退到 strict rtol=0.1 并留文档 |
| **对精度对齐体系的启发** | **永远不要轻易放宽 tolerance** — 先在可控硬件上大量重复验证，区分"算法固有差异"和"CI 环境 flake"。文档化 flake 原因比放宽阈值更好 |

---

#### Commit 4: Optimizer State Regression Test

| 字段 | 内容 |
|------|------|
| **Commit** | `aa1578c` |
| **时间** | 2026-05-26 |
| **问题背景** | 生产环境 qwen3.5-35B-a3b VL 在 h100x16 step 100 时 checkpoint save 挂死（DCP metadata broadcast race），但现有测试因 batch 太小从未触发 "未使用参数的 optimizer state" 路径 |
| **修改内容** | 新增 2-GPU FSDP2 测试，模型含 `unused` 子模块永不收梯度，验证完整 save+load 合约（包括 TensorProperties.__hash__ monkey-patch、placeholder state 创建、strict load、round-trip 精确性） |
| **对精度对齐体系的启发** | **Regression test 必须覆盖 edge case（未使用参数、部分 expert 未激活）** — 正常训练的 small-scale test 无法触发生产中的 corner case。commit message 明确记录了生产 bug 的复现条件 |

---

#### Commit 5: MoE Router Replay Hook

| 字段 | 内容 |
|------|------|
| **Commit** | `ac22e25` |
| **PR** | #719 |
| **时间** | 2026-05-15 |
| **涉及文件** | `veomni/utils/moe_router_replay.py` (+220), `tests/models/test_moe_router_replay_invariants.py` (+460), `tests/utils/test_moe_router_replay.py` (+253) |
| **问题背景** | RL 训练中 actor 和 rollout 需要 MoE routing 决策完全一致。由于 BF16 舍入噪声，同一输入在不同 context 下 router top-k 可能翻转 |
| **修改内容** | Record/Replay 基础设施：RECORD 模式记录原生路由结果，REPLAY 模式在后续 pass 中重放完全相同的 (indices, weights)，且验证 RECORD 模式与 no-RR baseline bitwise 一致 |
| **新增测试** | 两个独立 test 文件，bitwise invariant 验证 |
| **对精度对齐体系的启发** | **当算法固有存在非确定性（BF16 router top-k）时，需要 Record/Replay 机制而非仅依赖 determinism flags** — 这是 "无法通过 seed 解决的精度问题"的正确工程方案 |

---

#### Commit 6: SP Loss 修复

| 字段 | 内容 |
|------|------|
| **Commit** | `5fb669f` |
| **PR** | #339 |
| **时间** | 2025-12-29 |
| **涉及文件** | `veomni/utils/loss_utils.py` (+60), `tasks/train_torch.py` |
| **问题背景** | Sequence Parallel 启用后 loss 计算不正确 — 各 SP rank 只看到部分 tokens，但 loss normalization 未正确 all-reduce token count |
| **修改内容** | 新增 `mean_global_loss()` 函数，正确处理 SP 场景下 token count 的 all-reduce 和 fsdp_size 补偿 |
| **对精度对齐体系的启发** | **新增并行维度时，loss normalization 必须同步更新** — loss 值异常是最容易被发现的精度问题信号 |

---

#### Commit 7: Full Determinism 支持（不依赖 CUDA_LAUNCH_BLOCKING）

| 字段 | 内容 |
|------|------|
| **Commit** | `8447a8d` |
| **PR** | #318 |
| **时间** | 2025-12-23 |
| **涉及文件** | `veomni/utils/helper.py` (+33) |
| **问题背景** | 原实现要求 `CUDA_LAUNCH_BLOCKING=1` 才能开启 determinism，严重影响性能 |
| **修改内容** | 改用 `torch.use_deterministic_algorithms(True, warn_only=True)` + 各框架 env var（NCCL_DETERMINISTIC, FLASH_ATTENTION_DETERMINISTIC, CUBLAS_WORKSPACE_CONFIG），移除 CUDA_LAUNCH_BLOCKING 依赖 |
| **对精度对齐体系的启发** | **Determinism 不应该以 10x 性能下降为代价** — `warn_only=True` 允许在不支持 deterministic 的算子上给出警告而非报错，是实用工程取舍 |

---

#### Commit 8: 统一 Grad Norm Clipping

| 字段 | 内容 |
|------|------|
| **Commit** | `f331f6f` |
| **PR** | #205 |
| **时间** | 2025-12-06 |
| **涉及文件** | `veomni/distributed/fsdp2/clip_grad_norm.py`, `veomni/distributed/clip_grad_norm.py`, `tests/utils/test_ep_clip_grad_norm.py` (+272) |
| **问题背景** | EP + FSDP2 在 NPU 上由于缺少 PreSumMul ReduceOp，gradient divide 需要在 clip 内部处理而非通过 `set_gradient_divide_factor` |
| **修改内容** | 统一 GPU/NPU grad norm clipping 路径，所有梯度在 clip 前 cast 到 FP32 |
| **对精度对齐体系的启发** | **Grad norm 必须在 FP32 计算** — BF16 只有 7 bit mantissa，p-norm 在 BF16 下会严重不准 |

---

#### Commit 9: enable_high_precision_for_bf16 全覆盖

| 字段 | 内容 |
|------|------|
| **Commit** | `946ccca` + `b1d6bf5` |
| **PR** | #465 |
| **时间** | 2026-02-08/09 |
| **问题背景** | 部分 trainer entry point（flux, wan, omni, qwen_vl）遗漏了 `enable_high_precision_for_bf16()` 调用 |
| **修改内容** | 在所有 5 个遗漏的 task 入口补上调用 |
| **对精度对齐体系的启发** | **精度设置必须在 base class 中统一，不能依赖各 trainer 手动调用** — 后来该调用被移入 `BaseTrainer._setup()` |

---

#### Commit 10: DCP Multi-Node Broadcast Race Fix

| 字段 | 内容 |
|------|------|
| **Commit** | `f1e5886` |
| **时间** | 2026-05-28 |
| **问题背景** | 多节点训练中 DCP checkpoint save/load 存在 broadcast race condition 和 HDFS-FUSE soft hang |
| **修改内容** | 加入 CUDA sync inside DCP Metadata broadcast (`b7ced20`)，retry + local-staging fallback for HDFS |
| **对精度对齐体系的启发** | **Checkpoint 的一致性不仅是数值问题，还有分布式同步问题** — 如果 metadata broadcast 未完成就开始 load，会加载到不完整/错误的权重 |

---

#### Commit 11: Qwen3-MoE Router Weight Dtype Fix

| 字段 | 内容 |
|------|------|
| **Commit** | `90d4bc2` |
| **时间** | 2026-05-20 |
| **问题背景** | transformers 5.8 修改了 Qwen3MoeTopKRouter.forward，将 top-k weights cast 回 router_logits.dtype。VeOmni patch 保留了旧的 fp32 assumption |
| **修改内容** | Cast to `router_logits.dtype` 保持与 vanilla HF bitwise-equal |
| **对精度对齐体系的启发** | **上游依赖升级后必须验证 dtype 一致性** — bitwise logits test 正好捕获了此问题 |

---

## 5. 典型精度问题案例复盘

### Case 1: SP 启用后 Loss 计算不正确

| 字段 | 内容 |
|------|------|
| **现象** | SP 模式下 loss 值异常（比预期高/低） |
| **根因** | 各 SP rank 只看到 sequence 的一部分 tokens，但 loss 除以了本地 token count 而非全局 token count；FSDP 的隐式 gradient division 也未被补偿 |
| **如何定位** | 对比 SP=1 和 SP=2 的 loss 值 |
| **如何修复** | `mean_global_loss()` 正确 all-reduce token count，乘以 `fsdp_size` 补偿 FSDP division |
| **是否新增测试** | 后续在 `test_e2e_parallel.py` 中覆盖 |
| **借鉴** | **每增加一个并行维度，loss normalization 都要重新验证** |

### Case 2: BF16 MoE Router Top-K 翻转导致 EP=1 vs EP=2 Grad Norm 差异

| 字段 | 内容 |
|------|------|
| **现象** | EP=1 和 EP=2 的 grad_norm 差异超过 10%，CI flake |
| **根因** | BF16 精度下 router softmax 概率在 top-k 边界处的微小差异，导致不同 EP 配置下 expert 选择不同 |
| **如何定位** | 100 次重复实验分离硬件 flake 和算法差异 |
| **如何修复** | 在 H100 上验证确认 L20 特异性后，保持 strict tolerance 并文档化原因 |
| **是否新增测试** | 是（`test_e2e_parallel.py` 含 cross-EP tolerance 文档） |
| **借鉴** | **不同 EP 配置在数学上不等价（token composition 不同），tolerance 设计需考虑算法固有差异** |

### Case 3: Qwen3-MoE Router Double-Softmax Bug

| 字段 | 内容 |
|------|------|
| **现象** | Router logits 经过两次 softmax，导致梯度异常小、训练发散 |
| **根因** | VeOmni patch 和 HF 原始代码各做了一次 softmax |
| **如何定位** | Bitwise logits test 发现输出不匹配 |
| **如何修复** | 修正 patch 代码移除重复 softmax（`64557a8`） |
| **是否新增测试** | Bitwise logits test 覆盖 |
| **借鉴** | **Model patch 后的 bitwise parity test 可以自动发现此类 bug** |

### Case 4: Optimizer State Save 时未使用参数的 State 缺失

| 字段 | 内容 |
|------|------|
| **现象** | 生产环境 h100x16 训练在 step 100 时 checkpoint save 挂死 |
| **根因** | MoE 模型中部分 expert 在前 5 步内未被激活，其 optimizer state 为空。DCP metadata broadcast 因 `TensorProperties.__hash__` TypeError 导致 NCCL hang |
| **如何定位** | 从生产 log 追踪到 DCP save 路径，构造含 unused 模块的 minimal repro |
| **如何修复** | 1) Monkey-patch `TensorProperties.__hash__`; 2) `fill_missing_optimizer_states=True` 填充 placeholder |
| **是否新增测试** | 是（`aa1578c`：专门测试 unused 参数的 save/load round-trip） |
| **借鉴** | **Regression test 必须构造极端 case — 小 batch 正常训练永远不会触发此路径** |

### Case 5: enable_high_precision_for_bf16 遗漏

| 字段 | 内容 |
|------|------|
| **现象** | 部分 trainer（flux, wan）的训练精度低于预期 |
| **根因** | TF32 matmul 和 BF16 reduced precision reduction 未关闭 |
| **如何定位** | 代码审查发现 5 个 task entry point 遗漏了调用 |
| **如何修复** | 补上调用；后来移入 `BaseTrainer._setup()` 确保不再遗漏 |
| **是否新增测试** | 无直接测试 |
| **借鉴** | **精度相关设置必须在框架层集中处理，不能散落在各 trainer** |

### Case 6: DCP Multi-Node Broadcast Race

| 字段 | 内容 |
|------|------|
| **现象** | 多节点训练中 checkpoint resume 后权重不一致 |
| **根因** | DCP metadata broadcast 未在 CUDA sync 后进行，导致部分 rank 读取到不完整 metadata |
| **如何定位** | `b7ced20` 在 broadcast 前插入 `synchronize()` |
| **如何修复** | Force CUDA sync inside DCP Metadata broadcast |
| **是否新增测试** | 无单独测试（依赖 `test_trainer_saveload.py`） |
| **借鉴** | **Checkpoint 操作必须有显式同步 barrier** |

### Case 7: DeepSeek-V3 RoPE 非确定性

| 字段 | 内容 |
|------|------|
| **现象** | DeepSeek-V3 在某些 GPU 架构上第一次调用 RoPE 的结果与后续调用不同 |
| **根因** | 默认 PyTorch matmul 在特定硬件上存在非确定性 |
| **如何定位** | 通过 `enable_full_determinism` + 反复运行比对 |
| **如何修复** | 用 Triton deterministic kernel (`triton_deterministic.py`) 替换默认 matmul |
| **是否新增测试** | MoE LB loss `test_determinism` 测试覆盖类似场景 |
| **借鉴** | **硬件级非确定性需要 custom kernel 解决，seed alone 不够** |

### Case 8: NPU HCCL PreSumMul 不支持

| 字段 | 内容 |
|------|------|
| **现象** | NPU 上 gradient reduction 结果与 GPU 不一致 |
| **根因** | HCCL (NPU 通信库) 不支持 PreSumMul ReduceOp |
| **如何定位** | NPU CI 失败 |
| **如何修复** | Monkey-patch `torch.distributed.all_reduce/reduce_scatter` on NPU (`hccl_premul_sum.py`) |
| **是否新增测试** | NPU CI workflow |
| **借鉴** | **跨硬件精度对齐需要 per-backend communication patches** |

### Case 9: Patchgen Code Drift（非典型精度问题，但影响精度保证）

| 字段 | 内容 |
|------|------|
| **现象** | Patchgen 生成的代码随时间可能 drift（依赖版本变化导致重新生成不一致） |
| **根因** | Code generation 本身可能是非确定性的 |
| **如何定位** | `1a2f973` 新增 patchgen CI check |
| **如何修复** | CI 中运行 `check_patchgen.py` 验证生成代码与 committed 代码一致 |
| **是否新增测试** | `check_patchgen.yml` workflow |
| **借鉴** | **如果训练代码是自动生成的，生成过程本身也需要确定性保证** |

### Case 10: CI Save/Load Test 非确定性 Flake

| 字段 | 内容 |
|------|------|
| **现象** | `test_trainer_saveload` 偶尔 flake — 两次独立训练的权重不 bitwise 一致 |
| **根因** | 未开启 full determinism 时，cuDNN/NCCL/CUBLAS 可能导致训练结果 drift |
| **如何定位** | CI flake 分析 |
| **如何修复** | `3bf42c3` 在测试中开启 `enable_full_determinism` |
| **是否新增测试** | 修改现有测试配置 |
| **借鉴** | **精度比较测试必须在 full determinism 模式下运行** |

---

## 6. 三阶段精度对齐流程映射

### 阶段一：训练前准备与基础对齐

| 检查项 | VeOmni 状态 | 证据 |
|--------|------------|------|
| 配置一致性 | ✅ 有 `save_args()` 快照 | `base.py:248` |
| 环境一致性 | ✅ `enable_full_determinism()` 设环境变量 | `helper.py:407` |
| Seed/RNG | ✅ CPU+CUDA+numpy+python | `helper.py:438` |
| 数据顺序 | ✅ SeedSequence per-rank per-epoch | `dataset.py:453` |
| 模型结构 | ✅ Bitwise logits parity test | `test_models_logits_equal_v5.py` |
| 初始化权重 | ✅ Rank0 broadcast | `module_utils.py:384` |
| Dropout/正则 | ⚠️ 无 per-rank dropout tracker | 隐式通过 gradient checkpointing RNG |
| Deterministic flags | ✅ 全面 | NCCL_DETERMINISTIC, FA, CUBLAS等 |

**评估：阶段一较为完整**，主要缺口是 resume 时无配置 diff 验证。

### 阶段二：单卡/单步对齐

| 检查项 | VeOmni 状态 | 证据 |
|--------|------------|------|
| Forward loss | ⚠️ 间接（通过 FSDP equiv grad_norm） | `test_fsdp_equivalence.py` |
| Activation | ❌ 无 dump/compare 工具 | — |
| Backward gradient | ❌ 无 dump/compare 工具 | — |
| Optimizer update | ✅ Round-trip test | `test_trainer_saveload.py` |
| Scheduler | ⚠️ State dict save/load 但无独立测试 | — |
| Loss scaling | N/A (无 GradScaler, FSDP2 native) | — |
| Tensor dump | ❌ 无 | — |
| Operator-level compare | ✅ 算子级 numerical test | `test_kernel_registry_numerical.py` |

**评估：阶段二在算子层面做得好（bitwise/allclose），但缺少中间层（activation/gradient dump）的调试工具**。

### 阶段三：多步/分布式/长稳对齐

| 检查项 | VeOmni 状态 | 证据 |
|--------|------------|------|
| Loss curve | ⚠️ e2e test 比较两种并行配置的 curve，但无 golden baseline | `test_e2e_parallel.py` |
| Checkpoint resume | ✅ 权重+optimizer bitwise | `checkpoint_verification_utils.py` |
| DP correctness | ✅ Single-GPU vs FSDP2 | `test_fsdp_equivalence.py` |
| SP correctness | ✅ SP=1 vs SP=2 e2e | `test_e2e_parallel.py` |
| EP correctness | ✅ EP=1 vs EP=2 e2e | 同上 |
| TP correctness | ❌ TP 未实现 | NotImplementedError |
| PP correctness | ❌ PP 未实现 | NotImplementedError |
| Gradient accumulation | ✅ HSDP all-reduce delay | `_configure_hsdp_allreduce()` |
| Communication collectives | ⚠️ NPU patch | `hccl_premul_sum.py` |
| Mixed precision stability | ✅ reduce_dtype=FP32 + TF32 disabled | Config defaults |
| NaN/Inf monitoring | ❌ 无训练循环级检测 | — |
| CI regression | ✅ GPU + NPU dual CI | `gpu_unit_tests.yml`, `gpu_e2e_test.yml` |

**评估：阶段三在已实现的并行维度（FSDP、SP、EP）上有系统性验证，但缺少 golden baseline 和 NaN guard。**

---

## 7. 可复用设计模式

### Pattern 1: Bitwise Logits Parity Gate

| 属性 | 内容 |
|------|------|
| **设计目标** | 验证 VeOmni patched model 与 HuggingFace 原始模型输出完全一致 |
| **源码位置** | `tests/models/test_models_logits_equal_v5.py` |
| **工作流程** | 1) 加载 HF 模型 + VeOmni patched 模型; 2) 相同输入; 3) `torch.equal(logits_hf, logits_ve)` 断言 |
| **优点** | 零容差，任何 patch 引入的精度偏差都会被捕获 |
| **局限** | 只能在 FP32+eager 模式下保证 bitwise equal；BF16 下需放宽 tolerance |
| **迁移方式** | 对任何 model wrapper/patch，维护一个 "原始 vs 修改" bitwise 对比测试 |

### Pattern 2: Single-GPU vs Distributed Equivalence Test

| 属性 | 内容 |
|------|------|
| **设计目标** | 验证分布式训练不引入数值偏差 |
| **源码位置** | `tests/distributed/test_fsdp_equivalence.py` |
| **工作流程** | 1) 1GPU 训练 N 步记录 grad_norm; 2) FSDP2 2GPU 训练同样 N 步; 3) 比较 grad_norm series |
| **优点** | Proxy metric (grad_norm) 对 FSDP-induced errors 敏感；参数化覆盖多模型 |
| **局限** | grad_norm 是 aggregate，可能 miss 局部偏差；tolerance 1e-1 较松 |
| **迁移方式** | 核心思路：始终维护 "无并行 baseline" → "有并行 target" 的自动对比 CI |

### Pattern 3: Batch Invariant Ops 体系

| 属性 | 内容 |
|------|------|
| **设计目标** | 保证 matmul、RMSNorm、log_softmax 结果与 batch size 无关 |
| **源码位置** | `veomni/ops/batch_invariant_ops/`, `veomni/ops/kernels/rms_norm/triton_batch_invariant.py`, `veomni/ops/kernels/rotary/triton_deterministic.py` |
| **工作流程** | `set_batch_invariant_mode(True)` 作为 context manager 包裹 forward+backward，monkey-patch `aten::mm/addmm/log_softmax/mean` |
| **优点** | 解决了 RL 训练中 actor/rollout batch size 不同导致的数值 drift（DeepSeek V3 特有需求） |
| **局限** | 性能开销（FP32 accumulation + 可能更多 kernel launches）；只覆盖有限算子 |
| **迁移方式** | 对需要 batch-invariance 的场景（RL、serving vs training parity），用 Triton FP32 accumulation kernel 替换默认 CUDA kernel |

### Pattern 4: MoE Router Record/Replay

| 属性 | 内容 |
|------|------|
| **设计目标** | RL 训练中 actor 和 rollout 的 MoE routing 决策完全一致 |
| **源码位置** | `veomni/utils/moe_router_replay.py` |
| **工作流程** | RECORD mode: 记录 (expert_indices, routing_weights); REPLAY mode: 跳过 router forward，直接使用记录的结果 |
| **优点** | 解决 BF16 router softmax 在不同 context 下 top-k 翻转问题；验证 RECORD bitwise == no-RR |
| **局限** | 额外内存开销（存储 routing tensors）；只适用于 decode/generation 场景 |
| **迁移方式** | 对任何含 stochastic routing 的模型（MoE、early exit），提供 Record/Replay hook |

### Pattern 5: Checkpoint Round-Trip Bitwise Verification

| 属性 | 内容 |
|------|------|
| **设计目标** | 验证 save → load → use 全路径权重零精度损失 |
| **源码位置** | `tests/checkpoints/checkpoint_verification_utils.py` |
| **工作流程** | DCP save → DCP load → convert to bf16 → `torch.testing.assert_close(rtol=0, atol=0)` vs original |
| **优点** | Bitwise exact match，任何精度损失都会被发现 |
| **局限** | 只验证 model weights，optimizer state 验证在另一路径 |
| **迁移方式** | 对所有 checkpoint format 转换（DCP→HF, safetensors, DeepSpeed→PyTorch），都加 bitwise round-trip test |

### Pattern 6: Hardware-Adaptive Tolerance

| 属性 | 内容 |
|------|------|
| **设计目标** | 不同 GPU 架构有不同 ULP rounding 行为，tolerance 需适配 |
| **源码位置** | `tests/ops/test_fused_moe_split_vs_merged.py:145` |
| **工作流程** | `fwd_atol = 4e-3 if is_sm90_or_above() else 3.2e-2` |
| **优点** | 避免 Hopper vs Ada/L20 硬件差异导致的 CI flake |
| **局限** | 需要在多种硬件上采集数据确定 tolerance bound |
| **迁移方式** | 对 kernel-level 精度测试，根据 compute capability 设置不同 tolerance |

### Pattern 7: Implicit CUDA Sync Detection Gate

| 属性 | 内容 |
|------|------|
| **设计目标** | 检测 model forward 中的隐式 GPU-CPU 同步（性能和精度 hazard） |
| **源码位置** | `tests/models/test_model_forward_no_implicit_sync.py` |
| **工作流程** | `torch.cuda.set_sync_debug_mode("warn")` → run forward → 检查无非白名单 sync |
| **优点** | 自动捕获 `.item()`, `.tolist()`, `print(tensor)` 等隐式同步 |
| **局限** | 白名单维护成本；只检测 forward path |
| **迁移方式** | 对所有 production model forward path 维护 "sync-free" ratchet test |

---

## 8. 缺口分析与改造建议

### P0: 必须补齐

#### 8.1 GPU RNG State 写入 Checkpoint

| 属性 | 内容 |
|------|------|
| **问题** | `checkpoint_callback.py` 只保存 `torch.get_rng_state()` (CPU)，GPU RNG state 未保存 |
| **为什么重要** | Resume 后 dropout/stochastic ops 产生不同序列，training trajectory 不可复现 |
| **当前部分实现** | Gradient checkpointing 内部正确 save/restore GPU RNG（`distributed/checkpoint.py`） |
| **建议设计** | 在 `_save_checkpoint` 中增加 `"cuda_rng_state": torch.cuda.get_rng_state()` (per-rank)，或通过 `get_device_rng_state()` 抽象 |
| **涉及模块** | `veomni/trainer/callbacks/checkpoint_callback.py` |
| **预期收益** | Checkpoint resume 后训练 trajectory 完全可复现（在 full determinism 模式下） |

#### 8.2 Training Loop NaN/Inf Guard

| 属性 | 内容 |
|------|------|
| **问题** | 训练循环无 NaN/Inf 检测机制，异常梯度会静默进入 optimizer |
| **为什么重要** | BF16 训练容易出现 loss spike → NaN gradient → weight corruption → 需要手动 rollback |
| **当前部分实现** | Model-level `torch.isnan` (SeedOmni, WAN) 是局部修补 |
| **建议设计** | 在 `veomni_clip_grad_norm` 后检查 `torch.isfinite(grad_norm)`，选项: skip step / rollback / raise |
| **涉及模块** | `veomni/trainer/base.py`, `veomni/distributed/fsdp2/clip_grad_norm.py` |
| **预期收益** | 早发现 NaN/Inf，避免 weight corruption，减少需要 rollback 的 step 数 |

#### 8.3 Golden Loss Baseline Regression Test

| 属性 | 内容 |
|------|------|
| **问题** | 无 pre-committed golden loss value，无法检测 "loss 缓慢 drift" 类 regression |
| **为什么重要** | 算子升级/kernel 替换可能引入微小精度变化，在单步看不到差异但累积影响训练质量 |
| **当前部分实现** | `test_e2e_parallel.py` 比较两种配置 loss curve，但没有 "绝对 golden" |
| **建议设计** | 选定小模型 + fixed seed + fixed 100 tokens → 训练 50 步 → commit loss@step50 as golden → CI 验证 |
| **涉及模块** | `tests/e2e/`, 新增 `tests/regression/` |
| **预期收益** | 任何影响训练数值的 PR 都会在 CI 中被捕获 |

### P1: 强烈建议补齐

#### 8.4 Activation/Gradient Dump 工具

| 属性 | 内容 |
|------|------|
| **问题** | 无系统性 activation/gradient dump 能力，精度问题定位依赖经验 |
| **为什么重要** | 精度问题的 root cause 通常在某一层，需要逐层比对 |
| **当前部分实现** | `MoE monitor` 用 forward hook 提取 routing info，但无 general-purpose dump |
| **建议设计** | Hook-based `ActivationRecorder(module, step_range)` → dump to file → `compare_activations(dump1, dump2)` |
| **涉及模块** | 新增 `veomni/utils/precision_debug.py` |
| **预期收益** | 将精度问题定位从"猜测"降维到"逐层二分" |

#### 8.5 Config Diff on Resume

| 属性 | 内容 |
|------|------|
| **问题** | Resume 时不验证当前配置与 checkpoint 中保存的配置是否一致 |
| **为什么重要** | 误改 mixed precision 设置、learning rate、batch size 后 resume 会产生不可预期的结果 |
| **当前部分实现** | `save_args()` 保存配置，但 resume 时不读取对比 |
| **建议设计** | Resume 时加载 saved args → diff with current args → warning/error for critical fields |
| **涉及模块** | `veomni/trainer/callbacks/checkpoint_callback.py` |
| **预期收益** | 防止配置变更导致的隐性精度问题 |

#### 8.6 Per-Rank Distributed RNG Tracker

| 属性 | 内容 |
|------|------|
| **问题** | 所有 rank 使用相同 seed，dropout 产生相同 mask（DP ranks 应该不同） |
| **为什么重要** | 虽然 LLM 训练通常 dropout=0，但 VLM/DiT/pretrain 场景可能使用 dropout |
| **当前部分实现** | Dataset RNG 有 per-rank differentiation，但 model forward 没有 |
| **建议设计** | 类似 Megatron `CudaRNGStatesTracker`：per-rank offset seed for DP, shared seed for TP |
| **涉及模块** | `veomni/distributed/parallel_state.py`, `veomni/utils/helper.py` |
| **预期收益** | Dropout-heavy 训练中各 rank 正确 regularization |

### P2: 长期优化项

#### 8.7 自动 Bisect 工具

| 属性 | 内容 |
|------|------|
| **问题** | 精度 regression 定位依赖人工 git bisect |
| **建议设计** | `scripts/precision_bisect.py` — 给定 golden loss + test script，自动 git bisect 找到第一个引入偏差的 commit |
| **预期收益** | 将 regression 定位从 O(n) 降到 O(log n) |

#### 8.8 跨硬件数值对比 CI

| 属性 | 内容 |
|------|------|
| **问题** | GPU 和 NPU CI 独立运行，无 cross-platform 数值对比 |
| **建议设计** | 选定 small model + 10 steps → 分别在 GPU/NPU 运行 → 比较 loss curve |
| **预期收益** | 发现 platform-specific 精度 divergence |

#### 8.9 FP8 Training 支持

| 属性 | 内容 |
|------|------|
| **问题** | 当前仅有 FP8 推理权重转换（`fp8_cast_bf16.py`），无 FP8 训练路径 |
| **建议设计** | FSDP2 + FP8 `MixedPrecisionPolicy`（需 PyTorch 2.9+ FP8 support） |
| **预期收益** | H100/H200 上 2x throughput with bounded precision loss |

#### 8.10 Operator-Level Gradient Correctness (torch.autograd.gradcheck)

| 属性 | 内容 |
|------|------|
| **问题** | 无 numerical gradient check（有限差分法验证 backward 正确性） |
| **建议设计** | 对 custom autograd functions（ReduceLoss, chunk_logprobs, batch_invariant_ops）增加 `gradcheck` test |
| **预期收益** | 验证自定义 backward 实现的数值正确性 |

---

## 9. 推荐学习路线

### 第 1 步：读文档和配置

1. 本报告（精度对齐全景）
2. `veomni/arguments/arguments_types.py` — 理解所有精度相关配置
3. `configs/text/qwen3_5_sft.yaml` — 实际生产配置
4. `.github/workflows/gpu_e2e_test.yml` — CI 如何验证精度

### 第 2 步：跑测试

```bash
# 算子级精度测试
pytest tests/ops/test_kernel_registry_numerical.py -v
pytest tests/ops/test_fused_load_balancing_loss.py -v

# Model patch bitwise parity
pytest tests/models/test_models_logits_equal_v5.py -v -k "qwen3"

# Chunk logprobs bitwise parity (需要 batch_invariant_mode)
pytest tests/ops/test_chunk_logprobs.py -v

# FSDP equivalence (需要 2+ GPU)
torchrun --nproc_per_node=2 -m pytest tests/distributed/test_fsdp_equivalence.py -v

# E2E parallel alignment (需要 8 GPU)
pytest tests/e2e/test_e2e_parallel.py -v -k "text"
```

### 第 3 步：读源码（按精度对齐重要性排序）

| 顺序 | 文件 | 关注点 |
|------|------|--------|
| 1 | `veomni/utils/helper.py` | `enable_full_determinism`, `enable_high_precision_for_bf16`, `set_seed` |
| 2 | `veomni/utils/loss_utils.py` | `mean_global_loss` — 全局 loss normalization |
| 3 | `veomni/distributed/sequence_parallel/loss.py` | `ReduceLoss` autograd function |
| 4 | `veomni/ops/batch_invariant_ops/batch_invariant_ops.py` | Triton FP32 accumulation kernel |
| 5 | `veomni/distributed/torch_parallelize.py` | FSDP2 MP policy + EP gradient divide |
| 6 | `veomni/distributed/fsdp2/clip_grad_norm.py` | FP32 grad norm + two-stage EP reduction |
| 7 | `veomni/trainer/base.py` | 训练循环 + RNG 初始化 |
| 8 | `veomni/distributed/checkpoint.py` | Gradient checkpointing RNG save/restore |
| 9 | `veomni/utils/moe_router_replay.py` | Record/Replay 设计 |
| 10 | `tests/distributed/test_fsdp_equivalence.py` | Equivalence test 设计 |
| 11 | `tests/models/test_models_logits_equal_v5.py` | Bitwise parity test 设计 |
| 12 | `tests/e2e/test_e2e_parallel.py` + `utils.py` | E2E loss curve 对比 |

### 第 4 步：复现关键 Commit 中的问题

| Commit | 问题 | 复现方法 |
|--------|------|---------|
| `5fb669f` | SP loss 错误 | SP=2 训练，对比 loss 与 SP=1 |
| `4011a06` | BF16 MoE dtype divergence | 加载 HF MoE 模型，BF16 forward，对比 logits |
| `3290fe3`+`b6efb6e` | Cross-EP grad_norm drift | EP=1 vs EP=2 多次运行，统计 grad_norm variance |
| `aa1578c` | Optimizer state 缺失 | 构造含 unused 参数的 MoE 模型，save/load |

### 第 5 步：抽象设计模式

将上述 Pattern 1-7 整理为可复用组件：
1. `ModelPatchParity` — 验证 patch 后 bitwise equal
2. `DistributedEquivalenceTest` — 单卡 vs 分布式
3. `BatchInvariantMode` — Batch-invariant 算子替换
4. `RouterReplay` — Stochastic routing record/replay
5. `CheckpointRoundTrip` — Bitwise save/load 验证
6. `HardwareAdaptiveTolerance` — 硬件感知 tolerance
7. `ImplicitSyncGate` — 隐式同步检测

### 第 6 步：迁移到自己的训练系统

1. **立即可迁移**：`enable_full_determinism()` 函数（~30 行），直接移植
2. **低成本迁移**：`mean_global_loss()` 模式 — 确保你的 loss normalization 考虑所有并行维度
3. **中等成本**：FSDP equivalence test pattern — 需要你的 model 和 data pipeline 支持 1GPU 模式
4. **高成本但高价值**：Batch-invariant ops — 需要 Triton kernel 开发
5. **需要 RL 场景才有意义**：MoE router replay

---

## 10. 对自研分布式训练系统的迁移建议

### 10.1 最小可行精度对齐体系（MVP）

如果只能做 3 件事：

1. **Golden loss regression test** — 选一个小模型(125M)、fixed seed、fixed 100 tokens，训练 50 步，commit loss@step50。每 PR 验证。
2. **`enable_high_precision_for_bf16()` equivalent** — 在你的系统中全局禁用 TF32 + BF16 reduced precision reduction。
3. **Single-GPU vs distributed equivalence test** — 选你最常用的并行配置，验证 grad_norm 在 tolerance 内。

### 10.2 进阶体系

4. **Operator-level bitwise test** — 对每个 custom kernel，有 eager reference + `assert_close`
5. **Checkpoint round-trip** — save → load → `assert_close(rtol=0, atol=0)`
6. **NaN/Inf guard** — `if not torch.isfinite(grad_norm): skip_step()`
7. **Activation dump hook** — `register_forward_hook` + configurable module pattern

### 10.3 专家体系

8. **Batch-invariant mode** — 如果你做 RL/RLHF
9. **MoE router replay** — 如果你训练 MoE
10. **Cross-hardware CI** — 如果你支持多种加速器
11. **Auto bisect** — 如果你有频繁的精度 regression

---

## Appendix A. 检索关键词与命令记录

### 源码搜索关键词（已执行）

| 关键词组 | 状态 | 发现 |
|---------|------|------|
| allclose, rtol, atol, tolerance | ✅ | 大量（见 §3） |
| nan, inf, overflow | ✅ | 零散 model-level |
| fp32, tf32, fp16, bf16, fp8 | ✅ | 完整 MP 体系 |
| deterministic, determinism | ✅ | `enable_full_determinism` + batch_invariant |
| seed, random, rng, manual_seed | ✅ | `set_seed`, `SeedSequence` |
| golden, baseline, expected | ✅ | **未发现** golden values |
| loss, loss_fn, compute_loss | ✅ | `mean_global_loss`, `reduce_sequence_parallel_loss` |
| gradient, grad, clip_grad | ✅ | FP32 grad norm, unified clipping |
| dump, tensor_dump, activation_dump | ✅ | **未发现** general dump tools |
| all_reduce, reduce_scatter | ✅ | HCCL premul_sum patch |
| checkpoint, resume | ✅ | DCP + bitwise verification |
| GradScaler, loss_scale | ✅ | **未发现**（正确，FSDP2 不需要） |
| detect_anomaly | ✅ | **未发现** |
| gradcheck | ✅ | **未发现** |
| regression (as test category) | ✅ | **未发现**（无专门 regression test 目录） |
| batch_invariant | ✅ | 完整体系 |
| parity, bitwise, equivalence | ✅ | 多个 test |
| router, replay | ✅ | MoE router replay |

### Git 搜索命令（已执行）

```bash
git log --oneline --all --grep="precision"      # 5 结果
git log --oneline --all --grep="accuracy"       # 0 结果
git log --oneline --all --grep="determin"       # 9 结果
git log --oneline --all --grep="golden"         # 1 结果 (无关)
git log --oneline --all --grep="numerical"      # 2 结果
git log --oneline --all --grep="loss"           # 40+ 结果
git log --oneline --all --grep="bf16\|fp16"     # 11 结果
git log --oneline --all --grep="checkpoint"     # 30+ 结果
git log --oneline --all --grep="seed\|rng"      # 9 结果
git log --oneline --all --grep="gradient"       # 15+ 结果
git log --oneline --all --grep="all_reduce"     # 12 结果
git log --oneline --all --grep="parallel"       # 40+ 结果
git log --oneline --all --grep="parity\|bitwise\|equivalence" # 14 结果
git log --oneline --all --grep="batch_invariant" # 0 结果 (files exist but no commits with this in message)
git log --oneline --all --grep="NaN\|nan"       # 2 结果
git log --oneline --all --grep="replay\|router" # 8 结果
git log --oneline --all --grep="sync"           # 20+ 结果
```

---

## Appendix B. 关键文件清单

### 精度控制核心文件

| 文件 | 功能 | 行数 |
|------|------|------|
| `veomni/utils/helper.py` | seed, determinism, high_precision | ~440 |
| `veomni/utils/loss_utils.py` | mean_global_loss | ~90 |
| `veomni/distributed/sequence_parallel/loss.py` | SP loss reduction | ~65 |
| `veomni/distributed/torch_parallelize.py` | FSDP2 MP + EP gradient | ~450 |
| `veomni/distributed/fsdp2/clip_grad_norm.py` | FP32 grad norm | ~200 |
| `veomni/distributed/checkpoint.py` | Gradient ckpt RNG | ~135 |
| `veomni/ops/batch_invariant_ops/batch_invariant_ops.py` | Batch-invariant Triton kernels | ~200 |
| `veomni/ops/kernels/rotary/triton_deterministic.py` | Deterministic RoPE | — |
| `veomni/ops/kernels/rms_norm/triton_batch_invariant.py` | Batch-invariant RMSNorm | — |
| `veomni/ops/platform/npu/hccl_premul_sum.py` | NPU all_reduce fix | — |
| `veomni/utils/moe_router_replay.py` | MoE routing record/replay | ~220 |
| `veomni/trainer/callbacks/checkpoint_callback.py` | Checkpoint save/load | — |
| `veomni/arguments/arguments_types.py` | MixedPrecisionConfig | 241-305 |

### 精度测试文件

| 文件 | 验证类型 | Tolerance |
|------|---------|-----------|
| `tests/models/test_models_logits_equal_v5.py` | Bitwise HF parity | `torch.equal` |
| `tests/models/test_return_log_probs_e2e.py` | Log-prob bitwise parity | `torch.equal` |
| `tests/models/test_moe_router_replay_invariants.py` | Router RECORD bitwise == baseline | `torch.equal` |
| `tests/ops/test_kernel_registry_numerical.py` | Kernel vs eager | atol 1e-4 ~ 5e-2 |
| `tests/ops/test_fused_moe_split_vs_merged.py` | MoE split vs merged | Hardware-adaptive |
| `tests/ops/test_fused_load_balancing_loss.py` | Triton LB loss + determinism | 1e-4 + bitwise |
| `tests/ops/test_chunk_logprobs.py` | Chunk vs reference | Bitwise (rtol=0) |
| `tests/ops/test_chunk_topk_distill.py` | Distill loss parity | Bitwise + 1e-5 |
| `tests/ops/test_flash_attn_varlen_padding.py` | Flash attn padding | Bitwise |
| `tests/ops/test_seqcls_loss.py` | SeqCls loss + NaN edge | 1e-6 ~ 2e-3 |
| `tests/distributed/test_fsdp_equivalence.py` | 1GPU vs FSDP2 | rtol/atol 1e-1 |
| `tests/e2e/test_e2e_parallel.py` | SP/EP parallel align | rtol/atol 1e-1 |
| `tests/parallel/ulysses/test_ulysses.py` | SP attention parity | 1e-5 ~ 2e-3 |
| `tests/parallel/ulysses/test_qwen3_5_gated_deltanet_ulysses.py` | GatedDeltaNet SP + determinism | 100 repeats |
| `tests/checkpoints/checkpoint_verification_utils.py` | Checkpoint bitwise | rtol=0, atol=0 |
| `tests/checkpoints/test_trainer_saveload.py` | Save/load round-trip | assert_close default |
| `tests/optim/test_muon_fsdp2_parity.py` | Muon optimizer + FSDP2 | — |
| `tests/data/test_multisource_dataset.py` | Dataset determinism | Equal sequence |

### CI Workflow 文件

| 文件 | 范围 |
|------|------|
| `.github/workflows/gpu_unit_tests.yml` | L20-8, parallel/models/ops/ckpt/data/optim |
| `.github/workflows/gpu_e2e_test.yml` | E2E parallel + FSDP equiv + DiT |
| `.github/workflows/npu_unit_tests.yml` | NPU unit tests |
| `.github/workflows/npu_e2e_test.yml` | NPU e2e tests |
| `.github/workflows/check_patchgen.yml` | Patchgen determinism |
| `.github/workflows/device_api_check.yml` | No hardcoded "cuda" |

---

## Appendix C. 关键 Commits / PR / Issues 清单

| Commit | 日期 | 标题 | 精度相关分类 |
|--------|------|------|-------------|
| `4011a06` | 2026-04-17 | bitwise-equal HF vs veomni logits; fix bf16 MoE dtype divergences (#670) | Model patch parity |
| `30af894` | 2026-04-01 | add dummy_forward and FSDP equivalence tests (#620) | Distributed correctness |
| `aa1578c` | 2026-05-26 | regression test for missing optimizer state save+load | Checkpoint correctness |
| `3290fe3` | 2026-05-05 | relax cross-EP grad_norm tolerance for MoE e2e tests | EP tolerance |
| `b6efb6e` | 2026-05-05 | keep strict MoE e2e tolerance; document L20 cross-EP flake | EP tolerance (revert) |
| `ac22e25` | 2026-05-15 | add MoE router replay hook for RL training frameworks (#719) | RL determinism |
| `5fb669f` | 2025-12-29 | loss when sp enabled (#339) | SP loss normalization |
| `dee70a1` | 2025-12-29 | fix the loss calculation under sp | SP loss (initial fix) |
| `8447a8d` | 2025-12-23 | support enable_full_determinism without CUDA_LAUNCH_BLOCKING (#318) | Determinism |
| `f331f6f` | 2025-12-06 | unified veomni grad norm clipping (#205) | Gradient precision |
| `b1d6bf5` | 2026-02-09 | add enable_high_precision_for_bf16() in trainer | TF32 control |
| `946ccca` | 2026-02-08 | Add enable_high_precision_for_bf16 to all trainers missing it (#465) | TF32 control (coverage) |
| `2b21f29` | 2026-04-08 | add mixed_precision in fsdp_config (#627) | MP configuration |
| `3bf42c3` | 2026-05-05 | enable full determinism in trainer save/load test | Test determinism |
| `64557a8` | 2026-05-01 | qwen3_moe router double-softmax and idempotent _init_weights wrap (#715) | Model correctness |
| `90d4bc2` | 2026-05-20 | match HF 5.8 router weight dtype in Qwen3-MoE patch | Dtype correctness |
| `f1e5886` | 2026-05-28 | harden DCP checkpoint save/load against multi-node broadcast race | Checkpoint race |
| `b7ced20` | — | force CUDA sync inside DCP Metadata broadcast to close multi-node race | Distributed sync |
| `b0e38dc` | — | the issue where NPU does not support the PreSumMul operation (#263) | NPU communication |
| `c0b24b2` | 2025-11-12 | refactor fsdp2 grad norm clipping (#185) | Gradient precision |
| `18bf1c6` | 2025-10-23 | fix gradient_checkpoint in fsdp2 | GC correctness |
| `534e4f7` | — | add bitwise logits-equal tests for v5 models (#722) | Model parity (v5) |
| `2b889db` | — | add bitwise logits-equal tests for transformers v4 models (#721) | Model parity (v4) |
| `2946213` | 2026-03-31 | guard ReduceLoss against zero-division when SP group has no valid tokens (#618) | NaN prevention |
| `bfe4b65` | 2026-02-06 | Exclude padding batches from gradient updates (#455) | Gradient correctness |
| `1a2f973` | — | add patchgen CI check to prevent generated code drift | Code generation determinism |
| `fe5d65d` | — | use exact match for HF checkpoint verification (#440) | Checkpoint verification |
| `7e57748` | 2026-01-06 | refactor merge_dcp_to_hf.py and add checkpoint verification (#272) | Checkpoint verification |

---

*报告生成时间: 2026-06-03*
*分析范围: VeOmni main branch (commit 0d7a48e 及历史)*
*分析方法: 三 Subagent 并行搜索 + Git history 交叉验证*
*证据标准: 每个结论均有文件路径/commit hash 支撑；"未发现"明确标注*
