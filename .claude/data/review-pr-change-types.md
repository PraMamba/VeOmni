# PR Review: Change Type Detection Reference

Referenced by `.claude/commands/review-pr.md`

---

## CRITICAL Level (Must use Opus)

| Change Type           | File Path Pattern                                          | Code Pattern                                                  |
| --------------------- | ---------------------------------------------------------- | ------------------------------------------------------------- |
| **FSDP_CORE**         | `veomni/distributed/fsdp/`, `veomni/distributed/fsdp2/`   | `FSDP`, `FullyShardedDataParallel`, `fully_shard`             |
| **PARALLELIZE_CORE**  | `veomni/distributed/torch_parallelize.py`                  | `build_parallelize_model`                                     |
| **PARALLEL_STATE**    | `veomni/distributed/parallel_state.py`                     | `ParallelState`, `DeviceMesh`, `init_para_mesh`               |
| **MOE_EP**            | `veomni/distributed/moe/`                                  | `EPGroupGemm`, `token_pre_all2all`, `tokens_post_all2all`     |
| **DCP_CHECKPOINT**    | `veomni/checkpoint/`                                       | `DCPCheckpointer`, `dcp.save`, `dcp.load`                     |
| **SP_CORE**           | `veomni/distributed/sequence_parallel/`                    | `gather_seq_scatter_heads`, `AsyncUlysses`, `all_to_all`      |

## HIGH Level (Recommend Opus)

| Change Type           | File Path Pattern                         | Code Pattern                                                 |
| --------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| **DISTRIBUTED_COMM**  | -                                         | `all_reduce`, `all_gather`, `reduce_scatter`, `dist.`        |
| **TRAINER_CORE**      | `veomni/trainer/base.py`                  | `BaseTrainer`, `train_step`, `forward_backward_step`         |
| **MODEL_LOADER**      | `veomni/models/loader.py`, `auto.py`      | `MODELING_REGISTRY`, `build_foundation_model`                |
| **OPS_PATCH**         | `veomni/ops/`                             | `apply_ops_patch`, `flash_attn`, `fused_moe`                 |
| **PARALLEL_PLAN**     | `**/parallel_plan.py`                     | `ExpertParallel`, `apply_ep`, mesh configuration             |

## MEDIUM Level (Use Sonnet)

| Change Type             | File Path Pattern                    | Code Pattern                                                 |
| ----------------------- | ------------------------------------ | ------------------------------------------------------------ |
| **TENSOR_OPS**          | -                                    | `.view(`, `.reshape(`, `dtype=`, `.detach()`, `no_grad`      |
| **NUMERICAL**           | -                                    | `log(`, `softmax`, `cross_entropy`, `eps=`, `.clamp(`        |
| **TRAINER_SUB**         | `veomni/trainer/*.py` (not base)     | `TextTrainer`, `VLMTrainer`, `DITTrainer`                    |
| **API_CONFIG**          | `veomni/arguments/`                  | `@dataclass`, `__post_init__`, `field(`                      |
| **DATA_PIPELINE**       | `veomni/data/`                       | `DATASET_REGISTRY`, `DataLoader`, `Collator`                 |
| **CALLBACK**            | `veomni/trainer/callbacks/`          | `Callback`, `on_step_begin`, `on_step_end`                   |
| **PATCHGEN**            | `veomni/patchgen/`                   | `codegen`, `patch_gen_config`                                |
| **LOSS_UTILS**          | `veomni/utils/loss_utils.py`         | `mean_global_loss`, `count_loss_token`                       |

## LOW Level (Use Haiku)

| Change Type     | File Path Pattern            | Code Pattern |
| --------------- | ---------------------------- | ------------ |
| **TESTS**       | `tests/`, `test_*.py`        | -            |
| **DOCS**        | `docs/`, `*.md`              | -            |
| **CONFIG_ONLY** | `*.yaml`, `*.json`, `*.toml` | -            |

---

## Risk Linkage Rules

| Detected Change          | Auto-Linked Review                                  |
| ------------------------ | --------------------------------------------------- |
| FSDP changes             | Parallel state + checkpoint interaction check       |
| SP changes               | Attention + loss reduction + collator check         |
| MOE_EP changes           | FSDP mesh + load balancing + token alignment check  |
| PARALLEL_STATE changes   | All parallel strategy interaction check             |
| TRAINER_CORE changes     | Callback dispatch + all trainer subclass check      |
| MODEL_LOADER changes     | Registry + config resolution + weight loading check |
| OPS_PATCH changes        | Monkey-patch safety + transformers compat check     |
| DCP_CHECKPOINT changes   | FSDP state dict + distributed consistency check     |
| PATCHGEN changes         | Generated file consistency + CI check_patchgen      |

---

## Core Framework Paths (Must Use Opus)

**Distributed Core**:
- `veomni/distributed/parallel_state.py`
- `veomni/distributed/torch_parallelize.py`
- `veomni/distributed/fsdp/`
- `veomni/distributed/fsdp2/`
- `veomni/distributed/sequence_parallel/`
- `veomni/distributed/moe/`

**Checkpoint Core**:
- `veomni/checkpoint/`

**Trainer Core**:
- `veomni/trainer/base.py`
