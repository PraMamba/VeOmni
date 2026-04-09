# CLAUDE.md - VeOmni

## WHAT: Project Overview

VeOmni is a versatile, modular framework for scaling any modality model training
(language, vision, audio, diffusion, omni-models) across various accelerators
(GPUs, NPUs). Developed by ByteDance Seed Team.

**Tech Stack**: Python 3.11 | PyTorch 2.9+ | FSDP/FSDP2 | Flash Attention 2/3/4

**Core Directories**:

- `veomni/` - Main package
  - `arguments/` - Config dataclasses, CLI/YAML parser
  - `trainer/` - BaseTrainer + specialized trainers (Text, VLM, DiT, DPO, RL)
    - `callbacks/` - Checkpoint, profiling, WandB, evaluation, MoE monitoring
  - `models/` - Registry-based model loading, HF adaptation, patching
    - `transformers/` - 21 model families (Qwen2/3, Llama, DeepSeek, Flux, Janus, etc.)
    - `diffusers/` - Diffusion model support
    - `seed_omni/` - Omni model family
  - `data/` - Dataset registry, transforms, collators, dynamic batching
    - `multimodal/` - Image/video/audio preprocessing
  - `distributed/` - FSDP1/2, Sequence Parallel (Ulysses), Expert Parallel
    - `fsdp/` - FSDP1 initialization and extensions
    - `fsdp2/` - FSDP2 composable API
    - `sequence_parallel/` - Ulysses SP (sync + async modes)
    - `moe/` - MoE layer, token routing, all-to-all communication
  - `ops/` - Fused kernels (flash-attn, cross-entropy, MoE, group GEMM)
    - `npu_patch/` - NPU-specific operator patches
    - `batch_invariant_ops/` - Batch-invariant operations (for DeepSeek V3)
  - `checkpoint/` - DCP (Distributed Checkpoint) management
  - `optim/` - Optimizer and LR scheduler builders
  - `patchgen/` - Code generation for model patches (transformers v5)
  - `utils/` - Logging, device abstraction, registry, helpers
- `tasks/` - Training entry points
- `configs/` - YAML configuration files (text, multimodal, dit)
- `scripts/` - Utility scripts (model download, checkpoint conversion, profiling)
- `tests/` - Test suite (unit, e2e, ops, parallel, checkpoint)
- `docs/` - MkDocs documentation

## WHY: Purpose

- Enable efficient distributed training for any modality at scale (up to 671B params)
- PyTorch-native approach: FSDP/FSDP2, DeviceMesh, DTensor -- no HuggingFace Trainer
- Multi-accelerator: GPU (CUDA) and NPU (Ascend) from the same codebase
- Registry-based extensibility: models, datasets, transforms, and checkpointers are independently pluggable

## HOW: Core Commands

```bash
# Setup
uv sync --extra gpu --extra audio --dev
source .venv/bin/activate

# Format and lint (run before committing)
make style                  # ruff check --fix && ruff format
make quality                # ruff check && ruff format --check
make commit                 # pre-commit install && pre-commit run --all-files

# Run tests
pytest tests/               # All tests
pytest tests/models/        # Specific module
pytest tests/e2e/           # End-to-end

# Patch generation
make patchgen               # Generate model patches
make check-patchgen         # Validate patch consistency

# Training launch
./train.sh configs/text/qwen3.yaml --train.optimizer.lr 1e-5
```

## Boundaries

### Constraints

- Python `>=3.11, <3.12`; `transformers==4.57.3` (stable), v5 is experimental
- Distributed training requires multi-GPU/NPU hardware; e2e tests need L20x8 or equivalent
- NPU async mode: supports RMSNorm but NOT LayerNorm
- TP and PP are declared in config but raise `NotImplementedError`

### Always Do

- Read relevant files before modifying code
- Run `make commit` (pre-commit) before committing
- Follow existing code patterns in the same module
- Use `veomni/utils/device.py` abstractions -- never hardcode `torch.cuda.*`, `"cuda"`, or `"nccl"`
- Use `veomni/utils/logging.get_logger(__name__)` for logging, not `print()` or stdlib `logging`
- Guard heavy optional imports (`flash_attn`, `triton`, `liger_kernel`, `diffusers`, `megatron.energon`) inside functions
- Add tests for new functionality
- Use `raise ValueError` for config validation, not bare `assert`

### Ask First

- Modifying config dataclasses in `veomni/arguments/arguments_types.py`
- Adding new dependencies to `pyproject.toml`
- Changing the training loop in `veomni/trainer/base.py`
- Modifying parallel state or device mesh construction
- Deleting or renaming public APIs or registry keys
- Running GPU/distributed tests (check GPU first: `python -c "import torch; print(torch.cuda.is_available())"`)

### Never Do

- Hardcode `torch.cuda.*`, `"cuda"`, or `"nccl"` (enforced by CI `device_api_check.yml`)
- Subclass `BaseTrainer` -- use composition via `__new__` pattern
- Use wildcard imports (`from x import *`) except in `__init__.py`
- Skip pre-commit hooks
- Use `print()` for output -- use the logger
- Import heavy packages at module level
- Create global process groups in module-level code
- Guess cluster configs or rebuild CUDA/driver stacks

## Architecture Quick Reference

### Trainer System (Composition, Not Inheritance)

Specialized trainers compose `BaseTrainer` via `__new__`, calling its private helpers
step-by-step for full control over initialization order:

```python
class TextTrainer:
    def __init__(self, args):
        self.base = BaseTrainer.__new__(BaseTrainer)  # bypass __init__
        self.base.args = args
        self.base._setup()
        self.base._build_model()
        self._build_model_assets()      # custom override
        self._build_data_transform()    # custom override
        self.base._build_dataset()
        # ...
```

| Trainer | File | Entry Point | Arguments |
|---------|------|-------------|-----------|
| TextTrainer | `veomni/trainer/text_trainer.py` | `tasks/train_text.py` | `VeOmniArguments` |
| VLMTrainer | `veomni/trainer/vlm_trainer.py` | `tasks/train_vlm.py` | `VeOmniVLMArguments` |
| DiTTrainer | `veomni/trainer/dit_trainer.py` | `tasks/train_dit.py` | `VeOmniDiTArguments` |
| TextDPOTrainer | `veomni/trainer/text_dpo_trainer.py` | `tasks/train_text_dpo.py` | `VeOmniDPOArguments` |
| BaseRLTrainer | `veomni/trainer/base_rl_trainer.py` | `tasks/train_text_rl.py` | RL-specific args |

### Registry System

All plugin points use `Registry` from `veomni/utils/registry.py`:

| Registry | Location | Purpose |
|----------|----------|---------|
| `MODELING_REGISTRY` | `veomni/models/loader.py` | Model classes |
| `MODEL_CONFIG_REGISTRY` | `veomni/models/loader.py` | Model config loaders |
| `MODEL_PROCESSOR_REGISTRY` | `veomni/models/loader.py` | Processor classes |
| `DATASET_REGISTRY` | `veomni/data/dataset.py` | Dataset implementations |
| `DATA_TRANSFORM_REGISTRY` | `veomni/data/data_transform.py` | Data transforms |
| `DATALOADER_REGISTRY` | `veomni/data/data_loader.py` | DataLoader builders |
| `CHAT_TEMPLATE_REGISTRY` | `veomni/data/chat_template.py` | Chat templates |
| `PREPROCESSOR_REGISTRY` | `veomni/data/multimodal/preprocess.py` | Multimodal preprocessors |
| `CHECKPOINTER_REGISTRY` | `veomni/checkpoint/checkpointer.py` | Checkpoint backends |

Register new components via decorator:
```python
@MODELING_REGISTRY.register("my_model")
def register_my_model(architecture: str):
    return MyModelClass
```

### Config Hierarchy

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

Parse: `parse_args(VeOmniArguments)` -- loads YAML + CLI overrides via dot notation
(`--train.optimizer.lr 1e-4`).

### Distributed Training

**Parallel dimensions** (in `veomni/distributed/parallel_state.py`):
- `dp_shard` (FSDP/ZeRO-3), `dp_replicate` (HSDP), `ulysses` (SP), `cp` (context), `tp`, `pp`, `ep`
- Device mesh dimensions conditionally included (only when `d > 1` or `name == "dp_shard"`)

**Parallelization flow** (`veomni/distributed/torch_parallelize.py`):
1. Apply ExtraParallel (EP, Embed Parallel) via `parallel_plan.py`
2. Apply FSDP1 or FSDP2 wrapping
3. Register DCP extensions
4. Load distributed weights

**Sequence Parallel** (`veomni/distributed/sequence_parallel/`):
- Sync: `gather_seq_scatter_heads` on 3D tensors BEFORE reshape to 4D
- Async: reshape to 4D FIRST, then all-to-all (overlaps with compute)

**Loss**: `reduce_sequence_parallel_loss()` takes 2 `torch.Tensor` args; actual training
loss uses `mean_global_loss()` in `veomni/utils/loss_utils.py`.

### Data Pipeline

```
YAML config → build_data_transform() → build_dataset() → build_dataloader()
                                                             ↓
                                         MainCollator (composable pipeline):
                                           PrecomputePositionIDsCollator
                                           → PackingCollator
                                           → SequenceParallelCollator (if SP)
```

**Transform types**: `plaintext`, `conversation`, `dpo`, `classification`, multimodal variants
**Dynamic batching**: token-level batching (`dyn_bsz=True`) with warmup support

### Model Patching

Two strategies:
1. **Runtime monkey-patching** (transformers v4): `apply_veomni_xxx_gpu_patch()` functions
2. **Code generation** (transformers v5): `patchgen/` DSL produces committed `generated/patched_modeling_*_gpu.py` files

### Callback System

Eight callbacks in `veomni/trainer/callbacks/`:
- `EnvironMeterCallback` -- hardware metrics
- `TqdmCallback` -- progress bar
- `WandbTraceCallback` -- W&B logging
- `ProfileTraceCallback` -- PyTorch profiler
- `CheckpointerCallback` -- DCP checkpointing
- `HuggingfaceCkptCallback` / `HFLoraCkptCallback` -- HF-format saving
- `EvaluateCallback` -- evaluation loop
- `MoERouterMonitorCallback` -- MoE load monitoring

## Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Config dataclass | `XxxConfig` | `FSDPConfig`, `OptimizerConfig` |
| Top-level args | `XxxArguments` | `TrainingArguments`, `ModelArguments` |
| Trainer class | `XxxTrainer` | `TextTrainer`, `VLMTrainer` |
| Callback class | `XxxCallback` | `CheckpointerCallback`, `TqdmCallback` |
| Registry instance | `XXX_REGISTRY` | `MODELING_REGISTRY`, `DATASET_REGISTRY` |
| Build function | `build_xxx()` | `build_dataset()`, `build_optimizer()` |
| Patch function | `apply_veomni_xxx_patch()` | `apply_veomni_qwen3_gpu_patch()` |
| Availability check | `is_xxx_available()` | `is_flash_attn_2_available()` |
| Parallel dimension | `{dim}_size/rank/group/mesh` | `dp_shard_size`, `ulysses_rank` |

## Progressive Disclosure: Detailed Guides

| Task | Reference |
|------|-----------|
| Add Model | `veomni/models/transformers/qwen3/` (example), `veomni/models/loader.py` |
| Add Dataset | `veomni/data/dataset.py`, `veomni/data/data_transform.py` |
| Add Trainer | `veomni/trainer/text_trainer.py` (example pattern) |
| Distributed Training | `veomni/distributed/torch_parallelize.py`, `veomni/distributed/parallel_state.py` |
| Model Patching | `veomni/patchgen/patch_spec.py`, `veomni/patchgen/codegen.py` |
| Config System | `veomni/arguments/arguments_types.py`, `veomni/arguments/parser.py` |
| Checkpoint | `veomni/checkpoint/dcp_checkpointer.py` |
| Fused Kernels | `veomni/ops/flash_attn/`, `veomni/ops/fused_moe/`, `veomni/ops/fused_cross_entropy/` |
| Launch Training | `train.sh` (auto-detects GPU/NPU, sets up torchrun) |

## Git Workflow

- **PR title format**: `[{modules}] {type}: {description}`
  - Modules: `misc`, `ci`, `config`, `docs`, `data`, `dist`, `omni`, `logging`, `model`, `optim`, `ckpt`, `release`, `task`, `perf`, `ops`, `parallel`
  - Types: `feat`, `fix`, `refactor`, `chore`, `test`
  - Breaking: prepend `[BREAKING]` -- e.g., `[BREAKING][parallel, model] feat: dynamic batching`
- **Commits**: Conventional Commits, ~72 chars subject, imperative voice
- **Squash**: Squash WIP commits before opening PR
- **PR requirements**: Run pre-commit, document test coverage, note hardware limitations

## Extended Configuration

See `.claude/agents/`, `.claude/commands/`, and `.claude/rules/` for specialized instructions.

### Agents

| Agent | Purpose | Activation Trigger |
|-------|---------|-------------------|
| `planner` | Implementation planning | Before multi-file changes, new features, architectural decisions |
| `code-reviewer` | Code quality review | After code changes, before committing |
| `trainer-expert` | Trainer and callback system | Trainer/callback code changes or questions |
| `model-expert` | Model loading, patching, registry | Adding models, GPU/NPU patches, patchgen |
| `data-pipeline-expert` | Data pipeline | Datasets, dataloaders, transforms, collators |
| `fsdp-expert` | FSDP/FSDP2 distributed training | Model parallelization, sharding, device mesh |
| `sequence-parallel-expert` | Ulysses Sequence Parallel | SP communication, attention, loss reduction |
| `moe-expert` | MoE and Expert Parallel | MoE layers, expert routing, fused kernels, EP |

**Stage-by-Stage Agent Guidance**:

1. **Planning Stage** (Before coding): Use `planner` for architecture design and implementation planning
2. **Implementation Stage**: Use domain experts (`trainer-expert`, `model-expert`, `data-pipeline-expert`, `fsdp-expert`, `sequence-parallel-expert`, `moe-expert`) as needed
3. **Review Stage** (After coding): Use `code-reviewer` for quality checks

### Skills (Guided Development Workflows)

- `/add-model` - Guide for adding a new HuggingFace model
- `/add-dataset` - Guide for adding a new dataset type
- `/add-unit-tests` - Guide for adding unit tests
- `/debug-distributed` - Guide for debugging distributed training issues
- `/review-pr` - PR code review with VeOmni-specific templates
- `/create-pr` - Rebase, squash, and create PR with proper title format
- `/gen-commit-msg` - Generate commit message from staged changes

### Rules (Code Quality Standards)

- `code-style.md` - Design patterns, logging, performance, naming, imports
- `distributed.md` - Device abstraction, parallel state, FSDP, SP, MoE rules
- `testing.md` - Test structure, markers, GPU skip patterns, CI workflows
- `api-config.md` - Dataclass conventions, config hierarchy, CLI integration

## Supported Models

| Category | Models |
|----------|--------|
| LLMs | DeepSeek V3, Llama 3, Qwen 2/3 (up to 72B/671B) |
| Vision-Language | Qwen2.5-VL, Qwen3-VL, QVQ (2B-72B) |
| MoE | Qwen3-MoE, Qwen3.5-MoE, Qwen3-VL-MoE |
| Diffusion | Wan 2.1-I2V (up to 14B), Flux |
| Omni | Qwen2.5-Omni, Qwen3-Omni-MoE |

## Hardware Support

- **GPU**: CUDA 12.9 (NVIDIA), tested on L20x8
- **NPU**: Ascend (Huawei), via `torch-npu`
- Device abstraction: `veomni/utils/device.py` -- all device ops must go through this

## Code Intelligence & Navigation

When navigating and understanding code:

1. **ALWAYS prefer LSP tools over text search for code relationships**:
   - Use `goToDefinition` to jump to symbol definitions
   - Use `findReferences` to find all usages across the codebase
   - Use `goToImplementation` for interfaces/abstract methods
   - Use `workspaceSymbol` to search symbols across entire project

2. **Use Grep/Glob/Read ONLY for**:
   - Text/pattern searches in comments or strings
   - Searching configuration files (JSON, YAML)
   - Exploratory "fuzzy" searches when unsure what you're looking for
   - Finding files by name patterns

3. **Workflow**:
   - First: Use LSP to understand code structure and relationships
   - Second: Use text tools only when LSP cannot help (non-code content)
   - NEVER read entire large files to find references; use LSP instead
