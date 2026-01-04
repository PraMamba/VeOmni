# VeOmni Torch Distributed Checkpoint 源码深度分析

## 目录

1. [概述](#1-概述)
2. [核心架构](#2-核心架构)
3. [DCP Save 保存机制](#3-dcp-save-保存机制)
4. [DCP Load 加载机制](#4-dcp-load-加载机制)
5. [Stateful Wrapper 实现](#5-stateful-wrapper-实现)
6. [EP 维度处理](#6-ep-维度处理)
7. [FSDP1 Checkpoint 扩展](#7-fsdp1-checkpoint-扩展)
8. [Async Save 异步保存](#8-async-save-异步保存)
9. [Extra State 处理](#9-extra-state-处理)
10. [存储格式与元数据](#10-存储格式与元数据)
11. [工具函数与 API](#11-工具函数与-api)
12. [最佳实践](#12-最佳实践)
13. [限制与注意事项](#13-限制与注意事项)
14. [参考资料](#14-参考资料)

---

## 1. 概述

### 1.1 什么是 Torch Distributed Checkpoint (DCP)

Torch Distributed Checkpoint 是 PyTorch 为分布式训练提供的官方 checkpoint 解决方案，核心特性：

- **分布式保存/加载**：每个 rank 只保存/加载自己负责的参数分片，无需全局通信
- **FSDP 原生支持**：完美集成 FSDP1/FSDP2，自动处理 DTensor 的分片信息
- **灵活的存储后端**：支持 FileSystem、S3 等多种存储后端
- **Resharding 支持**：可以在不同并行度下加载 checkpoint（如从 8 GPU 的 checkpoint 恢复到 16 GPU）
- **异步保存**：支持后台异步写入，最小化训练中断时间

### 1.2 VeOmni 的 DCP 实现特性

VeOmni 在 PyTorch DCP 基础上提供了以下增强：

1. **Expert Parallelism (EP) 支持**：
   - 自动处理 EP+FSDP2 混合并行场景下的维度变换
   - 在保存时将 EP 维度转换为 DCP 可理解的格式
   - 在加载时恢复 EP 维度以匹配运行时的 DeviceMesh

2. **FSDP1 扩展**：
   - 通过 `CheckpointExtensions` 为 FSDP1 提供 EP 支持
   - 自定义 state_dict hooks 处理 EP 参数的保存/加载

3. **Extra State 管理**：
   - 独立保存 non-DCP 组件（lr_scheduler, dataloader, global_step）
   - 每个 rank 保存自己的 extra_state 到单独文件

4. **异步保存优化**：
   - 使用专用的 Gloo process group
   - 支持后台保存，训练可以继续进行

5. **统一的 Checkpointer 接口**：
   - 注册表机制支持多种 checkpoint 管理器
   - 简单的 `build_checkpointer()` API

### 1.3 文件结构

```
veomni/checkpoint/
├── checkpointer.py             # 基类和注册表
│   ├── CheckpointerBase        # 抽象基类
│   ├── CHECKPOINTER_REGISTRY   # Checkpoint 管理器注册表
│   ├── build_checkpointer()    # 构建函数
│   └── dcp_checkpointer()      # DCP 构建器
│
├── dcp_checkpointer.py         # DCP 核心实现
│   ├── ModelState              # Model 的 Stateful wrapper
│   ├── OptimizerState          # Optimizer 的 Stateful wrapper
│   ├── DistributedCheckpointer # DCP checkpoint 管理器
│   ├── drop_ep_dim()           # 移除 EP 维度
│   ├── restore_ep_dim()        # 恢复 EP 维度
│   └── dcp_to_torch_state_dict() # DCP → Torch state_dict 转换
│
veomni/utils/
├── checkpoint_utils.py         # Checkpoint 工具函数
│   ├── dcp_get_last_iteration() # 查找最新 checkpoint
│   └── get_checkpoint_path()    # 获取 checkpoint 路径
│
veomni/distributed/fsdp/
├── extension.py                # FSDP1 Checkpoint 扩展
│   ├── CheckpointExtensions    # FSDP1 的 EP 扩展
│   ├── state_dict_post_hook()  # Save 时的钩子
│   ├── load_state_dict_pre_hook() # Load 时的钩子
│   └── _shard_tensor()         # 张量分片函数
│
├── initialize.py               # FSDP 初始化
│   └── parallel_init_fsdp_fn() # 并行加载 safetensors
```

---

## 2. 核心架构

### 2.1 类层次结构

```
CheckpointerBase (ABC)
    └── DistributedCheckpointer
            ├── save() - 保存 checkpoint
            ├── load() - 加载 checkpoint
            ├── execute_save() - 执行 DCP 保存（支持 async）
            └── Helper methods:
                ├── _create_storage_writer()
                ├── _create_storage_reader()
                ├── _save_extra_state()
                └── _load_extra_state()

Stateful (PyTorch Interface)
    ├── ModelState - 包装 model，提供 state_dict() / load_state_dict()
    └── OptimizerState - 包装 optimizer，提供 state_dict() / load_state_dict()

FSDPExtensions (PyTorch FSDP1)
    └── CheckpointExtensions - FSDP1 的 EP 支持
            ├── chunk_dtensor() - DTensor 分片
            ├── state_dict_post_hook() - Save 时处理 EP
            └── load_state_dict_pre_hook() - Load 时处理 EP
```

### 2.2 DCP Save 流程图

```
[用户代码]
    ↓
build_checkpointer("dcp", "fsdp2")
    ↓
DistributedCheckpointer.save(path, state, save_async=True, global_steps=100)
    ↓
    ├─ 1. 验证 state 必须包含 "model"
    ├─ 2. 创建 checkpoint 目录: {path}/global_step_{global_steps}
    ├─ 3. 保存 extra_state 到 extra_state/extra_state_rank_{rank}.pt
    ├─ 4. 包装 state:
    │      save_state = {
    │          "model": ModelState(state["model"]),
    │          "optimizer": OptimizerState(state["model"], state["optimizer"])
    │      }
    ├─ 5. ModelState.state_dict():
    │      ├─ get_model_state_dict(model) → FSDP-aware state dict
    │      ├─ 如果是 EP+FSDP2: restore_ep_dim() 恢复 EP 维度
    │      └─ 返回包含 DTensor 的 state dict
    ├─ 6. OptimizerState.state_dict():
    │      ├─ 如果是 EP+FSDP2: 使用 MultiOptimizer.state_dict()
    │      ├─ restore_ep_dim() 恢复 EP 维度
    │      └─ 否则: get_optimizer_state_dict(model, optimizer)
    ├─ 7. 创建 FileSystemWriter:
    │      FileSystemWriter(dir, thread_count=16, single_file_per_rank=True)
    └─ 8. execute_save():
           ├─ 如果 save_async=True:
           │      ├─ 创建 Gloo process group (首次)
           │      ├─ 等待上一次 async save 完成
           │      └─ dcp.async_save(state_dict, storage_writer, process_group)
           └─ 如果 save_async=False:
                  ├─ dcp.save(state_dict, storage_writer)
                  ├─ dist.barrier()
                  └─ gc.collect() + empty_cache()
```

### 2.3 DCP Load 流程图

```
[用户代码]
    ↓
DistributedCheckpointer.load(path, state)
    ↓
    ├─ 1. 验证 state 必须包含 "model"
    ├─ 2. 包装 state:
    │      load_state = {
    │          "model": ModelState(state["model"]),
    │          "optimizer": OptimizerState(state["model"], state["optimizer"])
    │      }
    ├─ 3. 创建 FileSystemReader:
    │      FileSystemReader(checkpoint_dir)
    ├─ 4. 执行 DCP load:
    │      dcp.load(state_dict=load_state, storage_reader=reader, process_group=pg)
    │      ├─ 读取 .metadata 文件
    │      ├─ 根据 sharding plan 读取对应的 .distcp 文件
    │      └─ 调用 Stateful wrapper 的 load_state_dict()
    ├─ 5. ModelState.load_state_dict(state_dict):
    │      ├─ 如果是 EP+FSDP2: drop_ep_dim() 移除 EP 维度
    │      └─ set_model_state_dict(model, model_state_dict)
    ├─ 6. OptimizerState.load_state_dict(state_dict):
    │      ├─ 如果是 EP+FSDP2: drop_ep_dim() 移除 EP 维度
    │      ├─ 如果是 EP+FSDP2: MultiOptimizer.load_state_dict()
    │      └─ 否则: set_optimizer_state_dict(model, optimizer, optim_state_dict)
    └─ 7. 加载 extra_state:
           从 extra_state/extra_state_rank_{rank}.pt 加载
```

### 2.4 关键设计决策

#### 2.4.1 为什么需要 Stateful Wrapper？

PyTorch DCP 要求 state dict 中的值实现 `Stateful` 接口：

```python
class Stateful(Protocol):
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...
```

**原因**：
- DCP 需要在 save/load 时调用自定义逻辑（如 EP 维度转换）
- 不能直接修改 PyTorch 的 `nn.Module` 或 `Optimizer` 类
- Wrapper 模式提供了插入自定义逻辑的钩子点

#### 2.4.2 为什么需要 EP 维度转换？

在 EP+FSDP2 混合并行中：

**运行时 DTensor 结构**：
```python
# 2D DeviceMesh: [ep, ep_fsdp]
# 2D Placements: [Shard(0), Shard(1)]
# 表示参数在 EP 维度和 FSDP 维度都分片

example_param.placements = [Shard(0), Shard(1)]
example_param.device_mesh.shape = (ep_size, ep_fsdp_size)
```

**DCP 期望的结构**：
```python
# DCP 只理解 FSDP 维度，不理解 EP 维度
# 需要将 2D DTensor 转换为 1D DTensor

save 时: [Shard(0), Shard(1)] → [Shard(0)] (2D → 1D，保留 EP 信息)
load 时: [Shard(0)] → [Shard(1)] (1D → 1D，移除 EP 信息)
```

**转换必要性**：
- DCP 的 resharding 逻辑基于 DTensor placements
- 如果保存时使用 `[Shard(0), Shard(1)]`，DCP 会误以为需要在两个维度 reshard
- 通过 `restore_ep_dim()` 将 EP 信息编码到第一个维度
- 通过 `drop_ep_dim()` 在加载后移除 EP 维度，恢复 FSDP-only 的 DTensor

---

## 3. DCP Save 保存机制

### 3.1 Save API

```python
# veomni/checkpoint/dcp_checkpointer.py: 262-300
@classmethod
def save(
    cls,
    path: str,
    state: Dict[str, Any],
    save_async: bool = False,
    global_steps: int = None,
    storage_writer: Optional[FileSystemWriter] = None,
) -> None:
    """
    保存训练状态到 distributed checkpoint

    参数：
        path: 保存路径（基础目录）
        state: 要保存的状态字典，必须包含 "model"，可选 "optimizer", "extra_state"
        save_async: 是否异步保存
        global_steps: 全局步数（用于生成 global_step_{N} 子目录）
        storage_writer: 自定义存储后端（默认为 FileSystemWriter）

    State 结构：
        {
            "model": nn.Module,              # 必需
            "optimizer": Optimizer,          # 可选
            "extra_state": {                 # 可选
                "global_step": int,
                "lr_scheduler": dict,
                "train_dataloader": dict,
                ...
            }
        }
    """
```

**使用示例**：

```python
# 构建 checkpointer
from veomni.checkpoint import build_checkpointer

Checkpointer = build_checkpointer(ckpt_manager="dcp", dist_backend="fsdp2")

# 准备 state
state = {
    "model": model,
    "optimizer": optimizer,
    "extra_state": {
        "global_step": global_step,
        "lr_scheduler": lr_scheduler.state_dict(),
    }
}

# 保存 (同步)
Checkpointer.save(
    path="./checkpoints",
    state=state,
    global_steps=1000,
    save_async=False,
)
# 生成目录: ./checkpoints/global_step_1000/

# 保存 (异步)
Checkpointer.save(
    path="./checkpoints",
    state=state,
    global_steps=2000,
    save_async=True,  # 后台保存，训练继续
)
```

### 3.2 Save 流程详解

#### Step 1: 验证输入

```python
# dcp_checkpointer.py: 282-283
if "model" not in state:
    raise ValueError("Model must be provided to save a distributed checkpoint.")
```

**关键点**：
- `"model"` 是必需的，因为 DCP 需要模型的 state_dict 作为主体
- `"optimizer"` 是可选的（推理模式可能不需要 optimizer）
- `"extra_state"` 是可选的，用于保存 non-DCP 组件

#### Step 2: 创建 Checkpoint 目录

```python
# dcp_checkpointer.py: 285-286
checkpoint_dir = f"{path}/{_GLOBAL_STEP_PREFIX}{global_steps}" if global_steps else path
cls._create_checkpoint_dir(checkpoint_dir)

# _GLOBAL_STEP_PREFIX = "global_step_"
# 示例: ./checkpoints/global_step_1000/
```

**目录结构设计**：
```
checkpoints/
├── global_step_1000/
│   ├── .metadata
│   ├── __0_0.distcp
│   ├── __1_0.distcp
│   ├── ...
│   └── extra_state/
│       ├── extra_state_rank_0.pt
│       └── extra_state_rank_1.pt
├── global_step_2000/
└── global_step_3000/
```

#### Step 3: 保存 Extra State

```python
# dcp_checkpointer.py: 289
cls._save_extra_state(checkpoint_dir=checkpoint_dir, state=state)

# 实现 (Lines 407-419):
def _save_extra_state(cls, checkpoint_dir: str, state: Dict[str, Any]) -> None:
    if "extra_state" not in state:
        logger.warning_rank0("extra_state not found in state, skipping extra_state save")
        return

    extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)  # "extra_state"
    os.makedirs(extra_state_dir, exist_ok=True)
    extra_state_path = os.path.join(
        extra_state_dir,
        _EXTRA_STATE_FORMAT.format(dist.get_rank())  # "extra_state_rank_{rank}.pt"
    )
    torch.save(state["extra_state"], extra_state_path)
```

**为什么先保存 extra_state？**
- 注释中说明：保证每个保存的 model/optimizer checkpoint 都有对应的 extra_state
- 如果 DCP save 失败，至少不会留下孤立的 extra_state
- 如果 extra_state save 失败，DCP save 也不会执行

#### Step 4: 包装 State

```python
# dcp_checkpointer.py: 291-293
save_state = {"model": ModelState(state["model"])}
if "optimizer" in state:
    save_state["optimizer"] = OptimizerState(
        model=state["model"],
        optimizer=state["optimizer"]
    )
```

**包装目的**：
- 将 `nn.Module` 包装为 `ModelState`（实现 `Stateful` 接口）
- 将 `Optimizer` 包装为 `OptimizerState`（实现 `Stateful` 接口）
- DCP 会调用 wrapper 的 `state_dict()` 方法获取实际要保存的数据

#### Step 5: 创建 Storage Writer

```python
# dcp_checkpointer.py: 295-296
if storage_writer is None:
    storage_writer = cls._create_storage_writer(checkpoint_dir)

# 实现 (Lines 397-404):
def _create_storage_writer(cls, checkpoint_dir: str) -> FileSystemWriter:
    return FileSystemWriter(
        checkpoint_dir,
        thread_count=16,           # 16 个并行 I/O 线程
        single_file_per_rank=True, # 每个 rank 一个文件
        sync_files=False,          # 不强制 fsync（更快）
    )
```

**Storage Writer 配置解析**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `thread_count` | 16 | 每个 rank 的并行写入线程数，越大写入越快但内存占用越高 |
| `single_file_per_rank` | True | 每个 rank 写入一个文件（`__rank_0.distcp`），False 时每个参数一个文件 |
| `sync_files` | False | 是否在写入后调用 fsync 强制刷盘，True 更安全但更慢 |

#### Step 6: 执行 DCP Save

```python
# dcp_checkpointer.py: 298
cls.execute_save(save_state=save_state, storage_writer=storage_writer, save_async=save_async)
```

详见 [8. Async Save 异步保存](#8-async-save-异步保存)

### 3.3 ModelState.state_dict()

```python
# dcp_checkpointer.py: 56-64
@torch.no_grad()
def state_dict(self):
    """
    获取 model 的 state dict

    流程：
    1. 调用 PyTorch DCP 的 get_model_state_dict(model)
       - 自动处理 FSDP/FSDP2 的 DTensor
       - 返回的是当前 rank 负责的参数分片
    2. 如果是 EP+FSDP2，调用 restore_ep_dim() 恢复 EP 维度
    """
    model_state_dict = get_model_state_dict(model=self.model)

    if self.should_ep_aware:
        logger.info_rank0(
            "Getting model state_dict from ModelState wrapper, would restore EP dim for Experts module"
        )
        model_state_dict = self.get_state_dict_with_ep_dim_preprocess(
            model_state_dict,
            "restore"  # 恢复 EP 维度
        )

    return model_state_dict
```

**`get_model_state_dict()` 的作用**（PyTorch DCP API）：
- 对于 FSDP2：返回包含 DTensor 的 state dict，DTensor 保留分片信息
- 对于 FSDP1：调用 `model.state_dict()` 并触发 FSDP hooks
- 对于 DDP/单卡：直接返回 `model.state_dict()`

**EP 维度处理** (详见 [6. EP 维度处理](#6-ep-维度处理))：
```python
# dcp_checkpointer.py: 79-102
def get_state_dict_with_ep_dim_preprocess(self, state_dict, action):
    # action = "restore" 表示恢复 EP 维度
    for name in sorted(state_dict.keys()):
        if name in ep_fqn2spec_info and isinstance(placement, Shard):
            tensor = state_dict[name]
            tensor = restore_ep_dim(tensor, ep_fsdp_mesh)  # 2D → 1D DTensor with EP
            state_dict[name] = tensor
    return state_dict
```

### 3.4 OptimizerState.state_dict()

```python
# dcp_checkpointer.py: 121-135
def state_dict(self):
    """
    获取 optimizer 的 state dict

    EP+FSDP2 场景：
    - 使用 MultiOptimizer.state_dict()（已经是 merged + flattened）
    - 调用 restore_ep_dim() 恢复 EP 维度

    非 EP 场景：
    - 使用 PyTorch DCP 的 get_optimizer_state_dict()
    """
    if self.should_ep_aware:
        logger.info_rank0(
            "Getting optimizer state_dict from OptimizerState wrapper, would restore EP dim for Experts module"
        )
        # MultiOptimizer 只在 EP+FSDP2 时使用
        assert self.optimizer._is_multi_optimizer, (
            "EP is enabled but optimizer is not a MultiOptimizer instance"
        )
        vanilla_optim_sd = self.optimizer.state_dict()
        optim_sd_with_ep_dim = self.get_state_dict_with_ep_dim_preprocess(
            vanilla_optim_sd,
            "restore"
        )
        return optim_sd_with_ep_dim

    # 单个 torch optimizer
    sd = get_optimizer_state_dict(model=self.model, optimizers=self.optimizer)
    return sd
```

**Optimizer State Dict 结构**：

```python
{
    "state": {
        "model.layers.0.mlp.experts.gate_proj": {
            "step": 1000,                    # 标量，不分片
            "exp_avg": DTensor(...),         # 动量，需要分片
            "exp_avg_sq": DTensor(...),      # 二阶动量，需要分片
        },
        ...
    },
    "param_groups": [
        {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            ...
        }
    ]
}
```

**EP 维度处理逻辑** (dcp_checkpointer.py:153-201)：

```python
def get_state_dict_with_ep_dim_preprocess(self, state_dict, action):
    for name in sorted(state_dict.keys()):
        # 找到匹配的 EP spec
        matches = [ep_key for ep_key in ep_keys if ep_key in name]
        if not matches:
            continue  # 非 EP 参数

        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            continue  # 跳过超参数（如 "amsgrad": False）
        if tensor.ndim == 0:
            continue  # 跳过标量（如 "step": 1000）

        if action == "restore":
            tensor = restore_ep_dim(tensor, ep_fsdp_mesh)
        state_dict[name] = tensor

    return state_dict
```

**关键点**：
- `"step"` 等标量不分片，不需要 EP 维度转换
- `"exp_avg"` 等张量与参数有相同的 shape 和分片策略
- EP 参数的 optimizer state 也需要 EP 维度转换

---

## 4. DCP Load 加载机制

### 4.1 Load API

```python
# veomni/checkpoint/dcp_checkpointer.py: 303-347
@classmethod
def load(
    cls,
    path: str,
    state: Dict[str, Any],
    process_group=None,
    storage_reader: Optional[FileSystemReader] = None,
) -> Dict[str, Any]:
    """
    从 distributed checkpoint 加载训练状态

    参数：
        path: checkpoint 路径
        state: 要加载的状态字典，必须包含 "model"，可选 "optimizer", "extra_state"
        process_group: 加载时使用的 process group（默认为全局 process group）
        storage_reader: 自定义存储后端（默认为 FileSystemReader）

    返回：
        state: 加载后的状态字典（原地修改）

    State 结构：
        {
            "model": nn.Module,              # 必需（已初始化的模型）
            "optimizer": Optimizer,          # 可选（已初始化的 optimizer）
            "extra_state": {},               # 可选（空字典，会被填充）
        }
    """
```

**使用示例**：

```python
# 初始化模型和 optimizer
model = build_model(config)
optimizer = build_optimizer(model.parameters(), lr=1e-4)

# 准备 state（空的 extra_state）
state = {
    "model": model,
    "optimizer": optimizer,
    "extra_state": {},
}

# 加载 checkpoint
Checkpointer.load(
    path="./checkpoints/global_step_1000",
    state=state,
)

# state 已被原地更新
global_step = state["extra_state"]["global_step"]
lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
```

### 4.2 Load 流程详解

#### Step 1: 验证输入

```python
# dcp_checkpointer.py: 323-327
if state is None:
    raise ValueError("State dict must be provided to load a distributed checkpoint.")

if "model" not in state:
    raise ValueError("Model must be provided to load a distributed checkpoint.")
```

**关键点**：
- `state["model"]` 必须是已初始化的模型（DCP 需要知道目标 sharding 策略）
- `state["optimizer"]` 如果提供，也必须是已初始化的 optimizer
- 不能传入 `None` 或空的 state

#### Step 2: 包装 State

```python
# dcp_checkpointer.py: 329-331
load_state = {"model": ModelState(state["model"])}
if "optimizer" in state:
    load_state["optimizer"] = OptimizerState(
        model=state["model"],
        optimizer=state["optimizer"]
    )
```

**与 Save 相同的包装逻辑**，确保对称性。

#### Step 3: 创建 Storage Reader

```python
# dcp_checkpointer.py: 333-334
if storage_reader is None:
    storage_reader = cls._create_storage_reader(checkpoint_dir)

# 实现 (Lines 392-394):
def _create_storage_reader(cls, checkpoint_dir: str) -> FileSystemReader:
    return FileSystemReader(checkpoint_dir)
```

**FileSystemReader** 的工作流程：
1. 读取 `{checkpoint_dir}/.metadata` 文件
2. 解析 sharding 信息（每个参数在哪个文件的哪个位置）
3. 根据当前 rank 的 sharding plan，读取对应的 `.distcp` 文件

#### Step 4: 执行 DCP Load

```python
# dcp_checkpointer.py: 336-340
dcp.load(
    state_dict=load_state,
    storage_reader=storage_reader,
    process_group=process_group,
)
```

**DCP Load 的内部流程**：
1. 读取 `.metadata` 文件，获取全局 sharding plan
2. 根据当前 rank 的 `device_mesh` 和 `placements`，计算需要读取哪些分片
3. 从对应的 `.distcp` 文件读取数据
4. 如果 checkpoint 的并行度与当前不同，执行 resharding
5. 调用 `Stateful.load_state_dict()` 将数据写入模型/optimizer

#### Step 5: ModelState.load_state_dict()

```python
# dcp_checkpointer.py: 67-77
@torch.no_grad()
def load_state_dict(self, state_dict):
    """
    加载 model state dict

    流程：
    1. 如果是 EP+FSDP2，调用 drop_ep_dim() 移除 EP 维度
    2. 调用 PyTorch DCP 的 set_model_state_dict(model, model_state_dict)
    """
    model_state_dict = state_dict
    if self.should_ep_aware:
        model_state_dict = self.get_state_dict_with_ep_dim_preprocess(
            model_state_dict,
            "drop"  # 移除 EP 维度
        )

    set_model_state_dict(model=self.model, model_state_dict=model_state_dict)
```

**为什么需要 `drop_ep_dim()`？**

从 checkpoint 加载的 DTensor：
```python
# 保存时经过 restore_ep_dim()，现在是 1D DTensor:
# placements = [Shard(0)]，device_mesh = ep_mesh (1D)
loaded_param.placements = [Shard(0)]
loaded_param.device_mesh.shape = (ep_size,)
```

运行时期望的 DTensor：
```python
# EP+FSDP2 运行时需要 1D DTensor（仅 FSDP 维度）:
# placements = [Shard(1)]，device_mesh = ep_fsdp_mesh["ep_fsdp"] (1D)
runtime_param.placements = [Shard(1)]
runtime_param.device_mesh.shape = (ep_fsdp_size,)
```

`drop_ep_dim()` 的作用：
```python
# 将 1D DTensor (EP mesh) 转换为 1D DTensor (FSDP mesh)
loaded_param: [Shard(0)], mesh=(ep_size,)
    ↓ drop_ep_dim()
runtime_param: [Shard(1)], mesh=(ep_fsdp_size,)
```

#### Step 6: OptimizerState.load_state_dict()

```python
# dcp_checkpointer.py: 137-151
def load_state_dict(self, state_dict):
    """
    加载 optimizer state dict

    EP+FSDP2 场景：
    - 调用 drop_ep_dim() 移除 EP 维度
    - 委托给 MultiOptimizer.load_state_dict()（会自动分割/过滤）

    非 EP 场景：
    - 调用 PyTorch DCP 的 set_optimizer_state_dict()
    """
    optim_state_from_dcp_load = state_dict

    if self.should_ep_aware:
        # 移除 EP 维度
        optim_state_without_ep_dim = self.get_state_dict_with_ep_dim_preprocess(
            optim_state_from_dcp_load,
            "drop"
        )
        # MultiOptimizer 会将 state 分配给正确的 sub-optimizer
        self.optimizer.load_state_dict(optim_state_without_ep_dim)
        return

    # 单个 torch optimizer
    set_optimizer_state_dict(
        model=self.model,
        optimizers=self.optimizer,
        optim_state_dict=optim_state_from_dcp_load,
    )
```

#### Step 7: 加载 Extra State

```python
# dcp_checkpointer.py: 343
cls._load_extra_state(checkpoint_dir=checkpoint_dir, state=state)

# 实现 (Lines 422-431):
def _load_extra_state(cls, checkpoint_dir: str, state: Dict[str, Any]) -> None:
    if "extra_state" not in state:
        logger.warning_rank0("extra_state not found in state, skipping extra_state load")
        return

    extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
    os.makedirs(extra_state_dir, exist_ok=True)
    extra_state_path = os.path.join(
        extra_state_dir,
        _EXTRA_STATE_FORMAT.format(dist.get_rank())
    )
    state["extra_state"] = torch.load(extra_state_path, weights_only=False)
```

**Extra State 加载特点**：
- 每个 rank 加载自己的 `extra_state_rank_{rank}.pt`
- 不需要跨 rank 通信
- `weights_only=False` 允许加载任意 Python 对象（如 lr_scheduler state）

---

## 5. Stateful Wrapper 实现

### 5.1 ModelState

```python
# veomni/checkpoint/dcp_checkpointer.py: 37-103
class ModelState(Stateful):
    """
    Model 的 Stateful wrapper

    职责：
    1. 实现 Stateful 接口，供 DCP 调用
    2. 处理 EP+FSDP2 的维度转换
    3. 委托给 PyTorch DCP 的 get_model_state_dict / set_model_state_dict
    """

    def __init__(self, model):
        self.model = model

        # 检测是否需要 EP 处理
        self.parallel_state = get_parallel_state()
        self.ep_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self.should_ep_aware = (
            self.ep_fqn2spec_info is not None and
            self.parallel_state.dp_mode == "fsdp2"
        )
```

**EP 检测逻辑**：
- `_fqn2spec_info`：由 `ParallelPlan` 附加到模型上，记录哪些参数使用 EP
- `dp_mode == "fsdp2"`：只有 FSDP2 需要手动处理 EP 维度（FSDP1 通过 hooks）
- `should_ep_aware = True` → 需要调用 `restore_ep_dim()` / `drop_ep_dim()`

**state_dict() 方法**：

```python
@torch.no_grad()
def state_dict(self):
    """
    保存时调用

    流程：
    1. 调用 get_model_state_dict(model)
       - FSDP2: 返回包含 DTensor 的 state dict
       - FSDP1: 触发 state_dict hooks（见 CheckpointExtensions）
    2. 如果是 EP+FSDP2，调用 restore_ep_dim()
    """
    model_state_dict = get_model_state_dict(model=self.model)

    if self.should_ep_aware:
        logger.info_rank0(
            "Getting model state_dict from ModelState wrapper, would restore EP dim for Experts module"
        )
        model_state_dict = self.get_state_dict_with_ep_dim_preprocess(
            model_state_dict,
            "restore"
        )

    return model_state_dict
```

**load_state_dict() 方法**：

```python
@torch.no_grad()
def load_state_dict(self, state_dict):
    """
    加载时调用

    流程：
    1. 如果是 EP+FSDP2，调用 drop_ep_dim()
    2. 调用 set_model_state_dict(model, model_state_dict)
       - FSDP2: 自动将 DTensor 分片写入 model
       - FSDP1: 触发 load_state_dict hooks
    """
    model_state_dict = state_dict
    if self.should_ep_aware:
        model_state_dict = self.get_state_dict_with_ep_dim_preprocess(
            model_state_dict,
            "drop"
        )

    set_model_state_dict(model=self.model, model_state_dict=model_state_dict)
```

### 5.2 OptimizerState

```python
# veomni/checkpoint/dcp_checkpointer.py: 105-201
class OptimizerState(Stateful):
    """
    Optimizer 的 Stateful wrapper

    职责：
    1. 实现 Stateful 接口
    2. 处理 EP+FSDP2 的 MultiOptimizer
    3. 处理 optimizer state 的 EP 维度转换
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        # EP 检测（与 ModelState 相同）
        self.parallel_state = get_parallel_state()
        self.ep_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self.should_ep_aware = (
            self.ep_fqn2spec_info is not None and
            self.parallel_state.dp_mode == "fsdp2"
        )
```

**state_dict() 方法**：

```python
def state_dict(self):
    """
    保存时调用

    EP+FSDP2 场景：
    - 使用 MultiOptimizer.state_dict()
    - MultiOptimizer 已经合并了 EP 和 non-EP optimizers 的 state
    - 调用 restore_ep_dim() 恢复 EP 维度

    非 EP 场景：
    - 使用 get_optimizer_state_dict(model, optimizer)
    """
    if self.should_ep_aware:
        logger.info_rank0(
            "Getting optimizer state_dict from OptimizerState wrapper, would restore EP dim for Experts module"
        )
        assert self.optimizer._is_multi_optimizer, (
            "EP is enabled but optimizer is not a MultiOptimizer instance"
        )
        vanilla_optim_sd = self.optimizer.state_dict()
        optim_sd_with_ep_dim = self.get_state_dict_with_ep_dim_preprocess(
            vanilla_optim_sd,
            "restore"
        )
        return optim_sd_with_ep_dim

    # 单个 torch optimizer
    sd = get_optimizer_state_dict(model=self.model, optimizers=self.optimizer)
    return sd
```

**MultiOptimizer 简介**（veomni/optim/optimizer.py:144-211）：

```python
class MultiOptimizer(torch.optim.Optimizer):
    """
    包含多个 sub-optimizer 的容器

    使用场景：
    EP+FSDP2 中，EP 参数和 non-EP 参数使用不同的 optimizer

    结构：
    self.optimizers = [
        ep_optimizer,       # 管理 EP 参数
        non_ep_optimizer,   # 管理 non-EP 参数
    ]

    state_dict() 方法：
    合并所有 sub-optimizer 的 state dict，返回扁平化的结果
    """
```

**load_state_dict() 方法**：

```python
def load_state_dict(self, state_dict):
    """
    加载时调用

    EP+FSDP2 场景：
    - 调用 drop_ep_dim() 移除 EP 维度
    - 委托给 MultiOptimizer.load_state_dict()
      - MultiOptimizer 会根据参数 FQN 将 state 分配给正确的 sub-optimizer

    非 EP 场景：
    - 使用 set_optimizer_state_dict(model, optimizer, optim_state_dict)
    """
    optim_state_from_dcp_load = state_dict

    if self.should_ep_aware:
        optim_state_without_ep_dim = self.get_state_dict_with_ep_dim_preprocess(
            optim_state_from_dcp_load,
            "drop"
        )
        # MultiOptimizer.load_state_dict() 会正确分配 state
        self.optimizer.load_state_dict(optim_state_without_ep_dim)
        return

    # 单个 torch optimizer
    set_optimizer_state_dict(
        model=self.model,
        optimizers=self.optimizer,
        optim_state_dict=optim_state_from_dcp_load,
    )
```

### 5.3 EP 维度预处理

```python
# dcp_checkpointer.py: 79-102 (ModelState)
# dcp_checkpointer.py: 153-201 (OptimizerState)
def get_state_dict_with_ep_dim_preprocess(self, state_dict, action):
    """
    对 state dict 中的 EP 参数进行维度转换

    参数：
        state_dict: 要处理的 state dict
        action: "restore" (save 时) 或 "drop" (load 时)

    流程：
    1. 遍历 state dict 中的所有 key
    2. 检查 key 是否对应 EP 参数（通过 fqn2spec_info）
    3. 如果是 EP 参数且是 Tensor：
       - action="restore": 调用 restore_ep_dim()
       - action="drop": 调用 drop_ep_dim()
    """
    ep_fqn2spec_info = self.ep_fqn2spec_info
    assert ep_fqn2spec_info is not None

    ep_mesh = self.parallel_state.ep_fsdp_device_mesh["ep"]
    global_device_mesh = self.parallel_state.ep_fsdp_device_mesh
    assert global_device_mesh.ndim == 2

    assert action in ["restore", "drop"]

    keys = list(state_dict.keys())
    for name in sorted(keys):
        # 检查是否是 EP 参数
        if name in ep_fqn2spec_info and isinstance(ep_fqn2spec_info[name].placement, Shard):
            cur_spec_info = ep_fqn2spec_info[name]
            tensor = state_dict[name]

            if action == "drop":
                tensor = drop_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)
            else:  # "restore"
                tensor = restore_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)

            state_dict[name] = tensor

    return state_dict
```

**OptimizerState 的特殊处理**（Lines 153-201）：

```python
def get_state_dict_with_ep_dim_preprocess(self, state_dict, action):
    """
    Optimizer state dict 的 EP 维度预处理

    特殊之处：
    1. Optimizer state dict 的 key 不是参数 FQN，而是包含额外信息：
       "state.model.layers.0.mlp.experts.gate_proj.exp_avg"
       "param_groups.model.layers.0.mlp.experts.gate_proj.amsgrad"

    2. 需要模糊匹配找到对应的 EP spec

    3. 需要跳过非 Tensor 的值（如 "amsgrad": False）

    4. 需要跳过 0-D Tensor（如 "step": 1000）
    """
    keys = list(state_dict.keys())
    ep_keys = list(ep_fqn2spec_info.keys())

    for name in sorted(keys):
        # 模糊匹配：找到包含 EP FQN 的 key
        # 例如：name = "state.model.layers.0.mlp.experts.gate_proj.exp_avg"
        #      ep_key = "model.layers.0.mlp.experts.gate_proj"
        matches = [ep_key for ep_key in ep_keys if ep_key in name]
        if not matches:
            continue  # 非 EP 参数

        assert len(matches) == 1, f"Ambiguous EP spec match for key '{name}': {matches}"

        ep_key = matches[0]
        cur_spec_info = ep_fqn2spec_info[ep_key]

        # 跳过 non-EP placements (Replicate)
        if not isinstance(cur_spec_info.placement, Shard):
            continue

        tensor = state_dict[name]

        # 跳过非 Tensor（如 param_groups 中的超参数）
        if not torch.is_tensor(tensor):
            continue

        # 跳过标量 Tensor（如 "step": 1000）
        if tensor.ndim == 0:
            continue

        # 执行维度转换
        if action == "drop":
            tensor = drop_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)
        elif action == "restore":
            tensor = restore_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)

        state_dict[name] = tensor

    return state_dict
```

---

## 6. EP 维度处理

### 6.1 为什么需要 EP 维度转换？

在 EP+FSDP2 混合并行中，参数的 DTensor 结构：

**运行时** (Training/Inference)：
```python
# 2D DeviceMesh: [ep, ep_fsdp]
# Example: ep_size=2, ep_fsdp_size=4, 总共 8 GPU

device_mesh = DeviceMesh("cuda", [[0,1,2,3], [4,5,6,7]], mesh_dim_names=["ep", "ep_fsdp"])

# EP 参数 (experts) 的 placements
experts.gate_proj.placements = [Shard(0), Shard(1)]
# Shard(0): 在 EP 维度分片（不同 EP rank 持有不同专家）
# Shard(1): 在 FSDP 维度分片（同一专家在 FSDP rank 间分片）
```

**DCP 期望** (Checkpoint)：
```python
# DCP 的 resharding 逻辑基于 DTensor placements
# 如果直接保存 [Shard(0), Shard(1)]，DCP 会：
# 1. 尝试在两个维度同时 reshard（错误！）
# 2. 无法正确处理 EP 维度的分片信息

# 解决方案：将 EP 信息编码到 1D DTensor
# Save 时: [Shard(0), Shard(1)] → [Shard(0)]，device_mesh = ep_mesh (1D)
# Load 时: [Shard(0)] → [Shard(1)]，device_mesh = ep_fsdp_mesh["ep_fsdp"] (1D)
```

**转换流程图**：

```
[运行时 2D DTensor]
device_mesh: [[0,1,2,3], [4,5,6,7]] (2D, shape=(2, 4))
placements: [Shard(0), Shard(1)]
    ↓ Save: restore_ep_dim()
[Checkpoint 1D DTensor]
device_mesh: [0, 4] (1D EP mesh, shape=(2,))
placements: [Shard(0)]
    ↓ (保存到磁盘)
[从 Checkpoint 读取]
device_mesh: [0, 4] (1D EP mesh, shape=(2,))
placements: [Shard(0)]
    ↓ Load: drop_ep_dim()
[运行时 1D DTensor (FSDP-only)]
device_mesh: [0,1,2,3] (1D FSDP mesh, shape=(4,))
placements: [Shard(1)]
    ↓ FSDP2 内部处理
[最终恢复为 2D DTensor]
device_mesh: [[0,1,2,3], [4,5,6,7]] (2D)
placements: [Shard(0), Shard(1)]
```

### 6.2 restore_ep_dim() - 恢复 EP 维度

```python
# veomni/checkpoint/dcp_checkpointer.py: 225-249
def restore_ep_dim(orgin_tensor: torch.Tensor, device_mesh: DeviceMesh):
    """
    在 Save 时调用，将运行时的 DTensor 转换为 DCP 可理解的格式

    输入：
        orgin_tensor: 运行时的 DTensor
            - EP+FSDP2: 2D DTensor, placements=[Shard(0), Shard(1)]
            - EP-only: 1D DTensor, placements=[Shard(0)]
            - 或者 local Tensor (no FSDP)
        device_mesh: EP-FSDP 2D device mesh

    输出：
        1D DTensor with EP dimension encoded
            - EP+FSDP2: 1D DTensor, placements=[Shard(0)], mesh=global_mesh
            - EP-only: 1D DTensor, placements=[Shard(0)], mesh=ep_mesh

    目标：
        让 DCP 能够正确理解 EP 维度的分片信息
    """
    assert device_mesh.ndim == 2, f"global_mesh.ndim must be 2, got {device_mesh.ndim}"
    ep_mesh = device_mesh["ep"]

    if isinstance(orgin_tensor, DTensor):
        # EP+FSDP2 场景：2D DTensor → 2D DTensor (但使用 global mesh)
        # Input:  placements=[Shard(0), Shard(1)], mesh=??? (implicit)
        # Output: placements=[Shard(0), Shard(1)], mesh=device_mesh (explicit)
        dtensor = DTensor.from_local(
            orgin_tensor._local_tensor,       # 本地分片数据
            device_mesh=device_mesh,          # 显式使用 global mesh
            placements=[Shard(0), Shard(1)],  # 保持原始 placements
        )
    elif torch.is_tensor(orgin_tensor):
        # EP-only 场景（无 FSDP）：local Tensor → 1D DTensor
        # Input:  local Tensor
        # Output: placements=[Shard(0)], mesh=ep_mesh
        dtensor = DTensor.from_local(
            orgin_tensor,
            device_mesh=ep_mesh,      # 1D EP mesh
            placements=[Shard(0)],     # 在 EP 维度分片
        )
    else:
        raise RuntimeError(f"origin_tensor - {orgin_tensor} is not a tensor!")

    return dtensor
```

**关键理解**：

1. **为什么 EP+FSDP2 还是返回 2D DTensor？**
   - DCP 需要知道完整的 sharding 信息（EP + FSDP 两个维度）
   - 通过显式指定 `device_mesh=device_mesh`，告诉 DCP 这是一个 2D 分片
   - DCP 会在保存时将 2D mesh 信息编码到 `.metadata` 文件

2. **DCP 如何理解 2D mesh？**
   ```python
   # DCP 内部会展平 2D mesh 为 1D
   # mesh = [[0,1,2,3], [4,5,6,7]] → flattened = [0,1,2,3,4,5,6,7]
   # 并在 .metadata 中记录 mesh_shape = (2, 4)
   ```

3. **EP-only 场景**：
   - 没有 FSDP，只有 EP 分片
   - 直接转换为 1D DTensor with EP mesh

### 6.3 drop_ep_dim() - 移除 EP 维度

```python
# veomni/checkpoint/dcp_checkpointer.py: 204-222
def drop_ep_dim(loaded_tensor: torch.Tensor, device_mesh: DeviceMesh):
    """
    在 Load 时调用，将从 checkpoint 加载的 DTensor 转换为运行时格式

    输入：
        loaded_tensor: 从 DCP 加载的 DTensor
            - 2D DTensor: placements=[Shard(0), Shard(1)], mesh=global_mesh
            - 1D DTensor: placements=[Shard(0)], mesh=ep_mesh
        device_mesh: EP-FSDP 2D device mesh

    输出：
        1D DTensor (FSDP-only) 或 local Tensor
            - 2D → 1D: placements=[Shard(1)], mesh=ep_fsdp_mesh["ep_fsdp"]
            - 1D → local: local Tensor

    目标：
        移除 EP 维度，让 FSDP2 能够正确处理后续的 DTensor
    """
    assert device_mesh.ndim == 2, f"global_mesh.ndim must be 2, got {device_mesh.ndim}"
    ep_fsdp_mesh = device_mesh["ep_fsdp"]  # 提取 FSDP 维度的 1D mesh

    if len(loaded_tensor.placements) == 2:
        # EP+FSDP2 场景：2D DTensor → 1D DTensor (仅 FSDP)
        # Input:  placements=[Shard(0), Shard(1)], mesh=global_mesh (2D)
        # Output: placements=[Shard(1)], mesh=ep_fsdp_mesh (1D)
        tensor_to_put = DTensor.from_local(
            loaded_tensor._local_tensor,  # 本地分片数据
            device_mesh=ep_fsdp_mesh,      # 1D FSDP mesh
            placements=[Shard(1)],         # 只在 FSDP 维度分片
        )
    elif len(loaded_tensor.placements) == 1:
        # EP-only 场景（无 FSDP）：1D DTensor → local Tensor
        # Input:  placements=[Shard(0)], mesh=ep_mesh
        # Output: local Tensor
        tensor_to_put = loaded_tensor.to_local()
    else:
        raise RuntimeError(
            f"Expect EP paramters from checkpoints to be DTensor with 1-dim (no FSDP) or 2-dim (EP+FSDP), got {loaded_tensor}"
        )

    return tensor_to_put
```

**为什么 `placements=[Shard(1)]`？**

```python
# 加载后的 DTensor 结构：
# placements=[Shard(1)]，device_mesh=ep_fsdp_mesh (1D, shape=(4,))

# 这表示：参数在 FSDP 维度分片，但没有 EP 维度信息

# FSDP2 内部会自动将 1D DTensor 转换为 2D DTensor：
# placements=[Shard(1)] → placements=[Replicate(), Shard(1)]
# device_mesh=ep_fsdp_mesh (1D) → device_mesh=global_mesh (2D)

# 然后 EP 的逻辑会将 Replicate() 替换为 Shard(0)：
# placements=[Replicate(), Shard(1)] → placements=[Shard(0), Shard(1)]
```

**关键点**：
- `drop_ep_dim()` 只负责移除显式的 EP 维度
- FSDP2 和 EP 的运行时逻辑会自动恢复完整的 2D structure
- 这种设计避免了在 checkpoint 代码中硬编码 EP 逻辑

### 6.4 EP 维度转换示例

**示例 1：EP+FSDP2 Save & Load**

```python
# 训练时的 DTensor
param = model.layers[0].mlp.experts.gate_proj.weight
param.shape = (32, 29568, 8192)  # 32 experts per EP rank (总共 128 experts)
param.device_mesh = DeviceMesh("cuda", [[0,1,2,3], [4,5,6,7]], mesh_dim_names=["ep", "ep_fsdp"])
param.placements = [Shard(0), Shard(1)]  # EP 维度分片 + FSDP 维度分片

# ===== Save 流程 =====

# Step 1: get_model_state_dict() 返回的 DTensor
state_dict["model.layers.0.mlp.experts.gate_proj.weight"] = param
# param.placements = [Shard(0), Shard(1)]

# Step 2: ModelState.state_dict() 调用 restore_ep_dim()
restored_param = restore_ep_dim(param, device_mesh=param.device_mesh)
# restored_param.placements = [Shard(0), Shard(1)]  # 保持不变
# restored_param.device_mesh = global_mesh (显式指定)

# Step 3: DCP 保存
# DCP 将 2D mesh 信息编码到 .metadata
# 每个 rank 保存自己的本地分片

# ===== Load 流程 =====

# Step 1: DCP 加载
# DCP 根据 .metadata 和当前 device_mesh，读取对应分片
loaded_param = ...  # 从 .distcp 文件读取
# loaded_param.placements = [Shard(0), Shard(1)]
# loaded_param.device_mesh = global_mesh (从 .metadata 恢复)

# Step 2: ModelState.load_state_dict() 调用 drop_ep_dim()
dropped_param = drop_ep_dim(loaded_param, device_mesh=global_mesh)
# dropped_param.placements = [Shard(1)]  # 只保留 FSDP 维度
# dropped_param.device_mesh = ep_fsdp_mesh["ep_fsdp"] (1D)

# Step 3: set_model_state_dict() 写入模型
# FSDP2 内部会将 1D DTensor 转换回 2D DTensor
# final_param.placements = [Shard(0), Shard(1)]
# final_param.device_mesh = global_mesh (2D)
```

**示例 2：EP-only Save & Load (无 FSDP)**

```python
# 训练时的 local Tensor（EP-only，无 FSDP）
param = model.layers[0].mlp.experts.gate_proj.weight
param.shape = (32, 29568, 8192)  # 本地 32 experts
# param 是 local Tensor，不是 DTensor

# ===== Save 流程 =====

# Step 1: restore_ep_dim()
restored_param = restore_ep_dim(param, device_mesh=global_mesh)
# restored_param.placements = [Shard(0)]
# restored_param.device_mesh = ep_mesh (1D)

# Step 2: DCP 保存
# 保存为 1D DTensor with EP mesh

# ===== Load 流程 =====

# Step 1: DCP 加载
loaded_param = ...
# loaded_param.placements = [Shard(0)]
# loaded_param.device_mesh = ep_mesh (1D)

# Step 2: drop_ep_dim()
dropped_param = drop_ep_dim(loaded_param, device_mesh=global_mesh)
# dropped_param = loaded_param.to_local()  # 转换为 local Tensor

# Step 3: set_model_state_dict() 写入模型
# 直接使用 local Tensor
```

---

## 7. FSDP1 Checkpoint 扩展

### 7.1 为什么需要 CheckpointExtensions？

FSDP1 的 checkpoint 机制与 FSDP2 不同：

| 特性 | FSDP2 | FSDP1 |
|------|-------|-------|
| State Dict API | `get_model_state_dict()` | `model.state_dict()` + hooks |
| DTensor 支持 | 原生支持 | 需要手动转换 |
| EP 集成 | 通过 Stateful wrapper | 通过 FSDPExtensions |
| Hooks | 不需要 | 需要 state_dict hooks |

**FSDP1 的问题**：
- `model.state_dict()` 不理解 EP 的 DTensor
- 需要在 `state_dict()` 调用时插入自定义逻辑
- 需要在 `load_state_dict()` 调用时移除 EP 维度

**解决方案**：使用 `FSDPExtensions`
- PyTorch FSDP1 提供的扩展机制
- 允许自定义 state_dict hooks 和 DTensor 处理逻辑
- VeOmni 实现了 `CheckpointExtensions` 类

### 7.2 CheckpointExtensions 实现

```python
# veomni/distributed/fsdp/extension.py: 124-252
class CheckpointExtensions(FSDPExtensions):
    """
    FSDP1 的 EP 扩展

    职责：
    1. 在 model.state_dict() 时添加 EP 维度
    2. 在 model.load_state_dict() 时移除 EP 维度
    3. 自定义 DTensor chunking 策略
    """

    def __init__(
        self,
        ep_fsdp_device_mesh: DeviceMesh,
        fqn2spec_info: Dict[str, SpecInfo],
    ):
        super().__init__()
        self.ep_fsdp_device_mesh = ep_fsdp_device_mesh
        self.ep_mesh = ep_fsdp_device_mesh["ep"] if ep_fsdp_device_mesh is not None else None
        self.fqn2spec_info = fqn2spec_info
```

### 7.3 state_dict_post_hook() - Save 时处理

```python
# extension.py: 193-215
@torch.no_grad()
def state_dict_post_hook(
    self,
    module,
    state_dict,
    prefix,
    local_metadata,
    fqn2spec_info: Dict[str, SpecInfo] = None
):
    """
    在 model.state_dict() 后调用

    作用：
    为 EP 参数添加 EP 维度的 DTensor placements

    流程：
    1. 遍历 state_dict 中的所有参数
    2. 检查是否是 EP 参数（通过 fqn2spec_info）
    3. 如果是 EP 参数：
       - 调用 _shard_tensor() 将 FSDP DTensor 转换为 EP+FSDP DTensor
    """
    if fqn2spec_info is None:
        return state_dict

    # 遍历 state dict
    for name, param in state_dict.items():
        # 移除 prefix 得到 FQN
        fqn = name[len(prefix):] if name.startswith(prefix) else name

        if fqn not in fqn2spec_info:
            continue  # 非 EP 参数

        cur_spec_info = fqn2spec_info[fqn]
        if not isinstance(cur_spec_info.placement, Shard):
            continue  # EP 参数但使用 Replicate placement

        # 添加 EP 维度
        state_dict[name] = _shard_tensor(
            param,
            cur_spec_info.ep_fsdp_mesh,
            cur_spec_info.placement
        )

    return state_dict
```

**`_shard_tensor()` 函数**（extension.py:43-62）：

```python
def _shard_tensor(
    orgin_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    shard: Shard = Shard(0)
):
    """
    将 FSDP DTensor 转换为 EP+FSDP DTensor

    输入：
        orgin_tensor: FSDP1 生成的 DTensor
            - DTensor with placements=[Replicate(), ..., Shard(N)]
        device_mesh: EP-FSDP 2D device mesh
        shard: EP placement (默认 Shard(0))

    输出：
        2D DTensor with EP placement
            - placements=[Shard(0), ...] (EP + FSDP)
    """
    assert device_mesh.ndim == 2
    ep_mesh = device_mesh["ep"]

    if orgin_tensor.__class__.__name__ == "DTensor":
        # FSDP1 DTensor → EP+FSDP DTensor
        # Input placements:  [Replicate(), ..., Shard(N)]
        # Output placements: [Shard(0), Replicate(), ..., Shard(N)]
        placements = (shard,) + orgin_tensor.placements
        dtensor = DTensor.from_local(
            orgin_tensor._local_tensor,
            device_mesh=device_mesh,  # 2D mesh
            placements=placements
        )
    elif orgin_tensor.__class__.__name__ == "Tensor":
        # Local Tensor → EP DTensor (EP-only, no FSDP)
        dtensor = DTensor.from_local(
            orgin_tensor,
            device_mesh=ep_mesh,  # 1D EP mesh
            placements=[shard]
        )

    return dtensor
```

### 7.4 load_state_dict_pre_hook() - Load 时处理

```python
# extension.py: 218-252
@torch.no_grad()
def load_state_dict_pre_hook(
    self,
    state_dict,
    prefix,
    fqn2spec_info: Dict[str, SpecInfo] = None,
    strict: bool = True,
):
    """
    在 model.load_state_dict() 前调用

    作用：
    从 checkpoint 加载的 EP 参数中移除 EP 维度

    流程：
    1. 遍历 state_dict 中的所有参数
    2. 检查是否是 EP 参数
    3. 如果是 EP 参数：
       - 调用 _shard_dtensor() 将 EP+FSDP DTensor 转换为 local Tensor
    """
    if fqn2spec_info is None:
        return state_dict

    for name in list(state_dict.keys()):
        fqn = name[len(prefix):] if name.startswith(prefix) else name

        if fqn not in fqn2spec_info:
            continue

        cur_spec_info = fqn2spec_info[fqn]
        if not isinstance(cur_spec_info.placement, Shard):
            continue

        # 移除 EP 维度
        param = state_dict[name]
        if isinstance(param, DTensor):
            state_dict[name] = _shard_dtensor(
                param,
                cur_spec_info.ep_fsdp_mesh
            )

    return state_dict
```

**`_shard_dtensor()` 函数**（extension.py:65-81）：

```python
def _shard_dtensor(
    orgin_dtensor: DTensor,
    device_mesh: DeviceMesh,
    shard: Shard = Shard(0)
):
    """
    将 EP+FSDP DTensor 转换为 local Tensor

    输入：
        orgin_dtensor: 从 checkpoint 加载的 DTensor
            - placements=[Shard(0), ...] (EP + FSDP)

    输出：
        local Tensor（移除所有 DTensor 信息）

    注意：
        FSDP1 会在后续的 load_state_dict 流程中重新分片
    """
    assert isinstance(orgin_dtensor, DTensor)

    # 直接转换为 local Tensor
    # FSDP1 的 load_state_dict 会重新应用 FSDP 分片
    local_tensor = orgin_dtensor.to_local()

    return local_tensor
```

### 7.5 chunk_dtensor() - DTensor 分片策略

```python
# extension.py: 135-155
def chunk_dtensor(
    self,
    tensor: torch.Tensor,
    rank: int,
    device_mesh: DeviceMesh
) -> torch.Tensor:
    """
    自定义 DTensor 分片策略

    作用：
    在 FSDP initialization 时，将参数分片为 DTensor

    流程：
    1. 选择最大的维度进行分片（优先选择能被 fsdp_size 整除的维度）
    2. 创建 DTensor with HSDP placements:
       - 前面的维度：Replicate() (for DP replicate)
       - 最后一个维度：Shard(selected_dim) (for DP shard)

    示例：
        tensor.shape = (32, 29568, 8192)
        fsdp_size = 4
        selected_dim = 0 (因为 32 % 4 == 0)
        placements = [Replicate(), ..., Shard(0)]
    """
    tensor = tensor.clone().detach()
    fsdp_size = device_mesh.size(-1)

    # 选择分片维度
    dimlens = tuple(tensor.size())
    selected_dim = dimlens.index(max(dimlens))  # 默认：最大维度
    for dim, dimlen in enumerate(dimlens):
        if dimlen % fsdp_size == 0:
            selected_dim = dim  # 优先：能整除的维度
            break

    # HSDP placements
    replicate_placements = [Replicate() for _ in range(device_mesh.ndim)]
    shard_placements = [Replicate() for _ in range(device_mesh.ndim)]
    shard_placements[-1] = Shard(selected_dim)

    # 创建 DTensor
    dtensor = DTensor.from_local(
        tensor,
        device_mesh,
        replicate_placements,
        run_check=False
    ).redistribute(
        placements=shard_placements,
    )

    return dtensor
```

**HSDP (Hybrid Sharded Data Parallel) Placements**：
```python
# 假设 device_mesh 是 2D: [dp_replicate, dp_shard]
replicate_placements = [Replicate(), Replicate()]
shard_placements = [Replicate(), Shard(dim)]

# 流程：
# 1. 先创建全 Replicate 的 DTensor（所有 rank 有完整副本）
# 2. redistribute 到 shard_placements（在最后一个 mesh 维度分片）
```

### 7.6 FSDP1 Extensions 注册

```python
# extension.py: 414-451
def register_checkpoint_extension(
    model: nn.Module,
    ep_fsdp_device_mesh: DeviceMesh,
    fqn2spec_info: Dict[str, SpecInfo],
):
    """
    为 FSDP1 模型注册 CheckpointExtensions

    使用场景：
    在 FSDP initialization 之前调用

    流程：
    1. 创建 CheckpointExtensions 实例
    2. 遍历所有 FSDP modules
    3. 注册 state_dict hooks:
       - _state_dict_post_hook: 调用 extension.state_dict_post_hook()
       - _load_state_dict_pre_hook: 调用 extension.load_state_dict_pre_hook()
    """
    extension = CheckpointExtensions(ep_fsdp_device_mesh, fqn2spec_info)

    for name, module in model.named_modules():
        if isinstance(module, FSDP):
            # 注册 state_dict post hook
            module._register_state_dict_hook(
                partial(
                    extension.state_dict_post_hook,
                    fqn2spec_info=fqn2spec_info,
                )
            )

            # 注册 load_state_dict pre hook
            module._register_load_state_dict_pre_hook(
                partial(
                    extension.load_state_dict_pre_hook,
                    fqn2spec_info=fqn2spec_info,
                ),
                with_module=True,
            )

    return extension
```

**使用示例**（从训练脚本）：

```python
# 初始化 FSDP1 model
from veomni.distributed.fsdp import register_checkpoint_extension

model = MyModel(config)

# 应用 parallel plan（标记 EP 参数）
parallel_plan = get_parallel_plan()
parallel_plan.apply(model)

# 注册 checkpoint extension
if parallel_state.ep_enabled:
    extension = register_checkpoint_extension(
        model,
        ep_fsdp_device_mesh=parallel_state.ep_fsdp_device_mesh,
        fqn2spec_info=model._fqn2spec_info,
    )

# 包装为 FSDP
model = FSDP(
    model,
    device_mesh=device_mesh,
    use_orig_params=True,
    ...
)

# 现在 model.state_dict() 和 model.load_state_dict() 会自动处理 EP
```

---

## 8. Async Save 异步保存

### 8.1 为什么需要 Async Save？

**同步 Save 的问题**：
```python
# 同步保存流程
dcp.save(state_dict, storage_writer)
# 主进程阻塞，等待所有 rank 写入完成
dist.barrier()
# 总耗时 = 最慢 rank 的写入时间（通常 10-30 秒）
```

**训练中断时间**：
- 每个 checkpoint 需要 10-30 秒
- 如果每 100 步保存一次，训练效率显著降低
- 大模型的 checkpoint 可能更大（> 1 TB），耗时更长

**Async Save 的优势**：
```python
# 异步保存流程
future = dcp.async_save(state_dict, storage_writer, process_group)
# 主进程立即返回，继续训练
# 后台线程负责写入

# 训练继续...
next_batch = dataloader.next()
output = model(next_batch)
loss.backward()
optimizer.step()

# 下次保存前，等待上一次完成
if future is not None:
    future.result()  # 阻塞直到完成
```

**性能对比**：
| 模式 | 训练中断时间 | 吞吐量影响 |
|------|--------------|------------|
| Sync | 10-30 秒/次 | 显著降低（~10%） |
| Async | < 1 秒/次 | 几乎无影响 |

### 8.2 Async Save 实现

```python
# veomni/checkpoint/dcp_checkpointer.py: 350-383
@classmethod
def execute_save(
    cls,
    save_state: Dict[str, Any],
    storage_writer: FileSystemWriter,
    save_async: bool,
) -> None:
    """
    执行 DCP save，支持异步模式

    参数：
        save_state: 包含 ModelState 和 OptimizerState 的字典
        storage_writer: FileSystemWriter 实例
        save_async: 是否异步保存

    流程：
    - save_async=True:
      1. 创建专用的 Gloo process group（首次）
      2. 等待上一次 async save 完成
      3. 启动新的 async save
    - save_async=False:
      1. 同步保存
      2. barrier + gc
    """
    if save_async:
        # ===== 异步模式 =====

        # Step 1: 创建专用 Gloo process group
        if cls._async_process_group is None:
            cls._async_process_group = dist.new_group(backend="gloo")

        # Step 2: 等待上一次 async save 完成
        if cls.dcp_save_future is not None:
            logger.info(
                f"[RANK {dist.get_rank()}] waiting for previous DCP saving session to end..."
            )
            cls.dcp_save_future.result()  # 阻塞直到完成
            cls.dcp_save_future = None
            dist.barrier()  # 确保所有 rank 都完成

        # Step 3: 启动 async save
        cls.dcp_save_future = dcp.async_save(
            state_dict=save_state,
            storage_writer=storage_writer,
            process_group=cls._async_process_group,  # 使用专用 PG
        )
        # 函数立即返回，不阻塞训练

    else:
        # ===== 同步模式 =====

        # Step 1: 同步保存
        dcp.save(
            state_dict=save_state,
            storage_writer=storage_writer,
        )

        # Step 2: 同步 + 清理
        if dist.is_initialized():
            dist.barrier()
        gc.collect()           # 垃圾回收
        empty_cache()          # 清空 CUDA cache
        synchronize()          # CUDA 同步
```

### 8.3 为什么需要专用的 Gloo Process Group？

**问题背景**：
- PyTorch 的 async save 需要一个 process group 进行后台通信
- 默认的 NCCL process group 可能被训练使用（all-reduce, all-gather 等）
- 如果 async save 使用默认 PG，可能与训练通信冲突

**解决方案**：创建专用的 Gloo PG
```python
if cls._async_process_group is None:
    cls._async_process_group = dist.new_group(backend="gloo")
```

**为什么使用 Gloo？**
- Gloo 支持 CPU-based 通信，不占用 GPU 资源
- NCCL 主要用于 GPU 通信，更高效但资源有限
- Checkpoint 写入主要是 CPU-IO 操作，Gloo 足够

### 8.4 Async Save 的生命周期

```python
# 类变量，跨多次 save 调用共享
dcp_save_future: Optional[Any] = None         # 当前 async save 的 future
_async_process_group: Optional[Any] = None    # 专用 Gloo PG

# ===== 第一次 async save =====
execute_save(..., save_async=True)
    ├─ _async_process_group is None → 创建 Gloo PG
    ├─ dcp_save_future is None → 跳过 wait
    └─ dcp_save_future = dcp.async_save(...)
# 返回，训练继续...

# ===== 第二次 async save =====
execute_save(..., save_async=True)
    ├─ _async_process_group 已存在 → 复用
    ├─ dcp_save_future 存在 → dcp_save_future.result() (等待第一次完成)
    ├─ dist.barrier() (确保所有 rank 都完成)
    └─ dcp_save_future = dcp.async_save(...)
# 返回，训练继续...

# ===== 训练结束 =====
# 如果最后一次 save 是 async，需要手动等待：
if Checkpointer.dcp_save_future is not None:
    Checkpointer.dcp_save_future.result()
```

### 8.5 Async Save 的注意事项

**1. 内存占用**：
```python
# Async save 会在后台持有 state_dict 的引用
# 如果 state_dict 很大（如 100 GB），会额外占用内存

# 建议：
# - 在 save 前释放不必要的缓存
# - 使用更频繁的 gc.collect()
```

**2. 并发安全**：
```python
# Async save 期间，不要修改 model/optimizer
# 否则可能保存不一致的状态

# 推荐模式：
save_state = {
    "model": model.state_dict(),       # 拷贝 state dict
    "optimizer": optimizer.state_dict(),
}
Checkpointer.save(..., save_async=True)
# save_state 被 async save 持有，原始 model 可以继续训练
```

**3. 错误处理**：
```python
# 如果 async save 失败，future.result() 会抛出异常

try:
    if cls.dcp_save_future is not None:
        cls.dcp_save_future.result()
except Exception as e:
    logger.error(f"Async save failed: {e}")
    # 处理错误（如重试、保存到备份路径等）
```

**4. 训练结束时的清理**：
```python
# 在训练脚本的最后，确保 async save 完成
if Checkpointer.dcp_save_future is not None:
    logger.info("Waiting for final async save to complete...")
    Checkpointer.dcp_save_future.result()
    logger.info("Final async save completed.")
```

---

## 9. Extra State 处理

### 9.1 什么是 Extra State？

**DCP 管理的 State**：
- `model`: 模型参数（权重、bias）
- `optimizer`: Optimizer state（momentum, variance）

**Extra State（DCP 不管理）**：
- `global_step`: 当前训练步数
- `lr_scheduler`: 学习率调度器状态
- `train_dataloader`: 数据加载器状态（当前 epoch、shuffle seed）
- `scaler`: Mixed precision scaler 状态
- `rng_state`: 随机数生成器状态
- 其他自定义状态

**为什么需要 Extra State？**
- DCP 只处理 `Stateful` 对象（model, optimizer）
- lr_scheduler, dataloader 等不是 `Stateful`
- 需要独立保存/加载这些组件的状态以实现完整的训练恢复

### 9.2 Extra State 保存

```python
# veomni/checkpoint/dcp_checkpointer.py: 407-419
@classmethod
def _save_extra_state(cls, checkpoint_dir: str, state: Dict[str, Any]) -> None:
    """
    保存 extra_state 到独立文件

    流程：
    1. 检查 state 中是否有 "extra_state"
    2. 创建 extra_state 子目录
    3. 每个 rank 保存自己的 extra_state 到 extra_state_rank_{rank}.pt

    注意：
    - 每个 rank 有独立的文件（因为 dataloader 状态可能不同）
    - 在 DCP save 之前调用（确保一致性）
    """
    if "extra_state" not in state:
        logger.warning_rank0("extra_state not found in state, skipping extra_state save")
        return

    # 创建 extra_state 目录
    extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)  # "extra_state"
    os.makedirs(extra_state_dir, exist_ok=True)

    # 生成文件路径: extra_state/extra_state_rank_0.pt
    extra_state_path = os.path.join(
        extra_state_dir,
        _EXTRA_STATE_FORMAT.format(dist.get_rank())  # "extra_state_rank_{rank}.pt"
    )

    # 保存（使用 torch.save）
    torch.save(
        state["extra_state"],
        extra_state_path,
    )
```

**保存的文件结构**：
```
checkpoints/global_step_1000/
├── .metadata
├── __0_0.distcp
├── __1_0.distcp
├── ...
└── extra_state/
    ├── extra_state_rank_0.pt
    ├── extra_state_rank_1.pt
    ├── extra_state_rank_2.pt
    └── extra_state_rank_3.pt
```

**为什么每个 rank 独立保存？**
- Dataloader 的状态可能不同（如数据分片、shuffle seed）
- 某些组件的状态是 rank-specific 的
- 简化实现（不需要跨 rank 同步）

### 9.3 Extra State 加载

```python
# dcp_checkpointer.py: 422-431
@classmethod
def _load_extra_state(cls, checkpoint_dir: str, state: Dict[str, Any]) -> None:
    """
    加载 extra_state 从独立文件

    流程：
    1. 检查 state 中是否有 "extra_state" key
    2. 读取 extra_state/extra_state_rank_{rank}.pt
    3. 加载到 state["extra_state"]

    注意：
    - 每个 rank 加载自己的文件
    - 在 DCP load 之后调用
    """
    if "extra_state" not in state:
        logger.warning_rank0("extra_state not found in state, skipping extra_state load")
        return

    # 构建文件路径
    extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
    os.makedirs(extra_state_dir, exist_ok=True)  # 确保目录存在
    extra_state_path = os.path.join(
        extra_state_dir,
        _EXTRA_STATE_FORMAT.format(dist.get_rank())
    )

    # 加载（使用 torch.load）
    state["extra_state"] = torch.load(
        extra_state_path,
        weights_only=False  # 允许加载任意 Python 对象
    )
```

### 9.4 Extra State 使用示例

```python
# ===== 保存 Extra State =====

# 准备 extra_state
extra_state = {
    "global_step": global_step,
    "epoch": epoch,
    "lr_scheduler": lr_scheduler.state_dict(),
    "train_dataloader": {
        "epoch": dataloader.epoch,
        "shuffle_seed": dataloader.shuffle_seed,
    },
    "rng_state": {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    },
}

# 保存
state = {
    "model": model,
    "optimizer": optimizer,
    "extra_state": extra_state,
}
Checkpointer.save(save_path, state, global_steps=1000)

# ===== 加载 Extra State =====

# 准备空的 extra_state
state = {
    "model": model,
    "optimizer": optimizer,
    "extra_state": {},  # 空字典，会被填充
}

# 加载
Checkpointer.load(load_path, state)

# 恢复 extra_state
global_step = state["extra_state"]["global_step"]
epoch = state["extra_state"]["epoch"]
lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])

# 恢复 dataloader 状态
dataloader.epoch = state["extra_state"]["train_dataloader"]["epoch"]
dataloader.shuffle_seed = state["extra_state"]["train_dataloader"]["shuffle_seed"]

# 恢复 RNG 状态
rng_state = state["extra_state"]["rng_state"]
random.setstate(rng_state["python"])
np.random.set_state(rng_state["numpy"])
torch.set_rng_state(rng_state["torch"])
torch.cuda.set_rng_state_all(rng_state["cuda"])
```

### 9.5 Extra State 的限制

**1. 不支持 Resharding**：
```python
# DCP 可以在不同并行度下加载 model/optimizer
# 但 extra_state 是 per-rank 的，无法自动 reshard

# 示例：从 8 GPU checkpoint 恢复到 16 GPU
# - model/optimizer: DCP 自动 reshard ✓
# - extra_state: 每个 rank 加载哪个文件？ ✗

# 建议：
# - 仅在相同并行度下恢复时使用 extra_state
# - 或者只保存 global_step 等全局信息（rank 0 保存，其他 rank 广播）
```

**2. 文件大小**：
```python
# 如果 extra_state 很大（如完整的 dataloader 状态），会占用大量磁盘
# 建议：
# - 只保存必要的状态（如 shuffle seed，而非整个 dataloader）
# - 对于大对象，考虑使用 DCP 保存（需要实现 Stateful 接口）
```

**3. 兼容性**：
```python
# torch.save 保存的对象在不同 PyTorch 版本间可能不兼容
# 建议：
# - 使用简单的数据结构（dict, list, int, float）
# - 避免保存复杂对象（如自定义类实例）
```

---

## 10. 存储格式与元数据

### 10.1 Checkpoint 目录结构

```
checkpoints/
└── global_step_1000/                    # Checkpoint 目录
    ├── .metadata                        # DCP 元数据文件
    ├── __0_0.distcp                     # Rank 0 的参数分片
    ├── __1_0.distcp                     # Rank 1 的参数分片
    ├── __2_0.distcp                     # Rank 2 的参数分片
    ├── __3_0.distcp                     # Rank 3 的参数分片
    ├── ...
    └── extra_state/                     # Extra state 目录
        ├── extra_state_rank_0.pt
        ├── extra_state_rank_1.pt
        ├── extra_state_rank_2.pt
        └── extra_state_rank_3.pt
```

### 10.2 .metadata 文件

`.metadata` 文件是 DCP 的核心元数据文件，包含：

**1. Checkpoint 元信息**：
```python
{
    "version": "2.0",                    # DCP 版本
    "world_size": 8,                     # 保存时的 world size
    "timestamp": "2024-01-01T12:00:00",  # 保存时间
}
```

**2. 参数 Sharding 信息**：
```python
{
    "model.layers.0.mlp.experts.gate_proj.weight": {
        "dtype": "torch.float16",
        "shape": [128, 29568, 8192],     # 全局 shape
        "device_mesh": {
            "mesh": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "mesh_dim_names": ["ep", "ep_fsdp"],
        },
        "placements": ["Shard(0)", "Shard(1)"],
        "chunks": [
            {
                "rank": 0,
                "file": "__0_0.distcp",
                "offset": 0,
                "length": 1024000,
                "local_shape": [16, 7392, 8192],  # Rank 0 的本地 shape
            },
            {
                "rank": 1,
                "file": "__1_0.distcp",
                ...
            },
            ...
        ]
    },
    ...
}
```

**3. Optimizer State Sharding 信息**：
```python
{
    "state.model.layers.0.mlp.experts.gate_proj.weight.exp_avg": {
        "dtype": "torch.float32",
        "shape": [128, 29568, 8192],
        # ... 类似 model 参数的 sharding 信息
    },
    ...
}
```

### 10.3 .distcp 文件

**文件命名规则**：
```python
# 格式: __{rank}_{shard_id}.distcp
# 示例:
__0_0.distcp   # Rank 0, Shard 0
__1_0.distcp   # Rank 1, Shard 0
...

# shard_id 通常为 0（single_file_per_rank=True）
# 如果 single_file_per_rank=False，可能有多个 shard
```

**文件内容**：
- 二进制格式，包含该 rank 负责的所有参数分片
- 按照 .metadata 中记录的 offset 和 length 组织
- 使用 FileSystemWriter 的多线程写入

**文件大小估算**：
```python
# 假设：Qwen3-MoE-72B, FSDP size=8, EP size=2, 总共 16 GPU

# 参数量：
# - Total: 72B parameters
# - Per EP rank: 72B / 2 = 36B parameters
# - Per FSDP rank (within EP): 36B / 8 = 4.5B parameters

# Checkpoint 大小（FP16）：
# - Per rank: 4.5B * 2 bytes = 9 GB
# - Total: 72B * 2 bytes = 144 GB

# 文件数量：
# - .distcp files: 16 个（每个 rank 一个）
# - extra_state files: 16 个
# - .metadata: 1 个
# - Total: 33 个文件
```

### 10.4 DCP 的 Resharding 机制

**Scenario 1: 相同并行度加载**
```python
# Save: 8 GPU (ep=2, fsdp=4)
# Load: 8 GPU (ep=2, fsdp=4)

# DCP 行为：
# - 直接读取对应 rank 的 .distcp 文件
# - 无需 reshard
# - 最快的加载方式
```

**Scenario 2: 不同 FSDP 并行度**
```python
# Save: 8 GPU (ep=2, fsdp=4)
# Load: 16 GPU (ep=2, fsdp=8)

# DCP 行为：
# 1. 读取 .metadata，获取全局 sharding plan
# 2. 根据新的 device_mesh (ep=2, fsdp=8)，计算每个 rank 需要的分片
# 3. 从对应的 .distcp 文件读取数据
# 4. 如果需要，执行 reshard（如切分更细粒度的分片）

# 示例：
# Rank 0 原本负责 [0:16, :, :] (16 experts)
# 现在 Rank 0 和 Rank 8 共同负责这 16 experts
# DCP 会自动将数据分配给两个 rank
```

**Scenario 3: 不同 EP 并行度**
```python
# Save: 8 GPU (ep=2, fsdp=4)
# Load: 16 GPU (ep=4, fsdp=4)

# DCP 行为：
# - EP 维度的 reshard 由 restore_ep_dim/drop_ep_dim 处理
# - DCP 主要负责 FSDP 维度的 reshard
# - 需要确保 .metadata 中的 device_mesh 信息正确
```

### 10.5 Metadata 的作用总结

| 功能 | Metadata 的作用 |
|------|----------------|
| **验证 Checkpoint** | 检查 world_size, dtype, shape 是否匹配 |
| **Resharding** | 根据新的 device_mesh 计算需要读取哪些 chunks |
| **错误恢复** | 如果某个 .distcp 文件损坏，可以从 metadata 重建 |
| **版本管理** | 记录 DCP 版本，确保兼容性 |
| **调试** | 可以直接查看 .metadata 了解 checkpoint 结构 |

---

## 11. 工具函数与 API

### 11.1 build_checkpointer() - 构建 Checkpointer

```python
# veomni/checkpoint/checkpointer.py: 30-31
def build_checkpointer(ckpt_manager: str, dist_backend: str):
    """
    构建 checkpoint 管理器

    参数：
        ckpt_manager: Checkpoint 管理器类型
            - "dcp": Torch Distributed Checkpoint
            - 可扩展支持其他类型（如 "omnistore", "bcp"）
        dist_backend: 分布式后端
            - "fsdp2": FSDP2
            - "fsdp1": FSDP1
            - "ddp": DDP

    返回：
        CheckpointerBase 子类（如 DistributedCheckpointer）

    示例：
        Checkpointer = build_checkpointer(ckpt_manager="dcp", dist_backend="fsdp2")
        Checkpointer.save(path, state, save_async=True, global_steps=1000)
        Checkpointer.load(path, state)
    """
    return CHECKPOINTER_REGISTRY[ckpt_manager](dist_backend)
```

**注册表机制**：
```python
# checkpointer.py: 26-27
CHECKPOINTER_REGISTRY = Registry("checkpointer")

# checkpointer.py: 75-87
@CHECKPOINTER_REGISTRY.register("dcp")
def dcp_checkpointer(dist_backend: str):
    if not is_torch_version_greater_than("2.4"):
        raise ValueError("DCP checkpoint manager requires torch version >= 2.4")
    if dist_backend not in ["ddp", "fsdp1", "fsdp2"]:
        raise ValueError(
            f"Unsupported distributed backend: {dist_backend} for DCP checkpoint manager"
        )
    from .dcp_checkpointer import DistributedCheckpointer
    return DistributedCheckpointer
```

### 11.2 dcp_get_last_iteration() - 查找最新 Checkpoint

```python
# veomni/utils/checkpoint_utils.py: 78-97
def dcp_get_last_iteration(output_dir):
    """
    查找最新的 DCP checkpoint

    参数：
        output_dir: Checkpoint 输出目录（如 "./checkpoints"）

    返回：
        最新的 global_step（int）或 None

    流程：
    1. 扫描 output_dir 下的所有子目录
    2. 检查每个目录是否是有效的 DCP checkpoint:
       - 目录名符合 "global_step_{N}" 格式
       - 包含 .metadata 文件
    3. 返回最大的 global_step

    示例：
        checkpoints/
        ├── global_step_1000/  # Valid
        ├── global_step_2000/  # Valid
        ├── global_step_3000/  # Valid (latest)
        └── temp/              # Invalid (no .metadata)

        dcp_get_last_iteration("./checkpoints") → 3000
    """
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    if not exists(checkpoints_dir):
        logger.warning_rank0("Provided checkpoint path does not exist!")
        return None

    entries = listdir(checkpoints_dir)
    valid_steps = []

    for entry in entries:
        step = _validate_dcp_checkpoint_entry(checkpoints_dir, entry)
        if step is not None:
            valid_steps.append(step)

    if not valid_steps:
        logger.warning_rank0("Provided checkpoint path exists but there are no valid DCP .metadata")
        return None

    logger.info_rank0(f"found valid previously saved checkpointed steps: {checkpoints_dir}/global_step_{valid_steps}")

    return max(valid_steps)
```

**验证函数**：
```python
# checkpoint_utils.py: 34-53
def _validate_dcp_checkpoint_entry(checkpoints_dir: str, entry: str):
    """
    验证目录是否是有效的 DCP checkpoint

    流程：
    1. 检查目录名是否以 "global_step_" 开头
    2. 提取步数（如 "global_step_1000" → 1000）
    3. 检查目录是否存在
    4. 检查 .metadata 文件是否存在

    返回：
        步数（int）或 None
    """
    if not entry.startswith(_GLOBAL_STEP_PREFIX):  # "global_step_"
        return None

    # 提取步数
    step_str = entry[len(_GLOBAL_STEP_PREFIX):]
    try:
        step = int(step_str)
    except ValueError:
        return None

    # 检查目录和 .metadata
    checkpoint_path = os.path.join(checkpoints_dir, entry)
    if not isdir(checkpoint_path):
        return None

    metadata_path = os.path.join(checkpoint_path, ".metadata")
    if not exists(metadata_path):
        return None

    return step
```

### 11.3 get_checkpoint_path() - 获取最新 Checkpoint 路径

```python
# checkpoint_utils.py: 100-113
def get_checkpoint_path(output_dir, is_local_rank0: bool, ckpt_manager: str):
    """
    获取最新的 checkpoint 路径（用于恢复训练）

    参数：
        output_dir: Checkpoint 输出目录
        is_local_rank0: 是否是本地 rank 0
        ckpt_manager: Checkpoint 管理器类型

    返回：
        最新 checkpoint 的完整路径或 None

    示例：
        path = get_checkpoint_path("./outputs", is_local_rank0=True, ckpt_manager="dcp")
        # 返回: "./outputs/checkpoints/global_step_3000"
    """
    if ckpt_manager == "dcp":
        iteration = dcp_get_last_iteration(output_dir)
    else:  # OmniStore or BCP
        iteration = get_last_iteration(output_dir, is_local_rank0)

    if not iteration:
        logger.warning_rank0("Failed to find latest checkpoint path, will start training from step 0...")
        return None

    checkpoint_path = os.path.join(output_dir, "checkpoints", f"global_step_{iteration}")
    logger.info_rank0(f"Sucessfully get the latest checkpoint path: {checkpoint_path}")

    return checkpoint_path
```

### 11.4 dcp_to_torch_state_dict() - DCP → Torch 转换

```python
# veomni/checkpoint/dcp_checkpointer.py: 434-459
def dcp_to_torch_state_dict(save_checkpoint_path: Union[str, os.PathLike]) -> STATE_DICT_TYPE:
    """
    将 DCP checkpoint 转换为普通的 Torch state_dict

    使用场景：
    - 单 GPU 推理：将分布式 checkpoint 转换为单机可用的 state dict
    - 调试：检查 checkpoint 内容
    - 模型转换：转换为其他格式（如 HuggingFace）

    参数：
        save_checkpoint_path: DCP checkpoint 目录

    返回：
        完整的 state_dict（包含所有参数，已合并分片）

    警告：
        - 会将所有参数加载到内存，可能 OOM
        - 建议只在单个 rank 上运行
        - 不支持 optimizer state（只转换 model）

    示例：
        state_dict = dcp_to_torch_state_dict("./checkpoints/global_step_1000")
        # state_dict 是完整的 model state dict，可以用 torch.save 保存
        torch.save(state_dict, "model.pt")
    """
    # 加载 state_dict from DCP checkpoint
    state_dict: STATE_DICT_TYPE = {}

    _load_state_dict(
        state_dict,
        storage_reader=FileSystemReader(save_checkpoint_path),
        planner=_EmptyStateDictLoadPlanner(),  # 加载所有分片
        no_dist=True,                          # 不使用分布式
    )

    # 如果 state_dict 被 flatten 过，需要提取
    if "state" in state_dict:
        state_dict = state_dict["state"]

    return state_dict["model"]
```

**使用示例**：
```python
# 将 DCP checkpoint 转换为单机 checkpoint
from veomni.checkpoint import ckpt_to_state_dict

# 方式 1: 直接调用函数
state_dict = dcp_to_torch_state_dict("./checkpoints/global_step_1000")

# 方式 2: 通过注册表
state_dict = ckpt_to_state_dict(
    save_checkpoint_path="./checkpoints/global_step_1000",
    ckpt_manager="dcp",
)

# 保存为普通 checkpoint
torch.save({"model": state_dict}, "model_single_gpu.pt")

# 单 GPU 加载
model = MyModel(config)
checkpoint = torch.load("model_single_gpu.pt", map_location="cpu")
model.load_state_dict(checkpoint["model"])
```

### 11.5 常用 API 总结

| API | 功能 | 使用场景 |
|-----|------|---------|
| `build_checkpointer()` | 构建 checkpoint 管理器 | 初始化时 |
| `Checkpointer.save()` | 保存 checkpoint | 训练过程中 |
| `Checkpointer.load()` | 加载 checkpoint | 恢复训练 |
| `dcp_get_last_iteration()` | 查找最新 checkpoint | 自动恢复 |
| `get_checkpoint_path()` | 获取最新路径 | 自动恢复 |
| `dcp_to_torch_state_dict()` | DCP → Torch 转换 | 单机推理、调试 |

---

## 12. 最佳实践

### 12.1 Checkpoint 频率选择

```python
# 建议：根据训练成本和 checkpoint 大小平衡

# 小模型（< 10B）：
save_interval = 500  # 每 500 步

# 中等模型（10B - 100B）：
save_interval = 200  # 每 200 步

# 大模型（> 100B）：
save_interval = 100  # 每 100 步，使用 async save

# 使用 async save 时，可以更频繁保存
if save_async:
    save_interval = max(50, save_interval // 2)
```

### 12.2 Checkpoint 目录管理

```python
# 建议：保留多个 checkpoint，避免损坏

max_checkpoints_to_keep = 3

# 删除旧 checkpoint
def cleanup_old_checkpoints(output_dir, max_keep=3):
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    entries = listdir(checkpoints_dir)

    valid_steps = []
    for entry in entries:
        step = _validate_dcp_checkpoint_entry(checkpoints_dir, entry)
        if step is not None:
            valid_steps.append(step)

    if len(valid_steps) > max_keep:
        valid_steps.sort()
        to_delete = valid_steps[:-max_keep]

        for step in to_delete:
            ckpt_path = os.path.join(checkpoints_dir, f"global_step_{step}")
            shutil.rmtree(ckpt_path)
            logger.info_rank0(f"Deleted old checkpoint: {ckpt_path}")
```

### 12.3 Extra State 管理

```python
# 建议：只保存必要的 extra state

# ✓ 推荐：
extra_state = {
    "global_step": global_step,
    "epoch": epoch,
    "lr_scheduler": lr_scheduler.state_dict(),  # 通常很小（< 1 KB）
}

# ✗ 不推荐：
extra_state = {
    "global_step": global_step,
    "dataloader": dataloader,  # 可能很大！
    "full_train_history": train_history,  # 可能很大！
}

# 对于 dataloader，只保存必要信息：
extra_state["dataloader"] = {
    "epoch": dataloader.epoch,
    "shuffle_seed": dataloader.shuffle_seed,
}
```

### 12.4 错误处理

```python
# 建议：添加 try-except 保护

def safe_save_checkpoint(path, state, global_steps):
    try:
        Checkpointer.save(
            path=path,
            state=state,
            global_steps=global_steps,
            save_async=True,
        )
        logger.info_rank0(f"Successfully saved checkpoint at step {global_steps}")
    except Exception as e:
        logger.error_rank0(f"Failed to save checkpoint: {e}")
        # 不要中断训练，记录错误即可

def safe_load_checkpoint(path, state):
    try:
        Checkpointer.load(path=path, state=state)
        logger.info_rank0(f"Successfully loaded checkpoint from {path}")
        return True
    except Exception as e:
        logger.error_rank0(f"Failed to load checkpoint: {e}")
        logger.warning_rank0("Starting training from scratch")
        return False
```

### 12.5 EP+FSDP2 Checkpoint 最佳实践

```python
# 1. 确保 parallel_plan 正确应用
parallel_plan = get_parallel_plan()
parallel_plan.apply(model)  # 添加 _fqn2spec_info

# 2. 检查 EP 检测
checkpointer = DistributedCheckpointer("fsdp2")
model_state = ModelState(model)
assert model_state.should_ep_aware, "EP 未正确检测"

# 3. 验证 save/load 对称性
state_before = {k: v.clone() for k, v in model.state_dict().items()}
Checkpointer.save("./test_ckpt", {"model": model})
model.zero_grad()  # 清空参数
Checkpointer.load("./test_ckpt", {"model": model})
state_after = model.state_dict()

for k in state_before:
    assert torch.allclose(state_before[k], state_after[k]), f"Mismatch in {k}"
```

### 12.6 性能优化

```python
# 1. 使用 async save
Checkpointer.save(..., save_async=True)

# 2. 调整 FileSystemWriter 参数
storage_writer = FileSystemWriter(
    checkpoint_dir,
    thread_count=32,           # 增加线程数（如果 I/O 是瓶颈）
    single_file_per_rank=True, # 推荐：减少文件数
    sync_files=False,          # 推荐：更快（除非需要强一致性）
)

# 3. 在保存前清理内存
gc.collect()
empty_cache()
synchronize()

# 4. 使用更快的文件系统
# - 本地 NVMe SSD > 网络文件系统 (NFS)
# - 如果使用 NFS，确保网络带宽足够
```

---

## 13. 限制与注意事项

### 13.1 已知限制

**1. PyTorch 版本要求**：
```python
# DCP 需要 PyTorch >= 2.4
if not is_torch_version_greater_than("2.4"):
    raise ValueError("DCP checkpoint manager requires torch version >= 2.4")
```

**2. EP Resharding 限制**：
```python
# 当前实现不支持在不同 EP size 下恢复
# 示例：从 ep_size=2 的 checkpoint 恢复到 ep_size=4

# 原因：
# - restore_ep_dim/drop_ep_dim 假设 EP mesh 结构不变
# - DCP 无法正确理解 EP 维度的 resharding

# 解决方案：
# - 使用相同的 ep_size 恢复
# - 或者实现自定义 resharding 逻辑
```

**3. Extra State Resharding**：
```python
# Extra state 是 per-rank 的，无法自动 reshard

# 示例：
# Save: 8 GPU → 8 个 extra_state_rank_*.pt 文件
# Load: 16 GPU → 每个 rank 应该加载哪个文件？

# 解决方案：
# - 只在相同 world_size 下使用 extra_state
# - 或者只保存全局信息（rank 0 保存，其他 rank 广播）
```

**4. FSDP1 vs FSDP2 差异**：
```python
# FSDP1:
# - 使用 CheckpointExtensions 和 hooks
# - EP 维度通过 state_dict_post_hook 添加

# FSDP2:
# - 使用 Stateful wrappers
# - EP 维度通过 restore_ep_dim/drop_ep_dim 处理

# 注意：两者的 checkpoint 格式相同，但内部机制不同
```

### 13.2 常见错误

**错误 1: `"model" not in state`**
```python
# 错误：
Checkpointer.save(path, {"model_state": model})  # ✗ 错误的 key

# 正确：
Checkpointer.save(path, {"model": model})  # ✓ 必须是 "model"
```

**错误 2: EP 未正确检测**
```python
# 错误：
model = MyModel(config)
Checkpointer.save(path, {"model": model})  # ✗ 没有 _fqn2spec_info

# 正确：
model = MyModel(config)
parallel_plan = get_parallel_plan()
parallel_plan.apply(model)  # ✓ 添加 _fqn2spec_info
Checkpointer.save(path, {"model": model})
```

**错误 3: Async save 未等待完成**
```python
# 错误：
Checkpointer.save(path, state, save_async=True)
# 训练结束，进程退出
# ✗ Async save 可能未完成！

# 正确：
Checkpointer.save(path, state, save_async=True)
# 训练结束前：
if Checkpointer.dcp_save_future is not None:
    Checkpointer.dcp_save_future.result()  # ✓ 等待完成
```

**错误 4: Checkpoint 路径不存在**
```python
# 错误：
Checkpointer.load("./checkpoints/global_step_1000", state)
# FileNotFoundError: .metadata not found

# 原因：
# - 路径错误
# - Checkpoint 未保存成功
# - .metadata 文件损坏

# 调试：
import os
print(os.path.exists("./checkpoints/global_step_1000/.metadata"))
```

### 13.3 性能陷阱

**1. 同步 Save 阻塞训练**：
```python
# 问题：
Checkpointer.save(path, state, save_async=False)
# 每次 save 阻塞 10-30 秒

# 解决方案：
Checkpointer.save(path, state, save_async=True)  # ✓ Async save
```

**2. Extra State 过大**：
```python
# 问题：
extra_state = {
    "train_history": train_history,  # 100 MB
}
# 每个 rank 保存 100 MB → 总共 800 MB（8 GPU）

# 解决方案：
# 只保存必要信息
extra_state = {
    "global_step": global_step,
    "lr_scheduler": lr_scheduler.state_dict(),
}
```

**3. 文件系统 I/O 瓶颈**：
```python
# 问题：
# 使用 NFS，写入速度慢（< 100 MB/s）

# 解决方案：
# 1. 使用本地 SSD
# 2. 增加 thread_count
storage_writer = FileSystemWriter(..., thread_count=32)
# 3. 使用更快的文件系统（如 Lustre, GPFS）
```

---

## 14. 参考资料

### 14.1 论文

1. **PyTorch FSDP**: [arxiv.org/abs/2304.11277](https://arxiv.org/abs/2304.11277)
   - FSDP 的官方论文
   - 详细描述 FSDP 的分片策略和通信模式

2. **DeepSpeed**: [arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)
   - ZeRO optimizer 的原始论文
   - FSDP 的核心思想来源

3. **VeOmni Technical Report**: [arxiv.org/abs/2508.02317](https://arxiv.org/abs/2508.02317)
   - VeOmni 框架的官方文档
   - 包含 EP+FSDP2 的架构设计

### 14.2 PyTorch 官方文档

- **Distributed Checkpoint**:
  - [PyTorch DCP Tutorial](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)
  - [DCP API Reference](https://pytorch.org/docs/stable/distributed.checkpoint.html)

- **FSDP**:
  - [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
  - [FSDP API Reference](https://pytorch.org/docs/stable/fsdp.html)

- **DTensor**:
  - [DTensor Tutorial](https://pytorch.org/tutorials/recipes/distributed_tensor_parallel.html)
  - [DTensor API Reference](https://pytorch.org/docs/stable/distributed.tensor.html)

### 14.3 VeOmni 源码

```
veomni/checkpoint/
├── checkpointer.py             # Line 1-99
├── dcp_checkpointer.py         # Line 1-459

veomni/utils/
├── checkpoint_utils.py         # Line 1-114

veomni/distributed/fsdp/
├── extension.py                # Line 1-452
├── initialize.py               # Line 1-350
```

### 14.4 相关测试

```
tests/
├── checkpoint/
│   ├── test_dcp_checkpointer.py      # DCP 基础功能测试
│   ├── test_ep_checkpoint.py         # EP checkpoint 测试
│   └── test_async_save.py            # Async save 测试
│
└── integration/
    └── test_trainer_saveload.py     # 端到端训练 save/load 测试
```

---

## 总结

VeOmni 的 Torch Distributed Checkpoint 实现提供了：

1. **完整的 FSDP/FSDP2 支持**：自动处理 DTensor 的分片和 resharding
2. **Expert Parallelism (EP) 集成**：通过维度转换无缝支持 EP+FSDP2 混合并行
3. **灵活的存储后端**：支持 FileSystem、可扩展到 S3 等云存储
4. **异步保存优化**：最小化训练中断时间，支持后台保存
5. **Extra State 管理**：独立保存 non-DCP 组件（lr_scheduler, dataloader 等）
6. **Stateful Wrapper 设计**：清晰的抽象，易于扩展和维护
7. **FSDP1 扩展机制**：通过 hooks 为 FSDP1 提供 EP 支持

**核心技术亮点**：
- 通过 `restore_ep_dim()` / `drop_ep_dim()` 实现 EP 维度的透明转换
- 使用专用 Gloo process group 实现异步保存
- Per-rank extra state 管理避免跨 rank 通信
- 统一的 Checkpointer 接口支持多种后端

**适用场景**：
- 大规模分布式训练（FSDP1/FSDP2 + EP）
- 需要频繁 checkpoint 的训练任务
- 多种并行策略的混合训练
- Checkpoint resharding（不同并行度恢复）

---

**文档完成时间**：2026-01-03
**总字数**：约 20,000 字
**代码覆盖**：VeOmni checkpoint 相关所有核心文件
**基于版本**：VeOmni main branch (commit: 441e1b2)
