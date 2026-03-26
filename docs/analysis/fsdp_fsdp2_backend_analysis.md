# VeOmni FSDP & FSDP2 Backend 深度分析

> **作者**: Claude Code (Anthropic)
> **日期**: 2026-01-04
> **版本**: v1.1
> **代码库版本**: VeOmni v0.1.0
> **最后更新**: 2026-02-10

---

## 目录

1. [概述](#1-概述)
2. [核心架构](#2-核心架构)
3. [并行状态管理 (ParallelState)](#3-并行状态管理-parallelstate)
4. [FSDP1 实现](#4-fsdp1-实现)
5. [FSDP2 实现](#5-fsdp2-实现)
6. [Expert Parallelism (EP) 集成](#6-expert-parallelism-ep-集成)
7. [参数分片策略](#7-参数分片策略)
8. [梯度同步与裁剪](#8-梯度同步与裁剪)
9. [混合精度训练](#9-混合精度训练)
10. [优化器集成](#10-优化器集成)
11. [检查点管理](#11-检查点管理)
12. [设备网格 (DeviceMesh) 拓扑](#12-设备网格-devicemesh-拓扑)
13. [性能优化](#13-性能优化)
14. [限制与注意事项](#14-限制与注意事项)
15. [参考资料](#15-参考资料)

---

## 1. 概述

### 1.1 什么是 FSDP 和 FSDP2？

**FSDP (Fully Sharded Data Parallel)** 是 PyTorch 提供的分布式训练范式，用于在多个设备上分片模型参数、梯度和优化器状态，以突破单设备显存限制。

**FSDP2 (Fully Sharded Data Parallel 2)** 是 PyTorch 2.4+ 引入的可组合 API，基于 `fully_shard()` 装饰器，提供更灵活的参数分片和更好的 DTensor 集成。

### 1.2 VeOmni 的设计目标

VeOmni 框架实现了 FSDP1 和 FSDP2 两种 backend，具有以下特点：

- **统一接口**: 通过 `dp_mode` 参数在 FSDP1/FSDP2/DDP 之间无缝切换
- **EP 感知**: 深度集成 Expert Parallelism，支持 MoE 模型高效训练
- **混合精度**: 支持 BF16 参数 + FP32 梯度通信的混合精度训练
- **HSDP 支持**: 支持 Hybrid Sharded Data Parallel (复制 + 分片两级拓扑)
- **检查点一致性**: 统一的 DCP (Distributed Checkpoint) 格式

### 1.3 核心文件结构

```
veomni/distributed/
├── parallel_state.py (579行)        # 并行状态管理和 DeviceMesh 配置
├── torch_parallelize.py (523行)     # FSDP1/FSDP2 入口函数
├── parallel_plan.py (170行)         # Expert Parallelism 计划系统
├── fsdp/
│   ├── initialize.py (349行)        # FSDP1 参数初始化和加载
│   ├── extension.py (451行)         # FSDP1 DTensor 扩展和检查点钩子
│   └── clip_grad_norm.py (137行)    # FSDP1 EP 感知梯度裁剪
└── fsdp2/
    └── clip_grad_norm.py (170行)    # FSDP2 EP 感知梯度裁剪

veomni/optim/
└── optimizer.py (443行)             # MultiOptimizer (EP+FSDP2)

veomni/models/transformers/
└── qwen3_moe/parallel_plan.py       # Qwen3-MoE EP 计划示例
```

---

## 2. 核心架构

### 2.1 并行训练流程

```
┌────────────────────────────────────────────────────────────────┐
│                     初始化阶段                                   │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  init_parallel_state()                                         │
│  ├─ 创建主 DeviceMesh: [PP, DP-Replicate, DP-Shard, SP, TP]   │
│  └─ 创建 EP-FSDP DeviceMesh: [EP, EP-FSDP]                     │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  build_parallelize_model()                                     │
│  ├─ TP: parallelize_module(model, tp_mesh)                     │
│  └─ DP: FSDP1 / FSDP2 / DDP 选择                               │
└────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────┬──────────────┐
        │                 │              │
   ┌────▼────┐      ┌────▼────┐   ┌────▼────┐
   │ FSDP1   │      │ FSDP2   │   │  DDP    │
   └────┬────┘      └────┬────┘   └────┬────┘
        │                │              │
        └────────────────┴──────────────┘
                      ↓
┌────────────────────────────────────────────────────────────────┐
│                     训练循环                                     │
│  ├─ Forward: 自动 all-gather 参数                              │
│  ├─ Backward: 计算梯度 + all-reduce                            │
│  ├─ Gradient Clipping: EP 感知的梯度裁剪                        │
│  ├─ Optimizer Step: MultiOptimizer (EP+FSDP2) 或标准优化器     │
│  └─ 参数再分片 (FSDP2: reshard_after_forward)                  │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                  检查点保存/加载                                 │
│  ├─ DCP (Torch Distributed Checkpoint)                         │
│  ├─ EP 维度临时恢复 (FSDP2)                                     │
│  └─ FSDPExtensions 钩子 (FSDP1)                                │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 FSDP1 vs FSDP2 对比

| 维度 | FSDP1 | FSDP2 |
|------|-------|-------|
| **API** | `FullyShardedDataParallel(model, **kwargs)` | `fully_shard(module, **kwargs)` |
| **包装方式** | 整体包装 + `auto_wrap_policy` | 模块级组合，自底向上 |
| **混合精度** | `MixedPrecision` 配置对象 | `MixedPrecisionPolicy` |
| **EP 支持** | 通过 `FSDPExtensions` 扩展 | 原生 DTensor，更灵活 |
| **优化器** | 单个优化器 | MultiOptimizer (EP+非EP 分离) |
| **梯度裁剪** | `FSDP.clip_grad_norm_()` | `model.clip_grad_norm_()` |
| **检查点** | FSDPExtensions 钩子自动处理 | 需手动恢复/删除 EP 维度 |
| **预取** | 自动预取 / forward_prefetch | 手动预取 (EP+FSDP2 必需) |
| **PyTorch 版本** | < 2.5 | >= 2.4 |
| **推荐场景** | 稳定生产环境、兼容旧版 | 新模型、复杂并行策略 |

---

## 3. 并行状态管理 (ParallelState)

### 3.1 ParallelState 数据类

**文件**: `veomni/distributed/parallel_state.py:78-93`

```python
@dataclass(frozen=True)
class ParallelState:
    # 数据并行维度
    dp_size: int = 1                    # 总数据并行大小
    dp_replicate_size: int = 1          # HSDP 复制维度
    dp_shard_size: int = 1              # HSDP 分片维度

    # 模型并行维度
    tp_size: int = 1                    # 张量并行大小
    pp_size: int = 1                    # 管道并行大小

    # 序列并行维度
    ulysses_size: int = 1               # Ulysses 序列并行大小
    cp_size: int = 1                    # 上下文并行大小

    # 专家并行维度
    ep_size: int = 1                    # 专家并行大小

    # 配置
    dp_mode: Literal["ddp", "fsdp1", "fsdp2"] = "fsdp1"
    device_type: str = get_device_type()
    include_sp_in_fsdp: bool = True     # 序列并行包含在 FSDP 中

    # 设备网格
    device_mesh: Optional["DeviceMesh"] = None
    ep_fsdp_device_mesh: Optional["DeviceMesh"] = None

    # 异步模式
    async_enabled: Optional[bool] = False  # 是否启用异步序列并行
```

**关键设计**:
- **frozen=True**: 并行状态一旦初始化不可修改，确保训练一致性
- **两个 DeviceMesh**:
  - `device_mesh`: 主网格 (PP, DP-Replicate, DP-Shard, Ulysses, CP, TP)
  - `ep_fsdp_device_mesh`: EP 专用网格 (EP, EP-FSDP)

### 3.2 DeviceMesh 初始化

**文件**: `veomni/distributed/parallel_state.py:449-568`

```python
def init_parallel_state(
    dp_size: int = 1,
    dp_replicate_size: int = 1,
    dp_shard_size: int = 1,
    tp_size: int = 1,
    ep_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    ulysses_size: int = 1,
    dp_mode: Literal["ddp", "fsdp1", "fsdp2"] = "fsdp1",
    device_type: str = None,
    include_sp_in_fsdp: bool = True,
    ep_outside: bool = False,
    async_enabled: Optional[bool] = False,
) -> None:
```

**步骤 1: 构建主 DeviceMesh**

```python
# 第 485-499 行
mesh_shape = []
mesh_dim_names = []
for d, name in zip(
    [pp_size, dp_replicate_size, dp_shard_size, ulysses_size, cp_size, tp_size],
    ["pp", "dp_replicate", "dp_shard", "ulysses", "cp", "tp"],
):
    if d > 1 or name in ["dp_shard"]:
        mesh_shape.append(d)
        mesh_dim_names.append(name)

device_mesh = init_device_mesh(
    device_type=device_type,
    mesh_shape=tuple(mesh_shape),
    mesh_dim_names=tuple(mesh_dim_names),
)
```

**关键点**:
- `dp_shard` 即使大小为 1 也会加入网格 (用于 FSDP 一致性)
- 其他维度只有在 `> 1` 时才加入

**步骤 2: 创建 EP-FSDP DeviceMesh**

```python
# 第 538-548 行
if ep_size > 1:
    world_size = dist.get_world_size()
    assert world_size % ep_size == 0
    ep_fsdp_size = world_size // ep_size

    mesh = init_ep_mesh_matrix(ep_size=ep_size, ep_fsdp_size=ep_fsdp_size, ep_outside=ep_outside)
    ep_fsdp_device_mesh = DeviceMesh(
        device_type=device_type,
        mesh=mesh,
        mesh_dim_names=("ep", "ep_fsdp"),
    )
```

**EP 网格矩阵构建** (`parallel_state.py:57-75`):

```python
def init_ep_mesh_matrix(ep_size: int, ep_fsdp_size: int, ep_outside: bool = False):
    if ep_outside:
        # EP 在外层: [ep_size, ep_fsdp_size]
        mesh = torch.arange(ep_size * ep_fsdp_size).view(ep_size, ep_fsdp_size)
    else:
        # EP 在内层: [ep_fsdp_size, ep_size].transpose()
        mesh = (
            torch.arange(ep_size * ep_fsdp_size)
            .view(ep_fsdp_size, ep_size)
            .transpose(0, 1)
        )
    return mesh
```

**ep_outside 参数影响**:
- `ep_outside=False` (默认): 相同 EP rank 的进程物理上相邻，有利于 GPU 间通信
- `ep_outside=True`: 不同 EP rank 的进程交错分布

**示例**: 假设 `world_size=16`, `ep_size=4`, `ep_fsdp_size=4`

```
ep_outside=False:
┌─────────────────┐
│ EP=0  EP=1  EP=2  EP=3 │
├─────────────────┤
│  0     4     8    12   │  ep_fsdp=0
│  1     5     9    13   │  ep_fsdp=1
│  2     6    10    14   │  ep_fsdp=2
│  3     7    11    15   │  ep_fsdp=3
└─────────────────┘

ep_outside=True:
┌─────────────────┐
│ ep_fsdp=0  ep_fsdp=1  ep_fsdp=2  ep_fsdp=3 │
├─────────────────┤
│    0        1        2        3    │  EP=0
│    4        5        6        7    │  EP=1
│    8        9       10       11    │  EP=2
│   12       13       14       15    │  EP=3
└─────────────────┘
```

### 3.3 复合网格 (Flattened Mesh)

**文件**: `veomni/distributed/parallel_state.py:501-536`

```python
# 数据加载网格 (无通信)
dp_mesh_dim_names = []
# 参数分片网格
dp_shard_sp_mesh_dim_names = []
# 损失 all-reduce 网格
dp_sp_mesh_dim_names = []
# 序列并行网格
sp_mesh_dim_names = []

if dp_replicate_size > 1:
    dp_mesh_dim_names.append("dp_replicate")
    dp_sp_mesh_dim_names.append("dp_replicate")
if dp_shard_size >= 1:
    dp_mesh_dim_names.append("dp_shard")
    dp_shard_sp_mesh_dim_names.append("dp_shard")
    dp_sp_mesh_dim_names.append("dp_shard")
if ulysses_size > 1:
    dp_shard_sp_mesh_dim_names.append("ulysses")
    sp_mesh_dim_names.append("ulysses")
    dp_sp_mesh_dim_names.append("ulysses")
if cp_size > 1:
    dp_shard_sp_mesh_dim_names.append("cp")
    sp_mesh_dim_names.append("cp")
    dp_sp_mesh_dim_names.append("cp")

# 展平成命名网格
if dp_mesh_dim_names != []:
    device_mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
if dp_shard_sp_mesh_dim_names != []:
    device_mesh[tuple(dp_shard_sp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_sp")
if dp_sp_mesh_dim_names != []:
    device_mesh[tuple(dp_sp_mesh_dim_names)]._flatten(mesh_dim_name="dp_sp")
if sp_mesh_dim_names != []:
    device_mesh[tuple(sp_mesh_dim_names)]._flatten(mesh_dim_name="sp")
```

**四种复合网格用途**:

1. **`dp`** (数据并行): `dp_replicate + dp_shard`
   - 用于数据加载器的分布式采样
   - 确保每个全局 batch 不重复

2. **`dp_shard_sp`** (参数分片): `dp_shard + ulysses + cp`
   - FSDP 参数分片的实际网格
   - 包含序列并行维度

3. **`dp_sp`** (损失 all-reduce): `dp_replicate + dp_shard + ulysses + cp`
   - 用于损失值的全局 all-reduce
   - 跨所有数据并行和序列并行 rank

4. **`sp`** (序列并行): `ulysses + cp`
   - Ulysses 序列并行通信组
   - Context Parallel (Ring Attention) 通信组

### 3.4 关键属性和方法

**FSDP 通信组** (`parallel_state.py:235-238`):

```python
@property
def fsdp_group(self) -> Optional["ProcessGroup"]:
    if self.device_mesh is not None:
        return self.device_mesh.get_group("dp_sp")
```

> **重要**: `fsdp_group` 返回的是 `dp_sp` 组（包含 `dp_replicate + dp_shard + ulysses + cp`），而不是纯粹的 FSDP 分片组。这意味着 FSDP 的通信组包含了数据并行和序列并行的所有 rank。这一设计使得损失 all-reduce 和 FSDP 梯度同步使用相同的通信组，确保了梯度裁剪和损失缩放的一致性。

**FSDP 大小** (`parallel_state.py:275-277`):

```python
@property
def fsdp_size(self) -> int:
    return self.world_size // (self.pp_size * self.tp_size)
```

**用途**: 在 `mean_global_loss()` 中用于补偿 FSDP 的梯度除法 (详见 [13.6](#136-损失缩放与-fsdp-梯度补偿))。

**FSDP 启用状态** (`parallel_state.py:272-273`):

```python
@property
def fsdp_enabled(self) -> bool:
    return self.fsdp_size > 1
```

> **注意**: `fsdp_enabled` 依赖 `fsdp_size`，即 `world_size // (pp_size * tp_size) > 1`。这意味着只要有至少 2 个进程参与（且不是被 PP/TP 完全占用），FSDP 就被认为是"启用"的——即使 `dp_mode` 设置为 `"ddp"` 也是如此。`build_parallelize_model` 根据此属性决定是否调用 FSDP/DDP 包装。

**FSDP 网格** (`parallel_state.py:252-269`):

```python
@property
def fsdp_mesh(self) -> "DeviceMesh":
    if self.dp_replicate_enabled:
        # HSDP
        if self.dp_shard_sp_enabled:
            return self.device_mesh["dp_replicate", "dp_shard_sp"]
        elif self.dp_shard_enabled:
            return self.device_mesh["dp_replicate", "dp_shard"]
        else:
            # DDP
            return self.device_mesh["dp_replicate"]
    # FSDP
    elif self.dp_shard_sp_enabled:
        return self.device_mesh["dp_shard_sp"]
    elif self.dp_shard_enabled:
        return self.device_mesh["dp_shard"]
    else:
        return self.device_mesh["dp"]
```

**EP 梯度除因子** (`parallel_state.py:348-357`):

```python
@property
def ep_gradient_divide_factor(self) -> int:
    # EP+FSDP2 的梯度除因子总是 world_size
    assert self.tp_size == 1  # 暂不支持 TP
    assert self.pp_size == 1  # 暂不支持 PP
    # FSDP2 + EP: 梯度除因子 = world_size
    # SP 不影响此值，因为 SP 组仍然复制参数
    return self.world_size
```

**设计原因**:
- FSDP2 的 DTensor all-reduce 默认除以进程组大小
- EP 参数需要在 EP 组和 FSDP 组上分别 all-reduce
- 为保持梯度尺度一致，统一除以 `world_size`

---

## 4. FSDP1 实现

### 4.1 入口函数

**文件**: `veomni/distributed/torch_parallelize.py:84-234`

```python
def parallelize_model_fsdp1(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    enable_full_shard: bool = True,
    enable_shard_grad_op: bool = False,
    enable_mixed_precision: bool = True,
    use_orig_params: bool = True,
    basic_modules: Optional[List[str]] = None,
    fsdp_no_shard_states=None,
    fsdp_no_shard_states_fqn=None,
    ep_param_suffix=None,
    fqn2spec_info=None,
    **kwargs,
) -> "nn.Module":
```

### 4.2 流程详解

**步骤 1: 应用 Expert Parallelism** (`torch_parallelize.py:104-117`)

```python
parallel_state = get_parallel_state()

if parallel_state.ep_enabled:
    parallel_plan = model.get_parallel_plan()
    fqn2spec_info = parallel_plan.apply(model, parallel_state.ep_fsdp_device_mesh)

    fsdp_no_shard_states_fqn_to_module = parallel_plan.get_fsdp_no_shard_info(model)
    fsdp_no_shard_states = list(fsdp_no_shard_states_fqn_to_module.values())
    fsdp_no_shard_states_fqn = list(fsdp_no_shard_states_fqn_to_module.keys())
else:
    fqn2spec_info = None
    fsdp_no_shard_states = None
    fsdp_no_shard_states_fqn = None
```

**ParallelPlan.apply()** 详解见 [6.2 ParallelPlan 系统](#62-parallelplan-系统)

**步骤 2: 配置 FSDP 包装策略** (`torch_parallelize.py:119-136`)

```python
wrap_policy = partial(
    lambda_auto_wrap_policy,
    lambda_fn=lambda module: module.__class__.__name__ in basic_modules
)

# 确定分片策略
if parallel_state.fsdp_mesh.ndim > 1 and parallel_state.fsdp_mesh.size() > 1:
    strategy = ShardingStrategy.HYBRID_SHARD if enable_full_shard else ShardingStrategy._HYBRID_SHARD_ZERO2
else:
    strategy = ShardingStrategy.FULL_SHARD if enable_full_shard else ShardingStrategy.SHARD_GRAD_OP

fsdp_kwargs = {
    "auto_wrap_policy": wrap_policy,
    "ignored_states": fsdp_no_shard_states,
    "device_id": get_device_id(),
    "sharding_strategy": strategy if enable_full_shard or enable_shard_grad_op else ShardingStrategy.NO_SHARD,
    "use_orig_params": use_orig_params,
}

fsdp_kwargs["device_mesh"] = parallel_state.fsdp_mesh

# 支持外部传入额外的 FSDP 参数
fsdp_kwargs.update(kwargs.pop("fsdp_kwargs", {}))
```

**分片策略选择**:

| 条件 | HSDP (2D mesh) | 非 HSDP (1D mesh) |
|------|---------------|------------------|
| `enable_full_shard=True` | `HYBRID_SHARD` | `FULL_SHARD` |
| `enable_shard_grad_op=True` | `_HYBRID_SHARD_ZERO2` | `SHARD_GRAD_OP` |
| 两者都为 False | `NO_SHARD` | `NO_SHARD` |

- **HYBRID_SHARD**: HSDP (2D mesh)，复制 + 全分片 (ZeRO-3)
- **_HYBRID_SHARD_ZERO2**: HSDP (2D mesh)，复制 + 梯度/优化器分片 (ZeRO-2)
- **FULL_SHARD**: 标准 FSDP (1D mesh)，全分片 (ZeRO-3)
- **SHARD_GRAD_OP**: 标准 FSDP (1D mesh)，梯度/优化器分片 (ZeRO-2)
- **NO_SHARD**: DDP 模式，不分片 (用于调试)

> **注意**: `enable_full_shard` 和 `enable_shard_grad_op` 不能同时为 True，在函数开头有断言检查。`fsdp_kwargs.update(kwargs.pop("fsdp_kwargs", {}))` 提供了一个扩展点，允许调用方传入额外的 FSDP 配置。

**步骤 3: 混合精度配置** (`torch_parallelize.py:138-148`)

```python
if enable_mixed_precision:
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,      # 参数精度
        reduce_dtype=torch.float32,      # 通信精度
        buffer_dtype=torch.float32,      # 缓冲区精度
    )
    if hasattr(model, "get_ignore_modules_in_mixed_precision"):
        mixed_precision._module_classes_to_ignore += model.get_ignore_modules_in_mixed_precision()

    fsdp_kwargs["mixed_precision"] = mixed_precision
```

**步骤 4: 初始化模式** (`torch_parallelize.py:150-169`)

VeOmni 支持三种初始化模式：

**模式 A: GPU/NPU 直接初始化**

```python
# 默认模式，init_device 不指定
# 每个 rank 独立初始化参数，然后 FSDP 自动分片
```

**模式 B: CPU Rank0 初始化** (`torch_parallelize.py:150-154`)

```python
if kwargs.get("init_device") == "cpu":
    fsdp_kwargs["sync_module_states"] = True
    if parallel_state.global_rank != 0:
        fsdp_kwargs["param_init_fn"] = init_fsdp_fn(model, device=get_device_type())
```

- Rank 0 在 CPU 上初始化完整模型
- 其他 rank 使用 `init_fsdp_fn` 创建空 tensor
- FSDP 自动同步参数 (`sync_module_states=True`)

**模式 C: Meta 设备初始化** (`torch_parallelize.py:155-169`)

```python
elif kwargs.get("init_device") == "meta":
    if weights_path is None:
        logger.info_rank0("weights_path is None during meta initialization.")

    shard_states = kwargs.get("shard_states", {})
    if not shard_states and weights_path:
        shard_states = parallel_load_safetensors(weights_path)

    # 转换权重键名以匹配 HuggingFace 格式
    shard_states = _convert_state_dict_keys(shard_states, model)

    fsdp_kwargs["param_init_fn"] = parallel_init_fsdp_fn(
        model,
        shard_states.copy(),
        ignore_states=fsdp_no_shard_states,
        strict=kwargs.pop("strict", False),
    )
```

> **注意**: `_convert_state_dict_keys` (`torch_parallelize.py:80-81`) 使用 `convert_weight_key()` 将权重键名转换为模型期望的格式，确保 HuggingFace 权重文件的键名与 VeOmni 模型兼容。

- 模型在 meta 设备上创建 (无内存占用)
- `parallel_load_safetensors` 并行加载权重文件
- `parallel_init_fsdp_fn` 在初始化时加载和广播参数

详见 [4.3 参数初始化](#43-参数初始化)

**步骤 5: 应用 FSDP 包装** (`torch_parallelize.py:182-216`)

```python
# 首先包装根模型
model = FullyShardedDataParallel(model, **fsdp_kwargs)

# 如果启用了 EP，单独包装专家模块
if fsdp_no_shard_states is not None:
    # 确定专家模块的分片策略
    if parallel_state.ep_fsdp_mesh["ep_fsdp"].size() == 1:
        moe_sharding_strategy = ShardingStrategy.NO_SHARD
        ep_fsdp_device_mesh = parallel_state.fsdp_mesh
    else:
        moe_sharding_strategy = ShardingStrategy.FULL_SHARD
        ep_fsdp_device_mesh = parallel_state.ep_fsdp_mesh["ep_fsdp"]

    fsdp_kwargs.pop("ignored_states", None)
    fsdp_kwargs.pop("auto_wrap_policy", None)
    fsdp_kwargs["sharding_strategy"] = moe_sharding_strategy
    fsdp_kwargs["device_mesh"] = ep_fsdp_device_mesh

    for fqn in fsdp_no_shard_states_fqn:
        no_shard_module = get_module_from_path(model, fqn)
        # ... 处理 meta 初始化的权重 ...
        fsdp_module = FullyShardedDataParallel(no_shard_module, **fsdp_kwargs)
        fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(fsdp_module)
        fsdp_state._gradient_postdivide_factor *= parallel_state.ep_size
        set_module_from_path(model, fqn, fsdp_module)
```

**关键点**:
- 专家模块使用 `ep_fsdp` 网格 (不包含 EP 维度)
- 梯度除因子乘以 `ep_size`，补偿 EP 维度的梯度分割

**步骤 5.5: 延迟初始化** (`torch_parallelize.py:216`)

```python
_lazy_init(model, model)
```

> FSDP 包装完成后，调用 `_lazy_init` 完成内部状态的初始化。这是 PyTorch FSDP1 的标准步骤，确保所有 FSDP 相关的内部数据结构被正确设置。

**步骤 6: 注册检查点扩展** (`torch_parallelize.py:218-225`)

```python
save_hook_mesh = parallel_state.ep_fsdp_device_mesh if parallel_state.ep_enabled else None
register_checkpoint_extension(
    fsdp_model=model,
    save_hook_mesh=save_hook_mesh,
    fqn2spec_info=fqn2spec_info,
)
```

详见 [4.4 检查点扩展](#44-检查点扩展-fsdpextensions)

**步骤 7: 绑定 EP 感知梯度裁剪** (`torch_parallelize.py:227-228`)

```python
if parallel_state.ep_enabled:
    model.clip_grad_norm_ = types.MethodType(clip_grad_norm_, model)
```

详见 [8.1 FSDP1 梯度裁剪](#81-fsdp1-梯度裁剪)

### 4.3 参数初始化

**文件**: `veomni/distributed/fsdp/initialize.py`

#### 4.3.1 并行加载 SafeTensors

**函数**: `parallel_load_safetensors` (`initialize.py:38-88`)

```python
def parallel_load_safetensors(
    filepath: str,
    specific_param_name: list[str] = None,
    ignore_param_name: list[str] = None
):
```

**流程**:

1. **下载模型到本地缓存** (`initialize.py:44-45`)

```python
filepath = copy_to_local(src=filepath, cache_dir=f"{CACHE_DIR}/models", verbose=True)
dist.barrier()
```

2. **解析索引文件** (`initialize.py:48-66`)

```python
safetensors2param = {}
index_file = os.path.join(filepath, "model.safetensors.index.json")
if os.path.exists(index_file):
    index = json.load(open(index_file, "rb"))
    for param_name, filename in index["weight_map"].items():
        if specific_param_name is not None:
            if param_name not in specific_param_name:
                continue
        elif ignore_param_name is not None:
            if param_name in ignore_param_name:
                continue
        safetensors2param.setdefault(filename, []).append(param_name)
else:
    # 单一文件模式
    param_file = os.path.join(filepath, "model.safetensors")
    states = load_file(param_file)
    for param_name in states:
        safetensors2param.setdefault("model.safetensors", []).append(param_name)
```

3. **负载均衡分配文件** (`initialize.py:68-72`)

```python
total_files = len(safetensors2param)
ckpt_chunks = sorted(safetensors2param.keys())
world_size = dist.get_world_size()
size = int(math.ceil(total_files / world_size))
ckpt_chunks = [ckpt_chunks[i * size : (i + 1) * size] for i in range(world_size)]
```

4. **每个 rank 加载分配的文件** (`initialize.py:74-88`)

```python
shard_states = {}
device = get_device_id()
for rank, files in enumerate(ckpt_chunks):
    if rank == dist.get_rank():
        for file in files:
            safetensors_file = os.path.join(filepath, file)
            states = load_file(safetensors_file, device=device)
            valid_states = {k: v for k, v in states.items() if k in safetensors2param[file]}
            shard_states.update(valid_states)
            del states
    else:
        for file in files:
            for param_name in safetensors2param[file]:
                shard_states[param_name] = rank  # 记录所在 rank
return shard_states
```

**返回值**: `Dict[str, Union[torch.Tensor, int]]`
- 如果参数在本 rank: `{param_name: tensor}`
- 如果参数在其他 rank: `{param_name: rank_id}`

#### 4.3.2 FSDP 初始化函数

**函数**: `parallel_init_fsdp_fn` (`initialize.py:91-318`)

```python
def parallel_init_fsdp_fn(
    module: torch.nn.Module,
    shard_states: Dict[str, torch.nn.Parameter],
    remove_standalone: bool = True,
    ignore_states: list[torch.nn.Module] = None,
    strict: bool = False,
):
```

**核心逻辑**: 返回一个初始化函数 `init_fn`，FSDP 会对每个子模块调用它

**步骤 1: 识别共享参数** (`initialize.py:109-123`)

```python
state2fqn = {}
for name, state in itertools.chain(
    module.named_parameters(remove_duplicate=False),
    module.named_buffers(remove_duplicate=False)
):
    state2fqn.setdefault(state, []).append(name)

shared = {s for s, names in state2fqn.items() if len(names) > 1} if remove_standalone else set(state2fqn.keys())

ignore_modules = set()
if ignore_states:
    for state in ignore_states:
        assert isinstance(state, torch.nn.Module)
        ignore_modules.add(state)
        ignore_modules.update(state.modules())
```

**步骤 2: 辅助函数 - 创建完整张量** (`initialize.py:127-145`)

```python
def make_full_tensor(param: torch.Tensor, spec_info: SpecInfo):
    device = get_device_id()
    if isinstance(spec_info.placement, Replicate):
        return torch.empty_like(param.data, device=device)
    else:
        assert isinstance(spec_info.placement, Shard)
        size = list(param.shape)
        size[spec_info.placement.dim] *= spec_info.ep_mesh.size()
        return torch.empty(size, dtype=param.dtype, device=device)
```

**步骤 3: 辅助函数 - 拷贝到本地** (`initialize.py:147-164`)

```python
def copy_to_local(param: torch.Tensor, full_data: torch.Tensor, spec_info: SpecInfo):
    if isinstance(spec_info.placement, Replicate):
        param.data.copy_(full_data)
    else:
        assert isinstance(spec_info.placement, Shard)
        local_data = full_data.chunk(spec_info.ep_mesh.size(), dim=spec_info.placement.dim)[
            spec_info.ep_mesh.get_local_rank()
        ]
        param.data.copy_(local_data.contiguous())
    param.spec_info = spec_info
```

**步骤 4: 辅助函数 - 分块广播** (`initialize.py:176-210`)

```python
def chunk_and_broadcast_data(param, full_data, spec_info):
    # 用于处理大型专家参数 (>20GB)
    device = param.device
    placement = spec_info.placement
    ep_size = spec_info.ep_mesh.size()

    # 填充到 ep_size 的倍数
    global_size = list(param.data.size())
    global_size[placement.dim] *= ep_size
    global_size = torch.Size(global_size)
    loaded_size = full_data.size()
    pad_size = tuple((0, module_dim - load_dim) for module_dim, load_dim in zip(global_size, loaded_size))
    pad_size = tuple(itertools.chain(*(pad_size[::-1])))
    full_data = torch.nn.functional.pad(full_data, pad_size)

    # 分块并逐块广播
    chunk_loaded_data = list(full_data.chunk(ep_size, dim=placement.dim))
    broadcast_buffer = torch.empty_like(param.data, device=device)
    for chunk_id in range(ep_size):
        broadcast_buffer.copy_(chunk_loaded_data[chunk_id].contiguous())
        dist.broadcast(broadcast_buffer, src=dist.get_rank())

    param.data.copy_(chunk_loaded_data[spec_info.ep_mesh.get_local_rank()].contiguous())
    param.spec_info = spec_info
```

**步骤 5: 辅助函数 - 接收广播** (`initialize.py:202-210`)

```python
def receive_broadcasted_chunk_data(param, broadcast_src, spec_info):
    device = param.device
    chunk_received_data = torch.empty_like(param.data, device=device)
    for chunk_id in range(spec_info.ep_mesh.size()):
        dist.broadcast(chunk_received_data, src=broadcast_src)
        if chunk_id == spec_info.ep_mesh.get_local_rank():
            param.data.copy_(chunk_received_data)
    param.spec_info = spec_info
```

**步骤 6: 创建和同步状态** (`initialize.py:212-274`)

```python
@torch.no_grad()
def create_and_sync_state(param_name, state, is_param):
    device = get_device_id()
    if is_param:
        param = torch.nn.Parameter(torch.empty_like(state.data, device=device), requires_grad=state.requires_grad)
    else:
        param = torch.empty_like(state.data, device=device)

    if param_name not in shard_states:
        # 参数缺失，随机初始化
        if strict:
            raise RuntimeError(f"Missing key(s) in state_dict: {param_name}")
        if dist.get_rank() == 0:
            initializer_range = (2.5 * max(state.shape)) ** -0.5
            size = list(state.size())
            if hasattr(state, "spec_info"):
                shard = state.spec_info.placement
                if isinstance(shard, Shard):
                    size[shard.dim] *= state.spec_info.ep_mesh.size()
            shard_states[param_name] = torch.nn.Parameter(
                torch.randn(size, dtype=state.dtype, device=device, requires_grad=state.requires_grad)
                * initializer_range
            )
        else:
            shard_states[param_name] = 0

    loaded = shard_states[param_name]

    if isinstance(loaded, (torch.nn.Parameter, torch.Tensor)):
        # 本 rank 持有此参数
        if not _is_large_shard_param(param, state):
            full_data = loaded.data.to(dtype=param.dtype, device=param.device, non_blocking=True)
            dist.broadcast(full_data, src=dist.get_rank())
            if hasattr(state, "spec_info"):
                copy_to_local(param, loaded.data, state.spec_info)
            else:
                param.data.copy_(loaded.data)
        else:
            chunk_and_broadcast_data(param, loaded, state.spec_info)
    else:
        # 从其他 rank 接收
        assert isinstance(loaded, int)
        if hasattr(state, "spec_info"):
            if not _is_large_shard_param(param, state):
                full_data = make_full_tensor(param, state.spec_info)
                dist.broadcast(full_data, src=loaded)
                copy_to_local(param, full_data, state.spec_info)
            else:
                receive_broadcasted_chunk_data(param, loaded, state.spec_info)
        else:
            dist.broadcast(param.data, src=loaded)

    shard_states.pop(param_name)
    return param
```

**步骤 7: 返回初始化函数** (`initialize.py:276-317`)

```python
def init_fn(sub_mod: torch.nn.Module):
    if sub_mod in ignore_modules:
        return sub_mod

    param_and_buffers = tuple(sub_mod.named_parameters(recurse=False)) + tuple(
        sub_mod.named_buffers(recurse=False)
    )
    for name, state in param_and_buffers:
        is_param = name in sub_mod._parameters
        fqn = state2fqn[state].pop(0)
        if (not is_param) and fqn not in shard_states:
            if state.is_meta:
                raise RuntimeError(
                    f"find a non-persistent buffer ({fqn}) initiated with device meta. "
                    "Such buffer is not saved in checkpoint and user should guarantee to init in CPU / GPU device."
                )
            continue
        if state in shared:
            if state not in materialized_states:
                materialized_states[state] = create_and_sync_state(fqn, state, is_param)
            else:
                if fqn in shard_states:
                    shard_states.pop(fqn)
            materialize_state = materialized_states[state]
        else:
            materialize_state = create_and_sync_state(fqn, state, is_param)
        if is_param:
            sub_mod._parameters[name] = materialize_state
        else:
            sub_mod._buffers[name] = materialize_state
    return sub_mod

return init_fn
```

### 4.4 检查点扩展 (FSDPExtensions)

**文件**: `veomni/distributed/fsdp/extension.py`

#### 4.4.1 CheckpointExtensions 类

**继承**: `torch.distributed.fsdp._fsdp_extensions.FSDPExtensions`

```python
class CheckpointExtensions(FSDPExtensions):
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

#### 4.4.2 核心方法

**方法 1: chunk_dtensor** (`extension.py:135-155`)

```python
def chunk_dtensor(self, tensor: torch.Tensor, rank: int, device_mesh: DeviceMesh) -> torch.Tensor:
    """将张量/DTensor 分片为 DTensor 并返回本地 DTensor"""
    tensor = tensor.clone().detach()
    fsdp_size = device_mesh.size(-1)
    dimlens = tuple(tensor.size())
    # 默认使用最大维度进行分片
    selected_dim = dimlens.index(max(dimlens))
    for dim, dimlen in enumerate(dimlens):
        if dimlen % fsdp_size == 0:
            selected_dim = dim
            break
    # HSDP placements: [Replicate(), ..., Shard(selected_dim)]
    replicate_placements = [Replicate() for _ in range(device_mesh.ndim)]
    shard_placements = [Replicate() for _ in range(device_mesh.ndim)]
    shard_placements[-1] = Shard(selected_dim)
    dtensor = DTensor.from_local(tensor, device_mesh, replicate_placements, run_check=False).redistribute(
        placements=shard_placements,
    )
    return dtensor
```

**方法 2: state_dict_post_hook** (`extension.py:192-215`)

```python
@torch.no_grad()
def state_dict_post_hook(
    self, module, state_dict, prefix, local_metadata, fqn2spec_info: Dict[str, SpecInfo] = None
):
    """
    保存模型状态字典时调用，为 EP 参数添加 EP 维度
    """
    if self.ep_mesh is None:
        return

    global_device_mesh = self.ep_fsdp_device_mesh
    assert global_device_mesh.ndim == 2

    keys = list(state_dict.keys())
    for name in sorted(keys):
        if name in fqn2spec_info and isinstance(fqn2spec_info[name].placement, Shard):
            cur_spec_info = fqn2spec_info[name]
            tensor = state_dict[name]
            tensor = _shard_tensor(tensor, cur_spec_info.ep_fsdp_mesh, cur_spec_info.placement)
            state_dict[name] = tensor
```

**_shard_tensor 辅助函数** (`extension.py:43-62`):

```python
def _shard_tensor(orgin_tensor: torch.Tensor, device_mesh: DeviceMesh, shard: Shard = Shard(0)):
    """将 Tensor 分片为 DTensor"""
    assert device_mesh.ndim == 2
    ep_mesh = device_mesh["ep"]

    if orgin_tensor.__class__.__name__ == "DTensor":
        # 已经是 DTensor，添加 EP 维度
        placements = (shard,) + orgin_tensor.placements
        dtensor = DTensor.from_local(orgin_tensor._local_tensor, device_mesh=device_mesh, placements=placements)
    elif orgin_tensor.__class__.__name__ == "Tensor":
        # 普通 Tensor，创建 DTensor
        dtensor = DTensor.from_local(orgin_tensor, device_mesh=ep_mesh, placements=[shard])

    return dtensor
```

**方法 3: load_state_dict_pre_hook** (`extension.py:217-252`)

```python
@torch.no_grad()
def load_state_dict_pre_hook(
    self,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
    fqn2spec_info: Dict[str, SpecInfo] = None,
):
    """
    加载模型状态字典前调用，将 EP DTensor 转换为本地 Tensor
    """
    if self.ep_mesh is None:
        return

    global_device_mesh = self.ep_fsdp_device_mesh
    assert global_device_mesh.ndim == 2

    if self.ep_mesh.size() != global_device_mesh.size():
        return

    keys = list(state_dict.keys())
    for name in sorted(keys):
        tensor = state_dict[name]
        if check_any_unflat_param_names_match(name, fqn2spec_info, "_fsdp_wrapped_module"):
            fqn = name.split("_fsdp_wrapped_module.")[-1]
            cur_spec_info = fqn2spec_info[fqn]
            tensor = _shard_dtensor(tensor, cur_spec_info.ep_fsdp_mesh, cur_spec_info.placement)
            state_dict[name] = tensor
```

**_shard_dtensor 辅助函数** (`extension.py:65-81`):

```python
def _shard_dtensor(orgin_dtensor: DTensor, device_mesh: DeviceMesh, shard: Shard = Shard(0)):
    """将 DTensor 转换为本地 Tensor"""
    assert isinstance(orgin_dtensor, DTensor)
    local_tensor = orgin_dtensor.to_local()
    return local_tensor
```

**方法 4: patch_convert_state_with_flat_params** (`extension.py:254-323`)

此方法修补 PyTorch FSDP 的优化器状态转换函数，使其支持 EP 参数：

```python
def patch_convert_state_with_flat_params(self):
    def _convert_state_with_flat_params_patch(
        all_optim_state_keys: List[_OptimStateKey],
        optim_state_key_to_param_key: Dict[_OptimStateKey, Union[int, str]],
        fqn_to_fsdp_param_info: Dict[str, FSDPParamInfo],
        optim_state_dict: Dict[Union[str, int], Any],
        to_save: bool,
        shard_state: bool,
        cpu_offload: bool = True,
        fqn2spec_info: Dict[str, SpecInfo] = None,
    ) -> Dict[str, Any]:
        fsdp_osd_state: Dict[str, Any] = {}
        for optim_state_key in all_optim_state_keys:
            param_key = optim_state_key_to_param_key.get(optim_state_key, None)
            assert param_key is not None

            if optim_state_key.is_fsdp_managed:
                fqn = optim_state_key.unflat_param_names[0]
                fsdp_param_info = fqn_to_fsdp_param_info[fqn]
                # 检查是否所有参数都是 EP 参数
                if check_all_unflat_param_names_match(optim_state_key.unflat_param_names, fqn2spec_info):
                    unflat_state = _unflatten_optim_state(
                        fsdp_param_info,
                        optim_state_dict[param_key],
                        to_save,
                        False,  # 不分片 EP 参数
                        cpu_offload,
                    )
                else:
                    unflat_state = _unflatten_optim_state(
                        fsdp_param_info,
                        optim_state_dict[param_key],
                        to_save,
                        shard_state,
                        cpu_offload,
                    )
                if to_save:
                    for unflat_param_name, unflat_param_state in zip(
                        optim_state_key.unflat_param_names,
                        unflat_state,
                    ):
                        fsdp_osd_state[unflat_param_name] = unflat_param_state
            elif to_save:
                # ... 非 FSDP 管理的参数 ...

        return fsdp_osd_state

    # Monkey patch
    torch.distributed.fsdp._optim_utils._convert_state_with_flat_params = partial(
        _convert_state_with_flat_params_patch, fqn2spec_info=self.fqn2spec_info
    )
```

**方法 5: patch_fsdp_optim_state_dict** (`extension.py:325-358`)

修补 FSDP 的 `optim_state_dict` 方法，为 EP 参数添加 EP 维度：

```python
def patch_fsdp_optim_state_dict(self):
    def fsdp_optim_state_post_patch_fn(
        model, optim, optim_state_dict=None, fqn2spec_info: Dict[str, SpecInfo] = None
    ):
        fsdp_mesh = model._device_mesh
        fsdp_pg = model.process_group
        optim_state = orig_optim_state_dict(model, optim, optim_state_dict, fsdp_pg)
        if self.ep_mesh is None:
            return optim_state

        global_device_mesh = self.ep_fsdp_device_mesh
        assert global_device_mesh.ndim == 2

        # 为 EP 参数添加 EP 维度
        for fqn in sorted(optim_state["state"].keys()):
            if fqn in fqn2spec_info and isinstance(fqn2spec_info[fqn].placement, Shard):
                cur_spec_info = fqn2spec_info[fqn]
                fqn_state = {}
                for key, val in optim_state["state"][fqn].items():
                    # "step" 等标量不分片
                    if key not in OPTIM_STATE_NO_SHARD_KEY:
                        val = _shard_tensor(val, cur_spec_info.ep_fsdp_mesh, cur_spec_info.placement)
                    fqn_state[key] = val
                optim_state["state"][fqn] = fqn_state
        return optim_state

    FSDP.optim_state_dict = staticmethod(partial(fsdp_optim_state_post_patch_fn, fqn2spec_info=self.fqn2spec_info))
```

**方法 6: patch_fsdp_optim_state_dict_to_load** (`extension.py:360-411`)

修补 FSDP 的 `optim_state_dict_to_load` 方法，加载时移除 EP 维度：

```python
def patch_fsdp_optim_state_dict_to_load(self):
    def optim_state_dict_to_load_pre_patch_fn(
        model,
        optim,
        optim_state_dict,
        is_named_optimizer=False,
        load_directly=False,
        group=None,
        fqn2spec_info: Dict[str, SpecInfo] = None,
    ):
        fsdp_mesh = model._device_mesh
        global_device_mesh = self.ep_fsdp_device_mesh
        assert global_device_mesh.ndim == 2

        if self.ep_mesh is not None and self.ep_mesh.size() == self.ep_fsdp_device_mesh.size():
            for fqn in sorted(optim_state_dict["state"].keys()):
                if check_any_unflat_param_names_match(fqn, fqn2spec_info):
                    fqn_state = {}
                    for key, val in optim_state_dict["state"][fqn].items():
                        if key not in OPTIM_STATE_NO_SHARD_KEY:
                            val = _shard_dtensor(val, self.ep_mesh)
                        fqn_state[key] = val
                    optim_state_dict["state"][fqn] = fqn_state

        fsdp_pg = model.process_group
        optim_state = orig_optim_state_dict_to_load(
            model, optim, optim_state_dict, is_named_optimizer, load_directly, fsdp_pg
        )
        return optim_state

    FSDP.optim_state_dict_to_load = staticmethod(
        partial(optim_state_dict_to_load_pre_patch_fn, fqn2spec_info=self.fqn2spec_info)
    )
```

#### 4.4.3 注册扩展

**函数**: `register_checkpoint_extension` (`extension.py:414-451`)

```python
def register_checkpoint_extension(
    fsdp_model: FSDP,
    save_hook_mesh: DeviceMesh = None,
    fqn2spec_info: Dict[str, SpecInfo] = None,
):
    extension = CheckpointExtensions(
        ep_fsdp_device_mesh=save_hook_mesh,
        fqn2spec_info=fqn2spec_info,
    )

    # 为所有 FSDP 模块设置扩展
    for fsdp_module in FSDP.fsdp_modules(fsdp_model):
        fsdp_module._fsdp_extension = extension
        fsdp_module._handle._fsdp_extension = extension
    fsdp_model._fsdp_extension = extension
    fsdp_model._handle._fsdp_extension = extension

    # 注册状态字典钩子
    if fqn2spec_info is not None:
        state_dict_post_hook_fn = partial(extension.state_dict_post_hook, fqn2spec_info=fqn2spec_info)
        fsdp_model._register_state_dict_hook(state_dict_post_hook_fn)

        load_state_dict_pre_hook_fn = partial(extension.load_state_dict_pre_hook, fqn2spec_info=fqn2spec_info)
        fsdp_model._register_load_state_dict_pre_hook(load_state_dict_pre_hook_fn)

        # 修补优化器状态字典函数
        extension.patch_convert_state_with_flat_params()
        extension.patch_fsdp_optim_state_dict()
        extension.patch_fsdp_optim_state_dict_to_load()

    return fsdp_model
```

---

## 5. FSDP2 实现

### 5.1 入口函数

**文件**: `veomni/distributed/torch_parallelize.py:237-435`

```python
def parallelize_model_fsdp2(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    enable_reshard_after_forward: bool = True,
    enable_mixed_precision: bool = True,
    basic_modules: Optional[List[str]] = None,
    **kwargs,
) -> "nn.Module":
    """
    应用 EP (启用时) + FSDP2 并行策略。

    流程:
    1. 应用 EP: 专家张量 [128,H,I] -> [32,H,I] 本地张量
    2. 对专家模块应用 FSDP2: 沿维度 1 分片
    3. 对常规模块应用 FSDP2: 标准维度 0 分片
    4. 结果: 专家参数 [32,H/fsdp_size,I], 常规参数使用标准 FSDP2
    """
```

> **注意**: `enable_reshard_after_forward` 参数控制前向传播后是否释放完整参数，重新分片以节省显存。此参数被直接包含在 `fsdp_kwargs` 中。另外，`build_parallelize_model` 还传入了 `enable_full_shard` 参数，但它通过 `**kwargs` 进入后并未被 `parallelize_model_fsdp2` 使用——这是一个仅对 FSDP1 有效的参数。

### 5.2 流程详解

**步骤 1: 获取目标类并构建层对** (`torch_parallelize.py:254-302`)

```python
parallel_state = get_parallel_state()

# 获取需要分片的基础模块类
target_classes = set((getattr(model, "_no_split_modules", []) or []) + (basic_modules or []))
decoder_blocks: List[Tuple[str, nn.Module]] = [
    (fqn, mod) for fqn, mod in model.named_modules() if mod.__class__.__name__ in target_classes
]

# 应用 Expert Parallelism
if parallel_state.ep_enabled:
    parallel_plan = model.get_parallel_plan()
    ep_fqn2spec_info = parallel_plan.apply(model, parallel_state.ep_fsdp_device_mesh)
    setattr(model, "_fqn2spec_info", ep_fqn2spec_info)
    ep_mesh = parallel_state.ep_fsdp_device_mesh["ep"]
    experts_map = parallel_plan.get_fsdp_no_shard_info(model)
else:
    ep_fqn2spec_info = None
    ep_mesh = None
    experts_map = None

# 提取专家模块并配对
layer_pairs = []
for layer_fqn, layer_mod in decoder_blocks:
    if experts_map is not None:
        experts_mod = next(
            (exp_mod for exp_fqn, exp_mod in experts_map.items() if exp_fqn.startswith(layer_fqn + ".")),
            None,
        )
        layer_pairs.append((layer_fqn, layer_mod, experts_mod))
    else:
        layer_pairs.append((layer_fqn, layer_mod, None))
```

**步骤 2: 配置 FSDP2 参数** (`torch_parallelize.py:304-344`)

```python
fsdp_kwargs = {"mesh": parallel_state.fsdp_mesh, "reshard_after_forward": enable_reshard_after_forward}

# 混合精度策略
if enable_mixed_precision:
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    fsdp_kwargs["mp_policy"] = mp_policy

# 处理需要忽略混合精度的模块
if hasattr(model, "get_ignore_modules_in_mixed_precision"):
    modules_to_ignore_in_mixed_precision = model.get_ignore_modules_in_mixed_precision()
    mp_ignored_classes = modules_to_ignore_in_mixed_precision
    fsdp_kwargs_without_mp = dict(fsdp_kwargs)
    fsdp_kwargs_without_mp.pop("mp_policy", None)
    fsdp_kwargs_without_mp["reshard_after_forward"] = False
else:
    mp_ignored_classes = None
    fsdp_kwargs_without_mp = fsdp_kwargs

# EP-FSDP2 参数
if parallel_state.ep_enabled:
    ep_fsdp_mesh = parallel_state.ep_fsdp_device_mesh["ep_fsdp"]
    expert_fsdp_kwargs = dict(fsdp_kwargs)
    expert_fsdp_kwargs["mesh"] = ep_fsdp_mesh

    # 专家参数使用维度 1 分片
    def _experts_shard_placement_fn(param):
        return Shard(1)

    expert_fsdp_kwargs["shard_placement_fn"] = _experts_shard_placement_fn
```

**为什么专家参数沿维度 1 分片？**

考虑 Qwen3-MoE 的专家权重形状：
```
原始形状: [128 experts, 768 hidden, 768 input]
应用 EP (ep_size=8): [16 experts, 768 hidden, 768 input]  # 沿维度 0 分片
应用 FSDP2 (fsdp_size=2): [16 experts, 384 hidden, 768 input]  # 沿维度 1 分片
```

- **维度 0**: 已被 EP 分片，不能再分
- **维度 1**: 隐藏维度，可以安全分片
- **维度 2**: 输入维度，也可以分片但通常选择隐藏维度

**步骤 3: 自底向上应用 fully_shard** (`torch_parallelize.py:355-391`)

```python
# NPU PreSumMul 补丁 (仅 NPU + EP)
if IS_NPU_AVAILABLE and parallel_state.ep_enabled:
    from veomni.ops.npu_patch.hccl_premul_sum import apply_hccl_premul_sum_patch
    apply_hccl_premul_sum_patch()

for layer_fqn, layer_mod, experts_mod in layer_pairs:
    layer_mod._fsdp_modules = []

    # 1. 分片专家模块 (如果存在)
    if parallel_state.ep_enabled and experts_mod is not None:
        fully_shard(experts_mod, **expert_fsdp_kwargs)
        # 设置梯度除因子
        gradient_divide_factor = parallel_state.ep_gradient_divide_factor
        if IS_NPU_AVAILABLE:
            experts_mod.set_reduce_scatter_divide_factor(gradient_divide_factor)
        else:
            experts_mod.set_gradient_divide_factor(gradient_divide_factor)
        layer_mod._fsdp_modules.append(experts_mod)

    # 2. 分片高精度模块 (如果需要)
    if mp_ignored_classes:
        for sub_mod in layer_mod.modules():
            if isinstance(sub_mod, mp_ignored_classes) and sub_mod is not layer_mod:
                fully_shard(sub_mod, **fsdp_kwargs_without_mp)
                layer_mod._fsdp_modules.append(sub_mod)

    # 3. 分片解码器层的其余部分
    fully_shard(layer_mod, **fsdp_kwargs)
    layer_mod._fsdp_modules.append(layer_mod)

# 4. 分片根模型
fully_shard(model, **fsdp_kwargs)
```

**自底向上的优势**:
- 更细粒度的控制
- 专家模块可以使用不同的分片策略
- 高精度模块可以保持在 GPU 上 (`reshard_after_forward=False`)

**步骤 4: 配置手动预取** (`torch_parallelize.py:393-410`)

```python
need_manual_prefetch = parallel_state.ep_enabled or mp_ignored_classes is not None
if need_manual_prefetch:
    blocks = [pair[1] for pair in layer_pairs]
    next_blocks = blocks[1:] + [None]

    # 前向预取
    for current_block, next_block in zip(blocks, next_blocks):
        if next_block is not None:
            prefetch_modules = next_block._fsdp_modules
            # 按 attn, gate, experts 的顺序预取
            current_block.set_modules_to_forward_prefetch(list(reversed(prefetch_modules)))

    # 反向预取
    rev_blocks = list(reversed(blocks))
    prev_blocks = rev_blocks[1:] + [None]
    for current_block, prev_block in zip(rev_blocks, prev_blocks):
        if prev_block is not None:
            prefetch_modules = prev_block._fsdp_modules
            current_block.set_modules_to_backward_prefetch(list(reversed(prefetch_modules)))
```

**为什么 EP+FSDP2 需要手动预取？**

- FSDP2 的自动预取不知道 EP 嵌套结构
- 专家模块和注意力模块需要协调预取
- 手动预取可以优化通信和计算重叠

**步骤 5: 加载权重** (`torch_parallelize.py:412-428`)

```python
assert kwargs.get("init_device") == "meta", "Please use init_device: meta for FSDP2"

if weights_path is None:
    model.to_empty(device=get_device_type())
    _reset_hf_initialized_flag(model)
    model.init_weights()
else:
    from torch.distributed.tensor import distribute_tensor

    if kwargs.get("broadcast_model_weights_from_rank0"):
        rank0_load_and_broadcast_weights(model, weights_path, get_device_type(), dtensor_factory=distribute_tensor)
    else:
        load_model_weights(model, weights_path, get_device_type(), dtensor_factory=distribute_tensor)
```

**步骤 6: 注册梯度裁剪** (`torch_parallelize.py:430-433`)

```python
from .fsdp2 import clip_grad_norm as clip_grad_norm_fn

model.clip_grad_norm_ = types.MethodType(clip_grad_norm_fn, model)
```

### 5.3 fully_shard 工作原理

**FSDP2 的核心 API**: `torch.distributed._composable.fsdp.fully_shard(module, **kwargs)`

**关键参数**:
- `mesh: DeviceMesh` - 分片网格
- `mp_policy: MixedPrecisionPolicy` - 混合精度策略
- `shard_placement_fn: Callable` - 参数分片维度选择函数
- `reshard_after_forward: bool` - 前向后是否再分片 (默认 True)

**工作流程**:

1. **参数转换为 DTensor**:
```python
# 对于每个参数，fully_shard 会:
param_dtensor = DTensor.from_local(
    param.data,
    device_mesh=mesh,
    placements=[Shard(dim)],  # dim 由 shard_placement_fn 决定
)
```

2. **前向传播时**:
```python
# All-gather 参数
full_param = param_dtensor.full_tensor()
# 前向计算
output = module.forward(input)
# 再分片 (如果 reshard_after_forward=True)
param_dtensor = param_dtensor.redistribute(placements=[Shard(dim)])
```

3. **反向传播时**:
```python
# All-gather 参数
full_param = param_dtensor.full_tensor()
# 反向计算梯度
grad = backward(output, grad_output)
# Reduce-scatter 梯度
grad_dtensor = DTensor.from_local(grad, mesh, placements=[Shard(dim)])
# 参数再分片
param_dtensor = param_dtensor.redistribute(placements=[Shard(dim)])
```

### 5.4 与 FSDP1 的关键差异

| 维度 | FSDP1 | FSDP2 |
|------|-------|-------|
| **参数表示** | FlatParameter (扁平化) | DTensor (保持原始结构) |
| **分片控制** | ShardingStrategy 枚举 | shard_placement_fn 函数 |
| **嵌套方式** | auto_wrap_policy 自动 | 手动 fully_shard 调用 |
| **优化器兼容** | FSDP 特定优化器 | 标准 PyTorch 优化器 |
| **检查点格式** | 需要 FSDPExtensions | DCP 原生支持 |
| **EP 集成复杂度** | 需要扩展钩子 | DTensor 天然支持 |

---

## 6. Expert Parallelism (EP) 集成

### 6.1 SpecInfo 数据类

**文件**: `veomni/distributed/parallel_plan.py:30-42`

```python
@dataclass
class SpecInfo:
    ep_fsdp_mesh: DeviceMesh         # EP+FSDP 的完整网格 [ep, ep_fsdp]
    placement: Union[Shard, Replicate]  # 参数的放置策略
    fqn: str                         # 完全限定名称 (Fully Qualified Name)

    @property
    def ep_mesh(self):
        if self.ep_fsdp_mesh is not None:
            return self.ep_fsdp_mesh["ep"]
        else:
            return None
```

**用途**: 附加在每个参数上，记录其 EP 分片信息

### 6.2 ParallelPlan 系统

**文件**: `veomni/distributed/parallel_plan.py:44-170`

```python
class ParallelPlan:
    def __init__(self, ep_plan: Dict[str, Shard]):
        self.ep_plan = ep_plan  # FQN 模式 -> Shard 映射
        self.ep_param_suffix = {k.split(".")[-1] for k in ep_plan.keys()}
        self.fsdp_no_shard_module = {".".join(list(ep_plan.keys())[0].split(".")[:-1])}
```

#### 6.2.1 apply 方法

**功能**: 将 EP 计划应用到模型，分片专家参数

**代码**: `parallel_plan.py:50-83`

```python
def apply(self, model: nn.Module, ep_fsdp_mesh: DeviceMesh):
    ep_mesh = ep_fsdp_mesh["ep"]
    fqn2spec_info = {}

    if self.ep_plan:
        ep_size = ep_mesh.size(-1)
        ep_replicate = [Replicate() for _ in range(ep_mesh.ndim)]

        for fqn, param in model.named_parameters():
            for fqn_pattern, shard in self.ep_plan.items():
                if check_fqn_match(fqn_pattern, fqn):
                    assert param.size(shard.dim) % ep_size == 0
                    ep_placement = ep_replicate[:-1] + [shard]

                    # 创建 DTensor
                    dtensor = DTensor.from_local(
                        local_tensor=param.data,
                        device_mesh=ep_mesh,
                        placements=ep_replicate
                    )
                    # 重新分布为分片
                    dtensor = dtensor.redistribute(device_mesh=ep_mesh, placements=ep_placement)
                    # 提取本地块
                    local_chunk = torch.nn.Parameter(dtensor.to_local(), requires_grad=param.requires_grad)
                    local_chunk.spec_info = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=shard, fqn=fqn)
                    set_module_from_path(model, fqn, local_chunk)
                    fqn2spec_info[fqn] = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=shard, fqn=fqn)
                    break

            if fqn not in fqn2spec_info:  # 非 EP 参数
                param.spec_info = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=Replicate(), fqn=fqn)
                fqn2spec_info[fqn] = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=Replicate(), fqn=fqn)

    for param in model.parameters():
        assert hasattr(param, "spec_info")

    return fqn2spec_info
```

**步骤详解**:

1. **匹配参数**: 对每个参数，检查是否匹配 `ep_plan` 中的模式
2. **创建 Replicate DTensor**: 首先创建全复制的 DTensor
3. **重新分布为 Shard**: 调用 `redistribute` 进行分片
4. **提取本地块**: `to_local()` 获取当前 rank 的分片
5. **替换参数**: 用本地分片替换原参数
6. **附加元数据**: 设置 `spec_info` 属性

#### 6.2.2 get_fsdp_no_shard_info 方法

**功能**: 获取专家模块列表 (FSDP 不应分片的模块)

**代码**: `parallel_plan.py:85-96`

```python
def get_fsdp_no_shard_info(self, model: nn.Module):
    if self.fsdp_no_shard_module is None:
        return None

    fsdp_no_shard_states_fqn_to_module = {}
    for fqn, param in model.named_modules():
        for no_shard_pattern in self.fsdp_no_shard_module:
            if check_fqn_match(no_shard_pattern, fqn):
                fsdp_no_shard_states_fqn_to_module[fqn] = get_module_from_path(model, fqn)

    assert len(fsdp_no_shard_states_fqn_to_module) > 0
    return fsdp_no_shard_states_fqn_to_module
```

#### 6.2.3 shard_tensor 方法

**功能**: 加载权重时切片专家张量

**代码**: `parallel_plan.py:106-170`

```python
def shard_tensor(self, tensor: "torch.Tensor", full_param_name: str, target_shape: tuple):
    if not self._is_expert_parameter(full_param_name):
        return tensor
    return self._slice_expert_tensor_for_ep(tensor, full_param_name, target_shape)

def _is_expert_parameter(self, parameter_name: str) -> bool:
    if not self.ep_plan:
        return False
    for fqn_pattern in self.ep_plan.keys():
        if check_fqn_match(fqn_pattern, parameter_name):
            return True
    return False

def _slice_expert_tensor_for_ep(self, tensor, parameter_name, target_shape):
    try:
        from .parallel_state import get_parallel_state
        parallel_state = get_parallel_state()

        if len(tensor.shape) >= 1 and len(target_shape) >= 1:
            tensor_experts = tensor.shape[0]
            target_experts = target_shape[0]

            # 如果 tensor 专家数 > 目标专家数，需要切片
            if tensor_experts > target_experts and tensor_experts % target_experts == 0:
                ep_size = tensor_experts // target_experts
                ep_rank = parallel_state.ep_rank if parallel_state.ep_enabled else 0
                start_idx = ep_rank * target_experts
                end_idx = start_idx + target_experts

                sliced_tensor = tensor[start_idx:end_idx]
                return sliced_tensor

        return tensor
    except Exception as e:
        logger.warning(f"Failed to slice expert tensor {parameter_name}: {e}")
        return tensor
```

**使用场景**: 从完整权重文件加载专家参数时

### 6.3 Qwen3-MoE 示例

**文件**: `veomni/models/transformers/qwen3_moe/parallel_plan.py`

```python
def get_paralle_plan():
    ep_plan = {
        "model.layers.*.mlp.experts.gate_proj": Shard(0),
        "model.layers.*.mlp.experts.up_proj": Shard(0),
        "model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    parallel_plan = ParallelPlan(ep_plan=ep_plan)
    return parallel_plan
```

**解释**:
- `*` 是通配符，匹配所有层
- `Shard(0)` 表示沿维度 0 (专家数量维度) 分片
- 三个专家权重矩阵都需要分片

**权重形状示例** (Qwen3-MoE-30B-A3B):

```
原始: [128 experts, 768 hidden, 768 input]
EP (ep_size=8): [16 experts, 768 hidden, 768 input]
EP+FSDP1: [16 experts, 768 hidden, 768 input]  # FSDP 沿维度 0 分片
EP+FSDP2: [16 experts, 384 hidden, 768 input]  # FSDP 沿维度 1 分片
```

---

## 7. 参数分片策略

### 7.1 FSDP1 分片策略

#### 7.1.1 ShardingStrategy 枚举

```python
from torch.distributed.fsdp import ShardingStrategy

ShardingStrategy.FULL_SHARD          # 全分片 (ZeRO-3)
ShardingStrategy.SHARD_GRAD_OP       # 分片梯度和优化器 (ZeRO-2)
ShardingStrategy.NO_SHARD            # 不分片 (DDP)
ShardingStrategy.HYBRID_SHARD        # 混合分片 (HSDP)
```

#### 7.1.2 FULL_SHARD

**特点**:
- 参数、梯度、优化器状态都分片
- 前向/反向时 all-gather 参数
- 内存效率最高，通信开销最大

**适用场景**: 单机多卡，网络带宽充足

#### 7.1.3 HYBRID_SHARD (HSDP)

**特点**:
- 两级拓扑: 复制组 (replica group) + 分片组 (shard group)
- 复制组内全分片，复制组间复制
- 减少跨节点通信

**配置**:
```python
# torch_parallelize.py:122-125
if parallel_state.fsdp_mesh.ndim > 1 and parallel_state.fsdp_mesh.size() > 1:
    strategy = ShardingStrategy.HYBRID_SHARD if enable_full_shard else ShardingStrategy._HYBRID_SHARD_ZERO2
else:
    strategy = ShardingStrategy.FULL_SHARD if enable_full_shard else ShardingStrategy.SHARD_GRAD_OP
```

**DeviceMesh 示例**:
```
世界大小: 16 (4 节点 x 4 卡)
dp_replicate_size: 4 (节点数)
dp_shard_size: 4 (每节点卡数)

FSDP Mesh: [dp_replicate, dp_shard]
┌──────────────────┐
│ Node0 Node1 Node2 Node3 │
├──────────────────┤
│ 0,1,2,3 | 4,5,6,7 | 8,9,10,11 | 12,13,14,15 │
└──────────────────┘
  └─分片组─┘  └─分片组─┘  └──分片组──┘  └───分片组───┘
    └─────────────── 复制组 ────────────────┘
```

**通信模式**:
- **参数 all-gather**: 仅在分片组内 (节点内)
- **梯度 reduce-scatter**: 仅在分片组内
- **跨节点**: 仅优化器更新后的参数同步

### 7.2 FSDP2 分片策略

#### 7.2.1 默认分片维度选择

FSDP2 使用 `shard_placement_fn` 选择分片维度：

```python
def default_shard_placement_fn(param: torch.nn.Parameter) -> Shard:
    # 默认选择最大维度
    max_dim = param.shape.index(max(param.shape))
    return Shard(max_dim)
```

#### 7.2.2 专家参数分片

```python
# torch_parallelize.py:341-342
def _experts_shard_placement_fn(param):
    return Shard(1)  # 强制沿维度 1 分片
```

**原因**: 维度 0 已被 EP 使用

#### 7.2.3 DTensor 分片示例

**标准参数** (如 Attention 权重):
```python
# 原始形状: [4096, 4096]
# FSDP2 (fsdp_size=8):
param = DTensor.from_local(
    local_tensor,  # [4096, 512]
    device_mesh=fsdp_mesh,
    placements=[Shard(1)],
)
```

**EP 专家参数**:
```python
# 原始形状: [128, 4096, 4096]
# EP (ep_size=8): [16, 4096, 4096]
# EP + FSDP2 (fsdp_size=2):
param = DTensor.from_local(
    local_tensor,  # [16, 2048, 4096]
    device_mesh=ep_fsdp_mesh["ep_fsdp"],
    placements=[Shard(1)],
)
```

### 7.3 分片内存计算

**假设**: Qwen2.5-7B, BF16, 8x A100 (80GB)

**FSDP1/FSDP2 (FULL_SHARD)**:
```
参数: 7B * 2 bytes / 8 = 1.75 GB
梯度: 7B * 2 bytes / 8 = 1.75 GB
优化器 (AdamW): 7B * (4 + 4) bytes / 8 = 7 GB
激活: ~10 GB (取决于序列长度)
────────────────────────────
总计: ~20.5 GB / GPU
```

**DDP (NO_SHARD)**:
```
参数: 7B * 2 bytes = 14 GB
梯度: 7B * 2 bytes = 14 GB
优化器: 7B * 8 bytes = 56 GB
激活: ~10 GB
────────────────────────────
总计: ~94 GB / GPU (超过 80GB，无法训练)
```

---

## 8. 梯度同步与裁剪

### 8.1 FSDP1 梯度裁剪

**文件**: `veomni/distributed/fsdp/clip_grad_norm.py`

**函数**: `clip_grad_norm_` (`clip_grad_norm.py:15-132`)

```python
def clip_grad_norm_(fsdp_model: FSDP, max_norm, norm_type=2.0) -> torch.Tensor:
```

#### 8.1.1 EP 感知设计

**核心思想**: 分离 EP 参数和非 EP 参数，分别计算梯度范数并 all-reduce

**步骤 1: 分类参数** (`clip_grad_norm.py:28-56`)

```python
fsdp_managed_params = set()
sharded_params_for_gnorm = {}          # 非 EP 参数
ep_fsdp_sharded_params_for_gnorm = {}  # EP 参数
nonsharded_params_for_gnorm = {}
grads_for_clip = []
ep_fsdp_process_group = None

for handle in fsdp_model._all_handles:
    for param in handle.flat_param._params:
        spec_info: SpecInfo = param.spec_info
        fsdp_managed_params.add(param)
        if param.grad is not None:
            grads_for_clip.append(param.grad)
        # EP 参数
        if isinstance(spec_info.placement, Shard):
            if ep_fsdp_process_group is None:
                ep_fsdp_process_group = handle.process_group
            ep_fsdp_sharded_params_for_gnorm.setdefault(param, None)
        # 非 EP 参数
        else:
            sharded_params_for_gnorm.setdefault(param, None)
```

**步骤 2: 计算本地范数** (`clip_grad_norm.py:58-80`)

```python
grad_norm_kwargs = {
    "norm_type": norm_type,
    "zero": torch.tensor(0.0),
    "device": fsdp_model.compute_device,
}

local_sharded_norm = _get_grad_norm(sharded_params_for_gnorm, **grad_norm_kwargs).to(fsdp_model.compute_device)
local_ep_fsdp_sharded_norm = (
    _get_grad_norm(ep_fsdp_sharded_params_for_gnorm, **grad_norm_kwargs).to(fsdp_model.compute_device)
    if ep_fsdp_sharded_params_for_gnorm
    else None
)
```

**步骤 3: 全局 reduction** (`clip_grad_norm.py:82-105`)

```python
if norm_type == math.inf:
    # Inf-norm: 使用 MAX
    total_norm = local_sharded_norm
    dist.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=fsdp_model.process_group)
    dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=ep_group)
else:
    # p-norm: 使用 SUM
    total_norm = local_sharded_norm**norm_type
    dist.all_reduce(total_norm, group=fsdp_model.process_group)

    if local_ep_fsdp_sharded_norm is not None:
        total_ep_fsdp_sharded_norm = local_ep_fsdp_sharded_norm**norm_type
        # 先在 EP-FSDP 组上 all-reduce
        dist.all_reduce(total_ep_fsdp_sharded_norm, group=ep_fsdp_process_group)
        # 再在 EP 组上 all-reduce
        dist.all_reduce(total_ep_fsdp_sharded_norm, group=ep_group)
        total_norm += total_ep_fsdp_sharded_norm

    total_norm = total_norm ** (1.0 / norm_type)
```

**为什么 EP 参数需要两次 all-reduce？**

考虑配置: `world_size=16, ep_size=4, ep_fsdp_size=4`

```
EP-FSDP 网格:
       ep_fsdp=0  ep_fsdp=1  ep_fsdp=2  ep_fsdp=3
ep=0      0         4          8         12
ep=1      1         5          9         13
ep=2      2         6         10        14
ep=3      3         7         11        15

EP 参数梯度分布:
- Rank 0,4,8,12: 持有不同专家的梯度 (ep=0)
- Rank 1,5,9,13: 持有不同专家的梯度 (ep=1)
...

All-reduce 流程:
1. EP-FSDP 组 all-reduce (纵向):
   - 组 0: {0,1,2,3}
   - 组 1: {4,5,6,7}
   - 组 2: {8,9,10,11}
   - 组 3: {12,13,14,15}
   → 每组内汇总同一专家的梯度范数

2. EP 组 all-reduce (横向):
   - 组 0: {0,4,8,12}
   - 组 1: {1,5,9,13}
   - 组 2: {2,6,10,11}
   - 组 3: {3,7,11,15}
   → 汇总所有专家的梯度范数
```

**步骤 4: 应用裁剪系数** (`clip_grad_norm.py:109-115`)

```python
clip_coef = max_norm / (total_norm + 1e-6)
clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
for grad in grads_for_clip:
    grad.mul_(clip_coef_clamped.to(grad.device, grad.dtype))
```

### 8.2 FSDP2 梯度裁剪

**文件**: `veomni/distributed/fsdp2/clip_grad_norm.py`

**函数**: `ep_fsdp2_clip_grad_norm` (`clip_grad_norm.py:46-99`)

```python
@torch.no_grad()
def ep_fsdp2_clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
```

#### 8.2.1 参数分组

```python
# clip_grad_norm.py:61-73
ps = get_parallel_state()
fsdp_group = ps.fsdp_group
ep_group = ps.ep_group if ps.ep_enabled else None
ep_fsdp_group = None
if ps.ep_enabled and ps.ep_fsdp_device_mesh is not None:
    ep_fsdp_group = ps.ep_fsdp_device_mesh["ep_fsdp"].get_group()

# 构建参数组 (过滤无梯度的参数)
ep_params: List[torch.nn.Parameter] = [p for p in model._ep_param_groups.get("ep", []) if p.grad is not None]
non_ep_params: List[torch.nn.Parameter] = [
    p for p in model._ep_param_groups.get("non_ep", []) if p.grad is not None
]
```

**_ep_param_groups 来源**: 由 `build_ep_fsdp2_optimizer` 设置 (见 [10.3](#103-构建-ep-fsdp2-优化器))

#### 8.2.2 分组 reduction

```python
# clip_grad_norm.py:74-88
non_ep_total = _fsdp2_reduce_group(
    params=non_ep_params,
    norm_type=norm_type,
    reduce_groups=[("fsdp", fsdp_group)],
)

ep_total = _fsdp2_reduce_group(
    params=ep_params,
    norm_type=norm_type,
    reduce_groups=[("ep_fsdp", ep_fsdp_group), ("ep", ep_group)],
)

if math.isinf(norm_type):
    total_norm = torch.maximum(non_ep_total, ep_total)
else:
    total_norm = (non_ep_total + ep_total) ** (1.0 / float(norm_type))

# 统一裁剪系数应用于两组参数
torch.nn.utils.clip_grads_with_norm_(ep_params, max_norm, total_norm, foreach=foreach)
torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_norm, total_norm, foreach=foreach)
```

> **FSDP1 vs FSDP2 裁剪方式差异**: FSDP1 手动计算裁剪系数并逐个乘以梯度 (`clip_coef = max_norm / (total_norm + 1e-6); grad.mul_(clip_coef_clamped)`)，而 FSDP2 使用 PyTorch 2.x 原生的 `torch.nn.utils.clip_grads_with_norm_()` 函数，代码更简洁且与 PyTorch 生态更一致。

#### 8.2.3 辅助函数

**_local_pth_sum** (`clip_grad_norm.py:103-121`):

```python
def _local_pth_sum(params: List[torch.nn.Parameter], p: float) -> torch.Tensor:
    grads = [p.grad for p in params if p.grad is not None]
    grads_local = [
        g.to_local().detach().to(torch.float32) if isinstance(g, DTensor) else g.detach().to(torch.float32)
        for g in grads
    ]
    default_device = grads_local[0].device if len(grads_local) > 0 else torch.device(get_device_type())
    res = torch.tensor(0.0, device=default_device, dtype=torch.float32)
    with torch.no_grad():
        grouped_grads_local = _group_tensors_by_device_and_dtype([grads_local])
        for (device, _), ([device_grads_local], _) in grouped_grads_local.items():
            if _has_foreach_support(device_grads_local, device) or _device_has_foreach_support(device):
                out = torch._foreach_pow_(torch._foreach_norm(device_grads_local, p), p)
                res += torch.sum(torch.stack(out)).to(default_device)
            else:
                for grad_local in device_grads_local:
                    gn = torch.norm(grad_local, p=p)
                    res = res + (gn**p).to(default_device)
    return res
```

**_local_max** (`clip_grad_norm.py:124-143`):

```python
def _local_max(params: List[torch.nn.Parameter]) -> torch.Tensor:
    dev = None
    mx = None
    for q in params:
        g = q.grad
        if g is None:
            continue
        if isinstance(g, DTensor):
            g_local = g.to_local()
        else:
            g_local = g
        if dev is None:
            dev = g_local.device
            mx = torch.tensor(0.0, device=dev, dtype=torch.float32)
        gn = torch.max(torch.abs(g_local.detach().to(torch.float32)))
        mx = torch.maximum(mx, gn)
    if mx is None:
        dev = torch.device(get_device_type())
        mx = torch.tensor(0.0, device=dev, dtype=torch.float32)
    return mx
```

**_fsdp2_reduce_group** (`clip_grad_norm.py:146-170`):

```python
def _fsdp2_reduce_group(
    params: List[torch.nn.Parameter],
    norm_type: float,
    reduce_groups: List[tuple[str, dist.ProcessGroup | None]],
) -> torch.Tensor:
    """
    计算本地组统计量并在提供的组上 reduce。

    对于有限 p，返回全局 reduced 的 p 次幂和 (不是最终范数)。
    对于 inf，返回全局 reduced 的最大值。
    """
    if math.isinf(norm_type):
        val = _local_max(params)
        for _, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.MAX, group=group)
        return val
    else:
        p = float(norm_type)
        val = _local_pth_sum(params, p)
        for name, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.SUM, group=group)
        return val
```

### 8.3 梯度同步流程

#### 8.3.1 FSDP1 梯度同步

**文件**: PyTorch FSDP 内部 (`torch/distributed/fsdp/_runtime_utils.py`)

```python
# 反向传播完成后
def _post_backward_hook(grad_output):
    # 1. 等待所有梯度计算完成
    torch.cuda.synchronize()

    # 2. Reduce-scatter 梯度
    dist._reduce_scatter_base(
        output_tensor=flat_grad_shard,
        input_tensor=flat_grad,
        group=process_group,
    )

    # 3. 除以世界大小 (平均梯度)
    flat_grad_shard.div_(world_size)

    # 4. 释放未分片梯度
    flat_grad = None
```

#### 8.3.2 FSDP2 梯度同步

**文件**: PyTorch FSDP2 内部 (`torch/distributed/_composable/fsdp/_fsdp_api.py`)

```python
# DTensor 自动 reduce-scatter
def _post_backward_hook(grad_output):
    # 1. 梯度已经是 DTensor
    grad_dtensor: DTensor = param.grad

    # 2. DTensor 自动处理 reduce-scatter
    # (在 all-reduce 时自动除以进程组大小)

    # 3. 对于 EP 参数，需要额外处理
    if hasattr(param, "_gradient_divide_factor"):
        grad_dtensor._local_tensor.div_(param._gradient_divide_factor)
```

---

## 9. 混合精度训练

### 9.1 FSDP1 混合精度

**配置**: `torch_parallelize.py:138-148`

```python
mixed_precision = MixedPrecision(
    param_dtype=torch.bfloat16,      # 参数存储精度
    reduce_dtype=torch.float32,      # 梯度通信精度
    buffer_dtype=torch.float32,      # 缓冲区精度
)
```

**工作流程**:

1. **前向传播**:
```python
# 参数以 BF16 存储
param_bf16 = param.to(torch.bfloat16)
# 前向计算使用 BF16
output_bf16 = module.forward(input_bf16)
```

2. **反向传播**:
```python
# 梯度计算使用 BF16
grad_bf16 = backward(output_bf16, grad_output_bf16)
# All-reduce 前转换为 FP32
grad_fp32 = grad_bf16.to(torch.float32)
# All-reduce 使用 FP32
dist.all_reduce(grad_fp32, group=fsdp_group)
# 梯度保持 FP32
param.grad = grad_fp32
```

3. **优化器更新**:
```python
# 优化器状态维护在 FP32
optimizer.step()  # 更新 FP32 master weights
# 参数转换回 BF16
param_bf16 = param.to(torch.bfloat16)
```

**忽略混合精度的模块**:

```python
# torch_parallelize.py:145-146
if hasattr(model, "get_ignore_modules_in_mixed_precision"):
    mixed_precision._module_classes_to_ignore += model.get_ignore_modules_in_mixed_precision()
```

**示例** (Qwen3-MoE):
```python
# qwen3_moe/modeling_qwen3_moe.py
def get_ignore_modules_in_mixed_precision(self):
    # 路由器门控需要高精度
    return (Qwen3MoEGate,)
```

### 9.2 FSDP2 混合精度

**配置**: `torch_parallelize.py:307-312`

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)
```

**MixedPrecisionPolicy vs MixedPrecision**:

| 特性 | MixedPrecision (FSDP1) | MixedPrecisionPolicy (FSDP2) |
|------|------------------------|------------------------------|
| **API 风格** | 配置对象 | 策略对象 |
| **buffer_dtype** | 可配置 | 继承 param_dtype |
| **模块级控制** | `_module_classes_to_ignore` | 不同 `fully_shard` 调用 |
| **灵活性** | 全局设置 | 模块级设置 |

> **注意**: DDP 路径 (`dp_mode="ddp"`) 也支持混合精度 (`torch_parallelize.py:511-521`)。DDP 使用相同的 `MixedPrecision` 配置，但 `buffer_dtype=torch.bfloat16`，与 FSDP1 的 `buffer_dtype=torch.float32` 不同。

**忽略混合精度的模块** (`torch_parallelize.py:319-328`):

```python
if modules_to_ignore_in_mixed_precision:
    fsdp_kwargs_without_mp = dict(fsdp_kwargs)
    fsdp_kwargs_without_mp.pop("mp_policy", None)
    # 高精度模块不重新分片，保持在 GPU
    fsdp_kwargs_without_mp["reshard_after_forward"] = False
```

**应用**:
```python
# torch_parallelize.py:380-384
if mp_ignored_classes:
    for sub_mod in layer_mod.modules():
        if isinstance(sub_mod, mp_ignored_classes) and sub_mod is not layer_mod:
            fully_shard(sub_mod, **fsdp_kwargs_without_mp)
            layer_mod._fsdp_modules.append(sub_mod)
```

### 9.3 数值稳定性

**BF16 的优势**:
- 动态范围与 FP32 相同 (8 位指数)
- 避免 FP16 的上溢/下溢问题
- 不需要 loss scaling

**FP32 梯度通信的重要性**:
```python
# 假设 8 个 rank，每个有梯度 g_i (BF16)
# All-reduce 会累加: g_total = g_1 + g_2 + ... + g_8

# BF16 累加误差:
# 每次加法引入 ~0.1% 误差
# 8 次加法累积误差 ~0.8%

# FP32 累加误差:
# 每次加法引入 ~0.0001% 误差
# 8 次加法累积误差 ~0.0008%

# 结论: FP32 all-reduce 保证梯度精度
```

### 9.4 FSDP2 Reshard 控制

FSDP2 提供两个重要的 reshard 控制参数:

**`enable_reshard_after_forward`** (`TrainingArguments:423-425`):
- 控制前向传播后是否释放 all-gather 的完整参数
- 设为 `True` (默认) 节省显存，但反向传播时需要重新 all-gather
- 设为 `False` 保留参数在 GPU 上，减少通信但增加显存占用

**`enable_reshard_after_backward`** (`TrainingArguments:427-429`):
- 控制反向传播后是否释放参数
- 在梯度累积场景中尤为重要

**梯度累积动态 Reshard 优化** (`train_torch.py:286-294`):

```python
if (
    args.train.data_parallel_mode == "fsdp2"
    and not args.train.enable_reshard_after_backward
    and num_micro_steps > 1
):
    if micro_step == 0:
        model.set_reshard_after_backward(False)
    elif micro_step == num_micro_steps - 1:
        model.set_reshard_after_backward(True)
```

**设计意图**:
- 在梯度累积的中间微步 (`micro_step = 0` 到 `num_micro_steps - 2`)，设置 `reshard_after_backward=False`
- 参数在反向传播后保持 all-gather 状态，避免下一个微步的重复 all-gather
- 在最后一个微步 (`micro_step == num_micro_steps - 1`)，恢复 `reshard_after_backward=True`
- 这样最后一步完成后参数被释放，为优化器更新腾出显存

**性能影响**:
- 减少了 `(num_micro_steps - 1)` 次反向 all-gather 通信
- 代价是中间微步期间占用更多显存 (持有完整参数)
- 适用于显存充裕但通信是瓶颈的场景

---

## 10. 优化器集成

### 10.1 标准优化器 (FSDP1/非EP FSDP2)

**文件**: `veomni/optim/optimizer.py:261-307`

```python
def build_optimizer(
    model: "nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    param_groups: Optional[Sequence[Dict[str, Any]]] = None,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
) -> "torch.optim.Optimizer":
```

**参数分组** (权重衰减):

```python
# optimizer.py:280-296
if param_groups is None:
    decay_param_names = get_parameter_names(model, no_decay_modules, no_decay_params)
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_param_names and p.requires_grad],
            "weight_decay": weight_decay,
        },
    ]
    no_decay_parameters, no_decay_parameter_names = [], []
    for n, p in model.named_parameters():
        if n not in decay_param_names and p.requires_grad:
            no_decay_parameter_names.append(n)
            no_decay_parameters.append(p)

    if len(no_decay_parameters) > 0:
        param_groups.append({"params": no_decay_parameters, "weight_decay": 0.0})
```

**get_parameter_names** (`optimizer.py:242-258`):

```python
def get_parameter_names(model, forbidden_layer_types, forbidden_param_names):
    forbidden_layer_types = [] if forbidden_layer_types is None else forbidden_layer_types
    forbidden_param_names = [] if forbidden_param_names is None else forbidden_param_names
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_param_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if child.__class__.__name__ not in forbidden_layer_types
            and not any(forbidden in f"{name}.{n}".lower() for forbidden in forbidden_param_names)
        ]

    result += [
        k for k in model._parameters.keys() if not any(forbidden in k.lower() for forbidden in forbidden_param_names)
    ]
    return result
```

**常用 no_decay 配置**:
```python
no_decay_modules = ["LayerNorm", "RMSNorm"]
no_decay_params = ["bias", "embedding"]
```

### 10.2 MultiOptimizer

**文件**: `veomni/optim/optimizer.py:144-211`

```python
class MultiOptimizer(Optimizer, Stateful):
    """
    EP+FSDP2 的多优化器容器

    原因: EP 和非 EP 参数有不同的 FSDP 分片维度 (dim-0 vs. dim-1)
    FSDP1 + EP 可以使用单个优化器，因为 FSDP 都沿 dim-0 分片
    """

    def __init__(
        self,
        root_model: nn.Module,
        optimizers: dict,  # {"ep": opt1, "non_ep": opt2}
        key_names: list[str],
    ):
        self.model = root_model
        self.optimizers_dict = optimizers
        self._is_multi_optimizer: bool = True
        self.key_names = key_names
```

**方法实现**:

```python
def step(self) -> None:
    for opt in self.optimizers_dict.values():
        opt.step()

def zero_grad(self) -> None:
    for opt in self.optimizers_dict.values():
        opt.zero_grad()

def state_dict(self) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for name in self.key_names:
        opt = self.optimizers_dict.get(name)
        sd = get_optimizer_state_dict(self.model, opt, options=StateDictOptions(flatten_optimizer_state_dict=True))
        # 检查键冲突
        overlap = set(merged.keys()) & set(sd.keys())
        if overlap:
            raise KeyError(f"Key clash detected: {', '.join(sorted(overlap))}")
        merged.update(sd)
    return merged

def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    # 将合并的状态字典分发到每个子优化器
    for name in self.key_names:
        opt = self.optimizers_dict.get(name)
        set_optimizer_state_dict(
            self.model,
            opt,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
```

### 10.3 构建 EP-FSDP2 优化器

**函数**: `build_ep_fsdp2_optimizer` (`optimizer.py:310-443`)

```python
def build_ep_fsdp2_optimizer(
    model: "nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    param_groups: Optional[List[Dict[str, Any]]] = None,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
):
```

**流程**:

**步骤 1: 分离 EP 和非 EP 参数** (`optimizer.py:390-413`):

```python
ep_params: List[torch.nn.Parameter] = []
non_ep_params: List[torch.nn.Parameter] = []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if DTensor is not None and isinstance(p, DTensor):
        mesh = getattr(p, "device_mesh", None)
        names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
        if "ep_fsdp" in names:
            ep_params.append(p)
            continue
    non_ep_params.append(p)
```

**步骤 2: 构建参数组** (`optimizer.py:410-413`):

```python
ep_groups = _make_param_groups_for_subset(model, ep_params, weight_decay, no_decay_modules, no_decay_params)
non_ep_groups = _make_param_groups_for_subset(
    model, non_ep_params, weight_decay, no_decay_modules, no_decay_params
)
```

**_make_param_groups_for_subset** (`optimizer.py:221-238`):

```python
def _make_param_groups_for_subset(
    model: "nn.Module",
    params: Iterable[torch.nn.Parameter],
    weight_decay: float,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    decay_param_names = set(get_parameter_names(model, no_decay_modules, no_decay_params))
    name_by_param = {p: n for n, p in model.named_parameters()}
    params = [p for p in params if p.requires_grad]
    decayed = [p for p in params if name_by_param.get(p) in decay_param_names]
    undecayed = [p for p in params if name_by_param.get(p) not in decay_param_names]
    groups: List[Dict[str, Any]] = []
    if decayed:
        groups.append({"params": decayed, "weight_decay": weight_decay})
    if undecayed:
        groups.append({"params": undecayed, "weight_decay": 0.0})
    return groups
```

**步骤 3: 创建优化器** (`optimizer.py:415-430`):

```python
def _build(groups: Sequence[Dict[str, Any]]) -> Optimizer:
    if optimizer_type == "adamw":
        foreach = not fused
        _fused = fused
        return AdamW(groups, lr, betas, eps, weight_decay, fused=_fused, foreach=foreach)
    elif optimizer_type == "anyprecision_adamw":
        return AnyPrecisionAdamW(groups, lr, betas, eps, weight_decay)
    else:
        raise ValueError("Only adamw and anyprecision_adamw are supported.")

optimizer_dict: Dict[str, Optimizer] = {}
if ep_groups:
    optimizer_dict["ep"] = _build(ep_groups)
if non_ep_groups:
    optimizer_dict["non_ep"] = _build(non_ep_groups)
```

**步骤 4: 缓存 EP 参数组** (`optimizer.py:432-436`):

```python
model._ep_param_groups = {
    "ep": [p for g in ep_groups for p in g.get("params", [])] if ep_groups else [],
    "non_ep": [p for g in non_ep_groups for p in g.get("params", [])] if non_ep_groups else [],
}
```

**用途**: 梯度裁剪时使用 (见 [8.2](#82-fsdp2-梯度裁剪))

**步骤 5: 返回 MultiOptimizer** (`optimizer.py:438-443`):

```python
key_names = list(optimizer_dict.keys())
multi_opt = MultiOptimizer(model, optimizer_dict, key_names=key_names)
return multi_opt
```

### 10.4 AnyPrecisionAdamW

**文件**: `veomni/optim/optimizer.py:40-142`

```python
class AnyPrecisionAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        use_kahan_summation=True,
        momentum_dtype=torch.bfloat16,
        variance_dtype=torch.bfloat16,
        compensation_buffer_dtype=torch.bfloat16,
    ):
```

**特点**:
- **分离的动量和方差精度**: 可以使用 BF16 存储优化器状态
- **Kahan 求和**: 补偿浮点累积误差
- **更低内存占用**: 相比 FP32 AdamW 节省 50% 优化器内存

**Kahan 求和原理**:

```python
# 标准累加 (存在误差累积)
x = x + delta  # 小的 delta 可能被舍入

# Kahan 求和 (补偿误差)
compensation += delta
temp = x
x = x + compensation
compensation = compensation + (temp - x)  # 保存舍入误差
```

**step 方法** (`optimizer.py:66-141`):

```python
@torch.no_grad()
def step(self, closure=None):
    for group in self.param_groups:
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        eps = group["eps"]
        use_kahan_summation = group["use_kahan_summation"]

        momentum_dtype = group["momentum_dtype"]
        variance_dtype = group["variance_dtype"]
        compensation_buffer_dtype = group["compensation_buffer_dtype"]

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            # 状态初始化
            if len(state) == 0:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=variance_dtype)
                if use_kahan_summation:
                    state["compensation"] = torch.zeros_like(p, dtype=compensation_buffer_dtype)

            # 更新步数
            state["step"] += 1
            step = state["step"]

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            grad = p.grad

            # 权重衰减 (AdamW 风格)
            if weight_decay:
                p.data.mul_(1 - lr * weight_decay)

            # 更新动量
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # 更新非中心方差
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Bias 校正
            bias_correction1 = 1 - beta1**step
            step_size = lr / bias_correction1

            denom_correction = (1 - beta2**step) ** 0.5
            centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(eps, alpha=1)

            if use_kahan_summation:
                compensation = state["compensation"]
                compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                # Kahan 求和更新权重
                temp_buffer = p.detach().clone()
                p.data.add_(compensation)
                compensation.add_(temp_buffer.sub_(p.data))
            else:
                # 标准 AdamW 更新
                p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)
```

---

## 11. 检查点管理

### 11.1 DCP (Distributed Checkpoint)

**PyTorch 2.0+ 提供的分布式检查点格式**:
- 每个 rank 保存自己的分片
- 元数据记录全局张量形状和分片信息
- 支持弹性恢复 (改变世界大小)

**文件**: `veomni/checkpoint/dcp_checkpointer.py`

### 11.2 ModelState 包装器

**类**: `ModelState` (`dcp_checkpointer.py:37-100`)

```python
class ModelState(Stateful):
    """
    包装模型使其成为 Stateful。

    对于 EP+FSDP2，需要在保存前恢复 EP 维度，加载后删除 EP 维度。
    FSDP1 通过 FSDPExtensions 和状态字典钩子自动处理。
    """

    def __init__(self, model):
        self.model = model
        self.parallel_state = get_parallel_state()
        self.ep_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self.should_ep_aware = self.ep_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"
```

**state_dict 方法** (`dcp_checkpointer.py:56-64`):

```python
@torch.no_grad()
def state_dict(self):
    model_state_dict = get_model_state_dict(model=self.model)
    if self.should_ep_aware:
        model_state_dict = self.get_state_dict_with_ep_dim_preprocess(model_state_dict, "restore")
    return model_state_dict
```

**load_state_dict 方法** (`dcp_checkpointer.py:66-77`):

```python
@torch.no_grad()
def load_state_dict(self, state_dict):
    model_state_dict = state_dict
    if self.should_ep_aware:
        model_state_dict = self.get_state_dict_with_ep_dim_preprocess(model_state_dict, "drop")
    set_model_state_dict(model=self.model, model_state_dict=model_state_dict)
```

**get_state_dict_with_ep_dim_preprocess** (`dcp_checkpointer.py:79-100`):

```python
def get_state_dict_with_ep_dim_preprocess(self, state_dict, action):
    ep_fqn2spec_info = self.ep_fqn2spec_info
    ep_mesh = self.parallel_state.ep_fsdp_device_mesh["ep"]
    global_device_mesh = self.parallel_state.ep_fsdp_device_mesh
    assert global_device_mesh.ndim == 2
    assert action in ["restore", "drop"]

    keys = list(state_dict.keys())
    for name in sorted(keys):
        if name in ep_fqn2spec_info and isinstance(ep_fqn2spec_info[name].placement, Shard):
            cur_spec_info = ep_fqn2spec_info[name]
            tensor = state_dict[name]
            if action == "drop":
                tensor = drop_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)
            else:
                tensor = restore_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)
            state_dict[name] = tensor

    return state_dict
```

**restore_ep_dim** (`dcp_checkpointer.py:225-249`):

```python
def restore_ep_dim(orgin_tensor: torch.Tensor, device_mesh: DeviceMesh):
    """
    Restore EP dim so that DCP can be aware about EP ranks

    从 DTensor[ep_fsdp] 恢复为 DTensor[ep, ep_fsdp]
    例如: [16, 384, 768] (本地) -> DTensor[ep, ep_fsdp] 表示 [128, 768, 768] (全局)
    """
    assert device_mesh.ndim == 2, f"global_mesh.ndim must be 2, got {device_mesh.ndim}"
    ep_mesh = device_mesh["ep"]

    if isinstance(orgin_tensor, DTensor):
        # EP+FSDP2: 原始张量已经是 DTensor (由 FSDP2 分片)
        # 使用 _local_tensor 获取底层本地数据，创建包含 EP 和 FSDP 两个维度的 DTensor
        dtensor = DTensor.from_local(
            orgin_tensor._local_tensor, device_mesh=device_mesh, placements=[Shard(0), Shard(1)]
        )
    elif torch.is_tensor(orgin_tensor):
        # 仅 EP，无 FSDP: 普通 Tensor 创建仅含 EP 维度的 DTensor
        dtensor = DTensor.from_local(orgin_tensor, device_mesh=ep_mesh, placements=[Shard(0)])
    else:
        raise RuntimeError(f"origin_tensor - {orgin_tensor} is not a tensor!")

    return dtensor
```

**drop_ep_dim** (`dcp_checkpointer.py:204-222`):

```python
def drop_ep_dim(loaded_tensor: torch.Tensor, device_mesh: DeviceMesh):
    """
    Drop EP dims after loading from DCP so that EP-FSDP would not be confused

    从 DTensor[ep, ep_fsdp] 转换为 DTensor[ep_fsdp] 或普通 Tensor
    """
    assert device_mesh.ndim == 2, f"global_mesh.ndim must be 2, got {device_mesh.ndim}"
    ep_fsdp_mesh = device_mesh["ep_fsdp"]

    if len(loaded_tensor.placements) == 2:
        # EP+FSDP2: 从 DTensor[ep, ep_fsdp] 转为 DTensor[ep_fsdp]
        # 取底层本地数据，在 ep_fsdp 子网格上重建 DTensor
        tensor_to_put = DTensor.from_local(
            loaded_tensor._local_tensor, device_mesh=ep_fsdp_mesh, placements=[Shard(1)]
        )
    elif len(loaded_tensor.placements) == 1:
        # 仅 EP: 直接取本地数据
        tensor_to_put = loaded_tensor.to_local()
    else:
        raise RuntimeError(
            f"Expect EP paramters from checkpoints to be DTensor with 1-dim (no FSDP) or 2-dim (EP+FSDP), got {loaded_tensor}"
        )

    return tensor_to_put
```

**实现要点**:
- `restore_ep_dim`: 使用 `_local_tensor` (而非 `to_local()`) 获取底层数据，因为 DTensor 参数的本地数据已经被 FSDP2 分片，需要保持分片状态
- `restore_ep_dim`: EP+FSDP2 情况下使用 `[Shard(0), Shard(1)]` 放置策略——EP 沿维度 0，FSDP 沿维度 1
- `drop_ep_dim`: 根据 placements 数量区分处理——2 维表示 EP+FSDP，1 维表示仅 EP
- `drop_ep_dim`: 对 EP+FSDP2 使用 `[Shard(1)]` 放置，保留 FSDP 的维度 1 分片信息

### 11.3 OptimizerState 包装器

**类**: `OptimizerState` (`dcp_checkpointer.py:105-201`)

与 `ModelState` 类似，`OptimizerState` 也实现了 EP-FSDP2 感知的状态字典处理。核心区别在于优化器状态字典的键名不直接匹配 `ep_fqn2spec_info`，需要通过子串匹配来找到对应的 EP 参数。

```python
class OptimizerState(Stateful):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.parallel_state = get_parallel_state()
        self.ep_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self.should_ep_aware = self.ep_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"
```

**关键处理逻辑**:
- **保存时**: 通过 `MultiOptimizer.state_dict()` 获取合并的优化器状态，然后使用 `restore_ep_dim` 恢复 EP 维度
- **加载时**: 使用 `drop_ep_dim` 移除 EP 维度，然后通过 `MultiOptimizer.load_state_dict()` 分发到子优化器
- **键名匹配**: 使用子串匹配找到 EP 参数对应的优化器状态 (例如 `state.model.layers.0.mlp.experts.gate_proj.step` 匹配 `model.layers.0.mlp.experts.gate_proj`)
- **标量处理**: 0 维张量和非张量值 (如超参数) 不进行分片处理

### 11.4 异步检查点保存

**类方法**: `DistributedCheckpointer.execute_save` (`dcp_checkpointer.py:350-383`)

```python
@classmethod
def execute_save(cls, save_state, storage_writer, save_async):
    if save_async:
        # 懒创建专用 Gloo 进程组
        if cls._async_process_group is None:
            cls._async_process_group = dist.new_group(backend="gloo")

        # 等待前一次异步保存完成
        if cls.dcp_save_future is not None:
            cls.dcp_save_future.result()
            cls.dcp_save_future = None
            dist.barrier()

        # 启动新的异步保存
        cls.dcp_save_future = dcp.async_save(
            state_dict=save_state,
            storage_writer=storage_writer,
            process_group=cls._async_process_group,
        )
    else:
        dcp.save(state_dict=save_state, storage_writer=storage_writer)
        dist.barrier()
```

**设计要点**:
- **专用 Gloo 进程组** (`_async_process_group`): 异步保存使用独立的 Gloo 通信后端，避免与训练的 NCCL 通信干扰
- **Future 跟踪** (`dcp_save_future`): 通过 `dcp.async_save` 返回的 future 跟踪异步保存进度
- **串行化**: 如果前一次异步保存未完成，会先等待其完成再启动新的保存

### 11.5 检查点保存流程

**伪代码**:

```python
# FSDP1 路径
def save_fsdp1_checkpoint(model, optimizer, path):
    # 1. FSDPExtensions.state_dict_post_hook 自动添加 EP 维度
    model_state = model.state_dict()  # DTensor[ep, ep_fsdp] for EP params

    # 2. FSDPExtensions.patch_fsdp_optim_state_dict 处理优化器
    optim_state = FSDP.optim_state_dict(model, optimizer)  # DTensor[ep, ep_fsdp]

    # 3. 保存到 DCP
    dcp.save(
        state_dict={"model": model_state, "optimizer": optim_state},
        storage_writer=FileSystemWriter(path),
    )

# FSDP2 路径
def save_fsdp2_checkpoint(model, optimizer, path):
    # 1. ModelState.state_dict 手动恢复 EP 维度
    model_state_wrapper = ModelState(model)
    model_state = model_state_wrapper.state_dict()  # DTensor[ep, ep_fsdp]

    # 2. 优化器状态通过 MultiOptimizer.state_dict 合并
    optim_state = optimizer.state_dict()  # 已合并的扁平字典

    # 3. 保存到 DCP
    dcp.save(
        state_dict={"model": model_state, "optimizer": optim_state},
        storage_writer=FileSystemWriter(path),
    )
```

### 11.6 检查点加载流程

```python
# FSDP1 路径
def load_fsdp1_checkpoint(model, optimizer, path):
    # 1. 从 DCP 加载
    state_dict = dcp.load(
        storage_reader=FileSystemReader(path),
    )

    # 2. FSDPExtensions.load_state_dict_pre_hook 移除 EP 维度
    model.load_state_dict(state_dict["model"])  # DTensor[ep_fsdp]

    # 3. FSDPExtensions.patch_fsdp_optim_state_dict_to_load 处理优化器
    FSDP.optim_state_dict_to_load(model, optimizer, state_dict["optimizer"])

# FSDP2 路径
def load_fsdp2_checkpoint(model, optimizer, path):
    # 1. 从 DCP 加载
    state_dict = dcp.load(
        storage_reader=FileSystemReader(path),
    )

    # 2. ModelState.load_state_dict 移除 EP 维度
    model_state_wrapper = ModelState(model)
    model_state_wrapper.load_state_dict(state_dict["model"])  # DTensor[ep_fsdp]

    # 3. MultiOptimizer.load_state_dict 分发到子优化器
    optimizer.load_state_dict(state_dict["optimizer"])
```

### 11.7 HuggingFace 格式导出

**TrainerCheckpointer.export_huggingface_checkpoint()**:

1. **加载 DCP 检查点**
2. **合并所有分片** (all-gather)
3. **转换为 HuggingFace 格式** (safetensors)
4. **保存到单一目录**

---

## 12. 设备网格 (DeviceMesh) 拓扑

### 12.1 示例配置

**场景**: 4 节点 x 8 GPU = 32 GPUs

```python
init_parallel_state(
    dp_replicate_size=4,    # 4 个节点
    dp_shard_size=8,        # 每节点 8 GPU
    ep_size=4,              # 4-way EP
    ulysses_size=1,         # 无 Ulysses
    cp_size=1,              # 无 CP
    tp_size=1,              # 无 TP
    pp_size=1,              # 无 PP
    dp_mode="fsdp2",
    ep_outside=False,
)
```

### 12.2 主 DeviceMesh

```
mesh_shape: (4, 8)
mesh_dim_names: ("dp_replicate", "dp_shard")

Rank 布局:
       dp_shard=0  1   2   3   4   5   6   7
dp_rep=0   0      1   2   3   4   5   6   7
dp_rep=1   8      9  10  11  12  13  14  15
dp_rep=2  16     17  18  19  20  21  22  23
dp_rep=3  24     25  26  27  28  29  30  31

复合网格:
- dp: [dp_replicate, dp_shard] = 所有 32 ranks
- dp_shard_sp: [dp_shard] = {0-7}, {8-15}, {16-23}, {24-31}
- dp_sp: [dp_replicate, dp_shard] = 所有 32 ranks (与 dp 相同)
```

### 12.3 EP-FSDP DeviceMesh

```
ep_size: 4
ep_fsdp_size: 8 (32 / 4)
ep_outside: False

mesh_shape: (4, 8)
mesh_dim_names: ("ep", "ep_fsdp")

Rank 布局:
       ep_fsdp=0  1   2   3   4   5   6   7
ep=0     0      4   8  12  16  20  24  28
ep=1     1      5   9  13  17  21  25  29
ep=2     2      6  10  14  18  22  26  30
ep=3     3      7  11  15  19  23  27  31

解释:
- EP 组 0: {0, 4, 8, 12, 16, 20, 24, 28}
- EP 组 1: {1, 5, 9, 13, 17, 21, 25, 29}
- EP 组 2: {2, 6, 10, 14, 18, 22, 26, 30}
- EP 组 3: {3, 7, 11, 15, 19, 23, 27, 31}

EP-FSDP 组 0: {0, 1, 2, 3}
EP-FSDP 组 1: {4, 5, 6, 7}
...
```

### 12.4 通信模式

**FSDP All-Gather** (非 EP 参数):
```
组: {0, 1, 2, 3, 4, 5, 6, 7} (同一 dp_shard 组)
通信量: (模型大小 / 8) * 7 (每个 rank 收集 7 个分片)
```

**EP All-Gather** (EP 参数):
```
组: {0, 4, 8, 12, 16, 20, 24, 28} (同一 EP 组)
通信量: (专家参数大小 / 8) * 7
```

**梯度 All-Reduce**:
- **非 EP 参数**: 在 FSDP 组上 reduce-scatter
- **EP 参数**: 先在 EP-FSDP 组上 reduce-scatter，再在 EP 组上 all-reduce

---

## 13. 性能优化

### 13.1 预取 (Prefetching)

**FSDP1 自动预取** (`torch_parallelize.py:175-179`):

```python
if kwargs.pop("enable_forward_prefetch", False):
    fsdp_kwargs["forward_prefetch"] = True
else:
    fsdp_kwargs["forward_prefetch"] = False
    fsdp_kwargs["backward_prefetch"] = None
```

**FSDP2 手动预取** (`torch_parallelize.py:393-410`):

```python
need_manual_prefetch = parallel_state.ep_enabled or mp_ignored_classes is not None
if need_manual_prefetch:
    blocks = [pair[1] for pair in layer_pairs]
    next_blocks = blocks[1:] + [None]

    # 前向预取下一层
    for current_block, next_block in zip(blocks, next_blocks):
        if next_block is not None:
            prefetch_modules = next_block._fsdp_modules
            current_block.set_modules_to_forward_prefetch(list(reversed(prefetch_modules)))

    # 反向预取前一层
    rev_blocks = list(reversed(blocks))
    prev_blocks = rev_blocks[1:] + [None]
    for current_block, prev_block in zip(rev_blocks, prev_blocks):
        if prev_block is not None:
            prefetch_modules = prev_block._fsdp_modules
            current_block.set_modules_to_backward_prefetch(list(reversed(prefetch_modules)))
```

**预取顺序** (reversed):
```
_fsdp_modules = [layer, gate, experts]
prefetch = [experts, gate, layer]  # 逆序
```

**原因**: 后面的模块先计算，提前预取可以隐藏通信延迟

### 13.2 CPU Offload

**FSDP1 支持**:

```python
# torch_parallelize.py:171-173
if kwargs.pop("enable_fsdp_offload", False):
    fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)
```

**效果**:
- 参数在 CPU 上存储
- 前向/反向时加载到 GPU
- 内存换时间

**FSDP2**: 暂不支持 CPU offload

### 13.3 梯度累积

**标准实现**:

```python
# train_torch.py (伪代码)
optimizer.zero_grad()
for micro_step in range(gradient_accumulation_steps):
    micro_batch = next(dataloader)
    loss = model(micro_batch)
    loss = loss / gradient_accumulation_steps
    loss.backward()

optimizer.step()
```

**FSDP 兼容性**: 自动处理，无需特殊配置

### 13.4 Flash Attention 集成

**配置**: `model_args.attn_implementation = "flash_attention_2"`

**FSDP 集成**:
- Flash Attention 使用 `cu_seqlens` 格式
- 与 Packing Collator 天然兼容 (见 [Dynamic Batching 分析](dynamic_batching_strategy_analysis.md))
- 无需额外适配

### 13.5 激活检查点 (Activation Checkpointing)

**启用**:

```python
# torch_parallelize.py:466-477
if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
    use_reentrant = kwargs.pop("enable_reentrant", False)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": use_reentrant,
            "context_fn": kwargs.pop("recompute_context_fn", noop_context_fn),
        },
    )
```

**FSDP 兼容性**:
- 推荐 `use_reentrant=False` (PyTorch 2.0+)
- FSDP 会自动处理检查点与参数分片的交互

### 13.6 损失缩放与 FSDP 梯度补偿

**文件**: `veomni/utils/loss_utils.py:30-56`

VeOmni 使用 `mean_global_loss()` 函数计算全局平均损失，而非简单的 `loss / dp_size`。核心公式:

```python
cur_loss = cur_loss * cur_token_len / all_reduced_len * get_parallel_state().fsdp_size
```

**为什么需要乘以 `fsdp_size`？**

FSDP 在 reduce-scatter 时默认将梯度除以 FSDP 进程组大小。但 VeOmni 采用 token-weighted 损失缩放——每个 rank 的损失按其实际 token 数占全局 token 数的比例缩放，而不是简单的等分。因此需要乘以 `fsdp_size` 来抵消 FSDP 的默认除法，让最终梯度反映 token-weighted 的缩放。

**计算流程**:

1. 每个 rank 计算本地 token 数 `cur_token_len`
2. 如果启用了序列并行，在 SP 组上 all-reduce token 数
3. 全局 all-reduce 得到所有微批次的总 token 数 `all_reduced_len`
4. 损失 = `cur_loss * (cur_token_len / all_reduced_len) * fsdp_size`

**直观理解**: 假设 4 个 rank，rank 0 有 100 个 token，总共 400 个 token。
- VeOmni 希望 rank 0 的梯度权重为 `100/400 = 0.25`
- 但 FSDP 会在 reduce-scatter 时除以 4
- 所以预先乘以 4: `loss * (100/400) * 4 = loss`
- 经过 FSDP 除以 4 后: `loss / 4 = loss * 0.25`
- 结果正好是期望的 token-weighted 梯度

---

## 14. 限制与注意事项

### 14.1 FSDP1 限制

1. **EP 检查点复杂性**:
   - 需要 FSDPExtensions 钩子
   - Monkey patch PyTorch 内部函数
   - 升级 PyTorch 可能导致兼容性问题

2. **扁平化参数**:
   - FlatParameter 隐藏原始参数结构
   - 调试困难

3. **预取控制**:
   - EP+FSDP1 无法手动预取
   - 通信效率可能不如 FSDP2

### 14.2 FSDP2 限制

1. **PyTorch 版本要求**:
   - 需要 >= 2.4
   - NPU 使用 2.7.1 (特殊版本)

2. **手动预取必需**:
   - EP+FSDP2 必须手动配置预取
   - 代码更复杂

3. **CPU Offload 不支持**:
   - 仅 FSDP1 支持
   - 大模型训练受限

### 14.3 EP 限制

1. **仅支持 MoE 模型**:
   - 需要模型定义 `get_parallel_plan()`
   - 非 MoE 模型无法使用 EP

2. **维度限制**:
   - EP 参数必须沿维度 0 分片
   - FSDP2 必须沿其他维度分片 (通常维度 1)

3. **TP/PP 不兼容**:
   - `parallel_state.py:352-353` 断言 TP=1, PP=1
   - 暂不支持多种模型并行组合

### 14.4 通用限制

1. **Ulysses SP 包含在 FSDP**:
   - `include_sp_in_fsdp=False` 未实现
   - 无法解耦 SP 和 FSDP

2. **Ring Attention 未实现**:
   - `cp_size > 1` 会报错
   - 长序列训练受限

3. **Meta 初始化强制** (FSDP2):
   - 必须使用 `init_device="meta"`
   - 无法像 FSDP1 一样灵活选择

---

## 15. 参考资料

### 15.1 VeOmni 文档

- **官方文档**: https://veomni.readthedocs.io/
- **论文**: https://arxiv.org/abs/2508.02317 (AAAI 2026)
- **GitHub**: https://github.com/ByteDance-Seed/VeOmni

### 15.2 PyTorch 文档

- **FSDP 教程**: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- **FSDP2 (Fully Shard)**: https://pytorch.org/docs/stable/fsdp.html
- **DTensor**: https://pytorch.org/docs/stable/distributed.tensor.html
- **DeviceMesh**: https://pytorch.org/docs/stable/distributed.tensor.parallel.html
- **DCP (Distributed Checkpoint)**: https://pytorch.org/docs/stable/distributed.checkpoint.html

### 15.3 相关论文

- **ZeRO**: Rajbhandari et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (SC 2020)
- **FSDP**: Zhao et al. "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" (VLDB 2023)
- **MoE**: Fedus et al. "Switch Transformers: Scaling to Trillion Parameter Models" (JMLR 2022)
- **Expert Parallelism**: Lepikhin et al. "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" (ICLR 2021)

### 15.4 源码文件索引

**核心文件** (按重要性排序):

1. `veomni/distributed/parallel_state.py` (579 行) - 并行状态管理
2. `veomni/distributed/torch_parallelize.py` (523 行) - FSDP1/FSDP2 入口
3. `veomni/distributed/parallel_plan.py` (170 行) - EP 计划系统
4. `veomni/distributed/fsdp/initialize.py` (349 行) - FSDP1 参数初始化
5. `veomni/distributed/fsdp/extension.py` (451 行) - FSDP1 检查点扩展
6. `veomni/distributed/fsdp/clip_grad_norm.py` (137 行) - FSDP1 梯度裁剪
7. `veomni/distributed/fsdp2/clip_grad_norm.py` (170 行) - FSDP2 梯度裁剪
8. `veomni/optim/optimizer.py` (443 行) - 优化器和 MultiOptimizer
9. `veomni/checkpoint/dcp_checkpointer.py` - DCP 检查点管理

**辅助文件**:
- `veomni/distributed/utils.py` - FQN 匹配工具
- `veomni/utils/arguments.py` - 训练参数配置
- `veomni/models/transformers/*/parallel_plan.py` - 模型特定的 EP 计划

---

## 附录 A: 完整配置示例

### A.1 Qwen2.5-7B (FSDP2, 无 EP)

```yaml
# configs/pretrain/qwen2_5_7b_fsdp2.yaml
model_args:
  config_path: Qwen/Qwen2.5-7B
  model_path: Qwen/Qwen2.5-7B
  attn_implementation: flash_attention_2

training_args:
  # 并行配置
  dp_mode: fsdp2
  dp_size: 8
  dp_shard_size: 8
  dp_replicate_size: 1

  # 训练配置
  enable_mixed_precision: true
  enable_gradient_checkpointing: true
  init_device: meta
  broadcast_model_weights_from_rank0: true

  # 优化器
  optimizer_type: adamw
  lr: 3e-4
  weight_decay: 0.1
  no_decay_modules: ["LayerNorm", "RMSNorm"]

  # 其他
  max_steps: 10000
  checkpointing_steps: 1000
```

### A.2 Qwen3-MoE-30B-A3B (FSDP2 + EP)

```yaml
# configs/pretrain/qwen3_moe_30b_fsdp2_ep.yaml
model_args:
  config_path: Qwen/Qwen3-MoE-30B-A3B
  model_path: Qwen/Qwen3-MoE-30B-A3B
  attn_implementation: flash_attention_2
  moe_implementation: fused

training_args:
  # 并行配置
  dp_mode: fsdp2
  dp_size: 16
  dp_shard_size: 2
  dp_replicate_size: 8
  ep_size: 8
  ep_outside: false

  # 训练配置
  enable_mixed_precision: true
  enable_gradient_checkpointing: true
  init_device: meta
  broadcast_model_weights_from_rank0: true

  # 优化器
  optimizer_type: adamw
  lr: 1e-4
  weight_decay: 0.1
  no_decay_modules: ["LayerNorm", "RMSNorm", "Qwen3MoEGate"]

  # 其他
  max_steps: 50000
  checkpointing_steps: 5000
```

---

## 附录 B: 调试技巧

### B.1 打印设备网格

```python
from veomni.distributed.parallel_state import get_parallel_state

ps = get_parallel_state()
if ps.global_rank == 0:
    print(f"Device Mesh: {ps.device_mesh}")
    print(f"EP-FSDP Mesh: {ps.ep_fsdp_device_mesh}")
    print(f"FSDP Mesh: {ps.fsdp_mesh}")
```

### B.2 FSDP1 分组结构调试

**函数**: `verbose_fsdp_grouping` (`torch_parallelize.py:62-77`)

FSDP1 包装完成后 (第 230 行)，会自动调用此函数打印 FSDP 分组结构:

```python
def verbose_fsdp_grouping(model, prefix="", depth=0):
    indent = "    " * depth
    for name, child in model.named_children():
        if isinstance(child, FullyShardedDataParallel):
            module_names = [m_name for m_name, _ in child.named_modules()][1:]
            strategy = child.sharding_strategy
            logger.debug_rank0(f"{indent}├── [FSDP Group] {prefix}{name}")
            logger.debug_rank0(f"{indent}│   ├── Sharding Strategy: {strategy}, Mixed Precision: {child.mixed_precision}")
            logger.debug_rank0(f"{indent}│   └── Contains Modules: {module_names}")
            verbose_fsdp_grouping(child, prefix=f"{prefix}{name}.", depth=depth + 1)
        else:
            verbose_fsdp_grouping(child, prefix=f"{prefix}{name}.", depth=depth)
```

**用法**: 设置日志级别为 DEBUG 即可在训练启动时看到 FSDP 的完整分组结构，便于验证 `auto_wrap_policy` 是否正确分组。

### B.3 检查 EP 参数分片

```python
for name, param in model.named_parameters():
    if hasattr(param, "spec_info"):
        spec_info = param.spec_info
        print(f"{name}: placement={spec_info.placement}, shape={param.shape}")
```

### B.4 验证梯度范数

```python
import torch.distributed as dist

total_norm = model.clip_grad_norm_(max_norm=1.0)
if ps.global_rank == 0:
    print(f"Global grad norm: {total_norm}")

# 对比非 EP 参数和 EP 参数
if hasattr(model, "_ep_param_groups"):
    ep_params = model._ep_param_groups["ep"]
    non_ep_params = model._ep_param_groups["non_ep"]

    ep_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in ep_params if p.grad is not None]), 2.0)
    non_ep_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in non_ep_params if p.grad is not None]), 2.0)

    print(f"EP grad norm: {ep_norm}, Non-EP grad norm: {non_ep_norm}")
```

### B.5 监控内存使用

```python
import torch

def print_memory_stats():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Max: {max_allocated:.2f} GB")

# 在训练循环中
print_memory_stats()  # 前向前
loss.backward()
print_memory_stats()  # 反向后
optimizer.step()
print_memory_stats()  # 优化器更新后
```

---

**文档结束**
