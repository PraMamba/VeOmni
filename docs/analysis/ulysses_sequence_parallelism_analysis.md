# VeOmni 框架中 Ulysses Sequence Parallelism 实现详解

## 1. 概述

VeOmni 框架基于 DeepSpeed Ulysses 算法实现了 Sequence Parallelism（序列并行），支持同步（非异步）和异步两种模式。该实现允许模型在序列维度上进行分布式训练，从而能够处理超长序列，并有效降低单卡显存占用。

### 1.1 核心特性

- **同步模式（Non-Async）**：基础 Ulysses 实现，使用 all-to-all 通信原语
- **异步模式（Async）**：通信与计算重叠优化，显著提升训练效率
- **FSDP/FSDP2 集成**：与 PyTorch 原生分布式训练无缝集成
- **灵活的硬件支持**：同时支持 NVIDIA GPU 和 Ascend NPU
- **Attention 无关**：可与任意 attention 实现（FlashAttention、SDPA 等）配合使用

### 1.2 相关文件结构

```
veomni/distributed/sequence_parallel/
├── ulysses.py              # 核心 Ulysses 算法实现（同步模式）
├── async_ulysses.py        # 异步 Ulysses 优化实现
├── async_ulysses_dit.py    # DiT 模型专用异步实现
├── comm.py                 # 进程组管理
├── data.py                 # 数据预处理与后处理
├── loss.py                 # 损失函数聚合
└── utils.py                # 辅助工具函数

tests/parallel/ulysses/
├── test_ulysses.py         # 同步模式测试
├── test_async_ulysses.py   # 异步模式测试
└── attention.py            # 参考 Attention 实现
```

## 2. 核心算法原理

### 2.1 Ulysses Sequence Parallelism 基本思想

Ulysses 算法通过在序列维度和注意力头维度之间进行动态切换来实现序列并行：

**输入阶段**：每个 rank 持有序列的一部分
```
Rank 0: [batch, seq_len/P, hidden_dim]
Rank 1: [batch, seq_len/P, hidden_dim]
...
Rank P-1: [batch, seq_len/P, hidden_dim]
```

**Attention 计算前**：通过 all-to-all 通信，将序列维度聚合，注意力头维度分散
```
all-to-all (gather_seq_scatter_heads)
↓
Rank 0: [batch, seq_len, hidden_dim/P]
Rank 1: [batch, seq_len, hidden_dim/P]
...
Rank P-1: [batch, seq_len, hidden_dim/P]
```

**Attention 计算后**：通过 all-to-all 通信，将注意力头维度聚合，序列维度分散
```
all-to-all (gather_heads_scatter_seq)
↓
Rank 0: [batch, seq_len/P, hidden_dim]
Rank 1: [batch, seq_len/P, hidden_dim]
...
Rank P-1: [batch, seq_len/P, hidden_dim]
```

### 2.2 通信复杂度分析

对于序列长度 `S`、隐藏维度 `H`、并行度 `P`：

- **每次 all-to-all 通信量**：`4SH(P-1)/P²` bytes
- **通信效率**：
  - P=4: ~75% 效率
  - P=8: ~87.5% 效率
- **激活显存节省**：减少到原来的 `1/P`

## 3. 同步模式（Non-Async）实现详解

### 3.1 核心通信原语

#### 文件：`veomni/distributed/sequence_parallel/ulysses.py`

**all_to_all_tensor** (第 125-135 行)
```python
def all_to_all_tensor(
    x: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
    async_op: bool = False,
):
    if scatter_dim <= 1 and gather_dim <= 1:
        return _all_to_all_single(x, scatter_dim, gather_dim, group, async_op)
    else:
        return _all_to_all(x, scatter_dim, gather_dim, group, async_op)
```

该函数根据维度选择优化的通信路径：
- **维度 ≤ 1**：使用 `_all_to_all_single` 优化实现
- **维度 > 1**：使用通用 `_all_to_all` 实现

**_all_to_all_single** (第 86-122 行)
```python
def _all_to_all_single(
    x: Tensor, scatter_dim: int, gather_dim: int,
    group: Optional[dist.ProcessGroup] = None, async_op: bool = False
):
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group)

    # 对于 scatter_dim=1 的情况，需要重排张量
    if scatter_dim != 0:
        gather_dim_bef = x.shape[gather_dim]
        scatter_dim_bef = x.shape[scatter_dim]
        x = (
            x.reshape([gather_dim_bef, sp_world_size, scatter_dim_bef // sp_world_size]
                      + list(x.shape[2:]))
            .transpose(0, 1)
            .reshape([gather_dim_bef * sp_world_size, scatter_dim_bef // sp_world_size]
                     + list(x.shape[2:]))
            .contiguous()
        )

    output = torch.empty_like(x)
    comm = dist.all_to_all_single(output, x.contiguous(), group=group, async_op=async_op)

    if async_op:
        def wait():
            comm.wait()
            if scatter_dim == 0:
                return torch.cat(output.split(x.size(0) // sp_world_size), dim=gather_dim)
            else:
                return output
        return wait

    if scatter_dim == 0:
        output = torch.cat(output.split(x.size(0) // sp_world_size), dim=gather_dim)
    return output
```

**设计要点**：
1. 使用 PyTorch 的 `all_to_all_single` 而非 `all_to_all`，减少内存拷贝
2. 支持 `async_op` 参数，返回 wait 函数用于异步通信
3. 针对不同 scatter_dim 进行张量重排优化

### 3.2 Autograd 集成

**_SeqAllToAll** (第 138-160 行)
```python
class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        input_t = grad_output[0]
        # 反向传播时交换 scatter_dim 和 gather_dim
        return (
            None,
            all_to_all_tensor(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, False),
            None,
            None,
        )
```

**设计要点**：
- 反向传播时自动反转 scatter 和 gather 维度
- 梯度流正确传递，无需手动管理

### 3.3 核心 API

**gather_seq_scatter_heads** (第 232-250 行)
```python
def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    group: ProcessGroup = None,
) -> Tensor:
    """
    Attention 计算前调用：聚合序列，分散注意力头
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    # 执行 all-to-all 通信
    x = _SeqAllToAll.apply(group, x, head_dim, seq_dim)
    # 处理 padding（如果序列长度不能被 sp_world 整除）
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = x.size(seq_dim) - unpadded_dim_size
        x = unpad_tensor(x, seq_dim, padding_size)
    return x
```

**gather_heads_scatter_seq** (第 217-229 行)
```python
def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int,
                              group: ProcessGroup = None) -> Tensor:
    """
    Attention 计算后调用：聚合注意力头，分散序列
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    dim_size = x.size(seq_dim)
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    # 如果序列长度不能整除，需要先 padding
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = pad_tensor(x, seq_dim, padding_size)
    return _SeqAllToAll.apply(group, x, seq_dim, head_dim)
```

### 3.4 使用示例

以下是典型的 Attention 层集成方式（来自 `tests/parallel/ulysses/attention.py`）：

```python
class Attention(nn.Module):
    def forward(self, x: torch.Tensor, unpadded_seq_len: int):
        # QKV 投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch, seq, num_heads, head_dim]
        q = q.view(B, seq_len, self.num_heads, self.head_dim)
        k = k.view(B, seq_len, self.num_heads, self.head_dim)
        v = v.view(B, seq_len, self.num_heads, self.head_dim)

        # Attention 前：聚合序列，分散头
        q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2,
                                      unpadded_dim_size=unpadded_seq_len)
        k = gather_seq_scatter_heads(k, seq_dim=1, head_dim=2,
                                      unpadded_dim_size=unpadded_seq_len)
        v = gather_seq_scatter_heads(v, seq_dim=1, head_dim=2,
                                      unpadded_dim_size=unpadded_seq_len)

        # QK Normalization (可选)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Attention 计算（FlashAttention、SDPA 等）
        x = F.scaled_dot_product_attention(q, k, v, ...)

        # Attention 后：聚合头，分散序列
        x = gather_heads_scatter_seq(x, head_dim=2, seq_dim=1)

        # Output 投影
        x = x.view(B, seq_len, -1)
        x = self.proj_o(x)

        return x
```

## 4. 异步模式（Async）实现详解

### 4.1 异步优化动机

同步模式的性能瓶颈在于通信与计算串行执行：

```
[QKV 投影] → [Q all-to-all] → [K all-to-all] → [V all-to-all] → [Attention] → [Output all-to-all]
```

异步模式通过以下策略实现通信-计算重叠：

```
[Q 投影 + Q all-to-all (async)] → [K 投影 + K all-to-all (async)] → [V 投影 + V all-to-all (async)]
                                ↓
                         [Q wait + QK Norm]
                                ↓
                         [K wait + V wait + Attention]
                                ↓
                         [Output 投影 + Output all-to-all (async)]
```

### 4.2 核心实现

#### 文件：`veomni/distributed/sequence_parallel/async_ulysses.py`

**AsyncUlyssesQKVProjection** (第 48-209 行)

该类融合了 QKV 线性投影和 all-to-all 通信，实现通信-计算重叠。

**前向传播关键步骤**：

```python
def forward(ctx, hidden_states, seq_dimension, head_dimension,
            q_weight, k_weight, v_weight, ...):
    # 1. Q 投影
    q = F.linear(hidden_states, q_weight, q_bias)
    q = q.view(batch_size, -1, num_q_heads, head_dim)

    # 2. 启动 Q 的异步 all-to-all 通信
    q_res = all_to_all_tensor(
        q, scatter_dim=head_dimension, gather_dim=seq_dimension,
        group=sp_group, async_op=True
    )

    # 3. K 投影（在 Q 通信进行时）
    k = F.linear(hidden_states, k_weight, k_bias)
    k = k.view(batch_size, -1, num_kv_heads, head_dim)

    # 4. 处理 GQA：如果 ulysses_size > num_kv_heads，需要重复 KV heads
    if need_repeat_kv:
        k = torch.repeat_interleave(k, dim=2, repeats=n_repeat)

    # 5. 启动 K 的异步 all-to-all 通信
    k_res = all_to_all_tensor(
        k, scatter_dim=head_dimension, gather_dim=seq_dimension,
        group=sp_group, async_op=True
    )

    # 6. V 投影（在 K 通信进行时）
    v = F.linear(hidden_states, v_weight, v_bias)
    v = v.view(batch_size, -1, num_kv_heads, head_dim)

    if need_repeat_kv:
        v = torch.repeat_interleave(v, dim=2, repeats=n_repeat)

    # 7. 启动 V 的异步 all-to-all 通信
    v_res = all_to_all_tensor(
        v, scatter_dim=head_dimension, gather_dim=seq_dimension,
        group=sp_group, async_op=True
    )

    # 8. 等待 Q 通信完成
    q = q_res()
    q = unpadding_tensor_for_seqeunce_parallel(q, seq_dimension, unpadded_dim_size)

    # 9. 等待 K 通信完成
    k = k_res()
    k = unpadding_tensor_for_seqeunce_parallel(k, seq_dimension, unpadded_dim_size)

    q = q.contiguous()
    k = k.contiguous()

    # 10. QK Normalization（如果启用）
    if norm_type is not None:
        if norm_type == "rmsnorm":
            output_q, invvar_q = fused_layer_norm_cuda.rms_forward_affine(
                q, normalized_shape, norm_q_weight, eps
            )
            output_k, invvar_k = fused_layer_norm_cuda.rms_forward_affine(
                k, normalized_shape, norm_k_weight, eps
            )
        elif norm_type == "layernorm":
            output_q, mean_q, invvar_q = fused_layer_norm_cuda.forward_affine(
                q, normalized_shape, norm_q_weight, norm_q_bias, eps
            )
            output_k, mean_k, invvar_k = fused_layer_norm_cuda.forward_affine(
                k, normalized_shape, norm_k_weight, norm_k_bias, eps
            )
    else:
        output_q = q
        output_k = k

    # 11. 等待 V 通信完成
    v = v_res()
    v = unpadding_tensor_for_seqeunce_parallel(v, seq_dimension, unpadded_dim_size)

    # 保存中间结果用于反向传播
    ctx.save_for_backward(hidden_states, q_weight, k_weight, v_weight, ...)

    return output_q, output_k, v
```

**反向传播关键步骤** (第 211-416 行)：

反向传播遵循相反的顺序，同样实现通信-计算重叠：

```python
def backward(ctx, *grad_output):
    # 1. 启动 V 梯度的异步 all-to-all 通信
    grad_v = grad_output[2].contiguous()
    grad_v = padding_tensor_for_seqeunce_parallel(grad_v, dim=seq_dimension)
    grad_v_res = all_to_all_tensor(
        grad_v, scatter_dim=seq_dimension, gather_dim=head_dimension,
        group=sp_group, async_op=True
    )

    # 2. QK Normalization 反向传播（在 V 梯度通信进行时）
    if norm_type == "rmsnorm":
        grad_k, grad_norm_k_weight = fused_layer_norm_cuda.rms_backward_affine(
            grad_output[1].contiguous(), invvar_k, k, ...
        )
        grad_q, grad_norm_q_weight = fused_layer_norm_cuda.rms_backward_affine(
            grad_output[0].contiguous(), invvar_q, q, ...
        )

    # 3. 等待 V 梯度通信完成
    grad_v = grad_v_res()
    if need_repeat_kv:
        # 对重复的 KV heads 进行梯度聚合
        grad_v = grad_v.reshape(
            grad_v.shape[0], grad_v.shape[1], original_num_kv_heads, n_repeat, grad_v.shape[-1]
        ).sum(dim=3)

    # 4. 启动 K 梯度的异步 all-to-all 通信
    grad_k = padding_tensor_for_seqeunce_parallel(grad_k, dim=seq_dimension)
    grad_k_res = all_to_all_tensor(
        grad_k, scatter_dim=seq_dimension, gather_dim=head_dimension,
        group=sp_group, async_op=True
    )

    # 5. 计算 V 投影的梯度（在 K 梯度通信进行时）
    grad_v = grad_v.reshape(grad_v.shape[0], grad_v.shape[1], -1)
    grad_v_input = grad_v @ v_weight
    grad_v_weight = grad_v.transpose(-1, -2) @ hidden_states
    if v_bias is not None:
        grad_v_bias = grad_v.sum(0)

    # 6. 等待 K 梯度通信完成
    grad_k = grad_k_res()
    if need_repeat_kv:
        grad_k = grad_k.reshape(...).sum(dim=3)

    # 7. 启动 Q 梯度的异步 all-to-all 通信
    grad_q = padding_tensor_for_seqeunce_parallel(grad_q, dim=seq_dimension)
    grad_q_res = all_to_all_tensor(
        grad_q, scatter_dim=seq_dimension, gather_dim=head_dimension,
        group=sp_group, async_op=True
    )

    # 8. 计算 K 投影的梯度（在 Q 梯度通信进行时）
    grad_k = grad_k.reshape(grad_k.shape[0], grad_k.shape[1], -1)
    grad_k_input = grad_k @ k_weight
    grad_k_weight = grad_k.transpose(-1, -2) @ hidden_states

    # 9. 等待 Q 梯度通信完成
    grad_q = grad_q_res()

    # 10. 计算 Q 投影的梯度
    grad_q = grad_q.reshape(grad_q.shape[0], grad_q.shape[1], -1)
    grad_q_input = grad_q @ q_weight
    grad_q_weight = grad_q.transpose(-1, -2) @ hidden_states

    # 11. 聚合所有输入梯度
    grad_hidden_states = grad_q_input + grad_k_input + grad_v_input

    return (grad_hidden_states, None, None, grad_q_weight, grad_k_weight, grad_v_weight, ...)
```

**AsyncUlyssesOutputProjection** (第 419-503 行)

该类融合了输出线性投影和 all-to-all 通信。

**前向传播**：
```python
def forward(ctx, hidden_states, seq_dimension, head_dimension,
            proj_weight, proj_bias, ...):
    # 1. Padding（如果需要）
    hidden_states = padding_tensor_for_seqeunce_parallel(hidden_states, seq_dimension)

    # 2. all-to-all 通信：聚合头，分散序列
    hidden_states = all_to_all_tensor(
        hidden_states, scatter_dim=seq_dimension, gather_dim=head_dimension, group=sp_group
    )

    # 3. Reshape 并进行线性投影
    hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1)
    o = F.linear(hidden_states, proj_weight, proj_bias)

    ctx.save_for_backward(hidden_states, proj_weight, proj_bias)
    return o
```

**反向传播**：
```python
def backward(ctx, *grad_output):
    # 1. 计算输出梯度
    grad_o = grad_output[0] @ (proj_weight)
    grad_o = grad_o.reshape(grad_o.shape[0], -1, num_heads, head_dim)

    # 2. 启动异步 all-to-all 通信
    grad_out_res = all_to_all_tensor(
        grad_o, scatter_dim=head_dimension, gather_dim=seq_dimension,
        group=sp_group, async_op=True
    )

    # 3. 计算权重梯度（在通信进行时）
    grad_proj_weight = grad_output[0].transpose(-1, -2) @ (hidden_states)
    if proj_bias is not None:
        grad_proj_bias = grad_output[0].sum(0)

    # 4. 等待通信完成并 unpad
    grad_o = grad_out_res()
    grad_o = unpadding_tensor_for_seqeunce_parallel(grad_o, seq_dimension, unpadded_dim_size)

    return (grad_o, None, None, grad_proj_weight, grad_proj_bias, None, None)
```

### 4.3 GQA (Grouped Query Attention) 支持

对于 GQA 模型（如 Llama 3），`num_kv_heads < num_q_heads`。当 `ulysses_size > num_kv_heads` 时，需要特殊处理：

**前向传播**（第 83-119 行）：
```python
if ulysses_size > num_kv_heads:
    assert ulysses_size % num_kv_heads == 0
    need_repeat_kv = True
    n_repeat = ulysses_size // num_kv_heads
    original_num_kv_heads = num_kv_heads

    # 重复 KV heads
    k = torch.repeat_interleave(k, dim=2, repeats=n_repeat)
    v = torch.repeat_interleave(v, dim=2, repeats=n_repeat)
```

**反向传播**（第 336-340 行）：
```python
if need_repeat_kv:
    # 对重复的 heads 进行梯度求和
    grad_v = grad_v.reshape(
        grad_v.shape[0], grad_v.shape[1], original_num_kv_heads, n_repeat, grad_v.shape[-1]
    ).sum(dim=3)
```

### 4.4 QK Normalization 支持

异步模式支持融合 QK Normalization（第 138-177 行）：

**支持的 Normalization 类型**：
- **RMSNorm**：使用 `fused_layer_norm_cuda.rms_forward_affine`（GPU）或 `torch_npu.npu_rms_norm`（NPU）
- **LayerNorm**：使用 `fused_layer_norm_cuda.forward_affine`

**融合优势**：
- 减少一次 kernel launch
- 利用 Q/K 通信时间进行 normalization 计算

### 4.5 异步模式使用示例

```python
from veomni.distributed.sequence_parallel.async_ulysses import (
    async_ulysses_qkv_projection,
    async_ulysses_output_projection
)

class Attention(nn.Module):
    def forward(self, x: torch.Tensor, unpadded_seq_len: int):
        # 异步 QKV 投影 + all-to-all
        q, k, v = async_ulysses_qkv_projection(
            hidden_states=x,
            seq_dimension=1,
            head_dimension=2,
            q_weight=self.q_proj.weight,
            k_weight=self.k_proj.weight,
            v_weight=self.v_proj.weight,
            q_bias=self.q_proj.bias if hasattr(self.q_proj, 'bias') else None,
            k_bias=self.k_proj.bias if hasattr(self.k_proj, 'bias') else None,
            v_bias=self.v_proj.bias if hasattr(self.v_proj, 'bias') else None,
            norm_type="rmsnorm",  # 可选：QK normalization
            norm_q_weight=self.q_norm.weight if self.qk_norm else None,
            norm_k_weight=self.k_norm.weight if self.qk_norm else None,
            normalized_shape=self.head_dim,
            eps=1e-6,
            unpadded_dim_size=unpadded_seq_len,
            head_dim=self.head_dim,
            group=None,
        )

        # Attention 计算
        x = F.scaled_dot_product_attention(q, k, v, ...)

        # 异步 Output 投影 + all-to-all
        x = async_ulysses_output_projection(
            hidden_states=x,
            seq_dimension=1,
            head_dimension=2,
            proj_weight=self.proj_o.weight,
            proj_bias=self.proj_o.bias if hasattr(self.proj_o, 'bias') else None,
            unpadded_dim_size=unpadded_seq_len,
            group=None,
        )

        return x
```

## 5. 进程组管理

### 5.1 进程组初始化

#### 文件：`veomni/distributed/sequence_parallel/comm.py`

**init_sequence_parallel** (第 245-314 行)

```python
def init_sequence_parallel(
    ulysses_size: int = 1,
    sep_dp: bool = False,
    ulysses_group_key: str = "default",
    cp_size: int = 1
):
    """
    初始化序列并行进程组

    Args:
        ulysses_size: Ulysses 序列并行大小
        sep_dp: 是否创建独立的数据并行组
        ulysses_group_key: Ulysses 组的键（支持多个 Ulysses 组）
        cp_size: Context Parallel 大小（Ring Attention，尚未完全实现）
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    unified_sp_size = ulysses_size * cp_size
    data_parallel_size = world_size // unified_sp_size

    for i in range(data_parallel_size):
        # 创建 Ulysses 组
        if ulysses_size > 1:
            for j in range(cp_size):
                start_rank = i * unified_sp_size + j * ulysses_size
                end_rank = start_rank + ulysses_size
                ulysses_ranks = range(start_rank, end_rank)
                ulysses_group = dist.new_group(ulysses_ranks)
                ulysses_cpu_group = dist.new_group(ulysses_ranks, backend="gloo")
                if rank in ulysses_ranks:
                    set_ulysses_sequence_parallel_group(
                        group=ulysses_group, group_key=ulysses_group_key
                    )
                    set_ulysses_sequence_parallel_cpu_group(
                        group=ulysses_cpu_group, group_key=ulysses_group_key
                    )

        # 创建 Context Parallel 组（尚未完全支持）
        if cp_size > 1:
            for j in range(ulysses_size):
                cp_global_ranks = range(i * unified_sp_size + j, (i + 1) * unified_sp_size, ulysses_size)
                cp_group = dist.new_group(cp_global_ranks)
                if rank in cp_global_ranks:
                    set_context_parallel_group(cp_group=cp_group)

        # 创建统一的序列并行组
        unified_sp_ranks = range(i * unified_sp_size, (i + 1) * unified_sp_size)
        sp_group = dist.new_group(unified_sp_ranks)
        sp_cpu_group = dist.new_group(unified_sp_ranks, backend="gloo")
        if rank in unified_sp_ranks:
            set_unified_sequence_parallel_group(group=sp_group)
            set_unified_sequence_parallel_cpu_group(group=sp_cpu_group)

    # 创建独立的数据并行组（可选）
    if sep_dp:
        for j in range(unified_sp_size):
            dp_ranks = range(j, world_size, unified_sp_size)
            dp_group = dist.new_group(dp_ranks)
            if rank in dp_ranks:
                set_data_parallel_group(dp_group)
```

### 5.2 进程组拓扑示例

假设 world_size=16, ulysses_size=4, cp_size=1：

```
Data Parallel Size = 16 / 4 = 4

Ulysses Groups:
  Group 0: [Rank 0, 1, 2, 3]
  Group 1: [Rank 4, 5, 6, 7]
  Group 2: [Rank 8, 9, 10, 11]
  Group 3: [Rank 12, 13, 14, 15]

Unified SP Groups (same as Ulysses when cp_size=1):
  Group 0: [Rank 0, 1, 2, 3]
  Group 1: [Rank 4, 5, 6, 7]
  Group 2: [Rank 8, 9, 10, 11]
  Group 3: [Rank 12, 13, 14, 15]
```

### 5.3 多 Ulysses 组支持

VeOmni 支持在单个训练任务中创建多个 Ulysses 组（通过 `group_key`），用于复杂的多模态模型：

```python
# 创建 text Ulysses 组
init_sequence_parallel(ulysses_size=4, ulysses_group_key="text")

# 创建 vision Ulysses 组
init_sequence_parallel(ulysses_size=8, ulysses_group_key="vision")

# 使用特定组
with UlyssesGroupKeyManager("vision"):
    x = gather_seq_scatter_heads(x, seq_dim=1, head_dim=2)
```

## 6. 数据预处理与后处理

### 6.1 输入数据预处理

#### 文件：`veomni/distributed/sequence_parallel/data.py`

**sequence_parallel_preprocess** (第 144-196 行)

```python
def sequence_parallel_preprocess(
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    sp_group: Optional[ProcessGroup] = None,
):
    """
    对输入数据进行序列并行预处理

    主要操作：
    1. 将 input_ids 沿序列维度切分到各个 rank
    2. 将 labels 进行 shift 并切分（causal language modeling）
    3. 对 position_ids、attention_mask、cu_seqlens 进行 padding 和处理
    """
    if sp_group is not None:
        sp_size = dist.get_world_size(sp_group)
        padding_size = (sp_size - (input_ids.shape[-1] % sp_size)) % sp_size

        # 切分 input_ids
        input_ids = slice_input_tensor(
            input_ids, dim=-1, padding=True, padding_value=0, group=sp_group
        )

        # 处理 labels：先 shift，再 padding，最后切分
        if labels is not None:
            labels = labels[..., 1:].contiguous()  # 左移一位
            labels = F.pad(labels, (0, 1), "constant", IGNORE_INDEX)  # padding 到与 input_ids 相同长度
            labels = slice_input_tensor(
                labels, dim=-1, padding=True, padding_value=IGNORE_INDEX, group=sp_group
            )

        # Padding position_ids
        if position_ids is not None:
            position_ids = pad_tensor(position_ids, dim=-1, padding_size=padding_size, padding_value=0)

        # Padding attention_mask
        if attention_mask is not None:
            attn_mask_padding_value = 1 if position_ids is not None else 0
            attention_mask = pad_tensor(
                attention_mask, dim=-1, padding_size=padding_size, padding_value=attn_mask_padding_value
            )

        # Padding cu_seqlens
        if cu_seqlens is not None:
            cu_seqlens_padding_value = cu_seqlens[-1].item() + padding_size
            cu_seqlens = pad_tensor(
                cu_seqlens, dim=-1, padding_size=padding_size, padding_value=cu_seqlens_padding_value
            )

    return input_ids, labels, position_ids, attention_mask, cu_seqlens
```

**设计要点**：
1. **Labels Shifting**：因果语言建模需要将 labels 左移一位
2. **Padding 策略**：确保序列长度能被 sp_size 整除
3. **IGNORE_INDEX**：对 padding 的 labels 使用 IGNORE_INDEX，避免影响损失计算

### 6.2 输出数据聚合

**gather_outputs** (第 105-122 行)

```python
def gather_outputs(
    x: Tensor,
    gather_dim: int,
    padding_dim: Optional[int] = None,
    unpad_dim_size: Optional[int] = None,
    scale_grad=True,
    group: ProcessGroup = None,
):
    """
    聚合各个 rank 的输出，恢复完整序列
    """
    group = get_unified_sequence_parallel_group() if group is None else group
    if not group:
        return x
    # All-gather 操作
    x = _Gather.apply(group, x, gather_dim, scale_grad)
    # 移除 padding
    if padding_dim is not None:
        x = unpadding_tensor_for_seqeunce_parallel(x, padding_dim, unpad_dim_size, group)
    return x
```

### 6.3 VLM 特殊处理

对于 Vision-Language 模型，图像 tokens 数量可能在不同样本间变化，需要特殊的 all-to-all 处理：

**all_to_all_images** (第 316-321 行)

```python
def all_to_all_images(image_embeds, in_splits, out_splits):
    """
    处理可变长度的图像 embeddings

    Args:
        image_embeds: 图像 embeddings [total_tokens, hidden_dim]
        in_splits: 每个 rank 的输入 token 数 [rank0_tokens, rank1_tokens, ...]
        out_splits: 每个 rank 应接收的 token 数
    """
    if not in_splits:
        return image_embeds
    image_embeds = image_embeds[: sum(in_splits)]
    group = get_ulysses_sequence_parallel_group()
    return _AlltoAllRegion.apply(group, image_embeds, in_splits, out_splits)
```

## 7. 损失函数处理

### 7.1 序列并行损失聚合

#### 文件：`veomni/distributed/sequence_parallel/loss.py`

在序列并行中，每个 rank 只计算部分序列的损失，需要进行全局聚合：

```python
def reduce_sequence_parallel_loss(
    loss: torch.Tensor,
    num_valid_tokens: int,
    sp_group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    """
    聚合序列并行组内的损失

    Args:
        loss: 当前 rank 的损失（已经是 sum，非 mean）
        num_valid_tokens: 当前 rank 的有效 token 数（排除 IGNORE_INDEX）
        sp_group: 序列并行进程组

    Returns:
        全局平均损失
    """
    if sp_group is None:
        return loss / num_valid_tokens

    # 聚合所有 rank 的损失和有效 token 数
    global_loss = dist.all_reduce(loss, group=sp_group, async_op=False)
    global_num_valid_tokens = dist.all_reduce(
        torch.tensor(num_valid_tokens, device=loss.device),
        group=sp_group,
        async_op=False
    )

    # 全局平均
    return global_loss / global_num_valid_tokens
```

**使用示例**：

```python
# 计算当前 rank 的损失
logits = model(input_ids, ...)
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1),
                       ignore_index=IGNORE_INDEX, reduction='sum')
num_valid_tokens = (labels != IGNORE_INDEX).sum()

# 序列并行损失聚合
if sp_enabled:
    loss = reduce_sequence_parallel_loss(loss, num_valid_tokens, sp_group)
else:
    loss = loss / num_valid_tokens
```

## 8. 与 FSDP/FSDP2 集成

### 8.1 设备网格拓扑

VeOmni 使用 PyTorch 的 DeviceMesh 统一管理各种并行维度。

#### 文件：`veomni/distributed/parallel_state.py`

**设备网格维度**：
```
[dp_replicate, dp_shard, ulysses, cp, tp]
```

- **dp_replicate**：数据并行复制维度（ZeRO-0）
- **dp_shard**：数据并行分片维度（FSDP）
- **ulysses**：Ulysses 序列并行维度
- **cp**：Context Parallel 维度（Ring Attention，尚未完全支持）
- **tp**：Tensor Parallel 维度

**组合进程组**：
- **dp_shard_sp**：`[dp_shard, ulysses, cp]` - 用于 FSDP 参数分片
- **sp**：`[ulysses, cp]` - 用于序列并行操作
- **dp_sp**：`[dp_replicate, dp_shard, ulysses, cp]` - 用于损失 all-reduce

### 8.2 FSDP 与 Ulysses 的协同

**关键设计决策**：
1. **参数复制**：Ulysses SP 维度上参数完全复制（不进行额外分片）
2. **梯度同步**：梯度在 `dp_shard_sp` 组内进行 all-reduce
3. **激活分片**：激活在序列维度上分片，显存节省 `1/ulysses_size`

**配置示例**：

```python
from veomni.distributed.parallel_state import ParallelState

parallel_state = ParallelState(
    dp_size=1,              # 总数据并行大小（dp_replicate * dp_shard）
    dp_replicate_size=1,    # ZeRO-0 复制
    dp_shard_size=1,        # FSDP 分片
    ulysses_size=4,         # Ulysses 序列并行
    cp_size=1,              # Context Parallel（暂不支持）
    tp_size=1,              # Tensor Parallel（暂不支持）
    include_sp_in_fsdp=True # 必须为 True（解耦模式未实现）
)
```

### 8.3 训练流程

**前向传播**：
1. 数据预处理：`sequence_parallel_preprocess` 切分输入
2. Embedding 层：每个 rank 处理部分序列
3. Transformer 层：
   - Attention 前：`gather_seq_scatter_heads`
   - Attention 计算：每个 rank 处理部分 heads
   - Attention 后：`gather_heads_scatter_seq`
4. 输出层：每个 rank 产生部分序列的 logits

**反向传播**：
1. 损失计算：每个 rank 计算部分序列的损失
2. 损失聚合：`reduce_sequence_parallel_loss` 全局平均
3. 梯度反向传播：自动反转 all-to-all 操作
4. 梯度聚合：FSDP 在 `dp_shard_sp` 组内 all-reduce

## 9. 性能优化技巧

### 9.1 Padding 优化

**问题**：当序列长度不能被 `ulysses_size` 整除时，需要 padding。

**解决方案**：
- 使用动态 batching 尽量凑齐整除的序列长度
- Padding 时使用特殊值（0 for input_ids, IGNORE_INDEX for labels）
- 在 all-to-all 后及时 unpad，避免无效计算

### 9.2 通信-计算重叠

**异步模式优势**：
- QKV 投影计算与通信并行
- 理论加速比：接近 2x（取决于计算/通信比）

**最佳实践**：
- 优先使用异步模式（`sp_async=True`）
- 确保模型支持 QK Normalization 以最大化融合收益
- 对于小模型或高带宽网络，收益可能不明显

### 9.3 GQA 模型优化

**问题**：GQA 模型的 `num_kv_heads < num_q_heads`，可能导致负载不均衡。

**解决方案**：
- 选择 `ulysses_size` 能整除 `num_kv_heads` 或能被 `num_kv_heads` 整除
- 避免 `ulysses_size` 与 `num_kv_heads` 互质的情况

**示例**：
- Llama 3 70B: num_q_heads=64, num_kv_heads=8
  - 推荐 ulysses_size: 2, 4, 8（整除）或 16, 32（倍数）
  - 不推荐 ulysses_size: 3, 5, 6（需要重复 KV heads）

### 9.4 Flash Attention 集成

Ulysses 与 Flash Attention 完全兼容：

```python
from flash_attn import flash_attn_func

# Ulysses 处理后，每个 rank 持有：
# q, k, v: [batch, seq_len, num_heads/ulysses_size, head_dim]

# 直接使用 Flash Attention
output = flash_attn_func(q, k, v, causal=True)

# Ulysses 后处理
output = gather_heads_scatter_seq(output, head_dim=2, seq_dim=1)
```

## 10. 测试与验证

### 10.1 正确性验证

VeOmni 使用以下策略验证 Ulysses 实现的正确性：

**测试文件**：`tests/parallel/ulysses/test_ulysses.py`

**验证方法**：
1. **前向传播对比**：SP 输出与 DP 输出数值一致性
2. **反向传播对比**：梯度数值一致性
3. **Padding 场景**：测试序列长度不整除的情况

**测试用例**（第 65-113 行）：

```python
def test_self_attn(self):
    # 初始化 SP 组
    sp_group = get_ulysses_sequence_parallel_group()

    # 准备数据
    full_input = torch.randn(2, 8192, 1024).cuda()  # [batch, seq, hidden]
    part_input = slice_input_tensor(full_input, dim=1, group=sp_group)

    # 初始化两个相同权重的模型
    attn_dp = Attention(..., sp_async=False).cuda()
    attn_sp = Attention(..., sp_async=False).cuda()
    attn_sp.load_state_dict(attn_dp.state_dict())

    # SP 前向+反向
    sp_rst = attn_sp(part_input, unpadded_seq_len=8192)
    sp_full_rst = gather_outputs(sp_rst, gather_dim=1, ...)
    loss_sp = sp_rst.sum() * 2
    loss_sp.backward()

    # DP 前向+反向
    dp_rst = attn_dp(full_input, unpadded_seq_len=8192)
    loss_dp = dp_rst.sum() * 2
    loss_dp.backward()

    # 验证输出一致性
    torch.testing.assert_close(dp_rst, sp_full_rst, atol=1e-6, rtol=1e-5)

    # 验证梯度一致性
    torch.testing.assert_close(
        attn_dp.proj_o.weight.grad,
        attn_sp.proj_o.weight.grad,
        atol=1e-3, rtol=1e-4
    )
```

### 10.2 异步模式验证

**测试文件**：`tests/parallel/ulysses/test_async_ulysses.py`

**验证策略**：
- 异步模式输出与同步模式输出数值一致性
- 异步模式梯度与同步模式梯度一致性

**关键差异**（第 81 行）：
```python
# 同步模式
attn_dp = Attention(..., sp_async=False)

# 异步模式
attn_sp = Attention(..., sp_async=True)
```

### 10.3 性能测试

**建议的性能测试指标**：
1. **端到端吞吐量**：samples/sec 或 tokens/sec
2. **通信时间占比**：通过 profiler 测量
3. **内存占用**：峰值显存使用
4. **扩展性**：不同 ulysses_size 的加速比

**Profiling 示例**：

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True
) as prof:
    output = model(input_ids, ...)
    loss = loss_fn(output, labels)
    loss.backward()

# 分析通信开销
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 11. 限制与已知问题

### 11.1 当前限制

1. **Head 数量约束**：`num_heads` 必须能被 `ulysses_size` 整除
2. **Context Parallel 未完全支持**：`cp_size` 参数存在，但功能未实现
3. **解耦 SP 模式未实现**：`include_sp_in_fsdp` 必须为 `True`
4. **NPU 异步模式限制**：Ascend NPU 暂不支持异步 Ulysses

### 11.2 不支持的场景

1. **局部注意力**：Sliding window attention 需要特殊处理
2. **稀疏注意力**：稀疏模式的通信模式不同
3. **Cross-Attention**：Encoder-Decoder 架构的 cross-attention 需要额外适配

### 11.3 性能瓶颈

1. **小序列长度**：序列长度 < 4096 时，通信开销可能超过收益
2. **高带宽网络**：Infiniband EDR 及以上，通信占比降低，加速比受限
3. **小 Batch Size**：batch_size=1 时，通信延迟难以隐藏

## 12. 最佳实践总结

### 12.1 何时使用 Ulysses SP

**推荐场景**：
- 序列长度 > 8192
- 单卡显存不足以容纳全序列激活
- 使用 FlashAttention 等高效 attention 实现
- 训练长文本 LLM、高分辨率 VLM、长视频生成模型

**不推荐场景**：
- 序列长度 < 2048
- 已经使用了其他序列并行方法（如 Megatron-LM 的 SP）
- 网络带宽极低（< 10 Gbps）

### 12.2 配置建议

**Ulysses Size 选择**：
```python
# 经验公式
ulysses_size = min(
    num_gpus // fsdp_size,  # 不超过可用 GPU 数
    num_heads,              # 不超过注意力头数
    seq_len // 2048         # 确保每个 rank 至少 2048 tokens
)
```

**异步模式选择**：
```python
# GPU: 优先使用异步模式
sp_async = True  # 如果模型支持 QK normalization

# NPU: 使用同步模式
sp_async = False
```

### 12.3 调试技巧

**常见错误**：

1. **维度不匹配**：
```
RuntimeError: num_query_heads (64) must be divisible by ulysses_size (5)
```
解决：选择能整除 num_heads 的 ulysses_size

2. **梯度不一致**：
检查是否正确设置了 `scale_grad` 参数

3. **OOM**：
虽然激活显存减少，但参数显存未减少。考虑同时使用 FSDP

**日志监控**：
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看进程组信息
logger.debug(f"Ulysses group: {get_ulysses_sequence_parallel_group()}")
logger.debug(f"Ulysses world size: {get_ulysses_sequence_parallel_world_size()}")
logger.debug(f"Ulysses rank: {get_ulysses_sequence_parallel_rank()}")
```

## 13. 未来改进方向

### 13.1 待实现功能

1. **Context Parallel (Ring Attention)**：支持超长序列（> 1M tokens）
2. **解耦 SP 模式**：SP 维度独立于 FSDP 维度
3. **动态 Ulysses Size**：根据序列长度动态调整并行度
4. **Hybrid SP**：结合 Ulysses 和 Megatron SP

### 13.2 性能优化方向

1. **更细粒度的通信-计算重叠**：在 Attention 内部进行流水线
2. **通信压缩**：FP16/BF16 通信，降低带宽需求
3. **自适应 Padding**：根据实际序列长度分布优化 padding 策略

### 13.3 易用性改进

1. **自动配置**：根据模型和硬件自动选择最优 ulysses_size
2. **更好的错误提示**：参数配置错误时给出明确的修复建议
3. **性能分析工具**：集成 profiler，自动分析通信瓶颈

## 14. 参考资料

### 14.1 论文

1. **DeepSpeed Ulysses**: [arXiv:2309.14509](https://arxiv.org/abs/2309.14509) - "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models"
2. **VeOmni**: [arXiv:2508.02317](https://arxiv.org/abs/2508.02317) - "VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo"

### 14.2 相关代码

- DeepSpeed 官方实现：https://github.com/microsoft/DeepSpeed
- PyTorch FSDP2：https://pytorch.org/docs/stable/fsdp.html
- Flash Attention：https://github.com/Dao-AILab/flash-attention

### 14.3 VeOmni 文档

- 官方文档：https://veomni.readthedocs.io/
- GitHub 仓库：https://github.com/ByteDance-Seed/VeOmni
- Ulysses 使用示例：`docs/key_features/ulysses.md`

---

**文档版本**: 1.0
**最后更新**: 2026-01-03
**作者**: VeOmni Team Analysis
