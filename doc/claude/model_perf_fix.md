# model_perf_fix.md

# HT-Transformer-DS 模型性能修复清单（Claude Code 执行版）

本文档用于指导 Claude Code 对当前 `HT-Transformer-DS` 的 **model 相关性能瓶颈** 进行修复。  
目标不是改动总体架构，而是在 **保持 v2 设计方向不变** 的前提下，修复当前实现中的主要性能瓶颈与若干 correctness 问题。

---

## 一、修复目标

本轮修复的目标：

1. **显著降低 DeepSphere 分支的 Python 循环开销**
2. **避免在每个 DeepSphere block 中重复构造局部邻接**
3. **修复 CDCompression 与 `cd_mask` 的 padding 语义不一致问题**
4. **让 CD encoder 显式接收并遵守 `cd_mask`**
5. **降低 WP time bias 导致的 attention 慢路径开销**
6. **保留当前总体架构**
   - 保留 `HTTransformer`
   - 保留 `DeepSphereEncoder + CDCompression`
   - 保留 `WPProjector` 双支路
   - 保留 `SignedTimeBucketBias`
   - 保留 Hybrid Token Fusion 结构

---

## 二、禁止项

Claude Code **不要**做以下修改：

1. 不要删除 `DeepSphereEncoder`
2. 不要把 CD 分支改回 full attention
3. 不要删除 `SignedTimeBucketBias`
4. 不要重写整个 `HTTransformer`
5. 不要修改 loss 设计
6. 不要在本轮引入新依赖库
7. 不要为了追求性能而牺牲 `active pixel ids -> local neighbors` 的语义正确性

---

## 三、优先级总览

### P0（必须先做）
1. `deepsphere.py`
   - 把局部邻接构造从每层 block 中移出，只对每个 batch 构造一次
   - 把 `_aggregate_neighbors()` 改成张量化 gather
   - 显式支持 `cd_mask`
2. `deepsphere.py`
   - 修复 `CDCompression` 的 padding pixel id，统一使用 `-1`
3. `ht_transformer.py`
   - 接入新的 batch 级局部邻接缓存
   - 给 `cd_encoder` 显式传 `cd_mask`

### P1（高价值性能优化）
4. `ht_transformer.py`
   - 允许 `wp_time_bias` 只在前若干层启用
5. `wp_time_bias.py`
   - 增加轻量模式支持：`heads_shared=True` 更方便使用
6. `ht_transformer.py`
   - 让 `wp_time_bias` 只计算一次，并复用

### P2（工程收尾）
7. 清理未使用逻辑/参数
8. 补充注释
9. 增加调试日志（可选，默认关闭）

---

# 四、逐文件修复说明

---

## 文件 1：`models/components/deepsphere.py`

这是本轮修复的核心文件。

---

### 任务 1.1：把局部邻接映射移到 batch 级，只构造一次

#### 当前问题
当前 `build_local_neighbor_indices()` 是在 `DeepSphereBlock.forward()` 中被调用的。  
由于 `DeepSphereEncoder` 会堆叠多个 block，这会导致：

- 每个 block 都重新构造一次局部邻接
- 重复的 Python 字典与 Python for-loop 开销
- 当 `num_layers` 增大时，耗时近似线性放大

#### 目标改法
将“global pixel id -> local token index”映射的构造提升到 **encoder 级**，而不是 block 级。

建议：

- `DeepSphereEncoder.forward()` 中只构造一次：
  - `local_neighbor_indices`
  - `valid_neighbor_mask`
- 然后把这两者传给所有 `DeepSphereBlock`

---

### 参考代码：新增批级局部邻接构造函数

请将旧的 `build_local_neighbor_indices(...)` 替换为更明确的 batch 级版本，例如：

```python
import torch
from typing import Tuple


def build_local_neighbor_graph(
    pixel_ids: torch.Tensor,
    full_knn_adj: torch.Tensor,
    cd_mask: torch.Tensor | None = None,
    padding_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build local neighbor graph once per batch.

    Args:
        pixel_ids: (B, N) global HEALPix pixel ids for active/padded CD tokens
        full_knn_adj: (npix, k) global neighbor table
        cd_mask: (B, N) bool, True = padding
        padding_value: value for invalid local neighbor

    Returns:
        local_neighbor_indices: (B, N, k) local indices
        valid_neighbor_mask: (B, N, k) bool, True = valid
    """
    B, N = pixel_ids.shape
    k = full_knn_adj.shape[1]
    device = pixel_ids.device

    local_neighbor_indices = torch.full(
        (B, N, k), padding_value, dtype=torch.long, device=device
    )
    valid_neighbor_mask = torch.zeros(
        (B, N, k), dtype=torch.bool, device=device
    )

    for b in range(B):
        if cd_mask is not None:
            valid_nodes = (~cd_mask[b]).nonzero(as_tuple=False).squeeze(-1)
        else:
            valid_nodes = torch.arange(N, device=device)

        if valid_nodes.numel() == 0:
            continue

        active_pixel_ids = pixel_ids[b, valid_nodes]
        global_to_local = {
            int(active_pixel_ids[i].item()): int(valid_nodes[i].item())
            for i in range(active_pixel_ids.numel())
        }

        for local_token_idx in valid_nodes.tolist():
            global_pid = int(pixel_ids[b, local_token_idx].item())
            global_neighbors = full_knn_adj[global_pid]

            write_idx = 0
            for nbr_pid in global_neighbors.tolist():
                mapped = global_to_local.get(int(nbr_pid), None)
                if mapped is not None:
                    local_neighbor_indices[b, local_token_idx, write_idx] = mapped
                    valid_neighbor_mask[b, local_token_idx, write_idx] = True
                    write_idx += 1
                    if write_idx >= k:
                        break

    return local_neighbor_indices, valid_neighbor_mask
```

#### 注意
这份实现虽然仍然有 Python 循环，但它会从“每个 block 都构造一次”变为“每个 batch 构造一次”，先把最大冗余砍掉。  
后续如果还需要更快，再继续做张量化版本。

---

### 任务 1.2：修改 `DeepSphereBlock`，不再自己构造邻接

#### 当前问题
`DeepSphereBlock.forward()` 当前签名大致是：

```python
def forward(self, x, pixel_ids=None, full_knn_adj=None):
```

它内部自己调用 `build_local_neighbor_indices()`。这会导致重复计算。

#### 目标改法
改为：

```python
def forward(self, x, local_neighbor_indices=None, valid_neighbor_mask=None):
```

即 block 只负责：
- 用已经构造好的局部邻接做聚合
- 不再自己建图

---

### 参考代码：重写 `DeepSphereBlock.forward`

```python
def forward(
    self,
    x: torch.Tensor,
    local_neighbor_indices: torch.Tensor | None = None,
    valid_neighbor_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    x_norm = self.norm(x)

    if local_neighbor_indices is not None:
        neighbor_agg = self._aggregate_neighbors(
            x_norm,
            local_neighbor_indices,
            valid_neighbor_mask,
        )
    else:
        neighbor_agg = x_norm.mean(dim=1, keepdim=True).expand_as(x_norm)

    neighbor_feat = self.neighbor_proj(neighbor_agg)
    self_feat = self.self_proj(x_norm)
    combined = neighbor_feat + self_feat

    out = x + self.mlp(combined)
    return out
```

---

### 任务 1.3：把 `_aggregate_neighbors()` 改成张量化 gather，去掉 `for b in range(B)`

#### 当前问题
现在 `_aggregate_neighbors()` 里有：

```python
for b in range(B):
    idx = neighbor_indices[b]
    nf = x[b][idx]
```

这会严重拖慢训练。

#### 目标改法
改成真正的 batched gather。

---

### 参考代码：张量化 `_aggregate_neighbors()`

```python
def _aggregate_neighbors(
    self,
    x: torch.Tensor,
    neighbor_indices: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Args:
        x: (B, N, D)
        neighbor_indices: (B, N, k)
        valid_mask: (B, N, k), True = valid

    Returns:
        (B, N, D)
    """
    B, N, D = x.shape
    _, _, k = neighbor_indices.shape

    # 先把 -1 替换成 0，避免 gather 非法
    safe_indices = neighbor_indices.clamp(min=0)

    # x: (B,N,D)
    # 先扩成 (B,N,k,D)，再沿 token 维 gather
    x_expand = x.unsqueeze(2).expand(B, N, k, D)
    gather_index = safe_indices.unsqueeze(-1).expand(B, N, k, D)
    neighbor_features = torch.gather(x_expand, dim=1, index=gather_index)  # (B,N,k,D)

    if valid_mask is not None:
        masked = neighbor_features * valid_mask.unsqueeze(-1).to(x.dtype)
        denom = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(x.dtype)
        return masked.sum(dim=2) / denom
    else:
        return neighbor_features.mean(dim=2)
```

#### 注意
如果你发现 `torch.gather` 这一版维度不对，请优先保证输出形状严格为 `(B,N,k,D)`。  
必要时允许改成 flatten + offset 的 gather 写法，但禁止退回 batch Python 循环。

---

### 任务 1.4：让 `DeepSphereEncoder` 只构造一次 batch 邻接并复用给所有 block

### 参考代码：重写 `DeepSphereEncoder.forward`

```python
def forward(
    self,
    x: torch.Tensor,
    pixel_ids: torch.Tensor | None = None,
    full_knn_adj: torch.Tensor | None = None,
    cd_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    local_neighbor_indices = None
    valid_neighbor_mask = None

    if full_knn_adj is not None and pixel_ids is not None:
        local_neighbor_indices, valid_neighbor_mask = build_local_neighbor_graph(
            pixel_ids=pixel_ids,
            full_knn_adj=full_knn_adj,
            cd_mask=cd_mask,
            padding_value=-1,
        )

    for block in self.blocks:
        x = block(
            x,
            local_neighbor_indices=local_neighbor_indices,
            valid_neighbor_mask=valid_neighbor_mask,
        )
    return x
```

---

### 任务 1.5：修复 `CDCompression._healpix_pool()` 的 padding pixel id

#### 当前问题
现在 padding 仍然是 0，这会和合法 HEALPix pixel 0 冲突。

#### 目标改法
统一用 `-1` 表示 padding。

---

### 参考代码：修改 padding 初始化

把：

```python
pixel_ids_padded = torch.zeros(B, max_len, dtype=torch.long, device=x.device)
```

改成：

```python
pixel_ids_padded = torch.full(
    (B, max_len),
    fill_value=-1,
    dtype=torch.long,
    device=x.device,
)
```

同时保留：

```python
pooled_padded = torch.zeros(B, max_len, D, device=x.device, dtype=x.dtype)
```

因为 embedding padding 为 0 没问题，关键是 pixel id padding 要明确为 `-1`。

---

### 任务 1.6：让 `CDCompression` 尽量遵守输入 mask

#### 当前问题
`_healpix_pool()` 里虽然有 `pixel_ids`，但没有显式根据 `mask` 去过滤 padded token。

#### 目标改法
如果 `mask` 被传入，则只处理 `mask == False` 的有效 token。

---

### 参考代码：在 `_healpix_pool()` 中接入 mask

将函数签名改成：

```python
def _healpix_pool(
    self,
    x: torch.Tensor,
    pixel_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

在 batch 循环中加入：

```python
if mask is not None:
    valid_token_mask = ~mask[b]
    x_b = x[b][valid_token_mask]
    low_ids_b = low_ids_flat[b][valid_token_mask]
else:
    x_b = x[b]
    low_ids_b = low_ids_flat[b]
```

并在 `forward()` 中把 `mask` 传进去。

---

## 文件 2：`models/ht_transformer.py`

这是第二个核心文件。

---

### 任务 2.1：给 `cd_encoder` 显式传入 `cd_mask`

#### 当前问题
当前 `cd_encoder(...)` 没有接 `cd_mask`，CD encoder 无法显式知道哪些 token 是 padding。

#### 参考代码

把：

```python
cd_emb = self.cd_encoder(cd_emb, pixel_ids=cd_pixel_ids,
                         full_knn_adj=self.cd_knn_adj)
```

改成：

```python
cd_emb = self.cd_encoder(
    cd_emb,
    pixel_ids=cd_pixel_ids,
    full_knn_adj=self.cd_knn_adj,
    cd_mask=batch.get('cd_mask', None),
)
```

---

### 任务 2.2：压缩后 `cd_mask` 逻辑保持和 `-1` 约定一致

#### 当前代码
当前是：

```python
cd_mask = (cd_emb.abs().sum(dim=-1) == 0) if cd_pixel_ids_fused is None else \
          (cd_pixel_ids_fused == -1)
```

这一条基本对，但前提是 `CDCompression` 真的把 padding 写成 `-1`。  
本项无需大改，只要在 `deepsphere.py` 中修好 padding 语义即可。

---

### 任务 2.3：为 WP time bias 增加“只在前若干层启用”的能力

#### 当前问题
现在 `wp_time_bias` 会在所有 fusion layers 中使用，导致：
- 所有 WP self-attn 都走慢路径
- 每层都要付出 `N_wp × N_wp` bias 代价

#### 目标改法
新增配置项，例如：

```yaml
model:
  wp_time_bias_layers: 1
```

表示只在前 `k` 层启用 time bias，后续层不使用，从而恢复 SDPA 快路径。

---

### 参考代码：在 `HTTransformer.__init__` 中记录层数

```python
self.wp_time_bias_layers = model_cfg.get('wp_time_bias_layers', self.num_layers)
```

---

### 参考代码：在 forward 中按层选择是否启用 bias

把：

```python
for layer in self.encoder_layers:
    wp_emb, cd_emb, global_emb, query_emb = layer(
        wp_emb, cd_emb, global_emb, query_emb,
        wp_mask=batch['wp_mask'],
        cd_mask=cd_mask,
        wp_time_bias=wp_time_bias,
    )
```

改成：

```python
for layer_idx, layer in enumerate(self.encoder_layers):
    layer_wp_time_bias = wp_time_bias if layer_idx < self.wp_time_bias_layers else None

    wp_emb, cd_emb, global_emb, query_emb = layer(
        wp_emb, cd_emb, global_emb, query_emb,
        wp_mask=batch['wp_mask'],
        cd_mask=cd_mask,
        wp_time_bias=layer_wp_time_bias,
    )
```

---

### 任务 2.4：保留 `wp_time_bias` 只计算一次并复用
这一点当前其实已经做到了：`wp_time_bias` 在 `HTTransformer.forward()` 中只生成一次，然后复用给各层。  
请保留这一点，不要退化成“每层重复重新算 bias”。

---

### 任务 2.5：给 `MultiHeadAttention` 增加一个注释和 TODO，标注当前慢路径原因
不是功能修复，而是为了后续维护。

#### 建议注释
在 `MultiHeadAttention.forward()` 的 `else` 分支上方加入注释：

```python
# NOTE:
# When time_bias is provided, we currently fall back to manual full attention.
# This disables SDPA/FlashAttention fast path and can be a major performance bottleneck.
# Future optimization direction:
#   - encode bias into SDPA-compatible attn_mask if numerically safe
#   - or enable time bias only for early layers
```

---

## 文件 3：`models/components/wp_time_bias.py`

---

### 任务 3.1：确保轻量模式配置可用
当前 `heads_shared` 已经实现，但为了让后续配置更容易生效，请补一条注释说明建议使用方式。

#### 建议注释
在 `SignedTimeBucketBias.__init__()` 里添加：

```python
# Performance note:
# heads_shared=True is cheaper and usually recommended for large N_wp.
```

---

### 任务 3.2：可选增强——增加“快速禁用模式”
新增一个可选短路逻辑：如果 `num_buckets <= 1`，直接返回全零 bias。

#### 参考代码

在 `forward()` 开头加：

```python
if self.num_buckets <= 1:
    B, N = wp_times.shape
    return torch.zeros(
        B, self.num_heads, N, N,
        device=wp_times.device,
        dtype=wp_times.dtype,
    )
```

这不是必须项，但方便快速做 ablation / profiling。

---

## 文件 4：`models/components/token_projectors.py`

这一文件本轮不是瓶颈重点，不做大改。

### 任务 4.1：不改架构，只补注释
给 `WPProjector` 和 `CDProjector` 补充一句性能说明：

- `WPProjector` 当前不是主要瓶颈
- `CDProjector` 的 Conv1d 时间编码开销相对可接受

不做结构修改。

---

## 文件 5：`models/components/position_encoding.py`

### 任务 5.1：本轮不接入新逻辑
该文件当前不是运行时热点。  
如果它在当前 v2 主模型中未使用，可以：

- 保留
- 但在文件头注明“当前 v2 主模型未使用该模块，保留用于后续实验”

不要在本轮删除它。

---

## 文件 6：`models/losses/endpoint_loss.py`

### 任务 6.1：本轮不修改
loss 不是当前训练速度的主因，不做修改。

---

# 五、建议的配置修改

请在配置中新增或确认以下字段：

```yaml
model:
  wp_time_bias_layers: 1      # 只在前 1 层启用 time bias，可改为 0/1/2/...
  wp_time_bias_heads_shared: true
```

推荐默认：

- `wp_time_bias_layers: 1`
- `wp_time_bias_heads_shared: true`

这样能显著降低 WP 分支慢路径代价。

---

# 六、建议的 profiling 验证步骤

在完成修改后，请 Claude Code 不要只做静态修复，还要做一次最小 profiling 验证。

---

## 验证 1：前向可运行

随机构造一个最小 batch，确认：

- `HTTransformer.forward()` 正常返回
- 无 NaN
- shape 正确

---

## 验证 2：DeepSphere 邻接只构造一次

临时在 `build_local_neighbor_graph()` 中加日志或计数器，确认：

- 每个 batch 只调用一次
- 不是每个 block 都调用一次

---

## 验证 3：compression 后 mask 正确

构造包含合法 pixel id `0` 的样例，确认：

- 合法 pixel `0` 不会被 mask
- padding 才是 `-1`
- `cd_mask == (cd_pixel_ids_fused == -1)` 语义正确

---

## 验证 4：WP time bias 分层启用有效

当：

```yaml
wp_time_bias_layers: 1
```

时，确认：
- 第 0 层带 bias
- 第 1 层及以后 `time_bias is None`
- 这些层重新走 SDPA 路径

可以通过临时日志验证。

---

## 验证 5：性能回归检查

至少做两组对比：

### 对比 A：修改前 vs 修改后
记录：
- 单个 train step 耗时
- 单次 forward/backward 耗时

### 对比 B：不同 `wp_time_bias_layers`
测试：
- `0`
- `1`
- `2`

目标：
- 找到一个性能/效果折中较好的默认值

---

# 七、最终验收标准

Claude Code 完成修改后，必须满足以下条件：

1. `HTTransformer` 仍保持当前 v2 总体结构
2. `DeepSphereEncoder` 不再在每个 block 中重复建局部邻接
3. `_aggregate_neighbors()` 不再包含 `for b in range(B)` 的 batch Python 循环
4. `CDCompression` 的 padding pixel id 为 `-1`
5. `cd_encoder` 显式接收 `cd_mask`
6. 可以通过配置控制只在前若干层启用 `wp_time_bias`
7. 最小前向测试通过
8. 不引入新的功能性 bug
9. 训练速度相较当前实现应有可观察改善

---

# 八、执行顺序（严格遵守）

1. 先改 `deepsphere.py`
2. 再改 `ht_transformer.py`
3. 再补 `wp_time_bias.py` 的轻量辅助逻辑
4. 最后做配置修改与最小 profiling 验证

不要先改 `ht_transformer.py` 再改 `deepsphere.py`，否则接口会不一致。

---

# 九、交付要求

Claude Code 最终需要给出：

1. 修改后的代码
2. 简短变更说明
3. 最小 profiling 对比结果
4. 哪些优化已经完成
5. 哪些优化仍然只是建议、未在本轮实现
