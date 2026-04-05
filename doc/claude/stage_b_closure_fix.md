# Stage B 闭环修复清单

## 目的

这份文档用于指导 Claude Code 完成 **Stage B 闭环修复**。  
目标不是继续做局部性能小优化，而是把当前 **模型端已按 fixed HEALPix grid 假设改写、但数据端仍停留在 active patch 变长表示** 的不一致彻底修正。

本清单**只针对以下三个文件**：

- `data/preprocess.py`
- `data/dataset.py`
- `models/components/deepsphere.py`

并要求最终达到：

1. CD 数据在预处理和训练读取阶段都采用 **固定 HEALPix 稠密网格表示**
2. DeepSphere 直接运行在 **固定全局图** 上
3. CD compression 统一为 **dense fixed-grid pooling**
4. 删除或停用旧的 active-patch 变长路径，避免新旧逻辑并存导致语义错位

---

# 一、当前问题总结

当前仓库状态是：

- `models/components/deepsphere.py` 已经朝 fixed-grid 设计迁移
- `models/ht_transformer.py` 已经按 fixed-grid 风格在调用 `cd_encoder`
- 但 `data/dataset.py` 仍然返回 **active patch 变长 token**
- `data/preprocess.py` 也仍然在缓存这种 active patch 表示

这会导致一个根本问题：

> 模型端默认 token index 对应全局 HEALPix pixel id  
> 但数据端实际上仍然只给出当前事件 active patch 列表

因此 Stage B 还没有闭环。

---

# 二、最终目标数据表示

Stage B 完成后，CD 输入必须改成如下形式。

对于给定 `nside`，设：

- `npix = 12 * nside^2`

例如 `nside = 8` 时：

- `npix = 768`

则每个事件的 CD 表示统一为：

- `cd_unit_vecs_dense`: `(npix, 3)`
- `cd_stats_dense`: `(npix, 4)`，顺序保持当前约定 `[sumQ, count, t_min, t_mean]`
- `cd_time_bins_dense`: `(npix, num_time_bins)`
- `cd_mask_dense`: `(npix,)`，`True = padding / inactive patch`

这里的“padding / inactive patch”不是 batch 级 ragged padding，而是：

- 某个全局 HEALPix pixel 在当前事件没有命中时，视为 inactive
- feature 全 0
- `cd_mask_dense[p] = True`

也就是说，**每个事件的 CD 张量长度固定为 `npix`**。

---

# 三、执行原则

Claude Code 必须遵守以下原则：

## 1. 不改动任务语义
不能改变：
- 双端点有序回归任务
- loss 接口
- 训练输出格式

## 2. 不保留 active patch 作为主路径
可以保留旧函数用于兼容或临时对照，但：
- 默认训练主路径必须使用 dense HEALPix representation
- 不允许“模型按 dense-grid 假设、数据仍走 active patch”的半迁移状态继续存在

## 3. 不在 forward 主路径里做 runtime local graph construction
Stage B 的目标之一就是：
- fixed graph
- fixed token index
- no runtime remapping

## 4. fixed-grid 优先于省 token
只要低分辨率输出 token 数可接受，就优先固定 shape，而不是再做 active token compaction。

---

# 四、逐文件改造说明

---

## A. `data/dataset.py`

### 目标
把当前 CD tokenization 从 **active patch 列表** 改成 **dense HEALPix grid**。

---

### A1. 需要修改的数据输出格式

当前 `__getitem__()` 或等价 tokenization 逻辑中，CD 部分不应再返回：

- 仅 active patch 的 `cd_stats`
- 仅 active patch 的 `cd_time_bins`
- 仅 active patch 的 `cd_unit_vecs`
- `cd_pixel_ids` 作为主索引语义

而应改为返回固定 shape：

```python
{
    "cd_unit_vecs": cd_unit_vecs_dense,   # (npix, 3)
    "cd_stats": cd_stats_dense,           # (npix, 4)
    "cd_time_bins": cd_time_bins_dense,   # (npix, num_time_bins)
    "cd_mask": cd_mask_dense,             # (npix,)
}
```

其中：

- `cd_unit_vecs_dense[p]` 必须与全局 HEALPix pixel `p` 对应
- 对 inactive patch：
  - `cd_stats_dense[p] = 0`
  - `cd_time_bins_dense[p] = 0`
  - `cd_mask_dense[p] = True`

---

### A2. 不再把 `unique_pixels` 压缩成 active list
当前问题通常出在类似下面的逻辑：

```python
unique_pixels, inverse = np.unique(pixel_ids, return_inverse=True)
n_patches = len(unique_pixels)
...
cd_stats = np.zeros((n_patches, 4), ...)
cd_time_bins = np.zeros((n_patches, num_bins), ...)
```

这条逻辑必须改掉。

应改成：

- 直接分配 `(npix, ...)`
- 用 pixel id 作为全局索引写入

---

### A3. 参考代码：dense HEALPix 聚合函数

Claude Code 可直接在 `dataset.py` 中新增一个辅助函数，例如：

```python
import numpy as np

def build_dense_cd_patches(
    pixel_ids: np.ndarray,
    cd_unit_vecs_hits: np.ndarray,
    cd_times: np.ndarray,
    cd_charges: np.ndarray,
    pixel_center_vecs: np.ndarray,
    npix: int,
    num_time_bins: int,
    t_max: float,
):
    """
    Build dense HEALPix patch representation.

    Args:
        pixel_ids: (N_hits,) HEALPix pixel id per CD hit
        cd_unit_vecs_hits: (N_hits, 3) CD hit directions
        cd_times: (N_hits,)
        cd_charges: (N_hits,)
        pixel_center_vecs: (npix, 3) fixed HEALPix pixel centers
        npix: total number of pixels for nside
        num_time_bins: number of time bins
        t_max: max time for binning

    Returns:
        cd_unit_vecs_dense: (npix, 3)
        cd_stats_dense: (npix, 4)
        cd_time_bins_dense: (npix, num_time_bins)
        cd_mask_dense: (npix,) True = inactive
    """
    cd_unit_vecs_dense = pixel_center_vecs.astype(np.float32).copy()
    cd_stats_dense = np.zeros((npix, 4), dtype=np.float32)
    cd_time_bins_dense = np.zeros((npix, num_time_bins), dtype=np.float32)
    cd_mask_dense = np.ones((npix,), dtype=bool)

    if len(pixel_ids) == 0:
        return cd_unit_vecs_dense, cd_stats_dense, cd_time_bins_dense, cd_mask_dense

    # group hits by pixel
    for pix in np.unique(pixel_ids):
        mask = (pixel_ids == pix)
        hit_q = cd_charges[mask]
        hit_t = cd_times[mask]

        cd_stats_dense[pix, 0] = hit_q.sum()  # sumQ
        cd_stats_dense[pix, 1] = len(hit_q)   # count
        cd_stats_dense[pix, 2] = hit_t.min() if len(hit_t) > 0 else 0.0
        cd_stats_dense[pix, 3] = (hit_q * hit_t).sum() / (hit_q.sum() + 1e-10)

        # dense time bins
        bin_edges = np.linspace(0, t_max, num_time_bins + 1)
        bin_idx = np.clip(np.digitize(hit_t, bin_edges) - 1, 0, num_time_bins - 1)
        np.add.at(cd_time_bins_dense[pix], bin_idx, hit_q)

        cd_mask_dense[pix] = False

    return cd_unit_vecs_dense, cd_stats_dense, cd_time_bins_dense, cd_mask_dense
```

---

### A4. `cd_pixel_ids` 的处理策略
Stage B 完成后，**主路径不再需要 `cd_pixel_ids` 作为 forward 必需输入**。

建议：

- dataset 中可以不再返回 `cd_pixel_ids`
- 如果为了兼容老接口暂时保留，也应明确标注：
  - `deprecated`
  - model 主路径不使用它

---

### A5. batch collate 逻辑
由于 CD 已变成 fixed shape：

- `cd_unit_vecs`, `cd_stats`, `cd_time_bins`, `cd_mask`
  都应直接 `stack`
- 不再需要对 CD 做“按当前 batch 最大 patch 数 padding”的逻辑

如果你当前 `collate_fn` 是对 CD 变长 padding 的，必须同步清理。

---

### A6. 验收标准
Claude Code 修改后必须验证：

1. 单个 event 输出的 `cd_stats.shape[0] == npix`
2. 所有 event 的 CD shape 完全一致
3. inactive patch feature 全 0，`cd_mask=True`
4. active patch feature 非零，`cd_mask=False`

---

## B. `data/preprocess.py`

### 目标
让预处理缓存保存 **dense HEALPix representation**，而不是旧的 active patch representation。

---

### B1. 预处理缓存必须与 Stage B 数据格式一致
当前 `preprocess.py` 如果仍然只是把旧 `dataset.py` 的 active-patch 输出直接 `torch.save` 到 `.pt` 文件，那么 Stage B 就无法闭环。

预处理缓存中的每个 event 必须保存：

- `cd_unit_vecs`: `(npix, 3)`
- `cd_stats`: `(npix, 4)`
- `cd_time_bins`: `(npix, num_time_bins)`
- `cd_mask`: `(npix,)`

这意味着预处理阶段和在线 dataset 阶段必须使用**同一套 dense tokenization 逻辑**。

---

### B2. 禁止“预处理仍保存变长 active patch，但训练时假装 dense-grid”
这是当前最需要避免的错误状态。

必须确保：

- 原始 h5 -> dense event dict
- preprocess cache -> dense event dict
- train loader / val loader -> dense event dict

三者语义一致。

---

### B3. manifest / config hash 建议增加 representation 标识
建议在 `manifest.json` 或 config hash 中加入：

```python
"cd_representation": "dense_healpix"
```

这样可以防止：
- 旧缓存和新代码混用
- 看起来能读，实际语义不一致

---

### B4. 参考代码：manifest 增强
在写 manifest 时增加：

```python
manifest = {
    ...
    "cd_representation": "dense_healpix",
    "nside": config["data"]["nside"],
    "npix": 12 * config["data"]["nside"] * config["data"]["nside"],
    "num_time_bins": config["data"]["num_time_bins"],
}
```

同时在读取预处理缓存时检查：

- `cd_representation == "dense_healpix"`

如果不是，直接报错并提示重新预处理。

---

### B5. 强制缓存失效策略
如果之前已经有旧版 preprocessed 数据，Claude Code 必须：

- 检查旧 manifest 是否缺少 `cd_representation: dense_healpix`
- 如果缺少，不允许继续训练主路径静默使用旧缓存
- 应给出明确错误信息：
  - “existing preprocessed cache is old active-patch format; please re-run preprocess for dense_healpix stage B”

---

### B6. 验收标准
1. 新生成的 `.pt` 事件缓存中 CD shape 固定
2. manifest 明确记录 dense_healpix
3. 旧缓存不会被误用
4. 使用预处理缓存训练时，CD 输入 shape 与在线 dataset 一致

---

## C. `models/components/deepsphere.py`

### 目标
让 DeepSphere 与 compression **真正运行在 fixed dense HEALPix grid 上**，删除 Stage B 不再需要的 runtime remapping / active compaction 逻辑。

---

### C1. DeepSphereEncoder 不再接收 `pixel_ids`
Stage B 完成后，固定图语义是：

- token index == global HEALPix pixel id
- 所有样本共享同一个全局邻接表

因此 `DeepSphereEncoder.forward()` 应改成：

```python
def forward(self, x, knn_adj, mask=None):
    ...
```

不再需要：

- `pixel_ids`
- `build_local_neighbor_graph`
- 任何 global->local remap 逻辑

---

### C2. 删除或停用 `build_local_neighbor_graph`
如果该函数目前仍在文件中，Stage B 后应：

- 从主路径中彻底删除调用
- 可以保留为 legacy helper，但要标记 deprecated
- 更推荐直接移除，避免误用

因为 fixed dense grid 下它已经没有任何必要。

---

### C3. DeepSphereBlock 直接使用全局 `knn_adj`
参考实现思路：

```python
def _aggregate_neighbors_fixed(self, x, knn_adj, mask=None):
    """
    x: (B, npix, D)
    knn_adj: (npix, k)
    mask: (B, npix), True = inactive
    """
    B, N, D = x.shape
    k = knn_adj.shape[1]

    # expand adjacency for batch
    idx = knn_adj.unsqueeze(0).expand(B, -1, -1)  # (B, N, k)

    # gather neighbor features
    gather_idx = idx.unsqueeze(-1).expand(B, N, k, D)
    x_expand = x.unsqueeze(2).expand(B, N, k, D)
    nbr = torch.gather(x_expand, dim=1, index=gather_idx)  # (B, N, k, D)

    if mask is not None:
        nbr_mask = mask.gather(
            1, idx.reshape(B, -1)
        ).view(B, N, k)
        valid = (~nbr_mask).to(x.dtype)
        denom = valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
        nbr = nbr * valid.unsqueeze(-1)
        return nbr.sum(dim=2) / denom
    else:
        return nbr.mean(dim=2)
```

注意：
- 这里的 `mask` 是固定 dense-grid 下的 inactive patch mask
- 不再存在局部索引和 `-1` 邻居的问题

---

### C4. DeepSphereEncoder.forward 参考接口
```python
def forward(self, x, knn_adj, mask=None):
    for block in self.blocks:
        x = block(x, knn_adj=knn_adj, mask=mask)
    return x
```

每层 block 都共享相同的固定 `knn_adj`。

---

### C5. `CDCompression` 必须统一到 dense fixed-grid pooling
Stage B 完成后，`CDCompression` 不应再保留 active-token compaction 作为主路径。

如果你已经有 `_healpix_pool_dense(...)`，则：

- 让 `forward()` 主路径只走 dense version
- 不再用旧 `_healpix_pool(...)` 生成 variable-length active token list

---

### C6. 推荐的 dense pooling 参考代码
```python
def _healpix_pool_dense(self, x, mask=None):
    """
    x: (B, npix_in, D)
    mask: (B, npix_in), True = inactive
    returns:
        pooled: (B, npix_out, D)
        out_mask: (B, npix_out)
    """
    B, N, D = x.shape
    npix_out = int(self.npix_out.item())

    if mask is None:
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)

    high_to_low = self.high_to_low.view(1, N).expand(B, -1)  # (B, N)
    batch_offsets = (torch.arange(B, device=x.device) * npix_out).view(B, 1)
    bins = high_to_low + batch_offsets

    valid = ~mask
    bins_valid = bins[valid]
    x_valid = x[valid]

    out = torch.zeros(B * npix_out, D, device=x.device, dtype=x.dtype)
    counts = torch.zeros(B * npix_out, device=x.device, dtype=x.dtype)

    out.index_add_(0, bins_valid, x_valid)
    counts.index_add_(0, bins_valid, torch.ones_like(bins_valid, dtype=x.dtype))

    nonzero = counts > 0
    out = out / counts.clamp(min=1.0).unsqueeze(-1)

    out = out.view(B, npix_out, D)
    out_mask = ~nonzero.view(B, npix_out)

    return out, out_mask
```

然后 `forward()` 直接返回：

```python
pooled, out_mask = self._healpix_pool_dense(x, mask=mask)
return pooled, out_mask
```

注意：
- 不再返回 `pixel_ids_out` 作为主逻辑必需量
- 因为固定 grid 下 low-res pixel id 与 index 一致

---

### C7. 模型接口兼容性说明
如果 `ht_transformer.py` 当前还写着：

- `cd_emb, cd_pixel_ids_fused = self.cd_compression(...)`
- 然后 `cd_mask = (cd_pixel_ids_fused == -1)`

那么 Stage B 完成后建议同步改成：

```python
cd_emb, cd_mask = self.cd_compression(cd_emb, mask=cd_mask_input)
```

也就是：
- compression 直接返回 `pooled` 和 `mask`
- 不再依赖 `pixel_ids_out`

虽然这不在本清单的 3 个文件里，但 Claude Code 必须在提交说明中标出该联动修改需求。

---

### C8. 验收标准
1. DeepSphere forward 不再需要 `pixel_ids`
2. 主路径中不再调用 local graph builder
3. compression 统一为 dense fixed-grid pooling
4. compression 输出 shape 固定
5. forward 中不再存在 active-token compaction

---

# 五、执行顺序

Claude Code 必须按以下顺序实施：

## Step 1
先修改 `data/dataset.py`，让在线数据路径输出 dense HEALPix 表示

## Step 2
修改 `data/preprocess.py`，让缓存语义与在线数据路径一致，并强制旧缓存失效

## Step 3
修改 `models/components/deepsphere.py`，移除对 active patch 语义的依赖，统一改成 fixed-grid graph + dense pooling

## Step 4
检查并记录与 `models/ht_transformer.py` 的接口联动修改点
即使这份文档不要求直接改该文件，也必须在最终说明中明确指出：
- `cd_encoder`
- `cd_compression`
的返回和调用接口需要同步收口

---

# 六、最终验收清单

只有同时满足以下条件，Stage B 才算真正闭环：

- [ ] 在线 dataset 输出 dense HEALPix CD tensors
- [ ] preprocess cache 输出 dense HEALPix CD tensors
- [ ] 旧 active-patch cache 不会被误用
- [ ] DeepSphere 不再做 runtime local graph construction
- [ ] DeepSphere forward 不再依赖 `pixel_ids`
- [ ] CDCompression 不再做 active-token compaction
- [ ] compression 输出固定 shape
- [ ] 模型主路径的 CD token index 与全局 HEALPix pixel id 一一对应

---

# 七、禁止项

Claude Code 不允许：

1. 仅修改 `deepsphere.py` 而不改 `dataset.py / preprocess.py`
2. 保留 active patch 数据表示，却声称 Stage B 已完成
3. 同时保留 dense-grid 和 active-patch 两套主路径并静默切换
4. 让模型继续依赖 `pixel_ids` 做 runtime remapping
5. 继续把 compression 结果压成 variable-length active token list 作为主输出
6. 省略旧缓存失效检查

---

# 八、交付要求

Claude Code 完成后必须提交：

1. 修改后的三个文件
2. 一份简短变更说明，明确：
   - 新的 CD 数据格式
   - 是否仍保留旧路径
   - `ht_transformer.py` 需要同步修改的接口点
3. 一份最小自检结果：
   - 单 event shape 检查
   - batch collate shape 检查
   - DeepSphere forward shape 检查
   - compression output shape 检查

---

# 九、最短总结

Stage B 是否真正完成，关键不在于 `deepsphere.py` 写得多漂亮，  
而在于：

> **CD 数据是否已经变成 fixed dense HEALPix grid，并让模型端与数据端使用同一套固定图语义。**

这才是闭环。
