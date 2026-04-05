# diff_stage_b.md

## 目的

这份文档不是讨论稿，不是建议列表，也不是高层总结。  
这是 **Claude Code 必须执行的 Stage B 差异修复命令清单**。

你必须逐文件完成修复，直到仓库满足以下事实：

1. CD 数据表示是 **fixed dense HEALPix grid**
2. 预处理缓存保存的是 **fixed dense HEALPix grid**
3. DeepSphere 主路径不再依赖 `pixel_ids`
4. DeepSphere 主路径不再做 runtime local remapping
5. CDCompression 主路径不再做 active-token compaction
6. 主模型不再用 `SignedTimeBucketBias` 作为 WP 主路径
7. 主模型必须改用 `WPTimeEncoding`
8. `MultiHeadAttention` 主路径必须是 SDPA-only
9. 旧 active-patch 路径不得继续作为默认训练主路径存在

如果任一条没有完成，就不得宣称 Stage B 已完成。

---

# 总体判断

当前仓库 **没有完成 Stage B**。  
当前仓库的主要问题不是“差一点点”，而是：

- 数据端仍然输出 active patch 变长表示
- 预处理端仍然缓存 active patch 变长表示
- DeepSphere 仍然依赖 `pixel_ids`
- DeepSphere 仍然做 runtime local remapping
- Compression 仍然走 active pooled token 路径
- 主模型仍然在用 `SignedTimeBucketBias`
- `WPTimeEncoding` 虽然存在，但没有接入主模型主路径

你必须修复这些问题。

---

# 文件 1：`data/dataset.py`

## 当前是什么

当前 `dataset.py` 仍然在做以下事情：

1. 用 `np.unique(pixel_ids)` 生成 `unique_pixels`
2. 只为当前事件 active patch 分配：
   - `cd_stats`
   - `cd_time_bins`
   - `cd_patch_unit`
3. 返回：
   - `cd_pixel_ids`
   - 长度为 `n_patches` 的变长 CD token
4. `collate_fn` 仍然按当前 batch 的 `max_cd` 做 padding

这说明当前数据表示仍然是：

> **active patch 变长表示**

这与 Stage B 目标完全不一致。

---

## 应该变成什么

你必须把 `dataset.py` 改成：

> **固定 dense HEALPix grid 表示**

对于给定 `nside`：

- `npix = 12 * nside^2`

每个事件的 CD 数据必须固定输出：

- `cd_unit_vecs`: `(npix, 3)`
- `cd_stats`: `(npix, 4)`
- `cd_time_bins`: `(npix, num_time_bins)`
- `cd_mask`: `(npix,)`

语义要求：

- token index == global HEALPix pixel id
- active pixel:
  - feature 非零
  - `cd_mask[p] = False`
- inactive pixel:
  - feature 全零
  - `cd_mask[p] = True`

---

## 你必须修改什么

### 1. 删除 active patch 主路径
你必须删除或停用如下主逻辑：

- `unique_pixels`
- `inverse`
- `n_patches = len(unique_pixels)`
- 基于 active patch 数量构造 CD token

你不得再用 active patch 列表作为主输出。

### 2. 新增 dense CD 构造函数
你必须新增一个 dense HEALPix patch 构造函数，名字可自定，但必须具备如下功能：

- 输入：
  - hit-level `pixel_ids`
  - `charges`
  - `times`
  - 全局 `pixel_center_vecs`
  - `npix`
  - `num_time_bins`
  - `t_max`
- 输出：
  - `cd_unit_vecs_dense`
  - `cd_stats_dense`
  - `cd_time_bins_dense`
  - `cd_mask_dense`

### 3. 每个事件固定输出 `npix` 长度
你必须保证 `__getitem__()` 返回的 CD shape 与事件无关：

- 所有事件 shape 完全一致
- 不能再返回长度随事件变化的 CD token

### 4. `collate_fn` 必须同步改
由于 CD 已改成 fixed shape：

- `cd_unit_vecs`
- `cd_stats`
- `cd_time_bins`
- `cd_mask`

都必须直接 `stack`

你不得继续对 CD 做“按 batch 最大 patch 数 padding”的逻辑。

### 5. `cd_pixel_ids` 不得继续作为主模型必需输入
如果为了兼容你暂时保留 `cd_pixel_ids` 字段，可以保留，但必须满足：

- 模型主路径不依赖它
- 代码注释中明确写明 `deprecated`
- 不得再把它作为 DeepSphere 主路径输入

---

## 参考实现要求

你必须使用类似如下形式的 dense 构造函数：

```python
def build_dense_cd_patches(
    pixel_ids,
    cd_times,
    cd_charges,
    pixel_center_vecs,
    npix,
    num_time_bins,
    t_max,
):
    cd_unit_vecs_dense = pixel_center_vecs.astype(np.float32).copy()
    cd_stats_dense = np.zeros((npix, 4), dtype=np.float32)
    cd_time_bins_dense = np.zeros((npix, num_time_bins), dtype=np.float32)
    cd_mask_dense = np.ones((npix,), dtype=bool)

    if len(pixel_ids) == 0:
        return cd_unit_vecs_dense, cd_stats_dense, cd_time_bins_dense, cd_mask_dense

    bin_edges = np.linspace(0, t_max, num_time_bins + 1)

    for pix in np.unique(pixel_ids):
        m = (pixel_ids == pix)
        q = cd_charges[m]
        t = cd_times[m]

        cd_stats_dense[pix, 0] = q.sum()
        cd_stats_dense[pix, 1] = len(q)
        cd_stats_dense[pix, 2] = t.min() if len(t) > 0 else 0.0
        cd_stats_dense[pix, 3] = (q * t).sum() / (q.sum() + 1e-10)

        if len(t) > 0:
            bin_idx = np.clip(np.digitize(t, bin_edges) - 1, 0, num_time_bins - 1)
            np.add.at(cd_time_bins_dense[pix], bin_idx, q)

        cd_mask_dense[pix] = False

    return cd_unit_vecs_dense, cd_stats_dense, cd_time_bins_dense, cd_mask_dense
```

你可以优化实现方式，但输出语义必须完全一致。

---

## 为什么还没完成

因为当前代码仍然把 CD token index 当成“当前事件 active patch 序列下标”，不是“全局 HEALPix pixel id”。  
只要这一点不改，Stage B 就没有任何成立基础。

---

## 验收标准

你必须满足以下检查：

- 任意两个事件的 `cd_stats.shape[0]` 必须相同，且等于 `npix`
- `collate_fn` 不再为 CD 做变长 padding
- inactive pixel feature 全零
- `cd_mask` 对 inactive pixel 为 True
- 模型前向不再需要依赖 `cd_pixel_ids`

---

# 文件 2：`data/preprocess.py`

## 当前是什么

当前 `preprocess.py` 仍然只是把 dataset 输出直接缓存成 `.pt` 文件。  
由于 `dataset.py` 当前仍是 active-patch 变长表示，所以你现在的缓存语义也仍然是：

> **active patch 变长缓存**

同时，当前 manifest 没有强制记录 Stage B 所需的表示类型标识，也没有对旧缓存做强制失效检查。

---

## 应该变成什么

预处理缓存必须与新的 dense dataset 输出完全一致。

也就是说，缓存中的每个 event 必须保存：

- `cd_unit_vecs`: `(npix, 3)`
- `cd_stats`: `(npix, 4)`
- `cd_time_bins`: `(npix, num_time_bins)`
- `cd_mask`: `(npix,)`

缓存语义必须明确标记为：

> `cd_representation = "dense_healpix"`

---

## 你必须修改什么

### 1. 预处理输出必须跟随新的 dense dataset
你不得继续缓存 active patch 事件字典。  
你必须确保预处理阶段保存的是 dense CD 表示。

### 2. manifest 必须增加强约束字段
你必须在 manifest 中增加至少以下字段：

```json
{
  "cd_representation": "dense_healpix",
  "nside": "...",
  "npix": "...",
  "num_time_bins": "..."
}
```

### 3. 读取缓存前必须检查表示类型
你必须在使用 preprocessed cache 之前检查：

- `manifest["cd_representation"] == "dense_healpix"`

如果不满足，必须直接报错，禁止静默继续训练。

### 4. 旧缓存必须失效
当前逻辑如果看到已有 `manifest.json` 就直接退出，这不允许继续保持。  
你必须改成：

- 如果已有 manifest 且不是 dense_healpix，报错并要求重新预处理
- 不能继续把旧缓存当成新 Stage B 缓存使用

### 5. config hash 必须包含 representation 信息
你必须把 `cd_representation` 纳入 config hash 或 manifest 校验条件，否则旧缓存仍可能被误用。

---

## 参考实现要求

### manifest 示例
```python
manifest = {
    "original_events": total_events,
    "expand_times": expand_times,
    "total_events": total_events * (1 + expand_times),
    "batch_size": batch_size,
    "num_files": batch_idx,
    "events_per_file": events_per_batch,
    "config_hash": _config_hash(config),
    "cd_representation": "dense_healpix",
    "nside": config["data"]["nside"],
    "npix": 12 * config["data"]["nside"] * config["data"]["nside"],
    "num_time_bins": config["data"]["num_time_bins"],
}
```

### 强制失效检查示例
```python
if os.path.exists(manifest_path):
    with open(manifest_path, "r") as f:
        old_manifest = json.load(f)

    if old_manifest.get("cd_representation") != "dense_healpix":
        raise RuntimeError(
            "Existing preprocessed cache is old active-patch format. "
            "Delete preprocessed directory and re-run preprocess for dense_healpix Stage B."
        )
```

你可以改写实现，但语义必须一致。

---

## 为什么还没完成

因为当前缓存系统没有显式地区分：
- 旧 active-patch 缓存
- 新 dense-healpix 缓存

只要这一点不解决，就一定存在“旧缓存被新模型误用”的风险。

---

## 验收标准

你必须满足以下检查：

- 新生成 manifest 中有 `cd_representation: dense_healpix`
- 旧缓存不会被 Stage B 主路径静默读取
- 预处理后的 CD shape 固定
- 预处理缓存和在线 dataset 输出语义一致

---

# 文件 3：`models/components/deepsphere.py`

## 当前是什么

当前 `deepsphere.py` 仍然有以下旧路径特征：

1. 仍然有 `build_local_neighbor_indices(pixel_ids, full_knn_adj, ...)`
2. `DeepSphereBlock.forward()` 仍然接收 `pixel_ids`
3. `DeepSphereEncoder.forward()` 仍然接收 `pixel_ids` 和 `full_knn_adj`
4. encoder 仍然在构造局部 token graph
5. `CDCompression` 仍然依赖 `pixel_ids`
6. `CDCompression` 仍然保留 active pooled token 路径

这说明当前 DeepSphere 仍然是：

> **active-patch + runtime remap 语义**

不是 Stage B 目标。

---

## 应该变成什么

Stage B 完成后，DeepSphere 必须运行在：

- fixed dense HEALPix grid
- fixed global knn graph
- fixed output pooling grid

也就是说：

### DeepSphereEncoder
必须改成：

```python
def forward(self, x, knn_adj, mask=None):
    ...
```

不再需要：
- `pixel_ids`
- `full_knn_adj`
- `build_local_neighbor_indices`

### DeepSphereBlock
必须直接按全局 fixed 邻接聚合：

- token index == pixel id
- 所有样本共享同一张图

### CDCompression
必须直接在 dense fixed-grid 上做 pooling：

- 输入固定 `(B, npix_in, D)`
- 输出固定 `(B, npix_out, D)`
- 输出固定 `(B, npix_out)` mask
- 不再 active-token compaction
- 不再依赖 `pixel_ids`

---

## 你必须修改什么

### 1. 删除主路径里的 `build_local_neighbor_indices`
你必须停止在主路径中调用这个函数。  
如果为了兼容暂时保留定义，可以保留，但必须：

- 标注 `deprecated`
- 不允许主模型再调用它

更推荐直接删除。

### 2. 修改 `DeepSphereBlock.forward`
你必须把 `DeepSphereBlock.forward()` 改成只接：

- `x`
- `knn_adj`
- `mask`

例如：

```python
def forward(self, x, knn_adj, mask=None):
    ...
```

### 3. 修改 `DeepSphereEncoder.forward`
你必须把 `DeepSphereEncoder.forward()` 改成：

```python
def forward(self, x, knn_adj, mask=None):
    for block in self.blocks:
        x = block(x, knn_adj=knn_adj, mask=mask)
    return x
```

### 4. 改写邻居聚合为 fixed-grid gather
你必须直接使用固定 `knn_adj` 做 gather，而不是再做 global->local remap。

### 5. `CDCompression` 主路径必须改成 dense pooling
你必须让 compression 主路径只走 dense fixed-grid pooling。

### 6. compression 不得继续依赖 `pixel_ids`
Stage B 完成后：
- low-res output token index 自身就是 low-res pixel id
- 不需要 `pixel_ids_out` 作为主路径语义

### 7. compression 输出必须改成 `(pooled, out_mask)`
你不得再把 `pixel_ids_out` 作为模型主路径必要输出。  
主路径统一为：

```python
pooled, out_mask = self.cd_compression(x, mask=mask)
```

---

## 参考实现要求

### DeepSphere 聚合示例
```python
def _aggregate_neighbors_fixed(self, x, knn_adj, mask=None):
    B, N, D = x.shape
    k = knn_adj.shape[1]

    idx = knn_adj.unsqueeze(0).expand(B, -1, -1)
    gather_idx = idx.unsqueeze(-1).expand(B, N, k, D)
    x_expand = x.unsqueeze(2).expand(B, N, k, D)
    nbr = torch.gather(x_expand, dim=1, index=gather_idx)

    if mask is not None:
        nbr_mask = mask.gather(1, idx.reshape(B, -1)).view(B, N, k)
        valid = (~nbr_mask).to(x.dtype)
        denom = valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
        nbr = nbr * valid.unsqueeze(-1)
        return nbr.sum(dim=2) / denom
    else:
        return nbr.mean(dim=2)
```

### DeepSphereBlock 示例
```python
def forward(self, x, knn_adj, mask=None):
    x_norm = self.norm(x)
    neighbor_agg = self._aggregate_neighbors_fixed(x_norm, knn_adj, mask=mask)
    neighbor_feat = self.neighbor_proj(neighbor_agg)
    self_feat = self.self_proj(x_norm)
    combined = neighbor_feat + self_feat
    return x + self.mlp(combined)
```

### DeepSphereEncoder 示例
```python
def forward(self, x, knn_adj, mask=None):
    for block in self.blocks:
        x = block(x, knn_adj=knn_adj, mask=mask)
    return x
```

### Dense compression 示例
```python
def _healpix_pool_dense(self, x, mask=None):
    B, N, D = x.shape
    npix_out = int(self.npix_out.item())

    if mask is None:
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)

    high_to_low = self.high_to_low.view(1, N).expand(B, -1)
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

然后 `forward()` 主路径必须改成：

```python
def forward(self, x, mask=None):
    return self._healpix_pool_dense(x, mask=mask)
```

---

## 为什么还没完成

因为当前 `deepsphere.py` 仍然在用 active-patch 思维工作。  
只要还需要 `pixel_ids` 做 runtime remap，Stage B 就没有完成。

---

## 验收标准

你必须满足以下检查：

- `DeepSphereEncoder.forward()` 不再接收 `pixel_ids`
- 主路径不再调用 `build_local_neighbor_indices`
- `DeepSphereBlock` 直接用 fixed `knn_adj`
- `CDCompression` 主路径不再接收 `pixel_ids`
- `CDCompression` 主路径输出固定 shape
- 主路径中不再存在 active pooled token compaction

---

# 额外联动修改命令（必须执行）

虽然本清单只要求这 3 个文件，但你必须在提交说明中明确指出以下联动修改是 **必须同步完成** 的：

## `models/ht_transformer.py`
你必须同步修改：

1. `cd_encoder` 调用接口  
   从：
   ```python
   self.cd_encoder(cd_emb, pixel_ids=..., full_knn_adj=..., ...)
   ```
   改成：
   ```python
   self.cd_encoder(cd_emb, knn_adj=self.cd_knn_adj, mask=cd_mask_input)
   ```

2. `cd_compression` 调用接口  
   从：
   ```python
   cd_emb, cd_pixel_ids_fused = self.cd_compression(...)
   cd_mask = (cd_pixel_ids_fused == -1)
   ```
   改成：
   ```python
   cd_emb, cd_mask = self.cd_compression(cd_emb, mask=cd_mask_input)
   ```

如果你不做这两项联动修改，Stage B 仍然不成立。

---

# 最终验收清单

只有以下全部成立，才允许宣称 Stage B 已完成：

- [ ] `data/dataset.py` 输出 dense HEALPix CD tensors
- [ ] `data/preprocess.py` 缓存 dense HEALPix CD tensors
- [ ] 旧 active-patch cache 不会被误用
- [ ] `models/components/deepsphere.py` 主路径不再依赖 `pixel_ids`
- [ ] DeepSphere 主路径不再做 runtime local remap
- [ ] `CDCompression` 主路径不再做 active-token compaction
- [ ] `CDCompression` 输出 fixed dense low-res grid
- [ ] `models/ht_transformer.py` 已同步接到新的 encoder/compression 接口
- [ ] CD token index 与全局 HEALPix pixel id 一一对应

---

# 禁止项

你不得：

1. 只改 `deepsphere.py` 而不改 `dataset.py`
2. 只改 `dataset.py` 而不改 `preprocess.py`
3. 保留旧 active-patch 主路径并把它藏在条件分支里继续默认使用
4. 保留 `pixel_ids` 作为 DeepSphere 主路径必要输入
5. 保留 active pooled token compaction 作为 compression 默认路径
6. 省略旧缓存失效检查
7. 用“兼容老逻辑”作为理由跳过 Stage B 主路径替换

---

# 最终命令

Claude Code，按以下顺序执行：

1. 先重写 `data/dataset.py`，把 CD 输出改成 fixed dense HEALPix grid
2. 再重写 `data/preprocess.py`，让缓存语义与 dense-grid 完全一致，并强制旧缓存失效
3. 再重写 `models/components/deepsphere.py`，删除 active-patch 主路径，统一改为 fixed-grid graph + dense pooling
4. 再同步修改 `models/ht_transformer.py` 的 encoder/compression 调用接口
5. 再做 shape 验证、缓存验证、单 batch forward 验证
6. 在全部通过前，不允许说 Stage B 已完成
