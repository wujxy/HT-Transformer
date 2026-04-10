# HT-Transformer v3 升级方案（基于 v2.1）

> 目标：在 **v2.1（HT-Transformer-DS）** 的工程基础上，升级为 **v3：CD Sparse PMT Token + Late Fusion** 主架构。  
> 本文档用于指导 Claude Code 直接实施代码改造。  
> 要求：**不得使用模糊表述，不得做与本方案相反的结构折中，不得继续以 patch/DeepSphere 作为 v3 主路径。**

---

## 1. 结论先行：为什么必须从 v2.1 升级到 v3

v2.1 的主要路线是：

- WP：`WP hit token -> WPProjector -> WPTimeEncoding -> attention`
- CD：`dense HEALPix patch -> CDProjector -> DeepSphereEncoder -> CDCompression -> Fusion`

该路线的出发点是正确的：

1. CD PMT 数过多，不能直接全量做 full attention；
2. fixed dense HEALPix grid 有利于固定图拓扑、减少运行时图构造、改善编译与加速兼容性；
3. WP 采用 token-level time encoding 取代 pairwise time bias，可以保持 SDPA 兼容。

但是，当前实验结果表明：

- 与旧版 CD self-attention 路线相比，**没有出现质的提升**；
- 与 **仅 WP** 相比，也没有体现出 CD 信息的明显增益；
- 68% 角精度约为 2.0 度，中点距离约为 0.9 m，说明 **CD 信息没有真正有效融入最终预测**。

### 1.1 v2.1 的核心问题不是“训练没跑通”，而是“CD 表示路线限制了信息进入模型”

必须明确：

- v2.1 不是坏在优化器、调度器、训练框架；
- v2.1 的根本问题在于：**CD 先被 patch 化，再被图卷积/池化压缩，端点附近的精细位置与时序结构在进入融合前已经被抹平。**

对本任务（有序双端点回归）来说，真正重要的不是“CD 是否被高效压缩”，而是：

- 入射端附近是否保留了足够精细的几何-时序结构；
- 出射端附近是否保留了晚到达结构；
- 模型是否能从 CD 中读到区别于 WP 的补充信息。

v2.1 主路径在这三点上都不够强。

---

## 2. v3 的设计目标

v3 只做三件事：

1. **废除 CD 主路径的 patch 表示。**  
   CD 不再以 dense HEALPix patch 作为主输入。

2. **CD 改为 sparse PMT token 表示。**  
   先在预处理阶段完成 PMT 聚合和 TopK charge 选择，训练时直接读取固定长度 sparse token。

3. **融合改为单次晚融合。**  
   不允许在每一层融合层里继续堆叠大量 WP↔CD attention。必须将跨模态交互压缩为少量、明确、可控的 attention。

---

## 3. v3 的核心设计原则

### 原则 1：CD token 只按 charge 选择，不按时间挑选

这是 v3 的关键约束，必须严格执行。

原因：

- 若同时按 charge / early / late / coverage 进行混合挑选，会引入过强的手工偏置；
- 选择器本身会变得“既要又要”，污染实验解释性；
- 本任务首先需要验证的是：**只要 CD 不再被 patch 化，仅靠高 charge PMT token + time embedding，CD 是否就能带来提升。**

因此，v3 规定：

- **CD token 的保留规则只允许基于 charge；**
- **时间信息只允许通过 token 特征与 time embedding 进入模型；**
- **禁止在 token 选择阶段使用 early/late/covarage 等二级启发式规则。**

### 原则 2：CD 的 PMT 聚合与 TopK 选择必须前移到预处理阶段

原因：

- 这部分计算不属于模型学习，而是确定输入表示；
- 若在训练时在线执行，会增加 CPU 端和 DataLoader 端开销；
- 现有 DS 分支已经有离线预处理到 split HDF5 的框架，应直接复用；
- 将该步骤前移后，训练时只读取定长 `cd_tokens` 与 `cd_mask`，可以显著简化 dataset 逻辑。

因此，v3 规定：

- **CD 的 PMT 聚合、排序、TopK 选择，必须在 `data/preprocess.py` 中离线完成；**
- **训练阶段不得在 `Dataset.__getitem__()` 中再做 CD token 聚合和 TopK 选择。**

### 原则 3：CD 分支必须有自己的 self-attention 编码器，但不能把 CD self-attention 塞回每一层 fusion

原因：

- 如果 CD 不具备 branch 内 self-attention，它仍然只是被动的静态特征源；
- 如果将 CD self-attention 与双向 cross-attention 一起塞进每层 fusion，速度会显著下降；
- 本任务需要的是 **“先各自建模，再小规模融合”**，而不是每层重型交互。

因此，v3 规定：

- **CD 必须有独立的 sparse self-attention encoder；**
- **融合层只允许做单次晚融合，不允许恢复成多层重型双向融合。**

### 原则 4：v3 必须保留 v2.1 的 WP 路径优势

WP 路径已经具备如下优点：

- dual-branch projector；
- token-level time encoding；
- SDPA 兼容；
- 训练稳定、实现清晰。

因此，v3 规定：

- **WP 分支尽量不改主结构；**
- **v3 的重构重点只放在 CD 表示、CD 编码、CD 融合路径。**

---

## 4. v3 最终架构总览

## 4.1 总体结构

v3 总体结构为：

```text
WP raw hits
  -> WP tokenization (保持现有)
  -> WPProjector
  -> WPTimeEncoding
  -> WP Encoder x 2
  -> H_wp

CD raw hits
  -> [预处理阶段] PMT aggregation
  -> [预处理阶段] TopK by charge
  -> [写入预处理HDF5] cd_tokens, cd_mask
  -> [训练阶段读取] CDHitProjector
  -> CDTimeEmbedding
  -> CD Sparse Encoder x 1
  -> CD Latent Compression
  -> Z_cd

Global tokens G0
  -> attend [H_wp ; Z_cd]
  -> G

Query tokens Q0
  -> attend [H_wp ; Z_cd ; G]
  -> Q

Q_1 -> EndpointHead1 -> pred_u1
Q_2 -> EndpointHead2 -> pred_u2
```

### 4.2 与 v2.1 的本质区别

v2.1：

- CD = patch 表示
- CD = DeepSphere 主路径
- CD = 先压缩后融合
- Fusion = 每层做较重的多 attention 交互

v3：

- CD = sparse PMT token 表示
- CD = self-attention 主路径
- CD = 先 branch 内编码，再 latent 压缩
- Fusion = 单次晚融合

### 4.3 一句话定义 v3

**v3 = Sparse CD PMT Token Transformer with Offline Selection and Late Fusion**

---

## 5. v3 数据表示定义

## 5.1 WP 表示（保持 v2.1）

WP 仍使用 hit 级 token：

```text
wp_tokens: [N_wp, 5]
feature order = [ux, uy, uz, q, t]
```

其中：

- `ux, uy, uz`：WP PMT 方向单位向量；
- `q`：归一化电荷；
- `t`：归一化时间。

训练时继续使用 padding mask：

```text
wp_mask: [N_wp]  True = padding/inactive
```

## 5.2 CD 表示（v3 新定义）

CD 不再保存 patch，而是保存 **稀疏 PMT token**。

### 5.2.1 PMT 聚合定义

对单个事件中同一个 CD PMT 的所有命中进行聚合，得到一个 PMT token。

对每个 PMT 计算如下特征：

```text
[ux, uy, uz,
 q_sum,
 q_max,
 n_hits,
 t_first,
 t_mean,
 t_late,
 t_span]
```

定义说明：

- `ux, uy, uz`：该 PMT 的方向单位向量；
- `q_sum`：该 PMT 所有命中的总电荷；
- `q_max`：该 PMT 命中电荷最大值；
- `n_hits`：该 PMT 命中次数；
- `t_first`：最早命中时间；
- `t_mean`：电荷加权平均时间；
- `t_late`：晚到达时间，建议使用裁剪后的晚时刻统计（优先使用高分位时间，如 `t90`/`t95`，若实现复杂则先使用裁剪版 `t_last`）；
- `t_span = t_late - t_first`。

### 5.2.2 TopK 选择规则

只允许按 `q_sum` 排序，取前 `K_cd` 个 PMT token：

```text
score = q_sum
selected = topk(score, K_cd)
```

禁止：

- 用时间参与 token 选择；
- 追加 coverage 补点；
- 混入 early/late 多路 token 预算；
- 在训练时在线重新排序或重选。

### 5.2.3 预处理后写入的 CD 张量

预处理 HDF5 中每个事件必须直接存储：

```text
cd_tokens: [K_cd, 10]
cd_mask:   [K_cd]
```

其中：

- `cd_tokens[i]` 为单个 PMT token 的 10 维特征；
- `cd_mask[i] = False` 表示该位置为有效 token；
- 若某事件有效 PMT 少于 `K_cd`，则末尾补零，并将 `cd_mask=True`。

### 5.2.4 建议默认值

```text
K_cd = 640
feature_dim_cd = 10
```

若速度压力较大，可先用：

```text
K_cd = 512
```

但主实验建议以 `640` 为默认值。

---

## 6. 预处理输出 HDF5 新格式

## 6.1 必须新增的数据集字段

在预处理后的 split HDF5 中，必须新增如下字段：

```text
wp_tokens        float32 [N_events, K_wp, 5]      # 若WP仍在线tokenize可不落盘；若已预处理化则按现有逻辑
wp_mask          bool    [N_events, K_wp]

cd_tokens        float32 [N_events, K_cd, 10]
cd_mask          bool    [N_events, K_cd]

enter_point      float32 [N_events, 3]
exit_point       float32 [N_events, 3]
track_time       float32 [N_events]
```

如果当前 split HDF5 仍保留原始 event-level 索引和统计信息，可继续保留，但 **v3 训练路径不得再依赖 patch 相关字段。**

## 6.2 必须废弃的 CD 主字段

下列字段可以保留用于兼容旧实验，但 **不得作为 v3 主训练路径输入**：

```text
cd_unit_vecs_dense
cd_stats_dense
cd_time_bins_dense
cd_times_mean
cd_patch_mask
```

这些字段属于 patch/DeepSphere 体系，仅用于保留 v2.1 对照试验，不得继续驱动 v3 主架构。

## 6.3 预处理 metadata 必须记录的配置

预处理输出目录下的 metadata 中，必须记录：

```yaml
cd_representation: sparse_pmt_precomputed
cd_topk_score: q_sum
cd_max_tokens: 640
cd_feature_order:
  - ux
  - uy
  - uz
  - q_sum
  - q_max
  - n_hits
  - t_first
  - t_mean
  - t_late
  - t_span
```

这是为了保证训练与预测读取时不会出现字段歧义。

---

## 7. v3 模型模块定义

## 7.1 WPProjector（保留）

WPProjector 继续沿用 v2.1：

- geometry branch: `[ux, uy, uz]`
- qt branch: `[q, t]`
- MLP 融合到 `d_model`

此模块无需结构性重写，只需确保与 v3 其余部分接口兼容。

## 7.2 新增 CDHitProjector

必须新增 `CDHitProjector`，输入为：

```text
cd_tokens[..., 10]
```

建议结构：

- geo branch: `[ux, uy, uz]`
- physics-time branch: `[q_sum, q_max, n_hits, t_first, t_mean, t_late, t_span]`
- 两路 hidden 后 concat
- 经融合 MLP 输出 `d_model`

禁止直接复用 patch 版 `CDProjector`。

原因：

- 旧 `CDProjector` 的输入语义是 patch 统计，不是 PMT token；
- 旧 `CDProjector` 默认带时间直方图分支，不适合当前 v3。

## 7.3 新增 CDTimeEmbedding

必须新增 `CDTimeEmbedding`。

注意：

- 时间已经作为原始 token 特征进入 `CDHitProjector`；
- 仍然必须再提供一个独立 time embedding 分支，以增强时间结构的表达能力。

推荐输入：

```text
[t_first, t_mean, t_late, t_span]
```

输出：

```text
time_emb_cd: [B, K_cd, d_model]
```

并执行：

```text
cd_hidden = cd_projected + cd_time_emb + cd_type_emb + abs_pos_emb
```

## 7.4 CD Sparse Encoder

必须新增 **CDSparseEncoder**，作为 CD 的主干编码器。

要求：

- 只使用 self-attention；
- 起步版本只实现 `1` 层；
- 使用与 WP attention 相同的 SDPA 兼容 attention 内核；
- 保留 PreNorm + residual + FFN 结构；
- 使用 `cd_mask` 屏蔽 padding token。

禁止：

- 将 DeepSphere 继续作为 v3 的 CD 主路径；
- 在 CDSparseEncoder 中引入 patch 邻接图；
- 在 v3 首版中堆叠多层 CD encoder。

默认：

```text
L_cd = 1
```

## 7.5 CD Latent Compression

必须新增 CD latent compression 模块。

目标：

- 将 `K_cd` 个 CD sparse token 压缩为少量 `M_cd` 个 latent tokens；
- 作为融合阶段的 CD memory。

推荐实现：

- learnable latent queries `Z0_cd`；
- 一次 cross-attention：`Q=Z0_cd, K=H_cd, V=H_cd`。

输出：

```text
Z_cd: [B, M_cd, d_model]
```

默认：

```text
M_cd = 64
```

禁止继续使用 `healpix_pool` 作为 v3 主压缩方式。

原因：

- `healpix_pool` 属于 patch 体系；
- v3 的原则是 **先保留 PMT 级结构，再晚压缩**。

## 7.6 Global / Query Late Fusion

必须将当前 v2.1 的多层重型 fusion 改为 **单次晚融合**。

### 7.6.1 Global 聚合

保留 learnable global tokens `G0`：

```text
G = CrossAttn(Q=G0, K=[H_wp ; Z_cd], V=[H_wp ; Z_cd])
```

### 7.6.2 Query 解码

保留 `num_queries = 2`：

```text
Q = CrossAttn(Q=Q0, K=[H_wp ; Z_cd ; G], V=[H_wp ; Z_cd ; G])
```

### 7.6.3 严格禁止的融合行为

v3 首版中，禁止：

- 恢复 `WP <- CD`、`CD <- WP` 的每层双向交互；
- 在 fusion 层中再次加入 CD self-attention；
- 用多层 fusion 堆叠替代 branch encoder。

v3 的逻辑是：

- 各分支先独立编码；
- 融合只在末端做一次；
- 由 global/query 完成跨模态读取。

## 7.7 输出头（保留）

继续使用两个 endpoint head：

```text
pred_u1 = Head1(Q[:, 0])
pred_u2 = Head2(Q[:, 1])
```

输出必须为单位向量。

loss 继续沿用现有 ordered dual-endpoint 体系，首版不要同时改 loss。

---

## 8. v3 训练与推理流程

## 8.1 训练流程

```text
1. 预处理阶段离线生成 split HDF5
   - 读取原始事件
   - WP按现有逻辑处理
   - CD执行 PMT aggregation
   - CD按 q_sum 做 TopK
   - 写入 cd_tokens / cd_mask

2. Dataset 直接读取预处理 HDF5
   - 返回 wp tokens / wp mask
   - 返回 cd tokens / cd mask
   - 返回 enter/exit labels

3. 模型前向
   - WP branch 编码
   - CD branch 编码
   - CD latent compression
   - late fusion
   - dual endpoint heads

4. loss
   - ordered endpoint loss
```

## 8.2 推理流程

推理必须与训练完全一致：

- 使用相同的预处理产物格式；
- 不允许推理阶段再做不同版本的 CD token 选择；
- 若从原始 H5 做在线推理，必须复用与训练相同的 PMT 聚合 + TopK charge 逻辑。

---

## 9. 配置文件修改要求

必须在 `configs/default.yaml` 中加入 v3 相关配置，并保留 v2.1 兼容分支。

建议新增配置如下：

```yaml
model:
  version: "v3"
  cd_representation: "sparse_pmt_precomputed"

  d_model: 128
  num_heads: 4
  d_ff: 512

  wp_self_layers: 2
  cd_self_layers: 1

  num_global_tokens: 8
  num_queries: 2

  cd_max_tokens: 640
  cd_topk_score: "q_sum"
  cd_feature_dim: 10
  cd_latent_tokens: 64

  cd_time_embedding: true
  cd_time_hidden: 32
  cd_time_fourier_dim: 16

  fusion_mode: "late_global_query"

  # 兼容旧版 v2.1 配置，但 v3 主路径不得使用
  cd_deepsphere_layers: 4
  cd_compression: "healpix_pool"
  cd_fusion_tokens: 128
```

### 必须注意

- `cd_deepsphere_layers` / `cd_compression` / `cd_fusion_tokens` 可临时保留以兼容旧 checkpoint 和旧实验；
- 但当 `cd_representation == sparse_pmt_precomputed` 时，这些字段 **不得驱动模型主干**。

---

## 10. 逐文件改造方案（Claude Code 执行指令）

下面是最重要的部分。必须逐文件执行，不允许跳过。

---

# 10.1 `data/preprocess.py`

## 目标

将 CD 的 **PMT 聚合 + TopK charge 选择** 固化到离线预处理阶段，输出 v3 所需 HDF5 字段。

## 必须新增的函数

### 1）`aggregate_cd_hits_by_pmt(...)`

输入：

- 当前事件的 CD hit `copyno / charge / time`
- CD PMT 几何查找表

输出：

- 该事件所有有效 CD PMT 的聚合特征数组：
  - `unit_vecs`
  - `q_sum`
  - `q_max`
  - `n_hits`
  - `t_first`
  - `t_mean`
  - `t_late`
  - `t_span`

要求：

- 按 PMT 聚合，不按 patch 聚合；
- 必须是事件内唯一 PMT 列表；
- 所有特征统一使用与现有管线一致的归一化策略；
- `t_late` 建议优先采用高分位时间；若第一版实现复杂，可先使用裁剪版末次时间，但必须写明实现方式。

### 2）`select_topk_cd_tokens_by_charge(...)`

输入：

- 聚合后的 PMT 特征
- `K_cd`

输出：

- `cd_tokens: [K_cd, 10]`
- `cd_mask: [K_cd]`

要求：

- 排序分数只能使用 `q_sum`；
- 前 `K_cd` 个保留；
- 不足则补零并 mask；
- 不允许引入 early/late/coverage 等额外选择规则。

### 3）`build_v3_cd_tokens(...)`

功能：

- 串联 `aggregate_cd_hits_by_pmt()` 与 `select_topk_cd_tokens_by_charge()`；
- 作为预处理阶段构建 v3 CD 表示的统一入口。

## 必须修改的主预处理逻辑

在事件遍历中：

- 保留 WP 现有处理流程；
- 移除 v3 主路径对 `build_dense_cd_patches()` 的依赖；
- 生成并写入 `cd_tokens` 与 `cd_mask`；
- 在 metadata 中写入 v3 token 配置信息。

## 明确命令

- **不要删除旧 patch 预处理函数；** 仅将其降级为 v2.1 兼容路径；
- **新增 v3 主路径分支；**
- **将 `cd_representation` 写入 metadata；**
- **预处理输出文件结构必须让训练阶段无需再做 CD TopK 选择。**

---

# 10.2 `data/dataset.py`

## 目标

让 Dataset 在 v3 模式下直接读取 **预计算好的 sparse CD token**，而不是在线构造 patch 或在线做 PMT 聚合。

## 必须修改的点

### 1）新增 v3 读取分支

当配置：

```yaml
model.cd_representation: sparse_pmt_precomputed
```

时，`__getitem__()` 必须直接返回：

```python
{
    "wp_tokens": ...,
    "wp_mask": ...,
    "cd_tokens": ...,
    "cd_mask": ...,
    "enter": ...,
    "exit": ...,
    ...
}
```

### 2）禁止在 v3 路径中在线执行以下操作

- `build_dense_cd_patches()`
- patch 级 HEALPix 映射
- CD 的 PMT 聚合
- CD 的 TopK charge 排序
- CD patch time histogram 构造

### 3）保留旋转增强兼容性

当前旋转增强对方向向量和标签进行同步旋转。v3 中必须保证：

- `cd_tokens[..., 0:3]` 的方向单位向量支持旋转增强；
- 与标签 `enter/exit` 的旋转保持一致；
- 由于 token 选择只基于 charge，因此旋转不需要重新选择 token。

### 4）旧字段兼容

v2.1 路径可继续返回：

- `cd_unit_vecs`
- `cd_stats`
- `cd_time_bins`
- `cd_mask`

但 v3 路径 **不得依赖这些字段。**

## 明确命令

- **新增 `sparse_pmt_precomputed` 路径；**
- **不允许在 v3 中在线重建 CD 表示；**
- **保持 v2.1 路径作为 baseline 对照，不要直接删除。**

---

# 10.3 `models/components/token_projectors.py`

## 目标

新增适用于 v3 CD PMT token 的 projector 与时间编码模块。

## 必须新增的类

### 1）`CDHitProjector`

输入：

```text
[B, K_cd, 10]
```

结构建议：

- `geo = MLP([ux, uy, uz])`
- `feat = MLP([q_sum, q_max, n_hits, t_first, t_mean, t_late, t_span])`
- concat 后再过 fusion MLP
- 输出 `[B, K_cd, d_model]`

要求：

- 不要复用 patch projector 的 time-bin 分支；
- 不要让输入维度与 patch 表示混用。

### 2）`CDTimeEmbedding`

输入：

```text
[B, K_cd, 4]
# [t_first, t_mean, t_late, t_span]
```

输出：

```text
[B, K_cd, d_model]
```

要求：

- 形式上与 `WPTimeEncoding` 保持同类设计风格；
- 不得引入 pairwise time bias；
- 必须保持 SDPA 兼容思想，即时间作为 token-level additive embedding。

## 旧模块处理

- 保留 `WPProjector`；
- 保留 `WPTimeEncoding`；
- patch 版 `CDProjector` 可保留给 v2.1，但不得用于 v3 主路径。

---

# 10.4 `models/ht_transformer.py`

## 目标

新增 v3 主模型路径，彻底建立：

- WP branch encoder
- CD sparse branch encoder
- CD latent compression
- late global/query fusion

## 必须新增或重构的模块

### 1）v3 分支选择逻辑

根据配置：

```yaml
model.cd_representation: sparse_pmt_precomputed
```

进入 v3 主路径。

不要让 v3 继续实例化：

- `DeepSphereEncoder`
- `CDCompression(healpix_pool)`

这些模块只能保留给 v2.1 兼容分支。

### 2）新增 `CDSparseEncoderLayer`

结构与 WP attention layer 同型：

- PreNorm
- MultiHeadAttention
- residual
- FFN
- residual

要求：

- 只做 self-attention；
- 使用 `cd_mask`；
- 首版层数固定 `1`。

### 3）新增 `CDLatentCompression`

要求：

- learnable latent tokens
- cross-attend to CD tokens
- 输出 `Z_cd: [B, M_cd, d_model]`

### 4）新增 `LateFusionLayer` 或等价结构

要求：

- 用 global tokens 读取 `[H_wp ; Z_cd]`
- 用 query tokens 读取 `[H_wp ; Z_cd ; G]`
- 保持 query 数为 2

禁止：

- 恢复成 v2.1 的双向 WP↔CD 每层交互模式；
- 在 v3 首版中堆 2 层以上 fusion。

### 5）forward 路径必须重写清楚

v3 前向应显式包含：

```python
wp_tokens -> wp_proj -> wp_time -> wp_encoder -> H_wp
cd_tokens -> cd_proj -> cd_time -> cd_encoder -> H_cd
H_cd -> cd_latent_compression -> Z_cd
G0 -> global_attn([H_wp, Z_cd]) -> G
Q0 -> query_attn([H_wp, Z_cd, G]) -> Q
Q -> endpoint_heads -> pred_u1, pred_u2
```

不要把 v3 路径塞进旧 `HybridFusionLayer` 做隐式兼容。

## 旧模型兼容

- v2.1 路径必须保留，方便做 A/B 测试；
- 但 `HTTransformer` 内部必须明确支持两种 CD 表示：
  - `patch_dense`
  - `sparse_pmt_precomputed`

---

# 10.5 `models/components/deepsphere.py`

## 目标

该文件在 v3 中不再作为主路径。

## 明确命令

- 不要删除，以免破坏 v2.1 baseline；
- 不要继续尝试把它接进 v3；
- 不要对其进行为 v3 服务的复杂追加修改。

v3 的方向不是修补 DeepSphere，而是直接绕开它作为主干。

---

# 10.6 `configs/default.yaml`

## 必须新增 v3 默认配置

请将默认配置扩展为至少支持：

```yaml
model:
  version: "v3"
  cd_representation: "sparse_pmt_precomputed"

  d_model: 128
  num_heads: 4
  d_ff: 512
  dropout: 0.1

  wp_self_layers: 2
  cd_self_layers: 1

  num_global_tokens: 8
  num_queries: 2

  cd_max_tokens: 640
  cd_topk_score: "q_sum"
  cd_feature_dim: 10
  cd_latent_tokens: 64

  cd_time_embedding: true
  cd_time_hidden: 32
  cd_time_fourier_dim: 16

  fusion_mode: "late_global_query"
```

## 必须保留的旧配置

保留以下字段用于 baseline：

- `cd_deepsphere_layers`
- `cd_compression`
- `cd_fusion_tokens`

但要在注释中明确写出：

```yaml
# legacy v2.1 only; ignored when cd_representation == sparse_pmt_precomputed
```

---

# 10.7 `train.py` / `engine/*` / 训练入口

## 目标

保证训练入口正确读取 v3 数据格式，并正确实例化 v3 模型。

## 必须检查的点

1. 日志中输出当前：
   - `model.version`
   - `cd_representation`
   - `cd_max_tokens`
   - `cd_latent_tokens`

2. checkpoint metadata 中写入：
   - `cd_representation`
   - `cd_feature_dim`
   - `cd_topk_score`

3. 不允许在训练时 silent fallback 到旧 patch 路径。

也就是说：

- 如果配置是 `sparse_pmt_precomputed`，但数据里没有 `cd_tokens`，必须直接报错；
- 不允许默默退回到 `build_dense_cd_patches()`。

---

# 10.8 `predict.py` / `evaluation` 相关代码

## 目标

确保推理和评估与训练一致。

## 必须修改

- 当加载 v3 模型时，预测数据输入必须匹配 v3 的 `cd_tokens/cd_mask`；
- 若采用预处理 HDF5 推理，直接读取即可；
- 若从原始 H5 做在线推理，必须复用与预处理相同的 CD PMT 聚合与 TopK charge 逻辑。

不允许出现：

- 训练用 sparse PMT token，推理时又退回 patch；
- 训练和推理对 `t_late` 定义不一致。

---

## 11. 模型默认超参数建议

v3 首版建议：

```yaml
model:
  d_model: 128
  num_heads: 4
  d_ff: 512

  wp_self_layers: 2
  cd_self_layers: 1

  cd_max_tokens: 640
  cd_latent_tokens: 64

  num_global_tokens: 8
  num_queries: 2
```

这是一个平衡版本：

- 比全量 CD self-attention 显著省算力；
- 比 patch 路线更保留细粒度结构；
- 不会因为 fusion attention 过多导致速度崩掉。

若速度压力较大，可将：

```yaml
cd_max_tokens: 512
```

作为首轮 sanity check 配置。

---

## 12. 实施顺序（必须按顺序执行）

### Stage 1：完成数据前移

1. 修改 `data/preprocess.py`
2. 生成包含 `cd_tokens/cd_mask` 的新预处理 HDF5
3. 验证 HDF5 中 token shape、mask、feature 顺序正确

### Stage 2：完成 dataset 读取

1. 修改 `data/dataset.py`
2. 确保 v3 模式不再在线构造 patch
3. 确保旋转增强对 `cd_tokens[..., :3]` 生效

### Stage 3：完成模型主干切换

1. 新增 `CDHitProjector`
2. 新增 `CDTimeEmbedding`
3. 新增 `CDSparseEncoder`
4. 新增 `CDLatentCompression`
5. 新增 `LateFusion`
6. 在 `models/ht_transformer.py` 中接入 v3 forward

### Stage 4：训练闭环验证

1. 跑通 v3 单卡/多卡训练
2. 跑通验证与推理
3. 确认 checkpoint metadata 正确
4. 与 v2.1 做对照实验

---

## 13. 验收标准

Claude Code 完成修改后，必须满足以下验收条件。

### 13.1 数据层验收

- 预处理 HDF5 中存在 `cd_tokens` 与 `cd_mask`；
- `cd_tokens.shape[-1] == 10`；
- `cd_mask.shape[-1] == cd_max_tokens`；
- token 顺序与 metadata 一致；
- 不需要训练时在线执行 CD PMT 聚合和 TopK 选择。

### 13.2 模型层验收

- v3 配置下不再实例化 DeepSphere 主路径；
- v3 前向中存在独立的 CD sparse encoder；
- v3 前向中存在 CD latent compression；
- v3 前向中存在 late global/query fusion；
- 输出头与 loss 路径保持正确。

### 13.3 工程层验收

- v2.1 baseline 路径仍可运行；
- v3 路径可独立运行；
- 配置切换行为清晰且不会 silent fallback；
- 训练、验证、推理三者数据格式一致。

### 13.4 实验层验收

至少完成以下对照：

1. `v2.1 patch_dense baseline`
2. `v3 sparse_pmt_precomputed, K_cd=512`
3. `v3 sparse_pmt_precomputed, K_cd=640`

记录：

- 68% 角精度
- 中点距离
- step time / it/s
- GPU 显存

---

## 14. 不允许做的事

Claude Code 在执行本方案时，禁止以下行为：

1. **不要把 v3 再改回 patch 主路径。**
2. **不要把 DeepSphere 继续包装成 v3 的“局部优化版”。**
3. **不要把 token 选择器扩展成 charge+time+coverage 的混合启发式系统。**
4. **不要在训练时在线执行 CD PMT 聚合和 TopK。**
5. **不要把 CD self-attention 塞进每一层 fusion，导致 attention 数量爆炸。**
6. **不要在 v3 首版同时改 loss、改数据增强、改 scheduler，导致无法归因。**
7. **不要 silent fallback 到旧 patch 流程。**

---

## 15. 本方案的最终定义

本方案定义的 v3，不是对 v2.1 的小修小补，而是一次明确的主路径切换：

- 从 **CD patch / DeepSphere 主路径**
- 切换为 **CD sparse PMT token / self-attention / late fusion 主路径**

其根本目的不是让 CD 更“规整”，而是让 CD **重新保住端点任务真正需要的细粒度信息**。

一句话总结：

> **v3 = 预处理阶段离线生成 CD sparse PMT token，训练阶段直接读 token，经 CD self-attention 编码后压缩成 latent memory，再通过单次 late fusion 送入 query 做双端点回归。**

这就是本项目从 v2.1 升级到 v3 的正式执行方案。

