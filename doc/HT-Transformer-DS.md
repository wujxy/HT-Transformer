# HT-Transformer v2 详细实施方案
## —— 基于 DeepSphere 的 CD 分支重构与 WP 时间感知增强

---

## 1. 文档目标

本文档给出 **HT-Transformer v2** 的完整实施方案，用于指导在当前 HT-Transformer 源码基础上进行结构升级。该版本的核心目标是：

1. **保留整体 Hybrid Token Transformer 框架**；
2. 将 **CD 分支的 self-attention 整体替换为 DeepSphere block**，从根本上解除当前 CD 分支的主要计算瓶颈；
3. 将 **WP 分支升级为更强的 token encoder**，并为 WP self-attention 显式注入 **时间结构意识**；
4. 保留当前有序双端点回归任务定义、输出头设计、损失设计和整体训练/预测/评估闭环；
5. 引入 **更智能的学习率调度与早停策略**，解决当前训练在约 100 epoch 左右开始过拟合的问题；
6. 保持与现有项目目录结构、运行入口、预处理流程、预测/评估脚本尽可能兼容。

本文档面向实施，不是概念讨论文档。重点是：**哪些模块要改、怎么改、输入输出是什么、替换关系是什么、训练流程怎么更新、怎样保证性能不掉。**

---

## 2. v2 的最终决策

本版本固定采用如下决策，不再保留备选分支：

### 2.1 CD 分支决策

采用：

> **路线 B：将 CD self-attn 整体替换成 DeepSphere block，但保留整体 Hybrid Token 框架。**

具体含义：

- 保留当前 **CD patch token 构造方式**；
- 保留当前 **CDProjector** 的总体思路；
- 删除/替换当前 `HybridEncoderLayer` 中的 `cd_self_attn + cd_rpe + cd_knn_mask` 逻辑；
- 用 `DeepSphereBlock` / `DeepSphereEncoder` 承担 CD 局部球面建模；
- 保留 WP↔CD 融合、Global token、Query token、双端点输出头。

### 2.2 WP 分支决策

WP 分支采用两级增强：

1. **先把 WP token encoder 做强**；
2. **给 WP self-attention 显式加入时间结构意识**。

在“时间结构意识”的几种实现中，固定采用：

> **为 WP self-attention 引入 signed relative time bias（带符号相对时间偏置）**。

不采用“仅额外追加时间特征”作为主方案。

### 2.3 旋转增强决策

保留现有设计目标：

> **每个 event 自己随机采一个旋转副本。**

该行为是符合实验需求的，不需要改成固定旋转集合。

### 2.4 学习率调度决策

训练侧不再使用固定的 “warmup + 全程 cosine 衰减” 作为唯一策略，而采用：

> **Warmup + ReduceLROnPlateau + EarlyStopping**

其中：

- Warmup 负责训练初期稳定；
- ReduceLROnPlateau 根据验证集表现自动降学习率；
- EarlyStopping 防止在后期继续过拟合。

---

## 3. 为什么必须升级到 v2

### 3.1 当前 CD 分支的根本瓶颈

当前源码中的 CD 分支虽然在语义上采用了 “kNN local attention”，但其实现仍然保留了以下高成本步骤：

1. 先构造完整的 `QK^T` attention 分数矩阵；
2. 先构造完整的 CD-RPE 两两角距离矩阵；
3. 先构造完整的 CD-RPE 两两时间差矩阵；
4. 最后再通过 mask 屏蔽掉非邻居项。

这意味着当前 CD self-attention 的计算和显存消耗仍然近似 **二次复杂度**，而不是理想中的稀疏局部复杂度。

结论：

- 只要 `nside` 增大，CD token 数增加；
- 当前实现的代价会迅速上升；
- 因而当前版本不适合在不改 CD 算子的前提下继续做更细粒度球面建模。

### 3.2 为什么 DeepSphere 是正确方向

对于 CD 分支，DeepSphere / HEALPix 图卷积类方法有三个优势：

1. **天然适合球面局部邻域建模**；
2. **计算复杂度依赖局部邻域而不是全局两两关系**；
3. **更适合在更大 `nside` 下扩展**。

这非常符合 CD patch token 的物理与几何属性：

- 输入已是球面 patch；
- 需要保留局部拓扑；
- 又不应继续承担全局 full attention 的代价。

### 3.3 为什么 WP 还要增强

当前 WP token encoder 仅为：

```python
Linear([ux, uy, uz, q, t] -> d_model)
```

这太浅，存在两个问题：

1. WP token 的单点语义提炼不足；
2. 时间信息只停留在 token 级输入，没有真正进入 token-token 关系建模。

因此需要同时做两件事：

- 增强单 token 表征能力；
- 让 WP self-attention 显式感知相对时间结构。

---

## 4. v2 总体架构

v2 的总体流程如下：

```text
Raw PMT hits
  ├─ WP hits  ──> WPProjectorV2 ───────────────┐
  └─ CD hits  ──> HEALPix patching ─> CDProjector ─> DeepSphereEncoder ─> CDCompression ─┐
                                                                                          │
                                                          Global tokens / Query tokens ───┤
                                                                                          ▼
                                       Hybrid Fusion Encoder (WP self-attn + WP↔CD + Global + Query)
                                                                                          ▼
                                                    Ordered Dual Endpoint Heads (u1, u2)
                                                                                          ▼
                                                               Joint Loss / Metrics / Eval
```

相较于 v1，最大的变化是：

- **CD 不再在每层 encoder 中做 self-attention**；
- **CD 先独立经过 DeepSphereEncoder 编码，再压缩后进入融合阶段**；
- **WP self-attention 保留，但加入相对时间偏置**；
- **整体输出端、损失端、训练入口保持最大兼容性。**

---

## 5. 模块级实施方案

---

## 5.1 `DataLoader.py`：输入数据组织保持兼容，补充必要字段

### 目标

保持现有 H5 读取、CD/WP 分离、CD patch 聚合逻辑不变，尽量避免重写预处理链路。

### 保留内容

以下逻辑保持不变：

- H5 文件发现与读取；
- `copyno/charge/time` 读取；
- `enter/exit` 真值读取；
- WP/CD 分离；
- WP token 原始特征构造；
- CD patch aggregation；
- `cd_stats`、`cd_time_bins`、`cd_unit_vecs` 的构造；
- `u1/u2/p1/p2` 标签构造。

### 需要补充/调整的内容

#### （1）保留 CD patch 的原始 pixel_id

当前建议在数据项中新增：

```python
'cd_pixel_ids': torch.LongTensor  # (N_cd_patch,)
```

用途：

- 供 DeepSphereEncoder 或后续压缩模块追踪 patch 所属像素；
- 便于未来做多层级 HEALPix pooling；
- 便于调试和可视化。

#### （2）为 WP 相对时间偏置保留原始/规范化时间

当前已有：

```python
'wp_times': torch.FloatTensor
```

这部分保留即可，不需要额外改变数据格式。

#### （3）CD 邻接表后续不再供 self-attn 使用

当前 `cd_knn_adj` 主要是给 CD self-attn 构造 mask 用的。v2 中 CD self-attn 被替换掉后：

- 旧的 `cd_knn_adj` 可以不再作为主训练路径必需输入；
- 但如果 DeepSphereEncoder 仍需邻域图，可改为在初始化时构建 **HEALPix 邻接表** 或 **父子层级映射表**；
- 不再走 “attention mask” 语义，而改为 “graph / spherical conv adjacency” 语义。

### 是否需要重写 Preprocess

不需要整体重写。

预处理仍然可以：

- 先用现有 `H5EndpointDataset` 做 tokenization；
- 再把结果保存为 `.pt`；
- v2 只需确保新增字段（如 `cd_pixel_ids`）也被一并保存。

---

## 5.2 `TokenProjector.py`：升级 WPProjector，保留并复用 CDProjector

### 5.2.1 CDProjector

#### 决策

**总体保留当前 CDProjector。**

理由：

当前 CDProjector 已具备：

- 几何位置 `cd_unit_vecs`
- 一级统计量 `sumQ/count/t_min/t_mean`
- 二级时间序列 `time_bins`
- 通过 Conv1d 对 time-bin 做编码

这是一个合理且信息量充足的 patch projector，不是当前瓶颈来源。

#### 只做小改动

建议新增一个可配置选项：

```yaml
model:
  cd_projector_dropout: 0.0~0.1
```

并允许在 `stats_proj` / `fusion` 中插入轻量 dropout 或 RMSNorm，但不作为第一优先级。

---

### 5.2.2 新增 `WPProjectorV2`

#### 目标

将当前单层线性投影替换为 **双支路特征提炼 + 融合**。

#### 推荐结构

输入仍然为：

```python
[ux, uy, uz, q, t]
```

但编码方式改为：

```text
geometry branch: [ux, uy, uz] -> Linear -> GELU -> Linear
optical-time branch: [q, t]   -> Linear -> GELU -> Linear
concat -> fusion Linear -> GELU -> Linear(d_model)
```

#### 推荐实现

```python
class WPProjectorV2(nn.Module):
    def __init__(self, d_model, d_geo=32, d_qt=32, hidden=64, dropout=0.1):
        ...
```

#### 说明

这样做优于单层线性层的原因：

1. 几何特征和光学时间特征先各自提炼；
2. token 初始表征更强；
3. 为后续 self-attention 提供更高质量输入。

#### 替换关系

- 删除/停用 `WPProjector`
- 新增 `WPProjectorV2`
- `Model.py` 中改为实例化 `WPProjectorV2`

---

## 5.3 新增 `DeepSphere.py`：CD 分支主干模块

建议新建文件：

```text
python/DeepSphere.py
```

### 目标

为 CD 分支提供球面局部建模模块，替换掉当前基于 full attention 的 `cd_self_attn`。

### 建议包含的类

#### （1）`DeepSphereBlock`

职责：

- 对输入 patch embeddings 做一层球面局部特征聚合；
- 支持残差、归一化、GELU；
- 不做全局两两 attention。

#### （2）`DeepSphereEncoder`

职责：

- 堆叠多个 `DeepSphereBlock`；
- 输出增强后的 CD patch embeddings；
- 可选支持多层级 pooling。

#### （3）`CDCompression`

职责：

- 将较细粒度 CD patch embeddings 压缩到适合与 WP 做融合的 token 数；
- 避免把全部高分辨率 CD token 直接送入 dense cross-attention。

---

### 5.3.1 `DeepSphereBlock` 推荐接口

```python
class DeepSphereBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1, norm_type='rmsnorm')
    def forward(self, x, pixel_ids=None, batch_meta=None):
        # x: (B, N_cd, D)
        return x_out
```

### 5.3.2 关于归一化

此处优先推荐：

> **RMSNorm 用于 DeepSphereBlock 内部归一化。**

原因：

- 结构更简单；
- 往往对更深堆叠更稳；
- 比优先替换位置编码更有意义。

### 5.3.3 关于层数

建议起步：

```yaml
model:
  cd_deepsphere_layers: 3~4
  cd_deepsphere_hidden: 2*d_model
```

不要一开始过深。

### 5.3.4 `CDCompression` 的必要性

这一模块必须有。

#### 原因

DeepSphere 解决的是 **CD 内部建模的复杂度**。

但如果编码后仍把所有 CD token 原样送去：

- WP→CD cross-attn
- CD→WP cross-attn
- Global→All
- Query→All

则融合阶段仍会很重。

#### 推荐做法

以下三种方式任选其一作为起步：

1. **HEALPix 层级 pooling**（推荐）
2. top-k score pooling
3. mean pooling 到固定数量 token

首选：

> **HEALPix 层级 pooling**

即在较细 patch 上编码，在较粗层级上输出融合 token。

#### 建议输出 token 数

建议把进入融合阶段的 CD token 数控制在：

```text
~64 到 ~192
```

具体依赖 batch size、显存和 WP token 数。

---

## 5.4 `Model.py`：重构整体模型主干

### 目标

在保留整体 Hybrid Token 框架的前提下，完成以下替换：

1. 替换 WP projector；
2. 将 CD self-attn 从融合层中抽离；
3. 在融合前加入独立的 DeepSphereEncoder + CDCompression；
4. 给 WP self-attention 加入 relative time bias。

---

### 5.4.1 推荐的模型分层

将整体模型拆成三段：

#### 第一段：Input projection

- `WPProjectorV2`
- `CDProjector`
- type embedding
- absolute position encoding

#### 第二段：Modality-specific encoding

- WP 不在这里做额外预编码（可选）
- CD 通过 `DeepSphereEncoder`
- 然后 `CDCompression`

#### 第三段：Hybrid fusion encoder

只保留以下交互：

- WP self-attn
- WP↔CD cross-attn
- Global↔All
- Query↔All
- FFN

此时不再存在单独的 `cd_self_attn`。

---

### 5.4.2 重写 `HybridEncoderLayer`

建议将当前 `HybridEncoderLayer` 改写为新的融合层，例如：

```python
class HybridFusionLayer(nn.Module):
    ...
```

#### 旧版本包含

- WP self-attn
- CD self-attn
- WP→CD
- CD→WP
- Global→All
- Query→All

#### 新版本包含

- WP self-attn（带 relative time bias）
- WP→CD cross-attn
- CD→WP cross-attn
- Global→All
- Query→All
- FFN

即：

> **删除 CD self-attn 模块、CD-RPE 模块、CD-kNN-attn-mask 模块。**

---

### 5.4.3 给 WP self-attention 加 signed relative time bias

#### 这是 WP 时间结构意识的最终方案

建议新增一个模块，例如：

```text
python/WPTimeBias.py
```

或写入 `PositionEncoding.py`。

#### 推荐类名

```python
class SignedTimeBucketBias(nn.Module):
    ...
```

#### 输入

- `wp_times`: `(B, N_wp)`

#### 输出

- `time_bias`: `(B, H, N_wp, N_wp)`

#### 为什么要 signed

因为：

- `abs(dt)` 只表达接近程度；
- `signed dt` 还能表达先后顺序；
- 对端点方向任务，早到 hit 往往更有信息。

#### 如何接入

WP self-attention 建议新增参数：

```python
wp_time_bias = self.wp_time_bias(batch['wp_times'])
```

再以 `attn_bias` 形式加到 WP self-attention 的 logits 上。

如果你不想重构 SDPA 太多，可以：

- 为 WP self-attn 单独实现一个支持 bias 的 attention；
- 或在 PyTorch 版本允许的情况下把 bias 转为 additive mask 形式。

#### 注意

WP 的 relative time bias 和 CD 原来的 RPE 不是一回事：

- CD-RPE 将被删除；
- WP relative time bias 将新增；
- 这是 v2 唯一主保留的 attention bias。

---

### 5.4.4 全模型 forward 流程（v2）

推荐顺序：

```text
1. WP raw tokens -> WPProjectorV2 -> wp_emb
2. CD patch tokens -> CDProjector -> cd_emb
3. add token type embedding
4. add absolute position encoding
5. cd_emb -> DeepSphereEncoder -> cd_highres
6. cd_highres -> CDCompression -> cd_fused
7. 初始化 global/query tokens
8. 多层 HybridFusionLayer:
   - WP self-attn + signed time bias
   - WP<->CD cross-attn
   - Global<->All
   - Query<->All
   - FFN
9. query outputs + global pooled context
10. ordered dual endpoint heads -> pred_u1, pred_u2
```

---

## 5.5 `PositionEncoding.py`：保留绝对 PE，新增 WP 时间偏置模块

### 目标

不推翻当前位置编码体系，只做最小必要升级。

### 保留

- `FourierPositionEncoding` 保留；
- 继续用于 `wp_unit_vecs` 和 `cd_unit_vecs`。

### 删除/停用

- `BucketRPE` 不再用于 CD self-attention 主路径。

### 新增

建议新增：

```python
class SignedTimeBucketBias(nn.Module):
    ...
```

#### 参数建议

```yaml
model:
  wp_num_time_buckets: 32 or 64
  wp_time_bias_heads_shared: false
```

#### bucket 化建议

可采用：

- 对称 signed bucket
- 中间桶表示 `dt≈0`
- 左右两侧分别表示 “query 更早 / key 更早”

---

## 5.6 `LossFunction.py`：先保持不变

### 决策

v2 第一阶段 **不修改损失函数形式**。

保留当前三项联合损失：

1. `L_ang`
2. `L_len`
3. `L_dir`

### 理由

因为 v2 当前的主要变化在：

- CD 建模算子
- WP 编码与时间感知
- 训练调度

如果同时再改 loss，会导致 A/B 对照不清楚。

### 可选预留

仅在配置中允许后续调整权重：

```yaml
loss:
  lambda_ang: 1.0
  lambda_len: 0.5
  lambda_dir: 0.25
```

训练稳定后可再做权重消融。

---

## 5.7 `ModelTrain.py`：学习率调度升级为 Warmup + Plateau + EarlyStopping

这是 v2 训练策略升级的重点模块之一。

### 目标

解决当前训练：

- 前期可正常收敛；
- 中后期约 100 epoch 附近开始过拟合；
- 固定 cosine warmup 对验证反馈无响应的问题。

---

### 5.7.1 调度器总体策略

采用三阶段：

#### 阶段 A：Warmup

- 用线性 warmup；
- 持续前若干 epoch 或若干 step。

#### 阶段 B：ReduceLROnPlateau

- warmup 结束后转入 plateau scheduler；
- 每个 epoch 使用 `val_loss` 更新；
- 当验证集表现停滞时自动降学习率。

#### 阶段 C：EarlyStopping

- 用更任务相关的指标（推荐 `val_dir_ang_p68`）监控；
- 若连续若干次 detailed eval 无提升，则提前停止训练。

---

### 5.7.2 为什么 scheduler 监控 `val_loss`

因为：

- `val_loss` 每个 epoch 都可获得；
- 更新频率高；
- 数值更平滑；
- 不依赖额外的全量复杂评估。

因此 Plateau scheduler 建议：

```python
ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=4~6,
    threshold=1e-3,
    cooldown=1,
    min_lr=1e-6,
)
```

---

### 5.7.3 为什么 early stopping 监控 `val_dir_ang_p68`

因为你最终关心的是：

- 重建质量
- 几何方向误差
- 实际任务指标

而不是单纯的优化损失。

建议 early stopping 用：

```text
val_dir_ang_p68
```

原因：

- 比 `val_loss` 更接近任务目标；
- 比单点极端分位数更稳定；
- 已经在当前训练评估逻辑里存在。

---

### 5.7.4 配置新增建议

```yaml
train:
  scheduler_type: warmup_plateau
  warmup_epochs: 5
  plateau_factor: 0.5
  plateau_patience: 5
  plateau_threshold: 1.0e-3
  plateau_min_lr: 1.0e-6
  early_stop_metric: val_dir_ang_p68
  early_stop_patience: 6
  early_stop_mode: min
```

### 5.7.5 Trainer 代码级改造建议

#### 删除/替换

- 旧的 `get_cosine_with_warmup_scheduler()` 可以保留作兼容，但不再作为默认；
- `self.scheduler.step()` 不再每个 batch 都调用。

#### 新逻辑

- warmup 阶段：每个 step 更新 warmup scheduler；
- plateau 阶段：每个 epoch 结束后 `scheduler.step(val_loss)`；
- detailed eval 阶段：更新 early stopping 计数器。

### 5.7.6 训练日志中必须新增的内容

每个 epoch 输出：

- 当前 LR
- 当前是否处于 warmup / plateau 阶段
- plateau 是否触发过 LR 降低
- early stop 剩余 patience

---

## 5.8 `Metrics.py`：保持主指标体系，支持 early stopping

### 保留

保留当前：

- endpoint angular error
- direction angular error
- midpoint distance
- p68/p90/p95 等汇总

### 新增建议

新增一个辅助函数：

```python
def get_early_stop_metric(history_or_metrics, key='val_dir_ang_p68'):
    ...
```

让 Trainer 统一从 detailed eval 结果中拿 early stopping 指标。

### 注意

不要因为 scheduler 改动而大改 metrics。

指标侧只做：

- 对 Trainer 提供更清晰的接口；
- 避免分散主改动范围。

---

## 5.9 `Plotting.py`：兼容 scheduler 与 v2 指标记录

### 需要补充的内容

训练曲线里新增：

1. 学习率变化曲线（保留）
2. LR 触发衰减节点标记（可选）
3. early stopping 监控指标趋势（如 `val_dir_ang_p68`）

### 可选增强

针对 v2 新增：

- CD token 数压缩前后统计图
- WP token 数分布图
- batch padding 浪费统计图

这些不是必须，但对后续性能调优很有帮助。

---

## 5.10 `Preprocess.py`：保留随机旋转副本语义

### 决策

旋转增强逻辑总体保留，不改成固定旋转集。

### 只建议做两件事

#### （1）修正文档/注释

明确写成：

> 每个 event 在被读取时，会独立随机采样一个 SO(3) 旋转。

不要再写成“每次 expansion 对全体事件施加同一个固定旋转”。

#### （2）允许配置扩增倍数

例如：

```yaml
augmentation:
  rotation_expand_times: 2   # 总数据量≈原始的3倍
```

保持当前语义：

- 原始 1 份
- 随机旋转副本 2 份
- 总量约 3 倍

---

## 5.11 `Config.py`：新增 v2 配置项

建议新增配置块如下。

### 模型配置

```yaml
model:
  # WP
  wp_projector: v2
  wp_geo_hidden: 32
  wp_qt_hidden: 32
  wp_projector_dropout: 0.1
  wp_num_time_buckets: 64

  # CD
  cd_backbone: deepsphere
  cd_deepsphere_layers: 4
  cd_deepsphere_hidden: 256
  cd_deepsphere_norm: rmsnorm
  cd_compression: healpix_pool
  cd_fused_tokens: 128

  # Fusion
  fusion_layers: 4
  d_model: 128
  num_heads: 4
  d_ff: 512
```

### 训练配置

```yaml
train:
  scheduler_type: warmup_plateau
  warmup_epochs: 5
  plateau_factor: 0.5
  plateau_patience: 5
  plateau_threshold: 1.0e-3
  plateau_min_lr: 1.0e-6
  early_stop_metric: val_dir_ang_p68
  early_stop_patience: 6
```

### 增强配置

```yaml
augmentation:
  use_rotation_aug: true
  rotation_expand_times: 2
```

---

## 6. 兼容性与文件改动清单

---

## 6.1 必改文件

```text
python/Model.py
python/TokenProjector.py
python/PositionEncoding.py
python/ModelTrain.py
python/Config.py
python/DataLoader.py
python/Preprocess.py
```

---

## 6.2 新增文件

```text
python/DeepSphere.py
python/WPTimeBias.py   # 或并入 PositionEncoding.py
python/Norms.py        # 可选，若单独实现 RMSNorm
```

---

## 6.3 尽量少改的文件

```text
python/LossFunction.py
python/Metrics.py
python/ModelPredict.py
python/Plotting.py
python/RunModule.py
```

原则：

- 训练主干变了，但入口和输出格式尽量不变；
- 预测与评估逻辑尽量只做兼容性补丁。

---

## 7. 推荐实施顺序

为了控制风险，建议按如下顺序实施。

### Phase 1：WP 升级

1. 实现 `WPProjectorV2`
2. 给 WP self-attn 加 signed relative time bias
3. 保持其余结构不变
4. 先做一次基线对照

目标：

- 验证 WP 提升是否稳定；
- 尽量先在“小改动”条件下收敛一轮。

### Phase 2：CD DeepSphere 替换

1. 新增 `DeepSphere.py`
2. 在 `Model.py` 中用 `DeepSphereEncoder` 替代 `cd_self_attn`
3. 删除 CD-RPE 主路径
4. 引入 `CDCompression`

目标：

- 解决 CD 主瓶颈；
- 测试速度/显存变化；
- 验证性能不掉。

### Phase 3：训练策略升级

1. 引入 warmup + plateau scheduler
2. 引入 early stopping
3. 打通配置项
4. 记录调度状态日志

目标：

- 缓解约 100 epoch 后过拟合；
- 缩短无效训练尾段。

### Phase 4：系统联调

1. 更新预处理
2. 更新预测/评估兼容
3. 更新 plotting
4. 跑完整训练-预测-评估闭环

---

## 8. 验证与对照实验方案

### 必做对照

#### 对照 1：v1 vs v2（只改 WP）

看：

- 收敛速度
- `dir_ang_p68`
- `mid_dist_p68`

#### 对照 2：v1 vs v2（只改 CD）

看：

- 单 epoch 训练时间
- 显存占用
- 验证集核心指标

#### 对照 3：v2 + 新 scheduler vs v2 + 旧 scheduler

看：

- 最佳 epoch
- 是否缓解 100 epoch 以后过拟合
- best checkpoint 的最终表现

### 建议额外记录

- CD compression 前后 token 数
- 平均 batch padding 比例
- 训练总 wall-clock
- 每 epoch 平均耗时

---

## 9. 风险点与规避方案

### 风险 1：CD 改成 DeepSphere 后性能下降

#### 可能原因

- CD 对全局信息建模能力下降
- 融合 token 压得过狠
- DeepSphere 层数/宽度不足

#### 规避

- 保留 global/query 融合结构
- 初期不要把 `cd_fused_tokens` 压太小
- DeepSphere block 起步 3~4 层，不宜过浅

---

### 风险 2：WP relative time bias 不稳定

#### 可能原因

- bucket 太多/太少
- signed bucket 实现不当

#### 规避

- 起步用 32 或 64 buckets
- 做单独 ablation：无 bias / abs(dt) / signed(dt)
- 默认主线仍选 signed(dt)

---

### 风险 3：新 scheduler 与当前训练循环不兼容

#### 规避

- 保留旧 cosine 作为 fallback
- 先把 warmup 单独实现清楚
- 再切换 plateau
- early stopping 最后接入

---

### 风险 4：融合阶段仍然太慢

#### 规避

- CDCompression 必须落地
- 后续如有需要再做 batch bucketing
- 不要一开始就把 `nside` 拉太高

---

## 10. v2 的预期收益

### 10.1 结构收益

- CD 分支摆脱当前 full attention 瓶颈；
- WP 分支表征更强、时间利用更充分；
- 保留整个 Hybrid Token 设计的核心价值。

### 10.2 训练收益

- 显著降低 CD 内部建模代价；
- 中后期学习率更自适应；
- 过拟合出现时可自动降 LR 和提前停止。

### 10.3 工程收益

- 对当前项目代码侵入可控；
- 预测/评估接口可最大兼容；
- 有利于后续继续做 `nside`、compression、scheduler 的系统消融。

---

## 11. 一句话实施摘要

> **HT-Transformer v2 的核心，就是保留 Hybrid Token 总体框架，改掉 CD 的 full-attention 式内部建模，增强 WP 的单 token 表征和时间关系建模，并用 Warmup + Plateau + EarlyStopping 取代固定式学习率衰减。**

这将使项目从“可运行的第一版 Transformer 原型”升级为“更贴合球面探测器几何、更能扩展到更细粒度 patch、更具训练可控性的 v2 架构”。

---

## 12. 后续可继续保留的建议（供最终总结时并入）

以下建议虽然不在本次主实施范围内，但应在最终总结中保留：

1. 融合阶段可进一步做 token-count bucketing，减少 padding 浪费；
2. accelerate 模式下应测试恢复适量 `num_workers`；
3. 训练期详细评估频率可进一步按 wall-clock 优化；
4. loss 权重可在 v2 稳定后再做系统消融；
5. 若后续继续提升 `nside`，应同步关注 CDCompression 和融合层的 token 数控制。

