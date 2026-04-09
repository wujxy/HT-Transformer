# HT-Transformer v2.1: JUNO CDWP Muon Track Endpoint Reconstruction (WP-Backbone + CD-Auxiliary)

## 项目简介

HT-Transformer v2.1 是 HT-Transformer 的 **架构重设计版本**，基于 v2 的 Stage B 优化基础，进一步将模型从"WP/CD 对称双向深融合"转变为 **"WP 主干 + CD 辅助"的非对称架构**。

### 设计动机

在 JUNO CDWP 缪子径迹端点重建任务中：

- **WP（水池 PMT）** 提供切伦科夫光的方向先验，是主导信息源（WP-only 已有较好效果）
- **CD（中心探测器 PMT）** 提供进入/穿出点附近的辅助信息

v2 将两者做对称深融合（每层都做 WP self-attn + WP↔CD + CD→WP + Global + Query），不符合"WP 主导、CD 辅助"的物理先验。v2.1 重新设计信息流：

```
v2:  WP ←→ CD (bidirectional, 4 layers × 5 MHA = 20 QKV projections/step)
v2.1: WP → CD → Query (unidirectional, 5 MHA total)
```

### v2 → v2.1 核心改动

| 改动项 | v2 (旧版) | v2.1 (当前) | 收益 |
|--------|-----------|-------------|------|
| **架构范式** | 对称双向深融合 | WP 主干 + CD 辅助 | 符合物理先验 |
| **融合层** | `HybridFusionLayer` ×4 | `WPSelfAttentionLayer` ×2 + `CDConditioningLayer` + `CrossModalReadout` | 3 个专用组件替代 1 个通用组件 |
| **WP↔CD 交互** | 每层双向 (WP→CD + CD→WP) | 单次单向 WP→CD 条件注入 | CD 不干扰 WP |
| **MHA 次数/步** | 20 | 5 | ↓75% |
| **QKV 投影/步** | 60 | 15 | ↓75% |
| **CD DeepSphere 层数** | 4 | 2 | CD 编码器浅化 |
| **CDCompression** | 固定 dense pooling | 增加 identity 快速路径 (nside_in == nside_out) | 可完全跳过压缩 |
| **数据管线** | batch-level .pt 文件 | split-level HDF5 (train.h5/val.h5/test.h5) | 更高效、支持增量追加 |
| **参数量** | ~3.89M | ~1.84M | ↓53% |

### 核心特点（继承 v2）

- **Hybrid Token Transformer**: WP hit-level + CD patch-level 混合表示
- **DeepSphere CD Encoder**: 固定图局部球面聚合 (v2.1 减至 2 层)
- **Token-level Time Encoding**: WP 时间感知 SDPA-compatible
- **Fixed Dense HEALPix Grid**: CD token index = global pixel id
- **Ordered Dual-Endpoint**: 两个 Query Token 预测有序入射/出射点
- **Multi-GPU Training**: accelerate 库支持 DDP, bf16 混合精度
- **100% SDPA**: 所有 attention 使用 `F.scaled_dot_product_attention`

---

## 快速开始

### 环境依赖

```bash
pip install -r requirements.txt
```

主要依赖：

| 包 | 版本 | 用途 |
|---|---|---|
| torch | >= 2.0 | 深度学习框架 (SDPA/FlashAttention) |
| h5py | >= 3.0 | H5 数据读取 |
| healpy | >= 1.15 | HEALPix patch 映射 |
| pyyaml | >= 5.0 | 配置文件 |
| loguru | >= 0.6 | 日志 |
| tqdm | >= 4.60 | 进度条 |
| matplotlib | >= 3.4 | 可视化 |
| accelerate | >= 0.20 | 多 GPU 训练 |

### 数据准备

1. **H5 事件文件**: 包含 PMT hit 信息 (copyno, charge, hittime) 和真值端点
2. **PMT Geometry 文件**:
   - CD: `PMTPos_CD_LPMT.csv`
   - WP: `PMTPos_WP_LPMT.csv`

### 三步运行

#### 第 1 步: 数据预处理

将 H5 tokenize 为 split-level HDF5 格式并缓存:

```bash
python -m cli.run --config configs/default.yaml --Preprocess
```

**v2.1 数据管线改进**:
- 预处理直接写 `train.h5` / `val.h5` / `test.h5` 独立文件
- 训练 loader 是薄 reader（`SplitDataset`），按需读取 float16 数据并转 float32
- 支持 train 跳过检测：train.h5 已存在且事件数一致则跳过
- val/test 也支持旋转增强：原事例 + N 份旋转副本

#### 第 2 步: 训练

**单 GPU:**
```bash
python -m cli.run --config configs/default.yaml --Train
```

**多 GPU (推荐):**
```bash
accelerate launch --num_processes 2 -m cli.run --config configs/default.yaml --Train
```

#### 第 3 步: 预测与评估

```bash
# 预测
python -m cli.run --config configs/default.yaml --Predict

# 评估
python -m cli.run --config configs/default.yaml --Eval
```

---

## 项目结构

```
HT-Transformer/
├── configs/
│   └── default.yaml              # 默认配置 (v2.1 WP-backbone)
├── models/
│   ├── ht_transformer.py         # 主模型 (v2.1 asymmetric architecture)
│   ├── components/
│   │   ├── token_projectors.py   # WP/CD Projector
│   │   ├── deepsphere.py         # DeepSphere + CDCompression (identity fast path)
│   │   ├── wp_time_encoding.py   # Token-level WP time encoding
│   │   └── position_encoding.py  # Fourier position encoding
│   └── losses/
│       └── endpoint_loss.py      # 三项联合损失
├── data/
│   ├── dataset.py                # Dense HEALPix dataset
│   ├── preprocess.py             # Split-level HDF5 preprocessing ⭐
│   └── augmentation.py           # 数据增强 (rotation)
├── engine/
│   ├── trainer.py                # 训练循环 (accelerate)
│   └── predictor.py              # 预测/评估
├── geometry/
│   ├── detector_geometry.py      # PMT 坐标查找
│   └── healpix_mapper.py         # HEALPix 映射
└── cli/
    └── run.py                    # 统一入口
```

### 输出目录

```
output/{mission_name}/
├── checkpoints/
│   ├── best.pth                  # 最优 checkpoint
│   └── epoch_N.pth
├── preprocessed/                 # Split-level HDF5 format ⭐
│   ├── train.h5                  # float16, wp_offsets + wp_tokens_flat
│   ├── val.h5
│   ├── test.h5
│   ├── cd_unit_vecs.npy          # (npix, 3) shared
│   └── meta.json                 # 含 cd_representation: dense_healpix
└── training_history.json
```

---

## 模型架构 (v2.1)

### 总体流程

```
原始 PMT hit → Geometry 查找 → WP/CD 分离
    │
    ├── WP hits → [ux,uy,uz,q,t] → WPProjector (dual-branch) → WP tokens
    │                                                          ↓
    │                                               WPTimeEncoding (token-level)
    │                                                          ↓
    │                                        ┌─ WPSelfAttentionLayer ×N ──┐
    │                                        │  WP self-attn + FFN        │
    │                                        │  (CD 从不修改 WP)          │
    │                                        └────────────────────────────┘
    │                                                          ↓
    └── CD hits → HEALPix 聚合 → CDProjector → CD tokens
                                     ↓
                          DeepSphereEncoder (2 layers, fixed graph)
                                     ↓
                          CDCompression (dense pooling / identity skip)
                                     ↓
                    + Type Embedding + Fourier Position Encoding
                                     ↓
                          CDConditioningLayer ⭐
                          (CD queries WP for trajectory context)
                                     ↓
                          CrossModalReadout ⭐
                          (Query/Global read from [WP, CD])
                                     ↓
                          Endpoint Heads → pred_u1, pred_u2
```

### v2.1 新增组件

#### 1. WPSelfAttentionLayer — WP 主干层

WP 独立处理，通过 self-attention 建模 hit 间关系。CD 从不参与也不修改 WP 表示。

```python
class WPSelfAttentionLayer(nn.Module):
    """WP backbone: only WP self-attention + FFN."""
    def forward(self, wp_emb, wp_attn_mask):
        # Pre-LN WP self-attn
        wp_normed = self.wp_attn_norm(wp_emb)
        wp_out = wp_emb + self.wp_self_attn(wp_normed, wp_normed, wp_normed, mask=wp_attn_mask)
        # FFN
        wp_out = wp_out + self.wp_ffn(self.wp_ffn_norm(wp_out))
        return wp_out
```

- 参数: 4 × d_model² + 2 × d_model × d_ff (同 v2 单层 WP self-attn 部分)
- CD 从不修改 WP 表示（信息流单向: WP → CD → Query）

#### 2. CDConditioningLayer — WP→CD 单向条件注入

CD 从 WP 获取轨迹上下文，使其聚焦于与该径迹相关的响应模式。这是 WP→CD 的单向信息流，WP 不被修改。

```python
class CDConditioningLayer(nn.Module):
    """Single-direction WP→CD conditioning."""
    def forward(self, cd_emb, wp_emb, cd_cross_mask):
        # CD queries WP for trajectory context
        cd_normed = self.cd_cross_norm(cd_emb)
        cd_out = cd_emb + self.cd_cross_attn(cd_normed, wp_emb, wp_emb, mask=cd_cross_mask)
        cd_out = cd_out + self.cd_ffn(self.cd_ffn_norm(cd_out))
        return cd_out
```

- 仅 1 次 cross-attention（v2 每层都有 WP↔CD 双向，共 4×2=8 次）
- CD 通过读取 WP 知道"这条径迹大致从哪来、往哪去"，从而更好地解释 CD 响应

#### 3. CrossModalReadout — 最终读出层

Query 和 Global token 单向读取 [WP, CD] 的信息。这是 CD 信息进入预测路径的唯一入口。无双向更新。

```python
class CrossModalReadout(nn.Module):
    """Final readout: Query/Global attend to [WP, CD]."""
    def forward(self, wp_emb, cd_emb, global_emb, query_emb, mask_pack):
        # Global attends to [WP, CD]
        all_tokens = torch.cat([wp_emb, cd_emb], dim=1)
        global_out = global_emb + self.global_attn(global_normed, all_tokens, all_tokens, ...)
        # Query attends to [WP, CD, Global]
        all_with_global = torch.cat([wp_emb, cd_emb, global_out], dim=1)
        query_out = query_emb + self.query_attn(query_normed, all_with_global, all_with_global, ...)
        # FFN
        ...
        return global_out, query_out
```

### 信息流对比

```
v2 (对称双向深融合):
┌──────────────────────────────────────────────────────────┐
│  Layer 1: WP self-attn → WP→CD cross → CD→WP cross      │
│           → Global→All → Query→All → FFN                 │
│  Layer 2: WP self-attn → WP→CD cross → CD→WP cross      │
│           → Global→All → Query→All → FFN                 │
│  Layer 3: ... (×4 layers total)                          │
│  Layer 4: ...                                            │
│                                                          │
│  信息流: WP ←→ CD (双向) × 4 层, 共 20 MHA/步           │
│  问题: CD 干扰 WP 表示，违反 WP 主导的物理先验           │
└──────────────────────────────────────────────────────────┘

v2.1 (WP 主干 + CD 辅助):
┌──────────────────────────────────────────────────────────┐
│  WP Backbone: WPSelfAttentionLayer ×N                    │
│    → WP 独立处理，CD 从不参与                             │
│                                                          │
│  CD Conditioning: CDConditioningLayer ×1                 │
│    → CD 从 WP 获取轨迹上下文 (单向 WP→CD)                │
│                                                          │
│  Readout: CrossModalReadout ×1                           │
│    → Query/Global 读取 [WP, CD] (单向)                   │
│                                                          │
│  信息流: WP → CD → Query (单向), 共 5 MHA/步             │
│  优势: 符合物理先验，WP 作为主干不被 CD 干扰              │
└──────────────────────────────────────────────────────────┘
```

### 计算量对比

| 指标 | v2 (4 层对称) | v2.1 (WP 主干 + CD 辅助) |
|------|-------------|------------------------|
| MHA 次数/步 | 20 | 5 (2×WP self + 1×WP→CD + 1×Global + 1×Query) |
| QKV 投影/步 | 60 | 15 |
| O(N_wp²) 操作 | 4 | 2 |
| 双向 cross-attn | 4 层 × 2 方向 | 0 (仅 WP→CD 单向) |

### Token 类型

| Token | 粒度 | 数量 | 输入特征 |
|-------|------|------|----------|
| WP | hit-level | ~500-2000/事件 | `[ux, uy, uz, log1p(q), t_norm]` |
| CD | patch-level (HEALPix) | **固定 768 patches** (nside=8) | 统计量 + time-bin 序列 |
| Global | learnable | 8 | 可训练参数 |
| Query | learnable | 2 | 可训练参数 |

---

## 数据管线 (v2.1 Split-First)

### v2 → v2.1 数据管线改动

| 改动项 | v2 | v2.1 |
|--------|----|----|
| 存储格式 | batch-level .pt 文件 | split-level HDF5 (train/val/test 独立) |
| 浮点精度 | float32 | float16 存储，float32 读取 |
| WP tokens | 每 batch 独立存储 | `(N+1,)` cumulative offsets + flat array |
| Labels | 4 个 `(N, 3)` 数组 | 合并为 `(N, 12)` |
| cd_mask | 显式存储 | 从 `cd_stats[:,:,1]==0` 派生 |
| cd_unit_vecs | 每 batch 重复 | 全局共享 `.npy` 文件 |
| val/test 增强 | 无 | 原事例 + N 份旋转副本 (与 train 相同逻辑) |
| Train 跳过 | 无 | train.h5 已存在且事件数一致则跳过 |

### Split HDF5 文件布局

```
train.h5 / val.h5 / test.h5:
├── cd_stats       (N, npix, 4)    float16   [sumQ, count, t_min, t_mean]
├── cd_time_bins   (N, npix, 32)   float16   time histogram
├── wp_offsets     (N+1,)          int64     cumulative offsets
├── wp_tokens_flat (total_wp*5,)   float16   flattened [ux,uy,uz,q,t]
└── labels         (N, 12)         float16   [u1(3), u2(3), p1(3), p2(3)]
```

### 训练数据流

```
SplitDataset.__getitem__(idx):
    cd_stats     = float16 → float32     # (npix, 4)
    cd_time_bins = float16 → float32     # (npix, 32)
    wp_tokens    = flat[offsets[idx]:offsets[idx+1]] → reshape(-1, 5)
    labels       = float16 → float32     # (12,)
         ↓
collate_fn(batch):
    wp_tokens:  pad → (B, max_wp, 5)
    wp_mask:    (B, max_wp) bool, True = padding
    cd_mask:    从 cd_stats[:, :, 1] == 0 派生
    cd_unit_vecs: 全局 .npy 加载一次, expand to (B, npix, 3)
         ↓
模型输入 dict: {wp_tokens, wp_mask, wp_unit_vecs, wp_times, cd_unit_vecs, cd_stats, cd_time_bins, cd_mask, u1, u2}
```

---

## 损失函数

与 v2 相同，三项联合损失:

```
L = λ_ang · L_ang + λ_len · L_len + λ_dir · L_dir
```

| 损失项 | 公式 | 默认权重 | 作用 |
|--------|------|----------|------|
| L_ang | `0.5 * [(1 - cos_sim(pred_u1, gt_u1)) + (1 - cos_sim(pred_u2, gt_u2))]` | 1.0 | 端点角度回归 |
| L_len | `SmoothL1(‖pred_u2 - pred_u1‖, ‖gt_u2 - gt_u1‖)` | 0.5 | 防止端点塌缩 |
| L_dir | `1 - cosine_sim(normalize(pred_u2-pred_u1), normalize(gt_u2-gt_u1))` | 0.25 | 强化有序方向 |

---

## 配置说明

### 数据配置

```yaml
data:
  h5_path: "/path/to/h5"
  geometry_cd: "PMTPos_CD_LPMT.csv"
  geometry_wp: "PMTPos_WP_LPMT.csv"
  nside: 8                           # HEALPix nside → 768 patches
  num_time_bins: 32                  # CD time-bin 数量
  t_max: 800.0                       # 时间截断 (ns)
  train_ratio: 0.9
  val_ratio: 0.05
  test_ratio: 0.05
```

### 模型配置

```yaml
model:
  d_model: 128
  num_layers: 2                      # WP 主干层数 (v2.1 语义变化: 不再是融合层数)
  num_heads: 4
  d_ff: 512
  num_global_tokens: 8
  num_queries: 2

  # WP Token-level Time Encoding (继承 v2)
  wp_time_encoding: true
  wp_time_hidden: 32
  wp_time_fourier_dim: 16

  # CD DeepSphere (v2.1: 减至 2 层)
  cd_knn_k: 16
  cd_deepsphere_layers: 2            # v2=4, v2.1=2
  cd_deepsphere_hidden: 256
  cd_compression: "healpix_pool"
  cd_fusion_tokens: 128
  cd_compression_nside: 8            # = nside 时跳过压缩 (identity fast path)

  # 其他
  fourier_freq: 32
  dropout: 0.1
  norm_type: "rmsnorm"
```

**v2.1 配置变更**:

| 配置项 | v2 | v2.1 | 说明 |
|--------|----|----|------|
| `cd_deepsphere_layers` | 4 | 2 | CD encoder 浅化 |
| `num_layers` | 2 | 2 | 语义变化: "融合层数" → "WP 主干层数" |
| `cd_compression_nside` | 4 | 8 | 与 nside 相同时跳过压缩 |

### 训练配置

```yaml
train:
  lr: 1.0e-4
  weight_decay: 1.0e-2
  warmup_epochs: 5
  plateau_factor: 0.5
  plateau_patience: 5
  early_stop_patience: 4
  batch_size: 64
  num_epochs: 100
  precision: "bf16"
  use_accelerate: true
  gradient_accumulation_steps: 2

augmentation:
  rotation_expand_times: 2           # 原事例 + 2 份旋转副本 (train/val/test 均适用)
```

---

## 参数量对比

| 组件 | v2 | v2.1 | 变化 |
|------|----|----|------|
| WP Projector | ~25K | ~25K | 不变 |
| CD Projector | ~20K | ~20K | 不变 |
| WP Time Encoding | ~4K | ~4K | 不变 |
| Type + Position Encoding | ~33K | ~33K | 不变 |
| DeepSphere Encoder | ~396K (4 层) | ~198K (2 层) | ↓50% |
| CD Compression | ~0 (healpix_pool) | ~0 (identity skip) | 跳过 |
| **编码层** | **~3,422K** (4×HybridFusion) | **~1,508K** (2×WP + CD条件化 + Readout) | **↓56%** |
| Output Heads | ~33K | ~33K | 不变 |
| Learnable Tokens | ~2.6K | ~2.6K | 不变 |
| **Total** | **~3,89M** | **~1.84M** | **↓53%** |

### 编码层参数明细 (v2.1)

| 组件 | MHA 数 | 参数量 |
|------|--------|--------|
| WPSelfAttentionLayer ×2 | 2 | ~705K |
| CDConditioningLayer ×1 | 1 | ~198K |
| CrossModalReadout ×1 | 2 | ~605K |
| **合计** | **5** | **~1,508K** |

---

## v2 → v2.1 迁移指南

### 数据兼容性

**不兼容**: v2 batch-level .pt 缓存无法用于 v2.1

**迁移步骤**:
```bash
# 1. 删除旧缓存
rm -rf output/{mission_name}/preprocessed

# 2. 重新预处理 (自动生成 split-level HDF5 格式)
python -m cli.run --config configs/default.yaml --Preprocess

# 3. 训练
python -m cli.run --config configs/default.yaml --Train
```

### 代码兼容性

| 模块 | v2 API | v2.1 API |
|------|--------|----------|
| `HTTransformer.__init__` | `self.encoder_layers = ModuleList([HybridFusionLayer(...)])` | `self.wp_layers` + `self.cd_conditioning` + `self.readout` |
| `HTTransformer.forward` | 层循环 `for layer in encoder_layers` | 独立调用 WP backbone → CD conditioning → readout |
| `CDCompression.forward` | `(x, mask)` 始终做 scatter-add | `(x, mask)` 增加 nside_in==nside_out 快速路径 |

---

## 常见问题

**Q: 为什么要把 CD↔WP 双向交互改为单向 WP→CD？**

A: 物理上 WP（水池 PMT）提供切伦科夫光方向先验，是主导信息源。CD（中心探测器）提供进入/穿出点附近辅助信息。双向融合让 CD 干扰 WP 表示，违反物理先验。单向注入让 CD 从 WP 获取轨迹上下文，但 WP 表示保持纯净。

**Q: CD 只在最后被读取一次，信息是否足够？**

A: CD 在被读取前已经过两步处理：(1) DeepSphere 2 层局部球面聚合；(2) CDConditioningLayer 从 WP 获取轨迹上下文。这使得 CD 表示已经融合了局部空间信息和全局轨迹先验。如果精度下降，首先检查 CDCompression 的 nside_out 是否太小导致信息丢失。

**Q: 为什么 CD DeepSphere 从 4 层减到 2 层？**

A: v2.1 中 CD 不再参与深层双向融合，其主要任务是在被读取前完成局部特征提取。2 层 DeepSphere 足以完成 kNN 邻居聚合，更多层只会增加不必要的参数。

**Q: cd_compression_nside 设为 8 (与 nside 相同) 会怎样？**

A: 触发 identity 快速路径，直接返回 `(x, mask)` 跳过所有 scatter-add 计算。适合 CD 命中分布稀疏、希望保留全部 768 个 CD token 的场景。

**Q: 如何重新预处理而不影响已有的 train.h5？**

A: v2.1 预处理会检测 train.h5 是否存在且事件数一致。如果一致则跳过 train 处理，只重新生成 val.h5 和 test.h5。如需强制重新处理，删除 `output/{mission}/preprocessed/` 目录。

**Q: val/test 的旋转增强和 train 一样吗？**

A: 是的，v2.1 中 train/val/test 都使用"原事例 + N 份旋转副本"模式。`rotation_expand_times: 2` 意味着每个集合都是 3 份（原始 + 2 份旋转）。

---

## 附录: v2.1 验收清单

- [x] `WPSelfAttentionLayer`: WP 独立 self-attention + FFN
- [x] `CDConditioningLayer`: 单向 WP→CD cross-attention
- [x] `CrossModalReadout`: Query/Global 单向读取 [WP, CD]
- [x] 信息流 WP → CD → Query (never CD → WP)
- [x] CD DeepSphere 减至 2 层
- [x] CDCompression identity 快速路径 (nside_in == nside_out)
- [x] Split-level HDF5 数据管线 (train/val/test 独立文件)
- [x] Train 跳过检测 (已存在则跳过)
- [x] val/test 旋转增强 (原事例 + N 份旋转副本)
- [x] float16 存储减少磁盘占用
- [x] Mask precompute outside layer loop (继承 v2)
- [x] 100% SDPA 覆盖 (继承 v2)
