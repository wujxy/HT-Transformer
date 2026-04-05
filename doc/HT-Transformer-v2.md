# HT-Transformer v2: JUNO CDWP Muon Track Endpoint Reconstruction (Stage B Optimized)

## 项目简介

HT-Transformer v2 是 HT-Transformer 的 **Stage B 性能优化版本**，针对 JUNO 球形光学探测器（CD + WP 联合）的有序双端点回归任务进行了深度架构优化。

### v2 核心改进

| 优化项 | v1 (旧版) | v2 (当前) | 收益 |
|--------|-----------|-----------|------|
| **WP 时间编码** | `SignedTimeBucketBias` (显式 pairwise B,H,N,N) | `WPTimeEncoding` (token-level) | 显存 ↓, SDPA 全兼容 |
| **CD 表示** | 变长 active HEALPix patches | 固定 dense HEALPix grid | 编译友好, step time 稳定 |
| **CD Self-Attention** | kNN local attention + runtime graph build | DeepSphere on fixed graph | 无 runtime remap, O(N*k) |
| **CD Compression** | active-token compaction | fixed dense pooling | shape 稳定, 无 padding |
| **Attention Mask** | 每层重复构造 | 层循环外预计算 | 冗余计算 ↓ |
| **SDPA 覆盖率** | 5/6 attention | 6/6 attention | 100% FlashAttention |

### 核心特点

- **Hybrid Token Transformer**: WP hit-level + CD patch-level 混合表示
- **DeepSphere CD Encoder**: 固定图局部球面聚合, 无 runtime graph construction
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

将 H5 tokenize 为 dense HEALPix 格式并缓存:

```bash
python -m cli.run --config configs/default.yaml --Preprocess
```

**Stage B 注意**: 预处理输出固定 dense HEALPix grid (`npix = 12 * nside²`)，manifest 标记 `cd_representation: dense_healpix`。旧版 active-patch 缓存会被强制拒绝。

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
│   └── default.yaml              # 默认配置 (Stage B optimized)
├── models/
│   ├── ht_transformer.py         # 主模型 (mask precompute)
│   ├── components/
│   │   ├── token_projectors.py   # WP/CD Projector
│   │   ├── deepsphere.py         # DeepSphere + CDCompression
│   │   ├── wp_time_encoding.py   # Token-level WP time encoding ⭐
│   │   ├── wp_time_bias.py       # Deprecated pairwise bias
│   │   └── position_encoding.py  # Fourier position encoding
│   └── losses/
│       └── endpoint_loss.py      # 三项联合损失
├── data/
│   ├── dataset.py                # Dense HEALPix dataset ⭐
│   ├── preprocess.py             # Dense format preprocessing ⭐
│   └── augmentation.py           # 数据增强
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
├── preprocessed/                 # Stage B: dense_healpix format
│   ├── batch_0000.pt ~ batch_N.pt
│   └── manifest.json             # 含 cd_representation 标记
└── training_history.json
```

---

## 模型架构 (v2)

### 总体流程

```
原始 PMT hit → Geometry 查找 → WP/CD 分离
    │
    ├── WP hits → [ux,uy,uz,q,t] → WPProjectorV2 (dual-branch) → WP tokens
    │                                                              ↓
    │                                                    WPTimeEncoding ⭐
    │                                                              ↓
    └── CD hits → HEALPix 聚合 (dense grid) → CDProjector → CD tokens
                                                              ↓
                                        DeepSphereEncoder (fixed graph) ⭐
                                                              ↓
                                        CDCompression (dense pooling) ⭐
                                                              ↓
                            + Type Embedding + Fourier Position Encoding
                                                              ↓
                                    Hybrid Encoder Layers (×N)
                                                              ↓
                                        (mask precompute outside loop) ⭐
                                                              ↓
                                    Query Tokens → Endpoint Heads → pred_u1, pred_u2
```

### Token 类型

| Token | 粒度 | 数量 | 输入特征 |
|-------|------|------|----------|
| WP | hit-level | ~500-2000/事件 | `[ux, uy, uz, log1p(q), t_norm]` |
| CD | patch-level (HEALPix) | **固定 768 patches** (nside=8) | 统计量 + time-bin 序列 |
| Global | learnable | 8 | 可训练参数 |
| Query | learnable | 2 | 可训练参数 |

### v2 关键改进详解

#### 1. WP Token-level Time Encoding ⭐

**v1 (已废弃)**: `SignedTimeBucketBias` 构造显式 `(B, H, N_wp, N_wp)` bias 张量
- 显存开销大
- 阻止 SDPA 快速路径

**v2**: `WPTimeEncoding` 直接编码到 token embedding
```python
# (B, N_wp) → Fourier features + MLP → (B, N_wp, d_model)
wp_emb = wp_emb + self.wp_time_encoding(wp_times)
```
- 无 pairwise 张量
- **100% SDPA 兼容**

#### 2. CD Fixed Dense HEALPix Grid ⭐

**v1 (已废弃)**: 变长 active patches
- 每事件 patch 数不同
- runtime local graph construction
- compression 需 active-token compaction

**v2**: 固定 dense grid
```python
# 固定 shape: (B, npix, ...)，与事件无关
cd_unit_vecs:  (B, 768, 3)     # pixel center vectors
cd_stats:      (B, 768, 4)     # [sumQ, count, t_min, t_mean]
cd_time_bins:  (B, 768, 32)    # time histogram
cd_mask:       (B, 768)        # True = inactive (no hits)
```
- **Token index = Global HEALPix pixel id**
- 无 runtime remapping
- 所有事件共享固定图拓扑

#### 3. DeepSphere on Fixed Graph ⭐

```python
# v2: 固定 kNN 邻接表，无 per-batch 构造
cd_emb = self.cd_encoder(
    cd_emb,
    knn_adj=self.cd_knn_adj,  # (npix, k), fixed for all batches
    mask=cd_mask,
)
```

- 输入: `(B, npix, D)`
- 邻居聚合: 直接 gather from `knn_adj`，无 `build_local_neighbor_graph`

#### 4. CD Compression: Fixed Dense Pooling ⭐

HEALPix hierarchical pooling (nside=8 → nside=4):
```python
# 固定输出 shape: (B, npix_out, D)
cd_emb, cd_mask = self.cd_compression(cd_emb, mask=cd_mask_input)
# (B, 768, D) → (B, 192, D)
```

- 无 active-token compaction
- 无变长 padding
- 固定 low-res grid

#### 5. Mask Precompute Outside Layer Loop ⭐

```python
# HTTransformer.forward() 中预计算 (once per forward)
mask_pack = {
    'wp_attn': wp_attn_mask,      # (B, N_wp, N_wp)
    'wp_cross': wp_cross_mask,    # (B, N_wp, N_cd)
    'cd_cross': cd_cross_mask,    # (B, N_cd, N_wp)
    'global': global_attn_mask,   # (B, M, N_wp+N_cd)
    'query': query_attn_mask,     # (B, Q, N_wp+N_cd+M)
}

# Layer loop 中直接使用
for layer in self.encoder_layers:
    wp_emb, cd_emb, global_emb, query_emb = layer(
        wp_emb, cd_emb, global_emb, query_emb,
        mask_pack=mask_pack,  # 预计算 masks
    )
```

**优化效果**: 对于 `num_layers=2`，mask 构造从 **10 次** 减少到 **5 次**。

### Hybrid Encoder Layer (v2)

```
┌─────────────────────────────────────────────────────────┐
│              HybridFusionLayer (v2)                     │
│                                                         │
│  1. WP self-attention      (SDPA, no time bias)      ⭐  │
│  2. WP↔CD cross-attention  (bidirectional, SDPA)        │
│  3. CD↔WP cross-attention  (bidirectional, SDPA)        │
│  4. Global→All attention   (SDPA)                       │
│  5. Query→All attention    (SDPA cross-attn)            │
│  6. FFN (Pre-LN, RMSNorm)                               │
└─────────────────────────────────────────────────────────┘
```

**v2 变化**:
- CD self-attention 移入 `DeepSphereEncoder` (pre-fusion)
- 所有 attention 纯 SDPA，无 custom bias
- 接收预计算 `mask_pack`

### SDPA 100% 覆盖

v2 所有 6 个 attention 路径均使用 `F.scaled_dot_product_attention`:

| Attention | v1 后端 | v2 后端 |
|-----------|---------|---------|
| WP self-attn | manual (with bias) | **SDPA** ⭐ |
| CD self-attn | manual (kNN mask) | DeepSphere (fixed gather) |
| WP→CD cross | **SDPA** | **SDPA** |
| CD→WP cross | **SDPA** | **SDPA** |
| Global→All | **SDPA** | **SDPA** |
| Query→All | **SDPA** | **SDPA** |

---

## 损失函数

与 v1 相同，三项联合损失:

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
  h5_path: "/path/to/h5"           # H5 数据路径
  geometry_cd: "PMTPos_CD_LPMT.csv"
  geometry_wp: "PMTPos_WP_LPMT.csv"
  nside: 8                           # HEALPix nside → 768 patches
  num_time_bins: 32                  # CD time-bin 数量
  t_max: 800.0                       # 时间截断 (ns)
  train_ratio: 0.86
  val_ratio: 0.07
  test_ratio: 0.07
```

**Stage B 注意**: `nside` 决定固定 grid 大小: `npix = 12 * nside²`

### 模型配置

```yaml
model:
  d_model: 128
  num_layers: 2
  num_heads: 4
  d_ff: 512
  num_global_tokens: 8
  num_queries: 2

  # WP v2: Token-level time encoding
  wp_time_encoding: true
  wp_time_hidden: 32
  wp_time_fourier_dim: 16

  # CD v2: DeepSphere + Fixed dense compression
  cd_knn_k: 16                       # DeepSphere kNN 邻居数
  cd_deepsphere_layers: 4
  cd_deepsphere_hidden: 256
  cd_compression: "healpix_pool"     # 固定 dense pooling
  cd_fusion_tokens: 128              # 目标 token 数 (实际由 nside_out 决定)
  cd_compression_nside: 4            # nside=8 → nside=4 (768→192)

  # 其他
  fourier_freq: 32                   # Position encoding
  dropout: 0.1
  norm_type: "rmsnorm"
```

**注意**: `wp_time_encoding: true` 启用 v2 token-level 编码；`false` 则关闭时间编码。

### 训练配置

```yaml
train:
  lr: 1.0e-4
  weight_decay: 1.0e-2
  warmup_epochs: 5
  plateau_factor: 0.5
  plateau_patience: 5
  early_stop_patience: 4
  batch_size: 2
  num_epochs: 200
  precision: "bf16"
  use_accelerate: true
```

**Scheduler**: Warmup (5 epochs) → ReduceLROnPlateau

---

## v1 → v2 迁移指南

### 数据兼容性

**不兼容**: v1 active-patch 缓存无法用于 v2 模型

**迁移步骤**:
```bash
# 1. 删除旧缓存
rm -rf output/{mission_name}/preprocessed

# 2. 重新预处理 (自动生成 dense_healpix 格式)
python -m cli.run --config configs/default.yaml --Preprocess

# 3. 训练
python -m cli.run --config configs/default.yaml --Train
```

### 配置变更

| 配置项 | v1 | v2 | 说明 |
|--------|----|----|------|
| `wp_time_encoding` | - | `true` | 新增，启用 token-level 编码 |
| `wp_time_hidden` | - | `32` | 新增 |
| `wp_time_fourier_dim` | - | `16` | 新增 |
| `wp_time_bias` | `"signed_bucket"` | - | **删除**，已废弃 |
| `abs_posenc` | `"fourier"` | - | **删除**，硬编码 |

### 代码兼容性

| 模块 | v1 API | v2 API |
|------|--------|--------|
| `DeepSphereEncoder.forward` | `(x, pixel_ids, full_knn_adj, ...)` | `(x, knn_adj, mask)` |
| `CDCompression.forward` | `(x, pixel_ids, mask)` → 变长 | `(x, mask)` → 固定 shape |
| `HybridFusionLayer.forward` | `(..., wp_mask, cd_mask)` 每层构造 masks | `(..., mask_pack)` 预计算 |

---

## 性能对比

### 理论改进

| 指标 | v1 | v2 | 提升 |
|------|----|----|------|
| WP time 显存 | O(N²) | O(N) | N_wp ~ 1000 时 ↓~1000x |
| CD graph 构造 | O(N*k) per layer | O(1) | 完全消除 runtime 开销 |
| CD compression | 变长, padding | 固定 shape | compile 友好 |
| SDPA 覆盖率 | 5/6 | 6/6 | 100% FlashAttention |
| Step time 稳定性 | 变长导致波动 | 固定 shape | 更稳定 |

### 模型规模 (d_model=128, num_layers=2)

| 组件 | 参数量 |
|------|--------|
| WP Projector V2 | ~25K |
| CD Projector | ~20K |
| WP Time Encoding | ~4K |
| DeepSphere Encoder | ~1,200K |
| CD Compression | ~50K |
| Hybrid Fusion Layers (×2) | ~1,800K |
| Output Heads | ~33K |
| Learnable Tokens | ~2.6K |
| **Total** | **~3,135K** |

---

## 常见问题

**Q: 为什么预处理时提示 "old active-patch format" 错误？**

A: v2 要求 `cd_representation: dense_healpix` 格式。删除旧缓存重新预处理:
```bash
rm -rf output/{mission_name}/preprocessed
python -m cli.run --config configs/default.yaml --Preprocess
```

**Q: nside 是什么？如何调整？**

A: `nside` 是 HEALPix 分辨率参数，`npix = 12 * nside²`。
- nside=8 → 768 pixels (推荐)
- nside=4 → 192 pixels (compression 输出)

**Q: 如何禁用 WP 时间编码？**

A: 设置 `wp_time_encoding: false`:
```yaml
model:
  wp_time_encoding: false  # 关闭时间编码
```

**Q: DeepSphere 的 kNN 邻接表如何构建？**

A: 预计算一次，所有事件共享:
```python
# HTTransformer.__init__ 中
self.register_buffer('cd_knn_adj', build_healpix_knn_adjacency(nside=8, k=16))
```

**Q: 为何移除 SignedTimeBucketBias？**

A: 它构造显式 `(B, H, N, N)` 张量，显存开销大且阻止 SDPA。v2 的 `WPTimeEncoding` 提供等效时间感知且完全 SDPA 兼容。

---

## 附录: Stage B 验收清单

- [x] `data/dataset.py` 输出 dense HEALPix CD tensors
- [x] `data/preprocess.py` 缓存 dense_healpix 格式
- [x] 旧 active-patch cache 强制失效检查
- [x] `DeepSphere` 主路径不依赖 `pixel_ids`
- [x] `DeepSphere` 无 runtime local remap
- [x] `CDCompression` 无 active-token compaction
- [x] `CDCompression` 输出 fixed dense low-res grid
- [x] `HTTransformer` 同步新接口
- [x] CD token index = global HEALPix pixel id
- [x] Mask precompute outside layer loop
