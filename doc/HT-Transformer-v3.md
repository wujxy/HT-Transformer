# HT-Transformer v3: JUNO CDWP Muon Track Endpoint Reconstruction (CD Sparse PMT Token + Late Fusion)

## 项目简介

HT-Transformer v3 是 HT-Transformer 的 **CD 分支重构版本**，将 CD 数据表示从 HEALPix patch 聚合 + DeepSphere 图卷积彻底替换为 **Sparse PMT Token + Self-Attention + Late Fusion** 架构。

### 设计动机

v2.1 的 CD 分支采用 dense HEALPix patch 表示 + DeepSphere 图卷积编码。实验结果表明：

- 与 **仅 WP** 相比，CD 分支未能提供有效增益（效果 ≈ WP-only）
- 68% 角精度约 2.0°，中点距离约 0.9 m，CD 信息没有真正融入预测

**根本原因**：CD hit 先被 patch 化（768 个 HEALPix 像素），再被图卷积压缩，端点附近的精细几何-时序结构在进入融合前已被抹平。对本任务（有序双端点回归）来说，真正需要的是入射/出射端附近的细粒度 PMT 信息，而非全局均匀化的 patch 统计。

v3 的核心思路：

```
v2.1: CD hits → HEALPix patch 化 → DeepSphere → CDCompression → Fusion (信息损失严重)
v3:   CD hits → per-PMT 聚合 → TopK charge 选择 → self-attention → late fusion (全量保留细粒度)
```

### v2.1 → v3 核心改动

| 改动项 | v2.1 (旧版) | v3 (当前) | 收益 |
|--------|-------------|-----------|------|
| **CD 表示** | dense HEALPix patch (768 固定像素) | sparse PMT token (TopK by charge, K_cd=640) | 保留端点附近精细几何-时序信息 |
| **CD 编码器** | DeepSphere 图卷积 (2 层) | self-attention (1 层) | 纯 attention，无需图拓扑 |
| **CD token 投影** | CDProjector (统计量 + time-bin Conv1d) | CDHitProjector (双分支 geo+physics MLP) | 适配 PMT 级特征 |
| **CD 时间编码** | time-bin 直方图 (32 bins) | CDTimeEmbedding (Fourier + MLP) | 更紧凑，SDPA 兼容 |
| **CD 压缩** | HEALPix hierarchical pooling | 无压缩，全量进入 fusion | 最大化保留 CD 细粒度信息 |
| **WP→CD 条件化** | CDConditioningLayer (WP→CD cross-attn) | 移除 | 简化信息流，CD 独立编码 |
| **融合方式** | WP→CD 条件化 + CrossModalReadout | 单次 Late Fusion (CrossModalReadout) | 更简洁，CD 从不干扰 WP |
| **数据管线** | cd_stats + cd_time_bins + cd_unit_vecs | cd_tokens + cd_mask (预处理离线计算) | 训练时零 CD 计算开销 |
| **依赖** | healpy + HEALPix 相关模块 | 移除 healpy | 减少外部依赖 |
| **参数量** | ~1.84M | ~1.08M | ↓41% |

### 核心特点

**v3 新增：**
- **Sparse PMT Token**: CD 以 per-PMT 聚合 + TopK charge 选择作为主输入
- **CD Self-Attention Encoder**: CD 分支独立 self-attention 编码
- **Late Fusion**: Global/Query 单次读取 [WP, CD]，无压缩全量融合，CD 从不干扰 WP
- **CDTimeEmbedding**: Fourier + MLP token-level 时间编码

**继承自 v2.1：**
- **WP 双分支投影**: geometry [ux,uy,uz] + optical-time [q,t] 双分支 MLP
- **WP Token-level Time Encoding**: SDPA 兼容的 Fourier 时间编码
- **Ordered Dual-Endpoint**: 两个 Query Token 预测有序入射/出射点
- **三项联合损失**: 角度 + 长度 + 方向一致性
- **Multi-GPU Training**: accelerate 库支持 DDP, bf16 混合精度
- **100% SDPA**: 所有 attention 使用 `F.scaled_dot_product_attention`
- **RMSNorm**: 全局使用 RMSNorm 替代 LayerNorm
- **Split HDF5 数据管线**: train/val/test 独立文件，float16 存储

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
| numpy | >= 1.21 | 数值计算 |
| pandas | >= 1.3 | CSV 几何表读取 |
| pyyaml | >= 5.0 | 配置文件 |
| loguru | >= 0.6 | 日志 |
| tqdm | >= 4.60 | 进度条 |
| matplotlib | >= 3.4 | 可视化 |
| accelerate | >= 0.20 | 多 GPU 训练 |

**注意**: v3 已移除 `healpy` 依赖。

### 数据准备

1. **H5 事件文件**: 包含 PMT hit 信息 (copyno, charge, hittime) 和真值端点
2. **PMT Geometry 文件**:
   - CD: `PMTPos_CD_LPMT.csv`
   - WP: `PMTPos_WP_LPMT.csv`

### 三步运行

#### 第 1 步: 数据预处理

将 H5 tokenize 为 v3 格式 (CD sparse PMT token) 并缓存:

```bash
python -m cli.run --config configs/default.yaml --Preprocess
```

**v3 数据管线**:
- CD: per-PMT 聚合 → TopK by q_sum → 写入 `cd_tokens` + `cd_mask`
- WP: 沿用现有 tokenization
- 输出 split-level HDF5: `train.h5` / `val.h5` / `test.h5`

#### 第 2 步: 训练

**单 GPU:**
```bash
python -m cli.run --config configs/default.yaml --TrainModel
```

**多 GPU (推荐):**
```bash
accelerate launch --num_processes 2 -m cli.run --config configs/default.yaml --TrainModel
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
│   └── default.yaml                # 默认配置 (v3)
├── models/
│   ├── ht_transformer.py           # 主模型 (v3 architecture)
│   ├── components/
│   │   ├── token_projectors.py     # WPProjector + CDHitProjector + CDTimeEmbedding
│   │   ├── wp_time_encoding.py     # Token-level WP time encoding
│   │   └── norms.py                # RMSNorm
│   └── losses/
│       └── endpoint_loss.py        # 三项联合损失
├── data/
│   ├── dataset.py                  # H5EndpointDataset + v3 CD token 构造
│   ├── preprocess.py               # Split-level HDF5 预处理 (v3 CD sparse)
│   ├── augmentation.py             # 数据增强 (rotation)
│   └── normalization.py            # 归一化工具
├── engine/
│   ├── trainer.py                  # 训练循环 (accelerate)
│   └── predictor.py                # 预测/评估
├── geometry/
│   └── detector_geometry.py        # PMT 坐标查找 (CD + WP 双查找表)
├── metrics/
│   └── endpoint_metrics.py         # 评估指标
├── visualization/
│   └── plotting.py                 # 可视化
├── config/
│   └── loader.py                   # 配置加载 (YAML + CLI override)
├── cli/
│   └── run.py                      # 统一入口
└── doc/
    ├── HT-Transformer-v1.md
    ├── HT-Transformer-v2.md
    ├── HT-Transformer-v2.1.md
    └── HT-Transformer-v3.md        # 本文件
```

### 输出目录

```
output/{mission_name}/
├── checkpoints/
│   ├── best.pth                    # 最优 checkpoint
│   ├── final.pth                   # 最终 checkpoint
│   └── epoch_N.pth                 # 定期保存
├── preprocessed/                   # v3 format
│   ├── train.h5                    # cd_tokens(N,K_cd,10) + cd_mask(N,K_cd) + wp + labels
│   ├── val.h5
│   ├── test.h5
│   └── meta.json                   # cd_representation: sparse_pmt_precomputed
├── plots/
│   ├── training_curves.png         # 训练曲线
│   └── val_*.png                   # 评估分布图
└── training_history.json           # 完整训练历史
```

---

## 模型架构 (v3)

### 总体流程

```
原始 PMT hit → Geometry 查找 → WP/CD 分离
    │
    ├── WP hits → [ux,uy,uz,q,t] → WPProjector (dual-branch) → WP tokens
    │                                                          ↓
    │                                               WPTimeEncoding (token-level)
    │                                                          ↓
    │                                        ┌─ WPSelfAttentionLayer ×2 ─┐
    │                                        │  WP self-attn + FFN        │
    │                                        │  (CD 从不修改 WP)          │
    │                                        └────────────────────────────┘
    │                                                          ↓ H_wp
    │
    └── CD hits → [预处理] per-PMT 聚合 + TopK by charge
                   → cd_tokens (K_cd=640, 10-dim) + cd_mask
                   → CDHitProjector (dual-branch) → CD embeddings
                                                ↓
                                      CDTimeEmbedding (Fourier + MLP)
                                                ↓
                                      CDSparseEncoderLayer ×1 (self-attn)
                                                ↓ H_cd (全量 640 tokens)
                    + Type Embedding + Fourier Position Encoding
                                                ↓
                                  CrossModalReadout (Late Fusion)
                                  Global → attend [H_wp, H_cd]
                                  Query  → attend [H_wp, H_cd, Global]
                                                ↓
                                  Endpoint Heads → pred_u1, pred_u2
```

### v3 新增组件详解

#### 1. CDHitProjector — CD PMT Token 投影

将 10 维 CD PMT token 投影到 d_model 维度。采用与 WPProjector 类似的双分支结构：

```python
class CDHitProjector(nn.Module):
    """Dual-branch: geo[3] + physics-time[7] → d_model"""
    def __init__(self, d_model=128, d_geo=32, d_pt=64, hidden=64, dropout=0.1): ...
    def forward(self, cd_tokens):  # (B, K_cd, 10) → (B, K_cd, d_model)
```

- **geo branch**: `[ux, uy,uz]` → MLP → d_geo
- **physics-time branch**: `[q_sum, q_max, n_hits, t_first, t_mean, t_late, t_span]` → MLP → d_pt
- **融合**: concat → MLP → d_model

参数量: ~19K

#### 2. CDTimeEmbedding — CD 时间编码

为 CD PMT token 提供独立的时间增强编码，与 WPTimeEncoding 设计哲学一致：

```python
class CDTimeEmbedding(nn.Module):
    """Fourier features + MLP on [t_first, t_mean, t_late, t_span]"""
    def __init__(self, d_model=128, hidden=32, fourier_dim=16): ...
    def forward(self, cd_time_features):  # (B, K_cd, 4) → (B, K_cd, d_model)
```

- 输入: 4 维时间特征 `[t_first, t_mean, t_late, t_span]`
- 随机 Fourier 频率投影 + 原始时间拼接 → MLP → d_model
- SDPA 兼容：时间作为 token-level additive embedding

参数量: ~8K

#### 3. CDSparseEncoderLayer — CD Self-Attention 编码器

CD 分支独立 self-attention 编码层：

```python
class CDSparseEncoderLayer(nn.Module):
    """Pre-LN + self-attention + residual + FFN + residual"""
    def __init__(self, d_model=128, num_heads=4, d_ff=512, dropout=0.1): ...
    def forward(self, cd_emb, cd_attn_mask):  # (B, K_cd, D) → (B, K_cd, D)
```

- 结构与 `WPSelfAttentionLayer` 同型
- 使用 `cd_mask` 屏蔽 padding token
- v3 首版使用 1 层
- 100% SDPA 兼容

参数量: ~198K

#### 4. CrossModalReadout — 单次 Late Fusion

Query 和 Global token 单向读取 [WP, CD] 信息（继承自 v2.1）：

```python
class CrossModalReadout(nn.Module):
    """Global → [H_wp, H_cd], Query → [H_wp, H_cd, Global]"""
    def forward(self, wp_emb, cd_emb, global_emb, query_emb, mask_pack):
        # Global attends to [WP, CD]
        # Query attends to [WP, CD, Global]
        # + FFN
        return global_out, query_out
```

**v3 与 v2.1 的关键差异**：
- v2.1: CrossModalReadout 接收 CDCompression 输出（HEALPix low-res grid）
- v3: CrossModalReadout 直接接收全部 640 个 CD sparse token，无压缩

### 信息流对比

```
v2.1 (WP 主干 + CD 条件化):
┌──────────────────────────────────────────────────────────┐
│  WP Backbone: WPSelfAttentionLayer ×2                    │
│    → WP 独立处理                                          │
│                                                          │
│  CD Encoding: DeepSphere (2 layers) → CDCompression      │
│    → HEALPix patch 图卷积 + pooling 压缩                 │
│                                                          │
│  CD Conditioning: CDConditioningLayer ×1                 │
│    → CD 从 WP 获取轨迹上下文 (WP→CD 单向)                │
│                                                          │
│  Readout: CrossModalReadout ×1                           │
│    → Query/Global 读取 [WP, CD]                          │
│                                                          │
│  信息流: WP → CD → Query (5 MHA/步)                      │
│  问题: patch 化损失端点附近精细信息                        │
└──────────────────────────────────────────────────────────┘

v3 (CD Sparse PMT Token + Late Fusion):
┌──────────────────────────────────────────────────────────┐
│  WP Backbone: WPSelfAttentionLayer ×2                    │
│    → WP 独立处理 (不变)                                   │
│                                                          │
│  CD Encoding: CDHitProjector + CDTimeEmbedding            │
│             → CDSparseEncoderLayer ×1 (self-attn)        │
│    → PMT 级细粒度特征，全量保留无压缩                     │
│                                                          │
│  Late Fusion: CrossModalReadout ×1                       │
│    → Global 读取 [WP, CD]                                │
│    → Query  读取 [WP, CD, Global]                        │
│                                                          │
│  信息流: WP → Query, CD → Query (3 MHA/步)               │
│  优势: CD 全量保留 PMT 级信息，独立编码后单次融合          │
└──────────────────────────────────────────────────────────┘
```

### 计算量对比

| 指标 | v2.1 | v3 |
|------|------|-----|
| MHA 次数/步 | 5 (2×WP self + 1×WP→CD + 1×Global + 1×Query) | 3 (2×WP self + 1×Global + 1×Query) |
| CD 编码 | DeepSphere 图卷积 (2 层) | self-attention (1 层) |
| CD→WP 条件化 | 有 (WP→CD cross-attn) | 无 (CD 独立编码) |
| 融合交互 | 单次 (但含条件化) | 单次 (纯 late fusion，无压缩) |

---

## 数据管线 (v3 Sparse PMT Token)

### CD PMT Token 构造

v3 在预处理阶段将 CD hit 离线聚合为 sparse PMT token，训练时直接读取。

#### 第 1 步: Per-PMT 聚合

对每个事件中同一个 CD PMT 的所有 hit 进行聚合：

```python
aggregate_cd_hits_by_pmt(cd_copyno, cd_unit_vecs, cd_charges, cd_times, geometry, t_max)
# → pmt_features: (N_unique_pmts, 10)
```

输出 10 维特征：

| 特征 | 定义 | 归一化 |
|------|------|--------|
| `ux, uy, uz` | PMT 方向单位向量 (geometry 查表) | 无 (已是单位向量) |
| `q_sum` | 总电荷 | log10(x+1) + clip p99 + min-max |
| `q_max` | 最大电荷 | log10(x+1) + clip p99 + min-max |
| `n_hits` | hit 数 | log10(n+1) + min-max |
| `t_first` | 最早时间 | clip / t_max → [0, 1] |
| `t_mean` | 电荷加权平均时间 | clip / t_max → [0, 1] |
| `t_late` | 90th 分位时间 (t90) | clip / t_max → [0, 1] |
| `t_span` | t_late - t_first | clip / t_max → [0, 1] |

**t_late 特殊处理**: >=3 个 hit 时使用 `np.percentile(times, 90)` (t90)；<3 个 hit 时使用 `np.max(times)`。

#### 第 2 步: TopK by Charge

按 `q_sum` 降序排列，取前 `K_cd` 个 PMT：

```python
select_topk_cd_tokens_by_charge(pmt_features, K_cd=640)
# → cd_tokens: (K_cd, 10), cd_mask: (K_cd,) bool
```

- 选择规则 **仅基于 charge**，禁止引入 time/coverage 等启发式规则
- 不足 `K_cd` 个时零填充，`cd_mask=True` 标记 padding 位

### v3 HDF5 文件布局

```
train.h5 / val.h5 / test.h5:
├── cd_tokens       (N, K_cd, 10)   float16   [ux,uy,uz,q_sum,q_max,n_hits,t_first,t_mean,t_late,t_span]
├── cd_mask         (N, K_cd)       bool      True = padding
├── wp_offsets      (N+1,)          int64     cumulative offsets
├── wp_tokens_flat  (total_wp*5,)   float16   flattened [ux,uy,uz,q,t]
└── labels          (N, 12)         float16   [u1(3), u2(3), p1(3), p2(3)]
```

### v2.1 → v3 数据管线改动

| 改动项 | v2.1 | v3 |
|--------|------|-----|
| CD 数据字段 | `cd_stats` (N,768,4) + `cd_time_bins` (N,768,32) | `cd_tokens` (N,640,10) + `cd_mask` (N,640) |
| CD 聚合方式 | HEALPix patch 聚合 | per-PMT 聚合 |
| CD 选择方式 | 无 (固定 768 patch) | TopK by q_sum (640 PMT) |
| 全局共享数据 | `cd_unit_vecs.npy` (768,3) | 无 (方向向量编码在 cd_tokens 中) |
| 训练时 CD 计算 | 需从 cd_stats 派生 cd_mask | 直接读取 cd_tokens + cd_mask |
| HDF5 磁盘占用 | cd_stats + cd_time_bins | cd_tokens (更紧凑) |

### 训练数据流

```
SplitDataset.__getitem__(idx):
    cd_tokens     = float16 → float32     # (K_cd, 10)
    cd_mask       = bool                  # (K_cd,)
    wp_tokens     = flat[offset:offset+1] → reshape(-1, 5)
    labels        = float16 → float32     # (12,)
         ↓
collate_fn(batch):
    wp_tokens:  pad → (B, max_wp, 5)
    wp_mask:    (B, max_wp) bool, True = padding
    wp_unit_vecs: wp_tokens[:, :, :3]     # 派生
    wp_times:   wp_tokens[:, :, 4]        # 派生
    cd_tokens:  stack → (B, K_cd, 10)     # 固定长度，直接 stack
    cd_mask:    stack → (B, K_cd) bool
         ↓
模型输入 dict: {wp_tokens, wp_mask, wp_unit_vecs, wp_times,
               cd_tokens, cd_mask, u1, u2}
```

### 预处理 metadata

`meta.json` 记录 v3 配置：

```json
{
  "cd_representation": "sparse_pmt_precomputed",
  "cd_max_tokens": 640,
  "cd_topk_score": "q_sum",
  "cd_feature_order": ["ux", "uy", "uz", "q_sum", "q_max", "n_hits",
                        "t_first", "t_mean", "t_late", "t_span"]
}
```

---

## 损失函数

与 v2.1 相同，三项联合损失:

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
  t_max: 800.0                       # 时间截断 (ns)
  train_ratio: 0.9
  val_ratio: 0.05
  test_ratio: 0.05
```

**v3 移除**: `nside`, `num_time_bins` (不再使用 HEALPix)

### 模型配置

```yaml
model:
  version: "v3"
  cd_representation: "sparse_pmt_precomputed"

  d_model: 128
  num_heads: 4
  d_ff: 512
  dropout: 0.1

  # Encoder layers
  wp_self_layers: 2                  # WP 主干层数
  cd_self_layers: 1                  # CD sparse encoder 层数

  # Global/Query tokens
  num_global_tokens: 8
  num_queries: 2

  # v3 CD sparse token config
  cd_max_tokens: 640                 # K_cd: TopK 选择数量
  cd_topk_score: "q_sum"             # 选择评分字段
  cd_feature_dim: 10                 # CD token 特征维度

  # CD time embedding
  cd_time_embedding: true
  cd_time_hidden: 32
  cd_time_fourier_dim: 16

  # WP projector (不变)
  wp_geo_hidden: 32
  wp_qt_hidden: 32
  wp_projector_hidden: 64

  # WP time encoding (不变)
  wp_time_encoding: true
  wp_time_hidden: 32
  wp_time_fourier_dim: 16

  # Position encoding
  abs_posenc: "fourier"
  fourier_freq: 32

  # Normalization
  norm_type: "rmsnorm"
```

**v3 移除的配置项**:

| 配置项 | 说明 |
|--------|------|
| `nside` | 不再使用 HEALPix |
| `num_time_bins` | 不再使用 time-bin 直方图 |
| `cd_knn_k` | 不再使用 kNN 图 |
| `cd_deepsphere_layers` | 不再使用 DeepSphere |
| `cd_compression` | 不再使用 HEALPix pooling |
| `cd_fusion_tokens` | 不再使用 latent compression |
| `cd_compression_nside` | 不再使用 HEALPix 压缩 |
| `num_layers` | 改用 `wp_self_layers` + `cd_self_layers` |

### 训练配置

```yaml
train:
  optimizer: "adamw"
  lr: 1.0e-4
  weight_decay: 1.0e-2

  # Scheduler: Warmup + ReduceLROnPlateau
  warmup_epochs: 5
  plateau_factor: 0.5
  plateau_patience: 5
  plateau_min_lr: 1.0e-6

  # Early stopping
  early_stop_patience: 4
  early_stop_monitor: "val_dir_ang_p68"

  precision: "bf16"
  batch_size: 64
  num_epochs: 100
  use_accelerate: true
  gradient_accumulation_steps: 2
```

---

## 参数量对比

| 组件 | v2.1 | v3 | 变化 |
|------|------|-----|------|
| WP Projector | ~25K | ~13K | dual-branch 结构调整 |
| CD Projector | ~20K | ~19K (CDHitProjector) | 重写为 PMT 级投影 |
| WP Time Encoding | ~4K | ~5K | 不变 |
| CD Time Embedding | - | ~8K | **新增** |
| Type + Position Encoding | ~33K | ~9K | 精简 |
| DeepSphere Encoder | ~198K (2 层) | - | **移除** |
| CD Compression | ~0 (healpix_pool) | - | **移除** |
| CDConditioningLayer | ~198K | - | **移除** |
| WPSelfAttentionLayer ×2 | ~705K | ~396K | 参数调整 |
| CDSparseEncoderLayer ×1 | - | ~198K | **新增** |
| CrossModalReadout | ~605K | ~396K | 参数调整 |
| Output Heads | ~33K | ~34K | 不变 |
| Learnable Tokens | ~2.6K | ~1.3K | token 数调整 |
| **Total** | **~1.84M** | **~1.08M** | **↓41%** |

---

## v2.1 → v3 迁移指南

### 数据兼容性

**不兼容**: v2.1 HDF5 (cd_stats + cd_time_bins) 无法用于 v3

**迁移步骤**:
```bash
# 1. 删除旧缓存
rm -rf output/{mission_name}/preprocessed

# 2. 重新预处理 (自动生成 v3 sparse PMT token 格式)
python -m cli.run --config configs/default.yaml --Preprocess

# 3. 训练
python -m cli.run --config configs/default.yaml --TrainModel
```

### 配置变更

| 配置项 | v2.1 | v3 | 说明 |
|--------|------|-----|------|
| `mission_name` | `ht_transformer_v2.1` | `ht_transformer_v3` | 版本标识 |
| `model.version` | - | `"v3"` | 新增 |
| `model.cd_representation` | - | `"sparse_pmt_precomputed"` | 新增 |
| `model.wp_self_layers` | `num_layers` | `wp_self_layers: 2` | 重命名 |
| `model.cd_self_layers` | - | `1` | 新增 |
| `model.cd_max_tokens` | - | `640` | 新增 (TopK 数量) |
| `model.cd_latent_tokens` | `cd_fusion_tokens` | - | **移除**（不再使用 latent compression） |
| `data.nside` | `8` | - | **移除** |
| `data.num_time_bins` | `32` | - | **移除** |
| `model.cd_knn_k` | `16` | - | **移除** |
| `model.cd_deepsphere_layers` | `2` | - | **移除** |

### 代码兼容性

| 模块 | v2.1 | v3 |
|------|------|-----|
| `HTTransformer.__init__` | `self.cd_encoder` (DeepSphere) + `self.cd_compression` + `self.cd_conditioning` | `self.cd_hit_projector` + `self.cd_time_embedding` + `self.cd_sparse_encoder` |
| `HTTransformer.forward` | WP encode → CD encode → WP→CD condition → readout | WP encode → CD encode → late fusion readout (无压缩) |
| `collate_fn` 输入 | cd_stats + cd_time_bins + cd_unit_vecs | cd_tokens + cd_mask |
| 预处理输出 | cd_stats + cd_time_bins + cd_unit_vecs.npy | cd_tokens + cd_mask |

### 已移除的依赖和模块

**依赖移除**:
- `healpy` (requirements.txt 中已移除)

**代码移除**:
- `geometry/healpix_mapper.py` — HEALPix 映射模块
- `models/components/deepsphere.py` — DeepSphere 编码器 + CDCompression
- `models/components/wp_time_bias.py` — 已废弃的 pairwise time bias
- `models/components/position_encoding.py` — 已废弃的旧位置编码
- `CDProjector` 类 — v2.1 CD patch 投影器
- `CDConditioningLayer` 类 — v2.1 WP→CD 条件化层
- `CDLatentCompression` 类 — CD 隐压缩（v3 设计初期后移除）

---

## 常见问题

**Q: 为什么从 HEALPix patch 切换到 sparse PMT token？**

A: v2.1 的 patch 化将所有 CD PMT hit 均匀映射到 768 个 HEALPix 像素，然后做图卷积压缩。这个过程丢失了端点附近的精细几何-时序信息——而端点重建任务恰恰依赖这些细粒度信号。v3 直接保留 PMT 级信息，通过 TopK charge 选择保留最相关的 PMT。

**Q: TopK 选择为什么只用 charge，不考虑 time？**

A: 这是有意的设计约束。若同时按 charge/time/coverage 混合挑选，会引入过强手工偏置，污染实验解释性。v3 首先验证：只要 CD 不再被 patch 化，仅靠高 charge PMT token + time embedding，CD 是否就能带来提升。时间信息通过 CDTimeEmbedding 进入模型，而非参与 token 选择。

**Q: K_cd=640 太多/太少怎么办？**

A: 可通过 `model.cd_max_tokens` 调整。推荐范围 512-640。K_cd 影响的是 self-attention 的序列长度（K_cd² 计算量），如果速度压力大可先设 512 做初步验证。

**Q: 为什么移除了 CDLatentCompression？**

A: v3 的核心动机是保留端点附近的精细 PMT 信息。CDLatentCompression 将 640 个 CD token 压缩为 64 个，虽然通过 cross-attention 实现可学习压缩，但 10:1 的压缩比仍可能损失关键信息。移除后 CD 全量 640 token 直接进入 CrossModalReadout，确保细粒度信息完整保留。融合阶段的 attention 序列变长（~1000+ tokens），但只有 2 次 MHA 调用，计算开销可控。

**Q: 为什么移除了 CDConditioningLayer？**

A: v2.1 中 CDConditioningLayer 让 CD 从 WP 获取轨迹上下文（WP→CD 单向），但实验证明 CD patch 化后的信息损失使得条件化效果有限。v3 中 CD 直接以 PMT 级特征独立编码，不再需要从 WP 获取上下文——CD 的 self-attention 已足以建模 PMT 间关系。

**Q: v3 和 v2.1 的 WP 路径有区别吗？**

A: 没有。WP 路径完全保持不变：WPProjector (dual-branch) → WPTimeEncoding → WPSelfAttentionLayer ×2。v3 的所有改动都在 CD 分支。

**Q: 如何重新预处理而不影响已有的 train.h5？**

A: 预处理会检测 train.h5 是否存在且事件数一致。如需强制重新处理，删除 `output/{mission}/preprocessed/` 目录后重新运行 `--Preprocess`。

---

## 附录: v3 设计原则

### 原则 1: CD token 只按 charge 选择

- 选择规则仅基于 `q_sum`
- 时间信息只通过 token 特征和 CDTimeEmbedding 进入模型
- 禁止在 token 选择阶段使用 early/late/coverage 等二级启发式规则

### 原则 2: CD 聚合与 TopK 选择前移到预处理阶段

- PMT 聚合、排序、TopK 选择在 `data/preprocess.py` 中离线完成
- 训练时只读取定长 `cd_tokens` 与 `cd_mask`
- 训练阶段不得在 `Dataset.__getitem__()` 中再做 CD token 聚合

### 原则 3: CD 分支独立 self-attention + 单次晚融合

- CD 有独立的 sparse self-attention encoder
- 融合只在末端做一次 (CrossModalReadout)
- 禁止恢复每层双向 WP↔CD 重型交互

### 原则 4: WP 路径不改动

- WP 分支保持 v2.1 的完整架构
- 重构重点只放在 CD 表示、编码、融合路径

---

## 附录: v3 验收清单

- [x] CD PMT token 构造: `aggregate_cd_hits_by_pmt` + `select_topk_cd_tokens_by_charge`
- [x] 预处理输出 v3 HDF5: `cd_tokens` (N, K_cd, 10) + `cd_mask` (N, K_cd)
- [x] `CDHitProjector`: 双分支 geo+physics-time → d_model
- [x] `CDTimeEmbedding`: Fourier + MLP token-level 时间编码
- [x] `CDSparseEncoderLayer`: CD 独立 self-attention 编码
- [x] `CrossModalReadout`: 单次 late fusion (Global → [WP, CD], Query → [WP, CD, Global])，无压缩
- [x] 信息流: WP → Query, CD → Query (CD 从不干扰 WP)
- [x] 移除 DeepSphere / HEALPix / CDConditioningLayer 依赖
- [x] 移除 healpy 外部依赖
- [x] 移除 CDLatentCompression，CD 全量直接进入 fusion
- [x] 模型参数量: ~1.08M (↓41% vs v2.1)
- [x] 100% SDPA 覆盖
- [x] meta.json 记录 `cd_representation: sparse_pmt_precomputed`
- [x] WP 路径保持不变
