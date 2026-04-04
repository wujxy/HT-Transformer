# HT-Transformer 径迹端点重建 — 实施计划

## Context

基于 JUNO 球形光学探测器的 CDWP（CD + WP 联合）hit 数据，实现一套 Hybrid Token Transformer，用于**有序双端点回归**——预测粒子径迹与球面的入射点和出射点（单位向量）。

参考项目 `muon_track_reco_CDWP/muon_track_reco_transformer/python` 已有完整的 CDWP 数据加载、训练、预测框架（DataLoader.py / ModelTrain.py / ModelPredict.py / RunModule.py），本计划在其基础上**增量修改**，将标准 Transformer 替换为 Hybrid Token Transformer（CD HEALPix patch + WP hit-level token + kNN attention + Global token），并将输出改为有序双端点单位向量回归。

---

## 已确认的数据与资源

### H5 Schema（`output_cdwp_train_10459.h5`，4 个事件）

| Key | Shape | Dtype | 说明 |
|-----|-------|-------|------|
| `copyno` | (4, 20012) | uint16 | PMT 编号，CD < 50000，WP >= 50000 |
| `charge` | (4, 20012) | float32 | 电荷 (PE) |
| `time` | (4, 20012) | float32 | 相对触发时间 (ns) |
| `nhits` | (4,) | uint16 | 每事件实际 hit 数 |
| `tt_enter_x/y/z` | (4,) | float32 | 入射点坐标 (mm) |
| `tt_exit_x/y/z` | (4,) | float32 | 出射点坐标 (mm) |
| `track_time` | (4,) | uint64 | 时间戳 |

- 定长存储，padding 至 20012
- CD: ~17612 PMTs (copyno 0–17611)，WP: ~2400 PMTs (copyno 50000–52399)

### Geometry 文件

| 文件 | PMT 数 | Key 列 | 内容 |
|------|--------|--------|------|
| `PMTInfo.csv` | 17612 | PMTID_sim | CD PMT: X,Y,Z,Theta,Phi,Type,... |
| `PMTPos_WP_LPMT.csv` | 2400 | CopyNo | WP PMT: X,Y,Z,Orientation_theta,phi |
| `PMTPos_CD_LPMT.csv` | 17612 | CopyNo | CD PMT: X,Y,Z,Theta,Phi |

---

## 项目目录结构

```
HT-Transformer/
├── CLAUDE.md
├── 修订.md
├── CLAUDE_PLAN.md
├── output_cdwp_train_10459.h5
│
├── configs/
│   └── default.yaml                 # 默认配置文件
│
├── python/
│   ├── RunModule.py                 # 统一入口（训练/预测/评估/Schema检查）
│   ├── Config.py                    # YAML 配置加载 + CLI 覆盖
│   ├── Fuc.py                       # 工具函数（复用参考项目）
│   ├── DataLoader.py                # H5 数据加载 + WP/CD tokenization
│   ├── Geometry.py                  # DualPMTPositionLookup（复用参考项目）
│   ├── HEALPix.py                   # HEALPix patch 映射 + kNN 邻接表预计算
│   ├── Model.py                     # Hybrid Token Transformer 模型
│   ├── PositionEncoding.py          # Fourier 绝对 PE + Bucket RPE
│   ├── TokenProjector.py            # WPProjector / CDProjector + 类型嵌入
│   ├── LossFunction.py              # 三项损失（角度 + 长度 + 方向）
│   ├── ModelTrain.py                # 训练循环（复用参考项目框架）
│   ├── ModelPredict.py              # 预测（复用参考项目框架）
│   ├── Metrics.py                   # 评估指标计算
│   ├── Plotting.py                  # 可视化（训练曲线 + 结果分布 + 事件级）
│   ├── InspectH5.py                 # H5 schema 检查工具
│   └── Normalizer.py                # 归一化工具
│
├── scripts/
│   ├── inspect_h5.sh                # H5 schema 检查
│   ├── train.sh                     # 训练启动
│   ├── predict.sh                   # 预测启动
│   └── eval.sh                      # 评估启动
│
└── output/                          # 训练输出（自动创建）
    └── {mission_name}/
        ├── checkpoints/
        ├── plots/
        └── predict_results/
```

---

## Phase 0：数据理解与工具

### 0.1 H5 Schema 检查工具 — `InspectH5.py`

- 复用参考项目 `DataLoader.py` 的 H5 读取方式
- 打印顶层 keys、shape、dtype
- 随机抽取事件检查字段内容
- 统计 CD/WP hit 分布
- 统计时间范围、电荷分布
- 检查真值端点是否在球面上（||p|| ≈ R）
- 输出文本 schema 报告

### 0.2 Geometry 模块 — `Geometry.py`

从参考项目 `DataLoader.py` 中提取 `DualPMTPositionLookup` 类：
- 加载 `PMTInfo.csv`（CD）和 `PMTPos_WP_LPMT.csv`（WP）
- `get_positions_batch(copynos) → (N, 3)` 向量化查表
- `get_subsystem_tags(copynos) → ['CD'|'WP']`
- 返回单位方向向量 `u = xyz / ||xyz||`
- 新增：返回球面坐标 (theta, phi) 供 HEALPix 映射使用

### 0.3 单事件读取 Demo

用 `InspectH5.py` + `Geometry.py` 读取 1 个事件，验证：
- copyno → xyz 映射正确
- CD/WP 分离正确
- 真值端点归一化后 ||u|| ≈ 1.0
- 时间范围合理

**关键文件**：`InspectH5.py`, `Geometry.py`, `scripts/inspect_h5.sh`

---

## Phase 1：最小可训练闭环

### 1.1 配置系统 — `Config.py` + `configs/default.yaml`

采用 YAML 配置 + CLI 覆盖（与参考项目 argparse 风格兼容）：

```yaml
# configs/default.yaml
mission_name: "ht_transformer_v1"
output_path: "../output"
detector_type: "CDWP"

data:
  h5_path: "output_cdwp_train_10459.h5"
  h5_val_path: null                    # null 表示从 h5_path 自动 split
  geometry_cd: "../../PMTInfo.csv"
  geometry_wp: "../muon_track_reco_transformer/PMTPos_WP_LPMT.csv"
  h5_key_map:                          # h5 键名映射（以实际 h5 为准）
    copyno: "copyno"
    charge: "charge"
    time: "time"
    nhits: "nhits"
    enter_x: "tt_enter_x"
    enter_y: "tt_enter_y"
    enter_z: "tt_enter_z"
    exit_x: "tt_exit_x"
    exit_y: "tt_exit_y"
    exit_z: "tt_exit_z"
    track_time: "track_time"
  max_hits: 20012
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  nside: 8
  num_time_bins: 32
  t_max: null                           # null = 自动从数据估计 p99
  auto_t_max: true

model:
  d_model: 128
  num_layers: 4
  num_heads: 4
  d_ff: 512
  num_queries: 2
  num_global_tokens: 8
  cd_knn_k: 16
  abs_posenc: "fourier"
  rel_posenc: "bucket"
  patch_time_encoder: "conv1d"
  num_rpe_angle_buckets: 64
  num_rpe_time_buckets: 64
  dropout: 0.1

loss:
  lambda_ang: 1.0
  lambda_len: 0.5
  lambda_dir: 0.25

train:
  optimizer: "adamw"
  lr: 3.0e-4
  weight_decay: 1.0e-2
  scheduler: "cosine_with_warmup"
  warmup_steps: 1000
  precision: "bf16"
  dropout: 0.1
  grad_clip: 1.0
  batch_size: 4
  num_epochs: 200
  save_every: 50
  eval_every: 10
  use_accelerate: false

augmentation:
  use_rotation_aug: false
  use_time_jitter: false
  use_drop_hits: false
  use_noise_injection: false
```

`Config.py` 实现：
- `load_config(yaml_path)` 加载 YAML
- CLI argparse 参数覆盖 YAML 字段（兼容参考项目风格）
- 验证必填字段和值域

### 1.2 归一化工具 — `Normalizer.py`

复用参考项目 `DataLoader.py` 中 `Normalizer` 类的逻辑：

- `normalize_charge(charge, apply_log=True)`: log10(q+1) → clip p99 → min-max [0,1]，CD/WP 分别归一化
- `normalize_time(time, t_max)`: clip [0, t_max] → 除以 t_max
- `normalize_position(xyz)`: 除以 PMT_RADIUS (19433.975 mm)
- `normalize_endpoint(xyz)`: xyz / ||xyz|| → 单位向量

### 1.3 HEALPix 模块 — `HEALPix.py`

- `pixel_id = hp.ang2pix(nside, theta, phi)` — 将 CD PMT 方向映射到 HEALPix pixel
- 预计算 CD PMT copyno → HEALPix pixel_id 查找表（nside=8 时共 768 个 pixel）
- 预计算 HEALPix pixel 间球面角距 kNN 邻接表：
  - 对所有活跃 pixel 计算角距矩阵
  - 对每个 pixel 取 top-k 近邻
  - 输出 `adj[k] = [邻居 pixel 列表]`，存为固定 tensor
- `build_time_bins(times, charges, t_max, B)` — 为每个 patch 构建 time-bin 序列 s[b]

### 1.4 DataLoader — `DataLoader.py`

**继承参考项目 H5 读取模式**，但 tokenization 完全重写。

#### 数据读取层（复用）

- H5Dataset 类：直接读取 h5，支持 lazy loading + cached file handle
- `_process_event()` 改写为新的 tokenization 逻辑

#### Tokenization 层（新写）

对每个事件：

```python
# 1. 读取 hit 数据
copyno, charge, time, nhits = read_h5_event(...)
# 2. 取前 nhits 个有效 hit
valid_mask = np.arange(len(copyno)) < nhits
copyno = copyno[valid_mask]
charge = charge[valid_mask]
time = time[valid_mask]

# 3. Geometry lookup
positions = geo.get_positions_batch(copyno)       # (N, 3)
unit_vecs = positions / np.linalg.norm(positions, axis=1, keepdims=True)  # (N, 3)
subsystem = geo.get_subsystem_tags(copyno)         # ['CD'/'WP']

# 4. 分离 CD / WP
cd_mask = subsystem == 'CD'
wp_mask = subsystem == 'WP'

# 5. 归一化
# CD/WP charge 分别归一化
# time 统一归一化

# 6. WP tokens: [ux, uy, uz, q, t]  per hit
wp_tokens = stack([unit_vecs[wp_mask], norm_charge[wp_mask], norm_time[wp_mask]], dim=-1)  # (N_wp, 5)

# 7. CD patch tokens:
# 7a. 映射到 HEALPix pixel
cd_pixel_ids = healpix.lookup(cd_copyno)  # (N_cd,)
# 7b. 按 pixel 分组，聚合
# 一级统计量: sumQ, count, t_min, t_mean (charge-weighted)
# 二级 time-bin: s[b] = sum of charge in bin b
# 7c. time-bin 序列通过 Conv1d encoder → patch_time_emb
# 7d. CD patch token = [ux, uy, uz, sumQ, count, t_min, t_mean] + patch_time_emb

# 8. 标签
entry_xyz = [tt_enter_x, tt_enter_y, tt_enter_z]
exit_xyz = [tt_exit_x, tt_exit_y, tt_exit_z]
u1 = entry_xyz / ||entry_xyz||    # 单位向量
u2 = exit_xyz / ||exit_xyz||      # 单位向量
```

#### Batch 输出

```python
{
    'wp_tokens':     Tensor[B, N_wp_max, 5],      # padding to max in batch
    'wp_mask':       Tensor[B, N_wp_max],          # True = padding
    'cd_patch_tokens': Tensor[B, N_cd_patch_max, d_cd_raw],  # padding
    'cd_patch_mask': Tensor[B, N_cd_patch_max],
    'cd_time_bins':  Tensor[B, N_cd_patch_max, B], # time-bin 序列
    'cd_knn_adj':    Tensor[N_total_patches, k],   # 预计算 kNN 邻接
    'u1':            Tensor[B, 3],                 # 标签：入射端点单位向量
    'u2':            Tensor[B, 3],                 # 标签：出射端点单位向量
    'p1':            Tensor[B, 3],                 # 原始 xyz（用于后处理/可视化）
    'p2':            Tensor[B, 3],
}
```

### 1.5 Token Projector — `TokenProjector.py`

```python
class WPProjector(nn.Module):
    """WP hit token [ux,uy,uz,q,t] → d_model"""
    Linear(5, d_model)

class CDProjector(nn.Module):
    """CD patch token [stats + time_emb] → d_model"""
    # 一级统计量: Linear(6, d_stats)
    # time-bin encoder: Conv1d(B, d_time) → GELU → Conv1d → GELU → GAP → Linear
    # 融合: Linear(d_stats + d_time, d_model)

class TokenTypeEmbedding(nn.Module):
    """4 种类型嵌入: WP, CD, GLOBAL, QUERY"""
    nn.Embedding(4, d_model)

class FourierPositionEncoding(nn.Module):
    """基于单位向量 u=(ux,uy,uz) 的 Fourier features"""
    # 对 u 做 random Fourier features: [sin(w·u), cos(w·u)] → Linear → d_model
```

### 1.6 位置编码 — `PositionEncoding.py`

#### Fourier 绝对 PE

```python
class FourierPE(nn.Module):
    def __init__(self, d_model, num_frequencies=32):
        self.B = nn.Parameter(torch.randn(3, num_frequencies), requires_grad=False)  # 固定随机频率
        self.linear = nn.Linear(num_frequencies * 2, d_model)

    def forward(self, u):
        # u: (..., 3) 单位向量
        proj = u @ self.B                    # (..., num_freq)
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.linear(features)         # (..., d_model)
```

#### Bucket RPE

```python
class BucketRPE(nn.Module):
    def __init__(self, num_buckets=64, num_heads=4):
        self.angle_bias = nn.Embedding(num_buckets * num_heads, 1)
        self.time_bias = nn.Embedding(num_buckets * num_heads, 1)

    def forward(self, angle_ij, delta_t_ij):
        # angle_ij: 球面角距 (B, H, S, S)
        # delta_t_ij: 时间差 (B, H, S, S)
        a_bucket = bucketize(angle_ij, num_buckets)
        t_bucket = bucketize(delta_t_ij, num_buckets)
        return self.angle_bias(a_bucket) + self.time_bias(t_bucket)
```

### 1.7 Model — `Model.py`

#### 整体架构

```
Input:
  wp_tokens (B, N_wp, 5)
  cd_patch_tokens (B, N_cd, d_cd_raw)
  cd_time_bins (B, N_cd, B_bins)

Processing:
  1. WPProjector(wp_tokens) → wp_emb (B, N_wp, d_model)
  2. CDProjector(cd_patch_tokens, cd_time_bins) → cd_emb (B, N_cd, d_model)
  3. 加类型嵌入 (WP/CD)
  4. 加 Fourier 绝对 PE
  5. 拼接 Global tokens (learnable, M=8) + Query tokens (learnable, 2)
  6. 送入 Encoder (L=4 layers):
     - 每层内分区域注意力:
       a. WP↔WP: dense self-attention
       b. WP↔CD: dense cross-attention (双向)
       c. CD↔CD: kNN local self-attention (mask-based)
       d. Global↔All: dense
       e. Query↔All: dense cross-attention
     - 加入 Bucket RPE (角度 + 时间偏置)
     - Pre-LN + FFN
  7. 取 Query token 输出:
     query1 → EndpointHead → 3D vector → normalize → pred_u1
     query2 → EndpointHead → 3D vector → normalize → pred_u2

Output:
  pred_u1 (B, 3), pred_u2 (B, 3)  — 单位向量
```

#### Encoder Layer 实现

```python
class HybridEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, k_cd, num_global, dropout):
        # 注意力组件
        self.wp_self_attn = MultiheadAttention(d_model, num_heads)   # WP↔WP dense
        self.cd_self_attn = MultiheadAttention(d_model, num_heads)   # CD↔CD with kNN mask
        self.wp_cd_cross_attn = MultiheadAttention(d_model, num_heads)  # WP→CD
        self.cd_wp_cross_attn = MultiheadAttention(d_model, num_heads)  # CD→WP
        self.global_attn = MultiheadAttention(d_model, num_heads)     # Global↔All dense
        self.query_attn = MultiheadAttention(d_model, num_heads)      # Query→All dense

        # FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1..norm6 = nn.LayerNorm(d_model)

        # RPE
        self.rpe = BucketRPE(...)

    def forward(self, wp_emb, cd_emb, global_emb, query_emb,
                wp_mask, cd_mask, cd_knn_adj, rpe_angle, rpe_time):
        # 1. WP self-attention (dense)
        wp_out = wp_emb + self.wp_self_attn(wp_emb, wp_emb, wp_emb, wp_mask)

        # 2. CD self-attention (kNN masked)
        cd_knn_mask = build_knn_mask(cd_knn_adj, cd_mask)
        cd_out = cd_emb + self.cd_self_attn(cd_emb, cd_emb, cd_emb,
                                             cd_mask | cd_knn_mask)

        # 3. WP↔CD cross-attention (dense)
        wp_out2 = wp_out + self.wp_cd_cross_attn(wp_out, cd_out, cd_out, cd_mask)
        cd_out2 = cd_out + self.cd_wp_cross_attn(cd_out, wp_out, wp_out, wp_mask)

        # 4. Global↔All dense attention
        all_tokens = cat([wp_out2, cd_out2])
        global_out = global_emb + self.global_attn(global_emb, all_tokens, all_tokens)

        # 5. Query→All dense cross-attention
        all_with_global = cat([wp_out2, cd_out2, global_out])
        query_out = query_emb + self.query_attn(query_emb, all_with_global, all_with_global)

        # 6. FFN
        ... (Pre-LN style)

        return wp_out_final, cd_out_final, global_out, query_out
```

#### Endpoint Head

```python
class EndpointHead(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

    def forward(self, query):
        v = self.mlp(query)             # (B, 3)
        u = F.normalize(v, dim=-1)      # 归一化到单位球面
        return u
```

### 1.8 Loss — `LossFunction.py`

```python
class EndpointLoss(nn.Module):
    def __init__(self, lambda_ang=1.0, lambda_len=0.5, lambda_dir=0.25):
        ...

    def forward(self, pred_u1, pred_u2, gt_u1, gt_u2):
        # 1. 角度损失（稳定的 1-cos 形式）
        L_ang = 0.5 * ((1 - (pred_u1 * gt_u1).sum(-1)) +
                        (1 - (pred_u2 * gt_u2).sum(-1)))

        # 2. 径迹长度约束
        pred_dist = (pred_u2 - pred_u1).norm(dim=-1)
        gt_dist = (gt_u2 - gt_u1).norm(dim=-1)
        L_len = F.smooth_l1_loss(pred_dist, gt_dist)

        # 3. 方向一致性
        pred_dir = F.normalize(pred_u2 - pred_u1, dim=-1)
        gt_dir = F.normalize(gt_u2 - gt_u1, dim=-1)
        L_dir = 1 - (pred_dir * gt_dir).sum(-1).clamp(-1+eps, 1-eps)

        # 总损失
        loss = self.lambda_ang * L_ang.mean() + \
               self.lambda_len * L_len + \
               self.lambda_dir * L_dir.mean()
        return loss, {
            'loss_ang': L_ang.mean().item(),
            'loss_len': L_len.item(),
            'loss_dir': L_dir.mean().item(),
        }
```

### 1.9 训练框架 — `ModelTrain.py`

**复用参考项目 `ModelTrain.py` 的框架**，主要改动：

- 模型构建：使用新的 `Model.py`
- Loss：使用 `EndpointLoss`
- 数据加载：使用新的 `DataLoader.py`
- 保留：Accelerate 多 GPU 支持、checkpoint 保存/加载、日志格式、训练循环结构
- 新增：bf16 混合精度（AMP）、cosine_with_warmup scheduler、gradient clipping
- 输出目录结构：`output/{mission_name}/checkpoints/`, `output/{mission_name}/plots/`

### 1.10 入口 — `RunModule.py`

复用参考项目 `RunModule.py` 的 argparse 模式，新增：

```python
--InspectH5        # 运行 h5 schema 检查
--TrainModel       # 训练（原有）
--Predict          # 预测（原有）
--Eval             # 评估 + 可视化
--config           # YAML 配置文件路径（新增）
```

**关键文件**：`Config.py`, `DataLoader.py`, `HEALPix.py`, `TokenProjector.py`, `PositionEncoding.py`, `Model.py`, `LossFunction.py`, `ModelTrain.py`, `RunModule.py`, `scripts/train.sh`

---

## Phase 2：评估闭环

### 2.1 评估指标 — `Metrics.py`

```python
def compute_metrics(pred_u1, pred_u2, gt_u1, gt_u2):
    # 1. 每端点角度误差 (degrees)
    ang_err1 = arccos(clamp(pred_u1 · gt_u1, -1, 1)) * 180/pi
    ang_err2 = arccos(clamp(pred_u2 · gt_u2, -1, 1)) * 180/pi

    # 2. 平均角度误差
    mean_ang_err = (ang_err1 + ang_err2) / 2

    # 3. 分位数: median, 68%, 95%
    quantiles = compute_quantiles(ang_err, [0.5, 0.68, 0.95])

    # 4. 端点间距误差
    pred_dist = ||pred_u2 - pred_u1||
    gt_dist = ||gt_u2 - gt_u1||
    dist_err = |pred_dist - gt_dist|

    # 5. 方向余弦
    pred_dir = normalize(pred_u2 - pred_u1)
    gt_dir = normalize(gt_u2 - gt_u1)
    dir_cos = pred_dir · gt_dir

    return {metrics_dict}
```

### 2.2 可视化 — `Plotting.py`

#### 训练过程图
- total loss / angle loss / length loss / direction loss vs epoch
- learning rate vs epoch

#### 结果分布图
- endpoint angular error histogram (端点1/端点2/平均)
- angular error violin/box plot
- endpoint distance error distribution
- direction consistency distribution

#### 事件级可视化
- 球面 3D 散点图：PMT hit 热度 + 真值端点 + 预测端点
- 径迹方向箭头

### 2.3 预测 — `ModelPredict.py`

复用参考项目 `ModelPredict.py` 框架：
- 加载 checkpoint
- 批量预测
- 输出：pred_u1, pred_u2（单位向量）+ pred_p1, pred_p2（恢复球面 xyz = u × R）
- 保存为 CSV + NPZ

**关键文件**：`Metrics.py`, `Plotting.py`, `ModelPredict.py`, `scripts/eval.sh`, `scripts/predict.sh`

---

## Phase 3：结构增强与优化（Phase 1 & 2 完成后）

- RPE 消融实验（有/无 RPE 对比）
- Global token 数量消融 (0/4/8)
- B / nside / k 消融
- SDPA / FlashAttention 加速
- 可变长 bucket batching
- 可选数据增强（旋转 / time jitter / drop hits）

---

## 与参考项目的复用/改动对比

| 模块 | 复用 | 改动/新写 | 原因 |
|------|------|-----------|------|
| RunModule.py | 框架/argparse | 新增 --config, --InspectH5, --Eval | 扩展入口 |
| Config.py | — | 全新 | 原项目无 YAML 配置 |
| DataLoader.py | H5 读取、DualPMTPositionLookup | tokenization 完全重写 | CD HEALPix patch + WP hit 混合 token |
| Geometry.py | 提取自 DataLoader.py | 独立模块 | 解耦 |
| Normalizer.py | 提取自 DataLoader.py | 独立模块 | 解耦 |
| HEALPix.py | — | 全新 | 参考项目无 patch 机制 |
| Model.py | Pre-LN 结构/FFN 参考 | 完全新写 | Hybrid attention + query + global token |
| TokenProjector.py | — | 全新 | WP/CD 独立 projector |
| PositionEncoding.py | — | 全新 | Fourier PE + Bucket RPE |
| LossFunction.py | 参考 line_para_loss | 完全新写 | 有序双端点 + 三项损失 |
| ModelTrain.py | 训练循环框架 | 改 loss/模型/数据 | 适配新模型 |
| ModelPredict.py | 预测框架 | 改输出格式 | 单位向量 → 球面坐标 |
| Metrics.py | — | 全新 | 参考项目无此独立模块 |
| Plotting.py | 参考 ModelTrainingPlotter | 改可视化内容 | 双端点可视化 |

---

## 实施顺序（文件创建顺序）

```
Phase 0 (约 3 个文件):
  1. python/InspectH5.py
  2. python/Geometry.py (从参考项目提取 DualPMTPositionLookup)
  3. python/Normalizer.py (从参考项目提取)

Phase 1 (约 10 个文件):
  4. configs/default.yaml
  5. python/Config.py
  6. python/HEALPix.py
  7. python/DataLoader.py
  8. python/TokenProjector.py
  9. python/PositionEncoding.py
  10. python/Model.py
  11. python/LossFunction.py
  12. python/ModelTrain.py
  13. python/RunModule.py
  14. scripts/train.sh

Phase 2 (约 4 个文件):
  15. python/Metrics.py
  16. python/Plotting.py
  17. python/ModelPredict.py
  18. scripts/eval.sh, scripts/predict.sh
```

---

## 验证方案

### Phase 0 验证
```bash
python python/RunModule.py --InspectH5 --config configs/default.yaml
# 预期：输出完整 schema 报告，CD/WP hit 分布正确，真值端点 ||u|| ≈ 1
```

### Phase 1 验证
```bash
bash scripts/train.sh
# 预期：
# - DataLoader 正确分离 CD/WP，构造 patch token 和 hit token
# - 模型前向不报错，输出 pred_u1, pred_u2 为单位向量
# - Loss 三项均有值且合理（L_ang 初始 ~1.0, L_len 初始 <1.0, L_dir 初始 ~1.0）
# - 训练 loss 在 50 epoch 内下降
# - checkpoint 正确保存
```

### Phase 2 验证
```bash
bash scripts/eval.sh
# 预期：
# - 输出各项评估指标（角度误差、间距误差、方向一致性）
# - 生成训练 loss 曲线图
# - 生成结果分布图（histogram/violin）
# - 生成事件级 3D 可视化

bash scripts/predict.sh
# 预期：
# - 加载 checkpoint，对 h5 做预测
# - 输出 CSV/NPZ 文件
# - pred_u1, pred_u2 ||u|| ≈ 1.0
# - pred_p1, pred_p2 在球面上
```
