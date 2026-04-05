# HT-Transformer: JUNO CDWP Muon Track Endpoint Reconstruction

## 项目简介

HT-Transformer 是一个面向 JUNO 球形光学探测器（CD + WP 联合）的 **Hybrid Token Transformer**，用于 **有序双端点回归** —— 基于 PMT hit 信息，预测粒子径迹与球面的入射点和出射点。

### 核心特点

- **混合 token 架构**：WP 使用 hit-level token，CD 使用 HEALPix patch token
- **结构化注意力**：CD 局部 kNN 注意力 + WP/CD 交叉注意力 + 全局 token
- **有序双端点回归**：两个 Query Token 分别预测有序的两个端点（入射点 / 出射点）
- **三项联合损失**：角度损失 + 径迹长度约束 + 方向一致性约束
- **多 GPU 训练**：基于 accelerate 库支持 DDP 多卡训练，bf16 混合精度
- **SDPA 加速**：5/6 attention 调用使用 `F.scaled_dot_product_attention`（FlashAttention 后端）

---

## 快速开始

### 环境依赖

```bash
pip install -r requirements.txt
```

主要依赖：

| 包 | 版本 | 用途 |
|---|---|---|
| torch | >= 2.0 | 深度学习框架 |
| h5py | >= 3.0 | H5 数据读取 |
| healpy | >= 1.15 | HEALPix patch 映射 |
| pyyaml | >= 5.0 | 配置文件 |
| loguru | >= 0.6 | 日志 |
| tqdm | >= 4.60 | 进度条 |
| matplotlib | >= 3.4 | 可视化 |
| accelerate | >= 0.20 | 多 GPU 训练 |

### 数据准备

项目需要以下输入数据：

1. **H5 事件文件**：包含 PMT hit 信息（copyno, charge, hittime）和真值端点坐标
2. **PMT Geometry 文件**：
   - CD PMT 位置表（`PMTPos_CD_LPMT.csv`）
   - WP PMT 位置表（`PMTPos_WP_LPMT.csv`）

### 三步运行

#### 第 1 步：数据预处理

将 H5 数据 tokenize 并缓存为 `.pt` 文件，避免每个 epoch 重复 tokenization：

```bash
bash scripts/preprocess.sh
```

预处理结果保存在 `output/{mission_name}/preprocessed/` 下，包含：
- `batch_0000.pt` ~ `batch_NNNN.pt`：按批序列化的 tokenized 数据
- `manifest.json`：元信息（事件数、文件数等）

#### 第 2 步：训练

**单 GPU 训练：**

```bash
bash scripts/train.sh
```

**多 GPU 训练（推荐）：**

```bash
bash scripts/train.sh --accelerate --num_processes 2
```

训练过程中会自动检测预处理数据并加载，输出包括：
- tqdm 进度条显示 loss
- 定期保存 checkpoint 到 `output/{mission_name}/checkpoints/`
- 定期评估并保存指标图到 `output/{mission_name}/plots/`

#### 第 3 步：预测与评估

```bash
# 预测
bash scripts/predict.sh

# 评估（指标 + 可视化）
bash scripts/eval.sh
```

---

## 项目结构

```
HT-Transformer/
├── configs/
│   └── default.yaml             # 默认配置
├── python/
│   ├── RunModule.py             # 统一运行入口
│   ├── Config.py                # 配置加载（YAML + CLI 覆盖）
│   ├── Geometry.py              # PMT 坐标查找（CD + WP 双查找表）
│   ├── Normalizer.py            # 归一化工具
│   ├── HEALPix.py               # HEALPix patch 映射 + kNN 邻接
│   ├── DataLoader.py            # H5 数据加载 + tokenization
│   ├── Preprocess.py            # 预处理管线（H5 → .pt 文件）
│   ├── TokenProjector.py        # WP/CD token 投影 + 类型嵌入
│   ├── PositionEncoding.py      # Fourier 绝对位置编码 + Bucket RPE
│   ├── Model.py                 # HT-Transformer 模型
│   ├── LossFunction.py          # 三项联合损失
│   ├── ModelTrain.py            # 训练循环（accelerate + tqdm）
│   ├── ModelPredict.py          # 批量预测
│   ├── Metrics.py               # 评估指标
│   ├── Plotting.py              # 可视化
│   └── InspectH5.py             # H5 schema 检查工具
├── scripts/
│   ├── preprocess.sh            # 预处理脚本
│   ├── train.sh                 # 训练脚本（支持 --accelerate）
│   ├── predict.sh               # 预测脚本
│   ├── eval.sh                  # 评估脚本
│   └── inspect_h5.sh            # H5 检查脚本
├── doc/
│   ├── Tutorial.md              # 本文件
│   ├── Project.md               # 项目进展记录
│   └── 修订.md                  # 架构修订说明
├── requirements.txt             # Python 依赖
├── PMTPos_CD_LPMT.csv           # CD PMT 位置表
└── PMTPos_WP_LPMT.csv           # WP PMT 位置表
```

### 输出目录结构

训练完成后，输出目录组织如下：

```
output/{mission_name}/
├── checkpoints/
│   ├── best.pth                 # 最优 checkpoint（val loss 最低）
│   ├── final.pth                # 最终 checkpoint
│   └── epoch_N.pth              # 定期保存的 checkpoint
├── plots/
│   └── training_curves.png      # 训练曲线（loss + 指标）
├── preprocessed/
│   ├── batch_0000.pt ~ batch_N.pt
│   └── manifest.json
├── eval_results/
│   ├── metrics.json             # 评估指标
│   └── result_distributions.png # 结果分布图
├── predict_results/
│   └── predictions.npz          # 预测结果
└── training_history.json        # 完整训练历史
```

---

## 模型架构

### 总体流程

```
原始 PMT hit → Geometry 查找 → WP/CD 分离
    │
    ├── WP hits → [ux,uy,uz,q,t] → WPProjector → WP tokens
    │
    └── CD hits → HEALPix 聚合 → CDProjector → CD tokens
                                              ↓
                            + Type Embedding + Fourier Position Encoding
                                              ↓
                            Hybrid Encoder Layers (×N)
                                              ↓
                            Query Tokens → Endpoint Heads → pred_u1, pred_u2
```

### Token 类型

| Token | 粒度 | 数量 | 输入特征 |
|-------|------|------|----------|
| WP | hit-level | ~500-2000/事件 | `[ux, uy, uz, log1p(q), t_norm]` |
| CD | patch-level (HEALPix) | ~768 patches | 统计量 + time-bin 序列 |
| Global | learnable | 8 | 可训练参数 |
| Query | learnable | 2 | 可训练参数 |

### CD Patch Token 构造

CD PMT 通过 HEALPix（nside=8）映射为 ~768 个球面 patch，每个 patch token 包含两级时间表示：

- **一级统计量**：`[ux, uy, uz, sumQ, count, t_min, t_mean]` → Linear → 32d
- **二级 time-bin**：32 bin 电荷分布序列 → Conv1d → GELU → Conv1d → GELU → GAP → 32d
- **融合**：concat → Linear → d_model

### Hybrid Encoder Layer

每层包含 6 个 attention 子步骤：

```
┌─────────────────────────────────────────────────────┐
│                  HybridEncoderLayer                  │
│                                                      │
│  1. WP self-attention      (dense)                   │
│  2. CD self-attention      (kNN, k=16) + RPE         │
│  3. WP↔CD cross-attention  (bidirectional, dense)    │
│  4. Global→All attention   (dense)                    │
│  5. Query→All attention    (dense cross-attn)        │
│  6. FFN (Pre-LN)           (all groups)               │
└─────────────────────────────────────────────────────┘
```

**注意力连接规则：**

| 连接 | 类型 | 备注 |
|------|------|------|
| WP ↔ WP | dense self-attention | 全连接 |
| CD ↔ CD | kNN 局部注意力 | 球面角距 k=16 邻居 |
| WP ↔ CD | dense cross-attention | 双向 |
| Global ↔ All | dense | 全局信息汇聚 |
| Query ↔ All | dense cross-attention | 端点信息聚合 |

**CD 局部注意力**：基于 HEALPix 球面 kNN 邻接关系构建 attention mask，每个 CD patch 只关注其 k 个最近邻 patch。同时加入 Bucket RPE（相对位置编码），基于球面角距和时间差提供可学习的偏置。

### 输出头

两个独立 Endpoint Head，每个接收一个 Query Token：

```
query_token → Linear(d_model, d_model) → GELU → Dropout → Linear(d_model, 3) → Normalize
```

输出两个有序单位向量 `pred_u1`、`pred_u2`，可通过乘以球面半径恢复为 3D 坐标。

### SDPA 优化

除 CD self-attention（需 RPE bias）外，其余 5/6 attention 调用使用 `F.scaled_dot_product_attention`，自动选择 FlashAttention / Memory-Efficient 后端，将显存从 O(N²) 降至 O(N)。

---

## 损失函数

总损失为三项联合：

```
L = λ_ang · L_ang + λ_len · L_len + λ_dir · L_dir
```

| 损失项 | 公式 | 默认权重 | 作用 |
|--------|------|----------|------|
| L_ang | `0.5 * [(1 - pred_u1·gt_u1) + (1 - pred_u2·gt_u2)]` | 1.0 | 端点角度回归 |
| L_len | `SmoothL1(‖pred_u2 - pred_u1‖, ‖gt_u2 - gt_u1‖)` | 0.5 | 防止两端点塌缩 |
| L_dir | `1 - cosine_sim(normalize(pred_u2-pred_u1), normalize(gt_u2-gt_u1))` | 0.25 | 强化有序方向 |

---

## 配置说明

所有参数通过 `configs/default.yaml` 配置，也可通过 CLI 参数覆盖。

### 数据配置

```yaml
data:
  h5_path: "/path/to/h5/data"          # H5 数据路径（目录/文件/glob）
  geometry_cd: "PMTPos_CD_LPMT.csv"     # CD PMT 位置表
  geometry_wp: "PMTPos_WP_LPMT.csv"     # WP PMT 位置表
  nside: 8                              # HEALPix nside（768 patches）
  num_time_bins: 32                     # CD patch time-bin 数量
  t_max: 800.0                          # 时间截断（ns）
  max_hits: 20012                       # 最大 hit 数（截断/补全）
  train_ratio: 0.8                      # 训练集比例
  val_ratio: 0.1                        # 验证集比例
  preprocess_batch_size: 512            # 预处理每批事件数
```

### 模型配置

```yaml
model:
  d_model: 128                          # 模型维度
  num_layers: 4                         # Encoder 层数
  num_heads: 4                          # 注意力头数
  d_ff: 512                             # FFN 中间维度
  num_global_tokens: 8                  # 全局 token 数量
  num_queries: 2                        # Query token 数量（= 端点数）
  cd_knn_k: 16                          # CD kNN 邻居数
  abs_posenc: "fourier"                 # 绝对位置编码类型
  rel_posenc: "bucket"                  # 相对位置编码类型
  patch_time_encoder: "conv1d"          # CD time-bin 编码器
  num_rpe_angle_buckets: 64             # RPE 角度分桶数
  num_rpe_time_buckets: 64              # RPE 时间分桶数
  dropout: 0.1                          # Dropout 率
```

### 训练配置

```yaml
train:
  batch_size: 8                         # 批大小
  num_epochs: 200                       # 训练轮数
  lr: 3.0e-4                            # 学习率
  weight_decay: 1.0e-2                  # 权重衰减
  warmup_steps: 500                     # 预热步数
  precision: "bf16"                     # 混合精度（bf16/fp16/fp32）
  grad_clip: 1.0                        # 梯度裁剪
  use_accelerate: true                  # 启用 accelerate 多 GPU
  gradient_accumulation_steps: 1        # 梯度累积步数
  save_every: 50                        # checkpoint 保存间隔
  eval_every: 10                        # 评估间隔
  seed: 42                              # 随机种子
  num_workers: 16                       # DataLoader 工作进程数
```

### 损失配置

```yaml
loss:
  lambda_ang: 1.0                       # 角度损失权重
  lambda_len: 0.5                       # 长度损失权重
  lambda_dir: 0.25                      # 方向损失权重
```

### CLI 参数覆盖

所有配置项均可通过命令行覆盖，例如：

```bash
python python/RunModule.py --config configs/default.yaml --TrainModel \
    --batch_size 16 --lr 1e-4 --num_layers 6 --d_model 256
```

支持的 CLI 参数：`--h5_path`, `--max_hits`, `--d_model`, `--num_layers`, `--num_heads`, `--d_ff`, `--batch_size`, `--num_epochs`, `--lr`, `--mission_name`, `--output_path`

---

## 使用教程

### 自定义数据集

1. 准备 H5 文件，需包含以下字段（字段名可在配置中映射）：
   - `copyno`：PMT 编号（CD: 0~17611，WP: 50000~52399）
   - `charge`：每个 hit 的电荷
   - `time`：每个 hit 的相对触发时间
   - 真值端点坐标（enter_x/y/z, exit_x/y/z）

2. 修改 `configs/default.yaml` 中的 `data.h5_path` 和 H5 key 映射

3. 按顺序运行预处理 → 训练 → 评估

### 检查 H5 数据

在训练前，可使用 `InspectH5.py` 检查数据格式：

```bash
bash scripts/inspect_h5.sh
```

输出包含：顶层 key、shape、dtype、样例事件内容等。

### 恢复训练

Checkpoint 中保存了完整的训练状态（模型、优化器、调度器），支持断点续训。

### 多 GPU 训练

```bash
# 2 卡训练
bash scripts/train.sh --accelerate --num_processes 2

# 自动检测所有可用 GPU
bash scripts/train.sh --accelerate
```

注意事项：
- 使用 `find_unused_parameters=True`（因 Encoder Layer 中 FFN 存在未使用参数路径）
- 仅主进程输出 tqdm 进度条和模型摘要
- 验证集 loss 通过 `accelerator.gather` 跨 GPU 聚合

---

## 评估指标

| 指标 | 含义 |
|------|------|
| Endpoint Angular Error | 端点方向角度误差（度） |
| Direction Angular Error | 径迹方向角度误差（度） |
| Midpoint Distance | 径迹中点距离误差（mm） |
| p68 / p90 | 68% / 90% 分位数 |

训练过程中每 `eval_every` 个 epoch 自动计算并记录上述指标，同时生成分布图和趋势图。

---

## 默认参数下的模型规模

| 组件 | 参数量 |
|------|--------|
| Token Projectors | ~20K |
| Type Embedding | ~0.5K |
| Position Encoding | ~4K |
| Encoder Layers (×4) | ~3,700K |
| Output Heads | ~33K |
| Learnable Tokens | ~2.6K |
| **Total** | **~3,747K** |

---

## 常见问题

**Q: 显存不足怎么办？**

A: 减小 `batch_size`，或启用 `gradient_accumulation_steps` 模拟更大 batch。当前 batch_size=8 在 24GB 显存下接近上限（因变长 padding 和注意力矩阵）。

**Q: 训练速度慢？**

A: 确保：① 已运行预处理步骤（消除 tokenization 开销）；② 使用 `--accelerate` 多卡训练；③ SDPA 已启用（默认开启，需 PyTorch >= 2.0）。

**Q: 如何更换 H5 数据？**

A: 修改 `configs/default.yaml` 中的 `data.h5_path`，重新运行 `preprocess.sh` 和 `train.sh`。如果 H5 字段名不同，需同步修改 `data` 下的 key 映射。
