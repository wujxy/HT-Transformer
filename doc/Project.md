# HT-Transformer 项目进展

## 项目概述

基于 JUNO 球形光学探测器（CD + WP 联合）的 PMT hit 信息，实现 Hybrid Token Transformer 用于 **有序双端点回归** —— 预测粒子径迹与球面的入射点和出射点。

## 实施阶段与完成情况

### Phase 0: 数据理解 — ✅ 完成

| 模块 | 文件 | 状态 |
|------|------|------|
| H5 Schema 检查工具 | `python/InspectH5.py` | ✅ |
| Geometry 映射模块 | `python/Geometry.py` | ✅ |
| 归一化工具 | `python/Normalizer.py` | ✅ |

**关键发现**：
- H5 中 `copyno` 为带偏移的 PMT 编号：CD 0~17611，WP 50000~52399
- 端点真值在半径 20500mm 的球面上（WP 球面，非 CD 球面 19434mm）
- 时间范围约 1.5~1076ns，p99=620.6ns，默认 t_max=800.0
- CD geometry 文件已切换为 `PMTPos_CD_LPMT.csv`（key=CopyNo）

### Phase 1: 最小可训练闭环 — ✅ 完成

| 模块 | 文件 | 状态 |
|------|------|------|
| 配置系统 | `configs/default.yaml`, `python/Config.py` | ✅ |
| HEALPix patch 映射 | `python/HEALPix.py` | ✅ |
| DataLoader + tokenization | `python/DataLoader.py` | ✅ |
| Token Projector | `python/TokenProjector.py` | ✅ |
| 位置编码 | `python/PositionEncoding.py` | ✅ |
| Hybrid Token Transformer | `python/Model.py` | ✅ |
| 三项损失函数 | `python/LossFunction.py` | ✅ |
| 训练循环 | `python/ModelTrain.py` | ✅ |
| 统一入口 | `python/RunModule.py` | ✅ |

**模型架构**：d_model=128, 4层, 4头, d_ff=512, 8个 Global Token, 2个 Query Token

**已修复的关键问题**：
- 相对导入改为绝对导入（`from .X` → `from X`）
- Padding token 导致 softmax NaN → `nan_to_num(0.0)` 修复
- CD geometry 文件 header 解析：`skiprows=4` + `names` 显式指定列名

### Phase 2: 评估闭环 — ✅ 完成

| 模块 | 文件 | 状态 |
|------|------|------|
| 评估指标 | `python/Metrics.py` | ✅ |
| 可视化 | `python/Plotting.py` | ✅ |
| 预测模块 | `python/ModelPredict.py` | ✅ |
| Shell 脚本 | `scripts/{train,predict,eval}.sh` | ✅ |
| 依赖清单 | `requirements.txt` | ✅ |

### Phase 2.5: 训练中 eval 增强 — ✅ 完成

| 改动 | 文件 | 说明 |
|------|------|------|
| `compute_training_metrics()` | `Metrics.py` | 方向角度误差 + 中点距离 + 端点角度误差 + 分位数 |
| `plot_training_eval_distributions()` | `Plotting.py` | 3张分布图：方向角度、中点距离、EP1/EP2对比 |
| `_predict_val()` | `ModelTrain.py` | 验证集全量推理，收集预测结果 |
| `_eval_and_plot()` | `ModelTrain.py` | 计算重建指标 + 画分布图 + 记录到 history |
| `_plot_training_curves()` 扩展 | `ModelTrain.py` | 3x2 子图，新增方向角度 p68/p90 和中点距离 p68/p90 趋势 |

**训练时 eval_every 触发流程**：
1. `_plot_training_curves()` — loss 曲线 (3x2)
2. `_eval_and_plot()` → `_predict_val()` → `compute_training_metrics()` → `plot_training_eval_distributions()`
3. 日志输出 p68/p90 指标
4. history 中记录 `val_dir_ang_p68`, `val_mid_dist_p68` 等趋势数据

### Phase 3: 结构增强与优化 — ⏸ 未开始

- RPE 消融实验
- Global token 数量消融
- B / nside / k 消融
- SDPA / FlashAttention 加速

## 文件清单

```
HT-Transformer/
├── CLAUDE.md                    # 技术规范（不可违反的约束）
├── CLAUDE_PLAN.md               # 原始实施计划
├── 修订.md                      # 架构修订报告
├── Project.md                   # 本文件：项目进展
├── requirements.txt             # Python 依赖
├── configs/
│   └── default.yaml             # 默认配置
├── python/
│   ├── RunModule.py             # 统一入口
│   ├── Config.py                # YAML + CLI 配置
│   ├── Geometry.py              # PMT 坐标查找
│   ├── Normalizer.py            # 归一化工具
│   ├── HEALPix.py               # HEALPix patch 映射
│   ├── DataLoader.py            # H5 数据加载 + tokenization
│   ├── TokenProjector.py        # WP/CD projector + 类型嵌入
│   ├── PositionEncoding.py      # Fourier PE + Bucket RPE
│   ├── Model.py                 # HT-Transformer 模型
│   ├── LossFunction.py          # 三项损失
│   ├── ModelTrain.py            # 训练循环（含 eval 增强）
│   ├── ModelPredict.py          # 预测模块
│   ├── Metrics.py               # 评估指标
│   ├── Plotting.py              # 可视化
│   └── InspectH5.py             # H5 schema 检查
└── scripts/
    ├── train.sh
    ├── predict.sh
    └── eval.sh
```
