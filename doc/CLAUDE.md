# CLAUDE.md

## 项目目标

你需要在现有代码仓库中，实施一个 **面向 JUNO 球形光学探测器的径迹端点重建 Transformer 项目**。本项目的目标不是做通用轨迹点云拟合，而是基于单个事件中的 PMT hit 信息，重建粒子径迹与球面边界的两个交点：

- 第一个端点：沿粒子传播方向首先与球面相交的点
- 第二个端点：沿粒子传播方向随后与球面相交的点

这两个端点由 TT 提供真值，且具有物理顺序，因此本项目必须实现为：

> **有序双端点回归**

而不是无序 set prediction，也不是 Hungarian matching / min-permutation 训练范式。

---

## 你必须遵守的最终技术决策

以下内容已经确定，不要自行改成别的方案：

1. **任务形式固定为有序双端点回归**
   - 输出为两个有序端点
   - 不要改回 permutation-invariant / Hungarian matching
   - Query1 对应第一端点，Query2 对应第二端点

2. **输出形式固定为单位向量**
   - 每个端点先输出 3D 向量，再归一化到单位球面
   - 同时可在后处理阶段恢复为球面上的 3D 坐标
   - 不要用经纬度作为默认监督目标

3. **模型主线固定为 Hybrid Token Transformer**
   - WP：hit-level token
   - CD：HEALPix patch token
   - CD 使用 two-level time 表示
   - CD↔CD 采用球面 kNN 局部注意力
   - 使用 Global Token 做长程交互
   - 不在首版里实现 ISAB 或周期性全局层

4. **时间输入使用“相对触发时间”**
   - 输入 h5 中的 hittime 已经是相对触发时间语义
   - 首版不要再做 `t - t_min` 或 quantile 对齐
   - 仅做 clip 与 normalize

5. **首版基线不做数据增强**
   - 不启用旋转增强
   - 不启用 time jitter
   - 不启用 drop hits / drop patch
   - 不注入噪声
   - 这些增强接口可以预留，但默认关闭

6. **首版模型规模固定为**
   - `d_model = 128`
   - `num_layers = 4`
   - `num_heads = 4`
   - `d_ff = 512`

7. **首版不做不确定性建模**
   - 不实现 vMF loss 作为默认训练路径
   - 不实现 MC Dropout 作为默认输出
   - 首版只输出确定性双端点

---

## 参考项目约束

训练与预测框架必须**参考类似项目代码 `muon_track_reco_transformer`**，优先遵循其已有的：

- 项目目录组织方式
- 训练入口风格
- 配置文件组织方式
- DataLoader / Train / Predict 模块拆分方式
- bash 启动脚本风格
- 日志、模型保存、评估输出风格

### 实施原则

- **优先“在现有类似项目结构上增量修改”**，不要凭空重写成完全不同的新框架。
- 如果 `muon_track_reco_transformer` 中已经存在类似的模块命名，例如：
  - `DataLoader`
  - `ModelTrain`
  - `ModelPredict`
  - `RunModule`
  
  那么请尽量沿用相同或兼容的组织方式。
- 如果原项目已有公共工具模块（配置读取、日志、checkpoint、绘图、训练循环等），优先复用，不要重复造轮子。
- 如果原项目的输出格式、评估脚本、绘图脚本与当前任务不兼容，允许修改，但要明确说明改了哪些接口以及为什么改。

---

## 数据输入说明

我会提供一个示例 h5 文件。该 h5 文件中保存了 **N 个事件样本**。每个事件至少包含：

1. `hit pmt id`（即 `copyno`）
2. `charge`
3. `hittime`
4. TT 提供的径迹真值：
   - 入射点坐标
   - 出射点坐标

### 你必须先做的第一件事

**不要先假设 h5 的 key 名称、shape、dtype 完全符合你的想象。**

在真正写训练代码之前，先实现一个 **h5 schema inspection 工具**，用于：

- 打印顶层 key
- 打印每个数据集或 group 的 shape / dtype
- 随机抽取 1~3 个事件检查字段内容
- 明确单事件是定长存储还是变长存储
- 明确 `copyno / charge / hittime / truth` 的实际键名
- 输出一份 schema 报告，供后续 DataLoader 实现使用

### 对 h5 数据的实现要求

请按“**以实际样例 h5 为准**”的原则写代码，不要硬编码我没有明确给出的字段名。

你可以合理假设最终需要读取出如下逻辑字段：

- `copyno`: 每个 hit 对应的 PMT 编号
- `charge`: 每个 hit 的电荷
- `hittime`: 每个 hit 的相对触发时间
- `entry_point_xyz`: TT 给出的第一端点坐标
- `exit_point_xyz`: TT 给出的第二端点坐标

但这些只是逻辑意义，不代表 h5 内部字段名一定如此。

---

## 几何与 PMT 映射要求

注意：h5 中给的是 hit 的 `copyno`，而模型需要的是 PMT 的空间坐标和子探测器归属。

因此你必须实现或复用一个 **PMT geometry / lookup 模块**，至少支持：

1. `copyno -> PMT xyz`
2. `copyno -> subsystem tag`，即该 PMT 属于：
   - `CD`
   - `WP`
   - 如有其它子系统，先忽略，或按配置筛除

### 重要要求

- **不要硬编码虚假的 PMT 坐标。**
- 如果仓库中已有 geometry 表、PMT map、detector description，就优先复用。
- 如果 `muon_track_reco_transformer` 已有 `copyno -> xyz` 映射链路，则优先直接复用其读取方式。
- 如果当前仓库没有可用的 PMT geometry 文件，请把 geometry 文件路径设计成配置项，并清晰说明需要用户提供。

---

## 真值定义与标签构造

TT 提供的入射点与出射点坐标，构成本项目的监督标签。

### 标签语义

- `u1`: 粒子传播方向上首先与球面相交的点
- `u2`: 粒子传播方向上随后与球面相交的点

也就是说：

- 第一端点和第二端点是**有序的**
- 不要在训练时交换顺序
- 不要对两端点做最小两排列匹配

### 标签处理要求

训练时建议同时保留两种真值表示：

1. 原始球面坐标（xyz）
2. 单位向量标签（unit vector）

标准训练标签使用单位向量：

- `u1 = normalize(entry_point_xyz)`
- `u2 = normalize(exit_point_xyz)`

同时保留原始 xyz，便于：

- 后处理恢复球面坐标
- 作图
- 调试几何一致性

---

## 输入表示与 token 化要求

### 1. WP token

WP 使用 hit-level token，每个 hit 一个 token。

每个 WP token 最少应包含：

- PMT 单位方向向量 `u = (ux, uy, uz)`
- `charge`
- `time`

建议原始特征形式：

```text
[ux, uy, uz, q, t]
```

其中：

- `q = log1p(charge)` 后再做标准化或稳健标准化
- `t` 使用相对触发时间，经 clip / normalize 后输入

### 2. CD token

CD 不直接使用 hit-level token，而使用 HEALPix patch token。

默认设置：

- `nside = 8`

请将 CD hits 根据 PMT 方向映射到 HEALPix patch，并为每个活跃 patch 构造一个 token。

### 3. CD two-level time 表示

每个 CD patch token 必须同时包含：

#### 一级统计量

- `sumQ`
- `count`
- `t_min`
- `t_mean`（可做 charge-weighted）

#### 二级 time-bin 序列

按固定 bin 数 `B=32` 构造 patch 内时间序列：

```text
s[b] = sum of charge in time-bin b
```

然后将该序列通过一个轻量前端编码为 `patch-time embedding`。

默认实现：

- `Conv1d -> GELU -> Conv1d -> GELU -> GAP -> Linear`

备选实现：

- MLP

但首版默认请做 **1DConv**。

### 4. 统一 token 接口

WP token 与 CD token 必须通过各自独立的 projector 映射到统一的 `d_model` 维度：

- `WPProjector`
- `CDProjector`

不要强行共用同一个 projector。

同时加入类型嵌入：

- `WP`
- `CD`
- `GLOBAL`
- `QUERY`

---

## 时间处理要求

由于当前输入时间已经是相对触发时间，首版必须采用如下策略：

- 不做 `t - t_min`
- 不做 quantile 对齐
- 只做：
  - clip 到 `[0, T_max]` 或按配置给定范围
  - normalize

`T_max` 不要硬编码成奇怪的数值，应该：

- 从配置文件指定，或
- 基于训练集统计自动估计（例如 p99）

但自动估计逻辑必须可关闭。

---

## 位置编码要求

### 绝对位置编码

必须基于单位向量 `u=(ux,uy,uz)`，不要用经纬度直接做默认编码。

默认实现：

- Fourier features on unit vector

备选：

- MLP on unit vector

### 相对位置编码 RPE

必须支持在 attention logits 中加入相对偏置，至少包含：

- 球面角距 `alpha_ij`
- 时间差 `delta_t_ij`

默认实现建议：

- bucket-based learnable bias

即：

- angle bucket table
- time bucket table

RPE 应主要作用于需要结构感知的注意力层，尤其是 CD↔CD 局部交互部分。

---

## Transformer 主干实现要求

### 主干结构

总体结构固定为：

```text
Tokens -> Encoder -> 2 Query Tokens -> Endpoint Heads
```

### 注意力连接规则

请按以下规则实现：

1. `WP ↔ WP`: dense
2. `WP ↔ CD`: dense
3. `CD ↔ CD`: local kNN on sphere
4. `Query ↔ All memory`: dense cross-attention
5. `Global token ↔ All`: dense

### CD 局部注意力

CD patch 间的邻域关系基于球面角距 kNN：

- 默认 `k = 16`
- 可配置候选 `8 / 16 / 32`

如果实现 full sparse attention 太复杂，可以首版先用：

- 显式构造邻接 mask / allowed edges
- 在标准 attention 中通过 mask 限制可见性

先保证正确性，再做高性能优化。

### Global token

首版固定启用 Global Token 做长程拓扑交互：

- `num_global_tokens = 4` 或 `8`
- 默认建议 `8`

不要在首版里实现：

- ISAB
- 周期性全局层
- 更复杂的层次化全局方案

---

## 输出头要求

模型输出两个 query 对应的两个端点。

### 每个端点 head

建议为每个 query 使用独立 head：

```text
query_i
 -> Linear
 -> GELU
 -> Dropout
 -> Linear
 -> 3D vector
 -> Normalize
```

输出：

- `pred_u1`
- `pred_u2`

同时在预测输出中建议提供：

- `pred_u1`, `pred_u2`（单位向量）
- `pred_p1`, `pred_p2`（若球半径已知，则恢复后的球面 xyz）

---

## 损失函数要求

首版损失固定为三项联合：

### 1. 端点角度损失

每个端点做球面方向回归。

评估定义可以使用：

```text
alpha_k = arccos(pred_u_k dot gt_u_k)
```

但训练时建议用更稳定的实现，例如：

```text
L_ang = 0.5 * [(1 - pred_u1·u1) + (1 - pred_u2·u2)]
```

### 2. 端点间距 / 径迹长度约束

必须加入，用于防止两端点塌缩：

```text
L_len = SmoothL1( ||pred_u2 - pred_u1|| , ||u2 - u1|| )
```

### 3. 方向一致性约束

必须加入，用于强化有序回归语义：

```text
pred_d = normalize(pred_u2 - pred_u1)
gt_d   = normalize(u2 - u1)
L_dir  = 1 - pred_d · gt_d
```

### 4. 总损失

默认权重：

```text
L = 1.0 * L_ang + 0.5 * L_len + 0.25 * L_dir
```

### 明确禁止

- 首版不要改成无序双点最小匹配 loss
- 首版不要默认引入 vMF NLL
- 首版不要额外塞很多未经验证的辅助 loss

---

## 训练框架实施要求

请参考 `muon_track_reco_transformer` 的风格，实现或适配以下模块：

### 1. DataLoader

负责：

- 读取 h5
- 解析 schema
- 读取事件 hit 与真值
- 通过 geometry lookup 将 `copyno` 转换为 `xyz + subsystem`
- 路由为 WP/CD
- 生成 WP token 与 CD patch token
- 输出训练所需 batch 字段

请保证 DataLoader 支持：

- 训练集 / 验证集 / 测试集
- padding + mask
- 可变长事件
- 后续可扩展 bucket batching

### 2. ModelTrain

负责：

- 构建模型
- 前向
- loss 计算
- optimizer / scheduler
- checkpoint 保存
- log 输出
- 评估调用

### 3. ModelPredict

负责：

- 加载 checkpoint
- 对输入 h5 做批量预测
- 输出预测结果文件
- 可选导出中间指标与可视化

### 4. RunModule / main entry

应提供统一运行入口，尽量贴近参考项目风格，例如：

- 训练入口
- 预测入口
- 评估入口
- 配置驱动运行

### 5. bash 启动脚本

应提供至少：

- `train.sh`
- `predict.sh`
- `eval.sh`

如果参考项目已有类似脚本，请在其基础上适配，不要完全脱离原风格。

---

## 配置系统要求

请将如下内容做成可配置项，而不是硬编码：

### 数据相关

- h5 路径
- geometry 文件路径
- h5 key 映射
- `T_max`
- `nside`
- `B`
- train/val/test split

### 模型相关

- `d_model`
- `num_layers`
- `num_heads`
- `d_ff`
- `k_cd`
- `num_global_tokens`
- abs PE 类型
- RPE 类型
- patch-time encoder 类型

### 训练相关

- batch size 或 token budget
- learning rate
- weight decay
- warmup steps
- scheduler
- precision
- dropout
- grad clip
- epochs

### 损失相关

- `lambda_ang`
- `lambda_len`
- `lambda_dir`

---

## 评估与画图要求

至少实现以下指标和可视化。

### 指标

1. `endpoint1 angular error`
2. `endpoint2 angular error`
3. `mean angular error`
4. `median / 68% / 95% quantiles`
5. `endpoint distance error`
6. `direction cosine` 或方向误差

### 训练过程图

至少输出：

- total loss curve
- angle loss curve
- length loss curve
- direction loss curve
- learning rate curve

### 结果分布图

至少输出：

- endpoint angular error histogram
- endpoint angular error violin/box plot
- endpoint distance error distribution
- direction consistency distribution

### 事件级可视化

至少实现若干样本的球面/3D 可视化：

- PMT 响应热度
- 真值双端点
- 预测双端点
- 连接线 / 方向箭头

---

## 实施顺序要求

请按下面顺序推进，不要一开始就陷入“过度优化”。

### Phase 0：数据理解

先完成：

- h5 schema 检查工具
- geometry 映射确认
- 单事件读取和可视化
- 真值顺序检查

输出：

- schema report
- 一个最小事件读取 demo

### Phase 1：最小可训练闭环

先做一个能跑通的版本：

- DataLoader
- WP/CD tokenization
- baseline Transformer
- ordered dual-endpoint heads
- 三项 loss
- 基础训练脚本

不要在这个阶段引入复杂加速和花哨增强。

### Phase 2：评估闭环

补齐：

- 指标计算
- 绘图
- 预测导出
- 样本可视化

### Phase 3：结构增强与优化

在 baseline 跑通后，再逐步验证：

- RPE 是否带来收益
- global token 数量影响
- `B / nside / k` 消融
- 性能优化（SDPA / FlashAttention / xFormers）

---

## 高性能实现建议

首版目标是“正确、可复现、可训练”。

因此实现优先级是：

1. 正确读取数据
2. 正确构造 token
3. 正确实现 ordered dual-endpoint loss
4. 正确输出评估结果
5. 然后再做性能优化

在 baseline 稳定之后，可逐步尝试：

- `torch.nn.functional.scaled_dot_product_attention`
- FlashAttention 后端
- xFormers `attn_bias`
- token bucket batching
- variable-length attention
- `torch.compile`

但这些都不应阻碍首版闭环交付。

---

## 代码质量要求

请保证：

- 关键模块有清晰注释
- 不要把大量逻辑塞进一个超长脚本
- 函数和类职责明确
- 配置与代码解耦
- 保留必要的 debug 输出
- 对异常 schema、空事件、非法 hit 做防御性处理

### 必须写的检查项

至少加入以下检查：

- h5 key 缺失时报错信息清晰
- geometry lookup 失败时给出明确报错
- 空事件 / 空 patch 时不要 silent failure
- `pred_u` 归一化前后数值稳定
- `acos` 前要做 clamp
- loss 中除法位置要加 `eps`

---

## 你需要输出的交付物

请最终交付以下内容：

1. 可运行代码
2. 配置文件
3. h5 schema inspection 工具
4. 训练脚本
5. 预测脚本
6. 评估脚本
7. 基础可视化脚本
8. 关键模块说明文档
9. 与 `muon_track_reco_transformer` 相比修改了哪些接口的说明

如果你发现参考项目和当前任务存在明显不一致，例如：

- 原项目输出格式不是双端点
- 原项目评估逻辑只适配单目标
- 原项目绘图逻辑只适配别的真值格式

那么请明确列出：

- 哪些模块需要改
- 改动原因
- 改动后的新输入输出格式

---

## 最终默认配置

请以如下默认配置作为首版启动基线：

```yaml
model:
  d_model: 128
  num_layers: 4
  num_heads: 4
  d_ff: 512
  num_queries: 2
  num_global_tokens: 8
  cd_knn_k: 16
  abs_posenc: fourier
  rel_posenc: bucket
  patch_time_encoder: conv1d

data:
  nside: 8
  num_time_bins: 32
  use_rotation_aug: false
  use_time_jitter: false
  use_drop_hits: false
  use_noise_injection: false
  time_align_mode: none

loss:
  lambda_ang: 1.0
  lambda_len: 0.5
  lambda_dir: 0.25

train:
  optimizer: adamw
  lr: 3e-4
  weight_decay: 1e-2
  scheduler: cosine_with_warmup
  precision: bf16
  dropout: 0.1
  grad_clip: 1.0
```

---

## 明确禁止的事项

以下事情不要擅自做：

1. 不要把任务改成无序双点 set prediction
2. 不要默认启用 Hungarian matching
3. 不要默认引入 vMF / uncertainty head
4. 不要一上来就做复杂数据增强
5. 不要跳过 h5 schema 检查，直接猜字段名
6. 不要在没有 geometry 依据时伪造 `copyno -> xyz`
7. 不要为了“好看”而重写成和参考项目完全不同的工程结构

---

## 一句话总结

你的任务是：

> **基于示例 h5 数据与 `muon_track_reco_transformer` 的现有工程风格，增量实现一套面向 JUNO 的 Hybrid Token Transformer，用于 ordered dual-endpoint trajectory reconstruction；先完成可训练闭环，再补齐评估、绘图和优化。**

