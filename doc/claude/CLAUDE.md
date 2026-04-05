# CLAUDE.md

## HT-Transformer v2 实施指导文件
### 面向 Claude Code AI Agent 的开发约束、实施顺序与验收标准

---

## 0. 文档定位

本文件用于指导 Claude Code AI agent 在当前 **HT-Transformer** 项目基础上实施 **HT-Transformer v2**。

本文件不是概念讨论文档，而是 **执行指导文件**。Claude 必须把它理解为：

- 明确的改造目标；
- 不可违反的架构边界；
- 具体模块级任务列表；
- 编码时的接口约束；
- 分阶段实施顺序；
- 每阶段完成后的验收标准。

Claude 的输出目标不是“提出想法”，而是：

> **在尽量保持现有项目结构、运行入口、训练/预测/评估闭环兼容的前提下，完成 HT-Transformer v2 的工程实现。**

---

## 1. 必须遵守的总体原则

### 1.1 总目标

本次升级的最终目标固定为：

1. 保留整体 **Hybrid Token Transformer** 框架；
2. 将 **CD 分支的 self-attention 整体替换为 DeepSphere block**；
3. 升级 **WP token encoder**；
4. 为 **WP self-attention** 注入 **signed relative time bias**；
5. 保持当前任务定义：**有序双端点回归**；
6. 保持当前输出定义：`pred_u1`, `pred_u2`；
7. 保持当前 loss 设计先不变；
8. 引入 **Warmup + ReduceLROnPlateau + EarlyStopping** 训练调度；
9. 尽量保持 `RunModule.py` 的入口风格、训练/预测/评估流程、预处理流程兼容。

---

### 1.2 不允许偏离的关键决策

Claude **不得**擅自改成以下替代路线：

- 不要把整个模型重写成纯 DeepSphere 网络；
- 不要删除 WP/CD 融合、Global token、Query token；
- 不要把任务改成 midpoint + direction 回归；
- 不要把 loss 直接改成新的多项式组合并替换现有三项联合 loss；
- 不要把 WP 时间结构意识实现成“仅额外追加时间特征”作为主方案；
- 不要把旋转增强改成固定旋转集合；
- 不要重构整个项目目录；
- 不要破坏现有 `Train / Predict / Eval / Preprocess` 入口模式。

允许做的是：

- 在现有目录下新增少量必要文件；
- 在保持兼容的基础上重构 `Model.py` / `ModelTrain.py`；
- 为配置系统增加新字段；
- 为训练过程增加新 scheduler / early stop 逻辑；
- 为数据项增加少量必要字段（如 `cd_pixel_ids`）。

---

### 1.3 默认实现态度

Claude 在实现时必须遵循以下态度：

- **先保证结构正确，再追求极限优化**；
- **先做最小可运行闭环，再做增强**；
- **优先小步替换当前瓶颈模块，而不是大面积推翻重写**；
- **所有新增模块都必须有清晰的输入输出定义**；
- **所有改动尽量可通过配置开关控制**，便于 v1/v2 对照。

---

## 2. 现有项目中的事实基线（Claude 必须理解）

Claude 在动手前必须理解当前项目的关键事实：

1. 当前项目已经具备完整的训练/预测/评估/预处理闭环；
2. 当前训练可直接读取预处理好的 `.pt` 数据；
3. 当前瓶颈主要在 **CD 分支本身**，而不是 H5 在线读取；
4. 当前 CD 分支虽然语义上是局部 kNN attention，但实现上仍保留 full attention 代价；
5. 当前 WP 分支的 token encoder 太浅，仅是单层线性；
6. 当前 WP 时间信息只进入 token 输入，没有真正进入 token-token 关系建模；
7. 当前训练约在 100 epoch 左右开始出现过拟合；
8. 当前旋转增强“每个 event 自己随机采一个旋转副本”是符合实验需求的。

Claude 不能忽略这些事实，不能重复设计一个与项目现状脱节的架构。

---

## 3. v2 的架构决策（固定采用）

---

### 3.1 CD 分支：固定采用路线 B

CD 分支固定采用：

> **将 CD self-attn 整体替换为 DeepSphere block，但保留整体 Hybrid Token 框架。**

具体含义：

- 保留现有 CD patch token 构造；
- 保留现有 `CDProjector` 总体思路；
- 删除旧的 `cd_self_attn + cd_rpe + cd_knn_mask` 主训练路径；
- 新增 `DeepSphereEncoder`；
- 新增 `CDCompression`；
- CD 在进入融合层前先独立完成局部球面编码，再进行 token 压缩，然后再参与与 WP/global/query 的融合。

---

### 3.2 WP 分支：固定做两件事

#### 任务 A：增强 WP token encoder

把当前单线性层 WP projector 升级为 **双支路 projector**：

- geometry branch：`[ux, uy, uz]`
- optical-time branch：`[q, t]`
- fusion 到 `d_model`

#### 任务 B：给 WP self-attention 注入时间结构意识

固定采用：

> **signed relative time bias**

即：

- 让 WP self-attn 显式感知 token 间带符号的相对时间差；
- 这是最终主方案；
- 不把“仅增加时间特征列”作为主实现替代。

---

### 3.3 训练调度：固定采用三段式策略

训练调度固定采用：

> **Warmup + ReduceLROnPlateau + EarlyStopping**

原则如下：

- Warmup：训练初期稳定；
- Plateau：中后期根据验证表现自动降 LR；
- EarlyStopping：防止后期继续过拟合。

---

## 4. 推荐的目录级改动

Claude 应尽量只做以下级别的文件改动。

### 4.1 允许新增的文件

建议新增：

```text
python/DeepSphere.py
python/WPTimeBias.py
```

如有必要，也允许新增：

```text
python/Norms.py
```

仅用于封装 RMSNorm。

---

### 4.2 预计需要修改的文件

```text
python/DataLoader.py
python/TokenProjector.py
python/Model.py
python/PositionEncoding.py   # 若把 time bias 放这里
python/ModelTrain.py
python/Config.py
python/Preprocess.py         # 若新增字段需保存
python/RunModule.py          # 仅在确有必要时最小改动
configs/default.yaml         # 或等价配置文件
```

---

### 4.3 尽量不要修改的文件

除非确有必要，否则尽量不要大改：

```text
python/LossFunction.py
python/Metrics.py
python/ModelPredict.py
python/Plotting.py
python/InspectH5.py
```

原因：

- v2 的任务定义和输出定义不变；
- 预测结果结构不变；
- loss 先不变；
- 指标与可视化先尽量复用。

---

## 5. 模块级实施要求

---

## 5.1 `DataLoader.py`

### 目标

保持现有数据读取、CD/WP 分离、CD patch 聚合逻辑总体不变。

### 必做事项

1. 保持现有返回字段兼容；
2. 新增 `cd_pixel_ids` 字段：

```python
'cd_pixel_ids': torch.LongTensor  # (N_cd_patch,)
```

3. 确保该字段能被：
   - 普通 raw H5 路径返回；
   - 预处理 `.pt` 路径保存并恢复；
   - `collate_fn` 正确 pad 与 batch 化。

### 禁止项

- 不要重写整个数据读取流程；
- 不要把现有 tokenization 逻辑推翻重写；
- 不要破坏已有字段名和 batch dict 结构。

### 验收标准

- `H5EndpointDataset.__getitem__()` 仍返回原有字段；
- 新增 `cd_pixel_ids` 后训练/预处理不报错；
- `collate_fn` 后 batch 中该字段维度正确。

---

## 5.2 `TokenProjector.py`

### 目标

保留 `CDProjector`，新增 `WPProjectorV2`。

### 必做事项

#### （1）保留 `CDProjector`

可做极轻量增强，但不能把它推倒重写。

#### （2）新增 `WPProjectorV2`

推荐结构：

```text
geometry branch: [ux, uy, uz] -> MLP
optical-time branch: [q, t]   -> MLP
concat -> fusion MLP -> d_model
```

### 接口要求

输入仍然必须兼容：

```python
wp_tokens: (B, N_wp, 5)
```

输出：

```python
wp_emb: (B, N_wp, d_model)
```

### 禁止项

- 不要修改 WP 原始 token 的字段定义；
- 不要把 WP token 改成其他不兼容格式；
- 不要直接删除旧类但不处理兼容导入。

### 验收标准

- `Model.py` 能无缝切换到 `WPProjectorV2`；
- 维度对齐正常；
- 单 batch 前向不报错。

---

## 5.3 `DeepSphere.py`

### 目标

为 CD 分支提供新的主干模块。

### 必做模块

至少实现以下类：

#### （1）`DeepSphereBlock`

职责：

- 进行局部球面/图邻域特征聚合；
- 使用残差；
- 使用归一化；
- 使用 GELU；
- 不做 full attention。

#### （2）`DeepSphereEncoder`

职责：

- 堆叠多个 `DeepSphereBlock`；
- 输入 `(B, N_cd, D)`；
- 输出 `(B, N_cd, D)` 或等价编码结果。

#### （3）`CDCompression`

职责：

- 将 DeepSphere 编码后的 CD tokens 压缩到适合融合的数量；
- 推荐目标数量在 `64~192` 左右，可配置。

### 推荐实现态度

- 第一版先实现 **简单、稳定、可运行** 的 DeepSphere block；
- 不需要一开始就做过度复杂的多尺度金字塔；
- 但必须给后续扩展留接口。

### 归一化建议

优先在此处使用 **RMSNorm**，如无现成实现则新增轻量实现。

### 禁止项

- 不要把 DeepSphere 模块写成与 batch dict 强耦合、无法复用的碎片代码；
- 不要直接把所有 CD token 原样送入融合阶段而不做压缩；
- 不要保留旧的 CD self-attn 作为主路径同时又引入 DeepSphere，造成冗余双主干。

### 验收标准

- DeepSphereEncoder 可独立单测；
- 输入输出维度稳定；
- 与 `CDProjector` 串联后可前向；
- `CDCompression` 输出 token 数受配置控制；
- 训练主路径不再依赖旧的 `cd_self_attn`。

---

## 5.4 `WPTimeBias.py` 或 `PositionEncoding.py`

### 目标

实现 WP self-attention 的 **signed relative time bias**。

### 推荐类名

```python
class SignedTimeBucketBias(nn.Module):
    ...
```

### 职责

输入：

```python
wp_times: (B, N_wp)
```

输出：

```python
time_bias: (B, H, N_wp, N_wp)
```

### 实现要求

- 使用 bucketized signed delta time；
- 支持可配置 bucket 数；
- 支持 head-aware bias；
- bias 可直接加到 WP self-attention logits 上。

### 为什么必须这样实现

Claude 必须理解：

- 当前 WP 的问题不是没有时间输入；
- 而是没有把时间真正注入 token-token 关系；
- 因此时间偏置要进入 **attention logits**，而不是只停留在 token embedding 侧。

### 禁止项

- 不要把这个模块降级为“多拼几维时间特征”；
- 不要只用 `abs(delta_t)` 而忽略符号；
- 不要把它写成仅支持单头的硬编码模块。

### 验收标准

- `HybridFusionLayer` 能接入该 bias；
- 带 bias / 不带 bias 可通过配置切换；
- 前向图维度正确。

---

## 5.5 `Model.py`

### 目标

重构整体模型，使之符合 v2 架构。

### 必做事项

#### （1）输入投影阶段

- 使用 `WPProjectorV2`；
- 使用现有 `CDProjector`；
- 继续保留 type embedding；
- 继续保留 absolute position encoding。

#### （2）CD modality-specific encoding

新增流程：

```text
CDProjector -> DeepSphereEncoder -> CDCompression
```

#### （3）重写融合层

建议新增或改写为：

```python
class HybridFusionLayer(nn.Module):
    ...
```

该层应包含：

- WP self-attn（带 signed time bias）
- WP→CD cross-attn
- CD→WP cross-attn
- Global→All
- Query→All
- FFN

#### （4）删除旧的 CD self-attn 主路径

以下旧逻辑必须退出主训练路径：

- `cd_self_attn`
- `cd_rpe`
- `cd_knn_mask`
- `_compute_cd_rpe()`
- `_build_cd_knn_mask()`

可以保留旧代码片段一段时间用于对照，但最终主路径不得再依赖它们。

### 输出要求

模型最终输出仍必须保持：

```python
{
    'pred_u1': pred_u1,
    'pred_u2': pred_u2,
}
```

### 禁止项

- 不要改动输出头定义为其他任务形式；
- 不要删除 global token / query token；
- 不要把整个模型改成纯双塔并取消融合。

### 验收标准

- 单 batch 前向通过；
- 训练路径可计算 loss；
- 输出接口与旧版兼容；
- 预测/评估模块在最小改动下可继续使用。

---

## 5.6 `ModelTrain.py`

### 目标

接入新 scheduler 和 early stopping，并保持训练主流程稳定。

### 必做事项

#### （1）把 scheduler 从写死的 cosine warmup 改为可配置分派

必须支持至少以下模式：

- `cosine_with_warmup`（兼容旧版）
- `plateau`（v2 主方案）

#### （2）v2 主方案实现

默认主方案：

```text
Warmup -> ReduceLROnPlateau
```

即：

- 前若干 epoch / steps 使用 warmup；
- warmup 结束后，使用 `ReduceLROnPlateau`；
- 每个 epoch 根据 `val_loss` 调整学习率。

#### （3）新增 EarlyStopping

建议：

- 监控任务相关指标，优先 `val_dir_ang_p68`；
- 若该指标在若干次详细评估中无改善，则 early stop。

### 推荐参数

可先默认：

```yaml
train:
  scheduler: plateau
  warmup_epochs: 5
  plateau_factor: 0.5
  plateau_patience: 5
  plateau_min_lr: 1e-6
  early_stop_patience: 8
  early_stop_monitor: val_dir_ang_p68
```

### 训练逻辑要求

- `val_loss`：每 epoch 都有，可供 plateau 使用；
- `val_dir_ang_p68`：在详细评估阶段更新，可供 early stop 使用；
- 若未做详细评估，则 early stop 不应误触发。

### 禁止项

- 不要把 scheduler 逻辑写得完全耦合，导致以后不能切回旧版；
- 不要让 early stopping 依赖不存在的指标；
- 不要破坏 accelerate / 单卡双路径兼容。

### 验收标准

- 新 scheduler 能运行；
- plateau 模式能根据验证结果调整 LR；
- 早停逻辑生效；
- 单卡和 accelerate 模式不报错。

---

## 5.7 `Config.py` / 配置文件

### 目标

为 v2 增加必要配置项。

### 建议新增配置

```yaml
model:
  use_v2: true
  wp_projector: wp_v2
  wp_time_bias: signed_bucket
  wp_time_bias_num_buckets: 64
  cd_backbone: deepsphere
  cd_deepsphere_layers: 4
  cd_deepsphere_hidden: 256
  cd_compression: healpix_pool
  cd_fusion_tokens: 128
  norm_type: rmsnorm

train:
  scheduler: plateau
  warmup_epochs: 5
  plateau_factor: 0.5
  plateau_patience: 5
  plateau_min_lr: 1e-6
  early_stop_patience: 8
  early_stop_monitor: val_dir_ang_p68
```

### 要求

- 旧配置尽量仍可运行；
- v2 的新增配置有默认值；
- CLI override 如有必要可补充，但不是第一优先级。

### 验收标准

- 不填新字段时项目能回退到默认兼容逻辑；
- 打开 `use_v2=true` 时走新路径。

---

## 5.8 `Preprocess.py`

### 目标

保证新增字段（如 `cd_pixel_ids`）能随着预处理数据一起保存。

### 必做事项

- 若 dataset 新增字段，预处理输出必须包含它；
- `PreprocessedDataset` 读取时必须兼容；
- 不破坏当前随机旋转增强逻辑。

### 明确说明

当前随机旋转行为符合实验需求，因此：

- 不要把它改成固定旋转集；
- 不要修改其“每个 event 各自随机旋转”的主语义。

### 验收标准

- 预处理输出的新字段完整；
- preprocessed dataloader 训练正常。

---

## 6. 分阶段实施顺序（Claude 必须按顺序进行）

Claude 不要一口气同时改完所有模块。
必须按以下阶段推进。

---

### Phase 1：打通最小 v2 结构闭环

目标：

- 新增 `WPProjectorV2`
- 新增 `DeepSphere.py`
- 在 `Model.py` 中替换 CD self-attn 主路径
- 单 batch forward 成功

验收：

- 模型 `forward()` 跑通；
- 输出 `pred_u1/pred_u2` 正常。

---

### Phase 2：接入 WP signed time bias

目标：

- 新增 `SignedTimeBucketBias`
- 接入 WP self-attn
- 保证带 bias 的注意力正常前向

验收：

- 启用/关闭 bias 都能正常训练；
- 张量维度正确。

---

### Phase 3：接入训练调度器与早停

目标：

- 改造 scheduler 分派逻辑
- 接入 plateau
- 接入 early stopping

验收：

- 训练可跑；
- LR 可根据验证表现变化；
- early stop 能触发。

---

### Phase 4：补齐预处理与兼容性

目标：

- 预处理保存新增字段；
- Predict/Eval 保持兼容；
- RunModule 入口不破。

验收：

- `Preprocess / Train / Predict / Eval` 都可运行；
- 旧脚本最小改动即可继续用。

---

## 7. Claude 的代码风格要求

Claude 在写代码时必须遵守：

1. 优先复用当前项目命名风格；
2. 新增类必须有清晰 docstring；
3. 复杂模块必须给出输入输出 shape 注释；
4. 不要写大量神秘 hard-code；
5. 配置项必须集中管理；
6. 不要把实验性分支逻辑散落到多个文件里；
7. 能保留向后兼容时尽量保留；
8. 提交修改时，优先保证训练主路径先可运行。

---

## 8. Claude 的测试要求

每阶段至少做以下测试：

### 最小测试

1. 单 batch dataloader 输出 shape 检查；
2. 模型前向 shape 检查；
3. loss 计算检查；
4. 1 个 epoch 训练 smoke test；
5. Predict/Eval 接口 smoke test。

### 必须检查的具体内容

- `cd_pixel_ids` 是否正确 batch 化；
- CDCompression 后 token 数是否符合配置；
- WP relative time bias 维度是否为 `(B, H, N, N)`；
- scheduler 是否真的在更新 LR；
- early stopping 是否只在有指标时触发。

---

## 9. 任务完成标准

只有同时满足以下条件，Claude 才能认为 v2 实施完成：

1. 项目存在可运行的 v2 主路径；
2. CD self-attn 已被 DeepSphere 主路径替换；
3. WPProjectorV2 已接入；
4. WP signed relative time bias 已接入；
5. Warmup + Plateau + EarlyStopping 已接入；
6. 训练入口、预测入口、评估入口仍可使用；
7. 预处理数据链路仍可工作；
8. 输出结构和 loss 结构保持兼容；
9. 代码中没有留下未接线的伪实现主路径。

---

## 10. Claude 最后的执行提醒

Claude 在实施过程中必须始终记住：

- 这是一次 **以保持性能为前提的结构升级**；
- 不要追求“彻底重写成全新模型”；
- 核心是：
  - **CD 用 DeepSphere 解内部建模瓶颈**；
  - **WP 用更强 projector + 时间偏置增强表征**；
  - **训练用 Plateau + EarlyStopping 解决中后期过拟合**；
- 每做一步都要问自己：
  - 是否保持了现有项目的整体闭环？
  - 是否真的在消除当前瓶颈？
  - 是否引入了不必要的大规模破坏？

如果某个改动会明显破坏现有预测/评估接口，优先选择 **兼容式重构**，而不是推翻旧接口。

---

## 11. 推荐的实施输出形式

Claude 在真正执行代码改造时，建议输出顺序为：

1. 先给出改动计划；
2. 再逐文件实施；
3. 每完成一阶段汇报：
   - 改了哪些文件；
   - 新增了哪些类/配置；
   - 当前能否 forward/train；
   - 还剩什么没做；
4. 最后给出：
   - 改动清单；
   - 关键接口说明；
   - 训练配置建议；
   - 已知风险点。

---

**本文件为 HT-Transformer v2 的 Claude Code 实施指导文件。实现时以本文件和 `HT-Transformer-DS.md` 为最高级别项目内技术约束。**
