# HT-Transformer-DS 剩余问题修复清单

> 交由 Claude Code 执行。  
> 目标：把当前 **HT-Transformer-DS v2** 从“架构已迁移但工程未闭环”修到“可稳定训练 / 评估 / 继续迭代”的状态。  
> 执行原则：**按优先级顺序修改，不要并行大改，不要顺手重构无关代码。**

---

## 总体要求

1. **先修 P0 阻塞项**，确保：
   - 代码可导入
   - 旋转增强可运行
   - DeepSphere 的邻接语义正确
   - 独立 `--Eval` 能在 preprocessed 数据模式下运行
2. **再修 P1 正确性与一致性项**，确保：
   - 配置语义与训练逻辑一致
   - early stopping 收尾逻辑正确
   - v1 遗留接口清理完成
3. **最后做 P2 工程增强项**，只做低风险、可验证的改进。
4. 每完成一个条目后，必须做对应验收，不要攒到最后一起测。
5. 除明确要求外，**不要改模型任务定义、loss 形式、输入字段语义、输出格式。**

---

# P0 阻塞项（必须先修）

## P0-1 修复 `Augmentation.py` 语法错误

**文件**：`Augmentation.py`

### 问题
当前 `random_rotation_matrix()` 中 `np.array([...], dtype=np.float32)` 的括号闭合存在语法错误，会导致：
- `Preprocess.py` 顶层导入失败
- `DataLoader.py` 在 `apply_rotation_aug=True` 时失败

### 修改要求
1. 修复 `R = np.array([...], dtype=np.float32)` 的括号/方括号闭合错误。
2. 保持函数签名与返回值不变：
   - 输入：无
   - 输出：`(3, 3)` 的 `np.float32` 旋转矩阵
3. 不要改函数语义，不要引入固定旋转，不要改成确定性版本。

### 验收
执行：
```bash
python -m py_compile Augmentation.py
python -m py_compile Preprocess.py DataLoader.py
```
并额外执行：
```python
from Augmentation import random_rotation_matrix
R = random_rotation_matrix()
print(R.shape, R.dtype)
```
应得到 `(3, 3)` 和 `float32`。

---

## P0-2 让 DeepSphere 真正按 `active pixel ids` 建图

**文件**：`DeepSphere.py`、`Model.py`

### 问题
当前 `DeepSphereBlock._aggregate_neighbors()` 在收到共享邻接表时，直接使用 `neighbor_indices[:N]`。  
这相当于把“当前事件里的第 0..N-1 个活跃 patch”错误地当成“全局 HEALPix 的第 0..N-1 个像素”，没有真正使用 `pixel_ids` 建立当前事件的局部子图。

### 修改目标
让 DeepSphere 的局部聚合严格基于：
- 当前事件真实 active HEALPix pixel id
- 全局 HEALPix 邻接表
- 事件内的 `global_pixel -> local_token_index` 映射

### 修改要求
1. 在 `DeepSphere.py` 中新增 helper，推荐命名之一：
   - `build_local_neighbor_indices(...)`
   - `map_global_to_local_neighbors(...)`
2. 输入建议：
   - `pixel_ids: (B, N_cd)`
   - `full_knn_adj: (npix, k)`
3. 输出建议：
   - `local_neighbor_indices: (B, N_cd, k)`
   - `valid_neighbor_mask: (B, N_cd, k)`
4. 映射逻辑：
   - 先根据每个 batch event 的 `pixel_ids` 建立 `global_pixel -> local_idx`
   - 对全局邻接表中的每个邻居像素，查询是否存在于当前 active set
   - 若存在，写入对应 `local_idx`
   - 若不存在，标记为 invalid，不允许通过 `clamp(0, N-1)` 硬映射到某个 token
5. `_aggregate_neighbors()` 改为：
   - 使用 `local_neighbor_indices + valid_neighbor_mask`
   - 只聚合 valid 邻居
   - 使用有效邻居数做 mean pooling
   - 若某节点没有任何 valid 邻居，则至少退回到 self-feature 路径，不能产生 NaN
6. `DeepSphereEncoder.forward()` 要真正把 `pixel_ids` 传到上述逻辑中。
7. `Model.py` 中 `cd_encoder(cd_emb, pixel_ids=..., neighbor_indices=...)` 的调用保留，但要确保 `pixel_ids` 真正被 DeepSphere 使用。

### 禁止项
- 不允许继续使用“前 N 行邻接表”近似。
- 不允许用 `clamp` 把无效邻居静默映射到合法节点。

### 验收
1. 人工构造一个小例子：
   - 给出 4 个 `pixel_ids`
   - 检查局部邻接索引只引用这 4 个 token 的 local index
2. 对不存在的邻居：
   - `valid_neighbor_mask=False`
   - 不参与 mean pooling
3. 前向传播无 NaN。
4. 打印/断言至少一个 event 中映射前后的邻接关系，确认语义正确。

---

## P0-3 修复 `CDCompression` 后的 padding mask 逻辑

**文件**：`DeepSphere.py`、`Model.py`

### 问题
当前 `CDCompression._healpix_pool()` 里 padding 的 `pixel_ids_padded` 默认初始化为 `0`；但 `0` 是合法 HEALPix pixel id。  
`Model.forward()` 又用 `cd_pixel_ids_fused == 0` 构造 `cd_mask`，会把真实 token 错误当成 padding。

### 修改要求
1. 在 `DeepSphere.py` 的 `_healpix_pool()` 中：
   - 将 `pixel_ids_padded` 的 padding sentinel 改为 `-1`
   - 保留真实 low-res pixel id 原值，包括 `0`
2. 在 `Model.py` 中：
   - 将 `cd_mask = (cd_pixel_ids_fused == 0)` 改成 `cd_mask = (cd_pixel_ids_fused == -1)`
   - 保留当 `cd_pixel_ids_fused is None` 时的兜底逻辑
3. 检查下列使用点都拿的是修正后的 mask：
   - WP↔CD cross-attn
   - `_masked_mean(cd_emb, cd_mask)`
   - global/query 融合

### 验收
1. 构造一个 low-res pixel id 正好为 `0` 的例子，确认它不会被 mask。
2. 真正 padding 的位置应为 `-1` 且被正确 mask。
3. 前向传播输出形状不变。

---

## P0-4 修复独立 `--Eval` 与 preprocessed 数据集不兼容

**文件**：`Metrics.py`、`RunModule.py`、`ModelPredict.py`、`ModelTrain.py`

### 问题
当前 `Metrics.evaluate_model()` 仍访问 `dataloader.dataset.healpix.get_knn_adjacency(...)`。  
但 preprocessed 数据模式下使用的是 `PreprocessedDataset`，没有 `healpix` 属性。  
此外 v2 模型已经把 CD 邻接表作为模型内部 buffer 使用，外部再往 batch 注入 `cd_knn_adj` 已没有必要。

### 修改要求
1. `Metrics.evaluate_model()` 中：
   - 删除对 `dataloader.dataset.healpix` 的依赖
   - 删除向 `batch_gpu` 注入 `cd_knn_adj` 的逻辑
2. `ModelPredict.py` 中：
   - 删除手动构造 `adj_tensor`
   - 删除 `batch_gpu['cd_knn_adj'] = ...`
3. `ModelTrain.py` 中：
   - 删除训练/验证/预测阶段向 batch 手动塞 `cd_knn_adj` 的逻辑
   - 删除 `_get_knn_adj_tensor()` 及其相关无用调用（若完全不再需要）
4. `RunModule.py` 中无需再依赖 dataloader 提供邻接。
5. 如果保留任何兼容字段，必须保证模型前向完全不依赖它。

### 验收
1. 在存在 `preprocessed/manifest.json` 的情况下运行独立评估：
```bash
python RunModule.py --config ... --Eval
```
应可完成并生成：
- `metrics.json`
- `metrics.txt`
- `result_distributions.png`
2. 训练、预测流程不应因删除 `cd_knn_adj` 注入而报错。

---

# P1 正确性与一致性项

## P1-1 清理 v1/v2 混杂的 scheduler 配置语义

**文件**：`Config.py`、`ModelTrain.py`

### 问题
Trainer 实际已经使用 `WarmupPlateauScheduler + EarlyStopping`，但默认配置仍保留：
- `scheduler: 'cosine_with_warmup'`
- `warmup_steps`
等 v1 残留字段，容易误导实验配置。

### 修改要求
1. 在 `Config.py` 默认配置中：
   - 删除或明确废弃 `scheduler: 'cosine_with_warmup'`
   - 删除或明确废弃 `warmup_steps`
2. 保留并注释清楚 v2 实际使用字段：
   - `warmup_epochs`
   - `plateau_factor`
   - `plateau_patience`
   - `plateau_threshold`
   - `plateau_min_lr`
   - `early_stop_patience`
   - `early_stop_monitor`
   - `early_stop_mode`
3. 如果坚持保留 `scheduler` 配置项，则必须在 `ModelTrain.py` 中真正做调度器分派；否则直接统一成 v2 语义。
4. 在训练日志中打印：
   - 当前 scheduler 类型
   - warmup / plateau 的关键参数

### 验收
- 默认配置不再出现与实际训练逻辑冲突的旧字段
- 日志明确显示当前使用的是 warmup+plateau

---

## P1-2 修正 early stopping 触发后的训练收尾逻辑

**文件**：`ModelTrain.py`

### 问题
当前 early stopping 主体已经实现，但训练尾部的最终保存/最终评估应基于**真实停止 epoch**，而不是名义上的 `num_epochs - 1`。

### 修改要求
1. 在 `run()` 中维护：
   - `last_epoch`
   - 或 `stopped_epoch`
2. early stop 触发时：
   - 记录 `stopped_epoch`
   - 日志打印 best metric 与 best epoch
3. 训练尾部：
   - `_save_checkpoint('final.pth')` 保留
   - `_eval_and_plot(...)` 使用真实最后 epoch
   - `training_history.json` 中增加：
     - `stopped_early`
     - `stopped_epoch`
     - `best_epoch`
     - `best_monitor_value`
4. 如果 `best.pth` 与 `final.pth` 不同，日志中明确说明。

### 验收
- 把 `early_stop_patience` 调小做一次短跑测试，确认提前停止
- 最终 history 和日志中的 epoch/monitor 值一致

---

## P1-3 清理 v1 遗留的 `cd_knn_adj` 外部接口与死代码

**文件**：`ModelTrain.py`、`ModelPredict.py`、`Metrics.py`、必要时 `DataLoader.py`

### 问题
v2 已把 CD 邻接表放入模型内部，但训练/预测/评估流程里仍残留外部注入 `cd_knn_adj` 的旧逻辑与辅助函数，属于死代码/混乱接口。

### 修改要求
1. 删除或废弃：
   - `_get_knn_adj_tensor()`
   - 训练/验证/预测/评估中所有 `batch['cd_knn_adj'] = ...`
2. 检查 `DataLoader.py`：
   - docstring 中不要再写“Add shared cd_knn_adj”之类与现状不符的话
3. 检查 `Model.py`：
   - forward docstring 不要再宣称外部 batch 必须提供 `cd_knn_adj`
4. 确保 v2 模型对外接口简化为：
   - WP/CD tokens
   - masks
   - pixel_ids
   - labels

### 验收
- 全项目 grep `cd_knn_adj`，只保留模型内部真正需要的定义/buffer
- 删除后训练、预测、评估均可运行

---

## P1-4 统一 `Preprocess.py` 中旋转增强的注释与真实语义

**文件**：`Preprocess.py`

### 问题
当前你确认的实验需求是：**每个 event 自己重新随机采一个旋转副本**。  
这也是 `H5EndpointDataset.__getitem__()` 的真实行为。  
但 `Preprocess.py` 的注释和 `manifest['rotation_matrices']` 仍然暗示“每次 expansion 对全部事件施加一个固定旋转”。

### 修改要求
1. 修改注释，明确写清当前真实语义：
   - 每个 event 在 `apply_rotation_aug=True` 时独立随机采样一个 SO(3) 旋转
   - expansion 只是重复生成新的随机旋转副本集合
2. 若 `rotation_matrices` 不再具有真实对应关系，则二选一：
   - 删除 `manifest['rotation_matrices']`
   - 或明确改名并说明它只是“expansion 采样记录/占位”，不代表逐 event 实际旋转
3. 不要改动你确认过的“随机 event 副本”语义。

### 验收
- 注释与行为一致
- `manifest.json` 不再对使用者产生误导

---

# P2 工程增强项（在 P0/P1 全部通过后再做）

## P2-1 为 accelerate + preprocessed 数据模式恢复有限 `num_workers` 试验支持

**文件**：`DataLoader.py`、`Preprocess.py`（如需要）

### 背景
当前 `use_accelerate=True` 时仍强制 `num_workers=0`。这很稳，但会影响吞吐。你已经确认训练是直接读取 `.pt` 预处理数据，因此这里值得做一次低风险恢复测试。

### 修改要求
1. 不要默认直接改成高并发。
2. 增加一个显式配置项，例如：
   - `train.accelerate_num_workers`
   - 或 `data.accelerate_num_workers`
3. 当 `use_accelerate=True` 时：
   - 默认仍可保持 0
   - 但若配置非 0，则允许用小值（如 2/4）
4. 保留 `persistent_workers`、`prefetch_factor` 的合理逻辑。

### 验收
- 单卡和 accelerate 模式下都能正常起 loader
- 小 worker 配置不报错

---

## P2-2 为训练集引入 token-count bucketing（可选，但推荐）

**文件**：`DataLoader.py`、必要时新增 sampler 文件

### 背景
当前预处理数据虽然绕开了在线 tokenization，但 `collate_fn` 仍按 batch 内最大 `N_wp / N_cd` 做 padding。若 batch 内样本长度差异大，会浪费大量计算。

### 修改要求
1. 基于样本长度引入轻量 bucketing：
   - 可用 `N_wp + N_cd`
   - 或 `max(N_wp, N_cd)`
2. 优先只对 train loader 生效
3. 不要破坏当前 `collate_fn` 输出格式
4. 若实现复杂，允许作为独立 sampler 模块新增文件

### 验收
- 同一 batch 内长度分布更接近
- 不影响训练正确性
- 日志可选择打印一个 batch 的 token 统计作为 sanity check

---

## P2-3 Predictor 的 `h5_path` 参数要真正生效

**文件**：`ModelPredict.py`

### 问题
`predict(h5_path=...)` 接口虽然保留了参数，但当前内部仍直接依赖配置创建 dataloader；传入的新 `h5_path` 未必真正替换数据源。

### 修改要求
1. 若传入 `h5_path`：
   - 应基于该路径构造用于预测的 dataloader
   - 不要静默继续使用 config 中的 `data.h5_path`
2. 可行方案：
   - 临时复制配置并覆盖 `data.h5_path`
   - 然后再调用 `create_dataloaders(...)`

### 验收
- 显式传入不同的 `h5_path`，预测结果对应的数据源确实改变

---

## P2-4 更新文档与 docstring，消除 v1 描述残留

**文件**：`Model.py`、`DataLoader.py`、`ModelTrain.py`、`Metrics.py`、`RunModule.py`

### 修改要求
1. 更新以下说明，确保与 v2 现状一致：
   - CD 分支已使用 DeepSphere，不再是 CD self-attn + RPE
   - `cd_knn_adj` 不再要求由外部 batch 注入
   - scheduler 为 warmup+plateau，不再是 cosine warmup
   - 旋转增强为 per-event random rotation
2. 不要求写长文档，但至少保证 docstring 不误导。

### 验收
- 随机抽查上述文件，说明与实际代码一致

---

# 建议执行顺序

按以下顺序执行并提交：

1. `Augmentation.py` 语法修复  
2. DeepSphere active pixel 邻接修复  
3. CD compression mask 修复  
4. 删除外部 `cd_knn_adj` 注入并打通 `--Eval`  
5. scheduler 配置语义清理  
6. early stopping 收尾逻辑修复  
7. 旋转增强注释/manifest 语义清理  
8. 可选性能项（workers / bucketing / predictor h5_path / 文档）

---

# 最终验收清单

全部修改完成后，至少做以下检查：

## A. 静态检查
```bash
python -m py_compile *.py
```
应全部通过。

## B. 训练入口检查
```bash
python RunModule.py --config ... --TrainModel
```
至少能完成：
- 模型构建
- dataloader 构建
- 一个 epoch 的训练与验证
- 不出现 NaN / shape mismatch / missing attribute

## C. 预处理入口检查
```bash
python RunModule.py --config ... --Preprocess
```
应可生成新的 `preprocessed/` 数据。

## D. 独立评估检查
```bash
python RunModule.py --config ... --Eval
```
在 preprocessed 数据模式下应可完成并生成评估产物。

## E. 关键语义检查
1. DeepSphere 邻接确实基于 active `pixel_ids`。
2. low-res pixel `0` 不会被误当 padding。
3. WP time bias 仍正常接入 self-attention。
4. scheduler 日志显示 warmup/plateau 状态正确。
5. early stopping 若触发，history 与日志一致。

---

# 禁止项

1. 不要重写整个模型。
2. 不要把 DeepSphere 改回 attention。
3. 不要改 loss 定义。
4. 不要改任务标签含义。
5. 不要在修 bug 时顺手大规模改命名风格或目录结构。
6. 不要引入与本任务无关的新依赖，除非绝对必要。

