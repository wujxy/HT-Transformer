# HT-Transformer-DA Loss Upgrade Plan

## 1. 文档目标

本文档用于指导 Claude Code 对 `HT-Transformer-DA` 分支进行 **loss 函数升级**。目标不是调整模型主干，而是只围绕 **训练目标、验证指标、早停监控、配置项** 做一次闭环重构，使训练目标更贴合 JUNO muon 径迹重建的真实物理目标。

本次改造只处理 loss 与评估闭环，不处理 CD token 选择策略。

---

## 2. 当前实现的真实情况

### 2.1 当前模型输出

当前模型输出为两个有序端点单位向量：

- `pred_u1`: 第一端点单位向量
- `pred_u2`: 第二端点单位向量

模型头已经对输出做了 `F.normalize(..., dim=-1)`，因此预测结果天然位于单位球面上。

### 2.2 当前 loss

当前 `models/losses/endpoint_loss.py` 只有两项：

- `L_len`: 预测 chord length 与真值 chord length 的 SmoothL1
- `L_dir`: 预测方向与真值方向之间的 `1 - cos`

总 loss 为：

```math
L = \lambda_{len} L_{len} + \lambda_{dir} L_{dir}
```

### 2.3 当前评估与早停

当前训练/评估流程存在以下特点：

1. `metrics/endpoint_metrics.py` 已经计算：
   - `ep1_angle`
   - `ep2_angle`
   - `dir_angle`
   - `midpoint_dist`
2. 但 `engine/trainer.py` 当前趋势记录与 early stopping 主要监控：
   - `val_dir_ang_p68`
3. 当前 `configs/default.yaml` 中：
   - `loss.lambda_len = 0.5`
   - `loss.lambda_dir = 0.5`
   - `train.early_stop_monitor = "val_dir_ang_p68"`

这意味着：

- **训练目标** 主要鼓励“方向正确、长度合理”
- **验证闭环** 主要奖励“方向角 p68 更小”
- **最终真正关心的端点几何误差** 不是训练与早停的主目标

---

## 3. 物理背景下的核心问题

本项目任务不是普通向量回归，而是 **JUNO 球面探测器中的 through-going muon 径迹重建**。

模型要预测的是粒子轨迹与球面的两个交点：

- 入射点 `u1`
- 出射点 `u2`

并且这两个端点是 **有序的**，不能交换。

### 3.1 仅用 direction + length 的问题

对于一个球内 chord：

- 方向约束只能告诉模型“这条线朝哪走”
- 长度约束只能告诉模型“这条 chord 有多长”

但它们 **不能唯一决定 chord 在球体中的具体位置**。

模型可能出现以下退化解：

- 方向看起来很接近真值
- chord length 也接近真值
- 但整条 chord 在球内横向偏移
- 结果是两个端点在球面上的角误差仍然不小

这类错误对 muon track reconstruction 是实质性错误，因为最终需要的是 **entry / exit point 的精确定位**，而不是只有方向接近。

### 3.2 midpoint 的物理意义

对球面 chord 而言：

- chord direction 决定方向
- chord length 决定离球心距离的大小关系
- chord midpoint 决定 chord 在球体中的横向位置

因此，`midpoint_dist` 不是一个附属指标，而是区分“方向对了但整条轨迹偏了”的关键量。

### 3.3 端点本身必须成为主监督

既然最终输出本身就是两个球面端点单位向量，那么训练目标必须直接监督：

- 端点 1 是否对准真值端点 1
- 端点 2 是否对准真值端点 2

不能只靠 chord 的 proxy 几何量间接约束。

---

## 4. 升级目标

将当前 loss 升级为：

> **端点主监督 + 轨迹几何辅助约束 + 有序方向约束 + 轻量长度正则**

设计原则：

1. **端点误差必须成为主项**
2. **midpoint 误差必须进入训练目标**
3. **方向项保留，用于维持 ordered endpoints 的物理一致性**
4. **长度项降级为轻量正则，而不是主目标**
5. **所有子项先做无量纲归一化，再谈权重**
6. **验证与早停目标要与最终任务一致，不再只盯 direction**

---

## 5. 新 loss 设计

### 5.1 总形式

定义新总 loss：

```math
L = \lambda_{ep} L_{ep} + \lambda_{mid} L_{mid} + \lambda_{dir} L_{dir} + \lambda_{len} L_{len}
```

其中：

- `L_ep`: 端点直接监督，主项
- `L_mid`: chord midpoint 位置监督，关键辅助项
- `L_dir`: ordered direction consistency
- `L_len`: chord length regularization

---

### 5.2 Endpoint loss: `L_ep`

定义：

```math
L_{ep} = \frac{1}{2}\left[(1 - \hat u_1 \cdot u_1) + (1 - \hat u_2 \cdot u_2)\right]
```

说明：

- 不要在训练 loss 中直接使用 `arccos`
- 使用 `1 - cos` 作为光滑 surrogate
- 当前模型输出已经归一化，因此可以直接点乘
- 该项是整个 loss 的主项

性质：

- 直接惩罚球面端点偏差
- 与最终评估目标一致
- 对有序双端点任务天然适配

---

### 5.3 Midpoint loss: `L_mid`

设：

```math
\hat m = \frac{\hat u_1 + \hat u_2}{2}, \qquad m = \frac{u_1 + u_2}{2}
```

定义：

```math
L_{mid} = \text{SmoothL1}\left(\frac{\|\hat m - m\|}{1}\right)
```

如果实现使用物理单位，也可写成：

```math
L_{mid} = \text{SmoothL1}\left(\frac{\|R\hat m - Rm\|}{R}\right)
```

二者等价，本质上都是单位球上的无量纲 midpoint 偏差。

说明：

- midpoint loss 用于约束 chord 在球内的横向位置
- 这是修复当前 `length + direction` 退化自由度的关键项
- 该项的重要性通常高于 `L_len`

推荐实现方式：

- 直接使用单位球表示即可，无需强制转成 mm
- 若后续日志需要物理可解释性，再在 metrics 中单独输出 mm 量纲版本

---

### 5.4 Direction loss: `L_dir`

设：

```math
\hat d = \frac{\hat u_2 - \hat u_1}{\|\hat u_2 - \hat u_1\| + \epsilon},
\qquad
 d = \frac{u_2 - u_1}{\|u_2 - u_1\| + \epsilon}
```

定义：

```math
L_{dir} = 1 - \hat d \cdot d
```

说明：

- 保留当前方向约束的思想
- 但它不再是主项
- 该项主要用于：
  - 防止端点顺序学反
  - 保持 through-going track 的方向一致性

注意：

- 训练 loss 中 **不要取绝对值**
- 因为这是 ordered dual-endpoint regression，方向是有符号的
- metrics 中可继续保留 acute-angle 风格统计，但 loss 中必须保留有向性

---

### 5.5 Length loss: `L_len`

定义：

```math
L_{len} = \text{SmoothL1}\left(\frac{|\|\hat u_2 - \hat u_1\| - \|u_2 - u_1\||}{2}\right)
```

说明：

- 单位球上 chord length 最大为 2，因此除以 2 做归一化
- 此项只作为轻量几何正则
- 不应继续与 endpoint / midpoint 同级

---

## 6. 推荐默认权重

第一版默认推荐：

```yaml
loss:
  name: "endpoint_composite"
  lambda_ep: 1.0
  lambda_mid: 0.5
  lambda_dir: 0.25
  lambda_len: 0.05
```

解释：

- `lambda_ep = 1.0`：端点监督作为主锚点
- `lambda_mid = 0.5`：强辅助项，用来消除横向偏移退化
- `lambda_dir = 0.25`：保留 ordered 方向约束
- `lambda_len = 0.05`：只做轻量正则，避免过强主导

不要把 `lambda_len` 提得过高；否则模型会重新偏向“先学 chord 粗几何”，而不是优化端点本体。

---

## 7. 权重搜索策略

### 7.1 不要盲搜四维连续空间

不要一开始做大规模随机搜索或全量训练网格搜索。正确方法是：

1. 固定 `lambda_ep = 1.0`
2. 只搜索 `lambda_mid / lambda_dir / lambda_len`
3. 先做短程代理实验，再做全量训练

### 7.2 第一阶段搜索网格

推荐搜索：

```yaml
lambda_mid: [0.25, 0.5, 1.0]
lambda_dir: [0.10, 0.25, 0.50]
lambda_len: [0.02, 0.05, 0.10]
```

共 27 组。

### 7.3 短程代理实验

每组配置先做短程实验：

- 使用完整训练集的 20%~30%
- 或固定训练 epoch 为 10~20
- 不追求收敛，只看早期排序

从中筛出前 3 组，再用完整配置全量训练。

### 7.4 模型选择指标

不要用 `val_loss` 选模型，也不要再只用 `val_dir_ang_p68`。

定义新的验证选择分数：

```math
S_{val} = \text{mean_ep_ang_p68} + \alpha \cdot \text{mid_dist_norm_p68} + \beta \cdot \text{dir_ang_p68_norm}
```

实际工程中可简化为：

```python
selection_score = mean_ep_ang_p68 + 0.15 * dir_ang_p68 + 0.5 * mid_dist_unit_p68
```

实现建议：

- `mean_ep_ang_p68 = 0.5 * (ep1_ang_p68 + ep2_ang_p68)`
- `mid_dist_unit_p68` 使用单位球 midpoint 偏差的 p68
- `dir_ang_p68` 保留为辅助项

如果不想引入复合分数，也可以把 early stopping 直接改成：

```yaml
train:
  early_stop_monitor: "val_mean_ep_ang_p68"
  early_stop_mode: "min"
```

这是最简化、最稳妥的版本。

---

## 8. 必须新增的 metrics

### 8.1 当前 metrics 的不足

当前 `compute_training_metrics()` 已经有：

- `ep1_ang_*`
- `ep2_ang_*`
- `dir_ang_*`
- `mid_dist_*`（mm）

但还缺两个训练闭环里真正需要的量：

1. `mean_ep_ang_*`
2. `mid_dist_unit_*`

### 8.2 新增指标

新增：

```python
mean_ep_angle = 0.5 * (ep1_angle + ep2_angle)
```

并统计：

- `mean_ep_ang_mean`
- `mean_ep_ang_p50`
- `mean_ep_ang_p68`
- `mean_ep_ang_p90`
- `mean_ep_ang_p95`
- `mean_ep_ang_p99`

新增单位球 midpoint 偏差：

```python
pred_mid_unit = (pred_u1 + pred_u2) / 2.0
gt_mid_unit = (gt_u1 + gt_u2) / 2.0
midpoint_dist_unit = np.linalg.norm(pred_mid_unit - gt_mid_unit, axis=-1)
```

并统计：

- `mid_dist_unit_mean`
- `mid_dist_unit_p50`
- `mid_dist_unit_p68`
- `mid_dist_unit_p90`
- `mid_dist_unit_p95`
- `mid_dist_unit_p99`

### 8.3 训练历史记录

`engine/trainer.py` 中历史曲线至少新增：

- `val_mean_ep_ang_p68`
- `val_mean_ep_ang_p90`
- `val_mid_dist_unit_p68`
- `val_mid_dist_unit_p90`
- `val_selection_score`（如果启用复合分数）

---

## 9. 代码改造要求（逐文件）

## 9.1 `models/losses/endpoint_loss.py`

### 目标

将当前简单的 `EndpointLoss` 升级为支持四项损失的复合损失。

### 必做修改

1. 保留类名 `EndpointLoss`，避免训练器大范围改接口
2. 新增以下初始化参数：

```python
lambda_ep: float = 1.0
lambda_mid: float = 0.5
lambda_dir: float = 0.25
lambda_len: float = 0.05
use_unit_midpoint: bool = True
eps: float = 1e-8
```

3. `forward()` 中计算：
   - `loss_ep`
   - `loss_mid`
   - `loss_dir`
   - `loss_len`
4. 返回统一 `total_loss, loss_dict`
5. `loss_dict` 至少包含：

```python
{
    "loss_total": ...,
    "loss_ep": ...,
    "loss_mid": ...,
    "loss_dir": ...,
    "loss_len": ...,
}
```

### 实现要求

- 不要在 loss 中使用 `arccos`
- 不要在 direction loss 中取绝对值
- 所有分量必须先归一化成可比较量级
- 代码保持批处理张量形式，不要写逐样本 Python 循环

### 参考实现框架

```python
class EndpointLoss(nn.Module):
    def __init__(
        self,
        lambda_ep: float = 1.0,
        lambda_mid: float = 0.5,
        lambda_dir: float = 0.25,
        lambda_len: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.lambda_ep = lambda_ep
        self.lambda_mid = lambda_mid
        self.lambda_dir = lambda_dir
        self.lambda_len = lambda_len
        self.eps = eps

    def forward(self, pred_u1, pred_u2, gt_u1, gt_u2):
        eps = self.eps

        # endpoint cosine loss
        cos1 = (pred_u1 * gt_u1).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
        cos2 = (pred_u2 * gt_u2).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
        loss_ep = 0.5 * ((1.0 - cos1) + (1.0 - cos2))
        loss_ep = loss_ep.mean()

        # midpoint loss on unit sphere
        pred_mid = 0.5 * (pred_u1 + pred_u2)
        gt_mid = 0.5 * (gt_u1 + gt_u2)
        mid_dist = (pred_mid - gt_mid).norm(dim=-1)
        loss_mid = F.smooth_l1_loss(mid_dist, torch.zeros_like(mid_dist))

        # directed track vector loss
        pred_dir = pred_u2 - pred_u1
        gt_dir = gt_u2 - gt_u1
        pred_dir = pred_dir / (pred_dir.norm(dim=-1, keepdim=True) + eps)
        gt_dir = gt_dir / (gt_dir.norm(dim=-1, keepdim=True) + eps)
        dir_cos = (pred_dir * gt_dir).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
        loss_dir = (1.0 - dir_cos).mean()

        # normalized chord length regularizer
        pred_len = (pred_u2 - pred_u1).norm(dim=-1)
        gt_len = (gt_u2 - gt_u1).norm(dim=-1)
        len_err = (pred_len - gt_len).abs() / 2.0
        loss_len = F.smooth_l1_loss(len_err, torch.zeros_like(len_err))

        total = (
            self.lambda_ep * loss_ep
            + self.lambda_mid * loss_mid
            + self.lambda_dir * loss_dir
            + self.lambda_len * loss_len
        )

        return total, {
            "loss_total": float(total.detach().item()),
            "loss_ep": float(loss_ep.detach().item()),
            "loss_mid": float(loss_mid.detach().item()),
            "loss_dir": float(loss_dir.detach().item()),
            "loss_len": float(loss_len.detach().item()),
        }
```

注意：这是框架示例，Claude Code 可以在不改变数学目标的前提下做等价清理。

---

## 9.2 `metrics/endpoint_metrics.py`

### 目标

将当前 metrics 扩展到能支撑：

- endpoint 主导的模型选择
- midpoint 几何监控
- 与新 loss 一致的训练闭环

### 必做修改

在 `compute_training_metrics()` 中新增：

1. `mean_ep_angle`
2. `midpoint_dist_unit`
3. 对这两个量做完整 quantile summary
4. 返回：
   - `mean_ep_ang_p68`
   - `mean_ep_ang_p90`
   - `mid_dist_unit_p68`
   - `mid_dist_unit_p90`

### 推荐新增内容

```python
mean_ep_angle = 0.5 * (ep1_angle + ep2_angle)

pred_mid_unit = (pred_u1 + pred_u2) / 2.0
gt_mid_unit = (gt_u1 + gt_u2) / 2.0
midpoint_dist_unit = np.linalg.norm(pred_mid_unit - gt_mid_unit, axis=-1)
```

并加入 `_quantile_summary(...)`。

### 保留项

以下现有内容保留：

- `ep1_angle`
- `ep2_angle`
- `dir_angle`
- `midpoint_dist`（mm）

不要删除物理单位版本的 midpoint 指标；训练 loss 用单位球量，日志/论文图仍然需要 mm 版本。

---

## 9.3 `engine/trainer.py`

### 目标

使训练器能够正确记录新 loss 分量，并将 early stopping 从“方向优先”改为“端点优先”。

### 必做修改

#### A. loss 初始化

当前：

```python
criterion = EndpointLoss(
    lambda_len=..., 
    lambda_dir=..., 
)
```

改为读取四项参数：

```python
criterion = EndpointLoss(
    lambda_ep=loss_cfg.get("lambda_ep", 1.0),
    lambda_mid=loss_cfg.get("lambda_mid", 0.5),
    lambda_dir=loss_cfg.get("lambda_dir", 0.25),
    lambda_len=loss_cfg.get("lambda_len", 0.05),
)
```

#### B. train / val epoch 日志项

当前只累积：

- `loss_total`
- `loss_len`
- `loss_dir`

必须改成同时累积：

- `loss_total`
- `loss_ep`
- `loss_mid`
- `loss_dir`
- `loss_len`

并写入 `history`：

- `train_ep`
- `train_mid`
- `train_dir`
- `train_len`
- `val_ep`
- `val_mid`
- `val_dir`
- `val_len`

#### C. `_eval_and_plot()`

新增历史记录：

```python
self.history.setdefault('val_mean_ep_ang_p68', []).append(metrics['mean_ep_ang_p68'])
self.history.setdefault('val_mean_ep_ang_p90', []).append(metrics['mean_ep_ang_p90'])
self.history.setdefault('val_mid_dist_unit_p68', []).append(metrics['mid_dist_unit_p68'])
self.history.setdefault('val_mid_dist_unit_p90', []).append(metrics['mid_dist_unit_p90'])
```

如果实现复合选择分数，还要记录：

```python
selection_score = (
    metrics['mean_ep_ang_p68']
    + 0.15 * metrics['dir_ang_p68']
    + 0.5 * metrics['mid_dist_unit_p68']
)
self.history.setdefault('val_selection_score', []).append(selection_score)
```

#### D. early stopping 默认监控项

把默认监控项从：

```yaml
val_dir_ang_p68
```

改成：

```yaml
val_mean_ep_ang_p68
```

若实现复合分数，则可改为：

```yaml
val_selection_score
```

但第一阶段更推荐直接使用 `val_mean_ep_ang_p68`，更简单、更稳定。

#### E. 曲线绘图

`_plot_training_curves()` 需要同步更新：

- loss 曲线从 3 项扩展到 4 项或 5 项
- reconstruction trend 图不再只画 `val_dir_ang_p68 / p90`
- 至少增加 `val_mean_ep_ang_p68 / p90`

推荐：

- 子图 1：Total loss
- 子图 2：Endpoint / Midpoint loss
- 子图 3：Direction / Length loss
- 子图 4：Validation reconstruction metrics trend

---

## 9.4 `configs/default.yaml`

### 目标

把 loss 配置从旧版双项配置升级为新四项配置，并修改 early stop monitor。

### 必做修改

将：

```yaml
loss:
  lambda_len: 0.5
  lambda_dir: 0.5
```

替换为：

```yaml
loss:
  name: "endpoint_composite"
  lambda_ep: 1.0
  lambda_mid: 0.5
  lambda_dir: 0.25
  lambda_len: 0.05
```

并将：

```yaml
train:
  early_stop_monitor: "val_dir_ang_p68"
```

改为：

```yaml
train:
  early_stop_monitor: "val_mean_ep_ang_p68"
```

### 可选新增项

如果想把模型选择逻辑显式写进配置，可加：

```yaml
train:
  selection_score:
    use_composite: false
    dir_weight: 0.15
    mid_unit_weight: 0.5
```

第一阶段建议 `use_composite: false`，优先用 `val_mean_ep_ang_p68`。

---

## 10. 实施顺序

Claude Code 必须按以下顺序执行，不要跳步：

### Step 1
修改 `models/losses/endpoint_loss.py`

### Step 2
修改 `metrics/endpoint_metrics.py`

### Step 3
修改 `engine/trainer.py`

### Step 4
修改 `configs/default.yaml`

### Step 5
完成后检查以下一致性：

1. 训练日志中是否出现：
   - `loss_ep`
   - `loss_mid`
   - `loss_dir`
   - `loss_len`
2. validation history 中是否出现：
   - `val_mean_ep_ang_p68`
   - `val_mean_ep_ang_p90`
   - `val_mid_dist_unit_p68`
   - `val_mid_dist_unit_p90`
3. early stopping 是否真正读取新的 monitor key
4. checkpoint / history json / plot 是否不会因新字段缺失而报错

---

## 11. 验收标准

完成后，满足以下条件才算改造成功：

### 功能层面

1. 训练可以正常启动
2. train / val loss 统计不报错
3. eval 流程不报错
4. history 与 plot 正常生成

### 指标层面

至少能在日志中稳定看到：

- `train_ep`, `val_ep`
- `train_mid`, `val_mid`
- `train_dir`, `val_dir`
- `train_len`, `val_len`
- `val_mean_ep_ang_p68`
- `val_mean_ep_ang_p90`
- `val_mid_dist_unit_p68`
- `val_mid_dist_unit_p90`

### 目标层面

后续模型选择必须优先关注：

1. `val_mean_ep_ang_p68`
2. `val_mean_ep_ang_p90`
3. `val_mid_dist_p68`
4. `val_dir_ang_p68`

而不是继续把 `val_dir_ang_p68` 当作唯一主要目标。

---

## 12. 推荐实验计划

### 实验 A：只替换 loss，不改别的

目的：验证 loss 升级本身是否改善 endpoint 重建。

配置：

```yaml
lambda_ep: 1.0
lambda_mid: 0.5
lambda_dir: 0.25
lambda_len: 0.05
early_stop_monitor: "val_mean_ep_ang_p68"
```

### 实验 B：小网格定权

尝试：

```yaml
lambda_mid: [0.25, 0.5, 1.0]
lambda_dir: [0.10, 0.25, 0.50]
lambda_len: [0.02, 0.05, 0.10]
```

固定：

```yaml
lambda_ep: 1.0
```

筛选依据：

1. `val_mean_ep_ang_p68`
2. `val_mean_ep_ang_p90`
3. `val_mid_dist_p68`

### 实验 C：确认早停策略

对比：

- `early_stop_monitor = val_dir_ang_p68`
- `early_stop_monitor = val_mean_ep_ang_p68`

预期：

- 新监控项应更偏向优化端点本体
- direction 指标可能略有波动，但 endpoint 误差应更匹配任务目标

---

## 13. 明确禁止事项

Claude Code 在执行本方案时，不允许做以下事情：

1. **不要修改模型主干结构**
2. **不要修改 CD / WP token 生成逻辑**
3. **不要把 ordered endpoints 改成 permutation-invariant loss**
4. **不要在训练 loss 中使用 `abs(dir_cos)`**
5. **不要只保留复合 score 而删除原始物理指标**
6. **不要移除现有 midpoint mm 指标**

---

## 14. 最终结论

本次 loss 升级的核心不是简单“多加几项 loss”，而是完成以下闭环：

- 从 **方向/长度 proxy 主导**
- 升级为 **端点位置主导 + chord 几何辅助**
- 并让 **训练 loss、验证指标、早停监控、历史曲线** 四者一致

一句话概括：

> 当前任务的真实优化目标应当是“端点重建最优”，而不是“方向 proxy 最优”。

因此，推荐的首选方案是：

```yaml
loss:
  lambda_ep: 1.0
  lambda_mid: 0.5
  lambda_dir: 0.25
  lambda_len: 0.05

train:
  early_stop_monitor: "val_mean_ep_ang_p68"
```

这是本次改造的默认基线实现。
