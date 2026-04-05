# HT-Transformer-DS 目录重构执行文档（restructure.md）

本文件用于指导 Claude Code 对当前 **HT-Transformer-DS** 项目执行目录结构重构。

目标：
- 将当前所有 `.py` 文件从 `/python` 一级目录的“平铺结构”重构为按职责划分的包结构。
- 清理 `sys.path.insert(...)` 式脚本导入，统一为包内绝对导入。
- 保持现有功能不变，不在本次重构中引入新的模型逻辑修改。
- 重构后，训练 / 预测 / 评估 / 预处理入口都应可运行。

---

## 一、重构原则

### 必须遵守
1. **只做结构重构，不做算法改动。**
   - 不修改模型数学逻辑。
   - 不修改损失定义。
   - 不修改训练策略。
   - 不顺手修 unrelated bug，除非该 bug 是由 import/路径重构直接导致。

2. **统一导入风格为包内绝对导入。**
   目标风格：
   ```python
   from ht_transformer_ds.config.loader import load_config
   from ht_transformer_ds.data.dataset import create_dataloaders
   from ht_transformer_ds.models.ht_transformer import HTTransformer
   ```

3. **删除或最小化 `sys.path.insert(...)`。**
   重构后不得再依赖临时修改 `sys.path` 才能运行主程序。

4. **保留文件职责清晰。**
   - 数据处理归 `data/`
   - 几何和 HEALPix 归 `geometry/`
   - 模型组件归 `models/components/`
   - 损失归 `models/losses/`
   - 训练/预测执行器归 `engine/`
   - 指标归 `metrics/`
   - 可视化归 `visualization/`

5. **所有新目录必须包含 `__init__.py`。**

---

## 二、目标目录结构

将当前 `python/` 目录重构为：

```text
python/
├── ht_transformer_ds/
│   ├── __init__.py
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   └── run.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── loader.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── preprocess.py
│   │   ├── inspect_h5.py
│   │   ├── augmentation.py
│   │   └── normalization.py
│   │
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── detector_geometry.py
│   │   └── healpix_mapper.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ht_transformer.py
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── token_projectors.py
│   │   │   ├── position_encoding.py
│   │   │   ├── wp_time_bias.py
│   │   │   └── deepsphere.py
│   │   └── losses/
│   │       ├── __init__.py
│   │       └── endpoint_loss.py
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── predictor.py
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── endpoint_metrics.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plotting.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── constants.py   # 可选：如不需要可暂不创建
│
├── configs/
│   └── default.yaml
│
└── scripts/
    ├── train.sh
    ├── predict.sh
    ├── eval.sh
    └── preprocess.sh
```

说明：
- 如果当前项目根目录里本来就有 `configs/` 和 `scripts/`，则保持不动。
- 本次核心是重构 `python/` 内部结构。

---

## 三、逐文件迁移清单

按下面映射移动或重命名文件：

| 当前文件 | 新位置 |
|---|---|
| `python/RunModule.py` | `python/ht_transformer_ds/cli/run.py` |
| `python/Config.py` | `python/ht_transformer_ds/config/loader.py` |
| `python/DataLoader.py` | `python/ht_transformer_ds/data/dataset.py` |
| `python/Preprocess.py` | `python/ht_transformer_ds/data/preprocess.py` |
| `python/InspectH5.py` | `python/ht_transformer_ds/data/inspect_h5.py` |
| `python/Augmentation.py` | `python/ht_transformer_ds/data/augmentation.py` |
| `python/Normalizer.py` | `python/ht_transformer_ds/data/normalization.py` |
| `python/Geometry.py` | `python/ht_transformer_ds/geometry/detector_geometry.py` |
| `python/HEALPix.py` | `python/ht_transformer_ds/geometry/healpix_mapper.py` |
| `python/Model.py` | `python/ht_transformer_ds/models/ht_transformer.py` |
| `python/TokenProjector.py` | `python/ht_transformer_ds/models/components/token_projectors.py` |
| `python/PositionEncoding.py` | `python/ht_transformer_ds/models/components/position_encoding.py` |
| `python/WPTimeBias.py` | `python/ht_transformer_ds/models/components/wp_time_bias.py` |
| `python/DeepSphere.py` | `python/ht_transformer_ds/models/components/deepsphere.py` |
| `python/LossFunction.py` | `python/ht_transformer_ds/models/losses/endpoint_loss.py` |
| `python/ModelTrain.py` | `python/ht_transformer_ds/engine/trainer.py` |
| `python/ModelPredict.py` | `python/ht_transformer_ds/engine/predictor.py` |
| `python/Metrics.py` | `python/ht_transformer_ds/metrics/endpoint_metrics.py` |
| `python/Plotting.py` | `python/ht_transformer_ds/visualization/plotting.py` |
| `python/__init__.py` | `python/ht_transformer_ds/__init__.py` |

可选：
- 如果你希望后续统一常量管理，可新建：
  - `python/ht_transformer_ds/utils/constants.py`
- 当前如不想额外重构常量，可以先不启用。

---

## 四、逐模块 import 修改规则

下面是迁移后需要统一替换的 import 规则。

### 4.1 顶层统一规则

**旧风格示例：**
```python
from Config import load_config
from Model import HTTransformer
from DataLoader import create_dataloaders
```

**新风格统一改为：**
```python
from ht_transformer_ds.config.loader import load_config
from ht_transformer_ds.models.ht_transformer import HTTransformer
from ht_transformer_ds.data.dataset import create_dataloaders
```

### 4.2 数据模块相关

将以下旧导入：
```python
from Geometry import DualPMTPositionLookup, PMT_COPYNO_OFFSET
from Normalizer import Normalizer
from HEALPix import HEALPixMapper
from Augmentation import random_rotation_matrix
```
统一替换为：
```python
from ht_transformer_ds.geometry.detector_geometry import DualPMTPositionLookup, PMT_COPYNO_OFFSET
from ht_transformer_ds.data.normalization import Normalizer
from ht_transformer_ds.geometry.healpix_mapper import HEALPixMapper
from ht_transformer_ds.data.augmentation import random_rotation_matrix
```

### 4.3 模型组件相关

将以下旧导入：
```python
from TokenProjector import WPProjector, CDProjector, TokenTypeEmbedding, FourierPositionEncoding
from DeepSphere import DeepSphereEncoder, CDCompression, build_healpix_knn_adjacency, RMSNorm
from WPTimeBias import SignedTimeBucketBias
from PositionEncoding import BucketRPE
```
统一替换为：
```python
from ht_transformer_ds.models.components.token_projectors import (
    WPProjector, CDProjector, TokenTypeEmbedding, FourierPositionEncoding,
    TOKEN_WP, TOKEN_CD, TOKEN_GLOBAL, TOKEN_QUERY,
)
from ht_transformer_ds.models.components.deepsphere import (
    DeepSphereEncoder, CDCompression, build_healpix_knn_adjacency, RMSNorm,
)
from ht_transformer_ds.models.components.wp_time_bias import SignedTimeBucketBias
from ht_transformer_ds.models.components.position_encoding import BucketRPE
```

### 4.4 训练/预测/评估相关

将以下旧导入：
```python
from Model import HTTransformer
from LossFunction import EndpointLoss
from DataLoader import create_dataloaders
from Geometry import DualPMTPositionLookup
from HEALPix import HEALPixMapper
from Metrics import compute_training_metrics
from Plotting import plot_training_eval_distributions
```
统一替换为：
```python
from ht_transformer_ds.models.ht_transformer import HTTransformer
from ht_transformer_ds.models.losses.endpoint_loss import EndpointLoss
from ht_transformer_ds.data.dataset import create_dataloaders
from ht_transformer_ds.geometry.detector_geometry import DualPMTPositionLookup
from ht_transformer_ds.geometry.healpix_mapper import HEALPixMapper
from ht_transformer_ds.metrics.endpoint_metrics import compute_training_metrics
from ht_transformer_ds.visualization.plotting import plot_training_eval_distributions
```

### 4.5 CLI 入口相关

`run.py` 中应改为：
```python
from ht_transformer_ds.config.loader import load_config
```

并且根据 mode 分发：
```python
from ht_transformer_ds.data.inspect_h5 import inspect_h5
from ht_transformer_ds.data.preprocess import preprocess
from ht_transformer_ds.engine.trainer import Trainer
from ht_transformer_ds.engine.predictor import Predictor
from ht_transformer_ds.metrics.endpoint_metrics import evaluate_model, format_metrics
from ht_transformer_ds.visualization.plotting import ...
```

---

## 五、逐文件操作说明

下面按文件列出需要做的事情。

---

### 5.1 `cli/run.py`
**来源**：`RunModule.py`

#### 操作
1. 移动到 `ht_transformer_ds/cli/run.py`
2. 删除以下路径注入代码：
   ```python
   sys.path.insert(...)
   ```
3. 所有导入改为绝对导入。
4. 保留现有 mode 分发逻辑。
5. 如脚本模式运行，需要支持：
   ```bash
   python -m ht_transformer_ds.cli.run --TrainModel --config configs/default.yaml
   ```

#### 验收
- 不再依赖 `sys.path.insert`
- 可以通过 `python -m ht_transformer_ds.cli.run --help` 正常启动 argparse

---

### 5.2 `config/loader.py`
**来源**：`Config.py`

#### 操作
1. 迁移文件。
2. 仅调整模块路径，不改配置逻辑。
3. 如果内部没有 import 其他本地文件，可只改文件名。

#### 验收
- `load_config()` 仍返回原结构
- CLI override 逻辑不变

---

### 5.3 `data/dataset.py`
**来源**：`DataLoader.py`

#### 操作
1. 迁移文件。
2. 更新本地 import：
   - `Geometry` → `ht_transformer_ds.geometry.detector_geometry`
   - `Normalizer` → `ht_transformer_ds.data.normalization`
   - `HEALPix` → `ht_transformer_ds.geometry.healpix_mapper`
   - `Augmentation` → `ht_transformer_ds.data.augmentation`
3. 保持 `H5EndpointDataset`、`collate_fn`、`create_dataloaders` 对外接口不变。
4. 不要在本次重构里顺手修改数据逻辑。

#### 验收
- 文件可导入
- `create_dataloaders()` 可正常构造 dataloader

---

### 5.4 `data/preprocess.py`
**来源**：`Preprocess.py`

#### 操作
1. 迁移文件。
2. 更新 import：
   - `DataLoader` → `ht_transformer_ds.data.dataset`
   - `Augmentation` → `ht_transformer_ds.data.augmentation`
   - `Geometry` / `HEALPix` 对应更新
3. 保持 `preprocess(config)` 接口不变。

#### 验收
- `python -m py_compile python/ht_transformer_ds/data/preprocess.py` 通过
- 顶层导入不报错

---

### 5.5 `data/inspect_h5.py`
**来源**：`InspectH5.py`

#### 操作
1. 迁移文件。
2. 如无本地依赖，仅改路径与可执行入口。
3. 保持 `inspect_h5(h5_path, max_sample_events=3)` 接口不变。

#### 验收
- CLI 入口中可正常调用

---

### 5.6 `data/augmentation.py`
**来源**：`Augmentation.py`

#### 操作
1. 迁移文件。
2. 在迁移时修复已经存在的语法错误。
3. 不改函数语义，保留 `random_rotation_matrix()`。

#### 验收
- `python -m py_compile python/ht_transformer_ds/data/augmentation.py` 通过

---

### 5.7 `data/normalization.py`
**来源**：`Normalizer.py`

#### 操作
1. 迁移文件。
2. 无需改业务逻辑。

#### 验收
- `Normalizer.normalize_time / normalize_charge / normalize_endpoint` 可正常导入调用

---

### 5.8 `geometry/detector_geometry.py`
**来源**：`Geometry.py`

#### 操作
1. 迁移文件。
2. 保持导出对象不变：
   - `DualPMTPositionLookup`
   - `PMT_COPYNO_OFFSET`
   - `PMT_RADIUS`（如有）

#### 验收
- 其他模块原先依赖的类名和常量名仍可导入

---

### 5.9 `geometry/healpix_mapper.py`
**来源**：`HEALPix.py`

#### 操作
1. 迁移文件。
2. 保持 `HEALPixMapper` 对外接口不变。

#### 验收
- 模型训练初始化几何和 HEALPix 时可正常导入

---

### 5.10 `models/ht_transformer.py`
**来源**：`Model.py`

#### 操作
1. 迁移文件。
2. 更新所有组件 import 到 `models/components/...`
3. 保持 `HTTransformer` 类名和 `forward(batch)` 接口不变。
4. 不修改模型逻辑。

#### 验收
- `from ht_transformer_ds.models.ht_transformer import HTTransformer` 正常
- 可实例化模型

---

### 5.11 `models/components/token_projectors.py`
**来源**：`TokenProjector.py`

#### 操作
1. 迁移文件。
2. 保持导出：
   - `WPProjector`
   - `CDProjector`
   - `TokenTypeEmbedding`
   - `FourierPositionEncoding`
   - `TOKEN_WP/TOKEN_CD/TOKEN_GLOBAL/TOKEN_QUERY`
3. 保持 `__all__` 更新后路径正确。

#### 验收
- `ht_transformer.py` 可正常导入

---

### 5.12 `models/components/position_encoding.py`
**来源**：`PositionEncoding.py`

#### 操作
1. 迁移文件。
2. 保持 `BucketRPE` 导出不变。

#### 验收
- 依赖它的文件可正常导入

---

### 5.13 `models/components/wp_time_bias.py`
**来源**：`WPTimeBias.py`

#### 操作
1. 迁移文件。
2. 保持 `SignedTimeBucketBias` 和 `LogTimeBucketBias` 导出不变。

#### 验收
- `ht_transformer.py` 能正常导入并实例化

---

### 5.14 `models/components/deepsphere.py`
**来源**：`DeepSphere.py`

#### 操作
1. 迁移文件。
2. 保持导出：
   - `RMSNorm`
   - `DeepSphereBlock`
   - `DeepSphereEncoder`
   - `CDCompression`
   - `build_healpix_knn_adjacency`
3. 本次目录重构阶段不改数学逻辑。

#### 验收
- `ht_transformer.py` 可正常导入这些类和函数

---

### 5.15 `models/losses/endpoint_loss.py`
**来源**：`LossFunction.py`

#### 操作
1. 迁移文件。
2. 保持 `EndpointLoss` 类名不变。

#### 验收
- 训练器可正常导入 loss

---

### 5.16 `engine/trainer.py`
**来源**：`ModelTrain.py`

#### 操作
1. 迁移文件。
2. 更新 import 到新包路径。
3. 不修改训练逻辑。
4. 保持 `Trainer(config)` 和 `trainer.run()` 用法不变。

#### 验收
- `from ht_transformer_ds.engine.trainer import Trainer` 正常
- 可实例化 Trainer（不要求实际开训）

---

### 5.17 `engine/predictor.py`
**来源**：`ModelPredict.py`

#### 操作
1. 迁移文件。
2. 更新 import 到新包路径。
3. 保持 `Predictor(config, checkpoint_path)` 与 `predict(...)` 接口不变。

#### 验收
- `Predictor` 可导入并实例化

---

### 5.18 `metrics/endpoint_metrics.py`
**来源**：`Metrics.py`

#### 操作
1. 迁移文件。
2. 更新对 loss 的 import 到：
   `ht_transformer_ds.models.losses.endpoint_loss`
3. 保持函数名不变：
   - `compute_endpoint_metrics`
   - `compute_training_metrics`
   - `format_metrics`
   - `evaluate_model`

#### 验收
- CLI eval 流程可导入这些函数

---

### 5.19 `visualization/plotting.py`
**来源**：`Plotting.py`

#### 操作
1. 迁移文件。
2. 保持绘图函数名不变。

#### 验收
- `trainer.py` 和 `cli/run.py` 可正常导入 plotting 函数

---

## 六、`__init__.py` 组织建议

为减少导入路径长度，可以给部分包做轻量 re-export。

### 推荐做法

#### `ht_transformer_ds/models/__init__.py`
```python
from .ht_transformer import HTTransformer
```

#### `ht_transformer_ds/engine/__init__.py`
```python
from .trainer import Trainer
from .predictor import Predictor
```

#### `ht_transformer_ds/models/losses/__init__.py`
```python
from .endpoint_loss import EndpointLoss
```

#### `ht_transformer_ds/metrics/__init__.py`
```python
from .endpoint_metrics import (
    compute_endpoint_metrics,
    compute_training_metrics,
    format_metrics,
    evaluate_model,
)
```

注意：
- 不要过度做 `*` 导出。
- 只导出高频稳定接口。

---

## 七、脚本与运行方式调整

重构完成后，推荐主运行方式改为模块模式：

```bash
python -m ht_transformer_ds.cli.run --TrainModel --config configs/default.yaml
python -m ht_transformer_ds.cli.run --Predict --config configs/default.yaml
python -m ht_transformer_ds.cli.run --Eval --config configs/default.yaml
python -m ht_transformer_ds.cli.run --Preprocess --config configs/default.yaml
```

### 如项目有 shell 脚本
请同步修改 `scripts/*.sh` 中原本对 `RunModule.py` 的调用：

**旧：**
```bash
python python/RunModule.py --TrainModel --config configs/default.yaml
```

**新：**
```bash
python -m ht_transformer_ds.cli.run --TrainModel --config configs/default.yaml
```

如果当前 shell 脚本不在本次任务范围，可先至少在文档中标注需要同步修改。

---

## 八、执行顺序（必须按顺序做）

1. 新建目标目录树与所有 `__init__.py`
2. 移动物理文件到新路径
3. 逐文件修正 import
4. 删除 `RunModule.py` 中的 `sys.path.insert(...)`
5. 修正 shell 脚本中的入口调用（如存在）
6. 运行静态检查与导入检查
7. 提交迁移结果与未解决问题列表

不要一边改 import 一边做功能修改。

---

## 九、验收标准

Claude Code 完成后，至少满足以下验收项：

### 9.1 语法与导入检查
运行以下检查必须通过：

```bash
python -m py_compile $(find python/ht_transformer_ds -name "*.py")
```

### 9.2 包导入检查
以下导入应全部通过：

```python
from ht_transformer_ds.config.loader import load_config
from ht_transformer_ds.data.dataset import create_dataloaders
from ht_transformer_ds.geometry.detector_geometry import DualPMTPositionLookup
from ht_transformer_ds.geometry.healpix_mapper import HEALPixMapper
from ht_transformer_ds.models.ht_transformer import HTTransformer
from ht_transformer_ds.models.losses.endpoint_loss import EndpointLoss
from ht_transformer_ds.engine.trainer import Trainer
from ht_transformer_ds.engine.predictor import Predictor
from ht_transformer_ds.metrics.endpoint_metrics import evaluate_model
from ht_transformer_ds.visualization.plotting import plot_training_curves
```

### 9.3 CLI 入口检查
以下命令至少能正常进入 argparse / 初始化阶段：

```bash
python -m ht_transformer_ds.cli.run --help
python -m ht_transformer_ds.cli.run --InspectH5 --config configs/default.yaml
```

### 9.4 不再依赖 `sys.path.insert`
- `cli/run.py` 中不再使用临时插入路径做本地导入。

---

## 十、禁止项

Claude Code 在本次任务中禁止做以下事情：

1. 不要改模型逻辑
2. 不要改 loss 数学定义
3. 不要改训练超参数默认值
4. 不要顺手修复 DeepSphere 邻接语义 bug
5. 不要顺手改 scheduler 行为
6. 不要更换函数名、类名，除非文件路径改变后必须调整导出
7. 不要修改 `configs/default.yaml` 的内容，除非只是更新运行入口文档引用

本次任务是**目录重构任务**，不是功能修复任务。

---

## 十一、交付内容

Claude Code 最终应交付：

1. 新的目录结构
2. 所有迁移后的 `.py` 文件
3. 修改后的 import
4. 运行方式说明
5. 一份简短的迁移结果报告，至少包含：
   - 已迁移文件列表
   - 已修改 import 的文件列表
   - 是否仍存在旧路径残留引用
   - 验收命令是否通过

