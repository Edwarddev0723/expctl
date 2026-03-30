# expctl

> 🧪 **Git-native、config-driven 的實驗控制工具包**，把 tabular ML / Kaggle 專案中常見的實驗治理流程標準化。

`expctl` 專注解決這幾個常見痛點：交叉驗證切分不固定、YAML 設定錯誤太晚才爆、模型入口散落在各腳本、提交檔格式常出錯，以及實驗結果缺少一致的 registry 與摘要。這個套件把上述通用邏輯抽成可重用模組，讓專案特定的 feature engineering、模型 builder 與資料處理邏輯留在上層 repo。

## 📝 Project Title & Description

**expctl** 是一個用於建立可重現、可審計、可擴充機器學習實驗流程的 Python toolkit。

## ✨ Key Features

- 🔍 **Config validation**: 驗證 training、submission generation、experiment registration 三種工作流的 YAML 結構與欄位契約。
- 🔒 **Immutable split artifacts**: 產生並保存固定的 stratified / repeated-stratified fold artifact，降低 CV drift 與資料洩漏風險。
- 🧩 **Plugin-style model dispatch**: 透過 `ModelRegistry` 註冊 `model.type -> builder`，把模型實作與通用 workflow 解耦。
- 📊 **Reusable evaluation helpers**: 內建 `accuracy`、`f1_macro`、`roc_auc` 計算與 fold summary 聚合。
- ✅ **Submission validation**: 檢查 schema、列數、ID 對齊、預測值合法性，提早攔截提交檔錯誤。
- 🗂️ **Experiment registry utilities**: 提供 CSV registry 的讀寫、排序、row upsert、summary 解析與 metadata 推導。
- 🏗️ **Scaffold-ready**: 內附 `templates/expctl_repo/`，可作為新實驗 repo 的起始骨架。

## 🏗️ Architecture / Core Logic

`expctl` 採用「**通用治理邏輯下沉、專案特定邏輯上浮**」的設計：

| Layer | Module | Responsibility |
| --- | --- | --- |
| Config Contract | `expctl.config` | 驗證 YAML config、feature catalog、輸出路徑與 workflow 契約 |
| Split Management | `expctl.splits` | 建立、保存、讀取固定 fold artifact |
| Training Runtime | `expctl.training` | 模型 builder 註冊、fold 選取、summary aggregation |
| Evaluation | `expctl.evaluation` | metric 計算與 submission 檢查 |
| Experiment Governance | `expctl.registry` | registry row 生成、summary 讀取、實驗 metadata 維護 |

核心流程如下：

1. 由專案層提供 YAML config 與 `FeatureCatalog`。
2. 使用 `expctl.config` 在訓練前做結構驗證，提前攔截缺欄位、非法 metric、路徑錯誤與 feature 定義衝突。
3. 使用 `expctl.splits` 建立固定的 split artifact，確保所有實驗共享同一份 fold 定義。
4. 由專案層透過 `ModelRegistry` 註冊模型 builder，training script 只需依 `model.type` 解析模型。
5. 訓練完成後用 `expctl.evaluation` 彙整 metric、檢查 submission，並以 `expctl.registry` 寫回 registry 與 summary metadata。

```text
YAML Config
   -> FeatureCatalog Validation
   -> Fixed Split Artifact
   -> ModelRegistry Resolve
   -> Fold Metrics / Summary
   -> Registry & Submission Checks
```

## ⚙️ Prerequisites

| Item | Requirement |
| --- | --- |
| Python | `>=3.10` |
| Core Packages | `numpy`, `pandas`, `PyYAML`, `scikit-learn` |
| Environment | macOS / Linux / Windows 皆可，只要能建立 Python 虛擬環境 |
| Hardware | 核心套件不需要特殊硬體，CPU 即可 |
| Optional | 若下游專案啟用 `sentence_transformer` 類型的 text vectorizer，需自行安裝對應模型與推論依賴 |

## 🚀 Installation & Setup

```bash
git clone <repo-url> expctl
cd expctl

python -m venv .venv
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .

# Optional: install dev tools
python -m pip install -e ".[dev]"

# Run checks
ruff check .
python -m pytest
```

如果你想拿它當新專案骨架，也可以直接參考：

```bash
templates/expctl_repo/
```

## 💡 Quick Start / Usage

下面示範一個最小可執行流程：建立固定 split、註冊模型、計算 metric，並驗證 submission。

```python
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from expctl.evaluation import compute_metrics, validate_submission
from expctl.splits import build_stratified_folds, get_fold_indices, save_folds
from expctl.training import ModelRegistry

y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
X = np.random.randn(len(y), 3)

fold_artifact = build_stratified_folds(
    y,
    protocol_name="stratified_kfold_v1",
    n_splits=4,
    random_state=42,
)
save_folds(
    fold_artifact,
    Path("artifacts/splits/stratified_kfold_v1.json"),
    overwrite=True,
)

train_idx, valid_idx = get_fold_indices(fold_artifact, fold_id=0)

registry = ModelRegistry()
registry.register(
    "logistic_regression",
    lambda **params: LogisticRegression(**params),
)
model = registry.resolve("logistic_regression")(max_iter=200, random_state=42)

model.fit(X[train_idx], y[train_idx])
y_pred = model.predict(X[valid_idx])
y_proba = model.predict_proba(X[valid_idx])

metrics = compute_metrics(
    y_true=y[valid_idx],
    y_pred=y_pred,
    y_proba=y_proba,
    metric_names=["accuracy", "roc_auc"],
)
print("metrics:", metrics)

test_df = pd.DataFrame({"id": ["a", "b"], "feature": [1.0, 2.0]})
submission_df = pd.DataFrame({"id": ["a", "b"], "prediction": ["0", "1"]})

result = validate_submission(
    test_df=test_df,
    submission_df=submission_df,
    valid_prediction_values=("0", "1"),
)
print("submission valid:", result.is_valid)
```

若你要接入自己的實驗 repo，建議把專案特定邏輯放在上層專案中，例如：

- feature catalog
- model builders
- feature engineering / text processing
- training scripts 與 submission scripts

而把通用治理能力交給 `expctl` 處理。

## 📂 Package Layout

```text
expctl/
├── config/       # config contracts and validators
├── evaluation/   # metrics and submission validation
├── registry/     # experiment registry helpers
├── splits/       # fixed fold artifact management
└── training/     # model registry and fold runtime helpers
```

## 📄 License

MIT
