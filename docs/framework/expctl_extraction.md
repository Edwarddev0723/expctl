# `expctl` Extraction 說明

本次抽取把原本散在 `src/` 與 `scripts/` 的通用實驗治理邏輯，整理成可獨立移植的 `expctl/` 子套件。

## 已抽出的通用層

| 能力 | 新位置 | 說明 |
| --- | --- | --- |
| Config validator | `expctl/config/` | 驗證器改為吃 `FeatureCatalog`，不再直接 import 專案 feature 常數 |
| Split manager | `expctl/splits/` | 固定 split artifact 的 build/load/save contract |
| Submission validator | `expctl/evaluation/submission.py` | 改成 competition-schema 可注入 |
| Metrics | `expctl/evaluation/metrics.py` | 通用 metric 計算 |
| Experiment registry | `expctl/registry/` | CSV registry 讀寫、summary path、row upsert |
| Model dispatch | `expctl/training/model_registry.py` | plugin-style model registry |
| Fold runtime helpers | `expctl/training/runner.py` | fold selection、repeat output naming、summary aggregation |

## 保留在專案層的部分

以下仍屬於 `IM5033` 專案特定邏輯，沒有進入框架：

- `src/feature_engineering.py`
- `src/text_features.py`
- `src/training_row_policy.py`
- `src/submission_paths.py`
- `src/models/*` 的具體 builder 實作

## 相容層

現有 import path 仍可繼續使用：

- `src/config_validation.py`
- `src/validation/splitter.py`
- `src/evaluation/submission_checker.py`

這三個模組現在只負責把專案特定 catalog / schema 接到 `expctl`。

## 新增模型的方式

1. 在專案層新增 `src/models/<your_model>.py`
2. 在 [src/models/registry.py](/Users/edwardhuang/Documents/GitHub/IM5033-boy-and-girl-deep/src/models/registry.py) 註冊 `model.type -> builder`
3. 不需要再修改 `scripts/train_baseline.py` 與 `scripts/generate_submission.py`

## 拆成獨立 repo 的建議內容

第一批建議搬出的目錄：

- `expctl/`
- `docs/framework/`
- `templates/expctl_repo/`

保留在 competition repo 的目錄：

- `src/feature_engineering.py`
- `src/text_features.py`
- `src/submission_paths.py`
- `src/training_row_policy.py`
- `src/models/`
- `configs/experiments/`
- `artifacts/`
- `reports/`

## Standalone scaffold

`templates/expctl_repo/` 提供一個最小新 repo 骨架，包含：

- `AGENTS.md` 治理模板
- `pyproject.toml`
- config / registry / decisions 初始檔
- `src/project/` 作為 competition-specific adapter 放置點
