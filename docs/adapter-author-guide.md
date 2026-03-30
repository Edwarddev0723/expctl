# Adapter Author Guide

每個 `expctl` project 都要提供 `src/project/__init__.py`，並暴露三個穩定 entrypoints：

- `feature_catalog()`
- `model_registry()`
- `load_dataset(config)`

## feature_catalog()

必須回傳 `expctl.config.FeatureCatalog`。

## model_registry()

必須回傳 `expctl.training.ModelRegistry`。`config.model.type` 會透過這個 registry resolve 成 estimator builder。

## load_dataset(config)

必須回傳 `pandas.DataFrame`。建議直接使用 `config["data"]["train_path"]`，因為 runtime 會先把路徑 resolve 成絕對路徑再傳進來。

## What doctor validates

`expctl doctor` 會檢查：

- adapter 檔案是否存在
- 三個 entrypoints 是否存在且型別正確
- `config.model.type` 是否能被 registry resolve
- dataset 是否能被成功載入
- dataset 是否包含 target / id / group / time / feature columns
