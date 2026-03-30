# Tracking Backends

beta 版支援三種 backend：

- `local`
- `mlflow`
- `wandb`

## local

預設 backend。會在 run output directory 產生 `tracking.local.json`，保存：

- flattened params
- summary metrics
- tags
- config snapshot path
- artifact paths

## mlflow

beta 支援：

- params
- metrics
- tags
- config artifact
- fold metrics / summary / OOF artifact

需要額外安裝：

```bash
python -m pip install -e ".[tracking]"
```

## wandb

beta 支援範圍與 `mlflow` 相同，同樣需要安裝 optional dependency。

## Doctor checks

`expctl doctor` 會檢查 backend 依賴是否已安裝。若缺少 `mlflow` 或 `wandb`，會回傳 error 而不是等到 train 時才失敗。
