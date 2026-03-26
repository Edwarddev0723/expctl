# AI Agent Governance

## Hard Rules

1. 不得修改 `artifacts/splits/` 既有 split artifact。
2. 不得在訓練腳本 inline 建立 `KFold` / `train_test_split`；所有 fold 必須由 artifact 載入。
3. 不得手動編輯 `reports/experiment_registry.csv`；必須透過 script/CLI 寫入。
4. 不得修改原始資料目錄。
5. 不得刪除既有 experiment artifact 目錄。
6. 未通過 submission validation 不得產生正式提交檔。
7. 不得硬編碼 API key / secrets / PII。

## Repo Contracts

- Config-driven first: 先改 config，再改 code。
- Project-specific feature engineering 放在 `src/project/`。
- Framework logic 放在 `expctl/`。
- Registry 與 decisions 採 append-only。
