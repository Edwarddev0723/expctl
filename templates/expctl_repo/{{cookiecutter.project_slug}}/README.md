# {{cookiecutter.project_name}}

Git-native、config-driven 的 Kaggle / 資料科學實驗框架骨架。

## 結構

```text
{{cookiecutter.project_slug}}/
├── AGENTS.md
├── pyproject.toml
├── configs/
│   └── templates/
├── docs/
├── reports/
├── artifacts/
│   └── splits/
└── src/
    └── project/
```

## 設計原則

- 所有實驗都由 YAML 定義
- Split artifact 固定且不可被 inline CV 取代
- Registry / decisions 採 append-only 治理
- 專案特定 feature engineering 放在 `src/project/`
- 通用能力由 `expctl` 套件提供

## 下一步

1. 複製 `expctl/` 套件到這個 repo
2. 在 `src/project/` 實作 feature catalog、model registry、submission schema
3. 依 competition 需求客製 `AGENTS.md`
