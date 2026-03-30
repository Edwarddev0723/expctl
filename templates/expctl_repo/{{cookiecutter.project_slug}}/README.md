# {{cookiecutter.project_name}}

CLI-first、config-driven 的資料科學實驗框架骨架，對齊 `expctl` public beta contract。

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

1. 安裝 `expctl`
2. 在 `src/project/` 實作或擴充 feature catalog、model registry、dataset loader
3. 編輯 `configs/templates/experiment.yaml`
4. 先執行 `expctl doctor`
5. 執行 `expctl validate-config`, `expctl build-splits`, `expctl train`

## 參考文件

- `docs/README.md`
- `README.md`
