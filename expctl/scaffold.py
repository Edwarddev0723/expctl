from __future__ import annotations

import csv
from pathlib import Path

from expctl.registry import REGISTRY_COLUMNS


def create_project_scaffold(project_root: Path, *, force: bool = False) -> list[Path]:
    project_root = project_root.resolve()
    if project_root.exists() and any(project_root.iterdir()) and not force:
        raise FileExistsError(
            f"Target directory is not empty: {project_root}. Use --force to write scaffold anyway."
        )

    created: list[Path] = []
    for relative_path, content in _scaffold_files().items():
        path = project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing scaffold file: {path}")
        path.write_text(content, encoding="utf-8")
        created.append(path)

    registry_path = project_root / "reports" / "experiment_registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
    created.append(registry_path)
    return created


def _scaffold_files() -> dict[str, str]:
    return {
        "README.md": _readme_content(),
        "data/README.md": _data_readme_content(),
        "artifacts/README.md": _artifacts_readme_content(),
        "artifacts/experiments/README.md": _experiments_readme_content(),
        "artifacts/splits/README.md": _splits_readme_content(),
        "configs/README.md": _configs_readme_content(),
        "configs/experiments/example_classification.yaml": _example_config_content(),
        "configs/experiments/example_regression.yaml": _example_regression_config_content(),
        "configs/experiments/example_grouped_classification.yaml": (
            _example_grouped_config_content()
        ),
        "configs/experiments/example_time_series_regression.yaml": (
            _example_time_series_config_content()
        ),
        "docs/quickstart.md": _quickstart_content(),
        "docs/adapter-guide.md": _adapter_guide_content(),
        "docs/config-reference.md": _config_reference_content(),
        "reports/decisions.md": _decisions_content(),
        "src/project/__init__.py": _adapter_content(),
    }


def _readme_content() -> str:
    return """# expctl Project

This project was scaffolded with `expctl init`.

## Included Configs

- `configs/experiments/example_classification.yaml`
- `configs/experiments/example_regression.yaml`
- `configs/experiments/example_grouped_classification.yaml`
- `configs/experiments/example_time_series_regression.yaml`

## Quick Start

1. Put your training data at `data/train.csv`.
2. Choose the example config that matches your task.
3. Update the feature columns and model params.
4. Run:

```bash
expctl validate-config --config configs/experiments/example_classification.yaml
expctl build-splits --config configs/experiments/example_classification.yaml
expctl train --config configs/experiments/example_classification.yaml
expctl evaluate --config configs/experiments/example_classification.yaml
expctl register --config configs/experiments/example_classification.yaml
expctl doctor --config configs/experiments/example_classification.yaml
```

## Next Steps

- Edit `src/project/__init__.py` if your dataset loading or model registry differs from the starter.
- Add project-specific notes in `reports/decisions.md`.
- Expand the docs in `docs/` once your team conventions are clear.
"""


def _data_readme_content() -> str:
    return """# Data Directory

Place your tabular datasets here. The default scaffold expects:

- `data/train.csv`
- optional `data/test.csv`
"""


def _artifacts_readme_content() -> str:
    return """# Artifacts

This directory stores generated experiment outputs. Commit configuration and docs,
but avoid committing model outputs unless your team explicitly versions them.
"""


def _experiments_readme_content() -> str:
    return """# Experiment Outputs

Each run writes its fold metrics, OOF predictions, summary, tracking metadata,
and optional submission under a dedicated subdirectory.
"""


def _splits_readme_content() -> str:
    return """# Split Artifacts

Store immutable validation split definitions here. Treat these files as part of
the experiment contract rather than temporary outputs.
"""


def _configs_readme_content() -> str:
    return """# Configs

Keep one config per experiment family. Favor cloning from the closest example and
changing only the task-, validation-, and model-specific fields.
"""


def _example_config_content() -> str:
    return """schema_version: 1
task:
  type: binary_classification

data:
  train_path: data/train.csv
  test_path: data/test.csv
  target_col: target
  id_col: id

features:
  numeric:
    - feature_1
    - feature_2
  categorical: []
  derived:
    numeric: []
    categorical: []
  text: null
  ignore_text: true

validation:
  strategy: stratified_kfold
  protocol_name: stratified_kfold_v1
  split_path: artifacts/splits/stratified_kfold_v1.json
  n_splits: 5
  shuffle: true
  random_state: 42

model:
  type: logistic_regression
  params:
    max_iter: 200

evaluation:
  metrics:
    - accuracy
    - f1_macro
  primary_metric: accuracy

tracking:
  backend: local
  experiment_name: expctl
  run_name: exp_001_baseline
  tags:
    owner: team

output:
  dir: artifacts/experiments/exp_001_baseline
  fold_metrics_file: fold_metrics.csv
  oof_predictions_file: oof_predictions.csv
  summary_file: summary.yaml
  registry_path: reports/experiment_registry.csv
"""


def _example_regression_config_content() -> str:
    return """schema_version: 1
task:
  type: regression

data:
  train_path: data/train.csv
  test_path: data/test.csv
  target_col: target
  id_col: id
  group_col: group_id

features:
  numeric:
    - feature_1
    - feature_2
  categorical: []
  derived:
    numeric: []
    categorical: []
  text: null
  ignore_text: true

validation:
  strategy: group_kfold
  protocol_name: group_kfold_v1
  split_path: artifacts/splits/group_kfold_v1.json
  n_splits: 5
  shuffle: false
  random_state: 42

model:
  type: ridge_regression
  params:
    alpha: 1.0

evaluation:
  metrics:
    - rmse
    - mae
    - r2
  primary_metric: rmse

tracking:
  backend: local
  experiment_name: expctl
  run_name: exp_010_regression_baseline
  tags:
    owner: team

output:
  dir: artifacts/experiments/exp_010_regression_baseline
  fold_metrics_file: fold_metrics.csv
  oof_predictions_file: oof_predictions.csv
  summary_file: summary.yaml
  registry_path: reports/experiment_registry.csv
"""


def _example_grouped_config_content() -> str:
    return """schema_version: 1
task:
  type: binary_classification

data:
  train_path: data/train.csv
  test_path: data/test.csv
  target_col: target
  id_col: id
  group_col: customer_id

features:
  numeric:
    - feature_1
    - feature_2
  categorical:
    - region
  derived:
    numeric: []
    categorical: []
  text: null
  ignore_text: true

validation:
  strategy: stratified_group_kfold
  protocol_name: stratified_group_kfold_v1
  split_path: artifacts/splits/stratified_group_kfold_v1.json
  n_splits: 5
  shuffle: true
  random_state: 42

model:
  type: logistic_regression
  params:
    max_iter: 200

evaluation:
  metrics:
    - accuracy
    - f1_macro
    - roc_auc
  primary_metric: roc_auc

tracking:
  backend: local
  experiment_name: expctl
  run_name: exp_020_grouped_baseline
  tags:
    owner: team

output:
  dir: artifacts/experiments/exp_020_grouped_baseline
  fold_metrics_file: fold_metrics.csv
  oof_predictions_file: oof_predictions.csv
  summary_file: summary.yaml
  registry_path: reports/experiment_registry.csv
"""


def _example_time_series_config_content() -> str:
    return """schema_version: 1
task:
  type: regression

data:
  train_path: data/train.csv
  test_path: data/test.csv
  target_col: target
  id_col: id
  time_col: event_time

features:
  numeric:
    - feature_1
    - feature_2
  categorical: []
  derived:
    numeric: []
    categorical: []
  text: null
  ignore_text: true

validation:
  strategy: time_series_split
  protocol_name: time_series_split_v1
  split_path: artifacts/splits/time_series_split_v1.json
  n_splits: 3
  shuffle: false
  random_state: 42

model:
  type: ridge_regression
  params:
    alpha: 1.0

evaluation:
  metrics:
    - rmse
    - mae
    - r2
  primary_metric: rmse

tracking:
  backend: local
  experiment_name: expctl
  run_name: exp_030_time_series_baseline
  tags:
    owner: team

output:
  dir: artifacts/experiments/exp_030_time_series_baseline
  fold_metrics_file: fold_metrics.csv
  oof_predictions_file: oof_predictions.csv
  summary_file: summary.yaml
  registry_path: reports/experiment_registry.csv
"""


def _quickstart_content() -> str:
    return """# Quickstart

1. Put your CSVs in `data/`.
2. Pick a config from `configs/experiments/`.
3. Run:

```bash
expctl doctor --config configs/experiments/example_classification.yaml
expctl validate-config --config configs/experiments/example_classification.yaml
expctl build-splits --config configs/experiments/example_classification.yaml
expctl train --config configs/experiments/example_classification.yaml
expctl evaluate --config configs/experiments/example_classification.yaml
expctl register --config configs/experiments/example_classification.yaml
```
"""


def _adapter_guide_content() -> str:
    return """# Adapter Guide

Your project adapter must expose:

- `feature_catalog()`
- `model_registry()`
- `load_dataset(config)`

Return `expctl.config.FeatureCatalog` and `expctl.training.ModelRegistry`
instances from the first two functions. `load_dataset(config)` must return a
`pandas.DataFrame`.
"""


def _config_reference_content() -> str:
    return """# Config Reference

Stable top-level sections:

- `schema_version`
- `task`
- `data`
- `features`
- `validation`
- `model`
- `evaluation`
- `tracking`
- `output`

Prefer cloning one of the included example configs rather than writing a new
config from scratch.
"""


def _decisions_content() -> str:
    return """# Decisions

- Record naming conventions for experiments.
- Record when a split protocol changes.
- Record metric changes that affect leaderboard or offline comparisons.
"""


def _adapter_content() -> str:
    return '''from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from expctl.config import FeatureCatalog
from expctl.training import ModelRegistry


def feature_catalog() -> FeatureCatalog:
    return FeatureCatalog()


def model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(
        "logistic_regression",
        lambda **params: LogisticRegression(**params),
    )
    registry.register(
        "random_forest_classifier",
        lambda **params: RandomForestClassifier(**params),
    )
    registry.register(
        "ridge_regression",
        lambda **params: Ridge(**params),
    )
    return registry


def load_dataset(config: dict[str, object]) -> pd.DataFrame:
    train_path = Path(str(config["data"]["train_path"]))
    return pd.read_csv(train_path)
'''
