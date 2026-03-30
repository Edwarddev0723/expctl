# Config Reference

`schema_version: 1` 的穩定 top-level sections 如下：

- `schema_version`
- `task`
- `data`
- `features`
- `validation`
- `model`
- `evaluation`
- `tracking`
- `output`

## task

- `type`: `binary_classification | multiclass_classification | regression`

## data

- `train_path`: 必填
- `test_path`: 選填；若提供，通常也應提供 `id_col`
- `target_col`: 必填
- `id_col`: 建議提供，submission workflow 需要
- `group_col`: group-based validation 必填
- `time_col`: time-series validation 必填

## features

- `numeric`
- `categorical`
- `derived.numeric`
- `derived.categorical`
- `text`
- `ignore_text`
- `text_vectorizer`

## validation

- `strategy`: `stratified_kfold | repeated_stratified_kfold | group_kfold | stratified_group_kfold | time_series_split`
- `protocol_name`
- `split_path`
- `n_splits`
- `n_repeats`
- `shuffle`
- `random_state`

## model

- `type`
- `params`

## evaluation

- `metrics`
- `primary_metric`

## tracking

- `backend`: `local | mlflow | wandb`
- `experiment_name`
- `run_name`
- `tags`

## output

- `dir`
- `fold_metrics_file`
- `oof_predictions_file`
- `summary_file`
- `registry_path`
- `config_snapshot_file`
- `submission_file`
