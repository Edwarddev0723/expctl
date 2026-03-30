# Validation Strategies

`expctl` beta 內建 5 種 validation strategy。

## stratified_kfold

用於 classification，會維持 target label 分布。

## repeated_stratified_kfold

用於 classification，需要 `n_repeats >= 2`。

## group_kfold

用於 regression 或 classification。需要 `config.data.group_col`。

## stratified_group_kfold

用於 classification。需要 `config.data.group_col`，同時保留 group 邊界與 label 分布。

## time_series_split

用於 time-series style validation。需要 `config.data.time_col`，且 `shuffle` 必須為 `false`。

## Common failures

- regression 不能搭配 stratified strategy
- group-based strategy 缺 `group_col`
- time-series strategy 缺 `time_col`
- time-series strategy 設了 `shuffle: true`
