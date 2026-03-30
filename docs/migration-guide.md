# Migration Guide

## From legacy config to `schema_version: 1`

目前 beta 仍支援舊 config shape，但會發出 warning：

`Legacy config shape detected. expctl normalized it into schema_version=1 for beta compatibility`

建議遷移步驟：

1. 加入 `schema_version: 1`
2. 補齊 top-level sections：`task`, `data`, `features`, `validation`, `model`, `evaluation`, `tracking`, `output`
3. 明確設定 `validation.strategy`
4. 明確設定 `tracking.backend`
5. 為 `evaluation.metrics` 補 `primary_metric`

## Deprecation policy

- beta 期間會保留 legacy compat layer
- 相容路徑只保證 warning，不保證長期存在
- `0.6.x` 起若要移除 compat path，需先在 changelog 中公告
