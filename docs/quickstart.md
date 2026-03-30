# 5-Minute Quickstart

`expctl` 的 beta happy path 是：

```bash
expctl init my-project
cd my-project
expctl doctor --config configs/experiments/example_classification.yaml
expctl validate-config --config configs/experiments/example_classification.yaml
expctl build-splits --config configs/experiments/example_classification.yaml
expctl train --config configs/experiments/example_classification.yaml
expctl evaluate --config configs/experiments/example_classification.yaml
expctl register --config configs/experiments/example_classification.yaml
```

最少只需要補三件事：

1. 放入 `data/train.csv`，必要時加上 `data/test.csv`
2. 修改對應的 example config
3. 在 `src/project/__init__.py` 補上你的 dataset loader、feature catalog、model registry

若是既有專案導入，先看 `docs/migration-guide.md`。
