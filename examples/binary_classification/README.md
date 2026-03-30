# Binary Classification Example

This example shows the default happy path for tabular binary classification.

```bash
expctl doctor --repo-root examples/binary_classification --config examples/binary_classification/configs/experiments/experiment.yaml
expctl build-splits --repo-root examples/binary_classification --config examples/binary_classification/configs/experiments/experiment.yaml
expctl train --repo-root examples/binary_classification --config examples/binary_classification/configs/experiments/experiment.yaml
expctl register --repo-root examples/binary_classification --config examples/binary_classification/configs/experiments/experiment.yaml
```
