from __future__ import annotations

import numpy as np
import pytest

from expctl.evaluation.metrics import compute_metrics, validate_metric_names


def test_compute_metrics_supports_regression_metrics() -> None:
    metrics = compute_metrics(
        y_true=np.array([1.0, 2.0, 3.0]),
        y_pred=np.array([1.2, 2.1, 2.9]),
        y_proba=None,
        metric_names=["rmse", "mae", "r2"],
        task_type="regression",
    )

    assert set(metrics) == {"rmse", "mae", "r2"}
    assert metrics["rmse"] > 0


def test_validate_metric_names_rejects_wrong_task_metrics() -> None:
    with pytest.raises(ValueError, match="not supported for task 'regression'"):
        validate_metric_names("regression", ["accuracy"])
