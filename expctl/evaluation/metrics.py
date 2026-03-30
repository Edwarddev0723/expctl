from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

CLASSIFICATION_TASKS = frozenset({"binary_classification", "multiclass_classification"})
REGRESSION_TASKS = frozenset({"regression"})

TASK_METRICS: dict[str, frozenset[str]] = {
    "binary_classification": frozenset({"accuracy", "f1_macro", "roc_auc", "log_loss"}),
    "multiclass_classification": frozenset({"accuracy", "f1_macro", "roc_auc", "log_loss"}),
    "regression": frozenset({"rmse", "mae", "r2"}),
}
SUPPORTED_METRICS = frozenset().union(*TASK_METRICS.values())


def validate_metric_names(task_type: str, metric_names: list[str]) -> None:
    if task_type not in TASK_METRICS:
        raise ValueError(f"Unsupported task type: {task_type}")
    unsupported = [metric for metric in metric_names if metric not in TASK_METRICS[task_type]]
    if unsupported:
        raise ValueError(
            f"Metrics {unsupported} are not supported for task '{task_type}'. "
            f"Allowed={sorted(TASK_METRICS[task_type])}"
        )


def _binary_score_array(y_score: np.ndarray) -> np.ndarray:
    if y_score.ndim == 1:
        return y_score
    if y_score.ndim == 2 and y_score.shape[1] >= 2:
        return y_score[:, 1]
    raise ValueError("Binary classification metrics require a 1D score array or 2-class scores")


def compute_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    metric_names: list[str],
    task_type: str | None = None,
) -> dict[str, float]:
    if task_type is not None:
        validate_metric_names(task_type, metric_names)

    results: dict[str, float] = {}
    for metric in metric_names:
        if metric == "accuracy":
            results[metric] = float(accuracy_score(y_true, y_pred))
        elif metric == "f1_macro":
            results[metric] = float(f1_score(y_true, y_pred, average="macro"))
        elif metric == "roc_auc":
            if y_proba is None:
                raise ValueError("roc_auc requested but model has no score/probability output")
            unique_labels = np.unique(y_true)
            if unique_labels.shape[0] == 2:
                results[metric] = float(roc_auc_score(y_true, _binary_score_array(y_proba)))
            else:
                if y_proba.ndim != 2:
                    raise ValueError("Multiclass roc_auc requires a 2D probability array")
                results[metric] = float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="macro",
                    )
                )
        elif metric == "log_loss":
            if y_proba is None:
                raise ValueError("log_loss requested but model has no predict_proba output")
            results[metric] = float(log_loss(y_true, y_proba))
        elif metric == "rmse":
            results[metric] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif metric == "mae":
            results[metric] = float(mean_absolute_error(y_true, y_pred))
        elif metric == "r2":
            results[metric] = float(r2_score(y_true, y_pred))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return results
