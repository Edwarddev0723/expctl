from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SUPPORTED_METRICS = frozenset({"accuracy", "f1_macro", "roc_auc"})


def compute_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    metric_names: list[str],
) -> dict[str, float]:
    results: dict[str, float] = {}
    for metric in metric_names:
        if metric == "accuracy":
            results[metric] = float(accuracy_score(y_true, y_pred))
        elif metric == "f1_macro":
            results[metric] = float(f1_score(y_true, y_pred, average="macro"))
        elif metric == "roc_auc":
            if y_proba is None:
                raise ValueError("roc_auc requested but model has no predict_proba output")
            unique_labels = np.unique(y_true)
            if unique_labels.shape[0] == 2:
                results[metric] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                results[metric] = float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="macro",
                    )
                )
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return results
