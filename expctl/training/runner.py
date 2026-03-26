from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldRunOutput:
    fold_id: int
    repeat_id: int
    valid_indices: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray | None
    score_label: Any | None
    metrics: dict[str, float]


def pick_folds(
    *,
    fold_artifact: dict[str, Any],
    fold_id: int | None,
    repeat_id: int,
) -> list[dict[str, Any]]:
    folds = [fold for fold in fold_artifact["folds"] if int(fold["repeat_id"]) == repeat_id]
    if fold_id is not None:
        folds = [fold for fold in folds if int(fold["fold_id"]) == fold_id]
    if not folds:
        raise ValueError(
            f"No fold found for fold_id={fold_id}, repeat_id={repeat_id}. "
            f"Available folds: {[(f['repeat_id'], f['fold_id']) for f in fold_artifact['folds']]}"
        )
    return sorted(folds, key=lambda item: int(item["fold_id"]))


def pick_all_folds(fold_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        fold_artifact["folds"],
        key=lambda item: (int(item["repeat_id"]), int(item["fold_id"])),
    )


def summarize_metrics(fold_metrics_df: pd.DataFrame, metric_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_folds": int(fold_metrics_df.shape[0]),
    }
    for metric in metric_names:
        summary[f"{metric}_mean"] = float(fold_metrics_df[metric].mean())
        summary[f"{metric}_std"] = float(fold_metrics_df[metric].std(ddof=0))
    if "pseudo_label_count" in fold_metrics_df.columns:
        summary["pseudo_label_count_mean"] = float(fold_metrics_df["pseudo_label_count"].mean())
        summary["pseudo_label_count_std"] = float(fold_metrics_df["pseudo_label_count"].std(ddof=0))
    return summary


def derive_repeated_output_dir(output_dir: Path, fold_artifact: dict[str, Any]) -> Path:
    suffix = f"__rskf_{int(fold_artifact['n_splits'])}x{int(fold_artifact['n_repeats'])}"
    output_dir_str = str(output_dir)
    if output_dir_str.endswith(suffix):
        return output_dir
    return Path(f"{output_dir_str}{suffix}")


def extract_estimator_classes(estimator: Any) -> np.ndarray:
    if hasattr(estimator, "classes_"):
        return np.asarray(estimator.classes_)
    if hasattr(estimator, "named_steps"):
        model = estimator.named_steps.get("model")
        if model is not None and hasattr(model, "classes_"):
            return np.asarray(model.classes_)
    raise ValueError("Estimator does not expose classes_")
