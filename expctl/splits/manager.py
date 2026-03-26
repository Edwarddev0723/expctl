from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

REQUIRED_ARTIFACT_KEYS = {
    "protocol_name",
    "seed",
    "n_splits",
    "shuffle",
    "repeated",
    "n_repeats",
    "n_samples",
    "folds",
}

REQUIRED_FOLD_KEYS = {"fold_id", "repeat_id", "train_indices", "valid_indices"}


def build_stratified_folds(
    y: Sequence[int] | Sequence[str] | np.ndarray,
    *,
    protocol_name: str,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    repeated: bool = False,
    n_repeats: int = 1,
) -> dict[str, Any]:
    """Build fixed stratified fold indices for a classification target."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    if repeated and n_repeats == 1:
        raise ValueError("repeated=True requires n_repeats >= 2")

    y_array = np.asarray(y)
    if y_array.ndim != 1:
        raise ValueError("y must be a 1D sequence")
    if len(y_array) == 0:
        raise ValueError("y must not be empty")

    if repeated:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
    else:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    folds: list[dict[str, Any]] = []
    zeros = np.zeros(shape=(len(y_array), 1), dtype=np.int8)
    for global_fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(zeros, y_array)):
        repeat_id = global_fold_idx // n_splits if repeated else 0
        fold_id = global_fold_idx % n_splits
        folds.append(
            {
                "fold_id": int(fold_id),
                "repeat_id": int(repeat_id),
                "train_indices": train_idx.astype(int).tolist(),
                "valid_indices": valid_idx.astype(int).tolist(),
            }
        )

    return {
        "protocol_name": protocol_name,
        "seed": int(random_state),
        "n_splits": int(n_splits),
        "shuffle": bool(shuffle),
        "repeated": bool(repeated),
        "n_repeats": int(n_repeats),
        "n_samples": int(len(y_array)),
        "folds": folds,
    }


def save_folds(
    artifact: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Persist fold artifact to disk in JSON format."""
    _validate_artifact(artifact)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Split artifact already exists: {output_path}. "
            "Use overwrite=True to regenerate intentionally."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)


def load_folds(input_path: Path) -> dict[str, Any]:
    """Load and validate a persisted fold artifact."""
    with input_path.open("r", encoding="utf-8") as fh:
        artifact = json.load(fh)
    _validate_artifact(artifact)
    return artifact


def get_fold_indices(
    artifact: dict[str, Any],
    *,
    fold_id: int,
    repeat_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Get train/valid indices for a specific fold and repeat id."""
    _validate_artifact(artifact)
    for fold in artifact["folds"]:
        if int(fold["fold_id"]) == fold_id and int(fold["repeat_id"]) == repeat_id:
            train_idx = np.asarray(fold["train_indices"], dtype=int)
            valid_idx = np.asarray(fold["valid_indices"], dtype=int)
            return train_idx, valid_idx

    raise KeyError(f"Fold not found: fold_id={fold_id}, repeat_id={repeat_id}")


def _validate_artifact(artifact: dict[str, Any]) -> None:
    if not isinstance(artifact, dict):
        raise ValueError("artifact must be a dictionary")
    missing_root = REQUIRED_ARTIFACT_KEYS.difference(set(artifact.keys()))
    if missing_root:
        raise KeyError(f"artifact missing keys: {sorted(missing_root)}")

    folds = artifact["folds"]
    if not isinstance(folds, list) or len(folds) == 0:
        raise ValueError("artifact['folds'] must be a non-empty list")

    n_samples = int(artifact["n_samples"])
    for fold in folds:
        if not isinstance(fold, dict):
            raise ValueError("each fold entry must be a dictionary")
        missing_fold = REQUIRED_FOLD_KEYS.difference(set(fold.keys()))
        if missing_fold:
            raise KeyError(f"fold entry missing keys: {sorted(missing_fold)}")

        train_idx = np.asarray(fold["train_indices"], dtype=int)
        valid_idx = np.asarray(fold["valid_indices"], dtype=int)
        if train_idx.ndim != 1 or valid_idx.ndim != 1:
            raise ValueError("train_indices and valid_indices must be 1D")
        if np.intersect1d(train_idx, valid_idx).size > 0:
            raise ValueError("train/valid leakage detected in fold artifact")
        if train_idx.min(initial=0) < 0 or valid_idx.min(initial=0) < 0:
            raise ValueError("indices must be >= 0")
        if train_idx.max(initial=-1) >= n_samples or valid_idx.max(initial=-1) >= n_samples:
            raise ValueError("indices exceed n_samples")
