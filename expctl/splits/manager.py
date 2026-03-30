from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

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
SUPPORTED_SPLIT_STRATEGIES = frozenset(
    {
        "stratified_kfold",
        "repeated_stratified_kfold",
        "group_kfold",
        "stratified_group_kfold",
        "time_series_split",
    }
)


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
        splitter_kwargs: dict[str, Any] = {
            "n_splits": n_splits,
            "shuffle": shuffle,
        }
        if shuffle:
            splitter_kwargs["random_state"] = random_state
        splitter = StratifiedKFold(**splitter_kwargs)

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
        "strategy": "repeated_stratified_kfold" if repeated else "stratified_kfold",
        "seed": int(random_state),
        "n_splits": int(n_splits),
        "shuffle": bool(shuffle),
        "repeated": bool(repeated),
        "n_repeats": int(n_repeats),
        "n_samples": int(len(y_array)),
        "folds": folds,
    }


def build_validation_folds(
    *,
    strategy: str,
    protocol_name: str,
    n_splits: int,
    y: Sequence[int] | Sequence[str] | np.ndarray | None = None,
    groups: Sequence[Any] | np.ndarray | None = None,
    time_order: Sequence[Any] | np.ndarray | None = None,
    shuffle: bool = True,
    random_state: int = 42,
    n_repeats: int = 1,
) -> dict[str, Any]:
    if strategy not in SUPPORTED_SPLIT_STRATEGIES:
        raise ValueError(
            f"Unsupported validation strategy: {strategy}. "
            f"Supported={sorted(SUPPORTED_SPLIT_STRATEGIES)}"
        )

    if strategy == "stratified_kfold":
        if y is None:
            raise ValueError("stratified_kfold requires y")
        return build_stratified_folds(
            y,
            protocol_name=protocol_name,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
            repeated=False,
            n_repeats=1,
        )

    if strategy == "repeated_stratified_kfold":
        if y is None:
            raise ValueError("repeated_stratified_kfold requires y")
        return build_stratified_folds(
            y,
            protocol_name=protocol_name,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
            repeated=True,
            n_repeats=n_repeats,
        )

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    if strategy == "group_kfold":
        if groups is None:
            raise ValueError("group_kfold requires groups")
        return _build_group_folds(
            strategy=strategy,
            protocol_name=protocol_name,
            n_splits=n_splits,
            groups=np.asarray(groups),
            random_state=random_state,
        )

    if strategy == "stratified_group_kfold":
        if groups is None or y is None:
            raise ValueError("stratified_group_kfold requires both y and groups")
        return _build_stratified_group_folds(
            protocol_name=protocol_name,
            n_splits=n_splits,
            y=np.asarray(y),
            groups=np.asarray(groups),
            shuffle=shuffle,
            random_state=random_state,
        )

    if time_order is None:
        raise ValueError("time_series_split requires time_order")
    return _build_time_series_folds(
        protocol_name=protocol_name,
        n_splits=n_splits,
        time_order=np.asarray(time_order),
    )


def _build_group_folds(
    *,
    strategy: str,
    protocol_name: str,
    n_splits: int,
    groups: np.ndarray,
    random_state: int,
) -> dict[str, Any]:
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("groups must be a non-empty 1D sequence")
    splitter = GroupKFold(n_splits=n_splits)
    zeros = np.zeros(shape=(groups.shape[0], 1), dtype=np.int8)
    folds: list[dict[str, Any]] = []
    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(zeros, groups=groups)):
        folds.append(
            {
                "fold_id": int(fold_id),
                "repeat_id": 0,
                "train_indices": train_idx.astype(int).tolist(),
                "valid_indices": valid_idx.astype(int).tolist(),
            }
        )
    return {
        "protocol_name": protocol_name,
        "strategy": strategy,
        "seed": int(random_state),
        "n_splits": int(n_splits),
        "shuffle": False,
        "repeated": False,
        "n_repeats": 1,
        "n_samples": int(groups.shape[0]),
        "folds": folds,
    }


def _build_stratified_group_folds(
    *,
    protocol_name: str,
    n_splits: int,
    y: np.ndarray,
    groups: np.ndarray,
    shuffle: bool,
    random_state: int,
) -> dict[str, Any]:
    if y.ndim != 1 or groups.ndim != 1 or y.shape[0] != groups.shape[0]:
        raise ValueError("y and groups must be 1D sequences with the same length")
    splitter_kwargs: dict[str, Any] = {"n_splits": n_splits, "shuffle": shuffle}
    if shuffle:
        splitter_kwargs["random_state"] = random_state
    splitter = StratifiedGroupKFold(**splitter_kwargs)
    zeros = np.zeros(shape=(y.shape[0], 1), dtype=np.int8)
    folds: list[dict[str, Any]] = []
    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(zeros, y, groups=groups)):
        folds.append(
            {
                "fold_id": int(fold_id),
                "repeat_id": 0,
                "train_indices": train_idx.astype(int).tolist(),
                "valid_indices": valid_idx.astype(int).tolist(),
            }
        )
    return {
        "protocol_name": protocol_name,
        "strategy": "stratified_group_kfold",
        "seed": int(random_state),
        "n_splits": int(n_splits),
        "shuffle": bool(shuffle),
        "repeated": False,
        "n_repeats": 1,
        "n_samples": int(y.shape[0]),
        "folds": folds,
    }


def _build_time_series_folds(
    *,
    protocol_name: str,
    n_splits: int,
    time_order: np.ndarray,
) -> dict[str, Any]:
    if time_order.ndim != 1 or time_order.size == 0:
        raise ValueError("time_order must be a non-empty 1D sequence")

    ordered_indices = np.argsort(time_order, kind="stable")
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds: list[dict[str, Any]] = []
    zeros = np.zeros(shape=(time_order.shape[0], 1), dtype=np.int8)
    for fold_id, (train_pos, valid_pos) in enumerate(splitter.split(zeros[ordered_indices])):
        train_idx = ordered_indices[train_pos]
        valid_idx = ordered_indices[valid_pos]
        folds.append(
            {
                "fold_id": int(fold_id),
                "repeat_id": 0,
                "train_indices": train_idx.astype(int).tolist(),
                "valid_indices": valid_idx.astype(int).tolist(),
            }
        )
    return {
        "protocol_name": protocol_name,
        "strategy": "time_series_split",
        "seed": 0,
        "n_splits": int(n_splits),
        "shuffle": False,
        "repeated": False,
        "n_repeats": 1,
        "n_samples": int(time_order.shape[0]),
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
