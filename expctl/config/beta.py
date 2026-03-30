from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from expctl.config.contracts import FeatureCatalog
from expctl.config.validation import (
    ConfigValidationError,
    _as_mapping,
    _find_duplicates,
    _optional_int,
    _optional_str,
    _optional_str_list,
    _require_mapping,
    _require_str,
    _require_str_list,
    _resolve_repo_path,
    _validate_derived_features,
    _validate_text_vectorizer,
)
from expctl.evaluation.metrics import validate_metric_names
from expctl.splits import SUPPORTED_SPLIT_STRATEGIES

DEFAULT_SCHEMA_VERSION = 1
TASK_TYPES = frozenset({"binary_classification", "multiclass_classification", "regression"})
TRACKING_BACKENDS = frozenset({"local", "mlflow", "wandb"})
CLASSIFICATION_TASKS = frozenset({"binary_classification", "multiclass_classification"})
REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "task",
        "data",
        "features",
        "validation",
        "model",
        "evaluation",
        "tracking",
        "output",
    }
)
LEGACY_COMPAT_WARNING = (
    "Legacy config shape detected. expctl normalized it into schema_version=1 for beta "
    "compatibility; migrate to the new contract to avoid future breakage."
)


def load_config_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return _as_mapping(data, f"config<{path}>")


def dump_config_file(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def is_beta_config(cfg: dict[str, Any]) -> bool:
    return int(cfg.get("schema_version", 0) or 0) == DEFAULT_SCHEMA_VERSION


def normalize_config(
    cfg: dict[str, Any],
    *,
    repo_root: Path,
    config_path: Path,
    feature_catalog: FeatureCatalog,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if is_beta_config(cfg):
        normalized = _normalize_beta_config(cfg, config_path=config_path)
    else:
        normalized = _convert_legacy_config(cfg, repo_root=repo_root, config_path=config_path)
        warnings.append(LEGACY_COMPAT_WARNING)
    validate_beta_config(
        normalized,
        repo_root=repo_root,
        config_path=config_path,
        feature_catalog=feature_catalog,
    )
    return normalized, warnings


def validate_beta_config(
    cfg: dict[str, Any],
    *,
    repo_root: Path,
    config_path: Path,
    feature_catalog: FeatureCatalog,
) -> None:
    root = _as_mapping(cfg, f"config<{config_path}>")
    unknown_keys = sorted(set(root) - REQUIRED_TOP_LEVEL_KEYS)
    if unknown_keys:
        raise ConfigValidationError(
            f"Unknown top-level config keys are not allowed in schema_version=1: {unknown_keys}"
        )
    schema_version = root.get("schema_version")
    if schema_version != DEFAULT_SCHEMA_VERSION:
        raise ConfigValidationError(
            f"'config.schema_version' must equal {DEFAULT_SCHEMA_VERSION}, got {schema_version!r}"
        )

    task_cfg = _require_mapping(root, "task", "config")
    data_cfg = _require_mapping(root, "data", "config")
    feature_cfg = _require_mapping(root, "features", "config")
    validation_cfg = _require_mapping(root, "validation", "config")
    model_cfg = _require_mapping(root, "model", "config")
    evaluation_cfg = _require_mapping(root, "evaluation", "config")
    tracking_cfg = _require_mapping(root, "tracking", "config")
    output_cfg = _require_mapping(root, "output", "config")

    task_type = _require_str(task_cfg, "type", "config.task")
    if task_type not in TASK_TYPES:
        raise ConfigValidationError(
            f"Unsupported task type: {task_type}. Supported={sorted(TASK_TYPES)}"
        )

    train_path_like = _require_str(data_cfg, "train_path", "config.data")
    _resolve_repo_path(
        repo_root,
        train_path_like,
        "config.data.train_path",
        must_exist=True,
        must_be_file=True,
    )
    _require_str(data_cfg, "target_col", "config.data")
    id_col = _optional_str(data_cfg, "id_col", "config.data")
    group_col = _optional_str(data_cfg, "group_col", "config.data")
    time_col = _optional_str(data_cfg, "time_col", "config.data")
    test_path_like = _optional_str(data_cfg, "test_path", "config.data")
    if test_path_like is not None:
        _resolve_repo_path(
            repo_root,
            test_path_like,
            "config.data.test_path",
            must_exist=True,
            must_be_file=True,
        )

    numeric_cols = _optional_str_list(feature_cfg, "numeric", "config.features")
    categorical_cols = _optional_str_list(feature_cfg, "categorical", "config.features")
    derived_numeric, derived_categorical = _validate_derived_features(
        feature_cfg,
        "config.features",
        feature_catalog,
    )
    ignore_text = feature_cfg.get("ignore_text", True)
    if not isinstance(ignore_text, bool):
        raise ConfigValidationError(
            f"'config.features.ignore_text' must be a boolean, got {type(ignore_text).__name__}"
        )
    text_col = _optional_str(feature_cfg, "text", "config.features")
    text_vectorizer_cfg = _validate_text_vectorizer(
        feature_cfg,
        "config.features",
        feature_catalog,
    )
    if (
        not numeric_cols
        and not categorical_cols
        and not derived_numeric
        and not derived_categorical
        and (ignore_text or text_col is None)
    ):
        raise ConfigValidationError("At least one feature source is required in config.features")
    duplicates = _find_duplicates(
        numeric_cols + categorical_cols + derived_numeric + derived_categorical
    )
    if duplicates:
        raise ConfigValidationError(
            f"Duplicate feature names across config.features.* are not allowed: {duplicates}"
        )
    if ignore_text and text_vectorizer_cfg is not None:
        raise ConfigValidationError(
            "config.features.text_vectorizer requires config.features.ignore_text=false"
        )
    if not ignore_text and text_col is None:
        raise ConfigValidationError(
            "config.features.ignore_text=false requires config.features.text"
        )
    if not ignore_text and text_vectorizer_cfg is None:
        raise ConfigValidationError(
            "config.features.ignore_text=false requires config.features.text_vectorizer"
        )

    strategy = _require_str(validation_cfg, "strategy", "config.validation")
    if strategy not in SUPPORTED_SPLIT_STRATEGIES:
        raise ConfigValidationError(
            f"Unsupported validation strategy: {strategy}. "
            f"Supported={sorted(SUPPORTED_SPLIT_STRATEGIES)}"
        )
    split_path_like = _require_str(validation_cfg, "split_path", "config.validation")
    split_path = repo_root / split_path_like
    if split_path.exists() and not split_path.is_file():
        raise ConfigValidationError(
            f"'config.validation.split_path' must point to a file: {split_path}"
        )
    n_splits = _optional_int(validation_cfg, "n_splits", "config.validation")
    if n_splits is None or n_splits < 2:
        raise ConfigValidationError("'config.validation.n_splits' must be an integer >= 2")
    n_repeats = _optional_int(validation_cfg, "n_repeats", "config.validation")
    if strategy == "repeated_stratified_kfold":
        if n_repeats is None or n_repeats < 2:
            raise ConfigValidationError(
                "'config.validation.n_repeats' must be an integer >= 2 "
                "for repeated_stratified_kfold"
            )
    elif n_repeats is not None and n_repeats < 1:
        raise ConfigValidationError("'config.validation.n_repeats' must be >= 1")
    shuffle = validation_cfg.get("shuffle", strategy != "time_series_split")
    if not isinstance(shuffle, bool):
        raise ConfigValidationError(
            f"'config.validation.shuffle' must be a boolean, got {type(shuffle).__name__}"
        )
    random_state = validation_cfg.get("random_state", 42)
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise ConfigValidationError(
            "'config.validation.random_state' must be an integer, "
            f"got {type(random_state).__name__}"
        )
    if task_type == "regression" and strategy in {
        "stratified_kfold",
        "repeated_stratified_kfold",
        "stratified_group_kfold",
    }:
        raise ConfigValidationError(
            "Regression tasks require a non-stratified validation strategy"
        )
    if strategy in {"group_kfold", "stratified_group_kfold"} and group_col is None:
        raise ConfigValidationError(
            f"config.data.group_col is required when validation.strategy='{strategy}'"
        )
    if strategy == "time_series_split" and time_col is None:
        raise ConfigValidationError(
            "config.data.time_col is required when validation.strategy='time_series_split'"
        )
    if strategy == "time_series_split" and shuffle:
        raise ConfigValidationError("time_series_split requires config.validation.shuffle=false")
    if strategy in {"stratified_kfold", "repeated_stratified_kfold", "stratified_group_kfold"}:
        if task_type not in CLASSIFICATION_TASKS:
            raise ConfigValidationError(
                f"validation.strategy='{strategy}' is only valid for classification tasks"
            )

    _require_str(model_cfg, "type", "config.model")
    _require_mapping(model_cfg, "params", "config.model")

    metrics = _require_str_list(evaluation_cfg, "metrics", "config.evaluation")
    if not metrics:
        raise ConfigValidationError("config.evaluation.metrics must not be empty")
    try:
        validate_metric_names(task_type, metrics)
    except ValueError as exc:
        raise ConfigValidationError(str(exc)) from exc
    primary_metric = _optional_str(evaluation_cfg, "primary_metric", "config.evaluation")
    if primary_metric is not None and primary_metric not in metrics:
        raise ConfigValidationError(
            "config.evaluation.primary_metric must be one of config.evaluation.metrics"
        )

    backend = tracking_cfg.get("backend", "local")
    if not isinstance(backend, str) or backend not in TRACKING_BACKENDS:
        raise ConfigValidationError(
            f"Unsupported tracking backend: {backend!r}. Supported={sorted(TRACKING_BACKENDS)}"
        )
    _optional_str(tracking_cfg, "experiment_name", "config.tracking")
    _optional_str(tracking_cfg, "run_name", "config.tracking")
    tags = tracking_cfg.get("tags", {})
    if not isinstance(tags, dict):
        raise ConfigValidationError(
            f"'config.tracking.tags' must be a mapping/dict, got {type(tags).__name__}"
        )

    _require_str(output_cfg, "dir", "config.output")
    _require_str(output_cfg, "fold_metrics_file", "config.output")
    _require_str(output_cfg, "oof_predictions_file", "config.output")
    _require_str(output_cfg, "summary_file", "config.output")
    _optional_str(output_cfg, "registry_path", "config.output")
    _optional_str(output_cfg, "config_snapshot_file", "config.output")
    _optional_str(output_cfg, "submission_file", "config.output")

    if id_col is None and test_path_like is not None:
        raise ConfigValidationError(
            "config.data.id_col is required when config.data.test_path is set"
        )


def _normalize_beta_config(cfg: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    normalized = dict(cfg)
    normalized["schema_version"] = DEFAULT_SCHEMA_VERSION

    validation_cfg = dict(normalized.get("validation", {}))
    strategy = validation_cfg.get("strategy", "stratified_kfold")
    validation_cfg.setdefault("strategy", strategy)
    validation_cfg.setdefault("n_splits", 5)
    validation_cfg.setdefault("n_repeats", 2 if strategy == "repeated_stratified_kfold" else 1)
    validation_cfg.setdefault("shuffle", strategy != "time_series_split")
    validation_cfg.setdefault("random_state", 42)
    validation_cfg.setdefault("protocol_name", f"{strategy}_v1")
    validation_cfg.setdefault(
        "split_path",
        f"artifacts/splits/{validation_cfg['protocol_name']}.json",
    )
    normalized["validation"] = validation_cfg

    tracking_cfg = dict(normalized.get("tracking", {}))
    tracking_cfg.setdefault("backend", "local")
    tracking_cfg.setdefault("experiment_name", "expctl")
    tracking_cfg.setdefault("run_name", config_path.stem)
    tracking_cfg.setdefault("tags", {})
    normalized["tracking"] = tracking_cfg

    output_cfg = dict(normalized.get("output", {}))
    output_cfg.setdefault("dir", f"artifacts/experiments/{config_path.stem}")
    output_cfg.setdefault("fold_metrics_file", "fold_metrics.csv")
    output_cfg.setdefault("oof_predictions_file", "oof_predictions.csv")
    output_cfg.setdefault("summary_file", "summary.yaml")
    output_cfg.setdefault("registry_path", "reports/experiment_registry.csv")
    output_cfg.setdefault("config_snapshot_file", "config.snapshot.yaml")
    output_cfg.setdefault("submission_file", "submission.csv")
    normalized["output"] = output_cfg

    features_cfg = dict(normalized.get("features", {}))
    features_cfg.setdefault("numeric", [])
    features_cfg.setdefault("categorical", [])
    features_cfg.setdefault("derived", {"numeric": [], "categorical": []})
    features_cfg.setdefault("ignore_text", True)
    normalized["features"] = features_cfg

    evaluation_cfg = dict(normalized.get("evaluation", {}))
    metrics = list(evaluation_cfg.get("metrics", []))
    if metrics:
        evaluation_cfg.setdefault("primary_metric", metrics[0])
    normalized["evaluation"] = evaluation_cfg

    return normalized


def _convert_legacy_config(
    cfg: dict[str, Any],
    *,
    repo_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    root = _as_mapping(cfg, f"config<{config_path}>")
    data_cfg = dict(_require_mapping(root, "data", "config"))
    feature_cfg = dict(_require_mapping(root, "features", "config"))
    validation_cfg = dict(_require_mapping(root, "validation", "config"))
    model_cfg = dict(_require_mapping(root, "model", "config"))
    evaluation_cfg = dict(_require_mapping(root, "evaluation", "config"))
    output_cfg = dict(_require_mapping(root, "output", "config"))

    inferred = _infer_legacy_validation(validation_cfg, repo_root=repo_root)
    metrics = list(evaluation_cfg.get("metrics", []))

    beta_cfg: dict[str, Any] = {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "task": {
            "type": _infer_legacy_task_type(metrics),
        },
        "data": {
            "train_path": data_cfg["train_path"],
            "target_col": data_cfg["target_col"],
            "id_col": data_cfg.get("id_col"),
            "group_col": data_cfg.get("group_col"),
            "time_col": data_cfg.get("time_col"),
            "test_path": data_cfg.get("test_path"),
        },
        "features": {
            "numeric": feature_cfg.get("numeric", []),
            "categorical": feature_cfg.get("categorical", []),
            "derived": feature_cfg.get("derived", {"numeric": [], "categorical": []}),
            "text": feature_cfg.get("text"),
            "ignore_text": feature_cfg.get("ignore_text", True),
        },
        "validation": {
            "strategy": inferred["strategy"],
            "protocol_name": validation_cfg.get("protocol_name", inferred["protocol_name"]),
            "split_path": validation_cfg["split_path"],
            "n_splits": inferred["n_splits"],
            "n_repeats": inferred["n_repeats"],
            "shuffle": inferred["shuffle"],
            "random_state": inferred["random_state"],
            "default_repeat_id": validation_cfg.get("default_repeat_id", 0),
        },
        "model": model_cfg,
        "evaluation": {
            "metrics": metrics,
            "primary_metric": metrics[0] if metrics else None,
        },
        "tracking": {
            "backend": "local",
            "experiment_name": "expctl",
            "run_name": config_path.stem,
            "tags": {"compat_mode": "legacy"},
        },
        "output": {
            "dir": output_cfg["dir"],
            "fold_metrics_file": output_cfg.get("fold_metrics_file", "fold_metrics.csv"),
            "oof_predictions_file": output_cfg.get("oof_predictions_file", "oof_predictions.csv"),
            "summary_file": output_cfg["summary_file"],
            "registry_path": output_cfg.get("registry_path", "reports/experiment_registry.csv"),
            "config_snapshot_file": output_cfg.get("config_snapshot_file", "config.snapshot.yaml"),
            "submission_file": output_cfg.get("submission_file", "submission.csv"),
        },
    }
    if "text_vectorizer" in feature_cfg:
        beta_cfg["features"]["text_vectorizer"] = feature_cfg["text_vectorizer"]
    return _normalize_beta_config(beta_cfg, config_path=config_path)


def _infer_legacy_task_type(metrics: list[str]) -> str:
    if any(metric in {"rmse", "mae", "r2"} for metric in metrics):
        return "regression"
    return "binary_classification"


def _infer_legacy_validation(
    validation_cfg: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    split_path_like = str(
        validation_cfg.get("split_path", "artifacts/splits/stratified_kfold_v1.json")
    )
    split_path = repo_root / split_path_like
    protocol_name = str(validation_cfg.get("protocol_name", Path(split_path_like).stem))
    inferred = {
        "strategy": "stratified_kfold",
        "protocol_name": protocol_name,
        "n_splits": int(validation_cfg.get("n_splits", 5)),
        "n_repeats": int(validation_cfg.get("n_repeats", 1)),
        "shuffle": bool(validation_cfg.get("shuffle", True)),
        "random_state": int(validation_cfg.get("random_state", 42)),
    }
    if split_path.exists():
        try:
            with split_path.open("r", encoding="utf-8") as fh:
                artifact = json.load(fh)
            inferred["strategy"] = str(
                artifact.get(
                    "strategy",
                    "repeated_stratified_kfold" if artifact.get("repeated") else "stratified_kfold",
                )
            )
            inferred["n_splits"] = int(artifact.get("n_splits", inferred["n_splits"]))
            inferred["n_repeats"] = int(artifact.get("n_repeats", inferred["n_repeats"]))
            inferred["shuffle"] = bool(artifact.get("shuffle", inferred["shuffle"]))
            inferred["random_state"] = int(artifact.get("seed", inferred["random_state"]))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    hint = f"{split_path_like}::{protocol_name}".lower()
    if "time" in hint:
        inferred["strategy"] = "time_series_split"
        inferred["shuffle"] = False
    elif "stratified_group" in hint:
        inferred["strategy"] = "stratified_group_kfold"
    elif "group" in hint:
        inferred["strategy"] = "group_kfold"
    elif "repeat" in hint or "rskf" in hint:
        inferred["strategy"] = "repeated_stratified_kfold"
        inferred["n_repeats"] = max(inferred["n_repeats"], 2)
    return inferred
