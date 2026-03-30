from __future__ import annotations

from pathlib import Path
from typing import Any

from expctl.config.contracts import FeatureCatalog


class ConfigValidationError(ValueError):
    """Raised when a YAML config is structurally invalid for a given workflow."""


def _as_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigValidationError(
            f"{context} must be a mapping/dict, got {type(value).__name__}"
        )
    return value


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigValidationError(f"Missing required key '{context}.{key}'")
    return mapping[key]


def _require_str(mapping: dict[str, Any], key: str, context: str) -> str:
    value = _require(mapping, key, context)
    if not isinstance(value, str):
        raise ConfigValidationError(
            f"'{context}.{key}' must be a string, got {type(value).__name__}"
        )
    stripped = value.strip()
    if not stripped:
        raise ConfigValidationError(f"'{context}.{key}' must not be empty")
    return stripped


def _optional_str(mapping: dict[str, Any], key: str, context: str) -> str | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigValidationError(
            f"'{context}.{key}' must be a string or null, got {type(value).__name__}"
        )
    stripped = value.strip()
    if not stripped:
        raise ConfigValidationError(f"'{context}.{key}' must not be an empty string")
    return stripped


def _require_bool(mapping: dict[str, Any], key: str, context: str) -> bool:
    value = _require(mapping, key, context)
    if not isinstance(value, bool):
        raise ConfigValidationError(
            f"'{context}.{key}' must be a boolean, got {type(value).__name__}"
        )
    return value


def _optional_bool(mapping: dict[str, Any], key: str, context: str) -> bool | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if not isinstance(value, bool):
        raise ConfigValidationError(
            f"'{context}.{key}' must be a boolean, got {type(value).__name__}"
        )
    return value


def _optional_int(mapping: dict[str, Any], key: str, context: str) -> int | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(
            f"'{context}.{key}' must be an integer, got {type(value).__name__}"
        )
    return value


def _optional_float(mapping: dict[str, Any], key: str, context: str) -> float | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(
            f"'{context}.{key}' must be a float, got {type(value).__name__}"
        )
    return float(value)


def _require_mapping(mapping: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    return _as_mapping(_require(mapping, key, context), f"{context}.{key}")


def _optional_mapping(mapping: dict[str, Any], key: str, context: str) -> dict[str, Any] | None:
    if key not in mapping:
        return None
    return _as_mapping(mapping[key], f"{context}.{key}")


def _require_str_list(mapping: dict[str, Any], key: str, context: str) -> list[str]:
    value = _require(mapping, key, context)
    if not isinstance(value, list):
        raise ConfigValidationError(
            f"'{context}.{key}' must be a list, got {type(value).__name__}"
        )
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ConfigValidationError(
                f"'{context}.{key}[{idx}]' must be a string, got {type(item).__name__}"
            )
        stripped = item.strip()
        if not stripped:
            raise ConfigValidationError(f"'{context}.{key}[{idx}]' must not be empty")
        out.append(stripped)
    return out


def _optional_str_list(mapping: dict[str, Any], key: str, context: str) -> list[str]:
    if key not in mapping:
        return []
    return _require_str_list(mapping, key, context)


def _find_duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _validate_derived_features(
    feature_cfg: dict[str, Any],
    context: str,
    feature_catalog: FeatureCatalog,
) -> tuple[list[str], list[str]]:
    derived_cfg = _optional_mapping(feature_cfg, "derived", context)
    if derived_cfg is None:
        return [], []

    derived_numeric = _optional_str_list(derived_cfg, "numeric", f"{context}.derived")
    derived_categorical = _optional_str_list(derived_cfg, "categorical", f"{context}.derived")

    unsupported_numeric = sorted(
        set(derived_numeric) - set(feature_catalog.derived_numeric_features)
    )
    unsupported_categorical = sorted(
        set(derived_categorical) - set(feature_catalog.derived_categorical_features)
    )
    if unsupported_numeric:
        raise ConfigValidationError(
            "Unsupported derived numeric features in config.features.derived.numeric: "
            f"{unsupported_numeric}; supported={sorted(feature_catalog.derived_numeric_features)}"
        )
    if unsupported_categorical:
        raise ConfigValidationError(
            "Unsupported derived categorical features in config.features.derived.categorical: "
            f"{unsupported_categorical}; "
            f"supported={sorted(feature_catalog.derived_categorical_features)}"
        )
    return derived_numeric, derived_categorical


def _validate_text_vectorizer(
    feature_cfg: dict[str, Any],
    context: str,
    feature_catalog: FeatureCatalog,
) -> dict[str, Any] | None:
    text_vectorizer_cfg = _optional_mapping(feature_cfg, "text_vectorizer", context)
    if text_vectorizer_cfg is None:
        return None

    vectorizer_type = _require_str(text_vectorizer_cfg, "type", f"{context}.text_vectorizer")
    if vectorizer_type not in feature_catalog.text_vectorizers:
        raise ConfigValidationError(
            "Unsupported text vectorizer type in config.features.text_vectorizer.type: "
            f"{vectorizer_type}; supported={sorted(feature_catalog.text_vectorizers)}"
        )

    if vectorizer_type == "tfidf_svd":
        analyzer = _require_str(text_vectorizer_cfg, "analyzer", f"{context}.text_vectorizer")
        if analyzer not in feature_catalog.tfidf_analyzers:
            raise ConfigValidationError(
                "Unsupported analyzer in config.features.text_vectorizer.analyzer: "
                f"{analyzer}; supported={sorted(feature_catalog.tfidf_analyzers)}"
            )

        ngram_range = _require(text_vectorizer_cfg, "ngram_range", f"{context}.text_vectorizer")
        if (
            not isinstance(ngram_range, list)
            or len(ngram_range) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in ngram_range)
        ):
            raise ConfigValidationError(
                f"'{context}.text_vectorizer.ngram_range' must be a 2-item integer list"
            )
        if ngram_range[0] < 1 or ngram_range[0] > ngram_range[1]:
            raise ConfigValidationError(
                f"'{context}.text_vectorizer.ngram_range' must satisfy 1 <= low <= high"
            )

        max_features = _optional_int(
            text_vectorizer_cfg, "max_features", f"{context}.text_vectorizer"
        )
        if max_features is None or max_features < 1:
            raise ConfigValidationError(
                f"'{context}.text_vectorizer.max_features' must be an integer >= 1"
            )

        svd_components = _optional_int(
            text_vectorizer_cfg, "svd_components", f"{context}.text_vectorizer"
        )
        if svd_components is None or svd_components < 1:
            raise ConfigValidationError(
                f"'{context}.text_vectorizer.svd_components' must be an integer >= 1"
            )

        min_df = text_vectorizer_cfg.get("min_df", 1)
        if isinstance(min_df, bool) or not isinstance(min_df, (int, float)) or float(min_df) <= 0:
            raise ConfigValidationError(
                f"'{context}.text_vectorizer.min_df' must be a positive int/float"
            )

        max_df = text_vectorizer_cfg.get("max_df", 1.0)
        if isinstance(max_df, bool) or not isinstance(max_df, (int, float)) or float(max_df) <= 0:
            raise ConfigValidationError(
                f"'{context}.text_vectorizer.max_df' must be a positive int/float"
            )

        if "sublinear_tf" in text_vectorizer_cfg:
            _require_bool(text_vectorizer_cfg, "sublinear_tf", f"{context}.text_vectorizer")
        _optional_int(text_vectorizer_cfg, "random_state", f"{context}.text_vectorizer")
        return text_vectorizer_cfg

    _require_str(text_vectorizer_cfg, "model_name", f"{context}.text_vectorizer")
    batch_size = _optional_int(text_vectorizer_cfg, "batch_size", f"{context}.text_vectorizer")
    if batch_size is not None and batch_size < 1:
        raise ConfigValidationError(
            f"'{context}.text_vectorizer.batch_size' must be an integer >= 1"
        )
    if "normalize_embeddings" in text_vectorizer_cfg:
        _require_bool(text_vectorizer_cfg, "normalize_embeddings", f"{context}.text_vectorizer")
    _optional_str(text_vectorizer_cfg, "device", f"{context}.text_vectorizer")
    return text_vectorizer_cfg


def _validate_numeric_clip_quantiles(
    preprocessing_cfg: dict[str, Any] | None,
    context: str,
) -> tuple[float, float] | None:
    if preprocessing_cfg is None or "numeric_clip_quantiles" not in preprocessing_cfg:
        return None
    value = preprocessing_cfg["numeric_clip_quantiles"]
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigValidationError(
            f"'{context}.numeric_clip_quantiles' must be a 2-item list like [0.01, 0.99]"
        )
    lower_raw, upper_raw = value
    if isinstance(lower_raw, bool) or not isinstance(lower_raw, (int, float)):
        raise ConfigValidationError(f"'{context}.numeric_clip_quantiles[0]' must be a float")
    if isinstance(upper_raw, bool) or not isinstance(upper_raw, (int, float)):
        raise ConfigValidationError(f"'{context}.numeric_clip_quantiles[1]' must be a float")
    lower = float(lower_raw)
    upper = float(upper_raw)
    if not 0.0 <= lower < upper <= 1.0:
        raise ConfigValidationError(
            f"'{context}.numeric_clip_quantiles' must satisfy 0 <= lower < upper <= 1"
        )
    return lower, upper


def _validate_decision_config(root: dict[str, Any]) -> float | None:
    decision_cfg = _optional_mapping(root, "decision", "config")
    if decision_cfg is None:
        return None
    threshold = _optional_float(decision_cfg, "positive_class_threshold", "config.decision")
    if threshold is None:
        return None
    if not 0.0 < threshold < 1.0:
        raise ConfigValidationError(
            "'config.decision.positive_class_threshold' must satisfy 0 < threshold < 1"
        )
    return threshold


def _validate_seed_ensemble(model_cfg: dict[str, Any], context: str) -> list[int] | None:
    seed_ensemble_cfg = _optional_mapping(model_cfg, "seed_ensemble", context)
    if seed_ensemble_cfg is None:
        return None
    random_states = _require(seed_ensemble_cfg, "random_states", f"{context}.seed_ensemble")
    if not isinstance(random_states, list) or len(random_states) < 2:
        raise ConfigValidationError(
            f"'{context}.seed_ensemble.random_states' must be a list with at least 2 integers"
        )
    normalized: list[int] = []
    for idx, value in enumerate(random_states):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigValidationError(
                f"'{context}.seed_ensemble.random_states[{idx}]' must be an integer"
            )
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ConfigValidationError(
            f"'{context}.seed_ensemble.random_states' must not contain duplicates"
        )
    return normalized


def _validate_pseudo_labeling(root: dict[str, Any], repo_root: Path) -> dict[str, Any] | None:
    semi_supervised_cfg = _optional_mapping(root, "semi_supervised", "config")
    if semi_supervised_cfg is None:
        return None
    pseudo_cfg = _optional_mapping(semi_supervised_cfg, "pseudo_labeling", "config.semi_supervised")
    if pseudo_cfg is None:
        return None

    pool_path_like = _require_str(pseudo_cfg, "pool_path", "config.semi_supervised.pseudo_labeling")
    _resolve_repo_path(
        repo_root,
        pool_path_like,
        "config.semi_supervised.pseudo_labeling.pool_path",
        must_exist=True,
        must_be_file=True,
    )
    lower = _optional_float(
        pseudo_cfg, "lower_threshold", "config.semi_supervised.pseudo_labeling"
    )
    upper = _optional_float(
        pseudo_cfg, "upper_threshold", "config.semi_supervised.pseudo_labeling"
    )
    if lower is None or upper is None:
        raise ConfigValidationError(
            "config.semi_supervised.pseudo_labeling requires both "
            "lower_threshold and upper_threshold"
        )
    if not 0.0 < lower < 0.5 < upper < 1.0:
        raise ConfigValidationError(
            "config.semi_supervised.pseudo_labeling thresholds must satisfy "
            "0 < lower < 0.5 < upper < 1"
        )
    return pseudo_cfg


def _validate_training_config(root: dict[str, Any], repo_root: Path) -> dict[str, Any] | None:
    training_cfg = _optional_mapping(root, "training", "config")
    if training_cfg is None:
        return None

    suspicious_cfg = _optional_mapping(training_cfg, "suspicious_rows", "config.training")
    if suspicious_cfg is None:
        return training_cfg

    path_like = _require_str(suspicious_cfg, "path", "config.training.suspicious_rows")
    _resolve_repo_path(
        repo_root,
        path_like,
        "config.training.suspicious_rows.path",
        must_exist=True,
        must_be_file=True,
    )
    _require_str(suspicious_cfg, "score_col", "config.training.suspicious_rows")
    mode = _require_str(suspicious_cfg, "mode", "config.training.suspicious_rows")
    if mode not in {"drop", "downweight", "drop_then_downweight"}:
        raise ConfigValidationError(
            "config.training.suspicious_rows.mode must be "
            "'drop', 'downweight', or 'drop_then_downweight'"
        )
    top_k = _optional_int(suspicious_cfg, "top_k", "config.training.suspicious_rows")
    drop_top_k = _optional_int(
        suspicious_cfg,
        "drop_top_k",
        "config.training.suspicious_rows",
    )
    downweight_next_k = _optional_int(
        suspicious_cfg,
        "downweight_next_k",
        "config.training.suspicious_rows",
    )
    if top_k is None or top_k < 1:
        if mode in {"drop", "downweight"}:
            raise ConfigValidationError(
                "config.training.suspicious_rows.top_k must be an integer >= 1"
            )
    factor = _optional_float(
        suspicious_cfg,
        "downweight_factor",
        "config.training.suspicious_rows",
    )
    if mode == "downweight":
        if factor is None:
            raise ConfigValidationError(
                "config.training.suspicious_rows.downweight_factor is required "
                "when mode='downweight'"
            )
        if not 0.0 <= factor < 1.0:
            raise ConfigValidationError(
                "config.training.suspicious_rows.downweight_factor must satisfy 0 <= factor < 1"
            )
        if drop_top_k is not None or downweight_next_k is not None:
            raise ConfigValidationError(
                "config.training.suspicious_rows.drop_top_k and "
                "downweight_next_k are only valid when mode='drop_then_downweight'"
            )
    elif mode == "drop":
        if factor is not None:
            raise ConfigValidationError(
                "config.training.suspicious_rows.downweight_factor is only valid "
                "when mode='downweight' or mode='drop_then_downweight'"
            )
        if drop_top_k is not None or downweight_next_k is not None:
            raise ConfigValidationError(
                "config.training.suspicious_rows.drop_top_k and "
                "downweight_next_k are only valid when mode='drop_then_downweight'"
            )
    else:
        if top_k is not None:
            raise ConfigValidationError(
                "config.training.suspicious_rows.top_k is not used when "
                "mode='drop_then_downweight'; use drop_top_k + downweight_next_k"
            )
        if drop_top_k is None or drop_top_k < 1:
            raise ConfigValidationError(
                "config.training.suspicious_rows.drop_top_k must be an integer >= 1 "
                "when mode='drop_then_downweight'"
            )
        if downweight_next_k is None or downweight_next_k < 1:
            raise ConfigValidationError(
                "config.training.suspicious_rows.downweight_next_k must be an "
                "integer >= 1 when mode='drop_then_downweight'"
            )
        if factor is None:
            raise ConfigValidationError(
                "config.training.suspicious_rows.downweight_factor is required "
                "when mode='drop_then_downweight'"
            )
        if not 0.0 <= factor < 1.0:
            raise ConfigValidationError(
                "config.training.suspicious_rows.downweight_factor must satisfy 0 <= factor < 1"
            )
    if mode == "drop_then_downweight" and factor is None:
        raise ConfigValidationError(
            "config.training.suspicious_rows.downweight_factor is required "
            "when mode='drop_then_downweight'"
        )
    if mode not in {"downweight", "drop_then_downweight"} and factor is not None:
        raise ConfigValidationError(
            "config.training.suspicious_rows.downweight_factor is only valid "
            "when mode='downweight' or mode='drop_then_downweight'"
        )
    return training_cfg


def _resolve_repo_path(
    repo_root: Path,
    path_like: str,
    field: str,
    *,
    must_exist: bool,
    must_be_file: bool = False,
) -> Path:
    resolved = repo_root / path_like
    if must_exist and not resolved.exists():
        raise ConfigValidationError(f"'{field}' points to missing path: {resolved}")
    if must_be_file and resolved.exists() and not resolved.is_file():
        raise ConfigValidationError(f"'{field}' must point to a file: {resolved}")
    return resolved


def validate_train_experiment_config(
    cfg: dict[str, Any],
    *,
    repo_root: Path,
    config_path: Path,
    feature_catalog: FeatureCatalog,
) -> None:
    root = _as_mapping(cfg, f"config<{config_path}>")
    data_cfg = _require_mapping(root, "data", "config")
    feature_cfg = _require_mapping(root, "features", "config")
    validation_cfg = _require_mapping(root, "validation", "config")
    model_cfg = _require_mapping(root, "model", "config")
    eval_cfg = _require_mapping(root, "evaluation", "config")
    output_cfg = _require_mapping(root, "output", "config")
    preprocessing_cfg = _optional_mapping(root, "preprocessing", "config")

    train_path_like = _require_str(data_cfg, "train_path", "config.data")
    _resolve_repo_path(
        repo_root,
        train_path_like,
        "config.data.train_path",
        must_exist=True,
        must_be_file=True,
    )
    _require_str(data_cfg, "target_col", "config.data")
    _optional_str(data_cfg, "id_col", "config.data")

    numeric_cols = _require_str_list(feature_cfg, "numeric", "config.features")
    categorical_cols = _require_str_list(feature_cfg, "categorical", "config.features")
    derived_numeric, derived_categorical = _validate_derived_features(
        feature_cfg,
        "config.features",
        feature_catalog,
    )
    _require_str(feature_cfg, "text", "config.features")
    ignore_text = _require_bool(feature_cfg, "ignore_text", "config.features")
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
    ):
        raise ConfigValidationError("At least one feature column is required in config.features")
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
    if not ignore_text and text_vectorizer_cfg is None:
        raise ConfigValidationError(
            "config.features.ignore_text=false requires config.features.text_vectorizer"
        )

    split_path_like = _require_str(validation_cfg, "split_path", "config.validation")
    _resolve_repo_path(
        repo_root,
        split_path_like,
        "config.validation.split_path",
        must_exist=True,
        must_be_file=True,
    )
    _require_str(validation_cfg, "protocol_name", "config.validation")
    _optional_int(validation_cfg, "default_repeat_id", "config.validation")

    _require_str(model_cfg, "type", "config.model")
    _require_mapping(model_cfg, "params", "config.model")
    _validate_seed_ensemble(model_cfg, "config.model")

    metrics = _require_str_list(eval_cfg, "metrics", "config.evaluation")
    if not metrics:
        raise ConfigValidationError("config.evaluation.metrics must not be empty")
    unsupported = [metric for metric in metrics if metric not in feature_catalog.metrics]
    if unsupported:
        raise ConfigValidationError(
            "Unsupported metrics in config.evaluation.metrics: "
            f"{unsupported}; supported={sorted(feature_catalog.metrics)}"
        )

    _require_str(output_cfg, "dir", "config.output")
    _require_str(output_cfg, "fold_metrics_file", "config.output")
    _require_str(output_cfg, "oof_predictions_file", "config.output")
    _require_str(output_cfg, "summary_file", "config.output")

    if preprocessing_cfg is not None:
        _optional_bool(preprocessing_cfg, "use_missing_indicator", "config.preprocessing")
    _validate_numeric_clip_quantiles(preprocessing_cfg, "config.preprocessing")
    _validate_decision_config(root)
    _validate_pseudo_labeling(root, repo_root)
    training_cfg = _validate_training_config(root, repo_root)
    if training_cfg is not None and _optional_mapping(
        training_cfg, "suspicious_rows", "config.training"
    ) is not None and _optional_str(data_cfg, "id_col", "config.data") is None:
        raise ConfigValidationError(
            "config.data.id_col is required when config.training.suspicious_rows is enabled"
        )


def validate_generate_submission_config(
    cfg: dict[str, Any],
    *,
    repo_root: Path,
    config_path: Path,
    feature_catalog: FeatureCatalog,
) -> None:
    root = _as_mapping(cfg, f"config<{config_path}>")
    data_cfg = _require_mapping(root, "data", "config")
    feature_cfg = _require_mapping(root, "features", "config")
    model_cfg = _require_mapping(root, "model", "config")
    output_cfg = _require_mapping(root, "output", "config")
    preprocessing_cfg = _optional_mapping(root, "preprocessing", "config")

    train_path_like = _require_str(data_cfg, "train_path", "config.data")
    _resolve_repo_path(
        repo_root,
        train_path_like,
        "config.data.train_path",
        must_exist=True,
        must_be_file=True,
    )
    _require_str(data_cfg, "target_col", "config.data")
    if _optional_str(data_cfg, "id_col", "config.data") is None:
        raise ConfigValidationError("config.data.id_col is required for submission generation")

    numeric_cols = _require_str_list(feature_cfg, "numeric", "config.features")
    categorical_cols = _require_str_list(feature_cfg, "categorical", "config.features")
    derived_numeric, derived_categorical = _validate_derived_features(
        feature_cfg,
        "config.features",
        feature_catalog,
    )
    _require_str(feature_cfg, "text", "config.features")
    ignore_text = _require_bool(feature_cfg, "ignore_text", "config.features")
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
    ):
        raise ConfigValidationError("At least one feature column is required in config.features")
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
    if not ignore_text and text_vectorizer_cfg is None:
        raise ConfigValidationError(
            "config.features.ignore_text=false requires config.features.text_vectorizer"
        )

    _require_str(model_cfg, "type", "config.model")
    _require_mapping(model_cfg, "params", "config.model")
    _validate_seed_ensemble(model_cfg, "config.model")
    _require_str(output_cfg, "dir", "config.output")
    if preprocessing_cfg is not None:
        _optional_bool(preprocessing_cfg, "use_missing_indicator", "config.preprocessing")
    training_cfg = _validate_training_config(root, repo_root)
    if training_cfg is not None and _optional_mapping(
        training_cfg, "suspicious_rows", "config.training"
    ) is not None:
        _require_str(data_cfg, "id_col", "config.data")
    _validate_numeric_clip_quantiles(preprocessing_cfg, "config.preprocessing")
    _validate_decision_config(root)
    _validate_pseudo_labeling(root, repo_root)


def validate_register_experiment_config(
    cfg: dict[str, Any],
    *,
    repo_root: Path,
    config_path: Path,
    feature_catalog: FeatureCatalog,
) -> None:
    root = _as_mapping(cfg, f"config<{config_path}>")
    validation_cfg = _require_mapping(root, "validation", "config")
    feature_cfg = _require_mapping(root, "features", "config")
    eval_cfg = _require_mapping(root, "evaluation", "config")
    output_cfg = _require_mapping(root, "output", "config")
    preprocessing_cfg = _optional_mapping(root, "preprocessing", "config")
    model_cfg = _optional_mapping(root, "model", "config")

    split_path_like = _require_str(validation_cfg, "split_path", "config.validation")
    _resolve_repo_path(
        repo_root,
        split_path_like,
        "config.validation.split_path",
        must_exist=True,
        must_be_file=True,
    )

    numeric_cols = _require_str_list(feature_cfg, "numeric", "config.features")
    categorical_cols = _require_str_list(feature_cfg, "categorical", "config.features")
    derived_numeric, derived_categorical = _validate_derived_features(
        feature_cfg,
        "config.features",
        feature_catalog,
    )
    _require_str(feature_cfg, "text", "config.features")
    ignore_text = (
        _require_bool(feature_cfg, "ignore_text", "config.features")
        if "ignore_text" in feature_cfg
        else True
    )
    text_vectorizer_cfg = _validate_text_vectorizer(
        feature_cfg,
        "config.features",
        feature_catalog,
    )
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
    if not ignore_text and text_vectorizer_cfg is None:
        raise ConfigValidationError(
            "config.features.ignore_text=false requires config.features.text_vectorizer"
        )

    metrics = _require_str_list(eval_cfg, "metrics", "config.evaluation")
    if not metrics:
        raise ConfigValidationError("config.evaluation.metrics must not be empty")
    unsupported = [metric for metric in metrics if metric not in feature_catalog.metrics]
    if unsupported:
        raise ConfigValidationError(
            "Unsupported metrics in config.evaluation.metrics: "
            f"{unsupported}; supported={sorted(feature_catalog.metrics)}"
        )

    _require_str(output_cfg, "dir", "config.output")
    _require_str(output_cfg, "summary_file", "config.output")

    if preprocessing_cfg is not None:
        _optional_bool(preprocessing_cfg, "use_missing_indicator", "config.preprocessing")
    _validate_numeric_clip_quantiles(preprocessing_cfg, "config.preprocessing")
    _validate_decision_config(root)
    if model_cfg is not None and "type" in model_cfg:
        _require_str(model_cfg, "type", "config.model")
        _validate_seed_ensemble(model_cfg, "config.model")
    _validate_pseudo_labeling(root, repo_root)
