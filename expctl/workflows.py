from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from expctl.adapters import AdapterLoadError, ProjectAdapter, load_project_adapter
from expctl.config.beta import load_config_file, normalize_config
from expctl.evaluation import compute_metrics, validate_submission
from expctl.registry import (
    default_row,
    infer_exp_id,
    infer_feature_set,
    infer_split_version,
    infer_stage,
    load_summary,
    metric_value,
    read_registry,
    to_repo_rel,
    upsert_registry_row,
    write_registry,
)
from expctl.splits import build_validation_folds, load_folds, save_folds
from expctl.tracking import (
    check_tracking_backend_dependencies,
    create_tracker,
)
from expctl.training import extract_estimator_classes, pick_all_folds, summarize_metrics


@dataclass(frozen=True)
class RuntimeContext:
    repo_root: Path
    config_path: Path
    config: dict[str, Any]
    adapter: ProjectAdapter
    warnings: list[str]


@dataclass(frozen=True)
class DoctorResult:
    issues: list[str]
    warnings: list[str]

    @property
    def is_healthy(self) -> bool:
        return len(self.issues) == 0


class TextNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X: Any, y: Any | None = None) -> "TextNormalizer":
        return self

    def transform(self, X: Any) -> list[str]:
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            series = X
        else:
            array = np.asarray(X)
            if array.ndim == 2 and array.shape[1] == 1:
                array = array[:, 0]
            series = pd.Series(array)
        return series.fillna("").astype(str).tolist()


class SentenceTransformerEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self._model: Any | None = None

    def fit(self, X: Any, y: Any | None = None) -> "SentenceTransformerEncoder":
        self._ensure_model()
        return self

    def transform(self, X: Any) -> np.ndarray:
        self._ensure_model()
        texts = TextNormalizer().transform(X)
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            device=self.device,
        )
        return np.asarray(embeddings)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence_transformers is required when config.features.text_vectorizer.type="
                "'sentence_transformer'"
            ) from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)


def load_runtime_context(config_path: Path, *, repo_root: Path | None = None) -> RuntimeContext:
    config_path = config_path.resolve()
    repo_root = (repo_root or Path.cwd()).resolve()
    adapter = load_project_adapter(repo_root)
    raw_config = load_config_file(config_path)
    normalized, warnings = normalize_config(
        raw_config,
        repo_root=repo_root,
        config_path=config_path,
        feature_catalog=adapter.feature_catalog(),
    )
    return RuntimeContext(
        repo_root=repo_root,
        config_path=config_path,
        config=normalized,
        adapter=adapter,
        warnings=warnings,
    )


def run_validate_config(context: RuntimeContext) -> dict[str, Any]:
    return {
        "config_path": str(context.config_path),
        "schema_version": int(context.config["schema_version"]),
        "warnings": list(context.warnings),
    }


def run_build_splits(context: RuntimeContext) -> dict[str, Any]:
    dataset = _load_dataset(context)
    config = context.config
    validation_cfg = config["validation"]
    data_cfg = config["data"]
    strategy = validation_cfg["strategy"]
    task_type = config["task"]["type"]

    y: np.ndarray | None = None
    if task_type != "regression" or strategy == "stratified_group_kfold":
        y = dataset[data_cfg["target_col"]].to_numpy()

    groups = None
    if data_cfg.get("group_col"):
        groups = dataset[str(data_cfg["group_col"])].to_numpy()

    time_order = None
    if data_cfg.get("time_col"):
        time_order = pd.to_datetime(dataset[str(data_cfg["time_col"])]).to_numpy()

    artifact = build_validation_folds(
        strategy=strategy,
        protocol_name=validation_cfg["protocol_name"],
        n_splits=int(validation_cfg["n_splits"]),
        y=y,
        groups=groups,
        time_order=time_order,
        shuffle=bool(validation_cfg["shuffle"]),
        random_state=int(validation_cfg["random_state"]),
        n_repeats=int(validation_cfg.get("n_repeats", 1)),
    )
    split_path = context.repo_root / str(validation_cfg["split_path"])
    save_folds(artifact, split_path, overwrite=True)
    return {"split_path": str(split_path), "n_folds": len(artifact["folds"])}


def run_train(context: RuntimeContext) -> dict[str, Any]:
    dataset = _load_dataset(context)
    artifact = load_folds(context.repo_root / str(context.config["validation"]["split_path"]))
    estimator = _build_estimator(context)
    metrics = list(context.config["evaluation"]["metrics"])
    target_col = str(context.config["data"]["target_col"])

    output_dir = _output_dir(context)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = output_dir / str(context.config["output"]["config_snapshot_file"])
    with config_snapshot_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(context.config, fh, sort_keys=False)

    tracker = _create_tracker(context, output_dir=output_dir)
    tracker.log_config(context.config, config_path=context.config_path)
    tracker.log_params(
        {
            "task": context.config["task"]["type"],
            "validation": context.config["validation"],
            "model": context.config["model"],
            "evaluation": context.config["evaluation"],
        }
    )
    tracker.log_tags(context.config["tracking"].get("tags", {}))
    tracker.log_artifact(config_snapshot_path)

    fold_metric_rows: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    for fold in pick_all_folds(artifact):
        fold_id = int(fold["fold_id"])
        repeat_id = int(fold["repeat_id"])
        train_idx = np.asarray(fold["train_indices"], dtype=int)
        valid_idx = np.asarray(fold["valid_indices"], dtype=int)

        X_train = _feature_frame(context, dataset.iloc[train_idx])
        X_valid = _feature_frame(context, dataset.iloc[valid_idx])
        y_train = dataset.iloc[train_idx][target_col].to_numpy()
        y_valid = dataset.iloc[valid_idx][target_col].to_numpy()

        model = _clone_pipeline(estimator)
        model.fit(X_train, y_train)

        y_pred = np.asarray(model.predict(X_valid))
        y_score, score_labels = _predict_scores(
            model,
            X_valid,
            task_type=context.config["task"]["type"],
        )
        fold_metrics = compute_metrics(
            y_true=np.asarray(y_valid),
            y_pred=y_pred,
            y_proba=y_score,
            metric_names=metrics,
            task_type=context.config["task"]["type"],
        )
        fold_metric_rows.append(
            {
                "fold_id": fold_id,
                "repeat_id": repeat_id,
                **fold_metrics,
            }
        )
        oof_frames.append(
            _build_oof_frame(
                valid_indices=valid_idx,
                fold_id=fold_id,
                repeat_id=repeat_id,
                y_true=y_valid,
                y_pred=y_pred,
                y_score=y_score,
                score_labels=score_labels,
            )
        )

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    oof_df = pd.concat(oof_frames, ignore_index=True).sort_values(
        ["repeat_id", "fold_id", "row_index"]
    )
    summary = summarize_metrics(fold_metrics_df, metrics)

    fold_metrics_path = output_dir / str(context.config["output"]["fold_metrics_file"])
    oof_path = output_dir / str(context.config["output"]["oof_predictions_file"])
    summary_path = output_dir / str(context.config["output"]["summary_file"])
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    oof_df.to_csv(oof_path, index=False)
    _write_summary(summary_path, summary)

    tracker.log_artifact(fold_metrics_path)
    tracker.log_artifact(oof_path)
    tracker.log_artifact(summary_path)
    tracker.log_metrics(
        {key: float(value) for key, value in summary.items() if key.endswith("_mean")}
    )
    tracker.finish()

    return {
        "output_dir": str(output_dir),
        "fold_metrics_path": str(fold_metrics_path),
        "oof_predictions_path": str(oof_path),
        "summary_path": str(summary_path),
    }


def run_evaluate(
    context: RuntimeContext,
    *,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = _output_dir(context)
    fold_metrics_path = output_dir / str(context.config["output"]["fold_metrics_file"])
    summary_path = output_dir / str(context.config["output"]["summary_file"])
    metrics = list(context.config["evaluation"]["metrics"])

    if fold_metrics_path.exists():
        fold_metrics_df = pd.read_csv(fold_metrics_path)
        summary = summarize_metrics(fold_metrics_df, metrics)
    else:
        predictions_path = predictions_path or output_dir / str(
            context.config["output"]["oof_predictions_file"]
        )
        oof_df = pd.read_csv(predictions_path)
        y_score = _extract_scores_from_oof(oof_df)
        summary = compute_metrics(
            y_true=oof_df["y_true"].to_numpy(),
            y_pred=oof_df["prediction"].to_numpy(),
            y_proba=y_score,
            metric_names=metrics,
            task_type=context.config["task"]["type"],
        )
    _write_summary(summary_path, summary)
    return {"summary_path": str(summary_path)}


def run_register(context: RuntimeContext) -> dict[str, Any]:
    output_dir = _output_dir(context)
    summary_path = output_dir / str(context.config["output"]["summary_file"])
    summary = load_summary(summary_path)

    registry_path = context.repo_root / str(context.config["output"]["registry_path"])
    rows = read_registry(registry_path)

    exp_name = str(context.config["tracking"].get("run_name") or context.config_path.stem)
    try:
        exp_id = infer_exp_id(exp_name)
    except ValueError:
        exp_id = _allocate_exp_id(rows)

    primary_metric = str(
        context.config["evaluation"].get("primary_metric")
        or context.config["evaluation"]["metrics"][0]
    )
    row = default_row()
    row.update(
        {
            "exp_id": exp_id,
            "exp_name": exp_name,
            "date": date.today().isoformat(),
            "stage": infer_stage(exp_name),
            "split_version": infer_split_version(context.config),
            "config_path": to_repo_rel(context.config_path, repo_root=context.repo_root),
            "artifact_dir": to_repo_rel(output_dir, repo_root=context.repo_root),
            "model_family": str(context.config["model"]["type"]),
            "feature_set": infer_feature_set(context.config),
            "cv_score_mean": metric_value(summary, primary_metric, "mean"),
            "cv_score_std": metric_value(summary, primary_metric, "std"),
            "status": "completed",
            "owner": os.getenv("USER", ""),
            "notes": (
                f"task={context.config['task']['type']};"
                f"tracking={context.config['tracking']['backend']}"
            ),
        }
    )
    updated_rows, action = upsert_registry_row(rows, row)
    write_registry(registry_path, updated_rows)
    return {"registry_path": str(registry_path), "action": action, "exp_id": exp_id}


def run_make_submission(
    context: RuntimeContext,
    *,
    predictions_path: Path,
    output_path: Path | None = None,
    test_path: Path | None = None,
) -> dict[str, Any]:
    data_cfg = context.config["data"]
    id_col = data_cfg.get("id_col")
    if not id_col:
        raise ValueError("config.data.id_col is required for make-submission")

    resolved_test_path = test_path or (
        context.repo_root / str(data_cfg["test_path"]) if data_cfg.get("test_path") else None
    )
    if resolved_test_path is None:
        raise ValueError(
            "Test data path is required. "
            "Set config.data.test_path or pass --test-data."
        )

    test_df = pd.read_csv(resolved_test_path, dtype=str, keep_default_na=False)
    prediction_df = pd.read_csv(predictions_path, dtype=str, keep_default_na=False)

    if "prediction" not in prediction_df.columns:
        raise ValueError("Predictions CSV must contain a 'prediction' column")

    if "id" not in prediction_df.columns:
        if prediction_df.shape[0] != test_df.shape[0]:
            raise ValueError(
                "Predictions without an 'id' column must have the same row count as the test data"
            )
        submission_df = pd.DataFrame(
            {
                "id": test_df[str(id_col)],
                "prediction": prediction_df["prediction"],
            }
        )
    else:
        submission_df = prediction_df[["id", "prediction"]].copy()

    test_for_validation = test_df.rename(columns={str(id_col): "id"})
    result = validate_submission(test_for_validation, submission_df)
    if not result.is_valid:
        raise ValueError(f"Submission validation failed: {result.errors_by_check}")

    output_path = output_path or _output_dir(context) / str(
        context.config["output"]["submission_file"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return {"submission_path": str(output_path)}


def run_doctor(config_path: Path, *, repo_root: Path | None = None) -> DoctorResult:
    repo_root = (repo_root or Path.cwd()).resolve()
    warnings: list[str] = []
    issues: list[str] = []

    try:
        adapter = load_project_adapter(repo_root)
    except AdapterLoadError as exc:
        return DoctorResult(issues=[str(exc)], warnings=[])

    try:
        raw_config = load_config_file(config_path.resolve())
    except Exception as exc:  # noqa: BLE001
        return DoctorResult(issues=[f"Failed to load config: {exc}"], warnings=[])

    try:
        config, normalize_warnings = normalize_config(
            raw_config,
            repo_root=repo_root,
            config_path=config_path.resolve(),
            feature_catalog=adapter.feature_catalog(),
        )
        warnings.extend(normalize_warnings)
    except Exception as exc:  # noqa: BLE001
        return DoctorResult(issues=[str(exc)], warnings=warnings)

    backend = str(config["tracking"]["backend"])
    issues.extend(check_tracking_backend_dependencies(backend))

    feature_cfg = config["features"]
    text_vectorizer = feature_cfg.get("text_vectorizer")
    if isinstance(text_vectorizer, dict) and text_vectorizer.get("type") == "sentence_transformer":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            issues.append(
                "sentence_transformers is required when using config.features.text_vectorizer.type="
                "'sentence_transformer'"
            )

    split_path = repo_root / str(config["validation"]["split_path"])
    if not split_path.exists():
        warnings.append(
            f"Split artifact does not exist yet: {split_path}. Run `expctl build-splits` first."
        )

    try:
        registry = adapter.model_registry()
        registry.resolve(str(config["model"]["type"]))
    except Exception as exc:  # noqa: BLE001
        issues.append(f"Model registry cannot resolve config.model.type: {exc}")

    try:
        dataset = _load_dataset(
            RuntimeContext(
                repo_root=repo_root,
                config_path=config_path.resolve(),
                config=config,
                adapter=adapter,
                warnings=list(warnings),
            )
        )
    except Exception as exc:  # noqa: BLE001
        issues.append(f"Failed to load dataset through project adapter: {exc}")
        return DoctorResult(issues=issues, warnings=warnings)

    issues.extend(_validate_dataset_columns(dataset, config))

    return DoctorResult(issues=issues, warnings=warnings)


def _load_dataset(context: RuntimeContext) -> pd.DataFrame:
    resolved_config = _config_with_resolved_paths(context.config, repo_root=context.repo_root)
    dataset = context.adapter.load_dataset(resolved_config)
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("project.load_dataset(config) must return a pandas.DataFrame")
    return dataset


def _config_with_resolved_paths(config: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    copied = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in config.items()
    }
    data_cfg = dict(copied["data"])
    for key in ("train_path", "test_path"):
        if data_cfg.get(key):
            data_cfg[key] = str((repo_root / str(data_cfg[key])).resolve())
    copied["data"] = data_cfg
    validation_cfg = dict(copied["validation"])
    validation_cfg["split_path"] = str((repo_root / str(validation_cfg["split_path"])).resolve())
    copied["validation"] = validation_cfg
    output_cfg = dict(copied["output"])
    output_cfg["dir"] = str((repo_root / str(output_cfg["dir"])).resolve())
    if output_cfg.get("registry_path"):
        output_cfg["registry_path"] = str((repo_root / str(output_cfg["registry_path"])).resolve())
    copied["output"] = output_cfg
    return copied


def _output_dir(context: RuntimeContext) -> Path:
    return (context.repo_root / str(context.config["output"]["dir"])).resolve()


def _build_estimator(context: RuntimeContext) -> Any:
    registry = context.adapter.model_registry()
    model_cfg = context.config["model"]
    estimator = registry.resolve(str(model_cfg["type"]))(**dict(model_cfg["params"]))
    preprocessor = _build_preprocessor(context.config)
    if preprocessor is None:
        return estimator
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def _build_preprocessor(config: dict[str, Any]) -> ColumnTransformer | None:
    feature_cfg = config["features"]
    numeric_cols = list(feature_cfg.get("numeric", [])) + list(
        feature_cfg.get("derived", {}).get("numeric", [])
    )
    categorical_cols = list(feature_cfg.get("categorical", [])) + list(
        feature_cfg.get("derived", {}).get("categorical", [])
    )
    transformers: list[tuple[str, Any, Any]] = []

    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    if not feature_cfg.get("ignore_text", True):
        text_col = str(feature_cfg["text"])
        text_vectorizer = dict(feature_cfg["text_vectorizer"])
        if text_vectorizer["type"] == "tfidf_svd":
            text_pipeline: Any = Pipeline(
                [
                    ("normalize", TextNormalizer()),
                    (
                        "tfidf",
                        TfidfVectorizer(
                            analyzer=text_vectorizer["analyzer"],
                            ngram_range=tuple(text_vectorizer["ngram_range"]),
                            max_features=int(text_vectorizer["max_features"]),
                            min_df=text_vectorizer.get("min_df", 1),
                            max_df=text_vectorizer.get("max_df", 1.0),
                            sublinear_tf=bool(text_vectorizer.get("sublinear_tf", False)),
                        ),
                    ),
                    (
                        "svd",
                        TruncatedSVD(
                            n_components=int(text_vectorizer["svd_components"]),
                            random_state=text_vectorizer.get("random_state", 42),
                        ),
                    ),
                ]
            )
        else:
            text_pipeline = Pipeline(
                [
                    ("normalize", TextNormalizer()),
                    (
                        "encoder",
                        SentenceTransformerEncoder(
                            model_name=str(text_vectorizer["model_name"]),
                            batch_size=int(text_vectorizer.get("batch_size", 32)),
                            normalize_embeddings=bool(
                                text_vectorizer.get("normalize_embeddings", False)
                            ),
                            device=text_vectorizer.get("device"),
                        ),
                    ),
                ]
            )
        transformers.append(("text", text_pipeline, text_col))

    if not transformers:
        return None
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _feature_frame(context: RuntimeContext, dataset: pd.DataFrame) -> pd.DataFrame:
    feature_cfg = context.config["features"]
    columns = (
        list(feature_cfg.get("numeric", []))
        + list(feature_cfg.get("categorical", []))
        + list(feature_cfg.get("derived", {}).get("numeric", []))
        + list(feature_cfg.get("derived", {}).get("categorical", []))
    )
    if not feature_cfg.get("ignore_text", True) and feature_cfg.get("text") is not None:
        columns.append(str(feature_cfg["text"]))
    missing = [column for column in columns if column not in dataset.columns]
    if missing:
        raise KeyError(f"Dataset is missing required feature columns: {missing}")
    return dataset[columns].copy()


def _required_feature_columns(config: dict[str, Any]) -> list[str]:
    feature_cfg = config["features"]
    columns = (
        list(feature_cfg.get("numeric", []))
        + list(feature_cfg.get("categorical", []))
        + list(feature_cfg.get("derived", {}).get("numeric", []))
        + list(feature_cfg.get("derived", {}).get("categorical", []))
    )
    if not feature_cfg.get("ignore_text", True) and feature_cfg.get("text") is not None:
        columns.append(str(feature_cfg["text"]))
    return columns


def _validate_dataset_columns(dataset: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    data_cfg = config["data"]

    required_columns = [str(data_cfg["target_col"])] + _required_feature_columns(config)
    for optional_column_key in ("id_col", "group_col", "time_col"):
        if data_cfg.get(optional_column_key):
            required_columns.append(str(data_cfg[optional_column_key]))

    missing = sorted({column for column in required_columns if column not in dataset.columns})
    if missing:
        issues.append(f"Dataset is missing required columns: {missing}")
    return issues


def _predict_scores(
    estimator: Any,
    X_valid: pd.DataFrame,
    *,
    task_type: str,
) -> tuple[np.ndarray | None, list[str] | None]:
    if task_type == "regression":
        return None, None
    if hasattr(estimator, "predict_proba"):
        y_score = np.asarray(estimator.predict_proba(X_valid))
        labels = [str(label) for label in extract_estimator_classes(estimator)]
        return y_score, labels
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X_valid)), None
    return None, None


def _build_oof_frame(
    *,
    valid_indices: np.ndarray,
    fold_id: int,
    repeat_id: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    score_labels: list[str] | None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_index": valid_indices.astype(int),
            "fold_id": fold_id,
            "repeat_id": repeat_id,
            "y_true": y_true,
            "prediction": y_pred,
        }
    )
    if y_score is None:
        return frame
    if y_score.ndim == 1:
        frame["score"] = y_score
        return frame
    for idx in range(y_score.shape[1]):
        label = score_labels[idx] if score_labels is not None and idx < len(score_labels) else idx
        frame[f"score_{label}"] = y_score[:, idx]
    return frame


def _extract_scores_from_oof(oof_df: pd.DataFrame) -> np.ndarray | None:
    score_cols = [column for column in oof_df.columns if column.startswith("score_")]
    if score_cols:
        return oof_df[score_cols].to_numpy()
    if "score" in oof_df.columns:
        return oof_df["score"].to_numpy()
    return None


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(summary, fh, sort_keys=True)


def _create_tracker(context: RuntimeContext, *, output_dir: Path):
    tracking_cfg = context.config["tracking"]
    return create_tracker(
        backend=str(tracking_cfg["backend"]),
        output_dir=output_dir,
        experiment_name=str(tracking_cfg["experiment_name"]),
        run_name=str(tracking_cfg["run_name"]),
    )


def _allocate_exp_id(rows: list[dict[str, str]]) -> str:
    existing = []
    for row in rows:
        exp_id = row.get("exp_id", "")
        if exp_id.startswith("exp_"):
            suffix = exp_id.split("_", maxsplit=1)[1]
            if suffix.isdigit():
                existing.append(int(suffix))
    next_id = max(existing, default=0) + 1
    return f"exp_{next_id:03d}"


def _clone_pipeline(estimator: Any) -> Any:
    from sklearn.base import clone

    return clone(estimator)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
