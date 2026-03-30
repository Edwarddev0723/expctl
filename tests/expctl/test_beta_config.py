from __future__ import annotations

from pathlib import Path

import pytest

from expctl.config import FeatureCatalog
from expctl.config.beta import LEGACY_COMPAT_WARNING, normalize_config, validate_beta_config
from expctl.config.validation import ConfigValidationError


def _beta_config(train_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "task": {"type": "binary_classification"},
        "data": {
            "train_path": str(train_path),
            "target_col": "target",
            "id_col": "id",
        },
        "features": {
            "numeric": ["feature_1"],
            "categorical": [],
            "derived": {"numeric": [], "categorical": []},
            "text": None,
            "ignore_text": True,
        },
        "validation": {
            "strategy": "stratified_kfold",
            "split_path": "artifacts/splits/stratified_kfold_v1.json",
            "n_splits": 2,
            "shuffle": True,
            "random_state": 42,
        },
        "model": {"type": "logistic_regression", "params": {}},
        "evaluation": {"metrics": ["accuracy"], "primary_metric": "accuracy"},
        "tracking": {"backend": "local", "experiment_name": "expctl", "run_name": "exp_001"},
        "output": {
            "dir": "artifacts/experiments/exp_001",
            "fold_metrics_file": "fold_metrics.csv",
            "oof_predictions_file": "oof_predictions.csv",
            "summary_file": "summary.yaml",
        },
    }


def test_validate_beta_config_rejects_regression_with_stratified_strategy(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    train_path.write_text("id,feature_1,target\n1,1.0,0\n2,2.0,1\n", encoding="utf-8")
    config = _beta_config(train_path)
    config["task"] = {"type": "regression"}

    with pytest.raises(ConfigValidationError, match="Regression tasks require"):
        validate_beta_config(
            config,
            repo_root=tmp_path,
            config_path=tmp_path / "experiment.yaml",
            feature_catalog=FeatureCatalog(),
        )


def test_normalize_legacy_config_emits_compat_warning(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    train_path.write_text("id,feature_1,target\n1,1.0,0\n2,2.0,1\n", encoding="utf-8")
    legacy_config = {
        "data": {
            "train_path": str(train_path),
            "target_col": "target",
            "id_col": "id",
        },
        "features": {
            "numeric": ["feature_1"],
            "categorical": [],
            "derived": {"numeric": [], "categorical": []},
            "text": "text",
            "ignore_text": True,
        },
        "validation": {
            "split_path": "artifacts/splits/legacy.json",
            "protocol_name": "legacy",
        },
        "model": {"type": "logistic_regression", "params": {}},
        "evaluation": {"metrics": ["accuracy"]},
        "output": {
            "dir": "artifacts/experiments/legacy",
            "fold_metrics_file": "fold_metrics.csv",
            "oof_predictions_file": "oof_predictions.csv",
            "summary_file": "summary.yaml",
        },
    }

    normalized, warnings = normalize_config(
        legacy_config,
        repo_root=tmp_path,
        config_path=tmp_path / "legacy.yaml",
        feature_catalog=FeatureCatalog(),
    )

    assert normalized["schema_version"] == 1
    assert normalized["tracking"]["backend"] == "local"
    assert warnings == [LEGACY_COMPAT_WARNING]


def test_validate_beta_config_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    train_path.write_text("id,feature_1,target\n1,1.0,0\n2,2.0,1\n", encoding="utf-8")
    config = _beta_config(train_path)
    config["unexpected"] = {}

    with pytest.raises(ConfigValidationError, match="Unknown top-level config keys"):
        validate_beta_config(
            config,
            repo_root=tmp_path,
            config_path=tmp_path / "experiment.yaml",
            feature_catalog=FeatureCatalog(),
        )
