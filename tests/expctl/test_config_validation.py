from __future__ import annotations

from pathlib import Path

import pytest

from expctl.config import FeatureCatalog
from expctl.config.validation import ConfigValidationError, validate_register_experiment_config


def _register_config(split_path: Path) -> dict[str, object]:
    return {
        "validation": {
            "split_path": str(split_path),
        },
        "features": {
            "numeric": ["num_a"],
            "categorical": [],
            "text": "text_body",
            "ignore_text": True,
        },
        "evaluation": {
            "metrics": ["accuracy"],
        },
        "output": {
            "dir": "artifacts/exp_001",
            "summary_file": "summary.yaml",
        },
    }


def test_register_config_rejects_unsupported_metrics(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    split_path.write_text("{}", encoding="utf-8")
    config = _register_config(split_path)
    config["evaluation"] = {"metrics": ["not_supported"]}

    with pytest.raises(ConfigValidationError, match="Unsupported metrics"):
        validate_register_experiment_config(
            config,
            repo_root=tmp_path,
            config_path=tmp_path / "register.yaml",
            feature_catalog=FeatureCatalog(),
        )


def test_register_config_requires_text_vectorizer_when_text_enabled(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    split_path.write_text("{}", encoding="utf-8")
    config = _register_config(split_path)
    config["features"] = {
        "numeric": ["num_a"],
        "categorical": [],
        "text": "text_body",
        "ignore_text": False,
    }

    with pytest.raises(
        ConfigValidationError,
        match="config\\.features\\.ignore_text=false requires config\\.features\\.text_vectorizer",
    ):
        validate_register_experiment_config(
            config,
            repo_root=tmp_path,
            config_path=tmp_path / "register.yaml",
            feature_catalog=FeatureCatalog(),
        )


def test_register_config_accepts_supported_metric_and_text_vectorizer(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    split_path.write_text("{}", encoding="utf-8")
    config = _register_config(split_path)
    config["features"] = {
        "numeric": ["num_a"],
        "categorical": [],
        "text": "text_body",
        "ignore_text": False,
        "text_vectorizer": {
            "type": "tfidf_svd",
            "analyzer": "word",
            "ngram_range": [1, 2],
            "max_features": 128,
            "svd_components": 16,
        },
    }

    validate_register_experiment_config(
        config,
        repo_root=tmp_path,
        config_path=tmp_path / "register.yaml",
        feature_catalog=FeatureCatalog(),
    )
