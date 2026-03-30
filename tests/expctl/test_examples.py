from __future__ import annotations

from pathlib import Path

import pytest

from expctl.cli import main

EXAMPLE_CONFIGS = [
    (
        "binary_classification",
        "configs/experiments/experiment.yaml",
        "artifacts/experiments/exp_101_binary_classification/summary.yaml",
    ),
    (
        "regression",
        "configs/experiments/experiment.yaml",
        "artifacts/experiments/exp_102_regression/summary.yaml",
    ),
    (
        "grouped_classification",
        "configs/experiments/experiment.yaml",
        "artifacts/experiments/exp_103_grouped_classification/summary.yaml",
    ),
    (
        "time_series_regression",
        "configs/experiments/experiment.yaml",
        "artifacts/experiments/exp_104_time_series_regression/summary.yaml",
    ),
]


@pytest.mark.parametrize(("example_name", "config_relpath", "summary_relpath"), EXAMPLE_CONFIGS)
def test_official_examples_run_train_workflow(
    example_name: str,
    config_relpath: str,
    summary_relpath: str,
) -> None:
    repo_root = Path("examples") / example_name
    config_path = repo_root / config_relpath

    assert main(["doctor", "--repo-root", str(repo_root), "--config", str(config_path)]) == 0
    assert main(["build-splits", "--repo-root", str(repo_root), "--config", str(config_path)]) == 0
    assert main(["train", "--repo-root", str(repo_root), "--config", str(config_path)]) == 0
    assert (repo_root / summary_relpath).exists()
