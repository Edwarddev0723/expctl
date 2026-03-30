from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from expctl.cli import main


def test_cli_init_creates_scaffold(tmp_path: Path) -> None:
    project_root = tmp_path / "demo_project"

    exit_code = main(["init", str(project_root)])

    assert exit_code == 0
    assert (project_root / "configs" / "experiments" / "example_classification.yaml").exists()
    assert (project_root / "configs" / "experiments" / "example_regression.yaml").exists()
    assert (
        project_root / "configs" / "experiments" / "example_grouped_classification.yaml"
    ).exists()
    assert (
        project_root / "configs" / "experiments" / "example_time_series_regression.yaml"
    ).exists()
    assert (project_root / "src" / "project" / "__init__.py").exists()
    assert (project_root / "reports" / "experiment_registry.csv").exists()
    assert (project_root / "docs" / "quickstart.md").exists()
    assert (project_root / "artifacts" / "splits" / "README.md").exists()


def test_cli_end_to_end_local_workflow(tmp_path: Path) -> None:
    project_root = tmp_path / "workflow_project"
    assert main(["init", str(project_root)]) == 0

    train_df = pd.DataFrame(
        {
            "id": [f"train_{idx}" for idx in range(10)],
            "feature_1": [0.0, 1.0] * 5,
            "feature_2": [1.0, 0.0] * 5,
            "target": [0, 1] * 5,
        }
    )
    test_df = pd.DataFrame(
        {
            "id": ["test_1", "test_2"],
            "feature_1": [0.0, 1.0],
            "feature_2": [1.0, 0.0],
        }
    )
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    config_path = project_root / "configs" / "experiments" / "example_classification.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    config["validation"]["n_splits"] = 2
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    common_args = ["--config", str(config_path), "--repo-root", str(project_root)]
    assert main(["doctor", *common_args]) == 0
    assert main(["validate-config", *common_args]) == 0
    assert main(["build-splits", *common_args]) == 0
    assert main(["train", *common_args]) == 0
    assert main(["evaluate", *common_args]) == 0
    assert main(["register", *common_args]) == 0

    output_dir = project_root / "artifacts" / "experiments" / "exp_001_baseline"
    assert (output_dir / "fold_metrics.csv").exists()
    assert (output_dir / "oof_predictions.csv").exists()
    assert (output_dir / "summary.yaml").exists()
    assert (output_dir / "tracking.local.json").exists()

    registry_path = project_root / "reports" / "experiment_registry.csv"
    registry_df = pd.read_csv(registry_path)
    assert not registry_df.empty
    assert registry_df.iloc[0]["exp_id"] == "exp_001"

    predictions_path = project_root / "predictions.csv"
    pd.DataFrame({"prediction": ["0", "1"]}).to_csv(predictions_path, index=False)
    assert main(
        [
            "make-submission",
            *common_args,
            "--predictions",
            str(predictions_path),
        ]
    ) == 0
    assert (output_dir / "submission.csv").exists()


def test_cli_end_to_end_regression_group_workflow(tmp_path: Path) -> None:
    project_root = tmp_path / "regression_project"
    assert main(["init", str(project_root)]) == 0

    train_df = pd.DataFrame(
        {
            "id": [f"train_{idx}" for idx in range(9)],
            "account_id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "feature_1": [float(idx) for idx in range(9)],
            "feature_2": [float(idx + 10) for idx in range(9)],
            "target": [0.5 + idx * 0.2 for idx in range(9)],
        }
    )
    test_df = pd.DataFrame(
        {
            "id": ["test_1", "test_2"],
            "feature_1": [9.0, 10.0],
            "feature_2": [19.0, 20.0],
        }
    )
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    config_path = project_root / "configs" / "experiments" / "example_regression.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    config["data"]["group_col"] = "account_id"
    config["validation"]["n_splits"] = 3
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    common_args = ["--config", str(config_path), "--repo-root", str(project_root)]
    assert main(["doctor", *common_args]) == 0
    assert main(["build-splits", *common_args]) == 0
    assert main(["train", *common_args]) == 0
    assert main(["evaluate", *common_args]) == 0
    assert main(["register", *common_args]) == 0

    output_dir = project_root / "artifacts" / "experiments" / "exp_010_regression_baseline"
    assert (output_dir / "summary.yaml").exists()


def test_cli_end_to_end_time_series_workflow(tmp_path: Path) -> None:
    project_root = tmp_path / "time_series_project"
    assert main(["init", str(project_root)]) == 0

    train_df = pd.DataFrame(
        {
            "id": [f"train_{idx}" for idx in range(8)],
            "event_time": pd.date_range("2024-01-01", periods=8, freq="D"),
            "feature_1": [float(idx) for idx in range(8)],
            "feature_2": [float(idx + 100) for idx in range(8)],
            "target": [1.0 + idx * 0.3 for idx in range(8)],
        }
    )
    test_df = pd.DataFrame(
        {
            "id": ["test_1", "test_2"],
            "event_time": ["2024-01-09", "2024-01-10"],
            "feature_1": [8.0, 9.0],
            "feature_2": [108.0, 109.0],
        }
    )
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    config_path = project_root / "configs" / "experiments" / "example_time_series_regression.yaml"
    common_args = ["--config", str(config_path), "--repo-root", str(project_root)]
    assert main(["doctor", *common_args]) == 0
    assert main(["build-splits", *common_args]) == 0
    assert main(["train", *common_args]) == 0
    assert main(["evaluate", *common_args]) == 0

    output_dir = project_root / "artifacts" / "experiments" / "exp_030_time_series_baseline"
    assert (output_dir / "summary.yaml").exists()


def test_cli_doctor_reports_missing_feature_columns(tmp_path: Path) -> None:
    project_root = tmp_path / "doctor_project"
    assert main(["init", str(project_root)]) == 0

    train_df = pd.DataFrame(
        {
            "id": ["train_1", "train_2", "train_3", "train_4"],
            "feature_1": [0.0, 1.0, 0.0, 1.0],
            "target": [0, 1, 0, 1],
        }
    )
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": ["test_1"], "feature_1": [0.5]}).to_csv(data_dir / "test.csv", index=False)

    config_path = project_root / "configs" / "experiments" / "example_classification.yaml"
    exit_code = main(["doctor", "--config", str(config_path), "--repo-root", str(project_root)])

    assert exit_code == 1
