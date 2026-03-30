from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from expctl.tracking import (
    LocalTracker,
    check_tracking_backend_dependencies,
    create_tracker,
)


def test_local_tracker_writes_tracking_payload(tmp_path: Path) -> None:
    tracker = LocalTracker(
        output_dir=tmp_path,
        experiment_name="expctl",
        run_name="exp_001",
    )

    tracker.log_params({"model": {"type": "ridge"}})
    tracker.log_metrics({"rmse_mean": 0.5})
    tracker.log_tags({"owner": "team"})
    tracker.log_config({"schema_version": 1})
    tracker.log_artifact(tmp_path / "artifact.txt")
    tracker.finish()

    payload = (tmp_path / "tracking.local.json").read_text(encoding="utf-8")
    assert '"backend": "local"' in payload
    assert '"model.type": "ridge"' in payload


def test_check_tracking_backend_dependencies_reports_missing_backend(monkeypatch) -> None:
    monkeypatch.setattr("expctl.tracking.util.find_spec", lambda name: None)
    assert check_tracking_backend_dependencies("mlflow") == [
        "Tracking backend 'mlflow' is not installed in the current environment"
    ]


def test_create_tracker_supports_mlflow_with_stub(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    class StubMlflow:
        def set_experiment(self, name: str) -> None:
            calls.append(("set_experiment", name))

        def start_run(self, run_name: str):
            calls.append(("start_run", run_name))
            return object()

        def log_params(self, params):
            calls.append(("log_params", params))

        def log_metrics(self, metrics):
            calls.append(("log_metrics", metrics))

        def set_tags(self, tags):
            calls.append(("set_tags", tags))

        def log_artifact(self, path: str):
            calls.append(("log_artifact", path))

        def end_run(self):
            calls.append(("end_run", None))

    monkeypatch.setattr("expctl.tracking.util.find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "mlflow", StubMlflow())

    tracker = create_tracker(
        backend="mlflow",
        output_dir=tmp_path,
        experiment_name="expctl",
        run_name="exp_001",
    )
    tracker.log_params({"alpha": 1.0})
    tracker.finish()

    assert ("set_experiment", "expctl") in calls
    assert ("start_run", "exp_001") in calls
    assert ("log_params", {"alpha": 1.0}) in calls
    assert ("end_run", None) in calls


def test_create_tracker_supports_wandb_with_stub(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    class StubRun:
        def __init__(self) -> None:
            self.config = SimpleNamespace(update=self._update)

        def _update(self, payload, allow_val_change: bool = False):
            calls.append(("config.update", payload))

        def log(self, metrics):
            calls.append(("log", metrics))

        def save(self, path: str):
            calls.append(("save", path))

        def finish(self):
            calls.append(("finish", None))

    class StubWandb(ModuleType):
        def init(self, **kwargs):
            calls.append(("init", kwargs["name"]))
            return StubRun()

    monkeypatch.setattr("expctl.tracking.util.find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "wandb", StubWandb("wandb"))

    tracker = create_tracker(
        backend="wandb",
        output_dir=tmp_path,
        experiment_name="expctl",
        run_name="exp_001",
    )
    tracker.log_metrics({"accuracy_mean": 0.9})
    tracker.finish()

    assert ("init", "exp_001") in calls
    assert ("log", {"accuracy_mean": 0.9}) in calls
    assert ("finish", None) in calls
