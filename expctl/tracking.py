from __future__ import annotations

import json
from importlib import util
from pathlib import Path
from typing import Any

SUPPORTED_TRACKING_BACKENDS = frozenset({"local", "mlflow", "wandb"})
BACKEND_MODULES = {
    "local": None,
    "mlflow": "mlflow",
    "wandb": "wandb",
}


class TrackingError(RuntimeError):
    """Raised when a tracker cannot be initialized or used."""


class BaseTracker:
    def log_params(self, params: dict[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: dict[str, float]) -> None:
        return None

    def log_tags(self, tags: dict[str, Any]) -> None:
        return None

    def log_config(self, config: dict[str, Any], *, config_path: Path | None = None) -> None:
        return None

    def log_artifact(self, path: Path) -> None:
        return None

    def finish(self) -> None:
        return None


class LocalTracker(BaseTracker):
    def __init__(
        self,
        *,
        output_dir: Path,
        experiment_name: str,
        run_name: str,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._payload: dict[str, Any] = {
            "backend": "local",
            "experiment_name": experiment_name,
            "run_name": run_name,
            "params": {},
            "metrics": {},
            "tags": {},
            "config_path": None,
            "artifacts": [],
        }

    def log_params(self, params: dict[str, Any]) -> None:
        self._payload["params"].update(flatten_mapping(params))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._payload["metrics"].update(metrics)

    def log_tags(self, tags: dict[str, Any]) -> None:
        self._payload["tags"].update(flatten_mapping(tags))

    def log_config(self, config: dict[str, Any], *, config_path: Path | None = None) -> None:
        self._payload["config_path"] = str(config_path) if config_path is not None else None
        self._payload["config"] = config

    def log_artifact(self, path: Path) -> None:
        self._payload["artifacts"].append(str(path))

    def finish(self) -> None:
        tracking_path = self.output_dir / "tracking.local.json"
        with tracking_path.open("w", encoding="utf-8") as fh:
            json.dump(self._payload, fh, indent=2, sort_keys=True)


class MlflowTracker(BaseTracker):
    def __init__(
        self,
        *,
        output_dir: Path,
        experiment_name: str,
        run_name: str,
    ) -> None:
        mlflow = _import_dependency("mlflow")
        self._mlflow = mlflow
        self._mlflow.set_experiment(experiment_name)
        self._run = self._mlflow.start_run(run_name=run_name)
        self._output_dir = output_dir

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(flatten_mapping(params))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._mlflow.log_metrics(metrics)

    def log_tags(self, tags: dict[str, Any]) -> None:
        self._mlflow.set_tags(flatten_mapping(tags))

    def log_config(self, config: dict[str, Any], *, config_path: Path | None = None) -> None:
        if config_path is not None and config_path.exists():
            self._mlflow.log_artifact(str(config_path))

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))

    def finish(self) -> None:
        self._mlflow.end_run()


class WandbTracker(BaseTracker):
    def __init__(
        self,
        *,
        output_dir: Path,
        experiment_name: str,
        run_name: str,
    ) -> None:
        wandb = _import_dependency("wandb")
        self._wandb = wandb
        self._run = self._wandb.init(
            project=experiment_name,
            name=run_name,
            dir=str(output_dir),
            reinit=True,
        )

    def log_params(self, params: dict[str, Any]) -> None:
        self._run.config.update(flatten_mapping(params), allow_val_change=True)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._run.log(metrics)

    def log_tags(self, tags: dict[str, Any]) -> None:
        flattened = flatten_mapping(tags)
        self._run.config.update({f"tag.{key}": value for key, value in flattened.items()})

    def log_config(self, config: dict[str, Any], *, config_path: Path | None = None) -> None:
        self._run.config.update({"config": config}, allow_val_change=True)
        if config_path is not None and config_path.exists():
            self._run.save(str(config_path))

    def log_artifact(self, path: Path) -> None:
        self._run.save(str(path))

    def finish(self) -> None:
        self._run.finish()


def create_tracker(
    *,
    backend: str,
    output_dir: Path,
    experiment_name: str,
    run_name: str,
) -> BaseTracker:
    if backend == "local":
        return LocalTracker(
            output_dir=output_dir,
            experiment_name=experiment_name,
            run_name=run_name,
        )
    if backend == "mlflow":
        return MlflowTracker(
            output_dir=output_dir,
            experiment_name=experiment_name,
            run_name=run_name,
        )
    if backend == "wandb":
        return WandbTracker(
            output_dir=output_dir,
            experiment_name=experiment_name,
            run_name=run_name,
        )
    raise TrackingError(
        f"Unsupported tracking backend: {backend}. Supported={sorted(SUPPORTED_TRACKING_BACKENDS)}"
    )


def check_tracking_backend_dependencies(backend: str) -> list[str]:
    if backend == "local":
        return []
    if backend not in SUPPORTED_TRACKING_BACKENDS:
        return [f"Unsupported tracking backend: {backend}"]
    module_name = BACKEND_MODULES[backend]
    if module_name is None:
        return []
    if util.find_spec(module_name) is None:
        return [f"Tracking backend '{backend}' is not installed in the current environment"]
    return []


def flatten_mapping(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        key_str = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_mapping(value, prefix=key_str))
        else:
            flattened[key_str] = value
    return flattened


def _import_dependency(name: str) -> Any:
    module_spec = util.find_spec(name)
    if module_spec is None:
        raise TrackingError(f"Tracking backend dependency '{name}' is not installed")
    module = __import__(name)
    return module
