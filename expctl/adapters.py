from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pandas as pd

from expctl.config import FeatureCatalog
from expctl.training import ModelRegistry


class AdapterLoadError(RuntimeError):
    """Raised when a project adapter is missing or structurally invalid."""


@dataclass(frozen=True)
class ProjectAdapter:
    feature_catalog: Callable[[], FeatureCatalog]
    model_registry: Callable[[], ModelRegistry]
    load_dataset: Callable[[dict[str, Any]], pd.DataFrame]


def load_project_adapter(repo_root: Path) -> ProjectAdapter:
    module = _load_adapter_module(repo_root)

    feature_catalog = _require_callable(module, "feature_catalog")
    model_registry = _require_callable(module, "model_registry")
    load_dataset = _require_callable(module, "load_dataset")

    catalog = feature_catalog()
    if not isinstance(catalog, FeatureCatalog):
        raise AdapterLoadError("project.feature_catalog() must return expctl.config.FeatureCatalog")

    registry = model_registry()
    if not isinstance(registry, ModelRegistry):
        raise AdapterLoadError("project.model_registry() must return expctl.training.ModelRegistry")

    return ProjectAdapter(
        feature_catalog=feature_catalog,
        model_registry=model_registry,
        load_dataset=load_dataset,
    )


def adapter_exists(repo_root: Path) -> bool:
    return (repo_root / "src" / "project" / "__init__.py").exists()


def _load_adapter_module(repo_root: Path) -> ModuleType:
    module_path = repo_root / "src" / "project" / "__init__.py"
    if not module_path.exists():
        raise AdapterLoadError(
            f"Project adapter not found: {module_path}. "
            "Run `expctl init` or create src/project/__init__.py."
        )

    module_name = f"_expctl_project_adapter_{abs(hash(module_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AdapterLoadError(f"Failed to load adapter module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _require_callable(module: ModuleType, name: str) -> Callable[..., Any]:
    value = getattr(module, name, None)
    if value is None or not callable(value):
        raise AdapterLoadError(f"project adapter must define callable `{name}()`")
    return value
