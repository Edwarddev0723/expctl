from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from expctl.config import FeatureCatalog
from expctl.training import ModelRegistry


def feature_catalog() -> FeatureCatalog:
    return FeatureCatalog()


def model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("logistic_regression", lambda **params: LogisticRegression(**params))
    registry.register("ridge_regression", lambda **params: Ridge(**params))
    return registry


def load_dataset(config: dict[str, object]) -> pd.DataFrame:
    return pd.read_csv(Path(str(config["data"]["train_path"])))
