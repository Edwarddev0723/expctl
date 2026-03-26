from __future__ import annotations

import pytest

from expctl.training.model_registry import ModelRegistry, UnknownModelTypeError


def test_model_registry_resolves_registered_builder() -> None:
    builder = object()
    registry = ModelRegistry({"logistic_regression": lambda **_: builder})
    assert registry.resolve("logistic_regression")() is builder


def test_model_registry_raises_for_unknown_model_type() -> None:
    registry = ModelRegistry()
    with pytest.raises(UnknownModelTypeError):
        registry.resolve("missing_model")
