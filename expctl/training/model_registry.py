from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

ModelBuilder = Callable[..., Any]


class UnknownModelTypeError(ValueError):
    """Raised when a config asks for a model type that is not registered."""


class ModelRegistry:
    def __init__(self, builders: Mapping[str, ModelBuilder] | None = None) -> None:
        self._builders: dict[str, ModelBuilder] = dict(builders or {})

    def register(
        self,
        model_type: str,
        builder: ModelBuilder,
        *,
        overwrite: bool = False,
    ) -> None:
        if model_type in self._builders and not overwrite:
            raise ValueError(f"Model type already registered: {model_type}")
        self._builders[model_type] = builder

    def resolve(self, model_type: str) -> ModelBuilder:
        if model_type not in self._builders:
            raise UnknownModelTypeError(
                f"Unsupported model type: {model_type}. Supported: {self.supported_types}"
            )
        return self._builders[model_type]

    @property
    def supported_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))

    def as_dict(self) -> dict[str, ModelBuilder]:
        return dict(self._builders)
