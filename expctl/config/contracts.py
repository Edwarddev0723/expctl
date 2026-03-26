from __future__ import annotations

from dataclasses import dataclass, field

from expctl.evaluation.metrics import SUPPORTED_METRICS


@dataclass(frozen=True)
class FeatureCatalog:
    """Project-provided feature metadata consumed by generic config validators."""

    derived_numeric_features: frozenset[str] = field(default_factory=frozenset)
    derived_categorical_features: frozenset[str] = field(default_factory=frozenset)
    text_vectorizers: frozenset[str] = field(
        default_factory=lambda: frozenset({"tfidf_svd", "sentence_transformer"})
    )
    tfidf_analyzers: frozenset[str] = field(
        default_factory=lambda: frozenset({"word", "char", "char_wb"})
    )
    metrics: frozenset[str] = field(default_factory=lambda: frozenset(SUPPORTED_METRICS))
