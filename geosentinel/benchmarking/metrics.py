"""Task-appropriate benchmark metrics with markdown/LaTeX export.

Metrics come from scikit-learn rather than being hand-rolled: a
researcher citing this platform should be citing a metric
implementation with well-known reference behaviour, not a bespoke
accuracy function with undocumented edge-case handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from geosentinel.core.registry import BENCHMARK_REGISTRY

_METRIC_FUNCTIONS = {
    "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
    "precision_macro": lambda y_true, y_pred: precision_score(y_true, y_pred, average="macro", zero_division=0),
    "recall_macro": lambda y_true, y_pred: recall_score(y_true, y_pred, average="macro", zero_division=0),
    "f1_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro", zero_division=0),
}


@dataclass(slots=True)
class BenchmarkResult:
    metrics: dict[str, float]
    confusion: np.ndarray
    class_labels: list[str]

    def to_markdown_table(self) -> str:
        return pd.DataFrame([self.metrics]).to_markdown(index=False, floatfmt=".4f")

    def to_latex_table(self, caption: str = "Benchmark results") -> str:
        return pd.DataFrame([self.metrics]).to_latex(
            index=False, float_format="%.4f", caption=caption, label="tab:benchmark"
        )


@BENCHMARK_REGISTRY.register("classification")
class ClassificationBenchmark:
    """Standard classification benchmark: accuracy/precision/recall/F1 + confusion matrix."""

    def __init__(self, metric_names: list[str], class_labels: list[str] | None = None) -> None:
        unknown = set(metric_names) - set(_METRIC_FUNCTIONS)
        if unknown:
            raise ValueError(f"Unknown metric(s) {sorted(unknown)}. Available: {sorted(_METRIC_FUNCTIONS)}")
        self._metric_names = metric_names
        self._class_labels = class_labels

    def run(self, y_true: Any, y_pred: Any) -> BenchmarkResult:
        y_true_arr, y_pred_arr = np.asarray(y_true), np.asarray(y_pred)
        metrics = {name: float(_METRIC_FUNCTIONS[name](y_true_arr, y_pred_arr)) for name in self._metric_names}
        confusion = confusion_matrix(y_true_arr, y_pred_arr)
        labels = self._class_labels or [str(i) for i in sorted(set(y_true_arr.tolist()))]
        return BenchmarkResult(metrics=metrics, confusion=confusion, class_labels=labels)