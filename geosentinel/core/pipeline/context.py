"""Shared mutable state passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from geosentinel.core.configuration.schema import ExperimentConfig


@dataclass(slots=True)
class PipelineContext:
    """Carries config, intermediate artifacts, and metrics through a run.

    Stages communicate exclusively through `artifacts`/`metrics` rather
    than return values, because the pipeline is a sequence of stages of
    *varying and evolving number* (a segmentation pipeline has
    different stages than a change-detection one) -- a fixed function
    signature per stage would not scale to that.
    """

    experiment_id: str
    config: ExperimentConfig
    output_dir: Path
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    stage_log: list[str] = field(default_factory=list)

    def set_artifact(self, key: str, value: Any) -> None:
        self.artifacts[key] = value

    def get_artifact(self, key: str, expected_type: type[Any] | None = None) -> Any:
        if key not in self.artifacts:
            raise KeyError(
                f"Stage requires artifact '{key}', which no prior stage produced. "
                f"Produced so far: {sorted(self.artifacts)}"
            )
        value = self.artifacts[key]
        if expected_type is not None and not isinstance(value, expected_type):
            raise TypeError(
                f"Artifact '{key}' expected to be {expected_type.__name__}, got {type(value).__name__}"
            )
        return value

    def bound_logger(self) -> structlog.stdlib.BoundLogger:
        return structlog.get_logger("geosentinel.pipeline").bind(experiment_id=self.experiment_id)