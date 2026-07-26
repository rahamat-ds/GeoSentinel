"""Abstract base for a single pipeline stage."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from geosentinel.core.pipeline.context import PipelineContext


class PipelineStage(ABC):
    """One step of the ingestion -> ... -> decision-support flow.

    Stages are small and single-purpose (SRP): a stage should be
    describable in one sentence ("runs MC-Dropout inference and writes
    mean/variance maps"). Composing many small stages, over one large
    run_experiment() function, is what lets e.g. the benchmarking stage
    be reused across classification and segmentation pipelines
    unchanged.
    """

    name: ClassVar[str]

    def preconditions(self, context: PipelineContext) -> list[str]:
        """Required artifact keys this stage depends on.

        Checked before the stage runs, so a misconfigured stage order
        fails fast with a clear message instead of a KeyError three
        layers down inside the stage.
        """
        return []

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError


class PipelineStageError(RuntimeError):
    """Wraps an exception raised inside a stage with stage context."""

    def __init__(self, stage_name: str, original: Exception) -> None:
        self.stage_name = stage_name
        self.original = original
        super().__init__(f"Stage '{stage_name}' failed: {original}")