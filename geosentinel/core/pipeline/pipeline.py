"""The orchestration engine: Satellite Image -> ... -> Decision Support.

This module is deliberately the only place that knows about *timing*,
*error containment*, and *provenance finalization* -- cross-cutting
concerns that should not leak into individual stage implementations.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Sequence

from geosentinel.core.configuration.schema import ExperimentConfig
from geosentinel.core.pipeline.context import PipelineContext
from geosentinel.core.pipeline.stage import PipelineStage, PipelineStageError
from geosentinel.core.provenance.capture import ExperimentRecord, ProvenanceRecorder
from geosentinel.core.provenance.seed import set_global_seed


class Pipeline:
    """Executes an ordered sequence of stages under one experiment record."""

    def __init__(self, stages: Sequence[PipelineStage], config: ExperimentConfig) -> None:
        if not stages:
            raise ValueError("Pipeline requires at least one stage")
        self._stages = list(stages)
        self._config = config

    def run(self) -> tuple[PipelineContext, ExperimentRecord]:
        experiment_id = f"{self._config.experiment_name}-{uuid.uuid4().hex[:8]}"
        output_dir = self._config.reporting.output_dir / experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)

        set_global_seed(self._config.seed)

        recorder = ProvenanceRecorder(experiment_id=experiment_id, config=self._config)
        recorder.start()

        context = PipelineContext(experiment_id=experiment_id, config=self._config, output_dir=output_dir)
        logger = context.bound_logger()
        logger.info("pipeline_started", stages=[s.name for s in self._stages])

        for stage in self._stages:
            missing = [k for k in stage.preconditions(context) if k not in context.artifacts]
            if missing:
                raise PipelineStageError(
                    stage.name, RuntimeError(f"missing required artifacts from prior stages: {missing}")
                )

            stage_start = time.perf_counter()
            logger.info("stage_started", stage=stage.name)
            try:
                context = stage.run(context)
            except Exception as exc:  # noqa: BLE001 - re-raised with stage context below
                logger.error("stage_failed", stage=stage.name, error=str(exc))
                recorder.finalize(context, status="failed")
                raise PipelineStageError(stage.name, exc) from exc
            duration = time.perf_counter() - stage_start
            context.stage_log.append(stage.name)
            logger.info("stage_completed", stage=stage.name, duration_seconds=round(duration, 4))

        record = recorder.finalize(context, status="completed")
        record.to_json(output_dir / "experiment_record.json")
        logger.info("pipeline_completed", output_dir=str(output_dir))
        return context, record