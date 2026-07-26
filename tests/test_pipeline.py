"""Integration test: the orchestration engine with synthetic stages.

Deliberately torch-free, so this test (and the pattern it documents)
runs in any environment, including CI without a GPU.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from geosentinel.core.configuration.loader import ConfigLoader
from geosentinel.core.pipeline.context import PipelineContext
from geosentinel.core.pipeline.pipeline import Pipeline
from geosentinel.core.pipeline.stage import PipelineStage


class _FakeIngestionStage(PipelineStage):
    name = "ingestion"

    def run(self, context: PipelineContext) -> PipelineContext:
        context.set_artifact("y_true", [0, 1, 1, 0, 1])
        return context


class _FakeModelStage(PipelineStage):
    name = "model_inference"

    def preconditions(self, context: PipelineContext) -> list[str]:
        return ["y_true"]

    def run(self, context: PipelineContext) -> PipelineContext:
        y_true = context.get_artifact("y_true", list)
        context.set_artifact("y_pred", y_true)  # perfect predictions for the test
        return context


class _FakeBenchmarkStage(PipelineStage):
    name = "benchmarking"

    def preconditions(self, context: PipelineContext) -> list[str]:
        return ["y_true", "y_pred"]

    def run(self, context: PipelineContext) -> PipelineContext:
        y_true = context.get_artifact("y_true", list)
        y_pred = context.get_artifact("y_pred", list)
        accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
        context.metrics["accuracy"] = accuracy
        return context


@pytest.fixture
def minimal_config(tmp_path: Path):
    return ConfigLoader.load_dict(
        {
            "experiment_name": "unit_test_experiment",
            "seed": 7,
            "dataset": {"name": "synthetic", "version": "0.0.1", "root": str(tmp_path)},
            "model": {"name": "identity", "architecture": "identity", "task_type": "classification", "num_classes": 2},
            "reporting": {"output_dir": str(tmp_path / "experiments")},
        }
    )


def test_pipeline_runs_stages_in_order_and_captures_provenance(minimal_config):
    pipeline = Pipeline(
        stages=[_FakeIngestionStage(), _FakeModelStage(), _FakeBenchmarkStage()], config=minimal_config
    )
    context, record = pipeline.run()

    assert context.stage_log == ["ingestion", "model_inference", "benchmarking"]
    assert context.metrics["accuracy"] == 1.0
    assert record.status == "completed"
    assert record.seed == 7
    assert (context.output_dir / "experiment_record.json").exists()


def test_pipeline_fails_fast_on_missing_precondition(minimal_config):
    pipeline = Pipeline(stages=[_FakeBenchmarkStage()], config=minimal_config)  # skips ingestion
    with pytest.raises(Exception, match="missing required artifacts"):
        pipeline.run()