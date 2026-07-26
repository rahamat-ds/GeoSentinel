"""Automatic capture of everything the Research Standards require.

If provenance capture required the researcher to remember to log
something, it would eventually be forgotten under deadline pressure.
Everything here is captured by the pipeline itself, not opted into.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from geosentinel.core.configuration.schema import ExperimentConfig
from geosentinel.core.pipeline.context import PipelineContext


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _git_is_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5, check=True
        )
        return bool(result.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def capture_software_environment() -> dict[str, Any]:
    """Python version, OS, git commit, and installed package versions."""
    packages = {
        dist.metadata["Name"]: dist.version for dist in distributions() if dist.metadata.get("Name")
    }
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "git_dirty": _git_is_dirty(),
        "packages": packages,
    }


def capture_hardware_environment() -> dict[str, Any]:
    """CPU/GPU description. GPU details are best-effort via torch."""
    info: dict[str, Any] = {
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        else:
            info["gpu"] = "unavailable"
    except ImportError:
        info["gpu"] = "torch_not_installed"
    return info


@dataclass(slots=True)
class ExperimentRecord:
    """The full, serializable provenance record for one experiment run."""

    experiment_id: str
    experiment_name: str
    status: str
    config: dict[str, Any]
    seed: int
    timestamp_start_utc: float
    timestamp_end_utc: float | None
    runtime_seconds: float | None
    software_environment: dict[str, Any]
    hardware_environment: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    stage_log: list[str] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")


class ProvenanceRecorder:
    """Builds an ExperimentRecord around a pipeline run."""

    def __init__(self, experiment_id: str, config: ExperimentConfig) -> None:
        self._experiment_id = experiment_id
        self._config = config
        self._start_time: float | None = None

    def start(self) -> None:
        self._start_time = time.time()

    def finalize(self, context: PipelineContext, status: str) -> ExperimentRecord:
        if self._start_time is None:
            raise RuntimeError("ProvenanceRecorder.start() must be called before finalize()")
        end_time = time.time()
        return ExperimentRecord(
            experiment_id=self._experiment_id,
            experiment_name=self._config.experiment_name,
            status=status,
            config=self._config.model_dump(mode="json"),
            seed=self._config.seed,
            timestamp_start_utc=self._start_time,
            timestamp_end_utc=end_time,
            runtime_seconds=round(end_time - self._start_time, 4),
            software_environment=capture_software_environment(),
            hardware_environment=capture_hardware_environment(),
            metrics=context.metrics,
            artifacts=[str(v) for v in context.artifacts.values() if isinstance(v, Path)],
            stage_log=context.stage_log,
        )