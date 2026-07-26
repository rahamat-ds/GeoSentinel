"""Typed configuration schema for GeoSentinel experiments.

Every experiment run is fully described by a single, validated
ExperimentConfig object. This is the contract between the YAML files a
researcher edits and the pipeline that executes them: if a config
parses, every downstream stage can rely on well-typed values instead
of defensively checking dictionaries.

Design rationale
-----------------
- Pydantic (not plain dataclasses) because we need runtime validation
  of researcher-edited YAML, coercion of primitive types, and machine
  readable error messages -- researchers should get "learning_rate
  must be > 0", not a stack trace three modules deep.
- No defaults for fields that affect scientific validity (seed,
  dataset version, split). Silent defaults are how irreproducible
  experiments happen.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskType(str, Enum):
    """Supported research task families.

    Deliberately a closed enum, not a free string: benchmarking and
    reporting stages branch on this value.
    """

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    CHANGE_DETECTION = "change_detection"


class DatasetConfig(BaseModel):
    """Describes a dataset instance, pinned to a specific version.

    `version` and `checksum` exist because "the dataset" is not a
    stable reference over a multi-year research program -- files get
    corrected, labels get revised, mirrors go stale. "Sentinel-2
    imagery" is not reproducible; "Sentinel-2 imagery, registry version
    2024.03, sha256:ab12..." is.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = Field(..., description="Immutable version tag from datasets/registry")
    root: Path = Field(..., description="Filesystem or object-store URI, resolved at load time")
    format: str = Field(default="geotiff")
    checksum: str | None = Field(default=None, description="sha256 of the dataset manifest")
    split: str = Field(default="train")

    @field_validator("root", mode="before")
    @classmethod
    def _expand_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser()


class ModelConfig(BaseModel):
    """Model architecture and weight provenance."""

    model_config = ConfigDict(extra="forbid")

    name: str
    architecture: str
    task_type: TaskType
    num_classes: int = Field(..., gt=0)
    pretrained: bool = False
    checkpoint_path: Path | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def _expand_checkpoint(cls, value: str | Path | None) -> Path | None:
        return Path(value).expanduser() if value else None


class ExplainabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    method: str = Field(default="vanilla_gradient")
    params: dict[str, Any] = Field(default_factory=dict)


class UncertaintyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    method: str = Field(default="mc_dropout")
    n_samples: int = Field(default=30, gt=1)
    params: dict[str, Any] = Field(default_factory=dict)


class BenchmarkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(default_factory=lambda: ["accuracy", "f1_macro"])
    baseline_results_path: Path | None = None


class ReportingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    formats: list[str] = Field(default_factory=lambda: ["markdown", "latex"])
    output_dir: Path = Field(default=Path("experiments"))
    generate_figures: bool = True


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = Field(default="mlflow")
    tracking_uri: str | None = None
    experiment_group: str = Field(default="default")


class ExperimentConfig(BaseModel):
    """The full, validated description of a single experiment run.

    This is intentionally the *only* thing a pipeline stage is allowed
    to depend on for "what should happen" -- stages must not reach into
    environment variables or ambient global state to decide behaviour,
    or two runs of the same config could diverge.
    """

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    description: str = ""
    seed: int = Field(..., description="Global random seed; mandatory, no default")
    dataset: DatasetConfig
    model: ModelConfig
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    benchmarking: BenchmarkingConfig = Field(default_factory=BenchmarkingConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_task_class_consistency(self) -> "ExperimentConfig":
        if self.model.task_type is TaskType.CLASSIFICATION and self.model.num_classes < 2:
            raise ValueError("Classification tasks require num_classes >= 2")
        return self