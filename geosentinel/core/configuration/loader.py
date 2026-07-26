"""Configuration loading: YAML -> validated ExperimentConfig.

Supports:
- `extends: <path>` for base/override inheritance, so a researcher can
  keep one base_experiment.yaml and vary a single hyperparameter per
  run without duplicating the whole file.
- `${ENV_VAR}` / `${ENV_VAR:-default}` interpolation, so configs
  committed to git never contain machine-specific paths or secrets.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from geosentinel.core.configuration.schema import ExperimentConfig

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(:-[^}]*)?\}")


class ConfigurationError(Exception):
    """Raised when a YAML config is malformed or fails schema validation."""


def _interpolate_env_vars(raw: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2)
        if var_name in os.environ:
            return os.environ[var_name]
        if default is not None:
            return default[2:]  # strip leading ":-"
        raise ConfigurationError(
            f"Environment variable '{var_name}' referenced in config but not set, "
            f"and no default was provided (use ${{VAR:-default}})."
        )

    return _ENV_VAR_PATTERN.sub(_replace, raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Config file not found: {path}")

    text = _interpolate_env_vars(path.read_text(encoding="utf-8"))
    try:
        data: dict[str, Any] = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc

    extends = data.pop("extends", None)
    if extends is not None:
        base_data = _load_raw_yaml((path.parent / extends).resolve())
        data = _deep_merge(base_data, data)

    return data


class ConfigLoader:
    """Single entry point every stage, CLI command, and test should use
    to obtain an ExperimentConfig; nothing else should call
    `yaml.safe_load` directly.
    """

    @staticmethod
    def load(path: str | Path) -> ExperimentConfig:
        resolved = Path(path).resolve()
        raw = _load_raw_yaml(resolved)
        try:
            return ExperimentConfig.model_validate(raw)
        except ValidationError as exc:
            raise ConfigurationError(f"Configuration '{resolved}' failed validation:\n{exc}") from exc

    @staticmethod
    def load_dict(data: dict[str, Any]) -> ExperimentConfig:
        """Load from an in-memory dict (used heavily in tests)."""
        try:
            return ExperimentConfig.model_validate(data)
        except ValidationError as exc:
            raise ConfigurationError(f"Configuration dict failed validation:\n{exc}") from exc