"""Plugin contract for domain extensions (earth_observation, marine_monitoring, ...).

A GeoSentinel plugin is a Python distribution that exposes a
`geosentinel.plugins` entry point pointing at a callable. Core never
imports domain code directly -- this is what lets marine_monitoring
and climate live as independently versioned, independently citable
packages rather than being welded into core.
"""
from __future__ import annotations

from typing import Protocol


class GeoSentinelPlugin(Protocol):
    """Structural contract every plugin's entry-point callable satisfies."""

    def __call__(self) -> None:
        """Register the plugin's models/explainers/estimators/benchmarks.

        Implementations should only call `.register(...)` on the
        global registries (geosentinel.core.registry). Side effects
        beyond registration (network calls, file I/O) are strongly
        discouraged at registration time.
        """
        ...