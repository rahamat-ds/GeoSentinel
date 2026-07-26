"""Plugin discovery via Python entry points.

Uses importlib.metadata rather than a bespoke plugin-folder scanner so
plugins are ordinary, independently installable/versionable pip
packages -- consistent with "extensibility before convenience".
"""
from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points

import structlog

_ENTRY_POINT_GROUP = "geosentinel.plugins"
logger = structlog.get_logger(__name__)


class PluginManager:
    """Discovers and activates installed GeoSentinel plugins."""

    def __init__(self) -> None:
        self._activated: list[str] = []

    def discover(self) -> list[EntryPoint]:
        return list(entry_points(group=_ENTRY_POINT_GROUP))

    def activate_all(self) -> list[str]:
        """Import and call every discovered plugin's registration function.

        A failure in one plugin is logged and skipped rather than
        aborting the whole process -- a broken optional plugin should
        not prevent a researcher from running an unrelated experiment.
        """
        for ep in self.discover():
            try:
                register_fn = ep.load()
                register_fn()
                self._activated.append(ep.name)
                logger.info("plugin_activated", plugin=ep.name)
            except Exception as exc:  # noqa: BLE001 - intentionally broad, see docstring
                logger.error("plugin_activation_failed", plugin=ep.name, error=str(exc))
        return self._activated

    @property
    def activated(self) -> list[str]:
        return list(self._activated)