"""Generic, type-safe component registry.

GeoSentinel-AI's extensibility model (models, explainers, uncertainty
estimators, benchmarks, datasets) is built on this one pattern: a
name-keyed registry with a registration decorator. One auditable
mechanism, not five ad-hoc ones -- that matters when the code still
has to make sense in ten years.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    """Raised on duplicate registration or lookup of an unknown name."""


class Registry([T]):
    """A name -> component registry for one kind of pluggable component.

    Example
    -------
    >>> MODEL_REGISTRY: Registry = Registry("model")
    >>> @MODEL_REGISTRY.register("resnet_change_detector")
    ... class ResNetChangeDetector: ...
    >>> cls = MODEL_REGISTRY.get("resnet_change_detector")
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def _decorator(item: T) -> T:
            if name in self._items:
                raise RegistryError(
                    f"A {self._kind} named '{name}' is already registered "
                    f"({self._items[name]!r}). Names must be unique."
                )
            self._items[name] = item
            return item

        return _decorator

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<none registered>"
            raise RegistryError(
                f"No {self._kind} named '{name}' is registered. Available: {available}"
            ) from exc

    def list_registered(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

