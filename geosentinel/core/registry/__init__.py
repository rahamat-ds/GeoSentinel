"""Global component registries.

Importing from here (rather than instantiating ad-hoc registries)
guarantees every plugin registers into the same namespace the core
pipeline resolves against.
"""
from geosentinel.core.registry.registry import Registry, RegistryError

MODEL_REGISTRY: Registry = Registry("model")
EXPLAINER_REGISTRY: Registry = Registry("explainer")
UNCERTAINTY_REGISTRY: Registry = Registry("uncertainty_estimator")
DATASET_REGISTRY: Registry = Registry("dataset")
BENCHMARK_REGISTRY: Registry = Registry("benchmark")

__all__ = [
    "Registry",
    "RegistryError",
    "MODEL_REGISTRY",
    "EXPLAINER_REGISTRY",
    "UNCERTAINTY_REGISTRY",
    "DATASET_REGISTRY",
    "BENCHMARK_REGISTRY",
]