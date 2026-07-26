"""Monte Carlo Dropout uncertainty estimation (Gal & Ghahramani, 2016).

Chosen as the reference implementation because it needs no
architectural change to an existing model (unlike deep ensembles,
which need N trained models) -- lowest-friction path for a researcher
to add uncertainty estimates to a model they already have.
"""
from __future__ import annotations

from typing import Any

from geosentinel.core.registry import UNCERTAINTY_REGISTRY
from geosentinel.uncertainty.base import UncertaintyEstimator, UncertaintyResult


def _enable_dropout_only(model: Any) -> None:
    """eval() everything except Dropout layers, which stay stochastic.

    This is the crux of MC-Dropout: we want stochastic dropout masks at
    inference time, but batchnorm and everything else must stay in
    eval mode, or single-sample statistics would corrupt running
    batch-norm estimates.
    """
    import torch.nn as nn

    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


@UNCERTAINTY_REGISTRY.register("mc_dropout")
class MCDropoutEstimator(UncertaintyEstimator):
    name = "mc_dropout"

    def estimate(self, model: Any, inputs: Any, n_samples: int = 30) -> UncertaintyResult:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "MCDropoutEstimator requires PyTorch. Install with `pip install geosentinel-ai[torch]`."
            ) from exc

        import numpy as np

        _enable_dropout_only(model)
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                predictions.append(model(inputs).cpu().numpy())

        stacked = np.stack(predictions, axis=0)
        return UncertaintyResult(
            method=self.name, mean=stacked.mean(axis=0), std=stacked.std(axis=0), n_samples=n_samples
        )