"""Vanilla-gradient saliency: the simplest correct pixel-attribution baseline.

This is the reference Explainer implementation, deliberately NOT
Grad-CAM or Integrated Gradients -- those are better explanations but
worse baselines to sanity-check the interface against, since their
correctness is harder to eyeball. New explainers (Captum-backed
Integrated Gradients, SHAP) should be added as siblings here and
registered under EXPLAINER_REGISTRY.
"""
from __future__ import annotations

from typing import Any

from geosentinel.core.registry import EXPLAINER_REGISTRY
from geosentinel.explainability.base import Explainer, ExplanationResult


@EXPLAINER_REGISTRY.register("vanilla_gradient")
class VanillaGradientSaliency(Explainer):
    """dL/dx attribution. Requires torch; raises a clear error otherwise."""

    name = "vanilla_gradient"

    def explain(self, model: Any, inputs: Any, target: int | None = None) -> ExplanationResult:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "VanillaGradientSaliency requires PyTorch. Install with "
                "`pip install geosentinel-ai[torch]`."
            ) from exc

        model.eval()
        x = inputs.clone().detach().requires_grad_(True)
        logits = model(x)
        target_idx = target if target is not None else int(logits.argmax(dim=-1).item())
        score = logits[..., target_idx].sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        if x.grad is None:
            raise RuntimeError("No gradient reached the input; check model is differentiable.")

        attribution = x.grad.detach().abs().cpu().numpy()
        return ExplanationResult(method=self.name, attribution=attribution, metadata={"target_class": target_idx})