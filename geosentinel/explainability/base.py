"""Contract for explainability methods."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class ExplanationResult:
    method: str
    attribution: np.ndarray
    metadata: dict[str, Any]


class Explainer(ABC):
    """Produces per-input attributions for a trained model.

    Implementations must not mutate model parameters -- attribution
    should never change what the model would predict next.
    """

    name: str

    @abstractmethod
    def explain(self, model: Any, inputs: Any, target: int | None = None) -> ExplanationResult:
        raise NotImplementedError