"""Contract for uncertainty quantification methods."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class UncertaintyResult:
    method: str
    mean: np.ndarray
    std: np.ndarray
    n_samples: int


class UncertaintyEstimator(ABC):
    """Produces predictive mean and epistemic-uncertainty estimates.

    Must not assume any specific loss function or task type;
    benchmarking/reporting decide what to do with mean/std (e.g. flag
    predictions above a std threshold for expert review).
    """

    name: str

    @abstractmethod
    def estimate(self, model: Any, inputs: Any, n_samples: int) -> UncertaintyResult:
        raise NotImplementedError