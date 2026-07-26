"""Deterministic-experiment support.

Full bitwise reproducibility across GPU hardware/driver versions is
not achievable for deep learning -- what this module guarantees is
that every RNG source under our control is seeded and recorded. State
that limitation explicitly in any published methods section.
"""
from __future__ import annotations

import random

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        logger.debug("torch_not_installed_seed_skipped")

    logger.info("global_seed_set", seed=seed)