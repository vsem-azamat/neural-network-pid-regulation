"""Reproducibility helpers.

Every entry point seeds once at start-up so a rerun of the same command
produces the same figures and the same metric tables.
"""

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    """Seed Python, NumPy and Torch RNGs. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return seed
