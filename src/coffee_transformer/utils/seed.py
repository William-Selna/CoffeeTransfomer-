"""Seeding. The design runs 5 seeds and reports variance (Section 7/9), so
seeding must cover python, numpy, and torch, and hand back a torch.Generator
for the stochastic bits (randomized r, SMILES/slot augmentation, pair sampling).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> torch.Generator:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
