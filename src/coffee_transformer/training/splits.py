"""Split the labeled HTE pool into supervised / RL-environment / test sets.

Mirrors Section 7's ~60/20/20 structure but parameterized so the four target
runs are pure config: the SFT/RL ratio ({100/0, 90/10, 75/25, 50/50}) is the
experimental axis, at a fixed total labeled budget, holding the test split
fixed so every run is evaluated identically.

`data_fraction` additionally subsamples the SFT pool for the main-grid
data-efficiency axis ({0.5, 0.75, 1.0}).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from ..data.dataset import ReactionExample


@dataclass
class Pools:
    sft: list[ReactionExample]
    rl: list[ReactionExample]
    test: list[ReactionExample]


def split_pools(
    examples: Sequence[ReactionExample],
    sft_fraction: float,
    rl_fraction: float,
    test_fraction: float,
    data_fraction: float = 1.0,
    seed: int = 0,
) -> Pools:
    if abs((sft_fraction + rl_fraction) - 1.0) > 1e-6:
        raise ValueError(
            f"sft_fraction + rl_fraction must equal 1 (got {sft_fraction} + {rl_fraction})"
        )
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)

    n = len(idx)
    n_test = round(n * test_fraction)
    test_idx = idx[:n_test]
    rest = idx[n_test:]

    n_sft = round(len(rest) * sft_fraction)
    sft_idx = rest[:n_sft]
    rl_idx = rest[n_sft:]

    # data-efficiency subsampling of the supervised pool only
    n_use = round(len(sft_idx) * data_fraction)
    sft_idx = sft_idx[:n_use]

    pick = lambda ids: [examples[i] for i in ids]
    return Pools(sft=pick(sft_idx), rl=pick(rl_idx), test=pick(test_idx))
