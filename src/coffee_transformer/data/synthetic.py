"""Synthetic HTE-shaped data for smoke-testing the full pipeline offline.

This is NOT chemistry — it exists so the tokenizer, model, losses, trainers,
and eval harness can be exercised end-to-end on CPU before renting a GPU
(Section 8: "Debug everything on CPU/local GPU first"). Replace with
`load_buchwald_hartwig` for real runs.

The generator builds a small factorial cross of toy SMILES per slot and a
deterministic, non-linear yield function with additive "poisoning" effects so
that ranking/regression metrics are meaningful during smoke tests.
"""

from __future__ import annotations

import math
import random

from .dataset import ReactionExample
from .slots import BUCHWALD_HARTWIG_SLOTS

# Small pools of syntactically valid toy SMILES fragments per slot.
_POOLS: dict[str, list[str]] = {
    "ARYL_HALIDE": ["Clc1ccccc1", "Brc1ccccc1", "Ic1ccccc1", "Clc1ccncc1", "Brc1ccc(C)cc1"],
    "LIG": ["CC(C)c1ccccc1P", "c1ccc(P(c2ccccc2)c2ccccc2)cc1", "CP(C)C", "FC(F)(F)P"],
    "BASE": ["[Na+].[OH-]", "CC(C)(C)[O-]", "O=C([O-])[O-]"],
    "ADD": ["c1ccno1", "Cc1ccno1", "O=Cc1ccno1", "c1ccc(-c2ccno2)cc1", "N#Cc1ccno1"],
}


def _hash_unit(s: str, salt: int) -> float:
    """Deterministic pseudo-random value in [0, 1) from a string."""
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch) + salt) & 0xFFFFFFFF
    return (h % 100003) / 100003.0


def _toy_yield(components: list[tuple[str, str]]) -> float:
    """Non-linear yield with a dominant main effect and additive poisoning."""
    base = 0.0
    poison = 1.0
    for slot, smi in components:
        v = _hash_unit(smi, salt=len(slot))
        if slot == "ADD":
            # additives simulate catalyst poisoning: multiplicative penalty
            poison *= 0.4 + 0.6 * v
        else:
            base += v
    base /= max(1, len(components) - 1)
    y = 100.0 * poison * (0.5 + 0.5 * math.sin(3.0 * base))
    return float(max(0.0, min(100.0, y)))


def make_synthetic_bh(
    n: int = 2000,
    seed: int = 0,
    slots: tuple[str, ...] = BUCHWALD_HARTWIG_SLOTS,
) -> list[ReactionExample]:
    rng = random.Random(seed)
    examples: list[ReactionExample] = []
    for _ in range(n):
        components = [(slot, rng.choice(_POOLS[slot])) for slot in slots if slot in _POOLS]
        examples.append(
            ReactionExample(
                components=components,
                yield_value=_toy_yield(components),
                dataset="BH",
            )
        )
    return examples


def all_synthetic_smiles(slots: tuple[str, ...] = BUCHWALD_HARTWIG_SLOTS) -> list[str]:
    """Every toy SMILES — used to build the tokenizer vocab for smoke tests."""
    out: list[str] = []
    for slot in slots:
        out.extend(_POOLS.get(slot, []))
    return out
