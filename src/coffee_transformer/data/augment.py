"""Training-time augmentations (Section 4).

Both measurably helped Yield-BERT:
  * slot-order randomization — shuffle component order within a reaction;
  * SMILES randomization — re-render each molecule from a random atom.

SMILES randomization needs RDKit. If RDKit is not installed the function is a
no-op passthrough (so the synthetic/debug path still runs) and warns once.
"""

from __future__ import annotations

import random
import warnings
from typing import Sequence

try:  # optional dependency
    from rdkit import Chem  # type: ignore

    _HAS_RDKIT = True
except Exception:  # pragma: no cover - exercised only without rdkit
    _HAS_RDKIT = False

_WARNED = False


def randomize_smiles(smiles: str, rng: random.Random | None = None) -> str:
    """Return a randomized (non-canonical) SMILES for the same molecule."""
    global _WARNED
    if not _HAS_RDKIT:
        if not _WARNED:
            warnings.warn("RDKit not installed; SMILES randomization is a no-op.", stacklevel=2)
            _WARNED = True
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    n = mol.GetNumAtoms()
    order = list(range(n))
    (rng or random).shuffle(order)
    renumbered = Chem.RenumberAtoms(mol, order)
    return Chem.MolToSmiles(renumbered, canonical=False)


def randomize_slot_order(
    components: Sequence[tuple[str, str]],
    rng: random.Random | None = None,
) -> list[tuple[str, str]]:
    out = list(components)
    (rng or random).shuffle(out)
    return out
