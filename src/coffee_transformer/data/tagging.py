"""Heuristic agent-role tagging for USPTO reactions (design §6, Stage 2).

USPTO reaction strings are `reactants > agents > products`, with the agents
field an unordered, untyped soup of catalysts/ligands/bases/solvents. To feed
Stage-2 pretraining the same slot schema as HTE, we tag each agent by role using
**curated reagent lists** (the design's stated approach) with an `AGENT`
fallback. Reactants/products become REACTANT/PRODUCT spans.

Matching is exact on RDKit-canonical SMILES when RDKit is available (so
resonance/kekulé/atom-order variants still match); without RDKit it falls back
to raw-string membership, which is weaker — extend the curated sets with the
exact strings your corpus uses, or install RDKit.

The lists below are deliberately small but real; extend them for coverage. The
point is the mechanism, and that everything unmatched degrades to AGENT rather
than being dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

try:  # optional
    from rdkit import Chem  # type: ignore
    from rdkit import RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    _HAS_RDKIT = False


# Curated reagent -> role tables (extend as needed). Values are canonicalized
# on first use when RDKit is present.
_RAW_ROLES: dict[str, list[str]] = {
    "SOLVENT": [
        "O", "CO", "CCO", "C1CCOC1", "CN(C)C=O", "ClCCl", "ClCCCl", "Cc1ccccc1",
        "CC#N", "CS(C)=O", "C1COCCO1", "CC(C)=O", "CCOC(C)=O", "CCCCCC", "c1ccccc1",
    ],
    "BASE": [
        "O=C([O-])[O-]", "[OH-]", "CC(C)(C)[O-]", "CCN(CC)CC", "CCN(C(C)C)C(C)C",
        "O=P([O-])([O-])[O-]", "[H-]", "CC(=O)[O-]", "C(=O)([O-])[O-]",
    ],
    "LIG": [
        "c1ccc(P(c2ccccc2)c2ccccc2)cc1",       # PPh3
        "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1",  # XPhos-like
        "C1CCC(P(C2CCCCC2)C2CCCCC2)CC1",       # PCy3
    ],
    "ADD": [
        "[Cl-]", "[Br-]", "[I-]", "F[B-](F)(F)F",
    ],
}


def _canon(smiles: str) -> str:
    if not _HAS_RDKIT:
        return smiles
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else smiles


def _build_lookup() -> dict[str, str]:
    lut: dict[str, str] = {}
    for role, smis in _RAW_ROLES.items():
        for s in smis:
            lut[_canon(s)] = role
    return lut


_LOOKUP = _build_lookup()


def tag_agent(smiles: str) -> str:
    """Return a slot name for one agent SMILES (LIG/BASE/ADD/SOLVENT or AGENT)."""
    # transition-metal catalyst -> AGENT (kept generic; role is 'catalyst')
    role = _LOOKUP.get(_canon(smiles))
    return role if role is not None else "AGENT"


def tag_reaction(rxn_smiles: str) -> list[tuple[str, str]]:
    """`reactants>agents>products` -> ordered [(slot, smiles), ...] or [] if malformed."""
    segs = rxn_smiles.strip().split(">")
    if len(segs) != 3:
        return []
    reactants, agents, products = segs
    comps: list[tuple[str, str]] = []
    for r in filter(None, reactants.split(".")):
        comps.append(("REACTANT", r))
    for a in filter(None, agents.split(".")):
        comps.append((tag_agent(a), a))
    for p in filter(None, products.split(".")):
        comps.append(("PRODUCT", p))
    return comps


def write_reactions_jsonl(rxns: Iterable[str], out_path: str | Path) -> int:
    """Tag reaction strings and write one JSON list of [slot, smiles] per line.
    Returns the number of reactions written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as fh:
        for rxn in rxns:
            comps = tag_reaction(rxn)
            if not comps:
                continue
            fh.write(json.dumps([[s, m] for s, m in comps]) + "\n")
            n += 1
    return n
