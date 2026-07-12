"""Reaction slot schema.

A reaction is a set of typed components (an aryl halide, a ligand, a base, an
additive, ...). Each component is spelled as SMILES and prefixed with a slot
token; every token in the span also carries a learned slot-type embedding
(BERT segment-embedding style, added in the model's embedding layer).

The schema below is deliberately shared across pretraining and fine-tuning
(Section 4 of the design): USPTO/ORD reactivity pretraining tags agent roles
with the same slot tokens so the representation transfers cleanly to HTE SFT.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Canonical slot names. `AGENT` is the fallback role for USPTO reagents whose
# functional role can't be resolved heuristically (Section 6, Stage 2).
BUCHWALD_HARTWIG_SLOTS: tuple[str, ...] = (
    "ARYL_HALIDE",
    "LIG",
    "BASE",
    "ADD",
)

SUZUKI_MIYAURA_SLOTS: tuple[str, ...] = (
    "ELECTROPHILE",
    "NUCLEOPHILE",
    "LIG",
    "REAGENT",
    "SOLVENT",
)

# Union used to size the slot-type embedding table so a single encoder can be
# shared across datasets (Section 4, "multi-dataset handling").
ALL_SLOTS: tuple[str, ...] = (
    "ARYL_HALIDE",
    "ELECTROPHILE",
    "NUCLEOPHILE",
    "LIG",
    "BASE",
    "ADD",
    "REAGENT",
    "SOLVENT",
    "AGENT",
    "REACTANT",
    "PRODUCT",
)

# Sentinel slot id for special/structural tokens ([CLS], [PAD], separators)
# that do not belong to any chemical component.
NO_SLOT = "NONE"


@dataclass(frozen=True)
class SlotSchema:
    """Maps slot names <-> ids and exposes the special slot tokens.

    `slot_token(name)` -> the vocabulary string (e.g. "[LIG]") that the
    tokenizer emits as a single token to open a component span.
    """

    slots: tuple[str, ...] = field(default=ALL_SLOTS)

    def __post_init__(self) -> None:
        names = (NO_SLOT,) + tuple(self.slots)
        object.__setattr__(self, "_name_to_id", {n: i for i, n in enumerate(names)})
        object.__setattr__(self, "_id_to_name", {i: n for n, i in self._name_to_id.items()})

    @property
    def num_slot_types(self) -> int:
        """Includes the NO_SLOT sentinel at id 0."""
        return len(self._name_to_id)

    def slot_id(self, name: str) -> int:
        return self._name_to_id[name]

    def slot_name(self, idx: int) -> str:
        return self._id_to_name[idx]

    @staticmethod
    def slot_token(name: str) -> str:
        """Vocabulary string for a slot's opening special token."""
        return f"[{name}]"

    def slot_tokens(self) -> list[str]:
        return [self.slot_token(n) for n in self.slots]


DEFAULT_SCHEMA = SlotSchema()
