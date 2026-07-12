"""Reaction datasets and loaders.

`ReactionExample` is the schema-agnostic unit: an ordered list of
(slot_name, SMILES) components plus a measured yield in [0, 100].

`HTEDataset` tokenizes on the fly and applies augmentation. It returns plain
dicts of python lists; batching/padding happens in `collate.py`.

`load_buchwald_hartwig` reads the canonical Dreher_and_Doyle xlsx sheets. It is
written against the documented column layout but kept dependency-light: pandas
is imported lazily so the synthetic path needs neither pandas nor the file.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from torch.utils.data import Dataset

from .augment import randomize_slot_order, randomize_smiles
from .tokenizer import SmilesTokenizer


@dataclass
class ReactionExample:
    components: list[tuple[str, str]]  # ordered (slot_name, smiles)
    yield_value: float                 # measured yield, 0-100
    dataset: str = "BH"                # for per-dataset heads / conditioning
    meta: dict = field(default_factory=dict)


class HTEDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[ReactionExample],
        tokenizer: SmilesTokenizer,
        randomize_order: bool = False,
        randomize_smiles_prob: float = 0.0,
        max_length: int | None = None,
        seed: int = 0,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.randomize_order = randomize_order
        self.randomize_smiles_prob = randomize_smiles_prob
        self.max_length = max_length
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        components = ex.components
        if self.randomize_order:
            components = randomize_slot_order(components, self._rng)
        if self.randomize_smiles_prob > 0:
            components = [
                (slot, randomize_smiles(smi, self._rng))
                if self._rng.random() < self.randomize_smiles_prob
                else (slot, smi)
                for slot, smi in components
            ]

        input_ids, slot_type_ids = self.tokenizer.encode_reaction(components)
        if self.max_length is not None:
            input_ids = input_ids[: self.max_length]
            slot_type_ids = slot_type_ids[: self.max_length]

        return {
            "input_ids": input_ids,
            "slot_type_ids": slot_type_ids,
            "yield_value": float(ex.yield_value),
        }


# --- Buchwald-Hartwig canonical loader ------------------------------------

# Column names in Dreher_and_Doyle_input_data.xlsx (FullCV_01..10 sheets).
# Adjust here if a particular release differs.
_BH_COLUMNS = {
    "ARYL_HALIDE": "Aryl halide",
    "LIG": "Ligand",
    "BASE": "Base",
    "ADD": "Additive",
}
_BH_YIELD_COLUMN = "Output"


def load_buchwald_hartwig(
    xlsx_path: str | Path,
    sheet: str = "FullCV_01",
    smiles_columns: dict[str, str] | None = None,
    yield_column: str = _BH_YIELD_COLUMN,
) -> list[ReactionExample]:
    """Load one BH CV sheet into ReactionExamples.

    Sheets FullCV_01..10 are the canonical random folds; Tests 1-4 are the
    additive-holdout OOD splits (Section 2). Pass `sheet=` accordingly.
    """
    import pandas as pd  # lazy: only needed for the real corpus

    cols = smiles_columns or _BH_COLUMNS
    df = pd.read_excel(xlsx_path, sheet_name=sheet)

    examples: list[ReactionExample] = []
    for _, row in df.iterrows():
        components = [(slot, str(row[col])) for slot, col in cols.items() if col in df.columns]
        examples.append(
            ReactionExample(
                components=components,
                yield_value=float(row[yield_column]),
                dataset="BH",
            )
        )
    return examples
