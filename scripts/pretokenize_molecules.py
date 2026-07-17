#!/usr/bin/env python
"""Pre-tokenize an already-cleaned molecule .smi into the Stage-1 mmap store,
WITHOUT re-running the slow full prepare_corpus (RDKit clean of 10M molecules).

Use this when prepare_corpus already wrote data/processed/pubchem.smi (cleaned)
and tokenizer.json, and you just need the pre-tokenized `molecules.*` for the
fast Stage-1 path.

    python scripts/pretokenize_molecules.py    # uses the data/processed defaults
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from coffee_transformer.data.corpus import (
    augmented_molecule_records,
    molecule_records,
    pretokenize_records,
    read_smiles_lines,
)
from coffee_transformer.data.tokenizer import SmilesTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smi", default="data/processed/pubchem.smi", help="cleaned molecules, one/line")
    p.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    p.add_argument("--out", default="data/processed/molecules", help="output prefix")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--augment", type=int, default=1,
                   help="spellings per molecule: 1 = canonical only; 5 = canonical + 4 "
                        "randomized SMILES (teaches spelling-invariance; needs RDKit)")
    args = p.parse_args()

    tok = SmilesTokenizer.load(args.tokenizer)
    mols = read_smiles_lines(args.smi)
    if args.augment > 1:
        print(f"tokenizing {len(mols):,} molecules x{args.augment} spellings -> {args.out}.* "
              f"(this takes a while — RDKit re-renders each molecule) ...")
        records = augmented_molecule_records(mols, tok, args.max_length, args.augment)
    else:
        print(f"tokenizing {len(mols):,} molecules -> {args.out}.* ...")
        records = molecule_records(mols, tok, args.max_length)
    meta = pretokenize_records(records, args.out)
    print(f"done: {meta}")


if __name__ == "__main__":
    main()
