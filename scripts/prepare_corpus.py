#!/usr/bin/env python
"""Build the shared tokenizer + cleaned/pre-tokenized pretraining corpus.

This is the offline data step the design's Section 6 hygiene checklist calls
for. Run it once on the box with the raw corpora before pretraining.

    # real
    python scripts/prepare_corpus.py \
        --pubchem data/raw/pubchem.smi --limit 10000000 \
        --reactions data/raw/reactions.jsonl \
        --hte data/raw/Dreher_and_Doyle_input_data.xlsx --sheet FullCV_01 \
        --out data/processed

    # offline smoke (synthetic toy corpora, no downloads)
    python scripts/prepare_corpus.py --synthetic --out data/processed

Outputs in --out:
  pubchem.smi        cleaned, deduped molecule SMILES (Stage 1)
  reactions.*        pre-tokenized mmap store (Stage 2): tokens/slots/offsets/meta
  tokenizer.json     shared vocab covering PubChem + reactions + HTE (zero [UNK])

`--reactions` is JSONL, one reaction per line: a list of [slot_name, smiles]
pairs, e.g. [["ARYL_HALIDE","Clc1ccccc1"],["LIG","..."],["BASE","..."]].
Heuristic USPTO agent-role tagging / ORD schema extraction produce this file
upstream (see docs/PIPELINE.md) — kept separate so the parsing choices are
explicit and auditable.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from coffee_transformer.data.corpus import (
    assert_no_unk,
    check_no_leakage,
    clean_corpus,
    pretokenize_records,
    reaction_records,
    read_smiles_lines,
)
from coffee_transformer.data.dataset import load_buchwald_hartwig
from coffee_transformer.data.slots import DEFAULT_SCHEMA
from coffee_transformer.data.synthetic import (
    synthetic_molecule_corpus,
    synthetic_reaction_corpus,
)
from coffee_transformer.data.tokenizer import SmilesTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/processed")
    p.add_argument("--pubchem", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reactions", default=None, help="JSONL of [[slot,smiles],...] per line")
    p.add_argument("--hte", default=None, help="BH xlsx (for vocab + leakage check)")
    p.add_argument("--sheet", default="FullCV_01")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--synthetic", action="store_true")
    return p.parse_args()


def _load_reactions(path):
    reactions = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                reactions.append([tuple(pair) for pair in json.loads(line)])
    return reactions


def main():
    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- molecules (Stage 1) ---
    if args.synthetic or not args.pubchem:
        molecules = synthetic_molecule_corpus(800)
        print(f"[molecules] synthetic: {len(molecules)}")
    else:
        raw = read_smiles_lines(args.pubchem, limit=args.limit)
        molecules = clean_corpus(raw)
        print(f"[molecules] {len(raw)} raw -> {len(molecules)} cleaned/deduped")
    (out / "pubchem.smi").write_text("\n".join(molecules))

    # --- reactions (Stage 2) ---
    if args.synthetic or not args.reactions:
        reactions = synthetic_reaction_corpus(800)
        print(f"[reactions] synthetic: {len(reactions)}")
    else:
        reactions = _load_reactions(args.reactions)
        print(f"[reactions] loaded: {len(reactions)}")

    # --- HTE molecules (vocab coverage + leakage) ---
    hte_smiles = []
    if args.hte:
        hte = load_buchwald_hartwig(args.hte, sheet=args.sheet)
        hte_smiles = [smi for ex in hte for _, smi in ex.components]

    # --- shared tokenizer ---
    vocab_corpus = list(molecules)
    for comps in reactions:
        vocab_corpus.extend(smi for _, smi in comps)
    vocab_corpus.extend(hte_smiles)
    tokenizer = SmilesTokenizer.build(vocab_corpus, schema=DEFAULT_SCHEMA)
    tokenizer.save(out / "tokenizer.json")
    print(f"[tokenizer] vocab {tokenizer.vocab_size}")

    # --- hygiene gates ---
    if hte_smiles:
        assert_no_unk(hte_smiles, tokenizer, name="HTE")
        leaked = check_no_leakage(molecules, hte_smiles, canonicalized=True)
        if leaked:
            print(f"[WARN] {len(leaked)} HTE molecules appear in the PubChem corpus "
                  f"(fine for Stage 1 grammar; ensure no exact test REACTIONS leak in Stage 2)")

    # --- pre-tokenize reactions to mmap ---
    meta = pretokenize_records(
        reaction_records(reactions, tokenizer, args.max_length), out / "reactions"
    )
    print(f"[reactions] pretokenized -> {out}/reactions.* ({meta})")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
