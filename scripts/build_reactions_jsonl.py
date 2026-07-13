#!/usr/bin/env python
"""Turn raw USPTO/ORD reactions into the slot-tagged reactions.jsonl that
prepare_corpus.py consumes for Stage-2 pretraining.

    # USPTO rsmi (reaction SMILES in column 0, tab-separated)
    python scripts/build_reactions_jsonl.py --uspto data/raw/uspto_1976_Sep2016.rsmi.gz \
        --out data/raw/reactions.jsonl --limit 500000

    # offline smoke (synthetic toy reactions, no data)
    python scripts/build_reactions_jsonl.py --synthetic --out data/raw/reactions.jsonl

Agent roles are assigned by curated reagent lists (see data/tagging.py); extend
those lists for better coverage. ORD support needs the `ord-schema` package and
is stubbed with a clear message.
"""

from __future__ import annotations

import argparse
import gzip
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from coffee_transformer.data.tagging import write_reactions_jsonl


def _open(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def _uspto_reactions(path: str, smiles_col: int, sep: str, limit: int | None):
    n = 0
    with _open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(sep)
            if len(parts) <= smiles_col:
                continue
            rxn = parts[smiles_col]
            if ">" not in rxn:
                continue
            yield rxn
            n += 1
            if limit and n >= limit:
                return


def _synthetic_reactions(n: int):
    # toy reactants>agents>products with a mix of taggable + fallback agents
    import random
    rng = random.Random(0)
    reactants = ["CC(=O)O", "c1ccccc1N", "CCBr", "NCCO"]
    agents = ["[Pd]", "O=C([O-])[O-]", "CN(C)C=O", "CCN(CC)CC", "c1ccc(P(c2ccccc2)c2ccccc2)cc1"]
    products = ["CC(=O)Nc1ccccc1", "CCOC(C)=O", "O=C(O)c1ccccc1"]
    for _ in range(n):
        r = ".".join(rng.sample(reactants, 2))
        a = ".".join(rng.sample(agents, 2))
        p = rng.choice(products)
        yield f"{r}>{a}>{p}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--uspto", default=None, help="rsmi/tsv path (.gz ok)")
    p.add_argument("--ord", default=None, help="ORD dataset dir (needs ord-schema)")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--smiles-col", type=int, default=0)
    p.add_argument("--sep", default="\t")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.ord:
        raise SystemExit(
            "ORD ingestion needs the `ord-schema` package: iterate the dataset's "
            "reactions, serialize each as reactants>agents>products, and feed the "
            "strings through data.tagging.write_reactions_jsonl (ORD's typed roles "
            "can bypass the heuristic tagger — map catalyst/ligand/base/solvent "
            "directly to slots)."
        )
    if args.synthetic or not args.uspto:
        rxns = _synthetic_reactions(args.limit or 500)
    else:
        rxns = _uspto_reactions(args.uspto, args.smiles_col, args.sep, args.limit)

    written = write_reactions_jsonl(rxns, args.out)
    print(f"wrote {written} tagged reactions -> {args.out}")


if __name__ == "__main__":
    main()
