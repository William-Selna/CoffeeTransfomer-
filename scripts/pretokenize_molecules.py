#!/usr/bin/env python
"""Pre-tokenize an already-cleaned molecule .smi into the Stage-1 mmap store,
WITHOUT re-running the slow full prepare_corpus (RDKit clean of 10M molecules).

Parallel across cores — augmentation is embarrassingly parallel, so this rips
through 10M x5 spellings in minutes on a many-core box instead of hours.

    python scripts/pretokenize_molecules.py                       # canonical only
    python scripts/pretokenize_molecules.py --augment 5 --workers 64
"""

from __future__ import annotations

import argparse
import os
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from coffee_transformer.data.corpus import (
    pretokenize_from_arrays,
    pretokenize_records,
    molecule_records,
    read_smiles_lines,
)
from coffee_transformer.data.tokenizer import SmilesTokenizer

# --- worker (module-level so multiprocessing can pickle it) -----------------
_TOK: SmilesTokenizer | None = None


def _init_worker(tokenizer_path: str) -> None:
    global _TOK
    _TOK = SmilesTokenizer.load(tokenizer_path)


def _augment_chunk(payload):
    """Randomize + tokenize a chunk of molecules; return compact int16 arrays."""
    from coffee_transformer.data.augment import randomize_smiles

    smiles, n_aug, max_length, seed = payload
    rng = random.Random(seed)
    tok = _TOK
    toks: list[int] = []
    slots: list[int] = []
    lengths: list[int] = []
    for smi in smiles:
        forms = [smi]
        for _ in range(max(0, n_aug - 1)):
            forms.append(randomize_smiles(smi, rng))
        for f in forms:
            ids = ([tok.cls_id] + tok.encode_smiles(f))[:max_length]
            toks.extend(ids)
            slots.extend([0] * len(ids))
            lengths.append(len(ids))
    return (
        np.asarray(toks, dtype=np.int16),
        np.asarray(slots, dtype=np.int16),
        np.asarray(lengths, dtype=np.int64),
    )


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smi", default="data/processed/pubchem.smi", help="cleaned molecules, one/line")
    p.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    p.add_argument("--out", default="data/processed/molecules", help="output prefix")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--augment", type=int, default=1,
                   help="spellings per molecule (5 = canonical + 4 randomized; needs RDKit)")
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8),
                   help="parallel worker processes")
    p.add_argument("--chunk", type=int, default=20000, help="molecules per task")
    args = p.parse_args()

    mols = read_smiles_lines(args.smi)
    print(f"molecules={len(mols):,} augment=x{args.augment} workers={args.workers}")

    # canonical-only + single worker: keep the simple generator path
    if args.augment <= 1 and args.workers <= 1:
        tok = SmilesTokenizer.load(args.tokenizer)
        meta = pretokenize_records(molecule_records(mols, tok, args.max_length), args.out)
        print(f"done: {meta}")
        return

    import multiprocessing as mp

    payloads = [
        (chunk, args.augment, args.max_length, i)
        for i, chunk in enumerate(_chunks(mols, args.chunk))
    ]
    tok_parts, slot_parts, len_parts = [], [], []
    done = 0
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        for t, s, ln in pool.imap(_augment_chunk, payloads):
            tok_parts.append(t)
            slot_parts.append(s)
            len_parts.append(ln)
            done += 1
            if done % 50 == 0:
                print(f"  ~{done * args.chunk:,} molecules processed ...", flush=True)

    meta = pretokenize_from_arrays(
        np.concatenate(tok_parts),
        np.concatenate(slot_parts),
        np.concatenate(len_parts),
        args.out,
    )
    print(f"done: {meta}")


if __name__ == "__main__":
    main()
