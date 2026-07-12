"""Pretraining corpus: ingestion, hygiene, and pre-tokenization.

Covers the Section 6 corpus-hygiene checklist for Stages 1-2:
  * RDKit canonicalization, drop unparseable, dedup after canonicalization,
    salt stripping, size/element filters;
  * verify zero [UNK] tokens on the HTE molecules;
  * grep-check that no exact test reaction leaks into the pretraining corpus.

RDKit is imported lazily and every hygiene step degrades to a safe no-op
without it, so the synthetic smoke path needs no chemistry stack. For real
runs, `pip install -e ".[chem]"`.

Pre-tokenization packs the whole corpus into two flat memory-mapped arrays
(token ids + slot ids) plus offsets, so the training dataloader does zero
Python tokenization at step time — the difference between feeding an H100 and
starving it at this model scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from torch.utils.data import Dataset

from .tokenizer import SmilesTokenizer

try:  # optional
    from rdkit import Chem  # type: ignore
    from rdkit import RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    _HAS_RDKIT = False


# --- ingestion --------------------------------------------------------------

def read_smiles_lines(path: str | Path, limit: int | None = None) -> list[str]:
    """Read a .smi/.txt corpus, one SMILES per line (first whitespace field)."""
    out: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(line.split()[0])
            if limit is not None and len(out) >= limit:
                break
    return out


# --- hygiene (Section 6) ----------------------------------------------------

_ALLOWED_ELEMENTS = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "H", "Si"}


def _strip_salts(mol):
    # keep the largest fragment (drops counterions / solvents of crystallization)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def canonicalize(smiles: str, strip_salts: bool = True) -> str | None:
    """Canonical SMILES, or None if unparseable / filtered out."""
    if not _HAS_RDKIT:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if strip_salts:
        mol = _strip_salts(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def clean_corpus(
    smiles: Iterable[str],
    min_heavy: int = 2,
    max_heavy: int = 80,
    strip_salts: bool = True,
    element_filter: bool = True,
) -> list[str]:
    """Canonicalize, salt-strip, size/element filter, and dedup.

    Without RDKit this only dedups (canonicalization/filters are skipped with a
    single pass-through), so it stays runnable in the smoke path.
    """
    seen: set[str] = set()
    out: list[str] = []
    for smi in smiles:
        canon = canonicalize(smi, strip_salts=strip_salts)
        if canon is None or canon in seen:
            continue
        if _HAS_RDKIT and (min_heavy or max_heavy or element_filter):
            mol = Chem.MolFromSmiles(canon)
            if mol is None:
                continue
            nh = mol.GetNumHeavyAtoms()
            if nh < min_heavy or nh > max_heavy:
                continue
            if element_filter and any(
                a.GetSymbol() not in _ALLOWED_ELEMENTS for a in mol.GetAtoms()
            ):
                continue
        seen.add(canon)
        out.append(canon)
    return out


def assert_no_unk(smiles: Sequence[str], tokenizer: SmilesTokenizer, name: str = "corpus") -> None:
    """Fail loudly if any molecule tokenizes to [UNK] (Section 6 hygiene gate)."""
    bad = [s for s in smiles if tokenizer.unk_id in tokenizer.encode_smiles(s)]
    if bad:
        raise ValueError(f"{name}: {len(bad)} molecules produced [UNK], e.g. {bad[:3]}")


def check_no_leakage(
    corpus: Iterable[str],
    held_out: Iterable[str],
    canonicalized: bool = True,
) -> list[str]:
    """Return any held-out (test) SMILES that also appear in the corpus.

    Compare after canonicalization so trivial rewrites don't hide a leak.
    """
    norm = (lambda s: s) if canonicalized else (lambda s: canonicalize(s) or s)
    corpus_set = {norm(s) for s in corpus}
    return [s for s in held_out if norm(s) in corpus_set]


# --- pre-tokenization to a memory-mapped store ------------------------------

def pretokenize_records(
    records: Iterable[tuple[list[int], list[int]]],
    out_prefix: str | Path,
) -> dict:
    """Pack (input_ids, slot_type_ids) records into flat mmap-able arrays.

    Writes <prefix>.tokens.npy (int16), <prefix>.slots.npy (int16),
    <prefix>.offsets.npy (int64), and <prefix>.meta.json.
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    tokens: list[int] = []
    slots: list[int] = []
    offsets: list[int] = [0]
    for input_ids, slot_ids in records:
        tokens.extend(input_ids)
        slots.extend(slot_ids)
        offsets.append(len(tokens))

    tok_arr = np.asarray(tokens, dtype=np.int16)
    slot_arr = np.asarray(slots, dtype=np.int16)
    off_arr = np.asarray(offsets, dtype=np.int64)
    np.save(f"{out_prefix}.tokens.npy", tok_arr)
    np.save(f"{out_prefix}.slots.npy", slot_arr)
    np.save(f"{out_prefix}.offsets.npy", off_arr)
    meta = {"num_records": len(off_arr) - 1, "num_tokens": int(tok_arr.size)}
    Path(f"{out_prefix}.meta.json").write_text(json.dumps(meta))
    return meta


def molecule_records(
    smiles: Sequence[str], tokenizer: SmilesTokenizer, max_length: int
) -> Iterator[tuple[list[int], list[int]]]:
    for smi in smiles:
        ids = ([tokenizer.cls_id] + tokenizer.encode_smiles(smi))[:max_length]
        yield ids, [0] * len(ids)


def reaction_records(
    reactions: Sequence[Sequence[tuple[str, str]]],
    tokenizer: SmilesTokenizer,
    max_length: int,
) -> Iterator[tuple[list[int], list[int]]]:
    for components in reactions:
        ids, slots = tokenizer.encode_reaction(components)
        yield ids[:max_length], slots[:max_length]


class PackedTokenDataset(Dataset):
    """Reads a pre-tokenized mmap store; returns dicts for the MLM collate."""

    def __init__(self, prefix: str | Path):
        prefix = str(prefix)
        self.tokens = np.load(f"{prefix}.tokens.npy", mmap_mode="r")
        self.slots = np.load(f"{prefix}.slots.npy", mmap_mode="r")
        self.offsets = np.load(f"{prefix}.offsets.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, idx: int) -> dict:
        lo, hi = int(self.offsets[idx]), int(self.offsets[idx + 1])
        return {
            "input_ids": self.tokens[lo:hi].astype(np.int64).tolist(),
            "slot_type_ids": self.slots[lo:hi].astype(np.int64).tolist(),
        }
