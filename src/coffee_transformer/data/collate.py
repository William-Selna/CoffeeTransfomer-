"""Batching: pad variable-length reactions and build the attention mask.

Padding positions get `attention_mask == 0`; the model turns that into a
`-inf` additive bias before softmax so pad keys receive zero attention mass
(Section 4: never rely on zero vectors — a zero key still gets mass after
softmax renormalization). "Missing slots" are simply components that are
absent from a reaction; they never appear as tokens, so nothing to mask beyond
padding.
"""

from __future__ import annotations

from typing import Sequence

import torch


def collate_reactions(batch: Sequence[dict], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    slot_type_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    yields = torch.zeros(bsz, dtype=torch.float)

    for i, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[i, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
        slot_type_ids[i, :n] = torch.tensor(item["slot_type_ids"], dtype=torch.long)
        attention_mask[i, :n] = True
        yields[i] = item["yield_value"]

    return {
        "input_ids": input_ids,
        "slot_type_ids": slot_type_ids,
        "attention_mask": attention_mask,  # bool [B, T], True = real token
        "yield_value": yields,
    }


def make_collate_fn(pad_id: int):
    def _fn(batch: Sequence[dict]) -> dict[str, torch.Tensor]:
        return collate_reactions(batch, pad_id=pad_id)

    return _fn
