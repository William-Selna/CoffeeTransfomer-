"""Stages 1-2 — self-supervised pretraining (masked-SMILES modeling).

Stage 1 (molecular grammar, PubChem): mask ~15% of tokens, predict each via a
linear head over the ~300-token vocab, cross-entropy. Teaches valence/ring
syntax so later stages don't spend capacity on grammar.

Stage 2 (reactivity, USPTO+ORD): same objective over reaction strings, with the
option to span-mask entire slot contents ("infer the plausible ligand from
context") — nearly the downstream task in self-supervised clothing.

This module scaffolds the objective and loop against `MLMModel`. Wiring the
actual corpora (canonicalization, dedup, [UNK] and leakage checks — Section 6)
lives in the data pipeline; see `docs/PIPELINE.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.tokenizer import CORE_SPECIALS, SmilesTokenizer
from ..models.heads import MLMModel


class MoleculeMLMDataset(Dataset):
    """One molecule (or reaction) SMILES string per item, tokenized with [CLS]."""

    def __init__(self, smiles: list[str], tokenizer: SmilesTokenizer, max_length: int = 256):
        self.smiles = smiles
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
        ids = [self.tok.cls_id] + self.tok.encode_smiles(self.smiles[idx])
        ids = ids[: self.max_length]
        return {"input_ids": ids, "slot_type_ids": [0] * len(ids)}


def mask_tokens(
    input_ids: torch.Tensor,
    tokenizer: SmilesTokenizer,
    mlm_prob: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT 80/10/10 masking. Returns (masked_input_ids, labels) with labels
    set to -100 on unmasked positions."""
    labels = input_ids.clone()
    special_ids = {tokenizer.token_to_id[t] for t in CORE_SPECIALS}
    special_ids |= {tokenizer.slot_token_id(s) for s in tokenizer.schema.slots}

    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        special_mask |= input_ids == sid

    prob = torch.full(input_ids.shape, mlm_prob)
    prob.masked_fill_(special_mask, 0.0)
    masked = torch.bernoulli(prob, generator=generator).bool()
    labels[~masked] = -100

    # 80% -> [MASK]
    replace = torch.bernoulli(torch.full(input_ids.shape, 0.8), generator=generator).bool() & masked
    input_ids = input_ids.clone()
    input_ids[replace] = tokenizer.mask_id
    # 10% -> random token
    rand = torch.bernoulli(torch.full(input_ids.shape, 0.5), generator=generator).bool() & masked & ~replace
    random_tokens = torch.randint(tokenizer.vocab_size, input_ids.shape, generator=generator)
    input_ids[rand] = random_tokens[rand]
    # remaining 10% keep original
    return input_ids, labels


def mlm_collate(batch, pad_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    slot_type_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"])
        attention_mask[i, :n] = True
    return {"input_ids": input_ids, "slot_type_ids": slot_type_ids, "attention_mask": attention_mask}


@dataclass
class MLMResult:
    steps: int
    final_loss: float


class MLMTrainer:
    def __init__(self, model: MLMModel, tokenizer, device, mlm_prob=0.15, lr=3e-4, generator=None):
        self.model = model.to(device)
        self.tok = tokenizer
        self.device = device
        self.mlm_prob = mlm_prob
        self.generator = generator
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train(self, loader: DataLoader, epochs: int = 1, log_every: int = 50) -> MLMResult:
        step = 0
        last = float("nan")
        for _ in range(epochs):
            for batch in loader:
                self.model.train()
                masked, labels = mask_tokens(batch["input_ids"], self.tok, self.mlm_prob, self.generator)
                masked = masked.to(self.device)
                labels = labels.to(self.device)
                slot_type_ids = batch["slot_type_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits = self.model(masked, slot_type_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                if step % log_every == 0:
                    print(f"[mlm] step {step} loss {float(loss.detach()):.4f}")
                step += 1
                last = float(loss.detach())
        return MLMResult(steps=step, final_loss=last)
