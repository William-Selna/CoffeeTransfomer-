"""Input embeddings: token + slot-type + (learned) positional.

The slot-type embedding is the cheap arm of the typed-attention ablation
(Section 4): a learned vector per slot type, added to every token in that
component's span (BERT segment-embedding style).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class ReactionEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.token = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.slot_type = nn.Embedding(cfg.num_slot_types, cfg.d_model)
        self.position = nn.Embedding(cfg.max_position, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "position_ids", torch.arange(cfg.max_position).unsqueeze(0), persistent=False
        )

    def forward(self, input_ids: torch.Tensor, slot_type_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        pos = self.position_ids[:, :seq_len]
        emb = self.token(input_ids) + self.slot_type(slot_type_ids) + self.position(pos)
        return self.dropout(self.norm(emb))
