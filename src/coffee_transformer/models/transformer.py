"""Pre-norm transformer block (self-attention + FFN with residuals)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .config import ModelConfig

_ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        act = _ACTIVATIONS[cfg.activation]()
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            act,
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        slot_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(
            self.attn(self.norm1(x), key_padding_mask=key_padding_mask, slot_type_ids=slot_type_ids)
        )
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
