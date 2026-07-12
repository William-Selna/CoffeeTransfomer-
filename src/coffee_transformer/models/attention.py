"""Multi-head self-attention with key-padding masking and an optional typed
(per-slot-type) projection ablation.

`typed_attention="embedding"` (default) uses ordinary shared QKV projections;
the slot signal enters only through the slot-type embedding. This is the cheap
arm and, per the design's prior, expected to win at this scale.

`typed_attention="per_slot_kqv"` gives each slot type its own Q/K/V weights
(~heterogeneous-graph-transformer style, ~6x attention parameters). It is
implemented for the ablation; expect higher memory/compute.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class TypedLinear(nn.Module):
    """Per-slot-type linear map: token t uses weight W[slot_type[t]].

    weight: [S, D_in, D_out], bias: [S, D_out]. Used only for the
    `per_slot_kqv` ablation; the memory cost is O(B*T*D_in*D_out) for the
    gathered weights, acceptable at the ~5M-param scale this targets.
    """

    def __init__(self, num_types: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_types, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(num_types, d_out))
        nn.init.normal_(self.weight, std=(d_in ** -0.5))

    def forward(self, x: torch.Tensor, slot_type_ids: torch.Tensor) -> torch.Tensor:
        w = self.weight[slot_type_ids]          # [B, T, D_in, D_out]
        b = self.bias[slot_type_ids]            # [B, T, D_out]
        return torch.einsum("btd,btde->bte", x, w) + b


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.typed = cfg.typed_attention == "per_slot_kqv"

        if self.typed:
            self.q = TypedLinear(cfg.num_slot_types, cfg.d_model, cfg.d_model)
            self.k = TypedLinear(cfg.num_slot_types, cfg.d_model, cfg.d_model)
            self.v = TypedLinear(cfg.num_slot_types, cfg.d_model, cfg.d_model)
        else:
            self.q = nn.Linear(cfg.d_model, cfg.d_model)
            self.k = nn.Linear(cfg.d_model, cfg.d_model)
            self.v = nn.Linear(cfg.d_model, cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def _proj(self, layer, x, slot_type_ids):
        return layer(x, slot_type_ids) if self.typed else layer(x)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,dh]

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,  # bool [B,T], True = real token
        slot_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split(self._proj(self.q, x, slot_type_ids))
        k = self._split(self._proj(self.k, x, slot_type_ids))
        v = self._split(self._proj(self.v, x, slot_type_ids))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if key_padding_mask is not None:
            # set masked (pad) key logits to -inf before softmax (Section 4)
            bias = torch.zeros_like(key_padding_mask, dtype=scores.dtype)
            bias = bias.masked_fill(~key_padding_mask, float("-inf"))
            scores = scores + bias[:, None, None, :]

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v)                       # [B,H,T,dh]
        ctx = ctx.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return self.out(ctx)
