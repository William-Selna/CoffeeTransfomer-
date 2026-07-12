"""Pre-norm transformer block (self-attention + FFN with residuals).

The FFN supports both pointwise activations (gelu/relu/silu) and gated variants
(swiglu/geglu). Gated FFNs are the modern default (Llama/PaLM) and are the most
likely single-knob quality win here; because the same core block is applied r
times in the recurrent depth loop, a smooth gated nonlinearity is also a safer
choice than a hard ReLU for the iterated map.

Param matching: a gated FFN uses two input projections instead of one, so its
hidden width is scaled to 2/3 * d_ff — with the default d_ff=1536 that is 1024,
giving 3 * d * 1024 == 2 * d * 1536 projection parameters, i.e. the same as the
pointwise FFN (only the handful of bias terms differ). This keeps a
gelu-vs-swiglu comparison matched-parameter to within a fraction of a percent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .config import ModelConfig

_POINTWISE = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
_GATED = {"swiglu": nn.SiLU, "geglu": nn.GELU}  # name -> gate activation


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        act = cfg.activation
        self.dropout = nn.Dropout(cfg.dropout)
        if act in _GATED:
            self.gated = True
            hidden = max(8, int(round(2 * cfg.d_ff / 3)))
            self.gate = nn.Linear(cfg.d_model, hidden)
            self.value = nn.Linear(cfg.d_model, hidden)
            self.act = _GATED[act]()
            self.out = nn.Linear(hidden, cfg.d_model)
        elif act in _POINTWISE:
            self.gated = False
            self.net = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_ff),
                _POINTWISE[act](),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_ff, cfg.d_model),
            )
        else:
            raise ValueError(
                f"unknown activation {act!r}; use one of "
                f"{sorted(_POINTWISE) + sorted(_GATED)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            return self.out(self.dropout(self.act(self.gate(x)) * self.value(x)))
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)
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
