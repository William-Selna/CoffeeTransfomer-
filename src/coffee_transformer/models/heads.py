"""Output heads and task wrappers.

`HistogramHead` predicts a distribution over yield bins (Section 5): the label
is a scalar but the supervision is a full histogram. `expected_yield` reads the
distribution's mean back out for regression metrics.

`YieldModel` = recurrent-depth encoder + histogram head, returning per-iteration
logits so deep-supervision losses can grade every draft.

`MLMModel` = same encoder + a masked-token head for Stage 1/2 pretraining
(shares the encoder so representations transfer to SFT).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ModelConfig
from .recurrent_depth import RecurrentDepthEncoder


def bin_centers(cfg: ModelConfig, device=None) -> torch.Tensor:
    """Centers of the `num_bins` equal-width bins spanning [yield_min, yield_max]."""
    width = (cfg.yield_max - cfg.yield_min) / cfg.num_bins
    idx = torch.arange(cfg.num_bins, device=device, dtype=torch.float)
    return cfg.yield_min + (idx + 0.5) * width


class HistogramHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.num_bins),
        )
        self.register_buffer("centers", bin_centers(cfg), persistent=False)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)  # logits [B, num_bins]

    def expected_yield(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return (probs * self.centers).sum(dim=-1)


class MLMHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.d_model, cfg.d_model)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.decoder = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.act(self.dense(sequence)))
        return self.decoder(x)  # logits [B, T, vocab]


@dataclass
class YieldOutput:
    logits: torch.Tensor                 # final-iteration logits [B, num_bins]
    iteration_logits: list[torch.Tensor]  # per read-out iteration
    r_used: int


class YieldModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = RecurrentDepthEncoder(cfg)
        self.head = HistogramHead(cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        slot_type_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        r: int | None = None,
        deep_supervision: bool = False,
        generator: torch.Generator | None = None,
    ) -> YieldOutput:
        enc = self.encoder(
            input_ids,
            slot_type_ids,
            attention_mask=attention_mask,
            r=r,
            deep_supervision=deep_supervision,
            generator=generator,
        )
        iter_logits = [self.head(p) for p in enc.pooled_states]
        return YieldOutput(logits=iter_logits[-1], iteration_logits=iter_logits, r_used=enc.r_used)

    @torch.no_grad()
    def predict_yield(self, *args, **kwargs) -> torch.Tensor:
        out = self.forward(*args, **kwargs)
        return self.head.expected_yield(out.logits)


class MLMModel(nn.Module):
    """Stage 1/2 pretraining wrapper — encoder + masked-token head."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = RecurrentDepthEncoder(cfg)
        self.head = MLMHead(cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        slot_type_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        r: int | None = None,
    ) -> torch.Tensor:
        enc = self.encoder(input_ids, slot_type_ids, attention_mask=attention_mask, r=r)
        return self.head(enc.last_sequence)  # [B, T, vocab]

    def transfer_encoder_to(self, model: YieldModel) -> None:
        """Copy pretrained encoder weights into an SFT model (Stage 3)."""
        model.encoder.load_state_dict(self.encoder.state_dict())
