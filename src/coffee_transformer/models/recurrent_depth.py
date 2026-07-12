"""Recurrent-depth encoder (Geiping et al. 2025; Universal Transformer ancestor).

    prelude (runs once, embeds the reaction)
      -> recurrent core (a block applied r times with the SAME weights)
      -> coda (LayerNorm + pooling)

Key mechanics from Section 3 of the design:
  * state s_0 initialized with Gaussian noise;
  * input embedding h re-injected every iteration as (s + h) — prevents drift,
    anchors each pass to the specific reaction (mandatory input injection);
  * r randomized during training (sampled in [r_min, r_max]);
  * truncated BPTT — only the last `truncated_bptt_k` iterations carry grad
    (constant memory, literally truncated backprop through the depth-unroll);
  * deep supervision — the coda can be read out at EVERY iteration so a loss is
    applied to each "draft" (Section 5); this module returns per-iteration
    pooled states for that.

Set `cfg.recurrent = False` for the vanilla-transformer baseline: the core runs
exactly once (r == 1) with no input injection and no state noise.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ModelConfig
from .embeddings import ReactionEmbedding
from .transformer import TransformerBlock


@dataclass
class EncoderOutput:
    # one entry per read-out iteration (all iterations if deep_supervision else last only)
    sequence_states: list[torch.Tensor]  # each [B, T, D]
    pooled_states: list[torch.Tensor]     # each [B, D]
    r_used: int

    @property
    def last_sequence(self) -> torch.Tensor:
        return self.sequence_states[-1]

    @property
    def last_pooled(self) -> torch.Tensor:
        return self.pooled_states[-1]


def _pool(seq: torch.Tensor, mask: torch.Tensor | None, mode: str) -> torch.Tensor:
    if mode == "cls":
        return seq[:, 0]
    if mode == "mean":
        if mask is None:
            return seq.mean(dim=1)
        m = mask.unsqueeze(-1).to(seq.dtype)
        return (seq * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    raise ValueError(f"unknown pool mode: {mode}")


class RecurrentDepthEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = ReactionEmbedding(cfg)
        self.prelude = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.prelude_layers)])
        self.core = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.core_layers)])
        self.coda_norm = nn.LayerNorm(cfg.d_model)

    def _run_core(self, s, key_padding_mask, slot_type_ids):
        for layer in self.core:
            s = layer(s, key_padding_mask=key_padding_mask, slot_type_ids=slot_type_ids)
        return s

    def _sample_r(self, generator: torch.Generator | None) -> int:
        cfg = self.cfg
        if not cfg.recurrent:
            return 1
        if self.training and cfg.randomize_r:
            span = cfg.r_max - cfg.r_min + 1
            offset = int(torch.randint(0, span, (1,), generator=generator).item())
            return cfg.r_min + offset
        return cfg.resolved_train_r()

    def forward(
        self,
        input_ids: torch.Tensor,
        slot_type_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,   # bool [B,T], True = real token
        r: int | None = None,
        deep_supervision: bool = False,
        generator: torch.Generator | None = None,
    ) -> EncoderOutput:
        cfg = self.cfg
        slot_arg = slot_type_ids if cfg.typed_attention == "per_slot_kqv" else None

        h = self.embedding(input_ids, slot_type_ids)
        for layer in self.prelude:
            h = layer(h, key_padding_mask=attention_mask, slot_type_ids=slot_arg)

        if not cfg.recurrent:
            s = self._run_core(h, attention_mask, slot_arg)
            seq = self.coda_norm(s)
            return EncoderOutput([seq], [_pool(seq, attention_mask, cfg.pool)], r_used=1)

        r_used = r if r is not None else self._sample_r(generator)

        # s_0: Gaussian noise, same shape as h
        s = torch.randn_like(h) * cfg.state_init_std

        # truncated BPTT: first (r - k) iterations run without grad
        k = cfg.truncated_bptt_k if self.training else r_used
        n_nograd = max(0, r_used - k)

        seq_states: list[torch.Tensor] = []
        pooled_states: list[torch.Tensor] = []
        for i in range(r_used):
            grad_ctx = torch.no_grad() if i < n_nograd else contextlib.nullcontext()
            with grad_ctx:
                inject = (s + h) if cfg.input_injection else s
                s = self._run_core(inject, attention_mask, slot_arg)
            read_out = deep_supervision or (i == r_used - 1)
            if read_out:
                seq = self.coda_norm(s)
                seq_states.append(seq)
                pooled_states.append(_pool(seq, attention_mask, cfg.pool))

        return EncoderOutput(seq_states, pooled_states, r_used=r_used)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
