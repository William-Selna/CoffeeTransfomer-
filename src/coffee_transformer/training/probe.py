"""Linear-probe scoring — the metric for picking the better pretrained encoder.

This is the design's "probe-only column" (Section 6): freeze the pretrained
encoder, train only a fresh head, and read off HTE R². It measures how much
yield-relevant structure the encoder learned for free — exactly the right basis
for choosing between two pretraining runs before committing to the 4-fork SFT.

Implemented by reusing `SFTTrainer` with `linear_probe_steps` set beyond the
total step count, so the encoder never unfreezes.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from ..losses.combined import LossConfig, YieldLoss
from ..models.heads import YieldModel
from ..utils.config import TrainConfig
from .sft import SFTTrainer


def linear_probe_score(
    model: YieldModel,
    sft_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    generator: torch.Generator | None = None,
    probe_epochs: int = 3,
    eval_r: int = 4,
) -> float:
    """Return HTE R² of a frozen-encoder linear probe at recurrence `eval_r`."""
    cfg = TrainConfig(
        epochs=probe_epochs,
        linear_probe_steps=10**9,   # never unfreeze -> pure probe
        eval_r_values=[eval_r],
        device=device.type,
        log_every=10**9,
    )
    loss_fn = YieldLoss(replace(LossConfig(), deep_supervision=False))
    trainer = SFTTrainer(model, loss_fn, cfg, device, generator)
    result = trainer.train(sft_loader, test_loader)
    return result.ttc[eval_r]["r2"]
