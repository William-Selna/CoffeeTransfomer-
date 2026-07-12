"""Encoder checkpoints shared between pretraining and SFT.

A pretrained encoder is only useful downstream if the SFT model rebuilds the
*exact* same architecture and reuses the *same* tokenizer/vocab (Section 6:
pooling and vocab chosen once, kept identical across all stages). So the
checkpoint carries the ModelConfig alongside the encoder weights, and the
tokenizer is saved next to it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from ..data.tokenizer import SmilesTokenizer
from ..models.config import ModelConfig
from ..models.heads import MLMModel, YieldModel


def save_pretrained_encoder(
    out_dir: str | Path,
    model_cfg: ModelConfig,
    mlm_model: MLMModel,
    tokenizer: SmilesTokenizer,
    val_loss: float | None = None,
    filename: str = "encoder.pt",
    step: int | None = None,
) -> Path:
    """Save an encoder checkpoint (+ ModelConfig) and the shared tokenizer.

    `filename` lets callers keep a rolling `encoder_latest.pt` alongside the
    canonical `encoder.pt` (best-by-val) for crash recovery.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # unwrap a possible torch.compile wrapper
    encoder = getattr(mlm_model, "_orig_mod", mlm_model).encoder
    ckpt = {
        "model_config": dataclasses.asdict(model_cfg),
        "encoder_state": encoder.state_dict(),
        "val_loss": val_loss,
        "step": step,
    }
    torch.save(ckpt, out_dir / filename)
    tokenizer.save(out_dir / "tokenizer.json")
    return out_dir / filename


def load_encoder_into_yield_model(
    ckpt_path: str | Path,
    yield_model: YieldModel,
    strict: bool = True,
) -> YieldModel:
    """Copy pretrained encoder weights into an existing (SFT) YieldModel.

    The YieldModel must already be built with the checkpoint's ModelConfig
    (same dims) — `build_model_from_checkpoint` does that for you.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    yield_model.encoder.load_state_dict(ckpt["encoder_state"], strict=strict)
    return yield_model


def model_config_from_checkpoint(ckpt_path: str | Path) -> ModelConfig:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ModelConfig(**ckpt["model_config"])
