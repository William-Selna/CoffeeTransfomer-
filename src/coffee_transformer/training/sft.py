"""Stage 3 — supervised fine-tuning on HTE yields.

Default recipe (Section 6):
  1. linear probe — freeze the encoder, train only the fresh histogram head for
     a few hundred steps (guards pretrained representations from the random
     head's initial gradients);
  2. unfreeze — continue with the encoder LR ~10x lower than the head LR.

Loss is the full distributional objective (Section 5), deep-supervised across
recurrence iterations when `loss.deep_supervision` is set. Works from a random
encoder (baseline column) or a pretrained one (transfer column) unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ..eval.ttc import ttc_sweep
from ..losses.combined import YieldLoss
from ..models.heads import YieldModel
from ..utils.config import TrainConfig


@dataclass
class SFTResult:
    steps: int
    final_train_loss: float
    ttc: dict[int, dict[str, float]]


def _lr_lambda(step: int, warmup: int, total: int):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class SFTTrainer:
    def __init__(
        self,
        model: YieldModel,
        loss_fn: YieldLoss,
        cfg: TrainConfig,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device
        self.generator = generator
        self.deep_supervision = loss_fn.cfg.deep_supervision

    # -- optimizer construction -------------------------------------------
    def _head_params(self):
        return list(self.model.head.parameters())

    def _encoder_params(self):
        return list(self.model.encoder.parameters())

    def _build_optimizer(self, include_encoder: bool):
        groups = [{"params": self._head_params(), "lr": self.cfg.head_lr}]
        if include_encoder:
            groups.append(
                {
                    "params": self._encoder_params(),
                    "lr": self.cfg.head_lr * self.cfg.encoder_lr_scale,
                }
            )
        return torch.optim.AdamW(groups, weight_decay=self.cfg.weight_decay)

    # -- one forward/backward ---------------------------------------------
    def _step(self, batch, optimizer, scheduler) -> float:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        slot_type_ids = batch["slot_type_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        yields = batch["yield_value"].to(self.device)

        out = self.model(
            input_ids,
            slot_type_ids,
            attention_mask=attention_mask,
            deep_supervision=self.deep_supervision,
            generator=self.generator,
        )
        loss, _ = self.loss_fn(
            out,
            yields,
            self.model.head.centers,
            self.model.cfg.yield_min,
            self.model.cfg.yield_max,
            generator=self.generator,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        return float(loss.detach())

    # -- full run ----------------------------------------------------------
    def train(self, train_loader: DataLoader, test_loader: DataLoader | None = None) -> SFTResult:
        steps_per_epoch = max(1, len(train_loader))
        total_steps = self.cfg.epochs * steps_per_epoch
        probe_steps = min(self.cfg.linear_probe_steps, total_steps)

        # Phase 1: linear probe (encoder frozen)
        for p in self._encoder_params():
            p.requires_grad_(False)
        optimizer = self._build_optimizer(include_encoder=False)
        warmup = int(self.cfg.warmup_frac * total_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: _lr_lambda(s, warmup, total_steps)
        )

        global_step = 0
        last_loss = float("nan")
        unfrozen = probe_steps == 0
        if unfrozen:  # no probe phase
            for p in self._encoder_params():
                p.requires_grad_(True)
            optimizer = self._build_optimizer(include_encoder=True)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda s: _lr_lambda(s, warmup, total_steps)
            )

        for epoch in range(self.cfg.epochs):
            for batch in train_loader:
                if not unfrozen and global_step >= probe_steps:
                    # Phase 2: unfreeze encoder, rebuild optimizer/scheduler
                    for p in self._encoder_params():
                        p.requires_grad_(True)
                    optimizer = self._build_optimizer(include_encoder=True)
                    scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer,
                        lambda s, base=global_step: _lr_lambda(base + s, warmup, total_steps),
                    )
                    unfrozen = True

                last_loss = self._step(batch, optimizer, scheduler)
                if global_step % self.cfg.log_every == 0:
                    phase = "probe" if not unfrozen else "ft"
                    print(f"[sft:{phase}] step {global_step}/{total_steps} loss {last_loss:.4f}")
                global_step += 1

        ttc: dict[int, dict[str, float]] = {}
        if test_loader is not None:
            ttc = ttc_sweep(self.model, test_loader, self.cfg.eval_r_values, self.device)

        return SFTResult(steps=global_step, final_train_loss=last_loss, ttc=ttc)
