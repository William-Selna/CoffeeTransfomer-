"""Test-time-compute sweep (Section 3 / Section 7).

Train at r=4, evaluate at r in {1, 2, 4, 8, 16}. The accuracy-vs-iterations
curve on a scientific task at ~5M params is a headline figure. This helper runs
a trained YieldModel over a dataloader at each r and returns per-r metrics.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..models.heads import YieldModel
from .metrics import regression_report


@torch.no_grad()
def evaluate_at_r(
    model: YieldModel,
    loader: DataLoader,
    r: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds, targets = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        slot_type_ids = batch["slot_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids, slot_type_ids, attention_mask=attention_mask, r=r)
        preds.append(model.head.expected_yield(out.logits).cpu())
        targets.append(batch["yield_value"])
    pred = torch.cat(preds)
    target = torch.cat(targets)
    return regression_report(pred, target)


def ttc_sweep(
    model: YieldModel,
    loader: DataLoader,
    r_values: list[int],
    device: torch.device,
) -> dict[int, dict[str, float]]:
    return {r: evaluate_at_r(model, loader, r, device) for r in r_values}
