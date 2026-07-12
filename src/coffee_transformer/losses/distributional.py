"""Distributional yield supervision (Section 5).

The scalar yield is turned into a soft histogram target ("two-hot": a true
yield of 63% puts partial mass on the 60 and 65 bins). Cross-entropy over bins
gives denser gradients than MSE and calibrated uncertainty for free.

`multi_scale_ce` additionally pools the fine bins into coarser blocks and
applies the loss at each scale — strong low-variance gradients early, fine
granularity later.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def two_hot_targets(
    yields: torch.Tensor,
    num_bins: int,
    y_min: float,
    y_max: float,
) -> torch.Tensor:
    """Soft two-hot histogram targets, shape [B, num_bins], each row sums to 1.

    Mass is split linearly between the two bin centers bracketing each yield.
    Values outside [y_min, y_max] collapse onto the nearest edge bin.
    """
    width = (y_max - y_min) / num_bins
    pos = (yields - y_min) / width - 0.5           # continuous bin-center index
    lower_f = torch.floor(pos)
    frac = (pos - lower_f).clamp(0.0, 1.0)         # weight going to the upper bin

    lower = lower_f.long().clamp(0, num_bins - 1)
    upper = (lower_f.long() + 1).clamp(0, num_bins - 1)

    target = torch.zeros(yields.size(0), num_bins, device=yields.device, dtype=torch.float)
    target.scatter_add_(1, lower.unsqueeze(1), (1.0 - frac).unsqueeze(1))
    target.scatter_add_(1, upper.unsqueeze(1), frac.unsqueeze(1))
    # renormalize (rows where lower == upper on an edge already sum to 1)
    return target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)


def soft_cross_entropy(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    """CE with soft targets: -sum_k target_k * log_softmax(logits)_k, mean over batch."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1).mean()


def multi_scale_ce(
    logits: torch.Tensor,
    target_dist: torch.Tensor,
    scales: list[int],
) -> torch.Tensor:
    """Sum of soft-CE at each coarseness. Each scale is a coarse bin count that
    must divide the fine bin count; probabilities/targets are pooled by summing
    adjacent groups (Section 5, multi-scale histogram loss)."""
    num_bins = logits.size(-1)
    probs = F.softmax(logits, dim=-1)
    total = logits.new_zeros(())
    for coarse in scales:
        if num_bins % coarse != 0:
            raise ValueError(f"scale {coarse} does not divide num_bins {num_bins}")
        group = num_bins // coarse
        p = probs.view(probs.size(0), coarse, group).sum(dim=-1)
        t = target_dist.view(target_dist.size(0), coarse, group).sum(dim=-1)
        total = total + -(t * torch.log(p.clamp_min(1e-8))).sum(dim=-1).mean()
    return total
