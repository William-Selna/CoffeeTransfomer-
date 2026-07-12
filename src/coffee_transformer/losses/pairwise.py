"""Pairwise-difference supervision (Section 5).

For sampled pairs of reactions, predict the sign/magnitude of their yield
difference. This multiplies effective supervision and partially cancels
systematic measurement error. We sample pairs per batch (never enumerate all).

NOTE: the design's stronger variant restricts pairs to reactions differing in
exactly one component. That selection belongs in the data pipeline (emit a
`pair_index` alongside each example); here we sample random within-batch pairs,
which is the cheap default. See `docs/PIPELINE.md`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_difference_loss(
    pred_yields: torch.Tensor,   # [B] expected yields from the histogram head
    true_yields: torch.Tensor,  # [B]
    num_pairs: int,
    generator: torch.Generator | None = None,
    margin_weight: float = 0.0,
) -> torch.Tensor:
    b = pred_yields.size(0)
    if b < 2:
        return pred_yields.new_zeros(())
    num_pairs = min(num_pairs, b * (b - 1))
    # Sample on CPU with the (CPU) generator, then move — keeps this correct
    # whether the model runs on CPU or CUDA.
    i = torch.randint(0, b, (num_pairs,), generator=generator).to(pred_yields.device)
    j = torch.randint(0, b, (num_pairs,), generator=generator).to(pred_yields.device)
    valid = i != j
    i, j = i[valid], j[valid]
    if i.numel() == 0:
        return pred_yields.new_zeros(())

    pred_diff = pred_yields[i] - pred_yields[j]
    true_diff = true_yields[i] - true_yields[j]
    loss = F.mse_loss(pred_diff, true_diff)

    if margin_weight > 0:
        # ranking hinge on the sign of the difference
        sign = torch.sign(true_diff)
        loss = loss + margin_weight * F.relu(-sign * pred_diff).mean()
    return loss
