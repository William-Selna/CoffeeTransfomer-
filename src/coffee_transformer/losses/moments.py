"""Moment matching (Section 5).

Auxiliary MSE on the mean (optionally variance) of the predicted histogram —
keeps the distribution's center of mass honest and directly ties the
distributional head to the regression target.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def moment_matching_loss(
    logits: torch.Tensor,
    yields: torch.Tensor,
    centers: torch.Tensor,
    match_variance: bool = False,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    pred_mean = (probs * centers).sum(dim=-1)
    loss = F.mse_loss(pred_mean, yields)
    if match_variance:
        pred_var = (probs * (centers - pred_mean.unsqueeze(-1)) ** 2).sum(dim=-1)
        # target variance is unknown per-sample; regularize predicted spread toward
        # the squared error so confident-but-wrong predictions are penalized.
        target_var = (pred_mean.detach() - yields) ** 2
        loss = loss + F.mse_loss(pred_var, target_var)
    return loss
