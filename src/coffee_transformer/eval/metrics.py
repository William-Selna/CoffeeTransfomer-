"""Regression + ranking metrics.

R^2 is the headline the field reports (Section 2: RF baseline R^2 ~= 0.92 on
random splits). Ranking metrics (Spearman) matter for the RL stage, which
targets "correct yield ordering — what chemists actually want" (Section 6).
"""

from __future__ import annotations

import torch


def _ranks(x: torch.Tensor) -> torch.Tensor:
    order = x.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float)
    return ranks


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2).clamp_min(1e-8)
    return float(1.0 - ss_res / ss_tot)


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)))


def spearman(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() < 2:
        return float("nan")
    rp, rt = _ranks(pred), _ranks(target)
    rp = rp - rp.mean()
    rt = rt - rt.mean()
    denom = (rp.norm() * rt.norm()).clamp_min(1e-8)
    return float((rp * rt).sum() / denom)


def regression_report(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "r2": r2_score(pred, target),
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
        "spearman": spearman(pred, target),
    }
