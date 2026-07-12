"""Combined yield objective with per-term flags and deep supervision.

Every term is toggleable by config flag so the loss-term ablation (Section 7,
"the knockout table is the second contribution") is a pure config sweep. Fixed
lambda weights are the cheap default (Section 5); swap in annealing or
uncertainty weighting later if needed.

Deep supervision (Section 5): when the encoder reads out every iteration, the
base distributional loss is applied to each draft and combined with weights
that ramp toward the final iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..models.heads import YieldOutput
from .distributional import multi_scale_ce, soft_cross_entropy, two_hot_targets
from .moments import moment_matching_loss
from .pairwise import pairwise_difference_loss


@dataclass
class LossConfig:
    lambda_two_hot: float = 1.0

    use_multi_scale: bool = True
    lambda_multi_scale: float = 0.3
    multi_scale_bins: list[int] = field(default_factory=lambda: [10, 5, 2])

    use_moments: bool = True
    lambda_moment: float = 0.3
    match_variance: bool = False

    deep_supervision: bool = True
    deep_supervision_scheme: str = "linear"  # "uniform" | "linear" | "last"

    use_pairwise: bool = True
    lambda_pairwise: float = 0.3
    pairs_per_batch: int = 64
    pairwise_margin_weight: float = 0.0


def deep_supervision_weights(n: int, scheme: str, device=None) -> torch.Tensor:
    if n == 1 or scheme == "last":
        w = torch.zeros(n, device=device)
        w[-1] = 1.0
        return w
    if scheme == "uniform":
        w = torch.ones(n, device=device)
    elif scheme == "linear":
        w = torch.arange(1, n + 1, device=device, dtype=torch.float)
    else:
        raise ValueError(f"unknown deep-supervision scheme: {scheme}")
    return w / w.sum()


class YieldLoss(nn.Module):
    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def _base_loss(self, logits, yields, centers, y_min, y_max, target):
        cfg = self.cfg
        loss = cfg.lambda_two_hot * soft_cross_entropy(logits, target)
        parts = {"two_hot": loss.detach()}
        if cfg.use_multi_scale:
            ms = cfg.lambda_multi_scale * multi_scale_ce(logits, target, cfg.multi_scale_bins)
            loss = loss + ms
            parts["multi_scale"] = ms.detach()
        if cfg.use_moments:
            mom = cfg.lambda_moment * moment_matching_loss(
                logits, yields, centers, match_variance=cfg.match_variance
            )
            loss = loss + mom
            parts["moment"] = mom.detach()
        return loss, parts

    def forward(
        self,
        output: YieldOutput,
        yields: torch.Tensor,
        centers: torch.Tensor,
        y_min: float,
        y_max: float,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        num_bins = output.logits.size(-1)
        target = two_hot_targets(yields, num_bins, y_min, y_max)

        iters = output.iteration_logits
        weights = deep_supervision_weights(len(iters), cfg.deep_supervision_scheme, yields.device)

        total = yields.new_zeros(())
        components: dict[str, torch.Tensor] = {}
        for w, logits in zip(weights, iters):
            base, parts = self._base_loss(logits, yields, centers, y_min, y_max, target)
            total = total + w * base
            for k, v in parts.items():
                components[k] = components.get(k, 0.0) + w * v

        if cfg.use_pairwise:
            probs = torch.softmax(output.logits, dim=-1)
            pred_yields = (probs * centers).sum(dim=-1)
            pw = cfg.lambda_pairwise * pairwise_difference_loss(
                pred_yields, yields, cfg.pairs_per_batch, generator, cfg.pairwise_margin_weight
            )
            total = total + pw
            components["pairwise"] = pw.detach()

        components["total"] = total.detach()
        return total, components
