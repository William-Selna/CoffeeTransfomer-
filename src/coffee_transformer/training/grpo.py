"""Stage 4 — RL with GRPO on a held-out pool (Section 6).

Group-Relative Policy Optimization: no value network. For each reaction sample
k predictions from the histogram head (a categorical over yield bins), score
them, and use the group-relative advantage (reward minus the per-reaction group
mean, over the group std) as the policy-gradient weight.

Two reward shapes (Section 6):
  * "tolerance" — smooth partial credit exp(-|pred - true| / tolerance);
  * "ranking"   — within-batch order agreement (non-differentiable, genuinely
    RL-shaped; "what chemists actually want").

Caveat kept in view (Section 6): for pure regression with labels available,
reward-for-correctness RL is a higher-variance supervised signal — this stage
is here to test the "SFT memorizes, RL generalizes" hypothesis at tiny scale,
especially on the OOD splits, not to beat SFT on random splits.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from ..models.heads import YieldModel
from ..utils.config import GRPOConfig


def tolerance_reward(pred_vals: torch.Tensor, true: torch.Tensor, tol: float) -> torch.Tensor:
    # pred_vals [G, B], true [B] -> [G, B]
    return torch.exp(-(pred_vals - true.unsqueeze(0)).abs() / tol)


def ranking_reward(pred_vals: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Reward each sample by how well its predicted value orders against the
    true yields of the rest of the batch."""
    order_true = torch.sign(true.unsqueeze(1) - true.unsqueeze(0))       # [B, B]
    pred_order = torch.sign(pred_vals.unsqueeze(2) - true.view(1, 1, -1))  # [G, B, B]
    agree = (pred_order == order_true.unsqueeze(0)).float()               # [G, B, B]
    # ignore self-comparison (diagonal)
    mask = 1.0 - torch.eye(true.size(0), device=true.device).unsqueeze(0)
    return (agree * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)


@dataclass
class GRPOResult:
    steps: int
    final_loss: float
    mean_reward: float


class GRPOTrainer:
    def __init__(
        self,
        model: YieldModel,
        cfg: GRPOConfig,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.generator = generator
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.reference = None
        if cfg.kl_coef > 0:
            self.reference = copy.deepcopy(model).to(device)
            self.reference.eval()
            for p in self.reference.parameters():
                p.requires_grad_(False)

    def _reward(self, pred_vals: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if self.cfg.reward == "tolerance":
            return tolerance_reward(pred_vals, true, self.cfg.tolerance)
        if self.cfg.reward == "ranking":
            return ranking_reward(pred_vals, true)
        raise ValueError(f"unknown reward: {self.cfg.reward}")

    def _step(self, batch) -> tuple[float, float]:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        slot_type_ids = batch["slot_type_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        true = batch["yield_value"].to(self.device)

        out = self.model(
            input_ids, slot_type_ids, attention_mask=attention_mask, r=self.cfg.sample_r
        )
        logits = out.logits                              # [B, K]
        logp = F.log_softmax(logits, dim=-1)
        dist = Categorical(logits=logits)

        g = self.cfg.group_size
        samples = dist.sample((g,))                      # [G, B]
        centers = self.model.head.centers                 # [K]
        pred_vals = centers[samples]                      # [G, B]

        rewards = self._reward(pred_vals, true)           # [G, B]
        adv = (rewards - rewards.mean(0, keepdim=True)) / (rewards.std(0, keepdim=True) + 1e-6)

        sample_logp = logp.unsqueeze(0).expand(g, -1, -1).gather(-1, samples.unsqueeze(-1)).squeeze(-1)
        loss = -(adv.detach() * sample_logp).mean()

        if self.reference is not None:
            with torch.no_grad():
                ref_logits = self.reference(
                    input_ids, slot_type_ids, attention_mask=attention_mask, r=self.cfg.sample_r
                ).logits
                ref_logp = F.log_softmax(ref_logits, dim=-1)
            kl = (logp.exp() * (logp - ref_logp)).sum(-1).mean()
            loss = loss + self.cfg.kl_coef * kl

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.detach()), float(rewards.mean().detach())

    def train(self, rl_loader: DataLoader) -> GRPOResult:
        step = 0
        last_loss, last_reward = float("nan"), float("nan")
        for epoch in range(self.cfg.epochs):
            for batch in rl_loader:
                last_loss, last_reward = self._step(batch)
                if step % 20 == 0:
                    print(f"[grpo] step {step} loss {last_loss:.4f} reward {last_reward:.4f}")
                step += 1
        return GRPOResult(steps=step, final_loss=last_loss, mean_reward=last_reward)
