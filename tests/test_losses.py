import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import torch

from coffee_transformer.losses.combined import LossConfig, YieldLoss
from coffee_transformer.losses.distributional import multi_scale_ce, two_hot_targets
from coffee_transformer.models.heads import YieldOutput, bin_centers
from coffee_transformer.models.config import ModelConfig


def test_two_hot_rows_sum_to_one():
    y = torch.tensor([0.0, 12.5, 63.0, 100.0])
    t = two_hot_targets(y, num_bins=20, y_min=0.0, y_max=100.0)
    assert torch.allclose(t.sum(dim=1), torch.ones(4), atol=1e-5)


def test_two_hot_places_mass_on_bracketing_bins():
    # 63% with 20 bins of width 5 -> centers at 62.5 (bin 12) and 67.5 (bin 13)
    y = torch.tensor([63.0])
    t = two_hot_targets(y, num_bins=20, y_min=0.0, y_max=100.0)[0]
    assert t[12] > 0 and t[13] > 0
    assert torch.argmax(t).item() in (12, 13)


def test_multi_scale_requires_divisors():
    logits = torch.randn(3, 20)
    target = two_hot_targets(torch.tensor([10.0, 50.0, 90.0]), 20, 0.0, 100.0)
    val = multi_scale_ce(logits, target, [10, 5, 2])
    assert torch.isfinite(val)
    try:
        multi_scale_ce(logits, target, [7])
        assert False, "expected ValueError for non-divisor scale"
    except ValueError:
        pass


def test_combined_loss_backprops():
    cfg = ModelConfig(vocab_size=60, num_slot_types=12, d_model=16, num_bins=20)
    centers = bin_centers(cfg)
    logits = torch.randn(8, 20, requires_grad=True)
    out = YieldOutput(logits=logits, iteration_logits=[logits, logits], r_used=2)
    y = torch.rand(8) * 100
    loss_fn = YieldLoss(LossConfig())
    total, parts = loss_fn(out, y, centers, cfg.yield_min, cfg.yield_max)
    total.backward()
    assert torch.isfinite(total)
    assert logits.grad is not None
    assert "two_hot" in parts and "pairwise" in parts
