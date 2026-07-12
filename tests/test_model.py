import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import torch

from coffee_transformer.models.config import ModelConfig
from coffee_transformer.models.heads import YieldModel
from coffee_transformer.models.recurrent_depth import count_parameters


def _cfg(**kw):
    base = dict(vocab_size=60, num_slot_types=12, d_model=32, n_heads=4, d_ff=64,
                prelude_layers=1, core_layers=2, num_bins=20, max_position=64)
    base.update(kw)
    return ModelConfig(**base)


def _batch(b=4, t=10, cfg=None):
    input_ids = torch.randint(0, cfg.vocab_size, (b, t))
    slot_type_ids = torch.randint(0, cfg.num_slot_types, (b, t))
    attention_mask = torch.ones(b, t, dtype=torch.bool)
    attention_mask[:, -2:] = False
    return input_ids, slot_type_ids, attention_mask


def test_forward_shapes():
    cfg = _cfg()
    model = YieldModel(cfg).eval()
    ii, si, am = _batch(cfg=cfg)
    out = model(ii, si, attention_mask=am, r=4)
    assert out.logits.shape == (4, cfg.num_bins)
    assert out.r_used == 4


def test_deep_supervision_reads_every_iteration():
    cfg = _cfg()
    model = YieldModel(cfg).eval()
    ii, si, am = _batch(cfg=cfg)
    out = model(ii, si, attention_mask=am, r=5, deep_supervision=True)
    assert len(out.iteration_logits) == 5
    out2 = model(ii, si, attention_mask=am, r=5, deep_supervision=False)
    assert len(out2.iteration_logits) == 1


def test_vanilla_mode_runs_once():
    cfg = _cfg(recurrent=False)
    model = YieldModel(cfg).eval()
    ii, si, am = _batch(cfg=cfg)
    out = model(ii, si, attention_mask=am, deep_supervision=True)
    assert out.r_used == 1
    assert len(out.iteration_logits) == 1


def test_typed_attention_ablation_runs():
    cfg = _cfg(typed_attention="per_slot_kqv")
    model = YieldModel(cfg).eval()
    ii, si, am = _batch(cfg=cfg)
    out = model(ii, si, attention_mask=am, r=2)
    assert out.logits.shape == (4, cfg.num_bins)


def test_anchor_param_count_small():
    # the ~5M-param anchor config should land in single-digit millions
    cfg = ModelConfig(vocab_size=300, num_slot_types=12)
    model = YieldModel(cfg)
    assert count_parameters(model) < 15_000_000


def test_swiglu_forward_and_param_match():
    # gated FFN should run and stay parameter-matched to the pointwise FFN
    gelu = YieldModel(_cfg(activation="gelu", d_ff=192))
    swiglu = YieldModel(_cfg(activation="swiglu", d_ff=192))
    ii, si, am = _batch(cfg=_cfg())
    out = swiglu.eval()(ii, si, attention_mask=am, r=2)
    assert out.logits.shape == (4, 20)
    # 2/3 * 192 == 128 -> 3*d*128 == 2*d*192: FFN projection params match exactly;
    # only the handful of bias terms differ, so totals are within a fraction of 1%.
    pg, ps = count_parameters(gelu), count_parameters(swiglu)
    assert abs(pg - ps) / pg < 0.01


def test_unknown_activation_raises():
    import pytest
    with pytest.raises(ValueError):
        YieldModel(_cfg(activation="banana"))


def test_backward_flows_to_encoder():
    cfg = _cfg()
    model = YieldModel(cfg).train()
    ii, si, am = _batch(cfg=cfg)
    out = model(ii, si, attention_mask=am, r=4, deep_supervision=True)
    out.logits.sum().backward()
    grads = [p.grad is not None for p in model.encoder.parameters() if p.requires_grad]
    assert any(grads)
