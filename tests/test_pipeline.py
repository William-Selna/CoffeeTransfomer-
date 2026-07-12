import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import torch

from coffee_transformer.losses.combined import YieldLoss
from coffee_transformer.training.builder import (
    build_model,
    build_pools,
    build_tokenizer,
    load_examples,
    make_dataset,
    make_loader,
)
from coffee_transformer.training.grpo import GRPOTrainer
from coffee_transformer.training.sft import SFTTrainer
from coffee_transformer.utils.config import RunConfig
from coffee_transformer.utils.seed import set_seed


def _tiny_cfg():
    cfg = RunConfig()
    cfg.data.synthetic = True
    cfg.data.synthetic_n = 120
    cfg.data.sft_fraction = 0.75
    cfg.data.rl_fraction = 0.25
    cfg.model.d_model = 32
    cfg.model.n_heads = 4
    cfg.model.d_ff = 64
    cfg.model.prelude_layers = 1
    cfg.model.core_layers = 2
    cfg.train.epochs = 1
    cfg.train.batch_size = 16
    cfg.train.linear_probe_steps = 2
    cfg.train.device = "cpu"
    cfg.grpo.epochs = 1
    cfg.grpo.batch_size = 16
    return cfg


def test_split_fractions_partition_pool():
    cfg = _tiny_cfg()
    examples = load_examples(cfg)
    pools = build_pools(cfg, examples)
    assert len(pools.sft) > 0 and len(pools.rl) > 0 and len(pools.test) > 0
    # test held out, sft+rl partition the remainder
    assert len(pools.sft) + len(pools.rl) + len(pools.test) == len(examples)


def test_uspto_crude_4bin_sft_runs():
    # crude USPTO SFT with 25%-wide (4-bin) coarse target, from scratch
    cfg = _tiny_cfg()
    cfg.data.dataset = "USPTO"
    cfg.data.synthetic_n = 160
    cfg.model.num_bins = 4
    cfg.loss.multi_scale_bins = [2]      # must divide num_bins=4
    cfg.loss.use_pairwise = False
    gen = set_seed(0)
    device = torch.device("cpu")
    examples = load_examples(cfg)
    assert examples[0].dataset == "USPTO"
    tok = build_tokenizer(examples)
    model = build_model(cfg, tok)
    assert model.head.centers.numel() == 4    # 4 coarse chunks
    pools = build_pools(cfg, examples)
    sft_loader = make_loader(cfg, make_dataset(cfg, tok, pools.sft, True), tok, 16, True)
    test_loader = make_loader(cfg, make_dataset(cfg, tok, pools.test, False), tok, 16, False)
    res = SFTTrainer(model, YieldLoss(cfg.loss), cfg.train, device, gen).train(sft_loader, test_loader)
    assert res.steps > 0
    assert torch.isfinite(torch.tensor(res.ttc[cfg.train.eval_r_values[0]]["r2"]))


def test_sft_then_grpo_runs_and_is_finite():
    cfg = _tiny_cfg()
    gen = set_seed(0)
    device = torch.device("cpu")
    examples = load_examples(cfg)
    tok = build_tokenizer(examples)
    model = build_model(cfg, tok)
    pools = build_pools(cfg, examples)

    sft_loader = make_loader(cfg, make_dataset(cfg, tok, pools.sft, True), tok, 16, True)
    test_loader = make_loader(cfg, make_dataset(cfg, tok, pools.test, False), tok, 16, False)

    sft = SFTTrainer(model, YieldLoss(cfg.loss), cfg.train, device, gen)
    res = sft.train(sft_loader, test_loader)
    assert res.steps > 0
    assert all(torch.isfinite(torch.tensor(m["r2"])) for m in res.ttc.values())

    rl_loader = make_loader(cfg, make_dataset(cfg, tok, pools.rl, True), tok, 16, True)
    grpo = GRPOTrainer(model, cfg.grpo, device, gen)
    gres = grpo.train(rl_loader)
    assert gres.steps > 0
