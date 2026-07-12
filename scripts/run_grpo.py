#!/usr/bin/env python
"""Stage 4 GRPO from a saved SFT checkpoint (standalone).

Usually you just let `run_sft.py` run Stage 4 after SFT. Use this to re-run RL
alone — e.g. sweep the reward shape or KL coefficient against a fixed SFT model.

    python scripts/run_grpo.py --config configs/run_sft75_rl25.yaml \
        --checkpoint runs/sft75_rl25/model.pt --device cpu
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import torch

from coffee_transformer.eval.ttc import ttc_sweep
from coffee_transformer.training.builder import (
    build_model,
    build_pools,
    build_tokenizer,
    load_examples,
    make_dataset,
    make_loader,
)
from coffee_transformer.training.grpo import GRPOTrainer
from coffee_transformer.utils.config import load_run_config
from coffee_transformer.utils.device import get_device
from coffee_transformer.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--reward", default=None, help="override grpo.reward (tolerance|ranking)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_run_config(args.config)
    if args.device:
        cfg.train.device = args.device
    if args.reward:
        cfg.grpo.reward = args.reward
    cfg.grpo.enabled = True

    generator = set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)

    examples = load_examples(cfg)
    tokenizer = build_tokenizer(examples)
    model = build_model(cfg, tokenizer)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"loaded checkpoint {args.checkpoint}")

    pools = build_pools(cfg, examples)
    if not pools.rl:
        raise SystemExit("RL pool is empty (rl_fraction == 0); nothing to do.")
    rl_loader = make_loader(cfg, make_dataset(cfg, tokenizer, pools.rl, True), tokenizer, cfg.grpo.batch_size, True)
    test_loader = make_loader(cfg, make_dataset(cfg, tokenizer, pools.test, False), tokenizer, cfg.train.batch_size, False)

    result = GRPOTrainer(model, cfg.grpo, device, generator).train(rl_loader)
    print(f"grpo done: {result}")

    print("\n== post-GRPO test-time-compute sweep ==")
    for r, m in ttc_sweep(model, test_loader, cfg.train.eval_r_values, device).items():
        print(f"  r={r:>2}: R2={m['r2']:.3f} MAE={m['mae']:.2f} spearman={m['spearman']:.3f}")


if __name__ == "__main__":
    main()
