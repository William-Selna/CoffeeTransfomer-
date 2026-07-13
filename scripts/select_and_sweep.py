#!/usr/bin/env python
"""Pick the better pretrained encoder, then run the four SFT/RL forks on it.

The full plan end to end:
  1. scripts/prepare_corpus.py         # once: tokenizer + cleaned/packed corpus
  2. scripts/pretrain.py x2            # the 5M and 10M candidates (parallel GPUs)
  3. scripts/select_and_sweep.py       # <-- this: probe both, fork the winner

Selection metric = HTE linear-probe R² (the design's probe-only column): freeze
each encoder, train only a fresh head, compare. The winner is forked into the
four SFT/RL configs (each loading it as the transfer encoder).

    python scripts/select_and_sweep.py          # defaults to the 2x2 encoders

    # offline smoke on CPU
    python scripts/select_and_sweep.py \
        --encoders runs/pretrain_gelu_s0 runs/pretrain_swiglu_s0 \
        --device cpu --epochs 1 --probe-epochs 1 --batch-size 32
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import torch

from coffee_transformer.training.builder import (
    build_pools,
    load_examples,
    load_pretrained_bundle,
    make_dataset,
    make_loader,
)
from coffee_transformer.training.checkpoint import model_config_from_checkpoint
from coffee_transformer.training.probe import linear_probe_score
from coffee_transformer.utils.config import load_run_config
from coffee_transformer.utils.device import get_device
from coffee_transformer.utils.seed import set_seed

FORKS = [
    "configs/run_sft100_rl0.yaml",
    "configs/run_sft90_rl10.yaml",
    "configs/run_sft75_rl25.yaml",
    "configs/run_sft50_rl50.yaml",
]

# the 2x2 pretraining grid
ENCODERS = [
    "runs/pretrain_gelu_s0",
    "runs/pretrain_gelu_s1",
    "runs/pretrain_swiglu_s0",
    "runs/pretrain_swiglu_s1",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", default=ENCODERS, help="pretrained encoder dirs")
    p.add_argument("--data-config", default="configs/run_sft75_rl25.yaml",
                   help="HTE data settings used for the probe")
    p.add_argument("--forks", nargs="+", default=FORKS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None, help="override SFT epochs in the forks")
    p.add_argument("--batch-size", type=int, default=None, help="override batch size (probe + forks)")
    p.add_argument("--probe-epochs", type=int, default=3)
    p.add_argument("--probe-seeds", type=int, nargs="+", default=[0, 1],
                   help="average the probe R2 over these seeds (fresh head each) for a stable pick")
    p.add_argument("--eval-r", type=int, default=4)
    return p.parse_args()


def probe_encoder(enc_dir, data_cfg, device, probe_epochs, eval_r, seeds):
    """Average HTE linear-probe R2 over several seeds (fresh head each), so the
    encoder pick isn't decided by probe-init noise (gap fix)."""
    examples = load_examples(data_cfg)
    pools = build_pools(data_cfg, examples)  # pools fixed by data_cfg.train.seed
    scores = []
    for s in seeds:
        gen = set_seed(s)
        model, tokenizer = load_pretrained_bundle(enc_dir)  # fresh head per seed
        sft_loader = make_loader(data_cfg, make_dataset(data_cfg, tokenizer, pools.sft, True),
                                 tokenizer, data_cfg.train.batch_size, True)
        test_loader = make_loader(data_cfg, make_dataset(data_cfg, tokenizer, pools.test, False),
                                  tokenizer, data_cfg.train.batch_size, False)
        scores.append(linear_probe_score(model.to(device), sft_loader, test_loader, device,
                                          gen, probe_epochs=probe_epochs, eval_r=eval_r))
    return sum(scores) / len(scores), scores


def main():
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parent.parent
    data_cfg = load_run_config(root / args.data_config)
    if args.device:
        data_cfg.train.device = args.device
    if args.batch_size:
        data_cfg.train.batch_size = args.batch_size
    device = get_device(data_cfg.train.device)

    print(f"== probing candidate encoders (HTE linear-probe R2, seeds {args.probe_seeds}) ==")
    scores = {}
    by_activation: dict[str, list[float]] = {}
    for enc in args.encoders:
        mean_r2, per_seed = probe_encoder(
            enc, data_cfg, device, args.probe_epochs, args.eval_r, args.probe_seeds
        )
        scores[enc] = mean_r2
        act = model_config_from_checkpoint(f"{enc}/encoder.pt").activation
        by_activation.setdefault(act, []).append(mean_r2)
        detail = ", ".join(f"{r:.4f}" for r in per_seed)
        print(f"  {enc} [{act}]: probe R2 = {mean_r2:.4f}  (per-seed: {detail})")

    # the 2x2 comparison: mean over seeds per activation
    print("\n-- per-activation mean probe R2 (over seeds) --")
    for act, vals in by_activation.items():
        print(f"  {act}: {sum(vals)/len(vals):.4f}  (n={len(vals)})")

    best = max(scores, key=scores.get)
    print(f"\n== winner: {best} (mean probe R2 {scores[best]:.4f}) -> forking into the 4 runs ==")

    for fork in args.forks:
        for seed in args.seeds:
            cmd = [sys.executable, str(root / "scripts" / "run_sft.py"),
                   "--config", str(root / fork), "--pretrained", best, "--seed", str(seed)]
            if args.device:
                cmd += ["--device", args.device]
            if args.epochs is not None:
                cmd += ["--epochs", str(args.epochs)]
            if args.batch_size is not None:
                cmd += ["--batch-size", str(args.batch_size)]
            print(f"\n########## {fork} (seed {seed}) on {best} ##########")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
