#!/usr/bin/env python
"""Drive the four target runs — the SFT/RL-ratio axis at fixed total data.

    python scripts/sweep_sft_ratio.py                 # all 4 configs, default seed
    python scripts/sweep_sft_ratio.py --seeds 0 1 2   # add seeds for variance
    python scripts/sweep_sft_ratio.py --device cpu --epochs 2   # quick smoke

Each config runs Stage 3 (+ Stage 4 where enabled) via run_sft's machinery and
a compact comparison table is printed at the end. Add seeds to reproduce the
design's 5-seed variance reporting (Section 7/9).
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

RUNS = [
    "configs/run_sft100_rl0.yaml",
    "configs/run_sft90_rl10.yaml",
    "configs/run_sft75_rl25.yaml",
    "configs/run_sft50_rl50.yaml",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--configs", nargs="+", default=RUNS)
    return p.parse_args()


def main():
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parent.parent
    for config in args.configs:
        for seed in args.seeds:
            cmd = [sys.executable, str(root / "scripts" / "run_sft.py"), "--config", str(root / config), "--seed", str(seed)]
            if args.device:
                cmd += ["--device", args.device]
            if args.epochs is not None:
                cmd += ["--epochs", str(args.epochs)]
            print(f"\n########## {config} (seed {seed}) ##########")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
