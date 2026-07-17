#!/usr/bin/env python
"""Run one pretraining config (Stages 1->2) and save an encoder checkpoint.

    python scripts/pretrain.py --config configs/pretrain_5m.yaml
    python scripts/pretrain.py --config configs/pretrain_10m.yaml
    # offline smoke (no downloads): forces the synthetic toy corpora
    python scripts/pretrain.py --config configs/pretrain_5m.yaml --synthetic --device cpu \
        --stage1-epochs 1 --stage2-epochs 1

Writes <out_dir>/{encoder.pt, tokenizer.json}. Point the SFT forks at <out_dir>
via `pretrained_encoder:` or `run_sft.py --pretrained <out_dir>`.

Run the two candidates (5M and 10M) then pick with scripts/select_and_sweep.py.
For the ~1-hour target, run them on two GPUs in parallel.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# Cap CPU threads per process BEFORE importing torch — with several pretrainings
# sharing one box, each torch process otherwise grabs all cores for tiny CPU ops
# and the machine thrashes (load average >> cores). TORCH_NUM_THREADS overrides.
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("TORCH_NUM_THREADS", "8"))
# reduce CUDA fragmentation across the co-scheduled runs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import torch

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "8")))

from coffee_transformer.models.recurrent_depth import count_parameters
from coffee_transformer.training.checkpoint import save_pretrained_encoder
from coffee_transformer.training.pretrain import DivergenceError
from coffee_transformer.training.pretrain_pipeline import run_pretraining
from coffee_transformer.utils.config import load_pretrain_config
from coffee_transformer.utils.device import get_device
from coffee_transformer.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--synthetic", action="store_true", help="force offline toy corpora")
    p.add_argument("--stage1-epochs", type=int, default=None)
    p.add_argument("--stage2-epochs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None, help="override dataloader workers")
    p.add_argument("--batch-size", type=int, default=None, help="override batch size")
    p.add_argument("--resume", default=None,
                   help="continue from a saved encoder: a run dir (uses encoder.pt or "
                        "encoder_latest.pt) or a direct .pt path — for the canonical->augmented hot swap")
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_pretrain_config(args.config)
    if args.device:
        cfg.device = args.device
    if args.synthetic:
        cfg.synthetic = True
    if args.stage1_epochs is not None:
        cfg.stage1_epochs = args.stage1_epochs
    if args.stage2_epochs is not None:
        cfg.stage2_epochs = args.stage2_epochs
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.no_compile:
        cfg.compile = False

    resume_from = None
    if args.resume:
        rp = pathlib.Path(args.resume)
        if rp.is_dir():
            cand = rp / "encoder.pt"
            resume_from = str(cand if cand.exists() else rp / "encoder_latest.pt")
        else:
            resume_from = str(rp)

    generator = set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"== pretrain {cfg.name} | device {device} | synthetic={cfg.synthetic} "
          f"| activation={cfg.model.activation} | seed={cfg.seed}"
          f"{' | RESUME' if resume_from else ''} ==")

    try:
        model, tokenizer, val_loss = run_pretraining(cfg, device, generator, resume_from=resume_from)
    except DivergenceError as e:
        print(f"\n[DIVERGED] {e}\n"
              f"  A rolling checkpoint may exist at {cfg.out_dir}/encoder_latest.pt.\n"
              f"  Try lowering lr, raising warmup_frac, or reducing truncated_bptt_k / r_max.")
        sys.exit(2)
    print(f"params {count_parameters(model)/1e6:.2f}M | best held-out val MLM loss "
          f"{'n/a' if val_loss is None else f'{val_loss:.4f}'}")

    path = save_pretrained_encoder(cfg.out_dir, cfg.model, model, tokenizer, val_loss)
    (pathlib.Path(cfg.out_dir) / "pretrain_metrics.json").write_text(
        json.dumps({"name": cfg.name, "val_loss": val_loss,
                    "params_millions": count_parameters(model) / 1e6}, indent=2)
    )
    print(f"saved encoder -> {path}")


if __name__ == "__main__":
    main()
