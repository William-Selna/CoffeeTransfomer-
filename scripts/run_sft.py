#!/usr/bin/env python
"""Drive one run: Stage 3 SFT (+ optional Stage 4 GRPO) for a single config.

    python scripts/run_sft.py --config configs/run_sft75_rl25.yaml
    python scripts/run_sft.py --config configs/run_sft75_rl25.yaml --device cpu --epochs 2

The synthetic data path runs end-to-end on CPU with no downloads (Section 8:
debug locally first). Point `data.bh_xlsx` at the real sheet for a real run.
"""

from __future__ import annotations

import argparse
import json
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
    load_pretrained_bundle,
    make_dataset,
    make_loader,
)
from coffee_transformer.training.grpo import GRPOTrainer
from coffee_transformer.training.sft import SFTTrainer
from coffee_transformer.models.recurrent_depth import count_parameters
from coffee_transformer.utils.config import load_run_config
from coffee_transformer.utils.device import get_device
from coffee_transformer.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default=None, help="override train.device (cuda|cpu|mps)")
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--seed", type=int, default=None, help="override train.seed")
    p.add_argument("--no-grpo", action="store_true", help="skip Stage 4 even if enabled")
    p.add_argument("--synthetic", action="store_true",
                   help="force the offline toy data (keeps the CPU smoke path working "
                        "now that configs point at the real BH sheet)")
    p.add_argument("--pretrained", default=None,
                   help="pretrained encoder dir (encoder.pt + tokenizer.json); "
                        "overrides pretrained_encoder in the config")
    p.add_argument("--batch-size", type=int, default=None, help="override batch size")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_run_config(args.config)
    if args.device:
        cfg.train.device = args.device
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.pretrained is not None:
        cfg.pretrained_encoder = args.pretrained
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
        cfg.grpo.batch_size = min(cfg.grpo.batch_size, args.batch_size)
    if args.synthetic:
        cfg.data.synthetic = True

    generator = set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    print(f"== run {cfg.name} | device {device} | seed {cfg.train.seed} ==")

    examples = load_examples(cfg)
    if cfg.pretrained_encoder:
        # Transfer column: reuse the pretrained encoder + its shared tokenizer,
        # attach a fresh histogram head (Section 6, Stage 3). The head is sized
        # by the run config's num_bins (e.g. 4 coarse bins for the USPTO SFT),
        # independent of the encoder's pretraining bin count.
        model, tokenizer = load_pretrained_bundle(
            cfg.pretrained_encoder, override_num_bins=cfg.model.num_bins
        )
        print(f"loaded pretrained encoder from {cfg.pretrained_encoder}")
    else:
        tokenizer = build_tokenizer(examples)
        model = build_model(cfg, tokenizer)
    print(f"vocab {tokenizer.vocab_size} | params {count_parameters(model)/1e6:.2f}M "
          f"| recurrent={model.cfg.recurrent} train_r={model.cfg.resolved_train_r()}")

    pools = build_pools(cfg, examples)
    print(f"pools: sft={len(pools.sft)} rl={len(pools.rl)} test={len(pools.test)}")

    sft_ds = make_dataset(cfg, tokenizer, pools.sft, train=True)
    test_ds = make_dataset(cfg, tokenizer, pools.test, train=False)
    sft_loader = make_loader(cfg, sft_ds, tokenizer, cfg.train.batch_size, shuffle=True)
    test_loader = make_loader(cfg, test_ds, tokenizer, cfg.train.batch_size, shuffle=False)

    loss_fn = YieldLoss(cfg.loss)
    trainer = SFTTrainer(model, loss_fn, cfg.train, device, generator)
    sft_result = trainer.train(sft_loader, test_loader)

    print("\n== SFT test-time-compute sweep ==")
    for r, m in sft_result.ttc.items():
        print(f"  r={r:>2}: R2={m['r2']:.3f} MAE={m['mae']:.2f} spearman={m['spearman']:.3f}")

    grpo_result = None
    if cfg.grpo.enabled and not args.no_grpo and len(pools.rl) > 0:
        print("\n== Stage 4: GRPO on held-out RL pool ==")
        rl_ds = make_dataset(cfg, tokenizer, pools.rl, train=True)
        rl_loader = make_loader(cfg, rl_ds, tokenizer, cfg.grpo.batch_size, shuffle=True)
        grpo_trainer = GRPOTrainer(model, cfg.grpo, device, generator)
        grpo_result = grpo_trainer.train(rl_loader)

        from coffee_transformer.eval.ttc import ttc_sweep
        post = ttc_sweep(model, test_loader, cfg.train.eval_r_values, device)
        print("\n== post-GRPO test-time-compute sweep ==")
        for r, m in post.items():
            print(f"  r={r:>2}: R2={m['r2']:.3f} MAE={m['mae']:.2f} spearman={m['spearman']:.3f}")

    out_dir = pathlib.Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    tokenizer.save(out_dir / "tokenizer.json")
    metrics = {
        "name": cfg.name,
        "params_millions": count_parameters(model) / 1e6,
        "pools": {"sft": len(pools.sft), "rl": len(pools.rl), "test": len(pools.test)},
        "sft": {"steps": sft_result.steps, "final_train_loss": sft_result.final_train_loss,
                "ttc": sft_result.ttc},
        "grpo": None if grpo_result is None else {
            "steps": grpo_result.steps, "final_loss": grpo_result.final_loss,
            "mean_reward": grpo_result.mean_reward},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
