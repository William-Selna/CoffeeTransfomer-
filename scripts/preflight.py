#!/usr/bin/env python
"""Preflight: verify the environment and plumbing before spending GPU hours.

    python scripts/preflight.py

Checks torch + CUDA, the vendored HTE data, and runs a tiny synthetic
pretrain -> transfer -> SFT end-to-end on CPU. Prints a GO / NO-GO summary. A
non-zero exit means don't start the real run yet.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

ROOT = pathlib.Path(__file__).resolve().parent.parent
ok = True


def check(label, cond, detail="", hard=True):
    """hard=False makes a failing check a warning that doesn't block GO."""
    global ok
    mark = "OK  " if cond else ("FAIL" if hard else "WARN")
    if not cond and hard:
        ok = False
    print(f"[{mark}] {label}{(' — ' + detail) if detail else ''}")
    return cond


def main():
    print("== preflight ==")

    try:
        import torch
        check("torch import", True, f"{torch.__version__}")
        cuda = torch.cuda.is_available()
        check("CUDA available", cuda,
              torch.cuda.get_device_name(0) if cuda else "CPU only — fine for debug; the real run needs CUDA",
              hard=False)
    except Exception as e:
        check("torch import", False, str(e))
        print("\nNO-GO: install torch (see requirements.txt).")
        sys.exit(1)

    bh = ROOT / "data/hte/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx"
    check("BH data vendored", bh.exists(), str(bh))

    # end-to-end plumbing on synthetic data
    try:
        import torch
        from coffee_transformer.losses.combined import YieldLoss
        from coffee_transformer.training.builder import (
            build_pools, load_examples, make_dataset, make_loader, load_pretrained_bundle)
        from coffee_transformer.training.checkpoint import save_pretrained_encoder
        from coffee_transformer.training.pretrain_pipeline import run_pretraining
        from coffee_transformer.training.sft import SFTTrainer
        from coffee_transformer.utils.config import PretrainConfig, RunConfig
        from coffee_transformer.utils.seed import set_seed

        dev = torch.device("cpu")
        gen = set_seed(0)
        pc = PretrainConfig()
        pc.synthetic = True
        pc.synthetic_mol_n = 40; pc.synthetic_rxn_n = 40
        pc.stage1_epochs = 1; pc.stage2_epochs = 1
        pc.batch_size = 8; pc.num_workers = 0; pc.amp = False; pc.compile = False
        pc.lr = 1e-4  # gentle LR keeps the tiny toy run stable
        pc.model.d_model = 32; pc.model.n_heads = 4; pc.model.d_ff = 64
        pc.model.prelude_layers = 1; pc.model.core_layers = 2
        pc.out_dir = str(ROOT / "runs/_preflight")
        model, tok, val = run_pretraining(pc, dev, gen)
        bundle = ROOT / "runs/_preflight_bundle"
        save_pretrained_encoder(bundle, pc.model, model, tok, val)
        check("pretrain (stage1->2) + checkpoint", (bundle / "encoder.pt").exists(),
              f"val MLM loss {val:.3f}" if val else "")

        rc = RunConfig()
        rc.data.synthetic = True; rc.data.synthetic_n = 80
        rc.train.epochs = 1; rc.train.batch_size = 8; rc.train.device = "cpu"
        rc.train.linear_probe_steps = 2
        ym, tok2 = load_pretrained_bundle(str(bundle))
        ex = load_examples(rc); pools = build_pools(rc, ex)
        sl = make_loader(rc, make_dataset(rc, tok2, pools.sft, True), tok2, 8, True)
        tl = make_loader(rc, make_dataset(rc, tok2, pools.test, False), tok2, 8, False)
        res = SFTTrainer(ym, YieldLoss(rc.loss), rc.train, dev, gen).train(sl, tl)
        import math
        r2 = res.ttc[rc.train.eval_r_values[0]]["r2"]
        check("transfer + SFT + eval", math.isfinite(r2), f"probe path ran (R2={r2:.3f} on toy data)")
    except Exception as e:
        import traceback; traceback.print_exc()
        check("end-to-end plumbing", False, str(e))

    import shutil
    for d in ("runs/_preflight", "runs/_preflight_bundle"):
        shutil.rmtree(ROOT / d, ignore_errors=True)

    print("\n" + ("GO — plumbing verified." if ok else "NO-GO — fix the FAILs above."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
