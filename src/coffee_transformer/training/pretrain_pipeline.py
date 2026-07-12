"""Orchestrate Stage 1 -> Stage 2 into a single encoder checkpoint.

Stage 1 (molecular grammar) and Stage 2 (reactivity) train the SAME MLMModel in
sequence — Stage 2 continues from Stage 1's weights — then the best-by-val
encoder is saved with its ModelConfig and tokenizer for the SFT forks to load.

Hardening (so a 2-3 hr run is genuinely fire-and-forget):
  * each stage carves out a held-out val slice and logs val MLM loss;
  * warmup+cosine LR (in MLMTrainer) for early-training stability;
  * a rolling `encoder_latest.pt` is written every `ckpt_every` steps for crash
    recovery, and the best-by-val encoder becomes the canonical `encoder.pt`;
  * non-finite / exploding loss raises DivergenceError (caught by the script).
"""

from __future__ import annotations

import random
from functools import partial

from torch.utils.data import DataLoader, Subset

from ..data.corpus import PackedTokenDataset, read_smiles_lines
from ..data.synthetic import (
    all_synthetic_smiles,
    synthetic_molecule_corpus,
    synthetic_reaction_corpus,
)
from ..data.slots import DEFAULT_SCHEMA
from ..data.tokenizer import SmilesTokenizer
from ..models.heads import MLMModel
from ..utils.config import PretrainConfig
from .checkpoint import save_pretrained_encoder
from .pretrain import (
    InMemoryTokenDataset,
    MLMTrainer,
    MoleculeMLMDataset,
    mlm_collate,
)


def build_pretrain_tokenizer(cfg: PretrainConfig) -> SmilesTokenizer:
    # synthetic mode is self-contained — never depends on a prebuilt vocab file
    if cfg.synthetic:
        return SmilesTokenizer.build(all_synthetic_smiles(), schema=DEFAULT_SCHEMA)
    if cfg.tokenizer_path:
        return SmilesTokenizer.load(cfg.tokenizer_path)
    raise ValueError(
        "real pretraining needs a shared tokenizer; build one with "
        "scripts/prepare_corpus.py and set pretrain.tokenizer_path"
    )


def _split(dataset, val_frac: float, seed: int):
    """Deterministic shuffled train/val split (held-out MLM val — gap fix)."""
    n = len(dataset)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(n * val_frac)) if n > 1 else 0
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train = Subset(dataset, train_idx)
    val = Subset(dataset, val_idx) if n_val else None
    return train, val


def _loader(cfg: PretrainConfig, dataset, tokenizer, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=partial(mlm_collate, pad_id=tokenizer.pad_id),
        drop_last=False,
    )


def _stage1_dataset(cfg: PretrainConfig, tokenizer):
    if cfg.synthetic:
        mols = synthetic_molecule_corpus(cfg.synthetic_mol_n, seed=cfg.seed)
    else:
        if not cfg.pubchem_path:
            raise ValueError("stage1 needs pubchem_path (cleaned .smi) for a real run")
        mols = read_smiles_lines(cfg.pubchem_path, limit=cfg.pubchem_limit)
    return MoleculeMLMDataset(mols, tokenizer, cfg.max_length)


def _stage2_dataset(cfg: PretrainConfig, tokenizer):
    if cfg.synthetic:
        reactions = synthetic_reaction_corpus(cfg.synthetic_rxn_n, seed=cfg.seed + 1)
        records = []
        for comps in reactions:
            ids, slots = tokenizer.encode_reaction(comps)
            records.append({"input_ids": ids[: cfg.max_length], "slot_type_ids": slots[: cfg.max_length]})
        return InMemoryTokenDataset(records)
    if not cfg.stage2_reactions_path:
        raise ValueError("stage2 needs stage2_reactions_path (pretokenized prefix) for a real run")
    return PackedTokenDataset(cfg.stage2_reactions_path)


def run_pretraining(cfg: PretrainConfig, device, generator=None):
    """Returns (mlm_model, tokenizer, best_val_loss). The returned model holds
    the best-by-val encoder weights (canonical encoder.pt)."""
    tokenizer = build_pretrain_tokenizer(cfg)
    cfg.model.vocab_size = tokenizer.vocab_size
    cfg.model.num_slot_types = tokenizer.schema.num_slot_types
    model = MLMModel(cfg.model)

    best = {"state": None, "val": float("inf")}

    def make_checkpoint_cb(capture_best: bool):
        def cb(m, step, val_loss, is_best):
            # rolling latest for crash recovery
            save_pretrained_encoder(
                cfg.out_dir, cfg.model, m, tokenizer, val_loss,
                filename="encoder_latest.pt", step=step,
            )
            if capture_best and is_best and val_loss is not None:
                enc = getattr(m, "_orig_mod", m).encoder
                best["state"] = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
                best["val"] = val_loss
        return cb

    def run_stage(dataset, mask_mode, epochs, capture_best):
        train_ds, val_ds = _split(dataset, cfg.val_frac, cfg.seed)
        train_loader = _loader(cfg, train_ds, tokenizer, shuffle=True)
        val_loader = _loader(cfg, val_ds, tokenizer, shuffle=False) if val_ds else None
        trainer = MLMTrainer(
            model, tokenizer, device, cfg.mlm_prob, cfg.lr, generator,
            mask_mode=mask_mode, amp=cfg.amp, compile=cfg.compile,
        )
        print(f"[{mask_mode}] train={len(train_ds)} val={0 if val_ds is None else len(val_ds)} epochs={epochs}")
        trainer.train(
            train_loader, val_loader=val_loader, epochs=epochs,
            warmup_frac=cfg.warmup_frac, eval_every=cfg.eval_every,
            ckpt_every=cfg.ckpt_every, on_checkpoint=make_checkpoint_cb(capture_best),
        )
        return getattr(trainer.model, "_orig_mod", trainer.model)

    if cfg.stage1_enabled:
        model = run_stage(_stage1_dataset(cfg, tokenizer), "uniform", cfg.stage1_epochs,
                          capture_best=not cfg.stage2_enabled)

    if cfg.stage2_enabled:
        mode = "span" if cfg.span_mask_stage2 else "uniform"
        model = run_stage(_stage2_dataset(cfg, tokenizer), mode, cfg.stage2_epochs, capture_best=True)

    # restore best-by-val encoder as the canonical weights
    if best["state"] is not None:
        model.encoder.load_state_dict(best["state"])
        return model, tokenizer, best["val"]
    return model, tokenizer, None
