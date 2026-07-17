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
from pathlib import Path

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


def _split(dataset, val_frac: float, seed: int, val_max: int | None = None):
    """Deterministic shuffled train/val split (held-out MLM val — gap fix).

    `val_max` caps the held-out slice so eval cost stops scaling with corpus
    size — on the 5x-larger augmented store, 2% would be ~1M molecules and each
    eval a multi-thousand-batch pass. A fixed ~200k slice is more than enough for
    a stable val estimate.
    """
    n = len(dataset)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(n * val_frac)) if n > 1 else 0
    if val_max is not None and n_val > val_max:
        n_val = val_max
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train = Subset(dataset, train_idx)
    val = Subset(dataset, val_idx) if n_val else None
    return train, val


def _resolve_resume(resume_from: str, expected_vocab: int) -> str:
    """Resolve a --resume argument to a concrete checkpoint file.

    A direct .pt path is returned as-is (the caller vocab-checks it). A run DIR
    is resolved to the first checkpoint whose encoder vocab matches the tokenizer,
    trying the real trained snapshots first — this skips the classic wrong-file
    trap where a stale 38-vocab synthetic smoke encoder.pt sat next to the real
    622-vocab encoder and got picked blindly.
    """
    import torch

    p = Path(resume_from)
    if not p.is_dir():
        return str(p)
    tried = []
    for name in ("encoder_phase1.pt", "encoder_latest.pt", "encoder.pt"):
        c = p / name
        if not c.exists():
            continue
        v = int(torch.load(c, map_location="cpu")["encoder_state"]["embedding.token.weight"].shape[0])
        tried.append(f"{name}={v}")
        if v == expected_vocab:
            return str(c)
    raise ValueError(
        f"--resume {p}: no checkpoint matches tokenizer vocab {expected_vocab} "
        f"(found {', '.join(tried) or 'none'}). Pass the real encoder_phase1.pt directly."
    )


def _loader(cfg: PretrainConfig, dataset, tokenizer, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=partial(mlm_collate, pad_id=tokenizer.pad_id),
        drop_last=False,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )


def _stage1_dataset(cfg: PretrainConfig, tokenizer):
    if cfg.synthetic:
        mols = synthetic_molecule_corpus(cfg.synthetic_mol_n, seed=cfg.seed)
        return MoleculeMLMDataset(mols, tokenizer, cfg.max_length)
    # fast path: pre-tokenized mmap (tokenized once by prepare_corpus)
    if cfg.stage1_tokens_path and Path(f"{cfg.stage1_tokens_path}.tokens.npy").exists():
        return PackedTokenDataset(cfg.stage1_tokens_path)
    # fallback: tokenize the cleaned .smi on the fly (slower, CPU-bound)
    if cfg.pubchem_path:
        mols = read_smiles_lines(cfg.pubchem_path, limit=cfg.pubchem_limit)
        return MoleculeMLMDataset(mols, tokenizer, cfg.max_length)
    raise ValueError("stage1 needs stage1_tokens_path (preferred) or pubchem_path for a real run")


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


def run_pretraining(cfg: PretrainConfig, device, generator=None, resume_from: str | None = None):
    """Returns (mlm_model, tokenizer, best_val_loss). The returned model holds
    the best-by-val encoder weights (canonical encoder.pt).

    `resume_from` (path to an encoder.pt/encoder_latest.pt) continues from a
    saved ENCODER — used for the canonical->augmented hot swap. The MLM head is
    re-initialized (it isn't saved) and re-warms in a few hundred steps; the
    encoder, which is what matters, is preserved. Optimizer state is fresh, so
    the LR schedule warms up again on the new data (that's fine, arguably good
    when the data distribution changes)."""
    import torch

    tokenizer = build_pretrain_tokenizer(cfg)
    cfg.model.vocab_size = tokenizer.vocab_size
    cfg.model.num_slot_types = tokenizer.schema.num_slot_types
    model = MLMModel(cfg.model)
    if resume_from:
        resume_path = _resolve_resume(resume_from, tokenizer.vocab_size)
        ckpt = torch.load(resume_path, map_location="cpu")
        enc_vocab = int(ckpt["encoder_state"]["embedding.token.weight"].shape[0])
        if enc_vocab != tokenizer.vocab_size:
            raise ValueError(
                f"resume vocab mismatch: {resume_path} encoder has vocab {enc_vocab}, "
                f"but the tokenizer has {tokenizer.vocab_size}. This is almost always the "
                f"wrong checkpoint (a synthetic smoke-test encoder is 38-vocab). Point "
                f"--resume at the real encoder_phase1.pt / encoder_latest.pt."
            )
        model.encoder.load_state_dict(ckpt["encoder_state"])
        prev = ckpt.get("val_loss")
        print(f"resumed encoder from {resume_path}" + (f" (prev val {prev:.4f})" if prev else ""))

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
        train_ds, val_ds = _split(dataset, cfg.val_frac, cfg.seed, cfg.val_max)
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
