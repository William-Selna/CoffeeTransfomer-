"""Orchestrate Stage 1 -> Stage 2 into a single encoder checkpoint.

Stage 1 (molecular grammar) and Stage 2 (reactivity) train the SAME MLMModel in
sequence — Stage 2 continues from Stage 1's weights — then the encoder is saved
with its ModelConfig and tokenizer for the SFT forks to load.
"""

from __future__ import annotations

from functools import partial

from torch.utils.data import DataLoader

from ..data.corpus import PackedTokenDataset, read_smiles_lines
from ..data.slots import DEFAULT_SCHEMA
from ..data.synthetic import (
    all_synthetic_smiles,
    synthetic_molecule_corpus,
    synthetic_reaction_corpus,
)
from ..data.tokenizer import SmilesTokenizer
from ..models.heads import MLMModel
from ..utils.config import PretrainConfig
from .pretrain import (
    InMemoryTokenDataset,
    MoleculeMLMDataset,
    MLMTrainer,
    mlm_collate,
)


def build_pretrain_tokenizer(cfg: PretrainConfig) -> SmilesTokenizer:
    if cfg.tokenizer_path:
        return SmilesTokenizer.load(cfg.tokenizer_path)
    if cfg.synthetic:
        return SmilesTokenizer.build(all_synthetic_smiles(), schema=DEFAULT_SCHEMA)
    raise ValueError(
        "real pretraining needs a shared tokenizer; build one with "
        "scripts/prepare_corpus.py and set pretrain.tokenizer_path"
    )


def _loader(cfg: PretrainConfig, dataset, tokenizer):
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
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
    """Returns (mlm_model, tokenizer, val_loss)."""
    tokenizer = build_pretrain_tokenizer(cfg)
    cfg.model.vocab_size = tokenizer.vocab_size
    cfg.model.num_slot_types = tokenizer.schema.num_slot_types
    model = MLMModel(cfg.model)

    val_loss = None

    if cfg.stage1_enabled:
        ds1 = _stage1_dataset(cfg, tokenizer)
        loader1 = _loader(cfg, ds1, tokenizer)
        t1 = MLMTrainer(model, tokenizer, device, cfg.mlm_prob, cfg.lr, generator,
                        mask_mode="uniform", amp=cfg.amp, compile=cfg.compile)
        print(f"[stage1] molecules={len(ds1)} epochs={cfg.stage1_epochs}")
        t1.train(loader1, epochs=cfg.stage1_epochs)
        model = getattr(t1.model, "_orig_mod", t1.model)

    if cfg.stage2_enabled:
        ds2 = _stage2_dataset(cfg, tokenizer)
        loader2 = _loader(cfg, ds2, tokenizer)
        mode = "span" if cfg.span_mask_stage2 else "uniform"
        t2 = MLMTrainer(model, tokenizer, device, cfg.mlm_prob, cfg.lr, generator,
                        mask_mode=mode, amp=cfg.amp, compile=cfg.compile)
        print(f"[stage2] reactions={len(ds2)} epochs={cfg.stage2_epochs} mask={mode}")
        t2.train(loader2, epochs=cfg.stage2_epochs)
        val_loss = t2.evaluate(loader2)
        model = getattr(t2.model, "_orig_mod", t2.model)

    return model, tokenizer, val_loss
