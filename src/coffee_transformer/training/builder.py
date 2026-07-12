"""Turn a RunConfig into the concrete objects a trainer needs:
tokenizer, model, and the SFT/RL/test datasets.

Keeps the wiring in one place so `scripts/run_sft.py`, `scripts/run_grpo.py`,
and the tests all construct things identically (same tokenizer, same splits).
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from ..data.collate import make_collate_fn
from ..data.dataset import HTEDataset, ReactionExample, load_buchwald_hartwig
from ..data.slots import BUCHWALD_HARTWIG_SLOTS, DEFAULT_SCHEMA
from ..data.synthetic import make_synthetic_bh
from ..data.tokenizer import SmilesTokenizer
from ..models.heads import YieldModel
from ..utils.config import RunConfig
from .checkpoint import load_encoder_into_yield_model, model_config_from_checkpoint
from .splits import Pools, split_pools


def load_examples(cfg: RunConfig) -> list[ReactionExample]:
    d = cfg.data
    if d.synthetic:
        return make_synthetic_bh(n=d.synthetic_n, seed=cfg.train.seed)
    if d.bh_xlsx is None:
        raise ValueError("data.synthetic is False but data.bh_xlsx is not set")
    return load_buchwald_hartwig(d.bh_xlsx, sheet=d.sheet)


def build_tokenizer(examples: list[ReactionExample]) -> SmilesTokenizer:
    corpus = [smi for ex in examples for _, smi in ex.components]
    return SmilesTokenizer.build(corpus, schema=DEFAULT_SCHEMA)


def build_model(cfg: RunConfig, tokenizer: SmilesTokenizer) -> YieldModel:
    cfg.model.vocab_size = tokenizer.vocab_size
    cfg.model.num_slot_types = tokenizer.schema.num_slot_types
    return YieldModel(cfg.model)


def load_pretrained_bundle(ckpt_dir: str) -> tuple[YieldModel, SmilesTokenizer]:
    """Build an SFT YieldModel from a pretrained encoder checkpoint directory.

    Rebuilds the exact encoder architecture from the saved ModelConfig, loads
    the pretrained encoder weights, attaches a FRESH histogram head, and reuses
    the shared tokenizer saved beside it (Section 6: one vocab across stages).
    """
    ckpt_dir = str(ckpt_dir)
    encoder_pt = f"{ckpt_dir}/encoder.pt"
    tokenizer = SmilesTokenizer.load(f"{ckpt_dir}/tokenizer.json")
    model_cfg = model_config_from_checkpoint(encoder_pt)
    model = YieldModel(model_cfg)
    load_encoder_into_yield_model(encoder_pt, model)
    return model, tokenizer


def make_dataset(cfg: RunConfig, tokenizer: SmilesTokenizer, examples, train: bool) -> HTEDataset:
    d = cfg.data
    return HTEDataset(
        examples,
        tokenizer,
        randomize_order=d.randomize_order and train,
        randomize_smiles_prob=d.randomize_smiles_prob if train else 0.0,
        max_length=d.max_length,
        seed=cfg.train.seed,
    )


def build_pools(cfg: RunConfig, examples: list[ReactionExample]) -> Pools:
    d = cfg.data
    return split_pools(
        examples,
        sft_fraction=d.sft_fraction,
        rl_fraction=d.rl_fraction,
        test_fraction=d.test_fraction,
        data_fraction=d.data_fraction,
        seed=cfg.train.seed,
    )


def make_loader(cfg: RunConfig, dataset: HTEDataset, tokenizer, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(tokenizer.pad_id),
        drop_last=False,
    )
