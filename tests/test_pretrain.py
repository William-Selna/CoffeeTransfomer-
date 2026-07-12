import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import torch

from coffee_transformer.data.corpus import (
    PackedTokenDataset,
    clean_corpus,
    pretokenize_records,
    reaction_records,
)
from coffee_transformer.data.slots import DEFAULT_SCHEMA
from coffee_transformer.data.synthetic import (
    all_synthetic_smiles,
    synthetic_reaction_corpus,
)
from coffee_transformer.data.tokenizer import SmilesTokenizer
from coffee_transformer.training.builder import (
    build_pools,
    load_examples,
    make_dataset,
    make_loader,
)
from coffee_transformer.training.checkpoint import save_pretrained_encoder
from coffee_transformer.training.builder import load_pretrained_bundle
from coffee_transformer.training.pretrain import (
    DivergenceError,
    MLMTrainer,
    MoleculeMLMDataset,
    mlm_collate,
    span_mask_tokens,
)
from coffee_transformer.training.pretrain_pipeline import run_pretraining
from coffee_transformer.training.probe import linear_probe_score
from coffee_transformer.utils.config import PretrainConfig, RunConfig
from coffee_transformer.utils.seed import set_seed


def _tok():
    return SmilesTokenizer.build(all_synthetic_smiles(), schema=DEFAULT_SCHEMA)


def _small_model(cfg):
    cfg.model.d_model = 32
    cfg.model.n_heads = 4
    cfg.model.d_ff = 64
    cfg.model.prelude_layers = 1
    cfg.model.core_layers = 2
    return cfg


def test_clean_corpus_dedups():
    out = clean_corpus(["CP(C)C", "CP(C)C", "c1ccno1"])
    assert len(out) == len(set(out)) == 2


def test_pretokenize_roundtrip(tmp_path):
    tok = _tok()
    reactions = synthetic_reaction_corpus(10)
    prefix = tmp_path / "rx"
    meta = pretokenize_records(reaction_records(reactions, tok, 256), prefix)
    assert meta["num_records"] == 10
    ds = PackedTokenDataset(prefix)
    assert len(ds) == 10
    item = ds[0]
    assert len(item["input_ids"]) == len(item["slot_type_ids"]) > 0


def test_span_mask_hits_slot_spans():
    tok = _tok()
    comps = [("ARYL_HALIDE", "Clc1ccccc1"), ("LIG", "CP(C)C")]
    ids, slots = tok.encode_reaction(comps)
    input_ids = torch.tensor([ids])
    slot_ids = torch.tensor([slots])
    masked, labels = span_mask_tokens(input_ids, slot_ids, tok, mask_frac=1.0)
    # with mask_frac=1.0 every slot body token becomes a label and a [MASK]
    assert (labels != -100).any()
    assert (masked == tok.mask_id).any()


def _pretrain_cfg():
    cfg = _small_model(PretrainConfig())
    cfg.synthetic = True
    cfg.synthetic_mol_n = 60
    cfg.synthetic_rxn_n = 60
    cfg.stage1_epochs = 1
    cfg.stage2_epochs = 1
    cfg.batch_size = 16
    cfg.device = "cpu"
    cfg.amp = False
    cfg.compile = False
    cfg.num_workers = 0
    return cfg


def test_divergence_guard_raises():
    gen = set_seed(0)
    tok = _tok()
    from coffee_transformer.models.config import ModelConfig
    from coffee_transformer.models.heads import MLMModel
    from torch.utils.data import DataLoader
    from functools import partial

    m = MLMModel(ModelConfig(vocab_size=tok.vocab_size, num_slot_types=tok.schema.num_slot_types,
                             d_model=16, n_heads=2, d_ff=32, prelude_layers=1, core_layers=1))
    ds = MoleculeMLMDataset(["CP(C)C", "c1ccno1"] * 8, tok, 64)
    loader = DataLoader(ds, batch_size=4, collate_fn=partial(mlm_collate, pad_id=tok.pad_id))
    trainer = MLMTrainer(m, tok, torch.device("cpu"), generator=gen)
    # max_loss=0 forces every (positive) loss to trip the divergence guard
    import pytest
    with pytest.raises(DivergenceError):
        trainer.train(loader, epochs=1, max_loss=0.0)


def test_pretrain_then_transfer_and_probe(tmp_path):
    gen = set_seed(0)
    device = torch.device("cpu")
    cfg = _pretrain_cfg()
    cfg.out_dir = str(tmp_path / "pt")
    model, tokenizer, val_loss = run_pretraining(cfg, device, gen)
    assert val_loss is not None and val_loss > 0
    # hardening: a rolling checkpoint was written during training
    assert (tmp_path / "pt" / "encoder_latest.pt").exists()

    bundle = tmp_path / "bundle"
    save_pretrained_encoder(bundle, cfg.model, model, tokenizer, val_loss)
    assert (bundle / "encoder.pt").exists()
    assert (bundle / "tokenizer.json").exists()

    yield_model, tok2 = load_pretrained_bundle(str(bundle))
    assert tok2.vocab_size == tokenizer.vocab_size

    # the transferred model probes without error and returns a finite R2
    run_cfg = _small_model(RunConfig())
    run_cfg.data.synthetic = True
    run_cfg.data.synthetic_n = 120
    run_cfg.train.device = "cpu"
    examples = load_examples(run_cfg)
    pools = build_pools(run_cfg, examples)
    sft_loader = make_loader(run_cfg, make_dataset(run_cfg, tok2, pools.sft, True), tok2, 16, True)
    test_loader = make_loader(run_cfg, make_dataset(run_cfg, tok2, pools.test, False), tok2, 16, False)
    r2 = linear_probe_score(yield_model, sft_loader, test_loader, device, gen, probe_epochs=1, eval_r=4)
    assert torch.isfinite(torch.tensor(r2))
