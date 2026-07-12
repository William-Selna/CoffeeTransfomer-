"""YAML-backed run configuration.

A run config has five sections — `model`, `loss`, `data`, `train`, `grpo` —
each mapping onto a dataclass. The SFT/RL split (`data.sft_fraction` /
`data.rl_fraction`) is the experimental axis this scaffold targets: the four
`configs/run_sft_*.yaml` files differ only there.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..losses.combined import LossConfig
from ..models.config import ModelConfig


@dataclass
class DataConfig:
    dataset: str = "BH"                    # "BH" | "Suzuki" | "BH+Suzuki"
    # vendored BH data (data/hte/); flip `synthetic: false` in a run config to use it
    bh_xlsx: str | None = "data/hte/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx"
    sheet: str = "FullCV_01"               # canonical random fold; Tests 1-4 for OOD
    synthetic: bool = True                 # use the offline toy generator
    synthetic_n: int = 2000

    # split structure ~60/20/20 supervised / RL pool / test (Section 7)
    sft_fraction: float = 0.75             # fraction of the labeled pool used for SFT
    rl_fraction: float = 0.25              # held out as the GRPO environment
    test_fraction: float = 0.20            # held out from everything for eval

    # main-grid data-efficiency axis: fraction of the SFT pool actually used
    data_fraction: float = 1.0             # {0.5, 0.75, 1.0}

    # augmentation (Section 4)
    randomize_order: bool = True
    randomize_smiles_prob: float = 0.0     # >0 needs RDKit
    max_length: int = 256


@dataclass
class TrainConfig:
    seed: int = 0
    epochs: int = 20
    batch_size: int = 64
    head_lr: float = 3e-3
    encoder_lr_scale: float = 0.1          # encoder LR = head_lr * scale (Section 6)
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    linear_probe_steps: int = 200          # freeze encoder, train head first
    grad_clip: float = 1.0
    deep_supervision: bool = True          # read out every recurrence iteration
    eval_r_values: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    eval_every_epochs: int = 5
    device: str = "cuda"
    log_every: int = 20
    out_dir: str = "runs/default"


@dataclass
class GRPOConfig:
    enabled: bool = False
    group_size: int = 8                    # k predictions per reaction
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-4
    reward: str = "tolerance"              # "tolerance" | "ranking"
    tolerance: float = 10.0                # yield-% window for tolerance shaping
    kl_coef: float = 0.0                   # optional KL-to-reference regularizer
    sample_r: int = 4


@dataclass
class PretrainConfig:
    """Stages 1-2 self-supervised pretraining (produces an encoder checkpoint).

    The two target pretraining runs differ only in `pubchem_limit` (5M vs 10M);
    everything else — model, Stage 2 corpus, schedule — is held fixed so the
    downstream probe comparison isolates Stage-1 corpus scale.
    """

    name: str = "pretrain"
    model: ModelConfig = field(default_factory=ModelConfig)

    # data
    synthetic: bool = True
    synthetic_mol_n: int = 800
    synthetic_rxn_n: int = 800
    pubchem_path: str | None = None          # .smi/.txt, one SMILES per line
    pubchem_limit: int | None = None         # 5_000_000 or 10_000_000
    stage2_reactions_path: str | None = None  # pre-tokenized prefix (see prepare_corpus)
    tokenizer_path: str | None = None         # shared vocab built by prepare_corpus

    # schedule
    stage1_enabled: bool = True
    stage2_enabled: bool = True
    stage1_epochs: int = 10
    stage2_epochs: int = 5
    mlm_prob: float = 0.15
    span_mask_stage2: bool = True

    # hardening: held-out MLM val + warmup/cosine LR + divergence/checkpoint
    val_frac: float = 0.02             # held-out slice of each stage's corpus
    warmup_frac: float = 0.05
    eval_every: int = 500              # steps between held-out val evals
    ckpt_every: int = 1000             # steps between rolling encoder_latest.pt saves

    # optimization / perf
    seed: int = 0
    batch_size: int = 256
    lr: float = 3.0e-4
    max_length: int = 256
    device: str = "cuda"
    amp: bool = True                          # bf16 autocast on CUDA
    compile: bool = False                     # torch.compile
    num_workers: int = 4
    out_dir: str = "runs/pretrain"


@dataclass
class RunConfig:
    name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    # optional: path to a pretrained encoder checkpoint (encoder.pt) to load
    # into the SFT model before Stage 3 (Section 6 transfer column).
    pretrained_encoder: str | None = None


def _from_dict(dc_type, d: dict[str, Any] | None):
    if not d:
        return dc_type()
    fields = {f.name for f in dataclasses.fields(dc_type)}
    unknown = set(d) - fields
    if unknown:
        raise ValueError(f"{dc_type.__name__}: unknown keys {sorted(unknown)}")
    return dc_type(**{k: v for k, v in d.items() if k in fields})


def load_run_config(path: str | Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return RunConfig(
        name=raw.get("name", Path(path).stem),
        model=_from_dict(ModelConfig, raw.get("model")),
        loss=_from_dict(LossConfig, raw.get("loss")),
        data=_from_dict(DataConfig, raw.get("data")),
        train=_from_dict(TrainConfig, raw.get("train")),
        grpo=_from_dict(GRPOConfig, raw.get("grpo")),
        pretrained_encoder=raw.get("pretrained_encoder"),
    )


def load_pretrain_config(path: str | Path) -> PretrainConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    model = _from_dict(ModelConfig, raw.get("model"))
    top = {k: v for k, v in raw.items() if k != "model"}
    fields = {f.name for f in dataclasses.fields(PretrainConfig)} - {"model"}
    unknown = set(top) - fields
    if unknown:
        raise ValueError(f"PretrainConfig: unknown keys {sorted(unknown)}")
    return PretrainConfig(model=model, **{k: v for k, v in top.items() if k in fields})
