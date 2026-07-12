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
    bh_xlsx: str | None = None             # path to Dreher_and_Doyle_input_data.xlsx
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
class RunConfig:
    name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)


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
    )
