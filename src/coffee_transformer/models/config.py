"""Model hyperparameters.

Defaults describe the ~5M-parameter anchor point from the scale plan
(Section 6). The recurrent-depth vs. vanilla-transformer and matched-params vs.
matched-FLOPs comparisons (Section 7) are expressed by toggling `recurrent`,
`core_layers`, and `train_r` — see `configs/` for the sweep points.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # vocabulary / embeddings (filled from the tokenizer at build time)
    vocab_size: int = 300
    num_slot_types: int = 12
    max_position: int = 256

    # transformer width
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"

    # recurrent-depth structure (Section 3): prelude -> core (x r) -> coda
    prelude_layers: int = 2
    core_layers: int = 4
    recurrent: bool = True          # False -> vanilla transformer (train_r forced to 1)
    input_injection: bool = True    # re-inject embedding h as (s + h) every step
    state_init_std: float = 0.4     # noise std for s_0

    # test-time-compute schedule
    train_r: int = 4                # nominal recurrence at train time
    randomize_r: bool = True        # sample r each step (Section 3 stability trick)
    r_min: int = 1
    r_max: int = 8
    truncated_bptt_k: int = 4       # backprop through only the last k iterations

    # typed attention ablation (Section 4)
    typed_attention: str = "embedding"   # "embedding" | "per_slot_kqv"

    # pooling — chosen once, identical across all stages (Section 6, Stage 3)
    pool: str = "cls"               # "cls" | "mean"

    # supervision head
    num_bins: int = 20              # yield histogram bins over [0, 100]
    yield_min: float = 0.0
    yield_max: float = 100.0

    def resolved_train_r(self) -> int:
        return self.train_r if self.recurrent else 1
