# CoffeeTransformer — recurrent-depth transformers for reaction-yield prediction

A PyTorch scaffold for the research plan in *Small-Scale Recurrent-Depth
Transformers for Chemical Reaction Yield Prediction* (July 2026 design summary).
It implements the recurrent-depth architecture, the distributional-supervision
loss stack, and the four-stage `pretrain → SFT → RL` pipeline at the ~5M-param
scale the plan targets — wired so the **four SFT/RL-ratio runs** are drivable
from config.

Status: **runnable scaffold**. The architecture, losses, SFT trainer, GRPO
trainer, MLM pretraining, eval/TTC harness, and a synthetic data path all run
end-to-end on CPU today (16 unit tests + a full SFT+GRPO smoke run pass). What
remains before paper-grade results is wiring the *real* corpora — see
[Data](#data) and [What's stubbed](#whats-stubbed-vs-real).

---

## Install

```bash
pip install -e .                 # core: torch, numpy, pyyaml
pip install -e ".[chem]"         # + rdkit, pandas, openpyxl (real data / SMILES aug)
pip install -e ".[dev]"          # + pytest
```

Install a CUDA build of torch matching your driver, e.g. CUDA 12.1:
`pip install torch --index-url https://download.pytorch.org/whl/cu121`.

## Quickstart (no downloads — synthetic smoke test)

```bash
python scripts/run_sft.py --config configs/run_sft75_rl25.yaml --device cpu --epochs 3
pytest -q
```

The synthetic generator (`data/synthetic.py`) is toy chemistry that exists only
to exercise the plumbing on CPU before renting a GPU (the design's "debug
everything locally first" rule). On it you can already watch the signature
recurrent-depth effect — accuracy rising with test-time iterations `r`.

## The four target runs

`scripts/sweep_sft_ratio.py` drives all four; they differ **only** in the
SFT/RL split at a fixed total labeled budget (`configs/run_sft*_rl*.yaml`):

| Run | SFT % | RL % | Stage 4 | Question it answers |
|-----|------:|-----:|---------|---------------------|
| `sft100_rl0`  | 100 |  0 | off | Supervised-only control — the bar RL must clear |
| `sft90_rl10`  |  90 | 10 | GRPO | The LLM-era default: much SFT, a little RL |
| `sft75_rl25`  |  75 | 25 | GRPO | The design's canonical 60/20/20 split |
| `sft50_rl50`  |  50 | 50 | GRPO | Does a big RL share help, or just add variance? |

```bash
# all four, one seed (add --device cpu --epochs 2 for a quick pass)
python scripts/sweep_sft_ratio.py

# reproduce the design's 5-seed variance reporting
python scripts/sweep_sft_ratio.py --seeds 0 1 2 3 4
```

The test split is held fixed across all four so the comparison is clean. Each
run writes `runs/<name>/{model.pt, tokenizer.json, metrics.json}` including the
full test-time-compute sweep (R² / MAE / Spearman at r ∈ {1,2,4,8,16}).

> On the SFT/RL axis: the design lists SFT/RL ratios `{90/10, 75/25, 50/50}`
> (Section 6) plus SFT-only RL controls (Section 7). "4 runs with different SFT
> %" is read here as those three ratios **plus the 100/0 supervised-only
> control** = four. If you instead meant the main-grid data fractions
> `{50%, 75%, 100%}`, flip the axis via `data.data_fraction` — see
> [Clarifications](#clarifications).

## Target hardware (CUDA)

The model is 5–30M params, so it fits on anything; the question is only price
and throughput. Recommendation:

- **SFT + GRPO sweeps (these four runs): a single RTX 4090 (24 GB).** It is the
  cheapest training-class card (~$0.20–0.52/hr), and a 5–30M model at ~3–6k HTE
  reactions trains in minutes — a 5-seed × 4-run sweep is well under a
  GPU-hour. This is the right card for everything in this repo today.
- **Full pretraining (Stages 1–2, once you add the corpora): one A100 80 GB**
  (~$1.09–1.79/hr) as the comfortable default. 1–2B effective tokens over
  ~8–12 epochs is a few A100-hours (~$5–10). The large VRAM lets you push batch
  size and keep the whole corpus in a persistent volume.
- **Skip H100.** Per the design, it isn't worth the premium below ~200M params —
  a 5–30M model can't saturate it.

Concretely: rent a 4090 for the four runs; add an A100 only when you bolt on
Stage 1/2 pretraining. Prefer per-minute/per-second billing and a persistent
volume for the corpus. Everything is CUDA-ready — pass `train.device: cuda`
(the default in the run configs) and it uses the GPU; `--device cpu` is the
local debug path.

## Data

Synthetic by default. For real Buchwald–Hartwig data:

1. Get `data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx` from
   [rxn4chemistry/rxn_yields](https://github.com/rxn4chemistry/rxn_yields).
2. In a run config set `data.synthetic: false` and
   `data.bh_xlsx: <path>`; pick `data.sheet` (`FullCV_01..10` for the canonical
   random folds, `Test1..4` for additive-holdout OOD).
3. `pip install -e ".[chem]"` for the pandas/openpyxl reader (and RDKit if you
   enable `data.randomize_smiles_prob`).

Column mapping lives in `data/dataset.py` (`_BH_COLUMNS`); adjust if your
release names columns differently.

## Layout

```
src/coffee_transformer/
  data/        SMILES tokenizer, slot schema, HTE dataset + BH loader, augment, collate, synthetic
  models/      embeddings, (typed) attention, transformer block, recurrent-depth encoder, heads
  losses/      two-hot + multi-scale CE, moment matching, pairwise, deep-supervision combiner
  training/    splits, builder, SFT (Stage 3), GRPO (Stage 4), MLM pretrain (Stages 1–2)
  eval/        R²/MAE/Spearman, test-time-compute sweep
  utils/       config (YAML→dataclasses), seed, device
configs/       base.yaml (annotated) + the four run files
scripts/       run_sft.py, run_grpo.py, sweep_sft_ratio.py
tests/         tokenizer, model, losses, end-to-end pipeline
docs/          PIPELINE.md — maps every design section to the code
```

## What's stubbed vs. real

**Real and runnable now:** recurrent-depth encoder (prelude → weight-tied core
×r → coda) with noise-init state, mandatory `(s+h)` input injection, randomized
`r`, truncated BPTT, deep supervision; the full distributional loss stack;
SFT with linear-probe→unfreeze and split encoder/head LRs; GRPO (tolerance and
ranking rewards, group-relative advantage, optional KL); MLM pretraining loop;
the TTC eval sweep; both typed-attention arms.

**Stubbed / needs your data + judgment:** the actual PubChem/USPTO/ORD corpora
and their hygiene (RDKit canonicalization, dedup, salt stripping, `[UNK]` and
test-leakage checks); heuristic agent-role tagging for USPTO; the
"differ-in-exactly-one-component" pair sampler (currently random within-batch);
Suzuki multi-task heads; adaptive halting (ACT). Each is marked with a `NOTE:`
in code and detailed in `docs/PIPELINE.md`.

## Clarifications

A few choices I made that you may want to redirect — see the end of
`docs/PIPELINE.md` for the full list. The main one is the four-run axis
(SFT/RL ratios + 100/0 control vs. main-grid data fractions), described above.
