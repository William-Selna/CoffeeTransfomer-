# CoffeeTransformer — recurrent-depth transformers for reaction-yield prediction

A PyTorch scaffold for the research plan in *Small-Scale Recurrent-Depth
Transformers for Chemical Reaction Yield Prediction* (July 2026 design summary).
It implements the recurrent-depth architecture, the distributional-supervision
loss stack, and the four-stage `pretrain → SFT → RL` pipeline at the ~5M-param
scale the plan targets — wired so the **four SFT/RL-ratio runs** are drivable
from config.

Status: **runnable scaffold**. The architecture, losses, SFT trainer, GRPO
trainer, the full two-stage pretraining pipeline, the encoder-transfer +
probe-selection flow, eval/TTC harness, and a synthetic data path all run
end-to-end on CPU today (20 unit tests pass, plus a full
`prepare → pretrain ×2 → probe-select → 4-fork` smoke run). What remains before
paper-grade results is wiring the *real* corpora — see [Data](#data) and
[What's stubbed](#whats-stubbed-vs-real).

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

> These four are the SFT/RL ratios `{90/10, 75/25, 50/50}` (Section 6) plus the
> `100/0` supervised-only control (Section 7). They all fork from the *same*
> winning pretrained encoder — see [Full pipeline](#full-pipeline-pretrain-2-encoders--pick-the-best--fork-the-4-runs).
> The main-grid data-fraction axis `{50%, 75%, 100%}` is orthogonal and
> available via `data.data_fraction`.

## Full pipeline: pretrain 2 encoders → pick the best → fork the 4 runs

The plan is to pretrain **two** candidate encoders (5M vs 10M PubChem
molecules), pick the better one by HTE linear-probe R² (the design's
"probe-only column"), then fork that winner into the four SFT/RL runs.

```bash
# 1. once: build the shared tokenizer + cleaned/pre-tokenized corpus
python scripts/prepare_corpus.py \
    --pubchem data/raw/pubchem.smi --limit 10000000 \
    --reactions data/raw/reactions.jsonl \
    --hte data/raw/Dreher_and_Doyle_input_data.xlsx --sheet FullCV_01 \
    --out data/processed

# 2. the two candidates (run on two GPUs in parallel to stay near ~1 hr)
python scripts/pretrain.py --config configs/pretrain_5m.yaml
python scripts/pretrain.py --config configs/pretrain_10m.yaml

# 3. probe both, fork the winner into the four SFT/RL runs
python scripts/select_and_sweep.py --encoders runs/pretrain_5m runs/pretrain_10m
```

Offline smoke (no downloads, CPU) — same flow at toy scale:

```bash
python scripts/prepare_corpus.py --synthetic --out data/processed
python scripts/pretrain.py --config configs/pretrain_5m.yaml  --synthetic --device cpu \
    --stage1-epochs 1 --stage2-epochs 1 --num-workers 0 --batch-size 32
python scripts/pretrain.py --config configs/pretrain_10m.yaml --synthetic --device cpu \
    --stage1-epochs 1 --stage2-epochs 1 --num-workers 0 --batch-size 32
python scripts/select_and_sweep.py --encoders runs/pretrain_5m runs/pretrain_10m \
    --device cpu --epochs 1 --probe-epochs 1 --batch-size 32
```

The two pretrain configs differ **only** in `pubchem_limit` (5M vs 10M), so the
probe comparison cleanly answers "is 2× the Stage-1 corpus worth the extra
pretraining compute?" The encoder is ~11M params (the design's 10–15M
recurrent-depth point) — grow `model.d_model`/`core_layers` in the configs to
hit other scale-sweep points.

## Target hardware (CUDA)

The model is ~11–30M params, so it fits on anything; the question is only price
and throughput. For the full plan (two pretrainings + the fork), the honest
recommendation given a ~1-hour wall-clock target:

- **Two H100s, one per pretraining run, in parallel.** Each 10M-molecule
  pretraining (1–2B effective tokens, 8–12 epochs) is roughly a couple of
  GPU-hours on an A100 and faster on an H100; running the 5M and 10M candidates
  side by side keeps wall-clock at ~1 hr instead of ~2 hr serial. The SFT+GRPO
  forks that follow are minutes and pennies — they don't need their own card.
- **H100, not H200.** The H200 is the *same Hopper compute* as the H100 (same
  ~990 dense BF16 TFLOP/s) with more memory (141 GB vs 80 GB) and bandwidth. At
  ~11M params on short SMILES you use a fraction of 80 GB and aren't
  memory-bound, so H200 buys **zero** wall-clock over H100 for more money.
- **One H100 is fine if you serialize** — expect ~2 hr for both pretrainings.
  A single A100 80 GB works too (the design's comfortable default) at lower cost
  and modestly slower.

Reality check on hitting 1 hr: at this scale the GPU is *not* the bottleneck —
the input pipeline is. A tiny model starves on a naive dataloader, so the perf
knobs matter more than the card: bf16 (`amp: true`), `torch.compile`
(`compile: true`), a large `batch_size`, `num_workers`, and the pre-tokenized
mmap corpus that `prepare_corpus.py` writes (zero Python tokenization at step
time). All are already wired in the pretrain configs. Prefer per-second billing
and a persistent volume for the corpus. `--device cpu` is the local debug path.

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
  data/        SMILES tokenizer, slot schema, HTE dataset + BH loader, augment,
               collate, synthetic, corpus (ingestion/hygiene/pre-tokenization)
  models/      embeddings, (typed) attention, transformer block, recurrent-depth encoder, heads
  losses/      two-hot + multi-scale CE, moment matching, pairwise, deep-supervision combiner
  training/    splits, builder, SFT (Stage 3), GRPO (Stage 4), MLM pretrain +
               pretrain_pipeline (Stages 1–2), checkpoint (encoder transfer), probe (selection)
  eval/        R²/MAE/Spearman, test-time-compute sweep
  utils/       config (YAML→dataclasses), seed, device
configs/       base.yaml (annotated) + four run files + pretrain_5m/10m
scripts/       prepare_corpus.py, pretrain.py, select_and_sweep.py,
               run_sft.py, run_grpo.py, sweep_sft_ratio.py
tests/         tokenizer, model, losses, pipeline, pretrain
docs/          PIPELINE.md — maps every design section to the code
```

## What's stubbed vs. real

**Real and runnable now:** recurrent-depth encoder (prelude → weight-tied core
×r → coda) with noise-init state, mandatory `(s+h)` input injection, randomized
`r`, truncated BPTT, deep supervision; the full distributional loss stack;
the two-stage MLM pretraining pipeline (uniform Stage-1 + slot-span Stage-2)
with bf16/compile/mmap perf knobs; encoder-checkpoint transfer + linear-probe
selection; SFT with linear-probe→unfreeze and split encoder/head LRs; GRPO
(tolerance and ranking rewards, group-relative advantage, optional KL); the
corpus hygiene functions (RDKit canonicalization, salt strip, dedup, `[UNK]`
and leakage checks); the TTC eval sweep; both typed-attention arms.

**Stubbed / needs your data + judgment:** the actual PubChem/USPTO/ORD *bytes*
(the hygiene *functions* exist; you supply the raw files to `prepare_corpus.py`);
heuristic agent-role tagging that turns raw USPTO into the slot-tagged
`reactions.jsonl` it expects; the "differ-in-exactly-one-component" pair sampler
(currently random within-batch); Suzuki multi-task heads; adaptive halting
(ACT). Each is marked with a `NOTE:` in code and detailed in `docs/PIPELINE.md`.

## Clarifications

The four-run axis (SFT/RL ratios) and the two-pretrain plan (5M vs 10M, pick by
probe R²) are settled and wired. A couple of smaller open choices — histogram
bin count and the GRPO headline reward (`tolerance` vs `ranking`) — are listed
at the end of `docs/PIPELINE.md`.
