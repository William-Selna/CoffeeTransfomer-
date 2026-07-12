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

The plan is to pretrain **two identical** candidate encoders (same config,
different seed — pretraining is fickle at this scale), keep whichever probes
better on HTE (the design's "probe-only column"), then fork that winner into
the four SFT/RL runs.

```bash
# 0. gather data (BH is vendored; PubChem/USPTO fetch on the GPU box — see Data)
python scripts/fetch_data.py --pubchem --limit 10000000 --uspto --ord

# 1. once: build the shared tokenizer + cleaned/pre-tokenized corpus
python scripts/prepare_corpus.py \
    --pubchem data/raw/pubchem.smi --limit 10000000 \
    --reactions data/raw/reactions.jsonl \
    --hte data/hte/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx --sheet FullCV_01 \
    --out data/processed

# 2. the two identical candidates (seed 0 and seed 1)
python scripts/pretrain.py --config configs/pretrain_a.yaml
python scripts/pretrain.py --config configs/pretrain_b.yaml

# 3. probe both, fork the winner into the four SFT/RL runs
python scripts/select_and_sweep.py --encoders runs/pretrain_a runs/pretrain_b
```

Offline smoke (no downloads, CPU) — same flow at toy scale:

```bash
python scripts/prepare_corpus.py --synthetic --out data/processed
python scripts/pretrain.py --config configs/pretrain_a.yaml --synthetic --device cpu \
    --stage1-epochs 1 --stage2-epochs 1 --num-workers 0 --batch-size 32
python scripts/pretrain.py --config configs/pretrain_b.yaml --synthetic --device cpu \
    --stage1-epochs 1 --stage2-epochs 1 --num-workers 0 --batch-size 32
python scripts/select_and_sweep.py --encoders runs/pretrain_a runs/pretrain_b \
    --device cpu --epochs 1 --probe-epochs 1 --batch-size 32
```

The two pretrain configs are **identical except `seed`** — the probe just picks
the healthier of two runs. To make one an algorithm variant instead, change a
single `model` knob in `pretrain_b.yaml` (`truncated_bptt_k` = backprop
strength, or `activation`) and leave the rest matched. The encoder is ~11M
params (the design's 10–15M recurrent-depth point); grow `model.d_model` /
`core_layers` for other scale points — the SFT forks inherit the dims from the
checkpoint automatically.

## Target hardware (CUDA)

The model is ~11–30M params, so it fits on anything. Plan of record: **a single
H100 (80 GB), both pretrainings run serially.** Wall-clock isn't a constraint
here, so there's no need to parallelize across cards — the two pretrainings plus
the four SFT/RL forks finish comfortably in a couple of hours, and the SFT+GRPO
forks are minutes and pennies on top.

- **H100, not H200.** The H200 is the *same Hopper compute* as the H100 (same
  ~990 dense BF16 TFLOP/s) with more memory (141 GB vs 80 GB). At ~11M params on
  short SMILES you use a fraction of 80 GB and aren't memory-bound, so H200 buys
  nothing over H100 for more money.
- One card is plenty: ~11M params leaves the 80 GB almost empty, so push
  `batch_size` up and both pretrainings still fit back-to-back.

At this scale the GPU is rarely the bottleneck — the input pipeline is. A tiny
model starves on a naive dataloader, so keep the perf knobs on: bf16
(`amp: true`), `torch.compile` (`compile: true`), a large `batch_size`,
`num_workers`, and the pre-tokenized mmap corpus that `prepare_corpus.py` writes
(zero Python tokenization at step time). All are already wired in the pretrain
configs. Use a persistent volume for the corpus. `--device cpu` is the local
debug path.

## Data

The **Buchwald–Hartwig HTE dataset is vendored** in
`data/hte/Buchwald-Hartwig/` (~2 MB, 3,955 reactions, 16 sheets: `FullCV_01..10`
random folds + `Test1..4` additive-holdout OOD). To train on it instead of the
synthetic toy data, set `data.synthetic: false` in a run config — the default
`data.bh_xlsx` already points at the vendored file — and `pip install -e
".[chem]"` for the pandas/openpyxl reader.

Everything else is fetched with one script:

```bash
python scripts/fetch_data.py --all          # BH (already vendored) + PubChem + USPTO + ORD
python scripts/fetch_data.py --pubchem --limit 10000000 --uspto   # the big corpora only
```

**Network caveat:** from a Claude session, PubChem (NCBI) and USPTO (figshare)
are blocked by the proxy — only GitHub-hosted data (BH, ORD) downloads here. Run
the PubChem/USPTO fetch on the H100 box (open network). Full source table,
licenses, and sizes are in [`data/README.md`](data/README.md). Column mapping
for the BH reader is `_BH_COLUMNS` in `data/dataset.py`.

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
configs/       base.yaml (annotated) + four run files + pretrain_a/pretrain_b
scripts/       fetch_data.py, prepare_corpus.py, pretrain.py, select_and_sweep.py,
               run_sft.py, run_grpo.py, sweep_sft_ratio.py
data/          hte/ (vendored BH) + README (sources/licenses)
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
