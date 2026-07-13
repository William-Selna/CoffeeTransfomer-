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
end-to-end on CPU today (29 unit tests pass, and `make preflight` + `make smoke`
run the whole `prepare → pretrain → probe-select → 4-fork → USPTO` pipeline on
synthetic data). What remains before
paper-grade results is wiring the *real* corpora — see [Data](#data) and
[What's stubbed](#whats-stubbed-vs-real).

---

## Run it

Everything is driven by a Makefile (`make help` for all targets).

```bash
make setup        # pip install -e ".[chem,dev]"
make preflight    # env + plumbing check → GO / NO-GO (safe, CPU, ~1 min)
make smoke        # full pipeline on synthetic data, CPU — the debug-first gate

# on the H100 box (open network):
make fetch        # download PubChem + USPTO (BH already vendored)
make reactions    # USPTO → slot-tagged data/raw/reactions.jsonl
make prepare      # clean + pre-tokenize corpus → data/processed/
make mps          # start CUDA MPS (co-schedule the 4 runs)
make pretrain     # the 2×2 grid, four runs in parallel
make select       # probe all four, fork the winner into the 4 SFT/RL runs
# make all        # reactions → prepare → pretrain → select in one go
```

`make preflight` and `make smoke` both pass today (CPU), so the plumbing is
verified end-to-end before you spend a GPU-hour.

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

The plan is a **2×2 pretraining grid** — `{gelu, swiglu} × {seed 0, seed 1}` = four
candidate encoders — so the FFN comparison is disentangled from seed noise. The
gated SwiGLU FFN is param-matched to GELU (2/3·d_ff hidden width).
`select_and_sweep.py` probes all four on HTE (the design's "probe-only column"),
reports the per-activation mean over seeds, and forks the single best encoder
into the four SFT/RL runs. All four pretrainings fit in parallel on one H100 (see
[capacity](#target-hardware-cuda)).

```bash
# 0. gather data (BH is vendored; PubChem/USPTO fetch on the GPU box — see Data)
python scripts/fetch_data.py --pubchem --limit 10000000 --uspto --ord

# 1. once: build the shared tokenizer + cleaned/pre-tokenized corpus
python scripts/prepare_corpus.py \
    --pubchem data/raw/pubchem.smi --limit 10000000 \
    --reactions data/raw/reactions.jsonl \
    --hte data/hte/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx --sheet FullCV_01 \
    --out data/processed

# 2. the four candidates (run concurrently on one H100 — use CUDA MPS)
for c in gelu_s0 gelu_s1 swiglu_s0 swiglu_s1; do
    python scripts/pretrain.py --config configs/pretrain_$c.yaml &
done; wait

# 3. probe all four, fork the winner into the four SFT/RL runs
python scripts/select_and_sweep.py     # defaults to the 2x2 encoder dirs
```

Offline smoke (no downloads, CPU) — same flow at toy scale, two of the four:

```bash
python scripts/prepare_corpus.py --synthetic --out data/processed
for c in gelu_s0 swiglu_s0; do
  python scripts/pretrain.py --config configs/pretrain_$c.yaml --synthetic --device cpu \
    --stage1-epochs 1 --stage2-epochs 1 --num-workers 0 --batch-size 32
done
python scripts/select_and_sweep.py --encoders runs/pretrain_gelu_s0 runs/pretrain_swiglu_s0 \
    --device cpu --epochs 1 --probe-epochs 1 --batch-size 32
```

The encoder is ~11M params (the design's 10–15M recurrent-depth point); grow
`model.d_model` / `core_layers` for other scale points — the SFT forks inherit
the dims from the checkpoint automatically.

**How pretraining is judged.** Two levels: the **held-out MLM validation loss**
(intrinsic convergence/health, logged per stage) and the **HTE linear-probe R²**
(extrinsic — freeze the encoder, train only a head, measure yield R²). You
*select* on the probe R² (averaged over a couple of seeds, so the pick isn't
probe-init noise), because a low MLM loss doesn't guarantee good transfer.

**Robustness (it's meant to be fire-and-forget).** The pretraining trainer has
warmup+cosine LR, a divergence guard (non-finite / exploding loss aborts with a
message instead of burning hours), a rolling `encoder_latest.pt` for crash
recovery, and saves the **best-by-val** encoder as the canonical `encoder.pt`.
So `pretrain.py` → `select_and_sweep.py` is a single unattended chain.

## Target hardware (CUDA)

The model is ~11–30M params, so it fits on anything. Plan of record: **a single
H100 (80 GB) runs all four pretrainings concurrently.**

**Capacity.** One ~11M pretraining run costs ~0.2 GB of weights/optimizer state;
the rest is activations, which scale with `batch × seq²` (attention) × the ~16
retained layer-passes (4 core layers × `truncated_bptt_k` + prelude). At realistic
SMILES lengths (~60–150 tokens) that's ~3–8 GB/run, so **8–12+ runs fit in 80 GB**
— the 2×2 grid uses four. More importantly, an 11M model can't saturate the
H100's compute (it's kernel-launch bound on tiny ops), so co-scheduling the runs
*raises* aggregate throughput rather than just time-slicing. Launch them as
separate processes and enable **CUDA MPS** for clean concurrency.

- **H100, not H200.** The H200 is the *same Hopper compute* as the H100 (same
  ~990 dense BF16 TFLOP/s) with more memory you won't use at this scale — zero
  gain for more money.
- Push `batch_size` up (you have headroom) to improve per-run GPU utilization.
- Measure the real footprint on your box with `nvidia-smi` while a run is live —
  the numbers above are analytical estimates (I have no GPU here to profile).

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

### Crude USPTO SFT (optional, coarse 4-bin)

Raw USPTO yields are too noisy for fine regression (design §2), but a **coarse
4-bin (25%-wide) target launders that noise** into a usable weak signal.
`configs/run_uspto_crude.yaml` runs it: `dataset: USPTO`, `model.num_bins: 4`,
kept in its own `dataset="USPTO"` so labels are never blended with HTE. It's a
weak-signal experiment, not a replacement for the HTE SFT.

```bash
python scripts/run_sft.py --config configs/run_uspto_crude.yaml   # synthetic offline
# real: set data.synthetic false + data.uspto_path <rsmi>; optionally
# pretrained_encoder: runs/pretrain_swiglu_s0  (head auto-resizes to 4 bins)
```

## Layout

```
src/coffee_transformer/
  data/        SMILES tokenizer, slot schema, HTE dataset + BH/USPTO loaders,
               augment, collate, synthetic, corpus (hygiene/pre-tokenization), tagging
  models/      embeddings, (typed) attention, transformer block, recurrent-depth encoder, heads
  losses/      two-hot + multi-scale CE, moment matching, pairwise, deep-supervision combiner
  training/    splits, builder, SFT (Stage 3), GRPO (Stage 4), MLM pretrain +
               pretrain_pipeline (Stages 1–2), checkpoint (encoder transfer), probe (selection)
  eval/        R²/MAE/Spearman, test-time-compute sweep
  utils/       config (YAML→dataclasses), seed, device
configs/       base.yaml + four SFT/RL runs + run_uspto_crude + pretrain_{gelu,swiglu}_s{0,1}
scripts/       fetch_data.py, build_reactions_jsonl.py, prepare_corpus.py,
               pretrain.py, select_and_sweep.py, run_sft.py, run_grpo.py,
               sweep_sft_ratio.py, preflight.py, make_tarball.sh
Makefile       one-command orchestration (make help)
data/          hte/ (vendored BH) + README (sources/licenses)
tests/         tokenizer, model, losses, pipeline, pretrain, tagging
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
