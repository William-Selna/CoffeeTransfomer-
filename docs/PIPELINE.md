# Design → code map

Every section of the July 2026 design summary and where it lives in this
scaffold. Section numbers refer to that document.

## §3 Recurrent-depth architecture
`models/recurrent_depth.py`, `models/config.py`

- Prelude → weight-tied core (×r) → coda: `RecurrentDepthEncoder`
  (`prelude`, `core` shared across iterations, `coda_norm` + pooling).
- State init with noise: `s = torch.randn_like(h) * cfg.state_init_std`.
- **Mandatory input injection** `(s + h)` every step: `cfg.input_injection`.
- Randomize `r` in training: `_sample_r` over `[r_min, r_max]`.
- **Truncated BPTT** — only the last `truncated_bptt_k` iterations carry grad:
  the first `r - k` run inside `torch.no_grad()`.
- Vanilla-transformer baseline: `cfg.recurrent = False` (core runs once, no
  injection, no noise) — the matched-parameters comparison column.
- Matched-FLOPs comparison: set a deeper/wider vanilla `core_layers` so its
  single pass ≈ the recurrent model's `r` passes (Section 7 requires both
  columns).
- Test-time compute: `eval/ttc.py` sweeps `r ∈ {1,2,4,8,16}`.
- FFN activation (`models/transformer.py::FeedForward`): pointwise
  `gelu|relu|silu` or gated `swiglu|geglu` (param-matched via 2/3·d_ff hidden).
  A smooth/gated nonlinearity is the safer choice for the weight-tied core; the
  two pretrain configs use this axis (A=gelu, B=swiglu).
- **Stubbed:** adaptive halting (ACT / convergence exit) — hook would go in
  `RecurrentDepthEncoder.forward` to break early per-example.

## §4 Input representation & tokenization
`data/tokenizer.py`, `data/slots.py`, `data/augment.py`, `models/attention.py`

- Regex SMILES tokenizer (~300 vocab, keeps `Br`/`Cl`/`[nH]`): `SmilesTokenizer`.
- Slot tokens `[LIG]`/`[BASE]`/`[ADD]`/… + learned slot-type embedding added to
  every token in the span: `slots.py` + `ReactionEmbedding.slot_type`.
- Typed-attention ablation: `cfg.typed_attention` = `embedding` (shared QKV) vs
  `per_slot_kqv` (`TypedLinear`, per-slot Q/K/V).
- Missing slots via masking, never zero vectors: attention adds `-inf` to pad
  key logits before softmax (`MultiHeadSelfAttention`); absent components simply
  aren't emitted as tokens.
- Augmentation: `randomize_slot_order` (implemented), `randomize_smiles`
  (RDKit; no-op passthrough without it).
- **Stubbed:** per-dataset heads / dataset-conditioning token for Suzuki
  multi-task (schema in `slots.py` ready; heads not yet split).

## §5 Supervision design
`losses/`

- Two-hot distributional targets + CE: `distributional.two_hot_targets`,
  `soft_cross_entropy`.
- Multi-scale pooled CE: `distributional.multi_scale_ce` (`loss.multi_scale_bins`).
- Moment matching (mean, optional variance): `moments.moment_matching_loss`.
- Deep supervision across iterations: `combined.deep_supervision_weights` +
  `YieldLoss` (loss applied to every read-out iteration).
- Pairwise-difference task: `pairwise.pairwise_difference_loss`.
  **NOTE:** samples random within-batch pairs; the design's stronger
  "differ-in-exactly-one-component" variant needs the data pipeline to emit a
  neighbor index. Cheap default for now.
- Masked-component reconstruction: available via the MLM path (`pretrain.py`,
  span-masking); the design notes it saturates on BH alone, so it's most useful
  in the no-pretraining ablation.
- Loss weighting: fixed λ (`lambda_*`), swept by config. Annealed/uncertainty
  weighting not implemented (design calls fixed λ "usually sufficient").
- Every term is a config flag → the loss-term knockout table is a pure sweep.

## §6 Four-stage pipeline
`training/`

- Stage 1 molecular-grammar MLM (PubChem): `pretrain.MLMTrainer` (uniform
  `mask_tokens`, BERT 80/10/10) + `MoleculeMLMDataset`.
- Stage 2 reactivity MLM (USPTO+ORD), span-masking slot contents:
  `pretrain.span_mask_tokens` (mask entire slot spans). The corpus hygiene
  (`data/corpus.py`: canonicalize, salt-strip, dedup, `[UNK]`/leakage checks)
  and pre-tokenized mmap store are implemented; **you supply the raw bytes** and
  the heuristic USPTO→slot-tagged `reactions.jsonl` tagging.
- Orchestration + hardening: `pretrain_pipeline.run_pretraining` runs Stage 1→2
  on one MLMModel with per-stage **held-out MLM val**, warmup+cosine LR, a
  **divergence guard** (`DivergenceError` on non-finite/exploding loss), a
  rolling `encoder_latest.pt` for crash recovery, and **best-by-val** saved as
  the canonical `encoder.pt`. `scripts/pretrain.py` catches divergence and exits
  with a recovery hint.
- Evaluation criteria: intrinsic = held-out MLM val loss (health); extrinsic =
  HTE linear-probe R² (the selection metric). You pick on the probe, not the
  MLM loss — they can disagree.
- Two-encoder selection: `scripts/pretrain.py` ×2 (A=gelu / B=swiglu) →
  `training/probe.linear_probe_score` averaged over `--probe-seeds` →
  `scripts/select_and_sweep.py` picks the winner and forks it.
- Stage 3 SFT: `sft.SFTTrainer` — fresh histogram head, linear-probe→unfreeze,
  encoder LR = `head_lr × encoder_lr_scale`. Pooling chosen once (`cfg.pool`)
  and shared across stages. Pretrained encoder loaded via
  `builder.load_pretrained_bundle` (rebuilds exact dims, reuses shared vocab).
- Stage 4 GRPO: `grpo.GRPOTrainer` — k samples/reaction, group-relative
  advantage (no value net), tolerance or ranking reward, optional KL to a frozen
  reference.
- Build-in-reverse: the SFT-from-scratch baseline is the default runnable path
  (no `pretrained_encoder`); pretraining bolts underneath by pointing a fork at
  an encoder checkpoint dir.
- Splits (~60/20/20 supervised/RL/test): `training/splits.py`.

## §7 Experimental design
- SFT/RL ratio axis → the four `configs/run_sft*_rl*.yaml`, driven by
  `scripts/sweep_sft_ratio.py`.
- Main grid data fractions {50,75,100}% → `data.data_fraction`.
- Architecture columns {vanilla, recurrent matched-params, recurrent
  matched-FLOPs} → `model.recurrent` + `core_layers`/`train_r`.
- Pipeline ablation {scratch vs pretrained} × {SFT vs SFT+GRPO} → load or skip a
  pretrained encoder; `grpo.enabled`.
- Eval splits: `data.sheet` = `FullCV_*` (random) vs `Test*` (additive-holdout OOD).
- 5 seeds → `--seeds` in the sweep script; metrics carry Spearman for ranking.

## §8 Compute
- CUDA target: `utils/device.py`, `train.device: cuda`. CPU path for local debug.
- GPU recommendation and cost notes: README "Target hardware".

---

# Clarifications / choices to confirm

1. **The four-run axis — SETTLED.** SFT/RL ratios `{100/0, 90/10, 75/25,
   50/50}` (`configs/run_sft*_rl*.yaml`), forked from the winning pretrained
   encoder.
2. **Pretraining runs — SETTLED.** Two candidates, 5M vs 10M PubChem molecules
   (`configs/pretrain_5m.yaml` / `pretrain_10m.yaml`), identical otherwise;
   winner chosen by HTE linear-probe R².
3. **Scale — SETTLED at ~11M.** Pretrain configs use `d_model=384`,
   prelude 2 / core 4 (the design's 10–15M recurrent-depth point). Bump
   `d_model`/`core_layers` for other scale-sweep points; the SFT forks inherit
   the encoder's dims from the checkpoint automatically.
4. **Histogram bins.** Defaulted to 20 bins over 0–100% (5% width), the "~20
   buckets" in §5. Confirm bin count / whether yields can exceed 100 (two-hot
   clamps to edges if so).
5. **GRPO reward default.** Set to `tolerance` (smooth, low-variance). The
   `ranking` reward is implemented but noisier; which should be the headline?
6. **Real corpus bytes.** `prepare_corpus.py` implements the hygiene + packing;
   you supply raw PubChem `.smi`, a slot-tagged `reactions.jsonl` (from USPTO/ORD
   parsing), and the BH `.xlsx`. If your BH copy's headers differ from
   `_BH_COLUMNS` in `data/dataset.py`, share a header row and I'll fix it.
