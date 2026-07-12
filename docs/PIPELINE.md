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

- Stage 1 molecular-grammar MLM (PubChem): `pretrain.MLMTrainer` +
  `MoleculeMLMDataset` + `mask_tokens` (BERT 80/10/10).
- Stage 2 reactivity MLM (USPTO+ORD), span-masking slot contents: same trainer;
  **stubbed** = the corpora, hygiene, and heuristic agent-role tagging.
- Stage 3 SFT: `sft.SFTTrainer` — fresh histogram head, linear-probe→unfreeze,
  encoder LR = `head_lr × encoder_lr_scale`. Pooling chosen once (`cfg.pool`)
  and shared across stages. `MLMModel.transfer_encoder_to` copies pretrained
  weights in.
- Stage 4 GRPO: `grpo.GRPOTrainer` — k samples/reaction, group-relative
  advantage (no value net), tolerance or ranking reward, optional KL to a frozen
  reference.
- Build-in-reverse: the SFT-from-scratch baseline is the default runnable path;
  pretraining bolts underneath by loading an encoder checkpoint.
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

1. **The four-run axis.** Implemented as SFT/RL ratios `{100/0, 90/10, 75/25,
   50/50}` (three ratios from §6 + the §7 supervised-only control) = 4 runs
   "with different SFT %". If you meant the §7 main-grid **data fractions**
   `{50%, 75%, 100%}` instead (that's 3, not 4), tell me the fourth point and
   I'll switch the sweep to `data.data_fraction`.
2. **Histogram bins.** Defaulted to 20 bins over 0–100% (5% width), the "~20
   buckets" in §5. Confirm bin count / whether yields can exceed 100 (they
   shouldn't post-cleaning, but the two-hot clamps to edges if so).
3. **Scale anchor.** Defaults are the 5M-param anchor (`d_model=256`,
   prelude 2 / core 4). The 10–15M and 30M scale-sweep points and the
   matched-FLOPs vanilla config aren't pre-written — want me to add config
   files for those three scale points?
4. **GRPO reward default.** Set to `tolerance` (smooth, low-variance). The
   `ranking` reward is implemented but noisier; which should be the headline?
5. **Pretraining corpora.** Stages 1–2 are scaffolded but need real
   PubChem/USPTO/ORD data + hygiene. Out of scope for "scaffold the SFT runs" —
   confirm before I build the ingestion/cleaning pipeline.
6. **Real BH ingestion.** `load_buchwald_hartwig` targets the documented column
   layout of `Dreher_and_Doyle_input_data.xlsx`. If your copy differs, share a
   header row and I'll fix `_BH_COLUMNS`.
