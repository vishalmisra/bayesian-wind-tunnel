# Substrate-first training (exploration spec)

**Status:** parked for exploration, 2026-07-08. Run when GPUs free up
after the review-revision battery. **Owner:** VM. **Est. compute:**
~2-4 GPU-hours for all pilots (wind tunnel scale).

## The idea

The jlens study established that Bayesian computation in these models
factorizes into a routing substrate and writer computation, with
asymmetric training economics:

- The substrate (embeddings + QK geometry) forms by step ~400, is
  causally functional as soon as it forms (Phase 8: swap margin crosses
  +1 bit at steps 400-500), and plateaus by ~2,500.
- The remaining ~95% of training buys writer precision inside the
  already-working substrate (calibration improves ~1000x after the
  substrate exists).
- The dose-response (Phase 4 density series): 1% supervision density
  recovers most writer function against a present substrate; 5% fully
  restores.
- The substrate is weight-defined and content-free (corruption: frame
  causally specific only at the entry band); writers are the expensive,
  position-by-position-compiled part.

If the separation is real engineering and not just interpretation,
three training interventions follow. Each is falsifiable in the wind
tunnel in minutes, with exact calibration (entropy MAE vs the analytic
posterior) as the metric.

## Pilot A: substrate transplant

Skip the formation phase by initializing from a donor's substrate.

- Donor: bijection transformer at step 2,500 (substrate formed,
  calibration mediocre ~0.03 bits).
- Recipient: fresh model; copy tok_emb, pos_emb, and all wq/wk (variant:
  also wv/wo) from the donor. Two arms: transplanted-frozen,
  transplanted-trainable. Baseline: from-scratch.
- Metric: steps to reach 0.01 bits MAE, and final MAE at matched steps.
- Prediction: transplanted arms skip the ~2.5k-step formation phase.
- Strong version: transfer ACROSS tasks. Same vocabulary, different
  in-context task (e.g., a fixed-offset cipher, or the paired/query
  variant already in the repo). If routing is content-free, substrate
  transfers; if not, the substrate is more task-laden than the
  corruption experiments suggest. Informative either way.

## Pilot B: substrate freezing

Cut late-training cost on parameters that stopped learning.

- Train normally to step 2,500 (frame plateau, measurable without
  ground truth via the frame-overlap ratio). Then freeze tok_emb,
  pos_emb, wq, wk (variant: + wv/wo) and train the rest to 50k.
- Metric: final MAE vs the 0.0008-bit full-training baseline; optimizer
  state / gradient FLOPs saved on frozen params.
- Risk our own data raises: the consolidation finding (load migrating
  INTO frame coordinates through step 30k, Phase 8) may require the
  frame to keep moving. If frozen matches baseline, consolidation is
  writer-side adaptation to a fixed frame and freezing is free. If it
  does not match, we have localized what late frame drift is for.
  Informative either way.

## Pilot C: sparse supervision after formation

Weaponize the dose-response as a schedule.

- Dense loss (all key positions) to step 2,500, then drop to 5% (and
  1%) loss density to 50k. Baseline: dense throughout.
- Metric: final MAE at matched steps.
- Honest caveat: loss masking saves LABELS, not FLOPs (forward/backward
  still run). The claim monetizes where supervision is the scarce
  resource: RLHF, human annotation, synthetic-data budgets. The
  wind-tunnel version establishes the principle; the pitch is "most
  supervision is spent buying precision inside a substrate a fraction
  of the signal would buy."

## The free measurement tool

The frame-overlap ratio (J-lens Gram sweep + embedding-frame projection,
`run_phase1.py` machinery) is computable on any model WITHOUT ground
truth: weights and inputs only. That makes "is the substrate formed?"
a monitorable training signal - the switch condition for A/B/C and a
training diagnostic in its own right. Worth one figure in any writeup:
frame-ratio vs step as the phase-change detector.

## Implementation notes

- All three pilots are flags/loaders on `train_dense_checkpoints.py`:
  parameter groups (B), state-dict partial load (A), loss-density
  schedule (C, reuse the extra_loss_density idea from the K=5 family).
- Cross-task arm of A needs a second task generator; the paired/query
  variant (`train_paired_variant.py`) is already in the repo and
  shares the vocabulary.
- Evaluate with `calibration.eval_entropy_mae` + the Phase-8 swap
  margin per checkpoint (causal functionality, not just calibration).
- Seeds: 3 per arm minimum; the review battery taught us that lesson.

## Relation to prior art (check before writing anything public)

- Freezing embeddings / partial-freezing schedules exist as folklore
  and in efficient-training literature; the novelty here is the
  PRINCIPLED switch condition (measured substrate formation) and the
  causal story for WHY the schedule should work.
- Progressive stacking / growth methods (grow depth after early
  training) are adjacent; substrate transplant across tasks is closer
  to embedding-transfer literature. The differentiator: we transplant
  the ROUTING geometry specifically and predict which downstream
  quantity (formation time vs final precision) it buys.
- The LoRA corollaries in the note (writer-layer adapters move content,
  late adapters move precision) are the fine-tuning-side siblings of
  these pretraining-side pilots; a joint writeup could cover both.

## Success criteria for promoting this to a real project

Pilot A or B shows >=2x reduction in steps-to-calibration (or matched
final MAE with substrate frozen), replicated on 3 seeds, plus the
cross-task transfer arm producing a clean answer in either direction.
Then this becomes its own note: the first engineering payoff of the
separation principle.
