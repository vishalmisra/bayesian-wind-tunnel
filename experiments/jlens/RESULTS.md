# The Global Workspace Inside a Bayesian Wind Tunnel — Results Memo

**Status:** all phases complete (0–5). **Models:** bijection transformer
(`logs/bijection_v20_repl`, 2.69M, sep-vocab wq-variant), MLP control
(2.68M, trained to its 1.107-bit plateau), K=5 loss-horizon recurrence
models (3 seeds) + full-horizon control. **Reference:**
anthropics/jacobian-lens @ `581d3986`; deviations in `DEVIATIONS.md`.
**Compute:** ~2 GPU-hours total on one RTX 4090 (vs 60–120 budgeted; the
cost estimate assumed Jacobian sweeps are expensive — at 2.7M params with
Gram accumulation they are ~10 s per model).

## Preregistered scorecard

| Pred | Claim | Threshold | Outcome |
|------|-------|-----------|---------|
| P1a | J-space ≈ frame subspace | overlap ≥ 3× null in some band | **PASS (refined)** — 4.8–6.8× (embedding-mode frame, emb/block-0 band, all r, both reductions); head-pullback modes pass only at r=4 (3.5–4.6×). Random-init and MLP controls: 0.9–1.2× (exactly null). Horizon-robust (min_horizon=2 unchanged). |
| P1b | J-directions decode hypothesis identity | ≥ 90% | **PASS trivially (weak alone)** — 100%; but random-init also decodes 100%, so P1b carries evidence only jointly with P1a's trained-vs-control contrast. |
| P2 | L0 frame head is the dominant writer | ≥ 2× every other head | **FAIL → distributed-writer result** — exactly the spec's falsification branch. No single dominant writer; indispensability itself is distributed (all six L0 heads ≈ 0.11–0.15 bits ΔMAE). The replacement result is sharper: the 9 indispensable heads (L0 bank + L1H0/H1/H3) separate from the other 27 with **zero overlap** in absolute frame-projected write (min 2.06 vs max 0.75). Metric lesson: the ratio form ‖P_J w‖/‖w‖ is gamed by near-dead heads (ρ = −0.40 with ablation); the absolute frame-projected norm is the honest writer metric (ρ = +0.47, p=0.004). |
| P3 | J-contents decode the surviving set | ≥ 95% (balanced) | **PASS** — 0.977 at r=24, 0.998 at r=32 (layer 1); full-residual ceiling 1.000. The r-dependence (0.90→0.95→0.98→1.00 through r=16→32) is itself confirmatory: the workspace needs ~V=20 dimensions. Raw accuracy is uninformative here (base rate ≈ 0.9 late); balanced accuracy required. |
| P4 (swap) | J-swap redirects the posterior | ≥ 1 bit margin, ≥ 200 pairs | **PASS** — +2.86 bits (J r=16, emb); frame swap **+8.17 bits = full-residual ceiling** (complete redirect); random-subspace control −6.7 (no redirect). |
| P4 (axis) | Entropy readout is causally inert | calibration intact | **PASS** — axis ablation ΔMAE −0.001 bits (axis R² = 1.00 — perfectly decodable, causally nothing) vs frame ablation +0.75 bits. The Paper-III frame–precision boundary replicates inside the wind tunnel. Discharges the backlog (a,b)-probe-ablation question: probe-readable ≠ causally used. |
| P5 | Workspace collapses past the K=5 horizon | post-horizon overlap ≤ null | **FAIL → REVERSE FINDING (preregistered as reportable)** — calibration gap replicates exactly (0.005–0.11 bits in-horizon → 1.3–1.8 past) but the frame-aligned J-space persists at 4.8× null past the horizon (control: 6.7×). The discriminator: last-layer next-token decode collapses 0.77 → 0.19 across the boundary (control: 1.00 → 1.00). |

## The one-paragraph story

Anthropic's J-lens, run inside a wind tunnel with analytic ground truth,
finds that the emergent workspace **is** the hypothesis frame — in
token-identity coordinates, at the embedding/block-0 band, with clean
learned-vs-structural controls (P1). Its contents are the posterior's
surviving hypothesis set (P3), its content is causally load-bearing while
the uncertainty readout riding on it is causally inert (P4), and it is
written not by one head but by a coherent bank — the L0 heads plus three
L1 heads — whose frame-projected write magnitude separates load-bearing
from epiphenomenal heads with zero overlap (P2-refined). The formation
account: the workspace is in place early (already above the P1 threshold
at step 10k, converged by 30k) while calibration is still improving 2.4×
(P5-dynamics). And the sharpest result: at the K=5 compilation boundary,
the workspace *geometry* survives where the *computation* dies — frame
overlap stays at 4.8× null past the horizon while last-layer decode and
calibration collapse together. The compilation boundary of gradient
descent lives in the writers, not the workspace. The workspace is a
global routing substrate; what is positionally compiled is the machinery
that fills it.

## Mechanistic bonus findings

1. **The causal injection window closes after layer 1.** Evidence-region
   patches at blocks ≥ 1 have zero effect on the readout (swap margins
   pinned at the unpatched baseline, all bases including full-residual).
   Independently confirms P2's writer localization (L0 bank + L1) by a
   different method.
2. **Writer concentration is task-variant-dependent (reconciled).**
   The production checkpoint distributes indispensability across the L0
   bank (H3 0.150, H4 0.140, H2 0.135, H5 0.114 bits); a paper-grade
   retrain of Paper I's query-token variant reproduces the single-head
   claim (2.74× concentration). One phenomenon, two supervision
   geometries — see post-review addition 3.
3. **Key-orthogonality does not identify the frame head** (picks H1;
   ablation says H3). Ablation is the criterion that tracks function.
4. **Late-layer J-spaces are structurally degenerate** for strictly-future
   targets (causality: position p′ logits read only h_final at p′), so
   workspace claims are inherently early/mid-layer claims.
5. **`extrap_K5_density*` checkpoints are NOT clean loss-horizon models**
   (`extra_loss_density` gives 1–25% post-horizon supervision; they are
   calibrated everywhere). The Nature-paper models are
   `results/extrapolation/horizon_integer_seed*`. The density series is a
   ready-made dose-response follow-up (does partial supervision restore
   the writers?).

## Post-review additions (2026-07-07)

1. **Dose-response (P5 graded law).** Across `extra_loss_density`
   0%→1%→5%→25% (3 seeds each): post-horizon calibration MAE
   1.64→0.54→0.014→0.005 bits, last-layer decode 0.19→0.68→1.00→1.00,
   workspace ratio 4.8→4.8→7.6→7.4× null. Supervision density buys writer
   competence; the workspace never needed buying. Artifacts:
   `phase4_density{1,5,25}`.
2. **Uncensored formation curve.** Dense-checkpoint retrain
   (`logs/bijection_v20_dense`, checkpoints from step 100): frame ratio
   2.3× at step 300, crosses 3× at ~400, plateaus ~6.3× by 2,500;
   calibration improves another ~40× afterwards (0.03→0.0008 bits).
   Artifacts: `phase5_dense`.
3. **Frame-head reconciliation (RESOLVED — variant-dependent).** Both
   surviving paired-variant checkpoints were undertrained side runs (MAE
   7.6 / 1.25 bits); a paper-grade retrain (`logs/bijection_v20_paired_v2`,
   0.017 bits) settles it: on Paper I's variant the single-head claim
   REPLICATES (L0H2 at 2.74× second, 1 head above half-top) while the
   production sepvocab variant is DISTRIBUTED (1.13×, 9 heads above
   half-top). Writer concentration tracks readout demand (one query site
   → one binding head; readout at every key position → the L0 bank).
   Paper I needs no correction; the note states the reconciliation.
   Caveat: paired-variant ablation effect is mild in absolute terms
   (0.0026 bits on 0.0169 baseline). Gotcha recorded: paired-format
   supervision must put value labels at KEY positions (next-token
   convention); same-position labels teach copying (0.8-bit plateau).
4. **Note edits:** circularity-disarming sentence (cotangent restriction
   = output vocabulary = Anthropic's own definition); weight-sharing
   paragraph (geometry global because the parameterization cannot make it
   local — connects P5 to the Nature thesis).

5. **Writer corruption (Phase 7) — double dissociation.** Norm-preserving
   rotations at in-horizon positions: complement-rotation at block 3
   collapses prediction (1.00 -> 0.09-0.16) while frame-rotation spares it
   (0.98-0.99), all seeds + control. Depth profile mirrors the rescue:
   computation migrates into the complement as writers work. Artifacts:
   `phase7_corruption`.
6. **Mid-training causal function (Phase 8).** P4 frame-swap margin
   crosses +1 bit at steps 400-500 (as the frame crosses the P1
   threshold; calibration still 0.65 bits), grows monotonically to +11.5
   bits at 50k; corruption destroys from the first functional checkpoint;
   load consolidates into frame coordinates (frame-rotation 2.3x more
   destructive than complement by 30k). The workspace functions as soon
   as it forms. Artifacts: `phase8_midtraining`.

## Decision recommendation

The results support a **standalone note** positioned as the first
ground-truth grounding of the global-workspace finding, with the P5
reverse finding as the headline novelty (workspace ≠ compilation
boundary; geometry global, writers positional) and P2's
distributed-but-dichotomous writer structure as the second contribution.
An Act-3 extension of the unified paper would bury the P5 result, and a
blog pairing wastes the preregistration. Suggested title register:
"The Workspace Survives Where the Circuit Dies."

If a note is pursued, two cheap additions strengthen it:
dense-early-checkpoint retrain for the formation curve (~1 GPU-hour) and
the density dose-response on P5 (~30 min).

## Artifacts

- Figures: `results/figures/phase{1..5}_*.png`
- Summaries: `results/phase*_summary.json` (per-cell grids in the
  the GPU workstation artifacts dirs, `experiments/jlens/artifacts/`)
- Runners: `run_phase{0..5}.py`; library: `extract.py`, `subspaces.py`,
  `metrics.py`, `connectivity.py`, `interventions.py`, `calibration.py`,
  `models.py`, `data_gen.py`
- Provenance: `DEVIATIONS.md` (vs anthropics/jacobian-lens @ 581d3986)
