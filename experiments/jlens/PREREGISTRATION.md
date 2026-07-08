# Preregistered experiment spec (verbatim)

**Provenance note (honest dating).** This spec was authored 2026-07-07,
before any experiment in this study ran. Its commit to the public
repository postdates the first runs, so it does not constitute an
externally timestamped preregistration; it is an internal
prespecification whose thresholds Table 1 of the note scores against.
Deviations are logged in `DEVIATIONS.md`. One infrastructure reference
has been genericized; the scientific content is unedited.

---

# Experiment Spec: The Global Workspace Inside a Bayesian Wind Tunnel

**Status:** ready for handoff. **Owner:** VM / NA / SRD. **Est. compute:** ~60–120 GPU-hours total (small models; the cost is in Jacobian sweeps, not training).

## 1. Objective

Anthropic's global-workspace paper finds a small, densely-connected subspace (the "J-space") that holds flexibly reusable, causally load-bearing content in production LLMs, identified via Jacobians from internal activations to future output logits. Their stated open question: what mechanism decides what enters the workspace, and how it forms.

Our wind tunnels have (a) a known ground-truth posterior at every position, (b) a fully characterized internal mechanism (hypothesis frame at Layer 0, elimination in middle layers, refinement late), and (c) a derived formation account (advantage-based routing under normalized competition). This experiment runs the J-lens inside the wind tunnel to test whether their emergent workspace and our hypothesis frame are the same object.

If the identification holds, we ground the workspace in a setting with analytic ground truth and supply the formation mechanism their paper lacks. If it fails, we learn the workspace is a scale phenomenon our small models don't exhibit, which bounds both claims.

## 2. Preregistered predictions

- **P1 (Identity).** The J-space at each position substantially coincides with the hypothesis-frame subspace (the span of Layer-0 key/value directions for active hypotheses). Threshold: mean principal angle overlap / CKA against the frame subspace ≥ 3× the null (random subspace of equal dimension), and top-k J-space directions decode hypothesis identity with ≥ 90% accuracy.
- **P2 (Writer attribution).** The Layer-0 hypothesis-frame head is the dominant *writer* to the J-space: its write connectivity (norm of its output projected into J-space, aggregated over positions) exceeds every other head's by ≥ 2×, mirroring its unique indispensability under ablation.
- **P3 (Contents = posterior support).** At position k, J-space contents decode the *surviving* hypothesis set of the analytic posterior: a linear probe on the J-space predicts, per hypothesis, whether it has been eliminated, with accuracy ≥ 95%; eliminated hypotheses' directions decay in J-space projection norm as elimination proceeds.
- **P4 (Causal asymmetry, the frame-precision test).** Swapping hypothesis content inside the J-space (activation patching: replace hypothesis-h directions with hypothesis-h′ directions from a donor run) redirects the model's posterior to the Bayes posterior of the *swapped* evidence, measured in bits. By contrast, ablating the entropy-readout axis leaves calibration intact (replicating Paper III's boundary result in the wind tunnel). Content is load-bearing; the uncertainty readout is not.
- **P5 (Horizon boundary, Nature-paper bridge).** In the K=5 loss-horizon model, the J-space is well-formed at positions ≤ 5 and degrades to noise past position 5 (overlap with frame subspace falls to null level). The workspace's flexibility does not extend the compilation boundary.

Falsification is meaningful for each: P1 failing says the workspace is not the frame; P2 failing says workspace writing is distributed even in a minimal model; P4's swap failing says wind-tunnel content is not causally routed the way production content is; P5 *succeeding in reverse* (workspace intact past the horizon) would be evidence against the position-specific-circuit account and must be reported if found.

## 3. Models and data (all existing, no new training)

| Asset | Source | Purpose |
|---|---|---|
| Bijection transformer, 2.67M params, trained | wind-tunnel repo | primary model, P1–P4 |
| HMM transformer (K=20) | wind-tunnel repo | replication in transport regime |
| K=5 loss-horizon model | Nature-paper repo | P5 |
| Training checkpoints (bijection), ~10 spanning training | wind-tunnel repo | workspace formation dynamics (secondary) |
| 2,000 held-out bijections + analytic posteriors | wind-tunnel repo | ground truth for P3/P4 |
| MLP control, 2.70M | wind-tunnel repo | negative control: J-lens should find no low-dim reusable subspace |

If any checkpoint set is missing, retraining the bijection model is ~4 GPU-hours; budget included.

## 4. Method

### 4.1 J-lens operationalization
Follow Anthropic's released code where possible (their repo is public; pin the commit in the run config). Fallback operationalization if their code doesn't transplant to our architecture:

For residual stream activation x at (layer ℓ, position i), compute the Jacobian J = ∂logits(future positions) / ∂x. Stack Jacobians across a batch of sequences and positions; the J-space at (ℓ, i) is the top-r right singular subspace (r chosen by explained-variance elbow, report r=4, 8, 16). This is the "directions that matter for potential outputs" definition, which is the substance of their method.

- Compute per layer ℓ ∈ {0..L} and position i ∈ {1..2K}, batched over 256 held-out sequences.
- Cost note: models are 2.67M params; full Jacobians via vmap'd VJPs are cheap. Estimate ~2–4 GPU-hours per model for the full (ℓ, i) sweep at batch 256.

### 4.2 Frame subspace (comparison target)
As in Paper I: span of Layer-0 key directions for the V hypothesis tokens (and separately, value directions). Already extracted in the repo; re-derive per checkpoint for the dynamics analysis.

### 4.3 Metrics
- **Subspace overlap:** principal angles, projection-Frobenius overlap, and linear CKA between J-space and frame subspace. Null: 1,000 random orthonormal subspaces of matched dimension, report z-scores.
- **Read/write connectivity:** for each head/MLP, write = ‖P_J · output‖ / ‖output‖; read = sensitivity of the component's output to J-space perturbations of its input (finite-difference along top J-directions). Report per-component, ranked; this mirrors their ~100× connectivity claim at wind-tunnel scale.
- **Decoding:** logistic probes from J-space coordinates to (a) hypothesis identity, (b) per-hypothesis eliminated/surviving status. Train/test split across sequences, never within.
- **Causal:** swap intervention effect measured as KL(model posterior ‖ Bayes posterior of swapped evidence) in bits, versus KL to the *original* evidence's posterior. Success = swapped-KL < original-KL by ≥ 1 bit, per position, averaged over ≥ 200 swap pairs.

## 5. Phases and gates

**Phase 0 — Infra (1–2 days, ~10 GPU-hours).** Port/pin J-lens code; verify Jacobian sweep runs on one model; confirm J-space is stable across batch resamples (overlap between two disjoint-batch extractions ≥ 0.8). *Gate G0: stability. If the J-space isn't reproducible across batches at this scale, stop and report; everything downstream is noise.*

**Phase 1 — Identity (P1) (~15 GPU-hours).** Full (ℓ, i) sweep on bijection model; overlap metrics vs frame subspace and nulls; decoding probes. MLP control in parallel (expect: no stable low-rank J-space, or one with null-level structure). *Gate G1 = P1 thresholds. This is the go/no-go for the paper framing. If P1 fails, pivot: the writeup becomes "the workspace is not the frame," still publishable, skip P2, keep P3–P4.*

**Phase 2 — Writers (P2) (~10 GPU-hours).** Connectivity analysis; cross-check writer ranking against Paper I's head-ablation indispensability ranking (Spearman ρ, expect > 0.7).

**Phase 3 — Causality (P4) (~20 GPU-hours).** Swap interventions and entropy-axis ablations, same harness. This phase also discharges the probe-ablation experiment already on our backlog (the (a,b)-direction functional-use question) as a special case; fold it in.

**Phase 4 — Horizon (P5) (~10 GPU-hours).** J-lens sweep on the K=5 model, positions 1–10. Report overlap and decoding curves across the position-5 boundary.

**Phase 5 — Dynamics (secondary, ~15 GPU-hours).** J-space extraction across training checkpoints. Prediction from the frame-precision dissociation: J-space overlap with the final frame rises early and plateaus while calibration is still improving. Nice-to-have; do not block the writeup on it.

## 6. Engineering notes

- Everything fits on a single A100/H100; parallelism across (model, phase) only. No multi-node work.
- Deterministic seeds throughout; log all runs with the (ℓ, i, r, batch) config; store extracted subspaces as artifacts (they're tiny) so analyses re-run without recomputing Jacobians.
- Repo layout: `experiments/jlens/` with `extract.py` (Jacobian sweep), `subspaces.py` (frame + nulls), `metrics.py`, `interventions.py`, `configs/*.yaml` per phase. Analysis notebooks read artifacts only.
- Pin the Anthropic repo commit and record any deviations from their operationalization in `DEVIATIONS.md`; provenance matters if this becomes a public reconciliation.

## 7. Deliverables

1. Metrics tables for P1–P5 with nulls and preregistered thresholds, pass/fail marked.
2. Figures: overlap heatmap over (layer, position); writer-connectivity ranking; swap-intervention KL curves; K=5 boundary plot.
3. A short results memo (2 pages) sufficient to decide: standalone note, Act-3 extension of the unified paper, or blog post pairing with the Anthropic release.

## 8. Risks

- Their operationalization may not transfer cleanly to a 4-layer model (workspace claims are about deep networks). Mitigation: r-sweep and per-layer reporting; the identity claim only needs *some* layer/position band to carry it.
- Jacobian-defined subspaces can be dominated by the unembedding geometry in small models. Mitigation: recompute J-lens with logits taken ≥ 2 positions ahead (their "potential outputs" spirit) and check robustness.
- P4 swap donors must be evidence-matched (same observed positions, different hypothesis) or the intervention confounds content with position; the pairing script must enforce this.
