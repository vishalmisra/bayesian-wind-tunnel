# Deviations from Anthropic's released J-lens operationalization

Reference implementation: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens),
pinned at commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` (Apache-2.0,
released 2026-07-06 with "A global workspace in language models").

Provenance matters if this becomes a public reconciliation; every deviation
and its rationale is recorded here.

## 1. Per-(layer, position) J-spaces, not per-layer averages

Their `fitting.fit()` produces one `J_l` per layer, *averaged over source
positions* (mean over valid positions of the summed-over-future-targets
Jacobian). The wind tunnel's mechanism is position-specific (elimination
proceeds pair by pair), so the identity claim P1 is stated per (layer,
position). We therefore keep the Jacobian rows resolved by source position
and never average across positions.

We retain their estimator *design*: one-hot cotangents at output dimensions,
target positions batched, output dims riding the batch axis (their
`dim_batch` trick, our vocab-replication).

## 2. Targets are logits at future positions, not the final-layer residual

Their estimator backprops from the final-layer (or penultimate-layer)
*residual* and applies the model's unembedding afterward (`lens_l(h) =
unembed(J_l h)`). The spec's operationalization is `J = d logits(future) /
d x`, so we backprop from the logits directly. For TinyGPT the unembedding
is weight-tied and preceded by RMSNorm, so the two differ by the
unembedding geometry; the spec's risk section flags exactly this. Mitigation
implemented: `min_horizon` config (targets >= 1 vs >= 2 positions ahead)
as the robustness check.

## 3. Subspace, not transport map

Their lens is the transport map `J_l` itself (used to read out logits). Our
object is the top-r right singular subspace of the stacked rows ("the
directions that matter for potential outputs"), computed via on-the-fly
Gram accumulation (`G = M^T M`; top eigenspace of G = top right-singular
subspace of M). r is swept over {4, 8, 16} per the spec.

Two reductions are reported:
- `stacked`: every (target position, vocab dim) contributes its own row.
- `summed`: rows summed over future target positions first — the reduction
  closest to their released estimator.

## 4. `skip_first = 0`

Their `SKIP_FIRST_N_POSITIONS = 16` excludes attention-sink-dominated early
positions in natural-language prompts. Wind-tunnel sequences are 39 tokens
with meaningful structure from position 0 (the first key), and TinyGPT has
no BOS/sink token; we exclude nothing.

## 4b. Cotangent dims restricted to hypothesis logits

The production bijection checkpoints use a separate key/value vocabulary
(keys 0..V-1, values V..2V-1; vocab = 2V) and sequences
`[k1, v1+V, ..., kL]` with no trailing query token — every key position is
a query. Cotangents are injected only at the V value-token logits (the
hypothesis dimensions), not all 2V. Key-token logits are structurally
near-constant (keys are drawn without replacement by the *data*, not
predicted by the model's Bayesian machinery) and would only add
unembedding-geometry noise to the J-space. Config knob: `cot_dims`.

## 5. Token-id prompts

Their `LensModel.encode` takes text; wind-tunnel prompts are token-id
tensors already. No tokenizer is involved.

## 6. Everything else follows their release where applicable

- Graph-rooting trick: parameters frozen, earliest captured residual marked
  `requires_grad` so the retained graph spans the blocks only
  (their `hooks.py start_graph_at`).
- Forward-hook capture of block outputs, tuple-unwrapped (their
  `ActivationRecorder`; TinyGPT blocks return `(hidden, attn)`).
- fp32 accumulation on device, fp64 aggregation on CPU (they store fp16
  lenses; we keep fp64 Grams since ours are tiny).
