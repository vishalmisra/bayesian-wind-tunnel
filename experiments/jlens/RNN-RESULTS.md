# Cross-architecture J-lens: LSTM and Mamba (follow-up note material)

**Question:** is the workspace — and the routing-vs-computation
separation — architecture-general, or attention's particular way of
organizing Bayesian content?

**Setup:** same sep-vocab bijection task (accumulation-solvable: no
repeated-key queries, so Bayes-optimal calibration requires only
tracking spent values). Existing checkpoints: LSTM 6x192 (1.80M,
`logs/lstm_bijection_seed7777`), Mamba (1.45M,
`logs/mamba_bijection_v20_longer`); transformer 2.69M and MLP control
from the main study. All calibrated except the MLP: 0.007 / 0.009 /
0.013 / 1.107 bits.

## Findings (single seeds; robustness runs in flight)

| | Transformer | LSTM | Mamba | MLP |
|---|---|---|---|---|
| Calibration (bits) | 0.007 | 0.009 | 0.013 | 1.107 |
| Frame-aligned J-space | emb–block1 (6.1/3.9/1.9x) | **emb only** (6.9x; hidden ~1.0x) | graded (2.7/2.4/1.6x) | null |
| P3 contents decode | 0.998 (block 1) | **1.000 (layer 4, r=24)** | 0.965 (layer 5, r=32) | — |
| P4a swap redirect | +8.2 frame@emb | **+4.3 via own J-coords@hidden**; frame@hidden dead (−6.8) | **FAIL all layers, even full residual** (best −0.6) | — |
| P4b axis inert / frame load-bearing | ✓ / ✓ | ✓ (+0.0000) / ✓ at emb only (+0.96; hidden frame ablation free) | ✓ (−0.0002) / ✓ | — |

1. **The separation principle is architecture-general.** In all three
   calibrated architectures the J-lens finds a causal workspace whose
   contents decode the posterior support, with the precision readout
   perfectly decodable and causally inert. The frame–precision
   dissociation replicates cross-architecture.
2. **The coordinate system is architecture-specific.** The transformer
   holds the workspace in token-frame coordinates (attention's
   content-addressable keys require it). The LSTM re-encodes at the
   embedding boundary: frame coordinates are causally dead in every
   hidden layer (ablation free, swaps inert) while the LSTM's own
   J-space coordinates carry the content (decode 1.00, swap redirect
   +4.3 bits). Same content, private code.
3. **Mamba exposes a transport-topology difference.** No positional
   patch redirects it — not the frame, not its J-space, not the full
   residual (best −0.6 vs required +1). Evidence transport is
   distributed through per-block SSM states, so there is no
   position-by-layer bottleneck to intervene on. The workspace concept
   as a *positional residual subspace* is attention's (and, partially,
   the LSTM's layer-boundary) implementation; Mamba's substrate lives
   in state space, beyond the reach of residual-stream patching.
4. **Taxonomy echo.** Frame-alignment depth orders as transformer >
   Mamba > LSTM(hidden) — the Paper-I primitive ordering appears in
   the workspace *organization* even at matched function, because this
   task variant is solvable by accumulation alone.

**Follow-up-note thesis candidate:** the routing/computation separation
is universal; the *workspace* — a frame-aligned positional subspace —
is attention's implementation of the routing side. Recurrent
architectures implement the same separation in coordinates (LSTM) or
topologies (Mamba) the production J-lens would not see.

**Robustness (resolved 2026-07-08):**
- LSTM replicates on all three seeds: P3 = 1.000 (r=24, layer 3-4);
  frame@emb swap +6.6/+6.6/+6.9 bits; own-J-coords hidden swap
  +3.5/+4.2/+4.9 bits; frame ratios 6.7-6.9x at emb, ~1.0 hidden.
- Mamba identity profile holds shape on seed 2024 (1.8/1.5/1.3, graded
  monotone), weaker magnitude than seed 1 (2.7/2.4/1.6).
- **Phase 9 (state lever) resolves the Mamba question:** state-zero at
  the read boundary explodes KL 0.02 -> ~25 bits (states carry
  everything); evidence-matched STATE swap redirects +1.22 bits at the
  deep position (crosses the causal threshold), lands halfway at
  mid-sequence (local conv/skip/gate paths carry recent evidence).
  "Unpatchable" retired: the Mamba workspace is a state-space object.

**Status:** folded into the note as section 3.6 (option C) with the
cross-architecture figure. Remaining caveats: parameter counts
unmatched; task exercises accumulation only (binding variant = the
follow-up note's opening experiment).

Artifacts: `phase1_lstm`, `phase1_mamba`, `phase3_lstm`, `phase3_mamba`
(+ `phase1_mamba_s2024`, seed retrains) on the GPU workstation.
