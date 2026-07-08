# J-lens experiments: the global workspace inside a Bayesian wind tunnel

Runs Anthropic's Jacobian-lens ("A global workspace in language models",
2026; code anthropics/jacobian-lens @ `581d3986`) inside the wind tunnel
to test whether their emergent workspace and Paper I's hypothesis frame
are the same object. **Results: `RESULTS.md`. Method deviations:
`DEVIATIONS.md`.**

## Layout

| File | Role |
|------|------|
| `extract.py` | per-(layer, position) Jacobian sweep via Gram accumulation |
| `subspaces.py` | frame subspace (key/value/embedding modes), nulls, entropy axis |
| `metrics.py` | principal angles / projection overlap / CKA, probes (balanced) |
| `connectivity.py` | per-head OV write decomposition, read sensitivity (P2) |
| `interventions.py` | J-space swap patching, subspace ablations (P4) |
| `calibration.py` | entropy-calibration MAE, head-ablation utilities |
| `models.py` | checkpoint loaders: SepVocabTinyGPT / MLPControl / RecurrenceTransformerLens / legacy TinyGPT — one uniform interface |
| `data_gen.py` | annotated bijection batches (sepvocab + paired formats) |
| `run_phase{0..5}.py` | phase runners (see spec); each writes `<out>/phaseN_summary.json` + figures |
| `train_mlp_control.py` | trains the attention-free negative control |
| `smoke_test.py` | CPU correctness suite (brute-force Jacobian match etc.) |

## Assets

All models retrain from the committed scripts in minutes on one
consumer GPU; no checkpoints are distributed.

- Bijection transformer: `train_dense_checkpoints.py` (production
  format: sep-vocab, wq-variant. **Format gotcha:** vocab=2V, values are
  tokens 20-39, 37-token sequences, no query token;
  `src/models/tinygpt.py` cannot load these — use `models.load_model`).
- MLP control: `train_mlp_control.py`. LSTM seeds:
  `train_lstm_seeds.py`. Paired/query variant:
  `train_paired_variant.py` (supervise KEY positions; same-position
  labels teach copying).
- K=5 loss-horizon recurrence models: the companion manuscript's
  `recurrence_extrapolation.py --mode horizon --loss_horizon 5`
  (integer tokens, p=17). **Not** the `extra_loss_density > 0` runs —
  those receive post-horizon supervision and are calibrated everywhere.
- Full-horizon control: `construction_boundary/recurrence_bwt.py`
  (attention proj named `attn.out`, normalized on load).

## Running everything

```bash
python experiments/jlens/smoke_test.py          # CPU, ~2 min
# train the models with the scripts above, then run_phase0..9 with
# --checkpoint pointing at your local checkpoints; each phase is
# seconds to minutes on one consumer GPU (full study ~2 GPU-hours).
```
