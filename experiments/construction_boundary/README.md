# Construction boundary experiments

These experiments demonstrate the Shannon/Kolmogorov wall: gradient descent
compiles program-identification circuits with near-Bayesian precision at
rewarded positions, but these circuits do not generalize beyond the loss horizon.

## Experiments

### The π experiment (`recurrence_bwt.py`)

The core experiment. Trains a transformer on modular recurrences
x_{t+1} = ax_t + b mod 17 where (a, b) are freshly sampled each episode.

```bash
# Integer tokens (succeeds — 0.014-bit MAE)
python recurrence_bwt.py --seeds 42 --device cuda:0

# Opaque tokens (fails — 0.83-bit MAE, the compilation boundary)
python recurrence_bwt.py --opaque --seeds 42 --device cuda:1
```

### Loss horizon restriction (`recurrence_extrapolation.py`)

The construction boundary experiment. Restricts gradient signal to the
first K positions; measures inference precision at trained vs untrained positions.

```bash
# Loss horizon K=5 (0.020 bits at positions 1-5, 1.63 bits at positions 6-15)
python recurrence_extrapolation.py --loss_horizon 5 --seeds 42 --device cuda:0

# Length extrapolation (train on 8 tokens, test on 16/32/50)
python recurrence_extrapolation.py --sinusoidal_pe --train_len 8 --seeds 42 --device cuda:0
```

### Shuffled order control (`recurrence_shuffled_control.py`)

Presents (t, x_t) triples in random temporal order instead of sequential.
Tests whether positional routing determinism matters.

```bash
python recurrence_shuffled_control.py --seeds 42 --device cuda:0
```

### Modulus transfer (`recurrence_modulus_transfer.py`)

Fine-tunes a trained p=17 model on p=19 sequences. Tests whether the
learned circuit is abstract or modulus-specific.

```bash
python recurrence_modulus_transfer.py --seeds 42 --device cuda:0
```

## Key results

| Experiment | Trained MAE | Untrained MAE | Ratio |
|-----------|------------|---------------|-------|
| Standard (K=15) | 0.014 bits | — | 1x |
| Loss horizon K=5 | 0.020 bits | 1.63 bits | 83x |
| Loss horizon K=3 | 0.028 bits | 1.95 bits | 69x |
| Opaque tokens | 0.83 bits | — | (fails) |

## Support locality, causal substrate, and distributional error

These scripts back the held-out-prefix, causal-substrate, and full-distribution
results reported in the Nature manuscript. Result JSONs are under `results/`.

### Held-out-prefix control — support locality (`recurrence_heldout_prefix.py`)

Splits generic three-token prefixes into five hash buckets; trains on four,
evaluates the same weights on the fifth. Near-Bayesian on training-prefix support,
sharp failure on held-out prefixes governed by the identical rule.

```bash
python recurrence_heldout_prefix.py --p 17 --holdout_bucket 0 --seed 42 --device cuda:0
```

### p = 31 replication (`run_heldout_p31_all.sh`, `aggregate_heldout_p31.py`)

Replicates the control on a 6.2× larger support (150,000 steps per cell). Per-cell
metrics for all 15 bucket–seed cells are in `results/heldout_prefix_p31/`.

```bash
bash run_heldout_p31_all.sh          # trains + evaluates all 15 cells
python aggregate_heldout_p31.py      # aggregates per-cell metrics
```

### Causal substrate — decodable ≠ functional

- `recurrence_residual_probe.py` — linear probe recovers (a, b) from the residual stream.
- `recurrence_probe_ablation.py` — directional ablation of the probe subspace (+ random-projection control).
- `recurrence_interchange_intervention.py` — inject another sequence's (a, b) subspace.
- `recurrence_intervention.py` — input-level off-trajectory do-intervention (diagnostic only).

Results in `results/causal_substrate/` and `results/intervention_p17/`.

### Distributional error and posterior-extraction residual (Nature revision)

- `recurrence_kltv_reeval.py` — re-evaluates the π-experiment checkpoints and adds
  per-position KL divergence and total variation to the entropy metric (integer model:
  KL ≤ 1e-4 bits, TV ≤ 0.001 by t ≥ 6). Output: `results/recurrence_kltv_reeval.json`.
- `recurrence_residual_check.py` — off-mixture L1 residual for the implicit class-posterior
  extraction (mean 2.8e-5, max 7e-4). Output: `results/recurrence_residual.json`.

```bash
CKPT_DIR=/path/to/recurrence python recurrence_kltv_reeval.py
CKPT_DIR=/path/to/recurrence python recurrence_residual_check.py
```

## Trained-model checkpoints

The checkpoints (~11 MB each: integer/opaque π-experiment models, plus the 15 p = 31
held-out models) are archived on Zenodo alongside this release rather than committed
to git. Download them and point `CKPT_DIR` at the `recurrence/` directory (containing
`integer/seed_*/best_model.pt` and `opaque/seed_*/best_model.pt`).

## Requirements

- PyTorch >= 2.0
- numpy
- A single GPU (experiments run in ~8 hours on an RTX 4090)
