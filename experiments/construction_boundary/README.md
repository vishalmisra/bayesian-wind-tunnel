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

## Requirements

- PyTorch >= 2.0
- numpy
- A single GPU (experiments run in ~8 hours on an RTX 4090)
