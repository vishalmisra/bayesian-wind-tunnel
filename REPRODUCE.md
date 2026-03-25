# Reproducibility guide

Step-by-step instructions to reproduce the main results from the Bayesian Attention Trilogy.

## Quick verification (15 minutes, 1 GPU)

Run the construction boundary experiment at small scale to verify everything works:

```bash
git clone https://github.com/vishalmisra/bayesian-wind-tunnel.git
cd bayesian-wind-tunnel
pip install -r requirements.txt

# Train for 1000 steps (~2 minutes on any GPU)
python experiments/construction_boundary/recurrence_bwt.py \
    --n_steps 1000 --seeds 42 --device cuda:0 \
    --output_dir results/quick_test

# Expected: loss drops from ~2.8 to ~1.7, eval shows MAE improving
```

If this runs without errors and loss decreases, the environment is working.

## Prerequisites

```bash
pip install -r requirements.txt
```

For Paper III (production LLM probing), you also need HuggingFace authentication:
```bash
huggingface-cli login
```

### Hardware

| Experiment | GPU memory | Time |
|------------|-----------|------|
| Wind tunnels (Papers I, II) | 4 GB | 2-8 hours |
| Construction boundary | 4 GB | 8 hours |
| LLM probing (410M-1B) | 8 GB | 10-30 minutes |
| LLM probing (7B) | 24 GB | 30 minutes |

---

## Paper I: bijection wind tunnel

### Train transformer

```bash
python experiments/paper1_bijection/train.py \
    --V 20 --L 19 \
    --dim 192 --layers 6 --heads 6 \
    --batch_size 128 --lr 3e-4 \
    --max_steps 100000 \
    --seed 42
```

**Expected:** Final entropy MAE < 0.01 bits against Bayesian posterior.

### Evaluate entropy calibration

```bash
python experiments/paper1_bijection/eval_entropy.py \
    --checkpoint <path_to_checkpoint> \
    --V 20 --L 19 \
    --n_samples 2000
```

### Train architecture baselines

```bash
# LSTM (succeeds on bijection, fails on HMM)
python experiments/paper1_bijection/train_lstm.py --V 20 --L 19 --max_steps 100000

# Mamba (succeeds on bijection and HMM, struggles with recall)
python experiments/paper1_bijection/train_mamba.py --V 20 --L 19 --max_steps 100000
```

### Generate geometry figures

```bash
# Value manifold PCA (expects PC1 > 80%, strong entropy correlation)
python experiments/paper1_bijection/figures/geometric_analysis.py --checkpoint <path>

# Manifold visualization
python experiments/paper1_bijection/figures/manifold_visualization.py --checkpoint <path>

# QKV structure analysis
python experiments/paper1_bijection/figures/qkv_analysis.py --checkpoint <path>

# Primitives taxonomy figure (no checkpoint needed)
python experiments/paper1_bijection/figures/generate_primitives_figure.py
```

---

## Paper I: HMM wind tunnel

### Train and evaluate

```bash
# Train on K=20 sequences
python experiments/paper1_hmm/train.py --epochs 50 --seed 42

# Evaluate length generalization (K=20 → K=50)
python experiments/paper1_hmm/eval_length_gen.py --checkpoint <path> --test_lengths 20 30 50
```

**Expected:**
- K=20: MAE ~ 7.5e-5 bits
- K=30: MAE ~ 0.012 bits
- K=50: MAE ~ 0.029 bits

### Architecture baselines

```bash
python experiments/paper1_hmm/train_lstm.py --epochs 50
python experiments/paper1_hmm/train_mamba.py --epochs 50
```

### Mamba geometry

```bash
# Mamba analysis (needs trained mamba checkpoint)
python experiments/paper1_hmm/mamba_analysis.py --checkpoint <path>

# HMM figures
python experiments/paper1_hmm/figures/hmm_geometric_analysis.py --checkpoint <path>
python experiments/paper1_hmm/figures/hmm_figures_better_colors.py --checkpoint <path>
python experiments/paper1_hmm/figures/layerwise_analysis.py --checkpoint <path>
python experiments/paper1_hmm/figures/mamba_5clusters.py --checkpoint <mamba_checkpoint>
```

---

## Paper I: ablations

```bash
# Per-head ablation (identifies hypothesis-frame head)
python experiments/ablations/head_ablation.py --checkpoint <path> --domain-size 20 --seq-length 19

# Per-layer ablation (tests hierarchical compositionality)
python experiments/ablations/layer_ablation.py --checkpoint <path> --domain-size 20 --seq-length 19
```

---

## Paper II: EM vs SGD gradient dynamics

```bash
# Standard run (compares EM-like schedule vs SGD on sticky Markov chain)
python experiments/paper2_gradient/em_vs_sgd_multiseed.py

# Deep transformer variant
python experiments/paper2_gradient/em_vs_sgd_multiseed.py --deep --n_layers 4 --n_heads 4
```

**Expected:** EM-like schedule converges 2.3x faster (430 vs 1000 steps).

---

## Paper III: production LLM geometry

### Probe geometric signatures

```bash
# Pythia-410M
python experiments/paper3_llm/analyze_geometry.py --model EleutherAI/pythia-410m

# Phi-2
python experiments/paper3_llm/analyze_geometry.py --model microsoft/phi-2

# Llama-3.2-1B (requires HuggingFace auth)
python experiments/paper3_llm/analyze_geometry.py --model meta-llama/Llama-3.2-1B

# Mistral-7B (24GB GPU)
python experiments/paper3_llm/analyze_geometry.py --model mistralai/Mistral-7B-v0.1
```

### SULA in-context evaluation

```bash
python experiments/paper3_llm/sula_experiments.py --model EleutherAI/pythia-410m
```

### Full SULA pipeline

```bash
cd experiments/paper3_llm/sula_pipeline

# 1. Generate dataset
python generate_sula_dataset.py

# 2. Run main evaluation
python run_sula_main.py --model EleutherAI/pythia-410m

# 3. Extract geometry
python extract_geometry_sula.py --model EleutherAI/pythia-410m

# 4. Analyze and visualize
python analyze_icl_sula.py
python compare_sula_interventions.py
python visualize_sula_controls.py
```

---

## Construction boundary (Shannon/Kolmogorov wall)

### The π experiment

```bash
# Integer tokens (succeeds: 0.014-bit MAE)
python experiments/construction_boundary/recurrence_bwt.py \
    --seeds 42 --device cuda:0 --n_steps 150000

# Opaque tokens (fails: 0.83-bit MAE)
python experiments/construction_boundary/recurrence_bwt.py \
    --opaque --seeds 42 --device cuda:1 --n_steps 150000
```

### Loss horizon restriction

```bash
# K=5: 0.020 bits at trained positions, 1.63 bits at untrained (83x gap)
python experiments/construction_boundary/recurrence_extrapolation.py \
    --loss_horizon 5 --seeds 42 --device cuda:0

# Length extrapolation (train on 8 tokens, test on 16/32/50)
python experiments/construction_boundary/recurrence_extrapolation.py \
    --sinusoidal_pe --train_len 8 --seeds 42 --device cuda:0
```

### Controls

```bash
# Shuffled temporal order
python experiments/construction_boundary/recurrence_shuffled_control.py --seeds 42

# Modulus transfer (p=17 → p=19)
python experiments/construction_boundary/recurrence_modulus_transfer.py --seeds 42
```

---

## Verification checklist

| Paper | Experiment | Metric | Expected |
|-------|-----------|--------|----------|
| I | Bijection (transformer) | Entropy MAE | < 0.01 bits |
| I | Bijection (MLP) | Entropy MAE | > 1 bit |
| I | HMM K=20 | Per-position MAE | ~ 7.5e-5 bits |
| I | HMM K=50 | Per-position MAE | < 0.03 bits |
| I | Key orthogonality | Layer 0 cosine | < 0.1 |
| I | Value manifold | PC1 variance | > 80% |
| II | EM vs SGD | Convergence speed | EM ~2.3x faster |
| III | Pythia-410M | Key orthogonality | 0.11-0.13 |
| III | Domain restriction | PC1+PC2 collapse | single > mixed |
| CB | π experiment (integer) | Entropy MAE | 0.014 bits |
| CB | π experiment (opaque) | Entropy MAE | 0.83 bits |
| CB | Loss horizon K=5 | Trained/untrained gap | 83x |

---

## Troubleshooting

**Out of memory:** Reduce batch size or use `--dtype float16` for LLM experiments.

**HuggingFace auth:** `huggingface-cli login` with a token from https://huggingface.co/settings/tokens

**Import errors:** Make sure you're running from the repo root so `src/` is on the Python path.
