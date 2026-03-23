# Bayesian Wind Tunnel

**Code for the Bayesian Attention Trilogy + Construction Boundary Experiments**

This repository contains code to reproduce all experiments from three papers on Bayesian inference in transformers, plus the construction boundary experiments demonstrating the Shannon/Kolmogorov wall.

| Paper | Title | arXiv |
|-------|-------|-------|
| **I** | Attention Is Really What You Need: Exact Bayesian Inference in Transformers | [2512.22471](https://arxiv.org/abs/2512.22471) |
| **II** | Gradient Dynamics of Transformers: An EM Interpretation of Attention | [2512.22473](https://arxiv.org/abs/2512.22473) |
| **III** | Do Large Language Models Implement Bayesian Inference? A Geometric Analysis | [2512.23752](https://arxiv.org/abs/2512.23752) |

## Repository structure

```
experiments/
├── paper1_bijection/        # Paper I: bijection elimination wind tunnel
│   ├── train.py             # Train transformer on bijection task
│   ├── train_lstm.py        # LSTM baseline
│   ├── train_mamba.py       # Mamba comparison
│   ├── eval_entropy.py      # Evaluate entropy MAE vs Bayesian posterior
│   └── figures/             # Key orthogonality, value manifold, QKV analysis
│       ├── geometric_analysis.py
│       ├── manifold_visualization.py
│       ├── qkv_analysis.py
│       └── generate_primitives_figure.py
│
├── paper1_hmm/              # Paper I: HMM filtering wind tunnel
│   ├── train.py             # Train transformer on HMM task
│   ├── train_lstm.py        # LSTM baseline
│   ├── train_mamba.py       # Mamba comparison
│   ├── eval_length_gen.py   # Length generalization (K=20 → K=50)
│   ├── mamba_analysis.py    # Mamba belief-simplex geometry
│   └── figures/             # HMM figures, Mamba 5-clusters, layerwise analysis
│       ├── mamba_5clusters.py
│       ├── hmm_figures_better_colors.py
│       ├── hmm_geometric_analysis.py
│       └── layerwise_analysis.py
│
├── paper2_gradient/         # Paper II: EM vs SGD gradient dynamics
│   └── em_vs_sgd_multiseed.py   # EM-like schedule vs standard SGD
│
├── paper3_llm/              # Paper III: production LLM geometry
│   ├── analyze_geometry.py  # Probe geometric signatures in any HF model
│   ├── sula_experiments.py  # SULA in-context evaluation
│   └── sula_pipeline/       # Full SULA pipeline
│       ├── generate_sula_dataset.py
│       ├── run_sula_main.py
│       ├── extract_geometry_sula.py
│       ├── analyze_icl_sula.py
│       ├── compare_sula_interventions.py
│       └── visualize_sula_controls.py
│
├── construction_boundary/   # Shannon/Kolmogorov wall experiments
│   ├── recurrence_bwt.py              # The π experiment
│   ├── recurrence_extrapolation.py    # Loss horizon + length extrapolation
│   ├── recurrence_modulus_transfer.py # Modulus transfer (p=17 → p=19)
│   └── recurrence_shuffled_control.py # Shuffled temporal order control
│
└── ablations/               # Ablation experiments
    ├── head_ablation.py     # Per-head ablation (hypothesis-frame head)
    └── layer_ablation.py    # Per-layer ablation (hierarchical compositionality)
```

## Quick start

### Installation

```bash
git clone https://github.com/vishalmisra/bayesian-wind-tunnel.git
cd bayesian-wind-tunnel
pip install -r requirements.txt
```

### Paper I: bijection wind tunnel

Train a transformer on bijection elimination (the core wind tunnel):

```bash
# Transformer (achieves ~0.007-bit MAE)
python experiments/paper1_bijection/train.py --V 20 --L 19 --max_steps 100000

# LSTM baseline (achieves ~0.009-bit MAE on bijection, fails on HMM)
python experiments/paper1_bijection/train_lstm.py --V 20 --L 19 --max_steps 100000

# Evaluate entropy calibration against Bayesian posterior
python experiments/paper1_bijection/eval_entropy.py --checkpoint logs/bijection/ckpt_final.pt
```

### Paper I: HMM wind tunnel

```bash
# Train transformer on HMM filtering (achieves 7.5e-5 bit MAE at K=20)
python experiments/paper1_hmm/train.py --max_steps 100000

# Test length generalization (K=20 → K=50)
python experiments/paper1_hmm/eval_length_gen.py --checkpoint logs/hmm/ckpt_final.pt

# Mamba comparison (achieves 0.024-bit MAE, discrete cluster geometry)
python experiments/paper1_hmm/train_mamba.py --max_steps 100000
```

### Paper II: EM vs SGD dynamics

```bash
# Compare EM-like schedule (10x value LR) vs standard SGD
# EM converges 2.3x faster on sticky Markov chain task
python experiments/paper2_gradient/em_vs_sgd_multiseed.py

# Deep transformer variant
python experiments/paper2_gradient/em_vs_sgd_multiseed.py --deep --n_layers 4 --n_heads 4
```

### Paper III: production LLM geometry

```bash
# Probe geometric signatures in Pythia-410M
python experiments/paper3_llm/analyze_geometry.py --model EleutherAI/pythia-410m

# Probe Llama-3.2-1B (requires HuggingFace login)
huggingface-cli login
python experiments/paper3_llm/analyze_geometry.py --model meta-llama/Llama-3.2-1B

# Run SULA in-context evaluation
python experiments/paper3_llm/sula_experiments.py --model EleutherAI/pythia-410m
```

### Construction boundary (the Shannon/Kolmogorov wall)

```bash
# The π experiment — integer tokens succeed (0.014-bit MAE)
python experiments/construction_boundary/recurrence_bwt.py --seeds 42 --device cuda:0

# Opaque tokens fail (0.83-bit MAE) — the compilation boundary
python experiments/construction_boundary/recurrence_bwt.py --opaque --seeds 42 --device cuda:1

# Loss horizon restriction — the construction boundary
# 0.020 bits at positions 1-5, 1.63 bits at positions 6-15 (83x gap)
python experiments/construction_boundary/recurrence_extrapolation.py --loss_horizon 5 --seeds 42

# Shuffled temporal order (routing determinism control)
python experiments/construction_boundary/recurrence_shuffled_control.py --seeds 42

# Modulus transfer (p=17 → p=19, tests circuit abstraction)
python experiments/construction_boundary/recurrence_modulus_transfer.py --seeds 42
```

## Generating figures

Each paper's figure scripts reproduce the key plots:

```bash
# Paper I: key orthogonality, value manifolds, QKV geometry
python experiments/paper1_bijection/figures/geometric_analysis.py --checkpoint <path>
python experiments/paper1_bijection/figures/manifold_visualization.py --checkpoint <path>

# Paper I: Mamba 5-cluster geometry, HMM analysis
python experiments/paper1_hmm/figures/mamba_5clusters.py --checkpoint <path>
python experiments/paper1_hmm/figures/hmm_geometric_analysis.py --checkpoint <path>

# Paper I: primitives taxonomy figure
python experiments/paper1_bijection/figures/generate_primitives_figure.py
```

## Key results

| Experiment | Architecture | MAE (bits) | Notes |
|-----------|-------------|-----------|-------|
| Bijection (K=20) | Transformer | 0.007 | All 3 primitives |
| Bijection (K=20) | Mamba | 0.010 | 2 of 3 primitives |
| Bijection (K=20) | LSTM | 0.009 | 1 of 3 primitives |
| Bijection (K=20) | MLP | 1.85 | Fails |
| HMM (K=20) | Transformer | 7.5e-5 | Per-position |
| HMM (K=20) | Mamba | 0.024 | Outperforms transformer |
| HMM (K=50, extrapolation) | Transformer | 0.029 | Smooth degradation |
| EM vs SGD | Single-head attn | 2.3x faster | EM-motivated LR schedule |
| π experiment (integer) | Transformer | 0.014 | Program identification |
| π experiment (opaque) | Transformer | 0.83 | Compilation boundary |
| Loss horizon K=5 (trained) | Transformer | 0.020 | Near-Bayesian |
| Loss horizon K=5 (untrained) | Transformer | 1.63 | Construction boundary (83x gap) |

## Hardware

- Wind tunnel experiments (Papers I, II, construction boundary): single GPU, ~8 hours on RTX 4090
- Production LLM probing (Paper III): single GPU with ≥24GB VRAM for Pythia/Phi-2/Llama-3.2-1B
- Scaling experiments (construction boundary, 25M-316M): multi-GPU recommended

## Citation

```bibtex
@article{agarwal2025bayesian1,
  title={Attention Is Really What You Need: Exact Bayesian Inference in Transformers via Wind Tunnels},
  author={Agarwal, Naman and Dalal, Siddhartha R. and Misra, Vishal},
  journal={arXiv preprint arXiv:2512.22471},
  year={2025}
}

@article{agarwal2025gradient,
  title={Gradient Dynamics of Transformers: An EM Interpretation of Attention},
  author={Agarwal, Naman and Dalal, Siddhartha R. and Misra, Vishal},
  journal={arXiv preprint arXiv:2512.22473},
  year={2025}
}

@article{agarwal2025geometric3,
  title={Do Large Language Models Implement Bayesian Inference? A Geometric Analysis},
  author={Agarwal, Naman and Dalal, Siddhartha R. and Misra, Vishal},
  journal={arXiv preprint arXiv:2512.23752},
  year={2025}
}
```

## License

MIT
