# Bayesian Wind Tunnel

**Code for the Bayesian Attention Trilogy**

This repository contains code to reproduce the experiments from three papers demonstrating that transformers implement Bayesian inference through geometric mechanisms:

| Paper | Title | Status |
|-------|-------|--------|
| **I** | The Bayesian Geometry of Transformer Attention | [arXiv](https://arxiv.org) |
| **II** | Gradient Dynamics of Attention: How Cross-Entropy Sculpts Bayesian Manifolds | [arXiv](https://arxiv.org) |
| **III** | Geometric Scaling of Bayesian Inference in LLMs | [arXiv](https://arxiv.org) |

## Key Findings

**Paper I** introduces *Bayesian wind tunnels*—controlled environments with analytic posteriors where we prove transformers implement exact Bayesian inference:

- Small transformers achieve **< 0.01 bit MAE** between model and Bayes-optimal entropy
- Capacity-matched MLPs fail by orders of magnitude
- Geometric mechanism: orthogonal keys, entropy-aligned value manifolds, progressive attention focusing

**Paper III** shows these geometric signatures persist in production LLMs (Pythia, Llama, Mistral):

- Value manifolds organize along entropy-aligned axes (70-95% variance in PC1+PC2)
- Domain restriction collapses representations to low-dimensional manifolds
- SULA experiments demonstrate Bayesian-like evidence integration

## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/bayesian-wind-tunnel.git
cd bayesian-wind-tunnel
pip install -r requirements.txt
```

### Reproduce Paper I (Bijection Wind Tunnel)

**Train a TinyGPT on bijection learning:**
```bash
python experiments/paper1_bijection/train.py \
    --V 20 --L 19 \
    --max_steps 100000 \
    --output_dir logs/bijection_v20
```

**Evaluate entropy calibration:**
```bash
python experiments/paper1_bijection/eval_entropy.py \
    --checkpoint logs/bijection_v20/ckpt_final.pt \
    --output_dir results/bijection
```

Expected output: `MAE < 0.01 bits` (Bayesian inference achieved)

### Reproduce Paper III (LLM Geometry)

**Analyze geometric structure in Pythia:**
```bash
python experiments/paper3_llm/analyze_geometry.py \
    --model EleutherAI/pythia-410m \
    --output_dir results/pythia_geometry
```

**Analyze Llama-3 (requires HuggingFace authentication):**
```bash
huggingface-cli login
python experiments/paper3_llm/analyze_geometry.py \
    --model meta-llama/Llama-3.2-1B \
    --output_dir results/llama_geometry
```

## Repository Structure

```
bayesian-wind-tunnel/
├── src/
│   ├── models/          # TinyGPT and MLP architectures
│   ├── data/            # Bijection and HMM data generation
│   └── utils/           # Entropy and geometry utilities
├── experiments/
│   ├── paper1_bijection/  # Bijection wind tunnel experiments
│   ├── paper1_hmm/        # HMM wind tunnel experiments
│   └── paper3_llm/        # Production LLM geometry analysis
├── configs/             # Experiment configurations
├── checkpoints/         # Model checkpoints (download separately)
└── figures/             # Generated figures
```

## Experiments Overview

### Paper I: Wind Tunnel Experiments

| Experiment | Script | Key Metric |
|------------|--------|------------|
| Bijection entropy | `paper1_bijection/eval_entropy.py` | MAE < 0.01 bits |
| HMM length generalization | `paper1_hmm/eval_length_gen.py` | K=50 MAE < 0.03 bits |
| Value manifold PCA | `paper1_bijection/plot_geometry.py` | PC1 variance > 80% |

### Paper III: LLM Geometry

| Experiment | Script | Key Metric |
|------------|--------|------------|
| Key orthogonality | `paper3_llm/analyze_geometry.py` | Layer 0 < 0.15 |
| Value manifold | `paper3_llm/analyze_geometry.py` | PC1+PC2 > 70% |
| SULA trajectory | `paper3_llm/sula_experiments.py` | ρ(PC1, entropy) > 0.5 |

## Pretrained Checkpoints

Checkpoints for reproducing results will be available at:
- **Bijection V=20**: [Download link TBD]
- **HMM K=20**: [Download link TBD]

## Citation

If you use this code, please cite our papers:

```bibtex
@article{aggarwal2025bayesian_geometry,
  title={The Bayesian Geometry of Transformer Attention},
  author={Aggarwal, Naman and Dalal, Siddhartha R. and Misra, Vishal},
  journal={arXiv preprint},
  year={2025},
  note={Paper I of the Bayesian Attention Trilogy}
}

@article{aggarwal2025gradient_dynamics,
  title={Gradient Dynamics of Attention: How Cross-Entropy Sculpts Bayesian Manifolds},
  author={Aggarwal, Naman and Misra, Vishal and Dalal, Siddhartha R.},
  journal={arXiv preprint},
  year={2025},
  note={Paper II of the Bayesian Attention Trilogy}
}

@article{aggarwal2025geometric_scaling,
  title={Geometric Scaling of Bayesian Inference in LLMs},
  author={Aggarwal, Naman and Dalal, Siddhartha R. and Misra, Vishal},
  journal={arXiv preprint},
  year={2025},
  note={Paper III of the Bayesian Attention Trilogy}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- Naman Aggarwal (Dream Sports)
- Siddhartha R. Dalal (Columbia University)
- Vishal Misra (Columbia University)
