# Reproducibility Guide

This document provides step-by-step instructions to reproduce all main figures and results from the Bayesian Attention Trilogy papers.

## Prerequisites

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/bayesian-wind-tunnel.git
cd bayesian-wind-tunnel

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For LLM experiments (Paper III), authenticate with HuggingFace
huggingface-cli login
```

## Hardware Requirements

| Experiment | GPU Memory | Time |
|------------|------------|------|
| Bijection training | 4 GB | ~2 hours |
| Bijection evaluation | 2 GB | ~5 minutes |
| LLM geometry (410M) | 8 GB | ~10 minutes |
| LLM geometry (7B) | 24 GB | ~30 minutes |

---

## Paper I: The Bayesian Geometry of Transformer Attention

### Figure 2: Entropy Calibration (Bijection)

**Train the model (or use provided checkpoint):**
```bash
python experiments/paper1_bijection/train.py \
    --V 20 --L 19 \
    --dim 192 --n_layers 6 --n_heads 6 \
    --batch_size 128 --lr 3e-4 \
    --max_steps 100000 \
    --output_dir logs/bijection_v20
```

**Evaluate entropy:**
```bash
python experiments/paper1_bijection/eval_entropy.py \
    --checkpoint logs/bijection_v20/ckpt_final.pt \
    --V 20 --L 19 \
    --n_samples 2000 \
    --output_dir figures/paper1
```

**Expected output:**
- `figures/paper1/entropy_calibration.png`
- MAE < 0.01 bits

### Figure 3: Transformer vs MLP Comparison

**Train MLP baseline:**
```bash
python experiments/paper1_bijection/train.py \
    --V 20 --L 19 \
    --model_type mlp \
    --dim 192 --n_layers 3 \
    --max_steps 100000 \
    --output_dir logs/mlp_baseline
```

**Compare entropy curves:**
```bash
python experiments/paper1_bijection/plot_comparison.py \
    --transformer_ckpt logs/bijection_v20/ckpt_final.pt \
    --mlp_ckpt logs/mlp_baseline/ckpt_final.pt \
    --output figures/paper1/transformer_vs_mlp.png
```

**Expected:** Transformer MAE < 0.01 bits, MLP MAE > 1 bit

### Figure 5: Value Manifold PCA

```bash
python experiments/paper1_bijection/plot_geometry.py \
    --checkpoint logs/bijection_v20/ckpt_final.pt \
    --output_dir figures/paper1
```

**Expected output:**
- `figures/paper1/value_manifold.png`
- PC1 explains > 80% variance
- Strong entropy-PC1 correlation

### Figure 6: Key Orthogonality

```bash
python experiments/paper1_bijection/plot_geometry.py \
    --checkpoint logs/bijection_v20/ckpt_final.pt \
    --analysis key_orthogonality \
    --output_dir figures/paper1
```

**Expected:** Layer 0 off-diagonal cosine < 0.1

---

## Paper I: HMM Wind Tunnel

### Figure 7: Length Generalization

**Train on K=20:**
```bash
python experiments/paper1_hmm/train.py \
    --K 20 \
    --max_steps 150000 \
    --output_dir logs/hmm_k20
```

**Evaluate on K=30 and K=50:**
```bash
python experiments/paper1_hmm/eval_length_gen.py \
    --checkpoint logs/hmm_k20/ckpt_step150000.pt \
    --test_lengths 20 30 50 \
    --output_dir figures/paper1
```

**Expected:**
- K=20: MAE ~ 0.0001 bits
- K=30: MAE ~ 0.012 bits
- K=50: MAE ~ 0.028 bits

---

## Paper III: Geometric Scaling of Bayesian Inference in LLMs

### Figure 1: Value Manifold Across Models

**Run for each model family:**
```bash
# Pythia-410M
python experiments/paper3_llm/analyze_geometry.py \
    --model EleutherAI/pythia-410m \
    --output_dir results/pythia_410m

# Phi-2
python experiments/paper3_llm/analyze_geometry.py \
    --model microsoft/phi-2 \
    --output_dir results/phi2

# Llama-3.2-1B (requires auth)
python experiments/paper3_llm/analyze_geometry.py \
    --model meta-llama/Llama-3.2-1B \
    --output_dir results/llama_1b

# Mistral-7B (16GB+ GPU)
python experiments/paper3_llm/analyze_geometry.py \
    --model mistralai/Mistral-7B-v0.1 \
    --dtype float16 \
    --output_dir results/mistral_7b
```

**Combine into comparison figure:**
```bash
python experiments/paper3_llm/plot_figures.py \
    --results_dirs results/pythia_410m results/phi2 results/llama_1b results/mistral_7b \
    --output figures/paper3/cross_model_comparison.png
```

### Figure 2: Domain Restriction Collapse

```bash
python experiments/paper3_llm/domain_pca.py \
    --model EleutherAI/pythia-410m \
    --domains mixed math code philosophy \
    --output_dir figures/paper3
```

**Expected:** Single-domain PC1+PC2 > 80%, mixed-domain lower

### Figure 3: SULA Trajectory

**Generate SULA prompts:**
```bash
python experiments/paper3_llm/generate_sula.py \
    --output icl_sula_prompts.json
```

**Extract geometry:**
```bash
python experiments/paper3_llm/sula_experiments.py \
    --prompts icl_sula_prompts.json \
    --models EleutherAI/pythia-410m microsoft/phi-2 meta-llama/Llama-3.2-1B \
    --output_dir results/sula
```

**Plot trajectory:**
```bash
python experiments/paper3_llm/plot_sula.py \
    --results_dir results/sula \
    --output figures/paper3/sula_trajectory.png
```

**Expected:** PC1 moves monotonically with k, correlates with Bayes entropy

### Figure 4: Causal Interventions (Pythia)

```bash
python experiments/paper3_llm/causal_interventions.py \
    --model EleutherAI/pythia-410m \
    --intervention axis_ablation \
    --output_dir results/causal
```

---

## Verification Checklist

After running all experiments, verify:

| Paper | Figure | Metric | Expected |
|-------|--------|--------|----------|
| I | 2 | Bijection MAE | < 0.01 bits |
| I | 3 | MLP vs Transformer | Transformer >> MLP |
| I | 5 | PC1 variance | > 80% |
| I | 6 | Layer 0 orthogonality | < 0.1 |
| I | 7 | K=50 generalization | MAE < 0.03 bits |
| III | 1 | PC1+PC2 (domain-restricted) | > 70% |
| III | 2 | Domain collapse | Single > Mixed |
| III | 3 | SULA correlation | ρ > 0.5 |

---

## Troubleshooting

### Out of Memory

For large models, reduce batch size or use CPU offloading:
```bash
python experiments/paper3_llm/analyze_geometry.py \
    --model mistralai/Mistral-7B-v0.1 \
    --dtype float16 \
    --device_map auto
```

### HuggingFace Authentication

Some models (Llama) require authentication:
```bash
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

### CUDA Errors

If you encounter CUDA errors, try:
```bash
export CUDA_LAUNCH_BLOCKING=1
python experiments/...
```

---

## Contact

For issues with reproducibility, please open a GitHub issue or contact the authors.
