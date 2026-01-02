#!/usr/bin/env python3
"""
SULA (Structured Uncertainty Learning from Examples) experiments.

This script runs the SULA ICL experiments from Paper III, demonstrating
that LLMs move along their value manifold as evidence accumulates.

Usage:
    # Generate SULA prompts first
    python generate_sula.py --output icl_sula_prompts.json
    
    # Run experiments
    python sula_experiments.py --prompts icl_sula_prompts.json

Reference: Paper III, Section 4
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# SULA vocabulary
POSITIVE_WORDS = ["happy", "joyful", "wonderful", "excellent", "fantastic", "great", "amazing", "brilliant"]
NEGATIVE_WORDS = ["sad", "terrible", "awful", "horrible", "dreadful", "poor", "bad", "miserable"]
ALPHA = 0.9  # Noise parameter for label generation


def get_args():
    parser = argparse.ArgumentParser(description="Run SULA experiments")
    parser.add_argument("--prompts", type=str, default=None, help="Path to pre-generated prompts JSON")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-410m", help="Model to evaluate")
    parser.add_argument("--n_prompts_per_k", type=int, default=50, help="Prompts per k value")
    parser.add_argument("--output_dir", type=str, default="results/sula", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def find_shared_single_token_labels(tokenizer) -> Tuple[str, str]:
    """Find two gibberish labels that tokenize to single tokens."""
    candidates = ["Ġblurp", "Ġzorb", "Ġfloop", "Ġmork", "Ġglib", "Ġtwerk"]
    valid = []
    for c in candidates:
        tokens = tokenizer.encode(" " + c.replace("Ġ", ""), add_special_tokens=False)
        if len(tokens) == 1:
            valid.append(c.replace("Ġ", ""))
    if len(valid) >= 2:
        return valid[0], valid[1]
    # Fallback to simple labels
    return "A", "B"


def compute_bayesian_posterior(examples: List[Tuple[str, str]], L1: str, L2: str, true_label: str) -> Dict[str, float]:
    """Compute Bayesian posterior for SULA task."""
    n_match = sum(1 for _, lab in examples if lab == true_label)
    n_mismatch = len(examples) - n_match
    
    log_like_true = n_match * np.log(ALPHA) + n_mismatch * np.log(1 - ALPHA)
    log_like_other = n_mismatch * np.log(ALPHA) + n_match * np.log(1 - ALPHA)
    
    max_ll = max(log_like_true, log_like_other)
    p_true = np.exp(log_like_true - max_ll) / (np.exp(log_like_true - max_ll) + np.exp(log_like_other - max_ll))
    
    if true_label == L1:
        return {L1: float(p_true), L2: float(1 - p_true)}
    else:
        return {L1: float(1 - p_true), L2: float(p_true)}


def entropy_bits(prob_dict: Dict[str, float]) -> float:
    """Compute entropy in bits."""
    probs = np.array(list(prob_dict.values()))
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def generate_sula_prompts(
    k_shots: List[int],
    n_prompts_per_k: int,
    L1: str,
    L2: str,
    seed: int = 42
) -> List[Dict]:
    """Generate SULA prompts with Bayesian ground truth."""
    random.seed(seed)
    np.random.seed(seed)
    all_words = POSITIVE_WORDS + NEGATIVE_WORDS
    
    prompts = []
    for k in k_shots:
        for prompt_idx in range(n_prompts_per_k):
            test_word = random.choice(all_words)
            true_label = random.choice([L1, L2])
            
            examples = []
            for _ in range(k):
                w = random.choice(all_words)
                if random.random() < ALPHA:
                    lab = true_label
                else:
                    lab = L2 if true_label == L1 else L1
                examples.append((w, lab))
            
            if k == 0:
                prompt_text = f"{test_word}:"
            else:
                ex_str = ", ".join(f"{w}: {lab}" for w, lab in examples)
                prompt_text = f"{ex_str}. {test_word}:"
            
            bayes_post = compute_bayesian_posterior(examples, L1, L2, true_label)
            bayes_H = entropy_bits(bayes_post)
            
            prompts.append({
                "k": k,
                "prompt_idx": prompt_idx,
                "prompt_text": prompt_text,
                "test_word": test_word,
                "examples": examples,
                "true_label": true_label,
                "bayes_posterior": bayes_post,
                "bayes_entropy": bayes_H,
            })
    
    return prompts


def extract_sula_geometry(model, tokenizer, prompts: List[Dict], L1: str, L2: str, device):
    """Extract value manifold coordinates and model predictions for SULA prompts."""
    L1_id = tokenizer.encode(" " + L1, add_special_tokens=False)[0]
    L2_id = tokenizer.encode(" " + L2, add_special_tokens=False)[0]
    
    results = []
    hidden_states = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt["prompt_text"], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        
        # Last layer, last token hidden state
        hidden = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
        hidden_states.append(hidden)
        
        # Model prediction
        logits = outputs.logits[0, -1, :]
        L1_logit = logits[L1_id].item()
        L2_logit = logits[L2_id].item()
        
        probs = torch.softmax(torch.tensor([L1_logit, L2_logit]), dim=0)
        model_posterior = {L1: probs[0].item(), L2: probs[1].item()}
        model_entropy = entropy_bits(model_posterior)
        
        results.append({
            **prompt,
            "model_posterior": model_posterior,
            "model_entropy": model_entropy,
        })
    
    # PCA on hidden states
    X = np.stack(hidden_states)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_std)
    
    for i, r in enumerate(results):
        r["pc1"] = float(coords[i, 0])
        r["pc2"] = float(coords[i, 1])
    
    return results, pca.explained_variance_ratio_


def plot_sula_results(results: List[Dict], output_dir: Path, model_name: str):
    """Plot SULA experiment results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Group by k
    k_values = sorted(set(r["k"] for r in results))
    
    # Plot 1: Entropy calibration
    ax1 = axes[0]
    for k in k_values:
        k_results = [r for r in results if r["k"] == k]
        bayes_ents = [r["bayes_entropy"] for r in k_results]
        model_ents = [r["model_entropy"] for r in k_results]
        ax1.scatter([k] * len(bayes_ents), bayes_ents, c="blue", alpha=0.3, s=20, label="Bayes" if k == k_values[0] else "")
        ax1.scatter([k] * len(model_ents), model_ents, c="red", alpha=0.3, s=20, label="Model" if k == k_values[0] else "")
    
    # Mean lines
    bayes_means = [np.mean([r["bayes_entropy"] for r in results if r["k"] == k]) for k in k_values]
    model_means = [np.mean([r["model_entropy"] for r in results if r["k"] == k]) for k in k_values]
    ax1.plot(k_values, bayes_means, "b--", linewidth=2, label="Bayes mean")
    ax1.plot(k_values, model_means, "r-", linewidth=2, label="Model mean")
    ax1.set_xlabel("k (number of examples)")
    ax1.set_ylabel("Entropy (bits)")
    ax1.set_title("Entropy Calibration")
    ax1.legend()
    
    # Plot 2: Value manifold
    ax2 = axes[1]
    pc1 = [r["pc1"] for r in results]
    pc2 = [r["pc2"] for r in results]
    bayes_ent = [r["bayes_entropy"] for r in results]
    scatter = ax2.scatter(pc1, pc2, c=bayes_ent, cmap="viridis", s=30, alpha=0.7)
    plt.colorbar(scatter, ax=ax2, label="Bayes Entropy")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("Value Manifold")
    
    # Plot 3: Manifold trajectory
    ax3 = axes[2]
    for k in k_values:
        k_results = [r for r in results if r["k"] == k]
        pc1_vals = [r["pc1"] for r in k_results]
        ax3.scatter([k] * len(pc1_vals), pc1_vals, alpha=0.3, s=20)
    
    pc1_means = [np.mean([r["pc1"] for r in results if r["k"] == k]) for k in k_values]
    pc1_stds = [np.std([r["pc1"] for r in results if r["k"] == k]) for k in k_values]
    ax3.errorbar(k_values, pc1_means, yerr=pc1_stds, fmt="o-", capsize=5, linewidth=2)
    ax3.set_xlabel("k (number of examples)")
    ax3.set_ylabel("PC1 (Bayesian axis)")
    ax3.set_title("Manifold Trajectory")
    
    plt.suptitle(f"SULA Experiments: {model_name}", fontsize=14)
    plt.tight_layout()
    
    out_path = output_dir / "sula_results.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")


def main():
    args = get_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device = next(model.parameters()).device
    model.eval()
    
    # Find labels
    L1, L2 = find_shared_single_token_labels(tokenizer)
    print(f"Using labels: {L1}, {L2}")
    
    # Generate or load prompts
    if args.prompts and Path(args.prompts).exists():
        print(f"Loading prompts from {args.prompts}")
        with open(args.prompts) as f:
            data = json.load(f)
            prompts = data["prompts"]
            L1, L2 = data["labels"]["L1"], data["labels"]["L2"]
    else:
        print("Generating SULA prompts...")
        k_shots = [0, 1, 2, 4, 8]
        prompts = generate_sula_prompts(k_shots, args.n_prompts_per_k, L1, L2, args.seed)
    
    print(f"Total prompts: {len(prompts)}")
    
    # Run experiments
    print("Extracting geometry...")
    results, var_explained = extract_sula_geometry(model, tokenizer, prompts, L1, L2, device)
    
    # Compute metrics
    pc1 = np.array([r["pc1"] for r in results])
    bayes_ent = np.array([r["bayes_entropy"] for r in results])
    rho, _ = spearmanr(pc1, bayes_ent)
    
    print("\n" + "=" * 50)
    print("SULA RESULTS")
    print("=" * 50)
    print(f"PC1 variance: {var_explained[0]*100:.1f}%")
    print(f"PC1+PC2 variance: {sum(var_explained[:2])*100:.1f}%")
    print(f"PC1-entropy correlation: {rho:.3f}")
    
    # Save results
    summary = {
        "model": args.model,
        "n_prompts": len(prompts),
        "pc1_variance": float(var_explained[0]),
        "pc2_variance": float(var_explained[1]),
        "pc1_entropy_correlation": float(rho),
    }
    
    with open(output_dir / "sula_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Plot
    plot_sula_results(results, output_dir, args.model)
    
    if abs(rho) > 0.5:
        print("\n✓ Strong PC1-entropy correlation (Bayesian manifold confirmed)")
    else:
        print("\n~ Weak PC1-entropy correlation")


if __name__ == "__main__":
    main()
