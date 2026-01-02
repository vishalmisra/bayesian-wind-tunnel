#!/usr/bin/env python3
"""
Evaluate entropy calibration for bijection models.

This script computes the Mean Absolute Error (MAE) between the model's
predictive entropy and the Bayes-optimal entropy at each position.

For bijection learning without replacement:
    H_Bayes(k) = log₂(V - k + 1)

A well-trained model should achieve MAE < 0.01 bits.

Usage:
    python eval_entropy.py --checkpoint logs/bijection_v20/ckpt_final.pt

Reference: Paper I, Section 3.1 and Figure 2
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import load_tinygpt
from src.utils import evaluate_entropy_calibration


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate entropy calibration")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--V", type=int, default=20, help="Vocabulary size")
    parser.add_argument("--L", type=int, default=19, help="Context length")
    parser.add_argument("--n_samples", type=int, default=2000, help="Number of samples")
    parser.add_argument("--with_replacement", action="store_true", help="Sample with replacement")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def plot_entropy_curves(results: dict, output_path: Path):
    """Plot model vs Bayes entropy curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    positions = range(1, len(results["model_entropy"]) + 1)
    
    ax.plot(positions, results["bayes_entropy"], 'k--', linewidth=2, label="Bayes Optimal")
    ax.plot(positions, results["model_entropy"], 'b-', linewidth=2, label="Model")
    
    ax.set_xlabel("Position k", fontsize=12)
    ax.set_ylabel("Entropy (bits)", fontsize=12)
    ax.set_title(f"Entropy Calibration (MAE = {results['mae_bits']:.4f} bits)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_tinygpt(args.checkpoint, device=str(device))
    
    # Evaluate
    print(f"Evaluating entropy calibration (V={args.V}, L={args.L}, n={args.n_samples})")
    results = evaluate_entropy_calibration(
        model=model,
        V=args.V,
        L=args.L,
        device=device,
        n_samples=args.n_samples,
        with_replacement=args.with_replacement,
        seed=args.seed,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("ENTROPY CALIBRATION RESULTS")
    print("=" * 50)
    print(f"MAE (bits): {results['mae_bits']:.6f}")
    print(f"V: {results['V']}, L: {results['L']}")
    print(f"With replacement: {results['with_replacement']}")
    print(f"N samples: {results['n_samples']}")
    
    if results['mae_bits'] < 0.01:
        print("\n✓ PASS: MAE < 0.01 bits (Bayesian inference achieved)")
    elif results['mae_bits'] < 0.1:
        print("\n~ PARTIAL: MAE < 0.1 bits (good but not optimal)")
    else:
        print("\n✗ FAIL: MAE >= 0.1 bits (poor calibration)")
    
    # Save results
    results_path = output_dir / "entropy_calibration.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Plot
    plot_path = output_dir / "entropy_calibration.png"
    plot_entropy_curves(results, plot_path)


if __name__ == "__main__":
    main()
