#!/usr/bin/env python3
"""
HMM Length Generalization Evaluation

Evaluates trained HMM models on sequences longer than training length.
This tests whether the transformer has learned the underlying algorithm
rather than memorizing patterns for a fixed length.

Key finding from Paper I: The transformer generalizes to longer sequences,
suggesting it performs genuine Bayesian inference rather than lookup.

Usage:
    python experiments/paper1_hmm/eval_length_gen.py \\
        --checkpoint checkpoints/hmm/best_model.pt \\
        --train-length 15 \\
        --test-lengths 20 25 30 35 40
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hmm import HMMConfig, HMMTokenizer, generate_hmm_instance
from src.models.gpt_mini import load_gpt_mini


def kl_divergence_bits(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Compute KL(p || q) in bits."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log2(p / q)))


def evaluate_single_length(
    model: torch.nn.Module,
    tokenizer: HMMTokenizer,
    cfg: HMMConfig,
    seq_length: int,
    n_samples: int,
    device: torch.device,
    seed: int = 42
) -> Dict[str, float]:
    """Evaluate model on sequences of a specific length."""
    model.eval()
    rng = np.random.default_rng(seed)
    
    all_kl = []
    per_position_kl = [[] for _ in range(seq_length)]
    
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc=f"K={seq_length}", leave=False):
            # Generate instance
            instance = generate_hmm_instance(cfg, rng, sequence_length=seq_length)
            tokens = tokenizer.encode_instance(instance)
            
            input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            # Forward pass
            logits, _ = model(input_ids)
            
            # Get observation positions
            ids_np = input_ids[0].cpu().numpy()
            sep_pos = None
            for j, tok in enumerate(ids_np):
                if tok == tokenizer.id_sep:
                    sep_pos = j
                    break
            
            if sep_pos is None:
                continue
                
            header_len = sep_pos + 1
            
            # Compute KL at each position
            for t in range(seq_length):
                pos = header_len + 2 * t + 1
                if pos < logits.shape[1]:
                    pred_probs = torch.softmax(logits[0, pos, :], dim=-1).cpu().numpy()
                    true_probs = instance.posteriors[t]
                    
                    kl = kl_divergence_bits(true_probs, pred_probs)
                    all_kl.append(kl)
                    per_position_kl[t].append(kl)
    
    # Compute statistics
    per_pos_mean = [np.mean(kl_list) if kl_list else 0.0 for kl_list in per_position_kl]
    
    return {
        "seq_length": seq_length,
        "mean_kl": np.mean(all_kl) if all_kl else 0.0,
        "std_kl": np.std(all_kl) if all_kl else 0.0,
        "per_position_kl": per_pos_mean,
    }


def plot_length_generalization(
    results: List[Dict], 
    train_length: int, 
    output_path: Path
):
    """Plot length generalization results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean KL vs sequence length
    lengths = [r["seq_length"] for r in results]
    mean_kls = [r["mean_kl"] for r in results]
    std_kls = [r["std_kl"] for r in results]
    
    ax1.errorbar(lengths, mean_kls, yerr=std_kls, marker='o', capsize=5, linewidth=2, markersize=8)
    ax1.axvline(x=train_length, color='red', linestyle='--', label=f'Training length (K={train_length})')
    ax1.axhline(y=0.1, color='green', linestyle=':', alpha=0.7, label='Target threshold (0.1 bits)')
    ax1.set_xlabel("Sequence Length (K)", fontsize=12)
    ax1.set_ylabel("Mean KL Divergence (bits)", fontsize=12)
    ax1.set_title("Length Generalization", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Per-position KL for different lengths
    cmap = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        K = result["seq_length"]
        per_pos = result["per_position_kl"]
        positions = list(range(1, len(per_pos) + 1))
        ax2.plot(positions, per_pos, marker='.', color=cmap[i], label=f'K={K}', alpha=0.8)
    
    ax2.set_xlabel("Position in Sequence", fontsize=12)
    ax2.set_ylabel("KL Divergence (bits)", fontsize=12)
    ax2.set_title("Per-Position Error by Sequence Length", fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate HMM length generalization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--train-length", type=int, default=15, help="Training sequence length")
    parser.add_argument("--test-lengths", type=int, nargs="+", default=[10, 15, 20, 25, 30],
                       help="Sequence lengths to test")
    parser.add_argument("--n-samples", type=int, default=200, help="Samples per length")
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_gpt_mini(args.checkpoint, device=str(device))
    
    # Initialize
    tokenizer = HMMTokenizer()
    cfg = HMMConfig(seed=args.seed)
    
    # Evaluate each length
    results = []
    for length in args.test_lengths:
        print(f"\nEvaluating K={length}...")
        result = evaluate_single_length(
            model, tokenizer, cfg, length, 
            args.n_samples, device, args.seed
        )
        results.append(result)
        
        print(f"  Mean KL: {result['mean_kl']:.4f} ± {result['std_kl']:.4f} bits")
    
    # Summary table
    print("\n" + "="*60)
    print("LENGTH GENERALIZATION SUMMARY")
    print("="*60)
    print(f"{'Length':>8} | {'Mean KL':>12} | {'Status':>15}")
    print("-"*60)
    
    for result in results:
        K = result["seq_length"]
        kl = result["mean_kl"]
        is_train = "(training)" if K == args.train_length else ""
        status = "✓ PASS" if kl < 0.1 else "✗ FAIL"
        print(f"{K:>8} | {kl:>10.4f}   | {status:>10} {is_train}")
    
    # Plot results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_length_generalization(results, args.train_length, output_dir / "hmm_length_generalization.png")
    
    # Check generalization success
    train_kl = next((r["mean_kl"] for r in results if r["seq_length"] == args.train_length), None)
    longer_lengths = [r for r in results if r["seq_length"] > args.train_length]
    
    if longer_lengths and train_kl:
        avg_longer_kl = np.mean([r["mean_kl"] for r in longer_lengths])
        ratio = avg_longer_kl / train_kl if train_kl > 0 else 0
        
        print(f"\nGeneralization ratio (longer/train): {ratio:.2f}x")
        if ratio < 2.0:
            print("✓ Strong length generalization - suggests algorithmic learning")
        else:
            print("! Moderate generalization - may indicate some memorization")


if __name__ == "__main__":
    main()
