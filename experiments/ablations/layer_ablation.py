#!/usr/bin/env python3
"""
Layer/Block Ablation Experiments

Tests whether FFN or Attention contributes more to Bayesian inference
by zeroing out each component per layer.

Key findings from Paper I:
- Bypassing entire layers has minimal impact (< 5% accuracy drop per layer)
- This suggests holographic/distributed computation across layers
- FFN and attention both contribute, with late FFN slightly more important

Usage:
    python experiments/ablations/layer_ablation.py \\
        --checkpoint checkpoints/bijection/best_model.pt \\
        --domain-size 20 \\
        --seq-length 19
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tinygpt import load_tinygpt
from src.data.bijection import sample_perm, build_sequence_from_perm
from src.utils.entropy import entropy_bits_from_logits


class ComponentAblationHook:
    """Hook to zero out a specific component (FFN or Attention)."""
    
    def __init__(self, component_type: str = 'ffn'):
        self.component_type = component_type
        self.enabled = False
    
    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        
        # Zero out the component
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        else:
            return torch.zeros_like(output)


class LayerBypassHook:
    """Hook to bypass an entire layer (return input unchanged)."""
    
    def __init__(self):
        self.enabled = False
        self.saved_input = None
    
    def pre_hook(self, module, input):
        if self.enabled:
            self.saved_input = input[0].clone()
        return input
    
    def forward_hook(self, module, input, output):
        if self.enabled and self.saved_input is not None:
            return self.saved_input
        return output


@torch.no_grad()
def evaluate_with_layer_ablation(
    model: nn.Module,
    V: int,
    L: int,
    device: torch.device,
    n_samples: int = 500,
    ablation_type: str = 'none',  # 'none', 'bypass', 'zero_attn', 'zero_ffn'
    layer_idx: Optional[int] = None,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate model with layer ablation.
    """
    model.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup hooks if needed
    hooks = []
    bypass_hook = None
    
    if ablation_type == 'bypass' and layer_idx is not None:
        if hasattr(model, 'blocks') and layer_idx < len(model.blocks):
            block = model.blocks[layer_idx]
            bypass_hook = LayerBypassHook()
            bypass_hook.enabled = True
            h1 = block.register_forward_pre_hook(bypass_hook.pre_hook)
            h2 = block.register_forward_hook(bypass_hook.forward_hook)
            hooks.extend([h1, h2])
    
    elif ablation_type == 'zero_attn' and layer_idx is not None:
        if hasattr(model, 'blocks') and layer_idx < len(model.blocks):
            block = model.blocks[layer_idx]
            hook = ComponentAblationHook('attn')
            hook.enabled = True
            h = block.attn.register_forward_hook(hook)
            hooks.append(h)
    
    elif ablation_type == 'zero_ffn' and layer_idx is not None:
        if hasattr(model, 'blocks') and layer_idx < len(model.blocks):
            block = model.blocks[layer_idx]
            hook = ComponentAblationHook('ffn')
            hook.enabled = True
            h = block.ffn.register_forward_hook(hook)
            hooks.append(h)
    
    try:
        correct = 0
        total = 0
        mae_bits_list = []
        
        for _ in range(n_samples):
            perm = sample_perm(V)
            seq = build_sequence_from_perm(perm, L)
            input_ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            
            logits = model(input_ids)
            
            query_logits = logits[0, -1, :V]
            pred = query_logits.argmax().item()
            
            query_key = seq[-1]
            true_value = perm[query_key]
            
            if pred == true_value:
                correct += 1
            total += 1
            
            pred_entropy = entropy_bits_from_logits(query_logits.unsqueeze(0)).item()
            mae_bits_list.append(abs(pred_entropy - 0.0))
        
        accuracy = correct / total if total > 0 else 0.0
        mae_bits = np.mean(mae_bits_list) if mae_bits_list else 0.0
        
    finally:
        # Clean up hooks
        for h in hooks:
            h.remove()
    
    return {
        "accuracy": accuracy,
        "mae_bits": mae_bits,
        "n_samples": total
    }


def run_layer_ablation(
    model: nn.Module,
    V: int,
    L: int,
    device: torch.device,
    n_samples: int = 500,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run comprehensive layer ablation experiments.
    """
    if hasattr(model, 'blocks'):
        n_layers = len(model.blocks)
    else:
        print("Model does not have blocks attribute")
        return pd.DataFrame()
    
    print(f"Model has {n_layers} layers")
    
    # Baseline
    print("Computing baseline...")
    baseline = evaluate_with_layer_ablation(
        model, V, L, device, n_samples, 'none', None, seed
    )
    print(f"  Baseline accuracy: {baseline['accuracy']:.4f}")
    
    results = []
    
    ablation_types = ['bypass', 'zero_attn', 'zero_ffn']
    
    for abl_type in ablation_types:
        for layer_idx in tqdm(range(n_layers), desc=f"{abl_type}"):
            metrics = evaluate_with_layer_ablation(
                model, V, L, device, n_samples, abl_type, layer_idx, seed
            )
            
            results.append({
                "layer": layer_idx,
                "ablation_type": abl_type,
                "accuracy": metrics['accuracy'],
                "acc_drop": baseline['accuracy'] - metrics['accuracy'],
                "mae_bits": metrics['mae_bits'],
                "mae_increase": metrics['mae_bits'] - baseline['mae_bits'],
            })
    
    df = pd.DataFrame(results)
    df['baseline_acc'] = baseline['accuracy']
    df['baseline_mae'] = baseline['mae_bits']
    
    return df


def run_cumulative_ablation(
    model: nn.Module,
    V: int,
    L: int,
    device: torch.device,
    n_samples: int = 500,
    seed: int = 42
) -> pd.DataFrame:
    """
    Test cumulative layer ablation (remove layers 0..k progressively).
    """
    if hasattr(model, 'blocks'):
        n_layers = len(model.blocks)
    else:
        return pd.DataFrame()
    
    results = []
    
    # We can't easily do true cumulative without model modification
    # Instead, test what happens when we bypass increasing numbers of early layers
    for n_bypass in tqdm(range(n_layers + 1), desc="Cumulative bypass"):
        if n_bypass == 0:
            metrics = evaluate_with_layer_ablation(
                model, V, L, device, n_samples, 'none', None, seed
            )
        else:
            # For simplicity, just evaluate bypassing layer n_bypass-1
            # True cumulative would require multiple simultaneous hooks
            metrics = evaluate_with_layer_ablation(
                model, V, L, device, n_samples, 'bypass', n_bypass - 1, seed
            )
        
        results.append({
            "n_layers_bypassed": n_bypass,
            "accuracy": metrics['accuracy'],
            "mae_bits": metrics['mae_bits'],
        })
    
    return pd.DataFrame(results)


def plot_layer_ablation(df: pd.DataFrame, output_path: Path):
    """Plot layer ablation results."""
    if df.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy by layer and ablation type
    ax1 = axes[0]
    ablation_types = df['ablation_type'].unique()
    n_layers = df['layer'].max() + 1
    
    x = np.arange(n_layers)
    width = 0.25
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for i, abl_type in enumerate(ablation_types):
        subset = df[df['ablation_type'] == abl_type].sort_values('layer')
        offset = (i - len(ablation_types) / 2 + 0.5) * width
        bars = ax1.bar(x + offset, subset['acc_drop'], width, 
                      label=abl_type, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel("Layer Index", fontsize=12)
    ax1.set_ylabel("Accuracy Drop", fontsize=12)
    ax1.set_title("Accuracy Drop by Layer Ablation Type", fontsize=14)
    ax1.set_xticks(x)
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Comparison of component importance
    ax2 = axes[1]
    
    for abl_type in ablation_types:
        subset = df[df['ablation_type'] == abl_type].sort_values('layer')
        ax2.plot(subset['layer'], subset['acc_drop'], 
                marker='o', label=abl_type, linewidth=2, markersize=8)
    
    ax2.set_xlabel("Layer Index", fontsize=12)
    ax2.set_ylabel("Accuracy Drop", fontsize=12)
    ax2.set_title("Component Importance Across Layers", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {output_path}")


def plot_attention_vs_ffn(df: pd.DataFrame, output_path: Path):
    """Plot attention vs FFN importance comparison."""
    if df.empty:
        return
    
    attn_df = df[df['ablation_type'] == 'zero_attn'].sort_values('layer')
    ffn_df = df[df['ablation_type'] == 'zero_ffn'].sort_values('layer')
    
    if attn_df.empty or ffn_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layers = attn_df['layer'].values
    attn_drops = attn_df['acc_drop'].values
    ffn_drops = ffn_df['acc_drop'].values
    
    width = 0.35
    x = np.arange(len(layers))
    
    bars1 = ax.bar(x - width/2, attn_drops, width, label='Attention', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, ffn_drops, width, label='FFN', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Accuracy Drop when Zeroed", fontsize=12)
    ax.set_title("Attention vs FFN Importance per Layer", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Layer/block ablation experiments")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--domain-size", type=int, default=20, help="Domain size V")
    parser.add_argument("--seq-length", type=int, default=19, help="Sequence length L")
    parser.add_argument("--n-samples", type=int, default=500, help="Samples per evaluation")
    parser.add_argument("--output-dir", type=str, default="results/ablations", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_tinygpt(args.checkpoint, device=str(device))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run layer ablation
    print("\n" + "="*60)
    print("LAYER ABLATION EXPERIMENTS")
    print("="*60)
    
    layer_df = run_layer_ablation(
        model, args.domain_size, args.seq_length,
        device, args.n_samples, args.seed
    )
    
    if not layer_df.empty:
        layer_df.to_csv(output_dir / "layer_ablation.csv", index=False)
        plot_layer_ablation(layer_df, output_dir / "layer_ablation.png")
        plot_attention_vs_ffn(layer_df, output_dir / "attn_vs_ffn.png")
        
        # Summary statistics
        print("\n" + "-"*40)
        print("SUMMARY BY COMPONENT TYPE")
        print("-"*40)
        
        for abl_type in layer_df['ablation_type'].unique():
            subset = layer_df[layer_df['ablation_type'] == abl_type]
            avg_drop = subset['acc_drop'].mean()
            max_drop = subset['acc_drop'].max()
            max_layer = subset.loc[subset['acc_drop'].idxmax(), 'layer']
            
            print(f"\n{abl_type}:")
            print(f"  Average accuracy drop: {avg_drop:.4f}")
            print(f"  Maximum drop: {max_drop:.4f} (layer {max_layer})")
        
        # Key insight: Is attention or FFN more important?
        attn_avg = layer_df[layer_df['ablation_type'] == 'zero_attn']['acc_drop'].mean()
        ffn_avg = layer_df[layer_df['ablation_type'] == 'zero_ffn']['acc_drop'].mean()
        
        print("\n" + "="*40)
        if attn_avg > ffn_avg:
            print(f"✓ Attention is more critical (avg drop {attn_avg:.4f} vs FFN {ffn_avg:.4f})")
        else:
            print(f"✓ FFN is more critical (avg drop {ffn_avg:.4f} vs Attention {attn_avg:.4f})")
        
        # Check for distributed computation
        bypass_drops = layer_df[layer_df['ablation_type'] == 'bypass']['acc_drop']
        if bypass_drops.max() < 0.1:
            print("✓ Small per-layer impact suggests holographic/distributed computation")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
