"""
Geometric analysis of LSTM, Mamba, and Transformer representations.
Compare value manifolds and routing structures across architectures.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import random
import math
sys.path.insert(0, '/home/vishal/bayesg/bayesian-wind-tunnel/experiments/paper1_bijection')

from train import make_batch, TinyGPT
from train_lstm import LSTMBijection
from train_mamba import MambaBijection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

V, L = 20, 19

def compute_bayesian_entropies(x, V):
    """Compute Bayesian entropy at each position in bijection sequence.
    
    For bijection with unique keys (without replacement):
    - At position 2k-1 (k-th value), entropy = log2(V - k + 1)
    - After k unique observations, V-k values remain possible
    """
    batch_size, seq_len = x.shape
    entropies = torch.zeros(batch_size, seq_len)
    
    for b in range(batch_size):
        seen_values = set()
        for t in range(seq_len):
            if t % 2 == 1:  # Value position (odd indices)
                n_remaining = V - len(seen_values)
                if n_remaining > 0:
                    entropies[b, t] = np.log2(n_remaining)
                # Track the value we just saw
                val = x[b, t].item()
                if val >= V:  # Values are encoded as V + actual_value
                    seen_values.add(val - V)
            else:  # Key/query position (even indices)
                # Query entropy is same as next value position would be
                n_remaining = V - len(seen_values)
                if n_remaining > 0:
                    entropies[b, t] = np.log2(n_remaining)
    
    return entropies

def load_transformer():
    model = TinyGPT(vocab_size=2*V, dim=192, n_layers=6, n_heads=6, max_seq_len=2*L)
    ckpt = torch.load('/home/vishal/bayesg/bayesian-wind-tunnel/logs/bijection_v20_repl/ckpt_final.pt', 
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval()

def load_lstm():
    model = LSTMBijection(vocab_size=2*V, d_model=192, n_layers=6, max_seq_len=2*L)
    ckpt = torch.load('/home/vishal/bayesg/bayesian-wind-tunnel/logs/lstm_bijection_v20/ckpt_final.pt',
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval()

def load_mamba():
    model = MambaBijection(vocab_size=2*V, d_model=192, n_layers=6, max_seq_len=2*L)
    ckpt = torch.load('/home/vishal/bayesg/bayesian-wind-tunnel/logs/mamba_bijection_v20_stable/ckpt_final.pt',
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval()

def extract_all_positions(model, model_type, n_batches=50, batch_size=64):
    """Extract hidden states from ALL positions, with their entropies."""
    all_hidden = []
    all_entropies = []
    all_positions = []
    
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = make_batch(V, L, batch_size, with_replacement=False, device=device)
            entropies = compute_bayesian_entropies(x.cpu(), V)
            
            if model_type == 'transformer':
                h = model.tok_emb(x) + model.pos_emb(torch.arange(x.size(1), device=device))
                for block in model.blocks:
                    h = block(h, model.mask)
                h = model.norm(h)
            elif model_type == 'lstm':
                _, _, h = model(x)
            elif model_type == 'mamba':
                h = model.tok_emb(x) + model.pos_emb(torch.arange(x.size(1), device=device))
                for block in model.blocks:
                    h = block(h)
                h = model.norm(h)
            
            # Collect value positions (odd indices) where entropy varies
            for t in range(1, x.size(1), 2):  # Value positions only
                all_hidden.append(h[:, t, :].cpu().numpy())
                all_entropies.append(entropies[:, t].numpy())
                all_positions.extend([t] * batch_size)
    
    return np.vstack(all_hidden), np.concatenate(all_entropies), np.array(all_positions)

def analyze_manifold(representations, entropies, name):
    """Analyze the manifold structure of representations."""
    print(f"\n{'='*60}")
    print(f"Geometric Analysis: {name}")
    print(f"{'='*60}")
    print(f"Samples: {representations.shape[0]}, Dimensions: {representations.shape[1]}")
    print(f"Entropy range: [{entropies.min():.2f}, {entropies.max():.2f}] bits")
    
    # PCA analysis
    pca = PCA(n_components=min(10, representations.shape[1]))
    pca_coords = pca.fit_transform(representations)
    
    print(f"\nPCA Explained Variance Ratios:")
    for i, var in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.1f}%)")
    
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dim_90 = np.argmax(cumvar >= 0.90) + 1
    dim_95 = np.argmax(cumvar >= 0.95) + 1
    
    print(f"\nEffective Dimensionality:")
    print(f"  90% variance: {dim_90} dims")
    print(f"  95% variance: {dim_95} dims")
    
    # Entropy correlation with PCs
    print(f"\nEntropy Correlation with Principal Components:")
    max_corr = 0
    max_corr_pc = 0
    for i in range(min(5, pca_coords.shape[1])):
        corr = np.corrcoef(pca_coords[:, i], entropies)[0, 1]
        if not np.isnan(corr):
            print(f"  PC{i+1} vs Entropy: r = {corr:.4f}")
            if abs(corr) > abs(max_corr):
                max_corr = corr
                max_corr_pc = i + 1
    
    print(f"\n  => Strongest entropy signal in PC{max_corr_pc} (r={max_corr:.4f})")
    
    # Linear regression: can we predict entropy from representations?
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        representations, entropies, test_size=0.2, random_state=42
    )
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    r2_train = reg.score(X_train, y_train)
    r2_test = reg.score(X_test, y_test)
    print(f"\nLinear Entropy Prediction (R²):")
    print(f"  Train: {r2_train:.4f}")
    print(f"  Test:  {r2_test:.4f}")
    
    # Check for entropy ordering along the entropy-correlated PC
    entropy_pc = max_corr_pc - 1
    entropy_bins = np.digitize(entropies, np.linspace(entropies.min(), entropies.max() + 0.001, 10))
    bin_means_pc = []
    bin_entropy_means = []
    for b in range(1, 11):
        mask = entropy_bins == b
        if mask.sum() > 5:
            bin_means_pc.append(pca_coords[mask, entropy_pc].mean())
            bin_entropy_means.append(entropies[mask].mean())
    
    if len(bin_means_pc) > 2:
        diffs = np.diff(bin_means_pc)
        monotonic_score = np.abs(np.sum(np.sign(diffs))) / len(diffs)
        print(f"\nEntropy Ordering Score (PC{max_corr_pc}): {monotonic_score:.4f}")
        print(f"  (1.0 = perfectly monotonic manifold, 0.0 = random)")
    
    return pca_coords, pca, max_corr

def main():
    print("Device:", device)
    print(f"\nAnalyzing bijection models (V={V}, L={L})")
    print("Extracting representations from ALL value positions")
    print("="*60)
    
    results = {}
    summary = {}
    
    # Analyze Transformer
    print("\n[1/3] Loading and analyzing Transformer...")
    try:
        transformer = load_transformer()
        trans_hidden, trans_ent, _ = extract_all_positions(transformer, 'transformer')
        trans_pca, _, trans_corr = analyze_manifold(trans_hidden, trans_ent, "Transformer")
        results['transformer'] = (trans_hidden, trans_ent, trans_pca)
        summary['transformer'] = trans_corr
    except Exception as e:
        print(f"Transformer analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Analyze LSTM
    print("\n[2/3] Loading and analyzing LSTM...")
    try:
        lstm = load_lstm()
        lstm_hidden, lstm_ent, _ = extract_all_positions(lstm, 'lstm')
        lstm_pca, _, lstm_corr = analyze_manifold(lstm_hidden, lstm_ent, "LSTM")
        results['lstm'] = (lstm_hidden, lstm_ent, lstm_pca)
        summary['lstm'] = lstm_corr
    except Exception as e:
        print(f"LSTM analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Analyze Mamba
    print("\n[3/3] Loading and analyzing Mamba...")
    try:
        mamba = load_mamba()
        mamba_hidden, mamba_ent, _ = extract_all_positions(mamba, 'mamba')
        mamba_pca, _, mamba_corr = analyze_manifold(mamba_hidden, mamba_ent, "Mamba")
        results['mamba'] = (mamba_hidden, mamba_ent, mamba_pca)
        summary['mamba'] = mamba_corr
    except Exception as e:
        print(f"Mamba analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Create visualization
    if len(results) >= 1:
        fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
        if len(results) == 1:
            axes = [axes]
        
        for ax, (name, (repr_data, ent, pca_coords)) in zip(axes, results.items()):
            scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                               c=ent, cmap='viridis', alpha=0.3, s=5)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            corr_str = f"r={summary[name]:.3f}" if name in summary else ""
            ax.set_title(f'{name.upper()}\n{corr_str}')
            plt.colorbar(scatter, ax=ax, label='Entropy (bits)')
        
        plt.tight_layout()
        plt.savefig('/home/vishal/bayesg/bayesian-wind-tunnel/logs/geometric_comparison.png', dpi=150)
        print("\nVisualization saved to logs/geometric_comparison.png")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: ENTROPY-GEOMETRY CORRELATION")
    print("="*60)
    for name, corr in summary.items():
        print(f"  {name:12s}: r = {corr:.4f}")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    if len(summary) >= 2:
        corrs = list(summary.values())
        if all(abs(c) > 0.8 for c in corrs):
            print("\n=> ALL architectures show strong entropy-geometry correlation!")
            print("   This suggests the 1D entropy manifold is UNIVERSAL,")
            print("   not specific to attention mechanisms.")
        elif any(abs(c) > 0.8 for c in corrs) and any(abs(c) < 0.5 for c in corrs):
            print("\n=> MIXED results: some architectures show entropy structure, others don't.")
            print("   The geometric structure may depend on the mechanism.")
        else:
            print("\n=> Results inconclusive or weak correlations across the board.")

if __name__ == '__main__':
    main()
