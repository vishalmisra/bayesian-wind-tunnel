"""
Layer-by-layer geometric analysis: How do entropy manifolds emerge?
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '/home/vishal/bayesg/bayesian-wind-tunnel/experiments/paper1_bijection')

from train import make_batch, TinyGPT
from train_lstm import LSTMBijection
from train_mamba import MambaBijection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
V, L = 20, 19

def compute_bayesian_entropies(x, V):
    """Compute Bayesian entropy at each value position."""
    batch_size, seq_len = x.shape
    entropies = torch.zeros(batch_size, seq_len)
    
    for b in range(batch_size):
        seen_values = set()
        for t in range(seq_len):
            if t % 2 == 1:
                n_remaining = V - len(seen_values)
                if n_remaining > 0:
                    entropies[b, t] = np.log2(n_remaining)
                val = x[b, t].item()
                if val >= V:
                    seen_values.add(val - V)
            else:
                n_remaining = V - len(seen_values)
                if n_remaining > 0:
                    entropies[b, t] = np.log2(n_remaining)
    return entropies

def analyze_layer(representations, entropies, name):
    """Analyze a single layer's representations."""
    pca = PCA(n_components=min(10, representations.shape[1]))
    pca_coords = pca.fit_transform(representations)
    
    # Top PC variance
    top_pc_var = pca.explained_variance_ratio_[0]
    
    # Cumulative variance for effective dimensionality
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dim_90 = np.argmax(cumvar >= 0.90) + 1
    
    # Best entropy correlation across PCs
    max_corr = 0
    for i in range(min(5, pca_coords.shape[1])):
        corr = np.corrcoef(pca_coords[:, i], entropies)[0, 1]
        if not np.isnan(corr) and abs(corr) > abs(max_corr):
            max_corr = corr
    
    # Linear entropy prediction R²
    X_train, X_test, y_train, y_test = train_test_split(
        representations, entropies, test_size=0.2, random_state=42
    )
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    r2 = reg.score(X_test, y_test)
    
    return {
        'top_pc_var': top_pc_var,
        'dim_90': dim_90,
        'entropy_corr': max_corr,
        'linear_r2': r2,
        'pca_coords': pca_coords
    }

def extract_transformer_layers(model, n_batches=30, batch_size=64):
    """Extract representations at each layer of transformer."""
    layer_reps = {i: [] for i in range(7)}  # embedding + 6 layers
    all_entropies = []
    
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = make_batch(V, L, batch_size, with_replacement=False, device=device)
            entropies = compute_bayesian_entropies(x.cpu(), V)
            
            # Layer 0: after embedding
            h = model.tok_emb(x) + model.pos_emb(torch.arange(x.size(1), device=device))
            
            # Collect from value positions
            for t in range(1, x.size(1), 2):
                layer_reps[0].append(h[:, t, :].cpu().numpy())
                all_entropies.append(entropies[:, t].numpy())
            
            # Layers 1-6: after each block
            for i, block in enumerate(model.blocks):
                h = block(h, model.mask)
                for t in range(1, x.size(1), 2):
                    layer_reps[i+1].append(h[:, t, :].cpu().numpy())
    
    # Stack and deduplicate entropies (same for all layers)
    all_entropies = np.concatenate(all_entropies[:len(layer_reps[0])])
    layer_reps = {k: np.vstack(v) for k, v in layer_reps.items()}
    
    return layer_reps, all_entropies

def extract_mamba_layers(model, n_batches=30, batch_size=64):
    """Extract representations at each layer of Mamba."""
    layer_reps = {i: [] for i in range(7)}
    all_entropies = []
    
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = make_batch(V, L, batch_size, with_replacement=False, device=device)
            entropies = compute_bayesian_entropies(x.cpu(), V)
            
            h = model.tok_emb(x) + model.pos_emb(torch.arange(x.size(1), device=device))
            
            for t in range(1, x.size(1), 2):
                layer_reps[0].append(h[:, t, :].cpu().numpy())
                all_entropies.append(entropies[:, t].numpy())
            
            for i, block in enumerate(model.blocks):
                h = block(h)
                for t in range(1, x.size(1), 2):
                    layer_reps[i+1].append(h[:, t, :].cpu().numpy())
    
    all_entropies = np.concatenate(all_entropies[:len(layer_reps[0])])
    layer_reps = {k: np.vstack(v) for k, v in layer_reps.items()}
    
    return layer_reps, all_entropies

def extract_lstm_layers(model, n_batches=30, batch_size=64):
    """Extract representations at each LSTM layer."""
    # LSTM has n_layers internal layers
    n_layers = model.lstm.num_layers
    layer_reps = {i: [] for i in range(n_layers + 1)}  # embedding + LSTM layers
    all_entropies = []
    
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = make_batch(V, L, batch_size, with_replacement=False, device=device)
            entropies = compute_bayesian_entropies(x.cpu(), V)
            
            # Embedding layer
            tok_emb = model.tok_emb(x)
            pos_emb = model.pos_emb(torch.arange(x.size(1), device=device))
            h = tok_emb + pos_emb
            
            for t in range(1, x.size(1), 2):
                layer_reps[0].append(h[:, t, :].cpu().numpy())
                all_entropies.append(entropies[:, t].numpy())
            
            # Run LSTM and get all layer outputs
            # We need to manually process to get intermediate layers
            B, T, D = h.shape
            h_prev = [torch.zeros(B, D, device=device) for _ in range(n_layers)]
            c_prev = [torch.zeros(B, D, device=device) for _ in range(n_layers)]
            
            layer_outputs = [[] for _ in range(n_layers)]
            
            for t_idx in range(T):
                inp = h[:, t_idx, :]
                for layer_idx in range(n_layers):
                    # Get LSTM layer weights
                    w_ih = getattr(model.lstm, f'weight_ih_l{layer_idx}')
                    w_hh = getattr(model.lstm, f'weight_hh_l{layer_idx}')
                    b_ih = getattr(model.lstm, f'bias_ih_l{layer_idx}')
                    b_hh = getattr(model.lstm, f'bias_hh_l{layer_idx}')
                    
                    gates = inp @ w_ih.T + b_ih + h_prev[layer_idx] @ w_hh.T + b_hh
                    i, f, g, o = gates.chunk(4, dim=1)
                    i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
                    g = torch.tanh(g)
                    
                    c_prev[layer_idx] = f * c_prev[layer_idx] + i * g
                    h_prev[layer_idx] = o * torch.tanh(c_prev[layer_idx])
                    
                    inp = h_prev[layer_idx]  # Input to next layer
                    
                    if t_idx % 2 == 1:  # Value position
                        layer_outputs[layer_idx].append(h_prev[layer_idx].cpu().numpy())
            
            for layer_idx in range(n_layers):
                layer_reps[layer_idx + 1].extend(layer_outputs[layer_idx])
    
    all_entropies = np.concatenate(all_entropies[:len(layer_reps[0])])
    layer_reps = {k: np.vstack(v) for k, v in layer_reps.items()}
    
    return layer_reps, all_entropies

def main():
    print("Device:", device)
    print(f"\nLayer-wise Analysis of Entropy Manifold Emergence")
    print(f"Task: Bijection (V={V}, L={L})")
    print("="*70)
    
    results = {}
    
    # Transformer
    print("\n[1/3] Analyzing Transformer layer by layer...")
    transformer = TinyGPT(vocab_size=2*V, dim=192, n_layers=6, n_heads=6, max_seq_len=2*L)
    ckpt = torch.load('/home/vishal/bayesg/bayesian-wind-tunnel/logs/bijection_v20_repl/ckpt_final.pt',
                      map_location=device, weights_only=False)
    transformer.load_state_dict(ckpt['model'])
    transformer = transformer.to(device).eval()
    
    trans_layers, trans_ent = extract_transformer_layers(transformer)
    results['Transformer'] = {}
    print(f"\n{'Layer':<10} {'Top PC Var':>12} {'Dim@90%':>10} {'Entropy r':>12} {'Linear R²':>12}")
    print("-" * 60)
    for layer_idx in sorted(trans_layers.keys()):
        stats = analyze_layer(trans_layers[layer_idx], trans_ent, f"Layer {layer_idx}")
        results['Transformer'][layer_idx] = stats
        layer_name = 'Embed' if layer_idx == 0 else f'Block {layer_idx}'
        print(f"{layer_name:<10} {stats['top_pc_var']:>12.4f} {stats['dim_90']:>10d} {stats['entropy_corr']:>12.4f} {stats['linear_r2']:>12.4f}")
    
    # Mamba
    print("\n[2/3] Analyzing Mamba layer by layer...")
    mamba = MambaBijection(vocab_size=2*V, d_model=192, n_layers=6, max_seq_len=2*L)
    ckpt = torch.load('/home/vishal/bayesg/bayesian-wind-tunnel/logs/mamba_bijection_v20_stable/ckpt_final.pt',
                      map_location=device, weights_only=False)
    mamba.load_state_dict(ckpt['model'])
    mamba = mamba.to(device).eval()
    
    mamba_layers, mamba_ent = extract_mamba_layers(mamba)
    results['Mamba'] = {}
    print(f"\n{'Layer':<10} {'Top PC Var':>12} {'Dim@90%':>10} {'Entropy r':>12} {'Linear R²':>12}")
    print("-" * 60)
    for layer_idx in sorted(mamba_layers.keys()):
        stats = analyze_layer(mamba_layers[layer_idx], mamba_ent, f"Layer {layer_idx}")
        results['Mamba'][layer_idx] = stats
        layer_name = 'Embed' if layer_idx == 0 else f'Block {layer_idx}'
        print(f"{layer_name:<10} {stats['top_pc_var']:>12.4f} {stats['dim_90']:>10d} {stats['entropy_corr']:>12.4f} {stats['linear_r2']:>12.4f}")
    
    # LSTM
    print("\n[3/3] Analyzing LSTM layer by layer...")
    lstm = LSTMBijection(vocab_size=2*V, d_model=192, n_layers=6, max_seq_len=2*L)
    ckpt = torch.load('/home/vishal/bayesg/bayesian-wind-tunnel/logs/lstm_bijection_v20/ckpt_final.pt',
                      map_location=device, weights_only=False)
    lstm.load_state_dict(ckpt['model'])
    lstm = lstm.to(device).eval()
    
    lstm_layers, lstm_ent = extract_lstm_layers(lstm)
    results['LSTM'] = {}
    print(f"\n{'Layer':<10} {'Top PC Var':>12} {'Dim@90%':>10} {'Entropy r':>12} {'Linear R²':>12}")
    print("-" * 60)
    for layer_idx in sorted(lstm_layers.keys()):
        stats = analyze_layer(lstm_layers[layer_idx], lstm_ent, f"Layer {layer_idx}")
        results['LSTM'][layer_idx] = stats
        layer_name = 'Embed' if layer_idx == 0 else f'LSTM {layer_idx}'
        print(f"{layer_name:<10} {stats['top_pc_var']:>12.4f} {stats['dim_90']:>10d} {stats['entropy_corr']:>12.4f} {stats['linear_r2']:>12.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Entropy correlation by layer
    ax = axes[0, 0]
    for name, data in results.items():
        layers = sorted(data.keys())
        corrs = [abs(data[l]['entropy_corr']) for l in layers]
        ax.plot(layers, corrs, 'o-', label=name, linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('|Entropy Correlation| with best PC')
    ax.set_title('Entropy Structure Emergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Plot 2: Top PC variance by layer
    ax = axes[0, 1]
    for name, data in results.items():
        layers = sorted(data.keys())
        vars_ = [data[l]['top_pc_var'] for l in layers]
        ax.plot(layers, vars_, 'o-', label=name, linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Variance in PC1')
    ax.set_title('Representation Dimensionality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Linear R² by layer
    ax = axes[1, 0]
    for name, data in results.items():
        layers = sorted(data.keys())
        r2s = [data[l]['linear_r2'] for l in layers]
        ax.plot(layers, r2s, 'o-', label=name, linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Linear Entropy Prediction R²')
    ax.set_title('Entropy Decodability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Plot 4: Effective dimensionality
    ax = axes[1, 1]
    for name, data in results.items():
        layers = sorted(data.keys())
        dims = [data[l]['dim_90'] for l in layers]
        ax.plot(layers, dims, 'o-', label=name, linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Dimensions for 90% variance')
    ax.set_title('Effective Dimensionality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/vishal/bayesg/bayesian-wind-tunnel/logs/layerwise_analysis.png', dpi=150)
    print("\nVisualization saved to logs/layerwise_analysis.png")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: How Entropy Manifolds Emerge")
    print("="*70)
    
    for name, data in results.items():
        layers = sorted(data.keys())
        corrs = [abs(data[l]['entropy_corr']) for l in layers]
        max_layer = layers[np.argmax(corrs)]
        max_corr = max(corrs)
        print(f"\n{name}:")
        print(f"  Peak entropy correlation: r={max_corr:.4f} at layer {max_layer}")
        print(f"  Embedding correlation: r={abs(data[0]['entropy_corr']):.4f}")
        print(f"  Final layer correlation: r={abs(data[max(layers)]['entropy_corr']):.4f}")

if __name__ == '__main__':
    main()
