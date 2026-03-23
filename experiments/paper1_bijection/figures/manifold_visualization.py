"""
Visualize manifold evolution layer by layer for each architecture.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
sys.path.insert(0, '/home/vishal/bayesg/bayesian-wind-tunnel/experiments/paper1_bijection')

from train import make_batch, TinyGPT
from train_lstm import LSTMBijection
from train_mamba import MambaBijection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
V, L = 20, 19

def compute_bayesian_entropies(x, V):
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

def extract_all_layers(model, model_type, n_batches=40, batch_size=64):
    """Extract representations at all layers."""
    if model_type == 'lstm':
        n_layers = model.lstm.num_layers
        layer_reps = {i: [] for i in range(n_layers + 1)}
    else:
        layer_reps = {i: [] for i in range(7)}
    all_entropies = []
    
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = make_batch(V, L, batch_size, with_replacement=False, device=device)
            entropies = compute_bayesian_entropies(x.cpu(), V)
            
            if model_type == 'transformer':
                h = model.tok_emb(x) + model.pos_emb(torch.arange(x.size(1), device=device))
                for t in range(1, x.size(1), 2):
                    layer_reps[0].append(h[:, t, :].cpu().numpy())
                    all_entropies.append(entropies[:, t].numpy())
                for i, block in enumerate(model.blocks):
                    h = block(h, model.mask)
                    for t in range(1, x.size(1), 2):
                        layer_reps[i+1].append(h[:, t, :].cpu().numpy())
                        
            elif model_type == 'mamba':
                h = model.tok_emb(x) + model.pos_emb(torch.arange(x.size(1), device=device))
                for t in range(1, x.size(1), 2):
                    layer_reps[0].append(h[:, t, :].cpu().numpy())
                    all_entropies.append(entropies[:, t].numpy())
                for i, block in enumerate(model.blocks):
                    h = block(h)
                    for t in range(1, x.size(1), 2):
                        layer_reps[i+1].append(h[:, t, :].cpu().numpy())
                        
            elif model_type == 'lstm':
                tok_emb = model.tok_emb(x)
                pos_emb = model.pos_emb(torch.arange(x.size(1), device=device))
                h = tok_emb + pos_emb
                for t in range(1, x.size(1), 2):
                    layer_reps[0].append(h[:, t, :].cpu().numpy())
                    all_entropies.append(entropies[:, t].numpy())
                
                B, T, D = h.shape
                n_layers = model.lstm.num_layers
                h_prev = [torch.zeros(B, D, device=device) for _ in range(n_layers)]
                c_prev = [torch.zeros(B, D, device=device) for _ in range(n_layers)]
                
                for t_idx in range(T):
                    inp = h[:, t_idx, :]
                    for layer_idx in range(n_layers):
                        w_ih = getattr(model.lstm, f'weight_ih_l{layer_idx}')
                        w_hh = getattr(model.lstm, f'weight_hh_l{layer_idx}')
                        b_ih = getattr(model.lstm, f'bias_ih_l{layer_idx}')
                        b_hh = getattr(model.lstm, f'bias_hh_l{layer_idx}')
                        
                        gates = inp @ w_ih.T + b_ih + h_prev[layer_idx] @ w_hh.T + b_hh
                        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
                        i_gate, f_gate, o_gate = torch.sigmoid(i_gate), torch.sigmoid(f_gate), torch.sigmoid(o_gate)
                        g_gate = torch.tanh(g_gate)
                        
                        c_prev[layer_idx] = f_gate * c_prev[layer_idx] + i_gate * g_gate
                        h_prev[layer_idx] = o_gate * torch.tanh(c_prev[layer_idx])
                        inp = h_prev[layer_idx]
                        
                        if t_idx % 2 == 1:
                            layer_reps[layer_idx + 1].append(h_prev[layer_idx].cpu().numpy())
    
    all_entropies = np.concatenate(all_entropies[:len(layer_reps[0])])
    layer_reps = {k: np.vstack(v) for k, v in layer_reps.items()}
    return layer_reps, all_entropies

def create_manifold_figure(layer_reps, entropies, model_name, layers_to_show):
    """Create a figure showing manifold at selected layers."""
    n_layers = len(layers_to_show)
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    # Subsample for cleaner visualization
    n_points = min(3000, len(entropies))
    idx = np.random.choice(len(entropies), n_points, replace=False)
    
    for ax, layer_idx in zip(axes, layers_to_show):
        reps = layer_reps[layer_idx][idx]
        ent = entropies[idx]
        
        pca = PCA(n_components=2)
        coords = pca.fit_transform(reps)
        
        # Compute correlation
        corr = np.corrcoef(coords[:, 0], ent)[0, 1]
        if np.isnan(corr):
            corr = 0
        
        scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                           c=ent, cmap='viridis', alpha=0.5, s=8)
        
        if layer_idx == 0:
            title = f'Embedding'
        else:
            title = f'Layer {layer_idx}'
        
        var1 = pca.explained_variance_ratio_[0] * 100
        ax.set_title(f'{title}\nPC1: {var1:.1f}%, r={abs(corr):.2f}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        
        # Make axes equal for better visualization
        ax.set_aspect('equal', 'datalim')
    
    plt.colorbar(scatter, ax=axes[-1], label='Entropy (bits)')
    fig.suptitle(f'{model_name} Manifold Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def main():
    print("Device:", device)
    print("\nExtracting representations for manifold visualization...")
    
    # Load models
    print("\nLoading Transformer...")
    transformer = TinyGPT(vocab_size=2*V, dim=192, n_layers=6, n_heads=6, max_seq_len=2*L)
    ckpt = torch.load('/home/vishal/bayesg/bayesian-wind-tunnel/logs/bijection_v20_repl/ckpt_final.pt',
                      map_location=device, weights_only=False)
    transformer.load_state_dict(ckpt['model'])
    transformer = transformer.to(device).eval()
    
    print("Loading Mamba...")
    mamba = MambaBijection(vocab_size=2*V, d_model=192, n_layers=6, max_seq_len=2*L)
    ckpt = torch.load('/home/vishal/bayesg/bayesian-wind-tunnel/logs/mamba_bijection_v20_stable/ckpt_final.pt',
                      map_location=device, weights_only=False)
    mamba.load_state_dict(ckpt['model'])
    mamba = mamba.to(device).eval()
    
    print("Loading LSTM...")
    lstm = LSTMBijection(vocab_size=2*V, d_model=192, n_layers=6, max_seq_len=2*L)
    ckpt = torch.load('/home/vishal/bayesg/bayesian-wind-tunnel/logs/lstm_bijection_v20/ckpt_final.pt',
                      map_location=device, weights_only=False)
    lstm.load_state_dict(ckpt['model'])
    lstm = lstm.to(device).eval()
    
    # Extract representations
    print("\nExtracting Transformer representations...")
    trans_layers, trans_ent = extract_all_layers(transformer, 'transformer')
    
    print("Extracting Mamba representations...")
    mamba_layers, mamba_ent = extract_all_layers(mamba, 'mamba')
    
    print("Extracting LSTM representations...")
    lstm_layers, lstm_ent = extract_all_layers(lstm, 'lstm')
    
    # Create individual model figures
    layers_to_show = [0, 1, 3, 6]  # Embed, early, middle, late
    
    print("\nCreating visualizations...")
    
    fig = create_manifold_figure(trans_layers, trans_ent, 'TRANSFORMER', layers_to_show)
    fig.savefig('/home/vishal/bayesg/bayesian-wind-tunnel/logs/manifold_transformer.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    fig = create_manifold_figure(mamba_layers, mamba_ent, 'MAMBA', layers_to_show)
    fig.savefig('/home/vishal/bayesg/bayesian-wind-tunnel/logs/manifold_mamba.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    fig = create_manifold_figure(lstm_layers, lstm_ent, 'LSTM', layers_to_show)
    fig.savefig('/home/vishal/bayesg/bayesian-wind-tunnel/logs/manifold_lstm.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Create combined comparison figure: final layer of each
    print("\nCreating combined comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    n_points = 3000
    
    for ax, (name, layers, ent) in zip(axes, [
        ('Transformer', trans_layers, trans_ent),
        ('Mamba', mamba_layers, mamba_ent),
        ('LSTM', lstm_layers, lstm_ent)
    ]):
        idx = np.random.choice(len(ent), min(n_points, len(ent)), replace=False)
        reps = layers[6][idx]  # Final layer
        e = ent[idx]
        
        pca = PCA(n_components=2)
        coords = pca.fit_transform(reps)
        
        corr = np.corrcoef(coords[:, 0], e)[0, 1]
        if np.isnan(corr):
            corr = 0
        var1 = pca.explained_variance_ratio_[0] * 100
        
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=e, cmap='viridis', alpha=0.5, s=8)
        ax.set_title(f'{name}\nPC1: {var1:.1f}% var, |r|={abs(corr):.2f}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    
    plt.colorbar(scatter, ax=axes[-1], label='Entropy (bits)')
    fig.suptitle('Final Layer Manifolds (Layer 6)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig('/home/vishal/bayesg/bayesian-wind-tunnel/logs/manifold_comparison_final.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Create the key comparison: layer evolution side by side
    print("\nCreating evolution comparison...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    models = [
        ('Transformer', trans_layers, trans_ent),
        ('Mamba', mamba_layers, mamba_ent),
        ('LSTM', lstm_layers, lstm_ent)
    ]
    layers_to_show = [0, 1, 3, 6]
    
    for row, (name, layers, ent) in enumerate(models):
        idx = np.random.choice(len(ent), min(n_points, len(ent)), replace=False)
        e = ent[idx]
        
        for col, layer_idx in enumerate(layers_to_show):
            ax = axes[row, col]
            reps = layers[layer_idx][idx]
            
            pca = PCA(n_components=2)
            coords = pca.fit_transform(reps)
            
            corr = np.corrcoef(coords[:, 0], e)[0, 1]
            if np.isnan(corr):
                corr = 0
            var1 = pca.explained_variance_ratio_[0] * 100
            
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=e, cmap='viridis', alpha=0.4, s=5)
            
            if row == 0:
                if layer_idx == 0:
                    ax.set_title('Embedding', fontsize=12, fontweight='bold')
                else:
                    ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
            
            if col == 0:
                ax.set_ylabel(f'{name}\n\nPC2', fontsize=11)
            else:
                ax.set_ylabel('PC2')
            ax.set_xlabel('PC1')
            
            # Add stats as text
            ax.text(0.05, 0.95, f'PC1: {var1:.0f}%\n|r|: {abs(corr):.2f}', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label='Entropy (bits)')
    
    fig.suptitle('Manifold Evolution: Embedding → Layer 1 → Layer 3 → Layer 6', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    fig.savefig('/home/vishal/bayesg/bayesian-wind-tunnel/logs/manifold_evolution_grid.png', dpi=150, bbox_inches='tight')
    
    print("\nSaved visualizations:")
    print("  - manifold_transformer.png")
    print("  - manifold_mamba.png")
    print("  - manifold_lstm.png")
    print("  - manifold_comparison_final.png")
    print("  - manifold_evolution_grid.png")

if __name__ == '__main__':
    main()
