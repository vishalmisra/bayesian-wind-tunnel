"""
Geometric analysis of HMM task: Compare LSTM vs Mamba manifolds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

PROJECT_ROOT = Path('/home/vishal/bayesg/bayesian-wind-tunnel')
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hmm import HMMConfig, HMMTokenizer, HMMDataset, collate_hmm_batch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMHMM(nn.Module):
    def __init__(self, vocab_size, num_states, d_model=256, n_layers=9):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_states)
        self.d_model = d_model
        self.n_layers = n_layers

    def forward(self, x):
        h = self.tok_emb(x)
        h, _ = self.lstm(h)
        h = self.norm(h)
        return self.head(h), h


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32) * 0.1
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_inner)
    
    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)[:, :, :L]
        x_inner = x_inner.transpose(1, 2)
        x_inner = torch.nn.functional.silu(x_inner)
        
        x_dbl = self.x_proj(x_inner)
        dt, B_param, C_param = x_dbl.split([1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)).clamp(min=1e-4, max=10.0)
        A = -torch.exp(self.A_log.clamp(max=5.0))
        
        y = self.selective_scan(x_inner, dt, A, B_param, C_param)
        y = self.norm(y)
        y = y + self.D * x_inner
        y = y * torch.nn.functional.silu(z)
        return self.out_proj(y)
    
    def selective_scan(self, u, dt, A, B, C):
        B_batch, L, d_inner = u.shape
        d_state = self.d_state
        h = torch.zeros(B_batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(L):
            dA = torch.exp((dt[:, i, :, None] * A).clamp(min=-20, max=0))
            dB_u = (dt[:, i, :, None] * B[:, i, None, :] * u[:, i, :, None]).clamp(-10, 10)
            h = h * dA + dB_u
            h = h.clamp(-100, 100)
            y = (h * C[:, i, None, :]).sum(-1)
            ys.append(y)
        return torch.stack(ys, dim=1)


class MambaHMM(nn.Module):
    def __init__(self, vocab_size, num_states, d_model=256, n_layers=9, d_state=16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_states)
        self.d_model = d_model

    def forward(self, x):
        h = self.tok_emb(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.norm(h)
        return self.head(h), h


def compute_entropy_from_posteriors(posteriors):
    """Compute entropy from posterior distributions (K, S)."""
    eps = 1e-10
    # posteriors: (batch, K, S) -> entropy: (batch, K)
    entropy = -np.sum(posteriors * np.log2(posteriors + eps), axis=-1)
    return entropy


def extract_lstm_layers(model, dataloader, tokenizer, K, n_batches=30):
    """Extract representations at each LSTM layer."""
    n_layers = model.n_layers
    layer_reps = {i: [] for i in range(n_layers + 1)}
    all_entropies = []
    
    with torch.no_grad():
        batch_count = 0
        for batch in dataloader:
            if batch_count >= n_batches:
                break
            input_ids, targets = batch  # targets: (B, K, S) posteriors
            input_ids = input_ids.to(device)
            
            # Compute entropy from posteriors
            entropies = compute_entropy_from_posteriors(targets.numpy())  # (B, K)
            
            # Get observation positions (where posteriors are computed)
            # Find separator position to get header length
            sep_pos = (input_ids[0] == tokenizer.id_sep).nonzero(as_tuple=True)[0]
            if len(sep_pos) > 0:
                header_len = sep_pos[0].item() + 1
            else:
                header_len = 52
            obs_positions = [header_len + 2 * t + 1 for t in range(K)]
            
            # Embedding
            h = model.tok_emb(input_ids)
            
            for i, t in enumerate(obs_positions):
                if t < h.size(1):
                    layer_reps[0].append(h[:, t, :].cpu().numpy())
                    all_entropies.append(entropies[:, i])
            
            # Manual LSTM forward to get intermediate layers
            B, T, D = h.shape
            h_prev = [torch.zeros(B, D, device=device) for _ in range(n_layers)]
            c_prev = [torch.zeros(B, D, device=device) for _ in range(n_layers)]
            
            layer_outputs = {i: [[] for _ in obs_positions] for i in range(n_layers)}
            
            for t_idx in range(T):
                inp = h[:, t_idx, :]
                for layer_idx in range(n_layers):
                    w_ih = getattr(model.lstm, f'weight_ih_l{layer_idx}')
                    w_hh = getattr(model.lstm, f'weight_hh_l{layer_idx}')
                    b_ih = getattr(model.lstm, f'bias_ih_l{layer_idx}')
                    b_hh = getattr(model.lstm, f'bias_hh_l{layer_idx}')
                    
                    gates = inp @ w_ih.T + b_ih + h_prev[layer_idx] @ w_hh.T + b_hh
                    i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
                    i_gate = torch.sigmoid(i_gate)
                    f_gate = torch.sigmoid(f_gate)
                    o_gate = torch.sigmoid(o_gate)
                    g_gate = torch.tanh(g_gate)
                    
                    c_prev[layer_idx] = f_gate * c_prev[layer_idx] + i_gate * g_gate
                    h_prev[layer_idx] = o_gate * torch.tanh(c_prev[layer_idx])
                    inp = h_prev[layer_idx]
                    
                    if t_idx in obs_positions:
                        pos_idx = obs_positions.index(t_idx)
                        layer_outputs[layer_idx][pos_idx].append(h_prev[layer_idx].cpu().numpy())
            
            for layer_idx in range(n_layers):
                for pos_idx in range(len(obs_positions)):
                    if layer_outputs[layer_idx][pos_idx]:
                        layer_reps[layer_idx + 1].append(layer_outputs[layer_idx][pos_idx][0])
            
            batch_count += 1
    
    all_entropies = np.concatenate(all_entropies)
    layer_reps = {k: np.vstack(v) for k, v in layer_reps.items() if len(v) > 0}
    
    return layer_reps, all_entropies


def extract_mamba_layers(model, dataloader, tokenizer, K, n_batches=30):
    """Extract representations at each Mamba layer."""
    n_layers = len(model.blocks)
    layer_reps = {i: [] for i in range(n_layers + 1)}
    all_entropies = []
    
    with torch.no_grad():
        batch_count = 0
        for batch in dataloader:
            if batch_count >= n_batches:
                break
            input_ids, targets = batch
            input_ids = input_ids.to(device)
            
            entropies = compute_entropy_from_posteriors(targets.numpy())
            
            sep_pos = (input_ids[0] == tokenizer.id_sep).nonzero(as_tuple=True)[0]
            if len(sep_pos) > 0:
                header_len = sep_pos[0].item() + 1
            else:
                header_len = 52
            obs_positions = [header_len + 2 * t + 1 for t in range(K)]
            
            h = model.tok_emb(input_ids)
            
            for i, t in enumerate(obs_positions):
                if t < h.size(1):
                    layer_reps[0].append(h[:, t, :].cpu().numpy())
                    all_entropies.append(entropies[:, i])
            
            for layer_idx, block in enumerate(model.blocks):
                h = h + block(h)
                for i, t in enumerate(obs_positions):
                    if t < h.size(1):
                        layer_reps[layer_idx + 1].append(h[:, t, :].cpu().numpy())
            
            batch_count += 1
    
    all_entropies = np.concatenate(all_entropies)
    layer_reps = {k: np.vstack(v) for k, v in layer_reps.items() if len(v) > 0}
    
    return layer_reps, all_entropies


def analyze_layer(representations, entropies):
    pca = PCA(n_components=min(10, representations.shape[1]))
    pca_coords = pca.fit_transform(representations)
    
    top_pc_var = pca.explained_variance_ratio_[0]
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dim_90 = np.argmax(cumvar >= 0.90) + 1
    
    max_corr = 0
    for i in range(min(5, pca_coords.shape[1])):
        corr = np.corrcoef(pca_coords[:, i], entropies)[0, 1]
        if not np.isnan(corr) and abs(corr) > abs(max_corr):
            max_corr = corr
    
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
    }


def main():
    print("Device:", device)
    print(f"\nHMM Geometric Analysis")
    print("="*70)
    
    print("\nSetting up HMM data...")
    K = 15  # sequence length
    hmm_cfg = HMMConfig(sequence_length=K, seed=42)
    tokenizer = HMMTokenizer()
    
    dataset = HMMDataset(2000, hmm_cfg, tokenizer, seed=42)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_hmm_batch)
    
    vocab_size = tokenizer.vocab_size
    num_states = hmm_cfg.n_states
    print(f"Vocab size: {vocab_size}, States: {num_states}, Seq length: {K}")
    
    results = {}
    layer_data = {}
    entropy_data = {}
    
    # Load LSTM
    print("\n[1/2] Loading LSTM...")
    try:
        lstm = LSTMHMM(vocab_size=vocab_size, num_states=num_states, d_model=256, n_layers=9)
        ckpt = torch.load('/home/vishal/bayesg/bayesian-wind-tunnel/logs/lstm_hmm/ckpt_final.pt',
                          map_location=device, weights_only=False)
        lstm.load_state_dict(ckpt['model'])
        lstm = lstm.to(device).eval()
        print("LSTM loaded")
        
        lstm_layers, lstm_ent = extract_lstm_layers(lstm, dataloader, tokenizer, K)
        layer_data['LSTM'] = lstm_layers
        entropy_data['LSTM'] = lstm_ent
        results['LSTM'] = {}
        
        print(f"\n{'Layer':<10} {'PC1 Var':>10} {'|r|':>10} {'R²':>10}")
        print("-" * 45)
        for layer_idx in sorted(lstm_layers.keys()):
            stats = analyze_layer(lstm_layers[layer_idx], lstm_ent)
            results['LSTM'][layer_idx] = stats
            layer_name = 'Embed' if layer_idx == 0 else f'LSTM {layer_idx}'
            print(f"{layer_name:<10} {stats['top_pc_var']:>10.3f} {abs(stats['entropy_corr']):>10.3f} {stats['linear_r2']:>10.3f}")
    except Exception as e:
        print(f"LSTM failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Load Mamba
    print("\n[2/2] Loading Mamba...")
    try:
        mamba = MambaHMM(vocab_size=vocab_size, num_states=num_states, d_model=256, n_layers=9, d_state=16)
        ckpt = torch.load('/home/vishal/bayesg/bayesian-wind-tunnel/logs/mamba_hmm/ckpt_final.pt',
                          map_location=device, weights_only=False)
        mamba.load_state_dict(ckpt['model'])
        mamba = mamba.to(device).eval()
        print("Mamba loaded")
        
        mamba_layers, mamba_ent = extract_mamba_layers(mamba, dataloader, tokenizer, K)
        layer_data['Mamba'] = mamba_layers
        entropy_data['Mamba'] = mamba_ent
        results['Mamba'] = {}
        
        print(f"\n{'Layer':<10} {'PC1 Var':>10} {'|r|':>10} {'R²':>10}")
        print("-" * 45)
        for layer_idx in sorted(mamba_layers.keys()):
            stats = analyze_layer(mamba_layers[layer_idx], mamba_ent)
            results['Mamba'][layer_idx] = stats
            layer_name = 'Embed' if layer_idx == 0 else f'Block {layer_idx}'
            print(f"{layer_name:<10} {stats['top_pc_var']:>10.3f} {abs(stats['entropy_corr']):>10.3f} {stats['linear_r2']:>10.3f}")
    except Exception as e:
        print(f"Mamba failed: {e}")
        import traceback
        traceback.print_exc()
    
    if len(results) == 0:
        print("No models loaded!")
        return
    
    # Visualizations
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {'LSTM': 'green', 'Mamba': 'orange'}
    
    ax = axes[0]
    for name, data in results.items():
        layers = sorted(data.keys())
        corrs = [abs(data[l]['entropy_corr']) for l in layers]
        ax.plot(layers, corrs, 'o-', label=f"{name}", color=colors[name], linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('|Entropy Correlation|')
    ax.set_title('Entropy-Geometry Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    ax = axes[1]
    for name, data in results.items():
        layers = sorted(data.keys())
        vars_ = [data[l]['top_pc_var'] for l in layers]
        ax.plot(layers, vars_, 'o-', label=name, color=colors[name], linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Variance in PC1')
    ax.set_title('Dimensionality Compression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    for name, data in results.items():
        layers = sorted(data.keys())
        r2s = [data[l]['linear_r2'] for l in layers]
        ax.plot(layers, r2s, 'o-', label=name, color=colors[name], linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Linear R²')
    ax.set_title('Entropy Decodability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.suptitle('HMM Task: LSTM (FAIL: 0.416 bits) vs Mamba (0.031 bits)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/vishal/bayesg/bayesian-wind-tunnel/logs/hmm_layerwise_analysis.png', dpi=150)
    print("Saved: hmm_layerwise_analysis.png")
    
    # Manifold comparison
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for row, name in enumerate(['LSTM', 'Mamba']):
        if name not in layer_data:
            continue
        layers = layer_data[name]
        ent = entropy_data[name]
        
        layer_indices = sorted(layers.keys())
        layers_to_show = [layer_indices[i] for i in [0, 2, 4, 6, min(8, len(layer_indices)-1)]]
        
        n_points = min(1500, len(ent))
        idx = np.random.choice(len(ent), n_points, replace=False)
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
                ax.set_title(f'Layer {layer_idx}', fontsize=11, fontweight='bold')
            
            if col == 0:
                mae = 0.416 if name == 'LSTM' else 0.031
                ax.set_ylabel(f'{name}\n(MAE={mae})\n\nPC2')
            else:
                ax.set_ylabel('PC2')
            ax.set_xlabel('PC1')
            
            ax.text(0.05, 0.95, f'{var1:.0f}%\n|r|={abs(corr):.2f}', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.8))
    
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.015, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label='Entropy (bits)')
    
    fig.suptitle('HMM Manifold Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.93, 0.96])
    plt.savefig('/home/vishal/bayesg/bayesian-wind-tunnel/logs/hmm_manifold_evolution.png', dpi=150, bbox_inches='tight')
    print("Saved: hmm_manifold_evolution.png")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: HMM Geometric Analysis")
    print("="*70)
    print("\nPerformance: LSTM=0.416 bits (FAIL), Mamba=0.031 bits")
    
    for name, data in results.items():
        layers = sorted(data.keys())
        final = data[max(layers)]
        print(f"\n{name} Final Layer:")
        print(f"  |r| = {abs(final['entropy_corr']):.3f}, PC1 = {final['top_pc_var']*100:.1f}%, R² = {final['linear_r2']:.3f}")

if __name__ == '__main__':
    main()
