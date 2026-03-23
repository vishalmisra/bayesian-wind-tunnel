"""
HMM figures with better color contrast
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

PROJECT_ROOT = Path('/home/vishal/bayesg/bayesian-wind-tunnel')
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hmm import HMMConfig, HMMTokenizer, HMMDataset, collate_hmm_batch
from src.models.gpt_mini import GPTMini, GPTMiniConfig
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model definitions (same as before)
class LSTMHMM(nn.Module):
    def __init__(self, vocab_size, num_states, d_model=256, n_layers=9):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_states)
        self.d_model = d_model
        self.n_layers = n_layers

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
        x_inner = F.silu(self.conv1d(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2))
        x_dbl = self.x_proj(x_inner)
        dt, B_param, C_param = x_dbl.split([1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)).clamp(min=1e-4, max=10.0)
        A = -torch.exp(self.A_log.clamp(max=5.0))
        y = self.selective_scan(x_inner, dt, A, B_param, C_param)
        y = self.norm(y) + self.D * x_inner
        return self.out_proj(y * F.silu(z))
    
    def selective_scan(self, u, dt, A, B, C):
        B_batch, L, d_inner = u.shape
        h = torch.zeros(B_batch, d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(L):
            dA = torch.exp((dt[:, i, :, None] * A).clamp(min=-20, max=0))
            dB_u = (dt[:, i, :, None] * B[:, i, None, :] * u[:, i, :, None]).clamp(-10, 10)
            h = (h * dA + dB_u).clamp(-100, 100)
            ys.append((h * C[:, i, None, :]).sum(-1))
        return torch.stack(ys, dim=1)

class MambaHMM(nn.Module):
    def __init__(self, vocab_size, num_states, d_model=256, n_layers=9, d_state=16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_states)
        self.d_model = d_model

def compute_entropy(posteriors):
    eps = 1e-10
    return -np.sum(posteriors * np.log2(posteriors + eps), axis=-1)

def extract_final_layer(model, model_type, dataloader, tokenizer, K, n_batches=40):
    all_reps = []
    all_ent = []
    
    with torch.no_grad():
        batch_count = 0
        for batch in dataloader:
            if batch_count >= n_batches:
                break
            input_ids, targets = batch
            input_ids = input_ids.to(device)
            ent = compute_entropy(targets.numpy())
            
            sep_pos = (input_ids[0] == tokenizer.id_sep).nonzero(as_tuple=True)[0][0].item() + 1
            obs_pos = [sep_pos + 2*t + 1 for t in range(K)]
            
            if model_type == 'transformer':
                h = model.tok_emb(input_ids) + model.pos_emb(torch.arange(input_ids.size(1), device=device))
                mask = torch.triu(torch.ones(input_ids.size(1), input_ids.size(1), device=device, dtype=torch.bool), diagonal=1)
                for layer in model.encoder.layers:
                    h = layer(h, src_mask=mask, is_causal=True)
                h = model.ln_f(h)
            elif model_type == 'mamba':
                h = model.tok_emb(input_ids)
                for block in model.blocks:
                    h = h + block(h)
                h = model.norm(h)
            elif model_type == 'lstm':
                h = model.tok_emb(input_ids)
                h, _ = model.lstm(h)
                h = model.norm(h)
            
            for i, t in enumerate(obs_pos):
                all_reps.append(h[:, t, :].cpu().numpy())
                all_ent.append(ent[:, i])
            batch_count += 1
    
    return np.vstack(all_reps), np.concatenate(all_ent)

def main():
    print(f"Device: {device}")
    K = 15
    hmm_cfg = HMMConfig(sequence_length=K, seed=42)
    tokenizer = HMMTokenizer()
    dataset = HMMDataset(2500, hmm_cfg, tokenizer, seed=42)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_hmm_batch)
    
    # Load models
    print("Loading models...")
    
    config = GPTMiniConfig(vocab_size=tokenizer.vocab_size, d_model=256, n_heads=8, n_layers=9, 
                           num_states=hmm_cfg.n_states, max_seq_len=1024)
    transformer = GPTMini(config).to(device)
    transformer.load_state_dict(torch.load('logs/transformer_hmm/best_model.pt', map_location=device, weights_only=False)['model'])
    transformer.eval()
    
    mamba = MambaHMM(tokenizer.vocab_size, hmm_cfg.n_states, d_model=256, n_layers=9).to(device)
    mamba.load_state_dict(torch.load('logs/mamba_hmm/ckpt_final.pt', map_location=device, weights_only=False)['model'])
    mamba.eval()
    
    lstm = LSTMHMM(tokenizer.vocab_size, hmm_cfg.n_states, d_model=256, n_layers=9).to(device)
    lstm.load_state_dict(torch.load('logs/lstm_hmm/ckpt_final.pt', map_location=device, weights_only=False)['model'])
    lstm.eval()
    
    # Extract final layer representations
    print("Extracting representations...")
    t_reps, t_ent = extract_final_layer(transformer, 'transformer', dataloader, tokenizer, K)
    m_reps, m_ent = extract_final_layer(mamba, 'mamba', dataloader, tokenizer, K)
    l_reps, l_ent = extract_final_layer(lstm, 'lstm', dataloader, tokenizer, K)
    
    print(f"Entropy range: {t_ent.min():.2f} to {t_ent.max():.2f}")
    
    # Create figure with better colors
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Use percentile-based normalization for better contrast
    all_ent = np.concatenate([t_ent, m_ent, l_ent])
    vmin, vmax = np.percentile(all_ent, 5), np.percentile(all_ent, 95)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    data = [
        ('Transformer', t_reps, t_ent, 0.049),
        ('Mamba', m_reps, m_ent, 0.031),
        ('LSTM', l_reps, l_ent, 0.416)
    ]
    
    for ax, (name, reps, ent, mae) in zip(axes, data):
        idx = np.random.choice(len(ent), min(3000, len(ent)), replace=False)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(reps[idx])
        
        corr = np.corrcoef(coords[:, 0], ent[idx])[0, 1]
        if np.isnan(corr): corr = 0
        
        # Use RdYlBu_r colormap for better contrast (red=high, blue=low)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=ent[idx], cmap='RdYlBu_r', 
                       norm=norm, alpha=0.6, s=15, edgecolors='none')
        
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title(f'{name}\nMAE = {mae} bits', fontsize=14, fontweight='bold')
        
        # Add stats
        pc1_var = pca.explained_variance_ratio_[0] * 100
        ax.text(0.05, 0.95, f'PC1: {pc1_var:.0f}%\n|r|: {abs(corr):.2f}', 
               transform=ax.transAxes, fontsize=11, va='top',
               bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.colorbar(sc, ax=axes, label='Entropy (bits)', shrink=0.8)
    fig.suptitle('HMM Final Layer: Why LSTM Fails', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('logs/hmm_final_layer_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: hmm_final_layer_comparison.png")
    
    # Also create the evolution grid with better colors
    print("\nCreating evolution grid...")

if __name__ == '__main__':
    main()
