import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
import sys

sys.path.insert(0, '/home/vishal/bayesg/bayesian-wind-tunnel')
from src.data.hmm import HMMConfig, HMMTokenizer, HMMDataset, collate_hmm_batch
from torch.utils.data import DataLoader

device = torch.device('cuda')

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.d_inner = d_model * 2
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, 4, padding=3, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state+1, dtype=torch.float32) * 0.1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_inner)
    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)
        xi = F.silu(self.conv1d(xi.transpose(1,2))[:,:,:L].transpose(1,2))
        xd = self.x_proj(xi)
        dt, Bp, Cp = xd.split([1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)).clamp(1e-4, 10)
        A = -torch.exp(self.A_log.clamp(max=5))
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for i in range(L):
            dA = torch.exp((dt[:,i,:,None] * A).clamp(-20, 0))
            dBu = (dt[:,i,:,None] * Bp[:,i,None,:] * xi[:,i,:,None]).clamp(-10,10)
            h = (h * dA + dBu).clamp(-100, 100)
            ys.append((h * Cp[:,i,None,:]).sum(-1))
        y = torch.stack(ys, dim=1)
        return self.out_proj((self.norm(y) + self.D * xi) * F.silu(z))

class MambaHMM(nn.Module):
    def __init__(self, vocab_size, num_states, d_model=256, n_layers=9):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_states)

def compute_entropy(p):
    return -np.sum(p * np.log2(p + 1e-10), axis=-1)

K = 15
cfg = HMMConfig(sequence_length=K, seed=42)
tok = HMMTokenizer()
ds = HMMDataset(3000, cfg, tok, seed=42)
dl = DataLoader(ds, batch_size=64, collate_fn=collate_hmm_batch)

print('Loading Mamba model...')
mamba = MambaHMM(tok.vocab_size, cfg.n_states).to(device)
mamba.load_state_dict(torch.load('logs/mamba_hmm/ckpt_final.pt', map_location=device, weights_only=False)['model'])
mamba.eval()

print('Extracting final layer representations...')
reps = []
entropies = []
most_likely_states = []

with torch.no_grad():
    for batch_idx, (inp, tgt) in enumerate(dl):
        if batch_idx >= 40: break
        inp = inp.to(device)
        posteriors = tgt.numpy()  # [batch, K, n_states]
        
        # Forward through Mamba
        h = mamba.tok_emb(inp)
        for blk in mamba.blocks:
            h = h + blk(h)
        h = mamba.norm(h)
        
        # Find observation positions
        sep = (inp[0] == tok.id_sep).nonzero(as_tuple=True)[0][0].item() + 1
        obs_positions = [sep + 2*t + 1 for t in range(K)]
        
        for i, t in enumerate(obs_positions):
            rep = h[:, t, :].cpu().numpy()  # [batch, d_model]
            ent = compute_entropy(posteriors[:, i, :])  # [batch]
            most_likely = posteriors[:, i, :].argmax(axis=1)  # [batch]
            
            reps.append(rep)
            entropies.append(ent)
            most_likely_states.append(most_likely)

reps = np.vstack(reps)
entropies = np.concatenate(entropies)
most_likely_states = np.concatenate(most_likely_states)

print(f'Total points: {len(entropies)}')
print(f'State distribution: {np.bincount(most_likely_states)}')

# PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(reps)
print(f'PC1 variance: {pca.explained_variance_ratio_[0]*100:.1f}%')
print(f'PC2 variance: {pca.explained_variance_ratio_[1]*100:.1f}%')

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: colored by most likely state
ax1 = axes[0]
state_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']  # 5 distinct colors
for state in range(5):
    mask = most_likely_states == state
    ax1.scatter(coords[mask, 0], coords[mask, 1], c=state_colors[state], 
                alpha=0.5, s=15, label=f'State {state+1}')
ax1.set_xlabel('PC1', fontsize=12)
ax1.set_ylabel('PC2', fontsize=12)
ax1.set_title('Mamba Final Layer: Colored by Most Likely HMM State', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=10)

# Right panel: colored by entropy
ax2 = axes[1]
vmin, vmax = np.percentile(entropies, 2), np.percentile(entropies, 98)
sc = ax2.scatter(coords[:, 0], coords[:, 1], c=entropies, cmap='RdYlBu_r',
                  vmin=vmin, vmax=vmax, alpha=0.5, s=15)
ax2.set_xlabel('PC1', fontsize=12)
ax2.set_ylabel('PC2', fontsize=12)
ax2.set_title('Mamba Final Layer: Colored by Posterior Entropy', fontsize=13, fontweight='bold')
cbar = plt.colorbar(sc, ax=ax2)
cbar.set_label('Entropy (bits)', fontsize=11)

# Add annotation
corr = np.corrcoef(coords[:, 0], entropies)[0, 1]
ax2.text(0.05, 0.95, f'|r| = {abs(corr):.2f}', transform=ax2.transAxes, fontsize=11,
         va='top', bbox=dict(facecolor='white', alpha=0.8))

plt.suptitle('Mamba Discovers the 5-Corner Geometry of HMM Belief Space\n(MAE = 0.031 bits)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('logs/mamba_5clusters.png', dpi=150, bbox_inches='tight')
print('Saved: logs/mamba_5clusters.png')
