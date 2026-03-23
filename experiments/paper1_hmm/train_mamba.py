#!/usr/bin/env python3
"""
Train Mamba on HMM task for architecture comparison.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hmm import HMMConfig, HMMTokenizer, HMMDataset, collate_hmm_batch


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
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)
        nn.init.constant_(self.dt_proj.bias, 0.1)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32) * 0.1
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_inner)
    
    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)
        
        x_dbl = self.x_proj(x)
        dt, B_param, C_param = x_dbl.split([1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)).clamp(min=1e-4, max=10.0)
        A = -torch.exp(self.A_log.clamp(max=5.0))
        
        y = self.selective_scan(x, dt, A, B_param, C_param)
        y = self.norm(y)
        y = y + self.D * x
        y = y * F.silu(z)
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

    def forward(self, x):
        h = self.tok_emb(x)
        for block in self.blocks:
            h = h + block(h)
            if torch.isnan(h).any():
                h = torch.nan_to_num(h, nan=0.0)
        h = self.norm(h)
        return self.head(h)


def get_obs_positions(input_ids, tokenizer, K):
    B = input_ids.shape[0]
    device = input_ids.device
    positions = []
    for b in range(B):
        ids = input_ids[b].cpu().numpy()
        sep_pos = 52
        for i, tok in enumerate(ids):
            if tok == tokenizer.id_sep:
                sep_pos = i
                break
        header_len = sep_pos + 1
        obs_pos = [header_len + 2 * t + 1 for t in range(K)]
        positions.append(obs_pos)
    return torch.tensor(positions, dtype=torch.long, device=device)


def compute_loss(logits, targets, obs_positions):
    B, T, num_states = logits.shape
    K = targets.shape[1]
    batch_idx = torch.arange(B, device=logits.device).unsqueeze(1).expand(-1, K)
    pred_logits = logits[batch_idx, obs_positions]
    pred_probs = torch.softmax(pred_logits, dim=-1)
    eps = 1e-10
    loss = -torch.sum(targets * torch.log(pred_probs + eps), dim=-1)
    return loss.mean()


def compute_entropy_mae(pred_probs, targets):
    eps = 1e-10
    pred_ent = -torch.sum(pred_probs * torch.log2(pred_probs + eps), dim=-1)
    tgt_ent = -torch.sum(targets * torch.log2(targets + eps), dim=-1)
    return torch.abs(pred_ent - tgt_ent).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--n-val-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-length", type=int, default=20)
    parser.add_argument("--n-layers", type=int, default=9)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="logs/mamba_hmm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    hmm_cfg = HMMConfig(sequence_length=args.seq_length, seed=args.seed)
    tokenizer = HMMTokenizer()

    print(f"Generating {args.n_samples} training samples...")
    train_dataset = HMMDataset(args.n_samples, hmm_cfg, tokenizer, seed=args.seed)
    val_dataset = HMMDataset(args.n_val_samples, hmm_cfg, tokenizer, seed=args.seed + 1000)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_hmm_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_hmm_batch)

    model = MambaHMM(
        vocab_size=tokenizer.vocab_size,
        num_states=hmm_cfg.n_states,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Mamba parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 60)

    best_mae = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for input_ids, targets in train_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            obs_positions = get_obs_positions(input_ids, tokenizer, args.seq_length)

            logits = model(input_ids)
            loss = compute_loss(logits, targets, obs_positions)
            
            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_maes = []
        with torch.no_grad():
            for input_ids, targets in val_loader:
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                obs_positions = get_obs_positions(input_ids, tokenizer, args.seq_length)

                logits = model(input_ids)
                B, T, num_states = logits.shape
                K = targets.shape[1]
                batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)
                pred_logits = logits[batch_idx, obs_positions]
                pred_probs = torch.softmax(pred_logits, dim=-1)
                mae = compute_entropy_mae(pred_probs, targets)
                val_maes.append(mae)

        avg_mae = np.mean(val_maes)
        print(f"Epoch {epoch:3d}: loss={total_loss/len(train_loader):.4f}, val_MAE={avg_mae:.6f} bits")

        if avg_mae < best_mae:
            best_mae = avg_mae
            torch.save({'model': model.state_dict(), 'args': vars(args), 'mae': best_mae}, 
                       f"{args.output_dir}/ckpt_best.pt")

    print(f"\nBest MAE: {best_mae:.6f} bits")
    torch.save({'model': model.state_dict(), 'args': vars(args), 'mae': avg_mae}, 
               f"{args.output_dir}/ckpt_final.pt")


if __name__ == '__main__':
    main()
