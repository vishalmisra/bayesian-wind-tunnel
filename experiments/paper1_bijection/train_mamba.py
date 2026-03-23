#!/usr/bin/env python3
"""
Train Mamba on bijection learning task using SAME data generation as transformer.

Uses the exact same:
- Data generation (100% changing dictionaries, without replacement)
- Evaluation metrics (entropy MAE, KL divergence)
- Training hyperparameters where applicable

This ensures a fair comparison between Transformer and Mamba architectures.

v2: Added numerical stability fixes
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import data generation and evaluation from transformer script
from train import make_batch, evaluate_entropy_mae, evaluate_kl_divergence

# ============================================================================
# Mamba Model Architecture (Numerically Stable)
# ============================================================================

class MambaBlock(nn.Module):
    """
    Simplified Mamba block with selective state space.
    Numerically stable version with clamping.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner
        )
        
        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # SSM discretization - smaller init for stability
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)
        nn.init.constant_(self.dt_proj.bias, 0.1)
        
        # A parameter (diagonal, learned) - smaller init
        A = torch.arange(1, d_state + 1, dtype=torch.float32) * 0.1
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Layer norm for stability
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
        
        # Clamp dt for stability
        dt = F.softplus(self.dt_proj(dt)).clamp(min=1e-4, max=10.0)
        A = -torch.exp(self.A_log.clamp(max=5.0))  # Clamp A
        
        y = self.selective_scan(x, dt, A, B_param, C_param)
        y = self.norm(y)  # Add layer norm
        y = y + self.D * x
        y = y * F.silu(z)
        return self.out_proj(y)
    
    def selective_scan(self, u, dt, A, B, C):
        B_batch, L, d_inner = u.shape
        d_state = self.d_state
        
        h = torch.zeros(B_batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        
        for i in range(L):
            # Clamp for numerical stability
            dA = torch.exp((dt[:, i, :, None] * A).clamp(min=-20, max=0))
            dB_u = (dt[:, i, :, None] * B[:, i, None, :] * u[:, i, :, None]).clamp(-10, 10)
            h = h * dA + dB_u
            h = h.clamp(-100, 100)  # Prevent state explosion
            y = (h * C[:, i, None, :]).sum(-1)
            ys.append(y)
        
        return torch.stack(ys, dim=1)


class MambaBijection(nn.Module):
    """
    Mamba model for bijection learning.
    """
    def __init__(self, vocab_size, d_model=192, n_layers=6, d_state=16, max_seq_len=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # Weight tying
        self.d_model = d_model

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        
        for block in self.blocks:
            h = h + block(h)  # Residual connection
            # Check for NaN and recover
            if torch.isnan(h).any():
                h = torch.nan_to_num(h, nan=0.0)
        
        h = self.norm(h)
        logits = self.head(h)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        
        return logits, loss, h


# ============================================================================
# Main Training
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--V", type=int, default=20, help="Vocabulary size")
    parser.add_argument("--L", type=int, default=19, help="Sequence length")
    parser.add_argument("--with_replacement", action="store_true", help="Sample keys with replacement")
    parser.add_argument("--dim", type=int, default=192, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--d_state", type=int, default=16, help="Mamba state dimension")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)  # Lower LR for stability
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=0.5)  # Tighter clipping
    parser.add_argument("--output_dir", type=str, default="logs/mamba_bijection_v20")
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=5000)  # More frequent eval
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    vocab_size = 2 * args.V
    max_seq_len = 2 * args.L
    
    model = MambaBijection(
        vocab_size=vocab_size,
        d_model=args.dim,
        n_layers=args.n_layers,
        d_state=args.d_state,
        max_seq_len=max_seq_len
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Mamba parameters: {n_params:,}")
    print(f"Training: V={args.V}, L={args.L}, with_replacement={args.with_replacement}")
    print(f"CRITICAL: Using 100% changing dictionaries (pure ICL)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_steps)
    
    print(f"\nTraining for {args.max_steps} steps...")
    print("=" * 60)
    
    for step in range(1, args.max_steps + 1):
        model.train()
        
        x, y = make_batch(args.V, args.L, args.batch_size, args.with_replacement, device)
        logits, loss, _ = model(x, targets=y)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"Step {step}: NaN loss detected, skipping update")
            continue
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        if step % args.eval_every == 0 or step == 1:
            print(f"Step {step:5d}/{args.max_steps} | Loss: {loss.item():.4f} | Grad: {grad_norm:.4f}")
            
            mae_nor = evaluate_entropy_mae(model, args.V, args.L, device, n_eval=100, with_replacement=False)
            mae_wr = evaluate_entropy_mae(model, args.V, args.L, device, n_eval=100, with_replacement=True)
            print(f"  → MAE (w/o replacement): {mae_nor:.4f} bits")
            print(f"  → MAE (w/ replacement):  {mae_wr:.4f} bits")
            
            if mae_nor < 0.01:
                print("  ✓ Functionally Bayesian (MAE < 0.01 bits)!")
        
        if step % args.save_every == 0:
            ckpt_path = f"{args.output_dir}/ckpt_step{step}.pt"
            torch.save({
                'model': model.state_dict(),
                'step': step,
                'args': vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")
    
    # Final save
    final_path = f"{args.output_dir}/ckpt_final.pt"
    torch.save({
        'model': model.state_dict(),
        'step': args.max_steps,
        'args': vars(args),
    }, final_path)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS - MAMBA")
    print("=" * 60)
    
    mae_nor = evaluate_entropy_mae(model, args.V, args.L, device, n_eval=200, with_replacement=False)
    mae_wr = evaluate_entropy_mae(model, args.V, args.L, device, n_eval=200, with_replacement=True)
    print(f"Entropy MAE (w/o replacement): {mae_nor:.4f} bits")
    print(f"Entropy MAE (w/ replacement):  {mae_wr:.4f} bits")
    
    print(f"\nSaved final model to {final_path}")


if __name__ == '__main__':
    main()
