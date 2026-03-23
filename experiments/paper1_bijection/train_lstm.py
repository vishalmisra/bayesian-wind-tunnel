#!/usr/bin/env python3
"""
Train LSTM on bijection task for architecture comparison.
"""
import argparse
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train import make_batch, evaluate_entropy_mae

class LSTMBijection(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_layers=6, max_seq_len=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.d_model = d_model

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        h, _ = self.lstm(h)
        h = self.norm(h)
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss, h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--V", type=int, default=20)
    parser.add_argument("--L", type=int, default=19)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default="logs/lstm_bijection_v20")
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    model = LSTMBijection(vocab_size=2*args.V, d_model=args.dim, n_layers=args.n_layers, max_seq_len=2*args.L).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LSTM parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    
    print(f"\nTraining for {args.max_steps} steps...")
    print("=" * 60)
    
    for step in range(1, args.max_steps + 1):
        model.train()
        x, y = make_batch(args.V, args.L, args.batch_size, False, device)
        logits, loss, _ = model(x, targets=y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % args.eval_every == 0 or step == 1:
            mae = evaluate_entropy_mae(model, args.V, args.L, device, n_eval=100, with_replacement=False)
            print(f"Step {step:5d}: loss={loss.item():.4f}, MAE={mae:.4f} bits")
    
    # Final
    mae = evaluate_entropy_mae(model, args.V, args.L, device, n_eval=200, with_replacement=False)
    print(f"\nFINAL MAE: {mae:.4f} bits")
    
    torch.save({'model': model.state_dict(), 'args': vars(args)}, f"{args.output_dir}/ckpt_final.pt")

if __name__ == '__main__':
    main()
