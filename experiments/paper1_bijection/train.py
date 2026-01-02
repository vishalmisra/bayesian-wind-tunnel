#!/usr/bin/env python3
"""
Train TinyGPT on bijection learning task.

This script trains a small transformer to learn random bijections (permutations)
from in-context examples. The key experimental setup:
- V = 20 (vocabulary size)
- L = 19 (context length)
- Sampling without replacement (each key appears at most once)
- Query from context (query key is one of the context keys)

Usage:
    # Single GPU
    python train.py --output_dir logs/bijection_v20

    # Multi-GPU (8 GPUs)
    torchrun --standalone --nproc_per_node=8 train.py --output_dir logs/bijection_v20_ddp

Reference: Paper I, Section 3.1 (Bijection Wind Tunnel)
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import TinyGPT
from src.data import BijectionDataset


def get_args():
    parser = argparse.ArgumentParser(description="Train bijection learning model")
    
    # Data
    parser.add_argument("--V", type=int, default=20, help="Vocabulary size")
    parser.add_argument("--L", type=int, default=19, help="Context length")
    parser.add_argument("--n_train", type=int, default=1_000_000, help="Training samples")
    parser.add_argument("--n_val", type=int, default=10_000, help="Validation samples")
    parser.add_argument("--with_replacement", action="store_true", help="Sample keys with replacement")
    
    # Model
    parser.add_argument("--dim", type=int, default=192, help="Embedding dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=6, help="Number of attention heads")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=100_000, help="Maximum training steps")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="logs/bijection_v20", help="Output directory")
    parser.add_argument("--save_every", type=int, default=10_000, help="Save checkpoint every N steps")
    parser.add_argument("--eval_every", type=int, default=1_000, help="Evaluate every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    """Cosine schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            total_loss += loss.item() * x.size(0)
            
            # Accuracy on query position
            preds = logits[:, -1, :].argmax(dim=-1)
            targets = y[:, -1]
            total_correct += (preds == targets).sum().item()
            total_count += x.size(0)
    
    model.train()
    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
    }


def main():
    args = get_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Data
    train_dataset = BijectionDataset(
        V=args.V, L=args.L, n_samples=args.n_train,
        with_replacement=args.with_replacement,
        query_from_context=True,
        predict_all_values=True,
        seed=args.seed,
    )
    val_dataset = BijectionDataset(
        V=args.V, L=args.L, n_samples=args.n_val,
        with_replacement=args.with_replacement,
        query_from_context=True,
        predict_all_values=True,
        seed=args.seed + 1,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Model
    model = TinyGPT(
        vocab_size=args.V,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=2 * args.L + 1,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, args.max_steps)
    
    # Training loop
    step = 0
    train_iter = iter(train_loader)
    
    print(f"Training for {args.max_steps} steps...")
    
    while step < args.max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        step += 1
        
        # Logging
        if step % 100 == 0:
            print(f"Step {step}/{args.max_steps} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Evaluation
        if step % args.eval_every == 0:
            metrics = evaluate(model, val_loader, device)
            print(f"  Val Loss: {metrics['loss']:.4f} | Val Acc: {metrics['accuracy']:.4f}")
        
        # Save checkpoint
        if step % args.save_every == 0:
            ckpt_path = output_dir / f"ckpt_step{step}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "config": vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")
    
    # Final save
    final_path = output_dir / "ckpt_final.pt"
    torch.save({
        "model": model.state_dict(),
        "step": step,
        "config": vars(args),
    }, final_path)
    print(f"Training complete! Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
