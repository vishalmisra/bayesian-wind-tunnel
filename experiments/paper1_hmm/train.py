#!/usr/bin/env python3
"""
HMM Wind Tunnel Training Script

Trains GPT-Mini to predict Bayesian posteriors for Hidden Markov Models.
This is a "wind tunnel" for understanding how transformers perform inference.

Key features:
- Ground-truth posteriors computed using exact discretized parameters
- Cross-entropy loss between predicted and true posteriors
- Supports multi-GPU training with DDP

Usage:
    python experiments/paper1_hmm/train.py --n-samples 10000 --epochs 50
    
For multi-GPU:
    torchrun --nproc_per_node=8 experiments/paper1_hmm/train.py --n-samples 100000
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hmm import HMMConfig, HMMTokenizer, HMMDataset, collate_hmm_batch
from src.models.gpt_mini import GPTMini, GPTMiniConfig


def compute_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    obs_positions: torch.Tensor
) -> torch.Tensor:
    """
    Compute cross-entropy loss at observation positions.
    
    Args:
        logits: (B, T, num_states)
        targets: (B, K, num_states) ground-truth posteriors
        obs_positions: (B, K) positions of observation tokens
        
    Returns:
        loss: scalar tensor
    """
    B, T, num_states = logits.shape
    K = targets.shape[1]
    
    # Gather logits at observation positions
    # obs_positions: (B, K)
    batch_idx = torch.arange(B, device=logits.device).unsqueeze(1).expand(-1, K)
    pred_logits = logits[batch_idx, obs_positions]  # (B, K, num_states)
    
    # Softmax to get predictions
    pred_probs = torch.softmax(pred_logits, dim=-1)
    
    # Cross-entropy loss: -sum(target * log(pred))
    eps = 1e-10
    loss = -torch.sum(targets * torch.log(pred_probs + eps), dim=-1)  # (B, K)
    
    return loss.mean()


def get_observation_positions_batch(
    input_ids: torch.Tensor, 
    tokenizer: HMMTokenizer, 
    K: int
) -> torch.Tensor:
    """Get observation token positions for a batch."""
    B = input_ids.shape[0]
    device = input_ids.device
    
    positions = []
    for b in range(B):
        ids = input_ids[b].cpu().numpy()
        # Find separator position
        sep_pos = None
        for i, tok in enumerate(ids):
            if tok == tokenizer.id_sep:
                sep_pos = i
                break
        
        if sep_pos is None:
            sep_pos = 52  # fallback
            
        # Observation positions: sep_pos + 1 + 2*t + 1 for t=0..K-1
        header_len = sep_pos + 1
        obs_pos = [header_len + 2 * t + 1 for t in range(K)]
        positions.append(obs_pos)
    
    return torch.tensor(positions, dtype=torch.long, device=device)


def main():
    parser = argparse.ArgumentParser(description="Train GPT-Mini on HMM inference")
    parser.add_argument("--n-samples", type=int, default=10000, help="Training samples")
    parser.add_argument("--n-val-samples", type=int, default=1000, help="Validation samples")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seq-length", type=int, default=15, help="HMM sequence length (K)")
    parser.add_argument("--n-layers", type=int, default=9, help="Transformer layers")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--output-dir", type=str, default="checkpoints/hmm", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Setup device and distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    
    if is_distributed:
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_main = local_rank == 0
    
    # Set seeds
    torch.manual_seed(args.seed + local_rank)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data
    hmm_cfg = HMMConfig(sequence_length=args.seq_length, seed=args.seed)
    tokenizer = HMMTokenizer()
    
    if is_main:
        print(f"Generating {args.n_samples} training samples...")
    train_dataset = HMMDataset(args.n_samples, hmm_cfg, tokenizer, seed=args.seed)
    val_dataset = HMMDataset(args.n_val_samples, hmm_cfg, tokenizer, seed=args.seed + 1000)
    
    # Create data loaders
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_hmm_batch,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_hmm_batch,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model_cfg = GPTMiniConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        num_states=5,
        dropout=0.1
    )
    model = GPTMini(model_cfg).to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )
    
    # Training loop
    best_val_loss = float('inf')
    K = args.seq_length
    
    for epoch in range(args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main)
        for input_ids, targets in pbar:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            # Get observation positions
            obs_positions = get_observation_positions_batch(input_ids, tokenizer, K)
            
            # Forward pass
            logits, _ = model(input_ids)
            loss = compute_loss(logits, targets, obs_positions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        train_loss /= n_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        
        with torch.no_grad():
            for input_ids, targets in val_loader:
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                
                obs_positions = get_observation_positions_batch(input_ids, tokenizer, K)
                logits, _ = model(input_ids)
                loss = compute_loss(logits, targets, obs_positions)
                
                val_loss += loss.item()
                n_val += 1
        
        val_loss /= n_val
        
        if is_main:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    "model": model.module.state_dict() if is_distributed else model.state_dict(),
                    "config": {
                        "vocab_size": model_cfg.vocab_size,
                        "d_model": model_cfg.d_model,
                        "n_layers": model_cfg.n_layers,
                        "n_heads": model_cfg.n_heads,
                        "num_states": model_cfg.num_states,
                    },
                    "epoch": epoch,
                    "val_loss": val_loss,
                }
                torch.save(ckpt, output_dir / "best_model.pt")
                print(f"  Saved best model (val_loss={val_loss:.4f})")
    
    if is_distributed:
        dist.destroy_process_group()
    
    if is_main:
        print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
