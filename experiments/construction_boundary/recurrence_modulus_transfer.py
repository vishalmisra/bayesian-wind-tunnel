"""
Modulus Transfer Experiment — Phase 2, Experiment 2
====================================================

Tests whether gradient-compiled modular arithmetic circuits transfer
across different moduli.

Protocol:
  1. Load a model trained on recurrences mod 17 (the standard π experiment)
  2. Expand vocab from 17 → p_new (e.g., 19)
  3. Fine-tune on recurrences mod p_new
  4. Compare convergence speed vs training from scratch on p_new

If the modular arithmetic circuit is abstract (implementing actual
modular operations), fine-tuning should converge much faster than
from-scratch training. If it's a mod-17 lookup table, no advantage.

Usage:
    # Fine-tune from p=17 checkpoint
    python recurrence_modulus_transfer.py --mode finetune \
        --p_new 19 --checkpoint results/recurrence/integer/seed_42/best_model.pt \
        --seeds 42 --device cuda:0

    # Train from scratch on p=19 (baseline)
    python recurrence_modulus_transfer.py --mode scratch \
        --p_new 19 --seeds 42 --device cuda:1

    # Train from scratch on p=17 (control — should match existing results)
    python recurrence_modulus_transfer.py --mode scratch \
        --p_new 17 --seeds 42 --device cuda:2
"""

import sys
import os
import math
import argparse
import json
import copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recurrence_bwt import (
    RecurrenceConfig,
    generate_recurrence_sequence,
    _predictive_entropy,
)

torch = None
nn = None
F = None


def _ensure_torch():
    global torch, nn, F
    if torch is None:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        torch = _torch
        nn = _nn
        F = _F


# ============================================================================
# Model (same architecture as recurrence_bwt.py)
# ============================================================================

def _build_model_class():
    _ensure_torch()

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, n_heads, dropout=0.1):
            super().__init__()
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.out = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scale = self.d_head ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn.masked_fill(
                    mask.unsqueeze(0).unsqueeze(0), float('-inf')
                )
            alpha = torch.softmax(attn, dim=-1)
            alpha = self.dropout(alpha)
            out = (alpha @ v).transpose(1, 2).reshape(B, T, C)
            return self.out(out), alpha

    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
            super().__init__()
            d_ff = d_ff or 4 * d_model
            self.attn = MultiHeadAttention(d_model, n_heads, dropout)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x, mask=None):
            h, alpha = self.attn(self.ln1(x), mask)
            x = x + h
            x = x + self.ff(self.ln2(x))
            return x, alpha

    class RecurrenceTransformer(nn.Module):
        def __init__(self, vocab_size, n_tokens, d_model=192, n_layers=6,
                     n_heads=6, d_ff=768, dropout=0.1):
            super().__init__()
            self.vocab_size = vocab_size
            self.n_tokens = n_tokens
            self.d_model = d_model

            self.token_embed = nn.Embedding(vocab_size + 1, d_model,
                                            padding_idx=vocab_size)
            self.pos_embed = nn.Embedding(512, d_model)

            self.layers = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])

            self.ln_final = nn.LayerNorm(d_model)
            self.output_proj = nn.Linear(d_model, n_tokens)

        def forward(self, tokens):
            B, T = tokens.shape
            mask = torch.triu(
                torch.ones(T, T, device=tokens.device), diagonal=1
            ).bool()
            x = self.token_embed(tokens)
            positions = torch.arange(
                T, device=tokens.device
            ).unsqueeze(0).expand(B, -1)
            x = x + self.pos_embed(positions)
            for layer in self.layers:
                x, _ = layer(x, mask)
            x = self.ln_final(x)
            logits = self.output_proj(x)
            return logits

    return RecurrenceTransformer


def expand_model(model_old, p_old, p_new, d_model):
    """
    Expand a trained model from p_old tokens to p_new tokens.
    
    - token_embed: copy old rows, init new rows randomly
    - output_proj: copy old weights, init new rows randomly
    - All other weights: copy directly
    """
    _ensure_torch()
    RecurrenceTransformer = _build_model_class()

    model_new = RecurrenceTransformer(
        vocab_size=p_new,
        n_tokens=p_new,
        d_model=d_model,
        n_layers=len(model_old.layers),
        n_heads=model_old.layers[0].attn.n_heads,
        d_ff=model_old.layers[0].ff[0].out_features,
    )

    # Copy all shared weights
    old_state = model_old.state_dict()
    new_state = model_new.state_dict()

    for key in new_state:
        if key in old_state:
            old_shape = old_state[key].shape
            new_shape = new_state[key].shape
            if old_shape == new_shape:
                new_state[key] = old_state[key]
            elif key == 'token_embed.weight':
                # Copy old embeddings, keep new ones random
                n_copy = min(old_shape[0], new_shape[0])
                new_state[key][:n_copy] = old_state[key][:n_copy]
                print(f"  token_embed: copied {n_copy}/{new_shape[0]} rows, "
                      f"{new_shape[0] - n_copy} new")
            elif key == 'output_proj.weight':
                n_copy = min(old_shape[0], new_shape[0])
                new_state[key][:n_copy] = old_state[key][:n_copy]
                print(f"  output_proj.weight: copied {n_copy}/{new_shape[0]} rows")
            elif key == 'output_proj.bias':
                n_copy = min(old_shape[0], new_shape[0])
                new_state[key][:n_copy] = old_state[key][:n_copy]
                print(f"  output_proj.bias: copied {n_copy}/{new_shape[0]}")
            else:
                print(f"  WARNING: shape mismatch for {key}: "
                      f"{old_shape} -> {new_shape}, using random init")

    model_new.load_state_dict(new_state)
    return model_new


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model, p, pi, seq_len, n_eval=2000, device='cpu'):
    """Evaluate model on recurrence task at given prime."""
    _ensure_torch()
    model.eval()

    per_position = {}
    all_mae = []

    cfg = RecurrenceConfig(p=p, pi=pi, seq_len=seq_len, opaque=False)

    with torch.no_grad():
        for ep in range(n_eval):
            tokens, gt, metadata = generate_recurrence_sequence(cfg)
            header_len = metadata['header_len']
            n_tok = metadata['n_tokens']

            tokens_tensor = torch.tensor(
                tokens, dtype=torch.long
            ).unsqueeze(0).to(device)
            logits = model(tokens_tensor)
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()

            for gt_entry in gt:
                t = gt_entry['t']
                model_pos = header_len + t - 1
                if t == 0 or model_pos < 0 or model_pos >= len(probs):
                    continue

                p_model = probs[model_pos]
                n_tok_eval = min(n_tok, p_model.shape[0])

                H_model = -sum(
                    p_model[i] * math.log2(p_model[i])
                    for i in range(n_tok_eval) if p_model[i] > 1e-10
                )
                H_bayes = gt_entry['entropy']
                mae = abs(H_model - H_bayes)
                all_mae.append(mae)

                if t not in per_position:
                    per_position[t] = {
                        'H_model': [], 'H_bayes': [], 'mae': []
                    }
                per_position[t]['H_model'].append(H_model)
                per_position[t]['H_bayes'].append(H_bayes)
                per_position[t]['mae'].append(mae)

    metrics = {
        'mae_bits': float(np.mean(all_mae)) if all_mae else 0.0,
        'mae_std': float(np.std(all_mae)) if all_mae else 0.0,
    }

    per_pos_summary = {}
    for t in sorted(per_position.keys()):
        per_pos_summary[t] = {
            'H_model_mean': float(np.mean(per_position[t]['H_model'])),
            'H_bayes_mean': float(np.mean(per_position[t]['H_bayes'])),
            'mae_mean': float(np.mean(per_position[t]['mae'])),
            'count': len(per_position[t]['mae']),
        }

    return metrics, per_pos_summary


# ============================================================================
# Training
# ============================================================================

def train(args):
    _ensure_torch()
    RecurrenceTransformer = _build_model_class()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    p = args.p_new

    if args.mode == 'finetune' and args.checkpoint:
        # Load old model (p=17)
        p_old = args.p_old
        model_old = RecurrenceTransformer(
            vocab_size=p_old, n_tokens=p_old,
            d_model=args.d_model, n_layers=args.n_layers,
            n_heads=args.n_heads, d_ff=args.d_ff,
        )
        print(f"Loading checkpoint from {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location='cpu',
                           weights_only=True)
        model_old.load_state_dict(state)

        if p == p_old:
            model = model_old
            print(f"Same modulus ({p}), using model directly")
        else:
            print(f"Expanding model from p={p_old} to p={p}")
            model = expand_model(model_old, p_old, p, args.d_model)
        model = model.to(device)
    else:
        # Train from scratch
        model = RecurrenceTransformer(
            vocab_size=p, n_tokens=p,
            d_model=args.d_model, n_layers=args.n_layers,
            n_heads=args.n_heads, d_ff=args.d_ff,
            dropout=args.dropout,
        ).to(device)

    param_count = sum(pr.numel() for pr in model.parameters())
    print(f"Model: {param_count:,} parameters on {device}")
    print(f"Task: recurrence mod {p} ({args.mode})")
    print(f"  Sequence length: {args.seq_len}")

    # Evaluate before training (for finetune: how good is the pretrained model?)
    if args.mode == 'finetune':
        print(f"\n  Pre-finetune eval on p={p}:")
        m, pp = evaluate(model, p, args.pi, args.seq_len,
                         n_eval=1000, device=device)
        print(f"    MAE={m['mae_bits']:.4f} bits")
        for t in sorted(pp.keys()):
            if t <= 4 or t == max(pp.keys()):
                print(f"      t={t:2d}: MAE={pp[t]['mae_mean']:.4f}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_steps
    )

    PAD = p  # padding token = vocab_size
    os.makedirs(args.output_dir, exist_ok=True)
    best_mae = float('inf')
    losses = []
    mae_trajectory = []

    for step in range(1, args.n_steps + 1):
        model.train()
        cfg = RecurrenceConfig(
            p=p, pi=args.pi, seq_len=args.seq_len, opaque=False
        )

        batch_tokens = []
        max_len = 0
        for _ in range(args.batch_size):
            tokens, _, metadata = generate_recurrence_sequence(cfg)
            batch_tokens.append(tokens)
            max_len = max(max_len, len(tokens))

        padded = []
        for tokens in batch_tokens:
            padded.append(tokens + [PAD] * (max_len - len(tokens)))
        x = torch.tensor(padded, dtype=torch.long).to(device)

        logits = model(x)

        loss = torch.tensor(0.0, device=device)
        count = 0
        for b_idx in range(args.batch_size):
            seq_start = 0  # no header in integer mode
            seq_end = len(batch_tokens[b_idx])
            for t in range(seq_start, seq_end - 1):
                target = x[b_idx, t + 1]
                if target < p:
                    loss = loss + F.cross_entropy(
                        logits[b_idx, t, :p], target
                    )
                    count += 1

        if count > 0:
            loss = loss / count

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % args.log_every == 0:
            recent_loss = np.mean(losses[-args.log_every:])
            print(f"  Step {step}/{args.n_steps}: loss={recent_loss:.4f}")

        if step % args.eval_every == 0:
            m, pp = evaluate(
                model, p, args.pi, args.seq_len,
                n_eval=1000, device=device
            )
            print(f"  Eval: MAE={m['mae_bits']:.4f} bits")
            mae_trajectory.append({
                'step': step, 'mae': m['mae_bits']
            })

            for t in sorted(pp.keys()):
                print(f"    t={t:2d}: H_model={pp[t]['H_model_mean']:.4f}, "
                      f"H_bayes={pp[t]['H_bayes_mean']:.4f}, "
                      f"MAE={pp[t]['mae_mean']:.4f}")

            if m['mae_bits'] < best_mae:
                best_mae = m['mae_bits']
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, 'best_model.pt'))
                print(f"    New best model saved (MAE={best_mae:.6f})")

    # Final evaluation
    m, pp = evaluate(model, p, args.pi, args.seq_len,
                     n_eval=5000, device=device)

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS (modulus transfer, {args.mode}, p={p})")
    print(f"{'='*70}")
    print(f"  Entropy MAE: {m['mae_bits']:.6f} ± {m['mae_std']:.6f} bits")
    for t in sorted(pp.keys()):
        print(f"    t={t:2d}: H_model={pp[t]['H_model_mean']:.4f}, "
              f"H_bayes={pp[t]['H_bayes_mean']:.4f}, "
              f"MAE={pp[t]['mae_mean']:.4f}")

    results = {
        'mode': args.mode,
        'p_new': p,
        'p_old': args.p_old if args.mode == 'finetune' else None,
        'metrics': m,
        'per_position': {str(k): v for k, v in pp.items()},
        'mae_trajectory': mae_trajectory,
        'final_loss': float(np.mean(losses[-1000:])),
    }

    results_path = os.path.join(args.output_dir, 'modulus_transfer_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Modulus transfer experiment (Phase 2)')
    parser.add_argument('--mode', choices=['finetune', 'scratch'],
                        default='finetune')
    parser.add_argument('--p_new', type=int, default=19,
                        help='Target prime for evaluation')
    parser.add_argument('--p_old', type=int, default=17,
                        help='Source prime (for finetune mode)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--pi', type=float, default=0.5)
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_steps', type=int, default=150000)
    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--log_every', type=int, default=500)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--output_dir', type=str,
                        default='results/modulus_transfer')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"MODULUS TRANSFER ({args.mode.upper()}, "
              f"p={args.p_new}), SEED {seed}")
        print(f"{'='*70}")

        np.random.seed(seed)
        _ensure_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        seed_dir = os.path.join(
            args.output_dir,
            f'{args.mode}_p{args.p_new}_seed{seed}'
        )
        args_copy = argparse.Namespace(**vars(args))
        args_copy.output_dir = seed_dir

        train(args_copy)


if __name__ == '__main__':
    main()
