"""
Recurrence Extrapolation Experiment — Phase 2, Experiment 1
============================================================

Tests whether gradient-compiled mechanism identification generalizes
beyond the training horizon.

Two sub-experiments:

(A) Length extrapolation: sinusoidal PE, train on length L_train,
    evaluate on length L_test >> L_train. Does the compiled program
    circuit generalize to unseen positions?

(B) Loss horizon restriction: standard learned PE, train on full
    length 16 but compute loss only at positions 1..K. Evaluate at
    all positions 1..15. Does the program circuit produce correct
    predictions at positions it was never directly rewarded for?

Both test whether the model builds a general-purpose "run the
recurrence" circuit vs position-specific statistical patterns.

Usage:
    # Sub-experiment A: length extrapolation
    python recurrence_extrapolation.py --mode extrapolate \
        --train_seq_len 8 --eval_seq_lens 8 16 32 50 \
        --sinusoidal_pe --seeds 42 --device cuda:0

    # Sub-experiment B: loss horizon restriction
    python recurrence_extrapolation.py --mode horizon \
        --loss_horizon 5 --seeds 42 --device cuda:0

    # Both conditions (integer vs opaque)
    python recurrence_extrapolation.py --mode extrapolate --opaque \
        --train_seq_len 8 --eval_seq_lens 8 16 32 50 \
        --sinusoidal_pe --seeds 42 --device cuda:1
"""

import sys
import os
import math
import argparse
import json
import numpy as np

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recurrence_bwt import (
    RecurrenceConfig,
    generate_recurrence_sequence,
    bayesian_predictive_recurrence,
    class_posterior_recurrence,
    count_consistent_recurrences,
    _predictive_entropy,
)

# Lazy torch imports
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
# Model with sinusoidal PE option
# ============================================================================

def _build_model_class():
    _ensure_torch()

    class SinusoidalPE(nn.Module):
        """Fixed sinusoidal positional encoding — generalizes to unseen lengths."""
        def __init__(self, d_model, max_len=2048):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float()
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, positions):
            # positions: (B, T) long tensor
            return self.pe[0, positions]  # (B, T, d_model) via advanced indexing

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, n_heads, dropout=0.1):
            super().__init__()
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.out_proj = nn.Linear(d_model, d_model)
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
                attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            alpha = torch.softmax(attn, dim=-1)
            alpha = self.dropout(alpha)
            out = (alpha @ v).transpose(1, 2).reshape(B, T, C)
            return self.out_proj(out), alpha

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

    class RecurrenceTransformerExtrap(nn.Module):
        """Transformer with optional sinusoidal PE for extrapolation."""

        def __init__(self, vocab_size, n_tokens, d_model=192, n_layers=6,
                     n_heads=6, d_ff=768, dropout=0.1, sinusoidal_pe=False):
            super().__init__()
            self.vocab_size = vocab_size
            self.n_tokens = n_tokens
            self.d_model = d_model
            self.sinusoidal_pe = sinusoidal_pe

            self.token_embed = nn.Embedding(vocab_size + 1, d_model,
                                            padding_idx=vocab_size)
            if sinusoidal_pe:
                self.pos_embed = SinusoidalPE(d_model)
            else:
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
            positions = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, -1)
            if self.sinusoidal_pe:
                x = x + self.pos_embed(positions)
            else:
                x = x + self.pos_embed(positions)

            for layer in self.layers:
                x, _ = layer(x, mask)

            x = self.ln_final(x)
            logits = self.output_proj(x)
            return logits

    return RecurrenceTransformerExtrap


# ============================================================================
# Evaluation at arbitrary length
# ============================================================================

def evaluate_at_length(model, p, pi, seq_len, n_eval=2000, device='cpu',
                       opaque=False):
    """Evaluate model on sequences of given length."""
    _ensure_torch()
    model.eval()

    per_position = {}
    all_mae = []

    cfg = RecurrenceConfig(p=p, pi=pi, seq_len=seq_len, opaque=opaque)

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
    RecurrenceTransformerExtrap = _build_model_class()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    p = args.p

    if args.opaque:
        vocab_size = 2 * p + 2
        n_tokens = p
    else:
        vocab_size = p
        n_tokens = p

    use_sinusoidal = args.sinusoidal_pe or args.mode == 'extrapolate'

    model = RecurrenceTransformerExtrap(
        vocab_size=vocab_size,
        n_tokens=n_tokens,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        sinusoidal_pe=use_sinusoidal,
    ).to(device)

    param_count = sum(pr.numel() for pr in model.parameters())
    mode_str = "OPAQUE" if args.opaque else "INTEGER"
    pe_str = "sinusoidal" if use_sinusoidal else "learned"
    print(f"Model: {param_count:,} parameters on {device}")
    print(f"Task: recurrence extrapolation ({mode_str}, {pe_str} PE)")
    print(f"  Mode: {args.mode}")
    print(f"  Train seq_len: {args.train_seq_len}")
    if args.mode == 'horizon':
        print(f"  Loss horizon: positions 1-{args.loss_horizon} "
              f"(of {args.train_seq_len})")
    if args.mode == 'extrapolate':
        print(f"  Eval seq_lens: {args.eval_seq_lens}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_steps
    )

    PAD = vocab_size
    os.makedirs(args.output_dir, exist_ok=True)
    best_mae = float('inf')
    losses = []

    for step in range(1, args.n_steps + 1):
        model.train()
        cfg = RecurrenceConfig(
            p=p, pi=args.pi, seq_len=args.train_seq_len, opaque=args.opaque
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

        # Loss computation with optional horizon restriction
        loss = torch.tensor(0.0, device=device)
        count = 0
        for b_idx in range(args.batch_size):
            header_len = len(batch_tokens[b_idx]) - args.train_seq_len
            seq_start = header_len

            # Determine loss range
            if args.mode == 'horizon' and args.loss_horizon is not None:
                # Only compute loss at positions 1..loss_horizon
                # (predicting tokens at positions 1..loss_horizon)
                seq_end_loss = min(
                    seq_start + args.loss_horizon,
                    len(batch_tokens[b_idx]) - 1
                )
            else:
                seq_end_loss = len(batch_tokens[b_idx]) - 1

            for t in range(seq_start, seq_end_loss):
                target = x[b_idx, t + 1]
                if target >= vocab_size:
                    continue
                if args.opaque:
                    target_shifted = target - p
                    if 0 <= target_shifted < n_tokens:
                        loss = loss + F.cross_entropy(
                            logits[b_idx, t, :n_tokens], target_shifted
                        )
                        count += 1
                else:
                    if target < n_tokens:
                        loss = loss + F.cross_entropy(
                            logits[b_idx, t, :n_tokens], target
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
            # Evaluate at training length
            metrics, per_pos = evaluate_at_length(
                model, p, args.pi, args.train_seq_len,
                n_eval=1000, device=device, opaque=args.opaque
            )
            print(f"  Eval (len={args.train_seq_len}): "
                  f"MAE={metrics['mae_bits']:.4f} bits")
            for t in sorted(per_pos.keys()):
                pp = per_pos[t]
                print(f"    t={t:2d}: H_model={pp['H_model_mean']:.4f}, "
                      f"H_bayes={pp['H_bayes_mean']:.4f}, "
                      f"MAE={pp['mae_mean']:.4f}")

            if metrics['mae_bits'] < best_mae:
                best_mae = metrics['mae_bits']
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, 'best_model.pt'))
                print(f"    New best model saved (MAE={best_mae:.6f})")

            # For extrapolation mode: also eval at other lengths
            if args.mode == 'extrapolate':
                for eval_len in args.eval_seq_lens:
                    if eval_len == args.train_seq_len:
                        continue
                    m, pp = evaluate_at_length(
                        model, p, args.pi, eval_len,
                        n_eval=500, device=device, opaque=args.opaque
                    )
                    print(f"  Eval (len={eval_len}): "
                          f"MAE={m['mae_bits']:.4f} bits")
                    # Show a few key positions
                    for t in sorted(pp.keys()):
                        if t <= 3 or t == eval_len - 1 or t % 10 == 0:
                            print(f"    t={t:2d}: MAE={pp[t]['mae_mean']:.4f}")

            # For horizon mode: show per-position to see trained vs untrained
            if args.mode == 'horizon':
                print(f"  --- Positions 1-{args.loss_horizon}: "
                      f"TRAINED (loss computed) ---")
                trained_maes = [
                    per_pos[t]['mae_mean']
                    for t in per_pos if 1 <= t <= args.loss_horizon
                ]
                untrained_maes = [
                    per_pos[t]['mae_mean']
                    for t in per_pos if t > args.loss_horizon
                ]
                if trained_maes:
                    print(f"    Mean MAE (trained positions): "
                          f"{np.mean(trained_maes):.4f} bits")
                if untrained_maes:
                    print(f"    Mean MAE (untrained positions "
                          f"{args.loss_horizon+1}-{args.train_seq_len-1}): "
                          f"{np.mean(untrained_maes):.4f} bits")

    # ====================================================================
    # Final comprehensive evaluation
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION")
    print(f"{'='*70}")

    results = {'mode': args.mode, 'opaque': args.opaque}

    if args.mode == 'extrapolate':
        for eval_len in args.eval_seq_lens:
            m, pp = evaluate_at_length(
                model, p, args.pi, eval_len,
                n_eval=2000, device=device, opaque=args.opaque
            )
            print(f"\n  Length {eval_len}: MAE={m['mae_bits']:.4f} "
                  f"± {m['mae_std']:.4f} bits")
            for t in sorted(pp.keys()):
                print(f"    t={t:2d}: H_model={pp[t]['H_model_mean']:.4f}, "
                      f"H_bayes={pp[t]['H_bayes_mean']:.4f}, "
                      f"MAE={pp[t]['mae_mean']:.4f}")
            results[f'len_{eval_len}'] = {
                'metrics': m,
                'per_position': {str(k): v for k, v in pp.items()},
            }

    elif args.mode == 'horizon':
        m, pp = evaluate_at_length(
            model, p, args.pi, args.train_seq_len,
            n_eval=2000, device=device, opaque=args.opaque
        )
        print(f"\n  Full evaluation (len={args.train_seq_len}): "
              f"MAE={m['mae_bits']:.4f} ± {m['mae_std']:.4f} bits")
        print(f"\n  Loss horizon: 1-{args.loss_horizon}")
        for t in sorted(pp.keys()):
            marker = " [TRAINED]" if t <= args.loss_horizon else " [UNTRAINED]"
            print(f"    t={t:2d}: H_model={pp[t]['H_model_mean']:.4f}, "
                  f"H_bayes={pp[t]['H_bayes_mean']:.4f}, "
                  f"MAE={pp[t]['mae_mean']:.4f}{marker}")

        trained_maes = [
            pp[t]['mae_mean'] for t in pp if 1 <= t <= args.loss_horizon
        ]
        untrained_maes = [
            pp[t]['mae_mean'] for t in pp if t > args.loss_horizon
        ]
        print(f"\n  Mean MAE (trained, 1-{args.loss_horizon}): "
              f"{np.mean(trained_maes):.4f} bits")
        print(f"  Mean MAE (untrained, {args.loss_horizon+1}-"
              f"{args.train_seq_len-1}): {np.mean(untrained_maes):.4f} bits")
        print(f"  Ratio (untrained/trained): "
              f"{np.mean(untrained_maes)/np.mean(trained_maes):.2f}x")

        results['metrics'] = m
        results['per_position'] = {str(k): v for k, v in pp.items()}
        results['trained_mae'] = float(np.mean(trained_maes))
        results['untrained_mae'] = float(np.mean(untrained_maes))

    # Save results
    results_path = os.path.join(args.output_dir, 'extrapolation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {results_path}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Recurrence extrapolation experiment (Phase 2)')
    parser.add_argument('--mode', choices=['extrapolate', 'horizon'],
                        default='extrapolate',
                        help='extrapolate: test length generalization; '
                             'horizon: test loss-horizon generalization')
    parser.add_argument('--p', type=int, default=17)
    parser.add_argument('--pi', type=float, default=0.5)
    parser.add_argument('--train_seq_len', type=int, default=8,
                        help='Sequence length for training')
    parser.add_argument('--eval_seq_lens', type=int, nargs='+',
                        default=[8, 16, 32, 50],
                        help='Sequence lengths for evaluation (extrapolate mode)')
    parser.add_argument('--loss_horizon', type=int, default=5,
                        help='Compute loss only at positions 1..K (horizon mode)')
    parser.add_argument('--sinusoidal_pe', action='store_true', default=False,
                        help='Use sinusoidal PE (default for extrapolate mode)')
    parser.add_argument('--opaque', action='store_true', default=False)
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_steps', type=int, default=150000)
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=500)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--output_dir', type=str,
                        default='results/extrapolation')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    for seed in args.seeds:
        mode_str = "OPAQUE" if args.opaque else "INTEGER"
        print(f"\n{'='*70}")
        print(f"EXTRAPOLATION ({args.mode.upper()}, {mode_str}), "
              f"SEED {seed}, DEVICE {args.device}")
        print(f"{'='*70}")

        np.random.seed(seed)
        _ensure_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        enc_str = "opaque" if args.opaque else "integer"
        seed_dir = os.path.join(
            args.output_dir, f'{args.mode}_{enc_str}_seed{seed}'
        )
        args_copy = argparse.Namespace(**vars(args))
        args_copy.output_dir = seed_dir

        train(args_copy)


if __name__ == '__main__':
    main()
