"""
Shuffled Recurrence Control — routing determinism test for the π experiment.

Same task as recurrence_bwt.py, but observations are presented as
(time_index, value, SEP) triples in RANDOM temporal order, analogous
to the random-order affine control.

Sequential (original):  x_0, x_1, x_2, ..., x_T
                        Model predicts next token at each position.
                        Routing is fixed: "use last 3 values at positions 0,1,2."

Shuffled (this script):  (t_σ(0), x_{σ(0)}, SEP, t_σ(1), x_{σ(1)}, SEP, ...)
                         Model predicts value given time index and all previous pairs.
                         Routing is episode-dependent: must determine which
                         observations are temporally adjacent.

Bayesian ground truth: brute-force enumeration of all p^3 consistent
(a, b, x_0) triples. Tractable for p=17 (4913 triples, ~80K ops/prediction).

Prediction: FAIL (same as random-order affine). If it succeeds, the π result
is bulletproof. If it fails, routing determinism is confirmed as central.

Usage:
    python recurrence_shuffled_control.py --verify
    python recurrence_shuffled_control.py --seeds 42 --device cuda:0
"""

import numpy as np
import math
import argparse
import json
import os
from dataclasses import dataclass

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
# Recurrence arithmetic
# ============================================================================

def iterate_recurrence(a, b, x0, t, p):
    """Compute x_t by iterating x_{i+1} = a*x_i + b mod p from x_0."""
    x = x0
    for _ in range(t):
        x = (a * x + b) % p
    return x


def generate_full_sequence(a, b, x0, length, p):
    """Generate [x_0, x_1, ..., x_{length-1}]."""
    seq = [x0]
    x = x0
    for _ in range(length - 1):
        x = (a * x + b) % p
        seq.append(x)
    return seq


# ============================================================================
# Brute-force Bayesian inference for shuffled observations
# ============================================================================

def count_consistent_triples(observations, p):
    """
    Count (a, b, x_0) triples consistent with observed (time, value) pairs.
    Brute force over Z_p^3.
    """
    count = 0
    for a in range(p):
        for b in range(p):
            for x0 in range(p):
                ok = True
                for t, v in observations:
                    if iterate_recurrence(a, b, x0, t, p) != v:
                        ok = False
                        break
                if ok:
                    count += 1
    return count


def bayesian_predictive_shuffled(observations, t_query, p, pi):
    """
    Bayesian predictive P(x_{t_query} | observed pairs) under the mixture.

    Under H_P: enumerate consistent (a,b,x_0), predict x_{t_query} for each.
    Under H_R: uniform 1/p.

    Returns dict {value: probability}.
    """
    # Count consistent triples and their predictions
    pred_counts = [0] * p
    total_consistent = 0

    for a in range(p):
        for b in range(p):
            for x0 in range(p):
                ok = True
                for t, v in observations:
                    if iterate_recurrence(a, b, x0, t, p) != v:
                        ok = False
                        break
                if ok:
                    total_consistent += 1
                    x_pred = iterate_recurrence(a, b, x0, t_query, p)
                    pred_counts[x_pred] += 1

    # P(data | H_P) = total_consistent / p^3
    # P(data | H_R) = 1/p^k where k = len(observations)
    k = len(observations)

    if total_consistent == 0:
        # H_P falsified
        return {v: 1.0 / p for v in range(p)}, 0.0

    log_lik_hp = math.log(total_consistent) - 3 * math.log(p)
    log_lik_hr = -k * math.log(p)

    log_num = math.log(pi) + log_lik_hp
    log_den = np.logaddexp(log_num, math.log(1 - pi) + log_lik_hr)
    w = math.exp(log_num - log_den)

    pred = {}
    for v in range(p):
        p_hp_v = pred_counts[v] / total_consistent if total_consistent > 0 else 0.0
        p_hr_v = 1.0 / p
        pred[v] = w * p_hp_v + (1 - w) * p_hr_v

    return pred, w


def _predictive_entropy(pred_dist):
    H = 0.0
    for prob in pred_dist.values():
        if prob > 1e-15:
            H -= prob * math.log2(prob)
    return H


# ============================================================================
# Sequence generation
# ============================================================================

@dataclass
class ShuffledRecurrenceConfig:
    p: int = 17
    pi: float = 0.5
    seq_len: int = 16


def generate_shuffled_sequence(cfg):
    """
    Generate a shuffled recurrence sequence.

    Format: [t_σ(0), v_σ(0), SEP, t_σ(1), v_σ(1), SEP, ...]

    Token vocab:
      0..p-1:       value tokens
      p..p+T-1:     time-index tokens
      p+T:          SEP token
    """
    p = cfg.p
    T = cfg.seq_len
    SEP = p + T

    is_program = np.random.random() < cfg.pi

    if is_program:
        a = np.random.randint(0, p)
        b = np.random.randint(0, p)
        x0 = np.random.randint(0, p)
        full_seq = generate_full_sequence(a, b, x0, T, p)
        true_class = 'program'
    else:
        full_seq = [int(np.random.randint(0, p)) for _ in range(T)]
        true_class = 'random'

    # Shuffle presentation order
    order = np.random.permutation(T).tolist()

    tokens = []
    ground_truth = []
    observations = []  # (time_index, value) pairs seen so far

    for k_idx in range(T - 1):
        t_present = order[k_idx]
        v_present = full_seq[t_present]

        # Ground truth: predict v_present given observations so far
        pred_dist, w = bayesian_predictive_shuffled(
            observations, t_present, p, cfg.pi
        )
        H = _predictive_entropy(pred_dist)

        ground_truth.append({
            'k': k_idx,
            't': t_present,
            'entropy': H,
            'pred_dist': pred_dist,
            'p_program': w,
        })

        # Add tokens: [time_index, value, SEP]
        tokens.extend([t_present + p, v_present, SEP])  # time token offset by p

        observations.append((t_present, v_present))

    metadata = {
        'true_class': true_class,
        'p': p,
        'seq_len': T,
        'order': order,
        'vocab_size': p + T + 1,
    }
    return tokens, ground_truth, metadata


# ============================================================================
# Model (reuse from recurrence_bwt)
# ============================================================================

def _build_model_class():
    _ensure_torch()

    class _MultiHeadAttention(nn.Module):
        def __init__(self, d_model, n_heads, dropout=0.1):
            super().__init__()
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.out = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            B, T, D = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            scale = self.d_head ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            alpha = torch.softmax(attn, dim=-1)
            alpha = self.dropout(alpha)
            out = (alpha @ v).transpose(1, 2).reshape(B, T, D)
            return self.out(out), alpha

    class _TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
            super().__init__()
            d_ff = d_ff or 4 * d_model
            self.attn = _MultiHeadAttention(d_model, n_heads, dropout)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model), nn.Dropout(dropout),
            )

        def forward(self, x, mask=None):
            attn_out, alpha = self.attn(self.ln1(x), mask)
            x = x + attn_out
            x = x + self.ff(self.ln2(x))
            return x, alpha

    class ShuffledRecurrenceTransformer(nn.Module):
        def __init__(self, vocab_size, n_tokens, d_model=192, n_layers=6,
                     n_heads=6, d_ff=768, dropout=0.1):
            super().__init__()
            self.vocab_size = vocab_size
            self.n_tokens = n_tokens
            self.token_embed = nn.Embedding(vocab_size + 1, d_model,
                                            padding_idx=vocab_size)
            self.pos_embed = nn.Embedding(512, d_model)
            self.layers = nn.ModuleList([
                _TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
            self.ln_final = nn.LayerNorm(d_model)
            self.output_proj = nn.Linear(d_model, n_tokens)

        def forward(self, tokens):
            B, T = tokens.shape
            mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()
            x = self.token_embed(tokens)
            positions = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, -1)
            x = x + self.pos_embed(positions)
            all_alphas = []
            for layer in self.layers:
                x, alpha = layer(x, mask)
                all_alphas.append(alpha)
            x = self.ln_final(x)
            logits = self.output_proj(x)
            return logits, all_alphas

    return ShuffledRecurrenceTransformer


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_shuffled(model, cfg, n_eval=500, device='cpu'):
    """Evaluate model against Bayesian optimum. Fewer evals due to brute-force cost."""
    _ensure_torch()
    model.eval()
    p = cfg.p
    T = cfg.seq_len

    all_mae = []
    per_position = {}

    with torch.no_grad():
        for ep in range(n_eval):
            tokens, gt, metadata = generate_shuffled_sequence(cfg)
            tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            logits, _ = model(tokens_tensor)
            probs = torch.softmax(logits[0, :, :p], dim=-1).cpu().numpy()

            # Prediction positions: value token in each (time, value, SEP) triple
            for k_idx, gt_entry in enumerate(gt):
                # Value is at position k_idx*3 + 1, predicted from position k_idx*3
                # Model at pos (k_idx*3) sees time token, predicts value at (k_idx*3 + 1)
                model_pos = k_idx * 3  # position of time token
                if model_pos >= len(probs):
                    continue

                p_model = probs[model_pos]
                p_bayes = np.array([gt_entry['pred_dist'].get(v, 0.0) for v in range(p)])

                H_model = -sum(pm * math.log2(pm) for pm in p_model if pm > 1e-10)
                H_bayes = gt_entry['entropy']
                all_mae.append(abs(H_model - H_bayes))

                k = gt_entry['k']
                if k not in per_position:
                    per_position[k] = {'H_model': [], 'H_bayes': [], 'mae': []}
                per_position[k]['H_model'].append(H_model)
                per_position[k]['H_bayes'].append(H_bayes)
                per_position[k]['mae'].append(abs(H_model - H_bayes))

    metrics = {
        'mae_bits': float(np.mean(all_mae)) if all_mae else 0.0,
        'mae_std': float(np.std(all_mae)) if all_mae else 0.0,
    }

    per_pos_summary = {}
    for k in sorted(per_position.keys()):
        per_pos_summary[k] = {
            'H_model_mean': float(np.mean(per_position[k]['H_model'])),
            'H_bayes_mean': float(np.mean(per_position[k]['H_bayes'])),
            'mae_mean': float(np.mean(per_position[k]['mae'])),
            'count': len(per_position[k]['mae']),
        }

    return metrics, per_pos_summary


# ============================================================================
# Training
# ============================================================================

def get_prediction_positions_shuffled(tokens, p, T):
    """
    In (time, value, SEP) triples, the value is at position 3*k+1.
    The model predicts value from the time token at position 3*k.
    """
    positions = []
    for k in range((T - 1)):
        pos = 3 * k + 1  # value position
        if pos < len(tokens):
            positions.append(pos)
    return positions


def train_shuffled(args):
    _ensure_torch()
    Transformer = _build_model_class()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    p = args.p
    T = args.seq_len

    vocab_size = p + T + 1  # values + time indices + SEP
    n_tokens = p  # predict values in Z_p

    model = Transformer(
        vocab_size=vocab_size,
        n_tokens=n_tokens,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(pr.numel() for pr in model.parameters())
    print(f"Model: {param_count:,} parameters on {device}")
    print(f"Task: SHUFFLED recurrence vs random on Z_{p}")
    print(f"  Format: (time_idx, value, SEP) triples in random temporal order")
    print(f"  Sequence length: {T}")
    print(f"  Vocab: {vocab_size} (values 0..{p-1}, times {p}..{p+T-1}, SEP={p+T})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps)

    PAD = vocab_size
    os.makedirs(args.output_dir, exist_ok=True)
    best_mae = float('inf')
    losses = []

    for step in range(1, args.n_steps + 1):
        model.train()

        cfg = ShuffledRecurrenceConfig(p=p, pi=args.pi, seq_len=T)

        batch_tokens = []
        max_len = 0
        for _ in range(args.batch_size):
            tokens, _, _ = generate_shuffled_sequence(cfg)
            batch_tokens.append(tokens)
            max_len = max(max_len, len(tokens))

        padded = []
        for tokens in batch_tokens:
            padded.append(tokens + [PAD] * (max_len - len(tokens)))
        x = torch.tensor(padded, dtype=torch.long).to(device)

        logits, _ = model(x)

        loss = torch.tensor(0.0, device=device)
        count = 0
        for b_idx in range(args.batch_size):
            pred_positions = get_prediction_positions_shuffled(
                batch_tokens[b_idx], p, T)
            for pos in pred_positions:
                if pos < max_len:
                    target = x[b_idx, pos]
                    if target < p:  # value token
                        # Predict from time token at pos-1
                        loss = loss + F.cross_entropy(
                            logits[b_idx, pos - 1, :p], target)
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
            eval_cfg = ShuffledRecurrenceConfig(p=p, pi=args.pi, seq_len=T)
            metrics, per_pos = evaluate_shuffled(model, eval_cfg,
                                                  n_eval=500, device=device)
            print(f"  Eval: MAE={metrics['mae_bits']:.4f} bits")
            for k in sorted(per_pos.keys()):
                pp = per_pos[k]
                print(f"    k={k:2d}: H_model={pp['H_model_mean']:.4f}, "
                      f"H_bayes={pp['H_bayes_mean']:.4f}, MAE={pp['mae_mean']:.4f}")

            if metrics['mae_bits'] < best_mae:
                best_mae = metrics['mae_bits']
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, 'best_model.pt'))

    # Final eval
    eval_cfg = ShuffledRecurrenceConfig(p=p, pi=args.pi, seq_len=T)
    metrics, per_pos = evaluate_shuffled(model, eval_cfg, n_eval=1000,
                                          device=device)

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS (SHUFFLED recurrence)")
    print(f"{'='*70}")
    print(f"  Entropy MAE: {metrics['mae_bits']:.6f} ± {metrics['mae_std']:.6f} bits")
    for k in sorted(per_pos.keys()):
        pp = per_pos[k]
        print(f"    k={k:2d}: H_model={pp['H_model_mean']:.4f}, "
              f"H_bayes={pp['H_bayes_mean']:.4f}, MAE={pp['mae_mean']:.4f}")

    return metrics, per_pos, losses


# ============================================================================
# Verification
# ============================================================================

def verify_shuffled_bayesian(p=17):
    """Verify brute-force Bayesian computation matches sequential results."""
    print(f"Verifying shuffled Bayesian calculations for p={p}")
    log2p = math.log2(p)

    # Test 1: No observations → uniform
    print("\n  Test 1: No observations → uniform")
    pred, w = bayesian_predictive_shuffled([], 0, p, 0.5)
    H = _predictive_entropy(pred)
    assert abs(H - log2p) < 1e-10, f"H={H}"
    print(f"    PASS: H={H:.4f} = log2({p})")

    # Test 2: One observation → still uniform (same as sequential)
    print("\n  Test 2: One observation → uniform")
    pred, w = bayesian_predictive_shuffled([(0, 5)], 1, p, 0.5)
    H = _predictive_entropy(pred)
    assert abs(H - log2p) < 1e-10, f"H={H}"
    print(f"    PASS: H={H:.4f}")

    # Test 3: Two consecutive observations → uniform (same as sequential k=2)
    print("\n  Test 3: Two consecutive observations → uniform (generic)")
    a, b, x0 = 3, 7, 2
    seq = generate_full_sequence(a, b, x0, 10, p)
    if seq[0] != seq[1]:
        pred, w = bayesian_predictive_shuffled(
            [(0, seq[0]), (1, seq[1])], 2, p, 0.5)
        H = _predictive_entropy(pred)
        assert abs(H - log2p) < 1e-10, f"H={H}"
        print(f"    PASS: H={H:.4f} (BF=1 still)")

    # Test 4: Three consecutive observations of consistent sequence
    # → BF should be 1 (same as sequential k=3)
    print("\n  Test 4: Three observations → BF still 1")
    obs3 = [(0, seq[0]), (1, seq[1]), (2, seq[2])]
    pred, w = bayesian_predictive_shuffled(obs3, 3, p, 0.5)
    H = _predictive_entropy(pred)
    # After 3 obs, the sequential BF is 1, so w should be 0.5
    assert abs(w - 0.5) < 0.05, f"w={w}, expected ~0.5"
    print(f"    PASS: w={w:.4f} ≈ 0.5")

    # Test 5: Four observations → program detectable
    print("\n  Test 5: Four observations → BF > 1")
    obs4 = [(0, seq[0]), (1, seq[1]), (2, seq[2]), (3, seq[3])]
    pred, w = bayesian_predictive_shuffled(obs4, 4, p, 0.5)
    assert w > 0.8, f"w={w}, expected > 0.8"
    print(f"    PASS: w={w:.4f} (program detected)")

    # Test 6: Shuffled order gives same BF as sequential (information is the same)
    print("\n  Test 6: Shuffled order → same posterior as sequential")
    # Present observations (0,1,2,3) in order (2,0,3,1)
    obs_shuffled = [(2, seq[2]), (0, seq[0]), (3, seq[3]), (1, seq[1])]
    pred_shuf, w_shuf = bayesian_predictive_shuffled(obs_shuffled, 4, p, 0.5)
    pred_seq, w_seq = bayesian_predictive_shuffled(obs4, 4, p, 0.5)
    assert abs(w_shuf - w_seq) < 1e-10, f"w_shuf={w_shuf}, w_seq={w_seq}"
    print(f"    PASS: w_shuffled={w_shuf:.4f} = w_sequential={w_seq:.4f}")

    # Test 7: Verify the posterior trajectory matches sequential
    print("\n  Bonus: Posterior trajectory (sequential presentation)")
    for k in range(1, 8):
        obs = [(t, seq[t]) for t in range(k)]
        _, w = bayesian_predictive_shuffled(obs, k, p, 0.5)
        print(f"    k={k}: w={w:.6f}")

    print(f"\nAll verification tests passed!")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Shuffled recurrence control (routing determinism test)')
    parser.add_argument('--p', type=int, default=17)
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
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=500)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--output_dir', type=str, default='results/recurrence_shuffled')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verify', action='store_true')

    args = parser.parse_args()

    if args.verify:
        verify_shuffled_bayesian(p=args.p)
        return

    all_results = []
    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"SHUFFLED RECURRENCE, SEED {seed}, DEVICE {args.device}")
        print(f"{'='*70}")
        np.random.seed(seed)
        _ensure_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        seed_dir = os.path.join(args.output_dir, f'seed_{seed}')
        args_copy = argparse.Namespace(**vars(args))
        args_copy.output_dir = seed_dir

        metrics, per_pos, losses_list = train_shuffled(args_copy)

        result = {
            'seed': seed,
            'device': args.device,
            'metrics': metrics,
            'per_position': {str(k): v for k, v in per_pos.items()},
            'final_loss': float(np.mean(losses_list[-1000:])),
        }
        all_results.append(result)

    results_path = os.path.join(args.output_dir, 'shuffled_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
