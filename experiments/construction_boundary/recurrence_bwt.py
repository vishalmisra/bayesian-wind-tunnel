"""
Modular Recurrence Wind Tunnel — The π Experiment
===================================================

Tests the Shannon/Kolmogorov boundary using modular linear recurrences:
  H_P (program): x_{t+1} = ax_t + b mod p  (2-parameter program)
  H_R (random):  x_t ~ Uniform(Z_p) i.i.d.

Key property: output statistics are provably indistinguishable from random
for the first 3 observations. The Bayes factor is exactly 1 for t=0,1,2,
then jumps to p at t=3, p^2 at t=4, etc. This is the "π moment."

Sequence format: raw sequence x_0 x_1 x_2 ... x_T (no input-output pairs,
no SEP tokens). Model predicts next token at every position. Cross-entropy
loss at every position.

Required circuit: a = (x_2 - x_1)(x_1 - x_0)^{-1} mod p (modular inverse).
Same depth as affine detection. Routing is positionally fixed.

Prediction: Integer tokens succeed; opaque tokens fail.

Usage:
    # Verify Bayesian calculations
    python recurrence_bwt.py --verify

    # Integer tokens
    python recurrence_bwt.py --seeds 42 --device cuda:0

    # Opaque tokens
    python recurrence_bwt.py --opaque --seeds 42 --device cuda:4
"""

import numpy as np
import math
import argparse
import json
import os
from dataclasses import dataclass

# Lazy imports for torch (allows running --verify without GPU)
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
# Bayesian inference for modular linear recurrence
# ============================================================================

def count_consistent_recurrences(seq, p):
    """
    Count (a, b) pairs consistent with prefix x_0, x_1, ..., x_{k-1}.

    A recurrence x_{t+1} = ax_t + b mod p is consistent if all consecutive
    pairs satisfy the relation.

    Returns:
      k=0: p^2 (any a, b)
      k=1: p^2 (single value constrains nothing)
      k=2: p   (x_1 = a*x_0 + b, one equation in two unknowns)
      k>=3: 1 if consistent, 0 if not (generic case where x_0 != x_1)
    """
    k = len(seq)
    if k <= 1:
        return p * p
    if k == 2:
        # x_1 = a*x_0 + b mod p. For each a in Z_p, b = x_1 - a*x_0 mod p.
        return p
    # k >= 3: first two transitions determine (a, b)
    # x_1 = a*x_0 + b and x_2 = a*x_1 + b
    # => x_2 - x_1 = a*(x_1 - x_0) mod p
    dx0 = (seq[1] - seq[0]) % p
    dx1 = (seq[2] - seq[1]) % p
    if dx0 == 0:
        # x_0 == x_1 mod p: then b = x_1 - a*x_0 and x_2 = a*x_1 + b = a*(x_1-x_0) + x_1
        # Since x_0 == x_1: x_2 must equal x_1. If not, 0 consistent.
        # If x_2 == x_1, then any a works with b = x_1 - a*x_0, but
        # we also need all subsequent to equal x_1.
        if dx1 != 0:
            return 0
        # x_0 == x_1 == x_2. Check remaining: all must equal x_0.
        # For each a, b = x_0(1-a) mod p, and the fixed point is x_0.
        # The recurrence gives x_t = x_0 for all t. So all remaining must be x_0.
        for t in range(3, k):
            if seq[t] != seq[0]:
                return 0
        return p  # any a works (b determined)
    # dx0 != 0: a is uniquely determined
    dx0_inv = pow(dx0, p - 2, p)  # Fermat's little theorem
    a = (dx1 * dx0_inv) % p
    b = (seq[1] - a * seq[0]) % p
    # Verify remaining transitions
    for t in range(2, k - 1):
        if (a * seq[t] + b) % p != seq[t + 1]:
            return 0
    return 1


def recover_recurrence(seq, p):
    """
    Recover (a, b) from a sequence of length >= 3 with distinct consecutive values.
    Returns (a, b) or None if underdetermined or inconsistent.
    """
    if len(seq) < 3:
        return None
    dx0 = (seq[1] - seq[0]) % p
    dx1 = (seq[2] - seq[1]) % p
    if dx0 == 0:
        if dx1 != 0:
            return None
        # Fixed point case: any a works; return a=0, b=x_0
        return (0, seq[0] % p)
    dx0_inv = pow(dx0, p - 2, p)
    a = (dx1 * dx0_inv) % p
    b = (seq[1] - a * seq[0]) % p
    return (a, b)


def bayes_factor_recurrence(seq, p):
    """
    BF(H_P : H_R) for a consistent program sequence.

    P(data | H_P) = count_consistent / p^2
    P(data | H_R) = 1/p^k

    BF = count_consistent * p^{k-2}

    For the generic case (x_0 != x_1):
      k=0,1: BF = p^{k-2} * p^2 = p^k ... wait, let's be careful.
      k=0: P(data|H_P) = 1, P(data|H_R) = 1, BF = 1
      k=1: P(data|H_P) = p^2/(p^2) * (1/p) ... no.

    Let me think about this more carefully.

    P(x_0 | H_P) = 1/p (uniform initial state, any a,b)
    P(x_0 | H_R) = 1/p
    => BF(k=1) = 1

    P(x_0, x_1 | H_P) = sum_{a,b} (1/p^2) * (1/p) * [x_1 = ax_0+b]
                       = (1/p^2) * (1/p) * p  (for each x_1, p choices of (a,b) work)
                       = 1/p^2
    P(x_0, x_1 | H_R) = 1/p^2
    => BF(k=2) = 1

    P(x_0,...,x_2 | H_P) = sum_{a,b} (1/p^2)(1/p) prod [x_{t+1}=ax_t+b]
    With k=3: = (1/p^2)(1/p) * count_at_k3
    Generic (dx0 != 0): count = 1, so = 1/p^3
    P(data | H_R) = 1/p^3
    => BF(k=3) = 1  ... Hmm, that's not right either.

    Actually wait. Let me redo this.

    Under H_P: draw (a,b) ~ Uniform(Z_p^2), draw x_0 ~ Uniform(Z_p),
               then x_{t+1} = ax_t + b mod p deterministically.
    P(x_0,...,x_{k-1} | H_P) = (1/p^2) * (1/p) * I[seq consistent with some (a,b)]
                                ... summed over all (a,b).
    = (1/p^3) * count_consistent(seq, p)

    Under H_R: each x_t ~ Uniform(Z_p) independently.
    P(x_0,...,x_{k-1} | H_R) = 1/p^k

    BF = [(1/p^3) * count] / [1/p^k] = count * p^{k-3}

    k=1: count = p^2, BF = p^2 * p^{-2} = 1  ✓
    k=2: count = p, BF = p * p^{-1} = 1  ✓
    k=3 (generic, dx0≠0): count = 1, BF = 1 * 1 = 1  ✓
    k=4 (generic): count = 1, BF = p  ✓
    k=5: BF = p^2  ✓

    So BF = count * p^{k-3}, and for generic consistent sequences:
    BF(k<=3) = 1, BF(k) = p^{k-3} for k >= 3.

    The "π moment" is at k=4 (4th observation, index 3) where BF first
    exceeds 1, jumping to p.
    """
    k = len(seq)
    if k == 0:
        return 1.0
    count = count_consistent_recurrences(seq, p)
    if count == 0:
        return 0.0
    # BF = count * p^{k-3}
    log_bf = math.log(count) + (k - 3) * math.log(p)
    return math.exp(log_bf)


def class_posterior_recurrence(seq, p, pi):
    """
    P(H_P | x_0,...,x_{k-1}) with prior pi on H_P.

    w = pi * BF / (pi * BF + (1 - pi))
    """
    bf = bayes_factor_recurrence(seq, p)
    if bf == 0.0:
        return 0.0
    num = pi * bf
    den = num + (1 - pi)
    return num / den


def bayesian_predictive_recurrence(seq, p, pi):
    """
    Bayesian predictive P(x_t | x_0,...,x_{t-1}) under the mixture.

    Under H_P:
      - t=0: uniform 1/p
      - t=1: uniform 1/p (x_1 = ax_0 + b, marginal over (a,b) is uniform)
      - t=2: if x_0 = x_1, then all a give x_2 = x_1, so P(x_1) = 1.
              if x_0 ≠ x_1, then x_2 = a*x_1 + b where a is determined
              by (x_0, x_1, b), and marginalizing over the p consistent
              (a,b) pairs gives uniform 1/p.
              Actually: given x_0, x_1, the p consistent (a,b) are
              a ∈ Z_p, b = x_1 - a*x_0. Then x_2 = a*x_1 + b = a*(x_1-x_0) + x_1.
              If x_0 ≠ x_1, as a ranges over Z_p, a*(x_1-x_0) ranges over Z_p,
              so x_2 is uniform. If x_0 = x_1, x_2 = x_1 always.
      - t>=3 (generic, x_0≠x_1): (a,b) determined, prediction is deterministic.

    Under H_R: always 1/p.

    Returns: dict {value: probability} for all values in Z_p.
    """
    k = len(seq)  # number of observations so far (predicting position k)
    w = class_posterior_recurrence(seq, p, pi)

    pred = {}

    if k == 0:
        # Both predict uniform
        for v in range(p):
            pred[v] = 1.0 / p
        return pred

    if k == 1:
        # H_P: x_1 = ax_0 + b, marginalized over (a,b). For each value of x_1,
        # there are p choices of (a,b) giving that x_1. So uniform.
        # H_R: uniform.
        for v in range(p):
            pred[v] = 1.0 / p
        return pred

    if k == 2:
        # H_P: p consistent (a,b) pairs.
        dx = (seq[1] - seq[0]) % p
        if dx == 0:
            # x_0 = x_1: all consistent recurrences give x_2 = x_1
            pred_hp = {v: (1.0 if v == seq[0] else 0.0) for v in range(p)}
        else:
            # x_0 ≠ x_1: x_2 = a*(x_1-x_0) + x_1 mod p.
            # As a ranges over Z_p, this is uniform over Z_p.
            pred_hp = {v: 1.0 / p for v in range(p)}
        for v in range(p):
            pred[v] = w * pred_hp[v] + (1 - w) * (1.0 / p)
        return pred

    # k >= 3: check consistency
    count = count_consistent_recurrences(seq, p)

    if count == 0:
        # H_P falsified
        for v in range(p):
            pred[v] = 1.0 / p
        return pred

    # Recover (a, b) and predict deterministically under H_P
    ab = recover_recurrence(seq, p)
    if ab is None:
        # Shouldn't happen if count > 0, but fallback
        for v in range(p):
            pred[v] = 1.0 / p
        return pred

    a, b = ab
    x_next = (a * seq[-1] + b) % p

    # Handle the degenerate case: fixed point with multiple consistent (a,b)
    if count > 1:
        # All consistent (a,b) give x_next = seq[0] (the fixed point)
        pred_hp = {v: (1.0 if v == seq[0] else 0.0) for v in range(p)}
    else:
        pred_hp = {v: (1.0 if v == x_next else 0.0) for v in range(p)}

    for v in range(p):
        pred[v] = w * pred_hp[v] + (1 - w) * (1.0 / p)

    return pred


def _predictive_entropy(pred_dist):
    """Compute entropy in bits from a distribution dict."""
    H = 0.0
    for prob in pred_dist.values():
        if prob > 1e-15:
            H -= prob * math.log2(prob)
    return H


# ============================================================================
# Sequence generation
# ============================================================================

@dataclass
class RecurrenceConfig:
    p: int = 17
    pi: float = 0.5
    seq_len: int = 16      # number of tokens in the sequence
    opaque: bool = False    # opaque symbol relabeling


def sample_recurrence(p):
    """Sample a modular linear recurrence: draw (a, b, x_0), generate sequence."""
    a = np.random.randint(0, p)
    b = np.random.randint(0, p)
    x0 = np.random.randint(0, p)
    seq = [x0]
    x = x0
    for _ in range(p):  # generate enough tokens
        x = (a * x + b) % p
        seq.append(x)
    return (a, b), seq


def generate_recurrence_sequence(cfg):
    """
    Generate a recurrence vs random sequence for the BWT.

    With probability pi: draw (a, b) ~ Uniform(Z_p^2), x_0 ~ Uniform(Z_p),
                          then x_{t+1} = ax_t + b mod p.
    With probability 1-pi: draw each x_t ~ Uniform(Z_p) i.i.d.

    Returns:
        tokens: list of token ids (length = seq_len)
        ground_truth: list of dicts with Bayesian ground truth at each position
        metadata: dict with sequence-level info
    """
    p = cfg.p
    seq_len = cfg.seq_len

    is_program = np.random.random() < cfg.pi

    if is_program:
        (a, b), full_seq = sample_recurrence(p)
        seq = full_seq[:seq_len]
        true_class = 'program'
    else:
        seq = [int(np.random.randint(0, p)) for _ in range(seq_len)]
        true_class = 'random'

    # Compute ground truth at each prediction position
    ground_truth = []
    for t in range(seq_len):
        prefix = seq[:t]
        pred_dist = bayesian_predictive_recurrence(prefix, p, cfg.pi)
        H = _predictive_entropy(pred_dist)
        w = class_posterior_recurrence(prefix, p, cfg.pi)

        ground_truth.append({
            't': t,
            'entropy': H,
            'pred_dist': pred_dist,
            'p_program': w,
        })

    # Opaque relabeling
    if cfg.opaque:
        relabel = np.random.permutation(p).tolist()
        tokens = [relabel[x] for x in seq]
        # Shift token ids by p to distinguish from header
        # Format: [ORD, relabel[0], relabel[1], ..., relabel[p-1], SEP, opaque_seq...]
        ORD = 2 * p
        SEP = 2 * p + 1
        header = [ORD] + relabel + [SEP]
        tokens_shifted = [relabel[x] + p for x in seq]  # opaque tokens in range [p, 2p)

        # Relabel the predictive distributions to opaque token space
        gt_opaque = []
        for gt_entry in ground_truth:
            pred_opaque = {}
            for v, prob in gt_entry['pred_dist'].items():
                pred_opaque[relabel[v] + p] = prob
            gt_opaque.append({
                't': gt_entry['t'],
                'entropy': gt_entry['entropy'],
                'pred_dist': pred_opaque,
                'p_program': gt_entry['p_program'],
            })

        tokens_full = header + tokens_shifted
        metadata = {
            'true_class': true_class,
            'p': p,
            'opaque': True,
            'relabel': relabel,
            'header_len': len(header),
            'vocab_size': 2 * p + 2,
            'n_tokens': p,  # number of "data" token types per domain
        }
        return tokens_full, gt_opaque, metadata
    else:
        metadata = {
            'true_class': true_class,
            'p': p,
            'opaque': False,
            'vocab_size': p,
            'n_tokens': p,
            'header_len': 0,
        }
        return seq, ground_truth, metadata


# ============================================================================
# Model
# ============================================================================

class MultiHeadAttention(object):
    """Placeholder — will use torch version after _ensure_torch."""
    pass


def _build_model_class():
    """Build model classes after torch is imported."""
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
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x, mask=None):
            attn_out, alpha = self.attn(self.ln1(x), mask)
            x = x + attn_out
            x = x + self.ff(self.ln2(x))
            return x, alpha

    class RecurrenceTransformer(nn.Module):
        """
        Transformer for the recurrence BWT.

        Simpler than ModelSelectionTransformer: no SEP/ORD tokens in integer mode.
        Vocabulary:
          - Integer mode: tokens 0..p-1, vocab_size = p
          - Opaque mode: tokens 0..p-1 (header values) + p..2p-1 (opaque seq)
                         + 2p (ORD) + 2p+1 (SEP), vocab_size = 2p+2

        Predicts next token at every position.
        """

        def __init__(self, vocab_size, n_tokens, d_model=192, n_layers=6,
                     n_heads=6, d_ff=768, dropout=0.1):
            super().__init__()
            self.vocab_size = vocab_size
            self.n_tokens = n_tokens  # number of prediction classes
            self.d_model = d_model

            self.token_embed = nn.Embedding(vocab_size + 1, d_model,
                                            padding_idx=vocab_size)
            self.pos_embed = nn.Embedding(512, d_model)

            self.layers = nn.ModuleList([
                _TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])

            self.ln_final = nn.LayerNorm(d_model)
            # For integer: predict over Z_p (tokens 0..p-1)
            # For opaque: predict over opaque tokens (tokens p..2p-1)
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

    return RecurrenceTransformer


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_recurrence(model, cfg, n_eval=2000, device='cpu'):
    """Evaluate model against Bayesian optimum."""
    _ensure_torch()
    model.eval()
    p = cfg.p

    all_mae = []
    all_kl = []
    per_position = {}
    class_post_errors = []

    with torch.no_grad():
        for ep in range(n_eval):
            tokens, gt, metadata = generate_recurrence_sequence(cfg)
            header_len = metadata['header_len']
            n_tok = metadata['n_tokens']

            tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            logits, _ = model(tokens_tensor)

            if cfg.opaque:
                # Opaque: logits project to n_tokens = p classes
                # These correspond to opaque token range [p, 2p)
                probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            else:
                probs = torch.softmax(logits[0], dim=-1).cpu().numpy()

            for gt_entry in gt:
                t = gt_entry['t']
                # The model at position (header_len + t - 1) predicts token at (header_len + t)
                model_pos = header_len + t - 1
                if t == 0 or model_pos < 0 or model_pos >= len(probs):
                    # Can't predict position 0 (no prior context)
                    continue

                p_model = probs[model_pos]
                pred_dist = gt_entry['pred_dist']

                if cfg.opaque:
                    # pred_dist keys are in range [p, 2p). Model output indices are 0..p-1.
                    p_bayes = np.array([pred_dist.get(v + p, 0.0) for v in range(n_tok)])
                else:
                    p_bayes = np.array([pred_dist.get(v, 0.0) for v in range(n_tok)])

                H_model = -sum(pm * math.log2(pm) for pm in p_model[:n_tok] if pm > 1e-10)
                H_bayes = gt_entry['entropy']

                all_mae.append(abs(H_model - H_bayes))

                kl = sum(p_bayes[y] * math.log(p_bayes[y] / max(p_model[y], 1e-10))
                         for y in range(n_tok) if p_bayes[y] > 1e-10)
                all_kl.append(kl)

                # Class posterior extraction for t >= 4 (program determined)
                if t >= 4 and gt_entry['p_program'] > 0.01:
                    p_prog = gt_entry['p_program']
                    # Under H_P: deterministic prediction at some token
                    # Under H_R: uniform 1/p
                    # p_model(y*) = w + (1-w)/p => w = (p_model(y*) - 1/p) / (1 - 1/p)
                    if cfg.opaque:
                        best_y = max(pred_dist, key=pred_dist.get)
                        best_idx = best_y - p
                    else:
                        best_y = max(pred_dist, key=pred_dist.get)
                        best_idx = best_y
                    if 0 <= best_idx < n_tok:
                        p_model_best = float(p_model[best_idx])
                        denom = 1.0 - 1.0 / p
                        if abs(denom) > 1e-10:
                            w_model = (p_model_best - 1.0 / p) / denom
                            w_model = max(0.0, min(1.0, w_model))
                            class_post_errors.append(abs(w_model - p_prog))

                if t not in per_position:
                    per_position[t] = {'H_model': [], 'H_bayes': [], 'mae': []}
                per_position[t]['H_model'].append(H_model)
                per_position[t]['H_bayes'].append(H_bayes)
                per_position[t]['mae'].append(abs(H_model - H_bayes))

    metrics = {
        'mae_bits': float(np.mean(all_mae)) if all_mae else 0.0,
        'mae_std': float(np.std(all_mae)) if all_mae else 0.0,
        'kl_nats': float(np.mean(all_kl)) if all_kl else 0.0,
        'class_posterior_mae': float(np.mean(class_post_errors)) if class_post_errors else None,
        'class_posterior_std': float(np.std(class_post_errors)) if class_post_errors else None,
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

def train_recurrence(args):
    _ensure_torch()
    RecurrenceTransformer = _build_model_class()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    p = args.p

    if args.opaque:
        vocab_size = 2 * p + 2
        n_tokens = p
    else:
        vocab_size = p
        n_tokens = p

    model = RecurrenceTransformer(
        vocab_size=vocab_size,
        n_tokens=n_tokens,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(pr.numel() for pr in model.parameters())
    mode_str = "OPAQUE" if args.opaque else "INTEGER"
    print(f"Model: {param_count:,} parameters on {device}")
    print(f"Task: modular recurrence vs random on Z_{p} ({mode_str})")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Prior pi: {args.pi}")
    print(f"  BF trajectory: 1, 1, 1, {p}, {p**2}, {p**3}, ...")
    print(f"  Posterior (pi=0.5): 0.5, 0.5, 0.5, {p/(p+1):.3f}, {p**2/(p**2+1):.4f}, ...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps)

    PAD = vocab_size  # padding token
    os.makedirs(args.output_dir, exist_ok=True)
    best_mae = float('inf')
    losses = []

    for step in range(1, args.n_steps + 1):
        model.train()

        cfg = RecurrenceConfig(p=p, pi=args.pi, seq_len=args.seq_len, opaque=args.opaque)

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

        logits, _ = model(x)

        # Loss at every sequence position (next-token prediction)
        loss = torch.tensor(0.0, device=device)
        count = 0
        for b_idx in range(args.batch_size):
            header_len = len(batch_tokens[b_idx]) - args.seq_len
            seq_start = header_len
            seq_end = len(batch_tokens[b_idx])
            # Predict position t+1 from position t
            for t in range(seq_start, seq_end - 1):
                target = x[b_idx, t + 1]
                if target < vocab_size:  # not padding
                    if args.opaque:
                        # Target is in range [p, 2p). Shift to [0, p) for loss.
                        target_shifted = target - p
                        if 0 <= target_shifted < n_tokens:
                            loss = loss + F.cross_entropy(
                                logits[b_idx, t, :n_tokens], target_shifted)
                            count += 1
                    else:
                        if target < n_tokens:
                            loss = loss + F.cross_entropy(
                                logits[b_idx, t, :n_tokens], target)
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
            eval_cfg = RecurrenceConfig(p=p, pi=args.pi, seq_len=args.seq_len,
                                        opaque=args.opaque)
            metrics, per_pos = evaluate_recurrence(model, eval_cfg, n_eval=1000,
                                                   device=device)
            cp_str = (f", ClassPostMAE={metrics['class_posterior_mae']:.4f}"
                      if metrics['class_posterior_mae'] is not None else "")
            print(f"  Eval: MAE={metrics['mae_bits']:.4f} bits, "
                  f"KL={metrics['kl_nats']:.4f}{cp_str}")

            # Per-position summary
            for t in sorted(per_pos.keys()):
                pp = per_pos[t]
                print(f"    t={t:2d}: H_model={pp['H_model_mean']:.4f}, "
                      f"H_bayes={pp['H_bayes_mean']:.4f}, MAE={pp['mae_mean']:.4f}")

            if metrics['mae_bits'] < best_mae:
                best_mae = metrics['mae_bits']
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, 'best_model.pt'))
                print(f"    New best model saved (MAE={best_mae:.6f})")

    # Final evaluation
    eval_cfg = RecurrenceConfig(p=p, pi=args.pi, seq_len=args.seq_len,
                                opaque=args.opaque)
    metrics, per_pos = evaluate_recurrence(model, eval_cfg, n_eval=5000,
                                           device=device)

    mode_str = "opaque" if args.opaque else "integer"
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS (recurrence BWT, {mode_str})")
    print(f"{'='*70}")
    print(f"  Entropy MAE: {metrics['mae_bits']:.6f} ± {metrics['mae_std']:.6f} bits")
    print(f"  KL divergence: {metrics['kl_nats']:.6f} nats")
    if metrics['class_posterior_mae'] is not None:
        print(f"  Class posterior MAE: {metrics['class_posterior_mae']:.6f} "
              f"± {metrics['class_posterior_std']:.6f}")

    print(f"\n  Per-position entropy (model vs Bayes):")
    for t in sorted(per_pos.keys()):
        pp = per_pos[t]
        print(f"    t={t:2d}: H_model={pp['H_model_mean']:.4f}, "
              f"H_bayes={pp['H_bayes_mean']:.4f}, MAE={pp['mae_mean']:.4f}")

    return metrics, per_pos, losses


# ============================================================================
# Verification
# ============================================================================

def verify_bayesian_calculations(p=17):
    """Run unit tests on the Bayesian math."""
    print(f"Verifying recurrence BWT Bayesian calculations for p={p}")
    log2p = math.log2(p)
    pi = 0.5
    n_tests = 200

    # Test 1: BF = 1 after 3 observations of consistent sequence
    print(f"\n  Test 1: BF = 1 for first 3 observations (consistent seq)")
    for trial in range(n_tests):
        a = np.random.randint(0, p)
        b = np.random.randint(0, p)
        x = np.random.randint(0, p)
        seq = [x]
        for _ in range(3):
            x = (a * x + b) % p
            seq.append(x)
        # Skip degenerate case where x_0 = x_1
        if seq[0] == seq[1]:
            continue
        for k in [1, 2, 3]:
            bf = bayes_factor_recurrence(seq[:k], p)
            assert abs(bf - 1.0) < 1e-10, f"BF({k})={bf}, expected 1.0"
    print(f"    PASS: BF(1)=BF(2)=BF(3)=1 for generic sequences")

    # Test 2: BF = p after 4 consistent observations
    print(f"\n  Test 2: BF = p = {p} after 4 observations")
    found = 0
    for trial in range(n_tests):
        a = np.random.randint(1, p)  # a != 0 to avoid fixed points easily
        b = np.random.randint(0, p)
        x = np.random.randint(0, p)
        seq = [x]
        for _ in range(4):
            x = (a * x + b) % p
            seq.append(x)
        if seq[0] == seq[1]:
            continue
        bf = bayes_factor_recurrence(seq[:4], p)
        assert abs(bf - p) < 1e-6, f"BF(4)={bf}, expected {p}"
        found += 1
    assert found > 0, "No non-degenerate sequences found"
    print(f"    PASS: BF(4)={p} ({found} tests)")

    # Test 3: Consistent (a,b) count trajectory
    print(f"\n  Test 3: count_consistent trajectory (generic case)")
    for trial in range(n_tests):
        a = np.random.randint(1, p)
        b = np.random.randint(0, p)
        x = np.random.randint(0, p)
        seq = [x]
        for _ in range(6):
            x = (a * x + b) % p
            seq.append(x)
        if seq[0] == seq[1]:
            continue
        assert count_consistent_recurrences([], p) == p * p
        assert count_consistent_recurrences(seq[:1], p) == p * p
        assert count_consistent_recurrences(seq[:2], p) == p
        assert count_consistent_recurrences(seq[:3], p) == 1
        assert count_consistent_recurrences(seq[:4], p) == 1
        break  # one success is enough
    print(f"    PASS: counts = p^2, p^2, p, 1, 1, ...")

    # Test 4: Predictive entropy = log_2(p) for positions 0 and 1
    print(f"\n  Test 4: Predictive entropy = log2({p}) ≈ {log2p:.4f} for t=0,1")
    pred0 = bayesian_predictive_recurrence([], p, pi)
    H0 = _predictive_entropy(pred0)
    assert abs(H0 - log2p) < 1e-10, f"H(0)={H0}, expected {log2p}"

    for trial in range(n_tests):
        x0 = np.random.randint(0, p)
        pred1 = bayesian_predictive_recurrence([x0], p, pi)
        H1 = _predictive_entropy(pred1)
        assert abs(H1 - log2p) < 1e-10, f"H(1)={H1}, expected {log2p}"
    print(f"    PASS: H(0) = H(1) = {log2p:.4f} bits")

    # Test 5: Predictive entropy < log_2(p) at position 2 when H_P is true
    # and x_0 = x_1 (fixed point makes prediction deterministic under H_P)
    print(f"\n  Test 5: H(t=2) < log2(p) when x_0 = x_1 (fixed point)")
    # For generic case x_0 ≠ x_1, H(t=2) = log2(p) since marginalized pred is uniform.
    # But for x_0 = x_1, H(t=2) < log2(p) since H_P predicts x_2 = x_0.
    x0 = 5
    pred2_fp = bayesian_predictive_recurrence([x0, x0], p, pi)
    H2_fp = _predictive_entropy(pred2_fp)
    assert H2_fp < log2p - 0.01, f"H(2|fixed point)={H2_fp}, should be < {log2p}"
    # For generic case, entropy should still be log2(p)
    for trial in range(n_tests):
        x0 = np.random.randint(0, p)
        x1 = np.random.randint(0, p)
        if x0 == x1:
            continue
        pred2 = bayesian_predictive_recurrence([x0, x1], p, pi)
        H2 = _predictive_entropy(pred2)
        assert abs(H2 - log2p) < 1e-10, f"H(2|generic)={H2}, expected {log2p}"
        break
    print(f"    PASS: H(2|fixed point)={H2_fp:.4f} < {log2p:.4f}, "
          f"H(2|generic)={log2p:.4f}")

    # Test 6: Under H_R, predictive entropy stays at log_2(p) throughout
    print(f"\n  Test 6: Under pure random, entropy = log2(p) at all positions")
    # Set pi = 0 to force H_R
    for trial in range(n_tests):
        seq = [int(np.random.randint(0, p)) for _ in range(10)]
        for t in range(len(seq)):
            pred = bayesian_predictive_recurrence(seq[:t], p, pi=0.0)
            H = _predictive_entropy(pred)
            assert abs(H - log2p) < 1e-10, f"H(t={t})={H}, expected {log2p}"
        break  # one sequence is enough
    print(f"    PASS: H(t) = {log2p:.4f} for all t under pi=0")

    # Bonus: check that the posterior trajectory is correct
    print(f"\n  Bonus: Posterior trajectory for consistent program sequence")
    a, b = 3, 7
    x = 2
    seq = [x]
    for _ in range(8):
        x = (a * x + b) % p
        seq.append(x)
    if seq[0] != seq[1]:
        for k in range(1, len(seq) + 1):
            w = class_posterior_recurrence(seq[:k], p, pi)
            bf = bayes_factor_recurrence(seq[:k], p)
            print(f"    k={k}: BF={bf:.1f}, w={w:.6f}")

    print(f"\nAll verification tests passed!")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Modular recurrence BWT (the π experiment)')
    parser.add_argument('--p', type=int, default=17)
    parser.add_argument('--pi', type=float, default=0.5)
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--opaque', action='store_true', default=False,
                        help='Use opaque symbol relabeling')
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
    parser.add_argument('--output_dir', type=str, default='results/recurrence')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verify', action='store_true')

    args = parser.parse_args()

    if args.verify:
        verify_bayesian_calculations(p=args.p)
        return

    all_results = []
    for seed in args.seeds:
        mode_str = "OPAQUE" if args.opaque else "INTEGER"
        print(f"\n{'='*70}")
        print(f"RECURRENCE BWT ({mode_str}), SEED {seed}, DEVICE {args.device}")
        print(f"{'='*70}")
        np.random.seed(seed)
        _ensure_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        seed_dir = os.path.join(args.output_dir, mode_str.lower(), f'seed_{seed}')
        args_copy = argparse.Namespace(**vars(args))
        args_copy.output_dir = seed_dir

        metrics, per_pos, losses_list = train_recurrence(args_copy)

        result = {
            'seed': seed,
            'opaque': args.opaque,
            'device': args.device,
            'metrics': metrics,
            'per_position': {str(k): v for k, v in per_pos.items()},
            'final_loss': float(np.mean(losses_list[-1000:])),
        }
        all_results.append(result)

    # Save aggregate results
    mode_str = "opaque" if args.opaque else "integer"
    results_path = os.path.join(args.output_dir, mode_str,
                                'recurrence_bwt_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to {results_path}")

    maes = [r['metrics']['mae_bits'] for r in all_results]
    print(f"\nOverall MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f} bits")


if __name__ == '__main__':
    main()
