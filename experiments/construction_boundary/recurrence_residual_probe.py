"""
Residual-stream linear probe at unrewarded positions.

Tests whether (a, b) recurrence parameters are decodable from the
residual stream of a K=5 loss-horizon-trained model at positions
K+1..15, where the output predictions are at the uniform baseline.

Two outcomes:
  (i) (a, b) linearly decodable above chance -> construction-boundary
      claim survives in strong form ("the program exists internally
      but does not drive predictions without gradient pressure").
  (ii) (a, b) not decodable -> claim becomes even sharper (no internal
       representation either).

Either way, the result calibrates the construction-boundary framing.

Usage:
    python recurrence_residual_probe.py \
        --checkpoint results/extrapolation/horizon_integer_seed42/best_model.pt \
        --device cuda:0 --n_eval 4000 --output_dir results/probe_K5
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recurrence_bwt import (
    RecurrenceConfig,
    sample_recurrence,
)
from recurrence_extrapolation import _build_model_class


def generate_program_sequences(p, seq_len, n):
    """Generate n program-class sequences with their (a, b, x0) labels."""
    sequences = []
    for _ in range(n):
        (a, b), full_seq = sample_recurrence(p)
        sequences.append({
            "tokens": full_seq[:seq_len],
            "a": int(a),
            "b": int(b),
            "x0": int(full_seq[0]),
        })
    return sequences


def extract_residuals(model, sequences, device, p):
    """Forward sequences through the model; return residual stream and logits.

    Returns:
        residuals: tensor [N, T, d_model] (post final ln_final)
        logits: tensor [N, T, p]
    """
    model.eval()
    tokens = torch.tensor(
        [s["tokens"] for s in sequences], dtype=torch.long, device=device
    )
    B, T = tokens.shape

    with torch.no_grad():
        mask = torch.triu(
            torch.ones(T, T, device=device), diagonal=1
        ).bool()
        x = model.token_embed(tokens)
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = x + model.pos_embed(positions)
        for layer in model.layers:
            x, _ = layer(x, mask)
        x = model.ln_final(x)
        logits = model.output_proj(x)

    return x.detach().cpu().numpy(), logits.detach().cpu().numpy()


def predictive_entropy_per_position(logits, p):
    """Compute per-position predictive entropy in bits."""
    # logits: [N, T, p]
    log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    probs = np.exp(log_probs)
    entropy = -(probs * log_probs).sum(axis=-1) / np.log(2.0)
    return entropy.mean(axis=0)  # [T]


def train_linear_probe(X_train, y_train, X_test, y_test, n_classes,
                        l2=1e-3, n_steps=2000, lr=1e-2, device="cpu"):
    """Train a logistic-regression linear probe.

    X: numpy [N, d]; y: numpy int [N], values in [0, n_classes).
    Returns (test_acc, train_acc).
    """
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
    yte = torch.tensor(y_test, dtype=torch.long, device=device)

    d = Xtr.shape[1]
    W = nn.Linear(d, n_classes).to(device)
    optim = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=l2)

    for step in range(n_steps):
        logits = W(Xtr)
        loss = F.cross_entropy(logits, ytr)
        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        train_acc = (W(Xtr).argmax(-1) == ytr).float().mean().item()
        test_acc = (W(Xte).argmax(-1) == yte).float().mean().item()
    return test_acc, train_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                         help="Path to best_model.pt from K=5 horizon run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--p", type=int, default=17)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--K", type=int, default=5,
                         help="Loss horizon (positions 1..K were trained)")
    parser.add_argument("--n_eval", type=int, default=4000,
                         help="Total eval sequences (split 75/25 train/test)")
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=768)
    parser.add_argument("--probe_steps", type=int, default=3000)
    parser.add_argument("--probe_lr", type=float, default=5e-3)
    parser.add_argument("--probe_l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/probe_K5")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Build model with the same hyperparameters as the K=5 run (vocab_size=p
    # for integer mode; sinusoidal_pe=False for the saved K=5 horizon run).
    ModelClass = _build_model_class()
    model = ModelClass(
        vocab_size=args.p,
        n_tokens=args.p,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=0.0,
        sinusoidal_pe=False,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Param count: {sum(p.numel() for p in model.parameters()):,}")

    # Generate eval sequences and extract residuals.
    print(f"Generating {args.n_eval} program-class sequences...")
    seqs = generate_program_sequences(args.p, args.seq_len, args.n_eval)
    residuals, logits = extract_residuals(model, seqs, device, args.p)
    print(f"Residual shape: {residuals.shape}")
    print(f"Logits shape: {logits.shape}")

    # Sanity check: per-position predictive entropy.
    H_per_pos = predictive_entropy_per_position(logits, args.p)
    print(f"Per-position predictive entropy (bits):")
    for t in range(args.seq_len):
        marker = "[TRAINED]" if t <= args.K else "[UNTRAINED]"
        print(f"  t={t:2d}: H={H_per_pos[t]:.3f} bits  {marker}")

    # Labels: (a, b) for each sequence.
    a_labels = np.array([s["a"] for s in seqs])
    b_labels = np.array([s["b"] for s in seqs])
    x0_labels = np.array([s["x0"] for s in seqs])

    # Train/test split.
    n = len(seqs)
    perm = np.random.permutation(n)
    n_train = int(0.75 * n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    # Probe at every position in the sequence, with focus on K+1..seq_len-1.
    results = {
        "config": vars(args),
        "per_position_entropy_bits": H_per_pos.tolist(),
        "probe_accuracy": {},
    }

    print(f"\nTraining linear probes (positions 0..{args.seq_len-1})...")
    print(f"Chance accuracy: 1/{args.p} = {1.0/args.p:.4f}")
    print(f"Train/test split: {n_train}/{n - n_train}")
    print()

    for t in range(args.seq_len):
        X = residuals[:, t, :]
        results["probe_accuracy"][t] = {}

        for label_name, labels in [("a", a_labels), ("b", b_labels), ("x0", x0_labels)]:
            test_acc, train_acc = train_linear_probe(
                X[train_idx], labels[train_idx],
                X[test_idx], labels[test_idx],
                n_classes=args.p,
                l2=args.probe_l2,
                n_steps=args.probe_steps,
                lr=args.probe_lr,
                device=device,
            )
            results["probe_accuracy"][t][label_name] = {
                "test_acc": test_acc,
                "train_acc": train_acc,
            }

        marker = "[TRAINED]" if t <= args.K else "[UNTRAINED]"
        a_acc = results["probe_accuracy"][t]["a"]["test_acc"]
        b_acc = results["probe_accuracy"][t]["b"]["test_acc"]
        x0_acc = results["probe_accuracy"][t]["x0"]["test_acc"]
        print(f"  t={t:2d} {marker}  a: test={a_acc:.3f}  "
              f"b: test={b_acc:.3f}  x0: test={x0_acc:.3f}")

    # Save results.
    out_path = os.path.join(args.output_dir, "probe_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Summary.
    trained_a = np.mean([
        results["probe_accuracy"][t]["a"]["test_acc"]
        for t in range(1, args.K + 1)
    ])
    trained_b = np.mean([
        results["probe_accuracy"][t]["b"]["test_acc"]
        for t in range(1, args.K + 1)
    ])
    untrained_a = np.mean([
        results["probe_accuracy"][t]["a"]["test_acc"]
        for t in range(args.K + 1, args.seq_len)
    ])
    untrained_b = np.mean([
        results["probe_accuracy"][t]["b"]["test_acc"]
        for t in range(args.K + 1, args.seq_len)
    ])
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Chance accuracy: {1.0/args.p:.4f} (1/{args.p})")
    print(f"Trained positions (1..{args.K}):")
    print(f"  Mean a probe acc: {trained_a:.4f}")
    print(f"  Mean b probe acc: {trained_b:.4f}")
    print(f"Untrained positions ({args.K+1}..{args.seq_len-1}):")
    print(f"  Mean a probe acc: {untrained_a:.4f}")
    print(f"  Mean b probe acc: {untrained_b:.4f}")

    if untrained_a > 5 * (1.0 / args.p) or untrained_b > 5 * (1.0 / args.p):
        print()
        print("INTERPRETATION: probes recover (a, b) substantially above")
        print("chance at unrewarded positions -> the program is constructed")
        print("internally even when no gradient flows. The construction")
        print("boundary localizes to the output head, not representation.")
    elif untrained_a < 2 * (1.0 / args.p) and untrained_b < 2 * (1.0 / args.p):
        print()
        print("INTERPRETATION: probes do not recover (a, b) at unrewarded")
        print("positions -> no internal representation either. The")
        print("construction-boundary claim is sharper: gradient descent")
        print("has not constructed a generator program internally.")
    else:
        print()
        print("INTERPRETATION: partial decodability. The construction")
        print("boundary applies to representational quality as well as to")
        print("output, but with intermediate strength.")


if __name__ == "__main__":
    main()
