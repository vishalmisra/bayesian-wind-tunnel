"""
Probe ablation experiment: ablate (a, b) directions in residual stream
at trained positions of K=5 horizon model. Measure MAE before/after.

Design:
  1. Load K=5 horizon model.
  2. Train linear probes for a and b at each trained position 1..5
     (using the same protocol as recurrence_residual_probe.py).
  3. For each trained position, identify the probe-decodable subspace:
     span of rows of the stacked probe weight matrix [W_a; W_b], shape
     (34, d_model=192).
  4. Compute the orthogonal-complement projector Q = I - P^+ P, where
     P is the stacked probe matrix and P^+ its pseudo-inverse.
  5. Apply Q to the post-ln_final residual at each trained position
     before output_proj. Compute logits and MAE vs Bayesian posterior.
  6. Compare to no-ablation baseline.

Two outcomes:
  (a) MAE at trained positions degrades substantially after ablation:
      the probe-defined subspace is functionally relevant. The
      "program is in the residual stream and the readout consults it"
      framing is supported at trained positions.
  (b) MAE at trained positions stays similar after ablation:
      the probe is reading non-functional information. The truer
      framing is per-position circuits with computational residue
      that's decodable but not used.

Usage:
    python recurrence_probe_ablation.py \\
        --checkpoint results/extrapolation/horizon_integer_seed42/best_model.pt \\
        --device cuda:0 --output_dir results/probe_ablation_K5_seed42
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

from recurrence_bwt import RecurrenceConfig, sample_recurrence
from recurrence_extrapolation import _build_model_class


def generate_program_sequences(p, seq_len, n):
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


def forward_get_residuals_and_logits(model, sequences, device):
    """Forward pass returning post-ln_final residuals and original logits."""
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
        residuals = model.ln_final(x)  # post-ln residual
        logits = model.output_proj(residuals)
    return residuals, logits


def train_probe(X_train, y_train, n_classes, l2=1e-3, n_steps=3000,
                 lr=5e-3, device="cpu"):
    """Train a logistic-regression linear probe; return weight matrix (n_classes, d)."""
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    d = Xtr.shape[1]
    W = nn.Linear(d, n_classes).to(device)
    optim = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=l2)
    for _ in range(n_steps):
        logits = W(Xtr)
        loss = F.cross_entropy(logits, ytr)
        optim.zero_grad()
        loss.backward()
        optim.step()
    return W.weight.detach()  # shape (n_classes, d)


def predictive_entropy(logits, p):
    """logits shape (n, p) → mean entropy in bits per row, then average."""
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = log_probs.exp()
    H = -(probs * log_probs).sum(dim=-1) / np.log(2.0)
    return H.cpu().numpy()


def bayesian_predictive_entropy(p, sequences, K=15):
    """For each sequence, compute analytic Bayesian predictive entropy at each position."""
    from recurrence_bwt import bayesian_predictive_recurrence, _predictive_entropy
    pi = 0.5
    Hs = np.zeros((len(sequences), K))
    for i, s in enumerate(sequences):
        tokens = s["tokens"]
        for t in range(1, K):
            prefix = tokens[:t]
            pred_dist = bayesian_predictive_recurrence(prefix, p, pi)
            H = _predictive_entropy(pred_dist)
            Hs[i, t] = H
    return Hs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--p", type=int, default=17)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--n_eval", type=int, default=4000)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l2", type=float, default=1e-3,
                        help="probe L2 regularization (default 1e-3)")
    parser.add_argument("--random_subspace", action="store_true",
                        help="control: ablate 34 random Gaussian directions instead of probe-defined")
    parser.add_argument("--output_dir", default="results/probe_ablation_K5")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Build + load model
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

    # Generate sequences
    seqs = generate_program_sequences(args.p, args.seq_len, args.n_eval)
    a_labels = np.array([s["a"] for s in seqs])
    b_labels = np.array([s["b"] for s in seqs])

    # Forward pass: get residuals + baseline logits
    residuals, logits_baseline = forward_get_residuals_and_logits(
        model, seqs, device
    )
    print(f"Residuals: {residuals.shape}, Logits: {logits_baseline.shape}")

    # Compute Bayesian entropy ground truth
    H_bayes = bayesian_predictive_entropy(args.p, seqs, K=args.seq_len)

    # Train probes at each position
    probes = {}
    for t in range(1, args.seq_len):
        # Position t residual (B, d)
        X = residuals[:, t, :].cpu().numpy()
        # Train probes for a and b
        n = len(seqs)
        train_idx = np.arange(int(0.75 * n))
        W_a = train_probe(X[train_idx], a_labels[train_idx], args.p, l2=args.l2, device=device)
        W_b = train_probe(X[train_idx], b_labels[train_idx], args.p, l2=args.l2, device=device)
        probes[t] = {"W_a": W_a, "W_b": W_b}

    # For each test position, build ablation projector
    # P^{(t)} = stack([W_a, W_b]), shape (34, 192)
    # Q = I - P^+ P, shape (192, 192)
    print("\nBuilding ablation projectors...")
    Q_per_pos = {}
    d = residuals.shape[-1]
    I = torch.eye(d, device=device)
    for t in probes:
        if args.random_subspace:
            # Control: 34 random Gaussian directions, orthonormalized via QR.
            P_rand = torch.randn(2 * args.p, d, device=device)
            P_rand, _ = torch.linalg.qr(P_rand.T)  # d x 34, columns orthonormal
            P_rand = P_rand.T  # 34 x d
            P = P_rand
        else:
            P = torch.cat([probes[t]["W_a"], probes[t]["W_b"]], dim=0)  # (34, d)
        # Pseudo-inverse via SVD
        # Q = I - P^T (P P^T)^{-1} P
        PP_T = P @ P.T  # (34, 34)
        PP_T_inv = torch.linalg.pinv(PP_T)
        Q = I - P.T @ PP_T_inv @ P  # (d, d)
        Q_per_pos[t] = Q

    # Compute MAEs:
    # 1. Baseline (no ablation)
    # 2. Ablated at trained positions only
    # 3. Ablated at untrained positions only
    print("\nComputing MAEs...")

    # Baseline
    H_baseline = np.zeros_like(H_bayes)
    for t in range(1, args.seq_len):
        # logits at position t-1 predicts token at position t
        logits_t = logits_baseline[:, t - 1]  # (B, p)
        H_baseline[:, t] = predictive_entropy(logits_t, args.p)

    # Ablated at trained positions
    ablated_residuals_trained = residuals.clone()
    for t in range(1, args.K + 1):
        # logits at position t-1 used to predict position t
        # residual at position t-1 produces logits for position t
        # so we need to ablate position t-1 to affect prediction of position t
        # WAIT: actually the probe at "position t" was trained to predict the (a,b) of the sequence
        # using residuals at position t. The model's output at position t-1 predicts token t.
        # We want: remove the (a,b) info from residual at position t-1, see if logit at t-1 changes
        # But our probe was trained on residual at position t.
        # For the ablation to be meaningful: we should use the probe at position t-1
        # to ablate residual at position t-1.
        #
        # Actually, the cleanest setup: probe at position p extracts (a,b) from residual_p.
        # Logits at position p predict token p+1. So if we ablate the (a,b) info from residual_p,
        # the logits at position p (predicting token p+1) should change if the readout uses (a,b).
        ablated_residuals_trained[:, t - 1] = (
            Q_per_pos[t - 1] @ residuals[:, t - 1].T
        ).T if (t - 1) in Q_per_pos else residuals[:, t - 1]
    # Ablate position 0 too if probe exists (for t-1=0 when t=1)
    # Actually probe at 0 predicting (a,b) doesn't make sense (only x0 known)
    # Skip ablation at position 0
    with torch.no_grad():
        logits_ablated_trained = model.output_proj(ablated_residuals_trained)

    H_ablated_trained = np.zeros_like(H_bayes)
    for t in range(1, args.seq_len):
        H_ablated_trained[:, t] = predictive_entropy(
            logits_ablated_trained[:, t - 1], args.p
        )

    # Ablated at untrained positions (positions 5..14 affect predictions at 6..15)
    ablated_residuals_untrained = residuals.clone()
    for t in range(args.K + 1, args.seq_len):
        # ablate residual at position t-1 (which predicts token t)
        if (t - 1) in Q_per_pos:
            ablated_residuals_untrained[:, t - 1] = (
                Q_per_pos[t - 1] @ residuals[:, t - 1].T
            ).T

    with torch.no_grad():
        logits_ablated_untrained = model.output_proj(ablated_residuals_untrained)

    H_ablated_untrained = np.zeros_like(H_bayes)
    for t in range(1, args.seq_len):
        H_ablated_untrained[:, t] = predictive_entropy(
            logits_ablated_untrained[:, t - 1], args.p
        )

    # Compute MAEs
    mae_baseline = np.abs(H_baseline - H_bayes)
    mae_ab_trained = np.abs(H_ablated_trained - H_bayes)
    mae_ab_untrained = np.abs(H_ablated_untrained - H_bayes)

    print("\n" + "=" * 70)
    print(f"Per-position MAE (mean across {args.n_eval} sequences)")
    print("=" * 70)
    print(f"{'pos':>5} {'baseline':>12} {'ablate-trained':>18} {'ablate-untrained':>20} {'role':>12}")
    for t in range(1, args.seq_len):
        marker = "TRAINED" if t <= args.K else "UNTRAINED"
        print(f"{t:>5} {mae_baseline[:, t].mean():>12.4f} "
               f"{mae_ab_trained[:, t].mean():>18.4f} "
               f"{mae_ab_untrained[:, t].mean():>20.4f} {marker:>12}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    trained_baseline = mae_baseline[:, 1:args.K + 1].mean()
    trained_ablate_trained = mae_ab_trained[:, 1:args.K + 1].mean()
    untrained_baseline = mae_baseline[:, args.K + 1:].mean()
    untrained_ablate_untrained = mae_ab_untrained[:, args.K + 1:].mean()

    print(f"Trained positions (1-{args.K}):")
    print(f"  Baseline MAE: {trained_baseline:.4f} bits")
    print(f"  After ablating (a,b) directions: {trained_ablate_trained:.4f} bits")
    print(f"  Degradation: {trained_ablate_trained / max(1e-9, trained_baseline):.2f}x")
    print(f"")
    print(f"Untrained positions ({args.K + 1}-{args.seq_len-1}):")
    print(f"  Baseline MAE: {untrained_baseline:.4f} bits")
    print(f"  After ablating (a,b) directions at untrained positions: {untrained_ablate_untrained:.4f} bits")
    print(f"  Change: {untrained_ablate_untrained - untrained_baseline:+.4f} bits")
    print(f"")

    if trained_ablate_trained > 5 * trained_baseline:
        print("INTERPRETATION: ablating (a,b) directions DEGRADES trained-position prediction.")
        print("The (a,b) representation is functionally relevant; the readout consults it.")
        print("The 'program is in the residual stream' framing is supported at trained positions.")
    elif trained_ablate_trained < 1.5 * trained_baseline:
        print("INTERPRETATION: ablating (a,b) directions does NOT substantially affect")
        print("trained-position prediction. The probe is reading information the model isn't using.")
        print("Truer framing: per-position circuits with computational residue that's decodable")
        print("but not functional. The 'program is in the residual stream' framing overstates.")
    else:
        print("INTERPRETATION: partial degradation. The (a,b) directions matter to some extent")
        print("but not overwhelmingly. Mixed evidence.")

    out = {
        "mae_baseline_per_position": mae_baseline.mean(axis=0).tolist(),
        "mae_ablated_trained_per_position": mae_ab_trained.mean(axis=0).tolist(),
        "mae_ablated_untrained_per_position": mae_ab_untrained.mean(axis=0).tolist(),
        "trained_baseline": float(trained_baseline),
        "trained_ablated": float(trained_ablate_trained),
        "untrained_baseline": float(untrained_baseline),
        "untrained_ablated": float(untrained_ablate_untrained),
    }
    out_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
