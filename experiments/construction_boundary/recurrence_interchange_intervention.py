"""
Recurrence interchange-intervention experiment.

Complements the existing probe-ablation experiment. The ablation showed the
linearly-decodable (a, b) subspace at trained positions is not *necessary*
(projecting it out doesn't degrade prediction). This test asks whether it is
*sufficient*: if we INJECT a different (a, b) representation into the residual
at a trained position, does the model's next-token prediction shift toward
what it would predict under the injected parameters?

Design:
  - Generate N pairs of independent sequences. Each pair is (seq_A, seq_B)
    with (a_A, b_A) != (a_B, b_B) as a filter.
  - Train per-position linear probes for a and b on the union of all sequences,
    using the same protocol as recurrence_probe_ablation.py.
  - For each pair and each trained predict-position t in [2..K]:
      P_proj = projector onto the (W_a, W_b)-row span at position (t-1)
      r_patched = (I - P_proj) r_B[t-1] + P_proj r_A[t-1]
    Pass r_patched through model.output_proj to get patched logits.
  - Compute three predictive distributions per (pair, t):
      P_B   = softmax(output_proj(r_B[t-1]))           # baseline, seq_B
      P_A   = softmax(output_proj(r_A[t-1]))           # reference, seq_A
      P_inj = softmax(output_proj(r_patched))          # seq_B with (a,b)<-A
  - Score each on:
      target_B[t]         = (a_B * x_B[t-1] + b_B) mod p   # truth for seq_B
      target_A_swapped[t] = (a_A * x_B[t-1] + b_A) mod p   # what (a_A, b_A)
                                                          # would predict
                                                          # given x_B[t-1]
    Filter pairs/positions where target_B == target_A_swapped (the test is
    informative only when these differ).

Headline numbers reported per trained position and pooled across positions:
  - P_B(target_B)     baseline vs patched   (should DROP if subspace is used)
  - P_inj(target_A_swapped) vs P_B(target_A_swapped) (should RISE)
  - Shift fraction in log-prob space:
        shift = (logP_inj(A_swap) - logP_B(A_swap)) /
                (logP_A(A_swap)   - logP_B(A_swap))
    Bounded near 0 if no shift, near 1 if patched matches A's prediction.

Usage:
    python recurrence_interchange_intervention.py \\
        --checkpoint results/extrapolation/horizon_integer_seed42/best_model.pt \\
        --device cuda:0 --output_dir results/interchange_K5_seed42
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
        residuals = model.ln_final(x)
        logits = model.output_proj(residuals)
    return residuals, logits


def train_probe(X_train, y_train, n_classes, l2=1e-3, n_steps=3000,
                 lr=5e-3, device="cpu"):
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
    return W.weight.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--p", type=int, default=17)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--n_pairs", type=int, default=2000)
    parser.add_argument("--n_probe_train", type=int, default=4000,
                        help="number of sequences for probe training")
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/interchange_K5")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

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

    # 1) Probe training set: independent sequences for fitting probes only.
    probe_seqs = generate_program_sequences(args.p, args.seq_len, args.n_probe_train)
    probe_residuals, _ = forward_get_residuals_and_logits(model, probe_seqs, device)
    a_labels = np.array([s["a"] for s in probe_seqs])
    b_labels = np.array([s["b"] for s in probe_seqs])

    print(f"Training probes on {args.n_probe_train} sequences...")
    probes = {}
    for t in range(1, args.seq_len):
        X = probe_residuals[:, t, :].cpu().numpy()
        n_train = int(0.75 * len(probe_seqs))
        W_a = train_probe(X[:n_train], a_labels[:n_train], args.p, device=device)
        W_b = train_probe(X[:n_train], b_labels[:n_train], args.p, device=device)
        probes[t] = {"W_a": W_a, "W_b": W_b}
    del probe_residuals  # free memory

    # 2) Build (a,b)-subspace projectors per position.
    d = args.d_model
    I = torch.eye(d, device=device)
    P_proj_per_pos = {}
    for t in probes:
        P = torch.cat([probes[t]["W_a"], probes[t]["W_b"]], dim=0)  # (2p, d)
        PP_T = P @ P.T  # (2p, 2p)
        PP_T_inv = torch.linalg.pinv(PP_T)
        P_proj = P.T @ PP_T_inv @ P  # (d, d), projector onto row-span of P
        P_proj_per_pos[t] = P_proj

    # 3) Generate intervention pairs (independent of probe-training set).
    pair_seqs_A = generate_program_sequences(args.p, args.seq_len, args.n_pairs)
    pair_seqs_B = generate_program_sequences(args.p, args.seq_len, args.n_pairs)

    # Filter pairs where (a_A, b_A) == (a_B, b_B) — uninformative
    keep = []
    for i in range(args.n_pairs):
        if (pair_seqs_A[i]["a"], pair_seqs_A[i]["b"]) != (
            pair_seqs_B[i]["a"], pair_seqs_B[i]["b"]
        ):
            keep.append(i)
    pair_seqs_A = [pair_seqs_A[i] for i in keep]
    pair_seqs_B = [pair_seqs_B[i] for i in keep]
    n_pairs = len(pair_seqs_A)
    print(f"Pairs after distinct-(a,b) filter: {n_pairs}/{args.n_pairs}")

    # 4) Forward pairs.
    res_A, logits_A_full = forward_get_residuals_and_logits(model, pair_seqs_A, device)
    res_B, logits_B_full = forward_get_residuals_and_logits(model, pair_seqs_B, device)

    # 5) For each predict-position t in [2..K], patch residual at t-1.
    print("\nRunning interchange interventions...")
    rows = []
    for t in range(2, args.K + 1):
        s = t - 1  # residual position whose patch affects logits-for-token-t
        if s not in P_proj_per_pos:
            continue
        Pp = P_proj_per_pos[s]
        Q = I - Pp

        r_A = res_A[:, s]  # (n_pairs, d)
        r_B = res_B[:, s]
        # Patched residual: keep B's orthogonal complement, replace (a,b) with A's.
        r_patched = r_B @ Q.T + r_A @ Pp.T

        with torch.no_grad():
            logits_B = model.output_proj(r_B)         # (n_pairs, p)
            logits_A = model.output_proj(r_A)
            logits_inj = model.output_proj(r_patched)
            log_p_B = F.log_softmax(logits_B, dim=-1)
            log_p_A = F.log_softmax(logits_A, dim=-1)
            log_p_inj = F.log_softmax(logits_inj, dim=-1)

        # Build target tokens.
        # target_B[t]         = next token for seq_B under its true params
        # target_A_swapped[t] = next token IF we substitute (a_A, b_A) on x_B[s]
        # x at residual position s == seq_B.tokens[s]
        a_B = torch.tensor([s_["a"] for s_ in pair_seqs_B], dtype=torch.long, device=device)
        b_B = torch.tensor([s_["b"] for s_ in pair_seqs_B], dtype=torch.long, device=device)
        a_A = torch.tensor([s_["a"] for s_ in pair_seqs_A], dtype=torch.long, device=device)
        b_A = torch.tensor([s_["b"] for s_ in pair_seqs_A], dtype=torch.long, device=device)
        x_B_s = torch.tensor(
            [seq["tokens"][s] for seq in pair_seqs_B], dtype=torch.long, device=device
        )
        target_B = (a_B * x_B_s + b_B) % args.p
        target_A_swap = (a_A * x_B_s + b_A) % args.p

        informative = (target_B != target_A_swap)
        n_inf = informative.sum().item()
        if n_inf == 0:
            continue

        idx = torch.arange(n_pairs, device=device)

        log_p_B_at_B = log_p_B[idx, target_B]
        log_p_B_at_A = log_p_B[idx, target_A_swap]
        log_p_A_at_B = log_p_A[idx, target_B]
        log_p_A_at_A = log_p_A[idx, target_A_swap]
        log_p_inj_at_B = log_p_inj[idx, target_B]
        log_p_inj_at_A = log_p_inj[idx, target_A_swap]

        denom = (log_p_A_at_A - log_p_B_at_A)
        # Avoid division when denom is tiny (no anchor difference)
        ok = informative & (denom.abs() > 1e-3)
        if ok.sum().item() < 5:
            continue
        shift_frac = (log_p_inj_at_A - log_p_B_at_A) / denom
        shift_frac = shift_frac[ok]

        row = {
            "predict_position": t,
            "patch_residual_position": s,
            "n_informative_pairs": n_inf,
            "n_pairs_for_shift": int(ok.sum().item()),
            "P_B(target_B) baseline": log_p_B_at_B[informative].exp().mean().item(),
            "P_inj(target_B) patched": log_p_inj_at_B[informative].exp().mean().item(),
            "P_B(target_A_swap) baseline": log_p_B_at_A[informative].exp().mean().item(),
            "P_inj(target_A_swap) patched": log_p_inj_at_A[informative].exp().mean().item(),
            "P_A(target_A_swap) reference": log_p_A_at_A[informative].exp().mean().item(),
            "shift_fraction_logprob_mean": float(shift_frac.mean().item()),
            "shift_fraction_logprob_median": float(shift_frac.median().item()),
            "shift_fraction_logprob_p25": float(torch.quantile(shift_frac, 0.25).item()),
            "shift_fraction_logprob_p75": float(torch.quantile(shift_frac, 0.75).item()),
        }
        rows.append(row)

    # 6) Print + save.
    print("\n" + "=" * 110)
    print("INTERCHANGE INTERVENTION: trained-position predictions")
    print("=" * 110)
    print(
        f"{'pos':>4} {'n_inf':>7} {'P_B(B) base':>14} {'P_inj(B)':>10} "
        f"{'P_B(Asw) base':>14} {'P_inj(Asw)':>12} {'P_A(Asw) ref':>14} "
        f"{'shift_med':>10} {'shift_IQR':>14}"
    )
    for r in rows:
        iqr = f"[{r['shift_fraction_logprob_p25']:+.2f}, {r['shift_fraction_logprob_p75']:+.2f}]"
        print(
            f"{r['predict_position']:>4} {r['n_informative_pairs']:>7} "
            f"{r['P_B(target_B) baseline']:>14.4f} "
            f"{r['P_inj(target_B) patched']:>10.4f} "
            f"{r['P_B(target_A_swap) baseline']:>14.4f} "
            f"{r['P_inj(target_A_swap) patched']:>12.4f} "
            f"{r['P_A(target_A_swap) reference']:>14.4f} "
            f"{r['shift_fraction_logprob_median']:>+10.3f} {iqr:>14}"
        )

    # Pooled summary
    if rows:
        pooled_pB_base = np.mean([r["P_B(target_B) baseline"] for r in rows])
        pooled_pB_inj = np.mean([r["P_inj(target_B) patched"] for r in rows])
        pooled_pA_base = np.mean([r["P_B(target_A_swap) baseline"] for r in rows])
        pooled_pA_inj = np.mean([r["P_inj(target_A_swap) patched"] for r in rows])
        pooled_shift = np.mean([r["shift_fraction_logprob_median"] for r in rows])
        print("\n" + "=" * 70)
        print("POOLED across trained predict-positions 2..K")
        print("=" * 70)
        print(f"  P_B(target_B)         baseline -> patched: {pooled_pB_base:.4f} -> {pooled_pB_inj:.4f}")
        print(f"  P_*(target_A_swapped) baseline -> patched: {pooled_pA_base:.4f} -> {pooled_pA_inj:.4f}")
        print(f"  Mean median shift fraction (logprob):       {pooled_shift:+.3f}")
        print()
        print("INTERPRETATION GUIDE")
        if pooled_shift > 0.5 and pooled_pA_inj > 3 * pooled_pA_base:
            print("  Substantial shift -> the (a,b) subspace IS sufficient. The model")
            print("  uses the linearly-decodable representation at trained positions.")
            print("  This contradicts the ablation interpretation; investigate further")
            print("  before titling the paper around 'compiles correlations not models'.")
        elif abs(pooled_shift) < 0.15 and pooled_pA_inj < 2 * pooled_pA_base:
            print("  Negligible shift -> the (a,b) subspace is NOT sufficient either.")
            print("  Combined with ablation (not necessary), this gives a clean")
            print("  necessity-and-sufficiency falsification of the readout-binding")
            print("  reading. The probe is computational residue.")
        else:
            print("  Partial shift -> mixed evidence. Inspect per-position rows.")

    out = {
        "per_position": rows,
        "n_pairs_total": n_pairs,
    }
    out_path = os.path.join(args.output_dir, "interchange_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
