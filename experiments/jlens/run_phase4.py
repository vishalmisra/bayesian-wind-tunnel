#!/usr/bin/env python3
"""
Phase 4: the horizon boundary (P5, Nature-paper bridge).

In the K=5 loss-horizon recurrence model, is the J-space well-formed at
source positions inside the trained horizon and noise past it?

Per model and source position we report:
  * calibration MAE (sanity: replicate the trained/untrained gap),
  * disjoint-batch J-space stability (G0-style, r=8),
  * frame-subspace overlap ratio-to-null (embedding-mode token frame),
  * next-token decode from J-coordinates on program sequences at
    positions where the recurrence is determined (k >= 3).

Gate (preregistered): overlap well-formed at positions <= 5, falling to
null past 5. A full-horizon control model shows the curve is about
TRAINING, not position. P5 *succeeding in reverse* (workspace intact past
the horizon) must be reported if found.

Usage:
    python experiments/jlens/run_phase4.py \
        --k5-checkpoints <a.pt> <b.pt> ... \
        --control-checkpoint <full_horizon.pt> \
        --out experiments/jlens/artifacts/phase4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "construction_boundary"))

from recurrence_bwt import bayesian_predictive_recurrence  # noqa: E402

from experiments.jlens.extract import accumulate_grams, capture_names  # noqa: E402
from experiments.jlens.metrics import probe_from_coords, projection_overlap  # noqa: E402
from experiments.jlens.models import load_model  # noqa: E402
from experiments.jlens.subspaces import (  # noqa: E402
    default_hypothesis_tokens,
    frame_subspace,
    random_subspaces,
)

RANK = 8
REDUCTION = "stacked"


def generate_recurrence_batch(B: int, p: int, seq_len: int, pi: float, seed: int):
    """Tokens (B, T) + per-position Bayes predictive (B, T, p) + labels.

    bayes[b, i] is the predictive P(x_i | x_0..x_{i-1}); is_program marks
    sequences drawn from H_P.
    """
    rng = np.random.default_rng(seed)
    tokens = np.zeros((B, seq_len), dtype=np.int64)
    bayes = np.zeros((B, seq_len, p))
    is_program = np.zeros(B, dtype=bool)
    for b in range(B):
        if rng.random() < pi:
            a, off, x0 = rng.integers(0, p, size=3)
            seq = [int(x0)]
            for _ in range(seq_len - 1):
                seq.append(int((a * seq[-1] + off) % p))
            is_program[b] = True
        else:
            seq = [int(v) for v in rng.integers(0, p, size=seq_len)]
        tokens[b] = seq
        for i in range(seq_len):
            pred = bayesian_predictive_recurrence(seq[:i], p, pi)
            bayes[b, i] = [pred[v] for v in range(p)]
    return (
        torch.from_numpy(tokens),
        torch.from_numpy(bayes),
        torch.from_numpy(is_program),
    )


@torch.no_grad()
def calibration_curve(model, tokens, bayes) -> np.ndarray:
    """Per-position |H_model - H_bayes| in bits."""
    device = next(model.parameters()).device
    logits = model.logits(tokens.to(device)).float().cpu()
    pm = F.softmax(logits, dim=-1)
    h_model = -(pm * (pm + 1e-12).log2()).sum(-1)  # (B, T) at predicting pos i+1
    pb = bayes.float()
    h_bayes = -(pb * (pb + 1e-12).log2()).sum(-1)  # (B, T) at pos i
    # logits at position i predict token i+1: align to target position.
    T = tokens.shape[1]
    mae = torch.zeros(T)
    mae[0] = float("nan")  # position 0 has no model prediction
    for i in range(1, T):
        mae[i] = (h_model[:, i - 1] - h_bayes[:, i]).abs().mean()
    return mae.numpy()


def analyze_model(tag, ckpt, args, out):
    model = load_model(ckpt, device=args.device)
    hyp = default_hypothesis_tokens(model)
    p = model.vocab_size
    print(f"\n=== {tag}: {Path(ckpt).parent.name} "
          f"(p={p}, layers={model.n_layers}) ===")

    tokens, bayes, is_prog = generate_recurrence_batch(
        2 * args.n_seq, p, args.seq_len, args.pi, args.seed
    )
    mae = calibration_curve(model, tokens, bayes)
    print("calibration MAE by target position:",
          np.round(mae, 3).tolist())

    halves = [tokens[: args.n_seq], tokens[args.n_seq :]]
    sweeps = [
        accumulate_grams(
            model, h, seq_chunk=args.seq_chunk, cot_dims=hyp.tolist()
        )
        for h in halves
    ]
    names = capture_names(model.n_layers)
    T = args.seq_len
    frame = frame_subspace(model, mode="embedding", hypotheses=hyp)
    nulls = random_subspaces(model.dim, RANK, n=args.n_nulls, seed=123)
    null_mean = float(
        np.mean([projection_overlap(nulls[i], frame) for i in range(args.n_nulls)])
    )

    stability = np.full((len(names), T), np.nan)
    overlap_ratio = np.full((len(names), T), np.nan)
    for li, name in enumerate(names):
        for i in range(T):
            if sweeps[0].counts[REDUCTION][name][i] == 0:
                continue
            U0 = sweeps[0].top_subspace(REDUCTION, name, i, RANK)
            U1 = sweeps[1].top_subspace(REDUCTION, name, i, RANK)
            stability[li, i] = projection_overlap(U0, U1)
            overlap_ratio[li, i] = projection_overlap(U0, frame) / null_mean

    # Next-token decode from J-coords, program sequences, determined region
    # (k >= 3). Labels = the true next token.
    from experiments.jlens.interventions import capture_residuals

    prog = tokens[is_prog]
    decode = np.full((len(names), T), np.nan)
    for li, name in enumerate(names):
        resid = capture_residuals(model, prog, name).cpu().float()
        for i in range(3, T - 1):
            if sweeps[0].counts[REDUCTION][name][i] == 0:
                continue
            U = sweeps[0].top_subspace(REDUCTION, name, i, 16)
            coords = resid[:, i, :] @ U
            labels = prog[:, i + 1]
            decode[li, i] = probe_from_coords(coords, labels)

    # Summaries over the layer band that carries the workspace (best layer
    # by in-horizon overlap), source positions grouped by the boundary.
    in_pos = [i for i in range(2, 6)]
    out_pos = [i for i in range(8, T - 1)]
    per_layer_in = np.nanmean(overlap_ratio[:, in_pos], axis=1)
    best_li = int(np.nanargmax(per_layer_in))
    # The workspace-overlap claim is judged at the frame-carrying band
    # (best layer by in-horizon overlap, = emb in practice); the
    # COMPUTATION claim (is the next token computed?) is judged at the
    # last layer, where the prediction must live if it exists.
    decode_in = [i for i in in_pos if i >= 3]
    res = {
        "checkpoint": ckpt,
        "calibration_mae_by_position": mae.tolist(),
        "null_mean": null_mean,
        "best_layer": names[best_li],
        "overlap_ratio_best_layer": overlap_ratio[best_li].tolist(),
        "stability_best_layer": stability[best_li].tolist(),
        "decode_last_layer": decode[-1].tolist(),
        "in_horizon_ratio": float(np.nanmean(overlap_ratio[best_li, in_pos])),
        "post_horizon_ratio": float(np.nanmean(overlap_ratio[best_li, out_pos])),
        "in_horizon_stability": float(np.nanmean(stability[best_li, in_pos])),
        "post_horizon_stability": float(np.nanmean(stability[best_li, out_pos])),
        "in_horizon_decode_last": float(np.nanmean(decode[-1, decode_in])),
        "post_horizon_decode_last": float(np.nanmean(decode[-1, out_pos])),
        "grids": {
            "overlap_ratio": overlap_ratio.tolist(),
            "stability": stability.tolist(),
            "decode": decode.tolist(),
            "layer_names": names,
        },
    }
    print(f"best layer {names[best_li]}: "
          f"overlap ratio in-horizon={res['in_horizon_ratio']:.2f} "
          f"post-horizon={res['post_horizon_ratio']:.2f}; "
          f"stability {res['in_horizon_stability']:.2f} -> "
          f"{res['post_horizon_stability']:.2f}; "
          f"last-layer decode {res['in_horizon_decode_last']:.2f} -> "
          f"{res['post_horizon_decode_last']:.2f}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k5-checkpoints", nargs="+", required=True)
    ap.add_argument("--control-checkpoint", default=None,
                    help="full-horizon recurrence model (positive control)")
    ap.add_argument("--out", default="experiments/jlens/artifacts/phase4")
    ap.add_argument("--n-seq", type=int, default=256, help="per disjoint half")
    ap.add_argument("--seq-chunk", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--pi", type=float, default=0.5)
    ap.add_argument("--n-nulls", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = {"k5": [], "control": None}
    for ckpt in args.k5_checkpoints:
        results["k5"].append(analyze_model("K5", ckpt, args, out))
    if args.control_checkpoint:
        results["control"] = analyze_model(
            "control", args.control_checkpoint, args, out
        )

    in_r = float(np.mean([r["in_horizon_ratio"] for r in results["k5"]]))
    post_r = float(np.mean([r["post_horizon_ratio"] for r in results["k5"]]))
    p5_pass = in_r >= 3.0 and post_r <= 1.5
    reverse = post_r >= 3.0  # must be reported if found
    ctrl_note = None
    if results["control"]:
        ctrl_note = {
            "in_horizon_ratio": results["control"]["in_horizon_ratio"],
            "post_horizon_ratio": results["control"]["post_horizon_ratio"],
        }

    dec_in = float(np.mean([r["in_horizon_decode_last"] for r in results["k5"]]))
    dec_post = float(np.mean([r["post_horizon_decode_last"] for r in results["k5"]]))
    results["P5"] = {
        "k5_mean_in_horizon_ratio": in_r,
        "k5_mean_post_horizon_ratio": post_r,
        "k5_mean_decode_last_in": dec_in,
        "k5_mean_decode_last_post": dec_post,
        "pass": bool(p5_pass),
        "reverse_finding": bool(reverse),
        "control": ctrl_note,
        "interpretation": (
            "Workspace geometry (frame-aligned J-space) persists past the "
            "horizon while the last-layer next-token computation stops at "
            "it: the compilation boundary lives in the writers, not the "
            "workspace." if reverse and dec_in > 2 * max(dec_post, 1e-9)
            else None
        ),
    }
    print(f"last-layer decode: in={dec_in:.2f} post={dec_post:.2f}")
    print(f"\nP5: in-horizon ratio {in_r:.2f} (>=3 ?), "
          f"post-horizon {post_r:.2f} (<=1.5 ?) -> "
          f"{'PASS' if p5_pass else 'FAIL'}"
          + ("  [REVERSE FINDING: workspace intact past horizon]"
             if reverse else ""))
    if ctrl_note:
        print(f"control: in={ctrl_note['in_horizon_ratio']:.2f} "
              f"post={ctrl_note['post_horizon_ratio']:.2f}")

    # ---- figure -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        for r in results["k5"]:
            axes[0].plot(r["overlap_ratio_best_layer"], "-o", ms=3, alpha=0.7)
        if results["control"]:
            axes[0].plot(results["control"]["overlap_ratio_best_layer"], "-s",
                         ms=3, c="k", label="full-horizon control")
        axes[0].axvline(5, ls="--", c="crimson")
        axes[0].axhline(1.0, ls=":", c="gray")
        axes[0].set_xlabel("source position"); axes[0].set_ylabel("frame ratio-to-null")
        axes[0].set_title("P5: workspace across the horizon"); axes[0].legend(fontsize=7)

        for r in results["k5"]:
            axes[1].plot(r["stability_best_layer"], "-o", ms=3, alpha=0.7)
        if results["control"]:
            axes[1].plot(results["control"]["stability_best_layer"], "-s",
                         ms=3, c="k")
        axes[1].axvline(5, ls="--", c="crimson")
        axes[1].set_xlabel("source position"); axes[1].set_ylabel("disjoint-batch overlap")
        axes[1].set_title("J-space stability")

        for r in results["k5"]:
            axes[2].plot(r["calibration_mae_by_position"], "-o", ms=3, alpha=0.7)
        if results["control"]:
            axes[2].plot(results["control"]["calibration_mae_by_position"], "-s",
                         ms=3, c="k")
        axes[2].axvline(5, ls="--", c="crimson")
        axes[2].set_xlabel("target position"); axes[2].set_ylabel("MAE (bits)")
        axes[2].set_title("calibration (sanity)")
        fig.tight_layout()
        fig.savefig(out / "phase4_horizon.png", dpi=150)
        print(f"figures -> {out/'phase4_horizon.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"figure generation skipped: {exc}")

    with open(out / "phase4_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
