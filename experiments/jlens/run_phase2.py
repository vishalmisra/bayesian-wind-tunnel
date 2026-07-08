#!/usr/bin/env python3
"""
Phase 2: writer attribution (P2).

Preregistered prediction: the Layer-0 hypothesis-frame head is the dominant
writer to the J-space -- its write connectivity exceeds every other head's
by >= 2x, mirroring its unique indispensability under ablation. Cross-check:
Spearman rho between the write-connectivity ranking and the head-ablation
indispensability ranking, expected > 0.7.

Reuses the Phase-1 Gram artifact for the workspace bases (top eigenspace of
the position-pooled Gram per layer, stacked reduction, r=8).

Usage:
    python experiments/jlens/run_phase2.py \
        --checkpoint <ckpt.pt> \
        --gram experiments/jlens/artifacts/phase1/gram.pt \
        --out experiments/jlens/artifacts/phase2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.jlens.calibration import eval_entropy_mae, head_ablated  # noqa: E402
from experiments.jlens.connectivity import (  # noqa: E402
    pooled_subspace,
    read_connectivity_all,
    write_connectivity_all,
)
from experiments.jlens.data_gen import generate_batch  # noqa: E402
from experiments.jlens.extract import GramSweep, capture_names  # noqa: E402
from experiments.jlens.models import load_model  # noqa: E402

RANK = 8
REDUCTION = "stacked"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gram", required=True, help="Phase-1 gram.pt artifact")
    ap.add_argument("--out", default="experiments/jlens/artifacts/phase2")
    ap.add_argument("--n-seq", type=int, default=256)
    ap.add_argument("--V", type=int, default=20)
    ap.add_argument("--L", type=int, default=19)
    ap.add_argument("--seed", type=int, default=888)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device=args.device)
    fmt = "sepvocab" if model.vocab_size == 2 * args.V else "paired"
    batch = generate_batch(args.n_seq, V=args.V, L=args.L, seed=args.seed, fmt=fmt)
    key_pos = batch.key_positions().tolist()

    sweep = GramSweep.load(args.gram)
    names = capture_names(model.n_layers)
    bases = {
        name: pooled_subspace(sweep, REDUCTION, name, key_pos, RANK)
        for name in names
    }

    # ---- ablation indispensability, all heads -----------------------------
    baseline = eval_entropy_mae(model, batch)
    delta_mae = {}
    for l in range(model.n_layers):
        for h in range(model.blocks[l].attn.n_heads):
            with head_ablated(model, l, h):
                delta_mae[f"L{l}H{h}"] = eval_entropy_mae(model, batch) - baseline
    print(f"baseline calibration MAE {baseline:.4f} bits")
    top_abl = sorted(delta_mae, key=delta_mae.get, reverse=True)[:5]
    print("top-5 indispensable heads:",
          {k: round(delta_mae[k], 4) for k in top_abl})

    # ---- write + read connectivity ----------------------------------------
    writes = write_connectivity_all(model, batch.tokens, bases, key_pos)
    reads = read_connectivity_all(model, batch.tokens, bases, key_pos)

    # Frame-basis variant: project writes onto the P1-identified workspace
    # (the hypothesis-token frame) instead of per-landing-layer J-spaces.
    # Rationale (Phase-2 finding): the per-layer ratio metric is dominated
    # by near-dead heads with tiny frame-aligned writes; the absolute
    # frame-projected write separates load-bearing writers cleanly.
    from experiments.jlens.subspaces import default_hypothesis_tokens, frame_subspace

    hyp = default_hypothesis_tokens(model, V=args.V)
    F_basis = frame_subspace(model, mode="embedding", hypotheses=hyp)
    frame_bases = {name: F_basis for name in names}
    writes_frame = write_connectivity_all(model, batch.tokens, frame_bases, key_pos)

    head_names = [k for k in writes if "H" in k and k != "emb"]
    w_sorted = sorted(head_names, key=lambda k: writes[k]["connectivity"],
                      reverse=True)
    print("\nwrite connectivity (top 8):")
    for k in w_sorted[:8]:
        print(f"  {k:7} conn={writes[k]['connectivity']:.3f} "
              f"norm={writes[k]['write_norm']:.2f} "
              f"dMAE={delta_mae.get(k, float('nan')):+.4f} "
              f"read_ratio={reads.get(k, {}).get('ratio', float('nan')):.2f}")

    # ---- P2 gate ------------------------------------------------------------
    frame_head_name = max(
        (k for k in delta_mae if k.startswith("L0")), key=delta_mae.get
    )
    frame_conn = writes[frame_head_name]["connectivity"]
    other_max = max(
        writes[k]["connectivity"] for k in head_names if k != frame_head_name
    )
    p2_ratio = frame_conn / max(other_max, 1e-9)
    p2_pass = p2_ratio >= 2.0

    from scipy.stats import spearmanr

    w_vec = [writes[k]["connectivity"] for k in head_names]
    a_vec = [delta_mae[k] for k in head_names]
    rho, pval = spearmanr(w_vec, a_vec)
    # Absolute frame-projected write (the metric that separates writers).
    wf_vec = [
        writes_frame[k]["connectivity"] * writes_frame[k]["write_norm"]
        for k in head_names
    ]
    rho_f, pval_f = spearmanr(wf_vec, a_vec)

    # Dichotomy check: do frame-writes separate indispensable heads
    # (dMAE > 0.05 bits) from the rest with a clean margin?
    indisp = [k for k in head_names if delta_mae[k] > 0.05]
    disp = [k for k in head_names if delta_mae[k] <= 0.05]
    absw = dict(zip(head_names, wf_vec))
    sep = {
        "indispensable_heads": indisp,
        "min_framewrite_indispensable": min((absw[k] for k in indisp), default=None),
        "max_framewrite_dispensable": max((absw[k] for k in disp), default=None),
    }
    sep["clean_separation"] = bool(
        indisp
        and sep["min_framewrite_indispensable"] > sep["max_framewrite_dispensable"]
    )

    print(f"\nP2: frame head {frame_head_name} conn={frame_conn:.3f}, "
          f"best other={other_max:.3f}, ratio={p2_ratio:.2f} "
          f"({'PASS' if p2_pass else 'FAIL'} vs 2.0)")
    print(f"Spearman(write conn, dMAE) = {rho:.3f} (p={pval:.2g}); "
          f"frame-abs = {rho_f:.3f} (p={pval_f:.2g})")
    print(f"writer/indispensability separation: {sep}")

    # ---- figures --------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        ks = w_sorted
        colors = ["crimson" if k == frame_head_name else "steelblue" for k in ks]
        axes[0].bar(range(len(ks)), [writes[k]["connectivity"] for k in ks],
                    color=colors)
        axes[0].set_xticks(range(len(ks)), ks, rotation=90, fontsize=6)
        axes[0].set_ylabel("write connectivity ||P_J w|| / ||w||")
        axes[0].set_title(f"Writers to the workspace [{REDUCTION}, r={RANK}]")
        axes[1].scatter(w_vec, a_vec, s=18)
        axes[1].scatter([frame_conn], [delta_mae[frame_head_name]],
                        color="crimson", s=40, label=frame_head_name)
        axes[1].set_xlabel("write connectivity")
        axes[1].set_ylabel("ablation ΔMAE (bits)")
        axes[1].set_title(f"write vs indispensability (ρ={rho:.2f})")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(out / "phase2_writers.png", dpi=150)
        print(f"figures -> {out/'phase2_writers.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"figure generation skipped: {exc}")

    summary = {
        "config": vars(args),
        "baseline_mae_bits": baseline,
        "frame_head": frame_head_name,
        "delta_mae": delta_mae,
        "writes": writes,
        "writes_frame_basis": writes_frame,
        "reads": reads,
        "P2": {
            "frame_head_connectivity": frame_conn,
            "best_other_connectivity": other_max,
            "ratio": p2_ratio,
            "pass": bool(p2_pass),
            "spearman_write_vs_ablation": {"rho": float(rho), "p": float(pval)},
            "spearman_frame_abs": {"rho": float(rho_f), "p": float(pval_f)},
            "writer_separation": sep,
        },
    }
    with open(out / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
