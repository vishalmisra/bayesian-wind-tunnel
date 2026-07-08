#!/usr/bin/env python3
"""
Phase 3: contents and causality (P3 + P4).

P3 (contents = posterior support): per-hypothesis eliminated/surviving
probes on J-space coordinates, BALANCED accuracy >= 95% at the carrying
band; eliminated-hypothesis directions decay in J-projected residual norm.

P4 (causal asymmetry, the frame-precision test):
  a) Swap: patch the J-space component of the evidence region with an
     evidence-matched donor's (same key order, different permutation);
     the model's posterior at the read position should redirect to the
     donor's Bayes posterior. Success: KL margin >= 1 bit over >= 200
     pairs. Controls: random subspace (no redirect expected), full
     residual (ceiling).
  b) Entropy-axis ablation: projecting out the entropy-readout direction
     leaves calibration intact, while projecting out the frame/J subspace
     destroys it (the Paper-III boundary result inside the wind tunnel).

Usage:
    python experiments/jlens/run_phase3.py \
        --checkpoint <ckpt.pt> \
        --gram experiments/jlens/artifacts/phase1/gram.pt \
        --out experiments/jlens/artifacts/phase3
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.jlens.calibration import eval_entropy_mae  # noqa: E402
from experiments.jlens.connectivity import pooled_subspace  # noqa: E402
from experiments.jlens.data_gen import generate_batch  # noqa: E402
from experiments.jlens.extract import GramSweep, capture_names  # noqa: E402
from experiments.jlens.interventions import (  # noqa: E402
    capture_residuals,
    subspace_ablation_mae,
    swap_experiment,
)
from experiments.jlens.metrics import (  # noqa: E402
    elimination_probes,
    hypothesis_projection_curves,
    jspace_coordinates,
)
from experiments.jlens.models import load_model  # noqa: E402
from experiments.jlens.subspaces import (  # noqa: E402
    _normed_token_embeddings,
    default_hypothesis_tokens,
    entropy_axis,
    frame_subspace,
    random_subspaces,
)

REDUCTION = "stacked"
PROBE_RANK = 16
SWAP_LAYERS = ("emb", "0", "1", "2", "3", "4")
READ_T = (5, 10, 15, 18)  # pair-count depths; key position index = 2t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gram", required=True)
    ap.add_argument("--out", default="experiments/jlens/artifacts/phase3")
    ap.add_argument("--n-seq", type=int, default=512, help="P3 probe batch")
    ap.add_argument("--n-pairs", type=int, default=256, help="P4 swap pairs")
    ap.add_argument("--V", type=int, default=20)
    ap.add_argument("--L", type=int, default=19)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device=args.device)
    fmt = "sepvocab" if model.vocab_size == 2 * args.V else "paired"
    sweep = GramSweep.load(args.gram)
    names = capture_names(model.n_layers)
    hyp = default_hypothesis_tokens(model, V=args.V)

    # =====================================================================
    # P3: elimination probes (balanced) over the (layer, key-position) grid.
    # Probe rank is swept: a V-hypothesis frame needs ~V workspace
    # dimensions, so r=16 < V=20 caps decodability by construction; the
    # gate is judged at the best rank and the r-dependence is reported
    # (it is itself evidence for the identity claim).
    # =====================================================================
    batch = generate_batch(args.n_seq, V=args.V, L=args.L, seed=args.seed, fmt=fmt)
    key_pos = batch.key_positions()
    probe_ranks = (16, 24, 32)
    resid_cache = {
        name: capture_residuals(model, batch.tokens, name).cpu().float()
        for name in names
    }
    p3_grid = {}
    for rank in probe_ranks:
        p3_grid[rank] = {}
        for name in names:
            per_pos = {}
            for i in key_pos:
                if sweep.counts[REDUCTION][name][int(i)] == 0:
                    continue
                U = sweep.top_subspace(REDUCTION, name, int(i), rank)
                coords = resid_cache[name][:, int(i), :] @ U
                res = elimination_probes(coords, batch.eliminated[:, int(i)])
                if res["n_probes"] > 0:
                    per_pos[int(i)] = res
            if per_pos:
                p3_grid[rank][name] = {
                    "per_position": per_pos,
                    "mean_balanced_acc": float(
                        np.mean([r["mean_acc"] for r in per_pos.values()])
                    ),
                }
    p3_by_rank_layer = {
        rank: {n: g["mean_balanced_acc"] for n, g in grid.items()}
        for rank, grid in p3_grid.items()
    }
    best_rank, best_p3_layer = max(
        (
            (rank, layer)
            for rank, by_layer in p3_by_rank_layer.items()
            for layer in by_layer
        ),
        key=lambda rl: p3_by_rank_layer[rl[0]][rl[1]],
    )
    p3_best = p3_by_rank_layer[best_rank][best_p3_layer]
    p3_pass = p3_best >= 0.95
    for rank, by_layer in p3_by_rank_layer.items():
        print(f"P3 balanced-acc by layer (r={rank}): "
              f"{ {k: round(v, 3) for k, v in by_layer.items()} }")
    print(f"P3: best (r={best_rank}, layer {best_p3_layer}) = {p3_best:.3f} "
          f"({'PASS' if p3_pass else 'FAIL'} vs 0.95)")

    # Decay curves at the best layer + emb.
    frame_dirs = _normed_token_embeddings(model)[hyp]  # (V, d) un-orthogonalized
    decay = {}
    for name in {"emb", best_p3_layer}:
        jb = {
            int(i): sweep.top_subspace(REDUCTION, name, int(i), best_rank)
            for i in key_pos
            if sweep.counts[REDUCTION][name][int(i)] > 0
        }
        decay[name] = hypothesis_projection_curves(
            resid_cache[name], jb, frame_dirs, batch.eliminated, key_pos
        )

    # =====================================================================
    # P4a: evidence-matched J-space swaps
    # =====================================================================
    rng = random.Random(args.seed + 1)
    fixed_keys = rng.sample(range(args.V), args.L)
    orig = generate_batch(
        args.n_pairs, V=args.V, L=args.L, seed=args.seed + 2,
        fmt=fmt, fixed_keys=fixed_keys,
    )
    donor = generate_batch(
        args.n_pairs, V=args.V, L=args.L, seed=args.seed + 3,
        fmt=fmt, fixed_keys=fixed_keys,
    )
    assert not torch.equal(orig.perms, donor.perms)
    read_positions = [2 * t for t in READ_T]

    frame20 = frame_subspace(model, mode="embedding", hypotheses=hyp)
    rand16 = random_subspaces(model.dim, 16, n=1, seed=71)[0]

    swap_results = {}
    for layer in SWAP_LAYERS:
        bases = {
            "jspace_r8": pooled_subspace(sweep, REDUCTION, layer, key_pos.tolist(), 8),
            "jspace_r16": pooled_subspace(
                sweep, REDUCTION, layer, key_pos.tolist(), 16
            ),
            "frame20": frame20,
            "random16": rand16,
            "full": None,
        }
        swap_results[layer] = {}
        for tag, basis in bases.items():
            swap_results[layer][tag] = swap_experiment(
                model, orig, donor, layer, basis, read_positions
            )

    # Success per spec: margin >= 1 bit. Judge on the best (layer, J-basis).
    def margin(layer, tag):
        return float(
            np.mean(
                [swap_results[layer][tag][i]["redirect_margin_bits"]
                 for i in read_positions]
            )
        )

    p4a_table = {
        layer: {tag: round(margin(layer, tag), 3) for tag in
                ("jspace_r8", "jspace_r16", "frame20", "random16", "full")}
        for layer in SWAP_LAYERS
    }
    best_j_margin = max(
        margin(l, t) for l in SWAP_LAYERS for t in ("jspace_r8", "jspace_r16")
    )
    p4a_pass = best_j_margin >= 1.0
    print("\nP4a swap redirect margins (bits; rows=layer):")
    for layer, row in p4a_table.items():
        print(f"  {layer:4} {row}")
    print(f"P4a: best J-space margin = {best_j_margin:.2f} bits "
          f"({'PASS' if p4a_pass else 'FAIL'} vs 1.0)")

    # =====================================================================
    # P4b: entropy-axis vs frame/J ablation
    # =====================================================================
    ax_batch = generate_batch(
        args.n_seq, V=args.V, L=args.L, seed=args.seed + 4, fmt=fmt
    )
    baseline_mae = eval_entropy_mae(model, ax_batch)
    p4b = {"baseline_mae": baseline_mae, "per_layer": {}}
    gen = torch.Generator().manual_seed(17)
    for layer in SWAP_LAYERS:
        axis, r2 = entropy_axis(model, ax_batch, layer)
        rand_dir = torch.randn(model.dim, 1, generator=gen)
        rand_dir = rand_dir / rand_dir.norm()
        p4b["per_layer"][layer] = {
            "entropy_axis_r2": r2,
            "mae_entropy_axis_ablated": subspace_ablation_mae(
                model, ax_batch, layer, axis.unsqueeze(1)
            ),
            "mae_random_dir_ablated": subspace_ablation_mae(
                model, ax_batch, layer, rand_dir
            ),
            "mae_frame20_ablated": subspace_ablation_mae(
                model, ax_batch, layer, frame20
            ),
            "mae_jspace_r8_ablated": subspace_ablation_mae(
                model, ax_batch, layer,
                pooled_subspace(sweep, REDUCTION, layer, key_pos.tolist(), 8),
            ),
        }
        r = p4b["per_layer"][layer]
        print(f"P4b[{layer}] axis-r2={r2:.3f}  MAE: base={baseline_mae:.4f} "
              f"axis={r['mae_entropy_axis_ablated']:.4f} "
              f"rand={r['mae_random_dir_ablated']:.4f} "
              f"frame20={r['mae_frame20_ablated']:.4f} "
              f"J8={r['mae_jspace_r8_ablated']:.4f}")

    # Gate: at the layer with the best-fitting axis, axis ablation leaves
    # calibration within 0.05 bits of baseline while frame ablation
    # degrades it by >= 10x that.
    best_ax_layer = max(
        p4b["per_layer"], key=lambda l: p4b["per_layer"][l]["entropy_axis_r2"]
    )
    row = p4b["per_layer"][best_ax_layer]
    axis_delta = row["mae_entropy_axis_ablated"] - baseline_mae
    frame_delta = row["mae_frame20_ablated"] - baseline_mae
    p4b_pass = axis_delta < 0.05 and frame_delta > 10 * max(axis_delta, 0.005)
    print(f"P4b: axis dMAE={axis_delta:+.4f}, frame dMAE={frame_delta:+.4f} "
          f"at layer {best_ax_layer} ({'PASS' if p4b_pass else 'FAIL'})")

    # ---- figures ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
        tags = ("jspace_r16", "frame20", "random16", "full")
        xs = np.arange(len(SWAP_LAYERS))
        for k, tag in enumerate(tags):
            axes[0].bar(xs + 0.2 * k, [p4a_table[l][tag] for l in SWAP_LAYERS],
                        width=0.19, label=tag)
        axes[0].axhline(1.0, ls="--", c="k", lw=0.8)
        axes[0].set_xticks(xs + 0.3, SWAP_LAYERS)
        axes[0].set_xlabel("patched layer"); axes[0].set_ylabel("redirect margin (bits)")
        axes[0].set_title("P4a: evidence swap"); axes[0].legend(fontsize=7)

        for name, c in decay.items():
            axes[1].plot(c["position"], c["surviving_mean"], "-o", ms=3,
                         label=f"{name} surviving")
            axes[1].plot(c["position"], c["eliminated_mean"], "-s", ms=3,
                         label=f"{name} eliminated")
        axes[1].set_xlabel("position"); axes[1].set_ylabel("|<P_J h, f_hyp>|")
        axes[1].set_title("P3: eliminated-direction decay"); axes[1].legend(fontsize=7)

        labels = ["axis", "random dir", "frame20", "J r8"]
        vals = [row["mae_entropy_axis_ablated"], row["mae_random_dir_ablated"],
                row["mae_frame20_ablated"], row["mae_jspace_r8_ablated"]]
        axes[2].bar(labels, vals, color=["seagreen", "gray", "crimson", "darkorange"])
        axes[2].axhline(baseline_mae, ls="--", c="k", lw=0.8, label="baseline")
        axes[2].set_ylabel("calibration MAE (bits)")
        axes[2].set_title(f"P4b: ablations at layer {best_ax_layer}")
        axes[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "phase3_causality.png", dpi=150)
        print(f"figures -> {out/'phase3_causality.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"figure generation skipped: {exc}")

    summary = {
        "config": vars(args),
        "P3": {
            "by_rank_and_layer_mean_balanced_acc": p3_by_rank_layer,
            "best_rank": best_rank,
            "best_layer": best_p3_layer,
            "best": p3_best,
            "pass": bool(p3_pass),
            "grid": p3_grid,
            "decay_curves": decay,
        },
        "P4a": {
            "margins_bits": p4a_table,
            "best_jspace_margin_bits": best_j_margin,
            "pass": bool(p4a_pass),
            "detail": swap_results,
        },
        "P4b": p4b | {
            "best_axis_layer": best_ax_layer,
            "axis_delta_bits": axis_delta,
            "frame_delta_bits": frame_delta,
            "pass": bool(p4b_pass),
        },
    }
    with open(out / "phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nPhase 3: P3={'PASS' if p3_pass else 'FAIL'} "
          f"P4a={'PASS' if p4a_pass else 'FAIL'} "
          f"P4b={'PASS' if p4b_pass else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
