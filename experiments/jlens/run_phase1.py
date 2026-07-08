#!/usr/bin/env python3
"""
Phase 1: the identity test (P1).

Does the J-space at each (layer, position) coincide with the hypothesis-frame
subspace? Preregistered thresholds (spec section 2):

  P1a: mean overlap (projection / CKA) against the frame subspace >= 3x the
       matched-dimension random-subspace null, in some layer/position band.
  P1b: top-k J-space directions decode hypothesis identity >= 90%.

Outputs (under --out):
  phase1_summary.json   thresholds, pass/fail, per-layer tables
  overlaps.npz          per-(reduction, r, mode, layer, position) metrics
  probes.npz            per-(layer, position) decode accuracies
  gram.pt               the Gram sweep artifact
  *.png                 overlap + decoding heatmaps

Run the same command with the MLP-control checkpoint for the negative
control; the runner detects the absence of attention and skips the
frame-head ablation (frame modes fall back to embedding directions).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.jlens.calibration import eval_entropy_mae, frame_head_by_ablation  # noqa: E402
from experiments.jlens.data_gen import generate_batch  # noqa: E402
from experiments.jlens.extract import GramSweep, accumulate_grams, capture_names  # noqa: E402
from experiments.jlens.metrics import (  # noqa: E402
    elimination_probes,
    jspace_coordinates,
    linear_cka,
    probe_from_coords,
    projection_overlap,
)
from experiments.jlens.models import MLPControl, load_model  # noqa: E402
from experiments.jlens.subspaces import (  # noqa: E402
    default_hypothesis_tokens,
    frame_subspace,
    identify_frame_head,
    random_subspaces,
)

RANKS = (4, 8, 16)
REDUCTIONS = ("stacked", "summed")
PRIMARY = ("stacked", 8, "key")  # reduction, rank, frame mode for the gate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="experiments/jlens/artifacts/phase1")
    ap.add_argument("--n-seq", type=int, default=512)
    ap.add_argument("--seq-chunk", type=int, default=64)
    ap.add_argument("--V", type=int, default=20)
    ap.add_argument("--L", type=int, default=19)
    ap.add_argument("--min-horizon", type=int, default=1)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--n-nulls", type=int, default=1000)
    ap.add_argument("--probe-rank", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--reuse-gram", action="store_true",
                    help="load gram.pt from --out instead of re-sweeping")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device=args.device)
    # Attention-free models (MLP control, LSTM, Mamba) have no frame head
    # to ablate; the frame comparison falls back to embedding directions.
    is_mlp = not hasattr(model.blocks[0], "attn")
    fmt = "sepvocab" if model.vocab_size == 2 * args.V else "paired"
    hyp_tokens = default_hypothesis_tokens(model, V=args.V)
    batch = generate_batch(args.n_seq, V=args.V, L=args.L, seed=args.seed, fmt=fmt)

    # ---- model sanity + frame head ---------------------------------------
    mae = eval_entropy_mae(model, batch)
    print(f"model: {'MLP control' if is_mlp else 'transformer'}  "
          f"entropy-calibration MAE = {mae:.4f} bits")

    frame_info = {"calibration_mae_bits": mae}
    if is_mlp:
        frame_head = None
        modes = ("embedding",)
    else:
        frame_head, ablation_maes = frame_head_by_ablation(model, batch, layer=0)
        ortho_head, ortho_scores = identify_frame_head(model, hyp_tokens)
        frame_info |= {
            "frame_head_by_ablation": frame_head,
            "ablation_maes": ablation_maes,
            "frame_head_by_orthogonality": ortho_head,
            "orthogonality_scores": ortho_scores,
        }
        print(f"frame head: ablation -> {frame_head} "
              f"(MAE {ablation_maes[frame_head]:.3f} vs baseline "
              f"{ablation_maes[-1]:.3f}); orthogonality -> {ortho_head}")
        modes = ("key", "value", "embedding")

    frames = {
        mode: frame_subspace(model, mode=mode, head=frame_head, hypotheses=hyp_tokens)
        for mode in modes
    }

    # ---- Jacobian sweep ----------------------------------------------------
    gram_path = out / "gram.pt"
    if args.reuse_gram and gram_path.exists():
        sweep = GramSweep.load(gram_path)
        print(f"reusing {gram_path}")
    else:
        t0 = time.perf_counter()
        sweep = accumulate_grams(
            model,
            batch.tokens,
            min_horizon=args.min_horizon,
            seq_chunk=args.seq_chunk,
            cot_dims=hyp_tokens.tolist(),
        )
        print(f"sweep: {args.n_seq} seqs in {time.perf_counter() - t0:.1f}s")
        sweep.save(gram_path)

    names = capture_names(model.n_layers)
    T = sweep.T
    d = model.dim
    key_pos = batch.key_positions()
    val_pos = batch.value_positions()

    # ---- overlaps vs frame with nulls -------------------------------------
    nulls = {r: random_subspaces(d, r, n=args.n_nulls, seed=123) for r in RANKS}
    null_stats = {}  # (r, mode) -> (mean, std) per metric
    for r in RANKS:
        for mode, W in frames.items():
            pv = np.array([projection_overlap(nulls[r][i], W) for i in range(args.n_nulls)])
            cv = np.array([linear_cka(nulls[r][i], W) for i in range(args.n_nulls)])
            null_stats[(r, mode)] = {
                "projection": (float(pv.mean()), float(pv.std())),
                "cka": (float(cv.mean()), float(cv.std())),
            }

    overlaps = {}  # arrays[(reduction, r, mode, metric)] = (n_layers, T)
    for reduction in REDUCTIONS:
        for r in RANKS:
            for mode, W in frames.items():
                proj = np.full((len(names), T), np.nan)
                cka = np.full((len(names), T), np.nan)
                for li, name in enumerate(names):
                    for i in range(T):
                        if sweep.counts[reduction][name][i] == 0:
                            continue
                        U = sweep.top_subspace(reduction, name, i, r)
                        proj[li, i] = projection_overlap(U, W)
                        cka[li, i] = linear_cka(U, W)
                overlaps[f"{reduction}_r{r}_{mode}_projection"] = proj
                overlaps[f"{reduction}_r{r}_{mode}_cka"] = cka

    np.savez(out / "overlaps.npz", **overlaps,
             layer_names=np.array(names), key_positions=key_pos)

    # ---- P1a: ratio to null, per layer over key positions ------------------
    red, r, mode = PRIMARY
    if mode not in frames:
        mode = "embedding"
    p1a_table = {}
    for metric in ("projection", "cka"):
        arr = overlaps[f"{red}_r{r}_{mode}_{metric}"]
        null_mean = null_stats[(r, mode)][metric][0]
        per_layer = np.nanmean(arr[:, key_pos], axis=1)
        p1a_table[metric] = {
            "null_mean": null_mean,
            "per_layer_mean": dict(zip(names, per_layer.round(4).tolist())),
            "per_layer_ratio": dict(
                zip(names, (per_layer / max(null_mean, 1e-12)).round(2).tolist())
            ),
            "best_layer_ratio": float(np.nanmax(per_layer) / max(null_mean, 1e-12)),
        }
        print(f"P1a[{metric}] ratio-to-null by layer: "
              f"{p1a_table[metric]['per_layer_ratio']}")
    p1a_pass = (
        p1a_table["projection"]["best_layer_ratio"] >= 3.0
        and p1a_table["cka"]["best_layer_ratio"] >= 3.0
    )

    # ---- P1b: hypothesis-identity decoding --------------------------------
    # At value positions: decode the hypothesis just revealed.
    # At key positions: decode the upcoming value (Bayes-uncertain early;
    # reported for the curve, the P1b gate uses value positions).
    rp = args.probe_rank
    probe_val = np.full((len(names), T), np.nan)
    probe_key = np.full((len(names), T), np.nan)
    lo = args.V if fmt == "sepvocab" else 0
    for li, name in enumerate(names):
        for i in val_pos:
            if sweep.counts["stacked"][name][i] == 0:
                continue
            U = sweep.top_subspace("stacked", name, int(i), rp)
            coords = jspace_coordinates(model, batch.tokens, U, name, int(i))
            labels = batch.tokens[:, i] - lo
            probe_val[li, i] = probe_from_coords(coords, labels)
        for i in key_pos[:-1]:  # last key position's label is the answer
            if sweep.counts["stacked"][name][i] == 0:
                continue
            U = sweep.top_subspace("stacked", name, int(i), rp)
            coords = jspace_coordinates(model, batch.tokens, U, name, int(i))
            labels = batch.tokens[:, i + 1] - lo
            probe_key[li, i] = probe_from_coords(coords, labels)

    np.savez(out / "probes.npz", probe_val=probe_val, probe_key=probe_key,
             layer_names=np.array(names))
    best_probe_val = float(np.nanmax(np.nanmean(probe_val[:, val_pos], axis=1)))
    p1b_pass = best_probe_val >= 0.90
    print(f"P1b: best-layer mean hypothesis-identity decode "
          f"(value positions) = {best_probe_val:.3f}")

    # ---- P3 preview: elimination probes at the best layer ------------------
    best_layer = names[int(np.nanargmax(np.nanmean(probe_val[:, val_pos], axis=1)))]
    elim_summary = {}
    for i in (key_pos[len(key_pos) // 2], key_pos[-1]):
        U = sweep.top_subspace("stacked", best_layer, int(i), rp)
        coords = jspace_coordinates(model, batch.tokens, U, best_layer, int(i))
        elim_summary[int(i)] = elimination_probes(coords, batch.eliminated[:, i])
    print(f"P3 preview (layer {best_layer}): {elim_summary}")

    # ---- figures ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        arr = overlaps[f"{red}_r{r}_{mode}_projection"]
        im0 = axes[0].imshow(arr, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        axes[0].set_title(f"J-space vs frame({mode}) projection overlap "
                          f"[{red}, r={r}]")
        axes[0].set_xlabel("position"); axes[0].set_ylabel("layer")
        axes[0].set_yticks(range(len(names)), names)
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(probe_val, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title(f"hypothesis-identity decode from top-{rp} J-dirs")
        axes[1].set_xlabel("position"); axes[1].set_ylabel("layer")
        axes[1].set_yticks(range(len(names)), names)
        fig.colorbar(im1, ax=axes[1])
        fig.tight_layout()
        fig.savefig(out / "phase1_heatmaps.png", dpi=150)
        print(f"figures -> {out/'phase1_heatmaps.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"figure generation skipped: {exc}")

    # ---- summary -------------------------------------------------------------
    summary = {
        "config": {k: v for k, v in vars(args).items()},
        "format": fmt,
        "is_mlp_control": is_mlp,
        "frame": frame_info,
        "null_stats": {f"r{r}_{m}": s for (r, m), s in null_stats.items()},
        "P1a": p1a_table | {"pass": bool(p1a_pass)},
        "P1b": {
            "best_layer": best_layer,
            "best_layer_mean_decode_value_positions": best_probe_val,
            "pass": bool(p1b_pass),
        },
        "P3_preview": elim_summary,
        "P1_pass": bool(p1a_pass and p1b_pass),
    }
    with open(out / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nP1 (identity): P1a={'PASS' if p1a_pass else 'FAIL'} "
          f"P1b={'PASS' if p1b_pass else 'FAIL'}")
    return 0 if (p1a_pass and p1b_pass) or is_mlp else 1


if __name__ == "__main__":
    sys.exit(main())
