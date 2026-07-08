#!/usr/bin/env python3
"""
Referee-requested analyses that need no new training (review round 6):

  W3    P3 probe null: elimination decode from random r-dim subspaces
        vs the J-space, matched rank.
  W7    P4 random-subspace control as a distribution (100 draws), not
        a single draw.
  DA-10 r=20 J-space swap: quantify the gap to the frame/full ceiling.
  DA-7  Anthropic-faithful estimator row: P1 overlap ratios under the
        summed reduction (their cotangent aggregation), already computed
        in every sweep.
  W6    Null quantiles for the P1 overlap (99th percentile, not just the
        mean), for the headline cells.

Usage:
    python experiments/jlens/referee_analyses.py \
        --checkpoint <bijection ckpt> \
        --gram experiments/jlens/artifacts/phase1/gram.pt \
        --out experiments/jlens/artifacts/referee_analyses
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

from experiments.jlens.data_gen import generate_batch  # noqa: E402
from experiments.jlens.extract import GramSweep  # noqa: E402
from experiments.jlens.interventions import capture_residuals, swap_experiment  # noqa: E402
from experiments.jlens.metrics import elimination_probes, projection_overlap  # noqa: E402
from experiments.jlens.models import load_model  # noqa: E402
from experiments.jlens.subspaces import (  # noqa: E402
    default_hypothesis_tokens,
    frame_subspace,
    random_subspaces,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gram", required=True)
    ap.add_argument("--out", default="experiments/jlens/artifacts/referee_analyses")
    ap.add_argument("--V", type=int, default=20)
    ap.add_argument("--L", type=int, default=19)
    ap.add_argument("--n-seq", type=int, default=1024)
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    model = load_model(args.checkpoint, device=args.device)
    sweep = GramSweep.load(args.gram)
    hyp = default_hypothesis_tokens(model, V=args.V)
    batch = generate_batch(args.n_seq, V=args.V, L=args.L, seed=args.seed,
                           fmt="sepvocab")
    key_pos = batch.key_positions()

    # ---- W3: P3 probe null (random subspaces, matched rank) ----------------
    name = "1"
    resid = capture_residuals(model, batch.tokens, name).cpu().float()
    gen = torch.Generator().manual_seed(7)
    w3 = {}
    for r in (24, 32):
        j_accs, null_accs = [], []
        nulls = random_subspaces(model.dim, r, n=30, seed=11)
        for i in key_pos:
            U = sweep.top_subspace("stacked", name, int(i), r)
            res = elimination_probes(resid[:, int(i), :] @ U,
                                     batch.eliminated[:, int(i)])
            if res["n_probes"]:
                j_accs.append(res["mean_acc"])
        # Null: 30 random subspaces at 5 spread positions (compute budget)
        for ni in range(30):
            U = nulls[ni]
            for i in key_pos[::4]:
                res = elimination_probes(resid[:, int(i), :] @ U,
                                         batch.eliminated[:, int(i)])
                if res["n_probes"]:
                    null_accs.append(res["mean_acc"])
        w3[f"r{r}"] = {
            "jspace_mean": float(np.mean(j_accs)),
            "null_mean": float(np.mean(null_accs)),
            "null_p99": float(np.percentile(null_accs, 99)),
        }
        print(f"W3 r={r}: J-space {w3[f'r{r}']['jspace_mean']:.3f} vs "
              f"random-subspace null {w3[f'r{r}']['null_mean']:.3f} "
              f"(p99 {w3[f'r{r}']['null_p99']:.3f})")
    results["W3_probe_null"] = w3

    # ---- DA-10 + W7: r=20 J swap; random-swap distribution -----------------
    rng = random.Random(args.seed + 1)
    fixed_keys = rng.sample(range(args.V), args.L)
    orig = generate_batch(args.n_pairs, V=args.V, L=args.L,
                          seed=args.seed + 2, fmt="sepvocab",
                          fixed_keys=fixed_keys)
    donor = generate_batch(args.n_pairs, V=args.V, L=args.L,
                           seed=args.seed + 3, fmt="sepvocab",
                           fixed_keys=fixed_keys)
    read_positions = [20, 36]

    from experiments.jlens.connectivity import pooled_subspace

    j20 = pooled_subspace(sweep, "stacked", "emb", key_pos.tolist(), 20)
    swaps = swap_experiment(model, orig, donor, "emb", j20, read_positions)
    da10 = float(np.mean([swaps[i]["redirect_margin_bits"]
                          for i in read_positions]))
    print(f"DA-10: r=20 J-space swap margin = {da10:+.2f} bits "
          f"(r=16 was +2.9; frame20 ceiling +8.2)")
    results["DA10_r20_swap_margin"] = da10

    margins = []
    for k in range(100):
        R = random_subspaces(model.dim, 16, n=1, seed=1000 + k)[0]
        s = swap_experiment(model, orig, donor, "emb", R, [36])
        margins.append(s[36]["redirect_margin_bits"])
    results["W7_random_swap"] = {
        "mean": float(np.mean(margins)),
        "p95": float(np.percentile(margins, 95)),
        "max": float(np.max(margins)),
    }
    print(f"W7: random-16 swap margin over 100 draws: "
          f"mean {np.mean(margins):+.2f}, p95 {np.percentile(margins, 95):+.2f}, "
          f"max {np.max(margins):+.2f}")

    # ---- DA-7: summed-reduction (Anthropic-faithful aggregation) row -------
    frame = frame_subspace(model, mode="embedding", hypotheses=hyp)
    nulls1k = random_subspaces(model.dim, 8, n=1000, seed=123)
    null_vals = np.array([projection_overlap(nulls1k[i], frame)
                          for i in range(1000)])
    da7 = {}
    for reduction in ("stacked", "summed"):
        per_layer = {}
        for nm in sweep.names[:3]:
            ratios = []
            for i in key_pos:
                if sweep.counts[reduction][nm][int(i)] == 0:
                    continue
                U = sweep.top_subspace(reduction, nm, int(i), 8)
                ratios.append(projection_overlap(U, frame) / null_vals.mean())
            per_layer[nm] = round(float(np.mean(ratios)), 2)
        da7[reduction] = per_layer
    results["DA7_reduction_comparison"] = da7
    results["W6_null_quantiles_r8"] = {
        "mean": float(null_vals.mean()),
        "p99": float(np.percentile(null_vals, 99)),
        "p999": float(np.percentile(null_vals, 99.9)),
        "max": float(null_vals.max()),
    }
    print(f"DA-7 reduction comparison (ratio to null mean): {da7}")
    print(f"W6 null r=8: mean {null_vals.mean():.4f}, p99 "
          f"{np.percentile(null_vals, 99):.4f}, max {null_vals.max():.4f} "
          f"-> headline 6.1x mean = "
          f"{6.09 * null_vals.mean() / np.percentile(null_vals, 99):.1f}x the p99")

    with open(out / "referee_analyses.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
