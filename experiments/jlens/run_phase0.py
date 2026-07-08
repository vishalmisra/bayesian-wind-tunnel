#!/usr/bin/env python3
"""
Phase 0: infrastructure + G0 stability gate.

Extracts J-spaces from two disjoint batches and checks that the top-r
subspace at every (layer, position) is reproducible: projection overlap
between the two extractions >= the gate threshold (default 0.8, spec
section 5).

If the J-space is not reproducible at this scale, everything downstream is
noise: stop and report.

Usage:
    python experiments/jlens/run_phase0.py \
        --checkpoint logs/bijection_v20_repl/ckpt_final.pt \
        --out experiments/jlens/artifacts/phase0 \
        --n-seq 256 --seq-chunk 64
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

from experiments.jlens.data_gen import generate_batch  # noqa: E402
from experiments.jlens.extract import accumulate_grams, capture_names  # noqa: E402
from experiments.jlens.metrics import subspace_stability  # noqa: E402
from experiments.jlens.models import load_model  # noqa: E402
from experiments.jlens.subspaces import (  # noqa: E402
    default_hypothesis_tokens,
    identify_frame_head,
)

RANKS = (4, 8, 16)
GATE_THRESHOLD = 0.8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="experiments/jlens/artifacts/phase0")
    ap.add_argument("--n-seq", type=int, default=256, help="per disjoint half")
    ap.add_argument("--seq-chunk", type=int, default=64)
    ap.add_argument("--V", type=int, default=20)
    ap.add_argument("--L", type=int, default=19)
    ap.add_argument("--min-horizon", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gate", type=float, default=GATE_THRESHOLD)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device=args.device)
    print(f"model: dim={model.dim} layers={model.n_layers} vocab={model.vocab_size}")
    fmt = "sepvocab" if model.vocab_size == 2 * args.V else "paired"
    hyp_tokens = default_hypothesis_tokens(model, V=args.V)
    frame_head, ortho_scores = identify_frame_head(model, hyp_tokens)
    print(f"format={fmt}  hypothesis-frame head (L0): {frame_head}")
    print(f"key-orthogonality by head: {ortho_scores}")

    sweeps = {}
    for half in (0, 1):
        batch = generate_batch(
            args.n_seq, V=args.V, L=args.L, seed=args.seed + half * 10_000, fmt=fmt
        )
        t0 = time.perf_counter()
        sweep = accumulate_grams(
            model,
            batch.tokens,
            min_horizon=args.min_horizon,
            seq_chunk=args.seq_chunk,
            cot_dims=hyp_tokens.tolist(),
        )
        dt = time.perf_counter() - t0
        print(f"half {half}: {args.n_seq} seqs in {dt:.1f}s")
        sweep.save(out / f"gram_half{half}.pt")
        sweeps[half] = sweep

    names = capture_names(model.n_layers)
    T = sweeps[0].T
    results = {
        "per_cell": [],
        "config": vars(args) | {"frame_head": frame_head, "format": fmt},
    }
    worst = {}
    for reduction in ("stacked", "summed"):
        for r in RANKS:
            overlaps = np.zeros((len(names), T))
            for li, name in enumerate(names):
                # Position T-1 has no strictly-future targets for large
                # min_horizon; positions with zero rows are skipped.
                for i in range(T):
                    if sweeps[0].counts[reduction][name][i] == 0:
                        overlaps[li, i] = np.nan
                        continue
                    U1 = sweeps[0].top_subspace(reduction, name, i, r)
                    U2 = sweeps[1].top_subspace(reduction, name, i, r)
                    overlaps[li, i] = subspace_stability(U1, U2)
            valid = overlaps[~np.isnan(overlaps)]
            key = f"{reduction}_r{r}"
            worst[key] = {
                "min": float(valid.min()),
                "mean": float(valid.mean()),
                "frac_below_gate": float((valid < args.gate).mean()),
            }
            results["per_cell"].append(
                {"reduction": reduction, "r": r, "overlap": overlaps.tolist()}
            )
            print(
                f"{key}: mean={valid.mean():.3f} min={valid.min():.3f} "
                f"frac<{args.gate}={(valid < args.gate).mean():.3f}"
            )

    # Gate: judged on the mean at the paper's primary rank (r=8, stacked);
    # per-cell minima are reported so a localized failure band is visible.
    primary = worst["stacked_r8"]
    passed = primary["mean"] >= args.gate
    results["gate"] = {
        "threshold": args.gate,
        "primary": "stacked_r8_mean",
        "value": primary["mean"],
        "passed": passed,
        "summary": worst,
    }
    with open(out / "phase0_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nG0 {'PASS' if passed else 'FAIL'}: stacked_r8 mean overlap "
          f"{primary['mean']:.3f} vs gate {args.gate}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
