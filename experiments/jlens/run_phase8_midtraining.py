#!/usr/bin/env python3
"""
Phase 8: mid-training causal function (reviewer experiment, round 3).

The formation dynamics (Phase 5) showed the frame crosses the P1
threshold by step ~400 while calibration is still ~30x from converged.
Question: is the early workspace already causally load-bearing?

Per dense checkpoint we measure, on the bijection task with the P4
evidence-matched swap harness (NOT the rescue harness, which is
recurrence-specific):

  * frame-swap redirect margin (bits): patch the frame component of the
    evidence region with an evidence-matched donor's, read the posterior
    downstream (P4a protocol, frame basis recomputed per checkpoint);
  * corruption at block 0: calibration MAE under norm-preserving
    rotation of the frame component vs the frame-orthogonal complement
    (per-checkpoint frame basis).

Predictions (stated before running): swaps already redirect at step
400-600 with smaller margins; frame-side corruption already destroys
calibration (in the bijection model the emb-band frame carries the
injected evidence, per P4); margins grow with training.

Usage:
    python experiments/jlens/run_phase8_midtraining.py \
        --checkpoint-dir logs/bijection_v20_dense \
        --out experiments/jlens/artifacts/phase8_midtraining
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.jlens.calibration import eval_entropy_mae  # noqa: E402
from experiments.jlens.data_gen import generate_batch  # noqa: E402
from experiments.jlens.interventions import (  # noqa: E402
    _PatchHook,
    capture_residuals,
    swap_experiment,
)
from experiments.jlens.models import load_model  # noqa: E402
from experiments.jlens.run_phase7_corruption import rotation_operators  # noqa: E402
from experiments.jlens.subspaces import (  # noqa: E402
    default_hypothesis_tokens,
    frame_subspace,
)

STEPS = (100, 200, 300, 400, 500, 600, 800, 1000, 2000, 3000,
         5000, 10000, 30000, 50000)
READ_POSITIONS = (20, 36)  # key positions t=10, t=18
CORRUPT_LAYER = "0"


@torch.no_grad()
def rotated_mae(model, batch, layer: str, R: torch.Tensor) -> float:
    """Calibration MAE with the residual at `layer` rotated by R at every
    position (norm-preserving corruption)."""
    device = next(model.parameters()).device
    resid = capture_residuals(model, batch.tokens, layer)
    donor = resid @ R.to(device).T

    class _Shim:
        vocab_size = model.vocab_size

        def parameters(self):
            return model.parameters()

        def logits(self, tokens):
            with _PatchHook(model, layer, list(range(batch.T)), donor, None):
                return model.logits(tokens)

    return eval_entropy_mae(_Shim(), batch)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="logs/bijection_v20_dense")
    ap.add_argument("--out", default="experiments/jlens/artifacts/phase8_midtraining")
    ap.add_argument("--n-pairs", type=int, default=128)
    ap.add_argument("--n-seq", type=int, default=256)
    ap.add_argument("--V", type=int, default=20)
    ap.add_argument("--L", type=int, default=19)
    ap.add_argument("--seed", type=int, default=987)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    fixed_keys = rng.sample(range(args.V), args.L)
    orig = generate_batch(args.n_pairs, V=args.V, L=args.L,
                          seed=args.seed + 1, fmt="sepvocab",
                          fixed_keys=fixed_keys)
    donor = generate_batch(args.n_pairs, V=args.V, L=args.L,
                           seed=args.seed + 2, fmt="sepvocab",
                           fixed_keys=fixed_keys)
    cal_batch = generate_batch(args.n_seq, V=args.V, L=args.L,
                               seed=args.seed + 3, fmt="sepvocab")

    rows = []
    for step in STEPS:
        ckpt = Path(args.checkpoint_dir) / f"ckpt_step{step}.pt"
        if not ckpt.exists():
            continue
        model = load_model(str(ckpt), device=args.device)
        hyp = default_hypothesis_tokens(model, V=args.V)
        frame = frame_subspace(model, mode="embedding", hypotheses=hyp)
        ops = rotation_operators(frame, seed=17)

        mae = eval_entropy_mae(model, cal_batch)
        swaps = swap_experiment(model, orig, donor, "emb", frame,
                                READ_POSITIONS)
        margin = float(np.mean(
            [swaps[i]["redirect_margin_bits"] for i in READ_POSITIONS]
        ))
        mae_frame = rotated_mae(model, cal_batch, CORRUPT_LAYER,
                                ops["rotate_frame"])
        mae_comp = rotated_mae(model, cal_batch, CORRUPT_LAYER,
                               ops["rotate_complement"])
        rows.append({
            "step": step,
            "calibration_mae": mae,
            "frame_swap_margin_bits": margin,
            "mae_rotate_frame_block0": mae_frame,
            "mae_rotate_complement_block0": mae_comp,
        })
        print(f"step {step:>6}: MAE={mae:.4f}  swap-margin={margin:+.2f} bits  "
              f"corrupt(frame)={mae_frame:.3f}  corrupt(comp)={mae_comp:.3f}",
              flush=True)

    # ---- figure: causation + formation on one x-axis -----------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [r["step"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
        ax1.semilogx(steps, [r["frame_swap_margin_bits"] for r in rows],
                     "-o", c="steelblue", label="frame-swap redirect margin (bits)")
        ax1.axhline(1.0, ls="--", c="steelblue", lw=0.8)
        ax1.axhline(0.0, ls=":", c="gray", lw=0.8)
        ax1.set_xlabel("training step"); ax1.set_ylabel("redirect margin (bits)")
        ax2 = ax1.twinx()
        ax2.semilogx(steps, [r["mae_rotate_frame_block0"] for r in rows],
                     "-s", c="crimson", label="corrupt frame @ block 0 (MAE)")
        ax2.semilogx(steps, [r["mae_rotate_complement_block0"] for r in rows],
                     "-^", c="darkorange", label="corrupt complement @ block 0 (MAE)")
        ax2.semilogx(steps, [r["calibration_mae"] for r in rows],
                     "-x", c="gray", label="baseline MAE")
        ax2.set_ylabel("calibration MAE (bits)")
        l1, lab1 = ax1.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lab1 + lab2, fontsize=7, loc="center right")
        ax1.set_title("Workspace function across training")
        fig.tight_layout()
        fig.savefig(out / "phase8_midtraining.png", dpi=150)
        print(f"figure -> {out/'phase8_midtraining.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"figure generation skipped: {exc}")

    with open(out / "phase8_summary.json", "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
