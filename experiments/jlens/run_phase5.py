#!/usr/bin/env python3
"""
Phase 5 (secondary): workspace formation dynamics across training.

Prediction from the frame-precision dissociation: the J-space's overlap
with the FINAL model's frame rises early and plateaus while calibration is
still improving. Per checkpoint we measure:

  * entropy-calibration MAE (bits),
  * J-space overlap ratio-to-null vs the final checkpoint's frame
    (embedding-mode), at the carrying band (emb + block 0, key positions),
  * J-space overlap with the FINAL checkpoint's J-space (same cells) --
    the workspace-formation trajectory itself,
  * P3-style elimination decode (balanced, r=24) at block 1, mid-sequence.

Usage:
    python experiments/jlens/run_phase5.py \
        --checkpoints logs/bijection_v20_repl/ckpt_step10000.pt ... \
        --final-checkpoint logs/bijection_v20_repl/ckpt_final.pt \
        --out experiments/jlens/artifacts/phase5
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.jlens.calibration import eval_entropy_mae  # noqa: E402
from experiments.jlens.data_gen import generate_batch  # noqa: E402
from experiments.jlens.extract import accumulate_grams  # noqa: E402
from experiments.jlens.interventions import capture_residuals  # noqa: E402
from experiments.jlens.metrics import elimination_probes, projection_overlap  # noqa: E402
from experiments.jlens.models import load_model  # noqa: E402
from experiments.jlens.subspaces import (  # noqa: E402
    default_hypothesis_tokens,
    frame_subspace,
    random_subspaces,
)

RANK = 8
REDUCTION = "stacked"
BAND = ("emb", "0")  # the P1 carrying band


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--final-checkpoint", required=True)
    ap.add_argument("--out", default="experiments/jlens/artifacts/phase5")
    ap.add_argument("--n-seq", type=int, default=256)
    ap.add_argument("--seq-chunk", type=int, default=64)
    ap.add_argument("--V", type=int, default=20)
    ap.add_argument("--L", type=int, default=19)
    ap.add_argument("--seed", type=int, default=444)
    ap.add_argument("--n-nulls", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    final_model = load_model(args.final_checkpoint, device=args.device)
    hyp = default_hypothesis_tokens(final_model, V=args.V)
    fmt = "sepvocab" if final_model.vocab_size == 2 * args.V else "paired"
    batch = generate_batch(args.n_seq, V=args.V, L=args.L, seed=args.seed, fmt=fmt)
    key_pos = batch.key_positions()
    final_frame = frame_subspace(final_model, mode="embedding", hypotheses=hyp)

    nulls = random_subspaces(final_model.dim, RANK, n=args.n_nulls, seed=123)
    null_mean = float(
        np.mean(
            [projection_overlap(nulls[i], final_frame) for i in range(args.n_nulls)]
        )
    )

    def band_subspaces(sweep):
        return {
            (name, int(i)): sweep.top_subspace(REDUCTION, name, int(i), RANK)
            for name in BAND
            for i in key_pos
            if sweep.counts[REDUCTION][name][int(i)] > 0
        }

    final_sweep = accumulate_grams(
        final_model, batch.tokens, seq_chunk=args.seq_chunk, cot_dims=hyp.tolist()
    )
    final_J = band_subspaces(final_sweep)

    def step_of(path: str) -> int:
        m = re.search(r"step(\d+)", path)
        return int(m.group(1)) if m else 10**9  # final sorts last

    rows = []
    for ckpt in sorted(args.checkpoints, key=step_of) + [args.final_checkpoint]:
        model = load_model(ckpt, device=args.device)
        mae = eval_entropy_mae(model, batch)
        sweep = accumulate_grams(
            model, batch.tokens, seq_chunk=args.seq_chunk, cot_dims=hyp.tolist()
        )
        J = band_subspaces(sweep)
        frame_ratio = float(
            np.mean([projection_overlap(U, final_frame) for U in J.values()])
            / null_mean
        )
        j_vs_final = float(
            np.mean(
                [projection_overlap(J[k], final_J[k]) for k in J if k in final_J]
            )
        )
        # Elimination decode at block 1, mid-sequence key position, r=24.
        mid = int(key_pos[len(key_pos) // 2])
        U = sweep.top_subspace(REDUCTION, "1", mid, 24)
        resid = capture_residuals(model, batch.tokens, "1").cpu().float()
        elim = elimination_probes(resid[:, mid, :] @ U, batch.eliminated[:, mid])
        rows.append(
            {
                "checkpoint": ckpt,
                "step": step_of(ckpt),
                "calibration_mae_bits": mae,
                "frame_ratio_to_null": frame_ratio,
                "jspace_overlap_with_final": j_vs_final,
                "elim_balanced_acc_mid": elim["mean_acc"],
            }
        )
        print(f"step {rows[-1]['step']:>7}: MAE={mae:.4f}  "
              f"frame-ratio={frame_ratio:.2f}  J~final={j_vs_final:.3f}  "
              f"elim-acc={elim['mean_acc']:.3f}")

    # ---- figure -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [r["step"] if r["step"] < 10**9 else rows[-2]["step"] + 5000
                 for r in rows]
        fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
        ax1.plot(steps, [r["frame_ratio_to_null"] for r in rows], "-o",
                 c="steelblue", label="frame ratio-to-null (band)")
        ax1.plot(steps, [10 * r["jspace_overlap_with_final"] for r in rows],
                 "-s", c="seagreen", label="J-space ~ final (x10)")
        ax1.plot(steps, [10 * r["elim_balanced_acc_mid"] for r in rows], "-^",
                 c="darkorange", label="elim balanced acc (x10)")
        ax1.set_xlabel("training step"); ax1.set_ylabel("workspace structure")
        ax2 = ax1.twinx()
        ax2.semilogy(steps, [r["calibration_mae_bits"] for r in rows], "-x",
                     c="crimson", label="calibration MAE (bits)")
        ax2.set_ylabel("MAE (bits, log)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
        ax1.set_title("Workspace formation vs calibration")
        fig.tight_layout()
        fig.savefig(out / "phase5_dynamics.png", dpi=150)
        print(f"figures -> {out/'phase5_dynamics.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"figure generation skipped: {exc}")

    summary = {"config": vars(args), "null_mean": null_mean, "trajectory": rows}
    with open(out / "phase5_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
