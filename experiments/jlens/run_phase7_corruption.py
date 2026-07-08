#!/usr/bin/env python3
"""
Phase 7: writer corruption (the converse of the rescue).

The rescue showed a transplanted writer state restores post-horizon
prediction (readers position-general, writers broken). The converse
closes the causal loop from the other side: corrupt the writer's output
at an IN-horizon position -- where everything works -- and prediction
should collapse, while the workspace geometry (weight-defined) is
untouched by construction.

The corruption is a random orthogonal rotation of the residual at the
target position, in three variants that exploit the frame/complement
split found in the rescue:

  rotate_full        rotate the whole residual
  rotate_complement  rotate only the frame-orthogonal component
                     (where the in-flight computation lives -> collapse)
  rotate_frame       rotate only the frame component
                     (routing coordinates -> survival predicted)

Norms are preserved in every variant (rotations), so trivial
scale-destruction explanations are excluded.

Usage:
    python experiments/jlens/run_phase7_corruption.py \
        --checkpoints <k5 seeds + control> \
        --out experiments/jlens/artifacts/phase7_corruption
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

from experiments.jlens.interventions import _PatchHook, capture_residuals  # noqa: E402
from experiments.jlens.models import load_model  # noqa: E402
from experiments.jlens.run_phase4 import generate_recurrence_batch  # noqa: E402
from experiments.jlens.subspaces import (  # noqa: E402
    default_hypothesis_tokens,
    frame_subspace,
)

POSITIONS = (3, 4)  # in-horizon, recurrence determined
LAYERS = ("1", "2", "3")  # override with --layers


def _orthogonal(n: int, gen: torch.Generator) -> torch.Tensor:
    Q, R = torch.linalg.qr(torch.randn(n, n, generator=gen))
    return Q * torch.sign(torch.diagonal(R))


def rotation_operators(frame: torch.Tensor, seed: int = 5):
    """(R_full, R_complement, R_frame): d x d orthogonal maps that rotate
    the whole space, only the frame-orthogonal complement, or only the
    frame span, respectively."""
    gen = torch.Generator().manual_seed(seed)
    d, k = frame.shape
    # Complete the frame to an orthonormal basis of R^d.
    full, _ = torch.linalg.qr(
        torch.cat([frame, torch.randn(d, d - k, generator=gen)], dim=1)
    )
    Fb, Cb = full[:, :k], full[:, k:]
    R_full = _orthogonal(d, gen)
    R_comp = Fb @ Fb.T + Cb @ _orthogonal(d - k, gen) @ Cb.T
    R_frame = Fb @ _orthogonal(k, gen) @ Fb.T + Cb @ Cb.T
    # Dimension-matched control (reviewer W4): rotate a RANDOM k-dim
    # subspace, so frame-vs-complement asymmetry cannot be explained by
    # subspace dimension alone.
    Rnd, _ = torch.linalg.qr(torch.randn(d, k, generator=gen))
    Cnd, _r = torch.linalg.qr(
        torch.cat([Rnd, torch.randn(d, d - k, generator=gen)], dim=1)
    )
    Rb, Cb2 = Cnd[:, :k], Cnd[:, k:]
    R_randk = Rb @ _orthogonal(k, gen) @ Rb.T + Cb2 @ Cb2.T
    return {"rotate_full": R_full, "rotate_complement": R_comp,
            "rotate_frame": R_frame, "rotate_random_kdim": R_randk}


@torch.no_grad()
def corrupted_read(model, tokens, layer, q, resid, R):
    """Prediction stats at q with resid[:, q] replaced by R resid[:, q]."""
    device = next(model.parameters()).device
    donor = resid.clone()
    donor[:, q, :] = donor[:, q, :] @ R.to(device).T
    with _PatchHook(model, layer, [q], donor, None):
        logits = model.logits(tokens.to(device))
    lg = logits[:, q, :].float()
    pm = F.softmax(lg, dim=-1)
    ent = -(pm * (pm + 1e-12).log2()).sum(-1)
    return lg.argmax(-1).cpu(), float(ent.mean())


def analyze(tag, ckpt, args):
    model = load_model(ckpt, device=args.device)
    hyp = default_hypothesis_tokens(model)
    frame = frame_subspace(model, mode="embedding", hypotheses=hyp)
    ops = rotation_operators(frame, seed=args.seed)

    tokens, _, is_prog = generate_recurrence_batch(
        args.n_seq, model.vocab_size, args.seq_len, 1.0, args.seed
    )
    device = next(model.parameters()).device
    results = {}
    for q in POSITIONS:
        truth = tokens[:, q + 1]
        logits = model.logits(tokens.to(device))[:, q, :].float()
        base_acc = float((logits.argmax(-1).cpu() == truth).float().mean())
        row = {"none": {"acc": base_acc}}
        for layer in LAYERS:
            resid = capture_residuals(model, tokens, layer)
            for cond, R in ops.items():
                pred, ent = corrupted_read(model, tokens, layer, q, resid, R)
                row[f"{cond}@{layer}"] = {
                    "acc": float((pred == truth).float().mean()),
                    "entropy_bits": ent,
                }
        results[f"q{q}"] = row

    def mean_acc(cond, layer):
        return float(np.mean([results[k][f"{cond}@{layer}"]["acc"]
                              for k in results]))

    base = float(np.mean([results[k]["none"]["acc"] for k in results]))
    print(f"\n=== {tag} ({Path(ckpt).parent.name}) ===")
    print(f"in-horizon baseline acc: {base:.3f} "
          f"(chance {1/model.vocab_size:.3f})")
    for cond in ("rotate_full", "rotate_complement", "rotate_frame"):
        accs = {layer: round(mean_acc(cond, layer), 3) for layer in LAYERS}
        print(f"{cond:17} acc by corrupted layer: {accs}")
    return {"checkpoint": ckpt, "baseline_acc": base, "detail": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/jlens/artifacts/phase7_corruption")
    ap.add_argument("--n-seq", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=654)
    ap.add_argument("--layers", nargs="+", default=None,
                    help="override corrupted layers, e.g. --layers 1 2 3 4 5")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    global LAYERS
    if args.layers:
        LAYERS = tuple(args.layers)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"models": [analyze("model", c, args) for c in args.checkpoints]}

    with open(out / "phase7_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
