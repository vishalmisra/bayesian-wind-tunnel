#!/usr/bin/env python3
"""
Phase 6: horizon rescue by writer patching (reviewer experiment).

P5 showed the workspace geometry survives past the K=5 loss horizon while
the computation collapses. The causal localization: if the boundary lives
in the WRITERS, transplanting in-horizon writer output into a post-horizon
position should (partially) rescue the prediction there; if the READERS
are also positionally compiled, rescue will be limited. A random-subspace
transplant of the same content is the "wrong workspace, right writer"
control (no rescue expected).

State matching: for a modular linear recurrence, the same (a, b, x) state
occurs at different positions of shifted views of one orbit. The donor
sequence is the test sequence's orbit started `shift` steps later, so
donor position p carries exactly the recurrence state of test position
p + shift. We patch the frame-subspace component of the test residual at
post-horizon position q = p + shift with the donor's at in-horizon
position p (positional content stays the test position's own), then read
the model's prediction at q.

Conditions per (patch layer, position pair):
  none        unpatched baseline (post-horizon failure)
  frame       donor frame-component -> test frame-component
  random      donor content through a random rank-matched subspace
  full        entire residual transplanted (includes donor positional info)

Metric: top-1 accuracy of the (deterministic) next token at q, and
model-entropy error vs the Bayes optimum (0 bits at determined positions).

Usage:
    python experiments/jlens/run_phase6_rescue.py \
        --k5-checkpoints <3 seeds> --control-checkpoint <full-horizon> \
        --out experiments/jlens/artifacts/phase6_rescue
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

from experiments.jlens.extract import _ResidualRecorder  # noqa: E402
from experiments.jlens.interventions import _PatchHook  # noqa: E402
from experiments.jlens.models import load_model  # noqa: E402
from experiments.jlens.subspaces import (  # noqa: E402
    default_hypothesis_tokens,
    frame_subspace,
    random_subspaces,
)

SHIFT = 6
PAIRS = ((2, 8), (3, 9), (4, 10))  # (donor in-horizon p, test post-horizon q)
LAYERS = ("emb", "0", "1", "2", "3")


def generate_orbit_pairs(B: int, p: int, seq_len: int, shift: int, seed: int):
    """(test, donor) token batches from shared orbits, generic (x0 != x1)."""
    rng = np.random.default_rng(seed)
    test = np.zeros((B, seq_len), dtype=np.int64)
    donor = np.zeros((B, seq_len), dtype=np.int64)
    n = 0
    while n < B:
        a, b, x0 = rng.integers(0, p, size=3)
        orbit = [int(x0)]
        for _ in range(seq_len + shift):
            orbit.append(int((a * orbit[-1] + b) % p))
        if orbit[0] == orbit[1]:  # degenerate fixed prefix
            continue
        if orbit[shift] == orbit[shift + 1]:  # donor prefix must be generic too
            continue
        test[n] = orbit[:seq_len]
        donor[n] = orbit[shift : shift + seq_len]
        n += 1
    return torch.from_numpy(test), torch.from_numpy(donor)


@torch.no_grad()
def capture_all(model, tokens):
    device = next(model.parameters()).device
    with _ResidualRecorder(model) as rec:
        model.logits(tokens.to(device))
    return {k: v.detach() for k, v in rec.activations.items()}


@torch.no_grad()
def read_at(model, tokens, q, layer=None, positions=None, donor=None, basis=None):
    """Model prediction stats at position q, optionally patched."""
    device = next(model.parameters()).device
    if layer is None:
        logits = model.logits(tokens.to(device))
    else:
        with _PatchHook(model, layer, positions, donor, basis):
            logits = model.logits(tokens.to(device))
    lg = logits[:, q, :].float()
    pred = lg.argmax(-1).cpu()
    pm = F.softmax(lg, dim=-1)
    ent = -(pm * (pm + 1e-12).log2()).sum(-1).cpu()  # Bayes optimum: 0 bits
    return pred, ent


def analyze(tag, ckpt, args):
    model = load_model(ckpt, device=args.device)
    hyp = default_hypothesis_tokens(model)
    frame = frame_subspace(model, mode="embedding", hypotheses=hyp)
    rand = random_subspaces(model.dim, frame.shape[1], n=1, seed=99)[0]
    test, donor = generate_orbit_pairs(
        args.n_seq, model.vocab_size, args.seq_len, SHIFT, args.seed
    )
    device = next(model.parameters()).device
    donor_resid = {k: v for k, v in capture_all(model, donor).items()}

    results = {}
    for p_in, q in PAIRS:
        truth = test[:, q + 1]
        base_pred, base_ent = read_at(model, test, q)
        row = {
            "none": {
                "acc": float((base_pred == truth).float().mean()),
                "entropy_bits": float(base_ent.mean()),
            }
        }
        for layer in LAYERS:
            dr = donor_resid[layer]
            aligned = torch.zeros(
                test.shape[0], args.seq_len, model.dim, device=device
            )
            aligned[:, q, :] = dr[:, p_in, :]
            for cond, basis in (("frame", frame), ("random", rand), ("full", None)):
                pred, ent = read_at(
                    model, test, q, layer=layer, positions=[q],
                    donor=aligned, basis=basis,
                )
                row[f"{cond}@{layer}"] = {
                    "acc": float((pred == truth).float().mean()),
                    "entropy_bits": float(ent.mean()),
                }
        results[f"p{p_in}->q{q}"] = row

    # Console summary: best rescue layer per condition, averaged over pairs.
    def mean_acc(cond, layer):
        return float(np.mean([results[k][f"{cond}@{layer}"]["acc"] for k in results]))

    base = float(np.mean([results[k]["none"]["acc"] for k in results]))
    print(f"\n=== {tag} ({Path(ckpt).parent.name}) ===")
    print(f"unpatched post-horizon acc: {base:.3f} (chance {1/model.vocab_size:.3f})")
    for cond in ("frame", "random", "full"):
        accs = {layer: round(mean_acc(cond, layer), 3) for layer in LAYERS}
        best = max(accs.values())
        print(f"{cond:7} acc by patch layer: {accs}  best={best:.3f}")
    return {"checkpoint": ckpt, "baseline_acc": base, "detail": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k5-checkpoints", nargs="+", required=True)
    ap.add_argument("--control-checkpoint", default=None)
    ap.add_argument("--out", default="experiments/jlens/artifacts/phase6_rescue")
    ap.add_argument("--n-seq", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=321)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"k5": [analyze("K5", c, args) for c in args.k5_checkpoints]}
    if args.control_checkpoint:
        results["control"] = analyze("CTRL", args.control_checkpoint, args)

    # Verdict: rescue margin of the frame patch over baseline and over the
    # random-subspace control, at the best layer, averaged over seeds.
    def best_frame(entry):
        accs = []
        for layer in LAYERS:
            accs.append(
                np.mean(
                    [entry["detail"][k][f"frame@{layer}"]["acc"]
                     for k in entry["detail"]]
                )
            )
        return float(max(accs))

    def best_cond(entry, cond):
        return float(
            max(
                np.mean(
                    [entry["detail"][k][f"{cond}@{layer}"]["acc"]
                     for k in entry["detail"]]
                )
                for layer in LAYERS
            )
        )

    frame_acc = float(np.mean([best_frame(e) for e in results["k5"]]))
    rand_acc = float(np.mean([best_cond(e, "random") for e in results["k5"]]))
    base_acc = float(np.mean([e["baseline_acc"] for e in results["k5"]]))
    results["verdict"] = {
        "k5_baseline_acc": base_acc,
        "k5_frame_rescue_acc": frame_acc,
        "k5_random_control_acc": rand_acc,
        "rescued": bool(frame_acc > 2 * base_acc and frame_acc > 2 * rand_acc),
    }
    print(f"\nVERDICT: baseline {base_acc:.3f} -> frame rescue {frame_acc:.3f} "
          f"(random control {rand_acc:.3f})")

    with open(out / "phase6_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
