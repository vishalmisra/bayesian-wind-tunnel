#!/usr/bin/env python3
"""
Phase 9: Mamba state-level evidence swap (the right lever?).

Phase 3's P4a failed on Mamba at every layer, including full-residual
patching of the evidence region: SSM evidence transport is distributed
through per-block recurrent states, not positional residuals. Before
claiming "unpatchable", intervene on the actual state: at read boundary
q, replace every block's scan state h (accumulated through position
q-1) with an evidence-matched donor's, then read the posterior at q.

  redirect  -> the substrate is state-resident and causally patchable
               at the right lever; the workspace concept transfers to
               SSMs as a *state-space* object, not a positional one.
  no effect -> Mamba's evidence use is genuinely non-localizable by
               single-boundary intervention (report as-is).

Conditions: none / state_swap (evidence-matched donor) / state_zero
(destroys accumulated evidence; shows the lever is live).

Known contamination: conv1d (kernel 4) gives position q a local view of
tokens q-3..q from the ORIGINAL sequence; keys agree between donor and
original by construction, values of the immediately preceding pair
differ. Reported, not corrected.

Usage:
    python experiments/jlens/run_phase9_mamba_state.py \
        --checkpoint <mamba ckpt> --out experiments/jlens/artifacts/phase9
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.jlens.data_gen import generate_batch  # noqa: E402
from experiments.jlens.interventions import _bayes_at_position, kl_bits  # noqa: E402
from experiments.jlens.models import load_model  # noqa: E402

READ_POSITIONS = (20, 36)


class ScanController:
    """Shared mutable control block for the monkeypatched scans."""

    def __init__(self, n_blocks: int):
        self.mode = None  # None | "record" | "patch"
        self.record_positions = set()
        self.recorded = [dict() for _ in range(n_blocks)]  # [block][pos] -> h
        self.patch_at = None  # entering position i == patch_at: h <- donor
        self.donor = None  # list[dict] like `recorded`
        self.zero = False


def install_scans(model, controller: ScanController):
    """Replace each MambaBlock's selective_scan with a controllable clone
    of the original algorithm (same clamps, same order of operations)."""
    for b_idx, block in enumerate(model.inner.blocks):

        def scan(u, dt, A, B, C, _block=block, _idx=b_idx):
            B_batch, L, d_inner = u.shape
            h = torch.zeros(B_batch, d_inner, _block.d_state,
                            device=u.device, dtype=u.dtype)
            ys = []
            for i in range(L):
                if controller.mode == "patch" and controller.patch_at == i:
                    if controller.zero:
                        h = torch.zeros_like(h)
                    else:
                        h = controller.donor[_idx][i - 1].to(u.device)
                dA = torch.exp((dt[:, i, :, None] * A).clamp(min=-20, max=0))
                dB_u = (dt[:, i, :, None] * B[:, i, None, :]
                        * u[:, i, :, None]).clamp(-10, 10)
                h = h * dA + dB_u
                h = h.clamp(-100, 100)
                if controller.mode == "record" and i in controller.record_positions:
                    controller.recorded[_idx][i] = h.detach().clone()
                y = (h * C[:, i, None, :]).sum(-1)
                ys.append(y)
            return torch.stack(ys, dim=1)

        block.selective_scan = scan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="experiments/jlens/artifacts/phase9_mamba_state")
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--V", type=int, default=20)
    ap.add_argument("--L", type=int, default=19)
    ap.add_argument("--seed", type=int, default=1212)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device=args.device)
    n_blocks = len(model.inner.blocks)
    controller = ScanController(n_blocks)
    install_scans(model, controller)

    rng = random.Random(args.seed)
    fixed_keys = rng.sample(range(args.V), args.L)
    orig = generate_batch(args.n_pairs, V=args.V, L=args.L,
                          seed=args.seed + 1, fmt="sepvocab",
                          fixed_keys=fixed_keys)
    donor = generate_batch(args.n_pairs, V=args.V, L=args.L,
                           seed=args.seed + 2, fmt="sepvocab",
                           fixed_keys=fixed_keys)
    device = args.device

    # Record donor states at each read boundary (q - 1).
    controller.mode = "record"
    controller.record_positions = {q - 1 for q in READ_POSITIONS}
    with torch.no_grad():
        model.logits(donor.tokens.to(device))
    donor_states = [dict(d) for d in controller.recorded]

    results = {}
    for q in READ_POSITIONS:
        row = {}
        controller.mode = None
        with torch.no_grad():
            base = model.logits(orig.tokens.to(device))[:, q, :].float().cpu()

        controller.mode = "patch"
        controller.patch_at = q
        controller.donor = [
            {q - 1: d[q - 1]} for d in donor_states
        ]
        controller.zero = False
        with torch.no_grad():
            swapped = model.logits(orig.tokens.to(device))[:, q, :].float().cpu()
        controller.zero = True
        with torch.no_grad():
            zeroed = model.logits(orig.tokens.to(device))[:, q, :].float().cpu()
        controller.mode = None

        lo = args.V  # value-token slice
        bayes_orig = _bayes_at_position(orig, q).float()
        bayes_donor = _bayes_at_position(donor, q).float()
        for tag, lg in (("none", base), ("state_swap", swapped),
                        ("state_zero", zeroed)):
            pm = F.softmax(lg[:, lo:lo + args.V], dim=-1)
            row[tag] = {
                "kl_donor_bits": float(kl_bits(bayes_donor, pm).mean()),
                "kl_orig_bits": float(kl_bits(bayes_orig, pm).mean()),
                "redirect_margin_bits": float(
                    (kl_bits(bayes_orig, pm) - kl_bits(bayes_donor, pm)).mean()
                ),
            }
        results[f"q{q}"] = row
        print(f"q={q}: none margin={row['none']['redirect_margin_bits']:+.2f}  "
              f"state_swap={row['state_swap']['redirect_margin_bits']:+.2f}  "
              f"state_zero kl_orig={row['state_zero']['kl_orig_bits']:.2f} "
              f"(baseline kl_orig={row['none']['kl_orig_bits']:.2f})")

    margin = float(np.mean(
        [results[k]["state_swap"]["redirect_margin_bits"] for k in results]
    ))
    results["verdict"] = {
        "mean_state_swap_margin_bits": margin,
        "redirected": bool(margin >= 1.0),
    }
    print(f"\nVERDICT: state-swap margin = {margin:+.2f} bits "
          f"({'REDIRECTED — state is the lever' if margin >= 1.0 else 'no redirect'})")

    with open(out / "phase9_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
