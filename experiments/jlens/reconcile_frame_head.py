#!/usr/bin/env python3
"""
Reconciliation run: is Paper I's "single uniquely indispensable Layer-0
frame head" a property of the task variant?

The jlens Phase-2 result (production sep-vocab checkpoint, no query token)
found indispensability distributed across the full Layer-0 bank. Paper I's
claim was measured on the query-token shared-vocab variant. This script
runs the SAME per-head ablation protocol on both variants and reports the
concentration statistics, so the note can state a reconciliation (variant
changes writer concentration) or a correction.

Ablation: zero the head's head_mask (both architectures carry the buffer).
Metric: entropy-calibration MAE at truncated-context query positions
(Paper I's evaluation), batched.

Usage:
    python experiments/jlens/reconcile_frame_head.py \
        --checkpoints logs/bijection_v256_test/ckpt_final.pt \
                      logs/bijection_v64_highL/ckpt_final.pt \
                      <sepvocab ckpt> \
        --out experiments/jlens/artifacts/reconcile
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

from experiments.jlens.calibration import eval_entropy_mae, head_ablated  # noqa: E402
from experiments.jlens.data_gen import generate_batch  # noqa: E402
from experiments.jlens.models import LegacyTinyGPTWrapper, load_model  # noqa: E402


@torch.no_grad()
def paired_calibration_mae(model, V, L, device, n_seq=64, seed=7,
                           t_stride=1) -> float:
    """Paper I's evaluation, batched: for each context depth t, build n_seq
    sequences [k1,v1,...,k_{t-1},v_{t-1},k_t] (shared vocab, query = fresh
    key) and compare model entropy at the final position to log2(V-t+1)."""
    rng = random.Random(seed)
    total, count = 0.0, 0
    for t in range(1, L + 1, t_stride):
        seqs = []
        for _ in range(n_seq):
            perm = list(range(V))
            rng.shuffle(perm)
            keys = rng.sample(range(V), t)
            seq = []
            for k in keys[:-1]:
                seq.append(k)
                seq.append(perm[k])
            seq.append(keys[-1])
            seqs.append(seq)
        x = torch.tensor(seqs, dtype=torch.long, device=device)
        logits = model.logits(x)[:, -1, :].float()
        p = F.softmax(logits, dim=-1)
        h_model = -(p * (p + 1e-12).log2()).sum(-1)  # (n_seq,)
        h_bayes = float(np.log2(V - t + 1)) if V - t + 1 > 1 else 0.0
        total += float((h_model - h_bayes).abs().mean())
        count += 1
    return total / count


def concentration_stats(delta: dict) -> dict:
    """Concentration of indispensability across heads."""
    l0 = {k: v for k, v in delta.items() if k.startswith("L0")}
    ranked = sorted(delta, key=delta.get, reverse=True)
    top, second = ranked[0], ranked[1]
    n_big = sum(1 for v in delta.values() if v > 0.5 * delta[top])
    return {
        "top_head": top,
        "top_delta": delta[top],
        "second_head": second,
        "second_delta": delta[second],
        "top_over_second": delta[top] / max(delta[second], 1e-9),
        "n_heads_above_half_top": n_big,
        "l0_top": max(l0, key=l0.get),
        "single_head_concentrated": bool(
            delta[top] >= 2.0 * delta[second]
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/jlens/artifacts/reconcile")
    ap.add_argument("--n-seq", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    for ckpt in args.checkpoints:
        model = load_model(ckpt, device=args.device)
        is_paired = isinstance(model, LegacyTinyGPTWrapper)
        V = model.vocab_size if is_paired else model.vocab_size // 2
        # Infer L from the positional embedding length.
        inner = model.inner if is_paired else model
        pos_len = inner.pos_emb.weight.shape[0]
        L = (pos_len - 1) // 2 if is_paired else pos_len // 2
        tag = Path(ckpt).parent.name
        print(f"\n=== {tag}: {'paired/query' if is_paired else 'sepvocab'} "
              f"V={V} L={L} heads={model.n_heads}x{model.n_layers} ===")

        # Larger tasks: stride the context depths to keep this quick.
        t_stride = max(1, L // 20)

        if is_paired:
            def mae_fn():
                return paired_calibration_mae(
                    model, V, L, args.device, n_seq=args.n_seq, t_stride=t_stride
                )
        else:
            batch = generate_batch(256, V=V, L=L, seed=7, fmt="sepvocab")

            def mae_fn():
                return eval_entropy_mae(model, batch)

        baseline = mae_fn()
        delta = {}
        for l in range(model.n_layers):
            for h in range(model.blocks[l].attn.n_heads):
                with head_ablated(model, l, h):
                    delta[f"L{l}H{h}"] = mae_fn() - baseline
        stats = concentration_stats(delta)
        ranked = sorted(delta, key=delta.get, reverse=True)[:8]
        print(f"baseline MAE {baseline:.4f} bits; top-8 dMAE: "
              f"{ {k: round(delta[k], 4) for k in ranked} }")
        print(f"concentration: top {stats['top_head']} = "
              f"{stats['top_over_second']:.2f}x second "
              f"({'SINGLE-HEAD' if stats['single_head_concentrated'] else 'DISTRIBUTED'}); "
              f"{stats['n_heads_above_half_top']} heads above half-top")
        results[tag] = {
            "checkpoint": ckpt,
            "variant": "paired_query" if is_paired else "sepvocab",
            "V": V,
            "L": L,
            "baseline_mae": baseline,
            "delta_mae": delta,
            "concentration": stats,
        }

    with open(out / "reconcile_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
