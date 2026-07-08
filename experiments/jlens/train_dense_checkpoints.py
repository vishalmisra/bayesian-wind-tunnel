#!/usr/bin/env python3
"""
Retrain the bijection transformer with DENSE EARLY CHECKPOINTS so the
Phase-5 formation curve is not left-censored (the frame is already formed
by the production run's first checkpoint at step 10k).

Same architecture, data distribution and hyperparameters as
logs/bijection_v20_repl (SepVocabTinyGPT, sep-vocab sequences, keys
without replacement, AdamW 3e-4, cosine, batch 256). Checkpoint schedule:
every 100 steps to 1k, every 500 to 5k, every 1k to 10k, every 5k to 50k.

Usage:
    python experiments/jlens/train_dense_checkpoints.py \
        --out logs/bijection_v20_dense --max-steps 50000
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.jlens.models import SepVocabTinyGPT  # noqa: E402
from experiments.jlens.train_mlp_control import (  # noqa: E402
    IGNORE,
    eval_entropy_mae,
    sample_batch,
)


def checkpoint_steps(max_steps: int):
    steps = set()
    steps.update(range(100, 1001, 100))
    steps.update(range(1500, 5001, 500))
    steps.update(range(6000, 10001, 1000))
    steps.update(range(15000, max_steps + 1, 5000))
    steps.add(max_steps)
    return steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/bijection_v20_dense")
    ap.add_argument("--V", type=int, default=20)
    ap.add_argument("--L", type=int, default=19)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--untied", action="store_true",
                    help="untie the output head from the token embedding "
                         "(DA-1 control: is the frame alignment an artifact "
                         "of weight tying?)")
    ap.add_argument("--final-only", action="store_true",
                    help="save only the final checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    saves = checkpoint_steps(args.max_steps)

    model = SepVocabTinyGPT(
        vocab_size=2 * args.V,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        max_seq_len=2 * args.L,
    ).to(args.device)
    if args.untied:
        import torch.nn as nn
        model.head = nn.Linear(args.dim, 2 * args.V, bias=False).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SepVocabTinyGPT: {n_params/1e6:.2f}M params; "
          f"{len(saves)} checkpoints scheduled", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(args.max_steps - args.warmup, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    config = {
        "V": args.V, "L": args.L, "dim": args.dim, "n_layers": args.layers,
        "n_heads": args.heads, "batch_size": args.batch_size, "lr": args.lr,
        "max_steps": args.max_steps, "seed": args.seed,
        "format": "sepvocab", "n_params": n_params, "dense_checkpoints": True,
    }
    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        tokens, labels = sample_batch(
            args.batch_size, args.V, args.L, args.device, gen
        )
        logits = model(tokens)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=IGNORE
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step in saves and not args.final_only:
            torch.save(
                {"model": model.state_dict(), "step": step, "config": config},
                out / f"ckpt_step{step}.pt",
            )
        if step % 2000 == 0 or step == 1:
            mae = eval_entropy_mae(model, args.V, args.L, args.device)
            print(f"step {step}  loss {loss.item():.4f}  "
                  f"entropy-MAE {mae:.3f} bits  ({time.time()-t0:.0f}s)",
                  flush=True)

    torch.save(
        {"model": model.state_dict(), "step": args.max_steps, "config": config},
        out / "ckpt_final.pt",
    )
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
