#!/usr/bin/env python3
"""
Train additional LSTM seeds for the cross-architecture comparison.

The repo's train_lstm.py has drifted against train.py's make_batch
signature, so seed-robustness runs use the jlens stack instead: LSTMLens
(state-dict compatible with the original LSTMBijection) on the same
sep-vocab bijection data as every other model in the study.

Usage:
    python experiments/jlens/train_lstm_seeds.py \
        --out logs/lstm_bijection_seed1234 --seed 1234
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

from experiments.jlens.models import LSTMLens  # noqa: E402
from experiments.jlens.train_mlp_control import (  # noqa: E402
    IGNORE,
    eval_entropy_mae,
    sample_batch,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--V", type=int, default=20)
    ap.add_argument("--L", type=int, default=19)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=50000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    model = LSTMLens(
        vocab_size=2 * args.V, dim=args.dim, n_layers=args.layers,
        max_seq_len=2 * args.L,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LSTMLens: {n_params/1e6:.2f}M params seed={args.seed}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(args.max_steps - args.warmup, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    config = vars(args) | {"n_params": n_params, "format": "sepvocab",
                           "arch": "lstm"}
    t0 = time.time()
    model.train()
    for step in range(1, args.max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        tokens, labels = sample_batch(
            args.batch_size, args.V, args.L, args.device, gen
        )
        logits = model(tokens)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1),
            ignore_index=IGNORE,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 2000 == 0 or step == 1:
            model.eval()
            mae = eval_entropy_mae(model, args.V, args.L, args.device)
            model.train()
            print(f"step {step}  loss {loss.item():.4f}  "
                  f"entropy-MAE {mae:.4f} bits  ({time.time()-t0:.0f}s)",
                  flush=True)

    model.eval()
    torch.save(
        {"model": model.state_dict(), "step": args.max_steps, "config":
         {k: v for k, v in config.items() if k != "device"}},
        out / "ckpt_final.pt",
    )
    with open(out / "config.json", "w") as f:
        json.dump({k: v for k, v in config.items() if k != "device"}, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
