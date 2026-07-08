#!/usr/bin/env python3
"""
Train the MLP negative control (spec section 3) on the sep-vocab bijection
task -- identical data distribution to logs/bijection_v20_repl.

The attention-free per-position MLP cannot do in-context binding, so it
plateaus far above the Bayes bound; the control claim for P1 is that the
J-lens finds no frame-aligned low-rank subspace in it. We still train to
convergence so "no structure" cannot be attributed to an untrained model.

Usage:
    python experiments/jlens/train_mlp_control.py \
        --out logs/mlp_control_v20 --max-steps 50000
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

from experiments.jlens.models import MLPControl  # noqa: E402

IGNORE = -100


def sample_batch(B: int, V: int, L: int, device, generator) -> tuple:
    """Vectorized sep-vocab bijection batch.

    tokens: (B, 2L-1) = [k1, v1+V, ..., k_{L-1}, v_{L-1}+V, kL]
    labels: (B, 2L-1) with the upcoming value token at key positions,
            IGNORE elsewhere (matches train_v256_ddp.py supervision).
    """
    perms = torch.argsort(
        torch.rand(B, V, device=device, generator=generator), dim=1
    )  # (B, V) perms[b, k] = pi(k)
    keys = torch.argsort(
        torch.rand(B, V, device=device, generator=generator), dim=1
    )[:, :L]
    values = torch.gather(perms, 1, keys) + V  # (B, L) value tokens

    seq = torch.zeros(B, 2 * L, dtype=torch.long, device=device)
    seq[:, 0::2] = keys
    seq[:, 1::2] = values
    tokens = seq[:, : 2 * L - 1]

    labels = torch.full_like(tokens, IGNORE)
    labels[:, 0::2] = values  # key position 2t predicts value of pair t
    return tokens, labels


@torch.no_grad()
def eval_entropy_mae(model, V: int, L: int, device, n: int = 512, seed: int = 7) -> float:
    """Mean |H_model - H_bayes| in bits over key positions, value-slice
    softmax (H_bayes at key position 2t is log2(V - t))."""
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens, _ = sample_batch(n, V, L, device, gen)
    logits = model(tokens)
    key_pos = torch.arange(0, 2 * L - 1, 2, device=device)
    lv = logits[:, key_pos, V : 2 * V].float()
    p = F.softmax(lv, dim=-1)
    h_model = -(p * (p + 1e-12).log2()).sum(-1)  # (n, L)
    t = torch.arange(L, device=device)
    h_bayes = torch.log2((V - t).float())[None]
    return float((h_model - h_bayes).abs().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/mlp_control_v20")
    ap.add_argument("--V", type=int, default=20)
    ap.add_argument("--L", type=int, default=19)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=9)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=50000)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    model = MLPControl(
        vocab_size=2 * args.V,
        dim=args.dim,
        n_layers=args.layers,
        max_seq_len=2 * args.L,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLPControl: {n_params/1e6:.2f}M params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(args.max_steps - args.warmup, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    config = vars(args) | {"n_params": n_params, "format": "sepvocab"}
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

        if step % 1000 == 0 or step == 1:
            mae = eval_entropy_mae(model, args.V, args.L, args.device)
            print(
                f"step {step}  loss {loss.item():.4f}  entropy-MAE {mae:.3f} bits  "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.max_steps:
            torch.save(
                {"model": model.state_dict(), "step": step, "config": config},
                out / f"ckpt_step{step}.pt",
            )

    torch.save(
        {"model": model.state_dict(), "step": args.max_steps, "config": config},
        out / "ckpt_final.pt",
    )
    with open(out / "config.json", "w") as f:
        json.dump({k: v for k, v in config.items() if k != "device"}, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
