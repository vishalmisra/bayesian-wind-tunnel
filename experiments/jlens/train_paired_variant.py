#!/usr/bin/env python3
"""
Train the Paper-I task variant (paired shared-vocab sequences with an
explicit query token) to paper-grade calibration, for the frame-head
reconciliation: no well-calibrated checkpoint of this variant survives on
disk (the candidates are undertrained side runs), and the single-head
claim can only be adjudicated on a model that is actually Bayesian.

Settings mirror configs/bijection_v20.yaml: V=20, L=19, dim=192, 6L/6H,
keys without replacement, query from context, all value positions +
query supervised, AdamW 3e-4 cosine, batch 256.

Usage:
    python experiments/jlens/train_paired_variant.py \
        --out logs/bijection_v20_paired --max-steps 50000
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

from src.models.tinygpt import TinyGPT  # noqa: E402

IGNORE = -100


def sample_batch(B, V, L, device, generator):
    """Vectorized paired-format batch: [k1,v1,...,kL,vL,q], shared vocab.

    Supervision (next-token convention: logits at position p are paired
    with the token at p+1): every KEY position predicts its upcoming
    value --- the calibrated-posterior positions --- and the query
    position predicts the recalled answer (Paper I's
    predict_all_values setting).
    """
    perms = torch.argsort(torch.rand(B, V, device=device, generator=generator), dim=1)
    keys = torch.argsort(torch.rand(B, V, device=device, generator=generator), dim=1)[:, :L]
    values = torch.gather(perms, 1, keys)
    qidx = torch.randint(0, L, (B, 1), device=device, generator=generator)
    query = torch.gather(keys, 1, qidx).squeeze(1)
    answer = torch.gather(values, 1, qidx).squeeze(1)

    T = 2 * L + 1
    tokens = torch.zeros(B, T, dtype=torch.long, device=device)
    tokens[:, 0:2*L:2] = keys
    tokens[:, 1:2*L:2] = values
    tokens[:, -1] = query

    labels = torch.full_like(tokens, IGNORE)
    labels[:, 0:2*L:2] = values  # key position 2i predicts v_i (next token)
    labels[:, -1] = answer       # query predicts the recalled value
    return tokens, labels


@torch.no_grad()
def eval_entropy_mae(model, V, L, device, n=256, seed=7):
    """Paper I evaluation: entropy at truncated-context query positions
    vs log2(V - t + 1), averaged over depths t (strided for speed)."""
    import random
    rng = random.Random(seed)
    total, count = 0.0, 0
    for t in range(1, L + 1, 2):
        seqs = []
        for _ in range(n):
            perm = list(range(V))
            rng.shuffle(perm)
            ks = rng.sample(range(V), t)
            seq = []
            for k in ks[:-1]:
                seq.append(k)
                seq.append(perm[k])
            seq.append(ks[-1])
            seqs.append(seq)
        x = torch.tensor(seqs, dtype=torch.long, device=device)
        logits, _ = model(x)
        p = F.softmax(logits[:, -1, :].float(), dim=-1)
        h_model = -(p * (p + 1e-12).log2()).sum(-1)
        h_bayes = math.log2(V - t + 1) if V - t + 1 > 1 else 0.0
        total += float((h_model - h_bayes).abs().mean())
        count += 1
    return total / count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/bijection_v20_paired")
    ap.add_argument("--V", type=int, default=20)
    ap.add_argument("--L", type=int, default=19)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
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

    model = TinyGPT(
        vocab_size=args.V,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        max_seq_len=2 * args.L + 1,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyGPT (paired variant): {n_params/1e6:.2f}M params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(args.max_steps - args.warmup, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    config = {
        "V": args.V, "L": args.L, "dim": args.dim, "n_layers": args.layers,
        "n_heads": args.heads, "format": "paired", "n_params": n_params,
        "max_steps": args.max_steps, "seed": args.seed,
    }
    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        tokens, labels = sample_batch(args.batch_size, args.V, args.L, args.device, gen)
        logits, _ = model(tokens)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=IGNORE
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 2000 == 0 or step == 1:
            mae = eval_entropy_mae(model, args.V, args.L, args.device)
            print(f"step {step}  loss {loss.item():.4f}  "
                  f"entropy-MAE {mae:.4f} bits  ({time.time()-t0:.0f}s)", flush=True)
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
        json.dump(config, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
