#!/usr/bin/env python3
"""
Substrate-first training pilots (see SUBSTRATE-FIRST-TRAINING.md).

One trainer, four interventions, two tasks:

  --task bijection        accumulation task (keys w/o replacement)
  --task binding          keys WITH replacement, random consistent map:
                          unseen key -> uniform (log2 V bits), seen key
                          -> recall (0 bits). Different computation,
                          same vocabulary and dimensions.

  --donor CKPT --transplant {frozen,trainable}
                          Pilot A: initialize tok_emb, pos_emb, and all
                          wq/wk from the donor. Note: with tied
                          embeddings, freezing tok_emb also freezes the
                          output head (the substrate includes the
                          readout coordinates by construction).
  --freeze-at N           Pilot B: freeze the same substrate parameters
                          at step N (train them from scratch first).
  --sparse-after N --sparse-density D
                          Pilot C: after step N, keep the loss at each
                          supervised position with probability D.

Writes metrics.jsonl ({"step", "mae"} every eval) for
steps-to-calibration analysis.
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
    sample_batch as sample_bijection,
)

SUBSTRATE_PREFIXES = ("tok_emb.", "pos_emb.")
SUBSTRATE_SUFFIXES = (".attn.wq.weight", ".attn.wk.weight")


def is_substrate(name: str) -> bool:
    return name.startswith(SUBSTRATE_PREFIXES) or name.endswith(SUBSTRATE_SUFFIXES)


def sample_binding(B, V, L, device, generator):
    """Keys WITH replacement; per-sequence random key->value map (values
    can repeat). Supervision at key positions with the upcoming value."""
    keymap = torch.randint(0, V, (B, V), device=device, generator=generator)
    keys = torch.randint(0, V, (B, L), device=device, generator=generator)
    values = torch.gather(keymap, 1, keys) + V

    seq = torch.zeros(B, 2 * L, dtype=torch.long, device=device)
    seq[:, 0::2] = keys
    seq[:, 1::2] = values
    tokens = seq[:, : 2 * L - 1]
    labels = torch.full_like(tokens, IGNORE)
    labels[:, 0::2] = values
    return tokens, labels


@torch.no_grad()
def eval_mae(model, task, V, L, device, n=512, seed=7):
    gen = torch.Generator(device=device).manual_seed(seed)
    if task == "bijection":
        tokens, _ = sample_bijection(n, V, L, device, gen)
        key_pos = torch.arange(0, 2 * L - 1, 2, device=device)
        t = torch.arange(L, device=device)
        h_bayes = torch.log2((V - t).float())[None].expand(n, -1)
    else:
        tokens, _ = sample_binding(n, V, L, device, gen)
        key_pos = torch.arange(0, 2 * L - 1, 2, device=device)
        keys = tokens[:, key_pos]
        seen = torch.zeros(n, L, device=device, dtype=torch.bool)
        for t_i in range(1, L):
            seen[:, t_i] = (keys[:, :t_i] == keys[:, t_i : t_i + 1]).any(1)
        h_bayes = torch.where(seen, 0.0, math.log2(V))
    logits = model(tokens)
    lv = logits[:, key_pos, V : 2 * V].float()
    p = F.softmax(lv, dim=-1)
    h_model = -(p * (p + 1e-12).log2()).sum(-1)
    return float((h_model - h_bayes).abs().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", choices=("bijection", "binding"),
                    default="bijection")
    ap.add_argument("--donor", default=None)
    ap.add_argument("--transplant", choices=("frozen", "trainable"),
                    default=None)
    ap.add_argument("--transplant-set", choices=("full", "routing-only"),
                    default="full",
                    help="routing-only transplants pos_emb + wq/wk but NOT "
                         "tok_emb (and hence not the tied head), isolating "
                         "whether frozen cross-task failure is the tied "
                         "readout")
    ap.add_argument("--freeze-at", type=int, default=None)
    ap.add_argument("--sparse-after", type=int, default=None)
    ap.add_argument("--sparse-density", type=float, default=0.05)
    ap.add_argument("--V", type=int, default=20)
    ap.add_argument("--L", type=int, default=19)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    model = SepVocabTinyGPT(
        vocab_size=2 * args.V, dim=192, n_layers=6, n_heads=6,
        max_seq_len=2 * args.L,
    ).to(args.device)

    if args.donor:
        ck = torch.load(args.donor, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        sub = {k: v for k, v in sd.items() if is_substrate(k)}
        if args.transplant_set == "routing-only":
            sub = {k: v for k, v in sub.items()
                   if not k.startswith("tok_emb.")}
        missing, unexpected = model.load_state_dict(sub, strict=False)
        assert not unexpected, unexpected
        print(f"transplanted {len(sub)} substrate tensors from donor",
              flush=True)
        if args.transplant == "frozen":
            for name, p in model.named_parameters():
                if is_substrate(name) and name in sub:
                    p.requires_grad_(False)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.1,
    )
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    sample = sample_bijection if args.task == "bijection" else sample_binding

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(args.max_steps - args.warmup, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    metrics = open(out / "metrics.jsonl", "w")
    config = {k: v for k, v in vars(args).items() if k != "device"}
    json.dump(config, open(out / "config.json", "w"), indent=2)

    frozen_late = False
    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        if args.freeze_at and step == args.freeze_at and not frozen_late:
            for name, p in model.named_parameters():
                if is_substrate(name):
                    p.requires_grad_(False)
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr_at(step), weight_decay=0.1,
            )
            frozen_late = True
            print(f"substrate frozen at step {step}", flush=True)

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        tokens, labels = sample(args.batch_size, args.V, args.L,
                                args.device, gen)
        if args.sparse_after and step > args.sparse_after:
            keep = torch.rand(labels.shape, device=labels.device,
                              generator=gen) < args.sparse_density
            labels = torch.where(keep, labels, torch.full_like(labels, IGNORE))
        logits = model(tokens)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1),
            ignore_index=IGNORE,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == 1:
            model.eval()
            mae = eval_mae(model, args.task, args.V, args.L, args.device)
            model.train()
            metrics.write(json.dumps({"step": step, "mae": mae}) + "\n")
            metrics.flush()
            if step % 2000 == 0 or step == 1:
                print(f"step {step}  mae {mae:.4f}  ({time.time()-t0:.0f}s)",
                      flush=True)

    torch.save({"model": model.state_dict(), "step": args.max_steps,
                "config": config}, out / "ckpt_final.pt")
    metrics.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
