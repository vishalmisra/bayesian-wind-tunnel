#!/usr/bin/env python3
"""
Paper I bijection wind tunnel: incremental-loss training and hybrid entropy
evaluation. This is the recipe behind the paper's bijection transformer
(2.67M parameters, entropy MAE of about 3e-3 bits).

Training (`train`)
  - Every example is a full sequence [k1, v1, ..., kL, vL] built from a fresh
    random bijection. Cross-entropy is applied at every key position, where
    the model predicts the value that follows (teacher forcing at every y_k).
  - Keys and values share one vocabulary of size V.
  - Adam, constant learning rate, gradient clipping at 1.0.
  - Two stages: 100k steps with keys sampled with replacement, then resume
    and continue to 150k steps with keys sampled without replacement.

Evaluation (`eval`)
  - "Hybrid" entropy. For a key seen for the first time, the entropy of the
    model's output distribution is compared with log2(V - #distinct keys seen).
    For a repeated key, a pointer restricted to earlier occurrences of the
    same key is used, and the Bayes entropy is 0.
  - `--no_replacement` evaluates sequences in which keys never repeat
    (H_Bayes(k) = log2(V - k + 1)).

Usage: see REPRODUCE.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

IGNORE_INDEX = -100


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.register_buffer("head_mask", torch.ones(self.n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        qkv = self.qkv(x).view(B, T, 3, H, D).permute(0, 3, 1, 2, 4)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        # Causal mask, except that the last position attends to everything.
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        causal[-1, :] = False
        att = att.masked_fill(causal, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = torch.matmul(att, v) * self.head_mask.view(1, H, 1, 1)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, dropout)
        self.ln2 = RMSNorm(dim)
        self.mlp = MLP(dim, 4.0, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """Compact GPT with tied input/output embeddings."""

    def __init__(self, vocab_size: int, block_size: int, n_layers: int = 6,
                 n_heads: int = 6, dim: int = 192, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        logits = self.head(self.hidden(idx))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=IGNORE_INDEX)
        return logits, loss


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def sample_perm(V: int) -> List[int]:
    arr = list(range(V))
    random.shuffle(arr)
    return arr


class IncrementalLossDataset(Dataset):
    """Full [k1, v1, ..., kL, vL] sequences from a fresh bijection per item.
    The target at each key position is the value that follows it."""

    def __init__(self, V: int, L: int, n_samples: int, with_replacement: bool):
        self.V, self.L, self.n = V, L, n_samples
        self.with_replacement = with_replacement

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        V, L = self.V, self.L
        perm = sample_perm(V)
        if self.with_replacement:
            keys = [random.randrange(V) for _ in range(L)]
        else:
            keys = random.sample(range(V), min(L, V))
        seq: List[int] = []
        for k in keys:
            seq += [k, perm[k]]
        x = torch.tensor(seq, dtype=torch.long)
        y = torch.full((2 * L,), IGNORE_INDEX, dtype=torch.long)
        for i, k in enumerate(keys):
            y[2 * i] = perm[k]
        return x, y


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = IncrementalLossDataset(args.V, args.L, args.epoch_size, args.with_replacement)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers)

    model = TinyGPT(args.V, 2 * args.L + 1, args.layers, args.heads, args.dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt.get("step", 0)
        print(f"Resumed from {args.resume_from} at step {step}")

    def save(path: Path, loss: float) -> None:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "loss": loss,
                    "args": {"V": args.V, "L": args.L, "layers": args.layers,
                             "heads": args.heads, "dim": args.dim}}, str(path))

    model.train()
    last_loss = float("nan")
    while step < args.max_steps:
        for x, y in loader:
            if step >= args.max_steps:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_loss = loss.item()
            step += 1
            if step % args.save_every == 0:
                print(f"step {step:>7}  loss {last_loss:.4f}", flush=True)
                save(out / f"ckpt_step{step}.pt", last_loss)
    save(out / "ckpt_final.pt", last_loss)
    print(json.dumps({"final_step": step, "final_loss": last_loss,
                      "checkpoint": str(out / "ckpt_final.pt")}))


# ----------------------------------------------------------------------------
# Hybrid entropy evaluation
# ----------------------------------------------------------------------------
def entropy_bits(logits: torch.Tensor) -> float:
    lp = logits.log_softmax(dim=-1)
    return float(-(lp.exp() * lp).sum().item() / math.log(2.0))


@torch.no_grad()
def evaluate_hybrid(ckpt_path: str, V: int, L: int, n_sequences: int = 2000,
                    device: str = "cpu", with_replacement: bool = True,
                    seed: int = 1337) -> Dict:
    dev = torch.device(device)
    state = torch.load(ckpt_path, map_location=dev)
    a = state.get("args", {})
    model = TinyGPT(V, 2 * L + 1, a.get("layers", 6), a.get("heads", 6), a.get("dim", 192)).to(dev)
    model.load_state_dict(state["model"])
    model.eval()

    h_model = torch.zeros(L, dtype=torch.float64)
    h_bayes = torch.zeros(L, dtype=torch.float64)
    random.seed(seed)

    for _ in range(n_sequences):
        perm = sample_perm(V)
        if with_replacement:
            xs = [random.randrange(V) for _ in range(L)]
        else:
            xs = random.sample(range(V), min(L, V))
        seen: List[int] = []
        for t in range(1, L + 1):
            seq: List[int] = []
            for i in range(t - 1):
                seq += [xs[i], perm[xs[i]]]
            seq.append(xs[t - 1])
            x = torch.tensor([seq], dtype=torch.long, device=dev)
            T = x.size(1)
            h = model.hidden(x)
            base_logits = model.head(h)[0, -1, :]
            x_t = xs[t - 1]

            if x_t in seen:
                # Pointer restricted to earlier occurrences of the same key.
                keys_h = h[0, 0:T - 1:2, :]
                vals = x[0, 1:T - 1:2]
                match = (x[0, 0:T - 1:2] == x_t).nonzero(as_tuple=False).squeeze(-1)
                sim = (h[0, -1, :].unsqueeze(0) @ keys_h[match].T) / math.sqrt(h.size(-1))
                w = torch.softmax(sim, dim=-1).squeeze(0)
                pointer_logits = torch.full((V,), -1e9, device=dev)
                pointer_logits.scatter_(0, vals[match], torch.log(w + 1e-12))
                h_model[t - 1] += entropy_bits(pointer_logits)
            else:
                h_model[t - 1] += entropy_bits(base_logits)
                h_bayes[t - 1] += math.log(V - len(set(seen)), 2)
            seen.append(x_t)

    hm = (h_model / n_sequences).tolist()
    hb = (h_bayes / n_sequences).tolist()
    mae = sum(abs(m - b) for m, b in zip(hm, hb)) / L
    return {"MAE_bits": mae, "H_model_bits": hm, "H_bayes_bits": hb}


def cmd_eval(args: argparse.Namespace) -> None:
    res = evaluate_hybrid(args.ckpt, args.V, args.L, args.n, args.device,
                          with_replacement=not args.no_replacement, seed=args.seed)
    if args.out_csv:
        p = Path(args.out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            f.write("t,H_model_bits,H_bayes_bits\n")
            for t, (m, b) in enumerate(zip(res["H_model_bits"], res["H_bayes_bits"]), 1):
                f.write(f"{t},{m},{b}\n")
    print(json.dumps({"MAE_bits": res["MAE_bits"],
                      "keys": "without replacement" if args.no_replacement else "with replacement",
                      "csv": args.out_csv}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--V", type=int, default=20)
    t.add_argument("--L", type=int, default=19)
    t.add_argument("--layers", type=int, default=6)
    t.add_argument("--heads", type=int, default=6)
    t.add_argument("--dim", type=int, default=192)
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--batch_size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--max_steps", type=int, default=100000)
    t.add_argument("--epoch_size", type=int, default=100000,
                   help="sequences per pass of the sampler; every sequence is a fresh bijection")
    t.add_argument("--with_replacement", action="store_true", help="sample keys with replacement")
    t.add_argument("--resume_from", type=str, default="")
    t.add_argument("--save_every", type=int, default=10000)
    t.add_argument("--num_workers", type=int, default=4)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", type=str, default="cuda")
    t.add_argument("--output_dir", type=str, required=True)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--V", type=int, default=20)
    e.add_argument("--L", type=int, default=19)
    e.add_argument("--n", type=int, default=2000, help="number of evaluation sequences")
    e.add_argument("--seed", type=int, default=1337)
    e.add_argument("--device", type=str, default="cuda")
    e.add_argument("--no_replacement", action="store_true", help="evaluate with keys that never repeat")
    e.add_argument("--out_csv", type=str, default="")
    e.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
