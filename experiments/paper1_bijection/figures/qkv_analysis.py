#!/usr/bin/env python3
"""
QKV Geometry Analyses for TinyGPT (V=20, L=19)

Subcommands:
- orthogonality: Key orthogonality per layer/head
- qk-align: Query–Key alignment over time (single sequence)
- value-manifold: PCA of final-layer representations per position colored by H(t)
- svd: Singular value spectra of W_Q and W_K per layer
- grad-align: Gradient alignment with principal axes of W_Q/W_K

Outputs are written under --out (default: ablations/qkv).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Prefer the src/ package if available
try:
    from train import TinyGPT
except Exception:
    # Fallback to root path import style if needed
    import sys, os
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from train import TinyGPT  # type: ignore


def load_tinygpt_from_ckpt(ckpt_path: Path, device: torch.device) -> Tuple[TinyGPT, int, int, int, int]:
    state = torch.load(str(ckpt_path), map_location=device)
    model_state = state.get("model", state)

    # Infer architecture
    vocab_size = int(model_state["tok_emb.weight"].shape[0])
    block_size = int(model_state["pos_emb.weight"].shape[0])
    dim = int(model_state["tok_emb.weight"].shape[1])
    # Heuristic for #layers: count attn.qkv
    n_layers = max(
        [int(k.split(".")[1]) for k in model_state.keys() if k.startswith("blocks.") and k.endswith("attn.qkv.weight")]
        or [5]
    ) + 1
    # Heads must be read from one block
    qkv_w = model_state[f"blocks.0.attn.qkv.weight"]
    n_heads = None
    # Try to infer from saved args
    args_in = state.get("args", {})
    if isinstance(args_in, dict):
        n_heads = int(args_in.get("heads", 6))
    else:
        n_heads = int(getattr(args_in, "heads", 6))
    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads,
        dim=dim,
        dropout=0.0,
        pointer_lookup=True,
        pointer_mode="replace",
    ).to(device)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    # Derive V and L
    V = vocab_size
    # For BFI-5, L=25 is fixed; for bijection, infer from block_size
    if vocab_size == 36:  # BFI-5
        L = 25
    else:
        L = (block_size - 1) // 2 if block_size % 2 == 1 else block_size // 2
    return model, V, L, n_layers, n_heads


def generate_no_replacement_sequence(V: int, L: int, device: torch.device) -> Tuple[torch.Tensor, List[int]]:
    """Generate bijection-style sequence: [k1, v1, k2, v2, ...]"""
    perm = list(range(V))
    rng = np.random.default_rng(1337)
    rng.shuffle(perm)
    keys = list(range(V))
    rng.shuffle(keys)
    keys = keys[:L]
    seq: List[int] = []
    for i in range(L):
        k = keys[i]
        v = perm[k]
        seq.append(k)
        seq.append(v)
    # Start with first key only; caller will slice prefixes by t
    x_full = torch.tensor([seq], dtype=torch.long, device=device)
    return x_full, keys


def generate_bfi5_sequence(L: int, device: torch.device, seed: int = 1337) -> Tuple[torch.Tensor, List[int]]:
    """Generate BFI-5 sequence: [BOS, IN_1, OUT_1, SEP, ..., IN_L].
    Returns full sequence and list of input indices (for entropy tracking).
    """
    from bayesg.tasks.bfi5 import get_bfi5_vocab, idx_to_in_token_id, out_token_id
    vocab = get_bfi5_vocab()
    rng = np.random.default_rng(seed)
    # Random truth table and input order
    truth = rng.integers(0, 2, size=32, dtype=np.int32)
    order = rng.choice(32, size=L, replace=False)
    tokens: List[int] = [vocab.bos_id]
    input_indices: List[int] = []
    for t in range(1, L):
        in_idx = int(order[t - 1])
        y = int(truth[in_idx])
        tokens.append(idx_to_in_token_id(in_idx, vocab))
        input_indices.append(in_idx)
        tokens.append(out_token_id(y, vocab))
        tokens.append(vocab.sep_id)
    # Final IN_L
    in_idx_L = int(order[L - 1])
    tokens.append(idx_to_in_token_id(in_idx_L, vocab))
    input_indices.append(in_idx_L)
    x_full = torch.tensor([tokens], dtype=torch.long, device=device)
    return x_full, input_indices


def get_block_inputs(model: TinyGPT, x: torch.Tensor) -> List[torch.Tensor]:
    """Return the input tensor to each block's attn (after ln1 input), by running a forward pass."""
    # This mirrors TinyGPT forward
    pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
    h = model.tok_emb(x) + model.pos_emb(pos)
    h = model.drop(h)
    h_in_list: List[torch.Tensor] = []
    for blk in model.blocks:
        h_in = blk.ln1(h)
        h_in_list.append(h_in)
        a, _ = blk.attn(h_in, return_attn=False)
        h = h + a
        h = h + blk.mlp(blk.ln2(h))
    h = model.ln_f(h)
    return h_in_list


def cmd_orthogonality(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, V, L, n_layers, n_heads = load_tinygpt_from_ckpt(Path(args.ckpt), device)
    dim = model.blocks[0].attn.qkv.in_features
    head_dim = dim // n_heads

    tok_ids = torch.arange(V, device=device)
    E = model.tok_emb(tok_ids)  # (V, dim)

    rows = []
    for li, blk in enumerate(model.blocks):
        W = blk.attn.qkv.weight  # (3*dim, dim)
        Wk = W[dim:2*dim, :].view(n_heads, head_dim, dim)  # (H, Hd, dim)
        for hi in range(n_heads):
            K = E @ Wk[hi].T  # (V, Hd)
            K = F.normalize(K, dim=1)
            C = (K @ K.T).detach().cpu().numpy()
            offdiag = C[~np.eye(V, dtype=bool)]
            rows.append({
                "layer": li,
                "head": hi,
                "offdiag_mean": float(offdiag.mean()),
                "offdiag_std": float(offdiag.std()),
            })
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / "key_orthogonality_stats.csv", index=False)

    # Simple aggregate heatmap: mean over heads per layer
    agg = {}
    for r in rows:
        agg.setdefault(r["layer"], []).append(r["offdiag_mean"])
    layers = sorted(agg.keys())
    means = [np.mean(agg[l]) for l in layers]
    plt.figure(figsize=(6, 3))
    plt.bar(layers, means, edgecolor='black', alpha=0.7)
    plt.xlabel("Layer"); plt.ylabel("Mean off-diagonal cos(K,K)")
    plt.title("Key Orthogonality (lower is better)")
    plt.tight_layout()
    plt.savefig(out_dir / "key_orthogonality_bar.png", dpi=200)
    plt.close()


def cmd_qk_align(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, V, L, n_layers, n_heads = load_tinygpt_from_ckpt(Path(args.ckpt), device)
    dim = model.blocks[0].attn.qkv.in_features
    head_dim = dim // n_heads

    # Detect task type: BFI-5 has vocab_size=36, bijection has V=20 or similar
    is_bfi5 = (V == 36)
    if is_bfi5:
        x_full, input_indices = generate_bfi5_sequence(L, device)
    else:
        x_full, input_indices = generate_no_replacement_sequence(V, L, device)
        # For bijection, input_indices are the keys

    # Pre-allocate per-layer alignment matrices of shape (L, L-1)
    align_stack = [np.full((L, L - 1), np.nan, dtype=float) for _ in range(n_layers)]
    for t in range(1, L + 1):
        if is_bfi5:
            # BFI-5: [BOS, IN_1, OUT_1, SEP, ..., IN_t]
            # Length = 1 + 3*(t-1) + 1 = 3*t - 1
            T = 3 * t - 1
        else:
            # Bijection: [k1, v1, ..., k_t]
            T = 2 * (t - 1) + 1
        x_t = x_full[:, :T]
        h_in_list = get_block_inputs(model, x_t)
        for li, blk in enumerate(model.blocks):
            h_in = h_in_list[li]
            qkv = blk.attn.qkv(h_in)  # (B,T,3*dim)
            H, D = n_heads, head_dim
            qkv = qkv.view(1, x_t.size(1), 3, H, D).permute(0, 3, 1, 2, 4)  # (B,H,T,3,D)
            q = qkv[..., 0, :]  # (B,H,T,D)
            k = qkv[..., 1, :]  # (B,H,T,D)
            # Prior IN positions: for BFI-5, these are at positions 1, 4, 7, ... (1 + 3*i)
            # For bijection, these are at even positions 0, 2, 4, ...
            if is_bfi5:
                # Positions: 1, 4, 7, ..., T-4 (skip last IN which is the query)
                prior_positions = [1 + 3*i for i in range(t - 1)]
            else:
                prior_positions = list(range(0, x_t.size(1) - 1, 2))
            if len(prior_positions) == 0:
                continue
            q_last = F.normalize(q[0, :, -1, :], dim=-1)  # (H,D)
            k_prior = F.normalize(k[0, :, prior_positions, :], dim=-1)  # (H, P, D)
            # Cosine per head averaged over heads -> (P,)
            cos = torch.einsum('hd,hpd->hp', q_last, k_prior).mean(dim=0).detach().cpu().numpy()
            # Fill row t-1, columns 0..len(prior_positions)-1
            align_stack[li][t - 1, :len(prior_positions)] = cos
            if t == L:
                png = out_dir / f"qk_align_L{li}.png"
                plt.figure(figsize=(6, 4))
                # Mask NaN for cleaner visualization
                mask = ~np.isnan(align_stack[li])
                if mask.any():
                    vmin, vmax = np.nanmin(align_stack[li]), np.nanmax(align_stack[li])
                    plt.imshow(align_stack[li], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
                    plt.colorbar(label='cos(Q_t, K_prior)')
                    plt.xlabel('Prior IN index'); plt.ylabel('t (position)')
                    plt.title(f'Q–K Alignment (Layer {li})')
                    plt.tight_layout(); plt.savefig(png, dpi=200); plt.close()


def cmd_value_manifold(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, V, L, n_layers, n_heads = load_tinygpt_from_ckpt(Path(args.ckpt), device)

    # Detect task type
    is_bfi5 = (V == 36)
    
    # Use multiple sequences for better manifold visualization
    n_sequences = 100
    all_reps = []
    all_H_bayes = []
    all_positions = []
    
    for seq_idx in range(n_sequences):
        if is_bfi5:
            x_full, input_indices = generate_bfi5_sequence(L, device)
        else:
            x_full, input_indices = generate_no_replacement_sequence(V, L, device)

        reps = []
        H_bayes = []
        seen = set()
        for t in range(1, L + 1):
            if is_bfi5:
                T = 3 * t - 1  # [BOS, IN_1, OUT_1, SEP, ..., IN_t]
            else:
                T = 2 * (t - 1) + 1  # [k1, v1, ..., k_t]
            x_t = x_full[:, :T]
            # Forward to final layer representation
            pos = torch.arange(0, x_t.size(1), device=device).unsqueeze(0)
            h = model.tok_emb(x_t) + model.pos_emb(pos)
            h = model.drop(h)
            for blk in model.blocks:
                a, _ = blk.attn(blk.ln1(h))
                h = h + a
                h = h + blk.mlp(blk.ln2(h))
            h = model.ln_f(h)
            reps.append(h[0, -1, :].detach().cpu().numpy())
            # Bayes entropy: for BFI-5, use H(k) = 1 - k/32 (predictive entropy for random query)
            # for bijection, use log2(V - |seen|)
            if is_bfi5:
                # k = number of examples seen = t
                H_bayes.append(1.0 - (t / 32.0))
            else:
                H_bayes.append(math.log2(V - len(seen)))
            seen.add(input_indices[t-1])
        
        all_reps.extend(reps)
        all_H_bayes.extend(H_bayes)
        all_positions.extend(list(range(1, L + 1)))
    
    X = np.stack(all_reps, axis=0)
    # PCA 2D
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = U[:, :2] * S[:2]
    
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=np.array(all_H_bayes), cmap='viridis', s=30, alpha=0.6)
    plt.colorbar(sc, label='Bayes H(k) (bits)')
    # Add position labels for a few representative points
    if len(all_positions) <= 50:  # Only label if not too many points
        for i, (x, y) in enumerate(coords):
            plt.text(x, y, str(all_positions[i]), fontsize=6, alpha=0.7)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Value Manifold (PCA of final representation)')
    plt.tight_layout()
    plt.savefig(out_dir / 'value_manifold_pca.png', dpi=200)
    plt.close()


def cmd_svd(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, V, L, n_layers, n_heads = load_tinygpt_from_ckpt(Path(args.ckpt), device)
    dim = model.blocks[0].attn.qkv.in_features

    import pandas as pd
    rows = []
    for li, blk in enumerate(model.blocks):
        W = blk.attn.qkv.weight  # (3*dim, dim)
        Wq = W[:dim, :]
        Wk = W[dim:2*dim, :]
        for name, M in [("Wq", Wq), ("Wk", Wk)]:
            # SVD
            S = torch.linalg.svdvals(M).detach().cpu().numpy()
            energy = (S ** 2) / max(1e-9, float((S ** 2).sum()))
            for i, (s, e) in enumerate(zip(S, energy)):
                rows.append({"layer": li, "which": name, "idx": i, "sigma": float(s), "energy": float(e)})
    pd.DataFrame(rows).to_csv(out_dir / 'svd_spectrum.csv', index=False)

    # Plot per-matrix cumulative energy
    for name in ["Wq", "Wk"]:
        plt.figure(figsize=(6, 4))
        for li in range(n_layers):
            S = [r["energy"] for r in rows if r["layer"] == li and r["which"] == name]
            S = np.array(S)
            S = S.reshape(-1)
            S_sorted = np.sort(S)[::-1]
            cum = np.cumsum(S_sorted)
            plt.plot(cum, label=f'L{li}')
        plt.xlabel('Rank'); plt.ylabel('Cumulative energy')
        plt.title(f'SVD Energy Spectrum ({name})')
        plt.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(out_dir / f'svd_spectrum_{name}.png', dpi=200); plt.close()


def build_batch_for_grad(V: int, L: int, device: torch.device, n: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build batch for gradient computation. Supports both bijection and BFI-5."""
    is_bfi5 = (V == 36)
    xs = []
    ys = []
    rng = np.random.default_rng(1337)
    if is_bfi5:
        from bayesg.tasks.bfi5 import get_bfi5_vocab, idx_to_in_token_id, out_token_id
        vocab = get_bfi5_vocab()
        for _ in range(n):
            truth = rng.integers(0, 2, size=32, dtype=np.int32)
            order = rng.choice(32, size=L, replace=False)
            t = rng.integers(1, L + 1)
            seq = [vocab.bos_id]
            for i in range(t - 1):
                in_idx = int(order[i])
                y = int(truth[in_idx])
                seq.append(idx_to_in_token_id(in_idx, vocab))
                seq.append(out_token_id(y, vocab))
                seq.append(vocab.sep_id)
            # Final query
            in_idx_t = int(order[t - 1])
            seq.append(idx_to_in_token_id(in_idx_t, vocab))
            xs.append(seq)
            ys.append(out_token_id(int(truth[in_idx_t]), vocab))
    else:
        # Bijection style
        for _ in range(n):
            perm = list(range(V)); rng.shuffle(perm)
            keys = list(range(V)); rng.shuffle(keys); keys = keys[:L]
            t = rng.integers(1, L + 1)
            seq = []
            for i in range(t - 1):
                k = keys[i]; v = perm[k]
                seq.extend([k, v])
            seq.append(keys[t - 1])
            xs.append(seq)
            ys.append(perm[keys[t - 1]])
    maxT = max(len(s) for s in xs)
    X = torch.full((n, maxT), 0, dtype=torch.long, device=device)
    Y = torch.full((n, maxT), -100, dtype=torch.long, device=device)
    for i, s in enumerate(xs):
        X[i, :len(s)] = torch.tensor(s, device=device)
        Y[i, len(s) - 1] = ys[i]
    return X, Y


def cmd_grad_align(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, V, L, n_layers, n_heads = load_tinygpt_from_ckpt(Path(args.ckpt), device)
    dim = model.blocks[0].attn.qkv.in_features

    X, Y = build_batch_for_grad(V, L, device, n=128)
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    logits, loss = model(X, targets=Y)
    loss.backward()
    model.eval()

    import pandas as pd
    rows = []
    for li, blk in enumerate(model.blocks):
        W = blk.attn.qkv.weight  # (3*dim, dim)
        G = blk.attn.qkv.weight.grad  # (3*dim, dim)
        Wq = W[:dim, :].detach()
        Wk = W[dim:2*dim, :].detach()
        Gq = G[:dim, :].detach(); Gk = G[dim:2*dim, :].detach()
        for name, M, Gm in [("Wq", Wq, Gq), ("Wk", Wk, Gk)]:
            # SVD right singular vectors
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            V = Vh.T  # (dim, dim)
            denom = torch.norm(Gm)
            for k in [1, 2, 4, 8, 16, 32, 64, min(128, V.shape[1])]:
                proj = torch.norm(Gm @ V[:, :k])
                align = float((proj / (denom + 1e-12)).item())
                rows.append({"layer": li, "which": name, "k": k, "alignment": align})
    pd.DataFrame(rows).to_csv(out_dir / 'grad_alignment.csv', index=False)
    # Simple plot
    for name in ["Wq", "Wk"]:
        plt.figure(figsize=(6, 4))
        for li in range(n_layers):
            ks = [r["k"] for r in rows if r["which"] == name and r["layer"] == li]
            vals = [r["alignment"] for r in rows if r["which"] == name and r["layer"] == li]
            plt.plot(ks, vals, label=f'L{li}')
        plt.xlabel('k (top right-singular vectors)'); plt.ylabel('grad alignment')
        plt.title(f'Grad alignment with {name} principal axes')
        plt.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(out_dir / f'grad_alignment_{name}.png', dpi=200); plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, type=str)
    ap.add_argument('--out', required=True, type=str)
    ap.add_argument('--device', default='cuda:0', type=str)
    ap.add_argument('--subcmd', choices=['orthogonality', 'qk-align', 'value-manifold', 'svd', 'grad-align'], required=True)
    args = ap.parse_args()

    sub = args.subcmd
    if sub == 'orthogonality':
        cmd_orthogonality(args)
    elif sub == 'qk-align':
        cmd_qk_align(args)
    elif sub == 'value-manifold':
        cmd_value_manifold(args)
    elif sub == 'svd':
        cmd_svd(args)
    elif sub == 'grad-align':
        cmd_grad_align(args)
    else:
        raise SystemExit(2)


if __name__ == '__main__':
    main()


