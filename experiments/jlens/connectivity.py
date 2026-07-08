"""
Read/write connectivity between model components and the J-space (P2).

Write connectivity (spec 4.3): for each component (attention head or MLP),
    write = || P_J . output || / || output ||
computed row-wise over (sequence, position) and averaged; P_J projects onto
the workspace basis at the component's landing layer.

Read connectivity: sensitivity of the component's output to J-space
perturbations of its *input* residual, versus random-direction
perturbations of matched norm (central finite differences). The ratio
J-sensitivity / random-sensitivity mirrors Anthropic's connectivity-ratio
claim at wind-tunnel scale.

The workspace basis per layer is the top eigenspace of the position-pooled
Gram (sum of G_{l,i} over the pooled positions) -- the analogue of their
position-averaged lens.

SepVocabTinyGPT only (the production checkpoint family); the per-head
decomposition reads the module weights directly.
"""

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.jlens.extract import GramSweep, _ResidualRecorder  # noqa: E402


def pooled_subspace(
    sweep: GramSweep,
    reduction: str,
    name: str,
    positions: Sequence[int],
    r: int,
) -> torch.Tensor:
    """Top-r eigenspace of the position-pooled Gram: orthonormal (d, r)."""
    G = sweep.grams[reduction][name][list(positions)].sum(0)
    _, eigvecs = torch.linalg.eigh(G)
    return eigvecs[:, -r:].flip(-1).float()


@torch.no_grad()
def capture_block_inputs(model, tokens: torch.Tensor) -> List[torch.Tensor]:
    """Residual-stream inputs to each block: block_inputs[l] = x entering
    blocks[l], shape (B, T, d), on device. block_inputs[n_layers] is the
    final residual (input to the unembedding norm)."""
    device = next(model.parameters()).device
    with _ResidualRecorder(model) as recorder:
        model.logits(tokens.to(device))
    inputs = [recorder.activations["emb"].detach()]
    for l in range(model.n_layers):
        inputs.append(recorder.activations[str(l)].detach())
    return inputs  # inputs[l] is blocks[l]'s input; inputs[-1] final resid


def _per_head_writes(block, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-head residual writes for one SepVocab block.

    Args:
        block: model.blocks[l].
        x: (B, T, d) block input.

    Returns:
        (head_writes (H, B, T, d), attn_out (B, T, d)).
    """
    attn = block.attn
    H, hd = attn.n_heads, attn.head_dim
    B, T, d = x.shape
    xn = block.norm1(x)
    q = attn.wq(xn).view(B, T, H, hd).transpose(1, 2)
    k = attn.wk(xn).view(B, T, H, hd).transpose(1, 2)
    v = attn.wv(xn).view(B, T, H, hd).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / (hd**0.5)
    causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    ctx = probs @ v  # (B, H, T, hd)

    W_O = attn.wo.weight  # (d, d)
    writes = torch.empty(H, B, T, d, device=x.device)
    for h in range(H):
        sl = slice(h * hd, (h + 1) * hd)
        writes[h] = ctx[:, h] @ W_O[:, sl].T
    return writes, writes.sum(0)


def _row_connectivity(
    out: torch.Tensor, basis: torch.Tensor, positions: Sequence[int]
) -> float:
    """mean over (b, i in positions) of ||P_J out|| / ||out||."""
    rows = out[:, list(positions), :]  # (B, P, d)
    proj = rows @ basis  # (B, P, r)
    num = proj.norm(dim=-1)
    den = rows.norm(dim=-1).clamp(min=1e-9)
    return float((num / den).mean())


@torch.no_grad()
def write_connectivity_all(
    model,
    tokens: torch.Tensor,
    bases: Dict[str, torch.Tensor],
    positions: Sequence[int],
) -> Dict[str, Dict]:
    """Write connectivity of every head and MLP.

    Args:
        bases: workspace basis per capture name ('emb', '0', ..). Component
            writes in block l are projected onto bases[str(l)] (their
            landing layer's workspace).
        positions: positions to aggregate over (key positions).

    Returns:
        {component: {"connectivity": float, "write_norm": float}} with
        components "L{l}H{h}", "L{l}MLP", and "emb".
    """
    device = next(model.parameters()).device
    inputs = capture_block_inputs(model, tokens)
    results: Dict[str, Dict] = {}

    emb_basis = bases["emb"].to(device)
    results["emb"] = {
        "connectivity": _row_connectivity(inputs[0], emb_basis, positions),
        "write_norm": float(inputs[0][:, list(positions)].norm(dim=-1).mean()),
    }

    for l in range(model.n_layers):
        block = model.blocks[l]
        x = inputs[l]
        basis = bases[str(l)].to(device)
        head_writes, attn_out = _per_head_writes(block, x)
        for h in range(head_writes.shape[0]):
            results[f"L{l}H{h}"] = {
                "connectivity": _row_connectivity(head_writes[h], basis, positions),
                "write_norm": float(
                    head_writes[h][:, list(positions)].norm(dim=-1).mean()
                ),
            }
        x_mid = x + attn_out
        mlp_out = block.mlp(block.norm2(x_mid))
        results[f"L{l}MLP"] = {
            "connectivity": _row_connectivity(mlp_out, basis, positions),
            "write_norm": float(mlp_out[:, list(positions)].norm(dim=-1).mean()),
        }
    return results


@torch.no_grad()
def read_connectivity_all(
    model,
    tokens: torch.Tensor,
    bases: Dict[str, torch.Tensor],
    positions: Sequence[int],
    n_dirs: int = 8,
    n_random: int = 8,
    eps: float = 1e-2,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Read connectivity: finite-difference sensitivity of each component's
    output to J-space vs random perturbations of its input residual.

    The perturbation direction u is applied at every position of the block
    input (components are causal, so this upper-bounds per-position reads);
    sensitivity is ||out(x + eps u) - out(x - eps u)||_F / (2 eps), averaged
    over directions and normalized per-component by the random-direction
    sensitivity: ratio > 1 means the component preferentially reads the
    workspace.

    The J-directions for block l come from bases[input name of block l]
    ('emb' for block 0, str(l-1) otherwise).
    """
    device = next(model.parameters()).device
    inputs = capture_block_inputs(model, tokens)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    results: Dict[str, Dict] = {}
    pos = list(positions)

    for l in range(model.n_layers):
        block = model.blocks[l]
        x = inputs[l]
        in_name = "emb" if l == 0 else str(l - 1)
        J_dirs = bases[in_name][:, :n_dirs].T.to(device)  # (n_dirs, d)
        R = torch.randn(n_random, x.shape[-1], generator=gen)
        R = (R / R.norm(dim=-1, keepdim=True)).to(device)

        def sensitivities(dirs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Per-direction (head_sens (n, H), mlp_sens (n,))."""
            H = block.attn.n_heads
            head_s = torch.zeros(dirs.shape[0], H)
            mlp_s = torch.zeros(dirs.shape[0])
            for di, u in enumerate(dirs):
                x_p = x + eps * u
                x_m = x - eps * u
                hw_p, ao_p = _per_head_writes(block, x_p)
                hw_m, ao_m = _per_head_writes(block, x_m)
                dh = (hw_p - hw_m)[:, :, pos, :] / (2 * eps)
                head_s[di] = dh.flatten(1).norm(dim=1).cpu() / (
                    dh.shape[1] * len(pos)
                ) ** 0.5
                mo_p = block.mlp(block.norm2(x_p + ao_p))
                mo_m = block.mlp(block.norm2(x_m + ao_m))
                dm = (mo_p - mo_m)[:, pos, :] / (2 * eps)
                mlp_s[di] = float(dm.norm() / (dm.shape[0] * len(pos)) ** 0.5)
            return head_s, mlp_s

        jh, jm = sensitivities(J_dirs)
        rh, rm = sensitivities(R)
        for h in range(block.attn.n_heads):
            results[f"L{l}H{h}"] = {
                "j_sensitivity": float(jh[:, h].mean()),
                "rand_sensitivity": float(rh[:, h].mean()),
                "ratio": float(jh[:, h].mean() / max(rh[:, h].mean(), 1e-9)),
            }
        results[f"L{l}MLP"] = {
            "j_sensitivity": float(jm.mean()),
            "rand_sensitivity": float(rm.mean()),
            "ratio": float(jm.mean() / max(rm.mean(), 1e-9)),
        }
    return results
