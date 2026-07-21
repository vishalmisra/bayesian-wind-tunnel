"""
Comparison-target subspaces: the hypothesis frame, random nulls, and the
entropy-readout axis.

Frame subspace (spec section 4.2, following Paper I): the span of the
Layer-0 hypothesis-frame head's key directions for the hypothesis tokens,
and separately its value(-write) directions. Both are pulled back to
residual-stream coordinates (d_model) so they are directly comparable to
J-spaces measured on the residual stream:

  * mode "key":   W_K^head applied to the normalized token embedding gives
                  the key in head space; the residual-space read direction
                  for hypothesis h is W_K^head^T k_h (the direction of the
                  residual that maximally drives that key coordinate).
  * mode "value": the OV write direction W_O^head W_V^head e_h -- what the
                  head writes into the residual stream for hypothesis h.
  * mode "embedding": the raw token embedding directions e_h (baseline).

The hypothesis-frame head is identified per Paper I: the Layer-0 head whose
keys for the hypothesis tokens are most nearly orthogonal (minimum mean
absolute off-diagonal cosine similarity).

For sep-vocab bijection models (vocab = 2V) the hypothesis tokens are the
value tokens V..2V-1; for shared-vocab models they are all V tokens.

Handles both TinyGPT variants (fused qkv / separate wq-wk-wv-wo); see
experiments/jlens/models.py.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.geometry import measure_key_orthogonality  # noqa: E402


def _inner(model):
    return model.inner if hasattr(model, "inner") else model


def default_hypothesis_tokens(model, V: Optional[int] = None) -> torch.Tensor:
    """Hypothesis token ids: value tokens V..2V-1 for sep-vocab models,
    all tokens otherwise."""
    vocab = model.vocab_size
    if V is None:
        V = vocab // 2 if vocab % 2 == 0 else vocab
    if vocab == 2 * V:
        return torch.arange(V, 2 * V)
    return torch.arange(vocab)


def _l0_head_matrices(
    model, head: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(W_K_head (hd,d), W_V_head (hd,d), W_O_head (d,hd)) for Layer 0,
    fp32 on CPU, for either TinyGPT variant."""
    attn = _inner(model).blocks[0].attn
    hd = attn.head_dim
    sl = slice(head * hd, (head + 1) * hd)
    if hasattr(attn, "wk"):  # separate-projection variant
        W_K = attn.wk.weight.detach().float().cpu()[sl]
        W_V = attn.wv.weight.detach().float().cpu()[sl]
        W_O_head = attn.wo.weight.detach().float().cpu()[:, sl]
    else:  # fused qkv variant (src/models/tinygpt.py)
        d = _inner(model).dim
        qkv_w = attn.qkv.weight.detach().float().cpu()  # (3d, d)
        W_K = qkv_w[d : 2 * d][sl]
        W_V = qkv_w[2 * d : 3 * d][sl]
        W_O_head = attn.out_proj.weight.detach().float().cpu()[:, sl]
    return W_K, W_V, W_O_head


def _normed_token_embeddings(model) -> torch.Tensor:
    """Token embeddings as Layer 0's attention sees them: norm1(tok_emb).

    Positional embeddings are excluded: hypothesis tokens appear at many
    positions, and the frame claim is about token identity.
    """
    inner = _inner(model)
    emb_module = getattr(inner, "tok_emb", None) or inner.token_embed
    emb = emb_module.weight.detach().float().cpu()  # (vocab[+pad], d)
    block0 = inner.blocks[0]
    norm = getattr(block0, "norm1", None) or getattr(block0, "ln1", None) or getattr(
        block0, "ln", None
    )
    if norm is None:
        return emb
    if isinstance(norm, torch.nn.LayerNorm):
        return torch.nn.functional.layer_norm(
            emb, norm.normalized_shape, norm.weight.detach().cpu(),
            norm.bias.detach().cpu() if norm.bias is not None else None, norm.eps
        )
    scale = (norm.weight if hasattr(norm, "weight") else norm.scale).detach().float().cpu()
    rms = torch.rsqrt(emb.pow(2).mean(-1, keepdim=True) + norm.eps)
    return emb * rms * scale


def identify_frame_head(
    model, hypotheses: Optional[torch.Tensor] = None
) -> Tuple[int, Dict[int, float]]:
    """The Layer-0 head with the most orthogonal hypothesis keys."""
    if hypotheses is None:
        hypotheses = default_hypothesis_tokens(model)
    emb = _normed_token_embeddings(model)[hypotheses]
    n_heads = _inner(model).blocks[0].attn.n_heads
    scores: Dict[int, float] = {}
    for h in range(n_heads):
        W_K, _, _ = _l0_head_matrices(model, h)
        keys = (emb @ W_K.T).numpy()  # (k, hd)
        scores[h] = measure_key_orthogonality(keys)
    best = min(scores, key=scores.get)
    return best, scores


def frame_subspace(
    model,
    mode: str = "key",
    head: Optional[int] = None,
    hypotheses: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Orthonormal basis (d, k) of the hypothesis-frame subspace.

    Args:
        model: uniform-interface model.
        mode: "key" | "value" | "embedding" (see module docstring).
        head: Layer-0 head index; default = identify_frame_head(model).
        hypotheses: Hypothesis token ids; default = value tokens.
    """
    if hypotheses is None:
        hypotheses = default_hypothesis_tokens(model)
    emb = _normed_token_embeddings(model)[hypotheses]  # (k, d)

    if mode == "embedding":
        directions = emb
    else:
        if head is None:
            head, _ = identify_frame_head(model, hypotheses)
        W_K, W_V, W_O_head = _l0_head_matrices(model, head)
        if mode == "key":
            directions = (emb @ W_K.T) @ W_K  # (k, d): W_K^T k_h
        elif mode == "value":
            directions = (emb @ W_V.T) @ W_O_head.T  # (k, d): W_O W_V e_h
        else:
            raise ValueError(f"unknown mode {mode!r}")

    Q, _ = torch.linalg.qr(directions.T)  # (d, k)
    return Q


def random_subspaces(
    d: int, r: int, n: int = 1000, seed: int = 0, device: str = "cpu"
) -> torch.Tensor:
    """(n, d, r) stack of random orthonormal subspaces (QR of Gaussian)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(n, d, r, generator=gen)
    Q, _ = torch.linalg.qr(A)
    return Q.to(device)


def entropy_axis(
    model,
    batch,
    layer_name: str,
    positions: Optional[np.ndarray] = None,
    ridge: float = 1e-3,
) -> Tuple[torch.Tensor, float]:
    """The entropy-readout axis: the residual direction best predicting the
    analytic Bayes entropy, fit by ridge regression at the given layer.

    Ground truth: the surviving-set entropy log2(#surviving hypotheses)
    from the batch's elimination mask.

    Args:
        model: uniform-interface model.
        batch: JLensBatch (data_gen.generate_batch).
        layer_name: capture name ('emb' or block index string).
        positions: source positions to pool; default = batch.key_positions().

    Returns:
        (unit direction (d,), r-squared on a held-out half).
    """
    from extract import _ResidualRecorder

    device = next(model.parameters()).device
    tokens = batch.tokens.to(device)
    with _ResidualRecorder(model) as recorder, torch.no_grad():
        model.logits(tokens)
        resid = recorder.activations[layer_name].detach().cpu().float()

    surviving = (~batch.eliminated).sum(-1).clamp(min=1)  # (B, T)
    ent = torch.log2(surviving.float())  # (B, T)

    if positions is None:
        positions = batch.key_positions()

    X = resid[:, positions, :].reshape(-1, resid.shape[-1]).numpy()
    y = ent[:, positions].reshape(-1).numpy()

    n = X.shape[0]
    half = n // 2
    XtX = X[:half].T @ X[:half] + ridge * np.eye(X.shape[1])
    w = np.linalg.solve(XtX, X[:half].T @ y[:half])
    pred = X[half:] @ w
    ss_res = float(((y[half:] - pred) ** 2).sum())
    ss_tot = float(((y[half:] - y[half:].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    w_t = torch.from_numpy(w).float()
    return w_t / w_t.norm(), r2
