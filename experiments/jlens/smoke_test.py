#!/usr/bin/env python3
"""CPU smoke test for the J-lens pipeline: correctness, not science.

Covers BOTH model variants (sep-vocab wq/wk/wv/wo and legacy fused-qkv):
  1. data_gen ground truth agrees with src.utils.entropy.bayes_bijection_posterior.
  2. Causality: Jacobian rows vanish for sources at/after the target
     (counts bookkeeping) and the strictly-future mask is honored.
  3. Gram accumulation is chunk-invariant (one chunk == many chunks).
  4. Gram matches a brute-force autograd jacobian on a tiny case.
  5. Frame subspace bases are orthonormal, correct shape; metrics sane.
  6. Seeded determinism.
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.jlens.data_gen import generate_batch
from experiments.jlens.extract import _ResidualRecorder, accumulate_grams, capture_names
from experiments.jlens.metrics import (
    linear_cka,
    overlap_with_null,
    principal_angles,
    projection_overlap,
)
from experiments.jlens.models import LegacyTinyGPTWrapper, SepVocabTinyGPT
from experiments.jlens.subspaces import (
    default_hypothesis_tokens,
    frame_subspace,
    identify_frame_head,
    random_subspaces,
)
from src.models.tinygpt import TinyGPT
from src.utils.entropy import bayes_bijection_posterior

torch.manual_seed(0)

V, L = 8, 7
MODELS = {
    "sepvocab": SepVocabTinyGPT(
        vocab_size=2 * V, dim=32, n_layers=3, n_heads=2, max_seq_len=2 * L
    ),
    "paired": LegacyTinyGPTWrapper(
        TinyGPT(vocab_size=V, dim=32, n_layers=3, n_heads=2, max_seq_len=2 * L + 1)
    ),
}
for m in MODELS.values():
    m.eval()

# --- 1. data_gen ground truth -------------------------------------------------
for fmt in ("sepvocab", "paired"):
    batch = generate_batch(16, V=V, L=L, seed=1, fmt=fmt)
    for b in range(batch.B):
        n_pairs = L - 1 if fmt == "sepvocab" else L
        pairs = [
            (int(batch.keys[b, t]), int(batch.perms[b, batch.keys[b, t]]))
            for t in range(n_pairs)
        ]
        ref = bayes_bijection_posterior(V, pairs, int(batch.query[b]))
        assert np.allclose(batch.bayes_query[b].numpy(), ref), f"{fmt} b={b}"
        assert int(batch.eliminated[b, batch.T - 1].sum()) == n_pairs
    # Token ranges.
    if fmt == "sepvocab":
        assert batch.T == 2 * L - 1
        assert batch.tokens[:, 0::2].max() < V  # keys
        assert batch.tokens[:, 1::2].min() >= V  # values
    else:
        assert batch.T == 2 * L + 1
print("1. data_gen ground truth: OK")

# --- 2/3. Gram sweep: chunk invariance + causality ---------------------------
for fmt, model in MODELS.items():
    batch = generate_batch(8, V=V, L=L, seed=1, fmt=fmt)
    hyp = default_hypothesis_tokens(model, V=V).tolist()
    sweep_1 = accumulate_grams(model, batch.tokens, seq_chunk=8, cot_dims=hyp)
    sweep_4 = accumulate_grams(model, batch.tokens, seq_chunk=2, cot_dims=hyp)
    names = capture_names(model.n_layers)
    T = batch.T
    for reduction in ("stacked", "summed"):
        for name in names:
            a, b = sweep_1.grams[reduction][name], sweep_4.grams[reduction][name]
            assert torch.allclose(a, b, atol=1e-6, rtol=1e-4), (
                f"{fmt}/{reduction}/{name}: {(a - b).abs().max().item():.2e}"
            )
    for name in names:
        assert sweep_1.counts["stacked"][name][T - 1] == 0
        assert sweep_1.grams["stacked"][name][T - 1].abs().max() == 0
print("2/3. chunk invariance + causality (both variants): OK")

# --- 4. brute-force check on one sequence ------------------------------------
model = MODELS["sepvocab"]
batch = generate_batch(8, V=V, L=L, seed=1, fmt="sepvocab")
hyp = default_hypothesis_tokens(model, V=V).tolist()
T = batch.T
x = batch.tokens[:1]
sweep_bf = accumulate_grams(model, x, seq_chunk=1, cot_dims=hyp)


def brute_force_gram(layer_name: str, i: int) -> torch.Tensor:
    G = torch.zeros(model.dim, model.dim, dtype=torch.float64)
    for p in model.parameters():
        p.requires_grad_(False)
    for p_t in range(i + 1, T):
        for v in hyp:
            with _ResidualRecorder(model) as rec, torch.enable_grad():
                logits = model.logits(x)
                resid = rec.activations[layer_name]
                (row,) = torch.autograd.grad(logits[0, p_t, v], resid)
            r = row[0, i, :].double()
            G += torch.outer(r, r)
    return G


for layer_name, i in (("emb", 0), ("1", 3), (str(model.n_layers - 1), T - 2)):
    G_bf = brute_force_gram(layer_name, i)
    G_sw = sweep_bf.grams["stacked"][layer_name][i]
    assert torch.allclose(G_bf, G_sw, atol=1e-8, rtol=1e-4), (
        f"brute force mismatch at ({layer_name}, {i}): "
        f"{(G_bf - G_sw).abs().max().item():.2e}"
    )
print("4. brute-force Jacobian Gram match: OK")

# --- 5. subspace utilities ----------------------------------------------------
for fmt, model in MODELS.items():
    hyp_t = default_hypothesis_tokens(model, V=V)
    assert len(hyp_t) == V
    head, scores = identify_frame_head(model, hyp_t)
    for mode in ("key", "value", "embedding"):
        F = frame_subspace(model, mode=mode, hypotheses=hyp_t)
        assert F.shape == (model.dim, V), (fmt, mode, F.shape)
        assert torch.allclose(F.T @ F, torch.eye(V), atol=1e-5)
model = MODELS["sepvocab"]
U = sweep_bf.top_subspace("stacked", "0", 2, r=4)
assert U.shape == (model.dim, 4)
assert torch.allclose(U.T @ U, torch.eye(4), atol=1e-4)
F = frame_subspace(model, mode="key")
angles = principal_angles(U, F)
assert angles.shape[0] == 4
nulls = random_subspaces(model.dim, 4, n=50, seed=0)
stats = overlap_with_null(U, F, nulls)
assert 0.0 <= stats["observed"] <= 1.0
assert 0.0 <= projection_overlap(U, F) <= 1.0 + 1e-9
assert 0.0 <= linear_cka(U, F) <= 1.0 + 1e-9
print(f"5. subspaces + metrics: OK (rand-model null z={stats['z']:.2f})")

# --- 6. deterministic regeneration -------------------------------------------
b1 = generate_batch(16, V=V, L=L, seed=1, fmt="sepvocab")
b2 = generate_batch(16, V=V, L=L, seed=1, fmt="sepvocab")
assert torch.equal(b1.tokens, b2.tokens)
print("6. seeded determinism: OK")

print("\nALL SMOKE TESTS PASSED")
