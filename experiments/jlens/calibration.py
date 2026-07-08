"""
Entropy-calibration evaluation and head-ablation utilities (uniform model
interface). Used to identify the hypothesis-frame head by indispensability
(Paper I's criterion) and to sanity-check loaded checkpoints.
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.jlens.data_gen import JLensBatch  # noqa: E402


@torch.no_grad()
def eval_entropy_mae(model, batch: JLensBatch) -> float:
    """Mean |H_model - H_bayes| in bits over key positions.

    Model entropy is over the hypothesis (value-token) logit slice; the
    Bayes entropy is the surviving-set entropy from the batch annotation.
    """
    device = next(model.parameters()).device
    logits = model.logits(batch.tokens.to(device)).float().cpu()
    V = batch.V
    lo = V if model.vocab_size == 2 * V else 0
    key_pos = torch.from_numpy(batch.key_positions())
    lv = logits[:, key_pos, lo : lo + V]
    p = F.softmax(lv, dim=-1)
    h_model = -(p * (p + 1e-12).log2()).sum(-1)  # (B, n_key_pos)
    surviving = (~batch.eliminated[:, key_pos]).sum(-1).clamp(min=1)
    h_bayes = torch.log2(surviving.float())
    return float((h_model - h_bayes).abs().mean())


@contextmanager
def head_ablated(model, layer: int, head: int):
    """Zero one attention head via its head_mask buffer."""
    mask = model.blocks[layer].attn.head_mask
    original = mask[head].item()
    mask[head] = 0.0
    try:
        yield
    finally:
        mask[head] = original


def frame_head_by_ablation(
    model, batch: JLensBatch, layer: int = 0
) -> Tuple[int, Dict[int, float]]:
    """The layer-`layer` head whose ablation most degrades calibration.

    Returns (head, {head: MAE_when_ablated}). Baseline MAE is under key -1.
    """
    baseline = eval_entropy_mae(model, batch)
    n_heads = model.blocks[layer].attn.n_heads
    maes: Dict[int, float] = {-1: baseline}
    for h in range(n_heads):
        with head_ablated(model, layer, h):
            maes[h] = eval_entropy_mae(model, batch)
    best = max((h for h in maes if h >= 0), key=lambda h: maes[h])
    return best, maes
