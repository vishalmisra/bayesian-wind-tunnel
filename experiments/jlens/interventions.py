"""
Causal interventions on the workspace (P4): J-space swaps between
evidence-matched donor pairs, and direction/subspace ablations.

Swap design: donor pairs share the exact key order (data_gen fixed_keys)
but carry independent permutations, so the observed positions are
identical and only the hypothesis content differs. Patching the J-space
component of the evidence region with the donor's should redirect the
model's posterior to the Bayes posterior of the *donor's* evidence if and
only if the J-space carries the causally load-bearing content.

KL convention: KL(Bayes || model) in bits. The analytic posterior has
exact zeros (eliminated hypotheses), so KL(model || Bayes) is infinite by
construction; the reverse direction is finite because the model has full
support. Recorded as a deviation from the spec's notation.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.jlens.data_gen import JLensBatch  # noqa: E402
from experiments.jlens.extract import _ResidualRecorder  # noqa: E402


@torch.no_grad()
def capture_residuals(model, tokens: torch.Tensor, layer_name: str) -> torch.Tensor:
    """(B, T, d) residual at a capture point, detached, on device."""
    device = next(model.parameters()).device
    with _ResidualRecorder(model) as recorder:
        model.logits(tokens.to(device))
    return recorder.activations[layer_name].detach()


class _PatchHook:
    """Replace a subspace component of the residual at chosen positions.

    For capture name 'emb' the patch applies to blocks[0]'s input (forward
    pre-hook); for name str(l) it applies to blocks[l]'s output (forward
    hook). The patched value is
        h <- h - P h + P h_donor          (P = basis @ basis^T)
    at the given positions; a full-residual patch (basis=None) replaces h
    entirely.
    """

    def __init__(
        self,
        model,
        layer_name: str,
        positions: Sequence[int],
        donor_resid: torch.Tensor,
        basis: Optional[torch.Tensor],
    ):
        self.model = model
        self.layer_name = layer_name
        self.positions = list(positions)
        self.donor = donor_resid
        self.basis = basis
        self._handle = None

    def _patch(self, h: torch.Tensor) -> torch.Tensor:
        pos = self.positions
        if self.basis is None:
            h[:, pos, :] = self.donor[:, pos, :]
            return h
        Q = self.basis.to(h.device, h.dtype)  # (d, r)
        delta = (self.donor[:, pos, :] - h[:, pos, :]) @ Q @ Q.T
        h[:, pos, :] = h[:, pos, :] + delta
        return h

    def __enter__(self):
        if self.layer_name == "emb":

            def pre_hook(module, args):
                h = args[0].clone()
                return (self._patch(h), *args[1:])

            self._handle = self.model.emb_module.register_forward_pre_hook(pre_hook)
        else:
            block = self.model.blocks[int(self.layer_name)]

            def hook(module, inputs, output):
                tensor = output if torch.is_tensor(output) else output[0]
                patched = self._patch(tensor.clone())
                if torch.is_tensor(output):
                    return patched
                return (patched, *output[1:])

            self._handle = block.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@torch.no_grad()
def patched_logits(
    model,
    tokens: torch.Tensor,
    layer_name: str,
    positions: Sequence[int],
    donor_resid: torch.Tensor,
    basis: Optional[torch.Tensor],
) -> torch.Tensor:
    device = next(model.parameters()).device
    with _PatchHook(model, layer_name, positions, donor_resid, basis):
        return model.logits(tokens.to(device)).float().cpu()


def _bayes_at_position(batch: JLensBatch, i: int) -> torch.Tensor:
    """(B, V) analytic posterior at key position i: uniform over the
    surviving hypothesis set (keys are unique, so the key at i is unseen)."""
    surviving = (~batch.eliminated[:, i]).float()
    return surviving / surviving.sum(-1, keepdim=True)


def _model_posterior(logits: torch.Tensor, i: int, V: int, vocab: int) -> torch.Tensor:
    lo = V if vocab == 2 * V else 0
    return F.softmax(logits[:, i, lo : lo + V], dim=-1)


def kl_bits(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) in bits, rows; p may have zeros, q must be positive."""
    mask = p > 0
    ratio = torch.where(mask, p / q.clamp(min=1e-12), torch.ones_like(p))
    return (p * torch.log2(ratio)).sum(-1)


@torch.no_grad()
def swap_experiment(
    model,
    orig: JLensBatch,
    donor: JLensBatch,
    layer_name: str,
    basis: Optional[torch.Tensor],
    read_positions: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    """J-space swap: patch the evidence region (positions < read position)
    at `layer_name` with the donor's subspace component; read the posterior
    at each read position.

    Returns per read position: mean KL(Bayes_donor || model_patched),
    mean KL(Bayes_orig || model_patched), the redirect margin (orig - donor,
    >= 1 bit = success), and unpatched baselines.
    """
    donor_resid = capture_residuals(model, donor.tokens, layer_name)
    base_logits = model.logits(
        orig.tokens.to(next(model.parameters()).device)
    ).float().cpu()

    results: Dict[int, Dict[str, float]] = {}
    for i in read_positions:
        patch_pos = list(range(i))
        logits = patched_logits(
            model, orig.tokens, layer_name, patch_pos, donor_resid, basis
        )
        p_model = _model_posterior(logits, i, orig.V, model.vocab_size)
        p_base = _model_posterior(base_logits, i, orig.V, model.vocab_size)
        bayes_orig = _bayes_at_position(orig, i)
        bayes_donor = _bayes_at_position(donor, i)

        kl_donor = kl_bits(bayes_donor.float(), p_model)
        kl_orig = kl_bits(bayes_orig.float(), p_model)
        results[int(i)] = {
            "kl_donor_bits": float(kl_donor.mean()),
            "kl_orig_bits": float(kl_orig.mean()),
            "redirect_margin_bits": float((kl_orig - kl_donor).mean()),
            "frac_pairs_redirected": float((kl_orig > kl_donor).float().mean()),
            "baseline_kl_orig_bits": float(kl_bits(bayes_orig.float(), p_base).mean()),
            "baseline_kl_donor_bits": float(
                kl_bits(bayes_donor.float(), p_base).mean()
            ),
        }
    return results


@torch.no_grad()
def subspace_ablation_mae(
    model,
    batch: JLensBatch,
    layer_name: str,
    basis: Optional[torch.Tensor],
) -> float:
    """Calibration MAE with a subspace projected OUT of the residual at
    every position of `layer_name` (basis=None -> zero the whole residual).

    Reuses the patch hook with an all-zero donor: h - P h + P 0 = h - P h.
    """
    from experiments.jlens.calibration import eval_entropy_mae

    device = next(model.parameters()).device
    T = batch.T
    d = model.dim
    zero_donor = torch.zeros(batch.B, T, d, device=device)

    class _Shim:
        """eval_entropy_mae calls model.logits(tokens); wrap it patched."""

        def __init__(self, inner):
            self.inner = inner
            self.vocab_size = inner.vocab_size

        def parameters(self):
            return self.inner.parameters()

        def logits(self, tokens):
            with _PatchHook(
                self.inner, layer_name, list(range(T)), zero_donor, basis
            ):
                return self.inner.logits(tokens)

    return eval_entropy_mae(_Shim(model), batch)
