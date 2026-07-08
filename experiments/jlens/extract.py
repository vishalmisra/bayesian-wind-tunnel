"""
Per-(layer, position) J-space extraction for wind-tunnel transformer models.

Operationalization (see DEVIATIONS.md for the delta vs Anthropic's release):
for the residual-stream activation x at (layer l, position i), the J-space is
the top-r right singular subspace of the stacked Jacobian rows

    { d logits[p', v] / d x_{l,i}  :  v in cot_dims,  p' >= i + min_horizon,
                                      sequences b in batch }.

Rather than materializing the stacked matrix, we accumulate its Gram matrix
G_{l,i} = M^T M (d x d) on the fly; the top right-singular subspace of M is
the top eigenspace of G. This makes the full (l, i) sweep memory-trivial.

Two reductions are computed in one pass:
  * "stacked":  Gram of the individual (p', v) rows -- position-resolved.
  * "summed":   Gram of rows summed over future p' first (v rows only) --
                the reduction closest to Anthropic's estimator, which
                injects the cotangent at every valid target position at once.

Backward-pass count per batch chunk: T - min_horizon (one per target
position, all cotangent dims batched along the replicated batch axis),
following the dim-batching trick in anthropics/jacobian-lens fitting.py.

Models must expose the uniform interface from experiments/jlens/models.py:
.blocks, .emb_module, .logits(x), .dim, .vocab_size.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def capture_names(n_layers: int) -> List[str]:
    """Residual capture points: 'emb' is the post-embedding (pre-block-0)
    residual; integer name l is the output of blocks[l]."""
    return ["emb"] + [str(l) for l in range(n_layers)]


@dataclass
class GramSweep:
    """Accumulated Gram matrices for one model + batch config.

    grams[reduction][name] has shape (T, d, d): one Gram per source position.
    counts[reduction][name] has shape (T,): number of stacked rows per source
    position (diagnostics only; subspaces are scale-invariant).
    """

    T: int
    d: int
    names: List[str]
    min_horizon: int
    cot_dims: List[int]
    grams: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    counts: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    n_sequences: int = 0

    @classmethod
    def empty(
        cls, T: int, d: int, names: List[str], min_horizon: int, cot_dims: List[int]
    ) -> "GramSweep":
        sweep = cls(T=T, d=d, names=names, min_horizon=min_horizon, cot_dims=cot_dims)
        for reduction in ("stacked", "summed"):
            sweep.grams[reduction] = {
                name: torch.zeros(T, d, d, dtype=torch.float64) for name in names
            }
            sweep.counts[reduction] = {
                name: torch.zeros(T, dtype=torch.long) for name in names
            }
        return sweep

    def top_subspace(
        self, reduction: str, name: str, position: int, r: int
    ) -> torch.Tensor:
        """Top-r eigenvectors of G_{l,i}: orthonormal (d, r) basis, leading
        direction first."""
        G = self.grams[reduction][name][position]
        eigvals, eigvecs = torch.linalg.eigh(G)
        return eigvecs[:, -r:].flip(-1).float()

    def spectrum(self, reduction: str, name: str, position: int) -> torch.Tensor:
        G = self.grams[reduction][name][position]
        return torch.linalg.eigvalsh(G).flip(0).float()

    def save(self, path) -> None:
        torch.save(
            {
                "T": self.T,
                "d": self.d,
                "names": self.names,
                "min_horizon": self.min_horizon,
                "cot_dims": self.cot_dims,
                "grams": self.grams,
                "counts": self.counts,
                "n_sequences": self.n_sequences,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "GramSweep":
        state = torch.load(path, map_location="cpu", weights_only=True)
        sweep = cls(
            T=state["T"],
            d=state["d"],
            names=state["names"],
            min_horizon=state["min_horizon"],
            cot_dims=state["cot_dims"],
        )
        sweep.grams = state["grams"]
        sweep.counts = state["counts"]
        sweep.n_sequences = state["n_sequences"]
        return sweep


class _ResidualRecorder:
    """Hooks capturing the residual stream, autograd-visible.

    'emb' is captured with a forward *pre*-hook on model.emb_module (its
    input is the embedding residual) and marked requires_grad so it roots
    the autograd graph when parameters are frozen -- same trick as
    anthropics/jacobian-lens hooks.py start_graph_at. Block outputs are
    captured with forward hooks, tuple-unwrapped where needed.
    """

    def __init__(self, model):
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        def emb_pre_hook(module, args):
            h = args[0]
            if h.is_leaf:
                h.requires_grad_(True)
            self.activations["emb"] = h

        self._handles.append(
            self.model.emb_module.register_forward_pre_hook(emb_pre_hook)
        )
        for l, block in enumerate(self.model.blocks):

            def block_hook(module, inputs, output, _l=l):
                tensor = output if torch.is_tensor(output) else output[0]
                self.activations[str(_l)] = tensor

            self._handles.append(block.register_forward_hook(block_hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.enable_grad()
def accumulate_grams(
    model,
    tokens: torch.Tensor,
    min_horizon: int = 1,
    seq_chunk: int = 64,
    layers: Optional[List[str]] = None,
    cot_dims: Optional[Sequence[int]] = None,
    show_progress: bool = False,
) -> GramSweep:
    """Run the full (layer, position) Jacobian sweep on a token batch.

    Args:
        model: uniform-interface model (experiments/jlens/models.py), eval
            mode. Parameters are frozen for the duration; gradients are
            taken w.r.t. captured residuals only.
        tokens: (B, T) long tensor.
        min_horizon: Smallest target offset p' - i included. 1 = strictly
            future logits; 2 = the unembedding-geometry robustness check
            from the spec's risk section.
        seq_chunk: Sequences per forward pass; the effective batch is
            seq_chunk * len(cot_dims) (cotangent dims ride the batch axis).
        layers: Subset of capture names; default all.
        cot_dims: Output (vocab) dimensions to inject cotangents at.
            Default: all. For sep-vocab bijection models pass the value-token
            slice range(V, 2V) -- the hypothesis logits.

    Returns:
        A GramSweep with both reductions accumulated (float64 on CPU).
    """
    assert min_horizon >= 1
    model.eval()
    device = next(model.parameters()).device
    B, T = tokens.shape
    d = model.dim
    dims = list(cot_dims) if cot_dims is not None else list(range(model.vocab_size))
    K = len(dims)
    names = layers if layers is not None else capture_names(model.n_layers)

    param_grad_state = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in param_grad_state:
        p.requires_grad_(False)

    sweep = GramSweep.empty(
        T=T, d=d, names=names, min_horizon=min_horizon, cot_dims=dims
    )

    try:
        for start in range(0, B, seq_chunk):
            chunk = tokens[start : start + seq_chunk].to(device)
            Bc = chunk.shape[0]
            # dim-major replication: row (j * Bc + b) carries cotangent
            # dim dims[j].
            x_rep = chunk.repeat(K, 1)  # (K*Bc, T)
            dim_of_row = torch.tensor(dims, device=device).repeat_interleave(Bc)
            rows = torch.arange(K * Bc, device=device)

            # Per-chunk device-side fp32 accumulators; folded into the
            # global fp64 CPU sweep at chunk end (fp64 is slow on GPU).
            dev_stacked = {
                name: torch.zeros(T, d, d, device=device) for name in names
            }
            summed_rows = {
                name: torch.zeros(K * Bc, T, d, device=device) for name in names
            }

            with _ResidualRecorder(model) as recorder:
                logits = model.logits(x_rep)
                resids = [recorder.activations[name] for name in names]

                n_backwards = T - min_horizon
                for step, p_target in enumerate(
                    range(T - 1, min_horizon - 1, -1)
                ):
                    cot = torch.zeros_like(logits)
                    cot[rows, p_target, dim_of_row] = 1.0
                    grads = torch.autograd.grad(
                        outputs=logits,
                        inputs=resids,
                        grad_outputs=cot,
                        retain_graph=(step < n_backwards - 1),
                    )
                    i_max = p_target - min_horizon  # max source position
                    for name, g in zip(names, grads):
                        g_src = g[:, : i_max + 1, :]  # (K*Bc, i_max+1, d)
                        dev_stacked[name][: i_max + 1] += torch.einsum(
                            "rid,rie->ide", g_src, g_src
                        )
                        sweep.counts["stacked"][name][: i_max + 1] += K * Bc
                        summed_rows[name][:, : i_max + 1, :] += g_src
                    if show_progress and step % 8 == 0:
                        print(
                            f"  chunk {start // seq_chunk}: target {p_target}",
                            flush=True,
                        )

            for name in names:
                S = summed_rows[name]
                dev_summed = torch.einsum("rid,rie->ide", S, S)
                sweep.grams["summed"][name] += dev_summed.double().cpu()
                sweep.counts["summed"][name] += K * Bc
                sweep.grams["stacked"][name] += dev_stacked[name].double().cpu()

            sweep.n_sequences += Bc
    finally:
        for p, was in param_grad_state:
            p.requires_grad_(was)

    return sweep
