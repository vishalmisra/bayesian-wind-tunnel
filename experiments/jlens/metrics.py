"""
Subspace-comparison and decoding metrics for the J-lens experiments.

Overlap metrics (spec section 4.3): principal angles, projection-Frobenius
overlap, and linear CKA between orthonormal subspace bases, plus z-scores
against a matched-dimension random-subspace null.

Decoding: logistic probes from J-space coordinates to hypothesis identity
(P1) and per-hypothesis eliminated/surviving status (P3). Train/test split
is always across sequences, never within.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------
# Subspace overlap
# --------------------------------------------------------------------------

def principal_angles(U: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Principal angles (radians, ascending) between orthonormal bases
    U (d, r1) and W (d, r2)."""
    s = torch.linalg.svdvals(U.T @ W).clamp(-1.0, 1.0)
    return torch.arccos(s).flip(0).sort().values


def projection_overlap(U: torch.Tensor, W: torch.Tensor) -> float:
    """||P_U P_W||_F^2 / min(r1, r2) in [0, 1]; 1 iff the smaller subspace
    is contained in the larger."""
    s = torch.linalg.svdvals(U.T @ W)
    return float((s**2).sum() / min(U.shape[1], W.shape[1]))


def linear_cka(U: torch.Tensor, W: torch.Tensor) -> float:
    """Linear CKA between orthonormal bases:
    ||U^T W||_F^2 / sqrt(r1 * r2)."""
    num = float((torch.linalg.svdvals(U.T @ W) ** 2).sum())
    return num / float(np.sqrt(U.shape[1] * W.shape[1]))


def overlap_with_null(
    U: torch.Tensor,
    W: torch.Tensor,
    nulls: torch.Tensor,
    metric: str = "projection",
) -> Dict[str, float]:
    """Overlap(U, W) plus null distribution stats over `nulls` (n, d, r).

    The null replaces U (the J-space side) with random subspaces of the
    same dimension, keeping W (the frame) fixed.
    """
    fn = {"projection": projection_overlap, "cka": linear_cka}[metric]
    observed = fn(U, W)
    null_vals = np.array([fn(nulls[i], W) for i in range(nulls.shape[0])])
    mu, sd = float(null_vals.mean()), float(null_vals.std())
    return {
        "observed": observed,
        "null_mean": mu,
        "null_std": sd,
        "z": (observed - mu) / max(sd, 1e-12),
        "ratio_to_null": observed / max(mu, 1e-12),
    }


def subspace_stability(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """Reproducibility of a subspace across disjoint batches (G0 gate):
    projection overlap between the two extractions."""
    return projection_overlap(U1, U2)


# --------------------------------------------------------------------------
# Decoding probes
# --------------------------------------------------------------------------

def _logistic_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    balanced: bool = False,
) -> float:
    """Multinomial logistic probe accuracy (sklearn, lbfgs).

    balanced=True trains with class_weight='balanced' and scores balanced
    accuracy -- required for the P3 elimination probes, whose positive
    class base rate approaches 0.9 late in the sequence (a raw-accuracy
    probe would 'pass' by predicting the base rate).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        # Degenerate label set; probe is meaningless.
        return float("nan")
    clf = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced" if balanced else None
    )
    clf.fit(X_train, y_train)
    if balanced:
        return float(balanced_accuracy_score(y_test, clf.predict(X_test)))
    return float(clf.score(X_test, y_test))


def probe_from_coords(
    coords: torch.Tensor,
    labels: torch.Tensor,
    train_frac: float = 0.7,
    seed: int = 0,
    balanced: bool = False,
) -> float:
    """Probe accuracy from (N, r) coordinates to (N,) integer labels with a
    sequence-level split done by the caller (pass only one row per
    (sequence, position))."""
    rng = np.random.default_rng(seed)
    N = coords.shape[0]
    order = rng.permutation(N)
    n_train = int(train_frac * N)
    tr, te = order[:n_train], order[n_train:]
    X = coords.numpy()
    y = labels.numpy()
    return _logistic_probe(X[tr], y[tr], X[te], y[te], balanced=balanced)


def jspace_coordinates(
    model,
    tokens: torch.Tensor,
    basis: torch.Tensor,
    layer_name: str,
    position: int,
) -> torch.Tensor:
    """Project residuals at (layer, position) onto a J-space basis.

    Returns (B, r) coordinates on CPU.
    """
    from experiments.jlens.extract import _ResidualRecorder

    device = next(model.parameters()).device
    with _ResidualRecorder(model) as recorder, torch.no_grad():
        model.logits(tokens.to(device))
        resid = recorder.activations[layer_name].detach()
    h = resid[:, position, :].float().cpu()  # (B, d)
    return h @ basis  # (B, r)


def hypothesis_identity_probe(
    coords: torch.Tensor,
    value_tokens: torch.Tensor,
    train_frac: float = 0.7,
    seed: int = 0,
) -> float:
    """P1 decoding: from J-space coordinates at a value position, decode
    which hypothesis (value token) was just observed."""
    return probe_from_coords(coords, value_tokens, train_frac, seed)


def elimination_probes(
    coords: torch.Tensor,
    eliminated: torch.Tensor,
    train_frac: float = 0.7,
    seed: int = 0,
    balanced: bool = True,
) -> Dict[str, float]:
    """P3 decoding: per-hypothesis binary probes (eliminated vs surviving),
    scored with balanced accuracy by default (late-position base rates
    approach 0.9; raw accuracy would pass on the base rate alone).

    Args:
        coords: (B, r) J-space coordinates at one (layer, position).
        eliminated: (B, V) bool ground truth at that position.

    Returns:
        mean/min balanced accuracy over hypotheses with both classes present.
    """
    V = eliminated.shape[1]
    accs = []
    for v in range(V):
        y = eliminated[:, v].long()
        if y.min() == y.max():
            continue  # no variation at this position for this hypothesis
        accs.append(probe_from_coords(coords, y, train_frac, seed, balanced=balanced))
    accs = [a for a in accs if not np.isnan(a)]
    if not accs:
        return {"mean_acc": float("nan"), "min_acc": float("nan"), "n_probes": 0}
    return {
        "mean_acc": float(np.mean(accs)),
        "min_acc": float(np.min(accs)),
        "n_probes": len(accs),
    }


def hypothesis_projection_curves(
    resid: torch.Tensor,
    jbases: Dict[int, torch.Tensor],
    frame_dirs: torch.Tensor,
    eliminated: torch.Tensor,
    positions,
) -> Dict[str, list]:
    """P3 decay claim: |<P_J resid, f_h>| for eliminated vs surviving h,
    as elimination proceeds.

    Args:
        resid: (B, T, d) residual at one layer.
        jbases: {position: (d, r) J basis} per position.
        frame_dirs: (V, d) unit hypothesis directions (NOT orthogonalized;
            per-hypothesis interpretability matters more than orthogonality).
        eliminated: (B, T, V) ground truth.
        positions: positions to evaluate (key positions).

    Returns:
        {"position": [...], "surviving_mean": [...], "eliminated_mean": [...]}
    """
    out = {"position": [], "surviving_mean": [], "eliminated_mean": []}
    fd = frame_dirs / frame_dirs.norm(dim=-1, keepdim=True)  # (V, d)
    for i in positions:
        if int(i) not in jbases:
            continue
        Q = jbases[int(i)]  # (d, r)
        proj = resid[:, i, :].float().cpu() @ Q @ Q.T  # (B, d)
        coefs = (proj @ fd.T).abs()  # (B, V)
        elim = eliminated[:, i]  # (B, V) bool
        if elim.any() and (~elim).any():
            out["position"].append(int(i))
            out["eliminated_mean"].append(float(coefs[elim].mean()))
            out["surviving_mean"].append(float(coefs[~elim].mean()))
    return out


# --------------------------------------------------------------------------
# Read/write connectivity (P2)
# --------------------------------------------------------------------------

@torch.no_grad()
def write_connectivity(
    component_output: torch.Tensor, basis: torch.Tensor
) -> float:
    """||P_J o|| / ||o|| for a component's residual-stream write, averaged
    over rows of `component_output` (N, d)."""
    proj = component_output @ basis  # (N, r)
    num = proj.norm(dim=-1)
    den = component_output.norm(dim=-1).clamp(min=1e-12)
    return float((num / den).mean())
