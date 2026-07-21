"""
Held-out bijection batch generation with full ground-truth annotation.

Unlike the training datasets (which return only (x, y) pairs), the J-lens
analyses need per-position Bayesian ground truth: the surviving hypothesis
set at every position, the analytic posterior, and the raw (perm, keys)
metadata for evidence-matched intervention pairing (P4).

Two sequence formats:

  * "sepvocab" (default; matches logs/bijection_v20_repl training,
    train_v256_ddp.py family): [k1, v1+V, k2, v2+V, ..., kL] of length
    2L-1. Keys are tokens 0..V-1, values are tokens V..2V-1. There is no
    trailing query token: every key position is a query, supervised with
    the upcoming value token. Keys sampled without replacement.

  * "paired" (src/data/bijection.py style): [k1, v1, ..., kL, vL, q] of
    length 2L+1, shared key/value vocab, explicit query.

Hypothesis indices are always 0..V-1 (value identity v); for sepvocab the
corresponding token id is V + v.

All generation is seeded and deterministic.
"""

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass
class JLensBatch:
    """A batch of annotated bijection sequences.

    Attributes:
        tokens: (B, T) input token ids.
        fmt: "sepvocab" or "paired".
        V: domain size (hypothesis count; vocab may be 2V).
        perms: (B, V) the bijection per sequence: perms[b, k] = pi(k).
        keys: (B, L) context key order per sequence.
        query: (B,) query key: the final key position's key (sepvocab) or
            the explicit query token (paired).
        eliminated: (B, T, V) bool; eliminated[b, i, v] = True iff value v
            has been revealed in a *completed* key-value pair by position i
            (inclusive). Ground truth for the P3 surviving-set probes.
        n_observed: (B, T) int; number of completed pairs by position i.
        bayes_query: (B, V) analytic posterior over hypotheses at the final
            (query) position.
        answer: (B,) hypothesis index pi(query) in 0..V-1.
    """

    tokens: torch.Tensor
    fmt: str
    V: int
    perms: torch.Tensor
    keys: torch.Tensor
    query: torch.Tensor
    eliminated: torch.Tensor
    n_observed: torch.Tensor
    bayes_query: torch.Tensor
    answer: torch.Tensor

    @property
    def B(self) -> int:
        return self.tokens.shape[0]

    @property
    def T(self) -> int:
        return self.tokens.shape[1]

    @property
    def L(self) -> int:
        return self.keys.shape[1]

    def key_positions(self) -> np.ndarray:
        """Positions where the model predicts the upcoming value (the
        wind-tunnel posterior positions)."""
        if self.fmt == "sepvocab":
            return np.arange(0, self.T, 2)
        return np.array([self.T - 1])  # paired: only the query position

    def value_positions(self) -> np.ndarray:
        return np.arange(1, self.T, 2)

    def to(self, device) -> "JLensBatch":
        moved = {
            name: getattr(self, name).to(device)
            for name in (
                "tokens",
                "perms",
                "keys",
                "query",
                "eliminated",
                "n_observed",
                "bayes_query",
                "answer",
            )
        }
        return JLensBatch(fmt=self.fmt, V=self.V, **moved)


def generate_batch(
    B: int,
    V: int = 20,
    L: int = 19,
    seed: int = 0,
    fmt: str = "sepvocab",
    fixed_keys: Optional[List[int]] = None,
) -> JLensBatch:
    """Generate B annotated bijection sequences (keys without replacement).

    Args:
        B: Number of sequences.
        V: Domain size.
        L: Number of keys in context.
        seed: RNG seed (python random; independent of global state).
        fmt: "sepvocab" | "paired" (see module docstring).
        fixed_keys: If given, every sequence uses this exact key order
            (evidence-matched batches for P4 donor pairing).
    """
    assert fmt in ("sepvocab", "paired")
    rng = random.Random(seed)
    T = 2 * L - 1 if fmt == "sepvocab" else 2 * L + 1

    tokens = torch.zeros(B, T, dtype=torch.long)
    perms = torch.zeros(B, V, dtype=torch.long)
    keys_out = torch.zeros(B, L, dtype=torch.long)
    query_out = torch.zeros(B, dtype=torch.long)
    eliminated = torch.zeros(B, T, V, dtype=torch.bool)
    n_observed = torch.zeros(B, T, dtype=torch.long)
    bayes_query = torch.zeros(B, V, dtype=torch.float64)
    answer = torch.zeros(B, dtype=torch.long)

    for b in range(B):
        perm = list(range(V))
        rng.shuffle(perm)

        if fixed_keys is not None:
            assert len(fixed_keys) == L
            keys = list(fixed_keys)
        else:
            keys = rng.sample(range(V), L)

        if fmt == "sepvocab":
            query = keys[-1]  # final key position is the deepest query
            seq = []
            for t, k in enumerate(keys):
                seq.append(k)
                if t < L - 1:
                    seq.append(V + perm[k])
            # Completed pairs by position: pair t completes at its value
            # position 2t+1. The final key (position 2L-2) has L-1
            # observed pairs.
            observed_pairs = keys[: L - 1]
        else:
            query = rng.choice(keys)
            seq = []
            for k in keys:
                seq.append(k)
                seq.append(perm[k])
            seq.append(query)
            observed_pairs = keys

        tokens[b] = torch.tensor(seq)
        perms[b] = torch.tensor(perm)
        keys_out[b] = torch.tensor(keys)
        query_out[b] = query
        answer[b] = perm[query]

        elim = np.zeros(V, dtype=bool)
        count = 0
        for i in range(T):
            if i % 2 == 1:  # value position of pair t = (i - 1) // 2
                elim[perm[keys[(i - 1) // 2]]] = True
                count += 1
            eliminated[b, i] = torch.from_numpy(elim)
            n_observed[b, i] = count

        # Analytic posterior at the query position.
        observed_map = {k: perm[k] for k in observed_pairs}
        if query in observed_map:
            bayes_query[b, observed_map[query]] = 1.0
        else:
            observed_vals = set(observed_map.values())
            remaining = [v for v in range(V) if v not in observed_vals]
            bayes_query[b, remaining] = 1.0 / len(remaining)

    return JLensBatch(
        tokens=tokens,
        fmt=fmt,
        V=V,
        perms=perms,
        keys=keys_out,
        query=query_out,
        eliminated=eliminated,
        n_observed=n_observed,
        bayes_query=bayes_query,
        answer=answer,
    )
