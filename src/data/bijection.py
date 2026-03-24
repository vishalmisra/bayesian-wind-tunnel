"""
Bijection learning data generation for Bayesian wind tunnel experiments.

This module implements the bijection learning task where a model must learn
to invert a random bijection (permutation) from in-context examples.

The analytic posterior is:
    P(π(x_k) = y | context) = 1/(V - k + 1) if y not yet observed, else 0

Reference: "The Bayesian Geometry of Transformer Attention" (Paper I), Section 2.3
"""

import random
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def sample_permutation(V: int) -> List[int]:
    """Sample a random permutation (bijection) on V elements."""
    arr = list(range(V))
    random.shuffle(arr)
    return arr


def build_sequence(
    perm: List[int],
    L: int,
    with_replacement: bool = True,
    query_from_context: bool = True,
    predict_all_values: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a bijection learning sequence.
    
    Sequence format: [x₁, y(x₁), x₂, y(x₂), ..., xₗ, y(xₗ), x_query]
    
    Args:
        perm: The bijection (permutation) mapping keys to values
        L: Number of key-value pairs in context
        with_replacement: If True, keys can repeat; if False, unique keys
        query_from_context: If True, query key is sampled from context keys
        predict_all_values: If True, supervise all value positions (recommended)
    
    Returns:
        x: Input sequence of shape (2L + 1,)
        y: Target sequence with IGNORE_INDEX at non-supervised positions
    """
    V = len(perm)
    
    # Sample context keys
    if with_replacement:
        keys = [random.randrange(V) for _ in range(L)]
    else:
        keys = random.sample(range(V), min(L, V))
    
    # Sample query key
    if query_from_context and len(keys) > 0:
        query = random.choice(keys)
    else:
        query = random.randrange(V)
    
    # Build sequence: [k₁, v₁, k₂, v₂, ..., kₗ, vₗ, q]
    seq = []
    for k in keys:
        seq.append(k)
        seq.append(perm[k])
    seq.append(query)
    
    x = torch.tensor(seq, dtype=torch.long)
    y = torch.full((2 * L + 1,), IGNORE_INDEX, dtype=torch.long)
    
    # Supervise value positions
    if predict_all_values:
        for i, k in enumerate(keys):
            y[2 * i + 1] = perm[k]
    
    # Always supervise the query position
    y[-1] = perm[query]
    
    return x, y


class BijectionDataset(Dataset):
    """
    Dataset for bijection learning with fresh permutations per sample.
    
    This is the "ChangingDict" setting where each sample uses a new
    random bijection, requiring genuine in-context learning.
    """
    def __init__(
        self,
        V: int,
        L: int,
        n_samples: int,
        with_replacement: bool = True,
        query_from_context: bool = True,
        predict_all_values: bool = True,
        seed: Optional[int] = None,
    ):
        self.V = V
        self.L = L
        self.n_samples = n_samples
        self.with_replacement = with_replacement
        self.query_from_context = query_from_context
        self.predict_all_values = predict_all_values
        
        if seed is not None:
            random.seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        perm = sample_permutation(self.V)
        return build_sequence(
            perm, self.L,
            with_replacement=self.with_replacement,
            query_from_context=self.query_from_context,
            predict_all_values=self.predict_all_values,
        )


class FixedBijectionDataset(Dataset):
    """
    Dataset with a fixed bijection (for testing memorization vs ICL).
    """
    def __init__(
        self,
        V: int,
        L: int,
        n_samples: int,
        with_replacement: bool = True,
        query_from_context: bool = True,
        predict_all_values: bool = True,
        seed: int = 1337,
    ):
        self.V = V
        self.L = L
        self.n_samples = n_samples
        self.with_replacement = with_replacement
        self.query_from_context = query_from_context
        self.predict_all_values = predict_all_values
        
        # Fixed permutation
        rng = random.Random(seed)
        self.perm = list(range(V))
        rng.shuffle(self.perm)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_sequence(
            self.perm, self.L,
            with_replacement=self.with_replacement,
            query_from_context=self.query_from_context,
            predict_all_values=self.predict_all_values,
        )


class MixedBijectionDataset(Dataset):
    """
    Mixed dataset: p fraction uses fresh bijections, (1-p) uses fixed.
    
    Used to study the transition between memorization and ICL.
    """
    def __init__(
        self,
        V: int,
        L: int,
        n_samples: int,
        p_changing: float = 0.5,
        with_replacement: bool = True,
        query_from_context: bool = True,
        predict_all_values: bool = True,
        seed: int = 1337,
    ):
        self.V = V
        self.L = L
        self.n_samples = n_samples
        self.p_changing = p_changing
        self.with_replacement = with_replacement
        self.query_from_context = query_from_context
        self.predict_all_values = predict_all_values
        
        # Fixed permutation
        rng = random.Random(seed)
        self.fixed_perm = list(range(V))
        rng.shuffle(self.fixed_perm)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p_changing:
            perm = sample_permutation(self.V)
        else:
            perm = self.fixed_perm
        
        return build_sequence(
            perm, self.L,
            with_replacement=self.with_replacement,
            query_from_context=self.query_from_context,
            predict_all_values=self.predict_all_values,
        )

# Compatibility aliases
sample_perm = sample_permutation
build_sequence_from_perm = build_sequence
