"""
Entropy computation utilities for Bayesian wind tunnel experiments.

Provides functions to compute:
- Model predictive entropy from logits
- Bayesian posterior entropy for bijection tasks
- Mean absolute error (MAE) between model and Bayes-optimal entropy
"""

import math
from typing import List, Tuple, Optional
import numpy as np
import torch

LOG2E = 1.0 / math.log(2.0)


def entropy_bits_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy in bits from logits.
    
    Args:
        logits: Tensor of shape (..., vocab_size)
        
    Returns:
        Entropy in bits with same leading dimensions
    """
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    entropy_nats = -(probs * log_probs).sum(dim=-1)
    return entropy_nats * LOG2E


def entropy_bits_from_probs(probs: np.ndarray) -> float:
    """Compute entropy in bits from probability distribution."""
    probs = np.asarray(probs)
    probs = probs[probs > 0]  # Avoid log(0)
    return float(-np.sum(probs * np.log2(probs)))


def bayes_bijection_posterior(
    V: int,
    observed_pairs: List[Tuple[int, int]],
    query_key: int,
) -> np.ndarray:
    """
    Compute Bayesian posterior for bijection task.
    
    Given observed key-value pairs, compute P(π(query_key) = y) for all y.
    
    Args:
        V: Vocabulary size (domain size)
        observed_pairs: List of (key, value) pairs observed so far
        query_key: The key for which we want to predict the value
        
    Returns:
        Posterior distribution over values, shape (V,)
    """
    # Check if query key was already observed
    for key, val in observed_pairs:
        if key == query_key:
            # Already know the answer
            posterior = np.zeros(V)
            posterior[val] = 1.0
            return posterior
    
    # Find which values are still possible (not yet assigned to other keys)
    observed_values = set(val for _, val in observed_pairs)
    remaining = [v for v in range(V) if v not in observed_values]
    
    if len(remaining) == 0:
        # All values assigned - shouldn't happen with valid bijection
        return np.ones(V) / V
    
    # Uniform over remaining values
    posterior = np.zeros(V)
    p = 1.0 / len(remaining)
    for v in remaining:
        posterior[v] = p
    
    return posterior


def bayes_bijection_entropy(k: int, V: int) -> float:
    """
    Analytic Bayesian entropy at position k in bijection task.
    
    After observing k-1 key-value pairs with unique keys:
        H(k) = log₂(V - k + 1)
    
    Args:
        k: Position in sequence (1-indexed)
        V: Vocabulary size
        
    Returns:
        Bayesian posterior entropy in bits
    """
    remaining = V - k + 1
    if remaining <= 0:
        return 0.0
    return math.log2(remaining)


@torch.no_grad()
def evaluate_entropy_calibration(
    model,
    V: int,
    L: int,
    device: torch.device,
    n_samples: int = 1000,
    with_replacement: bool = False,
    seed: int = 42,
) -> dict:
    """
    Evaluate entropy calibration: compare model entropy to Bayes-optimal.
    
    Args:
        model: TinyGPT model
        V: Vocabulary size
        L: Context length
        device: Torch device
        n_samples: Number of sequences to evaluate
        with_replacement: Key sampling mode
        seed: Random seed
        
    Returns:
        Dictionary with MAE, per-position entropy curves, etc.
    """
    import random
    random.seed(seed)
    
    model.eval()
    
    ent_model_sum = torch.zeros(L, dtype=torch.float64, device=device)
    ent_bayes_sum = torch.zeros(L, dtype=torch.float64, device=device)
    
    for _ in range(n_samples):
        # Sample random permutation
        perm = list(range(V))
        random.shuffle(perm)
        
        # Sample keys
        if with_replacement:
            keys = [random.randrange(V) for _ in range(L)]
        else:
            keys = list(range(V))
            random.shuffle(keys)
            keys = keys[:L]
        
        # Track observations for Bayes posterior
        observed_pairs: List[Tuple[int, int]] = []
        
        for t in range(1, L + 1):
            # Build prefix: [k₁, v₁, ..., k_{t-1}, v_{t-1}, k_t]
            seq = []
            for i in range(t - 1):
                seq.append(keys[i])
                seq.append(perm[keys[i]])
            seq.append(keys[t - 1])
            
            x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)
            logits_last = logits[0, -1, :]
            
            # Model entropy
            ent = entropy_bits_from_logits(logits_last.unsqueeze(0))[0]
            ent_model_sum[t - 1] += ent.double()
            
            # Bayes entropy
            bayes_post = bayes_bijection_posterior(V, observed_pairs, keys[t - 1])
            hb = entropy_bits_from_probs(bayes_post)
            ent_bayes_sum[t - 1] += hb
            
            # Update observations
            observed_pairs.append((keys[t - 1], perm[keys[t - 1]]))
    
    ent_model_mean = (ent_model_sum / n_samples).cpu().numpy()
    ent_bayes_mean = (ent_bayes_sum / n_samples).cpu().numpy()
    mae = float(np.mean(np.abs(ent_model_mean - ent_bayes_mean)))
    
    return {
        "mae_bits": mae,
        "model_entropy": ent_model_mean.tolist(),
        "bayes_entropy": ent_bayes_mean.tolist(),
        "n_samples": n_samples,
        "V": V,
        "L": L,
        "with_replacement": with_replacement,
    }
