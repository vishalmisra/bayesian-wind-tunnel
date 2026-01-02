"""
Geometric analysis utilities for Bayesian wind tunnel experiments.

Provides tools for analyzing the geometric structure of transformer representations:
- Value manifold PCA
- Key orthogonality measurement
- Attention entropy analysis

Reference: "The Bayesian Geometry of Transformer Attention" (Paper I), Sections 5-6
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr


def measure_key_orthogonality(key_matrix: np.ndarray) -> float:
    """
    Measure orthogonality of key vectors.
    
    Computes mean absolute off-diagonal cosine similarity.
    Lower values indicate more orthogonal keys.
    
    Args:
        key_matrix: Key vectors of shape (n_tokens, d_head)
        
    Returns:
        Mean off-diagonal cosine similarity (0 = perfectly orthogonal)
    """
    # Normalize to unit vectors
    norms = np.linalg.norm(key_matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    normalized = key_matrix / norms
    
    # Compute cosine similarity matrix
    sim = normalized @ normalized.T
    
    # Mean absolute off-diagonal
    n = sim.shape[0]
    if n <= 1:
        return 0.0
    
    mask = ~np.eye(n, dtype=bool)
    off_diag = np.abs(sim[mask])
    
    return float(np.mean(off_diag))


def analyze_value_manifold(
    value_vectors: np.ndarray,
    entropies: np.ndarray,
    n_components: int = 2,
) -> Dict:
    """
    Analyze value manifold structure via PCA.
    
    Args:
        value_vectors: Value representations of shape (n_samples, d_value)
        entropies: Corresponding entropy values of shape (n_samples,)
        n_components: Number of PCA components
        
    Returns:
        Dictionary with PCA results, correlations, and metrics
    """
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(value_vectors)
    
    # PCA
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)
    
    # Variance explained
    var_explained = pca.explained_variance_ratio_
    
    # Correlation of PC1 with entropy
    pc1_entropy_corr, _ = spearmanr(coords[:, 0], entropies)
    
    # Align PC1 sign so positive correlation with entropy
    if pc1_entropy_corr < 0:
        coords[:, 0] = -coords[:, 0]
        pc1_entropy_corr = -pc1_entropy_corr
    
    return {
        "coordinates": coords,
        "variance_explained": var_explained.tolist(),
        "pc1_variance": float(var_explained[0]),
        "pc1_pc2_variance": float(sum(var_explained[:2])),
        "pc1_entropy_correlation": float(pc1_entropy_corr) if not np.isnan(pc1_entropy_corr) else 0.0,
        "pca_model": pca,
    }


def attention_entropy(attn_weights: np.ndarray) -> float:
    """
    Compute entropy of attention distribution.
    
    Args:
        attn_weights: Attention weights of shape (..., seq_len)
        
    Returns:
        Mean entropy in bits
    """
    # Flatten to (n_distributions, seq_len)
    flat = attn_weights.reshape(-1, attn_weights.shape[-1])
    
    # Compute entropy for each distribution
    eps = 1e-10
    flat = np.clip(flat, eps, 1.0)
    entropy = -np.sum(flat * np.log2(flat), axis=-1)
    
    return float(np.mean(entropy))


def measure_manifold_dimensionality(
    representations: np.ndarray,
    threshold: float = 0.95,
) -> int:
    """
    Measure effective dimensionality of representation manifold.
    
    Returns the number of PCA components needed to explain threshold variance.
    
    Args:
        representations: Data of shape (n_samples, d)
        threshold: Cumulative variance threshold
        
    Returns:
        Number of dimensions for threshold variance
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(representations)
    
    pca = PCA()
    pca.fit(X)
    
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_dims = int(np.searchsorted(cumvar, threshold) + 1)
    
    return n_dims


def compute_qk_alignment(
    queries: np.ndarray,
    keys: np.ndarray,
) -> Dict:
    """
    Measure query-key alignment across layers.
    
    Args:
        queries: Query vectors of shape (n_samples, d)
        keys: Key vectors of shape (n_samples, d)
        
    Returns:
        Dictionary with alignment metrics
    """
    # Normalize
    q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    k_norm = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarities
    cosine_sims = np.sum(q_norm * k_norm, axis=1)
    
    return {
        "mean_alignment": float(np.mean(cosine_sims)),
        "std_alignment": float(np.std(cosine_sims)),
        "min_alignment": float(np.min(cosine_sims)),
        "max_alignment": float(np.max(cosine_sims)),
    }
