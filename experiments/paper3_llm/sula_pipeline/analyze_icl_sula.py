"""
Analyze SULA ICL geometry outputs (entropy calibration, manifold stats, trajectories).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEFAULT_MODELS = [
    "EleutherAI/pythia-410m",
    "microsoft/phi-2",
    "meta-llama/Llama-3.2-1B",
]

RESULTS_ROOT = Path("results/icl_sula")


def safe_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def compute_mae_and_corr(model_entropies: np.ndarray, bayes_entropies: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(model_entropies - bayes_entropies)))
    if len(model_entropies) > 1 and bayes_entropies.std() > 0:
        corr = float(np.corrcoef(model_entropies, bayes_entropies)[0, 1])
    else:
        corr = float("nan")
    return {"mae": mae, "correlation": corr}


def compute_entropy_calibration(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    ent_model = np.array([r["model_entropy"] for r in results])
    ent_bayes = np.array([r["bayes_entropy"] for r in results])
    calib: Dict[str, Dict[str, float]] = {}
    k_values = sorted({r["k"] for r in results})
    for k in k_values:
        mask = np.array([r["k"] for r in results]) == k
        if not mask.any():
            continue
        sub_model = ent_model[mask]
        sub_bayes = ent_bayes[mask]
        calib[str(k)] = compute_mae_and_corr(sub_model, sub_bayes)
    return calib


def compute_pca_metrics(values: np.ndarray, bayes_entropies: np.ndarray) -> Dict[str, Any]:
    n_prompts = values.shape[0]
    final_layer = values[:, -1]
    flattened = final_layer.reshape(n_prompts, -1)
    
    # Canonical Protocol: Standardize features (zero mean, unit variance)
    scaler = StandardScaler()
    flattened_std = scaler.fit_transform(flattened)
    
    pca = PCA(n_components=10) # Computed top 10 for spectral analysis
    coords = pca.fit_transform(flattened_std)
    
    # Participation Ratio (PR) on eigenvalues
    # lambda_i = variance explained by component i
    eigenvalues = pca.explained_variance_
    pr_num = np.sum(eigenvalues) ** 2
    pr_den = np.sum(eigenvalues ** 2)
    participation_ratio = pr_num / (pr_den + 1e-12)

    pc1 = coords[:, 0]
    pc2 = coords[:, 1]
    pc1_corr = float(np.corrcoef(pc1, bayes_entropies)[0, 1]) if len(pc1) > 1 else float("nan")
    pc2_corr = float(np.corrcoef(pc2, bayes_entropies)[0, 1]) if len(pc2) > 1 else float("nan")
    
    # Bootstrap 95% CI for explained variance
    n_boot = 1000
    rng = np.random.default_rng(42)
    var_ratios_boot = []
    for _ in range(n_boot):
        indices = rng.integers(0, n_prompts, n_prompts)
        X_boot = flattened_std[indices]
        pca_boot = PCA(n_components=2)
        pca_boot.fit(X_boot)
        var_ratios_boot.append(pca_boot.explained_variance_ratio_)
    
    var_ratios_boot = np.array(var_ratios_boot)
    pc1_ci = np.percentile(var_ratios_boot[:, 0], [2.5, 97.5])
    pc2_ci = np.percentile(var_ratios_boot[:, 1], [2.5, 97.5])
    sum_ci = np.percentile(var_ratios_boot[:, 0] + var_ratios_boot[:, 1], [2.5, 97.5])

    return {
        "pc1_coords": pc1.tolist(),
        "pc2_coords": pc2.tolist(),
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_variance": float(pca.explained_variance_ratio_[1]),
        "pc1_variance_ci": pc1_ci.tolist(),
        "pc2_variance_ci": pc2_ci.tolist(),
        "pc1_pc2_variance_sum": float(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]),
        "pc1_pc2_variance_sum_ci": sum_ci.tolist(),
        "participation_ratio": float(participation_ratio),
        "top_eigenvalues": eigenvalues.tolist(),
        "pc1_entropy_corr": pc1_corr,
        "pc2_entropy_corr": pc2_corr,
        "entropies": bayes_entropies.tolist(),
        "pca_coords": coords[:, :2].tolist(),
    }


def compute_manifold_trajectory(pc1_coords: List[float], bayes_entropies: List[float], k_values: List[int]) -> Dict[str, Dict[str, Any]]:
    traj: Dict[str, Dict[str, Any]] = {}
    pc1 = np.array(pc1_coords)
    bayes_ent = np.array(bayes_entropies)
    ks = np.array(k_values)
    for k in sorted(set(ks.tolist())):
        mask = ks == k
        if not mask.any():
            continue
        vals = pc1[mask]
        ent = bayes_ent[mask]
        traj[str(k)] = {
            "pc1_mean": float(np.mean(vals)),
            "pc1_std": float(np.std(vals)),
            "entropy_mean": float(np.mean(ent)),
        }
    return traj


def compute_key_orthogonality(keys: np.ndarray) -> Dict[str, Any]:
    # keys: [n_prompts, n_layers, n_heads, d_head]
    if keys.ndim != 4:
        return {"scores": [0.0 for _ in range(keys.shape[1])]}
    
    n_prompts, n_layers, n_heads, d_head = keys.shape
    orth_scores: List[float] = []
    
    for layer in range(n_layers):
        per_prompt: List[float] = []
        for i in range(n_prompts):
            layer_keys = keys[i, layer]
            norms = np.linalg.norm(layer_keys, axis=1, keepdims=True)
            normalized = layer_keys / (norms + 1e-12)
            sim = normalized @ normalized.T
            # Mean off-diagonal absolute cosine similarity
            mask = ~np.eye(n_heads, dtype=bool)
            off_diag = np.mean(np.abs(sim[mask]))
            per_prompt.append(float(off_diag))
        orth_scores.append(float(np.mean(per_prompt)))

    # Theoretical baselines
    # E[|cos theta|] for random unit vectors in d dimensions: sqrt(2 / (pi * d))
    theoretical_gaussian = float(np.sqrt(2 / (np.pi * d_head)))
    empirical_init_min = 0.35
    empirical_init_max = 0.45

    return {
        "scores": orth_scores,
        "baseline_gaussian_theoretical": theoretical_gaussian,
        "baseline_initialization_empirical_range": [empirical_init_min, empirical_init_max],
        "d_head": d_head
    }


def compute_attention_entropy(attn: np.ndarray) -> List[float]:
    # attn: [n_prompts, n_layers, n_heads, seq_len]
    n_prompts, n_layers, n_heads, seq_len = attn.shape
    entropies: List[float] = []
    for layer in range(n_layers):
        layer_vals: List[float] = []
        for i in range(n_prompts):
            # Calculate entropy per head, then average
            head_entropies = []
            for h in range(n_heads):
                dist = attn[i, layer, h] # [seq_len]
                dist_prob = dist / (dist.sum() + 1e-12)
                H = -np.sum(dist_prob * np.log2(dist_prob + 1e-12))
                head_entropies.append(H)
            
            # Average entropy across heads for this prompt
            layer_vals.append(float(np.mean(head_entropies)))
        
        # Average across prompts
        entropies.append(float(np.mean(layer_vals)))
    return entropies


def analyze_model(model_name: str, tag: str) -> None:
    model_safe = safe_name(model_name)
    model_dir = RESULTS_ROOT / model_safe
    if tag == "main":
        jsonl_path = model_dir / f"icl_sula_results_{model_safe}.jsonl"
        arrays_path = model_dir / f"icl_sula_arrays_{model_safe}.npz"
    else:
        jsonl_path = model_dir / f"icl_sula_results_{tag}_{model_safe}.jsonl"
        arrays_path = model_dir / f"icl_sula_arrays_{tag}_{model_safe}.npz"
    if not jsonl_path.exists() or not arrays_path.exists():
        raise FileNotFoundError(f"Missing SULA data for {model_name}")
    records = load_jsonl(jsonl_path)
    arrays = np.load(arrays_path)
    values = arrays["values"]
    keys = arrays["keys"]
    attention = arrays["attention"]
    k_values = arrays["k_values"].tolist()
    bayes_entropies = arrays["bayes_entropies"].tolist()

    ent_calib = compute_entropy_calibration(records)
    calib_snapshot = {k: ent_calib[k] for k in sorted(ent_calib.keys(), key=int)}

    value_manifold = compute_pca_metrics(values, np.array(bayes_entropies))
    value_manifold["k_values"] = k_values
    manifold_traj = compute_manifold_trajectory(
        value_manifold["pc1_coords"], bayes_entropies, k_values
    )
    keys_orth_data = compute_key_orthogonality(keys)
    attn_entropy = compute_attention_entropy(attention)

    analysis = {
        "model": model_name,
        "num_prompts": values.shape[0],
        "entropy_calibration_by_k": calib_snapshot,
        "value_manifold": {
            **value_manifold,
            "values_shape": values.shape,
            "k_values": k_values,
        },
        "manifold_trajectory": manifold_traj,
        "key_orthogonality_by_layer": keys_orth_data["scores"],
        "key_orthogonality_metadata": keys_orth_data,
        "attention_entropy_by_layer": attn_entropy,
    }

    out_path = model_dir / f"icl_sula_analysis_{model_safe}.json"
    with out_path.open("w") as f:
        json.dump(analysis, f, indent=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze SULA ICL geometry outputs.")
    ap.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated list of model names to analyze.",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default="main",
        help="Condition tag used for this run (main, shuffled, axis_cut, etc.).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    for model_name in args.models.split(","):
        analyze_model(model_name, tag=args.tag)


if __name__ == "__main__":
    main()



