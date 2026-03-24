"""
Compute SULA main-condition manifold metrics and write them to the new results tree.

This script reuses the existing SULA geometry outputs under results/icl_sula/{model_safe}/
and produces:

  results/sula/{model_safe}/main.json
  results/sula/{model_safe}/manifold_main.png

For each model we:
  - Load icl_sula_arrays_{model_safe}.npz
  - Take final-layer value vectors at the last position
  - Run per-model PCA (2 components)
  - Use PC1 as a Bayesian axis:
      * report corr(PC1, Bayes entropy)
      * compute PC1 mean/std by k
      * store these metrics in main.json
  - Plot:
      * PC1 trajectory vs k (with error bars) overlaid across models
      * 2D PCA scatter (PC1 vs PC2) colored by Bayes entropy and marked by k
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from typing import Tuple; import numpy as np

def mean_ci(arr):
    m = np.mean(arr)
    se = np.std(arr) / np.sqrt(len(arr))
    return m, m - 1.96*se, m + 1.96*se

def safe_name(model_name):
    return model_name.replace('/', '_').replace('-', '_')




PROJECT_ROOT = Path(__file__).resolve().parents[2]
SULA_RESULTS_ROOT = PROJECT_ROOT / "results" / "icl_sula"
SULA_NEW_ROOT = PROJECT_ROOT / "results" / "sula"


def load_sula_arrays(model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_safe = safe_name(model_name)
    path = SULA_RESULTS_ROOT / model_safe / f"icl_sula_arrays_{model_safe}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing SULA arrays for {model_name} at {path}")
    arrs = np.load(path)
    values = arrs["values"]          # [n_prompts, n_layers, d_flat]
    k_values = arrs["k_values"]      # [n_prompts]
    bayes_entropies = arrs["bayes_entropies"]  # [n_prompts]
    return values, k_values, bayes_entropies


def per_model_pca(values: np.ndarray, bayes_entropies: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Run PCA on final-layer value vectors and return coords and corr(PC1, entropy).
    """
    n_prompts = values.shape[0]
    final_layer = values[:, -1]  # [n_prompts, d_flat]
    flat = final_layer.reshape(n_prompts, -1)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(flat)
    pc1 = coords[:, 0]
    if pc1.std() > 0 and bayes_entropies.std() > 0:
        corr = float(np.corrcoef(pc1, bayes_entropies)[0, 1])
    else:
        corr = float("nan")
    # Align sign so that correlation is non-negative where defined
    if np.isfinite(corr) and corr < 0:
        coords[:, 0] = -coords[:, 0]
        corr = -corr
    return coords, corr


def compute_pc1_trajectory(
    coords: np.ndarray, k_values: np.ndarray, bayes_entropies: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """
    Compute PC1 mean/std and Bayes entropy mean per k.
    """
    pc1 = coords[:, 0]
    traj: Dict[int, Dict[str, float]] = {}
    for k in sorted(set(k_values.tolist())):
        mask = k_values == k
        vals = pc1[mask]
        ents = bayes_entropies[mask]
        if vals.size == 0:
            continue
        mean_pc1 = float(vals.mean())
        std_pc1 = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        mean_ent = float(ents.mean())
        traj[int(k)] = {
            "pc1_mean": mean_pc1,
            "pc1_std": std_pc1,
            "entropy_mean": mean_ent,
            "n": int(vals.size),
        }
    return traj


def plot_pc1_trajectories(
    models: List[str],
    per_model_traj: Dict[str, Dict[int, Dict[str, float]]],
    out_path: Path,
) -> None:
    plt.figure(figsize=(7, 4))
    colors = plt.cm.tab10.colors
    for idx, model_name in enumerate(models):
        traj = per_model_traj[model_name]
        ks = sorted(traj.keys())
        means = [traj[k]["pc1_mean"] for k in ks]
        stds = [traj[k]["pc1_std"] for k in ks]
        color = colors[idx % len(colors)]
        plt.errorbar(
            ks,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            color=color,
            label=model_name,
        )
    plt.xlabel("k shots")
    plt.ylabel("PC1 (Bayesian axis)")
    plt.title("SULA main: manifold trajectory by k")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_global_scatter(
    models: List[str],
    coords_by_model: Dict[str, np.ndarray],
    k_by_model: Dict[str, np.ndarray],
    ent_by_model: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    plt.figure(figsize=(6, 5))
    markers = ["o", "s", "D", "^", "x"]

    all_coords = []
    all_k = []
    all_ent = []
    for model_name in models:
        all_coords.append(coords_by_model[model_name])
        all_k.append(k_by_model[model_name])
        all_ent.append(ent_by_model[model_name])
    all_coords = np.concatenate(all_coords, axis=0)
    all_k = np.concatenate(all_k, axis=0)
    all_ent = np.concatenate(all_ent, axis=0)

    ks_unique = sorted(np.unique(all_k).tolist())
    for k_idx, k in enumerate(ks_unique):
        mask = all_k == k
        sc = plt.scatter(
            all_coords[mask, 0],
            all_coords[mask, 1],
            c=all_ent[mask],
            cmap="viridis",
            marker=markers[k_idx % len(markers)],
            s=30,
            edgecolors="none",
            label=f"k={k}",
        )
    cbar = plt.colorbar(sc)
    cbar.set_label("Bayes entropy (bits)")
    plt.xticks([])
    plt.yticks([])
    plt.title("SULA main: global value manifold (per-model PCA concat)")
    plt.legend(title="k", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SULA main-condition manifolds into results/sula tree.")
    parser.add_argument(
        "--models",
        type=str,
        default="EleutherAI/pythia-410m,microsoft/phi-2,meta-llama/Llama-3.2-1B",
        help="Comma-separated list of model names to process.",
    )
    args = parser.parse_args()

    models = args.models.split(",")
    coords_by_model: Dict[str, np.ndarray] = {}
    k_by_model: Dict[str, np.ndarray] = {}
    ent_by_model: Dict[str, np.ndarray] = {}
    per_model_traj: Dict[str, Dict[int, Dict[str, float]]] = {}

    for model_name in models:
        values, k_values, bayes_entropies = load_sula_arrays(model_name)
        coords, corr_pc1_ent = per_model_pca(values, bayes_entropies)
        traj = compute_pc1_trajectory(coords, k_values, bayes_entropies)

        coords_by_model[model_name] = coords
        k_by_model[model_name] = k_values
        ent_by_model[model_name] = bayes_entropies
        per_model_traj[model_name] = traj

        model_safe = safe_name(model_name)
        out_dir = SULA_NEW_ROOT / model_safe
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": model_name,
            "model_safe": model_safe,
            "pc1_entropy_corr": corr_pc1_ent,
            "per_k": {int(k): traj[int(k)] for k in sorted(traj.keys())},
        }
        with (out_dir / "main.json").open("w") as f:
            json.dump(payload, f, indent=2)

    # Shared trajectory panel across models
    traj_out = SULA_NEW_ROOT / "manifold_main_trajectory.png"
    plot_pc1_trajectories(models, per_model_traj, traj_out)

    # Global scatter (concat all per-model PCAs just for visualization)
    scatter_out = SULA_NEW_ROOT / "manifold_main_global.png"
    plot_global_scatter(models, coords_by_model, k_by_model, ent_by_model, scatter_out)


if __name__ == "__main__":
    main()







