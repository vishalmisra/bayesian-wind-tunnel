"""
Visualize SULA control conditions (main + lexical remap + shuffled + evidence ablation)
for a single model as PC1 trajectories vs. k.

For a given model/safe-model and list of tags, this script:
  - Loads per-condition NPZ arrays (values, k_values) from results/icl_sula/{safe_model}/.
  - Builds a shared PCA over final-layer value vectors across all conditions.
  - Projects each prompt's final-layer value onto PC1.
  - For each condition and k, computes mean PC1 and bootstrap 95% CI.
  - Plots mean ± CI vs. k with one curve per condition.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


RESULTS_ROOT = Path("results/icl_sula")


def mean_ci_bootstrap(
    arr: np.ndarray, n_boot: int = 1000, ci: float = 0.95
) -> Tuple[float, float, float]:
    """
    Bootstrap mean and central (ci) interval.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        m = float(arr[0])
        return m, m, m
    rng = np.random.default_rng(0)
    means = []
    n = arr.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(arr[idx].mean()))
    means = np.array(means, dtype=float)
    m = float(means.mean())
    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(means, alpha))
    high = float(np.quantile(means, 1.0 - alpha))
    return m, low, high


def load_condition_arrays(
    safe_model: str, tag: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load values and k_values for a given model/condition tag.

    By convention:
      - main condition: icl_sula_arrays_{safe_model}.npz
      - others:         icl_sula_arrays_{tag}_{safe_model}.npz
    """
    model_dir = RESULTS_ROOT / safe_model
    if tag in {"main", "sula_main"}:
        path = model_dir / f"icl_sula_arrays_{safe_model}.npz"
    else:
        path = model_dir / f"icl_sula_arrays_{tag}_{safe_model}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing arrays for tag='{tag}', model_safe='{safe_model}': {path}")
    arrs = np.load(path)
    values = arrs["values"]      # [n_prompts, n_layers, n_heads, d_head]
    k_values = arrs["k_values"]  # [n_prompts]
    return values, k_values


def build_shared_pca_pc1(
    safe_model: str, tags: List[str]
) -> Tuple[PCA, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Build a PCA (PC1) over final-layer value vectors across all specified conditions.

    Returns:
      pca: fitted PCA
      flat_values: dict[tag] -> flattened final-layer values [n_prompts, d_v]
      k_values:    dict[tag] -> k_values [n_prompts]
    """
    flat_values: Dict[str, np.ndarray] = {}
    k_values_dict: Dict[str, np.ndarray] = {}
    all_flat: List[np.ndarray] = []

    for tag in tags:
        values, k_vals = load_condition_arrays(safe_model, tag)
        final_layer = values[:, -1]  # [n_prompts, n_heads, d_head]
        n_prompts = final_layer.shape[0]
        flat = final_layer.reshape(n_prompts, -1)  # [n_prompts, d_v]
        flat_values[tag] = flat
        k_values_dict[tag] = k_vals.astype(int)
        all_flat.append(flat)

    X = np.concatenate(all_flat, axis=0)
    pca = PCA(n_components=1)
    pca.fit(X)
    return pca, flat_values, k_values_dict


def condition_label_and_color(tag: str) -> Tuple[str, str]:
    """
    Map canonical tags to human-readable labels and colors.
    Defaults to matplotlib cycle if tag is unrecognized.
    """
    tag_norm = tag.lower()
    if tag_norm in {"main", "sula_main"}:
        return "Main", "C0"  # blue
    if "lexical" in tag_norm:
        return "Lexical remap", "C1"  # orange
    if "shuffled" in tag_norm:
        return "Shuffled labels", "C2"  # green
    if "ablation" in tag_norm:
        return "Evidence ablation", "C3"  # red
    # Fallback
    return tag, None


def plot_sula_controls_for_model(
    safe_model: str,
    tags: List[str],
    output_path: Path,
) -> None:
    pca, flat_values, k_values_dict = build_shared_pca_pc1(safe_model, tags)
    pc1_vec = pca.components_[0]  # [d_v]

    # Compute PC1 scores per condition
    pc1_scores: Dict[str, np.ndarray] = {}
    for tag in tags:
        flat = flat_values[tag]
        pc1_scores[tag] = flat @ pc1_vec  # [n_prompts]

    # Determine global k grid
    all_k = sorted(
        {int(k) for tag in tags for k in k_values_dict[tag].tolist()}
    )

    plt.figure(figsize=(6, 4))

    for tag in tags:
        scores = pc1_scores[tag]
        k_vals = k_values_dict[tag]
        label, color = condition_label_and_color(tag)
        means: List[float] = []
        lows: List[float] = []
        highs: List[float] = []
        for k in all_k:
            mask = k_vals == k
            if not np.any(mask):
                means.append(float("nan"))
                lows.append(float("nan"))
                highs.append(float("nan"))
                continue
            arr_k = scores[mask]
            m, lo, hi = mean_ci_bootstrap(arr_k)
            means.append(m)
            lows.append(lo)
            highs.append(hi)
        ks = np.array(all_k, dtype=float)
        means_arr = np.array(means, dtype=float)
        lows_arr = np.array(lows, dtype=float)
        highs_arr = np.array(highs, dtype=float)
        if color is None:
            plt.plot(ks, means_arr, marker="o", label=label)
            plt.fill_between(ks, lows_arr, highs_arr, alpha=0.2)
        else:
            plt.plot(ks, means_arr, marker="o", color=color, label=label)
            plt.fill_between(ks, lows_arr, highs_arr, color=color, alpha=0.2)

    plt.xlabel("Number of in-context examples (k)")
    plt.ylabel("PC1 coordinate (final-layer values)")
    plt.title(f"SULA controls: {safe_model}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize SULA control trajectories for a single model.")
    ap.add_argument(
        "--safe-model",
        type=str,
        required=True,
        help="Filesystem-safe model name (e.g., EleutherAI_pythia-410m).",
    )
    ap.add_argument(
        "--tags",
        type=str,
        required=True,
        help="Comma-separated list of condition tags (e.g., main,lexical_remap,shuffled,ablation).",
    )
    ap.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to output PDF.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    safe_model = args.safe_model
    tags = [t for t in args.tags.split(",") if t]
    output_path = Path(args.output_path)
    plot_sula_controls_for_model(safe_model, tags, output_path)


if __name__ == "__main__":
    main()



