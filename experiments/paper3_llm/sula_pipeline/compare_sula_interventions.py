"""
Compare baseline vs. intervention SULA runs for a single model.

This helper:
  - Loads baseline + intervention JSONL / NPZ outputs.
  - Summarizes entropy calibration (MAE / corr) overall and by k.
  - Optionally computes correlation between an entropy-aligned axis u_ent
    (precomputed from baseline) and entropy under each condition.

It is designed to be lightweight and text-only; figures are handled elsewhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


RESULTS_ROOT = Path("results/icl_sula")


def safe_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def gather_paths(model_name: str, tag: str) -> Tuple[Path, Path]:
    model_safe = safe_name(model_name)
    model_dir = RESULTS_ROOT / model_safe
    if tag == "main":
        jsonl_path = model_dir / f"icl_sula_results_{model_safe}.jsonl"
        arrays_path = model_dir / f"icl_sula_arrays_{model_safe}.npz"
    else:
        jsonl_path = model_dir / f"icl_sula_results_{tag}_{model_safe}.jsonl"
        arrays_path = model_dir / f"icl_sula_arrays_{tag}_{model_safe}.npz"
    return jsonl_path, arrays_path


def compute_mae_and_corr(model_entropies: np.ndarray, bayes_entropies: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(model_entropies - bayes_entropies)))
    if len(model_entropies) > 1 and bayes_entropies.std() > 0:
        corr = float(np.corrcoef(model_entropies, bayes_entropies)[0, 1])
    else:
        corr = float("nan")
    return {"mae": mae, "correlation": corr}


def summarize_entropy_calibration(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    ent_model = np.array([r["model_entropy"] for r in records], dtype=float)
    ent_bayes = np.array([r["bayes_entropy"] for r in records], dtype=float)
    calib: Dict[str, Dict[str, float]] = {}
    k_values = sorted({r["k"] for r in records})
    for k in k_values:
        mask = np.array([r["k"] for r in records]) == k
        if not mask.any():
            continue
        sub_model = ent_model[mask]
        sub_bayes = ent_bayes[mask]
        calib[str(k)] = compute_mae_and_corr(sub_model, sub_bayes)
    calib["all"] = compute_mae_and_corr(ent_model, ent_bayes)
    return calib


def compute_axis_entropy_corr(
    values: np.ndarray,
    entropies: np.ndarray,
    axis_path: Path,
    layer_idx: int,
) -> float:
    """
    Compute corr(v·u_ent, entropy) at a given layer using a precomputed axis file.
    """
    data = np.load(axis_path)
    if "u_ent" not in data:
        raise FileNotFoundError(f"Axis file {axis_path} missing 'u_ent'")
    u_ent = np.asarray(data["u_ent"], dtype=np.float32).reshape(-1)

    if layer_idx < 0:
        layer_idx = values.shape[1] - 1
    V_layer = values[:, layer_idx]  # [N, n_heads, d_head]
    N, n_heads, d_head = V_layer.shape
    V = V_layer.reshape(N, n_heads * d_head)
    if V.shape[1] != u_ent.shape[0]:
        raise ValueError(
            f"Axis dim {u_ent.shape[0]} does not match flattened value dim {V.shape[1]}"
        )
    coeff = V @ u_ent  # [N]
    ent = np.asarray(entropies, dtype=np.float64).reshape(-1)
    if coeff.std() == 0.0 or ent.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(coeff, ent)[0, 1])


def compare_conditions(
    model_name: str,
    baseline_tag: str,
    intervention_tags: List[str],
    layer_idx: int,
    axis_path: Path | None,
    axis_entropy_source: str,
) -> None:
    model_safe = safe_name(model_name)
    print(f"\n=== SULA comparison for {model_name} (safe={model_safe}) ===")
    print(f"Baseline tag: {baseline_tag}")
    print(f"Interventions: {intervention_tags}")

    # Load baseline
    base_jsonl, base_arrays = gather_paths(model_name, baseline_tag)
    base_records = load_jsonl(base_jsonl)
    base_npz = np.load(base_arrays)
    base_values = base_npz["values"]
    base_bayes_ent = base_npz["bayes_entropies"]
    base_model_ent = base_npz["model_entropies"]

    base_calib = summarize_entropy_calibration(base_records)

    # Optional axis correlation for baseline
    base_axis_corr = None
    if axis_path is not None and axis_path.exists():
        if axis_entropy_source == "bayes":
            ent_ref = base_bayes_ent
        else:
            ent_ref = base_model_ent
        base_axis_corr = compute_axis_entropy_corr(
            base_values, ent_ref, axis_path, layer_idx
        )

    print("\n--- Baseline entropy calibration (model vs Bayes) ---")
    for k in sorted([k for k in base_calib.keys() if k != "all"], key=int):
        stats = base_calib[k]
        print(f"k={k:>2}: MAE={stats['mae']:.4f}, corr={stats['correlation']:.4f}")
    stats_all = base_calib["all"]
    print(f"ALL : MAE={stats_all['mae']:.4f}, corr={stats_all['correlation']:.4f}")
    if base_axis_corr is not None:
        print(f"Baseline axis corr (v·u_ent vs {axis_entropy_source} entropy): {base_axis_corr:.4f}")

    # Interventions
    for tag in intervention_tags:
        print(f"\n--- Intervention: {tag} ---")
        int_jsonl, int_arrays = gather_paths(model_name, tag)
        int_records = load_jsonl(int_jsonl)
        int_npz = np.load(int_arrays)
        int_values = int_npz["values"]
        int_bayes_ent = int_npz["bayes_entropies"]
        int_model_ent = int_npz["model_entropies"]

        int_calib = summarize_entropy_calibration(int_records)

        # Axis correlation under intervention
        int_axis_corr = None
        if axis_path is not None and axis_path.exists():
            if axis_entropy_source == "bayes":
                ent_ref_int = int_bayes_ent
            else:
                ent_ref_int = int_model_ent
            int_axis_corr = compute_axis_entropy_corr(
                int_values, ent_ref_int, axis_path, layer_idx
            )

        # Print calibration and deltas vs baseline
        for k in sorted([k for k in base_calib.keys() if k != "all"], key=int):
            base_stats = base_calib[k]
            int_stats = int_calib.get(k)
            if int_stats is None:
                continue
            d_mae = int_stats["mae"] - base_stats["mae"]
            d_corr = int_stats["correlation"] - base_stats["correlation"]
            print(
                f"k={k:>2}: MAE={int_stats['mae']:.4f} "
                f"(Δ={d_mae:+.4f}), corr={int_stats['correlation']:.4f} "
                f"(Δ={d_corr:+.4f})"
            )
        base_all = base_calib["all"]
        int_all = int_calib["all"]
        d_mae_all = int_all["mae"] - base_all["mae"]
        d_corr_all = int_all["correlation"] - base_all["correlation"]
        print(
            f"ALL : MAE={int_all['mae']:.4f} (Δ={d_mae_all:+.4f}), "
            f"corr={int_all['correlation']:.4f} (Δ={d_corr_all:+.4f})"
        )
        if int_axis_corr is not None and base_axis_corr is not None:
            d_axis = int_axis_corr - base_axis_corr
            print(
                f"Axis corr (v·u_ent vs {axis_entropy_source} entropy): "
                f"{int_axis_corr:.4f} (Δ={d_axis:+.4f} vs baseline)"
            )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare baseline vs. intervention SULA runs for a single model."
    )
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (matching SULA outputs).",
    )
    ap.add_argument(
        "--baseline-tag",
        type=str,
        default="main",
        help="Baseline tag (default: main).",
    )
    ap.add_argument(
        "--intervention-tags",
        type=str,
        required=True,
        help="Comma-separated list of intervention tags to compare against baseline.",
    )
    ap.add_argument(
        "--layer-idx",
        type=int,
        default=-1,
        help="Layer index L* (0-based) for axis correlation. -1 = final layer.",
    )
    ap.add_argument(
        "--axis-path",
        type=str,
        default="",
        help="Optional explicit path to entropy axis .npz file. "
        "If empty, defaults to results/icl_sula/{model_safe}/entropy_axis_L{layer_idx}_model.npz.",
    )
    ap.add_argument(
        "--axis-entropy-source",
        type=str,
        default="model",
        choices=["model", "bayes"],
        help="Which entropy source u_ent was aligned to (for axis correlation reporting).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model
    intervention_tags = [t for t in args.intervention_tags.split(",") if t]
    if not intervention_tags:
        raise ValueError("No intervention tags provided via --intervention-tags")

    model_safe = safe_name(model_name)
    layer_idx = args.layer_idx

    if args.axis_path:
        axis_path = Path(args.axis_path)
    else:
        # Default: entropy axis aligned to model entropy at L*
        axis_suffix = "model" if args.axis_entropy_source == "model" else args.axis_entropy_source
        axis_path = (
            RESULTS_ROOT
            / model_safe
            / f"entropy_axis_L{layer_idx}_{axis_suffix}.npz"
        )

    compare_conditions(
        model_name=model_name,
        baseline_tag=args.baseline_tag,
        intervention_tags=intervention_tags,
        layer_idx=layer_idx,
        axis_path=axis_path if axis_path.exists() else None,
        axis_entropy_source=args.axis_entropy_source,
    )


if __name__ == "__main__":
    main()




