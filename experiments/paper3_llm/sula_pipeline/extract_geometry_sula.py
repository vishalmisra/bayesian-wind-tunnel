"""
Extract geometry and predictions for the SULA ICL prompts.

Inputs:
  - icl_sula_prompts.json (labels + prompts)
Outputs per model under results/icl_sula/{model_safe}/:
  - icl_sula_results_{model_safe}.jsonl
  - icl_sula_arrays_{model_safe}.npz
  - manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.icl_bayesian.hf_activations import HFExtractor, InterventionConfig

PROMPTS_DEFAULT = Path("icl_sula_prompts.json")
RESULTS_ROOT = Path("results/icl_sula")
LOGGER = logging.getLogger("icl_extract_sula")

MODELS = [
    "EleutherAI/pythia-410m",
    "microsoft/phi-2",
    "meta-llama/Llama-3.2-1B",
]


def safe_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def load_sula_prompts(path: Path) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    obj = json.loads(path.read_text())
    labels = obj["labels"]
    prompts = obj["prompts"]
    return labels, prompts


def compute_posterior_and_entropy_from_probs(
    probs: np.ndarray, label_ids: Tuple[int, int]
) -> Tuple[Dict[str, float], float]:
    """
    Compute model posterior over {L1, L2} and entropy from full-vocab probabilities.

    We restrict and renormalize to the two label tokens so results are comparable
    to the earlier logit-based implementation.
    """
    id1, id2 = label_ids
    p1_raw = float(probs[id1])
    p2_raw = float(probs[id2])
    denom = p1_raw + p2_raw + 1e-12
    p1 = p1_raw / denom
    p2 = p2_raw / denom
    entropy = -((p1 * np.log2(p1 + 1e-12)) + (p2 * np.log2(p2 + 1e-12)))
    return {"L1": float(p1), "L2": float(p2)}, float(entropy)


def run_model(
    model_name: str,
    labels: Dict[str, str],
    prompts: List[Dict[str, Any]],
    prompt_path: Path,
    tag: str,
    layer_idx: int | None = None,
    layer_indices: List[int] | None = None,
    intervention_type: str = "none",
    axis_source: str = "true",
    lambda_scale: float = 0.0,
    delta_sigma: float = 0.0,
    axis_path: Path | None = None,
    seed: int = 0,
) -> None:
    if intervention_type not in {"none", "axis_cut", "axis_only", "axis_shift"}:
        raise ValueError(f"Unsupported intervention_type={intervention_type}")
    if axis_source not in {"true", "random"}:
        raise ValueError(f"Unsupported axis_source={axis_source}")

    # Resolve axis path if using a true axis and caller did not explicitly pass one.
    model_safe = safe_name(model_name)
    if intervention_type != "none" and axis_source == "true" and axis_path is None:
        # Default to SULA entropy-axis location (model-entropy aligned).
        # For multi-layer interventions, we pass the model directory and let
        # the runtime pick per-layer files (entropy_axis_L{layer}_model.npz).
        if layer_indices is not None:
            axis_path = RESULTS_ROOT / model_safe
        else:
            axis_path = RESULTS_ROOT / model_safe / f"entropy_axis_L{layer_idx}_model.npz"

    if intervention_type == "none":
        cfg = InterventionConfig(value_mode="none", key_mode="none")
    else:
        cfg = InterventionConfig(
            value_mode=intervention_type,
            key_mode="none",
            layer_idx=layer_idx if layer_indices is None else None,
            layer_indices=layer_indices,
            axis_source=axis_source,
            axis_path=str(axis_path) if axis_path is not None else None,
            lambda_scale=lambda_scale,
            delta_sigma=delta_sigma,
            seed=seed,
        )

    extractor = HFExtractor(model_name, intervention=cfg)
    tokenizer = extractor.tokenizer

    # Resolve token IDs for the two labels (use leading space form)
    L1 = labels["L1"]
    L2 = labels["L2"]
    ids1 = tokenizer.encode(" " + L1, add_special_tokens=False)
    ids2 = tokenizer.encode(" " + L2, add_special_tokens=False)
    if len(ids1) != 1 or len(ids2) != 1:
        raise RuntimeError(f"Label tokenization not single-token: {L1}->{ids1}, {L2}->{ids2}")
    label_ids = (ids1[0], ids2[0])

    values_records: List[np.ndarray] = []
    keys_records: List[np.ndarray] = []
    attn_records: List[np.ndarray] = []
    model_results: List[Dict[str, Any]] = []

    for rec in prompts:
        tokens = extractor.tokenize(rec["prompt_text"])
        values_arr, keys_arr, attn_full, probs = extractor._run_and_collect(tokens)
        # Slice attention to the final query position to match previous shape
        seq_len = attn_full.shape[-1]
        final_pos = seq_len - 1
        attn = attn_full[:, :, final_pos, :]  # [layers, heads, seq]

        model_post, model_entropy = compute_posterior_and_entropy_from_probs(
            probs, label_ids
        )

        model_results.append(
            {
                "model": model_name,
                "k": rec["k"],
                "prompt_idx": rec["prompt_idx"],
                "prompt_text": rec["prompt_text"],
                "test_word": rec["test_word"],
                "examples": rec["examples"],
                "true_label": rec["true_label"],
                "bayes_entropy": rec["bayes_entropy"],
                "bayes_posterior": rec["bayes_posterior"],
                "model_entropy": model_entropy,
                "model_posterior": model_post,
                "values_shape": values_arr.shape,
                "keys_shape": keys_arr.shape,
                "attention_shape": attn.shape,
            }
        )

        values_records.append(values_arr)
        keys_records.append(keys_arr)
        attn_records.append(attn)

    model_safe = safe_name(model_name)
    out_dir = RESULTS_ROOT / model_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    # For backward compatibility, keep unsuffixed filenames for the main condition.
    if tag == "main":
        jsonl_path = out_dir / f"icl_sula_results_{model_safe}.jsonl"
        arrays_path = out_dir / f"icl_sula_arrays_{model_safe}.npz"
    else:
        jsonl_path = out_dir / f"icl_sula_results_{tag}_{model_safe}.jsonl"
        arrays_path = out_dir / f"icl_sula_arrays_{tag}_{model_safe}.npz"
    with jsonl_path.open("w") as f:
        for entry in model_results:
            f.write(json.dumps(entry) + "\n")

    max_seq_len = max(arr.shape[-1] for arr in attn_records)
    attn_padded = []
    for arr in attn_records:
        if arr.shape[-1] < max_seq_len:
            pad_width = ((0, 0), (0, 0), (0, max_seq_len - arr.shape[-1]))
            arr = np.pad(arr, pad_width)
        attn_padded.append(arr)
    attn_stack = np.stack(attn_padded)

    np.savez_compressed(
        arrays_path,
        values=np.stack(values_records),
        keys=np.stack(keys_records),
        attention=attn_stack,
        k_values=np.array([r["k"] for r in model_results]),
        bayes_entropies=np.array([r["bayes_entropy"] for r in model_results]),
        model_entropies=np.array([r["model_entropy"] for r in model_results]),
    )

    manifest = {
        "model": model_name,
        "prompt_file": str(prompt_path),
        "labels": labels,
        "num_prompts": len(model_results),
        "values_shape": np.stack(values_records).shape,
        "keys_shape": np.stack(keys_records).shape,
        "attention_shape": attn_stack.shape,
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract geometry for SULA ICL prompts.")
    ap.add_argument(
        "--prompts",
        type=str,
        default=str(PROMPTS_DEFAULT),
        help="Path to SULA prompt file (JSON). Defaults to icl_sula_prompts.json.",
    )
    ap.add_argument(
        "--models",
        type=str,
        default="EleutherAI/pythia-410m,microsoft/phi-2,meta-llama/Llama-3.2-1B",
        help="Comma-separated list of models to process.",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default="main",
        help="Condition tag (e.g. main, lexical_remap, shuffled, ablation).",
    )
    ap.add_argument(
        "--layer-idx",
        type=int,
        default=-1,
        help="Layer index L* (0-based) where interventions apply. Defaults to final layer (-1).",
    )
    ap.add_argument(
        "--layers",
        type=str,
        default="",
        help="Optional comma-separated list of layer indices for multi-layer interventions.",
    )
    ap.add_argument(
        "--intervention-type",
        type=str,
        default="none",
        choices=["none", "axis_cut", "axis_only", "axis_shift"],
        help="Value-side intervention type to apply at L*.",
    )
    ap.add_argument(
        "--axis-source",
        type=str,
        default="true",
        choices=["true", "random"],
        help="Which axis to use: 'true' (precomputed entropy axis) or 'random'.",
    )
    ap.add_argument(
        "--lambda-scale",
        type=float,
        default=0.0,
        help="Scaling factor for axis_cut (0.0 = full removal, 1.0 = no-op).",
    )
    ap.add_argument(
        "--delta-sigma",
        type=float,
        default=0.0,
        help="Shift in units of sigma_pc1 for axis_shift (e.g. +1, -1).",
    )
    ap.add_argument(
        "--axis-path",
        type=str,
        default="",
        help="Optional explicit path to an entropy axis .npz file. "
        "If empty, defaults to results/icl_sula/{model_safe}/entropy_axis_L{layer_idx}_model.npz.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random axis or key rotations.",
    )
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    args = parse_args()
    prompt_path = Path(args.prompts)
    labels, prompts = load_sula_prompts(prompt_path)
    axis_path_arg = Path(args.axis_path) if args.axis_path else None

    # Optional multi-layer interventions: parse comma-separated list of layer ids.
    layer_indices: List[int] | None = None
    layers_str = args.layers.strip()
    if layers_str:
        layer_indices = [int(tok) for tok in layers_str.split(",") if tok]

    for model_name in args.models.split(","):
        LOGGER.info("Processing %s (%d prompts)", model_name, len(prompts))
        # For single-layer interventions, layer_idx selects L*.
        # For multi-layer interventions, layer_indices selects the set.
        layer_idx = args.layer_idx
        if layer_indices is not None:
            layer_idx = None
        elif layer_idx == -1:
            # We do not need shapes yet; just mark as "final layer" using -1.
            layer_idx = -1
        run_model(
            model_name,
            labels,
            prompts,
            prompt_path,
            tag=args.tag,
            layer_idx=layer_idx,
             layer_indices=layer_indices,
            intervention_type=args.intervention_type,
            axis_source=args.axis_source,
            lambda_scale=args.lambda_scale,
            delta_sigma=args.delta_sigma,
            axis_path=axis_path_arg,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()



