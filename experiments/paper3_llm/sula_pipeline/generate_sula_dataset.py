"""
Generate SULA (Semantically Unrelated Label Assignment) ICL prompts.

Creates:
  - icl_sula_prompts.json
with fields:
  k, prompt_idx, prompt_text, test_word, examples, true_label,
  bayes_posterior (over {L1, L2}), bayes_entropy.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from transformers import AutoTokenizer

POSITIVE_WORDS = [
    "happy",
    "joyful",
    "excited",
    "cheerful",
    "delighted",
    "pleased",
    "content",
    "satisfied",
    "glad",
    "thrilled",
    "wonderful",
    "excellent",
    "fantastic",
    "amazing",
    "great",
    "beautiful",
    "lovely",
    "pleasant",
    "enjoyable",
    "terrific",
    "brilliant",
    "fabulous",
    "superb",
    "marvelous",
    "splendid",
    "peaceful",
    "calm",
    "relaxed",
    "serene",
    "tranquil",
    "optimistic",
    "hopeful",
    "confident",
    "proud",
    "grateful",
]

NEGATIVE_WORDS = [
    "sad",
    "angry",
    "upset",
    "annoyed",
    "frustrated",
    "disappointed",
    "unhappy",
    "miserable",
    "depressed",
    "gloomy",
    "terrible",
    "awful",
    "horrible",
    "dreadful",
    "bad",
    "ugly",
    "unpleasant",
    "disagreeable",
    "nasty",
    "disgusting",
    "worried",
    "anxious",
    "nervous",
    "stressed",
    "tense",
    "tired",
    "exhausted",
    "weary",
    "fatigued",
    "drained",
    "bitter",
    "resentful",
    "hostile",
    "hateful",
    "cruel",
]

assert len(POSITIVE_WORDS) == len(NEGATIVE_WORDS) == 35

ALPHA: float = 0.9

CANDIDATE_LABELS = [
    "dax",
    "lug",
    "mip",
    "zorb",
    "wug",
    "kep",
    "tuv",
    "gax",
]

MODEL_NAMES = [
    "EleutherAI/pythia-410m",
    "microsoft/phi-2",
    "meta-llama/Llama-3.2-1B",
]


def find_shared_single_token_labels() -> Tuple[str, str]:
    tokenizers = {m: AutoTokenizer.from_pretrained(m, use_fast=True) for m in MODEL_NAMES}
    shared: List[str] = []
    for label in CANDIDATE_LABELS:
        ok = True
        for tok in tokenizers.values():
            ids = tok.encode(" " + label, add_special_tokens=False)
            if len(ids) != 1:
                ok = False
                break
        if ok:
            shared.append(label)
    if len(shared) < 2:
        # Fallback: systematic CVC sweep to find more labels
        consonants = "bcdfghjklmnpqrstvwxyz"
        vowels = "aeiou"
        for c1 in consonants:
            for v in vowels:
                for c2 in consonants:
                    lab = f"{c1}{v}{c2}"
                    if lab in shared or lab in CANDIDATE_LABELS:
                        continue
                    ok = True
                    for tok in tokenizers.values():
                        ids = tok.encode(" " + lab, add_special_tokens=False)
                        if len(ids) != 1:
                            ok = False
                            break
                    if ok:
                        shared.append(lab)
                        if len(shared) >= 2:
                            break
                if len(shared) >= 2:
                    break
            if len(shared) >= 2:
                break
    if len(shared) < 2:
        raise RuntimeError(f"Could not find two shared single-token labels, found={shared}")
    return shared[0], shared[1]


def entropy_bits(prob_dict: Dict[str, float]) -> float:
    probs = np.array(list(prob_dict.values()), dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def compute_bayesian_posterior_iid(
    examples: List[Tuple[str, str]],
    L1: str,
    L2: str,
    true_label: str,
    alpha: float = ALPHA,
) -> Dict[str, float]:
    """
    Bayesian posterior over {L1, L2} for the *test word's* latent label.

    Generative model:
      - Prior: P(y = L1) = P(y = L2) = 0.5
      - For each example ℓ_i:
            P(ℓ_i = y | y)   = alpha
            P(ℓ_i ≠ y | y)   = 1 - alpha
      - Carrier words are ignored by the Bayes model (decorative only).
    """
    # Count how many example labels match / mismatch the true latent label
    n_match = sum(1 for _, lab in examples if lab == true_label)
    n_mismatch = len(examples) - n_match

    # Log-likelihoods under y = true_label and y = other_label
    log_like_true = n_match * np.log(alpha) + n_mismatch * np.log(1.0 - alpha)
    log_like_other = n_mismatch * np.log(alpha) + n_match * np.log(1.0 - alpha)

    # Prior is uniform, so posterior depends only on likelihood ratio
    max_ll = max(log_like_true, log_like_other)
    p_true_u = np.exp(log_like_true - max_ll)
    p_other_u = np.exp(log_like_other - max_ll)
    p_true = p_true_u / (p_true_u + p_other_u)
    p_other = 1.0 - p_true

    if true_label == L1:
        p_L1, p_L2 = p_true, p_other
    elif true_label == L2:
        p_L1, p_L2 = p_other, p_true
    else:
        raise RuntimeError("true_label must be L1 or L2")

    return {L1: float(p_L1), L2: float(p_L2)}


def generate_sula_icl_prompts_monotone(
    k_shots: List[int], n_prompts_per_k: int, L1: str, L2: str, seed: int = 42
) -> List[Dict]:
    """
    SULA prompts where each example label is an i.i.d. noisy glimpse of a
    single latent label y for the test word.

    This makes the *expected* Bayesian entropy monotonically decreasing in k.
    """
    random.seed(seed)
    np.random.seed(seed)
    all_words = POSITIVE_WORDS + NEGATIVE_WORDS  # carrier vocabulary

    prompts: List[Dict] = []
    for k in k_shots:
        for prompt_idx in range(n_prompts_per_k):
            # 1. Choose test word
            test_word = random.choice(all_words)

            # 2. Choose latent true label for test word
            true_label = random.choice([L1, L2])

            # 3. Sample k examples: arbitrary words + noisy labels around true_label
            examples: List[Tuple[str, str]] = []
            for _ in range(k):
                w = random.choice(all_words)
                if random.random() < ALPHA:
                    lab = true_label
                else:
                    lab = L2 if true_label == L1 else L1
                examples.append((w, lab))

            if k == 0:
                prompt_text = f"{test_word}:"
            else:
                ex_str = ", ".join(f"{w}: {lab}" for w, lab in examples)
                prompt_text = f"{ex_str}. {test_word}:"

            bayes_post = compute_bayesian_posterior_iid(
                examples, L1, L2, true_label, alpha=ALPHA
            )
            bayes_H = entropy_bits(bayes_post)

            prompts.append(
                {
                    "k": k,
                    "prompt_idx": prompt_idx,
                    "prompt_text": prompt_text,
                    "test_word": test_word,
                    "examples": examples,
                    "true_label": true_label,
                    "bayes_posterior": bayes_post,
                    "bayes_entropy": bayes_H,
                }
            )
    return prompts


def main() -> None:
    L1, L2 = find_shared_single_token_labels()
    print(f"Using SULA labels: L1={L1}, L2={L2}")
    k_shots = [0, 1, 2, 4, 8]
    prompts = generate_sula_icl_prompts_monotone(
        k_shots, n_prompts_per_k=50, L1=L1, L2=L2, seed=42
    )
    out_path = Path("icl_sula_prompts.json")
    with out_path.open("w") as f:
        json.dump(
            {
                "labels": {"L1": L1, "L2": L2},
                "prompts": prompts,
            },
            f,
            indent=2,
        )
    print(f"✅ Wrote {len(prompts)} SULA prompts with labels {L1}, {L2} to {out_path}")


if __name__ == "__main__":
    main()


