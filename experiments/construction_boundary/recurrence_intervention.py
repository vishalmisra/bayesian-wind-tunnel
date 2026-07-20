"""
Round 6 Reviewer Ask A1: Counterfactual Intervention Test
==========================================================

Tests whether a trained recurrence-BWT transformer (p=17 integer) has
learned the *mechanism* x_{t+1} = a*x_t + b mod p, versus merely a
*support-local* trajectory predictor that continues from the observed
(unintervened) future.

Protocol
--------
1. Load a trained checkpoint produced by
   experiments/construction_boundary/recurrence_bwt.py
2. For N trials:
     - sample (a, b, x_0) ~ Uniform(Z_p)^3 and generate the unintervened
       trajectory x_0, x_1, ..., x_T using the existing sample_recurrence().
     - skip degenerate trials with x_0 == x_1 (program underdetermined).
     - for each intervention position t in {3,4,5,6,7}:
         for each z in Z_p with z != x_t_unintervened:
             - build intervened prefix [x_0, ..., x_{t-1}, z]
             - feed through model -> P_model(x_{t+1} | intervened prefix)
             - mechanism target  : a*z + b mod p     (point mass)
             - trajectory target : a*x_t + b mod p   (next obs without intervention)
             - Bayes oracle (mechanism, pi=0.5, x_0!=x_1 implies (a,b)
               determined): P_HP = point mass on a*z+b, P_HR = uniform.
               At position t>=3 with mixture, the H_P weight w is
               essentially 1, so the Bayes-optimal post-intervention
               prediction is the mechanism target.

Reported metrics
----------------
- mechanism_acc       : fraction argmax(P_model) == a*z+b mod p
- trajectory_acc      : fraction argmax(P_model) == a*x_t+b mod p
- KL_model_mechanism  : E[ KL(P_model || mechanism point mass) ] in bits
                        (cross-entropy minus 0; equals -log2 P_model(a*z+b))
- KL_model_bayes      : E[ KL(P_model || Bayes-oracle) ] in bits.
                        The Bayes oracle is the mixture w*delta_{a*z+b} +
                        (1-w)*uniform. We use this rather than a strict point
                        mass to avoid -infs when the model leaves a sliver
                        of mass off the mechanism target.

Outputs
-------
A single JSON file at <output_dir>/intervention_results.json containing:
  - config (p, n_trials, intervention_positions, checkpoint, seed)
  - aggregate metrics
  - per-position breakdown
  - per-distance breakdown (|z - x_t_unintervened| mod p)
  - plot-ready arrays: per-trial mechanism_target_prob,
    trajectory_target_prob, kl_to_bayes, position, distance.

Usage
-----
    python -u recurrence_intervention.py \
        --checkpoint /home/vishal/bayesg/Model-Selection-BWT/experiments/results/recurrence/integer/seed_42/best_model.pt \
        --p 17 --n_trials 2000 --device cuda:0 \
        --output_dir /home/vishal/bayesg/Nature-Paper-Round6/intervention_p17/seed_42
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch

# Import the source training script. It lives at
# experiments/construction_boundary/recurrence_bwt.py on tokenprobe.
# We import-by-path so this script can be dropped next to it OR run from
# anywhere as long as --source_dir points at the parent of that file.
def _import_source(source_dir):
    if source_dir is None:
        candidates = [
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'construction_boundary'),
        ]
        for c in candidates:
            if os.path.isfile(os.path.join(c, 'recurrence_bwt.py')):
                source_dir = c
                break
    if source_dir is None or not os.path.isfile(
            os.path.join(source_dir, 'recurrence_bwt.py')):
        raise FileNotFoundError(
            "Could not locate recurrence_bwt.py. Pass --source_dir.")
    sys.path.insert(0, source_dir)
    import recurrence_bwt as src  # noqa: E402
    return src


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, p, src, device,
               d_model=192, n_layers=6, n_heads=6, d_ff=768, dropout=0.0):
    """Build the RecurrenceTransformer and load the checkpoint.

    Integer-mode defaults match the checkpoints at
        Model-Selection-BWT/experiments/results/recurrence/integer/seed_*/best_model.pt
    (token_embed [18,192], pos_embed [512,192], output_proj [17,192],
     6 layers, qkv [576,192] -> n_heads=6, d_ff=768).
    """
    RecurrenceTransformer = src._build_model_class()
    vocab_size = p          # integer mode
    n_tokens = p
    model = RecurrenceTransformer(
        vocab_size=vocab_size,
        n_tokens=n_tokens,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        print(f"  WARN: missing={missing} unexpected={unexpected}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Intervention machinery
# ---------------------------------------------------------------------------

def model_next_token_probs(model, prefix_tokens, device):
    """Return softmax probs over Z_p at the last position of prefix_tokens.

    `prefix_tokens` is a list of ints of length k>=1. The transformer is
    autoregressive: logits at position k-1 predict token at position k.
    """
    x = torch.tensor(prefix_tokens, dtype=torch.long,
                     device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(x)
    probs = torch.softmax(logits[0, -1, :], dim=-1).cpu().numpy()
    return probs  # shape (n_tokens,)


def kl_bits(p_dist, q_dist, eps=1e-12):
    """KL(p || q) in bits. p,q are 1D numpy arrays of equal length."""
    p_dist = np.asarray(p_dist, dtype=np.float64)
    q_dist = np.asarray(q_dist, dtype=np.float64)
    mask = p_dist > eps
    return float(np.sum(p_dist[mask] *
                        (np.log2(p_dist[mask]) -
                         np.log2(np.maximum(q_dist[mask], eps)))))


def run_intervention_trial(model, src, a, b, x0, p, intervention_t, device,
                           seq_len):
    """Run one intervention trial at position intervention_t for all z != x_t.

    Returns a list of dicts (one per z) with model/bayes/mechanism/trajectory
    statistics.
    """
    # Build the unintervened trajectory of length seq_len.
    x = x0
    seq = [x]
    while len(seq) < seq_len + 1:
        x = (a * x + b) % p
        seq.append(x)

    if seq[0] == seq[1]:
        # degenerate: (a,b) not uniquely determined from the prefix x_0,x_1,x_2
        return []

    x_t_orig = seq[intervention_t]

    rows = []
    for z in range(p):
        if z == x_t_orig:
            continue
        # Intervened prefix x_0, ..., x_{t-1}, z  (length intervention_t + 1)
        intervened_prefix = list(seq[:intervention_t]) + [z]
        # The model emits logits at every position; the last position predicts
        # x_{t+1} given the intervened context.
        p_model = model_next_token_probs(model, intervened_prefix, device)

        mechanism_target = (a * z + b) % p
        trajectory_target = (a * x_t_orig + b) % p  # = x_{t+1} (orig)

        # Bayes oracle on the intervened prefix.
        # Because the prefix x_0, x_1, x_2 (which is unmodified by an
        # intervention at t>=3) uniquely determines (a,b) in the generic
        # case, the intervened prefix is *not* consistent with the recovered
        # (a,b) at position t (we changed x_t from x_t_orig to z != x_t_orig).
        # Strictly, count_consistent on the full intervened prefix would
        # therefore be 0, which under the standard model means "H_P is
        # falsified" -> Bayes oracle = uniform.
        #
        # But that is the WRONG counterfactual for our purpose. The whole
        # point of an intervention is do(x_t := z), i.e. surgery on the
        # generative graph. Under do-semantics, (a,b) are still the
        # generating parameters and the next-token distribution is the point
        # mass on a*z+b. So we compute the Bayes-oracle for the intervened
        # prefix from the unmodified prefix x_0,x_1,x_2:
        ab = src.recover_recurrence(seq[:3], p)
        assert ab is not None, "x_0 != x_1 should determine (a,b)"
        a_rec, b_rec = ab
        x_next_mechanism = (a_rec * z + b_rec) % p
        # By construction a_rec==a and b_rec==b (we generated the seq).
        # Build mixture w * delta_{x_next} + (1-w)/p . At t>=3 with pi=0.5
        # the posterior weight on H_P is essentially 1, so we approximate
        # by treating it as the H_P-only prediction. To stay defensive we
        # actually compute w from the *unmodified* prefix of length t (which
        # is fully consistent with (a,b) and yields w ~ p^{t-3}/(p^{t-3}+1)).
        bayes_w = src.class_posterior_recurrence(seq[:intervention_t], p,
                                                 pi=0.5)
        bayes_dist = np.full(p, (1.0 - bayes_w) / p, dtype=np.float64)
        bayes_dist[x_next_mechanism] += bayes_w
        bayes_dist /= bayes_dist.sum()  # guard against fp drift

        mechanism_dist = np.zeros(p, dtype=np.float64)
        mechanism_dist[mechanism_target] = 1.0

        argmax_model = int(np.argmax(p_model[:p]))
        rows.append({
            'a': int(a), 'b': int(b), 'x0': int(x0),
            't': int(intervention_t),
            'z': int(z),
            'x_t_orig': int(x_t_orig),
            'distance': int((z - x_t_orig) % p),
            'mechanism_target': int(mechanism_target),
            'trajectory_target': int(trajectory_target),
            'argmax_model': argmax_model,
            'p_model_mechanism': float(p_model[mechanism_target]),
            'p_model_trajectory': float(p_model[trajectory_target]),
            'kl_model_mechanism_bits': kl_bits(mechanism_dist, p_model[:p]),
            'kl_model_bayes_bits': kl_bits(bayes_dist, p_model[:p]),
            'bayes_w': float(bayes_w),
            'mech_eq_traj': bool(mechanism_target == trajectory_target),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Counterfactual intervention test for recurrence BWT')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(
                            os.environ.get('CKPT_DIR',
                                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        'results', 'recurrence')),
                            'integer', 'seed_42', 'best_model.pt'))
    parser.add_argument('--p', type=int, default=17)
    parser.add_argument('--n_trials', type=int, default=2000)
    parser.add_argument('--intervention_positions', type=int, nargs='+',
                        default=[3, 4, 5, 6, 7])
    parser.add_argument('--seq_len', type=int, default=16,
                        help='length of the unintervened trajectory used '
                             'to anchor x_t_orig; must be > max intervention t')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'results', 'intervention_p17'))
    parser.add_argument('--source_dir', type=str, default=None,
                        help='dir containing recurrence_bwt.py')
    # Architecture overrides (defaults match existing p=17 checkpoints)
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=768)
    args = parser.parse_args()

    assert args.seq_len > max(args.intervention_positions), \
        "seq_len must exceed max intervention position"

    src = _import_source(args.source_dir)
    src._ensure_torch()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, args.p, src, device,
                       d_model=args.d_model, n_layers=args.n_layers,
                       n_heads=args.n_heads, d_ff=args.d_ff, dropout=0.0)
    n_params = sum(pr.numel() for pr in model.parameters())
    print(f"  model: {n_params:,} params on {device}")

    p = args.p
    rows = []
    attempts = 0
    while len(rows) < args.n_trials * len(args.intervention_positions) * (p - 1):
        attempts += 1
        a = int(np.random.randint(0, p))
        b = int(np.random.randint(0, p))
        x0 = int(np.random.randint(0, p))
        # Pre-skip degenerate (a, x0) combos where x_0 == x_1 = a*x_0 + b.
        if (a * x0 + b) % p == x0:
            continue
        # Cycle through intervention positions for this (a,b,x0) sample.
        for t in args.intervention_positions:
            trial_rows = run_intervention_trial(
                model, src, a, b, x0, p, t, device, args.seq_len)
            rows.extend(trial_rows)
        if attempts % 200 == 0:
            print(f"  attempt {attempts}: {len(rows):,} rows collected")
        if attempts >= args.n_trials:
            break

    print(f"Collected {len(rows):,} rows over {attempts} (a,b,x0) samples")

    # -----------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------
    def agg(filter_fn=lambda r: True):
        sub = [r for r in rows if filter_fn(r)]
        if not sub:
            return {'n': 0}
        # Exclude rows where mechanism == trajectory (no separation between
        # the two hypotheses, so they're uninformative for the contrast).
        sep = [r for r in sub if not r['mech_eq_traj']]
        sub_n = len(sub)
        sep_n = len(sep)
        return {
            'n': sub_n,
            'n_separable': sep_n,
            'mechanism_acc_all':
                float(np.mean([r['argmax_model'] == r['mechanism_target']
                               for r in sub])),
            'trajectory_acc_all':
                float(np.mean([r['argmax_model'] == r['trajectory_target']
                               for r in sub])),
            'mechanism_acc_separable':
                float(np.mean([r['argmax_model'] == r['mechanism_target']
                               for r in sep])) if sep else None,
            'trajectory_acc_separable':
                float(np.mean([r['argmax_model'] == r['trajectory_target']
                               for r in sep])) if sep else None,
            'kl_model_mechanism_bits_mean':
                float(np.mean([r['kl_model_mechanism_bits'] for r in sub])),
            'kl_model_bayes_bits_mean':
                float(np.mean([r['kl_model_bayes_bits'] for r in sub])),
            'p_model_mechanism_mean':
                float(np.mean([r['p_model_mechanism'] for r in sub])),
            'p_model_trajectory_mean':
                float(np.mean([r['p_model_trajectory'] for r in sub])),
            'kl_model_bayes_bits_separable_mean':
                float(np.mean([r['kl_model_bayes_bits'] for r in sep]))
                if sep else None,
        }

    metrics = {
        'overall': agg(),
        'per_position': {
            str(t): agg(lambda r, t=t: r['t'] == t)
            for t in args.intervention_positions
        },
        'per_distance': {
            str(d): agg(lambda r, d=d: r['distance'] == d)
            for d in range(1, p)
        },
    }

    config = {
        'checkpoint': args.checkpoint,
        'p': args.p,
        'n_trials_requested': args.n_trials,
        'attempts': attempts,
        'intervention_positions': args.intervention_positions,
        'seq_len': args.seq_len,
        'seed': args.seed,
        'device': str(device),
        'model_params': n_params,
        'arch': {
            'd_model': args.d_model, 'n_layers': args.n_layers,
            'n_heads': args.n_heads, 'd_ff': args.d_ff,
        },
    }

    # Plot-ready compact arrays.
    plot_arrays = {
        't': [r['t'] for r in rows],
        'distance': [r['distance'] for r in rows],
        'p_model_mechanism': [r['p_model_mechanism'] for r in rows],
        'p_model_trajectory': [r['p_model_trajectory'] for r in rows],
        'kl_model_bayes_bits': [r['kl_model_bayes_bits'] for r in rows],
        'kl_model_mechanism_bits': [r['kl_model_mechanism_bits'] for r in rows],
        'mech_eq_traj': [r['mech_eq_traj'] for r in rows],
        'argmax_eq_mechanism':
            [r['argmax_model'] == r['mechanism_target'] for r in rows],
        'argmax_eq_trajectory':
            [r['argmax_model'] == r['trajectory_target'] for r in rows],
    }

    out = {
        'config': config,
        'metrics': metrics,
        'plot_arrays': plot_arrays,
    }
    out_path = os.path.join(args.output_dir, 'intervention_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("INTERVENTION TEST RESULTS")
    print("=" * 70)
    print(f"  checkpoint        : {args.checkpoint}")
    print(f"  n_rows            : {len(rows):,}")
    ov = metrics['overall']
    print(f"  mechanism_acc     : {ov['mechanism_acc_all']:.4f} (all rows)")
    print(f"  trajectory_acc    : {ov['trajectory_acc_all']:.4f} (all rows)")
    if ov['mechanism_acc_separable'] is not None:
        print(f"  mechanism_acc*    : {ov['mechanism_acc_separable']:.4f} "
              f"(rows where mech != traj)")
        print(f"  trajectory_acc*   : {ov['trajectory_acc_separable']:.4f} "
              f"(rows where mech != traj)")
    print(f"  KL(model||mech)   : {ov['kl_model_mechanism_bits_mean']:.4f} bits")
    print(f"  KL(model||bayes)  : {ov['kl_model_bayes_bits_mean']:.4f} bits "
          "  <-- headline")
    print(f"  E[P_model(mech)]  : {ov['p_model_mechanism_mean']:.4f}")
    print(f"  E[P_model(traj)]  : {ov['p_model_trajectory_mean']:.4f}")
    print("\n  Per-position (mechanism_acc* / trajectory_acc* / KL_bayes):")
    for t in args.intervention_positions:
        m = metrics['per_position'][str(t)]
        print(f"    t={t}: "
              f"{m.get('mechanism_acc_separable')}  /  "
              f"{m.get('trajectory_acc_separable')}  /  "
              f"{m.get('kl_model_bayes_bits_mean')}")
    print(f"\n  Saved -> {out_path}")


if __name__ == '__main__':
    main()
