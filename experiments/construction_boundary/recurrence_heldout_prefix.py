"""
Round 6 Reviewer Ask A2: Held-Out-Prefix Replication at Larger Modulus
=======================================================================

Replicates the support-locality finding from the manuscript at p=31
(default; the script also accepts p in {17, 23, 43}). The original p=17
result was train-KL = 0.000399 bits vs heldout-KL = 0.542 bits across
5 buckets x 3 seeds. The reviewer wants the same separation at larger
p so it cannot be dismissed as the model memorizing the 4,624 generic
p=17 prefixes.

Methodology
-----------
1. Enumerate generic three-token prefixes (x_0, x_1, x_2) with x_0 != x_1.
   For p, the count is p * (p-1) * p.
2. Hash each prefix to bucket 0..4 with
       blake2b(f"{x_0},{x_1},{x_2}".encode(), digest_size=8).intdigest() % 5
3. For each (holdout_bucket, seed) cell:
     - Train one model on episodes whose first three tokens land in any
       of the four *training* buckets, using rejection sampling during
       sequence generation. H_R (random) episodes do not have a "generic
       prefix" assignment and are kept bucket-agnostic.
     - Train for up to --n_steps or until train-MAE < 0.02 bits.
     - At the end (and at the best checkpoint by train-bucket MAE),
       evaluate on:
           20,000 episodes drawn from training buckets only,
           20,000 episodes drawn from the held-out bucket only.
       Both splits are scored against the SAME full-task Bayes oracle
       (constructed only from p and pi=0.5; the oracle does NOT know
       about the bucket split).
     - Save best checkpoint and a metrics JSON per cell.

Output layout
-------------
<output_dir>/seed_<seed>_holdout_<bucket>/
    best_model.pt
    metrics.json
    train_log.txt   (also goes to stdout via nohup)

The companion runner script run_heldout_p31_all.sh launches all 15 cells
(5 buckets * 3 seeds) across the 8 RTX 4090 GPUs.
"""

import argparse
import json
import math
import os
import sys
import time
from hashlib import blake2b

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Import the source recurrence_bwt.py for Bayes math + model class.
# ---------------------------------------------------------------------------

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
    import recurrence_bwt as src
    return src


# ---------------------------------------------------------------------------
# Bucket hashing
# ---------------------------------------------------------------------------

def prefix_bucket(x0, x1, x2, n_buckets=5):
    """Deterministic bucket assignment for a generic prefix (x_0, x_1, x_2).

    Uses blake2b for stability across processes and Python versions.
    """
    h = blake2b(f"{int(x0)},{int(x1)},{int(x2)}".encode(),
                digest_size=8).digest()
    return int.from_bytes(h, 'big') % n_buckets


def enumerate_generic_prefixes(p):
    """Iterate generic prefixes (x_0 != x_1)."""
    for x0 in range(p):
        for x1 in range(p):
            if x1 == x0:
                continue
            for x2 in range(p):
                yield x0, x1, x2


def count_buckets(p, n_buckets=5):
    counts = [0] * n_buckets
    for pref in enumerate_generic_prefixes(p):
        counts[prefix_bucket(*pref, n_buckets=n_buckets)] += 1
    return counts


# ---------------------------------------------------------------------------
# Bucket-restricted sequence generation
# ---------------------------------------------------------------------------

def generate_episode_bucket_restricted(src, cfg, train_buckets, n_buckets,
                                       rng, max_retries=2000):
    """Generate one episode whose generic prefix lives in `train_buckets`.

    H_R (random, prob 1-pi) episodes are not bucket-restricted: their
    prefix has no semantic "generic" assignment in the program sense
    (they're not from the recurrence family). We do still apply the
    bucket filter to them so that the train/heldout EVAL splits below
    can pull H_R episodes consistently. (For an H_R episode the
    bucket is just a deterministic function of its random prefix.)
    """
    p = cfg.p
    # rng here is the Generator we use to PICK a per-episode seed for the
    # source script's np.random.* calls. We can't share state with the legacy
    # np.random.MT19937, so we reseed it deterministically each episode.
    for _ in range(max_retries):
        episode_seed = int(rng.integers(0, 2**31 - 1))
        np.random.seed(episode_seed)
        tokens, gt, metadata = src.generate_recurrence_sequence(cfg)
        x0, x1, x2 = tokens[0], tokens[1], tokens[2]
        if x0 == x1:
            # degenerate prefix; the bucket scheme is defined only for
            # generic prefixes, so reject and resample (matches the
            # manuscript's "generic" restriction).
            continue
        bkt = prefix_bucket(x0, x1, x2, n_buckets=n_buckets)
        if bkt in train_buckets:
            metadata['bucket'] = bkt
            return tokens, gt, metadata
    raise RuntimeError(
        f"Failed to sample bucket-restricted episode in {max_retries} tries")


def generate_episode_for_bucket(src, cfg, target_bucket, n_buckets,
                                rng, max_retries=10000):
    """Generate one episode whose generic prefix is exactly target_bucket."""
    return generate_episode_bucket_restricted(
        src, cfg, {target_bucket}, n_buckets, rng, max_retries=max_retries)


# ---------------------------------------------------------------------------
# Evaluation against the full-task Bayes oracle
# ---------------------------------------------------------------------------

def evaluate_on_buckets(model, src, cfg, eval_buckets, n_buckets, n_eval,
                        device, rng):
    """Evaluate a trained model on episodes whose prefix lies in eval_buckets.

    Returns dict of mean KL (nats), entropy MAE (bits), TV.
    Scoring is against bayesian_predictive_recurrence with cfg.pi as the
    program prior — i.e. the full-task oracle that does NOT know about
    the bucket split.
    """
    model.eval()
    p = cfg.p
    kls, maes, tvs = [], [], []
    with torch.no_grad():
        for _ in range(n_eval):
            tokens, gt, metadata = generate_episode_bucket_restricted(
                src, cfg, eval_buckets, n_buckets, rng)
            x = torch.tensor(tokens, dtype=torch.long,
                             device=device).unsqueeze(0)
            logits, _ = model(x)
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            header_len = metadata['header_len']
            n_tok = metadata['n_tokens']
            for gt_entry in gt:
                t = gt_entry['t']
                model_pos = header_len + t - 1
                if t == 0 or model_pos < 0 or model_pos >= len(probs):
                    continue
                p_model = probs[model_pos][:n_tok]
                pred_dist = gt_entry['pred_dist']
                if cfg.opaque:
                    p_bayes = np.array(
                        [pred_dist.get(v + p, 0.0) for v in range(n_tok)])
                else:
                    p_bayes = np.array(
                        [pred_dist.get(v, 0.0) for v in range(n_tok)])

                # KL(Bayes || Model) in nats — matches the manuscript convention
                kl = float(sum(p_bayes[y] *
                               math.log(p_bayes[y] /
                                        max(p_model[y], 1e-10))
                               for y in range(n_tok) if p_bayes[y] > 1e-10))
                kls.append(kl)

                H_model = float(-sum(pm * math.log2(pm) for pm in p_model
                                     if pm > 1e-10))
                H_bayes = gt_entry['entropy']
                maes.append(abs(H_model - H_bayes))

                tv = 0.5 * float(np.sum(np.abs(p_bayes - p_model)))
                tvs.append(tv)
    return {
        'n_predictions': len(kls),
        'kl_nats_mean': float(np.mean(kls)) if kls else 0.0,
        'kl_nats_std': float(np.std(kls)) if kls else 0.0,
        'kl_bits_mean': float(np.mean(kls) / math.log(2)) if kls else 0.0,
        'entropy_mae_bits_mean': float(np.mean(maes)) if maes else 0.0,
        'entropy_mae_bits_std': float(np.std(maes)) if maes else 0.0,
        'tv_mean': float(np.mean(tvs)) if tvs else 0.0,
    }


# ---------------------------------------------------------------------------
# Training loop (bucket-restricted)
# ---------------------------------------------------------------------------

def train_heldout(args, src):
    src._ensure_torch()
    RecurrenceTransformer = src._build_model_class()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    p = args.p
    n_buckets = 5

    train_buckets = set(range(n_buckets)) - {args.holdout_bucket}

    # Diagnostics
    bucket_counts = count_buckets(p, n_buckets=n_buckets)
    total_generic = sum(bucket_counts)
    print(f"  p={p}: {total_generic} generic prefixes, "
          f"per-bucket counts = {bucket_counts}")
    print(f"  train buckets: {sorted(train_buckets)}  "
          f"holdout: {args.holdout_bucket}  seed: {args.seed}")

    # Integer mode (manuscript replication target)
    vocab_size = p
    n_tokens = p
    model = RecurrenceTransformer(
        vocab_size=vocab_size,
        n_tokens=n_tokens,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(pr.numel() for pr in model.parameters())
    print(f"  model: {n_params:,} params on {device}")

    cfg = src.RecurrenceConfig(p=p, pi=args.pi, seq_len=args.seq_len,
                               opaque=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_steps)

    PAD = vocab_size
    os.makedirs(args.output_dir, exist_ok=True)

    # RNGs:
    # rng_train drives training data, rng_eval_train/heldout drive eval.
    rng_train = np.random.default_rng(args.seed)
    rng_eval_tr = np.random.default_rng(args.seed + 100_001)
    rng_eval_ho = np.random.default_rng(args.seed + 200_002)

    best_train_mae = float('inf')
    best_step = -1
    losses = []

    early_stop_threshold = args.early_stop_mae
    eval_history = []
    start_time = time.time()

    for step in range(1, args.n_steps + 1):
        model.train()
        batch_tokens = []
        max_len = 0
        for _ in range(args.batch_size):
            tokens, _, metadata = generate_episode_bucket_restricted(
                src, cfg, train_buckets, n_buckets, rng_train)
            batch_tokens.append(tokens)
            if len(tokens) > max_len:
                max_len = len(tokens)
        padded = [t + [PAD] * (max_len - len(t)) for t in batch_tokens]
        x = torch.tensor(padded, dtype=torch.long, device=device)
        logits, _ = model(x)

        # Per-position next-token CE loss (matches source script).
        loss = torch.tensor(0.0, device=device)
        count = 0
        for b_idx in range(args.batch_size):
            header_len = len(batch_tokens[b_idx]) - args.seq_len
            seq_start = header_len
            seq_end = len(batch_tokens[b_idx])
            for t in range(seq_start, seq_end - 1):
                target = x[b_idx, t + 1]
                if target < vocab_size and target < n_tokens:
                    loss = loss + F.cross_entropy(
                        logits[b_idx, t, :n_tokens], target)
                    count += 1
        if count > 0:
            loss = loss / count

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if step % args.log_every == 0:
            recent = float(np.mean(losses[-args.log_every:]))
            elapsed = time.time() - start_time
            rate = step / max(elapsed, 1.0)
            print(f"  step {step}/{args.n_steps}: loss={recent:.4f}  "
                  f"({rate:.1f} steps/s, "
                  f"ETA {(args.n_steps - step) / max(rate, 1e-6) / 60:.1f} min)")

        if step % args.eval_every == 0 or step == args.n_steps:
            train_eval = evaluate_on_buckets(
                model, src, cfg, train_buckets, n_buckets,
                n_eval=args.eval_n_during, device=device, rng=rng_eval_tr)
            print(f"    [eval@{step}] train-bucket: "
                  f"KL={train_eval['kl_bits_mean']:.6f} bits, "
                  f"MAE={train_eval['entropy_mae_bits_mean']:.6f} bits")
            eval_history.append({
                'step': step,
                'train_bucket': train_eval,
            })
            if train_eval['entropy_mae_bits_mean'] < best_train_mae:
                best_train_mae = train_eval['entropy_mae_bits_mean']
                best_step = step
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, 'best_model.pt'))
                print(f"    new best (train-bucket MAE={best_train_mae:.6f}) "
                      "-> saved")
            if best_train_mae < early_stop_threshold:
                print(f"    early stop: train-bucket MAE < "
                      f"{early_stop_threshold}")
                break

    # ---- final eval on both splits with the best checkpoint ----
    best_path = os.path.join(args.output_dir, 'best_model.pt')
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"\n  Final eval (best step {best_step}, "
          f"train-bucket MAE so far {best_train_mae:.6f}):")
    print(f"    sampling {args.eval_n_final} train-bucket episodes...")
    final_train = evaluate_on_buckets(
        model, src, cfg, train_buckets, n_buckets,
        n_eval=args.eval_n_final, device=device, rng=rng_eval_tr)
    print(f"    sampling {args.eval_n_final} holdout-bucket episodes...")
    final_heldout = evaluate_on_buckets(
        model, src, cfg, {args.holdout_bucket}, n_buckets,
        n_eval=args.eval_n_final, device=device, rng=rng_eval_ho)

    results = {
        'config': {
            'p': p, 'pi': args.pi, 'seq_len': args.seq_len,
            'seed': args.seed, 'holdout_bucket': args.holdout_bucket,
            'train_buckets': sorted(train_buckets),
            'n_buckets': n_buckets,
            'bucket_counts': bucket_counts,
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'arch': {
                'd_model': args.d_model, 'n_layers': args.n_layers,
                'n_heads': args.n_heads, 'd_ff': args.d_ff,
                'dropout': args.dropout,
            },
            'device': str(device),
            'eval_n_final': args.eval_n_final,
            'eval_n_during': args.eval_n_during,
            'early_stop_mae': early_stop_threshold,
        },
        'best_step': best_step,
        'best_train_mae_during': best_train_mae,
        'final_train_bucket': final_train,
        'final_heldout_bucket': final_heldout,
        'separation_kl_bits':
            final_heldout['kl_bits_mean'] - final_train['kl_bits_mean'],
        'separation_mae_bits':
            (final_heldout['entropy_mae_bits_mean'] -
             final_train['entropy_mae_bits_mean']),
        'eval_history': eval_history,
        'final_loss': float(np.mean(losses[-1000:])),
        'wall_clock_sec': time.time() - start_time,
    }

    out_path = os.path.join(args.output_dir, 'metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("HELD-OUT-PREFIX CELL RESULTS")
    print("=" * 70)
    print(f"  p={p}  holdout_bucket={args.holdout_bucket}  seed={args.seed}")
    print(f"  train  : KL={final_train['kl_bits_mean']:.6f} bits, "
          f"MAE={final_train['entropy_mae_bits_mean']:.6f} bits, "
          f"TV={final_train['tv_mean']:.4f}")
    print(f"  heldout: KL={final_heldout['kl_bits_mean']:.6f} bits, "
          f"MAE={final_heldout['entropy_mae_bits_mean']:.6f} bits, "
          f"TV={final_heldout['tv_mean']:.4f}")
    print(f"  delta-KL  = {results['separation_kl_bits']:.6f} bits")
    print(f"  delta-MAE = {results['separation_mae_bits']:.6f} bits")
    print(f"  saved -> {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Held-out-prefix replication at p in {17,23,31,43}')
    parser.add_argument('--p', type=int, default=31,
                        choices=[17, 23, 31, 43])
    parser.add_argument('--holdout_bucket', type=int, required=True,
                        choices=[0, 1, 2, 3, 4])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--pi', type=float, default=0.5)
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--n_steps', type=int, default=150_000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--log_every', type=int, default=500)
    parser.add_argument('--eval_every', type=int, default=10_000)
    parser.add_argument('--eval_n_during', type=int, default=1_000)
    parser.add_argument('--eval_n_final', type=int, default=20_000)
    parser.add_argument('--early_stop_mae', type=float, default=0.02)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--source_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'results', f'heldout_prefix_p{args.p}',
            f'seed_{args.seed}_holdout_{args.holdout_bucket}')

    src = _import_source(args.source_dir)
    np.random.seed(args.seed)
    src._ensure_torch()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=" * 70)
    print(f"HELD-OUT-PREFIX (p={args.p}, holdout={args.holdout_bucket}, "
          f"seed={args.seed})")
    print("=" * 70)
    train_heldout(args, src)


if __name__ == '__main__':
    main()
