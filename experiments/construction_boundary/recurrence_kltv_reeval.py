"""
Item C (reviewer audit): add per-position distributional error to the pi experiment.

Re-evaluates the EXISTING 2.8M pi-experiment checkpoints (no retraining) and reports,
per prediction position t, the model-vs-Bayes:
  - entropy MAE (bits)         [reproduction check vs paper: 0.014 integer, 0.83 opaque]
  - D_KL(P_Bayes || P_model)   (bits)   <-- new
  - total variation distance   (bits-free, in [0,1])  <-- new

Uses recurrence_bwt.py's own model class, sequence generator, and Bayes oracle so the
numbers are directly comparable to the published entropy figure.
"""
import os, sys, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# recurrence_bwt.py lives alongside this script in construction_boundary/
sys.path.insert(0, HERE)
import recurrence_bwt as R
R._ensure_torch()
torch = R.torch

# Directory holding the pi-experiment checkpoints (integer/opaque, seed_*/best_model.pt).
# These are large (~11 MB each) and are archived on Zenodo rather than in git; set CKPT_DIR
# to point at them, or place them under construction_boundary/results/recurrence/.
RESULTS = os.environ.get('CKPT_DIR', os.path.join(HERE, 'results', 'recurrence'))
SEEDS = [42, 1337, 2024, 7]
N_EVAL = 4000
DEVICE = os.environ.get('DEVICE', 'cuda:0')
EVAL_SEED = 12345  # fixed reanalysis seed across checkpoints

RTClass = R._build_model_class()


def build_model(opaque, cfg_json):
    p = 17
    if opaque:
        vocab_size, n_tokens = 2 * p + 2, p
    else:
        vocab_size, n_tokens = p, p
    return RTClass(
        vocab_size=vocab_size, n_tokens=n_tokens,
        d_model=cfg_json.get('d_model', 192),
        n_layers=cfg_json.get('n_layers', 6),
        n_heads=cfg_json.get('n_heads', 6),
        d_ff=cfg_json.get('d_ff', 768),
        dropout=0.1,
    )


def eval_kltv(model, opaque, n_eval, device):
    np.random.seed(EVAL_SEED)
    model.eval()
    p = 17
    cfg = R.RecurrenceConfig(p=p, pi=0.5, seq_len=16, opaque=opaque)
    per = {}
    with torch.no_grad():
        for _ in range(n_eval):
            tokens, gt, metadata = R.generate_recurrence_sequence(cfg)
            header_len = metadata['header_len']
            n_tok = metadata['n_tokens']
            tt = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            logits, _ = model(tt)
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            for e in gt:
                t = e['t']
                mp = header_len + t - 1
                if t == 0 or mp < 0 or mp >= len(probs):
                    continue
                pm = probs[mp][:n_tok]
                pd = e['pred_dist']
                if opaque:
                    p_bayes = np.array([pd.get(v + p, 0.0) for v in range(n_tok)])
                else:
                    p_bayes = np.array([pd.get(v, 0.0) for v in range(n_tok)])
                H_model = -sum(x * math.log2(x) for x in pm if x > 1e-10)
                H_bayes = e['entropy']
                kl_bits = sum(p_bayes[y] * math.log2(p_bayes[y] / max(pm[y], 1e-10))
                              for y in range(n_tok) if p_bayes[y] > 1e-10)
                tv = 0.5 * float(np.sum(np.abs(p_bayes - pm)))
                d = per.setdefault(t, {'mae': [], 'kl': [], 'tv': []})
                d['mae'].append(abs(H_model - H_bayes))
                d['kl'].append(kl_bits)
                d['tv'].append(tv)
    return {t: {'mae': float(np.mean(v['mae'])), 'kl_bits': float(np.mean(v['kl'])),
                'tv': float(np.mean(v['tv'])), 'n': len(v['mae'])}
            for t, v in sorted(per.items())}


def main():
    out = {}
    for cond in ['integer', 'opaque']:
        opaque = (cond == 'opaque')
        cfg_json = {}
        jpath = os.path.join(RESULTS, cond, 'recurrence_bwt_results.json')
        if os.path.exists(jpath):
            try:
                j = json.load(open(jpath))
                cfg_json = j.get('config', {}) if isinstance(j, dict) else {}
            except Exception:
                pass
        per_seed = []
        for s in SEEDS:
            ck = os.path.join(RESULTS, cond, f'seed_{s}', 'best_model.pt')
            if not os.path.exists(ck):
                print('MISSING', ck); continue
            m = build_model(opaque, cfg_json).to(DEVICE)
            m.load_state_dict(torch.load(ck, map_location=DEVICE))
            r = eval_kltv(m, opaque, N_EVAL, DEVICE)
            per_seed.append(r)
            locked = [r[t]['kl_bits'] for t in r if t >= 6]
            print(f"[{cond} seed {s}] overall MAE={np.mean([r[t]['mae'] for t in r]):.4f} "
                  f"overall KL={np.mean([r[t]['kl_bits'] for t in r]):.4f} bits "
                  f"t>=6 KL={np.mean(locked):.5f} bits", flush=True)
        agg = {}
        for t in sorted({t for r in per_seed for t in r}):
            kl = [r[t]['kl_bits'] for r in per_seed if t in r]
            ma = [r[t]['mae'] for r in per_seed if t in r]
            tv = [r[t]['tv'] for r in per_seed if t in r]
            agg[str(t)] = {'kl_bits_mean': float(np.mean(kl)), 'kl_bits_std': float(np.std(kl)),
                           'mae_mean': float(np.mean(ma)), 'tv_mean': float(np.mean(tv))}
        out[cond] = {'per_position': agg, 'seeds': SEEDS, 'n_eval': N_EVAL, 'eval_seed': EVAL_SEED}
    with open(os.path.join(HERE, 'results', 'recurrence_kltv_reeval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\n=== SUMMARY (per-position, mean across seeds) ===")
    for cond in out:
        print(f"\n{cond}:")
        print(f"  {'t':>3} {'KL(bits)':>10} {'MAE(bits)':>10} {'TV':>8}")
        for t in sorted(out[cond]['per_position'], key=int):
            e = out[cond]['per_position'][t]
            print(f"  {t:>3} {e['kl_bits_mean']:>10.4f} {e['mae_mean']:>10.4f} {e['tv_mean']:>8.4f}")


if __name__ == '__main__':
    main()
