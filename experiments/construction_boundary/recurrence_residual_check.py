"""
Item B (reviewer audit): off-mixture residual for the implicit-class-posterior
extraction, on the pi-experiment integer checkpoints.

The class posterior w is solved from the single coordinate y* (the H_P point mass):
    w = (P_model(y*) - 1/p) / (1 - 1/p),  clipped to [0,1].
This is only meaningful if P_model actually lies (near) the affine segment
  w * e_{y*} + (1-w) * Uniform.
We measure the L1 off-mixture residual  || P_model - [w e_{y*} + (1-w) U] ||_1
across ALL coordinates, at program-determined positions (t >= 4), and report the
distribution. A small residual certifies the recovered w is not a projection artefact.
"""
import os, sys, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # recurrence_bwt.py is in the same directory
import recurrence_bwt as R
R._ensure_torch()
torch = R.torch

# Checkpoints are archived on Zenodo (large); set CKPT_DIR or place under results/recurrence/.
RESULTS = os.environ.get('CKPT_DIR', os.path.join(HERE, 'results', 'recurrence'))
SEEDS = [42, 1337, 2024, 7]
N_EVAL = 4000
DEVICE = os.environ.get('DEVICE', 'cuda:0')
P = 17
RTClass = R._build_model_class()


def build_model():
    return RTClass(vocab_size=P, n_tokens=P, d_model=192, n_layers=6,
                   n_heads=6, d_ff=768, dropout=0.1)


def residuals(model):
    np.random.seed(12345)
    model.eval()
    cfg = R.RecurrenceConfig(p=P, pi=0.5, seq_len=16, opaque=False)
    res = []
    U = np.ones(P) / P
    with torch.no_grad():
        for _ in range(N_EVAL):
            tokens, gt, metadata = R.generate_recurrence_sequence(cfg)
            hl = metadata['header_len']; n_tok = metadata['n_tokens']
            tt = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
            logits, _ = model(tt)
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            for e in gt:
                t = e['t']; mp = hl + t - 1
                if t < 4 or mp < 0 or mp >= len(probs):
                    continue
                if e.get('p_program', 0) <= 0.99:
                    continue
                pm = probs[mp][:n_tok]
                ystar = max(e['pred_dist'], key=e['pred_dist'].get)
                if ystar >= n_tok:
                    continue
                w = (pm[ystar] - 1.0 / P) / (1.0 - 1.0 / P)
                w = min(1.0, max(0.0, w))
                recon = w * np.eye(P)[ystar] + (1 - w) * U
                res.append(float(np.sum(np.abs(pm - recon))))
    return np.array(res)


def main():
    allr = []
    for s in SEEDS:
        ck = os.path.join(RESULTS, 'integer', f'seed_{s}', 'best_model.pt')
        if not os.path.exists(ck):
            print('MISSING', ck); continue
        m = build_model().to(DEVICE)
        m.load_state_dict(torch.load(ck, map_location=DEVICE))
        r = residuals(m)
        allr.append(r)
        print(f"[seed {s}] n={len(r)} L1 residual  mean={r.mean():.4f} median={np.median(r):.4f} "
              f"p90={np.percentile(r,90):.4f} max={r.max():.4f}", flush=True)
    allr = np.concatenate(allr)
    out = {'n': int(len(allr)), 'mean': float(allr.mean()), 'median': float(np.median(allr)),
           'p90': float(np.percentile(allr, 90)), 'max': float(allr.max()),
           'note': 'L1 off-mixture residual, integer pi-experiment, t>=4, 4 seeds'}
    json.dump(out, open(os.path.join(HERE, 'results', 'recurrence_residual.json'), 'w'), indent=2)
    print("\nOVERALL:", json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
