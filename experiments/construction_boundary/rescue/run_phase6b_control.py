"""Reviewer follow-up controls for the writer-state rescue (Fig 3c).

Addresses the positional confound: the full-residual transplant also moves the
donor's transformed positional signature, so full-state rescue does not by itself
establish a position-general reader. Two controls:

(A) Writer-update transplant. Instead of replacing r_q with the donor residual,
    keep the recipient's own residual through block L-1 and add ONLY the donor
    block's computed update:
        r_q^{L} <- r_q^{L-1}(recipient) + [ r_p^{L} - r_p^{L-1} ](donor).
    This preserves the recipient's positional information. If it still rescues,
    the writer-side-localization claim is clean.  (full@L reproduced alongside.)

(B) Mismatched-donor redirect. Donor at position p shares the recipient's current
    token but has DIFFERENT (a,b) -> a different next-token target a'*t+b'. Does
    the transplant redirect the recipient toward the donor's target (content-
    specific control) rather than merely restoring a generic in-horizon signature?

Usage:
    python run_phase6b_control.py --k5-checkpoints s42.pt s1337.pt s2024.pt \
        --out phase6b_control.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import _ResidualRecorder          # noqa: E402
from interventions import _PatchHook            # noqa: E402
from models import load_model                   # noqa: E402

SHIFT = 6
PAIRS = ((2, 8), (3, 9), (4, 10))
BLOCKS = ["0", "1", "2", "3"]
PREV = {"0": "emb", "1": "0", "2": "1", "3": "2"}


def modinv(a, p):
    return pow(int(a) % p, p - 2, p)  # p prime


def gen_orbit_pairs(B, p, seq_len, shift, seed):
    rng = np.random.default_rng(seed)
    test = np.zeros((B, seq_len), dtype=np.int64)
    donor = np.zeros((B, seq_len), dtype=np.int64)
    n = 0
    while n < B:
        a, b, x0 = rng.integers(0, p, size=3)
        orbit = [int(x0)]
        for _ in range(seq_len + shift):
            orbit.append(int((a * orbit[-1] + b) % p))
        if orbit[0] == orbit[1] or orbit[shift] == orbit[shift + 1]:
            continue
        test[n] = orbit[:seq_len]
        donor[n] = orbit[shift:shift + seq_len]
        n += 1
    return torch.from_numpy(test), torch.from_numpy(donor)


def recover_ab(test, p):
    x0 = test[:, 0].numpy().astype(int); x1 = test[:, 1].numpy().astype(int); x2 = test[:, 2].numpy().astype(int)
    inv = np.array([modinv((x1[i] - x0[i]) % p, p) for i in range(len(x0))])
    a = ((x2 - x1) * inv) % p
    b = (x1 - a * x0) % p
    return a.astype(int), b.astype(int)


def gen_mismatch_donor(test, q_pos, p_pos, p, seq_len, seed):
    """Donor shares recipient's token at (q->p) but has different (a',b')."""
    rng = np.random.default_rng(seed + 777)
    B = test.shape[0]
    t = test[:, q_pos].numpy().astype(int)
    a_rec, b_rec = recover_ab(test, p)
    donor = np.zeros((B, seq_len), dtype=np.int64)
    don_t = np.zeros(B, dtype=np.int64)
    for i in range(B):
        while True:
            ap = int(rng.integers(1, p)); bp = int(rng.integers(0, p))
            if ap == a_rec[i] and bp == b_rec[i]:
                continue
            inv_ap = modinv(ap, p)
            orb = [0] * seq_len
            orb[p_pos] = int(t[i])
            for j in range(p_pos - 1, -1, -1):
                orb[j] = int(((orb[j + 1] - bp) * inv_ap) % p)
            for j in range(p_pos + 1, seq_len):
                orb[j] = int((ap * orb[j - 1] + bp) % p)
            if orb[0] == orb[1] or orb[p_pos] == orb[p_pos + 1]:
                continue
            cand_t = int((ap * int(t[i]) + bp) % p)
            if cand_t == int(test[i, q_pos + 1]):   # enforce donor target != recipient target
                continue
            donor[i] = orb
            don_t[i] = cand_t
            break
    return torch.from_numpy(donor), don_t, test[:, q_pos + 1].numpy().astype(int)


def capture_all(model, tokens):
    device = next(model.parameters()).device
    with _ResidualRecorder(model) as rec:
        model.logits(tokens.to(device))
    return {k: v.detach() for k, v in rec.activations.items()}


def read_pred(model, tokens, q, layer=None, positions=None, donor=None, basis=None):
    device = next(model.parameters()).device
    if layer is None:
        logits = model.logits(tokens.to(device))
    else:
        with _PatchHook(model, layer, positions, donor, basis):
            logits = model.logits(tokens.to(device))
    return logits[:, q, :].float().argmax(-1).cpu().numpy()


def make_aligned(model, q, seq_len, content):
    device = next(model.parameters()).device
    a = torch.zeros(content.shape[0], seq_len, model.dim, device=device)
    a[:, q, :] = content
    return a


def control_update(model, args):
    p = model.vocab_size
    test, donor = gen_orbit_pairs(args.n_seq, p, args.seq_len, SHIFT, args.seed)
    dR = capture_all(model, donor); tR = capture_all(model, test)
    out = {}
    for p_in, q in PAIRS:
        truth = test[:, q + 1].numpy().astype(int)
        base = read_pred(model, test, q)
        rec = {"none": float((base == truth).mean())}
        for L in BLOCKS:
            Lp = PREV[L]
            delta = (dR[L][:, p_in, :] - dR[Lp][:, p_in, :])
            upd = tR[Lp][:, q, :].to(delta.device) + delta
            pu = read_pred(model, test, q, layer=L, positions=[q],
                           donor=make_aligned(model, q, args.seq_len, upd), basis=None)
            pf = read_pred(model, test, q, layer=L, positions=[q],
                           donor=make_aligned(model, q, args.seq_len, dR[L][:, p_in, :]), basis=None)
            rec[f"update@{L}"] = float((pu == truth).mean())
            rec[f"full@{L}"] = float((pf == truth).mean())
        out[f"p{p_in}->q{q}"] = rec
    return out


def control_mismatch(model, args):
    p = model.vocab_size
    out = {}
    for p_in, q in PAIRS:
        test, _ = gen_orbit_pairs(args.n_seq, p, args.seq_len, SHIFT, args.seed)
        donor, don_t, rec_t = gen_mismatch_donor(test, q, p_in, p, args.seq_len, args.seed)
        dR = capture_all(model, donor); tR = capture_all(model, test)
        base = read_pred(model, test, q)
        rec = {"none_redirect": float((base == don_t).mean()),
               "none_retain": float((base == rec_t).mean())}
        for L in BLOCKS:
            Lp = PREV[L]
            delta = (dR[L][:, p_in, :] - dR[Lp][:, p_in, :])
            upd = tR[Lp][:, q, :].to(delta.device) + delta
            pf = read_pred(model, test, q, layer=L, positions=[q],
                           donor=make_aligned(model, q, args.seq_len, dR[L][:, p_in, :]), basis=None)
            pu = read_pred(model, test, q, layer=L, positions=[q],
                           donor=make_aligned(model, q, args.seq_len, upd), basis=None)
            rec[f"full@{L}_redirect"] = float((pf == don_t).mean())
            rec[f"full@{L}_retain"] = float((pf == rec_t).mean())
            rec[f"update@{L}_redirect"] = float((pu == don_t).mean())
            rec[f"update@{L}_retain"] = float((pu == rec_t).mean())
        out[f"p{p_in}->q{q}"] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k5-checkpoints", nargs="+", required=True)
    ap.add_argument("--n-seq", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=321)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="phase6b_control.json")
    args = ap.parse_args()

    results = {"update": [], "mismatch": []}
    for ckpt in args.k5_checkpoints:
        model = load_model(ckpt, device=args.device)
        seed_tag = Path(ckpt).parent.name
        u = control_update(model, args)
        m = control_mismatch(model, args)
        results["update"].append({"checkpoint": ckpt, "detail": u})
        results["mismatch"].append({"checkpoint": ckpt, "detail": m})

        def mean_over_pairs(d, key):
            return float(np.mean([d[k][key] for k in d]))
        print(f"\n=== {seed_tag} ===")
        print(f"  [A writer-update]  none={mean_over_pairs(u,'none'):.3f}")
        for L in BLOCKS:
            print(f"    block {L}: update={mean_over_pairs(u,f'update@{L}'):.3f}  "
                  f"full={mean_over_pairs(u,f'full@{L}'):.3f}")
        print(f"  [B mismatch redirect] chance={1/model.vocab_size:.3f}  "
              f"none_redirect={mean_over_pairs(m,'none_redirect'):.3f}")
        for L in BLOCKS:
            print(f"    block {L}: full redirect={mean_over_pairs(m,f'full@{L}_redirect'):.3f} "
                  f"retain={mean_over_pairs(m,f'full@{L}_retain'):.3f} | "
                  f"update redirect={mean_over_pairs(m,f'update@{L}_redirect'):.3f} "
                  f"retain={mean_over_pairs(m,f'update@{L}_retain'):.3f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
