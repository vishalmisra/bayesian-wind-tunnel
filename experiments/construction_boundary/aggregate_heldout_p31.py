import json, glob, numpy as np
rs_paths = sorted(glob.glob("heldout_prefix_p31/seed_*_holdout_*/metrics.json"))
rows = []
for p in rs_paths:
    d = json.load(open(p))
    cfg = d["config"]
    tr = d["final_train_bucket"]
    ho = d["final_heldout_bucket"]
    rows.append((cfg["seed"], cfg["holdout_bucket"],
                 tr["kl_bits_mean"], ho["kl_bits_mean"],
                 tr["entropy_mae_bits_mean"], ho["entropy_mae_bits_mean"],
                 tr.get("tv_mean", float("nan")), ho.get("tv_mean", float("nan"))))
rows.sort(key=lambda r: (r[0], r[1]))
print("  seed  holdout    tr_KL      ho_KL     delta    ratio")
for s, h, tk, hk, _, _, _, _ in rows:
    print(f"  {s:>4}  {h:>6}   {tk:8.5f}  {hk:8.5f}  {hk-tk:+8.5f}  {hk/max(1e-9,tk):5.1f}x")
tr_kl = np.array([r[2] for r in rows])
ho_kl = np.array([r[3] for r in rows])
tr_mae = np.array([r[4] for r in rows])
ho_mae = np.array([r[5] for r in rows])
print()
print("AGGREGATE", len(rows), "cells (5 buckets x 3 seeds)")
print(f"  train  KL  : mean {tr_kl.mean():.5f}  range [{tr_kl.min():.5f}, {tr_kl.max():.5f}]")
print(f"  heldout KL : mean {ho_kl.mean():.5f}  range [{ho_kl.min():.5f}, {ho_kl.max():.5f}]")
print(f"  delta KL   : mean {(ho_kl-tr_kl).mean():.5f}  range [{(ho_kl-tr_kl).min():.5f}, {(ho_kl-tr_kl).max():.5f}]")
ratio = ho_kl / np.clip(tr_kl, 1e-9, None)
print(f"  KL ratio   : mean {ratio.mean():.2f}x  range [{ratio.min():.2f}, {ratio.max():.2f}]x")
print(f"  train  MAE : mean {tr_mae.mean():.5f}  range [{tr_mae.min():.5f}, {tr_mae.max():.5f}]")
print(f"  heldout MAE: mean {ho_mae.mean():.5f}  range [{ho_mae.min():.5f}, {ho_mae.max():.5f}]")
