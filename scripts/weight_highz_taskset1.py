#!/usr/bin/env python3
"""Test high-z reweighting of the AION head on task set 1.

Embeddings are extracted once; then the head is retrained for several
reweighting strengths (alpha).  Metrics are reported overall and split at
z=1.5 so the high-z effect is visible.  PIT recalibration is refit per alpha.
"""

import argparse
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import aion_pz

DEFAULT_TRAIN = "public/pz_challenge_taskset_1_cardinal_training_1yr.hdf5"
Z_SPLIT = 1.5


def point_metrics(z_phot, z_true):
    if len(z_true) == 0:
        return {"bias": float("nan"), "sigma_MAD": float("nan"), "outlier_frac": float("nan")}
    dz = (z_phot - z_true) / (1.0 + z_true)
    return {
        "bias": float(np.median(dz)),
        "sigma_MAD": float(1.4826 * np.median(np.abs(dz - np.median(dz)))),
        "outlier_frac": float(np.mean(np.abs(dz) > 0.15)),
    }


def pmean(pz, grid):
    dz = grid[1] - grid[0]
    return (pz * grid).sum(axis=1) * dz / ((pz.sum(axis=1) * dz) + 1e-12)


def sub(d, idx):
    return {k: v[idx] for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_file", nargs="?", default=DEFAULT_TRAIN)
    ap.add_argument("--n-train", type=int, default=70000)
    ap.add_argument("--n-calib", type=int, default=10000)
    ap.add_argument("--n-eval", type=int, default=20000)
    ap.add_argument("--alphas", default="0.0,0.5,1.0")
    ap.add_argument("--device", default=os.environ.get("AION_PZ_DEVICE"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

    data = aion_pz.load_catalog(args.train_file)
    z = np.asarray(data[aion_pz.REDSHIFT_COL], dtype="float64")
    good = np.isfinite(z)
    data, z = sub(data, np.where(good)[0]), z[good]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(z))
    a, b, c = args.n_train, args.n_calib, args.n_eval
    tr, ca, ev = perm[:a], perm[a : a + b], perm[a + b : a + b + c]

    print("Loading AION + extracting embeddings once ...")
    t0 = time.time()
    model, cm, device = aion_pz.load_aion(device=args.device)
    grid = aion_pz.Z_GRID
    x_tr = aion_pz.build_design_matrix(model, cm, sub(data, tr), device)
    x_ca = aion_pz.build_design_matrix(model, cm, sub(data, ca), device)
    x_ev = aion_pz.build_design_matrix(model, cm, sub(data, ev), device)
    z_tr, z_ca, z_ev = z[tr], z[ca], z[ev]
    print(f"  embeddings ready in {time.time() - t0:.1f}s")
    n_hi = int(np.sum(z_ev >= Z_SPLIT))
    print(f"  eval: {len(ev)} objs, {n_hi} at z>={Z_SPLIT} ({n_hi/len(ev):.1%})")

    lo, hi = z_ev < Z_SPLIT, z_ev >= Z_SPLIT
    label = os.path.basename(args.train_file).replace("pz_challenge_", "").replace(".hdf5", "")
    print("\n" + "=" * 96)
    print(f"High-z reweighting  ({label}, eval N={len(ev)}; split at z={Z_SPLIT})")
    print("=" * 96)
    hdr = ["alpha", "grp", "N", "bias", "sigma_MAD", "outlier", "PIT_KS"]
    print("".join(f"{h:>11}" for h in hdr))
    for alpha in alphas:
        w = aion_pz.redshift_weights(z_tr, alpha=alpha) if alpha > 0 else None
        head = aion_pz.train_head(x_tr, z_tr, sample_weight=w)
        recal = aion_pz.fit_pit_recalibration(aion_pz.predict_pz(head, x_ca), grid, z_ca)
        pz = aion_pz.apply_pit_recalibration(aion_pz.predict_pz(head, x_ev), grid, recal)
        zmean = pmean(pz, grid)
        ks_all = aion_pz.pit_ks_stat(pz, grid, z_ev)
        for grp, mask, ks in [("all", np.ones(len(ev), bool), ks_all),
                              ("low", lo, float("nan")), ("high", hi, float("nan"))]:
            m = point_metrics(zmean[mask], z_ev[mask])
            ksv = ks if grp == "all" else float("nan")
            print(f"{alpha:>11.2f}{grp:>11}{int(mask.sum()):>11}"
                  f"{m['bias']:>11.4f}{m['sigma_MAD']:>11.4f}{m['outlier_frac']:>11.4f}{ksv:>11.4f}")
        print("-" * 96)
    print("alpha=0 is the current (unweighted) head. Watch high-group outlier/bias.")


if __name__ == "__main__":
    main()
