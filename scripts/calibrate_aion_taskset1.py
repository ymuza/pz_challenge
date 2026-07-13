#!/usr/bin/env python3
"""Measure PDF calibration for the AION trained head on task set 1.

Three-way split of the training file (which has truth): train the head, fit
calibration on a calibration split, and report point + PIT metrics on a held-out
eval split for: uncalibrated, temperature-scaled, and PIT-recalibrated PDFs.

Usage:
    python scripts/calibrate_aion_taskset1.py [TRAIN_FILE] \
        [--n-train N] [--n-calib N] [--n-eval N] [--device cpu|mps|cuda]
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


def point_metrics(z_phot, z_true):
    dz = (z_phot - z_true) / (1.0 + z_true)
    return {
        "bias": float(np.median(dz)),
        "sigma_MAD": float(1.4826 * np.median(np.abs(dz - np.median(dz)))),
        "outlier_frac": float(np.mean(np.abs(dz) > 0.15)),
    }


def posterior_mean(pz, grid):
    dz = grid[1] - grid[0]
    return (pz * grid).sum(axis=1) * dz / ((pz.sum(axis=1) * dz) + 1e-12)


def sub(data, idx):
    return {k: v[idx] for k, v in data.items()}


def summarize(name, pz, grid, z_eval, results):
    results[name] = {
        **point_metrics(posterior_mean(pz, grid), z_eval),
        "PIT_KS": aion_pz.pit_ks_stat(pz, grid, z_eval),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_file", nargs="?", default=DEFAULT_TRAIN)
    ap.add_argument("--n-train", type=int, default=80000)
    ap.add_argument("--n-calib", type=int, default=10000)
    ap.add_argument("--n-eval", type=int, default=10000)
    ap.add_argument("--device", default=os.environ.get("AION_PZ_DEVICE"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {args.train_file} ...")
    data = aion_pz.load_catalog(args.train_file)
    z = np.asarray(data[aion_pz.REDSHIFT_COL], dtype="float64")
    good = np.isfinite(z)
    data, z = sub(data, np.where(good)[0]), z[good]
    n = len(z)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    a, b, c = args.n_train, args.n_calib, args.n_eval
    tr, ca, ev = perm[:a], perm[a : a + b], perm[a + b : a + b + c]
    print(f"  train={len(tr)}  calib={len(ca)}  eval={len(ev)}")

    print("Loading AION ...")
    model, cm, device = aion_pz.load_aion(device=args.device)
    grid = aion_pz.Z_GRID

    print("Embeddings + training head ...")
    t0 = time.time()
    x_tr = aion_pz.build_design_matrix(model, cm, sub(data, tr), device)
    head = aion_pz.train_head(x_tr, z[tr])
    pz_ca = aion_pz.predict_pz(head, aion_pz.build_design_matrix(model, cm, sub(data, ca), device))
    pz_ev = aion_pz.predict_pz(head, aion_pz.build_design_matrix(model, cm, sub(data, ev), device))
    print(f"  done in {time.time() - t0:.1f}s")

    z_ca, z_ev = z[ca], z[ev]

    # Fit calibrators on the calibration split.
    T = aion_pz.fit_temperature(pz_ca, grid, z_ca)
    recal = aion_pz.fit_pit_recalibration(pz_ca, grid, z_ca)
    print(f"  fitted temperature T={T:.2f} (T>1 broadens)")

    results = {}
    summarize("uncalibrated", pz_ev, grid, z_ev, results)
    summarize(f"temperature(T={T:.2f})", aion_pz.apply_temperature(pz_ev, grid, T), grid, z_ev, results)
    summarize("PIT-recalibrated", aion_pz.apply_pit_recalibration(pz_ev, grid, recal), grid, z_ev, results)

    print("\n" + "=" * 74)
    print(f"Task set 1 PDF calibration  (eval N={len(ev)}, point metrics on posterior mean)")
    print("=" * 74)
    cols = ["bias", "sigma_MAD", "outlier_frac", "PIT_KS"]
    print(f"{'method':<22}" + "".join(f"{c:>13}" for c in cols))
    for name, m in results.items():
        print(f"{name:<22}" + "".join(f"{m[c]:>13.4f}" for c in cols))
    print("\nPIT_KS is the calibration target (0 = perfectly calibrated p(z)).")


if __name__ == "__main__":
    main()
