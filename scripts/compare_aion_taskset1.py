#!/usr/bin/env python3
"""Compare the AION trained-head vs zero-shot baseline on task set 1.

Uses the *training* file (which carries the true `redshift`), holds out a
validation split, and scores both methods against truth with standard photo-z
metrics.  This is the honest apples-to-apples comparison the README suggests.

Usage:
    python scripts/compare_aion_taskset1.py [TRAIN_FILE] \
        [--n-train N] [--n-val N] [--device cpu|mps|cuda]
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


def point_metrics(z_phot: np.ndarray, z_true: np.ndarray) -> dict[str, float]:
    dz = (z_phot - z_true) / (1.0 + z_true)
    bias = float(np.median(dz))
    sigma_mad = float(1.4826 * np.median(np.abs(dz - np.median(dz))))
    outlier = float(np.mean(np.abs(dz) > 0.15))
    return {"bias": bias, "sigma_MAD": sigma_mad, "outlier_frac": outlier}


def posterior_mean(pz: np.ndarray, grid: np.ndarray) -> np.ndarray:
    dz = grid[1] - grid[0]
    return (pz * grid).sum(axis=1) * dz / ((pz.sum(axis=1) * dz) + 1e-12)


def pit_ks(pz: np.ndarray, grid: np.ndarray, z_true: np.ndarray) -> float:
    """KS statistic of the PIT distribution vs Uniform(0,1). Lower = better."""
    dz = grid[1] - grid[0]
    cdf = np.cumsum(pz, axis=1) * dz
    cdf /= cdf[:, -1:] + 1e-12
    pit = np.array([np.interp(z_true[i], grid, cdf[i]) for i in range(len(z_true))])
    pit_sorted = np.sort(pit)
    n = len(pit_sorted)
    emp = np.arange(1, n + 1) / n
    return float(np.max(np.abs(pit_sorted - emp)))


def subsample(data: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[idx] for k, v in data.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("train_file", nargs="?", default=DEFAULT_TRAIN)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--device", default=os.environ.get("AION_PZ_DEVICE"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {args.train_file} ...")
    data = aion_pz.load_catalog(args.train_file)
    z = np.asarray(data[aion_pz.REDSHIFT_COL], dtype="float64")
    good = np.isfinite(z)
    data = subsample(data, np.where(good)[0])
    z = z[good]
    n = len(z)
    print(f"  {n} objects with finite redshift; columns: {sorted(data.keys())[:12]}...")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_tr = min(args.n_train, n - args.n_val)
    n_va = min(args.n_val, n - n_tr)
    tr_idx, va_idx = perm[:n_tr], perm[n_tr : n_tr + n_va]
    train = subsample(data, tr_idx)
    val = subsample(data, va_idx)
    z_val = z[va_idx]
    print(f"  train={n_tr}  val={n_va}")

    print("Loading AION (this downloads aion-base on first run) ...")
    t0 = time.time()
    model, cm, device = aion_pz.load_aion(device=args.device)
    print(f"  model on {device} in {time.time() - t0:.1f}s")

    grid = aion_pz.Z_GRID
    results = {}

    # --- Zero-shot baseline ---
    print("Running ZERO-SHOT baseline on val ...")
    t0 = time.time()
    pz_native, z_centers = aion_pz.zero_shot_pz(model, cm, val, device)
    pz_zs = aion_pz._resample_to_grid(pz_native, z_centers, grid)
    t_zs = time.time() - t0
    zmean_zs = posterior_mean(pz_zs, grid)
    results["zero-shot"] = {
        **point_metrics(zmean_zs, z_val),
        "PIT_KS": pit_ks(pz_zs, grid, z_val),
        "seconds": t_zs,
    }

    # --- Trained head ---
    print("Extracting embeddings + training HEAD ...")
    t0 = time.time()
    x_tr = aion_pz.build_design_matrix(model, cm, train, device)
    z_tr = z[tr_idx]
    head = aion_pz.train_head(x_tr, z_tr)
    x_va = aion_pz.build_design_matrix(model, cm, val, device)
    pz_head = aion_pz.predict_pz(head, x_va)
    t_head = time.time() - t0
    zmean_head = posterior_mean(pz_head, grid)
    results["trained-head"] = {
        **point_metrics(zmean_head, z_val),
        "PIT_KS": pit_ks(pz_head, grid, z_val),
        "seconds": t_head,
    }

    # --- Report ---
    print("\n" + "=" * 68)
    print(f"Task set 1 comparison  (val N={n_va}, point metrics on posterior mean)")
    print("=" * 68)
    cols = ["bias", "sigma_MAD", "outlier_frac", "PIT_KS", "seconds"]
    print(f"{'method':<14}" + "".join(f"{c:>13}" for c in cols))
    for name, m in results.items():
        print(f"{name:<14}" + "".join(f"{m[c]:>13.4f}" for c in cols))
    print("\nLower is better for every column. bias≈0 ideal; sigma_MAD=scatter;")
    print("outlier_frac=|Δz/(1+z)|>0.15; PIT_KS=0 means perfectly calibrated p(z).")


if __name__ == "__main__":
    main()
