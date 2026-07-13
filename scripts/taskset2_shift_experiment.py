#!/usr/bin/env python3
"""Quantify the Task Set 2 selection bias and test covariate-shift correction.

There is no truth for the faint TS2 test set, so we mirror the train->test shift
using truth we do have: train the head on the BRIGHT bulk of the TS2 training
set (i < FAINT_CUT) and validate on its FAINT tail (i >= FAINT_CUT), which
overlaps the test magnitude regime.  Compare:
  - naive (train on bright bulk only)
  - shift-corrected (reweight bright bulk toward the TS2 *test* magnitudes)
  - oracle (train on a representative mix incl. faint) as an upper bound
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

SIM, SC = "cardinal", "1yr"
FAINT_CUT = 23.5


def pm(z_phot, z_true):
    dz = (z_phot - z_true) / (1.0 + z_true)
    return dict(bias=float(np.median(dz)),
                sigma_MAD=float(1.4826 * np.median(np.abs(dz - np.median(dz)))),
                outlier=float(np.mean(np.abs(dz) > 0.15)))


def pmean(pz, grid):
    dz = grid[1] - grid[0]
    return (pz * grid).sum(axis=1) * dz / ((pz.sum(axis=1) * dz) + 1e-12)


def sub(d, idx):
    return {k: v[idx] for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=os.environ.get("AION_PZ_DEVICE"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train = aion_pz.load_catalog(f"public/pz_challenge_taskset_2_{SIM}_training_{SC}.hdf5")
    test = aion_pz.load_catalog(f"public/pz_challenge_taskset_2_{SIM}_test_{SC}.hdf5")
    z = np.asarray(train[aion_pz.REDSHIFT_COL], float)
    mi = np.asarray(train["mag_i_lsst"], float)
    good = np.isfinite(z) & np.isfinite(mi)
    train, z, mi = sub(train, np.where(good)[0]), z[good], mi[good]

    faint = mi >= FAINT_CUT
    bright = ~faint
    print(f"TS2 {SIM} {SC}: {bright.sum()} bright (i<{FAINT_CUT}), {faint.sum()} faint (i>={FAINT_CUT})")

    # Validation = faint tail; fit pool = bright bulk (split off a calib slice).
    rng = np.random.default_rng(args.seed)
    val_idx = np.where(faint)[0]
    bright_idx = rng.permutation(np.where(bright)[0])
    calib_idx = bright_idx[:8000]
    fit_idx = bright_idx[8000:]
    # oracle: a representative fit pool that also includes faint galaxies
    all_idx = rng.permutation(np.where(good[good])[0] if False else np.arange(len(z)))
    oracle_fit = np.setdiff1d(all_idx, val_idx)[: len(fit_idx)]

    print("Loading AION + embeddings (fit / calib / val / oracle) ...")
    t0 = time.time()
    model, cm, device = aion_pz.load_aion(device=args.device)
    grid = aion_pz.Z_GRID
    X_fit = aion_pz.build_design_matrix(model, cm, sub(train, fit_idx), device)
    X_cal = aion_pz.build_design_matrix(model, cm, sub(train, calib_idx), device)
    X_val = aion_pz.build_design_matrix(model, cm, sub(train, val_idx), device)
    X_ora = aion_pz.build_design_matrix(model, cm, sub(train, oracle_fit), device)
    print(f"  embeddings ready in {time.time()-t0:.1f}s")
    z_val = z[val_idx]

    # shift weights: reweight the bright fit pool toward the TS2 TEST magnitudes
    w_shift = aion_pz.feature_shift_weights(sub(train, fit_idx), test)

    def run(name, X, zt, w):
        head = aion_pz.train_head(X, zt, sample_weight=w)
        recal = aion_pz.fit_pit_recalibration(aion_pz.predict_pz(head, X_cal), grid, z[calib_idx])
        pz = aion_pz.apply_pit_recalibration(aion_pz.predict_pz(head, X_val), grid, recal)
        m = pm(pmean(pz, grid), z_val)
        m["PIT_KS"] = aion_pz.pit_ks_stat(pz, grid, z_val)
        return name, m

    rows = [
        run("naive (bright)", X_fit, z[fit_idx], None),
        run("shift-corrected", X_fit, z[fit_idx], w_shift),
        run("oracle (repr.)", X_ora, z[oracle_fit], None),
    ]

    print(f"\n{'='*74}\nTS2 shift test — validate on FAINT tail (i>={FAINT_CUT}, N={len(val_idx)}) with truth\n{'='*74}")
    cols = ["bias", "sigma_MAD", "outlier", "PIT_KS"]
    print(f"{'method':<18}" + "".join(f"{c:>13}" for c in cols))
    for name, m in rows:
        print(f"{name:<18}" + "".join(f"{m[c]:>13.4f}" for c in cols))
    print("\nnaive = train on bright only (the TS2 hazard); shift-corrected = reweight")
    print("bright toward test mags; oracle = train incl. faint (unavailable in real TS2).")


if __name__ == "__main__":
    main()
