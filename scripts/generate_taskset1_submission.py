#!/usr/bin/env python3
"""Generate AION qp outputs for every task-set-1 config (sim x scenario).

For each of cardinal/flagship x 1yr/10yr: train the head (with PIT
recalibration) on the full training file and write a qp estimate for the test
file, plus the pickled model, into submissions/aion/ using the challenge's
naming convention.

Usage:
    python scripts/generate_taskset1_submission.py [--public-area public]
        [--out-dir submissions/aion] [--device cpu|mps|cuda]
"""

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import aion_pz

SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "10yr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskset", type=int, default=1, help="task set number (1-4)")
    ap.add_argument("--public-area", default="public")
    ap.add_argument("--out-dir", default="submissions/aion")
    ap.add_argument("--device", default=os.environ.get("AION_PZ_DEVICE"))
    args = ap.parse_args()

    ts = args.taskset
    os.makedirs(args.out_dir, exist_ok=True)

    for sim in SIMS:
        for scenario in SCENARIOS:
            tag = f"taskset_{ts}_{sim}_{scenario}"
            train_file = f"{args.public_area}/pz_challenge_taskset_{ts}_{sim}_training_{scenario}.hdf5"
            test_file = f"{args.public_area}/pz_challenge_taskset_{ts}_{sim}_test_{scenario}.hdf5"
            model_file = f"{args.out_dir}/pz_challenge_taskset_{ts}_{sim}_pz_model_{scenario}.pkl"
            out_file = f"{args.out_dir}/pz_challenge_taskset_{ts}_{sim}_pz_estimate_{scenario}.hdf5"

            if not (os.path.exists(train_file) and os.path.exists(test_file)):
                print(f"[skip] {tag}: missing input files")
                continue

            print(f"[run ] {tag} ...", flush=True)
            t0 = time.time()
            aion_pz.train_and_estimate(
                train_file, test_file, out_file,
                device=args.device, save_model_to=model_file,
            )
            print(f"[done] {tag} in {time.time() - t0:.1f}s -> {out_file}", flush=True)

    print("ALL DONE")


if __name__ == "__main__":
    main()
