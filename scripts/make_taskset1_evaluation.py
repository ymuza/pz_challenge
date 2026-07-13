#!/usr/bin/env python3
"""Produce in-sample evaluation qp files for the official metrics.

The challenge's run_metrics.py compares predictions on the *training* file
(which carries truth) to that truth.  This script reuses the already-trained
models in submissions/aion/ to estimate p(z) on each training file and writes
evaluation/pz_challenge_taskset_1_{sim}_pz_evaluation_{scenario}.hdf5.
"""

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse

import aion_pz

SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "10yr"]
PUBLIC = "public"
SUBMIT = "submissions/aion"
EVAL = "evaluation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskset", type=int, default=1)
    args = ap.parse_args()
    ts = args.taskset

    os.makedirs(EVAL, exist_ok=True)
    device = os.environ.get("AION_PZ_DEVICE")
    for sim in SIMS:
        for scenario in SCENARIOS:
            tag = f"taskset_{ts}_{sim}_{scenario}"
            model = f"{SUBMIT}/pz_challenge_taskset_{ts}_{sim}_pz_model_{scenario}.pkl"
            train = f"{PUBLIC}/pz_challenge_taskset_{ts}_{sim}_training_{scenario}.hdf5"
            out = f"{EVAL}/pz_challenge_taskset_{ts}_{sim}_pz_evaluation_{scenario}.hdf5"
            if not (os.path.exists(model) and os.path.exists(train)):
                print(f"[skip] {tag}: missing model or training file")
                continue
            print(f"[run ] {tag} (estimate on training set) ...", flush=True)
            t0 = time.time()
            # estimate_only reads the training file as the "test" input: it has
            # the same magnitude columns + object_id, so predictions align.
            aion_pz.estimate_only(model, train, out, device=device)
            print(f"[done] {tag} in {time.time() - t0:.1f}s -> {out}", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
