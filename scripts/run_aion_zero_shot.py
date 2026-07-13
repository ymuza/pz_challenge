#!/usr/bin/env python3
"""Run the zero-shot AION generative-p(z) baseline on the challenge data.

Usage:
    python scripts/run_aion_zero_shot.py TEST_FILE OUTPUT_FILE
    # e.g.
    python scripts/run_aion_zero_shot.py \
        public/pz_challenge_taskset_1_cardinal_test_1yr.hdf5 \
        zeroshot_taskset1_cardinal_1yr.hdf5

Set AION_PZ_DEVICE=cpu to force CPU (CUDA is auto-detected otherwise).
Add --native to keep AION's own redshift bin centers instead of resampling
onto the fixed metrics grid.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import aion_pz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_file")
    parser.add_argument("output_file")
    parser.add_argument(
        "--native",
        action="store_true",
        help="Use AION's native redshift bins instead of resampling to Z_GRID.",
    )
    parser.add_argument("--model", default="polymathic-ai/aion-base")
    args = parser.parse_args()

    aion_pz.zero_shot_estimate(
        args.test_file,
        args.output_file,
        model_name=args.model,
        device=os.environ.get("AION_PZ_DEVICE"),
        resample=not args.native,
    )
    print(f"Wrote zero-shot p(z) to {args.output_file}")


if __name__ == "__main__":
    main()
