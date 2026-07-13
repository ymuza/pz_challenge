#!/usr/bin/env python3
"""Run the official challenge metrics for task set 1 only.

Same calls as scripts/run_metrics.py, restricted to taskset_1 (we only have
those data + evaluation files).  Saves the RAIL diagnostic plots and their
underlying data, and also prints the key numeric summaries.
"""

import argparse
import os

from pz_data_challenge import metrics

DATADIR = "public"
EVALUATION_DIR = "evaluation"
SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "10yr"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskset", type=int, default=1)
    args = ap.parse_args()
    TASKSETS = [f"taskset_{args.taskset}"]
    PLOTS_DIR = f"metric_plots_taskset_{args.taskset}"
    data_dict = {}
    for taskset_ in TASKSETS:
        for sim_ in SIMS:
            for scenario_ in SCENARIOS:
                prefix = f"{taskset_}_{sim_}_{scenario_}"
                sub = metrics.get_truth_and_qp_ensemble(
                    DATADIR, EVALUATION_DIR, taskset_, sim_, scenario_
                )
                test_data = sub[f"{prefix}_test"]
                submit_data = sub[f"{prefix}_evaluate"]

                data_dict.update(metrics.point_metrics_plot(f"{prefix}_point", test_data, submit_data))
                data_dict.update(metrics.point_v_redshfit_plot(f"{prefix}_point_v_redshift", test_data, submit_data))
                data_dict.update(metrics.point_v_mag_plot(f"{prefix}_point_v_mag", test_data, submit_data))
                data_dict.update(metrics.plot_pit_prob_plot(f"{prefix}_pit_prob", test_data, submit_data))
                data_dict.update(metrics.plot_pit_qq_plot(f"{prefix}_pit_qq", test_data, submit_data))
                print(f"[metrics] {prefix} done")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    for k, v in data_dict.items():
        try:
            v.savefig(k.replace("0.0", ""), PLOTS_DIR)
            v.savedata(PLOTS_DIR)
        except Exception as e:  # keep going if one plotter can't save
            print(f"  [warn] save failed for {k}: {e}")

    print(f"\nSaved {len(data_dict)} plots to {PLOTS_DIR}")
