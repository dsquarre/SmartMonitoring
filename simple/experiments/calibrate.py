"""Apply the calibration choice rule of ecg_protocol.md to one calibration step.

    python experiments/calibrate.py results/ecg/calib/step2 --order lr0.01,lr0.03,lr0.05,lr0.1

Every run in the directory is named <setting>_s<seed>. The rule: highest test AUPRC-AUC (mean test AUPRC over all
rounds) averaged over the tuning seeds; within --tie (0.005) of the best, the setting listed first in --order wins.
"""
import argparse
import glob
import os
import re

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir")
    p.add_argument("--order", required=True, help="settings in protocol order (tie-break), comma list")
    p.add_argument("--tie", type=float, default=0.005)
    a = p.parse_args()
    order = a.order.split(",")
    scores = {}
    for f in glob.glob(os.path.join(a.results_dir, "*_s*.csv")):
        if f.endswith(".partial.csv"):
            continue
        name = re.sub(r"_s\d+\.csv$", "", os.path.basename(f))
        df = pd.read_csv(f)
        scores.setdefault(name, []).append(df["test_auprc"].mean())
    missing = [s for s in order if s not in scores]
    if missing:
        raise SystemExit(f"missing runs for {missing}")
    table = {s: sum(scores[s]) / len(scores[s]) for s in order}
    best = max(table.values())
    choice = next(s for s in order if table[s] >= best - a.tie)
    for s in order:
        print(f"{s:20s} AUPRC-AUC {table[s]:.4f}  (seeds: {', '.join(f'{v:.4f}' for v in scores[s])})"
              + ("   <- chosen" if s == choice else ""))
    print(f"CHOICE {choice}")


if __name__ == "__main__":
    main()
