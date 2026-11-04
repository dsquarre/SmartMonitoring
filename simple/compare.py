"""Summarise finished runs and compare a target strategy with every other one, paired by seed.

    python compare.py results/main                       # target: select_train
    python compare.py results/main --target select --last 10

Per run: AUC = mean test AUPRC over all rounds (the supplement's "accuracy AUC", on AUPRC), the mean of the last
`--last` rounds, the final round, F2 / Brier over the last rounds, total client energy and latency, the server's cost of
its decisions (seconds per round in the controller, models scored on V and training passes over V per round) and how
often each rule was kept. Paired differences report the mean, a 95% t interval, the number of seeds where
the target is higher (better for AUPRC / F2; worse for Brier, energy and latency) and the paired t-test p-value.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ("auprc_auc", "auprc_last", "auprc_final", "f2_last", "brier_last", "energy_kj", "latency_h", "server_s",
           "server_evals")
INFO = ("server_finetunes", "rule_mix")


def summarise(df: pd.DataFrame, last: int) -> dict:
    df = df.sort_values("round")
    tail = df.tail(last)
    return {
        "rounds": int(df["round"].max()),
        "auprc_auc": df["test_auprc"].mean(),
        "auprc_last": tail["test_auprc"].mean(),
        "auprc_final": df["test_auprc"].iloc[-1],
        "f2_last": tail["test_f2"].mean(),
        "brier_last": tail["test_brier"].mean(),
        "energy_kj": df["cum_energy"].iloc[-1] / 1e3,
        "latency_h": df["cum_latency"].iloc[-1] / 3600,
        "server_s": df["server_s"].mean() if "server_s" in df else float("nan"),
        "server_evals": df["server_evals"].mean() if "server_evals" in df else float("nan"),
        "server_finetunes": df["server_finetunes"].mean() if "server_finetunes" in df else float("nan"),
        "rule_mix": " ".join(f"{k}:{v:.0%}" for k, v in df["choice"].value_counts(normalize=True).items()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir")
    p.add_argument("--target", default="select_train")
    p.add_argument("--last", type=int, default=10)
    a = p.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(a.results_dir, "*_s*.csv"))):
        if f.endswith(".partial.csv"):
            continue
        df = pd.read_csv(f)
        rows.append({"strategy": df["strategy"].iloc[0], "seed": int(df["seed"].iloc[0]), **summarise(df, a.last)})
    if not rows:
        raise SystemExit(f"no finished runs in {a.results_dir}")
    runs = pd.DataFrame(rows)

    lines = ["## Per strategy (mean ± sd over seeds)", "",
             "| strategy | seeds | " + " | ".join(METRICS + INFO) + " |",
             "|---" * (len(METRICS) + len(INFO) + 2) + "|"]
    for strat, g in runs.groupby("strategy"):
        cells = [f"{g[m].mean():.4f} ± {g[m].std(ddof=1) if len(g) > 1 else 0:.4f}" for m in METRICS]
        cells += [f"{g['server_finetunes'].mean():.2f}", g.sort_values("seed")["rule_mix"].iloc[0] + (" (first seed)" if len(g) > 1 else "")]
        lines.append(f"| {strat} | {len(g)} | " + " | ".join(cells) + " |")

    if a.target in set(runs["strategy"]):
        lines += ["", f"## {a.target} minus each strategy (paired by seed)", "",
                  "| vs | metric | n | mean diff | 95% CI | target higher | p (paired t) |",
                  "|---|---|---|---|---|---|---|"]
        tgt = runs[runs["strategy"] == a.target].set_index("seed")
        for strat, g in runs[runs["strategy"] != a.target].groupby("strategy"):
            other = g.set_index("seed")
            seeds = sorted(set(tgt.index) & set(other.index))
            if not seeds:
                continue
            for m in METRICS[:-2]:
                d = (tgt.loc[seeds, m] - other.loc[seeds, m]).to_numpy(dtype=float)
                n = len(d)
                half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
                pval = stats.ttest_1samp(d, 0.0).pvalue if n > 1 else float("nan")
                lines.append(f"| {strat} | {m} | {n} | {d.mean():+.4f} | ±{half:.4f} | {int((d > 0).sum())}/{n} | "
                             f"{pval:.3g} |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(a.results_dir, "comparison.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
