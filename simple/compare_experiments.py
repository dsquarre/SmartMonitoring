#!/usr/bin/env python3
"""
compare_experiments.py - summarise finished simulate_fl.py runs and test the first group against the others.

    python compare_experiments.py --dirs results/h1_select results/h1_fedavg results/h1_fedyogi \
        --labels Selector FedAvg FedYogi --best-of FedAvg,FedYogi -o results/hypothesis1

Each --dirs entry is a group: a run folder, a folder of s<seed> run folders, or a glob ("results/h1_select/s*").
The FIRST group is the method under test; every other group (and, with --best-of, the per-seed best of the named
groups) is a comparator. Comparisons are paired by seed (only seeds present in both groups).

Per comparison: mean difference with 95% CI, wins (seeds where the first group is better), one-sided paired t-test,
Wilcoxon signed-rank test and the Holm-adjusted t-test p-value across the comparisons of this call.
    --test sup        superiority: H0 the first group is not better (default)
    --test ni         non-inferiority: H0 the first group is worse by at least --margin (default 0.01)
    --metric          auprc_auc (default; mean test AUPRC over all rounds), auprc_late, auprc_final, f2_final,
                      brier_final, rounds_to_target (with --target, default 0.70), energy_kj, latency_h
Writes summary.md (printed too), per_seed.csv and plots (AUPRC vs round, rules kept, per-seed metric).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

METRICS = {   # name: (column in per-seed table, higher is better)
    "auprc_auc": True, "auprc_late": True, "auprc_final": True, "f2_final": True,
    "brier_final": False, "rounds_to_target": False, "energy_kj": False, "latency_h": False,
}


def find_runs(pattern):
    """Run folders (each holds metrics.csv) for a --dirs entry."""
    out = []
    for path in sorted(glob.glob(pattern)) or [pattern]:
        if os.path.isfile(os.path.join(path, "metrics.csv")):
            out.append(path)
        else:
            out += sorted(os.path.dirname(f) for f in glob.glob(os.path.join(path, "s*", "metrics.csv")))
    return out


def summarise_run(run_dir, target):
    df = pd.read_csv(os.path.join(run_dir, "metrics.csv")).sort_values("round")
    finished = os.path.exists(os.path.join(run_dir, "summary.json"))
    au = df["test_auprc"].astype(float)
    n = len(df)
    smooth = au.rolling(3, min_periods=3).mean()
    hit = np.flatnonzero(smooth.to_numpy() >= target)
    return {
        "seed": int(df["seed"].iloc[0]), "rounds": n, "finished": finished,
        "auprc_auc": au.mean(), "auprc_late": au.iloc[n // 2:].mean(), "auprc_final": au.iloc[-5:].mean(),
        "f2_final": df["test_f2"].iloc[-5:].mean(), "brier_final": df["test_brier"].iloc[-5:].mean(),
        "rounds_to_target": int(df["round"].iloc[hit[0]]) if len(hit) else n + 1,
        "energy_kj": df["cum_energy"].iloc[-1] / 1e3, "latency_h": df["cum_latency"].iloc[-1] / 3600,
        "server_s": df["server_s"].mean(),
        "v_scores": df["server_evals"].mean() if "server_evals" in df else 0.0,
        "v_trainings": df["server_finetunes"].mean() if "server_finetunes" in df else 0.0,
        "choices": df["choice"].value_counts(normalize=True).to_dict(),
        "curve": au.to_numpy(),
    }


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def compare(gain, test, margin):
    """gain = per-seed improvement of the first group (positive = better). One-sided t and Wilcoxon."""
    shift = margin if test == "ni" else 0.0
    n = len(gain)
    mean, sd = gain.mean(), gain.std(ddof=1) if n > 1 else 0.0
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else float("nan")
    p_t = float(stats.ttest_1samp(gain + shift, 0.0, alternative="greater").pvalue) if n > 1 and sd > 0 else (
        0.0 if mean + shift > 0 else 1.0)
    try:
        p_w = float(stats.wilcoxon(gain + shift, alternative="greater").pvalue)
    except ValueError:
        p_w = float("nan")
    return {"n": n, "mean": mean, "ci": half, "wins": int((gain > 0).sum()), "p_t": p_t, "p_w": p_w}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, help="groups: run folders, parents of s<seed> folders, or globs")
    ap.add_argument("--labels", nargs="+", help="a label per group (default: the folder names)")
    ap.add_argument("--best-of", action="append", default=[],
                    help="comma list of labels: also compare with their per-seed best (repeatable)")
    ap.add_argument("--metric", choices=list(METRICS), default="auprc_auc")
    ap.add_argument("--test", choices=["sup", "ni"], default="sup")
    ap.add_argument("--margin", type=float, default=0.01, help="non-inferiority margin (metric units)")
    ap.add_argument("--target", type=float, default=0.70, help="AUPRC target for rounds_to_target")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--allow-unfinished", action="store_true", help="include runs without summary.json")
    ap.add_argument("-o", "--output-dir", default=os.path.join("results", "comparison"))
    a = ap.parse_args()
    labels = a.labels or [os.path.basename(os.path.normpath(d.rstrip("*/\\"))) for d in a.dirs]
    if len(labels) != len(a.dirs):
        raise SystemExit("give one label per --dirs entry")
    os.makedirs(a.output_dir, exist_ok=True)

    groups, skipped = {}, []
    for label, pattern in zip(labels, a.dirs):
        runs = {}
        for d in find_runs(pattern):
            r = summarise_run(d, a.target)
            if r["finished"] or a.allow_unfinished:
                runs[r["seed"]] = r
            else:
                skipped.append(d)
        if not runs:
            raise SystemExit(f"no finished runs for {label} ({pattern})")
        groups[label] = runs

    L = [f"# Comparison: {' vs '.join(labels)}", "",
         f"Metric `{a.metric}` ({'higher' if METRICS[a.metric] else 'lower'} is better); test "
         f"{'superiority' if a.test == 'sup' else f'non-inferiority (margin {a.margin})'}; one-sided paired t-test "
         f"(Holm across the comparisons below) and Wilcoxon; alpha {a.alpha}.", ""]
    if skipped:
        L += [f"Unfinished runs left out: {len(skipped)} ({', '.join(skipped[:3])}{' ...' if len(skipped) > 3 else ''})", ""]

    L += ["## Groups (mean over seeds)", "",
          "| group | seeds | AUPRC-AUC | late | final | F2 (last 5) | Brier (last 5) | rounds to AUPRC "
          f"{a.target} | energy kJ | latency h | server s/round | V scores/round | V trainings/round | rules kept |",
          "|---" * 14 + "|"]
    for label, runs in groups.items():
        v = list(runs.values())
        m = lambda k: np.mean([r[k] for r in v])
        mix = pd.DataFrame([r["choices"] for r in v]).fillna(0).mean().sort_values(ascending=False)
        L.append(f"| {label} | {len(v)} | {m('auprc_auc'):.4f} | {m('auprc_late'):.4f} | {m('auprc_final'):.4f} | "
                 f"{m('f2_final'):.3f} | {m('brier_final'):.4f} | {m('rounds_to_target'):.1f} | {m('energy_kj'):.1f} | "
                 f"{m('latency_h'):.2f} | {m('server_s'):.2f} | {m('v_scores'):.1f} | {m('v_trainings'):.1f} | "
                 + ", ".join(f"{k} {x:.0%}" for k, x in mix.items()) + " |")

    first = labels[0]
    sign = 1.0 if METRICS[a.metric] else -1.0
    comparators = [(lab, [lab]) for lab in labels[1:]] + [(f"best of {b}", b.split(",")) for b in a.best_of]
    rows = []
    for name, members in comparators:
        missing = [m for m in members if m not in groups]
        if missing:
            raise SystemExit(f"--best-of names unknown labels: {missing}")
        seeds = sorted(set(groups[first]).intersection(*[set(groups[m]) for m in members]))
        if not seeds:
            rows.append((name, None))
            continue
        ours = np.array([groups[first][s][a.metric] for s in seeds])
        theirs = np.array([[groups[m][s][a.metric] for s in seeds] for m in members])
        best = theirs.max(axis=0) if METRICS[a.metric] else theirs.min(axis=0)
        rows.append((name, compare(sign * (ours - best), a.test, a.margin)))
    valid = [r for _, r in rows if r]
    adj = iter(holm([r["p_t"] for r in valid])) if valid else iter([])
    L += ["", f"## {first} vs each comparator", "",
          "| comparator | seeds | mean improvement | 95% CI | wins | every seed | t p | Wilcoxon p | Holm p | verdict |",
          "|---" * 10 + "|"]
    for name, r in rows:
        if r is None:
            L.append(f"| {name} | 0 | no common seeds ||||||||")
            continue
        p_holm = next(adj)
        ok = p_holm < a.alpha
        L.append(f"| {name} | {r['n']} | {r['mean']:+.4f} | ±{r['ci']:.4f} | {r['wins']}/{r['n']} | "
                 f"{'yes' if r['wins'] == r['n'] else 'no'} | {r['p_t']:.3g} | {r['p_w']:.3g} | {p_holm:.3g} | "
                 f"**{'supported' if ok else 'not supported'}** |")
    L += ["", "Improvement = first group minus comparator, signed so that positive is better."]
    text = "\n".join(L)
    with open(os.path.join(a.output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)

    pd.DataFrame([{"group": lab, **{k: v for k, v in r.items() if k not in ("choices", "curve")}}
                  for lab, runs in groups.items() for r in runs.values()]).to_csv(
        os.path.join(a.output_dir, "per_seed.csv"), index=False)
    plot(groups, a, labels)
    print(f"\nWritten to {os.path.abspath(a.output_dir)}")


def plot(groups, a, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5.5))
    for label in labels:
        curves = [r["curve"] for r in groups[label].values()]
        n = min(len(c) for c in curves)
        arr = np.array([c[:n] for c in curves])
        mean = arr.mean(axis=0)
        half = 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros(n)
        x = np.arange(1, n + 1)
        plt.plot(x, mean, label=f"{label} (n={len(arr)})")
        plt.fill_between(x, mean - half, mean + half, alpha=0.15)
    plt.xlabel("Federated round")
    plt.ylabel("Test AUPRC (mean over seeds, 95% CI)")
    plt.title("Test AUPRC vs round")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(a.output_dir, "comparison_auprc_overlay.png"), dpi=200)
    plt.close()

    mix = pd.DataFrame({lab: pd.DataFrame([r["choices"] for r in groups[lab].values()]).fillna(0).mean()
                        for lab in labels}).fillna(0).T
    if mix.shape[1] > 1:
        mix.plot(kind="bar", stacked=True, figsize=(9, 5), colormap="tab20")
        plt.ylabel("Share of rounds")
        plt.title("Server rule kept per round")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(a.output_dir, "comparison_rules_kept.png"), dpi=200)
        plt.close()

    plt.figure(figsize=(9, 5))
    plt.boxplot([[r[a.metric] for r in groups[lab].values()] for lab in labels])
    plt.ylabel(a.metric)
    plt.title(f"Per-seed {a.metric}")
    plt.xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(a.output_dir, f"comparison_{a.metric}_per_seed.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
