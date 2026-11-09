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
                      accuracy_final, loss_final, brier_final, rounds_to_target (with --target, default 0.70),
                      energy_kj, latency_h
The decision above uses --metric only. summary.md also lists the first group against each comparator on every
metric (two-sided paired t, uncorrected: descriptive), and hypothesis_test_results.csv has every pair of groups.

Outputs: summary.md (printed too), per_seed.csv, comparison_summary.csv, hypothesis_test_results.csv,
comparison_strategy_selection_distribution.csv and plots:
    comparison_{auprc,loss,f2_accuracy,accuracy}_overlay.png       metric vs round, mean over seeds with 95% band
    comparison_{energy,latency}_vs_round.png                       cumulative client energy / latency vs round
    comparison_costs.png                                           energy, latency, server time per round (bars)
    comparison_pareto_{auprc,f2}_vs_{energy,latency,joint_cost}.png   per-seed points, mean with 95% CI, frontier
    comparison_pareto_3d_{auprc,f2}_vs_energy_latency.png
    comparison_strategy_selection_frequency.png, comparison_rules_kept.png   server rule kept per round
    comparison_per_seed_metrics.png, comparison_<metric>_per_seed.png        per-seed box plots
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

METRICS = {   # per-seed metric: higher is better
    "auprc_auc": True, "auprc_late": True, "auprc_final": True, "f2_final": True, "accuracy_final": True,
    "loss_final": False, "brier_final": False, "rounds_to_target": False, "energy_kj": False, "latency_h": False,
}
CURVES = {    # per-round curve: metrics.csv column, scale
    "auprc": ("test_auprc", 1.0), "loss": ("test_loss", 1.0), "f2": ("test_f2", 1.0),
    "accuracy": ("test_accuracy", 1.0), "energy_kj": ("cum_energy", 1e-3), "latency_h": ("cum_latency", 1 / 3600),
}
COLORS = ["crimson", "royalblue", "forestgreen", "darkorange", "purple", "teal", "magenta", "chocolate", "navy",
          "olive"]


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
        "f2_final": df["test_f2"].iloc[-5:].mean(), "accuracy_final": df["test_accuracy"].iloc[-5:].mean(),
        "loss_final": df["test_loss"].iloc[-5:].mean(), "brier_final": df["test_brier"].iloc[-5:].mean(),
        "rounds_to_target": int(df["round"].iloc[hit[0]]) if len(hit) else n + 1,
        "energy_kj": df["cum_energy"].iloc[-1] / 1e3, "latency_h": df["cum_latency"].iloc[-1] / 3600,
        "server_s": df["server_s"].mean(),
        "v_scores": df["server_evals"].mean() if "server_evals" in df else 0.0,
        "v_trainings": df["server_finetunes"].mean() if "server_finetunes" in df else 0.0,
        "choices": df["choice"].value_counts(normalize=True).to_dict(),
        "choice_counts": df["choice"].value_counts().to_dict(),
        "curves": {k: df[col].astype(float).to_numpy() * scale for k, (col, scale) in CURVES.items()},
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
          "| group | seeds | AUPRC-AUC | late | final | F2 (last 5) | accuracy (last 5) | loss (last 5) | "
          f"Brier (last 5) | rounds to AUPRC {a.target} | energy kJ | latency h | server s/round | V scores/round | "
          "V trainings/round | rules kept |",
          "|---" * 16 + "|"]
    for label, runs in groups.items():
        v = list(runs.values())
        m = lambda k: np.mean([r[k] for r in v])
        mix = pd.DataFrame([r["choices"] for r in v]).fillna(0).mean().sort_values(ascending=False)
        L.append(f"| {label} | {len(v)} | {m('auprc_auc'):.4f} | {m('auprc_late'):.4f} | {m('auprc_final'):.4f} | "
                 f"{m('f2_final'):.3f} | {m('accuracy_final'):.4f} | {m('loss_final'):.4f} | "
                 f"{m('brier_final'):.4f} | {m('rounds_to_target'):.1f} | {m('energy_kj'):.1f} | "
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

    pairs = [(first, lab) for lab in labels[1:]] + [(x, y) for i, x in enumerate(labels[1:], 1)
                                                    for y in labels[i + 1:]]
    tests = [paired_test(x, groups[x], y, groups[y], k) for x, y in pairs for k in METRICS]
    tests = [t for t in tests if t]
    L += ["", f"## {first} vs each comparator on every metric (descriptive)", "",
          "Difference = first group minus comparator (raw, not signed); two-sided paired t-test, not corrected for "
          "multiple comparisons. Only the `--metric` table above decides the hypothesis.", "",
          "| metric | better | " + " | ".join(labels[1:]) + " |", "|---" * (len(labels) + 1) + "|"]
    for k, higher in METRICS.items():
        cells = []
        for lab in labels[1:]:
            t = next((t for t in tests if t["strategy_A"] == first and t["strategy_B"] == lab and t["metric"] == k),
                     None)
            cells.append(f"{t['mean_diff']:+.4g} (p {t['p_value']:.2g})" if t else "")
        L.append(f"| {k} | {'higher' if higher else 'lower'} | " + " | ".join(cells) + " |")
    text = "\n".join(L)
    with open(os.path.join(a.output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)

    drop = ("choices", "choice_counts", "curves")
    per_seed = pd.DataFrame([{"group": lab, **{k: v for k, v in r.items() if k not in drop}}
                             for lab, runs in groups.items() for r in runs.values()])
    per_seed.to_csv(os.path.join(a.output_dir, "per_seed.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(a.output_dir, "hypothesis_test_results.csv"), index=False)
    cols = list(METRICS) + ["server_s", "v_scores", "v_trainings"]
    agg = per_seed.groupby("group", sort=False)[cols].agg(["mean", "std"])
    agg.columns = [f"{k}_{s}" for k, s in agg.columns]
    agg.insert(0, "num_runs", per_seed.groupby("group", sort=False).size())
    agg.to_csv(os.path.join(a.output_dir, "comparison_summary.csv"))
    plot(groups, a, labels)
    print(f"\nWritten to {os.path.abspath(a.output_dir)}")


def paired_test(name_a, runs_a, name_b, runs_b, metric):
    """Two-sided paired t-test and Wilcoxon of one metric over the seeds both groups have."""
    seeds = sorted(set(runs_a) & set(runs_b))
    va = np.array([runs_a[s][metric] for s in seeds], dtype=float)
    vb = np.array([runs_b[s][metric] for s in seeds], dtype=float)
    ok = ~np.isnan(va) & ~np.isnan(vb)
    va, vb = va[ok], vb[ok]
    if len(va) == 0:
        return None
    p_t = p_w = 1.0
    t_stat = w_stat = 0.0
    if len(va) > 1 and np.any(va != vb):
        res = stats.ttest_rel(va, vb)
        t_stat, p_t = float(res.statistic), float(res.pvalue)
        if np.isnan(p_t):
            t_stat, p_t = 0.0, 1.0
        try:
            res = stats.wilcoxon(va, vb)
            w_stat, p_w = float(res.statistic), float(res.pvalue)
        except ValueError:
            w_stat, p_w = float("nan"), float("nan")
    return {"metric": metric, "strategy_A": name_a, "strategy_B": name_b, "n_pairs": len(va),
            "mean_A": va.mean(), "mean_B": vb.mean(), "mean_diff": va.mean() - vb.mean(),
            "t_statistic": t_stat, "p_value": p_t, "significant_p05": bool(p_t < 0.05),
            "wilcoxon_stat": w_stat, "wilcoxon_p_value": p_w}


def mean_ci(values):
    """Mean and 95% CI half-width (t) over seeds; values is (seeds, ...)."""
    arr = np.asarray(values, dtype=float)
    mean = arr.mean(axis=0)
    if len(arr) < 2:
        return mean, np.zeros_like(mean)
    return mean, stats.t.ppf(0.975, len(arr) - 1) * arr.std(axis=0, ddof=1) / np.sqrt(len(arr))


def plot(groups, a, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(style="darkgrid")
    except ImportError:
        pass
    out = a.output_dir
    color = {lab: COLORS[i % len(COLORS)] for i, lab in enumerate(labels)}

    def save(name):
        plt.tight_layout()
        plt.savefig(os.path.join(out, name), dpi=200)
        plt.close("all")

    def overlay(key, ylabel, title, name):
        plt.figure(figsize=(9, 5.5))
        for lab in labels:
            curves = [r["curves"][key] for r in groups[lab].values()]
            n = min(len(c) for c in curves)
            mean, half = mean_ci([c[:n] for c in curves])
            x = np.arange(1, n + 1)
            plt.plot(x, mean, color=color[lab], linewidth=1.8, label=f"{lab} (n={len(curves)})")
            plt.fill_between(x, mean - half, mean + half, color=color[lab], alpha=0.15)
        plt.xlabel("Federated round")
        plt.ylabel(f"{ylabel} (mean over seeds, 95% CI)")
        plt.title(title)
        plt.legend()
        save(name)

    overlay("auprc", "Test AUPRC", "Test AUPRC vs round", "comparison_auprc_overlay.png")
    overlay("loss", "Test loss", "Global test loss vs round", "comparison_loss_overlay.png")
    overlay("f2", "Test F2", "Test F2 vs round", "comparison_f2_accuracy_overlay.png")
    overlay("accuracy", "Test accuracy", "Test accuracy vs round", "comparison_accuracy_overlay.png")
    overlay("energy_kj", "Cumulative client energy (kJ)", "Cumulative energy vs round",
            "comparison_energy_vs_round.png")
    overlay("latency_h", "Cumulative round latency (h)", "Cumulative latency vs round",
            "comparison_latency_vs_round.png")

    # costs per run: client energy, latency, server time per round
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, (k, ylabel) in zip(axes, [("energy_kj", "Total client energy (kJ)"), ("latency_h", "Total latency (h)"),
                                      ("server_s", "Server time per round (s)")]):
        ci = [mean_ci([r[k] for r in groups[lab].values()]) for lab in labels]
        ax.bar(range(len(labels)), [m for m, _ in ci], yerr=[h for _, h in ci], capsize=5,
               color=[color[lab] for lab in labels], alpha=0.85, edgecolor="black")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
    fig.suptitle("Cost per run (mean over seeds, 95% CI)")
    save("comparison_costs.png")

    # trade-offs: per-seed points, group mean with 95% CI, empirical Pareto frontier (lower cost, higher score)
    scores = [("auprc_auc", "AUPRC-AUC (mean test AUPRC over rounds)", "auprc"),
              ("f2_final", "Final F2 (last 5 rounds)", "f2")]
    costs = [("energy_kj", "Total client energy (kJ)", "energy"), ("latency_h", "Total latency (h)", "latency"),
             ("edp", "Energy-delay product (kJ x h)", "joint_cost")]
    for yk, ylabel, ysuf in scores:
        yname = ylabel.split(" (")[0]
        for xk, xlabel, xsuf in costs:
            plt.figure(figsize=(9, 6))
            points = []
            for lab in labels:
                v = list(groups[lab].values())
                xs = np.array([r["energy_kj"] * r["latency_h"] if xk == "edp" else r[xk] for r in v])
                ys = np.array([r[yk] for r in v])
                (xm, xh), (ym, yh) = mean_ci(xs), mean_ci(ys)
                plt.scatter(xs, ys, color=color[lab], alpha=0.3, s=30)
                plt.errorbar(xm, ym, xerr=xh, yerr=yh, fmt="s", color=color[lab], markersize=9, capsize=6,
                             linewidth=2, label=f"{lab} (mean, 95% CI)")
                points.append((xm, ym))
            front, best = [], -np.inf
            for x, y in sorted(points):
                if y > best:
                    front.append((x, y))
                    best = y
            if len(front) > 1:
                plt.plot(*zip(*front), "k--", linewidth=1.5, label="Empirical Pareto frontier")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f"Trade-off: {yname} vs {xlabel.split(' (')[0].lower()}")
            plt.legend(loc="best")
            save(f"comparison_pareto_{ysuf}_vs_{xsuf}.png")

        fig = plt.figure(figsize=(10, 7.5))
        ax = fig.add_subplot(111, projection="3d")
        for lab in labels:
            v = list(groups[lab].values())
            lat, en, sc = (np.array([r[k] for r in v]) for k in ("latency_h", "energy_kj", yk))
            ax.scatter(lat, en, sc, color=color[lab], alpha=0.3, s=25)
            ax.scatter([lat.mean()], [en.mean()], [sc.mean()], color=color[lab], s=120, marker="s",
                       edgecolors="black", linewidth=1.5, label=f"{lab} (mean)")
            ax.plot([lat.mean()] * 2, [en.mean()] * 2, [sc.min(), sc.mean()], color=color[lab], linestyle=":")
        ax.set_xlabel("Total latency (h)")
        ax.set_ylabel("Total client energy (kJ)")
        ax.set_zlabel(yname)
        ax.set_title(f"3D trade-off: {yname} vs latency and energy")
        ax.legend(loc="upper left")
        save(f"comparison_pareto_3d_{ysuf}_vs_energy_latency.png")

    # server rule kept: mean rounds per seed (grouped bars, with CSV) and share of rounds (stacked)
    rows = []
    for lab in labels:
        v = list(groups[lab].values())
        counts = pd.DataFrame([r["choice_counts"] for r in v]).fillna(0)
        total = counts.to_numpy().sum()
        for rule in counts.columns:
            rows.append({"group": lab, "agg_strategy": rule, "num_runs": len(v),
                         "total_rounds_selected": int(counts[rule].sum()),
                         "avg_rounds_per_seed": round(float(counts[rule].mean()), 2),
                         "percentage_selected": round(100.0 * counts[rule].sum() / total, 2)})
    dist = pd.DataFrame(rows)
    dist.to_csv(os.path.join(out, "comparison_strategy_selection_distribution.csv"), index=False)
    rules = sorted(dist["agg_strategy"].unique())
    width = 0.8 / len(labels)
    x = np.arange(len(rules))
    plt.figure(figsize=(min(14, max(9, 0.6 * len(rules) * len(labels))), 5.5))
    for i, lab in enumerate(labels):
        d = dist[dist["group"] == lab].set_index("agg_strategy")["avg_rounds_per_seed"]
        bars = plt.bar(x + (i - len(labels) / 2 + 0.5) * width, [float(d.get(rule, 0.0)) for rule in rules], width,
                       label=lab, color=color[lab], edgecolor="black", alpha=0.85)
        for b in bars:
            if b.get_height() > 0:
                plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1f}", ha="center",
                         va="bottom", fontsize=8)
    plt.xticks(x, [rule.upper() for rule in rules])
    plt.xlabel("Server rule kept")
    plt.ylabel("Mean rounds kept per seed")
    plt.title("Server rule selection across seeds")
    plt.legend()
    save("comparison_strategy_selection_frequency.png")

    mix = pd.DataFrame({lab: pd.DataFrame([r["choices"] for r in groups[lab].values()]).fillna(0).mean()
                        for lab in labels}).fillna(0).T
    if mix.shape[1] > 1:
        mix.plot(kind="bar", stacked=True, figsize=(9, 5), colormap="tab20")
        plt.ylabel("Share of rounds")
        plt.title("Server rule kept per round")
        plt.xticks(rotation=20, ha="right")
        save("comparison_rules_kept.png")

    # per-seed distributions
    keys = ["auprc_auc", "auprc_final", "f2_final", "accuracy_final", "energy_kj", "latency_h"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, k in zip(axes.flat, keys):
        ax.boxplot([[r[k] for r in groups[lab].values()] for lab in labels])
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(f"{k} ({'higher' if METRICS[k] else 'lower'} is better)")
    fig.suptitle("Per-seed metrics")
    save("comparison_per_seed_metrics.png")

    plt.figure(figsize=(9, 5))
    plt.boxplot([[r[a.metric] for r in groups[lab].values()] for lab in labels])
    plt.ylabel(a.metric)
    plt.title(f"Per-seed {a.metric}")
    plt.xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right")
    save(f"comparison_{a.metric}_per_seed.png")


if __name__ == "__main__":
    main()
