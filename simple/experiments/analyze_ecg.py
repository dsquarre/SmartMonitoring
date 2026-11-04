"""Evaluate the hypotheses of ecg_protocol.md on the finished test runs and write results/ecg/REPORT.md.

    python experiments/analyze_ecg.py            # all papers
    python experiments/analyze_ecg.py --p1c-rate # only print the selector's FedYogi rate (the p of P1c)

Per run: AUC = mean test AUPRC over all rounds (primary), late = mean over the second half, final = mean of the last
5 rounds. Differences are paired by seed: mean, 95% t interval, wins (seeds with a positive difference).
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "ecg")
SEEDS = [42, 43, 44, 45, 46]


def load(dirname):
    runs = {}
    for f in glob.glob(os.path.join(ROOT, dirname, "*_s*.csv")):
        if f.endswith(".partial.csv"):
            continue
        name = re.sub(r"_s\d+\.csv$", "", os.path.basename(f))
        seed = int(re.search(r"_s(\d+)\.csv$", f).group(1))
        df = pd.read_csv(f).sort_values("round")
        n = len(df)
        runs.setdefault(name, {})[seed] = {
            "auc": df["test_auprc"].mean(), "late": df["test_auprc"].iloc[n // 2:].mean(),
            "final": df["test_auprc"].iloc[-5:].mean(), "f2": df["test_f2"].iloc[-5:].mean(),
            "brier": df["test_brier"].iloc[-5:].mean(),
            "server_s": df["server_s"].mean(), "evals": df["server_evals"].mean(),
            "finetunes": df["server_finetunes"].mean(),
            "yogi_rate": (df["choice"] == "fedyogi").mean(), "rounds": n,
            "choices": df["choice"].value_counts(normalize=True).to_dict(),
        }
    return runs


def vec(runs, name, key, seeds):
    return np.array([runs[name][s][key] for s in seeds])


def common(runs, names):
    seeds = set(SEEDS)
    for n in names:
        seeds &= set(runs.get(n, {}))
    return sorted(seeds)


def paired(d):
    n = len(d)
    if n < 2:
        return f"{d.mean():+.4f} (n={n})"
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    p = stats.ttest_1samp(d, 0.0).pvalue
    return f"{d.mean():+.4f} ± {half:.4f} ({int((d > 0).sum())}/{n}, p={p:.3g})"


def table(runs, names, lines):
    lines += ["| strategy | seeds | AUC | late | final | F2 (last 5) | Brier | server s/round | V scores/round | "
              "V trainings/round | kept |", "|---" * 11 + "|"]
    for n in names:
        if n not in runs:
            lines.append(f"| {n} | 0 | (not run yet) |||||||||")
            continue
        r = runs[n]
        s = [x for x in SEEDS if x in r]
        m = lambda k: np.mean([r[x][k] for x in s])
        mix = pd.DataFrame([r[x]["choices"] for x in s]).fillna(0).mean().sort_values(ascending=False)
        lines.append(f"| {n} | {len(s)} | {m('auc'):.4f} | {m('late'):.4f} | {m('final'):.4f} | {m('f2'):.3f} | "
                     f"{m('brier'):.4f} | {m('server_s'):.2f} | {m('evals'):.1f} | {m('finetunes'):.1f} | "
                     + " ".join(f"{k} {v:.0%}" for k, v in mix.items()) + " |")


def verdict(ok):
    return "**supported**" if ok else "**not supported**"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1c-rate", action="store_true")
    a = ap.parse_args()
    main_runs = load("main")
    if a.p1c_rate:
        s = common(main_runs, ["select"])
        print(f"{np.mean(vec(main_runs, 'select', 'yogi_rate', s)):.3f}")
        return

    L = ["# ECG results", "", "Protocol: experiments/ecg_protocol.md. Test seeds 42-46, 60 rounds. AUC = mean test AUPRC "
         "over all rounds; late = second half; final = last 5 rounds. Differences: mean ± 95% CI (wins/seeds, paired "
         "t p).", ""]

    # ---------------- Paper 1 ----------------
    L += ["## Paper 1: measure after the updates arrive", ""]
    table(main_runs, ["fedavg", "fedyogi", "select", "c_reject", "c_half", "c_normmatch", "random_mix"], L)
    s = common(main_runs, ["fedavg", "fedyogi", "select"])
    if s:
        best_static = np.maximum(vec(main_runs, "fedavg", "auc", s), vec(main_runs, "fedyogi", "auc", s))
        gain = vec(main_runs, "select", "auc", s) - best_static
        L += ["", f"- H1 select - per-seed best static (AUC): {paired(gain)} -> {verdict((gain > 0).sum() >= 4)}"]
        for k in ("late", "final"):
            bs = np.maximum(vec(main_runs, "fedavg", k, s), vec(main_runs, "fedyogi", k, s))
            L.append(f"  - {k}: {paired(vec(main_runs, 'select', k, s) - bs)}")
        for c in ("c_reject", "c_half", "c_normmatch"):
            sc = common(main_runs, ["fedavg", "fedyogi", "select", c])
            if not sc:
                continue
            bs = np.maximum(vec(main_runs, "fedavg", "auc", sc), vec(main_runs, "fedyogi", "auc", sc))
            g_sel = vec(main_runs, "select", "auc", sc) - bs
            g_c = vec(main_runs, c, "auc", sc) - bs
            match = (g_c >= 0.8 * g_sel) & (g_sel > 0)
            L.append(f"- P1b {c}: gain over best static {paired(g_c)}; matches the selector (>= 80% of its gain) on "
                     f"{int(match.sum())}/{len(sc)} seeds -> {'matches' if match.sum() > len(sc) / 2 else 'does not match'}")
        sr = common(main_runs, ["select", "random_mix"])
        if sr:
            d = vec(main_runs, "select", "auc", sr) - vec(main_runs, "random_mix", "auc", sr)
            L.append(f"- H1c select - random_mix (AUC): {paired(d)} -> {verdict((d > 0).sum() >= 4)}")
        L.append(f"- Selector kept FedYogi in {np.mean(vec(main_runs, 'select', 'yogi_rate', s)):.1%} of rounds.")

    # ---------------- Paper 2 ----------------
    L += ["", "## Paper 2: select and train", ""]
    for label, d in (("P2a full matched V (2,198)", "main"), ("P2b |V| = 280, matched", "v280"),
                     ("P2c |V| = 280 at 50% AFib", "v280_afib50")):
        runs = load(d)
        for st in ("fedavg", "fedyogi"):          # static rules do not use V: shared with P1a
            if st in main_runs:
                runs[st] = main_runs[st]
        L += [f"### {label}", ""]
        table(runs, ["fedavg", "fedyogi", "select", "train", "select_train", "train_select"], L)
        sc = common(runs, ["select", "train", "select_train"])
        if sc:
            better = np.maximum(vec(runs, "select", "auc", sc), vec(runs, "train", "auc", sc))
            dd = vec(runs, "select_train", "auc", sc) - better
            L += ["", f"- H2 select_train - better of (select, train): {paired(dd)} -> "
                      f"{verdict((dd >= 0).sum() >= 4)}"]
            for other in ("select", "train", "train_select"):
                so = common(runs, ["select_train", other])
                if so:
                    L.append(f"  - select_train - {other}: {paired(vec(runs, 'select_train', 'auc', so) - vec(runs, other, 'auc', so))}")
            ss = common(runs, ["select_train", "fedavg", "fedyogi"])
            if ss:
                bs = np.maximum(vec(runs, "fedavg", "auc", ss), vec(runs, "fedyogi", "auc", ss))
                dd = vec(runs, "select_train", "auc", ss) - bs
                tag = " (H2r)" if d == "v280_afib50" else ""
                L.append(f"- select_train - per-seed best static{tag}: {paired(dd)}"
                         + (f" -> {verdict((dd >= 0).sum() >= 4)}" if tag else ""))
        L.append("")

    # ---------------- Paper 3 ----------------
    L += ["## Paper 3: measurement vs FedStrategist under poisoning", ""]
    for label, d in (("P3a standard attack (5x honest update)", "poison_standard"),
                     ("P3b standard attack, flipped direction", "poison_flip")):
        runs = load(d)
        L += [f"### {label}", ""]
        names = ["fedavg", "median", "krum", "fs_paper", "fs_authors", "fs_paper_loss", "sara", "select_robust"]
        table(runs, names, L)
        fs = [n for n in ("fs_paper", "fs_authors", "fs_paper_loss") if n in runs]
        sc = common(runs, ["select_robust"] + fs)
        if fs and sc:
            best_fs = np.max([vec(runs, n, "auc", sc) for n in fs], axis=0)
            dd = vec(runs, "select_robust", "auc", sc) - best_fs
            L += ["", f"- H3 select_robust - per-seed best FedStrategist variant (AUC): {paired(dd)} -> "
                      f"{verdict((dd >= 0).sum() >= 4)}"]
            for n in fs:
                L.append(f"  - select_robust - {n}: {paired(vec(runs, 'select_robust', 'auc', sc) - vec(runs, n, 'auc', sc))}")
        ss = common(runs, ["select_robust", "sara"])
        if ss:
            dd = vec(runs, "select_robust", "auc", ss) - vec(runs, "sara", "auc", ss)
            L.append(f"- H3s select_robust - sara (AUC): {paired(dd)} -> {verdict((dd >= 0).sum() >= 4)}")
        learned = fs + (["sara"] if "sara" in runs else [])
        sl = common(runs, ["select_robust"] + learned)
        if "sara" in learned and sl:
            best = np.max([vec(runs, n, "auc", sl) for n in learned], axis=0)
            dd = vec(runs, "select_robust", "auc", sl) - best
            L.append(f"- select_robust - per-seed best learned controller (FedStrategist x3, SARA): {paired(dd)} -> "
                     f"{verdict((dd >= 0).sum() >= 4)}")
        st = [n for n in ("fedavg", "median", "krum") if n in runs]
        ss = common(runs, ["select_robust"] + st)
        if st and ss:
            bs = np.max([vec(runs, n, "auc", ss) for n in st], axis=0)
            L.append(f"- select_robust - per-seed best static rule: {paired(vec(runs, 'select_robust', 'auc', ss) - bs)}")
        L.append("")

    text = "\n".join(L)
    with open(os.path.join(ROOT, "REPORT.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
