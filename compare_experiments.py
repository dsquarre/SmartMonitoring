#!/usr/bin/env python3
"""
compare_experiments.py - Comparative Analysis & Paired Hypothesis Testing Script

Loads raw experiment metrics (metrics.csv or round_history.json) from single or multi-seed
run directories, overlays performance curves with error bands, and executes paired t-test
hypothesis analysis across paired seeds.
"""

import os
import sys
import argparse
import glob
import json
import csv
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


def load_run_metrics(run_dir):
    """Loads round metrics from metrics.csv or round_history.json in run_dir."""
    csv_path = os.path.join(run_dir, "metrics.csv")
    json_path = os.path.join(run_dir, "round_history.json")

    metrics = []
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed_row = {}
                for k, v in row.items():
                    try:
                        parsed_row[k] = float(v) if "." in v or "e" in v.lower() else int(v)
                    except ValueError:
                        parsed_row[k] = v
                metrics.append(parsed_row)
    elif os.path.exists(json_path):
        with open(json_path, "r") as f:
            metrics = json.load(f)

    return metrics


def parse_groups(dirs, labels):
    """
    Groups run directories by label.
    Accepts explicit directory paths or glob patterns.
    """
    grouped_runs = {}

    if labels and len(labels) == len(dirs):
        for label, dir_pattern in zip(labels, dirs):
            matched = glob.glob(dir_pattern) or [dir_pattern]
            matched_dirs = [d for d in matched if os.path.isdir(d)]
            grouped_runs[label] = matched_dirs
    else:
        for dir_pattern in dirs:
            matched = glob.glob(dir_pattern) or [dir_pattern]
            matched_dirs = [d for d in matched if os.path.isdir(d)]
            for d in matched_dirs:
                basename = os.path.basename(os.path.normpath(d))
                label = basename.split("_seed")[0].split("_s4")[0].replace("_", " ").title()
                grouped_runs.setdefault(label, []).append(d)

    return grouped_runs


def aggregate_group_metrics(run_dirs):
    """
    Aggregates per-round metrics across multiple seed runs for a single strategy group.
    Returns per-round statistics (mean, std, stderr) and per-seed metric series.
    """
    runs_data = []
    for d in run_dirs:
        m = load_run_metrics(d)
        if m:
            runs_data.append(m)

    if not runs_data:
        return None

    num_rounds = max(len(m) for m in runs_data)
    keys = ["loss", "f2", "accuracy", "f1", "precision", "recall", "auprc", "avg_comp_latency", "total_round_energy"]

    round_stats = {r: {k: [] for k in keys} for r in range(1, num_rounds + 1)}
    seed_map = {}

    for run_metrics in runs_data:
        seed = run_metrics[0].get("seed", 0) if run_metrics else 0
        seed_map[seed] = run_metrics
        for row in run_metrics:
            r = int(row.get("round", 0))
            if r in round_stats:
                for k in keys:
                    val = row.get(k, 0.0)
                    if val is not None:
                        round_stats[r][k].append(float(val))

    agg_by_round = []
    for r in range(1, num_rounds + 1):
        row_agg = {"round": r}
        for k in keys:
            vals = round_stats[r][k]
            if vals:
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals)) if len(vals) > 1 else 0.0
                stderr_v = float(std_v / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            else:
                mean_v, std_v, stderr_v = 0.0, 0.0, 0.0

            row_agg[f"{k}_mean"] = mean_v
            row_agg[f"{k}_std"] = std_v
            row_agg[f"{k}_stderr"] = stderr_v
        agg_by_round.append(row_agg)

    return {
        "agg_by_round": agg_by_round,
        "seed_map": seed_map,
        "raw_runs": runs_data
    }


def perform_paired_hypothesis_testing(group_A_name, group_A_data, group_B_name, group_B_data):
    """
    Executes paired t-test (ttest_rel) and Wilcoxon signed-rank test across paired seeds.
    """
    seed_map_A = group_A_data["seed_map"]
    seed_map_B = group_B_data["seed_map"]

    common_seeds = sorted(list(set(seed_map_A.keys()) & set(seed_map_B.keys())))

    # Fallback to index pairing if seeds are unspecified/unmatched
    if not common_seeds:
        min_len = min(len(group_A_data["raw_runs"]), len(group_B_data["raw_runs"]))
        paired_A_runs = group_A_data["raw_runs"][:min_len]
        paired_B_runs = group_B_data["raw_runs"][:min_len]
    else:
        paired_A_runs = [seed_map_A[s] for s in common_seeds]
        paired_B_runs = [seed_map_B[s] for s in common_seeds]

    if not paired_A_runs or not paired_B_runs:
        return []

    metrics_to_test = [
        ("Final Loss (L_T)", lambda run: run[-1].get("loss", np.nan)),
        ("Final F2 Score", lambda run: run[-1].get("f2", run[-1].get("accuracy", np.nan))),
        ("Final F1 Score", lambda run: run[-1].get("f1", np.nan)),
        ("Mean AUPRC", lambda run: float(np.mean([r.get("auprc", 0.0) for r in run if r.get("auprc") is not None]))),
        ("Total Latency", lambda run: float(np.sum([r.get("avg_comp_latency", 0.0) for r in run]))),
        ("Total Energy", lambda run: float(np.sum([r.get("total_round_energy", 0.0) for r in run])))
    ]

    results = []
    for metric_name, extractor in metrics_to_test:
        vals_A = np.array([extractor(run) for run in paired_A_runs], dtype=np.float64)
        vals_B = np.array([extractor(run) for run in paired_B_runs], dtype=np.float64)

        # Remove nan pairs
        valid_mask = ~np.isnan(vals_A) & ~np.isnan(vals_B)
        vA, vB = vals_A[valid_mask], vals_B[valid_mask]

        if len(vA) < 2 or np.all(vA == vB):
            t_stat, p_val = 0.0, 1.0
            w_stat, w_pval = 0.0, 1.0
        else:
            t_res = stats.ttest_rel(vA, vB)
            t_stat, p_val = float(t_res.statistic), float(t_res.pvalue)

            try:
                w_res = stats.wilcoxon(vA, vB)
                w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
            except Exception:
                w_stat, w_pval = np.nan, np.nan

        mean_A, mean_B = float(np.mean(vA)), float(np.mean(vB))
        mean_diff = mean_A - mean_B
        sig_flag = bool(p_val < 0.05)

        results.append({
            "metric": metric_name,
            "strategy_A": group_A_name,
            "strategy_B": group_B_name,
            "n_pairs": len(vA),
            "mean_A": mean_A,
            "mean_B": mean_B,
            "mean_diff": mean_diff,
            "t_statistic": t_stat,
            "p_value": p_val,
            "significant_p05": sig_flag,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p_value": w_pval
        })

    return results


def plot_comparative_overlay(group_data, output_dir):
    """Generates comparative overlay plots with shaded error bands across strategies."""
    colors = ['crimson', 'royalblue', 'forestgreen', 'darkorange', 'purple', 'teal']

    # 1. Overlay Loss vs Round
    plt.figure(figsize=(9, 5.5))
    for i, (label, data) in enumerate(group_data.items()):
        agg = data["agg_by_round"]
        rounds = [row["round"] for row in agg]
        means = [row["loss_mean"] for row in agg]
        stderrs = [row["loss_stderr"] for row in agg]
        color = colors[i % len(colors)]

        plt.plot(rounds, means, marker='o', color=color, linewidth=2, label=f"{label} (Mean)")
        if any(s > 0 for s in stderrs):
            plt.fill_between(rounds, np.array(means) - np.array(stderrs), np.array(means) + np.array(stderrs), color=color, alpha=0.2)

    plt.xlabel("Federated Round")
    plt.ylabel("Global Loss")
    plt.title("Global Test Loss vs Federated Round (Comparative Overlay)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_loss_overlay.png"), dpi=300)
    plt.close()

    # 2. Overlay F2 Score (Accuracy) vs Round
    plt.figure(figsize=(9, 5.5))
    for i, (label, data) in enumerate(group_data.items()):
        agg = data["agg_by_round"]
        rounds = [row["round"] for row in agg]
        means = [row["f2_mean"] for row in agg]
        stderrs = [row["f2_stderr"] for row in agg]
        color = colors[i % len(colors)]

        plt.plot(rounds, means, marker='s', color=color, linewidth=2, label=f"{label} (F2 Score)")
        if any(s > 0 for s in stderrs):
            plt.fill_between(rounds, np.array(means) - np.array(stderrs), np.array(means) + np.array(stderrs), color=color, alpha=0.2)

    plt.xlabel("Federated Round")
    plt.ylabel("F2 Score (Primary Accuracy)")
    plt.title("Global F2 Score Accuracy vs Federated Round (Comparative Overlay)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_f2_accuracy_overlay.png"), dpi=300)
    plt.close()

    # 3. Overlay AUPRC vs Round
    plt.figure(figsize=(9, 5.5))
    for i, (label, data) in enumerate(group_data.items()):
        agg = data["agg_by_round"]
        rounds = [row["round"] for row in agg]
        means = [row["auprc_mean"] for row in agg]
        color = colors[i % len(colors)]

        plt.plot(rounds, means, marker='D', color=color, linewidth=2, label=f"{label} (AUPRC)")

    plt.xlabel("Federated Round")
    plt.ylabel("AUPRC Score")
    plt.title("Global AUPRC vs Federated Round (Comparative Overlay)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_auprc_overlay.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Comparative Metrics & Paired Hypothesis Analysis")
    parser.add_argument("--dirs", nargs="+", required=True, help="Run directories or glob patterns")
    parser.add_argument("--labels", nargs="+", help="Strategy labels corresponding to dirs")
    parser.add_argument("-o", "--output-dir", type=str, default="results/comparison", help="Output directory")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    grouped_runs = parse_groups(args.dirs, args.labels)
    if not grouped_runs:
        print("[Error] No valid run directories matched.")
        sys.exit(1)

    print(f"[Compare] Found {len(grouped_runs)} strategy groups for analysis:")
    for label, dirs in grouped_runs.items():
        print(f"  - Group '{label}': {len(dirs)} run directories")

    group_data = {}
    for label, dirs in grouped_runs.items():
        data = aggregate_group_metrics(dirs)
        if data:
            group_data[label] = data

    if len(group_data) < 1:
        print("[Error] Failed to load metrics from directories.")
        sys.exit(1)

    # Plot overlay curves
    plot_comparative_overlay(group_data, args.output_dir)

    # Paired Hypothesis Analysis if at least 2 groups exist
    labels = list(group_data.keys())
    hypothesis_results = []
    if len(labels) >= 2:
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                res = perform_paired_hypothesis_testing(labels[i], group_data[labels[i]], labels[j], group_data[labels[j]])
                hypothesis_results.extend(res)

    # Save hypothesis testing results to CSV
    hyp_csv_path = os.path.join(args.output_dir, "hypothesis_test_results.csv")
    if hypothesis_results:
        fieldnames = list(hypothesis_results[0].keys())
        with open(hyp_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(hypothesis_results)
        print(f"\n Saved Paired Hypothesis Test Results to: {hyp_csv_path}")

    # Save summary stats to CSV
    summary_rows = []
    for label, data in group_data.items():
        agg = data["agg_by_round"]
        last_row = agg[-1] if agg else {}
        summary_rows.append({
            "strategy": label,
            "num_runs": len(data["raw_runs"]),
            "final_loss_mean": last_row.get("loss_mean", np.nan),
            "final_loss_std": last_row.get("loss_std", np.nan),
            "final_f2_mean": last_row.get("f2_mean", np.nan),
            "final_f2_std": last_row.get("f2_std", np.nan),
            "final_auprc_mean": last_row.get("auprc_mean", np.nan),
            "total_latency_mean": sum(r.get("avg_comp_latency_mean", 0) for r in agg),
            "total_energy_mean": sum(r.get("total_round_energy_mean", 0) for r in agg)
        })

    summary_csv_path = os.path.join(args.output_dir, "comparison_summary.csv")
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f" Saved Comparison Summary CSV to: {summary_csv_path}")

    print("\n" + "=" * 60)
    print(" Comparative Analysis & Hypothesis Testing Complete! ")
    print(f" Results saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
