#!/usr/bin/env python3
"""
Calibrates optimal federated rounds T = 1.5 * T0 for experimental runs.
T0 is defined as the round where full-participation FedAvg (N=100, K=100) curve plateaus for 10+ rounds.
"""

import os
import sys
import argparse
import csv
import numpy as np


def find_plateau_round(metrics_path: str, plateau_window: int = 10, loss_threshold: float = 0.005, acc_threshold: float = 0.005) -> tuple:
    """
    Finds T0: the first round where the loss or accuracy curve plateaus for plateau_window (10+) consecutive rounds.
    Returns: (t0, t_optimal, max_accuracy, final_loss, total_rounds)
    """
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics CSV file not found: {metrics_path}")

    rounds = []
    losses = []
    accuracies = []

    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(row["round"]))
            losses.append(float(row["loss"]))
            acc_val = float(row.get("f2", row.get("accuracy", 0.0)))
            accuracies.append(acc_val)

    total_rounds = len(rounds)
    max_accuracy = max(accuracies) if accuracies else 0.0
    final_loss = losses[-1] if losses else 0.0

    if total_rounds < plateau_window:
        print(f"[Warning] Run length ({total_rounds} rounds) is less than plateau window ({plateau_window} rounds).")
        return total_rounds, int(np.ceil(1.5 * total_rounds)), max_accuracy, final_loss, total_rounds

    t0 = total_rounds  # Default to max rounds if no plateau found

    for i in range(len(losses) - plateau_window + 1):
        window_losses = losses[i:i + plateau_window]
        window_accs = accuracies[i:i + plateau_window]

        loss_std = np.std(window_losses)
        acc_range = max(window_accs) - min(window_accs)

        if loss_std < loss_threshold or acc_range < acc_threshold:
            t0 = rounds[i]
            break

    t_optimal = int(np.ceil(1.5 * t0))
    return t0, t_optimal, max_accuracy, final_loss, total_rounds


def main():
    parser = argparse.ArgumentParser(description="Calibrate optimal rounds T = 1.5 * T0 from FedAvg baseline run")
    parser.add_argument("--metrics", type=str, default="results/to_calibration/metrics.csv", help="Path to full participation metrics.csv")
    parser.add_argument("--window", type=int, default=10, help="Plateau window size in rounds (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.005, help="Plateau threshold for loss std / acc range (default: 0.005)")

    args = parser.parse_args()

    t0, t_optimal, max_acc, final_loss, total_rounds = find_plateau_round(
        args.metrics, plateau_window=args.window, loss_threshold=args.threshold, acc_threshold=args.threshold
    )

    print("=" * 65)
    print(" SmartMonitoring Rounds Calibration (T = 1.5 * T0) ")
    print("=" * 65)
    print(f" Metrics Input File     : {args.metrics}")
    print(f" Total Baseline Rounds  : {total_rounds}")
    print(f" Max F2 Accuracy        : {max_acc:.4f}")
    print(f" Final Round Loss       : {final_loss:.4f}")
    print(f" Detected Plateau T0   : Round {t0}")
    print(f" Calibrated Rounds (T) : {t_optimal} (1.5 * T0)")
    print("=" * 65)
    print(f"\nRecommended command parameter for 30-seed hypothesis runs:")
    print(f"  simulate_fl.py --data-dir data/non_iid -n 100 -r {t_optimal} ...\n")


if __name__ == "__main__":
    main()

