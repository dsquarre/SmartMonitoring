#!/usr/bin/env python3
"""
nonfl_baselines.py - Non-federated reference baselines for the H1 comparison.

  --mode centralized : all clients' training data pooled in one place, one model (upper bound).
  --mode local       : every client trains its own model on its own data only, no communication (lower bound).

Uses the same model (server/model.py), Round-0 global output bias and global class weights, data partitions,
and evaluation as simulate_fl.py: threshold 0.3, confusion counts micro-aggregated over all clients' test sets,
loss = mean of per-client losses, AUPRC over all pooled test predictions. Writes metrics.csv with the same
columns as simulate_fl.py ("round" = training epoch), so compare_experiments.py can read it directly.

Training budget matches one FL run: in FL each round processes ~one full pass over the data
(K=20 of N=100 clients x E=5 local epochs), so 120 rounds ~= 120 epochs.
"""

import os
import sys
import csv
import json
import random
import argparse
import functools
import gc
import numpy as np

print = functools.partial(print, flush=True)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
for d in (os.path.join(PROJECT_ROOT, "server"), os.path.join(PROJECT_ROOT, "client")):
    if d not in sys.path:
        sys.path.insert(0, d)

import server.model as server_model_mod
sys.modules['model'] = server_model_mod
from client.model import Model as ClientModel
from server.model import Model as ServerModel


def load_partitions(data_dir):
    cache = {}
    for fn in sorted(os.listdir(data_dir)):
        if fn.endswith(".npz"):
            cid = fn[:-4]
            with np.load(os.path.join(data_dir, fn)) as d:
                cache[cid] = {
                    "X_train": ClientModel._to_array(d["X_train"]),
                    "y_train": np.asarray(d["y_train"]),
                    "X_test": ClientModel._to_array(d["X_test"]),
                    "y_test": np.asarray(d["y_test"]),
                }
    return cache


class ShuffledBatches(tf.keras.utils.Sequence):
    """Feeds in-memory numpy data to model.fit one batch at a time, reshuffled every epoch.
    Passing the whole pooled array (~840 MB) to fit() makes TF copy it to the GPU as one tensor,
    which fails when the GPU is shared ("Dst tensor is not initialized")."""

    def __init__(self, X, y, batch_size, seed):
        self.X, self.y, self.batch_size = X, y, batch_size
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, i):
        idx = np.sort(self.order[i * self.batch_size:(i + 1) * self.batch_size])
        return self.X[idx], self.y[idx]

    def on_epoch_end(self):
        self.order = self.rng.permutation(len(self.X))


def reset_optimizer(model):
    for v in model.optimizer.variables():
        v.assign(np.zeros(v.shape, dtype=v.dtype.as_numpy_dtype))


def evaluate_clients(models_for, cache, class_weights, batch_size):
    """models_for(cid) -> keras model to evaluate on that client's test set. Mirrors simulate_fl.py aggregation."""
    evals, ys, ss = [], [], []
    for cid, part in cache.items():
        cm = ClientModel(cid, batch_size=batch_size, model=models_for(cid), class_weights=class_weights, data=part)
        res, y_t, y_s = cm.evaluate(threshold=0.3, return_preds=True)
        if len(y_t) > 0:
            ys.append(y_t)
            ss.append(y_s)
        evals.append(res)
    tp = sum(int(e["tp"]) for e in evals); fp = sum(int(e["fp"]) for e in evals)
    tn = sum(int(e["tn"]) for e in evals); fn = sum(int(e["fn"]) for e in evals)
    prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    f2 = (5 * prec * rec / (4 * prec + rec)) if (4 * prec + rec) > 0 else 0.0
    s_tp = sum(e["soft_tp"] for e in evals); s_fp = sum(e["soft_fp"] for e in evals)
    s_fn = sum(e["soft_fn"] for e in evals); s_tn = sum(e["soft_tn"] for e in evals)
    s_prec = s_tp / (s_tp + s_fp + 1e-7); s_rec = s_tp / (s_tp + s_fn + 1e-7)
    s_f2 = (5 * s_prec * s_rec / (4 * s_prec + s_rec)) if (4 * s_prec + s_rec) > 0 else 0.0
    y_all, s_all = np.concatenate(ys), np.concatenate(ss)
    auprc = 0.5
    if len(np.unique(y_all)) > 1 and np.all(np.isfinite(s_all)):
        from sklearn.metrics import precision_recall_curve, auc
        p_c, r_c, _ = precision_recall_curve(y_all, s_all)
        auprc = float(auc(r_c, p_c))
    return {
        "loss": float(np.mean([e["loss"] for e in evals])),
        "accuracy": float(f2), "f2": float(f2), "f1": float(f1),
        "precision": float(prec), "recall": float(rec),
        "raw_accuracy": float((tp + tn) / max(1, tp + tn + fp + fn)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "auprc": auprc,
        "soft_tp": float(s_tp), "soft_fp": float(s_fp), "soft_tn": float(s_tn), "soft_fn": float(s_fn),
        "soft_precision": float(s_prec), "soft_recall": float(s_rec), "soft_f2": float(s_f2),
    }


def write_outputs(history, out_dir):
    with open(os.path.join(out_dir, "round_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def main():
    ap = argparse.ArgumentParser(description="Centralized / local-only reference baselines")
    ap.add_argument("--mode", choices=["centralized", "local"], required=True)
    ap.add_argument("--data-dir", default="datasets/iid")
    ap.add_argument("--epochs", type=int, default=120, help="Training epochs (matches 120 FL rounds of K=20, E=5)")
    ap.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs (local mode default 10)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--output-dir", required=True)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print(f" Non-FL baseline | mode={args.mode} | epochs={args.epochs} | seed={args.seed} | data={args.data_dir}")
    print("=" * 60)

    cache = load_partitions(args.data_dir)
    y_tr_all = np.concatenate([np.asarray(p["y_train"], dtype=np.int32) for p in cache.values()])
    n_all, n_pos = len(y_tr_all), int((y_tr_all == 1).sum())
    p_global = float(np.clip(n_pos / max(1, n_all), 1e-5, 1 - 1e-5))
    global_bias = float(np.log(p_global / (1 - p_global)))
    class_weights = {0: n_all / (2.0 * max(1, n_all - n_pos)), 1: n_all / (2.0 * max(1, n_pos))}
    print(f"[Setup] {len(cache)} clients | {n_all} train samples ({n_pos} pos) | bias={global_bias:.4f} | "
          f"class weights {{0: {class_weights[0]:.4f}, 1: {class_weights[1]:.4f}}}")

    model = ServerModel(initial_bias=global_bias).model
    history = []

    if args.mode == "centralized":
        X = np.concatenate([p["X_train"] for p in cache.values()])
        y = np.concatenate([np.asarray(p["y_train"]) for p in cache.values()]).astype(np.float32)
        batches = ShuffledBatches(X, y, args.batch_size, seed=args.seed)
        for ep in range(1, args.epochs + 1):
            h = model.fit(batches, epochs=1, class_weight=class_weights, verbose=0)
            batches.on_epoch_end()
            if ep % args.eval_every == 0 or ep == args.epochs:
                m = evaluate_clients(lambda cid: model, cache, class_weights, args.batch_size)
                row = {"seed": args.seed, "round": ep, **m, "avg_comp_latency": 0.0, "total_round_energy": 0.0,
                       "chosen_strategy": "centralized", "train_loss": float(h.history["loss"][-1])}
                history.append(row)
                write_outputs(history, args.output_dir)
                print(f" Epoch {ep}/{args.epochs} | Loss: {m['loss']:.4f} | Hard-F2: {m['f2']:.4f} | AUPRC: {m['auprc']:.4f} | "
                      f"Rec: {m['recall']:.4f} | Prec: {m['precision']:.4f}")
    else:
        # Each client: fresh copy of the same initial model and a fresh optimizer, trained only on its own data.
        # Weights are kept per client in memory and evaluated on that client's own test set.
        init_w = model.get_weights()
        client_w = {cid: init_w for cid in cache}
        eval_every = args.eval_every if args.eval_every > 1 else 10
        checkpoints = sorted(set(list(range(eval_every, args.epochs + 1, eval_every)) + [args.epochs]))
        done = 0
        for ck in checkpoints:
            n_ep = ck - done
            for cid, part in cache.items():
                model.set_weights(client_w[cid])
                reset_optimizer(model)
                model.fit(part["X_train"], np.asarray(part["y_train"]).astype(np.float32), batch_size=args.batch_size,
                          epochs=n_ep, shuffle=True, class_weight=class_weights, verbose=0)
                client_w[cid] = model.get_weights()
            done = ck

            eval_model = ServerModel(initial_bias=global_bias).model
            def model_for(cid):
                eval_model.set_weights(client_w[cid])
                return eval_model
            m = evaluate_clients(model_for, cache, class_weights, args.batch_size)
            history.append({"seed": args.seed, "round": ck, **m, "avg_comp_latency": 0.0, "total_round_energy": 0.0,
                            "chosen_strategy": "local_only"})
            write_outputs(history, args.output_dir)
            print(f" Epoch {ck}/{args.epochs} | Loss: {m['loss']:.4f} | Hard-F2: {m['f2']:.4f} | AUPRC: {m['auprc']:.4f} | "
                  f"Rec: {m['recall']:.4f} | Prec: {m['precision']:.4f}")
            del eval_model
            gc.collect()

    print("\n" + "=" * 60)
    print(" Research Simulation Completed Successfully! ")
    print(f" Logs saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
