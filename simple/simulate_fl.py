#!/usr/bin/env python3
"""
simulate_fl.py - SmartMonitoring single-process FL simulation (ECG atrial fibrillation, PTB-XL clients)

One run = one server strategy x one seed. The methods are the plugins of the distributed system
(distributed/server/plugins: client selectors, server rules and controllers, attacks), so a result here is a result
of the code the deployed system runs. Hypotheses and their commands: EXPERIMENTS.md.

    python simulate_fl.py -a select_train --seed 42 -o results/test       # one seed, written to results/test
    python simulate_fl.py -a fedavg -o results/h1_fedavg                   # base seeds 42-71 -> results/h1_fedavg/s42 .. s71
    python simulate_fl.py -a select --seeds 42-56 -o results/h1_select     # half the seeds (57-71 in a second window)

Round t:  S_t = selector picks K clients -> E epochs of local SGD on each (malicious clients then attack) ->
          w_{t+1} = server strategy(w_t, updates) -> metrics of w_{t+1} on the test set T (never used for a decision).
Per run folder: metrics.csv and round_history.json (every round), config.json, plots (every 10 rounds and at the
end), checkpoint.pkl while running (an interrupted run resumes from it), summary.json when finished (a finished run
is skipped unless --overwrite).
"""
import argparse
import csv
import functools
import glob
import importlib.util
import json
import math
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass

# Quiet TensorFlow; let several processes share one GPU.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

print = functools.partial(print, flush=True)

# -- the distributed system's plugins, loaded as the package `plugins` --------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PLUGINS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "distributed", "server", "plugins"))
if "plugins" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("plugins", os.path.join(PLUGINS_DIR, "__init__.py"),
                                                   submodule_search_locations=[PLUGINS_DIR])
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["plugins"] = _module
    _spec.loader.exec_module(_module)
from plugins import (CONTROLLERS, RULES, SELECTORS, ClientUpdate, RoundContext,  # noqa: E402
                     build_controller, craft_malicious_weights, get_selector_by_name)
from plugins.attacks import ATTACKS, update_norm  # noqa: E402
from plugins.oort import OortConfig  # noqa: E402

BASE_SEEDS = "42-71"     # 30 test seeds (experiments/ecg_protocol.md)
THRESHOLD = 0.3          # clinical operating point for thresholded metrics (recall-weighted F2)
INPUT_SHAPE = (1000, 12)


# =====================================================================================================================
# Data: client partitions, the server's validation set V and the test set T
# =====================================================================================================================
@dataclass
class FederatedData:
    client_x: list
    client_y: list
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray

    @property
    def client_sizes(self):
        return np.array([len(y) for y in self.client_y])


def to_array(X):
    """float32 (N, 1000, 12) with NaN/Inf zeroed; accepts (N, 12, 1000) too."""
    X = np.asarray(X)
    if X.ndim == 3 and X.shape[1] == 12 and X.shape[2] != 12:
        X = np.transpose(X, (0, 2, 1))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_records(X):
    """Per-record, per-lead z-score (chosen by calibration)."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return ((X - mu) / np.maximum(sd, 1e-6)).astype(np.float32)


def client_files(data_dir, num_clients=0):
    files = glob.glob(os.path.join(data_dir, "client_*.npz"))
    files.sort(key=lambda f: int(re.search(r"client_(\d+)\.npz$", f).group(1)))
    if not files:
        raise FileNotFoundError(f"no client_*.npz files in {data_dir}")
    return files[:num_clients] if num_clients else files


def build_federated_data(data_dir, seed, val_fraction=0.5, val_size=0, num_clients=0, val_pos_share=0.0,
                         normalize="record"):
    """Clients keep their train split. The clients' test splits are pooled and split once per seed into V (the
    server's validation set: selection and server training) and T (reported, never used for a decision).
    val_size subsamples V uniformly; with val_pos_share it is drawn with that AFib share (a label-skewed server set)."""
    prep = normalize_records if normalize == "record" else (lambda a: a)
    client_x, client_y, pool_x, pool_y = [], [], [], []
    for f in client_files(data_dir, num_clients):
        with np.load(f) as z:
            client_x.append(prep(to_array(z["X_train"])))
            client_y.append(np.asarray(z["y_train"], dtype=np.float32))
            pool_x.append(prep(to_array(z["X_test"])))
            pool_y.append(np.asarray(z["y_test"], dtype=np.float32))
    pool_x, pool_y = np.concatenate(pool_x), np.concatenate(pool_y)

    perm = np.random.default_rng(np.random.SeedSequence([seed, 0xDA7A])).permutation(len(pool_y))
    n_val = int(len(perm) * val_fraction)
    val_idx, test_idx = perm[:n_val], perm[n_val:]
    if val_size and val_pos_share > 0:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5CE, val_size, int(val_pos_share * 1000)]))
        pos, neg = val_idx[pool_y[val_idx] >= 0.5], val_idx[pool_y[val_idx] < 0.5]
        n_pos = int(round(val_size * val_pos_share))
        if n_pos > len(pos) or val_size - n_pos > len(neg):
            raise ValueError(f"V has {len(pos)} AFib / {len(neg)} normal records; cannot draw {val_size} at AFib share "
                             f"{val_pos_share}")
        val_idx = np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False),
                                          rng.choice(neg, val_size - n_pos, replace=False)]))
    elif val_size and val_size < len(val_idx):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x7A1, val_size]))
        val_idx = np.sort(rng.choice(val_idx, size=val_size, replace=False))
    return FederatedData(client_x, client_y, pool_x[val_idx], pool_y[val_idx], pool_x[test_idx], pool_y[test_idx])


def split_validation(n, select_frac, seed):
    """Index sets (scoring part, training part) of V; the same split as distributed/server/aggregator.py."""
    if select_frac <= 0:
        idx = np.arange(n)
        return idx, idx
    perm = np.random.default_rng(np.random.SeedSequence([seed, 0x5E1])).permutation(n)
    n_sel = max(1, int(round(n * select_frac)))
    return np.sort(perm[:n_sel]), np.sort(perm[n_sel:])


# =====================================================================================================================
# Model, training and metrics
# =====================================================================================================================
def build_model(initial_bias=-2.2):
    """1-D CNN on 12-lead, 1000-sample records."""
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="ecg_input")
    x = tf.keras.layers.Conv1D(64, kernel_size=15, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=7, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(256, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="afib",
                                   bias_initializer=tf.keras.initializers.Constant(initial_bias))(x)
    return tf.keras.Model(inputs=inputs, outputs=output)


def trainable_mask(model):
    """Per entry of model.get_weights(): True for trainable weights, False for BatchNorm moving statistics."""
    trainable = {id(v) for v in model.trainable_weights}
    return [id(v) in trainable for v in model.weights]


def bce(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def binary_metrics(y, p, threshold=THRESHOLD):
    """AUPRC, ROC-AUC, Brier, BCE and the thresholded metrics (None for AUPRC/ROC-AUC with one class only)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= threshold).astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    both = len(np.unique(y)) > 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "auprc": float(average_precision_score(y, p)) if both else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else 0.0,
        "loss": bce(y, p) if len(y) else 0.0,
        "precision": prec, "recall": rec,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "f2": 5 * prec * rec / (4 * prec + rec) if 4 * prec + rec else 0.0,
        "accuracy": (tp + tn) / len(y) if len(y) else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


@dataclass
class TrainResult:
    weights: list
    num_samples: int
    train_loss: float        # mean BCE over the training batches
    stat_utility: float      # Oort: |B_i| * sqrt(mean per-sample training loss^2)


class Trainer:
    """Local training as in the TMLR supplement: plain SGD, no momentum (a fresh optimizer state every call), unweighted
    binary cross-entropy, a compiled step. The server's training on V is a second Trainer at the server's step."""

    def __init__(self, model, lr, predict_batch=512):
        self.model = model
        self.predict_batch = predict_batch
        self.opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0)
        x_spec = tf.TensorSpec((None,) + INPUT_SHAPE, tf.float32)
        v_spec = tf.TensorSpec((None,), tf.float32)
        self._step = tf.function(self._train_step, input_signature=[x_spec, v_spec])
        self._forward = tf.function(lambda x: self.model(x, training=False), input_signature=[x_spec])

    def _train_step(self, x, y):
        with tf.GradientTape() as tape:
            p = tf.squeeze(self.model(x, training=True), axis=-1)
            per_sample = tf.keras.backend.binary_crossentropy(y, p)
            loss = tf.reduce_mean(per_sample)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return per_sample

    def train(self, x, y, *, epochs, batch_size, rng):
        y = np.asarray(y, dtype=np.float32)
        n = len(y)
        loss_sum, sq_sum, seen, batches = 0.0, 0.0, 0, 0
        for _ in range(epochs):
            perm = rng.permutation(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                per_sample = self._step(x[idx], y[idx]).numpy().astype(np.float64)
                loss_sum += per_sample.mean()
                sq_sum += float(np.sum(per_sample ** 2))
                seen += len(idx)
                batches += 1
        return TrainResult(self.model.get_weights(), n, loss_sum / max(batches, 1), n * (sq_sum / max(seen, 1)) ** 0.5)

    def predict(self, x):
        if len(x) == 0:
            return np.zeros(0, dtype=np.float32)
        out = [self._forward(x[i:i + self.predict_batch]).numpy() for i in range(0, len(x), self.predict_batch)]
        return np.concatenate(out).reshape(-1)


# =====================================================================================================================
# Client hardware profiles: reported latency / energy, and Oort's client durations
# =====================================================================================================================
EDGE_TIERS = {   # name: (cpu Hz, power coefficient alpha, tx power W, link bits/s)
    "wearable":   (0.8e9, 1.2e-27, 0.1, 5e6),
    "smartwatch": (1.5e9, 1.0e-27, 0.2, 15e6),
    "smartphone": (2.4e9, 0.8e-27, 0.4, 30e6),
    "edge_hub":   (3.0e9, 0.5e-27, 0.5, 50e6),
}
TIER_PROBS = (0.3, 0.4, 0.2, 0.1)


def build_profiles(client_sizes, local_epochs, model, seed):
    """Per client: tier, latency per round (T_cmp + download + upload) and energy (alpha f^3 T_cmp + radio)."""
    fwd = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            fwd += 2 * layer.output_shape[1] * layer.kernel_size[0] * layer.input_shape[-1] * layer.filters
        elif isinstance(layer, tf.keras.layers.Dense):
            fwd += 2 * layer.input_shape[-1] * layer.units
    flops, bits = 3 * fwd, model.count_params() * 32
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5157]))
    names = list(EDGE_TIERS)
    tiers = [names[i] for i in rng.choice(len(names), size=len(client_sizes), p=TIER_PROBS)]
    f, alpha, p_tx, rate = np.array([EDGE_TIERS[t] for t in tiers]).T
    t_cmp = flops * np.asarray(client_sizes, dtype=float) * local_epochs / f
    t_com = bits / rate
    return tiers, t_cmp + 2 * t_com, alpha * f ** 3 * t_cmp + p_tx * t_com


# =====================================================================================================================
# Plots (per run)
# =====================================================================================================================
def plot_metrics(history, out_dir):
    """Saves the run's plots; a locked file (a PNG open in a viewer) skips the plots for this round."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from collections import Counter

        rounds = [h["round"] for h in history]

        def line_plot(fname, title, ylabel, series):
            plt.figure(figsize=(8, 5))
            for key, label, style in series:
                plt.plot(rounds, [h.get(key) if h.get(key) is not None else np.nan for h in history], style, label=label)
            plt.xlabel("Federated round")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fname))
            plt.close()

        line_plot("auprc_vs_round.png", "Test AUPRC vs round", "AUPRC",
                  [("test_auprc", "Test AUPRC", "-"), ("test_roc_auc", "Test ROC-AUC", "--")])
        line_plot("loss_vs_round.png", "Test loss vs round", "BCE", [("test_loss", "Test loss", "-"),
                                                                     ("val_loss", "Validation loss (V)", "--")])
        line_plot("accuracy_vs_round.png", f"Thresholded metrics (threshold {THRESHOLD}) vs round", "Score",
                  [("test_f2", "F2", "-"), ("test_f1", "F1", "-"), ("test_recall", "Recall", "--"),
                   ("test_precision", "Precision", "--")])

        last = history[-1]
        cm = np.array([[last["test_tn"], last["test_fp"]], [last["test_fn"], last["test_tp"]]])
        plt.figure(figsize=(5, 4.5))
        plt.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center", fontsize=12)
        plt.xticks([0, 1], ["Normal", "AFib"])
        plt.yticks([0, 1], ["Normal", "AFib"])
        plt.xlabel(f"Predicted (threshold {THRESHOLD})")
        plt.ylabel("True")
        plt.title(f"Test confusion matrix, round {last['round']}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "confusion_matrix_latest.png"))
        plt.close()

        counts = Counter(h["choice"] for h in history)
        plt.figure(figsize=(7, 4.5))
        bars = plt.bar(list(counts), list(counts.values()), color="mediumpurple", edgecolor="black")
        for b in bars:
            plt.text(b.get_x() + b.get_width() / 2, b.get_height(), str(int(b.get_height())), ha="center", va="bottom")
        plt.xlabel("Rule kept by the server")
        plt.ylabel("Rounds")
        plt.title("Server rule kept per round")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "strategy_selection_freq.png"))
        plt.close()
    except OSError as e:
        print(f"[Plot Warning] Skipped saving plots this round: {e}")


# =====================================================================================================================
# Simulation
# =====================================================================================================================
def parse_seeds(text):
    seeds = []
    for part in str(text).split(","):
        a, _, b = part.partition("-")
        seeds += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return seeds


def parse_rounds(text):
    """'1-40,81-120' -> set of rounds ('' = every round)."""
    return set(parse_seeds(text)) if text else set()


def seeded(*key):
    return np.random.default_rng(np.random.SeedSequence(list(key)))


def run_config(args, seed):
    """Everything that defines a run (compared on resume)."""
    keys = ("data_dir", "num_clients", "select_k", "rounds", "local_epochs", "batch_size", "lr", "val_fraction",
            "val_size", "val_pos_share", "normalize", "selector", "aggregator", "rules", "rule_args", "controller_args",
            "server_lr", "ft_epochs", "ft_lr_scale", "select_frac", "select_score", "num_malicious", "attack",
            "attack_scale", "flip", "attack_rounds")
    return dict({k: getattr(args, k) for k in keys}, seed=seed)


def save_history(history, out_dir):
    with open(os.path.join(out_dir, "round_history.json"), "w") as f:
        json.dump(history, f, indent=1)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(history)


def _fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def run_simulation(args, seed, out_dir):
    """One strategy x one seed into out_dir."""
    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_path) and not args.overwrite:
        print(f"[Skip] seed {seed}: already finished in {out_dir} (use --overwrite to run again)")
        return
    os.makedirs(out_dir, exist_ok=True)
    cfg = run_config(args, seed)
    ckpt_path = os.path.join(out_dir, "checkpoint.pkl")
    if args.overwrite:
        for f in (summary_path, ckpt_path):
            if os.path.exists(f):
                os.remove(f)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    rules = [r for r in args.rules.split(",") if r] if args.rules else None
    rule_args = {"fedyogi": {"eta": args.server_lr}, "normmatch": {"eta": args.server_lr}}
    for k, v in args.rule_args.items():
        rule_args[k] = dict(rule_args.get(k, {}), **v)
    controller_args = dict(args.controller_args)
    if args.aggregator == "random_mix":
        controller_args.setdefault("seed", seed)      # a fresh coin-flip schedule per seed, not one shared schedule
    controller = build_controller(args.aggregator, rules=rules, rule_args=rule_args, **controller_args)
    attack_rounds = parse_rounds(args.attack_rounds)

    gpus = tf.config.list_physical_devices("GPU")
    print("=" * 64)
    print(" SmartMonitoring FL simulation (single process)")
    print("=" * 64)
    print(f" Data Directory    : {args.data_dir}")
    print(f" Clients (N)       : {args.num_clients or 'all'}   Selected per round (K): {args.select_k}")
    print(f" Rounds (R)        : {args.rounds}   Local epochs (E): {args.local_epochs}   Batch: {args.batch_size}")
    print(f" Server strategy   : {args.aggregator}  rules: {', '.join(controller.rules)}"
          + (f"  args: {json.dumps(args.controller_args)}" if args.controller_args else ""))
    print(f" Client selector   : {args.selector}")
    print(f" Client lr / FedYogi eta / server-training step: {args.lr} / {args.server_lr} / {args.ft_lr_scale} x client lr")
    if args.val_size or args.select_frac or args.select_score != "loss":
        print(f" Server set V      : size {args.val_size or 'all'}, AFib share {args.val_pos_share or 'matched'}, "
              f"scoring split {args.select_frac or 'none'}, score {args.select_score}")
    if args.num_malicious:
        print(f" Poisoning         : {args.num_malicious} malicious clients, attack {args.attack} x{args.attack_scale}"
              f"{', flipped' if args.flip else ''}{', rounds ' + args.attack_rounds if args.attack_rounds else ''}")
    print(f" Seed              : {seed}")
    print(f" Output Directory  : {out_dir}")
    print(f" Device            : {'GPU' if gpus else 'CPU (TensorFlow sees no GPU: run from an activated conda env)'}")
    print("=" * 64)

    data = build_federated_data(args.data_dir, seed, args.val_fraction, args.val_size, args.num_clients,
                                args.val_pos_share, args.normalize)
    sizes = data.client_sizes
    n_clients = len(sizes)
    print(f"[Round 0] {n_clients} clients, {int(sizes.sum())} training records "
          f"({int(sum(int(np.sum(y >= 0.5)) for y in data.client_y))} AFib); "
          f"V {len(data.val_y)} ({int(np.sum(data.val_y >= 0.5))} AFib); T {len(data.test_y)} "
          f"({int(np.sum(data.test_y >= 0.5))} AFib)")

    model = build_model()
    trainable = trainable_mask(model)
    trainer = Trainer(model, lr=args.lr)
    server_trainer = Trainer(model, lr=args.lr * args.ft_lr_scale)
    w = model.get_weights()

    sel_idx, ft_idx = split_validation(len(data.val_y), args.select_frac, seed)
    xv_sel, yv_sel = data.val_x[sel_idx], data.val_y[sel_idx]
    xv_ft, yv_ft = data.val_x[ft_idx], data.val_y[ft_idx]
    tiers, latency, energy = build_profiles(sizes, args.local_epochs, model, seed)
    selector_seed = int(np.random.SeedSequence([seed, 0x0027]).generate_state(1)[0])
    selector = (get_selector_by_name("oort", cfg=OortConfig(), seed=selector_seed) if args.selector == "oort"
                else get_selector_by_name(args.selector, seed=selector_seed))
    malicious = (set(seeded(seed, 0xBAD).choice(n_clients, min(args.num_malicious, n_clients), replace=False).tolist())
                 if args.num_malicious > 0 else set())
    context = {"client_samples": {c: float(n) for c, n in enumerate(sizes)},
               "client_durations": {c: float(latency[c]) for c in range(n_clients)}}

    def evaluate(weights):
        """What the server scores a candidate by on V (lower is better): BCE (default), Brier or 1 - AUPRC."""
        model.set_weights(weights)
        p = trainer.predict(xv_sel)
        acc = float(np.mean((p >= THRESHOLD) == (yv_sel >= 0.5)))
        if args.select_score == "brier":
            score = float(np.mean((p - yv_sel) ** 2))
        elif args.select_score == "auprc" and len(np.unique(yv_sel >= 0.5)) > 1:
            score = 1.0 - float(average_precision_score(yv_sel >= 0.5, p))
        else:
            score = bce(yv_sel, p)
        return {"loss": score, "accuracy": acc}

    def make_finetune(t):
        calls = [0]

        def finetune(weights):
            calls[0] += 1
            model.set_weights(weights)
            return server_trainer.train(xv_ft, yv_ft, epochs=args.ft_epochs, batch_size=args.batch_size,
                                        rng=seeded(seed, t, 0x5E7F, calls[0])).weights
        return finetune

    history, start, cum_latency, cum_energy = [], 1, 0.0, 0.0
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            ck = pickle.load(f)
        if ck["config"] != cfg:
            raise SystemExit(f"[Error] {out_dir} holds an unfinished run with other settings; use another -o or "
                             f"--overwrite")
        w, selector, history = ck["weights"], ck["selector"], ck["history"]
        controller.load_state_dict(ck["controller"])
        cum_latency, cum_energy, start = ck["cum_latency"], ck["cum_energy"], ck["round"] + 1
        print(f"[Resume] continuing after round {ck['round']}")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    t_run = time.time()
    for t in range(start, args.rounds + 1):
        print(f"\n--- Round {t}/{args.rounds} ---")
        t0 = time.time()
        selected = selector.select_clients(list(range(n_clients)), args.select_k, context=dict(context, round=t))
        print(f"Selected Clients for Round {t}: {selected}")

        results = {}
        for c in selected:
            model.set_weights(w)
            results[c] = trainer.train(data.client_x[c], data.client_y[c], epochs=args.local_epochs,
                                       batch_size=args.batch_size, rng=seeded(seed, t, c, 0xBA7C))
        attacking = (args.attack != "none" or args.flip) and (not attack_rounds or t in attack_rounds)
        bad = [c for c in selected if c in malicious] if attacking else []
        uploads = {c: r.weights for c, r in results.items()}
        if bad:
            benign = [c for c in selected if c not in malicious]
            benign_avg = (float(np.mean([update_norm([a - b for a, b in zip(results[c].weights, w)], trainable)
                                         for c in benign])) if benign else 0.0)
            for c in bad:
                uploads[c] = craft_malicious_weights(w, results[c].weights, trainable, args.attack,
                                                     benign_avg_norm=benign_avg, scale_factor=args.attack_scale,
                                                     flip=args.flip)
        if len(selected) <= 20:
            for c in selected:
                r = results[c]
                print(f" Client {c} [{tiers[c]}]: Trained | Samples: {r.num_samples} | Train loss: {r.train_loss:.4f} | "
                      f"Latency: {latency[c]:.2f}s | Energy: {energy[c]:.2f}J" + ("  (malicious)" if c in bad else ""))

        ctx = RoundContext(round=t, w_global=w, trainable=trainable,
                           updates=[ClientUpdate(c, uploads[c], results[c].num_samples) for c in selected],
                           evaluate=evaluate, finetune=make_finetune(t), num_malicious=len(bad))
        ts = time.time()
        w_new, info = controller.step(ctx)
        server_s = time.time() - ts
        selector.update_policy({"round": t, "selected_ids": selected,
                                "client_utilities": {c: r.stat_utility for c, r in results.items()},
                                "client_roundtrips": {c: float(latency[c]) for c in selected}})
        w = w_new
        choice = info.pop("choice")
        scores = {k[len("val_loss_"):]: v for k, v in info.items() if k.startswith("val_loss_")}
        msg = f"[Server {args.aggregator}] kept {choice}"
        if scores:
            msg += " | V score " + ", ".join(f"{k} {v:.4f}" for k, v in scores.items())
        if "reward" in info:
            msg += f" | reward {info['reward']:.4f}"
        print(msg + f" | V scores {info.get('server_evals', 0)}, V trainings {info.get('server_finetunes', 0)} "
                    f"| {server_s:.2f}s")

        model.set_weights(w)
        test = binary_metrics(data.test_y, trainer.predict(data.test_x))
        val = binary_metrics(data.val_y, trainer.predict(data.val_x))
        round_latency = float(latency[selected].max())
        round_energy = float(energy[selected].sum())
        cum_latency += round_latency
        cum_energy += round_energy
        elapsed = time.time() - t0
        row = {"seed": seed, "round": t, "strategy": args.aggregator, "choice": choice,
               **{f"test_{k}": v for k, v in test.items()},
               "val_auprc": val["auprc"], "val_loss": val["loss"],
               "server_s": server_s, **info,
               "mean_train_loss": float(np.mean([r.train_loss for r in results.values()])),
               "n_malicious": len(bad), "round_latency": round_latency, "cum_latency": cum_latency,
               "round_energy": round_energy, "cum_energy": cum_energy,
               "selected": " ".join(map(str, selected)), "wall_s": elapsed}
        history.append(row)
        print(f" Round {t} Results | Test AUPRC: {_fmt(test['auprc'])} | ROC-AUC: {_fmt(test['roc_auc'])} | "
              f"F2: {test['f2']:.4f} | Rec: {test['recall']:.4f} | Prec: {test['precision']:.4f} | "
              f"Brier: {test['brier']:.4f} | Loss: {test['loss']:.4f} | Time: {elapsed:.2f}s")

        save_history(history, out_dir)
        tmp = ckpt_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"config": cfg, "round": t, "weights": w, "selector": selector, "history": history,
                         "controller": controller.state_dict(), "cum_latency": cum_latency,
                         "cum_energy": cum_energy}, f)
        os.replace(tmp, ckpt_path)
        if t % 10 == 0 or t == args.rounds:
            plot_metrics(history, out_dir)
        if not math.isfinite(test["loss"]) or not all(np.all(np.isfinite(a)) for a in w):
            print(f"\n[CRITICAL] Round {t}: NaN/Inf in the global model; stopping this run.")
            plot_metrics(history, out_dir)
            break

    auprc = [h["test_auprc"] for h in history if h["test_auprc"] is not None]
    summary = {"seed": seed, "rounds": len(history), "auprc_auc": float(np.mean(auprc)) if auprc else None,
               "final_auprc": float(np.mean(auprc[-5:])) if auprc else None,
               "total_energy_j": cum_energy, "total_latency_s": cum_latency, "wall_min": (time.time() - t_run) / 60}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print("\n" + "=" * 64)
    print(f" Run finished: seed {seed}, AUPRC-AUC {_fmt(summary['auprc_auc'])}, final AUPRC {_fmt(summary['final_auprc'])}")
    print(f" Logs & plots: {os.path.abspath(out_dir)}")
    print("=" * 64)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("FL controls")
    g.add_argument("--data-dir", default=os.path.join("..", "datasets", "non_iid"), help="client partitions folder")
    g.add_argument("-n", "--num-clients", type=int, default=0, help="total clients N (0 = every client file)")
    g.add_argument("-k", "--select-k", type=int, default=20, help="clients selected per round K (default 20)")
    g.add_argument("-r", "--rounds", type=int, default=120, help="FL rounds (base 120)")
    g.add_argument("-e", "--local-epochs", type=int, default=5, help="local epochs per round (base 5)")
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--lr", type=float, default=0.1, help="client SGD learning rate (calibrated 0.1)")
    g.add_argument("--normalize", choices=["record", "none"], default="record", help="per-record z-score (calibrated)")
    g.add_argument("--seed", type=int, default=None, help="one seed, written to -o itself")
    g.add_argument("--seeds", default=None, help=f"seed range (default {BASE_SEEDS}), written to -o/s<seed>")
    g.add_argument("-o", "--output-dir", default=os.path.join("results", "run"))
    g.add_argument("--overwrite", action="store_true", help="run again even if the run already finished")

    g = p.add_argument_group("server strategy and client selection")
    g.add_argument("-a", "--aggregator", default="select_train",
                   help=f"a rule ({', '.join(sorted(RULES))}) or a controller ({', '.join(sorted(CONTROLLERS))})")
    g.add_argument("-s", "--selector", default="oort", choices=sorted(SELECTORS))
    g.add_argument("--rules", default="", help="candidate rules of a controller, comma list (default: its own)")
    g.add_argument("--mix-p", type=float, default=None, help="random_mix: probability of FedYogi each round")
    g.add_argument("--fs-preset", choices=["paper", "authors"], default=None, help="fedstrategist preset")
    g.add_argument("--fs-reward", choices=["accuracy", "loss"], default=None, help="fedstrategist reward")
    g.add_argument("--controller-args", default="{}", help='other controller arguments as JSON')
    g.add_argument("--rule-args", default="{}", help='rule arguments as JSON, e.g. {"krum": {"num_malicious": 2}}')
    g.add_argument("--server-lr", type=float, default=0.01, help="FedYogi step eta (calibrated 0.01)")
    g.add_argument("--ft-epochs", type=int, default=1, help="server-training epochs on V")
    g.add_argument("--ft-lr-scale", type=float, default=1.0, help="server-training step = client lr x this")

    g = p.add_argument_group("server set V")
    g.add_argument("--val-fraction", type=float, default=0.5, help="share of the pooled test splits that becomes V")
    g.add_argument("--val-size", type=int, default=0, help="subsample V to this many records (0 = all)")
    g.add_argument("--val-pos-share", type=float, default=0.0, help="with --val-size: AFib share of V")
    g.add_argument("--select-frac", type=float, default=0.0,
                   help="share of V used only for scoring (rest only for training); 0 = all of V for both")
    g.add_argument("--select-score", choices=["loss", "brier", "auprc"], default="loss",
                   help="what the server scores candidates by on V (lower is better)")

    g = p.add_argument_group("poisoning")
    g.add_argument("--num-malicious", type=int, default=0, help="malicious clients (fixed per seed)")
    g.add_argument("--attack", choices=ATTACKS, default="none")
    g.add_argument("--attack-scale", type=float, default=5.0)
    g.add_argument("--flip", action="store_true", help="reverse the malicious update direction")
    g.add_argument("--attack-rounds", default="", help="rounds when attackers are active, e.g. 1-40,81-120 (default all)")
    return p


def prepare_args(argv=None):
    """Parsed arguments with the JSON options decoded and the shortcut flags folded into the controller arguments."""
    args = build_parser().parse_args(argv)
    args.controller_args = json.loads(args.controller_args)
    args.rule_args = json.loads(args.rule_args)
    for key, value in (("p", args.mix_p), ("preset", args.fs_preset), ("reward_metric", args.fs_reward)):
        if value is not None:
            args.controller_args[key] = value
    if args.seed is not None and args.seeds:
        raise SystemExit("use --seed or --seeds, not both")
    return args


def main(argv=None):
    args = prepare_args(argv)
    if args.seed is not None:
        run_simulation(args, args.seed, args.output_dir)
        return
    seeds = parse_seeds(args.seeds or BASE_SEEDS)
    for i, seed in enumerate(seeds, 1):
        print(f"\n######## seed {seed} ({i}/{len(seeds)}) ########")
        run_simulation(args, seed, os.path.join(args.output_dir, f"s{seed}"))


if __name__ == "__main__":
    main()
