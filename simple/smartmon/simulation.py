"""Single-process federated simulation. The methods are the distributed system's plugins (distributed/server/plugins):
the client selector, the server controller with its rules, and the attacks, so a result here is a result of the code
that runs in the deployed system.

One round t:
    S_t     = selector.select_clients(K)                      (Oort: statistical utility x speed, explore/exploit)
    w_k     = local training of w_t on client k, k in S_t     (E epochs of plain SGD, unweighted BCE, as the
                                                               TMLR supplement); malicious clients then attack
    w_{t+1} = controller.step(w_t, {w_k})                     (static rule | select | train | select_train |
                                                               train_select | fedstrategist ...)
    selector <- each client's statistical utility and duration
    log test metrics of w_{t+1} on T (never used for a decision) and the server's cost of the decision

Per-round rows go to <out>/<name>_s<seed>.partial.csv and a checkpoint is written after every round, so an interrupted
run resumes where it stopped; the CSV is renamed to .csv when the run finishes.
"""
import csv
import json
import math
import os
import pickle
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import tensorflow as tf

from . import PLUGINS_DIR  # noqa: F401  (loads the plugin package)
from plugins import ClientUpdate, RoundContext, build_controller, craft_malicious_weights, get_selector_by_name
from plugins.attacks import update_norm
from plugins.oort import OortConfig

from .data import build_federated_data
from .ecg_model import THRESHOLD, Trainer, bce, binary_metrics, build_model, trainable_mask
from .profiles import build_profiles


@dataclass
class SimConfig:
    data_dir: str = "../datasets/non_iid"
    num_clients: int = 0             # 0: every client file in data_dir
    clients_per_round: int = 10      # K (Oort / FedScale default)
    rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    lr: float = 0.1                  # client SGD learning rate (ECG calibration, experiments/ecg_protocol.md)
    val_fraction: float = 0.5        # share of the pooled client test splits that becomes V (the rest is T)
    val_size: int = 0                # >0: subsample V to this many records (small-V ablation)
    val_pos_share: float = 0.0       # >0 (with val_size): draw V with this AFib share (label-skewed server set)
    normalize: str = "record"        # none | record (per-record, per-lead z-score; ECG calibration)
    seed: int = 42
    # -- plugins -------------------------------------------------------------------------------
    selector: str = "oort"
    strategy: str = "select_train"   # a controller (select_train, select, train, train_select, fedstrategist, sara) or a rule
    rules: tuple = ()                # () = the controller's default rules
    rule_args: dict = field(default_factory=dict)        # e.g. {"krum": {"num_malicious": 2}}
    controller_args: dict = field(default_factory=dict)  # e.g. {"preset": "authors", "lambda_cost": 0.5}
    server_lr: float = 0.01          # FedYogi eta (unless rule_args sets it)
    ft_epochs: int = 1               # server training on V: epochs
    ft_lr_scale: float = 1.0         # server training lr = client lr x this
    select_frac: float = 0.0         # >0: this share of V only scores candidates, the rest only trains
    # -- poisoning ---------------------------------------------------------------------------
    num_malicious: int = 0           # fixed malicious clients (drawn per seed); they attack whenever selected
    attack: str = "none"             # none | standard | stealth
    attack_scale: float = 5.0
    flip: bool = False
    name: str = ""                   # run label; default: the strategy
    oort: OortConfig = field(default_factory=OortConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rules"] = list(d["rules"])
        return d

    @property
    def label(self) -> str:
        return self.name or re.sub(r"[^A-Za-z0-9_.+-]", "-", self.strategy)


def run_stem(cfg: SimConfig, out_dir: str) -> str:
    return os.path.join(out_dir, f"{cfg.label}_s{cfg.seed}")


def _seeded(*key) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(list(key)))


def run(cfg: SimConfig, out_dir: str, data=None, verbose: bool = True, log_every: int = 0) -> str:
    """Runs one strategy and seed; returns the CSV path. Prints rounds 1-3 and every 10th, or every `log_every`-th."""
    stem = run_stem(cfg, out_dir)
    if os.path.exists(stem + ".csv"):
        return stem + ".csv"
    os.makedirs(out_dir, exist_ok=True)
    tf.random.set_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data = data if data is not None else build_federated_data(cfg.data_dir, cfg.seed, cfg.val_fraction,
                                                              cfg.val_size, cfg.num_clients, cfg.val_pos_share,
                                                              cfg.normalize)
    model = build_model()
    trainable = trainable_mask(model)
    trainer = Trainer(model, lr=cfg.lr)
    server_trainer = Trainer(model, lr=cfg.lr * cfg.ft_lr_scale)
    w = model.get_weights()

    rule_args = {"fedyogi": {"eta": cfg.server_lr}, "normmatch": {"eta": cfg.server_lr}}
    for k, v in cfg.rule_args.items():
        rule_args[k] = dict(rule_args.get(k, {}), **v)
    controller = build_controller(cfg.strategy, rules=list(cfg.rules) or None, rule_args=rule_args,
                                  **cfg.controller_args)

    sel_idx, ft_idx = _split_validation(len(data.val_y), cfg.select_frac, cfg.seed)
    xv_sel, yv_sel = data.val_x[sel_idx], data.val_y[sel_idx]
    xv_ft, yv_ft = data.val_x[ft_idx], data.val_y[ft_idx]

    sizes = data.client_sizes
    n_clients = len(sizes)
    profiles = build_profiles(sizes, cfg.local_epochs, model, cfg.seed)
    selector_seed = int(np.random.SeedSequence([cfg.seed, 0x0027]).generate_state(1)[0])
    selector = (get_selector_by_name("oort", cfg=cfg.oort, seed=selector_seed) if cfg.selector == "oort"
                else get_selector_by_name(cfg.selector, seed=selector_seed))
    malicious = set(_seeded(cfg.seed, 0xBAD).choice(n_clients, min(cfg.num_malicious, n_clients), replace=False)
                    .tolist()) if cfg.num_malicious > 0 else set()
    context = {"client_samples": {c: float(n) for c, n in enumerate(sizes)},
               "client_durations": {c: float(profiles.latency[c]) for c in range(n_clients)}}

    def evaluate(weights):
        model.set_weights(weights)
        p = trainer.predict(xv_sel)
        return {"loss": bce(yv_sel, p), "accuracy": float(np.mean((p >= THRESHOLD) == (yv_sel >= 0.5)))}

    def make_finetune(t):
        calls = [0]

        def finetune(weights):
            calls[0] += 1
            model.set_weights(weights)
            return server_trainer.train(xv_ft, yv_ft, epochs=cfg.ft_epochs, batch_size=cfg.batch_size,
                                        rng=_seeded(cfg.seed, t, 0x5E7F, calls[0])).weights
        return finetune

    # -- resume ----------------------------------------------------------------------------------
    ckpt_path, partial = stem + ".ckpt.pkl", stem + ".partial.csv"
    start, cum_latency, cum_energy = 1, 0.0, 0.0
    if os.path.exists(ckpt_path) and os.path.exists(partial):
        with open(ckpt_path, "rb") as f:
            ck = pickle.load(f)
        if ck["config"] == cfg.to_dict():
            w, selector, start = ck["weights"], ck["selector"], ck["round"] + 1
            controller.load_state_dict(ck["controller"])
            cum_latency, cum_energy = ck["cum_latency"], ck["cum_energy"]
            _truncate_csv(partial, ck["round"])
            if verbose:
                print(f"[{cfg.label} s{cfg.seed}] resuming after round {ck['round']}", flush=True)
    if start == 1 and os.path.exists(partial):
        os.remove(partial)
    with open(stem + ".json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    out_f = open(partial, "a", newline="")
    writer = None
    try:
        for t in range(start, cfg.rounds + 1):
            t0 = time.perf_counter()
            selected = selector.select_clients(list(range(n_clients)), cfg.clients_per_round,
                                               context=dict(context, round=t))
            results = {}
            for c in selected:
                model.set_weights(w)
                results[c] = trainer.train(data.client_x[c], data.client_y[c], epochs=cfg.local_epochs,
                                           batch_size=cfg.batch_size, rng=_seeded(cfg.seed, t, c, 0xBA7C))
            bad = [c for c in selected if c in malicious]
            uploads = {c: r.weights for c, r in results.items()}
            if bad and (cfg.attack != "none" or cfg.flip):
                benign = [c for c in selected if c not in malicious]
                benign_avg = (float(np.mean([update_norm([a - b for a, b in zip(results[c].weights, w)], trainable)
                                             for c in benign])) if benign else 0.0)
                for c in bad:
                    uploads[c] = craft_malicious_weights(w, results[c].weights, trainable, cfg.attack,
                                                         benign_avg_norm=benign_avg, scale_factor=cfg.attack_scale,
                                                         flip=cfg.flip)
            ctx = RoundContext(round=t, w_global=w, trainable=trainable,
                               updates=[ClientUpdate(c, uploads[c], results[c].num_samples) for c in selected],
                               evaluate=evaluate, finetune=make_finetune(t), num_malicious=len(bad))
            ts = time.perf_counter()
            w_new, info = controller.step(ctx)
            server_s = time.perf_counter() - ts
            selector.update_policy({"round": t, "selected_ids": selected,
                                    "client_utilities": {c: r.stat_utility for c, r in results.items()},
                                    "client_roundtrips": {c: float(profiles.latency[c]) for c in selected}})
            w = w_new

            round_latency = float(profiles.latency[selected].max())
            round_energy = float(profiles.energy[selected].sum())
            cum_latency += round_latency
            cum_energy += round_energy
            model.set_weights(w)
            test = binary_metrics(data.test_y, trainer.predict(data.test_x))
            val = binary_metrics(data.val_y, trainer.predict(data.val_x))
            choice = info.pop("choice")
            row = {
                "strategy": cfg.label, "seed": cfg.seed, "round": t, "choice": choice,
                **{f"test_{k}": test[k] for k in ("auprc", "roc_auc", "f1", "f2", "precision", "recall", "accuracy",
                                                    "brier", "loss")},
                "val_auprc": val["auprc"], "val_loss": val["loss"],
                "server_s": server_s, **info,
                "mean_train_loss": float(np.mean([r.train_loss for r in results.values()])),
                "n_malicious": len(bad),
                "round_latency": round_latency, "cum_latency": cum_latency,
                "round_energy": round_energy, "cum_energy": cum_energy,
                "selected": " ".join(map(str, selected)),
                "wall_s": time.perf_counter() - t0,
            }
            if hasattr(selector, "oort"):
                row.update(oort_exploit=len(selector.oort.exploit_clients),
                           oort_explore=len(selector.oort.explore_clients))
            if writer is None:
                writer = csv.DictWriter(out_f, fieldnames=list(row), extrasaction="ignore")
                if out_f.tell() == 0:
                    writer.writeheader()
            writer.writerow(row)
            out_f.flush()
            _save_checkpoint(ckpt_path, {"config": cfg.to_dict(), "round": t, "weights": w, "selector": selector,
                                         "controller": controller.state_dict(), "cum_latency": cum_latency,
                                         "cum_energy": cum_energy})
            if verbose and ((t % log_every == 0) if log_every else (t <= 3 or t % 10 == 0)):
                print(f"[{cfg.label} s{cfg.seed}] round {t:3d} choice={choice:13s} "
                      f"AUPRC={_fmt(test['auprc'])} F2={test['f2']:.3f} Brier={test['brier']:.4f} "
                      f"server {server_s:.1f}s ({row['wall_s']:.1f}s)", flush=True)
            if not math.isfinite(test["loss"]) or not all(np.all(np.isfinite(a)) for a in w):
                print(f"[{cfg.label} s{cfg.seed}] diverged at round {t}", flush=True)
                break
    finally:
        out_f.close()
    os.replace(partial, stem + ".csv")
    os.remove(ckpt_path)
    return stem + ".csv"


def _split_validation(n: int, select_frac: float, seed: int):
    """Index sets (selection part, training part) of V; the same split as distributed/server/aggregator.py."""
    if select_frac <= 0:
        idx = np.arange(n)
        return idx, idx
    perm = np.random.default_rng(np.random.SeedSequence([seed, 0x5E1])).permutation(n)
    n_sel = max(1, int(round(n * select_frac)))
    return np.sort(perm[:n_sel]), np.sort(perm[n_sel:])


def _fmt(v) -> str:
    return "  n/a" if v is None else f"{v:.3f}"


def _save_checkpoint(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)


def _truncate_csv(path: str, last_round: int) -> None:
    """Drop rows written after the checkpointed round (a round can be logged and then interrupted)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    keep = [rows[0]] + [r for r in rows[1:] if r and int(r[2]) <= last_round]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(keep)
