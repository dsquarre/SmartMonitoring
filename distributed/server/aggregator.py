"""
Server-side aggregation for the Celery task. The methods themselves are plugins (plugins/: rules and controllers,
shared with the simple simulator); this module is the adapter between them and the system:

    model files (.keras) on S3  ->  numpy weights  ->  controller.step(RoundContext)  ->  new global model file
    controller state (FedYogi moments, bandit) <-> models/global/server_state_<t>.pkl, since each Celery task builds
    a fresh aggregator

and it gives the controllers their two model-specific callbacks on the server's validation set V
(SERVER_VAL_PATH, an .npz with X and y): `evaluate` (BCE and accuracy at the 0.3 threshold) and `finetune` (plain SGD,
fresh optimizer, at the clients' learning rate x FL_FT_LR_SCALE and batch size, FL_FT_EPOCHS epochs).

FL_AGGREGATOR names the server strategy: a controller (select_train, select, train, train_select, fedstrategist, sara) or a
static rule (fedavg, fedyogi, median, krum, or static:<rule>). FL_RULES overrides the controller's candidate rules
(comma list), FL_PLUGIN_ARGS (JSON) passes controller arguments, e.g. {"preset": "authors", "lambda_cost": 0.5}.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import json
import os
import pickle
import gc
import numpy as np
import tensorflow as tf
from model import Model
from plugins import ClientUpdate, RoundContext, build_controller, parse_spec

THRESHOLD = 0.3

class ModelAggregator(ABC):
    @property
    def mode(self):
        """All strategies exchange model weights."""
        return "weights"

    @abstractmethod
    def aggregate(self, client_data: List[Tuple[str, float, float, str]], global_model_path: str, current_round: int):
        """
        Aggregate local client models and save the new global model.

        Args:
            client_data: List of tuples containing (local_model_filepath, number_of_samples, loss, client_id, ...).
            global_model_path: Filepath of the previous global model, overwritten with the new one.
            current_round: The current federated learning round index (1-based).
        """
        pass

# ----------------------------------------------------------------------------------------------
# Server validation set V
# ----------------------------------------------------------------------------------------------
_VAL_CACHE = {}

def load_server_validation(path: str):
    """Server validation set V: an .npz with X (n, 1000, 12) and y (n,), cached per worker process."""
    key = os.path.abspath(path)
    if key not in _VAL_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Server validation set not found at '{path}'. Build it with scripts/make_server_val.py "
                f"or set SERVER_VAL_PATH (static rules run without it).")
        with np.load(path) as data:
            X = np.asarray(data["X"])
            y = np.asarray(data["y"]).astype(np.float32).reshape(-1)
        if X.ndim == 3 and X.shape[1] == 12 and X.shape[2] != 12:
            X = np.transpose(X, (0, 2, 1))
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        # Per-record, per-lead z-score, as the clients (chosen by the ECG calibration)
        X = ((X - X.mean(axis=1, keepdims=True)) / np.maximum(X.std(axis=1, keepdims=True), 1e-6)).astype(np.float32)
        _VAL_CACHE[key] = (X, y)
    return _VAL_CACHE[key]

def bce(y, p) -> float:
    """Mean binary cross-entropy (the supplement selects on plain, unweighted cross-entropy)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), 1e-7, 1.0 - 1e-7)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

def trainable_mask(keras_model) -> list:
    """Per entry of get_weights(): True for trainable weights, False for BatchNorm moving statistics."""
    trainable = {id(v) for v in keras_model.trainable_weights}
    return [id(v) in trainable for v in keras_model.weights]

def split_validation(n: int, select_frac: float, seed: int):
    """Index sets (selection part, training part) of V. select_frac = 0: both are all of V."""
    if select_frac <= 0:
        idx = np.arange(n)
        return idx, idx
    perm = np.random.default_rng(np.random.SeedSequence([seed, 0x5E1])).permutation(n)
    n_sel = max(1, int(round(n * select_frac)))
    return np.sort(perm[:n_sel]), np.sort(perm[n_sel:])

# ----------------------------------------------------------------------------------------------
# The adapter
# ----------------------------------------------------------------------------------------------
def _env(name, cast, default):
    value = os.environ.get(name)
    return cast(value) if value not in (None, "") else default

class PluginAggregator(ModelAggregator):
    """
    Runs a plugin controller on the round's client models.
    """
    def __init__(self, strategy: str = "select_train", rules=None, rule_args=None, controller_args=None,
                 client_lr: float = 0.1, ft_lr_scale: float = 1.0, ft_epochs: int = 1, batch_size: int = 32,
                 select_frac: float = 0.0, num_malicious: int = 0, val_path: str = None, seed: int = 0):
        parse_spec(strategy)   # fail early on an unknown name
        self.strategy = strategy
        self.rules = list(rules) if rules else None
        self.rule_args = dict(rule_args or {})
        self.controller_args = dict(controller_args or {})
        self.client_lr = float(client_lr)
        self.ft_lr_scale = float(ft_lr_scale)
        self.ft_epochs = int(ft_epochs)
        self.batch_size = int(batch_size)
        self.select_frac = float(select_frac)
        self.num_malicious = int(num_malicious)
        self.val_path = val_path or os.environ.get("SERVER_VAL_PATH", "server_val.npz")
        self.seed = int(seed)
        self.last_info = {}

    @classmethod
    def from_env(cls, strategy: str):
        rules = os.environ.get("FL_RULES")
        return cls(strategy=strategy,
                   rules=[r.strip() for r in rules.split(",") if r.strip()] if rules else None,
                   rule_args={r: {"eta": _env("FL_SERVER_LR", float, 0.01)} for r in ("fedyogi", "normmatch")},
                   controller_args=json.loads(os.environ.get("FL_PLUGIN_ARGS") or "{}"),
                   client_lr=_env("FL_LR", float, 0.1), ft_lr_scale=_env("FL_FT_LR_SCALE", float, 1.0),
                   ft_epochs=_env("FL_FT_EPOCHS", int, 1), batch_size=_env("FL_BATCH_SIZE", int, 32),
                   select_frac=_env("FL_SELECT_FRAC", float, 0.0), num_malicious=_env("FL_NUM_MALICIOUS", int, 0),
                   val_path=os.environ.get("SERVER_VAL_PATH"), seed=_env("FL_SEED", int, 0))

    def config(self) -> dict:
        """Constructor arguments, passed through Celery (get_aggregator_by_name(name, **config))."""
        return {k: getattr(self, k) for k in ("strategy", "rules", "rule_args", "controller_args", "client_lr",
                                              "ft_lr_scale", "ft_epochs", "batch_size", "select_frac",
                                              "num_malicious", "val_path", "seed")}

    def build_controller(self):
        return build_controller(self.strategy, rules=self.rules, rule_args=self.rule_args, **self.controller_args)

    def _load_state(self, controller, current_round):
        from s3_helper import download_file
        local = f"tmp_server_state_{current_round - 1}.pkl"
        if current_round > 1 and download_file(f"models/global/server_state_{current_round - 1}.pkl", local):
            with open(local, "rb") as f:
                controller.load_state_dict(pickle.load(f))
            os.remove(local)

    def _save_state(self, controller, current_round):
        from s3_helper import upload_file
        local = f"tmp_server_state_{current_round}.pkl"
        with open(local, "wb") as f:
            pickle.dump(controller.state_dict(), f)
        ok = upload_file(local, f"models/global/server_state_{current_round}.pkl")
        os.remove(local)
        if not ok:
            raise RuntimeError("Failed to store the server state")

    def _validation_callbacks(self, model: Model):
        if not os.path.exists(self.val_path):
            return None, None, None
        X_val, y_val = load_server_validation(self.val_path)
        sel_idx, ft_idx = split_validation(len(y_val), self.select_frac, self.seed)

        def evaluate(weights):
            model.model.set_weights(weights)
            p = model.model.predict(X_val[sel_idx], batch_size=256, verbose=0).reshape(-1)
            y = y_val[sel_idx]
            return {"loss": bce(y, p), "accuracy": float(np.mean((p >= THRESHOLD) == (y >= 0.5)))}

        def finetune(weights):
            # server-ft: plain SGD (fresh, no momentum) on V at the clients' lr and batch size
            model.model.set_weights(weights)
            model.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.client_lr * self.ft_lr_scale,
                                                                  momentum=0.0), loss="binary_crossentropy")
            model.model.fit(X_val[ft_idx], y_val[ft_idx], epochs=self.ft_epochs, batch_size=self.batch_size,
                            shuffle=True, verbose=0)
            return model.model.get_weights()

        return evaluate, finetune, len(y_val)

    def aggregate(self, client_data, global_model_path, current_round):
        controller = self.build_controller()
        print(f"Using {self.strategy} ({controller.__class__.__name__}, rules: {', '.join(controller.rules)})")
        self._load_state(controller, current_round)

        model = Model()
        model.model.load_weights(global_model_path)
        w_global = model.model.get_weights()
        updates = []
        for item in client_data:
            model_path, samples, client_id = item[0], item[1], item[3]
            model.model.load_weights(model_path)
            updates.append(ClientUpdate(client_id=client_id, weights=model.model.get_weights(), num_samples=samples))

        evaluate, finetune, _ = self._validation_callbacks(model)
        ctx = RoundContext(round=current_round, w_global=w_global, trainable=trainable_mask(model.model),
                           updates=updates, evaluate=evaluate, finetune=finetune, num_malicious=self.num_malicious)
        w_new, info = controller.step(ctx)

        self._save_state(controller, current_round)
        model.model.set_weights(w_new)
        model.model.save(global_model_path)
        info = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in info.items()}
        self.last_info = {"server_choice": info.pop("choice"), **info}
        print(f"[{self.strategy}] round {current_round}: kept {self.last_info['server_choice']} "
              f"({', '.join(f'{k}={v:.4f}' for k, v in info.items() if k.startswith('val_loss_') and v is not None)})")
        del model
        tf.keras.backend.clear_session()
        gc.collect()

def get_aggregator_by_name(name: str, **kwargs) -> ModelAggregator:
    """
    The coordinator calls this with FL_AGGREGATOR (settings from the environment); the Celery task calls it with the
    class name and the coordinator's config().
    """
    if name == "PluginAggregator":
        return PluginAggregator(**kwargs)
    if kwargs:
        return PluginAggregator(strategy=name, **kwargs)
    return PluginAggregator.from_env(name)
