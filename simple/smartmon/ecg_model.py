"""ECG atrial-fibrillation model (1-D CNN on 12-lead, 1000-sample PTB-XL records): training, prediction, metrics.

Used by the simulated clients (local training) and server (training on V, scoring candidates). The distributed
system keeps its own Keras model (distributed/server/model.py, distributed/client/model.py).
"""
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

INPUT_SHAPE = (1000, 12)
THRESHOLD = 0.3          # clinical operating point for the thresholded metrics (recall-weighted F2)


def build_model(initial_bias: float = -2.2) -> tf.keras.Model:
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


def trainable_mask(model: tf.keras.Model) -> list:
    """Per entry of model.get_weights(): True for trainable weights, False for BatchNorm moving statistics."""
    trainable = {id(v) for v in model.trainable_weights}
    return [id(v) in trainable for v in model.weights]


def to_array(X) -> np.ndarray:
    """float32 (N, 1000, 12) with NaN/Inf zeroed; accepts (N, 12, 1000) too."""
    X = np.asarray(X)
    if X.ndim == 3 and X.shape[1] == 12 and X.shape[2] != 12:
        X = np.transpose(X, (0, 2, 1))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def bce(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary cross-entropy of probabilities p."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=np.float64)
    per = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.mean(per))


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = THRESHOLD) -> dict:
    """Ranking (AUPRC, ROC-AUC), calibration (Brier, BCE) and thresholded metrics. AUPRC / ROC-AUC are None when
    only one class is present (common for a single client's test split)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= threshold).astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    both = len(np.unique(y)) > 1
    out = {
        "n": int(len(y)),
        "loss": bce(y, p) if len(y) else 0.0,
        "auprc": float(average_precision_score(y, p)) if both else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    out.update(counts_to_scores(tp, fp, tn, fn))
    return out


def counts_to_scores(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Precision, recall, F1, F2 and accuracy from confusion counts (also used to pool clients' counts)."""
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if 4 * prec + rec else 0.0
    total = tp + fp + tn + fn
    return {"precision": prec, "recall": rec, "f1": f1, "f2": f2, "accuracy": (tp + tn) / total if total else 0.0}


@dataclass
class TrainResult:
    weights: list            # model.get_weights() after training
    num_samples: int
    train_loss: float        # mean BCE over the training batches
    stat_utility: float      # Oort: |B_i| * sqrt(mean per-sample training loss^2)


class Trainer:
    """Local training as in the TMLR supplement (hrl_fl/client.py local_train): plain SGD with no momentum, so every
    call starts from a fresh optimizer, on unweighted binary cross-entropy, with a compiled step. Also the server's
    training on V (a second Trainer on the same model at the server's learning rate)."""

    def __init__(self, model: tf.keras.Model, lr: float = 0.05, predict_batch: int = 512):
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

    def train(self, x: np.ndarray, y: np.ndarray, *, epochs: int, batch_size: int,
              rng: np.random.Generator) -> TrainResult:
        """`rng` fixes the batch order (seeded per seed, round and client by the caller)."""
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
        return TrainResult(weights=self.model.get_weights(), num_samples=n,
                           train_loss=loss_sum / max(batches, 1),
                           stat_utility=n * (sq_sum / max(seen, 1)) ** 0.5)


    def predict(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.zeros(0, dtype=np.float32)
        out = [self._forward(x[i:i + self.predict_batch]).numpy() for i in range(0, len(x), self.predict_batch)]
        return np.concatenate(out).reshape(-1)
