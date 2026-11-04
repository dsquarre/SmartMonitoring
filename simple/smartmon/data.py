"""Client partitions and the server's holdout.

Each `client_<i>.npz` holds X_train / y_train / X_test / y_test. Clients keep their train split. The clients' test
splits are pooled and split once per seed into the server's validation set V (used for selection and server
training) and the reported test set T, which is never used for any decision. This mirrors the TMLR supplement,
where the server holdout is the dataset's test split divided into V and T.

Options: `val_size` subsamples V uniformly; with `val_pos_share` the subsample is drawn with that AFib share instead
(a label-skewed server set, e.g. 0.5 for a curated, AFib-enriched set); `normalize="record"` z-scores every record per
lead (no cross-client statistics needed).
"""
import glob
import os
import re
from dataclasses import dataclass

import numpy as np

from .ecg_model import to_array


@dataclass
class FederatedData:
    client_x: list
    client_y: list
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray

    @property
    def client_sizes(self) -> np.ndarray:
        return np.array([len(y) for y in self.client_y])

    def label_counts(self):
        """(AFib positives, total) over all client train splits: the registration-time count handshake."""
        return int(sum(int(np.sum(y >= 0.5)) for y in self.client_y)), int(sum(len(y) for y in self.client_y))


def client_files(data_dir: str, num_clients: int = 0) -> list:
    files = glob.glob(os.path.join(data_dir, "client_*.npz"))
    files.sort(key=lambda f: int(re.search(r"client_(\d+)\.npz$", f).group(1)))
    if not files:
        raise FileNotFoundError(f"no client_*.npz files in {data_dir}")
    return files[:num_clients] if num_clients else files


def normalize_records(X: np.ndarray) -> np.ndarray:
    """Per-record, per-lead z-score."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return ((X - mu) / np.maximum(sd, 1e-6)).astype(np.float32)


def build_federated_data(data_dir: str, seed: int, val_fraction: float = 0.5, val_size: int = 0,
                         num_clients: int = 0, val_pos_share: float = 0.0, normalize: str = "none") -> FederatedData:
    if normalize not in ("none", "record"):
        raise ValueError("normalize must be none or record")
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
    if val_size and val_pos_share > 0:            # label-skewed V: fixed AFib share, own RNG stream
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5CE, val_size, int(val_pos_share * 1000)]))
        pos, neg = val_idx[pool_y[val_idx] >= 0.5], val_idx[pool_y[val_idx] < 0.5]
        n_pos = int(round(val_size * val_pos_share))
        if n_pos > len(pos) or val_size - n_pos > len(neg):
            raise ValueError(f"V has {len(pos)} AFib / {len(neg)} normal records; cannot draw {val_size} at share "
                             f"{val_pos_share}")
        val_idx = np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False),
                                          rng.choice(neg, val_size - n_pos, replace=False)]))
    elif val_size and val_size < len(val_idx):    # small-V ablation: uniform subsample, own RNG stream
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x7A1, val_size]))
        val_idx = np.sort(rng.choice(val_idx, size=val_size, replace=False))
    return FederatedData(client_x, client_y, pool_x[val_idx], pool_y[val_idx], pool_x[test_idx], pool_y[test_idx])
