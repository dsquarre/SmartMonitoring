"""Contextual bandits and the update diagnostics used by the FedStrategist controller.

Ported from the FedStrategist reimplementation (project x: src/bandit.py, src/diagnostics.py), numpy instead of torch.
"""
import numpy as np


class CountUCB:
    """The bandit shipped in the FedStrategist reference code: per-arm ridge regression (no intercept) refit on the
    full history, plus a count bonus alpha * sqrt(1 / (n_a + 1)). Not the LinUCB bound written in the paper, but what
    produced its numbers."""

    def __init__(self, num_actions, d, alpha=1.5, ridge=1.0):
        self.alpha = alpha
        self.ridge = ridge
        self.d = d
        self.X = [[] for _ in range(num_actions)]
        self.y = [[] for _ in range(num_actions)]

    def select(self, x):
        scores = []
        for X, y in zip(self.X, self.y):
            if X:
                X_arr, y_arr = np.asarray(X), np.asarray(y)
                theta = np.linalg.solve(X_arr.T @ X_arr + self.ridge * np.eye(self.d), X_arr.T @ y_arr)
                mu = float(x @ theta)
            else:
                mu = 0.0
            scores.append(mu + self.alpha * np.sqrt(1.0 / (len(X) + 1)))
        return int(np.argmax(scores))

    def update(self, action, x, reward):
        self.X[action].append(np.asarray(x, dtype=float))
        self.y[action].append(float(reward))


class LinUCB:
    """Disjoint LinUCB (Li et al., 2010), as written in the FedStrategist paper:
    a_t = argmax_a  x^T theta_a + alpha * sqrt(x^T A_a^{-1} x)."""

    def __init__(self, num_actions, d, alpha=1.5):
        self.alpha = alpha
        self.A = [np.eye(d) for _ in range(num_actions)]
        self.b = [np.zeros(d) for _ in range(num_actions)]

    def select(self, x):
        scores = []
        for A, b in zip(self.A, self.b):
            A_inv = np.linalg.inv(A)
            scores.append(float(x @ A_inv @ b) + self.alpha * np.sqrt(float(x @ A_inv @ x)))
        return int(np.argmax(scores))

    def update(self, action, x, reward):
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x


class OnlineStandardizer:
    """Running z-score of the raw state (Welford), clipped to [-clip, clip], with a constant bias feature appended."""

    def __init__(self, d, bias=True, clip=5.0):
        self.n = 0
        self.mean = np.zeros(d)
        self.m2 = np.zeros(d)
        self.bias = bias
        self.clip = clip

    @property
    def out_dim(self):
        return len(self.mean) + int(self.bias)

    def __call__(self, raw):
        raw = np.asarray(raw, dtype=float)
        self.n += 1
        delta = raw - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (raw - self.mean)
        std = np.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else np.ones_like(raw)
        z = np.clip((raw - self.mean) / np.maximum(std, 1e-12), -self.clip, self.clip)
        return np.append(z, 1.0) if self.bias else z


def state_vector(vectors: np.ndarray) -> list:
    """S_t = [variance of the L2 norms (unbiased, as torch.var), mean pairwise cosine similarity, norm of the mean].
    `vectors` is (n_clients, n_params): client updates (paper) or full client weights (reference code)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    norm_var = float(np.var(norms, ddof=1)) if len(norms) > 1 else 0.0
    unit = vectors / np.maximum(norms[:, None], 1e-12)
    sim = unit @ unit.T
    iu = np.triu_indices(len(vectors), k=1)
    avg_cos = float(sim[iu].mean()) if len(iu[0]) else 1.0
    mean_norm = float(np.linalg.norm(vectors.mean(axis=0)))
    return [norm_var, avg_cos, mean_norm]
