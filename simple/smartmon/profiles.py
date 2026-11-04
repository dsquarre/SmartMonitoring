"""Deterministic client hardware profiles and per-round latency / energy (reported, and Oort's client durations).

Each client is drawn into one of four wearable / edge tiers. Per round:
    T_cmp = F * n_i * E / f_i              F = training FLOPs per sample, counted from the model (1 FLOP per cycle)
    T_com = 2 S / r_i                      S = model size in bits; download the global model, upload the update
    E_i   = alpha_i f_i^3 T_cmp + p_i S / r_i     (dynamic CPU power alpha f^3; radio energy for the upload)
Server-side computation (scoring candidates, training on V) is not part of the client cost.
"""
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

EDGE_TIERS = {   # name: (cpu Hz, power coefficient alpha, tx power W, link bits/s)
    "wearable":   (0.8e9, 1.2e-27, 0.1, 5e6),
    "smartwatch": (1.5e9, 1.0e-27, 0.2, 15e6),
    "smartphone": (2.4e9, 0.8e-27, 0.4, 30e6),
    "edge_hub":   (3.0e9, 0.5e-27, 0.5, 50e6),
}
TIER_PROBS = (0.3, 0.4, 0.2, 0.1)


@dataclass
class ClientProfiles:
    tier: list
    latency: np.ndarray      # seconds per round
    energy: np.ndarray       # joules per round


def model_cost(model: tf.keras.Model):
    """(training FLOPs per sample ~ 3x forward, model size in bits at float32)."""
    fwd = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            length = layer.output_shape[1]
            fwd += 2 * length * layer.kernel_size[0] * layer.input_shape[-1] * layer.filters
        elif isinstance(layer, tf.keras.layers.Dense):
            fwd += 2 * layer.input_shape[-1] * layer.units
    return 3 * fwd, model.count_params() * 32


def build_profiles(client_sizes, local_epochs: int, model: tf.keras.Model, seed: int) -> ClientProfiles:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5157]))
    names = list(EDGE_TIERS)
    tiers = [names[i] for i in rng.choice(len(names), size=len(client_sizes), p=TIER_PROBS)]
    flops, bits = model_cost(model)
    spec = np.array([EDGE_TIERS[t] for t in tiers])
    f, alpha, p_tx, rate = spec.T
    t_cmp = flops * np.asarray(client_sizes, dtype=float) * local_epochs / f
    t_com = bits / rate
    latency = t_cmp + 2 * t_com
    energy = alpha * f ** 3 * t_cmp + p_tx * t_com
    return ClientProfiles(tier=tiers, latency=latency, energy=energy)
