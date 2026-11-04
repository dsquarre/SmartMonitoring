"""Client behaviours for the poisoning experiments (FedStrategist Sec. 4.3; project x: src/attacks.py).

A malicious client trains honestly, then turns its update  delta = w_local - w_global  into an attack:
    none      the honest update
    standard  same direction, scale_factor (5x) the norm ("loud")
    stealth   same direction, norm matched to the mean benign update norm of the round. This needs the other clients'
              updates, so only the simulator can run it (the reference code is an omniscient simulation too)
flip=True reverses the direction (a real poisoning attack; an extension, not in the FedStrategist paper).

The attack acts on the trainable weights only. BatchNorm moving statistics stay honest: scaling a variance delta
can make the variance negative, which no aggregation rule is meant to defend against.

This file is copied byte-identically to distributed/client/attacks.py (the client image has no server code);
distributed/tests/test_plugins.py checks the copy.
"""
import numpy as np

ATTACKS = ("none", "standard", "stealth")


def update_norm(delta, trainable) -> float:
    return float(np.sqrt(sum(float(np.sum(d.astype(np.float64) ** 2)) for d, t in zip(delta, trainable) if t)))


def craft_malicious_weights(w_global, w_local, trainable, attack: str, benign_avg_norm: float = None,
                            scale_factor: float = 5.0, flip: bool = False):
    """Returns the weights the malicious client uploads instead of w_local."""
    if attack not in ATTACKS:
        raise ValueError(f"unknown attack {attack!r}; choose from {ATTACKS}")
    if attack == "none" and not flip:
        return [w.copy() for w in w_local]
    delta = [l - g for l, g in zip(w_local, w_global)]
    sign = -1.0 if flip else 1.0
    norm = update_norm(delta, trainable)
    if norm == 0.0:
        target = 0.0
    elif attack == "standard":
        target = scale_factor * norm
    elif attack == "stealth":
        if benign_avg_norm is None:
            raise ValueError("the stealth attack needs the benign mean update norm (simulation only)")
        target = float(benign_avg_norm)
    else:                       # "none" with flip: reversed honest update
        target = norm
    scale = sign * (target / norm if norm > 0 else 0.0)
    return [(g + scale * d).astype(g.dtype) if t else l.copy()
            for g, l, d, t in zip(w_global, w_local, delta, trainable)]
