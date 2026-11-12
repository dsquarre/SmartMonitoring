"""Plugin interfaces, shared by the distributed system (the Celery task) and the simple simulator.

Plugins are numpy-only. A model is a list of numpy arrays in Keras get_weights() order, plus a `trainable` mask of the
same length (False for BatchNorm moving statistics). Anything that needs the model itself (scoring a candidate on the
server's validation set V, training on V) is passed in by the host as a callback in the RoundContext, so the same
plugin code runs in both hosts.

Three plugin kinds, each with a registry:
    client selectors (selectors.py)  which clients train this round
    server rules     (rules.py)      one way to combine the client updates into a candidate global model
    controllers      (controllers.py) which rule(s) to use each round, and whether to train on V
plus client behaviours (attacks.py) for the poisoning experiments.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

Weights = List[np.ndarray]


@dataclass
class ClientUpdate:
    client_id: object
    weights: Weights
    num_samples: float


@dataclass
class RoundContext:
    """Everything a controller may use in round `round` (1-based). `evaluate` and `finetune` are None when the server
    has no validation set; controllers that need them raise."""
    round: int
    w_global: Weights
    trainable: List[bool]
    updates: List[ClientUpdate]
    evaluate: Optional[Callable[[Weights], Dict[str, float]]] = None   # metrics on V (selection part): loss, accuracy
    finetune: Optional[Callable[[Weights], Weights]] = None           # training on V (training part)
    num_malicious: int = 0                                            # Krum's f, when the experiment tells it
    _cache: dict = field(default_factory=dict, repr=False)

    def fedavg(self) -> Weights:
        """Sample-size weighted average of the client models (McMahan et al. 2017), computed once per round."""
        if "fedavg" not in self._cache:
            n = np.array([float(u.num_samples) for u in self.updates])
            p = n / n.sum()
            self._cache["fedavg"] = [sum(pk * u.weights[i] for pk, u in zip(p, self.updates)).astype(self.w_global[i].dtype)
                                     for i in range(len(self.w_global))]
        return self._cache["fedavg"]

    def pseudo_gradient(self) -> list:
        """Delta_t = FedAvg - w_t on the trainable weights (None for BatchNorm statistics)."""
        return [a - g if t else None for a, g, t in zip(self.fedavg(), self.w_global, self.trainable)]

    def require(self, what: str, who: str):
        fn = getattr(self, what)
        if fn is None:
            raise ValueError(f"{who} needs the server validation set ({what}); none was provided")
        return fn


def update_norm(delta: list) -> float:
    return float(np.sqrt(sum(float(np.sum(d * d)) for d in delta if d is not None)))


class Registry(dict):
    """name -> class, filled by the @register decorator of each plugin module."""

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind

    def register(self, *names):
        def deco(cls):
            for n in names:
                self[n] = cls
            cls.name = names[0]
            return cls
        return deco

    def get_class(self, name: str):
        key = (name or "").lower()
        if key not in self:
            raise ValueError(f"unknown {self.kind} {name!r}; available: {', '.join(sorted(self))}")
        return self[key]


class Rule:
    """A server aggregation rule. `advance` is called once per round for every rule the controller holds (whether or
    not its candidate is used), so stateful rules such as FedYogi never apply stale state; `propose` then returns the
    rule's candidate global model."""
    name = ""
    finetunes_per_round = 0      # training passes on V that propose() itself makes (FLTrust's server update)

    def advance(self, ctx: RoundContext) -> None:
        pass

    def propose(self, ctx: RoundContext) -> Weights:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class Controller:
    """Turns one round's client updates into the next global model. `step` returns (w_new, info); info holds at least
    `choice` (the rule kept) and the server cost of the decision (`server_evals`: models scored on V,
    `server_finetunes`: training passes on V)."""
    name = ""
    default_rules: Tuple[str, ...] = ("fedavg",)

    def __init__(self, rules: Dict[str, Rule]):
        self.rules = rules

    def step(self, ctx: RoundContext) -> Tuple[Weights, dict]:
        raise NotImplementedError

    def advance_rules(self, ctx: RoundContext) -> None:
        for rule in self.rules.values():
            rule.advance(ctx)

    def state_dict(self) -> dict:
        return {"rules": {n: r.state_dict() for n, r in self.rules.items()}}

    def load_state_dict(self, state: dict) -> None:
        for n, s in state.get("rules", {}).items():
            if n in self.rules:
                self.rules[n].load_state_dict(s)
