"""Client selectors.

    oort    Oort guided participant selection (Lai et al., OSDI '21), the TMLR supplement's port (oort.py) with
            FedScale defaults. Clients are registered with reward = training-set size and their expected duration
            (`client_durations` in the context if the host knows it, as the simulator does; otherwise the median
            measured duration so far). After each round every client that reported back is updated with its
            statistical utility |B| * sqrt(mean per-sample training loss^2) and its duration.
    random  uniform sampling without replacement

Interface (the one server/main.py has always used):
    select_clients(client_ids, k, context)  context: round (1-based), client_samples, optional client_durations
    update_policy(round_summary)            round, selected_ids, client_utilities, client_roundtrips
"""
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from .base import Registry
from .oort import OortConfig, OortSelector

SELECTORS = Registry("client selector")


class ClientSelector(ABC):
    name = ""

    @abstractmethod
    def select_clients(self, client_ids: List[str], k: int, context: Dict[str, Any] = None) -> List[str]:
        """Select k of the available clients."""

    def update_policy(self, round_summary: Dict[str, Any]):
        """Feedback after the round (utility, duration)."""
        pass


@SELECTORS.register("random")
class RandomClientSelector(ClientSelector):
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)

    def select_clients(self, client_ids, k, context=None):
        if not client_ids:
            return []
        return self.rng.sample(list(client_ids), min(k, len(client_ids)))


@SELECTORS.register("oort")
class OortClientSelector(ClientSelector):
    def __init__(self, cfg: OortConfig = None, seed: int = None):
        if seed is None:
            seed = int(os.environ.get("FL_SEED", "0"))
        self.oort = OortSelector(cfg or OortConfig(), seed=seed)

    def _register(self, client_ids, client_samples: dict, client_durations: dict):
        measured = [a["duration"] for a in self.oort.arms.values() if a["count"] > 0]
        guess = float(np.median(measured)) if measured else 1.0
        for cid in client_ids:
            if cid not in self.oort.arms:
                self.oort.register_client(cid, reward=float(client_samples.get(cid, 1.0)),
                                          duration=float(client_durations.get(cid, guess)))
        if not client_durations:
            for arm in self.oort.arms.values():   # unexplored clients: expected duration = median measured duration
                if arm["count"] == 0:
                    arm["duration"] = guess

    def select_clients(self, client_ids, k, context=None):
        if not client_ids:
            return []
        context = context or {}
        self._register(client_ids, context.get("client_samples", {}), context.get("client_durations", {}))
        return list(self.oort.select_participants(min(k, len(client_ids)), cur_time=int(context.get("round", 1)),
                                                  feasible=list(client_ids)))

    def update_policy(self, round_summary):
        utilities = round_summary.get("client_utilities", {})
        durations = round_summary.get("client_roundtrips", {})
        for cid in round_summary.get("selected_ids", []):
            if cid in utilities and cid in self.oort.arms:
                self.oort.update_client_util(cid, reward=float(utilities[cid]),
                                             duration=float(durations.get(cid, round_summary.get("elapsed_round", 1.0))),
                                             time_stamp=int(round_summary["round"]))


def get_selector_by_name(name: str, **kwargs) -> ClientSelector:
    return SELECTORS.get_class(name or "oort")(**kwargs)
