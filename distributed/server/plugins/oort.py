"""Oort client selector (Lai et al., OSDI '21), ported from SymbioticLab/Oort `oort/oort.py` `_training_selector`.

This is the same port as the TMLR supplement (hrl_fl/oort.py), with its defaults from FedScale's config_parser.py.
Kept faithful to the reference, including its quirks:
  * the temporal-uncertainty bonus divides by the client's last-participation round (`time_stamp`),
    not by rounds-since-participation;
  * the pacer compares sums of per-round *mean* exploit utilities over two `pacer_step` windows;
  * exploration stops permanently once every client has been tried.
Departures, all needed to make it run standalone and reproducibly:
  * RNGs are per-instance (seeded) instead of global `np.random`;
  * guards for the reference's IndexError when exploitLen == number of explored clients, and for
    weighted sampling with fewer non-zero weights than requested;
  * `num` is capped at the number of feasible clients (a live server can have fewer connected clients than K).

Client ids can be any hashable (ints in the simulation, "client_3" strings in the distributed server).
numpy-only; used by both the distributed server and the simple simulator through plugins/selectors.py.
"""
import math
import random
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


@dataclass
class OortConfig:
    # Defaults follow FedScale's config_parser.py (the Oort authors' framework).
    exploration_factor: float = 0.9
    exploration_decay: float = 0.98
    exploration_min: float = 0.3
    round_threshold: float = 30.0   # percentile of client durations used as T_target
    round_penalty: float = 2.0      # beta in (T_target / T_i)^beta
    clip_bound: float = 0.9
    blacklist_rounds: int = -1      # -1 disables blacklisting
    blacklist_max_len: float = 0.3
    sample_window: float = 5.0
    pacer_step: int = 20
    pacer_delta: float = 5.0
    cut_off_util: float = 0.05


class OortSelector:
    def __init__(self, cfg: OortConfig = None, seed: int = 0):
        self.cfg = cfg or OortConfig()
        self.arms = OrderedDict()
        self.training_round = 0
        self.exploration = self.cfg.exploration_factor
        self.exploration_min = self.cfg.exploration_min
        self.round_threshold = self.cfg.round_threshold
        self.round_prefer_duration = float("inf")
        self.unexplored = set()
        self.successful = set()
        self.exploit_clients, self.explore_clients = [], []
        self.exploit_util_history, self.explore_util_history = [], []
        self.blacklist = set()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    # -- feedback ---------------------------------------------------------------------------
    def register_client(self, cid, reward: float, duration: float) -> None:
        """FedScale registers clients with reward = data size and their expected duration."""
        if cid not in self.arms:
            self.arms[cid] = {"reward": reward, "duration": duration, "time_stamp": self.training_round,
                              "count": 0, "status": True}
            self.unexplored.add(cid)

    def update_client_util(self, cid, reward: float, duration: float, time_stamp: int) -> None:
        """reward = statistical utility |B_i| * sqrt(mean per-sample loss^2) from the client's last training."""
        arm = self.arms[cid]
        arm["reward"], arm["duration"], arm["time_stamp"] = reward, duration, time_stamp
        arm["count"] += 1
        arm["status"] = True
        self.unexplored.discard(cid)
        self.successful.add(cid)

    # -- internals --------------------------------------------------------------------------
    def _mean_util(self, clients) -> float:
        cnt, util = 1e-4, 0.0
        for c in clients:
            if c in self.successful:
                cnt += 1
                util += self.arms[c]["reward"]
        return util / cnt

    def _pacer(self) -> None:
        self.explore_util_history.append(self._mean_util(self.explore_clients))
        self.exploit_util_history.append(self._mean_util(self.exploit_clients))
        self.successful = set()
        step = self.cfg.pacer_step
        if self.training_round >= 2 * step and self.training_round % step == 0:
            last = sum(self.exploit_util_history[-2 * step:-step])
            cur = sum(self.exploit_util_history[-step:])
            if abs(cur - last) <= last * 0.1:
                self.round_threshold = min(100.0, self.round_threshold + self.cfg.pacer_delta)
            elif abs(cur - last) >= last * 5:
                self.round_threshold = max(self.cfg.pacer_delta, self.round_threshold - self.cfg.pacer_delta)

    def _get_blacklist(self) -> set:
        if self.cfg.blacklist_rounds == -1:
            return set()
        ranked = sorted(self.arms, key=lambda c: self.arms[c]["count"], reverse=True)
        cap = int(self.cfg.blacklist_max_len * len(self.arms))
        return {c for c in ranked[:cap] if self.arms[c]["count"] > self.cfg.blacklist_rounds}

    @staticmethod
    def _get_norm(values, clip_bound=0.95, thres=1e-4):
        values = sorted(values)
        if not values:
            return 0.0, 0.0, thres, 0.0, float("inf")
        clip_value = values[min(int(len(values) * clip_bound), len(values) - 1)]
        vmax, vmin = values[-1], values[0] * 0.999
        return vmax, vmin, max(vmax - vmin, thres), sum(values) / max(1e-4, len(values)), clip_value

    def _penalise(self, score: float, duration: float) -> float:
        if duration > self.round_prefer_duration:
            score *= (self.round_prefer_duration / max(1e-4, duration)) ** self.cfg.round_penalty
        return score

    def _weighted_sample(self, pool, k, weights):
        w = np.asarray(weights, dtype=float)
        if k <= 0:
            return []
        if w.sum() <= 0 or np.count_nonzero(w) < k:
            w = w + 1e-12
        return [pool[i] for i in self.np_rng.choice(len(pool), size=k, replace=False, p=w / w.sum())]

    # -- selection --------------------------------------------------------------------------
    def select_participants(self, num: int, cur_time: int, feasible=None) -> list:
        """`cur_time` is the 1-based round index (the reference calls getTopK with training_round + 1).
        `feasible`: ids that may be picked this round (e.g. the connected clients); None = all registered."""
        self.training_round = cur_time
        self.blacklist = self._get_blacklist()
        self._pacer()

        feasible = set(self.arms) if feasible is None else set(feasible) & set(self.arms)
        ordered = [c for c in self.arms if c in feasible and c not in self.blacklist]
        num = min(num, len(ordered))

        if self.round_threshold < 100.0:
            durations = sorted(a["duration"] for a in self.arms.values())
            idx = min(int(len(durations) * self.round_threshold / 100.0), len(durations) - 1)
            self.round_prefer_duration = durations[idx]
        else:
            self.round_prefer_duration = float("inf")

        moving_reward = [self.arms[c]["reward"] for c in ordered if self.arms[c]["reward"] > 0]
        _, min_r, range_r, _, clip_value = self._get_norm(moving_reward, self.cfg.clip_bound)

        scores = {}
        for c in ordered:
            arm = self.arms[c]
            if arm["count"] > 0:
                creward = min(arm["reward"], clip_value)
                sc = (creward - min_r) / range_r + math.sqrt(0.1 * math.log(cur_time) / arm["time_stamp"])
                scores[c] = abs(self._penalise(sc, arm["duration"]))

        self.exploration = max(self.exploration * self.cfg.exploration_decay, self.exploration_min)
        exploit_len = min(int(num * (1.0 - self.exploration)), len(scores))

        picked = []
        if exploit_len > 0:
            ranked = sorted(scores, key=scores.get, reverse=True)
            cut_off = scores[ranked[min(exploit_len, len(ranked) - 1)]] * self.cfg.cut_off_util
            pool = []
            for c in ranked:
                if scores[c] < cut_off:
                    break
                pool.append(c)
            picked = self._weighted_sample(pool, exploit_len, [scores[c] for c in pool])
        self.exploit_clients = list(picked)

        unexplored = [c for c in self.arms if c in self.unexplored and c in feasible]
        if unexplored:
            init_reward = {c: self._penalise(self.arms[c]["reward"], self.arms[c]["duration"]) for c in unexplored}
            explore_len = min(len(unexplored), num - len(picked))
            window = min(int(self.cfg.sample_window * explore_len), len(init_reward))
            pool = sorted(init_reward, key=init_reward.get, reverse=True)[:window]
            chosen = self._weighted_sample(pool, explore_len, [init_reward[c] for c in pool])
            self.explore_clients = chosen
            picked = picked + chosen
        else:
            self.exploration_min = 0.0
            self.exploration = 0.0
            self.explore_clients = []

        while len(picked) < num:
            c = self.rng.choice(ordered)
            if c not in picked:
                picked.append(c)
        return picked
