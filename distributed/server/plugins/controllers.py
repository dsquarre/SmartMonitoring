"""Server controllers: which rule(s) to use each round, and whether to train on the server's validation set V.

    static:<rule>  always the same rule (fedavg, fedyogi, median, krum); the bare rule name works too
    select         build every rule's candidate from the same updates, keep the one with the lowest loss on V
                   (ties and non-finite losses go to the first rule): the TMLR supplement's "greedy-val" selector
    train          one rule (`base`), then training on V: the supplement's "server-ft" (Mai et al. 2023)
    select_train   SmartMonitoring's method: select as above, then train the kept model on V
    train_select   the hybrid the supplement leaves open (Sec. 5.6): candidates are every rule with and without
                   training on V, ranked on V. Use a selection split (select_frac > 0) so a trained candidate is not
                   scored on the data it was trained on
    fedstrategist  FedStrategist (Haque et al. 2025): a contextual bandit picks one rule per round from the update
                   diagnostics S_t and learns from  reward = 100 * (acc_V(w_t+1) - acc_V(w_t)) - lambda * cost(rule).
                   Presets follow the reproduction in project x: `authors` (CountUCB, S_t on full weights, raw
                   features) and `paper` (LinUCB, S_t on updates, standardised features). Both reward with accuracy on
                   V here; the authors' code used the test set, which no deployable server has
    sara           SARA (Hu et al., SEAMS 2025): UCB1 over the rules, updating only the applied rule's estimate,
                   reward = accuracy - loss of the new global model on V (reconstructed from the TMLR paper's
                   description; see the class)

Every controller advances every rule it holds each round (FedYogi's moments in particular), and reports its server
cost: `server_evals` (models scored on V) and `server_finetunes` (training passes over V).
"""
import math
import time
from typing import Dict, List, Sequence

import numpy as np

from .bandits import CountUCB, LinUCB, OnlineStandardizer, state_vector
from .base import Controller, Registry, RoundContext, Rule, update_norm
from .rules import build_rule

CONTROLLERS = Registry("server controller")


def _argmin_first(losses: Dict[str, float], order: Sequence[str]) -> str:
    finite = [n for n in order if math.isfinite(losses[n])]
    if not finite:
        return order[0]
    best = min(losses[n] for n in finite)
    return next(n for n in finite if losses[n] == best)


def _base_info(ctx: RoundContext, choice: str, evals: int = 0, finetunes: int = 0) -> dict:
    return {"choice": choice, "server_evals": evals, "server_finetunes": finetunes,
            "update_norm": update_norm(ctx.pseudo_gradient())}


@CONTROLLERS.register("static")
class Static(Controller):
    def __init__(self, rules: Dict[str, Rule], rule: str = None):
        super().__init__(rules)
        self.rule = rule or next(iter(rules))

    def step(self, ctx):
        self.advance_rules(ctx)
        return self.rules[self.rule].propose(ctx), _base_info(ctx, self.rule)


@CONTROLLERS.register("select")
class Select(Controller):
    default_rules = ("fedavg", "fedyogi")
    finetune_kept = False

    def step(self, ctx):
        evaluate = ctx.require("evaluate", self.name)
        self.advance_rules(ctx)
        order = list(self.rules)
        candidates = {n: r.propose(ctx) for n, r in self.rules.items()}
        losses = {n: float(evaluate(w)["loss"]) for n, w in candidates.items()}
        choice = _argmin_first(losses, order)
        w = candidates[choice]
        if self.finetune_kept:
            w = ctx.require("finetune", self.name)(w)
        info = _base_info(ctx, choice, evals=len(candidates), finetunes=int(self.finetune_kept))
        info.update({f"val_loss_{n}": v for n, v in losses.items()})
        return w, info


@CONTROLLERS.register("select_train")
class SelectTrain(Select):
    finetune_kept = True


@CONTROLLERS.register("train")
class Train(Controller):
    default_rules = ("fedavg", "fedyogi")

    def __init__(self, rules: Dict[str, Rule], base: str = "fedavg"):
        super().__init__(rules)
        if base not in rules:
            raise ValueError(f"train: base rule {base!r} is not among the rules {list(rules)}")
        self.base = base

    def step(self, ctx):
        finetune = ctx.require("finetune", self.name)
        self.advance_rules(ctx)
        return finetune(self.rules[self.base].propose(ctx)), _base_info(ctx, self.base, finetunes=1)


@CONTROLLERS.register("train_select")
class TrainSelect(Controller):
    default_rules = ("fedavg", "fedyogi")

    def step(self, ctx):
        evaluate = ctx.require("evaluate", self.name)
        finetune = ctx.require("finetune", self.name)
        self.advance_rules(ctx)
        candidates = {}
        for n, r in self.rules.items():
            w = r.propose(ctx)
            candidates[n] = w
            candidates[n + "+train"] = finetune(w)
        order = list(candidates)
        losses = {n: float(evaluate(w)["loss"]) for n, w in candidates.items()}
        choice = _argmin_first(losses, order)
        info = _base_info(ctx, choice, evals=len(candidates), finetunes=len(self.rules))
        info.update({f"val_loss_{n}": v for n, v in losses.items()})
        return candidates[choice], info


@CONTROLLERS.register("random_mix")
class RandomMix(Controller):
    """The supplement's control C4: each round commit the second rule with probability p, else the first, independent
    of any data (p = the selector's observed rate of keeping it). No validation set."""
    default_rules = ("fedavg", "fedyogi")

    def __init__(self, rules: Dict[str, Rule], p: float = 0.5, seed: int = 0):
        super().__init__(rules)
        if len(rules) != 2:
            raise ValueError("random_mix needs exactly two rules")
        self.p, self.seed = float(p), int(seed)

    def step(self, ctx):
        self.advance_rules(ctx)
        first, second = list(self.rules)
        u = np.random.default_rng(np.random.SeedSequence([self.seed, ctx.round, 0x313])).random()
        choice = second if u < self.p else first
        return self.rules[choice].propose(ctx), _base_info(ctx, choice)


@CONTROLLERS.register("fedstrategist")
class FedStrategist(Controller):
    default_rules = ("fedavg", "median", "krum")
    COSTS = {"fedavg": 0.1, "median": 0.4, "krum": 0.8}       # heuristic costs C_j (S1 Appendix)
    PRESETS = {"authors": dict(bandit="count_ucb", state_on="weights", standardize=False),
               "paper": dict(bandit="linucb", state_on="updates", standardize=True)}

    def __init__(self, rules: Dict[str, Rule], preset: str = "paper", lambda_cost: float = 0.5, alpha: float = 1.5,
                 costs: dict = None, reward_metric: str = "accuracy"):
        super().__init__(rules)
        if preset not in self.PRESETS:
            raise ValueError(f"fedstrategist preset must be one of {list(self.PRESETS)}")
        if reward_metric not in ("accuracy", "loss"):
            raise ValueError("reward_metric must be accuracy or loss")
        self.preset, self.lambda_cost, self.alpha, self.reward_metric = preset, float(lambda_cost), float(alpha), reward_metric
        self.costs = dict(self.COSTS, **(costs or {}))
        p = self.PRESETS[preset]
        self.arms: List[str] = list(rules)
        self.scaler = OnlineStandardizer(3) if p["standardize"] else None
        d = self.scaler.out_dim if self.scaler else 3
        self.bandit = (LinUCB if p["bandit"] == "linucb" else CountUCB)(len(self.arms), d, alpha=self.alpha)
        self.prev_metric = None

    def _metric(self, metrics: dict) -> float:
        # reward in percentage points of accuracy (as the reference), or loss decrease
        return 100.0 * metrics["accuracy"] if self.reward_metric == "accuracy" else -float(metrics["loss"])

    def step(self, ctx):
        evaluate = ctx.require("evaluate", self.name)
        t0 = time.perf_counter()
        self.advance_rules(ctx)
        on_weights = self.PRESETS[self.preset]["state_on"] == "weights"
        g = np.concatenate([a.ravel() for a in ctx.w_global]).astype(np.float64)
        vecs = np.stack([np.concatenate([a.ravel() for a in u.weights]).astype(np.float64) - (0.0 if on_weights else g)
                         for u in ctx.updates])
        raw = state_vector(vecs)
        x = self.scaler(raw) if self.scaler else np.asarray(raw)
        evals = 0
        if self.prev_metric is None:                          # the reference evaluates the initial model first
            self.prev_metric = self._metric(evaluate(ctx.w_global))
            evals += 1
        action = self.bandit.select(x)
        rule = self.arms[action]
        w = self.rules[rule].propose(ctx)
        cur = self._metric(evaluate(w))
        evals += 1
        reward = (cur - self.prev_metric) - self.lambda_cost * self.costs.get(rule, 0.0)
        self.bandit.update(action, x, reward)
        self.prev_metric = cur
        info = _base_info(ctx, rule, evals=evals)
        info.update(reward=reward, state_norm_var=raw[0], state_cosine_sim=raw[1], state_mean_norm=raw[2],
                    controller_s=time.perf_counter() - t0)
        return w, info

    def state_dict(self):
        return dict(super().state_dict(), bandit=self.bandit, scaler=self.scaler, prev_metric=self.prev_metric)

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.bandit = state.get("bandit", self.bandit)
        self.scaler = state.get("scaler", self.scaler)
        self.prev_metric = state.get("prev_metric")


@CONTROLLERS.register("sara")
class Sara(Controller):
    """SARA (Hu et al., SEAMS 2025) as the TMLR paper describes it: UCB over Byzantine-robust rules, updating only the
    estimate of the rule it applied, rewarded with the accuracy and loss of the resulting global model. Reconstructed
    from that description (no public code): UCB1, a_t = argmax mean_a + c * sqrt(2 ln t / n_a) after one pull of
    each arm in order, reward = acc_V(w_t+1) - loss_V(w_t+1). Scored on V; the paper's evaluation used the test set,
    which no deployable server has. One model scored on V per round."""
    default_rules = ("fedavg", "median", "krum")

    def __init__(self, rules: Dict[str, Rule], c: float = 1.0):
        super().__init__(rules)
        self.c = float(c)
        self.arms: List[str] = list(rules)
        self.counts = np.zeros(len(self.arms))
        self.sums = np.zeros(len(self.arms))

    def _pick(self) -> int:
        untried = np.flatnonzero(self.counts == 0)
        if untried.size:
            return int(untried[0])
        t = self.counts.sum() + 1
        return int(np.argmax(self.sums / self.counts + self.c * np.sqrt(2.0 * np.log(t) / self.counts)))

    def step(self, ctx):
        evaluate = ctx.require("evaluate", self.name)
        t0 = time.perf_counter()
        self.advance_rules(ctx)
        action = self._pick()
        rule = self.arms[action]
        w = self.rules[rule].propose(ctx)
        m = evaluate(w)
        reward = float(m["accuracy"]) - float(m["loss"])
        if not math.isfinite(reward):
            reward = -1e3
        self.counts[action] += 1
        self.sums[action] += reward
        info = _base_info(ctx, rule, evals=1)
        info.update(reward=reward, controller_s=time.perf_counter() - t0)
        return w, info

    def state_dict(self):
        return dict(super().state_dict(), counts=self.counts, sums=self.sums)

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.counts = state.get("counts", self.counts)
        self.sums = state.get("sums", self.sums)


def parse_spec(spec: str):
    """'select_train' | 'static:fedyogi' | 'fedyogi' (= static) | 'fedstrategist' ..."""
    spec = (spec or "").strip().lower()
    if ":" in spec:
        kind, arg = spec.split(":", 1)
        if kind != "static":
            raise ValueError(f"bad controller spec {spec!r}")
        return "static", arg
    if spec in CONTROLLERS:
        return spec, None
    from .rules import RULES
    if spec in RULES:
        return "static", spec
    raise ValueError(f"unknown server strategy {spec!r}; controllers: {', '.join(sorted(CONTROLLERS))}; "
                     f"rules: {', '.join(sorted(RULES))}")


def build_controller(spec: str, rules: Sequence[str] = None, rule_args: Dict[str, dict] = None, **kwargs) -> Controller:
    """Builds the controller for `spec` with its rules. `rules` defaults to the controller's own set (a static
    controller also holds FedYogi when asked for it); `rule_args` maps a rule name to its constructor arguments
    (e.g. {"fedyogi": {"eta": 0.01}, "krum": {"num_malicious": 2}}); `kwargs` go to the controller."""
    kind, arg = parse_spec(spec)
    cls = CONTROLLERS.get_class(kind)
    rule_args = rule_args or {}
    if kind == "static":
        names = list(rules) if rules else [arg]
        kwargs.setdefault("rule", arg)
    else:
        names = list(rules) if rules else list(cls.default_rules)
    built = {n: build_rule(n, **rule_args.get(n, {})) for n in names}
    return cls(built, **kwargs)
