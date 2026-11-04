"""Server aggregation rules.

    fedavg   sample-size weighted average (McMahan et al. 2017)
    fedyogi  adaptive server step on the FedAvg pseudo-gradient (Reddi et al., ICLR 2021; Alg. 2, no bias correction):
                 m <- b1 m + (1-b1) D,  v <- v - (1-b2) D^2 sign(v - D^2),  w <- w + eta m / (sqrt(v) + tau)
             with m0 = 0, v0 = tau^2, as in the TMLR supplement (hrl_fl/aggregators.py). The moments advance every
             round from the FedAvg pseudo-gradient, whichever rule is kept. Applied to the trainable weights; BatchNorm
             statistics take the FedAvg values (the supplement's CNN has no BatchNorm).
    median   coordinate-wise median; the lower middle value for an even number of clients, as torch.median in the
             FedStrategist reference code
    krum     Blanchard et al. 2017: the client model with the smallest sum of squared distances to its n - f - 2
             nearest neighbours; falls back to FedAvg when n <= 2f + 2 (as in the FedStrategist reproduction).
             f = the rule's `num_malicious`, else the round's RoundContext.num_malicious

The FedStrategist reference averages updates without sample weights; here every rule uses the system's sample-weighted
FedAvg where it averages, so the rules differ only in what they are meant to differ in.
"""
import numpy as np

from .base import Registry, Rule, RoundContext, Weights

RULES = Registry("server rule")


@RULES.register("fedavg")
class FedAvg(Rule):
    def propose(self, ctx: RoundContext) -> Weights:
        return ctx.fedavg()


@RULES.register("fedyogi")
class FedYogi(Rule):
    def __init__(self, eta: float = 0.01, tau: float = 1e-3, b1: float = 0.9, b2: float = 0.99):
        self.eta, self.tau, self.b1, self.b2 = float(eta), float(tau), float(b1), float(b2)
        self.m = self.v = None

    def advance(self, ctx: RoundContext) -> None:
        if self.m is None:
            self.m = [np.zeros_like(w) if t else None for w, t in zip(ctx.w_global, ctx.trainable)]
            self.v = [np.full_like(w, self.tau ** 2) if t else None for w, t in zip(ctx.w_global, ctx.trainable)]
        for i, d in enumerate(ctx.pseudo_gradient()):
            if d is not None:
                d2 = d * d
                self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * d
                self.v[i] = self.v[i] - (1 - self.b2) * d2 * np.sign(self.v[i] - d2)

    def propose(self, ctx: RoundContext) -> Weights:
        if self.m is None:
            raise RuntimeError("FedYogi.propose before advance")
        return [wa if m is None else (wg + self.eta * m / (np.sqrt(v) + self.tau)).astype(wg.dtype)
                for wg, wa, m, v in zip(ctx.w_global, ctx.fedavg(), self.m, self.v)]

    def state_dict(self) -> dict:
        return {"m": self.m, "v": self.v}

    def load_state_dict(self, state: dict) -> None:
        self.m, self.v = state.get("m"), state.get("v")


@RULES.register("median")
class Median(Rule):
    def propose(self, ctx: RoundContext) -> Weights:
        n = len(ctx.updates)
        return [np.sort(np.stack([u.weights[i] for u in ctx.updates]), axis=0)[(n - 1) // 2].astype(w.dtype)
                for i, w in enumerate(ctx.w_global)]


@RULES.register("krum")
class Krum(Rule):
    def __init__(self, num_malicious: int = None):
        self.num_malicious = num_malicious
        self.last_pick = None

    def propose(self, ctx: RoundContext) -> Weights:
        f = int(ctx.num_malicious if self.num_malicious is None else self.num_malicious)
        n = len(ctx.updates)
        if n <= 2 * f + 2:
            self.last_pick = None
            return ctx.fedavg()
        flat = np.stack([np.concatenate([a.ravel().astype(np.float64) for a in u.weights]) for u in ctx.updates])
        dists = np.zeros((n, n))
        for i in range(n):                  # exact squared distances (no |a|^2 + |b|^2 - 2ab cancellation)
            for j in range(i + 1, n):
                d = flat[i] - flat[j]
                dists[i, j] = dists[j, i] = float(d @ d)
        k = n - f - 2
        scores = np.sort(dists, axis=1)[:, 1:k + 1].sum(axis=1)   # column 0 is the distance to itself
        self.last_pick = int(np.argmin(scores))
        return [a.copy() for a in ctx.updates[self.last_pick].weights]


# -- mechanism controls of the TMLR supplement (Sec. 4.4, C1-C3): alternatives to the FedYogi candidate ----------
@RULES.register("reject")
class Reject(Rule):
    """C1: keep w_t (reject the round)."""

    def propose(self, ctx: RoundContext) -> Weights:
        return [w.copy() for w in ctx.w_global]


@RULES.register("half")
class Half(Rule):
    """C2: w_t + 0.5 * Delta_FedAvg on the trainable weights (BatchNorm statistics: FedAvg)."""

    def propose(self, ctx: RoundContext) -> Weights:
        return [wa if d is None else (wg + 0.5 * d).astype(wg.dtype)
                for wg, wa, d in zip(ctx.w_global, ctx.fedavg(), ctx.pseudo_gradient())]


@RULES.register("normmatch")
class NormMatch(Rule):
    """C3: FedAvg's direction at FedYogi's step norm for this round. Holds its own FedYogi, advanced every round from
    the same pseudo-gradients, so its moments equal those of a FedYogi rule with the same arguments."""

    def __init__(self, **yogi_args):
        self.yogi = FedYogi(**yogi_args)

    def advance(self, ctx: RoundContext) -> None:
        self.yogi.advance(ctx)

    def propose(self, ctx: RoundContext) -> Weights:
        delta = ctx.pseudo_gradient()
        w_yogi = self.yogi.propose(ctx)
        yogi_step = np.sqrt(sum(float(np.sum((a - g) ** 2)) for a, g, d in zip(w_yogi, ctx.w_global, delta)
                                if d is not None))
        avg_step = np.sqrt(sum(float(np.sum(d * d)) for d in delta if d is not None))
        scale = yogi_step / max(avg_step, 1e-12)
        return [wa if d is None else (wg + scale * d).astype(wg.dtype)
                for wg, wa, d in zip(ctx.w_global, ctx.fedavg(), delta)]

    def state_dict(self) -> dict:
        return self.yogi.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.yogi.load_state_dict(state)


def build_rule(name: str, **kwargs) -> Rule:
    return RULES.get_class(name)(**kwargs)
