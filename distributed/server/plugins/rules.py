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
    fltrust  Cao et al., NDSS 2021: client updates weighted by their cosine agreement with the server's own update
             on V and rescaled to its norm (needs the server validation set)

The FedStrategist reference averages updates without sample weights; here every rule uses the system's sample-weighted
FedAvg where it averages, so the rules differ only in what they are meant to differ in.
"""
import numpy as np

from .base import Registry, Rule, RoundContext, Weights, update_norm

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


@RULES.register("fltrust")
class FLTrust(Rule):
    """FLTrust (Cao et al., NDSS 2021). The server trains the current global model on its own clean data (the host's
    `finetune` on V, the paper's root dataset) to get a server update g0. Client update g_i (trainable weights) gets the
    trust score TS_i = ReLU(cos(g_i, g0)) and is rescaled to the norm of g0; the new model is
        w_t + sum_i TS_i * (|g0| / |g_i|) * g_i / sum_i TS_i            (global learning rate 1)
    If every score is 0 the round is rejected (w_t kept). BatchNorm statistics take the FedAvg values.
    Departures from the paper: the root set is V (2,198 records here, the paper uses 100), and g0 comes from the same
    server-training pass the `train` controllers use (ft_epochs of plain SGD on V) instead of exactly the clients'
    number of local iterations."""
    finetunes_per_round = 1

    def __init__(self):
        self.last_scores = None

    def propose(self, ctx: RoundContext) -> Weights:
        finetune = ctx.require("finetune", self.name)
        w0 = finetune([w.copy() for w in ctx.w_global])
        g0 = [(a - g).astype(np.float64) if t else None for a, g, t in zip(w0, ctx.w_global, ctx.trainable)]
        n0 = update_norm(g0)
        scores, steps = [], []
        for u in ctx.updates:
            gi = [(a - g).astype(np.float64) if t else None for a, g, t in zip(u.weights, ctx.w_global, ctx.trainable)]
            ni = update_norm(gi)
            dot = sum(float(np.sum(a * b)) for a, b in zip(gi, g0) if a is not None)
            cos = dot / (ni * n0) if ni > 0 and n0 > 0 else 0.0
            scores.append(max(cos, 0.0))
            steps.append((n0 / ni if ni > 0 else 0.0, gi))
        self.last_scores = scores
        total = sum(scores)
        out = []
        for i, (wg, wa, t) in enumerate(zip(ctx.w_global, ctx.fedavg(), ctx.trainable)):
            if not t:
                out.append(wa)
            elif total <= 0:
                out.append(wg.copy())
            else:
                step = sum(ts * scale * gi[i] for ts, (scale, gi) in zip(scores, steps)) / total
                out.append((wg + step).astype(wg.dtype))
        return out


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
