"""Plugin package (server/plugins): rules, controllers, bandits, attacks. numpy only, no TensorFlow."""
import filecmp
import os
import pickle
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))

from plugins import (CONTROLLERS, RULES, ClientUpdate, RoundContext, build_controller, build_rule,  # noqa: E402
                     craft_malicious_weights, parse_spec)
from plugins.bandits import CountUCB, LinUCB, OnlineStandardizer, state_vector  # noqa: E402


def make_round(t=1, n=4, seed=0, w_global=None, sizes=None):
    """A toy model: one trainable matrix, one trainable bias, one BatchNorm-like statistic."""
    rng = np.random.default_rng(seed)
    w_global = w_global or [rng.normal(size=(3, 2)).astype(np.float32), np.zeros(2, np.float32),
                            np.ones(2, np.float32)]
    sizes = sizes or [10, 20, 30, 40][:n]
    updates = [ClientUpdate(f"c{k}", [w + rng.normal(scale=0.1, size=w.shape).astype(np.float32) for w in w_global],
                            sizes[k]) for k in range(n)]
    return RoundContext(round=t, w_global=w_global, trainable=[True, True, False], updates=updates)


def weighted_mean(ctx):
    n = np.array([u.num_samples for u in ctx.updates], dtype=float)
    return [sum(p * u.weights[i] for p, u in zip(n / n.sum(), ctx.updates)) for i in range(len(ctx.w_global))]


class TestRules(unittest.TestCase):
    def test_fedavg_is_sample_weighted(self):
        ctx = make_round()
        for got, exp in zip(build_rule("fedavg").propose(ctx), weighted_mean(ctx)):
            np.testing.assert_allclose(got, exp, rtol=1e-6)

    def test_fedyogi_round1_matches_reddi(self):
        ctx = make_round()
        yogi = build_rule("fedyogi", eta=0.01, tau=1e-3)
        yogi.advance(ctx)
        out = yogi.propose(ctx)
        avg = weighted_mean(ctx)
        for got, wg, wa, t in zip(out, ctx.w_global, avg, ctx.trainable):
            if t:
                d = wa - wg
                m, v = 0.1 * d, 1e-6 - 0.01 * d * d * np.sign(1e-6 - d * d)
                np.testing.assert_allclose(got, wg + 0.01 * m / (np.sqrt(v) + 1e-3), rtol=1e-4, atol=1e-7)
            else:   # BatchNorm statistics take the FedAvg value
                np.testing.assert_allclose(got, wa, rtol=1e-6)

    def test_fedyogi_state_round_trip(self):
        a, b = build_rule("fedyogi"), build_rule("fedyogi")
        for t in (1, 2):
            ctx = make_round(t=t, seed=t)
            a.advance(ctx)
        b.load_state_dict(pickle.loads(pickle.dumps(a.state_dict())))
        ctx = make_round(t=3, seed=3)
        a.advance(ctx), b.advance(ctx)
        for x, y in zip(a.propose(ctx), b.propose(ctx)):
            np.testing.assert_array_equal(x, y)

    def test_median_is_lower_middle(self):
        ctx = make_round(n=4)
        out = build_rule("median").propose(ctx)
        stacked = np.sort(np.stack([u.weights[0] for u in ctx.updates]), axis=0)
        np.testing.assert_array_equal(out[0], stacked[1])

    def test_krum_drops_the_outlier_and_falls_back(self):
        ctx = make_round(n=6, sizes=[10] * 6)
        ctx.updates[5].weights = [w + 100.0 for w in ctx.updates[5].weights]
        krum = build_rule("krum", num_malicious=1)
        out = krum.propose(ctx)
        self.assertNotEqual(krum.last_pick, 5)
        np.testing.assert_array_equal(out[0], ctx.updates[krum.last_pick].weights[0])
        small = make_round(n=3)
        small.num_malicious = 1                     # n <= 2f + 2: FedAvg
        for got, exp in zip(build_rule("krum").propose(small), weighted_mean(small)):
            np.testing.assert_allclose(got, exp, rtol=1e-6)


    def test_fltrust_drops_opposed_updates_and_rescales(self):
        ctx = make_round(n=3, sizes=[10, 10, 10])
        d = [np.full_like(w, 0.1) if t else None for w, t in zip(ctx.w_global, ctx.trainable)]
        ctx.finetune = lambda w: [a + x if x is not None else a for a, x in zip(w, d)]       # server update g0 = d

        def shifted(k, bn):
            return [g + k * x if x is not None else g + bn for g, x in zip(ctx.w_global, d)]
        ctx.updates[0].weights = shifted(2.0, 0.0)      # agrees with g0, twice its norm
        ctx.updates[1].weights = shifted(0.5, 0.0)      # agrees, half its norm
        ctx.updates[2].weights = shifted(-5.0, 1.0)     # opposed and loud
        rule = build_rule("fltrust")
        out = rule.propose(ctx)
        self.assertEqual(rule.last_scores[2], 0.0)
        for got, g, x in zip(out[:2], ctx.w_global, d):  # both honest updates rescaled to |g0|: the step is g0
            np.testing.assert_allclose(got, g + x, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out[2], weighted_mean(ctx)[2], rtol=1e-6)   # BatchNorm statistic: FedAvg
        for u in ctx.updates:
            u.weights = shifted(-1.0, 0.0)
        ctx._cache.clear()
        for got, g in zip(build_rule("fltrust").propose(ctx)[:2], ctx.w_global):   # nobody trusted: keep w_t
            np.testing.assert_array_equal(got, g)
        _, info = build_controller("fltrust").step(ctx)
        self.assertEqual(info["server_finetunes"], 1)
        with self.assertRaises(ValueError):
            build_rule("fltrust").propose(make_round())                    # no server data


class TestControllers(unittest.TestCase):
    def test_spec_parsing(self):
        self.assertEqual(parse_spec("select_train"), ("select_train", None))
        self.assertEqual(parse_spec("fedyogi"), ("static", "fedyogi"))
        self.assertEqual(parse_spec("static:krum"), ("static", "krum"))
        with self.assertRaises(ValueError):
            parse_spec("qfedavg")
        self.assertEqual(set(CONTROLLERS) >= {"static", "select", "select_train", "train", "train_select",
                                              "fedstrategist"}, True)
        self.assertEqual(set(RULES), {"fedavg", "fedyogi", "median", "krum", "fltrust", "reject", "half", "normmatch"})

    def test_static(self):
        ctx = make_round()
        w, info = build_controller("fedavg").step(ctx)
        self.assertEqual(info["choice"], "fedavg")
        np.testing.assert_allclose(w[0], weighted_mean(ctx)[0], rtol=1e-6)

    def test_select_keeps_lower_loss_ties_to_fedavg(self):
        ctrl = build_controller("select")
        self.assertEqual(list(ctrl.rules), ["fedavg", "fedyogi"])
        ctx = make_round()
        target = ctrl.rules["fedyogi"]
        ctx.evaluate = lambda w: {"loss": 0.5}                       # tie -> fedavg
        _, info = ctrl.step(ctx)
        self.assertEqual(info["choice"], "fedavg")
        self.assertEqual(info["server_evals"], 2)
        ctx = make_round(t=2, seed=1)
        ctx.evaluate = lambda w: {"loss": float(np.abs(w[0]).sum())}   # FedYogi moves less: lower "loss"
        w, info = ctrl.step(ctx)
        lower = min(("fedavg", "fedyogi"), key=lambda n: info[f"val_loss_{n}"])
        self.assertEqual(info["choice"], lower)
        ctx = make_round(t=3, seed=2)
        ctx.evaluate = lambda w: {"loss": float("nan") if w is ctx.fedavg() else 1.0}
        self.assertEqual(ctrl.step(ctx)[1]["choice"], "fedyogi")
        self.assertIsNotNone(target.m)                                 # moments advanced every round

    def test_select_train_and_train_call_finetune_once(self):
        calls = []

        def finetune(w):
            calls.append(1)
            return [a + 1.0 for a in w]
        for spec, kept in (("select_train", None), ("train", "fedavg")):
            calls.clear()
            ctx = make_round()
            ctx.evaluate, ctx.finetune = (lambda w: {"loss": 1.0}), finetune
            w, info = build_controller(spec).step(ctx)
            self.assertEqual(len(calls), 1)
            self.assertEqual(info["server_finetunes"], 1)
            if kept:
                np.testing.assert_allclose(w[0], weighted_mean(ctx)[0] + 1.0, rtol=1e-6)

    def test_needs_validation(self):
        for spec in ("select", "select_train", "train", "train_select", "fedstrategist", "sara"):
            with self.assertRaises(ValueError):
                build_controller(spec).step(make_round())

    def test_train_select_scores_every_candidate(self):
        ctx = make_round()
        ctx.evaluate = lambda w: {"loss": float(np.abs(w[0] - 5.0).sum())}
        ctx.finetune = lambda w: [a + 5.0 for a in w]       # training moves toward 5: the trained candidates win
        w, info = build_controller("train_select").step(ctx)
        self.assertEqual(info["server_evals"], 4)
        self.assertTrue(info["choice"].endswith("+train"))

    def test_fedstrategist_presets_run_and_persist(self):
        for preset in ("paper", "authors"):
            ctrl = build_controller("fedstrategist", preset=preset, rule_args={"krum": {"num_malicious": 1}})
            self.assertEqual(list(ctrl.rules), ["fedavg", "median", "krum"])
            self.assertIsInstance(ctrl.bandit, LinUCB if preset == "paper" else CountUCB)
            choices = []
            for t in range(1, 6):
                ctx = make_round(t=t, n=6, seed=t, sizes=[10] * 6)
                ctx.evaluate = lambda w, t=t: {"accuracy": 0.5 + 0.01 * t, "loss": 1.0}
                _, info = ctrl.step(ctx)
                choices.append(info["choice"])
                self.assertIn("reward", info)
            again = build_controller("fedstrategist", preset=preset)
            again.load_state_dict(pickle.loads(pickle.dumps(ctrl.state_dict())))
            x = np.ones(again.scaler.out_dim if again.scaler else 3)
            self.assertEqual(again.bandit.select(x), ctrl.bandit.select(x))

    def test_fedstrategist_reward(self):
        ctrl = build_controller("fedstrategist", rules=["fedavg"], lambda_cost=0.5)
        accs = iter([0.40, 0.45])                     # initial model, then the new model
        ctx = make_round()
        ctx.evaluate = lambda w: {"accuracy": next(accs), "loss": 1.0}
        _, info = ctrl.step(ctx)
        self.assertAlmostEqual(info["reward"], 100 * 0.05 - 0.5 * 0.1)
        self.assertEqual(info["server_evals"], 2)

    def test_sara_ucb1_updates_only_the_applied_arm(self):
        ctrl = build_controller("sara", rule_args={"krum": {"num_malicious": 1}})
        self.assertEqual(ctrl.arms, ["fedavg", "median", "krum"])
        good = {"median": (0.9, 0.1)}                 # median earns reward 0.8, the others 0.5 - 1.0 = -0.5
        choices = []
        for t in range(1, 11):
            ctx = make_round(t=t, n=6, seed=t, sizes=[10] * 6)
            pending = {}

            def evaluate(w, ctx=ctx):
                acc, loss = good.get(pending["rule"], (0.5, 1.0))
                return {"accuracy": acc, "loss": loss}
            pending["rule"] = ctrl.arms[ctrl._pick()]
            ctx.evaluate = evaluate
            _, info = ctrl.step(ctx)
            choices.append(info["choice"])
            self.assertEqual(info["server_evals"], 1)
        self.assertEqual(choices[:3], ["fedavg", "median", "krum"])       # one pull of each arm, in order
        self.assertGreater(choices[3:].count("median"), 4)
        self.assertEqual(ctrl.counts.sum(), 10)
        np.testing.assert_allclose(ctrl.sums / ctrl.counts, [-0.5, 0.8, -0.5])
        again = build_controller("sara")
        again.load_state_dict(pickle.loads(pickle.dumps(ctrl.state_dict())))
        self.assertEqual(again._pick(), ctrl._pick())


class TestLoggingAndControls(unittest.TestCase):
    def test_select_logs_only_its_own_candidates(self):
        """Regression for the supplement's oracle bug (it logged every arm name, not only the arms it compared):
        a controller logs exactly the candidates it built."""
        for rules, expected in ((None, {"fedavg", "fedyogi"}), (["fedavg", "median", "krum"], {"fedavg", "median", "krum"}),
                                (["fedavg", "reject"], {"fedavg", "reject"})):
            ctrl = build_controller("select", rules=rules)
            ctx = make_round(n=6, sizes=[10] * 6)
            ctx.evaluate = lambda w: {"loss": 1.0}
            _, info = ctrl.step(ctx)
            logged = {k[len("val_loss_"):] for k in info if k.startswith("val_loss_")}
            self.assertEqual(logged, expected)
            self.assertEqual(info["server_evals"], len(expected))
        ctx = make_round()
        ctx.evaluate, ctx.finetune = (lambda w: {"loss": 1.0}), (lambda w: w)
        _, info = build_controller("train_select").step(ctx)
        self.assertEqual({k for k in info if k.startswith("val_loss_")},
                         {"val_loss_fedavg", "val_loss_fedyogi", "val_loss_fedavg+train", "val_loss_fedyogi+train"})

    def test_mechanism_controls(self):
        ctx = make_round()
        reject = build_rule("reject").propose(ctx)
        np.testing.assert_array_equal(reject[0], ctx.w_global[0])
        half = build_rule("half").propose(ctx)
        np.testing.assert_allclose(half[0], ctx.w_global[0] + 0.5 * (weighted_mean(ctx)[0] - ctx.w_global[0]), rtol=1e-5)
        yogi, nm = build_rule("fedyogi"), build_rule("normmatch")
        yogi.advance(ctx), nm.advance(ctx)
        wy, wn = yogi.propose(ctx), nm.propose(ctx)
        step = lambda w: np.sqrt(sum(float(np.sum((a - g) ** 2)) for a, g, t in zip(w, ctx.w_global, ctx.trainable) if t))
        self.assertAlmostEqual(step(wn), step(wy), places=5)          # FedYogi's step norm ...
        d_avg = weighted_mean(ctx)[0] - ctx.w_global[0]
        cos = float(np.sum((wn[0] - ctx.w_global[0]) * d_avg) / (np.linalg.norm(wn[0] - ctx.w_global[0]) * np.linalg.norm(d_avg)))
        self.assertAlmostEqual(cos, 1.0, places=5)                     # ... in FedAvg's direction

    def test_random_mix_is_data_blind_and_seeded(self):
        a = [build_controller("random_mix", p=0.3, seed=1).step(make_round(t=t))[1]["choice"] for t in range(1, 41)]
        b = [build_controller("random_mix", p=0.3, seed=1).step(make_round(t=t))[1]["choice"] for t in range(1, 41)]
        self.assertEqual(a, b)
        self.assertTrue(0 < a.count("fedyogi") < 40)


class TestBandits(unittest.TestCase):
    def test_state_vector_matches_definition(self):
        v = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        var, cos, mean_norm = state_vector(v)
        norms = np.linalg.norm(v, axis=1)
        self.assertAlmostEqual(var, float(np.var(norms, ddof=1)))
        self.assertAlmostEqual(cos, float(np.mean([0.0, 1 / np.sqrt(2), 1 / np.sqrt(2)])))
        self.assertAlmostEqual(mean_norm, float(np.linalg.norm(v.mean(0))))

    def test_linucb_learns_the_better_arm(self):
        bandit = LinUCB(2, 2, alpha=0.1)
        x = np.array([1.0, 0.0])
        for _ in range(20):
            a = bandit.select(x)
            bandit.update(a, x, 1.0 if a == 1 else 0.0)
        self.assertEqual(bandit.select(x), 1)

    def test_standardizer_appends_bias_and_clips(self):
        s = OnlineStandardizer(3)
        for _ in range(5):
            s([1.0, 2.0, 3.0])
        z = s([1e9, 2.0, 3.0])
        self.assertEqual(len(z), 4)
        self.assertEqual(z[-1], 1.0)
        self.assertLessEqual(z[0], 5.0)


class TestAttacks(unittest.TestCase):
    def setUp(self):
        self.g = [np.zeros((2, 2), np.float32), np.ones(2, np.float32)]
        self.l = [np.full((2, 2), 0.5, np.float32), np.full(2, 2.0, np.float32)]
        self.mask = [True, False]

    def test_standard_scales_trainable_delta_only(self):
        out = craft_malicious_weights(self.g, self.l, self.mask, "standard")
        np.testing.assert_allclose(out[0], 2.5)
        np.testing.assert_array_equal(out[1], self.l[1])          # BatchNorm statistic stays honest

    def test_stealth_and_flip(self):
        out = craft_malicious_weights(self.g, self.l, self.mask, "stealth", benign_avg_norm=2.0)
        self.assertAlmostEqual(float(np.linalg.norm(out[0])), 2.0, places=5)
        out = craft_malicious_weights(self.g, self.l, self.mask, "none", flip=True)
        np.testing.assert_allclose(out[0], -0.5)
        with self.assertRaises(ValueError):
            craft_malicious_weights(self.g, self.l, self.mask, "stealth")

    def test_client_copy_is_identical(self):
        self.assertTrue(filecmp.cmp(os.path.join(ROOT, "server", "plugins", "attacks.py"),
                                    os.path.join(ROOT, "client", "attacks.py"), shallow=False),
                        "client/attacks.py must be a byte-identical copy of server/plugins/attacks.py")


if __name__ == "__main__":
    unittest.main()
