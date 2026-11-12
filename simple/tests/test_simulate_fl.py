"""simulate_fl.py (model, metrics, data split, every strategy end to end, resume) and compare_experiments.py, on tiny
synthetic client files (CPU, about a minute). The plugins themselves are unit-tested in distributed/tests."""
import contextlib
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import simulate_fl as sim
from plugins import build_controller, get_selector_by_name

SIMPLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_clients(root, n_clients=6, seed=0):
    """Synthetic ECG-shaped clients: AFib records carry a small offset on lead 0, so the task is learnable."""
    rng = np.random.default_rng(seed)
    for i in range(n_clients):
        def split(n):
            y = (rng.random(n) < 0.3).astype(np.float32)
            x = rng.normal(size=(n, 1000, 12)).astype(np.float32) * 0.1
            x[:, :, 0] += y[:, None]
            return x, y
        xtr, ytr = split(6 + 2 * i)
        xte, yte = split(6)
        np.savez(os.path.join(root, f"client_{i}.npz"), X_train=xtr, y_train=ytr, X_test=xte, y_test=yte)


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


class TestModel(unittest.TestCase):
    def test_trainable_mask_marks_batchnorm_statistics(self):
        model = sim.build_model()
        mask = sim.trainable_mask(model)
        self.assertEqual(len(mask), len(model.get_weights()))
        self.assertEqual(mask.count(False), 6)             # 3 BatchNorm layers x (moving mean, moving variance)

    def test_plain_sgd_training_and_oort_utility(self):
        model = sim.build_model()
        rng = np.random.default_rng(0)
        x = rng.normal(size=(10, 1000, 12)).astype(np.float32)
        y = np.array([0, 1] * 5, dtype=np.float32)
        w0 = model.get_weights()
        r = sim.Trainer(model, lr=0.0).train(x, y, epochs=1, batch_size=4, rng=rng)      # lr 0: weights stay
        for a, b, t in zip(w0, r.weights, sim.trainable_mask(model)):
            if t:
                np.testing.assert_array_equal(a, b)
        self.assertGreaterEqual(r.stat_utility + 1e-6, r.num_samples * r.train_loss)
        r = sim.Trainer(model, lr=0.05).train(x, y, epochs=2, batch_size=4, rng=rng)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(w0, r.weights)))

    def test_metrics(self):
        m = sim.binary_metrics(np.array([1, 1, 0, 0, 0]), np.array([0.9, 0.2, 0.4, 0.1, 0.0]))
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (1, 1, 1, 2))
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["f2"], 0.5)
        self.assertIsNone(sim.binary_metrics(np.zeros(3), np.full(3, 0.1))["auprc"])

    def test_seed_and_round_ranges(self):
        self.assertEqual(sim.parse_seeds("42-44,50"), [42, 43, 44, 50])
        self.assertEqual(len(sim.parse_seeds(sim.BASE_SEEDS)), 30)
        self.assertEqual(sim.parse_rounds("1-2,5"), {1, 2, 5})
        self.assertEqual(sim.parse_rounds(""), set())


class TestSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data_dir = os.path.join(cls.tmp.name, "clients")
        os.makedirs(cls.data_dir)
        write_clients(cls.data_dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def argv(self, out, *extra, rounds=2):
        return ["--data-dir", self.data_dir, "-k", "3", "-r", str(rounds), "-e", "1", "--batch-size", "4",
                "--seed", "7", "-o", out, *extra]

    def history(self, out):
        return pd.read_csv(os.path.join(out, "metrics.csv"))

    def test_holdout_is_disjoint_and_seeded(self):
        a = sim.build_federated_data(self.data_dir, seed=1)
        b = sim.build_federated_data(self.data_dir, seed=1)
        self.assertEqual(len(a.val_y) + len(a.test_y), 6 * 6)
        np.testing.assert_array_equal(a.val_x, b.val_x)
        self.assertEqual(len(sim.build_federated_data(self.data_dir, seed=1, val_size=5).val_y), 5)

    def test_every_strategy_runs(self):
        expected = {"fedavg": {"fedavg"}, "fedyogi": {"fedyogi"}, "median": {"median"}, "krum": {"krum"},
                    "fltrust": {"fltrust"},
                    "select": {"fedavg", "fedyogi"}, "train": {"fedavg"}, "select_train": {"fedavg", "fedyogi"},
                    "train_select": {"fedavg", "fedyogi", "fedavg+train", "fedyogi+train"},
                    "random_mix": {"fedavg", "fedyogi"}, "fedstrategist": {"fedavg", "median", "krum"},
                    "sara": {"fedavg", "median", "krum"}}
        for strategy, choices in expected.items():
            out = os.path.join(self.tmp.name, "all", strategy)
            extra = ["-a", strategy] + (["--select-frac", "0.3"] if strategy == "train_select" else [])
            quiet(sim.main, self.argv(out, *extra))
            df = self.history(out)
            self.assertEqual(df["round"].tolist(), [1, 2], strategy)
            self.assertTrue(set(df["choice"]) <= choices, strategy)
            self.assertTrue(os.path.exists(os.path.join(out, "summary.json")))
            self.assertFalse(os.path.exists(os.path.join(out, "checkpoint.pkl")))
            self.assertTrue(os.path.exists(os.path.join(out, "auprc_vs_round.png")))

    def test_seed_range_writes_subfolders_and_skips_finished(self):
        out = os.path.join(self.tmp.name, "range")
        argv = ["--data-dir", self.data_dir, "-k", "3", "-r", "1", "-e", "1", "--batch-size", "4", "-a", "fedavg",
                "--seeds", "1-2", "-o", out]
        quiet(sim.main, argv)
        self.assertTrue(all(os.path.exists(os.path.join(out, f"s{s}", "summary.json")) for s in (1, 2)))
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            sim.main(argv)
        self.assertEqual(log.getvalue().count("[Skip]"), 2)

    def test_random_mix_draws_a_schedule_per_seed(self):
        out = os.path.join(self.tmp.name, "mix")
        quiet(sim.main, ["--data-dir", self.data_dir, "-k", "2", "-r", "12", "-e", "1", "--batch-size", "4",
                         "-a", "random_mix", "--mix-p", "0.5", "--seeds", "1-3", "-o", out])
        schedules = {tuple(pd.read_csv(os.path.join(out, f"s{s}", "metrics.csv"))["choice"]) for s in (1, 2, 3)}
        self.assertGreater(len(schedules), 1)

    def test_poisoning_and_attack_rounds(self):
        out = os.path.join(self.tmp.name, "poison")
        quiet(sim.main, self.argv(out, "-a", "select", "--rules", "fedavg,median,krum", "--num-malicious", "6",
                                  "--attack", "standard", "--attack-rounds", "2", rounds=2))
        df = self.history(out)
        self.assertEqual(df["n_malicious"].tolist(), [0, 3])          # all 6 clients malicious; active in round 2 only

    def test_resume_from_checkpoint(self):
        out = os.path.join(self.tmp.name, "resume")
        quiet(sim.main, self.argv(out, "-a", "select_train", rounds=2))
        os.remove(os.path.join(out, "summary.json"))                 # pretend the 3-round run stopped after round 2
        args = sim.prepare_args(self.argv(out, "-a", "select_train", rounds=3))
        with open(os.path.join(out, "round_history.json")) as f:
            history = json.load(f)
        with open(os.path.join(out, "checkpoint.pkl"), "wb") as f:
            pickle.dump({"config": sim.run_config(args, 7), "round": 2, "weights": sim.build_model().get_weights(),
                         "selector": get_selector_by_name("oort", seed=0), "history": history,
                         "controller": build_controller("select_train").state_dict(),
                         "cum_latency": 1.0, "cum_energy": 2.0}, f)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            sim.main(self.argv(out, "-a", "select_train", rounds=3))
        self.assertIn("[Resume] continuing after round 2", log.getvalue())
        self.assertEqual(self.history(out)["round"].tolist(), [1, 2, 3])

    def test_compare_experiments(self):
        root = os.path.join(self.tmp.name, "cmp")
        for strategy in ("select", "fedavg"):
            quiet(sim.main, ["--data-dir", self.data_dir, "-k", "3", "-r", "2", "-e", "1", "--batch-size", "4",
                             "-a", strategy, "--seeds", "1-3", "-o", os.path.join(root, strategy)])
        out = os.path.join(root, "result")
        res = subprocess.run([sys.executable, os.path.join(SIMPLE, "compare_experiments.py"),
                              "--dirs", os.path.join(root, "select"), os.path.join(root, "fedavg", "s*"),
                              "--labels", "Selector", "FedAvg", "--best-of", "FedAvg", "-o", out],
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        summary = open(os.path.join(out, "summary.md"), encoding="utf-8").read()
        self.assertIn("| Selector | 3 |", summary)
        self.assertIn("| best of FedAvg | 3 |", summary)
        self.assertEqual(len(pd.read_csv(os.path.join(out, "per_seed.csv"))), 6)


if __name__ == "__main__":
    unittest.main()
