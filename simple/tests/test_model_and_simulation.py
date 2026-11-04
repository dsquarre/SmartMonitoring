"""ECG model training / metrics and an end-to-end simulation of every plugin strategy on tiny synthetic client files
(CPU, ~1 minute). The plugins themselves are unit-tested in distributed/tests/test_plugins.py."""
import csv
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from smartmon.data import build_federated_data
from smartmon.ecg_model import Trainer, binary_metrics, build_model, trainable_mask
from smartmon.simulation import SimConfig, _save_checkpoint, run, run_stem
from plugins import build_controller, get_selector_by_name


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


class TestModel(unittest.TestCase):
    def test_trainable_mask_marks_batchnorm_statistics(self):
        mask = trainable_mask(build_model())
        self.assertEqual(len(mask), len(build_model().get_weights()))
        self.assertEqual(mask.count(False), 6)             # 3 BatchNorm layers x (moving mean, moving variance)

    def test_plain_sgd_training_and_oort_utility(self):
        model = build_model()
        rng = np.random.default_rng(0)
        x = rng.normal(size=(10, 1000, 12)).astype(np.float32)
        y = np.array([0, 1] * 5, dtype=np.float32)
        w0 = model.get_weights()
        r = Trainer(model, lr=0.0).train(x, y, epochs=1, batch_size=4, rng=rng)      # lr 0: weights stay
        for a, b, t in zip(w0, r.weights, trainable_mask(model)):
            if t:
                np.testing.assert_array_equal(a, b)
        self.assertGreaterEqual(r.stat_utility + 1e-6, r.num_samples * r.train_loss)
        r = Trainer(model, lr=0.05).train(x, y, epochs=2, batch_size=4, rng=rng)
        self.assertEqual(r.num_samples, 10)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(w0, r.weights)))

    def test_metrics(self):
        y = np.array([1, 1, 0, 0, 0])
        p = np.array([0.9, 0.2, 0.4, 0.1, 0.0])
        m = binary_metrics(y, p)
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (1, 1, 1, 2))
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["f2"], 0.5)
        self.assertIsNone(binary_metrics(np.zeros(3), np.full(3, 0.1))["auprc"])


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

    def cfg(self, strategy, rounds=2, **kw):
        return SimConfig(data_dir=self.data_dir, clients_per_round=3, rounds=rounds, local_epochs=1, batch_size=4,
                         seed=7, strategy=strategy, **kw)

    def test_holdout_is_disjoint_and_seeded(self):
        a = build_federated_data(self.data_dir, seed=1)
        b = build_federated_data(self.data_dir, seed=1)
        self.assertEqual(len(a.val_y) + len(a.test_y), 6 * 6)
        np.testing.assert_array_equal(a.val_x, b.val_x)
        self.assertEqual(len(build_federated_data(self.data_dir, seed=1, val_size=5).val_y), 5)

    def test_every_strategy_runs(self):
        out = os.path.join(self.tmp.name, "all")
        expected = {"fedavg": {"fedavg"}, "fedyogi": {"fedyogi"}, "median": {"median"}, "krum": {"krum"},
                    "select": {"fedavg", "fedyogi"}, "train": {"fedavg"}, "select_train": {"fedavg", "fedyogi"},
                    "train_select": {"fedavg", "fedyogi", "fedavg+train", "fedyogi+train"},
                    "fedstrategist": {"fedavg", "median", "krum"}}
        for strategy, choices in expected.items():
            cfg = self.cfg(strategy, select_frac=0.3 if strategy == "train_select" else 0.0)
            with open(run(cfg, out, verbose=False)) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([int(r["round"]) for r in rows], [1, 2], strategy)
            self.assertTrue(all(r["choice"] in choices for r in rows), strategy)
            self.assertTrue(all(float(r["server_s"]) >= 0 for r in rows))
            self.assertFalse(os.path.exists(run_stem(cfg, out) + ".ckpt.pkl"))

    def test_poisoning_run(self):
        out = os.path.join(self.tmp.name, "poison")
        cfg = self.cfg("select", rules=("fedavg", "median", "krum"), num_malicious=2, attack="standard",
                       name="select-robust")
        with open(run(cfg, out, verbose=False)) as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(os.path.basename(run_stem(cfg, out)).startswith("select-robust"))
        self.assertTrue(all(r["choice"] in ("fedavg", "median", "krum") for r in rows))
        self.assertIn("n_malicious", rows[0])

    def test_resume_from_checkpoint(self):
        out = os.path.join(self.tmp.name, "resume")
        cfg = self.cfg("select_train", rounds=3)
        stem = run_stem(cfg, out)
        # Simulate an interruption after round 2: run 2 rounds, then restore the partial files of a 3-round config.
        run(self.cfg("select_train", rounds=2), out, verbose=False)
        os.replace(stem + ".csv", stem + ".partial.csv")
        controller = build_controller("select_train")
        selector = get_selector_by_name("oort", seed=0)
        _save_checkpoint(stem + ".ckpt.pkl", {"config": cfg.to_dict(), "round": 2,
                                              "weights": build_model().get_weights(), "selector": selector,
                                              "controller": controller.state_dict(), "cum_latency": 1.0,
                                              "cum_energy": 2.0})
        with open(run(cfg, out, verbose=False)) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([int(r["round"]) for r in rows], [1, 2, 3])
        self.assertGreater(float(rows[-1]["cum_latency"]), 1.0)


if __name__ == "__main__":
    unittest.main()
