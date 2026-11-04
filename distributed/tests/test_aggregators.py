"""The Celery-side adapter (server/aggregator.py): .keras files -> plugin controller -> new global model, with the
controller state carried between rounds through storage (the mock bucket here) and the server validation set V."""
import sys
import os
import shutil
import tempfile
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

import s3_helper
from model import Model
from aggregator import PluginAggregator, bce, get_aggregator_by_name, trainable_mask

TMP = tempfile.mkdtemp(prefix="sm_aggregators_")
BUCKET = os.path.join(TMP, "bucket")


class TestPluginAggregator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The server state goes through s3_helper: use a local mock bucket for this class only
        cls._s3 = (s3_helper.S3_MOCK, s3_helper.MOCK_S3_DIR)
        s3_helper.S3_MOCK, s3_helper.MOCK_S3_DIR = True, BUCKET
        rng = np.random.default_rng(0)
        cls.val_path = os.path.join(TMP, "server_val.npz")
        np.savez(cls.val_path, X=rng.normal(size=(24, 1000, 12)).astype(np.float32),
                 y=np.array([1, 0, 0, 0] * 6, dtype=np.float32))

    @classmethod
    def tearDownClass(cls):
        s3_helper.S3_MOCK, s3_helper.MOCK_S3_DIR = cls._s3
        shutil.rmtree(TMP, ignore_errors=True)

    def setUp(self):
        self.tmp_dir = os.path.join(TMP, "models")
        os.makedirs(self.tmp_dir, exist_ok=True)
        shutil.rmtree(BUCKET, ignore_errors=True)
        self.global_path = os.path.join(self.tmp_dir, "global_model.keras")
        base = Model()
        base.model.save(self.global_path)
        self.w0 = base.model.get_weights()
        self.trainable = trainable_mask(base.model)

        # 3 client models, perturbed per client; samples 1000 / 2000 / 1000
        self.client_data, self.client_weights = [], []
        for i in range(3):
            cp = os.path.join(self.tmp_dir, f"client_{i}.keras")
            cm = Model()
            w = [layer + 0.01 * (i + 1) for layer in self.w0]
            cm.model.set_weights(w)
            cm.model.save(cp)
            self.client_weights.append(w)
            self.client_data.append((cp, 1000 * (2 if i == 1 else 1), 0.5, f"client_{i}"))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def agg(self, strategy, **kw):
        kw.setdefault("val_path", self.val_path)
        return PluginAggregator(strategy=strategy, **kw)

    def _load_global(self):
        m = Model()
        m.model.load_weights(self.global_path)
        return m.model.get_weights()

    def _fedavg(self):
        return [(w0 + 2 * w1 + w2) / 4 for w0, w1, w2 in zip(*self.client_weights)]

    def test_factory_and_celery_round_trip(self):
        for name in ("select_train", "select", "train", "train_select", "fedstrategist", "fedavg", "fedyogi",
                     "median", "krum", "static:fedyogi"):
            agg = get_aggregator_by_name(name)
            self.assertIsInstance(agg, PluginAggregator)
            again = get_aggregator_by_name(agg.__class__.__name__, **agg.config())   # what Celery receives
            self.assertEqual(again.config(), agg.config())
        for removed in ("qfedavg", "fedfv", "fedadam", "fedprox", "scaffold"):
            with self.assertRaises(ValueError):
                get_aggregator_by_name(removed)

    def test_fedavg_is_sample_weighted_mean(self):
        agg = self.agg("fedavg")
        agg.aggregate(self.client_data, self.global_path, current_round=1)
        self.assertEqual(agg.last_info["server_choice"], "fedavg")
        for got, exp in zip(self._load_global(), self._fedavg()):
            np.testing.assert_allclose(got, exp, rtol=1e-5, atol=1e-6)

    def test_fedyogi_matches_reddi_step(self):
        """Round 1 from the initial state: m = 0.1 d, v = tau^2 - 0.01 d^2 sign(tau^2 - d^2)."""
        eta, tau = 0.01, 1e-3
        self.agg("fedyogi", rule_args={"fedyogi": {"eta": eta, "tau": tau}}).aggregate(
            self.client_data, self.global_path, current_round=1)
        for got, w0, wa, t in zip(self._load_global(), self.w0, self._fedavg(), self.trainable):
            if t:
                d = wa - w0
                m, v = 0.1 * d, tau ** 2 - 0.01 * d * d * np.sign(tau ** 2 - d * d)
                np.testing.assert_allclose(got, w0 + eta * m / (np.sqrt(v) + tau), rtol=1e-4, atol=1e-6)
            else:   # BatchNorm moving statistics take the FedAvg values
                np.testing.assert_allclose(got, wa, rtol=1e-5, atol=1e-6)
        self.assertTrue(os.path.exists(os.path.join(BUCKET, "models/global/server_state_1.pkl")))

    def test_select_keeps_lower_validation_loss(self):
        agg = self.agg("select")
        agg.aggregate(self.client_data, self.global_path, current_round=1)
        info = agg.last_info
        lower = "fedyogi" if info["val_loss_fedyogi"] < info["val_loss_fedavg"] else "fedavg"
        self.assertEqual(info["server_choice"], lower)
        self.assertEqual(info["server_evals"], 2)

    def test_select_train_two_rounds_carries_state(self):
        agg = self.agg("select_train")
        agg.aggregate(self.client_data, self.global_path, current_round=1)
        self.assertEqual(agg.last_info["server_finetunes"], 1)
        out = self._load_global()           # training on V moved the model away from FedAvg
        self.assertGreater(max(float(np.abs(a - b).max()) for a, b in zip(out, self._fedavg())), 0.0)
        agg2 = get_aggregator_by_name("PluginAggregator", **agg.config())
        agg2.aggregate(self.client_data, self.global_path, current_round=2)
        self.assertTrue(os.path.exists(os.path.join(BUCKET, "models/global/server_state_2.pkl")))

    def test_server_training_is_plain_sgd_at_client_lr(self):
        """server-ft: SGD at client_lr * ft_lr_scale. Scale 0 leaves the trainable weights at FedAvg."""
        self.agg("train", client_lr=0.05, ft_lr_scale=0.0, batch_size=8).aggregate(
            self.client_data, self.global_path, current_round=1)
        for got, wa, t in zip(self._load_global(), self._fedavg(), self.trainable):
            if t:
                np.testing.assert_allclose(got, wa, rtol=1e-5, atol=1e-6)

    def test_fedstrategist_and_robust_rules(self):
        for strategy in ("median", "krum"):
            agg = self.agg(strategy, num_malicious=0)
            agg.aggregate(self.client_data, self.global_path, current_round=1)
            self.assertEqual(agg.last_info["server_choice"], strategy)
        agg = self.agg("fedstrategist", controller_args={"preset": "paper"})
        agg.aggregate(self.client_data, self.global_path, current_round=1)
        self.assertIn(agg.last_info["server_choice"], ("fedavg", "median", "krum"))
        self.assertIn("reward", agg.last_info)
        get_aggregator_by_name("PluginAggregator", **agg.config()).aggregate(
            self.client_data, self.global_path, current_round=2)

    def test_missing_validation_set(self):
        agg = self.agg("select_train", val_path=os.path.join(TMP, "does_not_exist.npz"))
        with self.assertRaises(ValueError):
            agg.aggregate(self.client_data, self.global_path, current_round=1)
        self.agg("fedyogi", val_path=os.path.join(TMP, "does_not_exist.npz")).aggregate(
            self.client_data, self.global_path, current_round=1)       # static rules need no V

    def test_bce_is_unweighted(self):
        y = np.array([1, 0, 0, 0])
        self.assertAlmostEqual(bce(y, np.full(4, 0.5)), float(np.log(2)), places=6)


if __name__ == "__main__":
    unittest.main()
