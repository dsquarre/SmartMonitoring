import sys
import os
import tempfile
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.modules.pop('model', None)
from client.model import Model


class TestClientModel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create dummy datasets (1000 time steps, 12 channels)
        # Partition 1: Standard shapes (N, 1000, 12) with balanced classes
        self.npz_path_standard = os.path.join(self.temp_dir.name, "client_standard.npz")
        X_train_std = np.random.randn(20, 1000, 12).astype(np.float32)
        y_train_std = np.array([0, 1] * 10, dtype=np.int32)
        X_test_std = np.random.randn(10, 1000, 12).astype(np.float32)
        y_test_std = np.array([0, 1] * 5, dtype=np.int32)

        np.savez(
            self.npz_path_standard,
            X_train=X_train_std,
            y_train=y_train_std,
            X_test=X_test_std,
            y_test=y_test_std
        )

        # Partition 2: Transposed signal shape (N, 12, 1000)
        self.npz_path_transposed = os.path.join(self.temp_dir.name, "client_transposed.npz")
        X_train_trans = np.random.randn(16, 12, 1000).astype(np.float32)
        y_train_trans = np.array([0, 1] * 8, dtype=np.int32)
        X_test_trans = np.random.randn(8, 12, 1000).astype(np.float32)
        y_test_trans = np.array([0, 1] * 4, dtype=np.int32)

        np.savez(
            self.npz_path_transposed,
            X_train=X_train_trans,
            y_train=y_train_trans,
            X_test=X_test_trans,
            y_test=y_test_trans
        )

        # Partition 3: Contaminated data containing NaNs and Infs
        self.npz_path_nan = os.path.join(self.temp_dir.name, "client_nan.npz")
        X_train_nan = np.random.randn(16, 1000, 12).astype(np.float32)
        X_train_nan[0, 5, 2] = np.nan
        X_train_nan[1, 10, 3] = np.inf
        X_train_nan[2, 15, 4] = -np.inf

        y_train_nan = np.array([0, 1] * 8, dtype=np.int32)
        X_test_nan = np.random.randn(8, 1000, 12).astype(np.float32)
        X_test_nan[0, 2, 1] = np.nan
        y_test_nan = np.array([0, 1] * 4, dtype=np.int32)

        np.savez(
            self.npz_path_nan,
            X_train=X_train_nan,
            y_train=y_train_nan,
            X_test=X_test_nan,
            y_test=y_test_nan
        )

        # Partition 4: Single class partition (all 0s) to test fallback
        self.npz_path_single_class = os.path.join(self.temp_dir.name, "client_single_class.npz")
        X_train_single = np.random.randn(10, 1000, 12).astype(np.float32)
        y_train_single = np.zeros(10, dtype=np.int32)
        X_test_single = np.random.randn(6, 1000, 12).astype(np.float32)
        y_test_single = np.zeros(6, dtype=np.int32)

        np.savez(
            self.npz_path_single_class,
            X_train=X_train_single,
            y_train=y_train_single,
            X_test=X_test_single,
            y_test=y_test_single
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_model_initialization_and_samples(self):
        client_model = Model(self.npz_path_standard, batch_size=4)
        self.assertEqual(client_model.get_samples(), 20)
        self.assertEqual(client_model.num_test_samples, 10)

    def test_transposed_data_generator(self):
        client_model = Model(self.npz_path_transposed, batch_size=4)
        ds = client_model.get_dataset(client_model.X_train, client_model.y_train, shuffle=False)
        for x_batch, y_batch in ds.take(1):
            self.assertEqual(x_batch.shape[1:], (1000, 12))
            self.assertEqual(y_batch.shape[0], 4)

    def test_nan_inf_data_handling(self):
        client_model = Model(self.npz_path_nan, batch_size=4)
        result = client_model.train(epochs=1, lr=0.05)
        self.assertTrue(np.isfinite(result["train_loss"]))
        self.assertTrue(np.isfinite(result["stat_utility"]))

        eval_res = client_model.evaluate()
        self.assertFalse(np.isnan(eval_res['loss']))
        self.assertFalse(np.isnan(eval_res['accuracy']))

    def test_single_class_fallback(self):
        client_model = Model(self.npz_path_single_class, batch_size=4)
        eval_res = client_model.evaluate()
        self.assertEqual(eval_res['roc_auc'], 0.5)

    def test_train_and_weights(self):
        client_model = Model(self.npz_path_standard, batch_size=4)
        w_before = client_model.get_weights()

        result = client_model.train(epochs=1, lr=0.05)
        self.assertIn("train_loss", result)
        self.assertGreater(result["stat_utility"], 0.0)

        w_after = client_model.get_weights()
        self.assertEqual(len(w_before), len(w_after))

        # Test setting weights back
        client_model.set_weights(w_before)
        w_reset = client_model.get_weights()
        for wb, wr in zip(w_before, w_reset):
            np.testing.assert_array_almost_equal(wb, wr)

    def test_plain_sgd_and_oort_utility(self):
        """TMLR supplement local_train: plain SGD (lr 0 leaves the trainable weights unchanged) and
        stat_utility = |B| * sqrt(mean per-sample loss^2) >= |B| * mean training loss."""
        client_model = Model(self.npz_path_standard, batch_size=4)
        trainable = [v.numpy().copy() for v in client_model.model.trainable_variables]
        result = client_model.train(epochs=1, lr=0.0)
        for before, v in zip(trainable, client_model.model.trainable_variables):
            np.testing.assert_array_equal(before, v.numpy())
        n = client_model.get_samples()
        self.assertGreaterEqual(result["stat_utility"] + 1e-6, n * result["train_loss"])

        result = client_model.train(epochs=2, lr=0.05, batch_size=5)
        self.assertEqual(client_model.batch_size, 5)
        moved = any(not np.allclose(b, v.numpy()) for b, v in zip(trainable, client_model.model.trainable_variables))
        self.assertTrue(moved)


    def test_evaluation_metrics_dict(self):
        client_model = Model(self.npz_path_standard, batch_size=4)
        eval_metrics = client_model.evaluate()

        required_keys = [
            "loss", "accuracy", "precision", "recall", "f1", "roc_auc",
            "tp", "fp", "tn", "fn"
        ]
        for key in required_keys:
            self.assertIn(key, eval_metrics)
            self.assertFalse(np.isnan(eval_metrics[key]), f"Metric {key} is NaN")

    def test_load_and_train_real_npz_dataset(self):
        real_npz_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "iid", "client_0.npz"
        )
        if not os.path.exists(real_npz_path):
            self.skipTest(f"Real dataset file not found at {real_npz_path}")

        client_model = Model(real_npz_path, batch_size=32)
        self.assertGreater(client_model.get_samples(), 0)

        result = client_model.train(epochs=1, lr=0.05)
        self.assertFalse(np.isnan(result["train_loss"]))

        eval_metrics = client_model.evaluate()
        self.assertFalse(np.isnan(eval_metrics['loss']))
        self.assertFalse(np.isnan(eval_metrics['accuracy']))


if __name__ == "__main__":
    unittest.main()
