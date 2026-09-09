import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from model import Model
from aggregator import (
    FedAvg, qFedAvg, FedFV, FedAdam, FedProx, Krum, SCAFFOLD, get_aggregator_by_name
)

class TestAggregators(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp_models")
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        self.global_path = os.path.join(self.tmp_dir, "global_model.keras")
        base_model = Model()
        base_model.model.save(self.global_path)
        
        # Create 3 dummy client models
        self.client_paths = []
        for i in range(3):
            cp = os.path.join(self.tmp_dir, f"client_{i}.keras")
            cm = Model()
            # perturb weights slightly per client
            w = cm.model.get_weights()
            w_new = [layer + 0.01 * (i + 1) for layer in w]
            cm.model.set_weights(w_new)
            cm.model.save(cp)
            self.client_paths.append(cp)

        # Standard client_data format: [(filepath, samples, loss, cid)]
        self.client_data = [
            (self.client_paths[0], 1000, 0.5, "client_0"),
            (self.client_paths[1], 1000, 0.8, "client_1"),
            (self.client_paths[2], 1000, 0.3, "client_2")
        ]

    def test_factory_function(self):
        agg = get_aggregator_by_name("fedavg")
        self.assertIsInstance(agg, FedAvg)
        
        agg = get_aggregator_by_name("fedprox", mu=0.05)
        self.assertIsInstance(agg, FedProx)
        self.assertEqual(agg.mu, 0.05)
        
        agg = get_aggregator_by_name("krum")
        self.assertIsInstance(agg, Krum)
        
        agg = get_aggregator_by_name("scaffold")
        self.assertIsInstance(agg, SCAFFOLD)

    def test_fedavg(self):
        agg = FedAvg()
        agg.aggregate(self.client_data, self.global_path, current_round=1)
        self.assertTrue(os.path.exists(self.global_path))

    def test_qfedavg(self):
        agg = qFedAvg(q=0.5)
        agg.aggregate(self.client_data, self.global_path, current_round=1)
        self.assertTrue(os.path.exists(self.global_path))

    def test_fedprox(self):
        agg = FedProx(mu=0.01)
        agg.aggregate(self.client_data, self.global_path, current_round=1)
        self.assertTrue(os.path.exists(self.global_path))

    def test_krum(self):
        agg = Krum(num_byzantine=1)
        agg.aggregate(self.client_data, self.global_path, current_round=1)
        self.assertTrue(os.path.exists(self.global_path))

    def test_scaffold(self):
        agg = SCAFFOLD(lr=1.0)
        agg.aggregate(self.client_data, self.global_path, current_round=1)
        self.assertTrue(os.path.exists(self.global_path))

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

if __name__ == "__main__":
    unittest.main()
