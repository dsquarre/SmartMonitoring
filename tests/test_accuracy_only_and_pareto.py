import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from rl_env import FederatedEnv

class TestAccuracyOnlyAndPareto(unittest.TestCase):
    def setUp(self):
        self.profiles = {0: {"cpu_frequency": 2.0e9, "tx_power": 0.2, "r_trans": 15e6}}
        self.env = FederatedEnv(self.profiles)

    def test_accuracy_only_reward_calculation(self):
        # Accuracy-Only: w_loss = 1.0, w_L = 0.0, w_E = 0.0, w_acc = 10.0
        client_ids = ["client_0"]
        selected_ids = ["client_0"]
        selected_metrics = {"client_0": {"t_total": 50.0, "E_total": 500.0}}
        client_losses = {"client_0": 0.3}

        c_rewards, scalar_r, meta_r = self.env.calculate_vector_rewards(
            client_ids, selected_ids, selected_metrics,
            global_loss_delta=0.05, client_losses=client_losses,
            w_loss=1.0, w_L=0.0, w_E=0.0, w_acc=10.0,
            global_acc_delta=0.05, total_round_energy=500.0, round_latency=50.0
        )

        self.assertGreaterEqual(c_rewards["client_0"], 0.0)
        self.assertLessEqual(c_rewards["client_0"], 1.0)
        self.assertGreaterEqual(meta_r, 0.0)
        self.assertLessEqual(meta_r, 1.0)

if __name__ == "__main__":
    unittest.main()
