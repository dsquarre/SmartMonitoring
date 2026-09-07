import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from rl_env import FederatedEnv
from simulate_fl import compute_dynamic_k

class TestRewardAndThresholds(unittest.TestCase):
    def setUp(self):
        self.profiles = {
            0: {"cpu_frequency": 2.0e9, "tx_power": 0.2, "r_trans": 15e6},
            1: {"cpu_frequency": 1.5e9, "tx_power": 0.3, "r_trans": 10e6},
            2: {"cpu_frequency": 2.5e9, "tx_power": 0.1, "r_trans": 20e6},
            3: {"cpu_frequency": 1.0e9, "tx_power": 0.4, "r_trans": 5e6},
        }
        self.env = FederatedEnv(self.profiles)
        self.client_ids = ["client_0", "client_1", "client_2", "client_3"]
        self.selected_ids = ["client_0", "client_1"]

    def test_dynamic_k_formula(self):
        self.assertEqual(compute_dynamic_k(10), 5)   # max(5, int(0.2*10)) = 5
        self.assertEqual(compute_dynamic_k(30), 6)   # max(5, int(0.2*30)) = 6
        self.assertEqual(compute_dynamic_k(3), 3)    # N < 5 -> 3
        self.assertEqual(compute_dynamic_k(1), 1)

    def test_subcontroller_reward_bounds(self):
        selected_metrics = {
            "client_0": {"t_total": 1.5, "E_total": 8.0},
            "client_1": {"t_total": 2.1, "E_total": 12.0}
        }
        client_losses = {"client_0": 0.4, "client_1": 0.7}

        c_rewards, scalar_r, meta_r = self.env.calculate_vector_rewards(
            self.client_ids, self.selected_ids, selected_metrics,
            global_loss_delta=0.1, client_losses=client_losses
        )

        for cid, r in c_rewards.items():
            self.assertGreaterEqual(r, 0.0, f"Reward for {cid} must be >= 0")
            self.assertLessEqual(r, 1.0, f"Reward for {cid} must be <= 1")

        self.assertGreaterEqual(scalar_r, 0.0)
        self.assertLessEqual(scalar_r, 1.0)
        self.assertGreaterEqual(meta_r, 0.0)
        self.assertLessEqual(meta_r, 1.0)

    def test_meta_reward_bounds(self):
        # r_t = 0.5 * (tanh(10 * z(ΔAcc) - z(E) - z(L)) + 1)
        r1 = self.env.calculate_meta_reward(global_acc_delta=0.05, total_round_energy=50.0, round_latency=2.0)
        r2 = self.env.calculate_meta_reward(global_acc_delta=-0.10, total_round_energy=100.0, round_latency=5.0)

        self.assertGreaterEqual(r1, 0.0)
        self.assertLessEqual(r1, 1.0)
        self.assertGreaterEqual(r2, 0.0)
        self.assertLessEqual(r2, 1.0)

if __name__ == "__main__":
    unittest.main()
