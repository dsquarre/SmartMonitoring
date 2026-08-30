import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from rl_env import FederatedEnv, RunningNormalizer

class TestRunningNormalizer(unittest.TestCase):
    def test_running_normalizer_convergence(self):
        norm = RunningNormalizer(alpha=0.1)
        
        # Feed baseline energy around ~10.0J
        for _ in range(20):
            val = norm.normalize(10.0 + np.random.normal(0, 1))
            self.assertFalse(np.isnan(val))
            self.assertFalse(np.isinf(val))

        # Mean should be around 10.0
        self.assertAlmostEqual(norm.mean, 10.0, delta=2.0)

    def test_scale_jump_robustness(self):
        norm = RunningNormalizer(alpha=0.1)
        
        # Start at 1.0J
        for _ in range(10):
            norm.normalize(1.0)
            
        self.assertAlmostEqual(norm.mean, 1.0, delta=0.5)
        
        # Sudden scale jump to 200.0J
        for _ in range(15):
            val = norm.normalize(200.0)
            self.assertFalse(np.isnan(val))
            self.assertFalse(np.isinf(val))

        # Mean should adapt toward 200.0 without exploding
        self.assertGreater(norm.mean, 100.0)

    def test_env_reward_standardization(self):
        profiles = {0: {"cpu_frequency": 2.0e9, "tx_power": 0.2, "r_trans": 15e6}}
        env = FederatedEnv(profiles)
        
        client_ids = ["client_0"]
        selected_ids = ["client_0"]
        selected_metrics = {
            "client_0": {"t_train": 2.0, "t_trans": 0.5, "t_total": 2.5, "E_train": 100.0, "E_trans": 1.0, "E_total": 101.0}
        }
        client_losses = {"client_0": 0.5}
        
        c_rewards, scalar_r = env.calculate_vector_rewards(
            client_ids, selected_ids, selected_metrics, 
            global_loss_delta=0.1, client_losses=client_losses
        )
        
        self.assertIn("client_0", c_rewards)
        self.assertFalse(np.isnan(scalar_r))

if __name__ == "__main__":
    unittest.main()
