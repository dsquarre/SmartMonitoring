import os
import tempfile
import unittest
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from selector import LinUCBAgent, WLSTSAgent, RLClientSelector, HierarchicalFLSelector, MetaAggregatorAgent
from rl_env import FederatedEnv


class TestSelectorSerialization(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.profiles = {i: {"cpu_frequency": 2.0e9, "tx_power": 0.2, "r_trans": 15e6} for i in range(10)}
        self.env = FederatedEnv(self.profiles)

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_linucb_save_load(self):
        agent = LinUCBAgent(feature_dim=7, gamma=0.95)
        # Modify A and b manually to simulate learning
        agent.A += np.eye(7) * 2.5
        agent.b += np.ones((7, 1)) * 3.1

        save_path = os.path.join(self.temp_dir, "linucb_agent.npz")
        agent.save_state(save_path)
        self.assertTrue(os.path.exists(save_path))

        loaded_agent = LinUCBAgent(feature_dim=7)
        loaded_agent.load_state(save_path)

        np.testing.assert_allclose(agent.A, loaded_agent.A)
        np.testing.assert_allclose(agent.b, loaded_agent.b)
        self.assertEqual(loaded_agent.feature_dim, 7)

    def test_wlsts_save_load(self):
        agent = WLSTSAgent(feature_dim=7, gamma=0.95, sigma=0.25)
        agent.A += np.eye(7) * 1.8
        agent.b += np.ones((7, 1)) * 4.2

        save_path = os.path.join(self.temp_dir, "wlsts_agent.npz")
        agent.save_state(save_path)
        self.assertTrue(os.path.exists(save_path))

        loaded_agent = WLSTSAgent(feature_dim=7)
        loaded_agent.load_state(save_path)

        np.testing.assert_allclose(agent.A, loaded_agent.A)
        np.testing.assert_allclose(agent.b, loaded_agent.b)

    def test_hierarchical_selector_save_load_and_freeze(self):
        meta_agent = MetaAggregatorAgent(feature_dim=20)
        sub_agent = LinUCBAgent(feature_dim=14)
        selector = HierarchicalFLSelector(meta_agent=meta_agent, sub_agent=sub_agent, env=self.env)

        # Simulate update to alter parameters
        selector.meta_agent.A[0] += np.eye(20) * 0.5
        selector.sub_agent.A += np.eye(14) * 1.2

        save_path = os.path.join(self.temp_dir, "hierarchical_selector.npz")
        selector.save_selector(save_path)
        self.assertTrue(os.path.exists(save_path))

        # Create new selector and load
        loaded_selector = HierarchicalFLSelector(env=self.env)
        loaded_selector.load_selector(save_path)

        np.testing.assert_allclose(selector.meta_agent.A[0], loaded_selector.meta_agent.A[0])
        np.testing.assert_allclose(selector.sub_agent.A, loaded_selector.sub_agent.A)

        # Test Freeze functionality
        loaded_selector.freeze()
        self.assertTrue(loaded_selector.is_frozen)
        self.assertTrue(loaded_selector.meta_agent.is_frozen)
        self.assertTrue(loaded_selector.sub_agent.is_frozen)

        # Save pre-update matrices
        pre_A_meta = loaded_selector.meta_agent.A[0].copy()
        pre_A_sub = loaded_selector.sub_agent.A.copy()

        # Call update when frozen
        round_summary = {
            "selected_ids": ["client_0", "client_1"],
            "client_id_map": {"client_0": 0, "client_1": 1},
            "client_samples": {"client_0": 100, "client_1": 100},
            "client_losses": {"client_0": 0.5, "client_1": 0.6},
            "global_loss_delta": 0.1,
            "round": 1,
            "rounds_left": 5
        }
        loaded_selector.last_client_state = np.zeros((10, 14))
        loaded_selector.last_global_state = np.zeros((1, 20))
        loaded_selector.last_action = [0, 1]
        loaded_selector.last_client_ids = [f"client_{i}" for i in range(10)]

        loaded_selector.update_policy(round_summary)

        # Check that matrices remained unchanged because policy was frozen
        np.testing.assert_allclose(loaded_selector.meta_agent.A[0], pre_A_meta)
        np.testing.assert_allclose(loaded_selector.sub_agent.A, pre_A_sub)


if __name__ == "__main__":
    unittest.main()
