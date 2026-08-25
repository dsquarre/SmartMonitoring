import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from rl_env import FederatedEnv
from selector import LinUCBAgent, RLClientSelector

class TestLinUCBSelector(unittest.TestCase):
    def setUp(self):
        self.profiles = {
            0: {"cpu_frequency": 2.0e9, "tx_power": 0.2, "r_trans": 15e6},
            1: {"cpu_frequency": 1.5e9, "tx_power": 0.3, "r_trans": 10e6},
            2: {"cpu_frequency": 2.5e9, "tx_power": 0.1, "r_trans": 20e6},
            3: {"cpu_frequency": 1.0e9, "tx_power": 0.4, "r_trans": 5e6},
        }
        self.env = FederatedEnv(self.profiles)
        self.agent = LinUCBAgent(alpha=1.0, feature_dim=8)
        self.selector = RLClientSelector(agent=self.agent, env=self.env)
        self.client_ids = ["client_0", "client_1", "client_2", "client_3"]
        self.client_id_map = {cid: i for i, cid in enumerate(self.client_ids)}

    def test_linucb_selection_and_masking(self):
        context = {
            "active_clients": ["client_0", "client_1", "client_2"],  # client_3 is offline
            "client_id_map": self.client_id_map,
            "client_samples": {cid: 1000 for cid in self.client_ids},
            "client_losses": {cid: 0.5 for cid in self.client_ids}
        }
        
        # Select k=2 clients out of 3 active ones
        selected = self.selector.select_clients(self.client_ids, k=2, context=context)
        
        self.assertEqual(len(selected), 2)
        self.assertNotIn("client_3", selected, "Offline client_3 must be disqualified by Action Masking!")
        
        # Verify staleness counters
        self.assertEqual(self.selector.client_staleness["client_3"], 1)

    def test_vector_reward_and_linucb_update(self):
        context = {
            "active_clients": self.client_ids,
            "client_id_map": self.client_id_map,
            "client_samples": {cid: 1000 for cid in self.client_ids},
            "client_losses": {cid: 0.8 for cid in self.client_ids}
        }
        
        selected = self.selector.select_clients(self.client_ids, k=2, context=context)
        
        round_summary = {
            "round": 1,
            "selected_ids": selected,
            "client_id_map": self.client_id_map,
            "client_samples": {cid: 1000 for cid in self.client_ids},
            "client_losses": {cid: 0.5 for cid in self.client_ids},
            "global_loss_delta": 0.3,
            "local_losses": [0.5, 0.4],
            "active_clients": self.client_ids,
            "client_roundtrips": {cid: 2.0 for cid in selected},
            "client_latencies": {cid: 1.5 for cid in selected},
            "client_energies": {cid: 10.0 for cid in selected}
        }
        
        # Update policy
        self.selector.update_policy(round_summary)
        
        # Verify model parameters updated
        self.assertTrue(self.agent.recompute_inv)
        for cid in selected:
            self.assertEqual(self.selector.client_has_telemetry[cid], 1.0)
            self.assertGreater(self.selector.client_ema_latency[cid], 0.0)


if __name__ == "__main__":
    unittest.main()
