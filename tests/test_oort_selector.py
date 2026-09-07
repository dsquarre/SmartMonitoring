import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from rl_env import FederatedEnv
from selector import LinUCBAgent, MetaAggregatorAgent, OortHierarchicalFLSelector, get_selector_by_name

class TestOortSelector(unittest.TestCase):
    def setUp(self):
        self.profiles = {
            0: {"cpu_frequency": 2.0e9, "tx_power": 0.2, "r_trans": 15e6},
            1: {"cpu_frequency": 1.5e9, "tx_power": 0.3, "r_trans": 10e6},
            2: {"cpu_frequency": 2.5e9, "tx_power": 0.1, "r_trans": 20e6},
            3: {"cpu_frequency": 1.0e9, "tx_power": 0.4, "r_trans": 5e6},
        }
        self.env = FederatedEnv(self.profiles)
        self.meta_agent = MetaAggregatorAgent(alpha=0.5, feature_dim=4)
        self.sub_agent = LinUCBAgent(alpha=1.0, feature_dim=11)
        self.selector = OortHierarchicalFLSelector(
            meta_agent=self.meta_agent,
            sub_agent=self.sub_agent,
            env=self.env
        )
        self.client_ids = ["client_0", "client_1", "client_2", "client_3"]
        self.client_id_map = {cid: i for i, cid in enumerate(self.client_ids)}

    def test_oort_state_dimensions(self):
        context = {
            "round": 1,
            "rounds_left": 10,
            "active_clients": ["client_0", "client_1", "client_2"],
            "client_id_map": self.client_id_map,
            "client_samples": {cid: 1000 for cid in self.client_ids},
            "client_losses": {cid: 0.7 for cid in self.client_ids},
            "global_loss_delta": 0.2,
        }
        
        selected_clients = self.selector.select_clients(self.client_ids, k=2, context=context)
        
        self.assertIn(self.selector.last_chosen_agg, MetaAggregatorAgent.STRATEGIES)
        self.assertEqual(len(selected_clients), 2)
        self.assertNotIn("client_3", selected_clients, "Offline client_3 must be excluded")
        
        # Meta state G_t dimension check (4 dims)
        self.assertEqual(self.selector.last_global_state.shape, (4,))
        # Client state X_t dimension check (4 clients x 11 features: 4 base + 7 one-hot)
        self.assertEqual(self.selector.last_client_state.shape, (4, 11))

    def test_factory_creation(self):
        selector_instance = get_selector_by_name("oort", env=self.env)
        self.assertIsInstance(selector_instance, OortHierarchicalFLSelector)

    def test_policy_updates(self):
        context = {
            "round": 1,
            "rounds_left": 10,
            "active_clients": self.client_ids,
            "client_id_map": self.client_id_map,
            "client_samples": {cid: 1000 for cid in self.client_ids},
            "client_losses": {cid: 0.6 for cid in self.client_ids}
        }
        
        selected = self.selector.select_clients(self.client_ids, k=2, context=context)
        
        round_summary = {
            "round": 1,
            "selected_ids": selected,
            "client_id_map": self.client_id_map,
            "client_samples": {cid: 1000 for cid in self.client_ids},
            "client_losses": {cid: 0.4 for cid in self.client_ids},
            "global_loss_delta": 0.35,
            "local_losses": [0.4, 0.3],
            "active_clients": self.client_ids,
            "client_roundtrips": {cid: 1.8 for cid in selected},
            "client_latencies": {cid: 1.2 for cid in selected},
            "client_energies": {cid: 8.0 for cid in selected}
        }
        
        self.selector.update_policy(round_summary)
        self.assertTrue(self.sub_agent.recompute_inv)

if __name__ == "__main__":
    unittest.main()
