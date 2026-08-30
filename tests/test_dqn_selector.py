import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from rl_env import FederatedEnv
from selector import (
    DQNAgent, MetaAggregatorAgent, RLClientSelector, HierarchicalFLSelector
)

class TestDQNSelector(unittest.TestCase):
    def setUp(self):
        self.profiles = {
            0: {"cpu_frequency": 2.0e9, "tx_power": 0.2, "r_trans": 15e6},
            1: {"cpu_frequency": 1.5e9, "tx_power": 0.3, "r_trans": 10e6},
            2: {"cpu_frequency": 2.5e9, "tx_power": 0.1, "r_trans": 20e6},
            3: {"cpu_frequency": 1.0e9, "tx_power": 0.4, "r_trans": 5e6},
        }
        self.env = FederatedEnv(self.profiles)
        self.client_ids = ["client_0", "client_1", "client_2", "client_3"]
        self.client_id_map = {cid: i for i, cid in enumerate(self.client_ids)}

    def test_standalone_dqn_selector(self):
        dqn_agent_8d = DQNAgent(feature_dim=8, hidden_dim=16, lr=0.001)
        selector = RLClientSelector(agent=dqn_agent_8d, env=self.env)
        
        context = {
            "active_clients": ["client_0", "client_1", "client_2"],
            "client_id_map": self.client_id_map,
            "client_samples": {cid: 1000 for cid in self.client_ids},
            "client_losses": {cid: 0.6 for cid in self.client_ids}
        }
        
        selected = selector.select_clients(self.client_ids, k=2, context=context)
        self.assertEqual(len(selected), 2)
        self.assertNotIn("client_3", selected, "Offline client_3 must be disqualified by Action Masking!")
        
        round_summary = {
            "round": 1,
            "selected_ids": selected,
            "client_id_map": self.client_id_map,
            "client_samples": {cid: 1000 for cid in self.client_ids},
            "client_losses": {cid: 0.4 for cid in self.client_ids},
            "global_loss_delta": 0.2,
            "local_losses": [0.4, 0.5],
            "active_clients": self.client_ids,
            "client_roundtrips": {cid: 2.0 for cid in selected},
            "client_latencies": {cid: 1.5 for cid in selected},
            "client_energies": {cid: 10.0 for cid in selected}
        }
        
        selector.update_policy(round_summary)
        self.assertGreater(len(dqn_agent_8d.replay_buffer), 0)

    def test_hierarchical_dqn_selector(self):
        meta_agent = MetaAggregatorAgent(alpha=0.5, feature_dim=5)
        dqn_agent_15d = DQNAgent(feature_dim=15, hidden_dim=16, lr=0.001)
        
        selector = HierarchicalFLSelector(
            meta_agent=meta_agent,
            sub_agent=dqn_agent_15d,
            env=self.env
        )
        
        context = {
            "round": 1,
            "rounds_left": 10,
            "active_clients": ["client_0", "client_1", "client_2"],
            "client_id_map": self.client_id_map,
            "client_samples": {cid: 1000 for cid in self.client_ids},
            "client_losses": {cid: 0.7 for cid in self.client_ids},
            "global_loss_delta": 0.15,
            "poison_alert_flag": 0.0
        }
        
        selected = selector.select_clients(self.client_ids, k=2, context=context)
        self.assertEqual(len(selected), 2)
        self.assertNotIn("client_3", selected)
        self.assertEqual(selector.last_client_state.shape, (4, 15))
        
        round_summary = {
            "round": 1,
            "selected_ids": selected,
            "client_id_map": self.client_id_map,
            "client_samples": {cid: 1000 for cid in self.client_ids},
            "client_losses": {cid: 0.5 for cid in self.client_ids},
            "global_loss_delta": 0.25,
            "local_losses": [0.5, 0.4],
            "active_clients": self.client_ids,
            "client_roundtrips": {cid: 1.8 for cid in selected},
            "client_latencies": {cid: 1.2 for cid in selected},
            "client_energies": {cid: 8.0 for cid in selected}
        }
        
        selector.update_policy(round_summary)
        self.assertGreater(len(dqn_agent_15d.replay_buffer), 0)

if __name__ == "__main__":
    unittest.main()
