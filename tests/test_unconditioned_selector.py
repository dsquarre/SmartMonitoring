import os
import unittest
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from selector import get_selector_by_name, UnconditionedHierarchicalFLSelector
from rl_env import FederatedEnv


class TestUnconditionedSelector(unittest.TestCase):
    def setUp(self):
        self.profiles = {i: {"cpu_frequency": 2.0e9, "tx_power": 0.2, "r_trans": 15e6} for i in range(10)}
        self.env = FederatedEnv(self.profiles)

    def test_unconditioned_selector_factory(self):
        selector = get_selector_by_name("unconditioned", env=self.env)
        self.assertIsInstance(selector, UnconditionedHierarchicalFLSelector)
        self.assertEqual(selector.meta_agent.feature_dim, 12)
        self.assertEqual(selector.sub_agent.feature_dim, 7)

    def test_unconditioned_state_dimensions(self):
        selector = UnconditionedHierarchicalFLSelector(env=self.env)
        context = {"round": 1, "rounds_left": 5, "active_clients": [f"client_{i}" for i in range(10)]}
        client_ids = [f"client_{i}" for i in range(10)]

        global_state = selector._build_global_state(context)
        self.assertEqual(global_state.shape, (12,))

        client_state = selector._build_conditioned_client_state(client_ids, agg_idx=0, context=context)
        self.assertEqual(client_state.shape, (10, 7))


if __name__ == "__main__":
    unittest.main()
