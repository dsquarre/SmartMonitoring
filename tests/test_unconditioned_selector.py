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

    def test_individual_client_selection_probability(self):
        from selector import compute_client_selection_probabilities, build_base_client_features
        client_ids = [f"client_{i}" for i in range(4)]
        selection_history = [
            ["client_0", "client_1"],
            ["client_0", "client_2"]
        ]
        # Total selections = 4 (kW). client_0 selected 2 times (p_0 = 2/4 = 0.5), client_1 selected 1 (p_1 = 0.25), client_2 selected 1 (p_2 = 0.25), client_3 selected 0 (p_3 = 0.0)
        probs = compute_client_selection_probabilities(selection_history, client_ids, window_size=10)
        self.assertAlmostEqual(probs["client_0"], 0.5)
        self.assertAlmostEqual(probs["client_1"], 0.25)
        self.assertAlmostEqual(probs["client_2"], 0.25)
        self.assertAlmostEqual(probs["client_3"], 0.0)

        # Check feature 7 in X_t matrix
        features = build_base_client_features(
            client_ids, {}, {}, np.inf, {}, {}, {}, {}, {}, selection_history, window_size=10
        )
        self.assertAlmostEqual(features[0, 6], 0.5)   # client_0 p_i
        self.assertAlmostEqual(features[1, 6], 0.25)  # client_1 p_i
        self.assertAlmostEqual(features[2, 6], 0.25)  # client_2 p_i
        self.assertAlmostEqual(features[3, 6], 0.0)   # client_3 p_i


if __name__ == "__main__":
    unittest.main()

