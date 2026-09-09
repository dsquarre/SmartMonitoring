import os
import unittest
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from selector import get_selector_by_name, HierarchicalFLSelector, OortHierarchicalFLSelector
from compare_experiments import get_round_to_target_acc
from rl_env import FederatedEnv


class TestDecoupledGammaAndTargetRound(unittest.TestCase):
    def setUp(self):
        self.profiles = {i: {"cpu_frequency": 2.0e9, "tx_power": 0.2, "r_trans": 15e6} for i in range(10)}
        self.env = FederatedEnv(self.profiles)

    def test_decoupled_gamma_initialization(self):
        selector = get_selector_by_name("hierarchical", env=self.env, gamma_meta=0.98, gamma_sub=0.90)
        self.assertIsInstance(selector, HierarchicalFLSelector)
        self.assertEqual(selector.meta_agent.gamma, 0.98)
        self.assertEqual(selector.sub_agent.gamma, 0.90)

    def test_oort_decoupled_gamma_initialization(self):
        selector = get_selector_by_name("oort", env=self.env, gamma_meta=1.0, gamma_sub=0.95)
        self.assertIsInstance(selector, OortHierarchicalFLSelector)
        self.assertEqual(selector.meta_agent.gamma, 1.0)
        self.assertEqual(selector.sub_agent.gamma, 0.95)

    def test_get_round_to_target_acc(self):
        dummy_run = [
            {"f2": 0.10},
            {"f2": 0.20},
            {"f2": 0.27},  # 90% of max 0.30 is 0.27
            {"f2": 0.30},
            {"f2": 0.29}
        ]
        round_idx = get_round_to_target_acc(dummy_run, target_ratio=0.90)
        self.assertEqual(round_idx, 3.0)


if __name__ == "__main__":
    unittest.main()
