import sys
import os
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from selector import (
    LinUCBAgent, WLSTSAgent, DQNAgent, MetaAggregatorAgent,
    HierarchicalFLSelector, OortHierarchicalFLSelector, get_selector_by_name
)

class TestGammaParameter(unittest.TestCase):
    def test_gamma_initialization(self):
        linucb = LinUCBAgent(gamma=1.0)
        self.assertEqual(linucb.gamma, 1.0)

        wlsts = WLSTSAgent(gamma=1.0)
        self.assertEqual(wlsts.gamma, 1.0)

        meta = MetaAggregatorAgent(gamma=1.0)
        self.assertEqual(meta.gamma, 1.0)

    def test_factory_gamma_propagation(self):
        hier_selector = get_selector_by_name("hierarchical", gamma=1.0)
        self.assertEqual(hier_selector.meta_agent.gamma, 1.0)
        self.assertEqual(hier_selector.sub_agent.gamma, 1.0)

        oort_selector = get_selector_by_name("oort", gamma=1.0)
        self.assertEqual(oort_selector.meta_agent.gamma, 1.0)
        self.assertEqual(oort_selector.sub_agent.gamma, 1.0)

if __name__ == "__main__":
    unittest.main()
