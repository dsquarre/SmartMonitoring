import sys
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "server"))

from selector import OortClientSelector, RandomClientSelector, get_selector_by_name


class TestOortSelector(unittest.TestCase):
    def setUp(self):
        self.clients = [f"client_{i}" for i in range(12)]
        self.samples = {cid: 50.0 + 10 * i for i, cid in enumerate(self.clients)}

    def _round(self, sel, t, k=4, utilities=None):
        chosen = sel.select_clients(self.clients, k, context={"round": t, "client_samples": self.samples})
        utilities = utilities or {cid: self.samples[cid] for cid in chosen}
        sel.update_policy({"round": t, "selected_ids": chosen, "client_utilities": utilities,
                           "client_roundtrips": {cid: 1.0 + int(cid.split("_")[1]) for cid in chosen},
                           "elapsed_round": 5.0})
        return chosen

    def test_factory(self):
        self.assertIsInstance(get_selector_by_name("oort"), OortClientSelector)
        self.assertIsInstance(get_selector_by_name(None), OortClientSelector)
        self.assertIsInstance(get_selector_by_name("random"), RandomClientSelector)

    def test_selects_k_distinct_connected_clients(self):
        sel = OortClientSelector(seed=0)
        for t in range(1, 8):
            chosen = self._round(sel, t)
            self.assertEqual(len(chosen), 4)
            self.assertEqual(len(set(chosen)), 4)
            self.assertTrue(set(chosen) <= set(self.clients))

    def test_only_feasible_clients_and_k_capped(self):
        sel = OortClientSelector(seed=0)
        self._round(sel, 1)
        connected = self.clients[:3]
        chosen = sel.select_clients(connected, 5, context={"round": 2, "client_samples": self.samples})
        self.assertEqual(sorted(chosen), sorted(connected))

    def test_feedback_reaches_oort(self):
        sel = OortClientSelector(seed=0)
        chosen = self._round(sel, 1, utilities={cid: 123.0 for cid in self.clients})
        for cid in chosen:
            arm = sel.oort.arms[cid]
            self.assertEqual(arm["count"], 1)
            self.assertEqual(arm["duration"], 1.0 + int(cid.split("_")[1]))

    def test_deterministic_for_a_seed(self):
        a, b = OortClientSelector(seed=7), OortClientSelector(seed=7)
        self.assertEqual([self._round(a, t) for t in range(1, 6)], [self._round(b, t) for t in range(1, 6)])


if __name__ == "__main__":
    unittest.main()
