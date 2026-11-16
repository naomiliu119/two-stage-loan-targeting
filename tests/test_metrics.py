"""Small hand-calculated checks of the ranking and targeting metrics."""
import unittest

import numpy as np

from scripts.targeting_metrics import ks, qini_and_deciles


class RankingMetricChecks(unittest.TestCase):
    def test_ks_hand_calculation(self):
        # Ranked by score: .8 (event), .4 (non-event), .35 (event), .1 (non-event).
        # Largest gap between cumulative event and non-event shares is 0.5.
        self.assertAlmostEqual(ks(np.array([0, 0, 1, 1]), np.array([.1, .4, .35, .8])), .5)


class TargetingMetricChecks(unittest.TestCase):
    def setUp(self):
        self.effect = np.arange(10, 0, -1, dtype=float)       # predicted effects, already descending
        self.uplift = np.array([1.] * 5 + [0.] * 5)            # per-letter adjusted uplift scores

    def test_perfect_ranking_qini_and_deciles(self):
        qini, deciles, tops, cum = qini_and_deciles(self.effect, self.uplift, np.arange(10))
        # Cumulative gain 1..5 then flat at 5; random line rises 0.5 per letter.
        # Mean gap = 12.5 / 10 = 1.25 per letter -> 1.25 / 10 * 1000 = 125 per 1,000.
        self.assertAlmostEqual(qini, 125.0)
        np.testing.assert_allclose(deciles, [100.] * 5 + [0.] * 5)
        np.testing.assert_allclose(list(tops.values()), [100., 100., 100.])
        np.testing.assert_allclose(cum, [1, 2, 3, 4, 5, 5, 5, 5, 5, 5])

    def test_reversed_ranking_is_negative_and_symmetric(self):
        qini, deciles, _, _ = qini_and_deciles(self.effect, self.uplift, np.arange(10)[::-1])
        self.assertAlmostEqual(qini, -125.0)
        np.testing.assert_allclose(deciles, [0.] * 5 + [100.] * 5)

    def test_constant_uplift_gives_zero_qini(self):
        qini, _, _, _ = qini_and_deciles(self.effect, np.full(10, .3), np.arange(10))
        self.assertAlmostEqual(qini, 0.0)


if __name__ == "__main__":
    unittest.main()
