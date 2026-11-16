"""Small hand-checks of the estimator and probability decomposition."""
import unittest
import numpy as np
from src.evaluation import counterfactual_scores, rule_scores
from src.diagnostics import loss_vectors


class EvaluationChecks(unittest.TestCase):
    def test_dr_known_example_and_no_clipping(self):
        # Balanced assignment; a residual can validly push a DR score above 1.
        scores = counterfactual_scores(np.array([1, 0]), np.array([1, 0]), np.array([.5, .5]), np.array([[.2, .3], [.4, .6]]))
        np.testing.assert_allclose(scores, [[.2, 1.7], [-.4, .6]])
        choices = np.array([[1, 0], [0, 1]])
        np.testing.assert_allclose(rule_scores(scores, choices).mean(axis=0), [.65, .4])

    def test_unsupported_assignment_stops(self):
        with self.assertRaises(ValueError):
            counterfactual_scores([1], [1], [0], [[.2, .3]])

    def test_binary_score_hand_calculation(self):
        losses = loss_vectors([1, 0], [.8, .2])
        np.testing.assert_allclose(losses["Log loss"], [-np.log(.8), -np.log(.8)])
        np.testing.assert_allclose(losses["Brier"], [.04, .04])


if __name__ == "__main__":
    unittest.main()
