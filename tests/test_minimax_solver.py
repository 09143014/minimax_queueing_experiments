import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game


class MinimaxSolverTests(unittest.TestCase):
    def test_pure_saddle_point(self):
        payoff = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = solve_zero_sum_matrix_game(payoff)

        self.assertAlmostEqual(result["value"], 3.0)
        np.testing.assert_allclose(result["defender_strategy"], [1.0, 0.0], atol=1e-8)

    def test_matching_pennies_value(self):
        payoff = np.array([[1.0, -1.0], [-1.0, 1.0]])

        result = solve_zero_sum_matrix_game(payoff)

        self.assertAlmostEqual(result["value"], 0.0, places=8)
        np.testing.assert_allclose(result["defender_strategy"], [0.5, 0.5], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
