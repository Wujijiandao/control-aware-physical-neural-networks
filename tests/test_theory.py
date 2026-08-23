import unittest
import numpy as np
from piha.theory import action_gap, action_preservation_sufficient, selected_action, subgaussian_action_mismatch_bound

class TheoryTests(unittest.TestCase):
    def test_action_gap(self):
        self.assertAlmostEqual(action_gap([0.1,0.3,0.4]), 0.2)

    def test_sufficient_condition_constructive(self):
        true = np.array([0.10, 0.50, 0.80])
        eps = 0.10
        approx = true + np.array([eps, -eps, eps])
        self.assertTrue(action_preservation_sufficient(action_gap(true), eps))
        self.assertEqual(selected_action(true), selected_action(approx))

    def test_subgaussian_mismatch_bound_decreases_with_margin(self):
        p_small = subgaussian_action_mismatch_bound([0.05, 0.08], 0.03)
        p_large = subgaussian_action_mismatch_bound([0.10, 0.16], 0.03)
        self.assertLess(p_large, p_small)

    def test_subgaussian_zero_noise(self):
        self.assertEqual(subgaussian_action_mismatch_bound([0.1, 0.2], 0.0), 0.0)

if __name__ == "__main__": unittest.main()
