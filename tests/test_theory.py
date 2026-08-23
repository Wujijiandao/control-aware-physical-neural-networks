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


def test_finite_horizon_first_divergence_bound():
    from piha.theory import finite_horizon_first_divergence_bound, finite_horizon_match_probability_lower_bound
    gaps=[np.array([0.4,0.7]),np.array([0.5,0.8])]
    b=finite_horizon_first_divergence_bound(gaps,0.1)
    assert 0.0 <= b <= 1.0
    assert np.isclose(finite_horizon_match_probability_lower_bound(gaps,0.1),1.0-b)
    # Larger margins cannot make the union bound worse.
    b2=finite_horizon_first_divergence_bound([g*2 for g in gaps],0.1)
    assert b2 <= b
