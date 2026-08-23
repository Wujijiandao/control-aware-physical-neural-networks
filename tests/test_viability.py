import unittest
import numpy as np
from piha.viability import viability_np

class ViabilityTests(unittest.TestCase):
    def test_safe_state_lower_than_bad_state(self):
        safe = float(viability_np(np.array([0.60, 0.45, 0.15])))
        bad = float(viability_np(np.array([0.10, 0.90, 0.90])))
        self.assertLess(safe, bad)

    def test_vectorization(self):
        x = np.array([[0.6,0.45,0.15],[0.1,0.9,0.9]])
        y = viability_np(x)
        self.assertEqual(y.shape, (2,))

if __name__ == "__main__": unittest.main()
