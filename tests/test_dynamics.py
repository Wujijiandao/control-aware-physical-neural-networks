import unittest
import numpy as np
from piha.dynamics import predicted_next

class DynamicsTests(unittest.TestCase):
    def test_state_bounds(self):
        for a in range(5):
            x = predicted_next([0.99,0.01,0.99], a, 0.9)
            self.assertTrue(np.all(x >= 0.0) and np.all(x <= 1.0))

if __name__ == "__main__": unittest.main()
