import unittest
import main
import numpy as np


class TestMain(unittest.TestCase):
    def test_compute_cost_function(self):
        self.assertEqual(main.compute_cost_function(np.array([1]), np.array([1]), 1, 1), 0.5)
        self.assertEqual(main.compute_cost_function(np.array([1, 2, 3]), np.array([1, 2, 3]), 1, 1), 0.5)

    def test_compute_gradient(self):
        self.assertEqual(main.compute_gradient(np.array([1]), np.array([1]), 1, 1), (1, 1))

    def test_gradient_descent(self):
        self.assertEqual(main.gradient_descent(np.array([1]), np.array([1]), 1, 1, 0.5, 1), (0.5, 0.5, [0]))


if __name__ == '__main__':
    unittest.main()
