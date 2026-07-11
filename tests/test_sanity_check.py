import unittest

import numpy as np

from sanity_check import displacement_summary


class SanityCheckTest(unittest.TestCase):
    def test_displacement_summary(self):
        seed = np.zeros((1, 2, 3), dtype=float)
        output = np.array([[[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]])
        summary = displacement_summary(seed, output)
        self.assertEqual(summary["valid"], 2)
        self.assertAlmostEqual(summary["distance_median_vox"], 2.5)
        self.assertEqual(summary["non_finite_values"], 0)


if __name__ == "__main__":
    unittest.main()
