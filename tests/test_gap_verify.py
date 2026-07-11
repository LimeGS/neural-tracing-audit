import unittest

import numpy as np

from gap_verify import displacement_stats, peak_locations


class GapVerifyTest(unittest.TestCase):
    def test_displacement_decomposition(self):
        seed = np.zeros((1, 1, 3))
        prediction = np.array([[[3.0, 4.0, 0.0]]])
        direction = np.array([[[1.0, 0.0, 0.0]]])
        signed, tangential, total = displacement_stats(seed, prediction, direction)
        self.assertAlmostEqual(float(signed[0, 0]), 3.0)
        self.assertAlmostEqual(float(tangential[0, 0]), 4.0)
        self.assertAlmostEqual(float(total[0, 0]), 5.0)

    def test_peak_locations_are_reported_in_profile_coordinates(self):
        offsets = np.arange(20, dtype=float)
        profile = np.zeros(20)
        profile[5] = 100
        profile[14] = 100
        peaks = peak_locations(profile, offsets, prominence=5, distance=5)
        np.testing.assert_array_equal(peaks, np.array([5.0, 14.0]))


if __name__ == "__main__":
    unittest.main()
