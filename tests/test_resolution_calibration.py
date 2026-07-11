import unittest

import numpy as np

from resolution_calibration import pitch_bin_rows, rescale_prediction, summarize_on_mask


class ResolutionCalibrationTest(unittest.TestCase):
    def test_rescale_prediction_preserves_seed_and_scales_displacement(self):
        seed = np.array([[1.0, 2.0, 3.0]])
        prediction = np.array([[5.0, 0.0, 7.0]])
        actual = rescale_prediction(seed, prediction, 0.5)
        np.testing.assert_allclose(actual, [[3.0, 1.0, 5.0]])

    def test_summarize_on_mask_uses_frozen_denominator(self):
        result = {
            "ok": np.array([True, True, True, False]),
            "excluded": np.array([False, False, False, True]),
            "correct": np.array([True, False, False, False]),
            "wrong_wrap": np.array([False, True, False, False]),
            "distance_miss": np.array([False, False, True, False]),
        }
        summary = summarize_on_mask(result, np.array([True, True, True, False]))
        self.assertEqual(summary["included"], 3)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["wrong_hop"], 2)
        self.assertAlmostEqual(summary["correct_pct"], 100.0 / 3.0)

    def test_summarize_on_mask_rejects_new_exclusions(self):
        result = {
            "ok": np.array([True, False]),
            "excluded": np.array([False, True]),
            "correct": np.array([True, False]),
            "wrong_wrap": np.array([False, False]),
            "distance_miss": np.array([False, False]),
        }
        with self.assertRaises(ValueError):
            summarize_on_mask(result, np.array([True, True]))

    def test_pitch_bin_rows_reports_numerator_and_denominator(self):
        result = {
            "gap": np.array([6.0, 8.0, 10.0, 17.0]),
            "correct": np.array([False, True, False, True]),
        }
        rows = pitch_bin_rows(
            result,
            np.ones(4, dtype=bool),
            bins=((0.0, 9.0), (9.0, np.inf)),
        )
        self.assertEqual(rows[0]["count"], 2)
        self.assertEqual(rows[0]["misses"], 1)
        self.assertEqual(rows[1]["count"], 2)
        self.assertEqual(rows[1]["misses"], 1)


if __name__ == "__main__":
    unittest.main()
