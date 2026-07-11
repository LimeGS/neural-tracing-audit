import unittest

import numpy as np

from threshold_sweep import row_for


class ThresholdSweepTest(unittest.TestCase):
    def test_row_for_reports_counts_and_distance_summaries(self):
        result = {
            "ok": np.array([True, True, False]),
            "excluded": np.array([False, False, True]),
            "correct": np.array([True, False, False]),
            "wrong_wrap": np.array([False, True, False]),
            "distance_miss": np.array([False, False, False]),
            "d_exp": np.array([2.0, 8.0, 99.0]),
            "gap": np.array([10.0, 20.0, 30.0]),
            "threshold": np.array([5.0, 10.0, 15.0]),
        }

        row = row_for("unit", "front", "gap_fraction", 0.5, result)

        self.assertEqual(row["config"], "unit")
        self.assertEqual(row["direction"], "front")
        self.assertEqual(row["scorable"], 2)
        self.assertEqual(row["excluded"], 1)
        self.assertEqual(row["hop_correct"], 1)
        self.assertEqual(row["wrong_hop"], 1)
        self.assertEqual(row["wrong_hop_pct"], "50.000")
        self.assertEqual(row["median_expected_dist_vox"], "5.000")
        self.assertEqual(row["median_local_gap_vox"], "15.000")
        self.assertEqual(row["median_threshold_vox"], "7.500")


if __name__ == "__main__":
    unittest.main()
