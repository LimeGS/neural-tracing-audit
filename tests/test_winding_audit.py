import unittest

import numpy as np

from winding_audit_v4 import result_summary


class WindingAuditTest(unittest.TestCase):
    def test_result_summary_preserves_count_arithmetic(self):
        result = {
            "ok": np.array([True, True, False]),
            "excluded": np.array([False, False, True]),
            "correct": np.array([True, False, False]),
            "wrong_wrap": np.array([False, True, False]),
            "distance_miss": np.array([False, False, False]),
            "d_exp": np.array([1.0, 3.0, 99.0]),
            "gap": np.array([4.0, 4.0, 1.0]),
            "ratio": np.array([0.25, 0.75, 99.0]),
            "wrap_index_error": np.array([0, 1, 0]),
        }
        summary = result_summary(result)
        self.assertEqual(summary["included"], 2)
        self.assertEqual(summary["wrong_hop"], 1)
        self.assertAlmostEqual(summary["wrong_hop_pct"], 50.0)
        self.assertAlmostEqual(summary["median_expected_dist_gap_ratio"], 0.5)
        self.assertEqual(summary["wrap_index_error_histogram"], {0: 1, 1: 1})


if __name__ == "__main__":
    unittest.main()
