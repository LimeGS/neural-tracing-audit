import unittest

import numpy as np
from scipy.spatial import cKDTree

from benchmark_core import (
    Reference,
    render_status,
    score_prediction,
    summarize_score,
    wrap_error_histogram,
)


def synthetic_reference():
    seed = np.zeros((2, 2, 3), dtype=np.float64)
    seed[0, 0] = [0.0, 0.0, 0.0]
    seed[0, 1] = [10.0, 0.0, 0.0]
    seed[1, 0] = [20.0, 0.0, 0.0]

    expected = np.array(
        [
            [0.0, 0.0, 10.0],
            [10.0, 0.0, 10.0],
            [20.0, 0.0, 10.0],
        ],
        dtype=np.float64,
    )
    wrong_wrap = np.array(
        [
            [0.0, 0.0, 100.0],
            [10.0, 0.0, 20.0],
            [20.0, 0.0, 100.0],
        ],
        dtype=np.float64,
    )

    rr = np.array([0, 0, 1])
    cc = np.array([0, 1, 0])
    return Reference(
        xyz=np.zeros((21, 21, 3), dtype=np.float64),
        valid=np.ones((21, 21), dtype=bool),
        row0=0,
        seed=seed,
        cls=np.zeros((21, 21), dtype=np.int64),
        trees={1: cKDTree(expected), 2: cKDTree(wrong_wrap)},
        rows_of={1: np.array([10, 10, 10]), 2: np.array([10, 10, 10])},
        cols_of={1: np.array([10, 10, 10]), 2: np.array([10, 10, 10])},
        pts_of={1: expected, 2: wrong_wrap},
        rr=rr,
        cc=cc,
        seed_cls=np.array([0, 0, 0]),
    )


class BenchmarkCoreTest(unittest.TestCase):
    def test_score_prediction_splits_correct_wrong_wrap_and_distance_miss(self):
        ref = synthetic_reference()
        pred = np.zeros_like(ref.seed)
        pred[0, 0] = [0.0, 0.0, 6.0]      # correct: distance 4 < half-gap 5
        pred[0, 1] = [10.0, 0.0, 20.0]    # nearest class is wrong wrap
        pred[1, 0] = [20.0, 0.0, 16.0]    # expected class, but 6 > half-gap

        result = score_prediction(ref, pred, expected_class=1)
        summary = summarize_score(result)

        self.assertEqual(summary["included"], 3)
        self.assertEqual(summary["excluded"], 0)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["wrong_wrap"], 1)
        self.assertEqual(summary["distance_miss"], 1)
        self.assertAlmostEqual(summary["wrong_hop_pct"], 66.6666666667)
        self.assertEqual(wrap_error_histogram(result), {0: 2, 1: 1})

    def test_edge_reference_points_are_excluded(self):
        ref = synthetic_reference()
        ref.rows_of[1] = np.array([0, 10, 10])
        pred = np.zeros_like(ref.seed)
        pred[0, 0] = [0.0, 0.0, 3.0]
        pred[0, 1] = [10.0, 0.0, 3.0]
        pred[1, 0] = [20.0, 0.0, 3.0]

        result = score_prediction(ref, pred, expected_class=1)

        self.assertEqual(int(result["excluded"].sum()), 1)
        self.assertEqual(int(result["ok"].sum()), 2)

    def test_column_edge_reference_points_are_excluded(self):
        ref = synthetic_reference()
        ref.cols_of[1] = np.array([0, 10, 10])
        pred = np.zeros_like(ref.seed)
        pred[0, 0] = [0.0, 0.0, 3.0]
        pred[0, 1] = [10.0, 0.0, 3.0]
        pred[1, 0] = [20.0, 0.0, 3.0]

        result = score_prediction(ref, pred, expected_class=1)

        self.assertEqual(int(result["excluded"].sum()), 1)

    def test_render_status_uses_stable_palette_and_scale(self):
        image = render_status(np.array([0, 1, 2, 3], dtype=np.int8), (2, 2))

        self.assertEqual(image.shape, (8, 8, 3))
        np.testing.assert_array_equal(image[0, 0], np.array([60, 180, 60]))
        np.testing.assert_array_equal(image[0, 4], np.array([40, 40, 220]))
        np.testing.assert_array_equal(image[4, 0], np.array([0, 165, 255]))
        np.testing.assert_array_equal(image[4, 4], np.array([90, 90, 90]))


if __name__ == "__main__":
    unittest.main()
