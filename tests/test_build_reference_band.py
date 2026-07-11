import unittest

import numpy as np

from build_reference_band import parse_ppm_header, parse_rows, row_byte_range


class BuildReferenceBandTest(unittest.TestCase):
    def test_header_and_row_range_arithmetic(self):
        header = b"width: 10\nheight: 20\nordered: true\n<>\n"
        width, height, offset = parse_ppm_header(header)
        self.assertEqual((width, height, offset), (10, 20, len(header)))
        self.assertEqual(
            row_byte_range(offset, width, row_start=2, row_count=3),
            (offset + 2 * 10 * 48, offset + 5 * 10 * 48 - 1),
        )

    def test_parse_rows_uses_six_little_endian_float64_channels(self):
        values = np.arange(2 * 3 * 6, dtype="<f8")
        rows = parse_rows(values.tobytes(), row_count=2, width=3)
        self.assertEqual(rows.shape, (2, 3, 6))
        np.testing.assert_array_equal(rows.ravel(), values)


if __name__ == "__main__":
    unittest.main()
