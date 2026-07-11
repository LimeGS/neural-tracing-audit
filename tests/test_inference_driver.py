import hashlib
import tempfile
import unittest

from inference_driver import build_argv, sha256_file


class InferenceDriverTest(unittest.TestCase):
    def test_build_argv_freezes_reduced_configuration(self):
        argv = build_argv("seed", "volume", "model", "mps", (128, 128, 128), 1, "out", True)
        self.assertIn("--no-tta", argv)
        crop = argv.index("--crop-size")
        self.assertEqual(argv[crop + 1:crop + 4], ["128", "128", "128"])
        self.assertEqual(argv[argv.index("--batch-size") + 1], "1")

    def test_build_argv_leaves_tta_on_for_native_configuration(self):
        argv = build_argv("seed", "volume", "model", "cuda", (128, 384, 384), 4, "out", False)
        self.assertNotIn("--no-tta", argv)
        crop = argv.index("--crop-size")
        self.assertEqual(argv[crop + 1:crop + 4], ["128", "384", "384"])

    def test_sha256_file(self):
        with tempfile.NamedTemporaryFile() as file:
            file.write(b"benchmark")
            file.flush()
            self.assertEqual(sha256_file(file.name), hashlib.sha256(b"benchmark").hexdigest())


if __name__ == "__main__":
    unittest.main()
