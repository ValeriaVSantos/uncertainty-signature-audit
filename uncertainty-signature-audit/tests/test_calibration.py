import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "src" / "ece_calibration_pipeline.py"
SPEC = importlib.util.spec_from_file_location("ece_calibration_pipeline", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CalculateEceTests(unittest.TestCase):
    def test_perfect_alignment_is_zero(self):
        values = np.array([5.0, 25.0, 55.0, 95.0])
        result, _, _ = MODULE.calculate_ece(values, values, n_bins=10)
        self.assertAlmostEqual(result, 0.0)

    def test_single_bin_matches_absolute_mean_gap(self):
        confidence = np.array([20.0, 40.0])
        reference = np.array([10.0, 30.0])
        result, _, _ = MODULE.calculate_ece(confidence, reference, n_bins=1)
        self.assertAlmostEqual(result, 10.0)

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            MODULE.calculate_ece(np.array([10.0]), np.array([10.0, 20.0]))


if __name__ == "__main__":
    unittest.main()
