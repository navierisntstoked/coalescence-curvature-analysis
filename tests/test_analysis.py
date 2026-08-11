from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coalescence_curvature import (  # noqa: E402
    AnalysisConfig,
    analyze_frame,
    fit_local_boundary,
    read_tiff_frame,
    tiff_metadata,
)


class CurvatureMathTests(unittest.TestCase):
    def test_circle_curvature_matches_inverse_radius(self):
        radius = 80.0
        x = np.linspace(-15.0, 15.0, 61)
        top_y = 100.0 + np.sqrt(radius**2 - x**2)
        bottom_y = 300.0 - np.sqrt(radius**2 - x**2)

        top_fit = fit_local_boundary(x, top_y, "maximum", half_width_px=15)
        bottom_fit = fit_local_boundary(x, bottom_y, "minimum", half_width_px=15)

        expected = 1.0 / radius
        self.assertAlmostEqual(top_fit.curvature_px_inv, expected, delta=expected * 0.02)
        self.assertAlmostEqual(
            bottom_fit.curvature_px_inv, expected, delta=expected * 0.02
        )


class SyntheticTiffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_path = PROJECT_ROOT / "data" / "michelle0912_sample.tif"
        cls.config = AnalysisConfig()

    def test_tiff_metadata(self):
        metadata = tiff_metadata(self.sample_path)
        self.assertEqual(metadata["frames"], 500)
        self.assertEqual(metadata["width_px"], 504)
        self.assertEqual(metadata["height_px"], 1008)

    def test_neck_grows_across_stack(self):
        first = analyze_frame(read_tiff_frame(self.sample_path, 0), 0, self.config)
        last = analyze_frame(read_tiff_frame(self.sample_path, 499), 499, self.config)
        self.assertGreater(
            last.measurement.neck_radius_px, first.measurement.neck_radius_px
        )
        self.assertTrue(np.isfinite(last.measurement.mean_curvature_px_inv))


if __name__ == "__main__":
    unittest.main()
