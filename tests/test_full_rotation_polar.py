import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate_visualizations import PolarResponseVisualizer
from polar_data_loader import PolarDataLoader

ANGLES = list(range(0, 91, 10))
FREQS = np.array([500.0, 1000.0, 2000.0])


def _side(level_for_angle):
    return {
        angle: {
            "magnitude": np.full(len(FREQS), level_for_angle(angle)),
            "phase": np.zeros(len(FREQS)),
            "unit": "dB SPL",
            "smoothing": "None",
            "metadata": {"title": f"capture {angle}"},
        }
        for angle in ANGLES
    }


class FullRotationPolarTest(unittest.TestCase):
    def setUp(self):
        # Each side falls off at its own rate, so a level shows which capture
        # was drawn: 0-180 deg loses 1 dB per 10 deg, 180-360 deg 2 dB.
        data = {
            "BATMAN": {
                "has_rear": True,
                "common_frequencies": FREQS,
                "angles": _side(lambda a: 90.0 - a / 10),
                "rear_angles": _side(lambda a: 80.0 - a / 10),
                "left_angles": _side(lambda a: 90.0 - a / 5),
                "left_rear_angles": _side(lambda a: 80.0 - a / 5),
            }
        }
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        path = root / "polar.h5"
        PolarDataLoader(connect_to_rew=False).save_to_hdf5(data, str(path))
        self.viz = PolarResponseVisualizer(
            str(path),
            static_plots_dir=root / "static",
            interactive_plots_dir=root / "interactive",
        )

    def test_polar_draws_the_180_360_half_instead_of_the_mirror(self):
        angles_rad, spl = self.viz._build_360_polar_data("BATMAN", 0)
        level = dict(zip(np.round(np.degrees(angles_rad)).astype(int).tolist(), (spl + 90.0).tolist()))

        self.assertAlmostEqual(level[30], 87.0)
        self.assertAlmostEqual(level[150], 77.0)
        self.assertAlmostEqual(level[330], 84.0)
        self.assertAlmostEqual(level[270], 72.0)
        self.assertAlmostEqual(level[210], 74.0)

    def test_contour_puts_the_180_360_half_at_negative_angles(self):
        angles, spl = self.viz._full_circle_contour_grid(self.viz.calc_results["BATMAN"])
        level = dict(zip(angles.tolist(), spl[0].tolist()))

        self.assertEqual(angles.tolist(), list(range(-180, 181, 2)))
        self.assertAlmostEqual(level[30], 87.0)
        self.assertAlmostEqual(level[150], 77.0)
        self.assertAlmostEqual(level[-30], 84.0)
        self.assertAlmostEqual(level[-150], 74.0)

    def test_directivity_uses_the_power_average_of_both_sides(self):
        calc = self.viz.calc_results["BATMAN"]["calculator"]
        expected = 10 * np.log10((10 ** 8.7 + 10 ** 8.4) / 2)
        self.assertAlmostEqual(calc.spl_matrix[0, 3], expected)


if __name__ == "__main__":
    unittest.main()
