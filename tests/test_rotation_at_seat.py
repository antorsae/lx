import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate_visualizations import PolarResponseVisualizer
from polar_data_loader import PolarDataLoader

ROTATIONS = list(range(0, 91, 10))
FREQS = np.array([500.0, 1000.0, 2000.0])


class RotationAtSeatTest(unittest.TestCase):
    def setUp(self):
        # The speaker points at the seat at 45 deg rotation: the level falls
        # 1 dB per 10 deg either side of it
        data = {
            "LX521.4 MGNEO": {
                "has_rear": False,
                "common_frequencies": FREQS,
                "angles": {
                    rotation: {
                        "magnitude": np.full(len(FREQS), 80.0 - abs(rotation - 45) / 10),
                        "phase": np.zeros(len(FREQS)),
                        "unit": "dB SPL",
                        "smoothing": "None",
                        "metadata": {"title": f"LX521 {rotation} DEG FRONT"},
                    }
                    for rotation in ROTATIONS
                },
            }
        }
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        path = self.root / "seat.h5"
        PolarDataLoader(connect_to_rew=False).save_to_hdf5(data, str(path))
        self.viz = PolarResponseVisualizer(
            str(path),
            static_plots_dir=self.root / "static",
            interactive_plots_dir=self.root / "interactive",
            rotation_at_seat=True,
        )

    def test_every_rotation_is_drawn_and_mirror_rotations_pair_up(self):
        res = self.viz.calc_results["LX521.4 MGNEO"]
        curves = self.viz._rotation_curves(np.array(res["angles"]), res["spl_matrix"])
        by_label = {label: (color, dashed) for label, color, dashed, _ in curves}

        self.assertEqual(list(by_label), [f"{r}°" for r in ROTATIONS])
        for low, high in [(40, 50), (30, 60), (0, 90)]:
            self.assertEqual(by_label[f"{low}°"][0], by_label[f"{high}°"][0])
            self.assertFalse(by_label[f"{low}°"][1])
            self.assertTrue(by_label[f"{high}°"][1])
        self.assertNotEqual(by_label["40°"][0], by_label["0°"][0])

    def test_no_plot_is_normalized_to_the_0_deg_rotation(self):
        self.viz.plot_frequency_response_by_angle()
        self.viz.plot_contour("LX521.4 MGNEO", normalized=False)
        core = self.root / "static" / "core"

        self.assertTrue((core / "LX521.4 MGNEO_freq_response_angles.png").exists())
        self.assertTrue((core / "LX521.4 MGNEO_contour_absolute.png").exists())
        self.assertFalse((core / "LX521.4 MGNEO_freq_response_normalized.png").exists())
        self.assertTrue((self.root / "interactive" / "LX521.4 MGNEO_freq_response_angles.html").exists())

    def test_polar_draws_each_rotation_once_relative_to_0_deg(self):
        entry = self.viz._build_polar_step_entry(
            self.viz.calc_results["LX521.4 MGNEO"], 1000.0, use_rear=False, rotation=True
        )

        np.testing.assert_allclose(entry["theta"], ROTATIONS)
        # 0 deg is 4.5 dB below the rotations that aim at the seat
        np.testing.assert_allclose(entry["r"], [4.5 - abs(r - 45) / 10 for r in ROTATIONS])

    def test_polar_explorer_shows_the_rotations_as_a_sector(self):
        self.viz.plot_polar_interactive_slider()
        html = (self.root / "interactive" / "polar" / "polar_explorer.html").read_text(encoding="utf-8")

        self.assertRegex(html, r'"sector":\s*\[45(\.0)?,\s*135(\.0)?\]')
        self.assertNotRegex(html, r'id="loadExtraPolar\w+Btn"')

    def test_explorer_speaks_of_rotation_and_overlays_no_off_axis_set(self):
        self.viz.plot_frequency_response_explorer()
        html = (self.root / "interactive" / "freq_response_explorer.html").read_text(encoding="utf-8")

        self.assertIn("<h3>Rotations</h3>", html)
        self.assertIn("Reference Rotation:", html)
        self.assertNotRegex(html, r'id="loadExtra\w+Btn"')


if __name__ == "__main__":
    unittest.main()
