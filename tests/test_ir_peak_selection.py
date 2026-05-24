import unittest
from unittest.mock import patch

import numpy as np

from polar_data_loader import PolarDataLoader, select_direct_ir_peak


class DirectIrPeakSelectionTest(unittest.TestCase):
    def test_loader_rejects_strongest_policy_without_legacy_override(self):
        with patch("config.ALLOW_UNSAFE_STRONGEST_IR_PEAK_POLICY", False):
            with self.assertRaisesRegex(ValueError, "unsafe for high-angle validation"):
                PolarDataLoader(
                    connect_to_rew=False,
                    direct_ir_peak_policy="strongest",
                )

    def test_loader_allows_strongest_policy_for_legacy_diagnostics_only(self):
        with patch("config.ALLOW_UNSAFE_STRONGEST_IR_PEAK_POLICY", True):
            loader = PolarDataLoader(
                connect_to_rew=False,
                direct_ir_peak_policy="strongest",
            )
        self.assertEqual(loader.direct_ir_peak_policy, "strongest")

    def test_loader_allows_ir_start_policy(self):
        loader = PolarDataLoader(
            connect_to_rew=False,
            direct_ir_peak_policy="ir-start",
        )
        self.assertEqual(loader.direct_ir_peak_policy, "ir-start")

    def test_ir_start_policy_uses_rew_ir_start_metadata(self):
        loader = PolarDataLoader(
            connect_to_rew=False,
            direct_ir_peak_policy="ir-start",
        )
        result = loader._auto_fix_timing(
            "unused",
            {"timeOfIRStartSeconds": -0.009333333333333332},
        )

        self.assertFalse(result["corrected"])
        self.assertFalse(result.get("peak_selection_failed", False))
        self.assertEqual(result["direct_ir_peak_policy"], "ir-start")
        self.assertAlmostEqual(result["selected_peak_time_ms"], -9.333333333333332)
        self.assertEqual(result["selected_peak_reason"], "REW stored IR start time")

    def test_ignores_early_event_and_selects_reference_window_peak(self):
        sample_rate_hz = 48_000.0
        start_time_s = -0.020
        ir = np.zeros(2048)

        def set_peak(time_s: float, amplitude: float) -> int:
            idx = int(round((time_s - start_time_s) * sample_rate_hz))
            ir[idx] = amplitude
            return idx

        set_peak(-0.0121, 0.30)
        direct_idx = set_peak(0.00012, 1.00)

        selection = select_direct_ir_peak(
            ir,
            start_time_s=start_time_s,
            sample_rate_hz=sample_rate_hz,
            threshold_fraction=0.10,
            reference_window_s=0.002,
            policy="strongest",
        )

        self.assertFalse(selection["rejected"])
        self.assertEqual(selection["index"], direct_idx)
        self.assertEqual(selection["reason"], "strongest significant peak inside reference window")

    def test_default_policy_selects_first_lobe_not_largest_lobe(self):
        sample_rate_hz = 48_000.0
        start_time_s = -0.020
        ir = np.zeros(2048)

        def set_peak(time_s: float, amplitude: float) -> int:
            idx = int(round((time_s - start_time_s) * sample_rate_hz))
            ir[idx] = amplitude
            return idx

        first_lobe_idx = set_peak(0.00004, 0.95)
        set_peak(0.00012, 1.00)

        selection = select_direct_ir_peak(
            ir,
            start_time_s=start_time_s,
            sample_rate_hz=sample_rate_hz,
            threshold_fraction=0.10,
            reference_window_s=0.002,
        )

        self.assertFalse(selection["rejected"])
        self.assertEqual(selection["index"], first_lobe_idx)
        self.assertEqual(selection["policy"], "first-strong")
        self.assertEqual(selection["reason"], "first strong near-reference lobe")

    def test_first_strong_policy_selects_first_lobe_not_largest_lobe(self):
        sample_rate_hz = 48_000.0
        start_time_s = -0.020
        ir = np.zeros(2048)

        def set_peak(time_s: float, amplitude: float) -> int:
            idx = int(round((time_s - start_time_s) * sample_rate_hz))
            ir[idx] = amplitude
            return idx

        first_lobe_idx = set_peak(0.00004, 0.95)
        later_lobe_idx = set_peak(0.00012, 1.00)

        strongest = select_direct_ir_peak(
            ir,
            start_time_s=start_time_s,
            sample_rate_hz=sample_rate_hz,
            threshold_fraction=0.10,
            reference_window_s=0.002,
            policy="strongest",
        )
        first_strong = select_direct_ir_peak(
            ir,
            start_time_s=start_time_s,
            sample_rate_hz=sample_rate_hz,
            threshold_fraction=0.10,
            reference_window_s=0.002,
            policy="first-strong",
            first_lobe_threshold_fraction=0.50,
            first_lobe_window_s=(-0.0005, 0.0008),
        )

        self.assertEqual(strongest["index"], later_lobe_idx)
        self.assertEqual(first_strong["index"], first_lobe_idx)
        self.assertEqual(first_strong["first_lobe_index"], first_lobe_idx)
        self.assertEqual(first_strong["reason"], "first strong near-reference lobe")

    def test_prefers_reference_window_peak_when_early_event_is_larger(self):
        sample_rate_hz = 48_000.0
        start_time_s = -0.020
        ir = np.zeros(2048)

        def set_peak(time_s: float, amplitude: float) -> int:
            idx = int(round((time_s - start_time_s) * sample_rate_hz))
            ir[idx] = amplitude
            return idx

        early_reflection_idx = set_peak(-0.0121, 1.00)
        direct_idx = set_peak(0.00012, 0.35)

        selection = select_direct_ir_peak(
            ir,
            start_time_s=start_time_s,
            sample_rate_hz=sample_rate_hz,
            threshold_fraction=0.10,
            reference_window_s=0.002,
            policy="strongest",
        )

        self.assertFalse(selection["rejected"])
        self.assertEqual(selection["global_index"], early_reflection_idx)
        self.assertEqual(selection["index"], direct_idx)
        self.assertEqual(selection["reason"], "strongest significant peak inside reference window")

    def test_first_strong_policy_rejects_without_strong_near_reference_lobe(self):
        sample_rate_hz = 48_000.0
        start_time_s = -0.020
        ir = np.zeros(2048)
        weak_idx = int(round((0.00012 - start_time_s) * sample_rate_hz))
        ir[weak_idx] = 0.30
        far_idx = int(round((0.006 - start_time_s) * sample_rate_hz))
        ir[far_idx] = 1.0

        selection = select_direct_ir_peak(
            ir,
            start_time_s=start_time_s,
            sample_rate_hz=sample_rate_hz,
            threshold_fraction=0.10,
            reference_window_s=0.002,
            policy="first-strong",
            first_lobe_threshold_fraction=0.50,
            first_lobe_window_s=(-0.0005, 0.0008),
        )

        self.assertTrue(selection["rejected"])
        self.assertIsNone(selection["index"])
        self.assertEqual(selection["reason"], "no first strong near-reference lobe")

    def test_rejects_when_only_large_peak_is_outside_reference_window(self):
        sample_rate_hz = 48_000.0
        start_time_s = -0.020
        ir = np.zeros(2048)
        far_reflection_idx = int(round((0.006 - start_time_s) * sample_rate_hz))
        ir[far_reflection_idx] = 1.0

        selection = select_direct_ir_peak(
            ir,
            start_time_s=start_time_s,
            sample_rate_hz=sample_rate_hz,
            threshold_fraction=0.10,
            reference_window_s=0.002,
            policy="strongest",
        )

        self.assertTrue(selection["rejected"])
        self.assertIsNone(selection["index"])
        self.assertEqual(selection["global_index"], far_reflection_idx)


if __name__ == "__main__":
    unittest.main()
