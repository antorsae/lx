import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from lx521_l22mg_baffle.metadata import load_measurement_notes, parse_height_reference
from polar_data_loader import PolarDataLoader


class MeasurementTimingMetadataTest(unittest.TestCase):
    def test_parse_height_reference_reads_juan_l22mg_lm_note(self):
        self.assertEqual(
            parse_height_reference("Measurement distance: 50 cm. Mic height: L22MG/LM."),
            "l22mg",
        )

    def test_direct_arrival_unsafe_angles_use_audited_peak_flags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "target.h5"
            with h5py.File(path, "w") as h5:
                h5.attrs["target_kind"] = "processed_measurement"
                angles = h5.create_group("L22MG").create_group("angles")
                clean = angles.create_group("70")
                clean.attrs["notes"] = "1m, altura UM\nusing IR start time"
                clean.attrs["timing_peak_time_ms"] = 0.128
                clean.attrs["timing_global_peak_time_ms"] = 0.128
                clean.attrs["timing_earliest_10pct_peak_time_ms"] = -12.142
                clean.attrs["timing_peak_policy"] = "first-strong"
                clean.attrs["timing_first_strong_near_ref_lobe_time_ms"] = 0.128
                clean.attrs["timing_selected_is_first_strong_near_ref_lobe"] = True
                clean.attrs["timing_mdat_window_ref_not_first_strong_near_ref_lobe"] = True
                clean.attrs["timing_mdat_window_ref_minus_first_strong_near_ref_lobe_path_mm"] = 28.6
                clean.attrs["timing_current_loader_selected_early_event"] = False
                clean.attrs["timing_current_loader_peak_rejected"] = False
                clean.attrs["timing_suspicious_window_ref_alignment"] = False

                early = angles.create_group("80")
                early.attrs["notes"] = "1m, altura UM\nusing IR start time"
                early.attrs["timing_current_loader_selected_early_event"] = True

                rejected = angles.create_group("90")
                rejected.attrs["notes"] = "1m, altura UM\nusing IR start time"
                rejected.attrs["timing_current_loader_peak_rejected"] = True

                window = angles.create_group("60")
                window.attrs["notes"] = "1m, altura UM\nusing estimated IR delay"
                window.attrs["timing_suspicious_window_ref_alignment"] = True

                lobe = angles.create_group("50")
                lobe.attrs["notes"] = "1m, altura UM\nusing estimated IR delay"
                lobe.attrs["timing_first_strong_near_ref_lobe_time_ms"] = 0.040
                lobe.attrs["timing_selected_is_first_strong_near_ref_lobe"] = False
                lobe.attrs["timing_selected_minus_first_strong_near_ref_lobe_path_mm"] = 28.6

                late = angles.create_group("40")
                late.attrs["notes"] = "1m, altura UM\nusing estimated IR delay"
                late.attrs["timing_late_window_peak_warning"] = True
                late.attrs["timing_late_window_peak_time_ms"] = 2.2

                strongest = angles.create_group("30")
                strongest.attrs["notes"] = "1m, altura UM\nusing estimated IR delay"
                strongest.attrs["timing_peak_policy"] = "strongest"
                strongest.attrs["timing_peak_time_ms"] = 0.123
                strongest.attrs["timing_global_peak_time_ms"] = 0.123
                strongest.attrs["timing_first_strong_near_ref_lobe_time_ms"] = 0.040

                ir_start = angles.create_group("20")
                ir_start.attrs["notes"] = "1m, altura UM\nusing IR start time"
                ir_start.attrs["timing_peak_policy"] = "ir-start"
                ir_start.attrs["timing_peak_time_ms"] = -0.250

            notes = load_measurement_notes(path, "L22MG", "angles")

        self.assertEqual(notes.ir_start_note_angles_deg, (20, 70, 80, 90))
        self.assertEqual(notes.peak_selected_early_event_angles_deg, (80,))
        self.assertEqual(notes.peak_rejected_angles_deg, (90,))
        self.assertEqual(notes.suspicious_window_ref_angles_deg, (60,))
        self.assertEqual(notes.selected_not_first_lobe_angles_deg, (50,))
        self.assertEqual(notes.mdat_window_ref_not_first_lobe_angles_deg, (70,))
        self.assertEqual(notes.peak_policy_unsafe_angles_deg, (30,))
        self.assertEqual(notes.late_window_warning_angles_deg, (40,))
        self.assertEqual(notes.direct_arrival_timing_unsafe_angles_deg, (30, 50, 60, 80, 90))
        self.assertNotIn(20, notes.direct_arrival_timing_unsafe_angles_deg)
        self.assertNotIn(70, notes.direct_arrival_timing_unsafe_angles_deg)
        self.assertNotIn(40, notes.direct_arrival_timing_unsafe_angles_deg)
        self.assertEqual(notes.target_kind, "processed_measurement")
        self.assertTrue(notes.acceptance_target_allowed)

    def test_diagnostic_hdf5_is_not_acceptance_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "target.h5"
            with h5py.File(path, "w") as h5:
                h5.attrs["target_kind"] = "raw_mdat_direct_gate_diagnostic"
                h5.attrs["diagnostic_only"] = True
                h5.attrs["not_acceptance_target"] = True
                angles = h5.create_group("L22MG").create_group("angles")
                angles.create_group("0").attrs["notes"] = "1m, altura UM"

            notes = load_measurement_notes(path, "L22MG", "angles")

        self.assertFalse(notes.acceptance_target_allowed)
        self.assertTrue(notes.diagnostic_only)
        self.assertTrue(notes.not_acceptance_target)
        self.assertIn("not_acceptance_target", notes.acceptance_target_reason)

    def test_hdf5_save_promotes_location_notes_to_structured_attrs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "target.h5"
            loader = PolarDataLoader(connect_to_rew=False)
            loader.save_to_hdf5(
                {
                    "L22MG": {
                        "has_rear": False,
                        "common_frequencies": np.array([300.0]),
                        "angles": {
                            0: {
                                "magnitude": np.array([85.0]),
                                "phase": np.array([0.0]),
                                "unit": "dB SPL",
                                "smoothing": "None",
                                "metadata": {
                                    "title": "F0-L22MG",
                                    "notes": "1m, altura UM\nMontado: L22MG, 10F8824, SEAS27T.",
                                    "sampleRate": 48000,
                                },
                            }
                        },
                    }
                },
                str(path),
            )

            with h5py.File(path, "r") as h5:
                attrs = h5["L22MG"]["angles"]["0"].attrs
                self.assertEqual(float(attrs["measurement_distance_m"]), 1.0)
                self.assertEqual(attrs["measurement_height_reference"], "um")
            notes = load_measurement_notes(path, "L22MG", "angles")

        self.assertEqual(notes.parsed_distance_m, 1.0)
        self.assertEqual(notes.parsed_height_reference, "um")
        self.assertIn("height=reference um", notes.summary)

    def test_load_measurement_notes_reads_passive_state_driver_attrs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "target.h5"
            with h5py.File(path, "w") as h5:
                driver = h5.create_group("L22MG")
                driver.attrs["passive_state_status"] = "unused_um_tweeter_state_unrecorded"
                driver.attrs["passive_state_evidence"] = "unknown passive geometry remains unknown"
                driver.attrs["passive_state_acceptance_use"] = (
                    "current_l22_target_but_passive_geometry_not_proven"
                )
                driver.attrs["passive_state_metadata_policy"] = (
                    "record uncertainty without guessing open/covered/mounted state"
                )
                angles = driver.create_group("angles")
                angles.create_group("0").attrs["notes"] = (
                    "Measurement distance: 50 cm. Mic height: L22MG/LM."
                )

            notes = load_measurement_notes(path, "L22MG", "angles")

        self.assertEqual(notes.passive_state_status, "unused_um_tweeter_state_unrecorded")
        self.assertEqual(
            notes.passive_state_acceptance_use,
            "current_l22_target_but_passive_geometry_not_proven",
        )
        self.assertIn("unknown passive geometry", notes.passive_state_evidence)

    def test_strongest_policy_hdf5_is_marked_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "target.h5"
            with patch("config.ALLOW_UNSAFE_STRONGEST_IR_PEAK_POLICY", True):
                loader = PolarDataLoader(
                    connect_to_rew=False,
                    direct_ir_peak_policy="strongest",
                )
            loader.save_to_hdf5(
                {
                    "L22MG": {
                        "has_rear": False,
                        "common_frequencies": np.array([300.0]),
                        "angles": {
                            70: {
                                "magnitude": np.array([70.0]),
                                "phase": np.array([0.0]),
                                "unit": "dB SPL",
                                "smoothing": "None",
                            }
                        },
                    }
                },
                str(path),
            )

            with h5py.File(path, "r") as h5:
                self.assertEqual(h5.attrs["direct_ir_peak_policy"], "strongest")
                self.assertEqual(h5.attrs["target_kind"], "legacy_strongest_lobe_diagnostic")
                self.assertTrue(bool(h5.attrs["diagnostic_only"]))
                self.assertTrue(bool(h5.attrs["not_acceptance_target"]))


if __name__ == "__main__":
    unittest.main()
