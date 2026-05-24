#!/usr/bin/env python3
"""
Polar Response Data Loader for LX521 Driver Measurements

This module loads acoustic measurements from REW .mdat files via the REW API,
applies time gating and smoothing in Python, and organizes them into a
structured format for polar response analysis.

Refactored to minimize REW API processing dependency.

Author: Andres Torrubia
Date: 2025-11-23
"""

import os
import re
import requests
import base64
import struct
import numpy as np
import h5py
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import config
from lx521_l22mg_baffle.metadata import parse_distance_m, parse_height_m, parse_height_reference


# ==================== Filename Pattern Definitions ====================
# "scanspeak" is the same naming convention as "juan"
_PATTERN_ALIASES = {
    "scanspeak": "juan",
}

_ANGLE_METADATA_ATTRS = (
    "title",
    "notes",
    "date",
    "measurement_distance_m",
    "measurement_height_m",
    "measurement_height_reference",
)

_PATTERN_DEFS = {
    "andres": {
        "regex": re.compile(r"^F(?P<angle>\d+)-(?P<driver>.+)$"),
        "side_from_match": lambda m: "F",
        "filename": lambda driver, angle, side: f"F{angle}-{driver}.mdat",
    },
    "juan": {
        "regex": re.compile(r"^(?P<driver>.+)\s+(?P<angle>\d+)\s+(?P<side>[FR])$"),
        "side_from_match": lambda m: m.group("side"),
        "filename": lambda driver, angle, side: f"{driver} {angle} {side}.mdat",
    },
    "juan_suffix": {
        "regex": re.compile(r"^(?P<driver>.+?)\s+(?P<angle>\d+)\s+(?P<side>[FR])(?:\s+.+)?$"),
        "side_from_match": lambda m: m.group("side"),
        "filename": lambda driver, angle, side: f"{driver} {angle} {side}.mdat",
    },
    "lx521_system": {
        "regex": re.compile(r"^(?P<driver>.+)\s+(?P<angle>\d+)\s+GRADOS\s+(?P<side>F|REAR)$"),
        "side_from_match": lambda m: "R" if m.group("side") == "REAR" else "F",
        "filename": lambda driver, angle, side: (
            f"{driver} {angle} GRADOS {'REAR' if side == 'R' else 'F'}.mdat"
        ),
    },
}


def select_direct_ir_peak(
    abs_ir: np.ndarray,
    start_time_s: float,
    sample_rate_hz: float,
    threshold_fraction: float = 0.10,
    reference_window_s: float = 0.002,
    policy: str = "first-strong",
    first_lobe_threshold_fraction: float = 0.50,
    first_lobe_window_s: tuple[float, float] = (-0.0005, 0.0008),
) -> Dict:
    """Select the direct-arrival IR peak near the acoustic timing reference."""

    abs_ir = np.asarray(abs_ir, dtype=float)
    policy = str(policy).replace("_", "-")
    if policy not in {"strongest", "first-strong"}:
        raise ValueError(f"Unknown direct IR peak policy: {policy}")

    global_max_idx = int(np.nanargmax(abs_ir))
    global_max_val = float(abs_ir[global_max_idx])
    global_peak_time_s = float(start_time_s) + (global_max_idx / float(sample_rate_hz))

    threshold = global_max_val * float(threshold_fraction)
    local_peak_mask = np.zeros_like(abs_ir, dtype=bool)
    local_peak_mask[1:-1] = (abs_ir[1:-1] > abs_ir[:-2]) & (abs_ir[1:-1] > abs_ir[2:])
    significant_peaks = np.where(local_peak_mask & (abs_ir > threshold))[0]
    peak_times_s = float(start_time_s) + (significant_peaks / float(sample_rate_hz))
    reference_window = np.abs(peak_times_s) <= float(reference_window_s)
    first_lobe_threshold = global_max_val * float(first_lobe_threshold_fraction)
    first_lobe_start_s, first_lobe_end_s = first_lobe_window_s
    first_lobe_window = (
        (peak_times_s >= float(first_lobe_start_s))
        & (peak_times_s <= float(first_lobe_end_s))
        & (abs_ir[significant_peaks] >= first_lobe_threshold)
    )
    first_lobe_peaks = significant_peaks[first_lobe_window]
    first_lobe_idx = int(first_lobe_peaks[np.argmin(peak_times_s[first_lobe_window])]) if len(first_lobe_peaks) else None

    base = {
        "global_index": global_max_idx,
        "global_peak_time_s": global_peak_time_s,
        "significant_peaks": significant_peaks,
        "policy": policy,
        "first_lobe_index": first_lobe_idx,
        "first_lobe_threshold_fraction": float(first_lobe_threshold_fraction),
        "first_lobe_window_s": (float(first_lobe_start_s), float(first_lobe_end_s)),
    }

    if policy == "first-strong":
        if first_lobe_idx is not None:
            return {
                **base,
                "index": first_lobe_idx,
                "reason": "first strong near-reference lobe",
                "rejected": False,
            }
        return {
            **base,
            "index": None,
            "reason": "no first strong near-reference lobe",
            "rejected": True,
        }

    if np.any(reference_window):
        window_peaks = significant_peaks[reference_window]
        target_peak_idx = int(window_peaks[np.argmax(abs_ir[window_peaks])])
        return {
            **base,
            "index": target_peak_idx,
            "reason": "strongest significant peak inside reference window",
            "rejected": False,
        }
    if abs(global_peak_time_s) <= float(reference_window_s):
        return {
            **base,
            "index": global_max_idx,
            "reason": "global peak inside reference window",
            "rejected": False,
        }
    return {
        **base,
        "index": None,
        "reason": "no significant reference-window peak; global peak outside reference window",
        "rejected": True,
    }


class PolarDataLoader:
    """Load and manage polar response measurements from REW"""

    def __init__(
        self,
        data_directory: str = ".",
        connect_to_rew: bool = True,
        pattern_type: str = "andres",
        direct_ir_peak_policy: str | None = None,
        driver_name_aliases: Dict[str, str] | None = None,
    ):
        """
        Initialize data loader

        Args:
            data_directory: Path to directory containing .mdat files
            connect_to_rew: Whether to verify/launch REW API connection (default: True)
            pattern_type: Filename pattern type ("andres" or "juan")
        """
        self.data_dir = Path(data_directory)
        self.measurements = {}
        self._rew_launch_attempted = False
        self.pattern_type = pattern_type
        self.direct_ir_peak_policy = str(direct_ir_peak_policy or config.DIRECT_IR_PEAK_POLICY).replace("_", "-")
        if self.direct_ir_peak_policy not in {"strongest", "first-strong", "ir-start"}:
            raise ValueError(f"Unknown direct IR peak policy: {self.direct_ir_peak_policy}")
        if (
            self.direct_ir_peak_policy == "strongest"
            and not config.ALLOW_UNSAFE_STRONGEST_IR_PEAK_POLICY
        ):
            raise ValueError(
                "direct_ir_peak_policy='strongest' is unsafe for high-angle validation because "
                "the absolute peak can be a reflected/scattered lobe rather than the direct "
                "arrival. Use the default 'first-strong' policy, or set "
                "ALLOW_UNSAFE_STRONGEST_IR_PEAK_POLICY=1 only for legacy diagnostics."
            )
        self._file_index = None
        if self._get_pattern_def() is None:
            valid = sorted(set(_PATTERN_DEFS.keys()) | set(_PATTERN_ALIASES.keys()))
            valid_list = ", ".join(valid)
            raise ValueError(
                f"Unknown pattern_type '{pattern_type}'. Expected one of: {valid_list}"
            )
        self._driver_name_aliases = {
            **getattr(config, "DRIVER_NAME_ALIASES", {}),
            **(driver_name_aliases or {}),
        }
        self._driver_name_reverse = {}
        for raw_name, canonical_name in self._driver_name_aliases.items():
            raw = raw_name.strip()
            canonical = canonical_name.strip()
            self._driver_name_reverse.setdefault(canonical, raw)

        if connect_to_rew:
            if not self._ensure_rew_running():
                raise RuntimeError("REW API is not accessible. Please ensure REW is running and the API server is started (Preferences -> API).")
            self._enable_blocking_mode()

    def _ensure_rew_running(self):
        """Check if REW is running, launch it if not (only once)"""
        url = f"{config.REW_API_BASE}/measurements"

        # Initial check - robust retry loop
        for i in range(3):
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                # If we get here, it works
                if i > 0:
                    print("✓ REW API connected after retry")
                else:
                    print("✓ REW API is accessible")
                return True
            except requests.exceptions.RequestException:
                if i < 2:
                    time.sleep(1)

        # API is not responding.
        # Attempt to launch/activate REW with API enabled (even if running)
        if not self._rew_launch_attempted:
            self._rew_launch_attempted = True
            print("REW API not responding. Attempting to launch REW with API enabled...")
            try:
                # Launch REW with API (MacOS specific per request)
                subprocess.Popen(
                    ["open", "-a", "REW.app", "--args", "-api"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print("Waiting for REW to start (15 seconds)...")
                time.sleep(15)

                # Retry connection loop
                for i in range(5):
                    try:
                        response = requests.get(url, timeout=5)
                        response.raise_for_status()
                        print("✓ REW started successfully and API is accessible")
                        return True
                    except requests.exceptions.RequestException:
                        print(f"  Waiting for API... ({i+1}/5)")
                        time.sleep(3)

                print("WARNING: REW launched but API not responding.")
                return False

            except Exception as e:
                print(f"ERROR: Failed to launch REW: {e}")
                return False
        else:
            print("WARNING: REW API is not accessible.")
            return False

    def _enable_blocking_mode(self):
        """Enable blocking mode in REW API for synchronous operations"""
        try:
            url = f"{config.REW_API_BASE}/application/blocking"
            response = requests.post(url, json=True, timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not enable blocking mode: {e}")

    def unload_measurement(self, uuid: str) -> bool:
        """Unload a measurement from REW memory (frees slot, file stays on disk).

        Args:
            uuid: The UUID of the measurement to unload

        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{config.REW_API_BASE}/measurements/{uuid}"
            response = requests.delete(url, timeout=10)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to unload measurement {uuid}: {e}")
            return False

    def unload_all_measurements(self) -> bool:
        """Unload ALL measurements from REW memory.

        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{config.REW_API_BASE}/measurements"
            response = requests.delete(url, timeout=10)
            if response.status_code == 200:
                print("✓ Unloaded all measurements from REW")
                return True
            return False
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to unload all measurements: {e}")
            return False

    def get_measurement_count(self) -> int:
        """Get the current number of measurements loaded in REW.

        Returns:
            Number of measurements, or -1 on error
        """
        try:
            url = f"{config.REW_API_BASE}/measurements"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            measurements = response.json()
            return len(measurements)
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not get measurement count: {e}")
            return -1

    def _decode_base64_floats(self, base64_string: str) -> np.ndarray:
        """Decode Base64-encoded float array from REW API"""
        byte_data = base64.b64decode(base64_string)
        num_floats = len(byte_data) // 4
        floats = struct.unpack(f'>{num_floats}f', byte_data)
        return np.array(floats)

    def _set_ir_window(
        self,
        measurement_uuid: str,
        left_ms: float,
        right_ms: float,
        ref_time_ms: Optional[float] = None,
    ):
        """Set IR window settings via REW API.

        When a direct-arrival peak has been selected, force REW's window
        reference to that peak instead of inheriting a possibly stale ref time.
        """
        url = f"{config.REW_API_BASE}/measurements/{measurement_uuid}/ir-windows"

        try:
            # 1. Get current settings
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            current_settings = response.json()

            # 2. Update widths
            current_settings["leftWindowWidthms"] = left_ms
            current_settings["rightWindowWidthms"] = right_ms
            if ref_time_ms is not None:
                current_settings["refTimems"] = float(ref_time_ms)
            # Ensure we use Tukey 0.25 as desired/default if not set
            if "leftWindowType" not in current_settings:
                 current_settings["leftWindowType"] = "Tukey 0.25"
            if "rightWindowType" not in current_settings:
                 current_settings["rightWindowType"] = "Tukey 0.25"

            # 3. Post back
            response = requests.post(url, json=current_settings, timeout=10)
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            print(f"    Warning: Failed to set IR window: {e}")
            # Fallback to simple set if GET failed (unlikely but safe)
            payload = {
                "leftWindowType": "Tukey 0.25",
                "rightWindowType": "Tukey 0.25",
                "leftWindowWidthms": left_ms,
                "rightWindowWidthms": right_ms,
                "refTimems": float(ref_time_ms) if ref_time_ms is not None else 0,
                "addFDW": False
            }
            requests.post(url, json=payload, timeout=10)

    def _get_smoothing_choices(self) -> List[str]:
        """Fetch valid smoothing choices from REW API"""
        url = f"{config.REW_API_BASE}/measurements/frequency-response/smoothing-choices"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"    Warning: Could not fetch smoothing choices: {e}")
            return []

    def _get_frequency_response(self, measurement_uuid: str, smoothing: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get frequency response from REW API"""

        # Determine the correct smoothing string
        if smoothing:
            smoothing_str = f"1/{smoothing}"
        else:
            # We want "No Smoothing"
            choices = self._get_smoothing_choices()
            # Common variants for "No Smoothing" in REW
            candidates = ["No smoothing", "None", "0", ""]

            smoothing_str = "None" # Default fallback

            # Case-insensitive match from available choices
            for choice in choices:
                if choice.lower() in [c.lower() for c in candidates]:
                    smoothing_str = choice
                    break

            if not choices:
                 print("    Warning: Using default 'None' for no smoothing (could not verify choices).")

        # print(f"    Requesting smoothing: '{smoothing_str}'")

        url = f"{config.REW_API_BASE}/measurements/{measurement_uuid}/frequency-response"
        params = {"smoothing": smoothing_str}

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        magnitude = self._decode_base64_floats(data["magnitude"])
        # API returns phase in "phase" field
        if "phase" in data:
            phase = self._decode_base64_floats(data["phase"])
        else:
            phase = np.zeros_like(magnitude)

        # Reconstruct frequency array
        start_freq = data["startFreq"]
        num_points = len(magnitude)

        if "ppo" in data and data["ppo"]:
            # Log-spaced data
            ppo = data["ppo"]
            indices = np.arange(num_points)
            frequencies = start_freq * np.exp(indices * np.log(2) / ppo)
        elif "freqStep" in data:
            # Linear-spaced data
            freq_step = data["freqStep"]
            frequencies = start_freq + np.arange(num_points) * freq_step
        else:
            # Fallback if neither (shouldn't happen with REW API)
            print("Warning: Could not determine frequency spacing, assuming linear 1Hz")
            frequencies = start_freq + np.arange(num_points)

        return frequencies, magnitude, phase

    def _auto_fix_timing(self, measurement_uuid: str, measurement_metadata: Optional[Dict] = None) -> dict:
        """
        Detect and fix timing anomalies.
        Prefer the first strong lobe near the acoustic timing reference. This
        avoids aligning t=0 to early pre-response artifacts or later/larger
        high-angle lobes.

        Returns:
            dict with 'corrected' (bool) and 'offset_ms' (float) if correction was applied
        """
        result = {'corrected': False, 'offset_ms': 0.0}
        if self.direct_ir_peak_policy == "ir-start":
            meta = measurement_metadata or {}
            if "timeOfIRStartSeconds" not in meta:
                result["peak_selection_failed"] = True
                result["timing_error"] = "REW metadata did not include timeOfIRStartSeconds"
                return result
            try:
                ir_start_ms = float(meta["timeOfIRStartSeconds"]) * 1000.0
            except (TypeError, ValueError) as exc:
                result["peak_selection_failed"] = True
                result["timing_error"] = f"Invalid REW timeOfIRStartSeconds: {exc}"
                return result
            if not np.isfinite(ir_start_ms):
                result["peak_selection_failed"] = True
                result["timing_error"] = "REW timeOfIRStartSeconds is not finite"
                return result
            result["selected_peak_time_ms"] = ir_start_ms
            result["selected_peak_reason"] = "REW stored IR start time"
            result["direct_ir_peak_policy"] = "ir-start"
            result["first_lobe_time_ms"] = float("nan")
            result["selected_is_first_lobe"] = False
            result["first_lobe_threshold_fraction"] = float("nan")
            result["first_lobe_window_start_ms"] = float("nan")
            result["first_lobe_window_end_ms"] = float("nan")
            result["rew_ir_start_time_ms"] = ir_start_ms
            return result
        try:
            # 1. Get IR Data
            url_ir = f"{config.REW_API_BASE}/measurements/{measurement_uuid}/impulse-response"
            response = requests.get(url_ir, params={"windowed": "false"}, timeout=10)
            response.raise_for_status()
            ir_info = response.json()

            if "data" in ir_info:
                ir_data = self._decode_base64_floats(ir_info["data"])
            elif "left" in ir_info:
                ir_data = self._decode_base64_floats(ir_info["left"])
            else:
                result["peak_selection_failed"] = True
                result["timing_error"] = "Impulse response payload did not contain data or left channel"
                return result

            sample_rate = ir_info["sampleRate"]
            start_time_s = ir_info["startTime"]

            # 2. Analyze Peaks
            abs_ir = np.abs(ir_data)

            # Prefer a strong peak close to the acoustic timing reference. The
            # previous "earliest >10% of max" rule could grab a pre-response
            # artifact at deep side-null angles and then gate around that event.
            peak_selection = select_direct_ir_peak(
                abs_ir,
                start_time_s,
                sample_rate,
                policy=self.direct_ir_peak_policy,
                first_lobe_threshold_fraction=config.DIRECT_IR_FIRST_LOBE_THRESHOLD_FRACTION,
                first_lobe_window_s=(
                    config.DIRECT_IR_FIRST_LOBE_START_MS / 1000.0,
                    config.DIRECT_IR_FIRST_LOBE_END_MS / 1000.0,
                ),
            )
            if peak_selection["rejected"]:
                if peak_selection["reason"] == "no first strong near-reference lobe":
                    detail = (
                        "no peak above "
                        f"{config.DIRECT_IR_FIRST_LOBE_THRESHOLD_FRACTION:.2f}x global peak "
                        f"inside {config.DIRECT_IR_FIRST_LOBE_START_MS:.2f}.."
                        f"{config.DIRECT_IR_FIRST_LOBE_END_MS:.2f} ms"
                    )
                else:
                    detail = (
                        f"global peak at {peak_selection['global_peak_time_s']*1000:.2f} ms "
                        "because it may be a reflection"
                    )
                print(
                    "    !! WARNING: No acceptable direct IR lobe found near the acoustic timing "
                    "reference; refusing to align to uncertain timing "
                    f"({detail})."
                )
                result["peak_selection_failed"] = True
                result["global_peak_time_ms"] = peak_selection["global_peak_time_s"] * 1000.0
                result["selected_peak_reason"] = str(peak_selection["reason"])
                return result

            target_peak_idx = int(peak_selection["index"])
            peak_reason = str(peak_selection["reason"])
            global_peak_time_s = float(peak_selection["global_peak_time_s"])
            first_lobe_idx = peak_selection.get("first_lobe_index")
            significant_peaks = np.asarray(peak_selection["significant_peaks"], dtype=int)
            if len(significant_peaks) > 0:
                earliest_idx = int(significant_peaks[0])
                earliest_time_s = start_time_s + (earliest_idx / sample_rate)
                target_time_preview_s = start_time_s + (target_peak_idx / sample_rate)
                if earliest_time_s < target_time_preview_s - 0.005:
                    print(
                        "    Timing note: ignored early significant IR event at "
                        f"{earliest_time_s*1000:.2f} ms; using peak near reference at "
                        f"{target_time_preview_s*1000:.2f} ms."
                    )

            # Calculate time of the target peak
            peak_time_s = start_time_s + (target_peak_idx / sample_rate)
            result["selected_peak_time_ms"] = peak_time_s * 1000.0
            result["selected_peak_reason"] = peak_reason
            result["global_peak_time_ms"] = global_peak_time_s * 1000.0
            result["direct_ir_peak_policy"] = str(peak_selection["policy"])
            result["first_lobe_threshold_fraction"] = float(
                peak_selection["first_lobe_threshold_fraction"]
            )
            first_lobe_window_s = peak_selection["first_lobe_window_s"]
            result["first_lobe_window_start_ms"] = float(first_lobe_window_s[0]) * 1000.0
            result["first_lobe_window_end_ms"] = float(first_lobe_window_s[1]) * 1000.0
            if first_lobe_idx is not None:
                first_lobe_time_s = start_time_s + (int(first_lobe_idx) / sample_rate)
                result["first_lobe_time_ms"] = first_lobe_time_s * 1000.0
                result["selected_is_first_lobe"] = int(first_lobe_idx) == target_peak_idx
            else:
                result["first_lobe_time_ms"] = float("nan")
                result["selected_is_first_lobe"] = False

            # If peak is not at 0 (tolerance 0.5ms), shift it
            if abs(peak_time_s) > 0.0005:
                # To shift Peak to 0, we set t=0 TO the current Peak Time.
                # REW 'Offset t=0' parameter is "time to become zero".
                shift_sec = peak_time_s

                print(f"    ! Aligning Peak: Found at {peak_time_s*1000:.2f} ms. Applying Offset {shift_sec*1000:.2f} ms...")

                url_cmd = f"{config.REW_API_BASE}/measurements/{measurement_uuid}/command"
                payload = {
                    "command": "Offset t=0",
                    "parameters": {
                        "offset": str(shift_sec),
                        "unit": "seconds"
                    }
                }
                resp_cmd = requests.post(url_cmd, json=payload, timeout=10)
                resp_cmd.raise_for_status()

                # Verify the shift
                time.sleep(0.5) # Wait for command to apply
                resp_verify = requests.get(url_ir, params={"windowed": "false"}, timeout=10)
                if resp_verify.ok:
                    info_v = resp_verify.json()
                    start_v = info_v["startTime"]
                    peak_time_v = start_v + (target_peak_idx / sample_rate)
                    print(f"    ✓ Verification: Peak is now at {peak_time_v*1000:.2f} ms")

                    if abs(peak_time_v) > 0.0001: # If still > 0.1ms off
                        print("    !! WARNING: Correction failed to move peak to 0. Result is still off.")
                        result["peak_selection_failed"] = True
                        result["timing_error"] = "Peak offset verification failed"
                        return result
                else:
                    result["peak_selection_failed"] = True
                    result["timing_error"] = "Could not verify peak offset"
                    return result

                # Reset Ref Time to 0 because the selected direct peak has
                # just been shifted to t=0.
                url_win = f"{config.REW_API_BASE}/measurements/{measurement_uuid}/ir-windows"
                resp_win = requests.get(url_win)
                if resp_win.ok:
                    settings = resp_win.json()
                    settings["refTimems"] = 0
                    requests.post(url_win, json=settings, timeout=10)

                # Record that correction was applied
                result['corrected'] = True
                result['offset_ms'] = shift_sec * 1000
                result["selected_peak_time_after_correction_ms"] = 0.0

        except Exception as e:
            print(f"    Warning: Failed to auto-fix timing: {e}")
            result["peak_selection_failed"] = True
            result["timing_error"] = str(e)

        return result

    def load_measurement(self, file_path: str, smoothing: Optional[int] = 12,
                        gate_left_ms: float = 0.0, gate_right_ms: float = 3.0) -> Dict:
        """
        Load a single measurement file, gate it using REW API, and retrieve smoothed response.
        """
        # Normalize path
        file_path = str(Path(file_path).absolute()).replace("\\", "/")

        for attempt in range(3):
            try:
                # Load file via API
                url = f"{config.REW_API_BASE}/measurements/command"
                payload = {"command": "Load", "parameters": [file_path]}
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()

                # Get measurements list to find the UUID
                url = f"{config.REW_API_BASE}/measurements"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                measurements = response.json()

                measurement_keys = sorted(measurements.keys(), key=int)
                last_key = measurement_keys[-1]
                measurement_uuid = measurements[last_key]["uuid"]

                # 0. Auto-fix timing anomalies
                timing_correction = self._auto_fix_timing(measurement_uuid, measurements[last_key])
                if timing_correction.get("peak_selection_failed", False):
                    detail = timing_correction.get("timing_error") or (
                        f"global peak at {timing_correction.get('global_peak_time_ms'):.2f} ms"
                        if "global_peak_time_ms" in timing_correction
                        else "unknown timing failure"
                    )
                    raise RuntimeError(
                        "Could not identify a direct IR peak near the acoustic timing reference "
                        f"for {file_path}; refusing to generate frequency response from uncertain timing "
                        f"({detail})."
                    )

                # 1. Apply Time Gating via REW API
                if gate_right_ms > 0 or gate_left_ms > 0:
                    # print(f"    Gating (REW): {gate_left_ms}ms / {gate_right_ms}ms")
                    gate_ref_time_ms = (
                        0.0
                        if timing_correction.get("corrected", False)
                        else timing_correction.get("selected_peak_time_ms")
                    )
                    self._set_ir_window(
                        measurement_uuid,
                        gate_left_ms,
                        gate_right_ms,
                        ref_time_ms=gate_ref_time_ms,
                    )

                # 2. Get Frequency Response from REW (Smoothing applied by REW)
                # if smoothing:
                #     print(f"    Smoothing (REW): 1/{smoothing} octave")

                frequencies, magnitude, phase = self._get_frequency_response(measurement_uuid, smoothing)

                return {
                    "frequencies": frequencies,
                    "magnitude": magnitude,
                    "phase": phase,
                    "unit": "dB SPL",
                    "smoothing": f"1/{smoothing}" if smoothing else "None",
                    "metadata": measurements[last_key],
                    "timing_corrected": timing_correction['corrected'],
                    "timing_offset_ms": timing_correction['offset_ms'],
                    "timing_peak_time_ms": timing_correction.get("selected_peak_time_ms", 0.0),
                    "timing_peak_selection_reason": timing_correction.get("selected_peak_reason", ""),
                    "timing_peak_policy": timing_correction.get("direct_ir_peak_policy", self.direct_ir_peak_policy),
                    "timing_first_strong_near_ref_lobe_time_ms": timing_correction.get("first_lobe_time_ms", float("nan")),
                    "timing_selected_is_first_strong_near_ref_lobe": timing_correction.get("selected_is_first_lobe", False),
                    "timing_first_lobe_threshold_fraction": timing_correction.get("first_lobe_threshold_fraction", config.DIRECT_IR_FIRST_LOBE_THRESHOLD_FRACTION),
                    "timing_first_lobe_window_start_ms": timing_correction.get("first_lobe_window_start_ms", config.DIRECT_IR_FIRST_LOBE_START_MS),
                    "timing_first_lobe_window_end_ms": timing_correction.get("first_lobe_window_end_ms", config.DIRECT_IR_FIRST_LOBE_END_MS),
                    "timing_rew_ir_start_time_ms": timing_correction.get(
                        "rew_ir_start_time_ms",
                        float(measurements[last_key].get("timeOfIRStartSeconds", float("nan"))) * 1000.0,
                    ),
                    "_uuid": measurement_uuid  # For tracking/unloading
                }
            except requests.exceptions.RequestException as e:
                print(f"    Warning: API request failed (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise e

    def load_all_drivers(self, driver_list: List[str] = None,
                        angles: List[int] = None,
                        smoothing: int = 12,
                        gate_left_ms: float = 0.0,
                        gate_right_ms: float = 3.0,
                        include_rear: bool = False,
                        batch_unload: bool = True) -> Dict:
        """
        Load complete polar measurements for all drivers

        Args:
            driver_list: List of drivers to load (auto-detected if None)
            angles: List of angles to load (auto-detected if None)
            smoothing: Smoothing factor (0 for none, 12 for 1/12 octave)
            gate_left_ms: Left gate time in ms
            gate_right_ms: Right gate time in ms
            include_rear: Whether to load rear measurements (if available)
            batch_unload: Unload measurements from REW after each driver (default: True)
                          This prevents hitting REW's ~100 measurement slot limit.
        """
        if driver_list is None:
            driver_list = self._detect_drivers()

        print(f"Loading polar data for {len(driver_list)} drivers...")
        display_names = [self._normalize_driver_name(name) for name in driver_list]
        print(f"Drivers: {', '.join(display_names)}")
        print(f"Pattern type: {self.pattern_type}")
        print(f"Gating: {gate_left_ms}ms / {gate_right_ms}ms")
        if smoothing:
            print(f"Smoothing: 1/{smoothing} octave")
        else:
            print("Smoothing: None")
        if include_rear:
            print("Including rear measurements")
        if batch_unload:
            print("Batch unload: ON (measurements unloaded after each driver)")
        print("=" * 60)

        all_data = {}

        for driver in driver_list:
            canonical = self._normalize_driver_name(driver)
            if canonical in all_data:
                raise ValueError(
                    f"Driver name collision after normalization: '{canonical}'."
                )
            label = canonical if canonical == driver else f"{canonical} (files: {driver})"
            print(f"\nLoading driver: {label}")
            driver_data = self.load_driver_polar_set(
                driver, angles, smoothing, gate_left_ms, gate_right_ms, include_rear
            )
            all_data[canonical] = driver_data

            # Unload this driver's measurements to free REW slots
            if batch_unload:
                loaded_uuids = driver_data.get("_loaded_uuids", [])
                if loaded_uuids:
                    unload_count = 0
                    for uuid in loaded_uuids:
                        if self.unload_measurement(uuid):
                            unload_count += 1
                    print(f"  ✓ Unloaded {unload_count}/{len(loaded_uuids)} measurements from REW")

        print("\n" + "=" * 60)
        return all_data

    def load_driver_polar_set(self, driver_name: str, angles: List[int] = None,
                               smoothing: int = 12,
                               gate_left_ms: float = 0.0,
                               gate_right_ms: float = 3.0,
                               include_rear: bool = False) -> Dict:
        """Load complete polar measurement set for a single driver

        Args:
            driver_name: Name of the driver
            angles: List of angles to load (auto-detected if None)
            smoothing: Smoothing factor (0 for none, 12 for 1/12 octave)
            gate_left_ms: Left gate time in ms
            gate_right_ms: Right gate time in ms
            include_rear: Whether to load rear measurements (if available)

        Returns:
            Dict with driver name, angles, and optionally rear_angles
        """
        canonical_name = self._normalize_driver_name(driver_name)
        # Auto-detect angles if not provided
        if angles is None:
            angles = self._detect_angles(driver_name, "F")
            if not angles:
                angles = list(range(0, 91, 10))  # Fallback default

        polar_data = {"driver": canonical_name, "angles": {}, "has_rear": False}
        loaded_uuids = []  # Track UUIDs for later unloading

        # Load front measurements
        for angle in angles:
            file_path = self._find_measurement_file(driver_name, angle, "F")
            filename = (
                file_path.relative_to(self.data_dir)
                if file_path.exists()
                else self._get_filename(driver_name, angle, "F")
            )

            if not file_path.exists():
                print(f"Warning: File not found: {filename}")
                continue

            print(f"  Loading {canonical_name} at {angle}° (Front)...")
            measurement = self.load_measurement(
                str(file_path), smoothing, gate_left_ms, gate_right_ms
            )
            polar_data["angles"][angle] = measurement
            # Track UUID for unloading
            if "_uuid" in measurement:
                loaded_uuids.append(measurement["_uuid"])
            time.sleep(2.0)

        # Load rear measurements if requested
        if include_rear:
            rear_angles = self._detect_angles(driver_name, "R")
            if rear_angles:
                polar_data["rear_angles"] = {}
                polar_data["has_rear"] = True

                for angle in rear_angles:
                    file_path = self._find_measurement_file(driver_name, angle, "R")
                    filename = (
                        file_path.relative_to(self.data_dir)
                        if file_path.exists()
                        else self._get_filename(driver_name, angle, "R")
                    )

                    if not file_path.exists():
                        print(f"Warning: File not found: {filename}")
                        continue

                    print(f"  Loading {canonical_name} at {angle}° (Rear)...")
                    measurement = self.load_measurement(
                        str(file_path), smoothing, gate_left_ms, gate_right_ms
                    )
                    polar_data["rear_angles"][angle] = measurement
                    # Track UUID for unloading
                    if "_uuid" in measurement:
                        loaded_uuids.append(measurement["_uuid"])
                    time.sleep(2.0)

        # Store loaded UUIDs in the polar_data dict
        polar_data["_loaded_uuids"] = loaded_uuids

        # Common frequency grid (using on-axis as reference)
        if polar_data["angles"]:
            ref_angle = sorted(polar_data["angles"].keys())[0]
            polar_data["common_frequencies"] = polar_data["angles"][ref_angle]["frequencies"]

        return polar_data

    def _iter_mdat_files(self) -> List[Path]:
        """Return all .mdat files below the source directory."""
        return sorted(self.data_dir.rglob("*.mdat"))

    def _get_file_index(self) -> Dict[Tuple[str, int, str], Path]:
        """Index parsed measurement files by raw driver name, angle, and side."""
        if self._file_index is not None:
            return self._file_index

        index = {}
        for file_path in self._iter_mdat_files():
            parsed = self._parse_filename(file_path.name)
            if not parsed:
                continue

            key = (
                self._normalize_driver_name(parsed["driver"]),
                parsed["angle"],
                parsed["side"],
            )
            if key in index:
                rel_existing = index[key].relative_to(self.data_dir)
                rel_new = file_path.relative_to(self.data_dir)
                print(
                    f"Warning: Duplicate measurement for {key}: "
                    f"{rel_existing} and {rel_new}. Using {rel_existing}."
                )
                continue

            index[key] = file_path

        self._file_index = index
        return index

    def _find_measurement_file(self, driver_name: str, angle: int, side: str = "F") -> Path:
        """Find a measurement path, including files in nested source folders."""
        file_driver_name = self._normalize_driver_name(driver_name)
        indexed_path = self._get_file_index().get((file_driver_name, angle, side))
        if indexed_path is not None:
            return indexed_path
        return self.data_dir / self._get_filename(driver_name, angle, side)

    def _parse_filename(self, filename: str) -> Optional[Dict]:
        """Parse measurement filename based on pattern type.

        Returns dict with: driver, angle, side ('F' or 'R'), or None if no match
        """
        stem = Path(filename).stem
        pattern_def = self._get_pattern_def()
        if not pattern_def:
            return None

        match = pattern_def["regex"].match(stem)
        if not match:
            return None

        return {
            "driver": match.group("driver"),
            "angle": int(match.group("angle")),
            "side": pattern_def["side_from_match"](match),
        }

    def _get_pattern_def(self) -> Optional[Dict]:
        """Resolve pattern type (with aliases) to its definition."""
        resolved = _PATTERN_ALIASES.get(self.pattern_type, self.pattern_type)
        return _PATTERN_DEFS.get(resolved)

    def _normalize_driver_name(self, driver_name: str) -> str:
        return self._driver_name_aliases.get(driver_name, driver_name).strip()

    def _resolve_driver_name_for_files(self, driver_name: str) -> str:
        if driver_name in self._driver_name_aliases:
            return driver_name
        return self._driver_name_reverse.get(driver_name, driver_name)

    def _detect_drivers(self) -> List[str]:
        """Auto-detect driver names from .mdat files"""
        drivers = set()

        for driver, _angle, _side in self._get_file_index().keys():
            # Skip combination measurements for andres pattern
            if self.pattern_type == "andres" and "con" in driver.lower():
                continue
            drivers.add(driver)

        return sorted(list(drivers))

    def _detect_angles(self, driver_name: str, side: str = "F") -> List[int]:
        """Auto-detect available angles for a driver and side"""
        file_driver_name = self._normalize_driver_name(driver_name)
        angles = set()

        for driver, angle, file_side in self._get_file_index().keys():
            if driver == file_driver_name and file_side == side:
                angles.add(angle)

        return sorted(list(angles))

    def _get_filename(self, driver_name: str, angle: int, side: str = "F") -> str:
        """Generate filename for given driver, angle, and side"""
        pattern_def = self._get_pattern_def()
        if not pattern_def:
            return ""
        file_driver_name = self._resolve_driver_name_for_files(driver_name)
        return pattern_def["filename"](file_driver_name, angle, side)

    # ==================== HDF5 Helper Methods ====================

    def _save_angle_metadata(self, angle_group, angle_data: Dict):
        """Save metadata and timing info to HDF5 angle group"""
        if 'metadata' in angle_data:
            meta = angle_data['metadata']
            for key in _ANGLE_METADATA_ATTRS:
                if key in meta:
                    angle_group.attrs[key] = meta.get(key, '')
            notes = str(meta.get("notes", ""))
            if notes:
                if "measurement_distance_m" not in angle_group.attrs:
                    distance_m = parse_distance_m(notes)
                    if distance_m is not None:
                        angle_group.attrs["measurement_distance_m"] = distance_m
                if "measurement_height_m" not in angle_group.attrs:
                    height_m = parse_height_m(notes)
                    if height_m is not None:
                        angle_group.attrs["measurement_height_m"] = height_m
                if "measurement_height_reference" not in angle_group.attrs:
                    height_reference = parse_height_reference(notes)
                    if height_reference is not None:
                        angle_group.attrs["measurement_height_reference"] = height_reference
            angle_group.attrs['sampleRate'] = meta.get('sampleRate', 0)
        angle_group.attrs['timing_corrected'] = angle_data.get('timing_corrected', False)
        angle_group.attrs['timing_offset_ms'] = angle_data.get('timing_offset_ms', 0.0)
        angle_group.attrs['timing_peak_time_ms'] = angle_data.get('timing_peak_time_ms', 0.0)
        angle_group.attrs['timing_peak_selection_reason'] = angle_data.get('timing_peak_selection_reason', '')
        angle_group.attrs['timing_peak_policy'] = angle_data.get('timing_peak_policy', '')
        angle_group.attrs['timing_first_strong_near_ref_lobe_time_ms'] = angle_data.get(
            'timing_first_strong_near_ref_lobe_time_ms',
            float('nan'),
        )
        angle_group.attrs['timing_selected_is_first_strong_near_ref_lobe'] = angle_data.get(
            'timing_selected_is_first_strong_near_ref_lobe',
            False,
        )
        angle_group.attrs['timing_first_lobe_threshold_fraction'] = angle_data.get(
            'timing_first_lobe_threshold_fraction',
            float('nan'),
        )
        angle_group.attrs['timing_first_lobe_window_start_ms'] = angle_data.get(
            'timing_first_lobe_window_start_ms',
            float('nan'),
        )
        angle_group.attrs['timing_first_lobe_window_end_ms'] = angle_data.get(
            'timing_first_lobe_window_end_ms',
            float('nan'),
        )
        angle_group.attrs['timing_rew_ir_start_time_ms'] = angle_data.get(
            'timing_rew_ir_start_time_ms',
            float('nan'),
        )

    def _load_angle_metadata(self, angle_group) -> Dict:
        """Load metadata and timing info from HDF5 angle group"""
        result = {}
        if 'title' in angle_group.attrs:
            result['metadata'] = {
                key: angle_group.attrs.get(key, '')
                for key in _ANGLE_METADATA_ATTRS
            }
            result['metadata']['sampleRate'] = angle_group.attrs.get('sampleRate', 0)
        result['timing_corrected'] = bool(angle_group.attrs.get('timing_corrected', False))
        result['timing_offset_ms'] = float(angle_group.attrs.get('timing_offset_ms', 0.0))
        result['timing_peak_time_ms'] = float(angle_group.attrs.get('timing_peak_time_ms', 0.0))
        result['timing_peak_selection_reason'] = str(angle_group.attrs.get('timing_peak_selection_reason', ''))
        result['timing_peak_policy'] = str(angle_group.attrs.get('timing_peak_policy', ''))
        result['timing_first_strong_near_ref_lobe_time_ms'] = float(
            angle_group.attrs.get('timing_first_strong_near_ref_lobe_time_ms', float('nan'))
        )
        result['timing_selected_is_first_strong_near_ref_lobe'] = bool(
            angle_group.attrs.get('timing_selected_is_first_strong_near_ref_lobe', False)
        )
        result['timing_first_lobe_threshold_fraction'] = float(
            angle_group.attrs.get('timing_first_lobe_threshold_fraction', float('nan'))
        )
        result['timing_first_lobe_window_start_ms'] = float(
            angle_group.attrs.get('timing_first_lobe_window_start_ms', float('nan'))
        )
        result['timing_first_lobe_window_end_ms'] = float(
            angle_group.attrs.get('timing_first_lobe_window_end_ms', float('nan'))
        )
        result['timing_rew_ir_start_time_ms'] = float(
            angle_group.attrs.get('timing_rew_ir_start_time_ms', float('nan'))
        )
        return result

    def _save_angles_group(self, parent_group, angles_dict: Dict, group_name: str = 'angles'):
        """Save angle measurements to HDF5 group"""
        group = parent_group.create_group(group_name)
        for angle, angle_data in angles_dict.items():
            ag = group.create_group(str(angle))
            ag.create_dataset('magnitude', data=angle_data['magnitude'])
            ag.create_dataset('phase', data=angle_data['phase'])
            ag.attrs['unit'] = angle_data['unit']
            ag.attrs['smoothing'] = angle_data['smoothing']
            self._save_angle_metadata(ag, angle_data)

    def _load_angles_group(self, parent_group, frequencies: np.ndarray, group_name: str = 'angles') -> Dict:
        """Load angle measurements from HDF5 group"""
        if group_name not in parent_group:
            return {}
        result = {}
        group = parent_group[group_name]
        for angle_str in group.keys():
            ag = group[angle_str]
            angle_data = {
                'frequencies': frequencies,
                'magnitude': np.array(ag['magnitude']),
                'phase': np.array(ag['phase']),
                'unit': ag.attrs['unit'],
                'smoothing': ag.attrs['smoothing'],
            }
            angle_data.update(self._load_angle_metadata(ag))
            result[int(angle_str)] = angle_data
        return result

    # ==================== HDF5 Save/Load ====================

    def save_to_hdf5(self, data: Dict, output_path: str,
                     gate_left_ms: float = 0.0, gate_right_ms: float = 0.0,
                     smoothing: int = 0):
        """Save polar data to HDF5 file

        Args:
            data: Polar measurement data dictionary
            output_path: Path to save HDF5 file
            gate_left_ms: Left gate time used (for metadata)
            gate_right_ms: Right gate time used (for metadata)
            smoothing: Smoothing factor used (for metadata)
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, 'w') as f:
            # Save global config as root attributes
            f.attrs['gate_left_ms'] = gate_left_ms
            f.attrs['gate_right_ms'] = gate_right_ms
            f.attrs['smoothing'] = smoothing
            f.attrs['smoothing_str'] = f"1/{smoothing}" if smoothing else "None"
            f.attrs['direct_ir_peak_policy'] = self.direct_ir_peak_policy
            if self.direct_ir_peak_policy == "strongest":
                f.attrs['target_kind'] = "legacy_strongest_lobe_diagnostic"
                f.attrs['diagnostic_only'] = True
                f.attrs['not_acceptance_target'] = True
                f.attrs['acceptance_note'] = (
                    "Unsafe strongest-peak timing policy: high-angle absolute peaks can be "
                    "reflected/scattered lobes, so this HDF5 is diagnostic-only."
                )
            elif self.pattern_type == "andres":
                f.attrs['target_kind'] = "andres_first_lobe_diagnostic"
                f.attrs['diagnostic_only'] = True
                f.attrs['not_acceptance_target'] = True
                f.attrs['acceptance_note'] = (
                    "Regenerated Andres first-strong-lobe HDF5 is diagnostic-only unless the "
                    "published polar explorer is regenerated from this same HDF5 in the same commit."
                )

            for driver_name, driver_data in data.items():
                driver_group = f.create_group(driver_name)
                driver_group.attrs['driver_name'] = driver_name
                driver_group.attrs['has_rear'] = driver_data.get('has_rear', False)

                if 'common_frequencies' in driver_data:
                    driver_group.create_dataset('frequencies',
                                               data=driver_data['common_frequencies'])

                # Save front angles
                self._save_angles_group(driver_group, driver_data['angles'], 'angles')

                # Save rear angles if present
                if driver_data.get('has_rear') and 'rear_angles' in driver_data:
                    self._save_angles_group(driver_group, driver_data['rear_angles'], 'rear_angles')

        print(f"Saved polar data to {output_path}")

    def load_from_hdf5(self, input_path: str) -> Dict:
        """Load polar data from HDF5 file

        Returns:
            Dictionary with 'config' key for global settings and driver names as keys
        """
        data = {}
        with h5py.File(input_path, 'r') as f:
            # Load global config
            data['_config'] = {
                'gate_left_ms': f.attrs.get('gate_left_ms', 0.0),
                'gate_right_ms': f.attrs.get('gate_right_ms', 0.0),
                'smoothing': f.attrs.get('smoothing', 0),
                'smoothing_str': f.attrs.get('smoothing_str', 'None'),
            }

            for driver_name in f.keys():
                driver_group = f[driver_name]

                # Skip drivers without frequency data (empty/misconfigured measurements)
                if 'frequencies' not in driver_group:
                    print(f"Warning: Skipping driver '{driver_name}' - no frequency data found")
                    continue

                has_rear = driver_group.attrs.get('has_rear', False)
                frequencies = np.array(driver_group['frequencies'])
                driver_data = {
                    'driver': driver_name,
                    'has_rear': has_rear,
                    'common_frequencies': frequencies,
                    'angles': self._load_angles_group(driver_group, frequencies, 'angles'),
                }

                # Load rear angles if present
                if has_rear:
                    rear_angles = self._load_angles_group(driver_group, frequencies, 'rear_angles')
                    if rear_angles:
                        driver_data['rear_angles'] = rear_angles

                data[driver_name] = driver_data
        return data

if __name__ == "__main__":
    loader = PolarDataLoader()
    # Test load with config defaults
    data = loader.load_all_drivers(
        smoothing=config.DEFAULT_SMOOTHING,
        gate_left_ms=config.GATE_LEFT_MS,
        gate_right_ms=config.GATE_RIGHT_MS
    )
    loader.save_to_hdf5(data, config.HDF5_FILE_PATH)
