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

class PolarDataLoader:
    """Load and manage polar response measurements from REW"""

    def __init__(self, data_directory: str = ".", connect_to_rew: bool = True,
                 pattern_type: str = "andres"):
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

    def _decode_base64_floats(self, base64_string: str) -> np.ndarray:
        """Decode Base64-encoded float array from REW API"""
        byte_data = base64.b64decode(base64_string)
        num_floats = len(byte_data) // 4
        floats = struct.unpack(f'>{num_floats}f', byte_data)
        return np.array(floats)

    def _set_ir_window(self, measurement_uuid: str, left_ms: float, right_ms: float):
        """Set IR window settings via REW API, preserving Ref Time"""
        url = f"{config.REW_API_BASE}/measurements/{measurement_uuid}/ir-windows"
        
        try:
            # 1. Get current settings
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            current_settings = response.json()
            
            # 2. Update widths
            current_settings["leftWindowWidthms"] = left_ms
            current_settings["rightWindowWidthms"] = right_ms
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
                "refTimems": 0, # Fallback default
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

    def _auto_fix_timing(self, measurement_uuid: str):
        """
        Detect and fix timing anomalies. 
        Looks for the earliest significant peak (direct sound) which might be shifted (e.g. -12ms).
        Aligns that peak to t=0 using REW's 'Offset t=0' command.
        """
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
                return

            sample_rate = ir_info["sampleRate"]
            start_time_s = ir_info["startTime"]
            
            # 2. Analyze Peaks
            abs_ir = np.abs(ir_data)
            global_max_idx = np.argmax(abs_ir)
            global_max_val = abs_ir[global_max_idx]
            
            # Search for earliest significant peak (10% of max) to catch direct sound
            # if it's earlier than the global max (reflection).
            # However, inspection showed the direct sound IS the max or close to it at -12ms.
            # But we'll be robust.
            
            threshold = global_max_val * 0.1
            candidates = np.where(abs_ir > threshold)[0]
            
            target_peak_idx = global_max_idx
            
            # Simple clustering: Take the first candidate that is a local maximum
            if len(candidates) > 0:
                for idx in candidates:
                    # Check local max condition
                    if idx > 0 and idx < len(abs_ir)-1:
                        if abs_ir[idx] > abs_ir[idx-1] and abs_ir[idx] > abs_ir[idx+1]:
                            target_peak_idx = idx
                            break
            
            # Calculate time of the target peak
            peak_time_s = start_time_s + (target_peak_idx / sample_rate)
            
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
                requests.post(url_cmd, json=payload, timeout=10)
                
                # Verify the shift
                time.sleep(0.5) # Wait for command to apply
                resp_verify = requests.get(url_ir, params={"windowed": "false"}, timeout=10)
                if resp_verify.ok:
                    info_v = resp_verify.json()
                    start_v = info_v["startTime"]
                    peak_time_v = start_v + (target_peak_idx / sample_rate)
                    print(f"    ✓ Verification: Peak is now at {peak_time_v*1000:.2f} ms")
                    
                    if abs(peak_time_v) > 0.1: # If still > 0.1ms off
                        print("    !! WARNING: Correction failed to move peak to 0. Result is still off.")

                # Reset Ref Time to 0
                url_win = f"{config.REW_API_BASE}/measurements/{measurement_uuid}/ir-windows"
                resp_win = requests.get(url_win)
                if resp_win.ok:
                    settings = resp_win.json()
                    settings["refTimems"] = 0
                    requests.post(url_win, json=settings, timeout=10)
                
        except Exception as e:
            print(f"    Warning: Failed to auto-fix timing: {e}")

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
                self._auto_fix_timing(measurement_uuid)

                # 1. Apply Time Gating via REW API
                if gate_right_ms > 0 or gate_left_ms > 0:
                    # print(f"    Gating (REW): {gate_left_ms}ms / {gate_right_ms}ms")
                    self._set_ir_window(measurement_uuid, gate_left_ms, gate_right_ms)
                
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
                    "metadata": measurements[last_key]
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
                        include_rear: bool = False) -> Dict:
        """
        Load complete polar measurements for all drivers

        Args:
            driver_list: List of drivers to load (auto-detected if None)
            angles: List of angles to load (auto-detected if None)
            smoothing: Smoothing factor (0 for none, 12 for 1/12 octave)
            gate_left_ms: Left gate time in ms
            gate_right_ms: Right gate time in ms
            include_rear: Whether to load rear measurements (if available)
        """
        if driver_list is None:
            driver_list = self._detect_drivers()

        print(f"Loading polar data for {len(driver_list)} drivers...")
        print(f"Drivers: {', '.join(driver_list)}")
        print(f"Pattern type: {self.pattern_type}")
        print(f"Gating: {gate_left_ms}ms / {gate_right_ms}ms")
        if smoothing:
            print(f"Smoothing: 1/{smoothing} octave")
        else:
            print("Smoothing: None")
        if include_rear:
            print("Including rear measurements")
        print("=" * 60)

        all_data = {}

        for driver in driver_list:
            print(f"\nLoading driver: {driver}")
            all_data[driver] = self.load_driver_polar_set(
                driver, angles, smoothing, gate_left_ms, gate_right_ms, include_rear
            )

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
        # Auto-detect angles if not provided
        if angles is None:
            angles = self._detect_angles(driver_name, "F")
            if not angles:
                angles = list(range(0, 91, 10))  # Fallback default

        polar_data = {"driver": driver_name, "angles": {}, "has_rear": False}

        # Load front measurements
        for angle in angles:
            filename = self._get_filename(driver_name, angle, "F")
            file_path = self.data_dir / filename

            if not file_path.exists():
                print(f"Warning: File not found: {filename}")
                continue

            print(f"  Loading {driver_name} at {angle}° (Front)...")
            measurement = self.load_measurement(
                str(file_path), smoothing, gate_left_ms, gate_right_ms
            )
            polar_data["angles"][angle] = measurement
            time.sleep(2.0)

        # Load rear measurements if requested
        if include_rear:
            rear_angles = self._detect_angles(driver_name, "R")
            if rear_angles:
                polar_data["rear_angles"] = {}
                polar_data["has_rear"] = True

                for angle in rear_angles:
                    filename = self._get_filename(driver_name, angle, "R")
                    file_path = self.data_dir / filename

                    if not file_path.exists():
                        print(f"Warning: File not found: {filename}")
                        continue

                    print(f"  Loading {driver_name} at {angle}° (Rear)...")
                    measurement = self.load_measurement(
                        str(file_path), smoothing, gate_left_ms, gate_right_ms
                    )
                    polar_data["rear_angles"][angle] = measurement
                    time.sleep(2.0)

        # Common frequency grid (using on-axis as reference)
        if polar_data["angles"]:
            ref_angle = sorted(polar_data["angles"].keys())[0]
            polar_data["common_frequencies"] = polar_data["angles"][ref_angle]["frequencies"]

        return polar_data

    def _parse_filename(self, filename: str) -> Optional[Dict]:
        """Parse measurement filename based on pattern type.

        Returns dict with: driver, angle, side ('F' or 'R'), or None if no match
        """
        stem = Path(filename).stem

        if self.pattern_type == "andres":
            # Pattern: F{angle}-{driver}
            match = re.match(r'F(\d+)-(.+)', stem)
            if match:
                return {"angle": int(match.group(1)), "driver": match.group(2), "side": "F"}

        elif self.pattern_type == "juan":
            # Pattern: {driver} {angle} {side}
            match = re.match(r'(.+)\s+(\d+)\s+([FR])$', stem)
            if match:
                return {"driver": match.group(1), "angle": int(match.group(2)), "side": match.group(3)}

        return None

    def _detect_drivers(self) -> List[str]:
        """Auto-detect driver names from .mdat files"""
        files = list(self.data_dir.glob("*.mdat"))
        drivers = set()

        for f in files:
            parsed = self._parse_filename(f.name)
            if parsed:
                driver = parsed["driver"]
                # Skip combination measurements for andres pattern
                if self.pattern_type == "andres" and "con" in driver.lower():
                    continue
                drivers.add(driver)

        return sorted(list(drivers))

    def _detect_angles(self, driver_name: str, side: str = "F") -> List[int]:
        """Auto-detect available angles for a driver and side"""
        files = list(self.data_dir.glob("*.mdat"))
        angles = set()

        for f in files:
            parsed = self._parse_filename(f.name)
            if parsed and parsed["driver"] == driver_name and parsed["side"] == side:
                angles.add(parsed["angle"])

        return sorted(list(angles))

    def _get_filename(self, driver_name: str, angle: int, side: str = "F") -> str:
        """Generate filename for given driver, angle, and side"""
        if self.pattern_type == "andres":
            return f"F{angle}-{driver_name}.mdat"
        elif self.pattern_type == "juan":
            return f"{driver_name} {angle} {side}.mdat"
        return ""

    def save_to_hdf5(self, data: Dict, output_path: str):
        """Save polar data to HDF5 file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, 'w') as f:
            for driver_name, driver_data in data.items():
                driver_group = f.create_group(driver_name)
                driver_group.attrs['driver_name'] = driver_name
                driver_group.attrs['has_rear'] = driver_data.get('has_rear', False)

                if 'common_frequencies' in driver_data:
                    driver_group.create_dataset('frequencies',
                                               data=driver_data['common_frequencies'])

                # Save front angles
                angles_group = driver_group.create_group('angles')
                for angle, angle_data in driver_data['angles'].items():
                    angle_group = angles_group.create_group(str(angle))
                    angle_group.create_dataset('magnitude', data=angle_data['magnitude'])
                    angle_group.create_dataset('phase', data=angle_data['phase'])
                    angle_group.attrs['unit'] = angle_data['unit']
                    angle_group.attrs['smoothing'] = angle_data['smoothing']

                # Save rear angles if present
                if driver_data.get('has_rear') and 'rear_angles' in driver_data:
                    rear_group = driver_group.create_group('rear_angles')
                    for angle, angle_data in driver_data['rear_angles'].items():
                        angle_group = rear_group.create_group(str(angle))
                        angle_group.create_dataset('magnitude', data=angle_data['magnitude'])
                        angle_group.create_dataset('phase', data=angle_data['phase'])
                        angle_group.attrs['unit'] = angle_data['unit']
                        angle_group.attrs['smoothing'] = angle_data['smoothing']

        print(f"Saved polar data to {output_path}")

    def load_from_hdf5(self, input_path: str) -> Dict:
        """Load polar data from HDF5 file"""
        data = {}
        with h5py.File(input_path, 'r') as f:
            for driver_name in f.keys():
                driver_group = f[driver_name]
                has_rear = driver_group.attrs.get('has_rear', False)
                driver_data = {
                    'driver': driver_name,
                    'angles': {},
                    'has_rear': has_rear,
                    'common_frequencies': np.array(driver_group['frequencies'])
                }

                # Load front angles
                angles_group = driver_group['angles']
                for angle_str in angles_group.keys():
                    angle = int(angle_str)
                    angle_group = angles_group[angle_str]
                    driver_data['angles'][angle] = {
                        'frequencies': driver_data['common_frequencies'],
                        'magnitude': np.array(angle_group['magnitude']),
                        'phase': np.array(angle_group['phase']),
                        'unit': angle_group.attrs['unit'],
                        'smoothing': angle_group.attrs['smoothing']
                    }

                # Load rear angles if present
                if has_rear and 'rear_angles' in driver_group:
                    driver_data['rear_angles'] = {}
                    rear_group = driver_group['rear_angles']
                    for angle_str in rear_group.keys():
                        angle = int(angle_str)
                        angle_group = rear_group[angle_str]
                        driver_data['rear_angles'][angle] = {
                            'frequencies': driver_data['common_frequencies'],
                            'magnitude': np.array(angle_group['magnitude']),
                            'phase': np.array(angle_group['phase']),
                            'unit': angle_group.attrs['unit'],
                            'smoothing': angle_group.attrs['smoothing']
                        }

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