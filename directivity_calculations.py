#!/usr/bin/env python3
"""
Directivity Calculations for Polar Response Analysis

This module provides functions for calculating directivity metrics from
polar response measurements, including:
- Directivity Index (DI)
- Beamwidth (-3dB, -6dB)
- Sound Power
- Listening Window
- Early Reflections
- Spinorama curves

Author: Andres Torrubia
Date: 2025-11-23
"""

import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from typing import Dict, Tuple, Optional, List


class DirectivityCalculator:
    """Calculate directivity metrics from polar measurements"""

    def __init__(self, frequencies: np.ndarray, angles: np.ndarray,
                 spl_matrix: np.ndarray):
        """
        Initialize calculator with polar measurement data

        Args:
            frequencies: Array of frequencies (Hz)
            angles: Array of measurement angles (degrees, typically 0-90)
            spl_matrix: 2D array [frequency, angle] of SPL values (dB)
        """
        self.frequencies = frequencies
        self.angles = np.array(angles)
        self.spl_matrix = spl_matrix

        # Validate dimensions
        assert spl_matrix.shape[0] == len(frequencies), \
            "SPL matrix first dimension must match frequencies"
        assert spl_matrix.shape[1] == len(angles), \
            "SPL matrix second dimension must match angles"

    def calculate_sound_power(self, method: str = "hemispherical") -> np.ndarray:
        """
        Calculate sound power from polar measurements

        Args:
            method: Integration method
                - "hemispherical": Front hemisphere (0-90°), appropriate for dipole
                - "spherical": Full sphere (assumes rear symmetry or actual data)

        Returns:
            Array of sound power values (dB) vs frequency
        """
        angles_rad = np.deg2rad(self.angles)
        power_db = np.zeros(len(self.frequencies))

        for i in range(len(self.frequencies)):
            # Convert SPL to linear intensity (proportional to pressure squared)
            intensity = 10 ** (self.spl_matrix[i, :] / 10)

            if method == "hemispherical":
                # Weight by solid angle element: sin(θ) dθ dφ
                # For hemisphere, φ integration gives 2π
                # Solid angle element: 2π sin(θ) dθ
                weights = np.sin(angles_rad)
                weights /= np.sum(weights)  # Normalize

                # Power = weighted average intensity
                power_linear = np.sum(intensity * weights)

            elif method == "spherical":
                # Full sphere integration
                # Assume rear hemisphere mirrors front (dipole)
                weights = np.sin(angles_rad)
                weights /= np.sum(weights) * 2  # Normalize for full sphere

                # Power = weighted average (front + rear)
                power_linear = 2 * np.sum(intensity * weights)

            else:
                raise ValueError(f"Unknown method: {method}")

            # Convert back to dB
            power_db[i] = 10 * np.log10(power_linear)

        return power_db

    def calculate_directivity_index(self, on_axis_spl: np.ndarray = None) -> np.ndarray:
        """
        Calculate Directivity Index (DI)

        DI = On-axis SPL - Sound Power

        Args:
            on_axis_spl: On-axis (0°) SPL values. If None, uses first column of spl_matrix

        Returns:
            Array of DI values (dB) vs frequency
        """
        if on_axis_spl is None:
            on_axis_spl = self.spl_matrix[:, 0]  # Assume 0° is first column

        sound_power = self.calculate_sound_power()
        di = on_axis_spl - sound_power

        return di

    def calculate_beamwidth(self, db_down: float = -6.0) -> np.ndarray:
        """
        Calculate beamwidth (coverage angle) vs frequency

        Args:
            db_down: dB threshold below on-axis (e.g., -3, -6, -10)

        Returns:
            Array of beamwidth values (degrees) vs frequency
        """
        on_axis_spl = self.spl_matrix[:, 0]
        beamwidth = np.zeros(len(self.frequencies))

        for i in range(len(self.frequencies)):
            # Normalize to on-axis
            spl_norm = self.spl_matrix[i, :] - on_axis_spl[i]

            # Find angle where SPL drops below threshold
            below_threshold = spl_norm < db_down

            if np.any(below_threshold):
                # Find first angle below threshold
                idx = np.where(below_threshold)[0][0]

                if idx > 0:
                    # Linear interpolation for precise angle
                    angle_before = self.angles[idx - 1]
                    angle_after = self.angles[idx]
                    spl_before = spl_norm[idx - 1]
                    spl_after = spl_norm[idx]

                    # Interpolate
                    angle_at_threshold = angle_before + \
                        (db_down - spl_before) * (angle_after - angle_before) / \
                        (spl_after - spl_before)

                    # Full beamwidth (both sides, assuming symmetry)
                    beamwidth[i] = 2 * angle_at_threshold
                else:
                    # Already below threshold at first non-zero angle
                    beamwidth[i] = 0
            else:
                # Never drops below threshold - very wide coverage
                beamwidth[i] = 180

        return beamwidth

    def calculate_listening_window(self, angle_weights: Dict[int, float] = None) -> np.ndarray:
        """
        Calculate Listening Window response (CEA-2034)

        Typical: Average of 0°, ±10°, ±20°, ±30° with equal weighting
        For horizontal-only data: Average of 0°, 10°, 20°, 30°

        Args:
            angle_weights: Dict of {angle: weight}. If None, uses default.

        Returns:
            Array of listening window SPL (dB) vs frequency
        """
        if angle_weights is None:
            # Default: 0°, 10°, 20°, 30° equally weighted
            angle_weights = {0: 1.0, 10: 1.0, 20: 1.0, 30: 1.0}

        lw = np.zeros(len(self.frequencies))

        for i in range(len(self.frequencies)):
            # Energy average (dB average in linear domain)
            intensities = []
            weights_list = []

            for angle, weight in angle_weights.items():
                if angle in self.angles:
                    angle_idx = np.where(self.angles == angle)[0][0]
                    intensity = 10 ** (self.spl_matrix[i, angle_idx] / 10)
                    intensities.append(intensity)
                    weights_list.append(weight)

            # Weighted energy average
            weights_array = np.array(weights_list)
            weights_array /= np.sum(weights_array)  # Normalize

            avg_intensity = np.sum(np.array(intensities) * weights_array)
            lw[i] = 10 * np.log10(avg_intensity)

        return lw

    def calculate_early_reflections(self, angle_weights: Dict[int, float] = None) -> np.ndarray:
        """
        Calculate Early Reflections response (CEA-2034)

        Approximation for horizontal measurements only:
        Average of 40°, 50°, 60°, 70° representing floor, ceiling, side walls

        Args:
            angle_weights: Dict of {angle: weight}. If None, uses default.

        Returns:
            Array of early reflections SPL (dB) vs frequency
        """
        if angle_weights is None:
            # Default: 40°, 50°, 60°, 70° equally weighted
            angle_weights = {40: 1.0, 50: 1.0, 60: 1.0, 70: 1.0}

        er = np.zeros(len(self.frequencies))

        for i in range(len(self.frequencies)):
            intensities = []
            weights_list = []

            for angle, weight in angle_weights.items():
                if angle in self.angles:
                    angle_idx = np.where(self.angles == angle)[0][0]
                    intensity = 10 ** (self.spl_matrix[i, angle_idx] / 10)
                    intensities.append(intensity)
                    weights_list.append(weight)

            # Weighted energy average
            weights_array = np.array(weights_list)
            weights_array /= np.sum(weights_array)

            avg_intensity = np.sum(np.array(intensities) * weights_array)
            er[i] = 10 * np.log10(avg_intensity)

        return er

    def calculate_predicted_in_room(self) -> np.ndarray:
        """
        Calculate Predicted In-Room Response (CEA-2034)

        PIR = 0.12 × Listening Window + 0.44 × Early Reflections + 0.44 × Sound Power

        Returns:
            Array of PIR values (dB) vs frequency
        """
        lw = self.calculate_listening_window()
        er = self.calculate_early_reflections()
        sp = self.calculate_sound_power()

        # Energy-weighted combination
        pir = 10 * np.log10(
            0.12 * 10 ** (lw / 10) +
            0.44 * 10 ** (er / 10) +
            0.44 * 10 ** (sp / 10)
        )

        return pir

    def interpolate_angles(self, target_angles: np.ndarray) -> np.ndarray:
        """
        Interpolate SPL matrix to finer angular resolution

        Args:
            target_angles: Array of desired angles (e.g., 0-90 in 1° steps)

        Returns:
            Interpolated SPL matrix [frequency, target_angles]
        """
        spl_interpolated = np.zeros((len(self.frequencies), len(target_angles)))

        for i in range(len(self.frequencies)):
            # Cubic interpolation for each frequency
            interpolator = interp1d(self.angles, self.spl_matrix[i, :],
                                   kind='cubic', fill_value='extrapolate')
            spl_interpolated[i, :] = interpolator(target_angles)

        return spl_interpolated


def fractional_octave_smooth(frequencies: np.ndarray, magnitude: np.ndarray,
                             fraction: int = 12) -> np.ndarray:
    """
    Apply fractional-octave smoothing to frequency response

    Args:
        frequencies: Array of frequencies (Hz)
        magnitude: Array of magnitude values (dB)
        fraction: Octave fraction (3=1/3, 6=1/6, 12=1/12, etc.)

    Returns:
        Smoothed magnitude array
    """
    smoothed = np.zeros_like(magnitude)

    for i, f in enumerate(frequencies):
        # Calculate bandwidth for this frequency
        # For 1/N octave: BW = f * (2^(1/(2N)) - 2^(-1/(2N)))
        bandwidth = f * (2 ** (1 / (2 * fraction)) - 2 ** (-1 / (2 * fraction)))

        # Find frequencies within bandwidth
        mask = (frequencies >= f - bandwidth / 2) & (frequencies <= f + bandwidth / 2)

        # Average magnitude in band (linear in dB)
        if np.any(mask):
            smoothed[i] = np.mean(magnitude[mask])
        else:
            smoothed[i] = magnitude[i]

    return smoothed


def normalize_polar_response(spl_matrix: np.ndarray, reference_angle_idx: int = 0) -> np.ndarray:
    """
    Normalize polar response to reference angle at each frequency

    Args:
        spl_matrix: 2D array [frequency, angle] of SPL values
        reference_angle_idx: Index of reference angle (typically 0 for on-axis)

    Returns:
        Normalized SPL matrix (dB relative to reference)
    """
    reference_spl = spl_matrix[:, reference_angle_idx:reference_angle_idx + 1]
    normalized = spl_matrix - reference_spl
    return normalized


def calculate_crossover_match_score(di1: np.ndarray, di2: np.ndarray,
                                   freq: np.ndarray, crossover_freq: float,
                                   bandwidth_octaves: float = 0.5) -> Dict:
    """
    Calculate directivity match quality at crossover frequency

    Args:
        di1: DI array for driver 1
        di2: DI array for driver 2
        freq: Frequency array
        crossover_freq: Crossover frequency (Hz)
        bandwidth_octaves: Bandwidth around crossover to analyze

    Returns:
        Dictionary with match metrics
    """
    # Find frequency range around crossover
    f_low = crossover_freq / (2 ** bandwidth_octaves)
    f_high = crossover_freq * (2 ** bandwidth_octaves)
    mask = (freq >= f_low) & (freq <= f_high)

    # Check if crossover frequency is within measurement range
    if crossover_freq < freq[0] or crossover_freq > freq[-1]:
        # Crossover outside measurement range
        return {
            "crossover_frequency": crossover_freq,
            "di_diff_at_crossover": np.nan,
            "di_diff_max_in_region": np.nan,
            "di_diff_avg_in_region": np.nan,
            "di_diff_rms_in_region": np.nan,
            "match_quality": "out_of_range",
            "note": f"Crossover at {crossover_freq} Hz is outside measurement range ({freq[0]:.1f} - {freq[-1]:.1f} Hz)"
        }

    # DI difference in crossover region
    di_diff = np.abs(di1[mask] - di2[mask])

    # Handle case where mask returns no values (shouldn't happen after range check, but be safe)
    if len(di_diff) == 0:
        di_at_xover_idx = np.argmin(np.abs(freq - crossover_freq))
        di_at_xover = di1[di_at_xover_idx] - di2[di_at_xover_idx]

        return {
            "crossover_frequency": crossover_freq,
            "di_diff_at_crossover": di_at_xover,
            "di_diff_max_in_region": np.nan,
            "di_diff_avg_in_region": np.nan,
            "di_diff_rms_in_region": np.nan,
            "match_quality": "insufficient_data"
        }

    # Match score metrics
    di_at_xover_idx = np.argmin(np.abs(freq - crossover_freq))
    di_at_xover = di1[di_at_xover_idx] - di2[di_at_xover_idx]

    return {
        "crossover_frequency": crossover_freq,
        "di_diff_at_crossover": di_at_xover,
        "di_diff_max_in_region": np.max(di_diff),
        "di_diff_avg_in_region": np.mean(di_diff),
        "di_diff_rms_in_region": np.sqrt(np.mean(di_diff ** 2)),
        "match_quality": "good" if np.abs(di_at_xover) < 2 else "acceptable" if np.abs(di_at_xover) < 4 else "poor"
    }


def create_polar_matrix_from_dict(driver_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert driver polar data dictionary to matrix format for calculations

    Args:
        driver_data: Dictionary from PolarDataLoader with 'angles' and 'common_frequencies'

    Returns:
        Tuple of (frequencies, angles, spl_matrix)
    """
    angles = sorted(driver_data['angles'].keys())
    frequencies = driver_data['common_frequencies']

    spl_matrix = np.zeros((len(frequencies), len(angles)))

    for i, angle in enumerate(angles):
        spl_matrix[:, i] = driver_data['angles'][angle]['magnitude']

    return frequencies, np.array(angles), spl_matrix


def main():
    """Example usage and testing"""
    # Create synthetic test data
    frequencies = np.logspace(np.log10(100), np.log10(20000), 200)
    angles = np.arange(0, 91, 10)

    # Simulate beaming driver (narrowing with frequency)
    spl_matrix = np.zeros((len(frequencies), len(angles)))
    for i, f in enumerate(frequencies):
        # On-axis: flat 90 dB
        # Off-axis: decreases with frequency and angle
        for j, angle in enumerate(angles):
            # Beaming factor increases with frequency
            beaming = (f / 1000) ** 0.5
            spl_matrix[i, j] = 90 - beaming * (angle / 30) ** 2

    # Calculate metrics
    calc = DirectivityCalculator(frequencies, angles, spl_matrix)

    di = calc.calculate_directivity_index()
    beamwidth = calc.calculate_beamwidth(-6)
    sound_power = calc.calculate_sound_power()
    lw = calc.calculate_listening_window()

    print("Directivity Calculator Test")
    print("=" * 60)
    print(f"Frequency range: {frequencies[0]:.1f} - {frequencies[-1]:.1f} Hz")
    print(f"Number of angles: {len(angles)}")
    print(f"\nDI at 1 kHz: {di[np.argmin(np.abs(frequencies - 1000))]:.2f} dB")
    print(f"Beamwidth at 1 kHz: {beamwidth[np.argmin(np.abs(frequencies - 1000))]:.1f}°")
    print(f"DI at 10 kHz: {di[np.argmin(np.abs(frequencies - 10000))]:.2f} dB")
    print(f"Beamwidth at 10 kHz: {beamwidth[np.argmin(np.abs(frequencies - 10000))]:.1f}°")


if __name__ == "__main__":
    main()
