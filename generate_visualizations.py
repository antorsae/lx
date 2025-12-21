#!/usr/bin/env python3
"""
Generate Polar Response Visualizations for LX521 Drivers

This script generates comprehensive polar response analysis visualizations including:
- Directivity Index (DI) plots
- Beamwidth plots
- Contour/heatmap plots
- Polar plots (including interactive 360° explorer)
- Crossover match analysis

Refactored to use centralized config.

Author: Andres Torrubia
Date: 2025-11-23
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from pathlib import Path
import gzip
import base64
import json
import html

# Import centralized configuration
import config

# ==================== HTML Template Constants ====================
# Reusable HTML fragments for embedded templates

HTML_DOCTYPE = '<!DOCTYPE html>\n<html lang="en">\n'

HTML_HEAD_START = '''<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
'''

HTML_PLOTLY_SCRIPT = '    <script src="https://cdn.plot.ly/plotly-3.3.0.min.js"></script>\n'
HTML_PAKO_SCRIPT = '    <script src="https://cdn.jsdelivr.net/npm/pako@2.1.0/dist/pako.min.js"></script>\n'
HTML_YAML_SCRIPT = '    <script src="https://cdn.jsdelivr.net/npm/js-yaml@4.1.0/dist/js-yaml.min.js"></script>\n'

FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

CSS_RESET = f'''* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: {FONT_STACK}; background: #f5f5f5; }}
'''

def build_html_page(title: str, styles: str, body: str, scripts: str = '', extra_head: str = '') -> str:
    """Build complete HTML page from components.

    Args:
        title: Page title
        styles: CSS styles (without <style> tags)
        body: Body content (without <body> tags)
        scripts: JavaScript (without <script> tags, but can include multiple script blocks)
        extra_head: Extra content for <head> (e.g., external script tags)

    Returns:
        Complete HTML document string
    """
    return f'''{HTML_DOCTYPE}{HTML_HEAD_START}    <title>{title}</title>
{extra_head}    <style>
{styles}
    </style>
</head>
<body>
{body}
{scripts}
</body>
</html>'''


def format_frequency_label(freq_hz: float) -> str:
    """Format frequency tick labels consistently across plots."""
    if freq_hz >= 1000:
        val = freq_hz / 1000
        return f'{int(val)}k' if float(val).is_integer() else f'{val}k'
    return str(int(freq_hz))


def get_valid_frequency_ticks(
    freq_min: float = config.FREQ_MIN,
    freq_max: float = config.FREQ_MAX,
):
    """Return (tick_values, tick_text) within configured range."""
    all_ticks = sorted(set(config.GRID_FREQS_MAJOR + config.GRID_FREQS_MINOR))
    valid_ticks = [f for f in all_ticks if freq_min <= f <= freq_max]
    tick_text = [format_frequency_label(f) for f in valid_ticks]
    return valid_ticks, tick_text

from polar_data_loader import PolarDataLoader
from directivity_calculations import (
    DirectivityCalculator, create_polar_matrix_from_dict,
    calculate_crossover_match_score
)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = config.DPI_STATIC
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

class PolarResponseVisualizer:
    """Generate comprehensive polar response visualizations"""

    def __init__(self, data_path: str = None, static_plots_dir: Path = None, interactive_plots_dir: Path = None):
        """
        Initialize visualizer

        Args:
            data_path: Path to HDF5 data file (default from config)
            static_plots_dir: Directory for static PNG plots (default from config)
            interactive_plots_dir: Directory for interactive HTML plots (default from config)
        """
        if data_path is None:
            data_path = config.HDF5_FILE_PATH

        # Set output directories (use provided or fall back to config defaults)
        self.static_plots_dir = Path(static_plots_dir) if static_plots_dir else config.STATIC_PLOTS_DIR
        self.interactive_plots_dir = Path(interactive_plots_dir) if interactive_plots_dir else config.INTERACTIVE_PLOTS_DIR

        self.data_path = Path(data_path)
        self.loader = PolarDataLoader(connect_to_rew=False)
        self.data = self.loader.load_from_hdf5(data_path)

        # Extract config (if present) and remove from driver list
        self.global_config = self.data.pop('_config', {
            'gate_left_ms': 0.0,
            'gate_right_ms': 0.0,
            'smoothing': 0,
            'smoothing_str': 'None'
        })

        self.all_drivers = sorted(self.data.keys())
        self.drivers = []

        # Calculate directivity metrics for all drivers
        # Note: DI/beamwidth/sound power are calculated from front hemisphere only (0-90 deg),
        # following industry standard practice. Rear data is stored separately for 360-deg polar plots.
        self.calc_results = {}
        self.skipped_drivers = {}
        for driver in self.all_drivers:
            try:
                freq, angles, spl_matrix, phase_matrix = create_polar_matrix_from_dict(self.data[driver])
                calc = DirectivityCalculator(freq, angles, spl_matrix)
            except ValueError as exc:
                print(f"Warning: Skipping driver '{driver}' due to invalid angle data: {exc}")
                self.skipped_drivers[driver] = str(exc)
                continue

            # Also create rear SPL matrix if available
            rear_spl_matrix = None
            rear_phase_matrix = None
            if self.data[driver].get('has_rear') and 'rear_angles' in self.data[driver]:
                _, rear_angles, rear_spl_matrix, rear_phase_matrix = create_polar_matrix_from_dict(
                    {'angles': self.data[driver]['rear_angles'],
                     'common_frequencies': self.data[driver]['common_frequencies']}
                )

            self.calc_results[driver] = {
                'frequencies': freq,
                'angles': angles,
                'spl_matrix': spl_matrix,
                'phase_matrix': phase_matrix,
                'rear_spl_matrix': rear_spl_matrix,
                'rear_phase_matrix': rear_phase_matrix,
                'has_rear': self.data[driver].get('has_rear', False),
                'calculator': calc,
                'di': calc.calculate_directivity_index(),
                'beamwidth_6db': calc.calculate_beamwidth(-6),
                'beamwidth_3db': calc.calculate_beamwidth(-3),
                'sound_power': calc.calculate_sound_power()
            }
            self.drivers.append(driver)

        if not self.drivers:
            raise ValueError(
                "No valid drivers found. Each driver needs at least 2 angles and must include 0°."
            )

        # Ensure output directories exist
        self.static_plots_dir.mkdir(parents=True, exist_ok=True)
        (self.static_plots_dir / "core").mkdir(exist_ok=True)
        (self.static_plots_dir / "crossover").mkdir(exist_ok=True)
        (self.static_plots_dir / "polar").mkdir(exist_ok=True)

        self.interactive_plots_dir.mkdir(parents=True, exist_ok=True)
        (self.interactive_plots_dir / "polar").mkdir(exist_ok=True)

    def _configure_interactive_axis(self, fig):
        """Helper to apply custom ticks to plotly figure x-axis"""
        valid_ticks, tick_text = get_valid_frequency_ticks()

        # Apply to all x-axes in the figure
        fig.update_xaxes(
            type="log",
            range=[np.log10(config.FREQ_MIN), np.log10(config.FREQ_MAX)],
            tickvals=valid_ticks,
            ticktext=tick_text,
            tickmode="array"
        )

    def _write_compressed_html(self, fig, filepath: Path, title: str = "Plot"):
        """Write Plotly figure as compressed HTML using pako for browser decompression.

        This significantly reduces file size by:
        1. Compressing the JSON data with gzip
        2. Base64 encoding for embedding in HTML
        3. Using pako.js to decompress in the browser

        Typical compression ratio: 70-85% smaller files.
        """
        # Compress and encode the figure JSON
        fig_json = fig.to_json()
        compressed = gzip.compress(fig_json.encode('utf-8'), compresslevel=9)
        b64_data = base64.b64encode(compressed).decode('ascii')

        # Build page using helper
        styles = '''        body { margin: 0; padding: 0; }
        #plot { width: 100vw; height: 100vh; }
        #loading {
            position: absolute; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            font-family: Arial, sans-serif; font-size: 18px;
        }'''

        body = '''    <div id="loading">Loading and decompressing data...</div>
    <div id="plot"></div>'''

        script = f'''    <script>
        // Compressed data (gzip + base64)
        const compressedData = "{b64_data}";

        // Decode base64 and decompress
        const binaryStr = atob(compressedData);
        const bytes = new Uint8Array(binaryStr.length);
        for (let i = 0; i < binaryStr.length; i++) {{
            bytes[i] = binaryStr.charCodeAt(i);
        }}

        // Decompress using pako
        const decompressed = pako.inflate(bytes, {{ to: 'string' }});
        const figData = JSON.parse(decompressed);

        // Hide loading message and render plot
        document.getElementById('loading').style.display = 'none';
        Plotly.newPlot('plot', figData.data, figData.layout, {{responsive: true, showTips: false}});
    </script>'''

        html = build_html_page(title, styles, body, script,
                               extra_head=HTML_PAKO_SCRIPT + HTML_PLOTLY_SCRIPT)

        with open(filepath, 'w') as f:
            f.write(html)

    def _build_360_polar_data(self, driver: str, freq_idx: int):
        """Build full 360° polar data for a driver at a specific frequency.

        For front-only data: mirrors front hemisphere to create symmetric pattern
        For front+rear data: creates full 360° continuous pattern

        Returns:
            angles_rad: Array of angles in radians (0 to 2π)
            spl_data: SPL values normalized to on-axis
        """
        res = self.calc_results[driver]
        front_angles = res['angles']  # e.g., [0, 10, 20, ..., 90] or [0, 15, 30, ..., 90]
        front_spl = res['spl_matrix'][freq_idx, :]  # SPL at this frequency

        has_rear = res.get('has_rear', False)
        rear_spl_matrix = res.get('rear_spl_matrix')

        if has_rear and rear_spl_matrix is not None:
            # Full 360° with real rear data
            rear_spl = rear_spl_matrix[freq_idx, :]

            # Build full pattern:
            # 0° to 90° (front right): front_spl[0:end]
            # 90° to 180° (rear right): interpolate from front_90 to rear, then rear angles reversed
            # 180° to 270° (rear left): mirror of rear
            # 270° to 360° (front left): mirror of front

            # Actually, simpler approach for dipole/open baffle:
            # Front: 0° to 90° = front measurements
            # Right side (90° to 180°): rear measurements from 90° down to 0° (as 90° to 180°)
            # Rear (180°): rear 0° measurement
            # Left side: mirror of right side

            # Angles for full 360°:
            # Front right quadrant: 0, 15, 30, 45, 60, 75, 90
            # Rear right quadrant: 105, 120, 135, 150, 165, 180 (rear angles mapped)
            # Rear left quadrant: 195, 210, 225, 240, 255, 270 (mirror of rear right)
            # Front left quadrant: 285, 300, 315, 330, 345 (mirror of front right)

            # For simplicity, let's build it piece by piece
            front_angles_arr = np.array(front_angles)

            # Front quadrant (0 to 90): use front data directly
            angles_0_90 = front_angles_arr
            spl_0_90 = front_spl

            # Rear quadrant (90 to 180): rear data, angles go 90->180
            # Rear measurements are at angles 0, 15, 30... which map to 180, 165, 150...
            # So rear_angle 0 = 180°, rear_angle 90 = 90°
            # For 90->180, we use rear data reversed
            rear_angles_arr = np.array(front_angles)  # Assuming same angles
            angles_90_180 = 180 - rear_angles_arr[::-1]  # [90, 105, 120, ..., 180] but needs fixing
            # Actually: rear measurement at angle X represents the sound at 180-X degrees
            # rear at 0° = behind (180°), rear at 90° = side (90°)
            # So for angles 90 to 180, we need rear[90], rear[75], ..., rear[0]
            angles_90_180 = 90 + front_angles_arr[1:]  # Skip 90 as it's shared: [105, 120, ...]
            # Map to rear: angle 105 -> rear at 180-105=75, angle 120 -> rear at 60, etc
            spl_90_180 = []
            for a in angles_90_180:
                rear_measurement_angle = 180 - a  # Which rear angle to use
                idx = np.abs(front_angles_arr - rear_measurement_angle).argmin()
                spl_90_180.append(rear_spl[idx])
            spl_90_180 = np.array(spl_90_180)

            # Rear center (180°) = rear at 0°
            angles_180 = np.array([180])
            spl_180 = np.array([rear_spl[0]])

            # Rear left quadrant (180 to 270): mirror of 90-180
            angles_180_270 = 360 - angles_90_180[::-1]  # [255, 240, 225, ...]
            spl_180_270 = spl_90_180[::-1]

            # Front left quadrant (270 to 360): mirror of 0-90
            angles_270_360 = 360 - front_angles_arr[::-1][:-1]  # [270, 285, 300, ...] skip 360=0
            spl_270_360 = front_spl[::-1][:-1]

            # Combine all
            all_angles = np.concatenate([angles_0_90, angles_90_180, angles_180, angles_180_270, angles_270_360])
            all_spl = np.concatenate([spl_0_90, spl_90_180, spl_180, spl_180_270, spl_270_360])

            # Sort by angle
            sort_idx = np.argsort(all_angles)
            all_angles = all_angles[sort_idx]
            all_spl = all_spl[sort_idx]

            # Close the loop (add 360° = 0°)
            all_angles = np.append(all_angles, 360)
            all_spl = np.append(all_spl, all_spl[0])

        else:
            # Front-only: mirror for left/right symmetry (180° pattern)
            # Original approach: -90 to +90 degrees
            angles_full = np.concatenate([[-a for a in reversed(front_angles) if a > 0], front_angles])
            data_full = np.concatenate([front_spl[::-1][:-1], front_spl])

            all_angles = angles_full
            all_spl = data_full

        # Normalize to on-axis (0°)
        on_axis_idx = np.abs(all_angles).argmin()
        spl_normalized = all_spl - all_spl[on_axis_idx]

        # Clamp extreme values
        spl_normalized = np.clip(spl_normalized, -40, 10)

        angles_rad = np.radians(all_angles)
        return angles_rad, spl_normalized

    def _interpolate_angle_grid(self, angles, spl_matrix, phase_matrix, target_angles):
        base_angles = np.array([int(a) for a in angles], dtype=float)
        target = np.array([int(a) for a in target_angles], dtype=float)

        base_order = np.argsort(base_angles)
        base_angles = base_angles[base_order]
        spl_matrix = spl_matrix[:, base_order]
        phase_matrix = phase_matrix[:, base_order]

        base_set = {int(a) for a in base_angles.tolist()}
        missing = [int(a) for a in target.tolist() if int(a) not in base_set]

        if len(base_angles) == len(target) and np.allclose(base_angles, target):
            return base_angles.astype(int).tolist(), spl_matrix, phase_matrix, missing

        # Interpolate using complex response (mag + unwrapped phase) to avoid phase discontinuities.
        mag = 10 ** (spl_matrix / 20.0)
        phase_rad = np.deg2rad(phase_matrix)
        phase_unwrapped = np.unwrap(phase_rad, axis=1)
        comp = mag * np.exp(1j * phase_unwrapped)

        comp_interp = np.empty((comp.shape[0], len(target)), dtype=np.complex128)
        for i in range(comp.shape[0]):
            real_interp = np.interp(target, base_angles, comp[i].real)
            imag_interp = np.interp(target, base_angles, comp[i].imag)
            comp_interp[i] = real_interp + 1j * imag_interp

        mag_interp = np.abs(comp_interp)
        spl_interp = 20 * np.log10(np.maximum(mag_interp, 1e-12))
        phase_interp = np.rad2deg(np.angle(comp_interp))

        return target.astype(int).tolist(), spl_interp, phase_interp, missing

    def _format_interpolation_note(self, target_angles, interp_info):
        target = sorted({int(a) for a in target_angles})
        if not interp_info:
            return "No interpolation needed."

        lines = []
        lines.append(f"Aligned to base grid: {', '.join(map(str, target))}")

        grouped = {}
        for driver, info in interp_info.items():
            key = (tuple(info.get("source_angles", [])), tuple(info.get("missing_angles", [])))
            grouped.setdefault(key, []).append(driver)

        for (source, missing), drivers in grouped.items():
            source_str = ", ".join(map(str, source)) if source else "none"
            missing_str = ", ".join(map(str, missing)) if missing else "none"
            driver_str = ", ".join(drivers)
            lines.append(f"{driver_str} measured: {source_str}; interpolated: {missing_str}")

        lines.append("Method: linear interpolation of complex response (magnitude + phase, phase unwrapped).")
        return "\n".join(lines)

    def _build_polar_data_from_result(self, res, freq_idx: int, use_rear: bool):
        front_angles = res['angles']
        front_spl = res['spl_matrix'][freq_idx, :]

        has_rear = use_rear and res.get('has_rear', False) and res.get('rear_spl_matrix') is not None
        rear_spl_matrix = res.get('rear_spl_matrix')

        if has_rear and rear_spl_matrix is not None:
            rear_spl = rear_spl_matrix[freq_idx, :]

            front_angles_arr = np.array(front_angles)

            angles_0_90 = front_angles_arr
            spl_0_90 = front_spl

            angles_90_180 = 90 + front_angles_arr[1:]
            spl_90_180 = []
            for a in angles_90_180:
                rear_measurement_angle = 180 - a
                idx = np.abs(front_angles_arr - rear_measurement_angle).argmin()
                spl_90_180.append(rear_spl[idx])
            spl_90_180 = np.array(spl_90_180)

            angles_180 = np.array([180])
            spl_180 = np.array([rear_spl[0]])

            angles_180_270 = 360 - angles_90_180[::-1]
            spl_180_270 = spl_90_180[::-1]

            angles_270_360 = 360 - front_angles_arr[::-1][:-1]
            spl_270_360 = front_spl[::-1][:-1]

            all_angles = np.concatenate([angles_0_90, angles_90_180, angles_180, angles_180_270, angles_270_360])
            all_spl = np.concatenate([spl_0_90, spl_90_180, spl_180, spl_180_270, spl_270_360])

            sort_idx = np.argsort(all_angles)
            all_angles = all_angles[sort_idx]
            all_spl = all_spl[sort_idx]

            all_angles = np.append(all_angles, 360)
            all_spl = np.append(all_spl, all_spl[0])
        else:
            angles_full = np.concatenate([[-a for a in reversed(front_angles) if a > 0], front_angles])
            data_full = np.concatenate([front_spl[::-1][:-1], front_spl])
            all_angles = angles_full
            all_spl = data_full

        on_axis_idx = np.abs(all_angles).argmin()
        spl_normalized = all_spl - all_spl[on_axis_idx]
        spl_normalized = np.clip(spl_normalized, -40, 10)

        angles_rad = np.radians(all_angles)
        return angles_rad, spl_normalized

    def _load_extra_set_data(self, set_name: str):
        mset = config.MEASUREMENT_SETS.get(set_name)
        if not mset:
            return None
        extra_path = config.DATA_DIR / mset.get("hdf5_file", "")
        if not extra_path.exists():
            return None
        try:
            if extra_path.resolve() == self.data_path.resolve():
                return None
        except OSError:
            pass
        extra_data = self.loader.load_from_hdf5(str(extra_path))
        extra_data.pop('_config', None)
        return extra_data

    def _build_explorer_payload(self, dataset: dict, target_points: int, target_angles=None):
        extra_all_data = {}
        extra_angles = set()
        extra_drivers = []
        interp_info = {}
        target_angles_list = sorted({int(a) for a in target_angles}) if target_angles else None
        if target_angles_list is not None:
            extra_angles.update(target_angles_list)

        for driver, driver_data in dataset.items():
            try:
                freq, angles, spl_matrix, phase_matrix = create_polar_matrix_from_dict(driver_data)
            except Exception as exc:
                print(f"Warning: Skipping extra driver '{driver}' due to invalid data: {exc}")
                continue

            if len(angles) == 0:
                continue

            extra_drivers.append(driver)

            n_points = len(freq)
            if n_points > target_points:
                log_indices = np.unique(np.logspace(0, np.log10(n_points - 1), target_points).astype(int))
                freq_dec = freq[log_indices]
                spl_dec = spl_matrix[log_indices, :]
                phase_dec = phase_matrix[log_indices, :]
            else:
                freq_dec = freq
                spl_dec = spl_matrix
                phase_dec = phase_matrix

            source_angles = [int(a) for a in angles]
            if target_angles_list is not None:
                interp_angles, spl_dec, phase_dec, missing = self._interpolate_angle_grid(
                    angles, spl_dec, phase_dec, target_angles_list
                )
                angles_used = interp_angles
                if missing:
                    interp_info[driver] = {
                        "source_angles": source_angles,
                        "missing_angles": missing
                    }
            else:
                angles_used = source_angles

            extra_all_data[driver] = {
                'freq': freq_dec.tolist(),
                'angles': angles_used,
                'spl': {int(angles_used[i]): spl_dec[:, i].tolist() for i in range(len(angles_used))},
                'phase': {int(angles_used[i]): phase_dec[:, i].tolist() for i in range(len(angles_used))}
            }
            if target_angles_list is None:
                extra_angles.update(angles_used)

        return extra_all_data, sorted(extra_angles), extra_drivers, interp_info

    def plot_di_comparison(self, save_static=True, save_interactive=True):
        """Generate DI comparison plot for all drivers"""
        print("Generating DI comparison plot...")

        if save_static:
            fig, ax = plt.subplots(figsize=config.FIG_SIZE_STATIC)
            self._plot_drivers_static(ax, 'di')
            self._finalize_static_plot(ax, ylabel='Directivity Index (dB)',
                                       title='Directivity Index vs Frequency - All Drivers')
            plt.tight_layout()
            plt.savefig(self.static_plots_dir / 'core/di_comparison.png', dpi=config.DPI_STATIC)
            plt.close()

        if save_interactive:
            fig = go.Figure()
            self._plot_drivers_interactive(fig, 'di')
            self._finalize_interactive_plot(fig, title='Directivity Index vs Frequency',
                                           ylabel='Directivity Index (dB)')
            fig.write_html(self.interactive_plots_dir / 'di_comparison.html')

    def plot_beamwidth_comparison(self, save_static=True, save_interactive=True):
        """Generate beamwidth comparison plot for all drivers"""
        print("Generating beamwidth comparison plot...")

        if save_static:
            fig, ax = plt.subplots(figsize=config.FIG_SIZE_STATIC)
            self._plot_drivers_static(ax, 'beamwidth_6db')
            self._finalize_static_plot(ax, ylabel='Beamwidth (degrees)',
                                       title='-6dB Beamwidth vs Frequency')
            ax.set_ylim(0, 180)
            plt.tight_layout()
            plt.savefig(self.static_plots_dir / 'core/beamwidth_comparison.png', dpi=config.DPI_STATIC)
            plt.close()

        if save_interactive:
            fig = go.Figure()
            self._plot_drivers_interactive(fig, 'beamwidth_6db')
            self._finalize_interactive_plot(fig, title='-6dB Beamwidth vs Frequency',
                                           ylabel='Beamwidth (degrees)')
            fig.update_layout(yaxis_range=[0, 180])
            fig.write_html(self.interactive_plots_dir / 'beamwidth_comparison.html')

    def plot_frequency_response_by_angle(self, save_static=True, save_interactive=True):
        """Generate frequency response plots for each driver at multiple angles.

        Shows SPL vs frequency with curves for:
        - 0° (on-axis): black
        - 30°: green
        - 60°: red
        - 90°: pink
        """
        print("Generating frequency response by angle plots...")

        # Define the angles and their colors
        angle_config = [
            (0, 'black', '0° (on-axis)'),
            (30, 'green', '30°'),
            (60, 'red', '60°'),
            (90, 'hotpink', '90°'),
        ]

        for driver in self.drivers:
            res = self.calc_results[driver]
            freq = res['frequencies']
            angles = np.array(res['angles'])  # Available angles like [0, 15, 30, 45, 60, 75, 90]
            spl_matrix = res['spl_matrix']  # Shape: [n_freqs, n_angles]

            if save_static:
                fig, ax = plt.subplots(figsize=config.FIG_SIZE_STATIC)

                for angle, color, label in angle_config:
                    angle_matches = np.where(angles == angle)[0]
                    if len(angle_matches) > 0:
                        angle_idx = angle_matches[0]
                        spl = spl_matrix[:, angle_idx]
                        ax.semilogx(freq, spl, label=label, linewidth=2, color=color)

                self._add_static_grid(ax)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('SPL (dB)')
                ax.set_title(f'{driver} - Frequency Response at Multiple Angles', fontweight='bold')
                ax.legend(loc='lower left')
                ax.set_xlim(config.FREQ_MIN, config.FREQ_MAX)

                # Add horizontal lines at 5 dB increments
                # 10 dB increments (20, 30, 40...) are solid/darker, 5 dB (25, 35...) are dotted/lighter
                y_min, y_max = ax.get_ylim()
                y_start = int(np.floor(y_min / 5) * 5)
                y_end = int(np.ceil(y_max / 5) * 5)
                for y_val in range(y_start, y_end + 1, 5):
                    if y_val % 10 == 0:
                        # 10 dB increments: solid, darker
                        ax.axhline(y_val, color='gray', linestyle='-', linewidth=0.8, alpha=0.6)
                    else:
                        # 5 dB increments: dotted, lighter
                        ax.axhline(y_val, color='gray', linestyle=':', linewidth=0.5, alpha=0.4)

                plt.tight_layout()
                plt.savefig(self.static_plots_dir / f'core/{driver}_freq_response_angles.png')
                plt.close()

                # Generate NORMALIZED version (on-axis = 0 dB reference)
                # Find on-axis (0°) data
                onaxis_matches = np.where(angles == 0)[0]
                if len(onaxis_matches) > 0:
                    onaxis_idx = onaxis_matches[0]
                    onaxis_spl = spl_matrix[:, onaxis_idx]

                    fig_norm, ax_norm = plt.subplots(figsize=config.FIG_SIZE_STATIC)

                    for angle, color, label in angle_config:
                        angle_matches = np.where(angles == angle)[0]
                        if len(angle_matches) > 0:
                            angle_idx = angle_matches[0]
                            spl = spl_matrix[:, angle_idx]
                            # Normalize: subtract on-axis response
                            spl_normalized = spl - onaxis_spl
                            ax_norm.semilogx(freq, spl_normalized, label=label, linewidth=2, color=color)

                    self._add_static_grid(ax_norm)
                    ax_norm.set_xlabel('Frequency (Hz)')
                    ax_norm.set_ylabel('Relative SPL (dB)')
                    ax_norm.set_title(f'{driver} - Normalized Frequency Response (0° = ref)', fontweight='bold')
                    ax_norm.legend(loc='lower left')
                    ax_norm.set_xlim(config.FREQ_MIN, config.FREQ_MAX)

                    # Set Y axis to show deviation from 0 dB
                    ax_norm.set_ylim(-30, 5)

                    # Add horizontal lines at 5 dB increments
                    for y_val in range(-30, 6, 5):
                        if y_val % 10 == 0:
                            ax_norm.axhline(y_val, color='gray', linestyle='-', linewidth=0.8, alpha=0.6)
                        else:
                            ax_norm.axhline(y_val, color='gray', linestyle=':', linewidth=0.5, alpha=0.4)

                    # Add 0 dB reference line (bold)
                    ax_norm.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

                    plt.tight_layout()
                    plt.savefig(self.static_plots_dir / f'core/{driver}_freq_response_normalized.png')
                    plt.close()

            if save_interactive:
                fig = go.Figure()

                for angle, color, label in angle_config:
                    angle_matches = np.where(angles == angle)[0]
                    if len(angle_matches) > 0:
                        angle_idx = angle_matches[0]
                        spl = spl_matrix[:, angle_idx]
                        fig.add_trace(go.Scatter(
                            x=freq, y=spl, name=label,
                            line=dict(width=2, color=color)
                        ))

                self._add_interactive_grid(fig)
                self._configure_interactive_axis(fig)

                # Add horizontal lines at 5 dB increments
                # 10 dB increments (20, 30, 40...) are solid/darker, 5 dB (25, 35...) are dotted/lighter
                # Get data range from all traces
                all_y = np.concatenate([spl_matrix[:, np.where(angles == a)[0][0]]
                                        for a, _, _ in angle_config
                                        if len(np.where(angles == a)[0]) > 0])
                y_min, y_max = np.min(all_y), np.max(all_y)
                y_start = int(np.floor(y_min / 5) * 5)
                y_end = int(np.ceil(y_max / 5) * 5)
                for y_val in range(y_start, y_end + 1, 5):
                    if y_val % 10 == 0:
                        # 10 dB increments: solid, darker
                        fig.add_hline(y=y_val, line_dash="solid", line_color="gray", opacity=0.6)
                    else:
                        # 5 dB increments: dotted, lighter
                        fig.add_hline(y=y_val, line_dash="dot", line_color="gray", opacity=0.4)

                fig.update_layout(
                    title=f'{driver} - Frequency Response at Multiple Angles',
                    xaxis_title='Frequency (Hz)',
                    yaxis_title='SPL (dB)',
                    legend=dict(x=0.02, y=0.02, xanchor='left', yanchor='bottom')
                )
                self._write_compressed_html(
                    fig,
                    self.interactive_plots_dir / f'{driver}_freq_response_angles.html',
                    f'{driver} Frequency Response'
                )

    def plot_frequency_response_explorer(self):
        """Generate advanced interactive frequency response explorer.

        Features:
        - Multi-driver selection (click to toggle, overlay multiple)
        - All angles available (0-90° in measurement increments)
        - Per-driver level offset sliders for comparison
        - Different markers for each driver to distinguish overlaid curves

        Performance optimizations:
        - Decimates data to ~500 log-spaced points (from 50K+)
        - Uses lines-only mode (no markers) for fast rendering
        """
        print("Generating frequency response explorer...")

        # Target number of points for interactive display (log-spaced)
        TARGET_POINTS = 500

        # Collect all data with decimation
        all_data = {}
        all_angles = set()

        for driver in self.drivers:
            res = self.calc_results[driver]
            freq = res['frequencies']
            angles = res['angles']
            spl_matrix = res['spl_matrix']
            phase_matrix = res['phase_matrix']

            # Decimate to TARGET_POINTS using log-spaced indices
            n_points = len(freq)
            if n_points > TARGET_POINTS:
                # Create log-spaced indices
                log_indices = np.unique(np.logspace(0, np.log10(n_points - 1), TARGET_POINTS).astype(int))
                freq_dec = freq[log_indices]
                spl_dec = spl_matrix[log_indices, :]
                phase_dec = phase_matrix[log_indices, :]
            else:
                freq_dec = freq
                spl_dec = spl_matrix
                phase_dec = phase_matrix

            all_data[driver] = {
                'freq': freq_dec.tolist(),
                'angles': [int(a) for a in angles],
                'spl': {int(angles[i]): spl_dec[:, i].tolist() for i in range(len(angles))},
                'phase': {int(angles[i]): phase_dec[:, i].tolist() for i in range(len(angles))}
            }
            all_angles.update(angles)

        all_angles = sorted(all_angles)

        extra_datasets = {}
        extra_set_name = "juan-baffleless"
        extra_data = self._load_extra_set_data(extra_set_name)
        extra_interp_note = ""
        if extra_data:
            extra_all_data, extra_angles, extra_drivers, extra_interp_info = self._build_explorer_payload(
                extra_data, TARGET_POINTS, target_angles=all_angles
            )
            if extra_drivers:
                extra_datasets[extra_set_name] = {
                    "label": "Juan baffleless drivers",
                    "drivers": extra_drivers,
                    "allData": extra_all_data,
                    "allAngles": [int(a) for a in extra_angles],
                }
                extra_interp_note = self._format_interpolation_note(all_angles, extra_interp_info)

        extra_controls_html = ""
        if extra_datasets:
            info_html = ""
            if extra_interp_note:
                note = html.escape(extra_interp_note).replace("\n", "&#10;")
                info_html = f'<span class="info-icon" title="{note}">(i)</span>'
            extra_driver_note = ""
            if extra_drivers:
                driver_list = ", ".join(sorted(extra_drivers))
                extra_driver_note = f'<div class="extra-driver-note">Adds: {driver_list}</div>'
            extra_controls_html = f'''
                <div class="extra-driver-controls">
                    <div class="extra-driver-row">
                        <button id="loadExtraJuanBtn" onclick="loadExtraDrivers('juan-baffleless')">Load Juan baffleless drivers</button>
                        {info_html}
                    </div>
                    {extra_driver_note}
                </div>
            '''

        # Build the HTML with embedded JavaScript
        html_content = f'''{HTML_DOCTYPE}{HTML_HEAD_START}    <title>Frequency Response Explorer</title>
{HTML_PLOTLY_SCRIPT}{HTML_YAML_SCRIPT}    <style>
{CSS_RESET}        .container {{
            display: flex;
            height: 100vh;
        }}
        .sidebar {{
            width: 320px;
            background: white;
            padding: 20px;
            overflow-y: auto;
            border-right: 1px solid #ddd;
            flex-shrink: 0;
        }}
        .plot-area {{
            flex: 1;
            padding: 10px;
        }}
        #plot {{
            width: 100%;
            height: 100%;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h2 {{
            font-size: 1.1rem;
            margin-bottom: 15px;
            color: #333;
            border-bottom: 2px solid #2563eb;
            padding-bottom: 8px;
        }}
        h3 {{
            font-size: 0.9rem;
            margin: 15px 0 10px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .driver-list {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .extra-driver-controls {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin: 6px 0 12px;
        }}
        .extra-driver-row {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .extra-driver-controls button {{
            padding: 6px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: #f8fafc;
            cursor: pointer;
            font-size: 0.8rem;
        }}
        .extra-driver-controls button:hover {{
            background: #eef2ff;
        }}
        .extra-driver-controls button:disabled {{
            cursor: default;
            opacity: 0.6;
        }}
        .info-icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            border: 1px solid #cbd5e1;
            color: #64748b;
            font-size: 0.7rem;
            cursor: help;
            background: #fff;
        }}
        .extra-driver-note {{
            font-size: 0.75rem;
            color: #666;
        }}
        .driver-item {{
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            transition: all 0.2s;
        }}
        .driver-item:hover {{
            border-color: #2563eb;
            background: #f8fafc;
        }}
        .driver-item.active {{
            border-color: var(--driver-color);
            background: color-mix(in srgb, var(--driver-color) 10%, white);
        }}
        .driver-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
        }}
        .driver-checkbox {{
            width: 18px;
            height: 18px;
            cursor: pointer;
            accent-color: var(--driver-color);
        }}
        .driver-line {{
            font-family: monospace;
            font-size: 1.1rem;
            letter-spacing: -2px;
            color: var(--driver-color);
        }}
        .driver-name {{
            flex: 1;
            font-weight: 500;
        }}
        .offset-control {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid #eee;
            font-size: 0.85rem;
            color: #666;
        }}
        .offset-control input[type="range"] {{
            flex: 1;
            min-width: 80px;
        }}
        .offset-control input[type="number"] {{
            width: 55px;
            padding: 3px 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
            text-align: center;
        }}
        .delay-control {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin-top: 6px;
            font-size: 0.85rem;
            color: #666;
        }}
        .delay-control input[type="number"] {{
            width: 70px;
            padding: 3px 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
            text-align: center;
        }}
        .invert-control {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin-top: 6px;
            font-size: 0.85rem;
            color: #666;
        }}
        .invert-control label {{
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
        }}
        .invert-control input[type="checkbox"] {{
            width: 16px;
            height: 16px;
            cursor: pointer;
            accent-color: var(--driver-color);
        }}
        .angle-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 6px;
        }}
        .angle-btn {{
            padding: 8px 4px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background: white;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
        }}
        .angle-btn:hover {{
            border-color: #2563eb;
        }}
        .angle-btn.active {{
            background: #2563eb;
            color: white;
            border-color: #2563eb;
        }}
        .quick-select {{
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
        }}
        .quick-btn {{
            padding: 6px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            cursor: pointer;
            font-size: 0.8rem;
        }}
        .quick-btn:hover {{
            background: #f0f0f0;
        }}
        /* Display options styles */
        .display-options {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 15px;
        }}
        .option-radio-row {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            font-size: 0.9rem;
            color: #444;
        }}
        .option-radio-row .option-label {{
            font-weight: 600;
            color: #444;
        }}
        .option-radio {{
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
        }}
        .option-radio input {{
            width: 16px;
            height: 16px;
            cursor: pointer;
        }}
        .option-checkbox {{
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            color: #444;
        }}
        .option-checkbox input {{
            width: 16px;
            height: 16px;
            cursor: pointer;
        }}
        /* Filter section styles */
        .filter-section {{
            margin-top: 20px;
            padding-top: 15px;
            border-top: 2px solid #e0e0e0;
        }}
        .filter-controls {{
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }}
        .timing-controls {{
            margin-bottom: 15px;
        }}
        .io-label {{
            min-width: 44px;
            font-size: 0.8rem;
            color: #666;
            font-weight: 600;
            letter-spacing: 0.3px;
        }}
        .filter-controls button {{
            padding: 6px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            cursor: pointer;
            font-size: 0.8rem;
        }}
        .filter-controls button:hover {{
            background: #f0f0f0;
        }}
        .filter-controls button.primary {{
            background: #2563eb;
            color: white;
            border-color: #2563eb;
        }}
        .filter-controls button.primary:hover {{
            background: #1d4ed8;
        }}
        .filter-controls button.danger {{
            background: #fee2e2;
            color: #dc2626;
            border-color: #fecaca;
        }}
        .filter-controls button.danger:hover {{
            background: #fecaca;
        }}
        .filter-driver-select {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
        }}
        .filter-driver-select select {{
            flex: 1;
            padding: 6px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.9rem;
        }}
        .filter-list {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: 400px;
            overflow-y: auto;
        }}
        .filter-card {{
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: #fafafa;
        }}
        .filter-card.disabled {{
            opacity: 0.5;
        }}
        .filter-card-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}
        .filter-card-header input[type="checkbox"] {{
            width: 16px;
            height: 16px;
        }}
        .filter-card-header select {{
            flex: 1;
            padding: 4px 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.85rem;
        }}
        .filter-card-header .delete-btn {{
            padding: 2px 8px;
            border: none;
            background: #fee2e2;
            color: #dc2626;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
        }}
        .filter-card-header .delete-btn:hover {{
            background: #fecaca;
        }}
        .filter-param {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin-top: 6px;
            font-size: 0.8rem;
        }}
        .filter-param label {{
            width: 45px;
            color: #666;
        }}
        .filter-param input[type="range"] {{
            flex: 1;
        }}
        .filter-param input[type="number"] {{
            width: 70px;
            padding: 3px 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
            text-align: right;
        }}
        .filter-param span {{
            width: 25px;
            color: #888;
        }}
        .no-filters {{
            color: #888;
            font-size: 0.85rem;
            font-style: italic;
            padding: 10px;
            text-align: center;
        }}
        input[type="file"] {{
            display: none;
        }}
        .optimize-section {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e0e0e0;
        }}
        .optimize-section h4 {{
            margin: 0 0 10px 0;
            font-size: 0.9rem;
            color: #444;
        }}
        .optimize-controls {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .opt-param {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.8rem;
        }}
        .opt-param label {{
            min-width: 90px;
            color: #666;
        }}
        .opt-param input[type="number"] {{
            width: 70px;
            padding: 4px 6px;
            border: 1px solid #ddd;
            border-radius: 4px;
            text-align: right;
        }}
        .opt-param select {{
            padding: 4px 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .opt-buttons {{
            display: flex;
            gap: 8px;
            margin-top: 5px;
        }}
        .opt-buttons button {{
            flex: 1;
        }}
        .opt-status {{
            font-size: 0.8rem;
            color: #666;
            padding: 5px 0;
            font-style: italic;
        }}
        .opt-checkbox {{
            display: flex;
            align-items: center;
            gap: 2px;
            font-size: 0.7rem;
            color: #666;
        }}
        .opt-checkbox input {{
            width: 14px;
            height: 14px;
        }}
        /* Visual Crossover Overlay styles */
        .crossover-section {{
            margin-top: 20px;
            padding-top: 15px;
            border-top: 2px solid #e0e0e0;
        }}
        .crossover-controls {{
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
        }}
        .crossover-controls button {{
            padding: 6px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            cursor: pointer;
            font-size: 0.8rem;
        }}
        .crossover-controls button:hover {{
            background: #f0f0f0;
        }}
        .crossover-controls button.primary {{
            background: #10b981;
            color: white;
            border-color: #10b981;
        }}
        .crossover-controls button.primary:hover {{
            background: #059669;
        }}
        .crossover-list {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: 300px;
            overflow-y: auto;
        }}
        .crossover-card {{
            padding: 10px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            background: #f0fdf4;
        }}
        .crossover-card.disabled {{
            opacity: 0.5;
            background: #fafafa;
        }}
        .crossover-card-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}
        .crossover-card-header input[type="checkbox"] {{
            width: 16px;
            height: 16px;
        }}
        .crossover-card-header select {{
            padding: 4px 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.85rem;
            background: white;
        }}
        .crossover-card-header .delete-btn {{
            padding: 2px 8px;
            border: none;
            background: #fee2e2;
            color: #dc2626;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            margin-left: auto;
        }}
        .crossover-card-header .delete-btn:hover {{
            background: #fecaca;
        }}
        .crossover-param {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin-top: 6px;
            font-size: 0.8rem;
        }}
        .crossover-param label {{
            width: 45px;
            color: #666;
        }}
        .crossover-param input[type="range"] {{
            flex: 1;
        }}
        .crossover-param input[type="number"] {{
            width: 70px;
            padding: 3px 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
            text-align: right;
        }}
        .crossover-param span {{
            width: 25px;
            color: #888;
        }}
        .crossover-drivers {{
            display: flex;
            gap: 12px;
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px dashed #d1d5db;
        }}
        .crossover-driver-select {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.8rem;
        }}
        .crossover-driver-select label {{
            color: #666;
            font-weight: 500;
        }}
        .crossover-driver-select select {{
            padding: 4px 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.8rem;
            background: white;
        }}
        .no-crossovers {{
            color: #888;
            font-size: 0.85rem;
            font-style: italic;
            padding: 10px;
            text-align: center;
        }}
        .apply-checkbox {{
            display: flex;
            align-items: center;
            gap: 2px;
            font-size: 0.7rem;
            color: #666;
            cursor: pointer;
        }}
        .apply-checkbox input {{
            width: 14px;
            height: 14px;
        }}
        /* X-O derived filter styles */
        .filter-card.from-crossover {{
            background: #f0f9ff;
            border-color: #7dd3fc;
        }}
        .filter-card.from-crossover.disabled {{
            background: #f8fafc;
            border-color: #e2e8f0;
        }}
        .xo-badge {{
            background: #0ea5e9;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
        }}
        .filter-card.disabled .xo-badge {{
            background: #94a3b8;
        }}
        .xo-filter-type {{
            font-weight: 500;
            font-size: 0.85rem;
            color: #334155;
        }}
        .xo-filter-status {{
            margin-left: auto;
            font-size: 1rem;
            color: #22c55e;
        }}
        .filter-card.disabled .xo-filter-status {{
            color: #cbd5e1;
        }}
        .filter-param.readonly {{
            padding: 4px 0;
        }}
        .readonly-value {{
            font-family: monospace;
            font-size: 0.85rem;
            color: #475569;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>Frequency Response Explorer</h2>

		            <h3>Drivers (click to toggle)</h3>
		            <div class="driver-list" id="driverList"></div>
                    {extra_controls_html}
                <div class="filter-controls timing-controls">
                    <button onclick="resetTimingAdjustments()" title="Clear per-driver delay and invert adjustments">Clear Delay/Invert</button>
                </div>
	
	            <h3>Angles</h3>
	            <div class="quick-select">
	                <button class="quick-btn" onclick="selectAngles([0])">0° only</button>
	                <button class="quick-btn" onclick="selectAngles([0,30,60,90])">0/30/60/90</button>
                <button class="quick-btn" onclick="selectAllAngles()">All</button>
                <button class="quick-btn" onclick="selectAngles([])">None</button>
            </div>
            <div class="angle-grid" id="angleGrid"></div>

	            <h3>Display Options</h3>
	            <div class="display-options">
	                <label class="option-checkbox">
	                    <input type="checkbox" id="showPhase" onchange="updatePlot()">
	                    Show Phase Response
	                </label>
                    <div class="option-radio-row">
                        <span class="option-label">SUM:</span>
                        <label class="option-radio" title="Hide SUM trace">
                            <input type="radio" name="sumMode" value="off" checked onchange="updatePlot()">
                            Off
                        </label>
                        <label class="option-radio" title="Show SUM along with individual drivers">
                            <input type="radio" name="sumMode" value="overlay" onchange="updatePlot()">
                            SUM + Drivers
                        </label>
                        <label class="option-radio" title="Show SUM only (hide individual drivers)">
                            <input type="radio" name="sumMode" value="only" onchange="updatePlot()">
                            SUM Only
                        </label>
                    </div>
	            </div>

	            <div class="filter-section">
	                <h3>Filters (IIR EQ)</h3>
	                <div class="filter-controls">
	                    <button class="primary" onclick="addFilter()">+ Add Filter</button>
                        <button class="danger" onclick="clearAllUserFilters()" title="Clear all user EQ filters (keeps crossovers)">Clear Filters</button>
	                </div>
                    <div class="filter-controls">
                        <span class="io-label">YAML:</span>
	                    <button onclick="document.getElementById('yamlInput').click()">Load</button>
	                    <button onclick="saveFiltersYaml()">Save</button>
                    </div>
                    <div class="filter-controls">
                        <span class="io-label">DSP:</span>
	                    <button onclick="document.getElementById('dspInput').click()" title="Load filters from Hypex Filter Design .dsp file">Load</button>
                        <button onclick="saveDspFile()" title="Export current filters/gain/delay/invert to Hypex Filter Design .dsp">Save</button>
                    </div>
	                <input type="file" id="yamlInput" accept=".yaml,.yml" onchange="loadYamlFile(event)" style="display:none">
	                <input type="file" id="dspInput" accept=".dsp" onchange="handleDspFileLoad(event)" style="display:none">
	                <div class="filter-driver-select">
	                    <label>Driver:</label>
	                    <select id="filterDriverSelect" onchange="selectFilterDriver(this.value)"></select>
	                </div>
                <div class="filter-list" id="filterList">
                    <div class="no-filters">No filters defined</div>
                </div>

                <div class="optimize-section">
                    <h4>Auto-Optimize</h4>
                    <div class="optimize-controls">
                        <div class="opt-param">
                            <label>Target Range:</label>
                            <input type="number" id="optFreqMin" value="200" min="20" max="20000" onchange="saveUiStateToLocalStorage()"> Hz to
                            <input type="number" id="optFreqMax" value="10000" min="20" max="20000" onchange="saveUiStateToLocalStorage()"> Hz
                        </div>
                        <div class="opt-param">
                            <label>Reference Angle:</label>
                            <select id="optRefAngle" onchange="saveUiStateToLocalStorage()"></select>
                        </div>
                        <div class="opt-buttons">
                            <button id="optimizeBtn" class="primary" onclick="runOptimization()">Optimize Selected</button>
                            <button id="undoOptBtn" onclick="undoOptimization()" style="display:none;">Undo</button>
                        </div>
                        <div id="optStatus" class="opt-status">Ready</div>
                    </div>
                </div>
            </div>

            <div class="crossover-section">
                <h3>Visual Crossover Overlay</h3>
                <div class="crossover-controls">
                    <button class="primary" onclick="addCrossover()">+ Add X-O</button>
                </div>
                <div class="crossover-list" id="crossoverList">
                    <div class="no-crossovers">No crossovers defined</div>
                </div>
            </div>
        </div>
        <div class="plot-area">
            <div id="plot"></div>
        </div>
    </div>

    <script>
        // Data embedded from Python
        const allData = {json.dumps(all_data)};
        const drivers = {json.dumps(self.drivers)};
        const allAngles = {json.dumps([int(a) for a in all_angles])};
        const driverColors = {json.dumps(config.DRIVER_COLORS)};
        const defaultColor = '#888888';
        const measurementSetName = '{self.interactive_plots_dir.parent.name}';
        const extraDriverDatasets = {json.dumps(extra_datasets)};

        // Filter types
        const filterTypes = ['Peaking', 'Lowpass', 'Highpass', 'Lowshelf', 'Highshelf', 'Allpass'];
        const filterNeedsGain = {{'Peaking': true, 'Lowshelf': true, 'Highshelf': true, 'Lowpass': false, 'Highpass': false, 'Allpass': false}};

        // Filter response defaults (CamillaDSP-style unless overridden per-filter)
        const DEFAULT_FILTER_SAMPLE_RATE = 48000;

        // State
        let activeDrivers = new Set([drivers[0]]);
        let activeAngles = new Set([0]);
        let driverOffsets = {{}};
        let driverPhaseOffsetsDeg = {{}};
        let driverDelaysMs = {{}};
        drivers.forEach(d => {{
            driverOffsets[d] = 0;
            driverPhaseOffsetsDeg[d] = 0;
            driverDelaysMs[d] = 0;
        }});

	        // Filter state
	        let driverFilters = {{}};
	        drivers.forEach(d => driverFilters[d] = []);
        let selectedFilterDriver = drivers[0];
        const STORAGE_KEY = 'lx521_filters_' + measurementSetName;

            // Track last-loaded DSP so we can reuse its header/template (and preserve unmapped channels).
            let lastLoadedDsp = null;
            let lastLoadedDspFilename = null;

        // Visual Crossover Overlay state
        let visualCrossovers = [];
        const CROSSOVER_STORAGE_KEY = 'lx521_crossovers_' + measurementSetName;
        const UI_STATE_STORAGE_KEY = 'lx521_ui_state_' + measurementSetName;
        let savedUiState = null;

        // Crossover types with their filter specifications and visual fade distance
        // filters: array of {{q: value}} for cascaded biquads
        // fadeOctaves: octaves from Fc to reach -40dB (where curve becomes invisible)
        const crossoverTypes = {{
            'LR2': {{ name: 'Linkwitz-Riley 2nd (12dB/oct)', filters: [{{q: 0.5}}], fadeOctaves: 3.33 }},
            'LR4': {{ name: 'Linkwitz-Riley 4th (24dB/oct)', filters: [{{q: 0.707}}, {{q: 0.707}}], fadeOctaves: 1.67 }},
            'LR8': {{ name: 'Linkwitz-Riley 8th (48dB/oct)', filters: [{{q: 0.707}}, {{q: 0.707}}, {{q: 0.707}}, {{q: 0.707}}], fadeOctaves: 0.83 }},
            'BW2': {{ name: 'Butterworth 2nd (12dB/oct)', filters: [{{q: 0.707}}], fadeOctaves: 3.33 }},
            'BW4': {{ name: 'Butterworth 4th (24dB/oct)', filters: [{{q: 0.541}}, {{q: 1.307}}], fadeOctaves: 1.67 }}
        }};

        // ============ DSP FILE LOADING ============
        // Channel mapping: DSP channel -> driver name(s)
        // Channel 1 = T (SEAS27T), 2 = unused, 3 = UM (10F8424, MU10), 4 = LM (L22MG), 5,6 = unused
        const DSP_CHANNEL_MAP = {{
            1: ['SEAS27T'],           // T - tweeter
            2: null,                   // Unused
            3: ['10F8424', 'MU10'],    // UM - upper-mid (both drivers get same filters)
            4: ['L22MG'],              // LM - lower-mid
            5: null,                   // Unused (W left)
            6: null                    // Unused (W right)
        }};

        // DSP filter type codes -> filter type names
        const DSP_TYPE_MAP = {{
            0: 'Unity',      // ftUnity - passthrough
            2: 'Lowpass',    // ftLowPass2
            4: 'Highpass',   // ftHighPass2
            6: 'Shelf',      // ftShelf2 (check shelfhl: 0=low, 1=high)
            8: 'Peaking',    // ftBoostCut
            10: 'Allpass'    // ftAllPass2
        }};

	        // Adjacent channel pairs for crossover detection: [HP channel, LP channel, HP drivers, LP drivers]
	        // Note: for an N-way, the higher-frequency band uses Highpass, the lower-frequency band uses Lowpass.
	        const DSP_CROSSOVER_PAIRS = [
	            // LM (ch4) lowpass <-> UM (ch3) highpass
	            {{ hpCh: 3, lpCh: 4, hpDrivers: ['10F8424', 'MU10'], lpDrivers: ['L22MG'], name: 'LM-UM' }},
	            // UM (ch3) lowpass <-> T (ch1) highpass
	            {{ hpCh: 1, lpCh: 3, hpDrivers: ['SEAS27T'], lpDrivers: ['10F8424', 'MU10'], name: 'UM-T' }}
	        ];

	        function parseDspFile(content) {{
	            const lines = content.split(/\\r?\\n/);
	            if (lines.length < 28) {{
	                throw new Error('Invalid DSP file: too few lines');
	            }}

	            const measSampleRate = parseFloat(lines[24]) || 48000;

	            // Parse header
	            const header = {{
	                productNr: lines[0],
	                formatVersion: lines[1],
	                buildNr: lines[2],
                    header24: lines[23] ?? '0',
	                measSampleRate: measSampleRate,
	                sampleRate: parseFloat(lines[25]) || 93750
	            }};

                const template = {{
                    channelPreambles: {{}},
                    channelExtraInt: {{}},
                    footerFlag2: '0',
                    anechstop: 65536,
                    smoothbw: 0.125,
                }};

                const channels = {{}};
            let lineIdx = 28;

            // Parse each channel (1-6)
            for (let ch = 1; ch <= 6; ch++) {{
                // Channels 2-6 have 3 preamble lines
                if (ch > 1) {{
                    template.channelPreambles[ch] = lines.slice(lineIdx, lineIdx + 3);
                    lineIdx += 3;
                }}

                // Channel header: 6 lines
                const biquadCount = parseInt(lines[lineIdx++]) || 15;
		                const delayRaw = parseFloat(lines[lineIdx++]) || 0;
		                const gain = parseFloat(lines[lineIdx++]) || 0;
		                const invert = lines[lineIdx++] === '1';
		                const enabled = lines[lineIdx++] === '1';
		                const extraInt = lines[lineIdx++] ?? '0';
                        template.channelExtraInt[ch] = extraInt;

                const filters = [];

                // Parse 15 filters (18 lines each)
                for (let f = 0; f < 15; f++) {{
                    const typeCode = parseInt(lines[lineIdx++]) || 0;
                    const f1 = parseFloat(lines[lineIdx++]) || 1000;
                    const f2 = parseFloat(lines[lineIdx++]) || 1000;
                    const filterGain = parseFloat(lines[lineIdx++]) || 0;
                    const q1 = parseFloat(lines[lineIdx++]) || 0.707;
                    const q2 = parseFloat(lines[lineIdx++]) || 0.707;
                    const shelfhl = parseInt(lines[lineIdx++]) || 0;

                    // Read spoles/szeros which indicate filter state
                    // True Unity: spoles=0, szeros=0
                    // Disabled filter: type=0 but spoles/szeros preserved from original type
                    const spoles = parseInt(lines[lineIdx++]) || 0;
                    const szeros = parseInt(lines[lineIdx++]) || 0;
                    lineIdx += 2; // zpoles, zzeros

                    // Read biquad coefficients to check for unity (passthrough)
                    lineIdx += 2; // sconst, zconst
                    const b0 = parseFloat(lines[lineIdx++]) || 0;
                    const b1 = parseFloat(lines[lineIdx++]) || 0;
                    const b2 = parseFloat(lines[lineIdx++]) || 0;
                    lineIdx += 2; // a1, a2

                    // Filter state detection based on type and spoles/szeros:
                    // In .dsp format, when a filter is disabled, type becomes 0 but spoles/szeros
                    // are preserved from the original filter type:
                    // - ftUnity always has spoles=0, szeros=0
                    // - ftBoostCut, ftShelf2, etc. have spoles=2, szeros=2
                    //
                    // So:
                    // - type=0 AND spoles=0 AND szeros=0 → was ftUnity → SKIP (not a real filter)
                    // - type=0 AND (spoles≠0 OR szeros≠0) → was non-Unity but DISABLED
                    // - type≠0 → ENABLED filter

                    const wasUnity = (spoles === 0 && szeros === 0);

                    // Skip Unity filter slots entirely (they're empty slots, not real filters)
                    if (typeCode === 0 && wasUnity) {{
                        continue;
                    }}

                    // Determine if filter is disabled
                    // type=0 with non-zero spoles/szeros means it was a real filter that got disabled
                    const isDisabled = (typeCode === 0 && !wasUnity);

                    // Determine filter type
                    let filterType = DSP_TYPE_MAP[typeCode] || 'Unknown';

                    // For disabled filters (type=0 but was non-Unity), infer original type
                    if (isDisabled) {{
                        // Most disabled filters were Peaking (ftBoostCut) or Shelf
                        // Use shelfhl to distinguish: shelfhl=1 means it was Highshelf
                        if (shelfhl === 1) {{
                            filterType = 'Highshelf';
                        }} else if (shelfhl === 0 && Math.abs(filterGain) > 0.01) {{
                            // Has gain, could be Lowshelf or Peaking
                            // Peaking is more common, but if Q is very low it might be shelf
                            filterType = 'Peaking';
                        }} else {{
                            filterType = 'Peaking';
                        }}
                    }}

                    if (filterType === 'Shelf') {{
                        filterType = shelfhl === 1 ? 'Highshelf' : 'Lowshelf';
                    }}

                    // Skip remaining Unity filters (no meaningful parameters)
                    if (filterType === 'Unity') continue;

                    filters.push({{
                        type: filterType,
                        freq: f1,
                        q: q1,
                        gain: filterGain,
                        enabled: !isDisabled,
                        sampleRate: header.sampleRate,
                        // Keep full raw details for potential round-trip export/debugging
                        raw: {{
                            typeCode,
                            f1,
                            f2,
                            gain: filterGain,
                            q1,
                            q2,
                            shelfhl,
                            spoles,
                            szeros,
                            b0,
                            b1,
                            b2,
                        }}
                    }});
                }}

		                // Hypex .dsp delay values are stored as microseconds.
	                // Convert to ms so phase adjustment uses correct units.
	                const delayMs = delayRaw / 1000;

	                channels[ch] = {{
	                    delay: delayMs,
	                    delayRaw: delayRaw,
	                    gain,
	                    invert,
	                    enabled,
	                    filters
	                }};

                // Skip blank line separator between channels (except after channel 6)
                if (ch < 6 && lineIdx < lines.length) {{
                    lineIdx++;
                }}
            }}

                // Parse footer (for template reuse)
                const footer = lines.slice(lineIdx);
                if (footer.length >= 6) {{
                    // Footer structure has a leading blank line, then a 10-line config block
                    if (footer[1] === '1') {{
                        template.footerFlag2 = footer[2] ?? '0';
                    }}
                    const maybeAnech = parseInt(footer[5], 10);
                    if (!Number.isNaN(maybeAnech)) template.anechstop = maybeAnech;
                    const maybeSmooth = parseFloat(footer[9]);
                    if (Number.isFinite(maybeSmooth)) template.smoothbw = maybeSmooth;
                }}

                return {{ header, template, channels }};
	        }}

        function detectCrossovers(dspData) {{
            const detected = [];

            for (const pair of DSP_CROSSOVER_PAIRS) {{
                const hpFilters = (dspData.channels[pair.hpCh]?.filters || [])
                    .filter(f => f.type === 'Highpass' && f.enabled !== false);
                const lpFilters = (dspData.channels[pair.lpCh]?.filters || [])
                    .filter(f => f.type === 'Lowpass' && f.enabled !== false);

                if (hpFilters.length === 0 || lpFilters.length === 0) continue;

                // Group HP filters by frequency (within 1% tolerance)
                const hpByFreq = {{}};
                for (const f of hpFilters) {{
                    const key = Math.round(f.freq);
                    if (!hpByFreq[key]) hpByFreq[key] = [];
                    hpByFreq[key].push(f);
                }}

                // Group LP filters by frequency
                const lpByFreq = {{}};
                for (const f of lpFilters) {{
                    const key = Math.round(f.freq);
                    if (!lpByFreq[key]) lpByFreq[key] = [];
                    lpByFreq[key].push(f);
                }}

                // Find matching frequencies
                for (const freqKey in hpByFreq) {{
                    const hpAtFreq = hpByFreq[freqKey];
                    // Look for LP at same frequency (±1%)
                    let lpAtFreq = null;
                    for (const lpKey in lpByFreq) {{
                        if (Math.abs(parseInt(lpKey) - parseInt(freqKey)) <= parseInt(freqKey) * 0.01) {{
                            lpAtFreq = lpByFreq[lpKey];
                            break;
                        }}
                    }}

                    if (!lpAtFreq) continue;

                    // Determine crossover type based on filter count and Q values
                    const hpCount = hpAtFreq.length;
                    const lpCount = lpAtFreq.length;
                    const avgHpQ = hpAtFreq.reduce((s, f) => s + f.q, 0) / hpCount;
                    const avgLpQ = lpAtFreq.reduce((s, f) => s + f.q, 0) / lpCount;

                    let xoType = null;

                    // LR2: 1 filter each, Q ≈ 0.5
                    if (hpCount === 1 && lpCount === 1 && Math.abs(avgHpQ - 0.5) < 0.1 && Math.abs(avgLpQ - 0.5) < 0.1) {{
                        xoType = 'LR2';
                    }}
                    // LR4: 2 filters each, Q ≈ 0.707
                    else if (hpCount === 2 && lpCount === 2 && Math.abs(avgHpQ - 0.707) < 0.1 && Math.abs(avgLpQ - 0.707) < 0.1) {{
                        xoType = 'LR4';
                    }}
                    // LR8: 4 filters each, Q ≈ 0.707
                    else if (hpCount === 4 && lpCount === 4 && Math.abs(avgHpQ - 0.707) < 0.1 && Math.abs(avgLpQ - 0.707) < 0.1) {{
                        xoType = 'LR8';
                    }}
                    // BW2: 1 filter each, Q ≈ 0.707
                    else if (hpCount === 1 && lpCount === 1 && Math.abs(avgHpQ - 0.707) < 0.1 && Math.abs(avgLpQ - 0.707) < 0.1) {{
                        xoType = 'BW2';
                    }}
                    // BW4: 2 filters each with Q ≈ 0.541 and 1.307
                    else if (hpCount === 2 && lpCount === 2) {{
                        const hpQs = hpAtFreq.map(f => f.q).sort((a, b) => a - b);
                        const lpQs = lpAtFreq.map(f => f.q).sort((a, b) => a - b);
                        if (Math.abs(hpQs[0] - 0.541) < 0.1 && Math.abs(hpQs[1] - 1.307) < 0.15 &&
                            Math.abs(lpQs[0] - 0.541) < 0.1 && Math.abs(lpQs[1] - 1.307) < 0.15) {{
                            xoType = 'BW4';
                        }}
                    }}

	                    // Even if we can't classify it as a standard topology, still surface it as a crossover.
	                    // We'll treat it as "Custom" and (when loaded) apply the exact DSP HP/LP stages.
	                    const inferredType = xoType || 'Custom';
	                    const fadeOctavesHP = hpCount ? (40 / (12 * hpCount)) : 1.5;
	                    const fadeOctavesLP = lpCount ? (40 / (12 * lpCount)) : 1.5;

	                    detected.push({{
	                        freq: parseFloat(freqKey),
	                        type: inferredType,
	                        hpDrivers: pair.hpDrivers,
	                        lpDrivers: pair.lpDrivers,
	                        name: pair.name,
	                        hpFilters: hpAtFreq,
	                        lpFilters: lpAtFreq,
	                        fadeOctavesHP: fadeOctavesHP,
	                        fadeOctavesLP: fadeOctavesLP,
	                    }});
	                }}
	            }}

            return detected;
        }}

	        function loadDspFilters(dspData, detectedCrossovers, clearExisting, unsupportedFilters) {{
	            const dspSampleRate = dspData.header?.sampleRate || DEFAULT_FILTER_SAMPLE_RATE;

	            if (clearExisting) {{
	                // Clear all existing filters and crossovers
	                drivers.forEach(d => {{
	                    driverFilters[d] = [];
	                    driverOffsets[d] = 0;
	                    driverPhaseOffsetsDeg[d] = 0;
	                    driverDelaysMs[d] = 0;
	                }});
	                visualCrossovers = [];
	            }}

	            // Apply channel gains/delay/invert as per-driver offsets
	            for (let ch = 1; ch <= 6; ch++) {{
	                const targetDrivers = DSP_CHANNEL_MAP[ch];
	                if (!targetDrivers) continue;

	                const chData = dspData.channels[ch] || {{}};
	                const channelGain = chData.gain || 0;
	                const channelDelayMs = chData.delay || 0;
	                const channelInvert = !!chData.invert;

		                for (const driver of targetDrivers) {{
		                    if (!drivers.includes(driver)) continue;

		                    driverOffsets[driver] = channelGain;
		                    driverDelaysMs[driver] = channelDelayMs;
		                    driverPhaseOffsetsDeg[driver] = channelInvert ? 180 : 0;

		                    // Update UI slider/input for this driver
		                    const item = document.querySelector(`.driver-item[data-driver="${{driver}}"]`);
		                    if (item) {{
		                        const gainSlider = item.querySelector('.gain-slider');
		                        const gainInput = item.querySelector('.gain-input');
		                        const delayInput = item.querySelector('.delay-input');
                                const invertInput = item.querySelector('.invert-checkbox');
		                        if (gainSlider) gainSlider.value = channelGain;
		                        if (gainInput) gainInput.value = channelGain;
		                        if (delayInput) delayInput.value = Math.round(channelDelayMs * 1000);
                                if (invertInput) invertInput.checked = channelInvert;
		                    }}
		                }}
		            }}

		            // Build set of filters that are part of crossovers we actually represent
		            // (only exclude crossover filters if we can create a corresponding visual crossover)
		            const crossoverFilterSet = new Set();
		
		            // Add crossovers to visualCrossovers
		            for (const xo of detectedCrossovers) {{
		                const hpDrivers = (xo.hpDrivers || []).filter(d => drivers.includes(d));
		                const lpDrivers = (xo.lpDrivers || []).filter(d => drivers.includes(d));
	
		                if (hpDrivers.length > 0 && lpDrivers.length > 0) {{
		                    xo.hpFilters.forEach(f => crossoverFilterSet.add(f));
		                    xo.lpFilters.forEach(f => crossoverFilterSet.add(f));
		
			                    visualCrossovers.push({{
			                        enabled: true,
			                        freq: xo.freq,
			                        type: xo.type,
		                        lpDriver: lpDrivers[0],
		                        hpDriver: hpDrivers[0],
		                        lpDrivers: lpDrivers,
		                        hpDrivers: hpDrivers,
		                        applyFilters: true,
		                        sampleRate: dspSampleRate,
		                        fadeOctavesLP: xo.fadeOctavesLP,
		                        fadeOctavesHP: xo.fadeOctavesHP,
		                        // For non-standard / asymmetric crossovers, keep the exact DSP stages so we can apply them.
		                        lpStages: xo.type === 'Custom' ? xo.lpFilters : null,
		                        hpStages: xo.type === 'Custom' ? xo.hpFilters : null,
		                    }});
			                }}
			            }}

	            // Add non-crossover filters to drivers
	            for (let ch = 1; ch <= 6; ch++) {{
	                const targetDrivers = DSP_CHANNEL_MAP[ch];
	                if (!targetDrivers) continue;

	                const channelFilters = dspData.channels[ch]?.filters || [];

	                for (const f of channelFilters) {{
	                    if (crossoverFilterSet.has(f)) continue; // Skip crossover filters
	                    if (f.enabled === false) continue; // Skip disabled filters
	                    if (!filterTypes.includes(f.type)) continue; // Skip unsupported types

	                    // Add to all target drivers
	                    for (const driver of targetDrivers) {{
	                        if (!drivers.includes(driver)) continue;

                        // Q scaling for Peaking: Q_Hypex = 2 * Q_Camilla
                        // DSP uses Hypex convention, explorer uses Camilla/REW convention
                        let q = f.q;
                        if (f.type === 'Peaking') {{
                            q = f.q / 2.0;
                        }}

	                        driverFilters[driver].push({{
	                            type: f.type,
	                            freq: f.freq,
	                            q: q,
	                            gain: f.gain,
	                            enabled: true,
	                            sampleRate: f.sampleRate || dspSampleRate
	                        }});
	                    }}
	                }}
	            }}

            // Save and update UI
            saveFiltersToLocalStorage();
            saveCrossoversToLocalStorage();
            syncCrossoverFilters();
            renderFilterList();
            renderCrossoverList();
            updatePlot();
	        }}

		        function resetTimingAdjustments() {{
		            drivers.forEach(d => {{
		                driverDelaysMs[d] = 0;
		                driverPhaseOffsetsDeg[d] = 0;
		                const item = document.querySelector(`.driver-item[data-driver="${{d}}"]`);
		                if (item) {{
		                    const delayInput = item.querySelector('.delay-input');
                            const invertInput = item.querySelector('.invert-checkbox');
		                    if (delayInput) delayInput.value = 0;
                            if (invertInput) invertInput.checked = false;
		                }}
		            }});
		            updatePlot();
		        }}

	        async function handleDspFileLoad(event) {{
            const file = event.target.files[0];
            if (!file) return;

            try {{
                const content = await file.text();
                const dspData = parseDspFile(content);
                lastLoadedDsp = dspData;
                lastLoadedDspFilename = file.name || null;
                const detectedCrossovers = detectCrossovers(dspData);

                // Collect unsupported filters and channel info
                const unsupportedFilters = [];
                let summary = 'DSP file loaded:\\n\\n';

                for (let ch = 1; ch <= 6; ch++) {{
                    const targetDrivers = DSP_CHANNEL_MAP[ch];
                    const chData = dspData.channels[ch];
                    const filters = chData?.filters || [];
	                    const gain = chData?.gain || 0;
	                    const delayMs = chData?.delay || 0;
	                    const invert = !!chData?.invert;

                    // Check for unsupported filter types
                    for (const f of filters) {{
                        if (f.type === 'Unknown') {{
                            unsupportedFilters.push(`Ch${{ch}}: Unknown filter type`);
                        }}
                    }}

	                    if (targetDrivers && (filters.length > 0 || gain !== 0 || delayMs !== 0 || invert)) {{
	                        const enabledCount = filters.filter(f => f.enabled !== false).length;
	                        const disabledCount = filters.length - enabledCount;
	                        let chInfo = `Channel ${{ch}} (${{targetDrivers.join(', ')}}): ${{enabledCount}} filters`;
	                        if (disabledCount > 0) chInfo += ` (+${{disabledCount}} disabled)`;
	                        if (gain !== 0) chInfo += `, gain=${{gain.toFixed(1)}}dB`;
	                        if (delayMs !== 0) chInfo += `, delay=${{delayMs.toFixed(3)}}ms`;
	                        if (invert) chInfo += `, invert`;
	                        summary += chInfo + '\\n';
	                    }}
	                }}

                if (detectedCrossovers.length > 0) {{
                    summary += '\\nDetected crossovers:\\n';
                    for (const xo of detectedCrossovers) {{
                        summary += `  ${{xo.name}}: ${{xo.type}} @ ${{xo.freq}} Hz\\n`;
                    }}
                }}

                // Warn about unsupported filters
                if (unsupportedFilters.length > 0) {{
                    summary += '\\n⚠️ Unsupported filters (will be skipped):\\n';
                    for (const uf of unsupportedFilters) {{
                        summary += `  ${{uf}}\\n`;
                    }}
                }}

                // Count existing filters/crossovers
                const existingFilters = Object.values(driverFilters).flat().filter(f => !f.fromCrossover).length;
                const existingXOs = visualCrossovers.length;

	                let confirmMsg = summary + '\\n';
	                const hasExisting = existingFilters > 0 || existingXOs > 0;
	
	                let clearExisting = true;
	                if (hasExisting) {{
	                    confirmMsg += `You have ${{existingFilters}} existing filters and ${{existingXOs}} crossovers.\\n`;
	                    confirmMsg += 'Clear existing filters and crossovers before loading?';
	                    clearExisting = confirm(confirmMsg);
	                }} else {{
	                    confirmMsg += 'Load these filters?';
	                    const proceed = confirm(confirmMsg);
	                    if (!proceed) {{
	                        event.target.value = '';
	                        return;
	                    }}
	                }}
	
	                loadDspFilters(dspData, detectedCrossovers, clearExisting, unsupportedFilters);
	                const loadedCount = Object.values(driverFilters).flat().filter(f => !f.fromCrossover).length;
	                alert(`Loaded ${{detectedCrossovers.length}} crossovers and ${{loadedCount}} filters from DSP file.`);
	            }} catch (err) {{
	                alert('Error loading DSP file: ' + err.message);
	                console.error(err);
	            }}

            // Reset file input
            event.target.value = '';
        }}

	        // ============ BIQUAD IIR CALCULATIONS ============
	        function calcBiquadCoeffs(type, freq, q, gain, sampleRate = 48000) {{
            const w0 = 2 * Math.PI * freq / sampleRate;
            const cosW0 = Math.cos(w0);
            const sinW0 = Math.sin(w0);
            const alpha = sinW0 / (2 * q);
            const A = Math.pow(10, gain / 40);  // sqrt of linear gain

            let b0, b1, b2, a0, a1, a2;

            switch (type) {{
                case 'Peaking':
                    b0 = 1 + alpha * A;
                    b1 = -2 * cosW0;
                    b2 = 1 - alpha * A;
                    a0 = 1 + alpha / A;
                    a1 = -2 * cosW0;
                    a2 = 1 - alpha / A;
                    break;
                case 'Lowpass':
                    b0 = (1 - cosW0) / 2;
                    b1 = 1 - cosW0;
                    b2 = (1 - cosW0) / 2;
                    a0 = 1 + alpha;
                    a1 = -2 * cosW0;
                    a2 = 1 - alpha;
                    break;
                case 'Highpass':
                    b0 = (1 + cosW0) / 2;
                    b1 = -(1 + cosW0);
                    b2 = (1 + cosW0) / 2;
                    a0 = 1 + alpha;
                    a1 = -2 * cosW0;
                    a2 = 1 - alpha;
                    break;
                case 'Lowshelf':
                    const sqrtA_ls = Math.sqrt(A);
                    b0 = A * ((A + 1) - (A - 1) * cosW0 + 2 * sqrtA_ls * alpha);
                    b1 = 2 * A * ((A - 1) - (A + 1) * cosW0);
                    b2 = A * ((A + 1) - (A - 1) * cosW0 - 2 * sqrtA_ls * alpha);
                    a0 = (A + 1) + (A - 1) * cosW0 + 2 * sqrtA_ls * alpha;
                    a1 = -2 * ((A - 1) + (A + 1) * cosW0);
                    a2 = (A + 1) + (A - 1) * cosW0 - 2 * sqrtA_ls * alpha;
                    break;
                case 'Highshelf':
                    const sqrtA_hs = Math.sqrt(A);
                    b0 = A * ((A + 1) + (A - 1) * cosW0 + 2 * sqrtA_hs * alpha);
                    b1 = -2 * A * ((A - 1) + (A + 1) * cosW0);
                    b2 = A * ((A + 1) + (A - 1) * cosW0 - 2 * sqrtA_hs * alpha);
                    a0 = (A + 1) - (A - 1) * cosW0 + 2 * sqrtA_hs * alpha;
                    a1 = 2 * ((A - 1) - (A + 1) * cosW0);
                    a2 = (A + 1) - (A - 1) * cosW0 - 2 * sqrtA_hs * alpha;
                    break;
                case 'Allpass':
                    // 2nd order allpass filter
                    b0 = 1 - alpha;
                    b1 = -2 * cosW0;
                    b2 = 1 + alpha;
                    a0 = 1 + alpha;
                    a1 = -2 * cosW0;
                    a2 = 1 - alpha;
                    break;
                default:
                    return {{ b0: 1, b1: 0, b2: 0, a1: 0, a2: 0 }};
            }}

            return {{
                b0: b0 / a0, b1: b1 / a0, b2: b2 / a0,
                a1: a1 / a0, a2: a2 / a0
            }};
        }}

        function biquadMagnitudeDb(coeffs, freq, sampleRate = 48000) {{
            const w = 2 * Math.PI * freq / sampleRate;
            const cosW = Math.cos(w);
            const cos2W = Math.cos(2 * w);
            const sinW = Math.sin(w);
            const sin2W = Math.sin(2 * w);

            const {{ b0, b1, b2, a1, a2 }} = coeffs;

            const numReal = b0 + b1 * cosW + b2 * cos2W;
            const numImag = -(b1 * sinW + b2 * sin2W);
            const denReal = 1 + a1 * cosW + a2 * cos2W;
            const denImag = -(a1 * sinW + a2 * sin2W);

            const numMag = Math.sqrt(numReal * numReal + numImag * numImag);
            const denMag = Math.sqrt(denReal * denReal + denImag * denImag);

            return 20 * Math.log10(numMag / denMag);
        }}

	        function biquadPhaseDeg(coeffs, freq, sampleRate = 48000) {{
	            // Calculate phase response of biquad filter in degrees
	            const w = 2 * Math.PI * freq / sampleRate;
            const cosW = Math.cos(w);
            const cos2W = Math.cos(2 * w);
            const sinW = Math.sin(w);
            const sin2W = Math.sin(2 * w);

            const {{ b0, b1, b2, a1, a2 }} = coeffs;

            const numReal = b0 + b1 * cosW + b2 * cos2W;
            const numImag = -(b1 * sinW + b2 * sin2W);
            const denReal = 1 + a1 * cosW + a2 * cos2W;
            const denImag = -(a1 * sinW + a2 * sin2W);

            // Phase = arg(H) = arg(num) - arg(den)
            const numPhase = Math.atan2(numImag, numReal);
            const denPhase = Math.atan2(denImag, denReal);

	            return (numPhase - denPhase) * 180 / Math.PI;
	        }}

            // ============ DSP EXPORT ============
            function formatDspSci(value) {{
                const v = Number.isFinite(value) ? value : 0;
                const isNegZero = Object.is(v, -0);
                const isNegative = v < 0 || isNegZero;
                const absExp = Math.abs(v).toExponential(14).replace('e', 'E');
                const parts = absExp.split('E');
                const mantissa = (isNegative ? '-' : '') + (parts[0] || '0.00000000000000');
                const expPart = parts[1] || '+0';
                const sign = expPart[0] === '-' ? '-' : '+';
                const digits = expPart.slice(1).padStart(4, '0');
                const formatted = `${{mantissa}}E${{sign}}${{digits}}`;
                return isNegative ? formatted : (' ' + formatted);
            }}

            function downloadTextFile(content, filename, mimeType = 'text/plain') {{
                const blob = new Blob([content], {{ type: mimeType }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                a.remove();
                URL.revokeObjectURL(url);
            }}

            function pickPrimaryDriverForChannel(mappedDrivers) {{
                if (!Array.isArray(mappedDrivers) || mappedDrivers.length === 0) return null;
                for (const d of mappedDrivers) {{
                    if (drivers.includes(d) && activeDrivers.has(d)) return d;
                }}
                for (const d of mappedDrivers) {{
                    if (drivers.includes(d)) return d;
                }}
                return null;
            }}

            function dspTypeCodeForFilterType(type) {{
                switch (type) {{
                    case 'Lowpass': return 2;
                    case 'Highpass': return 4;
                    case 'Lowshelf':
                    case 'Highshelf': return 6;
                    case 'Peaking': return 8;
                    case 'Allpass': return 10;
                    default: return 0;
                }}
            }}

            function dspShelfHlForFilterType(type) {{
                if (type === 'Highshelf') return 1;
                return 0;
            }}

            function dspPoleZeroCountsForTypeCode(baseTypeCode) {{
                // Mirrors observed Hypex Filter Design exports: Lowpass has szeros=0, others 2.
                if (baseTypeCode === 0) return {{ spoles: 0, szeros: 0, zpoles: 2, zzeros: 2 }};
                if (baseTypeCode === 2) return {{ spoles: 2, szeros: 0, zpoles: 2, zzeros: 2 }};
                return {{ spoles: 2, szeros: 2, zpoles: 2, zzeros: 2 }};
            }}

            function dspNormalizeZeroSigns(typeCode, b0, b1, b2) {{
                let outB1 = b1;
                let outB2 = b2;
                if (typeCode !== 10 && outB1 === 0) {{
                    if (typeCode === 0 && b0 < 0) outB1 = 0;
                    else outB1 = -0;
                }}
                if (outB2 === 0) {{
                    if (typeCode === 10) outB2 = -0;
                    else if (typeCode === 0 && b0 < 0) outB2 = -0;
                }}
                return {{ b1: outB1, b2: outB2 }};
            }}

            function buildDspFilterBlockLines(filter, sampleRate) {{
                const baseTypeCode = dspTypeCodeForFilterType(filter.type);
                if (baseTypeCode === 0) return null;

                const typeCode = filter.enabled ? baseTypeCode : 0;
                const f1 = filter.freq;
                const f2 = 1000.0;
                const gain = filterNeedsGain[filter.type] ? (filter.gain || 0) : 0.0;
                const q1 = filter.q;
                const q2 = 0.7;
                const shelfhl = dspShelfHlForFilterType(filter.type);

                const coeffs = calcBiquadCoeffs(filter.type, f1, Math.max(0.0001, q1), gain, sampleRate);
                const zconst = coeffs.b0;
                const counts = dspPoleZeroCountsForTypeCode(baseTypeCode);
                const zeros = dspNormalizeZeroSigns(typeCode, coeffs.b0, coeffs.b1, coeffs.b2);

                return [
                    String(typeCode),
                    formatDspSci(f1),
                    formatDspSci(f2),
                    formatDspSci(gain),
                    formatDspSci(q1),
                    formatDspSci(q2),
                    String(shelfhl),
                    String(counts.spoles),
                    String(counts.szeros),
                    String(counts.zpoles),
                    String(counts.zzeros),
                    formatDspSci(1.0),         // sconst
                    formatDspSci(zconst),      // zconst
                    formatDspSci(coeffs.b0),   // b0
                    formatDspSci(zeros.b1),    // b1
                    formatDspSci(zeros.b2),    // b2
                    formatDspSci(coeffs.a1),   // a1
                    formatDspSci(coeffs.a2),   // a2
                ];
            }}

            function buildDspUnityBlockLines() {{
                return [
                    '0',
                    formatDspSci(1000.0),
                    formatDspSci(1000.0),
                    formatDspSci(0.0),
                    formatDspSci(0.7),
                    formatDspSci(0.7),
                    '0',
                    '0',
                    '0',
                    '2',
                    '2',
                    formatDspSci(1.0),
                    formatDspSci(1.0),
                    formatDspSci(1.0),
                    formatDspSci(-0),
                    formatDspSci(0.0),
                    formatDspSci(0.0),
                    formatDspSci(0.0),
                ];
            }}

            function buildDspFileText(exportHeader, exportTemplate, channelsOut) {{
                const lines = [];

                const productNr = exportHeader.productNr ?? '3';
                const formatVersion = exportHeader.formatVersion ?? '2';
                const buildNr = exportHeader.buildNr ?? '3';
                const header24 = exportHeader.header24 ?? '0';
                const measSampleRate = exportHeader.measSampleRate ?? 48000.0;
                const sampleRate = exportHeader.sampleRate ?? 93750.0;

                // Header (28 lines)
                lines.push(productNr);
                lines.push(formatVersion);
                lines.push(buildNr);
                lines.push('0', '0', '1', '0', '0');
                lines.push('6');
                lines.push('15', '15', '15', '15', '15', '15');
                lines.push('10', '10', '10', '10', '10', '10');
                lines.push('1', '0', String(header24));
                lines.push(formatDspSci(measSampleRate));
                lines.push(formatDspSci(sampleRate));
                lines.push(formatDspSci(0.0));
                lines.push(formatDspSci(65536.0 / 48000.0));

                // Channels (1-6)
                for (let ch = 1; ch <= 6; ch++) {{
                    if (ch > 1) {{
                        const pre = exportTemplate.channelPreambles?.[ch] || ['1', '1', '0'];
                        lines.push(...pre);
                    }}

                    const chData = channelsOut[ch] || {{ delayRaw: 0, gain: 0, invert: false, enabled: true, filters: [] }};
                    const extraInt = exportTemplate.channelExtraInt?.[ch] ?? '0';
                    const delayRaw = chData.delayRaw || 0;
                    const gain = chData.gain || 0;
                    const invert = !!chData.invert;
                    const enabled = (chData.enabled ?? true) ? '1' : '0';

                    lines.push('15');
                    lines.push(formatDspSci(delayRaw));
                    lines.push(formatDspSci(gain));
                    lines.push(invert ? '1' : '0');
                    lines.push(enabled);
                    lines.push(String(extraInt));

                    const filters = Array.isArray(chData.filters) ? chData.filters : [];
                    for (let i = 0; i < 15; i++) {{
                        if (i < filters.length) {{
                            const block = buildDspFilterBlockLines(filters[i], sampleRate);
                            if (block) lines.push(...block);
                            else lines.push(...buildDspUnityBlockLines());
                        }} else {{
                            lines.push(...buildDspUnityBlockLines());
                        }}
                    }}

                    if (ch < 6) lines.push('');
                }}

                // Footer
                const footerFlag2 = exportTemplate.footerFlag2 ?? '0';
                const anechstop = exportTemplate.anechstop ?? 65536;
                const smoothbw = exportTemplate.smoothbw ?? 0.125;

                lines.push('');
                lines.push('1');
                lines.push(String(footerFlag2));
                lines.push('0');
                lines.push('0');
                lines.push(String(anechstop));
                lines.push('0');
                lines.push('0');
                lines.push('1');
                lines.push(formatDspSci(smoothbw));
                lines.push('0');
                lines.push('/N*');
                lines.push('');
                lines.push('*N/');
                for (let i = 0; i < 10; i++) lines.push('-1');

                return lines.join('\\r\\n') + '\\r\\n';
            }}

            function saveDspFile() {{
                // Build header/template defaults
                const baseHeader = lastLoadedDsp?.header || {{
                    productNr: '3',
                    formatVersion: '2',
                    buildNr: '3',
                    header24: '0',
                    measSampleRate: 48000.0,
                    sampleRate: 93750.0,
                }};
                const baseTemplate = lastLoadedDsp?.template || {{
                    channelPreambles: {{}},
                    channelExtraInt: {{}},
                    footerFlag2: '0',
                    anechstop: 65536,
                    smoothbw: 0.125,
                }};

                // Warn if we don't have a template (unmapped channels will be unity).
                if (!lastLoadedDsp) {{
                    const proceed = confirm(
                        'No DSP template loaded.\\n\\n' +
                        'Unmapped channels will be exported as Unity (empty).\\n' +
                        'If you want to preserve channels not shown in this explorer (e.g. W), load a .dsp first, then export.'
                    );
                    if (!proceed) return;
                }}

                const exportSampleRate = parseFloat(baseHeader.sampleRate) || 93750;

                // Build output channels (1-6)
                const channelsOut = {{}};
                const warnings = [];

                for (let ch = 1; ch <= 6; ch++) {{
                    const mappedDrivers = DSP_CHANNEL_MAP[ch];
                    const primaryDriver = pickPrimaryDriverForChannel(mappedDrivers);

                    if (primaryDriver) {{
                        const delayMs = driverDelaysMs[primaryDriver] || 0;
                        const delayRaw = delayMs * 1000;
                        const gain = driverOffsets[primaryDriver] || 0;
                        const phaseOffset = driverPhaseOffsetsDeg[primaryDriver] || 0;
                        const normPhase = ((phaseOffset % 360) + 360) % 360;
                        const invert = Math.abs(normPhase - 180) < 1e-6;
                        if (!(Math.abs(normPhase) < 1e-6 || invert)) {{
                            warnings.push(
                                'Channel ' + ch + ': driver ' + primaryDriver +
                                ' has non-180 phase offset (' + phaseOffset + '°); DSP export supports only invert.'
                            );
                        }}

                        const rawFilters = driverFilters[primaryDriver] || [];
                        const exportFilters = [];
                        const skipped = [];

                        rawFilters.forEach(f => {{
                            if (!filterTypes.includes(f.type)) {{
                                skipped.push(f.type || 'Unknown');
                                return;
                            }}
                            const baseTypeCode = dspTypeCodeForFilterType(f.type);
                            if (baseTypeCode === 0) {{
                                skipped.push(f.type);
                                return;
                            }}

                            let q = parseFloat(f.q);
                            if (!Number.isFinite(q) || q <= 0) q = 0.707;

                            // Convert Peaking Q to DSP convention (Hypex/Config.xml)
                            if (f.type === 'Peaking') q = q * 2.0;

                            exportFilters.push({{
                                type: f.type,
                                freq: parseFloat(f.freq) || 1000,
                                q: q,
                                gain: parseFloat(f.gain) || 0,
                                enabled: f.enabled !== false,
                            }});
                        }});

                        if (skipped.length > 0) {{
                            warnings.push('Channel ' + ch + ': skipped unsupported filter types: ' + Array.from(new Set(skipped)).join(', '));
                        }}
                        if (exportFilters.length > 15) {{
                            warnings.push('Channel ' + ch + ': has ' + exportFilters.length + ' filters; only first 15 will be exported.');
                            exportFilters.length = 15;
                        }}

                        channelsOut[ch] = {{
                            delayRaw: delayRaw,
                            gain: gain,
                            invert: invert,
                            enabled: true,
                            filters: exportFilters,
                        }};
                    }} else if (lastLoadedDsp?.channels?.[ch]) {{
                        // Preserve channels not mapped into this explorer.
                        const chData = lastLoadedDsp.channels[ch];
                        const exportFilters = [];
                        (chData.filters || []).forEach(f => {{
                            const baseTypeCode = dspTypeCodeForFilterType(f.type);
                            if (baseTypeCode === 0) return;
                            exportFilters.push({{
                                type: f.type,
                                freq: f.freq,
                                q: f.q,          // Already in DSP convention
                                gain: f.gain,
                                enabled: f.enabled !== false,
                            }});
                        }});
                        channelsOut[ch] = {{
                            delayRaw: chData.delayRaw || (chData.delay || 0) * 1000,
                            gain: chData.gain || 0,
                            invert: !!chData.invert,
                            enabled: chData.enabled !== false,
                            filters: exportFilters.slice(0, 15),
                        }};
                    }} else {{
                        // Default empty channel
                        channelsOut[ch] = {{ delayRaw: 0, gain: 0, invert: false, enabled: true, filters: [] }};
                    }}
                }}

                if (warnings.length > 0) {{
                    const proceed = confirm('DSP export warnings:\\n\\n' + warnings.join('\\n') + '\\n\\nContinue export?');
                    if (!proceed) return;
                }}

                const dspText = buildDspFileText(
                    {{ ...baseHeader, sampleRate: exportSampleRate }},
                    baseTemplate,
                    channelsOut
                );

                const defaultName = lastLoadedDspFilename
                    ? lastLoadedDspFilename.replace(/\\.dsp$/i, '_export.dsp')
                    : (measurementSetName + '_export.dsp');
                const filename = prompt('DSP filename:', defaultName) || defaultName;
                downloadTextFile(dspText, filename, 'text/plain');
            }}

	        function getFilterSampleRate(filter) {{
	            return filter.sampleRate || DEFAULT_FILTER_SAMPLE_RATE;
	        }}

	        function calcFilterChainResponse(filters, frequencies) {{
	            return frequencies.map(freq => {{
	                let totalDb = 0;
	                filters.forEach(filter => {{
	                    if (!filter.enabled) return;
	                    const sampleRate = getFilterSampleRate(filter);
	                    const coeffs = calcBiquadCoeffs(filter.type, filter.freq, filter.q, filter.gain || 0, sampleRate);
	                    totalDb += biquadMagnitudeDb(coeffs, freq, sampleRate);
	                }});
	                return totalDb;
	            }});
	        }}

	        function calcFilterChainPhaseResponse(filters, frequencies) {{
	            // Calculate total phase response of filter chain (in degrees)
	            return frequencies.map(freq => {{
	                let totalPhase = 0;
	                filters.forEach(filter => {{
	                    if (!filter.enabled) return;
	                    const sampleRate = getFilterSampleRate(filter);
	                    const coeffs = calcBiquadCoeffs(filter.type, filter.freq, filter.q, filter.gain || 0, sampleRate);
	                    totalPhase += biquadPhaseDeg(coeffs, freq, sampleRate);
	                }});
	                return totalPhase;
	            }});
	        }}

        function wrapPhase(phase) {{
            // Wrap phase to -180 to +180 range
            while (phase > 180) phase -= 360;
            while (phase < -180) phase += 360;
            return phase;
        }}

	        // ============ CROSSOVER FILTER MANAGEMENT ============
	        // Crossovers generate actual IIR filter entries in driverFilters
	        // These are marked with fromCrossover property and shown as read-only

	        function getXoDriverList(xo, side) {{
	            const list = side === 'lp' ? xo.lpDrivers : xo.hpDrivers;
	            if (Array.isArray(list) && list.length > 0) return list;
	            const single = side === 'lp' ? xo.lpDriver : xo.hpDriver;
	            return single ? [single] : [];
	        }}

	        function syncCrossoverFilters() {{
	            // Remove all existing crossover-derived filters and regenerate from current crossovers
	            drivers.forEach(driver => {{
                // Remove old crossover filters (use !== undefined, not !f.fromCrossover, since 0 is falsy)
                driverFilters[driver] = (driverFilters[driver] || []).filter(f => f.fromCrossover === undefined);

		                // Add new crossover filters
		                visualCrossovers.forEach((xo, xoIdx) => {{
		                    if (!xo.enabled) return;

		                    let filterType = null;
		                    if (getXoDriverList(xo, 'lp').includes(driver)) filterType = 'Lowpass';
		                    else if (getXoDriverList(xo, 'hp').includes(driver)) filterType = 'Highpass';
		                    if (!filterType) return;

		                    const sampleRate = xo.sampleRate || DEFAULT_FILTER_SAMPLE_RATE;

		                    // If this X-O came from DSP import and has explicit stages, apply them exactly.
		                    const dspStages = filterType === 'Lowpass' ? xo.lpStages : xo.hpStages;
		                    if (Array.isArray(dspStages) && dspStages.length > 0) {{
		                        dspStages.forEach((stage, stageIdx) => {{
		                            driverFilters[driver].push({{
		                                type: filterType,
		                                freq: xo.freq,
		                                q: stage.q,
		                                gain: stage.gain || 0,
		                                enabled: xo.applyFilters,  // Controlled by X-O "Filt" checkbox
		                                sampleRate: stage.sampleRate || sampleRate,
		                                optimize: false,
		                                fromCrossover: xoIdx,      // Mark as X-O derived
		                                xoType: xo.type,
		                                xoStage: stageIdx + 1,
		                                xoTotalStages: dspStages.length
		                            }});
		                        }});
		                        return;
		                    }}

		                    // Otherwise use the synthetic crossover generator (UI-created X-Os).
		                    const xoType = crossoverTypes[xo.type];
		                    if (!xoType) return;

		                    // Create a filter entry for each cascaded biquad stage
		                    xoType.filters.forEach((stage, stageIdx) => {{
		                        driverFilters[driver].push({{
		                            type: filterType,
		                            freq: xo.freq,
		                            q: stage.q,
		                            gain: 0,
		                            enabled: xo.applyFilters,  // Controlled by X-O "Filt" checkbox
		                            sampleRate: sampleRate,
		                            optimize: false,
		                            fromCrossover: xoIdx,      // Mark as X-O derived
		                            xoType: xo.type,
		                            xoStage: stageIdx + 1,
		                            xoTotalStages: xoType.filters.length
		                        }});
		                    }});
		                }});
		            }});
		        }}

	        function hasVisualClippingForDriver(driver) {{
	            // Check if driver has any enabled crossover (for visual clipping, regardless of applyFilters)
	            return visualCrossovers.some(xo =>
	                xo.enabled && (getXoDriverList(xo, 'lp').includes(driver) || getXoDriverList(xo, 'hp').includes(driver))
	            );
	        }}

        function getDriverFadeInfo(driver, frequencies) {{
            // Get fade information for each frequency point
            // Returns array of {{ show: boolean, fadeRatio: 0-1 }} where:
            //   show=false means clip (beyond fadeOctaves)
            //   fadeRatio=0 means full color, fadeRatio=1 means fully faded to white

            return frequencies.map(freq => {{
                let show = true;
                let maxFadeRatio = 0;  // Track maximum fade across all crossovers

	                visualCrossovers.forEach(xo => {{
	                    if (!xo.enabled) return;
	                    const xoType = crossoverTypes[xo.type];
	                    const fadeOctavesLP = xo.fadeOctavesLP || xoType?.fadeOctaves || 1.5;
	                    const fadeOctavesHP = xo.fadeOctavesHP || xoType?.fadeOctaves || 1.5;
	                    const logFc = Math.log2(xo.freq);
	                    const logF = Math.log2(freq);
		                    const octaveDistance = logF - logFc;  // positive = above Fc, negative = below

		                    // LP driver: fade as frequency goes above Fc
		                    if (getXoDriverList(xo, 'lp').includes(driver) && octaveDistance > 0) {{
		                        if (octaveDistance > fadeOctavesLP) {{
		                            show = false;
		                        }} else {{
	                            // Fade ratio: 0 at Fc, 1 at Fc + fadeOctaves
	                            const ratio = octaveDistance / fadeOctavesLP;
	                            maxFadeRatio = Math.max(maxFadeRatio, ratio);
	                        }}
		                    }}
		                    // HP driver: fade as frequency goes below Fc
		                    if (getXoDriverList(xo, 'hp').includes(driver) && octaveDistance < 0) {{
		                        if (octaveDistance < -fadeOctavesHP) {{
		                            show = false;
		                        }} else {{
	                            // Fade ratio: 0 at Fc, 1 at Fc - fadeOctaves
	                            const ratio = Math.abs(octaveDistance) / fadeOctavesHP;
	                            maxFadeRatio = Math.max(maxFadeRatio, ratio);
	                        }}
	                    }}
	                }});
                return {{ show, fadeRatio: maxFadeRatio }};
            }});
        }}

        function interpolateColor(color1, color2, ratio) {{
            // Interpolate between two hex colors
            // ratio: 0 = color1, 1 = color2
            const hex = (c) => parseInt(c, 16);
            const r1 = hex(color1.slice(1, 3)), g1 = hex(color1.slice(3, 5)), b1 = hex(color1.slice(5, 7));
            const r2 = hex(color2.slice(1, 3)), g2 = hex(color2.slice(3, 5)), b2 = hex(color2.slice(5, 7));
            const r = Math.round(r1 + (r2 - r1) * ratio);
            const g = Math.round(g1 + (g2 - g1) * ratio);
            const b = Math.round(b1 + (b2 - b1) * ratio);
            return `rgb(${{r}},${{g}},${{b}})`;
        }}

        function getDriverFadeColors(driver, frequencies, baseColor) {{
            // Get per-point colors with fade effect
            // Returns {{ colors: array, clipMask: array }}
            const fadeInfo = getDriverFadeInfo(driver, frequencies);
            const white = '#ffffff';
            const colors = fadeInfo.map(info => {{
                // Always return a valid color (Plotly needs colors for all points)
                // Clipped points will be handled by null in y-data
                if (info.fadeRatio === 0) return baseColor;
                return interpolateColor(baseColor, white, info.fadeRatio * 0.85);  // Max 85% fade toward white
            }});
            const clipMask = fadeInfo.map(info => info.show);
            return {{ colors, clipMask }};
        }}

        function createFadingSegments(xData, yData, fadeColors, clipMask, lineWidth, dashPattern, name, showInLegend) {{
            // Create multiple line segments, each with its own color, to simulate gradient fading
            // Uses quantized fade levels (10 steps) to avoid too many segments while showing gradient
            const segments = [];

            // Quantize colors to reduce number of segments
            // Handles both hex (#rrggbb) and rgb(r,g,b) formats
            function quantizeColor(color) {{
                let r, g, b;

                // Check for hex format
                if (color.startsWith('#')) {{
                    r = parseInt(color.slice(1, 3), 16);
                    g = parseInt(color.slice(3, 5), 16);
                    b = parseInt(color.slice(5, 7), 16);
                }} else {{
                    // Check for rgb format
                    const match = color.match(/rgb\((\d+),(\d+),(\d+)\)/);
                    if (match) {{
                        r = parseInt(match[1]);
                        g = parseInt(match[2]);
                        b = parseInt(match[3]);
                    }} else {{
                        return color;  // Unknown format, return as-is
                    }}
                }}

                // Quantize to 10 discrete levels (25.5 = 255/10)
                r = Math.round(r / 25.5) * 25.5;
                g = Math.round(g / 25.5) * 25.5;
                b = Math.round(b / 25.5) * 25.5;
                return `rgb(${{Math.round(r)}},${{Math.round(g)}},${{Math.round(b)}})`;
            }}

            const quantizedColors = fadeColors.map(quantizeColor);
            let segStart = 0;

            for (let i = 1; i <= xData.length; i++) {{
                // Check if we need to end a segment (color change, clip boundary, or end of data)
                const endSegment = i === xData.length ||
                    quantizedColors[i] !== quantizedColors[i-1] ||
                    clipMask[i] !== clipMask[i-1];

                if (endSegment && segStart < i) {{
                    // Create segment from segStart to i-1 (inclusive)
                    // Include next point for continuity (overlap between segments)
                    const endIdx = Math.min(i + 1, xData.length);
                    const segX = xData.slice(segStart, endIdx);
                    const segY = yData.slice(segStart, endIdx);

                    // Only add segment if it should be visible (clipMask)
                    if (clipMask[segStart]) {{
                        segments.push({{
                            x: segX,
                            y: segY,
                            name: name,
                            mode: 'lines',
                            line: {{
                                color: quantizedColors[segStart],
                                width: lineWidth,
                                dash: dashPattern
                            }},
                            showlegend: showInLegend && segments.length === 0,  // Only first segment shows in legend
                            legendgroup: name,
                            hoverinfo: 'x+y+name'
                        }});
                    }}
                    segStart = i;
                }}
            }}

            return segments;
        }}

        // ============ NELDER-MEAD OPTIMIZER ============
        function nelderMead(objective, x0, bounds, options = {{}}) {{
            const maxIter = options.maxIter || 300;
            const tol = options.tol || 1e-6;
            const alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5;
            const n = x0.length;
            const penalty = 1000;

            function penalizedObjective(x) {{
                let p = 0;
                for (let i = 0; i < n; i++) {{
                    if (x[i] < bounds.min[i]) p += penalty * Math.pow(bounds.min[i] - x[i], 2);
                    if (x[i] > bounds.max[i]) p += penalty * Math.pow(x[i] - bounds.max[i], 2);
                }}
                return objective(x) + p;
            }}

            let simplex = [x0.slice()];
            for (let i = 0; i < n; i++) {{
                const xi = x0.slice();
                xi[i] += (bounds.max[i] - bounds.min[i]) * 0.05;
                simplex.push(xi);
            }}

            let values = simplex.map(penalizedObjective);
            let lastBestValue = values[0];

            for (let iter = 0; iter < maxIter; iter++) {{
                const order = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
                simplex = order.map(([_, i]) => simplex[i]);
                values = order.map(([v, _]) => v);

                if (values[n] - values[0] < tol) break;

                const centroid = Array(n).fill(0);
                for (let i = 0; i < n; i++) {{
                    for (let j = 0; j < n; j++) centroid[j] += simplex[i][j] / n;
                }}

                const xr = centroid.map((c, j) => c + alpha * (c - simplex[n][j]));
                const fr = penalizedObjective(xr);

                if (fr < values[0]) {{
                    const xe = centroid.map((c, j) => c + gamma * (xr[j] - c));
                    const fe = penalizedObjective(xe);
                    simplex[n] = fe < fr ? xe : xr;
                    values[n] = Math.min(fe, fr);
                }} else if (fr < values[n - 1]) {{
                    simplex[n] = xr;
                    values[n] = fr;
                }} else {{
                    const xc = centroid.map((c, j) => c + rho * (simplex[n][j] - c));
                    const fc = penalizedObjective(xc);
                    if (fc < values[n]) {{
                        simplex[n] = xc;
                        values[n] = fc;
                    }} else {{
                        for (let i = 1; i <= n; i++) {{
                            simplex[i] = simplex[0].map((s0, j) => s0 + sigma * (simplex[i][j] - s0));
                            values[i] = penalizedObjective(simplex[i]);
                        }}
                    }}
                }}

                if (options.onIteration && iter % 20 === 0) {{
                    options.onIteration(iter, values[0]);
                }}
            }}

            return {{ x: simplex[0], value: values[0] }};
        }}

        function calcOptimizationObjective(params, filtersToOptimize, allFilters, measuredSpl, frequencies, freqRange) {{
            let paramIdx = 0;
            const updatedFilters = allFilters.map(f => {{
                if (!filtersToOptimize.includes(f)) return f;
                const updated = {{ ...f }};
                updated.freq = params[paramIdx++];
                updated.q = params[paramIdx++];
                if (filterNeedsGain[f.type]) {{
                    updated.gain = params[paramIdx++];
                }}
                return updated;
            }});

            const filterResponse = calcFilterChainResponse(updatedFilters, frequencies);
            const filteredSpl = measuredSpl.map((spl, i) => spl + filterResponse[i]);

            const inRange = [];
            for (let i = 0; i < frequencies.length; i++) {{
                if (frequencies[i] >= freqRange.min && frequencies[i] <= freqRange.max) {{
                    inRange.push(i);
                }}
            }}

            if (inRange.length === 0) return 1000;

            const splInRange = inRange.map(i => filteredSpl[i]);
            const mean = splInRange.reduce((a, b) => a + b, 0) / splInRange.length;
            const rmse = Math.sqrt(splInRange.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / splInRange.length);

            return rmse;
        }}

        let preOptFilters = null;

        async function runOptimization() {{
            const driver = selectedFilterDriver;
            const filters = driverFilters[driver];
            const toOptimize = filters.filter(f => f.optimize && f.enabled);

            if (toOptimize.length === 0) {{
                alert('No filters marked for optimization. Check the "Opt" checkbox on filters you want to optimize.');
                return;
            }}

            const refAngle = parseInt(document.getElementById('optRefAngle').value);
            const data = allData[driver];
            if (!data.spl[refAngle]) {{
                alert('Reference angle ' + refAngle + '° not available for ' + driver);
                return;
            }}
            const measuredSpl = data.spl[refAngle].map((v, i) => v + driverOffsets[driver]);

            const freqMin = parseFloat(document.getElementById('optFreqMin').value);
            const freqMax = parseFloat(document.getElementById('optFreqMax').value);

            const x0 = [], boundsMin = [], boundsMax = [];
            toOptimize.forEach(f => {{
                x0.push(f.freq);
                boundsMin.push(f.freq / 2);
                boundsMax.push(f.freq * 2);

                x0.push(f.q);
                if (f.type === 'Peaking') {{
                    boundsMin.push(0.2); boundsMax.push(3.0);
                }} else {{
                    boundsMin.push(0.5); boundsMax.push(2.0);
                }}

                if (filterNeedsGain[f.type]) {{
                    x0.push(f.gain);
                    boundsMin.push(-15); boundsMax.push(15);
                }}
            }});

            preOptFilters = JSON.parse(JSON.stringify(filters));

            const statusEl = document.getElementById('optStatus');
            const optimizeBtn = document.getElementById('optimizeBtn');
            statusEl.textContent = 'Optimizing...';
            optimizeBtn.disabled = true;

            await new Promise(r => setTimeout(r, 10));

            const result = nelderMead(
                (params) => calcOptimizationObjective(params, toOptimize, filters, measuredSpl, data.freq, {{ min: freqMin, max: freqMax }}),
                x0,
                {{ min: boundsMin, max: boundsMax }},
                {{ maxIter: 300, onIteration: (i, v) => {{ statusEl.textContent = `Iteration ${{i}}... RMSE: ${{v.toFixed(3)}} dB`; }} }}
            );

            let paramIdx = 0;
            toOptimize.forEach(f => {{
                f.freq = Math.round(result.x[paramIdx++]);
                f.q = Math.round(result.x[paramIdx++] * 100) / 100;
                if (filterNeedsGain[f.type]) {{
                    f.gain = Math.round(result.x[paramIdx++] * 10) / 10;
                }}
            }});

            statusEl.textContent = `Done! RMSE: ${{result.value.toFixed(3)}} dB`;
            optimizeBtn.disabled = false;
            document.getElementById('undoOptBtn').style.display = 'inline-block';

            renderFilterList();
            saveFiltersToLocalStorage();
            updatePlot();
        }}

        function undoOptimization() {{
            if (preOptFilters) {{
                driverFilters[selectedFilterDriver] = preOptFilters;
                preOptFilters = null;
                document.getElementById('undoOptBtn').style.display = 'none';
                document.getElementById('optStatus').textContent = 'Reverted to previous values';
                renderFilterList();
                saveFiltersToLocalStorage();
                updatePlot();
            }}
        }}

        // ============ LOCALSTORAGE ============
        function saveFiltersToLocalStorage() {{
            // Only save user-created filters, not X-O derived ones (they're regenerated from crossovers)
            const filtersToSave = {{}};
            drivers.forEach(d => {{
                filtersToSave[d] = (driverFilters[d] || []).filter(f => f.fromCrossover === undefined);
            }});
            localStorage.setItem(STORAGE_KEY, JSON.stringify(filtersToSave));
        }}

        function loadFiltersFromLocalStorage() {{
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) {{
                try {{
                    driverFilters = JSON.parse(saved);
                    // Ensure all drivers have arrays
                    drivers.forEach(d => {{
                        if (!driverFilters[d]) driverFilters[d] = [];
                    }});
                }} catch (e) {{
                    console.error('Failed to load filters from localStorage:', e);
                }}
            }}
        }}

        function saveUiStateToLocalStorage() {{
            const showPhase = document.getElementById('showPhase')?.checked ?? false;
            const sumMode = document.querySelector('input[name="sumMode"]:checked')?.value || 'off';
            const optMinInput = document.getElementById('optFreqMin');
            const optMaxInput = document.getElementById('optFreqMax');
            const optRefInput = document.getElementById('optRefAngle');

            const offsets = {{}};
            const delays = {{}};
            const phases = {{}};
            drivers.forEach(d => {{
                offsets[d] = Number.isFinite(driverOffsets[d]) ? driverOffsets[d] : 0;
                delays[d] = Number.isFinite(driverDelaysMs[d]) ? driverDelaysMs[d] : 0;
                phases[d] = Number.isFinite(driverPhaseOffsetsDeg[d]) ? driverPhaseOffsetsDeg[d] : 0;
            }});

            const state = {{
                activeDrivers: Array.from(activeDrivers),
                activeAngles: Array.from(activeAngles),
                driverOffsets: offsets,
                driverDelaysMs: delays,
                driverPhaseOffsetsDeg: phases,
                showPhase,
                sumMode,
                selectedFilterDriver,
                optFreqMin: optMinInput ? parseFloat(optMinInput.value) : null,
                optFreqMax: optMaxInput ? parseFloat(optMaxInput.value) : null,
                optRefAngle: optRefInput ? parseFloat(optRefInput.value) : null,
            }};

            localStorage.setItem(UI_STATE_STORAGE_KEY, JSON.stringify(state));
        }}

        function loadUiStateFromLocalStorage() {{
            const saved = localStorage.getItem(UI_STATE_STORAGE_KEY);
            if (!saved) return;
            try {{
                const state = JSON.parse(saved);
                savedUiState = state;

                if (Array.isArray(state.activeDrivers)) {{
                    const filtered = state.activeDrivers.filter(d => drivers.includes(d));
                    if (state.activeDrivers.length === 0) {{
                        activeDrivers = new Set();
                    }} else if (filtered.length) {{
                        activeDrivers = new Set(filtered);
                    }} else if (drivers.length) {{
                        activeDrivers = new Set([drivers[0]]);
                    }}
                }}

                if (Array.isArray(state.activeAngles)) {{
                    const filteredAngles = state.activeAngles
                        .map(a => parseInt(a, 10))
                        .filter(a => allAngles.includes(a));
                    if (state.activeAngles.length === 0) {{
                        activeAngles = new Set();
                    }} else if (filteredAngles.length) {{
                        activeAngles = new Set(filteredAngles);
                    }} else if (allAngles.length) {{
                        activeAngles = new Set([allAngles[0]]);
                    }}
                }}

                if (state.driverOffsets && typeof state.driverOffsets === 'object') {{
                    drivers.forEach(d => {{
                        const v = parseFloat(state.driverOffsets[d]);
                        if (Number.isFinite(v)) driverOffsets[d] = v;
                    }});
                }}
                if (state.driverDelaysMs && typeof state.driverDelaysMs === 'object') {{
                    drivers.forEach(d => {{
                        const v = parseFloat(state.driverDelaysMs[d]);
                        if (Number.isFinite(v)) driverDelaysMs[d] = v;
                    }});
                }}
                if (state.driverPhaseOffsetsDeg && typeof state.driverPhaseOffsetsDeg === 'object') {{
                    drivers.forEach(d => {{
                        const v = parseFloat(state.driverPhaseOffsetsDeg[d]);
                        if (Number.isFinite(v)) driverPhaseOffsetsDeg[d] = v;
                    }});
                }}

                const showPhaseEl = document.getElementById('showPhase');
                if (showPhaseEl && typeof state.showPhase === 'boolean') {{
                    showPhaseEl.checked = state.showPhase;
                }}
                if (state.sumMode) {{
                    const sumRadio = document.querySelector(`input[name="sumMode"][value="${{state.sumMode}}"]`);
                    if (sumRadio) sumRadio.checked = true;
                }}

                if (state.selectedFilterDriver && drivers.includes(state.selectedFilterDriver)) {{
                    selectedFilterDriver = state.selectedFilterDriver;
                }}

                const optMinInput = document.getElementById('optFreqMin');
                if (optMinInput && Number.isFinite(state.optFreqMin)) {{
                    optMinInput.value = state.optFreqMin;
                }}
                const optMaxInput = document.getElementById('optFreqMax');
                if (optMaxInput && Number.isFinite(state.optFreqMax)) {{
                    optMaxInput.value = state.optFreqMax;
                }}
            }} catch (e) {{
                console.error('Failed to load UI state from localStorage:', e);
            }}
        }}

        function renderDriverList() {{
            const driverList = document.getElementById('driverList');
            if (!driverList) return;
            driverList.innerHTML = '';

            drivers.forEach((driver) => {{
                const color = driverColors[driver] || defaultColor;
                const isActive = activeDrivers.has(driver);

                const item = document.createElement('div');
                item.className = 'driver-item' + (isActive ? ' active' : '');
                item.style.setProperty('--driver-color', color);
                item.dataset.driver = driver;

	                item.innerHTML = `
	                    <div class="driver-header">
	                        <input type="checkbox" class="driver-checkbox" ${{isActive ? 'checked' : ''}}>
	                        <span class="driver-line">———</span>
	                        <span class="driver-name">${{driver}}</span>
	                    </div>
		                    <div class="offset-control">
		                        <span>Gain:</span>
		                        <input type="range" class="gain-slider" min="-20" max="20" value="0">
		                        <input type="number" class="gain-input" value="0" min="-20" max="20">
		                        <span>dB</span>
		                    </div>
		                    <div class="delay-control">
		                        <span>Delay:</span>
		                        <input type="number" class="delay-input" value="0" step="1" min="-100000" max="100000">
		                        <span>µs</span>
		                    </div>
                            <div class="invert-control">
                                <label>
                                    <input type="checkbox" class="invert-checkbox">
                                    <span>Invert</span>
                                </label>
                            </div>
	                `;

                // Header click toggles driver
                const header = item.querySelector('.driver-header');
                const checkbox = item.querySelector('.driver-checkbox');

                // Make the checkbox itself behave normally (no double-toggle via header click)
                checkbox.onclick = (e) => {{
                    e.stopPropagation();
                }};
                checkbox.onchange = () => {{
                    toggleDriver(driver, item, checkbox.checked);
                }};

                header.onclick = (e) => {{
                    if (e.target === checkbox) return;
                    checkbox.checked = !checkbox.checked;
                    toggleDriver(driver, item, checkbox.checked);
                }};

                // Offset controls
	                const gainSlider = item.querySelector('.gain-slider');
	                const gainInput = item.querySelector('.gain-input');
	                const delayInput = item.querySelector('.delay-input');
                        const invertInput = item.querySelector('.invert-checkbox');
                        const savedGain = Number.isFinite(driverOffsets[driver]) ? driverOffsets[driver] : 0;
                        const savedDelayUs = Number.isFinite(driverDelaysMs[driver]) ? Math.round(driverDelaysMs[driver] * 1000) : 0;
                        if (gainSlider) gainSlider.value = savedGain;
                        if (gainInput) gainInput.value = savedGain;
                        if (delayInput) delayInput.value = savedDelayUs;
	                gainSlider.oninput = () => {{
	                    gainInput.value = gainSlider.value;
	                }};
	                gainSlider.onchange = () => {{
	                    setOffset(driver, gainSlider.value);
	                }};
	                gainInput.onchange = () => {{
	                    gainSlider.value = gainInput.value;
	                    setOffset(driver, gainInput.value);
	                }};
	                delayInput.onchange = () => {{
	                    setDelayUs(driver, delayInput.value);
	                }};
                        if (invertInput) {{
                            invertInput.checked = (driverPhaseOffsetsDeg[driver] || 0) === 180;
                            invertInput.onchange = () => {{
                                setInvert(driver, invertInput.checked);
                            }};
                        }}

                driverList.appendChild(item);
            }});
        }}

        function renderAngleGrid() {{
            const angleGrid = document.getElementById('angleGrid');
            if (!angleGrid) return;
            angleGrid.innerHTML = '';
            allAngles.forEach(angle => {{
                const btn = document.createElement('button');
                btn.className = 'angle-btn' + (activeAngles.has(angle) ? ' active' : '');
                btn.textContent = angle + '°';
                btn.onclick = () => toggleAngle(angle, btn);
                angleGrid.appendChild(btn);
            }});
        }}

        function refreshFilterDriverSelect() {{
            const select = document.getElementById('filterDriverSelect');
            if (!select) return;
            select.innerHTML = '';
            drivers.forEach(d => {{
                const opt = document.createElement('option');
                opt.value = d;
                opt.textContent = d;
                select.appendChild(opt);
            }});
            if (!drivers.includes(selectedFilterDriver)) {{
                selectedFilterDriver = drivers[0];
            }}
            select.value = selectedFilterDriver;
        }}

        function refreshOptRefAngleSelect() {{
            const refAngleSelect = document.getElementById('optRefAngle');
            if (!refAngleSelect) return;
            const currentValue = parseInt(refAngleSelect.value);
            refAngleSelect.innerHTML = '';
            allAngles.forEach(a => {{
                const opt = document.createElement('option');
                opt.value = a;
                opt.textContent = a + '°';
                refAngleSelect.appendChild(opt);
            }});

            if (Number.isFinite(currentValue) && allAngles.includes(currentValue)) {{
                refAngleSelect.value = currentValue;
            }} else if (Number.isFinite(savedUiState?.optRefAngle) && allAngles.includes(savedUiState.optRefAngle)) {{
                refAngleSelect.value = savedUiState.optRefAngle;
            }} else if (allAngles.includes(0)) {{
                refAngleSelect.value = 0;
            }} else if (allAngles.length) {{
                refAngleSelect.value = allAngles[0];
            }}
        }}

        // Initialize UI
        function initUI() {{
            renderDriverList();
            renderAngleGrid();
        }}

        function toggleDriver(driver, elem, checked) {{
            if (checked) {{
                activeDrivers.add(driver);
                elem.classList.add('active');
            }} else {{
                activeDrivers.delete(driver);
                elem.classList.remove('active');
            }}
            updatePlot();
        }}

        function toggleAngle(angle, btn) {{
            if (activeAngles.has(angle)) {{
                activeAngles.delete(angle);
                btn.classList.remove('active');
            }} else {{
                activeAngles.add(angle);
                btn.classList.add('active');
            }}
            updatePlot();
        }}

        function selectAngles(angles) {{
            activeAngles = new Set(angles);
            document.querySelectorAll('.angle-btn').forEach(btn => {{
                const angle = parseInt(btn.textContent);
                btn.classList.toggle('active', activeAngles.has(angle));
            }});
            updatePlot();
        }}

        function selectAllAngles() {{
            selectAngles(allAngles);
        }}

	        function setOffset(driver, value) {{
	            driverOffsets[driver] = parseFloat(value) || 0;
	            updatePlot();
	        }}

	        function setDelayUs(driver, value) {{
	            const delayUs = parseFloat(value) || 0;
	            driverDelaysMs[driver] = delayUs / 1000;
	            updatePlot();
	        }}

            function setInvert(driver, isInverted) {{
                driverPhaseOffsetsDeg[driver] = isInverted ? 180 : 0;
                updatePlot();
            }}

        function loadExtraDrivers(key) {{
            const dataset = extraDriverDatasets?.[key];
            if (!dataset || !dataset.allData) {{
                alert('Extra drivers not available for this dataset.');
                return;
            }}
            if (dataset.loaded) return;

            const extraDrivers = (dataset.drivers || []).filter(d => !drivers.includes(d));
            if (!extraDrivers.length) {{
                dataset.loaded = true;
                const btn = document.getElementById('loadExtraJuanBtn');
                if (btn) {{
                    btn.textContent = 'Juan baffleless drivers loaded';
                    btn.disabled = true;
                }}
                return;
            }}

            extraDrivers.forEach(d => {{
                drivers.push(d);
                if (!driverFilters[d]) driverFilters[d] = [];
                const savedOffset = savedUiState?.driverOffsets?.[d];
                const savedDelay = savedUiState?.driverDelaysMs?.[d];
                const savedPhase = savedUiState?.driverPhaseOffsetsDeg?.[d];
                driverOffsets[d] = Number.isFinite(savedOffset) ? savedOffset : 0;
                driverDelaysMs[d] = Number.isFinite(savedDelay) ? savedDelay : 0;
                driverPhaseOffsetsDeg[d] = Number.isFinite(savedPhase) ? savedPhase : 0;
            }});

            const savedActiveDrivers = savedUiState?.activeDrivers;
            if (Array.isArray(savedActiveDrivers)) {{
                savedActiveDrivers.forEach(d => {{
                    if (drivers.includes(d)) activeDrivers.add(d);
                }});
            }}

            Object.entries(dataset.allData || {{}}).forEach(([driver, data]) => {{
                allData[driver] = data;
            }});

            (dataset.allAngles || []).forEach(a => {{
                if (!allAngles.includes(a)) allAngles.push(a);
            }});
            allAngles.sort((a, b) => a - b);

            renderDriverList();
            renderAngleGrid();
            refreshFilterDriverSelect();
            refreshOptRefAngleSelect();
            renderFilterList();
            renderCrossoverList();
            syncCrossoverFilters();
            saveFiltersToLocalStorage();
            saveUiStateToLocalStorage();
            updatePlot();

            dataset.loaded = true;
            const btn = document.getElementById('loadExtraJuanBtn');
            if (btn) {{
                btn.textContent = 'Juan baffleless drivers loaded';
                btn.disabled = true;
            }}
        }}

        // ============ FILTER UI FUNCTIONS ============
        function initFilterUI() {{
            refreshFilterDriverSelect();
            refreshOptRefAngleSelect();

            const optMinInput = document.getElementById('optFreqMin');
            if (optMinInput && Number.isFinite(savedUiState?.optFreqMin)) {{
                optMinInput.value = savedUiState.optFreqMin;
            }}
            const optMaxInput = document.getElementById('optFreqMax');
            if (optMaxInput && Number.isFinite(savedUiState?.optFreqMax)) {{
                optMaxInput.value = savedUiState.optFreqMax;
            }}

            renderFilterList();
        }}

        function selectFilterDriver(driver) {{
            selectedFilterDriver = driver;
            renderFilterList();
            saveUiStateToLocalStorage();
        }}

	        function addFilter() {{
	            const newFilter = {{
	                type: 'Peaking',
	                freq: 1000,
	                q: 1.0,
	                gain: 0,
	                enabled: true,
	                optimize: false
	            }};
	            driverFilters[selectedFilterDriver].push(newFilter);
	            saveFiltersToLocalStorage();
	            renderFilterList();
	            updatePlot();
	        }}

            function clearAllUserFilters() {{
                const userFilterCount = drivers.reduce((sum, d) => {{
                    const filters = driverFilters[d] || [];
                    return sum + filters.filter(f => f.fromCrossover === undefined).length;
                }}, 0);

                if (userFilterCount === 0) {{
                    alert('No user filters to clear.');
                    return;
                }}

                const proceed = confirm(
                    'Clear ' + userFilterCount + ' user filter(s) across all drivers?\\n\\n' +
                    'Crossovers will be kept.'
                );
                if (!proceed) return;

                drivers.forEach(d => {{
                    // Keep crossover-derived filters only; they will be regenerated from visualCrossovers.
                    driverFilters[d] = (driverFilters[d] || []).filter(f => f.fromCrossover !== undefined);
                }});

                saveFiltersToLocalStorage();
                syncCrossoverFilters();
                renderFilterList();
                updatePlot();
            }}

	        function deleteFilter(index) {{
	            driverFilters[selectedFilterDriver].splice(index, 1);
	            saveFiltersToLocalStorage();
	            renderFilterList();
            updatePlot();
        }}

        function updateFilter(index, field, value) {{
            const filter = driverFilters[selectedFilterDriver][index];
            if (field === 'type') {{
                filter.type = value;
                // Reset gain if switching to a type that doesn't need it
                if (!filterNeedsGain[value]) {{
                    filter.gain = 0;
                }}
            }} else if (field === 'enabled' || field === 'optimize') {{
                filter[field] = value;
            }} else {{
                filter[field] = parseFloat(value) || 0;
            }}
            saveFiltersToLocalStorage();
            renderFilterList();
            updatePlot();
        }}

        function renderFilterList() {{
            const list = document.getElementById('filterList');
            const filters = driverFilters[selectedFilterDriver] || [];

            if (filters.length === 0) {{
                list.innerHTML = '<div class="no-filters">No filters defined</div>';
                return;
            }}

            list.innerHTML = '';
            filters.forEach((filter, idx) => {{
                const card = document.createElement('div');
                const isFromCrossover = filter.fromCrossover !== undefined;
                card.className = 'filter-card' + (filter.enabled ? '' : ' disabled') + (isFromCrossover ? ' from-crossover' : '');

                const needsGain = filterNeedsGain[filter.type];

                if (isFromCrossover) {{
                    // Read-only display for crossover-derived filters
                    const stageLabel = filter.xoTotalStages > 1 ? ` (${{filter.xoStage}}/${{filter.xoTotalStages}})` : '';
                    card.innerHTML = `
                        <div class="filter-card-header">
                            <span class="xo-badge" title="From X-O #${{filter.fromCrossover + 1}}">X-O</span>
                            <span class="xo-filter-type">${{filter.type}}${{stageLabel}}</span>
                            <span class="xo-filter-status">${{filter.enabled ? '●' : '○'}}</span>
                        </div>
                        <div class="filter-param readonly">
                            <label>Freq:</label>
                            <span class="readonly-value">${{Math.round(filter.freq)}} Hz</span>
                        </div>
                        <div class="filter-param readonly">
                            <label>Q:</label>
                            <span class="readonly-value">${{filter.q.toFixed(3)}}</span>
                        </div>
                    `;
                }} else {{
                    // Editable filter
                    card.innerHTML = `
                        <div class="filter-card-header">
                            <input type="checkbox" ${{filter.enabled ? 'checked' : ''}}
                                   onchange="updateFilter(${{idx}}, 'enabled', this.checked)" title="Enable filter">
                            <label class="opt-checkbox" title="Include in auto-optimization">
                                <input type="checkbox" ${{filter.optimize ? 'checked' : ''}}
                                       onchange="updateFilter(${{idx}}, 'optimize', this.checked)">Opt
                            </label>
                            <select onchange="updateFilter(${{idx}}, 'type', this.value)">
                                ${{filterTypes.map(t => `<option value="${{t}}" ${{t === filter.type ? 'selected' : ''}}>${{t}}</option>`).join('')}}
                            </select>
                            <button class="delete-btn" onclick="deleteFilter(${{idx}})">×</button>
                        </div>
                        <div class="filter-param">
                            <label>Freq:</label>
                            <input type="range" min="1.3" max="4.3" step="0.01" value="${{Math.log10(filter.freq)}}"
                                   oninput="this.nextElementSibling.value = Math.round(Math.pow(10, this.value))"
                                   onchange="updateFilter(${{idx}}, 'freq', Math.pow(10, this.value))">
                            <input type="number" value="${{Math.round(filter.freq)}}" min="20" max="20000"
                                   onchange="updateFilter(${{idx}}, 'freq', this.value); this.previousElementSibling.value = Math.log10(this.value)">
                            <span>Hz</span>
                        </div>
                        <div class="filter-param">
                            <label>Q:</label>
                            <input type="range" min="0.1" max="10" step="0.01" value="${{filter.q}}"
                                   oninput="this.nextElementSibling.value = parseFloat(this.value).toFixed(2)"
                                   onchange="updateFilter(${{idx}}, 'q', this.value)">
                            <input type="number" value="${{filter.q.toFixed(2)}}" min="0.1" max="10" step="0.01"
                                   onchange="updateFilter(${{idx}}, 'q', this.value); this.previousElementSibling.value = this.value">
                            <span></span>
                        </div>
                        ${{needsGain ? `
                        <div class="filter-param">
                            <label>Gain:</label>
                            <input type="range" min="-20" max="20" step="0.1" value="${{filter.gain}}"
                                   oninput="this.nextElementSibling.value = parseFloat(this.value).toFixed(1)"
                                   onchange="updateFilter(${{idx}}, 'gain', this.value)">
                            <input type="number" value="${{filter.gain.toFixed(1)}}" min="-20" max="20" step="0.1"
                                   onchange="updateFilter(${{idx}}, 'gain', this.value); this.previousElementSibling.value = this.value">
                            <span>dB</span>
                        </div>
                        ` : ''}}
                    `;
                }}
                list.appendChild(card);
            }});
        }}

        // ============ YAML IMPORT/EXPORT ============
        function loadYamlFile(event) {{
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {{
                try {{
                    const yaml = jsyaml.load(e.target.result);
                    const filters = [];
                    const unsupported = [];

                    for (const [name, def] of Object.entries(yaml.filters || {{}})) {{
                        if (def.type === 'Biquad' && def.parameters) {{
                            const filterType = def.parameters.type || 'Peaking';
                            // Check if filter type is supported
                            if (!filterTypes.includes(filterType)) {{
                                unsupported.push(`${{name}}: ${{filterType}}`);
                                continue;
                            }}
                            filters.push({{
                                type: filterType,
                                freq: def.parameters.freq || 1000,
                                q: def.parameters.q || 1.0,
                                gain: def.parameters.gain || 0,
                                enabled: true,
                                optimize: false
                            }});
                        }}
                    }}

                    if (filters.length > 0) {{
                        driverFilters[selectedFilterDriver] = filters;
                        syncCrossoverFilters();  // Re-add any X-O derived filters after import
                        saveFiltersToLocalStorage();
                        renderFilterList();
                        updatePlot();
                        let msg = `Loaded ${{filters.length}} filter(s) for ${{selectedFilterDriver}}`;
                        if (unsupported.length > 0) {{
                            msg += `\\n\\n⚠️ Skipped ${{unsupported.length}} unsupported filter(s):\\n${{unsupported.join('\\n')}}`;
                        }}
                        alert(msg);
                    }} else {{
                        let msg = 'No valid biquad filters found in YAML';
                        if (unsupported.length > 0) {{
                            msg += `\\n\\n⚠️ Found ${{unsupported.length}} unsupported filter(s):\\n${{unsupported.join('\\n')}}`;
                        }}
                        alert(msg);
                    }}
                }} catch (err) {{
                    alert('Error parsing YAML: ' + err.message);
                }}
            }};
            reader.readAsText(file);
            event.target.value = '';  // Reset input
        }}

        function saveFiltersYaml() {{
            // Export only user-created filters (exclude X-O derived ones)
            const filters = (driverFilters[selectedFilterDriver] || []).filter(f => f.fromCrossover === undefined);
            if (filters.length === 0) {{
                alert('No user filters to save for ' + selectedFilterDriver);
                return;
            }}

            const yamlObj = {{ filters: {{}}, pipeline: [] }};

            filters.forEach((f, idx) => {{
                const name = `filter_${{idx + 1}}`;
                const params = {{
                    type: f.type,
                    freq: f.freq,
                    q: f.q
                }};
                if (filterNeedsGain[f.type]) {{
                    params.gain = f.gain;
                }}
                yamlObj.filters[name] = {{
                    type: 'Biquad',
                    parameters: params
                }};
            }});

            yamlObj.pipeline = [{{
                type: 'Filter',
                channel: 0,
                names: Object.keys(yamlObj.filters)
            }}];

            const yamlStr = jsyaml.dump(yamlObj, {{ indent: 2 }});
            const blob = new Blob([yamlStr], {{ type: 'text/yaml' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${{selectedFilterDriver}}_filters.yml`;
            a.click();
            URL.revokeObjectURL(url);
        }}

	        // ============ VISUAL CROSSOVER UI FUNCTIONS ============
	        function addCrossover() {{
	            const lp = drivers.length > 0 ? drivers[0] : null;
	            const hp = drivers.length > 1 ? drivers[1] : null;
	            const newXO = {{
	                enabled: true,
	                freq: 1200,
	                type: 'LR4',
	                lpDriver: lp,
	                hpDriver: hp,
	                lpDrivers: lp ? [lp] : [],
	                hpDrivers: hp ? [hp] : [],
	                applyFilters: true  // Apply filters by default (create IIR entries)
	            }};
	            visualCrossovers.push(newXO);
            saveCrossoversToLocalStorage();
            syncCrossoverFilters();
            renderCrossoverList();
            renderFilterList();
            updatePlot();
        }}

        function deleteCrossover(index) {{
            visualCrossovers.splice(index, 1);
            saveCrossoversToLocalStorage();
            syncCrossoverFilters();
            renderCrossoverList();
            renderFilterList();
            updatePlot();
        }}

	        function updateCrossover(index, field, value) {{
	            const xo = visualCrossovers[index];
	            if (field === 'enabled') {{
	                xo.enabled = value;
            }} else if (field === 'freq') {{
                xo.freq = parseFloat(value) || 1000;
	            }} else if (field === 'type') {{
	                xo.type = value;
	                // If switching away from a DSP-derived custom crossover, drop the explicit stages
	                // so the UI selection uses the synthetic generator.
	                if (value !== 'Custom') {{
	                    xo.lpStages = null;
	                    xo.hpStages = null;
	                    xo.fadeOctavesLP = null;
	                    xo.fadeOctavesHP = null;
	                }}
		            }} else if (field === 'lpDriver') {{
		                xo.lpDriver = value === '' ? null : value;
		                xo.lpDrivers = xo.lpDriver ? [xo.lpDriver] : [];
	            }} else if (field === 'hpDriver') {{
	                xo.hpDriver = value === '' ? null : value;
	                xo.hpDrivers = xo.hpDriver ? [xo.hpDriver] : [];
	            }} else if (field === 'applyFilters') {{
	                xo.applyFilters = value;
	            }}
            saveCrossoversToLocalStorage();
            syncCrossoverFilters();
            renderCrossoverList();
            renderFilterList();
            updatePlot();
        }}

        function renderCrossoverList() {{
            const list = document.getElementById('crossoverList');

            if (visualCrossovers.length === 0) {{
                list.innerHTML = '<div class="no-crossovers">No crossovers defined</div>';
                return;
            }}

            list.innerHTML = '';
            visualCrossovers.forEach((xo, idx) => {{
                const card = document.createElement('div');
                card.className = 'crossover-card' + (xo.enabled ? '' : ' disabled');

	                let typeOptions = Object.entries(crossoverTypes).map(([key, val]) =>
	                    `<option value="${{key}}" ${{key === xo.type ? 'selected' : ''}}>${{key}}</option>`
	                ).join('');
	                if (!crossoverTypes[xo.type]) {{
	                    typeOptions = `<option value="${{xo.type}}" selected>${{xo.type}}</option>` + typeOptions;
	                }}

                const driverOptionsLP = ['<option value="">(None)</option>'].concat(
                    drivers.map(d => `<option value="${{d}}" ${{d === xo.lpDriver ? 'selected' : ''}}>${{d}}</option>`)
                ).join('');

                const driverOptionsHP = ['<option value="">(None)</option>'].concat(
                    drivers.map(d => `<option value="${{d}}" ${{d === xo.hpDriver ? 'selected' : ''}}>${{d}}</option>`)
                ).join('');

                card.innerHTML = `
                    <div class="crossover-card-header">
                        <input type="checkbox" ${{xo.enabled ? 'checked' : ''}}
                               onchange="updateCrossover(${{idx}}, 'enabled', this.checked)" title="Enable crossover">
                        <select onchange="updateCrossover(${{idx}}, 'type', this.value)" title="${{crossoverTypes[xo.type]?.name || ''}}">
                            ${{typeOptions}}
                        </select>
                        <label class="apply-checkbox" title="Apply LP/HP filters to response (not just visual clipping)">
                            <input type="checkbox" ${{xo.applyFilters ? 'checked' : ''}}
                                   onchange="updateCrossover(${{idx}}, 'applyFilters', this.checked)">Filt
                        </label>
                        <button class="delete-btn" onclick="deleteCrossover(${{idx}})">×</button>
                    </div>
                    <div class="crossover-param">
                        <label>Freq:</label>
                        <input type="range" min="1.3" max="4.3" step="0.01" value="${{Math.log10(xo.freq)}}"
                               oninput="this.nextElementSibling.value = Math.round(Math.pow(10, this.value))"
                               onchange="updateCrossover(${{idx}}, 'freq', Math.pow(10, this.value))">
                        <input type="number" value="${{Math.round(xo.freq)}}" min="20" max="20000"
                               onchange="updateCrossover(${{idx}}, 'freq', this.value); this.previousElementSibling.value = Math.log10(this.value)">
                        <span>Hz</span>
                    </div>
                    <div class="crossover-drivers">
                        <div class="crossover-driver-select">
                            <label>LP →</label>
                            <select onchange="updateCrossover(${{idx}}, 'lpDriver', this.value)">
                                ${{driverOptionsLP}}
                            </select>
                        </div>
                        <div class="crossover-driver-select">
                            <label>HP →</label>
                            <select onchange="updateCrossover(${{idx}}, 'hpDriver', this.value)">
                                ${{driverOptionsHP}}
                            </select>
                        </div>
                    </div>
                `;
                list.appendChild(card);
            }});
        }}

        function saveCrossoversToLocalStorage() {{
            localStorage.setItem(CROSSOVER_STORAGE_KEY, JSON.stringify(visualCrossovers));
        }}

        function loadCrossoversFromLocalStorage() {{
            const saved = localStorage.getItem(CROSSOVER_STORAGE_KEY);
            if (saved) {{
                try {{
                    visualCrossovers = JSON.parse(saved);
                }} catch (e) {{
                    console.error('Failed to load crossovers from localStorage:', e);
                    visualCrossovers = [];
                }}
            }}
        }}

	        function updatePlot() {{
	            const traces = [];
                const driverSumInfo = {{}};

            // Angle-based dash patterns: progressive density (0° solid, higher angles more dashed)
            // Maps angle to dash pattern - solid for on-axis, increasingly broken for off-axis
            function getAngleDash(angle) {{
                if (angle === 0) return 'solid';
                if (angle <= 10) return '12 4';      // long dash
                if (angle <= 20) return '8 4';       // medium dash
                if (angle <= 30) return '5 4';       // short dash
                if (angle <= 45) return '8 4 2 4';   // dash-dot
                if (angle <= 60) return '5 4 2 4';   // short dash-dot
                return '2 4';                         // dotted (90° and beyond)
            }}

            // Phase dash patterns: 2X longer than SPL patterns
            function getPhaseDash(angle) {{
                if (angle === 0) return 'solid';
                if (angle <= 10) return '24 8';      // 2X of '12 4'
                if (angle <= 20) return '16 8';      // 2X of '8 4'
                if (angle <= 30) return '10 8';      // 2X of '5 4'
                if (angle <= 45) return '16 8 4 8';  // 2X of '8 4 2 4'
                if (angle <= 60) return '10 8 4 8';  // 2X of '5 4 2 4'
                return '4 8';                         // 2X of '2 4'
            }}

		            const showPhaseTraces = document.getElementById('showPhase')?.checked ?? false;
                    const sumMode = document.querySelector('input[name="sumMode"]:checked')?.value || 'off';
                const showSumTraces = sumMode !== 'off';
                    const showIndividualDrivers = sumMode !== 'only';

                    saveUiStateToLocalStorage();

                    // Keep SUM first in the legend even when SUM traces are drawn last (on top)
                    if (showSumTraces && activeDrivers.size > 0 && activeAngles.size > 0) {{
                        traces.push({{
                            x: [],
                            y: [],
                            name: 'Σ SUM',
                            mode: 'lines',
                            line: {{ color: '#222', width: 3.5 }},
                            showlegend: true,
                            legendgroup: 'SUM',
                            hoverinfo: 'skip',
                            legendrank: 0
                        }});
                    }}

                function parseRgbColor(color) {{
                    if (!color) return {{ r: 136, g: 136, b: 136 }};
                    if (color.startsWith('#')) {{
                        if (color.length === 4) {{
                            const r = parseInt(color[1] + color[1], 16);
                            const g = parseInt(color[2] + color[2], 16);
                            const b = parseInt(color[3] + color[3], 16);
                            return {{ r, g, b }};
                        }}
                        const r = parseInt(color.slice(1, 3), 16);
                        const g = parseInt(color.slice(3, 5), 16);
                        const b = parseInt(color.slice(5, 7), 16);
                        return {{ r, g, b }};
                    }}
                    const match = color.match(/rgb\\((\\d+),\\s*(\\d+),\\s*(\\d+)\\)/);
                    if (match) {{
                        return {{ r: parseInt(match[1]), g: parseInt(match[2]), b: parseInt(match[3]) }};
                    }}
                    return {{ r: 136, g: 136, b: 136 }};
                }}

                function blendRgb(colorA, colorB, ratioToB) {{
                    const a = parseRgbColor(colorA);
                    const b = parseRgbColor(colorB);
                    const t = Math.min(1, Math.max(0, ratioToB));
                    const r = Math.round(a.r + (b.r - a.r) * t);
                    const g = Math.round(a.g + (b.g - a.g) * t);
                    const bch = Math.round(a.b + (b.b - a.b) * t);
                    return `rgb(${{r}},${{g}},${{bch}})`;
                }}

                const SUM_MIN_DB = -100;

                function buildUnionFrequencyGrid(drivers) {{
                    const allFreq = [];
                    drivers.forEach(d => {{
                        const freq = driverSumInfo[d]?.data?.freq;
                        if (Array.isArray(freq)) {{
                            freq.forEach(f => {{
                                if (Number.isFinite(f)) allFreq.push(f);
                            }});
                        }}
                    }});
                    if (!allFreq.length) return [];
                    allFreq.sort((a, b) => a - b);
                    const merged = [];
                    const relTol = 1e-6;
                    const absTol = 1e-3;
                    for (const f of allFreq) {{
                        if (!merged.length) {{
                            merged.push(f);
                            continue;
                        }}
                        const last = merged[merged.length - 1];
                        const tol = Math.max(absTol, Math.abs(last) * relTol);
                        if (Math.abs(f - last) > tol) merged.push(f);
                    }}
                    return merged;
                }}

                function interpolateComplexToGrid(freqSrc, reSrc, imSrc, freqTarget) {{
                    const n = freqTarget.length;
                    const outRe = new Array(n).fill(0);
                    const outIm = new Array(n).fill(0);
                    if (!freqSrc || freqSrc.length < 2) {{
                        return {{ re: outRe, im: outIm }};
                    }}

                    let j = 0;
                    const srcN = freqSrc.length;
                    const minF = freqSrc[0];
                    const maxF = freqSrc[srcN - 1];

                    for (let i = 0; i < n; i++) {{
                        const f = freqTarget[i];
                        if (!Number.isFinite(f) || f < minF || f > maxF) continue;
                        while (j < srcN - 2 && freqSrc[j + 1] < f) j++;
                        const f0 = freqSrc[j];
                        const f1 = freqSrc[j + 1];
                        if (f1 === f0) {{
                            outRe[i] = reSrc[j];
                            outIm[i] = imSrc[j];
                            continue;
                        }}
                        const t = (f - f0) / (f1 - f0);
                        outRe[i] = reSrc[j] + (reSrc[j + 1] - reSrc[j]) * t;
                        outIm[i] = imSrc[j] + (imSrc[j + 1] - imSrc[j]) * t;
                    }}
                    return {{ re: outRe, im: outIm }};
                }}

	            activeDrivers.forEach(driver => {{
	                const color = driverColors[driver] || defaultColor;
	                const offset = driverOffsets[driver];
	                const data = allData[driver];
	                const filters = driverFilters[driver] || [];
                    const delayMs = driverDelaysMs[driver] || 0;
                    const invertDeg = driverPhaseOffsetsDeg[driver] || 0;

                // Check what's active - X-O filters are now part of driverFilters
                const hasActiveFilters = filters.some(f => f.enabled);
                const hasVisualClipping = hasVisualClippingForDriver(driver);  // Any X-O enabled (for freq clipping/fading)

                // Calculate combined filter response (includes both EQ and X-O filters in driverFilters)
                let filterResponse = null;
                let filterPhaseResponse = null;
	                if (hasActiveFilters) {{
	                    filterResponse = calcFilterChainResponse(filters, data.freq);
	                    filterPhaseResponse = calcFilterChainPhaseResponse(filters, data.freq);
	                }}

                    // Cache per-driver info for SUM calculation
	                    driverSumInfo[driver] = {{
	                        color,
	                        offset,
	                        data,
                        hasActiveFilters,
                        filterResponse,
                        filterPhaseResponse,
	                        delayMs,
	                        invertDeg
	                    }};

                    if (!showIndividualDrivers) return;

	                // Get fade colors and clip mask for crossover visual effect
	                let fadeColors = null;
	                let clipMask = null;
	                if (hasVisualClipping) {{
                    const fadeInfo = getDriverFadeColors(driver, data.freq, color);
                    fadeColors = fadeInfo.colors;
                    clipMask = fadeInfo.clipMask;
                }}

                activeAngles.forEach(angle => {{
                    if (!data.spl[angle]) return;

                    const splOriginal = data.spl[angle].map(v => v + offset);
                    const lineWidth = angle === 0 ? 2.5 : 1.5;
                    const dashPattern = getAngleDash(angle);
                    const traceName = `${{driver}} @ ${{angle}}°` + (offset !== 0 ? ` (${{offset > 0 ? '+' : ''}}${{offset}}dB)` : '');

                    if (hasActiveFilters) {{
                        // Filters active - show ONLY filtered trace (NO original curve)
                        const splFiltered = splOriginal.map((v, i) => v + filterResponse[i]);

                        if (hasVisualClipping) {{
                            // Use segmented lines for color fading effect
                            const segments = createFadingSegments(
                                data.freq, splFiltered, fadeColors, clipMask,
                                lineWidth, dashPattern, traceName, true
                            );
                            traces.push(...segments);
                        }} else {{
                            traces.push({{
                                x: data.freq,
                                y: splFiltered,
                                name: traceName,
                                mode: 'lines',
                                line: {{ color: color, width: lineWidth, dash: dashPattern }},
                                connectgaps: false
                            }});
                        }}
                    }} else if (hasVisualClipping) {{
                        // Only visual clipping/fading, no filters - show faded original with segmented lines
                        const segments = createFadingSegments(
                            data.freq, splOriginal, fadeColors, clipMask,
                            lineWidth, dashPattern, traceName, true
                        );
                        traces.push(...segments);
                    }} else {{
                        // No filters or crossovers - show original trace
                        traces.push({{
                            x: data.freq,
                            y: splOriginal,
                            name: traceName,
                            mode: 'lines',
                            line: {{
                                color: color,
                                width: lineWidth,
                                dash: dashPattern
                            }}
                        }});
                    }}

	                    // Add phase trace if enabled and data available
		                    if (showPhaseTraces && data.phase && data.phase[angle]) {{
		                        const phaseOriginal = data.phase[angle];
		                        const phaseDash = getPhaseDash(angle);
		                        const phaseLineWidth = lineWidth * 0.8;
		                        const phaseName = `${{driver}} @ ${{angle}}° (φ)`;
		                        const needsPhaseAdjust = (hasActiveFilters && filterPhaseResponse) || invertDeg !== 0 || delayMs !== 0;

	                        // Apply filter phase response + invert/delay if active
		                        const phaseData = needsPhaseAdjust
		                            ? phaseOriginal.map((p, i) => {{
		                                let phase = p;
		                                if (hasActiveFilters && filterPhaseResponse) phase += filterPhaseResponse[i];
		                                if (invertDeg) phase += invertDeg;
		                                if (delayMs) phase -= 360 * data.freq[i] * (delayMs / 1000);
		                                return wrapPhase(phase);
		                            }})
		                            : phaseOriginal;

	                        if (hasVisualClipping) {{
	                            // Apply same crossover fading to phase
	                            const phaseSegments = createFadingSegments(
                                data.freq, phaseData, fadeColors, clipMask,
                                phaseLineWidth, phaseDash, phaseName, false
                            );
                            // Assign to xaxis2/yaxis2 (bottom subplot)
                            phaseSegments.forEach(seg => {{ seg.xaxis = 'x2'; seg.yaxis = 'y2'; }});
                            traces.push(...phaseSegments);
                        }} else {{
                            traces.push({{
                                x: data.freq,
                                y: phaseData,
                                name: phaseName,
                                mode: 'lines',
                                line: {{ color: color, width: phaseLineWidth, dash: phaseDash }},
                                xaxis: 'x2',
                                yaxis: 'y2',
                                showlegend: false,
                                hovertemplate: '%{{y:.1f}}°<extra></extra>'
                            }});
                        }}
                    }}
                }});
	            }});

                // ============ SUM TRACE (selected drivers) ============
                if (showSumTraces && activeDrivers.size > 0 && activeAngles.size > 0) {{
                    const selectedDrivers = Array.from(activeDrivers);
                    const xFreq = buildUnionFrequencyGrid(selectedDrivers);

                    if (xFreq.length) {{
                        const n = xFreq.length;
                        const clipAll = xFreq.map(() => true);

                        function contributionColorAtIndex(powByDriver, idx) {{
                            let top1Driver = null, top2Driver = null;
                            let top1 = 0, top2 = 0;

                            selectedDrivers.forEach(d => {{
                                const p = powByDriver[d]?.[idx] || 0;
                                if (p > top1) {{
                                    top2 = top1; top2Driver = top1Driver;
                                    top1 = p; top1Driver = d;
                                }} else if (p > top2) {{
                                    top2 = p; top2Driver = d;
                                }}
                            }});

                            const c1 = (top1Driver && driverSumInfo[top1Driver]?.color) ? driverSumInfo[top1Driver].color : defaultColor;
                            if (!top2Driver || top2 <= 0) return c1;

                            const c2 = driverSumInfo[top2Driver]?.color || defaultColor;
                            const denom = top1 + top2;
                            if (denom <= 0) return defaultColor;

                            const w1 = top1 / denom;
                            const w2 = top2 / denom;
                            const gamma = 2.0;  // emphasize dominant driver
                            const w1g = Math.pow(w1, gamma);
                            const w2g = Math.pow(w2, gamma);
                            const ratioTo2 = w2g / (w1g + w2g);
                            return blendRgb(c1, c2, ratioTo2);
                        }}

                        activeAngles.forEach(angle => {{
                            // Initialize complex sum and contribution tracking
                            const sumRe = new Array(n).fill(0);
                            const sumIm = new Array(n).fill(0);
                            const powByDriver = {{}};
                            selectedDrivers.forEach(d => {{ powByDriver[d] = new Array(n).fill(0); }});

                            // Accumulate each driver as complex pressure
                            selectedDrivers.forEach(driver => {{
                                const info = driverSumInfo[driver];
                                if (!info || !info.data) return;
                                const data = info.data;
                                const freqSrc = data.freq;
                                if (!freqSrc || freqSrc.length < 2) return;

                                const splArr = data.spl?.[angle];
                                if (!splArr) return;

                                const phaseArr = data.phase?.[angle];
                                const magAdj = (info.hasActiveFilters && info.filterResponse) ? info.filterResponse : null;
                                const phaseAdj = (info.hasActiveFilters && info.filterPhaseResponse) ? info.filterPhaseResponse : null;

                                const reSrc = new Array(freqSrc.length);
                                const imSrc = new Array(freqSrc.length);

                                for (let i = 0; i < freqSrc.length; i++) {{
                                    let splDb = splArr[i] + info.offset;
                                    if (magAdj) splDb += magAdj[i];
                                    if (!Number.isFinite(splDb) || splDb < SUM_MIN_DB) {{
                                        reSrc[i] = 0;
                                        imSrc[i] = 0;
                                        continue;
                                    }}

                                    const amp = Math.pow(10, splDb / 20);

                                    let phaseDeg = (phaseArr ? phaseArr[i] : 0);
                                    if (phaseAdj) phaseDeg += phaseAdj[i];
                                    if (info.invertDeg) phaseDeg += info.invertDeg;
                                    if (info.delayMs) phaseDeg -= 360 * freqSrc[i] * (info.delayMs / 1000);
                                    if (!Number.isFinite(phaseDeg)) {{
                                        reSrc[i] = 0;
                                        imSrc[i] = 0;
                                        continue;
                                    }}

                                    const rad = phaseDeg * Math.PI / 180;
                                    reSrc[i] = amp * Math.cos(rad);
                                    imSrc[i] = amp * Math.sin(rad);
                                }}

                                const interp = interpolateComplexToGrid(freqSrc, reSrc, imSrc, xFreq);
                                for (let i = 0; i < n; i++) {{
                                    const re = interp.re[i];
                                    const im = interp.im[i];
                                    sumRe[i] += re;
                                    sumIm[i] += im;
                                    powByDriver[driver][i] = (re * re) + (im * im);
                                }}
                            }});

                            // Build SPL/phase arrays and per-point colors
                            const sumSpl = new Array(n);
                            const sumPhase = new Array(n);
                            const sumColors = new Array(n);

                            for (let i = 0; i < n; i++) {{
                                const mag = Math.hypot(sumRe[i], sumIm[i]);
                                if (!Number.isFinite(mag) || mag <= 0) {{
                                    sumSpl[i] = null;
                                    sumPhase[i] = null;
                                    sumColors[i] = defaultColor;
                                    continue;
                                }}
                                const sumDb = 20 * Math.log10(mag);
                                if (sumDb < SUM_MIN_DB) {{
                                    sumSpl[i] = null;
                                    sumPhase[i] = null;
                                    sumColors[i] = defaultColor;
                                    continue;
                                }}
                                sumSpl[i] = sumDb;
                                sumPhase[i] = wrapPhase(Math.atan2(sumIm[i], sumRe[i]) * 180 / Math.PI);
                                sumColors[i] = contributionColorAtIndex(powByDriver, i);
                            }}

                            const sumLineWidth = angle === 0 ? 3.5 : 2.2;
                            const sumDash = getAngleDash(angle);
	                            const sumName = `Σ SUM @ ${{angle}}°`;
	                            const sumSegments = createFadingSegments(
	                                xFreq, sumSpl, sumColors, clipAll,
	                                sumLineWidth, sumDash, sumName, false
	                            );
	                            // Use a consistent legend group so SUM stays together
	                            sumSegments.forEach(seg => {{ seg.legendgroup = 'SUM'; }});
	                            traces.push(...sumSegments);

                            // Phase for SUM (optional)
                            if (showPhaseTraces) {{
                                const sumPhaseDash = getPhaseDash(angle);
	                                const sumPhaseName = `Σ SUM @ ${{angle}}° (φ)`;
	                                const phaseSegments = createFadingSegments(
	                                    xFreq, sumPhase, sumColors, clipAll,
	                                    sumLineWidth * 0.8, sumPhaseDash, sumPhaseName, false
	                                );
                                phaseSegments.forEach(seg => {{
                                    seg.xaxis = 'x2';
                                    seg.yaxis = 'y2';
                                    seg.showlegend = false;
                                    seg.legendgroup = 'SUM';
	                                    seg.hovertemplate = '%{{y:.1f}}°<extra></extra>';
                                }});
                                traces.push(...phaseSegments);
                            }}
                        }});
                    }}
                }}

	            // Add grid lines
	            const shapes = [];
            // Vertical frequency grid
            [100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000].forEach(f => {{
                shapes.push({{
                    type: 'line', x0: f, x1: f, y0: 0, y1: 1, yref: 'paper',
                    line: {{ color: f === 1000 || f === 10000 ? '#888' : '#ddd', width: 1, dash: 'dot' }}
                }});
            }});

            // Get Y range from SPL data only (exclude phase traces on yaxis2)
            let yMin = Infinity, yMax = -Infinity;
            traces.forEach(t => {{
                if (!t.y) return;
                if (t.yaxis === 'y2') return;  // Skip phase traces
                t.y.forEach(v => {{
                    if (v === null || v === undefined) return;  // Skip clipped/empty points
                    if (!Number.isFinite(v)) return;
                    if (v < yMin) yMin = v;
                    if (v > yMax) yMax = v;
                }});
            }});
            if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMin >= yMax) {{
                // No traces selected (or invalid data); use a sane default range
                yMin = 40;
                yMax = 100;
            }} else {{
                yMin = Math.floor(yMin / 5) * 5 - 5;
                yMax = Math.ceil(yMax / 5) * 5 + 5;
                if (yMin === yMax) {{
                    yMin -= 5;
                    yMax += 5;
                }}
            }}

            // Horizontal dB grid
            for (let db = yMin; db <= yMax; db += 5) {{
                shapes.push({{
                    type: 'line', x0: 0, x1: 1, xref: 'paper', y0: db, y1: db,
                    line: {{ color: db % 10 === 0 ? '#888' : '#ddd', width: db % 10 === 0 ? 1 : 0.5, dash: db % 10 === 0 ? 'solid' : 'dot' }}
                }});
            }}

            const showPhase = document.getElementById('showPhase')?.checked ?? false;

            // SPL always in top portion, phase in bottom when enabled
            const splDomain = showPhase ? [0.38, 1] : [0, 1];
            const phaseDomain = [0, 0.32];

            const layout = {{
                title: '<b>Frequency Response Explorer</b>',
                xaxis: {{
                    title: showPhase ? '' : 'Frequency (Hz)',
                    type: 'log',
                    range: [Math.log10(100), Math.log10(20000)],
                    gridcolor: 'transparent',
                    domain: [0, 1],
                    anchor: 'y'
                }},
                xaxis2: {{
                    title: 'Frequency (Hz)',
                    type: 'log',
                    range: [Math.log10(100), Math.log10(20000)],
                    gridcolor: 'transparent',
                    domain: [0, 1],
                    anchor: 'y2',
                    visible: showPhase
                }},
                yaxis: {{
                    title: 'SPL (dB)',
                    range: [yMin, yMax],
                    gridcolor: 'transparent',
                    domain: splDomain
                }},
                yaxis2: {{
                    title: 'Phase (°)',
                    range: [-180, 180],
                    dtick: 90,
                    showgrid: true,
                    gridcolor: '#f0f0f0',
                    zeroline: true,
                    zerolinecolor: '#ccc',
                    zerolinewidth: 1,
                    domain: phaseDomain,
                    anchor: 'x2',
                    visible: showPhase
                }},
                shapes: shapes,
                legend: {{
                    x: 1.02,
                    y: 1,
                    xanchor: 'left',
                    font: {{ size: 10 }}
                }},
                margin: {{ l: 60, r: 200, t: 50, b: 50 }},
                hovermode: 'closest'
            }};

            Plotly.react('plot', traces, layout, {{responsive: true}});
        }}

        // Initialize
        loadFiltersFromLocalStorage();
        loadCrossoversFromLocalStorage();
        loadUiStateFromLocalStorage();
        syncCrossoverFilters();  // Generate IIR filter entries from crossovers
        initUI();
        initFilterUI();
        renderCrossoverList();
        updatePlot();
    </script>
</body>
</html>'''

        # Write the HTML file
        output_path = self.interactive_plots_dir / 'freq_response_explorer.html'
        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"  Saved to {output_path}")

    def _add_static_grid(self, ax):
        """Helper to add standard grid to matplotlib axes"""
        # Major bold dotted lines (1k, 10k)
        for freq in config.GRID_FREQS_MAJOR:
            ax.axvline(freq, color='black', linestyle=':', linewidth=1.5, alpha=0.8)
        
        # Minor dotted lines (100, 200... 11k, 12k...)
        for freq in config.GRID_FREQS_MINOR:
            if freq <= config.FREQ_MAX:
                ax.axvline(freq, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        
        # Custom ticks
        valid_ticks, tick_text = get_valid_frequency_ticks()

        # Enforce ticks using FixedLocator/FixedFormatter to override LogScale defaults
        ax.xaxis.set_major_locator(ticker.FixedLocator(valid_ticks))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(tick_text))
        ax.xaxis.set_minor_locator(ticker.NullLocator()) # Hide default log minor ticks
        
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
        
        ax.grid(True, alpha=0.3, which='major', axis='y')

    def _add_interactive_grid(self, fig, row=None, col=None):
        """Helper to add standard grid to plotly figure"""
        for freq in config.GRID_FREQS_MAJOR:
            fig.add_vline(x=freq, line_dash="dot", line_color="gray", opacity=0.6, row=row, col=col)
        for freq in config.GRID_FREQS_MINOR:
            if freq <= config.FREQ_MAX:
                fig.add_vline(x=freq, line_dash="dot", line_color="lightgray", opacity=0.4, row=row, col=col)

    # ==================== Crossover Line Helpers ====================

    def _add_crossover_lines_static(self, ax, with_labels=True):
        """Add crossover reference lines to matplotlib axes"""
        for xo_freq in config.CROSSOVER_FREQUENCIES:
            ax.axvline(xo_freq, color='red', linestyle='--', linewidth=1, alpha=0.5)
            if with_labels:
                ax.text(xo_freq, ax.get_ylim()[1], f'{xo_freq} Hz',
                       ha='center', va='bottom', fontsize=8, color='red')

    def _add_crossover_lines_interactive(self, fig):
        """Add crossover reference lines to plotly figure"""
        for xo_freq in config.CROSSOVER_FREQUENCIES:
            fig.add_vline(x=xo_freq, line_dash="dash", line_color="red", opacity=0.5)

    # ==================== Driver Metric Plot Helpers ====================

    def _plot_drivers_static(self, ax, metric_key, **kwargs):
        """Add all driver metrics to matplotlib axes"""
        for driver in self.drivers:
            freq = self.calc_results[driver]['frequencies']
            data = self.calc_results[driver][metric_key]
            ax.semilogx(freq, data, label=driver, linewidth=2,
                       color=config.DRIVER_COLORS.get(driver), **kwargs)

    def _plot_drivers_interactive(self, fig, metric_key, **kwargs):
        """Add all driver metrics to plotly figure"""
        for driver in self.drivers:
            freq = self.calc_results[driver]['frequencies']
            data = self.calc_results[driver][metric_key]
            fig.add_trace(go.Scatter(
                x=freq, y=data, name=driver,
                line=dict(width=2, color=config.DRIVER_COLORS.get(driver)), **kwargs
            ))

    # ==================== Plot Finalization Helpers ====================

    def _finalize_static_plot(self, ax, ylabel=None, title=None, add_crossovers=True):
        """Apply consistent styling to static matplotlib plot"""
        self._add_static_grid(ax)
        ax.set_xlabel('Frequency (Hz)')
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontweight='bold')
        ax.set_xlim(config.FREQ_MIN, config.FREQ_MAX)
        if add_crossovers:
            self._add_crossover_lines_static(ax)
        ax.legend()

    def _finalize_interactive_plot(self, fig, title=None, ylabel=None, add_crossovers=True):
        """Apply consistent styling to interactive plotly plot"""
        self._add_interactive_grid(fig)
        self._configure_interactive_axis(fig)
        if add_crossovers:
            self._add_crossover_lines_interactive(fig)
        fig.update_layout(
            title=title or '',
            xaxis_title='Frequency (Hz)',
            yaxis_title=ylabel or ''
        )

    def plot_contour(self, driver, normalized=True, save_static=True, save_interactive=True):
        """Generate contour/heatmap plot for a driver

        When rear data is available, displays full 180° range:
        - Front (0° to 90°) on lower Y axis
        - Rear (90° to 180°) on upper Y axis (rear 0° = 180°, rear 90° = 90°)
        """
        print(f"Generating {'normalized' if normalized else 'absolute'} contour plot for {driver}...")

        freq = self.calc_results[driver]['frequencies']
        spl_matrix = self.calc_results[driver]['spl_matrix']
        calc = self.calc_results[driver]['calculator']
        has_rear = self.calc_results[driver].get('has_rear', False)
        rear_spl_matrix = self.calc_results[driver].get('rear_spl_matrix')

        # Interpolation for front hemisphere
        angles_fine_front = np.linspace(0, 90, 91)
        spl_interpolated_front = calc.interpolate_angles(angles_fine_front)

        # Check if we have rear data to include
        if has_rear and rear_spl_matrix is not None:
            # Create interpolator for rear data
            # Rear angles are 0-90, we'll display them as 180 down to 90
            # rear 0° -> 180°, rear 90° -> 90°
            rear_angles = self.calc_results[driver]['angles']  # Same angles as front
            angles_fine_rear = np.linspace(0, 90, 91)

            # Interpolate rear data
            from scipy.interpolate import interp1d
            spl_interpolated_rear = np.zeros((len(freq), len(angles_fine_rear)))
            for i in range(len(freq)):
                interpolator = interp1d(rear_angles, rear_spl_matrix[i, :],
                                       kind='cubic', fill_value='extrapolate')
                spl_interpolated_rear[i, :] = interpolator(angles_fine_rear)

            # Combine: front (0 to 90) + rear (90 to 180)
            # Rear 0° maps to 180°, rear 90° maps to 90°
            # So rear angles become: 180 - rear_angle
            # angles_fine_rear goes 0,1,2...90, so 180-angles = 180,179...90
            # We need 91,92...180 which is 180-89, 180-88, ... 180-0
            # That's the reverse of 180-angles_fine_rear
            angles_fine_rear_mapped = 180 - angles_fine_rear[::-1]  # 90, 91, ..., 180
            spl_rear_mapped = spl_interpolated_rear[:, ::-1]  # Flip to match 90 to 180 order

            # Combine (skip duplicate 90°)
            angles_combined = np.concatenate([angles_fine_front, angles_fine_rear_mapped[1:]])
            spl_combined = np.concatenate([spl_interpolated_front, spl_rear_mapped[:, 1:]], axis=1)

            angles_fine = angles_combined
            spl_interpolated = spl_combined
            y_lim = (0, 180)
            title_360 = " (180°)"
        else:
            angles_fine = angles_fine_front
            spl_interpolated = spl_interpolated_front
            y_lim = (0, 90)
            title_360 = ""

        if normalized:
            # Normalize to 0° (find index of 0° in combined array)
            zero_idx = np.abs(angles_fine - 0).argmin()
            spl_plot = spl_interpolated - spl_interpolated[:, zero_idx:zero_idx+1]
            cbar_label = 'Attenuation (dB)'
            vmax = 3
            vmin = -30
            title_suffix = "Normalized"
            cmap_name = 'RdYlBu_r'
        else:
            spl_plot = spl_interpolated
            cbar_label = 'SPL (dB)'
            vmin, vmax = np.percentile(spl_matrix, [5, 95])
            title_suffix = "Absolute SPL"
            cmap_name = 'RdYlBu_r'

        if save_static:
            fig_height = 12 if has_rear and rear_spl_matrix is not None else 8
            fig, ax = plt.subplots(figsize=(14, fig_height))

            if normalized:
                # Custom colormap with explicit stops to handle >0dB
                # Range is vmin to vmax (-30 to +3 typically)
                # We want 0dB to be Red, and positive to be Dark Red

                # Calculate normalized positions (0..1)
                span = vmax - vmin
                if span == 0: span = 1

                def get_pos(val):
                    return (val - vmin) / span

                # Define color stops
                # -30: Dark Blue (#000033)
                # -15: Blue (#0000FF)
                # -6:  Cyan (#00FFFF)
                # -3:  Green (#00FF00)
                # -1:  Yellow (#FFFF00)
                #  0:  Red (#FF0000)
                # +3:  Dark Red (#500000)

                stops = [
                    (0.0,  '#000033'), # Bottom
                    (get_pos(-15), '#0000FF'),
                    (get_pos(-10), '#00FFFF'),
                    (get_pos(-6),  '#00FF00'),
                    (get_pos(-3),  '#FFFF00'),
                    (get_pos(0),   '#FF0000'),
                    (1.0,          '#500000')  # Top (>0)
                ]
                # Filter out out-of-bounds stops just in case vmin/vmax change
                valid_stops = [(p, c) for p, c in stops if 0 <= p <= 1]
                # Ensure ends are covered
                if valid_stops[0][0] > 0: valid_stops.insert(0, (0.0, valid_stops[0][1]))
                if valid_stops[-1][0] < 1: valid_stops.append((1.0, valid_stops[-1][1]))

                cmap = mcolors.LinearSegmentedColormap.from_list('directivity_custom', valid_stops, N=256)
                levels = np.arange(vmin, vmax + 0.1, 0.2) # Fine steps for smooth gradient
                extend = 'both'
            else:
                cmap = cmap_name
                levels = np.arange(vmin, vmax + 1, 1)
                extend = 'both'

            contour = ax.contourf(freq, angles_fine, spl_plot.T, levels=levels, cmap=cmap, extend=extend)

            if normalized:
                contour_levels = [-20, -10, -6, -3, 0, 3]
                # Use dotted lines for contours
                cs = ax.contour(freq, angles_fine, spl_plot.T, levels=contour_levels,
                          colors='black', linestyles='dotted', linewidths=1.0, alpha=0.6)
                # Label the contours
                ax.clabel(cs, inline=True, fmt='%1.0f dB', fontsize=8, colors='black')

            # Add horizontal line at 90° for reference when showing rear data
            if has_rear and rear_spl_matrix is not None:
                ax.axhline(90, color='white', linestyle='-', linewidth=1.5, alpha=0.8)

            ax.set_xscale('log')
            self._add_static_grid(ax)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Angle (degrees)')
            ax.set_title(f'{driver} - Polar Response Contour ({title_suffix}){title_360}', fontweight='bold')
            ax.set_xlim(config.FREQ_MIN, config.FREQ_MAX)
            ax.set_ylim(y_lim)
            cbar = fig.colorbar(contour, ax=ax, label=cbar_label)

            # Add FRONT/REAR vertical labels on the right Y-axis when rear data is shown
            if has_rear and rear_spl_matrix is not None:
                # Create secondary Y-axis for labels
                ax2 = ax.twinx()
                ax2.set_ylim(y_lim)
                ax2.set_yticks([45, 135])
                ax2.set_yticklabels(['FRONT', 'REAR'], fontsize=12, fontweight='bold')
                ax2.tick_params(axis='y', colors='#1e40af', length=0)
                # Color the labels differently
                for i, label in enumerate(ax2.get_yticklabels()):
                    label.set_color('#1e40af' if i == 0 else '#dc2626')

            plt.tight_layout()
            suffix = "normalized" if normalized else "absolute"
            plt.savefig(self.static_plots_dir / f'core/{driver}_contour_{suffix}.png', bbox_inches='tight')
            plt.close()

        if save_interactive:
            # Subsample frequencies for interactive plots to reduce file size
            # (Full resolution kept for static plots, subsampled for interactive)
            step = 4  # Keep every 4th frequency point
            freq_sub = freq[::step]
            spl_sub = spl_plot[::step, :]

            # Convert numpy arrays to lists to avoid Plotly's binary bdata format
            # which can cause issues with browser rendering
            fig = go.Figure(data=go.Heatmap(
                z=spl_sub.T.tolist(),
                x=freq_sub.tolist(),
                y=angles_fine.tolist(),
                colorscale='RdYlBu_r' if not normalized else \
                           [[0, '#000033'], [0.2, '#0000FF'], [0.4, '#00FFFF'],
                            [0.6, '#00FF00'], [0.8, '#FFFF00'], [1, '#FF0000']],
                zmin=vmin, zmax=vmax,
                colorbar=dict(title=cbar_label),
                hovertemplate="Frequency: %{x:.1f} Hz<br>Angle: %{y:.1f}°<br>Level: %{z:.2f} dB<extra></extra>"
            ))

            # Add horizontal line at 90° for reference when showing rear data
            if has_rear and rear_spl_matrix is not None:
                fig.add_hline(y=90, line_color="white", line_width=2, opacity=0.8)

            self._add_interactive_grid(fig)
            self._configure_interactive_axis(fig)

            # Add FRONT/REAR annotations when rear data is shown
            annotations = []
            if has_rear and rear_spl_matrix is not None:
                annotations = [
                    dict(
                        x=1.06, y=0.25, xref='paper', yref='paper',
                        text='<b>FRONT</b>', showarrow=False,
                        font=dict(size=14, color='#1e40af'),
                        textangle=-90
                    ),
                    dict(
                        x=1.06, y=0.75, xref='paper', yref='paper',
                        text='<b>REAR</b>', showarrow=False,
                        font=dict(size=14, color='#dc2626'),
                        textangle=-90
                    )
                ]

            fig.update_layout(
                title=f'{driver} - Polar Response Contour ({title_suffix}){title_360}',
                yaxis_title='Angle (degrees)',
                yaxis_range=list(y_lim),
                annotations=annotations
            )

            suffix = "normalized" if normalized else "absolute"
            title_type = "Normalized" if normalized else "Absolute"
            self._write_compressed_html(
                fig,
                self.interactive_plots_dir / f'{driver}_contour_{suffix}.html',
                title=f'{driver} - Contour ({title_type})'
            )

    def plot_crossover_analysis(self, save_static=True, save_interactive=True):
        """Generate crossover match analysis"""
        print("Generating crossover match analysis...")

        # Define crossover scenarios (specific to LX521 drivers)
        # These are only relevant when processing the Andres measurement set
        crossovers = [
            {
                'freq': config.CROSSOVER_FREQUENCIES[0],
                'drivers': ['10F8424', 'L22MG'],
                'name': 'Crossover 120Hz'
            },
            {
                'freq': config.CROSSOVER_FREQUENCIES[1],
                'drivers': ['L22MG', 'MU10', '10F8424'],
                'name': 'Crossover 1000Hz'
            },
            {
                'freq': config.CROSSOVER_FREQUENCIES[2],
                'drivers': ['MU10', '10F8424', 'SEAS27T'],
                'name': 'Crossover 7000Hz'
            }
        ]

        for item in crossovers:
            # Skip if any required driver is not in this dataset
            if not all(d in self.drivers for d in item['drivers']):
                continue
            xo_freq = item['freq']
            drivers_list = item['drivers']
            title = item['name']
            
            # Zoom range for plot
            freq_min_zoom = xo_freq / (2 ** 1.5)
            freq_max_zoom = xo_freq * (2 ** 1.5)

            if save_static:
                fig, axes = plt.subplots(2, 1, figsize=(12, 10))
                
                # DI Panel
                for driver in drivers_list:
                    freq = self.calc_results[driver]['frequencies']
                    di = self.calc_results[driver]['di']
                    axes[0].semilogx(freq, di, label=driver, color=config.DRIVER_COLORS.get(driver), linewidth=2)
                
                axes[0].axvline(xo_freq, color='red', linestyle='--')
                axes[0].set_title(f'DI Comparison: {title} @ {xo_freq} Hz')
                axes[0].set_xlim(freq_min_zoom, freq_max_zoom)
                axes[0].legend()
                self._add_static_grid(axes[0])
                
                # Beamwidth Panel
                for driver in drivers_list:
                    freq = self.calc_results[driver]['frequencies']
                    bw = self.calc_results[driver]['beamwidth_6db']
                    axes[1].semilogx(freq, bw, label=driver, color=config.DRIVER_COLORS.get(driver), linewidth=2)
                
                axes[1].axvline(xo_freq, color='red', linestyle='--')
                axes[1].set_title(f'Beamwidth Comparison')
                axes[1].set_xlim(freq_min_zoom, freq_max_zoom)
                axes[1].set_ylim(0, 180)
                self._add_static_grid(axes[1])
                
                plt.tight_layout()
                plt.savefig(self.static_plots_dir / f'crossover/crossover_{xo_freq}Hz.png')
                plt.close()

            if save_interactive:
                fig = go.Figure()
                
                # DI Traces
                for driver in drivers_list:
                    freq = self.calc_results[driver]['frequencies']
                    di = self.calc_results[driver]['di']
                    fig.add_trace(go.Scatter(x=freq, y=di, name=f"{driver} DI",
                                           line=dict(width=2, color=config.DRIVER_COLORS.get(driver))))
                
                # Beamwidth Traces (dashed to distinguish?)
                # Or separate plots? Static uses subplots. Plotly can too.
                # Let's make subplots for interactive too.
                
                from plotly.subplots import make_subplots
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                   subplot_titles=(f'DI Comparison: {title}', 'Beamwidth Comparison'))
                
                # DI Panel (Row 1)
                for driver in drivers_list:
                    freq = self.calc_results[driver]['frequencies']
                    di = self.calc_results[driver]['di']
                    fig.add_trace(go.Scatter(x=freq, y=di, name=f"{driver} DI",
                                           line=dict(width=2, color=config.DRIVER_COLORS.get(driver))), row=1, col=1)
                
                # Beamwidth Panel (Row 2)
                for driver in drivers_list:
                    freq = self.calc_results[driver]['frequencies']
                    bw = self.calc_results[driver]['beamwidth_6db']
                    fig.add_trace(go.Scatter(x=freq, y=bw, name=f"{driver} BW",
                                           line=dict(width=2, dash='dash', color=config.DRIVER_COLORS.get(driver))), row=2, col=1)

                # Crossover Line
                fig.add_vline(x=xo_freq, line_dash="dash", line_color="red", opacity=0.5)

                self._add_interactive_grid(fig, row=1, col=1)
                self._add_interactive_grid(fig, row=2, col=1)
                self._configure_interactive_axis(fig)
                
                # Specific layout tweaks
                fig.update_layout(height=800, showlegend=True)
                fig.update_yaxes(title_text="Directivity Index (dB)", row=1, col=1)
                fig.update_yaxes(title_text="Beamwidth (deg)", range=[0, 180], row=2, col=1)
                fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
                
                # Zoom range
                fig.update_xaxes(range=[np.log10(freq_min_zoom), np.log10(freq_max_zoom)], type="log")

                fig.write_html(self.interactive_plots_dir / f'crossover_{xo_freq}Hz.html')

    def plot_dipole_analysis(self, save_static=True, save_interactive=True):
        """Generate Dipole Null (90deg) Analysis"""
        print("Generating dipole analysis...")
        
        if save_static:
            fig, ax = plt.subplots(figsize=config.FIG_SIZE_STATIC)
            for driver in self.drivers:
                freq = self.calc_results[driver]['frequencies']
                spl = self.calc_results[driver]['spl_matrix']
                # 0 deg is index 0 (assuming sorted angles 0..90)
                # 90 deg is index -1
                
                on_axis = spl[:, 0]
                off_axis_90 = spl[:, -1]
                
                null_depth = off_axis_90 - on_axis
                
                ax.semilogx(freq, null_depth, label=driver, linewidth=2,
                           color=config.DRIVER_COLORS.get(driver))

            ax.axhline(-6, color='gray', linestyle='--', alpha=0.5, label='-6 dB (Monopole)')
            ax.axhline(-20, color='black', linestyle='--', alpha=0.5, label='-20 dB (Target Null)')

            self._add_static_grid(ax)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('90° Level relative to On-Axis (dB)')
            ax.set_title('Dipole Null Analysis (90° Attenuation)', fontweight='bold')
            ax.legend()
            ax.set_xlim(config.FREQ_MIN, config.FREQ_MAX)
            ax.set_ylim(-40, 10)
            
            plt.tight_layout()
            plt.savefig(self.static_plots_dir / 'core/dipole_null_analysis.png')
            plt.close()

    def plot_polar_diagrams(self, freqs=[500, 1000, 2000, 4000], save_static=True):
        """Generate Circular Polar Plots at specific frequencies (Single Driver)

        Supports full 360° plots when rear data is available.
        """
        print("Generating circular polar diagrams (Single Driver)...")

        for driver in self.drivers:
            res = self.calc_results[driver]
            f_axis = res['frequencies']
            has_rear = res.get('has_rear', False)

            if save_static:
                # Dynamic grid size
                n_plots = len(freqs)
                cols = 3
                rows = (n_plots + cols - 1) // cols

                fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), subplot_kw={'projection': 'polar'})
                axes = axes.flatten()

                for i, target_f in enumerate(freqs):
                    ax = axes[i]
                    # Find nearest freq index
                    idx = np.abs(f_axis - target_f).argmin()
                    actual_f = f_axis[idx]

                    # Use helper to build polar data (handles 360° if rear available)
                    angles_rad, data_norm = self._build_360_polar_data(driver, idx)

                    ax.plot(angles_rad, data_norm, linewidth=2, color=config.DRIVER_COLORS.get(driver, 'blue'))
                    ax.set_title(f'{actual_f:.0f} Hz', va='bottom', fontweight='bold')
                    ax.set_theta_zero_location("N")
                    ax.set_theta_direction(-1)
                    ax.set_rlabel_position(45)
                    ax.set_ylim(-40, 5)
                    ax.grid(True)

                    # For 360° plots, show full circle
                    if has_rear:
                        ax.set_thetamin(0)
                        ax.set_thetamax(360)
                
                # Hide empty subplots
                for j in range(i + 1, len(axes)):
                    axes[j].axis('off')

                title_suffix = " (360°)" if has_rear else " (Normalized)"
                fig.suptitle(f'{driver} - Polar Response{title_suffix}', fontweight='bold', fontsize=16)
                plt.tight_layout()
                plt.savefig(self.static_plots_dir / f'polar/{driver}_polar_circular.png')
                plt.close()

    def plot_polar_multi_driver_comparison(self, freqs=[500, 1000, 2000, 4000], save_static=True):
        """Generate Circular Polar Plots with ALL drivers overlaid per frequency

        Supports full 360° plots when rear data is available.
        """
        print("Generating circular polar diagrams (Overlaid Comparison)...")

        if save_static:
            # Dynamic grid size
            n_plots = len(freqs)
            cols = 3
            rows = (n_plots + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), subplot_kw={'projection': 'polar'})
            axes = axes.flatten()

            # Check if any driver has rear data
            any_has_rear = any(self.calc_results[d].get('has_rear', False) for d in self.drivers)

            for i, target_f in enumerate(freqs):
                ax = axes[i]
                ax.set_title(f'{target_f} Hz', va='bottom', fontweight='bold', fontsize=14)

                for driver in self.drivers:
                    res = self.calc_results[driver]
                    f_axis = res['frequencies']

                    # Find nearest freq index
                    idx = np.abs(f_axis - target_f).argmin()

                    # Use helper to build polar data (handles 360° if rear available)
                    angles_rad, data_norm = self._build_360_polar_data(driver, idx)

                    ax.plot(angles_rad, data_norm, linewidth=2, label=driver,
                           color=config.DRIVER_COLORS.get(driver, 'blue'))

                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_rlabel_position(45)
                ax.set_ylim(-40, 5)
                ax.grid(True)

                # For 360° plots, show full circle
                if any_has_rear:
                    ax.set_thetamin(0)
                    ax.set_thetamax(360)

                # Legend only on first plot
                if i == 0:
                    ax.legend(loc='lower left', bbox_to_anchor=(-0.3, -0.3), fontsize=8)

            # Hide empty subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')

            fig.suptitle(f'Multi-Driver Polar Comparison (Normalized)', fontweight='bold', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.static_plots_dir / f'polar/polar_gallery_overlaid.png')
            plt.close()

    def plot_polar_interactive_slider(self):
        """Generate Interactive Polar Explorer with driver selection and 360° support"""
        print("Generating interactive polar explorer (with driver selection)...")

        # Plot Limits
        limit_min = -40
        limit_max = 10

        ref_driver = self.drivers[0]
        freqs = self.calc_results[ref_driver]['frequencies']

        # Check if any driver has rear data for full 360°
        any_has_rear = any(self.calc_results[d].get('has_rear', False) for d in self.drivers)

        # Efficient Frequency Sampling - use more steps for smoother slider
        n_steps = 300
        min_idx = np.abs(freqs - config.FREQ_MIN).argmin()
        max_idx = np.abs(freqs - config.FREQ_MAX).argmin()
        stride = max(1, (max_idx - min_idx) // n_steps)
        indices = list(range(min_idx, max_idx + 1, stride))

        # Store all data for slider steps
        # structure: all_step_data[step_index][driver_index] = {'theta': array, 'r': array}
        all_step_data = []

        for i in indices:
            step_data = []
            for driver in self.drivers:
                # Use _build_360_polar_data helper for proper 360° support
                angles_rad, spl_norm = self._build_360_polar_data(driver, i)
                # Convert radians to degrees for Plotly
                angles_deg = np.degrees(angles_rad)
                step_data.append({'theta': angles_deg, 'r': spl_norm})
            all_step_data.append(step_data)

        # Create initial traces
        initial_traces = []
        for d_idx, driver in enumerate(self.drivers):
            initial_traces.append(go.Scatterpolar(
                r=all_step_data[0][d_idx]['r'],
                theta=all_step_data[0][d_idx]['theta'],
                mode='lines',
                name=driver,
                visible=True,
                line=dict(width=3, color=config.DRIVER_COLORS.get(driver, 'blue'))
            ))

        # Create Slider Steps (Restyle) - update both r and theta
        base_trace_indices = list(range(len(self.drivers)))
        steps = []
        for i, idx in enumerate(indices):
            f = freqs[idx]
            step = dict(
                method="restyle",
                args=[{
                    "r": [all_step_data[i][d]['r'] for d in range(len(self.drivers))],
                    "theta": [all_step_data[i][d]['theta'] for d in range(len(self.drivers))]
                }, base_trace_indices],
                label=f"{f:.0f}"
            )
            steps.append(step)

        # Configure angular axis based on whether we have 360° data
        if any_has_rear:
            angular_axis = dict(
                direction="clockwise",
                rotation=90,
                gridcolor='lightgray',
                tickmode="linear",
                tick0=0,
                dtick=30,
                ticksuffix="°",
                tickfont=dict(size=9),
                ticks="inside",
                showticklabels=True,
                ticklabelstep=2
            )
        else:
            angular_axis = dict(
                direction="clockwise",
                rotation=90,
                gridcolor='lightgray',
                tickmode="array",
                tickvals=[-90, -60, -30, 0, 30, 60, 90],
                ticktext=["-90°", "-60°", "-30°", "<b>0°</b>", "30°", "60°", "90°"]
            )

        radial_axis = dict(
            range=[limit_min, limit_max],
            visible=True,
            showline=True,
            gridcolor='lightgray',
            showticklabels=True
        )
        if any_has_rear:
            radial_axis["ticks"] = "inside"

        if any_has_rear:
            legend_cfg = dict(
                yanchor="top",
                y=0.98,
                xanchor="right",
                x=0.98,
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.7)"
            )
            margin_cfg = dict(l=60, r=60, t=80, b=100)
        else:
            legend_cfg = dict(
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.05,
                font=dict(size=14),
                bgcolor="rgba(255,255,255,0.5)"
            )
            margin_cfg = dict(l=60, r=150, t=80, b=100)

        polar_config = dict(
            bgcolor='white',
            radialaxis=radial_axis,
            angularaxis=angular_axis
        )
        if any_has_rear:
            polar_config["domain"] = dict(x=[0.05, 0.95], y=[0.14, 0.98])

        # Layout
        layout = go.Layout(
            title=dict(
                text="<b>Polar Response Explorer</b>" + (" (360°)" if any_has_rear else ""),
                font=dict(size=24),
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            font=dict(family="Arial, sans-serif", size=12),
            polar=polar_config,
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "visible": False
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.05,
                "y": 0,
                "steps": steps
            }],
            legend=legend_cfg,
            margin=margin_cfg,
            paper_bgcolor="white"
        )

        fig = go.Figure(data=initial_traces, layout=layout)

        # Generate HTML with custom frequency input
        html_content = fig.to_html(
            include_plotlyjs='cdn',
            full_html=True,
            config={"responsive": True, "showTips": False}
        )
        if not html_content.lstrip().lower().startswith('<!doctype html>'):
            html_content = '<!DOCTYPE html>\n' + html_content

        # Build frequency lookup array for JavaScript
        freq_values = [freqs[idx] for idx in indices]
        freq_js_array = ','.join([f'{f:.1f}' for f in freq_values])

        extra_polar_payload = None
        extra_set_name = "juan-baffleless"
        extra_data = self._load_extra_set_data(extra_set_name)
        if extra_data:
            base_step_data = [
                [
                    {
                        "theta": (entry["theta"].tolist()
                                  if hasattr(entry["theta"], "tolist")
                                  else entry["theta"]),
                        "r": (entry["r"].tolist()
                              if hasattr(entry["r"], "tolist")
                              else entry["r"])
                    }
                    for entry in step
                ]
                for step in all_step_data
            ]
            extra_results = {}
            for driver, driver_data in extra_data.items():
                try:
                    freq, angles, spl_matrix, phase_matrix = create_polar_matrix_from_dict(driver_data)
                    DirectivityCalculator(freq, angles, spl_matrix)
                except ValueError as exc:
                    print(f"Warning: Skipping extra driver '{driver}' for polar explorer: {exc}")
                    continue

                rear_spl_matrix = None
                has_rear = False
                if driver_data.get('has_rear') and 'rear_angles' in driver_data:
                    _, rear_angles, rear_spl_matrix, rear_phase_matrix = create_polar_matrix_from_dict(
                        {
                            'angles': driver_data['rear_angles'],
                            'common_frequencies': driver_data['common_frequencies']
                        }
                    )
                    if np.array_equal(rear_angles, angles):
                        has_rear = True
                    else:
                        rear_spl_matrix = None

                extra_results[driver] = {
                    'frequencies': freq,
                    'angles': angles,
                    'spl_matrix': spl_matrix,
                    'rear_spl_matrix': rear_spl_matrix,
                    'has_rear': has_rear,
                }

            if extra_results:
                extra_drivers = sorted(extra_results.keys())
                extra_freqs = extra_results[extra_drivers[0]]['frequencies']
                extra_indices = [int(np.abs(extra_freqs - f).argmin()) for f in freq_values]

                extra_step_data = []
                for idx in extra_indices:
                    step = []
                    for driver in extra_drivers:
                        res = extra_results[driver]
                        angles_rad, spl_norm = self._build_polar_data_from_result(res, idx, use_rear=any_has_rear)
                        step.append({'theta': np.degrees(angles_rad).tolist(), 'r': spl_norm.tolist()})
                    extra_step_data.append(step)

                extra_polar_payload = {
                    "label": "Juan baffleless drivers",
                    "drivers": extra_drivers,
                    "stepData": extra_step_data,
                    "baseDrivers": self.drivers,
                    "baseStepData": base_step_data,
                }

        extra_polar_button_html = ""
        extra_polar_css = ""
        extra_polar_script = ""
        if extra_polar_payload:
            extra_polar_button_html = f'''
<div class="extra-driver-container" id="extraPolarControl">
    <div class="extra-driver-row">
        <button id="loadExtraPolarBtn" onclick="loadExtraPolarDrivers()">Load Juan baffleless drivers</button>
    </div>
</div>
'''
            extra_polar_css = '''
.extra-driver-container {
    position: absolute;
    top: 0;
    right: 0;
    z-index: 1000;
    background: white;
    padding: 6px 12px;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    font-family: Arial, sans-serif;
}
.extra-driver-container button {
    padding: 5px 12px;
    font-size: 13px;
    cursor: pointer;
    background: #eef2ff;
    color: #1e3a8a;
    border: 1px solid #c7d2fe;
    border-radius: 4px;
}
.extra-driver-container button:hover {
    background: #e0e7ff;
}
.extra-driver-container button:disabled {
    cursor: default;
    opacity: 0.6;
}
.extra-driver-row {
    display: flex;
    align-items: center;
    gap: 6px;
}
'''
            extra_polar_script = f'''
const extraPolarDataset = {json.dumps(extra_polar_payload)};
const polarDriverColors = {json.dumps(config.DRIVER_COLORS)};
console.log('[extra-polar] dataset', extraPolarDataset);

function buildPolarTraces(drivers, stepEntries) {{
    return drivers.map((driver, idx) => {{
        const entry = stepEntries[idx];
        const color = polarDriverColors[driver] || '#2563eb';
        return {{
            r: entry.r,
            theta: entry.theta,
            mode: 'lines',
            name: driver,
            type: 'scatterpolar',
            line: {{ width: 3, color: color }}
        }};
    }});
}}

function positionExtraPolarControl() {{
    const plotDiv = document.querySelector('.plotly-graph-div');
    const control = document.getElementById('extraPolarControl');
    if (!plotDiv || !control) {{
        console.warn('[extra-polar] missing plotDiv/control', {{ plotDiv, control }});
        return;
    }}

    if (!plotDiv.contains(control)) {{
        plotDiv.appendChild(control);
    }}
    if (!plotDiv.style.position) {{
        plotDiv.style.position = 'relative';
    }}

    const plotRect = plotDiv.getBoundingClientRect();
    const legend = plotDiv.querySelector('.legend');
    if (legend) {{
        const legendRect = legend.getBoundingClientRect();
        const top = Math.max(0, legendRect.bottom - plotRect.top + 8);
        const right = Math.max(12, plotRect.right - legendRect.right);
        control.style.top = `${{top}}px`;
        control.style.right = `${{right}}px`;
        control.style.left = 'auto';
        console.log('[extra-polar] positioned via legend', {{ top, right }});
        return;
    }}

    const layout = plotDiv._fullLayout;
    const size = layout?._size;
    if (layout?.legend && size) {{
        const legendLayout = layout.legend;
        const y = legendLayout.y ?? 1;
        const x = legendLayout.x ?? 1;
        const fontSize = legendLayout.font?.size || layout.font?.size || 12;
        const itemHeight = fontSize + 6;
        const count = (extraPolarDataset?.baseDrivers?.length || 0) + 1;
        const legendTop = size.t + (1 - y) * size.h;
        const legendBottom = legendTop + (itemHeight * count);
        const legendRight = size.l + (x * size.w);
        const top = Math.max(0, legendBottom + 8);
        const right = Math.max(12, plotRect.width - legendRight);
        control.style.top = `${{top}}px`;
        control.style.right = `${{right}}px`;
        control.style.left = 'auto';
        console.log('[extra-polar] positioned via layout fallback', {{ top, right }});
        return;
    }}

    control.style.top = '160px';
    control.style.right = '22px';
    control.style.left = 'auto';
    console.warn('[extra-polar] legend not found, using fallback');
}}

function loadExtraPolarDrivers() {{
    console.log('[extra-polar] load requested');
    if (!extraPolarDataset || extraPolarDataset.loaded) {{
        console.warn('[extra-polar] dataset missing or already loaded');
        return;
    }}
    const plotDiv = document.querySelector('.plotly-graph-div');
    if (!plotDiv) {{
        console.warn('[extra-polar] plotDiv not found');
        return;
    }}
    console.log('[extra-polar] plotDiv found', {{
        hasLayout: !!plotDiv._fullLayout,
        dataLen: plotDiv.data?.length
    }});
    const baseDrivers = extraPolarDataset.baseDrivers || [];
    const baseSteps = extraPolarDataset.baseStepData || [];
    const extraDrivers = extraPolarDataset.drivers || [];
    const extraSteps = extraPolarDataset.stepData || [];
    if (!baseDrivers.length || !baseSteps.length || !extraDrivers.length || !extraSteps.length) {{
        console.warn('[extra-polar] missing data', {{
            baseDrivers: baseDrivers.length,
            baseSteps: baseSteps.length,
            extraDrivers: extraDrivers.length,
            extraSteps: extraSteps.length
        }});
        return;
    }}

    const activeIdx = plotDiv.layout?.sliders?.[0]?.active ?? 0;
    const stepIdx = Math.max(0, Math.min(activeIdx, extraSteps.length - 1));
    const extraTraces = buildPolarTraces(extraDrivers, extraSteps[stepIdx] || []);
    const extraTraceIndices = extraDrivers.map((_, i) => baseDrivers.length + i);
    plotDiv._extraPolarTraceIndices = extraTraceIndices;
    plotDiv._extraPolarStepData = extraSteps;
    console.log('[extra-polar] add traces', {{ activeIdx, stepIdx, extraTraces: extraTraces.length }});

    Plotly.addTraces(plotDiv, extraTraces).then(() => {{
        console.log('[extra-polar] addTraces ok');
        console.log('[extra-polar] data length after add', plotDiv.data?.length);
        if (!plotDiv._extraPolarSliderHook) {{
            plotDiv._extraPolarSliderHook = true;
            plotDiv.on('plotly_sliderchange', (e) => {{
                const idx = e?.slider?.active ?? e?.stepIndex ?? plotDiv.layout?.sliders?.[0]?.active ?? 0;
                const step = extraSteps[Math.max(0, Math.min(idx, extraSteps.length - 1))] || [];
                Plotly.restyle(plotDiv, {{
                    r: step.map(d => d.r),
                    theta: step.map(d => d.theta)
                }}, extraTraceIndices);
            }});
        }}
        positionExtraPolarControl();

        extraPolarDataset.loaded = true;
        const btn = document.getElementById('loadExtraPolarBtn');
        if (btn) {{
            btn.textContent = 'Juan baffleless drivers loaded';
            btn.disabled = true;
        }}
    }}).catch((err) => {{
        console.error('[extra-polar] addTraces failed', err);
    }});
}}

function initExtraPolarControl() {{
    const plotDiv = document.querySelector('.plotly-graph-div');
    if (!plotDiv) {{
        console.warn('[extra-polar] plotDiv missing at init');
        return;
    }}
    positionExtraPolarControl();
    setTimeout(positionExtraPolarControl, 200);
    setTimeout(positionExtraPolarControl, 600);
    if (plotDiv.on) {{
        plotDiv.on('plotly_afterplot', positionExtraPolarControl);
        plotDiv.on('plotly_relayout', positionExtraPolarControl);
    }}
    plotDiv.addEventListener('click', () => {{
        setTimeout(positionExtraPolarControl, 50);
    }});
    if (window.Plotly && Plotly.Plots?.resize) {{
        Plotly.Plots.resize(plotDiv);
        setTimeout(() => Plotly.Plots.resize(plotDiv), 150);
    }}
}}
function waitForExtraPolarPlotReady(attempt = 0) {{
    const plotDiv = document.querySelector('.plotly-graph-div');
    if (!plotDiv) {{
        console.warn('[extra-polar] plotDiv missing, retry', attempt);
        if (attempt < 30) setTimeout(() => waitForExtraPolarPlotReady(attempt + 1), 200);
        return;
    }}
    if (!plotDiv._fullLayout) {{
        if (attempt % 5 === 0) {{
            console.log('[extra-polar] waiting for Plotly layout', attempt);
        }}
        if (attempt < 30) setTimeout(() => waitForExtraPolarPlotReady(attempt + 1), 200);
        return;
    }}
    console.log('[extra-polar] Plotly ready', {{
        hasLayout: !!plotDiv._fullLayout,
        hasLegend: !!plotDiv._fullLayout?.legend,
        dataLen: plotDiv.data?.length
    }});
    initExtraPolarControl();
}}

waitForExtraPolarPlotReady();
'''

        # Custom JavaScript for manual frequency entry
        custom_js = f'''
<style>
html, body {{
    height: 100%;
    margin: 0;
}}
body > div {{
    height: 100%;
}}
.plotly-graph-div {{
    width: 100%;
    height: 100%;
    min-height: 80vh;
}}
.freq-input-container {{
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    top: 0;
    z-index: 1000;
    font-family: Arial, sans-serif;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.freq-input-container label {{
    font-size: 13px;
    margin: 0;
}}
.freq-input-container input {{
    width: 90px;
    padding: 4px 6px;
    font-size: 13px;
    border: 1px solid #ccc;
    border-radius: 4px;
}}
.freq-input-unit {{
    font-size: 13px;
    color: #1f2937;
}}
    {extra_polar_css}
	</style>
	<div class="freq-input-container">
	    <label for="freqInput">Frequency:</label>
	    <input type="number" id="freqInput" min="{config.FREQ_MIN}" max="{config.FREQ_MAX}" placeholder="Hz">
        <span class="freq-input-unit">Hz</span>
	</div>
    {extra_polar_button_html}
	<script>
	var baseTraceIndices = Array.from({{length: {len(self.drivers)}}}, (_, i) => i);
	var freqValues = [{freq_js_array}];

function getClosestFreqIndex(targetFreq) {{
    // Find closest frequency index
    var closestIdx = 0;
    var minDiff = Math.abs(freqValues[0] - targetFreq);
    for (var i = 1; i < freqValues.length; i++) {{
        var diff = Math.abs(freqValues[i] - targetFreq);
        if (diff < minDiff) {{
            minDiff = diff;
            closestIdx = i;
        }}
    }}
    return closestIdx;
}}

function findSliderElement(plotDiv) {{
    return (
        plotDiv.querySelector('.slider-container') ||
        plotDiv.querySelector('.slider-group') ||
        plotDiv.querySelector('.slider') ||
        plotDiv.querySelector('.sliders')
    );
}}

function positionFrequencyInput() {{
    const plotDiv = document.querySelector('.plotly-graph-div');
    const container = document.querySelector('.freq-input-container');
    if (!plotDiv || !container) {{
        return;
    }}

    if (!plotDiv.contains(container)) {{
        plotDiv.appendChild(container);
    }}
    if (!plotDiv.style.position) {{
        plotDiv.style.position = 'relative';
    }}

    const plotRect = plotDiv.getBoundingClientRect();
    const sliderEl = findSliderElement(plotDiv);
    if (sliderEl) {{
        const sliderRect = sliderEl.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        const top = Math.max(0, sliderRect.top - plotRect.top - containerRect.height - 6);
        container.style.top = `${{top}}px`;
        container.style.bottom = 'auto';
        return;
    }}

    const layout = plotDiv._fullLayout;
    const size = layout?._size;
    const slider = layout?.sliders?.[0];
    if (size && slider) {{
        const padB = slider.pad?.b ?? 0;
        const sliderTop = size.t + size.h + padB;
        const containerRect = container.getBoundingClientRect();
        const top = Math.max(0, sliderTop - containerRect.height - 6);
        container.style.top = `${{top}}px`;
        container.style.bottom = 'auto';
    }}
}}

function setFrequencyFromInput() {{
    var inputEl = document.getElementById('freqInput');
    if (!inputEl) {{
        return;
    }}
    var targetFreq = parseFloat(inputEl.value);
    if (isNaN(targetFreq)) {{
        return;
    }}
    if (targetFreq < {config.FREQ_MIN}) {{
        targetFreq = {config.FREQ_MIN};
    }} else if (targetFreq > {config.FREQ_MAX}) {{
        targetFreq = {config.FREQ_MAX};
    }}

    var closestIdx = getClosestFreqIndex(targetFreq);
    // Update slider
    var plotDiv = document.querySelector('.plotly-graph-div');
    if (plotDiv) {{
        Plotly.relayout(plotDiv, {{'sliders[0].active': closestIdx}});
        // Also trigger the restyle to update the plot
        var slider = plotDiv.layout.sliders[0];
        if (slider && slider.steps && slider.steps[closestIdx]) {{
            var step = slider.steps[closestIdx];
            var traceIndices = (step.args && step.args[1]) ? step.args[1] : baseTraceIndices;
            Plotly.restyle(plotDiv, step.args[0], traceIndices);
            if (plotDiv._extraPolarTraceIndices && plotDiv._extraPolarStepData) {{
                var extraStep = plotDiv._extraPolarStepData[Math.max(0, Math.min(closestIdx, plotDiv._extraPolarStepData.length - 1))] || [];
                Plotly.restyle(plotDiv, {{
                    r: extraStep.map(d => d.r),
                    theta: extraStep.map(d => d.theta)
                }}, plotDiv._extraPolarTraceIndices);
            }}
        }}
    }}

    inputEl.value = Math.round(freqValues[closestIdx]);
}}

function syncInputToSlider(idx) {{
    var inputEl = document.getElementById('freqInput');
    if (!inputEl) {{
        return;
    }}
    inputEl.value = Math.round(freqValues[idx] || freqValues[0]);
}}

function schedulePlotResize() {{
    var plotDiv = document.querySelector('.plotly-graph-div');
    if (!plotDiv || !window.Plotly || !Plotly.Plots || !Plotly.Plots.resize) {{
        return;
    }}
    Plotly.Plots.resize(plotDiv);
    setTimeout(function() {{ Plotly.Plots.resize(plotDiv); }}, 150);
}}

function initFrequencyInput() {{
    var inputEl = document.getElementById('freqInput');
    if (!inputEl) {{
        return;
    }}
    syncInputToSlider(0);
    positionFrequencyInput();
    inputEl.addEventListener('change', setFrequencyFromInput);
    inputEl.addEventListener('blur', setFrequencyFromInput);
    inputEl.addEventListener('keypress', function(e) {{
        if (e.key === 'Enter') {{
            setFrequencyFromInput();
        }}
    }});

    function hookSliderChanges(attempt) {{
        var plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv || !plotDiv.on) {{
            if (attempt < 20) {{
                setTimeout(function() {{ hookSliderChanges(attempt + 1); }}, 200);
            }}
            return;
        }}
        if (plotDiv._freqInputHook) {{
            return;
        }}
        plotDiv._freqInputHook = true;
        plotDiv.on('plotly_sliderchange', function(e) {{
            var idx = (e && e.slider && e.slider.active);
            if (idx == null) {{
                idx = (e && e.stepIndex) || 0;
            }}
            syncInputToSlider(idx);
        }});
        plotDiv.on('plotly_afterplot', positionFrequencyInput);
        plotDiv.on('plotly_relayout', positionFrequencyInput);
    }}

    hookSliderChanges(0);
    schedulePlotResize();
    window.addEventListener('resize', function() {{
        setTimeout(positionFrequencyInput, 100);
    }});
}}

initFrequencyInput();
    {extra_polar_script}
	</script>
	'''

        # Insert custom HTML before closing body tag
        html_content = html_content.replace('</body>', custom_js + '</body>')

        output_path = self.interactive_plots_dir / "polar/polar_explorer.html"
        with open(output_path, 'w') as f:
            f.write(html_content)

    def generate_measurement_summary_html(self):
        """Generate HTML summary of all measurements with metadata"""
        print("Generating measurement summary HTML...")

        # CSS styles for measurement summary
        styles = f'''        body {{
            font-family: {FONT_STACK};
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
            background: #f8fafc;
            color: #1e293b;
        }}
        h1 {{ color: #2563eb; border-bottom: 2px solid #2563eb; padding-bottom: 0.5rem; }}
        h2 {{ color: #1e40af; margin-top: 2rem; }}
        .config-box {{
            background: #dbeafe;
            border: 1px solid #93c5fd;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin: 1rem 0 2rem;
        }}
        .config-box h3 {{ margin: 0 0 0.5rem; color: #1e40af; }}
        .config-box p {{ margin: 0.25rem 0; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #e2e8f0;
            padding: 0.75rem;
            text-align: left;
        }}
        th {{
            background: #1e40af;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{ background: #f1f5f9; }}
        tr:hover {{ background: #e0f2fe; }}
        .angle-label {{ font-weight: 600; color: #1e40af; }}
        .front {{ color: #059669; }}
        .rear {{ color: #dc2626; }}
        .notes {{ font-size: 0.875rem; color: #64748b; white-space: pre-line; }}
        .filename {{ font-family: monospace; font-size: 0.875rem; }}
        .date {{ font-size: 0.875rem; color: #64748b; }}
        .processing-notes {{ font-size: 0.875rem; color: #b45309; font-weight: 500; }}'''

        # Build body content
        body_parts = ['    <h1>Measurement Summary</h1>']

        # Global config
        body_parts.append(f'''
    <div class="config-box">
        <h3>Processing Configuration</h3>
        <p><strong>Time Gating:</strong> Left: {self.global_config.get('gate_left_ms', 0):.1f} ms, Right: {self.global_config.get('gate_right_ms', 0):.1f} ms</p>
        <p><strong>Smoothing:</strong> {self.global_config.get('smoothing_str', 'None')}</p>
    </div>''')

        # For each driver
        for driver in self.drivers:
            driver_data = self.data[driver]
            has_rear = driver_data.get('has_rear', False)

            body_parts.append(f'''
    <h2>{driver}</h2>
    <table>
        <thead>
            <tr>
                <th>Angle</th>
                <th>Measurement Name</th>
                <th>Notes</th>
                <th>Date</th>
                <th>Processing Notes</th>
            </tr>
        </thead>
        <tbody>''')

            # Front angles
            for angle in sorted(driver_data['angles'].keys()):
                angle_data = driver_data['angles'][angle]
                meta = angle_data.get('metadata', {})
                title = meta.get('title', '')
                notes = meta.get('notes', '').replace('\n', '<br>')
                date = meta.get('date', '')

                processing_notes = ''
                if angle_data.get('timing_corrected', False):
                    offset_ms = angle_data.get('timing_offset_ms', 0.0)
                    processing_notes = f'Peak aligned: {offset_ms:.2f}ms'

                body_parts.append(f'''
            <tr>
                <td class="angle-label"><span class="front">F{angle}</span></td>
                <td class="filename">{title}</td>
                <td class="notes">{notes}</td>
                <td class="date">{date}</td>
                <td class="processing-notes">{processing_notes}</td>
            </tr>''')

            # Rear angles if present
            if has_rear and 'rear_angles' in driver_data:
                for angle in sorted(driver_data['rear_angles'].keys()):
                    angle_data = driver_data['rear_angles'][angle]
                    meta = angle_data.get('metadata', {})
                    title = meta.get('title', '')
                    notes = meta.get('notes', '').replace('\n', '<br>')
                    date = meta.get('date', '')

                    processing_notes = ''
                    if angle_data.get('timing_corrected', False):
                        offset_ms = angle_data.get('timing_offset_ms', 0.0)
                        processing_notes = f'Peak aligned: {offset_ms:.2f}ms'

                    body_parts.append(f'''
            <tr>
                <td class="angle-label"><span class="rear">R{angle}</span></td>
                <td class="filename">{title}</td>
                <td class="notes">{notes}</td>
                <td class="date">{date}</td>
                <td class="processing-notes">{processing_notes}</td>
            </tr>''')

            body_parts.append('''
        </tbody>
    </table>''')

        body = '\n'.join(body_parts)

        # Build and write HTML using helper
        html = build_html_page('Measurement Summary', styles, body)
        output_path = self.interactive_plots_dir / "measurement_summary.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"  Saved to {output_path}")

    def generate_all_plots(self):
        """Generate all configured visualizations"""
        self.plot_di_comparison()
        self.plot_beamwidth_comparison()
        self.plot_dipole_analysis()
        self.plot_frequency_response_by_angle()
        self.plot_frequency_response_explorer()

        for driver in self.drivers:
            self.plot_contour(driver, normalized=True)
            self.plot_contour(driver, normalized=False)

        freqs_polar = [500, 1000, 2000, 4000, 6000, 6500, 7000, 8000, 10000, 15000, 20000]
        self.plot_polar_diagrams(freqs=freqs_polar)
        self.plot_polar_multi_driver_comparison(freqs=freqs_polar)
        self.plot_polar_interactive_slider()
        self.plot_crossover_analysis()
        self.generate_measurement_summary_html()
        print("\nAll visualizations generated.")

if __name__ == "__main__":
    viz = PolarResponseVisualizer()
    viz.generate_all_plots()
