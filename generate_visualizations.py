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

# Import centralized configuration
import config

# ==================== HTML Template Constants ====================
# Reusable HTML fragments for embedded templates

HTML_DOCTYPE = '<!DOCTYPE html>\n<html lang="en">\n'

HTML_HEAD_START = '''<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
'''

HTML_PLOTLY_SCRIPT = '    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>\n'
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

        self.loader = PolarDataLoader(connect_to_rew=False)
        self.data = self.loader.load_from_hdf5(data_path)

        # Extract config (if present) and remove from driver list
        self.global_config = self.data.pop('_config', {
            'gate_left_ms': 0.0,
            'gate_right_ms': 0.0,
            'smoothing': 0,
            'smoothing_str': 'None'
        })

        self.drivers = sorted(self.data.keys())

        # Calculate directivity metrics for all drivers
        # Note: DI/beamwidth/sound power are calculated from front hemisphere only (0-90 deg),
        # following industry standard practice. Rear data is stored separately for 360-deg polar plots.
        self.calc_results = {}
        for driver in self.drivers:
            freq, angles, spl_matrix = create_polar_matrix_from_dict(self.data[driver])
            calc = DirectivityCalculator(freq, angles, spl_matrix)

            # Also create rear SPL matrix if available
            rear_spl_matrix = None
            if self.data[driver].get('has_rear') and 'rear_angles' in self.data[driver]:
                _, rear_angles, rear_spl_matrix = create_polar_matrix_from_dict(
                    {'angles': self.data[driver]['rear_angles'],
                     'common_frequencies': self.data[driver]['common_frequencies']}
                )

            self.calc_results[driver] = {
                'frequencies': freq,
                'angles': angles,
                'spl_matrix': spl_matrix,
                'rear_spl_matrix': rear_spl_matrix,
                'has_rear': self.data[driver].get('has_rear', False),
                'calculator': calc,
                'di': calc.calculate_directivity_index(),
                'beamwidth_6db': calc.calculate_beamwidth(-6),
                'beamwidth_3db': calc.calculate_beamwidth(-3),
                'sound_power': calc.calculate_sound_power()
            }

        # Ensure output directories exist
        self.static_plots_dir.mkdir(parents=True, exist_ok=True)
        (self.static_plots_dir / "core").mkdir(exist_ok=True)
        (self.static_plots_dir / "crossover").mkdir(exist_ok=True)
        (self.static_plots_dir / "polar").mkdir(exist_ok=True)

        self.interactive_plots_dir.mkdir(parents=True, exist_ok=True)
        (self.interactive_plots_dir / "polar").mkdir(exist_ok=True)

    def _configure_interactive_axis(self, fig):
        """Helper to apply custom ticks to plotly figure x-axis"""
        # Calculate ticks
        all_ticks = sorted(list(set(config.GRID_FREQS_MAJOR + config.GRID_FREQS_MINOR)))
        valid_ticks = [f for f in all_ticks if config.FREQ_MIN <= f <= config.FREQ_MAX]

        def format_freq(x):
            if x >= 1000:
                val = x / 1000
                return f'{int(val)}k' if val.is_integer() else f'{val}k'
            return str(int(x))

        tick_text = [format_freq(f) for f in valid_ticks]

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
        Plotly.newPlot('plot', figData.data, figData.layout, {{responsive: true}});
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

            # Decimate to TARGET_POINTS using log-spaced indices
            n_points = len(freq)
            if n_points > TARGET_POINTS:
                # Create log-spaced indices
                log_indices = np.unique(np.logspace(0, np.log10(n_points - 1), TARGET_POINTS).astype(int))
                freq_dec = freq[log_indices]
                spl_dec = spl_matrix[log_indices, :]
            else:
                freq_dec = freq
                spl_dec = spl_matrix

            all_data[driver] = {
                'freq': freq_dec.tolist(),
                'angles': [int(a) for a in angles],
                'spl': {int(angles[i]): spl_dec[:, i].tolist() for i in range(len(angles))}
            }
            all_angles.update(angles)

        all_angles = sorted(all_angles)

        # Build the HTML with embedded JavaScript
        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Frequency Response Explorer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/js-yaml@4.1.0/dist/js-yaml.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
        }}
        .container {{
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
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>Frequency Response Explorer</h2>

            <h3>Drivers (click to toggle)</h3>
            <div class="driver-list" id="driverList"></div>

            <h3>Angles</h3>
            <div class="quick-select">
                <button class="quick-btn" onclick="selectAngles([0])">0° only</button>
                <button class="quick-btn" onclick="selectAngles([0,30,60,90])">0/30/60/90</button>
                <button class="quick-btn" onclick="selectAllAngles()">All</button>
                <button class="quick-btn" onclick="selectAngles([])">None</button>
            </div>
            <div class="angle-grid" id="angleGrid"></div>

            <div class="filter-section">
                <h3>Filters (IIR EQ)</h3>
                <div class="filter-controls">
                    <button class="primary" onclick="addFilter()">+ Add Filter</button>
                    <button onclick="document.getElementById('yamlInput').click()">Load YAML</button>
                    <button onclick="saveFiltersYaml()">Save YAML</button>
                </div>
                <input type="file" id="yamlInput" accept=".yaml,.yml" onchange="loadYamlFile(event)">
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
                            <input type="number" id="optFreqMin" value="200" min="20" max="20000"> Hz to
                            <input type="number" id="optFreqMax" value="10000" min="20" max="20000"> Hz
                        </div>
                        <div class="opt-param">
                            <label>Reference Angle:</label>
                            <select id="optRefAngle"></select>
                        </div>
                        <div class="opt-buttons">
                            <button id="optimizeBtn" class="primary" onclick="runOptimization()">Optimize Selected</button>
                            <button id="undoOptBtn" onclick="undoOptimization()" style="display:none;">Undo</button>
                        </div>
                        <div id="optStatus" class="opt-status">Ready</div>
                    </div>
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

        // Line style patterns for distinguishing drivers
        const dashPatterns = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot'];
        const dashLabels = ['———', '– – –', '· · ·', '– · –', '—  —', '—  ·'];

        // Filter types
        const filterTypes = ['Peaking', 'Lowpass', 'Highpass', 'Lowshelf', 'Highshelf'];
        const filterNeedsGain = {{'Peaking': true, 'Lowshelf': true, 'Highshelf': true, 'Lowpass': false, 'Highpass': false}};

        // State
        let activeDrivers = new Set([drivers[0]]);
        let activeAngles = new Set([0]);
        let driverOffsets = {{}};
        drivers.forEach(d => driverOffsets[d] = 0);

        // Filter state
        let driverFilters = {{}};
        drivers.forEach(d => driverFilters[d] = []);
        let selectedFilterDriver = drivers[0];
        const STORAGE_KEY = 'lx521_filters_' + measurementSetName;

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

        function calcFilterChainResponse(filters, frequencies) {{
            return frequencies.map(freq => {{
                let totalDb = 0;
                filters.forEach(filter => {{
                    if (!filter.enabled) return;
                    const coeffs = calcBiquadCoeffs(filter.type, filter.freq, filter.q, filter.gain || 0);
                    totalDb += biquadMagnitudeDb(coeffs, freq);
                }});
                return totalDb;
            }});
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
            localStorage.setItem(STORAGE_KEY, JSON.stringify(driverFilters));
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

        // Initialize UI
        function initUI() {{
            const driverList = document.getElementById('driverList');
            drivers.forEach((driver, idx) => {{
                const color = driverColors[driver] || defaultColor;
                const dashLabel = dashLabels[idx % dashLabels.length];
                const isActive = activeDrivers.has(driver);

                const item = document.createElement('div');
                item.className = 'driver-item' + (isActive ? ' active' : '');
                item.style.setProperty('--driver-color', color);
                item.dataset.driver = driver;

                item.innerHTML = `
                    <div class="driver-header">
                        <input type="checkbox" class="driver-checkbox" ${{isActive ? 'checked' : ''}}>
                        <span class="driver-line">${{dashLabel}}</span>
                        <span class="driver-name">${{driver}}</span>
                    </div>
                    <div class="offset-control">
                        <span>Offset:</span>
                        <input type="range" min="-20" max="20" value="0">
                        <input type="number" value="0" min="-20" max="20">
                        <span>dB</span>
                    </div>
                `;

                // Header click toggles driver
                const header = item.querySelector('.driver-header');
                const checkbox = item.querySelector('.driver-checkbox');
                header.onclick = () => {{
                    checkbox.checked = !checkbox.checked;
                    toggleDriver(driver, item, checkbox.checked);
                }};

                // Offset controls
                const slider = item.querySelector('input[type="range"]');
                const numInput = item.querySelector('input[type="number"]');
                slider.oninput = () => {{
                    numInput.value = slider.value;
                }};
                slider.onchange = () => {{
                    setOffset(driver, slider.value);
                }};
                numInput.onchange = () => {{
                    slider.value = numInput.value;
                    setOffset(driver, numInput.value);
                }};

                driverList.appendChild(item);
            }});

            const angleGrid = document.getElementById('angleGrid');
            allAngles.forEach(angle => {{
                const btn = document.createElement('button');
                btn.className = 'angle-btn' + (activeAngles.has(angle) ? ' active' : '');
                btn.textContent = angle + '°';
                btn.onclick = () => toggleAngle(angle, btn);
                angleGrid.appendChild(btn);
            }});
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

        // ============ FILTER UI FUNCTIONS ============
        function initFilterUI() {{
            // Populate driver select
            const select = document.getElementById('filterDriverSelect');
            drivers.forEach(d => {{
                const opt = document.createElement('option');
                opt.value = d;
                opt.textContent = d;
                select.appendChild(opt);
            }});
            select.value = selectedFilterDriver;

            // Populate reference angle select for optimization
            const refAngleSelect = document.getElementById('optRefAngle');
            allAngles.forEach(a => {{
                const opt = document.createElement('option');
                opt.value = a;
                opt.textContent = a + '°';
                refAngleSelect.appendChild(opt);
            }});
            refAngleSelect.value = 0;  // Default to on-axis

            renderFilterList();
        }}

        function selectFilterDriver(driver) {{
            selectedFilterDriver = driver;
            renderFilterList();
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
                card.className = 'filter-card' + (filter.enabled ? '' : ' disabled');

                const needsGain = filterNeedsGain[filter.type];

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

                    for (const [name, def] of Object.entries(yaml.filters || {{}})) {{
                        if (def.type === 'Biquad' && def.parameters) {{
                            filters.push({{
                                type: def.parameters.type || 'Peaking',
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
                        saveFiltersToLocalStorage();
                        renderFilterList();
                        updatePlot();
                        alert(`Loaded ${{filters.length}} filter(s) for ${{selectedFilterDriver}}`);
                    }} else {{
                        alert('No valid biquad filters found in YAML');
                    }}
                }} catch (err) {{
                    alert('Error parsing YAML: ' + err.message);
                }}
            }};
            reader.readAsText(file);
            event.target.value = '';  // Reset input
        }}

        function saveFiltersYaml() {{
            const filters = driverFilters[selectedFilterDriver] || [];
            if (filters.length === 0) {{
                alert('No filters to save for ' + selectedFilterDriver);
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

        function updatePlot() {{
            const traces = [];

            // Line dash patterns for different drivers (to distinguish when overlaid)
            const dashPatterns = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot'];

            activeDrivers.forEach(driver => {{
                const driverIdx = drivers.indexOf(driver);
                const color = driverColors[driver] || defaultColor;
                const dashBase = dashPatterns[driverIdx % dashPatterns.length];
                const offset = driverOffsets[driver];
                const data = allData[driver];
                const filters = driverFilters[driver] || [];
                const hasActiveFilters = filters.some(f => f.enabled);

                // Calculate filter response if there are active filters
                let filterResponse = null;
                if (hasActiveFilters) {{
                    filterResponse = calcFilterChainResponse(filters, data.freq);
                }}

                activeAngles.forEach(angle => {{
                    if (!data.spl[angle]) return;

                    const splOriginal = data.spl[angle].map(v => v + offset);
                    const angleBrightness = 1 - (angle / 90) * 0.4;

                    if (hasActiveFilters) {{
                        // Show original trace (dimmed, dotted)
                        traces.push({{
                            x: data.freq,
                            y: splOriginal,
                            name: `${{driver}} @ ${{angle}}° (orig)`,
                            mode: 'lines',
                            line: {{
                                color: color,
                                width: 1,
                                dash: 'dot'
                            }},
                            opacity: 0.3,
                            showlegend: false
                        }});

                        // Show filtered trace (bold)
                        const splFiltered = splOriginal.map((v, i) => v + filterResponse[i]);
                        traces.push({{
                            x: data.freq,
                            y: splFiltered,
                            name: `${{driver}} @ ${{angle}}° (EQ)`,
                            mode: 'lines',
                            line: {{
                                color: color,
                                width: angle === 0 ? 2.5 : 1.5,
                                dash: dashBase
                            }},
                            opacity: angleBrightness
                        }});
                    }} else {{
                        // No filters - show original trace only
                        traces.push({{
                            x: data.freq,
                            y: splOriginal,
                            name: `${{driver}} @ ${{angle}}°` + (offset !== 0 ? ` (${{offset > 0 ? '+' : ''}}${{offset}}dB)` : ''),
                            mode: 'lines',
                            line: {{
                                color: color,
                                width: angle === 0 ? 2.5 : 1.5,
                                dash: dashBase
                            }},
                            opacity: angleBrightness
                        }});
                    }}
                }});
            }});

            // Add grid lines
            const shapes = [];
            // Vertical frequency grid
            [100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000].forEach(f => {{
                shapes.push({{
                    type: 'line', x0: f, x1: f, y0: 0, y1: 1, yref: 'paper',
                    line: {{ color: f === 1000 || f === 10000 ? '#888' : '#ddd', width: 1, dash: 'dot' }}
                }});
            }});

            // Get Y range from data
            let yMin = 100, yMax = 0;
            traces.forEach(t => {{
                t.y.forEach(v => {{
                    if (v < yMin) yMin = v;
                    if (v > yMax) yMax = v;
                }});
            }});
            yMin = Math.floor(yMin / 5) * 5 - 5;
            yMax = Math.ceil(yMax / 5) * 5 + 5;

            // Horizontal dB grid
            for (let db = yMin; db <= yMax; db += 5) {{
                shapes.push({{
                    type: 'line', x0: 0, x1: 1, xref: 'paper', y0: db, y1: db,
                    line: {{ color: db % 10 === 0 ? '#888' : '#ddd', width: db % 10 === 0 ? 1 : 0.5, dash: db % 10 === 0 ? 'solid' : 'dot' }}
                }});
            }}

            const layout = {{
                title: '<b>Frequency Response Explorer</b>',
                xaxis: {{
                    title: 'Frequency (Hz)',
                    type: 'log',
                    range: [Math.log10(100), Math.log10(20000)],
                    gridcolor: 'transparent'
                }},
                yaxis: {{
                    title: 'SPL (dB)',
                    range: [yMin, yMax],
                    gridcolor: 'transparent'
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
        initUI();
        initFilterUI();
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
        all_ticks = sorted(list(set(config.GRID_FREQS_MAJOR + config.GRID_FREQS_MINOR)))
        valid_ticks = [f for f in all_ticks if config.FREQ_MIN <= f <= config.FREQ_MAX]
        
        # Format labels: 100, 200.. 1k, 1.1k etc.
        def format_freq(x):
            if x >= 1000:
                val = x / 1000
                return f'{int(val)}k' if val.is_integer() else f'{val}k'
            return str(int(x))

        # Enforce ticks using FixedLocator/FixedFormatter to override LogScale defaults
        ax.xaxis.set_major_locator(ticker.FixedLocator(valid_ticks))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter([format_freq(f) for f in valid_ticks]))
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
        steps = []
        for i, idx in enumerate(indices):
            f = freqs[idx]
            step = dict(
                method="restyle",
                args=[{
                    "r": [all_step_data[i][d]['r'] for d in range(len(self.drivers))],
                    "theta": [all_step_data[i][d]['theta'] for d in range(len(self.drivers))]
                }],
                label=f"{f:.0f}"
            )
            steps.append(step)

        # Create driver selection buttons (updatemenus)
        # "All" button shows all drivers, individual buttons show only that driver
        driver_buttons = []

        # "All" button
        driver_buttons.append(dict(
            label="All",
            method="restyle",
            args=[{"visible": [True] * len(self.drivers)}]
        ))

        # Individual driver buttons
        for d_idx, driver in enumerate(self.drivers):
            visibility = [False] * len(self.drivers)
            visibility[d_idx] = True
            driver_buttons.append(dict(
                label=driver,
                method="restyle",
                args=[{"visible": visibility}]
            ))

        # Configure angular axis based on whether we have 360° data
        if any_has_rear:
            angular_axis = dict(
                direction="clockwise",
                rotation=90,
                gridcolor='lightgray',
                tickmode="array",
                tickvals=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
                ticktext=["<b>0°</b>", "30°", "60°", "90°", "120°", "150°",
                          "180°", "210°", "240°", "270°", "300°", "330°"]
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
            polar=dict(
                bgcolor='white',
                radialaxis=dict(
                    range=[limit_min, limit_max],
                    visible=True,
                    showline=True,
                    gridcolor='lightgray',
                    showticklabels=True
                ),
                angularaxis=angular_axis
            ),
            updatemenus=[{
                "buttons": driver_buttons,
                "direction": "down",
                "showactive": True,
                "x": 1.15,
                "xanchor": "left",
                "y": 0.8,
                "yanchor": "top",
                "bgcolor": "white",
                "bordercolor": "lightgray",
                "font": {"size": 12}
            }],
            annotations=[{
                "text": "Driver:",
                "x": 1.15,
                "xref": "paper",
                "y": 0.85,
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14, "color": "black"}
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Frequency: ",
                    "suffix": " Hz",
                    "visible": True,
                    "xanchor": "center"
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.05,
                "y": 0,
                "steps": steps
            }],
            legend=dict(
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.05,
                font=dict(size=14),
                bgcolor="rgba(255,255,255,0.5)"
            ),
            margin=dict(l=60, r=150, t=80, b=100),
            paper_bgcolor="white"
        )

        fig = go.Figure(data=initial_traces, layout=layout)

        # Generate HTML with custom frequency input
        html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)

        # Build frequency lookup array for JavaScript
        freq_values = [freqs[idx] for idx in indices]
        freq_js_array = ','.join([f'{f:.1f}' for f in freq_values])

        # Custom JavaScript for manual frequency entry
        custom_js = f'''
<style>
.freq-input-container {{
    position: absolute;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 1000;
    background: white;
    padding: 8px 15px;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    font-family: Arial, sans-serif;
}}
.freq-input-container label {{
    font-size: 14px;
    margin-right: 8px;
}}
.freq-input-container input {{
    width: 80px;
    padding: 5px 8px;
    font-size: 14px;
    border: 1px solid #ccc;
    border-radius: 4px;
}}
.freq-input-container button {{
    padding: 5px 12px;
    font-size: 14px;
    margin-left: 5px;
    cursor: pointer;
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 4px;
}}
.freq-input-container button:hover {{
    background: #1d4ed8;
}}
.freq-input-container span {{
    margin-left: 10px;
    font-size: 12px;
    color: #666;
}}
</style>
<div class="freq-input-container">
    <label for="freqInput">Go to frequency:</label>
    <input type="number" id="freqInput" min="{config.FREQ_MIN}" max="{config.FREQ_MAX}" placeholder="Hz">
    <button onclick="goToFrequency()">Go</button>
    <span>(Range: {config.FREQ_MIN}-{config.FREQ_MAX} Hz)</span>
</div>
<script>
var freqValues = [{freq_js_array}];

function goToFrequency() {{
    var targetFreq = parseFloat(document.getElementById('freqInput').value);
    if (isNaN(targetFreq) || targetFreq < {config.FREQ_MIN} || targetFreq > {config.FREQ_MAX}) {{
        alert('Please enter a frequency between {config.FREQ_MIN} and {config.FREQ_MAX} Hz');
        return;
    }}

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

    // Update slider
    var plotDiv = document.querySelector('.plotly-graph-div');
    if (plotDiv) {{
        Plotly.relayout(plotDiv, {{'sliders[0].active': closestIdx}});
        // Also trigger the restyle to update the plot
        var slider = plotDiv.layout.sliders[0];
        if (slider && slider.steps && slider.steps[closestIdx]) {{
            var step = slider.steps[closestIdx];
            Plotly.restyle(plotDiv, step.args[0]);
        }}
    }}
}}

// Also allow Enter key to trigger search
document.getElementById('freqInput').addEventListener('keypress', function(e) {{
    if (e.key === 'Enter') {{
        goToFrequency();
    }}
}});
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
