"""
Configuration settings for LX521 Polar Analysis
"""

from pathlib import Path

# Paths
OUTPUT_DIR = Path("output")
DATA_DIR = OUTPUT_DIR / "data"

# Measurement Set Configurations
MEASUREMENT_SETS = {
    "andres": {
        "path": Path("../Mediciones Andres"),
        "pattern_type": "andres",  # F{angle}-{driver}.mdat
        "angles": list(range(0, 91, 10)),
        "has_rear": False,
        "hdf5_file": "polar_data_andres.h5",
        "output_dir": OUTPUT_DIR / "andres",
    },
    "juan": {
        "path": Path("../Mediciones Juan/GRS PT6816 A MIC ON AXIS"),
        "pattern_type": "juan",  # {driver} {angle} {side}.mdat
        "angles": [0, 15, 30, 45, 60, 75, 90],
        "has_rear": True,
        "hdf5_file": "polar_data_juan.h5",
        "output_dir": OUTPUT_DIR / "juan",
    },
}

DEFAULT_MEASUREMENT_SET = "andres"

# Legacy compatibility (for default measurement set)
MDAT_DIR = MEASUREMENT_SETS[DEFAULT_MEASUREMENT_SET]["path"]
HDF5_FILE_NAME = MEASUREMENT_SETS[DEFAULT_MEASUREMENT_SET]["hdf5_file"]
HDF5_FILE_PATH = DATA_DIR / HDF5_FILE_NAME
STATIC_PLOTS_DIR = MEASUREMENT_SETS[DEFAULT_MEASUREMENT_SET]["output_dir"] / "static_plots"
INTERACTIVE_PLOTS_DIR = MEASUREMENT_SETS[DEFAULT_MEASUREMENT_SET]["output_dir"] / "interactive"

# REW API
REW_API_BASE = "http://127.0.0.1:4735"
REW_TIMEOUT = 30

# Analysis Parameters
DEFAULT_SMOOTHING = 0  # No smoothing by default (use 12 for 1/12th octave)
GATE_LEFT_MS = 0.5
GATE_RIGHT_MS = 3.0
SAMPLE_RATE = 48000  # Default, will be updated from measurement



# Driver Definitions
DRIVERS = ['10F8824', 'L22MG', 'MU10', 'SEAS27T']

DRIVER_COLORS = {
    '10F8824': '#1f77b4',  # Blue - Woofer
    'L22MG': '#ff7f0e',    # Orange - Lower Mid
    'MU10': '#2ca02c',     # Green - Upper Mid
    'SEAS27T': '#d62728'   # Red - Tweeter
}

# Crossover Frequencies (Hz)
CROSSOVER_FREQUENCIES = [120, 1000, 7000]

# Visualization Settings
FREQ_MIN = 100
FREQ_MAX = 20000
GRID_FREQS_MAJOR = [1000, 10000]
GRID_FREQS_MINOR = (
    list(range(100, 1000, 100)) + 
    list(range(1000, 10000, 1000)) + 
    list(range(10000, 21000, 1000))
)

# Plotting
FIG_SIZE_STATIC = (12, 6)
DPI_STATIC = 300
