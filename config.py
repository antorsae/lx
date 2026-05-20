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
        "path": Path("measurements/andres"),
        "pattern_type": "andres",  # F{angle}-{driver}.mdat
        "angles": list(range(0, 91, 10)),
        "has_rear": False,
        "hdf5_file": "polar_data_andres.h5",
        "output_dir": OUTPUT_DIR / "andres",
    },
    "juan-baffleless": {
        # Combined Juan driver measurements.
        # This set merges data from multiple source directories
        "path": None,  # Not used - see 'sources' below
        "sources": [
            {"path": Path("measurements/juan/GRS PT6816 A MIC ON AXIS"), "pattern_type": "juan"},
            {"path": Path("measurements/juan/ScanSpeak 10F8414G10"), "pattern_type": "scanspeak"},
            {"path": Path("measurements/juan/SEAS L22MG NUDE MIC ON AXIS"), "pattern_type": "juan"},
            {"path": Path("measurements/juan/DAYTON ND25FW4 ANIDADOS 18 MM NUDE"), "pattern_type": "juan"},
            {"path": Path("measurements/juan/POLARES L10NEO"), "pattern_type": "juan"},
            {"path": Path("measurements/juan/SEAS MU10RB SL POLARES"), "pattern_type": "juan"},
            {"path": Path("measurements/juan/ScanSpeak 10F8424G00"), "pattern_type": "juan_suffix"},
            {"path": Path("measurements/juan/L26RO4Y POLARES EN BAFFLE CILINDRICO VER NOTA"), "pattern_type": "juan"},
        ],
        "pattern_type": "juan",  # Default pattern (not used when sources defined)
        "angles": [0, 15, 30, 45, 60, 75, 90],
        "has_rear": True,
        "hdf5_file": "polar_data_juan_baffleless.h5",
        "output_dir": OUTPUT_DIR / "juan-baffleless",
    },
    "lx521-system": {
        "path": Path("measurements/juan/LX521 POLARES 0_180 GRADOS"),
        "pattern_type": "lx521_system",  # {name} {angle} GRADOS {F|REAR}.mdat
        "angles": [0, 15, 30, 45, 60, 75, 90],
        "has_rear": True,
        "hdf5_file": "polar_data_lx521_system.h5",
        "output_dir": OUTPUT_DIR / "lx521-system",
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
DRIVERS = ['10F8424', 'L22MG', 'MU10', 'SEAS27T']

DRIVER_COLORS = {
    '10F8424': '#1f77b4',  # Blue - Woofer
    'L22MG': '#ff7f0e',    # Orange - Lower Mid
    'MU10': '#2ca02c',     # Green - Upper Mid
    'SEAS27T': '#d62728',  # Red - Tweeter
    'GRS PT6816': '#9467bd',     # Purple - Juan baffleless
    'SS10F8414G10': '#17becf',   # Cyan - Juan baffleless
    'L22MG (nude)': '#e377c2',   # Pink - Juan baffleless
    'ND25FW4 (nude 18mm)': '#bcbd22',  # Olive - Juan baffleless
    'L10NEO': '#8c564b',          # Brown - Juan baffleless
    'MU10RB-SL': '#7f7f7f',       # Gray - Juan baffleless
    'SS10F8424G00': '#1f77b4',    # Blue - Juan baffleless
    'L26RO4Y': '#ff9896',          # Light red - Juan measurements
}

DRIVER_NAME_ALIASES = {
    "SEAS L22MG A": "L22MG (nude)",
    "ND25 NEST 18 MM": "ND25FW4 (nude 18mm)",
    "SEAS L10NEO A": "L10NEO",
    "SEAS MU10RBSL A": "MU10RB-SL",
    "SEAS MU10RBSLA": "MU10RB-SL",
    "SEAS L26RO4Y": "L26RO4Y",
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
