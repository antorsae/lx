"""
Configuration settings for LX521 Polar Analysis
"""

import os
from pathlib import Path

# Paths
OUTPUT_DIR = Path("output")
DATA_DIR = OUTPUT_DIR / "data"

# Kaspar Reili's outdoor LX521 captures share one measurement condition; only
# the DSP channel gains differ between the three system profiles.
KASPAR_COMMON_NOTE = (
    "Kaspar Reili LX521, outdoors, 1 m on-axis, 18-Oct-2022. UMIK-1 (18 dB gain, "
    "cal 7057283) with REW acoustic timing reference, ungated. Measured downstream "
    "of the active crossover/EQ: acoustic channel outputs, not raw drivers."
)

KASPAR_DRIVER_NOTES = {
    "LM L22MG (Kaspar DSP)": "Lower midrange alone, DSP channel gains at 0.",
    "UM MU10RB (Kaspar DSP)": "Upper midrange alone, DSP channel gains at 0.",
    "Tweeter 27TFFNC (Kaspar DSP)": "Tweeter alone, DSP channel gains at 0.",
    "System P1 (Kaspar DSP)": "All drivers, gain profile 1: Woofer -4.8, LM -4.6, UM +2, HI -2 dB.",
    "System P2 (Kaspar DSP)": "All drivers, gain profile 2: Woofer -4.8, LM -6.2, UM +2, HI -2 dB.",
    "System P3 (Kaspar DSP)": "All drivers, gain profile 3: Woofer -4.8, LM -6.2, UM -2.8, HI 0 dB.",
}

# Measurement Set Configurations
MEASUREMENT_SETS = {
    "andres": {
        "path": Path("measurements/andres"),
        "pattern_type": "andres",  # F{angle}-{driver}.mdat
        "angles": list(range(0, 91, 10)),
        "has_rear": False,
        # Keep validation in parity with the already-published polar explorer.
        # First-lobe/direct-gate/strongest-lobe variants are diagnostics unless
        # the explorer is regenerated from the same HDF5 in the same commit.
        "hdf5_file": "polar_data_andres_early_peak_legacy.h5",
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
        "direct_ir_peak_policy": "ir-start",
        "measurement_metadata_overrides": {
            "L22MG (nude)": {
                "measurement_distance_m": 0.50,
                "measurement_height_reference": "l22mg",
                "notes": "Measurement distance: 50 cm from driver. Mic height: L22MG/LM.",
            },
        },
    },
    "juan-lx521-top-raw": {
        # Raw/no-crossover driver captures mounted in the LX521 top baffle.
        # The source files intentionally reuse names such as "SEAS L22MG A";
        # source-local aliases keep these mounted measurements distinct from
        # the naked/baffleless source captures.
        "path": None,
        "sources": [
            {
                "path": Path("measurements/juan/SEAS L22MG EN TOP BAFFLE LX521"),
                "pattern_type": "juan",
                "driver_name_aliases": {
                    "SEAS L22MG A": "L22MG (LX521 top raw)",
                },
            },
            {
                "path": Path("measurements/juan/SEAS L10NEO EN TOP BAFFLE LX521"),
                "pattern_type": "juan",
                "driver_name_aliases": {
                    "SEAS L10NEO A": "L10NEO (LX521 top raw)",
                },
            },
            {
                "path": Path(
                    "measurements/juan/SEAS L22MG + L10NEO + TWEETERS NO XOVER EN TOP BAFFLE LX521"
                ),
                "pattern_type": "juan",
                "driver_name_aliases": {
                    "SEAS L22MG A": "L22MG+L10NEO+Tweeters (LX521 top raw)",
                },
            },
        ],
        "pattern_type": "juan",
        "angles": [0, 15, 30, 45, 60, 75, 90],
        "has_rear": True,
        "hdf5_file": "polar_data_juan_lx521_top_raw.h5",
        "output_dir": OUTPUT_DIR / "juan-lx521-top-raw",
        # Match REW's stored IR-start window reference for the mounted raw
        # Juan top-baffle driver captures.
        "direct_ir_peak_policy": "ir-start",
        "measurement_metadata_overrides": {
            "L22MG (LX521 top raw)": {
                "measurement_distance_m": 0.50,
                "measurement_height_reference": "l22mg",
                "notes": (
                    "Measurement distance: 50 cm. Mic height: L22MG/LM. "
                    "LX521 top baffle mounted; raw/no crossover/no EQ."
                ),
            },
            "L10NEO (LX521 top raw)": {
                "measurement_distance_m": 0.50,
                "measurement_height_reference": "l22mg",
                "notes": (
                    "Measurement distance: 50 cm. Mic height: L22MG/LM. "
                    "LX521 top baffle mounted; raw/no crossover/no EQ."
                ),
            },
            "L22MG+L10NEO+Tweeters (LX521 top raw)": {
                "measurement_distance_m": 0.50,
                "measurement_height_reference": "l22mg",
                "notes": (
                    "Measurement distance: 50 cm. Mic height: L22MG. "
                    "L22MG + L10NEO + tweeters on LX521 top baffle; raw/no crossover/no EQ."
                ),
            },
        },
    },
    "kaspar": {
        # Kaspar Reili's LX521, measured outdoors at 1 m on-axis, 18 Oct 2022.
        # Split from a single 6-measurement .mdat into F0-{driver}.mdat files.
        # Per-driver captures are taken downstream of the active crossover/EQ,
        # so they are acoustic channel outputs, not raw drivers.
        "path": Path("measurements/kaspar"),
        "pattern_type": "kaspar",  # F{angle}-{driver}.mdat
        "angles": [0],
        "has_rear": False,
        # On-axis only: no polar/DI/beamwidth/contour output is meaningful here.
        "single_angle": True,
        # Outdoor capture with the first reflection ~6 ms after the direct
        # arrival and >=20 dB down. The default 3 ms gate would discard
        # everything below ~333 Hz, which is most of what these captures show,
        # so publish them ungated.
        "gate_left_ms": 0.0,
        "gate_right_ms": 0.0,
        # Ungated means full 0.37 Hz resolution with deep comb structure. The
        # explorer decimates to ~500 log-spaced points, and picking single bins
        # out of that aliases by up to ~10 dB, so smooth before decimating.
        "smoothing": 12,
        "hdf5_file": "polar_data_kaspar.h5",
        "output_dir": OUTPUT_DIR / "kaspar",
        "direct_ir_peak_policy": "ir-start",
        "measurement_metadata_overrides": {
            name: {
                "measurement_distance_m": 1.00,
                "notes": f"{KASPAR_COMMON_NOTE} {detail}",
            }
            for name, detail in KASPAR_DRIVER_NOTES.items()
        },
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

# Direct-arrival IR selection for regenerated HDF5s. Juan's USB/no-timing-ref
# captures are not absolute-time measurements: REW can peak-reference a later
# stronger reflection to 0 ms at high angle. Use REW's stored IR-start/onset as
# the default gate reference; first-strong/strongest are legacy diagnostics.
DIRECT_IR_PEAK_POLICY = os.environ.get("DIRECT_IR_PEAK_POLICY", "ir-start")
ALLOW_UNSAFE_STRONGEST_IR_PEAK_POLICY = os.environ.get(
    "ALLOW_UNSAFE_STRONGEST_IR_PEAK_POLICY",
    "",
).strip().lower() in {"1", "true", "yes"}
DIRECT_IR_FIRST_LOBE_THRESHOLD_FRACTION = float(
    os.environ.get("DIRECT_IR_FIRST_LOBE_THRESHOLD_FRACTION", "0.50")
)
DIRECT_IR_FIRST_LOBE_START_MS = float(os.environ.get("DIRECT_IR_FIRST_LOBE_START_MS", "-0.50"))
DIRECT_IR_FIRST_LOBE_END_MS = float(os.environ.get("DIRECT_IR_FIRST_LOBE_END_MS", "0.80"))



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
    'L22MG (LX521 top raw)': '#ff7f0e',  # Orange - mounted raw
    'L10NEO (LX521 top raw)': '#8c564b',  # Brown - mounted raw
    'L22MG+L10NEO+Tweeters (LX521 top raw)': '#111827',  # Black - mounted raw stack
    'LM L22MG (Kaspar DSP)': '#ff7f0e',        # Orange - Kaspar lower mid
    'UM MU10RB (Kaspar DSP)': '#2ca02c',       # Green - Kaspar upper mid
    'Tweeter 27TFFNC (Kaspar DSP)': '#d62728',  # Red - Kaspar tweeter
    'System P1 (Kaspar DSP)': '#111827',       # Black - Kaspar system profile 1
    'System P2 (Kaspar DSP)': '#6b7280',       # Gray - Kaspar system profile 2
    'System P3 (Kaspar DSP)': '#9467bd',       # Purple - Kaspar system profile 3
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
