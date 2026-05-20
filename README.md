# LX521 Polar Analysis Pipeline

A Python pipeline for processing acoustic polar response measurements from REW (Room EQ Wizard) and generating comprehensive visualizations for speaker driver analysis.

**Live Demo:** [https://antorsae.github.io/lx/](https://antorsae.github.io/lx/)

## Features

- Load and process `.mdat` measurement files via REW API
- Apply time gating to remove room reflections
- Support for multiple measurement sets with different naming conventions
- Full 360° polar plots when front and rear measurements are available
- Generate directivity analysis (DI, beamwidth)
- Interactive HTML plots (gzip compressed) and static PNG exports
- Interactive frequency response explorer with IIR EQ, per-driver gain/delay/invert, SUM view, and Hypex Filter Design `.dsp` + CamillaDSP YAML I/O (Config.xml conversion via external tools)
- Crossover match analysis for multi-driver systems
- **REW slot management**: Automatic batch unloading prevents hitting REW's ~100 measurement limit

## Requirements

- Python 3.11+
- [REW (Room EQ Wizard)](https://www.roomeqwizard.com/) with API enabled
- macOS (for automatic REW launch)

## Installation

```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Configuration

Edit `config.py` to configure measurement sets:

```python
MEASUREMENT_SETS = {
    "andres": {
        "path": Path("measurements/andres"),
        "pattern_type": "andres",  # F{angle}-{driver}.mdat
        "has_rear": False,
        "hdf5_file": "polar_data_andres.h5",
        "output_dir": OUTPUT_DIR / "andres",
    },
    "juan-baffleless": {
        # Combined Juan driver measurements (GRS PT6816, SS10F8414G10,
        # L22MG nude, ND25FW4 nude 18mm, L10NEO, MU10RB-SL,
        # SS10F8424G00, L26RO4Y).
        # Multi-source sets merge directories.
        "path": None,
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
        "pattern_type": "juan",  # default (ignored when sources defined)
        "has_rear": True,
        "hdf5_file": "polar_data_juan_baffleless.h5",
        "output_dir": OUTPUT_DIR / "juan-baffleless",
    },
    "lx521-system": {
        "path": Path("measurements/juan/LX521 POLARES 0_180 GRADOS"),
        "pattern_type": "lx521_system",  # {name} {angle} GRADOS {F|REAR}.mdat
        "has_rear": True,
        "hdf5_file": "polar_data_lx521_system.h5",
        "output_dir": OUTPUT_DIR / "lx521-system",
    },
}
```

### Supported Naming Conventions

| Pattern | Example | Description |
|---------|---------|-------------|
| `andres` | `F45-10F8424.mdat` | Front-only measurements |
| `juan` | `GRS PT6816 45 F.mdat` | Front (F) and Rear (R) measurements |
| `scanspeak` | `SS10F8414G10 45 F.mdat` | Same convention as `juan` |
| `juan_suffix` | `SS10F8424G00 45 F sn 074.mdat` | Same as `juan`, allowing text after `F`/`R` |
| `lx521_system` | `LX521 HIGH MID INV ORIGINAL 45 GRADOS F.mdat` | Full system measurements with GRADOS notation |

Measurement sources may be nested below each configured source directory. This supports source layouts such as `POLARES L10NEO/FRONTALES` and `POLARES L10NEO/TRASERAS`.

## Usage

### Enable REW API

1. Open REW
2. Go to Preferences → API
3. Enable the API server (default port: 4735)

### Run Pipeline

```bash
# Process default measurement set (andres)
python run_pipeline.py

# Process specific measurement set
python run_pipeline.py -m juan-baffleless

# Skip data loading (use existing HDF5)
python run_pipeline.py -m juan-baffleless --skip-loading

# Skip visualization generation
python run_pipeline.py --skip-viz

# Disable smoothing (raw data)
python run_pipeline.py --no-smoothing
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `-m, --measurement-set` | Which measurement set to process (default: andres) |
| `--skip-loading` | Skip REW loading, use existing HDF5 file |
| `--skip-viz` | Skip visualization generation |
| `--no-smoothing` | Disable frequency response smoothing (default: no smoothing) |

### Makefile Automation

```bash
make all              # Full rebuild: data + viz + sync
make data             # Load REW data → HDF5 (skips when HDF5 is newer than .mdat)
make viz              # Regenerate all visualizations (uses existing HDF5)
make viz-andres       # Regenerate only andres set
make sync             # Sync output/ to docs/ + regenerate landing pages
make deploy           # sync + commit + push to GitHub Pages
make help             # Show all targets
```

`make all` now runs `viz` and `sync` in parallel (default `JOBS=8`). Override with `JOBS=4 make all` or `make -j16 all`.

## Output Structure

```
output/
├── data/
│   ├── polar_data_andres.h5          # Processed data (andres set)
│   ├── polar_data_juan_baffleless.h5 # Processed data (juan-baffleless set)
│   └── polar_data_lx521_system.h5    # Processed data (lx521-system set)
├── andres/                            # Visualizations for andres set
│   ├── static_plots/
│   │   ├── core/                     # DI, beamwidth, contour plots
│   │   └── polar/                    # Polar diagrams
│   └── interactive/                  # HTML interactive plots (gzip compressed)
├── juan-baffleless/                   # Visualizations for juan-baffleless set
│   ├── static_plots/
│   └── interactive/
└── lx521-system/                      # Visualizations for lx521-system set
    ├── static_plots/
    └── interactive/
```

## Pipeline Workflow

1. **Connect to REW API** - Launches REW if not running
2. **Load Measurements** - Reads `.mdat` files for each driver/angle
3. **Auto-fix Timing** - Aligns impulse response peak to t=0
4. **Apply Time Gating** - Removes room reflections (default: 0.5ms / 3.0ms)
5. **Get Frequency Response** - Retrieves magnitude/phase data
6. **Unload from REW** - Frees REW memory slots after each driver (batch unload)
7. **Save to HDF5** - Stores processed data for visualization
8. **Generate Visualizations** - Creates all plots and analysis

## REW Slot Management

REW has a limit of ~100 measurement slots. When processing many drivers, the pipeline automatically unloads measurements after each driver to prevent hitting this limit.

```python
# Manual control (if needed)
from polar_data_loader import PolarDataLoader

loader = PolarDataLoader(data_directory="path/to/data")

# Check current count
print(f"Loaded: {loader.get_measurement_count()}")

# Unload specific measurement
loader.unload_measurement(uuid)

# Unload all measurements
loader.unload_all_measurements()

# Disable batch unload (not recommended for large datasets)
data = loader.load_all_drivers(batch_unload=False)
```

## Generated Visualizations

### Static Plots (PNG)
- Directivity Index (DI) comparison
- Beamwidth curves (-3dB, -6dB)
- Dipole null analysis
- Normalized/Absolute contour plots
- Polar diagrams (single driver and overlaid)
- Crossover match analysis
- Frequency response by angle (absolute and normalized to 0°)

### Interactive Plots (HTML)
- DI comparison with hover info
- Beamwidth comparison
- Contour heatmaps
- Polar explorer with frequency slider and manual entry
- Crossover analysis per frequency
- **Multi-driver frequency response explorer** with:
  - Driver overlay comparison with distinct markers
  - Per-driver gain offsets, delay (us), and invert toggles
  - Angle toggle grid with quick-select buttons
  - SUM display (off / overlay / only) with optional phase
  - **IIR filter editor**: Peaking, HP, LP, Highshelf, Lowshelf filters
  - Visual crossover overlay (LR/BW) and auto-optimize tools
  - Real-time filter simulation on all angles
  - Hypex Filter Design `.dsp` import/export (not miniDSP), Config.xml conversion via external tools, and CamillaDSP-compatible YAML import/export
  - LocalStorage persistence for filters, crossovers, and UI state
  - Optional extra driver datasets (e.g., Juan baffleless) loaded on demand

## File Structure

| File | Description |
|------|-------------|
| `run_pipeline.py` | Main entry point |
| `config.py` | Configuration settings |
| `polar_data_loader.py` | REW API interface and data loading |
| `generate_visualizations.py` | Plot generation |
| `directivity_calculations.py` | Acoustic calculations |
| `Makefile` | Build automation (data → viz → docs → deploy) |
| `requirements.txt` | Python dependencies |

## Adding New Measurement Sets

1. Add entry to `MEASUREMENT_SETS` in `config.py`
2. If using a new naming pattern, add an entry to `_PATTERN_DEFS` (and/or `_PATTERN_ALIASES`) in `polar_data_loader.py`
3. Run: `python run_pipeline.py -m your_new_set`
4. Run `make sync` to publish regenerated output under `docs/`

## License

MIT
