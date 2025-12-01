# LX521 Polar Analysis Pipeline

A Python pipeline for processing acoustic polar response measurements from REW (Room EQ Wizard) and generating comprehensive visualizations for speaker driver analysis.

## Features

- Load and process `.mdat` measurement files via REW API
- Apply time gating to remove room reflections
- Support for multiple measurement sets with different naming conventions
- Full 360° polar plots when front and rear measurements are available
- Generate directivity analysis (DI, beamwidth, ERDI)
- Spinorama curves (listening window, early reflections, predicted in-room)
- Interactive HTML plots and static PNG exports
- Crossover match analysis for multi-driver systems

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
        "path": Path("../Mediciones Andres"),
        "pattern_type": "andres",  # F{angle}-{driver}.mdat
        "angles": list(range(0, 91, 10)),
        "has_rear": False,
        "hdf5_file": "polar_data_andres.h5",
    },
    "juan": {
        "path": Path("../Mediciones Juan/GRS PT6816 A MIC ON AXIS"),
        "pattern_type": "juan",  # {driver} {angle} {side}.mdat
        "angles": [0, 15, 30, 45, 60, 75, 90],
        "has_rear": True,
        "hdf5_file": "polar_data_juan.h5",
    },
}
```

### Supported Naming Conventions

| Pattern | Example | Description |
|---------|---------|-------------|
| `andres` | `F45-10F8824.mdat` | Front-only measurements |
| `juan` | `GRS PT6816 45 F.mdat` | Front (F) and Rear (R) measurements |

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
python run_pipeline.py -m juan

# Skip data loading (use existing HDF5)
python run_pipeline.py -m juan --skip-loading

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
| `--no-smoothing` | Disable frequency response smoothing |

## Output Structure

```
output/
├── data/
│   ├── polar_data_andres.h5    # Processed data (andres set)
│   └── polar_data_juan.h5      # Processed data (juan set)
├── andres/                      # Visualizations for andres set
│   ├── static_plots/
│   │   ├── core/               # DI, beamwidth, contour plots
│   │   ├── polar/              # Polar diagrams
│   │   └── spinorama/          # Spinorama curves
│   └── interactive/            # HTML interactive plots
└── juan/                        # Visualizations for juan set
    ├── static_plots/
    └── interactive/
```

## Pipeline Workflow

1. **Connect to REW API** - Launches REW if not running
2. **Load Measurements** - Reads `.mdat` files for each driver/angle
3. **Auto-fix Timing** - Aligns impulse response peak to t=0
4. **Apply Time Gating** - Removes room reflections (default: 0.5ms / 3.0ms)
5. **Get Frequency Response** - Retrieves magnitude/phase data
6. **Save to HDF5** - Stores processed data for visualization
7. **Generate Visualizations** - Creates all plots and analysis

## Generated Visualizations

### Static Plots (PNG)
- Directivity Index (DI) comparison
- Beamwidth curves (-3dB, -6dB)
- ERDI (Early Reflections DI)
- Normalized/Absolute contour plots
- Polar diagrams (single driver and overlaid)
- Spinorama curves per driver
- Crossover match analysis

### Interactive Plots (HTML)
- DI comparison with hover info
- Beamwidth comparison
- Contour heatmaps
- Polar explorer with frequency slider
- Crossover analysis per frequency

## File Structure

| File | Description |
|------|-------------|
| `run_pipeline.py` | Main entry point |
| `config.py` | Configuration settings |
| `polar_data_loader.py` | REW API interface and data loading |
| `generate_visualizations.py` | Plot generation |
| `directivity_calculations.py` | Acoustic calculations |
| `requirements.txt` | Python dependencies |

## Adding New Measurement Sets

1. Add entry to `MEASUREMENT_SETS` in `config.py`
2. If using a new naming pattern, add parsing logic to `_parse_filename()` in `polar_data_loader.py`
3. Run: `python run_pipeline.py -m your_new_set`

## License

MIT
