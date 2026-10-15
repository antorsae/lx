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
        "hdf5_file": "polar_data_andres_l22mg_first_lobe.h5",
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
        "measurement_metadata_overrides": {
            "L22MG (nude)": {
                "measurement_distance_m": 0.50,
                "measurement_height_reference": "l22mg",
                "notes": "Measurement distance: 50 cm from driver. Mic height: L22MG/LM.",
            },
        },
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
| `kaspar` | `F0-UM MU10RB (Kaspar DSP).mdat` | Same shape as `andres`, kept separate from its combination-file and diagnostic-HDF5 rules |
| `juan` | `GRS PT6816 45 F.mdat` | Front (F) and Rear (R) measurements |
| `scanspeak` | `SS10F8414G10 45 F.mdat` | Same convention as `juan` |
| `juan_suffix` | `SS10F8424G00 45 F sn 074.mdat` | Same as `juan`, allowing text after `F`/`R` |
| `lx521_system` | `LX521 HIGH MID INV ORIGINAL 45 GRADOS F.mdat` | Full system measurements with GRADOS notation |
| `juan_360` | `BTM 250 DEG FRONT.mdat`, `BTM 120 DEG REA EQ.mdat`, `BTM LM+4 30 DEG F.mdat` | Absolute turntable rotation; 0-180 loads as front/rear and 180-360 as the other side of the same driver, and text after `FRONT`/`FRO`/`F`/`REA` names a separate sweep |

Measurement sources may be nested below each configured source directory. This supports source layouts such as `POLARES L10NEO/FRONTALES` and `POLARES L10NEO/TRASERAS`.

The Juan driver pages include **ND25FW4 (nude 18mm)** from
`DAYTON ND25FW4 ANIDADOS 18 MM NUDE.zip`: 14 REW captures taken on 9 December
2024, with front and rear measurements at 0–90° in 15° increments over approximately
500 Hz–20 kHz. These use the collection's 0.5 ms left / 3.0 ms right gate,
REW IR-start reference, and no smoothing. The source files have no timing reference
and do not specify microphone distance or height. The Dayton curves are also
available through the Juan baffleless overlay in the other measurement explorers.

The **juan-batman** set publishes `BATMAN POLARES 360 GRADOS.zip`: Juan's BATMAN
(MU10 plus nested tweeters on the outer side, "ANIDADOS EN EL EXTERIOR"; front tweeter
in inverted polarity) measured on 12 October 2026 at 50 cm, with the mic 108 cm high
and no timing reference. One
turntable sweep covers 0-350° in 10° steps, and a second sweep repeats 0-180° with
Equalizer APO linearization to the Harman curve. Files are named by absolute
rotation; the `juan_360` pattern loads 0-180° as the usual front and rear angles
and 180-360° as the driver's other side (`left_angles`, `left_rear_angles` in the
HDF5). The polar plots then draw the measured 180-360° half instead of mirroring
0-180°, the contour and angle pages show it as negative angles (rotation − 360°),
and the measurement summary lists it as LF/LR. This matters because the two halves agree within about 2.5 dB
below 2 kHz from 10° to 70°, but differ by 5-17 dB above 2 kHz, near the side nulls
and in the rear quadrant. DI and beamwidth use the power average of both sides; the
frequency explorer shows the 0-180° side. The Harman EQ sweep covers 0-180° only, so
its plots mirror it like the other sets. `BTM 360_0 0DEG FRON` repeats 0° at the
end of the sweep (0.07 dB RMS from the first capture) and is not loaded.
The set uses the usual 0.5 ms / 3.0 ms gate and REW's IR start: at the side nulls
(`BTM 260 DEG FRONT`, `BTM 100 DEG REA EQ`) a reflection 9-13 ms late is stronger
than the direct sound, so REW's t=0 sits on the reflection, but its IR start still
marks the direct arrival. Juan's `PANTALLAZOS` screenshots are ungated, so their
70-110° curves are dominated by those reflections.

The **juan-sweetspot** set publishes `LX521_4 MGNEO 0_90 DESDE SWEETSPOT.zip`,
`BATMAN 0_90 DESDE SWEETSPOT.zip` and
`BATMAN LM+4DB ANIDADOS INTERIOR 0_90 DESDE SWEETSPOT.zip`. The first two are Juan's
LX521.4 MGNEO and BATMAN, each as the right speaker with the UM not inverted at
PB -20 dB, measured on 13 October 2026 from the same listening seat, 20 minutes apart,
with BATMAN's nested tweeters on the outer side ("ANIDADOS EN EXTERIOR"; the third zip
repeats that screenshot under this name). On 15 October 2026 Juan measured BATMAN
again from the same seat with the nested tweeters on the inner side ("ANIDADOS EN
INTERIOR") and the LM raised 4 dB. The mic sat 114 cm high, 85 cm in front of the rear
wall, and aimed at the front wall, with the speaker distance not recorded and no
timing reference. It stayed at the seat while the speaker turned from 0°, with the
baffle facing the wall, to 90° in 10° steps (`LX521 {rotation} DEG FRONT`,
`BATM {rotation} DEG FRONT` and `BTM LM+4 {rotation} DEG F`, loaded with the
`juan_360` pattern), so the angles are speaker rotation, not off-axis angle. The
LX521.4 responses mirror about 45°: 40/50°, 30/60°, 20/70°, 10/80° and 0/90° agree
within 1.2 dB RMS above 1 kHz, against 2.3-2.5 dB for an axis at 40° or 50°. BATMAN
with the outer tweeters mirrors less well (about 3 dB RMS), as its turntable set
already showed its two sides differ, and its treble peaks near 50°. So both point at
the seat near 45-50°, and 0° and 90° are each about 45° off-axis.

The 15 October sweep is 3.6-4.4 dB louder than the 13 October BATMAN from 125 Hz to
1 kHz (median per rotation), which is the LM's 4 dB. Above 1.6 kHz the median
difference stays within 1 dB, but gated, single rotations differ by up to 6 dB at
3-6 kHz: with the tweeters on the inner side, the treble (8-16 kHz) is centred on 45°
instead of 50°, and the rotations mirror about 45° within 2.3 dB RMS, against 2.9 dB
with the outer tweeters. Its `PANTALLAZOS` also hold a screenshot titled "CON HIGH SHELF -9 DB Q5 EN
14 KHZ", saved before these captures were taken. That run's captures are not in the
zip; reloaded ungated with 1/12-octave smoothing, the captures match Juan's screenshot
without the shelf.

The set uses the site's usual 0.5 ms / 3.0 ms gate from REW's IR start, with
1/12-octave smoothing. The rear wall's reflection arrives 4.4-4.8 ms after the direct
sound and is strongest at 0-20°, only 3-10 dB below the direct sound, where the
speaker faces the wall; the gate ends before it and before a weaker arrival at
2.9-3.4 ms (12-25 dB down), so the plots show what the speakers radiate toward the
seat. Inside the gate, arrivals from 1.7 ms on are at least 19 dB down; the ones
within 1 ms, 11-20 dB down, come from the speakers themselves, since no room surface
is close enough to reflect that early. A 4.2 ms gate, the longest
that still ends before the wall, gives nearly the same curves above 3 kHz (95% of
points within 1.2 dB), but in the LX521.4's 1-3 kHz region single frequencies move by
up to 14 dB, so that region depends on the gate; both gates show its direct sound well
below the in-room level around 1.3-1.8 kHz, most at 0° and 90°. Juan's ungated
screenshots include the wall and the rest of the room, which fill in the low rotations:
above 2 kHz the spread between rotations at one frequency is a median of 2-4 dB
ungated against 6-10 dB gated (both with 1/12-octave smoothing), and at most 7-11 dB
against 18-22 dB. Gated, the LX521.4 dips 10-14 dB at 5-7 kHz at every rotation,
including 40° and 50° (18-37 dB deep unsmoothed); the room fills these in to a shallow
dip. The
published `lx521-system` set (UM inverted, 50 cm) has no such dip on-axis, and neither
BATMAN sweep has one at 50° (at most 5 dB at 40°).

The set has no DI or beamwidth pages. Its polar explorer draws each capture at its
rotation, relative to the 0° rotation like the other polar explorers, from -40 to
+25 dB because the rotations aiming at the seat reach 22 dB above 0°; nothing is
mirrored. Each angle page draws every rotation, with mirror pairs in one color and the
later one dashed, and the frequency explorer overlays the three sweeps at any rotation.

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
│   ├── polar_data_andres_l22mg_first_lobe.h5 # Andres first-lobe validation target
│   ├── polar_data_andres.h5                  # Legacy strongest-lobe diagnostic
│   ├── polar_data_juan_baffleless.h5         # Processed data (juan-baffleless set)
│   ├── polar_data_juan_batman.h5             # Processed data (juan-batman set)
│   ├── polar_data_juan_sweetspot.h5          # Processed data (juan-sweetspot set)
│   ├── polar_data_kaspar.h5                  # Processed data (kaspar set, on-axis only)
│   └── polar_data_lx521_system.h5            # Processed data (lx521-system set)
├── andres/                            # Visualizations for andres set
│   ├── static_plots/
│   │   ├── core/                     # DI, beamwidth, contour plots
│   │   └── polar/                    # Polar diagrams
│   └── interactive/                  # HTML interactive plots (gzip compressed)
├── juan-baffleless/                   # Visualizations for juan-baffleless set
│   ├── static_plots/
│   └── interactive/
├── lx521-system/                      # Visualizations for lx521-system set
│   ├── static_plots/
│   └── interactive/
├── juan-batman/                       # Visualizations for juan-batman set
│   ├── static_plots/
│   └── interactive/
├── juan-sweetspot/                    # Speakers turned at the seat: no polar/DI plots
│   ├── static_plots/
│   └── interactive/
└── kaspar/                            # On-axis-only set: no static/polar plots
    └── interactive/                   # Frequency response explorer + summary
```

### On-Axis-Only Sets

A set with a single measurement angle sets `"single_angle": True` in its
config entry. Directivity, beamwidth, contour and polar output is skipped
(there is no angular coverage to compute it from), and only the frequency
response explorer and measurement summary are generated. Such a set is still
offered as a loadable overlay in every other set's explorer, but it is not
interpolated onto their angle grids — its curves are simply hidden at angles
it was not measured at.

### Speaker Rotated at the Seat

A set whose captures turn the speaker in front of a mic fixed at the listening
seat, starting off-axis, sets `"speaker_rotation_at_seat": True`. Its angles are
speaker rotation, so the pipeline writes only the frequency response by rotation
(every rotation, instead of the 0/30/60/90° selection), the frequency explorer, a polar
explorer by rotation, the absolute contour with a rotation axis, and the measurement
summary. The polar explorer shows the measured rotations as a sector, with levels
relative to the 0° rotation and room above 0 dB, since 0° is off-axis and other
rotations are louder. Directivity, beamwidth, crossover and static polar plots take 0° as
on-axis and are skipped. So are the normalized contour and frequency response plots,
whose scales stop a few dB above 0 dB. Rotations do not line up with off-axis angles,
so such a set's explorers offer no other set as an overlay, and the other sets'
explorers do not offer it either.

Per-set `gate_left_ms`, `gate_right_ms` and `smoothing` keys override the
`config.py` defaults, for captures where the default 0.5/3.0 ms gate would
discard the band of interest.

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
