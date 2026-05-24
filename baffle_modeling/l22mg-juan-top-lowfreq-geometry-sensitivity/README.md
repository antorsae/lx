# L22MG Low-Frequency Null Geometry Sensitivity

This diagnostic changes only physical geometry or mic assumptions around the same Juan-fitted H1659 source and the selected measured target.

- This is diagnostic-only evidence, not an acceptance proof.
- No scalar dB correction, target-derived source correction, rear correction, or null-depth fitting is applied.
- Target data are used only as the scoring surface.

## Provenance

- Target HDF5: `output/data/polar_data_juan_lx521_top_raw.h5` / `L22MG (LX521 top raw)`.
- Target kind: `juan_lx521_top_raw`.
- Validation hypothesis: `juan_baffleless_to_juan_top_baffle_l22mg_raw`.
- Processing / peak policy: repo processed Juan LX521 top-baffle raw/no-crossover/no-EQ HDF5; measurement distance 0.5 m; height reference `l22mg`; Measurement distance: 50 cm. Mic height: L22MG/LM. LX521 top baffle mounted; raw/no crossover/no EQ. / Juan HDF5 direct_ir_peak_policy=first-strong.
- Gate/window policy: Juan processed-HDF5 gate_left_ms=0.5, gate_right_ms=3.0, smoothing=None.
- Normalization policy: per-frequency normalized polar shape, SPL(theta)-SPL(0 deg).
- Published explorer match: False, 1004 Hz max delta nan dB, path ``.
- Juan source HDF5: `output/data/polar_data_juan_baffleless.h5`; naked source radius 0.5 m; rear phase `dipole`.
- Source model: `h1659-modal-compact-2` profile rings, azimuth 24, regularization 1e-06, SVD rcond 0.
- Frequencies scored: 300.293 Hz, 304.321 Hz.
- Angles scored: 0, 15, 30, 45, 60, 75, 90 deg.
- BEM mesh: delaunay-local h 42 mm, boundary/local h 32/32 mm, q7, near q7.

## Result

- Best variant by hotspot score: `width205_um_face`.
- Best 0-60 deg RMS: 1.107 dB.
- Best 70-90 deg RMS: 13.301 dB.
- Best 80-90 deg RMS: 1.012 dB.
- Interpretation: these physical geometry/mic variants do not close the measured low-frequency high-angle null.

## Variants

| variant | diagnostic | Wmax | thickness | mic z | UM face | tweeter faces | aperture | RMS 0-60 | RMS 70-90 | RMS 80-90 | RMS 90 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| width205_um_face | True | 205.0 | 18.30 | 0.0 | True | False | andres-mounted | 1.107 | 13.301 | 1.012 | 1.012 |
| open_passive_holes | True | 305.0 | 18.30 | 0.0 | False | False | open-holes | 1.118 | 13.451 | 2.773 | 2.773 |
| solid_patch_no_passives | True | 305.0 | 18.30 | 0.0 | False | False | none | 1.106 | 13.477 | 3.316 | 3.316 |
| current_um_face | True | 305.0 | 18.30 | 0.0 | True | False | andres-mounted | 1.209 | 13.944 | 1.554 | 1.554 |
| um_plus_tweeter_faces | True | 305.0 | 18.30 | 0.0 | True | True | andres-mounted | 1.209 | 13.946 | 1.562 | 1.562 |
| width405_um_face | True | 405.0 | 18.30 | 0.0 | True | False | andres-mounted | 1.224 | 13.981 | 1.468 | 1.468 |

## Variant Notes

- `current_um_face`: custom CLI variant.
- `solid_patch_no_passives`: custom CLI variant.
- `open_passive_holes`: custom CLI variant.
- `um_plus_tweeter_faces`: custom CLI variant.
- `width205_um_face`: custom CLI variant.
- `width405_um_face`: custom CLI variant.

Files: `geometry_sensitivity.csv`, `variant_summary.csv`, `plots/geometry_sensitivity_summary.png`, and `plots/geometry_sensitivity_curves.png`.
