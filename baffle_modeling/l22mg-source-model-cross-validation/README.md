# L22MG naked-source cross-validation

This is a Juan-only leave-one-angle-out audit of equivalent-source models. Andres' mounted-baffle measurements are not read, fitted, or used for ranking.

## Method

- For each fold, one nonzero Juan angle is withheld from one side, while 0 deg remains in the training set as the normalization anchor.
- The source is fitted to the remaining naked front/rear polars and then evaluated at 0 deg plus the held-out angle.
- The scored quantity is normalized SPL, `SPL(theta)-SPL(0)`, so this checks angular source shape rather than absolute level.
- In-sample absolute source-fit RMS is reported separately because normalized shape cannot detect a uniform front/rear source gain error.
- In-sample phase RMS is reported separately because absolute SPL cannot detect a uniform rear timing or phase error.
- Asymmetric split-discrete rows use independent front/rear source depths and ring radii fitted from Juan only.
- Smoothness rows, when present, are adjacent-source Tikhonov penalties fitted from Juan only; they are not Andres-derived corrections.
- Active-surface annular rows fit normal-velocity amplitudes on an H1659-shaped source surface from Juan only. `annular` uses the H1659 profile annuli; `uniform-annular` uses configurable equal-area radial bands.
- Physical-diaphragm rows fit a single H1659 cone/surround prescribed-Neumann surface with four capped radial velocity shapes: piston, two cone bending/tilt shapes, and surround. They are Juan-only and do not use a rear scalar or Andres data.
- Physical rear-basket rows keep those four diaphragm modes coupled front/rear and add only a small low-order rear-side basket/directivity block. The rear block is Juan-only and cannot affect the front fit.
- Rear magnitude correction: none. The `dipole` rows only reconstruct rear phase as opposite polarity to front; magnitudes remain Juan's measured magnitudes.
- No Andres-derived gain, delay, rear correction, or baffle result is used.

## Inputs

- Juan source HDF5: `output/data/polar_data_juan_baffleless.h5` driver `L22MG (nude)`.
- Juan front notes: distance=0.5 m; height=unknown; first note=Measurement distance: 50 cm from driver..
- Juan rear notes: distance=0.5 m; height=unknown; first note=Measurement distance: 50 cm from driver..
- Equivalent-source reference radius: 0.5 m.
- Frequency grid: 300-2000 Hz at 24 points/octave.
- Candidate set: `compact`.

## Ranking

| Rank | Model | Phase | CV worst 300-1200 | CV front | CV rear | In-sample norm front/rear | In-sample abs front/rear | Phase front/rear deg | Failed folds |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `axisymmetric-directivity-table-front-duplicate` | dipole | 1.229 | 1.229 | 0.984 | 0.000/0.569 | 0.000/0.569 | 0.000/68.034 | 0 |
| 2 | `diagnostic-legendre-o8` | dipole | 2.201 | 2.201 | 1.437 | 9.203/8.925 | 9.145/8.844 | 23.738/62.252 | 0 |
| 3 | `h1659-modal-compact2-az24-reg1e-06` | dipole | 3.298 | 3.298 | 2.877 | 2.631/1.226 | 2.578/0.974 | 15.048/13.734 | 0 |
| 4 | `h1659-modal-full2-az24-reg1e-06` | dipole | 3.300 | 3.300 | 2.911 | 2.682/1.261 | 2.629/1.007 | 15.134/13.889 | 0 |
| 5 | `measured-phase-h1659-modal-full2-az24-reg1e-06` | measured | 3.494 | 3.494 | 2.743 | 2.682/5.967 | 2.629/6.200 | 15.134/62.847 | 0 |
| 6 | `h1659-modal-compact3-az24-reg1e-06` | dipole | 3.768 | 3.768 | 3.300 | 2.215/1.171 | 2.184/0.934 | 14.029/13.529 | 0 |
| 7 | `h1659-modal-full4-az24-reg1e-06-svd1e-05` | dipole | 3.800 | 2.767 | 3.800 | 1.050/0.952 | 1.039/0.868 | 12.999/12.921 | 0 |
| 8 | `h1659-profile-ring-full-az24-reg1e-06-svd1e-05` | dipole | 3.842 | 3.288 | 3.842 | 0.855/1.091 | 0.845/1.030 | 12.298/12.298 | 0 |
| 9 | `h1659-profile-ring-full-az24-reg1e-06` | dipole | 3.844 | 3.286 | 3.844 | 0.855/1.090 | 0.845/1.030 | 12.298/12.297 | 0 |
| 10 | `h1659-profile-ring-compact-az24-reg1e-06` | dipole | 4.090 | 3.290 | 4.090 | 0.865/1.093 | 0.856/1.033 | 12.283/12.269 | 0 |
| 11 | `h1659-modal-full3-az24-reg1e-06` | dipole | 4.090 | 4.090 | 3.827 | 2.184/1.204 | 2.156/0.965 | 13.920/13.608 | 0 |
| 12 | `h1659-profile-ring-compact-az24-reg1e-06-svd1e-05` | dipole | 4.092 | 3.291 | 4.092 | 0.865/1.093 | 0.856/1.033 | 12.283/12.269 | 0 |

## Current BEM Source Rows

- Stable corrected-geometry source: `stable-split-discrete-d45-r0_40_75-reg1e-06` rank 25, CV worst-side RMS 8.622 dB over 300-1200 Hz (8.245 dB over 300-2000 Hz), in-sample normalized front/rear 2.541/2.642 dB and absolute front/rear 2.507/2.598 dB; phase front/rear 16.234/16.308 deg.
- Current wide/SVD diagnostic source: `current-wide-split-discrete-d45-r0_35_70_95-reg1e-06-svd1e-05` rank 26, CV worst-side RMS 8.810 dB over 300-1200 Hz (8.167 dB over 300-2000 Hz), in-sample normalized front/rear 2.404/2.500 dB and absolute front/rear 2.381/2.469 dB; phase front/rear 14.889/14.961 deg.
- Best asymmetric split-discrete source in this candidate set: `asym-split-discrete-f45-r55-r0_35_70_95-reg1e-06-svd1e-05` rank 17, CV worst-side RMS 6.793 dB over 300-1200 Hz (7.441 dB over 300-2000 Hz), in-sample normalized front/rear 2.404/2.656 dB and absolute front/rear 2.381/2.627 dB; phase front/rear 14.889/15.474 deg.
- Best active-surface radial source in this candidate set: not included in this candidate set.
- Best physical-diaphragm active source in this candidate set: not included in this candidate set.
- Best physical rear-basket active source in this candidate set: not included in this candidate set.

## Juan-Only Recommendation

- Thresholds: finite non-Legendre source, failed folds <= 0, in-sample side-worst normalized RMS <= 3.000 dB, in-sample side-worst absolute SPL RMS <= 3.000 dB, in-sample side-worst phase RMS <= 30.000 deg, and held-out side-worst normalized RMS <= 4.000 dB over 300-1200 Hz.
- First Juan-only recommended source candidate: `h1659-modal-compact2-az24-reg1e-06` rank 3; CV worst-side RMS 3.298 dB; in-sample side-worst normalized/absolute 2.631/2.578 dB; phase 15.048 deg.
- Current wide/SVD source recommendation status: `rejected_weak_cv`; Juan held-out side-worst normalized RMS 8.810 dB exceeds 4.000 dB.
- Recommendation is not Andres acceptance: any recommended source still needs a full mounted-baffle BEM run, mesh convergence, and validation against Andres before it can replace the current baseline.
- Active-surface rows are Juan-fitted H1659 prescribed-Neumann source surfaces. They are finite source models, but still acoustic open-surface fits rather than coupled elastic cone/suspension simulations.
- The axisymmetric directivity-table row is diagnostic-only: it rotates Juan's measured angular field, but does not resolve a unique finite 3D source distribution.

## Interpretation

- A low in-sample Juan fit with a high held-out-angle error is a source-model overfit warning before any baffle simulation is considered.
- This audit does not prove Andres acceptance. It only checks whether the naked-source representation generalizes across Juan's measured angles.
- If the current BEM source is not among the better cross-validated rows, source support/regularization should be treated as an active uncertainty in the mounted-baffle residuals.

## Files

- `source_cv_summary.csv` summarizes each candidate.
- `source_cv_recommendations.csv` records the Juan-only recommendation status and reason for each candidate.
- `source_cv_folds.csv` contains every held-out-angle fold.
- `plots/source_cv_worst_side_300_1200.png` ranks candidates by held-out normalized-polar error.
- `plots/source_cv_vs_insample.png` compares in-sample fit to cross-validation error.
