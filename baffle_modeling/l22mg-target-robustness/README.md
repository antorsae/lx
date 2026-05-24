# L22MG Target Robustness Audit

This compares Andres L22MG target variants after each target is normalized to its own 0-degree response.
It quantifies validation-target movement only; it does not fit the model, exclude data, or change acceptance gates.

- Band: 300-1200 Hz.
- Delta convention: comparison target minus reference target.
- CSVs: `target_robustness_summary.csv`, `target_robustness_angle_summary.csv`, `target_robustness_worst_points.csv`.

## first-lobe-diagnostic-andres minus published-parity-andres

- Reference HDF5: `output/data/polar_data_andres_early_peak_legacy.h5`.
- Comparison HDF5: `output/data/polar_data_andres_l22mg_first_lobe.h5`.
- Contour plot: `first-lobe-diagnostic-andres_target_robustness_contours.png`.
- Primary target movement:
  - 0-90 deg RMS 5.229 dB; max -42.627 dB at 639.0 Hz / 70 deg.
  - Through 60 deg RMS 0.040 dB; through 80 deg RMS 4.326 dB.
  - 70-90 deg RMS 9.546 dB; 80-90 deg RMS 8.648 dB.

### Angle Summary

| angle | RMS delta | p95 abs | max delta | max Hz |
| ---: | ---: | ---: | ---: | ---: |
| 0 deg | 0.000 | 0.000 | 0.000 | 300.3 |
| 10 deg | 0.000 | 0.000 | 0.000 | 300.3 |
| 20 deg | 0.000 | 0.000 | 0.000 | 300.3 |
| 30 deg | 0.000 | 0.000 | 0.000 | 300.3 |
| 40 deg | 0.000 | 0.000 | 0.000 | 300.3 |
| 50 deg | 0.000 | 0.000 | 0.000 | 300.3 |
| 60 deg | 0.106 | 0.185 | 0.352 | 1199.7 |
| 70 deg | 11.126 | 22.580 | -42.627 | 639.0 |
| 80 deg | 6.678 | 12.791 | -29.745 | 1067.5 |
| 90 deg | 10.246 | 25.325 | 26.636 | 316.0 |

### Worst Points

| rank | Hz | angle | signed delta | abs delta |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 639.0 | 70 | -42.63 | 42.63 |
| 2 | 638.7 | 70 | -42.51 | 42.51 |
| 3 | 639.4 | 70 | -42.51 | 42.51 |
| 4 | 638.3 | 70 | -42.19 | 42.19 |
| 5 | 639.8 | 70 | -42.18 | 42.18 |
| 6 | 637.9 | 70 | -41.70 | 41.70 |
| 7 | 640.1 | 70 | -41.68 | 41.68 |
| 8 | 637.6 | 70 | -41.10 | 41.10 |
| 9 | 640.5 | 70 | -41.07 | 41.07 |
| 10 | 637.2 | 70 | -40.43 | 40.43 |
| 11 | 640.9 | 70 | -40.39 | 40.39 |
| 12 | 636.8 | 70 | -39.73 | 39.73 |
| 13 | 641.2 | 70 | -39.68 | 39.68 |
| 14 | 636.5 | 70 | -39.03 | 39.03 |
| 15 | 641.6 | 70 | -38.96 | 38.96 |

## legacy-strongest-andres minus published-parity-andres

- Reference HDF5: `output/data/polar_data_andres_early_peak_legacy.h5`.
- Comparison HDF5: `output/data/polar_data_andres.h5`.
- Contour plot: `legacy-strongest-andres_target_robustness_contours.png`.
- Primary target movement:
  - 0-90 deg RMS 5.098 dB; max -29.459 dB at 1070.1 Hz / 80 deg.
  - Through 60 deg RMS 0.000 dB; through 80 deg RMS 4.145 dB.
  - 70-90 deg RMS 9.307 dB; 80-90 deg RMS 8.897 dB.

### Angle Summary

| angle | RMS delta | p95 abs | max delta | max Hz |
| ---: | ---: | ---: | ---: | ---: |
| 0 deg | 0.000 | 0.000 | 0.000 | 300.3 |
| 10 deg | 0.000 | 0.000 | 0.000 | 300.3 |
| 20 deg | 0.000 | 0.000 | 0.000 | 300.3 |
| 30 deg | 0.000 | 0.000 | 0.000 | 300.3 |
| 40 deg | 0.000 | 0.000 | 0.000 | 300.3 |
| 50 deg | 0.000 | 0.000 | 0.000 | 300.3 |
| 60 deg | 0.000 | 0.000 | 0.000 | 300.3 |
| 70 deg | 10.078 | 20.892 | -26.501 | 644.5 |
| 80 deg | 7.284 | 17.426 | -29.459 | 1070.1 |
| 90 deg | 10.260 | 23.879 | 24.617 | 325.6 |

### Worst Points

| rank | Hz | angle | signed delta | abs delta |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1070.1 | 80 | -29.46 | 29.46 |
| 2 | 1070.4 | 80 | -29.38 | 29.38 |
| 3 | 1069.7 | 80 | -29.37 | 29.37 |
| 4 | 1070.8 | 80 | -29.14 | 29.14 |
| 5 | 1069.3 | 80 | -29.12 | 29.12 |
| 6 | 1071.2 | 80 | -28.76 | 28.76 |
| 7 | 1069.0 | 80 | -28.74 | 28.74 |
| 8 | 1071.5 | 80 | -28.28 | 28.28 |
| 9 | 1068.6 | 80 | -28.26 | 28.26 |
| 10 | 1071.9 | 80 | -27.72 | 27.72 |
| 11 | 1068.2 | 80 | -27.71 | 27.71 |
| 12 | 1067.9 | 80 | -27.13 | 27.13 |
| 13 | 1072.3 | 80 | -27.12 | 27.12 |
| 14 | 1067.5 | 80 | -26.53 | 26.53 |
| 15 | 1072.6 | 80 | -26.50 | 26.50 |

## direct-gate-0p8ms-diagnostic-andres minus published-parity-andres

- Reference HDF5: `output/data/polar_data_andres_early_peak_legacy.h5`.
- Comparison HDF5: `output/data/polar_data_andres_l22mg_direct_gate_0p8ms.h5`.
- Contour plot: `direct-gate-0p8ms-diagnostic-andres_target_robustness_contours.png`.
- Primary target movement:
  - 0-90 deg RMS 3.758 dB; max 21.233 dB at 322.3 Hz / 90 deg.
  - Through 60 deg RMS 0.836 dB; through 80 deg RMS 2.900 dB.
  - 70-90 deg RMS 6.741 dB; 80-90 deg RMS 6.291 dB.

### Angle Summary

| angle | RMS delta | p95 abs | max delta | max Hz |
| ---: | ---: | ---: | ---: | ---: |
| 0 deg | 0.000 | 0.000 | 0.000 | 300.3 |
| 10 deg | 0.128 | 0.197 | 0.198 | 424.1 |
| 20 deg | 0.316 | 0.471 | 0.474 | 438.4 |
| 30 deg | 0.405 | 0.596 | 0.604 | 1043.7 |
| 40 deg | 0.575 | 1.251 | 1.271 | 1133.8 |
| 50 deg | 1.076 | 2.607 | 2.679 | 1171.5 |
| 60 deg | 1.768 | 4.595 | 5.607 | 1199.7 |
| 70 deg | 7.561 | 11.881 | -12.256 | 300.3 |
| 80 deg | 3.693 | 6.615 | -6.653 | 357.4 |
| 90 deg | 8.094 | 20.473 | 21.233 | 322.3 |

### Worst Points

| rank | Hz | angle | signed delta | abs delta |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 322.3 | 90 | 21.23 | 21.23 |
| 2 | 322.6 | 90 | 21.23 | 21.23 |
| 3 | 321.9 | 90 | 21.23 | 21.23 |
| 4 | 323.0 | 90 | 21.23 | 21.23 |
| 5 | 321.5 | 90 | 21.23 | 21.23 |
| 6 | 323.4 | 90 | 21.23 | 21.23 |
| 7 | 321.2 | 90 | 21.23 | 21.23 |
| 8 | 323.7 | 90 | 21.23 | 21.23 |
| 9 | 320.8 | 90 | 21.23 | 21.23 |
| 10 | 324.1 | 90 | 21.23 | 21.23 |
| 11 | 320.4 | 90 | 21.23 | 21.23 |
| 12 | 324.5 | 90 | 21.23 | 21.23 |
| 13 | 320.1 | 90 | 21.23 | 21.23 |
| 14 | 324.8 | 90 | 21.22 | 21.22 |
| 15 | 319.7 | 90 | 21.22 | 21.22 |

