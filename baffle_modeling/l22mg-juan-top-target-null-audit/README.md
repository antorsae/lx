# Juan L22MG Top-Baffle Target Null Audit

This report audits measured Juan HDF5 data only. It does not run BEM, refit the source, or change validation gates.

- Target: `output/data/polar_data_juan_lx521_top_raw.h5` / `L22MG (LX521 top raw)`.
- Nude reference: `output/data/polar_data_juan_baffleless.h5` / `L22MG (nude)`.
- Corroboration capture: `output/data/polar_data_juan_lx521_top_raw.h5` / `L22MG+L10NEO+Tweeters (LX521 top raw)`.
- Band searched: 300-600 Hz, angles >= 60 deg.

## Key Finding

The deepest high-angle target point is 300.293 Hz / 75 deg: L22 top target -29.894 dB, L22 nude -9.594 dB, measured target-minus-nude transfer -20.300 dB, and L22+L10+tweeters top capture -18.024 dB.
The multi-driver top capture is +11.870 dB relative to the L22-alone target at the same frequency/angle. This is corroboration context only because that capture is not an L22-alone validation target.

## Same-Frequency Angle Table

| deg | L22 top target | L22 nude | target-nude transfer | combo top | combo-target |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 0.000 | +0.000 | 0.000 | +0.000 |
| 15 | -0.349 | -0.201 | -0.148 | -0.483 | -0.133 |
| 30 | -1.591 | -0.950 | -0.641 | -1.749 | -0.158 |
| 45 | -3.788 | -2.409 | -1.380 | -3.964 | -0.176 |
| 60 | -7.750 | -4.910 | -2.841 | -8.125 | -0.374 |
| 75 | -29.894 | -9.594 | -20.300 | -18.024 | +11.870 |
| 90 | -17.408 | -22.465 | +5.056 | -21.225 | -3.817 |

## Angle Summary

| deg | target min | min Hz | target mean | transfer at min | transfer mean | combo-target at min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 300.3 | 0.000 | +0.000 | +0.000 | +0.000 |
| 15 | -0.421 | 599.9 | -0.378 | -0.155 | -0.133 | -0.131 |
| 30 | -1.841 | 599.9 | -1.690 | -0.619 | -0.580 | -0.092 |
| 45 | -4.263 | 599.9 | -3.961 | -1.287 | -1.284 | -0.009 |
| 60 | -8.399 | 599.9 | -7.958 | -2.707 | -2.701 | -0.015 |
| 75 | -29.894 | 300.3 | -23.950 | -20.300 | -14.128 | +11.870 |
| 90 | -19.983 | 599.9 | -18.946 | -2.504 | +0.920 | -5.954 |

## Interpretation

The L22-alone top-baffle target contains a very deep 75-degree low-frequency null that is much deeper than the measured nude L22 polar at the same radius. The multi-driver top capture does not reproduce that 75-degree depth; it shifts the strongest low-frequency high-angle attenuation toward 90 degrees. That does not invalidate the L22-alone target, but it makes the 75-degree null a target-quality and repeatability item before treating it as decisive proof of source/BEM failure.

## Files

- `target_null_at_worst_frequency.csv`: same-frequency per-angle measured table.
- `target_null_angle_summary.csv`: 300-600 Hz per-angle summary.
- `plots/target_nude_combo_norm_curves.png`: normalized curves at 60/75/90 deg.
- `plots/l22_target_minus_nude_transfer.png`: measured normalized baffle transfer heatmap.
