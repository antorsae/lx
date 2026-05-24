# Juan L22MG Top-Baffle Front/Rear Consistency

This report audits measured Juan HDF5 data only. It does not run BEM, refit the source, exclude data, or change validation gates.

- Top-baffle HDF5: `output/data/polar_data_juan_lx521_top_raw.h5`.
- L22 target driver: `L22MG (LX521 top raw)`.
- Nude reference HDF5: `output/data/polar_data_juan_baffleless.h5` / `L22MG (nude)`.
- Corroboration capture: `L22MG+L10NEO+Tweeters (LX521 top raw)`.
- Band: 300-600 Hz, high angles >= 60 deg.

## Key Finding

The deepest L22-alone front high-angle point is 300.293 Hz / 75 deg: front -29.894 dB, rear -9.455 dB, front-minus-rear -20.440 dB.
At that same point the measured front baffle transfer is -20.300 dB, rear transfer is +1.000 dB, and front-minus-rear transfer is -21.300 dB.
The L22+L10NEO+tweeters capture is +11.870 dB above the L22-alone front target at the same point, so it does not corroborate the full 75-degree front-null depth.
Over 300-600 Hz and angles >= 60 deg, L22 top front/rear normalized RMS difference is 10.733 dB; front-minus-rear baffle-transfer RMS is 10.890 dB; combo front/rear normalized RMS difference is 4.288 dB.

## Same-Frequency Table

| deg | top front | top rear | front-rear | front transfer | rear transfer | transfer delta | combo front | combo rear | combo-L22 front |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| 15 | -0.349 | -0.277 | -0.073 | -0.148 | +0.234 | -0.382 | -0.483 | -0.325 | -0.133 |
| 30 | -1.591 | -1.095 | -0.496 | -0.641 | -0.396 | -0.245 | -1.749 | -1.186 | -0.158 |
| 45 | -3.788 | -2.507 | -1.281 | -1.380 | -0.092 | -1.287 | -3.964 | -2.755 | -0.176 |
| 60 | -7.750 | -4.946 | -2.805 | -2.841 | +0.259 | -3.100 | -8.125 | -5.225 | -0.374 |
| 75 | -29.894 | -9.455 | -20.440 | -20.300 | +1.000 | -21.300 | -18.024 | -9.920 | +11.870 |
| 90 | -17.408 | -26.319 | +8.910 | +5.056 | -1.813 | +6.869 | -21.225 | -27.743 | -3.817 |

## Angle Summary

| deg | L22 F/R RMS | L22 F/R max | transfer F/R RMS | transfer F/R max | combo F/R RMS | combo-L22 front RMS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 15 | 0.046 | 0.073 | 0.219 | 0.382 | 0.163 | 0.128 |
| 30 | 0.380 | 0.496 | 0.318 | 0.451 | 0.522 | 0.115 |
| 45 | 1.014 | 1.281 | 1.229 | 1.445 | 1.099 | 0.103 |
| 60 | 2.294 | 2.805 | 2.794 | 3.100 | 2.403 | 0.167 |
| 75 | 13.693 | 20.440 | 14.736 | 21.300 | 5.986 | 7.502 |
| 90 | 12.362 | 15.423 | 11.437 | 15.353 | 3.681 | 4.514 |

## Interpretation

The L22-alone front 75-degree low-frequency null is not mirrored by the L22-alone rear measurement at the same frequency and angle. The rear side keeps a more nude-like 75-degree response and places the deepest low-frequency high-angle attenuation closer to 90 degrees. The multi-driver no-crossover capture also does not reproduce the full L22-alone 75-degree front-null depth.
This strengthens the target-quality and repeatability warning for the 300 Hz / 75 deg validation blocker. It does not remove the point from validation, but it means a future source/BEM change should not be accepted just because it target-fits that isolated front-side null.

## Files

- `front_rear_consistency_at_deep_front_null.csv`: same-frequency per-angle measured table.
- `front_rear_consistency_summary.csv`: per-angle 300-600 Hz front/rear consistency summary.
- `plots/front_rear_norm_curves.png`: normalized front/rear curves at 60/75/90 deg.
- `plots/front_minus_rear_baffle_transfer_delta.png`: front-minus-rear baffle-transfer heatmap.
