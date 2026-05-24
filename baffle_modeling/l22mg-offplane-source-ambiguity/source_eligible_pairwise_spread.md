# L22MG Eligible Source Off-Plane Spread

This targeted report compares only Juan-CV `recommended_juan_only` finite source cases. It does not read Andres data and does not change the source fit.

- Juan HDF5: `output/data/polar_data_juan_baffleless.h5`.
- Source-fit/evaluation band: 300-1200 Hz at 24 points/octave.
- Equivalent-source fit radius: 0.75 m.
- Observer radius: 1 m.
- Off-plane observer z offset: 165.0 mm.
- Eligible source-CV cases: modal-full4-svd, profile-ring-full-svd, profile-ring-full, profile-ring-compact-svd, profile-ring-compact, modal-full2.

## Worst Eligible Spread

| case | worst recommended reference | RMS 70-90 | max loc |
| --- | --- | ---: | --- |
| modal-full4-svd | modal-full2 | 8.797 | 1069 Hz / 90 deg |
| profile-ring-full-svd | modal-full2 | 13.283 | 925 Hz / 90 deg |
| profile-ring-full | modal-full2 | 13.283 | 925 Hz / 90 deg |
| profile-ring-compact-svd | modal-full2 | 13.284 | 925 Hz / 90 deg |
| profile-ring-compact | modal-full2 | 13.284 | 925 Hz / 90 deg |
| modal-full2 | profile-ring-compact-svd | 13.284 | 925 Hz / 90 deg |

## Worst Surface Points

Rows locate the largest pointwise normalized-polar spread across the eligible source ensemble at Andres' UM-height observer plane.

| frequency | angle | max pairwise spread | low/high cases |
| ---: | ---: | ---: | --- |
| 925 Hz | 90 deg | 35.367 dB | `profile-ring-compact-svd` / `modal-full2` |
| 952 Hz | 90 deg | 34.943 dB | `profile-ring-compact-svd` / `modal-full2` |
| 899 Hz | 90 deg | 33.843 dB | `profile-ring-compact-svd` / `modal-full2` |
| 980 Hz | 90 deg | 32.429 dB | `profile-ring-compact-svd` / `modal-full2` |
| 873 Hz | 90 deg | 32.301 dB | `profile-ring-compact-svd` / `modal-full2` |
| 1069 Hz | 90 deg | 31.497 dB | `modal-full4-svd` / `modal-full2` |
| 849 Hz | 90 deg | 31.374 dB | `profile-ring-full` / `modal-full2` |
| 824 Hz | 90 deg | 30.882 dB | `profile-ring-full` / `modal-full2` |
| 801 Hz | 90 deg | 30.534 dB | `profile-ring-full` / `modal-full2` |
| 778 Hz | 90 deg | 30.126 dB | `profile-ring-full` / `modal-full2` |

## Interpretation

A large value means Juan's horizontal naked polars still allow materially different UM-height incident fields even within finite source models that pass the current Juan-CV thresholds.
The strict validation gate prefers this ensemble spread over the legacy wide-SVD-relative spread when this CSV is present.

## Files

- `source_eligible_pairwise_offplane_summary.csv` contains all case/reference comparisons by angle group.
- `source_eligible_pairwise_spread_surface.csv` contains the pointwise max pairwise spread across the eligible source ensemble.
- `plots/source_eligible_pairwise_offplane_70_90.png` plots the worst 70-90 deg spread per case.
- `plots/source_eligible_pairwise_spread_contour.png` plots the pointwise spread surface.
