# L22MG Measurement Geometry Provenance

This report audits where the measurement geometry is stored in the HDF5 files used by the L22MG workflow.

- HDF5 write mode: enabled.
- Structured attrs checked: `measurement_distance_m`, `measurement_height_m`, and `measurement_height_reference`.
- Note-derived fields are parsed from each angle group's `notes` attr and promoted only when the structured attr is missing or an explicit overwrite is requested.

## Summary

| target | HDF5 | group | angles | distance final | distance status | height reference final | height status | wrote attrs |
| --- | --- | --- | ---: | --- | --- | --- | --- | ---: |
| `juan_l22mg_naked_front` | `output/data/polar_data_juan_baffleless.h5` | `L22MG (nude)/angles` | 7 | 0.5 m | matches_expected | l22mg | present | 0 |
| `juan_l22mg_naked_rear` | `output/data/polar_data_juan_baffleless.h5` | `L22MG (nude)/rear_angles` | 7 | 0.5 m | matches_expected | l22mg | present | 0 |
| `juan_l22mg_top_front` | `output/data/polar_data_juan_lx521_top_raw.h5` | `L22MG (LX521 top raw)/angles` | 7 | 0.5 m | matches_expected | l22mg | matches_expected | 0 |
| `juan_l22mg_top_rear` | `output/data/polar_data_juan_lx521_top_raw.h5` | `L22MG (LX521 top raw)/rear_angles` | 7 | 0.5 m | matches_expected | l22mg | matches_expected | 0 |
| `juan_l10neo_top_front` | `output/data/polar_data_juan_lx521_top_raw.h5` | `L10NEO (LX521 top raw)/angles` | 7 | 0.5 m | matches_expected | l22mg | matches_expected | 0 |
| `juan_l10neo_top_rear` | `output/data/polar_data_juan_lx521_top_raw.h5` | `L10NEO (LX521 top raw)/rear_angles` | 7 | 0.5 m | matches_expected | l22mg | matches_expected | 0 |
| `juan_l22_l10_tweeters_top_front` | `output/data/polar_data_juan_lx521_top_raw.h5` | `L22MG+L10NEO+Tweeters (LX521 top raw)/angles` | 7 | 0.5 m | matches_expected | l22mg | matches_expected | 0 |
| `juan_l22_l10_tweeters_top_rear` | `output/data/polar_data_juan_lx521_top_raw.h5` | `L22MG+L10NEO+Tweeters (LX521 top raw)/rear_angles` | 7 | 0.5 m | matches_expected | l22mg | matches_expected | 0 |
| `andres_l22mg_published_parity` | `output/data/polar_data_andres_early_peak_legacy.h5` | `L22MG/angles` | 10 | 1 m | matches_expected | um | matches_expected | 0 |
| `andres_l22mg_legacy_strongest_diagnostic` | `output/data/polar_data_andres.h5` | `L22MG/angles` | 10 | 1 m | matches_expected | um | matches_expected | 0 |

## Interpretation

- Juan's naked L22MG front and rear measurements are structured as 0.50 m source-radius measurements in `output/data/polar_data_juan_baffleless.h5`.
- Juan's LX521 top-baffle raw L22MG, L10NEO, and multi-driver captures are structured as 0.50 m measurements at LM/L22MG height in `output/data/polar_data_juan_lx521_top_raw.h5`.
- Andres' canonical published-parity L22MG target is structured as 1.0 m horizontal radius at UM height (`measurement_height_reference=um`).
- These attrs are provenance for geometry setup only; they do not change the published-parity pressure data or make diagnostic target files acceptance targets.

Files: `measurement_geometry_attrs.csv`.
