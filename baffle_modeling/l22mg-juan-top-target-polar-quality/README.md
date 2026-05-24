# L22MG Target Polar Quality Audit

This report flags deep measured-target normalized-polar nulls that can dominate validation metrics.
It does not exclude data, fit to Andres, or change acceptance gates.

- Band: 300-600 Hz.
- Deep-null threshold: -25.0 dB re 0 deg.
- Same-angle local context window: 0.167 octaves.
- Interpretation: a flagged row is not proof of a bad measurement; it is a target-quality warning where a deep measured null can overwhelm model validation.
- Same-angle local contrast measures frequency-local sharpness; adjacent-angle contrast measures angular isolation at the null frequency.
- A low same-angle contrast with high adjacent-angle contrast means a broad-in-frequency but angularly isolated measured null.

## juan-top-l22

- Deep-null clusters: 1.

| rank | angle | min Hz | min norm | width Hz | local contrast | adjacent contrast |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 75 | 300.3 | -29.9 | 83.1 | -0.6 | 17.3 |

CSV: `target_polar_quality_summary.csv`.
