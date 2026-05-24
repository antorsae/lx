# L22MG Source-Support Hotspot Sweep

This diagnostic varies only the Juan-fitted source support and evaluates a fixed mounted-baffle BEM geometry at selected high-angle hotspot frequencies.

- No target-derived source, rear, or scalar correction is fitted.
- Target data are used only to score the already Juan-fitted candidate source supports.
- Target: `output/data/polar_data_juan_lx521_top_raw.h5` / `L22MG (LX521 top raw)` (Juan L22MG top baffle).
- Frequencies tested: 300.293 Hz, 304.321 Hz.
- BEM geometry: Wmax 305 mm, thickness 18.3 mm, Andres-mounted passive UM face and solid tweeter patches.
- Mic geometry: horizontal radius 0.5 m, z offset 0 mm relative to L22 center.
- BEM mesh: method delaunay-local, h 42 mm, boundary/local h 32/32 mm, panels 832.
- BEM quadrature: order 7, near order 7, near subdivisions 1, near distance factor 2.5.
- Source fit grid: 300-1200 Hz, 24 points/octave, Juan radius 0.5 m, rear phase `dipole`.
- Acoustic-center offsets swept: -20 mm, 0 mm, 20 mm.
- H1659 profile/modal ring candidates included: True.
- Candidates swept: 41 source supports x 3 acoustic-center offsets.

## Best Hotspot Candidate

- Candidate: `d65_r0-25-50-75-95_svd1e-6_ac-20mm`.
- Hotspot score: 7.361 dB.
- RMS 0-60 deg: 1.434 dB.
- RMS 70-90 deg: 7.361 dB.
- RMS 80-90 deg: 8.335 dB.
- Current-source comparison: current 70-90 deg RMS is 14.046 dB, so the best candidate improves this hotspot by 6.685 dB, but remains far from an acceptance-grade residual.

## Ranked Candidates

| candidate | model | profile | score | RMS 0-60 | RMS 70-90 | RMS 80-90 | front depth/radii | rear depth/radii | SVD rcond |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| d65_r0-25-50-75-95_svd1e-6_ac-20mm | split-discrete | symmetric_depth_rings | 7.361 | 1.434 | 7.361 | 8.335 | 65.0 mm / 0 25 50 75 95 | 65.0 mm / 0 25 50 75 95 | 1e-06 |
| d65_r0-25-50-75-95_svd1e-5_ac-20mm | split-discrete | symmetric_depth_rings | 7.371 | 1.435 | 7.371 | 8.295 | 65.0 mm / 0 25 50 75 95 | 65.0 mm / 0 25 50 75 95 | 1e-05 |
| d65_r0-35-70-95_svd1e-5_ac-20mm | split-discrete | symmetric_depth_rings | 8.003 | 1.130 | 8.003 | 7.697 | 65.0 mm / 0 35 70 95 | 65.0 mm / 0 35 70 95 | 1e-05 |
| d65_r0-35-70-95_svd1e-6_ac-20mm | split-discrete | symmetric_depth_rings | 8.012 | 1.126 | 8.012 | 7.727 | 65.0 mm / 0 35 70 95 | 65.0 mm / 0 35 70 95 | 1e-06 |
| d65_r0-25-50-75-95_svd1e-6_ac+0mm | split-discrete | symmetric_depth_rings | 9.834 | 1.150 | 9.834 | 2.131 | 65.0 mm / 0 25 50 75 95 | 65.0 mm / 0 25 50 75 95 | 1e-06 |
| d65_r0-25-50-75-95_svd1e-5_ac+0mm | split-discrete | symmetric_depth_rings | 9.856 | 1.147 | 9.856 | 2.149 | 65.0 mm / 0 25 50 75 95 | 65.0 mm / 0 25 50 75 95 | 1e-05 |
| d65_r0-35-70-95_svd1e-6_ac+0mm | split-discrete | symmetric_depth_rings | 10.198 | 0.921 | 10.198 | 2.647 | 65.0 mm / 0 35 70 95 | 65.0 mm / 0 35 70 95 | 1e-06 |
| d65_r0-35-70-95_svd1e-5_ac+0mm | split-discrete | symmetric_depth_rings | 10.210 | 0.920 | 10.210 | 2.657 | 65.0 mm / 0 35 70 95 | 65.0 mm / 0 35 70 95 | 1e-05 |
| h1659-modal-full2-az24-reg1e-06_ac+20mm | split-profile-ring | h1659-modal-full-2 | 10.915 | 0.704 | 10.915 | 2.594 | nan mm /  | nan mm /  |  |
| d55_r0-25-50-75-95_svd1e-6_ac+0mm | split-discrete | symmetric_depth_rings | 11.740 | 0.436 | 11.740 | 1.745 | 55.0 mm / 0 25 50 75 95 | 55.0 mm / 0 25 50 75 95 | 1e-06 |
| d55_r0-25-50-75-95_svd1e-5_ac+0mm | split-discrete | symmetric_depth_rings | 11.743 | 0.441 | 11.743 | 1.746 | 55.0 mm / 0 25 50 75 95 | 55.0 mm / 0 25 50 75 95 | 1e-05 |
| d55_r0-35-70-95_svd1e-5_ac+0mm | split-discrete | symmetric_depth_rings | 11.851 | 0.352 | 11.851 | 2.283 | 55.0 mm / 0 35 70 95 | 55.0 mm / 0 35 70 95 | 1e-05 |
| d55_r0-35-70-95_svd1e-6_ac+0mm | split-discrete | symmetric_depth_rings | 11.852 | 0.350 | 11.852 | 2.288 | 55.0 mm / 0 35 70 95 | 55.0 mm / 0 35 70 95 | 1e-06 |
| d45_r0-35-70-95-110_svd1e-5_ac+0mm | split-discrete | symmetric_depth_rings | 12.028 | 1.806 | 11.723 | 0.738 | 45.0 mm / 0 35 70 95 110 | 45.0 mm / 0 35 70 95 110 | 1e-05 |
| d45_r0-35-70-95-110_svd1e-6_ac+0mm | split-discrete | symmetric_depth_rings | 12.028 | 1.806 | 11.723 | 0.738 | 45.0 mm / 0 35 70 95 110 | 45.0 mm / 0 35 70 95 110 | 1e-06 |
| asym_f55r45_front25_rear35_svd1e-5_ac+0mm | split-discrete | asymmetric_depth_rings | 12.958 | 0.960 | 12.958 | 0.915 | 55.0 mm / 0 25 50 75 95 | 45.0 mm / 0 35 70 95 | 1e-05 |

Interpretation: this is a source-support diagnostic, not an acceptance proof. A useful candidate must improve the 70-90 deg hotspot without damaging 0-60 deg agreement; it still needs a full 300-1200 Hz BEM run and mesh-convergence check.

Files: `source_support_hotspot_sweep.csv`, `top_candidates.csv`, `plots/source_support_hotspot_summary.png`, and `plots/best_candidate_curves.png`.
