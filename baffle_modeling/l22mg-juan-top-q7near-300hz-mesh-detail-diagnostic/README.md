# L22MG Validation Decomposition Diagnostic

This report reads the exported `/validation_decomposition` HDF5 group. It does not refit, rerun BEM, or add any scalar/source/rear correction.

- Cases: focused_q7near_300hz.
- Error sign: positive normalized error means the model is less attenuated than the validation target at that angle.
- `target/model minus nude norm` is the mounted normalized response minus the synthetic-nude normalized response at the same validation mic geometry.
- CSV: `decomposition_band_summary.csv`, `decomposition_angle_summary.csv`, `decomposition_worst_points.csv`.
- Plots: `plots/*_norm_curves.png` and `plots/*_error_maps.png`.

## 300-1200 Hz Summary

| case | angle group | norm RMS | norm max | norm max loc | transfer RMS | target-nude mean | model-nude mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| focused_q7near_300hz | 0_60 | 1.209 | 2.424 | 300/60 | 1.209 | -0.845 | -0.030 |
| focused_q7near_300hz | 70_90 | 13.944 | 19.806 | 300/75 | 13.944 | -7.212 | 3.393 |
| focused_q7near_300hz | 80_90 | 1.553 | 1.589 | 304/90 | 1.553 | 4.703 | 6.256 |
| focused_q7near_300hz | 90 | 1.553 | 1.589 | 304/90 | 1.553 | 4.703 | 6.256 |

## Worst Points

| case | rank | Hz | deg | target norm | model norm | synthetic nude norm | target-nude | model-nude | norm err | transfer err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| focused_q7near_300hz | 1 | 300.3 | 75 | -29.894 | -10.088 | -10.619 | -19.275 | 0.531 | +19.806 | +19.815 |
| focused_q7near_300hz | 2 | 300.7 | 75 | -29.869 | -10.089 | -10.620 | -19.249 | 0.530 | +19.779 | +19.787 |
| focused_q7near_300hz | 3 | 301.0 | 75 | -29.843 | -10.090 | -10.621 | -19.222 | 0.530 | +19.752 | +19.758 |
| focused_q7near_300hz | 4 | 301.4 | 75 | -29.817 | -10.091 | -10.622 | -19.195 | 0.530 | +19.725 | +19.730 |
| focused_q7near_300hz | 5 | 301.8 | 75 | -29.791 | -10.092 | -10.623 | -19.168 | 0.530 | +19.698 | +19.701 |
| focused_q7near_300hz | 6 | 302.1 | 75 | -29.765 | -10.093 | -10.623 | -19.142 | 0.530 | +19.672 | +19.672 |
| focused_q7near_300hz | 7 | 302.5 | 75 | -29.739 | -10.094 | -10.624 | -19.115 | 0.530 | +19.645 | +19.644 |
| focused_q7near_300hz | 8 | 302.9 | 75 | -29.713 | -10.095 | -10.625 | -19.088 | 0.530 | +19.618 | +19.615 |

## Interpretation

At the worst current point, the validation target is much deeper than both the synthetic nude reference and the mounted model: 300.3 Hz / 75 deg has target -29.9 dB, synthetic nude -10.6 dB, and model -10.1 dB. Relative to synthetic nude, the target is -19.3 dB while the model is +0.5 dB. The modeled baffle/incident-field interaction is filling the high-angle null where the measured mounted speaker deepens it.

If this measured null is repeatable clean direct sound, it points at source/driver-surface and local scattering physics rather than scalar alignment. If the target-polar quality audit flags the same region as a deep high-sensitivity null, treat it as a measurement/window corroboration target before using it as decisive acceptance evidence.

## Mesh Convergence Context

This section reads the artifact's stored BEM convergence CSV. It does not rerun the solver.

| case | mesh/ref h mm | norm RMS | norm max | max Hz/deg | through-80 max | through-80 Hz/deg | per-angle detail CSV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| focused_q7near_300hz | 42/36 | 0.562 | 1.37 | 300/90 | 0.572 | 300/75 | yes |

If `per-angle detail CSV` is `no`, this artifact was generated before per-frequency/angle `bem_mesh_convergence_detail.csv` export existed; rerun the same case to inspect the full mesh-delta surface.

## Per-Angle Mesh Detail: focused_q7near_300hz

Nearest mesh-detail frequency to the worst validation point is 300.293 Hz. Mesh deltas are coarse mesh minus reference mesh, normalized as SPL(theta)-SPL(0).

| deg | target norm | model norm | nude norm | norm err | mesh norm delta | mesh abs delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 0.000 | 0.000 | +0.000 | +0.000 | -0.218 |
| 15 | -0.349 | -0.274 | -0.262 | +0.076 | -0.007 | -0.225 |
| 30 | -1.591 | -1.116 | -1.084 | +0.475 | -0.028 | -0.246 |
| 45 | -3.788 | -2.673 | -2.609 | +1.115 | -0.073 | -0.291 |
| 60 | -7.750 | -5.326 | -5.277 | +2.424 | -0.186 | -0.404 |
| 75 | -29.894 | -10.088 | -10.619 | +19.806 | -0.572 | -0.790 |
| 90 | -17.408 | -15.891 | -22.201 | +1.518 | -1.374 | -1.592 |

## Case Metadata

| case | source | solver | Juan radius m | validation mic | z offset mm | alignment |
| --- | --- | --- | ---: | --- | ---: | --- |
| focused_q7near_300hz | split_axisymmetric_ring_source | dense_neumann_bem | 0.5 | l22 | 0 | one scalar gain and one delay only; no source or rear level correction |
