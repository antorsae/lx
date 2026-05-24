# Juan L22MG Top-Baffle Acceptance Scorecard

This scorecard selects the strongest current Juan top-baffle BEM evidence from `docs/l22mg-validation-gate-summary/validation_gate_summary.csv` and expands the acceptance gates into one row per criterion.

- Selected artifact: `compact2-h32-h21-h28-h18-targetavg7-meshtarget-conv-smoke`.
- Artifact path: `output/l22mg-bem-juan-top-h1659-modal-compact2-solid-passive-h32-h21-h28-h18-q7near-targetavg7-meshtarget-conv-smoke`.
- Overall status: `fail`.
- Normalized polar RMS 300-1200 Hz: 3.920 dB all-angle, 0.875 dB through 60 deg, 3.491 dB through 80 deg.
- Mesh movement: 0.672 dB max over 300-1200 Hz.
- Blocking criteria: overall_acceptance, normalized_polar_shape, mesh_convergence, target_measurement_quality, target_null_repeatability, target_front_rear_consistency, target_polar_quality, passive_geometry.
- Target-warning-separated diagnostic: excluding the stored target-quality cluster gives 3.461 dB all-angle RMS and 2.858 dB through 80 deg, but this is diagnostic only and does not change acceptance.
- Separate width-sweep deliverable: `output/l22mg-bem-juan-top-h1659-modal-compact2-current-width-sweep-h42-h28-q7near-targetavg7` reports widths 205,255,305,355,405 mm with required 205/305/405 present `True`; status `qualitative_only` because the 305 mm baseline remains above the <=1.5 dB shape target.

The selected row is chosen from current 0.50 m Juan source/target rows. Full 300-1200 Hz coverage, target-angle parity, wavelength mesh-target sizing, source-CV pass, and lower mesh movement are preferred before lower RMS rows that lack those properties.

| criterion | status | blocks acceptance | evidence |
| --- | --- | --- | --- |
| overall_acceptance | fail | yes | Selected artifact `compact2-h32-h21-h28-h18-targetavg7-meshtarget-conv-smoke` from `output/l22mg-bem-juan-top-h1659-modal-compact2-solid-passive-h32-h21-h28-h18-q7near-targetavg7-meshtarget-conv-smoke`; normalized polar RMS 3.920 dB over 300-1200 Hz. |
| target_provenance | pass | no | HDF5 `output/data/polar_data_juan_lx521_top_raw.h5`, driver `L22MG (LX521 top raw)`, hypothesis `juan_baffleless_to_juan_top_baffle_l22mg_raw`. |
| measurement_geometry | pass | no | Juan source radius 0.500 m; target distance 0.500 m; mic reference `l22`, horizontal radius 0.500 m, z offset 0.000 mm. |
| angle_grid | pass | no | Policy `target_hdf5_angles`; model angles [0, 15, 30, 45, 60, 75, 90]; target angles [0, 15, 30, 45, 60, 75, 90]; interpolation used `False`. |
| validation_alignment_policy | pass | no | Gain 1.362 dB and delay 0.000 ms; scalar gain count 1, delay count 1, angle-dependent gain count 0, band-specific gain count 0, rear/source level corrections 0.000 / 0.000 dB. exactly one global scalar gain and one global delay are allowed for validation alignment; no angle-dependent, band-specific, rear-level, or source-level dB corrections are applied |
| coverage | pass | no | Covered band 300.2929915487766-1199.7071272432804 Hz; full coverage `True`. |
| normalized_polar_shape | fail | yes | All-angle RMS 3.920 dB vs 1.5 dB target; through-60 0.875 dB; through-80 3.491 dB; worst +19.102 dB at 300 Hz / 75 deg. |
| target_quality_masked_shape_diagnostic | diagnostic_only | no | Diagnostic only, not an acceptance metric: excluding the stored target-polar-quality cluster at 75 deg from 300.293-383.423 Hz removes 228 points and gives all-angle RMS 3.461 dB, through-60 0.875 dB, through-80 2.858 dB; remaining worst +13.956 dB at 384 Hz / 75 deg. Full-data acceptance remains unchanged. |
| mesh_convergence | fail | yes | Max normalized-polar movement 0.672 dB vs 0.5 dB gate. |
| mesh_target_wavelength_sizing | pass | no | Reference 1200 Hz; actual broad/boundary/local h = 32.000/21.000/21.000 mm; limits broad lambda/6 47.639 mm and boundary/local lambda/10 28.583 mm. BEM mesh spacing is within the recorded wavelength sizing targets: actual BEM mesh spacing is within lambda/6 broad and lambda/10 boundary/local targets |
| bem_linear_solver | pass | no | Solver `direct`, max relative residual 1.049e-15. |
| source_cv | pass | no | Source model id `h1659-modal-compact2-az24-reg1e-06`; Juan-only held-out side-worst 3.298 dB; recommendation `recommended_juan_only`. |
| source_offplane_generalization | pass | no | not required for acceptance because validation target is on the LM/L22MG height; Andres UM-height off-plane ambiguity remains diagnostic |
| target_measurement_quality | not_proven | yes | Juan target-null repeatability: Juan measured target has -20.300 dB top-minus-nude transfer at 300.293 Hz / 75 deg; multi-driver top capture is +11.870 dB relative to L22-alone target; this is also the artifact's worst normalized-polar residual; Juan front/rear consistency: Juan measured front-minus-rear baffle-transfer delta is -21.300 dB at 300.293 Hz / 75 deg; front-minus-rear normalized delta is -20.440 dB; multi-driver top capture is +11.870 dB relative to L22-alone front; this is also the artifact's worst normalized-polar residual; target polar quality: measured target has -29.894 dB normalized null at 300.293 Hz / 75 deg; cluster width 83.1 Hz; adjacent-angle contrast 17.315 dB; same-angle local contrast -0.589 dB; this is also the artifact's worst normalized-polar residual |
| target_null_repeatability | warning | yes | Juan measured target has -20.300 dB top-minus-nude transfer at 300.293 Hz / 75 deg; multi-driver top capture is +11.870 dB relative to L22-alone target; this is also the artifact's worst normalized-polar residual |
| target_front_rear_consistency | warning | yes | Juan measured front-minus-rear baffle-transfer delta is -21.300 dB at 300.293 Hz / 75 deg; front-minus-rear normalized delta is -20.440 dB; multi-driver top capture is +11.870 dB relative to L22-alone front; this is also the artifact's worst normalized-polar residual |
| target_polar_quality | warning | yes | measured target has -29.894 dB normalized null at 300.293 Hz / 75 deg; cluster width 83.1 Hz; adjacent-angle contrast 17.315 dB; same-angle local contrast -0.589 dB; this is also the artifact's worst normalized-polar residual |
| passive_geometry | not_proven | yes | Juan L22MG top-baffle target is L22-only; the validation target HDF5 records `passive_state_status=unused_um_tweeter_state_unrecorded` and `passive_state_acceptance_use=current_l22_target_but_passive_geometry_not_proven`, which is an explicit unknown, not evidence for open holes, covered patches, or mounted inactive drivers. See `docs/l22mg-juan-top-passive-state-audit/README.md` and diagnostic sensitivity `docs/l22mg-juan-top-passive-state-band-sensitivity/README.md` |
| driver_stl_geometry_provenance | pass | no | STL `linkwitz/H1659-08_U22REX_P-SL_driver.stl` status `present`; OD 221.000 mm, depth 90.600 mm, front proud height 0.600 mm, rear depth 90.000 mm, flange overlap past L22 cutout radius 15.500 mm. STL dimensions are recorded for provenance of H1659-derived face/frame/source geometry; recording these dimensions is not a raw-STL solve, feature-preserving remesh, or acceptance proof |
| audit_outputs | pass | no | Contour plot present `True`; decomposition CSV `pass`; HDF5 group `pass`. |
| width_sweep_status | qualitative_only | no | Width sweep CSV `output/l22mg-bem-juan-top-h1659-modal-compact2-solid-passive-h32-h21-h28-h18-q7near-targetavg7-meshtarget-conv-smoke/width_sweep_metrics.csv`; interpretation: The 305 mm baseline did not pass all current validation gates, so this width sweep is diagnostic only and cannot be read as accepted LX521 width behavior. Required widths present `False`; widths 305; missing required widths `205,405`. Baseline 305 mm 300-1200 metrics: F90-F0 -18.108 dB, DI 5.590 dB, beamwidth 112.410 deg, sensitivity 75.047 dB, rear/front symmetry 1.540 dB. Separate qualitative width-sweep deliverable `output/l22mg-bem-juan-top-h1659-modal-compact2-current-width-sweep-h42-h28-q7near-targetavg7` reports widths 205,255,305,355,405 mm; required 205/305/405 present `True`; recommended 205/255/305/355/405 present `True`; current-radius evidence `True`; deliverable status `qualitative_only`; 305 mm all-angle RMS 4.214 dB. This separate sweep remains qualitative and does not change the selected artifact acceptance status. |

Interpretation:

- The current Juan top-baffle path is not accepted.
- The selected row is the strongest current generated evidence because it uses the canonical Juan target, exact target HDF5 angles, 0.50 m source/target geometry, full 300-1200 Hz coverage, mesh-target-compliant sizing, direct dense-BEM residual metadata, source-CV pass, contour output, and validation decomposition output.
- The remaining blockers are the all-angle shape miss at the 300 Hz / 75 deg target null, all-angle mesh convergence above 0.5 dB, target measurement-quality warnings around that null, and unresolved L22-only passive-state provenance.
- The target-warning-separated diagnostic is provided to localize the blocker; it does not exclude data from acceptance and must not be used as a scalar or ad hoc correction.
- Source off-plane generalization is not an acceptance blocker for this current target because Juan top-baffle validation is at LM/L22MG height; Andres UM-height ambiguity remains a legacy diagnostic.
- Width sweeps remain qualitative until the 305 mm baseline passes acceptance; the separate current-radius width deliverable already reports the required 205/305/405 mm rows.

CSV: `juan_acceptance_scorecard.csv`.
Diagnostic CSV: `target_quality_masked_shape_diagnostic.csv`.
