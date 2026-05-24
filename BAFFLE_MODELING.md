# L22MG Baffle Modeling Handoff

This document is a restart prompt and cleanup guide for the failed L22MG/LX521 top-baffle modeling effort.

The goal was not achieved. We tried to predict the mounted LX521 top-baffle polar response of the same physical SEAS L22MG driver from:

- the naked/baffleless L22MG measurements,
- the driver structure and H1659/U22REX-derived geometry,
- the LX521 top-baffle geometry,
- finite-baffle acoustic simulation.

The validation target was Juan's same-room, same-driver top-baffle measurement. The naked and baffle-mounted measurements were both made at 50 cm and at the same L22MG/LM height. The intended model should therefore learn or infer the baffle transfer from the naked L22MG capture to the same L22MG mounted in the baffle, without using source/rear scalar hacks or target-derived corrections.

The current result is a failure: no model got close enough to the real mounted-baffle measurement over the acceptance band.

## Current Acceptance Target

Primary target:

- HDF5: `output/data/polar_data_juan_lx521_top_raw.h5`
- Driver/group: `L22MG (LX521 top raw)`
- Geometry: 0.50 m measurement distance, L22MG/LM measurement height.
- Angles: front 0, 15, 30, 45, 60, 75, 90 deg.
- Frequency scope used for main acceptance: 300-1200 Hz.
- Primary metric: per-frequency normalized polar shape, `SPL(theta) - SPL(0 deg)`.
- Acceptance target used in the repo: normalized polar RMS <= 1.5 dB over 300-1200 Hz, with full target angle grid, full coverage, no angle-dependent gain, no band-specific gain, no source/rear scalar correction, and mesh/solver provenance.

Primary naked source input:

- HDF5: `output/data/polar_data_juan_baffleless.h5`
- Driver/group: `L22MG (nude)`
- Geometry: 0.50 m measurement distance, L22MG/LM measurement height.
- Front and rear naked polars exist at the same 0/15/30/45/60/75/90 deg grid.

Geometry/provenance references:

- `baffle_modeling/l22mg-measurement-geometry-provenance/README.md`
- `baffle_modeling/l22mg-current-status/README.md`
- `baffle_modeling/l22mg-juan-acceptance-scorecard/README.md`

## Result Summary

No current model is accepted.

Best current populated-silent rerank and Wmax summaries show the same pattern:

| model / family | status | representative full-band RMS | representative through-60 RMS | main failure |
| --- | --- | ---: | ---: | --- |
| H1659 modal compact2 finite source | failed | about 4.2 dB | about 1.0 dB | misses 300.293 Hz / 75 deg null by about +18 to +19 dB |
| H1659 modal full2 finite source | failed | about 4.2 dB | about 1.0 dB | same low-frequency high-angle null miss |
| axisymmetric directivity diagnostic | failed / diagnostic | about 4.1 to 4.5 dB | about 1.4 to 1.9 dB | sometimes ranks well, but source is not physically accepted and disagrees directionally in Wmax sweep |
| active piston / active surface / split active surface | failed | about 6 to 10 dB | about 1.8 to 4.1 dB | adds larger residuals elsewhere |
| feature proxy / rear basket / physical diaphragm attempts | failed | typically worse | typically worse | did not close the target null or source evidence gates |
| source support / acoustic-center probes | diagnostic only | can reduce local high-angle miss but not enough | mixed | remains far from 1.5 dB and risks target fitting |

The strongest recent report is:

- `baffle_modeling/l22mg-juan-top-populated-silent-self-contained-report/README.md`

It reports:

- best literal/full-band current rank: `axisym directivity populated`, about 4.088 dB RMS, diagnostic only;
- best current target-above-minus-20 diagnostic rank: `modal full2 populated`, 1.587 dB, still above the 1.5 dB target and not an acceptance mask;
- conservative baseline: `compact2 populated`, 1.594 dB under the same diagnostic mask and 4.219 dB full-band.

The latest Wmax shoulder sweep is:

- `baffle_modeling/l22mg-bem-juan-top-populated-silent-wmax-model-agreement/README.md`

That sweep shows the three models do not agree directionally:

- `compact2` and `modal_full2`: widening Wmax from B toward C/D improves several metrics;
- `axisym_directivity`: widening worsens metrics and prefers actual B.

This is useful directional evidence, but not accepted baffle behavior because the baseline model still fails.

## Dominant Failure

The dominant blocker is the Juan L22-only top-baffle low-frequency high-angle null:

- Frequency: 300.293 Hz
- Angle: 75 deg front
- Target normalized level: -29.894 dB
- Naked L22 normalized level at same point: -9.594 dB
- Measured target-minus-nude transfer: -20.300 dB
- Typical model residual: about +18 to +20 dB at the same point

Measured-data audits show this is not just a model-output quirk:

- `baffle_modeling/l22mg-juan-top-target-null-audit/README.md`
- `baffle_modeling/l22mg-juan-top-front-rear-consistency/README.md`
- `baffle_modeling/l22mg-juan-top-target-polar-quality/README.md`

Important target-only findings:

- The multi-driver top capture, `L22MG+L10NEO+Tweeters (LX521 top raw)`, is -18.024 dB at the same 300.293 Hz / 75 deg point, which is 11.870 dB shallower than the L22-alone target.
- The L22-alone rear top-baffle measurement is -9.455 dB at the same point, not -29.894 dB.
- Front-minus-rear at that point is -20.440 dB.
- The target-only quality audit finds one deep 300-600 Hz cluster at 75 deg, 83.1 Hz wide, about 17.3 dB deeper than adjacent angles.

Interpretation: the 75 deg null may be a real condition-specific mounted-baffle effect, but it is not strongly corroborated by the rear capture or the multi-driver capture. A future model should not declare success merely by fitting this isolated front-side null.

## What Was Tried

### Measurement And Timing Provenance

Done and should not be repeated unless new raw data arrive:

- Confirmed Juan naked and top-baffle L22MG measurements are 0.50 m and at L22MG/LM height.
- Confirmed the target angle grid is 0/15/30/45/60/75/90 deg and should be solved directly, not interpolated.
- Separated Juan same-room/same-driver target from Andres published-parity target.
- Added gate/report logic to avoid accidentally using stale 0.75 m source-radius metadata.
- Added target-null, front/rear consistency, polar-quality, and passive-state provenance warnings.

Do not spend another iteration retiming or reclassifying the Andres artifacts unless the goal changes. Andres published parity is a useful legacy data-path check, but it is not the current same-driver/same-height baffle target.

Key references:

- `baffle_modeling/l22mg-measurement-geometry-provenance/README.md`
- `baffle_modeling/l22mg-juan-top-target-null-audit/README.md`
- `baffle_modeling/l22mg-juan-top-front-rear-consistency/README.md`
- `baffle_modeling/l22mg-juan-top-target-polar-quality/README.md`
- `baffle_modeling/l22mg-target-robustness/README.md`

### Source Models

Tried source families include:

- split discrete monopole clouds;
- axisymmetric directivity-table diagnostics;
- split axisymmetric ring sources;
- H1659 profile-ring finite sources;
- H1659 modal compact/full sources;
- SVD-regularized variants;
- measured rear phase and ideal dipole rear phase variants;
- active-surface modal Neumann sources;
- physical-diaphragm source surfaces;
- coupled rear-basket source variants;
- bounded rear-filter variants;
- acoustic-center offsets;
- source support sweeps.

What not to redo:

- Do not rerun simple H1659 modal compact2/full2 variants expecting them to pass. They are already negative evidence.
- Do not rerun active piston or basic active-surface variants expecting improvement. They made broader residuals worse.
- Do not use a free rear scalar, source-level scalar, angle scalar, or band scalar. Those can fit target nulls but violate the source-evidence policy.
- Do not trust a low in-sample naked-source fit alone. Cross-validation and mounted-baffle evidence contradicted several apparently plausible sources.

Key references:

- `baffle_modeling/l22mg-source-model-cross-validation/README.md`
- `baffle_modeling/l22mg-source-candidate-decision/README.md`
- `baffle_modeling/l22mg-offplane-source-ambiguity/README.md`
- `baffle_modeling/l22mg-juan-top-lowfreq-source-support-sweep/README.md`

### Baffle And Passive Geometry

Tried geometry/passive variants include:

- actual LX521 top baffle;
- finite thickness baffle;
- populated-silent passive positions;
- solid passive patches;
- open passive holes;
- raised UM passive face;
- raised tweeter passive faces;
- baffle thickness changes;
- mic z offsets;
- acoustic-center offsets;
- Wmax width/shoulder sweeps;
- simplified driver face/frame/proxy features.

What not to redo:

- Do not rerun only open/closed passive holes as if that is likely to solve the failure. The sparse passive-state sensitivity runs did not close the low-frequency high-angle null.
- Do not treat Wmax widening as an accepted design result. Width direction is model-dependent and the baseline fails.
- Do not claim passive-state parity. Juan's L22-only top-baffle HDF5 records the unused UM/tweeter state as unrecorded.

Key references:

- `baffle_modeling/l22mg-juan-top-passive-state-audit/README.md`
- `baffle_modeling/l22mg-juan-top-passive-state-band-sensitivity/README.md`
- `baffle_modeling/l22mg-juan-top-lowfreq-geometry-sensitivity/README.md`
- `baffle_modeling/l22mg-bem-juan-top-populated-silent-wmax-model-agreement/README.md`

### BEM And Numerical Work

Tried numerical improvements include:

- dense exterior Neumann BEM;
- edge solver smoke paths;
- mesh target metadata and gates;
- Delaunay local refinement around edges/cutouts;
- q7 quadrature;
- near-panel quadrature;
- target normal-trace averaging;
- direct LU residual metadata;
- GMRES/matrix-free smoke comparisons;
- mesh convergence pairs from coarse to finer h/boundary/local meshes;
- localized lobe probes.

What not to redo:

- Do not assume the 75 deg null failure is primarily a linear-solver residual problem. Direct-solve residuals are tiny.
- Do not assume q7 near-panel quadrature alone fixes it. It does not.
- Do not assume the huge 75 deg residual is explained by the measured mesh delta alone. Focused diagnostics found the 75 deg validation residual around +19.8 dB while local mesh movement was about -0.57 dB; the bigger local mesh movement was at 90 deg where the validation residual was much smaller.

Still unresolved:

- all-angle mesh convergence remained above the strict 0.5 dB gate in several paths;
- higher-order/adaptive or accelerated BEM is still the serious path if numerical fidelity is to be pushed further;
- constant-panel dense BEM may still be insufficient for final acceptance, but it is not the only or most obvious explanation for the dominant 300 Hz / 75 deg failure.

Key references:

- `baffle_modeling/l22mg-bem-upgrade-path/README.md`
- `baffle_modeling/l22mg-current-status/README.md`
- `baffle_modeling/l22mg-juan-top-q7near-300hz-mesh-detail-diagnostic/README.md`

## Hypotheses For Why The Modeling Failed

### 1. The validation target contains a fragile high-angle front-side null

The 300.293 Hz / 75 deg L22-alone front null is extremely deep relative to both the naked driver and the L22-alone rear capture. It also is not reproduced at the same depth by the multi-driver top capture. This could be due to:

- measurement repeatability limits;
- fixture or turntable geometry;
- a very specific front-side scattering condition;
- passive-driver state or nearby structure not recorded in the L22-only HDF5;
- gate/window sensitivity;
- small angular misalignment amplified by a deep null.

The point should remain visible in reports, but it should not be the only basis for accepting a future model.

### 2. Horizontal naked polars do not identify the 3D source well enough

The naked source is measured on one horizontal polar plane. That does not uniquely determine:

- the near-field source distribution across cone, surround, dustcap, frame, basket, and rear side;
- off-plane radiation;
- local phase distribution at the baffle cutout;
- source behavior inside the finite baffle aperture.

Several source models fit Juan naked data acceptably but fail once inserted into the baffle. This is a classic inverse problem: a family of sources can match far-field horizontal data while producing different near-field/baffle scattering.

### 3. The passive and inactive-driver physical state is under-specified

The L22-only top-baffle target confirms the baffle was mounted and the measurement was raw/no-crossover/no-EQ, but it does not prove whether unused UM/tweeter positions were:

- open holes,
- covered patches,
- mounted inactive drivers,
- lossy/elastic diaphragms,
- partially sealed cavities,
- or some mixed condition.

Simple rigid approximations did not solve the problem, but the actual condition remains an uncertainty.

### 4. The real driver is not a rigid prescribed acoustic source

The modeling mostly used rigid acoustic BEM plus prescribed equivalent source surfaces. It did not solve coupled elastic/acoustic motion of:

- cone and surround breakup/compliance,
- basket and rear cavities,
- frame lip diffraction,
- suspension and dustcap details,
- mounting compliance,
- lossy materials.

At deep high-angle nulls, small structural or phase errors can dominate normalized polar residuals.

### 5. The metric is dominated by deep-null regions

The through-60 deg errors are often near or below about 1 dB for the better finite-source candidates, while all-angle RMS remains around 4 dB because 75/90 deg low-frequency null regions dominate. A target-above-minus-20 diagnostic mask brings the best models close to 1.5 dB, but that mask is not acceptance. The honest conclusion is not "passed after masking"; it is "model is plausible over moderate angles and fails in deep high-angle nulls."

### 6. Dense constant-panel BEM is not the long-term solver

The solver stack became much more auditable, but the current dense constant-panel backend is still the wrong endpoint for high-confidence driver/baffle validation. If continuing simulation-first, use adaptive higher-order BEM or accelerated BEM with feature-preserving remeshing rather than another round of tiny flat-triangle dense solves.

## Cleanup Scripts

The untracked `docs/` artifact cleanup is encoded in two one-off scripts at the repository root:

- `move_baffle_related.sh` moves the selected baffle-modeling reports from untracked top-level `docs/` entries into `baffle_modeling/`.
- `delete_stale.sh` deletes the remaining stale untracked top-level `docs/` entries.

Both scripts are dry-run by default, support `--audit`, and require `--apply` before changing files. They classify only untracked top-level `docs/` entries reported by `git status --porcelain -z --untracked-files=all docs`; tracked docs changes, raw data, source code, `output/`, and `linkwitz/` are not touched.

Use this order:

```bash
./move_baffle_related.sh --audit
./delete_stale.sh --audit
./move_baffle_related.sh --apply
./delete_stale.sh --apply
```

The scripts intentionally refuse to run if any untracked `docs/` entry is unclassified. Re-run the audits first if new generated folders appear.

Current curation after running the move step:

- `baffle_modeling/` contains the 21 retained handoff artifacts.
- `baffle_modeling_stale/` contains 45 parked intermediate artifacts that were moved out of the final handoff set but not deleted.

## Recommended Next Approach

A more capable model should not merely rerun the existing script matrix. It should first decide whether the target data are sufficiently repeatable and identifiable to support the requested inverse problem.

Recommended next steps:

1. Re-measure or independently verify the Juan L22MG top-baffle 300-600 Hz high-angle region, especially 60/75/90 deg around 300-400 Hz.
2. Repeat the L22-only top-baffle front and rear captures with the unused UM/tweeter physical state explicitly photographed and documented.
3. Capture more angular data around the null, e.g. 65/70/75/80/85/90 deg, because a 15 deg grid can make a deep null look more stable than it is.
4. Capture phase/timing consistently enough to compare baffle transfer, not only normalized magnitude.
5. Add near-field or additional off-plane naked-driver measurements if the goal is to infer a 3D source from naked data.
6. If simulation continues, move to a feature-preserving higher-order/adaptive BEM or FEM-BEM path with exposed driver surfaces and a physically constrained source, not another free equivalent-source cloud.
7. Keep validation honest: no rear scalar, no source scalar, no angle/band scalar, no target-derived correction, and no masked acceptance.

## Prompt For A More Capable Model

Use this as the next-start prompt:

```text
We are modeling the baffle transfer of the SEAS L22MG in the LX521 top baffle.

Inputs:
- Naked/baffleless L22MG HDF5: output/data/polar_data_juan_baffleless.h5, group "L22MG (nude)".
- Mounted top-baffle L22MG target HDF5: output/data/polar_data_juan_lx521_top_raw.h5, group "L22MG (LX521 top raw)".
- Both measurements are same driver, same room/setup family, 50 cm distance, L22MG/LM height, raw/no-crossover/no-EQ.
- Geometry: linkwitz/lx521metric-baffle.pdf, linkwitz/H1659-08_U22REX_P-SL_driver.stl, and lx521_l22mg_baffle geometry code.

Goal:
Predict the mounted top-baffle normalized polar response from the naked L22MG data plus physical baffle/driver geometry. Acceptance is normalized polar RMS <= 1.5 dB over 300-1200 Hz at the target angles, without angle-dependent gain, band-specific gain, source-level gain, rear scalar, or target-derived correction.

Important prior result:
This project failed. Do not repeat the old model matrix blindly. Better finite-source models got about 4.2 dB all-angle RMS and about 1.0 dB through 60 deg, but missed a dominant 300.293 Hz / 75 deg front target null by about +18 to +20 dB. The target at that point is -29.894 dB normalized, the naked L22 is -9.594 dB, the L22 top rear is -9.455 dB, and the L22+L10NEO+tweeters top capture is -18.024 dB. The null is 83 Hz wide but angularly isolated and not corroborated at the same depth by rear or combo captures.

Read first:
- BAFFLE_MODELING.md
- baffle_modeling/l22mg-current-status/README.md
- baffle_modeling/l22mg-juan-top-populated-silent-self-contained-report/README.md
- baffle_modeling/l22mg-juan-top-target-null-audit/README.md
- baffle_modeling/l22mg-juan-top-front-rear-consistency/README.md
- baffle_modeling/l22mg-juan-top-target-polar-quality/README.md
- baffle_modeling/l22mg-source-model-cross-validation/README.md
- baffle_modeling/l22mg-source-candidate-decision/README.md
- baffle_modeling/l22mg-bem-upgrade-path/README.md

Do not redo:
- simple compact2/full2 H1659 modal BEM runs;
- axisymmetric directivity diagnostic as acceptance;
- active piston/basic active surface variants;
- free rear/source scalar fitting;
- Andres retiming work;
- passive open/closed/rigid-face sweeps without new passive-state evidence;
- width/Wmax design conclusions before the baseline passes.

First determine whether the target data are repeatable and sufficiently constrained. If not, specify the missing measurements. If yes, propose a physically constrained source/solver upgrade that can be falsified against the existing naked and mounted measurements.
```

## Bottom Line

The old effort did produce useful negative evidence:

- the data paths and measurement geometry are now auditable;
- the naked-source inverse problem is underconstrained;
- the dominant residual is localized to high-angle low-frequency target behavior;
- simple passive/baffle/source tweaks do not close it;
- the current BEM backend is not the final solver, but solver residuals are not the main explanation for the worst null.

The next successful attempt likely requires better validation data and a stronger physical source/driver model, not more combinations of the same equivalent-source and rigid-baffle approximations.
