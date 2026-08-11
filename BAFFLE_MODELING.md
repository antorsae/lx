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

Critical timing update:

- Juan captures were made over USB/no timing reference, so REW `t=0` is not an absolute acoustic reference. At high angle, REW can peak-reference a later stronger room reflection to `0 ms`.
- The active processing policy for Juan baffleless and Juan LX521 top raw is now `direct_ir_peak_policy=ir-start`.
- Use REW `timeOfIRStartSeconds` / IR onset as the gate reference for regenerated HDF5s. Do not gate these Juan high-angle measurements around the global peak or the first strong peak near `0 ms`.
- A raw-IR audit on 2026-05-25 found `L22MG (LX521 top raw)` front 75 deg has REW IR start at `-9.333 ms` while the global peak is at `0 ms`. The earlier event is physically plausible as the direct 75 deg response; the `0 ms` event is plausibly a later room reflection with about 3.2 m extra path.
- The same audit found `L10NEO` naked high-angle rows with similar early direct events around `-9` to `-11 ms`. `L10NEO (LX521 top raw)` did not show the same high-angle early-onset pattern.
- Gate sensitivity is large at the former failure point: for `L22MG (LX521 top raw)` front 75 deg, using IR-start rather than `0 ms` changes the 300-1200 Hz response by about `6.99 dB RMS` and raises the 300.293 Hz normalized response by about `13.12 dB`, from `-29.894 dB` to `-16.774 dB`.
- Therefore the old first-strong / peak-gated validation target is considered contaminated for the 300 Hz / 75 deg null. Existing model rankings that used that target are stale and must not be treated as final acceptance evidence.

Geometry/provenance references:

- `baffle_modeling/l22mg-measurement-geometry-provenance/README.md`
- `baffle_modeling/l22mg-current-status/README.md`
- `baffle_modeling/l22mg-juan-acceptance-scorecard/README.md`

## Current IR-Start Result Summary

No current model is accepted yet against the corrected `ir-start` target, but the nature of the failure changed materially.

The corrected real-baffle-only rerun used the actual LX521 top-baffle width, `B = 305 mm`, with populated-silent passive positions. The A/C/D width variants are deferred exercises and must not be used for design conclusions until the actual B model passes.

Current corrected-target model ranking:

| model / family | actual width | all-angle RMS 300-1200 Hz | through-60 RMS | through-80 RMS | worst signed error |
| --- | ---: | ---: | ---: | ---: | --- |
| H1659 modal full2, populated-silent | 305 mm | 3.119 dB | 0.460 dB | 1.880 dB | -14.047 dB at 424.072 Hz / 90 deg |
| H1659 modal compact2, populated-silent | 305 mm | 3.210 dB | 0.479 dB | 1.906 dB | -14.458 dB at 424.439 Hz / 90 deg |
| axisymmetric directivity diagnostic | 305 mm | 4.299 dB | 0.949 dB | 3.137 dB | +17.550 dB at 1199.707 Hz / 75 deg |
| H1659 modal full2, +5 mm acoustic-center probe | 305 mm | 5.133 dB | 1.495 dB | 3.928 dB | +21.145 dB at 1199.707 Hz / 75 deg |
| H1659 modal full2, +20 mm acoustic-center probe | 305 mm | 6.539 dB | 2.119 dB | 4.799 dB | +28.096 dB at 1199.707 Hz / 75 deg |

Current best:

- `output/l22mg-bem-juan-ir-start-populated-silent-modal-full2-real-baffle`
- `docs/l22mg-bem-juan-ir-start-populated-silent-modal-full2-real-baffle/README.md`
- self-contained summary: `baffle_modeling/l22mg-juan-ir-start-real-baffle-report/README.md`

Interpretation: the best corrected model is excellent through 60 deg and close, but still above target, through 80 deg. It fails full-angle acceptance because the simulation makes the 90 deg response too deep around 424 Hz. This is not the old 300 Hz / 75 deg failure.

## Historical Result Summary

The results below are historical first-strong / peak-gated runs. They remain useful as a map of what was tried, but the dominant L22 front 75 deg target null they failed to match is now confirmed stale enough to invalidate those rankings as final acceptance evidence.

Historical populated-silent rerank and Wmax summaries showed this pattern:

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

## Former Dominant Failure

The former dominant blocker was the Juan L22-only top-baffle low-frequency high-angle null in the legacy first-strong / peak-gated target:

- Frequency: 300.293 Hz
- Angle: 75 deg front
- Legacy target normalized level: -29.894 dB
- Naked L22 normalized level at same point: -9.594 dB
- Measured target-minus-nude transfer: -20.300 dB
- Typical historical model residual: about +18 to +20 dB at the same point

Older measured-data audits showed this was not just a model-output quirk, but those audits used the legacy target timing and must be rerun under `ir-start`:

- `baffle_modeling/l22mg-juan-top-target-null-audit/README.md`
- `baffle_modeling/l22mg-juan-top-front-rear-consistency/README.md`
- `baffle_modeling/l22mg-juan-top-target-polar-quality/README.md`

Important target-only findings:

- The multi-driver top capture, `L22MG+L10NEO+Tweeters (LX521 top raw)`, is -18.024 dB at the same 300.293 Hz / 75 deg point, which is 11.870 dB shallower than the L22-alone target.
- The L22-alone rear top-baffle measurement is -9.455 dB at the same point, not -29.894 dB.
- Front-minus-rear at that point is -20.440 dB.
- The target-only quality audit finds one deep 300-600 Hz cluster at 75 deg, 83.1 Hz wide, about 17.3 dB deeper than adjacent angles.

Corrected IR-start result: the same 300.293 Hz / 75 deg point is now `-16.774 dB`, not `-29.894 dB`. The corrected deepest high-angle target point in 300-600 Hz, angles >=60 deg is now 599.854 Hz / 90 deg:

- target normalized: `-19.925 dB`
- naked L22 normalized: `-17.580 dB`
- target-minus-nude transfer: `-2.344 dB`
- combo top capture: `-24.527 dB`
- combo-minus-target: `-4.602 dB`

Revised interpretation: the old 75 deg null was a timing/windowing artifact, not a reliable physical baffle null. The raw IR shows a plausible direct event at `-9.333 ms` and a later stronger event at `0 ms`; with no timing reference, the later event can be a reflection. The old target's `-29.894 dB` normalized point must not be treated as a decisive validation target.

## What Was Tried

### Measurement And Timing Provenance

Done and should not be repeated unless new raw data arrive:

- Confirmed Juan naked and top-baffle L22MG measurements are 0.50 m and at L22MG/LM height.
- Confirmed the target angle grid is 0/15/30/45/60/75/90 deg and should be solved directly, not interpolated.
- Separated Juan same-room/same-driver target from Andres published-parity target.
- Added gate/report logic to avoid accidentally using stale 0.75 m source-radius metadata.
- Added target-null, front/rear consistency, polar-quality, and passive-state provenance warnings.
- Rechecked Juan raw IRs for L22MG and L10NEO, both naked and top raw. The high-angle `0 ms` peaks are not reliable direct-arrival references because these are no-timing-ref USB captures.
- Confirmed the active Juan timing policy should be `ir-start`, not first-strong/global peak. This is especially important for `L22MG (LX521 top raw)` front 75 deg and naked L10NEO high-angle rows.

Do not spend another iteration retiming or reclassifying the Andres artifacts unless the goal changes. Andres published parity is a useful legacy data-path check, but it is not the current same-driver/same-height baffle target. The Juan-derived acceptance artifacts have now been regenerated from `ir-start`; most older baffle-modeling reports used the now-suspect peak-gated target and remain historical only.

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
- Do not treat Wmax widening or A/B/C/D outline exercises as accepted design results. The actual LX521 baffle is B = 305 mm; only run A/C/D after an actual-B model passes the acceptance gate.
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

## Updated Hypotheses

### 1. The former target was probably contaminated by wrong IR timing

This is the primary confirmed hypothesis as of 2026-05-25.

Juan's USB/no-timing-ref measurements do not provide an absolute acoustic `0 ms`. REW can place a later, stronger event at `0 ms`; at high horizontal angles that event can be a room reflection fed by a stronger radiation direction. For `L22MG (LX521 top raw)` front 75 deg, the raw IR contains a plausible direct response at REW IR start `-9.333 ms`, while the global peak is at `0 ms`. A 9.333 ms delay is about 3.2 m of extra path, which is plausible for a room reflection in the photographed setup.

Processing that row with `0 ms` / first-strong timing instead of IR-start changes the 300-1200 Hz response by about 6.99 dB RMS and moves the 300 Hz response by about +13.07 dB. That magnitude is large enough to explain much of the old 300 Hz / 75 deg "model failure." Therefore all historical acceptance claims and failures tied to the legacy peak-gated target must be rerun before drawing physical conclusions.

### 2. The validation target may still contain fragile high-angle behavior

After retiming, the 75 deg region may still be sensitive, but the old `-29.894 dB` normalized point is not a reliable physical target. Any remaining 300-600 Hz high-angle structure could be due to:

- measurement repeatability limits;
- fixture or turntable geometry;
- a very specific front-side scattering condition;
- passive-driver state or nearby structure not recorded in the L22-only HDF5;
- gate/window sensitivity;
- small angular misalignment amplified by a deep null.

The point should remain visible in reports, but it should not be the only basis for accepting a future model. First regenerate the target with IR-start timing, then reassess whether a deep null remains.

### 3. Horizontal naked polars do not identify the 3D source well enough

The naked source is measured on one horizontal polar plane. That does not uniquely determine:

- the near-field source distribution across cone, surround, dustcap, frame, basket, and rear side;
- off-plane radiation;
- local phase distribution at the baffle cutout;
- source behavior inside the finite baffle aperture.

Several source models fit Juan naked data acceptably but fail once inserted into the baffle. This is a classic inverse problem: a family of sources can match far-field horizontal data while producing different near-field/baffle scattering.

### 4. The passive and inactive-driver physical state is under-specified

The L22-only top-baffle target confirms the baffle was mounted and the measurement was raw/no-crossover/no-EQ, but it does not prove whether unused UM/tweeter positions were:

- open holes,
- covered patches,
- mounted inactive drivers,
- lossy/elastic diaphragms,
- partially sealed cavities,
- or some mixed condition.

Simple rigid approximations did not solve the problem, but the actual condition remains an uncertainty.

### 5. The real driver is not a rigid prescribed acoustic source

The modeling mostly used rigid acoustic BEM plus prescribed equivalent source surfaces. It did not solve coupled elastic/acoustic motion of:

- cone and surround breakup/compliance,
- basket and rear cavities,
- frame lip diffraction,
- suspension and dustcap details,
- mounting compliance,
- lossy materials.

At deep high-angle nulls, small structural or phase errors can dominate normalized polar residuals.

### 6. The metric can be dominated by deep-null regions

The through-60 deg errors are often near or below about 1 dB for the better finite-source candidates, while all-angle RMS remains around 4 dB because 75/90 deg low-frequency null regions dominate. A target-above-minus-20 diagnostic mask brings the best models close to 1.5 dB, but that mask is not acceptance. The honest conclusion is not "passed after masking"; it is "model is plausible over moderate angles and fails in deep high-angle nulls."

### 7. Dense constant-panel BEM is not the long-term solver

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

1. Regenerate `output/data/polar_data_juan_baffleless.h5` and `output/data/polar_data_juan_lx521_top_raw.h5` with `direct_ir_peak_policy=ir-start`, then regenerate the corresponding `docs/juan-baffleless/` and `docs/juan-lx521-top-raw/` plots.
2. Rerun target-null, front/rear consistency, combo-capture, and acceptance scorecard reports against the regenerated IR-start HDF5s. The old 300 Hz / 75 deg null report is stale.
3. Use the corrected actual-B baseline as the reference: the current best is H1659 modal full2, populated-silent, 305 mm actual baffle, with 3.119 dB all-angle RMS and 0.460 dB through-60 RMS.
4. Iterate geometry/source/solver upgrades only on the actual B = 305 mm baffle. Do not resume A/C/D width exercises until a passing actual-B simulation exists. Optimize for physically constrained improvement, not target-derived correction.
5. If a high-angle null remains after IR-start regeneration, consider re-measurement or independent verification of the Juan L22MG top-baffle 300-600 Hz high-angle region, especially 60/75/90 deg around 300-400 Hz.
6. Capture phase/timing consistently enough to compare baffle transfer if new measurements are made; an acoustic timing reference would remove the current ambiguity.
7. Add near-field or additional off-plane naked-driver measurements if the goal is to infer a 3D source from naked data.
8. Keep validation honest: no rear scalar, no source scalar, no angle/band scalar, no target-derived correction, and no masked acceptance.

## Prompt For A More Capable Model

Use this as the next-start prompt:

```text
We are modeling the baffle transfer of the SEAS L22MG in the LX521 top baffle.

Inputs:
- Naked/baffleless L22MG HDF5: output/data/polar_data_juan_baffleless.h5, group "L22MG (nude)".
- Mounted top-baffle L22MG target HDF5: output/data/polar_data_juan_lx521_top_raw.h5, group "L22MG (LX521 top raw)".
- Both measurements are same driver, same room/setup family, 50 cm distance, L22MG/LM height, raw/no-crossover/no-EQ.
- Geometry: linkwitz/lx521metric-baffle.pdf, linkwitz/H1659-08_U22REX_P-SL_driver.stl, and lx521_l22mg_baffle geometry code.
- Timing policy: direct_ir_peak_policy must be ir-start for Juan baffleless and Juan LX521 top raw. These USB/no-timing-reference captures cannot use REW peak-at-0 as an acoustic reference.
- Baffle scope: model only the actual LX521 top baffle, B = 305 mm. A/C/D widened/narrowed shoulder outlines are deferred exercises to run only after the actual-B model passes.

Goal:
Predict the mounted top-baffle normalized polar response from the naked L22MG data plus physical baffle/driver geometry. Acceptance is normalized polar RMS <= 1.5 dB over 300-1200 Hz at the target angles, without angle-dependent gain, band-specific gain, source-level gain, rear scalar, or target-derived correction.

Important prior result:
This project previously failed against a first-strong / peak-gated target. That failure is now stale because raw IR inspection found L22MG top raw front 75 deg has a plausible direct arrival at REW IR start -9.333 ms and a later stronger event at 0 ms. With no timing reference, the 0 ms event is plausibly a room reflection. Using IR-start instead of 0 ms changes L22 top raw F75 by about 6.99 dB RMS over 300-1200 Hz and raises the 300.293 Hz / 75 deg normalized target from -29.894 dB to -16.774 dB.

Current corrected baseline:
- Best actual-B run: output/l22mg-bem-juan-ir-start-populated-silent-modal-full2-real-baffle.
- Score: 3.119 dB all-angle RMS over 300-1200 Hz, 0.460 dB through 60 deg, 1.880 dB through 80 deg.
- Remaining failure: too-deep simulated 90 deg response around 424 Hz, worst signed error -14.047 dB.

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

Immediate work:
1. Confirm config.py has direct_ir_peak_policy="ir-start" for juan-baffleless and juan-lx521-top-raw, and that config.DIRECT_IR_PEAK_POLICY defaults to "ir-start".
2. Regenerate Juan baffleless and Juan LX521 top raw HDF5s and docs from raw .mdat using ir-start timing.
3. Rerun target-null/front-rear/combo/acceptance summaries and compare old first-strong scores to corrected ir-start scores.
4. Re-run or extend physically grounded simulations on the actual B = 305 mm baffle only. Start from the corrected H1659 modal full2 populated-silent baseline, then branch into source/geometry/solver variants that can reduce the remaining 90 deg residual without damaging the through-60 fit.
5. Iterate modeling for the best possible physically constrained result within the time budget. Save plots/tables for every meaningful actual-B attempt in a new baffle_modeling/ artifact.

Do not use:
- global peak or first-strong timing for Juan high-angle target/source HDF5s;
- free rear/source scalar fitting;
- angle-dependent or band-specific gain;
- target-derived correction;
- masked acceptance as a pass;
- Andres retiming work unless the goal explicitly changes.
- A/C/D width exercises until an actual-B simulation passes.

Deliver:
- best corrected model and ranking table;
- before/after timing comparison versus stale first-strong target;
- normalized polar maps and polar slices including 300-1200 Hz;
- concise statement of whether the timing hypothesis explains the old failure and what residual failure remains.
```

## Recommended /goal Prompt

Use this for a bounded modeling run:

```text
/goal 6h
Objective: With the corrected Juan IR-start timing hypothesis, regenerate and rescore the L22MG LX521 top-baffle modeling target, then iterate physically constrained baffle simulations for up to 6 hours to get the best possible simulation.

Context:
- Work in /Users/antor/gh/lx.
- Read BAFFLE_MODELING.md first.
- Juan baffleless and Juan LX521 top raw are USB/no-timing-reference REW captures. Do not use global peak or first-strong timing as the acoustic reference. Use direct_ir_peak_policy=ir-start / REW timeOfIRStartSeconds for regenerated Juan HDF5s.
- The old dominant 300 Hz / 75 deg failure likely came from gating L22MG top raw F75 around a later reflection at 0 ms instead of the plausible direct arrival at -9.333 ms.
- Model only the actual LX521 top baffle, B = 305 mm. Ignore A/C/D width variants until a passing actual-B model exists.

Tasks:
1. Verify config.py and loader behavior use ir-start for juan-baffleless and juan-lx521-top-raw.
2. Regenerate output/data/polar_data_juan_baffleless.h5, output/data/polar_data_juan_lx521_top_raw.h5, docs/juan-baffleless/, and docs/juan-lx521-top-raw/.
3. Recompute target-null, front/rear consistency, combo-capture, and acceptance scorecards under ir-start timing. Explicitly compare against the stale first-strong/peak-gated result.
4. Re-run or extend the best prior physically grounded simulation families against the corrected target on the actual B = 305 mm baffle only. Start with compact/full H1659 modal populated-silent variants, then iterate source/geometry/solver changes only if they are physically constrained and falsifiable.
5. Keep strict validation: no target-derived correction, no angle/band gain, no source/rear scalar hacks, no masked acceptance as a pass.
6. Save a new self-contained baffle_modeling/ artifact with plots, ranking tables, timing-before/after plots, and a concise final recommendation.

Success criteria:
- Best normalized polar RMS over 300-1200 Hz is reported with full angle coverage.
- The report states whether the IR-start timing correction explains the old 300 Hz / 75 deg residual.
- The best model's residual maps and polar slices are included.
- Stop after 6 hours even if not accepted; preserve the best result and the next most promising path.
```

## Bottom Line

The old effort did produce useful negative evidence:

- the data paths and measurement geometry are now auditable;
- the naked-source inverse problem is underconstrained;
- the former dominant residual was localized to high-angle low-frequency target behavior;
- the current best explanation for the former 300 Hz / 75 deg failure is wrong timing/gating of a no-timing-reference measurement, not necessarily a missing physical baffle effect;
- simple passive/baffle/source tweaks do not close it;
- the current BEM backend is not the final solver, but solver residuals are not the main explanation for the worst null.

The next successful attempt must first regenerate and rescore the Juan target with `ir-start` timing. Only then should it decide whether better validation data or a stronger physical source/driver model is still required.
