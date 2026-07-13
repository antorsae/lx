# V1LF R6F physical qualification record

Overall status: **PENDING — NOT PHYSICALLY QUALIFIED**

Floor-stand status: **PENDING — RELEASE NOT AUTHORIZED**  
No-floor-stand status: **PENDING — RELEASE NOT AUTHORIZED**

The hash-pinned MU10 reference mesh omits both electrical terminals, the
available datasheet does not dimension their carrier or removal geometry,
and the W22 STEP is a reference shrinkwrap rather than the installed custom
U22 driver. The CAD therefore uses explicit conservative proxies and keeps
`PHYSICAL_MEASURE_REQUIRED = True`. Generated STEP/STL files, passing CAD
tests, analytical strength screens, or a hash-valid candidate manifest are
not evidence that either physical state passed.

Qualify `floor_stand` and `no_floor_stand` independently. Evidence from one
state does not authorize the other. Do not change a state to **PASS — RELEASE
AUTHORIZED** until every applicable field below contains a measured result
and evidence reference, every rejection criterion passes, and the named
release approver signs that state. `PENDING`, a blank field, `N/A` without a
written justification, or a failed row keeps that state unauthorized.

## Per-state hardware and process identity

Record each state explicitly even when both use the same item; “same as other
state” is acceptable only with the shared lot/serial/evidence identifier.

| Required identity | `floor_stand` value / evidence | `no_floor_stand` value / evidence |
|---|---|---|
| Record revision / date | PENDING | PENDING |
| Source revision or commit | PENDING | PENDING |
| MU10RB-SL driver serial / lot | PENDING | PENDING |
| U22REX/P-SL driver serial / lot | PENDING | PENDING |
| Tweeter pair serial / lot, if fitted | PENDING | PENDING |
| 6.3 mm flag-Faston manufacturer and part number, both polarities | PENDING | PENDING |
| Insulation-boot manufacturer, part number and lot | PENDING | PENDING |
| LM / UM / T cable manufacturer, construction and part number | PENDING | PENDING |
| Heat-shrink manufacturer, type and recovery ratio | PENDING | PENDING |
| TPU manufacturer, grade, shore hardness and lot | PENDING | PENDING |
| PLA Tough+ manufacturer, colour, lot and conditioning history | PENDING | PENDING |
| Printer model / serial and nozzle identity | PENDING | PENDING |
| Slicer name / version and project-file hash | PENDING | PENDING |
| Layer height, line width, wall count, top/bottom layers and infill | PENDING | PENDING |
| Printed orientation and support settings for every qualified part | PENDING | PENDING |
| Heat-set manufacturer, part number and lot for M3 and M5 inserts | PENDING | PENDING |
| Insert tool, set temperature, dwell/process and operator | PENDING | PENDING |
| Bolt, washer, nut and stock-bridge/support hardware specification | PENDING | PENDING |
| Qualification ambient and maximum intended service temperature | PENDING | PENDING |

## Per-state candidate identity

Record hashes after the exact candidate artifacts have been generated. The
printed/sliced files must be traceable to these entries; a later regeneration
or process change requires a new qualification. Do not record the final
manifest hash here: the manifest hashes this record, which would create a
circular checksum. Instead hash a candidate artifact inventory that excludes
both the manifest and this qualification record.

| Required identity | `floor_stand` candidate | `no_floor_stand` candidate |
|---|---|---|
| Candidate/build ID | PENDING | PENDING |
| Candidate generation date | PENDING | PENDING |
| Inspector and inspection date | PENDING | PENDING |
| Candidate artifact-inventory path / SHA-256, excluding manifest and this record | PENDING | PENDING |
| LM carrier STL SHA-256 | PENDING | PENDING |
| UM carrier STL SHA-256 | PENDING | PENDING |
| Tweeter crescent STL SHA-256, if fitted | PENDING | PENDING |
| Selected external cable-retention identity, if fitted | PENDING | PENDING |
| Floor-support STL SHA-256 | PENDING | N/A — no floor support in this state |
| Stock bridge serial/revision and installation substrate | N/A — floor support owns this load path | PENDING |
| Final slicer project / G-code / print-job hashes | PENDING | PENDING |
| As-built mass and measured LM/UM centre spacing | PENDING | PENDING |
| Deviations from candidate CAD, hardware or documented process | PENDING | PENDING |

## Coupon, terminal, cable and strain-relief evidence

Use the exact printed coupons from the candidate state. Attach measurements,
photographs or video identifiers, instrument IDs and calibration dates rather
than recording only “looks good.”

| Required evidence | `floor_stand` result / evidence | `no_floor_stand` result / evidence |
|---|---|---|
| Coupon 7: physical U22 seat, M5 insert/pad and flushness | PENDING | PENDING |
| Coupon 9: terminal clock midway between 238° and 328° screws | PENDING | PENDING |
| Coupon 12: state-specific closed bore bump, full solid roof-to-bore saddle, cable passage and exact hardware clearance | PENDING | PENDING |
| Terminal carrier radius and rear Z | PENDING | PENDING |
| Terminal pitch, tab width/thickness and exposed length | PENDING | PENDING |
| Polarity order | PENDING | PENDING |
| Actual withdrawal axis, release stroke and peak force for each terminal | PENDING | PENDING |
| Positive disengagement/handling margin beyond the modeled 12 mm state | PENDING | PENDING |
| Measured receptacle and installed boot envelopes | PENDING | PENDING |
| One-at-a-time pull at 0/3/6/9/12 mm with opposite side installed | PENDING | PENDING |
| Installed LM / UM / T cable outer diameters | PENDING | PENDING |
| Cable-manufacturer minimum static and repeated-flex bend radii | PENDING | PENDING |
| Free D7.8 LM lead follows the 20.15 mm / 269.5° rear span without a printed micro-duct; center z=0.40..3.80, 1.00 mm outer clearance to the deepest z=5.3 pad/web rear datum, and floor-support physical-cable-plus-0.4 mm clearance verified | PENDING | PENDING |
| Finished OD8 bundle and both OD4 branch heat-shrink dimensions | PENDING | PENDING |
| Y-junction continuity, insulation, strain transfer and polarity labels | PENDING | PENDING |
| Selected external retention dimensions, material and buried-route/free-cable/service-envelope clearance | PENDING | PENDING |
| External retention installation around the terminated cable without shell damage | PENDING | PENDING |
| External retention load and deliberate tool/finger removal access | PENDING | PENDING |
| Full service-motion clearance to the physical MU10 and installed U22 | PENDING | PENDING |
| Final free-LM placement plus UM/T buried-span fishing and free-span placement, electrical continuity and insulation test | PENDING | PENDING |
| Final strain-relief pull transfers cable load away from MU tabs | PENDING | PENDING |
| No-floor bridge plate has a soft cubic blend into R113 and occupies z=5.3..18.3; four rear Ø6.4 x 6.8 insert bores retain a 6.2 mm front floor, the centered UM/T mouths at x=±5/y=82 open only at rear z=5.3, and no geometry extends behind the existing LM-pad envelope | N/A — no bridge plate; supported ring feeds retained | PENDING |
| Six actual Ø5 x 2 magnets are bonded face-flush—not bottomed—in Ø5.2 x 2.2 pockets: preserve the upper LM sites at 64°/116° with at least 2.2 mm insert gap and 0.86 mm route-cover gap; add lower LM sites at 224°/316° with at least 23.0 mm nearest-insert edge clearance, keeping 224° at z=12.55 and moving only 316° to z=15.40 for route clearance with a closed 0.30 mm front skin; verify buried-route and bridge/support clearance; retain UM sites at 50.5°/129.5°, z=15.1, with at least 1.1 mm insert/T-cover gap, 0.2 mm radial floor and 0.6 mm front skin; no proud ears | PENDING | PENDING |
| UM passage is buried only in LM and ends in a flush free-cable handoff; the UM carrier has no printed rear UM duct or D82 mouth, and the physical cable remains clear behind UM through its R15/R20 service path | PENDING | PENDING |
| T passage is buried only in LM/UM and ends in a flush free-cable handoff; the tweeter crescent has no printed T arc, conduit, socket, or horn, and the free cable remains clear behind the crescent | PENDING | PENDING |
| Physical UM/T centerlines cross at 82.67° with T above UM and retain the documented physical-envelope gap; no two-printed-duct separator web is claimed or required | PENDING | PENDING |
| Floor support has no obsolete LM magnet cups/arms; each Ø11.6 connected boss fits its Ø12.4 carrier clearance and retains 2.6 mm radial wall around the Ø6.4 heat-set cavity; modeled free-LM-cable clearance remains open | PENDING | N/A — no floor support in this state |
| All eight named duct bumps have continuous solid saddle material from conduit roof to bore floor; floor 300/240/180° axes contain only the exact hardware voids | PENDING | PENDING |

## Per-state structural proof evidence

Use a dummy mass, not valuable drivers. Apply the documented resultant at the
modeled `(y=230 mm, rear offset=70 mm)` assembly load point: 39.23 N for the
4 kg sustained-1g case, 117.68 N for transient 3g and 196.13 N for transient
5g. Loading only a convenient mounting hole does not reproduce the screened
moments. The upper joints must also be checked with their 0.85 kg distributed
mass and documented 120 mm plan / 70 mm rear lever case. Ramp every load over
at least 10 seconds; hold 1g for 24 hours, 3g for 60 seconds and 5g for 10
seconds. Reject cracks, insert motion, loss of torque, permanent deformation,
electrical damage or impaired cable service.

The no-floor analytical screen deducts the complete Ø8.2 and Ø6.0 centered
entry lumens from the 62 mm insert core and credits a conservative
47.8 × 13.0 mm member; exact sampled soft-outline cuts retain at least
53.5 mm. Its calculated 5g factor is about 1.10. The fusion screen separately credits only the actual
z=6.8..18.3 (11.5 mm) ring-lip overlap, giving a 2.87 factor at 5g. This
calculation remains screening context only and
does not change any PENDING field below into physical evidence.

| Required evidence | `floor_stand` result / evidence | `no_floor_stand` result / evidence |
|---|---|---|
| Fixture drawing/ID, load-cell ID/calibration and evidence-file hashes | PENDING | PENDING |
| Actual load application coordinates and direction | PENDING | PENDING |
| Pre-test temperature, dimensions, insert positions and fastener torques | PENDING | PENDING |
| 1g: load, 24 h duration, temperature and maximum deflection | PENDING | PENDING |
| 1g: unloaded residual deflection, insert movement, cracks and post-test torque | PENDING | PENDING |
| 3g: load, 60 s duration, temperature and maximum deflection | PENDING | PENDING |
| 3g: unloaded residual deflection, insert movement, cracks and post-test torque | PENDING | PENDING |
| 5g: load, 10 s duration, temperature and maximum deflection | PENDING | PENDING |
| 5g: unloaded residual deflection, insert movement, cracks and post-test torque | PENDING | PENDING |
| Upper LM-to-UM and UM-to-tweeter joint proof at the documented 0.85 kg case | PENDING | PENDING |
| Three lower W22-axis support screws, rails and NL8 panel remain sound | PENDING | N/A — no floor support in this state |
| Front-flush solid four-hole web, stock bridge and installation substrate remain sound | N/A — no stock bridge in this state | PENDING |
| Solid-backed cable bumps remain clear, cavity-free and undamaged after every load case | PENDING | PENDING |
| Post-proof cable service, continuity and Faston removal repeated successfully | PENDING | PENDING |
| Proof repeated at maximum intended service temperature, or lower rated limit recorded | PENDING | PENDING |
| Final structural decision and any operating restrictions | PENDING | PENDING |

## Evidence inventory

| Evidence item | `floor_stand` path / SHA-256 | `no_floor_stand` path / SHA-256 |
|---|---|---|
| Dimension and coupon report | PENDING | PENDING |
| Terminal service photographs/video | PENDING | PENDING |
| External cable-retention installation/load/removal evidence | PENDING | PENDING |
| Structural fixture and load-time history | PENDING | PENDING |
| Pre/post metrology and torque report | PENDING | PENDING |
| Electrical continuity/insulation report | PENDING | PENDING |

## State-specific release signoff

| Signoff field | `floor_stand` | `no_floor_stand` |
|---|---|---|
| Evidence reviewed by / date | PENDING | PENDING |
| Mechanical proof witnessed by / date | PENDING | PENDING |
| Terminal/electrical service witnessed by / date | PENDING | PENDING |
| Deviations and operating limits accepted by | PENDING | PENDING |
| Release approver printed name / role | PENDING | PENDING |
| Release approver signature / date | PENDING | PENDING |
| Final state decision: `PASS — RELEASE AUTHORIZED` or `FAIL/PENDING` | FAIL / PENDING | FAIL / PENDING |

Release gate: **FAIL / PENDING FOR BOTH STATES**. The modeled 12 mm pull
equals the provisional 12 mm exposed-tab length, so the current proxy has
zero positive release overtravel margin. Do not infer physical fit, structural
qualification or release authority from this template, generated geometry,
passing analytic tests, or a candidate manifest.
