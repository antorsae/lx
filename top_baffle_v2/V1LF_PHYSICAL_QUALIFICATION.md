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

The floor candidate has no separate support artifact. The canonical floor LM
and `lx521_top_v1lf_optional_lm_keyed_1of2_bottom.stl` each own the complete
integral W64 stem/foot, R12 root, buried floor lanes and NL8 panel. The floor
datum is world `Y=0`, exactly **200.981 mm** below the LM axis. All six LM
driver insert sites are ordinary blind carrier bores in both states.

Each state release manifest ships both the canonical monolithic LM carrier and
both halves of the optional keyed LM form. Although an installed assembly uses
exactly one form, release qualification must authorize **both shipped forms
independently**. Authorization of one form does not authorize the other, and
`N/A` is not permitted for either shipped form's identity, required evidence,
or print-form decision. A state remains unauthorized until both print-form
decisions pass. The keyed form is cut at world `Y=172.481 mm` with a zero-gap
planar butt; neither keyed half may be combined with the monolithic carrier in
an installed assembly. One concealed right-hand straight rounded tongue/blind-socket
registration pair is carved wholly inside the existing R110.6..R113 LM lip.
The tongue is 0.8 mm wide and engages 3.5 mm along its tangential insertion
axis, approximately 75.23° from +X. It adds no external protrusion,
envelope growth, extra screw, or standalone retention/load credit. Its
installed load path exists only after the LM driver flange and all
normal LM fasteners splice the seam.

The keyed socket's current CAD contract retains at least **0.65 mm inner wall**
and **0.60 mm outer wall**. Both are below the usual 0.8 mm printed-wall
convention, so positive CAD wall checks are not release evidence by themselves.
The candidate slicer preview must show continuous intended extrusion paths at
both walls, and a state/process-matched printed tongue/socket coupon must pass
before the keyed form can be authorized.

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
| Selected Bambu PLA grade (Tough+, Basic, Lite, Matte, or Silk+), colour, lot and conditioning history | PENDING | PENDING |
| Printer model / serial and nozzle identity | PENDING | PENDING |
| Slicer name / version and project-file hash | PENDING | PENDING |
| Layer height, line width, wall count, top/bottom layers and infill; floor must record 100% local-solid modifier through complete stem/root | PENDING | PENDING |
| Printed orientation and support settings for every qualified part | PENDING | PENDING |
| Heat-set manufacturer, part number and lot for M3 and M5 inserts | PENDING | PENDING |
| Insert tool, set temperature, dwell/process and operator | PENDING | PENDING |
| Bolt, washer, nut, stock-bridge hardware, and floor anti-tip tether/anchor specification | PENDING | PENDING |
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
| Canonical monolithic LM carrier STL SHA-256 — mandatory shipped form | PENDING | PENDING |
| Keyed LM bottom `lx521_top_v1lf_optional_lm_keyed_1of2_bottom.stl` SHA-256 — mandatory shipped form | PENDING | PENDING |
| Keyed LM top `lx521_top_v1lf_optional_lm_keyed_2of2_top.stl` SHA-256 — mandatory shipped form | PENDING | PENDING |
| Keyed split print orientation and verified in-bed footprint (bottom / top) | Floor bottom X=−90°, Z=0°; top Z=45°; record measured values, each axis ≤220 mm | Z26° bottom 198.79×205.51 mm / Z45° top 210.47×210.47 mm |
| UM carrier STL SHA-256 | PENDING | PENDING |
| Tweeter crescent STL SHA-256, if fitted | PENDING | PENDING |
| Selected external cable-retention identity, if fitted | PENDING | PENDING |
| Integral floor LM geometry identity: W64 stem/foot, R12 root, NL8 panel, floor Y=0 and 200.981 mm LM-axis height | PENDING | N/A — no integral stand in this state |
| Positive check that `lx521_top_v1lf_addon_mount_floor_support.stl` is absent | PENDING | PENDING |
| Stock bridge serial/revision and installation substrate | N/A — integral LM owns this load path | PENDING |
| Floor anti-tip tether/anchor make, attachment points, rating and installation record | PENDING | N/A — no integral floor stand in this state |
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
| Coupon 12: state-specific closed bore bump, full solid roof-to-ordinary-blind-bore saddle, and cable passage | PENDING | PENDING |
| Monolithic LM form physical fit: actual U22 seat/flange, all normal LM fasteners, insert access, front-datum seating and surrounding-clearance inspection on the printed monolithic carrier | PENDING | PENDING |
| Monolithic LM form cable check: actual UM/T cables fished through the intact buried lumens, free LM lead placed, cable service repeated without snag or insulation damage, and continuity/insulation verified | PENDING | PENDING |
| Keyed LM form physical fit: both printed halves fully seated front-face-down, actual U22 flange and all normal LM fasteners installed, front-datum coplanarity and post-fit crack/damage inspection | PENDING | PENDING |
| Keyed LM dedicated tongue/socket coupon: candidate material/process, assembly force, full seating of the straight rounded 0.8 mm tongue through its 3.5 mm engagement in the blind right-hand socket, socket clearance and straight tangential insertion along the ~75.23° axis | PENDING | PENDING |
| Keyed LM socket thin-wall acceptance: CAD minima confirmed at ≥0.65 mm inner and ≥0.60 mm outer, candidate slicer preview shows continuous intended extrusion paths at both sub-0.8 mm walls, and the state/process-matched printed coupon has no gaps, delamination or breakage | PENDING | PENDING |
| Keyed LM seam metrology: world Y=172.481 mm, closed zero-gap planar butt, no volumetric half overlap, both front faces registered on one flat datum, and continuous UM/T route seams | PENDING | PENDING |
| Keyed LM physical UM/T cable pull-through across both preserved seam lumens without snag, insulation damage or slicer-support residue, followed by continuity/insulation verification | PENDING | PENDING |
| Terminal carrier radius and rear Z | PENDING | PENDING |
| Terminal pitch, tab width/thickness and exposed length | PENDING | PENDING |
| Polarity order | PENDING | PENDING |
| Actual withdrawal axis, release stroke and peak force for each terminal | PENDING | PENDING |
| Positive disengagement/handling margin beyond the modeled 12 mm state | PENDING | PENDING |
| Measured receptacle and installed boot envelopes | PENDING | PENDING |
| One-at-a-time pull at 0/3/6/9/12 mm with opposite side installed | PENDING | PENDING |
| Installed LM / UM / T cable outer diameters | PENDING | PENDING |
| Cable-manufacturer minimum static and repeated-flex bend radii | PENDING | PENDING |
| Free D7.8 LM lead follows the 20.15 mm / 269.5° rear span without a printed micro-duct; center z=0.40..3.80 and 1.00 mm outer clearance to the deepest z=5.3 pad/web rear datum; floor state continues through the Ø9 buried integral-stem lane | PENDING | PENDING |
| Finished OD8 bundle and both OD4 branch heat-shrink dimensions | PENDING | PENDING |
| Y-junction continuity, insulation, strain transfer and polarity labels | PENDING | PENDING |
| Selected external retention dimensions, material and buried-route/free-cable/service-envelope clearance | PENDING | PENDING |
| External retention installation around the terminated cable without shell damage | PENDING | PENDING |
| External retention load and deliberate tool/finger removal access | PENDING | PENDING |
| Full service-motion clearance to the physical MU10 and installed U22 | PENDING | PENDING |
| Final free-LM placement plus UM/T buried-span fishing and free-span placement, electrical continuity and insulation test | PENDING | PENDING |
| Final strain-relief pull transfers cable load away from MU tabs | PENDING | PENDING |
| No-floor bridge plate has a soft cubic blend into R113 and occupies z=5.3..18.3; four rear Ø6.4 x 6.8 insert bores retain a 6.2 mm front floor, the centered UM/T mouths at x=±5/y=82 open only at rear z=5.3, and no geometry extends behind the existing LM-pad envelope | N/A — integral stand/lane geometry replaces it | PENDING |
| Six actual Ø5 x 2 magnets are bonded face-flush—not bottomed—in Ø5.2 x 2.2 pockets: preserve the upper LM sites at 64°/116° with at least 2.2 mm insert gap and 0.86 mm route-cover gap; add lower LM sites at 224°/316° with at least 23.0 mm nearest-insert edge clearance, keeping 224° at z=12.55 and moving only 316° to z=15.40 for route clearance with a closed 0.30 mm front skin; verify buried-route and bridge/integral-stand clearance; retain UM sites at 50.5°/129.5°, z=15.1, with at least 1.1 mm insert/T-cover gap, 0.2 mm radial floor and 0.6 mm front skin; no proud ears | PENDING | PENDING |
| UM passage is buried only in LM and ends in a flush free-cable handoff; the UM carrier has no printed rear UM duct or D82 mouth, and the physical cable remains clear behind UM through its R15/R20 service path | PENDING | PENDING |
| T passage is buried only in LM/UM and ends in a flush free-cable handoff; the tweeter crescent has no printed T arc, conduit, socket, or horn, and the free cable remains clear behind the crescent | PENDING | PENDING |
| Physical UM/T centerlines cross at 82.67° with T above UM and retain the documented physical-envelope gap; no two-printed-duct separator web is claimed or required | PENDING | PENDING |
| Integral floor geometry: floor Y=0, LM axis height 200.981 mm, full-depth W64 stem, W64×18.3 foot over z=−150..18.3, R12 root, rear NL8 panel/service cavity and three buried Ø9/Ø8.2/Ø6 lanes; no separate support/yoke/rail artifact | PENDING | N/A — no integral stand in this state |
| All six LM sites retain ordinary blind carrier insert bores; all eight named duct bumps have continuous solid saddle material from conduit roof to bore floor | PENDING | PENDING |

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

The floor integral-root calculation is an exact rectangle-minus-Ø9/Ø8.2/
Ø6 closed-form section screen, **not FEA or certification**. Its current
project-allowable vertical 1g/3g/5g safety factors and 1g diagnostic
deflections, all including the explicit 1.25 root geometry/model factor, are:
Tough+ 3.05/1.97/1.18 and 1.18 mm; Basic 4.39/2.78/1.67 and 1.05 mm; Lite
2.69/1.73/**1.04** and 1.40 mm; Matte 2.78/1.79/1.08 and 1.49 mm; Silk+
3.23/2.09/1.25 and 1.17 mm. PLA Lite is provisional pending a
product-specific official TDS and fails the vertical-5g threshold; it is not
accepted by this screen. The section result is valid only with a **100% local-
solid modifier through the complete stem/root**; sparse infill receives no
structural credit. These numbers do not complete any row below. Magnets and
the concealed keyed-split registration feature receive 0 N.

For every selected floor material/process, add a **2× service load for 24 h
at 35 °C** gate: reject crack, whitening, insert movement, or unloaded
residual set greater than **0.5 mm or 10% of loaded deflection**. Then add a
**1.5× service load for at least 168 h** creep gate at the worst credible room
temperature. The W64 foot is not a stability qualification: calculated tip
thresholds are only 0.139g lateral, 0.348g rearward, and 0.384g forward. A
positively attached anti-tip tether or anchor is mandatory and must remain
installed throughout floor-state testing and service.

For the optional LM split, analytical tongue/socket fit and containment checks
are registration checks only and the monolithic carrier calculation is not
transferable release evidence.
Perform the complete proof with the LM driver or an equivalent flange using
all normal LM fasteners across the seam. Give the concealed pair no standalone
retention or load credit and keep the flange splice installed through
sustained 1g, transient 3g and transient 5g loading.

| Required evidence | `floor_stand` result / evidence | `no_floor_stand` result / evidence |
|---|---|---|
| Fixture drawing/ID, load-cell ID/calibration and evidence-file hashes | PENDING | PENDING |
| Actual load application coordinates and direction | PENDING | PENDING |
| Pre-test temperature, dimensions, insert positions and fastener torques | PENDING | PENDING |
| Keyed LM dedicated proof setup: driver flange/all normal LM fasteners installed across the seam, tongue fully seated, front faces coplanar, and tongue/socket standalone retention/load credit recorded as 0 N | PENDING | PENDING |
| 1g: load, 24 h duration, temperature and maximum deflection | PENDING | PENDING |
| 1g: unloaded residual deflection, insert movement, cracks and post-test torque | PENDING | PENDING |
| 3g: load, 60 s duration, temperature and maximum deflection | PENDING | PENDING |
| 3g: unloaded residual deflection, insert movement, cracks and post-test torque | PENDING | PENDING |
| 5g: load, 10 s duration, temperature and maximum deflection | PENDING | PENDING |
| 5g: unloaded residual deflection, insert movement, cracks and post-test torque | PENDING | PENDING |
| Floor 2× service load: 24 h at 35 °C, maximum deflection, crack/whitening inspection, unloaded residual ≤0.5 mm or ≤10% | PENDING | N/A — no integral stand in this state |
| Floor 1.5× service load: ≥168 h creep history at worst credible room temperature, residual set and damage inspection | PENDING | N/A — no integral stand in this state |
| Positively attached anti-tip tether/anchor installed, proof-loaded, and retained during all tests | PENDING | N/A — no integral stand in this state |
| Keyed LM dedicated driver-installed 1g proof: 24 h load/time history, temperature, maximum and residual deflection, fastener/insert movement, seam/socket damage and post-test torque | PENDING | PENDING |
| Keyed LM dedicated driver-installed 3g proof: 60 s load/time history, temperature, maximum and residual deflection, fastener/insert movement, seam/socket damage and post-test torque | PENDING | PENDING |
| Keyed LM dedicated driver-installed 5g proof: 10 s load/time history, temperature, maximum and residual deflection, fastener/insert movement, seam/socket damage and post-test torque | PENDING | PENDING |
| Keyed LM post-proof cable pull-through, service, continuity and insulation repeated successfully with the driver flange splice installed | PENDING | PENDING |
| Upper LM-to-UM and UM-to-tweeter joint proof at the documented 0.85 kg case | PENDING | PENDING |
| Integral W64 stem/foot, R12 root, three buried lanes, service cavity and NL8 panel remain sound | PENDING | N/A — no integral stand in this state |
| Front-flush solid four-hole web, stock bridge and installation substrate remain sound | N/A — no stock bridge in this state | PENDING |
| Solid-backed cable bumps remain clear, cavity-free and undamaged after every load case | PENDING | PENDING |
| Post-proof cable service, continuity and Faston removal repeated successfully | PENDING | PENDING |
| Proof repeated at maximum intended service temperature, or lower rated limit recorded | PENDING | PENDING |
| Final structural decision and any operating restrictions | PENDING | PENDING |

## Evidence inventory

| Evidence item | `floor_stand` path / SHA-256 | `no_floor_stand` path / SHA-256 |
|---|---|---|
| Dimension and coupon report | PENDING | PENDING |
| Monolithic LM physical fit and cable-service evidence | PENDING | PENDING |
| Keyed LM physical fit, tongue/socket coupon, socket-wall slicer preview, full-seat/coplanarity, route-seam metrology and cable-pull-through evidence | PENDING | PENDING |
| Keyed LM dedicated driver-installed 1g/3g/5g fixture, load-time history and post-proof cable-service evidence | PENDING | PENDING |
| Terminal service photographs/video | PENDING | PENDING |
| External cable-retention installation/load/removal evidence | PENDING | PENDING |
| Structural fixture and load-time history | PENDING | PENDING |
| Floor 2×/24 h/35 °C proof and 1.5×/≥168 h creep histories | PENDING | N/A |
| Floor anti-tip tether/anchor installation and proof evidence | PENDING | N/A |
| Pre/post metrology and torque report | PENDING | PENDING |
| Electrical continuity/insulation report | PENDING | PENDING |

## State-specific release signoff

The two LM print-form decisions below are mandatory for every state manifest.
The final state decision may be `PASS — RELEASE AUTHORIZED` only when both form
decisions pass and every shared state requirement also passes.

| Signoff field | `floor_stand` | `no_floor_stand` |
|---|---|---|
| Evidence reviewed by / date | PENDING | PENDING |
| Mechanical proof witnessed by / date | PENDING | PENDING |
| Terminal/electrical service witnessed by / date | PENDING | PENDING |
| Deviations and operating limits accepted by | PENDING | PENDING |
| Monolithic LM form evidence reviewed by / date | PENDING | PENDING |
| Monolithic LM form authorization: `PASS — FORM AUTHORIZED` or `FAIL/PENDING` | FAIL / PENDING | FAIL / PENDING |
| Keyed LM form evidence reviewed by / date | PENDING | PENDING |
| Keyed LM form authorization: `PASS — FORM AUTHORIZED` or `FAIL/PENDING` | FAIL / PENDING | FAIL / PENDING |
| Release approver printed name / role | PENDING | PENDING |
| Release approver signature / date | PENDING | PENDING |
| Final state decision: `PASS — RELEASE AUTHORIZED` or `FAIL/PENDING` | FAIL / PENDING | FAIL / PENDING |

Release gate: **FAIL / PENDING FOR BOTH STATES**. The modeled 12 mm pull
equals the provisional 12 mm exposed-tab length, so the current proxy has
zero positive release overtravel margin. Do not infer physical fit, structural
qualification or release authority from this template, generated geometry,
passing analytic tests, or a candidate manifest.
