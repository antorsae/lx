# Obi-Wan physical qualification record

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
and `obiwan_optional_lm_keyed_1_of_2_bottom.stl` each own the complete
integral W64 stem/foot, convex constant-thickness R41-minimum Option-B
transition, buried floor lanes and NL8 panel. The floor
datum is world `Y=0`, exactly **200.981 mm** below the LM axis. All six LM
driver insert sites are ordinary blind carrier bores in both states.

At the LM-to-UM joint axes x=±32 mm, y=315.770 mm, the monolithic LM—or the
selected optional keyed LM top—must independently contain both rear Ø3.4
screw-clearance passages. The standalone UM must independently contain both
rear-opening blind Ø4.6 x 4.0 M3 heat-set receivers and their complete walls,
with a 1.9 mm solid acoustic-front floor. The closure-web/base teardrop remains
nominal Ø9, while each complete Z-owned cylindrical functional boss is locally
Ø9.8. Install the inserts in that individual UM print before assembly. The ear
halves retain a 0.20 mm axial gap and are
joined by rear-driven M3 screws only; no washer, nut, front bolt head, or
through-drilled UM floor is permitted.

At the UM-to-tweeter axes x=±24 mm, y=421.5 mm, the standalone UM must
independently contain both rear Ø3.4 screw-clearance passages. The standalone
crescent must independently contain both rear-opening blind Ø4.6 x 4.0 M3
heat-set receivers, complete 360° walls, and 1.9 mm solid acoustic-front
floors. Here too the closure-web/base teardrop remains nominal Ø9 while each
complete Z-owned cylindrical functional boss is locally Ø9.8. Install both
inserts in the individual crescent before assembly. The 0.20 mm axial gap may
not be filled by either print, and no receiver wall or floor may depend on UM
material.

Each state release manifest ships both the canonical monolithic LM carrier and
both halves of the optional keyed LM form. Although an installed assembly uses
exactly one form, release qualification must authorize **both shipped forms
independently**. Authorization of one form does not authorize the other, and
`N/A` is not permitted for either shipped form's identity, required evidence,
or print-form decision. A state remains unauthorized until both print-form
decisions pass. The keyed form is cut at world `Y=172.481 mm` with a zero-gap
planar butt; neither keyed half may be combined with the monolithic carrier in
an installed assembly. The bottom owns two symmetric concealed Ø1.60
cylindrical pins at `x=±109.187 mm`, `z=14.30 mm`; both point world +Y normal
to the seam, have 0.50 mm root overlap, and engage the top by 2.40 mm (2.90 mm
total male length). The top owns 2.65 mm-deep blind sockets with 0.12 mm radial
and 0.25 mm end clearance: right is round Ø1.84, while left is X-relieved to
1.96 × 1.84 mm. This round-plus-relieved constraint accepts ±0.30 mm relative
pitch error across the 218.374 mm spacing instead of binding like two tight
round sockets. Small exterior lands outside the LM recess reach R114.4036:
1.4036 mm beyond structural R113.0 and 0.6036 mm beyond the finalized R113.8
visible fairing, but add no extra screw or standalone retention/load
credit. Flat/graded provide 0.25 mm clearance pockets around the lands, wholly at
the hidden carrier interface between their front and rear faces; printed fit
remains coupon-qualified.
Its installed load path exists only after the LM driver flange and all normal
LM fasteners splice the seam.

With the monolithic LM, the flat/graded pockets remain as small hidden local reliefs.
The three magnetic datums and primary wing-retention geometry are unchanged,
but the former local saddle contact is not claimed to be geometrically identical.

The keyed sockets' current CAD contract retains at least **0.50 mm local
radial and blind-end wall**, **0.05 mm recess plan clearance**, and **0.13 mm
conservative W22-flange plan clearance**. Each horizontal pin is four nominal
0.4 mm nozzle widths,
but positive CAD wall checks are not release evidence by themselves. The
candidate slicer preview must show both complete pin toolpaths, both open blind
mouths, both support lands, and continuous intended extrusion paths at all
minimum socket walls. A state/process-matched two-pin/socket coupon and actual
U22 fit must pass before the keyed form can be authorized.

Every released acoustic candidate prints front-face-down, whether or not it
contains a magnet. The authoritative pause
and polarity record is `review/captive_magnet_slice_audit/CAPTIVE_MAGNET_PAUSE_MANIFEST.md`; an STL alone is not a valid
magnet-print instruction. At every pause, verify the marked pole and count,
seat each disc below the completed layer, clear the future roof toolpath, and
only then resume. A buried polarity error is irreversible and rejects the part.
Magnets remain alignment/anti-rattle devices and receive zero structural-load
credit even after this process passes.

The approved station uses an actual D5.0 × 2.0 disc in an Ø5.20 × 2.10 cavity,
0.45 mm axial skins on both faces, and a support-free 45° roof. No glue or
external access or local exterior location cue is permitted. All LM-lower,
LM-upper, and UM stations share the common source plane **Z = 15.10 mm**. The
LM-upper and UM ring stations retain
structural radii R113.0/R51.7 beneath continuous exposed R113.8/R52.5 side
fairings. Those fairings stop only inside the existing LM--UM and T--UM
cusp/service regions; the LM--UM stop preserves the 0.40 mm inter-carrier
gap. Their cavity construction datum is structural radius
+0.65 mm, 0.15 mm beneath the exposed surface, with no local pad, boss, flat,
or visible cue. Opposing carrier and flat/graded solids have zero physical mating
gap. The receiver cavity datum instead includes a 0.05 mm solid construction
standoff. Opposing magnet faces are therefore nominally 1.10 mm apart at
LM-lower, LM-upper, and UM (`0.45 + 0.15 + 0.05 + 0.45`). The LM-lower datum
is cubic parameter `u=0.50` on the shared curved shoulder; its right visible
point is `(45.285011,89.190370)` with outward normal
`(0.706451,-0.707762)`, and its left point is the exact X mirror. The D5 × 2
geometry and both 0.45 mm skins are unchanged.

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
| `review/captive_magnet_slice_audit/CAPTIVE_MAGNET_PAUSE_MANIFEST.md` revision / SHA-256 and applied pause markers | PENDING | PENDING |
| D5 × 2 magnet supplier, grade, lot and marked-pole convention | PENDING | PENDING |
| Heat-set manufacturer, part number and lot for M3 and M5 inserts, including the two LM-to-UM M3 x 3 inserts installed in the standalone UM carrier and two UM-to-tweeter M3 x 3 inserts installed in the standalone crescent | PENDING | PENDING |
| Insert tool, set temperature, dwell/process and operator | PENDING | PENDING |
| Screw, driver-clamp washer, stock-bridge hardware, unrelated through-bolt/nut hardware, and floor anti-tip tether/anchor specification; confirm both upper interfaces use rear-driven M3 screws only, with no washer/nut or front bolt head | PENDING | PENDING |
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
| Keyed LM bottom `obiwan_optional_lm_keyed_1_of_2_bottom.stl` SHA-256 — mandatory shipped form | PENDING | PENDING |
| Keyed LM top `obiwan_optional_lm_keyed_2_of_2_top.stl` SHA-256 — mandatory shipped form | PENDING | PENDING |
| Keyed split print orientation and verified in-bed footprint (bottom / top) | Both front-face-down with in-plane rotation only; record actual selected-printer clearances | Both front-face-down with in-plane rotation only; record actual selected-printer clearances |
| UM carrier STL SHA-256 | PENDING | PENDING |
| Tweeter crescent STL SHA-256, if fitted | PENDING | PENDING |
| Selected external cable-retention identity, if fitted | PENDING | PENDING |
| Integral floor LM geometry identity: W64 stem/foot, convex constant-thickness Option-B transition (75 mm span, 65 mm rise, centreline Rmin 41 mm), NL8 panel, floor Y=0 and 200.981 mm LM-axis height | PENDING | N/A — no integral stand in this state |
| Positive check that `lx521_top_obiwan_addon_mount_floor_support.stl` is absent | PENDING | PENDING |
| Stock bridge serial/revision and installation substrate | N/A — integral LM owns this load path | PENDING |
| Floor anti-tip tether/anchor make, attachment points, rating and installation record | PENDING | N/A — no integral floor stand in this state |
| Final slicer project / G-code / print-job hashes | PENDING | PENDING |
| G-code evidence that each pause is the first roof-closing layer after the last fully open layer | PENDING | PENDING |
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
| Keyed LM form physical fit: both printed halves front-face-down on one flat datum; top translated straight along world -Y so the two pins enter together without flex/twist; actual U22 flange and all normal LM fasteners installed; full seating, front-datum coplanarity and post-fit crack/damage inspection | PENDING | PENDING |
| Keyed LM dedicated two-pin/socket coupon: candidate material/process; two symmetric Ø1.60 +Y pins, 2.40 mm engagement; right Ø1.84 round and left 1.96 × 1.84 X-relieved blind sockets; assembly force; simultaneous straight -Y insertion; no pin used as a hinge | PENDING | PENDING |
| Keyed LM pin/socket slicer and thin-wall acceptance: CAD minima confirmed at ≥0.50 mm radial/end wall, ≥0.05 mm recess plan clearance and ≥0.13 mm conservative W22-flange plan clearance; candidate preview shows both complete four-nozzle-width horizontal pins, both blind mouths, both exterior support lands and continuous intended paths at all minimum walls; state/process-matched coupon has no gaps, droop, delamination or breakage; actual U22 fit clears the lands; flat/graded 0.25 mm hidden interface pockets clear both actual staged halves and pass a printed fit coupon | PENDING | PENDING |
| Keyed LM seam metrology: world Y=172.481 mm, closed zero-gap planar butt, no volumetric half overlap, both front faces registered on one flat datum, and continuous UM/T route seams | PENDING | PENDING |
| Keyed LM physical UM/T cable pull-through across both preserved seam lumens without snag, insulation damage or slicer-support residue, followed by continuity/insulation verification | PENDING | PENDING |
| Assembled LM–UM and T–UM closure-web inspection: all owners front-face-down on one flat datum; z=18.3 faces coplanar; solid tangent-blended webs seated across the 0.05 mm plan seams; each anti-void lens retains a continuous 0.45 mm fusion land and a visible Arachne wall path; no triangular/cusp opening, bounded front-plane void island, thin front skin, rear hollow, or sub-resolution detached shard; only the declared T cable mouth and functional fastener/interface/route clearances remain | PENDING | PENDING |
| Standalone LM-to-UM fastener features before assembly: the closure-web/base teardrops remain nominal Ø9; both monolithic LM or selected optional keyed LM-top ears have complete local Ø9.8 Z-owned cylindrical functional bosses and unobstructed rear Ø3.4 clearance passages; both UM ears have complete local Ø9.8 bosses, independently accessible rear-opening blind Ø4.6 x 4.0 receivers, complete 360° walls, and 1.9 mm solid acoustic-front floors; no feature depends on material from the opposing print | PENDING | PENDING |
| LM-to-UM insert installation before assembly: both M3 x 3 heat-sets installed square and flush through the individual UM carrier's rear/mating openings; no cracking, lateral opening, over-melt, insert motion, receiver-wall damage, or acoustic-face mark/breakthrough; insertion process, temperature, dwell, and photographs recorded | PENDING | PENDING |
| LM-to-UM dry assembly: complete LM and UM ears engage with the specified 0.20 mm axial gap; rear-driven screws pass through the LM Ø3.4 bores and achieve full UM-insert engagement without bottoming; both front faces remain coplanar; no washer, nut, or front bolt head is present | PENDING | PENDING |
| Standalone UM-to-tweeter fastener features before assembly: both UM ears have complete local Ø9.8 Z-owned bosses and unobstructed rear Ø3.4 passages; both crescent ears have complete local Ø9.8 bosses, independently accessible rear-opening blind Ø4.6 x 4.0 receivers, complete 360° walls, and 1.9 mm solid acoustic-front floors; no feature depends on material from the opposing print | PENDING | PENDING |
| UM-to-tweeter insert installation before assembly: both M3 x 3 heat-sets installed square and flush through the individual crescent's rear/mating openings; no cracking, lateral opening, over-melt, insert motion, receiver-wall damage, or acoustic-face mark/breakthrough; insertion process, temperature, dwell, and photographs recorded | PENDING | PENDING |
| UM-to-tweeter dry assembly: complete UM and crescent ears engage with the specified 0.20 mm axial gap; rear-driven screws pass through the UM Ø3.4 bores and achieve full crescent-insert engagement without bottoming; both front faces remain coplanar; no washer, nut, or front bolt head is present | PENDING | PENDING |
| Process-matched M3 insert pullout/pry qualification for both D4.6-receiver interfaces: test the individual UM and individual crescent receiver constructions to at least the documented 5g demand of 393.9 N per insert without insert motion, wall opening, floor damage, or boss fracture | PENDING | PENDING |
| Terminal carrier radius and rear Z | PENDING | PENDING |
| Terminal pitch, tab width/thickness and exposed length | PENDING | PENDING |
| Polarity order | PENDING | PENDING |
| Actual withdrawal axis, release stroke and peak force for each terminal | PENDING | PENDING |
| Positive disengagement/handling margin beyond the modeled 12 mm state | PENDING | PENDING |
| Measured receptacle and installed boot envelopes | PENDING | PENDING |
| One-at-a-time pull at 0/3/6/9/12 mm with opposite side installed | PENDING | PENDING |
| Installed LM / UM / T cable outer diameters | PENDING | PENDING |
| Cable-manufacturer minimum static and repeated-flex bend radii | PENDING | PENDING |
| Free D7.8 LM lead follows the 20.15 mm / 269.5° rear span without a printed micro-duct; the LM carrier has a minimum-radius 3.96 mm rear-open subtractive clearance around center z=0.40..3.80, with 1.00 mm outer-station clearance to the deepest z=5.3 pad/web rear datum; floor state continues through the Ø9 buried integral-stem lane | PENDING | PENDING |
| Finished OD8 bundle and both OD4 branch heat-shrink dimensions | PENDING | PENDING |
| Y-junction continuity, insulation, strain transfer and polarity labels | PENDING | PENDING |
| Selected external retention dimensions, material and buried-route/free-cable/service-envelope clearance | PENDING | PENDING |
| External retention installation around the terminated cable without shell damage | PENDING | PENDING |
| External retention load and deliberate tool/finger removal access | PENDING | PENDING |
| Full service-motion clearance to the physical MU10 and installed U22 | PENDING | PENDING |
| Final free-LM placement plus UM/T buried-span fishing and free-span placement, electrical continuity and insulation test | PENDING | PENDING |
| Final strain-relief pull transfers cable load away from MU tabs | PENDING | PENDING |
| No-floor bridge plate has a soft cubic blend into R113 and occupies z=5.3..18.3; four rear insert bores retain the unchanged 6.8 mm total depth and 6.2 mm front floor, with a Ø6.5 x 2.0 entry followed by Ø6.4; LM/T/UM mouths are packed wholly inside the D20 support opening as LM above, T lower-left, UM lower-right and open only at rear z=5.3; no geometry extends behind the existing LM-pad envelope | N/A — integral stand/lane geometry replaces it | PENDING |
| Six actual Ø5 x 2 magnets are pause-inserted and fully buried in Ø5.20 x 2.10 surface-normal cavities with continuous 0.45 mm axial skins and support-free 45° roofs: preserve upper LM axes 64°/116°, the lower LM pair at cubic parameter `u=0.50` on the shared shoulder, with right visible datum `(x,y,z)=(45.285011,89.190370,15.10)`, outward normal `(0.706451,-0.707762)`, and an exact-X-mirrored left datum, and UM axes 50.5°/129.5° at the same common source Z=15.10; verify R113.0/R51.7 structural rings, smooth exposed R113.8/R52.5 side fairings clipped only inside the existing LM--UM and T--UM cusp/service regions with the 0.40 mm LM--UM inter-carrier gap preserved, and ring cavity construction datums at structural radius +0.65 mm / 0.15 mm beneath the exposed surface; verify there is no local pad, boss, flat, visible cue, external access, or proud ear; verify floor/no-floor station coincidence, route/insert/structure keepouts, three matching flat/graded receivers per side, a 0.05 mm solid receiver construction standoff with zero physical mating gap, 1.10 mm nominal magnet-face separation at LM-lower, LM-upper, and UM | PENDING | PENDING |
| Every released acoustic STL printed front-face-down; for each magnet-bearing STL, sliced preview/G-code records the lowest open, representative open, last fully open, first closing and fully sealed layers; retaining walls remain continuous; each disc was inserted with manifest polarity, fully seated below the completed layer and clear of the resumed nozzle path | PENDING | PENDING |
| Coupon-equivalent regression marker on the tested P2S 0.4 mm / 0.16 mm Arachne profile is Z=5.96 mm for every common-plane Obi-Wan LM/UM transverse station; unrelated families use their own sliced schedules rather than copying this value | PENDING | PENDING |
| UM passage is buried only in LM and ends in a flush free-cable handoff; the UM carrier has no printed rear UM duct or D82 mouth, and the physical cable remains clear behind UM through its R15/R20 service path | PENDING | PENDING |
| T passage is buried only in LM/UM and ends in a flush free-cable handoff; the tweeter crescent has no printed T arc, conduit, socket, or horn, and the free cable remains clear behind the crescent | PENDING | PENDING |
| Physical UM/T centerlines cross at 82.95° with T above UM and retain a 2.00 mm physical-envelope gap; both LM-owned lumens finish at R112.95 and their 0.8 mm covers at R113.75 beneath the continuous visible R113.8 carrier exterior, retaining a 0.85 mm solid outside skin with no groove; no two-printed-duct separator web is claimed or required | PENDING | PENDING |
| Integral floor geometry: floor Y=0, LM axis height 200.981 mm, full-depth W64 stem, W64×18.3 foot over z=−150..18.3, convex constant-thickness Option-B transition (75 mm span, 65 mm rise, centreline Rmin 41 mm), rear NL8 panel/service cavity and three buried Ø9/Ø8.2/Ø6 lanes; no separate support/yoke/rail artifact | PENDING | N/A — no integral stand in this state |
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
electrical damage or impaired cable service. At both interfaces, inspect the
two heat-sets, complete receiver walls, and 1.9 mm front floors after every
load case; the M3 screw-tension screen does not independently qualify insert
pull-out, receiver pry-out, or front-floor integrity.

The no-floor analytical screen deducts the complete Ø9/Ø8.2/Ø6.0 D20-packed
entry lumens from the 62 mm insert core and credits a conservative
38.8 × 13.0 mm member; exact sampled soft-outline cuts retain at least
45.73 mm. Its calculated 5g factor is about 0.82, so 5g is not analytically
qualified. The fusion screen separately credits only the actual
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
both concealed keyed-split pins/sockets receive 0 N.

For every selected floor material/process, add a **2× service load for 24 h
at 35 °C** gate: reject crack, whitening, insert movement, or unloaded
residual set greater than **0.5 mm or 10% of loaded deflection**. Then add a
**1.5× service load for at least 168 h** creep gate at the worst credible room
temperature. The W64 foot is not a stability qualification: calculated tip
thresholds are only 0.139g lateral, 0.348g rearward, and 0.384g forward. A
positively attached anti-tip tether or anchor is mandatory and must remain
installed throughout floor-state testing and service.

For the optional LM split, analytical two-pin/socket fit and containment
checks are registration checks only and the monolithic carrier calculation is
not transferable release evidence.
Perform the complete proof with the LM driver or an equivalent flange using
all normal LM fasteners across the seam. Give both concealed pins/sockets no
standalone retention or load credit and keep the flange splice installed through
sustained 1g, transient 3g and transient 5g loading.

| Required evidence | `floor_stand` result / evidence | `no_floor_stand` result / evidence |
|---|---|---|
| Fixture drawing/ID, load-cell ID/calibration and evidence-file hashes | PENDING | PENDING |
| Actual load application coordinates and direction | PENDING | PENDING |
| Pre-test temperature, dimensions, insert positions and fastener torques | PENDING | PENDING |
| Keyed LM dedicated proof setup: driver flange/all normal LM fasteners installed across the seam, both pins fully seated in their identified right-round/left-relieved sockets, front faces coplanar, and pin/socket standalone retention/load credit recorded as 0 N | PENDING | PENDING |
| 1g: load, 24 h duration, temperature and maximum deflection | PENDING | PENDING |
| 1g: unloaded residual deflection, insert movement, cracks and post-test torque | PENDING | PENDING |
| 3g: load, 60 s duration, temperature and maximum deflection | PENDING | PENDING |
| 3g: unloaded residual deflection, insert movement, cracks and post-test torque | PENDING | PENDING |
| 5g: load, 10 s duration, temperature and maximum deflection | PENDING | PENDING |
| 5g: unloaded residual deflection, insert movement, cracks and post-test torque | PENDING | PENDING |
| Floor 2× service load: 24 h at 35 °C, maximum deflection, crack/whitening inspection, unloaded residual ≤0.5 mm or ≤10% | PENDING | N/A — no integral stand in this state |
| Floor 1.5× service load: ≥168 h creep history at worst credible room temperature, residual set and damage inspection | PENDING | N/A — no integral stand in this state |
| Positively attached anti-tip tether/anchor installed, proof-loaded, and retained during all tests | PENDING | N/A — no integral stand in this state |
| Keyed LM dedicated driver-installed 1g proof: 24 h load/time history, temperature, maximum and residual deflection, fastener/insert movement, seam/pin/socket damage and post-test torque | PENDING | PENDING |
| Keyed LM dedicated driver-installed 3g proof: 60 s load/time history, temperature, maximum and residual deflection, fastener/insert movement, seam/pin/socket damage and post-test torque | PENDING | PENDING |
| Keyed LM dedicated driver-installed 5g proof: 10 s load/time history, temperature, maximum and residual deflection, fastener/insert movement, seam/pin/socket damage and post-test torque | PENDING | PENDING |
| Keyed LM post-proof cable pull-through, service, continuity and insulation repeated successfully with the driver flange splice installed | PENDING | PENDING |
| Upper LM-to-UM joint proof at the documented 0.85 kg case: both local Ø9.8 functional bosses remain intact; rear-driven M3 screws remain engaged in both UM-owned inserts; no insert motion/pull-out, receiver-wall crack/pry-out, 1.9 mm front-floor damage, loss of 0.20 mm axial-gap control, or post-test torque loss | PENDING | PENDING |
| Upper UM-to-tweeter joint proof at the documented 0.85 kg case: both complete local Ø9.8 UM/crescent bosses remain intact; rear-driven M3 screws remain engaged in both crescent-owned inserts; no insert motion/pull-out, receiver-wall crack/pry-out, 1.9 mm front-floor damage, loss of 0.20 mm axial-gap control, or post-test torque loss | PENDING | PENDING |
| Integral W64 stem/foot, convex R41-minimum Option-B transition, three buried lanes, service cavity and NL8 panel remain sound | PENDING | N/A — no integral stand in this state |
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
| Keyed LM physical fit, two-pin/socket coupon, pin/socket-wall slicer preview, full-seat/coplanarity, route-seam metrology and cable-pull-through evidence | PENDING | PENDING |
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
