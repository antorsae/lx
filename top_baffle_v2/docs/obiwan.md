# Obi-Wan R6F — extreme two-collar barebone

Obi-Wan removes the full baffle outline and keeps only what carries a driver:
one LM collar and one UM collar, plus optional tweeter crescent and optional
Ac/Ae acoustic wings. It shares no seam, duct, or attachment interface with the
R6P proud family. Catalog entry: [`artifacts/obiwan/`](../artifacts/obiwan/).

This product is a **candidate**: its state manifests record
`release_authorized: false`, and the physical qualification record in
[`obiwan_physical_qualification.md`](obiwan_physical_qualification.md) is
pending.

## Source modules

| File | What |
|---|---|
| `src/lx521_baffle/obiwan/carriers.py` / `src/lx521_baffle/obiwan/split.py` | Extreme Obi-Wan core: structural LM/UM flush-driver collars at R113.0/R51.7 with smooth exposed R113.8/R52.5 side fairings clipped only inside the existing LM--UM and T--UM cusp/service regions, with the 0.40 mm LM--UM inter-carrier gap preserved; rounded LM-to-UM M3 half-laps whose closure-web/base teardrops remain nominal Ø9 while each complete Z-owned cylindrical functional boss is locally Ø9.8, with standalone rear Ø3.4 LM clearance bores and standalone rear-opening blind Ø4.6 x 4.0 UM heat-set receivers; six pause-and-bury captive magnet stations (two upper LM ring-radial, two lower LM shoulder-normal, and two UM ring-radial), all with cavity datums hidden 0.15 mm beneath a continuous carrier surface and no local pad/boss/flat/cue; buried UM/T route spans; and free rear cable continuations. Floor and no-floor share the exact upper LM shoulder used by the wings. Floor has no shallow material below its y=60 shoulder tangent; no-floor alone retains the shallow four-insert bridge. |
| `src/lx521_baffle/obiwan/lm_split.py` | Optional, mutually exclusive two-print form of the finalized Obi-Wan LM carrier: exact zero-gap world-Y butt seam plus two symmetric Ø1.60 cylindrical pins normal to the seam (world +Y). The pins engage 2.40 mm; the right blind socket is round Ø1.84 and the left is X-relieved to 1.96 × 1.84 mm so the 218.374 mm pitch cannot bind like two tight round fits. Two tiny exterior lands outside the LM recess retain 0.12 mm radial and 0.25 mm end clearance, at least 0.50 mm local radial/end wall, at least 0.05 mm recess plan clearance and 0.13 mm conservative W22-flange clearance. Their worst-case reach is R114.4036: 1.4036 mm beyond the structural R113.0 ring and 0.6036 mm beyond the finalized R113.8 visible fairing. They add no extra fastener or standalone retention/load credit; the monolithic LM remains canonical. |
| `src/lx521_baffle/obiwan/route.py` | Exact R6F printed-owner segments and physical cable continuations: 0.8 mm minimum walls and 0.85 mm seat roof on the surviving buried UM/T spans; no-floor LM/T/UM entries packed inside the one D20 support opening; LM-owned UM/T envelopes buried 0.05 mm beneath their outside owner limits, leaving a continuous 0.85 mm skin to visible R113.8 with no groove; full-width burial webs and solid roof-to-bore saddles; free UM behind the UM carrier; free T behind the crescent; and the 82.95° crown crossing |
| `src/lx521_baffle/obiwan/bridge.py` | Universal lower-LM front profile (filled exterior union of the historical floor stem and no-floor bridge), immutable no-floor four-hole datum, fused 62 mm insert core with soft cubic shoulders and three rear cable entries packed in the D20 support opening at the deepest existing LM-pad depth (no separate keel or rear ribs), hardware proxies, and an opening-aware biaxial 4 kg sustained-1g/3g/5g structural screen |
| `src/lx521_baffle/obiwan/floor.py` / `src/lx521_baffle/obiwan/floor_strength.py` | Floor-only integral W64 stem/foot with the constant-thickness convex Option-B transition (75 mm span, 65 mm rise, centreline Rmin 41 mm), rear NL8 panel/service cavity and three buried cable continuations; closed-form five-material net-section screen. This is part of the LM carrier, not an add-on, and the analysis is not FEA or physical qualification. |
| `src/lx521_baffle/obiwan/attachments.py` | Optional tweeter crescent with complete standalone blind-M3 receiver ears; any cable retention is external/non-modeled, and magnets receive zero structural load credit |
| `src/lx521_baffle/obiwan/assembled.py` | Review assembly containing the R6F core, selected add-ons, and the explicitly non-manufacturing terminal/Faston proxy |
| `src/lx521_baffle/obiwan/wings.py` | STEP-first Ac/Ae Obi-Wan acoustic attachments: one canonical monolith per side, three exact surface-normal captive D5 × 2 magnet receivers per side (LM lower shoulder, LM upper, and UM), one saddle compatible with the shared floor/no-floor upper LM shoulder, the approved constant-depth Ac or monotonic LM/UM/T-weighted Ae rear, and two print options. Both wings start at the Option-B vertical tangent y=74.15 and grow outward through a G1 cubic that joins the outer flank at the LM-aperture lower tangent y=105.981, so no wing panel hangs below the bend. Option A preserves the original three exact-mask pieces; option B keeps the identical lower piece and fuses LM-upper plus UM into one upper piece by restoring only their former clearance seam. Each physical side retains the lower→upper V1L-style through-local-thickness XY dovetail; option A also retains its middle→UM dovetail. Female clearance is 0.05 mm, exposed split clearance closes over the final 2 mm at both endpoints, and the keys add no envelope growth. They register/interlock in XY but provide no independent Z retention. Ae’s complete internal protected-land perimeter is accepted only when paired actual-BREP probes show a C0 jump ≤0.03 mm |
| `scripts/export_obiwan_wings.py` | Transactional Ac/Ae exporter: canonical/A/B assembled STEP, ten strict front-face-down STLs with ten exact adjacent `.print.json` authorities, facts, hash manifest, and CAD-derived QA renders under `build/wings/ac/` or `build/wings/ae/`; every review PNG uses hash-validated staged BREPs for a neutral no-floor LM-upper/UM/tweeter reference plus the two coincident LM-lower outlines—blue dash-dot for no-floor and green dotted for floor stand; the side view keeps its useful acoustic-depth scale and includes a complete-depth floor inset |
| `tests/test_obiwan_wings.py` | Remote-only Ac/Ae BREP, print-inventory, STEP, STL, mirror, depth, receiver, dovetail/clearance, endpoint-closure, bed-fit, provenance, render, and exact dual-state lower-LM front-profile gates |

## Geometry and interfaces

Obi-Wan is no longer a flush-recessed copy of the full outline. Its
mandatory geometry is only:

- an LM flush carrier with Ø190 opening, Ø221.2 seat, **R113.0 structural
  radius**, and a smooth **R113.8 exposed side radius**;
- an UM flush carrier with Ø82 opening, Ø98.6 seat, **R51.7 structural
  radius**, and a smooth **R52.5 exposed side radius**;
- two compact half-lap pairs at x=±32.0, y=315.770 that establish the
  165.100 mm driver-center spacing without entering either flange seat. Their
  closure-web/base teardrops remain nominal **Ø9**, while each complete
  Z-owned cylindrical functional boss is locally **Ø9.8** to preserve the
  joint screen with the Ø4.6 receiver. Each LM rear Z-half owns a complete
  standalone Ø3.4
  rear-driven screw-clearance passage; each UM front Z-half owns a complete
  standalone rear-opening blind Ø4.6 x 4.0 receiver for an M3 x 3 heat-set.
  The receiver retains a **1.9 mm solid acoustic-front floor**, and the LM and
  UM ear halves retain a **0.20 mm axial gap**. Install the inserts in the
  individual UM print before assembly, then drive the screws from the LM rear;
  this interface has no washer, nut, or front bolt head;
- exactly six surface-normal D5×2 alignment/anti-rattle interfaces using
  captive Ø5.20 × 2.10 cavities: four LM and two UM. Each magnet is enclosed
  between 0.45 mm axial skins and a self-supporting 45° closing roof, with no
  glue or external access opening. The upper LM pair retains the world polar
  64°/116° axes (±26° from top), has no proud ear, and retains at least 2.2 mm cavity-edge to the nearest
  insert-pad edge and 0.86 mm to its route covers. The lower LM pair is
  captive at parameter 0.5 of the shared curved shoulder. The right visible
  datum is **`(x,y,z)=(45.285011,89.190370,15.10)`** with outward normal
  `(0.706451,-0.707762)`; the left is its exact mirror. The cavity datums are
  0.15 mm inward from those continuous surfaces. These shoulder stations are
  identical in floor/no-floor carriers, lie wholly above the Option-B bend
  tangent, and clear the inserts, buried routes, and load path. The upper LM
  pair and the UM pair use
  that same source Z=15.10; the UM pair retains its 50.5°/129.5° axes. The LM
  and UM structural ring radii stay
  R113.0/R51.7, while their exposed sides are continuous cylindrical
  R113.8/R52.5 fairings. The fairings stop only inside the existing LM--UM and
  T--UM cusp/service regions; the LM--UM stop keeps the 0.40 mm inter-carrier
  gap open. At each ring-radial or lower-shoulder station the
  cavity construction datum is structural radius **+0.65 mm**, or **0.15 mm
  beneath the exposed surface**. The D5×2 cavity and 0.45 mm skin remain
  unchanged, and there is no magnet-local backing, boss, relief, rear cap,
  flat, or visible pocket cue. The exterior is the immutable magnet-free
  carrier surface. All
  six have
  **zero structural load credit**;
- Ac and Ae provide three coaxial captive receivers on each physical
  side—one at LM lower, one at LM upper, and one at UM—so all six carrier
  magnet axes have matching wing cavities. The mating surfaces are flush with
  zero physical air gap; the receiver retains 0.05 mm as a solid internal
  spacing standoff. Nominal paired magnet-face separation is **1.10 mm** at
  LM-lower, LM-upper, and UM (`0.45 + 0.15 + 0.05 + 0.45`);
- six Obi-Wan-only LM axes at 0/60/120/180/240/300° on radius 104.75 mm,
  leaving the crown clear; both states own six ordinary blind carrier
  inserts;
- two compact direct UM-to-tweeter half-lap ears at **x=±24,
  y=421.5**. UM owns complete rear ears with Ø3.4 passages; the crescent owns
  complete front ears with rear-opening blind Ø4.6 x 4.0 receivers, 360°
  walls, and 1.9 mm acoustic-front floors, so each part is independently
  printable and no fastener breaks the acoustic front; and
- complementary, tangent-blended **full-depth** closure webs at both
  LM–UM and T–UM junctions. LM owns the lower LM–UM web, UM owns the upper
  LM–UM and lower T–UM webs, and the tweeter crescent owns the upper T–UM
  webs. Every owner spans z=6.8..18.3 and overlaps its own ring/crescent by
  0.40 mm; the local anti-void lens fills retain a separate 0.45 mm
  Arachne-compatible fusion land. Sub-resolution Boolean shards are discarded,
  while the independently printed owners retain the normal 0.05 mm plan
  seam. The functional bosses at both LM-to-UM and UM-to-tweeter are a
  Z-owned exception: each base closure teardrop remains nominal Ø9, but every
  complete cylindrical boss is locally Ø9.8. Each ear remains wholly in its
  assigned axial half, and the opposing print is fully notched over that half
  so the plan seam cannot split a bore, receiver wall, or front floor. The
  separate 0.20 mm axial gaps remain open. These are
  solid members behind the common z=18.3 front plane,
  not front skins over cavities; the only non-functional opening between the
  upper rings is the central ±6 mm T free-cable mouth; and
- an Ø8.2 UM passage buried only in the LM carrier and an Ø6.0 T passage
  buried in the LM and UM carriers, each with 0.8 mm minimum walls and a
  0.85 mm seat roof on its printed span. The UM cable exits the LM passage
  and remains free behind the UM carrier; there is no printed UM-carrier rear
  duct or D82 mouth. T exits the UM passage and remains free behind the
  tweeter crescent; the crescent has no printed cable arc. Their physical
  centerlines cross at 82.95° with T above UM and retain a 2.00 mm physical
  envelope gap. In no-floor state the D7.8 LM lead uses the buried Ø9 branch
  from the D20 entry cluster; floor state uses its integral Ø9 lane. The
  LM-owned UM/T lumens finish at R112.95 and their covers at R113.75 beneath
  the uninterrupted R113.8 carrier exterior, leaving a 0.85 mm solid outside
  skin and no groove.

The load-bearing outer lips extend 2.4 mm past the flange-seat radii. Smooth
0.8 mm side fairings cover those structural lips at exposed radii R113.8 and
R52.5; they are clipped only inside the existing LM--UM and T--UM cusp/service
regions, and the LM--UM stop keeps the nominal 0.4 mm gap between the
structural collar envelopes open. The LM's six
insert-pad buttons, both pilot patterns, and flush seats remain; the old
5.5/7.5 mm annular floors and perimeter skin have been removed. Each seat
keeps only a 0.85 mm two-extrusion membrane. Narrow outer lips, local
blind-insert floors/bosses, calculated spokes, surviving buried-route covers, and
the explicit mechanical interfaces are the retained material.
The guarded closure acceptance clips the actual independently printable
LM/UM/crescent BREPs through fixed physical windows, not a window generated
from the closure target. It checks the actual front-face-down Bambu schedule
(0.20 mm first layer, then 0.16 mm layers) plus both sides of
each half-lap transition against frozen conservative front silhouettes,
proves the standalone LM clearance passages and UM blind receivers retain
their complete local Ø9.8 functional bosses, 360° walls, and the 1.9 mm UM
front floor; rejects exact 3-D owner overlap and proud material above z=18.3;
and rejects
any bounded residual void component beyond the declared fit seams, fastener
interfaces, route lumen, and T cable mouth. Thus a self-shrunk target, an open
cusp connected to a driver aperture, or a thin front skin over a rear cavity
cannot satisfy the release gate.

The canonical LM carrier remains one monolithic large-format release part.
Its mandatory front-face-down footprint is approximately 236.41 x 313.75 mm
in both states, so it is **not P2S-printable**. On a P2S it must instead be
printed as the mutually exclusive pair
`obiwan_optional_lm_keyed_1_of_2_bottom.stl` and
`obiwan_optional_lm_keyed_2_of_2_top.stl`; do not install either half
with the monolithic LM. The pair is cut from the finalized state-specific LM
at world **Y=172.481 mm** with an exact **zero-gap planar butt**, so both buried
route lumens cross the seam without being redrawn. The bottom owns two
symmetric Ø1.60 cylindrical pins at `x=±109.187`, `z=14.30`; each points world
+Y normal to the seam, has 0.50 mm root overlap, and engages the top by
2.40 mm (2.90 mm total male length). The top owns two 2.65 mm-deep blind
sockets with 0.12 mm radial and 0.25 mm end clearance: right is round Ø1.84,
while left is X-relieved to 1.96 × 1.84 mm. This round-plus-relieved constraint
tolerates ±0.30 mm relative pitch error across the 218.374 mm spacing instead
of binding like two tight round sockets. Two small exterior support lands
grow outward from the R113 lip, outside the LM recess. They retain at least
0.50 mm local radial and blind-end wall, 0.05 mm recess plan clearance, and
0.13 mm conservative W22-flange plan clearance. Their worst-case reach is
R114.4036: 1.4036 mm beyond the structural R113.0 ring and 0.6036 mm beyond
the finalized R113.8 visible fairing. Ac and Ae include a hidden 0.25 mm
clearance pocket around each land at the carrier interface, wholly between the
front and rear faces. CAD compatibility is gated; physical printed fit remains
coupon-qualified. With the monolithic LM these pockets are only small hidden
local reliefs; the three magnetic datums and primary wing retention are unchanged.
The pins create no extra screw or standalone retention/load credit.
Print both halves front-face-down. Assemble them front-face-down on one flat
datum, bring the top toward the bottom along world -Y so both pins enter
together without flexing, and confirm full seating, coplanarity, and
route-seam continuity. Then install the LM driver:
its flange and all normal LM fasteners are the installed structural splice
across the seam. Both keyed halves now print front-face-down with only
in-plane bed rotation. The former Z26°/Z45° and floor-bottom X=−90° footprint
figures are obsolete because those out-of-plane orientations cannot support
the captive-magnet pause. Revalidate the generated front-down footprint on the
selected printer. Each horizontal Ø1.60 pin is four nominal 0.4 mm nozzle
widths: release requires a process-matched coupon and sliced preview proving
both complete pin paths, both blind mouths, the exterior lands, and continuous
minimum-wall paths.
This option is still **PENDING** until two-pin/socket fit, full-seat and
coplanarity evidence, route-seam inspection, cable pull-through, and
driver-installed 1g/3g/5g proof are recorded; monolithic-LM evidence does not
qualify the split form.

The captive-magnet release audit does not create monolith G-code or a fake
monolith pause. Instead, every monolith station is source-contract matched to
the corresponding same-state keyed half, and coverage is accepted only after
that actual half passes the normal P2S cavity/toolpath gates. The pause
manifest distinguishes `not_p2s_printable__cavity_covered_by_exact_split`
coverage from actual keyed-half pause groups. Scaling, clipping, tilting, and
virtual-bed overrides are prohibited because all pieces must retain the same
front-face-down texture and insertion geometry.

The floor stand is not an add-on. In `build/floor_stand/`, both the canonical LM
carrier and the optional keyed bottom own the complete floor structure: a
full-depth W64 stem softly blended into the lower LM cap, a W64 × 18.3 mm
foot extending from `z=-150..18.3`, and one constant-thickness convex
Option-B wall transition over a 75 mm rear span and 65 mm rise. Its tangent
cubic has a 41 mm minimum centreline radius and no curvature reversal,
and the rear NL8 panel/service cavity. The floor plane is world `Y=0`, so the
LM-axis-to-floor distance is exactly **200.981 mm**. The rear panel is
`z=-150..-146`, 44 mm high, with an Ø31 NL8 cutout centered at
`(x=0,y=22)` and four Ø3.2 holes on a 29.2 mm square. The outer stem/foot
is solid except for the necessary connector service cavity and three buried
continuation lumens (LM Ø9, UM Ø8.2, shared T Ø6) through R14 turns.
There is no yoke, open rail, support fastener, or
`lx521_top_obiwan_addon_mount_floor_support` file. All six LM driver insert
bores are ordinary blind carrier bores in both states.

The floor and no-floor lower profiles intentionally diverge below the shared
upper shoulder. No-floor owns the complete shallow bridge down to world Y=0;
floor mode owns only the shoulder material above Y=60 plus its full-depth
Option-B wall. There is no shallow perimeter box and no lower magnet rail
under the bend. Ac and Ae share the exact local shoulder saddle, contain no
material below the Option-B vertical tangent at Y=74.15, and grow outward
through a G1 cubic to the LM-aperture lower tangent at Y=105.981. The floor's
deep load path and the no-floor bridge's four blind inserts remain
state-specific.

The optional V1 face-to-face tweeter crescent remains a separate add-on with
complete local-Ø9.8 blind-M3 receiver half-laps, 360° walls, 1.9 mm front
floors, and no printed T-cable arc.
Obi-Wan has no printed grommet; selected external cable retention remains a
physical-fit item, and cable load must never reach the MU tabs. No-floor
support is not an add-on: a 62 mm insert-bearing plate with soft cubic
shoulders is fused into the LM carrier around the unchanged holes at
(±20,20)/(±20,70). Three rear-facing mouths are packed inside the D20 support
opening centered at `(0,60)`: LM above, T lower-left, and UM lower-right. They
enter at z=5.3 and rise internally; the acoustic front stays solid. The plate occupies z=5.3..18.3, flush
with the acoustic front and no deeper than the six existing LM insert-pad rear
faces. It has no X-frame, acoustic-front window, rear rib, or other depth structure. Its four
6.8 mm-total bores open from the rear with a Ø6.5 × 2.0 entry followed by
Ø6.4; they retain the existing 6.2 mm solid front floor.
No bridge geometry extends behind the existing LM pad envelope. No-floor
geometry is otherwise unchanged. Select the tweeter module and an
independently qualified external retention method required by the
installation.

The conservative room-temperature PLA Tough+ screen assumes a 4.0 kg
installed mass, y=230 mm center of mass, and 70 mm rear offset. The 62 mm
insert core is reduced by the complete Ø9/Ø8.2/Ø6.0 entry lumens, with no
thin-skin credit, to a conservative **38.8 mm** design section. Exact 0.01 mm
sampled cuts through the soft outline retain at least 45.73 mm. At 13.0 mm
depth the credited section's in-plane/rear moduli are **3261.8/1092.9 mm³**.
Conservatively summing in-plane and rear bending gives approximately
**4.40/13.19/21.99 MPa** and safety factors **1.82/1.36/0.82** at sustained
1g/8 MPa, transient 3g/18 MPa, and transient 5g/18 MPa. Its 68° lower-ring fusion cradle physically follows the plate
to z=5.3, but the existing ring lip begins at z=6.8, so the load screen
credits only the actual 11.5 mm-deep monolithic interface. That interface
retains 118.5 mm effective width after deducting one Ø8.2 UM tunnel plus the
complete Ø6.0 tweeter tunnel. Its in-plane/rear section moduli are
**26908.4/2611.6 mm³**, with biaxial factors about **6.37/4.78/2.87**. The
combined normal/rear 5g insert reaction is 434.2 N, giving 1.38 pull-out
safety factor under the same assumed 600 N per insert; magnets contribute
exactly 0 N. These are design calculations, not certification. The two-ear
upper joints use the actual 0.43 kg MU + 0.20 kg tweeters plus carrier,
crescent, wire, and hardware allowance: 0.85 kg total over conservative
120 mm plan and 70 mm rear levers. Both receiver interfaces co-govern with
contact factors about 2.85/2.14/1.28 and M3 screw-tension factor about 1.28 at
5g. Those screens do not independently qualify either heat-set process,
receiver wall, or 1.9 mm front floor; the 5g pullout demand is approximately
393.9 N per insert. Magnets receive no credit in any case. The
finished print, inserts, screws, stock bridge, and installation substrate
remain
inside the physical proof-test boundary. The factors apply only near
room temperature; direct sun, a closed vehicle, or any service
approaching Bambu's published 61 °C heat-deflection temperature
invalidates them.

The integral floor stem has its own conservative closed-form
rectangle-minus-lumens screen. It is explicitly **not FEA, certification, or
physical release evidence**. The 4.0 kg load model uses `y=230 mm`, a 70 mm
rear eccentricity, and 1g/3g/5g load cases. The net W64 × 18.3 root deducts
the complete Ø9/Ø8.2/Ø6 lane sections; magnets and both optional concealed
split pins/sockets receive 0 N credit. Current project-allowable results are:

| Material | Vertical 1g/3g/5g SF | Anchored lateral 1g/3g/5g SF | 1g diagnostic deflection | Result |
|---|---:|---:|---:|---|
| Bambu PLA Tough+ | 3.05 / 1.97 / 1.18 | 4.22 / 2.73 / 1.64 | 1.18 mm | analytical pass |
| Bambu PLA Basic | 4.39 / 2.78 / 1.67 | 6.09 / 3.85 / 2.31 | 1.05 mm | analytical pass |
| Bambu PLA Lite | 2.69 / 1.73 / **1.04** | 3.73 / 2.40 / 1.44 | 1.40 mm | **FAIL at vertical 5g; provisional data** |
| Bambu PLA Matte | 2.78 / 1.79 / 1.08 | 3.85 / 2.49 / 1.49 | 1.49 mm | analytical pass |
| Bambu PLA Silk+ | 3.23 / 2.09 / 1.25 | 4.47 / 2.90 / 1.74 | 1.17 mm | analytical pass |

PLA Lite is provisional because no product-specific official TDS was
available; its comparison-sheet input is fail-closed. Every reported stress
includes an explicit **1.25 root geometry/model factor**. PLA Lite fails the
1.05 minimum at vertical 5g and is not an accepted material under this screen;
the other four pass only the analytical thresholds (2.0/1.5/1.05 and ≤2.0
mm), not the physical gate. The result is valid only with a **100% local-solid
modifier through the complete stem/root**; sparse infill receives no structural
credit. The W64 footprint also tips before material failure under lateral
acceleration: the calculated free-standing thresholds are only 0.139g
lateral, 0.348g rearward, and 0.384g forward. Therefore the installed speaker
**must use a positively attached anti-tip tether or anchor**; the rectangular
foot alone is not a safety restraint. Qualification remains pending a
2×-service-load 24 h proof at 35 °C with no cracking/whitening and residual
set ≤0.5 mm or ≤10% of loaded deflection, followed by at least 168 h at
1.5× service load for creep. Service above 35 °C, direct sun, radiator hot
soak, changed material/process, or the optional keyed LM form requires its own
recorded qualification.

The terminal service reference lies on the **283-degree axis**, exactly
midway between mounting screws 238 and 328 degrees; coupon 9 is the
physical clocking witness—there is no cosmetic collar engraving. Printed
UM conduit stops at the LM owner boundary; the Ø7.0 cable is free behind the
UM carrier, follows the modeled R15 approach to the 283° reference, and
continues through the R20 service turn to
a Y breakout comprising a 4 mm-long OD8 collar and two OD4 branch sleeves.
Two provisional Ø3.2/R8 slack leads enter separate provisional low-profile
flag Fastons. Service states pull one connector at a time through
0/3/6/9/12 mm while the other remains installed. The STEP fit model adds
closed Ø98/Ø80/Ø60 MU and conservative stepped W22 rear-body keep-outs.
The W22 source and transform are hash-pinned and recorded above.

`PHYSICAL_MEASURE_REQUIRED = True`, so terminal qualification remains
pending. The raw MU reference is an open acoustic surface and omits the
terminals; the datasheet also does not dimension them. The 12 mm maximum
pull exactly equals the provisional exposed-tab length and has zero
positive release overtravel margin. Measure and record carrier radius,
tab pitch/projection, 8.5/9.5 mm proxy body widths, flag orientation,
polarity, real withdrawal, cable/Y fit, and the selected external cable
retention before committing a full print. The proxy is a keep-clear aid,
not manufacturer
geometry or release proof.

See [`VARIANTS.md`](VARIANTS.md) for the variant/add-on catalog and the
compatibility matrix, [`obiwan_acoustic_wings_spec.md`](obiwan_acoustic_wings_spec.md)
for the Ac/Ae wing design authority, and [`PRINTING.md`](PRINTING.md) for
filament choice, print settings, fastener torques, and insert installation.

## Printable pieces

| STL in `build/<state>/stl/` | Footprint (mm) | Used by |
|---|---|---|
| `obiwan_core_1_of_2_lm_carrier.stl` | Structural Ø226 (R113.0) collar with a smooth exposed R113.8 side fairing, clipped only inside the LM--UM cusp to retain the 0.40 mm gap; six ordinary blind LM insert bores at 0/60/120/180/240/300°; two complete rear LM-to-UM ears with locally Ø9.8 cylindrical functional bosses and standalone Ø3.4 rear-driven screw-clearance passages at x=±32/y=315.770; two captive upper ring-magnet stations plus two captive lower shoulder stations, all hidden 0.15 mm beneath continuous surfaces. The right lower visible datum is `(x,y)=(45.285011,89.190370)` on shoulder parameter 0.5 with outward normal `(0.706451,-0.707762)`; the left is its exact mirror. All four LM magnets share source Z=15.10 with the UM pair. The LM also owns the buried UM/T route segments and continuous Ø9/R14 LM handoff. Floor state owns the full-height bent W64 stand and only the upper shallow shoulder; it has no lower box or magnet rails. No-floor owns the shallow four-insert bridge. | canonical large-format release form of the mandatory R6F LM carrier; use it on a verified larger bed **or** both optional keyed halves, never both forms. |
| `obiwan_optional_lm_keyed_1_of_2_bottom.stl` | front-face-down; in-plane bed rotation only; verified within 220 mm in both states | optional replacement print form for the canonical LM; in floor state it inherits the **entire** stem/foot/NL8 panel but remains the bed-checked alternative to the oversized monolith; requires the matching top half |
| `obiwan_optional_lm_keyed_2_of_2_top.stl` | front-face-down; in-plane bed rotation only; inherits both complete LM-to-UM ears, their local Ø9.8 cylindrical functional bosses, and their standalone Ø3.4 rear clearance passages | optional replacement print form for the canonical LM; requires the matching bottom half |
| `obiwan_core_2_of_2_um_carrier.stl` | Structural Ø103.4 (R51.7) collar with a smooth exposed R52.5 side fairing, clipped only inside the LM--UM and T--UM cusp/service regions while retaining the 0.40 mm LM--UM gap; two complete front LM-to-UM ears with standalone rear-opening blind Ø4.6 x 4.0 M3 heat-set receivers and 1.9 mm acoustic-front floors; two complete rear UM-to-tweeter ears with standalone Ø3.4 screw-clearance passages; locally Ø9.8 cylindrical functional bosses at both interfaces; two captive ring-magnet stations hidden 0.15 mm beneath the fairing; and the buried T continuation with fully solid-webbed 328°/58° insert bypasses. The UM cable is free behind this carrier and has no printed rear duct. | mandatory R6F UM core; install both LM-to-UM inserts in this individual print before assembly |
| `obiwan_addon_tweeter_crescent.stl` | cropped V1 crescent plus two complete front UM-to-tweeter ears with locally Ø9.8 functional bosses, standalone rear-opening blind Ø4.6 x 4.0 M3 heat-set receivers, complete 360° walls, and 1.9 mm acoustic-front floors; no printed T-cable arc or conduit | optional R6F face-to-face tweeter carrier; install both inserts in this individual print before assembly, then attach at x=±24, y=421.5 with the T cable free behind it |

Stable routing/fit review files in each state folder are
`obiwan_split.step` (mandatory two-carrier core),
`obiwan_lm_split.step` (the optional two-print LM form;
mutually exclusive with the monolithic LM carrier),
`obiwan_attachments.step` (optional tweeter add-on only),
`obiwan_assembled.step` (review assembly), and
`um_fit.step` (non-manufacturing Faston proxy,
standard/V1L/Obi-Wan UM Ø7 cable references, and the proud/V1L profile-fitted
split inserts). Obi-Wan has no printed grommet. The V1L grommet halves are also exported as the stable
STLs listed above.
The assembled R6F STEP also shows the independent LM Ø7.8 reference.

## Tweeter options

Obi-Wan carries the same default arrangement as Stock and Slim — the
face-to-face Dayton ND25FW-4 pair — but on its own optional add-on, the
tweeter crescent `obiwan_addon_tweeter_crescent.stl`, fastened to the UM
collar at `x=±24, y=421.5`. Leaving it off is a supported configuration.

The opposed TEBM35C10-4 BMR vase is **not** available for Obi-Wan: it is a
seam-B vase piece with the regular proud-family female dovetails, and Obi-Wan
has no seam B. It is released in Stock and Slim envelope profiles only.

## Cable routing (R6F)

- `baffle_cable_routing_obiwan.png` documents **R6F**, the surviving buried
  UM/T owner segments, the free rear UM and tweeter spans, the short
  un-ducted LM free span, solid-backed insert-bypass bumps, the physical
  T-over-UM crown crossing, state-specific support, and optional tweeter
  carrier. In addition to
  plan routing, it contains the true longitudinal side profiles and local
  nominal diametric u-z sections with authoritative vertical limits through
  representative UM and T bump/pilot axes.

R6F rotates only its six LM inserts to **0/60/120/180/240/300°** on the
unchanged Ø209.5 PCD, leaving the 90° crown clear. The physical UM and T
cable envelopes are Ø7.0/Ø5.2. The UM cable uses an Ø8.2 buried passage only
inside the LM carrier, then runs free behind the UM carrier. The T cable uses
an Ø6.0 buried passage through the LM and UM carriers, then runs free behind
the tweeter crescent; the crescent owns no printed cable arc. In no-floor
state all three mouths are packed inside the one D20 support opening centered
at `(0,60)`: LM Ø9 at `(0,64.76)`, T Ø6 at `(-4.75,55.91)`, and UM Ø8.2 at
`(3.17,55.91)`. Their radial D20 rim is at least 0.725 mm and every lumen pair
retains at least 0.800 mm of web. The LM branch follows a buried Ø9 path to
the common analytic R14 rear handoff; floor state reaches that same handoff
through its buried integral-stem lane. The UM route rises in the right LM arc
and exits the LM-owned
passage before continuing as free cable behind the UM carrier. The tweeter
route rises in the left LM arc, stays buried through the UM carrier, passes
the 328° and 58° pilots on shallow covered Z bumps, and exits before the
tweeter crescent. Both LM-owned ring lumens finish at R112.95; their 0.8 mm
covers finish at R113.75 beneath the continuous R113.8 carrier exterior. This
leaves a solid 0.85 mm outside skin and no radial groove. At the crown the
physical routes cross at **82.95°**: T
is the higher +Z cable and UM the lower cable, with **2.00 mm** between their
physical envelopes. This is not a two-printed-duct crossover and has no
separator-web claim.

The free UM cable follows the modeled R15 terminal approach to the immutable
283° service axis with a clockwise circumferential **193° tangent** at z=2.7,
then continues with exact G1 continuity through R20, clearing
the known Ø60 motor and terminal-carrier proxy before reaching the named
Y breakout. That breakout has a 4 mm-long OD8 collar with two OD4 branch
sleeves. Two provisional Ø3.2 conductors use R8-minimum slack paths into
separate provisional low-profile flag Fastons (8.5 mm receptacle / 9.5 mm
boot at 11 mm pitch). The review states move one connector at a time
through **0/3/6/9/12 mm** while the other remains installed.

Every surviving printed UM/T owner segment is continuously covered and has no
cable window. The
non-load-bearing wall is two complete 0.4 mm extrusion widths (**0.8 mm
minimum**); the seat roof is 0.85 mm to avoid a tangent BREP union. Insert
bypasses move only in Z and retain at least 0.4 mm to the complete
pad/bore envelope. Each of the eight named bypasses has a local full-width
solid saddle from the conduit roof to the applicable blind-bore floor; there
is no hollow trapped between the duct and bore, and the saddle never extends
behind the existing conduit bump. The LM-owned UM/T low runs and the
UM-owned T low run also have continuous full-width longitudinal webs from
the rear half of the conduit to the seat membrane. Those webs close both
shoulder cavities on either side of the 328°/58° UM bypasses while retaining
only the functional D6 lumen, blind insert bores, captive-magnet cavity voids and
half-lap mating clearances. In floor mode the 300/240/180° saddles retain the
same ordinary blind carrier insert floors as the other three LM axes; all
surrounding saddle volume is solid. The routing
PNG's nominal diametric u-z sections show the authoritative
vertical saddle limits without pretending to be exact octagonal BREP slices.
Obi-Wan deliberately exports no printed grommet or tunnel clip. Keep any
external cable retention outside the modeled buried-route, free-cable,
driver and Faston
service envelopes, and qualify it with the measured cable.

`PHYSICAL_MEASURE_REQUIRED = True`; qualification remains pending. The
MU reference omits both terminals and the datasheet leaves their carrier
and withdrawal geometry un-dimensioned. The maximum modeled pull is
12 mm, exactly the provisional exposed-tab length, so it has **zero
positive release overtravel margin**. The real MU, both Fastons and boots,
one-at-a-time withdrawal, cable, Y breakout, and selected external retention
must pass and record a physical dry fit before release. The completed record
must also bind each state to its exact candidate artifacts and document the
required 1g/3g/5g structural proof. Record all evidence and per-state signoff
in `obiwan_physical_qualification.md`; its current pending record and checksum
are bound into every R6F candidate manifest.

## Assembly

**R6F:** first prove the real MU terminal/Faston fit with coupon 9 and
the review STEP. If the optional LM print split is selected, use both halves
and omit the monolithic LM. With both front faces down on one flat datum, seat
the bottom half's two symmetric Ø1.60 +Y pins simultaneously in the top
half's right round and left X-relieved blind sockets by bringing the top along
world -Y without flexing or twisting. Verify full seating, coplanarity, the
closed route seam, and unobstructed UM/T cable pull-through. Hold that
registration while lifting the LM for driver fit-up. The pins/sockets have no
standalone retention or load credit; only the installed LM flange and its
normal fasteners splice the seam.
Install both LM-to-UM M3 inserts through the individual UM carrier's rear
receiver openings before assembly. On a flat front-face datum, engage the
rounded x=±32.0, y=315.770 half-lap ears while preserving their 0.20 mm axial
gap, then drive two M3 screws from the LM rear through its Ø3.4 clearance
passages into the UM's blind Ø4.6 x 4.0 receivers. Use no washer/nut and do not
drill through the 1.9 mm UM front floor. Verify the
165.100 mm axis spacing. Place the LM cable in its short free span, dry-fish
the UM and shared tweeter cables through their buried owner segments, and
rehearse the free UM span behind the UM carrier and free T span behind the
tweeter crescent. Confirm the physical T-over-UM crown crossing and covered
328°/58° T-route bumps are unobstructed. All six LM screws use the same
ordinary blind carrier inserts in floor and no-floor states. In floor mode,
verify that the integral W64 stem/foot, convex R41-minimum Option-B transition,
three buried continuations,
connector service cavity, and NL8 panel are unobstructed, then install a
positive anti-tip tether or anchor before loading the assembly. In no-floor
mode, bolt the stock bridge directly to the four rear-entry inserts in the
fused front-flush LM web. Fit only the selected add-ons;
fasten the crescent at x=±24, y=421.5 with rear-driven M3 screws into
its blind inserts. Obi-Wan has no TPU tunnel clip; keep the selected external
cable retention clear of the buried-route mouths, free cable, and service
envelope. Clock the physical terminals between screws 238/328 on the 283°
coupon-9 service axis. Confirm each measured flag Faston fits separately,
its lead follows the polarity-specific slack path, and each one-at-a-time
0/3/6/9/12 mm review state clears the installed opposite connector and
both drivers before final driver installation. This is still not release
proof: the 12 mm state has zero positive overtravel beyond the provisional
12 mm exposed tab. The numbered procedure and hardware
cautions are in PRINTING.md.

**Ac/Ae wing segments:** slide each through-local-thickness dovetail along
local Z while holding the acoustic front faces on a common datum. The two keys
provide XY registration and in-plane interlock only; they do not independently
retain the segments against Z separation and carry no structural-retention
claim. If handling or the experiment requires Z retention, use the same
documented rear tape or light-bond method on every compared wing. This Ac/Ae
contract supersedes their former wavy butt-glue/epoxy seams only; it does not
change the adhesive instructions for legacy R6P attachments or other splits.
