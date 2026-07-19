# Variant catalog

There are now two intentionally isolated systems:

- The **R6P proud family** prints as four pieces joined at three seams:
  `piece_bottom`, `piece_mid_left`, `piece_mid_right`, and `piece_top`.
  All five variants retain the same seams (A: y=120, teeth ±66/±103;
  B: y=315.95, teeth −19/+30; C: x=−5.6, tooth y=305) and matching
  seam mouths. B2/C7/V0/V1 use the standard Ø8.2/G1/R14 UM outlet at
  (33.446, 301.492). V1L is a keyed routing exception: its Ø8.2
  alternate tail stays wholly in `piece_mid_right` and exits the
  z=6.8 rear face at Q=(13.497063, 307.618796), radius 60.0 mm on the
  283-degree terminal axis. Because it never reaches seam B, top/vase
  interchangeability is preserved.
- **R6F Obi-Wan** is an extreme skeletal system: only one LM carrier and
  one UM collar are mandatory. It has no proud-family seams or full
  outline. UM is buried only through the LM
  carrier and is free behind UM; T is buried through LM/UM and is free behind
  the tweeter crescent. The surviving printed spans retain 0.8 mm minimum
  walls, a 0.85 mm seat roof, and covered Z bumps backed by solid roof-to-bore
  saddles. The short LM lead is a free D7.8 span in a minimum-radius 3.96 mm
  rear-open subtractive clearance, with no printed micro-duct.
  No-floor mode fuses the stock four-hole solid front
  web into the LM core and has no separate keel.
  Floor mode instead fuses the full-height W64 stem/foot, R12 root, buried
  floor lanes and NL8 panel directly into the LM carrier. There is no separate
  floor-support part or support fastener. Tweeter, outline, and retention
  remain selectable add-ons. The canonical floor LM is a large-format
  monolith; an optional two-print keyed split may replace it, but is never
  added to it, and its bottom half inherits the complete stand.

Both systems are generated in `floor_stand/` and `no_floor_stand/`.
Their review sheets are `baffle_cable_routing_proud.png` (normal R6P
route plus its labeled V1L-only alternate tail) and
`baffle_cable_routing_obiwan.png`; there is no generic shared sheet.

Drivers (LX521.4 production): LM = SEAS **U22REX/P-SL** (H1659-08,
flange O220.6 x 6.0 measured); UM = SEAS **MU10RB-SL** (H1658-04,
flange O98 x 4.0 measured). Older comments naming the LX521 prototype
drivers (W22EX001 / 10F) refer to the same cutout/pilot geometry.

## Base variants

| Variant | Replaces | Geometry | STLs |
|---|---|---|---|
| **B2** | (baseline, all 4) | Full 18.3 everywhere. Constant-wall mini-vase (walls tangent to r=50.83 about the UM). | `lx521_top_base_1..4of4` |
| **C7** | bottom + mids (+B2 vase) | LM knife taper: REAR-side smoothstep 18.3 -> 0.5 over 19 mm from the flank/chamfer edges; recovery lands at both seams; full bottom strip. Front plane intact. | `lx521_top_c7base_1..4of4` |
| **V0** | vase | Rear knife band: REAR-side 18.3 -> 0.5 over the last 2.8 mm of the vase outline (same sculpted side as C7); front intact. | `lx521_top_v0_4of4_vase` |
| **V1** | vase | Thin FLUSH vase: 11.5 (material z 6.8..18.3). Crescent re-derived (4.0 clamp seat at stock z); tweeter septum 11.5 (shorter standoffs, pair spacing -6.8); one shared front plane. | `lx521_top_v1_4of4_vase` |
| **V1L** | bottom + mids | Thin FLUSH LM section: 11.5 (z 6.8..18.3 -- SAME plane as the V1 vase: no seam-B step), including both 6-mm seam-B male teeth that project into the vase. Smoothstep ramp y=78..96 to the full strip. Ø8.2 LM duct is the 11.5 binder. Its keyed Ø8.2 UM alternate exits `mid_right` at Q=(13.497063, 307.618796, 6.8) on the 283° axis; seam B/top are untouched. | `lx521_top_v1l_1..3of4` (its `--variant v1l` export bundles the unchanged V1 vase = the complete ~12 mm baffle) |
| **Obi-Wan R6F** | legacy four-piece baffle; replaced by two mandatory carriers plus add-ons | **Extreme barebone flush carriers**: LM Ø190 opening / Ø221.2 seat / nominal R113.0 lip; UM Ø82 opening / Ø98.6 seat / R51.7 outside. Rounded M3 half-lap pairs sit at x=±32.0/y=315.770 and x=±24/y=421.5. At both interfaces the closure-web/base teardrops remain nominal Ø9, while every complete Z-owned cylindrical functional boss is locally Ø9.8. LM and UM respectively own complete standalone rear Ø3.4 passages; UM and the crescent respectively own complete standalone rear-opening blind Ø4.6 × 4.0 receivers with 360° walls and 1.9 mm front floors. Each joint retains a 0.20 mm axial gap. Install inserts in the individual UM and crescent prints before assembly; neither interface uses a washer, nut, front bolt head, or cross-owner receiver wall. Complementary tangent-blended LM–UM and T–UM closure webs are solid through z=6.8..18.3 and share the coplanar z=18.3 front; only the central T cable mouth remains open between the upper rings. Obi-Wan-only LM axes rotate to 0/60/120/180/240/300° on the unchanged Ø209.5 PCD; all six are ordinary blind carrier insert bores in both states. Six actual Ø5×2 magnets use captive Ø5.20×2.10 cavities with 0.45 mm axial skins and a 45° closing roof: preserve the upper LM ring-radial axes at 64°/116°; place the lower LM pair in the straight base side faces at `(x,y,z)=(±32,18,12.55)` with outward ±X normals; and keep the UM pair ring-radial at 50.5°/129.5°, z=15.1. The shared floor/no-floor lower profile makes the two base-side datums identical in both states. The Obi-Wan ring-radial stations use a local +0.60 mm outward backing boss because moving the captive envelope inward would collide with the flange seat. Magnets are inserted at the authoritative pause and permanently buried, with no glue or external opening. Neither carrier has proud magnet ears. Ac/Ae provide three matching captive receivers per physical side: LM lower, LM upper, and UM. The preserved 0.05 mm mating gap plus two skins gives 0.95 mm nominal magnet separation. UM is buried in an Ø8.2 passage only inside LM, then runs free behind UM with no printed UM-carrier rear duct. T is buried in an Ø6.0 passage through LM/UM, then runs free behind the tweeter crescent, which has no printed cable arc. Every surviving named insert bypass has a deep full-width burial web; the D7.8 LM lead is a free span in a minimum-radius 3.96 mm rear-open subtractive clearance without a printed micro-duct. The physical T/UM routes cross at 82.67° with no two-duct separator-web claim. No-floor LM includes the unchanged front-flush bridge plate at z=5.3..18.3. Floor LM instead owns the complete W64 × 18.3 stem/foot from z=−150..18.3, R12 root, three buried floor continuations, connector service cavity and rear NL8 panel; floor Y=0 keeps the LM axis exactly 200.981 mm above the floor. The tweeter carrier is a separate add-on; Obi-Wan has no printed grommet. | `lx521_top_obiwan_core_1of2_lm_carrier.stl`, `lx521_top_obiwan_core_2of2_um_carrier.stl` |

The Obi-Wan LM print form is a separate choice inside the same R6F variant. The
canonical `lx521_top_obiwan_core_1of2_lm_carrier.stl` is one solid. On a 220 mm
square bed, replace it with **both**
`lx521_top_obiwan_optional_lm_keyed_1of2_bottom.stl` and
`lx521_top_obiwan_optional_lm_keyed_2of2_top.stl`; never combine either half with
the monolithic carrier. Their state-specific final geometry is cut at world
Y=172.481 mm with a zero-gap planar butt. Both halves in both stand states
print front-face-down with only in-plane bed rotation. The former no-floor
Z26°/Z45° and floor-bottom X=−90° footprints are obsolete because those
out-of-plane orientations do not permit the captive-magnet pause. Revalidate
each actual front-down footprint against the selected printer. The bottom owns
two symmetric Ø1.60 cylindrical pins at `x=±109.187`, `z=14.30`;
both point world +Y normal to the seam, overlap the root by 0.50 mm, and engage
the top by 2.40 mm (2.90 mm total length). The top's 2.65 mm-deep blind sockets
retain 0.12 mm radial and 0.25 mm end clearance: right is round Ø1.84, while
left is X-relieved to 1.96 × 1.84 mm. The round-plus-relieved pair accepts
±0.30 mm relative pitch error across 218.374 mm. Small exterior support lands
outside the LM recess retain ≥0.50 mm radial/end wall, ≥0.05 mm recess plan
clearance, and ≥0.13 mm clearance from the conservative W22 flange proxy.
They add about 1.40 mm local perimeter growth and register the loose halves,
but add no extra screw and have no standalone retention/load credit; the installed LM driver flange
and all normal LM fasteners provide the structural splice. This optional form
is geometrically compatible with Ac/Ae through matching 0.25 mm clearance
pockets cut only into the hidden carrier interface between the wing faces;
physical printed fit remains coupon-qualified. It otherwise
remains pending slicer proof of both four-nozzle-width horizontal pins, both
lands and all minimum walls, process-matched two-pin/socket fit, actual U22
fit, full-seat, coplanarity, route-seam, cable-pull-through, and driver-installed
1g/3g/5g proof.

With the monolithic LM, the same Ac/Ae pockets remain as small hidden local
reliefs. They do not move the three magnetic datums or change the primary
magnetic retention contract.

**V1 vs V1L:** V1 = thin TOP piece; V1L = thin BOTTOM+MIDS. Pair them
for the full front-flush thin baffle; either also works alone on the
other family's pieces (see matrix). The V1L outlet change is confined to
its keyed `mid_right`; it does not create a V1L-specific top.

## Add-ons (outline experiments)

| Family | Pieces | Fits | Anchoring |
|---|---|---|---|
| **A-comp shoulders** (18.3) | 4: `addonA_1..4of4` | B2 vase only | 2 captive magnets/side: flare wall zc=5.0 + crescent arc zc=10.7; outline kinks register |
| **B1 wings** (18.3) | 2: `addonB1_1..2of2` | B2 vase only | same two sites |
| **V1 A-shoulders** (11.5) | 4: `v1addonA_*` | V1 vase (V1L sets) | two captive stations/side: lower zc=12.5, upper zc=14.4; common curved-interface tangent land is <=0.134666 mm on the base and receiver-relieved to the real 0.05 mm gap |
| **V1 B1-wings** (11.5) | 2: `v1addonB1_*` | V1 vase (V1L sets) | same captive stations and microscopic qualified tangent-land adaptation |
| V0 scarf family | (concept only; no released print) | V0 | no released mate or pairing polarity; V0's orphan rear-axis base cavities alone are migrated and print front-face-down. Detached `(±46,324)` was first moved to connected `(±37.697,326.470)`, but the mirrored-left site failed T-route clearance (2.605 < 8.000 mm). Final release sites are right `(37.697,326.470)` and left `(-7.250,321.200)`; the latter clears the D82 cutout, all UM pilots, grown seam-B keepout, and all ducts. |
| **R6F tweeter crescent** | `lx521_top_obiwan_addon_tweeter_crescent.stl` | Obi-Wan UM collar only | direct half-laps at x=±24, y=421.5; UM owns complete rear Ø3.4 ears and the crescent owns complete front local-Ø9.8 ears with standalone blind Ø4.6 x 4.0 insert receivers, 360° walls, 1.9 mm front floors, and a 0.20 mm axial gap; no printed T-cable arc or conduit, so T remains free behind the crescent |
| **R6P split grommet** | `lx521_top_proud_addon_um_grommet_half_{a,b}.stl` | standard B2/C7/V0/V1 R14 outlet; **not V1L** | TPU; short curved D8 shank follows the final bore and seats at rear z=0 |
| **V1L split grommet** | `lx521_top_v1l_addon_um_grommet_half_{a,b}.stl` | keyed V1L 283° outlet only | TPU; Ø8 curved body / Ø7.1 bore / 2.5 mm insertion / Ø13 × 2 flange seated at rear z=6.8; no fastener |

B2 addons on V0/V1: NO (knife/thin walls — no receiver seats).
V1 addons on B2/V0: NO (the zc=12.5/14.4 captive stations exist only on the V1 vase).
R6F add-ons fit only Obi-Wan interfaces and never fit R6P parts. The floor stem,
foot and NL8 panel are state-owned LM geometry, not an add-on.
No acoustic perimeter skin is emitted in R6F: that absence is the
barebone experiment. Any later skin/wing or cable retainer must use the
defined alignment or threaded interfaces as an optional module, must
not obstruct any buried tunnel or covered Z bump, and may
not add material back to the mandatory collars. Magnets may register a
module but may never be counted in its structural load path.

All released magnet-bearing variants now share the coupon-proven captive
system: D5.0 × 2.0 magnet, Ø5.20 × 2.10 internal cavity, 0.45 mm axial skin on
both faces, vertically open loading cradle, and support-free 45° roof. Every
part prints front-face-down. Exact pause heights, grouped sites, local-axis
polarity, and counts live in
[`CAPTIVE_MAGNET_PAUSE_MANIFEST.md`](review/captive_magnet_slice_audit/CAPTIVE_MAGNET_PAUSE_MANIFEST.md); STL files cannot carry
pause markers. Concept-only drawings, diagnostic renders, historical fit
coupons, and the unreleased V0 scarf mate are not release outputs and are not
converted as production parts. The `coupons/obiwan_ae_embed/` coupon remains the
reference implementation rather than an installed assembly component.

## Compatibility matrix (bottom+mids x vase)

Any of {B2, C7, V1L} bottom+mids joins any of {B2, V0, V1} vase — the
front plane is always continuous at z=18.3, and the seams, keys, and
duct mouths still match. V1L's alternate UM tail is self-contained in
`mid_right`, so it does not alter this matrix or the top. Rear-side
steps at seam B land on the hidden side:

| bottom+mids \ vase | B2 (18.3) | V0 (knife band) | V1 (11.5) |
|---|---|---|---|
| **B2** (18.3) | stock reference | edge experiment | vase-thickness experiment (rear step 6.8) |
| **C7** (LM knife) | LM-edge experiment | full knife-edge baffle | knife LM + thin vase |
| **V1L** (11.5) | thin LM only (vase protrudes 6.8 rearward) | thin LM + knife vase | **complete UNIFORM 11.5 front-flush baffle** |

Notes: mixed-thickness key joints mate on the thinner piece's depth
(the through-pockets leave a shallow open notch on the hidden rear —
cosmetic). Tweeter through-bolt length follows the vase septum (18.3:
~M4x35; 11.5: ~M4x30).

**R6F Obi-Wan is outside this matrix.** Its two collars, rear-driven
insert-fastened M3 half-laps, and integral buried routes replace the
proud-family shell, seams, and
subtractive cable network. Do not combine an R6F carrier/add-on with an
R6P base, or generate a Obi-Wan STEP under `LX_ROUTING_PROFILE=proud`.
R6P pieces remain mutually interchangeable exactly as shown above;
choosing the V1L `mid_right` selects its 283-degree UM exit, while a
B2/C7 `mid_right` retains the standard R14 exit.

The V1L physical rear-face aperture is centered at
**Q=(13.497063, 307.618796, 6.8)**, exactly 60.0 mm from the MU axis on
the 283-degree line. Its nominal outside cutter continuation ends at
(11.080158, 308.797599, −2.0), 2.689 mm farther in XY, and must not be
mistaken for the aperture center. The MU reference mesh omits the real
tabs. Dry-fit the physical driver, Fastons, boots, measured withdrawal
stroke, cable, and dedicated V1L split grommet in the printed V1L
`mid_right`; the standard R14 coupon and proud curved grommet do not
validate this keyed outlet. The V1L grommet solid clears the conservative
Faston motion box, while the installed cable intentionally enters that
box to reach the terminals; both facts still require physical hardware
confirmation.

The Obi-Wan LM/UM/T physical cables are Ø7.8/Ø7.0/Ø5.2. UM uses a printed
Ø8.2 passage only through LM; T uses a printed Ø6.0 passage through LM and
UM. The LM cable floats over a short 20.15 mm radial
span at 269.5° behind the carrier with no printed micro-duct or cover; a
minimum-radius 3.96 mm rear-open relief is subtracted from the LM carrier
without moving the cable centerline;
No-floor UM/T cables enter through two centered rear bridge-plate mouths at
`(+5,82,5.3)` and `(-5,82,5.3)` and rise internally before the LM ring;
floor mode continues LM/UM/T through the three buried integral-stem lanes.
The UM cable exits its
LM-owned passage and remains free behind the UM carrier; there is no printed
UM-carrier rear duct or D82 mouth. Its free centerline follows the modeled R15
terminal approach to the 283° service axis and then the R20 service turn. The
T path stays buried through the UM carrier, crosses the free UM cable at
**82.67°** with T above UM, then exits and remains free behind the tweeter
crescent. The crescent owns no printed cable arc, conduit, socket, or horn.
The gap between the physical cable envelopes at the crossing is 1.85 mm;
there is no longer a two-printed-duct separator web. All eight named insert
bypasses remain covered Z excursions on the surviving printed spans,
and each has a full-width solid saddle from its conduit roof to the applicable
blind-bore floor with no added rear depth. Continuous full-width longitudinal
webs back the LM-owned low runs and the UM-owned T low run to their seat
membranes, eliminating both shoulder cavities beside the 328°/58° UM
bypasses outside the exact D6 lumen, blind-bore, captive-magnet and half-lap
interface voids. Every surviving
buried span retains a 0.8 mm minimum wall and 0.85 mm seat roof. There are no
cable windows in those printed spans; the specified UM and T free spans are
intentionally visible from the rear. The state-specific
`baffle_cable_routing_obiwan.png` is the routing
reference and includes true longitudinal side profiles plus nominal diametric
u-z bump/pilot sections with authoritative vertical limits.
The fit STEP continues the physical Ø7.0 free UM cable behind the UM carrier
to the 283° terminal reference and through the R20 service turn to a Y breakout.
The breakout uses a 4 mm-long OD8 collar and two OD4 branch sleeves. Two
provisional Ø3.2 conductors then follow R8-minimum slack paths into
separate low-profile flag boots. One connector at a time is checked at
0/3/6/9/12 mm pull while the other remains installed.

The Obi-Wan free UM service path reaches the **283-degree axis**, the exact
midpoint between mounting screws 238 and 328 degrees. Use coupon 9 as
the physical witness and clock the MU terminals to that axis.
`top_baffle_nd25fw4_um_fit.step` shows the V1L legacy withdrawal volume and,
for Obi-Wan, the installed non-overlapping low-profile 6.3 mm flag-Faston
proxies, two provisional Ø3.2/R8 slack leads, and two independent 12 mm
pull-sweep envelopes. Source/test service compositions evaluate each terminal
at 0/3/6/9/12 mm while the opposite side remains installed. Closed
Ø98/Ø80/Ø60 MU and conservative stepped W22 rear-body keep-outs screen the
service harness. The W22 reference is
the hash-pinned manufacturer shrinkwrap `E0022_W22EX001.stp`, SHA-256
`7fc2be551c86006e11c32a570b046772987cb86dcf65350f77c6e34709aa5ab6`.
Its declared transform rotates +90° about X (native +Y to world +Z,
native +Z to world -Y) and translates by `(0, 200.981, -47.498931)`.
Cached native bounds
`(-110.5,-37,-110.5)..(110.5,65.798931,110.5)` map to world
`(-110.5,90.481,-84.498931)..(110.5,311.481,18.3)`, placing native max-Y
at front datum z=18.3. The guarded W22 geometry phase imports the pinned STEP,
verifies that transform/bounds, and proves exact containment by the stepped
service proxy. It does not qualify the installed custom U22, terminals or
leads.

`PHYSICAL_MEASURE_REQUIRED = True`, and terminal qualification is
pending. The raw MU mesh omits the tabs and is an open acoustic surface,
while the datasheet does not dimension their carrier. The 12 mm modeled
pull equals the provisional 12 mm exposed tab length and therefore has
zero positive release overtravel margin. Physical checks must record tab
pitch/radius/projection, boot widths, flag orientation, polarity, actual
withdrawal, the OD8/OD4 Y breakout, and the selected external cable
retention before release. Each stand state also requires its own exact candidate
identity, coupon evidence, structural proof and signed authorization in
`obiwan_physical_qualification.md`.

## Hardware

* W22: 6x M5 x 5.8 x O6.3 heat-sets (bore O6.4 x 6.8).
* 10F/MU10: 4x M3 x 3 x O5 heat-sets (bore O4.6 x 4.0), pattern clocked
  to 58/148/238/328 degrees. For Obi-Wan, select the equivalent driver
  rotation that puts the terminal carrier at the 283-degree witness.
* R6F LM-to-UM collar structural joint: 2x rear-driven M3 screws pass through
  the LM's standalone Ø3.4 clearance bores into M3 x 3 heat-set inserts in the
  UM's standalone rear-opening blind Ø4.6 x 4.0 receivers. Each receiver
  sits inside a complete local Ø9.8 cylindrical functional boss; the
  closure-web/base teardrop remains nominal Ø9. Each receiver retains a 1.9 mm
  acoustic-front floor; the ear halves retain a 0.20 mm axial gap. Install both
  inserts in the individual UM print before assembly and
  choose screw length for full engagement without bottoming. No washer, nut,
  or front bolt head belongs to this interface.
* R6F optional LM split: assemble both front faces down on one flat datum and
  move the top straight along world -Y so both bottom-owned Ø1.60 +Y pins seat
  together in the top's right round and left X-relieved blind sockets. Do not
  flex, twist, or use one pin as a hinge. The pair registers alignment only;
  assign it no standalone retention/load credit. The LM flange and all normal
  LM fasteners provide the installed splice across the Y=172.481 mm seam.
* R6F LM: all six 0/60/120/180/240/300° driver sites use ordinary blind
  carrier heat-sets in both states. Floor mode has no support screws or
  secondary inserts: its stand is monolithic with the LM. No-floor additionally
  uses the four immutable bridge inserts at
  (±20,20)/(±20,70) in the monolithic LM solid web. The Ø6.4 × 6.8
  bores open at z=5.3 and retain a 6.2 mm solid front floor.
* R6F tweeter crescent: 2x rear-driven M3 screws at x=±24, y=421.5 into
  standalone rear-opening blind Ø4.6 × 4.0 insert receivers in complete local
  Ø9.8 crescent half-laps. Install both inserts in the individual crescent
  before assembly; require complete 360° walls and 1.9 mm front floors. The
  UM owns the complete opposing Ø3.4 passages, the 0.20 mm axial gap remains
  open, and the acoustic front remains uninterrupted.
* R6F alignment: 6x D5 × 2 N52 magnets in the core's surface-normal captive
  Ø5.20 × 2.10 cavities (four LM + two UM; three total per physical side).
  Preserve the upper LM ring axes at 64°/116° with the validated insert/route
  clearances. The lower LM sites remain in the shared straight base
  faces at `(x,y,z)=(±32,18,12.55)`, with left/right outward normals
  `(-1,0)`/`(1,0)` and verified buried-route and bridge/integral-stand
  clearance. UM keeps its 50.5°/129.5° axes at z=15.1. Ring-radial stations
  use the local +0.60 mm outward backing boss required to avoid the flange
  seat. Every magnet is inserted at its manifest pause and buried between two
  0.45 mm skins under a 45° roof; there is no glue, access hole, or proud ear.
  Keep a marked polarity standard and use the manifest's local-axis table for
  mirrored parts. Magnets are alignment/anti-rattle
  devices only and receive **zero structural load credit**. Ac/Ae use all
  three sites on each side through matching LM-lower, LM-upper, and UM
  receivers. The 0.05 mm mating gap and two skins give 0.95 mm nominal
  magnet-face separation.
* Tweeter pair (when its carrier is fitted): M4 through-bolts + nyloc,
  clamping the crescent.
* R6P magnets: D5 x 2 N52 discs in the same Ø5.20 × 2.10 captive base
  and receiver cavities. Print front-face-down, insert them only at the
  manifest pauses, and verify seating and polarity before the roof is closed.

## R6F structural screen and release gate

The fused four-hole bridge interface has a 62 mm insert core and soft cubic
shoulders occupying z=5.3..18.3, exactly the deepest existing LM-pad envelope.
Two centered rear cable lumens enter at x=±8/y=82 while the acoustic front
remains solid. It has no X-frame or additional rear-depth structure and is screened
conservatively for a 4.0 kg installed assembly, y=230 mm center of mass and
70 mm rear offset. The conservative member width is **47.8 mm**: the 62 mm
core minus the complete Ø8.2 and Ø6.0 entry lumens, with no credit for their
thin skins. Exact sampled soft-outline sections retain at least 53.5 mm. The
47.8 × 13.0 mm design section has in-plane/rear moduli of
**4950.5/1346.4 mm³**. Its summed biaxial stresses are about
**3.28/9.84/16.41 MPa**, giving safety factors about **2.44/1.83/1.10** at
sustained 1g/8 MPa, transient 3g/18 MPa, and transient 5g/18 MPa. The physical 68° fusion cradle
extends to z=5.3, but the existing ring lip begins at z=6.8, so only its actual
11.5 mm-deep monolithic interface is credited. It retains an effective width
of **118.5 mm** after deducting one Ø8.2 UM tunnel plus the complete Ø6.0
tweeter tunnel. Its in-plane/rear section moduli are **26908.4/2611.6 mm³**,
and its biaxial factors are about **6.37/4.78/2.87**. The combined-axis worst
insert reaction is 434.2 N,
so the four inserts retain **1.38** pull-out safety factor at 5g under the
assumed 600 N capacity. Magnets contribute 0 N.

Floor mode instead has a closed-form net-section screen of the integral W64
× 18.3 stem root after deducting the complete Ø9 LM, Ø8.2 UM and Ø6 shared-T
lumens. The 4.0 kg model uses y=230 mm, 70 mm rear eccentricity, and 1g/3g/5g
vertical proof loads. Project-allowable results are:

| Material | Vertical 1g/3g/5g SF | Anchored lateral 1g/3g/5g SF | 1g diagnostic deflection | Result |
|---|---:|---:|---:|---|
| Bambu PLA Tough+ | 3.05 / 1.97 / 1.18 | 4.22 / 2.73 / 1.64 | 1.18 mm | analytical pass |
| Bambu PLA Basic | 4.39 / 2.78 / 1.67 | 6.09 / 3.85 / 2.31 | 1.05 mm | analytical pass |
| Bambu PLA Lite | 2.69 / 1.73 / **1.04** | 3.73 / 2.40 / 1.44 | 1.40 mm | **FAIL at vertical 5g; provisional data** |
| Bambu PLA Matte | 2.78 / 1.79 / 1.08 | 3.85 / 2.49 / 1.49 | 1.49 mm | analytical pass |
| Bambu PLA Silk+ | 3.23 / 2.09 / 1.25 | 4.47 / 2.90 / 1.74 | 1.17 mm | analytical pass |

This is a conservative analytical screen, **not FEA, certification, or
release evidence**. Bambu coupon values are not stand allowables; the code
applies project derating for weak direction, creep, print process, and
environment plus an explicit **1.25 root geometry/model factor**. PLA Lite is
provisional pending a product-specific official TDS and fails the 1.05
vertical-5g threshold; it is not accepted by this screen. The other four meet
the 2.0/1.5/1.05 1g/3g/5g and ≤2.0 mm 1g thresholds only when the complete
stem/root has a **100% local-solid modifier**; sparse infill receives no
structural credit. Magnets and both concealed split pins/sockets contribute
0 N.

The upper joint case uses the actual 0.43 kg MU + 0.20 kg tweeters plus printed and
hardware allowance, **0.85 kg total**, over conservative 120 mm plan and
70 mm rear levers. Both receiver interfaces co-govern with contact factors
about **2.85/2.14/1.28**, with M3 screw-tension factor about **1.28** at 5g.
Those screens do not qualify either heat-set installation, complete receiver
wall, or 1.9 mm front floor; the 5g pullout demand is approximately 393.9 N
per insert. Magnets contribute 0 N throughout.

These numbers are analytical screening, not release evidence. Every
finished web/core/insert/fastener combination must pass a documented
physical proof test at the distributed 4 kg sustained-1g, transient-3g
and transient-5g bridge/integral-stand design loads, while the upper joints see
their actual 0.85 kg distributed mass and lever arms. Record the
fixture, duration, temperature, permanent deflection, insert movement,
cracking, and post-test torque. Ramp over at least 10 s and hold 1g for
24 h, 3g for 60 s and 5g for 10 s. The calculations cover the modeled
insert groups, minimum printed sections, ear net/neck/bearing and M3
shear; they do not independently qualify the NL8 panel, stock bridge,
real insert process, LM-to-UM receiver wall/front floor, or installation
substrate. The integral floor state must
additionally pass a 2×-service-load 24 h proof at 35 °C with no cracking or
whitening and unloaded residual set ≤0.5 mm or ≤10% of loaded deflection,
followed by a 1.5×-service-load creep hold for at least 168 h. Bambu's
published 61 °C heat-deflection temperature is not an assembly service
rating; direct sun, a closed vehicle, or another hot environment
requires a suitable material and a proof test at the actual maximum
service temperature.

Strength does not establish free-standing stability. The W64 foot's screened
tip thresholds are only 0.139g lateral, 0.348g rearward, and 0.384g forward.
Every floor installation therefore requires a positively attached anti-tip
tether or anchor; neither the foot nor magnets may be treated as a safety
restraint.

The monolithic-LM screen does not qualify the optional keyed split by
inheritance. Its two concealed pins/sockets remain at zero standalone retention
and structural credit; separately record slicer-path proof, simultaneous
two-pin/socket fit, full seating, coplanarity, route-seam continuity, cable pull-through, and
driver-flange-installed 1g/3g/5g proof for each selected floor/no-floor split
candidate.
Record the complete floor/no-floor candidate identity, print and insert
process, load fixture/history, measurements, evidence hashes and independent
release signoff in `obiwan_physical_qualification.md`. Evidence from one state
does not authorize the other.

## Stable routing review and fit artifacts

Each stand-state folder contains:

- `top_baffle_nd25fw4_obiwan_split.step` — mandatory two-collar core;
- `top_baffle_nd25fw4_obiwan_lm_split.step` — optional two-print LM form,
  mutually exclusive with the monolithic LM carrier;
- `stl/lx521_top_obiwan_optional_lm_keyed_{1of2_bottom,2of2_top}.stl` —
  the two parts required when that optional LM form is selected;
- `top_baffle_nd25fw4_obiwan_attachments.step` — optional tweeter attachment;
  the floor structure is already part of the floor-state LM carrier;
- `top_baffle_nd25fw4_obiwan_assembled.step` — core, add-ons, and fit
  proxy together for collision review;
- `top_baffle_nd25fw4_um_fit.step` — terminal/Faston proxy, standard,
  V1L, and Obi-Wan D7 cable envelopes, plus the proud/V1L split strain-relief
  profiles (not manufacturer hardware geometry; Obi-Wan has no printed grommet);
- `stl/lx521_top_v1l_addon_um_grommet_half_{a,b}.stl` — printable
  keyed V1L TPU strain relief;
- `baffle_cable_routing_proud.png` and
  `baffle_cable_routing_obiwan.png` — the two isolated route sheets; the
  proud sheet includes both the normal R6P UM tail and the labeled
  V1L-only 283-degree alternate;
- `lx521_coupon_9_um_faston_clocking.stl` — the physical MU clocking
  gate; the complete current coupon list is in PRINTING.md; and
- `obiwan_release_manifest.json` — hashes the state candidate and its pending
  qualification record; it is provenance, not physical-release authority.

See PRINTING.md for print settings and torques; `make check` guards
clearances, seam keys, complete-route smoothness, eroded-outline
containment, service envelopes, state manifests, and cutter health.
