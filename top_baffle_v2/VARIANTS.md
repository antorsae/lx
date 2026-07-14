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
- **R6F V1LF** is an extreme skeletal system: only one LM collar and
  one UM collar are mandatory. It has no proud-family seams, full
  outline, stand, or tweeter carrier. UM is buried only through the LM
  carrier and is free behind UM; T is buried through LM/UM and is free behind
  the tweeter crescent. The surviving printed spans retain 0.8 mm minimum
  walls, a 0.85 mm seat roof, and covered Z bumps backed by solid roof-to-bore
  saddles. The short LM lead is a free D7.8 span with no printed micro-duct.
  No-floor mode fuses the stock four-hole solid front
  web into the LM core and has no separate keel.
  Floor mode has no bridge tail and requires its separate bolted floor
  support before carrying drivers; tweeter, outline, and retention remain
  selectable add-ons. The canonical LM is monolithic; an optional two-print
  keyed split may replace it, but is never added to it.

Both systems are generated in `floor_stand/` and `no_floor_stand/`.
Their review sheets are `baffle_cable_routing_proud.png` (normal R6P
route plus its labeled V1L-only alternate tail) and
`baffle_cable_routing_v1lf.png`; there is no generic shared sheet.

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
| **V1LF R6F** | legacy four-piece baffle; replaced by two mandatory carriers plus add-ons | **Extreme barebone flush carriers**: LM Ø190 opening / Ø221.2 seat / nominal R113.0 lip; UM Ø82 opening / Ø98.6 seat / R51.7 outside. Two Ø9 rounded M3 half-lap ears at x=±32.0 set the 165.100 mm spacing. V1LF-only LM axes rotate to 0/60/120/180/240/300° on the unchanged Ø209.5 PCD. Six actual Ø5×2 magnets use global Ø5.2×2.2 pockets: preserve the flush upper LM pair at 64°/116°, add a flush lower LM pair at 224°/316° with at least 23.0 mm nearest-insert edge clearance, and keep the UM pair flush at 50.5°/129.5°, z=15.1. The LM 224° pocket remains at z=12.55; the route-adjacent 316° pocket uses the Z-preferred z=15.40 site and retains a closed 0.30 mm front skin. Hold all six magnets flush during bonding; the extra 0.2 mm pocket depth is adhesive allowance, not a bottoming datum. Neither carrier has proud magnet ears. Direct UM-to-tweeter ears sit at x=±24, y=421.5. UM is buried in an Ø8.2 passage only inside LM, then runs free behind UM with no printed UM-carrier rear duct. T is buried in an Ø6.0 passage through LM/UM, then runs free behind the tweeter crescent, which has no printed cable arc. Every surviving named insert bypass has a deep full-width burial web; the D7.8 LM lead is a free span without a printed micro-duct. The physical T/UM routes cross at 82.67° with no two-duct separator-web claim. No-floor LM includes the front-flush bridge plate at z=5.3..18.3, with a soft cubic blend into R113, four unchanged rear insert bores, two centered rear cable mouths at x=±5/y=82, 6.2 mm front floors, and no geometry behind the existing LM-pad envelope; the floor LM has no bridge plate and keeps supported ring feeds. The tweeter carrier is a separate add-on; V1LF has no printed grommet. | `lx521_top_v1lf_core_1of2_lm_carrier.stl`, `lx521_top_v1lf_core_2of2_um_carrier.stl` |

The V1LF LM print form is a separate choice inside the same R6F variant. The
canonical `lx521_top_v1lf_core_1of2_lm_carrier.stl` is one solid. On a 220 mm
square bed, replace it with **both**
`lx521_top_v1lf_optional_lm_keyed_1of2_bottom.stl` and
`lx521_top_v1lf_optional_lm_keyed_2of2_top.stl`; never combine either half with
the monolithic carrier. Their state-specific final geometry is cut at world
Y=172.481 mm with a zero-gap planar butt. The authoritative measured Z26°/Z45°
X×Y footprints are 198.79×138.22 / 210.47×210.47 mm in floor state and
198.79×205.51 / 210.47×210.47 mm in no-floor state. One concealed right-hand
straight rounded tongue/blind-socket pair is carved wholly inside the existing
R110.6..R113 lip. The tongue is 0.8 mm wide and engages 3.5 mm along its
tangential insertion axis, approximately 75.23° from +X. It adds no external
protrusion, envelope growth, or extra screw, and registers the loose halves but
has no
standalone retention/load credit; the installed LM driver flange
and all normal LM fasteners provide the structural splice. This optional form
remains pending tongue/socket fit, full-seat, coplanarity, route-seam,
cable-pull-through, and driver-installed 1g/3g/5g proof.

**V1 vs V1L:** V1 = thin TOP piece; V1L = thin BOTTOM+MIDS. Pair them
for the full front-flush thin baffle; either also works alone on the
other family's pieces (see matrix). The V1L outlet change is confined to
its keyed `mid_right`; it does not create a V1L-specific top.

## Add-ons (outline experiments)

| Family | Pieces | Fits | Anchoring |
|---|---|---|---|
| **A-comp shoulders** (18.3) | 4: `addonA_1..4of4` | B2 vase only | 2 FLUSH magnets/side: flare wall zc=5.0 + crescent arc zc=10.7; outline kinks register |
| **B1 wings** (18.3) | 2: `addonB1_1..2of2` | B2 vase only | same two sites |
| **V1 A-shoulders** (11.5) | 4: `v1addonA_*` | V1 vase (V1L sets) | TWO pins/side: lower zc=12.5, upper zc=14.4 (in-wall, no bosses) |
| **V1 B1-wings** (11.5) | 2: `v1addonB1_*` | V1 vase (V1L sets) | two pins/side: lower zc=12.5, upper zc=14.4 (in-wall, no bosses) |
| V0 scarf family | (designed, not built) | V0 | would scarf onto the knife band; pending |
| **R6F floor support** | `lx521_top_v1lf_addon_mount_floor_support.stl` | floor-state V1LF LM collar only; **required for the floor assembly** | three rear-installed support heat-sets at rotated 180/240/300° W22 axes, reached by long driver-side M5 screws through Ø5.5 carrier clearances; no obsolete LM magnet cups/arms; open twin rails + NL8 panel and only physical-cable-plus-clearance space for the free LM lead. No-floor has no separate support print. |
| **R6F tweeter crescent** | `lx521_top_v1lf_addon_tweeter_crescent.stl` | V1LF UM collar only | direct rounded half-laps at x=±24, y=421.5; rear-driven M3 screws enter blind crescent inserts; no printed T-cable arc or conduit, so T remains free behind the crescent |
| **R6P split grommet** | `lx521_top_proud_addon_um_grommet_half_{a,b}.stl` | standard B2/C7/V0/V1 R14 outlet; **not V1L** | TPU; short curved D8 shank follows the final bore and seats at rear z=0 |
| **V1L split grommet** | `lx521_top_v1l_addon_um_grommet_half_{a,b}.stl` | keyed V1L 283° outlet only | TPU; Ø8 curved body / Ø7.1 bore / 2.5 mm insertion / Ø13 × 2 flange seated at rear z=6.8; no fastener |

B2 addons on V0/V1: NO (knife/thin walls — no receiver seats).
V1 addons on B2/V0: NO (the zc=12.5/14.4 pockets exist only on the V1 vase).
R6F add-ons fit only V1LF interfaces and never fit R6P parts.
No acoustic perimeter skin is emitted in R6F: that absence is the
barebone experiment. Any later skin/wing or cable retainer must use the
defined alignment or threaded interfaces as an optional module, must
not obstruct any buried tunnel or covered Z bump, and may
not add material back to the mandatory collars. Magnets may register a
module but may never be counted in its structural load path.

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

**R6F V1LF is outside this matrix.** Its two collars, M3 half-laps, and
integral buried routes replace the proud-family shell, seams, and
subtractive cable network. Do not combine an R6F carrier/add-on with an
R6P base, or generate a V1LF STEP under `LX_ROUTING_PROFILE=proud`.
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

The V1LF LM/UM/T physical cables are Ø7.8/Ø7.0/Ø5.2. UM uses a printed
Ø8.2 passage only through LM; T uses a printed Ø6.0 passage through LM and
UM. The LM cable floats over a short 20.15 mm radial
span at 269.5° behind the carrier with no printed micro-duct, cover, or cutter;
the floor support retains only the physical cable plus 0.4 mm clearance.
No-floor UM/T cables enter through two centered rear bridge-plate mouths at
`(+5,82,5.3)` and `(-5,82,5.3)` and rise internally before the LM ring;
floor mode keeps the supported R114 ring entries. The UM cable exits its
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
bypasses outside the exact D6 lumen, blind-bore, flush-magnet and half-lap
interface voids. Floor mode removes only the exact
grown Ø12.4 printed-boss/Ø5.8 shank clearances at 300/240/180°; the mating
Ø11.6 support bosses retain 2.6 mm radial wall around their Ø6.4 heat-set
cavities, and all surrounding carrier material remains solid. Every surviving
buried span retains a 0.8 mm minimum wall and 0.85 mm seat roof. There are no
cable windows in those printed spans; the specified UM and T free spans are
intentionally visible from the rear. The state-specific
`baffle_cable_routing_v1lf.png` is the routing
reference and includes true longitudinal side profiles plus nominal diametric
u-z bump/pilot sections with authoritative vertical limits.
The fit STEP continues the physical Ø7.0 free UM cable behind the UM carrier
to the 283° terminal reference and through the R20 service turn to a Y breakout.
The breakout uses a 4 mm-long OD8 collar and two OD4 branch sleeves. Two
provisional Ø3.2 conductors then follow R8-minimum slack paths into
separate low-profile flag boots. One connector at a time is checked at
0/3/6/9/12 mm pull while the other remains installed.

The V1LF free UM service path reaches the **283-degree axis**, the exact
midpoint between mounting screws 238 and 328 degrees. Use coupon 9 as
the physical witness and clock the MU terminals to that axis.
`top_baffle_nd25fw4_um_fit.step` shows the V1L legacy withdrawal volume and,
for V1LF, the installed non-overlapping low-profile 6.3 mm flag-Faston
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
`V1LF_PHYSICAL_QUALIFICATION.md`.

## Hardware

* W22: 6x M5 x 5.8 x O6.3 heat-sets (bore O6.4 x 6.8).
* 10F/MU10: 4x M3 x 3 x O5 heat-sets (bore O4.6 x 4.0), pattern clocked
  to 58/148/238/328 degrees. For V1LF, select the equivalent driver
  rotation that puts the terminal carrier at the 283-degree witness.
* R6F collar structural joint: 2x M3 through-bolts through the half-laps;
  choose length, washers, and nuts against the printed 11.5 mm stack.
* R6F optional LM split: assemble both front faces down on one flat datum and
  fully seat the bottom half's one concealed right-hand straight rounded tongue
  in the top half's blind socket. The pair registers alignment only; assign it no
  standalone retention/load credit. The LM flange and all normal
  LM fasteners provide the installed splice across the Y=172.481 mm seam.
* R6F floor support: use long driver-side M5 screws through the rotated
  180/240/300° carrier clearances into three rear-installed support
  heat-sets. It has no LM magnet cups/arms; the free LM-lead opening is only
  cable clearance. No-floor instead uses the four immutable bridge inserts at
  (±20,20)/(±20,70) in the monolithic LM solid web. The Ø6.4 × 6.8
  bores open at z=5.3 and retain a 6.2 mm solid front floor.
* R6F tweeter crescent: 2x rear-driven M3 screws at x=±24, y=421.5 into
  blind Ø4.6 × 4.0 insert receivers in the crescent half-laps. The
  acoustic front remains uninterrupted.
* R6F alignment: 6x D5 × 2 N52 magnets in the core's radial Ø5.2 × 2.2
  pockets (four LM + two UM; three total per physical side). Preserve the
  upper flush LM sites at 64°/116° with at least 2.2 mm insert gap, and add
  the lower LM sites face-flush at 224°/316° with at least 23.0 mm nearest-
  insert edge clearance. The 224° site remains at z=12.55 and the route-
  adjacent 316° site uses z=15.40 with a closed 0.30 mm front skin; both have
  verified buried-route and bridge/support clearance. UM keeps its
  flush 50.5°/129.5° sites at z=15.1 to keep
  at least 1.1 mm from the conservative T-cover envelope, a 0.2 mm radial
  floor and a 0.6 mm front skin. Hold every magnet flush during bonding;
  the extra 0.2 mm pocket depth is adhesive allowance, not a bottoming datum.
  No site has a proud ear. Keep a marked polarity standard:
  core pole OUT, add-on pole IN. Magnets are alignment/anti-rattle
  devices only and receive **zero structural load credit**.
* Tweeter pair (when its carrier is fitted): M4 through-bolts + nyloc,
  clamping the crescent.
* R6P magnets: D5 x 2 N52 discs in Ø5.2 × 2.2 base and receiver
  pockets. Fixture both magnets flush while bonding so they meet level;
  bottoming either disc would leave it recessed by the 0.2 mm adhesive allowance.

## R6F structural screen and release gate

The fused four-hole bridge interface has a 62 mm insert core and soft cubic
shoulders occupying z=5.3..18.3, exactly the deepest existing LM-pad envelope.
Two centered rear cable lumens enter at x=±5/y=82 while the acoustic front
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

Floor mode has its own 456.9 mm³ minimum U-section screen, with required
sections about 128.1/170.8/284.7 mm³ and safety factors about
3.57/2.67/1.60 at 1g/3g/5g; its combined-axis insert factor is about
2.00 at 5g. Floor load enters the LM through only three
spokes; their direct-shear factors are about 17.1/12.9/7.7. The upper
joint case uses the actual 0.43 kg MU + 0.20 kg tweeters plus printed and
hardware allowance, **0.85 kg total**, over conservative 120 mm plan and
70 mm rear levers. Its governing contact factors are about
**2.82/2.12/1.27**, with M3 tension factor about **1.17** at 5g. Magnets
contribute 0 N throughout.

These numbers are analytical screening, not release evidence. Every
finished web/core/insert/bolt combination must pass a documented
physical proof test at the distributed 4 kg sustained-1g, transient-3g
and transient-5g bridge/support design loads, while the upper joints see
their actual 0.85 kg distributed mass and lever arms. Record the
fixture, duration, temperature, permanent deflection, insert movement,
cracking, and post-test torque. Ramp over at least 10 s and hold 1g for
24 h, 3g for 60 s and 5g for 10 s. The calculations cover the modeled
insert groups, minimum printed sections, ear net/neck/bearing and M3
shear; they do not independently qualify floor rails/panel, stock bridge,
real insert process, or installation substrate. Bambu's
published 61 °C heat-deflection temperature is not an assembly service
rating; direct sun, a closed vehicle, or another hot environment
requires a suitable material and a proof test at the actual maximum
service temperature.

The monolithic-LM screen does not qualify the optional keyed split by
inheritance. Its concealed tongue/socket remains at zero standalone retention and
structural credit; separately record tongue/socket fit, full seating,
coplanarity, route-seam continuity, cable pull-through, and
driver-flange-installed 1g/3g/5g proof for each selected floor/no-floor split
candidate.
Record the complete floor/no-floor candidate identity, print and insert
process, load fixture/history, measurements, evidence hashes and independent
release signoff in `V1LF_PHYSICAL_QUALIFICATION.md`. Evidence from one state
does not authorize the other.

## Stable routing review and fit artifacts

Each stand-state folder contains:

- `top_baffle_nd25fw4_v1lf_split.step` — mandatory two-collar core;
- `top_baffle_nd25fw4_v1lf_lm_split.step` — optional two-print LM form,
  mutually exclusive with the monolithic LM carrier;
- `stl/lx521_top_v1lf_optional_lm_keyed_{1of2_bottom,2of2_top}.stl` —
  the two parts required when that optional LM form is selected;
- `top_baffle_nd25fw4_v1lf_attachments.step` — state-owned printable
  attachments; its floor support is required in floor state, while the
  remaining modules are optional;
- `top_baffle_nd25fw4_v1lf_assembled.step` — core, add-ons, and fit
  proxy together for collision review;
- `top_baffle_nd25fw4_um_fit.step` — terminal/Faston proxy, standard,
  V1L, and V1LF D7 cable envelopes, plus the proud/V1L split strain-relief
  profiles (not manufacturer hardware geometry; V1LF has no printed grommet);
- `stl/lx521_top_v1l_addon_um_grommet_half_{a,b}.stl` — printable
  keyed V1L TPU strain relief;
- `baffle_cable_routing_proud.png` and
  `baffle_cable_routing_v1lf.png` — the two isolated route sheets; the
  proud sheet includes both the normal R6P UM tail and the labeled
  V1L-only 283-degree alternate;
- `lx521_coupon_9_um_faston_clocking.stl` — the physical MU clocking
  gate; the complete current coupon list is in PRINTING.md; and
- `v1lf_release_manifest.json` — hashes the state candidate and its pending
  qualification record; it is provenance, not physical-release authority.

See PRINTING.md for print settings and torques; `make check` guards
clearances, seam keys, complete-route smoothness, eroded-outline
containment, service envelopes, state manifests, and cutter health.
