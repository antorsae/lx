# Variant catalog

There are now two intentionally isolated systems:

- The **proud family** — Stock and Slim — prints as four pieces joined at
  three seams:
  `piece_bottom`, `piece_mid_left`, `piece_mid_right`, and `piece_top`.
  All five variants retain the same seams (A: y=120, teeth ±66/±103;
  B: y=315.95, teeth −19/+29; C: x=−5.6, tooth y=305) and matching
  seam mouths. Seam B also carries the common hidden radial M3×20 joint at
  `(x,z)=(0,12.55)`: `mid_right` owns the recessed Ø3.4 screw passage and
  every vase owns the blind Ø4.6×4.0 heat-set receiver. B2 and V1 use the ordinary Ø8.2/G1/R14 UM outlet at
  (33.446, 301.492). V1L is a keyed routing exception: its Ø8.2
  alternate tail stays wholly in `piece_mid_right` and exits the
  z=6.8 rear face at Q=(13.497063, 307.618796), radius 60.0 mm on the
  283-degree terminal axis. Because it never reaches seam B, top/vase
  interchangeability is preserved.
  In no-floor form, every Stock/Slim variant also shares Obi-Wan's exact
  three-port D20 interface relative to the four bridge inserts: LM
  `(-0.35,64.76)`, one Ø6 shared-T trunk `(-4.75,55.91)`, and UM
  `(3.17,55.91)`.
- **Obi-Wan** is an extreme skeletal system: only one LM carrier and
  one UM collar are mandatory. It has no proud-family seams or full
  outline. UM is buried only through the LM
  carrier and is free behind UM; T is buried through LM/UM and is free behind
  the tweeter crescent. The surviving printed spans retain 0.8 mm minimum
  walls, a 0.85 mm seat roof, and covered Z bumps backed by solid roof-to-bore
  saddles. No-floor LM/T/UM enter through the one D20 support opening and the
  D7.8 LM reference follows its buried Ø9 branch to the common R14 handoff.
  No-floor mode fuses the stock four-hole solid front
  web into the LM core and has no separate keel.
  Floor mode instead fuses the full-height W64 stem/foot, convex
  constant-thickness Option-B transition (centreline Rmin 41 mm), buried
  floor lanes and NL8 panel directly into the LM carrier. There is no separate
  floor-support part or support fastener. Tweeter, outline, and retention
  remain selectable add-ons. The canonical floor LM is a large-format
  monolith; an optional two-print keyed split may replace it, but is never
  added to it, and its bottom half inherits the complete stand.

Both systems are generated in `build/floor_stand/` and `build/no_floor_stand/`.
Their review sheets are `baffle_cable_routing_proud.png` (normal proud
route plus its labeled V1L-only alternate tail) and
`baffle_cable_routing_obiwan.png`; there is no generic shared sheet.

> **Retired labels.** The proud routing family was historically labeled
> **R6P** (sixth cable-routing revision, proud channels) and the Obi-Wan
> system **R6F** (the same revision, flush/barebone). Both labels are retired
> from the documentation: say "proud routing" or name the product. The strings
> survive only inside code — test file and case-ID names, `LX_R6F_*`
> environment variables, the `.check-stamps/r6f/` tree and the
> `R6F_NATIVE_STAGE_SCHEMA_VERSION` schema key — where they are internal
> identifiers, not descriptions.

Drivers (LX521.4 production): LM = SEAS **U22REX/P-SL** (H1659-08,
flange O220.6 x 6.0 measured); UM = SEAS **MU10RB-SL** (H1658-04,
flange O98 x 4.0 measured). Older comments naming the LX521 prototype
drivers (W22EX001 / 10F) refer to the same cutout/pilot geometry.

## Base variants

> **C7 and V0 were retired from the build in August 2026** — no build targets,
> exports, or catalog entries remain; their geometry modules live in git history
> only. Their rows below and in the compatibility matrix are kept as design
> history and are marked *(retired)*.

| Variant | Replaces | Geometry | STLs |
|---|---|---|---|
| **B2** | (baseline, all 4) | Full 18.3 everywhere. Constant-wall mini-vase (walls tangent to r=50.83 about the UM). | `stock_1..4_of_4_*` |
| **C7** *(retired)* | bottom + mids (+B2 vase) | LM knife taper: REAR-side smoothstep 18.3 -> 0.5 over 19 mm from the flank/chamfer edges; recovery lands at both seams; full bottom strip. Front plane intact. | `lx521_top_c7base_1..4of4` |
| **V0** *(retired)* | vase | Rear knife band: REAR-side 18.3 -> 0.5 over the last 2.8 mm of the vase outline (same sculpted side as C7); front intact. | `lx521_top_v0_4of4_vase` |
| **V1** | vase | Thin FLUSH vase: 11.5 (material z 6.8..18.3). Crescent re-derived (4.0 clamp seat at stock z); tweeter septum 11.5 (shorter standoffs, pair spacing -6.8); one shared front plane. | `slim_4_of_4_vase_b2` (from the `--variant v1l` export; the duplicate standalone `lx521_top_v1_4of4_vase` was retired in August 2026, and `--variant v1` now emits only the slim receivers) |
| **V1L** | bottom + mids | Thin FLUSH LM section: 11.5 (z 6.8..18.3 -- SAME plane as the V1 vase: no seam-B step), including both 6-mm seam-B male dovetails that project into the vase. All joints use B2's regular through-local-thickness profiles. Rear-thickness ramp back to the full strip is stand-state dependent: no-floor keeps the smoothstep y=78..96; the floor stand runs one quintic smootherstep in PATH LENGTH over s=0..159.589 mm (43.85 mm of flat plate from y=118 down to the Option-B vertical tangent, then the 115.739 mm bend sweep), reaching full 18.3 mm depth exactly at the arc's HORIZONTAL tangent where the foot begins, 12.393 mm at the vertical tangent, and staying slim through the seam-A dovetails. Its three floor cable lanes are rerouted convex-ward as quintics to follow the thinning concave face (rear covers LM 1.650 / UM 2.149 / TS 1.941 mm, lane radii R47.5 / R42.5 / R46.9). Ø8.2 LM duct is the 11.5 binder. Its keyed Ø8.2 UM alternate exits `mid_right` at Q=(13.497063, 307.618796, 6.8) on the 283° axis; seam B/top are untouched. | `slim_1..3_of_4_*` (its `--variant v1l` export bundles the unchanged V1 vase = the complete ~12 mm baffle) |
| **Obi-Wan** | legacy four-piece baffle; replaced by two mandatory carriers plus add-ons | **Extreme barebone flush carriers**: LM Ø190 opening / Ø221.2 seat / structural R113.0 lip with smooth exposed R113.8 side fairing; UM Ø82 opening / Ø98.6 seat / structural R51.7 lip with smooth exposed R52.5 side fairing. The fairings stop only inside the existing LM–UM and T–UM cusp/service regions; the LM–UM stop preserves the 0.40 mm inter-carrier gap. Rounded M3 half-lap pairs sit at x=±32.0/y=315.770 and x=±24/y=421.5. At both interfaces the closure-web/base teardrops remain nominal Ø9, while every complete Z-owned cylindrical functional boss is locally Ø9.8. LM and UM respectively own complete standalone rear Ø3.4 passages; UM and the crescent respectively own complete standalone rear-opening blind Ø4.6 × 4.0 receivers with 360° walls and 1.9 mm front floors. Each joint retains a 0.20 mm axial gap. Install inserts in the individual UM and crescent prints before assembly; neither interface uses a washer, nut, front bolt head, or cross-owner receiver wall. Complementary tangent-blended LM–UM and T–UM closure webs are solid through z=6.8..18.3 and share the coplanar z=18.3 front; only the central T cable mouth remains open between the upper rings. Obi-Wan-only LM axes rotate to 0/60/120/180/240/300° on the unchanged Ø209.5 PCD; all six are ordinary blind carrier insert bores in both states. Six actual Ø5×2 magnets use captive Ø5.20×2.10 cavities with 0.45 mm axial skins and a 45° closing roof. Upper LM remains ring-radial at 64°/116°; lower LM sits at cubic parameter `u=0.50` on the shared curved shoulder, with right visible datum `(x,y,z)=(45.285011,89.190370,15.10)`, outward normal `(0.706451,-0.707762)`, and an exact-X-mirrored left datum; UM remains ring-radial at 50.5°/129.5°. Every carrier station shares source Z=15.10. Both states expose the same upper shoulder at that station, with no floor-state rail or shallow box below it. LM-upper/UM cavity datums are structural radius +0.65 mm; the lower shoulder datum is independently inset 0.15 mm. All three lie 0.15 mm beneath their continuous exposed surfaces. The magnet-free carrier exterior is immutable: there is no magnet-local backing, boss, relief, rear cap, flat, or visible pocket cue. Magnets are inserted at the authoritative pause and permanently buried, with no glue or external opening. Neither carrier has proud magnet ears. Flat/graded provide three matching captive receivers per physical side: LM lower, LM upper, and UM. Their mating surfaces are flush with zero physical air gap; the receiver's 0.05 mm allowance is a solid spacing standoff, not an air-gap cutter. Nominal paired magnet-face separation is 1.10 mm at LM-lower, LM-upper, and UM. UM is buried in an Ø8.2 passage only inside LM, then runs free behind UM with no printed UM-carrier rear duct. T is buried in an Ø6.0 passage through LM/UM, then runs free behind the tweeter crescent, which has no printed cable arc. Every surviving named insert bypass has a deep full-width burial web. In no-floor state, LM/T/UM enter through the D20 support opening as LM above, T lower-left and UM lower-right; the D7.8 LM lead follows a buried Ø9 branch to the common R14 handoff. The LM-owned UM/T lumens finish at R112.95 and their 0.8 mm covers at R113.75 beneath the uninterrupted visible R113.8 carrier exterior, retaining a 0.85 mm solid outside skin with no groove. The physical T/UM routes cross at 82.95° with a 2.00 mm physical-envelope gap and no two-duct separator-web claim. No-floor LM includes the unchanged front-flush bridge plate at z=5.3..18.3. Floor LM instead owns the complete W64 × 18.3 stem/foot from z=−150..18.3, its convex constant-thickness Option-B transition (75 mm span, 65 mm rise, centreline Rmin 41 mm), three buried floor continuations, connector service cavity and rear NL8 panel; floor Y=0 keeps the LM axis exactly 200.981 mm above the floor. The tweeter carrier is a separate add-on; Obi-Wan has no printed grommet. | `obiwan_core_1_of_2_lm_carrier.stl`, `obiwan_core_2_of_2_um_carrier.stl` |

The Obi-Wan LM print form is a separate choice inside the same Obi-Wan variant. The
canonical `obiwan_core_1_of_2_lm_carrier.stl` is one solid. On a 220 mm
square bed, replace it with **both**
`obiwan_optional_lm_keyed_1_of_2_bottom.stl` and
`obiwan_optional_lm_keyed_2_of_2_top.stl`; never combine either half with
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
Their worst-case reach is R114.4036: 1.4036 mm beyond structural R113.0 and
0.6036 mm beyond the finalized R113.8 visible fairing. They register the loose halves,
but add no extra screw and have no standalone retention/load credit; the installed LM driver flange
and all normal LM fasteners provide the structural splice. This optional form
is geometrically compatible with flat/graded through matching 0.25 mm clearance
pockets cut only into the hidden carrier interface between the wing faces;
physical printed fit remains coupon-qualified. It otherwise
remains pending slicer proof of both four-nozzle-width horizontal pins, both
lands and all minimum walls, process-matched two-pin/socket fit, actual U22
fit, full-seat, coplanarity, route-seam, cable-pull-through, and driver-installed
1g/3g/5g proof.

With the monolithic LM, the same flat/graded pockets remain as small hidden local
reliefs. They do not move the three magnetic datums or change the primary
magnetic retention contract.

**V1 vs V1L:** V1 = thin TOP piece; V1L = thin BOTTOM+MIDS. Pair them
for the full front-flush thin baffle. The V1L outlet change is confined to
its keyed `mid_right`; it does not change the acoustic outline of the top.
The complete V1L export retains the unchanged V1 top and its regular
through-thickness female pockets.

## Opposed TEBM35C10-4 BMR vase alternative

![Opposed BMR vase](../images/generated/iso/tweeter_tebm35c10_4_vase.png)

That is the ISO cell from `make iso_matrix`, drawn on the shared
tweeter-option scale. Tracked review renders of both profiles — acoustic
front, ISO and rear — are in
[`review/vase_tebm35c10_4/family/`](../review/vase_tebm35c10_4/family/), and
the release snapshots are in
[`review/tebm_release_snapshots/`](../review/tebm_release_snapshots/).

`vase_TEBM35C10-4` is a first-class optional vase family with two envelope
profiles. Both replace the Dayton tweeter crescent with two opposed
TEBM35C10-4 BMRs: the lower driver faces front, the upper driver faces rear,
both use four M2 × 4 × Ø3.2 insert bores, and each default Ø63 land has two captive
D5 × 2 side magnets. The driver pockets keep 1.2 mm blind rear/front walls.
The shared T route terminates at the lower pocket while a separate tangent
branch reaches the upper pocket; the guarded BREP gates allow no exterior
opening except the seam-B inlet and the two declared pocket outlets.

The land choice is independent of the Stock/Slim rear envelope. `full` is a
clipped Ø63 circle whose side-magnet faces at `x=±31.326666` set a 62.653 mm
maximum width, moving each face inward by about 1.508 mm. `bmr-slim` is an
unqualified driver-following alternative: a
Ø56 core, four local M2 pads and two discrete side-magnet lobes reaching the
same faces and width. It retains the magnets instead of deleting or moving
them, but reduces the local land plan area by about 19.5%. The drawing gives
no tolerance, flange thickness, lug detail or terminal envelope, so both new
land topologies are prototypes and `physical_measure_required` remains true.

The acoustic plane remains z=18.3 and both profiles reach the same local BMR
rear plane z=-6.8, giving the published 25.1 mm flush driver depth. Both use
the same regular seam-B XY profile; only the local vase thickness differs:

| Profile | Lower-vase rear at seam B | Smooth rear growth | Seam-B joint |
|---|---:|---:|---|
| Stock BMR | z=0.0 | 6.8 mm; begins at y=391.709 | regular 10/14/6 mm female pockets through 18.3 mm |
| Slim BMR | z=6.8 | 13.6 mm; C2 growth begins at seam B with zero slope/curvature | same regular female pockets through 11.5 mm |

The Stock and Slim BMR vases use the same seam-B plan interface as the normal
B2/V1 vases. Mixed thicknesses retain the documented hidden rear step/open
notch behavior. The two default full-land CAD children live under
`build/vase_TEBM35C10-4/{stock,slim}/`; their BMR-slim counterparts live under
`build/bmr_slim_TEBM35C10-4/proud/{stock,slim}/`. Build the full-land profiles with
`make vase_tebm35c10_4_cad`, or one named child with
`make vase_tebm35c10_4_{stock,slim}_cad`. Ready Bambu projects are local-only
targets named `vase_tebm35c10_4_{stock,slim}_3mf` and are promoted to the
stable sibling file `vase_TEBM35C10-4.gcode.3mf` in each child root.
Build all four CAD-only BMR-slim alternatives with
`make bmr_slim_candidates_cad`; that target intentionally creates no slicer,
release-catalog or `to_print` artifact.

## Candidate TEBM35C10-4 BMR crescents (Obi-Wan)

Obi-Wan has no seam B, so the vase above cannot fit it. Its BMR options mount
on the tweeter crescent's interface instead. There are two of them, both
candidates, both built by the local-only `make obiwan_bmr_crescent_cad` into
`build/bmr_crescent_TEBM35C10-4/`, and both keeping the released ND25FW-4
crescent's UM-collar mate exactly — so any of the three is swappable for the
others without touching the UM print — while keeping nothing else of that
crescent's outline. Each default artifact uses one or two clipped Ø63 lands,
joined to the collar by the same solid flush skirt, fed by the same hidden
Ø6.00 cable entry on the mate face, and carrying the vase layout's captive
D5 × 2 side magnets.

- **`obiwan_bmr_crescent_TEBM35C10-4`** — *coaxial*. Both BMRs on the fixed
  axis `(0, 452.494193)`, back to back, 15.699 mm below the released tweeter axis,
  which puts them 86.413 mm from the MU10 axis instead of 102.112 mm. Two
  25.1 mm envelopes stack to 50.2 mm. The full-land prototype is 62.653 mm
  wide and has **2 captive magnets** on its one outward land.
- **`obiwan_bmr_crescent_opposed_TEBM35C10-4`** — *opposed*. The vase's
  side-by-side layout on the crescent mount: the same lower axis at
  `(0, 452.494193)` facing front and a second at `(0, 501.794193)` facing
  rear, one vase pitch (49.3 mm) above it, both inside one 25.1 mm envelope.
  The full-land prototype is 62.653 mm wide and has **4 captive magnets**,
  two per land.

Full descriptions, including the per-variant open qualification items, are in
[`obiwan.md`](obiwan.md#tweeter-options).

![Candidate coaxial BMR crescent](../images/generated/iso/tweeter_tebm35c10_4_crescent.png)

![Candidate opposed BMR crescent](../images/generated/iso/tweeter_tebm35c10_4_crescent_opposed.png)

All three BMR arrangements use the same driver, the same fixed acoustic axes,
the same 1.20 mm blind pocket wall and the same side-magnet interface. Their
default `full` topology is the clipped Ø63 land; the unqualified BMR-slim
alternative substitutes a Ø56 core with M2 pads and local lobes without
moving the magnet faces. What differs is how the pair is arranged and what
carries it, and that drives everything else below:

| | Opposed vase (Stock, Slim) | Coaxial crescent (Obi-Wan) | Opposed crescent (Obi-Wan) |
|---|---|---|---|
| Driver axes | two, 49.3 mm apart in Y at y=443.931 and y=493.231 | one fixed axis at y=452.494193; changing land topology does not move it | two fixed axes at y=452.494193 and y=501.794193, preserving the same 49.3 mm pitch |
| Facing | lower driver front, upper driver rear | front driver +z, rear driver −z, back to back | lower driver front, upper driver rear |
| Local depth | 25.1 mm; both drivers share one envelope, z=18.3 to z=−6.8 | 50.2 mm; the envelopes stack, z=18.3 to z=−31.9 | 25.1 mm; both drivers share one envelope, z=18.3 to z=−6.8 |
| Shared wall | none — the two pockets sit side by side | a 2.40 mm partition, two independent 1.20 mm blind walls back to back, with one declared Ø4.60 lead pass | none — the two pockets sit side by side, 6.374 mm apart on the axis line |
| Mounts on | the seam-B vase interface, replacing piece `04` | the UM half-lap ears at x=±24, y=421.5 | the same UM half-lap ears |
| Surrounding structure | the full vase piece it replaces | one constant-plan Ø63 land, or the Ø56-core BMR-slim land, plus a solid skirt filling the plan between it and the collar | two constant-plan lands plus the same solid skirt under the lower one; the default Ø63 circles overlap by 13.7 mm |
| Cabling | the vase's own Ø4.6 lead outlets | one Ø6.00 entry on the UM mate face, in line with the collar's T emergence; nothing opens on the assembled exterior | the same Ø6.00 entry into the lower chamber, then one Ø4.60 branch across the waist to the upper one; nothing opens on the exterior |
| Captive magnets | 4, two per selected land | 2, the vase's lower/front pair on its one outward land | 4, the vase's own two pairs |
| Maximum land width | 62.653 mm for both full and BMR-slim | 62.653 mm for both full and BMR-slim | 62.653 mm for both full and BMR-slim |
| Status | prototype; not release-authorized after the land change | candidate; not release-authorized | candidate; not release-authorized |

## Add-ons (outline experiments)

| Family | Pieces | Fits | Anchoring |
|---|---|---|---|
| **A-comp shoulders** (18.3) | 4: `addonA_1..4of4` | B2 vase only | 2 captive magnets/side: flare wall and curved crescent stations share source Z=15.10; lower/upper nominal pair spacing 0.95/1.09 mm; outline kinks register |
| **B1 wings** (18.3) | 2: `addonB1_1..2of2` | B2 vase only | same two sites |
| **V1 A-shoulders** (11.5) | 4: `v1addonA_*` | V1 vase (V1L sets) | both captive stations share source Z=15.10; the upper land is contained by the broad symmetric smooth taper shelf, with no local magnet geometry; lower/upper nominal pair spacing 0.95/1.09 mm |
| **V1 B1-wings** (11.5) | 2: `v1addonB1_*` | V1 vase (V1L sets) | same common-Z captive stations and broad smooth taper shelf |
| V0 scarf family *(retired)* | (concept only; never released) | V0 | no released mate or pairing polarity; V0's two rear-axis base cavities print front-face-down at symmetric `(±6.690,321.290)`. The detached legacy `(±46,324)` and interim `(±37.697,326.470)` pair are rejected. Both released R3.20 lands fit wholly inside the immutable post-bevel host, clear the D82 cutout, UM pilots, grown seam-B keepout, and all ducts, and require no local backing, boss, or visible rear cue. |
| **Obi-Wan BMR crescents** *(candidates)* | `obiwan_bmr_crescent_TEBM35C10-4.stl`, `obiwan_bmr_crescent_opposed_TEBM35C10-4.stl` | Obi-Wan UM collar only; mutually exclusive with the ND25FW-4 crescent and with each other | the identical half-laps at x=±24, y=421.5, blind Ø4.6 x 4.0 receivers, 1.9 mm front floors and 0.20 mm axial gap, proven ear-for-ear against the released crescent and by assembling on the staged UM collar; one or two constant-plan Ø63 lands are joined to the collar by a solid skirt landing on the released crescent's own seam, with no inherited crescent outline and no M4 clamp holes; the lower axis stays fixed at y=452.494193 independently of land radius; one hidden Ø6.00 cable entry on the mate face and no exterior opening at all; the vase's captive D5 × 2 side stations at source Z=15.10 on each land's own flat — 2 on the coaxial pod, 4 on the opposed one — remain at x=±31.326666 and are buried behind the 0.45 mm skin; the optional BMR-slim plan keeps those stations on discrete lobes around a Ø56 core but remains physically unqualified; both parts stay **not release-authorized** and absent from the release inventory, the stage manifests and the released captive-magnet catalog |
| **Obi-Wan tweeter crescent** | `obiwan_addon_tweeter_crescent.stl` | Obi-Wan UM collar only | direct half-laps at x=±24, y=421.5; UM owns complete rear Ø3.4 ears and the crescent owns complete front local-Ø9.8 ears with standalone blind Ø4.6 x 4.0 insert receivers, 360° walls, 1.9 mm front floors, and a 0.20 mm axial gap; no printed T-cable arc or conduit, so T remains free behind the crescent |
| **Proud split grommet** | `stock_um_grommet_half_{a,b}.stl` | ordinary B2/V1 R14 outlet; **not V1L** | TPU; short curved D8 shank follows the final bore and seats at rear z=0 |
| **V1L split grommet** | `slim_um_grommet_half_{a,b}.stl` | keyed V1L 283° outlet only | TPU; Ø8 curved body / Ø7.1 bore / 2.5 mm insertion / Ø13 × 2 flange seated at rear z=6.8; no fastener |

B2 addons on V1: NO (thin walls — no receiver seats).
V1 addons on B2: NO (the source-Z=15.10 slim cavity contract exists only on the V1 vase).
Obi-Wan add-ons fit only Obi-Wan interfaces and never fit proud-family parts. The floor stem,
foot and NL8 panel are state-owned LM geometry, not an add-on.
No acoustic perimeter skin is emitted on Obi-Wan: that absence is the
barebone experiment. Any later skin/wing or cable retainer must use the
defined alignment or threaded interfaces as an optional module, must
not obstruct any buried tunnel or covered Z bump, and may
not add material back to the mandatory collars. Magnets may register a
module but may never be counted in its structural load path.

All released magnet-bearing variants now share the coupon-proven captive
system: D5.0 × 2.0 magnet, Ø5.20 × 2.10 internal cavity, 0.45 mm axial skin on
both faces, vertically open loading cradle, and support-free 45° roof. Every
paired transverse station in stock, slim, and Obi-Wan uses source Z=15.10.
The magnet-free exterior is immutable: cavities and lands remain internal,
with no magnet-local backing, boss, relief, rear cap, flat, or visible cue.
Mating surfaces are flush with zero physical air gap; the receiver's 0.05 mm
allowance is a solid spacing standoff, not an air-gap cutter. Stock lower
and curved-upper pair spacings are respectively 0.95 and 1.09 mm; every
Obi-Wan LM-lower/LM-upper/UM pair is 1.10 mm. Slim
hosts contain the upper land with a broad, symmetric, smooth taper shelf, not
a station-shaped patch. Every part prints front-face-down. Exact pause heights, grouped sites, local-axis
polarity, and counts live in
[`CAPTIVE_MAGNET_PAUSE_MANIFEST.md`](../review/captive_magnet_slice_audit/CAPTIVE_MAGNET_PAUSE_MANIFEST.md); STL files cannot carry
pause markers. Concept-only drawings, diagnostic renders, historical fit
coupons, and the retired V0 scarf mate are not release outputs and are not
converted as production parts. The `coupons/obiwan_ae_embed/` coupon remains the
reference implementation rather than an installed assembly component.

## Envelope compatibility matrix (bottom+mids x vase)

The acoustic outline, front plane, and duct-mouth datums permit the following
combinations. V1L's alternate UM tail is self-contained in
`mid_right`, so it does not alter this matrix or the top. Rear-side
steps at seam B land on the hidden side:

| bottom+mids \ vase | B2 (18.3) | V0 *(retired)* | V1 (11.5) |
|---|---|---|---|
| **B2** (18.3) | stock reference | edge experiment | vase-thickness experiment (rear step 6.8) |
| **C7** *(retired)* | LM-edge experiment | full knife-edge baffle | knife LM + thin vase |
| **V1L** (11.5) | thin LM only (vase protrudes 6.8 rearward) | thin LM + knife vase | **complete UNIFORM 11.5 front-flush baffle** |

Notes: mixed-thickness dovetails mate on the thinner piece's depth; the
through-pocket leaves a shallow open notch on the hidden rear. Tweeter
through-bolt length follows the vase septum (18.3:
~M4x35; 11.5: ~M4x30).

**Obi-Wan is outside this matrix.** Its two collars, rear-driven
insert-fastened M3 half-laps, and integral buried routes replace the
proud-family shell, seams, and
subtractive cable network. Do not combine an Obi-Wan carrier/add-on with a
proud-family base, or generate an Obi-Wan STEP under
`LX_ROUTING_PROFILE=proud`.
Proud pieces remain mutually interchangeable exactly as shown above;
choosing the V1L `mid_right` selects its 283-degree UM exit, while a
B2 `mid_right` retains the standard R14 exit.

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
UM. No-floor LM/T/UM enter through three bores packed wholly inside the D20
support opening centered at `(0,60)`: LM Ø9 at `(0,64.76)`, T Ø6 at
`(-4.75,55.91)`, and UM Ø8.2 at `(3.17,55.91)`. LM follows a buried Ø9 path
to the common R14 rear handoff; floor mode continues LM/UM/T through the
three buried integral-stem lanes. Both LM-owned UM/T lumens finish at R112.95
and their 0.8 mm covers at R113.75, beneath the uninterrupted visible R113.8
carrier exterior. The resulting outside skin is 0.85 mm with no groove.
The UM cable exits its
LM-owned passage and remains free behind the UM carrier; there is no printed
UM-carrier rear duct or D82 mouth. Its free centerline follows the modeled R15
terminal approach to the 283° service axis and then the R20 service turn. The
T path stays buried through the UM carrier, crosses the free UM cable at
**82.95°** with T above UM, then exits and remains free behind the tweeter
crescent. The crescent owns no printed cable arc, conduit, socket, or horn.
The gap between the physical cable envelopes at the crossing is 2.00 mm;
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
`um_fit.step` shows the V1L legacy withdrawal volume and,
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

* W22: 6x M5 x 5.8 x O6.3 heat-sets (unchanged 6.8 mm total bore:
  O6.5 x 2.0 entry, then O6.4).
* 10F/MU10: 4x M3 x 3 x O5 heat-sets (bore O4.6 x 4.0), pattern clocked
  to 58/148/238/328 degrees. For Obi-Wan, select the equivalent driver
  rotation that puts the terminal carrier at the 283-degree witness.
* Obi-Wan LM-to-UM collar structural joint: 2x rear-driven M3 screws pass through
  the LM's standalone Ø3.4 clearance bores into M3 x 3 heat-set inserts in the
  UM's standalone rear-opening blind Ø4.6 x 4.0 receivers. Each receiver
  sits inside a complete local Ø9.8 cylindrical functional boss; the
  closure-web/base teardrop remains nominal Ø9. Each receiver retains a 1.9 mm
  acoustic-front floor; the ear halves retain a 0.20 mm axial gap. Install both
  inserts in the individual UM print before assembly and
  choose screw length for full engagement without bottoming. No washer, nut,
  or front bolt head belongs to this interface.
* Obi-Wan optional LM split: assemble both front faces down on one flat datum and
  move the top straight along world -Y so both bottom-owned Ø1.60 +Y pins seat
  together in the top's right round and left X-relieved blind sockets. Do not
  flex, twist, or use one pin as a hinge. The pair registers alignment only;
  assign it no standalone retention/load credit. The LM flange and all normal
  LM fasteners provide the installed splice across the Y=172.481 mm seam.
* Obi-Wan LM: all six 0/60/120/180/240/300° driver sites use ordinary blind
  carrier heat-sets in both states. Floor mode has no support screws or
  secondary inserts: its stand is monolithic with the LM. No-floor additionally
  uses the four immutable bridge inserts at
  (±20,20)/(±20,70) in the monolithic LM solid web. The unchanged
  6.8 mm-total bores open at z=5.3 with a Ø6.5 × 2.0 entry followed by
  Ø6.4 and retain the existing 6.2 mm solid front floor.
* Obi-Wan tweeter crescent: 2x rear-driven M3 screws at x=±24, y=421.5 into
  standalone rear-opening blind Ø4.6 × 4.0 insert receivers in complete local
  Ø9.8 crescent half-laps. Install both inserts in the individual crescent
  before assembly; require complete 360° walls and 1.9 mm front floors. The
  UM owns the complete opposing Ø3.4 passages, the 0.20 mm axial gap remains
  open, and the acoustic front remains uninterrupted.
* Obi-Wan alignment: 6x D5 × 2 N52 magnets in the core's surface-normal captive
  Ø5.20 × 2.10 cavities (four LM + two UM; three total per physical side).
  Preserve the upper LM ring axes at 64°/116° with the validated insert/route
  clearances. The lower LM sites remain at cubic parameter `u=0.50` on the
  shared curved shoulder. The right visible datum is
  `(x,y,z)=(45.285011,89.190370,15.10)` with outward normal
  `(0.706451,-0.707762)`; the left is its exact X mirror, with verified
  buried-route and bridge/integral-stand clearance. Upper LM and UM use that
  same source Z=15.10; UM keeps its
  50.5°/129.5° axes. The R113.0/R51.7
  structural rings use smooth exposed R113.8/R52.5 side fairings, clipped only
  inside the existing LM–UM and T–UM cusp/service regions with the 0.40 mm
  LM–UM inter-carrier gap preserved. Ring cavity
  construction datums are structural radius +0.65 mm, 0.15 mm beneath the
  exposed surface; there is no magnet-local backing, boss, relief, rear cap,
  flat, or visible cue. Every
  magnet is inserted at its manifest pause and buried between two 0.45 mm
  skins under a 45° roof; there is no glue, access hole, or proud ear.
  Keep a marked polarity standard and use the manifest's local-axis table for
  mirrored parts. Magnets are alignment/anti-rattle
  devices only and receive **zero structural load credit**. Flat/graded use all
  three sites on each side through matching LM-lower, LM-upper, and UM
  receivers. Mating surfaces are flush with zero physical air gap; the
  receiver's 0.05 mm allowance is a solid spacing standoff. Nominal paired
  magnet-face separation is 1.10 mm at LM-lower, LM-upper, and UM.
* Tweeter pair (when its carrier is fitted): M4 through-bolts + nyloc,
  clamping the crescent.
* Proud-family magnets: D5 x 2 N52 discs in the same Ø5.20 × 2.10 captive base
  and receiver cavities. Print front-face-down, insert them only at the
  manifest pauses, and verify seating and polarity before the roof is closed.

## Obi-Wan structural screen and release gate

The fused four-hole bridge interface has a 62 mm insert core and soft cubic
shoulders occupying z=5.3..18.3, exactly the deepest existing LM-pad envelope.
Three rear cable lumens enter inside the D20 opening in the LM-above,
T-lower-left, UM-lower-right layout while the acoustic front remains solid.
It has no X-frame or additional rear-depth structure and is screened
conservatively for a 4.0 kg installed assembly, y=230 mm center of mass and
70 mm rear offset. The conservative member width is **38.8 mm**: the 62 mm
core minus the complete Ø9/Ø8.2/Ø6.0 entry lumens, with no credit for their
thin skins. Exact sampled soft-outline sections retain at least 45.73 mm. The
38.8 × 13.0 mm design section has in-plane/rear moduli of
**3261.8/1092.9 mm³**. Its summed biaxial stresses are about
**4.40/13.19/21.99 MPa**, giving safety factors about **1.82/1.36/0.82** at
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

- `obiwan_split.step` — mandatory two-collar core;
- `obiwan_lm_split.step` — optional two-print LM form,
  mutually exclusive with the monolithic LM carrier;
- `stl/obiwan_optional_lm_keyed_{1_of_2_bottom,2_of_2_top}.stl` —
  the two parts required when that optional LM form is selected;
- `obiwan_attachments.step` — optional tweeter attachment;
  the floor structure is already part of the floor-state LM carrier;
- `obiwan_assembled.step` — core, add-ons, and fit
  proxy together for collision review;
- `um_fit.step` — terminal/Faston proxy, stock,
  V1L, and Obi-Wan D7 cable envelopes, plus the proud/V1L split strain-relief
  profiles (not manufacturer hardware geometry; Obi-Wan has no printed grommet);
- `stl/slim_um_grommet_half_{a,b}.stl` — printable
  keyed V1L TPU strain relief;
- `baffle_cable_routing_proud.png` and
  `baffle_cable_routing_obiwan.png` — the two isolated route sheets; the
  proud sheet includes both the normal proud UM tail and the labeled
  V1L-only 283-degree alternate;
- `lx521_coupon_9_um_faston_clocking.stl` — the physical MU clocking
  gate; the complete current coupon list is in PRINTING.md; and
- `obiwan_release_manifest.json` — hashes the state candidate and its pending
  qualification record; it is provenance, not physical-release authority.

See PRINTING.md for print settings and torques; `make check` guards
clearances, seam keys, complete-route smoothness, eroded-outline
containment, service envelopes, state manifests, and cutter health.

## Retired variant design history

The two knife-edge experiments below were removed from the build in August 2026. They are kept here so the reasoning survives; nothing in this section is buildable.

### Variant C7 — LM knife-edge taper (retired)

> **Retired from the build in August 2026.** This variant no longer has build targets, exports, or catalog entries; its geometry modules remain in git history only. The description below is kept as design history.

An experimental replacement for the three LM-section pieces: full
18.3 mm around the W22, then the REAR face tapers (smoothstep over the
last 19 mm inside the flank/chamfer outline) down to a ~0.5 mm knife at
the edge -- the front face stays a full plane, exactly like the
crescent rear taper. It tests SL's "ideally the baffle would be even
thinner" in the band where the LM section's edges act (upper LM /
lower UM octaves), removing ~70 cm³ net (taper minus the T ribs).
The ducts sit at FIXED z from the rear face, so the binding rule is
z-interval containment: the rear cut over a duct must stay above
z_duct - r - skin (3.25 for the mid-plane mains; ~0 for the
rear-skinned T ducts).

- The cut fades in above the bottom strip (y 52..70: foot/bridge
  interface keeps full depth) and fades out toward seam B (y 270..~304:
  protected land and a flush joint to the shared vase piece). The four
  seam-A dovetail envelopes at ±66/±103 remain inside qualified material.
- The standard proud routing is shared by B2 and C7, so those pieces mix
  freely across the seams. Every duct remains inside the protected
  full-depth corridor; the tapered rear face carries no ribs or marks.
  This was asserted by the former test_c7_duct_corridor clearance check
  and verified with duct-envelope probes on the built piece solids.
- Print: same bed footprints as the B2 pieces; the taper prints
  front-face-down with layers shrinking as they rise (support-free).

### Variant V0 — minimalist UM vase (front slide) (retired)

> **Retired from the build in August 2026.** This variant no longer has build targets, exports, or catalog entries; its geometry modules remain in git history only. The description below is kept as design history.

An alternate piece_top for the low-crossover (3-4 kHz) experiments:
a REAR-side knife bevel (same side and philosophy as the C7 LM taper;
front plane fully intact) -- 18.3 -> ~0.5 over the last 2.8 mm inside
the flare/chamfer outline, fading out at the seam-B land and blending
into the crescent's rear taper above y~400. The band is capped at
2.8 mm by the shared O6.0 T duct (z=11.5) hugging the left vase
walls at ~1.6.
The standard top-piece routing was identical for B2/C7/V0/V1; V0
mixed with B2 or C7 bottom/mids freely. One D5 x 2 captive station per side
uses the common Ø5.20 x 2.10 pause-and-bury cavity. The old orphan centres at
`(±46.000, 324.000)` were 5.263 mm outside the exact B2 flare: even the
Ø5.20 cavity was detached by 2.663 mm. The interim `(±37.697, 326.470)`
pair was also rejected: its left site violated the T-route clearance and its
outboard right land required a visible rear-bevel backfill. The release uses
the symmetric inboard centres **`(±6.690, 321.290)`**. Each complete R3.20
land is already contained by the immutable post-bevel host, so the cavity
operation is subtractive only: there is no local keep, backing, boss, rear
block, or other exterior magnet-location cue. The pair retains at least
1.088 mm beyond the D82-cutout rule, 12.847 mm beyond the nearest-pilot rule,
1.089 mm beyond the grown-seam rule, and 18.579 mm beyond every route rule.
The rear axes, 45° conical closures, two 0.45 mm skins, driver seat, inserts,
and provisional rearward marked-pole directions remain unchanged. These
stations still have no released mate or pairing polarity.
The B2-family shoulders/wings do NOT fit V0. This was guarded by the
former test_v0_duct_corridor and test_v0_captive_geometry clearance
checks, which exported the retired `lx521_top_v0_4of4_vase`.
