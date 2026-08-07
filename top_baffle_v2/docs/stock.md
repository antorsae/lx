# Stock R6P — full-depth B2 product

The canonical product: the complete 18.3 mm baffle outline printed as four
registered pieces, with mutually exclusive A-comp shoulders or B1 wings as the
optional perimeter. Catalog entry: [`artifacts/stock/`](../artifacts/stock/).
Slim reuses this product's seams, key dimensions, and magnet contract, so this
file is the authority for both; see [`slim.md`](slim.md) for what Slim changes
and [`obiwan.md`](obiwan.md) for the unrelated R6F carrier system.

## Source modules

| File | What |
|---|---|
| `src/lx521_baffle/base.py` | Geometry library (drawing outline, holes, pilots); its own gen_step is the un-compromised aligned drawing (no artifacts kept) |
| `src/lx521_baffle/proud/a_comp.py` | Variant A-comp: straight-sided tower — vertical flanks at ±60.65 (tangent to B2's flare crest) from the extended top edge down to the LM chamfer-extension; tweeter section at the B2 drop. Buildable as B2 pieces + 4 shoulder pieces |
| `src/lx521_baffle/proud/attachments.py` / `build/common/attachments.step` | The 6 attachment pieces (exact boolean complements): 2+2 A-comp shoulders (top/bottom per side, split at the crest tangent), 2 B1 wings |
| `src/lx521_baffle/proud/b.py` | Shared B-family builder: mini-LM upper-mid vase (no shelf corners) + tweeter section lowered 9.0 mm. Governing clearance is on the FRONT face: the lower tweeter faces forward (stock LX521.4 arrangement), so its D104 faceplate shares the front plane with the 10F's D97.5 flange -- axis spacing 102.84 mm vs 100.75 mm contact leaves a 2.1 mm edge gap (drawing spacing allows an 11.1 mm drop max). Scallop-to-flange 14.1 mm; scallop-to-D82 web 21.9 mm. Total height 459.3 mm. Below the y=306 seam identical to A. |
| `src/lx521_baffle/proud/b1.py` | B1: flank is ONE straight line from the crescent horn corner (36.8, 432.9) through the max-width point (83.8, 399.6) to the V-waist at (+/-56.12, 306.5) -- extended to the horn so the top magnet site lands in the B1 wing |
| `src/lx521_baffle/proud/b2.py` | B2: constant wall around the 10F -- flare and chamfer keep the LM tilts but are both tangent to the r=50.83 circle about the UM center (9.8 mm wall at the D82, 2.1 mm to the D97.5 flange at both tangential points). Chamfer runs from the flare corner (+/-60.65, 391.71; max width 121.3 mm) to the crescent's D102.11 arc extended to (+/-10.08, 418.18); waist (+/-38.1, 315.95). |
| `src/lx521_baffle/proud/b2_split.py` | 4-piece print split of variant B2 (the universal **R6P proud-family** base set) with seven regular through-thickness dovetails |
| `src/lx521_baffle/proud/vase_tebm35c10_4.py` | Parameterized Stock/Slim alternative vase for two opposed Tectonic TEBM35C10-4 BMRs: front/rear mounting, eight M2 insert bores, four captive side magnets, blind pocket walls, independent cable branches, smooth rear growth, and the regular seam-B female dovetails |
| `src/lx521_baffle/cables.py` | Proud-family **R6P** subtractive routing and routing-profile dispatch: standard B2/V1 UM tail plus the keyed V1L-only 283-degree alternate |
| `src/lx521_baffle/um_fit.py` | 283-degree MU terminal service model: terminal-less MU body, hash-pinned W22 reference and declared-placement conservative rear keepout, independent low-profile flag-Faston pull states, physical OD8/OD4 Y-breakout harness, and the proud/V1L split strain reliefs; `PHYSICAL_MEASURE_REQUIRED` remains true |

## Key dimensions (from the drawing, verified against printed dims)

- Outline: bottom 152.4 → ±152.4 @ y≈256.1 → neck 114.3 (y 306–409) →
  121.84 across the tweeter prongs; top scallop cut from Ø78.50,
  corner arcs from Ø102.11 (both centered ≈ (0, 483.05) = rear tweeter axis).
- Lower-mid cutout Ø190 @ (0, 200.98); upper-mid Ø82 @ (0, 366.08) — the
  drawing had it at 371.94; all variants now align the UM (and tweeter
  section, and the perimeter above the neck) to the stock LX521.4 baffle
  (`lx521 baffle metric.dxf`, UM at 368.3 with LM at 203.2, LM-aligned).
- 4 bridge mounting points @ (±20.0, 20.0 / 70.0) — measured on the
  actual bridge (40.0 × 50.0 pattern; the V2 plano's positions were
  wrong). BRASS HEAT-SET inserts identical to the W22/LM (unchanged 6.8 mm
  total bore: Ø6.5 for the first 2.0 mm, then Ø6.4; M5 × 5.8 × Ø6.3),
  but bored BLIND from the REAR face (opposite
  the front-mounted driver inserts): the stock bridge screws in from
  behind with M5 machine screws. no-stand only. Front face stays solid
  (no through-hole, no countersink).
- 2 corner holes Ø4.5 @ (±66.2, 10.0) — OPTIONAL, disabled by default
  (set CORNER_HOLES_ENABLED = True in src/lx521_baffle/base.py to cut them).
  When enabled: M5 machine screws thread-form through the full 18.3 mm
  (pre-run the screw once to cut the threads).
- Blind driver mounts, front face only:
  - Upper mid (production SEAS MU10RB-SL H1658-04, historically called
    “10F” in this project; four-hole flange on pitch D89.5): 4 x Ø4.6
    bores at 58/148/238/328 deg (a square clocked
    +13 deg from 45 -- 45/90 grids are geometrically impossible, see
    VARIANTS.md), 4.0 mm deep, for BRASS HEAT-SET inserts M3 x 3 long
    x Ø5 OD (soldering-iron set). The ring sits 3.75 from the D82 cutout
    wall; the slim bore keeps 1.45 mm on its inboard side. The shared TS duct
    (z=11.5) clears the rotated pattern IN PLAN (>=6.8 to every bore;
    pilot floor z=14.3).
  - Lower mid (production SEAS U22REX/P-SL H1659-08; the hash-pinned
    W22EX001 shrinkwrap supplies the reference mounting template): 6 x
    D5.0 flange holes with D8.8 head recesses on pitch D209.5 and 6 x
    unchanged 6.8 mm-total bores with a Ø6.5 × 2.0 entry followed by Ø6.4,
    clocked at 0/60/120/180/240/300 degrees (shared with Obi-Wan), for BRASS
    HEAT-SET inserts M5 x 5.8 long x Ø6.3 OD. The structural screen assumes
    600 N pull-out per correctly installed insert; it does not qualify the
    actual print process, creep, reuse cycles or unlimited removal/refit.
    M5 screws pass the D5.0 flange holes natively and seat in the D8.8 recesses.
    Floor z=11.5; the ring is plan-clear of every front-half duct. The shared
    clock puts two inserts in `piece_bottom`, leaves no bore on the narrow
    +Y/seam-C land, and keeps the LM terminal axis vertical.
- 2 tweeter-clamp holes @ (±32.56, 451.24) — drawing Ø4.0, printed Ø4.4
  for M4 clearance. The face-to-face ND25FW-4 pair bolts through these,
  sandwiching the baffle crescent between the two faceplates; the pair's
  upper two holes are joined by standoffs above the baffle.
- CRESCENT REAR TAPER: the horseshoe that carries the tweeter pair
  thins from the REAR (the front face stays a full plane): 18.3 at the
  bottom of the scallop, 4.0 at the clamp pass-throughs, feathering to
  ~0.4 at the horn tips. Thickness follows the arc angle about the
  scallop center through two C1 smoothstep segments (zero slope at the
  bottom blend AND across the clamp ring, so the rear faceplate gets a
  locally flat 4 mm seat); cut as a loft of radial sections. Each
  section holds full cut depth from r=36 out to a knee r=51.5 (covering
  the D102.11 arc joint at r≈51.05), then SMOOTHSTEP-FADES the cut back
  to 0 by r=62 (just inside the flank's top corner at r≈62.4). The fade
  carries the SAME taper across the arc joint
  into the crescent's outboard neighbours — the A-comp TOP SHOULDERS
  and B1 WINGS — so when magnetically installed, their rear faces are FLUSH
  with the tapered crescent (no proud step), then ramp back to full
  18.3 depth before their outboard vertical flank/top edges (which stay
  full for the silhouette) and before the crest (y=391.71, where the
  top shoulder meets the full-depth bottom shoulder). Beside the horn
  tips, where the crescent feathers to ~0.4 mm, the shoulder feathers
  with it; the chamfer/flare walls at larger r keep full depth.
  Consequences handled: the T duct tails keep a >=1.3 mm floor where
  the taper starts (cut only ~0.5 deep there); the upper magnet site
  sits where the taper, the T ducts, and the shoulder's chamfer mating
  face balance (see the magnet section); the bottom shoulders are
  untouched (full depth).

## Print split (256×256×256 bed)

- Seam A: y=120 (two ~58 mm lands beside the Ø190 cutout), with four
  through-thickness dovetails at x=±66 and ±103. The inner teeth are
  7/9/3 mm (neck/head/depth); the outer teeth are 6/7/5 mm.
- piece_bottom carries a FUSED stand foot (STAND_FOOT flag in
  src/lx521_baffle/base.py). The foot starts as the baffle's own bottom
  strip (18.3 tall, side faces continuing the flank slopes: ±76.2 at
  the floor widening to ±81.6, sharing the floor plane with the plate
  -- no step), runs 150 mm rearward, and TAPERS in plan continuously
  (one straight line per side, from the strip corners to 38 wide at
  the panel inner face, z=-146). The plate/foot inner corner is a
  plain 90-deg joint (no rib -- printed front-face-down the joint is
  continuous perimeter walls, but it still requires the solid-infill
  modifier and proof procedure in PRINTING.md). The dressed baffle's CG
  sits ~52 mm behind the front face, so it stands upright with no
  front toe.
  The foot's far end carries a minimal 38 × 44 × 4 panel wall for a
  Neutrik NL8MPXX-BAG speakON: Ø31 cutout centered at (0, 20.5) plus
  4 × Ø3.2 screw pass-throughs on the 29.2 × 29.2 pattern (flange is
  38.7 sq -- 0.35 mm/side overhang past the 38 panel, cosmetic). The
  tongue center is channeled to a 4.0 floor between 2.0-thick side
  rails (interior 34 wide: >=1.75 around the Ø30.5 body, which reaches
  ~z=-113); the channel's step face sits at z=-99.
  With the flag ON: the four bridge pass-throughs (and countersinks)
  are omitted, and the cable ducts no longer break the rear face --
  each continues down the plate, drifts to its packed foot lane, takes
  a 90-deg vertical-plane elbow (R14 -- the largest radius that wraps
  the plate/foot inner corner with >=1.4 mm clearance), and runs
  rearward at y=10.5 (LM x=-5.45, UM x=+5.4) / y=5.5 (T1 x=+13.9,
  T2 x=-13.9), exiting through FOUR holes in the channel's step face
  -- ~40 mm of open channel between the cable outs and the connector
  tabs for dressing/Faston access. (Lanes are packed by Δx alone --
  8.45 + 10.85 + 8.5 webs -- because each pair of descent curves
  crosses in the (y,z) plane.) At the driver end, LM uses the shared
  continuous Ø9/R14 rear handoff below the Ø190 opening and TS pierces
  the open tweeter scallop. The proud-family UM
  route instead stays Ø8.2 through one continuous G1 R14 turn and
  leaves the rear face at (33.446, 301.492); there is no separate UM bore
  into the Ø82 opening. With the flag OFF: the original flat piece,
  bridge holes, and the same three rear-face entry datums as Obi-Wan.
  The stock support plate has a Ø20 hole centered at `(0,60)`, with the
  upper edge tangent to the upper screw line at y=70. LM is at
  `(-0.35,64.76)`, the one shared Ø6 tweeter trunk carrying both AWG24
  pairs is at `(-4.75,55.91)`, and UM is at `(3.17,55.91)`. Relative to
  the four rear bridge inserts at `(±20,20)/(±20,70)`, Stock, Slim and
  Obi-Wan therefore present one identical LM-above/T-lower-left/UM-lower-right
  cable interface. The port rim is at least 0.72 mm and every neighbouring
  lumen pair retains at least 0.80 mm wall. Fish all three with the plate off
  the support.
  Print orientation: plate flat FRONT FACE DOWN (rotate 180 deg about X;
  foot rises as an 18.3-thick wall, the panel just widens its top --
  no supports: the step face looks upward and the NL8 holes print as
  vertical-axis circles). Standing the part on its foot is not a released
  orientation because it would give the acoustic front a different texture.
- Seam B: y=315.95, exactly through B2's waist kinks, with two regular
  dovetails at x=-19 and +29 (neck 10 / head 14 / depth 6). Both pieces get
  OBTUSE corners at
  this seam (top foot ≈107° against the flare, mids ≈152° against the
  chamfer) — no brittle knife-tips — and the glue line hides in the crease.
  One additional radial M3×20 socket-cap screw runs on `(x,z)=(0,12.55)`
  from a fully recessed Ø6.2×3.4 head pocket in the LM cutout wall, through
  `mid_right`, into a blind Ø4.6×4.0 M3 heat-set receiver in the vase. The
  screw is serviceable before installing the W22 and completely hidden after.
- Seam C: x=-5.6 between A and B (~20 mm land above the cutout), with one
  regular dovetail at y=305.0 (neck 7 / head 8.5 / depth 4). The offset keeps
  its clearance-grown pocket clear of the 90-degree W22 insert bore.
- All seven dovetails run through the complete local part thickness. Female
  sides use the shared 0.05 mm mitred in-plan clearance. Qualify the fit on
  coupons 1 and 2 before printing a complete four-piece set.

These seams and the foot/bridge behavior apply only to the R6P proud
family. Obi-Wan R6F is not a thinned four-piece shell: its mandatory core
print set is the two collars described below. In floor state the complete
stand is fused into the LM carrier itself; there is no separate
`lx521_top_obiwan_addon_mount_floor_support` artifact. The no-floor bridge
remains fused into the LM core exactly as before.

Filament choice, slicer profile, fastener torques, insert installation, and the
numbered coupon procedure are in [`PRINTING.md`](PRINTING.md).

## Printable pieces

| STL in `build/<state>/stl/` | Footprint (mm) | Used by |
|---|---|---|
| stock_1_of_4_bottom | ~223.8 × 125.0 in plan; 168.3 tall with fused floor stand, 18.3 without | R6P proud family |
| stock_2_of_4_mid_left | ~146.7 × 201.9 | R6P proud family |
| stock_3_of_4_mid_right | ~162.0 × 201.9 | R6P proud family |
| stock_4_of_4_vase_b2 | ~121.3 × 137.4 | R6P proud family |
| stock_shoulder_1..2_of_4_top_l/r | 50.6 × 61.8 | A-comp only |
| stock_shoulder_3..4_of_4_bottom_l/r | 22.5 × 85.9 | A-comp only |
| stock_wing_1..2_of_2_l/r | 73.7 × 125.1 | B1 only |
| `stock_um_grommet_half_{a,b}.stl` | split TPU insert with short curved shank | ordinary B2/V1 R14-bore strain relief; not V1L |

Building the variants: B2 = the four base pieces. A-comp = B2 + the four
shoulder pieces. B1 = B2 + the two wings. Captive magnets retain the
attachments against piece_top_b2's flanks. The magnet interface surfaces meet
flush with zero physical air gap; the receiver's 0.05 mm allowance is a solid
spacing standoff behind that interface. Outline kinks, the notch corner, and
the crescent arc register the pieces. Magnets receive no structural-load
credit. The A bottom shoulders and B1 wings extend below
seam B and register against the mids.

## Cable routing (R6P proud)

Routing is now deliberately split into two physically incompatible
profiles. Generate and review both sheets; the generic routing image no
longer exists:

- `baffle_cable_routing_proud.png` documents **R6P**. It shows the
  normal B2/V1 UM path and, on the same sheet, the clearly
  labeled V1L-only 283-degree alternate tail.

Every proud-family LM route now finishes with one analytic R14 bend through
the established mouth at **(−10.5, 95.981)**. The constrained planar spline,
circular bend, and external tangent lead form one continuous cutter rather
than intersecting horizontal and rear-normal bores. Its last 10 mm flares
monotonically from Ø8.2 to Ø9 for the estimated Ø7.8 cable. Stock and Slim
use the same mouth; their different rear depths merely intersect the R14 at
different tangent angles.

R6P keeps the UM cable space **Ø8.2 end-to-end**. For B2 and V1, the
planar main follows the outer U22 arc, returns through the broad lower
neck, and joins an analytic R14 three-dimensional quarter-turn with
constrained G1 tangency. The same sweep reaches a vertical rear tangent
and leaves the rear face at **(33.446, 301.492)**; it is not assembled
from intersecting planar and vertical cylinders. For the estimated Ø7
cable this provides 0.6 mm nominal radial slack. Its 297.376-degree
bearing remains between mounting screws 238 and 328 degrees but is
14.376 degrees away from the 283-degree Faston pull axis. The
conservative D7 rear continuation and profile-fitted curved grommet are
collision-checked against the full 32 × 40 × 10 mm outboard service envelope.

The conservative W22 keepout records the placement of the hash-pinned
manufacturer reference shrinkwrap `E0022_W22EX001.stp`, SHA-256
`7fc2be551c86006e11c32a570b046772987cb86dcf65350f77c6e34709aa5ab6`.
Its declared native-to-world transform rotates +90° about X (native +Y
to world +Z and native +Z to world -Y), then translates by
`(0, 200.981, -47.498931)`. Native bounds
`(-110.5,-37,-110.5)..(110.5,65.798931,110.5)` therefore map to world
`(-110.5,90.481,-84.498931)..(110.5,311.481,18.3)`, with the LM centre at
`(0,200.981)` and native max-Y on the front datum z=18.3. These are cached,
hash-bound placement facts for the conservative proxy, not a runtime proof
that it contains every surface of the STEP or the installed U22. The
physical U22 and service harness still require the recorded fit check.

The proud-family route set is:

| Driver | Cable | Duct | R6P route |
|---|---|---|---|
| LM (U22/W22) | 2 × 2.5 mm² twisted | Ø8.2, flaring to Ø9 over the final 10 mm | z=12.55 main through the lower insert sector, then one continuous G1 R14 bend through the retained rear mouth below the Ø190 opening and an external tangent lead |
| UM (MU10/10F) | estimated Ø7 twisted pair | Ø8.2 | B2 and V1: outer U22 arc, single-curvature W22-pilot bypass, continuous G1 R14 handoff to (33.446, 301.492). V1L only: its own single-curvature guide reaches the keyed 283-degree rear-face aperture Q=(13.497063, 307.618796, 6.8) in `piece_mid_right`; both tails are regression-gated against heading reversal |
| T1+T2 (both ND25) | 2 × (2 × AWG24) | shared Ø6.0, flattened to W6.6 × H4.4 under the MU seat | strip feeders merge through the Ø6.8 step, rise along the left vase flank, and make one head-on scallop pierce near (−3.3, 430) |

The route suites sample the complete physical centerlines, including the
standard proud R14 handoff, the V1L alternate tail and rear-face handoff,
every R6F covered Z bump, physical crown crossing, printed-to-free owner
handoffs, independent LM lead, free rear UM/tweeter spans, and the R6F
cable's review-only G1 R20 turn to its Y breakout. Two separate Ø3.2 conductors then
retain R8-minimum slack paths into non-overlapping low-profile flag-Faston
boots and one-at-a-time 0/3/6/9/12 mm pull states. The printed terminal
approach is R15 and the exact G1 free continuation is R20; Ø3.2/R8 remains
provisional until the physical lead and manufacturer bend requirement are
measured.
For R6P, `test_um_eroded_outline_containment` erodes the exact outline by
duct radius plus the 1.6 mm proud-family skin, tests the complete
interpolated route `LineString` for containment (not just sampled
vertices), and reports true normal distance to the boundary. R6F instead
uses its state-specific 0.8 mm wall/0.85 mm roof checks plus final assembled
BREP shell subtraction and an independent 0.76..0.90 mm manufactured-BREP
normal-wall bracket. The former horizontal-gap approximation is not used
in either family.
Pilot and duct-pair checks retain the 1.5 mm separation rule. R6P ducts
cross its glue seams, so fish a cable or pull string through each open
segment during assembly. R6F UM/T cables must be dry-fished through their
buried owner segments and rehearsed across their free rear spans, while the
LM cable must be rehearsed over its free rear span and, in floor state,
through the integral stem continuation before driver installation.

## Tweeter options

The default arrangement is the face-to-face ND25FW-4 pair: two Dayton
ND25FW-4 dome tweeters with waveguide, bolted through the clamp holes at
`(±32.56, 451.24)` so their faceplates sandwich the B2 crescent. It is
integral to the vase piece and needs no separate part.

The alternative replaces the crescent with the opposed TEBM35C10-4 BMR vase:
two Tectonic TEBM35C10-4 BMRs, the lower facing front and the upper facing
rear, released in a Stock envelope profile. Build it with
`make vase_tebm35c10_4_cad` and take it from `build/vase_TEBM35C10-4/stock/`;
see [`VARIANTS.md`](VARIANTS.md#opposed-tebm35c10-4-bmr-vase-alternative).

## Magnet attachment (swappable shoulders/wings)

Attachments mount with neodymium N52 D5 x 2 disc magnets (superimanes
ref D-05-02-N52; supplier figure 0.68 kg/pair; 12 needed + spares) so B2 <-> A-comp <->
B1 are interchangeable without magnet adhesive. TWO sites per flank side
(4 magnets in the base total). Every released base and receiver uses the same
pause-and-bury captive cavity derived from
`coupons/obiwan_ae_embed/obiwan_ae_embed_coupon.py`: actual magnet D5.0 × 2.0,
internal cavity Ø5.20 × 2.10, 0.45 mm plastic skin at each axial face, an
upward-open printable cradle during insertion, and a self-supporting 45°
closing roof. The finished magnet is completely buried, has no glue and has
no external access opening. All paired transverse stations in stock, slim,
and Obi-Wan use the same front-biased source Z=**15.10 mm**. Their cavity
booleans may remove only internal material: the magnet-free exterior is
immutable, and no station may add a local backing, boss, relief, rear cap,
flat, or other visible cue. The supplier's 0.68 kg/pair figure is not proof of
achieved pull through the production spacing; qualify retention with a
physical pull test. The outline kinks/corners and saddles provide
registration; magnets receive no shear or structural-load credit.

The attachment surfaces themselves meet flush, so the physical air gap is
**0 mm**. The receiver's 0.05 mm allowance is solid material between that
interface and its 0.45 mm retaining skin, not an air-gap cutter. At the
stock lower straight station, the two skins plus that solid standoff give
**0.95 mm** nominal magnet-face separation. At the rounded stock upper
station, the base cavity datum is recessed 0.14 mm into the existing host—just
beyond the true arc's 0.134666 mm maximum tangent deviation—so its separation
is **1.09 mm** (`0.45 + 0.14 + 0.05 + 0.45`). Neither adaptation changes the
exterior. The thin V1/V1L host instead uses one broad, symmetric, smooth
rear-taper shelf through the upper band; it is not a magnet-shaped pad or
backfill. Obi-Wan's ring and lower-shoulder datums remain 0.15 mm beneath
their smooth surfaces, giving **1.10 mm** at LM-lower, LM-upper, and UM.

This replaces every old face-flush/adhesive-pocket and mating-air-gap
assumption. Magnet axes and polarity are unchanged. Neo stacks ship uniformly oriented; sharpie-mark one
pole before separating the stack and follow the site-by-site polarity table in
[`CAPTIVE_MAGNET_PAUSE_MANIFEST.md`](../review/captive_magnet_slice_audit/CAPTIVE_MAGNET_PAUSE_MANIFEST.md), including mirrored
parts. Never infer polarity from left/right appearance:

| Site (right; left mirrored) | Wall | Serves | Placement rationale |
|---|---|---|---|
| (40.0, 322.4, 15.10) | flare, waist-kink end | A bottom shoulder, B1 wing lower end | the flank's farthest point from the UM driver (59.2 mm); the complete captive land and T-duct clearance fit inside the unchanged host |
| (17.88, 420.37, 15.10) | crescent arc, theta=-69.5 deg | A top shoulder, B1 wing top end | as far down-arc as the receiver permits; `make check` verifies the complete internal land against the chamfer, smooth taper shelf, front face, and TS duct without exterior growth |

For the legacy R6P B2/A/B1 arrangement specifically, magnet count per baffle
is 4 base + 4 per attachment set (12 with both sets; 24 for a stereo pair).
Obi-Wan, its keyed alternative, flat/graded, and the calibration coupon use their own
per-part counts in the authoritative pause manifest; do not apply this R6P
count to them.

Polarity discipline: use the "MARCADO NORTE" batch and the manifest's local
axis convention; verify every magnet against a marked master before its
insertion pause. Once the roof is printed the polarity cannot be corrected.
Insertion direction is independent of polarity: every released site must be
loaded vertically downward from above the paused part (its +Z side) along
print -Z, exactly `print_insertion_direction_xyz = [0, 0, -1]`. The catalog
consumer fails closed if the front-face-down transform produces any other
loading direction.

The other inventory magnets are not suitable here:
D18 exceeds the 18.3 mm wall, adhesive tape magnets are too weak for a
structural joint, D10x5 only fits the receiver side.

Retired concept-only drawings, historical diagnostic renders, and non-release
fit coupons are excluded from the released part migration because they are not
printable release outputs. The
`coupons/obiwan_ae_embed/` coupon is not an assembly part; it remains the
physical reference implementation and regression evidence. The retired V0
variant's hypothetical scarf mate is likewise not a release output.

## Assembly

**R6P:** dry-fit and tune coupons 1 and 2 before gluing. Install the seam-B
M3 heat-set in the vase first. Assemble seam C, then A, then B on a flat
front-face datum; fish each cable segment as its seam closes. Drive the
M3×20 socket-cap screw radially from the LM opening and verify it seats below
the Ø6.2 access mouth without bottoming. Set the driver and rear bridge inserts square, mount
the drivers at the low torques in PRINTING.md, and re-torque after the
first preload-settling interval. For V1L, dry-fish the actual keyed
`mid_right` alternate outlet before glue-up; confirm that its aperture
is centered at the 283-degree rear-face witness and physically rehearse
the real Fastons, boots, service loop, dedicated split TPU grommet, and
measured withdrawal before installing the MU.
