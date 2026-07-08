# LX521.4 top baffle — ND25FW-4 face-to-face mod (V2)

3D-printable version of the modified top baffle from
`plano top baffle con anidados V2.pdf` (exact 1:1 vector geometry extracted
from the PDF, not redrawn). Overall 304.8 × 468.31 × 18.3 mm.

## Files

| File | What |
|---|---|
| `top_baffle_nd25fw4.py` | Geometry library (drawing outline, holes, pilots); its own gen_step is the un-compromised aligned drawing (no artifacts kept) |
| `top_baffle_nd25fw4_a_comp.py` | Variant A-comp: straight-sided tower — vertical flanks at ±60.65 (tangent to B2's flare crest) from the extended top edge down to the LM chamfer-extension; tweeter section at the B2 drop. Buildable as B2 pieces + 4 shoulder pieces |
| `top_baffle_nd25fw4_attachments.py` / `.step` | The 6 attachment pieces (exact boolean complements): 2+2 A-comp shoulders (top/bottom per side, split at the crest tangent), 2 B1 wings |
| `top_baffle_nd25fw4_b.py` | Shared B-family builder: mini-LM upper-mid vase (no shelf corners) + tweeter section lowered 9.0 mm. Governing clearance is on the FRONT face: the lower tweeter faces forward (stock LX521.4 arrangement), so its D104 faceplate shares the front plane with the 10F's D97.5 flange -- axis spacing 102.84 mm vs 100.75 mm contact leaves a 2.1 mm edge gap (drawing spacing allows an 11.1 mm drop max). Scallop-to-flange 14.1 mm; scallop-to-D82 web 21.9 mm. Total height 459.3 mm. Below the y=306 seam identical to A. |
| `top_baffle_nd25fw4_b1.py` | B1: flank is ONE straight line from the crescent horn corner (36.8, 432.9) through the max-width point (83.8, 399.6) to the V-waist at (+/-56.12, 306.5) -- extended to the horn so the top magnet site lands in the B1 wing |
| `top_baffle_nd25fw4_b2.py` | B2: constant wall around the 10F -- flare and chamfer keep the LM tilts but are both tangent to the r=50.83 circle about the UM center (9.8 mm wall at the D82, 2.1 mm to the D97.5 flange at both tangential points). Chamfer runs from the flare corner (+/-60.65, 391.71; max width 121.3 mm) to the crescent's D102.11 arc extended to (+/-10.08, 418.18); waist (+/-38.1, 315.95). |
| `top_baffle_nd25fw4_b2_split.py` | 4-piece print split of variant B2 (the universal base set), shown assembled |
| `top_baffle_nd25fw4_c7.py` | Variant C7: B2 with the LM section rear-tapered to a ~0.5 knife edge over the last 19 mm inside the flank/chamfer outline (front face stays a full plane). Full-depth land kept at the bottom strip (foot/bridge) and before seam B; half-round ribs (r=5.4) carry the T ducts across the band. See "Variant C7" below |
| `top_baffle_nd25fw4_c7_split.py` | C7 print split: same seams/dovetails/ducts as B2 -- the three LM pieces are drop-in replacements, piece_top and all attachments are shared |
| `export_piece_stls.py` | Exports the print-ready piece STLs (`--outdir`) |
| `export_steps.py` | Exports a module's `gen_step()` to STEP via build123d's native exporter (`<module.py> --output <path>`) — no CAD-skill dependency |
| `Makefile` | `make -j8` generates STEPs/STLs/PNGs for BOTH stand-foot states into `floor_stand/` and `no_floor_stand/` (see "Generated artifact layout") |
| `<variant>/stl/lx521_top_*.stl` | Print-ready pieces (flat, Z = thickness, front face up): 4 base + 4 addon-A + 2 addon-B1 |

Regenerate everything: `make -j8 PYTHON=<venv>/bin/python`. Self-contained
— the only dependencies are the pip packages `build123d`, `shapely`,
`matplotlib`, and `numpy` (no external CAD tooling).
`make check` runs the analytic clearance suite (`test_clearances.py`):
duct-duct and duct-pilot separations, foot-lane webs, magnet-pocket
walls, and the variant-outline splice assertions.

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
  wrong). BRASS HEAT-SET inserts identical to the W22/LM (bore Ø6.4 ×
  6.8, M5 × 5.8 × Ø6.3), but bored BLIND from the REAR face (opposite
  the front-mounted driver inserts): the stock bridge screws in from
  behind with M5 machine screws. no-stand only. Front face stays solid
  (no through-hole, no countersink).
- 2 corner holes Ø4.5 @ (±66.2, 10.0) — OPTIONAL, disabled by default
  (set CORNER_HOLES_ENABLED = True in top_baffle_nd25fw4.py to cut them).
  When enabled: M5 machine screws thread-form through the full 18.3 mm
  (pre-run the screw once to cut the threads).
- Blind driver mounts, front face only:
  - Upper mid (Scan-Speak 10F/8424G00, 4 x D3.8 flange holes on pitch
    D89.5): 4 x Ø4.6 bores at 58/148/238/328 deg (a square clocked
    +13 deg from 45 -- 45/90 grids are geometrically impossible, see
    VARIANTS.md), 4.0 mm deep, for BRASS HEAT-SET inserts M3 x 3 long
    x Ø5 OD (soldering-iron set; M3 screws pass the D3.8 flange holes
    natively). The ring sits 3.75 from the D82 cutout wall; the slim
    bore keeps 1.45 mm on its inboard side. The shared TS duct
    (z=11.5) clears the rotated pattern IN PLAN (>=6.8 to every bore;
    pilot floor z=14.3).
  - Lower mid (SEAS W22EX001, 6 x D5.0 flange holes + D8.8 head recess on
    pitch D209.5, measured from E0022_W22EX001.stp): 6 x Ø6.4 bores,
    6.8 mm deep, aligned VERTICALLY (30/90/...330 deg), for BRASS
    HEAT-SET inserts M5 x 5.8 long x Ø6.3 OD (recommended hole Ø6.4;
    set with a soldering iron; ~700 N pull-out each holds the 2.6 kg
    driver with a wide margin and survives unlimited R&R). M5 screws
    pass the D5.0 flange holes natively and seat in the D8.8 recesses.
    Floor z=11.5; the ring is plan-clear of every front-half duct
    (seam C clears the 90-deg bore by 2.25 mm, the LM duct keeps
    3.05 mm to the 270-deg bore).
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
  and B1 WINGS — so when they are glued on, their rear faces are FLUSH
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

- Seam A: y=120 (two ~58 mm lands beside the Ø190 cutout), 2 dovetails
  (±89, neck 7 / head 9 / depth 5) — in the full-depth window between
  the T duct arc (crosses seam A at x≈72.6; UM at x≈66) and the C7
  taper boundary (x≈92): keys clear of every duct and fully OUT of the
  C7 taper in all variants.
- piece_bottom carries a FUSED stand foot (STAND_FOOT flag in
  top_baffle_nd25fw4.py). The foot starts as the baffle's own bottom
  strip (18.3 tall, side faces continuing the flank slopes: ±76.2 at
  the floor widening to ±81.6, sharing the floor plane with the plate
  -- no step), runs 150 mm rearward, and TAPERS in plan continuously
  (one straight line per side, from the strip corners to 38 wide at
  the panel inner face, z=-146). The plate/foot inner corner is a
  plain 90-deg joint (no rib -- printed front-face-down the joint is
  continuous perimeter walls, plenty strong). The dressed baffle's CG
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
  crosses in the (y,z) plane.) The driver-side exit bores into the
  Ø190/Ø82 cutouts are unchanged. With the flag OFF: the original flat
  piece, bridge holes, and rear-face breakouts aimed at the SUPPORT
  WINDOW -- the stock support plate has a Ø20 hole (center (0, 60):
  horizontally centered, top edge tangent to the upper screw line
  y=70) that all four cables must pass. Packing: LM/UM breakouts side
  by side (steep ramps crossing z=0 at (∓5.2, 60.5), tips lancing
  their mains at (−8, 68.5, 12.55) / (+8, 60, 12.55)); twin Ø4.6 T
  ramps at the window's lower edge, breakouts (+3.8, 52.2) /
  (−3.1, 52.7) with far lips up to ~1.4 past the rim (the floppy
  AWG24 pairs duck in), lancing the strip feeders (t1f z=3.7 /
  t2f z=9.5) that merge into the Ø6.8 z-step west of the LM column.
  Fish the LM/UM steep ramps with the plate off the support.
  Print orientation: plate flat FRONT FACE DOWN (rotate 180 deg about X;
  foot rises as an 18.3-thick wall, the panel just widens its top --
  no supports: the step face looks upward and the NL8 holes print as
  vertical-axis circles) or standing on the foot for the strongest
  joint.
- Seam B: y=315.95, exactly through B2's waist kinks, 2 dovetails
  (left -19, right +21.5; neck 10 / head 14 / depth 6 — the right one
  outboard so the UM corridor passes at x=8.3 and the T elbow at x=33). Both pieces get OBTUSE corners at
  this seam (top foot ≈107° against the flare, mids ≈152° against the
  chamfer) — no brittle knife-tips — and the glue line hides in the crease.
- Seam C: x=-5.6 between A and B (~20 mm land above the cutout; offset
  left so its dovetail pocket clears the 90-deg W22 insert bore by
  1.55 mm), 1 dovetail (neck 6 / head 8 / depth 4 at y=300.5).
- Dovetails are through-thickness, 0.10 mm clearance on female sides.

### Generated artifact layout

`make -j8` (see the Makefile; `PYTHON=<venv>/bin/python` to pick an
interpreter) builds BOTH stand-foot states in parallel:

    floor_stand/      LX_STAND_FOOT=1: fused foot + NL8 panel, no
      stl/  *.step  *.png     bridge holes, cables through the foot
    no_floor_stand/   LX_STAND_FOOT=0: flat piece_bottom, bridge
      stl/  *.step  *.png     pass-throughs, rear-face cable breakouts

Each folder is a complete print set. piece_bottom is the only
functionally different piece; the other base STLs differ between the
folders by <0.05 mm of duct-wall position (the stand-foot entry knots
shift the shared duct splines microscopically — well under the 0.10
seam clearance, but the files are not byte-identical).
`attachments.step` is flag-independent and stays at the top level. The
STAND_FOOT flag is the `LX_STAND_FOOT` env var (default 1).

| STL in `<variant>/stl/` | Footprint (mm) | Used by |
|---|---|---|
| lx521_top_base_1of4_bottom | 223.8 × 125.0, 168.3 tall (fused stand foot; flip front-face-down or stand on the foot — see above) | all variants |
| lx521_top_base_2of4_mid_left | 146.7 × 201.9 | all variants |
| lx521_top_base_3of4_mid_right | 162.0 × 201.9 | all variants |
| lx521_top_base_4of4_vase_b2 | 121.3 × 137.4 | all variants |
| lx521_top_addonA_1..2of4_shoulder_top_l/r | 50.6 × 61.8 | A-comp only |
| lx521_top_addonA_3..4of4_shoulder_bottom_l/r | 22.5 × 85.9 | A-comp only |
| lx521_top_addonB1_1..2of2_wing_l/r | 73.7 × 125.1 | B1 only |
| lx521_top_c7base_1of4_bottom | as base 1of4 | C7 (LM knife taper) |
| lx521_top_c7base_2..3of4_mid_l/r | as base 2/3of4 | C7 (LM knife taper) |
| lx521_top_c7base_4of4_vase_b2 | same part as base 4of4 (re-tessellated file) | C7 = same vase |

Building the variants: B2 = the four base pieces. A-comp = B2 + the four
shoulder pieces. B1 = B2 + the two wings. Attachments are edge-glued onto
piece_top_b2's flanks (zero designed clearance); the kinks, notch corner,
and crescent arc on their inner faces self-register them. The A bottom
shoulders and the B1 wings extend below seam B (bonding ~9-12 mm onto the
mids), so they also splint the top-to-mid glue line.

## Internal cable ducts

FOUR fully internal spline pipes: LM and UM are big mid-plane bores
(z=9.15) sized for TWISTED pairs; T1/T2 run deeper at z=3.7 (they pass
under the 10F pilot ring -- see the pilot note above) -- each tweeter of
the face-to-face pair carries its own AWG24 pair. No duct intersects any
hole or pocket (blind included) in plan.

The in-plane MAIN of each duct (the routing in the table below) is common
to both variants; only the ENTRY and the driver-side EXIT differ:
- Entries depend on STAND_FOOT (see the Print-split section). Flag ON:
  each duct drops down the plate, takes a 90-deg R14 elbow, and runs
  through the foot to exit the connector-channel step face. Flag OFF:
  four oblique bores break the REAR face, packed into the support
  plate's Ø20 window (LM/UM inside it, T1/T2 at its lower edge).
- Exits are common to both: oblique bores into the driver-cutout walls,
  invisible with drivers mounted:

| Driver | Cable | Duct | Route |
|---|---|---|---|
| LM (W22) | 2x 2.5 mm^2 twisted | D8.2 | planar z=12.55, drifting past the 270-deg insert bore (plan-clear 3.0); near-level exit bore -> D190-rim opening at z~12 |
| UM (10F) | twisted 2x2.0 mm^2 pair (~O7.0; 2x2.5 no longer fits) | D7.8 | planar z=12.55 END-TO-END: R26 fan fillet, ONE arc r=119.5 OUTSIDE the W22 pilot ring, R50 fillet onto a straight diagonal tangent to the 30-deg pilot keep-out, one R~10 window bend, vase tail; exit into the D82 rim at z~12; ONE routing shared by ALL piece variants |
| T1+T2 (both ND25) | 2x (2x AWG24) | D6.0 shared ("ts") | planar z=11.5 up the LEFT flank: strip feeders (t1f z=3.7 from the right, t2f z=9.5) merge in a Ø6.8 z-step, tangent line onto the r=114 arc outside the pilot ring, left vase flank lane 5.1 inside the walls, crest transition, notch-corridor dive (the largest bore that corridor admits), SINGLE head-on pierce of the D78.5 scallop rim at ~(−3.3, 430); both pairs dress to their tweeters through the open scallop void |

Min bend (enforced by test_route_smoothness on the real splines):
LM >= 25, UM >= 10 (the window bend between the 90-deg pilot and the
right B-key), TS >= 4.5 (the crest transition), feeders >= 6.
Min walls: >=1.6 mm skins everywhere (the TS lane runs 5.1 inside the
vase walls; the notch-corridor dive keeps >=1.6 to the D82 rim and the
chamfer edge -- the corridor is consumed exactly). Verified centerline
separations (`make check` re-measures them): every duct pair >= its
two radii + 1.5 mm; every W22 AND 10F pilot >= bore radius + duct
radius + 1.5 mm in plan. The ducts cross the glue seams -- fish each cable (or a
pull string) through each piece's short open segment during assembly.
Seam-A dovetails sit at +/-89 (n7/h9/d5, full-depth, clear of both arcs);
seam-C dovetail at (300.5, n6/h8/d4).

## Variant C7 — LM knife-edge taper

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
  full-depth land, flush joint to the shared vase piece, dovetails at
  full section). Seam-A dovetails (±97) sit inboard of the band --
  full depth.
- The common cable routing (see the duct table) is shared by ALL
  variants, so B2 and C7 pieces mix freely across the seams. ALL four
  ducts stay buried in the full-depth core (UM at r=100.7 and the T
  lower mains at r=110, both deep at z=5.7 under the heat-set pilot
  ring) -- the tapered rear face carries no ribs or marks. Asserted by
  test_c7_duct_corridor (`make check`) and verified with duct-envelope
  probes on the built piece solids.
- Print: same bed footprints as the B2 pieces; the taper prints
  front-face-down with layers shrinking as they rise (support-free).
- Concept sheet: `gen_lm_knife_draft.py` ->
  `baffle_lm_knife_draft.png`.

## Variant V0 — minimalist UM vase (front slide)

An alternate piece_top for the low-crossover (3-4 kHz) experiments:
a REAR-side knife bevel (same side and philosophy as the C7 LM taper;
front plane fully intact) -- 18.3 -> ~0.5 over the last 2.8 mm inside
the flare/chamfer outline, fading out at the seam-B land and blending
into the crescent's rear taper above y~400. The band is capped at
2.8 mm by the shared O6.0 T duct (z=11.5) hugging the left vase
walls at ~1.6.
ALL duct routing identical; V0 mixes with B2 or C7 bottom/mids
freely. One D5 x 2 FLUSH magnet per side (the SAME magnets as all
attachments) in a vertical rear-face pocket at (+-46, 324); scarf
attachments add receivers + register on the outline kinks. The
B2-family shoulders/wings do NOT fit V0. Guarded by
test_v0_duct_corridor (`make check`); STL: lx521_top_v0_4of4_vase
(--variant v0).

## Variant V1 — 11.5 mm UM vase (minimum-thickness field)

The vase field thinned to t=11.5 -- the absolute practical minimum
with all buried routing kept (binding constraint: the O7.8 UM exit
tail needs z 0.2..11.2; the T lanes alone would allow 7.2). Front-side
cut, REAR plane and ALL ducts untouched; sharp step exactly at seam B
(keys auto-trim to 11.5 on both sides); the WHOLE top is flush at
11.5: the crescent taper re-derives on the 6.8..18.3 slab (same 4.0
clamp seat / 0.4 tips), the tweeter pair clamps an 11.5 septum
(shorter standoffs; pair spacing -6.8, which raises the pair's dipole
peak -- helpful for the 3-4 kHz XO). With the round-4 front-datum
routing the vase thins from the REAR, so V1 mounts FRONT-FLUSH with
the LM section -- no driver misalignment. Pair with V1L for the
complete thin baffle. 10F mounting: 4 x O4.6 x 4.0 bores from
the new front for M3 x 3 x O5 brass heat-sets (floor z=7.5 stays 1.9
above the T-lane roofs at the ring crossings). Two D5 x 2 FLUSH
magnets per side in the flank walls (zc 12.5/14.4); B2 wall
pockets are skipped (B2 attachments do not fit V1). Guarded by
test_v1_field (`make check`); STL: lx521_top_v1_4of4_vase
(--variant v1). Thinner is possible only by externalizing cables to
rear-face grooves (~7) or through-bolting the 10F (~5-6) -- see the
constraint ladder in the V0/V1 discussion.

## Variant V1L — 11.5 mm LM section (front-flush)

The bottom + both mids thinned to t=11.5 (material z 6.8..18.3 above
the foot strip): the ENTIRE baffle then shares one front plane (use
with the V1 vase -- same rear plane, NO step at seam B). Binding
constraint: the O8.2 LM duct window. The bottom strip keeps full 18.3 (smoothstep ramp
y=78 -> 96: full past the top pass-through seats +5, thin 10 short
of the D190 edge) for the fused foot / bridge hardware / cable
feeders; W22 heat-sets unchanged (floor keeps a 4.5 wall). Enabled by
the round-4 "front-datum" routing shared by ALL variants:

* LM O8.2 at z=12.55 (plan unchanged).
* UM O7.8 at z=12.55 END-TO-END on ONE arc r=119.5 OUTSIDE the W22
  pilot ring, then a diagonal threading the 0.9 mm window between the
  90-deg pilot keep-out and the right seam-B key (moved to cx=28).
* T1+T2 SHARE one O6.0 duct ("ts") at z=11.5 up the LEFT flank -- the
  largest bore the notch corridor (D82 rim vs vase chamfer) admits --
  with a SINGLE scallop exit at (-3.3, 430); both pairs dress to their
  tweeters through the open scallop void. Pair feeders (O3.8, t1f
  z=3.7 / t2f z=9.5) cross the full-depth strip under the LM/UM
  columns and merge into a O6.8 z-step west of the LM column. 10F pilot pattern rotated to
  (58/148/238/328) so its left pair clears the lane and dive.
* Seam-A keys at +-63 (full-depth in every variant, clear of both
  crossings); the RIGHT vase flank carries no duct at all.

STLs: lx521_top_v1l_{1of4_bottom,2of4_mid_left,3of4_mid_right}
(--variant v1l) + lx521_top_v1_4of4_vase. Structural note: ~30% of
stock bending stiffness -- measure assembly modes before trusting the
W22 on it.

See VARIANTS.md for the variant/add-on catalog and the
compatibility matrix, and PRINTING.md for filament choice, print settings, fastener
torques, and insert installation.

## Magnet attachment (swappable shoulders/wings)

Attachments mount with neodymium N52 D5 x 2 disc magnets (superimanes
ref D-05-02-N52, 0.68 kg/pair; 12 needed + spares) so B2 <-> A-comp <->
B1 are interchangeable without glue. TWO sites per flank side (4 magnets
in the base total), all FLUSH (a 2.0 mm disc in a 2.0 mm pocket on
BOTH faces, meeting level; the outline kinks/corners self-register --
a flush disc gives no shear key). Pockets: base and receiver both
D5.4 x 2.0. Polarity: neo stacks ship uniformly oriented
-- sharpie-dot the top face of each as you peel; dots face OUT in the
base, IN in the attachments:

| Site (right; left mirrored) | Wall | Serves | Placement rationale |
|---|---|---|---|
| (40.0, 322.4) | flare, waist-kink end | A bottom shoulder, B1 wing lower end | the flank's farthest point from the UM driver (59.2 mm); flush pocket 2.0 deep, ample clearance to the T duct |
| (17.88, 420.37) | crescent arc, theta=-69.5 deg | A top shoulder, B1 wing top end | as far down-arc as the RECEIVER allows: its bore sits in the narrowing wedge between the arc and the chamfer face the shoulder/wing mates against B2, and its bottom corner keeps 1.3 mm to that face; the rear taper leaves ~12.2 mm of wall (bore raised to z=10.7: 1.7 mm behind the pocket floor); ~7.9 mm to the TS duct, 21.7 from the clamp hole, 57.2 from the UM driver (all re-measured by `make check`) |

Magnet count per baffle: 4 base + 4 per attachment set
(12 with both sets; 24 for a stereo pair).

Gluing: epoxy or CA, magnets degreased. Polarity discipline: use the
"MARCADO NORTE" batch on the base with NORTH facing out, and mount all
attachment magnets SOUTH out (check each against a marked one before
gluing). Glue the base magnets first, use them to locate the mating
receivers' magnets. The other inventory magnets are not suitable here:
D18 exceeds the 18.3 mm wall, adhesive tape magnets are too weak for a
structural joint, D10x5 only fits the receiver side.

If you prefer permanent assembly, the same pockets take glue (fill with
epoxy and clamp); the outline kinks register the parts.

## Printing

- Flat on the bed, no supports needed (all holes are vertical through-holes).
- PETG or PLA; 5–6 perimeters, 40–50 % gyroid/cubic infill, 5 top/bottom
  layers — the baffle carries three drivers, err stiff. ~945 cm³ solid
  volume total (≈1.1–1.3 kg at these settings).
- piece_bottom is 250.6 mm wide: orient it square to the bed axes.

## Assembly

1. Dry-fit all seams first; if a dovetail binds, warm-file the male flanks
   (designed clearance is 0.10 mm on the female sides).
2. Glue order: mid_left + mid_right (seam C), then the mid pair onto
   bottom (seam A), then top onto the mids (seam B). Epoxy (30 min) or
   polyurethane glue preferred; CA works on PLA but is brittle in shock.
3. Assemble on a flat surface, front face down, so the seams cure flush.
   Clamp lightly across each seam; check the Ø190 rim stays circular where
   seams A and C cross it (the driver flange will bridge these seams).
4. After cure, optionally lay a bead of epoxy along the rear seam lines and
   the rear rim of the Ø190 cutout as a fillet — the L22 flange clamps the
   front face, so rear-side reinforcement is invisible.
5. Set the brass heat-set inserts with a soldering iron (flush,
   square): six M5 x 5.8 x Ø6.3 in the W22 pilots, four M3 x 3 x Ø5 in the
   10F pilots. Mount the W22EX001 lower-mid with M5 screws and the
   10F/8424G00 upper-mid with M3 screws into the inserts, then bolt the
   ND25FW-4 pair through the (±32.56, 451.24) holes with M4 screws and cut
   the inter-faceplate standoffs to the printed thickness (18.3 mm).
6. Screw to the bridge through the four countersunk holes (5 mm oval-head
   wood screws, from the front).
