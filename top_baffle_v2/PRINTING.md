# Printing the top baffle — PLA+ Tough (Bambu Studio)

Settings and engineering numbers for printing the piece sets in
Bambu PLA Tough+ on a 0.4 mm nozzle, for both `floor_stand/` and
`no_floor_stand/`. Everything here follows from measured heat-set
pull-out data, the piece geometry (front plane flat in every variant;
sculpted/stepped rears), and PLA's one real weakness for this job:
**preload creep**.

## Loads and fasteners

Drivers are the LX521.4 production SEAS customs: "W22" rows = the
U22REX/P-SL (H1659-08), "10F" rows = the MU10RB-SL (H1658-04) — same
cutout/pilot geometry, real flanges O220.6 x 6.0 / O98 x 4.0
(owner-measured; see the V1LF recess note below).

The baffle carries ~3.2 kg of drivers: W22 ~2.6 kg on six inserts,
10F 0.43 kg on four, the ND25 pair ~0.2 kg clamped. Static load per
W22 insert is ~5 N against >=600 N conservative pull-out for an
M5 x 5.8 x O6.3 heat-set in PLA (published tests: 900-1400 N) — retention
margin is >100x everywhere. The governing effect is instead that PLA
relaxes 30-50 % of bolt preload over the first days.

| Fastener | Spec | Torque | Notes |
|---|---|---|---|
| W22 | M5 x 14 pan + flat washer into M5 x 5.8 x O6.3 heat-set (bore O6.4 x 6.8) | 0.8–1.0 N·m | wave washers; re-torque at 24 h and ~2 weeks |
| W22 on **V1LF** | **M5 x 12** pan + flat washer, same inserts (bore O6.4 x **6.2** from the recess seat — sized to the 5.8 insert so the rear pads stay 1.5 low; an M5 x 14 can bottom out before clamping) | 0.8–1.0 N·m | same wave-washer / re-torque routine |
| 10F | M3 x 8 into M3 x 3 x O5 heat-set (bore O4.6 x 4.0) | 0.30–0.40 N·m | short engagement — do not overdrive |
| Bridge (no-stand) | M5 machine screw from the bridge (behind) into M5 x 5.8 x O6.3 heat-set (bore O6.4 x 6.8, REAR face) | hand-snug | 4 off; same insert as the W22, set from the rear |
| Tweeter pair | M4 through-bolts + nyloc + wave washer; length = septum + faceplates (stock 18.3 septum → ~M4 x 35; V1's 11.5 → ~M4 x 30 — verify stacked) | snug, ~0.5 N·m | clamps the 4.0 mm crescent seat; recheck after a week |

**Installing the inserts:** soldering iron at 230–250 °C, press
slowly and square, stop flush. The bores carry +1.0 mm melt room by
design.

**Temperature limit:** PLA+ Tough Tg is ~58–62 °C. Indoors: fine.
Direct summer sun on a dark baffle or a closed car will creep the W22
mounts and the tweeter clamp. If the speakers may see that, print
PETG/ASA instead — decide before committing ~2 kg of filament.

## Orientation — the setting that matters most

**Print every piece front face DOWN.** The front plane is the only
flat face in every variant (C7 / V0 / V1 / V1L sculpt or step the
rear), and for the floor-stand `piece_bottom` (223.8 x 125 x 168.3)
front-down puts the plate on the bed with the 150 mm foot rising as a
self-standing tower. Stock B2 pieces could print rear-down, but use
front-down anyway for a uniform front finish. Smooth/satin PEI gives
a clean front face; textured PEI gives uniform grain — either works.

## Bambu Studio profile (0.4 nozzle)

* **Layer height** 0.20 mm.
* **Walls: 6 loops** (2.4 mm) — makes the material around every
  insert bore, dovetail key, and knife edge fully solid.
  Top/bottom **6/5** layers; *Ensure vertical shell thickness: All*.
* **Infill:** gyroid, **30 %** for the mids and vase, **40 % for
  piece_bottom in both variants** (foot standing moment ~8 N·m, or
  the bridge bolts). *Detect narrow internal solid infill: on.*
* **Strength tuning:** nozzle 225 °C (top-middle of the Tough range —
  hotter = better layer adhesion), bed 55–60 °C, **max fan 60 %**
  (overhang fan 100 %), outer wall <=120 mm/s, keep the filament
  profile's volumetric limit (~12–16 mm³/s). Strength lives in layer
  adhesion, not speed.
* **Dimensional fits** (dovetails run a SNUG 0.05 mm clearance --
  tune hole compensation on the coupon until the key slides firmly;
  insert bores are sized exact): *Precise wall: on*,
  **elephant-foot compensation 0.15 mm**, X-Y hole compensation
  starting at **+0.05**. Print the EIGHT `stl/lx521_coupon_*.stl`
  files first (each arrives laid flat, one body per file): a B-key
  male/female pair, the O6.4 and O4.6 insert bores, the V1
  upper-pocket wall section (1.2 front / 1.6 floor walls), FOUR
  fishing-rehearsal blocks carved with the real round-5 duct geometry
  (entry cluster + O6.8 Y-step, the UM arc-top exit, the TS notch
  dive, a stand-foot R14 elbow pair), and TWO V1LF flush blocks:
  **7_recess_seat** — a ~46° sector of the U22 seat with through-void
  inboard of the cutout edge. Method: driver CONE-UP on its magnet on
  the table, then flip the block front-face-DOWN onto it so the seat
  lands on the flange edge (the void clears the cone; the motor never
  matters). Straightedge across block-face-vs-flange = flushness;
  rotate the driver to line a flange hole up with the block's pilot
  for a real M5×12 clamp test into the insert-on-pad stack; and
  **8_fish_um_oval** — the whole vase
  run (morph -> lane -> crest -> notch under the MU10 seat -> exit
  morph, the worst pull in the project; dry-fish BOTH tweeter pairs),
  which also carries the MU10 seat's left arc + cutout void for the
  same flange-edge drop-in test. Tune hole compensation, verify the
  thin wall and both seats, and dry-fish before committing the big
  pieces.
* **Internal voids:** the cable ducts (O3.8–9.3, arched ceilings)
  self-support — no supports on any flat piece. For the floor-stand
  bottom, preview the foot: the connector channel's internal ceiling
  bridges up to ~38 mm — default thick bridges handle it; add tree
  supports (on build plate only) only if the preview sags inside the
  channel, and paint support blockers over the duct bores.
* **V1LF flange recesses (front-down = recess floor is a ceiling over
  the bed):** the two seat annuli bridge 6.0 / 4.0 mm above the bed
  over a 10–16 mm radial span. Add **normal supports painted into the
  two recess rings only** (support/raft gap 0.2, 2 dense interface
  layers), keep blockers over everything else. The seat surface is a
  supported face: expect ±0.1–0.2 roughness — the drivers hide it,
  but FLUSHNESS depends on the real seat depth. Print coupon block
  **7_recess_seat** first, drop the actual driver flange edge into
  it, and caliper front-face-to-flange: adjust `LM_FLANGE_T_MM` /
  `UM_FLANGE_T_MM` in `top_baffle_nd25fw4_flush.py` and rebuild if
  it sits proud or sunk. NOTE the datasheets disagree with the
  measured thicknesses (U22 drawing 5.5±0.2 vs 6.0 measured; MU10
  drawing 5.4±0.2 vs 4.0 measured) — re-measure before the vase
  print especially: the MU10 seat deeper than ~4.5 leaves no room
  for the tweeter duct underneath (the design is at its limit).
* **Seam:** aligned, painted onto the outline's rear/hidden edges —
  keep it off the V0/C7 knife bevels and the front perimeter.
* **Floor-stand foot junction (required):** the foot meets the plate
  at a SHARP 90° inside corner with no gusset (the duct elbows own the
  corner interior), and printed front-down the joint loads layer
  adhesion in tension with a x2-3 stress concentration. At 30-40 %
  infill a hard knock or carrying the speaker by the baffle peaks the
  corner near the layer-adhesion limit. Fix in the slicer: add a
  **height-range modifier over print-z 12-32 mm with 100 % infill**
  (Bambu Studio: right-click the part > Add height range). That makes
  the joint cross-section solid (~5x the bending margin) for ~60 g.
  Handle the finished speaker by the FOOT, not the baffle top.

## Polar-index base (measurement turntable)

`floor_stand/stl/lx521_polar_base_{1of2_base,2of2_rotor}.stl`: print
both flat, no supports, PLA+ (30 % infill, 4-5 walls). The rotor's
fenced pocket takes the foot's floor footprint (0.4/side clearance;
NL8 plug gap at the rear); the base's two flex-arm noses click into
the rotor's 72-socket underside ring: 5-deg steps, firmer at the
10-deg majors, scale readable against the rotor's front notch. The
rotation axis sits 84.15 mm behind the front baffle plane (footprint
center — stability over eccentricity); correct per-angle mic distance
in post with r = 84.15 if you want absolute polars, or ignore it for
variant-vs-variant comparisons (identical bias). Double-side tape the
base to the stand; mark tape position for cross-session repeats.

## Cable fishing protocol

Nobody ever pulls a full route: every duct is open at each seam face,
so the worst single segment is ~200 mm (a mid piece).

1. **Dry-fish each piece right after printing** — it is the per-piece
   QC (a collapsed bore costs one reprint now, a rebuild later).
2. **Thread seam by seam at glue-up**, bottom -> mids -> vase. The
   dovetail keys align mating duct mouths to <=0.05 mm, and every
   crossing has a 1 mm lateral funnel relief on both faces.
3. **Leader**: ~1.5 mm nylon trimmer line or a guitar string, SOLDERED
   and heat-shrunk to the pair (no taped lumps — the LM pull has only
   0.4 mm of slack). Talc or silicone grease helps the long UM pull.
4. **Shared T duct**: fish the T2 pair first (it enters higher,
   z=9.5), then the T1 pair alongside it through the O6.8 step.
5. Hardest moments, in order: the UM window bend (R~10, end of its
   ~500 mm route), the LM snug section (~50 mm at 0.4 slack), the
   second pair past the T step junction.

## Assembly notes that interact with the material

* Glue the seams with 5–30 min **epoxy** (open time to seat the
  dovetail keys); CA gel is fine for the pin magnets. No solvent
  bonding — PLA ignores acetone.
* After the first loud listening session, re-torque all driver screws
  once — that is when the PLA under the flanges finishes settling.
* For V1/V1L: knock-test the assembled trunk before trusting the W22
  on the 11.5 mm section long-term. At ~25 % of stock bending
  stiffness the PLA+ Tough damping helps, but verify — SL's bridge
  exists precisely to keep the baffle from being excited.
