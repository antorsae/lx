# Printing the top baffle — PLA+ Tough (Bambu Studio)

Settings and engineering numbers for printing the piece sets in
Bambu PLA Tough+ on a 0.4 mm nozzle, for both `floor_stand/` and
`no_floor_stand/`. R6P proud-family pieces and the R6F V1LF collars are
covered separately where their geometry diverges. Everything here
combines manufacturer data, published reference tests, conservative
assumptions and analytical screens. None substitutes for owner-specific
coupon and assembly proof. PLA's governing weakness for this job is
**preload creep**.

## Loads and fasteners

Drivers are the LX521.4 production SEAS customs: "W22" rows = the
U22REX/P-SL (H1659-08), "10F" rows = the MU10RB-SL (H1658-04) — same
cutout/pilot geometry, real flanges O220.6 x 6.0 / O98 x 4.0
(owner-measured; see the V1LF recess note below).

The baffle carries ~3.2 kg of drivers: W22 ~2.6 kg on six inserts,
10F 0.43 kg on four, and the ND25 pair ~0.2 kg clamped. Static load per
W22 driver insert is ~5 N against an assumed 600 N conservative pull-out
capacity for an M5 x 5.8 x O6.3 heat-set in PLA (published reference tests:
900-1400 N), so the analytic driver-mount margin is >100x. The R6F
bridge-web load case is treated explicitly below. PLA still relaxes
30-50 % of bolt preload over the first days.

| Fastener | Spec | Torque | Notes |
|---|---|---|---|
| W22 | M5 x 14 pan + flat washer into M5 x 5.8 x O6.3 heat-set (bore O6.4 x 6.8) | 0.8–1.0 N·m | wave washers; re-torque at 24 h and ~2 weeks |
| W22 on **V1LF** | **M5 x 12** pan + flat washer into carrier heat-sets | 0.8–1.0 N·m | all six sites in no-floor; floor keeps carrier inserts only at 0/60/120° |
| W22 + R6F floor support, 180/240/300° sites | long driver-side M5 screw through the carrier's Ø5.5 clearance into a rear-installed Ø6.4 × 6.8 support heat-set | set only after a physical stack mock-up | floor state only; no rear head/front nut and no double-used insert; qualify actual screw engagement |
| 10F | M3 x 8 into M3 x 3 x O5 heat-set (bore O4.6 x 4.0) | 0.30–0.40 N·m | short engagement — do not overdrive |
| R6P bridge (no-stand) | M5 machine screw from the bridge (behind) into M5 x 5.8 x O6.3 heat-set (bore O6.4 x 6.8, REAR face) | hand-snug | 4 off; same insert as the W22, set from the rear |
| R6F collar half-laps | 2 x M3 through-bolts + washers/nuts; choose length against the printed 11.5 mm stack | hand-snug | structural UM/tweeter-to-LM load path; tighten evenly on a flat front-face datum |
| R6F fused bridge plate | four stock holes at (±20,20)/(±20,70), rear-opening Ø6.4 × 6.8 inserts | hand-snug for fit; final torque only after proof test | no-floor LM only; 62 mm insert core with soft cubic shoulders, centered rear UM/T entries at x=±5/y=82/z=5.3, solid acoustic front, immutable 40 × 50 pattern, and no geometry behind the existing LM pads; magnets receive zero load credit |
| R6F optional LM keyed seam | one concealed right-hand straight rounded tongue/blind-socket pair wholly inside the existing R110.6..R113 lip; world Y=172.481 mm, zero-gap planar butt; tongue width 0.8 mm, engagement 3.5 mm along the tangential insertion axis (~75.23° from +X) | registration only | mutually exclusive replacement print form for the canonical monolithic LM; no external protrusion, envelope growth, extra screw, or standalone retention/load credit. Assemble front-face-down on a flat datum; the installed LM driver flange and its normal fasteners are the structural splice. |
| R6F UM-to-tweeter half-laps | 2 x rear-driven M3 screws at x=±24, y=421.5 into blind M3 x 3 inserts in Ø4.6 x 4.0 receivers | hand-snug | no front bolt head; keep the acoustic face uninterrupted |
| R6F alignment magnets | Six D5 × 2 N52 magnets in Ø5.2 × 2.2 radial pockets | — | LM has four: preserve the upper flush 64°/116° pair with its verified 2.2 mm insert and 0.86 mm route-cover gaps, and add a lower face-flush pair at 224°/316° with at least 23.0 mm nearest-insert edge clearance. The 224° pocket remains at z=12.55; the route-adjacent 316° pocket moves only in Z to 15.40 and retains a closed 0.30 mm front skin. Both clear buried routes and the bridge/support load path. UM keeps its flush 50.5°/129.5° pair at z=15.1, with at least 1.1 mm insert/T-cover gap, 0.2 mm radial floor and 0.6 mm front skin. Fixture all six flush during cure—0.2 mm extra pocket depth is adhesive allowance; no proud ears; alignment/anti-rattle only, **zero structural load credit** |
| Tweeter pair | M4 through-bolts + nyloc + wave washer; length = septum + faceplates (stock 18.3 septum → ~M4 x 35; V1/R6F crescent → ~M4 x 30 — verify stacked) | snug, ~0.5 N·m | clamps the 4.0 mm crescent seat; recheck after a week |

**Installing the inserts:** soldering iron at 230–250 °C, press
slowly and square, stop flush. The bores carry +1.0 mm melt room by
design.

### R6F structural screens

The fused bridge web and four-hole group use a conservative **4.0 kg**
installed mass, y=230 mm center of mass, and 70 mm rear offset.
Bambu's published XY flexural
strength is 65 MPa for PLA Tough+; this screen deliberately uses much
lower 18 MPa short-duration and 8 MPa sustained allowables for printed
anisotropy, stress concentrations, and creep. The softly blended plate has a
62 mm insert core and occupies z=5.3..18.3, ending at the deepest existing
LM-pad rear datum with no added rear projection. The structural model deducts
the complete Ø8.2 and Ø6.0 centered entry lumens and credits only a
**47.8 × 13.0 mm** section; exact 0.01 mm sampled cuts retain at least 53.5 mm.
Its in-plane/rear section moduli are **4950.5/1346.4 mm³**. Conservatively
summing the two bending stresses gives about **3.28/9.84/16.41 MPa** and
safety factors **2.44/1.83/1.10** at sustained 1g/8 MPa, transient 3g/18 MPa,
and transient 5g/18 MPa. The physical 68° lower-ring cradle
extends to z=5.3 with the web, but the existing ring lip starts at z=6.8;
the calculation therefore credits only the actual 11.5 mm-deep monolithic
interface. After conservatively deducting one Ø8.2 UM tunnel plus the
complete Ø6.0 tweeter tunnel, its effective width is **118.5 mm**, with
in-plane/rear section moduli of **26908.4/2611.6 mm³** and biaxial safety
factors of approximately **6.37/4.78/2.87**. The combined
normal-plus-rear worst insert reaction is **434.2 N** at 5g, retaining
**1.38** assumed pull-out safety factor. Magnets contribute 0 N. The
material-property reference is the
[Bambu PLA Tough+ data sheet](https://us.store.bambulab.com/products/pla-tough-upgrade?id=624483921975980068).

The floor support uses a separately derived **456.9 mm³** narrowest
10×6 flange/twin-4×12-web section. Its required section is approximately
**128.1/170.8/284.7 mm³** at 1g/3g/5g, for safety factors about
**3.57/2.67/1.60**; the three support inserts retain about **2.00**
combined-axis assumed pull-out safety factor at 5g. The two collar ears and the two
tweeter-interface ears carry only the upper assembly, not the LM/bridge.
The ledgered 0.43 kg MU + 0.20 kg tweeters plus printed/hardware allowance
gives a **0.85 kg** case over conservative 120 mm plan and 70 mm rear
levers. Governing contact factors are about **2.82/2.12/1.27** at
1g/3g/5g; M3 tension factor is about **1.17** at 5g. The three floor-load
LM spokes screen at about **17.1/12.9/7.7** in direct shear. Magnets
contribute 0 N to every calculation.

These screens cover the modeled bridge/floor insert reactions, minimum
printed transfer sections, ear neck/net/bearing areas and M3 shear. They
do **not** independently qualify the floor rails/NL8 panel, stock bridge,
installation substrate, real insert process, or a changed print orientation.
Those items remain inside the system proof-test boundary.

They also do not transfer automatically from the canonical monolithic LM to
the optional LM keyed split. Its one concealed right-hand tongue/socket pair is a
registration aid with no standalone retention/load credit. That
print form remains pending until tongue/socket fit, full seating, front-datum
coplanarity, UM/T route-seam continuity and cable pull-through, and the
complete driver-installed 1g/3g/5g proof have passed and been recorded.

This calculation is a screening model, not permission to hang drivers
from an untested print. Before service, every final combination of
filament batch, slicer settings, frame, carrier, inserts, and bolts must
pass a documented physical proof test through the distributed 4 kg
sustained-1g, 3g and 5g bridge/support cases in the governing normal and
rear-moment directions. The upper joints must simultaneously carry their
actual 0.85 kg distributed mass at the stated lever arms.
Prefer a dummy mass rather than valuable drivers. Apply **39.23 / 117.68 /
196.13 N** (4/12/20 kgf equivalent) through the modeled center of mass,
including its 230 mm
plan coordinate and 70 mm rear offset; simply hanging 4 kg at a mount
does not reproduce the screened moments. Record load, fixture, duration,
temperature, deflection, insert motion, cracks, and post-test torque.
Ramp each load over at least 10 s; hold sustained 1g for 24 h, 3g for
60 s, and 5g for 10 s before the unloaded inspection; reject any
permanent movement or damage.
Record the exact per-state candidate identity, print/insert process,
fixture, load history, temperature, deflection, damage inspection and
release signoff in `V1LF_PHYSICAL_QUALIFICATION.md`.

**Temperature limit:** Bambu publishes a **61 °C heat-deflection
temperature (0.45 MPa)** for PLA Tough+; that is not a guaranteed Tg or
an assembly service rating. Indoors near room temperature is the only
condition screened here.
Direct summer sun on a dark baffle or a closed car will creep the W22
mounts, bridge-web interface, and tweeter clamp. The safety factors above are
room-temperature values and are invalid near 60 °C. If the speakers may
see elevated temperatures, choose a suitable material before printing
and repeat the structural proof test at the actual maximum service
temperature; substituting PETG/ASA does not preserve the PLA calculation
automatically.

## Orientation — the setting that matters most

**R6P:** print every baffle piece front face down. The front plane is
the reliable datum while C7/V0/V1/V1L sculpt or step the rear. For the
floor-stand `piece_bottom` (223.8 × 125 × 168.3), front-down leaves the
150 mm foot rising as a self-standing wall. Smooth/satin PEI gives a
clean front; textured PEI gives uniform grain.

**R6F:** floor-state collars print front-face-down so their mounting
planes share the bed datum. The monolithic no-floor LM+bridge web is
too long for any flat Z rotation on 256 × 256 mm; its exporter therefore
uses the validated **45° X tilt**. Preserve that orientation and review
supports, interface faces, and the complete Z height before slicing.
The optional LM keyed split replaces that monolithic LM with two flat prints:
the authoritative measured Z26° bottom footprint is
**198.79 × 138.22 mm** in floor state or **198.79 × 205.51 mm** in no-floor
state; the authoritative measured Z45° top is **210.47 × 210.47 mm** in either
state. Print both front-face-down; the single right-hand straight rounded
0.8 mm tongue and blind socket remain concealed inside the existing lip and
add no envelope growth. The tongue engages 3.5 mm along its tangential insertion axis
(~75.23° from +X). Never mix either half with the monolithic LM.
Place the separate floor support (required for a floor-state assembly) on
its broad rear face. The UM route is covered only in LM, T is covered in
LM/UM, and their specified rear continuations are free; the short LM lead is
also intentionally free. Keep generated support out of functional buried-route
mouths/free-cable clearance and inspect every rear bump. Floor LM uses 28° Z rotation and
the floor/NL8 support uses 70°.

## Bambu Studio profile (0.4 nozzle)

* **Layer height** 0.20 mm.
* **Walls: 6 loops** (2.4 mm) — makes the material around every
  insert bore, dovetail key, and knife edge fully solid.
  Top/bottom **6/5** layers; *Ensure vertical shell thickness: All*.
* **Infill:** gyroid, **30 %** for R6P mids/vase and the R6F collars,
  **40 % for the R6P piece_bottom in both stand states** (foot standing
  moment ~8 N·m, or bridge bolts), and **40 % for the R6F fused bridge
  web, required floor support add-on, and tweeter crescent/direct joint ears**. *Detect narrow internal solid
  infill: on.*
* **Strength tuning:** nozzle 225 °C (top-middle of the Tough range —
  hotter = better layer adhesion), bed 55–60 °C, **max fan 60 %**
  (overhang fan 100 %), outer wall <=120 mm/s, keep the filament
  profile's volumetric limit (~12–16 mm³/s). Strength lives in layer
  adhesion, not speed.
* **Dimensional fits:** R6P dovetails use 0.05 mm working clearance;
  tune compensation until the coupon key slides firmly. Insert bores
  are nominal. Use *Precise wall: on*, **elephant-foot compensation
  0.15 mm**, and start X-Y hole compensation at **+0.05**.

Print the applicable stable `stl/lx521_coupon_*.stl` files before the
large parts:

1. `lx521_coupon_1_fit_plate.stl` — female dovetail, Ø6.4/Ø4.6 insert
   bores, and the global Ø5.2 × 2.2 magnet pocket in the V1 upper-pocket
   wall. Use it to establish a face-flush adhesive fixture; do not bottom
   the Ø5 × 2 magnet.
2. `lx521_coupon_2_fit_key.stl` — matching loose male dovetail.
3. `lx521_coupon_3_fish_entry.stl` — no-foot entry cluster and Ø6.8
   tweeter-pair merge.
4. `lx521_coupon_4_um_outlet_proud.stl` — the real B2 outline and the
   complete standard B2/C7/V0/V1 R6P Ø8.2/G1/R14 rear outlet at
   (33.446, 301.492). It does **not** represent the keyed V1L outlet.
5. `lx521_coupon_5_fish_ts_dive.stl` — proud-family tweeter notch
   passage.
6. `lx521_coupon_6_fish_foot.stl` — stand-foot R14 elbow pair.
7. `lx521_coupon_7_recess_seat.stl` — actual V1LF LM-core seat sector
   and rear insert pad.
   Put the U22 cone-up on its magnet and
   lower the coupon front-face-down over the flange edge; check
   face-to-flange level with a straightedge, then perform the real
   M5×12 insert/pad clamp test.
8. `lx521_coupon_8_fish_ts_oval_proud.stl` — complete proud-family
   round-to-oval tweeter passage; dry-fish both AWG24 pairs.
9. `lx521_coupon_9_um_faston_clocking.stl` — D104 gauge marking the
   238/328-degree mounting screws and their 283-degree terminal
   midpoint. Place it against the physical MU and record where the
   actual tabs, boots, and carrier land.
12. `lx521_coupon_12_v1lf_closed_bore_bump.stl` — state-specific R6F
    LM-collar sector around the 300° axis, including the enclosed tunnel and
    full-width solid saddle from conduit roof to bore floor. The floor copy
    removes only its exact grown insert/shank hardware clearance; there is no
    trapped hollow and no cable is exposed.
R6P uses a short curved split grommet that follows the final R14 bore;
print `lx521_top_proud_addon_um_grommet_half_{a,b}.stl` and test it with
coupon 4. That grommet fits B2/C7/V0/V1 only; do not fit it to V1L.
V1L has no separate coupon in this set: its printed `mid_right` is the
dry-fishing and physical service-fit article for the 283-degree exit.
Use `lx521_top_v1l_addon_um_grommet_half_{a,b}.stl`, not the proud
halves. The V1L TPU insert follows the keyed R14 with a Ø8 body around a
Ø7.1 bore, enters 2.5 mm, and seats its Ø13 × 2 flange on the z=6.8 rear
face. Test both halves around the measured cable and in the printed
`mid_right`; the analytic model clears the Faston motion box, but the
real terminals and boots remain the release gate.
V1LF has no printed grommet or tunnel clip. Any selected external cable
retention remains **physical-fit pending** and must clear the buried-route
mouth, free cable, driver and Faston service envelopes with the measured UM cable.

* **R6P internal voids:** the cable ducts (Ø3.8–9.3, arched ceilings)
  self-support on the flat pieces. For the floor-stand bottom, preview
  the connector channel's ~38 mm ceiling; use build-plate-only tree
  supports only if the preview shows an unsafe bridge, and paint
  support blockers over every duct bore.
* **V1LF collar recesses (front-down = seat floor is a ceiling over
  the bed):** the LM and UM annuli bridge 6.0 / 4.0 mm above the bed.
  Add **normal supports painted into the two seat rings only**
  (support/raft gap 0.2, 2 dense interface
  layers), keep blockers over everything else. The seat surface is a
  supported face: expect ±0.1–0.2 roughness — the drivers hide it,
  but FLUSHNESS depends on the real seat depth. Print coupon block
  **7_recess_seat** first, drop the actual driver flange edge into
  it, and caliper front-face-to-flange: adjust `LM_FLANGE_T_MM` /
  `UM_FLANGE_T_MM` in `top_baffle_nd25fw4_flush.py` and rebuild if
  either sits proud or sunk. The datasheets disagree with the measured
  thicknesses (U22 drawing 5.5±0.2 vs 6.0 measured; MU10 drawing
  5.4±0.2 vs 4.0 measured), so re-measure both physical flanges. The
  mirrored Ø6 tweeter tunnel runs in the rear web and does not
  reduce the flush seating surface.
* **R6F routes:** preview the physical Ø7.0 UM cable inside its Ø8.2
  LM-owned passage and the Ø5.2 T cable inside its Ø6.0 LM/UM-owned passage
  at fine layer resolution. The UM cable must run free behind the UM carrier
  with no printed rear duct; T must run free behind the tweeter crescent with
  no crescent-owned printed arc. The true minimum non-load wall on each
  surviving buried span is 0.8 mm and the seat roof is 0.85 mm. The D7.8
  LM lead is different: it floats over a short 20.15 mm span behind the carrier
  with no printed micro-duct, cover, or cutter. Confirm its z=0.40..3.80
  centerline and 1.00 mm outer clearance to the deepest z=5.3 pad/web rear
  datum, and keep the floor support's
  physical-cable-plus-0.4 mm clearance open. Block support from every
  functional mouth/free-cable span and
  inspect the complete solid-backed rear bumps. Use the routing PNG's nominal
  diametric u-z sections—not only its longitudinal station plot—to confirm each
  conduit roof is joined solidly to its blind-bore floor. Verify the 82.67°
  physical crown crossing has T above UM and retains the 1.85 mm
  physical-cable gap; there is no two-duct separator web to inspect. Then
  follow T around the covered 328°/58° UM-carrier bumps to its flush exit,
  after which it remains free behind the crescent. Floor-state 300/240/180° support
  crossings retain only the modeled hardware-clearance cylinders; all
  surrounding saddle material remains solid. Any support strand
  left in a tunnel reduces cable space. Measure the actual cables before
  relying on a retainer or strain relief.
* **R6F terminal gate:** do not treat the STEP proxy as hardware CAD.
  Use coupon 9 and the real MU driver to verify the 283-degree clock,
  tab/boot volume, polarity order, and real withdrawal direction before
  printing the UM collar and any nearby add-ons. There is no printed
  UM-carrier rear duct or D82 mouth: the free cable follows the modeled R15
  approach to that immutable service axis with a clockwise circumferential
  193° tangent, then continues exact-G1 through R20. The fit
  STEP's closed Ø98/Ø80/Ø60 body keep-out proves clearance to the known
  terminal-less body, but cannot prove the omitted tabs. Its W22 keepout
  records a conservative proxy placement from hash-pinned
  `E0022_W22EX001.stp` SHA-256
  `7fc2be551c86006e11c32a570b046772987cb86dcf65350f77c6e34709aa5ab6`,
  using +90° about X and translation `(0,200.981,-47.498931)`: native +Y
  maps to world +Z, native +Z maps to world -Y, and native max-Y lands at
  front z=18.3. The cached native bounds map to world
  `(-110.5,90.481,-84.498931)..(110.5,311.481,18.3)`. The guarded W22
  geometry phase imports that exact STEP, verifies the transform/bounds, and
  proves the stepped proxy contains it. This still does not qualify the
  installed custom U22 or any leads.
  `PHYSICAL_MEASURE_REQUIRED = True`. The 12 mm maximum modeled pull
  equals the provisional exposed-tab length, so it has zero positive
  release overtravel margin and qualification remains pending.
  Complete `V1LF_PHYSICAL_QUALIFICATION.md` against the real hardware and
  structural proof loads for each stand state; the pending record is
  checksum-bound into generated candidate manifests but authorizes no release.
* **R6P seam:** align it on the outline's rear/hidden edges —
  keep it off the V0/C7 knife bevels and the front perimeter.
* **R6P floor-stand foot junction (required):** the foot meets the plate
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

### R6P proud family

1. **Dry-fish each piece immediately after printing.** Every duct is
   open at a seam, so a collapsed segment should cost one piece rather
   than an assembled baffle.
2. **Thread during glue-up**, bottom → mids → vase. The dovetails align
   mating mouths and the crossings carry funnel relief.
3. Use ~1.5 mm nylon trimmer line or a guitar string as the leader,
   soldered and heat-shrunk to the pair; do not create a taped lump.
   The LM bundle remains the tightest nominal fit. Use only a
   plastic-safe dry lubricant or a small amount of silicone grease.
4. Fish T2 first (the higher z=9.5 feeder), then T1 beside it through
   the shared Ø6.8 step.
5. For B2/C7/V0/V1, dry-fish
   `lx521_coupon_4_um_outlet_proud.stl` with the actual UM cable. The
   route is Ø8.2 from main through its G1 R14 rear turn, so the estimated
   Ø7 cable has 0.6 mm nominal radial slack and no discrete corner. Then
   rehearse the snug LM segment and the second tweeter pair through the
   merge.
6. For V1L, dry-fish the actual printed `piece_mid_right`; coupon 4 and
   its curved R14 grommet are the wrong geometry. Verify that the
   physical aperture is centered at Q=(13.497063, 307.618796, 6.8),
   radius 60.0 mm on the 283-degree terminal axis. The nominal cutter
   continues outside to (11.080158, 308.797599, −2.0); do not use that
   endpoint as the aperture center. Rehearse the measured cable, edge
   protection, dedicated V1L split TPU grommet, service loop, boots, and
   physically measured Faston withdrawal against the real MU. Install the two grommet
   halves only after the cable passes the dry-fishing rehearsal. The
   tail is wholly in `piece_mid_right`, so no special fishing or
   modification of the top/vase is required.

### R6F V1LF

R6F has a UM passage buried only in LM, a T passage buried in LM/UM, free
rear spans behind UM and the tweeter crescent, and a separate un-ducted LM
lead. Rehearse every cable before the drivers or optional modules hide the
working area:

1. Place the measured D7.8 LM cable over the modeled 20.15 mm free span at 269.5°
   behind the carrier; do not add a printed micro-duct. Confirm it floats clear
   of the LM/web and passes through the floor support's cable-only clearance
   when that support is fitted. Add external retention that does not refill
   the modeled clearance or load the cable termination.
2. In no-floor mode, begin the UM cable at the centered bridge rear mouth
   `(5,82,5.3)`; floor mode retains the supported ring mouth. Fish it through
   each covered LM-owned pad bump, then let it exit to a free span behind the
   UM carrier. Confirm the UM carrier has no printed rear duct or D82 mouth.
   Keep the free Ø7.0 cable below the physical crown crossing, follow its R15
   terminal approach to the 283° reference, and continue through R20 to
   the named Y breakout. Verify its 4 mm-long OD8 collar and both OD4
   branches fit, then verify the provisional Ø3.2/R8 slack leads enter
   their own low-profile flag Fastons. Exercise one connector at a time at
   0/3/6/9/12 mm while the other stays installed, without loading either
   tab.
3. In no-floor mode, begin the tweeter bundle at `(-5,82,5.3)`; floor mode
   keeps its supported ring mouth. Fish it through the crown above the UM
   cable and the covered 328°/58° pilot bumps. Confirm the LM-to-UM T
   ownership mouths are flush, then let T exit the UM-owned passage and run
   free behind the tweeter crescent. The crescent has no printed cable arc,
   conduit, socket, or horn. If the crescent is used, position the free cable
   before bolting the direct ears at x=±24, y=421.5; do not trap it behind the
   blind-M3 joint.
4. V1LF has no printed UM grommet. Fit only the selected external cable
   retention after successful dry-fishing, and keep it out of the buried-route
   mouths, free cable paths, and Faston pull envelope.

The CAD proxy assumes non-overlapping 8.5 mm receptacles and 9.5 mm
low-profile flag boots at 11 mm pitch. `PHYSICAL_MEASURE_REQUIRED` remains
true because the MU mesh omits the terminals. Connect physical hardware
only after measuring those widths, terminal clock, polarity order, flag
orientation, cable and Y-breakout fit, and pull-off stroke. The 12 mm
modeled endpoint is equal to the provisional 12 mm exposed tab and has no
positive release overtravel; it is not proof of disengagement. Oversize
straight boots are not equivalent: choose a measured compatible flag
connector or revise the service geometry.

## Assembly notes that interact with the material

### R6P proud family

* Glue the seams with 5–30 min **epoxy** (open time to seat the
  dovetail keys); CA gel is fine for the magnets. Every version uses
  Ø5.2 × 2.2 pockets for the actual Ø5 × 2 discs. The extra 0.2 mm depth
  is adhesive allowance: fixture each disc flush with its mating face
  during cure and do not bottom it. No solvent
  bonding — PLA ignores acetone.
* After the first loud listening session, re-torque all driver screws
  once — that is when the PLA under the flanges finishes settling.
* For V1/V1L: knock-test the assembled trunk before trusting the W22
  on the 11.5 mm section long-term. At ~25 % of stock bending
  stiffness the bridge/support remains important.
* For V1L specifically: inspect and dry-fish its keyed 283-degree outlet
  before seam glue or driver installation. The real MU terminal tabs are
  absent from the reference mesh, so the physical Fastons, boots,
  removal stroke, cable, and dedicated V1L split grommet remain
  mandatory release checks even when the analytic route checks pass.

### R6F V1LF barebone

Before hardware assembly, inspect
`top_baffle_nd25fw4_v1lf_split.step` for the mandatory core,
`top_baffle_nd25fw4_v1lf_lm_split.step` if the optional two-print LM form is
selected,
`top_baffle_nd25fw4_v1lf_attachments.step` for the selected printed
modules, and `top_baffle_nd25fw4_v1lf_assembled.step` with
`top_baffle_nd25fw4_um_fit.step` for the service keep-clear. The last
two contain reference geometry and are not an instruction to print the
Faston proxy.

Choose exactly one LM print form for each state: either the canonical
`lx521_top_v1lf_core_1of2_lm_carrier.stl`, or both
`lx521_top_v1lf_optional_lm_keyed_1of2_bottom.stl` and
`lx521_top_v1lf_optional_lm_keyed_2of2_top.stl`. The optional seam is at world
Y=172.481 mm with a closed zero-gap planar butt. Before step 4, place both
front faces down on one flat datum and fully seat the bottom half's single
concealed right-hand straight rounded tongue in the top half's blind socket.
Inspect tongue/socket fit, full seating, coplanarity and route-seam continuity,
then pull the actual UM/T cables through both preserved lumen handoffs. The tongue/socket
adds no external protrusion, extra screw, or standalone retention/load credit.
Hold registration when lifting the LM from the datum for driver fit-up; the
installed flange and all normal LM fasteners provide the structural splice.
Do not load an unspliced split LM.

1. Complete coupons 7, 9, and 12, then measure the real MU
   terminal carrier, tabs, boots, and cable. Stop if the physical parts
   exceed the modeled service space.
2. Install all six rotated LM carrier inserts in no-floor mode. In floor
   mode install carrier inserts only at 0/60/120°; install the other three
   heat-sets from the rear into the support bosses at 180/240/300° and
   leave the matching carrier holes as Ø5.5 clearances. Install four UM
   inserts and the two blind M3 inserts in the crescent half-laps. No-floor
   additionally owns four rear-opening bridge inserts in the 62 mm solid web;
   each Ø6.4 × 6.8 bore starts at z=5.3 and leaves a 6.2 mm front floor. Set every
   insert square and reject any cracked, loose, or over-melted boss.
3. Glue six D5×2 magnets into the Ø5.2×2.2 radial pockets. LM uses four:
   preserve the upper flush 64°/116° pair and its at-least-2.2 mm nearest
   insert gap, then fit the new lower face-flush pair at approximately
   224°/316° only after confirming clearance from lower inserts, buried
   routes, and the bridge/support load path. UM keeps the flush
   50.5°/129.5° pair centered at z=15.1; verify the 0.2 mm radial pocket
   floors, 0.6 mm front skins and 1.1 mm minimum T-cover gap. Hold each
   magnet flush while the adhesive cures; do not bottom it into the
   0.2 mm adhesive allowance. No site has a proud ear. Use the marked polarity standard (core pole OUT,
   mating add-on pole IN). These magnets align and suppress rattle only;
   assign them **zero load capacity** in every assembly and test.
4. Put both collars front-face-down on one flat reference. Engage the
   two rounded half-lap ears at x=±32.0, y=315.770, insert the two M3
   through-bolts,
   and tighten evenly while holding both front faces coplanar. Verify
   the 165.100 mm driver-center spacing before loading the drivers.
5. In floor mode, install the required support add-on: seat its receivers
   at 180/240/300° and run all three long driver-side M5 screws through
   the carrier into the support heat-sets with verified engagement.
   The support has no obsolete LM magnet cups/arms; its LM opening is solely
   physical-cable-plus-clearance space for the free lead.
   In no-floor mode, bolt the stock bridge directly to the immutable
   (±20,20)/(±20,70) pattern in the fused front-flush web; there is no separate
   no-floor keel. Feed UM/T through the plate's centered rear mouths before
   the bridge blocks access. The floor LM has no bridge tail. Confirm every closed
   solid-backed cable bump clears real hardware before tightening. Magnets
   remain alignment/anti-rattle aids, never a load path. Add only the remaining
   optional modules needed. Attach the tweeter
   crescent directly at x=±24, y=421.5
   with two rear-driven M3 screws into its blind inserts; no bolt head
   may break the acoustic front. V1LF has no printed grommet; keep the
   selected external cable retention outside the buried-route mouths and free
   cable paths; do not rely on it until its physical cable/retention fit passes.
   Keep every module clear of all driver, service, buried-route, and free
   cable envelopes.
6. Rotate the physical MU to the equivalent mounting orientation that
   puts its terminal carrier on the free rear-UM lead and the
   **283-degree service axis**, verified with coupon 9 and the routing sheet,
   exactly between the 238- and 328-degree screw positions. Fit and
   remove each measured low-profile flag Faston separately with the
   cables and adjacent modules in place; its polarity-specific Ø3.2 lead
   must clear the installed opposite connector and both drivers at each
   one-at-a-time 0/3/6/9/12 mm review state. Record the physical release
   stroke: the modeled 12 mm endpoint has zero positive overtravel beyond
   the provisional exposed tab and does not itself prove disengagement.
7. Follow the R6F cable-fishing protocol above. Confirm every covered
   LM-pad bump, the physical T-over-UM crown crossing, and the covered
   328°/58° T-route bumps follow the modeled smooth curves without a kink.
   Confirm the UM span is free behind the UM carrier and the T span is free
   behind the crescent, with neither deleted duct recreated by support or
   retention. Use the true
   local u-z views to confirm the full solid saddle to each bore floor. Add the
   chosen retention/strain relief (one form is mandatory) and connect the
   Fastons only after this inspection. Cable load must be reacted away
   from the MU tabs.
8. Before entrusting drivers to the support, proof-test the finished
   carrier/web-or-support/insert/bolt system with a dummy 4.0 kg mass through the
   sustained-1g, 3g, and 5g cases in the governing normal and
   rear-moment directions. Follow the documentation and rejection
   criteria in “R6F structural screens”; magnets receive no credit.
   For the optional LM split, install the representative LM driver or an
   equivalent flange-and-all-fasteners splice during the complete 1g/3g/5g
   proof; give the concealed tongue/socket no standalone retention or load credit
   and record the prior tongue/socket fit, full-seat, coplanarity, route-seam and
   cable-pull-through evidence with the result.
   Repeat after any filament, slicer, material, insert-process, or
   service-temperature change.
9. Mount the drivers, apply the same low torques in the table, and
   re-check collar coplanarity and all fasteners after 24 hours and the
   first loud session.
