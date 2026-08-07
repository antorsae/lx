# Printing the top baffle — Bambu PLA families (Bambu Studio)

Settings and engineering numbers for printing the piece sets in
Bambu PLA Tough+, PLA Basic, PLA Lite, PLA Matte, or PLA Silk+ on a 0.4 mm
nozzle, for both `build/floor_stand/` and
`build/no_floor_stand/`. Proud-family pieces and the Obi-Wan collars are
covered separately where their geometry diverges. Everything here
combines manufacturer data, published reference tests, conservative
assumptions and analytical screens. None substitutes for owner-specific
coupon and assembly proof. PLA's governing weakness for this job is
**preload creep**.

## Loads and fasteners

Drivers are the LX521.4 production SEAS customs: "W22" rows = the
U22REX/P-SL (H1659-08), "10F" rows = the MU10RB-SL (H1658-04) — same
cutout/pilot geometry, real flanges O220.6 x 6.0 / O98 x 4.0
(owner-measured; see the Obi-Wan recess note below).

The baffle carries ~3.2 kg of drivers: W22 ~2.6 kg on six inserts,
10F 0.43 kg on four, and the ND25 pair ~0.2 kg clamped. Static load per
W22 driver insert is ~5 N against an assumed 600 N conservative pull-out
capacity for an M5 x 5.8 x O6.3 heat-set in PLA (published reference tests:
900-1400 N), so the analytic driver-mount margin is >100x. The Obi-Wan
bridge-web load case is treated explicitly below. PLA still relaxes
30-50 % of bolt preload over the first days.

| Fastener | Spec | Torque | Notes |
|---|---|---|---|
| W22 | M5 x 14 pan + flat washer into M5 x 5.8 x O6.3 heat-set (unchanged 6.8 mm total bore: O6.5 x 2.0 entry, then O6.4) | 0.8–1.0 N·m | Stock, Slim and Obi-Wan share the 0/60/120/180/240/300° clock; wave washers; re-torque at 24 h and ~2 weeks |
| W22 on **Obi-Wan** | **M5 x 12** pan + flat washer into carrier heat-sets | 0.8–1.0 N·m | all six 0/60/120/180/240/300° sites are ordinary blind carrier inserts in both states; each keeps its 6.2 mm total depth and uses an O6.5 x 2.0 entry followed by O6.4; floor mode has no secondary support inserts or long through-screws |
| 10F | M3 x 8 into M3 x 3 x O5 heat-set (bore O4.6 x 4.0) | 0.30–0.40 N·m | short engagement — do not overdrive |
| Proud bridge (no-stand) | M5 machine screw from the bridge (behind) into M5 x 5.8 x O6.3 heat-set (unchanged 6.8 mm total: O6.5 x 2.0 entry, then O6.4, from the REAR face) | hand-snug | 4 off; same insert as the W22, set from the rear. Stock/Slim use Obi-Wan's exact three-port D20 cable interface relative to these inserts: LM `(-0.35,64.76)`, shared T `(-4.75,55.91)`, UM `(3.17,55.91)`. |
| Proud mid-right-to-vase seam | **M3 x 20 socket-cap**, no washer, through a Ø3.4 radial passage into one M3 x 3 heat-set in a blind Ø4.6 x 4.0 vase receiver | hand-snug | axis `(x,z)=(0,12.55)`, +Y from the LM cutout. Ø6.2 x 3.4 head pocket clears a normal D5.5 x H3.0 socket head and hides it completely after W22 installation. Nominal insert engagement 3.381 mm; tip-to-floor margin 0.619 mm. Do not substitute M3 x 25. |
| Obi-Wan LM-to-UM half-laps | 2 x rear-driven M3 screws through the LM's standalone Ø3.4 clearance bores into M3 x 3 heat-set inserts installed in the UM's standalone rear-opening Ø4.6 x 4.0 blind receivers | hand-snug | x=±32, y=315.770; the closure-web/base teardrops remain nominal Ø9, while the complete Z-owned cylindrical functional bosses are locally Ø9.8. The UM receivers retain a 1.9 mm acoustic-front floor and the Z-halves retain a 0.20 mm axial gap. Select screw length for full insert engagement without bottoming. No washer, nut, or front bolt head belongs to this interface. |
| Obi-Wan fused bridge plate | four stock holes at (±20,20)/(±20,70), rear-opening unchanged 6.8 mm-total bores with Ø6.5 x 2.0 entries followed by Ø6.4 | hand-snug for fit; final torque only after proof test | no-floor LM only; 62 mm insert core with soft cubic shoulders; LM/T/UM entries packed inside the D20 support opening as LM above, T lower-left, UM lower-right at z=5.3; solid acoustic front; immutable 40 × 50 pattern; no geometry behind the existing LM pads; magnets receive zero load credit |
| Obi-Wan optional LM keyed seam | two symmetric Ø1.60 cylindrical pins at x=±109.187/z=14.30, normal to the world-Y=172.481 mm zero-gap seam and pointing +Y; 2.40 mm engagement plus 0.50 mm root overlap. Right blind socket Ø1.84 round; left blind socket 1.96 × 1.84 mm X-relieved; both 2.65 mm deep with 0.12 mm radial and 0.25 mm end clearance. Small exterior lands outside the LM recess preserve ≥0.50 mm radial/end walls, ≥0.05 mm recess plan clearance and ≥0.13 mm conservative W22-flange clearance. | registration only | mutually exclusive replacement print form for the canonical monolithic LM. Round+relieved sockets tolerate ±0.30 mm relative pitch error; the lands reach R114.4036, which is 1.4036 mm beyond structural R113.0 but only 0.6036 mm beyond the finalized R113.8 visible fairing. They add no extra screw or standalone retention/load credit. Print and assemble front-face-down on a flat datum, moving the top straight along -Y; the installed LM driver flange and its normal fasteners are the structural splice. |
| Obi-Wan UM-to-tweeter half-laps | 2 x rear-driven M3 screws through the UM's standalone Ø3.4 clearance bores into M3 x 3 heat-set inserts installed in the crescent's standalone rear-opening Ø4.6 x 4.0 blind receivers | hand-snug | x=±24, y=421.5; nominal Ø9 closure-base teardrops, complete local Ø9.8 Z-owned functional bosses, 0.20 mm axial gap, complete 360° receiver walls, and 1.9 mm acoustic-front floors. Install both inserts in the individual crescent before assembly; no front bolt head or cross-owner receiver wall. |
| Obi-Wan alignment magnets | Six D5 × 2 N52 magnets in captive Ø5.20 × 2.10 surface-normal cavities | — | LM has four: preserve the upper ring-radial 64°/116° axes and the lower pair at cubic parameter `u=0.50` on the shared curved shoulder. The right visible datum is `(x,y,z)=(45.285011,89.190370,15.10)` with outward normal `(0.706451,-0.707762)`; the left is its exact X mirror. UM keeps its 50.5°/129.5° axes; all six stations share source Z=15.10. Every station has 0.45 mm axial skins and a 45° support-free roof. The R113.0/R51.7 structural rings have continuous exposed R113.8/R52.5 side fairings, clipped only inside the existing LM--UM and T--UM cusp/service regions; the LM--UM stop preserves the 0.40 mm gap. Ring cavity datums sit at structural radius +0.65 mm; ring and shoulder cavity datums are 0.15 mm beneath their exposed surfaces. There is no magnet-local backing, boss, relief, rear cap, flat, or visible cue: the magnet-free exterior is immutable. Magnets are fully buried at the manifest pause, never glued or externally accessible. Flat/graded have matching captive LM-lower, LM-upper, and UM receivers. Their mating surfaces are flush with zero physical air gap; the receiver's 0.05 mm allowance is a solid standoff. Nominal paired magnet-face separation is 1.10 mm at LM-lower, LM-upper, and UM. Alignment/anti-rattle only: **zero structural load credit**. |
| Tweeter pair | M4 through-bolts + nyloc + wave washer; length = septum + faceplates (stock 18.3 septum → ~M4 x 35; V1/Obi-Wan crescent → ~M4 x 30 — verify stacked) | snug, ~0.5 N·m | clamps the 4.0 mm crescent seat; recheck after a week |
| Optional TEBM35C10-4 BMR pair | 8 × M2 screws into M2 × 4 × Ø3.2 heat-set inserts | hand-snug; establish on a coupon | lower BMR mounts from the acoustic front and upper BMR from the rear; each pocket has a 1.2 mm blind opposite wall. Use the Stock BMR vase only with Stock receivers and the Slim BMR vase only with Slim/V1L receivers. Both print front-face-down with supports explicitly disabled; four same-height captive side magnets share one audited insertion pause. |

**Installing the inserts:** soldering iron at 230–250 °C, press
slowly and square, stop flush. The bores carry +1.0 mm melt room by
design. Before bringing any joint halves together, install both LM-to-UM M3
inserts through the rear/mating openings of the individual UM print and both
UM-to-tweeter M3 inserts through the rear/mating openings of the individual
crescent. Every Ø4.6 x 4.0 receiver must remain fully surrounded by its local
Ø9.8 functional boss and retain its 1.9 mm solid front floor; reject any
insert that cracks, laterally opens, moves, or marks the acoustic face.

### Obi-Wan structural screens

The fused bridge web and four-hole group use a conservative **4.0 kg**
installed mass, y=230 mm center of mass, and 70 mm rear offset.
Bambu's published XY flexural
strength is 65 MPa for PLA Tough+; this screen deliberately uses much
lower 18 MPa short-duration and 8 MPa sustained allowables for printed
anisotropy, stress concentrations, and creep. The softly blended plate has a
62 mm insert core and occupies z=5.3..18.3, ending at the deepest existing
LM-pad rear datum with no added rear projection. The structural model deducts
the complete Ø9/Ø8.2/Ø6.0 D20-packed entry lumens and credits only a
**38.8 × 13.0 mm** section; exact 0.01 mm sampled cuts retain at least 45.73 mm.
Its in-plane/rear section moduli are **3261.8/1092.9 mm³**. Conservatively
summing the two bending stresses gives about **4.40/13.19/21.99 MPa** and
safety factors **1.82/1.36/0.82** at sustained 1g/8 MPa, transient 3g/18 MPa,
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

The floor state has no separate support. Its LM carrier owns a full-depth W64
stem, W64 × 18.3 foot over z=−150..18.3, convex constant-thickness Option-B
transition with a 41 mm minimum centreline radius, three buried lanes and
rear NL8 panel. World floor Y=0 puts the LM axis exactly **200.981 mm** above
the floor. A closed-form rectangle-minus-lumens screen deducts the complete
Ø9 LM, Ø8.2 UM and Ø6 shared-T lane sections from the root and uses the
same 4.0 kg/y=230/rear-offset=70 mm load model:

| Material | Vertical 1g/3g/5g SF | Anchored lateral 1g/3g/5g SF | 1g diagnostic deflection | Result |
|---|---:|---:|---:|---|
| Bambu PLA Tough+ | 3.05 / 1.97 / 1.18 | 4.22 / 2.73 / 1.64 | 1.18 mm | analytical pass |
| Bambu PLA Basic | 4.39 / 2.78 / 1.67 | 6.09 / 3.85 / 2.31 | 1.05 mm | analytical pass |
| Bambu PLA Lite | 2.69 / 1.73 / **1.04** | 3.73 / 2.40 / 1.44 | 1.40 mm | **FAIL at vertical 5g; provisional data** |
| Bambu PLA Matte | 2.78 / 1.79 / 1.08 | 3.85 / 2.49 / 1.49 | 1.49 mm | analytical pass |
| Bambu PLA Silk+ | 3.23 / 2.09 / 1.25 | 4.47 / 2.90 / 1.74 | 1.17 mm | analytical pass |

These use deliberately derated project allowables, not unmodified Bambu
coupon values. PLA Lite is provisional because no product-specific official
TDS was available. Source records are Bambu's
[Tough+ V3](https://store.bblcdn.eu/s8/default/f0874452d01249dba4ab6fc68ca972e4/BambuPLA_Tough_TechnicalData_Sheet_%282%29.pdf),
[Basic V3](https://store.bblcdn.eu/s8/default/073e722a4aa44f7cbfdc419d597475cc/Bambu_PLA_Basic_Technical_Data_Sheet.pdf),
[Matte V3](https://store.bblcdn.eu/s8/default/82bab351a9494e318ab485f7c31a01b3/Bambu_PLA_Matte_Technical_Data_Sheet.pdf),
[Silk+ V1](https://store.bblcdn.eu/s8/default/d0de0f57694b406dbf3e9b2345b7dbb9/Bambu_PLA_Silk__Technical_Data_Sheet.pdf),
and the provisional
[PLA Pure comparison sheet](https://store.bblcdn.com/s7/default/ecb663b46ebb4fb984786d33befb8d2f/PLA_Pure_TDS.pdf)
used as a fail-closed Lite proxy. The screen is **closed-form analytical work, not FEA,
certification, or release authority**. All stresses include an explicit
**1.25 root geometry/model factor**. PLA Lite fails the 1.05 vertical-5g
threshold and is not accepted; the other four meet the 2.0/1.5/1.05 at
1g/3g/5g and ≤2.0 mm at 1g only with the required **100% local-solid modifier
through the complete stem/root**. Sparse infill receives no structural credit.
Magnets and both optional concealed split pins/sockets receive 0 N structural
credit.

The two LM-to-UM insert-fastened ears and the two tweeter-interface ears carry
only the upper assembly, not the LM/stand.
The ledgered 0.43 kg MU + 0.20 kg tweeters plus printed/hardware allowance
gives a **0.85 kg** case over conservative 120 mm plan and 70 mm rear
levers. Both D4.6-receiver interfaces co-govern with contact factors of about
**2.85/2.14/1.28** at 1g/3g/5g; the M3 screw-tension factor is about **1.28**
at 5g. Those numbers are analytical screens, not qualification of either
heat-set installation, receiver wall, or 1.9 mm front floor. Each interface
reaches approximately **393.9 N per insert** in the 5g pullout-demand case
and therefore remains subject to physical pullout qualification. Magnets
contribute 0 N to every
calculation.

These screens cover the modeled bridge/integral-root reactions, minimum
printed transfer sections, ear neck/net/bearing areas and M3 shear. They
do **not** independently qualify the NL8 panel, stock bridge,
installation substrate, real insert process, or a changed print orientation.
Those items remain inside the system proof-test boundary.

They also do not transfer automatically from the canonical monolithic LM to
the optional LM keyed split. Its two concealed Ø1.60 +Y pins and
right-round/left-X-relieved blind sockets are registration aids with no
standalone retention/load credit. The round+relieved pair prevents redundant
pitch constraint. The horizontal pins are four nominal 0.4 mm nozzle widths
and their sockets sit in small exterior lands outside the LM recess. That
print form remains pending until sliced pin/land/wall paths, actual U22 fit,
process-matched fit, full seating, front-datum coplanarity, UM/T route-seam
continuity and cable pull-through, and the complete driver-installed 1g/3g/5g
proof have passed and been recorded.

This calculation is a screening model, not permission to hang drivers
from an untested print. Before service, every final combination of
filament batch, slicer settings, frame, carrier, inserts, and fasteners must
pass a documented physical proof test through the distributed 4 kg
sustained-1g, 3g and 5g bridge/integral-stand cases in the governing normal and
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
permanent movement or damage. The integral floor candidate must additionally
hold **2× service load for 24 h at 35 °C**, with no crack or whitening and
unloaded residual set no greater than **0.5 mm or 10% of loaded deflection**,
then hold **1.5× service load for at least 168 h** for the creep gate.
Record the exact per-state candidate identity, print/insert process,
fixture, load history, temperature, deflection, damage inspection and
release signoff in `obiwan_physical_qualification.md`.

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

**Anti-tip is mandatory.** The W64 foot reaches material-strength limits only
after the assembly has already become unstable: calculated free-standing tip
thresholds are about **0.139g lateral, 0.348g rearward, and 0.384g forward**.
Install a positively attached tether or anchor before fitting valuable
drivers or putting the speaker into service. The foot and magnets are not a
safety restraint.

## Orientation — the setting that matters most

**Proud family:** print every baffle piece front face down. The front plane is
the reliable datum while V1/V1L sculpt or step the rear. For the
floor-stand `piece_bottom` (223.8 × 125 × 168.3), front-down leaves the
150 mm foot rising as a self-standing wall. Smooth/satin PEI gives a
clean front; textured PEI gives uniform grain.

**Obi-Wan:** every released printable baffle/acoustic part also prints
front-face-down. This includes the floor/no-floor monolithic LM, both optional
keyed LM halves, the UM carrier, tweeter crescent, and flat/graded segments. Only an
in-plane XY rotation about the bed normal may be used; the former 45° X tilt
and floor-face-down keyed-bottom
orientations are not valid for captive-magnet insertion. The floor-state
LM+integral stand is therefore a large-format print. Revalidate the actual
front-down footprint against the selected printer instead of relying on the
obsolete ≤220 mm tilted-orientation figures. The optional LM keyed split
replaces that monolithic LM with two front-down prints. On a P2S, this split
is mandatory: both canonical LM monoliths are approximately 236.41 x 313.75 mm
front-face-down and are not P2S-printable. Do not scale, clip, tilt, or use a
virtual bed. Two symmetric Ø1.60 pins point +Y normal to the seam and engage
2.40 mm; the right socket is round Ø1.84 and the left is X-relieved to
1.96 × 1.84 mm. Both blind sockets are 2.65 mm deep, retain 0.12 mm radial and
0.25 mm end clearance, and preserve at least 0.50 mm local radial/end wall in
small exterior lands outside the LM recess. They preserve at least 0.05 mm
recess and 0.13 mm conservative W22-flange plan clearance. Their worst-case
reach is R114.4036: 1.4036 mm beyond structural R113.0 but only 0.6036 mm
beyond the finalized R113.8 visible fairing. Never mix either half with the
monolithic LM. Flat/graded include matching 0.25 mm hidden interface pockets for
these lands; printed fit remains coupon-qualified. Each horizontal pin is four nominal nozzle widths; reject a
slice missing either complete pin, either land, or a continuous socket-wall
path.
With a monolithic LM, the same pockets remain as small hidden local reliefs;
the three magnetic datums and primary retention geometry are unchanged.

For the keyed split, the shelf also provides the mutually exclusive
`obiwan_01_02_03_04_LM_UM_combo_no_floor_stand` and
`obiwan_01_02_03_04_LM_UM_combo_floor_stand` single-plate alternatives to
the corresponding four individual 01/02/03/04 projects. They do not
regenerate or fuse geometry: four released, same-state front-down STLs are
translated to locked positions inside one Bambu object. Each project carries
the three released state-specific duct blockers, all four support fields
pinned globally and per object, and one six-magnet pause at Z=5.96 mm. Both
combined core projects use the hash-pinned saved TINMORRY PETG-GF preset and
eight walls. The no-floor project uses 40% gyroid globally plus a 100%-solid
zig-zag parameter modifier through the complete `01a` bridge/root; the floor
project preserves the integral-floor bottom's global 100% zig-zag contract.
Promotion requires exact four-volume
project/STL equivalence, actual support under every carrier, no support under
the tweeter footprint, and zero support-bead collisions against every LM/UM
functional duct. Never print a combined plate with its individual 01, 02, 03,
or 04 files, and never mix the two stand states.

For the flat and graded B wing splits, the shelf also provides
`obiwan_flat_wings_split2_combo` and
`obiwan_graded_wings_split2_combo` as mutually exclusive single-plate
alternatives to their four individual B projects. Each rigidly rotates and
translates the four exact released front-face-down STLs without
regenerating, changing, or fusing their geometry. The locked footprint is
248.813 × 248.815 mm on the 256 mm bed, with 3.587 mm minimum inter-part gap
and 3.592 mm minimum edge clearance. Use the ready project: it has 30% gyroid,
all four support fields pinned off globally and per object, zero support
feature blocks, and one Z=5.96 mm pause for all six wing magnets. Promotion
requires exact 3MF/STL equivalence (15,692 triangles for flat; 958,546 for graded),
an identity outer
placement, and passing toolpath audits for all four captive-cavity regions.
Never print either plate together with its four individual B files.
The flat/graded wings and shoulders do not carry the LM/UM cantilever load and stay
on the standard Bambu PLA Tough+ profile with six walls; do not apply the
PETG-GF structural-core profile to them.

The UM route is covered only in LM, T is covered in
LM/UM, and their specified rear continuations are free; the short LM lead is
also intentionally free inside a rear-open subtractive clearance, not a
printed micro-duct. Keep generated support out of functional buried-route
mouths/free-cable clearance, the NL8 service cavity, and every rear bump.

Every released nonpolar STL—including all ten production coupons and every
flat/graded segment—must travel with its exact adjacent `<stem>.print.json`. That
record hash-binds the mesh to its X180 plus optional in-bed-Z transform and
origin translation. Treat a missing, extra/orphaned, hash-stale, tilted, or
translation-inconsistent sidecar as a hard stop; run `scripts/check_manifold.py` on
the release STL directory before slicing. Do not substitute a sidecar from
another state or a similarly named piece.

## Bambu Studio profile (0.4 nozzle)

The settings below are the general structural profile. For every captive-
magnet part, the authoritative pause audit overrides the general layer height:
use the pinned **P2S 0.4 mm / 0.16 mm High Quality / Arachne wall** profile and
the exact first-closing-layer Z values in the generated pause manifest. Do not
reuse a 0.20-mm pause height or derive one by scaling.

* **Layer height:** **0.16 mm** for every generated captive-magnet job.
* **Walls: 6 loops maximum** (2.4 mm where geometry permits) for ordinary
  parts. Both standalone Obi-Wan keyed LM bottoms (`01a` no-floor and `01b`
  floor) and both combined `01+02+03+04` core plates use **8 loops** under
  their PETG-GF structural profile. The wall setting makes the material
  around every insert bore, dovetail key, and knife edge fully solid.
  Every 0.45 mm captive-magnet retaining skin is the exception: Arachne must
  slice it as exactly one bounded variable-width bead/traversal, never six
  paths. *Detect thin wall: on* remains pinned and the actual toolpath is
  audited independently. Transverse skin widths have a nominal
  0.42--0.67 mm bound; the audit permits only a 0.005 mm lower-side Arachne
  tolerance (effective floor 0.415 mm), and the pinned release reaches
  0.415656 mm at its narrowest path. The orthogonal 0.45 mm coupon spacing is
  deterministically 0.484336 mm at 0.16 mm layers; angled Obi-Wan LM/UM skins
  reached 0.586 mm and the legacy V1 adaptive inner bead reached the full-run
  maximum of 0.661027 mm. The audit selects a skin by its cavity-facing bead
  edge within 0.06 mm of nominal, then still requires exactly one path in
  every scan bin; a secondary overlapping-path guard rejects anything except
  the bounded same-path V1 edge return. Each transverse station must retain at least 2.0 mm of
  path-width-aware free loading slot (observed full-run range
  2.047--2.083 mm). Axial skin widths remain nominally 0.42--0.65 mm with only
  a 0.000005 mm lower serialization tolerance (observed maximum 0.631 mm), and
  only while the sliced loading aperture remains
  at least D5.0. A two-component axial ring is accepted only as two long,
  non-overlapping cyclic intervals with complementary ray coverage and
  endpoint-local seam anomalies whose two physical junctions retain bead-
  footprint contact within the 0.52 mm connectivity cap—not as disconnected
  arcs or a full ring plus a stray path.
  Top/bottom **6/5** layers; *Ensure vertical shell thickness: All*.
* **Infill:** gyroid, **30 %** for proud mids/vase and the Obi-Wan collars,
  **40 % for the proud piece_bottom in both stand states** (foot standing
  moment ~8 N·m, or bridge bolts), and **40 % for the Obi-Wan fused bridge
  web and tweeter crescent/direct joint ears**. For the no-floor Obi-Wan
  `01a`, use eight walls and a **100% local-solid modifier** through the
  complete W64 stem/root; do not depend on nominal infill through the root or around
  the three buried lanes. *Detect narrow internal solid
  infill: on.* The generated job embeds and audits that **100% zig-zag
  parameter modifier** while retaining 40% gyroid globally. The
  integral-floor keyed-bottom job deliberately uses global **100% zig-zag**;
  Bambu rejects gyroid at 100%.
* **Support:** off for every job except both optional keyed LM halves and the
  UM carrier in both floor states. Those generated projects pin **Enable
  support: on**,
  **On build plate only: on**, **Support critical regions only: on**, and
  **Remove small overhangs: on**, both globally and per object. Each project
  embeds a hash-bound blocker over every functional duct segment owned by that
  carrier. Do not broaden support manually: promotion requires a zero-collision
  audit between deposited support beads and the functional duct volumes.
  Bambu Studio's `floating cantilever` warning is a hard release failure even
  though the CLI describes it as non-critical.
  Every other generated project pins the same four settings off globally and
  per object, then verifies that its G-code emits no support feature.
* **Strength tuning:** nozzle 225 °C (top-middle of the Tough range —
  hotter = better layer adhesion), bed 55–60 °C, **max fan 60 %**
  (overhang fan 100 %), outer wall <=120 mm/s, keep the filament
  profile's volumetric limit (~12–16 mm³/s). Strength lives in layer
  adhesion, not speed. The PETG-GF structural-core jobs instead use the exact
  saved `TINMORRY PETG-GF Profile @BBL P2S` values: 260 °C nozzle, 80 °C
  textured bed, 0.93 flow ratio, and 12 mm³/s maximum volumetric speed. Do
  not copy this material/profile to flat/graded wings or shoulders.
* **Dimensional fits:** proud regular through-thickness dovetails use 0.05 mm
  in-plan female clearance; tune compensation until coupon 2 slides snugly
  through coupon 1 without forcing the seam open. Insert bores
  are nominal. Use *Precise wall: on*, **elephant-foot compensation
  0.15 mm**, and start X-Y hole compensation at **+0.05** for general fit
  coupons only. Captive-magnet artifacts require **0.00 mm**: +0.05 mm was
  observed to delete the 0.45 mm retaining skin.

### Captive-magnet pause procedure

All released magnet-bearing parts print **front-face-down**. The approved
system encloses each actual D5.0 × 2.0 disc in an Ø5.20 × 2.10 cavity with
0.45 mm plastic at each axial face, a vertically open loading cradle, and a
self-supporting 45° roof. There is no magnet glue and no external access
opening. Stock, slim, and Obi-Wan paired transverse stations all use source
Z=**15.10 mm**. Their magnet-free exterior is immutable: no cavity operation
may add a local backing, boss, relief, rear cap, flat, or visible pocket cue.
The mating surfaces are flush with **0 mm physical air gap**. The receiver's
0.05 mm allowance is a solid internal spacing standoff, not an air-gap cutter.
Stock lower stations are therefore **0.95 mm** apart. The `standard_curved`
upper base datum is recessed 0.14 mm inside
the unchanged host, giving **1.09 mm**. Obi-Wan LM-upper and UM ring datums are
0.15 mm beneath their smooth carrier fairings, and the Obi-Wan LM-lower datum
is 0.15 mm beneath its smooth shared shoulder; all three interfaces therefore
give **1.10 mm** to matching flat/graded receivers. The slim upper station is
contained by a broad, symmetric,
smooth rear-taper shelf rather than a local magnet-shaped patch. The D5 × 2
cavities and both 0.45 mm Arachne one-bead skins are unchanged.

Use the exact part/variant rows in
[`CAPTIVE_MAGNET_PAUSE_MANIFEST.md`](../review/captive_magnet_slice_audit/CAPTIVE_MAGNET_PAUSE_MANIFEST.md). Pause at the listed
Bambu Studio marker—the first layer whose toolpath begins closing the roof,
after the last completely open layer—not at a height inferred from CAD alone.
Before starting, mark one pole of every magnet and stage each site in manifest
order. At the pause:

1. From above the paused part (its print +Z side), insert exactly the listed
   count vertically downward along print -Z
   (`print_insertion_direction_xyz = [0, 0, -1]`), with the listed
   local-axis polarity.
2. Confirm every disc lies fully seated below the completed layer and cannot
   protrude into the next-layer nozzle path.
3. Remove tools and loose magnets from the build volume, then resume.

Polarity cannot be inspected or corrected after burial. Mirrored parts do not
automatically use the same visible face. The physically validated coupon
regression for the P2S 0.4 mm nozzle / 0.16 mm profile is the common
**LM/UM Z=5.96 mm** marker; never reuse it for a different geometry or
orientation. V0 also prints front-face-down. Its legacy orphan sites at
`(±46.000, 324.000)` were completely detached from the B2 flare, so—there
being no released mate—the first correction moved both to
`(±37.697, 326.470)`. Do not use that interim pair: the left station violated
the T-route rule, while the outboard right land required a visible rear-bevel
backfill. The release uses symmetric **`(±6.690, 321.290)`** centres below the
D82 cutout and between the seam-B integral-joint stations. The minimum qualified residuals
across the pair are 1.088 mm beyond the cutout rule, 12.847 mm beyond the
nearest-pilot rule, 1.089 mm beyond the grown seam rule, and 18.579 mm beyond
every route rule. Each complete R3.20 land already exists in the immutable
post-bevel host; the cavity operation adds no local keep, backing, boss, rear
block, or visible location cue. The rear axes, 45° conical closures, both
0.45 mm skins, and provisional marked-pole directions remain unchanged. V0
still has no released mating part or pairing polarity.

The floor and no-floor canonical Obi-Wan LM monoliths intentionally have no P2S
pause rows: neither fits the bed front-face-down. The manifest identifies each
as not P2S-printable and binds its four source-identical cavity contracts to
the same-state keyed bottom/top halves. Use only the actual keyed-half pause
rows on a P2S. A monolith proxy-coverage row is not a G-code pause and must
never be entered manually in Bambu Studio.

Print the applicable stable `stl/lx521_coupon_*.stl` files before the
large parts:

1. `lx521_coupon_1_fit_plate.stl` — clearance-grown regular female dovetail,
   stepped M5 bore (Ø6.5 for the first 2.0 mm, then Ø6.4 to the unchanged
   6.8 mm total depth), Ø4.6 insert bore, and the released
   V1-upper captive-magnet regression station. Print
   it front-face-down and use the exact pause generated for its own sliced
   G-code; its former exposed-pocket inspection notch has been deleted so the
   45° roof retains the full post-apex seal. This is an **unpaired** regression
   station: install the marked/N pole along source -Y, which is print +Y under
   its X180/Z0 transform. There is no mating magnet and no attraction claim.
2. `lx521_coupon_2_fit_key.stl` — matching loose full-thickness male dovetail.
   It contains no magnet and does not define a polarity pair with coupon 1.
3. `lx521_coupon_3_fish_entry.stl` — no-foot entry cluster and Ø6.8
   tweeter-pair merge.
4. `lx521_coupon_4_um_outlet_proud.stl` — the real B2 outline and the
   complete stock B2/V1 proud Ø8.2/G1/R14 rear outlet at
   (33.446, 301.492). It does **not** represent the keyed V1L outlet.
5. `lx521_coupon_5_fish_ts_dive.stl` — proud-family tweeter notch
   passage.
6. `lx521_coupon_6_fish_foot.stl` — stand-foot R14 elbow pair.
7. `lx521_coupon_7_recess_seat.stl` — actual Obi-Wan LM-core seat sector
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
12. `lx521_coupon_12_obiwan_closed_bore_bump.stl` — state-specific Obi-Wan
    LM-collar sector around the 300° axis, including the enclosed tunnel and
    full-width solid saddle from conduit roof to the ordinary blind insert
    floor; there is no trapped hollow and no cable is exposed.

Before any captive-magnet production run, also print and slice the dedicated
reference in `coupons/obiwan_ae_embed/`. Its source and README define the proven
cradle/skin/roof implementation and the common 5.96 mm regression marker.
The proud family uses a short curved split grommet that follows the final R14 bore;
print `stock_um_grommet_half_{a,b}.stl` and test it with
coupon 4. That grommet fits B2/C7/V0/V1 only; do not fit it to V1L.
V1L has no separate coupon in this set: its printed `mid_right` is the
dry-fishing and physical service-fit article for the 283-degree exit.
Use `slim_um_grommet_half_{a,b}.stl`, not the proud
halves. The V1L TPU insert follows the keyed R14 with a Ø8 body around a
Ø7.1 bore, enters 2.5 mm, and seats its Ø13 × 2 flange on the z=6.8 rear
face. Test both halves around the measured cable and in the printed
`mid_right`; the analytic model clears the Faston motion box, but the
real terminals and boots remain the release gate.
Obi-Wan has no printed grommet or tunnel clip. Any selected external cable
retention remains **physical-fit pending** and must clear the buried-route
mouth, free cable, driver and Faston service envelopes with the measured UM cable.

* **Proud internal voids:** the cable ducts (Ø3.8–9.3, arched ceilings)
  self-support on the flat pieces. For the floor-stand bottom, preview
  the connector channel's ~38 mm ceiling. The generated projects keep support
  disabled; they are rejected if any support feature is emitted. If a future
  revision enables support, it must first add a hash-bound blocker for every
  duct and pass the support-toolpath-versus-duct collision gate.
* **Obi-Wan collar recesses (front-down = seat floor is a ceiling over
  the bed):** the LM and UM annuli bridge 6.0 / 4.0 mm above the bed.
  Use the generated, audited project rather than repainting support. Its
  remove-small-overhang/critical-region/build-plate-only policy and embedded
  duct blockers are part of the release contract. The seat surface is a
  supported face: expect ±0.1–0.2 roughness — the drivers hide it,
  but FLUSHNESS depends on the real seat depth. Print coupon block
  **7_recess_seat** first, drop the actual driver flange edge into
  it, and caliper front-face-to-flange: adjust `LM_FLANGE_T_MM` /
  `UM_FLANGE_T_MM` in `src/lx521_baffle/flush.py` and rebuild if
  either sits proud or sunk. The datasheets disagree with the measured
  thicknesses (U22 drawing 5.5±0.2 vs 6.0 measured; MU10 drawing
  5.4±0.2 vs 4.0 measured), so re-measure both physical flanges. The
  mirrored Ø6 tweeter tunnel runs in the rear web and does not
  reduce the flush seating surface.
* **Obi-Wan routes:** preview the physical Ø7.0 UM cable inside its Ø8.2
  LM-owned passage and the Ø5.2 T cable inside its Ø6.0 LM/UM-owned passage
  at fine layer resolution. The UM cable must run free behind the UM carrier
  with no printed rear duct; T must run free behind the tweeter crescent with
  no crescent-owned printed arc. The true minimum non-load wall on each
  surviving buried span is 0.8 mm and the seat roof is 0.85 mm. In no-floor
  state verify all three LM/T/UM entries are open inside the D20 support
  opening, with LM above, T lower-left, and UM lower-right. Pull the D7.8 LM
  reference through its nominal Ø9 branch and common R14 handoff; in floor
  state verify the corresponding buried Ø9 continuation through the integral
  stem. Confirm the LM-owned UM/T lumens finish at R112.95 and their covers at
  R113.75 beneath the uninterrupted R113.8 carrier exterior, retaining a
  0.85 mm solid outside skin with no groove. Block support from every
  functional mouth/free-cable span and
  inspect the complete solid-backed rear bumps. Use the routing PNG's nominal
  diametric u-z sections—not only its longitudinal station plot—to confirm each
  conduit roof is joined solidly to its blind-bore floor. Verify the 82.95°
  physical crown crossing has T above UM and retains the 2.00 mm
  physical-cable gap; there is no two-duct separator web to inspect. Then
  follow T around the covered 328°/58° UM-carrier bumps to its flush exit,
  after which it remains free behind the crescent. All six LM insert bypasses
  retain ordinary blind-bore floors and all surrounding saddle material
  remains solid. Any support strand
  left in a tunnel reduces cable space. Measure the actual cables before
  relying on a retainer or strain relief.
* **Obi-Wan terminal gate:** do not treat the STEP proxy as hardware CAD.
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
  Complete `obiwan_physical_qualification.md` against the real hardware and
  structural proof loads for each stand state; the pending record is
  checksum-bound into generated candidate manifests but authorizes no release.
* **Proud seam:** align it on the outline's rear/hidden edges —
  keep it off the V0/C7 knife bevels and the front perimeter.
* **Proud floor-stand foot junction (required):** the foot meets the plate
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

`build/floor_stand/stl/lx521_polar_base_{1of2_base,2of2_rotor}.stl`: print
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

These are non-acoustic measurement-jig parts with no baffle/front-face datum;
they are outside the acoustic build-plate texture contract. Do not X180 either
jig part: that would put the base spigot/detent noses or rotor fence into the
build plate. Accordingly these two floor-only polar STLs are the sole release
exception and intentionally have no `.print.json` sidecars.

## Cable fishing protocol

### Proud family

1. **Dry-fish each piece immediately after printing.** Every duct is
   open at a seam, so a collapsed segment should cost one piece rather
   than an assembled baffle.
2. **Thread during glue-up.** Keep `mid_right` front-face-down on a flat datum;
   slide `mid_left` along local Z to engage seam C, then engage the bottom's
   four seam-A males and lower the vase's two female seam-B pockets onto the
   mid-piece males. The regular dovetails align mating duct mouths while the
   crossings retain funnel relief. Never flex a tooth sideways.
3. Use ~1.5 mm nylon trimmer line or a guitar string as the leader,
   soldered and heat-shrunk to the pair; do not create a taped lump.
   The LM bundle remains the tightest nominal fit. Use only a
   plastic-safe dry lubricant or a small amount of silicone grease.
4. In floor-stand form, fish T2 first (the higher z=9.5 feeder), then T1
   beside it through the shared Ø6.8 step. In no-floor form, both pairs use
   the one Ø6 rear mouth at `(-4.75,55.91)`, matching Obi-Wan; pull them
   sequentially through that shared trunk rather than looking for the deleted
   legacy second T entry.
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

### Obi-Wan

Obi-Wan has a UM passage buried only in LM, a T passage buried in LM/UM, free
rear spans behind UM and the tweeter crescent, and a separate un-ducted LM
lead. Rehearse every cable before the drivers or optional modules hide the
working area:

Before cable fishing, install the two LM-to-UM inserts in the standalone UM
carrier, then dry-assemble LM, UM, and the tweeter crescent on one flat
front-face datum. The complementary full-depth junction webs must meet
with only their designed fit seams; reject a visible triangular/cusp void,
front step, thin membrane, or enclosed pocket at either LM–UM or T–UM. The
central T free-cable mouth is intentional and must remain unobstructed. At
x=±32, verify that each LM Ø3.4 passage is open in the individual LM print,
each UM Ø4.6 x 4.0 receiver is fully enclosed by its local Ø9.8 functional
boss in the individual UM print, and the UM acoustic face remains closed by its
1.9 mm floor.

1. Place the measured D7.8 LM cable over the modeled 20.15 mm free span at 269.5°
   behind the carrier; do not add a printed micro-duct. Confirm it seats in the
   rear-open subtractive clearance, remains clear of the LM/web and, in floor
   state, pulls freely through its Ø9 buried
   integral-stem continuation. Add external retention that does not refill
   the modeled clearance or load the cable termination.
2. In no-floor mode, begin the UM cable at the centered bridge rear mouth
   `(5,82,5.3)`; floor mode begins in the Ø8.2 buried stem lane. Fish it through
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
   begins in the Ø6 buried stem lane. Fish it through the crown above the UM
   cable and the covered 328°/58° pilot bumps. Confirm the LM-to-UM T
   ownership mouths are flush, then let T exit the UM-owned passage and run
   free behind the tweeter crescent. The crescent has no printed cable arc,
   conduit, socket, or horn. If the crescent is used, position the free cable
   before bolting the direct ears at x=±24, y=421.5; do not trap it behind the
   blind-M3 joint.
4. Obi-Wan has no printed UM grommet. Fit only the selected external cable
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

### Proud family

* Glue the seams with 5–30 min **epoxy** (open time to seat the
  dovetail keys). Magnets are not glued: they must already be captive under
  their printed roofs using the manifest pause procedure above. No solvent
  bonding — PLA ignores acetone.
* After the first loud listening session, re-torque all driver screws
  once — that is when the PLA under the flanges finishes settling.
* For V1/V1L: knock-test the assembled trunk before trusting the W22
  on the 11.5 mm section long-term. At ~25 % of stock bending
  stiffness the bridge or integral stand remains important.
* For V1L specifically: inspect and dry-fish its keyed 283-degree outlet
  before seam glue or driver installation. The real MU terminal tabs are
  absent from the reference mesh, so the physical Fastons, boots,
  removal stroke, cable, and dedicated V1L split grommet remain
  mandatory release checks even when the analytic route checks pass.

### Obi-Wan barebone

Before hardware assembly, inspect
`obiwan_split.step` for the mandatory core,
`obiwan_lm_split.step` if the optional two-print LM form is
selected,
`obiwan_attachments.step` for the selected printed
modules, and `obiwan_assembled.step` with
`um_fit.step` for the service keep-clear. The last
two contain reference geometry and are not an instruction to print the
Faston proxy.

Choose exactly one LM print form for each state: either the canonical
`obiwan_core_1_of_2_lm_carrier.stl`, or both
`obiwan_optional_lm_keyed_1_of_2_bottom.stl` and
`obiwan_optional_lm_keyed_2_of_2_top.stl`. The optional seam is at world
Y=172.481 mm with a closed zero-gap planar butt. Before step 4, place both
front faces down on one flat datum. With the bottom stationary, move the top
straight along world -Y and seat both bottom-owned Ø1.60 +Y pins together—one
in the right round socket and one in the left X-relieved socket. Do not twist,
spread, hammer, or use one pin as a hinge. Inspect both fits, full seating,
coplanarity and route-seam continuity, then pull the actual UM/T cables through
both preserved lumen handoffs. The pin/socket lands reach R114.4036—1.4036 mm
beyond structural R113.0 and 0.6036 mm beyond the finalized R113.8 visible
fairing—but add no extra screw or standalone retention/load credit.
Hold registration when lifting the LM from the datum for driver fit-up; the
installed flange and all normal LM fasteners provide the structural splice.
Do not load an unspliced split LM.

For the P2S, choose the two keyed halves: the canonical monolith is retained
for a verified larger-format printer only and is not sliced by the P2S captive
magnet pipeline. The split stations are exact same-state cavity-audit proxies
for the monolith as well as the actual P2S prints; follow their own G-code
pause rows. No monolith pause is synthesized.

1. Complete coupons 7, 9, and 12, then measure the real MU
   terminal carrier, tabs, boots, and cable. Stop if the physical parts
   exceed the modeled service space.
2. Install all six rotated LM carrier inserts at 0/60/120/180/240/300° in
   both states; every site is an ordinary blind carrier bore. Floor mode has
   no secondary support heat-sets or through-clearance sites. Install the four
   UM driver inserts, then install two additional M3 inserts through the
   standalone UM carrier's rear/mating openings at x=±32, y=315.770. Those
   joint inserts belong wholly to the UM print in blind Ø4.6 x 4.0 receivers
   inside local Ø9.8 functional bosses, with a 1.9 mm front floor. Install the
   two crescent M3 inserts through that standalone print's rear/mating
   openings at x=±24, y=421.5; each must likewise sit in a complete local
   Ø9.8 boss with an independently printable 360° receiver wall and 1.9 mm
   front floor. No-floor
   additionally owns four rear-opening bridge inserts in the 62 mm solid web;
   each unchanged 6.8 mm-total bore starts at z=5.3 with a Ø6.5 × 2.0
   rear entry followed by Ø6.4 and leaves the existing 6.2 mm front floor. Set every
   insert square and reject any cracked, loose, or over-melted boss.
3. During printing, bury six D5×2 magnets in the Ø5.20×2.10
   surface-normal captive cavities at the exact manifest pauses. LM uses
   four: preserve the upper ring-radial 64°/116° axes and the validated
   nearest-insert/route clearances, then fit the lower captive pair at the
   shared shoulder's cubic midpoint. The right visible datum is
   `(x,y,z)=(45.285011,89.190370,15.10)` with outward normal
   `(0.706451,-0.707762)`; the left is its exact X mirror. Confirm that these
   coincident floor/no-floor datums remain clear of lower inserts, buried
   routes, and the
   bridge/integral-stand load path. Upper LM and UM use that same source
   Z=15.10; UM keeps the 50.5°/129.5° axes. Verify the R113.0/R51.7 structural rings and continuous exposed
   R113.8/R52.5 side fairings, clipped only inside the existing LM--UM and
   T--UM cusp/service regions, with the 0.40 mm LM--UM inter-carrier gap open.
   Each ring cavity datum is structural radius
   +0.65 mm, 0.15 mm beneath the exposed surface; there must be no
   magnet-local backing, boss, relief, rear cap, flat, or visible cue. Verify
   unchanged 0.45 mm axial skins, a
   continuous loading cradle, and the support-free 45° closing roof in sliced
   preview with Arachne. The mating surfaces must remain flush with zero air
   gap; the receiver's 0.05 mm spacing allowance is solid material. Nominal
   paired magnet-face separation is 1.10 mm at LM-lower, LM-upper, and UM.
   Use the manifest's marked local-axis polarity for
   every carrier and mating receiver; do not infer it from mirroring. These magnets
   align and suppress rattle only;
   assign them **zero load capacity** in every assembly and test. For flat/graded,
   verify three coaxial receiver pairs per side: LM lower, LM upper, and UM.
4. After the two UM-owned joint inserts have been installed and inspected in
   the individual UM print, put both collars front-face-down on one flat
   reference. Engage the two rounded half-lap ears at x=±32.0, y=315.770;
   preserve their 0.20 mm axial gap. Drive two M3 screws from the LM rear,
   through the LM's Ø3.4 clearance bores, into the UM inserts. Tighten evenly
   without bottoming the screws while holding both front faces coplanar. Do
   not add washers/nuts or drill through the UM front floor. Verify
   the 165.100 mm driver-center spacing before loading the drivers.
5. In floor mode, inspect the integral W64 stem/foot and convex R41-minimum
   Option-B transition, fish all
   three buried floor lanes, fit the NL8 receptacle through the rear panel and
   confirm the service cavity remains usable. Install a positive anti-tip
   tether or anchor before loading the assembly; the W64 foot is not a safety
   restraint. In no-floor mode, bolt the stock bridge directly to the immutable
   (±20,20)/(±20,70) pattern in the fused front-flush web; there is no separate
   no-floor keel. Feed UM/T through the plate's centered rear mouths before
   the bridge blocks access. The floor LM has no bridge tail. Confirm every closed
   solid-backed cable bump clears real hardware before tightening. Magnets
   remain alignment/anti-rattle aids, never a load path. Add only the remaining
   optional modules needed. Attach the tweeter
   crescent directly at x=±24, y=421.5, preserve the 0.20 mm axial gap, and
   drive two M3 screws through the UM's standalone Ø3.4 passages into the
   pre-installed crescent inserts; no bolt head
   may break the acoustic front. Obi-Wan has no printed grommet; keep the
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
7. Follow the Obi-Wan cable-fishing protocol above. Confirm every covered
   LM-pad bump, the physical T-over-UM crown crossing, and the covered
   328°/58° T-route bumps follow the modeled smooth curves without a kink.
   Confirm the UM span is free behind the UM carrier and the T span is free
   behind the crescent, with neither deleted duct recreated by support or
   retention. Use the true
   local u-z views to confirm the full solid saddle to each bore floor. Add the
   chosen retention/strain relief (one form is mandatory) and connect the
   Fastons only after this inspection. Cable load must be reacted away
   from the MU tabs.
8. Before entrusting drivers to the carrier, proof-test the finished
   carrier/web-or-integral-stand/insert/fastener system with a dummy 4.0 kg mass through the
   sustained-1g, 3g, and 5g cases in the governing normal and
   rear-moment directions. Follow the documentation and rejection
   criteria in “Obi-Wan structural screens”; magnets receive no credit.
   For floor mode, also complete the 2×/24 h/35 °C proof and the
   1.5×/≥168 h creep gate, and record the installed anti-tip anchor/tether.
   For the optional LM split, install the representative LM driver or an
   equivalent flange-and-all-fasteners splice during the complete 1g/3g/5g
   proof; give both concealed pins/sockets no standalone retention or load
   credit and record the prior slicer-path gate, two-pin fit, full-seat,
   coplanarity, route-seam and cable-pull-through evidence with the result.
   Repeat after any filament, slicer, material, insert-process, or
   service-temperature change.
9. Mount the drivers, apply the same low torques in the table, and
   re-check collar coplanarity and all fasteners after 24 hours and the
   first loud session.

## Release-form print constraints

These constraints moved here from the project README during the product-first restructure. The per-product geometry they refer to lives in [`stock.md`](stock.md), [`slim.md`](slim.md), and [`obiwan.md`](obiwan.md).

- Run the applicable coupons from the list above before committing a full print.
- Proud baffle pieces print front-face-down. Support-disabled projects must emit
  no support features; every support-enabled Obi-Wan keyed LM half and UM
  carrier embeds its generated duct blocker and must pass the final
  support/duct collision gate. A Bambu `floating cantilever` warning blocks
  release.
- The canonical floor Obi-Wan LM with its integral stand and the no-floor monolith
  are not P2S-printable at their approximately 236.41 x 313.75 mm front-down
  footprint. On a verified larger-format machine, print them front-face-down
  and keep support out of buried route mouths, connector cavities, and
  free-cable paths. Preview
  every closed bump and optional add-on separately.
- If the optional Obi-Wan LM keyed split is selected instead, print both halves
  front-face-down with in-plane rotation only. Recheck each generated
  footprint against the actual printer; this is the required P2S form. It replaces,
  rather than accompanies, the monolithic LM. Its two concealed Ø1.60 +Y
  pins and right-round/left-X-relieved blind sockets sit on small exterior
  lands outside the LM recess. Their worst-case reach is R114.4036: 1.4036 mm
  beyond the structural R113.0 ring and 0.6036 mm beyond the finalized R113.8
  visible fairing.
  flat/graded include matching 0.25 mm interface pockets around those lands;
  physical fit remains coupon-qualified.
  Preview the four-nozzle-width horizontal pins, ≥0.50 mm
  socket/end walls and both lands, then qualify their simultaneous straight-
  pull fit and actual U22 clearance with a process-matched coupon/print.
- Flat and graded wing sides each print as lower, middle, and UM segments cut from
  the finalized monolith. The lower segment owns the 7/9/4 mm male dovetail
  into the middle segment; the middle segment owns the 7/8.5/4 mm male
  dovetail into the UM segment. Both female complements use 0.05 mm clearance.
  The clearance collapses to exact closure over the final 2 mm at each exposed
  split endpoint, and neither key may grow the installed plan or depth
  envelope. Both complete keys retain at least 2.0 mm measured exterior plan
  ligament. Qualify the fit on a process-matched coupon before a complete wing.

### Magnet insertion and the print sidecar

Print every released baffle/acoustic part front-face-down, including
non-magnet pieces such as the Obi-Wan tweeter crescent. For magnet-bearing parts,
pause at the listed Bambu Studio marker, which is the first layer whose
toolpath begins closing the roof after the last fully open layer. At the pause, insert the exact
count vertically downward from the +Z side along print -Z, with the polarity
shown in the manifest; ensure every disc is fully seated below the completed
layer, clear the toolpath, and resume. Do not use the
coupon's common **LM/UM 5.96 mm** regression marker for unrelated parts; that
value applies only to matching Obi-Wan coupon geometry on the tested P2S 0.4
mm / 0.16 mm profile.

The sibling `.print.json` beside each released nonpolar STL is mandatory
machine-readable print authority, not optional documentation. It binds the
exact STL hash and size to X180 plus an optional bed-normal Z rotation and the
origin translation. Keep each pair together and rerun `scripts/check_manifold.py`
after copying artifacts; do not print a missing, orphaned, hash-stale, tilted,
or translation-inconsistent pair. The only release STLs intentionally without
this acoustic/front-down sidecar are the floor-state polar base and rotor,
which are measurement jigs with their own flat orientation.
