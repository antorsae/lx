# Obi-Wan Ac/Ae acoustic-wing specification

This document is the complete acoustic-wing contract for the Obi-Wan top
baffle. Only **Ac** and **Ae** are supported. Earlier W-series, B1, B2, frame,
perforated, honeycomb, slot, and chirped-edge Obi-Wan wing concepts are retired
and must not appear in builds, manifests, catalogs, or the wing design map.

## Scope

The wings are removable acoustic-boundary attachments for the Obi-Wan R6F LM
and UM carriers. They reuse the carrier geometry and magnetic interfaces; they
do not modify the driver carriers, mounting axes, cable paths, floor/no-floor
structure, or tweeter crescent.

Both variants accept either the canonical monolithic LM or the same-state
optional keyed LM pair. This is geometric compatibility only; keyed-land fit
remains process-matched coupon-qualified.

The two supported comparisons deliberately share one installed plan and two
qualified print-split options:

- **Ac:** constant solid depth reference.
- **Ae:** the same plan and flat acoustic front, with a weighted rear depth
  field that reduces material toward the exposed edge.

Neither wing has structural load credit. The carrier and its mechanical joints
remain responsible for driver and stand loads.

## Coordinate and geometry contract

- Units: millimetres.
- Installed acoustic/front plane: `z = 18.3`.
- Nominal wing depth: `11.5`, occupying `z = 6.8..18.3` where full-depth.
- Plan family: the finalized straight-taper A envelope from
  `scripts/gen_obiwan_wing_design_map.py`.
- Each physical side is one monolithic design body before print subdivision.
- Left and right geometry must be exact mirrors.
- Floor and no-floor Obi-Wan states expose the same wing-contact outline.
- No wing material may invade driver seats, carrier keep-outs, cable passages,
  the tweeter crescent envelope, or the matched receiver construction
  envelope. Carrier and wing solids contact with zero physical mating gap.
- The upper wing follows the complete released B2/Obi-Wan tweeter-crescent
  profile: the lower arc (`center = (-0.016809, 468.219063)`,
  `R = 51.051679`) followed by the released cubic horn flank. Because the
  wings are exact mirrors while the source profile has micron-scale
  left/right differences, the interface authority is the released plan
  unioned with its mirror and offset by `0.20` normal to the profile. A larger
  conservative proxy circle is forbidden because its oblique projection
  creates an excessive visible slot; extrapolating the lower circle is also
  forbidden because it undercuts the real horn near its tip.

Each physical side contains one hidden local clearance pocket for the optional
split's exterior socket land. The cutter offsets the worst-case X-relieved land
by `0.25` radially and at both axial ends; the right wing uses that worst case
and the left is its exact mirror. The pocket stays wholly between the rear and
front faces, does not reach receiver lands or voids, dovetails, or the exposed
acoustic edge, and remains as a small local relief with the monolithic LM.
Magnet datums and primary retention geometry are unchanged.

The generated design map is `obiwan_wing_design_map.png`. It must show Ac and
Ae only.

## Magnetic interface

Each physical side has three carrier-matched receiver stations:

1. LM lower, normal to the shared curved bridge shoulder at cubic parameter
   `u=0.50`.
2. LM upper, radial to the LM carrier.
3. UM, radial to the UM carrier.

The production interface contract is:

- Magnet: `D5 x 2` N52 disc.
- Captive cavity: `D5.20 x 2.10`.
- Intact axial skin: `0.45` on both carrier and wing.
- Common source plane: `Z = 15.10` for LM-lower, LM-upper, and UM.
- Physical mating gap: `0.00`; the receiver cavity datum has a `0.05` solid
  construction standoff, not an air gap.
- The carrier's structural ring radii remain `LM R113.0` and `UM R51.7`.
  Their smooth exposed side radii are `LM R113.8` and `UM R52.5`; the
  fairings stop only inside the existing LM--UM and T--UM cusp/service
  regions. The LM--UM stop keeps the `0.40` inter-carrier gap open.
- At LM-upper and UM, the carrier cavity construction datum is structural
  radius `+0.65`, which is `0.15` beneath the continuous exposed surface.
  The `0.45` cavity skin is unchanged. There is no local pad, boss, flat, or
  visible cue at a ring magnet station.
- At LM-lower, the right visible shoulder datum is
  `(x,y)=(45.285011,89.190370)` with outward normal
  `(0.706451,-0.707762)`; the left is its exact X mirror. Its cavity datum is
  likewise `0.15` beneath the uninterrupted shoulder.
- Nominal installed magnet-face separation is `1.10` at LM-lower, LM-upper,
  and UM (`0.45 + 0.15 + 0.05 + 0.45`). Ac and Ae place their matching
  receivers from the visible carrier datums.
- Every cavity is wholly buried in the immutable host; neither carrier nor
  wing may show a local pad, box, flat, dent, silhouette change, or other
  magnet-location cue on any exterior surface.
- Magnets are pause-inserted and permanently buried; no external access hole.
- Polarity follows the hash-bound print sidecar/catalog authority.
- Alignment and anti-rattle only: zero structural load credit.

The process reference remains `coupons/obiwan_ae_embed/`.

The lower Ac/Ae outline contains no material below the Option-B vertical
tangent at `Y=74.15`. A G1 cubic starts there with the bend tangent and joins
the released outer flank at the LM-aperture lower tangent `Y=105.981`; this
is the common lower-root contract for Ac/Ae, left/right, and both A/B print
splits.

## Ac: constant-depth wing

Ac is the mass and geometry reference:

- `11.5` rear-field depth wherever material remains; captive-receiver voids
  and the keyed-land interface relief are the only declared subtractive
  exceptions.
- Flat front at `z = 18.3` and flat rear at `z = 6.8`.
- Full-depth receiver roots, carrier lands, joint bands, and tweeter-contact
  cap.
- The exposed edge remains a 90-degree constant-depth edge.

Ac and Ae must use the same installed plan, receiver axes, split options, and
dovetail definitions so depth is the controlled variable.

## Ae: weighted-depth wing

Ae retains the Ac front surface and installed plan. Its rear is one continuous,
single-valued LM/UM/tweeter-weighted field:

- Local depth range: `0.24..11.5`.
- The eligible exposed edge is constant at `0.24`; this is a coupon-gated
  one-layer target for the documented `0.20` layer process.
- LM retains depth longest, UM is intermediate, and the tweeter sheds depth
  fastest.
- There is no fixed run or plateau.
- Maximum audited field slope: `6:1`.
- Every carrier contact, receiver envelope, joint band, support land, and
  tweeter-seat contact remains full-depth.
- The local tweeter top-contact blend is the only declared edge exception.
- The rear field is calculated on the monolithic side before the print split,
  so adjacent pieces inherit the same surface.
- Dovetail interfaces must not introduce a rear-depth discontinuity greater
  than `0.15`.

Ae is experimental until its edge, surface finish, and acoustic behavior have
physical evidence. CAD validation alone does not qualify the one-layer edge.

## Print subdivision

Each side has two alternative front-face-down subdivisions. Option A is the
established three-piece split:

1. `lm_lower`
2. `lm_upper`
3. `um`

Option B is a two-piece split:

1. `lm_lower`, geometrically identical to option A's lower piece
2. `lm_um_upper`, one continuous solid combining option A's `lm_upper` and
   `um`

Option B retains the lower dovetail and its `0.05` female clearance exactly.
The former middle-to-UM seam, key, and clearance are restored as solid wing
material; a B upper containing a hidden slit or multiple solids is invalid.

There are therefore ten STLs for Ac and ten for Ae. The split contract is:

- Lower-to-middle key: lower male, `7/9/4` neck/head/depth.
- Middle-to-UM key: middle male, `7/8.5/4` neck/head/depth.
- Female clearance: `0.05`.
- Exposed seam clearance closes over the final `2.0` at both endpoints.
- Keys remain inside the monolithic installed envelope.
- Dovetails register/interlock in XY but do not independently retain Z.
- Every print must fit a `220 x 220` bed using in-plane rotation only.
- Every STL requires an adjacent hash-bound `.print.json` authority.

## Canonical outputs

For slug `ac` or `ae`, the release transaction produces:

```text
build/wings/<slug>/top_baffle_nd25fw4_obiwan_wing_<slug>.step
build/wings/<slug>/top_baffle_nd25fw4_obiwan_wing_<slug>_assembled.step
build/wings/<slug>/top_baffle_nd25fw4_obiwan_wing_<slug>_assembled_b.step
build/wings/<slug>/obiwan_wing_<slug>_facts.json
build/wings/<slug>/obiwan_wing_<slug>_print_manifest.json
build/wings/<slug>/stl/lx521_top_obiwan_wing_<slug>_<side>_<n>of3_<role>.stl
build/wings/<slug>/stl/lx521_top_obiwan_wing_<slug>_<side>_<n>of3_<role>.print.json
build/wings/<slug>/stl/lx521_top_obiwan_wing_<slug>_<side>_b_<n>of2_<role>.stl
build/wings/<slug>/stl/lx521_top_obiwan_wing_<slug>_<side>_b_<n>of2_<role>.print.json
build/wings/<slug>/review/obiwan_wing_<slug>_<view>.png
```

No other Obi-Wan wing slug or output tree is valid.

## Required validation

The clean release must prove:

- Ac/Ae are the only accepted exporter choices and artifact directories.
- Each monolithic pair contains two mirrored solids.
- The A print assembly contains six valid solids, the B assembly contains four,
  and the transaction contains exactly ten STL/sidecar pairs.
- STEP, STL, facts, manifests, and review images are hash-consistent.
- Every STL is closed and strict-manifold.
- Receiver axes, common source Z, cavities, intact skins, 0.05 mm solid
  receiver construction standoff, zero physical mating gap, and polarity
  match the carrier contract.
- Both Ac/Ae sides clear the exact staged floor/no-floor canonical LM and both
  same-state keyed halves.
- Actual BREP land-to-final-wing clearance is at least `0.25` at both optional
  keyed socket lands.
- Each keyed-land pocket misses receiver lands and cutters, dovetails, and the
  exposed acoustic edge.
- Both dovetails reconstruct the monolithic field with the prescribed
  clearance, ownership, endpoint closure, and minimum ligament.
- The exposed upper T-to-wing boundary retains `0.20 +/- 0.005` normal plan
  clearance to the complete released crescent profile, independently of A/B
  subdivision.
- Every B lower is exactly the corresponding A lower; every B upper contains
  both established upper pieces plus the restored former upper clearance,
  is one valid solid, and reconstructs the monolith with only the lower fit
  clearance.
- Ac is constant-depth.
- Ae meets its depth range, edge constancy, protected full-depth lands,
  monotonic section, slope, and joint-mismatch gates.
- All twenty prints are front-face-down and bed-fit.
- CAD snapshots show no disconnected segment, unintended alternative wing,
  or floor/no-floor interface mismatch.

Physical fit, magnet insertion, cable service, proof loading, and acoustic
measurement remain required before release authorization. In particular, the
printed keyed-LM/wing fit remains process-matched coupon-pending even after the
geometric compatibility gates pass.
