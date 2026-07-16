# V1LF removable acoustic-wing specification

Status: design and experiment specification; no wing is acoustically qualified
until measured.

This document is intentionally self-contained enough for an implementation LLM
to design, generate, validate, and document the first family of removable V1LF
acoustic wings. It does not authorize changes to the released V1LF carriers,
their cable routing, driver positions, no-floor bridge/integral-floor
interfaces, or tweeter crescent.

## 1. Purpose

V1LF deliberately reduces the LX521.4 upper assembly to nearly baffleless LM
and UM carriers. Six flush surface-normal magnets were retained—four
ring-radial and two lower-LM base-side—so removable shoulders or wings can be
attached without rebuilding the load-bearing core.

The wings serve two related purposes:

1. Restore controlled portions of an LX521.4-like outer envelope and determine
   how much of the original measured dipole behavior came from that outline.
2. Explore whether additive manufacturing can create a spatially graded
   acoustic-impedance baffle: neither simply solid nor absent, but intentionally
   more transparent in some frequency ranges and more obstructive in others.

The implementation must produce all fourteen variants in Section 8. A bare
V1LF and the existing full B2 baffle are measurement controls, not substitutes
for those fourteen printed variants. Every variant, including the deliberately
open controls, uses the same realizable three-root skeleton on each physical
side and has acoustic treatment in both the lower LM and upper UM zones.

## 2. Acoustic motivation

The original LX521 mid/tweeter outline is an acoustic interference device, not
cosmetic trim. Front and rear radiation meet around its edges; edge distance,
outline, thickness, driver directivity, and crossover phase jointly determine
the on- and off-axis response. Linkwitz records that the shape was arrived at
empirically through extensive free-field polar measurements and that design
above a few kilohertz becomes educated iteration rather than a dependable
point-source calculation.

V1LF removes much of that empirically tuned outline. This offers possible
benefits--less panel area, fewer stored-energy modes, and less rear obstruction--
but it also changes the LM/UM dipole path lengths, edge diffraction, efficiency,
and crossover summation. The existing LX521.4 equalization and crossover must
not be presumed valid for bare V1LF or any wing.

Removable wings make the core an experimental fixture. Every acoustic geometry
can use identical drivers, centers, carrier seats, cable routes, and support
hardware. Differences can therefore be attributed primarily to the wing,
provided wing mass and carrier vibration are also recorded.

Relevant local background:

- `linkwitz/linkwitzlab_archive.md`, especially the discussions near lines
  6855, 6861, 6920, 15239, and 15243.
- `top_baffle_v2/V1LF_R6F_CAD_BRIEF.md` for the released carrier contract.
- `top_baffle_v2/V1LF_PHYSICAL_QUALIFICATION.md` for unresolved physical gates.
- `top_baffle_v2/VARIANTS.md` and `top_baffle_v2/README.md` for artifact and
  manufacturing conventions.

## 3. Terminology and physical model

In this specification:

- **Acoustically visible/opaque** means a surface presents enough acoustic
  transfer impedance to sustain a useful front-to-rear pressure difference and
  force appreciable radiation around its outer edge.
- **Acoustically transparent** means front and rear pressure communicate through
  the surface with comparatively little impedance.
- **Cell pitch** is the center-to-center spacing of repeated apertures or cells.
- **Active porosity** `phi_active` is actual open projected area divided by the
  patternable acoustic-field area after removing roots, skeleton, perimeter
  band, ties, keep-outs, and seams.
- **Gross porosity** `phi_gross` is that same actual open projected area divided
  by the entire installed side-module plan area. Both values are mandatory;
  an active-field target must never be presented as whole-wing porosity.
- **Effective path length** `l_eff` is physical channel length plus end effects
  and any intentional tortuous extension.
- **Hydraulic aperture size** is the equivalent diameter controlling local air
  motion and viscous loss; it is not merely the visible macro-cell dimension.
- **Knee frequency** is an experimental transition region, not a mathematically
  sharp cutoff.

At approximately 20 degrees C:

- wavelength at 1 kHz is about 343 mm;
- wavelength at 3.5 kHz is about 98 mm;
- wavelength at 4 kHz is about 85.8 mm;
- wavelength at 7 kHz is about 49.0 mm; and
- wavelength at 10 kHz is about 34.3 mm.

One kilohertz is approximately the LM upper crossover region and therefore the
shortest important LM wavelength, not its longest wavelength. The UM is active
well above 1 kHz and the tweeter handoff is near 7 kHz, so a pattern that is
subwavelength at 1 or 4 kHz is not necessarily homogenized at the upper handoff.
The 3.5 kHz row captures the reported upper-mid directivity-change region; the
10 kHz row is a resonance/scattering audit point above the nominal handoff, not
a claim that the wing should deliberately control the tweeter there.

Use the following scale table when selecting and reporting geometry. The phase
column is the free-field phase accrued by one millimeter of path difference;
it does not include local end correction, incidence angle, or material loss.

| Frequency | Wavelength | `lambda/10` | `lambda/4` | Phase per mm |
|---:|---:|---:|---:|---:|
| 1.0 kHz | 343.0 mm | 34.3 mm | 85.8 mm | 1.05 deg/mm |
| 3.5 kHz | 98.0 mm | 9.8 mm | 24.5 mm | 3.67 deg/mm |
| 4.0 kHz | 85.8 mm | 8.6 mm | 21.4 mm | 4.20 deg/mm |
| 7.0 kHz | 49.0 mm | 4.9 mm | 12.3 mm | 7.35 deg/mm |
| 10.0 kHz | 34.3 mm | 3.4 mm | 8.6 mm | 10.50 deg/mm |

The V1LF carrier dimensions impose a hard acoustic lower bound even when a
wing is made visually small. The LM outside radius is approximately
`113/343 = 0.33 lambda` at 1 kHz. The UM outside radius is approximately
`51.7/85.8 = 0.60 lambda` at 4 kHz and `51.7/49.0 = 1.06 lambda` at 7 kHz.
Consequently an added upper solid shoulder is already acoustically large in the
UM/tweeter band; it must not be described as a scaled-down LM-equivalent merely
because its drawing looks smaller. The available wing depth from `z = 6.8` to
`z = 18.3` is 11.5 mm, corresponding to about 12 degrees at 1 kHz, 48 degrees
at 4 kHz, and 85 degrees at 7 kHz before channel end effects.

A useful first-order perforated-sheet model is:

```text
Z_sheet ~= R + j * omega * rho * l_eff / phi_active
```

This model is only a design guide. It shows why cell size alone cannot make a
honeycomb "solid at 4 kHz." A fine, high-porosity mesh can remain transparent.
The controlling variables are porosity, aperture size, channel length,
resistance, leakage, and angle of incidence. A straight perforated sheet is
generally more transparent at low frequency and increasingly obstructive at
high frequency because the air slugs in its apertures have inertia.

For order of magnitude only, a roughly 3 mm effective path at 10 percent local
active porosity can develop significant impedance in the few-kilohertz region.
Moving similar behavior toward 1 kHz normally requires only a few percent local
active porosity, substantially longer/tortuous paths, deliberate resistance,
or some combination.
Making the lower-LM honeycomb simply larger and more open normally moves it in
the wrong direction.

For equal target inertive impedance in two zones, the first-order scaling is
`f * l_eff / phi_active ~= constant`. A 1 kHz LM zone therefore needs
approximately four times the `l_eff/phi_active` of a 4 kHz UM zone. This
relationship motivates the separate aperture-gradient and path-gradient
controls below; it does not prove their knees. For example, at equal pitch and
path length, about 4 percent
porosity at 1 kHz and about 15-16 percent at 4 kHz are a useful matched starting
pair. Viscous resistance and aperture end correction still depend on hydraulic
diameter, so coupon measurements are mandatory.

Periodic patterns can create coherent angle-dependent scattering when their
pitch is no longer small relative to wavelength. A pitch of 8 mm is only about
`lambda/10.7` at 4 kHz and is already about `lambda/6.1` at 7 kHz. A 4 mm pitch
is about `lambda/21.5` at 4 kHz, `lambda/12.3` at 7 kHz, and `lambda/8.6` at
10 kHz. Therefore use pitch no greater than 8 mm only for a nominal 4 kHz
homogenized field, and no greater than approximately 4.5 mm when the same field
is intended to behave as an effective sheet through the 7 kHz handoff. A
larger pitch is an intentional scattering/diffraction variable, not a
homogeneous-sheet assumption. Avoid perfect long-range periodicity. Larger
visible macro-cells may be used only if each macro-cell contains a separately
controlled diaphragm/channel field and the actual hydraulic aperture and
porosity targets remain satisfied.

Path lengths above 8 mm deserve special scrutiny: their approximate open-pipe
quarter-wave frequencies are at or below 10.7 kHz, inside the measurement band,
and a 12 mm path has a quarter-wave near 7.1 kHz. Long paths shall therefore be
distributed or damped, modeled with end corrections, and rejected if coupon
data reveal a dominant high-Q feature.

## 4. Scope and non-goals

### In scope

- Left and right removable wing modules.
- LX521.4/B2-like and intermediate outer outlines.
- Solid, perforated, graded, open-honeycomb, and tortuous acoustic surfaces.
- Matching receiver magnets and carrier-conforming registration saddles.
- Canonical monolithic CAD plus bed-fitting derived print splits.
- Printable acoustic-process coupons.
- STEP-first artifacts, STL exports, assemblies, review sheets, parameter
  manifests, deterministic tests, and acoustic/modal test instructions.

### Out of scope

- Moving or resizing any carrier magnet or driver fastener.
- Adding holes, keys, clips, or load paths to the released LM/UM carriers.
- Changing driver centers, clocks, seats, or the LM-to-UM ears.
- Changing the buried cable routes, free cable spans, no-floor bridge,
  integral floor stand, or tweeter crescent.
- Assigning any driver, bridge, stand, or transport load to the wing magnets.
- Claiming the stock crossover/EQ remains correct.
- Treating simulated diffraction, calculated sheet impedance, or a pretty
  honeycomb render as acoustic qualification.
- Sealed internal air pockets, uncontrolled quarter-wave tubes, or hollow
  trapped volumes.

## 5. Authoritative geometry and coordinate system

Units are millimeters. Preserve the existing project frame:

- XY is the baffle/front plane.
- +Z points toward the acoustic/front face.
- Acoustic front datum: `z = 18.3`.
- V1LF rear carrier datum for wing clearance: `z = 6.8`.
- LM center: `(0, 200.981)`.
- UM center: `(0, 366.081)`.
- LM carrier outside radius: `113.0`.
- UM carrier outside radius: `51.7`.
- Nominal LM-to-UM ring gap: `0.4`.

The implementation must import these values from current source rather than
copying them into a second unlinked geometry model:

- `top_baffle_nd25fw4.py`
- `top_baffle_nd25fw4_b2.py`
- `top_baffle_nd25fw4_v1lf.py`
- `top_baffle_nd25fw4_v1lf_attachments.py`
- `top_baffle_nd25fw4_v1lf_route.py`

`top_baffle_nd25fw4_b2.OUTLINE_B2` is the default full-envelope plan boundary.
The exact vector `top_baffle_nd25fw4.OUTLINE` may be emitted as an additional
outline comparator, but must not silently replace B2 in the required variants.

There is one measured geometric exception required for realizability. At the
UM side-magnet axes (129.5 and 50.5 degrees), the R51.7 V1LF carrier lies
outside the historic B2 side outline. A receiver constrained literally to B2
has essentially zero radial land and cannot contain a D5.2 x 2.2 pocket. Every
variant therefore uses the same **common physical-envelope correction**:

- an 18.0 mm radial annular side lobe outside the UM carrier, clipped by the LM,
  tweeter-crescent, center-seam, route, terminal, and service keep-outs;
- one enlarged UM receiver pad and carrier-following arc bridge per side; and
- one outboard inter-carrier spine joining the LM-upper and UM root systems
  without crossing either carrier.

`OUTLINE_B2` remains the acoustic comparator and the limit for the variant
field away from this named correction. The lobe/root/spine correction is
identical in W1-W14, is reported separately in edge-distance and area facts,
and is never counted as a private repair to one variant. Its 18 mm scale is
about 19 degrees at 1 kHz, 76 degrees at 4 kHz, and 132 degrees at 7 kHz, so it
is acoustically consequential and must appear in every measurement/control
interpretation rather than being dismissed as a negligible mounting ear.

The common wing keep-out is the union of both floor and no-floor V1LF states,
including:

- monolithic and optional split LM carrier envelopes;
- UM carrier and both LM/UM half-lap ears;
- no-floor fused bridge web;
- integral floor stem/foot where it intersects the wing Z range;
- optional tweeter crescent and its direct ears;
- installed driver/flange and screw-service envelopes;
- every printed route and modeled free-cable/service envelope;
- the UM terminal/service-fit model.

One wing family must fit both stand states and both LM print forms. Do not make
state-specific wing acoustics unless a later experiment explicitly requests it.

## 6. Immutable magnetic interface

Each physical side has three surface-normal magnetic stations. Obtain their
exact face points and outward normals from `side_magnet_sites()`.

| Side | Carrier station | Carrier face datum | Outward XY normal | Pocket center Z |
|---|---|---|---:|---:|
| Left | LM upper | R113 lip at 116.0 deg | radial | 12.55 |
| Left | LM lower | `(-32, 18)` base side | `(-1, 0)` | 12.55 |
| Left | UM | R51.7 lip at 129.5 deg | radial | 15.10 |
| Right | LM upper | R113 lip at 64.0 deg | radial | 12.55 |
| Right | LM lower | `(32, 18)` base side | `(1, 0)` | 12.55 |
| Right | UM | R51.7 lip at 50.5 deg | radial | 15.10 |

Carrier pockets are immutable `diameter 5.2 x 2.2 deep` pockets along each
listed outward surface normal for actual `diameter 5 x 2` magnets. Carrier
magnets remain flush, never proud. The two lower base-side datums are identical
in the floor and no-floor LM carriers.

Each wing shall provide a matching inward-facing coaxial pocket at every
station on its side. Ac and Ae therefore have exactly three receivers per
physical side: LM lower, LM upper, and UM.

- Pocket: `diameter 5.2 x 2.2 deep`.
- Actual magnet: `diameter 5 x 2`.
- Wing magnet face flush with the wing's carrier-facing datum.
- Exact CAD axis coincidence with the carrier pocket axis.
- Nominal nonmagnetic arc-saddle clearance: `0.20`.
- Exact modeled carrier-to-wing datum gap: `0.00`. A physical assembly may
  measure 0.00 to 0.05 after seating and adhesive cure. Do not
  accidentally inherit the 0.20 mm arc clearance at the magnetic faces; that
  would unnecessarily weaken retention.
- Extra 0.2 mm pocket depth is adhesive allowance, not a bottoming datum.
- Bond polarity must be established against the actual marked carrier; never
  infer polarity from left/right mirroring.
- Receiver material outside the pocket must remain positive and printable.
- Preserve the source carrier's retained skin at every pocket; in particular,
  retain the 3.15 mm front skin at each lower base-side site and the 0.60 mm
  front skin at each UM site.
- No magnet, cup, pod, handle, or registration feature may protrude beyond the
  selected **physical** plan envelope (B2/mid/chirped field plus the common
  correction defined in Section 5) or in front of `z = 18.3`.

The inner edge of each wing shall include shallow carrier-conforming ring and
straight-base saddles with nominal 0.20 mm normal clearance away from the
magnetic roots.
Locally relieve or step those saddles so each opposing magnet pair closes to
the exact zero-gap CAD datum without plastic interference; 0.00-0.05 mm is
only the accepted physical seated result. The nonparallel
three-station geometry and the arc saddles provide registration. Do not add
carrier holes or external snap handles. If a compliant seal is used, make it an
identical, replaceable, documented part for every compared variant; it must not
raise the front acoustic surface.

Magnets receive zero credit for driver, bridge, floor-stand, transport, or
safety loads. They may retain only the lightweight experimental wing in a
controlled test. Record wing mass and prove pull-off/slide retention with an
independent safety tether during early tests.

## 7. Common wing geometry

Every variant shall obey the following rules unless the variant section is
explicitly stricter.

### 7.1 Module topology

- Generate one deterministic left design and derive the right as an exact
  acoustic mirror, including any aperiodic pattern seed.
- Each installed side must be a connected assembly using all three magnetic
  stations on that side.
- Every side must physically span the LM lower root, LM upper root, and UM root,
  and every variant must include a defined lower LM acoustic zone and upper
  UM/tweeter acoustic zone outside that structure. A lower-only or upper-only
  decorative lobe is not a wing variant in this experiment.
- A wing must not bridge mechanically from left to right.
- A wing may approach but must not depend structurally on the tweeter crescent.
- Keep at least 0.20 mm CAD clearance from all common keep-outs.
- No wing geometry may extend rearward of `z = 6.8` or forward of `z = 18.3`.
  This preserves the free UM/T cable and service volume behind the carriers.

### 7.2 Front datum and outer edge

- The material portions of the acoustic front are coplanar at `z = 18.3`.
- No root step greater than 0.15 mm is permitted between wing and carrier front
  in CAD; physical mismatch is a fit-test item.
- A "razor" edge means a printable knife edge, not zero thickness.
- Default final outer-edge thickness: 0.80 mm, equal to two 0.4 mm extrusion
  widths.
- Form the edge by a smooth rear-only taper over at least 12 mm while preserving
  the front datum. Use C1-continuous transitions and avoid a rear shelf.
- The variant acoustic field and final edge must not exceed `OUTLINE_B2` for
  full-envelope variants except for the identical, named UM lobe/root/spine
  correction in Section 5. No other local root island or pattern-specific
  protrusion is permitted.

### 7.3 Structural behavior

- Use one common realizable three-root skeleton for W1-W14. Generate it once,
  Boolean it into every side module, and prohibit variant-specific weakening.
  The nominal LM root-pad plan envelope is 8.8 mm tangential x 6.0 mm radial;
  its hard post-Boolean minimum is 7.6 x 3.4 mm. Because the B2 comparator is
  inboard of the retained UM station, the nominal UM pad is 13.0 mm tangential
  x 8.0 mm radial and its hard minimum is 11.8 x 5.4 mm. The UM pad shall join
  the wider shoulder through the carrier-following arc/fan bridge from Section
  5; tangent contact with B2 is not a connection.
- Join LM lower to LM upper and LM upper to UM with a continuous exterior-side
  spine. The spine, common outer perimeter band, and every root-to-field or
  root-to-band tie are nominally 2.40 mm wide. They may grow where required by
  Boolean robustness or stress, but shall not be narrowed privately for a
  pattern. Preserve at least 1.20 mm positive printed ligament around pockets,
  apertures, split joints, and all load paths unless the immutable restrictive
  pocket-floor allowance in Section 6 explicitly governs a smaller front skin.
- The common skeleton must remain connected after its projected XY geometry is
  eroded inward by 0.60 mm. The eroded result on each physical side must be one
  connected component touching all three root-pad witness regions and the
  exterior band. This is the deterministic proof of a positive 1.20 mm neck;
  visual overlap or centerline intersection is not sufficient.
- A continuous-skin wing must use a minimum 1.20 mm acoustic skin except for
  the explicit 0.80 mm outer knife edge.
- Minimum printed web or cell wall: 0.80 mm; prefer 1.20 mm at load transfer
  paths and print seams.
- Rear ribs or cellular reinforcement must be open to the rear. Never trap a
  closed air volume between a front skin and rear structure.
- Use aperiodic, radial, or fan-like reinforcement rather than a large regular
  drum panel when it does not conflict with the acoustic variant.
- The three magnet roots must connect through the common positive-material
  skeleton. Do not ask a perforated membrane, lattice cell, edge serration, or
  nominal geometric touch to transfer wing handling loads.
- Provide flat, labeled rear sensor locations near each magnet root, the
  widest-span panel region, and one outer-edge region. Labels may live in the
  review drawing rather than be embossed into the acoustic surface.

### 7.4 Pattern behavior

- Apply a variant's acoustic pattern only to the field remaining after the full
  common skeleton, band, ties, root pads, keep-outs, and split seam reserves are
  removed. Pattern clipping and validation shall use the exact projected
  footprint of every round hole, slot, louvre, dog-leg mouth, cell wall, and end
  radius--never only its center point, pitch line, or nominal bounding cell.
- Maintain at least 1.20 mm ligament from every aperture footprint to the common
  skeleton, root pocket, structural tie, split joint, and adjacent aperture
  where that ligament transfers structure. The explicit 0.80 mm minimum is
  allowed only for non-load-bearing local cell walls declared by the variant.
- All apertures and channels must open to both sides or to a documented vent.
- No unintentional blind tube, sealed macro-cell, trapped support volume, or
  unsupported bridge roof is permitted.
- Any straight or tortuous channel deeper than 8 mm must be checked for
  quarter-/half-wave features inside 500 Hz to 20 kHz.
- Deterministic aperiodicity is allowed and encouraged; record the random seed.
- Preserve mirror symmetry between installed left and right wings.
- Calculate and export exact open projected area, `phi_active`, `phi_gross`,
  minimum wall/ligament, aperture hydraulic-size distribution, and effective
  path-length distribution for the LM, blend, and UM/tweeter zones separately.
  Report targets and achieved values side by side; do not infer porosity from
  pitch and nominal diameter after pattern clipping.

## 8. Required variants

All variants use the identical three-root-per-side skeleton and physical-envelope
correction from Sections 5 and 7.3, magnetic roots, common keep-outs, front
datum, material, and print orientation.
Except where the outer edge itself is the declared W12 variable, they also use
the common outer-edge treatment. Every pattern is clipped around the skeleton
using actual feature footprints, and every side contains both lower LM and
upper UM/tweeter acoustic treatment. That commonality is necessary for a
realizable build and a meaningful comparison.

| ID | Artifact slug | Main variable | Design hypothesis |
|---|---|---|---|
| W1 | `solid_lx` | Full B2 envelope, continuous skin | Establish the maximum effect of restoring a thin solid LX-like outline. |
| W2 | `solid_mid` | Intermediate solid outline | Separate outline/path-length effects from permeability effects. |
| W3 | `perforated_4k` | Uniform low-porosity straight channels | Explore low-frequency transparency with increasing obstruction around 3-5 kHz. |
| W4 | `graded_1k_4k` | Spatially graded impedance | Make the LM region effective lower in frequency than the UM/tweeter region. |
| W5 | `open_honeycomb` | High-open-area skeletal screen | Negative/control geometry for mechanical attachment and visually open lattice behavior. |
| W6 | `tortuous_aperiodic` | Offset/tortuous distributed channels | Explore a broader, less periodic transition without a direct line of sight. |
| W7 | `vertical_aperture_gradient` | Constant pitch/path; hydraulic diameter and porosity vary with Y | Test the first-order `f/phi` scaling independently of path length. |
| W8 | `vertical_path_gradient` | Constant mouth field/porosity; path length varies with Y | Test the first-order `f*l_eff` scaling independently of aperture area. |
| W9 | `bimodal_apertures` | Two hydraulic diameters at matched mean porosity | Test viscous/end-effect sensitivity without changing gross open area. |
| W10 | `radial_edge_loaded` | Low-porosity outer field, high-porosity inner field | Test an acoustically loaded distant edge while retaining a leaking inner field. |
| W11 | `radial_edge_released` | High-porosity outer field, low-porosity inner field | Matched inverse of W10 for a softened/released distant edge. |
| W12 | `solid_chirped_edge` | Solid sheet with inward aperiodic edge-distance modulation | Decorrelate upper-band edge arrivals without adding through leakage. |
| W13 | `perimeter_frame` | Solid outer frame plus open interior and common skeleton | Isolate the contribution of the distant outer edge from panel opacity. |
| W14 | `solid_radial_leak_slots` | Mostly solid sheet with sparse radial-LM / tangent-UM through-slots | Test directional distributed leakage in both driver zones without forcing an impossible upper slot. |

### 8.1 W1 `solid_lx`

- Acoustic-field boundary: full `OUTLINE_B2` minus the common wing keep-out,
  partitioned into connected left/right modules, then unioned with the common
  UM lobe/root/spine correction from Section 5.
- Acoustic surface: continuous minimum 1.20 mm front skin.
- Rear reinforcement: open-backed, deterministic, aperiodic/radial ribs tying
  all three magnet roots to the skin and outer perimeter.
- Outer edge: common 0.80 mm rear-tapered knife edge.
- No acoustic perforations.
- This is a thin V1LF wing reference, not an exact stock B2 acoustic clone;
  the existing full B2 assembly remains the stock-depth measurement control.

### 8.2 W2 `solid_mid`

- Construction and thickness: identical to W1.
- Acoustic-field boundary: halfway between the V1LF collar envelope and
  `OUTLINE_B2`, unioned with the unchanged common UM lobe/root/spine correction.
- Construct the boundary along rays from the LM and UM centers:
  `r_mid(theta) = r_core + 0.5 * (r_B2(theta) - r_core)`.
- Blend the LM-derived and UM-derived regions smoothly through the neck; do not
  create a kink, cusp, or local narrowing at the 0.4 mm inter-ring gap.
- Expand only where needed to retain the exact common magnet roots and minimum
  structural webs.
- Record the actual driver-to-edge distance versus angle for LM and UM and
  compare it to W1 and bare V1LF.

### 8.3 W3 `perforated_4k`

- Plan boundary: same full envelope as W1.
- Active sheet physical thickness: 2.0 to 3.0 mm, front at `z = 18.3`.
- Starting `phi_active` target: 8 to 12 percent in the lower LM field and 7 to
  12 percent in the narrow upper UM/tweeter field after subtracting roots,
  skeleton, ties, and edge band; record the lower `phi_gross` separately. The
  one-point upper relaxation reflects exact boundary clipping, not a larger
  pitch or an omitted upper field.
- Starting hydraulic aperture: 0.9 to 1.4 mm, using rounded hexagonal or round
  through-channels compatible with a 0.4 mm nozzle.
- Pattern pitch: no greater than 8 mm in the LM field and no greater than
  4.5 mm above `y = 335` if homogeneous-sheet behavior through 7 kHz is claimed.
- Minimum wall: 0.80 mm.
- Channels: straight through Z, with no blind ends.
- Break long-range periodicity by deterministic size/position modulation while
  maintaining the target local porosity.
- Target, not guarantee: a broad impedance transition centered between 3 and
  5 kHz, with materially greater transmission below approximately 1 kHz.
- Generate at least three coupon subvariants spanning the allowed porosity and
  path-length range before selecting the full-wing field.

### 8.4 W4 `graded_1k_4k`

Plan boundary is the same full envelope as W1. The acoustic property is graded,
not merely the visible cell size.

Define three smooth zones:

1. **LM zone**, nominally `y <= 315`: target `phi_active` 2 to 5 percent and
   `l_eff` 6 to 12 mm. Target transition region: approximately 0.8 to 1.5 kHz.
2. **Blend zone**, nominally `315 < y < 335`: smoothly interpolate porosity,
   hydraulic aperture, and path length with zero slope at both ends.
3. **UM/tweeter zone**, nominally `y >= 335`: target `phi_active` 8 to 15
   percent and `l_eff` 2.5 to 4.0 mm. Target transition region: approximately 3 to
   5 kHz.

If the local wing is too thin to provide the requested LM path length, use one
of these methods, in preference order:

1. an open-backed dog-leg/tortuous channel wholly inside the common Z envelope;
2. a large visible macro-honeycomb with a printed microperforated diaphragm;
3. lower porosity with a shorter path;
4. a locally continuous solid zone.

Do not enlarge lower-LM holes and call that lower-frequency acoustic visibility.
The exported manifest must report the actual impedance-control geometry, not
only the decorative macro-cell dimensions.

### 8.5 W5 `open_honeycomb`

- Plan boundary: same full envelope as W1.
- Projected `phi_active`: 65 to 85 percent in the broad LM field and 45 to 65
  percent in the structurally dense UM lobe, outside roots, common skeleton,
  ties, and edge bands. Target 65 to 80 percent over the combined active field
  and record `phi_gross`, which will be materially lower. The upper range is a
  declared consequence of the realizable D5.2 receiver/spine reserves.
- Straight-through open cells; no continuous acoustic skin.
- Minimum wall: 0.80 mm.
- Preferred pitch: 4 to 8 mm for a nominal 4 kHz control. Any region above
  4.5 mm pitch shall be explicitly labeled a scattering control at 7 kHz, not
  a homogenized sheet.
- Mild deterministic grading is allowed; maximize upper open area without
  violating the receiver, 0.80 mm wall, or 1.20 mm structural reserves.
- This is a negative/control wing. Its purpose is to reveal mechanical loading,
  attachment effects, lattice scattering, and any false assumption that a fine
  honeycomb is automatically acoustically solid.
- Do not tune its mass with closed ballast cavities. Record its naturally lower
  mass and interpret the collar-mode test accordingly.

### 8.6 W6 `tortuous_aperiodic`

- Plan boundary: same full envelope as W1.
- Use two offset aperture planes, louvres, or another printable channel field
  that prevents a direct front-to-rear line of sight.
- Starting `phi_active`: 10 to 20 percent in both acoustic zones; record
  `phi_gross` separately.
- Use a finer hydraulic aperture/pitch pair in the narrow UM field when needed
  to hit that range while preserving the no-projected-line-of-sight offset.
- Effective path-length range: 4 to 10 mm, intentionally distributed by at
  least +/-20 percent to avoid one dominant pipe resonance.
- All paths must remain open, inspectable, support-free, and drainable.
- Do not create Helmholtz cavities unless a later revision supplies an explicit
  target, neck/cavity calculation, damping method, and test coupon.
- Use deterministic aperiodicity and mirror the completed left field to the
  right.
- This is exploratory. Its predicted impedance curve must be labeled as a
  hypothesis until coupon and full polar measurements exist.

### 8.7 W7 `vertical_aperture_gradient`

- Plan boundary and straight-through physical path length: the same full
  envelope and nominal 2.5-3.0 mm active sheet as W3.
- Use a constant 4.0 mm nominal pitch in both acoustic zones. Apply only small,
  zero-mean deterministic position modulation; do not change the zone-average
  pitch to obtain the porosity gradient.
- At `y <= 315`, use nominal diameter 0.90 mm round apertures, which gives about
  4.0 percent ideal cell porosity before clipping.
- At `y >= 335`, use nominal diameter 2.05 mm round apertures. On an infinite
  grid this is about 20.6 percent, but the exact realizable UM field achieves
  approximately 12-13 percent after receiver/spine and full-footprint
  containment. This is partial rather than perfect fourfold compensation.
- Interpolate aperture area, not diameter, with zero slope at each end through
  `315 < y < 335`. Preserve the actual 1.20 mm structural ligaments and report
  achieved `phi_active`/`phi_gross` after exact-footprint clipping.
- The ideal ratios `1 kHz / 0.04` and `4 kHz / 0.15` are intentionally close
  in the first-order inertance model. The achieved upper porosity is lower and
  its residual mismatch must be reported, not hidden. W7 isolates the direction
  of this aperture/porosity scaling;
  it must not also vary channel depth or add tortuosity.
- Include both diameter endpoints and three blend positions in the coupon set.
  Label the target knee as a hypothesis because hydraulic-size-dependent loss
  and end correction do not scale only with projected area.

### 8.8 W8 `vertical_path_gradient`

- Plan boundary: the same full envelope as W1.
- Use one shared 4.0 mm mouth grid and one nominal 1.20 mm hydraulic diameter
  throughout. Hold mouth areal density and `phi_active` constant with Y as
  skeleton clipping permits; report the necessarily geometry-dependent
  `phi_gross` in every zone. Any residual active-zone mismatch must be reported
  rather than hidden by retuning aperture diameter.
- At `y <= 315`, target `l_eff = 10-12 mm`. At `y >= 335`, target
  `l_eff = 2.5-3.0 mm`. Interpolate effective length with zero slope at both
  zone boundaries. The approximately 4:1 length ratio is the isolated
  first-order counterpart to W7's porosity ratio.
- Realize the longer LM path with open-backed, support-free dog-legs or offset
  mouths wholly inside `z = 6.8...18.3`. Every path must remain open at both
  ends, visible in section, drainable, and free of a trapped plenum.
- Apply a documented deterministic +/-10 percent length dither about the local
  target without changing projected mouths. Report quarter-/half-wave estimates
  for every path and reject the geometry if a common high-Q family remains near
  7-10 kHz.
- Coupons shall hold the same mouth field while stepping only path length.

### 8.9 W9 `bimodal_apertures`

- Plan boundary and nominal path: the same full envelope and 2.5-3.0 mm
  straight-through construction as W3.
- Use a 4.0 mm nominal pitch with an equal-by-count deterministic mixture of
  diameter 0.90 mm and diameter 1.80 mm round apertures in each acoustic zone.
  The ideal unclipped mean is approximately 9.9 percent porosity.
- Interleave the two populations with a reproducible blue-noise or equivalent
  nonperiodic assignment. Do not segregate all large apertures into the UM zone;
  each defined LM, blend, and UM/tweeter reporting window must contain both
  populations in the declared ratio.
- Numerically match W3's comparison coupon and full-wing achieved open area,
  path length, and active-field thickness within 1 percent. Report each
  population's count, exact open area, hydraulic diameter, and nearest-neighbor
  distribution.
- W9 tests whether resistance and end effects depend on hydraulic-diameter
  distribution when mean projected porosity is held fixed. It is not a vertical
  crossover-gradient claim.

### 8.10 W10 `radial_edge_loaded`

- Plan boundary: the same full envelope as W1.
- Define normalized field distance between the relevant carrier/skeleton edge
  and the outer plan edge along rays from the LM center below the blend and the
  UM center above it. Blend the two coordinate fields smoothly through
  `315 < y < 335`.
- In the outermost nominal 18 mm field band, use low `phi_active`, initially
  4-6 percent. In the inner field use 14-18 percent. Use straight 2.5-3.0 mm
  paths, pitch no greater than 4.5 mm in the UM/tweeter zone, and smooth the
  radial transition over at least 8 mm.
- The common 2.40 mm perimeter band remains solid. "Edge loaded" refers to the
  less-permeable field immediately inside it, not a larger structural rim.
- W10 and W11 form a matched pair. Use the same multiset of exact aperture
  footprints, channel depths, active-field area, and deterministic seed; change
  only their radial assignment. Solve placement so achieved total open area and
  solid volume match W11 within 1 percent after clipping.
- Apply the loaded/released assignment in both the LM and UM/tweeter zones and
  export radial porosity profiles for each driver center.

### 8.11 W11 `radial_edge_released`

- Use W10's plan, common skeleton, exact aperture-footprint multiset, sheet
  thickness, seed, and radial coordinate.
- In the outermost nominal 18 mm field band, use high `phi_active`, initially
  14-18 percent. In the inner field use 4-6 percent, with the same at-least-8 mm
  smooth transition.
- Preserve the solid 2.40 mm perimeter band and every structural ligament. The
  high-porosity field begins inside that band; apertures never break the outer
  edge or turn the module into unsupported teeth.
- Match W10's exact gross open area and solid volume within 1 percent and report
  any unavoidable mismatch caused by root or keep-out clipping. W11 tests a
  softened/released outer acoustic edge, not a lighter or differently attached
  wing.

### 8.12 W12 `solid_chirped_edge`

- Construction: W1's continuous 1.20 mm minimum solid skin and embedded common
  three-root skeleton, with no acoustic through-apertures.
- Starting acoustic-field boundary: `OUTLINE_B2`. Move that field edge inward
  only; never exceed B2. Keep the Section 5 correction unchanged. Modulate
  driver-to-edge distance with a deterministic, aperiodic chirp of
  4-12 mm radial depth and 8-24 mm tangential feature length. Use smooth C1
  scallops, not printable saw teeth or zero-radius notches.
- Apply a complete chirped sequence to both the lower LM perimeter and upper
  UM/tweeter perimeter. Maintain the common root lands, 2.40 mm band following
  the new edge, 12 mm rear-only taper, and 0.80 mm final knife edge. Reduce a
  local chirp depth if necessary to preserve that buildable section; do not
  silently pinch the band or taper.
- Export edge distance versus polar angle for both driver centers plus the
  spatial spectrum/autocorrelation of the chirp. The 4-12 mm depths are about
  `0.05-0.14 lambda` at 4 kHz and `0.08-0.24 lambda` at 7 kHz; W12 is therefore
  an upper-band edge-arrival experiment, not a 1 kHz path-length replacement.

### 8.13 W13 `perimeter_frame`

- Plan boundary: the same full envelope as W1.
- Retain an 8.0 mm solid frame measured inward normal to the B2 perimeter, plus
  the common three-root skeleton and the minimum number of 2.40 mm radial/fan
  ties needed to connect the spine to that frame. Open all remaining field
  directly front to rear; do not add a perforated membrane or hidden rear skin.
- The 8 mm frame includes the common 2.40 mm structural band. Its exterior uses
  the common rear-only knife-edge taper; its inner boundary is smoothly rounded
  and remains at least 1.20 mm thick.
- The frame and open field must extend through both LM and UM/tweeter zones.
  Report open area both as a fraction of the patternable field and of the whole
  side module. Do not describe this sparse geometry as a homogenized sheet.
- W13 isolates whether the distant B2-like perimeter and its delayed edge path
  matter when most intervening panel opacity is removed. Compare it directly
  with W1 and W5, while retaining its naturally different mass in the record.

### 8.14 W14 `solid_radial_leak_slots`

- Construction: a W1-like continuous solid skin containing sparse rounded-end
  through-slots; the common skeleton remains positive material and is excluded
  from slot generation.
- Orient LM slots in deterministic radial fans about the LM center below the
  blend. In the narrow UM lobe, use compact carrier-tangent C1 arc slots: the
  retained UM receiver arc and inter-carrier spine leave no honest 4 mm radial
  capsule with the required ligaments. Blend or terminate orientation cleanly
  through `315 < y < 335`; no cusp or slot convergence may occur at the neck.
- Starting slot width: 1.20-1.60 mm. Starting projected length: 8-28 mm in the
  broad LM field and 4-8 mm in the narrow UM field, each with a chirped,
  aperiodic distribution. The shorter upper range is a realizability constraint
  imposed by the common UM lobe and its receiver/spine reserves, not a silent
  deletion of upper treatment. Straight-through path length: 2.5-3.0 mm.
  Round every end, preserve at least 1.20 mm structural ligament, and keep each
  full slot footprint clear of roots, spine, ties, split seams, and the common
  outer band.
- Target `phi_active` is 1-3 percent in both acoustic zones. This intentionally
  sparse control tests discrete secondary paths rather than trying to imitate
  W3's distributed sheet impedance. Slots shall not
  open through the outer perimeter; "leak" means controlled front-to-rear area,
  not a weakened edge comb.
- Export slot width/length/orientation distributions and a 2D structure-factor
  or equivalent periodicity check. W14 tests whether sparse directional leakage
  behaves differently from W3's near-isotropic microperforation at comparable
  gross open area.

## 9. Parametric source contract

Create a single source of truth, nominally:

```text
top_baffle_v2/top_baffle_nd25fw4_v1lf_wings.py
```

Use named, serializable parameters. At minimum expose:

```text
variant_id
outline_family
outline_fraction
front_z
rear_limit_z
root_gap
root_pad_nominal_tangential
root_pad_nominal_radial
root_pad_hard_min_tangential
root_pad_hard_min_radial
structural_spine_width
structural_band_width
structural_tie_width
structural_ligament_minimum
connectivity_erosion
outer_edge_thickness
outer_taper_width
skin_thickness
cell_pattern
cell_seed
cell_pitch_range
aperture_hydraulic_range
aperture_population_by_zone
porosity_active_target_by_zone
porosity_gross_target_by_zone
path_length_target_by_zone
zone_boundaries_y
radial_field_parameters
minimum_wall
split_enabled
print_bed_xy
```

The source API shall provide, at minimum:

```text
wing_interface_facts()
wing_keepouts(state)
wing_common_skeleton(side)
wing_skeleton_facts(side)
wing_acoustic_zones(variant_id, side)
wing_plan(variant_id, side)
wing_patternable_field(variant_id, side, zone)
wing_aperture_footprints(variant_id, side, zone)
wing_monolithic(variant_id, side)
wing_print_parts(variant_id, side)
wing_acoustic_facts(variant_id, side)
wing_wavelength_facts(variant_id, side)
wing_review_assembly(variant_id, state)
gen_step(variant_id)
```

Function names may follow existing project style, but equivalent facts and
outputs are mandatory. Labels must be verbose enough to identify variant,
side, segment, acoustic zone, and magnetic station in STEP topology/review
output.

`wing_common_skeleton()` must be one shared implementation used by all fourteen
variants, not fourteen visually similar reconstructions. The pattern API must
return exact planar feature footprints before 3D cutting so ligament,
connectivity, and porosity tests use the geometry that is actually Booleaned.
`wing_skeleton_facts()` must expose pad lands, minimum widths, root/band witness
intersections, and the post-0.60-mm-erosion component count.
`wing_acoustic_facts()` must distinguish nominal cell arithmetic from achieved
active/gross values after all clipping, and must report lower LM, blend, and
upper UM/tweeter zones even when a variant is uniform.
`wing_wavelength_facts()` must derive, not hand-copy, the feature ratios and
path-resonance estimates at the Section 3 audit frequencies.

Do not resolve fit by Boolean subtraction from a tessellated STL. Build from
the authoritative parametric source and exact STEP/BREP keep-outs.

## 10. Print splitting and assembly

The canonical design for each side is monolithic. Derive printable segments
from that finalized BREP after all cells, tapers, roots, and clearances exist.
Never redraw the acoustic field independently in each segment.

- Printer envelope: 256 x 256 mm XY, with at least 3 mm practical edge/brim
  allowance.
- Search Z rotation and prove each segment's bed footprint.
- Use the minimum segment count that fits; two segments per side are preferred,
  but a third is allowed when required by the actual B2 outline.
- Put seams through a solid web, never through a minimum-wall pore field,
  magnet pocket, knife edge, high-curvature outline feature, or service keep-out.
- Use concealed straight cylindrical pins or a rounded tongue/blind-socket
  wholly inside the existing envelope. No external snap handle, latch, screw
  boss, or envelope growth is permitted.
- Default FDM fit clearance: 0.15 mm per side, parameterized and proven with a
  small coupon before printing a full wing.
- The assembled front faces must remain coplanar within a 0.20 mm physical
  target.
- A seam may be reversibly taped or lightly bonded from the rear for an acoustic
  experiment, but the method must be identical across compared variants and
  recorded in the manifest.
- No print seam may create a sealed air pocket or a front-to-rear whistle slot.

Provide both the canonical monolithic STEP and all required printable STL
segments. The monolithic file is the geometric authority; print segments are a
manufacturing option.

### 10.1 Ac/Ae V1L-style dovetail split contract

Ac (constant 11.5 mm depth) and Ae (monotonic LM/UM/T-weighted rear) use three
print segments per physical side: lower, middle, and UM. Their former 0.16 mm
wavy butt-glue/epoxy seam contract is superseded by exactly one
through-local-thickness XY dovetail at each of the two interfaces. This
subsection is specific to Ac/Ae; it does not revise legacy adhesive joints or
the general W1-W14 split experiments elsewhere in this document.

- At the lower-to-middle interface, the lower segment owns the male key. Its
  trapezoid has 7.0 mm neck width, 9.0 mm head width, and 4.0 mm penetration.
- At the middle-to-UM interface, the middle segment owns the male key. Its
  trapezoid has 7.0 mm neck width, 8.5 mm head width, and 4.0 mm penetration.
- The mating female complement is offset by 0.05 mm around the key flanks and
  head. This 0.05 mm Ac/Ae fit overrides the generic 0.15 mm-per-side starting
  value above and must be qualified with a process-matched coupon.
- The split-line clearance must taper continuously to zero over the final
  2.0 mm at both external endpoints. The assembled outside edge and carrier
  contact boundary therefore remain closed, without a front-to-rear whistle
  slot.
- Derive both nominal ownership masks and printable clearance masks from the
  finalized monolithic Ac/Ae geometry. The male tabs, female reliefs, and
  endpoint tapers must remain wholly inside the existing wing plan and local
  depth: allowed envelope growth is exactly 0.0 mm. Preserve at least 2.0 mm
  measured plan ligament from each complete male key to the exterior field
  boundary; the upper A key is the binding location.
- The dovetails slide together along local Z. They provide XY registration and
  in-plane interlock, but no independent Z retention and no standalone
  structural-retention credit. Do not describe Ac/Ae as a fully glue-free
  structural assembly. When handling or an experiment requires Z retention,
  use an identical documented rear tape or light-bond method on every compared
  wing.

Ac/Ae facts and tests must report the two key profiles, male ownership, female
clearance, 2.0 mm endpoint-taper length, measured envelope delta, and the exact
nominal/print-mask reconstruction. They must also prove that each key is inside
the finalized solid web, that the external endpoint gap is zero, and that no
receiver, protected Ae land, carrier contact, or knife edge is cut.

## 11. Manufacturing assumptions

- Material baseline: Bambu PLA Tough+.
- Nozzle: 0.4 mm.
- Acoustic front prints face-down where practical.
- Minimum ordinary wall: two extrusion widths, 0.80 mm.
- Continuous acoustic skin baseline: 1.20 mm.
- Magnet pockets remain `diameter 5.2 x 2.2` for actual `diameter 5 x 2`
  magnets everywhere.
- Hold magnets flush during bonding; do not bottom them in the adhesive-depth
  allowance.
- Keep magnet polarity controlled with a marked master magnet and a bonding jig.
- Pores and tortuous passages must be slicer-resolvable without internal
  support. Validate sliced previews, not only CAD openings.
- Print all acoustic-comparison wings with the same material lot, orientation,
  perimeter policy, and relevant flow settings when possible.
- Weigh every finished segment and assembled side.

## 12. Required artifacts and naming

Use a dedicated tree so common wings are not duplicated as floor/no-floor
manufacturing files:

```text
top_baffle_v2/wings/<variant_slug>/
```

For every required variant emit:

```text
top_baffle_nd25fw4_v1lf_wing_<slug>.step
top_baffle_nd25fw4_v1lf_wing_<slug>_assembled.step
stl/lx521_top_v1lf_wing_<slug>_left_<n>of<m>.stl
stl/lx521_top_v1lf_wing_<slug>_right_<n>of<m>.stl
v1lf_wing_<slug>_facts.json
v1lf_wing_<slug>_print_manifest.json
```

Also emit:

```text
top_baffle_v2/wings/v1lf_wing_variants.json
top_baffle_v2/review/v1lf_wing_<slug>_front.png
top_baffle_v2/review/v1lf_wing_<slug>_rear.png
top_baffle_v2/review/v1lf_wing_<slug>_side_section.png
top_baffle_v2/review/v1lf_wing_<slug>_floor_assembly.png
top_baffle_v2/review/v1lf_wing_<slug>_no_floor_assembly.png
```

Generate printable process coupons for W3-W11 and W14. Coupon geometry must use
the exact same aperture-footprint/field generator, print Z orientation, wall,
and channel construction as the corresponding wing. W7 coupons cover both
diameter endpoints and the blend; W8 coupons isolate path length; W9 includes a
matched uniform-diameter control; W10/W11 coupons preserve their radial matched
pair. Provide a parameterized circular impedance-tube diameter plus a
60 x 60 mm flat process coupon; do not assume a specific test fixture diameter
without recording it. W12 and W13 require manufacturing witness pieces for edge
taper/frame quality if their full-wing print orientation cannot be represented
by an existing coupon, but a small coupon is not an acoustic substitute for
their complete perimeter geometry.

The facts JSON must include:

- source revision and content hash;
- exact common-interface values imported from V1LF source;
- exact common-skeleton pad, spine, band, tie, ligament, and eroded-connectivity
  facts, including witnesses that all three roots are reached on each side;
- exact area/bounds and driver-to-edge effect of the common 18 mm UM lobe,
  enlarged UM pads, arc bridges, and outboard inter-carrier spines;
- material/nozzle assumptions;
- side and segment bounding boxes;
- calculated bed rotation and footprint;
- volume and predicted mass;
- surface area, exact aperture/open projected area, and patternable field area;
- achieved `phi_active` and `phi_gross` by LM, blend, and UM/tweeter zone;
- aperture and path-length distributions;
- complete aperture/slot footprints or a content-addressed exact-footprint
  companion file, plus pitch/nearest-neighbor and periodicity statistics;
- minimum walls/skins;
- outer driver-to-edge distance versus angle;
- relevant feature-to-wavelength ratios at 1, 3.5, 4, 7, and 10 kHz and
  quarter-/half-wave estimates for every path family;
- magnet axes, gaps, polarity record placeholder, and retained skin;
- predicted knee ranges and `f*l_eff/phi_active` comparisons clearly labeled
  as first-order estimates;
- modal/acoustic measurement placeholders.

## 13. Deterministic CAD validation

Add a focused test module, nominally `top_baffle_v2/test_v1lf_wings.py`.
Required checks:

1. Every monolithic side and printable segment is a valid positive-volume
   solid or a deliberately labeled compound of valid solids.
2. Installed left/right wings mirror exactly in plan and acoustic field.
3. All six mating magnet axes coincide with `side_magnet_sites()` and use exact
   `diameter 5.2 x 2.2` pockets.
4. The one shared skeleton reports nominal LM pads of 8.8 x 6.0 mm and UM pads
   of 13.0 x 8.0 mm, with no post-Boolean land below their respective hard
   7.6 x 3.4 and 11.8 x 5.4 mm projected minima. Every relevant pocket/slice
   retains its specified 1.20 mm ligament or the explicit Section 6 restrictive
   skin allowance.
5. Eroding each side's projected skeleton inward by 0.60 mm leaves exactly one
   positive connected component that intersects witness regions at LM lower,
   LM upper, UM, and the common exterior band. The nominal 2.40 mm spine, band,
   and ties are measured from exact geometry rather than construction lines.
6. Every W1-W14 side has positive acoustic treatment/field area in both the
   lower LM and upper UM/tweeter zones, and both zones remain structurally
   connected to all three roots through the common skeleton.
7. No wing alters or intersects either carrier, either stand state, either LM
   print form, the tweeter crescent, fasteners, cable routes, or terminal/service
   envelopes.
8. No geometry lies behind `z = 6.8` or in front of `z = 18.3`.
9. Material front faces lie on `z = 18.3`; outer knife edges are at least
   0.80 mm thick.
10. Minimum wall, skin, root ligaments, pocket floors, and split-joint walls are
   positive and meet their variant limits.
11. Every aperture, slot, louvre, cell, and channel-mouth test uses its exact
    projected footprint including rounded ends and position modulation. No
    footprint intersects the skeleton/seam exclusion or violates its declared
    1.20 or 0.80 mm ligament; center-only containment is forbidden.
12. Variant acoustic-field geometry remains inside `OUTLINE_B2`, and W12's
    complete chirped field perimeter remains inside it. Geometry outside B2 is
    exactly the content-addressed common Section 5 correction, identical in
    W1-W14; the test rejects any additional or variant-specific protrusion.
13. W2 has the recorded 50 percent edge-distance interpolation except where
   explicit root preservation is required.
14. Exact open area, patternable area, `phi_active`, `phi_gross`, hydraulic-size
    distribution, pitch, and `l_eff` fall inside the declared range separately
    for LM, blend, and UM/tweeter zones for W3-W11 and W14.
15. W7 achieves its declared diameter/area endpoints at constant 4.0 mm pitch;
    W8 holds the same mouth field while varying path; and W9 contains both
    aperture populations in every reporting zone and matches its uniform
    comparison's achieved open area within 1 percent.
16. W10 and W11 use the same exact aperture-footprint multiset and match total
    open area and solid volume within 1 percent while inverting radial
    assignment. Their exported radial porosity profiles demonstrate both LM and
    UM/tweeter loaded/released zones.
17. W12 meets its chirp depth/spacing and C1 edge requirements; W13 has the
    complete 8.0 mm frame, connected ties, and direct open field; W14's exact
    rounded slot footprints meet declared width, length, orientation, porosity,
    ligament, and nonperiodicity limits in both acoustic zones.
18. All intended channels are open at both ends; no unintended enclosed void
   remains.
19. No field claiming homogenized behavior violates its frequency-dependent
    maximum pitch. Deterministic seeds reproduce byte-equivalent exact-footprint
    and geometry facts.
20. Every path above 8 mm has exported quarter-/half-wave estimates over
    500 Hz-20 kHz and a traceable coupon member; no generated fact labels the
    first-order estimate as measured.
21. Derived print pieces reconstruct the monolithic geometry within tolerance,
   fit the 256 mm bed after recorded rotation, and add no external feature.
22. Each assembled wing uses all three stations on its side and cannot be
   installed with a left/right or segment-order ambiguity.
23. Generated STLs have zero open edges and zero non-manifold or over-shared
   edges.

Generate and visually inspect STEP snapshots from front, rear, both grazing
rear angles, side, outer-edge close-up, magnet-root close-up, split exploded
view, common-skeleton isolation, 0.60 mm eroded-connectivity overlay, exact
aperture-footprint/ligament overlay, both driver-zone overlays, and
representative pore/channel sections. W10/W11 require a matched-pair overlay;
W12-W14 require edge/frame/slot close-ups. A passing Boolean/test suite is not
a substitute for snapshot review.

Ordinary project builds use the repository's remote CAD execution path. Do not
run full OCC carrier/wing builds locally on macOS unless the user explicitly
selects `LX_CAD_EXECUTION=local`. Use explicit targets rather than a directory-
wide rebuild.

## 14. Acoustic experiment protocol

Geometry is complete only when it enables a controlled measurement campaign.

### 14.1 Controls

Measure, with identical driver units and setup:

1. Existing full B2 assembly.
2. Bare V1LF with tweeter crescent installed.
3. W1 through W14 in the same sequence on left and right speakers.

Use an identical optional root seal/tape method for every wing. Record ambient
temperature, microphone position, gating window, amplifier voltage, wing mass,
and exact artifact hashes.

### 14.2 Driver measurements

- Measure LM, UM, and tweeter independently before applying a crossover.
- Keep the other installed diaphragms and all hardware in the same state.
- Capture on-axis response plus horizontal front and rear polars.
- Preferred angular resolution: 5 degrees through critical transition regions;
  10 degrees is the minimum useful coarse sweep.
- Measure at least 0 to 180 degrees; full 360 degrees is preferred.
- Use dense frequency resolution from 600 Hz through 10 kHz, with explicit
  analysis windows around 1, 3.5, 4, and 7 kHz, while measuring broadly enough
  to observe lower-band behavior, out-of-band peaks, and channel resonances.
- Compare normalized polar curves `L(theta,f) - L(0,f)`, not only raw on-axis
  level.
- Record front-to-rear behavior, crossover-region directivity, EQ magnitude
  required for a flat axis, and any narrow angle-dependent feature.
- For every variant, compare lower LM and upper UM/tweeter effects separately.
  In addition, make the declared matched contrasts: W7 versus W8, W9 versus its
  uniform-porosity control, W10 versus W11, W12 versus W1, W13 versus W1/W5,
  and W14 versus the closest achieved-gross-porosity W3 coupon/wing.

On-axis equalization may compensate smooth sensitivity changes. It cannot fix
irregular polar response, angle-dependent cancellation, panel radiation, or a
crossover directivity discontinuity. Rank variants primarily by smooth spectral
tracking across angle and by crossover compatibility.

### 14.3 Coupon measurements

For W3-W11 and W14, test coupons before committing to full wing prints:

- Measure or estimate airflow/acoustic transfer with a documented two-microphone
  fixture or impedance tube.
- Record complex transmission/phase if the fixture permits it.
- Locate the actual transition region and all narrow resonances.
- Test and report both the LM and UM/tweeter endpoints of every graded field.
  For W7 separate hydraulic-diameter effects from porosity arithmetic; for W8
  resolve the predicted 7-10 kHz path family; for W9 compare bimodal versus
  uniform diameter; and for W10/W11 preserve matched total open area.
- Reject a coupon with support-blocked pores, whistles, large manufacturing
  variation, or an unintended high-Q feature.
- Update the facts manifest with measured, not merely nominal, aperture and
  porosity samples from the print, including measured `phi_active` and derived
  `phi_gross` for the coupon's represented zone.

No calculated `Z_sheet` or knee frequency may be labeled measured.

## 15. Carrier and wing modal test

The wing can improve acoustic path length while turning the carrier/wing into a
secondary radiator. Static strength does not exclude this failure.

Use a very light accelerometer where possible; a piezo contact microphone is
acceptable for locating resonance frequencies but not for calibrated amplitude.

1. Excite one installed driver at a time with a stepped sine or sweep to locate
   suspicious structural peaks.
2. At each peak, use a 10-30-cycle sine burst with Hann/Tukey-shaped attack and
   release. High-Q candidates need a sufficiently long central plateau.
3. Record sensor output and amplifier voltage simultaneously, including the
   decay after excitation stops.
4. Rove the sensor across the declared rear locations: all three roots, widest
   skin/cell span, split seam, and outer edge.
5. Repeat for bare V1LF, every wing, and representative low/high drive levels.

A structural mode is indicated by a narrow normalized vibration peak, a spatial
node/antinode pattern, and ringing after the electrical burst stops. Flag a mode
for rejection or redesign when it lies in an active driver band and coincides
with an SPL/polar anomaly, audible buzz, or magnetic/seam motion. Normalize to
input voltage; acceleration naturally emphasizes high frequency, so inspect
velocity or displacement-derived views when comparing broad bands.

## 16. Experiment ranking and release gates

These are research wings. Do not choose a winner from appearance or on-axis
flatness alone. For every variant report:

- normalized horizontal polar smoothness;
- front/rear and side-null behavior;
- LM-to-UM and UM-to-tweeter crossover compatibility;
- on-axis EQ burden;
- narrow acoustic resonance count and Q;
- collar/wing structural resonance frequencies and decay;
- installed mass and magnetic retention margin;
- print time, material, segmentation, assembly repeatability, and damage;
- sensitivity to reseating/remounting.

Minimum gates before a wing can be called a qualified V1LF option:

1. All CAD, collision, printability, manifold, and fit checks pass.
2. The common skeleton passes exact-footprint ligament, 0.60 mm eroded
   connectivity, three-root, and both-acoustic-zone checks on each side.
3. Actual magnets seat flush with no polarity error, rocking, slide, or buzz.
4. Wing survives repeated installation and removal without carrier damage.
5. No new passband structural/acoustic resonance remains unexplained.
6. Front and rear polar measurements are complete for each active driver.
7. Required coupons have measured transfer/phase data and no unexplained path
   resonance or manufacturing blockage.
8. A new crossover/EQ is derived or the measured evidence demonstrates that a
   named existing network remains suitable.
9. Artifact hashes, print settings, measurements, and decision are recorded.

Failure to meet an acoustic hypothesis does not invalidate a well-documented
experimental variant. W5 in particular is useful even if it proves highly
transparent.

## 17. Implementation sequence for an LLM

An implementation LLM shall work in this order:

1. Read this entire specification and the authoritative files in Section 5.
2. Inspect the final floor/no-floor V1LF core, tweeter crescent, routes, and UM
   service model; do not infer interfaces from screenshots.
3. Implement the shared interface, acoustic-zone, and facts functions using
   `side_magnet_sites()` and exact source keep-outs.
4. Build `wing_common_skeleton()` once: nominal/hard root pads, 2.40 mm
   spine/band/ties, 1.20 mm ligaments, and exact receiver pockets. Before any
   acoustic field exists, prove 0.60 mm eroded connectivity from LM lower
   through LM upper to UM and the outer band on each side.
5. Build and validate W1 around that skeleton. Its keep-outs, receiver roots,
   split strategy, two-zone reporting, and review assembly become the common
   fixture for every other variant.
6. Derive W2 from the same solid construction with only the plan boundary
   changed.
7. Implement one exact-footprint acoustic-field generator shared by W3-W11 and
   W14. It must clip complete apertures/slots around the immutable skeleton and
   calculate achieved active/gross porosity from Booleaned footprints.
8. Implement W3-W6 and their coupons as the baseline straight, graded, open,
   and tortuous families.
9. Implement W7-W9 as controlled studies: aperture/porosity gradient, path
   gradient, then bimodal hydraulic diameter. Generate and evaluate coupons
   before producing their complete fields.
10. Implement W10 and W11 together from one aperture multiset and verify their
    matched open area/volume before accepting either.
11. Implement W12-W14 from the shared skeleton, proving chirped-edge
    containment, connected perimeter-frame topology, and exact radial-slot
    ligaments respectively.
12. Record calculated wavelength, pitch, path-resonance, aperture, porosity,
    and structural facts before any full-wing artifact is labeled complete.
13. Derive print splits from each finalized monolithic BREP.
14. Run explicit remote CAD generation, deterministic inspection, tests, STL
   manifold checks, and mandatory snapshots.
15. Produce the facts/print manifests and annotated comparison sheet with
    lower LM and upper UM/tweeter zones visible for all fourteen variants.
16. Hand STEP artifacts to CAD Viewer for live review.
17. Leave every acoustic conclusion marked `UNMEASURED` until physical data are
    attached.

Do not repair one variant with a private carrier change. If the immutable
interface cannot support a proposed surface, document that variant as infeasible
and revise the wing, not V1LF.

## 18. Definition of done for the initial design pass

The initial design pass is complete when:

- all fourteen required variants exist as parameterized monolithic left/right STEP
  geometry;
- every side has bed-fitting derived print segments and manifold STLs;
- all use the unchanged six V1LF stations with flush `diameter 5.2 x 2.2`
  mating pockets;
- all use the one shared LM 8.8 x 6.0 mm / UM 13.0 x 8.0 mm root system, the
  exact common UM lobe/arc/inter-carrier correction, and 2.40 mm minimum
  spine/band/ties; each side passes the 0.60 mm eroded-connectivity test across
  all three roots and the outer band;
- every variant has verified positive geometry and declared acoustic treatment
  in both the lower LM and upper UM/tweeter zones;
- floor/no-floor, monolithic/split LM, cable, terminal, crescent, and fastener
  collisions are proven absent;
- W3-W11 and W14 have exact-process coupons and calculated acoustic facts;
- every patterned variant reports actual feature footprints, achieved
  `phi_active`/`phi_gross`, hydraulic size, pitch, effective-path distributions,
  and wavelength/path-resonance comparisons; W10/W11 additionally pass their
  matched-pair tolerances;
- common and variant-specific deterministic tests pass;
- required snapshots and annotated sections have been visually reviewed;
- an experiment manifest exists with blank fields for physical fit, modal, and
  polar data;
- no generated artifact is described as acoustically qualified.
