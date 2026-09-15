# Dayton ND25FN-4 waveguide — integrated Obiwan upper

The **Dayton ND25FN-4 waveguide** is the project's third tweeter family,
alongside the Dayton ND25FW-4 face-to-face pair and Tectonic TEBM35C10-4 BMR.
It combines the MU10 upper-mid (UM) carrier and two printed tweeter
waveguides into one organic body. The lower tweeter faces front; the upper
one faces rear. Two removable caps and two retainers provide service access.

![Obiwan upper comparison: ND25FW-4, coaxial BMR, opposed BMR and ND25FN-4 waveguide](../images/generated/iso/rows/obiwan_upper_row.png)

The ND25FN-4 is the rightmost upper, aligned to the same LM joint and scale
as the other three arrangements. These views show the exported printed
parts with drivers and service caps omitted. See the
[three-family selection table](TWEETER_OPTIONS.md) for compatibility.

The ND25FN-4 is a 4-ohm, 25 mm silk-dome element supplied without a
faceplate, intended for custom mounting. Its nominal envelope is Ø41 mm
with a 21 mm depth and Ø34 mm cutout. The ND25FW-4 has a different mounting
arrangement; the two drivers are not interchangeable in these printed parts.
[Dayton Audio specifications and 3D reference](https://www.daytonaudio.com/product/1194/nd25fn-4-1-neo-silk-dome-tweeter-element-4-ohm).

![Actual H2C assembly with matching continuous graded wings](../build/h2c/views/H2C_Dayton_ND25FN4_graded_front.png)

## Compatibility and quantities

Use this upper with **Obiwan only**. No Stock or Slim adapter is currently
provided. It replaces both the regular UM collar and its separate tweeter
crescent. The fused body is identical for the floor-stand and no-floor-stand
configurations; choose the appropriate LM carrier underneath it.

| Part per speaker | Quantity | Selection |
|---|---:|---|
| Obiwan H2C LM carrier | 1 | Floor stand or no floor stand |
| ND25FN-4 fused UM/waveguide body | 1 | Shared between stand states |
| ND25FN-4 service cap | 2 | Included on one accessories plate |
| ND25FN-4 M3 retainer | 2 | Included on the same accessories plate |
| Matching continuous left and right wings | 1 of each, optional | Choose flat **or** graded |
| Dayton ND25FN-4 tweeter element | 2 | Front/rear pair |
| MU10 UM driver | 1 | Existing project driver |

Print one accessories plate per speaker: it already contains two caps and
two retainers. Double the quantities for a stereo pair. Caps and retainers
remain separate service parts.

**Regular Obiwan wings do not match this upper.** The ND25FN-4 wings follow
the curved body and have two Ø6 mm UM contacts per side. Regular wings
have a different upper interface. The H2C matching wing is continuous from
LM to UM; the older P2S delivery used separate upper and lower sections.

## H2C print files

Open the [current H2C file catalog](../to_print/h2c/README.md) and select the
`dayton_nd25fn4` group. It contains separate projects and audited slices for:

- **Tinmorry PETG-GF + PLA:** [file folder](../to_print/h2c/dayton_nd25fn4/petg_gf_pla/).
- **PETG Translucent + PLA Translucent:** [file folder](../to_print/h2c/dayton_nd25fn4/petg_translucent_pla/).

The canonical body is
[`h2c_dayton_nd25fn4_body.stl`](../to_print/h2c/STL/dayton_nd25fn4/h2c_dayton_nd25fn4_body.stl).
The [cap](../to_print/h2c/STL/dayton_nd25fn4/h2c_dayton_nd25fn4_cap.stl) and
[retainer](../to_print/h2c/STL/dayton_nd25fn4/h2c_dayton_nd25fn4_retainer.stl)
are single-part STLs. The full wing filenames begin
`h2c_dayton_nd25fn4_wing_` in the [H2C STL folder](../to_print/h2c/STL/).

Use the prepared `.3mf` or audited `.gcode.3mf` to retain the material mapping,
modifiers, support exclusions and measured magnet pauses. An imported STL
alone does not carry those settings. No ZIP is needed.

The H2C jobs use two **0.6 mm High Flow nozzles**: model material on the
left and PLA interfaces on the right. They use an Engineering Plate with
glue at 70 °C, a 5 mm outer brim and no raft. Earlier P2S AMS slot numbers
are not H2C nozzle assignments. The H2C jobs retain native machine priming
and standby programs; the earlier same-nozzle purge workaround is specific
to P2S.

| Region | Current infill policy |
|---|---|
| Structural UM portion | 100% zig-zag, matching the regular UM |
| Tweeter portion of the same body | 15% gyroid, applied by a modifier |
| Matching wings | 10% gyroid |
| Caps and retainers | 15% gyroid |

Six walls are used throughout. PETG forms support bases; PLA forms three
dense, zero-gap interface layers above and below. Both cap ceilings receive
PLA directly underneath them; retainers print without support. The red
nonprinting volumes exclude supports from the cable gallery, magnet loading
paths and all 12 insert interiors. Actual extrusion is audited after slicing.

The translucent body uses Classic, outer-first walls to reduce exterior
line-width changes around buried magnets. A smooth physical finish remains
a print qualification item. PETG Translucent with PLA Translucent is an
explicit custom-material exception to Bambu's narrower mutual-support guide;
it is not vendor-qualified by that guide. See the generated
[H2C policies and exceptions](../to_print/h2c/README.md#policy-exceptions).

## Magnets and fasteners

| Location | Magnets |
|---|---|
| Fused body | 4 × Ø6 × 3 mm N45, buried at the curved UM shoulders |
| Each matching full wing: upper contacts | 2 × Ø6 × 3 mm N45 |
| Each matching full wing: LM contacts | 2 × Ø5 × 2 mm N52 |
| Obiwan LM contacts | Existing Ø5 × 2 mm N52 |

One body plus a wing pair takes **eight Ø6 × 3 mm magnets**. The Ø5 and Ø6
sizes are not substitutes. The larger upper pockets follow the sloping
surface and preserve an uninterrupted exterior cover. Install the magnets
at the embedded, slice-derived pauses, with matching polarity and full
seating. Do not copy pause heights from older P2S files: H2C wing orientation
and closure timing differ. [Selection and pocket details](../candidates/nd25fn4_crescent/MAGNET_SELECTION.md).

Use the existing **Hanglife HLTI-M3-001, M3 × 5 × 4 mm inserts**: M3 thread,
Ø5 mm metal outside diameter and 4 mm length. The shared printed pilot is
Ø4.6 × 4 mm. The body contains 12 M3 insert sites: four for MU10, two for
LM receivers and six for tweeter retainers. The two retainers use **six
M3 × 8 mm socket-head screws** in total. The original LM/UM M2 tie remains
its documented separate exception. Retain the cap seals, two 53 × 2 mm
O-rings and the specified driver gasket/pad sets.
[Hardware measurements](../candidates/nd25fn4_crescent/hardware_validation.json)
and [insert inventory](INSERT_CATALOG.md).

## Assembly order

1. Print one body, one accessories plate and the optional matching wing pair
   from a single material lane. Install the magnets at the embedded pauses,
   checking polarity against their mates before burial.
2. Remove supports and confirm that the enclosed cable gallery, cap seats
   and insert interiors are clear. Install the twelve M3 inserts in the
   individual body, using the specified Ø5 × 4 mm metal inserts and blind
   Ø4.6 × 4 mm pilots. Keep the common LM/UM M2 tie as its separate hardware
   convention.
3. Rehearse the actual UM terminals and tweeter leads before closing the
   assembly. Route the shared tweeter cable through the covered LM handoff
   and enclosed upper gallery, keeping the UM lead on its own service path.
4. Mate the body to the selected LM at the preserved half-laps. Fit the
   existing LM-to-UM fasteners and service tie, retain the 0.20 mm axial
   clearance, and verify the lower front is flush and leaves the LM face
   uncovered. The H2C LM is one piece; older P2S LM halves must first be
   assembled according to their keyed-split instructions.
5. Fit the MU10 and both ND25FN-4 elements with their specified gaskets and
   pads. Secure the two retainers with six M3 × 8 screws, then fit the two
   service caps and 53 × 2 mm O-rings. Check screw engagement and cap sealing
   on the real parts before final tightening.
6. Fit the optional ND25FN-4 wings and check all upper and LM magnetic
   contacts. Complete the actual driver, cable, material and loaded-retention
   checks recorded by the project before treating the assembly as qualified.

## Geometry and verification

The surround has a broad shallow bowl, front wider than rear, with an
inclined outer wall, rounded lower waist and a smooth UM-to-tweeter loft.
The cable gallery is enclosed inside the surround. Solid backing beneath
the MU10 seat and a covered LM cable handoff are retained.

Installed axes are X lateral, Y up and Z forward. The LM and lower UM front
faces meet at Z18.30 mm. The shared receiver axes, 0.20 mm half-lap clearance,
driver seats and service tie are preserved. Native H2C LM checks verify
both stand states for interference, screw alignment, front-face step and
LM front occlusion. [Interface report](../build/h2c/obiwan_interface_validation.json).

The tweeter axes are 62 mm apart. The lower tweeter axis is approximately
84.57 mm above the MU10 axis. Tweeter depth is 35.8 mm; the UM's projecting
bowl makes the complete body approximately 38.2 mm deep. The tightened
spacing and altered flare/surround require acoustic validation; the shape
alone does not establish directivity or crossover performance.

[Front view](../build/h2c/views/H2C_Dayton_ND25FN4_graded_front.png) ·
[Rear oblique view](../build/h2c/views/H2C_Dayton_ND25FN4_flat_rear_oblique.png) ·
[Original surface and routing evidence](../candidates/nd25fn4_crescent/README.md).
The H2C views use the actual exported geometry, with drivers omitted.

The current catalog reports digital slice qualification separately from
physical qualification. The user has printed an earlier body, but the
H2C hardware/material trial, loaded retention and acoustic qualification
remain pending.

## Source identity and rebuilding

“V4” was the supplied package's revision label, not a separate product name.
Current H2C deliverables use **Dayton ND25FN-4 waveguide** and
`dayton_nd25fn4`. Historical source/archive filenames retain their original
revision strings so the approved input hashes and prior print evidence
remain traceable. The rename does not alter the approved body, cap or
retainer geometry.

The naming/provenance mapping lives in
[`h2c/dayton.py`](../src/lx521_baffle/h2c/dayton.py); the three-family registry
lives in [`tweeter_options.py`](../src/lx521_baffle/tweeter_options.py).
The H2C build produces the continuous matching wings and verifies the
retained meshes against their original sources. From the project root:

```sh
make h2c_prepare
make h2c_validate
make h2c_review
make h2c_docs
```
