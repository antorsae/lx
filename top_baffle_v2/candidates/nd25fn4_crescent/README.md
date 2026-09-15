# ND25FN-4 V4 + UM — reference-shaped surround

One shared printed body integrates the Obi-Wan UM carrier and retained V4 crescent. It fits both floor-stand and no-floor-stand LM assemblies. Use this body with the newly matched upper wings below.

The UM follows the supplied reference through its rounded middle, tighter lower waist and flowing upper shoulders. Its lower edge follows the actual LM top contour and meets the LM front at Z18.30 mm without covering it. The outline was traced using the MU10 driver flange as the scale. The front remains wider than the rear, with a continuous inclined side and a broad shallow bowl face.

The four Ø6×3 mm magnets are buried behind the uninterrupted side surface at the broadest shoulders. Their axes follow the inclined wall. Both upper-wing families have a slimmer backing and matching pockets. The closed internal cable gallery passes inward beneath the driver mounting ledge.

The front-edge nick and rear shoulder dent at the UM/tweeter waist have been removed in the actual STL. The outer transition now follows one continuous radial loft; rear-surface sampling stays within the tweeter footprint. [Before/after STL renders](views/waist_comparison.png) and [16 measured surface sections](waist_validation.json) cover both former defects.

[Reference overlay](views/UM_reference_overlay.png) · [Measured outline comparison](views/UM_reference_outline_comparison.png) · [Reference validation](reference_validation.json) · [Front](views/UM_outline_front.png) · [Side](views/UM_depth_side.png) · [Oblique](views/UM_outline_oblique.png)

![Actual STL outline compared with the supplied reference](views/UM_reference_overlay.png)

## Prepared print jobs

**[Engineering Plate: translucent body and caps/retainers — 560 mm³ purge](print_translucent_changeover/README.md)** use the confirmed **P2S 0.6 mm HF nozzle, PETG Translucent in AMS slot 4 and PLA Translucent in slot 2**, with **glue, a 70 °C bed, 5 mm outer brim and no raft**. These projects use 560 mm³ purge each way and an explicit 12 mm³/s PLA flush. The cap ceilings have three dense PLA interface layers and zero contact gaps; the body keeps its 9.80 mm magnet pause. Bambu's mutual-support guide excludes PETG Translucent and covers PLA Basic, so this material/plate combination remains a custom, physically unqualified setup. The file guide and policy record the applicable settings and exceptions.

**[Separate PETG Translucent + PLA Translucent print files](print_translucent/README.md)** use the confirmed **P2S 0.6 mm High Flow nozzle, PETG in AMS slot 4 and PLA in slot 2**. The `SMOOTH_WALLS` body uses constant-width Classic outer walls printed first to remove the measured line-width change over the magnet covers. The same geometry, infill and magnet pauses are retained. A [small surface test](print_translucent/qualification/README.md) checks the physical finish before another full-body print; caps/retainers and both upper-wing families are included in the separate material set.

**[Sliced P2S 0.6 HF Tinmorry PETG-GF + PLA print files and instructions](print/README.md)** are now available. The [shared body 3MF](print/01_UM_Crescent_V4_06HF_PETG_GF_PLA.gcode.3mf) uses **15% gyroid in the tweeters and the regular UM's 100% zig-zag infill**, with six walls throughout. Matching caps/retainers and both upper-wing families are also sliced. The body includes its magnet pause at 9.80 mm; each wing pauses at 5.96 and 9.80 mm.

The native jobs use [print variants](print/README.md#buried-magnets-and-automatic-pauses) with internal magnet-loading relief. The approved exterior, seats and mounting surfaces are preserved. Use the prepared jobs or those print variants for printing.

The accessories plate supports both cap ceilings with PETG-GF bases and three dense PLA interface layers. [Actual ceiling-support preview and checks](print/README.md#cap-ceiling-supports).

The red volumes in Bambu Studio are nonprinting support blockers. The body project now blocks all **12 insert sites**—four UM driver inserts, two LM receivers and six tweeter inserts—with flat 4 mm-deep M3 tweeter insert bores. Their coverage follows the existing Obi-Wan LM/UM projects. The final audit checks support extrusion inside these bores as well as the cable passages.

The [current LM assembly measurement](print/LM_assembly_sections.png) confirms **flush front faces at Z18.30 mm, zero LM front-face coverage and zero interference** in both stand configurations. The M3 axes remain aligned and the half-laps retain their 0.20 mm assembly clearance. [Orthographic interface render](views/LM_interface_orthographic_3D.png) · [Interactive joint](http://127.0.0.1:3246/?file=nd25fn4_crescent%2Fviews%2FLM_interface_detail.glb).

## Approved design STLs

The [shared UM + V4 design](STL/01_UM_Crescent_V4.stl) fits either LM configuration and replaces the standalone UM carrier and separate crescent. There is one connected material body with four closed magnet cavities. The original approved design STLs below remain available as geometry references; their internal loading roofs precede the print preparation.

| Part | Quantity |
|---|---:|
| [Shared fused UM + V4](STL/01_UM_Crescent_V4.stl) | 1 |
| [Closed V4 cap](STL/02_Closed_Cap_PRINT_TWO.stl) | 2 |
| [V4 tweeter retainer](STL/03_Tweeter_Retainer_PRINT_TWO.stl) | 2 |

Choose the wing family matching your existing lower wings. Reuse the lower wing pieces.

| Family | Left upper wing | Right upper wing |
|---|---|---|
| Flat | [Left STL](STL/wings/V4_flat_left_UPPER.stl) | [Right STL](STL/wings/V4_flat_right_UPPER.stl) |
| Graded | [Left STL](STL/wings/V4_graded_left_UPPER.stl) | [Right STL](STL/wings/V4_graded_right_UPPER.stl) |

The seven direct STL files and their checksums are indexed in [delivery_manifest.json](delivery_manifest.json). No deliverable ZIP is produced. Earlier iterations are archived through [superseded.json](superseded.json).

## Magnets and assembly

Use **eight new Ø6×3 mm magnets** per body plus upper-wing pair: four in the body and two in each wing. The two retained LM pockets in those wings still take the existing Ø5×2 mm magnets. The new UM pockets are **Ø6.20×3.10 mm**; Ø5×2 magnets do not fill them.

Recommended: [Superimanes D-06-03, Ø6×3 mm N45](https://www.superimanes.com/imanes-de-neodimio/discos/iman-neodimio-disco-6x3-mm). The vendor lists approximately 0.99 kg pull. This is a recommendation for the larger curved cover; actual retention through the printed pair is unmeasured. [Magnet comparison and installation notes](MAGNET_SELECTION.md) explain the alternatives and limits.

The four sites lie at polar angles **12°, 168°, −12° and 192°**, with their exterior contact datums at installed depth **Z=13.4 mm**. Each pair's axis follows the actual inclined wall normal, approximately 27° out of the XY plane. The complete pocket—including its loading chimney and closure roof—is placed beneath the free surface. Nominal wing clearance is 0.35 mm. [Depth and magnet checks](depth_magnet_validation.json) record full-cover measurements, signed taper direction, local slope consistency and paired face separation; [wing checks](wing_validation.json) record clearances and preserved LM interfaces.

These are captive, closed cavities and require **pause-and-bury installation** before the slicer closes their roofs. Check polarity, measured magnet dimensions and full seating. The original Ø5×2 coupon does not qualify this larger pocket. The prepared jobs contain the [slice-derived pause heights and insertion map](print/README.md#buried-magnets-and-automatic-pauses); physical fit and retention remain unmeasured.

The caps retain the supplied V4 geometry. The body and retainers now use **six M3×8 socket-head screws**, Ø3.4 mm retainer clearances and Ø4.6 × 4 mm blind insert bores on a Ø50.2 mm circle. Use the existing **Hanglife HLTI-M3-001 inserts: M3 thread, Ø5 mm outside, 4 mm long**. Ø4.6 mm is the printed heat-set bore; 8 mm is the screw length under its head. Use the new body and retainers together. The cap entry, two 53×2 mm O-rings, driver seats and compliant gasket/pad sets are retained. The imported M2 retention coupon is historical; it does not qualify the new M3 hardware. [Measured hardware geometry](hardware_validation.json) · [Current generated policies](../../docs/PRINT_POLICIES.md). LM-to-UM hardware is unchanged.

## Print status

**The PETG-GF and translucent material sets have passed slice checks**, including the two [alternate translucent body and caps jobs](print_translucent_changeover/README.md). Each linked file guide records quantities, material mapping, estimates and embedded magnet pauses. The user reports printing the earlier body with PETG Translucent and PLA. The new changeover calibration has not yet been printed, and finish, fit and retention qualification remain pending.

The body uses a diagonal, front-facing-down pose. The free lip rises nominally 2.4 mm ahead of the fixed Z18.3 driver face; the tweeter itself retains its 35.8 mm depth. Exact installed and print bounds are recorded in [validation.json](validation.json). The prepared slice includes removable support under recessed front areas, a 5 mm outer brim, a purge tower and support blockers keeping the closed cable passages clear. Older standalone UM and upper-wing G-code does not match these parts.

Digital checks do not establish print finish, insert strength, wing retention or acoustic performance. The altered UM contour and steeper lower tweeter flare still require acoustic validation.

## Preserved interfaces and routing

Installed axes: X lateral, Y upward, Z forward; dimensions in millimetres.

| Interface | Retained dimensions |
|---|---|
| LM rear-driven M3 axes | X=±32; Y=315.770102 |
| UM front half-lap | Native mating material Z=12.4–18.3; exterior flush with LM at Z18.3 |
| Blind insert receivers | Ø4.6; floor Z=16.4 |
| Receiver front floor / half-lap gap | At least the native 1.9 / 0.2 |
| LM-to-UM M2 tie | Original axis, bearing plane and screw length; rounded access mouth R0.5 |
| MU10 driver mount | Ø82 opening, Ø98.6 recess, Z14.3 seat and four original mounting pilots |
| UM driver centre | Y=366.081 |
| Opposed tweeter axes | Y=450.648 / 512.648; 62 mm pitch |

The tweeter pair remains 20.13 mm lower than the first retained-package placement. The forward tweeter's lower radial expansion is divided by 1.5, retaining its Ø37.2 throat and 8.8 mm flare depth. The upper flare and all tweeter retention interfaces are unchanged. The nominal 2.4 mm front ligament sets the selected UM/tweeter spacing.

The closed T cable handoff, solid annular material beneath the MU10 seat, rounded M2 screw access and independent UM lead corridor are retained. The independent lead opening behind the lower UM is a functional cable clearance. The curved rear band flows around the exact LM receiver lands, and a direct rear height loft joins the UM to the tweeter with matched slope and curvature. Its intersection with the inclined side is rounded.

The C2 T gallery follows an approximately R45.6 path through the UM, moving in depth to pass below both right-hand driver pilots. Its nominal Ø6 mm lumen eases down to Ø5.4 mm beside the upper pilot to retain wall thickness, then blends into the retained T taper at Y439. That join is approximately Ø5.4 mm, continuing into the original Ø4.8 mm branch. Both centreline and radius meet smoothly. The LM handoff stays at Y326. The full Ø82 driver insertion opening remains clear. Actual bend radius, screw clearance and closed cover are reported in [validation.json](validation.json), [lower-band validation](lower_band_validation.json), and the [no-floor](um_finish_no_floor_validation.json) / [floor](um_finish_floor_validation.json) finish checks.

The raised base apron has been removed. The exposed lower front is Z18.30 mm through installed Y322, then eases into the existing bowl by Y336 with matched slope and curvature. The exact LM contour governs the lower edge; the reference-outline comparison starts at Y322. The upper wings are relieved around this foot while retaining their LM magnets and lower split joints. Receiver screw axes, blind floors, half-lap faces and the M2 bearing remain fixed.

## Review and provenance

[Interactive with flat wings](http://127.0.0.1:3246/?file=nd25fn4_crescent%2Fviews%2Fnd25fn4_flat_wings.glb) · [Graded wings](http://127.0.0.1:3246/?file=nd25fn4_crescent%2Fviews%2Fnd25fn4_graded_wings.glb) · [Body and drivers](http://127.0.0.1:3246/?file=nd25fn4_crescent%2Fviews%2Fnd25fn4_color_review.glb)

Blue is printed material, orange is actual driver reference geometry, grey is LM/wings, and purple in the magnet section marks cavity volumes. MU10 electrical terminals are omitted from its visual reference; checks use the project's conservative physical envelope. GLBs are decimated review copies of the actual STL geometry.

[Top/front/side sheet](views/orthographic_top_front_side.png) · [Complete rear](views/rear_installed.png) · [Magnet section](views/magnet_alignment_section.png) · [Cable section](views/wire_connection_section.png) · [LM seam](views/LM_interface.png) · [Lower rear band](views/UM_lower_band_oblique.png)

The `.print.json` sidecars contain exact installed-to-print transforms and hashes. [build_manifest.json](build_manifest.json) records generator and input hashes. The immutable user-provided retained package is verified against all 271 payload hashes on each build. This is an explicitly STL-focused adaptation; no faceted STEP substitute or release-shelf promotion is generated.

## Rebuild and verify

From `top_baffle_v2`, using the project Python environment:

```bash
../.venv/bin/python candidates/nd25fn4_crescent/rebuild.py
../.venv/bin/python candidates/nd25fn4_crescent/build_wings.py
LX_CAD_EXECUTION=local ../.venv/bin/python scripts/run_memory_guarded.py ../.venv/bin/python candidates/nd25fn4_crescent/validate.py --geometry-only
../.venv/bin/python candidates/nd25fn4_crescent/validate.py --wings-only
../.venv/bin/python candidates/nd25fn4_crescent/verify_m3_stock.py
../.venv/bin/python candidates/nd25fn4_crescent/validate_um_finish.py
../.venv/bin/python candidates/nd25fn4_crescent/validate_lower_band.py
../.venv/bin/python candidates/nd25fn4_crescent/validate_outline.py
../.venv/bin/python candidates/nd25fn4_crescent/validate_waist.py
../.venv/bin/python candidates/nd25fn4_crescent/validate_reference.py
../.venv/bin/python candidates/nd25fn4_crescent/validate_depth_magnets.py
../.venv/bin/python candidates/nd25fn4_crescent/review.py
../.venv/bin/python candidates/nd25fn4_crescent/waist_review.py
../.venv/bin/python candidates/nd25fn4_crescent/depth_review.py
../.venv/bin/python candidates/nd25fn4_crescent/lower_band_review.py
../.venv/bin/python candidates/nd25fn4_crescent/outline_review.py
../.venv/bin/python candidates/nd25fn4_crescent/orthographic_views.py
../.venv/bin/python candidates/nd25fn4_crescent/deliver_stls.py
```

Regenerate the body and wing snapshot jobs, inspect current views, and refresh `snapshot_validation.json` before indexing delivery. `validate_depth_magnets.py` samples complete cavity surfaces and the exterior around all four stations without any flat-pad exemption. The comparison views and depth sections come from the exported triangles. The final index rejects stale geometry, validation or image hashes.
