# ND25FN-4 V4 — prepared P2S print jobs

These are **sliced Bambu Studio projects** for the **P2S, hardened 0.6 mm High Flow nozzle, Tinmorry PETG-GF and PLA support interfaces**. Geometry, material assignments, embedded magnet pauses and toolpaths have passed static checks. A physical print has not yet been performed.

The fused body uses **15% gyroid in the tweeter region** and the regular Obi-Wan UM's **100% zig-zag infill in the UM region**, with six walls throughout. One shared body fits both floor-stand and no-floor-stand LM assemblies.

**LM assembly check: both front faces are flush at Z18.30 mm and the LM face remains uncovered.** The exact mesh inside the sliced body project has zero interference and zero projected front-face overlap with either LM top configuration. Thirteen seam measurements show a maximum depth step below 0.001 mm; M3 axes remain aligned and the receiver half-laps retain 0.20 mm assembly clearance. This verifies the digital assembly; physical fit has not yet been tested. [Measured sections](LM_assembly_sections.png) · [Hash-bound assembly measurements](LM_assembly_check.json).

[Orthographic 3D interface sheet](../views/LM_interface_orthographic_3D.png) · [Front](../views/LM_interface_ortho_front.png) · [Side](../views/LM_interface_ortho_side.png). Blue is the current print body; grey is the existing LM top. These renders show the actual assembled meshes without changing their geometry or spacing.

## Files and quantities

For one speaker, print the body plate once, the accessories plate once, and **one left/right upper-wing pair** matching your existing lower wings. Reuse those lower wings. Flat and graded are alternatives.

| Plate | Print quantity | Estimated time | PETG-GF | PLA |
|---|---:|---:|---:|---:|
| [01 — fused UM + crescent](01_UM_Crescent_V4_06HF_PETG_GF_PLA.gcode.3mf) | 1 | 12 h 37 min | 304.8 g | 23.8 g |
| [02 — two caps and two retainers](02_Caps_TWO_Retainers_TWO_06HF_PETG_GF_PLA.gcode.3mf) | 1 plate; all four parts included | 1 h 37 min | 37.9 g | — |
| [Flat left upper wing](V4_flat_left_UPPER_06HF_PETG_GF_PLA.gcode.3mf) | 1 if flat | 2 h 41 min | 63.0 g | 3.2 g |
| [Flat right upper wing](V4_flat_right_UPPER_06HF_PETG_GF_PLA.gcode.3mf) | 1 if flat | 2 h 45 min | 63.4 g | 3.7 g |
| [Graded left upper wing](V4_graded_left_UPPER_06HF_PETG_GF_PLA.gcode.3mf) | 1 if graded | 2 h 25 min | 53.6 g | 3.7 g |
| [Graded right upper wing](V4_graded_right_UPPER_06HF_PETG_GF_PLA.gcode.3mf) | 1 if graded | 2 h 25 min | 53.6 g | 3.7 g |

Material estimates include the slicer's purge and support usage. Times exclude time spent manually inserting magnets.

Open the selected `.gcode.3mf` in Bambu Studio and use its prepared slice. Map **filament 1 to Tinmorry PETG-GF** and **filament 2 to PLA** in your actual AMS slots. The files retain the orientation, modifiers, support blockers, brim, tower and pauses. Changes to orientation, layer height or geometry require reslicing and checking the magnet pauses again.

**Red cylinders and tubes are nonprinting support blockers.** The body includes bore-sized exclusions at all 12 insert locations: four UM driver mounts, two LM-to-UM receivers and three mounts on each tweeter. Each tweeter exclusion also reaches its narrower blind tip. These use the regular Obi-Wan LM/UM convention of 0.25 mm clearance plus 0.02 mm margin. Cable passages and magnet cavities have their own blockers. The waist geometry includes the [front/rear dent correction](../views/waist_comparison.png).

The supported rear recess has a removable PLA interface immediately below its insert mouths. The audit distinguishes this surface contact from support inside a printed bore: the recess interface ends at Z24.84 and the first insert-wall layer is Z25.00. It records the 0.060 mm difference from the nominal CAD surface, within half of the 0.16 mm layer, separately from the clear bore interiors.

## Process retained from regular Obi-Wan UM

The process authority is the existing [regular UM PETG-GF/PLA project](../../../to_print/obiwan/3mf_06hf_petg-gf_pla/obiwan_03_UM_carrier_1_of_1_GUI.3mf), using its frozen resolved filament profiles.

| Setting | Prepared value |
|---|---|
| Layers | 0.16 mm; first layer 0.20 mm |
| Walls and shells | Six walls; ten top and five bottom layers; top ironing |
| Model material | Tinmorry PETG-GF, 260 °C, flow 0.93, volumetric cap 12 mm³/s |
| Plate | Textured PEI, 80 °C |
| Supports | PETG-GF body; PLA interfaces at 220 °C |
| Interface | Three layers; zero top Z gap and zero interface spacing |
| Support form | Normal/snug; 0.7 mm object XY separation |
| Purging | 280 mm³ each direction; flushing into model, infill and supports disabled |
| Adhesion | 5 mm outer brim; body front down, diagonally placed |
| Upper-wing infill | 30% gyroid; supports enabled for the curved overhangs |
| Caps and retainers | 15% gyroid; supports disabled |

The body modifier starts at **installed Y = 421 mm**, above the UM flange. It changes only infill density and pattern, so the UM mounting region and neck below that boundary retain the original solid infill. Walls, thin sections and top/bottom shells remain dense even in the tweeter region.

![Actual sliced infill: teal UM, amber tweeters](infill_preview.png)

## Buried magnets and automatic pauses

| Job | Pause before layer Z | Insert at that pause |
|---|---:|---|
| Fused body | **9.80 mm** | Four Ø6×3 mm magnets |
| Each upper wing | **5.96 mm** | One retained LM Ø5×2 mm magnet |
| Each upper wing | **9.80 mm** | Two Ø6×3 mm UM magnets |

For the body plus one wing pair, prepare **eight Ø6×3 mm magnets and two Ø5×2 mm magnets**. Match the polarity of each body/wing pair and the existing LM before burying them. Seat the tilted magnets fully in their inclined pockets; they must stay below the completed surface before you resume. The embedded pause lowers the bed for access and restores the printing height on resume.

![Body magnet positions on the actual print bed](magnet_pause_map.png)

The print meshes include a small internal loading relief so the inclined Ø6×3 magnets can be seated before the slicer closes the roofs. The body loses only **13.23 mm³ internally**; its approved exterior, driver seats and LM interfaces are preserved. These loading paths remain covered in the finished part. Measured cover samples are at least 0.786 mm on the body and 0.798 mm on the wings. The original approved design STLs remain in `../STL/`; the native jobs use the following print variants:

| Print mesh | File |
|---|---|
| Shared body | [01_UM_Crescent_V4_PRINT.stl](geometry/01_UM_Crescent_V4_PRINT.stl) |
| Flat left / right | [Left](geometry/V4_flat_left_UPPER_PRINT.stl) · [Right](geometry/V4_flat_right_UPPER_PRINT.stl) |
| Graded left / right | [Left](geometry/V4_graded_left_UPPER_PRINT.stl) · [Right](geometry/V4_graded_right_UPPER_PRINT.stl) |

Use these variants if importing STLs into another project. STL alone does not retain the infill modifier, material settings, support blockers or pauses. The two [caps](../STL/02_Closed_Cap_PRINT_TWO.stl) and two [retainers](../STL/03_Tweeter_Retainer_PRINT_TWO.stl) use the unchanged retained geometry. See the [assembly and hardware notes](../README.md#magnets-and-assembly).

## Verification and rebuild

[manifest.json](manifest.json) binds all six native jobs to their exact source geometry, process authority and passing audit reports. The audit compares every mesh triangle in the exported project, verifies the nonprinting infill modifier, counts actual material extrusion, checks the native pauses and tests both cable routes and all insert bores against support toolpaths. The closest sampled support bead remains **0.496 mm outside the cable lumen**. All 12 insert interiors, including six narrower tip reliefs, pass the support-path check. Surface-boundary contacts are reported separately as described above. Both previews are bound to the final body G-code by [views_manifest.json](views_manifest.json).

Bambu Studio **02.07.01.62** produced all six jobs without native slice warnings. The supplemental static G-code check passes while retaining the standard Bambu machine commands; it does not simulate the printer firmware. Surface finish, support removal, magnet fit/retention and acoustic performance remain physical checks.

From `top_baffle_v2`, verify the delivered files without changing anything:

```bash
../.venv/bin/python candidates/nd25fn4_crescent/verify_print.py
```

To reproduce preparation and slicing in this workspace, using the frozen profiles in `review/nd25fn4_print/profiles` and the regular UM process authority:

```bash
../.venv/bin/python candidates/nd25fn4_crescent/prepare_magnet_loading.py --body
../.venv/bin/python candidates/nd25fn4_crescent/prepare_magnet_loading.py
../.venv/bin/python candidates/nd25fn4_crescent/prepare_print.py --body --accessories --wings
../.venv/bin/python candidates/nd25fn4_crescent/slice_print.py
../.venv/bin/python candidates/nd25fn4_crescent/finalize_print.py
../.venv/bin/python candidates/nd25fn4_crescent/print_views.py
../.venv/bin/python candidates/nd25fn4_crescent/verify_lm_assembly.py
../.venv/bin/python candidates/nd25fn4_crescent/render_lm_interface.py
../.venv/bin/python candidates/nd25fn4_crescent/verify_print.py
```

No deliverable ZIP is produced. These print jobs remain separate from the regular 42-choice release shelf.
