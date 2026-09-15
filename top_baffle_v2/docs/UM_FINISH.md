# UM seat and cable-handoff revision

The shared ND25FN-4 V4 body and both standalone Obi-Wan UM carriers now have a closed T-cable collar at the LM handoff, solid backing beneath the MU10 driver seat, and rounded access to the LM/UM M2 tie. The crescent's backing continues into its curved lower exterior; the separate UM-driver lead retains its rear service clearance.

The T collar reaches the existing LM cover with a nominal 0.05 mm fit seam. The former circular underside recess is filled, while the Ø82 driver bore, Ø98.6 flange recess, Z14.3 seat, driver pilots, M3 half-laps and captive magnets remain functional. The M2 bearing plane stays at Y319.304764 mm on X−17/Z10; the head corridor opens into the driver bore through an R0.5 mm mouth. Install this screw before installing the UM driver.

The fused crescent's subsequent rear-band revision removes the rectangular thickness steps left by the original carrier. A continuous curved surface joins the lower UM surround to the flat LM receiver lands, including the side fairings. The fixed duct, solid seat and screw access are retained. [Rear detail](../candidates/nd25fn4_crescent/views/UM_lower_band_oblique.png) and [exported-surface checks](../candidates/nd25fn4_crescent/lower_band_validation.json) cover this additional change; the standalone carriers and their 3MFs retain the preceding finish revision.

The latest fused-body revision also sculpts the UM's depth and regenerates its upper wings. The front outline is wider than the rear, with a consistent inclined side from the lower neck through the upper shoulder and rounded transitions into the shallow front lip. Exterior magnetic planes are removed completely; new Ø6×3 mm pockets follow the inclined skin and remain fully buried in the body and matching wings. The original lower LM mating boundary, driver seat and functional cable openings remain fixed. [Depth comparison](../candidates/nd25fn4_crescent/views/UM_depth_comparison.png), [outline measurements](../candidates/nd25fn4_crescent/outline_validation.json) and [full pocket-cover checks](../candidates/nd25fn4_crescent/depth_magnet_validation.json) document this candidate-only change. Its curved front relief requires fresh slicing and bed/support review.

## Current files

| Use | STL | Geometry checks |
|---|---|---|
| Fused ND25FN crescent + UM, either LM | [One shared body](../candidates/nd25fn4_crescent/STL/01_UM_Crescent_V4.stl) | [Complete candidate report](../candidates/nd25fn4_crescent/validation.json) |
| Standalone UM, shared shelf part | [UM carrier](../to_print/obiwan/stl/obiwan_03_UM_carrier_1_of_1.stl) | [No-floor assembly](../build/no_floor_stand/um_finish/validation.json) |
| Standalone UM in the floor-state source set | [Floor-state UM](../build/floor_stand/stl/obiwan_core_2_of_2_um_carrier.stl) | [Floor assembly](../build/floor_stand/um_finish/validation.json) |

The standalone UM's native STEP files and build records are under `build/{state}/um_finish/`. Both `obiwan_split.step` and `obiwan_assembled.step` contain the new UM; their other children retain their geometry. The full release's original source-stage record remains historical, with the focused UM revision identified in each `obiwan_release_manifest.json`.

The standalone UM GUI 3MF and both LM/UM combo GUI 3MFs were re-exported with current normal meshes and regenerated duct support blockers. Their existing placements, magnet pauses and 100% infill settings were retained. They contain no sliced G-code. [Delivery inventory](../to_print/delivery_manifest.json) binds and audits the current files.

## Validation and views

Both native STLs are watertight, with one connected material component and two intentional closed magnet cavities. The crescent has one connected material component and four intended magnet cavities. Checks cover driver clearance, original M2 bearing position, a Ø3.9 mm head-insertion gauge, a Ø5.6 mm T-cable gauge and 6,912 handoff-shell witnesses per assembly. Uncovered shell witnesses are confined to the normal fit seam, approximately 0.05 mm from both parts. Native UM/LM overlap is zero to numerical precision; the crescent's measured overlap is approximately 0.000001 mm³. The full crescent checks also retain the separate Ø7.8 mm UM cable clearance, rear-loft curvature and matching wing interfaces.

[Closed T handoff](../candidates/nd25fn4_crescent/views/duct_handoff_closed.png) · [Crescent underside](../candidates/nd25fn4_crescent/views/UM_solid_underside.png) · [Rounded M2 entrance](../candidates/nd25fn4_crescent/views/M2_tie_rounded_mouth.png) · [Native UM STEP snapshot](../build/no_floor_stand/um_finish/UM_rear.png)

The shared candidate remains an unsliced STL deliverable. Physical fit, printed wall quality and acoustic behavior have not been measured; filling the geometric cavity is not a resonance test. No ZIP deliverable was generated.

## Sources

`src/lx521_baffle/obiwan/carriers.py` owns the solid standalone seat and rounded screw mouth. `src/lx521_baffle/obiwan/bumps.py` owns the extended collar and matching negative passage. `candidates/nd25fn4_crescent/v4_model.py` retains those mechanisms within the curved body and backs the lower annulus while preserving both cable paths.

`scripts/rebuild_obiwan_um.py` provides a guarded, focused native STEP/STL export. The normal project build also uses the same updated carrier source. `scripts/check_um_finish.py` measures exported meshes independently of the fill and collar constructors. The candidate's existing rebuild, wing, validation, orthographic and review scripts regenerate its direct outputs. The bounded refresh scripts and previous files are recorded in `review/nd25fn4_iterations/um_finish_development/` and the archive named in the candidate's `superseded.json`.
