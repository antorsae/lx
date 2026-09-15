# ND25FN-4 waveguide magnet surface test

[Prepared test 3MF](00_Magnet_Surface_Test_06HF_PETG_TRANSLUCENT_PLA_TRANSLUCENT.gcode.3mf) — approximately **72 minutes**, excluding the insertion pause.

This is a 26 × 26 mm crop of one actual body magnet station, with its original curved outside surface, cover thickness, inclined loading pocket and print orientation. It uses the revised body's **Classic / outer-first / 0.52 mm outer-wall** settings and 100% body infill. No design geometry is modified.

Use the P2S **0.6 mm High Flow nozzle**, **PETG Translucent in AMS slot 4** and **PLA Translucent in slot 2**. Both material bed settings are 70 °C. The native job includes PETG supports with PLA interfaces.

Have **one Ø6×3 mm N45 magnet** ready. At the embedded pause before **Z9.80 mm**, orient and fully seat it below the printing plane, then resume. After cooling, inspect the curved outside face in raking light and by touch: look for the pocket outline, a texture/gloss change or a ridge. Also check that the cover and resumed layers are bonded and the magnet is retained. The cut sides of this test are crop boundaries, not production surfaces.

The revised toolpaths remove the measured exterior bead-width change. This test lets you check the physical result; a crop does not reproduce every thermal effect of the full body. Record filament lot, printer/nozzle and observations in `physical_result.json` alongside this file. Physical surface finish and retention are not yet verified.

[Qualification measurements](qualification.json) bind this file to its geometry, material-role, D6 wall, pause and static G-code checks.
