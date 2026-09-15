# waveguide — PETG Translucent + PLA support, alternate changeover calibration

**Earlier P2S files.** For the current H2C printer, use the
[H2C catalog](../../../to_print/h2c/README.md) and
[Dayton ND25FN-4 waveguide guide](../../../docs/DAYTON_ND25FN4_WAVEGUIDE.md).
Original revision strings in these filenames are retained for provenance.


Sliced native Bambu Studio files for the **P2S, 0.6 mm High Flow nozzle**, using the **Engineering Plate with glue, 70 °C, a 5 mm outer brim and no raft**. The plate selection is stored in each project; re-open the updated files before printing.
Use **Bambu PETG Translucent in AMS slot 4** and **Bambu PLA Translucent in AMS slot 2**, as previously confirmed. Open as a project and map the two filaments in the Send dialog. Native nozzle-map numbers are not AMS slots.

| File | Estimated time |
|---|---:|
| [ND25FN-4 fused UM/waveguide body](01_UM_Crescent_V4_SMOOTH_WALLS_PURGE560_06HF_PETG_TRANSLUCENT_PLA_TRANSLUCENT.gcode.3mf) | 12 h 22 min |
| [Two caps + two M3 retainers](02_Caps_TWO_Retainers_TWO_PURGE560_06HF_PETG_TRANSLUCENT_PLA_TRANSLUCENT.gcode.3mf) | 2 h 10 min |

This alternate pair uses **560 mm³ purge in both directions**, multiplier **1**, and an explicit **12 mm³/s PLA flush**. The preceding translucent files used 298 mm³ PETG → PLA and 575 mm³ PLA → PETG; their generated PLA flush already ran at 12 mm³/s. The new calibration increases the PETG → PLA purge and standardizes the return purge. It has not been physically shown to resolve a blockage.

PETG printing: **250 °C first layer / 245 °C later**, flow 0.95, printing limit 16 mm³/s. PLA printing: **220 °C**, flow 0.98, printing limit 12 mm³/s. Both materials keep the Engineering Plate at **70 °C**, including during material changes. The native changeover temperature requests are preserved from the existing translucent recipe.

The [Bambu mutual-support guide](https://wiki.bambulab.com/en/filament-acc/filament/h2d-pla-and-petg-mutual-support) **explicitly excludes PETG Translucent and only covers PLA Basic paired with PETG Basic/HF**. It lists PEI/High Temp plates. This user-selected material/plate combination is a custom setup, not a guide-qualified preset. We retain its material-specific temperatures and cooling instead of importing the guide's 60 °C bed and 230 °C PLA Basic values. Open the P2S door and/or top cover while PLA is in use, and dry each filament according to its own supplier instructions. The [policy](../translucent_changeover_policy.json) records every retained exception.

The body retains Classic outer-first walls, 0.52 mm exterior paths, six walls, **100% UM infill** and **15% gyroid in the tweeter region**. Driver seats, the flush LM interface, cable routing, insert/magnet support blockers and source geometry are retained. Body magnet insertion pauses before **Z9.80 mm**; have four Ø6×3 mm N45 discs ready and fully seat them below the printing plane before resuming.

Support bodies use PETG; **PLA is used for the support interfaces**. Top and bottom Z gaps are zero. Both cap ceilings have **three dense PLA contact layers** at Z8.68, Z8.84 and Z9.00 mm before the PETG ceiling at Z9.16 mm. The measured inner ceiling coverage is at least **99.35%**. The two M3 retainers require no generated support. Cap/retainer infill remains 15% gyroid.

The cap ceiling has an approximately **44.6 mm unsupported span** without support. Bridging that distance might work in a test, but sagging could spoil the internal clearance and surface. These production files retain removable supports; no unsupported bridge has been physically qualified. A raft is not needed in this setup: the existing outer brim supplies extra adhesion without raising the whole part on a sacrificial base.

Both jobs pass exact source-mesh/placement comparison, full material-recipe checks, actual model/support/interface material checks, the changeover-volume/flow gate, native-warning checks and supplemental static G-code validation. Body D6 wall continuity and insertion timing pass; all 12 insert bores and both cable routes remain clear of support. No physical qualification or printer operation is claimed.

[Manifest and checks](manifest.json) · [Policy overlay](../translucent_changeover_policy.json) · [Existing translucent wing alternatives](../print_translucent/README.md)

Rebuild from the project root:

```bash
../.venv/bin/python candidates/nd25fn4_crescent/translucent_changeover_print.py prepare
# Inspect review/nd25fn4_translucent_changeover/dry_run.json.
../.venv/bin/python candidates/nd25fn4_crescent/translucent_changeover_print.py slice
../.venv/bin/python candidates/nd25fn4_crescent/translucent_changeover_print.py publish
```

Verify without reslicing:

```bash
../.venv/bin/python candidates/nd25fn4_crescent/translucent_changeover_print.py verify
```
