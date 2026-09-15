# PETG-GF / PLA changeover update

Applied the user-requested calibration for the P2S **0.6 mm High Flow** nozzle:

- Purge: **560 mm³ in each direction**, multiplier 1 (previously 280 mm³).
- PLA flush flow: **12 mm³/s** (previous native automatic request: 40 mm³/s).
- Printing temperatures, printing flow limits, geometry, supports, infill and magnet pause instructions retain their previous values.

Seven native sliced files were regenerated: the six V4 production plates and the D6 body/wing test. All **152** material changes passed a check of the actual native purge requests and virtual flush lengths. Geometry/material/support/wall/pause checks and supplemental static G-code validation pass.

Eight regular Obiwan GUI projects carry the new settings and pass mesh/settings validation. **Slice those GUI projects in Bambu Studio** to produce new G-code; previously exported files do not update automatically.

All seven separate translucent native files remain byte-identical. Their independent recipes were re-audited after updating the shared code and policy provenance.

## Files

| File | Estimated time |
|---|---:|
| [01_UM_Crescent_V4_06HF_PETG_GF_PLA.gcode.3mf](../../candidates/nd25fn4_crescent/print/01_UM_Crescent_V4_06HF_PETG_GF_PLA.gcode.3mf) | 12 h 51 min |
| [02_Caps_TWO_Retainers_TWO_06HF_PETG_GF_PLA.gcode.3mf](../../candidates/nd25fn4_crescent/print/02_Caps_TWO_Retainers_TWO_06HF_PETG_GF_PLA.gcode.3mf) | 2 h 18 min |
| [V4_flat_left_UPPER_06HF_PETG_GF_PLA.gcode.3mf](../../candidates/nd25fn4_crescent/print/V4_flat_left_UPPER_06HF_PETG_GF_PLA.gcode.3mf) | 2 h 29 min |
| [V4_flat_right_UPPER_06HF_PETG_GF_PLA.gcode.3mf](../../candidates/nd25fn4_crescent/print/V4_flat_right_UPPER_06HF_PETG_GF_PLA.gcode.3mf) | 2 h 34 min |
| [V4_graded_left_UPPER_06HF_PETG_GF_PLA.gcode.3mf](../../candidates/nd25fn4_crescent/print/V4_graded_left_UPPER_06HF_PETG_GF_PLA.gcode.3mf) | 2 h 21 min |
| [V4_graded_right_UPPER_06HF_PETG_GF_PLA.gcode.3mf](../../candidates/nd25fn4_crescent/print/V4_graded_right_UPPER_06HF_PETG_GF_PLA.gcode.3mf) | 2 h 22 min |
| [D6_body_wing_fit_06HF_PETG_GF_PLA.gcode.3mf](../../candidates/nd25fn4_crescent/print/qualification/D6_body_wing_fit_06HF_PETG_GF_PLA.gcode.3mf) | 1 h 23 min |

The **83-minute test pair** is the next physical check. Have two Ø6×3 mm N45 magnets ready for its embedded insertion pauses. Record whether both swaps extrude continuously, whether residual material remains, and whether any clicking/loading error occurs. The update has not been physically qualified and no printer job was sent or started.

[Before/after and changeover verification](verification.json) · [Shared policies](../../docs/PRINT_POLICIES.md) · [V4 print instructions](../../candidates/nd25fn4_crescent/print/README.md)

Reproduction starts with `scripts/update_petg_gf_changeover.py` (unsliced metadata only). The existing `finalize_print.py` and `build_magnet_coupon.py` use the pinned native Bambu profiles to reslice and audit. `scripts/verify_purge_update.py` checks before/after settings, native changeover volumes/flow, preserved geometry, pause and temperature requests, and the separate translucent bytes.

Focused regression checks: 18 tests passed for policy mapping, stale changeover rejection, material-number parsing and delivery contracts. The current regular shelf validates 42 choices and 76 projects.
