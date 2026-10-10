# Slim — H2C print files

11 of 11 jobs digitally audited. [All products, materials and policies](../README.md).

Slim baffle. Choose the LM for your stand state and one upper:
- **No floor stand:** `lm/slim_lm_no_floor_stand`.
- **Floor stand:** `lm/slim_lm_lower_floor_stand` + `lm/slim_lm_upper_floor_stand`.
- **Upper, one of:** `slim_upper` (ND25FW-4) or `slim_upper_bmr` (TEBM35C10-4 BMR, candidate). Slim has no ND25FN-4 upper.
- **Wings (optional, ND25FW-4 upper only):** the four A shoulders or the two B1 wings.

Each part keeps its files side by side: `<part>.stl`, `<part>.petg-gf.3mf` (editable project) and `<part>.petg-gf.gcode.3mf` (audited slice). Print from the `.3mf` or `.gcode.3mf`: an STL alone loses the supports, blockers, infill modifiers and magnet pauses.

## Combined files: one file per stand state

Each file holds every part for that stand state, arranged on named plates. Plates are grouped by option: print the shared plates plus the plates of the one tweeter option (A to F) and wing style you chose. Each part appears once; every plate is sliced and audited like the single-part jobs below (per-part settings, blockers, seam placement and magnet pauses are kept).

### Slim, floor stand

[Project (.3mf)](slim_floor_stand.petg-gf.3mf) · [Sliced, all plates (.gcode.3mf)](slim_floor_stand.petg-gf.gcode.3mf) · [Audit](../../build/h2c/plates/slim_floor_stand/audit.json)

| Plate | Contents | Parts | Magnet pauses (mm) |
|---:|---|---|---|
| 1 | LM lower with floor stand (shared by every option) | `slim_lm_lower_floor_stand` | none |
| 2 | LM upper (shared by every option) | `slim_lm_upper_floor_stand` | none |
| 3 | A · ND25FW-4 · upper module | `slim_upper` | 6.12 |
| 4 | B · BMR · upper module | `slim_upper_bmr` | 6.12 |
| 5 | Wings for A · A shoulders (4 pieces) | `slim_a_shoulder_bottom_left_1_of_4`, `slim_a_shoulder_top_left_2_of_4`, `slim_a_shoulder_bottom_right_3_of_4`, `slim_a_shoulder_top_right_4_of_4` | 5.96, 6.12 |
| 6 | Wings for A · B1 wings left + right | `slim_b1_wing_left_1_of_2`, `slim_b1_wing_right_2_of_2` | 5.96, 6.12 |

### Slim, no floor stand

[Project (.3mf)](slim_no_floor_stand.petg-gf.3mf) · [Sliced, all plates (.gcode.3mf)](slim_no_floor_stand.petg-gf.gcode.3mf) · [Audit](../../build/h2c/plates/slim_no_floor_stand/audit.json)

| Plate | Contents | Parts | Magnet pauses (mm) |
|---:|---|---|---|
| 1 | LM (shared by every option) | `slim_lm_no_floor_stand` | none |
| 2 | A · ND25FW-4 · upper module | `slim_upper` | 6.12 |
| 3 | B · BMR · upper module | `slim_upper_bmr` | 6.12 |
| 4 | Wings for A · A shoulders (4 pieces) | `slim_a_shoulder_bottom_left_1_of_4`, `slim_a_shoulder_top_left_2_of_4`, `slim_a_shoulder_bottom_right_3_of_4`, `slim_a_shoulder_top_right_4_of_4` | 5.96, 6.12 |
| 5 | Wings for A · B1 wings left + right | `slim_b1_wing_left_1_of_2`, `slim_b1_wing_right_2_of_2` | 5.96, 6.12 |

## Individual parts

## LM (lower module)

| Part | Tweeters | Infill | Magnets | Files | Status |
|---|---|---|---:|---|---|
| `slim_lm_no_floor_stand` | ND25FW-4, TEBM35C10-4 | 100% | 0 | [STL](lm/slim_lm_no_floor_stand.stl) · [PETG-GF project](lm/slim_lm_no_floor_stand.petg-gf.3mf) · [PETG-GF sliced](lm/slim_lm_no_floor_stand.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_slim_lm_no_floor_stand__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `slim_lm_lower_floor_stand` | ND25FW-4, TEBM35C10-4 | 100% | 0 | [STL](lm/slim_lm_lower_floor_stand.stl) · [PETG-GF project](lm/slim_lm_lower_floor_stand.petg-gf.3mf) · [PETG-GF sliced](lm/slim_lm_lower_floor_stand.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_slim_lm_lower_floor_stand__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `slim_lm_upper_floor_stand` | ND25FW-4, TEBM35C10-4 | 100% | 0 | [STL](lm/slim_lm_upper_floor_stand.stl) · [PETG-GF project](lm/slim_lm_upper_floor_stand.petg-gf.3mf) · [PETG-GF sliced](lm/slim_lm_upper_floor_stand.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_slim_lm_upper_floor_stand__petg_gf_pla/audit.json) | Digital pass; physical trial pending |

## Upper

| Part | Tweeters | Infill | Magnets | Files | Status |
|---|---|---|---:|---|---|
| `slim_upper` | ND25FW-4 | 100% | 4 | [STL](upper/slim_upper.stl) · [PETG-GF project](upper/slim_upper.petg-gf.3mf) · [PETG-GF sliced](upper/slim_upper.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_slim_upper__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `slim_upper_bmr` (candidate) | TEBM35C10-4 | 100% | 4 | [STL](upper/slim_upper_bmr.stl) · [PETG-GF project](upper/slim_upper_bmr.petg-gf.3mf) · [PETG-GF sliced](upper/slim_upper_bmr.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_slim_upper_bmr__petg_gf_pla/audit.json) | Digital pass; physical trial pending |

## Wings

| Part | Tweeters | Infill | Magnets | Files | Status |
|---|---|---|---:|---|---|
| `slim_a_shoulder_bottom_left_1_of_4` | ND25FW-4 | 10% | 1 | [STL](wings/slim_a_shoulder_bottom_left_1_of_4.stl) · [PETG-GF project](wings/slim_a_shoulder_bottom_left_1_of_4.petg-gf.3mf) · [PETG-GF sliced](wings/slim_a_shoulder_bottom_left_1_of_4.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_slim_05_A_shoulder_bottom_left_1_of_4__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `slim_a_shoulder_top_left_2_of_4` | ND25FW-4 | 10% | 1 | [STL](wings/slim_a_shoulder_top_left_2_of_4.stl) · [PETG-GF project](wings/slim_a_shoulder_top_left_2_of_4.petg-gf.3mf) · [PETG-GF sliced](wings/slim_a_shoulder_top_left_2_of_4.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_slim_06_A_shoulder_top_left_2_of_4__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `slim_a_shoulder_bottom_right_3_of_4` | ND25FW-4 | 10% | 1 | [STL](wings/slim_a_shoulder_bottom_right_3_of_4.stl) · [PETG-GF project](wings/slim_a_shoulder_bottom_right_3_of_4.petg-gf.3mf) · [PETG-GF sliced](wings/slim_a_shoulder_bottom_right_3_of_4.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_slim_07_A_shoulder_bottom_right_3_of_4__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `slim_a_shoulder_top_right_4_of_4` | ND25FW-4 | 10% | 1 | [STL](wings/slim_a_shoulder_top_right_4_of_4.stl) · [PETG-GF project](wings/slim_a_shoulder_top_right_4_of_4.petg-gf.3mf) · [PETG-GF sliced](wings/slim_a_shoulder_top_right_4_of_4.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_slim_08_A_shoulder_top_right_4_of_4__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `slim_b1_wing_left_1_of_2` | ND25FW-4 | 10% | 2 | [STL](wings/slim_b1_wing_left_1_of_2.stl) · [PETG-GF project](wings/slim_b1_wing_left_1_of_2.petg-gf.3mf) · [PETG-GF sliced](wings/slim_b1_wing_left_1_of_2.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_slim_09_B1_wing_left_1_of_2__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `slim_b1_wing_right_2_of_2` | ND25FW-4 | 10% | 2 | [STL](wings/slim_b1_wing_right_2_of_2.stl) · [PETG-GF project](wings/slim_b1_wing_right_2_of_2.petg-gf.3mf) · [PETG-GF sliced](wings/slim_b1_wing_right_2_of_2.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_slim_10_B1_wing_right_2_of_2__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
