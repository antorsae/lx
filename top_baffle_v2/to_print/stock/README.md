# Stock — H2C print files

17 of 17 jobs digitally audited. [All products, materials and policies](../README.md).

Stock baffle. Choose the LM for your stand state and one upper:
- **No floor stand:** `lm/stock_lm_no_floor_stand`.
- **Floor stand:** `lm/stock_lm_lower_floor_stand` + `lm/stock_lm_upper_floor_stand`.
- **Upper, one of:** `stock_upper` (ND25FW-4), `stock_upper_bmr` (TEBM35C10-4 BMR, candidate), or the ND25FN-4 vase `stock_nd25fn4_tweeters_{right,left}_upper` + its two caps (candidate; pick the hand, see below).
- **Wings (optional, ND25FW-4 upper only):** the four A shoulders or the two B1 wings.

Each part keeps its files side by side: `<part>.stl`, `<part>.petg-gf.3mf` (editable project) and `<part>.petg-gf.gcode.3mf` (audited slice). Print from the `.3mf` or `.gcode.3mf`: an STL alone loses the supports, blockers, infill modifiers and magnet pauses.

**ND25FN-4 hands.** The waveguides are not left/right symmetric. `tweeters_right` has the tweeter column on the right of the MU10 seen from the front; `tweeters_left` is its mirror. For a mirror-imaged pair with the tweeters inboard, the left speaker takes `tweeters_right` and the right speaker `tweeters_left`. Caps and wings are hand-specific. [ND25FN-4 guide](../../docs/DAYTON_ND25FN4_WAVEGUIDE.md).

## Combined files: one file per stand state

Each file holds every part for that stand state, arranged on named plates. Plates are grouped by option: print the shared plates plus the plates of the one tweeter option (A to F) and wing style you chose. Each part appears once; every plate is sliced and audited like the single-part jobs below (per-part settings, blockers, seam placement and magnet pauses are kept).

### Stock, floor stand

[Project (.3mf)](stock_floor_stand.petg-gf.3mf) · [Sliced, all plates (.gcode.3mf)](stock_floor_stand.petg-gf.gcode.3mf) · [Audit](../../build/h2c/plates/stock_floor_stand/audit.json)

| Plate | Contents | Parts | Magnet pauses (mm) |
|---:|---|---|---|
| 1 | LM lower with floor stand (shared by every option) | `stock_lm_lower_floor_stand` | none |
| 2 | LM upper (shared by every option) | `stock_lm_upper_floor_stand` | none |
| 3 | A · ND25FW-4 · upper module | `stock_upper` | 6.12 |
| 4 | B · BMR · upper module | `stock_upper_bmr` | 6.12 |
| 5 | C · ND25FN-4 tweeters right · vase + caps | `stock_nd25fn4_tweeters_right_upper`, `stock_nd25fn4_tweeters_right_cap_front`, `stock_nd25fn4_tweeters_right_cap_rear` | none |
| 6 | C · ND25FN-4 tweeters left · vase + caps | `stock_nd25fn4_tweeters_left_upper`, `stock_nd25fn4_tweeters_left_cap_front`, `stock_nd25fn4_tweeters_left_cap_rear` | none |
| 7 | Wings for A · A shoulders (4 pieces) | `stock_a_shoulder_bottom_left_1_of_4`, `stock_a_shoulder_top_left_2_of_4`, `stock_a_shoulder_bottom_right_3_of_4`, `stock_a_shoulder_top_right_4_of_4` | 6.12 |
| 8 | Wings for A · B1 wings left + right | `stock_b1_wing_left_1_of_2`, `stock_b1_wing_right_2_of_2` | 6.12 |

### Stock, no floor stand

[Project (.3mf)](stock_no_floor_stand.petg-gf.3mf) · [Sliced, all plates (.gcode.3mf)](stock_no_floor_stand.petg-gf.gcode.3mf) · [Audit](../../build/h2c/plates/stock_no_floor_stand/audit.json)

| Plate | Contents | Parts | Magnet pauses (mm) |
|---:|---|---|---|
| 1 | LM (shared by every option) | `stock_lm_no_floor_stand` | none |
| 2 | A · ND25FW-4 · upper module | `stock_upper` | 6.12 |
| 3 | B · BMR · upper module | `stock_upper_bmr` | 6.12 |
| 4 | C · ND25FN-4 tweeters right · vase + caps | `stock_nd25fn4_tweeters_right_upper`, `stock_nd25fn4_tweeters_right_cap_front`, `stock_nd25fn4_tweeters_right_cap_rear` | none |
| 5 | C · ND25FN-4 tweeters left · vase + caps | `stock_nd25fn4_tweeters_left_upper`, `stock_nd25fn4_tweeters_left_cap_front`, `stock_nd25fn4_tweeters_left_cap_rear` | none |
| 6 | Wings for A · A shoulders (4 pieces) | `stock_a_shoulder_bottom_left_1_of_4`, `stock_a_shoulder_top_left_2_of_4`, `stock_a_shoulder_bottom_right_3_of_4`, `stock_a_shoulder_top_right_4_of_4` | 6.12 |
| 7 | Wings for A · B1 wings left + right | `stock_b1_wing_left_1_of_2`, `stock_b1_wing_right_2_of_2` | 6.12 |

## Individual parts

## LM (lower module)

| Part | Tweeters | Infill | Magnets | Files | Status |
|---|---|---|---:|---|---|
| `stock_lm_no_floor_stand` | ND25FW-4, TEBM35C10-4, ND25FN-4 | 100% | 0 | [STL](lm/stock_lm_no_floor_stand.stl) · [PETG-GF project](lm/stock_lm_no_floor_stand.petg-gf.3mf) · [PETG-GF sliced](lm/stock_lm_no_floor_stand.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_stock_lm_no_floor_stand__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `stock_lm_lower_floor_stand` | ND25FW-4, TEBM35C10-4, ND25FN-4 | 100% | 0 | [STL](lm/stock_lm_lower_floor_stand.stl) · [PETG-GF project](lm/stock_lm_lower_floor_stand.petg-gf.3mf) · [PETG-GF sliced](lm/stock_lm_lower_floor_stand.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_stock_lm_lower_floor_stand__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `stock_lm_upper_floor_stand` | ND25FW-4, TEBM35C10-4, ND25FN-4 | 100% | 0 | [STL](lm/stock_lm_upper_floor_stand.stl) · [PETG-GF project](lm/stock_lm_upper_floor_stand.petg-gf.3mf) · [PETG-GF sliced](lm/stock_lm_upper_floor_stand.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_stock_lm_upper_floor_stand__petg_gf_pla/audit.json) | Digital pass; physical trial pending |

## Upper

| Part | Tweeters | Infill | Magnets | Files | Status |
|---|---|---|---:|---|---|
| `stock_upper` | ND25FW-4 | 100% | 4 | [STL](upper/stock_upper.stl) · [PETG-GF project](upper/stock_upper.petg-gf.3mf) · [PETG-GF sliced](upper/stock_upper.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_stock_upper__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `stock_upper_bmr` (candidate) | TEBM35C10-4 | 100% | 4 | [STL](upper/stock_upper_bmr.stl) · [PETG-GF project](upper/stock_upper_bmr.petg-gf.3mf) · [PETG-GF sliced](upper/stock_upper_bmr.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_stock_upper_bmr__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `stock_nd25fn4_tweeters_right_upper` (candidate) | ND25FN-4 | 100% | 0 | [STL](upper/stock_nd25fn4_tweeters_right_upper.stl) · [PETG-GF project](upper/stock_nd25fn4_tweeters_right_upper.petg-gf.3mf) · [PETG-GF sliced](upper/stock_nd25fn4_tweeters_right_upper.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_stock_tweeters_right_upper__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `stock_nd25fn4_tweeters_right_cap_front` (candidate) | ND25FN-4 | 15% | 0 | [STL](upper/stock_nd25fn4_tweeters_right_cap_front.stl) · [PETG-GF project](upper/stock_nd25fn4_tweeters_right_cap_front.petg-gf.3mf) · [PETG-GF sliced](upper/stock_nd25fn4_tweeters_right_cap_front.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_stock_tweeters_right_cap_front_tweeter__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `stock_nd25fn4_tweeters_right_cap_rear` (candidate) | ND25FN-4 | 15% | 0 | [STL](upper/stock_nd25fn4_tweeters_right_cap_rear.stl) · [PETG-GF project](upper/stock_nd25fn4_tweeters_right_cap_rear.petg-gf.3mf) · [PETG-GF sliced](upper/stock_nd25fn4_tweeters_right_cap_rear.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_stock_tweeters_right_cap_rear_tweeter__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `stock_nd25fn4_tweeters_left_upper` (candidate) | ND25FN-4 | 100% | 0 | [STL](upper/stock_nd25fn4_tweeters_left_upper.stl) · [PETG-GF project](upper/stock_nd25fn4_tweeters_left_upper.petg-gf.3mf) · [PETG-GF sliced](upper/stock_nd25fn4_tweeters_left_upper.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_stock_tweeters_left_upper__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `stock_nd25fn4_tweeters_left_cap_front` (candidate) | ND25FN-4 | 15% | 0 | [STL](upper/stock_nd25fn4_tweeters_left_cap_front.stl) · [PETG-GF project](upper/stock_nd25fn4_tweeters_left_cap_front.petg-gf.3mf) · [PETG-GF sliced](upper/stock_nd25fn4_tweeters_left_cap_front.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_stock_tweeters_left_cap_front_tweeter__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `stock_nd25fn4_tweeters_left_cap_rear` (candidate) | ND25FN-4 | 15% | 0 | [STL](upper/stock_nd25fn4_tweeters_left_cap_rear.stl) · [PETG-GF project](upper/stock_nd25fn4_tweeters_left_cap_rear.petg-gf.3mf) · [PETG-GF sliced](upper/stock_nd25fn4_tweeters_left_cap_rear.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_stock_tweeters_left_cap_rear_tweeter__petg_gf_pla/audit.json) | Digital pass; physical trial pending |

## Wings

| Part | Tweeters | Infill | Magnets | Files | Status |
|---|---|---|---:|---|---|
| `stock_a_shoulder_bottom_left_1_of_4` | ND25FW-4 | 10% | 1 | [STL](wings/stock_a_shoulder_bottom_left_1_of_4.stl) · [PETG-GF project](wings/stock_a_shoulder_bottom_left_1_of_4.petg-gf.3mf) · [PETG-GF sliced](wings/stock_a_shoulder_bottom_left_1_of_4.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_stock_05_A_shoulder_bottom_left_1_of_4__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `stock_a_shoulder_top_left_2_of_4` | ND25FW-4 | 10% | 1 | [STL](wings/stock_a_shoulder_top_left_2_of_4.stl) · [PETG-GF project](wings/stock_a_shoulder_top_left_2_of_4.petg-gf.3mf) · [PETG-GF sliced](wings/stock_a_shoulder_top_left_2_of_4.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_stock_06_A_shoulder_top_left_2_of_4__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `stock_a_shoulder_bottom_right_3_of_4` | ND25FW-4 | 10% | 1 | [STL](wings/stock_a_shoulder_bottom_right_3_of_4.stl) · [PETG-GF project](wings/stock_a_shoulder_bottom_right_3_of_4.petg-gf.3mf) · [PETG-GF sliced](wings/stock_a_shoulder_bottom_right_3_of_4.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_stock_07_A_shoulder_bottom_right_3_of_4__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `stock_a_shoulder_top_right_4_of_4` | ND25FW-4 | 10% | 1 | [STL](wings/stock_a_shoulder_top_right_4_of_4.stl) · [PETG-GF project](wings/stock_a_shoulder_top_right_4_of_4.petg-gf.3mf) · [PETG-GF sliced](wings/stock_a_shoulder_top_right_4_of_4.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_stock_08_A_shoulder_top_right_4_of_4__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `stock_b1_wing_left_1_of_2` | ND25FW-4 | 10% | 2 | [STL](wings/stock_b1_wing_left_1_of_2.stl) · [PETG-GF project](wings/stock_b1_wing_left_1_of_2.petg-gf.3mf) · [PETG-GF sliced](wings/stock_b1_wing_left_1_of_2.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_stock_09_B1_wing_left_1_of_2__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `stock_b1_wing_right_2_of_2` | ND25FW-4 | 10% | 2 | [STL](wings/stock_b1_wing_right_2_of_2.stl) · [PETG-GF project](wings/stock_b1_wing_right_2_of_2.petg-gf.3mf) · [PETG-GF sliced](wings/stock_b1_wing_right_2_of_2.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_stock_10_B1_wing_right_2_of_2__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
