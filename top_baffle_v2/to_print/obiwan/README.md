# Obiwan — H2C print files

37 of 37 jobs digitally audited. [All products, materials and policies](../README.md).

Obiwan baffle. Choose an LM for your stand state, then one upper arrangement:
- **LM, one of:** the one-piece `lm/obiwan_lm_carrier_{floor_stand,no_floor_stand}`, or the keyed halves `lm/obiwan_lm_keyed_bottom_{floor_stand,no_floor_stand}` + `lm/obiwan_lm_keyed_top` (the keyed top is the same part for both stand states).
- **ND25FW-4 or BMR upper:** `upper/obiwan_um_carrier` + one crescent (`obiwan_crescent_nd25fw4`, or the BMR candidates `obiwan_crescent_bmr` / `obiwan_crescent_bmr_opposed`); optional regular wings `wings/obiwan_wing_{flat,graded}_{left,right}`. Any LM above fits.
- **ND25FN-4 waveguide (candidate):** the keyed bottom for your stand state + `upper/obiwan_nd25fn4_tweeters_{right,left}_top` (it replaces the keyed top, the UM and the crescent) + its two caps; optional matching wings `wings/obiwan_nd25fn4_tweeters_<hand>_wing_*`. Regular wings do not fit it.
- **Dipole tweeter (candidate; GRS PT2522-4 planar or Mundorf AMT17D2.1), one driver, upright or rotated:** *T above* — any LM above + `dipole/obiwan_<pt2522|amt17d2>_t_above_<upright|rotated>` (UM + tweeter frame in one piece, replaces the UM carrier and crescent). *T below* — the keyed bottom for your stand state + `dipole/obiwan_<driver>_t_below_<orientation>` (keyed LM top + tweeter + UM in one piece); the PT2522 upright comes as two prints, a custom keyed LM top `dipole/obiwan_pt2522_t_below_upright_lm_top` + its T + UM piece `_um_t`, joined like the regular LM top and UM (they only fit each other). Drivers fix with M3 screws into M3 heat-set inserts. No wings for the dipole parts.
- **Floor stand add-on:** `addons/obiwan_nl8_service_lid` closes the underside NL8 bay.

Each part keeps its files side by side: `<part>.stl`, `<part>.petg-gf.3mf` (editable project) and `<part>.petg-gf.gcode.3mf` (audited slice). Print from the `.3mf` or `.gcode.3mf`: an STL alone loses the supports, blockers, infill modifiers and magnet pauses.

**ND25FN-4 hands.** The waveguides are not left/right symmetric. `tweeters_right` has the tweeter column on the right of the MU10 seen from the front; `tweeters_left` is its mirror. For a mirror-imaged pair with the tweeters inboard, the left speaker takes `tweeters_right` and the right speaker `tweeters_left`. Caps and wings are hand-specific. [ND25FN-4 guide](../../docs/DAYTON_ND25FN4_WAVEGUIDE.md).

## Combined files: one file per stand state

Each file holds every part for that stand state, arranged on named plates. Plates are grouped by option: print the shared plates plus the plates of the one tweeter option (A to F) and wing style you chose. Each part appears once; every plate is sliced and audited like the single-part jobs below (per-part settings, blockers, seam placement and magnet pauses are kept).

### Obiwan, floor stand

[Project (.3mf)](obiwan_floor_stand.petg-gf.3mf) · [Sliced, all plates (.gcode.3mf)](obiwan_floor_stand.petg-gf.gcode.3mf) · [Audit](../../build/h2c/plates/obiwan_floor_stand/audit.json)

| Plate | Contents | Parts | Magnet pauses (mm) |
|---:|---|---|---|
| 1 | NL8 service lid (every option) | `obiwan_nl8_service_lid` | none |
| 2 | A-C, E · LM carrier (shared by A, B, C and the dipole T above) | `obiwan_lm_carrier_floor_stand` | 5.96, 6.12 |
| 3 | A-C · UM carrier | `obiwan_um_carrier` | 6.12 |
| 4 | A · ND25FW-4 · crescent | `obiwan_crescent_nd25fw4` | none |
| 5 | B · BMR coaxial · crescent | `obiwan_crescent_bmr` | 6.12 |
| 6 | C · BMR opposed · crescent | `obiwan_crescent_bmr_opposed` | 6.12 |
| 7 | Wings A-C · flat · left | `obiwan_wing_flat_left` | 6.12 |
| 8 | Wings A-C · flat · right | `obiwan_wing_flat_right` | 6.12 |
| 9 | Wings A-C · graded · left | `obiwan_wing_graded_left` | 6.12 |
| 10 | Wings A-C · graded · right | `obiwan_wing_graded_right` | 6.12 |
| 11 | D, F · keyed LM bottom (also A-C, E with the keyed LM top instead of the LM carrier) | `obiwan_lm_keyed_bottom_floor_stand` | 5.96 |
| 12 | A-C, E · keyed LM top (with the keyed LM bottom, instead of the LM carrier) | `obiwan_lm_keyed_top` | 6.12 |
| 13 | D · ND25FN-4 tweeters right · vase top + caps | `obiwan_nd25fn4_tweeters_right_top`, `obiwan_nd25fn4_tweeters_right_cap_front`, `obiwan_nd25fn4_tweeters_right_cap_rear` | 6.12 |
| 14 | D · ND25FN-4 tweeters left · vase top + caps | `obiwan_nd25fn4_tweeters_left_top`, `obiwan_nd25fn4_tweeters_left_cap_front`, `obiwan_nd25fn4_tweeters_left_cap_rear` | 6.12 |
| 15 | D · ND25FN-4 tweeters right · flat wings left + right | `obiwan_nd25fn4_tweeters_right_wing_flat_left`, `obiwan_nd25fn4_tweeters_right_wing_flat_right` | 6.12 |
| 16 | D · ND25FN-4 tweeters right · graded wings left + right | `obiwan_nd25fn4_tweeters_right_wing_graded_left`, `obiwan_nd25fn4_tweeters_right_wing_graded_right` | 6.12 |
| 17 | D · ND25FN-4 tweeters left · flat wings left + right | `obiwan_nd25fn4_tweeters_left_wing_flat_left`, `obiwan_nd25fn4_tweeters_left_wing_flat_right` | 6.12 |
| 18 | D · ND25FN-4 tweeters left · graded wings left + right | `obiwan_nd25fn4_tweeters_left_wing_graded_left`, `obiwan_nd25fn4_tweeters_left_wing_graded_right` | 6.12 |
| 19 | E · PT2522-4 · T above · upright | `obiwan_pt2522_t_above_upright` | none |
| 20 | E · PT2522-4 · T above · rotated | `obiwan_pt2522_t_above_rotated` | none |
| 21 | E · AMT17D2.1 · T above · upright | `obiwan_amt17d2_t_above_upright` | none |
| 22 | E · AMT17D2.1 · T above · rotated | `obiwan_amt17d2_t_above_rotated` | none |
| 23 | F · PT2522-4 · T below · upright · keyed LM top | `obiwan_pt2522_t_below_upright_lm_top` | none |
| 24 | F · PT2522-4 · T below · upright · T + UM | `obiwan_pt2522_t_below_upright_um_t` | none |
| 25 | F · PT2522-4 · T below · rotated | `obiwan_pt2522_t_below_rotated` | none |
| 26 | F · AMT17D2.1 · T below · upright | `obiwan_amt17d2_t_below_upright` | none |
| 27 | F · AMT17D2.1 · T below · rotated | `obiwan_amt17d2_t_below_rotated` | none |

### Obiwan, no floor stand

[Project (.3mf)](obiwan_no_floor_stand.petg-gf.3mf) · [Sliced, all plates (.gcode.3mf)](obiwan_no_floor_stand.petg-gf.gcode.3mf) · [Audit](../../build/h2c/plates/obiwan_no_floor_stand/audit.json)

| Plate | Contents | Parts | Magnet pauses (mm) |
|---:|---|---|---|
| 1 | A-C, E · LM carrier (shared by A, B, C and the dipole T above) | `obiwan_lm_carrier_no_floor_stand` | 5.96, 6.12 |
| 2 | A-C · UM carrier | `obiwan_um_carrier` | 6.12 |
| 3 | A · ND25FW-4 · crescent | `obiwan_crescent_nd25fw4` | none |
| 4 | B · BMR coaxial · crescent | `obiwan_crescent_bmr` | 6.12 |
| 5 | C · BMR opposed · crescent | `obiwan_crescent_bmr_opposed` | 6.12 |
| 6 | Wings A-C · flat · left | `obiwan_wing_flat_left` | 6.12 |
| 7 | Wings A-C · flat · right | `obiwan_wing_flat_right` | 6.12 |
| 8 | Wings A-C · graded · left | `obiwan_wing_graded_left` | 6.12 |
| 9 | Wings A-C · graded · right | `obiwan_wing_graded_right` | 6.12 |
| 10 | D, F · keyed LM bottom (also A-C, E with the keyed LM top instead of the LM carrier) | `obiwan_lm_keyed_bottom_no_floor_stand` | 5.96 |
| 11 | A-C, E · keyed LM top (with the keyed LM bottom, instead of the LM carrier) | `obiwan_lm_keyed_top` | 6.12 |
| 12 | D · ND25FN-4 tweeters right · vase top + caps | `obiwan_nd25fn4_tweeters_right_top`, `obiwan_nd25fn4_tweeters_right_cap_front`, `obiwan_nd25fn4_tweeters_right_cap_rear` | 6.12 |
| 13 | D · ND25FN-4 tweeters left · vase top + caps | `obiwan_nd25fn4_tweeters_left_top`, `obiwan_nd25fn4_tweeters_left_cap_front`, `obiwan_nd25fn4_tweeters_left_cap_rear` | 6.12 |
| 14 | D · ND25FN-4 tweeters right · flat wings left + right | `obiwan_nd25fn4_tweeters_right_wing_flat_left`, `obiwan_nd25fn4_tweeters_right_wing_flat_right` | 6.12 |
| 15 | D · ND25FN-4 tweeters right · graded wings left + right | `obiwan_nd25fn4_tweeters_right_wing_graded_left`, `obiwan_nd25fn4_tweeters_right_wing_graded_right` | 6.12 |
| 16 | D · ND25FN-4 tweeters left · flat wings left + right | `obiwan_nd25fn4_tweeters_left_wing_flat_left`, `obiwan_nd25fn4_tweeters_left_wing_flat_right` | 6.12 |
| 17 | D · ND25FN-4 tweeters left · graded wings left + right | `obiwan_nd25fn4_tweeters_left_wing_graded_left`, `obiwan_nd25fn4_tweeters_left_wing_graded_right` | 6.12 |
| 18 | E · PT2522-4 · T above · upright | `obiwan_pt2522_t_above_upright` | none |
| 19 | E · PT2522-4 · T above · rotated | `obiwan_pt2522_t_above_rotated` | none |
| 20 | E · AMT17D2.1 · T above · upright | `obiwan_amt17d2_t_above_upright` | none |
| 21 | E · AMT17D2.1 · T above · rotated | `obiwan_amt17d2_t_above_rotated` | none |
| 22 | F · PT2522-4 · T below · upright · keyed LM top | `obiwan_pt2522_t_below_upright_lm_top` | none |
| 23 | F · PT2522-4 · T below · upright · T + UM | `obiwan_pt2522_t_below_upright_um_t` | none |
| 24 | F · PT2522-4 · T below · rotated | `obiwan_pt2522_t_below_rotated` | none |
| 25 | F · AMT17D2.1 · T below · upright | `obiwan_amt17d2_t_below_upright` | none |
| 26 | F · AMT17D2.1 · T below · rotated | `obiwan_amt17d2_t_below_rotated` | none |

## Individual parts

## LM (lower module)

| Part | Tweeters | Infill | Magnets | Files | Status |
|---|---|---|---:|---|---|
| `obiwan_lm_carrier_no_floor_stand` | ND25FW-4, TEBM35C10-4, GRS PT2522-4, Mundorf AMT17D2.1 | 100% | 4 | [STL](lm/obiwan_lm_carrier_no_floor_stand.stl) · [PETG-GF project](lm/obiwan_lm_carrier_no_floor_stand.petg-gf.3mf) · [PETG-GF sliced](lm/obiwan_lm_carrier_no_floor_stand.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_obiwan_core_lm_carrier_no_floor_stand__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_lm_keyed_bottom_no_floor_stand` | ND25FW-4, TEBM35C10-4, ND25FN-4, GRS PT2522-4, Mundorf AMT17D2.1 | 100% | 2 | [STL](lm/obiwan_lm_keyed_bottom_no_floor_stand.stl) · [PETG-GF project](lm/obiwan_lm_keyed_bottom_no_floor_stand.petg-gf.3mf) · [PETG-GF sliced](lm/obiwan_lm_keyed_bottom_no_floor_stand.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_obiwan_lm_keyed_bottom_no_floor_stand__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_lm_carrier_floor_stand` | ND25FW-4, TEBM35C10-4, GRS PT2522-4, Mundorf AMT17D2.1 | 100% | 4 | [STL](lm/obiwan_lm_carrier_floor_stand.stl) · [PETG-GF project](lm/obiwan_lm_carrier_floor_stand.petg-gf.3mf) · [PETG-GF sliced](lm/obiwan_lm_carrier_floor_stand.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_obiwan_core_lm_carrier_floor_stand__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_lm_keyed_bottom_floor_stand` | ND25FW-4, TEBM35C10-4, ND25FN-4, GRS PT2522-4, Mundorf AMT17D2.1 | 100% | 2 | [STL](lm/obiwan_lm_keyed_bottom_floor_stand.stl) · [PETG-GF project](lm/obiwan_lm_keyed_bottom_floor_stand.petg-gf.3mf) · [PETG-GF sliced](lm/obiwan_lm_keyed_bottom_floor_stand.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_obiwan_lm_keyed_bottom_floor_stand__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_lm_keyed_top` | ND25FW-4, TEBM35C10-4, GRS PT2522-4, Mundorf AMT17D2.1 | 100% | 2 | [STL](lm/obiwan_lm_keyed_top.stl) · [PETG-GF project](lm/obiwan_lm_keyed_top.petg-gf.3mf) · [PETG-GF sliced](lm/obiwan_lm_keyed_top.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_obiwan_lm_keyed_top__petg_gf_pla/audit.json) | Digital pass; physical trial pending |

## Upper

| Part | Tweeters | Infill | Magnets | Files | Status |
|---|---|---|---:|---|---|
| `obiwan_um_carrier` | ND25FW-4, TEBM35C10-4 | 100% | 2 | [STL](upper/obiwan_um_carrier.stl) · [PETG-GF project](upper/obiwan_um_carrier.petg-gf.3mf) · [PETG-GF sliced](upper/obiwan_um_carrier.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_obiwan_core_2_of_2_um_carrier__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_crescent_nd25fw4` | ND25FW-4 | 30% | 0 | [STL](upper/obiwan_crescent_nd25fw4.stl) · [PETG-GF project](upper/obiwan_crescent_nd25fw4.petg-gf.3mf) · [PETG-GF sliced](upper/obiwan_crescent_nd25fw4.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_obiwan_addon_tweeter_crescent__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_crescent_bmr` (candidate) | TEBM35C10-4 | 30% | 2 | [STL](upper/obiwan_crescent_bmr.stl) · [PETG-GF project](upper/obiwan_crescent_bmr.petg-gf.3mf) · [PETG-GF sliced](upper/obiwan_crescent_bmr.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_obiwan_bmr_crescent_TEBM35C10-4__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_crescent_bmr_opposed` (candidate) | TEBM35C10-4 | 30% | 4 | [STL](upper/obiwan_crescent_bmr_opposed.stl) · [PETG-GF project](upper/obiwan_crescent_bmr_opposed.petg-gf.3mf) · [PETG-GF sliced](upper/obiwan_crescent_bmr_opposed.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_obiwan_bmr_crescent_opposed_TEBM35C10-4__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_nd25fn4_tweeters_right_top` (candidate) | ND25FN-4 | 100% | 2 | [STL](upper/obiwan_nd25fn4_tweeters_right_top.stl) · [PETG-GF project](upper/obiwan_nd25fn4_tweeters_right_top.petg-gf.3mf) · [PETG-GF sliced](upper/obiwan_nd25fn4_tweeters_right_top.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_obiwan_tweeters_right_top__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_nd25fn4_tweeters_right_cap_front` (candidate) | ND25FN-4 | 15% | 0 | [STL](upper/obiwan_nd25fn4_tweeters_right_cap_front.stl) · [PETG-GF project](upper/obiwan_nd25fn4_tweeters_right_cap_front.petg-gf.3mf) · [PETG-GF sliced](upper/obiwan_nd25fn4_tweeters_right_cap_front.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_obiwan_tweeters_right_cap_front_tweeter__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_nd25fn4_tweeters_right_cap_rear` (candidate) | ND25FN-4 | 15% | 0 | [STL](upper/obiwan_nd25fn4_tweeters_right_cap_rear.stl) · [PETG-GF project](upper/obiwan_nd25fn4_tweeters_right_cap_rear.petg-gf.3mf) · [PETG-GF sliced](upper/obiwan_nd25fn4_tweeters_right_cap_rear.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_obiwan_tweeters_right_cap_rear_tweeter__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_nd25fn4_tweeters_left_top` (candidate) | ND25FN-4 | 100% | 2 | [STL](upper/obiwan_nd25fn4_tweeters_left_top.stl) · [PETG-GF project](upper/obiwan_nd25fn4_tweeters_left_top.petg-gf.3mf) · [PETG-GF sliced](upper/obiwan_nd25fn4_tweeters_left_top.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_obiwan_tweeters_left_top__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_nd25fn4_tweeters_left_cap_front` (candidate) | ND25FN-4 | 15% | 0 | [STL](upper/obiwan_nd25fn4_tweeters_left_cap_front.stl) · [PETG-GF project](upper/obiwan_nd25fn4_tweeters_left_cap_front.petg-gf.3mf) · [PETG-GF sliced](upper/obiwan_nd25fn4_tweeters_left_cap_front.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_obiwan_tweeters_left_cap_front_tweeter__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_nd25fn4_tweeters_left_cap_rear` (candidate) | ND25FN-4 | 15% | 0 | [STL](upper/obiwan_nd25fn4_tweeters_left_cap_rear.stl) · [PETG-GF project](upper/obiwan_nd25fn4_tweeters_left_cap_rear.petg-gf.3mf) · [PETG-GF sliced](upper/obiwan_nd25fn4_tweeters_left_cap_rear.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_obiwan_tweeters_left_cap_rear_tweeter__petg_gf_pla/audit.json) | Digital pass; physical trial pending |

## Dipole tweeters (GRS PT2522-4, Mundorf AMT17D2.1)

| Part | Tweeters | Infill | Magnets | Files | Status |
|---|---|---|---:|---|---|
| `obiwan_pt2522_t_above_upright` (candidate) | GRS PT2522-4 | 100% | 0 | [STL](dipole/obiwan_pt2522_t_above_upright.stl) · [PETG-GF project](dipole/obiwan_pt2522_t_above_upright.petg-gf.3mf) · [PETG-GF sliced](dipole/obiwan_pt2522_t_above_upright.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_grs_pt2522_4_obiwan_t_above_upright__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_pt2522_t_above_rotated` (candidate) | GRS PT2522-4 | 100% | 0 | [STL](dipole/obiwan_pt2522_t_above_rotated.stl) · [PETG-GF project](dipole/obiwan_pt2522_t_above_rotated.petg-gf.3mf) · [PETG-GF sliced](dipole/obiwan_pt2522_t_above_rotated.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_grs_pt2522_4_obiwan_t_above_rotated__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_amt17d2_t_above_upright` (candidate) | Mundorf AMT17D2.1 | 100% | 0 | [STL](dipole/obiwan_amt17d2_t_above_upright.stl) · [PETG-GF project](dipole/obiwan_amt17d2_t_above_upright.petg-gf.3mf) · [PETG-GF sliced](dipole/obiwan_amt17d2_t_above_upright.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_mundorf_amt17d2_1_obiwan_t_above_upright__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_amt17d2_t_above_rotated` (candidate) | Mundorf AMT17D2.1 | 100% | 0 | [STL](dipole/obiwan_amt17d2_t_above_rotated.stl) · [PETG-GF project](dipole/obiwan_amt17d2_t_above_rotated.petg-gf.3mf) · [PETG-GF sliced](dipole/obiwan_amt17d2_t_above_rotated.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_mundorf_amt17d2_1_obiwan_t_above_rotated__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_pt2522_t_below_upright_lm_top` (candidate) | GRS PT2522-4 | 100% | 0 | [STL](dipole/obiwan_pt2522_t_below_upright_lm_top.stl) · [PETG-GF project](dipole/obiwan_pt2522_t_below_upright_lm_top.petg-gf.3mf) · [PETG-GF sliced](dipole/obiwan_pt2522_t_below_upright_lm_top.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_grs_pt2522_4_obiwan_t_below_upright_lm_top__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_pt2522_t_below_upright_um_t` (candidate) | GRS PT2522-4 | 100% | 0 | [STL](dipole/obiwan_pt2522_t_below_upright_um_t.stl) · [PETG-GF project](dipole/obiwan_pt2522_t_below_upright_um_t.petg-gf.3mf) · [PETG-GF sliced](dipole/obiwan_pt2522_t_below_upright_um_t.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_grs_pt2522_4_obiwan_t_below_upright_um_t__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_pt2522_t_below_rotated` (candidate) | GRS PT2522-4 | 100% | 0 | [STL](dipole/obiwan_pt2522_t_below_rotated.stl) · [PETG-GF project](dipole/obiwan_pt2522_t_below_rotated.petg-gf.3mf) · [PETG-GF sliced](dipole/obiwan_pt2522_t_below_rotated.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_grs_pt2522_4_obiwan_t_below_rotated__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_amt17d2_t_below_upright` (candidate) | Mundorf AMT17D2.1 | 100% | 0 | [STL](dipole/obiwan_amt17d2_t_below_upright.stl) · [PETG-GF project](dipole/obiwan_amt17d2_t_below_upright.petg-gf.3mf) · [PETG-GF sliced](dipole/obiwan_amt17d2_t_below_upright.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_mundorf_amt17d2_1_obiwan_t_below_upright__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_amt17d2_t_below_rotated` (candidate) | Mundorf AMT17D2.1 | 100% | 0 | [STL](dipole/obiwan_amt17d2_t_below_rotated.stl) · [PETG-GF project](dipole/obiwan_amt17d2_t_below_rotated.petg-gf.3mf) · [PETG-GF sliced](dipole/obiwan_amt17d2_t_below_rotated.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_mundorf_amt17d2_1_obiwan_t_below_rotated__petg_gf_pla/audit.json) | Digital pass; physical trial pending |

## Wings

| Part | Tweeters | Infill | Magnets | Files | Status |
|---|---|---|---:|---|---|
| `obiwan_wing_flat_left` | ND25FW-4, TEBM35C10-4 | 10% | 3 | [STL](wings/obiwan_wing_flat_left.stl) · [PETG-GF project](wings/obiwan_wing_flat_left.petg-gf.3mf) · [PETG-GF sliced](wings/obiwan_wing_flat_left.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_obiwan_wing_flat_left__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_wing_flat_right` | ND25FW-4, TEBM35C10-4 | 10% | 3 | [STL](wings/obiwan_wing_flat_right.stl) · [PETG-GF project](wings/obiwan_wing_flat_right.petg-gf.3mf) · [PETG-GF sliced](wings/obiwan_wing_flat_right.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_obiwan_wing_flat_right__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_wing_graded_left` | ND25FW-4, TEBM35C10-4 | 10% | 3 | [STL](wings/obiwan_wing_graded_left.stl) · [PETG-GF project](wings/obiwan_wing_graded_left.petg-gf.3mf) · [PETG-GF sliced](wings/obiwan_wing_graded_left.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_obiwan_wing_graded_left__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_wing_graded_right` | ND25FW-4, TEBM35C10-4 | 10% | 3 | [STL](wings/obiwan_wing_graded_right.stl) · [PETG-GF project](wings/obiwan_wing_graded_right.petg-gf.3mf) · [PETG-GF sliced](wings/obiwan_wing_graded_right.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_obiwan_wing_graded_right__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_nd25fn4_tweeters_right_wing_flat_left` (candidate) | ND25FN-4 | 10% | 2 | [STL](wings/obiwan_nd25fn4_tweeters_right_wing_flat_left.stl) · [PETG-GF project](wings/obiwan_nd25fn4_tweeters_right_wing_flat_left.petg-gf.3mf) · [PETG-GF sliced](wings/obiwan_nd25fn4_tweeters_right_wing_flat_left.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_obiwan_tweeters_right_wing_flat_left__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_nd25fn4_tweeters_right_wing_flat_right` (candidate) | ND25FN-4 | 10% | 2 | [STL](wings/obiwan_nd25fn4_tweeters_right_wing_flat_right.stl) · [PETG-GF project](wings/obiwan_nd25fn4_tweeters_right_wing_flat_right.petg-gf.3mf) · [PETG-GF sliced](wings/obiwan_nd25fn4_tweeters_right_wing_flat_right.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_obiwan_tweeters_right_wing_flat_right__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_nd25fn4_tweeters_right_wing_graded_left` (candidate) | ND25FN-4 | 10% | 2 | [STL](wings/obiwan_nd25fn4_tweeters_right_wing_graded_left.stl) · [PETG-GF project](wings/obiwan_nd25fn4_tweeters_right_wing_graded_left.petg-gf.3mf) · [PETG-GF sliced](wings/obiwan_nd25fn4_tweeters_right_wing_graded_left.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_obiwan_tweeters_right_wing_graded_left__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_nd25fn4_tweeters_right_wing_graded_right` (candidate) | ND25FN-4 | 10% | 2 | [STL](wings/obiwan_nd25fn4_tweeters_right_wing_graded_right.stl) · [PETG-GF project](wings/obiwan_nd25fn4_tweeters_right_wing_graded_right.petg-gf.3mf) · [PETG-GF sliced](wings/obiwan_nd25fn4_tweeters_right_wing_graded_right.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_obiwan_tweeters_right_wing_graded_right__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_nd25fn4_tweeters_left_wing_flat_left` (candidate) | ND25FN-4 | 10% | 2 | [STL](wings/obiwan_nd25fn4_tweeters_left_wing_flat_left.stl) · [PETG-GF project](wings/obiwan_nd25fn4_tweeters_left_wing_flat_left.petg-gf.3mf) · [PETG-GF sliced](wings/obiwan_nd25fn4_tweeters_left_wing_flat_left.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_obiwan_tweeters_left_wing_flat_left__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_nd25fn4_tweeters_left_wing_flat_right` (candidate) | ND25FN-4 | 10% | 2 | [STL](wings/obiwan_nd25fn4_tweeters_left_wing_flat_right.stl) · [PETG-GF project](wings/obiwan_nd25fn4_tweeters_left_wing_flat_right.petg-gf.3mf) · [PETG-GF sliced](wings/obiwan_nd25fn4_tweeters_left_wing_flat_right.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_obiwan_tweeters_left_wing_flat_right__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_nd25fn4_tweeters_left_wing_graded_left` (candidate) | ND25FN-4 | 10% | 2 | [STL](wings/obiwan_nd25fn4_tweeters_left_wing_graded_left.stl) · [PETG-GF project](wings/obiwan_nd25fn4_tweeters_left_wing_graded_left.petg-gf.3mf) · [PETG-GF sliced](wings/obiwan_nd25fn4_tweeters_left_wing_graded_left.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_obiwan_tweeters_left_wing_graded_left__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
| `obiwan_nd25fn4_tweeters_left_wing_graded_right` (candidate) | ND25FN-4 | 10% | 2 | [STL](wings/obiwan_nd25fn4_tweeters_left_wing_graded_right.stl) · [PETG-GF project](wings/obiwan_nd25fn4_tweeters_left_wing_graded_right.petg-gf.3mf) · [PETG-GF sliced](wings/obiwan_nd25fn4_tweeters_left_wing_graded_right.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_dayton_nd25fn4_obiwan_tweeters_left_wing_graded_right__petg_gf_pla/audit.json) | Digital pass; physical trial pending |

## Add-ons

| Part | Tweeters | Infill | Magnets | Files | Status |
|---|---|---|---:|---|---|
| `obiwan_nl8_service_lid` | ND25FW-4, TEBM35C10-4, GRS PT2522-4, Mundorf AMT17D2.1 | 100% | 0 | [STL](addons/obiwan_nl8_service_lid.stl) · [PETG-GF project](addons/obiwan_nl8_service_lid.petg-gf.3mf) · [PETG-GF sliced](addons/obiwan_nl8_service_lid.petg-gf.gcode.3mf) · [PETG-GF audit](../../build/h2c/jobs/h2c_obiwan_addon_nl8_service_lid__petg_gf_pla/audit.json) | Digital pass; physical trial pending |
