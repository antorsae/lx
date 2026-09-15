# P2S files and slicer estimates

Generated from `catalog.json` and `delivery_manifest.json`; run `make PRINTER=P2S delivery_refresh` to update.

For the current printer, use the [H2C file catalog](h2c/README.md). The three families below are alternative upper selections, with their own compatible parts.

![Three tweeter families and five carrier arrangements at one scale](../images/generated/iso/rows/tweeter_row.png)

| Tweeter family | Earlier P2S selection | Print guide |
|---|---|---|
| Dayton ND25FW-4 | Stock/Slim standard vase or regular Obiwan UM + crescent | Jobs below |
| Tectonic TEBM35C10-4 BMR | Obiwan coaxial/opposed crescent on regular UM; separate Stock/Slim BMR vase delivery | Obiwan jobs below; [other mounts](../docs/VARIANTS.md) |
| Dayton ND25FN-4 waveguide | Fused Obiwan UM/body + two caps + two M3 retainers; matching optional wings | [PETG-GF + PLA](../candidates/nd25fn4_crescent/print/README.md) · [PETG Translucent + PLA](../candidates/nd25fn4_crescent/print_translucent/README.md) |

The ND25FN-4 material bundles have their own manifests and estimates. Counts below describe the earlier regular ND25FW-4/BMR shelf.

42 choices; 68 sliced projects and 8 GUI projects across all lanes. These are alternatives, not a per-speaker part count.

Times and grams are slicer estimates per job, including its encoded setup/purge where reported. GUI estimates remain pending until sliced. Multiply the chosen jobs by two for stereo.

| Part / plate | Lane | State | Delivery | Time | Filament (g) |
|---|---|---|---|---|---|
| [Stock LM lower, 1 of 3 — no-floor stand](stock/3mf_04/stock_01_LM_bottom_1_of_3_no_floor_stand.gcode.3mf) | pla04 | no_floor_stand | Sliced | 10h 38m 9s | 218.59 |
| [Stock LM lower, 1 of 3 — no-floor stand](stock/3mf_06hf/stock_01_LM_bottom_1_of_3_no_floor_stand_06hf.gcode.3mf) | pla06hf | no_floor_stand | Sliced | 5h 37m 38s | 216.11 |
| [Stock LM lower, 1 of 3 — floor stand](stock/3mf_04/stock_01_LM_bottom_1_of_3_floor_stand.gcode.3mf) | pla04 | floor_stand | Sliced | 18h 15m 36s | 367.37 |
| [Stock LM lower, 1 of 3 — floor stand](stock/3mf_06hf/stock_01_LM_bottom_1_of_3_floor_stand_06hf.gcode.3mf) | pla06hf | floor_stand | Sliced | 10h 1m 58s | 353.22 |
| [Stock LM middle-left, 2 of 3](stock/3mf_04/stock_02_LM_mid_left_2_of_3.gcode.3mf) | pla04 | shared | Sliced | 5h 22m 25s | 115.84 |
| [Stock LM middle-left, 2 of 3](stock/3mf_06hf/stock_02_LM_mid_left_2_of_3_06hf.gcode.3mf) | pla06hf | shared | Sliced | 3h 2m 30s | 113.87 |
| [Stock LM middle-right, 3 of 3 — shared](stock/3mf_04/stock_03_LM_mid_right_3_of_3.gcode.3mf) | pla04 | shared | Sliced | 5h 30m 28s | 119.49 |
| [Stock LM middle-right, 3 of 3 — shared](stock/3mf_06hf/stock_03_LM_mid_right_3_of_3_06hf.gcode.3mf) | pla06hf | shared | Sliced | 3h 9m 33s | 116.92 |
| [Stock UM vase/top, 1 of 1](stock/3mf_04/stock_04_UM_vase_1_of_1.gcode.3mf) | pla04 | shared | Sliced | 3h 15m 11s | 68.21 |
| [Stock UM vase/top, 1 of 1](stock/3mf_06hf/stock_04_UM_vase_1_of_1_06hf.gcode.3mf) | pla06hf | shared | Sliced | 1h 59m 0s | 66.32 |
| [Stock A shoulder bottom-left, 1 of 4](stock/3mf_04/stock_05_A_shoulder_bottom_left_1_of_4.gcode.3mf) | pla04 | shared | Sliced | 46m 51s | 14.85 |
| [Stock A shoulder bottom-left, 1 of 4](stock/3mf_06hf/stock_05_A_shoulder_bottom_left_1_of_4_06hf.gcode.3mf) | pla06hf | shared | Sliced | 30m 1s | 14.18 |
| [Stock A shoulder top-left, 2 of 4](stock/3mf_04/stock_06_A_shoulder_top_left_2_of_4.gcode.3mf) | pla04 | shared | Sliced | 55m 14s | 17.38 |
| [Stock A shoulder top-left, 2 of 4](stock/3mf_06hf/stock_06_A_shoulder_top_left_2_of_4_06hf.gcode.3mf) | pla06hf | shared | Sliced | 35m 2s | 16.98 |
| [Stock A shoulder bottom-right, 3 of 4](stock/3mf_04/stock_07_A_shoulder_bottom_right_3_of_4.gcode.3mf) | pla04 | shared | Sliced | 46m 43s | 14.86 |
| [Stock A shoulder bottom-right, 3 of 4](stock/3mf_06hf/stock_07_A_shoulder_bottom_right_3_of_4_06hf.gcode.3mf) | pla06hf | shared | Sliced | 29m 55s | 14.18 |
| [Stock A shoulder top-right, 4 of 4](stock/3mf_04/stock_08_A_shoulder_top_right_4_of_4.gcode.3mf) | pla04 | shared | Sliced | 55m 32s | 17.37 |
| [Stock A shoulder top-right, 4 of 4](stock/3mf_06hf/stock_08_A_shoulder_top_right_4_of_4_06hf.gcode.3mf) | pla06hf | shared | Sliced | 34m 55s | 16.97 |
| [Stock B1 wing left, 1 of 2](stock/3mf_04/stock_09_B1_wing_left_1_of_2.gcode.3mf) | pla04 | shared | Sliced | 1h 48m 56s | 36.99 |
| [Stock B1 wing left, 1 of 2](stock/3mf_06hf/stock_09_B1_wing_left_1_of_2_06hf.gcode.3mf) | pla06hf | shared | Sliced | 1h 3m 39s | 36.17 |
| [Stock B1 wing right, 2 of 2](stock/3mf_04/stock_10_B1_wing_right_2_of_2.gcode.3mf) | pla04 | shared | Sliced | 1h 48m 55s | 37.00 |
| [Stock B1 wing right, 2 of 2](stock/3mf_06hf/stock_10_B1_wing_right_2_of_2_06hf.gcode.3mf) | pla06hf | shared | Sliced | 1h 3m 53s | 36.19 |
| [Slim LM lower, 1 of 3 — no-floor stand](slim/3mf_04/slim_01_LM_bottom_1_of_3_no_floor_stand.gcode.3mf) | pla04 | no_floor_stand | Sliced | 9h 36m 20s | 200.59 |
| [Slim LM lower, 1 of 3 — no-floor stand](slim/3mf_06hf/slim_01_LM_bottom_1_of_3_no_floor_stand_06hf.gcode.3mf) | pla06hf | no_floor_stand | Sliced | 5h 6m 47s | 197.83 |
| [Slim LM lower, 1 of 3 — floor stand](slim/3mf_04/slim_01_LM_bottom_1_of_3_floor_stand.gcode.3mf) | pla04 | floor_stand | Sliced | 16h 12m 51s | 334.10 |
| [Slim LM lower, 1 of 3 — floor stand](slim/3mf_06hf/slim_01_LM_bottom_1_of_3_floor_stand_06hf.gcode.3mf) | pla06hf | floor_stand | Sliced | 9h 7m 18s | 320.51 |
| [Slim LM middle-left, 2 of 3](slim/3mf_04/slim_02_LM_mid_left_2_of_3.gcode.3mf) | pla04 | shared | Sliced | 3h 37m 0s | 81.85 |
| [Slim LM middle-left, 2 of 3](slim/3mf_06hf/slim_02_LM_mid_left_2_of_3_06hf.gcode.3mf) | pla06hf | shared | Sliced | 2h 8m 38s | 80.41 |
| [Slim LM middle-right, 3 of 3 — shared](slim/3mf_04/slim_03_LM_mid_right_3_of_3.gcode.3mf) | pla04 | shared | Sliced | 3h 40m 30s | 84.69 |
| [Slim LM middle-right, 3 of 3 — shared](slim/3mf_06hf/slim_03_LM_mid_right_3_of_3_06hf.gcode.3mf) | pla06hf | shared | Sliced | 2h 13m 2s | 82.60 |
| [Slim UM vase/top, 1 of 1](slim/3mf_04/slim_04_UM_vase_1_of_1.gcode.3mf) | pla04 | shared | Sliced | 2h 19m 5s | 46.81 |
| [Slim UM vase/top, 1 of 1](slim/3mf_06hf/slim_04_UM_vase_1_of_1_06hf.gcode.3mf) | pla06hf | shared | Sliced | 1h 27m 27s | 45.70 |
| [Slim A shoulder bottom-left, 1 of 4](slim/3mf_04/slim_05_A_shoulder_bottom_left_1_of_4.gcode.3mf) | pla04 | shared | Sliced | 33m 43s | 9.83 |
| [Slim A shoulder bottom-left, 1 of 4](slim/3mf_06hf/slim_05_A_shoulder_bottom_left_1_of_4_06hf.gcode.3mf) | pla06hf | shared | Sliced | 22m 48s | 9.43 |
| [Slim A shoulder top-left, 2 of 4](slim/3mf_04/slim_06_A_shoulder_top_left_2_of_4.gcode.3mf) | pla04 | shared | Sliced | 40m 36s | 11.94 |
| [Slim A shoulder top-left, 2 of 4](slim/3mf_06hf/slim_06_A_shoulder_top_left_2_of_4_06hf.gcode.3mf) | pla06hf | shared | Sliced | 26m 48s | 11.71 |
| [Slim A shoulder bottom-right, 3 of 4](slim/3mf_04/slim_07_A_shoulder_bottom_right_3_of_4.gcode.3mf) | pla04 | shared | Sliced | 33m 40s | 9.84 |
| [Slim A shoulder bottom-right, 3 of 4](slim/3mf_06hf/slim_07_A_shoulder_bottom_right_3_of_4_06hf.gcode.3mf) | pla06hf | shared | Sliced | 22m 43s | 9.44 |
| [Slim A shoulder top-right, 4 of 4](slim/3mf_04/slim_08_A_shoulder_top_right_4_of_4.gcode.3mf) | pla04 | shared | Sliced | 40m 35s | 11.93 |
| [Slim A shoulder top-right, 4 of 4](slim/3mf_06hf/slim_08_A_shoulder_top_right_4_of_4_06hf.gcode.3mf) | pla06hf | shared | Sliced | 26m 49s | 11.71 |
| [Slim B1 wing left, 1 of 2](slim/3mf_04/slim_09_B1_wing_left_1_of_2.gcode.3mf) | pla04 | shared | Sliced | 1h 13m 59s | 24.39 |
| [Slim B1 wing left, 1 of 2](slim/3mf_06hf/slim_09_B1_wing_left_1_of_2_06hf.gcode.3mf) | pla06hf | shared | Sliced | 45m 34s | 23.95 |
| [Slim B1 wing right, 2 of 2](slim/3mf_04/slim_10_B1_wing_right_2_of_2.gcode.3mf) | pla04 | shared | Sliced | 1h 13m 37s | 24.39 |
| [Slim B1 wing right, 2 of 2](slim/3mf_06hf/slim_10_B1_wing_right_2_of_2_06hf.gcode.3mf) | pla06hf | shared | Sliced | 45m 32s | 23.96 |
| [Obi-Wan keyed LM bottom, 1 of 2 — no-floor stand](obiwan/3mf_06hf_petg-gf_pla/obiwan_01_LM_bottom_keyed_1_of_2_no_floor_stand_GUI.3mf) | petg_gf_gui | no_floor_stand | Slice in GUI | Pending slice | Pending slice |
| [Obi-Wan keyed LM bottom, 1 of 2 — floor stand](obiwan/3mf_06hf_petg-gf_pla/obiwan_01_LM_bottom_keyed_1_of_2_floor_stand_GUI.3mf) | petg_gf_gui | floor_stand | Slice in GUI | Pending slice | Pending slice |
| [Obi-Wan keyed LM top, 2 of 2 — shared](obiwan/3mf_06hf_petg-gf_pla/obiwan_02_LM_top_keyed_2_of_2_GUI.3mf) | petg_gf_gui | shared | Slice in GUI | Pending slice | Pending slice |
| [Obi-Wan UM carrier, 1 of 1 — shared](obiwan/3mf_06hf_petg-gf_pla/obiwan_03_UM_carrier_1_of_1_GUI.3mf) | petg_gf_gui | shared | Slice in GUI | Pending slice | Pending slice |
| [Obi-Wan tweeter crescent, 1 of 1](obiwan/3mf_06hf_petg-gf_pla/obiwan_04_T_tweeter_crescent_1_of_1_GUI.3mf) | petg_gf_gui | shared | Slice in GUI | Pending slice | Pending slice |
| [Obi-Wan NL8 service-trough snap lid, 1 of 1 — floor stand](obiwan/3mf_06hf_petg-gf_pla/obiwan_NL8_service_lid_1_of_1_GUI.3mf) | petg_gf_gui | floor_stand | Slice in GUI | Pending slice | Pending slice |
| [Obi-Wan no-floor-stand 01+02+03+04 locked combo plate](obiwan/3mf_06hf_petg-gf_pla/obiwan_01_02_03_04_LM_UM_combo_no_floor_stand_GUI.3mf) | petg_gf_gui | no_floor_stand | Slice in GUI | Pending slice | Pending slice |
| [Obi-Wan floor-stand 01+02+03+04 locked combo plate](obiwan/3mf_06hf_petg-gf_pla/obiwan_01_02_03_04_LM_UM_combo_floor_stand_GUI.3mf) | petg_gf_gui | floor_stand | Slice in GUI | Pending slice | Pending slice |
| [Obi-Wan flat wing split2, LM lower-left, 1 of 2](obiwan/3mf_04/obiwan_05_split2_flat_wing_LM_lower_left_1_of_2.gcode.3mf) | pla04 | shared | Sliced | 48m 29s | 19.71 |
| [Obi-Wan flat wing split2, LM lower-left, 1 of 2](obiwan/3mf_06hf/obiwan_05_split2_flat_wing_LM_lower_left_1_of_2_06hf.gcode.3mf) | pla06hf | shared | Sliced | 33m 7s | 18.86 |
| [Obi-Wan flat wing split2, fused LM/UM upper-left, 2 of 2](obiwan/3mf_04/obiwan_06_split2_flat_wing_LM_UM_upper_left_2_of_2.gcode.3mf) | pla04 | shared | Sliced | 1h 29m 8s | 40.33 |
| [Obi-Wan flat wing split2, fused LM/UM upper-left, 2 of 2](obiwan/3mf_06hf/obiwan_06_split2_flat_wing_LM_UM_upper_left_2_of_2_06hf.gcode.3mf) | pla06hf | shared | Sliced | 57m 54s | 38.78 |
| [Obi-Wan flat wing split2, LM lower-right, 1 of 2](obiwan/3mf_04/obiwan_08_split2_flat_wing_LM_lower_right_1_of_2.gcode.3mf) | pla04 | shared | Sliced | 48m 21s | 19.64 |
| [Obi-Wan flat wing split2, LM lower-right, 1 of 2](obiwan/3mf_06hf/obiwan_08_split2_flat_wing_LM_lower_right_1_of_2_06hf.gcode.3mf) | pla06hf | shared | Sliced | 32m 57s | 18.89 |
| [Obi-Wan flat wing split2, fused LM/UM upper-right, 2 of 2](obiwan/3mf_04/obiwan_09_split2_flat_wing_LM_UM_upper_right_2_of_2.gcode.3mf) | pla04 | shared | Sliced | 1h 30m 34s | 40.34 |
| [Obi-Wan flat wing split2, fused LM/UM upper-right, 2 of 2](obiwan/3mf_06hf/obiwan_09_split2_flat_wing_LM_UM_upper_right_2_of_2_06hf.gcode.3mf) | pla06hf | shared | Sliced | 58m 23s | 38.84 |
| [Obi-Wan flat split2 left/right wings, locked four-piece combo plate](obiwan/3mf_04/obiwan_flat_wings_split2_combo.gcode.3mf) | pla04 | shared | Sliced | 6h 30m 33s | 147.31 |
| [Obi-Wan flat split2 left/right wings, locked four-piece combo plate](obiwan/3mf_06hf/obiwan_flat_wings_split2_combo_06hf.gcode.3mf) | pla06hf | shared | Sliced | 3h 45m 51s | 144.60 |
| [Obi-Wan flat split2 left/right wings, locked four-piece combo plate](obiwan/3mf_06hf_petg-gf/obiwan_flat_wings_split2_combo_06hf_petg-gf.gcode.3mf) | petg_gf_wings | shared | Sliced | 4h 29m 57s | 137.11 |
| [Obi-Wan graded wing split2, LM lower-left, 1 of 2](obiwan/3mf_04/obiwan_11_split2_graded_wing_LM_lower_left_1_of_2.gcode.3mf) | pla04 | shared | Sliced | 47m 1s | 17.74 |
| [Obi-Wan graded wing split2, LM lower-left, 1 of 2](obiwan/3mf_06hf/obiwan_11_split2_graded_wing_LM_lower_left_1_of_2_06hf.gcode.3mf) | pla06hf | shared | Sliced | 31m 21s | 16.62 |
| [Obi-Wan graded wing split2, fused LM/UM upper-left, 2 of 2](obiwan/3mf_04/obiwan_12_split2_graded_wing_LM_UM_upper_left_2_of_2.gcode.3mf) | pla04 | shared | Sliced | 1h 24m 18s | 34.18 |
| [Obi-Wan graded wing split2, fused LM/UM upper-left, 2 of 2](obiwan/3mf_06hf/obiwan_12_split2_graded_wing_LM_UM_upper_left_2_of_2_06hf.gcode.3mf) | pla06hf | shared | Sliced | 53m 30s | 31.95 |
| [Obi-Wan graded wing split2, LM lower-right, 1 of 2](obiwan/3mf_04/obiwan_14_split2_graded_wing_LM_lower_right_1_of_2.gcode.3mf) | pla04 | shared | Sliced | 46m 30s | 17.67 |
| [Obi-Wan graded wing split2, LM lower-right, 1 of 2](obiwan/3mf_06hf/obiwan_14_split2_graded_wing_LM_lower_right_1_of_2_06hf.gcode.3mf) | pla06hf | shared | Sliced | 31m 35s | 16.66 |
| [Obi-Wan graded wing split2, fused LM/UM upper-right, 2 of 2](obiwan/3mf_04/obiwan_15_split2_graded_wing_LM_UM_upper_right_2_of_2.gcode.3mf) | pla04 | shared | Sliced | 1h 24m 34s | 34.16 |
| [Obi-Wan graded wing split2, fused LM/UM upper-right, 2 of 2](obiwan/3mf_06hf/obiwan_15_split2_graded_wing_LM_UM_upper_right_2_of_2_06hf.gcode.3mf) | pla06hf | shared | Sliced | 53m 32s | 31.91 |
| [Obi-Wan graded split2 left/right wings, locked four-piece combo plate](obiwan/3mf_04/obiwan_graded_wings_split2_combo.gcode.3mf) | pla04 | shared | Sliced | 5h 13m 55s | 118.72 |
| [Obi-Wan graded split2 left/right wings, locked four-piece combo plate](obiwan/3mf_06hf/obiwan_graded_wings_split2_combo_06hf.gcode.3mf) | pla06hf | shared | Sliced | 3h 5m 15s | 114.02 |
| [Obi-Wan graded split2 left/right wings, locked four-piece combo plate](obiwan/3mf_06hf_petg-gf/obiwan_graded_wings_split2_combo_06hf_petg-gf.gcode.3mf) | petg_gf_wings | shared | Sliced | 3h 57m 1s | 115.16 |
| [CANDIDATE Obi-Wan BMR crescent, coaxial TEBM35C10-4 pair, 1 of 1 — both drivers on one axis 86.413 mm from the MU10 axis; buries two captive magnets. Not release-authorized: absent from the released catalog and the release inventory, pending physical qualification.](obiwan/3mf_04/obiwan_17_BMR_crescent_coaxial_1_of_1.gcode.3mf) | pla04 | shared | Sliced | 3h 29m 36s | 80.17 |
| [CANDIDATE Obi-Wan BMR crescent, opposed TEBM35C10-4 pair, 1 of 1 — the two drivers fire back to back across a shared 2.40 mm partition; buries four captive magnets. Not release-authorized: absent from the released catalog and the release inventory, pending physical qualification.](obiwan/3mf_04/obiwan_18_BMR_crescent_opposed_1_of_1.gcode.3mf) | pla04 | shared | Sliced | 3h 14m 14s | 74.09 |
