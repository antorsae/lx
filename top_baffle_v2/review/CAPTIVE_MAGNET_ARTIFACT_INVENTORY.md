# Captive-magnet artifact inventory

Generated from the authoritative release catalog and the successful full
Bambu P2S slice release of 2026-07-24. There are no Rhino `.3dm` files in this
release; the directly loadable files are G-code-bearing Bambu
`.gcode.3mf` projects. Each ready project already contains its exact
`PausePrint` event, front-face-down arrangement, Arachne profile, and verified
process settings.

The 56 ready projects and their raw G-code live in the local, Git-ignored
`captive_magnet_slice_audit/slices/` workspace (about 1.3 GB). Regenerate that
workspace on a Bambu Studio host with `make bambu_slice_release`; the compact,
tracked manifest below is the release authority.

The machine-readable authority is
[`captive_magnet_pause_manifest.json`](captive_magnet_slice_audit/captive_magnet_pause_manifest.json),
with the human-readable insertion/polarity table in
[`CAPTIVE_MAGNET_PAUSE_MANIFEST.md`](captive_magnet_slice_audit/CAPTIVE_MAGNET_PAUSE_MANIFEST.md).

## Floor stand

| STL / directly loadable Bambu project | Descriptive piece | Magnets |
|---|---|---:|
| [`lx521_top_addonA_1of4_shoulder_top_left.stl`](../floor_stand/stl/lx521_top_addonA_1of4_shoulder_top_left.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_A_lx521_top_addonA_1of4_shoulder_top_left/ready/ready_to_print.gcode.3mf) | Stock A shoulder — top-left, 1 of 4 | 1 |
| [`lx521_top_addonA_2of4_shoulder_top_right.stl`](../floor_stand/stl/lx521_top_addonA_2of4_shoulder_top_right.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_A_lx521_top_addonA_2of4_shoulder_top_right/ready/ready_to_print.gcode.3mf) | Stock A shoulder — top-right, 2 of 4 | 1 |
| [`lx521_top_addonA_3of4_shoulder_bottom_left.stl`](../floor_stand/stl/lx521_top_addonA_3of4_shoulder_bottom_left.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_A_lx521_top_addonA_3of4_shoulder_bottom_left/ready/ready_to_print.gcode.3mf) | Stock A shoulder — bottom-left, 3 of 4 | 1 |
| [`lx521_top_addonA_4of4_shoulder_bottom_right.stl`](../floor_stand/stl/lx521_top_addonA_4of4_shoulder_bottom_right.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_A_lx521_top_addonA_4of4_shoulder_bottom_right/ready/ready_to_print.gcode.3mf) | Stock A shoulder — bottom-right, 4 of 4 | 1 |
| [`lx521_top_addonB1_1of2_wing_left.stl`](../floor_stand/stl/lx521_top_addonB1_1of2_wing_left.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_B1_lx521_top_addonB1_1of2_wing_left/ready/ready_to_print.gcode.3mf) | Stock B1 wing — left, 1 of 2 | 2 |
| [`lx521_top_addonB1_2of2_wing_right.stl`](../floor_stand/stl/lx521_top_addonB1_2of2_wing_right.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_B1_lx521_top_addonB1_2of2_wing_right/ready/ready_to_print.gcode.3mf) | Stock B1 wing — right, 2 of 2 | 2 |
| [`lx521_top_base_4of4_vase_b2.stl`](../floor_stand/stl/lx521_top_base_4of4_vase_b2.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_B2_lx521_top_base_4of4_vase_b2/ready/ready_to_print.gcode.3mf) | Stock B2 vase/top, 4 of 4 | 4 |
| [`lx521_top_obiwan_optional_lm_keyed_1of2_bottom.stl`](../floor_stand/stl/lx521_top_obiwan_optional_lm_keyed_1of2_bottom.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_Obi-Wan-split_lx521_top_obiwan_optional_lm_keyed_1of2_bottom/ready/ready_to_print.gcode.3mf) | Obi-Wan keyed LM bottom — LM-lower pair, integral floor state | 2 |
| [`lx521_top_obiwan_optional_lm_keyed_2of2_top.stl`](../floor_stand/stl/lx521_top_obiwan_optional_lm_keyed_2of2_top.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_Obi-Wan-split_lx521_top_obiwan_optional_lm_keyed_2of2_top/ready/ready_to_print.gcode.3mf) | Obi-Wan keyed LM top — LM-upper pair | 2 |
| [`lx521_top_obiwan_core_1of2_lm_carrier.stl`](../floor_stand/stl/lx521_top_obiwan_core_1of2_lm_carrier.stl) · **no direct P2S 3MF; use the two keyed LM projects above** | Obi-Wan LM monolith/core, P2S-oversize; exact split-proxy coverage | 4 |
| [`lx521_top_obiwan_core_2of2_um_carrier.stl`](../floor_stand/stl/lx521_top_obiwan_core_2of2_um_carrier.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_Obi-Wan_lx521_top_obiwan_core_2of2_um_carrier/ready/ready_to_print.gcode.3mf) | Obi-Wan UM carrier/core, 2 of 2 | 2 |
| [`lx521_top_v1addonA_shoulder_bottom_left.stl`](../floor_stand/stl/lx521_top_v1addonA_shoulder_bottom_left.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_V1-A_lx521_top_v1addonA_shoulder_bottom_left/ready/ready_to_print.gcode.3mf) | Slim V1 A shoulder — bottom-left | 1 |
| [`lx521_top_v1addonA_shoulder_bottom_right.stl`](../floor_stand/stl/lx521_top_v1addonA_shoulder_bottom_right.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_V1-A_lx521_top_v1addonA_shoulder_bottom_right/ready/ready_to_print.gcode.3mf) | Slim V1 A shoulder — bottom-right | 1 |
| [`lx521_top_v1addonA_shoulder_top_left.stl`](../floor_stand/stl/lx521_top_v1addonA_shoulder_top_left.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_V1-A_lx521_top_v1addonA_shoulder_top_left/ready/ready_to_print.gcode.3mf) | Slim V1 A shoulder — top-left | 1 |
| [`lx521_top_v1addonA_shoulder_top_right.stl`](../floor_stand/stl/lx521_top_v1addonA_shoulder_top_right.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_V1-A_lx521_top_v1addonA_shoulder_top_right/ready/ready_to_print.gcode.3mf) | Slim V1 A shoulder — top-right | 1 |
| [`lx521_top_v1addonB1_wing_left.stl`](../floor_stand/stl/lx521_top_v1addonB1_wing_left.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_V1-B1_lx521_top_v1addonB1_wing_left/ready/ready_to_print.gcode.3mf) | Slim V1 B1 wing — left | 2 |
| [`lx521_top_v1addonB1_wing_right.stl`](../floor_stand/stl/lx521_top_v1addonB1_wing_right.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_V1-B1_lx521_top_v1addonB1_wing_right/ready/ready_to_print.gcode.3mf) | Slim V1 B1 wing — right | 2 |
| [`lx521_top_v1l_4of4_vase_b2.stl`](../floor_stand/stl/lx521_top_v1l_4of4_vase_b2.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_V1L_lx521_top_v1l_4of4_vase_b2/ready/ready_to_print.gcode.3mf) | V1L full-slim-set vase/top, 4 of 4 | 4 |
| [`lx521_coupon_1_fit_plate.stl`](../floor_stand/stl/lx521_coupon_1_fit_plate.stl) · [ready 3MF](captive_magnet_slice_audit/slices/floor_stand_coupon1_lx521_coupon_1_fit_plate/ready/ready_to_print.gcode.3mf) | Captive-magnet fit plate, coupon 1 | 1 |

## No floor stand

| STL / directly loadable Bambu project | Descriptive piece | Magnets |
|---|---|---:|
| [`lx521_top_addonA_1of4_shoulder_top_left.stl`](../no_floor_stand/stl/lx521_top_addonA_1of4_shoulder_top_left.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_A_lx521_top_addonA_1of4_shoulder_top_left/ready/ready_to_print.gcode.3mf) | Stock A shoulder — top-left, 1 of 4 | 1 |
| [`lx521_top_addonA_2of4_shoulder_top_right.stl`](../no_floor_stand/stl/lx521_top_addonA_2of4_shoulder_top_right.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_A_lx521_top_addonA_2of4_shoulder_top_right/ready/ready_to_print.gcode.3mf) | Stock A shoulder — top-right, 2 of 4 | 1 |
| [`lx521_top_addonA_3of4_shoulder_bottom_left.stl`](../no_floor_stand/stl/lx521_top_addonA_3of4_shoulder_bottom_left.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_A_lx521_top_addonA_3of4_shoulder_bottom_left/ready/ready_to_print.gcode.3mf) | Stock A shoulder — bottom-left, 3 of 4 | 1 |
| [`lx521_top_addonA_4of4_shoulder_bottom_right.stl`](../no_floor_stand/stl/lx521_top_addonA_4of4_shoulder_bottom_right.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_A_lx521_top_addonA_4of4_shoulder_bottom_right/ready/ready_to_print.gcode.3mf) | Stock A shoulder — bottom-right, 4 of 4 | 1 |
| [`lx521_top_addonB1_1of2_wing_left.stl`](../no_floor_stand/stl/lx521_top_addonB1_1of2_wing_left.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_B1_lx521_top_addonB1_1of2_wing_left/ready/ready_to_print.gcode.3mf) | Stock B1 wing — left, 1 of 2 | 2 |
| [`lx521_top_addonB1_2of2_wing_right.stl`](../no_floor_stand/stl/lx521_top_addonB1_2of2_wing_right.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_B1_lx521_top_addonB1_2of2_wing_right/ready/ready_to_print.gcode.3mf) | Stock B1 wing — right, 2 of 2 | 2 |
| [`lx521_top_base_4of4_vase_b2.stl`](../no_floor_stand/stl/lx521_top_base_4of4_vase_b2.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_B2_lx521_top_base_4of4_vase_b2/ready/ready_to_print.gcode.3mf) | Stock B2 vase/top, 4 of 4 | 4 |
| [`lx521_top_obiwan_optional_lm_keyed_1of2_bottom.stl`](../no_floor_stand/stl/lx521_top_obiwan_optional_lm_keyed_1of2_bottom.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_Obi-Wan-split_lx521_top_obiwan_optional_lm_keyed_1of2_bottom/ready/ready_to_print.gcode.3mf) | Obi-Wan keyed LM bottom — LM-lower pair, no-floor bridge state | 2 |
| [`lx521_top_obiwan_optional_lm_keyed_2of2_top.stl`](../no_floor_stand/stl/lx521_top_obiwan_optional_lm_keyed_2of2_top.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_Obi-Wan-split_lx521_top_obiwan_optional_lm_keyed_2of2_top/ready/ready_to_print.gcode.3mf) | Obi-Wan keyed LM top — LM-upper pair | 2 |
| [`lx521_top_obiwan_core_1of2_lm_carrier.stl`](../no_floor_stand/stl/lx521_top_obiwan_core_1of2_lm_carrier.stl) · **no direct P2S 3MF; use the two keyed LM projects above** | Obi-Wan LM monolith/core, P2S-oversize; exact split-proxy coverage | 4 |
| [`lx521_top_obiwan_core_2of2_um_carrier.stl`](../no_floor_stand/stl/lx521_top_obiwan_core_2of2_um_carrier.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_Obi-Wan_lx521_top_obiwan_core_2of2_um_carrier/ready/ready_to_print.gcode.3mf) | Obi-Wan UM carrier/core, 2 of 2 | 2 |
| [`lx521_top_v1addonA_shoulder_bottom_left.stl`](../no_floor_stand/stl/lx521_top_v1addonA_shoulder_bottom_left.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_V1-A_lx521_top_v1addonA_shoulder_bottom_left/ready/ready_to_print.gcode.3mf) | Slim V1 A shoulder — bottom-left | 1 |
| [`lx521_top_v1addonA_shoulder_bottom_right.stl`](../no_floor_stand/stl/lx521_top_v1addonA_shoulder_bottom_right.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_V1-A_lx521_top_v1addonA_shoulder_bottom_right/ready/ready_to_print.gcode.3mf) | Slim V1 A shoulder — bottom-right | 1 |
| [`lx521_top_v1addonA_shoulder_top_left.stl`](../no_floor_stand/stl/lx521_top_v1addonA_shoulder_top_left.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_V1-A_lx521_top_v1addonA_shoulder_top_left/ready/ready_to_print.gcode.3mf) | Slim V1 A shoulder — top-left | 1 |
| [`lx521_top_v1addonA_shoulder_top_right.stl`](../no_floor_stand/stl/lx521_top_v1addonA_shoulder_top_right.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_V1-A_lx521_top_v1addonA_shoulder_top_right/ready/ready_to_print.gcode.3mf) | Slim V1 A shoulder — top-right | 1 |
| [`lx521_top_v1addonB1_wing_left.stl`](../no_floor_stand/stl/lx521_top_v1addonB1_wing_left.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_V1-B1_lx521_top_v1addonB1_wing_left/ready/ready_to_print.gcode.3mf) | Slim V1 B1 wing — left | 2 |
| [`lx521_top_v1addonB1_wing_right.stl`](../no_floor_stand/stl/lx521_top_v1addonB1_wing_right.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_V1-B1_lx521_top_v1addonB1_wing_right/ready/ready_to_print.gcode.3mf) | Slim V1 B1 wing — right | 2 |
| [`lx521_top_v1l_4of4_vase_b2.stl`](../no_floor_stand/stl/lx521_top_v1l_4of4_vase_b2.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_V1L_lx521_top_v1l_4of4_vase_b2/ready/ready_to_print.gcode.3mf) | V1L full-slim-set vase/top, 4 of 4 | 4 |
| [`lx521_coupon_1_fit_plate.stl`](../no_floor_stand/stl/lx521_coupon_1_fit_plate.stl) · [ready 3MF](captive_magnet_slice_audit/slices/no_floor_stand_coupon1_lx521_coupon_1_fit_plate/ready/ready_to_print.gcode.3mf) | Captive-magnet fit plate, coupon 1 | 1 |

## Shared Obi-Wan wings

| STL / directly loadable Bambu project | Descriptive piece | Magnets |
|---|---|---:|
| [`lx521_top_obiwan_wing_ac_left_1of3_lm_lower.stl`](../wings/ac/stl/lx521_top_obiwan_wing_ac_left_1of3_lm_lower.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ac_obiwan_wing_ac_left_1of3_lm_lower/ready/ready_to_print.gcode.3mf) | Obi-Wan Ac wing — left LM-lower segment, 1 of 3 | 1 |
| [`lx521_top_obiwan_wing_ac_left_2of3_lm_upper.stl`](../wings/ac/stl/lx521_top_obiwan_wing_ac_left_2of3_lm_upper.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ac_obiwan_wing_ac_left_2of3_lm_upper/ready/ready_to_print.gcode.3mf) | Obi-Wan Ac wing — left LM-upper segment, 2 of 3 | 1 |
| [`lx521_top_obiwan_wing_ac_left_3of3_um.stl`](../wings/ac/stl/lx521_top_obiwan_wing_ac_left_3of3_um.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ac_obiwan_wing_ac_left_3of3_um/ready/ready_to_print.gcode.3mf) | Obi-Wan Ac wing — left UM segment, 3 of 3 | 1 |
| [`lx521_top_obiwan_wing_ac_left_b_1of2_lm_lower.stl`](../wings/ac/stl/lx521_top_obiwan_wing_ac_left_b_1of2_lm_lower.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ac_obiwan_wing_ac_left_b_1of2_lm_lower/ready/ready_to_print.gcode.3mf) | Obi-Wan Ac wing B — left LM-lower segment, 1 of 2 | 1 |
| [`lx521_top_obiwan_wing_ac_left_b_2of2_lm_um_upper.stl`](../wings/ac/stl/lx521_top_obiwan_wing_ac_left_b_2of2_lm_um_upper.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ac_obiwan_wing_ac_left_b_2of2_lm_um_upper/ready/ready_to_print.gcode.3mf) | Obi-Wan Ac wing B — left fused LM/UM upper, 2 of 2 | 2 |
| [`lx521_top_obiwan_wing_ac_right_1of3_lm_lower.stl`](../wings/ac/stl/lx521_top_obiwan_wing_ac_right_1of3_lm_lower.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ac_obiwan_wing_ac_right_1of3_lm_lower/ready/ready_to_print.gcode.3mf) | Obi-Wan Ac wing — right LM-lower segment, 1 of 3 | 1 |
| [`lx521_top_obiwan_wing_ac_right_2of3_lm_upper.stl`](../wings/ac/stl/lx521_top_obiwan_wing_ac_right_2of3_lm_upper.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ac_obiwan_wing_ac_right_2of3_lm_upper/ready/ready_to_print.gcode.3mf) | Obi-Wan Ac wing — right LM-upper segment, 2 of 3 | 1 |
| [`lx521_top_obiwan_wing_ac_right_3of3_um.stl`](../wings/ac/stl/lx521_top_obiwan_wing_ac_right_3of3_um.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ac_obiwan_wing_ac_right_3of3_um/ready/ready_to_print.gcode.3mf) | Obi-Wan Ac wing — right UM segment, 3 of 3 | 1 |
| [`lx521_top_obiwan_wing_ac_right_b_1of2_lm_lower.stl`](../wings/ac/stl/lx521_top_obiwan_wing_ac_right_b_1of2_lm_lower.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ac_obiwan_wing_ac_right_b_1of2_lm_lower/ready/ready_to_print.gcode.3mf) | Obi-Wan Ac wing B — right LM-lower segment, 1 of 2 | 1 |
| [`lx521_top_obiwan_wing_ac_right_b_2of2_lm_um_upper.stl`](../wings/ac/stl/lx521_top_obiwan_wing_ac_right_b_2of2_lm_um_upper.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ac_obiwan_wing_ac_right_b_2of2_lm_um_upper/ready/ready_to_print.gcode.3mf) | Obi-Wan Ac wing B — right fused LM/UM upper, 2 of 2 | 2 |
| [`lx521_top_obiwan_wing_ae_left_1of3_lm_lower.stl`](../wings/ae/stl/lx521_top_obiwan_wing_ae_left_1of3_lm_lower.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ae_obiwan_wing_ae_left_1of3_lm_lower/ready/ready_to_print.gcode.3mf) | Obi-Wan Ae wing — left LM-lower segment, 1 of 3 | 1 |
| [`lx521_top_obiwan_wing_ae_left_2of3_lm_upper.stl`](../wings/ae/stl/lx521_top_obiwan_wing_ae_left_2of3_lm_upper.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ae_obiwan_wing_ae_left_2of3_lm_upper/ready/ready_to_print.gcode.3mf) | Obi-Wan Ae wing — left LM-upper segment, 2 of 3 | 1 |
| [`lx521_top_obiwan_wing_ae_left_3of3_um.stl`](../wings/ae/stl/lx521_top_obiwan_wing_ae_left_3of3_um.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ae_obiwan_wing_ae_left_3of3_um/ready/ready_to_print.gcode.3mf) | Obi-Wan Ae wing — left UM segment, 3 of 3 | 1 |
| [`lx521_top_obiwan_wing_ae_left_b_1of2_lm_lower.stl`](../wings/ae/stl/lx521_top_obiwan_wing_ae_left_b_1of2_lm_lower.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ae_obiwan_wing_ae_left_b_1of2_lm_lower/ready/ready_to_print.gcode.3mf) | Obi-Wan Ae wing B — left LM-lower segment, 1 of 2 | 1 |
| [`lx521_top_obiwan_wing_ae_left_b_2of2_lm_um_upper.stl`](../wings/ae/stl/lx521_top_obiwan_wing_ae_left_b_2of2_lm_um_upper.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ae_obiwan_wing_ae_left_b_2of2_lm_um_upper/ready/ready_to_print.gcode.3mf) | Obi-Wan Ae wing B — left fused LM/UM upper, 2 of 2 | 2 |
| [`lx521_top_obiwan_wing_ae_right_1of3_lm_lower.stl`](../wings/ae/stl/lx521_top_obiwan_wing_ae_right_1of3_lm_lower.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ae_obiwan_wing_ae_right_1of3_lm_lower/ready/ready_to_print.gcode.3mf) | Obi-Wan Ae wing — right LM-lower segment, 1 of 3 | 1 |
| [`lx521_top_obiwan_wing_ae_right_2of3_lm_upper.stl`](../wings/ae/stl/lx521_top_obiwan_wing_ae_right_2of3_lm_upper.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ae_obiwan_wing_ae_right_2of3_lm_upper/ready/ready_to_print.gcode.3mf) | Obi-Wan Ae wing — right LM-upper segment, 2 of 3 | 1 |
| [`lx521_top_obiwan_wing_ae_right_3of3_um.stl`](../wings/ae/stl/lx521_top_obiwan_wing_ae_right_3of3_um.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ae_obiwan_wing_ae_right_3of3_um/ready/ready_to_print.gcode.3mf) | Obi-Wan Ae wing — right UM segment, 3 of 3 | 1 |
| [`lx521_top_obiwan_wing_ae_right_b_1of2_lm_lower.stl`](../wings/ae/stl/lx521_top_obiwan_wing_ae_right_b_1of2_lm_lower.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ae_obiwan_wing_ae_right_b_1of2_lm_lower/ready/ready_to_print.gcode.3mf) | Obi-Wan Ae wing B — right LM-lower segment, 1 of 2 | 1 |
| [`lx521_top_obiwan_wing_ae_right_b_2of2_lm_um_upper.stl`](../wings/ae/stl/lx521_top_obiwan_wing_ae_right_b_2of2_lm_um_upper.stl) · [ready 3MF](captive_magnet_slice_audit/slices/shared_Obi-Wan-Ae_obiwan_wing_ae_right_b_2of2_lm_um_upper/ready/ready_to_print.gcode.3mf) | Obi-Wan Ae wing B — right fused LM/UM upper, 2 of 2 | 2 |

## Count reconciliation

| Scope | Catalog artifacts | Catalog magnet stations | Direct ready projects | Direct insertions |
|---|---:|---:|---:|---:|
| Floor stand | 19 | 35 | 18 | 31 |
| No floor stand | 19 | 35 | 18 | 31 |
| Shared wings | 20 | 24 | 20 | 24 |
| **Total** | **58** | **94** | **56** | **86** |

All 86 directly sliced insertions are transverse sites. The remaining eight
catalog stations are the two four-magnet Obi-Wan
LM monoliths; they duplicate the exact keyed-half cavities for audit coverage
and have no P2S project. Floor/no-floor, Ac/Ae, and monolith/keyed split are
alternative choices, so 94 is an audit-inventory count rather than a bill of
materials. A physical Obi-Wan build using either LM form and one wing family
uses 12 magnets: six in the carrier and six in the wing receivers.
