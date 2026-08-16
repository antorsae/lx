; HEADER_BLOCK_START
; BambuStudio 02.07.01.62
; model printing time: 12m 25s; total estimated time: 19m 27s
; total layer number: 72
; total filament length [mm] : 2616.90
; total filament volume [cm^3] : 6294.39
; total filament weight [g] : 7.93
; filament_density: 1.26
; filament_diameter: 1.75
; max_z_height: 11.56
; filament: 1
; HEADER_BLOCK_END

; CONFIG_BLOCK_START
; accel_to_decel_enable = 0
; accel_to_decel_factor = 50%
; activate_air_filtration = 0
; additional_cooling_fan_speed = 70
; additional_fan_full_speed_layer = 0
; alternate_extra_wall = 0
; apply_scarf_seam_on_circles = 1
; auxiliary_fan = 1
; avoid_crossing_wall_includes_support = 0
; bed_custom_model = 
; bed_custom_texture = 
; bed_exclude_area = 
; bed_temperature_formula = by_first_filament
; before_layer_change_gcode = 
; best_object_pos = 0.5,0.5
; bottom_color_penetration_layers = 3
; bottom_shell_layers = 5
; bottom_shell_thickness = 0
; bottom_surface_density = 100%
; bottom_surface_pattern = monotonic
; bridge_angle = 0
; bridge_flow = 1
; bridge_no_support = 0
; bridge_speed = 30
; brim_object_gap = 0.1
; brim_type = auto_brim
; brim_width = 5
; chamber_temperatures = 0
; change_filament_gcode = ;======== P2S filament_change gcode ==========\n;===== 2026/05/15 =====\n\nM620 S[next_filament_id]A\nM204 S9000\n{if toolchange_count > 1 && (z_hop_types[current_filament_id] == 0 || z_hop_types[current_filament_id] == 3)}\nG17\nG2 Z{z_after_toolchange + 0.4} I0.86 J0.86 P1 F10000 ; spiral lift a little from second lift\n{endif}\n\n;nozzle_change_gcode\n\nG1 Z{max_layer_z + 3.0} F1200\n\nM400\nM106 P1 S0\n\n{if toolchange_count == 2}\n; get travel path for change filament\n;M620.1 X[travel_point_1_x] Y[travel_point_1_y] F21000 P0\n;M620.1 X[travel_point_2_x] Y[travel_point_2_y] F21000 P1\n;M620.1 X[travel_point_3_x] Y[travel_point_3_y] F21000 P2\n{endif}\n\n{if ((filament_type[current_filament_id] == \"PLA\") || (filament_type[current_filament_id] == \"PLA-CF\") || (filament_type[current_filament_id] == \"PETG\")) && (nozzle_diameter_at_nozzle_id[current_nozzle_id] == 0.2)}\nM620.10 A0 F74.8347 L[flush_length] H{nozzle_diameter_at_nozzle_id[current_nozzle_id]} T{flush_temperatures[current_filament_id]} P[old_filament_temp] S1\n{else}\nM620.10 A0 F{flush_volumetric_speeds[current_filament_id]/2.4053*60} L[flush_length] H{nozzle_diameter_at_nozzle_id[current_nozzle_id]} T{flush_temperatures[current_filament_id]} P[old_filament_temp] S1\n{endif}\n\n{if ((filament_type[next_filament_id] == \"PLA\") || (filament_type[next_filament_id] == \"PLA-CF\") || (filament_type[next_filament_id] == \"PETG\")) && (nozzle_diameter_at_nozzle_id[next_nozzle_id] == 0.2)}\nM620.10 A1 F74.8347 L[flush_length] H{nozzle_diameter_at_nozzle_id[next_nozzle_id]} T{flush_temperatures[next_filament_id]} P[new_filament_temp] S1\n{else}\nM620.10 A1 F{flush_volumetric_speeds[next_filament_id]/2.4053*60} L[flush_length] H{nozzle_diameter_at_nozzle_id[next_nozzle_id]} T{flush_temperatures[next_filament_id]} P[new_filament_temp] S1\n{endif}\n\nM620.15 C{new_filament_temp - filament_cooling_before_tower[next_filament_id]}\n\n{if long_retraction_when_cut}\nM620.11 P1 L0 I[current_filament_id] E-{retraction_distance_when_cut} F{max((flush_volumetric_speeds[current_filament_id]/2.4053*60), 200)}\n{else}\nM620.11 P0 L0 I[current_filament_id] E0\n{endif}\n\nM620.11 K0 I[current_filament_id] R0\n\n\nT[next_filament_id]\n\n;deretract\n{if filament_type[next_filament_id] == \"TPU\"}\n{else}\n{if filament_type[next_filament_id] == \"PA\"}\n;VG1 E1 F{max(new_filament_e_feedrate, 200)}\n;VG1 E1 F{max(new_filament_e_feedrate/2, 100)}\n{else}\n;VG1 E4 F{max(new_filament_e_feedrate, 200)}\n;VG1 E4 F{max(new_filament_e_feedrate/2, 100)}\n{endif}\n{endif}\n\n; VFLUSH_START\n{if flush_length>41.5}\n;VG1 E41.5 F{min(old_filament_e_feedrate,new_filament_e_feedrate)}\n;VG1 E{flush_length-41.5} F{new_filament_e_feedrate}\n{else}\n;VG1 E{flush_length} F{min(old_filament_e_feedrate,new_filament_e_feedrate)}\n{endif}\nSYNC T{ceil(flush_length / 80) * 5}\n; VFLUSH_END\n\nM1002 set_filament_type:{filament_type[next_filament_id]}\n\nM400\nM83\n{if next_filament_id < 255}\nM620.10 R{retract_length_toolchange[filament_map[next_filament_id]-1]}\nM628 S0\n;VM109 S[new_filament_temp]\n\nM629\nM400\nM983.3 F{filament_max_volumetric_speed[next_filament_id]/2.4} A0.4 R{retract_length_toolchange[filament_map[next_filament_id]-1]}\nM400\nG1 Y247 F30000\nG1 Y217 F18000\n\nG1 Z{max_layer_z + 3.0} F3000\n{if layer_z <= (initial_layer_print_height + 0.001)}\nM204 S[initial_layer_acceleration]\n{else}\nM204 S[travel_acceleration]\n{endif}\n\n{else}\nG1 X[x_after_toolchange] Y[y_after_toolchange] Z[z_after_toolchange] F12000\n{endif}\n\nM621 S[next_filament_id]A\n\nM622.1 S0 ;for prev version, default skip\nM1002 judge_flag powerloss_resume_flag\nM622 J1\nM983.3 F{filament_max_volumetric_speed[next_filament_id]/2.4} A0.4 R{retract_length_toolchange[filament_map[next_filament_id]-1]}\nM400\nG1 Y247 F30000\nG1 Y217 F18000\nG1 Z{max_layer_z + 3.0} F3000\n{if layer_z <= (initial_layer_print_height + 0.001)}\nM204 S[initial_layer_acceleration]\n{else}\nM204 S[travel_acceleration]\n{endif}\nM1002 set_flag powerloss_resume_flag=0\nM623\n\n{if (filament_type[next_filament_id] == \"PLA\") ||  (filament_type[next_filament_id] == \"PETG\")\n ||  (filament_type[next_filament_id] == \"PLA-CF\")  ||  (filament_type[next_filament_id] == \"PETG-CF\")}\nM1015.4 S1 K1 H{nozzle_diameter_at_nozzle_id[next_nozzle_id]} ;enable E air printing detect\n{else}\nM1015.4 S0 K0 H{nozzle_diameter_at_nozzle_id[next_nozzle_id]} ;disable E air printing detect\n{endif}\n\nM620.6 I[next_filament_id] W1 ;enable ams air printing detect\n\nG1 Y256 F18000\n\n{if (overall_chamber_temperature < 40)}\n{if (layer_num + 1 <= close_additional_fan_first_x_layers[next_filament_id])}\n    {if (min_vitrification_temperature <= 50)}\n        M106 P2 S{first_x_layer_fan_speed[next_filament_id]*255.0/100.0 };set first x_layer aux fan\n    {endif}\n{elsif (layer_num + 1 < additional_fan_full_speed_layer[next_filament_id] && additional_fan_full_speed_layer[next_filament_id] > close_additional_fan_first_x_layers[next_filament_id])}\n    {if (min_vitrification_temperature <= 50)}\n        M106 P2 S{(first_x_layer_fan_speed[next_filament_id] + (additional_cooling_fan_speed[next_filament_id] - first_x_layer_fan_speed[next_filament_id]) * (layer_num + 1 - close_additional_fan_first_x_layers[next_filament_id]) / max(additional_fan_full_speed_layer[next_filament_id] - close_additional_fan_first_x_layers[next_filament_id], 1)) * 255.0/100.0}\n    {endif}\n{else}\n    {if (min_vitrification_temperature <= 50)}\n        {if (nozzle_diameter_at_nozzle_id[current_nozzle_id] == 0.2)}\n            M142 P1 R30 S40 U{max_additional_fan/100.0} V1.0 O45; set PLA/TPU ND0.2 chamber autocooling\n        {else}\n            M142 P1 R30 S40 U{max_additional_fan/100.0} V1.0 O45; set PLA/TPU ND0.4 chamber autocooling\n        {endif}\n    {endif}\n{endif}\n{endif}\n\nM622.1 S0\nM1002 judge_flag ventobox_replace_aux1_fan_flag\nM622 J0\n{if (layer_num + 1 <= close_additional_fan_first_x_layers[next_filament_id])}\n    M106 P10 S{first_x_layer_fan_speed[next_filament_id]*255.0/100.0 };set first x_layer left aux fan\n{elsif (layer_num + 1 < additional_fan_full_speed_layer[next_filament_id] && additional_fan_full_speed_layer[next_filament_id] > close_additional_fan_first_x_layers[next_filament_id])}\n    M106 P10 S{(first_x_layer_fan_speed[next_filament_id] + (additional_cooling_fan_speed[next_filament_id] - first_x_layer_fan_speed[next_filament_id]) * (layer_num + 1 - close_additional_fan_first_x_layers[next_filament_id]) / max(additional_fan_full_speed_layer[next_filament_id] - close_additional_fan_first_x_layers[next_filament_id], 1)) * 255.0/100.0}\n{else}\n    M106 P10 S{additional_cooling_fan_speed[next_filament_id]*255.0/100.0};set left aux fan\n{endif}\nM623\n\n;not set fan changing filament
; circle_compensation_manual_offset = 0
; circle_compensation_speed = 200
; close_additional_fan_first_x_layers = 1
; close_fan_the_first_x_layers = 1
; compatible_printers_condition = 
; complete_print_exhaust_fan_speed = 70
; cool_plate_temp = 35
; cool_plate_temp_initial_layer = 35
; cooling_filter_enabled = 0
; cooling_perimeter_transition_distance = 10
; cooling_slowdown_logic = uniform_cooling
; counter_coef_1 = 0
; counter_coef_2 = 0.0025
; counter_coef_3 = 0.014
; counter_limit_max = 0.076
; counter_limit_min = 0.014
; curr_bed_type = Textured PEI Plate
; default_acceleration = 10000
; default_filament_colour = ""
; default_filament_profile = "Bambu PLA Basic @BBL P2S 0.6 nozzle"
; default_jerk = 0
; default_nozzle_volume_type = High Flow
; default_print_profile = 0.30mm Standard @BBL P2S 0.6 nozzle
; deretraction_speed = 30
; detect_floating_vertical_shell = 1
; detect_narrow_internal_solid_infill = 1
; detect_overhang_wall = 1
; detect_thin_wall = 1
; diameter_limit = 50
; different_settings_to_system = ;;
; draft_shield = disabled
; during_print_exhaust_fan_speed = 70
; elefant_foot_compensation = 0.15
; embedding_wall_into_infill = 0
; enable_arc_fitting = 1
; enable_circle_compensation = 0
; enable_filament_dynamic_map = 0
; enable_height_slowdown = 0
; enable_long_retraction_when_cut = 2
; enable_mixed_color_sublayer = 0
; enable_order_independent_overlap_carving = 0
; enable_overhang_bridge_fan = 1
; enable_overhang_speed = 1
; enable_pre_heating = 0
; enable_pressure_advance = 0
; enable_prime_tower = 1
; enable_support = 0
; enable_support_ironing = 0
; enable_tower_interface_features = 0
; enable_wrapping_detection = 0
; enforce_support_layers = 0
; eng_plate_temp = 55
; eng_plate_temp_initial_layer = 55
; ensure_vertical_shell_thickness = enabled
; exclude_object = 1
; extruder_ams_count = 
; extruder_clearance_dist_to_rod = 36.5
; extruder_clearance_height_to_lid = 141.5
; extruder_clearance_height_to_rod = 32.5
; extruder_clearance_max_radius = 72
; extruder_colour = #018001
; extruder_max_nozzle_count = 1
; extruder_nozzle_stats = 
; extruder_offset = 0x0
; extruder_printable_area = 
; extruder_type = Direct Drive
; extruder_variant_list = "Direct Drive Standard,Direct Drive High Flow"
; fan_cooling_layer_time = 100
; fan_direction = right
; fan_max_speed = 100
; fan_min_speed = 100
; filament_adaptive_volumetric_speed = 0
; filament_adhesiveness_category = 100
; filament_bridge_speed = 25
; filament_change_length = 5
; filament_change_length_nc = 10
; filament_colour = #00AE42
; filament_cooling_before_tower = 10
; filament_cost = 24.99
; filament_density = 1.26
; filament_dev_ams_drying_ams_limitations = 1
; filament_dev_ams_drying_heat_distortion_temperature = 45
; filament_dev_ams_drying_temperature = 45
; filament_dev_ams_drying_time = 12
; filament_dev_chamber_drying_bed_temperature = 70
; filament_dev_chamber_drying_time = 12
; filament_dev_drying_cooling_temperature = 45
; filament_dev_drying_softening_temperature = 50
; filament_diameter = 1.75
; filament_enable_overhang_speed = 1
; filament_end_gcode = "; filament end gcode \n\n"
; filament_extruder_compatibility = 0
; filament_extruder_variant = "Direct Drive Standard"
; filament_flow_ratio = 0.98
; filament_flush_temp = 0
; filament_flush_temp_fast = 0
; filament_flush_volumetric_speed = 0
; filament_ids = GFA00
; filament_is_mixed = 0
; filament_is_support = 0
; filament_map = 1
; filament_map_2 = 0
; filament_map_mode = Auto For Flush
; filament_max_volumetric_speed = 21
; filament_metal_stickiness = None
; filament_minimal_purge_on_wipe_tower = 15
; filament_mixed_components = ""
; filament_mixed_gradient = 0
; filament_mixed_gradient_curve = ""
; filament_mixed_gradient_per_part = 0
; filament_mixed_gradient_range = ""
; filament_mixed_sublayer_ratios = ""
; filament_notes = 
; filament_nozzle_map = 0
; filament_overhang_1_4_speed = 0
; filament_overhang_2_4_speed = 50
; filament_overhang_3_4_speed = 30
; filament_overhang_4_4_speed = 10
; filament_overhang_totally_speed = 10
; filament_pre_cooling_temperature = 0
; filament_pre_cooling_temperature_nc = 0
; filament_preheat_temperature_delta = 10
; filament_prime_volume = 30
; filament_prime_volume_nc = 60
; filament_printable = 3
; filament_ramming_travel_time = 0
; filament_ramming_travel_time_nc = 0
; filament_ramming_volumetric_speed = -1
; filament_ramming_volumetric_speed_nc = -1
; filament_retract_length_nc = 14
; filament_retraction_distances_when_cut = 10
; filament_retraction_length = 0.4
; filament_scarf_gap = 0%
; filament_scarf_height = 10%
; filament_scarf_length = 10
; filament_scarf_seam_type = none
; filament_self_index = 1
; filament_settings_id = "Bambu PLA Basic @BBL P2S 0.6 nozzle"
; filament_shrink = 100%
; filament_soluble = 0
; filament_start_gcode = "; filament start gcode\n"
; filament_tower_interface_pre_extrusion_dist = 10
; filament_tower_interface_pre_extrusion_length = 0
; filament_tower_interface_print_temp = -1
; filament_tower_interface_purge_volume = 20
; filament_tower_ironing_area = 4
; filament_type = PLA
; filament_velocity_adaptation_factor = 1
; filament_vendor = "Bambu Lab"
; filament_volume_map = 0
; filament_wipe = 1
; filament_wipe_distance = 1
; filename_format = {input_filename_base}_{filament_type[0]}_{print_time}.gcode
; fill_multiline = 1
; filter_out_gap_fill = 0
; first_layer_print_sequence = 0
; first_x_layer_fan_speed = 40
; first_x_layer_part_fan_speed = 0
; flush_into_infill = 0
; flush_into_objects = 0
; flush_into_support = 1
; flush_multiplier = 1
; flush_multiplier_fast = 1.2
; flush_volumes_matrix = 0,280,280,280,280,0,280,280,280,280,0,280,280,280,280,0
; flush_volumes_vector = 140,140,140,140,140,140,140,140
; full_fan_speed_layer = 0
; fuzzy_skin = none
; fuzzy_skin_first_layer = 0
; fuzzy_skin_mode = displacement
; fuzzy_skin_noise_type = classic
; fuzzy_skin_octaves = 4
; fuzzy_skin_persistence = 0.5
; fuzzy_skin_point_distance = 0.8
; fuzzy_skin_scale = 1
; fuzzy_skin_thickness = 0.3
; gap_infill_speed = 50
; gcode_add_line_number = 0
; gcode_flavor = marlin
; grab_length = 0
; group_algo_with_time = 0
; has_filament_switcher = 0
; has_scarf_joint_seam = 0
; head_wrap_detect_zone = 
; hole_coef_1 = 0
; hole_coef_2 = -0.0028
; hole_coef_3 = 0.12
; hole_limit_max = 0.12
; hole_limit_min = 0.05
; hot_plate_temp = 55
; hot_plate_temp_initial_layer = 55
; hotend_cooling_rate = 2
; hotend_heating_rate = 2
; impact_strength_z = 13.8
; independent_support_layer_height = 1
; infill_combination = 0
; infill_direction = 45
; infill_instead_top_bottom_surfaces = 0
; infill_jerk = 9
; infill_lock_depth = 1
; infill_rotate_step = 0
; infill_shift_step = 0.4
; infill_wall_overlap = 15%
; inherits_group = ;;
; initial_layer_acceleration = 500
; initial_layer_flow_ratio = 1
; initial_layer_infill_speed = 55
; initial_layer_jerk = 9
; initial_layer_line_width = 0.62
; initial_layer_print_height = 0.2
; initial_layer_speed = 35
; initial_layer_travel_acceleration = 6000
; inner_wall_acceleration = 0
; inner_wall_jerk = 9
; inner_wall_line_width = 0.62
; inner_wall_speed = 300
; interface_shells = 0
; interlocking_beam = 0
; interlocking_beam_layer_count = 2
; interlocking_beam_width = 0.8
; interlocking_boundary_avoidance = 2
; interlocking_depth = 2
; interlocking_orientation = 22.5
; internal_bridge_support_thickness = 0.8
; internal_solid_infill_line_width = 0.62
; internal_solid_infill_pattern = zig-zag
; internal_solid_infill_speed = 250
; ironing_direction = 45
; ironing_fan_speed = -1
; ironing_flow = 10%
; ironing_inset = 0.31
; ironing_pattern = zig-zag
; ironing_spacing = 0.15
; ironing_speed = 30
; ironing_type = no ironing
; is_infill_first = 0
; layer_change_gcode = ;======== P2S layer_change gcode ==========\n;===== 2026/05/15 ====\n\n{if (layer_num + 1 == 1)}\n{if (overall_chamber_temperature >= 40)}\n    ;not reset filter fan in first layer\n    ;not reset fan\n{else}\n{if (min_vitrification_temperature > 50)}\n    ;not reset filter fan in first layer\n    ;not reset fan\n{endif}\n{endif}\n{endif}\n\n{if (layer_num + 1 <= close_additional_fan_first_x_layers[current_filament_id])}\n{if (overall_chamber_temperature < 40)}\n    {if (min_vitrification_temperature <= 50)}\n        M106 P2 S{first_x_layer_fan_speed[current_filament_id]*255.0/100.0}\n    {endif}\n{endif}\n    M622.1 S0\n    M1002 judge_flag ventobox_replace_aux1_fan_flag\n    M622 J0\n    M106 P10 S{first_x_layer_fan_speed[current_filament_id]*255.0/100.0}; set first x_layer left aux fan\n    M623\n;not reset fan\n{elsif (layer_num + 1 < additional_fan_full_speed_layer[current_filament_id] && additional_fan_full_speed_layer[current_filament_id] > close_additional_fan_first_x_layers[current_filament_id])}\n{if (overall_chamber_temperature < 40)}\n    {if (min_vitrification_temperature <= 50)}\n        M106 P2 S{(first_x_layer_fan_speed[current_filament_id] + (additional_cooling_fan_speed[current_filament_id] - first_x_layer_fan_speed[current_filament_id]) * (layer_num + 1 - close_additional_fan_first_x_layers[current_filament_id]) / max(additional_fan_full_speed_layer[current_filament_id] - close_additional_fan_first_x_layers[current_filament_id], 1)) * 255.0/100.0}\n    {endif}\n{endif}\n    M622.1 S0\n    M1002 judge_flag ventobox_replace_aux1_fan_flag\n    M622 J0\n    M106 P10 S{(first_x_layer_fan_speed[current_filament_id] + (additional_cooling_fan_speed[current_filament_id] - first_x_layer_fan_speed[current_filament_id]) * (layer_num + 1 - close_additional_fan_first_x_layers[current_filament_id]) / max(additional_fan_full_speed_layer[current_filament_id] - close_additional_fan_first_x_layers[current_filament_id], 1)) * 255.0/100.0}\n    M623\n;not reset fan\n{elsif (layer_num + 1 == max(close_additional_fan_first_x_layers[current_filament_id] + 1, additional_fan_full_speed_layer[current_filament_id]))}\n{if (overall_chamber_temperature < 40)}\n    ;updata chamber autocooling in Xth layer\n    {if (min_vitrification_temperature <= 50)}\n        {if (nozzle_diameter_at_nozzle_id[current_nozzle_id] == 0.2)}\n            M142 P1 R30 S40 U{max_additional_fan/100.0} V1.0 O45; set PLA/TPU ND0.2 chamber autocooling\n        {else}\n            M142 P1 R30 S40 U{max_additional_fan/100.0} V1.0 O45; set PLA/TPU ND0.4 chamber autocooling\n        {endif}\n    {else}\n            ;not reset filter fan in Xth layer\n    {endif}\n{else}\n        ;not reset filter fan in Xth layer\n{endif}\n    M622.1 S0\n    M1002 judge_flag ventobox_replace_aux1_fan_flag\n    M622 J0\n    M106 P10 S{additional_cooling_fan_speed[current_filament_id]*255.0/100.0}; set left aux fan\n    M623\n;not reset fan\n{endif}\n\n; update layer progress\nM73 L{layer_num+1}\nM991 S0 P{layer_num} ;notify layer change\n\n
; layer_height = 0.16
; line_width = 0.62
; locked_skeleton_infill_pattern = zigzag
; locked_skin_infill_pattern = crosszag
; long_retractions_when_cut = 0
; long_retractions_when_ec = 0
; machine_bed_mass_Y = 0
; machine_end_gcode = ;======== P2S end gcode ==========\n;===== 2026/05/18 =====\nM400 ; wait for buffer to clear\nG92 E0 ; zero the extruder\nM211 Z1\n\nG90\nG1 Z{max_layer_z + 0.4} F900 ; lower z a little\nM1002 judge_flag timelapse_record_flag\nM622 J1\n    G150.3\n    M400 ; wait all motion done\n    M991 S0 P-1 ;end smooth timelapse at safe pos\n    M400 S5 ;wait for last picture to be taken\nM623  ;end of \"timelapse_record_flag\n\nG90\nG1 Z{max_layer_z + 10} F900 ; lower z a little\n\nM140 S0 ; turn off bed\nM106 S0 ; turn off fan\nM106 P2 S0 ; turn off remote part cooling fan\nM106 P3 S0 ; turn off chamber cooling fan\nM106 P10 S0 ; turn off left aux fan\n\n; pull back filament to AMS\nM620 S65535\nT65535\nG150.1 F8000\nM621 S65535\n\nG150.3\nM104 S0 ; turn off hotend\nM400 ; wait all motion done\nM17 S\nM17 Z0.4 ; lower z motor current to reduce impact if there is something in the bottom\n{if (80.0 - max_layer_z/2) > 0}\n    {if (max_layer_z + 80.0 - max_layer_z/2) < 256}\n        G1 Z{max_layer_z + 80.0 - max_layer_z/2} F600\n        G1 Z{max_layer_z + 78.0 - max_layer_z/2}\n    {else}\n        G1 Z256 F600\n        G1 Z256\n    {endif}\n{else}\n    {if (max_layer_z + 4.0) < 256}\n        G1 Z{max_layer_z + 4.0} F600\n        G1 Z{max_layer_z + 2.0}\n    {else}\n        G1 Z256 F600\n        G1 Z256\n    {endif}\n{endif}\nM400 P100\nM17 R ; restore z current\n\n\nM220 S100  ; Reset feedrate magnitude\nM201.2 K1.0 ; Reset acc magnitude\nM73.2 R1.0 ;Reset left time magnitude\nM1002 set_gcode_claim_speed_level : 0\n\nM1015.3 S0 ;disable clog detect\nM1015.4 S0 K0 ;disable air printing detect\n\n;=====printer finish air purification=========\nM622.1 S0\nM1002 judge_flag print_finish_air_filt_flag\n\nM622 J1\nM1002 gcode_claim_action : 66\nM145 P1\nM106 P2 S255\nM400 S180\nM106 P2 S0\nM623\n\nM622 J2\nM1002 gcode_claim_action : 66\nM145 P0\nM106 P3 S255\nM400 S180\nM106 P3 S0\nM623\n;=====printer finish air purification=========\n\n;=====printer finish  sound=========\nM17\nM400 S1\nM1006 S1\nM1006 A53 B10 L50 C53 D10 M50 E53 F10 N50 \nM1006 A57 B10 L50 C57 D10 M50 E57 F10 N50 \nM1006 A0 B15 L0 C0 D15 M0 E0 F15 N0 \nM1006 A53 B10 L50 C53 D10 M50 E53 F10 N50 \nM1006 A57 B10 L50 C57 D10 M50 E57 F10 N50 \nM1006 A0 B15 L0 C0 D15 M0 E0 F15 N0 \nM1006 A48 B10 L50 C48 D10 M50 E48 F10 N50 \nM1006 A0 B15 L0 C0 D15 M0 E0 F15 N0 \nM1006 A60 B10 L50 C60 D10 M50 E60 F10 N50 \nM1006 W\n;=====printer finish  sound=========\nM400\nM18\n
; machine_hotend_change_time = 0
; machine_load_filament_time = 26
; machine_max_acceleration_e = 5000,5000
; machine_max_acceleration_extruding = 20000,20000
; machine_max_acceleration_retracting = 5000,5000
; machine_max_acceleration_travel = 10000,10000
; machine_max_acceleration_x = 20000,20000
; machine_max_acceleration_y = 20000,20000
; machine_max_acceleration_z = 500,500
; machine_max_force_Y = 0
; machine_max_jerk_e = 2.5,2.5
; machine_max_jerk_x = 9,9
; machine_max_jerk_y = 9,9
; machine_max_jerk_z = 3,3
; machine_max_printed_mass = 0
; machine_max_speed_e = 30,30
; machine_max_speed_x = 600,600
; machine_max_speed_y = 600,600
; machine_max_speed_z = 20,20
; machine_min_extruding_rate = 0
; machine_min_travel_rate = 0
; machine_pause_gcode = M400 U1
; machine_prepare_compensation_time = 370
; machine_start_gcode = ;M1002 set_flag extrude_cali_flag=1\n;M1002 set_flag g29_before_print_flag=1\n;M1002 set_flag auto_cali_toolhead_offset_flag=1\n;M1002 set_flag build_plate_detect_flag=1\n\n;======== P2S start gcode==========\n;===== 2026/05/18 =====\n\n  M140 S[bed_temperature_initial_layer_single] ; heat heatbed first\n  M993 A0 B0 C0 ; nozzle cam detection not allowed.\n  M400\n\n;=====printer start sound ===================\nM17\nM400 S1\nM1006 S1\nM1006 A53 B9 L50 C53 D9 M50 E53 F9 N50\nM1006 A56 B9 L50 C56 D9 M50 E56 F9 N50\nM1006 A61 B9 L50 C61 D9 M50 E61 F9 N50\nM1006 A53 B9 L50 C53 D9 M50 E53 F9 N50\nM1006 A56 B9 L50 C56 D9 M50 E56 F9 N50\nM1006 A61 B18 L50 C61 D18 M50 E61 F18 N50\nM1006 W\n;=====printer start sound ===================\n\n  M620 M ;enable remap\n  G389\n\n;===== avoid end stop =================\n  G91\n  G380 S2 Z22 F1200\n  G380 S2 Z-12 F1200\n  G90\n;===== avoid end stop =================\n\n;===== reset machine status =================\n  M204 S10000\n  M630 S0 P1\n  G90\n  M17 D ; reset motor current to default\n  M960 S5 P1 ; turn on logo lamp\n  G90\n  M220 S100 ;Reset Feedrate\n  M1002 set_gcode_claim_speed_level: 5\n  M221 S100 ;Reset Flowrate\n  M73.2   R1.0 ;Reset left time magnitude\n  G29.1 Z{+0.0} ; clear z-trim value first\n  M983.1 M1\n  M982.2 S1 ; turn on cog noise reduction\n  M983.4 S0\n;===== reset machine status =================\n\n;==== set airduct mode ====\n;==== if Chamber Cooling is necessary ====\n{if (overall_chamber_temperature >= 40)}\nM145 P1 ; set airduct mode to heating mode for heating\nM106 P2 S255 ; turn on filter fan\nM622.1 S0\nM1002 judge_flag ventobox_replace_aux1_fan_flag\nM622 J0\nM106 P10 S0 ; turn off left aux fan\nM623\n{else}\n{if (min_vitrification_temperature <= 50)}\nM145 P0 ; set airduct mode to cooling mode for cooling\nM106 P2 S255 ; turn on auxiliary fan for cooling\nM106 P3 S127 ; turn on chamber fan for cooling\nM1002 gcode_claim_action : 29\nM191 S0 ; wait for chamber temp\nM106 P2 S102 ; turn on chamber cooling fan\nM622.1 S0\nM1002 judge_flag ventobox_replace_aux1_fan_flag\nM622 J0\nM106 P10 S0 ; turn off left aux fan\nM623\nM142 P6 R30 S40 U0.3 V0.8 ; set PETG exhaust chamber autocooling\n{else}\nM145 P1 ; set airduct mode to heating mode for heating\nM106 P2 S127 ; turn on 50% filter fan\nM142 P6 R30 S40 U0.3 V0.8 ; set PLA/TPU exhaust chamber autocooling\n{endif}\n{endif}\n;==== set airduct mode ====\n\n;===== start to heat heatbed & hotend==========\n  M1002 gcode_claim_action : 2\n  M1002 set_filament_type:{filament_type[initial_no_support_filament_id]}\n  M104 S140 A\n\n  G29.2 S0 ; avoid invalid abl data\n\n;===== first homing start =====\n  M1002 gcode_claim_action : 13\n  G28 X T300\n  G150.1 F8000 ; wipe mouth to avoid filament stick to heatbed\n  G150.3\n  M972 S24 P0\n  M972 S26 P0 C0\n  M972 S42 P0 T5000\n  G150.1 F8000 ; wipe mouth to avoid filament stick to heatbed\n  G90\n  G1 X128 Y128 F30000\n  G28 Z P0 T400\n  M400\n;===== first homign end =====\n\n;===== detection start =====\n  M1002 gcode_claim_action : 11\n  M104 S{nozzle_temperature_initial_layer[initial_no_support_filament_id]-80} A ; rise temp in advance\n  M972 S19 P0 T5000 ;plate type detection\n\n  {if max_print_z >= 145}\n    M1002 gcode_claim_action : 75 ;  Detect obstacles at the botton of the heated bed\n    G150.3\n    M104 S{nozzle_temperature_initial_layer[initial_no_support_filament_id]} ; rise temp in advance\n    G3811 Z{max_print_z}  ; Detect obstacles at the bottom of the heated bed\n  {endif}\n;===== detection end =====\n\n;===== prepare print temperature and material ==========\n  M400\n  M211 X0 Y0 Z0 ;turn off soft endstop\n  M975 S1 ; turn on input shaping\n\n  G29.2 S0 ; avoid invalid abl data\n  G150.3\n{if ((filament_type[initial_no_support_filament_id] == \"PLA\") || (filament_type[initial_no_support_filament_id] == \"PLA-CF\") || (filament_type[initial_no_support_filament_id] == \"PETG\")) && (nozzle_diameter_at_nozzle_id[initial_nozzle_id] == 0.2)}\nM620.10 A0 F74.8347 H{nozzle_diameter_at_nozzle_id[initial_nozzle_id]} T{flush_temperatures[initial_no_support_filament_id]} P{nozzle_temperature_initial_layer[initial_no_support_filament_id]} S1\nM620.10 A1 F74.8347 H{nozzle_diameter_at_nozzle_id[initial_nozzle_id]} T{flush_temperatures[initial_no_support_filament_id]} P{nozzle_temperature_initial_layer[initial_no_support_filament_id]} S1\n{else}\nM620.10 A0 F{flush_volumetric_speeds[initial_no_support_filament_id]/2.4053*60} H{nozzle_diameter_at_nozzle_id[initial_nozzle_id]} T{flush_temperatures[initial_no_support_filament_id]} P{nozzle_temperature_initial_layer[initial_no_support_filament_id]} S1\nM620.10 A1 F{flush_volumetric_speeds[initial_no_support_filament_id]/2.4053*60} H{nozzle_diameter_at_nozzle_id[initial_nozzle_id]} T{flush_temperatures[initial_no_support_filament_id]} P{nozzle_temperature_initial_layer[initial_no_support_filament_id]} S1\n{endif}\n\n M620.11 P0 L0 I[initial_no_support_filament_id] E0\n M620.11 K0 I[initial_no_support_filament_id] R0\n\n  M620 S[initial_no_support_filament_id]A   ; switch material if AMS exist\n  M1002 gcode_claim_action : 4\n  M1002 set_filament_type:UNKNOWN\n  M400\n  T[initial_no_support_filament_id]\n  M400\n  M628 S0\n  M629\n  M400\n  M1002 set_filament_type:{filament_type[initial_no_support_filament_id]}\n  M621 S[initial_no_support_filament_id]A\n  M104 S{nozzle_temperature_initial_layer[initial_no_support_filament_id]}\n  M400\n  M106 P1 S0\n  M400\n  G29.2 S1\n;===== prepare print temperature and material ==========\n\n\n;===== auto extrude cali start =========================\n  M975 S1\n  M1002 judge_flag extrude_cali_flag\n  M622 J0\n    M983.3 F{filament_max_volumetric_speed[initial_no_support_filament_id]/2.4} A0.4 ; cali dynamic extrusion compensation\n  M623\n\n  M622 J1\n    M1002 set_filament_type:{filament_type[initial_no_support_filament_id]}\n    M1002 gcode_claim_action : 8\n    M109 S{nozzle_temperature[initial_no_support_filament_id]}\n    G90\n    M83\n    M983.3 F{filament_max_volumetric_speed[initial_no_support_filament_id]/2.4} A0.4 ; cali dynamic extrusion compensation\n    M400\n    M106 P1 S255\n    M400 S5\n    M106 P1 S0\n    G150.3\n  M623\n\n  M622 J2\n    M1002 set_filament_type:{filament_type[initial_no_support_filament_id]}\n    M1002 gcode_claim_action : 8\n    M109 S{nozzle_temperature[initial_no_support_filament_id]}\n    G90\n    M83\n    M983.3 F{filament_max_volumetric_speed[initial_no_support_filament_id]/2.4} A0.4 ; cali dynamic extrusion compensation\n    M400\n    M106 P1 S255\n    M400 S5\n    M106 P1 S0\n    G150.3\n  M623\n;===== auto extrude cali end =========================\n\n  {if hold_chamber_temp_for_flat_print}\n    M1002 gcode_claim_action : 58\n    M104 S{first_layer_temperature[initial_no_support_filament_id]}\n    {if bed_temperature_initial_layer_single > 89}\n        M1030 S1800\n        SYNC R0 T1800\n    {else}\n        M1030 S300\n        SYNC R0 T300\n    {endif}\n    M1030 C\n  {endif}\n\n  {if filament_type[initial_filament_id] == \"TPU\" || filament_type[initial_filament_id] == \"PVA\"}\n  {else}\n    M83\n    G1 E-3 F1800\n    M400 P500\n  {endif}\n  G150.2\n  G150.1 F8000\n  G150.2\n  G150.1 F8000\n\n  G91\n  G1 Y-16 F12000 ; move away from the trash bin\n  G90\n  M400\n\n  M104 S{nozzle_temperature_initial_layer[initial_no_support_filament_id]-80} A\n\n;===== wipe right nozzle start =====\n  M1002 gcode_claim_action : 14\n  G150 T{nozzle_temperature_initial_layer[initial_no_support_filament_id]}\n  M400\n\n{if filament_type[initial_filament_id] == \"PC\"}\n  M109 S170 A\n{else}\n  M109 S140 A\n{endif}\n  G91\n  G1 Z5 F1200\n  G90\n  M400\n  G150.1\n;===== wipe left nozzle end =====\n\n\n;===== mech mode sweep start =====\n  M1002 gcode_claim_action : 3\n  G90\n  G1 X128 Y128 F20000\n  G1 Z5 F1200\n  M400 P200\n  M970.3 Q1 A5 K0 O1\n  M970.2 Q1 K1 W74 Z0.01\n  M974 Q1 S2 P0\n  M970.3 Q0 A7 K0 O1\n  M970.2 Q0 K1 W74 Z0.01\n  M974 Q0 S2 P0\n  M975 S1\n  M400\n;===== mech mode sweep end =====\n\n;===== bed leveling ==================================\n  M1002 gcode_claim_action : 54\n  M190 S[bed_temperature_initial_layer_single]; ensure bed temp\n  M109 S140 A\n  M106 S0 ; turn off fan , too noisy\n  M1002 judge_flag g29_before_print_flag\n  M622 J1\n    M1002 gcode_claim_action : 1\n    {if hold_chamber_temp_for_flat_print}\n      G29 H\n    {else}\n      G29 A1 X{first_layer_print_min[0]} Y{first_layer_print_min[1]} I{first_layer_print_size[0]} J{first_layer_print_size[1]}\n    {endif}\n    M400\n  M623\n\n  M622 J2\n    M1002 gcode_claim_action : 1\n    {if hold_chamber_temp_for_flat_print}\n      G29 H\n    {else}\n      G29 A2 X{first_layer_print_min[0]} Y{first_layer_print_min[1]} I{first_layer_print_size[0]} J{first_layer_print_size[1]}\n    {endif}\n    M400\n  M623\n\n  M622 J0\n    G28\n  M623\n  G29.2 S1\n  G28\n;===== bed leveling end ================================\n\n  M985.1 U0 E2\n  M985.1 U1 E2\n\n  M104 S{nozzle_temperature_initial_layer[initial_filament_id]} A\n  G150.3 ; move to garbage can to wait for temp\n\n;===== wait temperature reaching the reference value =======\n  M190 S[bed_temperature_initial_layer_single]\n\n  ;========turn off light and fans =============\n  M960 S1 P0 ; turn off laser\n  M960 S2 P0 ; turn off laser\n  M106 S0 ; turn off cooling fan\n\n;===== wait temperature reaching the reference value =======\n\n  M1002 gcode_claim_action : 255\n  M400\n  M975 S1 ; turn on mech mode supression\n\n;============switch again==================\n  M211 X0 Y0 Z0 ;turn off soft endstop\n  G91\n  G1 Z6 F1200\n  G90\n  M1002 set_filament_type:{filament_type[initial_no_support_filament_id]}\n  M620 S[initial_no_support_filament_id]A\n  M400\n  T[initial_no_support_filament_id]\n  M400\n  M628 S0\n  M629\n  M400\n  M621 S[initial_no_support_filament_id]A\n;============switch again==================\n\n;===== for Textured PEI Plate , lower the nozzle as the nozzle was touching topmost of the texture when homing ==\n  {if bed_temperature_initial_layer_single > 89}\n    {if curr_bed_type==\"Textured PEI Plate\"}\n      G29.1 Z{-0.02} ; for Textured PEI Plate\n    {else}\n      G29.1 Z{0.0}\n    {endif}\n  {else}\n    {if curr_bed_type==\"Textured PEI Plate\"}\n      G29.1 Z{0.01} ; for Textured PEI Plate\n    {else}\n      G29.1 Z{0.03}\n    {endif}\n  {endif}\n\n\n;===== nozzle load line ===============================\nM1002 gcode_claim_action : 51\n  G29.2 S1 ; ensure z comp turn on\n  G90\n  M83\n  M400 P50\n  M500 D1\n  M400 S3\n  M109 S{nozzle_temperature_initial_layer[initial_no_support_filament_id]}\n  G0 X100 Y0 F24000\n  M400\n  ;G130 O0 X100 Y-0.4 Z0.8 F{filament_max_volumetric_speed[initial_no_support_filament_id]/2/2.4053} L40 E20 D5\n  G130 O0 X100 Y-0.2 Z0.6 F{filament_max_volumetric_speed[initial_no_support_filament_id]/2/2.4053} L40 E12 D4\n  G90\n  M83\n  G1 Z1\n  M400\n;===== noozle load line end ===========================\nM1002 gcode_claim_action : 0\n  G29.99\n\n{if (filament_type[initial_no_support_filament_id] == \"TPU\") ||\n(filament_type[initial_no_support_filament_id] == \"PLA\") ||  (filament_type[initial_no_support_filament_id] == \"PETG\")}\nM1015.3 S1 H{nozzle_diameter_at_nozzle_id[initial_nozzle_id]};enable tpu, pla and petg clog detect\n{else}\nM1015.3 S0;disable clog detect\n{endif}\n\n{if (filament_type[initial_no_support_filament_id] == \"PLA\") ||  (filament_type[initial_no_support_filament_id] == \"PETG\")\n ||  (filament_type[initial_no_support_filament_id] == \"PLA-CF\")  ||  (filament_type[initial_no_support_filament_id] == \"PETG-CF\")}\nM1015.4 S1 K1 H{nozzle_diameter_at_nozzle_id[initial_nozzle_id]} ;enable E air printing detect\n{else}\nM1015.4 S0 K0 H{nozzle_diameter_at_nozzle_id[initial_nozzle_id]} ;disable E air printing detect\n{endif}\n\nM620.6 I[initial_no_support_filament_id] W1 ;enable ams air printing detect\n\nM1010 Q0 B0.023 S0.01\nM1010 Q1 B0.005 S0.01\nM1010.1 S1\n
; machine_switch_extruder_time = 0
; machine_unload_filament_time = 31
; master_extruder_id = 1
; max_bridge_length = 0
; max_layer_height = 0.42
; max_travel_detour_distance = 0
; min_bead_width = 85%
; min_feature_size = 25%
; min_layer_height = 0.12
; minimum_sparse_infill_area = 15
; mmu_segmented_region_interlocking_depth = 0
; mmu_segmented_region_max_width = 0
; monotonic_travel_into_wall = 45%
; no_slow_down_for_cooling_on_outwalls = 0
; nozzle_diameter = 0.6
; nozzle_flush_dataset = 0
; nozzle_height = 4.2
; nozzle_temperature = 220
; nozzle_temperature_initial_layer = 220
; nozzle_temperature_range_high = 240
; nozzle_temperature_range_low = 190
; nozzle_type = hardened_steel
; nozzle_volume = 110
; nozzle_volume_type = Standard
; only_one_wall_first_layer = 0
; ooze_prevention = 0
; other_layers_print_sequence = 0
; other_layers_print_sequence_nums = 0
; outer_wall_acceleration = 5000
; outer_wall_jerk = 9
; outer_wall_line_width = 0.52
; outer_wall_speed = 60
; overhang_1_4_speed = 0
; overhang_2_4_speed = 50
; overhang_3_4_speed = 15
; overhang_4_4_speed = 10
; overhang_fan_speed = 100
; overhang_fan_threshold = 50%
; overhang_threshold_participating_cooling = 95%
; overhang_totally_speed = 10
; override_filament_scarf_seam_setting = 0
; override_process_overhang_speed = 0
; physical_extruder_map = 0
; post_process = 
; pre_start_fan_time = 0
; precise_outer_wall = 1
; precise_z_height = 0
; pressure_advance = 0.02
; prime_tower_brim_width = -1
; prime_tower_enable_framework = 0
; prime_tower_extra_rib_length = 0
; prime_tower_fillet_wall = 1
; prime_tower_flat_ironing = 1
; prime_tower_infill_gap = 150%
; prime_tower_lift_height = -1
; prime_tower_lift_speed = 90
; prime_tower_max_speed = 90
; prime_tower_rib_wall = 1
; prime_tower_rib_width = 8
; prime_tower_skip_points = 1
; prime_tower_width = 60
; prime_volume_mode = Default
; print_compatible_printers = "Bambu Lab P2S 0.6 nozzle"
; print_extruder_id = 1
; print_extruder_variant = "Direct Drive Standard"
; print_flow_ratio = 1
; print_in_clockwise = 1
; print_sequence = by layer
; print_settings_id = 0.18mm Balanced Quality @BBL P2S 0.6 nozzle
; printable_area = 0x0,256x0,256x256,0x256
; printable_height = 256
; printer_extruder_id = 1
; printer_extruder_variant = "Direct Drive Standard"
; printer_model = Bambu Lab P2S
; printer_notes = 
; printer_settings_id = Bambu Lab P2S 0.6 nozzle
; printer_structure = corexy
; printer_technology = FFF
; printer_variant = 0.6
; printing_by_object_gcode = 
; process_notes = 
; raft_contact_distance = 0.1
; raft_expansion = 1.5
; raft_first_layer_density = 90%
; raft_first_layer_expansion = -1
; raft_layers = 0
; reduce_crossing_wall = 0
; reduce_fan_stop_start_freq = 1
; reduce_infill_retraction_mode = Auto
; required_nozzle_HRC = 3
; resolution = 0.012
; retract_before_wipe = 0%
; retract_length_toolchange = 2
; retract_lift_above = 0
; retract_lift_below = 249
; retract_restart_extra = 0
; retract_restart_extra_toolchange = 0
; retract_when_changing_layer = 1
; retraction_distances_when_cut = 18
; retraction_distances_when_ec = 0
; retraction_length = 1.4
; retraction_minimum_travel = 1
; retraction_speed = 30
; role_base_wipe_speed = 1
; scan_first_layer = 0
; scarf_angle_threshold = 155
; seam_gap = 15%
; seam_placement_away_from_overhangs = 0
; seam_position = aligned
; seam_slope_conditional = 1
; seam_slope_entire_loop = 0
; seam_slope_gap = 0
; seam_slope_inner_walls = 1
; seam_slope_min_length = 10
; seam_slope_start_height = 10%
; seam_slope_steps = 10
; seam_slope_type = none
; silent_mode = 0
; single_extruder_multi_material = 1
; skeleton_infill_density = 15%
; skeleton_infill_line_width = 0.62
; skin_infill_density = 15%
; skin_infill_depth = 2
; skin_infill_line_width = 0.62
; skirt_distance = 2
; skirt_height = 1
; skirt_loops = 0
; skirt_per_object = 1
; slice_closing_radius = 0.049
; slicing_mode = regular
; slow_down_for_layer_cooling = 1
; slow_down_layer_time = 4
; slow_down_min_speed = 20
; slowdown_end_acc = 100000
; slowdown_end_height = 400
; slowdown_end_speed = 1000
; slowdown_start_acc = 100000
; slowdown_start_height = 0
; slowdown_start_speed = 1000
; small_perimeter_speed = 50%
; small_perimeter_threshold = 0
; smooth_coefficient = 4
; smooth_speed_discontinuity_area = 1
; solid_infill_filament = 0
; sparse_infill_acceleration = 100%
; sparse_infill_anchor = 400%
; sparse_infill_anchor_max = 20
; sparse_infill_density = 10%
; sparse_infill_filament = 0
; sparse_infill_lattice_angle_1 = -45
; sparse_infill_lattice_angle_2 = 45
; sparse_infill_line_width = 0.62
; sparse_infill_pattern = gyroid
; sparse_infill_speed = 270
; spiral_mode = 0
; spiral_mode_max_xy_smoothing = 200%
; spiral_mode_smooth = 0
; standby_temperature_delta = -5
; start_end_points = 30x-3,54x245
; supertack_plate_temp = 40
; supertack_plate_temp_initial_layer = 40
; support_air_filtration = 0
; support_angle = 0
; support_base_pattern = default
; support_base_pattern_spacing = 2.5
; support_bottom_interface_spacing = 0.5
; support_bottom_z_distance = 0.18
; support_chamber_temp_control = 0
; support_cooling_filter = 0
; support_critical_regions_only = 0
; support_expansion = 0
; support_fast_purge_mode = 0
; support_filament = 0
; support_interface_bottom_layers = 2
; support_interface_filament = 0
; support_interface_loop_pattern = 0
; support_interface_not_for_body = 1
; support_interface_pattern = auto
; support_interface_spacing = 0.5
; support_interface_speed = 80
; support_interface_top_layers = 2
; support_ironing_direction = 0
; support_ironing_flow = 10%
; support_ironing_inset = 0
; support_ironing_pattern = zig-zag
; support_ironing_spacing = 0.15
; support_ironing_speed = 30
; support_line_width = 0.62
; support_object_first_layer_gap = 0.2
; support_object_skip_flush = 1
; support_object_xy_distance = 0.35
; support_on_build_plate_only = 0
; support_remove_small_overhang = 0
; support_speed = 150
; support_style = default
; support_threshold_angle = 30
; support_top_z_distance = 0.18
; support_type = tree(auto)
; symmetric_infill_y_axis = 0
; temperature_vitrification = 45
; template_custom_gcode = 
; textured_plate_temp = 55
; textured_plate_temp_initial_layer = 55
; thick_bridges = 0
; thumbnail_size = 50x50
; time_lapse_gcode = ;======== P2S timelapes gcode ==========\n;===== 2025/06/16 ====\n; SKIPPABLE_START\n; SKIPTYPE: timelapse\nM622.1 S1 ; for prev firware, default turned on\n\nM1002 judge_flag timelapse_record_flag\nM622 J1\n{if timelapse_type == 0} ; timelapse without wipe tower\n  M971 S11 C10 O0\n  M1004 S5 P1  ; external shutter\n{elsif timelapse_type == 1} ; timelapse with wipe tower\n  G150.3 ; move to garbage can\n  M400\n  M1004 S5 P1  ; external shutter\n  M400 P300\n  M971 S11 C10 O0\n  M400 P350\n  \n  G90\n  G1 Z{max_layer_z + 3.0} F1200\n  G1 Y247 F30000\n  G1 Y217 F18000\n{endif}\nM623\n; SKIPPABLE_END\n
; timelapse_type = 0
; top_area_threshold = 200%
; top_color_penetration_layers = 3
; top_one_wall_type = all top
; top_shell_layers = 6
; top_shell_thickness = 0.8
; top_solid_infill_flow_ratio = 1
; top_surface_acceleration = 2000
; top_surface_density = 100%
; top_surface_jerk = 9
; top_surface_line_width = 0.62
; top_surface_pattern = monotonicline
; top_surface_speed = 150
; top_z_overrides_xy_distance = 0
; travel_acceleration = 10000
; travel_jerk = 9
; travel_short_distance_acceleration = 250
; travel_speed = 600
; travel_speed_z = 0
; tree_support_branch_angle = 45
; tree_support_branch_diameter = 2
; tree_support_branch_diameter_angle = 5
; tree_support_branch_distance = 5
; tree_support_wall_count = -1
; upward_compatible_machine = "Bambu Lab A1 0.6 nozzle";"Bambu Lab H2S 0.6 nozzle";"Bambu Lab H2D 0.6 nozzle";"Bambu Lab H2D Pro 0.6 nozzle";"Bambu Lab H2C 0.6 nozzle";"Bambu Lab X2D 0.6 nozzle";"Bambu Lab A2L 0.6 nozzle"
; use_firmware_retraction = 0
; use_relative_e_distances = 1
; vertical_shell_speed = 80%
; volumetric_speed_coefficients = "0 0 0 0 0 0"
; wall_distribution_count = 1
; wall_filament = 0
; wall_generator = arachne
; wall_loops = 4
; wall_sequence = inner wall/outer wall
; wall_transition_angle = 10
; wall_transition_filter_deviation = 25%
; wall_transition_length = 100%
; wipe = 1
; wipe_distance = 2
; wipe_speed = 80%
; wipe_tower_no_sparse_layers = 0
; wipe_tower_rotation_angle = 0
; wipe_tower_x = 15
; wipe_tower_y = 220
; wrapping_detection_gcode = ;======== P2S 20250822 clumping ========\n{if !spiral_mode}\n    M622.1 S0 ; for previous firmware, default turn off\n    M1002 set_flag g39_forced_detection_flag=1\n    M1002 judge_flag g39_forced_detection_flag\n    M622 J1\n        {if layer_num == 3 || layer_num == 10 || layer_num == 19}\n            M993 A2 B2 C2 ; nozzle cam detection allow status save.\n            M993 A0 B0 C0 ; nozzle cam detection not allowed.\n\n            M400 P100\n\n            G39\n\n            G90\n            G1 Y247 F30000\n            G1 Y217 F18000\n            \n            M993 A3 B3 C3 ; nozzle cam detection allow status restore.\n        {endif}\n    M623\n{endif}
; wrapping_detection_layers = 20
; wrapping_exclude_area = 153x256,216x256,216x235,153x235
; xy_contour_compensation = 0
; xy_hole_compensation = 0
; z_direction_outwall_speed_continuous = 0
; z_hop = 0.4
; z_hop_types = Auto Lift
; CONFIG_BLOCK_END

; EXECUTABLE_BLOCK_START
M73 P0 R19
M201 X20000 Y20000 Z500 E5000
M203 X600 Y600 Z20 E30
M204 P20000 R5000 T20000
M205 X9.00 Y9.00 Z3.00 E2.50
M106 S0
M106 P2 S0
; FEATURE: Custom
;M1002 set_flag extrude_cali_flag=1
;M1002 set_flag g29_before_print_flag=1
;M1002 set_flag auto_cali_toolhead_offset_flag=1
;M1002 set_flag build_plate_detect_flag=1

;======== P2S start gcode==========
;===== 2026/05/18 =====

  M140 S55 ; heat heatbed first
  M993 A0 B0 C0 ; nozzle cam detection not allowed.
  M400

;=====printer start sound ===================
M17
M400 S1
M1006 S1
M1006 A53 B9 L50 C53 D9 M50 E53 F9 N50
M1006 A56 B9 L50 C56 D9 M50 E56 F9 N50
M1006 A61 B9 L50 C61 D9 M50 E61 F9 N50
M1006 A53 B9 L50 C53 D9 M50 E53 F9 N50
M1006 A56 B9 L50 C56 D9 M50 E56 F9 N50
M1006 A61 B18 L50 C61 D18 M50 E61 F18 N50
M1006 W
;=====printer start sound ===================

  M620 M ;enable remap
  G389

;===== avoid end stop =================
  G91
  G380 S2 Z22 F1200
  G380 S2 Z-12 F1200
  G90
;===== avoid end stop =================

;===== reset machine status =================
  M204 S10000
  M630 S0 P1
  G90
  M17 D ; reset motor current to default
  M960 S5 P1 ; turn on logo lamp
  G90
  M220 S100 ;Reset Feedrate
  M1002 set_gcode_claim_speed_level: 5
  M221 S100 ;Reset Flowrate
  M73.2   R1.0 ;Reset left time magnitude
  G29.1 Z0 ; clear z-trim value first
  M983.1 M1
  M982.2 S1 ; turn on cog noise reduction
  M983.4 S0
;===== reset machine status =================

;==== set airduct mode ====
;==== if Chamber Cooling is necessary ====


M145 P0 ; set airduct mode to cooling mode for cooling
M106 P2 S255 ; turn on auxiliary fan for cooling
M106 P3 S127 ; turn on chamber fan for cooling
M1002 gcode_claim_action : 29
M191 S0 ; wait for chamber temp
M106 P2 S102 ; turn on chamber cooling fan
M622.1 S0
M1002 judge_flag ventobox_replace_aux1_fan_flag
M622 J0
M106 P10 S0 ; turn off left aux fan
M623
M142 P6 R30 S40 U0.3 V0.8 ; set PETG exhaust chamber autocooling


;==== set airduct mode ====

;===== start to heat heatbed & hotend==========
  M1002 gcode_claim_action : 2
  M1002 set_filament_type:PLA
  M104 S140 A

  G29.2 S0 ; avoid invalid abl data

;===== first homing start =====
  M1002 gcode_claim_action : 13
  G28 X T300
  G150.1 F8000 ; wipe mouth to avoid filament stick to heatbed
  G150.3
  M972 S24 P0
  M972 S26 P0 C0
  M972 S42 P0 T5000
  G150.1 F8000 ; wipe mouth to avoid filament stick to heatbed
  G90
  G1 X128 Y128 F30000
  G28 Z P0 T400
  M400
;===== first homign end =====

;===== detection start =====
  M1002 gcode_claim_action : 11
  M104 S140 A ; rise temp in advance
  M972 S19 P0 T5000 ;plate type detection

  
;===== detection end =====

;===== prepare print temperature and material ==========
  M400
  M211 X0 Y0 Z0 ;turn off soft endstop
  M975 S1 ; turn on input shaping

  G29.2 S0 ; avoid invalid abl data
  G150.3

M620.10 A0 F523.843 H0.6 T240 P220 S1
M620.10 A1 F523.843 H0.6 T240 P220 S1


 M620.11 P0 L0 I0 E0
 M620.11 K0 I0 R0

  M620 S0A   ; switch material if AMS exist
  M1002 gcode_claim_action : 4
  M1002 set_filament_type:UNKNOWN
  M400
  T0
  M400
  M628 S0
  M629
  M400
  M1002 set_filament_type:PLA
  M621 S0A
  M104 S220
  M400
  M106 P1 S0
  M400
  G29.2 S1
;===== prepare print temperature and material ==========


;===== auto extrude cali start =========================
  M975 S1
  M1002 judge_flag extrude_cali_flag
  M622 J0
    M983.3 F8.75 A0.4 ; cali dynamic extrusion compensation
  M623

  M622 J1
    M1002 set_filament_type:PLA
    M1002 gcode_claim_action : 8
    M109 S220
    G90
    M83
    M983.3 F8.75 A0.4 ; cali dynamic extrusion compensation
    M400
    M106 P1 S255
    M400 S5
    M106 P1 S0
    G150.3
  M623

  M622 J2
    M1002 set_filament_type:PLA
    M1002 gcode_claim_action : 8
    M109 S220
    G90
    M83
    M983.3 F8.75 A0.4 ; cali dynamic extrusion compensation
    M400
    M106 P1 S255
    M400 S5
    M106 P1 S0
    G150.3
  M623
;===== auto extrude cali end =========================

  

  
    M83
    G1 E-3 F1800
    M400 P500
  
  G150.2
  G150.1 F8000
  G150.2
  G150.1 F8000

  G91
  G1 Y-16 F12000 ; move away from the trash bin
  G90
  M400

  M104 S140 A

;===== wipe right nozzle start =====
  M1002 gcode_claim_action : 14
  G150 T220
  M400


  M109 S140 A

  G91
M73 P2 R19
  G1 Z5 F1200
  G90
  M400
  G150.1
;===== wipe left nozzle end =====


;===== mech mode sweep start =====
  M1002 gcode_claim_action : 3
  G90
M73 P3 R18
  G1 X128 Y128 F20000
  G1 Z5 F1200
  M400 P200
  M970.3 Q1 A5 K0 O1
  M970.2 Q1 K1 W74 Z0.01
  M974 Q1 S2 P0
  M970.3 Q0 A7 K0 O1
  M970.2 Q0 K1 W74 Z0.01
  M974 Q0 S2 P0
  M975 S1
  M400
;===== mech mode sweep end =====

;===== bed leveling ==================================
  M1002 gcode_claim_action : 54
  M190 S55; ensure bed temp
  M109 S140 A
  M106 S0 ; turn off fan , too noisy
  M1002 judge_flag g29_before_print_flag
  M622 J1
    M1002 gcode_claim_action : 1
    
      G29 A1 X111.577 Y98.6759 I32.8504 J58.6363
    
    M400
  M623

  M622 J2
    M1002 gcode_claim_action : 1
    
      G29 A2 X111.577 Y98.6759 I32.8504 J58.6363
    
    M400
  M623

  M622 J0
    G28
  M623
  G29.2 S1
  G28
;===== bed leveling end ================================

  M985.1 U0 E2
  M985.1 U1 E2

  M104 S220 A
  G150.3 ; move to garbage can to wait for temp

;===== wait temperature reaching the reference value =======
  M190 S55

  ;========turn off light and fans =============
  M960 S1 P0 ; turn off laser
  M960 S2 P0 ; turn off laser
  M106 S0 ; turn off cooling fan

;===== wait temperature reaching the reference value =======

  M1002 gcode_claim_action : 255
  M400
  M975 S1 ; turn on mech mode supression

;============switch again==================
  M211 X0 Y0 Z0 ;turn off soft endstop
  G91
  G1 Z6 F1200
  G90
  M1002 set_filament_type:PLA
  M620 S0A
  M400
  T0
  M400
  M628 S0
  M629
  M400
  M621 S0A
;============switch again==================

;===== for Textured PEI Plate , lower the nozzle as the nozzle was touching topmost of the texture when homing ==
  
    
      G29.1 Z0.01 ; for Textured PEI Plate
    
  


;===== nozzle load line ===============================
M1002 gcode_claim_action : 51
  G29.2 S1 ; ensure z comp turn on
  G90
  M83
  M400 P50
  M500 D1
  M400 S3
  M109 S220
  G0 X100 Y0 F24000
  M400
  ;G130 O0 X100 Y-0.4 Z0.8 F4.36536 L40 E20 D5
  G130 O0 X100 Y-0.2 Z0.6 F4.36536 L40 E12 D4
  G90
  M83
  G1 Z1
  M400
;===== noozle load line end ===========================
M1002 gcode_claim_action : 0
  G29.99


M1015.3 S1 H0.6;enable tpu, pla and petg clog detect



M1015.4 S1 K1 H0.6 ;enable E air printing detect


M620.6 I0 W1 ;enable ams air printing detect

M1010 Q0 B0.023 S0.01
M1010 Q1 B0.005 S0.01
M1010.1 S1
; MACHINE_START_GCODE_END
; filament start gcode
;VT0 H-1
G90
G21
M83 ; use relative distances for extrusion
M981 S1 P20000 ;open spaghetti detector
; CHANGE_LAYER
; Z_HEIGHT: 0.2
; LAYER_HEIGHT: 0.2
G1 E-.4 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====









    
        M106 P2 S102
    

    M622.1 S0
    M1002 judge_flag ventobox_replace_aux1_fan_flag
    M622 J0
    M106 P10 S102; set first x_layer left aux fan
    M623
;not reset fan


; update layer progress
M73 L1
M991 S0 P0 ;notify layer change


M106 S0
M204 S6000
M73 P35 R12
G1 Z.4 F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.299 Y104.287
G1 Z.2
M73 P36 R12
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.61999
G1 F2100
M204 S500
G3 X123.623 Y117.923 I-51.322 J-19.914 E.73832
G1 X120.675 Y121.44 E.2158
G3 X115.219 Y126.653 I-40.86 J-37.305 E.35512
G1 X118.153 Y128.83 E.17183
G3 X120.994 Y134.509 I-4.392 J5.747 E.30913
G3 X120.826 Y136.147 I-13.424 J-.552 E.07749
G3 X130.645 Y141.919 I-22.187 J48.982 E.53659
G3 X142.215 Y153.374 I-32.793 J44.69 E.76841
G1 X142.215 Y104.857 E2.2815
G1 X141.839 Y104.94 E.0181
G1 X132.751 Y104.94 E.42733
G1 X132.525 Y104.927 E.01067
G1 X131.948 Y104.773 E.0281
G1 X131.453 Y104.467 E.02735
G1 X131.358 Y104.356 E.0069
; WIPE_START
G1 X130.942 Y105.266 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S6000
G1 X131.356 Y103.036 Z.6 F36000
G1 Z.2
G1 E.4 F1800
; LINE_WIDTH: 0.65787
G1 F2100
M204 S500
G1 X131.296 Y102.786 E.01288
; LINE_WIDTH: 0.66526
G1 X131.304 Y102.69 E.00491
G1 X130.915 Y103.687 E.05428
; LINE_WIDTH: 0.61999
G3 X123.18 Y117.552 I-50.742 J-19.214 E.74928
G1 X120.232 Y121.069 E.2158
G3 X114.314 Y126.651 I-40.649 J-37.171 E.38288
G1 X114.294 Y126.685 E.00189
G1 X117.809 Y129.293 E.20582
G3 X119.132 Y130.641 I-4.79 J6.022 E.08901
G1 X119.648 Y131.452 E.04521
G3 X120.386 Y135.203 I-5.854 J3.1 E.18234
G3 X120.135 Y136.467 I-11.649 J-1.651 E.06062
G3 X130.304 Y142.384 I-21.65 J48.897 E.55436
G3 X142.792 Y155.162 I-32.391 J44.148 E.84393
G1 X142.792 Y103.916 E2.40976
G1 X142.451 Y104.227 E.0217
G1 X141.839 Y104.363 E.02946
G1 X132.751 Y104.363 E.42733
G1 X132.177 Y104.244 E.02758
G1 X131.824 Y104.025 E.01953
G1 X131.5 Y103.638 E.02376
; LINE_WIDTH: 0.65787
G1 X131.377 Y103.124 E.02648
M204 S6000
G1 X131.913 Y102.948 F36000
; LINE_WIDTH: 0.61999
G1 F2100
M204 S500
G1 X131.89 Y102.841 E.00514
G1 X131.984 Y102.311 E.02534
; LINE_WIDTH: 0.665876
G1 X131.983 Y102.189 E.00619
; LINE_WIDTH: 0.711762
G1 X131.983 Y102.067 E.00664
; LINE_WIDTH: 0.757648
G1 X131.982 Y101.945 E.0071
; LINE_WIDTH: 0.803534
G1 X131.981 Y101.823 E.00755
; LINE_WIDTH: 0.84942
G3 X132.017 Y101.535 I.646 J-.065 E.01929
; LINE_WIDTH: 0.83693
G1 X132.155 Y100.9 E.042
; LINE_WIDTH: 0.796655
G1 X132.293 Y100.266 E.03987
; LINE_WIDTH: 0.75638
G1 X132.371 Y99.848 E.02472
; LINE_WIDTH: 0.75531
G1 X131.684 Y99.813 E.03994
; LINE_WIDTH: 0.75638
G1 X131.603 Y100.093 E.01696
; LINE_WIDTH: 0.796655
G1 X131.426 Y100.718 E.03987
; LINE_WIDTH: 0.83693
G1 X131.249 Y101.342 E.042
; LINE_WIDTH: 0.84942
G3 X131.142 Y101.602 I-.609 J-.098 E.01865
; LINE_WIDTH: 0.803534
G1 X131.081 Y101.708 E.00755
; LINE_WIDTH: 0.757648
G1 X131.02 Y101.814 E.0071
; LINE_WIDTH: 0.711762
G1 X130.96 Y101.919 E.00664
; LINE_WIDTH: 0.665876
G1 X130.899 Y102.025 E.00619
; LINE_WIDTH: 0.61999
G3 X122.738 Y117.182 I-50.787 J-17.571 E.81297
G1 X119.79 Y120.698 E.2158
G3 X113.335 Y126.692 I-41.103 J-37.793 E.41467
G1 X117.466 Y129.757 E.24186
G3 X118.664 Y130.979 I-4.407 J5.519 E.08068
G1 X119.149 Y131.742 E.04254
G3 X119.406 Y136.783 I-5.375 J2.801 E.24481
G3 X130.728 Y143.429 I-21.167 J49.034 E.61891
G3 X142.606 Y155.918 I-32.113 J42.433 E.81406
G1 X142.759 Y156.144 E.01286
G1 X143.369 Y156.031 E.02917
G1 X143.369 Y104.444 E2.42582
; LINE_WIDTH: 0.63311
G1 X143.362 Y103.572 E.04193
; LINE_WIDTH: 0.66358
M73 P37 R12
G1 X143.347 Y102.92 E.03298
; LINE_WIDTH: 0.67165
G3 X143.356 Y101.522 I23.877 J-.548 E.07166
; LINE_WIDTH: 0.64582
G1 X143.369 Y100.343 E.05791
; LINE_WIDTH: 0.61999
G1 X143.369 Y99.76 E.02742
G3 X142.139 Y99.785 I-.892 J-13.595 E.05787
G1 X142.261 Y100.445 E.03155
; LINE_WIDTH: 0.64582
G1 X142.488 Y101.602 E.05791
; LINE_WIDTH: 0.67165
G1 X142.716 Y102.758 E.06039
; LINE_WIDTH: 0.67265
G1 X142.718 Y103.075 E.01623
G1 X142.591 Y103.362 E.01614
; LINE_WIDTH: 0.63311
G1 X142.206 Y103.704 E.02476
; LINE_WIDTH: 0.61999
G1 X141.839 Y103.786 E.01768
G1 X132.751 Y103.786 E.42733
G1 X132.39 Y103.707 E.0174
G1 X132.195 Y103.583 E.01085
G1 X132 Y103.351 E.01426
G1 X131.932 Y103.036 E.01516
; WIPE_START
G1 X131.89 Y102.841 E-.0757
G1 X131.984 Y102.311 E-.20474
G1 X131.983 Y102.189 E-.0463
G1 X131.983 Y102.067 E-.0463
G1 X131.982 Y102.049 E-.00696
; WIPE_END
G1 E-.02 F1800
M204 S6000
G1 X139.516 Y100.825 Z.6 F36000
G1 X142.844 Y100.285 Z.6
G1 Z.2
G1 E.4 F1800
; LINE_WIDTH: 0.58294
G1 F2100
M204 S500
G2 X142.878 Y100.343 I-.03 J.056 E.01438
; WIPE_START
G1 X142.844 Y100.402 E-.076
G1 X142.777 Y100.402 E-.076
G1 X142.743 Y100.343 E-.076
G1 X142.777 Y100.285 E-.076
G1 X142.844 Y100.285 E-.076
; WIPE_END
G1 E-.02 F1800
M204 S6000
G1 X135.439 Y102.135 Z.6 F36000
G1 X132.489 Y102.872 Z.6
G1 Z.2
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.61999
G1 F2100
M204 S500
G1 X133.171 Y99.186 E.17625
G3 X131.188 Y99.137 I1.773 J-111.239 E.09329
G3 X122.279 Y116.797 I-51.225 J-14.763 E.93554
G1 X119.327 Y120.319 E.2161
G3 X112.343 Y126.702 I-40.562 J-37.366 E.44547
G1 X117.109 Y130.237 E.27903
G1 X117.524 Y130.596 E.0258
G3 X118.592 Y137.09 I-3.647 J3.935 E.33316
G3 X130.371 Y143.909 I-19.868 J47.899 E.64185
G3 X142.482 Y156.805 I-31.74 J41.945 E.83589
G1 X143.967 Y156.529 E.07102
G1 X143.967 Y99.203 E2.69567
G1 X143.967 Y99.136 E.00316
G3 X141.42 Y99.187 I-2.204 J-46.297 E.11985
G1 X142.102 Y102.874 E.17633
G3 X141.839 Y103.187 I-.293 J.021 E.02128
G1 X132.751 Y103.187 E.42733
G1 X132.58 Y103.125 E.00859
G1 X132.52 Y103.053 E.0044
G1 X132.504 Y102.961 E.00442
; WIPE_START
G1 X132.678 Y101.976 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S6000
G1 X131.431 Y102.208 Z.6 F36000
G1 Z.2
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.62804
G1 F2100
M204 S500
G1 X131.368 Y102.449 E.01187
; LINE_WIDTH: 0.66526
G1 X131.304 Y102.69 E.01263
; WIPE_START
G1 X131.368 Y102.449 E-.19
G1 X131.431 Y102.208 E-.19
; WIPE_END
G1 E-.02 F1800
M204 S6000
G1 X138.709 Y104.508 Z.6 F36000
G1 X140.807 Y105.171 Z.6
G1 Z.2
G1 E.4 F1800
; FEATURE: Bottom surface
; LINE_WIDTH: 0.62482
G1 F3300
M204 S500
G1 X141.724 Y106.088 E.0615
G1 X141.724 Y106.911 E.03902
G1 X140.244 Y105.431 E.09927
G1 X139.421 Y105.431 E.03902
G1 X141.724 Y107.734 E.15445
G1 X141.724 Y108.557 E.03902
G1 X138.598 Y105.431 E.20964
G1 X137.775 Y105.431 E.03902
G1 X141.724 Y109.38 E.26482
G1 X141.724 Y110.203 E.03902
G1 X136.952 Y105.431 E.32001
G1 X136.129 Y105.431 E.03902
G1 X141.724 Y111.026 E.37519
G1 X141.724 Y111.849 E.03902
G1 X135.306 Y105.431 E.43038
G1 X134.483 Y105.431 E.03902
G1 X141.724 Y112.672 E.48556
G1 X141.724 Y113.494 E.03902
G1 X133.66 Y105.431 E.54075
G1 X132.837 Y105.431 E.03902
G1 X141.724 Y114.317 E.59593
G1 X141.724 Y115.14 E.03902
G1 X131.821 Y105.237 E.66412
G3 X131.51 Y105.081 I.102 J-.592 E.0167
G1 X131.316 Y105.555 E.02428
G1 X141.724 Y115.963 E.69797
G1 X141.724 Y116.786 E.03902
G1 X131.077 Y106.139 E.71401
M73 P38 R12
G3 X130.824 Y106.709 I-7.735 J-3.076 E.0296
G1 X141.724 Y117.609 E.73092
G1 X141.724 Y118.432 E.03902
G1 X130.567 Y107.275 E.74817
G1 X130.31 Y107.841 E.02947
G1 X141.724 Y119.255 E.76542
G1 X141.724 Y120.078 E.03902
G1 X130.053 Y108.407 E.78266
G3 X129.782 Y108.958 I-9.192 J-4.173 E.02916
G1 X141.724 Y120.901 E.80084
G1 X141.724 Y121.724 E.03902
G1 X129.507 Y109.507 E.81925
G1 X129.222 Y110.044 E.02886
G1 X141.724 Y122.547 E.83839
G1 X141.724 Y123.37 E.03902
G1 X128.936 Y110.582 E.85753
G1 X128.651 Y111.119 E.02886
G1 X141.724 Y124.193 E.87667
G1 X141.724 Y125.015 E.03902
G1 X128.36 Y111.651 E.8962
G1 X128.059 Y112.173 E.02858
M73 P38 R11
G1 X141.724 Y125.838 E.91635
G1 X141.724 Y126.661 E.03902
G1 X127.758 Y112.695 E.93656
G1 X127.443 Y113.203 E.02834
G1 X141.724 Y127.484 E.95769
G1 X141.724 Y128.307 E.03902
G1 X127.127 Y113.71 E.97883
G1 X126.812 Y114.218 E.02834
G1 X141.724 Y129.13 E.99997
G1 X141.724 Y129.953 E.03902
G1 X126.495 Y114.724 E1.02125
G1 X126.166 Y115.218 E.02815
G1 X141.724 Y130.776 E1.04328
G1 X141.724 Y131.599 E.03902
G1 X125.835 Y115.709 E1.06552
G1 X125.497 Y116.195 E.02803
G1 X141.724 Y132.422 E1.08816
G1 X141.724 Y133.245 E.03902
G1 X125.158 Y116.679 E1.11089
G1 X124.812 Y117.155 E.02793
G1 X141.724 Y134.068 E1.13414
G1 X141.724 Y134.891 E.03902
G1 X124.462 Y117.628 E1.15757
G1 X124.107 Y118.096 E.02785
G1 X141.724 Y135.714 E1.1814
G1 X141.724 Y136.537 E.03902
G1 X123.737 Y118.55 E1.20617
G1 X123.362 Y118.997 E.0277
M73 P39 R11
G1 X141.724 Y137.359 E1.23133
G1 X141.724 Y138.182 E.03902
G1 X122.987 Y119.445 E1.2565
G1 X122.612 Y119.893 E.0277
G1 X141.724 Y139.005 E1.28166
G1 X141.724 Y139.828 E.03902
G1 X122.236 Y120.34 E1.30683
G1 X121.861 Y120.788 E.0277
G1 X141.724 Y140.651 E1.33199
G1 X141.724 Y141.474 E.03902
G1 X121.486 Y121.236 E1.35716
G1 X121.111 Y121.683 E.0277
G1 X141.724 Y142.297 E1.38232
G1 X141.724 Y143.12 E.03902
G1 X120.719 Y122.115 E1.40856
G3 X120.321 Y122.539 I-7.587 J-6.731 E.02761
G1 X141.724 Y143.943 E1.4353
G1 X141.724 Y144.766 E.03902
G1 X119.914 Y122.956 E1.46255
G1 X119.508 Y123.372 E.02759
G1 X141.724 Y145.589 E1.4898
G1 X141.724 Y146.412 E.03902
G1 X119.101 Y123.789 E1.51706
G3 X118.684 Y124.195 I-6.819 J-6.594 E.0276
G1 X141.724 Y147.235 E1.54503
G1 X141.724 Y148.058 E.03902
G1 X118.265 Y124.598 E1.57313
G3 X117.841 Y124.998 I-9.439 J-9.597 E.02761
G1 X141.724 Y148.88 E1.60155
G1 X141.724 Y149.703 E.03902
G1 X117.413 Y125.393 E1.63024
G3 X116.978 Y125.78 I-7.242 J-7.717 E.02764
G1 X141.724 Y150.526 E1.65947
G1 X141.724 Y151.349 E.03902
G1 X116.538 Y126.163 E1.68893
G1 X116.099 Y126.547 E.02766
M73 P40 R11
G1 X116.673 Y127.121 E.03849
G1 X118.449 Y128.439 E.10488
G3 X119.738 Y129.676 I-5.873 J7.406 E.08481
G1 X120.31 Y130.468 E.04635
G1 X120.682 Y131.13 E.03602
G1 X131.518 Y141.966 E.72662
G2 X128.665 Y139.936 I-34.069 J44.863 E.16603
G1 X121.194 Y132.465 E.50102
G3 X121.409 Y133.503 I-5.05 J1.587 E.05034
G1 X126.436 Y138.53 E.33712
G2 X124.512 Y137.429 I-18.84 J30.688 E.10514
G1 X121.48 Y134.397 E.20332
G1 X121.455 Y135.195 E.03786
G1 X123.409 Y137.149 E.13102
; CHANGE_LAYER
; Z_HEIGHT: 0.36
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F3300
G1 X122.702 Y136.441 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





    ;updata chamber autocooling in Xth layer
    
        
            M142 P1 R30 S40 U0.7 V1.0 O45; set PLA/TPU ND0.4 chamber autocooling
        
    

    M622.1 S0
    M1002 judge_flag ventobox_replace_aux1_fan_flag
    M622 J0
    M106 P10 S178.5; set left aux fan
    M623
;not reset fan


; update layer progress
M73 L2
M991 S0 P1 ;notify layer change


M106 S255
; open powerlost recovery
M1003 S1
M204 S10000
G17
G3 Z.6 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.268 Y103.735
G1 Z.36
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.199 Y115.369 I-51.411 J-19.421 E.50217
G3 X123.072 Y118.224 I-28.999 J-19.382 E.13602
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-40.381 J-36.837 E.29804
G1 X118.015 Y129.012 E.15054
G3 X118.932 Y129.849 I-4.61 J5.973 E.04747
G3 X120.075 Y131.53 I-5.869 J5.22 E.07784
G1 X120.45 Y132.48 E.03898
G3 X120.765 Y134.489 I-7.431 J2.192 E.07789
G1 X120.698 Y135.504 E.03881
G1 X120.538 Y136.268 E.02981
G3 X131.285 Y142.689 I-21.795 J48.68 E.47907
G3 X142.443 Y154.07 I-32.931 J43.448 E.61069
G1 X142.443 Y104.511 E1.89211
G1 X142.253 Y104.619 E.00836
G1 X141.651 Y104.714 E.02324
G1 X132.939 Y104.714 E.33262
G1 X132.652 Y104.693 E.01098
G1 X132.129 Y104.538 E.02085
G1 X131.693 Y104.264 E.01966
G1 X131.311 Y103.836 E.02189
G1 X131.303 Y103.818 E.00075
G1 X131.633 Y103.175 F36000
; LINE_WIDTH: 0.748456
G1 F10374.599
G1 X131.615 Y103.139 E.00185
; LINE_WIDTH: 0.791276
G1 F10249.577
G1 X131.559 Y103.022 E.0064
; LINE_WIDTH: 0.834096
G1 F9846.705
G1 X131.503 Y102.905 E.00676
; LINE_WIDTH: 0.876916
G1 F9346.294
G1 X131.447 Y102.788 E.00713
; LINE_WIDTH: 0.923496
G1 F8856.674
G1 X131.428 Y102.733 E.0034
; LINE_WIDTH: 0.970076
G1 F8415.8
G1 X131.408 Y102.677 E.00358
; LINE_WIDTH: 1.01666
G1 F8016.736
G1 X131.389 Y102.622 E.00376
; LINE_WIDTH: 1.06324
G1 F7653.805
G1 X131.37 Y102.566 E.00394
; LINE_WIDTH: 1.10982
G1 F7322.312
G1 X131.35 Y102.511 E.00411
G1 X131.31 Y102.556 E.00425
; LINE_WIDTH: 1.06324
G1 F7653.805
G1 X131.27 Y102.602 E.00406
; LINE_WIDTH: 1.01666
G1 F8016.736
G1 X131.229 Y102.647 E.00388
; LINE_WIDTH: 0.970076
G1 F8415.8
G1 X131.189 Y102.692 E.0037
; LINE_WIDTH: 0.923496
G1 F8856.674
G1 X131.149 Y102.737 E.00351
; LINE_WIDTH: 0.876916
G1 F9346.294
G1 X131.073 Y102.882 E.00896
; LINE_WIDTH: 0.834096
G1 F9846.705
G1 X130.997 Y103.026 E.0085
; LINE_WIDTH: 0.791276
G1 F10403.731
G1 X130.921 Y103.17 E.00805
; LINE_WIDTH: 0.748456
G1 F10925.576
G1 X130.845 Y103.315 E.00759
; LINE_WIDTH: 0.705636
G1 F11460.174
G1 X130.769 Y103.459 E.00714
; LINE_WIDTH: 0.662816
G1 F12007.541
G1 X130.693 Y103.603 E.00668
; LINE_WIDTH: 0.619996
G1 F13404.012
G1 X130.547 Y103.975 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.211 Y104.827 E.01969
G3 X124.712 Y115.043 I-50.391 J-20.536 E.44378
G3 X122.623 Y117.848 I-28.401 J-18.974 E.1336
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-41.712 J-38.365 E.32254
G1 X117.666 Y129.482 E.17935
G3 X118.523 Y130.268 I-4.175 J5.414 E.04446
G3 X119.208 Y131.169 I-7.776 J6.624 E.04324
G1 X119.562 Y131.812 E.02803
G1 X119.901 Y132.683 E.03569
G3 X120.136 Y133.793 I-9.548 J2.596 E.04331
G1 X120.18 Y134.525 E.028
G1 X120.115 Y135.453 E.03553
G1 X119.931 Y136.289 E.03267
G1 X119.836 Y136.599 E.01238
G3 X130.932 Y143.156 I-21.042 J48.274 E.49332
G3 X142.92 Y155.769 I-32.359 J42.763 E.66731
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.009 E1.97536
; LINE_WIDTH: 0.639246
G1 F13018.467
G1 X143.019 Y103.261 E.02951
; LINE_WIDTH: 0.653996
G1 F12708.583
G1 X143.012 Y103.003 E.01042
G1 X142.938 Y103.246 E.01027
; LINE_WIDTH: 0.639246
G1 F13018.467
G1 X142.549 Y103.791 E.02641
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.072 Y104.062 E.02094
G1 X141.651 Y104.128 E.01626
G1 X132.939 Y104.128 E.33262
G1 X132.739 Y104.113 E.00768
G1 X132.346 Y103.992 E.01569
G1 X132.067 Y103.813 E.01265
G1 X131.784 Y103.49 E.01642
; LINE_WIDTH: 0.662816
G1 F12170.303
G1 X131.728 Y103.373 E.00531
; LINE_WIDTH: 0.705636
G1 F11730.97
G1 X131.672 Y103.256 E.00568
G1 X132.187 Y102.852 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.161 Y102.76 E.00366
G1 X132.174 Y102.622 E.00527
G1 X132.745 Y99.54 E.1197
G3 X131.452 Y99.491 I1.001 J-43.416 E.0494
G3 X124.226 Y114.718 I-51.937 J-15.32 E.64612
G3 X122.174 Y117.472 I-27.806 J-18.568 E.13118
G1 X119.605 Y120.537 E.15272
G3 X112.93 Y126.698 I-40.356 J-37.03 E.34719
G1 X117.317 Y129.952 E.20855
G1 X118.114 Y130.687 E.04139
G3 X118.695 Y131.451 I-9.694 J7.974 E.03667
G1 X119.049 Y132.095 E.02803
G1 X119.352 Y132.887 E.0324
G3 X119.551 Y133.828 I-11.628 J2.947 E.03673
G1 X119.596 Y134.56 E.028
G1 X119.531 Y135.403 E.03225
G1 X119.361 Y136.154 E.02939
G1 X119.08 Y136.91 E.03082
G3 X130.578 Y143.623 I-20.72 J48.693 E.50968
G3 X142.647 Y156.415 I-31.955 J42.239 E.67457
G1 X143.615 Y156.235 E.03756
G1 X143.615 Y99.504 E2.16597
G3 X141.847 Y99.542 I-1.29 J-18.687 E.06754
G1 X142.418 Y102.63 E.11989
G3 X142.381 Y103.037 I-.767 J.135 E.01581
G1 X142.164 Y103.35 E.01454
G1 X141.892 Y103.505 E.01195
G1 X141.651 Y103.543 E.00928
G1 X132.939 Y103.543 E.33262
G1 X132.601 Y103.465 E.01327
G3 X132.28 Y103.178 I.339 J-.701 E.01665
G1 X132.212 Y102.939 E.00951
M204 S250
G1 X132.717 Y102.723 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X133.409 Y98.989 E.12023
G1 X132.441 Y98.986 E.03065
G2 X131.038 Y98.937 I-1.215 J14.55 E.04447
G3 X123.763 Y114.414 I-51.143 J-14.589 E.54381
G3 X121.747 Y117.121 I-27.157 J-18.125 E.10691
G1 X119.177 Y120.186 E.12664
G3 X112.01 Y126.704 I-40.297 J-37.117 E.30711
G1 X116.988 Y130.396 E.19622
G1 X117.728 Y131.083 E.03196
G3 X118.211 Y131.718 I-22.56 J17.654 E.02527
G1 X118.565 Y132.361 E.02324
G1 X118.834 Y133.08 E.02429
G1 X118.954 Y133.638 E.01807
G1 X119.044 Y134.594 E.03041
G1 X118.98 Y135.355 E.02418
G1 X118.824 Y136.026 E.02181
G1 X118.547 Y136.754 E.02464
G1 X118.31 Y137.19 E.01573
G3 X130.248 Y144.067 I-19.583 J47.796 E.4375
G3 X142.39 Y157.025 I-31.645 J41.818 E.56494
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.232 E1.81928
G1 X144.167 Y98.937 E.00935
G1 X143.859 Y98.938 E.00977
G3 X141.182 Y98.99 I-1.878 J-28.059 E.08479
G1 X141.874 Y102.725 E.12028
G1 X141.863 Y102.843 E.00376
G1 X141.721 Y102.979 E.00621
G1 X141.651 Y102.99 E.00223
G1 X132.939 Y102.99 E.27583
G1 X132.795 Y102.938 E.00486
G1 X132.735 Y102.808 E.00452
; WIPE_START
M204 S10000
G1 X132.909 Y101.823 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X140.49 Y102.708 Z.76 F36000
G1 X143.012 Y103.003 Z.76
G1 Z.36
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.656356
G1 F12660.365
G1 X143.011 Y102.512 E.01988
; LINE_WIDTH: 0.688786
G1 F12033.005
G1 X142.994 Y102.336 E.00757
; LINE_WIDTH: 0.7345
G1 F11247.37
G1 X142.972 Y102.087 E.01141
; LINE_WIDTH: 0.780214
G1 F10558.034
G1 X142.949 Y101.838 E.01216
; LINE_WIDTH: 0.825927
G1 F9948.317
G1 X142.926 Y101.589 E.0129
; LINE_WIDTH: 0.871641
G1 F9405.176
G1 X142.903 Y101.34 E.01365
; LINE_WIDTH: 0.917355
G1 F8918.271
G1 X142.88 Y101.091 E.01439
; LINE_WIDTH: 0.963069
G1 F8479.299
G1 X142.857 Y100.842 E.01514
; LINE_WIDTH: 1.00878
G1 F8081.514
G1 X142.834 Y100.593 E.01588
; LINE_WIDTH: 1.0545
G1 F7719.378
G1 X142.812 Y100.344 E.01663
; WIPE_START
G1 X142.834 Y100.593 E-.095
G1 X142.857 Y100.842 E-.095
G1 X142.88 Y101.091 E-.095
G1 X142.903 Y101.34 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.31 Y102.11 Z.76 F36000
G1 X131.35 Y102.511 Z.76
G1 Z.36
G1 E.4 F1800
; LINE_WIDTH: 1.10982
G1 F7322.312
G1 X131.36 Y102.472 E.00284
; LINE_WIDTH: 1.10326
G1 F7367.249
G1 X131.446 Y102.147 E.02343
; LINE_WIDTH: 1.05337
G1 F7727.912
G1 X131.532 Y101.822 E.02233
; LINE_WIDTH: 1.00348
G1 F8125.705
G1 X131.617 Y101.496 E.02124
; LINE_WIDTH: 0.953596
G1 F8566.674
G1 X131.626 Y101.46 E.00226
; LINE_WIDTH: 0.948456
G1 F8614.843
G1 X131.705 Y101.145 E.01932
; LINE_WIDTH: 0.908281
G1 F9010.865
G1 X131.784 Y100.831 E.01847
; LINE_WIDTH: 0.868106
G1 F9445.051
G1 X131.862 Y100.516 E.01762
; LINE_WIDTH: 0.827931
G1 F9923.198
G1 X131.941 Y100.202 E.01677
; LINE_WIDTH: 0.787756
G1 F10452.338
G1 X131.946 Y100.183 E.00096
; WIPE_START
G1 X131.941 Y100.202 E-.00744
G1 X131.862 Y100.516 E-.1232
G1 X131.784 Y100.831 E-.1232
G1 X131.705 Y101.145 E-.1232
G1 X131.703 Y101.153 E-.00295
; WIPE_END
G1 E-.02 F1800
G1 X132.043 Y104.754 Z.76 F36000
G1 Z.36
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.627406
G1 F13278.366
G1 X130.96 Y105.837 E.05925
G3 X130.294 Y107.342 I-18.891 J-7.461 E.06363
G1 X132.478 Y105.157 E.11944
G2 X133.263 Y105.212 I.612 J-3.141 E.03047
G1 X129.549 Y108.925 E.20303
G3 X128.639 Y110.674 I-28.421 J-13.683 E.07626
G1 X134.101 Y105.212 E.29868
G1 X134.94 Y105.212 E.03243
G1 X127.564 Y112.588 E.40328
G1 X126.151 Y114.839 E.10278
G1 X135.779 Y105.212 E.5264
G1 X136.618 Y105.212 E.03243
G1 X124.13 Y117.699 E.68275
G1 X123.83 Y118.095 E.01922
G1 X120.881 Y121.614 E.1775
G3 X115.652 Y126.639 I-40.978 J-37.401 E.28058
G1 X115.868 Y126.8 E.01042
G1 X137.456 Y105.212 E1.18035
G1 X138.295 Y105.212 E.03243
G1 X116.35 Y127.157 E1.19988
G1 X116.832 Y127.514 E.02318
G1 X139.134 Y105.212 E1.21941
G1 X139.972 Y105.212 E.03243
G1 X117.313 Y127.871 E1.23894
G1 X117.795 Y128.228 E.02318
G1 X140.811 Y105.212 E1.25847
G1 X141.65 Y105.212 E.03243
G1 X118.276 Y128.586 E1.278
G3 X118.736 Y128.964 I-5.994 J7.744 E.02304
G1 X141.945 Y105.755 E1.26901
G1 X141.945 Y106.594 E.03243
G1 X119.158 Y129.382 E1.24596
G3 X119.558 Y129.82 I-1.402 J1.684 E.02302
G1 X141.945 Y107.432 E1.22406
G1 X141.945 Y108.271 E.03243
G1 X119.922 Y130.294 E1.20415
G3 X120.246 Y130.809 I-3.208 J2.377 E.02354
G1 X141.945 Y109.11 E1.18644
G1 X141.945 Y109.949 E.03243
M73 P41 R11
G1 X120.541 Y131.353 E1.17032
G3 X120.784 Y131.949 I-3.776 J1.884 E.02491
G1 X141.945 Y110.787 E1.15706
G1 X141.945 Y111.626 E.03243
G1 X120.992 Y132.58 E1.14568
G1 X121.145 Y133.265 E.02715
G1 X141.945 Y112.465 E1.13728
G1 X141.945 Y113.304 E.03243
G1 X121.234 Y134.015 E1.13241
G3 X121.256 Y134.832 I-5.698 J.559 E.03163
G1 X141.945 Y114.142 E1.13123
G1 X141.945 Y114.981 E.03243
G1 X121.151 Y135.776 E1.13698
G1 X121.111 Y135.983 E.00817
G1 X121.566 Y136.199 E.01949
G1 X141.945 Y115.82 E1.11426
G1 X141.945 Y116.658 E.03243
G1 X122.135 Y136.469 E1.08314
G3 X122.703 Y136.739 I-3.456 J7.975 E.02433
G1 X141.945 Y117.497 E1.0521
G1 X141.945 Y118.336 E.03243
G1 X123.255 Y137.026 E1.0219
G1 X123.808 Y137.312 E.02405
G1 X141.945 Y119.175 E.99171
G1 X141.945 Y120.013 E.03243
G1 X124.36 Y137.599 E.96151
G3 X124.898 Y137.9 I-4.172 J8.092 E.02383
G1 X141.945 Y120.852 E.9321
G1 X141.945 Y121.691 E.03243
G1 X125.435 Y138.202 E.90275
G3 X125.969 Y138.506 I-4.674 J8.816 E.02378
G1 X141.945 Y122.53 E.87355
G1 X141.945 Y123.368 E.03243
G1 X126.492 Y138.822 E.84495
G1 X127.015 Y139.138 E.02362
G1 X141.945 Y124.207 E.81636
G1 X141.945 Y125.046 E.03243
G1 X127.53 Y139.461 E.78816
G1 X128.04 Y139.79 E.02345
G1 X141.945 Y125.884 E.7603
G1 X141.945 Y126.723 E.03243
G1 X128.55 Y140.119 E.73244
G3 X129.048 Y140.459 I-5.211 J8.171 E.02334
G1 X141.945 Y127.562 E.70518
G1 X141.945 Y128.401 E.03243
G1 X129.544 Y140.802 E.67807
G3 X130.04 Y141.145 I-4.977 J7.722 E.02331
G1 X141.945 Y129.239 E.65096
G1 X141.945 Y130.078 E.03243
G1 X130.521 Y141.502 E.62464
G1 X131.003 Y141.86 E.02318
G1 X141.945 Y130.917 E.59832
G1 X141.945 Y131.756 E.03243
G1 X131.484 Y142.217 E.57199
G3 X131.956 Y142.584 I-6.415 J8.751 E.02311
G1 X141.945 Y132.594 E.54618
G1 X141.945 Y133.433 E.03243
G1 X132.426 Y142.953 E.5205
G1 X132.896 Y143.322 E.02309
G1 X141.945 Y134.272 E.49482
G1 X141.945 Y135.11 E.03243
G1 X133.355 Y143.701 E.46971
G1 X133.812 Y144.083 E.02302
G1 X141.945 Y135.949 E.44472
G1 X141.945 Y136.788 E.03243
G1 X134.269 Y144.464 E.41973
G3 X134.716 Y144.856 I-5.888 J7.18 E.02298
G1 X141.945 Y137.627 E.39527
G1 X141.945 Y138.465 E.03243
G1 X135.16 Y145.251 E.37101
G1 X135.603 Y145.646 E.02297
G1 X141.945 Y139.304 E.34676
G1 X141.945 Y140.143 E.03243
G1 X136.033 Y146.055 E.32328
G1 X136.462 Y146.465 E.02294
G1 X141.945 Y140.982 E.2998
G1 X141.945 Y141.82 E.03243
G1 X136.892 Y146.874 E.27633
G3 X137.314 Y147.291 I-5.989 J6.493 E.02293
G1 X141.945 Y142.659 E.25324
G1 X141.945 Y143.498 E.03243
G1 X137.729 Y147.714 E.23054
G1 X138.144 Y148.138 E.02293
G1 X141.945 Y144.336 E.20784
G1 X141.945 Y145.175 E.03243
G1 X138.55 Y148.571 E.18564
G1 X138.954 Y149.005 E.02294
G1 X141.945 Y146.014 E.16355
G1 X141.945 Y146.853 E.03243
G1 X139.352 Y149.446 E.1418
G1 X139.746 Y149.891 E.02297
G1 X141.945 Y147.691 E.12028
G1 X141.945 Y148.53 E.03243
G1 X140.136 Y150.339 E.09892
G1 X140.52 Y150.794 E.02301
G1 X141.945 Y149.369 E.07792
G1 X141.945 Y150.208 E.03243
G1 X140.9 Y151.253 E.05716
G1 X141.275 Y151.717 E.02306
G1 X141.945 Y151.046 E.03667
G1 X141.945 Y151.885 E.03243
G1 X141.458 Y152.373 E.02666
; CHANGE_LAYER
; Z_HEIGHT: 0.52
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13278.366
G1 X141.945 Y151.885 E-.26204
G1 X141.945 Y151.575 E-.11796
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L3
M991 S0 P2 ;notify layer change


G17
G3 Z.76 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.266 Y103.74
G1 Z.52
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.206 Y115.358 I-51.41 J-19.427 E.50146
G3 X123.072 Y118.225 I-29.032 J-19.392 E.13652
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-41.028 J-37.519 E.29802
G1 X118.015 Y129.012 E.15054
G3 X118.974 Y129.892 I-4.302 J5.65 E.04977
G3 X120.075 Y131.529 I-6.068 J5.269 E.07551
G1 X120.45 Y132.48 E.03904
G3 X120.765 Y134.489 I-7.443 J2.193 E.07786
G1 X120.698 Y135.504 E.03882
G1 X120.538 Y136.268 E.02983
G3 X130.508 Y142.101 I-22.327 J49.595 E.44182
G3 X142.443 Y154.069 I-32.646 J44.495 E.64786
G1 X142.443 Y104.514 E1.89194
G1 X142.255 Y104.621 E.00825
G1 X141.654 Y104.716 E.02324
G1 X132.937 Y104.716 E.33282
G1 X132.65 Y104.695 E.01099
G1 X132.13 Y104.541 E.02068
G1 X131.692 Y104.267 E.01976
G1 X131.307 Y103.837 E.02203
G1 X131.301 Y103.823 E.00055
G1 X131.63 Y103.177 F36000
; LINE_WIDTH: 0.749101
G1 F10367.331
G1 X131.612 Y103.139 E.00194
; LINE_WIDTH: 0.792136
G1 F10236.588
G1 X131.556 Y103.022 E.00642
; LINE_WIDTH: 0.835171
G1 F9833.488
G1 X131.5 Y102.905 E.00678
; LINE_WIDTH: 0.878206
G1 F9332.007
G1 X131.444 Y102.788 E.00714
; LINE_WIDTH: 0.924184
G1 F8849.827
G1 X131.425 Y102.733 E.00337
; LINE_WIDTH: 0.970162
G1 F8415.026
G1 X131.406 Y102.678 E.00355
; LINE_WIDTH: 1.01614
G1 F8020.95
G1 X131.387 Y102.623 E.00372
; LINE_WIDTH: 1.06212
G1 F7662.131
G1 X131.368 Y102.568 E.00389
; LINE_WIDTH: 1.1081
G1 F7334.042
G1 X131.349 Y102.513 E.00407
G1 X131.309 Y102.558 E.0042
; LINE_WIDTH: 1.06212
G1 F7662.131
G1 X131.269 Y102.603 E.00402
; LINE_WIDTH: 1.01614
G1 F8020.95
G1 X131.229 Y102.648 E.00384
; LINE_WIDTH: 0.970162
G1 F8415.026
G1 X131.189 Y102.693 E.00366
; LINE_WIDTH: 0.924184
G1 F8849.827
G1 X131.149 Y102.738 E.00348
; LINE_WIDTH: 0.878206
G1 F9332.007
G1 X131.073 Y102.882 E.00897
; LINE_WIDTH: 0.835171
G1 F9833.488
G1 X130.997 Y103.026 E.00852
; LINE_WIDTH: 0.792136
G1 F10391.925
G1 X130.921 Y103.17 E.00806
; LINE_WIDTH: 0.749101
G1 F10913.498
G1 X130.845 Y103.315 E.0076
; LINE_WIDTH: 0.706066
G1 F11447.841
G1 X130.769 Y103.459 E.00714
; LINE_WIDTH: 0.663031
G1 F11994.955
G1 X130.693 Y103.603 E.00669
; LINE_WIDTH: 0.619996
G1 F13390.715
G1 X130.547 Y103.975 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.208 Y104.835 E.01998
G3 X124.719 Y115.033 I-50.387 J-20.543 E.443
G3 X122.623 Y117.848 I-28.435 J-18.985 E.1341
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-40.124 J-36.67 E.32255
G1 X117.666 Y129.482 E.17935
G1 X118.105 Y129.848 E.02184
G3 X119.208 Y131.168 I-5.586 J5.787 E.06578
G1 X119.562 Y131.811 E.02803
G1 X119.901 Y132.684 E.03574
G3 X120.135 Y133.79 I-9.572 J2.601 E.04321
G1 X120.18 Y134.525 E.02808
G1 X120.115 Y135.453 E.03554
G1 X119.933 Y136.284 E.03246
G1 X119.836 Y136.599 E.01259
G3 X130.932 Y143.157 I-21.043 J48.275 E.49336
G3 X142.92 Y155.769 I-32.376 J42.777 E.66727
G1 X143.029 Y155.749 E.00421
G1 X143.029 Y104.01 E1.97533
; LINE_WIDTH: 0.637816
G1 F13049.315
G1 X143.02 Y103.262 E.02941
; LINE_WIDTH: 0.651456
G1 F12760.89
G1 X143.013 Y103.005 E.01037
G1 X142.94 Y103.248 E.01022
; LINE_WIDTH: 0.637816
G1 F13049.315
G1 X142.552 Y103.793 E.02634
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.075 Y104.064 E.02094
G1 X141.654 Y104.13 E.01626
G1 X132.937 Y104.13 E.33282
G1 X132.736 Y104.116 E.00769
G1 X132.346 Y103.996 E.01557
G1 X132.066 Y103.816 E.01271
G1 X131.781 Y103.49 E.01652
; LINE_WIDTH: 0.663031
G1 F12163.194
G1 X131.725 Y103.373 E.00532
; LINE_WIDTH: 0.706066
G1 F11723.461
G1 X131.669 Y103.258 E.00561
G1 X132.185 Y102.854 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.158 Y102.761 E.00368
G1 X132.171 Y102.625 E.00523
G1 X132.742 Y99.542 E.1197
G3 X131.452 Y99.491 I.984 J-41.197 E.04931
G3 X124.233 Y114.707 I-51.937 J-15.32 E.64564
G3 X122.174 Y117.472 I-27.834 J-18.574 E.13167
G1 X119.604 Y120.538 E.15272
G3 X112.93 Y126.698 I-40.673 J-37.373 E.34717
G1 X117.317 Y129.952 E.20855
G3 X118.134 Y130.709 I-3.528 J4.633 E.04258
G3 X118.695 Y131.45 I-10.598 J8.586 E.0355
G1 X119.049 Y132.094 E.02803
G1 X119.352 Y132.888 E.03245
G3 X119.551 Y133.826 I-11.675 J2.958 E.03663
G1 X119.595 Y134.56 E.02808
G1 X119.531 Y135.403 E.03226
G1 X119.363 Y136.148 E.02918
G1 X119.08 Y136.91 E.03102
G3 X130.579 Y143.624 I-20.72 J48.692 E.50972
G3 X142.648 Y156.415 I-31.964 J42.247 E.67453
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.504 E2.16594
G3 X141.849 Y99.544 I-1.29 J-18.061 E.06745
G1 X142.421 Y102.632 E.11989
G3 X142.383 Y103.039 I-.767 J.135 E.01581
G1 X142.166 Y103.352 E.01454
G1 X141.894 Y103.507 E.01195
G1 X141.654 Y103.545 E.00928
G1 X132.937 Y103.545 E.33282
G1 X132.6 Y103.468 E.0132
G3 X132.277 Y103.18 I.337 J-.702 E.01674
G1 X132.209 Y102.94 E.0095
M204 S250
G1 X132.715 Y102.725 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X133.406 Y98.991 E.12023
G1 X132.441 Y98.988 E.03057
G2 X131.038 Y98.937 I-1.214 J13.969 E.04448
G3 X123.77 Y114.404 I-51.141 J-14.588 E.54341
G3 X121.747 Y117.121 I-27.193 J-18.137 E.10731
G1 X119.177 Y120.187 E.12664
G3 X112.01 Y126.704 I-40.321 J-37.143 E.3071
G1 X116.988 Y130.396 E.19622
G1 X117.372 Y130.719 E.0159
G3 X118.21 Y131.717 I-5.287 J5.292 E.0413
G1 X118.564 Y132.36 E.02325
G1 X118.834 Y133.08 E.02434
G1 X118.953 Y133.631 E.01785
G1 X119.044 Y134.594 E.03063
G1 X118.98 Y135.355 E.02418
G1 X118.825 Y136.021 E.02163
G1 X118.547 Y136.754 E.02481
G1 X118.31 Y137.19 E.01572
G3 X130.249 Y144.067 I-19.805 J48.182 E.43751
G3 X142.39 Y157.025 I-31.646 J41.817 E.56491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.232 E1.81929
G1 X144.167 Y98.937 E.00934
G2 X142.833 Y98.99 I-.077 J14.765 E.04229
G3 X141.185 Y98.992 I-.88 J-64.082 E.0522
G1 X141.876 Y102.727 E.12028
G1 X141.865 Y102.845 E.00376
G1 X141.724 Y102.981 E.00621
G1 X141.654 Y102.992 E.00223
G1 X132.937 Y102.992 E.27599
G1 X132.793 Y102.94 E.00485
G1 X132.732 Y102.81 E.00453
; WIPE_START
M204 S10000
G1 X132.906 Y101.826 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X140.487 Y102.71 Z.92 F36000
G1 X143.013 Y103.005 Z.92
G1 Z.52
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.653816
G1 F12712.275
G1 X143.012 Y102.515 E.01978
; LINE_WIDTH: 0.686496
M73 P42 R11
G1 F12075.259
G1 X142.996 Y102.337 E.0076
; LINE_WIDTH: 0.732211
G1 F11284.257
G1 X142.973 Y102.088 E.01137
; LINE_WIDTH: 0.777926
G1 F10590.514
G1 X142.95 Y101.839 E.01212
; LINE_WIDTH: 0.823641
G1 F9977.133
G1 X142.927 Y101.59 E.01286
; LINE_WIDTH: 0.869356
G1 F9430.912
G1 X142.904 Y101.341 E.01361
; LINE_WIDTH: 0.915071
G1 F8941.397
G1 X142.881 Y101.092 E.01435
; LINE_WIDTH: 0.960786
G1 F8500.19
G1 X142.858 Y100.843 E.0151
; LINE_WIDTH: 1.0065
G1 F8100.477
G1 X142.836 Y100.594 E.01584
; LINE_WIDTH: 1.05222
G1 F7736.669
G1 X142.813 Y100.345 E.01659
; WIPE_START
G1 X142.836 Y100.594 E-.095
G1 X142.858 Y100.843 E-.095
G1 X142.881 Y101.092 E-.095
G1 X142.904 Y101.341 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.311 Y102.111 Z.92 F36000
G1 X131.349 Y102.513 Z.92
G1 Z.52
G1 E.4 F1800
; LINE_WIDTH: 1.1081
G1 F7334.042
G1 X131.359 Y102.474 E.00283
; LINE_WIDTH: 1.10156
G1 F7378.985
G1 X131.423 Y102.23 E.01759
; LINE_WIDTH: 1.06404
G1 F7647.859
G1 X131.487 Y101.985 E.01697
; LINE_WIDTH: 1.02652
G1 F7937.068
G1 X131.552 Y101.741 E.01635
; LINE_WIDTH: 0.988996
G1 F8249.011
G1 X131.616 Y101.496 E.01573
; LINE_WIDTH: 0.951476
G1 F8586.476
G1 X131.625 Y101.46 E.00225
; LINE_WIDTH: 0.946376
G1 F8634.491
G1 X131.704 Y101.145 E.01927
; LINE_WIDTH: 0.906216
G1 F9032.206
G1 X131.783 Y100.831 E.01842
; LINE_WIDTH: 0.866056
G1 F9468.331
G1 X131.861 Y100.516 E.01757
; LINE_WIDTH: 0.825896
G1 F9948.71
G1 X131.94 Y100.202 E.01672
; LINE_WIDTH: 0.785736
G1 F10480.437
G1 X131.945 Y100.183 E.00097
; WIPE_START
G1 X131.94 Y100.202 E-.00752
G1 X131.861 Y100.516 E-.12316
G1 X131.783 Y100.831 E-.12316
G1 X131.704 Y101.145 E-.12316
G1 X131.702 Y101.153 E-.003
; WIPE_END
G1 E-.02 F1800
G1 X138.619 Y104.38 Z.92 F36000
G1 X142.209 Y106.055 Z.92
G1 Z.52
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.626766
G1 F13292.711
G1 X141.368 Y105.214 E.04595
G1 X140.53 Y105.214 E.03236
G1 X141.945 Y106.629 E.07731
G1 X141.945 Y107.467 E.03236
G1 X139.692 Y105.214 E.12307
G1 X138.854 Y105.214 E.03236
G1 X141.945 Y108.305 E.16883
G1 X141.945 Y109.143 E.03236
G1 X138.016 Y105.214 E.21459
G1 X137.179 Y105.214 E.03236
G1 X141.945 Y109.981 E.26035
G1 X141.945 Y110.819 E.03236
G1 X136.341 Y105.214 E.30611
G1 X135.503 Y105.214 E.03236
G1 X141.945 Y111.656 E.35187
G1 X141.945 Y112.494 E.03236
G1 X134.665 Y105.214 E.39763
G1 X133.827 Y105.214 E.03236
G1 X141.945 Y113.332 E.44339
G1 X141.945 Y114.17 E.03236
G1 X132.989 Y105.214 E.48915
G3 X132.57 Y105.187 I-.03 J-2.851 E.01626
G1 X131.939 Y105.001 E.0254
G1 X141.945 Y115.008 E.54654
G1 X141.945 Y115.845 E.03236
G1 X131.242 Y105.142 E.58458
G1 X131 Y105.738 E.02483
G1 X141.945 Y116.683 E.59781
G1 X141.945 Y117.521 E.03236
G1 X130.754 Y106.33 E.61125
G1 X130.492 Y106.906 E.02443
G1 X141.945 Y118.359 E.62555
G1 X141.945 Y119.197 E.03236
G1 X130.23 Y107.482 E.63985
G1 X129.968 Y108.058 E.02444
G1 X141.945 Y120.035 E.65415
G1 X141.945 Y120.872 E.03236
G1 X129.698 Y108.625 E.66892
G1 X129.42 Y109.185 E.02414
G1 X141.945 Y121.71 E.68409
G1 X141.945 Y122.548 E.03236
G1 X129.139 Y109.741 E.69948
G1 X128.846 Y110.287 E.0239
G1 X141.945 Y123.386 E.71544
G1 X141.945 Y124.224 E.03236
G1 X128.554 Y110.832 E.7314
G1 X128.262 Y111.378 E.0239
G1 X141.945 Y125.062 E.74736
G1 X141.945 Y125.899 E.03236
G1 X127.956 Y111.91 E.76407
G1 X127.65 Y112.442 E.0237
G1 X141.945 Y126.737 E.78079
G1 X141.945 Y127.575 E.03236
G1 X127.332 Y112.962 E.79815
G1 X127.011 Y113.479 E.0235
G1 X141.945 Y128.413 E.81567
G1 X141.945 Y129.251 E.03236
G1 X126.69 Y113.995 E.8332
G1 X126.369 Y114.512 E.0235
G1 X141.945 Y130.088 E.85073
G1 X141.945 Y130.926 E.03236
G1 X126.034 Y115.015 E.86905
G1 X125.698 Y115.517 E.02333
G1 X141.945 Y131.764 E.88738
G1 X141.945 Y132.602 E.03236
G1 X125.354 Y116.01 E.9062
G1 X125.005 Y116.499 E.0232
G1 X141.945 Y133.44 E.92526
G1 X141.945 Y134.278 E.03236
G1 X124.656 Y116.988 E.94432
G3 X124.302 Y117.472 I-8.454 J-5.795 E.02316
G1 X141.945 Y135.115 E.96362
G1 X141.945 Y135.953 E.03236
G1 X123.941 Y117.949 E.98336
G3 X123.565 Y118.411 I-4.955 J-3.645 E.02301
G1 X141.945 Y136.791 E1.00388
G1 X141.945 Y137.629 E.03236
G1 X123.183 Y118.867 E1.02475
G1 X122.801 Y119.322 E.02297
G1 X141.945 Y138.467 E1.04562
G1 X141.945 Y139.305 E.03236
G1 X122.419 Y119.778 E1.06648
G1 X122.037 Y120.234 E.02297
G1 X141.945 Y140.142 E1.08735
G1 X141.945 Y140.98 E.03236
G1 X121.655 Y120.69 E1.10822
G1 X121.273 Y121.145 E.02297
G1 X141.945 Y141.818 E1.12908
G1 X141.945 Y142.656 E.03236
G1 X120.891 Y121.601 E1.14995
G3 X120.489 Y122.038 I-4.827 J-4.037 E.02291
G1 X141.945 Y143.494 E1.17187
G1 X141.945 Y144.331 E.03236
G1 X120.082 Y122.468 E1.19413
G1 X119.668 Y122.892 E.02288
G1 X141.945 Y145.169 E1.21674
G1 X141.945 Y146.007 E.03236
G1 X119.254 Y123.316 E1.23935
G3 X118.839 Y123.738 I-7.142 J-6.606 E.02288
G1 X141.945 Y146.845 E1.26203
G1 X141.945 Y147.683 E.03236
G1 X118.412 Y124.149 E1.28534
G1 X117.985 Y124.56 E.02288
G1 X141.945 Y148.521 E1.30864
G1 X141.945 Y149.358 E.03236
G1 X117.549 Y124.962 E1.3325
G1 X117.109 Y125.36 E.02291
G1 X141.945 Y150.196 E1.3565
G1 X141.945 Y151.034 E.03236
G1 X116.669 Y125.758 E1.38051
G3 X116.225 Y126.151 I-6.646 J-7.071 E.02293
G1 X141.945 Y151.872 E1.4048
G1 X141.945 Y152.574 E.02711
G2 X131.359 Y142.124 I-43.355 J33.332 E.57627
G1 X120.494 Y131.258 E.59345
G3 X120.954 Y132.414 I-7.121 J3.508 E.0481
G1 X120.996 Y132.598 E.00728
G1 X128.46 Y140.062 E.40765
G2 X126.215 Y138.655 I-23.914 J35.664 E.10231
G1 X121.209 Y133.649 E.27341
G1 X121.265 Y134.542 E.03457
G1 X124.279 Y137.557 E.16466
G1 X122.55 Y136.666 E.07513
G1 X120.963 Y135.078 E.0867
; WIPE_START
G1 X121.67 Y135.785 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X118.06 Y129.061 Z.92 F36000
G1 X117.786 Y128.55 Z.92
G1 Z.52
G1 E.4 F1800
G1 F13292.711
G1 X115.585 Y126.35 E.12018
; CHANGE_LAYER
; Z_HEIGHT: 0.68
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13292.711
G1 X116.292 Y127.057 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L4
M991 S0 P3 ;notify layer change


G17
G3 Z.92 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.264 Y103.746
G1 Z.68
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.131 I-51.425 J-19.441 E.49077
G3 X123.072 Y118.224 I-29.872 J-19.715 E.14698
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-40.695 J-37.169 E.29803
G1 X118.03 Y129.023 E.15128
G3 X118.977 Y129.895 I-4.327 J5.647 E.0492
G3 X120.076 Y131.532 I-6.088 J5.276 E.07546
G1 X120.45 Y132.48 E.03892
G3 X120.765 Y134.489 I-7.4 J2.187 E.07787
G1 X120.698 Y135.504 E.03881
G1 X120.538 Y136.268 E.02981
G3 X131.285 Y142.689 I-21.886 J48.832 E.47905
G3 X142.443 Y154.069 I-32.923 J43.443 E.61068
G1 X142.443 Y104.518 E1.89181
G1 X142.258 Y104.623 E.00814
G1 X141.657 Y104.718 E.02324
G1 X132.934 Y104.718 E.33301
G1 X132.646 Y104.697 E.01102
G1 X132.117 Y104.538 E.02112
G1 X131.683 Y104.263 E.01961
G1 X131.303 Y103.837 E.0218
G1 X131.3 Y103.829 E.00034
G1 X131.628 Y103.178 F36000
; LINE_WIDTH: 0.749761
G1 F10358.901
G1 X131.609 Y103.14 E.00201
; LINE_WIDTH: 0.793016
G1 F10223.348
G1 X131.553 Y103.022 E.00643
; LINE_WIDTH: 0.836271
G1 F9819.998
G1 X131.497 Y102.905 E.0068
; LINE_WIDTH: 0.879526
G1 F9317.432
G1 X131.44 Y102.788 E.00717
; LINE_WIDTH: 0.9249
G1 F8842.712
G1 X131.422 Y102.733 E.00334
; LINE_WIDTH: 0.970274
G1 F8414.02
G1 X131.403 Y102.679 E.00351
; LINE_WIDTH: 1.01565
G1 F8024.971
G1 X131.384 Y102.624 E.00368
; LINE_WIDTH: 1.06102
G1 F7670.31
G1 X131.366 Y102.57 E.00385
; LINE_WIDTH: 1.1064
G1 F7345.671
G1 X131.347 Y102.515 E.00402
G1 X131.308 Y102.56 E.00415
; LINE_WIDTH: 1.06102
G1 F7670.31
G1 X131.268 Y102.604 E.00397
; LINE_WIDTH: 1.01565
G1 F8024.971
G1 X131.229 Y102.649 E.0038
; LINE_WIDTH: 0.970274
G1 F8414.02
G1 X131.189 Y102.693 E.00362
; LINE_WIDTH: 0.9249
G1 F8842.712
G1 X131.15 Y102.738 E.00345
; LINE_WIDTH: 0.879526
G1 F9317.432
G1 X131.074 Y102.882 E.00899
; LINE_WIDTH: 0.836271
G1 F9819.998
G1 X130.998 Y103.026 E.00853
; LINE_WIDTH: 0.793016
G1 F10379.871
G1 X130.921 Y103.17 E.00807
; LINE_WIDTH: 0.749761
G1 F10901.225
G1 X130.845 Y103.315 E.00761
; LINE_WIDTH: 0.706506
G1 F11435.354
G1 X130.769 Y103.459 E.00715
; LINE_WIDTH: 0.663251
G1 F11982.241
G1 X130.693 Y103.603 E.00669
; LINE_WIDTH: 0.619996
G1 F13377.281
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.215 Y104.819 E.01934
G3 X124.88 Y114.797 I-50.408 J-20.534 E.43276
G3 X122.623 Y117.848 I-29.318 J-19.328 E.14497
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-39.806 J-36.329 E.32257
G1 X117.677 Y129.49 E.17987
G1 X118.105 Y129.848 E.02133
G3 X119.209 Y131.171 I-5.592 J5.791 E.0659
G1 X119.563 Y131.814 E.02801
G1 X119.901 Y132.684 E.03564
G3 X120.135 Y133.793 I-9.441 J2.573 E.0433
G1 X120.18 Y134.525 E.02801
G1 X120.115 Y135.453 E.03553
G1 X119.931 Y136.289 E.03268
G1 X119.836 Y136.599 E.01238
G3 X130.931 Y143.156 I-21.125 J48.415 E.4933
G3 X142.92 Y155.769 I-32.361 J42.765 E.66733
G1 X143.029 Y155.749 E.00421
G1 X143.029 Y104.011 E1.97529
; LINE_WIDTH: 0.636396
G1 F13080.094
G1 X143.021 Y103.264 E.02932
; LINE_WIDTH: 0.648936
G1 F12813.212
G1 X143.014 Y103.007 E.01032
G1 X142.942 Y103.25 E.01017
; LINE_WIDTH: 0.636396
G1 F13080.094
G1 X142.554 Y103.795 E.02627
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.077 Y104.066 E.02094
G1 X141.657 Y104.133 E.01626
G1 X132.934 Y104.133 E.33301
G1 X132.733 Y104.118 E.00771
G1 X132.336 Y103.994 E.01588
G1 X132.059 Y103.814 E.01261
G1 X131.777 Y103.491 E.01635
; LINE_WIDTH: 0.663251
G1 F12155.948
G1 X131.721 Y103.374 E.00533
; LINE_WIDTH: 0.706506
G1 F11715.787
G1 X131.666 Y103.26 E.00556
G1 X132.182 Y102.855 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.156 Y102.762 E.00371
G1 X132.169 Y102.627 E.00518
G1 X132.74 Y99.544 E.1197
G3 X131.452 Y99.491 I.981 J-39.461 E.04922
G3 X124.399 Y114.463 I-51.964 J-15.33 E.63434
G3 X122.174 Y117.472 I-28.762 J-18.94 E.14296
G1 X119.605 Y120.537 E.15272
G3 X112.93 Y126.698 I-40.405 J-37.082 E.34719
G1 X117.323 Y129.957 E.20885
G3 X118.137 Y130.711 I-3.534 J4.627 E.04243
G3 X118.696 Y131.453 I-10.692 J8.648 E.03547
G1 X119.05 Y132.096 E.02801
G1 X119.352 Y132.887 E.03235
G3 X119.531 Y135.403 I-6.115 J1.699 E.09692
G1 X119.361 Y136.154 E.0294
G1 X119.08 Y136.91 E.03081
G3 X130.578 Y143.623 I-20.901 J49.002 E.50965
G3 X142.648 Y156.415 I-31.956 J42.241 E.67458
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.505 E2.16591
G3 X141.852 Y99.546 I-1.288 J-17.451 E.06736
G1 X142.423 Y102.634 E.11989
G3 X142.386 Y103.041 I-.767 J.135 E.01581
G1 X142.169 Y103.355 E.01454
G1 X141.897 Y103.509 E.01195
G1 X141.657 Y103.547 E.00928
G1 X132.934 Y103.547 E.33301
G1 X132.593 Y103.468 E.01338
G3 X132.274 Y103.181 I.342 J-.7 E.01659
G1 X132.207 Y102.942 E.00949
M204 S250
G1 X132.712 Y102.727 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X133.404 Y98.993 E.12023
G1 X132.441 Y98.99 E.03049
G2 X131.038 Y98.937 I-1.213 J13.429 E.04448
G3 X123.946 Y114.147 I-51.15 J-14.591 E.53357
G3 X121.747 Y117.121 I-28.179 J-18.531 E.11715
G1 X119.177 Y120.186 E.12664
G3 X112.01 Y126.704 I-38.869 J-35.546 E.30714
G1 X116.989 Y130.398 E.19629
G1 X117.372 Y130.719 E.01583
G3 X118.212 Y131.719 I-5.263 J5.272 E.04139
M73 P43 R11
G1 X118.565 Y132.362 E.02323
G3 X118.98 Y135.355 I-5.028 J2.223 E.09691
G1 X118.824 Y136.026 E.02181
G1 X118.547 Y136.753 E.02463
G1 X118.31 Y137.19 E.01573
G3 X130.248 Y144.067 I-19.584 J47.798 E.43749
G3 X142.39 Y157.025 I-31.645 J41.818 E.56495
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.231 E1.8193
G1 X144.167 Y98.937 E.00932
G2 X142.832 Y98.992 I-.071 J14.417 E.04232
G3 X141.187 Y98.994 I-.871 J-53.926 E.05209
G1 X141.879 Y102.729 E.12028
G1 X141.868 Y102.848 E.00376
G1 X141.726 Y102.983 E.00621
G1 X141.657 Y102.994 E.00223
G1 X132.934 Y102.994 E.27615
G1 X132.789 Y102.941 E.00488
G1 X132.73 Y102.812 E.0045
; WIPE_START
M204 S10000
G1 X132.904 Y101.828 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X140.485 Y102.712 Z1.08 F36000
G1 X143.014 Y103.007 Z1.08
G1 Z.68
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.651276
G1 F12764.614
G1 X143.013 Y102.517 E.01968
; LINE_WIDTH: 0.684206
G1 F12117.809
G1 X142.997 Y102.338 E.00763
; LINE_WIDTH: 0.729922
G1 F11321.387
G1 X142.974 Y102.089 E.01134
; LINE_WIDTH: 0.775639
G1 F10623.194
G1 X142.951 Y101.84 E.01208
; LINE_WIDTH: 0.821355
G1 F10006.115
G1 X142.928 Y101.591 E.01283
; LINE_WIDTH: 0.867071
G1 F9456.79
G1 X142.905 Y101.342 E.01357
; LINE_WIDTH: 0.912787
G1 F8964.641
G1 X142.882 Y101.093 E.01432
; LINE_WIDTH: 0.958504
G1 F8521.183
G1 X142.86 Y100.844 E.01506
; LINE_WIDTH: 1.00422
G1 F8119.53
G1 X142.837 Y100.595 E.01581
; LINE_WIDTH: 1.04994
G1 F7754.037
G1 X142.814 Y100.346 E.01655
; WIPE_START
G1 X142.837 Y100.595 E-.095
G1 X142.86 Y100.844 E-.095
G1 X142.882 Y101.093 E-.095
G1 X142.905 Y101.342 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.312 Y102.113 Z1.08 F36000
G1 X131.347 Y102.515 Z1.08
G1 Z.68
G1 E.4 F1800
; LINE_WIDTH: 1.1064
G1 F7345.671
G1 X131.357 Y102.476 E.00283
; LINE_WIDTH: 1.09984
G1 F7390.896
G1 X131.422 Y102.231 E.01761
; LINE_WIDTH: 1.06221
G1 F7661.438
G1 X131.486 Y101.986 E.01699
; LINE_WIDTH: 1.02459
G1 F7952.538
G1 X131.551 Y101.741 E.01637
; LINE_WIDTH: 0.986961
G1 F8266.633
G1 X131.615 Y101.495 E.01575
; LINE_WIDTH: 0.949336
G1 F8606.558
G1 X131.625 Y101.459 E.00224
; LINE_WIDTH: 0.944216
G1 F8654.988
G1 X131.703 Y101.145 E.01922
; LINE_WIDTH: 0.904066
G1 F9054.535
G1 X131.782 Y100.83 E.01837
; LINE_WIDTH: 0.863916
G1 F9492.757
G1 X131.86 Y100.516 E.01752
; LINE_WIDTH: 0.823766
G1 F9975.553
G1 X131.939 Y100.202 E.01668
; LINE_WIDTH: 0.783616
G1 F10510.091
G1 X131.944 Y100.183 E.00095
; WIPE_START
G1 X131.939 Y100.202 E-.00739
G1 X131.86 Y100.516 E-.12313
G1 X131.782 Y100.83 E-.12313
G1 X131.703 Y101.145 E-.12313
G1 X131.701 Y101.153 E-.00322
; WIPE_END
G1 E-.02 F1800
G1 X132.042 Y104.757 Z1.08 F36000
G1 Z.68
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.627366
G1 F13279.262
G1 X130.958 Y105.841 E.05923
G3 X130.292 Y107.346 I-18.907 J-7.469 E.06364
G1 X132.476 Y105.162 E.11937
G2 X133.26 Y105.216 I.611 J-3.17 E.03049
G1 X129.547 Y108.929 E.203
G3 X128.654 Y110.661 I-46.354 J-22.818 E.07537
G1 X134.099 Y105.216 E.29772
G1 X134.938 Y105.216 E.03242
G1 X127.538 Y112.616 E.40456
G3 X126.153 Y114.84 I-46.22 J-27.249 E.10132
G1 X135.776 Y105.216 E.52617
G1 X136.615 Y105.216 E.03242
G1 X125.506 Y116.325 E.60735
G3 X125.801 Y116.607 I-1.992 J2.376 E.0158
G1 X125.921 Y116.749 E.00716
G1 X137.454 Y105.216 E.63053
G1 X138.293 Y105.216 E.03242
G1 X126.271 Y117.238 E.65727
G3 X126.56 Y117.788 I-53.426 J28.439 E.02401
G1 X139.131 Y105.216 E.68732
G1 X139.97 Y105.216 E.03242
G1 X126.784 Y118.402 E.7209
G3 X126.876 Y118.811 I-2.976 J.88 E.01624
G1 X126.901 Y119.124 E.01211
G1 X140.809 Y105.216 E.76037
G1 X141.647 Y105.216 E.03242
G1 X126.393 Y120.47 E.83398
G1 X126.479 Y119.993 F36000
; FEATURE: Top surface
; LINE_WIDTH: 0.62
G1 F9000
M204 S2000
G1 X122.324 Y124.147 E.22432
G1 X122.127 Y124.345
G1 X121.322 Y124.321
G1 X121.519 Y124.124
G1 X126.577 Y119.066 E.27311
G1 X126.774 Y118.869
G1 X126.638 Y118.177
G1 X126.441 Y118.374
G1 X120.86 Y123.955 E.30135
G1 X120.662 Y124.153
G1 X120.063 Y123.923
G1 X120.261 Y123.726
G1 X126.197 Y117.789 E.32055
G1 X126.395 Y117.592
G1 X126.099 Y117.059
G1 X125.902 Y117.256
G1 X119.746 Y123.412 E.33239
G1 X119.549 Y123.61
G1 X119.098 Y123.233
G1 X119.295 Y123.035
G1 X125.535 Y116.795 E.33695
M204 S10000
G1 X125.142 Y116.401 F36000
; FEATURE: Gap infill
; LINE_WIDTH: 0.501137
G1 F3000
G1 X124.956 Y116.621 E.00876
; LINE_WIDTH: 0.462951
G1 X124.771 Y116.841 E.00804
; LINE_WIDTH: 0.424765
G1 X124.585 Y117.061 E.00732
; LINE_WIDTH: 0.386221
G1 X124.396 Y117.285 E.00672
; LINE_WIDTH: 0.344333
G1 X124.18 Y117.532 E.00663
; LINE_WIDTH: 0.299654
G1 X123.964 Y117.78 E.00568
; LINE_WIDTH: 0.254649
G1 X123.745 Y118.031 E.00478
; LINE_WIDTH: 0.210868
G1 X123.41 Y118.396 E.00571
; LINE_WIDTH: 0.168364
G1 X123.075 Y118.762 E.00433
; LINE_WIDTH: 0.125861
G1 X122.74 Y119.128 E.00296
; WIPE_START
G1 X123.075 Y118.762 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X126.274 Y120.652 Z1.08 F36000
G1 Z.68
G1 E.4 F1800
; LINE_WIDTH: 0.119866
G1 F3000
G1 X126.218 Y120.734 E.00055
; LINE_WIDTH: 0.151631
G1 X126.163 Y120.817 E.00076
; LINE_WIDTH: 0.189819
G1 X126.085 Y120.927 E.00136
; LINE_WIDTH: 0.233428
G1 X125.993 Y121.048 E.00197
; LINE_WIDTH: 0.276017
G1 X125.902 Y121.169 E.00239
; LINE_WIDTH: 0.308254
G1 X125.842 Y121.245 E.00173
; LINE_WIDTH: 0.341779
G3 X125.674 Y121.443 I-3.864 J-3.114 E.00521
; LINE_WIDTH: 0.38123
G1 X125.365 Y121.78 E.01033
; LINE_WIDTH: 0.421303
G1 X125.057 Y122.117 E.01152
; LINE_WIDTH: 0.461376
G1 X124.748 Y122.453 E.01271
; LINE_WIDTH: 0.509484
G3 X123.979 Y123.228 I-5.54 J-4.731 E.03385
; LINE_WIDTH: 0.479371
G1 X123.884 Y123.31 E.00364
; LINE_WIDTH: 0.452492
G1 X123.758 Y123.411 E.00439
; LINE_WIDTH: 0.423054
G1 X123.683 Y123.468 E.00239
; LINE_WIDTH: 0.388949
G1 X123.569 Y123.552 E.00328
; LINE_WIDTH: 0.34561
G1 X123.454 Y123.636 E.00288
; LINE_WIDTH: 0.30227
G1 X123.34 Y123.72 E.00248
; LINE_WIDTH: 0.258931
G1 X123.225 Y123.804 E.00208
; LINE_WIDTH: 0.219262
G1 X122.923 Y124.013 E.00443
G1 X115.681 Y126.989 F36000
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.627366
G1 F13279.262
G1 X119.249 Y123.42 E.19509
G2 X119.721 Y123.788 I2.603 J-2.854 E.02313
G1 X116.351 Y127.158 E.18423
G1 X116.832 Y127.515 E.02318
G1 X120.263 Y124.084 E.18757
G2 X120.885 Y124.301 I1.699 J-3.867 E.02548
G1 X117.314 Y127.872 E.19524
G1 X117.795 Y128.229 E.02318
G1 X121.57 Y124.454 E.20638
G2 X122.118 Y124.487 I.482 J-3.478 E.02124
G1 X122.401 Y124.462 E.011
G1 X118.277 Y128.586 E.2255
G3 X118.737 Y128.965 I-3.125 J4.272 E.02305
G1 X141.945 Y105.756 E1.26886
G1 X141.945 Y106.595 E.03242
G1 X119.161 Y129.38 E1.24572
G3 X119.559 Y129.82 I-1.549 J1.801 E.02302
G1 X141.945 Y107.434 E1.22394
G1 X141.945 Y108.272 E.03242
G1 X119.923 Y130.295 E1.20404
G3 X120.247 Y130.81 I-3.177 J2.358 E.02354
G1 X141.945 Y109.111 E1.18633
G1 X141.945 Y109.95 E.03242
M73 P43 R10
G1 X120.541 Y131.354 E1.17023
G3 X120.784 Y131.95 I-3.784 J1.887 E.0249
G1 X141.945 Y110.788 E1.15697
G1 X141.945 Y111.627 E.03242
G1 X120.992 Y132.581 E1.1456
G1 X121.145 Y133.266 E.02714
G1 X141.945 Y112.466 E1.1372
G1 X141.945 Y113.304 E.03242
G1 X121.235 Y134.015 E1.13232
G3 X121.256 Y134.833 I-6.627 J.581 E.03163
G1 X141.945 Y114.143 E1.13116
G1 X141.945 Y114.982 E.03242
G1 X121.151 Y135.777 E1.13691
G1 X121.11 Y135.984 E.00815
G1 X121.565 Y136.201 E.01948
G1 X141.945 Y115.82 E1.11424
G1 X141.945 Y116.659 E.03242
G1 X122.133 Y136.472 E1.0832
G1 X122.701 Y136.742 E.02432
G1 X141.945 Y117.498 E1.05216
G1 X141.945 Y118.336 E.03242
G1 X123.259 Y137.023 E1.02167
G1 X123.809 Y137.311 E.02403
G1 X141.945 Y119.175 E.99154
G1 X141.945 Y120.014 E.03242
G1 X124.36 Y137.599 E.96142
G3 X124.898 Y137.9 I-4.594 J8.839 E.02383
G1 X141.945 Y120.852 E.93202
G1 X141.945 Y121.691 E.03242
G1 X125.435 Y138.202 E.90267
G3 X125.969 Y138.507 I-4.671 J8.793 E.02377
G1 X141.945 Y122.53 E.87349
G1 X141.945 Y123.368 E.03242
G1 X126.492 Y138.822 E.8449
G1 X127.015 Y139.138 E.02362
G1 X141.945 Y124.207 E.8163
G1 X141.945 Y125.046 E.03242
G1 X127.53 Y139.461 E.78813
G1 X128.038 Y139.792 E.02343
G1 X141.945 Y125.885 E.76035
G1 X141.945 Y126.723 E.03242
G1 X128.546 Y140.122 E.73257
G3 X129.05 Y140.457 I-4.757 J7.702 E.02339
G1 X141.945 Y127.562 E.70503
G1 X141.945 Y128.401 E.03242
G1 X129.544 Y140.802 E.67804
G1 X130.037 Y141.147 E.02328
G1 X141.945 Y129.239 E.65106
G1 X141.945 Y130.078 E.03242
G1 X130.523 Y141.5 E.62448
G1 X131.004 Y141.858 E.02317
G1 X141.945 Y130.917 E.59822
G1 X141.945 Y131.755 E.03242
G1 X131.484 Y142.217 E.57195
G3 X131.956 Y142.583 I-7.051 J9.56 E.02311
G1 X141.945 Y132.594 E.54615
G1 X141.945 Y133.433 E.03242
G1 X132.426 Y142.952 E.52048
G1 X132.895 Y143.321 E.02309
G1 X141.945 Y134.271 E.4948
G1 X141.945 Y135.11 E.03242
G1 X133.353 Y143.703 E.46979
G1 X133.808 Y144.086 E.02301
G1 X141.945 Y135.949 E.44491
G1 X141.945 Y136.787 E.03242
G1 X134.263 Y144.47 E.42004
G1 X134.718 Y144.854 E.02301
G1 X141.945 Y137.626 E.39516
G1 X141.945 Y138.465 E.03242
G1 X135.161 Y145.249 E.37095
G1 X135.601 Y145.647 E.02296
G1 X141.945 Y139.303 E.34685
G1 X141.945 Y140.142 E.03242
G1 X136.034 Y146.053 E.32317
G1 X136.463 Y146.463 E.02293
G1 X141.945 Y140.981 E.29975
G1 X141.945 Y141.819 E.03242
G1 X136.891 Y146.873 E.27632
G3 X137.313 Y147.29 I-6.349 J6.853 E.02293
G1 X141.945 Y142.658 E.25325
G1 X141.945 Y143.497 E.03242
G1 X137.729 Y147.714 E.23055
G1 X138.144 Y148.137 E.02293
G1 X141.945 Y144.335 E.20785
G1 X141.945 Y145.174 E.03242
G1 X138.55 Y148.57 E.18566
G1 X138.954 Y149.005 E.02294
G1 X141.945 Y146.013 E.16358
G1 X141.945 Y146.851 E.03242
G1 X139.351 Y149.445 E.14183
G1 X139.745 Y149.891 E.02297
G1 X141.945 Y147.69 E.12031
G1 X141.945 Y148.529 E.03242
G1 X140.136 Y150.338 E.09895
G1 X140.52 Y150.793 E.02301
M73 P44 R10
G1 X141.945 Y149.367 E.07795
G1 X141.945 Y150.206 E.03242
G1 X140.899 Y151.252 E.0572
G1 X141.274 Y151.716 E.02305
G1 X141.945 Y151.045 E.03671
G1 X141.945 Y151.883 E.03242
G1 X141.457 Y152.371 E.02669
; CHANGE_LAYER
; Z_HEIGHT: 0.84
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13279.262
G1 X141.945 Y151.883 E-.2623
G1 X141.945 Y151.574 E-.1177
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L5
M991 S0 P4 ;notify layer change


G17
G3 Z1.08 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X124.543 Y117.962
G1 Z.84
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X124.669 Y118.094 E.00695
G1 X124.944 Y118.583 E.02145
G1 X125.064 Y119.092 E.01995
G1 X125.044 Y119.62 E.02019
G1 X124.904 Y120.08 E.01835
G1 X124.618 Y120.536 E.02055
G1 X123.422 Y121.963 E.07109
G1 X123.022 Y122.324 E.02059
G1 X122.549 Y122.559 E.02015
G1 X122.075 Y122.654 E.01846
G1 X121.578 Y122.628 E.01899
G1 X120.971 Y122.409 E.02464
G3 X120.103 Y121.724 I8.098 J-11.166 E.04224
G3 X114.848 Y126.662 I-40.656 J-37.999 E.27549
G1 X118.015 Y129.012 E.15054
G1 X118.416 Y129.344 E.01989
G1 X119.155 Y130.103 E.04046
G1 X119.749 Y130.931 E.03889
G1 X120.215 Y131.838 E.03893
G1 X120.538 Y132.787 E.03826
G3 X120.765 Y134.489 I-7.487 J1.862 E.06572
G1 X120.698 Y135.504 E.03882
G1 X120.538 Y136.268 E.0298
G3 X130.508 Y142.101 I-22.067 J49.147 E.44185
G3 X142.443 Y154.069 I-32.644 J44.493 E.64785
G1 X142.443 Y104.522 E1.89167
G1 X142.26 Y104.625 E.00803
G1 X141.659 Y104.72 E.02324
G1 X132.932 Y104.72 E.3332
G3 X132.304 Y104.616 I0 J-1.949 E.02442
G1 X131.884 Y104.415 E.01776
G1 X131.296 Y103.832 E.0316
G1 X131.259 Y103.76 E.00311
G3 X123.803 Y117.309 I-51.413 J-19.466 E.5924
G1 X124.376 Y117.789 E.02855
G1 X124.48 Y117.897 E.00573
G1 X124.11 Y118.352 F36000
G1 F13446.369
G1 X124.205 Y118.451 E.00522
G1 X124.407 Y118.821 E.01611
G1 X124.488 Y119.283 E.01791
G1 X124.437 Y119.654 E.01428
G1 X124.278 Y120.01 E.01489
G1 X124.169 Y120.16 E.00706
G1 X122.973 Y121.587 E.07109
G1 X122.645 Y121.87 E.01657
G1 X122.229 Y122.041 E.01717
G1 X121.789 Y122.067 E.01683
G3 X121.237 Y121.886 I.139 J-1.357 E.02235
G3 X120.051 Y120.917 I17.263 J-22.327 E.05849
G3 X113.893 Y126.683 I-40.448 J-37.022 E.32241
G1 X117.666 Y129.482 E.17935
G1 X118.108 Y129.851 E.02197
G1 X118.726 Y130.502 E.03429
G1 X119.266 Y131.262 E.03561
G1 X119.689 Y132.095 E.03565
G1 X119.98 Y132.964 E.035
G3 X120.115 Y135.453 I-6.555 J1.603 E.09573
G1 X119.931 Y136.289 E.03266
G1 X119.836 Y136.599 E.01238
G3 X130.931 Y143.156 I-21.119 J48.404 E.49331
G3 X142.92 Y155.769 I-32.174 J42.587 E.66735
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.012 E1.97526
; LINE_WIDTH: 0.634966
G1 F13111.234
G1 X143.021 Y103.266 E.02922
; LINE_WIDTH: 0.646396
G1 F12866.386
G1 X143.016 Y103.008 E.01027
G1 X142.944 Y103.252 E.01012
; LINE_WIDTH: 0.634966
G1 F13111.234
G1 X142.557 Y103.798 E.0262
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.08 Y104.068 E.02094
G1 X141.659 Y104.135 E.01626
G1 X132.932 Y104.135 E.3332
G1 X132.492 Y104.062 E.01701
G1 X132.199 Y103.921 E.01242
G1 X131.788 Y103.513 E.02211
G1 F11947.84
G1 X131.596 Y103.048 E.01922
; LINE_WIDTH: 0.667752
G1 F10303.752
G1 X131.572 Y102.992 E.00251
; LINE_WIDTH: 0.715508
G1 F10113.129
G1 X131.548 Y102.936 E.0027
; LINE_WIDTH: 0.763264
G1 F9924.286
G1 X131.523 Y102.88 E.00289
; LINE_WIDTH: 0.81102
G1 F9737.222
G1 X131.499 Y102.824 E.00308
; LINE_WIDTH: 0.858776
G1 F9551.938
G1 X131.475 Y102.768 E.00327
; LINE_WIDTH: 0.906532
G1 F9028.934
G1 X131.451 Y102.713 E.00346
; LINE_WIDTH: 0.954288
G1 F8560.23
G1 X131.426 Y102.657 E.00365
; LINE_WIDTH: 1.00204
G1 F8137.787
G1 X131.402 Y102.601 E.00384
; LINE_WIDTH: 1.0498
G1 F7755.076
G1 X131.378 Y102.545 E.00403
; LINE_WIDTH: 1.09756
G1 F7406.745
G1 X131.354 Y102.489 E.00422
G1 X131.31 Y102.54 E.00466
; LINE_WIDTH: 1.0498
G1 F7755.076
G1 X131.266 Y102.591 E.00445
; LINE_WIDTH: 1.00204
G1 F8137.787
G1 X131.222 Y102.642 E.00424
; LINE_WIDTH: 0.954288
G1 F8560.23
G1 X131.178 Y102.693 E.00403
; LINE_WIDTH: 0.906532
G1 F9028.934
G1 X131.134 Y102.744 E.00382
; LINE_WIDTH: 0.858776
G1 F9551.938
G1 X131.09 Y102.795 E.00362
; LINE_WIDTH: 0.81102
G1 F10139.259
G1 X131.046 Y102.845 E.00341
; LINE_WIDTH: 0.763264
G1 F10350.257
G1 X131.002 Y102.896 E.0032
; LINE_WIDTH: 0.715508
G1 F10563.403
G1 X130.958 Y102.947 E.00299
; LINE_WIDTH: 0.667752
G1 F10778.7
G1 X130.914 Y102.998 E.00278
; LINE_WIDTH: 0.619996
G1 F12103.804
G1 X130.772 Y103.372 E.01527
G1 F13446.369
G1 X130.63 Y103.746 E.01527
G1 X130.327 Y104.53 E.03209
G3 X122.999 Y117.399 I-50.328 J-20.138 E.56716
G1 X124 Y118.238 E.04986
G1 X124.048 Y118.287 E.00262
G1 X123.69 Y118.768 F36000
G1 F13446.369
G1 X123.809 Y118.913 E.00717
G1 X123.899 Y119.207 E.01171
G1 X123.873 Y119.495 E.01104
G1 X123.721 Y119.783 E.01246
G1 X122.524 Y121.21 E.07109
G1 X122.281 Y121.404 E.01186
G1 X121.987 Y121.486 E.01169
G1 X121.717 Y121.46 E.01035
G1 X121.428 Y121.307 E.01249
G1 X119.978 Y120.092 E.07222
G1 X119.604 Y120.538 E.02221
G3 X112.93 Y126.698 I-39.315 J-35.902 E.3472
G1 X117.325 Y129.958 E.20894
G1 X117.73 Y130.299 E.02021
G1 X118.297 Y130.9 E.03155
G1 X118.783 Y131.594 E.03233
G1 X119.162 Y132.352 E.03237
G3 X119.531 Y135.403 I-5.633 J2.229 E.11863
G1 X119.361 Y136.154 E.02939
G1 X119.08 Y136.91 E.03082
G3 X130.578 Y143.623 I-20.652 J48.576 E.50968
G3 X142.648 Y156.415 I-31.779 J42.074 E.67461
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.506 E2.16588
G3 X141.854 Y99.548 I-1.287 J-16.894 E.06726
G1 X142.426 Y102.636 E.11989
G3 X142.388 Y103.044 I-.767 J.135 E.01581
G1 X142.171 Y103.357 E.01454
G1 X141.899 Y103.511 E.01195
G1 X141.659 Y103.549 E.00928
G1 X132.932 Y103.549 E.3332
G3 X132.514 Y103.427 I0 J-.778 E.01686
G1 X132.27 Y103.18 E.01324
G1 X132.17 Y102.929 E.01034
G1 X132.175 Y102.581 E.01329
G1 X132.737 Y99.546 E.11784
G3 X131.452 Y99.491 I.974 J-37.77 E.04913
G3 X124.03 Y114.993 I-51.776 J-15.259 E.65901
G3 X122.174 Y117.472 I-28.946 J-19.742 E.11825
G1 X123.624 Y118.687 E.07222
G1 X123.633 Y118.698 E.00057
M204 S250
G1 X123.269 Y119.11 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X123.348 Y119.261 E.0054
G1 X123.297 Y119.428 E.00554
G1 X122.101 Y120.855 E.05895
G1 X121.945 Y120.935 E.00555
G1 X121.783 Y120.883 E.00539
G1 X120.779 Y120.042 E.04147
; LINE_WIDTH: 0.523196
G1 X120.119 Y119.493 E.02737
; LINE_WIDTH: 0.544336
G1 X120.047 Y119.191 E.01032
G1 X119.181 Y120.182 E.04378
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.854 J-35.521 E.30731
G1 X116.99 Y130.398 E.19631
G1 X117.374 Y130.721 E.01591
G1 X117.892 Y131.276 E.02402
G1 X118.327 Y131.906 E.02425
G1 X118.665 Y132.594 E.02427
G3 X118.98 Y135.355 I-5.169 J1.988 E.08893
G1 X118.824 Y136.026 E.0218
G1 X118.548 Y136.752 E.0246
G1 X118.31 Y137.19 E.01577
G3 X130.248 Y144.067 I-19.579 J47.789 E.43749
G3 X142.39 Y157.025 I-31.644 J41.818 E.56495
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.231 E1.81931
G1 X144.167 Y98.937 E.00931
G2 X142.832 Y98.994 I-.064 J14.109 E.04234
G3 X141.19 Y98.996 I-.862 J-45.509 E.05199
G1 X141.881 Y102.731 E.12028
G1 X141.87 Y102.85 E.00376
G1 X141.729 Y102.985 E.00621
G1 X132.932 Y102.996 E.27851
G1 X132.811 Y102.961 E.004
G1 X132.711 Y102.816 E.00556
G3 X133.037 Y100.962 I61.223 J9.819 E.05962
G1 X133.401 Y98.995 E.06332
G1 X132.441 Y98.992 E.03041
G2 X131.038 Y98.937 I-1.212 J12.933 E.04448
G3 X122.131 Y116.662 I-50.886 J-14.468 E.6318
; LINE_WIDTH: 0.521596
G1 X121.574 Y117.33 E.02762
; LINE_WIDTH: 0.544336
G1 X121.27 Y117.729 E.01669
G1 X121.608 Y117.716 E.01125
; LINE_WIDTH: 0.521596
G1 X122.264 Y118.268 E.02725
; LINE_WIDTH: 0.519996
G1 X123.2 Y119.053 E.03865
; WIPE_START
M204 S10000
G1 X123.348 Y119.261 E-.09731
G1 X123.297 Y119.428 E-.06648
G1 X122.931 Y119.864 E-.21621
; WIPE_END
G1 E-.02 F1800
G1 X128.778 Y114.958 Z1.24 F36000
G1 X143.016 Y103.008 Z1.24
G1 Z.84
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.648756
G1 F12816.967
G1 X143.015 Y102.519 E.01959
; LINE_WIDTH: 0.681936
G1 F12160.285
G1 X142.998 Y102.339 E.00766
; LINE_WIDTH: 0.727651
G1 F11358.474
G1 X142.975 Y102.09 E.0113
; LINE_WIDTH: 0.773366
G1 F10655.86
G1 X142.952 Y101.841 E.01204
; LINE_WIDTH: 0.819081
G1 F10035.107
G1 X142.929 Y101.592 E.01279
; LINE_WIDTH: 0.864796
G1 F9482.698
G1 X142.906 Y101.343 E.01353
; LINE_WIDTH: 0.910511
G1 F8987.931
G1 X142.884 Y101.094 E.01428
; LINE_WIDTH: 0.956226
G1 F8542.234
G1 X142.861 Y100.845 E.01502
; LINE_WIDTH: 1.00194
G1 F8138.651
G1 X142.838 Y100.596 E.01577
; LINE_WIDTH: 1.04766
G1 F7771.484
G1 X142.815 Y100.347 E.01651
; WIPE_START
G1 X142.838 Y100.596 E-.095
G1 X142.861 Y100.845 E-.095
G1 X142.884 Y101.094 E-.095
G1 X142.906 Y101.343 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.311 Y102.096 Z1.24 F36000
G1 X131.354 Y102.489 Z1.24
G1 Z.84
G1 E.4 F1800
; LINE_WIDTH: 1.09756
G1 F7406.745
G1 X131.369 Y102.431 E.00415
; LINE_WIDTH: 1.08786
G1 F7474.941
G1 X131.454 Y102.107 E.02301
; LINE_WIDTH: 1.03928
G1 F7836.237
G1 X131.539 Y101.783 E.02195
; LINE_WIDTH: 0.99071
G1 F8234.232
G1 X131.624 Y101.459 E.02089
; LINE_WIDTH: 0.942136
G1 F8674.819
G1 X131.702 Y101.144 E.01917
; LINE_WIDTH: 0.901986
G1 F9076.241
G1 X131.781 Y100.83 E.01833
; LINE_WIDTH: 0.861836
G1 F9516.617
G1 X131.859 Y100.516 E.01748
; LINE_WIDTH: 0.821686
G1 F10001.905
G1 X131.938 Y100.202 E.01663
; LINE_WIDTH: 0.781536
G1 F10539.348
G1 X131.943 Y100.183 E.00094
; WIPE_START
G1 X131.938 Y100.202 E-.00734
G1 X131.859 Y100.516 E-.12313
G1 X131.781 Y100.83 E-.12312
G1 X131.702 Y101.144 E-.12313
G1 X131.7 Y101.153 E-.00328
; WIPE_END
G1 E-.02 F1800
G1 X127.635 Y107.613 Z1.24 F36000
G1 X121.27 Y117.729 Z1.24
G1 Z.84
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X120.047 Y119.191 E.06335
; WIPE_START
M204 S10000
G1 X120.689 Y118.424 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.557 Y124.841 Z1.24 F36000
G1 X115.586 Y126.349 Z1.24
G1 Z.84
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.626686
G1 F13294.505
G1 X117.792 Y128.555 E.12049
; WIPE_START
G1 X117.085 Y127.848 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.696 Y134.573 Z1.24 F36000
G1 X120.97 Y135.083 Z1.24
G1 Z.84
G1 E.4 F1800
G1 F13294.505
G1 X122.563 Y136.677 E.08703
G3 X124.282 Y137.558 I-11.388 J24.333 E.07459
G1 X121.261 Y134.537 E.16497
G1 X121.21 Y133.648 E.0344
G1 X126.22 Y138.658 E.27361
G3 X128.477 Y140.078 I-19.979 J34.271 E.103
G1 X120.998 Y132.598 E.40847
G2 X120.471 Y131.234 I-5.665 J1.403 E.05662
G1 X131.365 Y142.128 E.59495
G3 X141.945 Y152.573 I-32.7 J43.703 E.57589
G1 X141.945 Y151.87 E.02714
G1 X116.225 Y126.15 E1.40458
G2 X116.669 Y125.757 I-5.659 J-6.832 E.02292
G1 X141.945 Y151.033 E1.38032
G1 X141.945 Y150.195 E.03235
G1 X117.108 Y125.357 E1.35639
G1 X117.546 Y124.958 E.0229
G1 X141.945 Y149.357 E1.33246
G1 X141.945 Y148.52 E.03235
G1 X117.984 Y124.558 E1.30852
G2 X118.414 Y124.15 I-6.44 J-7.209 E.02288
G1 X141.945 Y147.682 E1.28506
G1 X141.945 Y146.844 E.03235
G1 X118.839 Y123.738 E1.26183
G2 X119.255 Y123.316 I-7.551 J-7.839 E.02288
G1 X141.945 Y146.006 E1.23915
G1 X141.945 Y145.169 E.03235
G1 X119.669 Y122.892 E1.21655
G1 X120.083 Y122.468 E.02288
G1 X141.945 Y144.331 E1.19394
G1 X141.945 Y143.493 E.03235
G1 X121.579 Y123.127 E1.11221
G2 X122.398 Y123.109 I.348 J-2.741 E.03176
G1 X141.945 Y142.656 E1.06747
G1 X141.945 Y141.818 E.03235
G1 X123.025 Y122.898 E1.03324
G2 X123.526 Y122.561 I-2.885 J-4.832 E.02332
G1 X141.945 Y140.98 E1.00589
G1 X141.945 Y140.142 E.03235
G1 X123.932 Y122.129 E.98373
G1 X124.313 Y121.672 E.02297
G1 X141.945 Y139.305 E.96292
G1 X141.945 Y138.467 E.03235
G1 X124.694 Y121.215 E.94212
G1 X125.075 Y120.759 E.02297
G1 X141.945 Y137.629 E.92132
G1 X141.945 Y136.792 E.03235
G1 X125.379 Y120.226 E.90467
G2 X125.545 Y119.554 I-1.787 J-.798 E.02686
G1 X141.945 Y135.954 E.89561
G1 X141.945 Y135.116 E.03235
G1 X125.482 Y118.653 E.89906
G2 X125.277 Y118.119 I-1.928 J.433 E.02216
G2 X124.483 Y117.229 I-2.705 J1.614 E.04634
G1 X124.657 Y116.99 E.0114
G1 X141.945 Y134.279 E.94411
G1 X141.945 Y133.441 E.03235
G1 X125.01 Y116.506 E.92482
G2 X125.355 Y116.012 I-10.998 J-8.038 E.02324
G1 X141.945 Y132.603 E.90603
G1 X141.945 Y131.765 E.03235
G1 X125.698 Y115.518 E.88726
G2 X126.035 Y115.017 I-11.716 J-8.227 E.02331
M73 P45 R10
G1 X141.945 Y130.928 E.86889
G1 X141.945 Y130.09 E.03235
G1 X126.369 Y114.513 E.85065
G1 X126.694 Y114 E.02344
G1 X141.945 Y129.252 E.83291
G1 X141.945 Y128.415 E.03235
G1 X127.018 Y113.487 E.81517
G2 X127.332 Y112.964 I-8.32 J-5.343 E.02358
G1 X141.945 Y127.577 E.79803
G1 X141.945 Y126.739 E.03235
G1 X127.643 Y112.437 E.78104
G1 X127.954 Y111.91 E.02362
G1 X141.945 Y125.901 E.76405
G1 X141.945 Y125.064 E.03235
G1 X128.257 Y111.376 E.74751
G1 X128.551 Y110.831 E.02388
G1 X141.945 Y124.226 E.73149
G1 X141.945 Y123.388 E.03235
G1 X128.844 Y110.287 E.71547
G1 X129.137 Y109.742 E.02388
G1 X141.945 Y122.551 E.69945
G1 X141.945 Y121.713 E.03235
G1 X129.42 Y109.187 E.68404
G1 X129.697 Y108.627 E.02414
G1 X141.945 Y120.875 E.66888
G1 X141.945 Y120.037 E.03235
G1 X129.968 Y108.06 E.65411
G1 X130.229 Y107.484 E.02443
G1 X141.945 Y119.2 E.63981
G1 X141.945 Y118.362 E.03235
G1 X130.491 Y106.908 E.62552
G1 X130.753 Y106.332 E.02443
G1 X141.945 Y117.524 E.61122
G1 X141.945 Y116.687 E.03235
G1 X130.999 Y105.741 E.59777
G1 X131.242 Y105.145 E.02482
G1 X141.945 Y115.849 E.58453
G1 X141.945 Y115.011 E.03235
G1 X131.913 Y104.979 E.54788
G1 X132.14 Y105.088 E.00972
G2 X132.99 Y105.218 I.902 J-3.044 E.03332
G1 X141.945 Y114.174 E.48906
G1 X141.945 Y113.336 E.03235
G1 X133.828 Y105.218 E.44331
G1 X134.665 Y105.218 E.03235
G1 X141.945 Y112.498 E.39756
G1 X141.945 Y111.66 E.03235
G1 X135.503 Y105.218 E.35182
G1 X136.341 Y105.218 E.03235
G1 X141.945 Y110.823 E.30607
G1 X141.945 Y109.985 E.03235
G1 X137.179 Y105.218 E.26032
G1 X138.016 Y105.218 E.03235
G1 X141.945 Y109.147 E.21457
G1 X141.945 Y108.31 E.03235
G1 X138.854 Y105.218 E.16883
G1 X139.692 Y105.218 E.03235
G1 X141.945 Y107.472 E.12308
G1 X141.945 Y106.634 E.03235
G1 X140.529 Y105.218 E.07733
G1 X141.367 Y105.218 E.03235
G1 X142.209 Y106.06 E.04598
; CHANGE_LAYER
; Z_HEIGHT: 1
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13294.505
G1 X141.502 Y105.353 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L6
M991 S0 P5 ;notify layer change


G17
G3 Z1.24 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.284 Y118.861
G1 Z1
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.287 Y119.245 E.01463
G1 X125.186 Y119.67 E.0167
G3 X124.48 Y120.7 I-3.234 J-1.459 E.04792
G1 X123.196 Y122.233 E.07636
G1 X122.832 Y122.569 E.01891
G1 X122.358 Y122.816 E.02039
G1 X121.898 Y122.92 E.018
G1 X121.414 Y122.909 E.01849
G1 X120.871 Y122.745 E.02167
G1 X120.449 Y122.475 E.01913
G1 X119.861 Y121.982 E.02927
G3 X114.848 Y126.662 I-42.75 J-40.769 E.26199
G1 X118.015 Y129.012 E.15054
G3 X118.983 Y129.902 I-4.284 J5.629 E.05029
G1 X119.567 Y130.648 E.03618
G1 X120.076 Y131.531 E.03891
G1 X120.45 Y132.48 E.03894
G3 X120.765 Y134.489 I-7.463 J2.196 E.07787
G1 X120.698 Y135.507 E.03893
G1 X120.538 Y136.268 E.02969
G3 X130.509 Y142.102 I-22.153 J49.293 E.44189
G3 X142.443 Y154.069 I-32.645 J44.492 E.64781
G1 X142.443 Y104.525 E1.89154
G1 X142.263 Y104.627 E.00792
G1 X141.662 Y104.722 E.02324
G1 X132.929 Y104.722 E.33339
G1 X132.641 Y104.701 E.01105
G1 X132.12 Y104.547 E.02073
G1 X131.681 Y104.27 E.01981
G1 X131.296 Y103.838 E.0221
G1 X131.26 Y103.757 E.00338
G3 X124.027 Y117.012 I-51.132 J-19.302 E.57839
G1 X124.603 Y117.519 E.0293
G1 X124.971 Y117.929 E.02104
G1 X125.129 Y118.214 E.01242
G1 X125.284 Y118.758 E.0216
G1 X125.284 Y118.771 E.00051
G1 X124.694 Y118.878 F36000
G1 F13446.369
G1 X124.713 Y119.072 E.00746
G1 X124.635 Y119.473 E.0156
G1 X124.453 Y119.816 E.01484
G3 X124.032 Y120.324 I-7.677 J-5.945 E.0252
G1 X122.747 Y121.857 E.07636
G1 X122.408 Y122.147 E.01702
G1 X122.061 Y122.296 E.01446
G1 X121.638 Y122.343 E.01624
G1 X121.313 Y122.288 E.01259
G1 X120.872 Y122.063 E.0189
G3 X119.812 Y121.177 I297.616 J-356.816 E.05276
G3 X113.893 Y126.683 I-40.273 J-37.355 E.30893
G1 X117.666 Y129.482 E.17935
G3 X118.562 Y130.31 I-3.899 J5.123 E.04665
G1 X119.1 Y131.002 E.03347
G1 X119.563 Y131.813 E.03566
G1 X119.901 Y132.684 E.03566
G3 X120.135 Y133.793 I-9.641 J2.615 E.04329
G1 X120.18 Y134.525 E.02801
G1 X120.114 Y135.456 E.03565
G1 X119.931 Y136.289 E.03256
G1 X119.836 Y136.599 E.01238
G3 X130.932 Y143.156 I-21.122 J48.409 E.49331
G3 X142.92 Y155.769 I-32.175 J42.588 E.66735
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.013 E1.97522
; LINE_WIDTH: 0.633546
G1 F13142.306
G1 X143.022 Y103.267 E.02912
; LINE_WIDTH: 0.643876
G1 F12919.58
G1 X143.017 Y103.01 E.01021
G1 X142.945 Y103.253 E.01007
; LINE_WIDTH: 0.633546
G1 F13142.306
G1 X142.559 Y103.8 E.02613
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.082 Y104.07 E.02094
G1 X141.662 Y104.137 E.01626
G1 X132.929 Y104.137 E.33339
G1 X132.727 Y104.122 E.00773
G1 X132.337 Y104.001 E.01561
G1 X132.056 Y103.82 E.01275
G1 F12942.786
G1 X131.771 Y103.493 E.01657
; LINE_WIDTH: 0.663671
G1 F11458.534
G1 X131.715 Y103.376 E.00535
; LINE_WIDTH: 0.707346
G1 F11030.26
G1 X131.659 Y103.258 E.00572
; LINE_WIDTH: 0.751021
G1 F10610.142
G1 X131.603 Y103.14 E.00609
; LINE_WIDTH: 0.794696
G1 F10198.153
G1 X131.546 Y103.023 E.00646
; LINE_WIDTH: 0.838371
G1 F9794.35
G1 X131.49 Y102.905 E.00683
; LINE_WIDTH: 0.882046
G1 F9289.734
G1 X131.434 Y102.787 E.0072
; LINE_WIDTH: 0.926228
G1 F8829.545
G1 X131.416 Y102.734 E.00328
; LINE_WIDTH: 0.97041
G1 F8412.797
G1 X131.398 Y102.68 E.00344
; LINE_WIDTH: 1.01459
G1 F8033.616
G1 X131.38 Y102.627 E.00361
; LINE_WIDTH: 1.05877
G1 F7687.143
G1 X131.362 Y102.573 E.00377
; LINE_WIDTH: 1.10296
G1 F7369.317
G1 X131.344 Y102.52 E.00393
G1 X131.305 Y102.564 E.00405
; LINE_WIDTH: 1.05877
G1 F7687.143
G1 X131.267 Y102.607 E.00389
; LINE_WIDTH: 1.01459
G1 F8033.616
G1 X131.228 Y102.651 E.00372
; LINE_WIDTH: 0.97041
G1 F8412.797
G1 X131.19 Y102.695 E.00355
; LINE_WIDTH: 0.926228
G1 F8829.545
G1 X131.151 Y102.738 E.00338
; LINE_WIDTH: 0.882046
G1 F9289.734
G1 X131.075 Y102.882 E.00902
; LINE_WIDTH: 0.838371
G1 F9794.35
G1 X130.998 Y103.027 E.00855
; LINE_WIDTH: 0.794696
G1 F10356.937
G1 X130.922 Y103.171 E.00809
; LINE_WIDTH: 0.751021
G1 F10877.809
G1 X130.846 Y103.315 E.00762
; LINE_WIDTH: 0.707346
G1 F11411.476
G1 X130.769 Y103.459 E.00716
; LINE_WIDTH: 0.663671
G1 F11957.876
G1 X130.693 Y103.603 E.00669
; LINE_WIDTH: 0.619996
G1 F13351.536
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.208 Y104.834 E.01998
G3 X123.213 Y117.118 I-50.366 J-20.546 E.54121
G1 X124.227 Y117.968 E.05049
G1 X124.484 Y118.255 E.01472
G1 X124.674 Y118.683 E.01789
G1 X124.685 Y118.788 E.00403
G1 X124.109 Y118.92 F36000
G1 F13446.369
G1 X124.129 Y119.013 E.00364
G1 X124.062 Y119.328 E.01231
G1 X123.947 Y119.513 E.00831
G1 X122.298 Y121.481 E.09801
G1 X122.077 Y121.662 E.01093
G1 X121.78 Y121.755 E.01187
G1 X121.48 Y121.727 E.01152
G1 X121.201 Y121.577 E.01207
G1 X119.752 Y120.362 E.07222
G3 X112.93 Y126.698 I-39.384 J-35.565 E.35592
G1 X117.317 Y129.952 E.20855
G3 X118.142 Y130.717 I-3.512 J4.614 E.04301
G1 X118.633 Y131.356 E.03077
G1 X119.049 Y132.096 E.0324
G1 X119.352 Y132.887 E.03237
G3 X119.551 Y133.828 I-11.788 J2.981 E.03672
G1 X119.596 Y134.56 E.02801
G1 X119.531 Y135.406 E.03236
G1 X119.361 Y136.154 E.02928
G1 X119.08 Y136.91 E.03082
G3 X130.578 Y143.623 I-20.58 J48.453 E.50969
G3 X142.648 Y156.415 I-31.78 J42.074 E.67461
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.509 E2.16577
G1 X143.067 Y99.543 E.02094
G1 X141.856 Y99.548 E.04623
G1 X142.428 Y102.638 E.11997
G3 X142.391 Y103.046 I-.766 J.135 E.01581
G1 X142.174 Y103.359 E.01454
G1 X141.902 Y103.513 E.01195
G1 X141.662 Y103.551 E.00928
G1 X132.929 Y103.551 E.33339
G1 X132.591 Y103.474 E.01324
G3 X132.268 Y103.184 I.338 J-.701 E.01679
G1 X132.151 Y102.763 E.01666
G1 X132.164 Y102.631 E.00509
G1 X132.735 Y99.548 E.11971
G3 X131.452 Y99.491 I.957 J-35.997 E.04904
G3 X122.401 Y117.201 I-51.462 J-15.131 E.76378
G1 X123.851 Y118.416 E.07222
G1 X124.061 Y118.694 E.01328
G1 X124.09 Y118.832 E.00539
M204 S250
G1 X123.575 Y118.989 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X123.523 Y119.158 E.00561
G1 X121.871 Y121.129 E.08144
G1 X121.71 Y121.206 E.00564
G1 X121.556 Y121.153 E.00514
G1 X120.282 Y120.085 E.05263
; LINE_WIDTH: 0.523196
G1 X119.892 Y119.763 E.01613
; LINE_WIDTH: 0.544336
G1 X119.821 Y119.461 E.01032
G1 X119.181 Y120.182 E.03206
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-40.285 J-37.094 E.30728
G1 X116.988 Y130.396 E.19622
G3 X117.745 Y131.102 I-3.141 J4.13 E.03282
G1 X118.193 Y131.69 E.0234
G1 X118.565 Y132.362 E.02432
G1 X118.834 Y133.08 E.02427
G3 X118.999 Y133.862 I-26.564 J6.017 E.02531
G1 X119.044 Y134.599 E.02337
G1 X118.98 Y135.358 E.02412
G1 X118.824 Y136.026 E.02171
G1 X118.548 Y136.752 E.02461
G1 X118.31 Y137.19 E.01577
G3 X130.248 Y144.067 I-19.58 J47.79 E.43749
G3 X142.39 Y157.025 I-31.645 J41.819 E.56495
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.229 E1.81937
G1 X144.167 Y98.937 E.00924
G3 X141.192 Y98.997 I-2.225 J-36.431 E.09425
G1 X141.884 Y102.734 E.1203
G1 X141.873 Y102.852 E.00376
G1 X141.731 Y102.987 E.00621
G1 X141.662 Y102.998 E.00223
G1 X132.929 Y102.998 E.27646
G1 X132.785 Y102.946 E.00487
G1 X132.704 Y102.77 E.00614
G1 X132.707 Y102.731 E.00122
G1 X133.399 Y98.997 E.12024
G1 X132.441 Y98.994 E.03034
G2 X131.038 Y98.937 I-1.21 J12.464 E.04448
G3 X122.131 Y116.663 I-50.888 J-14.469 E.63181
; LINE_WIDTH: 0.521596
G1 X121.8 Y117.06 E.01641
; LINE_WIDTH: 0.544336
G1 X121.496 Y117.459 E.01669
G1 X121.834 Y117.446 E.01125
; LINE_WIDTH: 0.521596
G1 X122.221 Y117.771 E.01604
; LINE_WIDTH: 0.519996
G1 X123.495 Y118.84 E.05267
G1 X123.532 Y118.909 E.00248
; WIPE_START
M204 S10000
G1 X123.523 Y119.158 E-.09464
G1 X123.041 Y119.734 E-.28536
; WIPE_END
G1 E-.02 F1800
G1 X128.893 Y114.834 Z1.4 F36000
G1 X143.017 Y103.01 Z1.4
G1 Z1
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.646216
G1 F12870.172
G1 X143.016 Y102.522 E.01949
; LINE_WIDTH: 0.680386
G1 F12189.46
G1 X142.999 Y102.336 E.00787
; LINE_WIDTH: 0.726103
G1 F11383.903
G1 X142.976 Y102.087 E.01127
; LINE_WIDTH: 0.771819
G1 F10678.22
G1 X142.953 Y101.838 E.01202
; LINE_WIDTH: 0.817535
G1 F10054.92
G1 X142.93 Y101.589 E.01276
; LINE_WIDTH: 0.863251
G1 F9500.371
G1 X142.907 Y101.34 E.01351
; LINE_WIDTH: 0.908967
G1 F9003.795
G1 X142.884 Y101.091 E.01425
; LINE_WIDTH: 0.954684
G1 F8556.552
G1 X142.862 Y100.842 E.015
; LINE_WIDTH: 1.0004
G1 F8151.636
G1 X142.839 Y100.593 E.01574
; LINE_WIDTH: 1.04612
G1 F7783.313
G1 X142.816 Y100.344 E.01649
; WIPE_START
G1 X142.839 Y100.593 E-.095
G1 X142.862 Y100.842 E-.095
G1 X142.884 Y101.091 E-.095
G1 X142.907 Y101.34 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.314 Y102.115 Z1.4 F36000
G1 X131.344 Y102.52 Z1.4
G1 Z1
G1 E.4 F1800
; LINE_WIDTH: 1.10296
G1 F7369.317
G1 X131.354 Y102.481 E.00282
; LINE_WIDTH: 1.09646
G1 F7414.417
G1 X131.419 Y102.234 E.01765
; LINE_WIDTH: 1.05863
G1 F7688.215
G1 X131.483 Y101.988 E.01702
; LINE_WIDTH: 1.02081
G1 F7983.01
G1 X131.548 Y101.741 E.01639
; LINE_WIDTH: 0.982981
G1 F8301.314
G1 X131.613 Y101.495 E.01576
; LINE_WIDTH: 0.945156
G1 F8646.056
G1 X131.623 Y101.458 E.00223
; LINE_WIDTH: 0.940036
G1 F8694.933
G1 X131.701 Y101.144 E.01913
; LINE_WIDTH: 0.899896
G1 F9098.157
G1 X131.78 Y100.83 E.01828
; LINE_WIDTH: 0.859756
G1 F9540.598
G1 X131.858 Y100.516 E.01743
; LINE_WIDTH: 0.819616
G1 F10028.271
G1 X131.937 Y100.201 E.01659
; LINE_WIDTH: 0.779476
G1 F10568.485
G1 X131.942 Y100.183 E.00093
; WIPE_START
G1 X131.937 Y100.201 E-.00725
G1 X131.858 Y100.516 E-.12312
G1 X131.78 Y100.83 E-.12312
G1 X131.701 Y101.144 E-.12312
G1 X131.699 Y101.153 E-.00339
; WIPE_END
G1 E-.02 F1800
G1 X127.651 Y107.623 Z1.4 F36000
G1 X121.496 Y117.459 Z1.4
G1 Z1
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X119.821 Y119.461 E.0868
; WIPE_START
M204 S10000
G1 X120.462 Y118.694 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X119.962 Y122.714 Z1.4 F36000
G1 Z1
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X122.094 Y123.397 I1.858 J-2.131 E.08787
G3 X122.646 Y125.166 I-4.679 J2.429 E.07113
G3 X122.018 Y127.523 I-3.015 J.459 E.09578
G2 X119.552 Y129.81 I730.647 J790.216 E.12842
G3 X121.111 Y135.983 I-5.761 J4.738 E.25113
G3 X129.336 Y140.661 I-25.079 J53.674 E.36166
G2 X129.994 Y139.305 I-2.921 J-2.256 E.05794
G2 X129.882 Y136.477 I-3.803 J-1.264 E.11043
G2 X127.979 Y134.592 I-6.076 J4.23 E.10283
G3 X126.01 Y131.764 I4.035 J-4.91 E.13323
G3 X126.123 Y128.937 I3.803 J-1.264 E.11043
G3 X128.025 Y127.052 I6.076 J4.229 E.10283
G2 X130.186 Y123.281 I-3.506 J-4.514 E.1702
G2 X129.559 Y120.925 I-3.015 J-.459 E.09578
G3 X127.023 Y118.568 I394.535 J-427.175 E.13218
G3 X125.82 Y115.34 I3.889 J-3.287 E.13416
G2 X127.827 Y112.137 I-68.491 J-45.141 E.14435
G2 X128.982 Y111.028 I-6.337 J-7.754 E.06119
G2 X130.184 Y107.584 I-3.894 J-3.291 E.14241
G2 X131.425 Y104.694 I-35.049 J-16.761 E.12011
G2 X132.56 Y105.193 I1.459 J-1.777 E.04795
G2 X134.295 Y105.22 I1.048 J-11.652 E.06631
G2 X135.566 Y106.315 I5.913 J-5.579 E.06414
G3 X137.535 Y109.142 I-4.035 J4.91 E.13323
G3 X137.422 Y111.97 I-3.803 J1.264 E.11043
G3 X135.52 Y113.855 I-6.076 J-4.229 E.10283
G2 X133.55 Y116.683 I4.035 J4.91 E.13323
G2 X133.663 Y119.511 I3.803 J1.264 E.11043
G2 X135.566 Y121.396 I6.076 J-4.229 E.10283
G3 X137.535 Y124.224 I-4.035 J4.91 E.13323
G3 X137.422 Y127.052 I-3.803 J1.264 E.11043
G3 X135.52 Y128.937 I-6.076 J-4.229 E.10283
G2 X133.55 Y131.764 I4.035 J4.91 E.13323
G2 X133.663 Y134.592 I3.803 J1.264 E.11043
G2 X135.566 Y136.477 I6.076 J-4.229 E.10283
G3 X137.727 Y140.248 I-3.506 J4.514 E.1702
G3 X136.643 Y143.075 I-3.029 J.461 E.12093
G2 X134.688 Y144.838 I10.01 J13.071 E.10062
G3 X141.945 Y152.574 I-35.321 J40.411 E.40564
G1 X141.945 Y148.494 E.15575
G3 X140.899 Y145.903 I4.57 J-3.352 E.10779
G3 X141.945 Y143.114 I3.004 J-.464 E.11884
G1 X141.945 Y133.413 E.37039
G3 X140.899 Y130.822 I4.57 J-3.352 E.10779
G3 X141.945 Y128.033 I3.004 J-.464 E.11884
G1 X141.945 Y118.332 E.37039
G3 X140.899 Y115.741 I4.57 J-3.352 E.10779
G3 X141.945 Y112.952 I3.004 J-.464 E.11884
G1 X141.945 Y110.609 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 1.16
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y111.609 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L7
M991 S0 P6 ;notify layer change


G17
G3 Z1.4 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.447 Y118.647
G1 Z1.16
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.463 Y118.738 E.00352
M73 P46 R10
G1 X125.43 Y119.183 E.01707
G1 X125.3 Y119.601 E.01671
G1 X125.009 Y120.069 E.02103
G1 X123.027 Y122.434 E.1178
G1 X122.63 Y122.791 E.02039
G1 X122.23 Y122.999 E.01721
G1 X121.793 Y123.109 E.01719
G1 X121.394 Y123.121 E.01528
G1 X120.864 Y123.007 E.02067
G1 X120.593 Y122.883 E.01139
G3 X119.684 Y122.167 I4.734 J-6.954 E.0442
G3 X114.848 Y126.662 I-39.448 J-37.591 E.25223
G1 X118.015 Y129.012 E.15054
G1 X118.675 Y129.585 E.03339
G1 X119.183 Y130.134 E.02855
G1 X119.749 Y130.931 E.03734
G1 X120.215 Y131.838 E.03893
G1 X120.536 Y132.779 E.03795
G3 X120.765 Y134.489 I-7.543 J1.878 E.06601
G1 X120.698 Y135.504 E.03881
G1 X120.538 Y136.268 E.02983
G3 X130.508 Y142.101 I-21.912 J48.886 E.44185
G3 X142.443 Y154.069 I-32.646 J44.495 E.64786
G1 X142.443 Y104.529 E1.89141
G1 X142.265 Y104.629 E.00781
G1 X141.664 Y104.724 E.02324
G1 X132.927 Y104.724 E.33358
G1 X132.638 Y104.703 E.01106
G1 X132.117 Y104.548 E.02075
G1 X131.677 Y104.272 E.01983
G1 X131.293 Y103.838 E.02212
G1 X131.258 Y103.762 E.00318
G3 X124.167 Y116.819 I-51.44 J-19.483 E.56902
G1 X124.768 Y117.323 E.02994
G1 X125.151 Y117.756 E.02207
G1 X125.371 Y118.219 E.01954
G1 X125.431 Y118.558 E.01317
G1 X124.859 Y118.717 F36000
G1 F13446.369
G1 X124.878 Y118.866 E.00574
G1 X124.801 Y119.272 E.01577
G1 X124.618 Y119.619 E.015
G3 X123.864 Y120.524 I-13.541 J-10.521 E.04495
G1 X122.579 Y122.056 E.07636
G1 X122.302 Y122.306 E.01427
G1 X121.885 Y122.496 E.01748
G1 X121.437 Y122.537 E.01719
G1 X121.066 Y122.457 E.01446
G1 X120.661 Y122.222 E.0179
G1 X119.638 Y121.365 E.05097
G3 X113.893 Y126.683 I-40.778 J-38.283 E.29914
G1 X117.666 Y129.482 E.17935
G1 X118.288 Y130.024 E.03151
G1 X118.745 Y130.523 E.02585
G1 X119.266 Y131.263 E.03453
G1 X119.689 Y132.095 E.03564
G1 X119.978 Y132.957 E.0347
G1 X120.114 Y133.622 E.02593
G1 X120.18 Y134.525 E.03456
G1 X120.115 Y135.453 E.03553
G1 X119.932 Y136.284 E.03247
G1 X119.836 Y136.599 E.01258
G3 X130.932 Y143.157 I-21.325 J48.752 E.49333
G3 X142.92 Y155.769 I-32.174 J42.586 E.6673
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.014 E1.97518
; LINE_WIDTH: 0.632126
G1 F13173.523
G1 X143.023 Y103.269 E.02903
; LINE_WIDTH: 0.641356
G1 F12973.215
G1 X143.018 Y103.012 E.01016
G1 X142.947 Y103.255 E.01002
; LINE_WIDTH: 0.632126
G1 F13173.523
G1 X142.562 Y103.802 E.02606
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.085 Y104.072 E.02094
G1 X141.664 Y104.139 E.01626
G1 X132.927 Y104.139 E.33358
G1 X132.725 Y104.124 E.00774
G1 X132.334 Y104.003 E.01563
G1 X132.053 Y103.822 E.01276
G1 F12931.783
G1 X131.768 Y103.494 E.01658
; LINE_WIDTH: 0.663881
G1 F11446.802
G1 X131.712 Y103.376 E.00536
; LINE_WIDTH: 0.707766
G1 F11018.234
G1 X131.655 Y103.258 E.00573
; LINE_WIDTH: 0.751651
G1 F10597.829
G1 X131.599 Y103.141 E.0061
; LINE_WIDTH: 0.795536
G1 F10185.6
G1 X131.543 Y103.023 E.00648
; LINE_WIDTH: 0.839421
G1 F9781.576
G1 X131.487 Y102.905 E.00685
; LINE_WIDTH: 0.883306
G1 F9275.947
G1 X131.431 Y102.787 E.00722
; LINE_WIDTH: 0.9269
G1 F8822.898
G1 X131.413 Y102.734 E.00325
; LINE_WIDTH: 0.970494
G1 F8412.042
G1 X131.395 Y102.681 E.00341
; LINE_WIDTH: 1.01409
G1 F8037.748
G1 X131.378 Y102.628 E.00357
; LINE_WIDTH: 1.05768
G1 F7695.344
G1 X131.36 Y102.575 E.00373
; LINE_WIDTH: 1.10128
G1 F7380.921
G1 X131.342 Y102.522 E.00389
G1 X131.304 Y102.565 E.00401
; LINE_WIDTH: 1.05768
G1 F7695.344
G1 X131.266 Y102.609 E.00384
; LINE_WIDTH: 1.01409
G1 F8037.748
G1 X131.228 Y102.652 E.00368
; LINE_WIDTH: 0.970494
G1 F8412.042
G1 X131.19 Y102.695 E.00351
; LINE_WIDTH: 0.9269
G1 F8822.898
G1 X131.152 Y102.738 E.00335
; LINE_WIDTH: 0.883306
G1 F9275.947
G1 X131.075 Y102.883 E.00903
; LINE_WIDTH: 0.839421
G1 F9781.576
G1 X130.999 Y103.027 E.00856
; LINE_WIDTH: 0.795536
G1 F10345.508
G1 X130.922 Y103.171 E.0081
; LINE_WIDTH: 0.751651
G1 F10866.134
G1 X130.846 Y103.315 E.00763
; LINE_WIDTH: 0.707766
G1 F11399.54
G1 X130.769 Y103.459 E.00716
; LINE_WIDTH: 0.663881
G1 F11945.726
G1 X130.693 Y103.603 E.0067
; LINE_WIDTH: 0.619996
G1 F13338.698
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.208 Y104.834 E.01998
G3 X123.368 Y116.914 I-50.374 J-20.55 E.53143
G1 X124.392 Y117.772 E.05101
G1 X124.675 Y118.1 E.01654
G1 X124.822 Y118.426 E.01367
G1 X124.848 Y118.628 E.00777
G1 X124.28 Y118.752 F36000
G1 F13446.369
G1 X124.293 Y118.817 E.00254
G1 X124.228 Y119.13 E.01222
G1 X124.111 Y119.317 E.0084
G1 X122.132 Y121.679 E.11765
G1 X121.87 Y121.88 E.01262
G1 X121.574 Y121.954 E.01165
G1 X121.268 Y121.907 E.0118
G1 X120.923 Y121.678 E.01583
G1 X119.587 Y120.558 E.06657
G3 X112.93 Y126.698 I-40.531 J-37.263 E.34615
G1 X117.317 Y129.952 E.20855
G1 X117.901 Y130.463 E.02962
G1 X118.308 Y130.912 E.02314
G1 X118.783 Y131.594 E.03172
G1 X119.162 Y132.352 E.03236
G1 X119.42 Y133.134 E.03145
G1 X119.539 Y133.731 E.02323
G1 X119.596 Y134.56 E.03174
G1 X119.531 Y135.403 E.03225
G1 X119.363 Y136.149 E.0292
G1 X119.08 Y136.91 E.03101
G3 X130.579 Y143.624 I-20.429 J48.194 E.50975
G3 X142.647 Y156.415 I-31.779 J42.072 E.67456
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.51 E2.16574
G1 X143.067 Y99.545 E.02094
G1 X141.859 Y99.55 E.04614
G1 X142.431 Y102.64 E.11997
G3 X142.393 Y103.048 I-.766 J.135 E.01581
G1 X142.176 Y103.361 E.01454
G1 X141.904 Y103.515 E.01195
G1 X141.664 Y103.553 E.00928
G1 X132.927 Y103.553 E.33358
G1 X132.588 Y103.476 E.01326
G3 X132.265 Y103.185 I.338 J-.701 E.01681
G1 X132.148 Y102.764 E.01668
G1 X132.161 Y102.633 E.00505
G1 X132.732 Y99.55 E.11972
G3 X131.452 Y99.491 I.949 J-34.509 E.04894
G3 X122.572 Y116.995 I-51.747 J-15.245 E.75358
G2 X124.015 Y118.221 I43.419 J-49.678 E.07227
G1 X124.226 Y118.5 E.01335
G1 X124.261 Y118.664 E.00641
M204 S250
G1 X123.74 Y118.808 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X123.688 Y118.962 E.00516
G1 X121.707 Y121.325 E.09762
G1 X121.548 Y121.402 E.00561
G1 X121.392 Y121.35 E.0052
G1 X119.922 Y120.117 E.06074
; LINE_WIDTH: 0.523196
G1 X119.728 Y119.959 E.00797
; LINE_WIDTH: 0.544336
G1 X119.656 Y119.657 E.01032
G1 X119.181 Y120.182 E.02355
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-40.287 J-37.096 E.30728
G1 X116.988 Y130.396 E.19622
G1 X117.535 Y130.878 E.02309
G1 X117.895 Y131.28 E.01708
G1 X118.327 Y131.907 E.0241
G1 X118.665 Y132.594 E.02427
G1 X118.893 Y133.302 E.02354
G1 X118.996 Y133.834 E.01714
G1 X119.044 Y134.594 E.02412
G1 X118.98 Y135.355 E.02418
G1 X118.825 Y136.021 E.02165
G1 X118.548 Y136.752 E.02476
G1 X118.31 Y137.19 E.01577
G3 X130.249 Y144.067 I-19.958 J48.447 E.43749
G3 X142.39 Y157.025 I-31.645 J41.817 E.56491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.229 E1.81938
G1 X144.167 Y98.937 E.00923
G3 X141.194 Y98.999 I-2.222 J-35.183 E.09417
G1 X141.886 Y102.736 E.1203
G1 X141.875 Y102.854 E.00376
G1 X141.734 Y102.989 E.00621
G1 X141.664 Y103 E.00223
G1 X132.927 Y103 E.27662
G1 X132.782 Y102.948 E.00487
G1 X132.701 Y102.772 E.00614
G1 X132.705 Y102.734 E.00121
G1 X133.396 Y98.999 E.12024
G1 X132.441 Y98.996 E.03026
G2 X131.038 Y98.937 I-1.21 J12.043 E.04448
G3 X122.131 Y116.662 I-50.888 J-14.469 E.6318
; LINE_WIDTH: 0.544336
G1 X121.661 Y117.263 E.02536
G1 X121.999 Y117.25 E.01125
; LINE_WIDTH: 0.521596
G1 X122.855 Y117.969 E.03553
; LINE_WIDTH: 0.519996
G1 X123.66 Y118.644 E.03326
G1 X123.701 Y118.727 E.00293
; WIPE_START
M204 S10000
G1 X123.688 Y118.962 E-.08941
G1 X123.197 Y119.548 E-.29059
; WIPE_END
G1 E-.02 F1800
G1 X129.057 Y114.659 Z1.56 F36000
G1 X143.018 Y103.012 Z1.56
G1 Z1.16
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.643696
G1 F12923.396
G1 X143.017 Y102.524 E.01939
; LINE_WIDTH: 0.678136
G1 F12232.061
G1 X143 Y102.336 E.00791
; LINE_WIDTH: 0.723851
G1 F11421.072
G1 X142.977 Y102.088 E.01124
; LINE_WIDTH: 0.769566
G1 F10710.935
G1 X142.954 Y101.839 E.01198
; LINE_WIDTH: 0.815281
G1 F10083.937
G1 X142.931 Y101.59 E.01273
; LINE_WIDTH: 0.860996
G1 F9526.288
G1 X142.908 Y101.341 E.01347
; LINE_WIDTH: 0.906711
G1 F9027.082
G1 X142.886 Y101.092 E.01422
; LINE_WIDTH: 0.952426
G1 F8577.591
G1 X142.863 Y100.843 E.01496
; LINE_WIDTH: 0.998141
G1 F8170.741
G1 X142.84 Y100.594 E.01571
; LINE_WIDTH: 1.04386
G1 F7800.737
G1 X142.817 Y100.345 E.01645
; WIPE_START
G1 X142.84 Y100.594 E-.095
G1 X142.863 Y100.843 E-.095
G1 X142.886 Y101.092 E-.095
G1 X142.908 Y101.341 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.315 Y102.116 Z1.56 F36000
G1 X131.342 Y102.522 Z1.56
G1 Z1.16
G1 E.4 F1800
; LINE_WIDTH: 1.10128
G1 F7380.921
G1 X131.352 Y102.483 E.00281
; LINE_WIDTH: 1.09472
G1 F7426.583
G1 X131.417 Y102.236 E.01766
; LINE_WIDTH: 1.05682
G1 F7701.862
G1 X131.482 Y101.989 E.01702
; LINE_WIDTH: 1.01892
G1 F7998.335
G1 X131.547 Y101.742 E.01639
; LINE_WIDTH: 0.981016
G1 F8318.546
G1 X131.612 Y101.495 E.01576
; LINE_WIDTH: 0.943116
G1 F8665.464
G1 X131.621 Y101.459 E.00222
; LINE_WIDTH: 0.938036
G1 F8714.176
G1 X131.7 Y101.144 E.0191
; LINE_WIDTH: 0.897861
G1 F9119.598
G1 X131.779 Y100.83 E.01825
; LINE_WIDTH: 0.857686
G1 F9564.585
G1 X131.857 Y100.515 E.0174
; LINE_WIDTH: 0.817511
G1 F10055.225
G1 X131.936 Y100.201 E.01655
; LINE_WIDTH: 0.777336
G1 F10598.924
G1 X131.941 Y100.183 E.0009
; WIPE_START
G1 X131.936 Y100.201 E-.00706
G1 X131.857 Y100.515 E-.12319
G1 X131.779 Y100.83 E-.12318
G1 X131.7 Y101.144 E-.12318
G1 X131.698 Y101.153 E-.00338
; WIPE_END
G1 E-.02 F1800
G1 X127.662 Y107.631 Z1.56 F36000
G1 X121.661 Y117.263 Z1.56
G1 Z1.16
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X119.656 Y119.657 E.10381
; WIPE_START
M204 S10000
G1 X120.298 Y118.89 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X119.974 Y123.06 Z1.56 F36000
G1 Z1.16
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X122.177 Y123.531 I1.619 J-2.184 E.08869
G3 X122.732 Y125.166 I-4.264 J2.36 E.06628
G3 X121.788 Y127.994 I-2.817 J.631 E.11961
G2 X119.586 Y129.844 I10.843 J15.141 E.1099
G3 X121.113 Y135.986 I-5.758 J4.692 E.24961
G3 X129.315 Y140.642 I-25.294 J54.102 E.36046
G2 X130.273 Y138.362 I-4.724 J-3.327 E.09514
G2 X129.787 Y136.006 I-2.88 J-.634 E.09462
G2 X128.13 Y134.592 I-6.133 J5.509 E.0834
G3 X125.731 Y130.822 I3.685 J-4.992 E.17453
G3 X126.218 Y128.465 I2.88 J-.634 E.09462
G3 X127.874 Y127.052 I6.133 J5.509 E.0834
G2 X130.273 Y123.281 I-3.685 J-4.992 E.17453
G2 X129.787 Y120.925 I-2.88 J-.634 E.09462
G2 X128.13 Y119.511 I-6.133 J5.509 E.0834
G3 X125.717 Y115.489 I3.363 J-4.752 E.1843
G2 X127.969 Y111.891 I-61.032 J-40.701 E.16207
G2 X130.325 Y107.274 I-3.36 J-4.624 E.2054
G2 X131.424 Y104.697 I-31.305 J-14.87 E.10698
G1 X131.91 Y105.003 E.02194
G2 X134.071 Y105.222 I1.585 J-4.868 E.08356
G1 X135.415 Y106.315 E.06613
G3 X137.814 Y110.085 I-3.685 J4.992 E.17453
G3 X137.327 Y112.442 I-2.88 J.634 E.09462
G3 X135.671 Y113.855 I-6.133 J-5.508 E.0834
G2 X133.272 Y117.626 I3.685 J4.992 E.17453
G2 X133.758 Y119.982 I2.88 J.634 E.09462
G2 X135.415 Y121.396 I6.133 J-5.508 E.0834
G3 X137.814 Y125.166 I-3.685 J4.992 E.17453
G3 X137.327 Y127.523 I-2.88 J.634 E.09462
G3 X135.671 Y128.937 I-6.133 J-5.508 E.0834
G2 X133.272 Y132.707 I3.685 J4.992 E.17453
G2 X133.758 Y135.063 I2.88 J.634 E.09462
G2 X135.415 Y136.477 I6.133 J-5.508 E.0834
G3 X137.814 Y140.248 I-3.685 J4.992 E.17453
G3 X136.869 Y143.075 I-2.817 J.631 E.11961
G2 X134.723 Y144.869 I10.943 J15.267 E.10688
G3 X141.945 Y152.574 I-35.33 J40.356 E.40383
G1 X141.945 Y148.421 E.15854
G3 X140.813 Y145.903 I4.587 J-3.577 E.10643
G3 X141.945 Y142.927 I2.818 J-.631 E.12876
G1 X141.945 Y133.34 E.36604
G3 X140.813 Y130.822 I4.587 J-3.577 E.10643
G3 X141.945 Y127.846 I2.818 J-.631 E.12876
G1 X141.945 Y118.259 E.36605
G3 X140.813 Y115.741 I4.587 J-3.577 E.10643
G3 X141.945 Y112.765 I2.818 J-.631 E.12876
G1 X141.945 Y110.422 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 1.32
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y111.422 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L8
M991 S0 P7 ;notify layer change


G17
G3 Z1.56 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.576 Y118.483
G1 Z1.32
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.593 Y118.571 E.00344
G1 X125.561 Y119.021 E.0172
G1 X125.432 Y119.442 E.01685
G1 X125.139 Y119.914 E.0212
G1 X122.901 Y122.584 E.13301
G1 X122.523 Y122.93 E.01956
G1 X122.125 Y123.144 E.01725
G1 X121.648 Y123.266 E.01881
G1 X121.166 Y123.267 E.01842
G1 X120.919 Y123.219 E.0096
G1 X120.485 Y123.05 E.01781
G3 X119.543 Y122.314 I5.992 J-8.635 E.04563
G3 X114.848 Y126.662 I-40.088 J-38.578 E.24447
G1 X118.031 Y129.023 E.1513
G1 X118.485 Y129.403 E.0226
G1 X119.157 Y130.105 E.03712
G1 X119.749 Y130.932 E.03881
G1 X120.215 Y131.838 E.03892
G1 X120.535 Y132.776 E.03781
G3 X120.765 Y134.489 I-7.532 J1.881 E.06615
G1 X120.698 Y135.504 E.03882
G1 X120.538 Y136.268 E.02982
G3 X130.507 Y142.101 I-22.053 J49.127 E.44183
G3 X142.443 Y154.069 I-32.646 J44.495 E.64786
G1 X142.443 Y104.532 E1.89127
G1 X142.268 Y104.631 E.0077
G1 X141.667 Y104.727 E.02324
G1 X132.924 Y104.727 E.33377
G1 X132.635 Y104.705 E.01109
G1 X132.113 Y104.55 E.02076
G1 X131.674 Y104.273 E.01985
G1 X131.289 Y103.839 E.02214
G1 X131.256 Y103.768 E.00298
G3 X124.286 Y116.655 I-52.209 J-19.907 E.561
G1 X124.897 Y117.168 E.03044
G1 X125.299 Y117.629 E.02337
G1 X125.492 Y118.036 E.01719
G1 X125.559 Y118.394 E.0139
G1 X124.994 Y118.565 F36000
G1 F13446.369
G1 X125.008 Y118.704 E.00531
G1 X124.932 Y119.113 E.01589
G1 X124.748 Y119.464 E.01512
G1 X124.69 Y119.538 E.0036
G1 X122.444 Y122.218 E.13353
G1 X122.188 Y122.45 E.01317
G1 X121.772 Y122.646 E.01755
G1 X121.324 Y122.693 E.01722
G3 X120.762 Y122.534 I.083 J-1.362 E.02248
G3 X119.5 Y121.513 I10.053 J-13.721 E.06199
G3 X113.893 Y126.683 I-40.207 J-37.978 E.29141
G1 X117.677 Y129.49 E.17988
G1 X118.108 Y129.851 E.02145
G1 X118.728 Y130.504 E.03438
G1 X119.266 Y131.263 E.03553
G1 X119.689 Y132.095 E.03564
G1 X119.977 Y132.953 E.03457
G1 X120.114 Y133.621 E.02603
G1 X120.18 Y134.525 E.03459
G1 X120.115 Y135.453 E.03554
G1 X119.933 Y136.284 E.03245
G1 X119.836 Y136.599 E.01259
G3 X130.932 Y143.157 I-21.132 J48.426 E.49335
G3 X142.92 Y155.769 I-32.176 J42.588 E.6673
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.015 E1.97515
; LINE_WIDTH: 0.630696
G1 F13205.113
G1 X143.024 Y103.271 E.02893
; LINE_WIDTH: 0.638816
G1 F13027.727
G1 X143.019 Y103.014 E.01011
G1 X142.949 Y103.257 E.00997
; LINE_WIDTH: 0.630696
G1 F13205.113
G1 X142.564 Y103.804 E.02599
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.087 Y104.074 E.02094
G1 X141.667 Y104.141 E.01626
G1 X132.924 Y104.141 E.33377
G1 X132.722 Y104.126 E.00776
G1 X132.33 Y104.005 E.01563
G1 X132.049 Y103.823 E.01278
G1 F12921.581
G1 X131.765 Y103.495 E.0166
; LINE_WIDTH: 0.66408
G1 F11435.868
G1 X131.708 Y103.377 E.00536
; LINE_WIDTH: 0.708163
G1 F11007.007
G1 X131.652 Y103.259 E.00574
; LINE_WIDTH: 0.752246
G1 F10586.327
G1 X131.596 Y103.141 E.00612
; LINE_WIDTH: 0.79633
G1 F10173.843
G1 X131.54 Y103.023 E.00649
; LINE_WIDTH: 0.840413
G1 F9769.543
G1 X131.484 Y102.905 E.00687
; LINE_WIDTH: 0.884496
G1 F9262.963
G1 X131.428 Y102.787 E.00724
; LINE_WIDTH: 0.927508
G1 F8816.891
G1 X131.41 Y102.734 E.00322
; LINE_WIDTH: 0.97052
G1 F8411.808
G1 X131.393 Y102.682 E.00338
; LINE_WIDTH: 1.01353
G1 F8042.312
G1 X131.375 Y102.629 E.00353
; LINE_WIDTH: 1.05654
G1 F7703.911
G1 X131.358 Y102.577 E.00369
; LINE_WIDTH: 1.09956
G1 F7392.839
G1 X131.34 Y102.524 E.00384
G1 X131.303 Y102.567 E.00396
; LINE_WIDTH: 1.05654
G1 F7703.911
G1 X131.265 Y102.61 E.0038
; LINE_WIDTH: 1.01353
G1 F8042.312
G1 X131.228 Y102.653 E.00364
; LINE_WIDTH: 0.97052
G1 F8411.808
G1 X131.19 Y102.696 E.00348
; LINE_WIDTH: 0.927508
G1 F8816.891
G1 X131.152 Y102.739 E.00332
; LINE_WIDTH: 0.884496
G1 F9262.963
G1 X131.076 Y102.883 E.00904
; LINE_WIDTH: 0.840413
G1 F9769.543
G1 X130.999 Y103.027 E.00857
; LINE_WIDTH: 0.79633
G1 F10334.736
G1 X130.923 Y103.171 E.00811
; LINE_WIDTH: 0.752246
G1 F10855.145
G1 X130.846 Y103.315 E.00764
; LINE_WIDTH: 0.708163
G1 F11388.323
G1 X130.769 Y103.459 E.00717
; LINE_WIDTH: 0.66408
G1 F11934.267
G1 X130.693 Y103.603 E.0067
; LINE_WIDTH: 0.619996
G1 F13326.588
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.208 Y104.835 E.01998
G3 X123.49 Y116.752 I-50.283 J-20.495 E.5237
G1 X124.521 Y117.616 E.05138
G1 X124.817 Y117.965 E.01745
G1 X124.972 Y118.344 E.01566
G1 X124.985 Y118.476 E.00505
G1 X124.413 Y118.634 F36000
G1 F13446.369
G1 X124.397 Y118.863 E.00875
G1 X124.241 Y119.162 E.01286
G1 X121.999 Y121.838 E.1333
G1 X121.75 Y122.031 E.01202
G1 X121.503 Y122.104 E.00982
G1 X121.212 Y122.085 E.01114
G1 X120.907 Y121.928 E.01311
G1 X119.449 Y120.706 E.07262
G3 X112.93 Y126.698 I-39.899 J-36.873 E.33843
G1 X117.323 Y129.957 E.20886
G1 X117.731 Y130.299 E.0203
G1 X118.298 Y130.902 E.03163
G1 X118.783 Y131.594 E.03225
G1 X119.162 Y132.352 E.03236
G1 X119.419 Y133.131 E.03132
G1 X119.539 Y133.73 E.02333
G1 X119.596 Y134.56 E.03177
G1 X119.531 Y135.403 E.03226
G1 X119.363 Y136.148 E.02917
G1 X119.08 Y136.91 E.03103
G3 X130.579 Y143.624 I-20.653 J48.577 E.50973
G3 X142.648 Y156.415 I-31.78 J42.073 E.67456
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.511 E2.16569
G1 X143.067 Y99.547 E.02094
G1 X141.861 Y99.552 E.04605
G1 X142.433 Y102.642 E.11998
G3 X142.396 Y103.05 I-.767 J.135 E.01581
G1 X142.179 Y103.363 E.01454
G1 X141.907 Y103.517 E.01195
G1 X141.667 Y103.555 E.00928
G1 X132.924 Y103.555 E.33377
G1 X132.585 Y103.478 E.01327
G3 X132.262 Y103.187 I.339 J-.701 E.01682
G1 X132.146 Y102.765 E.01669
G1 X132.159 Y102.635 E.005
G1 X132.729 Y99.555 E.11959
G3 X131.452 Y99.491 I1.277 J-38.031 E.04883
G3 X122.687 Y116.843 I-51.362 J-15.053 E.74636
G1 X124.145 Y118.065 E.07263
G1 X124.314 Y118.264 E.00996
G1 X124.411 Y118.546 E.01137
M204 S250
G1 X123.87 Y118.651 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X123.818 Y118.807 E.00519
G1 X121.576 Y121.481 E.11048
G1 X121.435 Y121.556 E.00506
G1 X121.262 Y121.505 E.00571
G1 X119.658 Y120.16 E.06625
; LINE_WIDTH: 0.523206
G1 X119.597 Y120.114 E.00244
; LINE_WIDTH: 0.544336
G1 X119.526 Y119.812 E.01031
G1 X119.181 Y120.183 E.01684
; LINE_WIDTH: 0.520436
G1 X118.674 Y120.73 E.02365
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.744 J-36.513 E.28368
G1 X116.993 Y130.4 E.19642
G1 X117.385 Y130.731 E.01625
G1 X117.893 Y131.278 E.02364
G1 X118.327 Y131.907 E.02418
G1 X118.665 Y132.594 E.02427
G1 X118.893 Y133.299 E.02344
M73 P47 R10
G1 X118.996 Y133.833 E.01723
G1 X119.044 Y134.594 E.02413
G1 X118.98 Y135.355 E.02419
G1 X118.825 Y136.02 E.02162
G1 X118.548 Y136.752 E.02477
G1 X118.31 Y137.19 E.01577
G3 X130.249 Y144.067 I-19.804 J48.18 E.4375
G3 X142.39 Y157.025 I-31.645 J41.817 E.56491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.229 E1.81938
G1 X144.167 Y98.938 E.00922
G3 X141.197 Y99.001 I-2.218 J-34.087 E.0941
G1 X141.889 Y102.738 E.1203
G1 X141.878 Y102.856 E.00376
G1 X141.736 Y102.991 E.00621
G1 X141.667 Y103.002 E.00223
G1 X132.924 Y103.002 E.27678
G1 X132.78 Y102.95 E.00487
G1 X132.699 Y102.773 E.00615
G1 X132.702 Y102.736 E.0012
G1 X133.394 Y99.002 E.12021
G1 X132.742 Y99.002 E.02062
G2 X131.038 Y98.937 I-3.033 J56.47 E.05402
G3 X122.131 Y116.662 I-50.886 J-14.468 E.63181
; LINE_WIDTH: 0.544336
G1 X121.791 Y117.108 E.01863
G1 X122.129 Y117.094 E.01125
; LINE_WIDTH: 0.521596
G1 X122.822 Y117.677 E.02875
; LINE_WIDTH: 0.519996
G1 X123.79 Y118.489 E.04
G1 X123.83 Y118.571 E.00289
; WIPE_START
M204 S10000
G1 X123.818 Y118.807 E-.08985
G1 X123.327 Y119.392 E-.29015
; WIPE_END
G1 E-.02 F1800
G1 X129.195 Y114.511 Z1.72 F36000
G1 X143.019 Y103.014 Z1.72
G1 Z1.32
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.641176
G1 F12977.062
G1 X143.018 Y102.526 E.01929
; LINE_WIDTH: 0.675876
G1 F12275.151
G1 X143.001 Y102.337 E.00794
; LINE_WIDTH: 0.721591
G1 F11458.63
G1 X142.978 Y102.088 E.0112
; LINE_WIDTH: 0.767306
G1 F10743.96
G1 X142.955 Y101.839 E.01195
; LINE_WIDTH: 0.813021
G1 F10113.204
G1 X142.932 Y101.59 E.01269
; LINE_WIDTH: 0.858736
G1 F9552.402
G1 X142.91 Y101.342 E.01344
; LINE_WIDTH: 0.904451
G1 F9050.528
G1 X142.887 Y101.093 E.01418
; LINE_WIDTH: 0.950166
G1 F8598.758
G1 X142.864 Y100.844 E.01493
; LINE_WIDTH: 0.995881
G1 F8189.945
G1 X142.841 Y100.595 E.01567
; LINE_WIDTH: 1.0416
G1 F7818.24
G1 X142.818 Y100.346 E.01642
; WIPE_START
G1 X142.841 Y100.595 E-.095
G1 X142.864 Y100.844 E-.095
G1 X142.887 Y101.093 E-.095
G1 X142.91 Y101.342 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.317 Y102.118 Z1.72 F36000
G1 X131.34 Y102.524 Z1.72
G1 Z1.32
G1 E.4 F1800
; LINE_WIDTH: 1.09956
G1 F7392.839
G1 X131.35 Y102.485 E.0028
; LINE_WIDTH: 1.09304
G1 F7438.368
G1 X131.416 Y102.238 E.01767
; LINE_WIDTH: 1.05504
G1 F7715.293
G1 X131.481 Y101.99 E.01704
; LINE_WIDTH: 1.01704
G1 F8013.636
G1 X131.546 Y101.742 E.0164
; LINE_WIDTH: 0.979036
G1 F8335.98
G1 X131.611 Y101.495 E.01577
; LINE_WIDTH: 0.941036
G1 F8685.343
G1 X131.62 Y101.458 E.00221
; LINE_WIDTH: 0.935916
G1 F8734.666
G1 X131.699 Y101.144 E.01905
; LINE_WIDTH: 0.895756
G1 F9141.883
G1 X131.778 Y100.83 E.0182
; LINE_WIDTH: 0.855596
G1 F9588.926
G1 X131.856 Y100.515 E.01735
; LINE_WIDTH: 0.815436
G1 F10081.937
G1 X131.935 Y100.201 E.0165
; LINE_WIDTH: 0.775276
G1 F10628.392
G1 X131.938 Y100.187 E.00068
; WIPE_START
G1 X131.935 Y100.201 E-.00537
G1 X131.856 Y100.515 E-.12315
G1 X131.778 Y100.83 E-.12315
G1 X131.699 Y101.144 E-.12315
G1 X131.696 Y101.157 E-.00517
; WIPE_END
G1 E-.02 F1800
G1 X127.669 Y107.641 Z1.72 F36000
G1 X121.791 Y117.108 Z1.72
G1 Z1.32
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X119.526 Y119.812 E.11727
; WIPE_START
M204 S10000
G1 X120.168 Y119.045 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.007 Y123.326 Z1.72 F36000
G1 Z1.32
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X122.234 Y123.626 I1.395 J-1.942 E.0893
G3 X122.94 Y126.109 I-5.306 J2.85 E.09934
G3 X122.045 Y127.994 I-2.286 J.07 E.08273
G2 X119.617 Y129.878 I11.658 J17.531 E.11746
G3 X121.113 Y135.985 I-5.797 J4.656 E.24784
G3 X129.294 Y140.631 I-25.039 J53.621 E.35959
G2 X130.481 Y137.42 I-4.219 J-3.384 E.13299
G2 X129.586 Y135.535 I-2.286 J-.07 E.08273
G1 X128.293 Y134.592 E.0611
G3 X125.524 Y129.879 I3.26 J-5.086 E.21656
G3 X126.418 Y127.994 I2.286 J-.07 E.08273
G1 X127.712 Y127.052 E.0611
G2 X130.481 Y122.339 I-3.26 J-5.086 E.21656
G2 X129.586 Y120.453 I-2.286 J-.07 E.08273
G1 X128.293 Y119.511 E.0611
G3 X125.623 Y115.625 I3.287 J-5.118 E.18485
G2 X128.08 Y111.69 I-61.747 J-41.278 E.17713
G2 X130.461 Y106.975 I-3.43 J-4.691 E.20934
G2 X131.422 Y104.7 I-27.651 J-13.029 E.09433
G1 X131.906 Y105.005 E.02183
G2 X133.815 Y105.224 I1.421 J-3.945 E.07401
G1 X135.252 Y106.315 E.06888
G3 X138.021 Y111.028 I-3.26 J5.086 E.21656
G3 X137.127 Y112.913 I-2.286 J.07 E.08273
G1 X135.833 Y113.855 E.0611
G2 X133.064 Y118.568 I3.26 J5.086 E.21656
G2 X133.959 Y120.453 I2.286 J.07 E.08273
G1 X135.252 Y121.396 E.0611
G3 X138.021 Y126.109 I-3.26 J5.086 E.21656
G3 X137.127 Y127.994 I-2.286 J.07 E.08273
G1 X135.833 Y128.937 E.0611
G2 X133.064 Y133.65 I3.26 J5.086 E.21656
G2 X133.959 Y135.535 I2.286 J.07 E.08273
G1 X135.252 Y136.477 E.0611
G3 X138.021 Y141.19 I-3.26 J5.086 E.21656
G3 X137.127 Y143.075 I-2.286 J.07 E.08273
G2 X134.761 Y144.903 I11.761 J17.671 E.11422
G3 X141.945 Y152.574 I-35.361 J40.318 E.40191
G1 X141.945 Y148.364 E.16071
G3 X140.605 Y144.961 I4.101 J-3.581 E.14244
G3 X141.945 Y142.75 I2.507 J.009 E.10368
G1 X141.945 Y133.283 E.36145
G3 X140.605 Y129.879 I4.101 J-3.581 E.14244
G3 X141.945 Y127.669 I2.507 J.009 E.10368
G1 X141.945 Y118.202 E.36146
G3 X140.605 Y114.798 I4.101 J-3.581 E.14244
G3 X141.945 Y112.588 I2.507 J.009 E.10368
G1 X141.945 Y110.245 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 1.48
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y111.245 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L9
M991 S0 P8 ;notify layer change


G17
G3 Z1.72 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.688 Y118.427
G1 Z1.48
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.692 Y118.735 E.01177
G3 X125.247 Y119.786 I-2.24 J-.33 E.04402
G1 X122.782 Y122.726 E.1465
G1 X122.383 Y123.081 E.0204
G1 X121.887 Y123.319 E.02098
G1 X121.387 Y123.408 E.0194
G1 X120.85 Y123.358 E.0206
G1 X120.317 Y123.145 E.02191
G3 X119.427 Y122.435 I10.069 J-13.546 E.04347
G3 X114.848 Y126.662 I-39.592 J-38.296 E.23805
G1 X118.022 Y129.017 E.15087
G3 X118.932 Y129.849 I-4.637 J5.988 E.04716
G1 X119.565 Y130.646 E.03882
G1 X120.076 Y131.531 E.03903
G1 X120.45 Y132.477 E.03885
G3 X120.765 Y134.489 I-7.461 J2.199 E.07797
G1 X120.698 Y135.507 E.03892
G1 X120.538 Y136.268 E.02971
G3 X130.508 Y142.102 I-21.907 J48.876 E.44188
G3 X142.443 Y154.069 I-32.646 J44.493 E.64782
G1 X142.443 Y104.536 E1.89114
G1 X142.27 Y104.634 E.00759
G1 X141.669 Y104.729 E.02324
G1 X132.922 Y104.729 E.33396
G1 X132.632 Y104.707 E.01109
G1 X132.11 Y104.552 E.02079
G1 X131.67 Y104.274 E.01986
G1 X131.286 Y103.839 E.02216
G1 X131.254 Y103.773 E.0028
G3 X124.386 Y116.519 I-51.029 J-19.275 E.55443
G1 X125.005 Y117.039 E.03087
G1 X125.357 Y117.425 E.01994
G1 X125.577 Y117.845 E.01809
G1 X125.686 Y118.281 E.01719
G1 X125.687 Y118.337 E.00213
G1 X125.1 Y118.429 F36000
G1 F13446.369
G1 X125.117 Y118.569 E.00538
G1 X125.041 Y118.981 E.016
G1 X124.857 Y119.334 E.01522
G1 X124.798 Y119.409 E.00363
G1 X122.337 Y122.346 E.14631
G3 X121.361 Y122.823 I-1.095 J-1.003 E.04246
G1 X120.906 Y122.767 E.01747
G1 X120.59 Y122.626 E.01321
G3 X119.383 Y121.635 I21.878 J-27.867 E.05965
G3 X113.893 Y126.683 I-38.831 J-36.719 E.28499
G1 X117.671 Y129.486 E.17958
G1 X118.523 Y130.268 E.04419
G1 X119.098 Y131 E.03554
G1 X119.563 Y131.813 E.03575
G1 X119.901 Y132.681 E.03557
G3 X120.136 Y133.793 I-9.658 J2.622 E.0434
G1 X120.18 Y134.525 E.028
G1 X120.114 Y135.456 E.03564
G1 X119.933 Y136.283 E.03234
G1 X119.836 Y136.599 E.0126
G3 X130.932 Y143.156 I-21.308 J48.724 E.4933
G3 X142.92 Y155.769 I-32.242 J42.652 E.66733
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.016 E1.97511
; LINE_WIDTH: 0.629266
G1 F13236.852
G1 X143.024 Y103.272 E.02884
; LINE_WIDTH: 0.636296
G1 F13082.265
G1 X143.021 Y103.016 E.01006
G1 X142.951 Y103.259 E.00992
; LINE_WIDTH: 0.629266
G1 F13236.852
G1 X142.567 Y103.806 E.02592
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.09 Y104.076 E.02094
G1 X141.669 Y104.143 E.01626
G1 X132.922 Y104.143 E.33396
G1 X132.719 Y104.128 E.00776
G1 X132.327 Y104.007 E.01565
G1 X132.046 Y103.825 E.01278
G1 F12910.871
G1 X131.761 Y103.496 E.01661
; LINE_WIDTH: 0.664288
G1 F11424.946
G1 X131.705 Y103.378 E.00537
; LINE_WIDTH: 0.70858
G1 F10995.617
G1 X131.649 Y103.26 E.00575
; LINE_WIDTH: 0.752871
G1 F10574.51
G1 X131.593 Y103.141 E.00613
; LINE_WIDTH: 0.797163
G1 F10161.597
G1 X131.537 Y103.023 E.00651
; LINE_WIDTH: 0.841455
G1 F9756.934
G1 X131.481 Y102.905 E.00689
; LINE_WIDTH: 0.885746
G1 F9249.364
G1 X131.425 Y102.787 E.00726
; LINE_WIDTH: 0.928172
G1 F8810.341
G1 X131.408 Y102.735 E.00319
; LINE_WIDTH: 0.970598
G1 F8411.108
G1 X131.39 Y102.683 E.00334
; LINE_WIDTH: 1.01302
G1 F8046.488
G1 X131.373 Y102.631 E.0035
; LINE_WIDTH: 1.05545
G1 F7712.165
G1 X131.356 Y102.579 E.00365
; LINE_WIDTH: 1.09788
G1 F7404.517
G1 X131.339 Y102.527 E.0038
G1 X131.301 Y102.569 E.00391
; LINE_WIDTH: 1.05545
G1 F7712.165
G1 X131.264 Y102.611 E.00375
; LINE_WIDTH: 1.01302
G1 F8046.488
G1 X131.227 Y102.654 E.0036
; LINE_WIDTH: 0.970598
G1 F8411.108
G1 X131.19 Y102.696 E.00344
; LINE_WIDTH: 0.928172
G1 F8810.341
G1 X131.153 Y102.739 E.00329
; LINE_WIDTH: 0.885746
G1 F9249.364
G1 X131.076 Y102.883 E.00906
; LINE_WIDTH: 0.841455
G1 F9756.934
G1 X131 Y103.027 E.00859
; LINE_WIDTH: 0.797163
G1 F10323.447
G1 X130.923 Y103.171 E.00812
; LINE_WIDTH: 0.752871
G1 F10843.858
G1 X130.846 Y103.315 E.00765
; LINE_WIDTH: 0.70858
G1 F11377.035
G1 X130.769 Y103.459 E.00718
; LINE_WIDTH: 0.664288
G1 F11922.992
G1 X130.693 Y103.604 E.0067
; LINE_WIDTH: 0.619996
G1 F13314.674
G1 X130.546 Y103.976 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.209 Y104.835 E.01996
G3 X123.59 Y116.617 I-50.075 J-20.381 E.51729
G1 X124.629 Y117.488 E.05176
G1 X124.892 Y117.782 E.01506
G1 X125.07 Y118.178 E.01657
G1 X125.089 Y118.339 E.00622
G1 X124.516 Y118.469 F36000
G1 F13446.369
G1 X124.531 Y118.553 E.00329
G1 X124.467 Y118.843 E.01133
G1 X124.349 Y119.033 E.00852
G1 X121.891 Y121.966 E.14611
G1 X121.64 Y122.16 E.01211
G1 X121.334 Y122.238 E.01206
G1 X121.075 Y122.206 E.00997
G1 X120.799 Y122.057 E.01198
G1 X119.335 Y120.83 E.07294
G3 X112.93 Y126.698 I-39.834 J-37.049 E.332
G1 X117.32 Y129.954 E.20869
G1 X118.114 Y130.687 E.04127
G1 X118.632 Y131.355 E.03226
G1 X119.049 Y132.096 E.03246
G1 X119.352 Y132.885 E.03228
G3 X119.551 Y133.829 I-11.812 J2.989 E.03682
G1 X119.596 Y134.56 E.028
G1 X119.531 Y135.405 E.03236
G1 X119.363 Y136.148 E.02906
G1 X119.08 Y136.91 E.03104
G3 X130.578 Y143.623 I-20.427 J48.191 E.50971
G3 X142.647 Y156.415 I-31.84 J42.131 E.67459
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.512 E2.16566
G1 X143.067 Y99.549 E.02095
G1 X141.864 Y99.554 E.04595
G1 X142.436 Y102.644 E.11998
G3 X142.398 Y103.052 I-.767 J.135 E.01581
G1 X142.181 Y103.365 E.01454
G1 X141.909 Y103.519 E.01195
G1 X141.669 Y103.557 E.00928
G1 X132.922 Y103.557 E.33396
G1 X132.583 Y103.48 E.01329
G3 X132.26 Y103.188 I.339 J-.701 E.01683
G1 X132.143 Y102.766 E.01671
G1 X132.156 Y102.637 E.00495
G1 X132.727 Y99.557 E.11959
G3 X131.452 Y99.49 I1.253 J-36.311 E.04874
G3 X122.789 Y116.709 I-51.394 J-15.07 E.73993
G1 X124.253 Y117.936 E.07295
G1 X124.475 Y118.243 E.01444
G1 X124.5 Y118.38 E.00533
M204 S250
G1 X123.978 Y118.521 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X123.926 Y118.678 E.00523
G1 X121.471 Y121.607 E.121
G1 X121.309 Y121.686 E.00569
G1 X121.154 Y121.633 E.00519
G1 X119.978 Y120.648 E.04858
; LINE_WIDTH: 0.522786
G1 X119.49 Y120.243 E.0202
; LINE_WIDTH: 0.529446
G1 X119.442 Y120.201 E.00204
; LINE_WIDTH: 0.544336
G1 X119.418 Y119.941 E.00869
G1 X118.674 Y120.73 E.03606
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.041 J-36.844 E.28367
G1 X116.988 Y130.397 E.19625
G1 X117.728 Y131.083 E.03193
G1 X118.192 Y131.689 E.02418
G1 X118.565 Y132.362 E.02435
G3 X118.979 Y135.363 I-5.126 J2.237 E.09712
G1 X118.825 Y136.02 E.02137
G1 X118.548 Y136.752 E.02479
G1 X118.31 Y137.19 E.01577
G3 X130.248 Y144.067 I-19.585 J47.799 E.43749
G3 X142.39 Y157.025 I-31.527 J41.708 E.56496
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.228 E1.81939
G1 X144.167 Y98.938 E.0092
G3 X141.199 Y99.004 I-2.216 J-32.984 E.09402
G1 X141.891 Y102.74 E.1203
G1 X141.88 Y102.858 E.00376
G1 X141.739 Y102.993 E.00621
G1 X141.669 Y103.004 E.00223
G1 X132.922 Y103.004 E.27694
G1 X132.777 Y102.952 E.00488
G1 X132.696 Y102.775 E.00615
G1 X132.7 Y102.738 E.00119
G1 X133.391 Y99.004 E.1202
G1 X132.742 Y99.004 E.02054
G2 X131.038 Y98.936 I-3.037 J54.832 E.05402
G3 X122.631 Y116 I-51.1 J-14.573 E.60551
; LINE_WIDTH: 0.521116
G1 X122.193 Y116.582 E.02311
; LINE_WIDTH: 0.544336
G1 X121.899 Y116.979 E.01641
G1 X122.197 Y116.933 E.01004
; LINE_WIDTH: 0.526776
G1 X122.794 Y117.435 E.02503
; LINE_WIDTH: 0.519996
G1 X123.898 Y118.36 E.0456
G1 X123.938 Y118.441 E.00286
; WIPE_START
M204 S10000
G1 X123.926 Y118.678 E-.09025
G1 X123.436 Y119.262 E-.28975
; WIPE_END
G1 E-.02 F1800
G1 X129.31 Y114.389 Z1.88 F36000
G1 X143.021 Y103.016 Z1.88
G1 Z1.48
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.638636
G1 F13031.609
G1 X143.02 Y102.529 E.01919
; LINE_WIDTH: 0.673606
G1 F12318.739
G1 X143.002 Y102.338 E.00797
; LINE_WIDTH: 0.719323
G1 F11496.582
G1 X142.979 Y102.089 E.01116
; LINE_WIDTH: 0.765039
G1 F10777.301
G1 X142.956 Y101.84 E.01191
; LINE_WIDTH: 0.810755
G1 F10142.723
G1 X142.934 Y101.591 E.01265
; LINE_WIDTH: 0.856471
G1 F9578.719
G1 X142.911 Y101.342 E.0134
; LINE_WIDTH: 0.902188
G1 F9074.137
G1 X142.888 Y101.093 E.01414
; LINE_WIDTH: 0.947904
G1 F8620.053
G1 X142.865 Y100.844 E.01489
; LINE_WIDTH: 0.99362
G1 F8209.251
G1 X142.842 Y100.596 E.01563
; LINE_WIDTH: 1.03934
G1 F7835.821
G1 X142.819 Y100.347 E.01638
; WIPE_START
G1 X142.842 Y100.596 E-.095
G1 X142.865 Y100.844 E-.095
G1 X142.888 Y101.093 E-.095
G1 X142.911 Y101.342 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.318 Y102.119 Z1.88 F36000
G1 X131.339 Y102.527 Z1.88
G1 Z1.48
G1 E.4 F1800
; LINE_WIDTH: 1.09788
G1 F7404.517
G1 X131.349 Y102.487 E.0028
; LINE_WIDTH: 1.09134
G1 F7450.332
G1 X131.414 Y102.239 E.0177
; LINE_WIDTH: 1.05322
G1 F7729.075
G1 X131.479 Y101.991 E.01706
; LINE_WIDTH: 1.0151
G1 F8029.488
G1 X131.545 Y101.742 E.01643
; LINE_WIDTH: 0.976976
G1 F8354.197
G1 X131.61 Y101.494 E.01579
; LINE_WIDTH: 0.938856
G1 F8706.276
G1 X131.62 Y101.457 E.00221
; LINE_WIDTH: 0.933776
G1 F8755.448
G1 X131.698 Y101.143 E.01899
; LINE_WIDTH: 0.893626
G1 F9164.543
G1 X131.777 Y100.829 E.01815
; LINE_WIDTH: 0.853476
G1 F9613.742
G1 X131.855 Y100.515 E.0173
; LINE_WIDTH: 0.813326
G1 F10109.245
G1 X131.934 Y100.2 E.01645
; LINE_WIDTH: 0.773176
G1 F10658.601
G1 X131.937 Y100.187 E.00066
; WIPE_START
G1 X131.934 Y100.2 E-.00519
G1 X131.855 Y100.515 E-.1231
G1 X131.777 Y100.829 E-.12309
G1 X131.698 Y101.143 E-.12309
G1 X131.695 Y101.157 E-.00552
; WIPE_END
G1 E-.02 F1800
G1 X127.677 Y107.647 Z1.88 F36000
G1 X121.899 Y116.979 Z1.88
G1 Z1.48
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X119.418 Y119.941 E.12845
; WIPE_START
M204 S10000
G1 X120.06 Y119.174 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.043 Y123.559 Z1.88 F36000
G1 Z1.48
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X122.292 Y123.691 I1.254 J-2.147 E.08926
G3 X123.108 Y126.109 I-5.746 J3.287 E.09804
G3 X122.352 Y127.994 I-2.036 J.278 E.0811
G2 X119.637 Y129.924 I17.134 J26.989 E.12725
G3 X121.113 Y135.984 I-5.83 J4.63 E.24573
G3 X129.304 Y140.635 I-22.625 J49.392 E.36008
G2 X130.649 Y137.42 I-4.793 J-3.893 E.13487
G2 X129.893 Y135.535 I-2.036 J-.278 E.0811
G1 X128.473 Y134.592 E.06508
M73 P48 R10
G3 X125.356 Y129.879 I3.058 J-5.409 E.22395
G3 X126.111 Y127.994 I2.036 J-.278 E.0811
G1 X127.532 Y127.052 E.06508
G2 X130.649 Y122.339 I-3.058 J-5.409 E.22395
G2 X129.893 Y120.453 I-2.036 J-.278 E.0811
G1 X128.473 Y119.511 E.06508
G3 X125.538 Y115.748 I3.397 J-5.676 E.1864
G2 X128.177 Y111.525 I-46.378 J-31.93 E.19019
G2 X130.628 Y106.607 I-3.358 J-4.743 E.21852
G2 X131.421 Y104.702 I-23.162 J-10.766 E.0788
G2 X133.511 Y105.226 I1.791 J-2.715 E.08385
G1 X135.072 Y106.315 E.07265
G3 X138.189 Y111.028 I-3.058 J5.409 E.22395
G3 X137.434 Y112.913 I-2.036 J.278 E.0811
G1 X136.013 Y113.855 E.06508
G2 X132.896 Y118.568 I3.058 J5.409 E.22395
G2 X133.652 Y120.453 I2.036 J.278 E.0811
G1 X135.072 Y121.396 E.06508
G3 X138.189 Y126.109 I-3.058 J5.409 E.22395
G3 X137.434 Y127.994 I-2.036 J.278 E.0811
G1 X136.013 Y128.937 E.06508
G2 X132.896 Y133.65 I3.058 J5.409 E.22395
G2 X133.652 Y135.535 I2.036 J.278 E.0811
G1 X135.072 Y136.477 E.06508
G3 X138.189 Y141.19 I-3.058 J5.409 E.22395
G3 X137.434 Y143.075 I-2.036 J.278 E.0811
G2 X134.795 Y144.932 I11.946 J19.778 E.1233
G3 X141.945 Y152.574 I-36.437 J41.263 E.40015
G1 X141.945 Y148.362 E.16079
G3 X140.616 Y145.903 I4.53 J-4.037 E.10769
G3 X140.508 Y144.018 I3.473 J-1.146 E.07292
G3 X141.945 Y142.576 I2.679 J1.233 E.07939
G1 X141.945 Y133.281 E.35487
G3 X140.616 Y130.822 I4.53 J-4.037 E.10769
G3 X140.508 Y128.937 I3.473 J-1.146 E.07292
G3 X141.945 Y127.495 I2.679 J1.233 E.07939
G1 X141.945 Y118.2 E.35487
G3 X140.616 Y115.741 I4.53 J-4.037 E.10769
G3 X140.508 Y113.855 I3.473 J-1.146 E.07292
G3 X141.945 Y112.413 I2.679 J1.233 E.07939
G1 X141.945 Y110.071 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 1.64
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y111.071 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L10
M991 S0 P9 ;notify layer change


G17
G3 Z1.88 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.779 Y118.313
G1 Z1.64
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.775 Y118.619 E.0117
G3 X125.27 Y119.758 I-2.057 J-.231 E.04832
G1 X122.701 Y122.824 E.15272
G1 X122.309 Y123.178 E.02016
G1 X121.816 Y123.421 E.02099
G1 X121.314 Y123.516 E.01953
G1 X120.772 Y123.469 E.02075
G1 X120.583 Y123.416 E.0075
G1 X120.117 Y123.185 E.01984
G3 X119.329 Y122.538 I25.981 J-32.458 E.03896
G3 X114.848 Y126.662 I-40.268 J-39.251 E.23263
G1 X118.015 Y129.012 E.15054
G3 X118.932 Y129.849 I-4.611 J5.973 E.04749
G1 X119.566 Y130.648 E.03895
G1 X120.076 Y131.531 E.0389
G1 X120.45 Y132.477 E.03885
G3 X120.765 Y134.489 I-7.456 J2.198 E.07798
G1 X120.698 Y135.506 E.0389
G1 X120.538 Y136.268 E.02972
G3 X130.507 Y142.101 I-22.155 J49.297 E.44182
G3 X142.443 Y154.069 I-32.645 J44.494 E.64787
G1 X142.443 Y104.539 E1.891
G1 X142.273 Y104.636 E.00748
G1 X141.672 Y104.731 E.02324
G1 X132.919 Y104.731 E.33416
G1 X132.628 Y104.709 E.01115
G1 X132.079 Y104.541 E.02191
G1 X131.652 Y104.263 E.01944
G2 X131.245 Y103.8 I-6.018 J4.888 E.02357
G3 X124.469 Y116.405 I-51.055 J-19.32 E.54793
G1 X125.096 Y116.93 E.03125
G1 X125.426 Y117.285 E.01849
G1 X125.646 Y117.681 E.01731
G1 X125.776 Y118.164 E.01907
G1 X125.777 Y118.223 E.00226
G1 X125.189 Y118.313 F36000
G1 F13446.369
G1 X125.197 Y118.454 E.00539
G3 X124.822 Y119.381 I-1.462 J-.052 E.03896
G1 X122.252 Y122.447 E.15272
G3 X121.282 Y122.931 I-1.103 J-.996 E.04238
G1 X120.833 Y122.881 E.01723
G1 X120.445 Y122.7 E.01636
G3 X119.285 Y121.737 I51.387 J-63.096 E.05755
G3 X113.893 Y126.683 I-39.755 J-37.928 E.27956
G1 X117.666 Y129.482 E.17935
G3 X118.523 Y130.268 I-4.175 J5.414 E.04447
G1 X119.1 Y131.003 E.03566
G1 X119.562 Y131.813 E.03562
G1 X119.901 Y132.681 E.03557
G3 X120.135 Y133.792 I-9.634 J2.617 E.04339
G1 X120.18 Y134.525 E.02801
G1 X120.114 Y135.455 E.03561
G1 X119.931 Y136.289 E.03259
G1 X119.836 Y136.599 E.01239
G3 X130.932 Y143.157 I-21.128 J48.42 E.49335
G3 X142.92 Y155.769 I-32.176 J42.588 E.6673
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.017 E1.97508
; LINE_WIDTH: 0.627836
G1 F13268.746
G1 X143.025 Y103.274 E.02874
; LINE_WIDTH: 0.633776
G1 F13137.262
G1 X143.022 Y103.018 E.01001
G1 X142.953 Y103.261 E.00987
; LINE_WIDTH: 0.627836
G1 F13268.746
G1 X142.569 Y103.808 E.02585
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.092 Y104.079 E.02094
G1 X141.672 Y104.145 E.01626
G1 X132.919 Y104.145 E.33416
G1 X132.715 Y104.13 E.0078
G1 X132.306 Y103.999 E.01642
G1 X132.033 Y103.818 E.0125
G1 F12376.996
G1 X131.756 Y103.493 E.0163
; LINE_WIDTH: 0.666926
G1 F10948.709
G1 X131.715 Y103.394 E.00443
; LINE_WIDTH: 0.713856
G1 F10603.24
G1 X131.674 Y103.294 E.00476
; LINE_WIDTH: 0.760786
G1 F10263.293
G1 X131.633 Y103.195 E.00509
; LINE_WIDTH: 0.807716
G1 F9928.9
G1 X131.592 Y103.096 E.00542
; LINE_WIDTH: 0.854646
G1 F9600.03
G1 X131.55 Y102.997 E.00574
; LINE_WIDTH: 0.901576
G1 F9080.532
G1 X131.509 Y102.897 E.00607
; LINE_WIDTH: 0.948506
G1 F8614.372
G1 X131.468 Y102.798 E.0064
; LINE_WIDTH: 0.995436
G1 F8193.737
G1 X131.427 Y102.699 E.00673
; LINE_WIDTH: 1.04237
G1 F7812.268
G1 X131.386 Y102.6 E.00706
; LINE_WIDTH: 1.0893
G1 F7464.738
G1 X131.345 Y102.5 E.00739
G1 X131.302 Y102.551 E.00457
; LINE_WIDTH: 1.04237
G1 F7812.268
G1 X131.258 Y102.601 E.00437
; LINE_WIDTH: 0.995436
G1 F8193.737
G1 X131.215 Y102.652 E.00417
; LINE_WIDTH: 0.948506
G1 F8614.372
G1 X131.172 Y102.702 E.00396
; LINE_WIDTH: 0.901576
G1 F9080.532
G1 X131.128 Y102.752 E.00376
; LINE_WIDTH: 0.854646
G1 F9600.03
G1 X131.085 Y102.803 E.00356
; LINE_WIDTH: 0.807716
G1 F10182.576
G1 X131.042 Y102.853 E.00335
; LINE_WIDTH: 0.760786
G1 F10391.57
G1 X130.998 Y102.904 E.00315
; LINE_WIDTH: 0.713856
G1 F10602.687
G1 X130.955 Y102.954 E.00295
; LINE_WIDTH: 0.666926
G1 F10815.882
G1 X130.911 Y103.004 E.00274
; LINE_WIDTH: 0.619996
G1 F12143.204
G1 X130.768 Y103.378 E.01527
G1 F13446.369
G1 X130.625 Y103.751 E.01527
G1 X130.211 Y104.82 E.04374
G3 X123.674 Y116.502 I-50.079 J-20.355 E.51243
G1 X124.72 Y117.379 E.05213
G1 X124.951 Y117.627 E.01294
G1 X125.151 Y118.036 E.01738
G1 X125.177 Y118.224 E.00725
G1 X124.606 Y118.357 F36000
G1 F13446.369
G1 X124.622 Y118.442 E.00328
G3 X124.374 Y119.004 I-.986 J-.099 E.02388
G1 X121.804 Y122.07 E.15272
G1 X121.557 Y122.265 E.01202
G1 X121.25 Y122.346 E.01211
G1 X120.958 Y122.306 E.01124
G1 X120.708 Y122.166 E.01097
G1 X119.238 Y120.934 E.07321
G3 X112.93 Y126.698 I-38.932 J-36.275 E.32661
G1 X117.317 Y129.952 E.20855
G1 X118.114 Y130.687 E.04139
G1 X118.634 Y131.357 E.03237
G1 X119.049 Y132.095 E.03234
G1 X119.352 Y132.885 E.03229
G3 X119.551 Y133.828 I-11.764 J2.98 E.03681
G1 X119.596 Y134.56 E.02801
G1 X119.531 Y135.405 E.03233
G1 X119.362 Y136.153 E.02931
G1 X119.08 Y136.91 E.03083
G3 X130.579 Y143.624 I-20.582 J48.456 E.50973
G3 X142.648 Y156.415 I-31.78 J42.073 E.67456
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.512 E2.16563
G1 X143.067 Y99.551 E.02095
G1 X141.866 Y99.556 E.04586
G1 X142.438 Y102.646 E.11998
G3 X142.401 Y103.054 I-.767 J.135 E.01581
G1 X142.184 Y103.367 E.01454
G1 X141.912 Y103.521 E.01195
G1 X141.672 Y103.559 E.00928
G1 X132.919 Y103.559 E.33416
G1 X132.569 Y103.476 E.01374
G3 X132.157 Y102.937 I.438 J-.762 E.02661
G1 X132.165 Y102.578 E.01371
G1 X132.724 Y99.559 E.11722
G3 X131.452 Y99.49 I1.235 J-34.78 E.04865
G3 X122.874 Y116.596 I-51.397 J-15.071 E.73452
G1 X124.344 Y117.828 E.07323
G1 X124.563 Y118.128 E.01419
G1 X124.59 Y118.269 E.00549
M204 S250
G1 X124.069 Y118.412 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G3 X123.947 Y118.653 I-.366 J-.034 E.00876
G1 X121.377 Y121.718 E.12664
G1 X121.22 Y121.794 E.00552
G1 X121.063 Y121.742 E.00525
G1 X119.773 Y120.661 E.05329
; LINE_WIDTH: 0.522426
G1 X119.399 Y120.351 E.01545
; LINE_WIDTH: 0.536066
G1 X119.307 Y120.223 E.00516
; LINE_WIDTH: 0.544336
G1 X119.327 Y120.049 E.0058
G1 X118.674 Y120.73 E.03137
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.756 J-36.526 E.28368
G1 X116.988 Y130.396 E.19622
G1 X117.728 Y131.083 E.03196
G1 X118.194 Y131.692 E.02427
G1 X118.565 Y132.361 E.02425
G1 X118.833 Y133.077 E.02421
G1 X118.953 Y133.631 E.01793
G1 X119.033 Y134.411 E.02484
G1 X118.98 Y135.357 E.02999
G1 X118.824 Y136.026 E.02174
G1 X118.547 Y136.754 E.02466
G1 X118.31 Y137.19 E.01572
G3 X130.249 Y144.067 I-19.576 J47.784 E.43753
G3 X142.39 Y157.025 I-31.646 J41.818 E.56491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.228 E1.8194
G1 X144.167 Y98.938 E.00919
G3 X141.202 Y99.006 I-2.213 J-31.964 E.09394
G1 X141.894 Y102.742 E.1203
G1 X141.883 Y102.86 E.00376
G1 X141.741 Y102.996 E.00621
G1 X132.919 Y103.007 E.2793
G1 X132.773 Y102.952 E.00495
G1 X132.698 Y102.826 E.00464
G3 X133.025 Y100.973 I62.839 J10.113 E.05957
G1 X133.389 Y99.007 E.06332
G3 X132.132 Y98.986 I-.321 J-18.43 E.03979
G2 X131.038 Y98.936 I-.887 J7.542 E.03473
G3 X122.631 Y116 I-51.1 J-14.573 E.60551
; LINE_WIDTH: 0.520686
G1 X122.282 Y116.463 E.01837
; LINE_WIDTH: 0.544336
G1 X121.99 Y116.87 E.01667
G1 X122.198 Y116.792 E.00739
; LINE_WIDTH: 0.533656
G1 X122.327 Y116.858 E.00473
; LINE_WIDTH: 0.520686
G1 X122.786 Y117.243 E.01899
; LINE_WIDTH: 0.519996
G1 X123.989 Y118.252 E.04969
G1 X124.029 Y118.331 E.00283
; WIPE_START
M204 S10000
G1 X124.017 Y118.569 E-.09059
G1 X123.947 Y118.653 E-.04136
G1 X123.527 Y119.153 E-.24805
; WIPE_END
G1 E-.02 F1800
G1 X129.407 Y114.287 Z2.04 F36000
G1 X143.022 Y103.018 Z2.04
G1 Z1.64
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.636116
G1 F13086.179
G1 X143.021 Y102.531 E.0191
; LINE_WIDTH: 0.671356
G1 F12362.25
G1 X143.003 Y102.339 E.008
; LINE_WIDTH: 0.717071
G1 F11534.491
G1 X142.98 Y102.09 E.01113
; LINE_WIDTH: 0.762786
G1 F10810.626
G1 X142.957 Y101.841 E.01187
; LINE_WIDTH: 0.808501
G1 F10172.251
G1 X142.935 Y101.592 E.01262
; LINE_WIDTH: 0.854216
G1 F9605.065
G1 X142.912 Y101.343 E.01336
; LINE_WIDTH: 0.899931
G1 F9097.789
G1 X142.889 Y101.094 E.01411
; LINE_WIDTH: 0.945646
G1 F8641.407
G1 X142.866 Y100.845 E.01485
; LINE_WIDTH: 0.991361
G1 F8228.626
G1 X142.843 Y100.596 E.0156
; LINE_WIDTH: 1.03708
G1 F7853.481
G1 X142.82 Y100.347 E.01634
; WIPE_START
G1 X142.843 Y100.596 E-.095
G1 X142.866 Y100.845 E-.095
G1 X142.889 Y101.094 E-.095
G1 X142.912 Y101.343 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.317 Y102.103 Z2.04 F36000
G1 X131.345 Y102.5 Z2.04
G1 Z1.64
G1 E.4 F1800
; LINE_WIDTH: 1.0893
G1 F7464.738
G1 X131.364 Y102.43 E.00503
; LINE_WIDTH: 1.07738
G1 F7550.047
G1 X131.449 Y102.106 E.02277
; LINE_WIDTH: 1.02882
M73 P48 R9
G1 F7918.659
G1 X131.534 Y101.782 E.02171
; LINE_WIDTH: 0.98027
G1 F8325.112
G1 X131.618 Y101.458 E.02065
; LINE_WIDTH: 0.931716
G1 F8775.547
G1 X131.697 Y101.143 E.01896
; LINE_WIDTH: 0.891561
G1 F9186.62
G1 X131.776 Y100.829 E.01811
; LINE_WIDTH: 0.851406
G1 F9638.097
G1 X131.854 Y100.514 E.01726
; LINE_WIDTH: 0.811251
G1 F10136.245
G1 X131.933 Y100.2 E.01641
; LINE_WIDTH: 0.771096
G1 F10688.692
G1 X131.936 Y100.187 E.00064
; WIPE_START
G1 X131.933 Y100.2 E-.00503
G1 X131.854 Y100.514 E-.12313
G1 X131.776 Y100.829 E-.12313
G1 X131.697 Y101.143 E-.12313
G1 X131.694 Y101.157 E-.00557
; WIPE_END
G1 E-.02 F1800
G1 X127.683 Y107.651 Z2.04 F36000
G1 X121.99 Y116.87 Z2.04
G1 Z1.64
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X119.327 Y120.049 E.13786
; WIPE_START
M204 S10000
G1 X119.969 Y119.283 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.1 Y123.732 Z2.04 F36000
G1 Z1.64
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X122.343 Y123.725 I1.114 J-2.084 E.08925
G3 X123.339 Y127.052 I-4.698 J3.218 E.13473
G1 X123.179 Y127.523 E.019
G3 X121.996 Y128.465 I-2.332 J-1.713 E.05843
G2 X119.661 Y129.956 I6.147 J12.202 E.10592
G3 X121.111 Y135.984 I-5.813 J4.586 E.24423
G3 X129.292 Y140.63 I-25.1 J53.726 E.35962
G2 X130.879 Y136.477 I-4.734 J-4.188 E.17344
G2 X130.285 Y135.535 I-1.359 J.198 E.0438
G2 X128.679 Y134.592 I-8.739 J13.055 E.07115
G3 X125.125 Y128.937 I2.858 J-5.74 E.26822
G3 X125.719 Y127.994 I1.359 J.198 E.0438
G3 X127.326 Y127.052 I8.739 J13.055 E.07115
G2 X130.879 Y121.396 I-2.858 J-5.74 E.26822
G2 X130.285 Y120.453 I-1.359 J.198 E.0438
G2 X128.679 Y119.511 I-8.739 J13.055 E.07115
G3 X125.469 Y115.848 I3.126 J-5.978 E.19023
G2 X128.257 Y111.377 I-45.373 J-31.404 E.20125
G2 X130.879 Y106.315 I-3.775 J-5.165 E.22561
G1 X130.828 Y106.162 E.00614
G1 X131.421 Y104.704 E.06014
G2 X133.127 Y105.228 I1.598 J-2.16 E.06948
G2 X134.866 Y106.315 I5.313 J-6.57 E.07847
G3 X138.42 Y111.97 I-2.858 J5.74 E.26822
G3 X137.826 Y112.913 I-1.359 J-.198 E.0438
G3 X136.219 Y113.855 I-8.739 J-13.054 E.07115
G2 X132.666 Y119.511 I2.858 J5.74 E.26822
G2 X133.26 Y120.453 I1.359 J-.198 E.0438
G2 X134.866 Y121.396 I8.739 J-13.055 E.07115
G3 X138.42 Y127.052 I-2.858 J5.74 E.26822
G3 X137.826 Y127.994 I-1.359 J-.198 E.0438
G3 X136.219 Y128.937 I-8.739 J-13.054 E.07115
G2 X132.666 Y134.592 I2.858 J5.74 E.26822
G2 X133.26 Y135.535 I1.359 J-.198 E.0438
G2 X134.866 Y136.477 I8.739 J-13.055 E.07115
G3 X138.42 Y142.133 I-2.858 J5.74 E.26822
G1 X138.26 Y142.604 E.019
G3 X137.077 Y143.547 I-2.332 J-1.713 E.05843
G2 X134.825 Y144.959 I6.464 J12.808 E.10164
G3 X141.945 Y152.574 I-35.409 J40.249 E.39866
G1 X141.945 Y148.334 E.16188
G3 X140.206 Y144.018 I4.569 J-4.349 E.18194
G3 X140.801 Y143.075 I1.359 J.198 E.0438
G3 X141.945 Y142.386 I6.374 J9.296 E.05105
G1 X141.945 Y133.252 E.34873
G3 X140.206 Y128.937 I4.569 J-4.349 E.18194
G3 X140.801 Y127.994 I1.359 J.198 E.0438
G3 X141.945 Y127.305 I6.374 J9.296 E.05105
G1 X141.945 Y118.171 E.34873
G3 X140.206 Y113.855 I4.569 J-4.349 E.18194
G3 X140.801 Y112.913 I1.359 J.198 E.0438
G3 X141.945 Y112.224 I6.374 J9.296 E.05105
G1 X141.945 Y109.881 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 1.8
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y110.881 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L11
M991 S0 P10 ;notify layer change


G17
G3 Z2.04 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.855 Y118.217
G1 Z1.8
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.861 Y118.522 E.01165
G1 X125.765 Y118.966 E.01732
G3 X125.195 Y119.848 I-2.578 J-1.041 E.04036
G1 X122.625 Y122.914 E.15272
G1 X122.247 Y123.259 E.01955
G1 X121.755 Y123.508 E.02105
G1 X121.252 Y123.607 E.0196
G1 X120.706 Y123.564 E.02089
G1 X120.174 Y123.36 E.02175
G2 X119.246 Y122.627 I-41.809 J52.015 E.04517
G3 X114.848 Y126.662 I-40.244 J-39.442 E.22798
G1 X118.015 Y129.012 E.15054
G3 X118.932 Y129.849 I-4.609 J5.971 E.04749
G1 X119.567 Y130.649 E.03896
G1 X120.076 Y131.532 E.03893
G1 X120.45 Y132.48 E.03892
G3 X120.765 Y134.49 I-7.467 J2.197 E.07788
G1 X120.698 Y135.504 E.0388
G1 X120.538 Y136.268 E.0298
G3 X131.285 Y142.689 I-21.894 J48.845 E.47905
G3 X142.443 Y154.069 I-32.658 J43.183 E.61071
G1 X142.443 Y104.543 E1.89087
G1 X142.275 Y104.638 E.00737
G1 X141.674 Y104.733 E.02324
G1 X132.917 Y104.733 E.33435
G3 X132.284 Y104.627 I0 J-1.95 E.02459
G1 X131.863 Y104.424 E.01785
G2 X131.243 Y103.805 I-6.237 J5.642 E.03347
G3 X124.54 Y116.307 I-51.092 J-19.344 E.54311
G1 X125.173 Y116.838 E.03156
G1 X125.46 Y117.136 E.01579
G1 X125.7 Y117.536 E.0178
G1 X125.852 Y118.063 E.02096
G1 X125.853 Y118.127 E.00243
G1 X125.27 Y118.264 F36000
G1 F13446.369
G1 X125.264 Y118.571 E.01173
G1 X125.13 Y118.964 E.01587
G3 X124.746 Y119.472 I-2.532 J-1.517 E.02434
G1 X122.176 Y122.537 E.15272
G3 X121.215 Y123.023 I-1.106 J-.997 E.04205
G1 X120.785 Y122.981 E.01651
G1 X120.436 Y122.835 E.01443
G3 X119.202 Y121.824 I27.312 J-34.605 E.06092
G3 X113.893 Y126.683 I-39.242 J-37.543 E.27497
G1 X117.666 Y129.482 E.17935
G3 X118.523 Y130.268 I-4.173 J5.412 E.04447
G1 X119.1 Y131.003 E.03567
G1 X119.563 Y131.814 E.03565
G1 X119.901 Y132.684 E.03564
G3 X120.136 Y133.793 I-9.663 J2.619 E.04332
G1 X120.18 Y134.525 E.028
G1 X120.115 Y135.453 E.03552
G1 X119.931 Y136.289 E.03266
G1 X119.836 Y136.599 E.01239
G3 X130.931 Y143.156 I-21.129 J48.421 E.4933
G3 X142.92 Y155.769 I-32.391 J42.793 E.66732
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.018 E1.97504
; LINE_WIDTH: 0.626406
G1 F13300.794
G1 X143.026 Y103.275 E.02864
; LINE_WIDTH: 0.631256
G1 F13192.723
G1 X143.023 Y103.02 E.00996
G1 X142.955 Y103.263 E.00982
; LINE_WIDTH: 0.626406
G1 F13300.794
G1 X142.572 Y103.81 E.02578
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.095 Y104.081 E.02094
G1 X141.674 Y104.147 E.01626
G1 X132.917 Y104.147 E.33435
G1 X132.474 Y104.073 E.01713
G1 X132.18 Y103.931 E.01249
G1 X131.768 Y103.519 E.02225
; LINE_WIDTH: 0.664681
G1 F11457.409
G1 X131.71 Y103.397 E.00556
; LINE_WIDTH: 0.709366
G1 F11013.209
G1 X131.651 Y103.275 E.00595
; LINE_WIDTH: 0.754051
G1 F10577.762
G1 X131.593 Y103.152 E.00635
; LINE_WIDTH: 0.798736
G1 F10151.099
G1 X131.535 Y103.03 E.00674
; LINE_WIDTH: 0.843421
G1 F9733.218
G1 X131.477 Y102.908 E.00713
; LINE_WIDTH: 0.888106
G1 F9223.796
G1 X131.419 Y102.786 E.00753
; LINE_WIDTH: 0.929372
G1 F8798.53
G1 X131.402 Y102.735 E.00313
; LINE_WIDTH: 0.970638
G1 F8410.748
G1 X131.385 Y102.684 E.00328
; LINE_WIDTH: 1.0119
G1 F8055.706
G1 X131.369 Y102.633 E.00342
; LINE_WIDTH: 1.05317
G1 F7729.424
G1 X131.352 Y102.582 E.00357
; LINE_WIDTH: 1.09444
G1 F7428.545
G1 X131.335 Y102.531 E.00371
G1 X131.299 Y102.573 E.00382
; LINE_WIDTH: 1.05317
G1 F7729.424
G1 X131.263 Y102.614 E.00367
; LINE_WIDTH: 1.0119
G1 F8055.706
G1 X131.227 Y102.656 E.00352
; LINE_WIDTH: 0.970638
G1 F8410.748
G1 X131.19 Y102.698 E.00337
; LINE_WIDTH: 0.929372
G1 F8798.53
G1 X131.154 Y102.739 E.00322
; LINE_WIDTH: 0.888106
G1 F9223.796
G1 X131.077 Y102.883 E.00908
; LINE_WIDTH: 0.843421
G1 F9733.218
G1 X131 Y103.027 E.00861
; LINE_WIDTH: 0.798736
G1 F10302.198
G1 X130.924 Y103.171 E.00813
; LINE_WIDTH: 0.754051
G1 F10821.751
G1 X130.847 Y103.315 E.00766
; LINE_WIDTH: 0.709366
G1 F11354.085
G1 X130.77 Y103.459 E.00718
; LINE_WIDTH: 0.664681
G1 F11899.153
G1 X130.693 Y103.603 E.0067
; LINE_WIDTH: 0.619996
G1 F13289.481
G1 X130.548 Y103.976 E.01527
G1 F13446.369
G1 X130.331 Y104.532 E.02278
G3 X123.745 Y116.405 I-50.129 J-20.047 E.51976
G1 X124.797 Y117.287 E.05244
G3 X125.166 Y117.775 I-.876 J1.046 E.02355
G1 X125.272 Y118.144 E.01467
G1 X125.271 Y118.174 E.00114
G1 X124.682 Y118.261 F36000
G1 F13446.369
G1 X124.699 Y118.347 E.00337
G1 X124.636 Y118.64 E.01146
G3 X124.297 Y119.096 I-1.723 J-.929 E.02174
G1 X121.728 Y122.161 E.15272
G1 X121.486 Y122.354 E.0118
G1 X121.179 Y122.438 E.01214
G1 X120.909 Y122.407 E.01039
G1 X120.631 Y122.258 E.01205
G1 X119.157 Y121.022 E.07344
G3 X112.93 Y126.698 I-41.016 J-38.743 E.32198
G1 X117.317 Y129.952 E.20856
G1 X118.114 Y130.687 E.04139
G1 X118.634 Y131.357 E.03238
G1 X119.05 Y132.096 E.03236
G1 X119.352 Y132.888 E.03235
G3 X119.551 Y133.829 I-11.818 J2.986 E.03674
M73 P49 R9
G1 X119.595 Y134.561 E.028
G1 X119.531 Y135.403 E.03225
G1 X119.361 Y136.154 E.02938
G1 X119.08 Y136.91 E.03082
G3 X130.578 Y143.623 I-20.586 J48.462 E.50968
G3 X142.648 Y156.415 I-31.988 J42.271 E.67457
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.513 E2.16559
G1 X143.067 Y99.553 E.02095
G1 X141.869 Y99.558 E.04576
G1 X142.441 Y102.648 E.11998
G3 X142.403 Y103.056 I-.767 J.135 E.01581
G1 X142.186 Y103.369 E.01454
G1 X141.914 Y103.524 E.01195
G1 X141.674 Y103.561 E.00928
G1 X132.917 Y103.561 E.33435
G3 X132.496 Y103.438 I0 J-.778 E.01696
G1 X132.254 Y103.191 E.01323
G1 X132.138 Y102.768 E.01673
G1 X132.722 Y99.561 E.12445
G3 X131.452 Y99.49 I1.216 J-33.333 E.04856
G3 X122.946 Y116.5 I-51.409 J-15.075 E.72993
G1 X124.421 Y117.736 E.07346
G1 X124.631 Y118.014 E.01332
G1 X124.664 Y118.172 E.00616
M204 S250
G1 X124.146 Y118.319 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X124.094 Y118.477 E.00528
G1 X121.301 Y121.809 E.13762
G1 X121.145 Y121.886 E.00553
G1 X120.986 Y121.834 E.0053
G1 X119.599 Y120.672 E.05729
; LINE_WIDTH: 0.522296
G1 X119.322 Y120.443 E.01143
; LINE_WIDTH: 0.541216
G1 X119.226 Y120.211 E.00829
; LINE_WIDTH: 0.544336
G1 X119.25 Y120.141 E.00245
G1 X118.674 Y120.73 E.02739
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.703 J-36.467 E.28368
G1 X116.988 Y130.396 E.19622
G1 X117.728 Y131.083 E.03196
G1 X118.194 Y131.692 E.02428
G1 X118.565 Y132.362 E.02427
G1 X118.834 Y133.08 E.02426
G1 X118.952 Y133.628 E.01776
G1 X119.033 Y134.416 E.02507
G1 X118.98 Y135.355 E.02979
G1 X118.824 Y136.026 E.0218
G1 X118.547 Y136.754 E.02465
G1 X118.31 Y137.19 E.01572
G3 X130.248 Y144.067 I-19.575 J47.782 E.43749
G3 X142.39 Y157.025 I-31.674 J41.846 E.56495
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.228 E1.81941
G1 X144.167 Y98.938 E.00918
G3 X141.204 Y99.008 I-2.21 J-30.975 E.09387
G1 X141.896 Y102.744 E.1203
G1 X141.885 Y102.862 E.00376
G1 X141.744 Y102.998 E.00621
G1 X132.917 Y103.009 E.27946
G3 X132.724 Y102.901 I0 J-.226 E.00728
G1 X132.695 Y102.742 E.00513
G1 X133.386 Y99.009 E.1202
G3 X132.133 Y98.987 I-.32 J-17.854 E.0397
G2 X131.038 Y98.936 I-.887 J7.324 E.03474
G3 X122.631 Y116 I-51.099 J-14.572 E.60551
; LINE_WIDTH: 0.538916
G1 X122.146 Y116.674 E.02729
; LINE_WIDTH: 0.544336
G1 X122.067 Y116.778 E.00436
G1 X122.173 Y116.71 E.00421
; LINE_WIDTH: 0.538916
G1 X122.404 Y116.766 E.00782
; LINE_WIDTH: 0.520516
G1 X123.508 Y117.692 E.04568
; LINE_WIDTH: 0.519996
G1 X124.066 Y118.16 E.02303
G1 X124.106 Y118.239 E.0028
; WIPE_START
M204 S10000
G1 X124.094 Y118.477 E-.09088
G1 X123.605 Y119.061 E-.28912
; WIPE_END
G1 E-.02 F1800
G1 X129.489 Y114.2 Z2.2 F36000
G1 X143.023 Y103.02 Z2.2
G1 Z1.8
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.633576
G1 F13141.647
G1 X143.022 Y102.533 E.019
; LINE_WIDTH: 0.669086
G1 F12406.46
G1 X143.004 Y102.34 E.00804
; LINE_WIDTH: 0.7148
G1 F11572.99
G1 X142.981 Y102.091 E.01109
; LINE_WIDTH: 0.760514
G1 F10844.458
G1 X142.959 Y101.842 E.01183
; LINE_WIDTH: 0.806228
G1 F10202.215
G1 X142.936 Y101.593 E.01258
; LINE_WIDTH: 0.851941
G1 F9631.791
G1 X142.913 Y101.344 E.01332
; LINE_WIDTH: 0.897655
G1 F9121.776
G1 X142.89 Y101.095 E.01407
; LINE_WIDTH: 0.943369
G1 F8663.058
G1 X142.867 Y100.846 E.01482
; LINE_WIDTH: 0.989083
G1 F8248.266
G1 X142.844 Y100.597 E.01556
; LINE_WIDTH: 1.0348
G1 F7871.379
G1 X142.821 Y100.348 E.01631
; WIPE_START
G1 X142.844 Y100.597 E-.095
G1 X142.867 Y100.846 E-.095
G1 X142.89 Y101.095 E-.095
G1 X142.913 Y101.344 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.32 Y102.122 Z2.2 F36000
G1 X131.335 Y102.531 Z2.2
G1 Z1.8
G1 E.4 F1800
; LINE_WIDTH: 1.09444
G1 F7428.545
G1 X131.345 Y102.492 E.00279
; LINE_WIDTH: 1.08794
G1 F7474.374
G1 X131.411 Y102.242 E.01773
; LINE_WIDTH: 1.04964
G1 F7756.329
G1 X131.477 Y101.993 E.01708
; LINE_WIDTH: 1.01134
G1 F8060.389
G1 X131.542 Y101.743 E.01644
; LINE_WIDTH: 0.973036
G1 F8389.262
G1 X131.608 Y101.494 E.01579
; LINE_WIDTH: 0.934736
G1 F8746.113
G1 X131.617 Y101.457 E.00219
; LINE_WIDTH: 0.929636
G1 F8795.935
G1 X131.696 Y101.143 E.01891
; LINE_WIDTH: 0.889476
G1 F9209.019
G1 X131.775 Y100.829 E.01807
; LINE_WIDTH: 0.849316
G1 F9662.815
G1 X131.853 Y100.514 E.01722
; LINE_WIDTH: 0.809156
G1 F10163.652
G1 X131.932 Y100.2 E.01637
; LINE_WIDTH: 0.768996
G1 F10719.245
G1 X131.935 Y100.187 E.00061
; WIPE_START
G1 X131.932 Y100.2 E-.00485
G1 X131.853 Y100.514 E-.12315
G1 X131.775 Y100.829 E-.12315
G1 X131.696 Y101.143 E-.12315
G1 X131.692 Y101.158 E-.0057
; WIPE_END
G1 E-.02 F1800
G1 X127.688 Y107.655 Z2.2 F36000
G1 X122.067 Y116.778 Z2.2
G1 Z1.8
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X119.25 Y120.141 E.14585
; WIPE_START
M204 S10000
G1 X119.892 Y119.375 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.156 Y123.891 Z2.2 F36000
G1 Z1.8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X122.397 Y123.744 I.983 J-2.168 E.08928
G3 X123.7 Y127.052 I-6.956 J4.651 E.13676
G1 X123.683 Y127.523 E.018
G1 X123.338 Y127.994 E.0223
G1 X122.432 Y128.465 E.03898
G2 X119.683 Y129.983 I4.555 J11.501 E.12024
G3 X121.11 Y135.984 I-5.831 J4.558 E.2429
G3 X129.289 Y140.624 I-25.379 J54.257 E.35937
G2 X131.24 Y136.477 I-6.41 J-5.549 E.17717
G1 X131.224 Y136.006 E.018
G1 X130.879 Y135.535 E.0223
G1 X129.973 Y135.063 E.03898
G3 X126.802 Y133.178 I4.127 J-10.553 E.14149
G3 X124.764 Y128.937 I6.301 J-5.637 E.18206
G1 X124.781 Y128.465 E.018
G1 X125.126 Y127.994 E.0223
G1 X126.031 Y127.523 E.03898
G2 X129.203 Y125.638 I-4.127 J-10.554 E.14149
G2 X131.24 Y121.396 I-6.301 J-5.637 E.18206
G1 X131.224 Y120.925 E.018
G1 X130.879 Y120.453 E.0223
G2 X128.927 Y119.511 I-9.323 J16.809 E.08278
G3 X125.401 Y115.946 I2.607 J-6.106 E.19633
G2 X128.312 Y111.264 I-63.154 J-42.526 E.21054
G2 X129.203 Y110.556 I-2.443 J-3.986 E.04353
G2 X131.24 Y106.315 I-6.301 J-5.637 E.18206
G1 X131.224 Y105.843 E.018
G1 X131.053 Y105.61 E.01103
G1 X131.43 Y104.684 E.03817
G2 X132.542 Y105.202 I1.505 J-1.778 E.04741
G1 X132.666 Y105.372 E.00804
G1 X133.572 Y105.843 E.03898
G3 X136.743 Y107.729 I-4.127 J10.553 E.14149
G3 X138.781 Y111.97 I-6.301 J5.637 E.18206
G1 X138.764 Y112.442 E.018
G1 X138.419 Y112.913 E.0223
G1 X137.514 Y113.384 E.03898
G2 X134.342 Y115.269 I4.127 J10.553 E.14149
G2 X132.305 Y119.511 I6.3 J5.637 E.18206
G1 X132.321 Y119.982 E.018
G1 X132.666 Y120.453 E.0223
G1 X133.572 Y120.925 E.03898
G3 X136.743 Y122.81 I-4.127 J10.553 E.14149
G3 X138.781 Y127.052 I-6.301 J5.637 E.18206
G1 X138.764 Y127.523 E.018
G1 X138.419 Y127.994 E.0223
G1 X137.514 Y128.465 E.03898
G2 X134.342 Y130.351 I4.127 J10.553 E.14149
G2 X132.305 Y134.592 I6.3 J5.637 E.18206
G1 X132.321 Y135.063 E.018
G1 X132.666 Y135.535 E.0223
G1 X133.572 Y136.006 E.03898
G3 X136.743 Y137.891 I-4.127 J10.553 E.14149
G3 X138.781 Y142.133 I-6.301 J5.637 E.18206
G1 X138.764 Y142.604 E.018
G1 X138.419 Y143.075 E.0223
G1 X137.514 Y143.547 E.03898
G2 X134.862 Y144.979 I4.829 J12.111 E.11534
G3 X141.945 Y152.574 I-38.353 J42.875 E.39705
G1 X141.945 Y148.314 E.16263
G3 X139.845 Y144.018 I6.012 J-5.6 E.18525
G1 X139.862 Y143.547 E.018
G1 X140.207 Y143.075 E.0223
G3 X141.945 Y142.229 I8.366 J14.97 E.07386
G1 X141.945 Y133.233 E.34346
G3 X139.845 Y128.937 I6.012 J-5.6 E.18525
G1 X139.862 Y128.465 E.018
G1 X140.207 Y127.994 E.0223
G3 X141.945 Y127.148 I8.366 J14.97 E.07386
G1 X141.945 Y118.152 E.34346
G3 X139.845 Y113.855 I6.012 J-5.6 E.18525
G1 X139.862 Y113.384 E.018
G1 X140.207 Y112.913 E.0223
G3 X141.945 Y112.066 I8.366 J14.97 E.07386
G1 X141.945 Y109.724 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 1.96
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y110.724 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L12
M991 S0 P11 ;notify layer change


G17
G3 Z2.2 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.611 Y119.312
G1 Z1.96
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.481 Y119.507 E.00896
G3 X125.119 Y119.939 I-3.66 J-2.7 E.02153
G1 X122.549 Y123.004 E.15272
G1 X122.194 Y123.329 E.01836
G1 X121.702 Y123.582 E.02112
G1 X121.199 Y123.685 E.01961
G1 X120.65 Y123.644 E.02101
G1 X120.335 Y123.547 E.01258
G1 X119.878 Y123.286 E.02011
G1 X119.173 Y122.698 E.03505
G3 X114.848 Y126.662 I-38.474 J-37.634 E.22413
G1 X118.015 Y129.012 E.15055
G3 X118.932 Y129.849 I-4.606 J5.968 E.04747
G1 X119.565 Y130.646 E.03884
G1 X120.076 Y131.531 E.03904
G1 X120.45 Y132.48 E.03892
G3 X120.765 Y134.488 I-7.448 J2.194 E.07784
G1 X120.698 Y135.506 E.03896
G1 X120.538 Y136.268 E.02972
G3 X130.507 Y142.101 I-21.911 J48.883 E.44182
G3 X142.443 Y154.069 I-32.645 J44.495 E.64788
G1 X142.443 Y104.546 E1.89074
G1 X142.278 Y104.64 E.00726
G1 X141.677 Y104.735 E.02324
G1 X132.914 Y104.735 E.33454
G1 X132.623 Y104.713 E.01116
G1 X132.1 Y104.557 E.02083
G1 X131.659 Y104.277 E.01992
G2 X131.242 Y103.806 I-8.155 J6.79 E.02404
G3 X124.6 Y116.224 I-51.135 J-19.364 E.53919
G1 X125.239 Y116.76 E.03183
G1 X125.487 Y117.009 E.0134
G1 X125.749 Y117.421 E.01866
G1 X125.89 Y117.83 E.01653
G1 X125.936 Y118.285 E.01747
G1 X125.875 Y118.74 E.01751
G1 X125.715 Y119.156 E.01701
G1 X125.661 Y119.237 E.00371
G1 X125.157 Y118.929 F36000
G1 F13446.369
G1 X125.091 Y119.054 E.00541
G3 X124.673 Y119.559 I-7.327 J-5.651 E.02502
G1 X122.103 Y122.624 E.15272
G3 X121.159 Y123.1 I-1.102 J-1.011 E.04128
G1 X120.726 Y123.061 E.01659
G1 X120.555 Y123.004 E.00689
G1 X120.189 Y122.785 E.01628
G1 X119.131 Y121.898 E.05271
G3 X113.893 Y126.683 I-39.218 J-37.668 E.27106
G1 X117.666 Y129.482 E.17936
G3 X118.523 Y130.268 I-4.173 J5.411 E.04445
G1 X119.099 Y131 E.03556
G1 X119.563 Y131.814 E.03576
G1 X119.901 Y132.683 E.03563
G3 X120.135 Y133.79 I-9.584 J2.604 E.04321
G1 X120.18 Y134.524 E.02806
G1 X120.114 Y135.456 E.03567
G1 X119.932 Y136.284 E.03238
G1 X119.836 Y136.599 E.01257
G3 X130.931 Y143.156 I-21.304 J48.717 E.49328
G3 X142.92 Y155.769 I-32.374 J42.778 E.66732
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.019 E1.975
; LINE_WIDTH: 0.624966
G1 F13333.221
G1 X143.026 Y103.277 E.02855
; LINE_WIDTH: 0.628716
G1 F13249.101
G1 X143.025 Y103.021 E.0099
G1 X142.956 Y103.264 E.00978
; LINE_WIDTH: 0.624966
G1 F13333.221
G1 X142.574 Y103.812 E.02571
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.097 Y104.083 E.02094
G1 X141.677 Y104.149 E.01626
G1 X132.914 Y104.149 E.33454
G1 X132.71 Y104.134 E.00781
G1 X132.318 Y104.012 E.01568
G1 X132.036 Y103.829 E.01283
G1 F12880.753
G1 X131.751 Y103.498 E.01666
; LINE_WIDTH: 0.664881
G1 F11391.884
G1 X131.695 Y103.38 E.0054
; LINE_WIDTH: 0.709766
G1 F10961.812
G1 X131.639 Y103.261 E.00578
; LINE_WIDTH: 0.754651
G1 F10540
G1 X131.583 Y103.142 E.00617
; LINE_WIDTH: 0.799536
G1 F10126.463
G1 X131.527 Y103.023 E.00655
; LINE_WIDTH: 0.844421
G1 F9721.203
G1 X131.471 Y102.905 E.00693
; LINE_WIDTH: 0.889306
G1 F9210.85
G1 X131.415 Y102.786 E.00732
; LINE_WIDTH: 0.929996
G1 F8792.4
G1 X131.399 Y102.735 E.0031
; LINE_WIDTH: 0.970686
G1 F8410.317
G1 X131.383 Y102.685 E.00324
; LINE_WIDTH: 1.01138
G1 F8060.059
G1 X131.366 Y102.634 E.00338
; LINE_WIDTH: 1.05207
G1 F7737.808
G1 X131.35 Y102.584 E.00352
; LINE_WIDTH: 1.09276
G1 F7440.336
G1 X131.334 Y102.533 E.00367
G1 X131.298 Y102.574 E.00377
; LINE_WIDTH: 1.05207
G1 F7737.808
G1 X131.262 Y102.616 E.00362
; LINE_WIDTH: 1.01138
G1 F8060.059
G1 X131.226 Y102.657 E.00348
; LINE_WIDTH: 0.970686
G1 F8410.317
G1 X131.19 Y102.698 E.00333
; LINE_WIDTH: 0.929996
G1 F8792.4
G1 X131.155 Y102.739 E.00319
; LINE_WIDTH: 0.889306
G1 F9210.85
G1 X131.078 Y102.883 E.0091
; LINE_WIDTH: 0.844421
G1 F9721.203
G1 X131.001 Y103.027 E.00862
; LINE_WIDTH: 0.799536
G1 F10291.428
G1 X130.924 Y103.171 E.00814
; LINE_WIDTH: 0.754651
G1 F10810.937
G1 X130.847 Y103.315 E.00767
; LINE_WIDTH: 0.709766
G1 F11343.237
G1 X130.77 Y103.459 E.00719
; LINE_WIDTH: 0.664881
G1 F11888.298
G1 X130.693 Y103.603 E.00671
; LINE_WIDTH: 0.619996
G1 F13278.01
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.209 Y104.835 E.01998
G3 X123.805 Y116.322 I-49.995 J-20.345 E.50336
G1 X124.863 Y117.209 E.05271
G1 X125.036 Y117.383 E.00938
G1 X125.277 Y117.811 E.01876
G1 X125.351 Y118.254 E.01715
G1 X125.277 Y118.695 E.01707
G1 X125.198 Y118.849 E.0066
G1 X124.63 Y118.679 F36000
G1 F13446.369
G1 X124.583 Y118.754 E.00339
G1 X121.658 Y122.244 E.17387
G1 X121.425 Y122.43 E.01136
G1 X121.119 Y122.516 E.01215
G1 X120.789 Y122.467 E.01272
G1 X120.565 Y122.336 E.00991
G1 X119.087 Y121.097 E.07364
G3 X112.93 Y126.698 I-39.427 J-37.158 E.3181
G1 X117.317 Y129.952 E.20856
G1 X118.114 Y130.687 E.04138
G1 X118.632 Y131.355 E.03228
G1 X119.05 Y132.096 E.03247
G1 X119.352 Y132.887 E.03234
G3 X119.551 Y133.826 I-11.691 J2.962 E.03665
G1 X119.596 Y134.56 E.02806
G1 X119.531 Y135.405 E.03238
G1 X119.363 Y136.149 E.0291
G1 X119.08 Y136.91 E.03101
G3 X130.578 Y143.623 I-20.427 J48.19 E.5097
G3 X142.647 Y156.415 I-31.963 J42.247 E.67458
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.513 E2.16561
G3 X141.871 Y99.562 I-1.242 J-13.193 E.06663
G1 X142.443 Y102.651 E.11993
G3 X142.406 Y103.058 I-.767 J.135 E.01581
G1 X142.189 Y103.371 E.01454
G1 X141.917 Y103.526 E.01195
G1 X141.677 Y103.564 E.00928
G1 X132.914 Y103.564 E.33454
G1 X132.574 Y103.485 E.01333
G3 X132.251 Y103.192 I.34 J-.7 E.01689
G1 X132.136 Y102.769 E.01675
G3 X132.714 Y99.567 I316.413 J55.48 E.12421
G3 X131.452 Y99.49 I1.939 J-42.26 E.04828
G3 X123.008 Y116.418 I-51.425 J-15.082 E.72603
G1 X124.487 Y117.658 E.07366
G1 X124.69 Y117.921 E.01273
G1 X124.765 Y118.254 E.01302
G1 X124.702 Y118.561 E.01197
G1 X124.677 Y118.602 E.00183
M204 S250
G1 X124.159 Y118.399 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.237 Y121.886 E.14403
G1 X121.081 Y121.964 E.00554
G1 X120.92 Y121.912 E.00534
G1 X119.451 Y120.681 E.06069
; LINE_WIDTH: 0.521786
G1 X119.257 Y120.521 E.008
; LINE_WIDTH: 0.544286
G1 X119.192 Y120.209 E.01057
G1 X118.674 Y120.73 E.02441
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.764 J-36.535 E.28369
G1 X116.988 Y130.396 E.19623
G1 X117.728 Y131.083 E.03195
G1 X118.192 Y131.689 E.0242
G1 X118.565 Y132.362 E.02436
G1 X118.834 Y133.079 E.02424
G1 X118.953 Y133.631 E.01786
G1 X119.033 Y134.423 E.0252
G1 X118.98 Y135.358 E.02965
G1 X118.825 Y136.021 E.02157
G1 X118.547 Y136.754 E.02481
G1 X118.31 Y137.19 E.01572
G3 X130.248 Y144.067 I-19.585 J47.799 E.43749
G3 X142.39 Y157.025 I-31.644 J41.818 E.56495
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.229 E1.81939
G1 X144.167 Y98.938 E.00919
G3 X141.207 Y99.01 I-2.183 J-28.875 E.09379
G1 X141.899 Y102.746 E.12029
G1 X141.888 Y102.864 E.00376
G1 X141.746 Y103 E.00621
M73 P50 R9
G1 X141.677 Y103.011 E.00223
G1 X132.914 Y103.011 E.27742
G1 X132.769 Y102.958 E.00489
G1 X132.689 Y102.78 E.00617
G1 X132.692 Y102.744 E.00116
G1 X133.384 Y99.011 E.12021
G3 X132.132 Y98.989 I-.318 J-17.357 E.03964
G2 X131.038 Y98.936 I-.886 J7.107 E.03472
G3 X122.631 Y116 I-51.1 J-14.573 E.60551
; LINE_WIDTH: 0.544336
G1 X122.152 Y116.678 E.02759
G1 X122.471 Y116.687 E.01061
; LINE_WIDTH: 0.521746
G1 X123.49 Y117.544 E.04232
; LINE_WIDTH: 0.519996
G1 X124.132 Y118.081 E.02649
G1 X124.212 Y118.24 E.00563
G1 X124.188 Y118.314 E.00246
; WIPE_START
M204 S10000
G1 X123.551 Y119.085 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.439 Y114.228 Z2.36 F36000
G1 X143.025 Y103.021 Z2.36
G1 Z1.96
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.631036
G1 F13197.588
G1 X143.023 Y102.536 E.0189
; LINE_WIDTH: 0.666316
G1 F12460.838
G1 X143.006 Y102.343 E.00795
; LINE_WIDTH: 0.712034
G1 F11620.23
G1 X142.983 Y102.095 E.01104
; LINE_WIDTH: 0.757751
G1 F10885.869
G1 X142.96 Y101.846 E.01179
; LINE_WIDTH: 0.803469
G1 F10238.808
G1 X142.937 Y101.597 E.01254
; LINE_WIDTH: 0.849186
G1 F9664.356
G1 X142.914 Y101.348 E.01328
; LINE_WIDTH: 0.894904
G1 F9150.939
G1 X142.891 Y101.099 E.01402
; LINE_WIDTH: 0.940621
G1 F8689.321
G1 X142.869 Y100.85 E.01477
; LINE_WIDTH: 0.986339
G1 F8272.038
G1 X142.846 Y100.601 E.01552
; LINE_WIDTH: 1.03206
G1 F7892.996
G1 X142.823 Y100.352 E.01626
; WIPE_START
G1 X142.846 Y100.601 E-.095
G1 X142.869 Y100.85 E-.095
G1 X142.891 Y101.099 E-.095
G1 X142.914 Y101.348 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.324 Y100.546 Z2.36 F36000
G1 X131.934 Y100.188 Z2.36
G1 Z1.96
G1 E.4 F1800
; LINE_WIDTH: 0.766916
G1 F10749.68
G1 X131.931 Y100.199 E.00058
; LINE_WIDTH: 0.807051
G1 F10191.339
G1 X131.852 Y100.514 E.01632
; LINE_WIDTH: 0.847186
G1 F9688.135
G1 X131.774 Y100.828 E.01717
; LINE_WIDTH: 0.887321
G1 F9232.285
G1 X131.695 Y101.142 E.01801
; LINE_WIDTH: 0.927456
G1 F8817.405
G1 X131.617 Y101.457 E.01886
; LINE_WIDTH: 0.932556
G1 F8767.341
G1 X131.607 Y101.493 E.00219
; LINE_WIDTH: 0.970971
G1 F8407.758
G1 X131.541 Y101.743 E.01581
; LINE_WIDTH: 1.00939
G1 F8076.509
G1 X131.475 Y101.993 E.01646
; LINE_WIDTH: 1.0478
G1 F7770.372
G1 X131.41 Y102.244 E.01711
; LINE_WIDTH: 1.08622
G1 F7486.595
G1 X131.344 Y102.494 E.01775
; LINE_WIDTH: 1.09276
G1 F7440.336
G1 X131.334 Y102.533 E.00278
; WIPE_START
G1 X131.344 Y102.494 E-.01532
G1 X131.41 Y102.244 E-.09838
G1 X131.475 Y101.993 E-.09838
G1 X131.541 Y101.743 E-.09838
G1 X131.588 Y101.566 E-.06953
; WIPE_END
G1 E-.02 F1800
G1 X127.545 Y108.04 Z2.36 F36000
G1 X122.152 Y116.678 Z2.36
G1 Z1.96
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G2 X120.496 Y118.652 I437.208 J368.36 E.08567
G1 X119.192 Y120.208 E.06749
; WIPE_START
M204 S10000
G1 X119.835 Y119.442 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X125.65 Y114.499 Z2.36 F36000
G1 X127.207 Y113.176 Z2.36
G1 Z1.96
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X128.38 Y111.149 I-25.097 J-15.873 E.08945
G2 X130.359 Y109.097 I-3.046 J-4.919 E.10997
G2 X131.772 Y106.211 I-195.711 J-97.684 E.12268
G1 X132.244 Y105.88 E.02199
G3 X133.658 Y106.101 I.249 J3.049 E.05514
G3 X137.899 Y109.188 I-1.739 J6.847 E.20522
G3 X139.313 Y112.074 I-195.86 J97.758 E.12268
G1 X139.784 Y112.405 E.02199
G2 X141.945 Y111.918 I-.22 J-6.019 E.08505
G1 X141.945 Y118.134 E.2373
G3 X140.727 Y116.638 I4.459 J-4.876 E.07393
G3 X139.313 Y113.752 I195.412 J-97.538 E.12268
G1 X138.842 Y113.421 E.02199
G2 X137.428 Y113.641 I-.249 J3.049 E.05514
G2 X133.186 Y116.729 I1.739 J6.847 E.20522
G2 X131.772 Y119.614 I195.412 J97.538 E.12268
G1 X131.301 Y119.945 E.02199
G3 X129.887 Y119.725 I-.249 J-3.049 E.05514
G3 X126.342 Y117.599 I1.971 J-7.306 E.15987
G3 X125.969 Y119.69 I-2.548 J.625 E.08351
G1 X122.924 Y123.332 E.18125
G3 X122.464 Y123.743 I-1.639 J-1.371 E.02362
G1 X122.818 Y124.269 E.02423
G3 X124.232 Y127.155 I-195.487 J97.575 E.12268
G1 X124.703 Y127.486 E.02199
G2 X126.117 Y127.266 I.249 J-3.049 E.05514
G2 X130.359 Y124.178 I-1.739 J-6.847 E.20521
G2 X131.772 Y121.293 I-195.639 J-97.649 E.12268
G1 X132.244 Y120.961 E.02199
G3 X133.658 Y121.182 I.249 J3.049 E.05514
G3 X137.899 Y124.269 I-1.739 J6.847 E.20521
G3 X139.313 Y127.155 I-195.935 J97.795 E.12268
G1 X139.784 Y127.486 E.02199
G2 X141.945 Y127 I-.22 J-6.018 E.08505
G1 X141.945 Y133.215 E.2373
G3 X140.727 Y131.719 I4.459 J-4.876 E.07393
G3 X139.313 Y128.833 I195.412 J-97.538 E.12268
G1 X138.842 Y128.502 E.02199
G2 X137.428 Y128.723 I-.249 J3.049 E.05514
G2 X133.186 Y131.81 I1.739 J6.847 E.20522
G2 X131.772 Y134.696 I195.412 J97.538 E.12268
G1 X131.301 Y135.027 E.02199
G3 X129.887 Y134.806 I-.249 J-3.049 E.05514
G3 X125.646 Y131.719 I1.739 J-6.847 E.20522
G3 X124.232 Y128.833 I195.711 J-97.684 E.12268
G1 X123.761 Y128.502 E.02199
G1 X123.289 Y128.5 E.01799
G2 X119.705 Y130.012 I1.553 J8.688 E.14976
G3 X121.113 Y135.986 I-5.892 J4.542 E.24154
G3 X129.287 Y140.622 I-25.32 J54.16 E.35916
G2 X130.359 Y139.259 I-4.896 J-4.953 E.06637
G2 X131.772 Y136.374 I-195.711 J-97.684 E.12268
G1 X132.244 Y136.043 E.02199
G3 X133.658 Y136.263 I.249 J3.049 E.05514
G3 X137.899 Y139.351 I-1.739 J6.847 E.20522
G3 X139.313 Y142.236 I-195.86 J97.758 E.12268
G1 X139.784 Y142.567 E.02199
G2 X141.945 Y142.081 I-.22 J-6.019 E.08505
G1 X141.945 Y148.296 E.2373
G3 X140.727 Y146.8 I4.459 J-4.876 E.07393
G3 X139.313 Y143.915 I195.487 J-97.575 E.12268
G1 X138.842 Y143.583 E.02199
G1 X138.371 Y143.581 E.01799
G2 X134.902 Y145.015 I1.542 J8.639 E.1444
G3 X136.617 Y146.611 I-26.945 J30.681 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 2.12
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X135.885 Y145.93 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L13
M991 S0 P12 ;notify layer change


G17
G3 Z2.36 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.668 Y119.244
G1 Z2.12
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.537 Y119.44 E.00901
G1 X122.502 Y123.06 E.18035
G1 X122.149 Y123.388 E.01838
G1 X121.659 Y123.644 E.02114
G1 X121.155 Y123.75 E.01966
G1 X120.603 Y123.713 E.02112
G1 X120.237 Y123.596 E.01467
G1 X119.772 Y123.313 E.02079
G1 X119.112 Y122.76 E.03289
G3 X114.848 Y126.662 I-38.246 J-37.505 E.2208
G1 X118.015 Y129.012 E.15055
G3 X118.933 Y129.849 I-4.609 J5.972 E.0475
G1 X119.565 Y130.646 E.03882
G1 X120.076 Y131.531 E.03902
G1 X120.45 Y132.48 E.03894
G1 X120.652 Y133.335 E.03353
G3 X120.765 Y134.489 I-5.82 J1.149 E.04434
G1 X120.698 Y135.507 E.03895
G1 X120.538 Y136.268 E.02971
G3 X130.507 Y142.101 I-22.136 J49.269 E.4418
G3 X142.443 Y154.069 I-32.645 J44.495 E.64788
G1 X142.443 Y104.55 E1.8906
G1 X142.28 Y104.642 E.00715
G1 X141.679 Y104.737 E.02324
G1 X132.912 Y104.737 E.33473
G1 X132.62 Y104.715 E.01116
G1 X132.097 Y104.558 E.02086
G1 X131.656 Y104.279 E.01994
G3 X131.246 Y103.807 I2.871 J-2.911 E.02388
G3 X124.65 Y116.153 I-51.583 J-19.622 E.53584
G3 X125.506 Y116.9 I-4.485 J6.005 E.04342
G1 X125.787 Y117.318 E.01923
G1 X125.939 Y117.736 E.01699
G1 X125.992 Y118.214 E.01835
G1 X125.931 Y118.67 E.01757
G1 X125.772 Y119.088 E.01707
G1 X125.718 Y119.169 E.00372
G1 X125.133 Y118.99 F36000
G1 F13446.369
G1 X124.623 Y119.618 E.03088
G1 X122.054 Y122.684 E.15272
G3 X121.111 Y123.166 I-1.109 J-1.003 E.04133
G1 X120.676 Y123.129 E.01667
G1 X120.469 Y123.058 E.00835
G1 X120.115 Y122.836 E.01595
G1 X119.071 Y121.961 E.05204
G3 X113.893 Y126.683 I-39.657 J-38.281 E.26773
G1 X117.666 Y129.482 E.17936
G3 X118.524 Y130.268 I-4.176 J5.414 E.04448
G1 X119.099 Y131 E.03554
G1 X119.563 Y131.813 E.03574
G3 X120.135 Y133.793 I-6.735 J3.022 E.07892
G1 X120.18 Y134.524 E.02799
G1 X120.114 Y135.456 E.03567
G1 X119.933 Y136.283 E.03234
G1 X119.836 Y136.599 E.0126
G3 X130.931 Y143.156 I-21.131 J48.424 E.4933
G3 X142.92 Y155.769 I-32.242 J42.652 E.66734
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.02 E1.97497
; LINE_WIDTH: 0.623526
G1 F13365.808
G1 X143.027 Y103.279 E.02845
; LINE_WIDTH: 0.626196
G1 F13305.512
G1 X143.026 Y103.023 E.00985
G1 X142.958 Y103.266 E.00973
; LINE_WIDTH: 0.623526
G1 F13365.808
G1 X142.577 Y103.814 E.02564
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.1 Y104.085 E.02094
G1 X141.679 Y104.151 E.01626
G1 X132.912 Y104.151 E.33473
G1 X132.708 Y104.136 E.00781
G1 X132.315 Y104.014 E.0157
G1 X132.033 Y103.831 E.01284
G1 F12871.443
G1 X131.748 Y103.499 E.01668
; LINE_WIDTH: 0.66507
G1 F11381.769
G1 X131.692 Y103.38 E.00541
; LINE_WIDTH: 0.710143
G1 F10951.374
G1 X131.636 Y103.261 E.00579
; LINE_WIDTH: 0.755216
G1 F10529.259
G1 X131.58 Y103.143 E.00618
; LINE_WIDTH: 0.80029
G1 F10115.439
G1 X131.524 Y103.024 E.00656
; LINE_WIDTH: 0.845363
G1 F9709.916
G1 X131.468 Y102.905 E.00695
; LINE_WIDTH: 0.890436
G1 F9198.692
G1 X131.412 Y102.786 E.00734
; LINE_WIDTH: 0.930556
G1 F8786.906
G1 X131.396 Y102.736 E.00307
; LINE_WIDTH: 0.970676
G1 F8410.407
G1 X131.38 Y102.686 E.00321
; LINE_WIDTH: 1.0108
G1 F8064.847
G1 X131.364 Y102.636 E.00335
; LINE_WIDTH: 1.05092
G1 F7746.562
G1 X131.348 Y102.585 E.00348
; LINE_WIDTH: 1.09104
G1 F7452.446
G1 X131.332 Y102.535 E.00362
G1 X131.297 Y102.576 E.00372
; LINE_WIDTH: 1.05092
G1 F7746.562
G1 X131.261 Y102.617 E.00358
; LINE_WIDTH: 1.0108
G1 F8064.847
G1 X131.226 Y102.658 E.00344
; LINE_WIDTH: 0.970676
G1 F8410.407
G1 X131.191 Y102.699 E.0033
; LINE_WIDTH: 0.930556
G1 F8786.906
G1 X131.155 Y102.74 E.00316
; LINE_WIDTH: 0.890436
G1 F9198.692
G1 X131.078 Y102.884 E.00911
; LINE_WIDTH: 0.845363
G1 F9709.916
G1 X131.001 Y103.028 E.00863
; LINE_WIDTH: 0.80029
G1 F10281.306
G1 X130.924 Y103.171 E.00815
; LINE_WIDTH: 0.755216
G1 F10800.585
G1 X130.847 Y103.315 E.00767
; LINE_WIDTH: 0.710143
G1 F11332.686
G1 X130.77 Y103.459 E.00719
; LINE_WIDTH: 0.66507
G1 F11877.551
G1 X130.693 Y103.603 E.00671
; LINE_WIDTH: 0.619996
G1 F13266.651
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.209 Y104.835 E.01998
G3 X123.856 Y116.251 I-49.991 J-20.343 E.50004
G1 X124.919 Y117.142 E.05293
G1 X125.067 Y117.287 E.0079
G1 X125.325 Y117.722 E.01934
G1 X125.404 Y118.098 E.01465
G1 X125.364 Y118.525 E.01638
G1 X125.202 Y118.905 E.01577
G1 X125.19 Y118.92 E.00074
G1 X124.689 Y118.598 F36000
G1 F13446.369
G1 X124.639 Y118.688 E.00392
G1 X121.605 Y122.307 E.18029
G1 X121.374 Y122.494 E.01138
G1 X121.067 Y122.582 E.01216
G1 X120.773 Y122.548 E.01131
G1 X120.491 Y122.388 E.01238
G1 X119.027 Y121.16 E.07295
G3 X112.93 Y126.698 I-39.665 J-37.546 E.31477
G1 X117.317 Y129.952 E.20856
G1 X118.114 Y130.688 E.04141
G1 X118.632 Y131.355 E.03225
G1 X119.049 Y132.096 E.03245
G3 X119.531 Y135.406 I-5.593 J2.503 E.12934
G1 X119.363 Y136.148 E.02906
G1 X119.08 Y136.91 E.03104
G3 X130.578 Y143.623 I-20.532 J48.37 E.50969
G3 X142.647 Y156.415 I-31.84 J42.131 E.6746
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.514 E2.16558
G3 X141.874 Y99.564 I-1.241 J-12.863 E.06653
G1 X142.446 Y102.653 E.11993
G3 X142.408 Y103.06 I-.767 J.135 E.01581
G1 X142.191 Y103.373 E.01454
G1 X141.919 Y103.528 E.01195
G1 X141.679 Y103.566 E.00928
G1 X132.912 Y103.566 E.33473
G1 X132.571 Y103.487 E.01334
G3 X132.248 Y103.194 I.341 J-.7 E.0169
G1 X132.133 Y102.77 E.01676
G3 X132.713 Y99.568 I280.649 J49.17 E.12423
G3 X131.452 Y99.49 I1.603 J-36.405 E.04825
G3 X123.08 Y116.323 I-51.492 J-15.113 E.72146
G1 X123.086 Y116.37 E.00183
G1 X124.542 Y117.591 E.07257
G1 X124.739 Y117.84 E.01213
G1 X124.819 Y118.136 E.01171
G1 X124.779 Y118.438 E.01162
G1 X124.734 Y118.52 E.00357
M204 S250
G1 X124.215 Y118.333 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.182 Y121.951 E.14949
G1 X121.026 Y122.031 E.00555
G1 X120.864 Y121.979 E.00538
G1 X119.325 Y120.689 E.06358
; LINE_WIDTH: 0.521636
G1 X119.201 Y120.587 E.0051
; LINE_WIDTH: 0.544336
G1 X119.134 Y120.275 E.01062
G1 X118.674 Y120.73 E.02152
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.044 J-36.847 E.28368
G1 X116.988 Y130.396 E.19622
G1 X117.728 Y131.083 E.03197
G1 X118.192 Y131.69 E.02418
G1 X118.565 Y132.362 E.02434
G3 X119.034 Y134.426 I-5.182 J2.261 E.06741
G1 X118.98 Y135.358 E.02955
G1 X118.825 Y136.02 E.02153
G1 X118.548 Y136.753 E.02479
G1 X118.31 Y137.19 E.01576
G3 X130.248 Y144.067 I-19.573 J47.779 E.43749
G3 X142.39 Y157.025 I-31.527 J41.708 E.56497
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.228 E1.8194
G1 X144.167 Y98.938 E.00918
G3 X141.21 Y99.012 I-2.204 J-28.974 E.09371
G1 X141.901 Y102.748 E.12029
G1 X141.89 Y102.866 E.00376
G1 X141.749 Y103.002 E.00621
G1 X141.679 Y103.013 E.00223
G1 X132.912 Y103.013 E.27758
G1 X132.766 Y102.96 E.0049
G1 X132.686 Y102.782 E.00617
G1 X132.69 Y102.746 E.00115
G1 X133.381 Y99.013 E.12021
G3 X132.132 Y98.99 I-.317 J-16.845 E.03957
G2 X131.038 Y98.936 I-.886 J6.909 E.03472
G3 X122.631 Y116 I-51.1 J-14.573 E.60551
; LINE_WIDTH: 0.544336
G1 X122.187 Y116.632 E.02568
G1 X122.526 Y116.621 E.01127
; LINE_WIDTH: 0.520466
G1 X123.475 Y117.417 E.03926
; LINE_WIDTH: 0.519996
G1 X124.187 Y118.015 E.02944
G1 X124.268 Y118.173 E.00561
G1 X124.243 Y118.247 E.00247
; WIPE_START
M204 S10000
G1 X123.606 Y119.018 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.498 Y114.165 Z2.52 F36000
G1 X143.026 Y103.023 Z2.52
G1 Z2.12
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.628516
G1 F13253.561
G1 X143.025 Y102.538 E.0188
; LINE_WIDTH: 0.664056
G1 F12505.559
G1 X143.007 Y102.344 E.00798
; LINE_WIDTH: 0.709771
G1 F11659.154
G1 X142.984 Y102.095 E.01101
; LINE_WIDTH: 0.755486
G1 F10920.059
G1 X142.961 Y101.847 E.01175
; LINE_WIDTH: 0.801201
G1 F10269.083
G1 X142.938 Y101.598 E.0125
; LINE_WIDTH: 0.846916
G1 F9691.354
G1 X142.915 Y101.349 E.01324
; LINE_WIDTH: 0.892631
G1 F9175.167
G1 X142.893 Y101.1 E.01399
; LINE_WIDTH: 0.938346
G1 F8711.187
G1 X142.87 Y100.851 E.01473
; LINE_WIDTH: 0.984061
G1 F8291.874
G1 X142.847 Y100.602 E.01548
; LINE_WIDTH: 1.02978
G1 F7911.075
G1 X142.824 Y100.353 E.01622
; WIPE_START
G1 X142.847 Y100.602 E-.095
G1 X142.87 Y100.851 E-.095
G1 X142.893 Y101.1 E-.095
G1 X142.915 Y101.349 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.323 Y102.127 Z2.52 F36000
G1 X131.332 Y102.535 Z2.52
G1 Z2.12
G1 E.4 F1800
; LINE_WIDTH: 1.09104
G1 F7452.446
G1 X131.342 Y102.496 E.00277
; LINE_WIDTH: 1.08452
G1 F7498.714
G1 X131.408 Y102.246 E.01776
; LINE_WIDTH: 1.04602
G1 F7784.043
G1 X131.474 Y101.995 E.01711
; LINE_WIDTH: 1.00753
G1 F8091.945
G1 X131.54 Y101.744 E.01646
; LINE_WIDTH: 0.969031
G1 F8425.208
G1 X131.606 Y101.493 E.01581
; LINE_WIDTH: 0.930536
G1 F8787.101
G1 X131.615 Y101.457 E.00218
; LINE_WIDTH: 0.925456
G1 F8837.195
G1 X131.694 Y101.142 E.01883
; LINE_WIDTH: 0.885296
G1 F9254.255
G1 X131.773 Y100.828 E.01798
; LINE_WIDTH: 0.845136
G1 F9712.63
G1 X131.851 Y100.514 E.01713
; LINE_WIDTH: 0.804976
G1 F10218.78
G1 X131.93 Y100.199 E.01628
; LINE_WIDTH: 0.764816
G1 F10780.584
G1 X131.933 Y100.188 E.00056
; WIPE_START
G1 X131.93 Y100.199 E-.00448
G1 X131.851 Y100.514 E-.12315
G1 X131.773 Y100.828 E-.12315
G1 X131.694 Y101.142 E-.12315
G1 X131.69 Y101.158 E-.00607
; WIPE_END
G1 E-.02 F1800
G1 X127.5 Y107.537 Z2.52 F36000
G1 X119.134 Y120.275 Z2.52
G1 Z2.12
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G2 X120.858 Y118.22 I-279.218 J-236.053 E.08917
G2 X122.187 Y116.632 I-33.506 J-29.391 E.06885
; WIPE_START
M204 S10000
G1 X121.545 Y117.399 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X127.247 Y113.105 Z2.52 F36000
G1 Z2.12
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X128.419 Y111.077 I-24.891 J-15.733 E.08945
G2 X130.359 Y109.228 I-3.736 J-5.86 E.10295
G3 X131.772 Y106.83 I155.697 J90.17 E.1063
G3 X132.715 Y106.244 I1.115 J.744 E.04368
G3 X137.899 Y109.057 I-.156 J6.471 E.23381
G2 X139.313 Y111.455 I155.427 J-90.011 E.1063
G2 X140.256 Y112.041 I1.115 J-.744 E.04368
G2 X141.945 Y111.781 I.22 J-4.199 E.06573
G1 X141.945 Y118.13 E.24239
G3 X140.727 Y116.769 I4.257 J-5.036 E.06997
G2 X139.313 Y114.37 I-155.874 J90.274 E.1063
G2 X138.371 Y113.785 I-1.115 J.744 E.04368
G2 X133.186 Y116.598 I.156 J6.471 E.23381
G3 X131.772 Y118.996 I-155.799 J-90.23 E.1063
G3 X130.83 Y119.581 I-1.115 J-.744 E.04368
G3 X126.448 Y117.75 I.087 J-6.364 E.1858
G3 X126.026 Y119.623 I-2.482 J.424 E.07522
G1 X122.879 Y123.386 E.18728
G3 X122.513 Y123.73 I-1.361 J-1.082 E.01924
G3 X124.232 Y126.537 I-56.662 J36.635 E.12568
G2 X125.174 Y127.122 I1.115 J-.744 E.04368
G2 X130.359 Y124.309 I-.156 J-6.471 E.23381
G3 X131.772 Y121.911 I155.551 J90.084 E.1063
G3 X132.715 Y121.326 I1.115 J.744 E.04368
G3 X137.899 Y124.138 I-.156 J6.471 E.23381
G2 X139.313 Y126.537 I155.427 J-90.011 E.1063
G2 X140.256 Y127.122 I1.115 J-.744 E.04368
G2 X141.945 Y126.862 I.22 J-4.198 E.06573
G1 X141.945 Y133.211 E.24239
G3 X140.727 Y131.85 I4.257 J-5.037 E.06997
G2 X139.313 Y129.452 I-155.799 J90.23 E.1063
G2 X138.371 Y128.866 I-1.115 J.744 E.04368
G2 X133.186 Y131.679 I.156 J6.471 E.23381
G3 X131.772 Y134.077 I-155.874 J-90.274 E.1063
G3 X130.83 Y134.663 I-1.115 J-.744 E.04368
G3 X125.646 Y131.85 I.156 J-6.471 E.23381
G2 X124.232 Y129.452 I-155.551 J90.084 E.1063
G2 X123.289 Y128.866 I-1.115 J.744 E.04368
G2 X119.719 Y130.03 I.053 J6.219 E.14565
G3 X121.113 Y135.986 I-5.907 J4.523 E.24065
G3 X129.288 Y140.627 I-25.181 J53.879 E.35932
G2 X130.359 Y139.391 I-4.696 J-5.145 E.06258
G3 X131.772 Y136.992 I155.551 J90.084 E.1063
G3 X132.715 Y136.407 I1.115 J.744 E.04368
G3 X137.899 Y139.22 I-.156 J6.471 E.23381
G2 X139.313 Y141.618 I155.572 J-90.097 E.1063
G2 X140.256 Y142.203 I1.115 J-.744 E.04368
G2 X141.945 Y141.944 I.22 J-4.199 E.06573
G1 X141.945 Y148.292 E.24239
G3 X140.727 Y146.931 I4.257 J-5.037 E.06997
G2 X139.313 Y144.533 I-155.874 J90.274 E.1063
G2 X138.371 Y143.947 I-1.115 J.744 E.04368
G2 X134.924 Y145.035 I-.006 J5.985 E.14016
G3 X136.639 Y146.631 I-26.964 J30.679 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 2.28
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X135.907 Y145.95 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L14
M991 S0 P13 ;notify layer change


G17
G3 Z2.52 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.715 Y119.187
G1 Z2.28
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.584 Y119.384 E.00905
G1 X122.456 Y123.115 E.18586
G1 X122.111 Y123.438 E.01806
M73 P51 R9
G1 X121.622 Y123.697 E.02113
G1 X121.117 Y123.806 E.0197
G1 X120.563 Y123.771 E.0212
G1 X120.296 Y123.695 E.01059
G1 X119.827 Y123.447 E.02026
G1 X119.059 Y122.811 E.03805
G3 X114.848 Y126.662 I-39.048 J-38.477 E.21797
G1 X118.015 Y129.012 E.15055
G3 X118.933 Y129.849 I-4.607 J5.969 E.0475
G1 X119.565 Y130.646 E.03882
G1 X120.075 Y131.529 E.03892
G1 X120.45 Y132.48 E.03905
G3 X120.765 Y134.489 I-7.463 J2.197 E.07785
G1 X120.698 Y135.505 E.0389
G1 X120.538 Y136.268 E.02975
G3 X130.508 Y142.102 I-22.139 J49.273 E.44185
G3 X142.443 Y154.07 I-32.638 J44.485 E.64786
G1 X142.443 Y104.553 E1.89049
G1 X142.283 Y104.644 E.00704
G1 X141.682 Y104.739 E.02324
G1 X132.909 Y104.739 E.33492
G1 X132.617 Y104.717 E.01118
G1 X132.093 Y104.56 E.02088
G1 X131.652 Y104.28 E.01996
G1 X131.243 Y103.803 E.02398
G3 X124.692 Y116.092 I-51.339 J-19.475 E.53315
G3 X125.521 Y116.808 I-5.21 J6.874 E.04184
G1 X125.819 Y117.23 E.01972
G1 X125.981 Y117.656 E.0174
G1 X126.039 Y118.154 E.01912
G1 X125.979 Y118.612 E.01763
G1 X125.82 Y119.031 E.01712
G1 X125.765 Y119.112 E.00372
G1 X125.231 Y118.851 F36000
G1 F13446.369
G1 X125.135 Y119.008 E.00704
G1 X122.008 Y122.739 E.18586
G3 X121.071 Y123.222 I-1.111 J-1.003 E.04113
G1 X120.634 Y123.186 E.01673
G1 X120.168 Y122.971 E.0196
G1 X119.02 Y122.014 E.05708
G3 X113.893 Y126.683 I-39.047 J-37.724 E.26493
G1 X117.666 Y129.482 E.17936
G3 X118.523 Y130.268 I-4.172 J5.41 E.04448
G1 X119.099 Y131 E.03554
G1 X119.561 Y131.811 E.03564
G1 X119.901 Y132.684 E.03576
G3 X120.136 Y133.793 I-9.653 J2.618 E.04331
G1 X120.18 Y134.524 E.02797
G1 X120.114 Y135.455 E.03562
G1 X119.933 Y136.284 E.03239
G1 X119.836 Y136.599 E.01259
G3 X130.931 Y143.156 I-21.131 J48.424 E.4933
G3 X142.92 Y155.769 I-32.374 J42.778 E.66732
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.02 E1.97493
; LINE_WIDTH: 0.622076
G1 F13398.782
G1 X143.028 Y103.28 E.02836
; LINE_WIDTH: 0.623656
G1 F13362.86
G1 X143.027 Y103.025 E.0098
G1 X142.96 Y103.268 E.00968
; LINE_WIDTH: 0.622076
G1 F13398.782
G1 X142.579 Y103.816 E.02557
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.102 Y104.087 E.02094
G1 X141.682 Y104.153 E.01626
G1 X132.909 Y104.153 E.33492
G1 X132.705 Y104.138 E.00782
G1 X132.312 Y104.016 E.01572
G1 X132.03 Y103.832 E.01285
G1 F12525.457
G1 X131.743 Y103.496 E.01685
; LINE_WIDTH: 0.665248
G1 F11041.452
G1 X131.703 Y103.398 E.00438
; LINE_WIDTH: 0.7105
G1 F10697.74
G1 X131.663 Y103.299 E.00469
; LINE_WIDTH: 0.755752
G1 F10359.473
G1 X131.623 Y103.201 E.005
; LINE_WIDTH: 0.801004
G1 F10026.63
G1 X131.583 Y103.102 E.00532
; LINE_WIDTH: 0.846256
G1 F9699.232
G1 X131.543 Y103.003 E.00563
; LINE_WIDTH: 0.891508
G1 F9187.188
G1 X131.503 Y102.905 E.00595
; LINE_WIDTH: 0.93676
G1 F8726.497
G1 X131.463 Y102.806 E.00626
; LINE_WIDTH: 0.982012
G1 F8309.803
G1 X131.423 Y102.707 E.00657
; LINE_WIDTH: 1.02726
G1 F7931.09
G1 X131.383 Y102.609 E.00689
; LINE_WIDTH: 1.07252
G1 F7585.39
G1 X131.343 Y102.51 E.0072
G1 X131.3 Y102.56 E.00448
; LINE_WIDTH: 1.02726
G1 F7931.09
G1 X131.257 Y102.61 E.00428
; LINE_WIDTH: 0.982012
G1 F8309.803
G1 X131.213 Y102.66 E.00409
; LINE_WIDTH: 0.93676
G1 F8726.497
G1 X131.17 Y102.71 E.00389
; LINE_WIDTH: 0.891508
G1 F9187.188
G1 X131.127 Y102.76 E.0037
; LINE_WIDTH: 0.846256
G1 F9699.232
G1 X131.083 Y102.81 E.0035
; LINE_WIDTH: 0.801004
G1 F10271.722
G1 X131.04 Y102.86 E.00331
; LINE_WIDTH: 0.755752
G1 F10480.473
G1 X130.996 Y102.91 E.00311
; LINE_WIDTH: 0.7105
G1 F10691.345
G1 X130.953 Y102.96 E.00292
; LINE_WIDTH: 0.665248
G1 F10904.271
G1 X130.91 Y103.009 E.00272
; LINE_WIDTH: 0.619996
G1 F12236.849
G1 X130.766 Y103.383 E.01527
G1 F13446.369
G1 X130.623 Y103.756 E.01527
G1 X130.206 Y104.834 E.0441
G3 X123.899 Y116.192 I-50.046 J-20.356 E.49721
G1 X124.966 Y117.086 E.05313
G1 X125.285 Y117.475 E.01923
G1 X125.413 Y117.799 E.01328
G1 X125.448 Y118.255 E.01747
G1 X125.344 Y118.667 E.01622
G1 X125.278 Y118.774 E.00478
G1 X124.737 Y118.542 F36000
G1 F13446.369
G1 X124.686 Y118.631 E.00394
G1 X121.559 Y122.362 E.18586
G1 X121.33 Y122.548 E.01126
G1 X121.024 Y122.638 E.01217
G1 X120.696 Y122.594 E.01263
G1 X120.462 Y122.459 E.01032
G1 X118.976 Y121.213 E.07404
G3 X112.93 Y126.698 I-39.148 J-37.082 E.31197
G1 X117.317 Y129.952 E.20856
G1 X118.114 Y130.687 E.0414
G1 X118.632 Y131.355 E.03226
G1 X119.048 Y132.093 E.03236
G1 X119.352 Y132.888 E.03247
G3 X119.551 Y133.829 I-11.802 J2.984 E.03673
G1 X119.596 Y134.56 E.02797
G1 X119.531 Y135.404 E.03234
G1 X119.363 Y136.148 E.02911
G1 X119.08 Y136.91 E.03103
G3 X130.578 Y143.623 I-20.589 J48.468 E.50968
G3 X142.648 Y156.415 I-31.963 J42.247 E.67458
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.515 E2.16554
G3 X141.876 Y99.566 I-1.24 J-12.547 E.06644
G1 X142.448 Y102.655 E.11993
G3 X142.411 Y103.062 I-.767 J.135 E.01581
G1 X142.194 Y103.375 E.01454
G1 X141.922 Y103.53 E.01195
G1 X141.682 Y103.568 E.00928
G1 X132.909 Y103.568 E.33492
G1 X132.568 Y103.489 E.01336
G3 X132.146 Y102.943 I.431 J-.77 E.02706
G1 X132.157 Y102.576 E.01404
G2 X132.712 Y99.568 I-1695.501 J-314.735 E.11676
G3 X131.452 Y99.49 I1.335 J-31.759 E.04823
G3 X123.104 Y116.289 I-51.482 J-15.109 E.71989
G1 X124.59 Y117.535 E.07401
G1 X124.78 Y117.772 E.0116
G1 X124.866 Y118.078 E.01217
G1 X124.827 Y118.381 E.01166
G1 X124.781 Y118.463 E.00359
M204 S250
G1 X124.262 Y118.276 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.135 Y122.007 E.15413
G1 X120.98 Y122.087 E.00552
G3 X120.751 Y121.98 I.016 J-.333 E.0082
G1 X119.219 Y120.695 E.06332
; LINE_WIDTH: 0.521576
G1 X119.154 Y120.643 E.00263
; LINE_WIDTH: 0.546076
G1 X119.085 Y120.33 E.0107
G1 X118.674 Y120.729 E.01913
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.703 J-36.466 E.28369
G1 X116.988 Y130.396 E.19622
G1 X117.728 Y131.083 E.03197
G1 X118.192 Y131.689 E.02418
G1 X118.564 Y132.36 E.02427
G1 X118.834 Y133.08 E.02435
G3 X118.999 Y133.862 I-26.721 J6.05 E.02531
G1 X119.044 Y134.598 E.02335
G1 X118.98 Y135.357 E.0241
G1 X118.825 Y136.02 E.02158
G1 X118.548 Y136.753 E.02478
G1 X118.31 Y137.19 E.01576
G3 X130.248 Y144.067 I-19.805 J48.181 E.43747
G3 X142.39 Y157.025 I-31.645 J41.818 E.56495
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.228 E1.8194
G1 X144.167 Y98.939 E.00917
G3 X141.212 Y99.014 I-2.202 J-28.151 E.09364
G1 X141.904 Y102.75 E.12029
G1 X141.893 Y102.868 E.00376
G1 X141.751 Y103.004 E.00621
G1 X132.909 Y103.015 E.27994
G1 X132.764 Y102.962 E.0049
G1 X132.688 Y102.834 E.0047
G3 X133.014 Y100.981 I64.203 J10.36 E.05956
G1 X133.379 Y99.015 E.06332
G3 X132.131 Y98.991 I-.316 J-16.377 E.03951
G2 X131.038 Y98.936 I-.886 J6.723 E.03471
G3 X122.631 Y116 I-51.1 J-14.573 E.60551
; LINE_WIDTH: 0.544336
G1 X122.233 Y116.575 E.02323
G1 X122.573 Y116.565 E.01131
; LINE_WIDTH: 0.520286
G1 X123.462 Y117.31 E.03675
; LINE_WIDTH: 0.519996
G1 X124.234 Y117.958 E.03193
G1 X124.315 Y118.116 E.0056
G1 X124.29 Y118.191 E.00249
; WIPE_START
M204 S10000
G1 X123.653 Y118.962 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.548 Y114.113 Z2.68 F36000
G1 X143.027 Y103.025 Z2.68
G1 Z2.28
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.625976
G1 F13310.46
G1 X143.026 Y102.54 E.0187
; LINE_WIDTH: 0.661766
G1 F12551.202
G1 X143.008 Y102.345 E.00801
; LINE_WIDTH: 0.707483
G1 F11698.796
G1 X142.985 Y102.096 E.01097
; LINE_WIDTH: 0.753199
G1 F10954.808
G1 X142.962 Y101.847 E.01172
; LINE_WIDTH: 0.798915
G1 F10299.79
G1 X142.939 Y101.598 E.01246
; LINE_WIDTH: 0.844631
G1 F9718.683
G1 X142.917 Y101.35 E.01321
; LINE_WIDTH: 0.890347
G1 F9199.646
G1 X142.894 Y101.101 E.01395
; LINE_WIDTH: 0.936064
G1 F8733.237
G1 X142.871 Y100.852 E.0147
; LINE_WIDTH: 0.98178
G1 F8311.84
G1 X142.848 Y100.603 E.01544
; LINE_WIDTH: 1.0275
G1 F7929.237
G1 X142.825 Y100.354 E.01619
; WIPE_START
G1 X142.848 Y100.603 E-.095
G1 X142.871 Y100.852 E-.095
G1 X142.894 Y101.101 E-.095
G1 X142.917 Y101.35 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.321 Y102.104 Z2.68 F36000
G1 X131.341 Y102.499 Z2.68
G1 Z2.28
G1 E.4 F1800
; LINE_WIDTH: 1.0808
G1 F7525.371
G1 X131.36 Y102.428 E.00499
; LINE_WIDTH: 1.069
G1 F7611.196
G1 X131.445 Y102.104 E.0226
; LINE_WIDTH: 1.02042
G1 F7986.114
G1 X131.53 Y101.78 E.02154
; LINE_WIDTH: 0.97185
G1 F8399.881
G1 X131.615 Y101.456 E.02048
; LINE_WIDTH: 0.923276
G1 F8858.866
G1 X131.693 Y101.142 E.01877
; LINE_WIDTH: 0.883131
G1 F9277.86
G1 X131.772 Y100.827 E.01793
; LINE_WIDTH: 0.842986
G1 F9738.454
G1 X131.85 Y100.513 E.01708
; LINE_WIDTH: 0.802841
G1 F10247.169
G1 X131.929 Y100.199 E.01623
; LINE_WIDTH: 0.762696
G1 F10811.962
G1 X131.932 Y100.188 E.00053
; WIPE_START
G1 X131.929 Y100.199 E-.00422
G1 X131.85 Y100.513 E-.12311
G1 X131.772 Y100.827 E-.12311
G1 X131.693 Y101.142 E-.12311
G1 X131.689 Y101.158 E-.00644
; WIPE_END
G1 E-.02 F1800
G1 X127.496 Y107.536 Z2.68 F36000
G1 X119.085 Y120.33 Z2.68
G1 Z2.28
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.546076
G1 F3600
M204 S5000
G1 X119.2 Y120.199 E.00582
; LINE_WIDTH: 0.544336
G1 X120.858 Y118.22 E.08582
G2 X122.233 Y116.575 I-33.094 J-29.045 E.07129
; WIPE_START
M204 S10000
G1 X121.592 Y117.342 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X126.839 Y111.8 Z2.68 F36000
G1 X129.538 Y108.949 Z2.68
G1 Z2.28
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.448 Y111.023 I-31.651 J-15.3 E.08945
G2 X129.887 Y109.92 I-2.428 J-4.659 E.06956
G2 X131.772 Y107.229 I-17.921 J-14.562 E.12554
G3 X133.186 Y106.472 I1.57 J1.231 E.06302
G3 X137.428 Y108.365 I-.103 J5.929 E.18221
G3 X139.313 Y111.056 I-17.92 J14.561 E.12554
G2 X140.256 Y111.736 I1.506 J-1.092 E.04515
G2 X141.945 Y111.655 I.686 J-3.318 E.06528
G1 X141.945 Y118.133 E.24732
G3 X141.198 Y117.461 I2.625 J-3.67 E.03844
G3 X139.313 Y114.77 I17.918 J-14.56 E.12554
G2 X137.899 Y114.013 I-1.57 J1.231 E.06302
G2 X133.658 Y115.905 I.103 J5.929 E.18221
G2 X131.772 Y118.596 I17.918 J14.56 E.12554
G3 X130.359 Y119.354 I-1.57 J-1.231 E.06302
G3 X126.513 Y117.846 I.101 J-5.915 E.16108
G3 X126.073 Y119.566 I-2.474 J.283 E.06932
G1 X122.833 Y123.44 E.19279
G3 X122.564 Y123.709 I-1.071 J-.802 E.0146
G3 X124.232 Y126.137 I-20.184 J15.656 E.11251
G2 X125.646 Y126.894 I1.57 J-1.231 E.06302
G2 X129.887 Y125.001 I-.103 J-5.929 E.18221
G2 X131.772 Y122.311 I-17.921 J-14.562 E.12554
G3 X133.186 Y121.553 I1.57 J1.231 E.06302
G3 X137.428 Y123.446 I-.103 J5.929 E.18221
G3 X139.313 Y126.137 I-17.922 J14.562 E.12554
G2 X140.256 Y126.818 I1.506 J-1.092 E.04515
G2 X141.945 Y126.736 I.686 J-3.319 E.06528
G1 X141.945 Y133.214 E.24732
G3 X141.198 Y132.542 I2.624 J-3.669 E.03844
G3 X139.313 Y129.851 I17.918 J-14.56 E.12554
G2 X137.899 Y129.094 I-1.57 J1.231 E.06302
G2 X133.658 Y130.987 I.103 J5.929 E.18221
G2 X131.772 Y133.677 I17.918 J14.56 E.12554
G3 X130.359 Y134.435 I-1.57 J-1.231 E.06302
G3 X126.117 Y132.542 I.103 J-5.929 E.18221
G3 X124.232 Y129.851 I17.921 J-14.562 E.12554
G2 X122.818 Y129.094 I-1.57 J1.231 E.06302
G2 X119.73 Y130.045 I.064 J5.698 E.12507
G3 X121.113 Y135.985 I-5.785 J4.478 E.24021
G3 X129.299 Y140.632 I-25.249 J54.021 E.35975
G2 X129.887 Y140.083 I-2.154 J-2.9 E.03079
G2 X131.772 Y137.392 I-17.919 J-14.56 E.12554
G3 X133.186 Y136.635 I1.57 J1.231 E.06302
G3 X137.428 Y138.527 I-.103 J5.929 E.18221
G3 X139.313 Y141.218 I-17.922 J14.562 E.12554
G2 X140.256 Y141.899 I1.506 J-1.092 E.04515
G2 X141.945 Y141.817 I.686 J-3.318 E.06528
G1 X141.945 Y148.295 E.24732
G3 X141.198 Y147.623 I2.624 J-3.669 E.03844
G3 X139.313 Y144.933 I17.918 J-14.56 E.12554
G2 X137.899 Y144.175 I-1.57 J1.231 E.06302
G2 X134.935 Y145.056 I.008 J5.454 E.11971
G3 X136.654 Y146.647 I-19.858 J23.174 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 2.44
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X135.92 Y145.968 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L15
M991 S0 P14 ;notify layer change


G17
G3 Z2.68 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.755 Y119.14
G1 Z2.44
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.623 Y119.338 E.00909
G1 X122.418 Y123.161 E.19047
G1 X122.013 Y123.526 E.0208
G1 X121.593 Y123.74 E.018
G1 X121.087 Y123.851 E.01978
G1 X120.53 Y123.818 E.02128
G1 X120.084 Y123.668 E.01798
G3 X119.017 Y122.855 I3.453 J-5.635 E.05132
G3 X114.848 Y126.662 I-38.026 J-37.451 E.21567
G1 X118.015 Y129.012 E.15054
G3 X118.933 Y129.849 I-4.608 J5.97 E.0475
G1 X119.567 Y130.649 E.03895
G1 X120.075 Y131.529 E.03881
G1 X120.45 Y132.48 E.03904
G3 X120.765 Y134.489 I-7.456 J2.196 E.07787
G1 X120.698 Y135.504 E.03882
G1 X120.538 Y136.268 E.0298
G3 X130.508 Y142.101 I-22.165 J49.313 E.44184
G3 X142.443 Y154.07 I-32.637 J44.484 E.64788
G1 X142.443 Y104.557 E1.89037
G1 X142.285 Y104.646 E.00693
G1 X141.684 Y104.741 E.02324
G1 X132.907 Y104.741 E.33511
G1 X132.614 Y104.719 E.0112
G1 X132.09 Y104.562 E.0209
G1 X131.648 Y104.281 E.01998
G1 X131.242 Y103.806 E.02385
G3 X124.727 Y116.043 I-51.286 J-19.454 E.53068
G3 X125.533 Y116.732 I-6.064 J7.899 E.04051
G1 X125.841 Y117.153 E.0199
G1 X126.031 Y117.659 E.02064
G1 X126.078 Y118.104 E.0171
G1 X126.019 Y118.563 E.01767
G1 X125.859 Y118.983 E.01716
G1 X125.805 Y119.065 E.00373
G1 X125.261 Y118.831 F36000
G1 F13446.369
G1 X125.174 Y118.961 E.006
G1 X121.969 Y122.785 E.19047
G1 X121.686 Y123.04 E.01456
G1 X121.392 Y123.19 E.01259
G1 X121.038 Y123.268 E.01384
G1 X120.6 Y123.234 E.01678
G1 X120.336 Y123.14 E.01067
G1 X119.918 Y122.845 E.01954
G1 X118.978 Y122.058 E.04681
G3 X113.893 Y126.683 I-39.187 J-37.97 E.26262
G1 X117.666 Y129.482 E.17935
G3 X118.523 Y130.268 I-4.173 J5.412 E.04448
G1 X119.1 Y131.003 E.03566
G1 X119.562 Y131.811 E.03553
G1 X119.901 Y132.684 E.03575
G3 X120.135 Y133.792 I-9.623 J2.611 E.04329
G1 X120.18 Y134.525 E.02801
G1 X120.115 Y135.453 E.03554
G1 X119.931 Y136.289 E.03268
G1 X119.835 Y136.599 E.01237
G3 X130.932 Y143.157 I-21.13 J48.423 E.49334
G3 X142.92 Y155.769 I-32.176 J42.588 E.66731
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.021 E1.97489
; LINE_WIDTH: 0.620646
G1 F13431.463
G1 X143.029 Y103.282 E.02826
; LINE_WIDTH: 0.621136
G1 F13420.247
G1 X143.028 Y103.027 E.00975
G1 X142.962 Y103.27 E.00963
; LINE_WIDTH: 0.620646
G1 F13431.463
G1 X142.582 Y103.818 E.0255
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.105 Y104.089 E.02094
G1 X141.684 Y104.155 E.01626
G1 X132.907 Y104.155 E.33511
G1 X132.702 Y104.14 E.00784
G1 X132.309 Y104.017 E.01573
G1 X132.026 Y103.833 E.01287
G1 F12470.777
G1 X131.739 Y103.497 E.01687
; LINE_WIDTH: 0.666098
G1 F10988.707
G1 X131.699 Y103.399 E.00439
; LINE_WIDTH: 0.7122
G1 F10645.561
G1 X131.659 Y103.3 E.00471
; LINE_WIDTH: 0.758302
G1 F10307.829
G1 X131.619 Y103.202 E.00503
; LINE_WIDTH: 0.804404
G1 F9975.57
G1 X131.578 Y103.103 E.00535
; LINE_WIDTH: 0.850506
G1 F9648.726
G1 X131.538 Y103.004 E.00567
; LINE_WIDTH: 0.896608
G1 F9132.849
G1 X131.498 Y102.906 E.00599
; LINE_WIDTH: 0.94271
G1 F8669.337
G1 X131.458 Y102.807 E.00631
; LINE_WIDTH: 0.988812
G1 F8250.6
G1 X131.417 Y102.709 E.00663
; LINE_WIDTH: 1.03491
G1 F7870.451
G1 X131.377 Y102.61 E.00695
; LINE_WIDTH: 1.08102
G1 F7523.789
G1 X131.337 Y102.511 E.00727
G1 X131.294 Y102.561 E.00449
; LINE_WIDTH: 1.03491
G1 F7870.451
G1 X131.251 Y102.611 E.00429
; LINE_WIDTH: 0.988812
G1 F8250.6
G1 X131.208 Y102.661 E.00409
; LINE_WIDTH: 0.94271
G1 F8669.337
G1 X131.166 Y102.711 E.00389
; LINE_WIDTH: 0.896608
G1 F9132.849
G1 X131.123 Y102.761 E.00369
; LINE_WIDTH: 0.850506
G1 F9648.726
G1 X131.08 Y102.811 E.0035
; LINE_WIDTH: 0.804404
G1 F10226.371
G1 X131.037 Y102.861 E.0033
; LINE_WIDTH: 0.758302
G1 F10433.357
G1 X130.995 Y102.911 E.0031
; LINE_WIDTH: 0.7122
G1 F10642.417
G1 X130.952 Y102.961 E.0029
; LINE_WIDTH: 0.666098
G1 F10853.55
G1 X130.909 Y103.011 E.00271
; LINE_WIDTH: 0.619996
G1 F12183.114
G1 X130.766 Y103.384 E.01527
G1 F13446.369
G1 X130.622 Y103.758 E.01527
G1 X130.204 Y104.833 E.04405
G3 X123.935 Y116.143 I-50.424 J-20.557 E.49487
G1 X125.005 Y117.04 E.05328
G1 X125.326 Y117.433 E.0194
G1 X125.477 Y117.882 E.01809
G1 X125.472 Y118.321 E.01675
G1 X125.339 Y118.714 E.01583
G1 X125.311 Y118.756 E.00193
G1 X124.776 Y118.495 F36000
G1 F13446.369
G1 X124.725 Y118.585 E.00394
G1 X121.52 Y122.409 E.19047
G1 X121.295 Y122.593 E.0111
G1 X120.989 Y122.684 E.0122
G1 X120.677 Y122.647 E.01198
G1 X120.332 Y122.429 E.01559
G1 X118.934 Y121.257 E.06964
G3 X112.93 Y126.698 I-39.178 J-37.201 E.30966
G1 X117.317 Y129.952 E.20855
G1 X118.114 Y130.687 E.04141
G1 X118.634 Y131.357 E.03237
G1 X119.048 Y132.093 E.03225
G1 X119.352 Y132.888 E.03247
G3 X119.551 Y133.828 I-11.752 J2.973 E.03671
G1 X119.596 Y134.56 E.02801
G1 X119.531 Y135.403 E.03227
G1 X119.361 Y136.154 E.0294
G1 X119.08 Y136.91 E.0308
G3 X130.579 Y143.624 I-20.589 J48.468 E.50972
G3 X142.648 Y156.415 I-31.78 J42.074 E.67457
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.516 E2.16551
G3 X141.879 Y99.568 I-1.239 J-12.24 E.06635
G1 X142.451 Y102.657 E.11993
G3 X142.413 Y103.064 I-.767 J.135 E.01581
G1 X142.196 Y103.377 E.01454
G1 X141.924 Y103.532 E.01195
G1 X141.684 Y103.57 E.00928
G1 X132.907 Y103.57 E.33511
G1 X132.565 Y103.491 E.01337
G3 X132.144 Y102.945 I.432 J-.77 E.02707
G1 X132.155 Y102.575 E.01412
G1 X132.712 Y99.569 E.11673
G3 X131.452 Y99.49 I1.136 J-28.29 E.0482
G3 X123.14 Y116.241 I-51.468 J-15.103 E.71758
G1 X124.628 Y117.488 E.07417
G1 X124.812 Y117.713 E.01107
G1 X124.905 Y118.031 E.01264
G1 X124.866 Y118.334 E.01168
G1 X124.82 Y118.417 E.00361
M204 S250
G1 X124.301 Y118.23 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.096 Y122.053 E.15795
G1 X120.942 Y122.133 E.00549
G3 X120.671 Y121.992 I.036 J-.4 E.00993
G1 X119.139 Y120.707 E.06332
; LINE_WIDTH: 0.549136
G1 X119.045 Y120.375 E.01156
G1 X118.674 Y120.73 E.01723
; LINE_WIDTH: 0.520206
G1 X117.999 Y121.433 E.03086
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.906 J-39.309 E.25281
G1 X116.988 Y130.396 E.19622
G1 X117.728 Y131.083 E.03197
G1 X118.194 Y131.692 E.02427
G1 X118.564 Y132.36 E.02418
G1 X118.834 Y133.08 E.02436
G3 X118.999 Y133.862 I-26.437 J5.991 E.0253
G1 X119.044 Y134.599 E.02337
G1 X118.98 Y135.355 E.02405
G1 X118.824 Y136.026 E.02181
M73 P52 R9
G1 X118.548 Y136.753 E.0246
G1 X118.31 Y137.19 E.01575
G3 X130.249 Y144.067 I-19.572 J47.777 E.43752
G3 X142.39 Y157.025 I-31.646 J41.818 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.228 E1.81941
G1 X144.167 Y98.939 E.00916
G3 X141.215 Y99.016 I-2.198 J-27.356 E.09356
G1 X141.906 Y102.752 E.12029
G1 X141.895 Y102.87 E.00376
G1 X141.754 Y103.006 E.00621
G1 X132.907 Y103.017 E.2801
G1 X132.761 Y102.964 E.00491
G1 X132.685 Y102.836 E.0047
G3 X133.012 Y100.984 I64.473 J10.409 E.05955
G1 X133.376 Y99.017 E.06332
G3 X132.131 Y98.993 I-.315 J-15.928 E.03944
G2 X131.038 Y98.936 I-.886 J6.546 E.0347
G3 X122.631 Y116 I-50.902 J-14.475 E.60554
; LINE_WIDTH: 0.545776
G1 X122.27 Y116.528 E.0213
G1 X122.611 Y116.519 E.01138
; LINE_WIDTH: 0.520156
G1 X123.451 Y117.223 E.0347
; LINE_WIDTH: 0.519996
G1 X124.273 Y117.912 E.03398
G1 X124.353 Y118.069 E.00559
G1 X124.329 Y118.144 E.0025
; WIPE_START
M204 S10000
G1 X123.692 Y118.915 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.589 Y114.07 Z2.84 F36000
G1 X143.028 Y103.027 Z2.84
G1 Z2.44
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.623456
G1 F13367.397
G1 X143.027 Y102.543 E.01861
; LINE_WIDTH: 0.659506
G1 F12596.574
G1 X143.009 Y102.346 E.00804
; LINE_WIDTH: 0.705222
G1 F11738.206
G1 X142.986 Y102.097 E.01093
; LINE_WIDTH: 0.750939
G1 F10989.356
G1 X142.963 Y101.848 E.01168
; LINE_WIDTH: 0.796655
G1 F10330.325
G1 X142.941 Y101.599 E.01242
; LINE_WIDTH: 0.842371
G1 F9745.866
G1 X142.918 Y101.35 E.01317
; LINE_WIDTH: 0.888088
G1 F9223.998
G1 X142.895 Y101.101 E.01391
; LINE_WIDTH: 0.933804
G1 F8755.181
G1 X142.872 Y100.853 E.01466
; LINE_WIDTH: 0.97952
G1 F8331.713
G1 X142.849 Y100.604 E.0154
; LINE_WIDTH: 1.02524
G1 F7947.321
G1 X142.826 Y100.355 E.01615
; WIPE_START
G1 X142.849 Y100.604 E-.095
G1 X142.872 Y100.853 E-.095
G1 X142.895 Y101.101 E-.095
G1 X142.918 Y101.35 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.323 Y102.112 Z2.84 F36000
G1 X131.337 Y102.511 Z2.84
G1 Z2.44
G1 E.4 F1800
; LINE_WIDTH: 1.08102
G1 F7523.789
G1 X131.359 Y102.428 E.0059
; LINE_WIDTH: 1.0669
G1 F7626.675
G1 X131.444 Y102.104 E.02254
; LINE_WIDTH: 1.01834
G1 F8002.995
G1 X131.529 Y101.78 E.02148
; LINE_WIDTH: 0.96979
G1 F8418.379
G1 X131.613 Y101.456 E.02042
; LINE_WIDTH: 0.921236
G1 F8879.243
G1 X131.692 Y101.142 E.01873
; LINE_WIDTH: 0.881081
G1 F9300.322
G1 X131.771 Y100.827 E.01789
; LINE_WIDTH: 0.840926
G1 F9763.326
G1 X131.849 Y100.513 E.01704
; LINE_WIDTH: 0.800771
G1 F10274.845
G1 X131.928 Y100.199 E.01619
; LINE_WIDTH: 0.760616
G1 F10842.926
G1 X131.931 Y100.188 E.00051
; WIPE_START
G1 X131.928 Y100.199 E-.00409
G1 X131.849 Y100.513 E-.12313
G1 X131.771 Y100.827 E-.12313
G1 X131.692 Y101.142 E-.12313
G1 X131.688 Y101.158 E-.00651
; WIPE_END
G1 E-.02 F1800
G1 X127.493 Y107.535 Z2.84 F36000
G1 X119.045 Y120.375 Z2.84
G1 Z2.44
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.549136
G1 F3600
M204 S5000
G1 X119.2 Y120.198 E.00789
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15231
; LINE_WIDTH: 0.545776
G1 X122.27 Y116.528 E.00682
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.07771
G1 X121.632 Y117.297 E-.30229
; WIPE_END
G1 E-.02 F1800
G1 X126.872 Y111.748 Z2.84 F36000
G1 X129.563 Y108.897 Z2.84
G1 Z2.44
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.484 Y110.976 I-37.97 J-18.402 E.08944
G1 X128.945 Y110.74 E.01976
G1 X129.887 Y109.984 E.04615
G2 X131.772 Y107.541 I-15.799 J-14.139 E.11793
G3 X133.186 Y106.684 I1.9 J1.542 E.06437
G3 X137.428 Y108.301 I.25 J5.715 E.17821
G3 X139.313 Y110.744 I-15.801 J14.141 E.11793
G2 X141.945 Y111.538 I1.906 J-1.56 E.11136
G1 X141.945 Y118.124 E.25145
G1 X141.198 Y117.524 E.03658
G3 X139.313 Y115.081 I15.798 J-14.138 E.11793
G2 X137.899 Y114.225 I-1.9 J1.542 E.06437
G2 X133.658 Y115.842 I-.25 J5.715 E.17821
G2 X131.772 Y118.285 I15.796 J14.137 E.11793
G3 X130.359 Y119.141 I-1.9 J-1.542 E.06437
G3 X126.554 Y117.875 I-.273 J-5.528 E.15665
G3 X126.112 Y119.52 I-2.593 J.185 E.06624
G1 X122.612 Y123.676 E.20745
G3 X124.232 Y125.826 I-18.61 J15.706 E.1028
G2 X125.646 Y126.682 I1.9 J-1.542 E.06437
G2 X129.887 Y125.065 I.25 J-5.715 E.17821
G2 X131.772 Y122.622 I-15.8 J-14.14 E.11793
G3 X133.186 Y121.766 I1.9 J1.542 E.06437
G3 X137.428 Y123.382 I.25 J5.716 E.17821
G3 X139.313 Y125.826 I-15.801 J14.141 E.11793
G2 X141.945 Y126.619 I1.906 J-1.561 E.11136
G1 X141.945 Y133.205 E.25145
G1 X141.198 Y132.606 E.03658
G3 X139.313 Y130.162 I15.796 J-14.137 E.11793
G2 X137.899 Y129.306 I-1.9 J1.542 E.06437
G2 X133.658 Y130.923 I-.25 J5.715 E.17821
G2 X131.772 Y133.366 I15.798 J14.138 E.11793
G3 X130.359 Y134.223 I-1.9 J-1.542 E.06437
G3 X126.117 Y132.606 I-.25 J-5.715 E.17821
G3 X124.232 Y130.162 I15.799 J-14.139 E.11793
G2 X122.818 Y129.306 I-1.9 J1.542 E.06437
G2 X119.738 Y130.055 I-.38 J5.146 E.12302
G3 X121.111 Y135.983 I-5.885 J4.486 E.23943
G3 X129.288 Y140.627 I-25.153 J53.816 E.35943
G1 X129.887 Y140.146 E.02932
G2 X131.772 Y137.703 I-15.8 J-14.14 E.11793
G3 X133.186 Y136.847 I1.9 J1.542 E.06437
G3 X137.428 Y138.464 I.25 J5.715 E.17821
G3 X139.313 Y140.907 I-15.8 J14.14 E.11793
G2 X141.945 Y141.701 I1.906 J-1.561 E.11136
G1 X141.945 Y148.287 E.25145
G1 X141.198 Y147.687 E.03658
G3 X139.313 Y145.244 I15.798 J-14.138 E.11793
G2 X137.899 Y144.387 I-1.9 J1.542 E.06437
G2 X134.95 Y145.069 I-.418 J4.91 E.11748
G3 X136.673 Y146.656 I-33.571 J38.173 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 2.6
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X135.937 Y145.979 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L16
M991 S0 P15 ;notify layer change


G17
G3 Z2.84 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.787 Y119.1
G1 Z2.6
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.654 Y119.3 E.00918
G1 X122.386 Y123.198 E.19421
G1 X122.037 Y123.524 E.01821
G1 X121.575 Y123.772 E.02003
G1 X121.141 Y123.88 E.01707
G1 X120.74 Y123.89 E.01532
G1 X120.267 Y123.793 E.01846
G1 X119.639 Y123.44 E.02746
G1 X118.983 Y122.89 E.03273
G3 X114.848 Y126.662 I-39.289 J-38.905 E.21379
G1 X118.015 Y129.012 E.15055
G3 X119.01 Y129.934 I-4.232 J5.567 E.05188
G1 X119.571 Y130.654 E.03486
G1 X120.048 Y131.475 E.03624
G1 X120.445 Y132.465 E.04072
G1 X120.656 Y133.35 E.03474
G3 X120.765 Y134.489 I-5.848 J1.135 E.04376
G1 X120.698 Y135.506 E.03891
G1 X120.538 Y136.268 E.02973
G3 X130.507 Y142.101 I-22.114 J49.231 E.44181
G3 X142.443 Y154.069 I-32.644 J44.494 E.64787
G1 X142.443 Y104.56 E1.8902
G1 X142.288 Y104.648 E.00682
G1 X141.687 Y104.743 E.02324
G1 X132.904 Y104.743 E.33531
G1 X132.611 Y104.721 E.01122
G1 X132.087 Y104.563 E.02091
G1 X131.645 Y104.282 E.02
G1 X131.241 Y103.81 E.02372
G3 X124.755 Y116.002 I-51.264 J-19.448 E.52865
G1 X125.412 Y116.553 E.03274
G1 X125.843 Y117.063 E.02547
G1 X126.041 Y117.537 E.01961
G1 X126.11 Y118.048 E.01968
G1 X126.05 Y118.524 E.01833
G1 X125.891 Y118.945 E.0172
G1 X125.837 Y119.025 E.00368
G1 X125.33 Y118.72 F36000
G1 F13446.369
G1 X125.265 Y118.847 E.00545
G3 X124.507 Y119.757 I-13.093 J-10.139 E.04522
G1 X121.938 Y122.822 E.15272
G1 X121.647 Y123.082 E.01487
G1 X121.227 Y123.268 E.01757
G1 X120.786 Y123.306 E.01689
G1 X120.366 Y123.205 E.01648
G1 X119.888 Y122.884 E.02196
G1 X118.944 Y122.093 E.04704
G3 X113.893 Y126.683 I-41.891 J-41.022 E.26073
G1 X117.666 Y129.482 E.17936
G3 X118.588 Y130.339 I-3.854 J5.068 E.04815
G1 X119.103 Y131.007 E.03219
G1 X119.536 Y131.759 E.03313
G3 X120.136 Y133.793 I-6.606 J3.053 E.08126
G1 X120.18 Y134.525 E.02798
G1 X120.114 Y135.456 E.03563
G1 X119.933 Y136.284 E.03237
G1 X119.836 Y136.599 E.01259
G3 X130.931 Y143.156 I-21.126 J48.417 E.4933
G3 X142.92 Y155.769 I-32.374 J42.778 E.66733
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.284 E2.00306
; LINE_WIDTH: 0.619206
G1 F13464.532
G1 X143.03 Y103.029 E.00971
G1 X142.964 Y103.272 E.00959
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.584 Y103.821 E.02547
G1 X142.107 Y104.091 E.02094
G1 X141.687 Y104.158 E.01626
G1 X132.904 Y104.158 E.33531
G1 X132.699 Y104.142 E.00785
G1 X132.306 Y104.019 E.01574
G1 X132.023 Y103.835 E.01288
G1 F12480.853
G1 X131.736 Y103.498 E.01689
; LINE_WIDTH: 0.665932
G1 F10996.773
G1 X131.696 Y103.4 E.00438
; LINE_WIDTH: 0.711868
G1 F10654.096
G1 X131.656 Y103.301 E.0047
; LINE_WIDTH: 0.757804
G1 F10316.826
G1 X131.616 Y103.203 E.00501
; LINE_WIDTH: 0.80374
G1 F9984.98
G1 X131.576 Y103.104 E.00533
; LINE_WIDTH: 0.849676
G1 F9658.548
G1 X131.536 Y103.006 E.00565
; LINE_WIDTH: 0.895612
G1 F9143.411
G1 X131.496 Y102.908 E.00597
; LINE_WIDTH: 0.941548
G1 F8680.441
G1 X131.455 Y102.809 E.00629
; LINE_WIDTH: 0.987484
G1 F8262.096
G1 X131.415 Y102.711 E.00661
; LINE_WIDTH: 1.03342
G1 F7882.221
G1 X131.375 Y102.612 E.00692
; LINE_WIDTH: 1.07936
G1 F7535.741
G1 X131.335 Y102.514 E.00724
G1 X131.292 Y102.564 E.00447
; LINE_WIDTH: 1.03342
G1 F7882.221
G1 X131.25 Y102.613 E.00427
; LINE_WIDTH: 0.987484
G1 F8262.096
G1 X131.207 Y102.663 E.00407
; LINE_WIDTH: 0.941548
G1 F8680.441
G1 X131.165 Y102.713 E.00388
; LINE_WIDTH: 0.895612
G1 F9143.411
G1 X131.122 Y102.763 E.00368
; LINE_WIDTH: 0.849676
G1 F9658.548
G1 X131.079 Y102.813 E.00349
; LINE_WIDTH: 0.80374
G1 F10235.196
G1 X131.037 Y102.862 E.00329
; LINE_WIDTH: 0.757804
G1 F10441.783
G1 X130.994 Y102.912 E.00309
; LINE_WIDTH: 0.711868
G1 F10650.434
G1 X130.951 Y102.962 E.0029
; LINE_WIDTH: 0.665932
G1 F10861.149
G1 X130.909 Y103.012 E.0027
; LINE_WIDTH: 0.619996
G1 F12191.165
G1 X130.765 Y103.385 E.01527
G1 F13446.369
G1 X130.622 Y103.759 E.01527
G1 X130.206 Y104.834 E.044
G3 X123.964 Y116.103 I-50.061 J-20.363 E.49303
G1 X125.036 Y117.002 E.05341
G1 X125.352 Y117.384 E.01892
G1 X125.476 Y117.69 E.01261
G1 X125.524 Y118.059 E.0142
G1 X125.452 Y118.483 E.01642
G1 X125.372 Y118.64 E.00674
G1 X124.807 Y118.457 F36000
G1 F13446.369
G1 X124.756 Y118.548 E.00395
G1 X121.489 Y122.446 E.19421
G1 X121.218 Y122.653 E.01301
G1 X120.913 Y122.724 E.01193
G1 X120.592 Y122.664 E.01249
G1 X120.222 Y122.4 E.01737
G1 X118.9 Y121.292 E.06583
G3 X112.93 Y126.698 I-40.926 J-39.202 E.30776
G1 X117.317 Y129.952 E.20856
G3 X118.166 Y130.745 I-3.472 J4.567 E.04442
G1 X118.636 Y131.359 E.02952
G1 X119.024 Y132.043 E.03002
G1 X119.35 Y132.88 E.03429
G1 X119.51 Y133.592 E.02786
G1 X119.596 Y134.56 E.03711
G1 X119.531 Y135.405 E.03235
G1 X119.363 Y136.148 E.02909
G1 X119.08 Y136.91 E.03102
G3 X130.578 Y143.623 I-20.518 J48.347 E.50968
G3 X142.647 Y156.415 I-31.963 J42.247 E.67459
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.516 E2.16548
G1 X142.963 Y99.566 E.02494
G1 X141.881 Y99.57 E.0413
G1 X142.453 Y102.659 E.11993
G3 X142.416 Y103.066 I-.767 J.135 E.01581
G1 X142.199 Y103.38 E.01454
G1 X141.927 Y103.534 E.01195
G1 X141.687 Y103.572 E.00928
G1 X132.904 Y103.572 E.33531
G1 X132.563 Y103.493 E.01338
G3 X132.141 Y102.947 I.432 J-.77 E.02707
G1 X132.153 Y102.575 E.01421
G1 X132.709 Y99.571 E.11663
G3 X131.495 Y99.491 I.185 J-11.99 E.04646
G1 X131.441 Y99.528 E.00252
G3 X123.169 Y116.201 I-51.449 J-15.138 E.71423
G1 X124.66 Y117.451 E.07429
G1 X124.84 Y117.669 E.0108
G1 X124.936 Y117.992 E.01287
G1 X124.897 Y118.296 E.01171
G1 X124.851 Y118.379 E.00362
M204 S250
G1 X124.333 Y118.193 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.065 Y122.091 E.16105
G1 X120.921 Y122.17 E.0052
G1 X120.747 Y122.119 E.00573
G1 X119.811 Y121.334 E.03867
; LINE_WIDTH: 0.521466
G1 X119.084 Y120.727 E.03008
; LINE_WIDTH: 0.551576
G1 X119.013 Y120.412 E.01088
G1 X117.999 Y121.433 E.0485
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.725 J-39.104 E.25281
G1 X116.988 Y130.396 E.19623
G3 X118.194 Y131.692 I-3.629 J4.588 E.05627
G1 X118.54 Y132.312 E.02247
G1 X118.833 Y133.076 E.02591
G1 X118.969 Y133.706 E.02042
G1 X119.044 Y134.594 E.0282
G1 X118.98 Y135.357 E.02426
G1 X118.825 Y136.021 E.02156
G1 X118.548 Y136.753 E.02479
G1 X118.31 Y137.19 E.01575
G3 X130.248 Y144.066 I-19.805 J48.181 E.43746
G3 X142.39 Y157.025 I-31.645 J41.818 E.56496
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.228 E1.81942
G1 X144.167 Y98.939 E.00914
G1 X142.921 Y99.015 E.03953
G3 X141.217 Y99.019 I-1.146 J-132.169 E.05395
G1 X141.909 Y102.754 E.12029
G1 X141.898 Y102.873 E.00376
G1 X141.756 Y103.008 E.00621
G1 X132.904 Y103.019 E.28026
G1 X132.758 Y102.966 E.00491
G1 X132.683 Y102.838 E.0047
G3 X133.009 Y100.986 I64.814 J10.471 E.05955
G1 X133.374 Y99.019 E.06332
G3 X132.131 Y98.994 I-.314 J-15.504 E.03936
G2 X131.038 Y98.936 I-.886 J6.381 E.0347
G3 X122.631 Y116 I-51.1 J-14.573 E.60552
; LINE_WIDTH: 0.548256
G1 X122.301 Y116.489 E.01977
G1 X122.643 Y116.481 E.01146
; LINE_WIDTH: 0.519996
G1 X124.305 Y117.875 E.06866
G1 X124.385 Y118.031 E.00558
G1 X124.36 Y118.107 E.00251
; WIPE_START
M204 S10000
G1 X123.723 Y118.877 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.622 Y114.035 Z3 F36000
G1 X143.03 Y103.029 Z3
G1 Z2.6
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.620936
G1 F13424.822
G1 X143.028 Y102.545 E.01851
; LINE_WIDTH: 0.657246
G1 F12642.277
G1 X143.01 Y102.347 E.00806
; LINE_WIDTH: 0.70296
G1 F11777.925
G1 X142.987 Y102.098 E.0109
; LINE_WIDTH: 0.748674
G1 F11024.202
G1 X142.965 Y101.849 E.01164
; LINE_WIDTH: 0.794388
G1 F10361.144
G1 X142.942 Y101.6 E.01239
; LINE_WIDTH: 0.840101
G1 F9773.322
G1 X142.919 Y101.351 E.01313
; LINE_WIDTH: 0.885815
G1 F9248.616
G1 X142.896 Y101.102 E.01388
; LINE_WIDTH: 0.931529
G1 F8777.382
G1 X142.873 Y100.853 E.01462
; LINE_WIDTH: 0.977242
G1 F8351.838
G1 X142.85 Y100.604 E.01537
; LINE_WIDTH: 1.02296
G1 F7965.649
G1 X142.827 Y100.356 E.01611
; WIPE_START
G1 X142.85 Y100.604 E-.095
G1 X142.873 Y100.853 E-.095
G1 X142.896 Y101.102 E-.095
G1 X142.919 Y101.351 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.325 Y102.113 Z3 F36000
G1 X131.335 Y102.514 Z3
G1 Z2.6
G1 E.4 F1800
; LINE_WIDTH: 1.07936
G1 F7535.741
G1 X131.358 Y102.428 E.00608
; LINE_WIDTH: 1.0648
G1 F7642.218
G1 X131.443 Y102.104 E.0225
; LINE_WIDTH: 1.01624
G1 F8020.112
G1 X131.528 Y101.78 E.02144
; LINE_WIDTH: 0.96769
G1 F8437.319
G1 X131.612 Y101.456 E.02038
; LINE_WIDTH: 0.919136
G1 F8900.317
G1 X131.691 Y101.141 E.01869
; LINE_WIDTH: 0.878976
G1 F9323.5
G1 X131.77 Y100.827 E.01784
; LINE_WIDTH: 0.838816
G1 F9788.933
G1 X131.848 Y100.512 E.017
; LINE_WIDTH: 0.798656
G1 F10303.277
G1 X131.927 Y100.198 E.01615
; LINE_WIDTH: 0.758496
G1 F10874.67
G1 X131.929 Y100.188 E.00048
; WIPE_START
G1 X131.927 Y100.198 E-.00384
G1 X131.848 Y100.512 E-.12315
G1 X131.77 Y100.827 E-.12315
G1 X131.691 Y101.141 E-.12315
G1 X131.687 Y101.158 E-.00671
; WIPE_END
G1 E-.02 F1800
G1 X127.49 Y107.534 Z3 F36000
G1 X119.013 Y120.412 Z3
G1 Z2.6
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.551576
G1 F3600
M204 S5000
G1 X119.2 Y120.198 E.00958
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15231
; LINE_WIDTH: 0.548256
G1 X122.301 Y116.489 E.00849
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.09629
G1 X121.663 Y117.26 E-.28371
; WIPE_END
G1 E-.02 F1800
G1 X126.902 Y111.709 Z3 F36000
G1 X129.572 Y108.88 Z3
G1 Z2.6
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.484 Y110.955 I-31.273 J-15.073 E.08945
G1 X128.945 Y110.744 E.01935
G1 X129.887 Y110.049 E.04469
G2 X131.772 Y107.801 I-13.375 J-13.128 E.11214
G3 X133.186 Y106.877 I2.247 J1.895 E.06541
G3 X137.428 Y108.236 I.634 J5.322 E.17536
G3 X139.313 Y110.484 I-13.377 J13.13 E.11214
G2 X141.945 Y111.431 I2.092 J-1.684 E.11234
M73 P53 R9
G1 X141.945 Y118.14 E.25616
G1 X141.198 Y117.59 E.03542
G3 X139.313 Y115.342 I13.373 J-13.126 E.11214
G2 X137.899 Y114.418 I-2.247 J1.895 E.06541
G2 X133.658 Y115.776 I-.634 J5.322 E.17536
G2 X131.772 Y118.025 I13.374 J13.127 E.11214
G3 X130.359 Y118.949 I-2.247 J-1.895 E.06541
G3 X126.601 Y117.947 I-.645 J-5.126 E.15222
G3 X126.144 Y119.482 I-2.708 J.029 E.0621
G1 X122.658 Y123.632 E.20692
G3 X124.232 Y125.565 I-16.505 J15.045 E.09521
G2 X125.646 Y126.489 I2.247 J-1.895 E.06541
G2 X129.887 Y125.131 I.634 J-5.322 E.17536
G2 X131.772 Y122.882 I-13.375 J-13.128 E.11214
G3 X133.186 Y121.958 I2.247 J1.895 E.06541
G3 X137.428 Y123.317 I.634 J5.322 E.17536
G3 X139.313 Y125.565 I-13.376 J13.128 E.11214
G2 X141.945 Y126.512 I2.092 J-1.684 E.11234
G1 X141.945 Y133.222 E.25616
G1 X141.198 Y132.671 E.03542
G3 X139.313 Y130.423 I13.374 J-13.127 E.11214
G2 X137.899 Y129.499 I-2.247 J1.895 E.06541
G2 X133.658 Y130.857 I-.634 J5.322 E.17536
G2 X131.772 Y133.106 I13.373 J13.126 E.11214
G3 X130.359 Y134.03 I-2.247 J-1.895 E.06541
G3 X126.117 Y132.671 I-.634 J-5.322 E.17536
G3 X124.232 Y130.423 I13.376 J-13.129 E.11214
G2 X122.818 Y129.499 I-2.247 J1.895 E.06541
G2 X119.744 Y130.06 I-.723 J4.739 E.12151
G3 X121.113 Y135.985 I-6.048 J4.517 E.23897
G3 X129.307 Y140.64 I-25.226 J53.95 E.36016
G1 X129.887 Y140.212 E.02753
G2 X131.772 Y137.963 I-13.377 J-13.129 E.11214
G3 X133.186 Y137.04 I2.247 J1.895 E.06541
G3 X137.428 Y138.398 I.634 J5.322 E.17536
G3 X139.313 Y140.647 I-13.376 J13.128 E.11214
G2 X141.945 Y141.593 I2.092 J-1.684 E.11234
G1 X141.945 Y148.303 E.25616
G1 X141.198 Y147.753 E.03542
G3 X139.313 Y145.504 I13.375 J-13.127 E.11214
G2 X138.371 Y144.745 I-2.296 J1.888 E.04652
G2 X134.969 Y145.076 I-1.327 J4.02 E.13432
G3 X136.682 Y146.673 I-26.996 J30.67 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 2.76
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X135.951 Y145.991 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L17
M991 S0 P16 ;notify layer change


G17
G3 Z3 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.812 Y119.07
G1 Z2.76
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.679 Y119.271 E.00919
G1 X122.362 Y123.228 E.19713
G1 X121.969 Y123.584 E.02025
G1 X121.548 Y123.802 E.0181
G1 X121.042 Y123.917 E.0198
G1 X120.484 Y123.887 E.02136
G1 X119.984 Y123.713 E.02021
G3 X118.955 Y122.917 I4.693 J-7.12 E.0497
G3 X114.848 Y126.662 I-39.378 J-39.06 E.21232
G1 X118.015 Y129.012 E.15056
G3 X118.932 Y129.849 I-4.61 J5.972 E.04746
G1 X119.565 Y130.646 E.03885
G1 X120.076 Y131.531 E.03904
G1 X120.45 Y132.479 E.0389
G1 X120.657 Y133.354 E.03434
G3 X120.765 Y134.489 I-5.877 J1.132 E.04358
G1 X120.698 Y135.506 E.03892
G1 X120.538 Y136.268 E.02972
G3 X130.506 Y142.1 I-21.912 J48.886 E.44179
G3 X142.443 Y154.069 I-32.644 J44.495 E.64792
G1 X142.443 Y104.564 E1.89007
G1 X142.29 Y104.65 E.00671
G1 X141.689 Y104.745 E.02324
G1 X132.902 Y104.745 E.3355
G1 X132.608 Y104.723 E.01124
G1 X132.083 Y104.565 E.02093
G1 X131.641 Y104.283 E.02001
G1 X131.239 Y103.814 E.02359
G3 X124.777 Y115.971 I-51.248 J-19.446 E.52703
G1 X125.437 Y116.524 E.03287
G1 X125.858 Y117.017 E.02476
G1 X126.063 Y117.495 E.01983
G1 X126.134 Y118.018 E.02018
G1 X126.075 Y118.493 E.01827
G1 X125.916 Y118.915 E.01723
G1 X125.862 Y118.996 E.00369
G1 X125.355 Y118.691 F36000
G1 F13446.369
G1 X125.29 Y118.818 E.00545
G3 X124.482 Y119.786 I-13.91 J-10.773 E.04815
G1 X121.913 Y122.852 E.15272
G1 X121.638 Y123.101 E.01417
G1 X121.344 Y123.254 E.01266
G1 X120.99 Y123.334 E.01385
G1 X120.55 Y123.302 E.01684
G1 X120.249 Y123.191 E.01224
G1 X119.826 Y122.882 E.02001
G1 X118.917 Y122.121 E.04526
G3 X113.893 Y126.683 I-38.358 J-37.194 E.25929
G1 X117.666 Y129.482 E.17937
G3 X118.523 Y130.268 I-4.175 J5.413 E.04444
G1 X119.099 Y131 E.03557
G1 X119.563 Y131.814 E.03575
G3 X120.114 Y135.456 I-6.103 J2.787 E.14247
G1 X119.932 Y136.285 E.03241
G1 X119.836 Y136.599 E.01254
G3 X130.932 Y143.157 I-21.322 J48.748 E.49333
G3 X142.92 Y155.769 I-32.375 J42.777 E.66728
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y108.023 E1.82211
G3 X143.03 Y103.285 I1547.285 J-2 E.18089
; LINE_WIDTH: 0.617746
G1 F13498.227
G1 X143.031 Y103.031 E.00968
G1 X142.966 Y103.274 E.00956
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.587 Y103.823 E.02546
G1 X142.11 Y104.093 E.02094
G1 X141.689 Y104.16 E.01626
G1 X132.902 Y104.16 E.3355
G1 X132.696 Y104.144 E.00786
G1 X132.303 Y104.021 E.01575
G1 X132.02 Y103.836 E.01289
G1 F12490.794
G1 X131.733 Y103.499 E.0169
; LINE_WIDTH: 0.665768
G1 F11004.733
G1 X131.693 Y103.401 E.00437
; LINE_WIDTH: 0.71154
G1 F10662.509
G1 X131.653 Y103.302 E.00468
; LINE_WIDTH: 0.757312
G1 F10325.691
G1 X131.613 Y103.204 E.005
; LINE_WIDTH: 0.803084
G1 F9994.279
G1 X131.573 Y103.106 E.00532
; LINE_WIDTH: 0.848856
G1 F9668.272
G1 X131.533 Y103.008 E.00564
; LINE_WIDTH: 0.894628
G1 F9153.869
G1 X131.493 Y102.909 E.00595
; LINE_WIDTH: 0.9404
G1 F8691.44
G1 X131.453 Y102.811 E.00627
; LINE_WIDTH: 0.986172
G1 F8273.484
G1 X131.413 Y102.713 E.00658
; LINE_WIDTH: 1.03194
G1 F7893.882
G1 X131.373 Y102.614 E.0069
; LINE_WIDTH: 1.07772
G1 F7547.585
G1 X131.333 Y102.516 E.00722
G1 X131.291 Y102.566 E.00445
; LINE_WIDTH: 1.03194
G1 F7893.882
G1 X131.248 Y102.615 E.00425
; LINE_WIDTH: 0.986172
G1 F8273.484
G1 X131.206 Y102.665 E.00406
; LINE_WIDTH: 0.9404
G1 F8691.44
G1 X131.163 Y102.715 E.00386
; LINE_WIDTH: 0.894628
G1 F9153.869
G1 X131.121 Y102.765 E.00367
; LINE_WIDTH: 0.848856
G1 F9668.272
G1 X131.078 Y102.814 E.00347
; LINE_WIDTH: 0.803084
G1 F10243.931
G1 X131.036 Y102.864 E.00328
; LINE_WIDTH: 0.757312
G1 F10450.097
G1 X130.993 Y102.914 E.00308
; LINE_WIDTH: 0.71154
G1 F10658.363
G1 X130.951 Y102.963 E.00289
; LINE_WIDTH: 0.665768
G1 F10868.638
G1 X130.908 Y103.013 E.00269
; LINE_WIDTH: 0.619996
G1 F12199.099
G1 X130.765 Y103.387 E.01527
G1 F13446.369
G1 X130.622 Y103.76 E.01527
G1 X130.206 Y104.834 E.04395
G3 X123.986 Y116.072 I-50.068 J-20.366 E.49157
G1 X125.061 Y116.973 E.05351
G1 X125.37 Y117.343 E.01842
G1 X125.498 Y117.652 E.01277
G1 X125.548 Y118.028 E.0145
G1 X125.477 Y118.453 E.01645
G1 X125.396 Y118.61 E.00676
G1 X124.828 Y118.441 F36000
G1 F13446.369
G1 X124.781 Y118.518 E.00345
G1 X121.464 Y122.475 E.19713
G1 X121.195 Y122.681 E.01292
G1 X120.937 Y122.751 E.01021
G1 X120.614 Y122.712 E.01241
G1 X120.367 Y122.572 E.01084
G1 X118.874 Y121.32 E.07441
G3 X112.93 Y126.698 I-38.47 J-36.545 E.30633
G1 X117.317 Y129.952 E.20857
G1 X118.114 Y130.687 E.04137
G1 X118.632 Y131.355 E.03228
G1 X119.049 Y132.096 E.03247
G1 X119.352 Y132.887 E.03233
G1 X119.511 Y133.597 E.02777
G1 X119.586 Y134.411 E.03123
G1 X119.531 Y135.405 E.03802
G1 X119.362 Y136.15 E.02913
G1 X119.08 Y136.91 E.03098
G3 X130.579 Y143.624 I-20.427 J48.191 E.50974
G3 X142.648 Y156.415 I-31.964 J42.246 E.67454
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.517 E2.16545
G1 X142.963 Y99.568 E.02495
G1 X141.884 Y99.572 E.0412
G1 X142.456 Y102.661 E.11993
G3 X142.418 Y103.069 I-.767 J.135 E.01581
G1 X142.201 Y103.382 E.01454
G1 X141.929 Y103.536 E.01195
G1 X141.689 Y103.574 E.00928
G1 X132.902 Y103.574 E.3355
G1 X132.56 Y103.495 E.0134
G3 X132.138 Y102.948 I.433 J-.77 E.02707
G1 X132.151 Y102.574 E.01429
G1 X132.707 Y99.573 E.11653
G3 X131.451 Y99.495 I2.707 J-53.477 E.04805
G3 X123.191 Y116.17 I-51.451 J-15.101 E.71408
G1 X124.684 Y117.422 E.07439
G1 X124.903 Y117.72 E.01412
G1 X124.963 Y118.024 E.01184
G1 X124.901 Y118.323 E.01165
G1 X124.875 Y118.364 E.00187
M204 S250
G1 X124.357 Y118.163 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.037 Y122.125 E.16366
G1 X120.888 Y122.2 E.00528
G1 X120.722 Y122.148 E.00548
G1 X119.755 Y121.337 E.03996
; LINE_WIDTH: 0.521286
G1 X119.06 Y120.756 E.02878
; LINE_WIDTH: 0.553516
G1 X118.987 Y120.441 E.01094
G1 X117.999 Y121.433 E.04739
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.93 J-39.337 E.25281
G1 X116.988 Y130.396 E.19623
G1 X117.728 Y131.082 E.03194
G1 X118.192 Y131.689 E.0242
G1 X118.565 Y132.362 E.02435
G1 X118.834 Y133.079 E.02423
G1 X118.97 Y133.711 E.02046
G1 X119.035 Y134.445 E.02332
G1 X118.98 Y135.358 E.02896
G1 X118.825 Y136.022 E.02159
G1 X118.547 Y136.754 E.02478
G1 X118.31 Y137.19 E.01573
G3 X130.249 Y144.067 I-19.584 J47.797 E.43752
G3 X142.39 Y157.025 I-31.645 J41.817 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.227 E1.81943
G1 X144.167 Y98.939 E.00913
G1 X142.92 Y99.017 E.03956
G3 X141.22 Y99.021 I-1.128 J-122.665 E.05384
G1 X141.911 Y102.756 E.12029
G1 X141.9 Y102.875 E.00376
G1 X141.759 Y103.01 E.00621
G1 X132.902 Y103.021 E.28042
G1 X132.756 Y102.968 E.00492
G1 X132.68 Y102.84 E.0047
G3 X133.007 Y100.988 I65.141 J10.53 E.05954
G1 X133.371 Y99.021 E.06332
G3 X132.129 Y98.996 I-.313 J-15.087 E.03935
G2 X131.037 Y98.938 I-.87 J6.069 E.03465
G3 X122.631 Y116 I-51.099 J-14.574 E.60547
; LINE_WIDTH: 0.550176
G1 X122.325 Y116.459 E.01856
G1 X122.691 Y116.472 E.01234
; LINE_WIDTH: 0.519996
G1 X124.329 Y117.845 E.06766
G1 X124.409 Y118.002 E.00557
G1 X124.385 Y118.078 E.00252
; WIPE_START
M204 S10000
G1 X123.747 Y118.848 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.649 Y114.008 Z3.16 F36000
G1 X143.031 Y103.031 Z3.16
G1 Z2.76
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.618396
G1 F13483.205
G1 X143.03 Y102.547 E.01841
; LINE_WIDTH: 0.654966
G1 F12688.721
G1 X143.011 Y102.348 E.00809
; LINE_WIDTH: 0.700683
G1 F11818.181
G1 X142.989 Y102.099 E.01086
; LINE_WIDTH: 0.746399
G1 F11059.424
G1 X142.966 Y101.85 E.0116
; LINE_WIDTH: 0.792115
G1 F10392.217
G1 X142.943 Y101.601 E.01235
; LINE_WIDTH: 0.837831
G1 F9800.933
G1 X142.92 Y101.352 E.01309
; LINE_WIDTH: 0.883547
G1 F9273.311
G1 X142.897 Y101.103 E.01384
; LINE_WIDTH: 0.929264
G1 F8799.596
G1 X142.874 Y100.854 E.01459
; LINE_WIDTH: 0.97498
G1 F8371.926
G1 X142.851 Y100.605 E.01533
; LINE_WIDTH: 1.0207
G1 F7983.9
G1 X142.829 Y100.356 E.01608
; WIPE_START
G1 X142.851 Y100.605 E-.095
G1 X142.874 Y100.854 E-.095
G1 X142.897 Y101.103 E-.095
G1 X142.92 Y101.352 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.326 Y102.115 Z3.16 F36000
G1 X131.333 Y102.516 Z3.16
G1 Z2.76
G1 E.4 F1800
; LINE_WIDTH: 1.07772
G1 F7547.585
G1 X131.357 Y102.427 E.00625
; LINE_WIDTH: 1.0627
G1 F7657.824
G1 X131.442 Y102.103 E.02245
; LINE_WIDTH: 1.01415
G1 F8037.245
G1 X131.527 Y101.779 E.02139
; LINE_WIDTH: 0.965603
G1 F8456.225
G1 X131.611 Y101.455 E.02033
; LINE_WIDTH: 0.917056
G1 F8921.29
G1 X131.69 Y101.141 E.01865
; LINE_WIDTH: 0.876891
G1 F9346.571
G1 X131.769 Y100.826 E.0178
; LINE_WIDTH: 0.836726
G1 F9814.431
G1 X131.847 Y100.512 E.01695
; LINE_WIDTH: 0.796561
G1 F10331.596
G1 X131.926 Y100.198 E.01611
; LINE_WIDTH: 0.756396
G1 F10906.297
G1 X131.928 Y100.188 E.00046
; WIPE_START
G1 X131.926 Y100.198 E-.0037
G1 X131.847 Y100.512 E-.12317
G1 X131.769 Y100.826 E-.12316
G1 X131.69 Y101.141 E-.12316
G1 X131.686 Y101.158 E-.00681
; WIPE_END
G1 E-.02 F1800
G1 X127.488 Y107.533 Z3.16 F36000
G1 X118.987 Y120.441 Z3.16
G1 Z2.76
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.553516
G1 F3600
M204 S5000
G1 X119.2 Y120.198 E.01092
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.687 E.15232
; LINE_WIDTH: 0.550176
G1 X122.325 Y116.459 E.0098
; WIPE_START
M204 S10000
G1 X122.143 Y116.687 E-.11074
G1 X121.688 Y117.23 E-.26926
; WIPE_END
G1 E-.02 F1800
G1 X126.924 Y111.677 Z3.16 F36000
G1 X129.585 Y108.854 Z3.16
G1 Z2.76
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.504 Y110.932 I-45.253 J-22.218 E.08944
G1 X128.945 Y110.753 E.01816
G1 X129.887 Y110.118 E.04339
G2 X131.772 Y108.028 I-12.199 J-12.903 E.10755
G3 X133.186 Y107.053 I2.62 J2.287 E.06626
G3 X137.428 Y108.167 I.95 J5.013 E.17307
G3 X139.313 Y110.257 I-12.2 J12.904 E.10755
G2 X141.945 Y111.332 I2.291 J-1.85 E.11329
G1 X141.945 Y118.162 E.26075
G1 X141.198 Y117.658 E.03439
G3 X139.313 Y115.569 I12.199 J-12.903 E.10755
G2 X137.899 Y114.594 I-2.62 J2.287 E.06626
G2 X133.658 Y115.708 I-.95 J5.013 E.17307
G2 X131.772 Y117.797 I12.199 J12.903 E.10755
G3 X130.359 Y118.772 I-2.62 J-2.287 E.06626
G3 X126.625 Y118 I-.95 J-4.82 E.14949
G3 X126.168 Y119.452 I-2.702 J-.052 E.05891
G1 X122.7 Y123.599 E.20639
G3 X124.232 Y125.338 I-18.55 J17.887 E.08853
G2 X125.646 Y126.313 I2.62 J-2.287 E.06626
G2 X129.887 Y125.199 I.95 J-5.013 E.17307
G2 X131.772 Y123.11 I-12.2 J-12.904 E.10755
G3 X133.186 Y122.135 I2.62 J2.287 E.06626
G3 X137.428 Y123.249 I.95 J5.013 E.17307
G3 X139.313 Y125.338 I-12.201 J12.905 E.10755
G2 X141.945 Y126.413 I2.291 J-1.85 E.11329
G1 X141.945 Y133.243 E.26075
G1 X141.198 Y132.739 E.03439
G3 X139.313 Y130.65 I12.199 J-12.903 E.10755
G2 X137.899 Y129.675 I-2.62 J2.287 E.06626
G2 X133.658 Y130.789 I-.95 J5.013 E.17307
G2 X131.772 Y132.879 I12.199 J12.903 E.10755
G3 X130.359 Y133.854 I-2.62 J-2.287 E.06626
G3 X126.117 Y132.739 I-.95 J-5.013 E.17307
G3 X124.232 Y130.65 I12.199 J-12.903 E.10755
G2 X123.289 Y129.872 I-2.728 J2.344 E.04689
G2 X119.744 Y130.063 I-1.572 J3.841 E.14008
G3 X121.113 Y135.986 I-5.936 J4.492 E.2391
G3 X129.332 Y140.655 I-25.374 J54.234 E.36124
G1 X129.887 Y140.28 E.02559
G2 X131.772 Y138.191 I-12.199 J-12.903 E.10755
G3 X133.186 Y137.216 I2.62 J2.287 E.06626
G3 X137.428 Y138.33 I.95 J5.013 E.17307
G3 X139.313 Y140.419 I-12.2 J12.904 E.10755
G2 X141.945 Y141.494 I2.291 J-1.85 E.11329
G1 X141.945 Y148.324 E.26075
G1 X141.198 Y147.821 E.03439
G3 X139.313 Y145.732 I12.199 J-12.903 E.10755
G2 X138.371 Y144.953 I-2.728 J2.345 E.04689
G2 X134.968 Y145.085 I-1.558 J3.755 E.13413
G3 X136.686 Y146.678 I-19.856 J23.137 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 2.92
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X135.953 Y145.998 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L18
M991 S0 P17 ;notify layer change


G17
G3 Z3.16 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
M73 P53 R8
G1 X125.83 Y119.049
G1 Z2.92
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.696 Y119.25 E.00921
G1 X122.344 Y123.249 E.1992
G1 X121.934 Y123.616 E.02103
G1 X121.483 Y123.84 E.01925
G1 X120.947 Y123.943 E.02083
G1 X120.399 Y123.893 E.02102
G1 X119.952 Y123.727 E.0182
G3 X118.936 Y122.936 I5.17 J-7.69 E.04919
G3 X114.848 Y126.662 I-38.556 J-38.197 E.21129
G1 X118.015 Y129.012 E.15056
G1 X118.485 Y129.403 E.02334
G1 X119.157 Y130.105 E.03709
G1 X119.749 Y130.932 E.03885
G1 X120.215 Y131.839 E.03895
G1 X120.501 Y132.664 E.03332
G3 X120.698 Y135.506 I-6.999 J1.912 E.10949
G1 X120.538 Y136.268 E.02971
G3 X130.508 Y142.101 I-22.089 J49.185 E.44185
G3 X142.443 Y154.07 I-32.664 J44.51 E.64787
G1 X142.443 Y104.567 E1.88996
G1 X142.293 Y104.652 E.0066
G1 X141.692 Y104.747 E.02324
G1 X132.899 Y104.747 E.33569
G1 X132.603 Y104.725 E.01133
G1 X132.03 Y104.543 E.02299
G1 X131.612 Y104.262 E.01921
G1 X131.252 Y103.841 E.02113
G1 X131.006 Y104.409 E.02362
G3 X124.793 Y115.948 I-50.798 J-19.91 E.50158
G1 X125.454 Y116.503 E.03296
G1 X125.868 Y116.985 E.02423
G1 X126.077 Y117.465 E.02
G1 X126.152 Y117.997 E.02054
G1 X126.093 Y118.471 E.01824
G1 X125.933 Y118.894 E.01725
G1 X125.88 Y118.974 E.00369
G1 X125.373 Y118.67 F36000
G1 F13446.369
G1 X125.307 Y118.797 E.00546
G3 X124.465 Y119.807 I-14.526 J-11.254 E.05022
G1 X121.895 Y122.872 E.15272
G1 X121.609 Y123.13 E.01471
G1 X121.201 Y123.314 E.01707
G1 X120.75 Y123.357 E.01729
G1 X120.404 Y123.285 E.01352
G1 X119.974 Y123.041 E.01886
G1 X118.899 Y122.14 E.05356
G3 X113.893 Y126.683 I-41.254 J-40.427 E.25823
G1 X117.666 Y129.482 E.17937
G1 X118.108 Y129.851 E.02197
G1 X118.727 Y130.503 E.03435
G1 X119.266 Y131.263 E.03557
G3 X120.135 Y133.793 I-5.873 J3.432 E.10279
G1 X120.18 Y134.525 E.028
G1 X120.114 Y135.456 E.03563
G1 X119.931 Y136.289 E.03258
G1 X119.836 Y136.599 E.01238
G3 X130.932 Y143.157 I-21.133 J48.428 E.49333
G3 X142.92 Y155.769 I-32.59 J42.982 E.66726
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y112.024 E1.66935
G3 X143.031 Y103.287 I1741.239 J-4 E.33358
; LINE_WIDTH: 0.616306
G1 F13531.627
G1 X143.032 Y103.033 E.00964
G1 X142.967 Y103.275 E.00953
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.589 Y103.825 E.02546
G1 X142.112 Y104.095 E.02094
G1 X141.692 Y104.162 E.01626
G1 X132.899 Y104.162 E.33569
G1 X132.692 Y104.146 E.00792
G1 X132.265 Y104.005 E.01717
G1 X131.999 Y103.822 E.01234
G1 F12744.103
G1 X131.732 Y103.504 E.01586
; LINE_WIDTH: 0.665991
G1 F11332.584
G1 X131.676 Y103.384 E.00545
; LINE_WIDTH: 0.711986
G1 F10900.608
G1 X131.62 Y103.264 E.00584
; LINE_WIDTH: 0.757981
G1 F10477.026
G1 X131.565 Y103.144 E.00624
; LINE_WIDTH: 0.803976
G1 F10061.85
G1 X131.509 Y103.024 E.00663
; LINE_WIDTH: 0.849971
G1 F9655.055
G1 X131.453 Y102.904 E.00703
; LINE_WIDTH: 0.895966
G1 F9139.655
G1 X131.397 Y102.784 E.00743
; LINE_WIDTH: 0.942604
G1 F8670.353
G1 X131.379 Y102.725 E.00368
; LINE_WIDTH: 0.989241
G1 F8246.894
G1 X131.36 Y102.665 E.00387
; LINE_WIDTH: 1.03588
G1 F7862.872
G1 X131.342 Y102.606 E.00406
; LINE_WIDTH: 1.08252
G1 F7513.023
G1 X131.324 Y102.547 E.00425
G1 X131.282 Y102.595 E.00436
; LINE_WIDTH: 1.03588
G1 F7862.872
G1 X131.241 Y102.644 E.00417
; LINE_WIDTH: 0.989241
G1 F8246.894
G1 X131.199 Y102.692 E.00397
; LINE_WIDTH: 0.942604
G1 F8670.353
G1 X131.158 Y102.741 E.00378
; LINE_WIDTH: 0.895966
G1 F9139.655
G1 X131.08 Y102.884 E.00917
; LINE_WIDTH: 0.849971
G1 F9655.055
G1 X131.003 Y103.028 E.00868
; LINE_WIDTH: 0.803976
G1 F10232.057
G1 X130.925 Y103.172 E.00819
; LINE_WIDTH: 0.757981
G1 F10750.308
G1 X130.848 Y103.316 E.0077
; LINE_WIDTH: 0.711986
G1 F11281.389
G1 X130.77 Y103.459 E.00721
; LINE_WIDTH: 0.665991
G1 F11825.245
G1 X130.693 Y103.603 E.00672
; LINE_WIDTH: 0.619996
G1 F13211.368
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.401 Y104.348 E.01527
G1 X130.237 Y104.76 E.01691
G3 X124.002 Y116.05 I-50.721 J-20.642 E.49357
G1 X125.078 Y116.952 E.05358
G1 X125.382 Y117.314 E.01806
G1 X125.514 Y117.625 E.01288
G1 X125.566 Y118.006 E.01471
G1 X125.495 Y118.432 E.01647
G1 X125.414 Y118.59 E.00677
G1 X124.846 Y118.42 F36000
G1 F13446.369
G1 X124.798 Y118.498 E.00346
G1 X121.447 Y122.496 E.1992
G1 X121.183 Y122.7 E.0127
G1 X120.889 Y122.774 E.0116
G1 X120.595 Y122.732 E.01132
G1 X120.35 Y122.593 E.01077
G1 X118.855 Y121.339 E.07448
G3 X112.93 Y126.698 I-40.427 J-38.748 E.30526
G1 X117.317 Y129.953 E.20857
G1 X117.731 Y130.299 E.02059
G1 X118.298 Y130.902 E.03161
G1 X118.783 Y131.594 E.03229
G3 X119.531 Y135.405 I-5.171 J2.993 E.15101
G1 X119.361 Y136.154 E.0293
G1 X119.08 Y136.91 E.03082
G3 X130.579 Y143.624 I-20.535 J48.375 E.50972
G3 X142.648 Y156.415 I-32.179 J42.45 E.67452
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.518 E2.16541
G1 X142.963 Y99.57 E.02494
G1 X141.886 Y99.574 E.04112
M73 P54 R8
G1 X142.458 Y102.663 E.11993
G3 X142.421 Y103.071 I-.767 J.135 E.01581
G1 X142.204 Y103.384 E.01454
G1 X141.932 Y103.538 E.01195
G1 X141.692 Y103.576 E.00928
G1 X132.899 Y103.576 E.33569
G1 X132.537 Y103.487 E.01423
G3 X132.233 Y103.201 I.362 J-.689 E.01615
G1 X132.121 Y102.774 E.01684
G1 X132.134 Y102.656 E.00454
G1 X132.704 Y99.575 E.11962
G3 X131.493 Y99.491 I.187 J-11.407 E.04639
G1 X131.442 Y99.525 E.00235
G3 X123.207 Y116.148 I-51.356 J-15.09 E.71184
G1 X124.702 Y117.401 E.07446
G1 X124.919 Y117.695 E.01395
G1 X124.98 Y118.003 E.01198
G1 X124.918 Y118.302 E.01166
G1 X124.893 Y118.344 E.00188
M204 S250
G1 X124.375 Y118.142 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.023 Y122.141 E.16519
G1 X120.891 Y122.218 E.00483
G1 X120.746 Y122.196 E.00464
G1 X119.737 Y121.357 E.04156
; LINE_WIDTH: 0.521156
G1 X119.042 Y120.777 E.02872
; LINE_WIDTH: 0.529296
G1 X119.004 Y120.749 E.00154
; LINE_WIDTH: 0.554876
G1 X118.969 Y120.462 E.00981
G1 X117.999 Y121.433 E.04659
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-37.701 J-36.804 E.25284
G1 X116.988 Y130.397 E.19623
G3 X118.995 Y133.833 I-3.276 J4.218 E.12911
G3 X118.98 Y135.357 I-7.135 J.691 E.04837
G1 X118.824 Y136.026 E.02173
G1 X118.548 Y136.752 E.02461
G1 X118.31 Y137.19 E.01576
G3 X130.249 Y144.067 I-19.568 J47.771 E.43751
G3 X142.39 Y157.025 I-31.527 J41.707 E.56494
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.227 E1.81944
G1 X144.167 Y98.939 E.00912
G1 X142.92 Y99.019 E.03959
G3 X141.222 Y99.023 I-1.112 J-114.421 E.05374
G1 X141.914 Y102.759 E.12029
G1 X141.903 Y102.877 E.00376
G1 X141.761 Y103.012 E.00621
G1 X141.692 Y103.023 E.00223
G1 X132.899 Y103.023 E.27837
G1 X132.75 Y102.967 E.00504
G1 X132.674 Y102.791 E.00609
G1 X132.677 Y102.756 E.00109
G1 X133.369 Y99.023 E.12021
G3 X132.128 Y98.997 I-.312 J-14.695 E.03928
G2 X131.038 Y98.936 I-.884 J6.053 E.03464
G3 X122.633 Y115.998 I-50.903 J-14.475 E.60545
; LINE_WIDTH: 0.551596
G1 X122.342 Y116.438 E.01779
G1 X122.686 Y116.433 E.01163
; LINE_WIDTH: 0.519996
G1 X124.347 Y117.824 E.06859
G1 X124.427 Y117.981 E.00556
G1 X124.402 Y118.057 E.00253
; WIPE_START
M204 S10000
G1 X123.765 Y118.827 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.667 Y113.988 Z3.32 F36000
G1 X143.032 Y103.033 Z3.32
G1 Z2.92
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.615876
G1 F13541.632
G1 X143.031 Y102.55 E.01832
; LINE_WIDTH: 0.652696
G1 F12735.301
G1 X143.013 Y102.349 E.00812
; LINE_WIDTH: 0.698411
G1 F11858.602
G1 X142.99 Y102.1 E.01082
; LINE_WIDTH: 0.744126
G1 F11094.832
G1 X142.967 Y101.851 E.01157
; LINE_WIDTH: 0.789841
G1 F10423.492
G1 X142.944 Y101.602 E.01231
; LINE_WIDTH: 0.835556
G1 F9828.761
G1 X142.921 Y101.353 E.01306
; LINE_WIDTH: 0.881271
G1 F9298.235
G1 X142.898 Y101.104 E.0138
; LINE_WIDTH: 0.926986
G1 F8822.048
G1 X142.875 Y100.855 E.01455
; LINE_WIDTH: 0.972701
G1 F8392.257
G1 X142.853 Y100.606 E.01529
; LINE_WIDTH: 1.01842
G1 F8002.399
G1 X142.83 Y100.357 E.01604
; WIPE_START
G1 X142.853 Y100.606 E-.095
G1 X142.875 Y100.855 E-.095
G1 X142.898 Y101.104 E-.095
G1 X142.921 Y101.353 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.331 Y100.549 Z3.32 F36000
G1 X131.927 Y100.189 Z3.32
G1 Z2.92
G1 E.4 F1800
; LINE_WIDTH: 0.754336
G1 F10937.501
G1 X131.925 Y100.198 E.00043
; LINE_WIDTH: 0.794496
G1 F10359.662
G1 X131.846 Y100.512 E.01606
; LINE_WIDTH: 0.834656
G1 F9839.815
G1 X131.768 Y100.826 E.01691
; LINE_WIDTH: 0.874816
G1 F9369.647
G1 X131.689 Y101.141 E.01776
; LINE_WIDTH: 0.914976
G1 F8942.361
G1 X131.61 Y101.455 E.0186
; LINE_WIDTH: 0.920056
G1 F8891.073
G1 X131.601 Y101.491 E.00215
; LINE_WIDTH: 0.959046
G1 F8516.184
G1 X131.534 Y101.745 E.01584
; LINE_WIDTH: 0.998036
G1 F8171.631
G1 X131.467 Y101.999 E.01651
; LINE_WIDTH: 1.03703
G1 F7853.873
G1 X131.401 Y102.254 E.01718
; LINE_WIDTH: 1.07602
G1 F7559.903
G1 X131.334 Y102.508 E.01784
; LINE_WIDTH: 1.08252
G1 F7513.023
G1 X131.324 Y102.547 E.00274
; WIPE_START
G1 X131.334 Y102.508 E-.01526
G1 X131.401 Y102.254 E-.09985
G1 X131.467 Y101.999 E-.09985
G1 X131.534 Y101.745 E-.09985
G1 X131.578 Y101.579 E-.06519
; WIPE_END
G1 E-.02 F1800
G1 X127.339 Y107.927 Z3.32 F36000
G1 X118.969 Y120.462 Z3.32
G1 Z2.92
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.554876
G1 F3600
M204 S5000
G1 X119.2 Y120.198 E.01187
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15231
; LINE_WIDTH: 0.551596
G1 X122.342 Y116.438 E.01075
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.12116
G1 X121.705 Y117.21 E-.25884
; WIPE_END
G1 E-.02 F1800
G1 X126.937 Y111.652 Z3.32 F36000
G1 X129.584 Y108.84 Z3.32
G1 Z2.92
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.512 Y110.922 I-29.134 J-13.685 E.08945
G1 X128.945 Y110.767 E.01754
G1 X129.887 Y110.188 E.04223
G2 X131.772 Y108.233 I-9.748 J-11.286 E.10383
G3 X133.186 Y107.216 I3.031 J2.723 E.06701
G3 X137.428 Y108.097 I1.222 J4.767 E.17126
G3 X139.313 Y110.052 I-9.749 J11.287 E.10383
G2 X141.945 Y111.241 I2.509 J-2.047 E.1143
G1 X141.945 Y118.188 E.26523
G1 X141.198 Y117.729 E.03348
G3 X139.313 Y115.774 I9.749 J-11.287 E.10383
G2 X137.899 Y114.757 I-3.032 J2.723 E.06701
G2 X133.658 Y115.638 I-1.222 J4.767 E.17126
G2 X131.772 Y117.593 I9.748 J11.286 E.10383
G3 X130.359 Y118.609 I-3.031 J-2.723 E.06701
G3 X126.642 Y118.051 I-1.212 J-4.583 E.14754
G3 X126.186 Y119.432 I-2.726 J-.135 E.05618
G1 X122.741 Y123.55 E.20499
G3 X124.232 Y125.133 I-7.912 J8.943 E.08314
G2 X125.646 Y126.15 I3.031 J-2.723 E.06701
G2 X129.887 Y125.269 I1.222 J-4.767 E.17126
G2 X131.772 Y123.314 I-9.748 J-11.286 E.10383
G3 X133.186 Y122.297 I3.031 J2.723 E.06701
G3 X137.428 Y123.178 I1.222 J4.767 E.17126
G3 X139.313 Y125.133 I-9.748 J11.287 E.10383
G2 X141.945 Y126.322 I2.509 J-2.047 E.1143
G1 X141.945 Y133.269 E.26523
G1 X141.198 Y132.81 E.03348
G3 X139.313 Y130.855 I9.748 J-11.286 E.10383
G2 X137.899 Y129.838 I-3.031 J2.723 E.06701
G2 X133.658 Y130.719 I-1.222 J4.767 E.17126
G2 X131.772 Y132.674 I9.749 J11.287 E.10383
G3 X129.887 Y133.82 I-2.719 J-2.349 E.08562
G3 X125.174 Y131.922 I-.594 J-5.324 E.20211
G2 X123.289 Y130.062 I-6.817 J5.025 E.10152
G2 X119.747 Y130.058 I-1.775 J3.652 E.13995
G3 X121.111 Y135.984 I-5.96 J4.491 E.2391
G3 X129.358 Y140.676 I-25.048 J53.626 E.36267
G1 X129.887 Y140.351 E.0237
G2 X131.772 Y138.396 I-9.748 J-11.286 E.10383
G3 X133.186 Y137.379 I3.031 J2.723 E.06701
G3 X137.428 Y138.259 I1.222 J4.767 E.17126
G3 X139.313 Y140.214 I-9.748 J11.287 E.10383
G2 X141.945 Y141.403 I2.509 J-2.047 E.1143
G1 X141.945 Y148.35 E.26523
G3 X140.256 Y147.003 I2.396 J-4.738 E.08309
G2 X138.371 Y145.143 I-6.818 J5.025 E.10152
G2 X134.979 Y145.085 I-1.76 J3.727 E.13347
G3 X136.696 Y146.679 I-39.52 J44.29 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 3.08
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X135.963 Y145.999 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L19
M991 S0 P18 ;notify layer change


G17
G3 Z3.32 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.841 Y119.036
G1 Z3.08
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.707 Y119.236 E.00921
G1 X122.333 Y123.262 E.20056
G1 X121.925 Y123.629 E.02093
G1 X121.474 Y123.853 E.01924
G1 X120.936 Y123.957 E.0209
G1 X120.389 Y123.907 E.02098
G1 X119.932 Y123.736 E.01866
G3 X118.924 Y122.949 I5.51 J-8.095 E.04885
G3 X114.848 Y126.662 I-39.269 J-39.006 E.21061
G1 X118.015 Y129.012 E.15054
G1 X118.483 Y129.401 E.02323
G1 X119.156 Y130.105 E.0372
G1 X119.749 Y130.932 E.03885
G1 X120.215 Y131.838 E.03891
G1 X120.497 Y132.65 E.03282
G3 X120.698 Y135.504 I-6.982 J1.926 E.10994
G1 X120.538 Y136.268 E.02983
G3 X129.976 Y141.712 I-22.086 J49.192 E.41669
G3 X142.443 Y154.069 I-32.175 J44.931 E.67302
G1 X142.443 Y104.571 E1.8898
G1 X142.295 Y104.654 E.00649
G1 X141.694 Y104.749 E.02324
G1 X132.897 Y104.749 E.33588
G3 X132.244 Y104.637 I0 J-1.95 E.02542
G1 X131.649 Y104.298 E.02611
G1 X131.234 Y103.822 E.02414
G3 X124.803 Y115.934 I-51.109 J-19.374 E.52496
G1 X125.466 Y116.49 E.03303
G1 X125.875 Y116.963 E.02389
G1 X126.087 Y117.445 E.0201
G1 X126.163 Y117.984 E.02078
G1 X126.105 Y118.457 E.01821
G1 X125.945 Y118.88 E.01726
G1 X125.891 Y118.961 E.00369
G1 X125.384 Y118.656 F36000
G1 F13446.369
G1 X125.319 Y118.783 E.00546
G3 X124.454 Y119.82 I-14.861 J-11.512 E.05158
G1 X121.884 Y122.886 E.15272
G1 X121.599 Y123.142 E.01464
G1 X121.192 Y123.327 E.01707
G1 X120.74 Y123.37 E.01734
G1 X120.389 Y123.298 E.01365
G1 X119.962 Y123.055 E.01876
G1 X118.886 Y122.153 E.0536
G3 X113.893 Y126.683 I-38.377 J-37.285 E.25758
G1 X117.666 Y129.482 E.17935
G1 X118.106 Y129.849 E.02186
G1 X118.727 Y130.503 E.03446
G1 X119.266 Y131.263 E.03556
G3 X120.136 Y133.793 I-5.882 J3.436 E.10282
G1 X120.18 Y134.525 E.02798
G1 X120.115 Y135.453 E.03554
G1 X119.933 Y136.283 E.03243
G1 X119.836 Y136.599 E.01262
G3 X130.162 Y142.574 I-21.164 J48.486 E.45647
G3 X142.92 Y155.769 I-31.902 J43.613 E.70415
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y116.025 E1.5166
G3 X143.031 Y103.289 I1818.358 J-6 E.48628
; LINE_WIDTH: 0.614846
G1 F13565.659
G1 X143.033 Y103.035 E.00961
G1 X142.969 Y103.277 E.0095
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.592 Y103.827 E.02545
G1 X142.115 Y104.097 E.02094
G1 X141.694 Y104.164 E.01626
G1 X132.897 Y104.164 E.33588
G1 X132.44 Y104.085 E.0177
G1 X132.002 Y103.829 E.01937
G1 F13054.417
G1 X131.735 Y103.514 E.01578
; LINE_WIDTH: 0.647896
G1 F11632.345
G1 X131.539 Y103.036 E.02064
; LINE_WIDTH: 0.695714
G1 F9971.857
G1 X131.515 Y102.981 E.00261
; LINE_WIDTH: 0.743532
G1 F9785.226
G1 X131.492 Y102.925 E.0028
; LINE_WIDTH: 0.79135
G1 F9600.358
G1 X131.468 Y102.869 E.00299
; LINE_WIDTH: 0.839167
G1 F9417.265
G1 X131.444 Y102.813 E.00318
; LINE_WIDTH: 0.886985
G1 F9235.923
G1 X131.42 Y102.758 E.00337
; LINE_WIDTH: 0.934803
G1 F8745.466
G1 X131.397 Y102.702 E.00356
; LINE_WIDTH: 0.982621
G1 F8304.471
G1 X131.373 Y102.646 E.00375
; LINE_WIDTH: 1.03044
G1 F7905.816
G1 X131.349 Y102.59 E.00394
; LINE_WIDTH: 1.07826
G1 F7543.682
G1 X131.326 Y102.535 E.00412
G1 X131.282 Y102.585 E.0045
; LINE_WIDTH: 1.03044
G1 F7905.816
G1 X131.239 Y102.635 E.0043
; LINE_WIDTH: 0.982621
G1 F8304.471
G1 X131.196 Y102.685 E.00409
; LINE_WIDTH: 0.934803
G1 F8745.466
G1 X131.153 Y102.735 E.00388
; LINE_WIDTH: 0.886985
G1 F9235.923
G1 X131.11 Y102.785 E.00368
; LINE_WIDTH: 0.839167
G1 F9784.662
G1 X131.067 Y102.835 E.00347
; LINE_WIDTH: 0.79135
G1 F9988.431
G1 X131.024 Y102.885 E.00326
; LINE_WIDTH: 0.743532
G1 F10194.303
G1 X130.98 Y102.936 E.00306
; LINE_WIDTH: 0.695714
G1 F10402.252
G1 X130.937 Y102.986 E.00285
; LINE_WIDTH: 0.647896
G1 F11704.687
G1 X130.788 Y103.357 E.016
G1 F12650.201
G1 X130.685 Y103.613 E.01106
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X130.538 Y103.986 E.01527
G1 X130.078 Y105.145 E.04763
G3 X124.013 Y116.036 I-49.808 J-20.605 E.477
G1 X125.089 Y116.938 E.05363
G1 X125.391 Y117.295 E.01782
G1 X125.524 Y117.607 E.01295
G1 X125.577 Y117.992 E.01485
G1 X125.506 Y118.418 E.01648
G1 X125.425 Y118.576 E.00678
G1 X124.857 Y118.407 F36000
G1 F13446.369
G1 X124.81 Y118.484 E.00346
G1 X121.435 Y122.51 E.20055
G1 X121.173 Y122.713 E.01266
G1 X120.878 Y122.787 E.01163
G1 X120.582 Y122.745 E.01139
G1 X120.338 Y122.606 E.0107
G1 X118.843 Y121.352 E.07452
G3 X112.93 Y126.698 I-38.419 J-36.554 E.30462
G1 X117.317 Y129.952 E.20855
G1 X117.728 Y130.297 E.02049
G1 X118.298 Y130.902 E.03172
G1 X118.783 Y131.594 E.03228
G3 X119.531 Y135.403 I-5.17 J2.993 E.15093
G1 X119.363 Y136.148 E.02916
G1 X119.08 Y136.91 E.03105
G3 X129.816 Y143.047 I-20.426 J48.194 E.47322
G3 X142.647 Y156.415 I-31.73 J43.298 E.71102
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.519 E2.16538
G1 X142.963 Y99.572 E.02495
G1 X141.889 Y99.576 E.04101
G1 X142.461 Y102.665 E.11993
G3 X142.423 Y103.073 I-.767 J.135 E.01581
G1 X142.206 Y103.386 E.01454
G1 X141.934 Y103.54 E.01195
G1 X141.694 Y103.578 E.00928
G1 X132.897 Y103.578 E.33588
G1 X132.636 Y103.533 E.0101
G1 X132.386 Y103.387 E.01105
G1 X132.233 Y103.207 E.009
G1 X132.13 Y102.933 E.01118
G1 X132.131 Y102.658 E.01052
G1 X132.702 Y99.577 E.11963
G3 X131.491 Y99.491 I.188 J-11.143 E.04637
G1 X131.444 Y99.522 E.00216
G3 X123.218 Y116.134 I-51.474 J-15.148 E.71129
G1 X124.713 Y117.387 E.07451
G1 X124.929 Y117.678 E.01384
G1 X124.992 Y117.989 E.01208
G1 X124.93 Y118.288 E.01167
G1 X124.904 Y118.33 E.00188
M204 S250
G1 X124.386 Y118.129 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.012 Y122.155 E.16631
G1 X120.88 Y122.231 E.00482
G1 X120.734 Y122.209 E.00469
G1 X119.711 Y121.358 E.04213
; LINE_WIDTH: 0.521066
G1 X119.031 Y120.79 E.02811
; LINE_WIDTH: 0.530866
G1 X118.985 Y120.755 E.00186
; LINE_WIDTH: 0.555776
G1 X118.958 Y120.475 E.00958
G1 X117.999 Y121.433 E.04607
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.955 J-39.366 E.25281
G1 X116.988 Y130.396 E.19622
G3 X118.995 Y133.833 I-3.284 J4.223 E.12912
G3 X118.98 Y135.361 I-7.059 J.692 E.04846
G1 X118.825 Y136.02 E.02144
G1 X118.548 Y136.753 E.02482
G1 X118.31 Y137.19 E.01573
G3 X129.492 Y143.495 I-19.843 J48.261 E.40746
G3 X142.39 Y157.025 I-31.403 J42.848 E.59492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.227 E1.81944
G1 X144.167 Y98.939 E.00911
G1 X142.918 Y99.021 E.03962
G3 X141.225 Y99.025 I-1.1 J-108.104 E.05363
G1 X141.916 Y102.761 E.12029
G1 X141.905 Y102.879 E.00376
G1 X141.764 Y103.014 E.00621
G1 X132.897 Y103.025 E.28073
G1 X132.749 Y102.97 E.005
G1 X132.674 Y102.838 E.00478
G3 X133.002 Y100.992 I32.901 J4.884 E.05938
G1 X133.366 Y99.025 E.06332
G3 X132.128 Y98.999 I-.31 J-14.334 E.03922
G2 X131.039 Y98.936 I-.883 J5.903 E.03459
G3 X122.643 Y115.984 I-51.127 J-14.589 E.60488
; LINE_WIDTH: 0.552516
G1 X122.353 Y116.424 E.01782
G1 X122.696 Y116.418 E.01161
; LINE_WIDTH: 0.519996
G1 X124.358 Y117.811 E.06866
G1 X124.438 Y117.967 E.00556
G1 X124.414 Y118.043 E.00253
; WIPE_START
M204 S10000
M73 P55 R8
G1 X123.776 Y118.814 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.68 Y113.976 Z3.48 F36000
G1 X143.033 Y103.035 Z3.48
G1 Z3.08
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.613336
G1 F13601.038
G1 X143.032 Y102.552 E.01822
; LINE_WIDTH: 0.650416
G1 F12782.432
G1 X143.014 Y102.35 E.00814
; LINE_WIDTH: 0.696131
G1 F11899.456
G1 X142.991 Y102.101 E.01079
; LINE_WIDTH: 0.741846
G1 F11130.585
G1 X142.968 Y101.852 E.01153
; LINE_WIDTH: 0.787561
G1 F10455.044
G1 X142.945 Y101.603 E.01228
; LINE_WIDTH: 0.833276
G1 F9856.81
G1 X142.922 Y101.354 E.01302
; LINE_WIDTH: 0.878991
G1 F9323.334
G1 X142.899 Y101.105 E.01377
; LINE_WIDTH: 0.924706
G1 F8844.639
G1 X142.877 Y100.856 E.01451
; LINE_WIDTH: 0.970421
G1 F8412.698
G1 X142.854 Y100.607 E.01526
; LINE_WIDTH: 1.01614
G1 F8020.982
G1 X142.831 Y100.358 E.016
; WIPE_START
G1 X142.854 Y100.607 E-.095
G1 X142.877 Y100.856 E-.095
G1 X142.899 Y101.105 E-.095
G1 X142.922 Y101.354 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.329 Y102.127 Z3.48 F36000
G1 X131.326 Y102.535 Z3.48
G1 Z3.08
G1 E.4 F1800
; LINE_WIDTH: 1.07826
G1 F7543.682
G1 X131.384 Y102.317 E.0153
; LINE_WIDTH: 1.0429
G1 F7808.163
G1 X131.442 Y102.1 E.01478
; LINE_WIDTH: 1.00754
G1 F8091.862
G1 X131.522 Y101.786 E.02058
; LINE_WIDTH: 0.963361
G1 F8476.63
G1 X131.603 Y101.472 E.01965
; LINE_WIDTH: 0.919186
G1 F8899.814
G1 X131.684 Y101.158 E.01871
; LINE_WIDTH: 0.875011
G1 F9367.473
G1 X131.764 Y100.843 E.01778
; LINE_WIDTH: 0.830836
G1 F9887.006
G1 X131.773 Y100.808 E.00188
; LINE_WIDTH: 0.826356
G1 F9942.932
G1 X131.848 Y100.499 E.01641
; LINE_WIDTH: 0.790886
G1 F10409.095
G1 X131.924 Y100.191 E.01567
; WIPE_START
G1 X131.848 Y100.499 E-.12076
G1 X131.773 Y100.808 E-.12076
G1 X131.764 Y100.843 E-.01375
G1 X131.684 Y101.158 E-.12328
G1 X131.683 Y101.161 E-.00146
; WIPE_END
G1 E-.02 F1800
G1 X127.484 Y107.535 Z3.48 F36000
G1 X118.958 Y120.475 Z3.48
G1 Z3.08
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.555776
G1 F3600
M204 S5000
G1 X119.2 Y120.198 E.01249
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15231
; LINE_WIDTH: 0.552516
G1 X122.353 Y116.424 E.01137
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.1279
G1 X121.717 Y117.196 E-.2521
; WIPE_END
G1 E-.02 F1800
G1 X126.952 Y111.642 Z3.48 F36000
G1 X129.591 Y108.841 Z3.48
G1 Z3.08
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.513 Y110.92 I-38.006 J-18.385 E.08944
G2 X129.887 Y110.262 I-.864 J-3.565 E.05861
G2 X131.772 Y108.422 I-7.803 J-9.881 E.10076
G3 X133.186 Y107.367 I3.496 J3.213 E.06773
G3 X137.428 Y108.023 I1.465 J4.568 E.16989
G3 X139.313 Y109.863 I-7.804 J9.882 E.10076
G2 X141.945 Y111.157 I2.751 J-2.272 E.1154
G1 X141.945 Y118.219 E.26961
G1 X141.198 Y117.802 E.03266
G3 X139.313 Y115.962 I7.804 J-9.882 E.10076
G2 X137.899 Y114.908 I-3.496 J3.214 E.06773
G2 X133.658 Y115.564 I-1.465 J4.568 E.16989
G2 X131.772 Y117.404 I7.803 J9.881 E.10076
G3 X130.359 Y118.458 I-3.496 J-3.213 E.06773
G3 X126.653 Y118.101 I-1.446 J-4.398 E.14628
G3 X126.197 Y119.418 I-2.784 J-.226 E.05377
G1 X122.79 Y123.492 E.20278
G3 X124.232 Y124.945 I-6.183 J7.58 E.07829
G2 X125.646 Y125.999 I3.496 J-3.213 E.06773
G2 X129.887 Y125.343 I1.465 J-4.567 E.16989
G2 X131.772 Y123.503 I-7.803 J-9.881 E.10076
G3 X133.186 Y122.449 I3.496 J3.213 E.06773
G3 X137.428 Y123.105 I1.465 J4.568 E.16989
G3 X139.313 Y124.945 I-7.803 J9.881 E.10076
G2 X141.945 Y126.238 I2.751 J-2.272 E.1154
G1 X141.945 Y133.3 E.26961
G1 X141.198 Y132.884 E.03266
G3 X139.313 Y131.044 I7.803 J-9.881 E.10076
G2 X137.899 Y129.989 I-3.496 J3.213 E.06773
G2 X133.658 Y130.645 I-1.465 J4.568 E.16989
G2 X131.772 Y132.485 I7.804 J9.882 E.10076
G3 X130.359 Y133.54 I-3.496 J-3.213 E.06773
G3 X126.117 Y132.884 I-1.465 J-4.568 E.16989
G3 X124.232 Y131.044 I7.803 J-9.881 E.10076
G2 X119.742 Y130.051 I-2.923 J2.57 E.18782
G3 X121.113 Y135.984 I-5.973 J4.505 E.23944
G3 X129.399 Y140.696 I-26.125 J55.585 E.36431
G1 X129.887 Y140.424 E.02133
G2 X131.772 Y138.584 I-7.803 J-9.881 E.10076
G3 X133.186 Y137.53 I3.496 J3.213 E.06773
G3 X137.428 Y138.186 I1.465 J4.568 E.16989
G3 X139.313 Y140.026 I-7.804 J9.882 E.10076
G2 X141.945 Y141.32 I2.751 J-2.272 E.1154
G1 X141.945 Y148.381 E.26961
G1 X141.198 Y147.965 E.03266
G3 X139.313 Y146.125 I7.803 J-9.881 E.10076
G2 X134.974 Y145.086 I-2.903 J2.543 E.18162
G3 X136.693 Y146.677 I-37.599 J42.322 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 3.24
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X135.959 Y145.998 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L20
M991 S0 P19 ;notify layer change


G17
G3 Z3.48 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.03
G1 Z3.24
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.231 E.00922
G1 X122.328 Y123.268 E.20109
G1 X121.942 Y123.62 E.01997
G1 X121.522 Y123.839 E.01807
G1 X121.016 Y123.956 E.01982
G1 X120.456 Y123.928 E.02144
G1 X119.959 Y123.758 E.02003
G3 X118.919 Y122.954 I5.017 J-7.57 E.05025
G3 X114.848 Y126.662 I-39.244 J-38.989 E.21034
G1 X118.015 Y129.012 E.15054
G3 X118.933 Y129.849 I-4.609 J5.971 E.04749
G1 X119.565 Y130.646 E.03883
G1 X120.075 Y131.529 E.03892
G1 X120.45 Y132.478 E.03896
G1 X120.66 Y133.369 E.03496
G3 X120.765 Y134.489 I-5.914 J1.119 E.04302
G1 X120.698 Y135.506 E.03891
G1 X120.538 Y136.268 E.02971
G3 X130.508 Y142.102 I-22.167 J49.317 E.44188
G3 X142.443 Y154.069 I-32.646 J44.494 E.64782
G1 X142.443 Y104.574 E1.88966
G1 X142.298 Y104.657 E.00638
G1 X141.697 Y104.752 E.02324
G1 X132.894 Y104.752 E.33607
G1 X132.599 Y104.729 E.0113
G1 X132.073 Y104.57 E.02098
G1 X131.63 Y104.287 E.02007
G1 X131.246 Y103.843 E.02242
G1 X130.991 Y104.447 E.02503
G3 X124.807 Y115.928 I-51.266 J-20.207 E.49905
G1 X125.47 Y116.484 E.03305
G1 X125.877 Y116.955 E.02375
G1 X126.091 Y117.437 E.02015
G1 X126.167 Y117.978 E.02087
G1 X126.109 Y118.452 E.0182
G1 X125.949 Y118.875 E.01727
G1 X125.896 Y118.955 E.0037
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.323 Y118.778 E.00547
G3 X124.449 Y119.826 I-15.029 J-11.643 E.05212
G1 X121.88 Y122.891 E.15272
G1 X121.609 Y123.138 E.01397
G1 X121.316 Y123.291 E.01264
G1 X120.962 Y123.373 E.01387
G1 X120.52 Y123.343 E.01689
G1 X120.197 Y123.221 E.01318
G3 X118.881 Y122.158 I9.119 J-12.631 E.06461
G3 X113.893 Y126.683 I-38.38 J-37.3 E.25731
G1 X117.666 Y129.482 E.17935
G3 X118.523 Y130.268 I-4.176 J5.414 E.04447
G1 X119.099 Y131 E.03554
G1 X119.561 Y131.811 E.03564
G1 X119.901 Y132.682 E.03568
G3 X120.114 Y135.456 I-6.636 J1.906 E.10696
G1 X119.931 Y136.289 E.03258
G1 X119.836 Y136.599 E.01238
G3 X130.932 Y143.157 I-21.129 J48.42 E.49335
G3 X142.92 Y155.769 I-32.375 J42.776 E.66728
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y114.026 E1.59292
G3 X143.032 Y103.29 I1197.211 J-5 E.40989
; LINE_WIDTH: 0.613396
G1 F13599.629
G1 X143.035 Y103.036 E.00958
G1 X142.971 Y103.279 E.00947
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.594 Y103.829 E.02544
G1 X142.117 Y104.099 E.02094
G1 X141.697 Y104.166 E.01626
G1 X132.894 Y104.166 E.33607
G1 X132.688 Y104.15 E.00791
G1 X132.293 Y104.026 E.01578
G1 X132.01 Y103.841 E.01293
G1 F12521.164
G1 X131.723 Y103.502 E.01695
; LINE_WIDTH: 0.66527
G1 F11029.111
G1 X131.684 Y103.404 E.00434
; LINE_WIDTH: 0.710544
G1 F10688.306
G1 X131.644 Y103.306 E.00465
; LINE_WIDTH: 0.755818
G1 F10352.832
G1 X131.605 Y103.208 E.00497
; LINE_WIDTH: 0.801092
G1 F10022.706
G1 X131.565 Y103.11 E.00528
; LINE_WIDTH: 0.846366
G1 F9697.918
G1 X131.526 Y103.012 E.00559
; LINE_WIDTH: 0.89164
G1 F9185.774
G1 X131.486 Y102.914 E.0059
; LINE_WIDTH: 0.936914
G1 F8725.009
G1 X131.447 Y102.816 E.00621
; LINE_WIDTH: 0.982188
G1 F8308.259
G1 X131.407 Y102.718 E.00652
; LINE_WIDTH: 1.02746
G1 F7929.508
G1 X131.368 Y102.621 E.00683
; LINE_WIDTH: 1.07274
G1 F7583.783
G1 X131.328 Y102.523 E.00715
G1 X131.286 Y102.572 E.0044
; LINE_WIDTH: 1.02746
G1 F7929.508
G1 X131.244 Y102.622 E.00421
; LINE_WIDTH: 0.982188
G1 F8308.259
G1 X131.202 Y102.671 E.00401
; LINE_WIDTH: 0.936914
G1 F8725.009
G1 X131.16 Y102.72 E.00382
; LINE_WIDTH: 0.89164
G1 F9185.774
G1 X131.118 Y102.77 E.00363
; LINE_WIDTH: 0.846366
G1 F9697.918
G1 X131.075 Y102.819 E.00344
; LINE_WIDTH: 0.801092
G1 F10270.543
G1 X131.033 Y102.869 E.00325
; LINE_WIDTH: 0.755818
G1 F10475.559
G1 X130.991 Y102.918 E.00306
; LINE_WIDTH: 0.710544
G1 F10682.556
G1 X130.949 Y102.968 E.00286
; LINE_WIDTH: 0.66527
G1 F10891.578
G1 X130.907 Y103.017 E.00267
; LINE_WIDTH: 0.619996
G1 F12223.402
G1 X130.764 Y103.39 E.01527
G1 F13446.369
G1 X130.62 Y103.764 E.01527
G1 X130.204 Y104.833 E.0438
G3 X124.017 Y116.03 I-50.462 J-20.574 E.48955
G1 X125.094 Y116.933 E.05365
G1 X125.394 Y117.287 E.01772
G1 X125.528 Y117.6 E.01299
G1 X125.582 Y117.986 E.0149
G1 X125.511 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.814 Y118.479 E.00346
G1 X121.431 Y122.515 E.20109
G1 X121.165 Y122.72 E.01281
G1 X120.907 Y122.79 E.01021
G1 X120.577 Y122.75 E.01268
G1 X120.334 Y122.612 E.01068
G1 X118.838 Y121.357 E.07454
G3 X112.93 Y126.698 I-38.412 J-36.557 E.30434
G1 X117.317 Y129.952 E.20856
G1 X118.114 Y130.687 E.0414
G1 X118.632 Y131.355 E.03226
G1 X119.048 Y132.093 E.03236
G1 X119.352 Y132.886 E.03239
G1 X119.514 Y133.611 E.02838
G1 X119.587 Y134.425 E.0312
G1 X119.531 Y135.405 E.03748
G1 X119.361 Y136.154 E.0293
G1 X119.08 Y136.91 E.03082
G3 X130.579 Y143.624 I-20.588 J48.466 E.50973
G3 X142.648 Y156.415 I-31.963 J42.246 E.67453
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.52 E2.16535
G1 X142.963 Y99.574 E.02496
G1 X141.891 Y99.578 E.04092
G1 X142.463 Y102.667 E.11993
G3 X142.426 Y103.075 I-.767 J.135 E.01581
G1 X142.209 Y103.388 E.01454
G1 X141.937 Y103.542 E.01195
G1 X141.697 Y103.58 E.00928
G1 X132.894 Y103.58 E.33607
G1 X132.551 Y103.501 E.01344
G3 X132.131 Y102.953 I.435 J-.769 E.02708
G1 X132.145 Y102.573 E.01454
G1 X132.699 Y99.579 E.11624
G3 X131.489 Y99.491 I.189 J-10.896 E.04634
G1 X131.443 Y99.522 E.00215
G3 X123.222 Y116.128 I-51.437 J-15.126 E.711
G1 X124.718 Y117.382 E.07453
G1 X124.933 Y117.672 E.0138
G1 X124.996 Y117.983 E.01212
G1 X124.934 Y118.282 E.01168
G1 X124.909 Y118.325 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.004 Y122.164 E.16693
G1 X120.855 Y122.24 E.00527
G1 X120.689 Y122.188 E.00551
G1 X119.679 Y121.341 E.04172
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.027
; LINE_WIDTH: 0.531516
G1 X118.978 Y120.758 E.00199
; LINE_WIDTH: 0.556136
G1 X118.953 Y120.48 E.00948
G1 X117.999 Y121.433 E.04586
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.967 J-39.379 E.25281
G1 X116.988 Y130.396 E.19622
G1 X117.728 Y131.083 E.03197
G1 X118.192 Y131.689 E.02418
G1 X118.564 Y132.36 E.02427
G1 X118.834 Y133.078 E.02429
G1 X118.973 Y133.725 E.02096
G1 X119.035 Y134.459 E.0233
G1 X118.98 Y135.357 E.02851
G1 X118.824 Y136.026 E.02173
G1 X118.547 Y136.754 E.02465
G1 X118.31 Y137.19 E.01572
G3 X130.249 Y144.067 I-19.567 J47.769 E.43753
G3 X142.39 Y157.025 I-31.645 J41.817 E.56491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.227 E1.81945
G1 X144.167 Y98.939 E.0091
G1 X142.917 Y99.023 E.03966
G3 X141.227 Y99.027 I-1.085 J-100.714 E.05352
G1 X141.919 Y102.763 E.12029
G1 X141.908 Y102.881 E.00376
G1 X141.766 Y103.016 E.00621
G1 X132.894 Y103.027 E.28089
G1 X132.748 Y102.974 E.00493
G1 X132.673 Y102.846 E.0047
G3 X132.999 Y100.994 I66.161 J10.716 E.05953
G1 X133.364 Y99.027 E.06332
G3 X132.128 Y99 I-.309 J-13.991 E.03914
G2 X131.038 Y98.936 I-.884 J5.775 E.03463
G3 X122.647 Y115.978 I-50.883 J-14.467 E.60465
; LINE_WIDTH: 0.552856
G1 X122.357 Y116.419 E.01785
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.806 E.06866
G1 X124.443 Y117.962 E.00556
G1 X124.418 Y118.038 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.685 Y113.972 Z3.64 F36000
G1 X143.035 Y103.036 Z3.64
G1 Z3.24
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.610816
G1 F13660.493
G1 X143.033 Y102.554 E.01812
; LINE_WIDTH: 0.648156
G1 F12829.496
G1 X143.015 Y102.351 E.00817
; LINE_WIDTH: 0.693871
G1 F11940.232
G1 X142.992 Y102.102 E.01075
; LINE_WIDTH: 0.739586
G1 F11166.254
G1 X142.969 Y101.853 E.01149
; LINE_WIDTH: 0.785301
G1 F10486.508
G1 X142.946 Y101.604 E.01224
; LINE_WIDTH: 0.831016
G1 F9884.772
G1 X142.923 Y101.355 E.01298
; LINE_WIDTH: 0.876731
G1 F9348.347
G1 X142.901 Y101.106 E.01373
; LINE_WIDTH: 0.922446
G1 F8867.145
G1 X142.878 Y100.857 E.01447
; LINE_WIDTH: 0.968161
G1 F8433.058
G1 X142.855 Y100.608 E.01522
; LINE_WIDTH: 1.01388
G1 F8039.488
G1 X142.832 Y100.359 E.01596
; WIPE_START
G1 X142.855 Y100.608 E-.095
G1 X142.878 Y100.857 E-.095
G1 X142.901 Y101.106 E-.095
G1 X142.923 Y101.355 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.329 Y102.12 Z3.64 F36000
G1 X131.328 Y102.523 Z3.64
G1 Z3.24
G1 E.4 F1800
; LINE_WIDTH: 1.07274
G1 F7583.783
G1 X131.354 Y102.426 E.00676
; LINE_WIDTH: 1.0564
G1 F7705.028
G1 X131.439 Y102.102 E.02231
; LINE_WIDTH: 1.00785
G1 F8089.258
G1 X131.524 Y101.778 E.02125
; LINE_WIDTH: 0.959303
G1 F8513.821
G1 X131.608 Y101.454 E.02019
; LINE_WIDTH: 0.910756
G1 F8985.419
G1 X131.687 Y101.14 E.01851
; LINE_WIDTH: 0.870606
G1 F9416.816
G1 X131.766 Y100.826 E.01767
; LINE_WIDTH: 0.830456
G1 F9891.725
G1 X131.844 Y100.511 E.01682
; LINE_WIDTH: 0.790306
G1 F10417.081
G1 X131.923 Y100.197 E.01597
; LINE_WIDTH: 0.750156
G1 F11001.37
G1 X131.925 Y100.189 E.00038
; WIPE_START
G1 X131.923 Y100.197 E-.00313
G1 X131.844 Y100.511 E-.12313
G1 X131.766 Y100.826 E-.12313
G1 X131.687 Y101.14 E-.12313
G1 X131.682 Y101.159 E-.00747
; WIPE_END
G1 E-.02 F1800
G1 X127.483 Y107.533 Z3.64 F36000
G1 X118.953 Y120.48 Z3.64
G1 Z3.24
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556136
G1 F3600
M204 S5000
G1 X119.2 Y120.198 E.01275
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15231
; LINE_WIDTH: 0.552856
G1 X122.357 Y116.419 E.01161
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.13057
G1 X121.721 Y117.191 E-.24943
; WIPE_END
G1 E-.02 F1800
G1 X126.954 Y111.634 Z3.64 F36000
G1 X129.583 Y108.843 Z3.64
G1 Z3.24
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.51 Y110.925 I-29.138 J-13.69 E.08945
G2 X130.83 Y109.573 I-1.51 J-5.257 E.10356
G3 X132.715 Y107.779 I10.242 J8.872 E.09952
G3 X136.485 Y107.473 I2.2 J3.732 E.14945
G3 X138.371 Y108.712 I-1.883 J4.918 E.08681
G2 X140.256 Y110.506 I10.241 J-8.872 E.09952
G2 X141.945 Y111.08 I1.933 J-2.917 E.06889
G1 X141.945 Y118.255 E.27392
G3 X140.256 Y117.114 I1.756 J-4.423 E.07845
G2 X138.371 Y115.319 I-10.241 J8.872 E.09952
G2 X134.6 Y115.013 I-2.2 J3.732 E.14945
G2 X132.715 Y116.253 I1.883 J4.918 E.08681
G3 X130.83 Y118.047 I-10.241 J-8.872 E.09952
G3 X126.657 Y118.151 I-2.178 J-3.648 E.16659
G3 X126.202 Y119.412 I-2.88 J-.326 E.05169
G1 X122.841 Y123.429 E.19996
G3 X124.703 Y125.22 I-23.18 J25.968 E.09865
G2 X128.945 Y125.894 I2.671 J-3.127 E.17252
G2 X130.83 Y124.654 I-1.883 J-4.918 E.08681
G3 X132.715 Y122.86 I10.24 J8.871 E.09952
G3 X136.485 Y122.554 I2.2 J3.732 E.14945
G3 X138.371 Y123.793 I-1.883 J4.918 E.08681
G2 X140.256 Y125.588 I10.241 J-8.872 E.09952
G2 X141.945 Y126.161 I1.933 J-2.917 E.06889
G1 X141.945 Y133.336 E.27392
G3 X140.256 Y132.195 I1.756 J-4.423 E.07845
G2 X138.371 Y130.401 I-10.241 J8.872 E.09952
G2 X134.6 Y130.095 I-2.2 J3.732 E.14945
G2 X132.715 Y131.334 I1.883 J4.917 E.08681
G3 X130.83 Y133.128 I-10.242 J-8.873 E.09952
G3 X127.06 Y133.434 I-2.2 J-3.732 E.14945
G3 X125.174 Y132.195 I1.883 J-4.918 E.08681
G2 X123.289 Y130.401 I-10.24 J8.871 E.09952
G2 X119.727 Y130.041 I-2.168 J3.644 E.14113
G3 X120.836 Y132.09 I-7.146 J5.19 E.08924
G1 X141.945 Y138.9 F36000
G1 F13446.283
G1 X141.945 Y141.243 E.08944
G3 X140.256 Y140.669 I.243 J-3.49 E.06889
G3 X138.371 Y138.875 I8.357 J-10.667 E.09952
G2 X136.485 Y137.635 I-3.768 J3.678 E.08681
G2 X132.244 Y138.309 I-1.571 J3.8 E.17252
G3 X129.887 Y140.501 I-19.74 J-18.857 E.12295
G1 X129.434 Y140.729 E.01937
G3 X134.964 Y145.082 I-34.357 J49.335 E.26883
G3 X138.371 Y145.482 I1.257 J3.996 E.13494
G3 X140.256 Y147.276 I-8.356 J10.666 E.09952
G2 X141.945 Y148.417 I3.446 J-3.282 E.07845
G1 X141.945 Y150.76 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 3.4
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y149.76 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L21
M991 S0 P20 ;notify layer change


G17
G3 Z3.64 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z3.4
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00922
G1 X122.328 Y123.268 E.20115
G1 X121.942 Y123.62 E.01991
G1 X121.522 Y123.84 E.01811
G1 X121.016 Y123.957 E.01984
G1 X120.455 Y123.928 E.02144
G1 X119.958 Y123.758 E.02005
G3 X118.918 Y122.954 I5.031 J-7.586 E.05023
G3 X114.848 Y126.662 I-39.228 J-38.972 E.21031
G1 X118.015 Y129.012 E.15054
G3 X118.932 Y129.849 I-4.61 J5.973 E.0475
G1 X119.566 Y130.647 E.03888
G1 X120.076 Y131.532 E.039
G1 X120.45 Y132.48 E.03892
G1 X120.649 Y133.318 E.03287
G3 X120.765 Y134.489 I-5.789 J1.164 E.04501
G1 X120.698 Y135.505 E.03887
G1 X120.538 Y136.268 E.02975
G3 X129.901 Y141.659 I-22.058 J49.138 E.41319
G3 X142.443 Y154.069 I-32.112 J44.995 E.67652
G1 X142.443 Y104.578 E1.88953
G1 X142.301 Y104.659 E.00626
G1 X141.699 Y104.754 E.02324
G1 X132.892 Y104.754 E.33626
G1 X132.596 Y104.731 E.01131
G1 X132.07 Y104.572 E.02101
G1 X131.627 Y104.288 E.02008
G1 X131.242 Y103.844 E.02244
G1 X130.991 Y104.447 E.02496
G3 X124.807 Y115.927 I-51.267 J-20.208 E.49901
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.091 Y117.436 E.02015
G1 X126.168 Y117.978 E.02088
G1 X126.11 Y118.451 E.0182
G1 X125.95 Y118.874 E.01727
G1 X125.896 Y118.955 E.00369
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.324 Y118.777 E.00547
G3 X124.449 Y119.826 I-15.05 J-11.66 E.05218
G1 X121.879 Y122.892 E.15272
G1 X121.609 Y123.138 E.01393
G1 X121.315 Y123.292 E.01267
G1 X120.961 Y123.374 E.01388
G1 X120.52 Y123.343 E.0169
G1 X120.196 Y123.222 E.0132
G3 X118.881 Y122.159 I9.146 J-12.663 E.0646
G3 X113.893 Y126.683 I-38.385 J-37.306 E.25728
G1 X117.666 Y129.482 E.17935
G3 X118.523 Y130.268 I-4.176 J5.415 E.04448
G1 X119.099 Y131.001 E.0356
G1 X119.563 Y131.814 E.03571
G1 X119.901 Y132.684 E.03564
M73 P56 R8
G3 X120.136 Y133.793 I-9.088 J2.498 E.04331
G1 X120.18 Y134.525 E.02799
G1 X120.114 Y135.455 E.03559
G1 X119.931 Y136.289 E.03262
G1 X119.836 Y136.599 E.01237
G3 X130.161 Y142.574 I-21.583 J49.211 E.45641
G3 X142.92 Y155.769 I-31.901 J43.613 E.70419
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y112.027 E1.66924
G3 X143.033 Y103.292 I796.908 J-4 E.33351
; LINE_WIDTH: 0.611936
G1 F13634.003
G1 X143.036 Y103.038 E.00954
G1 X142.973 Y103.281 E.00944
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.597 Y103.831 E.02544
G1 X142.12 Y104.102 E.02094
G1 X141.699 Y104.168 E.01626
G1 X132.892 Y104.168 E.33626
G1 X132.685 Y104.152 E.00791
G1 X132.29 Y104.028 E.0158
G1 X132.007 Y103.842 E.01294
G1 F12802.215
G1 X131.722 Y103.506 E.01681
; LINE_WIDTH: 0.66651
G1 F11305.553
G1 X131.667 Y103.386 E.00547
; LINE_WIDTH: 0.713023
G1 F10872.62
G1 X131.611 Y103.265 E.00587
; LINE_WIDTH: 0.759536
G1 F10448.123
G1 X131.555 Y103.145 E.00627
; LINE_WIDTH: 0.80605
G1 F10032.065
G1 X131.499 Y103.024 E.00668
; LINE_WIDTH: 0.852563
G1 F9624.473
G1 X131.444 Y102.904 E.00708
; LINE_WIDTH: 0.899076
G1 F9106.785
G1 X131.388 Y102.784 E.00748
; LINE_WIDTH: 0.943651
G1 F8660.366
G1 X131.371 Y102.726 E.00357
; LINE_WIDTH: 0.988226
G1 F8255.67
G1 X131.353 Y102.668 E.00374
; LINE_WIDTH: 1.0328
G1 F7887.107
G1 X131.336 Y102.611 E.00392
; LINE_WIDTH: 1.07738
G1 F7550.047
G1 X131.319 Y102.553 E.00409
G1 X131.279 Y102.6 E.00419
; LINE_WIDTH: 1.0328
G1 F7887.107
G1 X131.239 Y102.647 E.00401
; LINE_WIDTH: 0.988226
G1 F8255.67
G1 X131.199 Y102.694 E.00383
; LINE_WIDTH: 0.943651
G1 F8660.366
G1 X131.159 Y102.741 E.00365
; LINE_WIDTH: 0.899076
G1 F9106.785
G1 X131.082 Y102.885 E.00921
; LINE_WIDTH: 0.852563
G1 F9624.473
G1 X131.004 Y103.029 E.00871
; LINE_WIDTH: 0.80605
G1 F10204.567
G1 X130.926 Y103.172 E.00822
; LINE_WIDTH: 0.759536
G1 F10722.257
G1 X130.848 Y103.316 E.00772
; LINE_WIDTH: 0.713023
G1 F11252.784
G1 X130.771 Y103.46 E.00723
; LINE_WIDTH: 0.66651
G1 F11796.091
G1 X130.693 Y103.603 E.00673
; LINE_WIDTH: 0.619996
G1 F13180.552
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.209 Y104.835 E.01998
G3 X124.018 Y116.03 I-50.128 J-20.413 E.48956
G1 X125.095 Y116.932 E.05365
G1 X125.394 Y117.287 E.01771
G1 X125.529 Y117.599 E.01299
G1 X125.582 Y117.986 E.01491
G1 X125.512 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.165 Y122.72 E.01279
G1 X120.906 Y122.791 E.01022
G1 X120.577 Y122.75 E.01268
G1 X120.334 Y122.612 E.01068
G1 X118.837 Y121.358 E.07454
G3 X112.93 Y126.698 I-38.414 J-36.559 E.30431
G1 X117.317 Y129.952 E.20855
G1 X118.114 Y130.687 E.0414
G1 X118.633 Y131.356 E.03231
G1 X119.05 Y132.096 E.03243
G1 X119.352 Y132.887 E.03235
G1 X119.503 Y133.56 E.0263
G1 X119.596 Y134.56 E.03836
G1 X119.531 Y135.404 E.03231
G1 X119.361 Y136.154 E.02934
G1 X119.08 Y136.91 E.03081
G3 X129.816 Y143.047 I-20.529 J48.375 E.4732
G3 X142.647 Y156.415 I-31.728 J43.297 E.71103
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.518 E2.16542
G1 X142.966 Y99.575 E.02485
G1 X141.894 Y99.58 E.04094
G1 X142.466 Y102.669 E.11993
G3 X142.428 Y103.077 I-.767 J.135 E.01581
G1 X142.212 Y103.39 E.01454
G1 X141.939 Y103.544 E.01195
G1 X141.699 Y103.582 E.00928
G1 X132.892 Y103.582 E.33626
G1 X132.548 Y103.503 E.01345
G3 X132.224 Y103.205 I.343 J-.699 E.01704
G1 X132.114 Y102.777 E.01688
G1 X132.126 Y102.662 E.0044
G1 X132.697 Y99.581 E.11963
G3 X131.488 Y99.49 I.19 J-10.667 E.04632
G1 X131.443 Y99.52 E.00205
G3 X123.222 Y116.127 I-51.438 J-15.125 E.71104
G1 X124.718 Y117.381 E.07453
G1 X124.934 Y117.671 E.01379
G1 X124.997 Y117.982 E.01212
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.003 Y122.165 E.16698
G1 X120.855 Y122.24 E.00527
G1 X120.689 Y122.189 E.00551
G1 X119.678 Y121.342 E.04174
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02697
; LINE_WIDTH: 0.531596
G1 X118.977 Y120.758 E.00201
; LINE_WIDTH: 0.556156
G1 X118.953 Y120.481 E.00946
G1 X117.999 Y121.433 E.04583
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.979 J-39.393 E.25281
G1 X116.987 Y130.396 E.19622
G1 X117.728 Y131.083 E.03197
G1 X118.193 Y131.69 E.02422
G1 X118.565 Y132.362 E.02432
G1 X118.834 Y133.08 E.02425
G1 X118.962 Y133.674 E.01924
G1 X119.044 Y134.594 E.02924
G1 X118.98 Y135.356 E.02423
G1 X118.824 Y136.026 E.02177
G1 X118.548 Y136.753 E.02461
G1 X118.31 Y137.19 E.01575
G3 X129.492 Y143.495 I-19.744 J48.085 E.40749
G3 X142.39 Y157.025 I-31.403 J42.846 E.59491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.426 E1.81313
G1 X144.167 Y98.943 E.01531
G2 X142.917 Y99.025 I.097 J11.055 E.03968
G3 X141.23 Y99.029 I-1.056 J-86.228 E.05343
G1 X141.921 Y102.765 E.12029
G1 X141.911 Y102.883 E.00376
G1 X141.769 Y103.019 E.00621
G1 X141.699 Y103.03 E.00223
G1 X132.892 Y103.029 E.27885
G1 X132.745 Y102.976 E.00494
G1 X132.666 Y102.796 E.00622
G1 X132.67 Y102.763 E.00106
G1 X133.361 Y99.029 E.12021
G3 X132.128 Y99.002 I-.308 J-13.659 E.03907
G2 X131.038 Y98.936 I-.884 J5.649 E.03463
G3 X122.648 Y115.977 I-50.882 J-14.466 E.60463
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00556
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.686 Y113.972 Z3.8 F36000
G1 X143.036 Y103.038 Z3.8
G1 Z3.4
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.608276
G1 F13720.948
G1 X143.035 Y102.556 E.01803
; LINE_WIDTH: 0.645876
G1 F12877.328
G1 X143.016 Y102.352 E.0082
; LINE_WIDTH: 0.691591
G1 F11981.652
G1 X142.993 Y102.103 E.01071
; LINE_WIDTH: 0.737306
G1 F11202.471
G1 X142.97 Y101.854 E.01146
; LINE_WIDTH: 0.783021
G1 F10518.443
G1 X142.947 Y101.605 E.0122
; LINE_WIDTH: 0.828736
G1 F9913.143
G1 X142.925 Y101.356 E.01295
; LINE_WIDTH: 0.874451
G1 F9373.718
G1 X142.902 Y101.107 E.01369
; LINE_WIDTH: 0.920166
G1 F8889.968
G1 X142.879 Y100.858 E.01444
; LINE_WIDTH: 0.965881
G1 F8453.699
G1 X142.856 Y100.609 E.01518
; LINE_WIDTH: 1.0116
G1 F8058.245
G1 X142.833 Y100.36 E.01593
; WIPE_START
G1 X142.856 Y100.609 E-.095
G1 X142.879 Y100.858 E-.095
G1 X142.902 Y101.107 E-.095
G1 X142.925 Y101.356 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.332 Y102.139 Z3.8 F36000
G1 X131.319 Y102.553 Z3.8
G1 Z3.4
G1 E.4 F1800
; LINE_WIDTH: 1.07738
G1 F7550.047
G1 X131.329 Y102.514 E.00273
; LINE_WIDTH: 1.07092
G1 F7597.098
G1 X131.396 Y102.258 E.0179
; LINE_WIDTH: 1.03161
G1 F7896.558
G1 X131.463 Y102.002 E.01722
; LINE_WIDTH: 0.992296
G1 F8220.595
G1 X131.531 Y101.746 E.01654
; LINE_WIDTH: 0.952986
G1 F8572.363
G1 X131.598 Y101.49 E.01586
; LINE_WIDTH: 0.913676
G1 F8955.581
G1 X131.608 Y101.454 E.00213
; LINE_WIDTH: 0.908636
G1 F9007.207
G1 X131.686 Y101.139 E.01846
; LINE_WIDTH: 0.868486
G1 F9440.749
G1 X131.765 Y100.825 E.01762
; LINE_WIDTH: 0.828336
G1 F9918.137
G1 X131.843 Y100.511 E.01677
; LINE_WIDTH: 0.788186
G1 F10446.376
G1 X131.922 Y100.197 E.01592
; LINE_WIDTH: 0.748036
G1 F11034.048
G1 X131.924 Y100.189 E.00036
; WIPE_START
G1 X131.922 Y100.197 E-.00293
G1 X131.843 Y100.511 E-.12311
G1 X131.765 Y100.825 E-.12311
G1 X131.686 Y101.139 E-.12311
G1 X131.681 Y101.159 E-.00775
; WIPE_END
G1 E-.02 F1800
G1 X127.482 Y107.533 Z3.8 F36000
G1 X118.953 Y120.481 Z3.8
G1 Z3.4
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556156
G1 F3600
M204 S5000
G1 X119.2 Y120.198 E.01278
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.687 E.15231
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.687 E-.1308
G1 X121.722 Y117.19 E-.2492
; WIPE_END
G1 E-.02 F1800
G1 X126.959 Y111.638 Z3.8 F36000
G1 X129.584 Y108.856 Z3.8
G1 Z3.4
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.505 Y110.936 I-37.978 J-18.38 E.08944
G2 X130.83 Y109.707 I-1.189 J-5.064 E.1015
G3 X133.186 Y107.642 I9.728 J8.724 E.11991
G3 X138.371 Y108.578 I1.899 J4.308 E.21345
G2 X140.727 Y110.643 I9.728 J-8.725 E.11991
G2 X141.945 Y111.009 I1.52 J-2.845 E.0489
G1 X141.945 Y118.295 E.27818
G3 X140.256 Y117.247 I1.458 J-4.237 E.07655
G2 X137.899 Y115.183 I-9.728 J8.725 E.11991
G2 X132.715 Y116.119 I-1.899 J4.308 E.21345
G3 X130.359 Y118.184 I-9.727 J-8.724 E.11991
G3 X126.655 Y118.201 I-1.871 J-4.136 E.14564
G3 X126.202 Y119.412 I-2.485 J-.24 E.04991
G1 X122.897 Y123.363 E.19669
G3 X125.174 Y125.433 I-51.416 J58.86 E.11752
G2 X129.887 Y125.5 I2.41 J-3.735 E.18967
G2 X131.772 Y123.849 I-5.049 J-7.668 E.09597
G3 X134.6 Y122.315 I3.551 J3.172 E.12529
G3 X138.371 Y123.659 I.405 J4.822 E.15756
G2 X140.727 Y125.724 I9.728 J-8.725 E.11991
G2 X141.945 Y126.091 I1.52 J-2.845 E.0489
G1 X141.945 Y133.377 E.27818
G3 X140.256 Y132.329 I1.458 J-4.237 E.07655
G2 X137.899 Y130.264 I-9.728 J8.724 E.11991
G2 X132.715 Y131.2 I-1.899 J4.308 E.21345
G3 X130.359 Y133.265 I-9.728 J-8.724 E.11991
G3 X125.174 Y132.329 I-1.899 J-4.308 E.21345
G2 X122.818 Y130.264 I-9.727 J8.724 E.11991
G2 X119.713 Y130.023 I-1.857 J3.81 E.12174
G3 X120.828 Y132.07 I-7.108 J5.196 E.08923
G1 X141.945 Y138.829 F36000
G1 F13446.283
G1 X141.945 Y141.172 E.08944
G3 X140.727 Y140.806 I.301 J-3.211 E.0489
G3 X138.371 Y138.741 I7.371 J-10.789 E.11991
G2 X134.6 Y137.396 I-3.365 J3.477 E.15756
G2 X131.772 Y138.93 I.723 J4.706 E.12529
G3 X129.887 Y140.581 I-6.934 J-6.017 E.09597
G1 X129.492 Y140.759 E.01654
G3 X134.951 Y145.071 I-42.804 J59.794 E.26571
G3 X137.899 Y145.345 I1.127 J3.869 E.11575
G3 X140.256 Y147.41 I-7.371 J10.789 E.11991
G2 X141.945 Y148.458 I3.148 J-3.189 E.07655
G1 X141.945 Y150.801 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 3.56
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y149.801 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L22
M991 S0 P21 ;notify layer change


G17
G3 Z3.8 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z3.56
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00922
G1 X122.328 Y123.268 E.20115
G1 X121.942 Y123.619 E.01991
G1 X121.523 Y123.84 E.01809
G1 X121.016 Y123.957 E.01986
G1 X120.455 Y123.928 E.02143
G1 X119.96 Y123.759 E.02001
G3 X118.918 Y122.954 I5.013 J-7.567 E.05028
G3 X114.848 Y126.662 I-39.407 J-39.168 E.2103
G1 X118.015 Y129.012 E.15055
G3 X118.933 Y129.849 I-4.613 J5.975 E.04748
G1 X119.566 Y130.648 E.03893
G1 X120.076 Y131.532 E.03895
G1 X120.45 Y132.478 E.03886
G3 X120.765 Y134.489 I-7.443 J2.195 E.07791
G1 X120.698 Y135.506 E.03893
G1 X120.538 Y136.268 E.02973
G3 X131.285 Y142.689 I-21.871 J48.81 E.47907
G3 X142.443 Y154.069 I-32.726 J43.249 E.61068
G1 X142.443 Y104.581 E1.8894
G1 X142.303 Y104.661 E.00615
G1 X141.702 Y104.756 E.02324
G1 X132.889 Y104.756 E.33646
G1 X132.593 Y104.733 E.01133
G1 X132.066 Y104.574 E.02102
G1 X131.623 Y104.289 E.02011
G1 X131.238 Y103.844 E.02247
G3 X124.807 Y115.927 I-51.8 J-19.815 E.52395
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.091 Y117.436 E.02015
G1 X126.168 Y117.978 E.02088
G1 X126.11 Y118.451 E.0182
G1 X125.95 Y118.874 E.01727
G1 X125.896 Y118.955 E.00369
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.324 Y118.777 E.00547
G3 X124.449 Y119.826 I-15.078 J-11.683 E.05218
G1 X121.879 Y122.892 E.15272
G1 X121.609 Y123.138 E.01393
G1 X121.316 Y123.292 E.01266
G1 X120.961 Y123.374 E.01389
G1 X120.52 Y123.343 E.01689
G1 X120.196 Y123.222 E.0132
G3 X118.881 Y122.159 I9.15 J-12.668 E.0646
G3 X113.893 Y126.683 I-41.605 J-40.855 E.25725
G1 X117.666 Y129.482 E.17936
G3 X118.523 Y130.268 I-4.176 J5.415 E.04446
G1 X119.1 Y131.003 E.03564
G1 X119.563 Y131.814 E.03566
G1 X119.901 Y132.682 E.03558
G3 X120.135 Y133.79 I-9.576 J2.603 E.04325
G1 X120.18 Y134.524 E.02809
G1 X120.114 Y135.456 E.03564
G1 X119.933 Y136.283 E.03234
G1 X119.836 Y136.599 E.01262
G3 X130.932 Y143.156 I-21.129 J48.421 E.49332
G3 X142.92 Y155.769 I-32.37 J42.773 E.6673
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y112.028 E1.66921
G3 X143.034 Y103.293 I674.018 J-4 E.33348
; LINE_WIDTH: 0.610486
G1 F13668.317
G1 X143.037 Y103.04 E.00951
G1 X142.975 Y103.283 E.00941
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.599 Y103.833 E.02543
G1 X142.122 Y104.104 E.02094
G1 X141.702 Y104.17 E.01626
G1 X132.889 Y104.17 E.33646
G1 X132.682 Y104.154 E.00793
G1 X132.287 Y104.03 E.01581
G1 X132.003 Y103.843 E.01296
G1 F12442.932
G1 X131.717 Y103.503 E.01698
; LINE_WIDTH: 0.668837
G1 F10953.063
G1 X131.673 Y103.395 E.00483
; LINE_WIDTH: 0.717678
G1 F10577.331
G1 X131.63 Y103.286 E.00521
; LINE_WIDTH: 0.76652
G1 F10208.156
G1 X131.586 Y103.178 E.00558
; LINE_WIDTH: 0.815361
G1 F9845.55
G1 X131.543 Y103.069 E.00595
; LINE_WIDTH: 0.864202
G1 F9489.49
G1 X131.499 Y102.961 E.00632
; LINE_WIDTH: 0.913043
G1 F8962.036
G1 X131.456 Y102.853 E.0067
; LINE_WIDTH: 0.961884
G1 F8490.129
G1 X131.412 Y102.744 E.00707
; LINE_WIDTH: 1.01073
G1 F8065.435
G1 X131.368 Y102.636 E.00744
; LINE_WIDTH: 1.05957
G1 F7681.203
G1 X131.325 Y102.527 E.00781
G1 X131.278 Y102.582 E.0048
; LINE_WIDTH: 1.01073
G1 F8065.435
G1 X131.232 Y102.637 E.00457
; LINE_WIDTH: 0.961884
G1 F8490.129
G1 X131.185 Y102.691 E.00434
; LINE_WIDTH: 0.913043
G1 F8962.036
G1 X131.139 Y102.746 E.00411
; LINE_WIDTH: 0.864202
G1 F9489.49
G1 X131.092 Y102.801 E.00389
; LINE_WIDTH: 0.815361
G1 F10082.911
G1 X131.046 Y102.855 E.00366
; LINE_WIDTH: 0.76652
G1 F10307.639
G1 X130.999 Y102.91 E.00343
; LINE_WIDTH: 0.717678
G1 F10534.844
G1 X130.952 Y102.965 E.0032
; LINE_WIDTH: 0.668837
G1 F10764.525
G1 X130.906 Y103.02 E.00297
; LINE_WIDTH: 0.619996
G1 F12088.784
G1 X130.763 Y103.393 E.01527
G1 F13446.369
G1 X130.619 Y103.766 E.01527
G1 X130.204 Y104.833 E.04369
G3 X124.018 Y116.03 I-50.462 J-20.574 E.48953
G1 X125.094 Y116.932 E.05365
G1 X125.394 Y117.286 E.01771
G1 X125.529 Y117.599 E.01299
G1 X125.582 Y117.986 E.01491
G1 X125.512 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.165 Y122.72 E.01278
G1 X120.906 Y122.791 E.01023
G1 X120.577 Y122.75 E.01269
G1 X120.334 Y122.612 E.01068
G1 X118.837 Y121.358 E.07454
G3 X112.93 Y126.698 I-40.624 J-39.004 E.30428
G1 X117.317 Y129.952 E.20857
G1 X118.114 Y130.687 E.04139
G1 X118.634 Y131.357 E.03236
G1 X119.05 Y132.096 E.03238
G1 X119.352 Y132.886 E.0323
G3 X119.551 Y133.826 I-11.668 J2.958 E.03668
G1 X119.595 Y134.56 E.02809
G1 X119.531 Y135.405 E.03235
G1 X119.363 Y136.147 E.02906
G1 X119.08 Y136.91 E.03106
G3 X130.579 Y143.623 I-20.591 J48.471 E.5097
G3 X142.648 Y156.415 I-31.96 J42.244 E.67456
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.519 E2.16539
G1 X142.966 Y99.577 E.02485
G1 X141.896 Y99.583 E.04084
G1 X142.468 Y102.671 E.11993
G3 X142.431 Y103.079 I-.767 J.135 E.01581
G1 X142.214 Y103.392 E.01454
G1 X141.942 Y103.546 E.01195
G1 X141.702 Y103.584 E.00928
G1 X132.889 Y103.584 E.33646
G1 X132.545 Y103.504 E.01347
G3 X132.125 Y102.956 I.436 J-.769 E.02709
G1 X132.141 Y102.572 E.01471
G1 X132.694 Y99.583 E.11605
G3 X131.486 Y99.49 I.191 J-10.437 E.04629
G1 X131.444 Y99.519 E.00195
G3 X123.222 Y116.127 I-51.437 J-15.123 E.7111
G1 X124.718 Y117.381 E.07453
G1 X124.934 Y117.671 E.01379
G1 X124.997 Y117.982 E.01212
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.003 Y122.165 E.16698
G1 X120.855 Y122.24 E.00527
G1 X120.689 Y122.189 E.00551
G1 X119.678 Y121.342 E.04174
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02697
; LINE_WIDTH: 0.531596
G1 X118.977 Y120.758 E.00201
; LINE_WIDTH: 0.556156
G1 X118.953 Y120.481 E.00946
G1 X117.999 Y121.433 E.04583
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.723 J-39.102 E.25281
G1 X116.988 Y130.396 E.19623
G1 X117.728 Y131.083 E.03196
G1 X118.194 Y131.691 E.02426
G1 X118.565 Y132.362 E.02428
G1 X118.834 Y133.078 E.02422
G1 X118.953 Y133.632 E.01792
G1 X119.036 Y134.469 E.02664
G1 X118.98 Y135.357 E.02818
G1 X118.825 Y136.02 E.02153
G1 X118.548 Y136.753 E.02482
G1 X118.31 Y137.19 E.01575
G3 X130.248 Y144.067 I-19.564 J47.764 E.43751
G3 X142.39 Y157.025 I-31.644 J41.817 E.56493
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.424 E1.81319
G1 X144.167 Y98.943 E.01524
G2 X142.916 Y99.027 I.098 J10.843 E.03972
G3 X141.232 Y99.031 I-1.045 J-80.925 E.05332
G1 X141.924 Y102.767 E.12029
G1 X141.913 Y102.885 E.00376
M73 P57 R8
G1 X141.771 Y103.021 E.00621
G1 X132.889 Y103.032 E.28121
G1 X132.743 Y102.978 E.00494
G1 X132.668 Y102.85 E.0047
G3 X132.994 Y100.998 I66.891 J10.848 E.05952
G1 X133.359 Y99.032 E.06332
G3 X132.127 Y99.003 I-.306 J-13.336 E.039
G2 X131.038 Y98.936 I-.884 J5.525 E.03462
G3 X122.648 Y115.977 I-50.882 J-14.466 E.60463
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00556
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.686 Y113.972 Z3.96 F36000
G1 X143.037 Y103.04 Z3.96
G1 Z3.56
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.605736
G1 F13781.941
G1 X143.036 Y102.559 E.01793
; LINE_WIDTH: 0.643586
G1 F12925.728
G1 X143.017 Y102.353 E.00822
; LINE_WIDTH: 0.689303
G1 F12023.521
G1 X142.994 Y102.104 E.01067
; LINE_WIDTH: 0.735019
G1 F11239.044
G1 X142.971 Y101.855 E.01142
; LINE_WIDTH: 0.780735
G1 F10550.661
G1 X142.949 Y101.606 E.01216
; LINE_WIDTH: 0.826451
G1 F9941.739
G1 X142.926 Y101.357 E.01291
; LINE_WIDTH: 0.872167
G1 F9399.268
G1 X142.903 Y101.108 E.01365
; LINE_WIDTH: 0.917884
G1 F8912.934
G1 X142.88 Y100.859 E.0144
; LINE_WIDTH: 0.9636
G1 F8474.451
G1 X142.857 Y100.61 E.01514
; LINE_WIDTH: 1.00932
G1 F8077.089
G1 X142.834 Y100.361 E.01589
; WIPE_START
G1 X142.857 Y100.61 E-.095
G1 X142.88 Y100.859 E-.095
G1 X142.903 Y101.108 E-.095
G1 X142.926 Y101.357 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.331 Y102.116 Z3.96 F36000
G1 X131.328 Y102.517 Z3.96
G1 Z3.56
G1 E.4 F1800
; LINE_WIDTH: 1.06758
G1 F7621.656
G1 X131.352 Y102.426 E.00636
; LINE_WIDTH: 1.05222
G1 F7736.669
G1 X131.437 Y102.102 E.02222
; LINE_WIDTH: 1.00367
G1 F8124.141
G1 X131.522 Y101.778 E.02116
; LINE_WIDTH: 0.955123
G1 F8552.47
G1 X131.606 Y101.454 E.0201
; LINE_WIDTH: 0.906576
G1 F9028.479
G1 X131.685 Y101.139 E.01842
; LINE_WIDTH: 0.866426
G1 F9464.122
G1 X131.764 Y100.825 E.01758
; LINE_WIDTH: 0.826276
G1 F9943.936
G1 X131.842 Y100.511 E.01673
; LINE_WIDTH: 0.786126
G1 F10475.001
G1 X131.921 Y100.196 E.01588
; LINE_WIDTH: 0.745976
G1 F11065.989
G1 X131.923 Y100.189 E.00034
; WIPE_START
G1 X131.921 Y100.196 E-.00275
G1 X131.842 Y100.511 E-.12313
G1 X131.764 Y100.825 E-.12313
G1 X131.685 Y101.139 E-.12313
G1 X131.68 Y101.159 E-.00785
; WIPE_END
G1 E-.02 F1800
G1 X127.481 Y107.533 Z3.96 F36000
G1 X118.953 Y120.481 Z3.96
G1 Z3.56
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556156
G1 F3600
M204 S5000
G1 X119.2 Y120.198 E.01277
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.687 E.15231
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.687 E-.13081
G1 X121.722 Y117.19 E-.24919
; WIPE_END
G1 E-.02 F1800
G1 X126.959 Y111.638 Z3.96 F36000
G1 X129.57 Y108.87 Z3.96
G1 Z3.56
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.497 Y110.951 I-29.132 J-13.703 E.08945
G2 X130.83 Y109.841 I-.885 J-4.865 E.09982
G3 X132.715 Y108.08 I27.62 J27.682 E.0985
G3 X138.371 Y108.444 I2.6 J3.714 E.23373
G2 X140.256 Y110.205 I27.624 J-27.687 E.0985
G1 X141.198 Y110.746 E.04149
G1 X141.945 Y110.945 E.02952
G1 X141.945 Y118.341 E.28239
G3 X140.256 Y117.381 I1.178 J-4.042 E.07486
G2 X138.371 Y115.621 I-27.615 J27.678 E.0985
G2 X132.715 Y115.985 I-2.6 J3.714 E.23373
G3 X130.83 Y117.746 I-27.615 J-27.678 E.0985
G3 X126.648 Y118.255 I-2.564 J-3.629 E.16756
G3 X126.202 Y119.412 I-2.461 J-.284 E.04784
G1 X122.956 Y123.292 E.19314
G3 X125.174 Y125.286 I-25.288 J30.354 E.11391
G2 X130.83 Y124.922 I2.6 J-3.714 E.23373
G3 X132.715 Y123.161 I27.62 J27.682 E.0985
G3 X138.371 Y123.525 I2.6 J3.714 E.23373
G2 X140.256 Y125.286 I27.624 J-27.687 E.0985
G1 X141.198 Y125.827 E.04149
G1 X141.945 Y126.026 E.02952
G1 X141.945 Y133.422 E.28239
G3 X140.256 Y132.463 I1.178 J-4.042 E.07486
G2 X138.371 Y130.702 I-27.606 J27.668 E.0985
G2 X132.715 Y131.066 I-2.6 J3.714 E.23373
G3 X130.83 Y132.827 I-27.62 J-27.682 E.0985
G3 X125.174 Y132.463 I-2.6 J-3.714 E.23373
G2 X123.289 Y130.702 I-27.611 J27.673 E.0985
G2 X119.695 Y130 I-2.53 J3.4 E.14457
G3 X120.816 Y132.043 I-5.755 J4.486 E.08934
G1 X141.945 Y138.764 F36000
G1 F13446.283
G1 X141.945 Y141.107 E.08944
G1 X141.198 Y140.908 E.02952
G1 X140.256 Y140.367 E.04149
G3 X138.371 Y138.607 I25.739 J-29.448 E.0985
G2 X133.658 Y137.702 I-3.128 J3.566 E.1921
G2 X131.772 Y139.094 I2.267 J5.043 E.09016
G3 X129.887 Y140.665 I-5.956 J-5.229 E.09407
G1 X129.542 Y140.803 E.0142
G3 X134.944 Y145.054 I-36.165 J51.517 E.26259
G3 X138.371 Y145.783 I.949 J3.955 E.13828
G3 X140.256 Y147.544 I-25.73 J29.439 E.0985
G2 X141.945 Y148.503 I2.868 J-3.083 E.07486
G1 X141.945 Y150.846 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 3.72
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y149.846 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L23
M991 S0 P22 ;notify layer change


G17
G3 Z3.96 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z3.72
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00922
G1 X122.328 Y123.268 E.20115
G1 X121.921 Y123.634 E.02088
G1 X121.47 Y123.859 E.01925
G1 X120.934 Y123.963 E.02086
G1 X120.385 Y123.913 E.02102
G1 X119.922 Y123.739 E.01887
G3 X118.918 Y122.954 I5.678 J-8.297 E.0487
G3 X114.848 Y126.662 I-39.397 J-39.158 E.21031
G1 X118.015 Y129.012 E.15056
G1 X118.484 Y129.402 E.02331
G1 X119.157 Y130.105 E.03713
G1 X119.749 Y130.932 E.03884
G1 X120.215 Y131.838 E.03891
G1 X120.498 Y132.652 E.0329
G1 X120.687 Y133.5 E.03317
G1 X120.765 Y134.489 E.03787
G1 X120.698 Y135.507 E.03893
G1 X120.538 Y136.268 E.02971
G3 X130.507 Y142.101 I-22.431 J49.771 E.44177
G3 X142.443 Y154.069 I-32.644 J44.495 E.6479
G1 X142.443 Y104.585 E1.88926
G1 X142.306 Y104.663 E.00604
G1 X141.704 Y104.758 E.02324
G1 X132.887 Y104.758 E.33665
G1 X132.59 Y104.735 E.01134
G1 X132.063 Y104.575 E.02104
G1 X131.62 Y104.29 E.02012
G1 X131.231 Y103.846 E.02252
G3 X124.807 Y115.927 I-51.459 J-19.614 E.52376
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.091 Y117.436 E.02015
G1 X126.168 Y117.978 E.02088
G1 X126.11 Y118.451 E.0182
G1 X125.95 Y118.874 E.01727
G1 X125.896 Y118.955 E.00369
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.324 Y118.777 E.00547
G3 X124.449 Y119.826 I-15.078 J-11.683 E.05218
G1 X121.879 Y122.892 E.15272
G1 X121.595 Y123.148 E.01461
G1 X121.188 Y123.333 E.01707
G1 X120.736 Y123.376 E.01732
G1 X120.383 Y123.303 E.01376
G1 X119.957 Y123.061 E.01871
G1 X118.881 Y122.159 E.05363
G3 X113.893 Y126.683 I-41.603 J-40.853 E.25725
G1 X117.666 Y129.482 E.17937
G1 X118.107 Y129.85 E.02193
G1 X118.727 Y130.503 E.03439
G1 X119.266 Y131.263 E.03556
G1 X119.689 Y132.095 E.03563
G3 X120.112 Y133.613 I-6.424 J2.612 E.06029
G1 X120.18 Y134.525 E.03491
G1 X120.114 Y135.456 E.03564
G1 X119.932 Y136.285 E.03239
G1 X119.835 Y136.599 E.01256
G3 X130.932 Y143.157 I-21.102 J48.375 E.49333
G3 X142.92 Y155.769 I-32.175 J42.587 E.66732
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y110.029 E1.74553
G3 X143.034 Y103.295 I1676.696 J-2 E.2571
; LINE_WIDTH: 0.609016
G1 F13703.279
G1 X143.038 Y103.042 E.00948
G1 X142.977 Y103.285 E.00938
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.602 Y103.835 E.02542
G1 X142.125 Y104.106 E.02094
G1 X141.704 Y104.172 E.01626
G1 X132.887 Y104.172 E.33665
G1 X132.679 Y104.156 E.00794
G1 X132.284 Y104.032 E.01583
G1 X132 Y103.845 E.01297
G1 F12382.242
G1 X131.713 Y103.504 E.017
; LINE_WIDTH: 0.66975
G1 F10894.639
G1 X131.67 Y103.396 E.00483
; LINE_WIDTH: 0.719503
G1 F10520.592
G1 X131.627 Y103.288 E.00521
; LINE_WIDTH: 0.769256
G1 F10153.061
G1 X131.583 Y103.179 E.00559
; LINE_WIDTH: 0.81901
G1 F9792.093
G1 X131.54 Y103.071 E.00597
; LINE_WIDTH: 0.868763
G1 F9437.619
G1 X131.497 Y102.963 E.00635
; LINE_WIDTH: 0.918516
G1 F8906.559
G1 X131.453 Y102.854 E.00673
; LINE_WIDTH: 0.96827
G1 F8432.08
G1 X131.41 Y102.746 E.0071
; LINE_WIDTH: 1.01802
G1 F8005.599
G1 X131.367 Y102.638 E.00748
; LINE_WIDTH: 1.06778
G1 F7620.181
G1 X131.323 Y102.529 E.00786
G1 X131.277 Y102.584 E.00483
; LINE_WIDTH: 1.01802
G1 F8005.599
G1 X131.23 Y102.639 E.0046
; LINE_WIDTH: 0.96827
G1 F8432.08
G1 X131.184 Y102.693 E.00436
; LINE_WIDTH: 0.918516
G1 F8906.559
G1 X131.138 Y102.748 E.00413
; LINE_WIDTH: 0.868763
G1 F9437.619
G1 X131.091 Y102.802 E.0039
; LINE_WIDTH: 0.81901
G1 F10036.024
G1 X131.045 Y102.857 E.00367
; LINE_WIDTH: 0.769256
G1 F10259.705
G1 X130.998 Y102.912 E.00343
; LINE_WIDTH: 0.719503
G1 F10485.85
G1 X130.952 Y102.966 E.0032
; LINE_WIDTH: 0.66975
G1 F10714.441
G1 X130.905 Y103.021 E.00297
; LINE_WIDTH: 0.619996
G1 F12035.704
G1 X130.762 Y103.394 E.01527
G1 F13433.767
G1 X130.619 Y103.768 E.01527
G1 F13446.369
G1 X130.475 Y104.141 E.01527
G1 X130.204 Y104.833 E.02837
G3 X124.018 Y116.03 I-50.462 J-20.574 E.48952
G1 X125.094 Y116.932 E.05365
G1 X125.394 Y117.286 E.01771
G1 X125.529 Y117.599 E.01299
G1 X125.582 Y117.986 E.01491
G1 X125.512 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.169 Y122.718 E.01264
G1 X120.874 Y122.793 E.01162
G1 X120.577 Y122.75 E.01145
G1 X120.334 Y122.612 E.01068
G1 X118.837 Y121.358 E.07455
G3 X112.93 Y126.698 I-40.623 J-39.003 E.30428
G1 X117.317 Y129.953 E.20857
G1 X117.73 Y130.298 E.02056
G1 X118.298 Y130.902 E.03165
G1 X118.783 Y131.594 E.03228
G3 X119.538 Y133.726 I-5.663 J3.204 E.08676
G1 X119.596 Y134.56 E.03194
G1 X119.531 Y135.405 E.03236
G1 X119.363 Y136.149 E.02911
G1 X119.08 Y136.91 E.031
G3 X130.579 Y143.624 I-20.648 J48.568 E.5097
G3 X142.648 Y156.415 I-31.779 J42.073 E.67458
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.519 E2.16536
G1 X142.966 Y99.579 E.02486
G1 X141.899 Y99.585 E.04075
G1 X142.471 Y102.673 E.11993
G3 X142.433 Y103.081 I-.767 J.135 E.01581
G1 X142.217 Y103.394 E.01454
G1 X141.944 Y103.549 E.01195
G1 X141.704 Y103.586 E.00928
G1 X132.887 Y103.586 E.33665
G1 X132.543 Y103.506 E.01348
G3 X132.123 Y102.958 I.436 J-.769 E.02709
G1 X132.139 Y102.571 E.01479
G1 X132.692 Y99.585 E.11595
G3 X131.485 Y99.49 I.191 J-10.207 E.04626
G1 X131.444 Y99.518 E.00187
G3 X123.222 Y116.127 I-51.438 J-15.122 E.71115
G1 X124.718 Y117.381 E.07453
G1 X124.934 Y117.671 E.01379
G1 X124.997 Y117.982 E.01212
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.007 Y122.161 E.16681
G1 X120.845 Y122.241 E.00571
G1 X120.689 Y122.189 E.00523
G1 X119.678 Y121.342 E.04174
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02697
; LINE_WIDTH: 0.531626
G1 X118.977 Y120.758 E.00202
; LINE_WIDTH: 0.556176
G1 X118.953 Y120.481 E.00946
G1 X117.999 Y121.433 E.04584
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.722 J-39.101 E.25281
G1 X116.988 Y130.397 E.19623
G3 X118.995 Y133.832 I-3.265 J4.212 E.1291
G1 X119.044 Y134.594 E.02417
G1 X118.98 Y135.358 E.02427
G1 X118.825 Y136.021 E.02157
G1 X118.548 Y136.753 E.02479
G1 X118.31 Y137.19 E.01573
G3 X130.248 Y144.067 I-19.805 J48.182 E.43749
G3 X142.39 Y157.025 I-31.645 J41.817 E.56493
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.423 E1.81324
G1 X144.167 Y98.943 E.01519
G2 X142.915 Y99.029 I.1 J10.643 E.03975
G3 X141.235 Y99.033 I-1.033 J-75.94 E.05321
G1 X141.926 Y102.769 E.12029
G1 X141.916 Y102.887 E.00376
G1 X141.774 Y103.023 E.00621
G1 X132.887 Y103.034 E.28137
G1 X132.74 Y102.98 E.00495
G1 X132.665 Y102.852 E.00469
G3 X132.992 Y101 I67.265 J10.916 E.05952
G1 X133.356 Y99.034 E.06332
G3 X132.127 Y99.004 I-.305 J-13.035 E.03894
G2 X131.038 Y98.936 I-.885 J5.405 E.03462
G3 X122.648 Y115.977 I-50.883 J-14.466 E.60463
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00556
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.687 Y113.973 Z4.12 F36000
G1 X143.038 Y103.042 Z4.12
G1 Z3.72
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.603216
G1 F13842.992
G1 X143.037 Y102.561 E.01784
; LINE_WIDTH: 0.641326
G1 F12973.856
G1 X143.018 Y102.354 E.00825
; LINE_WIDTH: 0.687042
G1 F12065.154
G1 X142.995 Y102.105 E.01064
; LINE_WIDTH: 0.732759
G1 F11275.411
G1 X142.973 Y101.856 E.01138
; LINE_WIDTH: 0.778475
G1 F10582.705
G1 X142.95 Y101.607 E.01213
; LINE_WIDTH: 0.824191
G1 F9970.185
G1 X142.927 Y101.358 E.01287
; LINE_WIDTH: 0.869907
G1 F9424.691
G1 X142.904 Y101.109 E.01362
; LINE_WIDTH: 0.915624
G1 F8935.79
G1 X142.881 Y100.86 E.01436
; LINE_WIDTH: 0.96134
G1 F8495.112
G1 X142.858 Y100.611 E.01511
; LINE_WIDTH: 1.00706
G1 F8095.855
G1 X142.835 Y100.362 E.01585
; WIPE_START
G1 X142.858 Y100.611 E-.095
G1 X142.881 Y100.86 E-.095
G1 X142.904 Y101.109 E-.095
G1 X142.927 Y101.358 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.333 Y102.124 Z4.12 F36000
G1 X131.323 Y102.529 Z4.12
G1 Z3.72
G1 E.4 F1800
; LINE_WIDTH: 1.06778
G1 F7620.181
G1 X131.351 Y102.425 E.00727
; LINE_WIDTH: 1.05012
G1 F7752.663
G1 X131.436 Y102.101 E.02217
; LINE_WIDTH: 1.00157
G1 F8141.779
G1 X131.521 Y101.777 E.02111
; LINE_WIDTH: 0.953023
G1 F8572.02
G1 X131.605 Y101.453 E.02005
; LINE_WIDTH: 0.904476
G1 F9050.269
G1 X131.684 Y101.139 E.01839
; LINE_WIDTH: 0.864316
G1 F9488.182
G1 X131.763 Y100.825 E.01754
; LINE_WIDTH: 0.824156
G1 F9970.627
G1 X131.841 Y100.51 E.01669
; LINE_WIDTH: 0.783996
G1 F10504.763
G1 X131.92 Y100.196 E.01584
; LINE_WIDTH: 0.743836
G1 F11099.367
G1 X131.922 Y100.19 E.00029
; WIPE_START
G1 X131.92 Y100.196 E-.00241
G1 X131.841 Y100.51 E-.12317
G1 X131.763 Y100.825 E-.12316
G1 X131.684 Y101.139 E-.12316
G1 X131.679 Y101.16 E-.0081
; WIPE_END
G1 E-.02 F1800
G1 X127.481 Y107.534 Z4.12 F36000
G1 X118.953 Y120.481 Z4.12
G1 Z3.72
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556176
G1 F3600
M204 S5000
G1 X119.2 Y120.199 E.01276
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15232
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.13088
G1 X121.722 Y117.19 E-.24912
; WIPE_END
G1 E-.02 F1800
G1 X126.965 Y111.644 Z4.12 F36000
G1 X129.565 Y108.893 Z4.12
G1 Z3.72
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.486 Y110.972 I-37.977 J-18.403 E.08944
G2 X130.83 Y109.977 I-.598 J-4.667 E.09847
G3 X132.715 Y108.222 I112.808 J119.291 E.09833
G3 X138.371 Y108.308 I2.775 J3.493 E.2339
G2 X140.256 Y110.063 I112.801 J-119.285 E.09833
G1 X141.198 Y110.649 E.04237
G1 X141.945 Y110.885 E.02992
G1 X141.945 Y118.392 E.28658
G1 X141.198 Y118.132 E.0302
G3 X139.313 Y116.633 I3.276 J-6.055 E.09243
G2 X136.485 Y114.878 I-4.178 J3.578 E.12907
G2 X132.715 Y115.849 I-.858 J4.475 E.15358
G3 X130.83 Y117.604 I-112.808 J-119.291 E.09833
G3 X126.641 Y118.314 I-2.76 J-3.57 E.16891
G3 X126.202 Y119.412 I-2.45 J-.341 E.04558
G1 X123.021 Y123.215 E.1893
G1 X123.289 Y123.39 E.01221
G2 X125.174 Y125.145 I112.726 J-119.204 E.09833
G2 X130.83 Y125.058 I2.775 J-3.493 E.2339
G3 X132.715 Y123.303 I112.726 J119.204 E.09833
G3 X138.371 Y123.39 I2.775 J3.493 E.2339
G2 X140.256 Y125.145 I112.801 J-119.285 E.09833
G1 X141.198 Y125.73 E.04237
G1 X141.945 Y125.967 E.02992
G1 X141.945 Y133.473 E.28658
G1 X141.198 Y133.213 E.0302
G3 X139.313 Y131.714 I3.276 J-6.055 E.09243
G2 X136.485 Y129.96 I-4.178 J3.578 E.12907
G2 X132.715 Y130.93 I-.858 J4.474 E.15358
G3 X130.83 Y132.685 I-112.726 J-119.204 E.09833
G3 X125.174 Y132.598 I-2.775 J-3.493 E.2339
G2 X123.289 Y130.844 I-112.808 J119.291 E.09833
G2 X119.686 Y129.969 I-2.721 J3.351 E.14634
G3 X120.813 Y132.012 I-7.454 J5.445 E.08931
G1 X141.945 Y138.705 F36000
G1 F13446.283
G1 X141.945 Y141.048 E.08944
G1 X141.198 Y140.812 E.02992
G1 X140.256 Y140.226 E.04237
G3 X138.371 Y138.471 I110.998 J-121.128 E.09833
G2 X134.6 Y137.5 I-2.912 J3.504 E.15358
G2 X131.772 Y139.255 I1.35 J5.332 E.12907
G3 X129.887 Y140.754 I-5.161 J-4.557 E.09243
G1 X129.612 Y140.849 E.01112
G3 X134.909 Y145.034 I-35.207 J50.015 E.25786
G3 X136.485 Y145.041 I.77 J3.878 E.06058
G3 X139.313 Y146.795 I-1.35 J5.332 E.12907
G2 X141.198 Y148.294 I5.161 J-4.557 E.09243
G1 X141.945 Y148.554 E.0302
G1 X141.945 Y150.897 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 3.88
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y149.897 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L24
M991 S0 P23 ;notify layer change


G17
G3 Z4.12 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z3.88
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00922
M73 P58 R8
G1 X122.328 Y123.268 E.20115
G1 X121.921 Y123.634 E.02089
G1 X121.47 Y123.859 E.01924
G1 X120.932 Y123.963 E.02091
G1 X120.385 Y123.913 E.02098
G1 X119.922 Y123.739 E.01887
G3 X118.918 Y122.954 I5.679 J-8.299 E.0487
G3 X114.848 Y126.662 I-39.198 J-38.939 E.21031
G1 X118.015 Y129.012 E.15055
G1 X118.483 Y129.401 E.02323
G1 X119.157 Y130.105 E.03721
G1 X119.749 Y130.932 E.03884
G1 X120.215 Y131.838 E.03891
G3 X120.715 Y133.69 I-7.732 J3.081 E.07341
G1 X120.765 Y134.489 E.03056
G1 X120.698 Y135.504 E.03882
G1 X120.538 Y136.268 E.02982
G3 X130.507 Y142.101 I-22.042 J49.108 E.4418
G3 X142.443 Y154.069 I-32.644 J44.495 E.6479
G1 X142.443 Y104.588 E1.88913
G1 X142.308 Y104.665 E.00594
G1 X141.707 Y104.76 E.02324
G1 X132.884 Y104.76 E.33684
G3 X132.242 Y104.651 I0 J-1.949 E.02497
G1 X131.818 Y104.443 E.01805
G1 X131.229 Y103.84 E.03215
G3 X124.807 Y115.927 I-51.223 J-19.466 E.52393
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.091 Y117.436 E.02015
G1 X126.168 Y117.978 E.02088
G1 X126.11 Y118.451 E.0182
G1 X125.95 Y118.874 E.01727
G1 X125.896 Y118.955 E.00369
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.324 Y118.777 E.00547
G3 X124.449 Y119.826 I-15.052 J-11.662 E.05218
G1 X121.879 Y122.892 E.15272
G1 X121.594 Y123.148 E.01461
G1 X121.187 Y123.333 E.01707
G1 X120.735 Y123.376 E.01735
G1 X120.383 Y123.303 E.01372
G1 X119.957 Y123.061 E.01871
G1 X118.881 Y122.159 E.05363
G3 X113.893 Y126.683 I-38.392 J-37.313 E.25728
G1 X117.666 Y129.482 E.17936
G1 X118.106 Y129.849 E.02186
G1 X118.727 Y130.503 E.03447
G1 X119.266 Y131.263 E.03555
G3 X120.136 Y133.793 I-5.913 J3.447 E.1028
G1 X120.18 Y134.525 E.02799
G1 X120.115 Y135.453 E.03554
G1 X119.933 Y136.284 E.03245
G1 X119.836 Y136.599 E.01259
G3 X130.932 Y143.157 I-21.129 J48.422 E.49334
G3 X142.92 Y155.769 I-32.175 J42.587 E.66732
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y112.03 E1.66914
G3 X143.035 Y103.297 I1918.826 J-3 E.33343
; LINE_WIDTH: 0.607556
G1 F13738.182
G1 X143.04 Y103.044 E.00945
G1 X142.978 Y103.286 E.00935
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.604 Y103.837 E.02542
G1 X142.127 Y104.108 E.02094
G1 X141.707 Y104.174 E.01626
G1 X132.884 Y104.174 E.33684
G1 X132.435 Y104.098 E.01739
G1 X132.138 Y103.952 E.01263
G1 F13365.285
G1 X131.725 Y103.53 E.02254
; LINE_WIDTH: 0.66701
G1 F11334.107
G1 X131.668 Y103.406 E.00566
; LINE_WIDTH: 0.714023
G1 F10885.8
G1 X131.61 Y103.281 E.00608
; LINE_WIDTH: 0.761036
G1 F10446.54
G1 X131.552 Y103.156 E.0065
; LINE_WIDTH: 0.80805
G1 F10016.325
G1 X131.495 Y103.032 E.00692
; LINE_WIDTH: 0.855063
G1 F9595.156
G1 X131.437 Y102.907 E.00734
; LINE_WIDTH: 0.902076
G1 F9075.299
G1 X131.379 Y102.783 E.00777
; LINE_WIDTH: 0.944626
G1 F8651.09
G1 X131.363 Y102.727 E.00345
; LINE_WIDTH: 0.987176
G1 F8264.767
G1 X131.346 Y102.671 E.00361
; LINE_WIDTH: 1.02973
G1 F7911.472
G1 X131.33 Y102.616 E.00377
; LINE_WIDTH: 1.07228
G1 F7587.144
G1 X131.314 Y102.56 E.00393
G1 X131.275 Y102.605 E.00402
; LINE_WIDTH: 1.02973
G1 F7911.472
G1 X131.237 Y102.651 E.00385
; LINE_WIDTH: 0.987176
G1 F8264.767
G1 X131.199 Y102.696 E.00369
; LINE_WIDTH: 0.944626
G1 F8651.09
G1 X131.161 Y102.742 E.00352
; LINE_WIDTH: 0.902076
G1 F9075.299
G1 X131.083 Y102.885 E.00924
; LINE_WIDTH: 0.855063
G1 F9595.156
G1 X131.005 Y103.029 E.00874
; LINE_WIDTH: 0.80805
G1 F10178.188
G1 X130.927 Y103.172 E.00824
; LINE_WIDTH: 0.761036
G1 F10695.143
G1 X130.849 Y103.316 E.00774
; LINE_WIDTH: 0.714023
G1 F11224.902
G1 X130.771 Y103.459 E.00724
; LINE_WIDTH: 0.66701
G1 F11767.467
G1 X130.693 Y103.603 E.00674
; LINE_WIDTH: 0.619996
G1 F13150.293
G1 X130.548 Y103.976 E.01527
G1 F13446.369
G1 X130.331 Y104.532 E.02278
G3 X124.018 Y116.03 I-50.152 J-20.057 E.50205
G1 X125.095 Y116.932 E.05365
G1 X125.394 Y117.286 E.01771
G1 X125.529 Y117.599 E.01299
G1 X125.582 Y117.986 E.01491
G1 X125.512 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.168 Y122.718 E.01264
G1 X120.873 Y122.793 E.01163
G1 X120.577 Y122.751 E.01143
G1 X120.333 Y122.612 E.01068
G1 X118.837 Y121.358 E.07454
G3 X112.93 Y126.698 I-38.419 J-36.565 E.30432
G1 X117.317 Y129.952 E.20856
G1 X117.728 Y130.297 E.02049
G1 X118.298 Y130.902 E.03173
G1 X118.783 Y131.594 E.03227
G3 X119.502 Y133.543 I-5.551 J3.154 E.07965
G1 X119.596 Y134.56 E.03902
G1 X119.531 Y135.403 E.03227
G1 X119.363 Y136.148 E.02917
G1 X119.08 Y136.91 E.03102
G3 X130.579 Y143.624 I-20.519 J48.349 E.50972
G3 X142.647 Y156.415 I-31.78 J42.073 E.67458
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.52 E2.16534
G1 X142.966 Y99.581 E.02486
G1 X141.901 Y99.587 E.04065
G1 X142.473 Y102.676 E.11994
G3 X142.436 Y103.083 I-.766 J.135 E.01581
G1 X142.219 Y103.396 E.01454
G1 X141.947 Y103.551 E.01195
G1 X141.707 Y103.589 E.00928
G1 X132.884 Y103.589 E.33684
G3 X132.458 Y103.462 I0 J-.779 E.0172
G1 X132.215 Y103.209 E.0134
G1 X132.106 Y102.779 E.01692
G1 X132.689 Y99.587 E.1239
G3 X131.483 Y99.49 I.193 J-10.007 E.04624
G1 X131.444 Y99.516 E.00177
G3 X123.222 Y116.127 I-51.438 J-15.12 E.71121
G1 X124.718 Y117.381 E.07453
G1 X124.934 Y117.671 E.01379
G1 X124.997 Y117.982 E.01212
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.007 Y122.161 E.16681
G1 X120.875 Y122.237 E.00482
G1 X120.728 Y122.215 E.00471
G1 X119.699 Y121.359 E.04239
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02783
; LINE_WIDTH: 0.531606
G1 X118.977 Y120.758 E.00202
; LINE_WIDTH: 0.556156
G1 X118.953 Y120.481 E.00946
G1 X117.999 Y121.433 E.04583
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-40.017 J-39.436 E.25281
G1 X116.988 Y130.396 E.19622
G3 X118.995 Y133.832 I-3.258 J4.208 E.12913
G1 X119.044 Y134.594 E.02416
G1 X118.98 Y135.355 E.02419
G1 X118.825 Y136.021 E.02163
G1 X118.548 Y136.753 E.02479
G1 X118.31 Y137.19 E.01575
G3 X130.249 Y144.067 I-19.805 J48.182 E.4375
G3 X142.39 Y157.025 I-31.646 J41.818 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.421 E1.8133
G1 X144.167 Y98.943 E.01512
G2 X142.914 Y99.031 I.1 J10.442 E.03979
G3 X141.237 Y99.035 I-1.021 J-70.497 E.0531
G1 X141.929 Y102.771 E.12029
G1 X141.918 Y102.889 E.00376
G1 X141.776 Y103.025 E.00621
G1 X132.884 Y103.036 E.28153
G3 X132.69 Y102.926 I0 J-.226 E.00738
G1 X132.662 Y102.769 E.00504
G1 X133.354 Y99.036 E.12021
G3 X132.127 Y99.006 I-.304 J-12.751 E.03886
G2 X131.038 Y98.936 I-.885 J5.296 E.03462
G3 X122.648 Y115.977 I-51.095 J-14.571 E.6046
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00556
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.687 Y113.973 Z4.28 F36000
G1 X143.04 Y103.044 Z4.28
G1 Z3.88
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.600676
G1 F13905.077
G1 X143.039 Y102.563 E.01774
; LINE_WIDTH: 0.639046
G1 F13022.773
G1 X143.019 Y102.354 E.00827
; LINE_WIDTH: 0.684762
G1 F12107.446
G1 X142.997 Y102.106 E.0106
; LINE_WIDTH: 0.730479
G1 F11312.34
G1 X142.974 Y101.857 E.01135
; LINE_WIDTH: 0.776195
G1 F10615.229
G1 X142.951 Y101.608 E.01209
; LINE_WIDTH: 0.821911
G1 F9999.048
G1 X142.928 Y101.359 E.01284
; LINE_WIDTH: 0.867628
G1 F9450.478
G1 X142.905 Y101.11 E.01358
; LINE_WIDTH: 0.913344
G1 F8958.969
G1 X142.882 Y100.861 E.01433
; LINE_WIDTH: 0.95906
G1 F8516.057
G1 X142.859 Y100.612 E.01507
; LINE_WIDTH: 1.00478
G1 F8114.877
G1 X142.837 Y100.363 E.01582
; WIPE_START
G1 X142.859 Y100.612 E-.095
G1 X142.882 Y100.861 E-.095
G1 X142.905 Y101.11 E-.095
G1 X142.928 Y101.359 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.336 Y102.144 Z4.28 F36000
G1 X131.314 Y102.56 Z4.28
G1 Z3.88
G1 E.4 F1800
; LINE_WIDTH: 1.07228
G1 F7587.144
G1 X131.324 Y102.521 E.00271
; LINE_WIDTH: 1.0658
G1 F7634.809
G1 X131.392 Y102.263 E.01795
; LINE_WIDTH: 1.02619
G1 F7939.669
G1 X131.459 Y102.005 E.01726
; LINE_WIDTH: 0.986586
G1 F8269.888
G1 X131.527 Y101.747 E.01657
; LINE_WIDTH: 0.946981
G1 F8628.767
G1 X131.595 Y101.489 E.01588
; LINE_WIDTH: 0.907376
G1 F9020.206
G1 X131.605 Y101.453 E.00211
; LINE_WIDTH: 0.902336
G1 F9072.582
G1 X131.683 Y101.139 E.01833
; LINE_WIDTH: 0.862196
G1 F9512.478
G1 X131.762 Y100.824 E.01748
; LINE_WIDTH: 0.822056
G1 F9997.208
G1 X131.84 Y100.51 E.01664
; LINE_WIDTH: 0.781916
G1 F10533.99
G1 X131.919 Y100.196 E.01579
; LINE_WIDTH: 0.741776
G1 F11131.687
G1 X131.92 Y100.19 E.00028
; WIPE_START
G1 X131.919 Y100.196 E-.00235
G1 X131.84 Y100.51 E-.12311
G1 X131.762 Y100.824 E-.12311
G1 X131.683 Y101.139 E-.12311
G1 X131.678 Y101.16 E-.00834
; WIPE_END
G1 E-.02 F1800
G1 X127.48 Y107.534 Z4.28 F36000
G1 X118.953 Y120.481 Z4.28
G1 Z3.88
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556156
G1 F3600
M204 S5000
G1 X119.2 Y120.199 E.01276
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15232
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.13088
G1 X121.722 Y117.19 E-.24912
; WIPE_END
G1 E-.02 F1800
G1 X126.967 Y111.645 Z4.28 F36000
G1 X129.546 Y108.918 Z4.28
G1 Z3.88
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.468 Y110.997 I-35.84 J-17.268 E.08944
G2 X130.83 Y110.115 I-.325 J-4.474 E.09756
G2 X132.715 Y108.359 I-50.328 J-55.913 E.09837
G3 X138.371 Y108.17 I2.944 J3.373 E.23386
G3 X140.256 Y109.926 I-50.312 J55.896 E.09837
G2 X141.945 Y110.832 I2.702 J-3.013 E.0739
G1 X141.945 Y118.447 E.29077
G1 X141.198 Y118.225 E.02976
G3 X139.313 Y116.793 I2.622 J-5.409 E.09098
G2 X136.485 Y114.922 I-4.663 J3.977 E.13115
G2 X132.715 Y115.71 I-1.059 J4.345 E.15203
G2 X130.83 Y117.467 I50.311 J55.895 E.09837
G3 X126.632 Y118.378 I-2.963 J-3.524 E.17062
G3 X126.202 Y119.412 I-2.47 J-.421 E.0431
G1 X123.091 Y123.131 E.18514
G1 X123.289 Y123.251 E.00883
G3 X125.174 Y125.007 I-50.311 J55.895 E.09837
G2 X130.83 Y125.197 I2.944 J-3.373 E.23386
G2 X132.715 Y123.44 I-50.311 J-55.895 E.09837
G3 X138.371 Y123.251 I2.944 J3.373 E.23386
G3 X140.256 Y125.007 I-50.295 J55.878 E.09837
G2 X141.945 Y125.913 I2.702 J-3.013 E.0739
G1 X141.945 Y128.255 E.08944
G1 X120.799 Y131.97 F36000
G1 F13446.283
G2 X119.661 Y129.932 I-8.926 J3.652 E.08932
G3 X123.289 Y130.981 I.69 J4.413 E.14888
G2 X125.174 Y132.737 I52.196 J-54.139 E.09837
G2 X130.83 Y132.548 I2.712 J-3.563 E.23386
G3 X132.715 Y130.792 I52.179 J54.12 E.09837
G3 X138.371 Y130.981 I2.712 J3.563 E.23386
G2 X140.256 Y132.737 I52.212 J-54.156 E.09837
G1 X141.198 Y133.306 E.04204
G1 X141.945 Y133.529 E.02976
G1 X141.945 Y140.994 E.28503
G3 X140.256 Y140.088 I1.013 J-3.918 E.0739
G2 X138.371 Y138.332 I-52.197 J54.14 E.09837
G2 X132.715 Y138.522 I-2.712 J3.563 E.23386
G3 X130.83 Y140.278 I-52.213 J-54.157 E.09837
G1 X129.887 Y140.847 E.04204
G1 X129.69 Y140.906 E.00786
G3 X134.876 Y145.005 I-35.03 J49.652 E.25251
G3 X138.371 Y146.062 I.579 J4.389 E.14367
G2 X140.256 Y147.819 I52.212 J-54.156 E.09837
G1 X141.198 Y148.388 E.04204
G1 X141.945 Y148.61 E.02976
G1 X141.945 Y150.953 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 4.04
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y149.953 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L25
M991 S0 P24 ;notify layer change


G17
G3 Z4.28 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z4.04
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00922
G1 X122.328 Y123.268 E.20115
G1 X121.942 Y123.62 E.01994
G1 X121.523 Y123.84 E.01806
G1 X121.016 Y123.957 E.01987
G1 X120.455 Y123.928 E.02143
G1 X119.954 Y123.756 E.02023
G3 X118.918 Y122.954 I5.098 J-7.659 E.05006
G3 X114.848 Y126.662 I-38.493 J-38.165 E.21031
G1 X118.015 Y129.012 E.15055
G3 X118.932 Y129.849 I-4.611 J5.973 E.04748
G1 X119.566 Y130.648 E.03895
G1 X120.075 Y131.53 E.03885
G1 X120.45 Y132.48 E.039
G1 X120.665 Y133.394 E.03585
G1 X120.765 Y134.489 E.04199
G1 X120.698 Y135.506 E.03893
G1 X120.538 Y136.268 E.02971
G3 X130.504 Y142.099 I-22.118 J49.239 E.44167
G3 X142.443 Y154.07 I-32.633 J44.486 E.64805
G1 X142.443 Y104.592 E1.88903
G1 X142.311 Y104.667 E.00582
G1 X141.709 Y104.762 E.02324
G1 X132.882 Y104.762 E.33703
G3 X132.224 Y104.648 I0 J-1.949 E.02559
G1 X131.628 Y104.305 E.02627
G1 X131.226 Y103.837 E.02355
G3 X124.807 Y115.927 I-50.997 J-19.326 E.524
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.091 Y117.436 E.02015
G1 X126.168 Y117.978 E.02088
G1 X126.11 Y118.451 E.0182
G1 X125.95 Y118.874 E.01727
G1 X125.896 Y118.955 E.00369
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.324 Y118.777 E.00547
G3 X124.449 Y119.826 I-15.05 J-11.66 E.05218
G1 X121.879 Y122.892 E.15272
G1 X121.609 Y123.138 E.01395
G1 X121.316 Y123.292 E.01263
G1 X120.961 Y123.374 E.0139
G1 X120.52 Y123.343 E.01689
G1 X120.196 Y123.222 E.0132
G3 X118.881 Y122.159 I9.148 J-12.665 E.0646
G3 X113.893 Y126.683 I-41.227 J-40.439 E.25725
G1 X117.666 Y129.482 E.17936
G3 X118.523 Y130.268 I-4.175 J5.414 E.04446
G1 X119.1 Y131.003 E.03566
G1 X119.562 Y131.812 E.03557
G1 X119.901 Y132.683 E.03571
G1 X120.092 Y133.515 E.03256
G1 X120.18 Y134.525 E.0387
G1 X120.114 Y135.456 E.03564
G1 X119.932 Y136.284 E.03235
G1 X119.836 Y136.599 E.0126
G3 X130.932 Y143.157 I-21.143 J48.445 E.49332
G3 X142.92 Y155.769 I-32.412 J42.813 E.66729
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y112.031 E1.6691
G3 X143.036 Y103.298 I1716.72 J-3 E.3334
; LINE_WIDTH: 0.606106
G1 F13773.021
G1 X143.041 Y103.046 E.00941
G1 X142.98 Y103.288 E.00932
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.607 Y103.839 E.02541
G1 X142.13 Y104.11 E.02094
G1 X141.709 Y104.176 E.01626
G1 X132.882 Y104.176 E.33703
G1 X132.422 Y104.097 E.01782
G1 X131.982 Y103.838 E.01948
G1 X131.715 Y103.519 E.01587
; LINE_WIDTH: 0.651356
G1 F12078.932
G1 X131.521 Y103.042 E.02071
; LINE_WIDTH: 0.697387
G1 F10389.333
G1 X131.498 Y102.987 E.00259
; LINE_WIDTH: 0.743418
G1 F10201.065
G1 X131.476 Y102.931 E.00277
; LINE_WIDTH: 0.78945
G1 F10014.531
G1 X131.453 Y102.876 E.00295
; LINE_WIDTH: 0.835481
G1 F9829.689
G1 X131.431 Y102.821 E.00313
; LINE_WIDTH: 0.881512
G1 F9295.594
G1 X131.408 Y102.765 E.00331
; LINE_WIDTH: 0.927543
G1 F8816.55
G1 X131.385 Y102.71 E.00349
; LINE_WIDTH: 0.973574
G1 F8384.459
G1 X131.363 Y102.654 E.00367
; LINE_WIDTH: 1.01961
G1 F7992.742
G1 X131.34 Y102.599 E.00385
; LINE_WIDTH: 1.06564
G1 F7635.994
G1 X131.318 Y102.543 E.00403
G1 X131.275 Y102.593 E.00439
; LINE_WIDTH: 1.01961
G1 F7992.742
G1 X131.233 Y102.643 E.00419
; LINE_WIDTH: 0.973574
G1 F8384.459
G1 X131.191 Y102.692 E.00399
; LINE_WIDTH: 0.927543
G1 F8816.55
G1 X131.149 Y102.742 E.0038
; LINE_WIDTH: 0.881512
G1 F9295.594
G1 X131.106 Y102.792 E.0036
; LINE_WIDTH: 0.835481
G1 F9829.689
G1 X131.064 Y102.842 E.00341
; LINE_WIDTH: 0.78945
G1 F10428.898
G1 X131.022 Y102.891 E.00321
; LINE_WIDTH: 0.743418
G1 F10636.37
G1 X130.98 Y102.941 E.00302
; LINE_WIDTH: 0.697387
G1 F10845.884
G1 X130.937 Y102.991 E.00282
; LINE_WIDTH: 0.651356
G1 F12174.992
G1 X130.787 Y103.361 E.01609
G1 F12762.958
G1 X130.685 Y103.614 E.01095
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X130.538 Y103.986 E.01527
G1 X130.078 Y105.145 E.04762
G3 X124.018 Y116.03 I-49.809 J-20.606 E.4767
G1 X125.095 Y116.932 E.05365
G1 X125.394 Y117.286 E.01771
G1 X125.529 Y117.599 E.01299
G1 X125.582 Y117.986 E.01491
G1 X125.512 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.165 Y122.72 E.01279
G1 X120.906 Y122.791 E.01023
G1 X120.577 Y122.751 E.01267
G1 X120.334 Y122.612 E.01069
G1 X118.837 Y121.358 E.07455
G3 X112.93 Y126.698 I-40.369 J-38.722 E.30429
G1 X117.317 Y129.952 E.20856
G1 X118.114 Y130.687 E.04139
G1 X118.634 Y131.357 E.03237
G1 X119.049 Y132.094 E.03228
G1 X119.352 Y132.887 E.03243
G1 X119.519 Y133.636 E.02928
G1 X119.595 Y134.56 E.03542
G1 X119.531 Y135.405 E.03236
G1 X119.363 Y136.148 E.02907
G1 X119.08 Y136.91 E.03104
G3 X130.579 Y143.624 I-20.52 J48.35 E.50972
G3 X142.648 Y156.415 I-32.008 J42.289 E.67455
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.521 E2.16531
G1 X142.966 Y99.583 E.02487
G1 X141.904 Y99.589 E.04056
G1 X142.476 Y102.678 E.11994
G3 X142.438 Y103.085 I-.766 J.135 E.01581
G1 X142.222 Y103.398 E.01454
M73 P58 R7
G1 X141.949 Y103.553 E.01195
G1 X141.709 Y103.591 E.00928
G1 X132.882 Y103.591 E.33703
G1 X132.619 Y103.545 E.01017
G1 X132.368 Y103.397 E.01112
G1 X132.216 Y103.216 E.00906
G1 X132.114 Y102.942 E.01115
G1 X132.135 Y102.57 E.01423
G1 X132.687 Y99.589 E.11576
G3 X131.481 Y99.49 I.195 J-9.822 E.04621
G1 X131.445 Y99.515 E.00167
G3 X123.222 Y116.127 I-51.439 J-15.119 E.71127
G1 X124.718 Y117.381 E.07453
G1 X124.934 Y117.671 E.01379
G1 X124.997 Y117.982 E.01212
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.003 Y122.165 E.16698
G1 X120.855 Y122.24 E.00527
G1 X120.689 Y122.189 E.00551
G1 X119.678 Y121.342 E.04175
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02697
; LINE_WIDTH: 0.531606
G1 X118.977 Y120.758 E.00202
; LINE_WIDTH: 0.556156
G1 X118.953 Y120.481 E.00946
G1 X117.999 Y121.433 E.04583
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-37.694 J-36.796 E.25284
G1 X116.988 Y130.396 E.19623
G1 X117.728 Y131.083 E.03196
G1 X118.194 Y131.692 E.02427
G1 X118.564 Y132.36 E.02421
G1 X118.834 Y133.08 E.02432
G1 X118.978 Y133.75 E.02171
M73 P59 R7
G1 X119.044 Y134.594 E.0268
G1 X118.98 Y135.358 E.02426
G1 X118.825 Y136.02 E.02153
G1 X118.548 Y136.753 E.02482
G1 X118.31 Y137.19 E.01574
G3 X130.248 Y144.067 I-19.584 J47.798 E.43751
G3 X142.39 Y157.025 I-31.644 J41.817 E.56493
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.419 E1.81336
G1 X144.167 Y98.943 E.01506
G2 X142.913 Y99.033 I.102 J10.263 E.03983
G3 X141.24 Y99.037 I-1.011 J-66.313 E.05299
G1 X141.931 Y102.773 E.12029
G1 X141.921 Y102.891 E.00376
G1 X141.779 Y103.027 E.00621
G1 X132.882 Y103.038 E.28169
G1 X132.733 Y102.982 E.00503
G1 X132.659 Y102.85 E.00479
G3 X132.987 Y101.004 I77.28 J12.774 E.05934
G1 X133.351 Y99.038 E.06332
G3 X132.127 Y99.007 I-.303 J-12.457 E.03879
G2 X131.038 Y98.936 I-.885 J5.189 E.03461
G3 X122.648 Y115.977 I-51.094 J-14.57 E.6046
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00556
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.688 Y113.974 Z4.44 F36000
G1 X143.041 Y103.046 Z4.44
G1 Z4.04
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.598156
G1 F13967.224
G1 X143.04 Y102.566 E.01764
; LINE_WIDTH: 0.636776
G1 F13071.842
G1 X143.02 Y102.355 E.0083
; LINE_WIDTH: 0.682494
G1 F12149.826
G1 X142.998 Y102.106 E.01056
; LINE_WIDTH: 0.728211
G1 F11349.308
G1 X142.975 Y101.858 E.01131
; LINE_WIDTH: 0.773929
G1 F10647.756
G1 X142.952 Y101.609 E.01205
; LINE_WIDTH: 0.819646
G1 F10027.888
G1 X142.929 Y101.36 E.0128
; LINE_WIDTH: 0.865364
G1 F9476.222
G1 X142.906 Y101.111 E.01354
; LINE_WIDTH: 0.911081
G1 F8982.088
G1 X142.883 Y100.862 E.01429
; LINE_WIDTH: 0.956799
G1 F8536.933
G1 X142.86 Y100.613 E.01503
; LINE_WIDTH: 1.00252
G1 F8133.819
G1 X142.838 Y100.364 E.01578
; WIPE_START
G1 X142.86 Y100.613 E-.095
G1 X142.883 Y100.862 E-.095
G1 X142.906 Y101.111 E-.095
G1 X142.929 Y101.36 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.336 Y102.134 Z4.44 F36000
G1 X131.318 Y102.543 Z4.44
G1 Z4.04
G1 E.4 F1800
; LINE_WIDTH: 1.06564
G1 F7635.994
G1 X131.349 Y102.425 E.00826
; LINE_WIDTH: 1.04592
G1 F7784.851
G1 X131.434 Y102.101 E.02209
; LINE_WIDTH: 0.997356
G1 F8177.401
G1 X131.519 Y101.776 E.02103
; LINE_WIDTH: 0.948796
G1 F8611.64
G1 X131.604 Y101.452 E.01997
; LINE_WIDTH: 0.900236
G1 F9094.584
G1 X131.682 Y101.138 E.01829
; LINE_WIDTH: 0.860091
G1 F9536.728
G1 X131.761 Y100.824 E.01744
; LINE_WIDTH: 0.819946
G1 F10024.059
G1 X131.839 Y100.509 E.01659
; LINE_WIDTH: 0.779801
G1 F10563.877
G1 X131.918 Y100.195 E.01574
; LINE_WIDTH: 0.739656
G1 F11165.145
G1 X131.919 Y100.19 E.00025
; WIPE_START
G1 X131.918 Y100.195 E-.00208
G1 X131.839 Y100.509 E-.12311
G1 X131.761 Y100.824 E-.12311
G1 X131.682 Y101.138 E-.12311
G1 X131.677 Y101.16 E-.00858
; WIPE_END
G1 E-.02 F1800
G1 X127.479 Y107.534 Z4.44 F36000
G1 X118.953 Y120.481 Z4.44
G1 Z4.04
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556156
G1 F3600
M204 S5000
G1 X119.2 Y120.199 E.01276
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15232
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.13088
G1 X121.722 Y117.19 E-.24912
; WIPE_END
G1 E-.02 F1800
G1 X126.974 Y111.652 Z4.44 F36000
G1 X129.538 Y108.948 Z4.44
G1 Z4.04
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.457 Y111.026 I-37.945 J-18.421 E.08944
G2 X130.83 Y110.259 I-.078 J-4.294 E.09661
G2 X132.715 Y108.494 I-19.824 J-23.067 E.09862
G3 X137.428 Y107.502 I3.208 J3.547 E.1927
G3 X139.313 Y108.871 I-2.076 J4.84 E.0897
G2 X141.198 Y110.471 I6.294 J-5.507 E.09474
G1 X141.945 Y110.783 E.03091
G1 X141.945 Y118.509 E.29496
G1 X141.198 Y118.324 E.02938
G3 X139.313 Y116.955 I2.076 J-4.84 E.0897
G2 X136.485 Y114.961 I-5.228 J4.414 E.13354
G2 X132.715 Y115.567 I-1.254 J4.232 E.15077
G2 X130.83 Y117.332 I19.83 J23.072 E.09862
G3 X126.618 Y118.448 I-3.177 J-3.486 E.17294
G3 X126.202 Y119.412 I-2.675 J-.58 E.04033
G1 X123.167 Y123.04 E.18059
G1 X123.289 Y123.108 E.00533
G3 X125.174 Y124.872 I-19.824 J23.067 E.09862
G2 X129.887 Y125.864 I3.208 J-3.547 E.1927
G2 X131.772 Y124.495 I-2.076 J-4.84 E.0897
G3 X134.6 Y122.502 I5.228 J4.414 E.13354
G3 X138.371 Y123.108 I1.254 J4.232 E.15077
G3 X140.256 Y124.872 I-19.821 J23.064 E.09862
G2 X141.945 Y125.864 I2.969 J-3.123 E.07546
G1 X141.945 Y128.207 E.08944
G1 X120.768 Y131.911 F36000
G1 F13446.283
G2 X119.61 Y129.889 I-8.135 J3.315 E.08924
G3 X123.289 Y131.116 I.493 J4.651 E.15269
G2 X125.174 Y132.881 I21.715 J-21.308 E.09862
G2 X129.887 Y133.093 I2.531 J-3.773 E.18939
G2 X131.772 Y131.493 I-4.409 J-7.107 E.09474
G3 X133.658 Y130.124 I3.961 J3.471 E.0897
G3 X138.371 Y131.116 I1.505 J4.539 E.1927
G2 X140.256 Y132.881 I21.717 J-21.311 E.09862
G1 X141.198 Y133.405 E.04118
G1 X141.945 Y133.59 E.02938
G1 X141.945 Y140.945 E.28083
G3 X140.256 Y139.954 I1.28 J-4.115 E.07546
G2 X138.371 Y138.189 I-21.707 J21.299 E.09862
G2 X133.658 Y137.977 I-2.531 J3.773 E.18939
G2 X131.772 Y139.576 I4.409 J7.107 E.09474
G3 X129.788 Y140.97 I-4.456 J-4.234 E.0932
G3 X134.834 Y144.967 I-36.05 J50.695 E.24588
G3 X138.371 Y146.197 I.376 J4.619 E.14716
G2 X140.256 Y147.962 I21.712 J-21.305 E.09862
G1 X141.198 Y148.486 E.04118
G1 X141.945 Y148.671 E.02938
G1 X141.945 Y151.014 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 4.2
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y150.014 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L26
M991 S0 P25 ;notify layer change


G17
G3 Z4.44 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z4.2
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00922
G1 X122.328 Y123.268 E.20115
G1 X121.942 Y123.62 E.01993
G1 X121.522 Y123.84 E.01811
G1 X121.016 Y123.957 E.01981
G1 X120.455 Y123.928 E.02145
G1 X119.958 Y123.758 E.02006
G3 X118.918 Y122.954 I5.038 J-7.594 E.05022
G3 X114.848 Y126.662 I-39.173 J-38.912 E.21031
G1 X118.015 Y129.012 E.15054
G3 X118.933 Y129.849 I-4.605 J5.967 E.0475
G1 X119.565 Y130.646 E.03882
G1 X120.076 Y131.531 E.03903
G1 X120.45 Y132.48 E.03895
G3 X120.718 Y133.724 I-10.105 J2.821 E.04862
G1 X120.765 Y134.489 E.02923
G1 X120.698 Y135.504 E.03884
G1 X120.538 Y136.268 E.0298
G3 X130.504 Y142.099 I-22.375 J49.673 E.44165
G3 X142.443 Y154.069 I-32.633 J44.487 E.64804
G1 X142.443 Y104.595 E1.88888
G1 X142.313 Y104.669 E.00571
G1 X141.712 Y104.764 E.02324
G1 X132.879 Y104.764 E.33722
G1 X132.579 Y104.741 E.0115
G1 X131.979 Y104.544 E.02409
G1 X131.227 Y103.847 E.03917
G3 X124.807 Y115.927 I-51.217 J-19.47 E.52366
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.091 Y117.436 E.02015
G1 X126.168 Y117.978 E.02088
G1 X126.11 Y118.451 E.0182
G1 X125.95 Y118.874 E.01727
G1 X125.896 Y118.955 E.00369
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.324 Y118.777 E.00547
G3 X124.449 Y119.826 I-15.045 J-11.656 E.05218
G1 X121.879 Y122.892 E.15272
G1 X121.609 Y123.138 E.01394
G1 X121.315 Y123.292 E.01267
G1 X120.961 Y123.374 E.01386
G1 X120.52 Y123.343 E.0169
G1 X120.196 Y123.222 E.0132
G3 X118.881 Y122.159 I9.152 J-12.671 E.0646
G3 X113.893 Y126.683 I-38.403 J-37.326 E.25728
G1 X117.666 Y129.482 E.17935
G3 X118.523 Y130.268 I-4.172 J5.41 E.04448
G1 X119.099 Y131 E.03554
G1 X119.563 Y131.814 E.03575
G1 X119.901 Y132.684 E.03566
G3 X120.135 Y133.79 I-10.082 J2.708 E.04319
G1 X120.18 Y134.524 E.02808
G1 X120.115 Y135.453 E.03556
G1 X119.931 Y136.289 E.03268
G1 X119.836 Y136.599 E.01237
G3 X130.932 Y143.157 I-21.095 J48.364 E.49334
G3 X142.92 Y155.769 I-32.373 J42.775 E.66729
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y114.032 E1.59271
G3 X143.037 Y103.3 I1906.259 J-4 E.40973
; LINE_WIDTH: 0.604626
G1 F13808.766
G1 X143.042 Y103.048 E.00938
G1 X142.982 Y103.29 E.00929
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.609 Y103.841 E.0254
G1 X142.132 Y104.112 E.02094
G1 X141.712 Y104.178 E.01626
G1 X132.879 Y104.178 E.33722
G1 X132.669 Y104.162 E.00805
G1 X132.224 Y104.011 E.01794
G1 X132.011 Y103.809 E.0112
G1 F12682.281
G1 X131.721 Y103.535 E.01527
; LINE_WIDTH: 0.667326
G1 F11324.974
G1 X131.663 Y103.409 E.0057
; LINE_WIDTH: 0.714656
G1 F10874.172
G1 X131.605 Y103.284 E.00613
; LINE_WIDTH: 0.761986
G1 F10432.51
G1 X131.547 Y103.158 E.00655
; LINE_WIDTH: 0.809316
G1 F10000.003
G1 X131.489 Y103.033 E.00698
; LINE_WIDTH: 0.856646
G1 F9576.681
G1 X131.431 Y102.908 E.0074
; LINE_WIDTH: 0.903976
G1 F9055.472
G1 X131.373 Y102.782 E.00783
; LINE_WIDTH: 0.945196
G1 F8645.676
G1 X131.358 Y102.728 E.00337
; LINE_WIDTH: 0.986416
G1 F8271.365
G1 X131.342 Y102.673 E.00352
; LINE_WIDTH: 1.02764
G1 F7928.119
G1 X131.326 Y102.619 E.00367
; LINE_WIDTH: 1.06886
G1 F7612.226
G1 X131.31 Y102.564 E.00382
G1 X131.273 Y102.609 E.00391
; LINE_WIDTH: 1.02764
G1 F7928.119
G1 X131.236 Y102.653 E.00375
; LINE_WIDTH: 0.986416
G1 F8271.365
G1 X131.199 Y102.698 E.0036
; LINE_WIDTH: 0.945196
G1 F8645.676
G1 X131.162 Y102.742 E.00344
; LINE_WIDTH: 0.903976
G1 F9055.472
G1 X131.083 Y102.886 E.00926
; LINE_WIDTH: 0.856646
G1 F9576.681
G1 X131.005 Y103.029 E.00876
; LINE_WIDTH: 0.809316
G1 F10161.553
G1 X130.927 Y103.173 E.00826
; LINE_WIDTH: 0.761986
G1 F10678.35
G1 X130.849 Y103.316 E.00775
; LINE_WIDTH: 0.714656
G1 F11207.965
G1 X130.771 Y103.46 E.00725
; LINE_WIDTH: 0.667326
G1 F11750.397
G1 X130.693 Y103.603 E.00674
; LINE_WIDTH: 0.619996
G1 F13132.248
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.401 Y104.348 E.01527
G1 X130.251 Y104.726 E.01552
G3 X124.018 Y116.03 I-50.744 J-20.612 E.49399
G1 X125.094 Y116.932 E.05365
G1 X125.394 Y117.286 E.01771
G1 X125.529 Y117.599 E.01299
G1 X125.582 Y117.986 E.01491
G1 X125.512 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.165 Y122.72 E.0128
G1 X120.906 Y122.791 E.01021
G1 X120.577 Y122.75 E.01269
G1 X120.334 Y122.612 E.01068
G1 X118.837 Y121.358 E.07455
G3 X112.93 Y126.698 I-38.423 J-36.569 E.30431
G1 X117.317 Y129.952 E.20856
G1 X118.114 Y130.687 E.0414
G1 X118.632 Y131.355 E.03226
G1 X119.049 Y132.096 E.03247
G1 X119.352 Y132.888 E.03237
G3 X119.551 Y133.826 I-12.564 J3.145 E.03662
G1 X119.596 Y134.56 E.02808
G1 X119.531 Y135.403 E.03227
G1 X119.361 Y136.154 E.0294
G1 X119.08 Y136.91 E.0308
G3 X130.579 Y143.624 I-20.743 J48.732 E.5097
G3 X142.648 Y156.415 I-31.962 J42.246 E.67455
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.522 E2.16528
G1 X142.966 Y99.585 E.02487
G1 X141.906 Y99.591 E.04046
G1 X142.479 Y102.68 E.11994
G3 X142.441 Y103.087 I-.767 J.135 E.01581
G1 X142.224 Y103.4 E.01454
G1 X141.952 Y103.555 E.01195
G1 X141.712 Y103.593 E.00928
G1 X132.879 Y103.593 E.33722
G1 X132.505 Y103.497 E.01473
G1 X132.21 Y103.212 E.0157
G1 X132.101 Y102.781 E.01695
G1 X132.684 Y99.591 E.12382
G3 X131.48 Y99.49 I.196 J-9.629 E.04619
G1 X131.445 Y99.513 E.00157
G3 X123.222 Y116.127 I-51.361 J-15.079 E.71134
G1 X124.718 Y117.381 E.07453
G1 X124.934 Y117.671 E.01379
G1 X124.997 Y117.982 E.01212
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.003 Y122.165 E.16698
G1 X120.855 Y122.24 E.00527
G1 X120.689 Y122.189 E.00551
G1 X119.678 Y121.342 E.04175
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02697
; LINE_WIDTH: 0.531606
G1 X118.977 Y120.758 E.00201
; LINE_WIDTH: 0.556196
G1 X118.953 Y120.481 E.00947
G1 X117.999 Y121.433 E.04584
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-40.046 J-39.468 E.25281
G1 X116.988 Y130.396 E.19622
G1 X117.728 Y131.083 E.03197
G1 X118.192 Y131.689 E.02418
G1 X118.565 Y132.362 E.02435
G1 X118.834 Y133.08 E.02427
G3 X118.999 Y133.86 I-29.206 J6.58 E.02523
G1 X119.044 Y134.598 E.02344
G1 X118.98 Y135.355 E.02405
G1 X118.824 Y136.026 E.02182
G1 X118.548 Y136.753 E.0246
G1 X118.31 Y137.19 E.01575
G3 X130.249 Y144.067 I-20.138 J48.759 E.43746
G3 X142.39 Y157.025 I-31.645 J41.818 E.56493
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.418 E1.81341
G1 X144.167 Y98.944 E.01501
G2 X142.912 Y99.035 I.104 J10.096 E.03986
G3 X141.242 Y99.039 I-1 J-61.702 E.05288
G1 X141.934 Y102.775 E.12029
G1 X141.923 Y102.893 E.00376
G1 X141.781 Y103.029 E.00621
G1 X132.879 Y103.04 E.28185
G3 X132.685 Y102.929 I0 J-.226 E.0074
G1 X132.657 Y102.773 E.00503
G1 X133.349 Y99.04 E.12021
G3 X132.126 Y99.009 I-.301 J-12.194 E.03872
G2 X131.038 Y98.936 I-.885 J5.088 E.03461
G3 X122.648 Y115.977 I-50.883 J-14.467 E.60463
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00556
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.688 Y113.974 Z4.6 F36000
G1 X143.042 Y103.048 Z4.6
G1 Z4.2
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.595636
G1 F14029.933
G1 X143.041 Y102.568 E.01755
; LINE_WIDTH: 0.634516
G1 F13121.065
G1 X143.022 Y102.356 E.00832
; LINE_WIDTH: 0.680231
G1 F12192.385
G1 X142.999 Y102.107 E.01053
; LINE_WIDTH: 0.725946
G1 F11386.476
G1 X142.976 Y101.858 E.01127
; LINE_WIDTH: 0.771661
G1 F10680.502
G1 X142.953 Y101.609 E.01202
; LINE_WIDTH: 0.817376
G1 F10056.959
G1 X142.93 Y101.361 E.01276
; LINE_WIDTH: 0.863091
G1 F9502.206
G1 X142.907 Y101.112 E.01351
; LINE_WIDTH: 0.908806
G1 F9005.455
G1 X142.884 Y100.863 E.01425
; LINE_WIDTH: 0.954521
G1 F8558.063
G1 X142.862 Y100.614 E.015
; LINE_WIDTH: 1.00024
G1 F8153.018
G1 X142.839 Y100.365 E.01574
; WIPE_START
G1 X142.862 Y100.614 E-.095
G1 X142.884 Y100.863 E-.095
G1 X142.907 Y101.112 E-.095
G1 X142.93 Y101.361 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.338 Y102.147 Z4.6 F36000
G1 X131.31 Y102.564 Z4.6
G1 Z4.2
G1 E.4 F1800
; LINE_WIDTH: 1.06886
G1 F7612.226
G1 X131.32 Y102.526 E.0027
; LINE_WIDTH: 1.0624
G1 F7660.059
G1 X131.389 Y102.266 E.01798
; LINE_WIDTH: 1.0226
G1 F7968.551
G1 X131.457 Y102.007 E.01728
; LINE_WIDTH: 0.982796
G1 F8302.934
G1 X131.525 Y101.747 E.01658
; LINE_WIDTH: 0.942996
G1 F8666.608
G1 X131.593 Y101.488 E.01589
; LINE_WIDTH: 0.903196
G1 F9063.601
G1 X131.603 Y101.452 E.0021
; LINE_WIDTH: 0.898156
G1 F9116.483
G1 X131.681 Y101.138 E.01825
; LINE_WIDTH: 0.858001
G1 F9560.927
G1 X131.76 Y100.823 E.0174
; LINE_WIDTH: 0.817846
G1 F10050.925
G1 X131.838 Y100.509 E.01655
; LINE_WIDTH: 0.777691
G1 F10593.862
G1 X131.917 Y100.195 E.0157
; LINE_WIDTH: 0.737536
G1 F11198.806
G1 X131.918 Y100.19 E.00022
; WIPE_START
G1 X131.917 Y100.195 E-.00182
G1 X131.838 Y100.509 E-.12314
G1 X131.76 Y100.823 E-.12314
G1 X131.681 Y101.138 E-.12314
G1 X131.676 Y101.16 E-.00877
; WIPE_END
G1 E-.02 F1800
G1 X127.478 Y107.535 Z4.6 F36000
G1 X118.953 Y120.481 Z4.6
G1 Z4.2
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556196
G1 F3600
M204 S5000
G1 X119.2 Y120.198 E.01278
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.687 E.15231
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.687 E-.1308
G1 X121.722 Y117.19 E-.2492
; WIPE_END
G1 E-.02 F1800
G1 X126.977 Y111.655 Z4.6 F36000
G1 X129.515 Y108.982 Z4.6
G1 Z4.2
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.432 Y111.059 I-34.732 J-16.794 E.08944
G2 X130.83 Y110.408 I.315 J-3.582 E.09685
G2 X132.715 Y108.628 I-11.997 J-14.59 E.09908
G3 X137.899 Y107.593 I3.367 J3.369 E.21396
M73 P60 R7
G3 X140.256 Y109.657 I-6.66 J9.982 E.11994
G2 X141.945 Y110.739 I3.256 J-3.226 E.07723
G1 X141.945 Y118.575 E.29919
G3 X140.256 Y117.949 I.38 J-3.616 E.06953
G3 X138.371 Y116.168 I11.996 J-14.589 E.09908
G2 X133.186 Y115.134 I-3.367 J3.369 E.21396
G2 X130.83 Y117.198 I6.66 J9.982 E.11994
G3 X126.598 Y118.523 I-3.405 J-3.453 E.17583
G3 X126.202 Y119.412 I-2.605 J-.629 E.03735
G1 X123.254 Y122.937 E.17546
G1 X123.289 Y122.958 E.00159
G3 X125.174 Y124.739 I-11.995 J14.588 E.09908
G2 X130.359 Y125.773 I3.367 J-3.369 E.21396
G2 X132.715 Y123.709 I-6.659 J-9.981 E.11994
G3 X137.899 Y122.675 I3.367 J3.369 E.21396
G3 X140.256 Y124.739 I-6.66 J9.982 E.11994
G2 X141.945 Y125.82 I3.256 J-3.226 E.07723
G1 X141.945 Y128.163 E.08944
G1 X120.745 Y131.847 F36000
G1 F13446.283
G2 X119.569 Y129.835 I-6.988 J2.735 E.08935
G3 X123.289 Y131.25 I.269 J4.892 E.1565
G2 X125.646 Y133.313 I9.017 J-7.918 E.11994
G2 X130.83 Y132.279 I1.817 J-4.403 E.21396
G3 X133.186 Y130.215 I9.016 J7.917 E.11994
G3 X138.371 Y131.25 I1.817 J4.403 E.21396
G2 X140.727 Y133.313 I9.017 J-7.918 E.11994
G2 X141.945 Y133.657 I1.453 J-2.823 E.04866
G1 X141.945 Y140.902 E.2766
G3 X140.256 Y139.82 I1.566 J-4.307 E.07723
G2 X137.899 Y137.756 I-9.017 J7.918 E.11994
G2 X132.715 Y138.79 I-1.817 J4.403 E.21396
G3 X130.359 Y140.854 I-9.016 J-7.918 E.11994
G1 X129.897 Y141.046 E.01908
G3 X134.781 Y144.92 I-36.634 J51.199 E.23808
G3 X138.371 Y146.331 I.164 J4.856 E.15143
G2 X140.727 Y148.395 I9.017 J-7.918 E.11994
G2 X141.945 Y148.738 I1.453 J-2.823 E.04866
G1 X141.945 Y151.081 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 4.36
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y150.081 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L27
M991 S0 P26 ;notify layer change


G17
G3 Z4.6 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z4.36
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00922
G1 X122.328 Y123.268 E.20115
G1 X121.941 Y123.62 E.01997
G1 X121.522 Y123.84 E.01807
G1 X121.016 Y123.957 E.01984
G1 X120.455 Y123.928 E.02144
G1 X119.96 Y123.759 E.01999
G3 X118.918 Y122.954 I5.013 J-7.568 E.05028
G3 X114.848 Y126.662 I-39.397 J-39.158 E.2103
G1 X118.015 Y129.012 E.15054
G3 X118.932 Y129.849 I-4.609 J5.972 E.04748
G1 X119.565 Y130.647 E.03888
G1 X120.075 Y131.529 E.03888
G1 X120.45 Y132.479 E.03902
G3 X120.765 Y134.489 I-7.446 J2.195 E.07788
G1 X120.698 Y135.506 E.03893
G1 X120.538 Y136.268 E.02971
G3 X130.504 Y142.099 I-22.174 J49.329 E.44166
G3 X142.443 Y154.069 I-32.713 J44.569 E.64803
G1 X142.443 Y104.599 E1.88873
G1 X142.316 Y104.671 E.0056
G1 X141.714 Y104.766 E.02324
G1 X132.877 Y104.766 E.33741
G1 X132.579 Y104.743 E.01141
G1 X132.049 Y104.582 E.02112
G1 X131.605 Y104.295 E.02019
G1 X131.225 Y103.851 E.02231
G3 X124.807 Y115.927 I-51.679 J-19.72 E.52347
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.091 Y117.436 E.02015
G1 X126.168 Y117.978 E.02088
G1 X126.11 Y118.451 E.0182
G1 X125.95 Y118.874 E.01727
G1 X125.896 Y118.955 E.00369
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.324 Y118.777 E.00547
G3 X124.449 Y119.826 I-15.052 J-11.662 E.05218
G1 X121.879 Y122.892 E.15272
G1 X121.609 Y123.138 E.01397
G1 X121.315 Y123.292 E.01264
G1 X120.961 Y123.374 E.01388
G1 X120.52 Y123.343 E.01689
G1 X120.196 Y123.222 E.0132
G3 X118.881 Y122.159 I9.15 J-12.668 E.0646
G3 X113.893 Y126.683 I-41.606 J-40.856 E.25725
G1 X117.666 Y129.482 E.17935
G3 X118.523 Y130.268 I-4.174 J5.413 E.04446
G1 X119.099 Y131.001 E.03559
G1 X119.561 Y131.811 E.0356
G1 X119.901 Y132.683 E.03574
G3 X120.135 Y133.791 I-9.586 J2.605 E.04324
G1 X120.18 Y134.524 E.02807
G1 X120.114 Y135.456 E.03564
G1 X119.931 Y136.288 E.03254
G1 X119.836 Y136.599 E.01242
G3 X130.932 Y143.156 I-21.128 J48.42 E.49331
G3 X142.92 Y155.769 I-32.175 J42.587 E.66734
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y114.033 E1.59267
G3 X143.037 Y103.302 I1740.439 J-4 E.40971
; LINE_WIDTH: 0.603166
G1 F13844.207
G1 X143.043 Y103.05 E.00935
G1 X142.984 Y103.292 E.00926
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.612 Y103.843 E.0254
G1 X142.135 Y104.114 E.02094
G1 X141.714 Y104.181 E.01626
G1 X132.877 Y104.181 E.33741
G1 X132.668 Y104.164 E.00799
G1 X132.271 Y104.039 E.01588
G1 X131.987 Y103.851 E.01302
G1 F12757.874
G1 X131.703 Y103.511 E.0169
; LINE_WIDTH: 0.66749
G1 F11255.422
G1 X131.647 Y103.39 E.00551
; LINE_WIDTH: 0.714983
G1 F10820.523
G1 X131.592 Y103.268 E.00593
; LINE_WIDTH: 0.762476
G1 F10394.163
G1 X131.537 Y103.147 E.00634
; LINE_WIDTH: 0.80997
G1 F9976.373
G1 X131.481 Y103.025 E.00676
; LINE_WIDTH: 0.857463
G1 F9567.18
G1 X131.426 Y102.903 E.00717
; LINE_WIDTH: 0.904956
G1 F9045.279
G1 X131.37 Y102.782 E.00758
; LINE_WIDTH: 0.945511
G1 F8642.688
G1 X131.355 Y102.728 E.00333
; LINE_WIDTH: 0.986066
G1 F8274.406
G1 X131.339 Y102.674 E.00347
; LINE_WIDTH: 1.02662
G1 F7936.229
G1 X131.324 Y102.62 E.00362
; LINE_WIDTH: 1.06718
G1 F7624.608
G1 X131.309 Y102.567 E.00377
G1 X131.272 Y102.61 E.00385
; LINE_WIDTH: 1.02662
G1 F7936.229
G1 X131.235 Y102.654 E.0037
; LINE_WIDTH: 0.986066
G1 F8274.406
G1 X131.199 Y102.698 E.00355
; LINE_WIDTH: 0.945511
G1 F8642.688
G1 X131.162 Y102.742 E.0034
; LINE_WIDTH: 0.904956
G1 F9045.279
G1 X131.084 Y102.886 E.00928
; LINE_WIDTH: 0.857463
G1 F9567.18
G1 X131.006 Y103.029 E.00877
; LINE_WIDTH: 0.80997
G1 F10152.994
G1 X130.927 Y103.173 E.00826
; LINE_WIDTH: 0.762476
G1 F10669.614
G1 X130.849 Y103.316 E.00776
; LINE_WIDTH: 0.714983
G1 F11199.082
G1 X130.771 Y103.46 E.00725
; LINE_WIDTH: 0.66749
G1 F11741.326
G1 X130.693 Y103.603 E.00675
; LINE_WIDTH: 0.619996
G1 F13122.659
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.209 Y104.835 E.01998
G3 X124.018 Y116.03 I-50.128 J-20.413 E.48956
G1 X125.094 Y116.932 E.05365
G1 X125.394 Y117.286 E.01771
G1 X125.529 Y117.599 E.01299
G1 X125.582 Y117.986 E.01491
G1 X125.512 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.165 Y122.72 E.0128
G1 X120.906 Y122.791 E.01022
G1 X120.577 Y122.75 E.01269
G1 X120.333 Y122.612 E.01068
G1 X118.837 Y121.358 E.07454
G3 X112.93 Y126.698 I-40.624 J-39.004 E.30428
G1 X117.317 Y129.952 E.20855
G1 X118.114 Y130.687 E.04139
G1 X118.633 Y131.356 E.03231
G1 X119.048 Y132.093 E.03232
G1 X119.352 Y132.887 E.03245
G3 X119.551 Y133.826 I-11.694 J2.963 E.03667
G1 X119.596 Y134.56 E.02807
G1 X119.531 Y135.405 E.03236
G1 X119.362 Y136.153 E.02926
G1 X119.08 Y136.91 E.03086
G3 X130.578 Y143.623 I-20.597 J48.482 E.50969
G3 X142.648 Y156.415 I-31.779 J42.074 E.6746
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.522 E2.16525
G1 X142.966 Y99.587 E.02487
G1 X141.909 Y99.593 E.04036
G1 X142.481 Y102.682 E.11994
G3 X142.443 Y103.089 I-.767 J.135 E.01581
G1 X142.227 Y103.403 E.01454
G1 X141.954 Y103.557 E.01195
G1 X141.714 Y103.595 E.00928
G1 X132.877 Y103.595 E.33741
G1 X132.531 Y103.514 E.01354
G3 X132.207 Y103.213 I.345 J-.698 E.01714
G1 X132.099 Y102.782 E.01697
G1 X132.111 Y102.675 E.00412
G1 X132.682 Y99.593 E.11967
G3 X131.478 Y99.49 I.197 J-9.445 E.04617
G1 X131.308 Y99.983 E.0199
G3 X123.222 Y116.127 I-51.397 J-15.645 E.69264
G1 X124.718 Y117.381 E.07453
G1 X124.934 Y117.671 E.01379
G1 X124.997 Y117.982 E.01212
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.003 Y122.165 E.16698
G1 X120.855 Y122.24 E.00527
G1 X120.689 Y122.189 E.00551
G1 X119.678 Y121.342 E.04174
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02697
; LINE_WIDTH: 0.531596
G1 X118.977 Y120.758 E.00201
; LINE_WIDTH: 0.556156
G1 X118.953 Y120.481 E.00946
G1 X117.999 Y121.433 E.04583
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.723 J-39.102 E.25281
G1 X116.988 Y130.396 E.19622
G1 X117.728 Y131.083 E.03196
G1 X118.193 Y131.69 E.02422
G1 X118.564 Y132.36 E.02424
G3 X118.98 Y135.357 I-5.038 J2.227 E.09707
G1 X118.824 Y136.025 E.0217
G1 X118.548 Y136.753 E.02466
G1 X118.31 Y137.19 E.01574
G3 X130.248 Y144.067 I-19.758 J48.1 E.43748
G3 X142.39 Y157.025 I-31.645 J41.818 E.56494
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.416 E1.81346
G1 X144.167 Y98.944 E.01495
G2 X142.911 Y99.037 I.105 J9.924 E.0399
G3 X141.245 Y99.041 I-.992 J-58.08 E.05277
G1 X141.937 Y102.777 E.12029
G1 X141.926 Y102.896 E.00376
G1 X141.784 Y103.031 E.00621
G1 X141.714 Y103.042 E.00223
G1 X132.877 Y103.042 E.2798
G1 X132.729 Y102.987 E.00497
G1 X132.651 Y102.806 E.00625
G1 X132.655 Y102.775 E.00099
G1 X133.346 Y99.042 E.12021
G3 X132.126 Y99.01 I-.3 J-11.94 E.03865
G2 X131.038 Y98.936 I-.886 J4.991 E.03461
G3 X122.648 Y115.977 I-50.883 J-14.466 E.60463
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00556
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.689 Y113.975 Z4.76 F36000
G1 X143.043 Y103.05 Z4.76
G1 Z4.36
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.593096
G1 F14093.708
G1 X143.042 Y102.57 E.01745
; LINE_WIDTH: 0.632236
G1 F13171.101
G1 X143.023 Y102.357 E.00834
; LINE_WIDTH: 0.677951
G1 F12235.576
G1 X143 Y102.108 E.01049
; LINE_WIDTH: 0.723666
G1 F11424.137
G1 X142.977 Y101.859 E.01123
; LINE_WIDTH: 0.769381
G1 F10713.631
G1 X142.954 Y101.61 E.01198
; LINE_WIDTH: 0.815096
G1 F10086.327
G1 X142.931 Y101.361 E.01272
; LINE_WIDTH: 0.860811
G1 F9528.42
G1 X142.908 Y101.112 E.01347
; LINE_WIDTH: 0.906526
G1 F9028.997
G1 X142.886 Y100.864 E.01421
; LINE_WIDTH: 0.952241
G1 F8579.32
G1 X142.863 Y100.615 E.01496
; LINE_WIDTH: 0.997956
G1 F8172.309
G1 X142.84 Y100.366 E.0157
; WIPE_START
G1 X142.863 Y100.615 E-.095
G1 X142.886 Y100.864 E-.095
G1 X142.908 Y101.112 E-.095
G1 X142.931 Y101.361 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.34 Y102.149 Z4.76 F36000
G1 X131.309 Y102.567 Z4.76
G1 Z4.36
G1 E.4 F1800
; LINE_WIDTH: 1.06718
G1 F7624.608
G1 X131.319 Y102.528 E.00269
; LINE_WIDTH: 1.06072
G1 F7672.597
G1 X131.387 Y102.268 E.01799
; LINE_WIDTH: 1.02081
G1 F7983.01
G1 X131.455 Y102.008 E.01729
; LINE_WIDTH: 0.980896
G1 F8319.6
G1 X131.524 Y101.748 E.01659
; LINE_WIDTH: 0.940986
G1 F8685.822
G1 X131.592 Y101.488 E.01589
; LINE_WIDTH: 0.901076
G1 F9085.771
G1 X131.602 Y101.452 E.00209
; LINE_WIDTH: 0.896056
G1 F9138.7
G1 X131.68 Y101.138 E.0182
; LINE_WIDTH: 0.855911
G1 F9585.248
G1 X131.759 Y100.823 E.01735
; LINE_WIDTH: 0.815766
G1 F10077.679
G1 X131.837 Y100.509 E.0165
; LINE_WIDTH: 0.775621
G1 F10623.445
G1 X131.916 Y100.195 E.01565
; LINE_WIDTH: 0.735476
G1 F11231.709
G1 X131.917 Y100.19 E.00021
; WIPE_START
G1 X131.916 Y100.195 E-.00175
G1 X131.837 Y100.509 E-.1231
G1 X131.759 Y100.823 E-.12309
G1 X131.68 Y101.138 E-.1231
G1 X131.674 Y101.16 E-.00897
; WIPE_END
G1 E-.02 F1800
G1 X127.477 Y107.535 Z4.76 F36000
G1 X118.953 Y120.481 Z4.76
G1 Z4.36
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556156
G1 F3600
M204 S5000
G1 X119.2 Y120.198 E.01277
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15231
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.13088
G1 X121.722 Y117.19 E-.24912
; WIPE_END
G1 E-.02 F1800
G1 X126.985 Y111.663 Z4.76 F36000
G1 X129.502 Y109.02 Z4.76
G1 Z4.36
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.418 Y111.096 I-75.509 J-38.12 E.08944
G2 X131.772 Y109.751 I.436 J-3.768 E.14382
G3 X133.658 Y107.976 I8.681 J7.33 E.09908
G3 X137.899 Y107.457 I2.634 J3.943 E.16924
G3 X139.313 Y108.534 I-2.411 J4.632 E.06819
G2 X141.198 Y110.309 I8.681 J-7.33 E.09908
G1 X141.945 Y110.7 E.03219
G1 X141.945 Y118.648 E.30346
G3 X139.313 Y117.291 I.285 J-3.784 E.11615
G2 X137.428 Y115.516 I-8.681 J7.33 E.09908
G2 X133.186 Y114.998 I-2.634 J3.943 E.16924
G2 X131.772 Y116.075 I2.411 J4.632 E.06819
G3 X129.887 Y117.85 I-8.681 J-7.33 E.09908
G3 X126.577 Y118.607 I-2.628 J-3.878 E.13266
G3 X126.202 Y119.412 I-2.6 J-.721 E.03406
G1 X123.337 Y122.837 E.1705
G3 X125.174 Y124.604 I-7.566 J9.706 E.09752
G2 X129.887 Y126.081 I3.673 J-3.463 E.19703
G2 X131.772 Y124.832 I-1.48 J-4.282 E.08725
G3 X133.658 Y123.057 I8.681 J7.33 E.09908
G3 X137.899 Y122.539 I2.634 J3.943 E.16924
G3 X139.313 Y123.615 I-2.411 J4.632 E.06819
G2 X141.198 Y125.391 I8.681 J-7.33 E.09908
G1 X141.945 Y125.781 E.03219
G1 X141.945 Y128.124 E.08944
G1 X120.717 Y131.771 F36000
G1 F13446.283
G2 X119.519 Y129.771 I-7.151 J2.924 E.08936
G3 X123.289 Y131.384 I.033 J5.135 E.16106
G2 X125.174 Y133.188 I9.382 J-7.918 E.09981
G2 X128.945 Y133.424 I2.128 J-3.751 E.14929
G2 X130.83 Y132.145 I-2.01 J-4.99 E.08764
G3 X132.715 Y130.341 I9.382 J7.917 E.09981
G3 X136.485 Y130.105 I2.128 J3.751 E.14929
G3 X138.371 Y131.384 I-2.009 J4.99 E.08764
G2 X140.256 Y133.188 I9.382 J-7.917 E.09981
G2 X141.945 Y133.73 I1.855 J-2.878 E.06853
G1 X141.945 Y140.863 E.27233
G3 X140.256 Y139.686 I1.871 J-4.488 E.07922
G2 X138.371 Y137.882 I-9.382 J7.917 E.09981
G2 X134.6 Y137.645 I-2.128 J3.751 E.14929
G2 X132.715 Y138.924 I2.01 J4.99 E.08764
G3 X130.83 Y140.728 I-9.382 J-7.918 E.09981
G3 X130.003 Y141.12 I-1.646 J-2.408 E.03506
G3 X134.729 Y144.862 I-39.965 J55.318 E.2302
G3 X138.371 Y146.465 I-.068 J5.093 E.15608
G2 X140.256 Y148.269 I9.382 J-7.917 E.09981
G2 X141.945 Y148.811 I1.855 J-2.878 E.06853
G1 X141.945 Y151.154 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 4.52
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y150.154 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L28
M991 S0 P27 ;notify layer change


G17
G3 Z4.76 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z4.52
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00922
G1 X122.328 Y123.268 E.20115
G1 X121.942 Y123.62 E.01993
G1 X121.522 Y123.84 E.01812
G1 X121.016 Y123.957 E.01982
G1 X120.455 Y123.928 E.02144
G1 X119.954 Y123.756 E.02022
G3 X118.918 Y122.954 I5.099 J-7.661 E.05006
G3 X114.848 Y126.662 I-38.496 J-38.169 E.21031
G1 X118.015 Y129.012 E.15054
G3 X118.932 Y129.849 I-4.608 J5.971 E.04749
G1 X119.565 Y130.646 E.03884
G1 X120.076 Y131.531 E.03903
G1 X120.45 Y132.48 E.03893
G1 X120.681 Y133.47 E.03882
G3 X120.698 Y135.506 I-8.012 J1.083 E.07792
G1 X120.538 Y136.268 E.02974
G3 X130.504 Y142.099 I-21.912 J48.884 E.44168
G3 X142.443 Y154.069 I-32.642 J44.497 E.64802
G1 X142.443 Y104.602 E1.88859
G1 X142.318 Y104.673 E.00549
G1 X141.717 Y104.768 E.02324
G1 X132.874 Y104.768 E.3376
G1 X132.575 Y104.745 E.01144
G1 X132.046 Y104.584 E.02113
G1 X131.602 Y104.296 E.02022
G1 X131.224 Y103.854 E.02219
G3 X124.807 Y115.927 I-51.676 J-19.722 E.52335
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.091 Y117.436 E.02015
G1 X126.168 Y117.978 E.02088
G1 X126.11 Y118.451 E.0182
G1 X125.95 Y118.874 E.01727
G1 X125.896 Y118.955 E.0037
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.324 Y118.777 E.00547
G3 X124.449 Y119.826 I-15.06 J-11.668 E.05218
G1 X121.879 Y122.892 E.15272
G1 X121.609 Y123.138 E.01394
G1 X121.315 Y123.292 E.01268
G1 X120.961 Y123.374 E.01387
G1 X120.52 Y123.343 E.01689
G1 X120.196 Y123.222 E.0132
G3 X118.881 Y122.159 I9.148 J-12.666 E.0646
G3 X113.893 Y126.683 I-41.223 J-40.434 E.25725
G1 X117.666 Y129.482 E.17935
G3 X118.523 Y130.268 I-4.173 J5.412 E.04447
G1 X119.099 Y131.001 E.03556
G1 X119.563 Y131.814 E.03574
G1 X119.901 Y132.684 E.03564
G3 X120.114 Y135.455 I-6.618 J1.903 E.10686
G1 X119.932 Y136.285 E.03243
G1 X119.835 Y136.599 E.01255
G3 X130.933 Y143.157 I-21.278 J48.674 E.49334
G3 X142.92 Y155.769 I-32.175 J42.587 E.6673
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y116.034 E1.51628
G3 X143.038 Y103.303 I1899.501 J-5 E.48604
; LINE_WIDTH: 0.601696
G1 F13880.078
G1 X143.045 Y103.051 E.00932
G1 X142.986 Y103.294 E.00923
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.615 Y103.846 E.02539
G1 X142.137 Y104.116 E.02094
G1 X141.717 Y104.183 E.01626
G1 X132.874 Y104.183 E.3376
G1 X132.665 Y104.166 E.008
G1 X132.268 Y104.041 E.01589
G1 X131.984 Y103.852 E.01304
G1 F12871.755
G1 X131.702 Y103.516 E.01677
; LINE_WIDTH: 0.665146
G1 F11374.316
G1 X131.505 Y103.039 E.02118
; LINE_WIDTH: 0.714893
G1 F9735.688
G1 X131.481 Y102.979 E.0029
; LINE_WIDTH: 0.764639
G1 F9537.073
G1 X131.456 Y102.918 E.00311
; LINE_WIDTH: 0.814385
G1 F9340.517
G1 X131.432 Y102.858 E.00332
; LINE_WIDTH: 0.864131
G1 F9145.996
G1 X131.408 Y102.797 E.00353
; LINE_WIDTH: 0.913877
G1 F8953.532
G1 X131.383 Y102.736 E.00374
; LINE_WIDTH: 0.963624
G1 F8474.235
G1 X131.359 Y102.676 E.00396
; LINE_WIDTH: 1.01337
G1 F8043.645
G1 X131.335 Y102.615 E.00417
; LINE_WIDTH: 1.06312
G1 F7654.698
G1 X131.31 Y102.555 E.00438
G1 X131.265 Y102.609 E.00476
; LINE_WIDTH: 1.01337
G1 F8043.645
G1 X131.219 Y102.663 E.00452
; LINE_WIDTH: 0.963624
G1 F8474.235
G1 X131.174 Y102.717 E.00429
; LINE_WIDTH: 0.913877
G1 F8953.532
G1 X131.128 Y102.772 E.00406
; LINE_WIDTH: 0.864131
G1 F9490.296
G1 X131.082 Y102.826 E.00384
; LINE_WIDTH: 0.814385
G1 F9705.494
G1 X131.037 Y102.88 E.00361
; LINE_WIDTH: 0.764639
G1 F9923.105
G1 X130.991 Y102.934 E.00338
; LINE_WIDTH: 0.714893
G1 F10143.104
G1 X130.945 Y102.989 E.00314
; LINE_WIDTH: 0.665146
G1 F11429.695
G1 X130.793 Y103.359 E.01645
G1 F12335.906
G1 X130.691 Y103.607 E.01104
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X130.545 Y103.979 E.01527
G1 X130.209 Y104.835 E.0351
G3 X124.018 Y116.03 I-50.13 J-20.415 E.48956
G1 X125.095 Y116.932 E.05365
G1 X125.394 Y117.286 E.01771
G1 X125.529 Y117.599 E.01299
G1 X125.582 Y117.986 E.01491
G1 X125.512 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.165 Y122.72 E.0128
G1 X120.906 Y122.791 E.01022
G1 X120.577 Y122.751 E.01268
G1 X120.334 Y122.612 E.01069
G1 X118.837 Y121.358 E.07455
G3 X112.93 Y126.698 I-40.367 J-38.72 E.30429
G1 X117.317 Y129.952 E.20856
G1 X118.114 Y130.687 E.04139
G1 X118.632 Y131.355 E.03227
G1 X119.05 Y132.096 E.03246
G1 X119.352 Y132.887 E.03235
G1 X119.535 Y133.712 E.03227
G1 X119.588 Y134.43 E.02745
G1 X119.531 Y135.405 E.03729
G1 X119.362 Y136.149 E.02916
G1 X119.08 Y136.91 E.03098
G3 X130.579 Y143.624 I-20.758 J48.757 E.50972
G3 X142.647 Y156.415 I-31.78 J42.073 E.67456
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.523 E2.16522
G1 X142.966 Y99.589 E.02487
G1 X141.912 Y99.595 E.04028
M73 P61 R7
G1 X142.484 Y102.684 E.11994
G3 X142.446 Y103.092 I-.766 J.135 E.01581
G1 X142.229 Y103.405 E.01454
G1 X141.957 Y103.559 E.01195
G1 X141.717 Y103.597 E.00928
G1 X132.874 Y103.597 E.3376
G1 X132.528 Y103.516 E.01355
G3 X132.205 Y103.216 I.346 J-.698 E.01707
G1 X132.106 Y102.942 E.01112
G1 X132.109 Y102.677 E.01015
G1 X132.679 Y99.595 E.11967
G3 X131.476 Y99.49 I.198 J-9.273 E.04615
G1 X131.297 Y100.005 E.02081
G3 X123.222 Y116.127 I-51.263 J-15.592 E.69171
G1 X124.718 Y117.381 E.07453
G1 X124.934 Y117.671 E.01379
G1 X124.997 Y117.982 E.01212
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.003 Y122.165 E.16698
G1 X120.855 Y122.24 E.00527
G1 X120.689 Y122.189 E.00551
G1 X119.678 Y121.342 E.04175
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02697
; LINE_WIDTH: 0.531626
G1 X118.977 Y120.758 E.00202
; LINE_WIDTH: 0.556176
G1 X118.953 Y120.481 E.00946
G1 X117.999 Y121.433 E.04584
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-37.694 J-36.796 E.25284
G1 X116.988 Y130.396 E.19622
G1 X117.728 Y131.083 E.03196
G1 X118.192 Y131.69 E.02419
G1 X118.565 Y132.362 E.02435
G1 X118.834 Y133.08 E.02425
G1 X118.994 Y133.827 E.0242
G1 X119.036 Y134.463 E.02019
G1 X118.98 Y135.357 E.02835
G1 X118.825 Y136.022 E.02161
G1 X118.548 Y136.752 E.02474
G1 X118.31 Y137.19 E.01576
G3 X130.249 Y144.067 I-19.584 J47.798 E.43753
G3 X142.39 Y157.025 I-31.645 J41.817 E.56491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.414 E1.81352
G1 X144.167 Y98.944 E.01489
G2 X142.911 Y99.039 I.106 J9.763 E.03993
G3 X141.247 Y99.044 I-.983 J-54.383 E.05267
G1 X141.939 Y102.779 E.12029
G1 X141.928 Y102.898 E.00376
G1 X141.786 Y103.033 E.00621
G1 X132.874 Y103.044 E.28216
G1 X132.727 Y102.989 E.00497
G1 X132.651 Y102.854 E.0049
G3 X132.979 Y101.011 I36.747 J5.587 E.0593
G1 X133.344 Y99.044 E.06332
G3 X132.126 Y99.012 I-.299 J-11.688 E.03858
G2 X131.039 Y98.936 I-.885 J4.891 E.03457
G3 X122.648 Y115.977 I-51.405 J-14.726 E.60457
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00556
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.689 Y113.975 Z4.92 F36000
G1 X143.045 Y103.051 Z4.92
G1 Z4.52
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.590576
G1 F14157.56
G1 X143.044 Y102.573 E.01736
; LINE_WIDTH: 0.629976
G1 F13221.074
G1 X143.024 Y102.358 E.00837
; LINE_WIDTH: 0.675691
G1 F12278.693
G1 X143.001 Y102.109 E.01045
; LINE_WIDTH: 0.721406
G1 F11461.715
G1 X142.978 Y101.86 E.0112
; LINE_WIDTH: 0.767121
G1 F10746.673
G1 X142.955 Y101.611 E.01194
; LINE_WIDTH: 0.812836
G1 F10115.608
G1 X142.932 Y101.362 E.01269
; LINE_WIDTH: 0.858551
G1 F9554.546
G1 X142.91 Y101.113 E.01343
; LINE_WIDTH: 0.904266
G1 F9052.454
G1 X142.887 Y100.864 E.01418
; LINE_WIDTH: 0.949981
G1 F8600.495
G1 X142.864 Y100.615 E.01492
; LINE_WIDTH: 0.995696
G1 F8191.521
G1 X142.841 Y100.366 E.01567
; WIPE_START
G1 X142.864 Y100.615 E-.095
G1 X142.887 Y100.864 E-.095
G1 X142.91 Y101.113 E-.095
G1 X142.932 Y101.362 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.34 Y102.141 Z4.92 F36000
G1 X131.31 Y102.555 Z4.92
G1 Z4.52
G1 E.4 F1800
; LINE_WIDTH: 1.06312
G1 F7654.698
G1 X131.372 Y102.326 E.01587
; LINE_WIDTH: 1.0259
G1 F7942.031
G1 X131.433 Y102.097 E.0153
; LINE_WIDTH: 0.988676
G1 F8251.777
G1 X131.513 Y101.783 E.02018
; LINE_WIDTH: 0.944501
G1 F8652.278
G1 X131.594 Y101.469 E.01925
; LINE_WIDTH: 0.900326
G1 F9093.639
G1 X131.675 Y101.155 E.01832
; LINE_WIDTH: 0.856151
G1 F9582.45
G1 X131.755 Y100.84 E.01738
; LINE_WIDTH: 0.811976
G1 F10126.794
G1 X131.764 Y100.805 E.00182
; LINE_WIDTH: 0.807496
G1 F10185.474
G1 X131.839 Y100.499 E.0159
; LINE_WIDTH: 0.772276
G1 F10671.601
G1 X131.914 Y100.192 E.01518
; WIPE_START
G1 X131.839 Y100.499 E-.11991
G1 X131.764 Y100.805 E-.11991
G1 X131.755 Y100.84 E-.01366
G1 X131.675 Y101.155 E-.12328
G1 X131.673 Y101.163 E-.00324
; WIPE_END
G1 E-.02 F1800
G1 X127.475 Y107.537 Z4.92 F36000
G1 X118.953 Y120.481 Z4.92
G1 Z4.52
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556176
G1 F3600
M204 S5000
G1 X119.2 Y120.199 E.01276
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15232
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.13088
G1 X121.722 Y117.19 E-.24912
; WIPE_END
G1 E-.02 F1800
G1 X126.991 Y111.668 Z4.92 F36000
G1 X129.479 Y109.06 Z4.92
G1 Z4.52
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.395 Y111.136 I-65.004 J-32.632 E.08943
G2 X131.772 Y109.932 I.686 J-3.415 E.14382
G3 X133.658 Y108.051 I10.356 J8.496 E.10183
G3 X137.899 Y107.312 I2.864 J3.898 E.17035
G3 X139.313 Y108.353 I-1.903 J4.065 E.06747
G2 X141.198 Y110.234 I10.357 J-8.497 E.10183
G1 X141.945 Y110.666 E.03295
G1 X141.945 Y118.728 E.3078
G3 X139.313 Y117.472 I.026 J-3.441 E.11498
G2 X137.428 Y115.592 I-10.357 J8.497 E.10183
G2 X133.186 Y114.853 I-2.864 J3.898 E.17035
G2 X131.772 Y115.894 I1.902 J4.065 E.06746
G3 X129.887 Y117.775 I-10.357 J-8.496 E.10183
G3 X126.554 Y118.699 I-2.87 J-3.876 E.13509
G3 X126.202 Y119.412 I-2.852 J-.964 E.03041
G1 X123.424 Y122.733 E.16531
G3 X125.174 Y124.468 I-5.988 J7.788 E.09436
G2 X129.887 Y126.2 I3.934 J-3.427 E.19999
G2 X131.772 Y125.013 I-1.066 J-3.784 E.08624
G3 X133.658 Y123.132 I10.357 J8.497 E.10183
G3 X137.899 Y122.394 I2.864 J3.898 E.17035
G3 X139.313 Y123.435 I-1.902 J4.065 E.06747
G2 X141.198 Y125.315 I10.357 J-8.497 E.10183
G1 X141.945 Y125.747 E.03295
G1 X141.945 Y128.09 E.08944
G1 X120.686 Y131.696 F36000
G1 F13446.283
G2 X119.469 Y129.706 I-8.8 J4.017 E.08926
G1 X119.519 Y129.702 E.0019
G3 X123.289 Y131.52 I-.258 J5.354 E.16418
G2 X125.174 Y133.355 I7.637 J-5.958 E.10077
G2 X128.945 Y133.401 I1.932 J-3.826 E.14906
G2 X130.83 Y132.009 I-2.361 J-5.17 E.09011
G3 X132.715 Y130.174 I7.636 J5.957 E.10077
G3 X136.485 Y130.128 I1.932 J3.826 E.14906
G3 X138.371 Y131.52 I-2.361 J5.17 E.09011
G2 X140.256 Y133.355 I7.637 J-5.957 E.10077
G2 X141.945 Y133.809 I1.656 J-2.793 E.06763
G1 X141.945 Y140.829 E.268
G3 X140.256 Y139.55 I2.191 J-4.65 E.08149
G2 X138.371 Y137.714 I-7.637 J5.957 E.10077
G2 X134.6 Y137.668 I-1.932 J3.826 E.14906
G2 X132.715 Y139.06 I2.361 J5.17 E.09011
G3 X130.83 Y140.896 I-7.637 J-5.958 E.10077
G3 X130.124 Y141.208 I-1.269 J-1.916 E.0296
G3 X134.63 Y144.787 I-46.831 J63.581 E.21973
G3 X138.371 Y146.601 I-.281 J5.343 E.16304
G2 X140.256 Y148.437 I7.637 J-5.958 E.10077
G2 X141.945 Y148.891 I1.656 J-2.793 E.06764
G1 X141.945 Y151.233 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 4.68
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y150.233 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L29
M991 S0 P28 ;notify layer change


G17
G3 Z4.92 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z4.68
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00923
G1 X123.613 Y121.735 E.12479
G1 X122.328 Y123.268 E.07636
G1 X121.943 Y123.619 E.0199
G1 X121.523 Y123.839 E.01809
G1 X121.016 Y123.957 E.01986
G1 X120.455 Y123.928 E.02144
G1 X120.071 Y123.81 E.01535
G1 X119.581 Y123.51 E.02194
G1 X118.918 Y122.954 E.03305
G3 X114.848 Y126.662 I-38.671 J-38.348 E.2103
G1 X118.015 Y129.012 E.15056
G3 X118.933 Y129.85 I-4.61 J5.971 E.0475
G1 X119.567 Y130.649 E.03893
G1 X120.076 Y131.532 E.03894
G1 X120.45 Y132.48 E.0389
G3 X120.718 Y133.723 I-10.228 J2.85 E.04857
G3 X120.698 Y135.504 I-8.742 J.795 E.06814
G1 X120.538 Y136.268 E.02979
G3 X130.506 Y142.1 I-22.359 J49.649 E.44173
G3 X142.443 Y154.069 I-32.643 J44.495 E.64794
G1 X142.443 Y104.606 E1.88846
G1 X142.321 Y104.675 E.00538
G1 X141.719 Y104.77 E.02324
G1 X132.872 Y104.77 E.3378
G1 X132.572 Y104.747 E.01146
G1 X132.043 Y104.585 E.02115
G1 X131.598 Y104.297 E.02023
G1 X131.223 Y103.857 E.02208
G3 X124.807 Y115.927 I-51.672 J-19.724 E.52322
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.107 Y117.493 E.02237
G1 X126.168 Y117.988 E.01905
G1 X126.11 Y118.45 E.01778
G1 X125.95 Y118.874 E.01728
G1 X125.896 Y118.954 E.0037
G1 X125.361 Y118.695 F36000
G1 F13446.369
G1 X125.264 Y118.854 E.00713
G1 X121.879 Y122.892 E.20115
G1 X121.609 Y123.138 E.01393
G1 X121.316 Y123.292 E.01266
G1 X120.961 Y123.374 E.01389
G1 X120.52 Y123.343 E.0169
G1 X120.3 Y123.271 E.00883
G1 X119.862 Y122.981 E.02005
G1 X118.881 Y122.159 E.04888
G3 X113.893 Y126.683 I-39.141 J-38.139 E.25727
G1 X117.666 Y129.482 E.17937
G3 X118.524 Y130.269 I-4.176 J5.414 E.04448
G1 X119.1 Y131.003 E.03564
G1 X119.563 Y131.814 E.03565
G1 X119.901 Y132.683 E.03561
G3 X120.135 Y133.792 I-10.132 J2.719 E.04327
G1 X120.18 Y134.525 E.02803
G1 X120.115 Y135.454 E.03557
G1 X119.932 Y136.285 E.0325
G1 X119.836 Y136.599 E.01253
G3 X130.932 Y143.157 I-21.097 J48.366 E.49335
G3 X142.92 Y155.769 I-32.175 J42.586 E.6673
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y116.035 E1.51624
G3 X143.039 Y103.305 I1756.836 J-5 E.48601
; LINE_WIDTH: 0.600226
G1 F13916.132
G1 X143.046 Y103.053 E.00929
G1 X142.987 Y103.296 E.0092
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.617 Y103.848 E.02538
G1 X142.14 Y104.118 E.02094
G1 X141.719 Y104.185 E.01626
G1 X132.872 Y104.185 E.3378
G1 X132.662 Y104.169 E.00802
G1 X132.265 Y104.043 E.0159
G1 X131.981 Y103.853 E.01305
G1 F12744.869
G1 X131.696 Y103.513 E.01694
; LINE_WIDTH: 0.66779
G1 F11240.525
G1 X131.641 Y103.391 E.00553
; LINE_WIDTH: 0.715583
G1 F10804.919
G1 X131.586 Y103.269 E.00595
; LINE_WIDTH: 0.763376
G1 F10377.935
G1 X131.53 Y103.147 E.00636
; LINE_WIDTH: 0.81117
G1 F9959.558
G1 X131.475 Y103.025 E.00678
; LINE_WIDTH: 0.858963
G1 F9549.777
G1 X131.42 Y102.903 E.0072
; LINE_WIDTH: 0.906756
G1 F9026.617
G1 X131.365 Y102.781 E.00762
; LINE_WIDTH: 0.946006
G1 F8637.995
G1 X131.35 Y102.729 E.00324
; LINE_WIDTH: 0.985256
G1 F8281.454
G1 X131.335 Y102.676 E.00338
; LINE_WIDTH: 1.02451
G1 F7953.179
G1 X131.32 Y102.624 E.00352
; LINE_WIDTH: 1.06376
G1 F7649.939
G1 X131.305 Y102.571 E.00366
G1 X131.27 Y102.614 E.00374
; LINE_WIDTH: 1.02451
G1 F7953.179
G1 X131.234 Y102.657 E.0036
; LINE_WIDTH: 0.985256
G1 F8281.454
G1 X131.198 Y102.7 E.00345
; LINE_WIDTH: 0.946006
G1 F8637.995
G1 X131.163 Y102.742 E.00331
; LINE_WIDTH: 0.906756
G1 F9026.617
G1 X131.085 Y102.886 E.0093
; LINE_WIDTH: 0.858963
G1 F9549.777
G1 X131.006 Y103.029 E.00879
; LINE_WIDTH: 0.81117
G1 F10137.31
G1 X130.928 Y103.173 E.00828
; LINE_WIDTH: 0.763376
G1 F10653.61
G1 X130.85 Y103.316 E.00777
; LINE_WIDTH: 0.715583
G1 F11182.762
G1 X130.771 Y103.46 E.00726
; LINE_WIDTH: 0.66779
G1 F11724.693
G1 X130.693 Y103.603 E.00675
; LINE_WIDTH: 0.619996
G1 F13105.074
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.209 Y104.835 E.01998
G3 X124.018 Y116.03 I-50.129 J-20.414 E.48957
G1 X125.095 Y116.932 E.05365
G1 X125.39 Y117.279 E.01739
G1 X125.546 Y117.667 E.01597
G1 X125.577 Y118.094 E.01636
G1 X125.474 Y118.51 E.01636
G1 X125.408 Y118.618 E.00483
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.165 Y122.72 E.01278
G1 X120.907 Y122.791 E.01023
G1 X120.577 Y122.751 E.01267
G1 X120.238 Y122.532 E.01538
G1 X118.837 Y121.358 E.0698
G3 X112.93 Y126.698 I-38.903 J-37.1 E.30431
G1 X117.317 Y129.953 E.20857
G1 X118.115 Y130.688 E.04141
G1 X118.634 Y131.357 E.03235
G1 X119.05 Y132.096 E.03237
G1 X119.352 Y132.887 E.03233
G3 X119.551 Y133.827 I-12.639 J3.161 E.0367
G1 X119.596 Y134.56 E.02803
G1 X119.531 Y135.403 E.03229
G1 X119.362 Y136.15 E.02922
G1 X119.08 Y136.91 E.03096
G3 X130.579 Y143.624 I-20.846 J48.908 E.50971
G3 X142.648 Y156.415 I-31.779 J42.072 E.67456
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.524 E2.16519
G1 X142.966 Y99.591 E.02488
G1 X141.914 Y99.597 E.04018
G1 X142.486 Y102.686 E.11994
G3 X142.448 Y103.094 I-.767 J.135 E.01581
G1 X142.232 Y103.407 E.01454
G1 X141.959 Y103.561 E.01195
G1 X141.719 Y103.599 E.00928
G1 X132.872 Y103.599 E.3378
G1 X132.526 Y103.518 E.01357
G3 X132.201 Y103.216 I.346 J-.697 E.01717
G1 X132.094 Y102.784 E.017
G1 X132.106 Y102.679 E.00403
G1 X132.677 Y99.596 E.11968
G3 X131.474 Y99.49 I.199 J-9.109 E.04612
G1 X131.308 Y99.983 E.01986
G3 X123.222 Y116.127 I-51.394 J-15.644 E.69264
G1 X124.718 Y117.381 E.07453
G1 X124.937 Y117.679 E.01409
G1 X124.997 Y117.982 E.0118
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00189
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.003 Y122.165 E.16698
G1 X120.855 Y122.24 E.00527
G1 X120.689 Y122.189 E.00551
G1 X119.678 Y121.341 E.04176
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02696
; LINE_WIDTH: 0.531626
G1 X118.977 Y120.758 E.00202
; LINE_WIDTH: 0.556176
G1 X118.953 Y120.481 E.00946
G1 X118 Y121.433 E.04582
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.72 J-39.098 E.25283
G1 X116.988 Y130.397 E.19623
G1 X117.728 Y131.083 E.03197
G1 X118.194 Y131.692 E.02426
G1 X118.565 Y132.362 E.02427
G1 X118.834 Y133.079 E.02424
G3 X118.999 Y133.861 I-29.648 J6.672 E.02529
G1 X119.044 Y134.594 E.02325
G1 X118.98 Y135.356 E.02421
G1 X118.825 Y136.022 E.02166
G1 X118.548 Y136.753 E.02474
G1 X118.31 Y137.19 E.01575
G3 X130.249 Y144.067 I-20.138 J48.759 E.43748
G3 X142.39 Y157.025 I-31.645 J41.817 E.56491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.412 E1.81359
G1 X144.167 Y98.944 E.01481
G1 X142.91 Y99.041 E.03994
G3 X141.25 Y99.046 I-.974 J-50.727 E.05256
G1 X141.942 Y102.782 E.12029
G1 X141.931 Y102.9 E.00376
G1 X141.789 Y103.035 E.00621
G1 X141.719 Y103.046 E.00223
G1 X132.872 Y103.046 E.28012
G1 X132.724 Y102.991 E.00498
G1 X132.646 Y102.81 E.00626
G1 X132.65 Y102.779 E.00097
G1 X133.341 Y99.046 E.12021
G3 X132.126 Y99.013 I-.297 J-11.451 E.03851
G2 X131.038 Y98.936 I-.886 J4.807 E.0346
G3 X122.648 Y115.977 I-51.094 J-14.57 E.6046
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00555
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
M73 P62 R7
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.69 Y113.976 Z5.08 F36000
G1 X143.046 Y103.053 Z5.08
G1 Z4.68
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.588036
G1 F14222.504
G1 X143.045 Y102.575 E.01726
; LINE_WIDTH: 0.627696
G1 F13271.876
G1 X143.025 Y102.359 E.00839
; LINE_WIDTH: 0.673411
G1 F12322.498
G1 X143.002 Y102.11 E.01042
; LINE_WIDTH: 0.719126
G1 F11499.878
G1 X142.979 Y101.861 E.01116
; LINE_WIDTH: 0.764841
G1 F10780.215
G1 X142.956 Y101.612 E.01191
; LINE_WIDTH: 0.810556
G1 F10145.321
G1 X142.934 Y101.363 E.01265
; LINE_WIDTH: 0.856271
G1 F9581.051
G1 X142.911 Y101.114 E.0134
; LINE_WIDTH: 0.901986
G1 F9076.241
G1 X142.888 Y100.865 E.01414
; LINE_WIDTH: 0.947701
G1 F8621.964
G1 X142.865 Y100.616 E.01489
; LINE_WIDTH: 0.993416
G1 F8210.995
G1 X142.842 Y100.367 E.01563
; WIPE_START
G1 X142.865 Y100.616 E-.095
G1 X142.888 Y100.865 E-.095
G1 X142.911 Y101.114 E-.095
G1 X142.934 Y101.363 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.342 Y102.152 Z5.08 F36000
G1 X131.305 Y102.571 Z5.08
G1 Z4.68
G1 E.4 F1800
; LINE_WIDTH: 1.06376
G1 F7649.939
G1 X131.315 Y102.532 E.00268
; LINE_WIDTH: 1.0573
G1 F7698.248
G1 X131.384 Y102.271 E.01802
; LINE_WIDTH: 1.0172
G1 F8012.331
G1 X131.453 Y102.01 E.01731
; LINE_WIDTH: 0.977096
G1 F8353.134
G1 X131.522 Y101.748 E.01661
; LINE_WIDTH: 0.936996
G1 F8724.216
G1 X131.59 Y101.487 E.0159
; LINE_WIDTH: 0.896896
G1 F9129.8
G1 X131.6 Y101.451 E.00208
; LINE_WIDTH: 0.891856
G1 F9183.46
G1 X131.678 Y101.137 E.01811
; LINE_WIDTH: 0.851711
G1 F9634.501
G1 X131.757 Y100.823 E.01726
; LINE_WIDTH: 0.811566
G1 F10132.137
G1 X131.835 Y100.508 E.01641
; LINE_WIDTH: 0.771421
G1 F10683.979
G1 X131.914 Y100.194 E.01557
; LINE_WIDTH: 0.731276
G1 F11299.395
G1 X131.915 Y100.191 E.00016
; WIPE_START
G1 X131.914 Y100.194 E-.00135
G1 X131.835 Y100.508 E-.12311
G1 X131.757 Y100.823 E-.12311
G1 X131.678 Y101.137 E-.12311
G1 X131.672 Y101.161 E-.00933
; WIPE_END
G1 E-.02 F1800
G1 X127.475 Y107.536 Z5.08 F36000
G1 X118.953 Y120.481 Z5.08
G1 Z4.68
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556176
G1 F3600
M204 S5000
G1 X119.2 Y120.199 E.01276
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15232
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.13087
G1 X121.722 Y117.19 E-.24913
; WIPE_END
G1 E-.02 F1800
G1 X126.998 Y111.675 Z5.08 F36000
G1 X129.457 Y109.105 Z5.08
G1 Z4.68
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.371 Y111.18 I-65.835 J-33.151 E.08943
G2 X131.772 Y110.126 I.871 J-3.204 E.14352
G3 X133.658 Y108.123 I12.478 J9.86 E.10513
G3 X137.899 Y107.157 I3.116 J3.886 E.17188
G3 X139.313 Y108.159 I-1.46 J3.558 E.06674
G2 X141.198 Y110.162 I12.478 J-9.86 E.10513
G1 X141.945 Y110.637 E.03381
G1 X141.945 Y118.814 E.31222
G3 X139.313 Y117.666 I-.206 J-3.119 E.11392
G2 X137.428 Y115.664 I-12.48 J9.862 E.10513
G2 X133.186 Y114.698 I-3.116 J3.886 E.17188
G2 X131.772 Y115.7 I1.459 J3.557 E.06674
G3 X129.887 Y117.702 I-12.478 J-9.86 E.10513
G3 X126.522 Y118.802 I-3.131 J-3.882 E.1382
G3 X126.202 Y119.412 I-2.407 J-.871 E.02636
G1 X123.521 Y122.617 E.15953
G3 X125.174 Y124.328 I-5.049 J6.529 E.09118
G2 X129.887 Y126.328 I4.23 J-3.418 E.20351
G2 X131.772 Y125.207 I-.705 J-3.331 E.08525
G3 X133.658 Y123.205 I12.479 J9.861 E.10513
G3 X137.899 Y122.238 I3.116 J3.886 E.17188
G3 X139.313 Y123.241 I-1.46 J3.558 E.06674
G2 X141.198 Y125.243 I12.479 J-9.861 E.10513
G1 X141.945 Y125.718 E.03381
G1 X141.945 Y128.061 E.08944
G1 X120.656 Y131.604 F36000
G1 F13446.283
G2 X119.411 Y129.63 I-9.261 J4.461 E.08931
G1 X119.519 Y129.626 E.00414
G3 X123.289 Y131.66 I-.565 J5.56 E.16785
G2 X125.174 Y133.535 I6.403 J-4.55 E.10202
G2 X128.002 Y133.738 I1.678 J-3.592 E.11069
G2 X130.83 Y131.869 I-1.886 J-5.927 E.13106
G3 X132.715 Y129.993 I6.402 J4.549 E.10202
G3 X135.543 Y129.791 I1.678 J3.591 E.11069
G3 X138.371 Y131.66 I-1.885 J5.927 E.13106
G2 X140.256 Y133.535 I6.403 J-4.55 E.10202
G2 X141.945 Y133.896 I1.468 J-2.743 E.06684
G1 X141.945 Y140.799 E.26358
G1 X141.198 Y140.324 E.03381
G3 X138.842 Y137.867 I89.269 J-87.962 E.12999
G2 X136.485 Y137.167 I-1.998 J2.409 E.09645
G2 X132.715 Y139.2 I.565 J5.56 E.16785
G3 X130.83 Y141.076 I-6.403 J-4.55 E.10202
G3 X130.269 Y141.313 I-.931 J-1.422 E.02335
G3 X134.6 Y144.707 I-83.818 J111.397 E.2101
G3 X138.371 Y146.741 I-.565 J5.56 E.16785
G2 X140.256 Y148.617 I6.403 J-4.55 E.10202
G2 X141.945 Y148.977 I1.468 J-2.743 E.06684
G1 X141.945 Y151.32 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 4.84
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y150.32 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L30
M991 S0 P29 ;notify layer change


G17
G3 Z5.08 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z4.84
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00922
G1 X122.328 Y123.268 E.20115
G1 X121.921 Y123.634 E.02089
G1 X121.47 Y123.859 E.01925
G1 X120.934 Y123.963 E.02085
G1 X120.385 Y123.913 E.02103
G1 X119.922 Y123.739 E.01887
G3 X118.918 Y122.954 I5.678 J-8.297 E.0487
G3 X114.848 Y126.662 I-39.399 J-39.16 E.2103
G1 X118.015 Y129.012 E.15055
G1 X118.485 Y129.402 E.02333
G1 X119.157 Y130.105 E.03711
G1 X119.749 Y130.932 E.03883
G3 X120.715 Y133.69 I-6.561 J3.845 E.11229
G1 X120.765 Y134.489 E.03056
G1 X120.698 Y135.504 E.03882
G1 X120.538 Y136.268 E.0298
G3 X130.504 Y142.099 I-21.932 J48.915 E.44169
G3 X142.443 Y154.069 I-32.636 J44.489 E.64804
G1 X142.443 Y104.609 E1.88834
G1 X142.323 Y104.677 E.00527
G1 X141.722 Y104.772 E.02324
G1 X132.869 Y104.772 E.33799
G1 X132.567 Y104.749 E.01158
G1 X131.954 Y104.544 E.02465
G1 X131.222 Y103.858 E.03831
G3 X124.807 Y115.927 I-51.206 J-19.477 E.52322
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.091 Y117.436 E.02015
G1 X126.168 Y117.978 E.02088
G1 X126.11 Y118.451 E.0182
G1 X125.95 Y118.874 E.01727
G1 X125.896 Y118.955 E.0037
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.324 Y118.777 E.00547
G3 X124.449 Y119.826 I-15.06 J-11.668 E.05218
G1 X121.879 Y122.892 E.15272
G1 X121.594 Y123.148 E.01461
G1 X121.188 Y123.333 E.01707
G1 X120.736 Y123.376 E.01731
G1 X120.383 Y123.303 E.01376
G1 X119.957 Y123.061 E.01871
G1 X118.881 Y122.159 E.05363
G3 X113.893 Y126.683 I-41.603 J-40.853 E.25725
G1 X117.666 Y129.482 E.17936
G1 X118.108 Y129.85 E.02196
G1 X118.727 Y130.503 E.03437
G1 X119.266 Y131.263 E.03556
G3 X120.135 Y133.792 I-5.94 J3.456 E.10278
G1 X120.18 Y134.525 E.02801
G1 X120.115 Y135.453 E.03554
G1 X119.931 Y136.288 E.03264
G1 X119.836 Y136.599 E.01241
G3 X130.932 Y143.157 I-21.28 J48.676 E.49331
G3 X142.92 Y155.769 I-32.176 J42.588 E.66732
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y118.036 E1.43984
G3 X143.04 Y103.306 I1890.781 J-6 E.56235
; LINE_WIDTH: 0.598746
G1 F13952.625
G1 X143.047 Y103.055 E.00925
G1 X142.989 Y103.297 E.00917
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.62 Y103.85 E.02538
G1 X142.142 Y104.12 E.02094
G1 X141.722 Y104.187 E.01626
G1 X132.869 Y104.187 E.33799
G1 X132.657 Y104.17 E.00811
G1 X132.204 Y104.013 E.01833
G1 F13348.453
G1 X131.996 Y103.814 E.01099
G1 F12338.304
G1 X131.707 Y103.538 E.01527
; LINE_WIDTH: 0.668465
G1 F11000.054
G1 X131.663 Y103.427 E.00491
; LINE_WIDTH: 0.716934
G1 F10617.61
G1 X131.619 Y103.317 E.00528
; LINE_WIDTH: 0.765403
G1 F10241.903
G1 X131.575 Y103.207 E.00566
; LINE_WIDTH: 0.813872
G1 F9872.992
G1 X131.531 Y103.096 E.00603
; LINE_WIDTH: 0.862341
G1 F9510.82
G1 X131.487 Y102.986 E.00641
; LINE_WIDTH: 0.91081
G1 F8984.872
G1 X131.443 Y102.876 E.00678
; LINE_WIDTH: 0.959278
G1 F8514.047
G1 X131.399 Y102.766 E.00716
; LINE_WIDTH: 1.00775
G1 F8090.107
G1 X131.355 Y102.655 E.00753
; LINE_WIDTH: 1.05622
G1 F7706.384
G1 X131.311 Y102.545 E.00791
G1 X131.266 Y102.599 E.00469
; LINE_WIDTH: 1.00775
G1 F8090.107
G1 X131.22 Y102.653 E.00447
; LINE_WIDTH: 0.959278
G1 F8514.047
G1 X131.175 Y102.707 E.00425
; LINE_WIDTH: 0.91081
G1 F8984.872
G1 X131.13 Y102.76 E.00403
; LINE_WIDTH: 0.862341
G1 F9510.82
G1 X131.084 Y102.814 E.0038
; LINE_WIDTH: 0.813872
G1 F10102.17
G1 X131.039 Y102.868 E.00358
; LINE_WIDTH: 0.765403
G1 F10322.783
G1 X130.993 Y102.922 E.00336
; LINE_WIDTH: 0.716934
G1 F10545.823
G1 X130.948 Y102.976 E.00314
; LINE_WIDTH: 0.668465
G1 F10771.202
G1 X130.902 Y103.03 E.00291
; LINE_WIDTH: 0.619996
G1 F12095.859
G1 X130.759 Y103.403 E.01527
G1 F13446.369
G1 X130.616 Y103.777 E.01527
G1 X130.246 Y104.735 E.03923
G3 X124.018 Y116.03 I-50.105 J-20.265 E.49361
G1 X125.095 Y116.932 E.05365
G1 X125.394 Y117.286 E.01771
G1 X125.529 Y117.599 E.01299
G1 X125.582 Y117.986 E.01491
G1 X125.512 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.168 Y122.718 E.01264
G1 X120.874 Y122.793 E.01161
G1 X120.577 Y122.75 E.01146
G1 X120.334 Y122.612 E.01068
G1 X118.837 Y121.358 E.07455
G3 X112.93 Y126.698 I-40.623 J-39.003 E.30428
G1 X117.317 Y129.952 E.20856
G1 X117.73 Y130.298 E.02058
G1 X118.298 Y130.902 E.03162
G1 X118.783 Y131.594 E.03228
G3 X119.505 Y133.559 I-5.511 J3.139 E.08028
G1 X119.596 Y134.56 E.03839
G1 X119.531 Y135.403 E.03226
G1 X119.362 Y136.153 E.02936
G1 X119.08 Y136.91 E.03085
G3 X130.579 Y143.624 I-20.428 J48.193 E.50973
G3 X142.647 Y156.415 I-31.78 J42.074 E.67458
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.525 E2.16516
G1 X142.966 Y99.593 E.02488
G1 X141.917 Y99.599 E.04008
G1 X142.489 Y102.688 E.11994
G3 X142.451 Y103.096 I-.767 J.135 E.01581
G1 X142.234 Y103.409 E.01454
G1 X141.962 Y103.563 E.01195
G1 X141.722 Y103.601 E.00928
G1 X132.869 Y103.601 E.33799
G1 X132.489 Y103.502 E.01498
G1 X132.197 Y103.215 E.01566
G1 X132.105 Y102.969 E.01001
G1 X132.125 Y102.567 E.01537
G1 X132.674 Y99.598 E.11528
G3 X131.473 Y99.49 I.2 J-8.943 E.0461
G1 X131.308 Y99.983 E.01984
G3 X123.222 Y116.127 I-51.32 J-15.607 E.69265
G1 X124.718 Y117.381 E.07453
G1 X124.934 Y117.671 E.01379
G1 X124.997 Y117.982 E.01212
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.007 Y122.161 E.16681
G1 X120.845 Y122.241 E.00571
G1 X120.689 Y122.189 E.00523
G1 X119.678 Y121.342 E.04174
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02697
; LINE_WIDTH: 0.531606
G1 X118.977 Y120.758 E.00202
; LINE_WIDTH: 0.556156
G1 X118.953 Y120.481 E.00946
G1 X117.999 Y121.433 E.04583
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.722 J-39.101 E.25281
G1 X116.988 Y130.396 E.19623
G1 X117.374 Y130.721 E.01599
G1 X117.893 Y131.278 E.02408
G1 X118.327 Y131.907 E.0242
G3 X118.995 Y133.831 I-5.113 J2.853 E.06483
G1 X119.044 Y134.594 E.02419
G1 X118.98 Y135.355 E.02419
G1 X118.824 Y136.025 E.02178
G1 X118.547 Y136.754 E.02467
G1 X118.31 Y137.19 E.01572
G3 X130.248 Y144.067 I-19.803 J48.177 E.43749
G3 X142.39 Y157.025 I-31.645 J41.818 E.56493
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.41 E1.81364
G1 X144.167 Y98.944 E.01475
G1 X142.909 Y99.043 E.03998
G3 X141.252 Y99.048 I-.965 J-47.693 E.05245
G1 X141.944 Y102.784 E.12029
G1 X141.933 Y102.902 E.00376
G1 X141.791 Y103.037 E.00621
G1 X132.869 Y103.048 E.28248
G1 X132.759 Y103.02 E.0036
G1 X132.647 Y102.865 E.00603
G3 X132.974 Y101.015 I69.842 J11.384 E.05949
G1 X133.339 Y99.048 E.06332
G3 X132.125 Y99.015 I-.296 J-11.225 E.03844
G2 X131.038 Y98.936 I-.887 J4.721 E.0346
G3 X122.648 Y115.977 I-50.883 J-14.466 E.60463
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00556
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.69 Y113.976 Z5.24 F36000
G1 X143.047 Y103.055 Z5.24
G1 Z4.84
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.585496
G1 F14288.048
G1 X143.046 Y102.577 E.01717
; LINE_WIDTH: 0.625416
G1 F13323.07
G1 X143.026 Y102.36 E.00841
; LINE_WIDTH: 0.671131
G1 F12366.619
G1 X143.003 Y102.111 E.01038
; LINE_WIDTH: 0.716846
G1 F11538.294
G1 X142.98 Y101.862 E.01112
; LINE_WIDTH: 0.762561
G1 F10813.966
G1 X142.958 Y101.613 E.01187
; LINE_WIDTH: 0.808276
G1 F10175.209
G1 X142.935 Y101.364 E.01261
; LINE_WIDTH: 0.853991
G1 F9607.701
G1 X142.912 Y101.115 E.01336
; LINE_WIDTH: 0.899706
G1 F9100.154
G1 X142.889 Y100.866 E.0141
; LINE_WIDTH: 0.945421
G1 F8643.541
G1 X142.866 Y100.617 E.01485
; LINE_WIDTH: 0.991136
G1 F8230.561
G1 X142.843 Y100.368 E.01559
; WIPE_START
G1 X142.866 Y100.617 E-.095
G1 X142.889 Y100.866 E-.095
G1 X142.912 Y101.115 E-.095
G1 X142.935 Y101.364 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.341 Y102.136 Z5.24 F36000
G1 X131.311 Y102.545 Z5.24
G1 Z4.84
G1 E.4 F1800
; LINE_WIDTH: 1.05622
G1 F7706.384
G1 X131.344 Y102.423 E.00843
; LINE_WIDTH: 1.03542
G1 F7866.505
G1 X131.429 Y102.099 E.02186
; LINE_WIDTH: 0.98687
G1 F8267.427
G1 X131.514 Y101.775 E.0208
; LINE_WIDTH: 0.938323
G1 F8711.413
G1 X131.599 Y101.451 E.01974
; LINE_WIDTH: 0.889776
G1 F9205.79
G1 X131.677 Y101.137 E.01806
; LINE_WIDTH: 0.849631
G1 F9659.081
G1 X131.756 Y100.822 E.01722
; LINE_WIDTH: 0.809486
G1 F10159.324
G1 X131.834 Y100.508 E.01637
; LINE_WIDTH: 0.769341
G1 F10714.214
G1 X131.913 Y100.194 E.01552
; LINE_WIDTH: 0.729196
G1 F11333.219
G1 X131.914 Y100.191 E.00014
; WIPE_START
G1 X131.913 Y100.194 E-.00115
G1 X131.834 Y100.508 E-.1231
G1 X131.756 Y100.822 E-.1231
G1 X131.677 Y101.137 E-.1231
G1 X131.671 Y101.161 E-.00956
; WIPE_END
G1 E-.02 F1800
G1 X127.474 Y107.536 Z5.24 F36000
G1 X118.953 Y120.481 Z5.24
G1 Z4.84
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556156
G1 F3600
M204 S5000
G1 X119.2 Y120.199 E.01276
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15232
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.13088
G1 X121.722 Y117.19 E-.24912
; WIPE_END
G1 E-.02 F1800
G1 X127.006 Y111.683 Z5.24 F36000
G1 X129.432 Y109.155 Z5.24
G1 Z4.84
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.339 Y111.227 I-36.135 J-17.732 E.08944
G2 X130.83 Y111.11 I1.088 J-3.411 E.09724
G2 X131.772 Y110.338 I-1.62 J-2.941 E.04677
G3 X133.658 Y108.193 I13.794 J10.223 E.10915
G3 X137.899 Y106.99 I3.402 J3.916 E.17386
G3 X139.313 Y107.947 I-1.064 J3.094 E.06596
G2 X141.198 Y110.092 I13.793 J-10.221 E.10915
G1 X141.945 Y110.612 E.03476
G1 X141.945 Y118.908 E.31673
G3 X139.313 Y117.879 I-.417 J-2.814 E.11293
G2 X137.428 Y115.734 I-13.797 J10.225 E.10915
G2 X133.186 Y114.53 I-3.403 J3.916 E.17386
G2 X131.772 Y115.488 I1.064 J3.094 E.06596
G3 X129.887 Y117.633 I-13.794 J-10.223 E.10915
G3 X126.472 Y118.934 I-3.598 J-4.312 E.14214
G3 X126.202 Y119.412 I-2.254 J-.954 E.02099
G1 X123.632 Y122.485 E.15297
G3 X125.174 Y124.182 I-4.736 J5.854 E.0879
G2 X129.887 Y126.465 I4.547 J-3.381 E.20783
G2 X131.772 Y125.419 I-.382 J-2.91 E.08428
G3 X133.658 Y123.274 I13.793 J10.221 E.10915
G3 X137.899 Y122.071 I3.403 J3.916 E.17386
G3 X139.313 Y123.028 I-1.064 J3.094 E.06596
G2 X141.198 Y125.173 I13.792 J-10.221 E.10915
G1 X141.945 Y125.694 E.03476
G1 X141.945 Y128.036 E.08944
G1 X120.585 Y131.486 F36000
G1 F13446.283
G2 X119.306 Y129.539 I-6.843 J3.101 E.08929
G3 X123.289 Y131.806 I-.595 J5.677 E.18005
G2 X124.703 Y133.417 I7.658 J-5.294 E.08203
G2 X127.06 Y133.985 I1.821 J-2.382 E.09527
G2 X130.83 Y131.723 I-1.031 J-5.99 E.17177
G3 X132.244 Y130.112 I7.658 J5.294 E.08203
G3 X134.6 Y129.544 I1.821 J2.382 E.09527
G3 X138.371 Y131.806 I-1.03 J5.99 E.17177
G2 X139.784 Y133.417 I7.658 J-5.294 E.08203
G2 X141.945 Y133.99 I1.728 J-2.159 E.08786
G1 X141.945 Y140.775 E.25906
G3 X140.256 Y139.264 I3.438 J-5.544 E.087
G2 X138.842 Y137.652 I-7.657 J5.293 E.08203
G2 X136.485 Y137.084 I-1.821 J2.382 E.09527
G2 X132.715 Y139.347 I1.031 J5.99 E.17177
G3 X131.301 Y140.958 I-7.658 J-5.294 E.08203
G3 X130.426 Y141.432 I-1.535 J-1.792 E.03829
G3 X134.451 Y144.622 I-54.614 J73.045 E.19611
G1 X134.6 Y144.625 E.00571
G3 X138.371 Y146.887 I-1.031 J5.99 E.17177
G2 X139.784 Y148.499 I7.658 J-5.294 E.08203
G2 X141.945 Y149.071 I1.728 J-2.159 E.08786
G1 X141.945 Y151.414 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 5
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y150.414 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L31
M991 S0 P30 ;notify layer change


G17
G3 Z5.24 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z5
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00922
G1 X122.328 Y123.268 E.20115
G1 X121.942 Y123.62 E.01993
G1 X121.522 Y123.84 E.0181
G1 X121.016 Y123.957 E.01984
G1 X120.455 Y123.928 E.02144
G1 X119.96 Y123.759 E.02
G3 X118.918 Y122.954 I5.013 J-7.567 E.05028
G3 X114.848 Y126.662 I-39.399 J-39.159 E.2103
G1 X118.015 Y129.012 E.15055
G3 X118.932 Y129.849 I-4.609 J5.971 E.04749
G1 X119.566 Y130.647 E.03888
G1 X120.076 Y131.531 E.03897
G1 X120.45 Y132.48 E.03896
G1 X120.671 Y133.42 E.03686
G1 X120.765 Y134.489 E.04096
G1 X120.698 Y135.505 E.03889
G1 X120.538 Y136.268 E.02974
G3 X131.286 Y142.69 I-21.89 J48.838 E.47909
G3 X142.443 Y154.069 I-32.822 J43.343 E.61065
G1 X142.443 Y104.613 E1.88819
G1 X142.326 Y104.679 E.00516
G1 X141.724 Y104.775 E.02324
G1 X132.867 Y104.775 E.33818
G1 X132.567 Y104.751 E.01149
G1 X132.036 Y104.589 E.02119
G1 X131.591 Y104.299 E.02027
G1 X131.22 Y103.863 E.02184
G3 X124.807 Y115.927 I-51.666 J-19.728 E.52296
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.091 Y117.436 E.02015
G1 X126.168 Y117.978 E.02088
G1 X126.11 Y118.451 E.0182
G1 X125.95 Y118.874 E.01727
G1 X125.896 Y118.955 E.0037
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.324 Y118.777 E.00547
G3 X124.449 Y119.826 I-15.06 J-11.668 E.05218
G1 X121.879 Y122.892 E.15272
G1 X121.609 Y123.138 E.01394
G1 X121.315 Y123.292 E.01266
G1 X120.961 Y123.374 E.01388
G1 X120.52 Y123.343 E.01689
G1 X120.196 Y123.222 E.0132
G3 X118.881 Y122.159 I9.151 J-12.669 E.0646
G3 X113.893 Y126.683 I-41.609 J-40.86 E.25725
G1 X117.666 Y129.482 E.17936
G3 X118.523 Y130.268 I-4.174 J5.412 E.04447
M73 P63 R7
G1 X119.099 Y131.001 E.0356
G1 X119.563 Y131.813 E.03569
G1 X119.901 Y132.684 E.03567
G1 X120.098 Y133.541 E.03358
G1 X120.18 Y134.525 E.03768
G1 X120.114 Y135.455 E.0356
G1 X119.931 Y136.289 E.0326
G1 X119.836 Y136.599 E.01239
G3 X130.932 Y143.157 I-21.126 J48.416 E.49334
G3 X142.92 Y155.769 I-32.372 J42.774 E.66728
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y118.037 E1.43981
G3 X143.04 Y103.308 I1768.749 J-6 E.56232
; LINE_WIDTH: 0.597286
G1 F13988.809
G1 X143.049 Y103.057 E.00922
G1 X142.991 Y103.299 E.00914
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.622 Y103.852 E.02537
G1 X142.145 Y104.122 E.02094
G1 X141.724 Y104.189 E.01626
G1 X132.867 Y104.189 E.33818
G1 X132.657 Y104.173 E.00804
G1 X132.259 Y104.046 E.01593
G1 X131.974 Y103.856 E.01307
G1 F12732.621
G1 X131.69 Y103.515 E.01697
; LINE_WIDTH: 0.668081
G1 F11226.105
G1 X131.635 Y103.392 E.00554
; LINE_WIDTH: 0.716166
G1 F10789.84
G1 X131.579 Y103.27 E.00597
; LINE_WIDTH: 0.764251
G1 F10362.22
G1 X131.524 Y103.148 E.00639
; LINE_WIDTH: 0.812336
G1 F9943.246
G1 X131.469 Y103.025 E.00681
; LINE_WIDTH: 0.860421
G1 F9532.919
G1 X131.414 Y102.903 E.00723
; LINE_WIDTH: 0.908506
G1 F9008.546
G1 X131.359 Y102.781 E.00765
; LINE_WIDTH: 0.946464
G1 F8633.662
G1 X131.345 Y102.729 E.00316
; LINE_WIDTH: 0.984421
G1 F8288.733
G1 X131.33 Y102.678 E.0033
; LINE_WIDTH: 1.02238
G1 F7970.305
G1 X131.316 Y102.627 E.00343
; LINE_WIDTH: 1.06034
G1 F7675.439
G1 X131.302 Y102.575 E.00356
G1 X131.267 Y102.617 E.00363
; LINE_WIDTH: 1.02238
G1 F7970.305
G1 X131.233 Y102.659 E.0035
; LINE_WIDTH: 0.984421
G1 F8288.733
G1 X131.198 Y102.701 E.00336
; LINE_WIDTH: 0.946464
G1 F8633.662
G1 X131.164 Y102.743 E.00323
; LINE_WIDTH: 0.908506
G1 F9008.546
G1 X131.085 Y102.886 E.00932
; LINE_WIDTH: 0.860421
G1 F9532.919
G1 X131.007 Y103.03 E.0088
; LINE_WIDTH: 0.812336
G1 F10122.108
G1 X130.928 Y103.173 E.00829
; LINE_WIDTH: 0.764251
G1 F10638.1
G1 X130.85 Y103.316 E.00778
; LINE_WIDTH: 0.716166
G1 F11166.903
G1 X130.771 Y103.46 E.00727
; LINE_WIDTH: 0.668081
G1 F11708.533
G1 X130.693 Y103.603 E.00675
; LINE_WIDTH: 0.619996
G1 F13087.988
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.209 Y104.835 E.01998
G3 X124.018 Y116.03 I-50.128 J-20.414 E.48956
G1 X125.095 Y116.932 E.05365
G1 X125.394 Y117.287 E.01771
G1 X125.529 Y117.599 E.01299
G1 X125.582 Y117.986 E.01491
G1 X125.512 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.165 Y122.72 E.01279
G1 X120.906 Y122.791 E.01022
G1 X120.577 Y122.75 E.01269
G1 X120.334 Y122.612 E.01067
G1 X118.837 Y121.358 E.07455
G3 X112.93 Y126.698 I-40.622 J-39.002 E.30428
G1 X117.317 Y129.952 E.20856
G1 X118.114 Y130.687 E.04139
G1 X118.633 Y131.356 E.03231
G1 X119.049 Y132.096 E.03241
G1 X119.352 Y132.888 E.03238
G1 X119.524 Y133.662 E.0303
G1 X119.596 Y134.56 E.03439
G1 X119.531 Y135.404 E.03232
G1 X119.361 Y136.154 E.02932
G1 X119.08 Y136.91 E.03082
G3 X130.579 Y143.624 I-20.519 J48.349 E.50973
G3 X142.647 Y156.415 I-31.962 J42.245 E.67454
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.525 E2.16513
G1 X142.966 Y99.595 E.02489
G1 X141.919 Y99.601 E.03998
G1 X142.491 Y102.69 E.11994
G3 X142.453 Y103.098 I-.767 J.135 E.01581
G1 X142.237 Y103.411 E.01454
G1 X141.964 Y103.565 E.01195
G1 X141.724 Y103.603 E.00928
G1 X132.867 Y103.603 E.33818
G1 X132.52 Y103.522 E.0136
G3 X132.195 Y103.218 I.347 J-.697 E.01721
G1 X132.089 Y102.785 E.01703
G1 X132.101 Y102.683 E.00393
G1 X132.672 Y99.6 E.11969
G3 X131.471 Y99.49 I.202 J-8.797 E.04607
G1 X131.308 Y99.983 E.01983
G3 X123.222 Y116.127 I-51.397 J-15.645 E.69264
G1 X124.718 Y117.381 E.07453
G1 X124.934 Y117.671 E.01379
G1 X124.997 Y117.982 E.01212
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.003 Y122.165 E.16698
G1 X120.855 Y122.24 E.00527
G1 X120.689 Y122.189 E.00551
G1 X119.678 Y121.342 E.04175
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02697
; LINE_WIDTH: 0.531596
G1 X118.977 Y120.758 E.00201
; LINE_WIDTH: 0.556196
G1 X118.953 Y120.481 E.00947
G1 X117.999 Y121.433 E.04584
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.725 J-39.104 E.25281
G1 X116.988 Y130.396 E.19622
G1 X117.728 Y131.083 E.03196
G1 X118.193 Y131.69 E.02422
G1 X118.565 Y132.362 E.0243
G1 X118.834 Y133.08 E.02428
G1 X118.984 Y133.776 E.02255
G1 X119.044 Y134.594 E.02595
G1 X118.98 Y135.357 E.02423
G1 X118.824 Y136.026 E.02175
G1 X118.548 Y136.753 E.02462
G1 X118.31 Y137.19 E.01575
G3 X130.249 Y144.067 I-19.583 J47.796 E.43752
G3 X142.39 Y157.025 I-31.645 J41.817 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.409 E1.81369
G1 X144.167 Y98.944 E.0147
G1 X142.908 Y99.045 E.04001
G3 X141.255 Y99.05 I-.956 J-44.479 E.05233
G1 X141.947 Y102.786 E.12029
G1 X141.936 Y102.904 E.00376
G1 X141.794 Y103.039 E.00621
G1 X141.724 Y103.05 E.00223
G1 X132.867 Y103.05 E.28044
G1 X132.719 Y102.995 E.00499
G1 X132.641 Y102.813 E.00627
G1 X132.645 Y102.784 E.00095
G1 X133.336 Y99.05 E.12021
G3 X132.125 Y99.016 I-.295 J-10.993 E.03837
G2 X131.038 Y98.936 I-.887 J4.636 E.0346
G3 X122.648 Y115.977 I-50.883 J-14.467 E.60463
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00556
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.69 Y113.977 Z5.4 F36000
G1 X143.049 Y103.057 Z5.4
G1 Z5
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.582976
G1 F14353.675
G1 X143.047 Y102.58 E.01707
; LINE_WIDTH: 0.623146
G1 F13374.435
G1 X143.027 Y102.361 E.00843
; LINE_WIDTH: 0.668863
G1 F12410.834
G1 X143.004 Y102.112 E.01034
; LINE_WIDTH: 0.714579
G1 F11576.755
G1 X142.982 Y101.863 E.01109
; LINE_WIDTH: 0.760295
G1 F10847.725
G1 X142.959 Y101.614 E.01183
; LINE_WIDTH: 0.806011
G1 F10205.074
G1 X142.936 Y101.365 E.01258
; LINE_WIDTH: 0.851727
G1 F9634.31
G1 X142.913 Y101.116 E.01332
; LINE_WIDTH: 0.897444
G1 F9124.009
G1 X142.89 Y100.867 E.01407
; LINE_WIDTH: 0.94316
G1 F8665.047
G1 X142.867 Y100.618 E.01481
; LINE_WIDTH: 0.988876
G1 F8250.048
G1 X142.844 Y100.369 E.01556
; WIPE_START
G1 X142.867 Y100.618 E-.095
G1 X142.89 Y100.867 E-.095
G1 X142.913 Y101.116 E-.095
G1 X142.936 Y101.365 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.344 Y102.155 Z5.4 F36000
G1 X131.302 Y102.575 Z5.4
G1 Z5
G1 E.4 F1800
; LINE_WIDTH: 1.06034
G1 F7675.439
G1 X131.312 Y102.537 E.00267
; LINE_WIDTH: 1.0539
G1 F7723.92
G1 X131.381 Y102.274 E.01805
; LINE_WIDTH: 1.01359
G1 F8041.828
G1 X131.45 Y102.012 E.01734
; LINE_WIDTH: 0.973286
G1 F8387.029
G1 X131.519 Y101.749 E.01662
; LINE_WIDTH: 0.932981
G1 F8763.194
G1 X131.588 Y101.486 E.01591
; LINE_WIDTH: 0.892676
G1 F9174.687
G1 X131.598 Y101.451 E.00206
; LINE_WIDTH: 0.887656
G1 F9228.661
G1 X131.676 Y101.136 E.01802
; LINE_WIDTH: 0.847521
G1 F9684.144
G1 X131.755 Y100.822 E.01717
; LINE_WIDTH: 0.807386
G1 F10186.922
G1 X131.833 Y100.508 E.01633
; LINE_WIDTH: 0.767251
G1 F10744.766
G1 X131.912 Y100.193 E.01548
; LINE_WIDTH: 0.727116
G1 F11367.247
G1 X131.913 Y100.191 E.00011
; WIPE_START
G1 X131.912 Y100.193 E-.00094
G1 X131.833 Y100.508 E-.12311
G1 X131.755 Y100.822 E-.12311
G1 X131.676 Y101.136 E-.12311
G1 X131.67 Y101.161 E-.00974
; WIPE_END
G1 E-.02 F1800
G1 X127.473 Y107.536 Z5.4 F36000
G1 X118.953 Y120.481 Z5.4
G1 Z5
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556196
G1 F3600
M204 S5000
G1 X119.2 Y120.198 E.01278
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15231
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.13088
G1 X121.722 Y117.19 E-.24912
; WIPE_END
G1 E-.02 F1800
G1 X127.016 Y111.693 Z5.4 F36000
G1 X129.409 Y109.208 Z5.4
G1 Z5
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X128.314 Y111.278 I-62.611 J-31.796 E.08943
G2 X130.83 Y111.326 I1.325 J-3.555 E.09793
G2 X131.772 Y110.576 I-1.202 J-2.48 E.04635
G3 X133.658 Y108.26 I16.159 J11.228 E.11412
G3 X137.899 Y106.808 I3.741 J4.004 E.17635
G3 X139.313 Y107.709 I-.702 J2.661 E.06505
G2 X141.198 Y110.025 I16.157 J-11.226 E.11412
G1 X141.945 Y110.593 E.03584
G1 X141.945 Y119.01 E.32136
G3 X139.313 Y118.117 I-.61 J-2.526 E.11198
G2 X137.428 Y115.801 I-16.162 J11.23 E.11412
G2 X133.186 Y114.348 I-3.741 J4.004 E.17635
G2 X131.772 Y115.25 I.702 J2.661 E.06505
G3 X129.887 Y117.565 I-16.157 J-11.226 E.11412
G3 X126.418 Y119.048 I-3.727 J-3.922 E.14712
G3 X126.202 Y119.412 I-1.724 J-.774 E.0162
G1 X123.76 Y122.331 E.14531
G1 X124.232 Y122.79 E.02513
G2 X126.117 Y125.106 I16.158 J-11.227 E.11412
G2 X130.359 Y126.559 I3.741 J-4.004 E.17635
G2 X131.772 Y125.657 I-.702 J-2.661 E.06505
G3 X133.658 Y123.341 I16.157 J11.226 E.11412
G3 X137.899 Y121.889 I3.741 J4.004 E.17635
G3 X139.313 Y122.79 I-.702 J2.661 E.06505
G2 X141.198 Y125.106 I16.155 J-11.225 E.11412
G1 X141.945 Y125.674 E.03584
G1 X141.945 Y128.017 E.08944
G1 X120.533 Y131.343 F36000
G1 F13446.283
G2 X119.205 Y129.427 I-8.125 J4.214 E.08925
G3 X123.289 Y131.961 I-.8 J5.848 E.18899
G1 X124.232 Y133.198 E.05936
G2 X127.06 Y134.075 I2.125 J-1.854 E.119
G2 X130.83 Y131.567 I-1.371 J-6.15 E.17684
G1 X131.772 Y130.331 E.05936
G3 X134.6 Y129.454 I2.125 J1.854 E.119
G3 X138.371 Y131.961 I-1.371 J6.15 E.17684
G1 X139.313 Y133.198 E.05936
G2 X141.945 Y134.092 I2.022 J-1.633 E.11198
G1 X141.945 Y140.756 E.25443
G3 X140.256 Y139.108 I3.821 J-5.609 E.09057
G1 X139.313 Y137.871 E.05936
G2 X136.485 Y136.994 I-2.125 J1.854 E.119
G2 X132.715 Y139.502 I1.371 J6.15 E.17684
G1 X131.772 Y140.739 E.05936
G3 X130.605 Y141.561 I-2.093 J-1.733 E.05516
G3 X134.326 Y144.512 I-41.157 J55.703 E.18135
G3 X138.371 Y147.043 I-.846 J5.851 E.18752
G1 X139.313 Y148.279 E.05936
G2 X141.945 Y149.173 I2.022 J-1.633 E.11198
G1 X141.945 Y151.515 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 5.16
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y150.515 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L32
M991 S0 P31 ;notify layer change


G17
G3 Z5.4 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z5.16
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00922
G1 X122.328 Y123.268 E.20115
G1 X121.941 Y123.62 E.01996
G1 X121.522 Y123.84 E.01806
G1 X121.016 Y123.957 E.01983
G1 X120.455 Y123.928 E.02145
G1 X119.954 Y123.756 E.02022
G3 X118.918 Y122.954 I5.1 J-7.663 E.05006
G3 X114.848 Y126.662 I-38.5 J-38.174 E.21031
G1 X118.015 Y129.012 E.15055
G3 X118.933 Y129.849 I-4.605 J5.967 E.0475
G1 X119.565 Y130.647 E.03887
G1 X120.075 Y131.529 E.03887
G1 X120.45 Y132.478 E.03897
G1 X120.672 Y133.427 E.03721
G1 X120.765 Y134.489 E.04071
G1 X120.698 Y135.505 E.03888
G1 X120.538 Y136.268 E.02976
G3 X130.506 Y142.1 I-22.174 J49.33 E.44176
G3 X142.443 Y154.069 I-32.639 J44.489 E.64795
G1 X142.443 Y104.616 E1.88807
G1 X142.328 Y104.682 E.00505
G1 X141.727 Y104.777 E.02324
G1 X132.864 Y104.777 E.33837
G1 X132.563 Y104.753 E.01151
G1 X132.032 Y104.59 E.02121
G1 X131.587 Y104.3 E.02029
G1 X131.219 Y103.866 E.02172
G3 X124.807 Y115.927 I-51.663 J-19.73 E.52284
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.091 Y117.436 E.02015
G1 X126.168 Y117.978 E.02088
G1 X126.11 Y118.451 E.0182
G1 X125.95 Y118.874 E.01727
G1 X125.896 Y118.955 E.0037
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.324 Y118.777 E.00547
G3 X124.449 Y119.826 I-15.057 J-11.666 E.05218
G1 X121.879 Y122.892 E.15272
G1 X121.609 Y123.138 E.01397
G1 X121.315 Y123.292 E.01264
G1 X120.961 Y123.374 E.01387
G1 X120.52 Y123.343 E.0169
G1 X120.196 Y123.222 E.0132
G3 X118.881 Y122.159 I9.151 J-12.669 E.0646
G3 X113.893 Y126.683 I-41.231 J-40.444 E.25725
G1 X117.666 Y129.482 E.17936
G3 X118.524 Y130.268 I-4.17 J5.409 E.04448
G1 X119.099 Y131.001 E.03558
G1 X119.561 Y131.811 E.03559
G1 X119.901 Y132.682 E.03568
G1 X120.099 Y133.548 E.03392
G1 X120.18 Y134.525 E.03742
G1 X120.114 Y135.454 E.03559
G1 X119.931 Y136.289 E.03263
G1 X119.836 Y136.599 E.01237
G3 X130.932 Y143.157 I-21.127 J48.417 E.49333
G3 X142.92 Y155.769 I-32.375 J42.777 E.66729
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y118.038 E1.43977
G3 X143.041 Y103.31 I1660.087 J-6 E.56229
; LINE_WIDTH: 0.595806
G1 F14025.684
G1 X143.05 Y103.059 E.00919
G1 X142.993 Y103.301 E.00911
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.625 Y103.854 E.02536
G1 X142.147 Y104.124 E.02094
G1 X141.727 Y104.191 E.01626
G1 X132.864 Y104.191 E.33837
G1 X132.654 Y104.175 E.00806
G1 X132.256 Y104.048 E.01594
G1 X131.971 Y103.858 E.01309
G1 F12428.712
G1 X131.685 Y103.513 E.01707
; LINE_WIDTH: 0.669645
G1 F10932.257
G1 X131.639 Y103.404 E.00493
; LINE_WIDTH: 0.719294
G1 F10549.873
G1 X131.592 Y103.294 E.00532
; LINE_WIDTH: 0.768942
G1 F10174.296
G1 X131.546 Y103.185 E.0057
; LINE_WIDTH: 0.818591
G1 F9805.527
G1 X131.499 Y103.075 E.00609
; LINE_WIDTH: 0.86824
G1 F9443.536
G1 X131.453 Y102.965 E.00647
; LINE_WIDTH: 0.917889
G1 F8912.884
G1 X131.406 Y102.856 E.00686
; LINE_WIDTH: 0.967538
G1 F8438.694
G1 X131.359 Y102.746 E.00724
; LINE_WIDTH: 1.01719
G1 F8012.413
G1 X131.313 Y102.637 E.00763
; LINE_WIDTH: 1.04166
G1 F7817.775
G1 X131.286 Y102.598 E.00312
G1 X131.263 Y102.625 E.00231
; LINE_WIDTH: 1.01719
G1 F8012.413
G1 X131.192 Y102.747 E.00905
; LINE_WIDTH: 0.967538
G1 F8438.694
G1 X131.121 Y102.869 E.0086
; LINE_WIDTH: 0.917889
G1 F8912.884
G1 X131.05 Y102.991 E.00814
; LINE_WIDTH: 0.86824
G1 F9443.536
G1 X130.978 Y103.113 E.00768
; LINE_WIDTH: 0.818591
G1 F10041.378
G1 X130.907 Y103.235 E.00722
; LINE_WIDTH: 0.768942
G1 F10484.849
G1 X130.836 Y103.357 E.00677
; LINE_WIDTH: 0.719294
G1 F10937.902
G1 X130.765 Y103.479 E.00631
; LINE_WIDTH: 0.669645
G1 F11400.537
G1 X130.694 Y103.601 E.00585
; LINE_WIDTH: 0.619996
G1 F12762.237
G1 X130.547 Y103.973 E.01527
G1 F13446.369
G1 X130.401 Y104.346 E.01527
G1 X130.209 Y104.835 E.02007
G3 X124.018 Y116.03 I-50.127 J-20.413 E.48956
G1 X125.095 Y116.932 E.05365
G1 X125.394 Y117.287 E.01771
G1 X125.529 Y117.599 E.01299
G1 X125.582 Y117.986 E.01491
G1 X125.512 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.165 Y122.72 E.0128
G1 X120.907 Y122.791 E.01021
G1 X120.577 Y122.751 E.01268
G1 X120.334 Y122.612 E.01069
G1 X118.837 Y121.358 E.07455
G3 X112.93 Y126.698 I-40.368 J-38.721 E.30429
G1 X117.317 Y129.952 E.20856
G1 X118.114 Y130.687 E.04141
G1 X118.633 Y131.356 E.0323
G1 X119.048 Y132.093 E.03231
G1 X119.352 Y132.886 E.0324
G1 X119.526 Y133.669 E.03063
G1 X119.596 Y134.56 E.03413
G1 X119.531 Y135.404 E.03231
G1 X119.361 Y136.154 E.02936
G1 X119.08 Y136.91 E.0308
G3 X130.579 Y143.624 I-20.546 J48.395 E.50971
G3 X142.648 Y156.415 I-31.964 J42.247 E.67455
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.526 E2.1651
G1 X142.966 Y99.597 E.0249
G1 X141.922 Y99.603 E.03988
G1 X142.494 Y102.692 E.11994
G3 X142.456 Y103.1 I-.767 J.135 E.01581
G1 X142.239 Y103.413 E.01454
G1 X141.967 Y103.567 E.01195
G1 X141.727 Y103.605 E.00928
G1 X132.864 Y103.605 E.33837
G1 X132.517 Y103.524 E.01361
G3 X132.191 Y103.219 I.347 J-.697 E.01728
G1 X132.094 Y102.716 E.01956
G3 X132.305 Y101.569 I28.151 J4.604 E.04453
G1 X132.669 Y99.602 E.07636
G3 X131.47 Y99.49 I.203 J-8.647 E.04605
G1 X131.308 Y99.983 E.01981
G3 X123.222 Y116.127 I-51.321 J-15.607 E.69265
G1 X124.718 Y117.381 E.07453
G1 X124.934 Y117.671 E.01379
G1 X124.997 Y117.982 E.01212
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.003 Y122.165 E.16698
G1 X120.855 Y122.24 E.00527
G1 X120.689 Y122.189 E.00551
G1 X119.678 Y121.342 E.04175
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02697
; LINE_WIDTH: 0.531586
G1 X118.977 Y120.758 E.00201
; LINE_WIDTH: 0.556156
G1 X118.953 Y120.481 E.00947
G1 X117.999 Y121.433 E.04583
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-37.697 J-36.8 E.25284
G1 X116.988 Y130.396 E.19622
G1 X117.728 Y131.083 E.03197
G1 X118.193 Y131.69 E.02421
G1 X118.564 Y132.36 E.02423
G1 X118.834 Y133.078 E.0243
G1 X118.985 Y133.783 E.02283
G1 X119.044 Y134.594 E.02573
G1 X118.98 Y135.356 E.02422
G1 X118.824 Y136.026 E.02178
G1 X118.548 Y136.752 E.02459
G1 X118.31 Y137.19 E.01577
G3 X130.248 Y144.067 I-19.556 J47.749 E.43751
G3 X142.39 Y157.025 I-31.645 J41.818 E.56493
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.407 E1.81375
G1 X144.167 Y98.944 E.01464
G1 X142.907 Y99.047 E.04005
G3 X141.257 Y99.052 I-.949 J-41.83 E.05222
G1 X141.949 Y102.788 E.12029
G1 X141.938 Y102.906 E.00376
M73 P64 R7
G1 X141.796 Y103.041 E.00621
G1 X141.727 Y103.052 E.00223
G1 X132.864 Y103.052 E.28059
G1 X132.716 Y102.997 E.00499
G1 X132.642 Y102.786 E.0071
G1 X133.334 Y99.052 E.12021
G3 X132.125 Y99.018 I-.294 J-10.785 E.0383
G2 X131.038 Y98.936 I-.887 J4.556 E.0346
M73 P64 R6
G3 X122.648 Y115.977 I-50.883 J-14.466 E.60463
; LINE_WIDTH: 0.552916
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00556
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.691 Y113.977 Z5.56 F36000
G1 X143.05 Y103.059 Z5.56
G1 Z5.16
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.580436
G1 F14420.437
G1 X143.049 Y102.582 E.01698
; LINE_WIDTH: 0.620866
G1 F13426.425
G1 X143.028 Y102.362 E.00845
; LINE_WIDTH: 0.666583
G1 F12455.591
G1 X143.006 Y102.113 E.0103
; LINE_WIDTH: 0.712299
G1 F11615.687
G1 X142.983 Y101.864 E.01105
; LINE_WIDTH: 0.758015
G1 F10881.902
G1 X142.96 Y101.615 E.01179
; LINE_WIDTH: 0.803731
G1 F10235.315
G1 X142.937 Y101.366 E.01254
; LINE_WIDTH: 0.849447
G1 F9661.258
G1 X142.914 Y101.117 E.01328
; LINE_WIDTH: 0.895164
G1 F9148.175
G1 X142.891 Y100.868 E.01403
; LINE_WIDTH: 0.94088
G1 F8686.84
G1 X142.868 Y100.619 E.01477
; LINE_WIDTH: 0.986596
G1 F8269.801
G1 X142.846 Y100.37 E.01552
; WIPE_START
G1 X142.868 Y100.619 E-.095
G1 X142.891 Y100.868 E-.095
G1 X142.914 Y101.117 E-.095
G1 X142.937 Y101.366 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.348 Y100.557 Z5.56 F36000
G1 X131.911 Y100.191 Z5.56
G1 Z5.16
G1 E.4 F1800
; LINE_WIDTH: 0.725016
G1 F11401.808
G1 X131.911 Y100.193 E.00009
; LINE_WIDTH: 0.765156
G1 F10775.568
G1 X131.832 Y100.507 E.01543
; LINE_WIDTH: 0.805296
G1 F10214.539
G1 X131.754 Y100.822 E.01628
; LINE_WIDTH: 0.845436
G1 F9709.037
G1 X131.675 Y101.136 E.01713
; LINE_WIDTH: 0.885576
G1 F9251.211
G1 X131.597 Y101.45 E.01798
; LINE_WIDTH: 0.890336
G1 F9199.766
G1 X131.588 Y101.484 E.00196
; LINE_WIDTH: 0.930486
G1 F8787.592
G1 X131.519 Y101.748 E.01593
; LINE_WIDTH: 0.970636
G1 F8410.766
G1 X131.449 Y102.012 E.01664
; LINE_WIDTH: 1.01079
G1 F8064.929
G1 X131.38 Y102.275 E.01736
; LINE_WIDTH: 1.05094
G1 F7746.409
G1 X131.311 Y102.539 E.01807
; LINE_WIDTH: 1.05766
G1 F7695.54
G1 X131.301 Y102.58 E.00278
; WIPE_START
G1 X131.311 Y102.539 E-.01584
G1 X131.38 Y102.275 E-.10362
G1 X131.449 Y102.012 E-.10362
G1 X131.519 Y101.748 E-.10362
G1 X131.554 Y101.612 E-.0533
; WIPE_END
G1 E-.02 F1800
G1 X127.315 Y107.959 Z5.56 F36000
G1 X118.953 Y120.481 Z5.56
G1 Z5.16
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556156
G1 F3600
M204 S5000
G1 X119.2 Y120.198 E.01277
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15231
; LINE_WIDTH: 0.552916
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.13087
G1 X121.722 Y117.19 E-.24913
; WIPE_END
G1 E-.02 F1800
G1 X118.759 Y123.811 Z5.56 F36000
G1 Z5.16
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X117.054 Y125.417 I-34.662 J-35.104 E.08944
G3 X115.685 Y126.664 I-2.494 J-1.364 E.072
G3 X118.668 Y128.899 I-26.666 J38.695 E.14234
G1 X119.063 Y129.288 E.02117
G3 X123.289 Y132.131 I-.914 J5.921 E.20086
G1 X124.232 Y133.475 E.06266
G2 X126.117 Y134.314 I1.72 J-1.325 E.08211
G2 X130.83 Y131.398 I-.481 J-6.044 E.21976
G1 X131.772 Y130.054 E.06266
G3 X133.658 Y129.215 I1.72 J1.325 E.08211
G3 X138.371 Y132.131 I-.481 J6.044 E.21976
G1 X139.313 Y133.475 E.06267
G2 X141.198 Y134.314 I1.72 J-1.325 E.08211
G1 X141.945 Y134.202 E.02884
G1 X141.945 Y140.76 E.25036
G3 X140.256 Y138.939 I6.072 J-7.328 E.09512
G1 X139.313 Y137.595 E.06266
G2 X137.428 Y136.755 I-1.72 J1.325 E.08211
G2 X132.715 Y139.671 I.481 J6.044 E.21976
G1 X131.772 Y141.015 E.06266
G3 X130.83 Y141.73 I-2.292 J-2.043 E.04545
G3 X134.159 Y144.371 I-69.121 J90.549 E.16225
G3 X138.371 Y147.212 I-.929 J5.92 E.2003
G1 X139.313 Y148.556 E.06266
G2 X141.198 Y149.396 I1.72 J-1.325 E.08211
G1 X141.945 Y149.284 E.02884
G1 X141.945 Y151.626 E.08944
; WIPE_START
G1 X141.945 Y150.626 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X135.782 Y146.125 Z5.56 F36000
G1 X121.2 Y135.475 Z5.56
G1 Z5.16
G1 E.4 F1800
G1 F13446.283
G3 X121.111 Y135.983 I-2.447 J-.17 E.01972
G1 X122.66 Y136.717 E.06544
G3 X124.09 Y137.457 I-.238 J2.211 E.06292
G3 X126.131 Y138.606 I-30.374 J56.333 E.08943
G1 X141.945 Y128.021 F36000
G1 F13446.283
G1 X141.945 Y125.678 E.08944
G1 X141.67 Y125.475 E.01307
G3 X139.313 Y122.514 I11.975 J-11.946 E.1448
G2 X137.899 Y121.689 I-1.776 J1.422 E.06392
G2 X134.129 Y122.972 I-.072 J5.97 E.15502
G2 X131.772 Y125.934 I11.976 J11.947 E.1448
G3 X130.359 Y126.758 I-1.776 J-1.421 E.06392
G3 X126.588 Y125.475 I-.072 J-5.97 E.15502
G3 X124.232 Y122.514 I11.975 J-11.946 E.1448
G1 X123.887 Y122.179 E.01834
G1 X126.332 Y119.201 E.14711
G2 X129.416 Y117.935 I-.534 J-5.69 E.12914
G2 X131.772 Y114.973 I-11.975 J-11.946 E.14481
G3 X133.186 Y114.149 I1.776 J1.422 E.06392
G3 X136.957 Y115.431 I.072 J5.97 E.15502
G3 X139.313 Y118.393 I-11.975 J11.946 E.14481
G1 X139.784 Y118.851 E.0251
G2 X141.945 Y119.121 I1.388 J-2.334 E.08555
G1 X141.945 Y110.597 E.32544
G1 X141.67 Y110.394 E.01307
G3 X139.313 Y107.432 I11.976 J-11.946 E.14481
G2 X137.899 Y106.608 I-1.776 J1.422 E.06392
G2 X134.129 Y107.891 I-.072 J5.97 E.15502
G2 X131.772 Y110.853 I11.976 J11.946 E.14481
G3 X130.83 Y111.567 I-1.75 J-1.329 E.04573
G3 X128.283 Y111.333 I-.907 J-4.094 E.09924
G2 X129.379 Y109.263 I-56.3 J-31.149 E.08943
; CHANGE_LAYER
; Z_HEIGHT: 5.32
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X128.911 Y110.147 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L33
M991 S0 P32 ;notify layer change


G17
G3 Z5.56 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z5.32
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00922
G1 X122.328 Y123.268 E.20115
G1 X121.921 Y123.634 E.02089
G1 X121.469 Y123.859 E.01926
G1 X120.934 Y123.963 E.02084
G1 X120.385 Y123.913 E.02102
G1 X119.922 Y123.739 E.01887
G3 X118.918 Y122.954 I5.678 J-8.297 E.0487
G3 X114.848 Y126.662 I-39.087 J-38.817 E.21031
G1 X118.015 Y129.012 E.15056
G1 X118.485 Y129.402 E.02332
G1 X119.156 Y130.104 E.03707
G1 X119.749 Y130.931 E.03885
G1 X120.215 Y131.839 E.03895
G1 X120.508 Y132.676 E.03386
G1 X120.686 Y133.494 E.03195
G1 X120.765 Y134.489 E.03813
G1 X120.698 Y135.507 E.03894
G1 X120.538 Y136.268 E.02969
G3 X130.504 Y142.099 I-21.93 J48.912 E.44167
G3 X142.443 Y154.069 I-32.713 J44.568 E.64803
G1 X142.443 Y104.62 E1.88792
G1 X142.331 Y104.684 E.00494
G1 X141.729 Y104.779 E.02324
G1 X132.862 Y104.779 E.33856
G3 X132.199 Y104.663 I0 J-1.95 E.02583
G1 X131.599 Y104.315 E.02647
G1 X131.217 Y103.863 E.02259
G3 X124.807 Y115.927 I-50.987 J-19.352 E.52295
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.091 Y117.436 E.02015
G1 X126.168 Y117.978 E.02088
G1 X126.11 Y118.451 E.0182
G1 X125.95 Y118.874 E.01727
G1 X125.896 Y118.955 E.0037
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.324 Y118.777 E.00547
G3 X124.449 Y119.826 I-15.06 J-11.668 E.05218
G1 X121.879 Y122.892 E.15272
G1 X121.594 Y123.148 E.01461
G1 X121.187 Y123.333 E.01708
G1 X120.736 Y123.376 E.0173
G1 X120.383 Y123.303 E.01375
G1 X119.957 Y123.061 E.01871
G1 X118.881 Y122.159 E.05363
G3 X113.893 Y126.683 I-38.408 J-37.331 E.25728
G1 X117.666 Y129.482 E.17937
G1 X118.108 Y129.85 E.02195
G1 X118.727 Y130.503 E.03433
G1 X119.266 Y131.263 E.03558
G1 X119.689 Y132.096 E.03566
G1 X119.951 Y132.858 E.03077
G1 X120.112 Y133.608 E.02932
G1 X120.18 Y134.525 E.03508
G1 X120.114 Y135.456 E.03565
G1 X119.931 Y136.289 E.03256
G1 X119.836 Y136.599 E.01238
G3 X130.933 Y143.157 I-21.275 J48.667 E.49335
G3 X142.92 Y155.769 I-32.176 J42.587 E.66729
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y118.038 E1.43974
G3 X143.042 Y103.311 I1563.944 J-6 E.56227
; LINE_WIDTH: 0.594326
G1 F14062.752
G1 X143.051 Y103.061 E.00916
G1 X142.995 Y103.303 E.00908
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.627 Y103.856 E.02536
G1 X142.15 Y104.127 E.02094
G1 X141.729 Y104.193 E.01626
G1 X132.862 Y104.193 E.33856
G1 X132.398 Y104.112 E.01798
G1 X131.956 Y103.849 E.01962
G1 F13287.966
G1 X131.689 Y103.526 E.016
; LINE_WIDTH: 0.666216
G1 F11833.104
G1 X131.491 Y103.045 E.02143
; LINE_WIDTH: 0.714471
G1 F10144.336
G1 X131.468 Y102.984 E.00287
; LINE_WIDTH: 0.762726
G1 F9943.071
G1 X131.444 Y102.924 E.00308
; LINE_WIDTH: 0.810981
G1 F9743.833
G1 X131.421 Y102.863 E.00328
; LINE_WIDTH: 0.859236
G1 F9546.612
G1 X131.398 Y102.803 E.00349
; LINE_WIDTH: 0.907491
G1 F9019.018
G1 X131.374 Y102.742 E.00369
; LINE_WIDTH: 0.955746
G1 F8546.684
G1 X131.351 Y102.682 E.00389
; LINE_WIDTH: 1.004
G1 F8121.362
G1 X131.327 Y102.622 E.0041
; LINE_WIDTH: 1.05226
G1 F7736.364
G1 X131.304 Y102.561 E.0043
G1 X131.259 Y102.615 E.00466
; LINE_WIDTH: 1.004
G1 F8121.362
G1 X131.214 Y102.669 E.00444
; LINE_WIDTH: 0.955746
G1 F8546.684
G1 X131.169 Y102.723 E.00422
; LINE_WIDTH: 0.907491
G1 F9019.018
G1 X131.124 Y102.777 E.004
; LINE_WIDTH: 0.859236
G1 F9546.612
G1 X131.079 Y102.831 E.00378
; LINE_WIDTH: 0.810981
G1 F10139.768
G1 X131.034 Y102.885 E.00356
; LINE_WIDTH: 0.762726
G1 F10360.251
G1 X130.989 Y102.939 E.00334
; LINE_WIDTH: 0.714471
G1 F10583.081
G1 X130.944 Y102.993 E.00312
; LINE_WIDTH: 0.666216
G1 F11896.456
G1 X130.79 Y103.362 E.01648
G1 F12462.81
G1 X130.685 Y103.614 E.01123
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X130.538 Y103.986 E.01527
G1 X130.079 Y105.145 E.04762
G3 X124.018 Y116.03 I-49.701 J-20.547 E.47671
G1 X125.094 Y116.932 E.05365
G1 X125.394 Y117.286 E.01771
G1 X125.529 Y117.599 E.01299
G1 X125.582 Y117.986 E.01491
G1 X125.512 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.168 Y122.718 E.01264
G1 X120.874 Y122.793 E.01161
G1 X120.577 Y122.751 E.01145
G1 X120.334 Y122.612 E.01068
G1 X118.837 Y121.358 E.07455
G3 X112.93 Y126.698 I-38.426 J-36.573 E.30431
G1 X117.317 Y129.953 E.20857
G1 X117.73 Y130.298 E.02057
G1 X118.297 Y130.901 E.03159
G1 X118.783 Y131.594 E.0323
G3 X119.537 Y133.723 I-5.618 J3.189 E.08669
G1 X119.596 Y134.56 E.03204
G1 X119.531 Y135.406 E.03237
G1 X119.361 Y136.154 E.02928
G1 X119.08 Y136.91 E.03082
G3 X130.579 Y143.624 I-20.746 J48.738 E.50973
G3 X142.648 Y156.415 I-31.78 J42.073 E.67455
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.527 E2.16507
G1 X142.966 Y99.598 E.02491
G1 X141.924 Y99.605 E.03978
G1 X142.496 Y102.694 E.11994
G3 X142.458 Y103.102 I-.767 J.135 E.01581
G1 X142.242 Y103.415 E.01454
G1 X141.969 Y103.569 E.01195
G1 X141.729 Y103.607 E.00928
G1 X132.862 Y103.607 E.33856
G1 X132.597 Y103.561 E.01026
G1 X132.345 Y103.411 E.0112
G1 X132.192 Y103.227 E.00913
G1 X132.093 Y102.95 E.01123
G1 X132.119 Y102.566 E.0147
G1 X132.667 Y99.604 E.11499
G3 X131.468 Y99.49 I.204 J-8.503 E.04602
G1 X131.308 Y99.983 E.01978
G3 X123.222 Y116.127 I-51.397 J-15.645 E.69265
G1 X124.718 Y117.381 E.07453
G1 X124.934 Y117.671 E.01379
G1 X124.997 Y117.982 E.01212
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.007 Y122.161 E.16681
G1 X120.845 Y122.241 E.00571
G1 X120.689 Y122.189 E.00523
G1 X119.678 Y121.342 E.04174
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02697
; LINE_WIDTH: 0.531626
G1 X118.977 Y120.758 E.00202
; LINE_WIDTH: 0.556176
G1 X118.953 Y120.481 E.00946
G1 X117.999 Y121.433 E.04584
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-40.128 J-39.562 E.25281
G1 X116.988 Y130.397 E.19623
G1 X117.374 Y130.721 E.01598
G1 X117.892 Y131.277 E.02405
G1 X118.327 Y131.906 E.02422
G3 X118.995 Y133.832 I-5.096 J2.847 E.06485
G1 X119.044 Y134.594 E.02419
G1 X118.98 Y135.358 E.02427
G1 X118.824 Y136.026 E.02171
G1 X118.548 Y136.753 E.02462
G1 X118.31 Y137.19 E.01576
G3 X130.249 Y144.068 I-19.955 J48.443 E.4375
G3 X142.39 Y157.025 I-31.646 J41.817 E.5649
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.405 E1.81381
G1 X144.167 Y98.945 E.01457
G1 X142.906 Y99.049 E.04009
G3 X141.26 Y99.054 I-.941 J-39.026 E.05211
G1 X141.952 Y102.79 E.12029
G1 X141.941 Y102.908 E.00376
G1 X141.799 Y103.044 E.00621
G1 X132.862 Y103.055 E.28296
G1 X132.712 Y102.998 E.00507
G1 X132.639 Y102.864 E.00482
G3 X132.967 Y101.021 I83.785 J13.97 E.05927
G1 X133.331 Y99.054 E.06332
G3 X132.125 Y99.019 I-.293 J-10.577 E.03823
G2 X131.038 Y98.936 I-.887 J4.478 E.0346
G3 X122.648 Y115.977 I-50.882 J-14.466 E.60463
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00556
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.691 Y113.978 Z5.72 F36000
G1 X143.051 Y103.061 Z5.72
G1 Z5.32
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.577916
G1 F14487.289
G1 X143.05 Y102.584 E.01688
; LINE_WIDTH: 0.618606
G1 F13478.358
G1 X143.03 Y102.363 E.00848
; LINE_WIDTH: 0.664322
G1 F12500.274
G1 X143.007 Y102.114 E.01027
; LINE_WIDTH: 0.710039
G1 F11654.538
G1 X142.984 Y101.865 E.01101
; LINE_WIDTH: 0.755755
G1 F10915.991
G1 X142.961 Y101.616 E.01176
; LINE_WIDTH: 0.801471
G1 F10265.469
G1 X142.938 Y101.367 E.0125
; LINE_WIDTH: 0.847188
G1 F9688.12
G1 X142.915 Y101.118 E.01325
; LINE_WIDTH: 0.892904
G1 F9172.255
G1 X142.892 Y100.869 E.01399
; LINE_WIDTH: 0.93862
G1 F8708.55
G1 X142.87 Y100.62 E.01474
; LINE_WIDTH: 0.984336
G1 F8289.474
G1 X142.847 Y100.371 E.01548
; WIPE_START
G1 X142.87 Y100.62 E-.095
G1 X142.892 Y100.869 E-.095
G1 X142.915 Y101.118 E-.095
G1 X142.938 Y101.367 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.349 Y100.558 Z5.72 F36000
G1 X131.91 Y100.191 Z5.72
G1 Z5.32
G1 E.4 F1800
; LINE_WIDTH: 0.722876
G1 F11437.245
G1 X131.91 Y100.193 E.00005
; LINE_WIDTH: 0.763031
G1 F10806.991
G1 X131.831 Y100.507 E.01539
; LINE_WIDTH: 0.803186
G1 F10242.571
G1 X131.753 Y100.821 E.01624
; LINE_WIDTH: 0.843341
G1 F9734.181
G1 X131.674 Y101.136 E.01709
; LINE_WIDTH: 0.883496
G1 F9273.872
G1 X131.596 Y101.45 E.01794
; LINE_WIDTH: 0.932043
G1 F8772.354
G1 X131.511 Y101.774 E.0196
; LINE_WIDTH: 0.98059
G1 F8322.295
G1 X131.426 Y102.098 E.02066
; LINE_WIDTH: 1.02914
G1 F7916.164
G1 X131.342 Y102.422 E.02172
; LINE_WIDTH: 1.05226
G1 F7736.364
G1 X131.304 Y102.561 E.00957
; WIPE_START
G1 X131.342 Y102.422 E-.05478
G1 X131.426 Y102.098 E-.12726
G1 X131.511 Y101.774 E-.12726
G1 X131.558 Y101.594 E-.07071
; WIPE_END
G1 E-.02 F1800
G1 X127.321 Y107.942 Z5.72 F36000
G1 X118.953 Y120.481 Z5.72
G1 Z5.32
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556176
G1 F3600
M204 S5000
G1 X119.2 Y120.199 E.01276
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.687 E.15232
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.687 E-.13081
G1 X121.722 Y117.19 E-.24919
; WIPE_END
G1 E-.02 F1800
G1 X119.198 Y123.838 Z5.72 F36000
G1 Z5.32
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X118.947 Y123.628 E.0125
G3 X117.49 Y125.021 I-29.567 J-29.461 E.07694
G3 X116.22 Y126.72 I-5.744 J-2.969 E.08139
G1 X115.931 Y126.846 E.01203
G3 X118.901 Y129.113 I-15.641 J23.578 E.14277
G3 X123.289 Y132.321 I-1.125 J6.144 E.2147
G2 X124.703 Y134.261 I6.382 J-3.165 E.0921
G2 X126.117 Y134.488 I1.047 J-2.003 E.05563
G2 X130.83 Y131.208 I-.868 J-6.273 E.22753
G3 X132.244 Y129.268 I6.382 J3.165 E.0921
G3 X133.658 Y129.04 I1.047 J2.003 E.05563
G3 X138.371 Y132.321 I-.868 J6.273 E.22753
G2 X139.784 Y134.261 I6.382 J-3.165 E.0921
G2 X141.198 Y134.488 I1.047 J-2.003 E.05563
G1 X141.945 Y134.322 E.02922
G1 X141.945 Y140.753 E.24551
G3 X139.784 Y137.958 I5.679 J-6.625 E.13581
G2 X138.371 Y136.604 I-2.437 J1.129 E.07652
G2 X135.543 Y137.152 I-.415 J5.425 E.1113
G2 X132.715 Y139.861 I2.868 J5.824 E.15187
G3 X131.301 Y141.802 I-6.382 J-3.165 E.0921
G1 X131.063 Y141.905 E.0099
G3 X133.942 Y144.185 I-76.07 J99.005 E.14022
G3 X138.371 Y147.402 I-1.091 J6.158 E.21629
G2 X139.784 Y149.342 I6.382 J-3.165 E.0921
G2 X141.198 Y149.57 I1.047 J-2.003 E.05563
G1 X141.945 Y149.404 E.02922
G1 X141.945 Y147.061 E.08944
G1 X126.558 Y138.862 F36000
G1 F13446.283
G2 X124.526 Y137.696 I-18.342 J29.591 E.08945
G2 X123.289 Y136.604 I-2.111 J1.146 E.0643
G2 X122.37 Y136.58 I-.527 J2.516 E.03531
G1 X121.111 Y135.983 E.05319
G2 X121.238 Y135.044 I-4.385 J-1.07 E.03627
G1 X141.945 Y128.014 F36000
G1 F13446.283
G1 X141.945 Y125.672 E.08944
G3 X141.198 Y124.978 I2.69 J-3.649 E.03899
G3 X139.313 Y122.176 I37.928 J-27.55 E.12897
G2 X137.899 Y121.469 I-1.449 J1.13 E.0624
M73 P65 R6
G2 X133.658 Y123.469 I.325 J6.187 E.18363
G2 X131.772 Y126.271 I37.928 J27.55 E.12897
G3 X130.359 Y126.979 I-1.449 J-1.13 E.0624
G3 X125.174 Y123.668 I.391 J-6.326 E.24522
G2 X124.042 Y121.995 I-7.472 J3.84 E.07731
G1 X126.202 Y119.412 E.12858
G2 X129.887 Y117.438 I-1.219 J-6.701 E.16222
G2 X131.772 Y114.635 I-37.928 J-27.55 E.12897
G3 X133.186 Y113.928 I1.449 J1.13 E.0624
G3 X136.485 Y115.079 I-.376 J6.381 E.13512
G3 X138.842 Y118.03 I-5.382 J6.715 E.14535
G2 X140.256 Y119.384 I2.437 J-1.129 E.07652
G2 X141.945 Y119.241 I.549 J-3.569 E.06535
G1 X141.945 Y110.59 E.33028
G3 X141.198 Y109.897 I2.691 J-3.649 E.03899
G3 X139.313 Y107.095 I37.928 J-27.55 E.12897
G2 X137.899 Y106.387 I-1.449 J1.13 E.0624
G2 X134.6 Y107.539 I.376 J6.381 E.13512
G2 X132.244 Y110.489 I5.382 J6.715 E.14535
G3 X130.83 Y111.844 I-2.437 J-1.129 E.07652
G3 X128.252 Y111.391 I-.464 J-4.923 E.10115
G2 X129.352 Y109.323 I-48.469 J-27.137 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 5.48
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X128.882 Y110.206 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L34
M991 S0 P33 ;notify layer change


G17
G3 Z5.72 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z5.48
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00922
G1 X124.897 Y120.203 E.04844
G1 X122.328 Y123.268 E.15272
G1 X121.942 Y123.62 E.01993
G1 X121.522 Y123.84 E.01811
G1 X121.016 Y123.957 E.01981
G1 X120.455 Y123.928 E.02145
G1 X120.071 Y123.81 E.01536
G1 X119.581 Y123.51 E.02192
G1 X118.918 Y122.954 E.03305
G3 X114.848 Y126.662 I-38.7 J-38.382 E.2103
G1 X118.015 Y129.011 E.15054
G3 X118.933 Y129.85 I-4.613 J5.975 E.04752
G1 X119.565 Y130.646 E.03882
G1 X120.076 Y131.531 E.03903
G1 X120.45 Y132.48 E.03893
G3 X120.718 Y133.725 I-10.199 J2.841 E.04865
G1 X120.765 Y134.489 E.02921
G1 X120.698 Y135.506 E.03893
G1 X120.538 Y136.268 E.02972
G3 X130.504 Y142.099 I-22.55 J49.975 E.44164
G3 X142.443 Y154.069 I-32.641 J44.496 E.64802
G1 X142.443 Y104.623 E1.88779
G1 X142.333 Y104.686 E.00483
G1 X141.732 Y104.781 E.02324
G1 X132.859 Y104.781 E.33875
G1 X132.557 Y104.757 E.01155
G1 X132.026 Y104.594 E.02124
G1 X131.58 Y104.303 E.02032
G1 X131.217 Y103.873 E.02149
G3 X124.807 Y115.927 I-51.215 J-19.499 E.5226
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.091 Y117.436 E.02015
G1 X126.168 Y117.978 E.02088
G1 X126.11 Y118.451 E.0182
G1 X125.95 Y118.874 E.01727
G1 X125.896 Y118.955 E.00369
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.324 Y118.777 E.00547
G3 X124.449 Y119.826 I-15.045 J-11.656 E.05218
G1 X121.879 Y122.892 E.15272
G1 X121.609 Y123.138 E.01394
G1 X121.315 Y123.292 E.01267
G1 X120.961 Y123.374 E.01386
G1 X120.52 Y123.343 E.0169
G1 X120.3 Y123.271 E.00884
G1 X119.845 Y122.967 E.0209
G1 X118.881 Y122.159 E.04802
G3 X113.893 Y126.683 I-39.146 J-38.144 E.25727
G1 X117.666 Y129.482 E.17935
G3 X118.524 Y130.269 I-4.176 J5.415 E.0445
G1 X119.099 Y131.001 E.03553
G1 X119.563 Y131.814 E.03575
G1 X119.901 Y132.684 E.03565
G3 X120.135 Y133.791 I-10.225 J2.739 E.04323
G1 X120.18 Y134.525 E.02806
G1 X120.114 Y135.456 E.03564
G1 X119.932 Y136.286 E.03245
G1 X119.835 Y136.599 E.0125
G3 X130.933 Y143.157 I-21.085 J48.346 E.49336
G3 X142.92 Y155.769 I-32.371 J42.773 E.66727
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y116.039 E1.51606
G3 X143.042 Y103.313 I1278.461 J-5 E.48588
; LINE_WIDTH: 0.592856
G1 F14099.766
G1 X143.052 Y103.063 E.00912
G1 X142.997 Y103.305 E.00905
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.63 Y103.858 E.02535
G1 X142.152 Y104.129 E.02094
G1 X141.732 Y104.195 E.01626
G1 X132.859 Y104.195 E.33875
G1 X132.648 Y104.179 E.00808
G1 X132.25 Y104.051 E.01597
G1 X131.964 Y103.861 E.01311
G1 F12715.395
G1 X131.68 Y103.517 E.01702
; LINE_WIDTH: 0.668503
G1 F11205.617
G1 X131.625 Y103.394 E.00557
; LINE_WIDTH: 0.71701
G1 F10768.332
G1 X131.57 Y103.271 E.00599
; LINE_WIDTH: 0.765516
G1 F10339.735
G1 X131.515 Y103.148 E.00642
; LINE_WIDTH: 0.814023
G1 F9919.84
G1 X131.46 Y103.025 E.00684
; LINE_WIDTH: 0.86253
G1 F9508.65
G1 X131.405 Y102.902 E.00727
; LINE_WIDTH: 0.911036
G1 F8982.549
G1 X131.35 Y102.78 E.0077
; LINE_WIDTH: 0.959103
G1 F8515.663
G1 X131.332 Y102.714 E.00411
; LINE_WIDTH: 1.00717
G1 F8094.912
G1 X131.315 Y102.648 E.00432
; LINE_WIDTH: 1.05524
G1 F7713.781
G1 X131.297 Y102.582 E.00454
G1 X131.253 Y102.636 E.00462
; LINE_WIDTH: 1.00717
G1 F8094.912
G1 X131.209 Y102.69 E.0044
; LINE_WIDTH: 0.959103
G1 F8515.663
G1 X131.165 Y102.743 E.00419
; LINE_WIDTH: 0.911036
G1 F8982.549
G1 X131.086 Y102.887 E.00934
; LINE_WIDTH: 0.86253
G1 F9508.65
G1 X131.008 Y103.03 E.00883
; LINE_WIDTH: 0.814023
G1 F10100.212
G1 X130.929 Y103.173 E.00831
; LINE_WIDTH: 0.765516
G1 F10615.764
G1 X130.85 Y103.317 E.00779
; LINE_WIDTH: 0.71701
G1 F11144.131
G1 X130.772 Y103.46 E.00728
; LINE_WIDTH: 0.668503
G1 F11685.331
G1 X130.693 Y103.603 E.00676
; LINE_WIDTH: 0.619996
G1 F13063.457
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.209 Y104.835 E.01998
G3 X124.018 Y116.03 I-50.029 J-20.36 E.48957
G1 X125.094 Y116.932 E.05365
G1 X125.394 Y117.287 E.01771
G1 X125.529 Y117.599 E.01299
G1 X125.582 Y117.986 E.01491
G1 X125.512 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.165 Y122.72 E.0128
G1 X120.907 Y122.791 E.01021
G1 X120.577 Y122.751 E.01267
G1 X120.221 Y122.518 E.01624
G1 X118.837 Y121.358 E.06894
G3 X112.93 Y126.698 I-38.904 J-37.101 E.30431
G1 X117.317 Y129.952 E.20855
G1 X118.115 Y130.688 E.04142
G1 X118.632 Y131.355 E.03225
G1 X119.05 Y132.096 E.03246
G1 X119.352 Y132.887 E.03236
G3 X119.551 Y133.827 I-12.796 J3.194 E.03666
G1 X119.596 Y134.56 E.02806
G1 X119.531 Y135.405 E.03235
G1 X119.362 Y136.15 E.02917
G1 X119.08 Y136.91 E.03094
G3 X130.579 Y143.624 I-20.878 J48.963 E.50971
G3 X142.648 Y156.415 I-31.962 J42.244 E.67452
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.528 E2.16504
G1 X142.966 Y99.6 E.02492
G1 X141.927 Y99.607 E.03969
G1 X142.499 Y102.696 E.11994
G3 X142.461 Y103.104 I-.766 J.135 E.01581
G1 X142.244 Y103.417 E.01454
G1 X141.972 Y103.572 E.01195
G1 X141.732 Y103.609 E.00928
G1 X132.859 Y103.609 E.33875
G1 X132.511 Y103.527 E.01364
G3 X132.186 Y103.223 I.348 J-.697 E.01726
G1 X132.082 Y102.788 E.01707
G1 X132.094 Y102.689 E.00379
G1 X132.665 Y99.606 E.11971
G3 X131.466 Y99.49 I.206 J-8.369 E.046
G3 X123.222 Y116.127 I-51.856 J-15.334 E.71243
G1 X124.718 Y117.381 E.07453
G1 X124.934 Y117.671 E.01379
G1 X124.997 Y117.982 E.01212
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.003 Y122.165 E.16698
G1 X120.855 Y122.24 E.00527
G1 X120.689 Y122.189 E.00551
G1 X119.678 Y121.341 E.04176
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02696
; LINE_WIDTH: 0.531586
G1 X118.977 Y120.758 E.00201
; LINE_WIDTH: 0.556176
G1 X118.953 Y120.481 E.00947
G1 X118 Y121.433 E.04582
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.721 J-39.099 E.25283
G1 X116.987 Y130.396 E.19622
G1 X117.728 Y131.083 E.03199
G1 X118.192 Y131.69 E.02417
G1 X118.565 Y132.362 E.02435
G1 X118.834 Y133.08 E.02426
G1 X118.981 Y133.764 E.02216
G1 X119.044 Y134.594 E.02635
G1 X118.98 Y135.357 E.02426
G1 X118.824 Y136.023 E.02163
G1 X118.548 Y136.753 E.02472
G1 X118.31 Y137.19 E.01576
G3 X130.249 Y144.067 I-19.556 J47.75 E.43754
G3 X142.39 Y157.025 I-31.646 J41.817 E.5649
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.403 E1.81386
G1 X144.167 Y98.945 E.01452
G1 X142.905 Y99.051 E.04012
G3 X141.262 Y99.056 I-.933 J-36.623 E.052
G1 X141.954 Y102.792 E.12029
G1 X141.943 Y102.91 E.00376
G1 X141.801 Y103.046 E.00621
G1 X141.732 Y103.057 E.00223
G1 X132.859 Y103.057 E.28091
G1 X132.711 Y103.001 E.005
G1 X132.634 Y102.818 E.00629
G1 X132.637 Y102.79 E.00091
G1 X133.329 Y99.057 E.12021
G3 X132.124 Y99.02 I-.291 J-10.384 E.03816
G2 X131.038 Y98.936 I-.888 J4.405 E.03459
G3 X122.648 Y115.977 I-51.094 J-14.57 E.6046
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00556
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.692 Y113.979 Z5.88 F36000
G1 X143.052 Y103.063 Z5.88
G1 Z5.48
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.575396
G1 F14554.764
G1 X143.051 Y102.587 E.01679
; LINE_WIDTH: 0.616346
G1 F13530.697
G1 X143.031 Y102.364 E.0085
; LINE_WIDTH: 0.66206
G1 F12545.328
G1 X143.008 Y102.115 E.01023
; LINE_WIDTH: 0.707774
G1 F11693.736
G1 X142.985 Y101.866 E.01098
; LINE_WIDTH: 0.753488
G1 F10950.409
G1 X142.962 Y101.617 E.01172
; LINE_WIDTH: 0.799201
G1 F10295.935
G1 X142.939 Y101.368 E.01247
; LINE_WIDTH: 0.844915
G1 F9715.282
G1 X142.916 Y101.119 E.01321
; LINE_WIDTH: 0.890629
G1 F9196.625
G1 X142.894 Y100.87 E.01396
; LINE_WIDTH: 0.936342
G1 F8730.539
G1 X142.871 Y100.621 E.0147
; LINE_WIDTH: 0.982056
G1 F8309.417
G1 X142.848 Y100.372 E.01545
; WIPE_START
G1 X142.871 Y100.621 E-.095
G1 X142.894 Y100.87 E-.095
G1 X142.916 Y101.119 E-.095
G1 X142.939 Y101.368 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.348 Y102.16 Z5.88 F36000
G1 X131.297 Y102.582 Z5.88
G1 Z5.48
G1 E.4 F1800
; LINE_WIDTH: 1.05524
G1 F7713.781
G1 X131.307 Y102.544 E.00265
; LINE_WIDTH: 1.04878
G1 F7762.904
G1 X131.377 Y102.279 E.01809
; LINE_WIDTH: 1.00818
G1 F8086.503
G1 X131.446 Y102.014 E.01737
; LINE_WIDTH: 0.967586
G1 F8438.254
G1 X131.516 Y101.75 E.01664
; LINE_WIDTH: 0.926991
G1 F8821.998
G1 X131.585 Y101.485 E.01592
; LINE_WIDTH: 0.886396
G1 F9242.307
G1 X131.595 Y101.45 E.00204
; LINE_WIDTH: 0.881396
G1 F9296.863
G1 X131.673 Y101.135 E.01789
; LINE_WIDTH: 0.841256
G1 F9759.332
G1 X131.752 Y100.821 E.01704
; LINE_WIDTH: 0.801116
G1 F10270.222
G1 X131.83 Y100.507 E.01619
; LINE_WIDTH: 0.760976
G1 F10837.554
G1 X131.909 Y100.193 E.01535
; LINE_WIDTH: 0.720836
G1 F11471.233
G1 X131.909 Y100.192 E.00004
; WIPE_START
G1 X131.909 Y100.193 E-.00033
G1 X131.83 Y100.507 E-.12311
G1 X131.752 Y100.821 E-.12311
G1 X131.673 Y101.135 E-.12311
G1 X131.667 Y101.162 E-.01036
; WIPE_END
G1 E-.02 F1800
G1 X127.471 Y107.537 Z5.88 F36000
G1 X118.953 Y120.481 Z5.88
G1 Z5.48
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556176
G1 F3600
M204 S5000
G1 X119.2 Y120.198 E.01277
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15231
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.13088
G1 X121.722 Y117.19 E-.24912
; WIPE_END
G1 E-.02 F1800
G1 X119.517 Y124.051 Z5.88 F36000
G1 Z5.48
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X118.945 Y123.627 I1.139 J-2.131 E.02727
G3 X117.774 Y124.759 I-22.576 J-22.184 E.06221
G3 X116.691 Y126.725 I-46.962 J-24.579 E.08571
G1 X116.268 Y127.096 E.02148
G3 X118.628 Y128.866 I-21.079 J30.554 E.11267
G3 X123.289 Y132.542 I-1.314 J6.459 E.23513
G2 X124.232 Y134.266 I14.453 J-6.785 E.07505
G1 X124.703 Y134.678 E.0239
G1 X125.174 Y134.79 E.01849
G2 X130.83 Y130.986 I-.382 J-6.675 E.27304
G3 X131.772 Y129.263 I14.453 J6.785 E.07505
G1 X132.244 Y128.851 E.0239
G1 X132.715 Y128.739 E.01849
G3 X138.371 Y132.542 I-.382 J6.675 E.27304
G2 X139.313 Y134.266 I14.451 J-6.784 E.07505
G1 X139.784 Y134.678 E.0239
G1 X140.256 Y134.79 E.01849
G2 X141.945 Y134.453 I-.107 J-4.941 E.06612
G1 X141.945 Y140.752 E.24052
G3 X139.784 Y137.602 I5.966 J-6.409 E.14702
G1 X139.313 Y136.804 E.03541
G1 X138.842 Y136.391 E.0239
G1 X138.371 Y136.28 E.01849
G2 X132.715 Y140.083 I.382 J6.675 E.27304
G3 X131.772 Y141.807 I-14.453 J-6.785 E.07505
G1 X131.386 Y142.144 E.01959
G2 X133.658 Y143.932 I232.131 J-292.623 E.11036
G3 X138.371 Y147.624 I-1.28 J6.488 E.23719
G2 X139.313 Y149.347 I14.452 J-6.784 E.07505
G1 X139.784 Y149.759 E.0239
G1 X140.256 Y149.871 E.01849
G2 X141.945 Y149.534 I-.107 J-4.941 E.06612
G1 X141.945 Y147.191 E.08944
G1 X126.86 Y139.045 F36000
G1 F13446.283
G2 X124.837 Y137.864 I-20.394 J32.625 E.08944
G2 X124.232 Y136.804 I-8.909 J4.376 E.04665
G1 X123.761 Y136.391 E.0239
G1 X123.289 Y136.28 E.01849
G2 X122.117 Y136.461 I-.075 J3.401 E.04551
G1 X121.115 Y135.986 E.04232
G2 X121.252 Y134.762 I-5.349 J-1.217 E.04714
G1 X141.945 Y128.014 F36000
G1 F13446.283
G1 X141.945 Y125.671 E.08944
G3 X140.727 Y124.262 I4.337 J-4.982 E.07136
G2 X139.313 Y121.722 I-219.018 J120.258 E.11098
G1 X138.842 Y121.31 E.0239
G1 X138.371 Y121.198 E.01849
G2 X133.186 Y124.185 I.43 J6.739 E.23665
G3 X131.772 Y126.725 I-218.767 J-120.118 E.11098
G1 X131.301 Y127.137 E.0239
G1 X130.83 Y127.249 E.01849
G3 X125.646 Y124.262 I.43 J-6.739 E.23665
G2 X124.247 Y121.749 I-235.505 J129.372 E.10981
G1 X126.04 Y119.606 E.10666
G2 X130.359 Y116.721 I-1.339 J-6.68 E.20342
G3 X131.772 Y114.182 I218.318 J119.868 E.11098
G1 X132.244 Y113.769 E.0239
G1 X132.715 Y113.658 E.01849
G3 X137.899 Y116.645 I-.43 J6.739 E.23665
G2 X139.313 Y119.185 I218.216 J-119.813 E.11098
G1 X139.784 Y119.597 E.0239
G1 X140.256 Y119.708 E.01849
G2 X141.945 Y119.371 I-.107 J-4.942 E.06612
G1 X141.945 Y110.59 E.33528
G3 X140.727 Y109.181 I4.337 J-4.982 E.07136
G2 X139.313 Y106.641 I-218.891 J120.187 E.11098
G1 X138.842 Y106.229 E.0239
G1 X138.371 Y106.117 E.01849
G2 X133.186 Y109.104 I.43 J6.739 E.23665
G3 X131.772 Y111.644 I-218.891 J-120.187 E.11098
G1 X131.301 Y112.056 E.0239
G1 X130.83 Y112.168 E.01849
G3 X128.218 Y111.452 I.553 J-7.147 E.10403
G2 X129.32 Y109.386 I-46.081 J-25.903 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 5.64
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X128.849 Y110.268 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L35
M991 S0 P34 ;notify layer change


G17
G3 Z5.88 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z5.64
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00922
G1 X122.328 Y123.268 E.20115
G1 X121.942 Y123.62 E.01995
G1 X121.522 Y123.84 E.01809
G1 X121.016 Y123.957 E.01981
G1 X120.455 Y123.928 E.02145
G1 X119.957 Y123.758 E.02009
G3 X118.918 Y122.954 I5.048 J-7.605 E.05019
G3 X114.848 Y126.662 I-39.062 J-38.79 E.21031
G1 X118.015 Y129.012 E.15055
G3 X118.933 Y129.85 I-4.601 J5.963 E.0475
G1 X119.565 Y130.646 E.03884
G1 X120.075 Y131.53 E.03895
G1 X120.45 Y132.48 E.03899
G3 X120.765 Y134.489 I-7.454 J2.195 E.07787
G1 X120.698 Y135.504 E.03885
G1 X120.538 Y136.268 E.0298
G3 X130.509 Y142.102 I-21.909 J48.881 E.44189
G3 X142.443 Y154.069 I-32.642 J44.488 E.64782
G1 X142.443 Y104.627 E1.88767
G1 X142.336 Y104.688 E.00472
G1 X141.734 Y104.783 E.02324
G1 X132.857 Y104.783 E.33895
G3 X132.207 Y104.671 I0 J-1.95 E.02529
G1 X131.779 Y104.458 E.01823
G1 X131.217 Y103.872 E.03101
G3 X124.807 Y115.927 I-51.195 J-19.488 E.52263
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.091 Y117.436 E.02015
G1 X126.168 Y117.978 E.02088
G1 X126.11 Y118.451 E.0182
G1 X125.95 Y118.874 E.01727
G1 X125.896 Y118.955 E.0037
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.324 Y118.777 E.00547
G3 X124.449 Y119.826 I-15.06 J-11.668 E.05218
G1 X121.879 Y122.892 E.15272
G1 X121.609 Y123.138 E.01395
G1 X121.315 Y123.292 E.01265
G1 X120.961 Y123.374 E.01386
G1 X120.52 Y123.343 E.0169
G1 X120.196 Y123.222 E.0132
G3 X118.881 Y122.159 I9.149 J-12.666 E.0646
G3 X113.893 Y126.683 I-38.413 J-37.337 E.25728
G1 X117.666 Y129.482 E.17936
G3 X118.524 Y130.269 I-4.167 J5.405 E.04448
G1 X119.099 Y131.001 E.03556
G1 X119.562 Y131.812 E.03566
G1 X119.901 Y132.683 E.0357
G3 X120.135 Y133.791 I-9.611 J2.609 E.04325
G1 X120.18 Y134.525 E.02805
G1 X120.115 Y135.454 E.03557
G1 X119.933 Y136.283 E.03241
G1 X119.836 Y136.599 E.01262
G3 X130.932 Y143.156 I-21.323 J48.749 E.4933
G3 X142.92 Y155.769 I-32.175 J42.588 E.66734
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y116.04 E1.51602
G3 X143.043 Y103.315 I1211.137 J-5 E.48586
; LINE_WIDTH: 0.591356
G1 F14137.733
G1 X143.054 Y103.064 E.00909
G1 X142.998 Y103.307 E.00902
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.632 Y103.86 E.02535
G1 X142.155 Y104.131 E.02094
G1 X141.734 Y104.197 E.01626
G1 X132.857 Y104.197 E.33895
G1 X132.402 Y104.119 E.01761
G1 X132.103 Y103.97 E.01275
G1 F13124.421
G1 X131.948 Y103.806 E.00861
G1 F12335.903
G1 X131.675 Y103.514 E.01527
; LINE_WIDTH: 0.667547
G1 F10997.788
G1 X131.633 Y103.408 E.00471
; LINE_WIDTH: 0.715098
G1 F10629.91
G1 X131.592 Y103.301 E.00507
; LINE_WIDTH: 0.76265
G1 F10268.29
G1 X131.551 Y103.195 E.00542
; LINE_WIDTH: 0.810201
G1 F9912.928
G1 X131.509 Y103.088 E.00578
; LINE_WIDTH: 0.857752
G1 F9563.824
G1 X131.468 Y102.982 E.00613
; LINE_WIDTH: 0.905303
G1 F9041.679
G1 X131.427 Y102.875 E.00648
; LINE_WIDTH: 0.952854
G1 F8573.596
G1 X131.385 Y102.769 E.00684
; LINE_WIDTH: 1.00041
G1 F8151.592
G1 X131.344 Y102.663 E.00719
; LINE_WIDTH: 1.04796
G1 F7769.184
G1 X131.303 Y102.556 E.00755
G1 X131.235 Y102.672 E.00887
; LINE_WIDTH: 1.00041
G1 F8151.592
G1 X131.168 Y102.788 E.00845
; LINE_WIDTH: 0.952854
G1 F8573.596
G1 X131.1 Y102.904 E.00803
; LINE_WIDTH: 0.905303
G1 F9041.679
G1 X131.032 Y103.02 E.00762
; LINE_WIDTH: 0.857752
G1 F9563.824
G1 X130.965 Y103.136 E.0072
; LINE_WIDTH: 0.810201
G1 F10149.97
G1 X130.897 Y103.251 E.00679
; LINE_WIDTH: 0.76265
G1 F10573.079
G1 X130.83 Y103.367 E.00637
; LINE_WIDTH: 0.715098
G1 F11004.828
G1 X130.762 Y103.483 E.00595
; LINE_WIDTH: 0.667547
G1 F11445.216
G1 X130.694 Y103.599 E.00554
; LINE_WIDTH: 0.619996
G1 F12809.506
G1 X130.549 Y103.972 E.01527
G1 F13446.369
G1 X130.404 Y104.345 E.01527
G1 X130.331 Y104.532 E.00767
G3 X124.018 Y116.03 I-50.247 J-20.108 E.50204
G1 X125.094 Y116.932 E.05365
G1 X125.394 Y117.287 E.01771
G1 X125.529 Y117.599 E.01299
G1 X125.582 Y117.986 E.01491
G1 X125.512 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.165 Y122.72 E.0128
G1 X120.907 Y122.791 E.01021
G1 X120.577 Y122.751 E.01269
G1 X120.334 Y122.612 E.01068
G1 X118.837 Y121.358 E.07455
G3 X112.93 Y126.698 I-38.426 J-36.573 E.30431
G1 X117.317 Y129.952 E.20856
G1 X118.114 Y130.687 E.0414
G1 X118.633 Y131.355 E.03228
G1 X119.049 Y132.094 E.03238
G1 X119.352 Y132.887 E.03241
G3 X119.551 Y133.827 I-11.738 J2.971 E.03668
G1 X119.596 Y134.56 E.02805
G1 X119.531 Y135.403 E.03229
G1 X119.363 Y136.148 E.02913
G1 X119.08 Y136.91 E.03105
G3 X130.578 Y143.623 I-20.428 J48.192 E.50972
G3 X142.647 Y156.415 I-31.779 J42.074 E.67459
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.529 E2.16501
G1 X142.966 Y99.602 E.02493
G1 X141.929 Y99.609 E.03959
M73 P66 R6
G1 X142.501 Y102.699 E.11994
G3 X142.463 Y103.106 I-.767 J.135 E.01581
G1 X142.247 Y103.419 E.01454
G1 X141.974 Y103.574 E.01195
G1 X141.734 Y103.612 E.00928
G1 X132.857 Y103.612 E.33895
G3 X132.426 Y103.482 I0 J-.779 E.0174
G1 X132.182 Y103.222 E.01363
G1 X132.09 Y102.696 E.02036
G1 X132.662 Y99.608 E.11992
G3 X131.464 Y99.49 I.207 J-8.237 E.04599
G3 X123.222 Y116.127 I-51.738 J-15.271 E.71242
G1 X124.718 Y117.381 E.07453
G1 X124.934 Y117.671 E.01379
G1 X124.997 Y117.982 E.01212
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.003 Y122.165 E.16698
G1 X120.855 Y122.24 E.00527
G1 X120.689 Y122.189 E.00551
G1 X119.678 Y121.342 E.04175
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02697
; LINE_WIDTH: 0.531606
G1 X118.977 Y120.758 E.00202
; LINE_WIDTH: 0.556156
G1 X118.953 Y120.481 E.00946
G1 X117.999 Y121.433 E.04583
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-40.153 J-39.59 E.25281
G1 X116.988 Y130.396 E.19623
G1 X117.728 Y131.083 E.03197
G1 X118.193 Y131.69 E.0242
G1 X118.564 Y132.361 E.02429
G3 X118.98 Y135.361 I-5.125 J2.238 E.09711
G1 X118.825 Y136.02 E.02142
G1 X118.548 Y136.753 E.02482
G1 X118.31 Y137.19 E.01574
G3 X130.248 Y144.067 I-19.584 J47.797 E.4375
G3 X142.39 Y157.025 I-31.645 J41.818 E.56494
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.401 E1.81392
G1 X144.167 Y98.945 E.01445
G1 X142.903 Y99.053 E.04016
G3 X141.265 Y99.058 I-.925 J-33.886 E.05189
G1 X141.957 Y102.794 E.12029
G1 X141.946 Y102.912 E.00376
G1 X141.804 Y103.048 E.00621
G1 X132.857 Y103.059 E.28327
G3 X132.661 Y102.946 I0 J-.226 E.00749
G1 X132.652 Y102.698 E.00783
G1 X133.326 Y99.059 E.1172
G3 X132.124 Y99.022 I-.29 J-10.19 E.03809
G2 X131.038 Y98.936 I-.888 J4.334 E.03459
G3 X122.648 Y115.977 I-50.883 J-14.466 E.60463
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00556
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.692 Y113.979 Z6.04 F36000
G1 X143.054 Y103.064 Z6.04
G1 Z5.64
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.572856
G1 F14623.414
G1 X143.052 Y102.589 E.01669
; LINE_WIDTH: 0.614066
G1 F13583.911
G1 X143.032 Y102.365 E.00852
; LINE_WIDTH: 0.659783
G1 F12591.011
G1 X143.009 Y102.116 E.01019
; LINE_WIDTH: 0.705499
G1 F11733.374
G1 X142.986 Y101.867 E.01094
; LINE_WIDTH: 0.751215
G1 F10985.123
G1 X142.963 Y101.618 E.01168
; LINE_WIDTH: 0.796931
G1 F10326.583
G1 X142.94 Y101.369 E.01243
; LINE_WIDTH: 0.842647
G1 F9742.535
G1 X142.918 Y101.12 E.01317
; LINE_WIDTH: 0.888364
G1 F9221.016
G1 X142.895 Y100.871 E.01392
; LINE_WIDTH: 0.93408
G1 F8752.493
G1 X142.872 Y100.622 E.01466
; LINE_WIDTH: 0.979796
G1 F8329.279
G1 X142.849 Y100.373 E.01541
; WIPE_START
G1 X142.872 Y100.622 E-.095
G1 X142.895 Y100.871 E-.095
G1 X142.918 Y101.12 E-.095
G1 X142.94 Y101.369 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.347 Y102.143 Z6.04 F36000
G1 X131.303 Y102.556 Z6.04
G1 Z5.64
G1 E.4 F1800
; LINE_WIDTH: 1.04796
G1 F7769.184
G1 X131.34 Y102.421 E.00924
; LINE_WIDTH: 1.02496
G1 F7949.568
G1 X131.424 Y102.097 E.02162
; LINE_WIDTH: 0.976403
G1 F8359.282
G1 X131.509 Y101.773 E.02057
; LINE_WIDTH: 0.92785
G1 F8813.523
G1 X131.594 Y101.449 E.01951
; LINE_WIDTH: 0.879296
G1 F9319.969
G1 X131.672 Y101.135 E.01785
; LINE_WIDTH: 0.839146
G1 F9784.919
G1 X131.751 Y100.821 E.017
; LINE_WIDTH: 0.798996
G1 F10298.695
G1 X131.829 Y100.506 E.01615
; LINE_WIDTH: 0.758846
G1 F10869.416
G1 X131.908 Y100.192 E.0153
; WIPE_START
G1 X131.829 Y100.506 E-.12313
G1 X131.751 Y100.821 E-.12313
G1 X131.672 Y101.135 E-.12313
G1 X131.665 Y101.162 E-.01061
; WIPE_END
G1 E-.02 F1800
G1 X127.47 Y107.538 Z6.04 F36000
G1 X118.953 Y120.481 Z6.04
G1 Z5.64
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556156
G1 F3600
M204 S5000
G1 X119.2 Y120.199 E.01276
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15232
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.13087
G1 X121.722 Y117.19 E-.24913
; WIPE_END
G1 E-.02 F1800
G1 X119.766 Y124.213 Z6.04 F36000
G1 Z5.64
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X118.947 Y123.628 I2.185 J-3.922 E.03852
G3 X117.99 Y124.555 I-19.751 J-19.418 E.05088
G2 X116.753 Y127.456 I19.088 J9.852 E.1205
G1 X118.185 Y128.518 E.06807
G3 X121.875 Y130.569 I-2.546 J8.928 E.1626
G3 X123.289 Y132.819 I-6.254 J5.499 E.10191
G1 X124.232 Y135.15 E.096
G2 X125.174 Y135.188 I.503 J-.774 E.0378
G2 X129.416 Y132.96 I-2.315 J-9.561 E.18477
G2 X130.83 Y130.71 I-6.255 J-5.499 E.10191
G1 X131.772 Y128.379 E.096
G3 X132.715 Y128.341 I.503 J.774 E.0378
G3 X136.957 Y130.569 I-2.315 J9.561 E.18477
G3 X138.371 Y132.819 I-6.255 J5.499 E.10191
G1 X139.313 Y135.15 E.096
G2 X140.256 Y135.188 I.503 J-.774 E.0378
G2 X141.945 Y134.594 I-3.967 J-13.994 E.06842
G1 X141.945 Y140.759 E.23537
G3 X140.256 Y138.25 I5.1 J-5.259 E.11631
G1 X139.313 Y135.919 E.096
G2 X138.371 Y135.882 I-.503 J.774 E.0378
G2 X134.129 Y138.109 I2.315 J9.561 E.18477
G2 X132.715 Y140.36 I6.254 J5.499 E.10191
G1 X131.85 Y142.5 E.08814
G3 X133.215 Y143.583 I-20.733 J27.541 E.06655
G3 X136.957 Y145.65 I-2.517 J8.976 E.16466
G3 X138.712 Y148.744 I-7.79 J6.464 E.13654
G3 X140.106 Y150.304 I-50.927 J46.938 E.07987
G2 X141.945 Y149.675 I-4.61 J-16.488 E.07425
G1 X141.945 Y147.333 E.08944
G1 X127.09 Y139.188 F36000
G1 F13446.283
G2 X125.072 Y137.998 I-18.639 J29.288 E.08945
G1 X124.232 Y135.919 E.08559
G2 X123.289 Y135.882 I-.503 J.774 E.0378
G2 X121.91 Y136.355 I3.14 J11.415 E.0557
G1 X121.113 Y135.984 E.03356
G2 X121.261 Y134.531 I-8.582 J-1.61 E.05583
G1 X141.945 Y128.02 F36000
G1 F13446.283
G1 X141.945 Y125.678 E.08944
G3 X140.727 Y124.126 I4.522 J-4.805 E.07561
G3 X139.313 Y120.838 I21.629 J-11.25 E.13676
G1 X138.842 Y120.69 E.01887
G2 X136.485 Y121.471 I3.662 J14.998 E.09488
G2 X133.186 Y124.322 I2.292 J5.987 E.16986
G2 X131.772 Y127.609 I21.629 J11.25 E.13676
G1 X131.301 Y127.758 E.01886
G3 X129.887 Y127.345 I1.898 J-9.121 E.0563
G3 X126.117 Y124.858 I2.186 J-7.416 E.17499
G3 X124.485 Y121.464 I12.382 J-8.044 E.14415
G1 X125.785 Y119.91 E.07736
G2 X127.06 Y119.436 I-3.205 J-10.567 E.05194
G2 X130.359 Y116.585 I-2.292 J-5.987 E.16986
G2 X131.772 Y113.298 I-21.629 J-11.249 E.13676
G1 X132.244 Y113.149 E.01886
G3 X134.6 Y113.93 I-3.661 J14.995 E.09488
G3 X137.899 Y116.781 I-2.292 J5.987 E.16986
G3 X139.313 Y120.069 I-21.631 J11.25 E.13676
G1 X139.784 Y120.217 E.01886
G2 X141.945 Y119.513 I-3.473 J-14.318 E.08687
G1 X141.945 Y110.596 E.34042
G3 X140.727 Y109.045 I4.522 J-4.805 E.07561
G3 X139.313 Y105.757 I21.631 J-11.25 E.13676
G1 X138.842 Y105.608 E.01887
G2 X136.485 Y106.389 I3.661 J14.995 E.09488
G2 X133.186 Y109.24 I2.292 J5.987 E.16986
G2 X131.772 Y112.528 I21.631 J11.25 E.13676
G1 X131.301 Y112.677 E.01886
G3 X128.179 Y111.516 I4.813 J-17.728 E.12736
G2 X129.288 Y109.453 I-49.338 J-27.868 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 5.8
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X128.815 Y110.334 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L36
M991 S0 P35 ;notify layer change


G17
G3 Z6.04 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.846 Y119.029
G1 Z5.8
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.712 Y119.23 E.00922
G1 X122.328 Y123.268 E.20115
G1 X121.942 Y123.619 E.01991
G1 X121.523 Y123.84 E.01809
G1 X121.016 Y123.957 E.01987
G1 X120.455 Y123.928 E.02144
G1 X119.954 Y123.756 E.02022
G3 X118.918 Y122.954 I5.099 J-7.661 E.05006
G3 X114.848 Y126.662 I-38.498 J-38.172 E.21031
G1 X118.015 Y129.012 E.15054
G3 X118.932 Y129.849 I-4.609 J5.972 E.04749
G1 X119.567 Y130.648 E.03895
G1 X120.076 Y131.532 E.03893
G1 X120.45 Y132.48 E.03893
G3 X120.766 Y134.507 I-7.484 J2.202 E.07857
G1 X120.698 Y135.506 E.0382
G1 X120.538 Y136.268 E.02975
G3 X130.506 Y142.1 I-21.909 J48.88 E.44178
G3 X142.443 Y154.069 I-32.645 J44.496 E.64792
G1 X142.443 Y104.63 E1.88752
G1 X142.338 Y104.69 E.00461
G1 X141.737 Y104.785 E.02324
G1 X132.854 Y104.785 E.33914
G1 X132.552 Y104.761 E.01158
G1 X132.019 Y104.597 E.02128
G1 X131.573 Y104.305 E.02036
G1 X131.214 Y103.879 E.02125
G3 X124.807 Y115.927 I-51.65 J-19.738 E.52232
G1 X125.471 Y116.484 E.03305
G1 X125.877 Y116.954 E.02374
G1 X126.091 Y117.436 E.02015
G1 X126.168 Y117.978 E.02088
G1 X126.11 Y118.451 E.0182
G1 X125.95 Y118.874 E.01727
G1 X125.896 Y118.955 E.00369
G1 X125.389 Y118.65 F36000
G1 F13446.369
G1 X125.324 Y118.777 E.00547
G3 X124.449 Y119.826 I-15.05 J-11.66 E.05218
G1 X121.879 Y122.892 E.15272
G1 X121.609 Y123.138 E.01393
G1 X121.316 Y123.292 E.01265
G1 X120.961 Y123.374 E.0139
G1 X120.52 Y123.343 E.01689
G1 X120.196 Y123.222 E.0132
G3 X118.881 Y122.159 I9.15 J-12.668 E.0646
G3 X113.893 Y126.683 I-41.226 J-40.438 E.25725
G1 X117.666 Y129.482 E.17935
G3 X118.523 Y130.268 I-4.173 J5.412 E.04447
G1 X119.1 Y131.003 E.03566
G1 X119.563 Y131.814 E.03564
G1 X119.901 Y132.684 E.03564
G3 X120.135 Y133.79 I-9.583 J2.603 E.04321
G1 X120.181 Y134.538 E.02858
G1 X120.114 Y135.455 E.03513
G1 X119.933 Y136.284 E.03238
G1 X119.836 Y136.599 E.01259
G3 X130.932 Y143.157 I-21.323 J48.748 E.49333
G3 X142.92 Y155.769 I-32.175 J42.587 E.66731
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y116.041 E1.51599
G3 X143.044 Y103.316 I1151.272 J-5 E.48583
; LINE_WIDTH: 0.589876
G1 F14175.399
G1 X143.055 Y103.066 E.00906
G1 X143 Y103.308 E.00899
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.635 Y103.862 E.02534
G1 X142.158 Y104.133 E.02094
G1 X141.737 Y104.199 E.01626
G1 X132.854 Y104.199 E.33914
G1 X132.642 Y104.183 E.00811
G1 X132.243 Y104.055 E.016
G1 X131.958 Y103.863 E.01314
G1 F12704.948
G1 X131.674 Y103.519 E.01705
; LINE_WIDTH: 0.668766
G1 F11193.112
G1 X131.619 Y103.395 E.00558
; LINE_WIDTH: 0.717536
G1 F10755.126
G1 X131.564 Y103.272 E.00601
; LINE_WIDTH: 0.766306
G1 F10325.851
G1 X131.509 Y103.149 E.00644
; LINE_WIDTH: 0.815076
G1 F9905.348
G1 X131.454 Y103.026 E.00687
; LINE_WIDTH: 0.863846
G1 F9493.557
G1 X131.399 Y102.902 E.0073
; LINE_WIDTH: 0.912616
G1 F8966.39
G1 X131.345 Y102.779 E.00773
; LINE_WIDTH: 0.95901
G1 F8516.521
G1 X131.328 Y102.715 E.004
; LINE_WIDTH: 1.0054
G1 F8109.64
G1 X131.311 Y102.651 E.0042
; LINE_WIDTH: 1.0518
G1 F7739.862
G1 X131.294 Y102.587 E.0044
G1 X131.251 Y102.639 E.00448
; LINE_WIDTH: 1.0054
G1 F8109.64
G1 X131.208 Y102.691 E.00427
; LINE_WIDTH: 0.95901
G1 F8516.521
G1 X131.166 Y102.744 E.00407
; LINE_WIDTH: 0.912616
G1 F8966.39
G1 X131.087 Y102.887 E.00936
; LINE_WIDTH: 0.863846
G1 F9493.557
G1 X131.008 Y103.03 E.00884
; LINE_WIDTH: 0.815076
G1 F10086.586
G1 X130.929 Y103.173 E.00832
; LINE_WIDTH: 0.766306
G1 F10601.823
G1 X130.85 Y103.317 E.0078
; LINE_WIDTH: 0.717536
G1 F11129.938
G1 X130.772 Y103.46 E.00728
; LINE_WIDTH: 0.668766
G1 F11670.843
G1 X130.693 Y103.603 E.00676
; LINE_WIDTH: 0.619996
G1 F13048.139
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.209 Y104.835 E.01998
G3 X124.018 Y116.03 I-50.129 J-20.414 E.48957
G1 X125.095 Y116.932 E.05365
G1 X125.394 Y117.286 E.01771
G1 X125.529 Y117.599 E.01299
G1 X125.582 Y117.986 E.01491
G1 X125.512 Y118.412 E.01648
G1 X125.43 Y118.57 E.00678
G1 X124.862 Y118.401 F36000
G1 F13446.369
G1 X124.815 Y118.478 E.00346
G1 X121.43 Y122.516 E.20115
G1 X121.165 Y122.72 E.01278
G1 X120.906 Y122.791 E.01023
G1 X120.577 Y122.751 E.01267
G1 X120.333 Y122.612 E.01069
G1 X118.837 Y121.358 E.07454
G3 X112.93 Y126.698 I-40.369 J-38.722 E.30429
G1 X117.317 Y129.952 E.20855
G1 X118.114 Y130.687 E.0414
G1 X118.634 Y131.357 E.03238
G1 X119.05 Y132.096 E.03236
G1 X119.352 Y132.887 E.03236
G3 X119.551 Y133.826 I-11.681 J2.959 E.03664
G1 X119.596 Y134.568 E.02836
G1 X119.531 Y135.405 E.03205
G1 X119.363 Y136.148 E.0291
G1 X119.08 Y136.91 E.03103
G3 X130.579 Y143.624 I-20.428 J48.193 E.50975
G3 X142.648 Y156.415 I-31.78 J42.073 E.67457
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.529 E2.16502
G1 X142.967 Y99.604 E.02489
G1 X141.932 Y99.611 E.03953
G1 X142.504 Y102.701 E.11995
G3 X142.466 Y103.108 I-.767 J.135 E.01581
G1 X142.249 Y103.421 E.01454
G1 X141.977 Y103.576 E.01195
G1 X141.737 Y103.614 E.00928
G1 X132.854 Y103.614 E.33914
G1 X132.506 Y103.531 E.01367
G3 X132.18 Y103.225 I.349 J-.696 E.01729
G1 X132.077 Y102.789 E.0171
G1 X132.089 Y102.693 E.0037
G1 X132.66 Y99.61 E.11973
G3 X131.463 Y99.49 I.209 J-8.118 E.04596
G3 X123.222 Y116.127 I-51.69 J-15.243 E.7124
G1 X124.718 Y117.381 E.07453
G1 X124.934 Y117.671 E.01379
G1 X124.997 Y117.982 E.01212
G1 X124.935 Y118.282 E.01168
G1 X124.909 Y118.324 E.00188
M204 S250
G1 X124.391 Y118.123 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.003 Y122.165 E.16698
G1 X120.855 Y122.24 E.00527
G1 X120.689 Y122.189 E.00551
G1 X119.678 Y121.342 E.04174
; LINE_WIDTH: 0.521016
G1 X119.026 Y120.796 E.02697
; LINE_WIDTH: 0.531606
G1 X118.977 Y120.758 E.00202
; LINE_WIDTH: 0.556156
G1 X118.953 Y120.481 E.00946
G1 X117.999 Y121.433 E.04583
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-37.697 J-36.8 E.25284
G1 X116.988 Y130.396 E.19622
G1 X117.728 Y131.083 E.03196
G1 X118.194 Y131.692 E.02428
G1 X118.565 Y132.362 E.02426
G1 X118.834 Y133.08 E.02426
G1 X118.953 Y133.631 E.01784
G1 X119.04 Y134.534 E.02874
G1 X118.98 Y135.357 E.02612
G1 X118.825 Y136.02 E.02157
G1 X118.548 Y136.753 E.02478
G1 X118.31 Y137.19 E.01576
G3 X130.249 Y144.067 I-19.584 J47.798 E.43752
G3 X142.39 Y157.025 I-31.645 J41.817 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.349 E1.81559
G1 X144.167 Y98.943 E.01284
G1 X142.903 Y99.055 E.04019
G3 X141.267 Y99.06 I-.917 J-31.349 E.05179
G1 X141.959 Y102.796 E.12029
G1 X141.948 Y102.914 E.00376
G1 X141.806 Y103.05 E.00621
G1 X141.737 Y103.061 E.00223
G1 X132.854 Y103.061 E.28123
G1 X132.706 Y103.005 E.00501
M73 P67 R6
G1 X132.629 Y102.822 E.0063
G1 X132.632 Y102.794 E.00089
G1 X133.323 Y99.061 E.12021
G3 X132.124 Y99.023 I-.288 J-10.003 E.03802
G2 X131.038 Y98.936 I-.889 J4.266 E.03459
G3 X122.648 Y115.977 I-50.883 J-14.466 E.60463
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01786
G1 X122.701 Y116.412 E.01162
; LINE_WIDTH: 0.519996
G1 X124.363 Y117.805 E.06866
G1 X124.443 Y117.961 E.00556
G1 X124.419 Y118.037 E.00253
; WIPE_START
M204 S10000
G1 X123.781 Y118.808 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.693 Y113.98 Z6.2 F36000
G1 X143.055 Y103.066 Z6.2
G1 Z5.8
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.570336
G1 F14692.165
G1 X143.054 Y102.591 E.0166
; LINE_WIDTH: 0.611796
G1 F13637.309
G1 X143.033 Y102.365 E.00854
; LINE_WIDTH: 0.657511
G1 F12636.901
G1 X143.01 Y102.116 E.01016
; LINE_WIDTH: 0.703226
G1 F11773.237
G1 X142.987 Y101.868 E.0109
; LINE_WIDTH: 0.748941
G1 F11020.075
G1 X142.964 Y101.619 E.01165
; LINE_WIDTH: 0.794656
G1 F10357.481
G1 X142.942 Y101.37 E.01239
; LINE_WIDTH: 0.840371
G1 F9770.048
G1 X142.919 Y101.121 E.01314
; LINE_WIDTH: 0.886086
G1 F9245.672
G1 X142.896 Y100.872 E.01388
; LINE_WIDTH: 0.931801
G1 F8774.716
G1 X142.873 Y100.623 E.01463
; LINE_WIDTH: 0.977516
G1 F8349.415
G1 X142.85 Y100.374 E.01537
; WIPE_START
G1 X142.873 Y100.623 E-.095
G1 X142.896 Y100.872 E-.095
G1 X142.919 Y101.121 E-.095
G1 X142.942 Y101.37 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.35 Y102.163 Z6.2 F36000
G1 X131.294 Y102.587 Z6.2
G1 Z5.8
G1 E.4 F1800
; LINE_WIDTH: 1.0518
G1 F7739.862
G1 X131.304 Y102.548 E.00264
; LINE_WIDTH: 1.04538
G1 F7789.01
G1 X131.373 Y102.282 E.01812
; LINE_WIDTH: 1.00458
G1 F8116.506
G1 X131.443 Y102.016 E.01739
; LINE_WIDTH: 0.963786
G1 F8472.753
G1 X131.513 Y101.75 E.01666
; LINE_WIDTH: 0.922991
G1 F8861.707
G1 X131.583 Y101.485 E.01593
; LINE_WIDTH: 0.882196
G1 F9288.09
G1 X131.593 Y101.449 E.00203
; LINE_WIDTH: 0.877196
G1 F9343.189
G1 X131.671 Y101.135 E.0178
; LINE_WIDTH: 0.837061
G1 F9810.335
G1 X131.75 Y100.821 E.01695
; LINE_WIDTH: 0.796926
G1 F10326.651
G1 X131.828 Y100.506 E.0161
; LINE_WIDTH: 0.756791
G1 F10900.334
G1 X131.907 Y100.192 E.01526
; WIPE_START
G1 X131.828 Y100.506 E-.12309
G1 X131.75 Y100.821 E-.12309
G1 X131.671 Y101.135 E-.12309
G1 X131.664 Y101.162 E-.01074
; WIPE_END
G1 E-.02 F1800
G1 X127.469 Y107.538 Z6.2 F36000
G1 X118.953 Y120.481 Z6.2
G1 Z5.8
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.556156
G1 F3600
M204 S5000
G1 X119.2 Y120.199 E.01276
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15232
; LINE_WIDTH: 0.552896
G1 X122.357 Y116.418 E.01164
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.13088
G1 X121.722 Y117.19 E-.24912
; WIPE_END
G1 E-.02 F1800
G1 X119.969 Y124.302 Z6.2 F36000
G1 Z5.8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X118.947 Y123.628 I1.504 J-3.392 E.04695
G1 X118.158 Y124.399 E.04213
G2 X117.366 Y127.052 I5.769 J3.166 E.10648
G1 X117.451 Y127.523 E.01828
G1 X117.847 Y127.994 E.02351
G2 X119.635 Y128.937 I8.56 J-14.072 E.07723
G3 X123.557 Y134.592 I-2.906 J6.202 E.27529
G1 X123.472 Y135.063 E.01828
G1 X123.076 Y135.535 E.02351
G3 X121.711 Y136.27 I-6.659 J-10.734 E.05924
G3 X125.269 Y138.108 I-22.463 J47.84 E.15293
G3 X124.907 Y136.477 I4.957 J-1.956 E.06403
G1 X124.991 Y136.006 E.01828
G1 X125.388 Y135.535 E.02351
G3 X127.176 Y134.592 I8.56 J14.072 E.07723
G2 X131.098 Y128.937 I-2.906 J-6.202 E.27529
G1 X131.013 Y128.465 E.01828
G1 X130.617 Y127.994 E.02351
G2 X128.828 Y127.052 I-8.56 J14.072 E.07723
G3 X124.907 Y121.396 I2.906 J-6.202 E.27529
G1 X124.991 Y120.925 E.01828
G1 X125.388 Y120.453 E.02351
G3 X127.176 Y119.511 I8.56 J14.072 E.07723
G2 X131.098 Y113.855 I-2.906 J-6.202 E.27529
G1 X131.013 Y113.384 E.01828
G1 X130.617 Y112.913 E.02351
G2 X128.14 Y111.584 I-107.93 J198.142 E.10733
G2 X129.253 Y109.523 I-74.223 J-41.441 E.08943
G1 X141.945 Y112.96 F36000
G1 F13446.283
G1 X141.945 Y110.617 E.08944
G3 X139.988 Y106.315 I5.323 J-5.018 E.18376
G1 X140.073 Y105.843 E.01828
G1 X140.469 Y105.372 E.02351
G1 X140.626 Y105.283 E.0069
G1 X138.233 Y105.283 E.09138
G3 X136.369 Y106.315 I-5.54 J-7.805 E.0815
G2 X132.447 Y111.97 I2.906 J6.202 E.27529
G1 X132.532 Y112.442 E.01828
G1 X132.928 Y112.913 E.02351
G2 X134.717 Y113.855 I8.559 J-14.071 E.07723
G3 X138.638 Y119.511 I-2.906 J6.202 E.27529
G1 X138.554 Y119.982 E.01828
G1 X138.157 Y120.453 E.02351
G3 X136.369 Y121.396 I-8.559 J-14.07 E.07723
G2 X132.447 Y127.052 I2.906 J6.202 E.27529
G1 X132.532 Y127.523 E.01828
G1 X132.928 Y127.994 E.02351
G2 X134.717 Y128.937 I8.559 J-14.071 E.07723
G3 X138.638 Y134.592 I-2.906 J6.202 E.27529
G1 X138.554 Y135.063 E.01828
G1 X138.157 Y135.535 E.02351
G3 X136.369 Y136.477 I-8.559 J-14.07 E.07723
G2 X132.447 Y142.133 I2.906 J6.202 E.27529
G1 X132.532 Y142.604 E.01828
G1 X132.928 Y143.075 E.02351
G2 X134.717 Y144.018 I8.559 J-14.071 E.07723
G3 X138.389 Y148.397 I-2.671 J5.969 E.22582
G3 X140.419 Y150.675 I-41.525 J39.047 E.11652
G3 X141.945 Y149.826 I5.695 J8.444 E.06676
G1 X141.945 Y140.78 E.3454
G3 X139.988 Y136.477 I5.323 J-5.018 E.18376
G1 X140.073 Y136.006 E.01828
G1 X140.469 Y135.535 E.02351
G3 X141.945 Y134.745 I7.159 J11.612 E.06397
G1 X141.945 Y125.698 E.3454
G3 X139.988 Y121.396 I5.323 J-5.018 E.18376
G1 X140.073 Y120.925 E.01828
G1 X140.469 Y120.453 E.02351
G3 X141.945 Y119.664 I7.159 J11.612 E.06397
G1 X141.945 Y117.321 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 5.96
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y118.321 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L37
M991 S0 P36 ;notify layer change


G17
G3 Z6.2 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.793 Y119.094
G1 Z5.96
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.661 Y119.292 E.00906
G1 X122.368 Y123.22 E.19568
G1 X121.962 Y123.58 E.02073
G1 X121.51 Y123.801 E.01922
G1 X120.975 Y123.902 E.02079
G1 X120.428 Y123.85 E.02098
G1 X120.015 Y123.699 E.01678
G3 X118.975 Y122.897 I4.257 J-6.598 E.05021
G3 X114.848 Y126.662 I-39.552 J-39.209 E.21337
G1 X118.015 Y129.012 E.15055
G1 X118.485 Y129.403 E.02334
G1 X119.156 Y130.104 E.03709
G1 X119.749 Y130.932 E.03885
G1 X120.215 Y131.839 E.03894
G1 X120.505 Y132.666 E.03345
G3 X120.761 Y134.433 I-7.479 J1.985 E.06833
G1 X120.698 Y135.505 E.041
G1 X120.538 Y136.268 E.02978
G3 X130.506 Y142.1 I-22.261 J49.482 E.44174
G3 X142.443 Y154.069 I-32.639 J44.49 E.64795
G1 X142.443 Y104.634 E1.8874
G1 X142.341 Y104.692 E.0045
G1 X141.739 Y104.787 E.02324
G1 X132.852 Y104.787 E.33933
G1 X132.549 Y104.763 E.01159
G1 X132.015 Y104.599 E.02131
G1 X131.569 Y104.306 E.02039
G1 X131.213 Y103.882 E.02113
G3 X124.761 Y115.993 I-51.673 J-19.751 E.52526
G1 X125.419 Y116.545 E.03278
G1 X125.847 Y117.05 E.02527
G1 X126.047 Y117.525 E.01967
G1 X126.117 Y118.039 E.01982
G1 X126.058 Y118.515 E.01831
G1 X125.898 Y118.937 E.01721
G1 X125.843 Y119.019 E.0038
G1 X125.338 Y118.711 F36000
G1 F13446.369
G1 X125.272 Y118.839 E.00548
G3 X124.493 Y119.774 I-13.459 J-10.424 E.0465
G1 X121.923 Y122.84 E.15272
G1 X121.639 Y123.091 E.0145
G1 X121.231 Y123.274 E.01705
G1 X120.781 Y123.314 E.01726
G1 X120.445 Y123.246 E.01307
G1 X120.009 Y123 E.01913
G1 X118.936 Y122.101 E.05342
G3 X113.893 Y126.683 I-41.652 J-40.777 E.26031
G1 X117.666 Y129.482 E.17936
G1 X118.108 Y129.851 E.02196
G1 X118.727 Y130.503 E.03435
G1 X119.266 Y131.263 E.03557
G1 X119.689 Y132.096 E.03566
G1 X119.949 Y132.848 E.03039
G1 X120.082 Y133.46 E.02391
G1 X120.177 Y134.468 E.03867
G1 X120.114 Y135.454 E.03772
G1 X119.933 Y136.283 E.0324
G1 X119.836 Y136.599 E.01261
G3 X130.932 Y143.157 I-21.129 J48.421 E.49334
G3 X142.92 Y155.769 I-32.177 J42.588 E.66731
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y116.042 E1.51595
G3 X143.045 Y103.318 I1096.326 J-5 E.48581
; LINE_WIDTH: 0.588386
G1 F14213.52
G1 X143.056 Y103.068 E.00903
G1 X143.002 Y103.31 E.00896
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.637 Y103.864 E.02533
G1 X142.16 Y104.135 E.02094
G1 X141.739 Y104.201 E.01626
G1 X132.852 Y104.201 E.33933
G1 X132.64 Y104.185 E.00811
G1 X132.24 Y104.057 E.01601
G1 X131.954 Y103.865 E.01315
G1 F12531.8
G1 X131.668 Y103.516 E.01723
; LINE_WIDTH: 0.667178
G1 F11015.309
G1 X131.627 Y103.41 E.00469
; LINE_WIDTH: 0.714361
G1 F10648.482
G1 X131.586 Y103.304 E.00504
; LINE_WIDTH: 0.761543
G1 F10287.878
G1 X131.545 Y103.197 E.00539
; LINE_WIDTH: 0.808725
G1 F9933.475
G1 X131.504 Y103.091 E.00574
; LINE_WIDTH: 0.855907
G1 F9585.294
G1 X131.463 Y102.985 E.00609
; LINE_WIDTH: 0.90309
G1 F9064.714
G1 X131.422 Y102.879 E.00644
; LINE_WIDTH: 0.950272
G1 F8597.767
G1 X131.381 Y102.773 E.00679
; LINE_WIDTH: 0.997454
G1 F8176.57
G1 X131.34 Y102.667 E.00714
; LINE_WIDTH: 1.04464
G1 F7794.715
G1 X131.299 Y102.561 E.00749
G1 X131.232 Y102.676 E.0088
; LINE_WIDTH: 0.997454
G1 F8176.57
G1 X131.165 Y102.791 E.00838
; LINE_WIDTH: 0.950272
G1 F8597.767
G1 X131.098 Y102.907 E.00797
; LINE_WIDTH: 0.90309
G1 F9064.714
G1 X131.031 Y103.022 E.00756
; LINE_WIDTH: 0.855907
G1 F9585.294
G1 X130.963 Y103.138 E.00715
; LINE_WIDTH: 0.808725
G1 F10169.31
G1 X130.896 Y103.253 E.00674
; LINE_WIDTH: 0.761543
G1 F10590.83
G1 X130.829 Y103.368 E.00633
; LINE_WIDTH: 0.714361
G1 F11020.91
G1 X130.762 Y103.484 E.00592
; LINE_WIDTH: 0.667178
G1 F11459.549
G1 X130.694 Y103.599 E.00551
; LINE_WIDTH: 0.619996
G1 F12824.669
G1 X130.548 Y103.971 E.01527
G1 F13446.369
G1 X130.402 Y104.344 E.01527
G1 X130.209 Y104.835 E.02015
G3 X123.97 Y116.095 I-50.105 J-20.403 E.49262
G1 X125.043 Y116.994 E.05344
G1 X125.357 Y117.373 E.01878
G1 X125.483 Y117.679 E.01265
G1 X125.531 Y118.05 E.01429
G1 X125.46 Y118.475 E.01643
G1 X125.379 Y118.631 E.00672
G1 X124.814 Y118.449 F36000
G1 F13446.369
G1 X124.763 Y118.539 E.00395
G1 X121.477 Y122.46 E.1953
G1 X121.215 Y122.659 E.01258
G1 X120.921 Y122.732 E.01157
G1 X120.634 Y122.691 E.01107
G1 X120.385 Y122.551 E.01092
G1 X118.893 Y121.3 E.07434
G3 X112.93 Y126.698 I-40.783 J-39.061 E.30734
G1 X117.317 Y129.952 E.20856
G1 X117.73 Y130.299 E.02058
G1 X118.298 Y130.901 E.0316
G1 X118.783 Y131.594 E.03229
G3 X119.508 Y133.575 I-5.458 J3.119 E.08093
G1 X119.592 Y134.504 E.0356
G1 X119.531 Y135.404 E.03444
G1 X119.363 Y136.148 E.02912
G1 X119.08 Y136.91 E.03105
G3 X130.579 Y143.624 I-20.519 J48.349 E.50973
G3 X142.648 Y156.415 I-31.78 J42.074 E.67457
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.529 E2.16499
G1 X142.967 Y99.606 E.02489
G1 X141.934 Y99.614 E.03944
G1 X142.506 Y102.703 E.11995
G3 X142.468 Y103.11 I-.766 J.135 E.01581
G1 X142.252 Y103.423 E.01454
G1 X141.979 Y103.578 E.01195
G1 X141.739 Y103.616 E.00928
G1 X132.852 Y103.616 E.33933
G1 X132.503 Y103.533 E.01368
G3 X132.176 Y103.224 I.349 J-.696 E.01741
G1 X132.085 Y102.701 E.0203
G1 X132.657 Y99.612 E.11993
G3 X131.461 Y99.49 I.211 J-7.995 E.04594
G3 X123.175 Y116.192 I-51.656 J-15.221 E.71544
G1 X124.667 Y117.443 E.07432
G1 X124.846 Y117.659 E.01072
G1 X124.943 Y117.983 E.01294
G1 X124.905 Y118.288 E.01171
G1 X124.858 Y118.37 E.00362
M204 S250
G1 X124.34 Y118.184 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.057 Y122.101 E.1618
G1 X120.895 Y122.18 E.00568
G1 X120.74 Y122.127 E.00519
G1 X120.542 Y121.962 E.00816
G1 X120.345 Y121.796 E.00816
G1 X120.147 Y121.63 E.00816
G1 X119.95 Y121.465 E.00816
G1 X119.795 Y121.335 E.00638
; LINE_WIDTH: 0.520236
G1 X119.678 Y121.237 E.00484
; LINE_WIDTH: 0.520526
G1 X119.528 Y121.112 E.0062
; LINE_WIDTH: 0.520826
G1 X119.377 Y120.986 E.00621
; LINE_WIDTH: 0.521126
G1 X119.227 Y120.861 E.00621
; LINE_WIDTH: 0.521416
G1 X119.077 Y120.735 E.00621
; LINE_WIDTH: 0.552136
G1 X119.006 Y120.421 E.01089
G1 X117.999 Y121.433 E.04819
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.723 J-39.102 E.25281
G1 X116.988 Y130.396 E.19623
G1 X117.374 Y130.721 E.01599
G1 X117.893 Y131.277 E.02406
G1 X118.327 Y131.907 E.02421
G3 X118.995 Y133.832 I-5.07 J2.838 E.06485
G1 X119.04 Y134.538 E.0224
G1 X118.98 Y135.356 E.02599
G1 X118.825 Y136.02 E.02158
G1 X118.548 Y136.753 E.0248
G1 X118.31 Y137.19 E.01576
G3 X130.249 Y144.067 I-19.584 J47.797 E.43752
G3 X142.39 Y157.025 I-31.645 J41.818 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.348 E1.8156
G1 X144.167 Y98.943 E.01282
G1 X142.902 Y99.057 E.04022
G3 X141.27 Y99.062 I-.91 J-29.163 E.05169
G1 X141.962 Y102.798 E.12029
G1 X141.951 Y102.916 E.00376
G1 X141.809 Y103.052 E.00621
G1 X141.739 Y103.063 E.00223
G1 X132.852 Y103.063 E.28139
G1 X132.703 Y103.007 E.00502
G1 X132.656 Y102.949 E.00236
G1 X132.648 Y102.697 E.00799
G1 X133.321 Y99.063 E.11703
G3 X132.124 Y99.025 I-.287 J-9.819 E.03795
G2 X131.038 Y98.936 I-.889 J4.199 E.03459
G3 X122.631 Y116 I-50.904 J-14.476 E.60555
; LINE_WIDTH: 0.548796
G1 X122.308 Y116.481 E.01942
G1 X122.65 Y116.473 E.01148
; LINE_WIDTH: 0.520006
G1 X122.695 Y116.511 E.00185
; LINE_WIDTH: 0.519996
G1 X122.739 Y116.548 E.00185
G1 X123.249 Y116.975 E.02105
G1 X123.78 Y117.421 E.02195
G1 X124.312 Y117.866 E.02195
G1 X124.392 Y118.023 E.00557
G1 X124.367 Y118.099 E.00251
; WIPE_START
M204 S10000
G1 X123.73 Y118.869 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.639 Y114.038 Z6.36 F36000
G1 X143.056 Y103.068 Z6.36
G1 Z5.96
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.567796
G1 F14762.121
G1 X143.055 Y102.594 E.01651
; LINE_WIDTH: 0.609516
G1 F13691.367
G1 X143.034 Y102.366 E.00856
; LINE_WIDTH: 0.655231
G1 F12683.305
G1 X143.011 Y102.117 E.01012
; LINE_WIDTH: 0.700946
G1 F11813.505
G1 X142.988 Y101.868 E.01086
; LINE_WIDTH: 0.746661
G1 F11055.348
G1 X142.966 Y101.619 E.01161
; LINE_WIDTH: 0.792376
G1 F10388.635
G1 X142.943 Y101.371 E.01235
; LINE_WIDTH: 0.838091
G1 F9797.762
G1 X142.92 Y101.122 E.0131
; LINE_WIDTH: 0.883806
G1 F9270.487
G1 X142.897 Y100.873 E.01384
; LINE_WIDTH: 0.929521
G1 F8797.065
G1 X142.874 Y100.624 E.01459
; LINE_WIDTH: 0.975236
G1 F8369.647
G1 X142.851 Y100.375 E.01533
; WIPE_START
G1 X142.874 Y100.624 E-.095
G1 X142.897 Y100.873 E-.095
G1 X142.92 Y101.122 E-.095
G1 X142.943 Y101.371 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.35 Y102.147 Z6.36 F36000
G1 X131.299 Y102.561 Z6.36
G1 Z5.96
G1 E.4 F1800
; LINE_WIDTH: 1.04464
G1 F7794.715
G1 X131.338 Y102.421 E.00956
; LINE_WIDTH: 1.02072
G1 F7983.738
G1 X131.422 Y102.097 E.02153
; LINE_WIDTH: 0.972176
G1 F8396.955
G1 X131.507 Y101.773 E.02047
; LINE_WIDTH: 0.923636
G1 F8855.28
G1 X131.592 Y101.449 E.01941
; LINE_WIDTH: 0.875096
G1 F9366.526
G1 X131.67 Y101.135 E.01774
; LINE_WIDTH: 0.834976
G1 F9835.882
G1 X131.749 Y100.82 E.0169
; LINE_WIDTH: 0.794856
G1 F10354.758
G1 X131.827 Y100.506 E.01605
; LINE_WIDTH: 0.754736
G1 F10931.428
G1 X131.906 Y100.192 E.0152
; WIPE_START
G1 X131.827 Y100.506 E-.12303
G1 X131.749 Y100.82 E-.12303
G1 X131.67 Y101.135 E-.12303
G1 X131.663 Y101.162 E-.01092
; WIPE_END
G1 E-.02 F1800
G1 X127.471 Y107.541 Z6.36 F36000
G1 X119.006 Y120.421 Z6.36
G1 Z5.96
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.552136
G1 F3600
M204 S5000
G1 X119.2 Y120.198 E.00996
; LINE_WIDTH: 0.544336
G1 X122.143 Y116.688 E.15231
; LINE_WIDTH: 0.548796
G1 X122.308 Y116.481 E.00887
; WIPE_START
M204 S10000
G1 X122.143 Y116.688 E-.10046
G1 X121.67 Y117.251 E-.27954
; WIPE_END
G1 E-.02 F1800
G1 X120.109 Y124.275 Z6.36 F36000
G1 Z5.96
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X119.003 Y123.571 I3.056 J-6.016 E.05012
G1 X118.273 Y124.287 E.03905
G2 X117.702 Y127.052 I5.164 J2.507 E.10891
G1 X117.893 Y127.523 E.01941
G2 X119.865 Y128.937 I4.913 J-4.771 E.09315
G3 X122.626 Y131.764 I-3.059 J5.747 E.15329
G3 X123.221 Y134.592 I-5.225 J2.577 E.11149
G1 X123.03 Y135.063 E.01941
G3 X121.548 Y136.192 I-4.117 J-3.868 E.07146
G3 X125.427 Y138.197 I-22.734 J48.74 E.16675
G3 X125.243 Y136.477 I3.538 J-1.249 E.06664
G1 X125.433 Y136.006 E.01941
G3 X127.406 Y134.592 I4.913 J4.771 E.09315
G2 X130.166 Y131.764 I-3.059 J-5.748 E.15329
G2 X130.762 Y128.937 I-5.225 J-2.577 E.11149
G1 X130.571 Y128.465 E.01941
G2 X128.598 Y127.052 I-4.913 J4.771 E.09315
G3 X125.838 Y124.224 I3.059 J-5.747 E.15329
G3 X125.243 Y121.396 I5.225 J-2.577 E.11149
G1 X125.433 Y120.925 E.01941
G3 X127.406 Y119.511 I4.913 J4.771 E.09315
G2 X130.166 Y116.683 I-3.059 J-5.748 E.15329
G2 X130.762 Y113.855 I-5.225 J-2.577 E.11149
G2 X130.126 Y112.913 I-1.526 J.344 E.04443
G2 X128.101 Y111.651 I-54.517 J85.256 E.09108
G2 X129.218 Y109.593 I-33.295 J-19.398 E.08945
G1 X141.945 Y112.983 F36000
G1 F13446.283
G1 X141.945 Y110.64 E.08944
G3 X140.919 Y109.142 I4.535 J-4.208 E.06956
G3 X140.324 Y106.315 I5.225 J-2.577 E.11149
G3 X141.093 Y105.285 I2.027 J.711 E.04983
G1 X137.749 Y105.285 E.12766
G3 X136.139 Y106.315 I-6.573 J-8.501 E.07306
G2 X133.379 Y109.142 I3.059 J5.748 E.15329
G2 X132.783 Y111.97 I5.225 J2.577 E.11149
G1 X132.974 Y112.442 E.01941
G2 X134.947 Y113.855 I4.913 J-4.771 E.09315
G3 X137.707 Y116.683 I-3.059 J5.748 E.15329
G3 X138.302 Y119.511 I-5.225 J2.577 E.11149
G1 X138.111 Y119.982 E.01941
G3 X136.139 Y121.396 I-4.913 J-4.771 E.09315
G2 X133.379 Y124.224 I3.059 J5.748 E.15329
G2 X132.783 Y127.052 I5.225 J2.577 E.11149
G1 X132.974 Y127.523 E.01941
G2 X134.947 Y128.937 I4.913 J-4.771 E.09315
G3 X137.707 Y131.764 I-3.059 J5.747 E.15329
G3 X138.302 Y134.592 I-5.225 J2.577 E.11149
G1 X138.111 Y135.063 E.01941
G3 X136.139 Y136.477 I-4.913 J-4.771 E.09315
G2 X133.379 Y139.305 I3.059 J5.747 E.15329
G2 X132.783 Y142.133 I5.225 J2.577 E.11149
G1 X132.974 Y142.604 E.01941
G2 X134.947 Y144.018 I4.913 J-4.771 E.09315
G3 X138.167 Y148.161 I-2.882 J5.564 E.20671
G3 X140.647 Y150.947 I-80.627 J74.264 E.14242
G3 X141.945 Y149.988 I3.979 J4.028 E.06183
G1 X141.945 Y140.802 E.3507
G3 X140.919 Y139.305 I4.536 J-4.209 E.06956
G3 X140.324 Y136.477 I5.225 J-2.577 E.11149
G1 X140.515 Y136.006 E.01941
G3 X141.945 Y134.907 I4.052 J3.793 E.06919
G1 X141.945 Y125.721 E.3507
G3 X140.919 Y124.224 I4.535 J-4.208 E.06956
G3 X140.324 Y121.396 I5.225 J-2.577 E.11149
G1 X140.515 Y120.925 E.01941
G3 X141.945 Y119.826 I4.052 J3.793 E.06919
G1 X141.945 Y117.483 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 6.12
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y118.483 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L38
M991 S0 P37 ;notify layer change


G17
G3 Z6.36 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.688 Y119.221
G1 Z6.12
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.558 Y119.414 E.00889
M73 P68 R6
G1 X122.482 Y123.084 E.18283
G1 X122.132 Y123.41 E.01826
G1 X121.642 Y123.669 E.02116
G1 X121.138 Y123.776 E.01968
G1 X120.585 Y123.739 E.02115
G1 X120.199 Y123.615 E.01549
G1 X119.735 Y123.326 E.02085
G1 X119.088 Y122.784 E.03224
G3 X114.848 Y126.662 I-38.108 J-37.401 E.21951
G1 X118.015 Y129.012 E.15056
G3 X118.932 Y129.849 I-4.607 J5.969 E.04748
G1 X119.565 Y130.646 E.03883
G1 X120.076 Y131.532 E.03904
G1 X120.45 Y132.48 E.03891
G3 X120.765 Y134.489 I-7.466 J2.197 E.07786
G1 X120.698 Y135.505 E.0389
G1 X120.538 Y136.268 E.02975
G3 X130.504 Y142.099 I-21.932 J48.915 E.44168
G3 X142.443 Y154.069 I-32.638 J44.492 E.64803
G1 X142.443 Y104.637 E1.88727
G1 X142.343 Y104.694 E.00439
G1 X141.742 Y104.789 E.02324
G1 X132.849 Y104.789 E.33952
G1 X132.546 Y104.765 E.01162
G1 X132.012 Y104.6 E.02132
G1 X131.565 Y104.307 E.0204
G1 X131.212 Y103.885 E.02102
G3 X124.67 Y116.125 I-51.304 J-19.552 E.53129
G3 X125.514 Y116.858 I-4.781 J6.361 E.0427
G1 X125.802 Y117.278 E.01946
G1 X125.958 Y117.7 E.01717
G1 X126.014 Y118.186 E.0187
G1 X125.953 Y118.643 E.0176
G1 X125.794 Y119.062 E.01709
G1 X125.738 Y119.146 E.00386
G1 X125.205 Y118.883 F36000
G1 F13446.369
G1 X125.109 Y119.038 E.00697
G1 X122.033 Y122.708 E.18283
G3 X121.093 Y123.192 I-1.11 J-1.002 E.04128
G1 X120.657 Y123.155 E.01669
G1 X120.436 Y123.079 E.00893
G1 X120.065 Y122.839 E.01686
G1 X119.047 Y121.985 E.05072
G3 X113.893 Y126.683 I-39.071 J-37.69 E.26645
G1 X117.666 Y129.482 E.17937
G3 X118.523 Y130.268 I-4.173 J5.411 E.04446
G1 X119.099 Y131 E.03555
G1 X119.563 Y131.814 E.03576
G1 X119.901 Y132.683 E.03562
G3 X120.136 Y133.793 I-9.672 J2.622 E.04332
G1 X120.18 Y134.524 E.02797
G1 X120.114 Y135.455 E.03562
G1 X119.931 Y136.29 E.03263
G1 X119.836 Y136.599 E.01236
G3 X130.932 Y143.157 I-21.324 J48.75 E.49332
G3 X142.92 Y155.769 I-32.375 J42.777 E.66729
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y116.043 E1.51591
G3 X143.045 Y103.32 I1046.985 J-5 E.48578
; LINE_WIDTH: 0.586906
G1 F14251.59
G1 X143.057 Y103.07 E.009
G1 X143.004 Y103.312 E.00893
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.64 Y103.866 E.02533
G1 X142.163 Y104.137 E.02094
G1 X141.742 Y104.203 E.01626
G1 X132.849 Y104.203 E.33952
G1 X132.637 Y104.187 E.00813
G1 X132.237 Y104.059 E.01602
G1 X131.951 Y103.866 E.01317
G1 F12542.705
G1 X131.665 Y103.517 E.01725
; LINE_WIDTH: 0.666994
G1 F11024.105
G1 X131.624 Y103.411 E.00468
; LINE_WIDTH: 0.713992
G1 F10657.804
G1 X131.583 Y103.305 E.00503
; LINE_WIDTH: 0.76099
G1 F10297.703
G1 X131.542 Y103.199 E.00538
; LINE_WIDTH: 0.807987
G1 F9943.79
G1 X131.502 Y103.093 E.00573
; LINE_WIDTH: 0.854985
G1 F9596.065
G1 X131.461 Y102.987 E.00608
; LINE_WIDTH: 0.901983
G1 F9076.276
G1 X131.42 Y102.881 E.00642
; LINE_WIDTH: 0.948981
G1 F8609.904
G1 X131.379 Y102.775 E.00677
; LINE_WIDTH: 0.995978
G1 F8189.118
G1 X131.339 Y102.669 E.00712
; LINE_WIDTH: 1.04298
G1 F7807.543
G1 X131.298 Y102.563 E.00747
G1 X131.231 Y102.678 E.00876
; LINE_WIDTH: 0.995978
G1 F8189.118
G1 X131.164 Y102.793 E.00835
; LINE_WIDTH: 0.948981
G1 F8609.904
G1 X131.097 Y102.908 E.00794
; LINE_WIDTH: 0.901983
G1 F9076.276
G1 X131.03 Y103.023 E.00754
; LINE_WIDTH: 0.854985
G1 F9596.065
G1 X130.963 Y103.138 E.00713
; LINE_WIDTH: 0.807987
G1 F10179.008
G1 X130.896 Y103.254 E.00672
; LINE_WIDTH: 0.76099
G1 F10599.733
G1 X130.828 Y103.369 E.00631
; LINE_WIDTH: 0.713992
G1 F11028.994
G1 X130.761 Y103.484 E.0059
; LINE_WIDTH: 0.666994
G1 F11466.759
G1 X130.694 Y103.599 E.00549
; LINE_WIDTH: 0.619996
G1 F12832.296
G1 X130.548 Y103.971 E.01527
G1 F13446.369
G1 X130.402 Y104.344 E.01527
G1 X130.208 Y104.835 E.02015
G3 X123.876 Y116.224 I-50.244 J-20.48 E.49873
G1 X124.94 Y117.116 E.05302
G1 X125.266 Y117.517 E.01974
G1 X125.389 Y117.838 E.01312
G1 X125.422 Y118.287 E.01718
G1 X125.318 Y118.698 E.01619
G1 X125.252 Y118.806 E.00482
G1 X124.711 Y118.572 F36000
G1 F13446.369
G1 X124.661 Y118.662 E.00393
G1 X121.584 Y122.332 E.18283
G1 X121.354 Y122.519 E.01134
G1 X121.048 Y122.608 E.01217
G1 X120.749 Y122.573 E.01148
G1 X120.442 Y122.39 E.01365
G1 X119.004 Y121.184 E.07164
G3 X112.93 Y126.698 I-39.214 J-37.098 E.31349
G1 X117.317 Y129.953 E.20857
G1 X118.114 Y130.687 E.04138
G1 X118.632 Y131.355 E.03227
G1 X119.05 Y132.096 E.03247
G1 X119.352 Y132.887 E.03234
G3 X119.551 Y133.829 I-11.823 J2.988 E.03675
G1 X119.596 Y134.56 E.02797
G1 X119.531 Y135.404 E.03234
G1 X119.361 Y136.154 E.02935
G1 X119.08 Y136.91 E.0308
G3 X130.579 Y143.624 I-20.428 J48.193 E.50973
G3 X142.647 Y156.415 I-31.963 J42.247 E.67455
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.531 E2.16493
G1 X142.966 Y99.608 E.02493
G1 X141.937 Y99.616 E.03931
G1 X142.509 Y102.705 E.11995
G3 X142.471 Y103.112 I-.767 J.135 E.01581
G1 X142.254 Y103.425 E.01454
G1 X141.982 Y103.58 E.01195
G1 X141.742 Y103.618 E.00928
G1 X132.849 Y103.618 E.33952
G1 X132.5 Y103.535 E.0137
G3 X132.173 Y103.226 I.349 J-.696 E.01742
G1 X132.083 Y102.703 E.02027
G1 X132.655 Y99.614 E.11994
G3 X131.46 Y99.49 I.212 J-7.874 E.04591
G3 X123.081 Y116.322 I-51.952 J-15.36 E.7215
G1 X124.564 Y117.565 E.0739
G1 X124.758 Y117.809 E.01189
G1 X124.841 Y118.11 E.01192
G1 X124.801 Y118.412 E.01164
G1 X124.755 Y118.494 E.00358
M204 S250
G1 X124.237 Y118.307 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.161 Y121.977 E.15161
G1 X121.005 Y122.057 E.00554
G1 X120.928 Y122.031 E.00257
G1 X120.851 Y122.004 E.00257
G1 X120.824 Y121.989 E.00098
G1 X120.809 Y121.976 E.00063
G1 X120.154 Y121.427 E.02707
G1 F3450
G1 X119.498 Y120.878 E.02707
G1 F3300
G1 X119.276 Y120.692 E.00917
; LINE_WIDTH: 0.520116
G1 F3600
G1 X119.269 Y120.686 E.00029
; LINE_WIDTH: 0.520456
G1 X119.248 Y120.669 E.00085
; LINE_WIDTH: 0.520786
G1 X119.227 Y120.652 E.00085
; LINE_WIDTH: 0.521126
G1 X119.207 Y120.635 E.00085
; LINE_WIDTH: 0.521466
G1 X119.186 Y120.618 E.00086
; LINE_WIDTH: 0.521556
G1 X119.18 Y120.613 E.00025
; LINE_WIDTH: 0.531016
G1 X119.151 Y120.483 E.0043
; LINE_WIDTH: 0.544336
G1 X119.112 Y120.3 E.00622
G1 X118.674 Y120.729 E.02039
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.705 J-36.469 E.28369
G1 X116.988 Y130.397 E.19623
G1 X117.728 Y131.083 E.03195
G1 X118.192 Y131.689 E.02419
G1 X118.565 Y132.362 E.02436
G1 X118.834 Y133.079 E.02424
G3 X118.999 Y133.862 I-26.763 J6.06 E.02533
G1 X119.044 Y134.598 E.02334
G1 X118.98 Y135.357 E.0241
G1 X118.824 Y136.026 E.02177
G1 X118.548 Y136.753 E.02463
G1 X118.31 Y137.19 E.01573
G3 X130.249 Y144.067 I-19.802 J48.177 E.43749
G3 X142.39 Y157.025 I-31.645 J41.818 E.56493
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.396 E1.81409
G1 X144.167 Y98.945 E.01428
G1 X142.901 Y99.059 E.04026
G3 X141.272 Y99.064 I-.904 J-27.539 E.05157
G1 X141.964 Y102.8 E.12029
G1 X141.953 Y102.918 E.00376
G1 X141.811 Y103.054 E.00621
G1 X141.742 Y103.065 E.00223
G1 X132.849 Y103.065 E.28155
G1 X132.7 Y103.009 E.00502
G1 X132.653 Y102.951 E.00237
G1 X132.646 Y102.697 E.00806
G1 X133.318 Y99.065 E.11694
G3 X132.123 Y99.026 I-.286 J-9.65 E.03788
G2 X131.038 Y98.936 I-.889 J4.135 E.03459
G3 X122.631 Y116 I-51.1 J-14.573 E.60552
; LINE_WIDTH: 0.544336
G1 X122.208 Y116.606 E.02456
G1 X122.401 Y116.6 E.00642
; LINE_WIDTH: 0.530696
G1 X122.547 Y116.595 E.00473
; LINE_WIDTH: 0.520386
G1 X122.58 Y116.622 E.00134
; LINE_WIDTH: 0.520366
G1 X122.712 Y116.734 E.00548
; LINE_WIDTH: 0.520316
G1 X122.845 Y116.845 E.00548
; LINE_WIDTH: 0.520256
G1 X122.977 Y116.956 E.00548
; LINE_WIDTH: 0.520196
G1 X123.11 Y117.067 E.00548
; LINE_WIDTH: 0.520146
G1 X123.243 Y117.179 E.00548
; LINE_WIDTH: 0.520086
G1 X123.375 Y117.29 E.00548
; LINE_WIDTH: 0.519996
G1 X123.544 Y117.431 E.00697
G1 X123.651 Y117.521 E.00441
G1 X123.757 Y117.61 E.00441
G1 X123.864 Y117.7 E.00441
G1 X123.971 Y117.789 E.00441
G1 X124.078 Y117.879 E.00441
G1 X124.184 Y117.968 E.00441
G3 X124.244 Y118.058 I-.075 J.114 E.00349
G1 X124.289 Y118.147 E.00317
G1 X124.265 Y118.221 E.00248
; WIPE_START
M204 S10000
G1 X123.628 Y118.992 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.111 Y113.682 Z6.52 F36000
G1 X142.852 Y100.376 Z6.52
G1 Z6.12
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.972976
G1 F8389.798
G1 X142.875 Y100.625 E.0153
; LINE_WIDTH: 0.927259
G1 F8819.356
G1 X142.898 Y100.874 E.01455
; LINE_WIDTH: 0.881541
G1 F9295.271
G1 X142.921 Y101.123 E.01381
; LINE_WIDTH: 0.835824
G1 F9825.481
G1 X142.944 Y101.371 E.01306
; LINE_WIDTH: 0.790106
G1 F10419.837
G1 X142.967 Y101.62 E.01232
; LINE_WIDTH: 0.744389
G1 F11090.73
G1 X142.99 Y101.869 E.01157
; LINE_WIDTH: 0.698671
G1 F11853.96
G1 X143.012 Y102.118 E.01083
; LINE_WIDTH: 0.652954
G1 F12730
G1 X143.035 Y102.367 E.01008
; LINE_WIDTH: 0.607236
G1 F13745.855
G1 X143.056 Y102.596 E.00857
; LINE_WIDTH: 0.565256
G1 F14832.745
G1 X143.057 Y103.07 E.01641
; WIPE_START
G1 X143.056 Y102.596 E-.18019
G1 X143.035 Y102.367 E-.08724
G1 X143.012 Y102.118 E-.095
G1 X143.008 Y102.072 E-.01757
; WIPE_END
G1 E-.02 F1800
G1 X135.382 Y102.392 Z6.52 F36000
G1 X131.298 Y102.563 Z6.52
G1 Z6.12
G1 E.4 F1800
; LINE_WIDTH: 1.04298
G1 F7807.543
G1 X131.337 Y102.42 E.00972
; LINE_WIDTH: 1.01866
G1 F8000.447
G1 X131.421 Y102.096 E.02149
; LINE_WIDTH: 0.970103
G1 F8415.56
G1 X131.506 Y101.772 E.02043
; LINE_WIDTH: 0.92155
G1 F8876.107
G1 X131.591 Y101.448 E.01937
; LINE_WIDTH: 0.872996
G1 F9389.979
G1 X131.669 Y101.134 E.01769
; LINE_WIDTH: 0.832896
G1 F9861.502
G1 X131.748 Y100.82 E.01685
; LINE_WIDTH: 0.792796
G1 F10382.882
G1 X131.826 Y100.506 E.016
; LINE_WIDTH: 0.752696
G1 F10962.47
G1 X131.905 Y100.193 E.01516
; WIPE_START
G1 X131.826 Y100.506 E-.12298
G1 X131.748 Y100.82 E-.12297
G1 X131.669 Y101.134 E-.12298
G1 X131.662 Y101.163 E-.01108
; WIPE_END
G1 E-.02 F1800
G1 X127.476 Y107.545 Z6.52 F36000
G1 X119.112 Y120.3 Z6.52
G1 Z6.12
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G2 X122.208 Y116.606 I-864.366 J-727.607 E.16025
; WIPE_START
M204 S10000
G1 X121.566 Y117.372 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X126.861 Y113.737 Z6.52 F36000
G1 Z6.12
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X128.064 Y111.727 I-32.055 J-20.558 E.08945
G2 X129.772 Y112.913 I32.86 J-45.502 E.07939
G3 X130.585 Y114.798 I-1.306 J1.681 E.08179
G3 X127.601 Y119.511 I-6.089 J-.554 E.22114
G1 X126.233 Y120.453 E.06344
G2 X125.42 Y122.339 I1.306 J1.681 E.08179
G2 X128.403 Y127.052 I6.089 J-.554 E.22114
G1 X129.772 Y127.994 E.06344
G3 X130.585 Y129.879 I-1.306 J1.681 E.08179
G3 X127.601 Y134.592 I-6.089 J-.554 E.22114
G1 X126.233 Y135.535 E.06344
G2 X125.42 Y137.42 I1.306 J1.681 E.08179
G1 X125.559 Y138.272 E.03295
G2 X121.389 Y136.115 I-28.482 J49.958 E.17929
G2 X122.937 Y134.592 I-1.477 J-3.05 E.08439
G2 X122.89 Y132.707 I-3.395 J-.858 E.0729
G2 X120.06 Y128.937 I-6.3 J1.781 E.18411
G1 X118.692 Y127.994 E.06344
G3 X117.879 Y126.109 I1.306 J-1.681 E.08179
G3 X118.374 Y124.187 I8.082 J1.056 E.07596
G1 X119.117 Y123.457 E.03976
G2 X120.208 Y124.151 I2.733 J-3.097 E.04957
G1 X141.945 Y143.177 F36000
G1 F13446.283
G1 X141.945 Y140.835 E.08944
G3 X140.655 Y138.362 I4.474 J-3.908 E.10748
G3 X140.608 Y136.477 I3.349 J-1.027 E.0729
G3 X141.945 Y135.1 I2.783 J1.364 E.07453
G1 X141.945 Y125.754 E.35683
G3 X140.655 Y123.281 I4.474 J-3.908 E.10748
G3 X140.608 Y121.396 I3.349 J-1.027 E.0729
G3 X141.945 Y120.018 I2.783 J1.364 E.07453
G1 X141.945 Y110.672 E.35683
G3 X140.501 Y107.257 I4.913 J-4.091 E.14362
G3 X141.438 Y105.287 I2.136 J-.193 E.08738
G1 X137.395 Y105.287 E.15434
G1 X135.944 Y106.315 E.06789
G2 X132.96 Y111.028 I3.105 J5.267 E.22114
G2 X133.773 Y112.913 I2.119 J.204 E.08179
G1 X135.142 Y113.855 E.06344
G3 X138.125 Y118.568 I-3.105 J5.267 E.22114
G3 X137.312 Y120.453 I-2.119 J.204 E.08179
G1 X135.944 Y121.396 E.06344
G2 X132.96 Y126.109 I3.105 J5.267 E.22114
G2 X133.773 Y127.994 I2.119 J.204 E.08179
G1 X135.142 Y128.937 E.06344
G3 X138.125 Y133.65 I-3.105 J5.267 E.22114
G3 X137.312 Y135.535 I-2.119 J.204 E.08179
G1 X135.944 Y136.477 E.06344
G2 X132.96 Y141.19 I3.105 J5.267 E.22114
G2 X133.773 Y143.075 I2.119 J.204 E.08179
G1 X135.142 Y144.018 E.06344
G3 X138.005 Y147.996 I-3.074 J5.233 E.19262
G3 X140.821 Y151.154 I-63.078 J59.081 E.16158
G3 X141.945 Y150.181 I3.077 J2.419 E.05711
G1 X141.945 Y147.838 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 6.28
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y148.838 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L39
M991 S0 P38 ;notify layer change


G17
G3 Z6.52 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.584 Y119.344
G1 Z6.28
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.456 Y119.537 E.00884
G3 X125.149 Y119.902 I-3.329 J-2.479 E.01822
G1 X122.58 Y122.968 E.15272
G1 X122.215 Y123.302 E.01889
G1 X121.724 Y123.553 E.02107
G1 X121.22 Y123.655 E.01963
G1 X120.672 Y123.613 E.02097
G1 X120.38 Y123.525 E.01165
G1 X119.926 Y123.273 E.01981
G1 X119.201 Y122.67 E.036
G3 X114.848 Y126.662 I-38.646 J-37.767 E.22564
G1 X118.015 Y129.012 E.15055
G3 X118.933 Y129.849 I-4.608 J5.97 E.0475
G1 X119.566 Y130.648 E.03893
G1 X120.076 Y131.532 E.03894
G1 X120.45 Y132.48 E.03892
G3 X120.718 Y133.726 I-10.226 J2.847 E.04869
G1 X120.765 Y134.488 E.02914
G1 X120.698 Y135.506 E.03898
G1 X120.538 Y136.268 E.02971
G3 X130.504 Y142.099 I-22.548 J49.971 E.44162
G3 X142.443 Y154.069 I-32.641 J44.497 E.64804
G1 X142.443 Y104.641 E1.88712
G1 X142.346 Y104.696 E.00428
G1 X141.744 Y104.791 E.02324
G1 X132.846 Y104.791 E.33971
G1 X132.543 Y104.767 E.01164
G1 X132.009 Y104.602 E.02134
G1 X131.562 Y104.308 E.02042
G1 X131.21 Y103.889 E.0209
G3 X124.577 Y116.256 I-51.496 J-19.657 E.53728
G1 X125.214 Y116.79 E.03172
G1 X125.477 Y117.058 E.01433
G1 X125.732 Y117.467 E.01841
G1 X125.867 Y117.873 E.01632
G1 X125.911 Y118.318 E.01707
G1 X125.849 Y118.772 E.01748
G1 X125.69 Y119.187 E.01698
G1 X125.634 Y119.27 E.00381
G1 X125.131 Y118.959 F36000
G1 F13446.369
G1 X125.06 Y119.081 E.00541
G3 X124.696 Y119.532 I-2.251 J-1.447 E.02217
G1 X122.126 Y122.597 E.15272
G3 X121.181 Y123.07 I-1.098 J-1.014 E.04124
G1 X120.749 Y123.03 E.01657
G1 X120.276 Y122.803 E.02002
G3 X119.158 Y121.869 I191.555 J-230.381 E.0556
G3 X113.893 Y126.683 I-38.501 J-36.826 E.27258
G1 X117.666 Y129.482 E.17936
G3 X118.524 Y130.268 I-4.172 J5.41 E.04448
G1 X119.1 Y131.003 E.03564
G1 X119.563 Y131.814 E.03565
G1 X119.901 Y132.684 E.03563
G3 X120.135 Y133.792 I-10.247 J2.744 E.04327
G1 X120.18 Y134.523 E.02798
G1 X120.114 Y135.456 E.03569
G1 X119.932 Y136.286 E.03244
G1 X119.836 Y136.599 E.01251
G3 X130.932 Y143.157 I-21.089 J48.353 E.49335
G3 X142.92 Y155.769 I-32.175 J42.586 E.66731
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y114.044 E1.59224
G3 X143.046 Y103.321 I843.895 J-4 E.4094
; LINE_WIDTH: 0.585406
G1 F14290.382
G1 X143.059 Y103.072 E.00896
G1 X143.006 Y103.314 E.0089
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.642 Y103.869 E.02532
G1 X142.165 Y104.139 E.02094
G1 X141.744 Y104.206 E.01626
G1 X132.846 Y104.206 E.33971
G1 X132.634 Y104.189 E.00814
G1 X132.234 Y104.06 E.01603
G1 X131.948 Y103.868 E.01318
G1 F12690.62
G1 X131.664 Y103.521 E.0171
; LINE_WIDTH: 0.669146
G1 F11175.266
G1 X131.609 Y103.397 E.0056
; LINE_WIDTH: 0.718296
G1 F10736.215
G1 X131.555 Y103.273 E.00604
; LINE_WIDTH: 0.767446
G1 F10305.975
G1 X131.5 Y103.15 E.00647
; LINE_WIDTH: 0.816596
G1 F9884.505
G1 X131.445 Y103.026 E.0069
; LINE_WIDTH: 0.865746
G1 F9471.862
G1 X131.391 Y102.902 E.00734
; LINE_WIDTH: 0.914896
G1 F8943.173
G1 X131.336 Y102.778 E.00777
; LINE_WIDTH: 0.958823
G1 F8518.242
G1 X131.32 Y102.716 E.00383
; LINE_WIDTH: 1.00275
G1 F8131.858
G1 X131.304 Y102.655 E.00401
; LINE_WIDTH: 1.04668
G1 F7779.007
G1 X131.289 Y102.593 E.00419
G1 X131.248 Y102.643 E.00426
; LINE_WIDTH: 1.00275
G1 F8131.858
G1 X131.207 Y102.694 E.00408
; LINE_WIDTH: 0.958823
G1 F8518.242
G1 X131.167 Y102.744 E.00389
; LINE_WIDTH: 0.914896
G1 F8943.173
G1 X131.088 Y102.887 E.00939
; LINE_WIDTH: 0.865746
G1 F9471.862
G1 X131.009 Y103.03 E.00886
; LINE_WIDTH: 0.816596
G1 F10066.986
G1 X130.93 Y103.174 E.00834
; LINE_WIDTH: 0.767446
G1 F10581.853
G1 X130.851 Y103.317 E.00782
; LINE_WIDTH: 0.718296
G1 F11109.544
G1 X130.772 Y103.46 E.00729
; LINE_WIDTH: 0.669146
G1 F11650.045
G1 X130.693 Y103.603 E.00677
; LINE_WIDTH: 0.619996
G1 F13026.148
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.209 Y104.835 E.01998
G3 X123.782 Y116.354 I-49.998 J-20.346 E.50487
G1 X124.838 Y117.239 E.0526
G1 X125.022 Y117.426 E.01002
G1 X125.255 Y117.851 E.01851
G1 X125.325 Y118.284 E.01676
G1 X125.252 Y118.726 E.0171
G1 X125.173 Y118.879 E.00656
G1 X124.604 Y118.709 F36000
G1 F13446.369
G1 X124.558 Y118.784 E.00337
G3 X124.251 Y119.15 I-1.996 J-1.361 E.01826
G1 X121.681 Y122.216 E.15272
G1 X121.449 Y122.4 E.01134
G1 X121.142 Y122.486 E.01215
G1 X120.807 Y122.434 E.01296
G1 X120.59 Y122.306 E.0096
G1 X119.114 Y121.068 E.07356
G3 X112.93 Y126.698 I-39.583 J-37.269 E.31961
G1 X117.317 Y129.952 E.20856
G1 X118.114 Y130.687 E.04141
G1 X118.634 Y131.357 E.03236
G1 X119.05 Y132.096 E.03237
G1 X119.352 Y132.887 E.03235
G3 X119.551 Y133.828 I-12.849 J3.205 E.0367
G1 X119.596 Y134.559 E.02798
G1 X119.531 Y135.405 E.0324
G1 X119.362 Y136.15 E.02916
G1 X119.08 Y136.91 E.03095
G3 X130.579 Y143.624 I-20.875 J48.958 E.5097
G3 X142.647 Y156.415 I-31.779 J42.072 E.67457
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.531 E2.16491
G1 X142.966 Y99.61 E.02493
G1 X141.939 Y99.618 E.03922
G1 X142.511 Y102.707 E.11995
G3 X142.473 Y103.114 I-.766 J.135 E.01581
G1 X142.257 Y103.428 E.01454
G1 X141.984 Y103.582 E.01195
G1 X141.744 Y103.62 E.00928
G1 X132.846 Y103.62 E.33971
G1 X132.497 Y103.537 E.01371
G3 X132.172 Y103.229 I.349 J-.696 E.01734
G1 X132.07 Y102.792 E.01715
G1 X132.081 Y102.7 E.00355
G1 X132.652 Y99.616 E.11975
G3 X131.458 Y99.49 I.214 J-7.762 E.04589
G3 X122.984 Y116.45 I-51.62 J-15.194 E.72763
G1 X124.461 Y117.688 E.07359
G1 X124.668 Y117.958 E.01299
G1 X124.74 Y118.284 E.01275
G1 X124.676 Y118.592 E.01199
G1 X124.651 Y118.633 E.00183
M204 S250
G1 X124.134 Y118.429 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.262 Y121.856 E.14156
G1 X121.106 Y121.934 E.00554
G1 X121.015 Y121.905 E.00301
G3 X120.897 Y121.841 I.02 J-.178 E.00434
G1 X120.688 Y121.666 E.00865
G1 X120.478 Y121.49 E.00865
G1 X120.269 Y121.315 E.00865
G1 F3450
G1 X120.06 Y121.139 E.00865
G1 F3300
G1 X119.85 Y120.964 E.00865
G1 F3150
G1 X119.641 Y120.788 E.00865
G1 F3600
G1 X119.508 Y120.677 E.00548
; LINE_WIDTH: 0.520216
G1 X119.487 Y120.66 E.00085
; LINE_WIDTH: 0.520546
G1 X119.455 Y120.633 E.00134
; LINE_WIDTH: 0.520876
G1 X119.422 Y120.606 E.00134
; LINE_WIDTH: 0.521206
G1 X119.39 Y120.58 E.00134
; LINE_WIDTH: 0.521546
G1 X119.357 Y120.553 E.00134
; LINE_WIDTH: 0.521876
G1 X119.324 Y120.526 E.00134
; LINE_WIDTH: 0.522206
G1 X119.292 Y120.499 E.00134
; LINE_WIDTH: 0.522306
G1 X119.282 Y120.491 E.00041
; LINE_WIDTH: 0.530906
G1 X119.246 Y120.376 E.00388
; LINE_WIDTH: 0.543266
G1 X119.195 Y120.211 E.00573
G1 X118.674 Y120.729 E.02438
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.899 J-36.686 E.28369
G1 X116.988 Y130.396 E.19622
G1 X117.728 Y131.083 E.03197
G1 X118.194 Y131.692 E.02426
G1 X118.565 Y132.362 E.02427
G1 X118.834 Y133.08 E.02425
M73 P69 R6
G1 X118.981 Y133.763 E.02214
G1 X119.044 Y134.593 E.02634
G1 X118.98 Y135.358 E.0243
G1 X118.824 Y136.022 E.02161
G1 X118.548 Y136.753 E.02474
G1 X118.31 Y137.19 E.01574
G3 X130.249 Y144.067 I-20.138 J48.759 E.43747
G3 X142.39 Y157.025 I-31.645 J41.817 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.394 E1.81415
G1 X144.167 Y98.945 E.01421
G1 X142.9 Y99.061 E.0403
G1 X141.275 Y99.066 E.05145
G1 X141.967 Y102.802 E.12029
G1 X141.956 Y102.921 E.00376
G1 X141.814 Y103.056 E.00621
G1 X132.846 Y103.067 E.28391
G1 X132.698 Y103.011 E.00503
G1 X132.621 Y102.827 E.00631
G1 X133.316 Y99.067 E.12106
G3 X132.123 Y99.028 I-.285 J-9.479 E.03781
G2 X131.038 Y98.936 I-.89 J4.071 E.03459
G3 X122.631 Y116 I-51.377 J-14.709 E.60548
; LINE_WIDTH: 0.543796
G1 X122.155 Y116.68 E.02757
G1 X122.329 Y116.702 E.00584
; LINE_WIDTH: 0.531216
G1 X122.446 Y116.716 E.00382
; LINE_WIDTH: 0.522786
G1 X122.495 Y116.757 E.00205
; LINE_WIDTH: 0.522646
G1 X122.645 Y116.883 E.00622
; LINE_WIDTH: 0.522256
M73 P69 R5
G1 X122.794 Y117.009 E.00621
; LINE_WIDTH: 0.521856
G1 X122.944 Y117.135 E.00621
; LINE_WIDTH: 0.521456
G1 X123.093 Y117.261 E.0062
; LINE_WIDTH: 0.521066
G1 X123.242 Y117.386 E.0062
; LINE_WIDTH: 0.520666
G1 X123.392 Y117.512 E.00619
; LINE_WIDTH: 0.520266
G1 X123.497 Y117.601 E.00436
; LINE_WIDTH: 0.519996
G1 X123.559 Y117.653 E.00256
G1 X123.647 Y117.727 E.00363
G1 X123.735 Y117.8 E.00363
G1 X123.823 Y117.874 E.00363
G1 X123.91 Y117.947 E.00363
G1 X123.998 Y118.021 E.00363
G1 X124.086 Y118.095 E.00363
G3 X124.141 Y118.181 I-.073 J.107 E.00333
G1 X124.186 Y118.27 E.00319
G1 X124.162 Y118.344 E.00245
; WIPE_START
M204 S10000
G1 X123.526 Y119.115 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.424 Y114.271 Z6.68 F36000
G1 X143.059 Y103.072 Z6.68
G1 Z6.28
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.562736
G1 F14903.483
G1 X143.058 Y102.598 E.01632
; LINE_WIDTH: 0.604966
G1 F13800.537
G1 X143.036 Y102.368 E.00859
; LINE_WIDTH: 0.650683
G1 F12776.91
G1 X143.014 Y102.119 E.01004
; LINE_WIDTH: 0.696399
G1 F11894.649
G1 X142.991 Y101.87 E.01079
; LINE_WIDTH: 0.742115
G1 F11126.359
G1 X142.968 Y101.621 E.01154
; LINE_WIDTH: 0.787831
G1 F10451.298
G1 X142.945 Y101.372 E.01228
; LINE_WIDTH: 0.833547
G1 F9853.465
G1 X142.922 Y101.123 E.01302
; LINE_WIDTH: 0.879264
G1 F9320.327
G1 X142.899 Y100.874 E.01377
; LINE_WIDTH: 0.92498
G1 F8841.92
G1 X142.876 Y100.626 E.01452
; LINE_WIDTH: 0.970696
G1 F8410.228
G1 X142.854 Y100.377 E.01526
; WIPE_START
G1 X142.876 Y100.626 E-.095
G1 X142.899 Y100.874 E-.095
G1 X142.922 Y101.123 E-.095
G1 X142.945 Y101.372 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.354 Y102.167 Z6.68 F36000
G1 X131.289 Y102.593 Z6.68
G1 Z6.28
G1 E.4 F1800
; LINE_WIDTH: 1.04668
G1 F7779.007
G1 X131.299 Y102.555 E.00262
; LINE_WIDTH: 1.04028
G1 F7828.499
G1 X131.369 Y102.287 E.01816
; LINE_WIDTH: 0.999181
G1 F8161.933
G1 X131.439 Y102.019 E.01742
; LINE_WIDTH: 0.958086
G1 F8525.035
G1 X131.51 Y101.751 E.01668
; LINE_WIDTH: 0.916991
G1 F8921.947
G1 X131.58 Y101.484 E.01593
; LINE_WIDTH: 0.875896
G1 F9357.622
G1 X131.59 Y101.448 E.00201
; LINE_WIDTH: 0.870916
G1 F9413.326
G1 X131.668 Y101.134 E.01764
; LINE_WIDTH: 0.830826
G1 F9887.131
G1 X131.747 Y100.82 E.0168
; LINE_WIDTH: 0.790736
G1 F10411.159
G1 X131.825 Y100.507 E.01595
; LINE_WIDTH: 0.750646
G1 F10993.845
G1 X131.904 Y100.193 E.01511
; WIPE_START
G1 X131.825 Y100.507 E-.12293
G1 X131.747 Y100.82 E-.12293
G1 X131.668 Y101.134 E-.12293
G1 X131.661 Y101.163 E-.01121
; WIPE_END
G1 E-.02 F1800
G1 X127.673 Y107.67 Z6.68 F36000
G1 X122.145 Y116.69 Z6.68
G1 Z6.28
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G2 X119.21 Y120.19 I664.202 J560.033 E.15187
; WIPE_START
M204 S10000
G1 X119.852 Y119.423 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.295 Y124.016 Z6.68 F36000
G1 Z6.28
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X119.231 Y123.343 I1.397 J-3.383 E.0483
G1 X118.465 Y124.096 E.04101
G2 X118.042 Y126.109 I6.012 J2.316 E.07886
G2 X118.978 Y127.994 I2.432 J-.033 E.0831
G1 X120.233 Y128.937 E.05993
G3 X122.52 Y131.764 I-3.773 J5.39 E.14068
G3 X122.689 Y134.592 I-3.997 J1.658 E.11019
G3 X121.256 Y136.052 I-3.517 J-2.018 E.07898
G3 X125.67 Y138.332 I-25.067 J53.963 E.18973
G1 X125.583 Y137.42 E.03497
G3 X126.519 Y135.535 I2.432 J.033 E.0831
G1 X127.774 Y134.592 E.05993
G2 X130.422 Y129.879 I-3.298 J-4.953 E.21425
G2 X129.486 Y127.994 I-2.432 J.033 E.0831
G1 X128.231 Y127.052 E.05993
G3 X125.583 Y122.339 I3.298 J-4.953 E.21425
G3 X126.519 Y120.453 I2.432 J.033 E.0831
G1 X127.774 Y119.511 E.05993
G2 X130.422 Y114.798 I-3.298 J-4.953 E.21425
G2 X129.486 Y112.913 I-2.432 J.033 E.0831
G3 X128.019 Y111.803 I22.041 J-30.654 E.07023
G3 X126.813 Y113.811 I-33.233 J-18.585 E.08945
G1 X141.945 Y113.023 F36000
G1 F13446.283
G1 X141.945 Y110.681 E.08944
G3 X140.664 Y107.257 I4.02 J-3.456 E.14248
G3 X141.711 Y105.289 I2.523 J.079 E.08815
G1 X137.107 Y105.289 E.17574
G1 X135.771 Y106.315 E.06431
G2 X133.123 Y111.028 I3.298 J4.953 E.21425
G2 X134.059 Y112.913 I2.432 J-.033 E.0831
G1 X135.314 Y113.855 E.05993
G3 X137.962 Y118.568 I-3.298 J4.953 E.21425
G3 X137.026 Y120.453 I-2.432 J-.033 E.0831
G1 X135.771 Y121.396 E.05993
G2 X133.123 Y126.109 I3.298 J4.953 E.21425
G2 X134.059 Y127.994 I2.432 J-.033 E.0831
G1 X135.314 Y128.937 E.05993
G3 X137.962 Y133.65 I-3.298 J4.953 E.21425
G3 X137.026 Y135.535 I-2.432 J-.033 E.0831
G1 X135.771 Y136.477 E.05993
G2 X133.123 Y141.19 I3.298 J4.953 E.21425
G2 X134.059 Y143.075 I2.432 J-.033 E.0831
G1 X135.314 Y144.018 E.05993
G3 X137.872 Y147.788 I-3.62 J5.208 E.17792
G1 X137.879 Y147.864 E.0029
G3 X140.981 Y151.351 I-41.035 J39.633 E.17823
G3 X141.945 Y150.356 I2.957 J1.903 E.05322
G1 X141.945 Y140.843 E.36321
G3 X140.664 Y137.42 I4.02 J-3.456 E.14248
G3 X141.945 Y135.275 I2.609 J.104 E.09946
G1 X141.945 Y125.762 E.36321
G3 X140.664 Y122.339 I4.02 J-3.457 E.14248
G3 X141.945 Y120.194 I2.609 J.104 E.09946
G1 X141.945 Y117.851 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 6.44
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y118.851 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L40
M991 S0 P39 ;notify layer change


G17
G3 Z6.68 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.302 Y119.715
G1 Z6.44
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.257 Y119.774 E.00283
G1 X122.688 Y122.839 E.15272
G1 X122.298 Y123.193 E.02012
G1 X121.805 Y123.438 E.021
G1 X121.302 Y123.533 E.01955
G1 X120.76 Y123.487 E.02078
G1 X120.558 Y123.43 E.00801
G1 X120.09 Y123.194 E.02
G3 X119.313 Y122.554 I30.367 J-37.685 E.03843
G3 X114.848 Y126.662 I-38.568 J-37.439 E.23177
G1 X118.015 Y129.012 E.15056
G3 X118.932 Y129.849 I-4.608 J5.969 E.04747
G1 X119.567 Y130.649 E.03896
G1 X120.076 Y131.531 E.0389
G1 X120.45 Y132.479 E.03892
G3 X120.718 Y133.726 I-10.238 J2.85 E.04871
G1 X120.765 Y134.488 E.02913
G1 X120.698 Y135.504 E.03889
G1 X120.538 Y136.268 E.0298
G3 X130.504 Y142.098 I-22.536 J49.953 E.44162
G3 X142.443 Y154.07 I-32.633 J44.486 E.64807
G1 X142.443 Y104.644 E1.88701
G1 X142.348 Y104.698 E.00417
G1 X141.747 Y104.793 E.02324
G1 X132.844 Y104.793 E.3399
G1 X132.536 Y104.769 E.0118
G1 X131.891 Y104.544 E.02609
G1 X131.211 Y103.886 E.03611
G3 X124.483 Y116.386 I-51.106 J-19.449 E.54353
G1 X125.111 Y116.913 E.0313
G1 X125.433 Y117.257 E.01798
G1 X125.657 Y117.655 E.01744
G1 X125.79 Y118.145 E.01938
G3 X125.515 Y119.436 I-1.992 J.25 E.05136
G1 X125.357 Y119.643 E.00995
G1 X124.815 Y119.389 F36000
G1 F13446.369
G1 X124.808 Y119.397 E.00039
G1 X122.239 Y122.463 E.15272
G3 X121.269 Y122.948 I-1.104 J-.994 E.04238
G1 X120.816 Y122.897 E.01742
G1 X120.421 Y122.711 E.01666
G3 X119.269 Y121.754 I61.075 J-74.66 E.0572
G3 X113.893 Y126.683 I-38.594 J-36.695 E.2787
G1 X117.666 Y129.482 E.17937
G3 X118.523 Y130.268 I-4.173 J5.411 E.04445
G1 X119.1 Y131.003 E.03567
G1 X119.563 Y131.813 E.03561
G1 X119.901 Y132.683 E.03564
G3 X120.135 Y133.792 I-10.252 J2.745 E.0433
G1 X120.18 Y134.523 E.02796
G1 X120.115 Y135.454 E.03561
G1 X119.933 Y136.284 E.03244
G1 X119.835 Y136.599 E.01259
G3 X130.932 Y143.156 I-21.093 J48.359 E.49333
G3 X142.92 Y155.769 I-32.177 J42.589 E.66733
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y110.045 E1.74492
G3 X143.047 Y103.323 I507.216 J-2 E.25665
; LINE_WIDTH: 0.583916
G1 F14329.126
G1 X143.06 Y103.074 E.00893
G1 X143.007 Y103.316 E.00887
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.645 Y103.871 E.02531
G1 X142.168 Y104.141 E.02094
G1 X141.747 Y104.208 E.01626
G1 X132.844 Y104.208 E.3399
G1 X132.628 Y104.191 E.00825
G1 X132.152 Y104.019 E.01933
G1 X131.939 Y103.803 E.01158
G1 F12382.381
G1 X131.658 Y103.518 E.01527
; LINE_WIDTH: 0.666627
G1 F11041.675
G1 X131.618 Y103.413 E.00466
; LINE_WIDTH: 0.713258
G1 F10676.437
G1 X131.577 Y103.307 E.00501
; LINE_WIDTH: 0.75989
G1 F10317.342
G1 X131.537 Y103.201 E.00535
; LINE_WIDTH: 0.806521
G1 F9964.361
G1 X131.497 Y103.096 E.00569
; LINE_WIDTH: 0.853152
G1 F9617.552
G1 X131.456 Y102.99 E.00604
; LINE_WIDTH: 0.899783
G1 F9099.348
G1 X131.416 Y102.884 E.00638
; LINE_WIDTH: 0.946414
G1 F8634.133
G1 X131.375 Y102.779 E.00673
; LINE_WIDTH: 0.993045
G1 F8214.173
G1 X131.335 Y102.673 E.00707
; LINE_WIDTH: 1.03968
G1 F7833.171
G1 X131.294 Y102.567 E.00741
G1 X131.228 Y102.682 E.00869
; LINE_WIDTH: 0.993045
G1 F8214.173
G1 X131.161 Y102.797 E.00829
; LINE_WIDTH: 0.946414
G1 F8634.133
G1 X131.094 Y102.911 E.00788
; LINE_WIDTH: 0.899783
G1 F9099.348
G1 X131.028 Y103.026 E.00748
; LINE_WIDTH: 0.853152
G1 F9617.552
G1 X130.961 Y103.14 E.00708
; LINE_WIDTH: 0.806521
G1 F10198.341
G1 X130.894 Y103.255 E.00668
; LINE_WIDTH: 0.75989
G1 F10617.515
G1 X130.828 Y103.37 E.00627
; LINE_WIDTH: 0.713258
G1 F11045.131
G1 X130.761 Y103.484 E.00587
; LINE_WIDTH: 0.666627
G1 F11481.143
G1 X130.694 Y103.599 E.00547
; LINE_WIDTH: 0.619996
G1 F12847.513
G1 X130.549 Y103.972 E.01527
G1 F13446.369
G1 X130.403 Y104.344 E.01527
G1 X130.275 Y104.667 E.01326
G3 X123.687 Y116.483 I-50.712 J-20.528 E.51785
G1 X124.735 Y117.362 E.05219
G1 X124.96 Y117.602 E.01258
G1 X125.164 Y118.013 E.01752
G1 X125.222 Y118.436 E.01632
G3 X124.959 Y119.206 I-1.574 J-.108 E.03139
G1 X124.871 Y119.319 E.00549
G1 X124.356 Y119.025 F36000
G1 F13446.369
G1 X121.79 Y122.087 E.15251
G1 X121.543 Y122.282 E.01201
G1 X121.237 Y122.364 E.01212
G1 X120.94 Y122.322 E.01145
G1 X120.693 Y122.183 E.01081
G1 X119.223 Y120.951 E.07326
G3 X112.93 Y126.698 I-39.768 J-37.225 E.32572
G1 X117.317 Y129.953 E.20857
G1 X118.114 Y130.687 E.04138
G1 X118.634 Y131.357 E.03239
G1 X119.049 Y132.095 E.03233
G1 X119.352 Y132.887 E.03236
G3 X119.551 Y133.828 I-12.851 J3.206 E.03673
G1 X119.596 Y134.559 E.02796
G1 X119.531 Y135.403 E.03233
G1 X119.363 Y136.148 E.02916
G1 X119.08 Y136.91 E.03102
G3 X130.578 Y143.623 I-20.874 J48.957 E.50968
G3 X142.648 Y156.415 I-31.78 J42.074 E.67459
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.532 E2.16488
G1 X142.966 Y99.612 E.02494
G1 X141.942 Y99.62 E.03912
G1 X142.514 Y102.709 E.11995
G3 X142.476 Y103.117 I-.767 J.135 E.01581
G1 X142.259 Y103.43 E.01454
G1 X141.987 Y103.584 E.01195
G1 X141.747 Y103.622 E.00928
G1 X132.844 Y103.622 E.3399
G1 X132.463 Y103.523 E.01502
G1 X132.167 Y103.229 E.01593
G1 X132.078 Y102.707 E.02021
G1 X132.65 Y99.617 E.11996
G3 X131.455 Y99.494 I.265 J-8.42 E.04587
G3 X122.888 Y116.578 I-51.568 J-15.173 E.73355
G1 X124.359 Y117.81 E.07327
G1 X124.577 Y118.107 E.01404
G1 X124.637 Y118.424 E.01233
G3 X124.415 Y118.958 I-1.014 J-.108 E.02237
M204 S250
G1 X123.929 Y118.675 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.363 Y121.736 E.12645
G1 X121.206 Y121.812 E.00552
G1 X121.117 Y121.782 E.00297
G3 X121.005 Y121.723 I.019 J-.171 E.00411
G1 X120.816 Y121.565 E.00779
G1 X120.628 Y121.407 E.0078
G1 X120.439 Y121.249 E.00779
G1 F3450
G1 X120.25 Y121.09 E.0078
G1 F3300
G1 X120.061 Y120.932 E.0078
G1 F3150
G1 X119.873 Y120.774 E.0078
G1 F3600
G1 X119.74 Y120.662 E.0055
; LINE_WIDTH: 0.520246
G1 X119.704 Y120.633 E.00147
; LINE_WIDTH: 0.520586
G1 X119.653 Y120.591 E.00208
; LINE_WIDTH: 0.520926
G1 X119.603 Y120.549 E.00209
; LINE_WIDTH: 0.521266
G1 X119.552 Y120.507 E.00209
; LINE_WIDTH: 0.521606
G1 X119.501 Y120.465 E.00209
; LINE_WIDTH: 0.521946
G1 X119.451 Y120.423 E.00209
; LINE_WIDTH: 0.522286
G1 X119.4 Y120.381 E.00209
; LINE_WIDTH: 0.522386
G1 X119.385 Y120.368 E.00064
; LINE_WIDTH: 0.528396
G1 X119.345 Y120.309 E.0023
; LINE_WIDTH: 0.537056
G1 X119.289 Y120.222 E.00338
; LINE_WIDTH: 0.544336
G1 X119.313 Y120.067 E.00523
G1 X118.674 Y120.729 E.0306
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-39.043 J-36.846 E.28369
G1 X116.988 Y130.397 E.19623
G1 X117.728 Y131.083 E.03195
G1 X118.194 Y131.692 E.02428
G1 X118.565 Y132.362 E.02424
G1 X118.834 Y133.079 E.02426
G1 X118.981 Y133.764 E.02216
G1 X119.044 Y134.593 E.02633
G1 X118.98 Y135.356 E.02424
G1 X118.825 Y136.021 E.02161
G1 X118.548 Y136.752 E.02477
G1 X118.31 Y137.19 E.01576
G3 X130.248 Y144.067 I-20.138 J48.759 E.43745
G3 X142.39 Y157.025 I-31.644 J41.817 E.56494
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.392 E1.81421
G1 X144.167 Y98.945 E.01415
G1 X142.899 Y99.063 E.04034
G1 X141.277 Y99.069 E.05134
G1 X141.969 Y102.804 E.12029
G1 X141.958 Y102.923 E.00376
G1 X141.816 Y103.058 E.00621
G1 X132.844 Y103.069 E.28407
G3 X132.648 Y102.955 I0 J-.226 E.00753
G1 X132.642 Y102.696 E.00821
G1 X133.313 Y99.069 E.11677
G3 X132.123 Y99.029 I-.283 J-9.323 E.03774
G2 X131.038 Y98.936 I-.89 J4.012 E.03458
G3 X122.631 Y116 I-50.903 J-14.475 E.60555
; LINE_WIDTH: 0.520636
G1 X122.297 Y116.444 E.01761
; LINE_WIDTH: 0.544336
G1 X122.004 Y116.853 E.01671
G1 X122.194 Y116.773 E.00685
; LINE_WIDTH: 0.534696
G1 X122.278 Y116.812 E.00302
; LINE_WIDTH: 0.526666
G1 X122.342 Y116.84 E.00224
; LINE_WIDTH: 0.520636
G1 X122.358 Y116.854 E.00065
; LINE_WIDTH: 0.520606
G1 X122.421 Y116.907 E.00262
; LINE_WIDTH: 0.520516
G1 X122.484 Y116.96 E.00262
; LINE_WIDTH: 0.520426
G1 X122.547 Y117.013 E.00262
; LINE_WIDTH: 0.520336
G1 X122.611 Y117.066 E.00262
; LINE_WIDTH: 0.520236
G1 X122.674 Y117.119 E.00262
; LINE_WIDTH: 0.520146
G1 X122.737 Y117.173 E.00262
; LINE_WIDTH: 0.519996
G1 X122.906 Y117.314 E.00697
G1 F3150
G1 X123.082 Y117.462 E.00728
G1 F3300
G1 X123.258 Y117.609 E.00728
G1 F3450
G1 X123.434 Y117.757 E.00728
G1 F3600
G1 X123.611 Y117.905 E.00728
G1 X123.787 Y118.053 E.00728
G1 X123.963 Y118.2 E.00728
G3 X124.038 Y118.304 I-.082 J.139 E.00417
G1 X124.084 Y118.394 E.0032
G3 X123.992 Y118.61 I-.424 J-.053 E.00753
; WIPE_START
M204 S10000
G1 X123.348 Y119.375 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X129.23 Y114.511 Z6.84 F36000
G1 X143.06 Y103.074 Z6.84
G1 Z6.44
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.560196
G1 F14975.469
G1 X143.059 Y102.601 E.01623
; LINE_WIDTH: 0.602686
G1 F13855.9
G1 X143.038 Y102.369 E.00861
; LINE_WIDTH: 0.648403
G1 F12824.35
G1 X143.015 Y102.12 E.01001
; LINE_WIDTH: 0.694119
G1 F11935.753
G1 X142.992 Y101.871 E.01075
; LINE_WIDTH: 0.739835
G1 F11162.316
G1 X142.969 Y101.622 E.0115
; LINE_WIDTH: 0.785551
G1 F10483.018
G1 X142.946 Y101.373 E.01224
; LINE_WIDTH: 0.831267
G1 F9881.657
G1 X142.923 Y101.124 E.01299
; LINE_WIDTH: 0.876984
G1 F9345.546
G1 X142.9 Y100.875 E.01373
; LINE_WIDTH: 0.9227
G1 F8864.613
G1 X142.878 Y100.626 E.01448
; LINE_WIDTH: 0.968416
G1 F8430.756
G1 X142.855 Y100.377 E.01522
; WIPE_START
G1 X142.878 Y100.626 E-.095
G1 X142.9 Y100.875 E-.095
G1 X142.923 Y101.124 E-.095
G1 X142.946 Y101.373 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.353 Y102.151 Z6.84 F36000
G1 X131.294 Y102.567 Z6.84
G1 Z6.44
G1 E.4 F1800
; LINE_WIDTH: 1.03968
G1 F7833.171
G1 X131.335 Y102.42 E.01004
; LINE_WIDTH: 1.01444
G1 F8034.895
G1 X131.419 Y102.096 E.02139
; LINE_WIDTH: 0.96589
G1 F8453.622
G1 X131.504 Y101.772 E.02033
; LINE_WIDTH: 0.917343
G1 F8918.393
G1 X131.589 Y101.448 E.01927
; LINE_WIDTH: 0.868796
G1 F9437.242
G1 X131.667 Y101.134 E.01759
; LINE_WIDTH: 0.828731
G1 F9913.205
G1 X131.746 Y100.82 E.01674
; LINE_WIDTH: 0.788666
G1 F10439.729
G1 X131.824 Y100.507 E.0159
; LINE_WIDTH: 0.748601
G1 F11025.321
G1 X131.902 Y100.193 E.01506
; WIPE_START
G1 X131.824 Y100.507 E-.12287
G1 X131.746 Y100.82 E-.12287
G1 X131.667 Y101.134 E-.12287
G1 X131.66 Y101.163 E-.01139
; WIPE_END
G1 E-.02 F1800
G1 X127.66 Y107.663 Z6.84 F36000
G1 X122.004 Y116.853 Z6.84
G1 Z6.44
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X119.313 Y120.067 E.13938
; WIPE_START
M204 S10000
G1 X119.955 Y119.3 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.342 Y123.876 Z6.84 F36000
G1 Z6.44
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X119.346 Y123.227 I.939 J-2.531 E.04573
G1 X118.529 Y124.033 E.0438
G2 X118.224 Y125.166 I2.955 J1.404 E.04506
M73 P70 R5
G2 X118.764 Y127.523 I2.924 J.569 E.09505
G2 X120.937 Y129.408 I24.35 J-25.872 E.10986
G3 X122.648 Y134.121 I-3.471 J3.927 E.1996
G3 X121.131 Y135.994 I-3.771 J-1.503 E.09344
G3 X125.771 Y138.39 I-41.567 J86.176 E.1994
G3 X126.305 Y136.006 I2.931 J-.595 E.0961
G3 X127.931 Y134.592 I6.625 J5.978 E.08248
G2 X130.24 Y130.822 I-3.715 J-4.866 E.17269
G2 X129.699 Y128.465 I-2.924 J-.569 E.09505
G2 X128.073 Y127.052 I-6.625 J5.978 E.08248
G3 X125.764 Y123.281 I3.715 J-4.866 E.17269
G3 X126.305 Y120.925 I2.924 J-.569 E.09505
G3 X127.931 Y119.511 I6.625 J5.978 E.08248
G2 X130.24 Y115.741 I-3.715 J-4.866 E.17269
G2 X129.699 Y113.384 I-2.924 J-.569 E.09505
G2 X127.974 Y111.885 I-15.363 J15.937 E.08731
G3 X126.763 Y113.89 I-71.773 J-41.955 E.08944
G1 X141.945 Y143.248 F36000
G1 F13446.283
G1 X141.945 Y140.906 E.08944
G3 X140.846 Y138.362 I4.578 J-3.489 E.10683
G3 X141.945 Y135.453 I2.895 J-.568 E.12508
G1 X141.945 Y125.824 E.36762
G3 X140.846 Y123.281 I4.578 J-3.489 E.10683
G3 X141.945 Y120.372 I2.895 J-.568 E.12508
G1 X141.945 Y110.743 E.36762
G3 X140.846 Y108.2 I4.578 J-3.489 E.10683
G3 X141.945 Y105.291 I2.895 J-.568 E.12508
G1 X136.861 Y105.291 E.19413
G1 X135.614 Y106.315 E.0616
G2 X133.305 Y110.085 I3.715 J4.866 E.17269
G2 X133.846 Y112.442 I2.924 J.569 E.09505
G2 X135.472 Y113.855 I6.624 J-5.977 E.08248
G3 X137.781 Y117.626 I-3.715 J4.866 E.17269
G3 X137.24 Y119.982 I-2.924 J.569 E.09505
G3 X135.614 Y121.396 I-6.625 J-5.978 E.08248
G2 X133.305 Y125.166 I3.715 J4.866 E.17269
G2 X133.846 Y127.523 I2.924 J.569 E.09505
G2 X135.472 Y128.937 I6.624 J-5.977 E.08248
G3 X137.781 Y132.707 I-3.715 J4.866 E.17269
G3 X137.24 Y135.063 I-2.924 J.569 E.09505
G3 X135.614 Y136.477 I-6.625 J-5.978 E.08248
G2 X133.305 Y140.248 I3.715 J4.866 E.17269
G2 X133.846 Y142.604 I2.924 J.569 E.09505
G2 X135.472 Y144.018 I6.624 J-5.977 E.08248
G3 X137.774 Y147.76 I-3.707 J4.86 E.17156
G3 X141.108 Y151.511 I-73.626 J68.786 E.19163
G3 X141.945 Y150.534 I3.671 J2.302 E.04931
G1 X141.945 Y148.192 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 6.6
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y149.192 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L41
M991 S0 P40 ;notify layer change


G17
G3 Z6.84 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.465 Y119.447
G1 Z6.6
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.403 Y119.555 E.00476
G3 X125.25 Y119.782 I-1.949 J-1.153 E.01046
G1 X122.79 Y122.716 E.14617
G1 X122.442 Y123.041 E.01818
G1 X122.029 Y123.271 E.01803
G1 X121.534 Y123.399 E.01956
G1 X121.033 Y123.396 E.01911
G1 X120.733 Y123.331 E.01171
G1 X120.281 Y123.129 E.01891
G3 X119.424 Y122.439 I12.128 J-15.946 E.04202
G3 X114.848 Y126.662 I-40.167 J-38.925 E.23788
G1 X118.021 Y129.016 E.15085
G3 X118.932 Y129.849 I-4.632 J5.983 E.04718
G1 X119.566 Y130.648 E.03893
G1 X120.076 Y131.531 E.03894
G1 X120.451 Y132.48 E.03896
G3 X120.718 Y133.725 I-10.119 J2.824 E.04862
G1 X120.765 Y134.487 E.02917
G1 X120.698 Y135.505 E.03893
G1 X120.538 Y136.268 E.02977
G3 X129.713 Y141.523 I-22.022 J49.083 E.40431
G3 X142.443 Y154.07 I-31.968 J45.169 E.68542
G1 X142.443 Y104.648 E1.88688
G1 X142.351 Y104.7 E.00406
G1 X141.749 Y104.795 E.02324
G1 X132.841 Y104.795 E.34009
G1 X132.537 Y104.771 E.01167
G1 X132.002 Y104.605 E.02138
G1 X131.555 Y104.31 E.02046
G1 X131.208 Y103.895 E.02066
G3 X124.388 Y116.516 I-51.061 J-19.438 E.54928
G1 X125.008 Y117.035 E.03089
G1 X125.359 Y117.42 E.01986
G1 X125.579 Y117.84 E.01811
G1 X125.689 Y118.278 E.01724
G1 X125.695 Y118.732 E.01734
G3 X125.586 Y119.166 I-2.241 J-.33 E.01711
G1 X125.501 Y119.364 E.00825
G1 X124.925 Y119.206 F36000
G1 F13446.369
G1 X124.86 Y119.331 E.00537
G1 X124.801 Y119.406 E.00363
G1 X122.342 Y122.34 E.14618
G1 X122.007 Y122.628 E.01686
G3 X121.358 Y122.827 I-.711 J-1.164 E.02617
G1 X120.902 Y122.77 E.01754
G1 X120.586 Y122.629 E.01323
G3 X119.38 Y121.638 I22.357 J-28.44 E.05959
G3 X113.893 Y126.683 I-39.95 J-37.942 E.28481
G1 X117.671 Y129.486 E.17959
G1 X118.523 Y130.268 E.04418
G1 X119.1 Y131.002 E.03564
G1 X119.563 Y131.814 E.03566
G1 X119.901 Y132.684 E.03567
G3 X120.135 Y133.79 I-10.108 J2.714 E.04319
G1 X120.18 Y134.523 E.02802
G1 X120.114 Y135.454 E.03564
G1 X119.932 Y136.286 E.03252
G1 X119.836 Y136.599 E.01248
G3 X130.159 Y142.572 I-21.554 J49.159 E.45631
G3 X142.92 Y155.769 I-31.897 J43.613 E.70429
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y108.046 E1.82124
G3 X143.048 Y103.324 I216.315 J-1.5 E.18027
; LINE_WIDTH: 0.582416
G1 F14352.939
G1 X143.061 Y103.076 E.0089
G1 F14349.578
G1 X143.009 Y103.317 E.00884
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.647 Y103.873 E.02531
G1 X142.17 Y104.143 E.02094
G1 X141.749 Y104.21 E.01626
G1 X132.841 Y104.21 E.34009
G1 X132.628 Y104.193 E.00816
G1 X132.228 Y104.064 E.01607
G1 X131.941 Y103.87 E.01321
G1 F13355.702
G1 X131.659 Y103.526 E.01698
; LINE_WIDTH: 0.669386
G1 F11810.504
G1 X131.47 Y103.062 E.02075
; LINE_WIDTH: 0.715875
G1 F10183.269
G1 X131.447 Y103.002 E.00284
; LINE_WIDTH: 0.762364
G1 F9984.192
G1 X131.424 Y102.943 E.00304
; LINE_WIDTH: 0.808852
G1 F9787.079
G1 X131.402 Y102.883 E.00323
; LINE_WIDTH: 0.855341
G1 F9591.904
G1 X131.379 Y102.823 E.00342
; LINE_WIDTH: 0.90183
G1 F9077.875
G1 X131.357 Y102.763 E.00362
; LINE_WIDTH: 0.948319
G1 F8616.14
G1 X131.334 Y102.703 E.00381
; LINE_WIDTH: 0.994807
G1 F8199.101
G1 X131.311 Y102.643 E.00401
; LINE_WIDTH: 1.0413
G1 F7820.569
G1 X131.289 Y102.584 E.0042
G1 X131.247 Y102.635 E.00433
; LINE_WIDTH: 0.997115
G1 F8179.449
G1 X131.205 Y102.686 E.00413
; LINE_WIDTH: 0.952934
G1 F8572.852
G1 X131.164 Y102.737 E.00395
; LINE_WIDTH: 0.908753
G1 F9006.009
G1 X131.122 Y102.788 E.00376
; LINE_WIDTH: 0.864571
G1 F9485.267
G1 X131.081 Y102.839 E.00357
; LINE_WIDTH: 0.82039
G1 F10018.4
G1 X131.039 Y102.89 E.00338
; LINE_WIDTH: 0.776209
G1 F10615.032
G1 X130.998 Y102.941 E.00319
; LINE_WIDTH: 0.732028
G1 F10826.361
G1 X130.956 Y102.992 E.003
; LINE_WIDTH: 0.687846
G1 F11931.628
G1 X130.824 Y103.3 E.01425
; LINE_WIDTH: 0.653921
G1 F12710.122
G1 X130.691 Y103.607 E.01351
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X130.545 Y103.979 E.01527
G1 X130.207 Y104.834 E.0351
G3 X123.593 Y116.613 I-50.683 J-20.713 E.51708
G1 X124.632 Y117.484 E.05177
G1 X124.894 Y117.777 E.015
G1 X125.072 Y118.173 E.01659
G1 X125.119 Y118.565 E.01507
G1 X125.044 Y118.978 E.01601
G1 X124.966 Y119.127 E.00642
G1 X124.398 Y118.956 F36000
G1 F13446.369
G1 X124.352 Y119.03 E.00331
G1 X121.893 Y121.964 E.14617
G1 X121.638 Y122.163 E.01235
G1 X121.332 Y122.242 E.01208
G1 X121.071 Y122.209 E.01001
G1 X120.796 Y122.06 E.01195
G1 X119.332 Y120.833 E.07295
G3 X112.93 Y126.698 I-41.574 J-38.956 E.3318
G1 X117.32 Y129.955 E.20871
G1 X118.114 Y130.687 E.04124
G1 X118.634 Y131.357 E.03235
G1 X119.049 Y132.096 E.03237
G1 X119.352 Y132.888 E.03238
G3 X119.551 Y133.826 I-12.591 J3.151 E.03662
G1 X119.596 Y134.559 E.02802
G1 X119.531 Y135.404 E.03236
G1 X119.362 Y136.151 E.02924
G1 X119.08 Y136.91 E.03092
G3 X129.814 Y143.046 I-20.502 J48.327 E.47314
G3 X142.647 Y156.415 I-31.729 J43.299 E.71109
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.533 E2.16486
G1 X142.966 Y99.614 E.02494
G1 X141.944 Y99.622 E.03903
G1 X142.516 Y102.711 E.11995
G3 X142.478 Y103.119 I-.766 J.135 E.01581
G1 X142.262 Y103.432 E.01454
G1 X141.989 Y103.586 E.01195
G1 X141.749 Y103.624 E.00928
G1 X132.841 Y103.624 E.34009
G1 X132.491 Y103.541 E.01374
G3 X132.167 Y103.234 I.35 J-.695 E.01729
G1 X132.071 Y102.955 E.01126
G1 X132.076 Y102.704 E.0096
G1 X132.647 Y99.619 E.11977
G3 X131.454 Y99.49 I.225 J-7.662 E.04586
G3 X122.791 Y116.705 I-51.436 J-15.097 E.7398
G1 X124.256 Y117.933 E.07296
G1 X124.478 Y118.239 E.01442
G1 X124.534 Y118.55 E.01208
G1 X124.47 Y118.84 E.01134
G1 X124.445 Y118.879 E.00178
M204 S250
G1 X123.929 Y118.675 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.466 Y121.612 E.12137
G1 X121.306 Y121.689 E.00561
G1 X121.219 Y121.66 E.00293
G3 X121.112 Y121.604 I.018 J-.165 E.0039
G1 X120.942 Y121.461 E.00703
G1 X120.772 Y121.319 E.00703
G1 X120.602 Y121.176 E.00703
G1 F3450
G1 X120.432 Y121.034 E.00703
G1 F3300
G1 X120.262 Y120.891 E.00703
G1 F3150
G1 X120.092 Y120.749 E.00703
G1 F3600
G1 X119.972 Y120.648 E.00495
; LINE_WIDTH: 0.520276
G1 X119.923 Y120.608 E.002
; LINE_WIDTH: 0.520676
G1 X119.854 Y120.551 E.00284
; LINE_WIDTH: 0.521076
G1 X119.785 Y120.493 E.00284
; LINE_WIDTH: 0.521466
G1 X119.716 Y120.436 E.00285
; LINE_WIDTH: 0.521866
G1 X119.647 Y120.379 E.00285
; LINE_WIDTH: 0.522256
G1 X119.578 Y120.322 E.00285
; LINE_WIDTH: 0.522656
G1 X119.509 Y120.264 E.00285
; LINE_WIDTH: 0.522776
G1 X119.487 Y120.246 E.00091
; LINE_WIDTH: 0.525566
G1 X119.467 Y120.228 E.00085
; LINE_WIDTH: 0.529666
G1 X119.438 Y120.202 E.00127
; LINE_WIDTH: 0.544336
G1 X119.415 Y119.944 E.00861
G1 X118.674 Y120.73 E.03592
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.705 J-36.469 E.28368
G1 X116.989 Y130.398 E.19629
G1 X117.728 Y131.083 E.0319
G1 X118.194 Y131.691 E.02426
G1 X118.565 Y132.362 E.02427
G1 X118.834 Y133.08 E.02428
G3 X118.999 Y133.86 I-29.418 J6.624 E.02524
G1 X119.044 Y134.597 E.02339
G1 X118.98 Y135.356 E.02411
G1 X118.824 Y136.023 E.02168
G1 X118.548 Y136.753 E.02471
G1 X118.31 Y137.19 E.01574
G3 X129.492 Y143.495 I-19.701 J48.009 E.40749
G3 X142.39 Y157.025 I-31.404 J42.848 E.59491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.39 E1.81427
G1 X144.167 Y98.945 E.01409
G1 X142.898 Y99.065 E.04037
G1 X141.28 Y99.071 E.05123
G1 X141.972 Y102.807 E.12029
G1 X141.961 Y102.925 E.00376
G1 X141.819 Y103.06 E.00621
G1 X132.841 Y103.071 E.28423
G1 X132.693 Y103.015 E.00504
G1 X132.618 Y102.877 E.00496
G3 X132.947 Y101.038 I40.739 J6.33 E.05917
G1 X133.311 Y99.071 E.06332
G3 X132.123 Y99.031 I-.282 J-9.152 E.03767
G2 X131.039 Y98.936 I-.89 J3.949 E.03454
G3 X122.631 Y116 I-51.13 J-14.59 E.60553
; LINE_WIDTH: 0.521096
G1 X122.196 Y116.579 E.02296
; LINE_WIDTH: 0.544336
G1 X121.902 Y116.975 E.01642
G1 X122.198 Y116.928 E.00998
; LINE_WIDTH: 0.526996
G1 X122.266 Y116.986 E.00287
; LINE_WIDTH: 0.526186
G1 X122.335 Y117.043 E.00286
; LINE_WIDTH: 0.525386
G1 X122.403 Y117.101 E.00286
; LINE_WIDTH: 0.524576
G1 X122.471 Y117.158 E.00285
; LINE_WIDTH: 0.523776
G1 X122.54 Y117.216 E.00285
; LINE_WIDTH: 0.522966
G1 X122.608 Y117.273 E.00285
; LINE_WIDTH: 0.522166
G1 X122.677 Y117.331 E.00284
; LINE_WIDTH: 0.521366
G1 X122.745 Y117.388 E.00284
; LINE_WIDTH: 0.520556
G1 X122.793 Y117.428 E.002
; LINE_WIDTH: 0.519996
G1 X122.906 Y117.523 E.00465
G1 F3150
G1 X123.065 Y117.657 E.0066
G1 F3300
G1 X123.225 Y117.79 E.0066
G1 F3450
G1 X123.385 Y117.924 E.0066
G1 F3600
G1 X123.544 Y118.058 E.0066
G1 X123.704 Y118.192 E.0066
G1 X123.864 Y118.326 E.0066
G3 X123.936 Y118.427 I-.08 J.133 E.00403
G1 X123.981 Y118.518 E.00322
G1 X123.957 Y118.589 E.00238
; WIPE_START
M204 S10000
G1 X123.321 Y119.361 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X128.795 Y114.042 Z7 F36000
G1 X142.856 Y100.378 Z7
G1 Z6.6
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.966156
G1 F8451.204
G1 X142.879 Y100.627 E.01519
; LINE_WIDTH: 0.92044
G1 F8887.222
G1 X142.902 Y100.876 E.01444
; LINE_WIDTH: 0.874724
G1 F9370.678
G1 X142.924 Y101.125 E.0137
; LINE_WIDTH: 0.829008
G1 F9909.759
G1 X142.947 Y101.374 E.01295
; LINE_WIDTH: 0.783291
G1 F10514.652
G1 X142.97 Y101.623 E.01221
; LINE_WIDTH: 0.737575
G1 F11198.189
G1 X142.993 Y101.872 E.01146
; LINE_WIDTH: 0.691859
G1 F11976.777
G1 X143.016 Y102.121 E.01072
; LINE_WIDTH: 0.646142
G1 F12871.723
G1 X143.039 Y102.37 E.00997
; LINE_WIDTH: 0.600426
G1 F13911.217
G1 X143.06 Y102.603 E.00863
; LINE_WIDTH: 0.557676
G1 F15047.58
G1 X143.061 Y103.076 E.01613
; WIPE_START
G1 X143.06 Y102.603 E-.17968
G1 X143.039 Y102.37 E-.08885
G1 X143.016 Y102.121 E-.095
G1 X143.012 Y102.078 E-.01647
; WIPE_END
G1 E-.02 F1800
G1 X135.386 Y102.407 Z7 F36000
G1 X131.289 Y102.584 Z7
G1 Z6.6
G1 E.4 F1800
; LINE_WIDTH: 1.0413
G1 F7820.569
G1 X131.354 Y102.338 E.01667
; LINE_WIDTH: 1.00137
G1 F8143.491
G1 X131.42 Y102.093 E.01601
; LINE_WIDTH: 0.961436
G1 F8494.23
G1 X131.501 Y101.779 E.01961
; LINE_WIDTH: 0.917251
G1 F8919.319
G1 X131.581 Y101.465 E.01867
; LINE_WIDTH: 0.873066
G1 F9389.196
G1 X131.662 Y101.15 E.01774
; LINE_WIDTH: 0.828881
G1 F9911.334
G1 X131.743 Y100.836 E.0168
; LINE_WIDTH: 0.784696
G1 F10494.964
G1 X131.751 Y100.801 E.00174
; LINE_WIDTH: 0.780296
G1 F10556.867
G1 X131.825 Y100.498 E.01518
; LINE_WIDTH: 0.745446
G1 F11074.237
G1 X131.899 Y100.195 E.01447
; WIPE_START
G1 X131.825 Y100.498 E-.11863
G1 X131.751 Y100.801 E-.11863
G1 X131.743 Y100.836 E-.01353
G1 X131.662 Y101.15 E-.12328
G1 X131.658 Y101.165 E-.00592
; WIPE_END
G1 E-.02 F1800
G1 X127.65 Y107.661 Z7 F36000
G1 X121.902 Y116.975 Z7
G1 Z6.6
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X119.415 Y119.944 E.12874
; WIPE_START
M204 S10000
G1 X120.057 Y119.178 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.371 Y123.713 Z7 F36000
G1 Z6.6
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X119.457 Y123.116 I1.187 J-2.814 E.04191
G1 X118.572 Y123.99 E.04747
G1 X118.482 Y124.224 E.00959
G2 X118.654 Y127.052 I3.794 J1.188 E.11059
G2 X120.538 Y128.937 I6.203 J-4.316 E.10231
G3 X122.615 Y132.707 I-3.521 J4.396 E.16863
G3 X121.153 Y135.818 I-3.407 J.298 E.13752
G1 X121.115 Y135.986 E.00656
G3 X125.864 Y138.449 I-35.284 J73.852 E.20428
G3 X126.195 Y136.477 I2.98 J-.514 E.07777
G3 X128.079 Y134.592 I6.203 J4.316 E.10231
G2 X129.982 Y131.764 I-4.052 J-4.781 E.13179
G2 X129.81 Y128.937 I-3.794 J-1.188 E.11059
G2 X127.925 Y127.052 I-6.203 J4.316 E.10231
G3 X126.022 Y124.224 I4.052 J-4.781 E.13179
G3 X126.195 Y121.396 I3.794 J-1.188 E.11059
G3 X128.079 Y119.511 I6.203 J4.317 E.10231
G2 X130.156 Y115.741 I-3.521 J-4.396 E.16863
G2 X129.48 Y113.384 I-3.075 J-.393 E.09622
G2 X127.921 Y111.966 I-16.923 J17.043 E.08048
G3 X126.707 Y113.969 I-33.483 J-18.938 E.08944
G1 X141.945 Y143.328 F36000
G1 F13446.283
G1 X141.945 Y140.985 E.08944
G3 X140.93 Y138.362 I4.57 J-3.278 E.10854
G3 X141.945 Y135.654 I3.072 J-.392 E.1149
G1 X141.945 Y125.904 E.37225
G3 X140.93 Y123.281 I4.57 J-3.278 E.10854
G3 X141.945 Y120.573 I3.072 J-.392 E.1149
G1 X141.945 Y110.823 E.37225
G3 X140.93 Y108.2 I4.57 J-3.278 E.10854
G3 X141.945 Y105.492 I3.072 J-.392 E.1149
G1 X141.945 Y105.265 E.00866
G1 X136.642 Y105.293 E.20249
G3 X135.466 Y106.315 I-6.023 J-5.745 E.05956
G2 X133.563 Y109.142 I4.052 J4.78 E.13179
G2 X133.735 Y111.97 I3.795 J1.188 E.11059
G2 X135.62 Y113.855 I6.203 J-4.317 E.10231
G3 X137.522 Y116.683 I-4.052 J4.78 E.13179
G3 X137.35 Y119.511 I-3.794 J1.188 E.11059
G3 X135.466 Y121.396 I-6.203 J-4.317 E.10231
G2 X133.563 Y124.224 I4.052 J4.78 E.13179
G2 X133.735 Y127.052 I3.795 J1.188 E.11059
G2 X135.62 Y128.937 I6.203 J-4.317 E.10231
G3 X137.522 Y131.764 I-4.052 J4.78 E.13179
G3 X137.35 Y134.592 I-3.794 J1.188 E.11059
G3 X135.466 Y136.477 I-6.203 J-4.317 E.10231
G2 X133.563 Y139.305 I4.052 J4.78 E.13179
G2 X133.735 Y142.133 I3.795 J1.188 E.11059
G2 X135.62 Y144.018 I6.203 J-4.317 E.10231
G3 X137.671 Y147.649 I-3.49 J4.366 E.16317
G3 X141.23 Y151.661 I-40.126 J39.19 E.20484
G3 X141.945 Y150.735 I3.346 J1.846 E.04482
G1 X141.945 Y148.393 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 6.76
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y149.393 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L42
M991 S0 P41 ;notify layer change


G17
G3 Z7 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.289 Y119.676
G1 Z6.76
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.147 Y119.905 E.01028
G1 X124.178 Y121.061 E.0576
G1 X122.893 Y122.594 E.07636
G1 X122.526 Y122.932 E.01906
G1 X122.112 Y123.156 E.01797
G1 X121.617 Y123.279 E.01948
G1 X121.12 Y123.271 E.01899
G1 X120.573 Y123.107 E.02181
G1 X120.146 Y122.836 E.0193
G1 X119.535 Y122.323 E.03047
G3 X114.848 Y126.662 I-40.736 J-39.297 E.24399
G1 X118.015 Y129.012 E.15054
G3 X118.932 Y129.849 I-4.609 J5.971 E.04747
G1 X119.566 Y130.648 E.03895
G1 X120.076 Y131.532 E.03894
G1 X120.45 Y132.48 E.03892
G3 X120.718 Y133.723 I-10.228 J2.849 E.04856
G3 X120.698 Y135.505 I-8.328 J.8 E.06818
G1 X120.538 Y136.268 E.02975
G3 X129.976 Y141.712 I-22.124 J49.253 E.4167
G3 X142.443 Y154.069 I-32.176 J44.932 E.67302
G1 X142.443 Y104.651 E1.88672
G1 X142.353 Y104.702 E.00395
G1 X141.752 Y104.797 E.02324
G1 X132.839 Y104.797 E.34029
G1 X132.534 Y104.773 E.01169
G1 X131.998 Y104.607 E.0214
G1 X131.551 Y104.311 E.02048
G1 X131.207 Y103.898 E.02054
G3 X124.294 Y116.645 I-51.593 J-19.73 E.55524
G1 X124.905 Y117.158 E.03047
G1 X125.303 Y117.614 E.02312
G1 X125.498 Y118.022 E.01726
G1 X125.601 Y118.561 E.02093
G1 X125.569 Y119.011 E.01721
G1 X125.44 Y119.433 E.01685
G1 X125.336 Y119.599 E.00749
G1 X124.821 Y119.331 F36000
G1 F13446.369
G1 X124.756 Y119.454 E.00532
G1 X124.698 Y119.529 E.0036
G1 X122.444 Y122.218 E.13396
G1 X122.144 Y122.484 E.01531
G1 X121.797 Y122.646 E.01463
G1 X121.447 Y122.705 E.01355
G1 X121.06 Y122.663 E.01488
G1 X120.795 Y122.564 E.01079
G3 X119.491 Y121.522 I8.129 J-11.505 E.06376
G3 X113.893 Y126.683 I-39.891 J-37.653 E.29094
G1 X117.666 Y129.482 E.17935
G3 X118.523 Y130.268 I-4.175 J5.414 E.04446
G1 X119.1 Y131.002 E.03566
G1 X119.563 Y131.814 E.03566
G1 X119.901 Y132.684 E.03563
G3 X120.135 Y133.79 I-10.189 J2.731 E.04322
G1 X120.181 Y134.536 E.02853
G1 X120.114 Y135.455 E.03515
G1 X119.931 Y136.289 E.03261
G1 X119.836 Y136.599 E.01239
G3 X130.162 Y142.574 I-21.722 J49.45 E.45644
G3 X142.92 Y155.769 I-31.902 J43.612 E.70415
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y108.047 E1.8212
G3 X143.048 Y103.326 I207.899 J-1.5 E.18025
; LINE_WIDTH: 0.580916
G1 F14352.287
G1 X143.062 Y103.078 E.00887
G1 F14349.022
G1 X143.011 Y103.319 E.00881
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.65 Y103.875 E.0253
G1 X142.173 Y104.145 E.02094
G1 X141.752 Y104.212 E.01626
G1 X132.839 Y104.212 E.34029
G1 X132.625 Y104.195 E.00818
G1 X132.225 Y104.066 E.01607
G1 X131.938 Y103.872 E.01322
G1 F12586.29
G1 X131.652 Y103.52 E.01731
; LINE_WIDTH: 0.666261
G1 F11059.34
G1 X131.612 Y103.414 E.00464
; LINE_WIDTH: 0.712525
G1 F10695.154
G1 X131.572 Y103.309 E.00498
; LINE_WIDTH: 0.75879
G1 F10337.036
G1 X131.532 Y103.204 E.00532
; LINE_WIDTH: 0.805054
G1 F9985.056
G1 X131.491 Y103.098 E.00566
; LINE_WIDTH: 0.851318
G1 F9639.134
G1 X131.451 Y102.993 E.006
; LINE_WIDTH: 0.897583
G1 F9122.538
G1 X131.411 Y102.888 E.00634
; LINE_WIDTH: 0.943847
G1 F8658.499
G1 X131.371 Y102.782 E.00668
; LINE_WIDTH: 0.990112
G1 F8239.383
G1 X131.331 Y102.677 E.00702
; LINE_WIDTH: 1.03638
G1 F7858.967
G1 X131.291 Y102.572 E.00736
G1 X131.225 Y102.686 E.00862
; LINE_WIDTH: 0.990112
G1 F8239.383
G1 X131.158 Y102.8 E.00822
; LINE_WIDTH: 0.943847
G1 F8658.499
G1 X131.092 Y102.914 E.00783
; LINE_WIDTH: 0.897583
G1 F9122.538
G1 X131.026 Y103.028 E.00743
; LINE_WIDTH: 0.851318
G1 F9639.134
G1 X130.96 Y103.142 E.00703
; LINE_WIDTH: 0.805054
G1 F10217.748
G1 X130.893 Y103.257 E.00663
; LINE_WIDTH: 0.75879
G1 F10635.325
G1 X130.827 Y103.371 E.00623
; LINE_WIDTH: 0.712525
G1 F11061.265
G1 X130.761 Y103.485 E.00584
; LINE_WIDTH: 0.666261
G1 F11495.568
G1 X130.694 Y103.599 E.00544
; LINE_WIDTH: 0.619996
G1 F12862.771
G1 X130.548 Y103.971 E.01527
G1 F13446.369
G1 X130.402 Y104.344 E.01527
G1 X130.207 Y104.834 E.02015
G3 X123.497 Y116.742 I-50.655 J-20.698 E.5232
G1 X124.529 Y117.607 E.0514
G1 X124.823 Y117.951 E.01728
G1 X124.979 Y118.332 E.01572
G1 X125.016 Y118.694 E.01388
G1 X124.94 Y119.103 E.0159
G1 X124.863 Y119.251 E.00636
G1 X124.302 Y119.051 F36000
G1 F13446.369
G1 X124.249 Y119.152 E.00436
G1 X121.996 Y121.841 E.13396
G1 X121.733 Y122.044 E.01267
M73 P71 R5
G1 X121.427 Y122.119 E.01205
G1 X121.205 Y122.095 E.00849
G1 X120.899 Y121.938 E.01316
G1 X119.441 Y120.716 E.07264
G3 X112.93 Y126.698 I-39.642 J-36.611 E.33796
G1 X117.317 Y129.952 E.20856
G1 X118.114 Y130.687 E.04138
G1 X118.634 Y131.357 E.03237
G1 X119.05 Y132.096 E.03238
G1 X119.352 Y132.887 E.03235
G3 X119.551 Y133.826 I-12.755 J3.186 E.03665
G1 X119.596 Y134.567 E.02834
G1 X119.531 Y135.404 E.03205
G1 X119.362 Y136.153 E.02933
G1 X119.08 Y136.91 E.03083
G3 X129.816 Y143.047 I-20.578 J48.46 E.47321
G3 X142.647 Y156.415 I-31.731 J43.299 E.71102
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.533 E2.16483
G1 X142.966 Y99.615 E.02495
G1 X141.947 Y99.624 E.03893
G1 X142.519 Y102.713 E.11995
G3 X142.481 Y103.121 I-.767 J.135 E.01581
G1 X142.264 Y103.434 E.01454
G1 X141.992 Y103.588 E.01195
G1 X141.752 Y103.626 E.00928
G1 X132.839 Y103.626 E.34029
G1 X132.488 Y103.543 E.01376
G3 X132.162 Y103.231 I.351 J-.695 E.01749
G1 X132.073 Y102.711 E.02015
G1 X132.645 Y99.621 E.11997
G3 X131.453 Y99.491 I.227 J-7.577 E.04583
G3 X122.695 Y116.833 I-51.467 J-15.108 E.74589
G1 X124.153 Y118.056 E.07265
G1 X124.321 Y118.252 E.00986
G1 X124.425 Y118.556 E.01227
G1 X124.405 Y118.853 E.01139
G1 X124.344 Y118.971 E.00507
M204 S250
G1 X123.826 Y118.797 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.569 Y121.49 E.11125
G1 X121.407 Y121.567 E.00566
G1 X121.32 Y121.537 E.00289
G3 X121.249 Y121.51 I-.001 J-.106 E.00246
G1 X121.229 Y121.494 E.00082
G1 X121.21 Y121.477 E.00082
G1 X121.19 Y121.46 E.00082
G1 F3450
G1 X120.132 Y120.574 E.04369
G1 F3600
G1 X119.64 Y120.162 E.02032
; LINE_WIDTH: 0.520666
G1 X119.63 Y120.154 E.00042
; LINE_WIDTH: 0.521746
G1 X119.613 Y120.141 E.00068
; LINE_WIDTH: 0.522836
G1 X119.595 Y120.128 E.00069
; LINE_WIDTH: 0.523196
G1 X119.59 Y120.124 E.00023
; LINE_WIDTH: 0.531646
G1 X119.561 Y120.003 E.00402
; LINE_WIDTH: 0.544336
G1 X119.518 Y119.822 E.00619
G1 X119.173 Y120.191 E.0168
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-40.299 J-37.128 E.30691
G1 X116.992 Y130.4 E.1964
G1 X117.728 Y131.083 E.03177
G1 X118.194 Y131.691 E.02427
G1 X118.565 Y132.362 E.02428
G1 X118.834 Y133.08 E.02426
G1 X118.981 Y133.766 E.02223
G1 X119.042 Y134.562 E.02526
G1 X118.98 Y135.356 E.02524
G1 X118.824 Y136.026 E.02176
G1 X118.548 Y136.753 E.02464
G1 X118.31 Y137.19 E.01574
G3 X129.492 Y143.495 I-19.764 J48.121 E.40747
G3 X142.39 Y157.025 I-31.404 J42.849 E.59492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.389 E1.81432
G1 X144.167 Y98.945 E.01403
G1 X142.897 Y99.067 E.04041
G1 X141.282 Y99.073 E.05112
G1 X141.974 Y102.809 E.12029
G1 X141.963 Y102.927 E.00376
G1 X141.821 Y103.062 E.00621
G1 X132.839 Y103.073 E.28439
G1 X132.69 Y103.017 E.00504
G1 X132.643 Y102.959 E.00237
G1 X132.638 Y102.695 E.00837
G1 X133.308 Y99.073 E.1166
G3 X132.122 Y99.032 I-.281 J-9.002 E.0376
G2 X131.038 Y98.936 I-.891 J3.896 E.03458
G3 X122.131 Y116.662 I-50.994 J-14.522 E.6318
; LINE_WIDTH: 0.544336
G1 X121.799 Y117.098 E.01821
G1 X121.902 Y117.101 E.00343
; LINE_WIDTH: 0.537436
G1 X122.005 Y117.104 E.00338
; LINE_WIDTH: 0.530536
G1 X122.108 Y117.107 E.00333
; LINE_WIDTH: 0.523636
G1 X122.162 Y117.108 E.00174
; LINE_WIDTH: 0.519996
G1 X122.175 Y117.119 E.00052
G1 X122.202 Y117.141 E.00111
G1 F3450
G1 X122.228 Y117.164 E.00111
G1 F3300
G1 X122.255 Y117.186 E.00111
G1 F3150
G1 X122.365 Y117.279 E.00455
G1 F3300
G1 X122.639 Y117.508 E.01131
G1 F3450
G1 X122.913 Y117.738 E.01131
G1 F3600
G1 X123.187 Y117.967 E.01131
G1 X123.461 Y118.197 E.01131
G1 X123.735 Y118.426 E.01131
G3 X123.833 Y118.55 I-.092 J.174 E.00514
G1 X123.878 Y118.642 E.00324
G1 X123.855 Y118.712 E.00235
; WIPE_START
M204 S10000
G1 X123.219 Y119.484 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X128.69 Y114.162 Z7.16 F36000
G1 X142.857 Y100.379 Z7.16
G1 Z6.76
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.963876
G1 F8471.933
G1 X142.88 Y100.628 E.01515
; LINE_WIDTH: 0.91816
G1 F8910.148
G1 X142.903 Y100.877 E.0144
; LINE_WIDTH: 0.872444
G1 F9396.17
G1 X142.926 Y101.126 E.01366
; LINE_WIDTH: 0.826728
G1 F9938.273
G1 X142.948 Y101.375 E.01291
; LINE_WIDTH: 0.781011
G1 F10546.758
G1 X142.971 Y101.624 E.01217
; LINE_WIDTH: 0.735295
G1 F11234.614
G1 X142.994 Y101.873 E.01142
; LINE_WIDTH: 0.689579
G1 F12018.452
G1 X143.017 Y102.122 E.01068
; LINE_WIDTH: 0.643862
G1 F12919.872
G1 X143.04 Y102.371 E.00993
; LINE_WIDTH: 0.598146
G1 F13967.472
G1 X143.061 Y102.605 E.00865
; LINE_WIDTH: 0.555136
G1 F15120.969
G1 X143.062 Y103.078 E.01604
; WIPE_START
G1 X143.061 Y102.605 E-.17951
G1 X143.04 Y102.371 E-.08939
G1 X143.017 Y102.122 E-.095
G1 X143.013 Y102.08 E-.0161
; WIPE_END
G1 E-.02 F1800
G1 X135.387 Y102.4 Z7.16 F36000
G1 X131.291 Y102.572 Z7.16
G1 Z6.76
G1 E.4 F1800
; LINE_WIDTH: 1.03638
G1 F7858.967
G1 X131.333 Y102.419 E.01035
; LINE_WIDTH: 1.01026
G1 F8069.309
G1 X131.417 Y102.095 E.0213
; LINE_WIDTH: 0.96171
G1 F8491.726
G1 X131.502 Y101.771 E.02024
; LINE_WIDTH: 0.913163
G1 F8960.813
G1 X131.587 Y101.447 E.01918
; LINE_WIDTH: 0.864616
G1 F9484.753
G1 X131.665 Y101.134 E.01749
; LINE_WIDTH: 0.824591
G1 F9965.14
G1 X131.743 Y100.82 E.01664
; LINE_WIDTH: 0.784566
G1 F10496.782
G1 X131.822 Y100.507 E.0158
; LINE_WIDTH: 0.744541
G1 F11088.349
G1 X131.9 Y100.193 E.01496
; WIPE_START
G1 X131.822 Y100.507 E-.12277
G1 X131.743 Y100.82 E-.12276
G1 X131.665 Y101.134 E-.12277
G1 X131.658 Y101.163 E-.0117
; WIPE_END
G1 E-.02 F1800
G1 X127.642 Y107.654 Z7.16 F36000
G1 X121.799 Y117.098 Z7.16
G1 Z6.76
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X119.518 Y119.822 E.1181
; WIPE_START
M204 S10000
G1 X120.16 Y119.055 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.359 Y123.553 Z7.16 F36000
G1 Z6.76
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X119.569 Y123.001 I2.397 J-4.272 E.03686
G3 X118.596 Y123.973 I-21.077 J-20.133 E.05252
G2 X118.837 Y127.052 I3.882 J1.245 E.12089
G2 X120.679 Y128.937 I7.494 J-5.482 E.10098
G3 X122.411 Y131.764 I-4.068 J4.435 E.12826
G3 X121.179 Y135.633 I-3.578 J.99 E.164
G1 X121.111 Y135.983 E.01364
G3 X125.943 Y138.494 I-35.433 J74.111 E.20795
G3 X126.377 Y136.477 I3.132 J-.38 E.08023
G3 X128.22 Y134.592 I7.495 J5.482 E.10098
G2 X129.952 Y131.764 I-4.068 J-4.435 E.12826
G2 X129.627 Y128.937 I-3.814 J-.994 E.11118
G2 X127.784 Y127.052 I-7.494 J5.482 E.10098
G3 X126.052 Y124.224 I4.068 J-4.435 E.12826
G3 X126.377 Y121.396 I3.814 J-.994 E.11119
G3 X128.22 Y119.511 I7.495 J5.483 E.10098
G2 X129.952 Y116.683 I-4.068 J-4.435 E.12826
G2 X129.627 Y113.855 I-3.814 J-.994 E.11118
G2 X127.873 Y112.051 I-7.214 J5.255 E.0964
G3 X126.654 Y114.05 I-33.437 J-19.013 E.08944
G1 X141.945 Y143.43 F36000
G1 F13446.283
G1 X141.945 Y141.087 E.08944
G3 X141.007 Y138.362 I4.591 J-3.106 E.1113
G3 X141.945 Y135.859 I3.307 J-.187 E.10506
G1 X141.945 Y126.006 E.3762
G3 X141.007 Y123.281 I4.591 J-3.106 E.1113
G3 X141.945 Y120.778 I3.307 J-.187 E.10506
G1 X141.945 Y110.924 E.3762
G3 X141.007 Y108.2 I4.591 J-3.106 E.1113
G3 X141.945 Y105.697 I3.307 J-.187 E.10506
G1 X141.945 Y105.267 E.0164
G1 X136.442 Y105.295 E.2101
G3 X135.325 Y106.315 I-7.846 J-7.478 E.0578
G2 X133.593 Y109.142 I4.068 J4.435 E.12826
G2 X133.918 Y111.97 I3.814 J.994 E.11118
G2 X135.761 Y113.855 I7.494 J-5.481 E.10098
G3 X137.493 Y116.683 I-4.068 J4.436 E.12826
G3 X137.167 Y119.511 I-3.814 J.994 E.11118
G3 X135.325 Y121.396 I-7.494 J-5.482 E.10098
G2 X133.593 Y124.224 I4.068 J4.435 E.12826
G2 X133.918 Y127.052 I3.814 J.994 E.11118
G2 X135.761 Y128.937 I7.494 J-5.481 E.10098
G3 X137.493 Y131.764 I-4.068 J4.436 E.12826
G3 X137.167 Y134.592 I-3.814 J.994 E.11118
G3 X135.325 Y136.477 I-7.495 J-5.482 E.10098
G2 X133.593 Y139.305 I4.068 J4.435 E.12826
G2 X133.918 Y142.133 I3.814 J.994 E.11118
G2 X135.761 Y144.018 I7.494 J-5.482 E.10098
G3 X137.59 Y147.565 I-3.489 J4.044 E.15616
G3 X141.34 Y151.798 I-40.086 J39.298 E.21599
G3 X141.945 Y150.941 I3.424 J1.777 E.04018
G1 X141.945 Y148.598 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 6.92
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y149.598 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L43
M991 S0 P42 ;notify layer change


G17
G3 Z7.16 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.17 Y119.841
G1 Z6.92
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X125.044 Y120.027 E.00857
G1 X122.996 Y122.471 E.12174
G1 X122.61 Y122.823 E.01992
G1 X122.195 Y123.041 E.01791
G1 X121.701 Y123.158 E.01938
G1 X121.207 Y123.146 E.01887
G1 X120.689 Y122.991 E.02066
G1 X120.249 Y122.713 E.01987
G1 X119.646 Y122.207 E.03006
G3 X114.848 Y126.662 I-40.674 J-38.991 E.25012
G1 X118.015 Y129.012 E.15055
G3 X118.933 Y129.849 I-4.606 J5.968 E.04749
G1 X119.566 Y130.648 E.03893
G1 X120.076 Y131.531 E.03894
G1 X120.45 Y132.478 E.03884
G1 X120.681 Y133.47 E.03891
G3 X120.698 Y135.506 I-7.934 J1.083 E.07795
G1 X120.538 Y136.268 E.02971
G3 X130.505 Y142.1 I-21.924 J48.901 E.44174
G3 X142.443 Y154.069 I-32.639 J44.491 E.64798
G1 X142.443 Y104.655 E1.8866
G1 X142.356 Y104.705 E.00384
G1 X141.754 Y104.8 E.02324
G1 X132.836 Y104.8 E.34048
G1 X132.531 Y104.775 E.0117
G1 X131.995 Y104.609 E.02142
G1 X131.547 Y104.313 E.0205
G1 X131.205 Y103.901 E.02042
G3 X124.199 Y116.775 I-51.595 J-19.735 E.56123
G1 X124.803 Y117.281 E.03005
G3 X125.263 Y117.844 I-1.253 J1.494 E.02794
G1 X125.457 Y118.367 E.02131
G1 X125.499 Y118.843 E.01823
G1 X125.432 Y119.284 E.01703
G1 X125.273 Y119.687 E.01653
G1 X125.22 Y119.767 E.00367
G1 X124.689 Y119.501 F36000
G1 F13446.369
G1 X124.596 Y119.651 E.00673
G1 X122.547 Y122.095 E.12174
G1 X122.234 Y122.37 E.01591
G1 X121.886 Y122.527 E.01458
G3 X120.91 Y122.448 I-.368 J-1.511 E.03802
G1 X120.467 Y122.131 E.02081
G1 X119.6 Y121.405 E.04317
G3 X113.893 Y126.683 I-40.171 J-37.711 E.29705
G1 X117.677 Y129.49 E.17987
G1 X118.523 Y130.268 E.0439
G1 X119.1 Y131.002 E.03564
G1 X119.563 Y131.814 E.03566
G1 X119.901 Y132.681 E.03556
G1 X120.108 Y133.591 E.03563
G3 X120.114 Y135.456 I-7.969 J.958 E.07134
G1 X119.931 Y136.289 E.03258
G1 X119.836 Y136.599 E.01238
G3 X130.932 Y143.156 I-21.317 J48.738 E.4933
G3 X142.92 Y155.769 I-32.176 J42.589 E.66734
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y108.048 E1.82117
G3 X143.049 Y103.328 I200.107 J-1.5 E.18022
; LINE_WIDTH: 0.579416
G1 F14351.606
G1 X143.064 Y103.079 E.00884
G1 F14348.439
G1 X143.013 Y103.321 E.00878
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.652 Y103.877 E.0253
G1 X142.175 Y104.147 E.02094
G1 X141.754 Y104.214 E.01626
G1 X132.836 Y104.214 E.34048
G1 X132.623 Y104.197 E.00819
G1 X132.222 Y104.067 E.01609
G1 X131.934 Y103.873 E.01323
G1 X131.649 Y103.521 E.01733
G1 X131.476 Y102.947 E.02287
G1 X131.473 Y102.84 E.00408
G1 X131.126 Y102.416 E.02092
G3 X123.401 Y116.87 I-51.021 J-17.979 E.62812
G1 X124.426 Y117.729 E.05109
G1 X124.749 Y118.123 E.01943
G1 X124.884 Y118.49 E.01491
G1 X124.906 Y118.927 E.01672
G1 X124.799 Y119.323 E.01566
G1 X124.736 Y119.425 E.00457
G1 X124.196 Y119.189 F36000
G1 F13446.369
G1 X124.147 Y119.275 E.0038
G1 X122.091 Y121.728 E.12219
G1 X121.828 Y121.925 E.01254
G1 X121.522 Y121.997 E.01201
G1 X121.177 Y121.926 E.01344
G3 X119.55 Y120.598 I32.321 J-41.27 E.08021
G3 X112.93 Y126.698 I-40.062 J-36.836 E.34407
G1 X117.323 Y129.957 E.20885
G1 X118.114 Y130.687 E.0411
G1 X118.634 Y131.357 E.03235
G1 X119.049 Y132.096 E.03237
G1 X119.352 Y132.885 E.03228
G3 X119.531 Y135.405 I-6.109 J1.701 E.09709
G1 X119.361 Y136.154 E.0293
G1 X119.08 Y136.91 E.03082
G3 X130.578 Y143.623 I-20.821 J48.865 E.50967
G3 X142.648 Y156.415 I-31.78 J42.074 E.6746
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y99.534 E2.16481
G1 X142.966 Y99.617 E.02495
G1 X141.949 Y99.626 E.03884
G1 X142.521 Y102.715 E.11995
G3 X142.483 Y103.123 I-.767 J.135 E.01581
G1 X142.267 Y103.436 E.01454
G1 X141.994 Y103.59 E.01195
G1 X141.754 Y103.628 E.00928
G1 X132.836 Y103.628 E.34048
G1 X132.486 Y103.545 E.01377
G3 X132.159 Y103.233 I.351 J-.695 E.01751
G1 X132.058 Y102.829 E.01586
G1 X132.114 Y102.473 E.01377
G1 X131.952 Y102.458 E.00624
G1 X131.791 Y102.343 E.00754
G1 X131.619 Y102.108 E.01112
G1 X131.548 Y101.698 E.01588
G1 F13418.266
G1 X131.587 Y101.296 E.01541
G1 F12008.298
G1 X131.626 Y100.898 E.01527
; LINE_WIDTH: 0.668638
G1 F10688.584
G1 X131.61 Y100.804 E.00395
; LINE_WIDTH: 0.717279
G1 F10385.115
G1 X131.594 Y100.71 E.00425
; LINE_WIDTH: 0.76592
G1 F10086.017
G1 X131.579 Y100.616 E.00455
; LINE_WIDTH: 0.814561
G1 F9791.288
G1 X131.563 Y100.522 E.00485
; LINE_WIDTH: 0.863203
G1 F9500.93
G1 X131.547 Y100.428 E.00516
; LINE_WIDTH: 0.911844
G1 F8974.283
G1 X131.531 Y100.334 E.00546
; LINE_WIDTH: 0.960485
G1 F8502.954
G1 X131.515 Y100.24 E.00576
; LINE_WIDTH: 1.00913
G1 F8078.663
G1 X131.499 Y100.146 E.00606
; LINE_WIDTH: 1.03838
G1 F7843.313
G1 X131.49 Y100.103 E.00287
G1 X131.441 Y100.178 E.00587
; LINE_WIDTH: 0.99189
G1 F8224.085
G1 X131.393 Y100.254 E.0056
; LINE_WIDTH: 0.945403
G1 F8643.715
G1 X131.345 Y100.33 E.00533
; LINE_WIDTH: 0.898916
G1 F9108.47
G1 X131.296 Y100.405 E.00506
; LINE_WIDTH: 0.85243
G1 F9626.041
G1 X131.248 Y100.481 E.00478
; LINE_WIDTH: 0.805943
G1 F10205.978
G1 X131.199 Y100.556 E.00451
; LINE_WIDTH: 0.759456
G1 F10488.753
G1 X131.151 Y100.632 E.00424
; LINE_WIDTH: 0.71297
G1 F10775.364
G1 X131.103 Y100.707 E.00397
; LINE_WIDTH: 0.666483
G1 F11065.821
G1 X131.054 Y100.783 E.0037
; LINE_WIDTH: 0.619996
G1 F12407.95
G1 X130.928 Y101.163 E.01527
G1 F13446.369
G1 X130.802 Y101.542 E.01527
G1 X130.589 Y102.165 E.02514
G3 X122.598 Y116.961 I-50.866 J-17.914 E.64464
G1 X124.05 Y118.178 E.07234
G1 X124.234 Y118.403 E.01109
G1 X124.328 Y118.742 E.01343
G1 X124.284 Y119.034 E.01127
G1 X124.241 Y119.11 E.00336
M204 S250
G1 X123.723 Y118.92 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.672 Y121.366 E.10106
G1 X121.525 Y121.443 E.00527
G1 X121.43 Y121.414 E.00315
G3 X121.3 Y121.344 I.018 J-.189 E.00481
G1 X121.053 Y121.137 E.0102
G1 X120.806 Y120.93 E.0102
G1 X120.559 Y120.723 E.0102
G1 F3450
G1 X120.312 Y120.516 E.0102
G1 F3300
G1 X120.065 Y120.309 E.0102
G1 F3150
G1 X119.845 Y120.124 E.0091
; LINE_WIDTH: 0.520456
G1 F3600
G1 X119.823 Y120.106 E.00089
; LINE_WIDTH: 0.520976
G1 X119.798 Y120.087 E.001
; LINE_WIDTH: 0.521486
G1 X119.774 Y120.067 E.001
; LINE_WIDTH: 0.522006
G1 X119.749 Y120.047 E.001
; LINE_WIDTH: 0.522516
G1 X119.725 Y120.027 E.001
; LINE_WIDTH: 0.523026
G1 X119.701 Y120.008 E.001
; LINE_WIDTH: 0.523196
G1 X119.692 Y120.001 E.00033
; LINE_WIDTH: 0.531656
G1 X119.664 Y119.88 E.00402
; LINE_WIDTH: 0.544336
G1 X119.621 Y119.699 E.00619
G1 X119.181 Y120.183 E.02174
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.855 J-35.522 E.30731
G1 X116.989 Y130.398 E.1963
G1 X117.728 Y131.083 E.03189
G1 X118.194 Y131.691 E.02426
G1 X118.565 Y132.362 E.02427
G1 X118.833 Y133.078 E.0242
G1 X118.994 Y133.827 E.02426
G1 X119.035 Y134.451 E.01979
G1 X118.98 Y135.357 E.02876
G1 X118.824 Y136.026 E.02173
G1 X118.548 Y136.753 E.02463
G1 X118.31 Y137.19 E.01575
G3 X130.248 Y144.067 I-19.584 J47.798 E.4375
G3 X142.39 Y157.025 I-31.644 J41.817 E.56494
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.387 E1.81438
G1 X144.167 Y98.946 E.01397
G2 X142.896 Y99.069 I.106 J7.699 E.04049
G1 X141.285 Y99.075 E.05101
G1 X141.977 Y102.811 E.12029
G1 X141.966 Y102.929 E.00376
G1 X141.824 Y103.064 E.00621
G1 X132.836 Y103.075 E.28455
G1 X132.687 Y103.019 E.00505
G1 X132.611 Y102.844 E.00605
G3 X132.78 Y101.918 I96.132 J17.031 E.0298
G2 X132.392 Y101.989 I-.072 J.701 E.01263
G1 X132.169 Y101.939 E.00725
M73 P72 R5
G1 X132.098 Y101.752 E.00632
G2 X132.244 Y100.214 I-128.488 J-12.943 E.04892
; LINE_WIDTH: 0.55526
G1 X132.272 Y100.14 E.00268
; LINE_WIDTH: 0.590523
G1 X132.301 Y100.067 E.00286
; LINE_WIDTH: 0.625786
G1 X132.329 Y99.993 E.00304
; LINE_WIDTH: 0.663756
G1 X132.364 Y99.962 E.0019
; LINE_WIDTH: 0.701726
G1 X132.398 Y99.931 E.00202
; LINE_WIDTH: 0.739696
G1 X132.433 Y99.901 E.00213
; LINE_WIDTH: 0.744396
G1 X133.022 Y99.993 E.02759
G1 X133.171 Y99.188 E.03792
G1 X132.621 Y99.176 E.02545
; LINE_WIDTH: 0.736706
G1 X132.454 Y99.146 E.0078
; LINE_WIDTH: 0.699733
G1 X132.286 Y99.116 E.00739
; LINE_WIDTH: 0.66276
G1 X132.119 Y99.086 E.00698
; LINE_WIDTH: 0.625786
G1 X131.924 Y99.039 E.00771
; LINE_WIDTH: 0.590523
G1 X131.73 Y98.991 E.00725
; LINE_WIDTH: 0.55526
G1 X131.536 Y98.943 E.00679
; LINE_WIDTH: 0.519996
G1 X131.038 Y98.936 E.01578
G3 X122.131 Y116.662 I-50.994 J-14.522 E.6318
; LINE_WIDTH: 0.544336
G1 X121.696 Y117.221 E.02353
G1 X121.892 Y117.213 E.00652
; LINE_WIDTH: 0.531156
G1 X122.034 Y117.208 E.00461
; LINE_WIDTH: 0.521596
G1 X122.066 Y117.234 E.00132
; LINE_WIDTH: 0.521526
G1 X122.182 Y117.332 E.00483
; LINE_WIDTH: 0.521296
G1 X122.299 Y117.43 E.00482
; LINE_WIDTH: 0.521066
G1 X122.415 Y117.528 E.00482
; LINE_WIDTH: 0.520836
G1 X122.531 Y117.626 E.00482
; LINE_WIDTH: 0.520616
G1 X122.647 Y117.723 E.00482
; LINE_WIDTH: 0.520386
G1 X122.764 Y117.821 E.00481
; LINE_WIDTH: 0.520156
G1 X122.846 Y117.89 E.00339
; LINE_WIDTH: 0.519996
G1 X122.932 Y117.962 E.00357
G1 X123.055 Y118.065 E.00506
G1 X123.177 Y118.168 E.00506
G1 X123.3 Y118.27 E.00506
G1 X123.422 Y118.373 E.00506
G1 X123.544 Y118.476 E.00506
G1 X123.667 Y118.578 E.00506
G3 X123.73 Y118.673 I-.076 J.119 E.00371
G1 X123.776 Y118.765 E.00326
G1 X123.752 Y118.835 E.00232
; WIPE_START
M204 S10000
G1 X123.117 Y119.607 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X128.994 Y114.738 Z7.32 F36000
G1 X143.064 Y103.079 Z7.32
G1 Z6.92
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.552616
G1 F15194.491
G1 X143.063 Y102.607 E.01595
; LINE_WIDTH: 0.595886
G1 F14023.686
G1 X143.041 Y102.372 E.00866
; LINE_WIDTH: 0.641603
G1 F12967.953
G1 X143.018 Y102.123 E.0099
; LINE_WIDTH: 0.687319
G1 F12060.049
G1 X142.995 Y101.874 E.01064
; LINE_WIDTH: 0.733035
G1 F11270.952
G1 X142.972 Y101.625 E.01139
; LINE_WIDTH: 0.778751
G1 F10578.778
G1 X142.95 Y101.376 E.01213
; LINE_WIDTH: 0.824467
G1 F9966.699
G1 X142.927 Y101.127 E.01288
; LINE_WIDTH: 0.870184
G1 F9421.575
G1 X142.904 Y100.878 E.01362
; LINE_WIDTH: 0.9159
G1 F8932.99
G1 X142.881 Y100.629 E.01437
; LINE_WIDTH: 0.961616
G1 F8492.581
G1 X142.858 Y100.38 E.01511
; WIPE_START
G1 X142.881 Y100.629 E-.095
G1 X142.904 Y100.878 E-.095
G1 X142.927 Y101.127 E-.095
G1 X142.95 Y101.376 E-.095
; WIPE_END
G1 E-.02 F1800
G1 X135.364 Y100.533 Z7.32 F36000
G1 X131.49 Y100.103 Z7.32
G1 Z6.92
G1 E.4 F1800
; LINE_WIDTH: 1.05502
G1 F7715.445
G1 X131.607 Y99.734 E.02576
; WIPE_START
G1 X131.49 Y100.103 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X127.699 Y106.728 Z7.32 F36000
G1 X121.696 Y117.221 Z7.32
G1 Z6.92
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X119.621 Y119.699 E.10746
; WIPE_START
M204 S10000
G1 X120.263 Y118.932 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.353 Y123.369 Z7.32 F36000
G1 Z6.92
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X119.681 Y122.886 I1.593 J-2.923 E.03167
G3 X118.612 Y123.958 I-23.743 J-22.603 E.05779
G2 X119.007 Y127.052 I3.98 J1.064 E.12211
G2 X120.816 Y128.937 I9.286 J-7.102 E.09997
G3 X122.387 Y131.764 I-4.05 J4.1 E.12519
G3 X121.557 Y135.063 I-3.619 J.843 E.13491
G2 X121.111 Y135.984 I.591 J.855 E.04082
G3 X126.01 Y138.532 I-35.837 J74.895 E.21088
G3 X126.547 Y136.477 I3.313 J-.231 E.08252
G3 X128.357 Y134.592 I9.286 J7.102 E.09997
G2 X129.657 Y132.707 I-3.727 J-3.962 E.0881
G2 X129.457 Y128.937 I-3.962 J-1.68 E.14923
G2 X127.648 Y127.052 I-9.286 J7.102 E.09997
G3 X126.347 Y125.166 I3.727 J-3.962 E.0881
G3 X126.547 Y121.396 I3.962 J-1.68 E.14923
G3 X128.357 Y119.511 I9.287 J7.103 E.09997
G2 X129.657 Y117.626 I-3.727 J-3.962 E.0881
G2 X129.457 Y113.855 I-3.962 J-1.68 E.14923
G2 X127.824 Y112.137 I-8.567 J6.506 E.09071
G3 X126.607 Y114.138 I-40.092 J-23.012 E.08944
G1 X141.945 Y143.549 F36000
G1 F13446.283
G1 X141.945 Y141.207 E.08944
G3 X141.158 Y139.305 I6.14 J-3.657 E.07886
G3 X141.945 Y136.062 I3.591 J-.845 E.13221
G1 X141.945 Y126.126 E.37938
G3 X141.158 Y124.224 I6.14 J-3.657 E.07886
G3 X141.945 Y120.981 I3.592 J-.845 E.13221
G1 X141.945 Y111.044 E.37938
G3 X141.158 Y109.142 I6.14 J-3.657 E.07886
G3 X141.945 Y105.9 I3.592 J-.845 E.13221
G1 X141.945 Y105.27 E.02405
G1 X136.257 Y105.297 E.21718
G3 X135.188 Y106.315 I-11.055 J-10.544 E.05636
G2 X133.888 Y108.2 I3.726 J3.962 E.0881
G2 X134.088 Y111.97 I3.962 J1.68 E.14923
G2 X135.897 Y113.855 I9.286 J-7.102 E.09997
G3 X137.198 Y115.741 I-3.726 J3.962 E.0881
G3 X136.998 Y119.511 I-3.962 J1.68 E.14923
G3 X135.188 Y121.396 I-9.287 J-7.103 E.09997
G2 X133.888 Y123.281 I3.726 J3.962 E.0881
G2 X134.088 Y127.052 I3.962 J1.68 E.14923
G2 X135.897 Y128.937 I9.286 J-7.102 E.09997
G3 X137.198 Y130.822 I-3.726 J3.962 E.0881
G3 X136.998 Y134.592 I-3.962 J1.68 E.14923
G3 X135.188 Y136.477 I-9.288 J-7.104 E.09997
G2 X133.888 Y138.362 I3.726 J3.962 E.0881
G2 X134.088 Y142.133 I3.962 J1.68 E.14923
G2 X135.897 Y144.018 I9.287 J-7.103 E.09997
G3 X137.525 Y147.502 I-3.466 J3.741 E.15055
G3 X141.434 Y151.914 I-39.502 J38.94 E.22516
G3 X141.945 Y151.144 I2.774 J1.285 E.03545
G1 X141.945 Y148.801 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 7.08
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y149.801 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L44
M991 S0 P43 ;notify layer change


G17
G3 Z7.32 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X125.066 Y119.966
G1 Z7.08
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X124.942 Y120.15 E.00849
G1 X123.099 Y122.349 E.10952
G1 X122.747 Y122.676 E.01835
G1 X122.284 Y122.923 E.02002
G1 X121.808 Y123.035 E.01868
G1 X121.332 Y123.027 E.01817
G1 X120.84 Y122.89 E.01949
G1 X120.36 Y122.597 E.02147
G1 X119.757 Y122.091 E.03006
G3 X114.848 Y126.662 I-41.266 J-39.394 E.25624
G1 X118.03 Y129.023 E.15128
G1 X118.485 Y129.402 E.02261
G1 X119.155 Y130.103 E.03704
G1 X119.749 Y130.931 E.03888
G1 X120.215 Y131.838 E.03893
G1 X120.498 Y132.638 E.03241
G1 X120.674 Y133.433 E.03106
G1 X120.765 Y134.489 E.04049
G1 X120.698 Y135.504 E.0388
G1 X120.538 Y136.268 E.02982
G3 X129.809 Y141.592 I-21.932 J48.922 E.40884
G3 X142.443 Y154.069 I-32.037 J45.077 E.68087
G1 X142.443 Y104.658 E1.88645
G1 X142.358 Y104.707 E.00373
G1 X141.757 Y104.802 E.02324
G1 X132.834 Y104.802 E.34067
G1 X132.528 Y104.777 E.01172
G2 X131.716 Y104.568 I-1.383 J3.674 E.03206
G1 X131.236 Y104.363 E.01993
G1 X131.077 Y104.227 E.00799
G3 X124.105 Y116.905 I-51.399 J-20.011 E.55401
G1 X124.7 Y117.403 E.02963
G1 X125.019 Y117.743 E.01779
G1 X125.196 Y118.035 E.01304
G1 X125.364 Y118.538 E.02023
G1 X125.396 Y118.974 E.01669
G3 X125.16 Y119.807 I-2.181 J-.166 E.03327
G1 X125.112 Y119.888 E.00361
G1 X124.615 Y119.578 F36000
G1 F13446.369
G1 X124.55 Y119.7 E.00527
G3 X123.925 Y120.452 I-11.285 J-8.762 E.03733
G1 X122.64 Y121.984 E.07636
G1 X122.361 Y122.231 E.01423
G1 X121.944 Y122.417 E.01742
G1 X121.499 Y122.456 E.01706
G1 X121.07 Y122.351 E.01687
G1 X120.701 Y122.118 E.01666
G1 X119.709 Y121.287 E.0494
G3 X113.893 Y126.683 I-40.186 J-37.484 E.30317
G1 X117.677 Y129.49 E.17987
G1 X118.108 Y129.85 E.02145
G1 X118.726 Y130.502 E.0343
G1 X119.266 Y131.262 E.0356
G3 X120.1 Y133.55 I-6.03 J3.494 E.09342
G1 X120.18 Y134.525 E.03737
G1 X120.115 Y135.453 E.03553
G1 X119.932 Y136.287 E.0326
G1 X119.836 Y136.599 E.01246
G3 X130.16 Y142.573 I-21.517 J49.095 E.45634
G3 X142.92 Y155.769 I-31.893 J43.608 E.70426
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y108.049 E1.82113
G3 X143.05 Y103.329 I192.869 J-1.5 E.18019
; LINE_WIDTH: 0.577916
G1 F14350.962
G1 X143.065 Y103.081 E.00881
G1 F14347.891
G1 X143.015 Y103.323 E.00875
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.655 Y103.879 E.02529
G1 X142.178 Y104.15 E.02094
G1 X141.757 Y104.216 E.01626
G1 X132.834 Y104.216 E.34067
G3 X131.536 Y103.86 I.844 J-5.622 E.05152
G1 X131.16 Y103.53 E.01909
G1 X130.95 Y102.904 E.02519
G3 X123.304 Y116.998 I-50.79 J-18.434 E.61443
G1 X124.324 Y117.852 E.05078
G1 X124.547 Y118.09 E.01245
G1 X124.755 Y118.508 E.01783
G1 X124.81 Y118.951 E.01704
G1 X124.733 Y119.355 E.01569
G1 X124.657 Y119.499 E.00622
G1 X124.093 Y119.312 F36000
G1 F13446.369
G1 X124.044 Y119.398 E.00377
G1 X122.195 Y121.603 E.10986
G3 X121.433 Y121.855 I-.617 J-.585 E.03194
G1 X121.104 Y121.693 E.01399
G1 X119.655 Y120.478 E.07222
G3 X112.93 Y126.698 I-39.798 J-36.282 E.35017
G1 X117.323 Y129.957 E.20885
G1 X117.73 Y130.298 E.02029
G1 X118.297 Y130.9 E.03156
G1 X118.783 Y131.594 E.03232
G3 X119.526 Y133.666 I-5.475 J3.132 E.08449
G1 X119.596 Y134.561 E.03424
G1 X119.531 Y135.403 E.03225
G1 X119.362 Y136.152 E.02932
G1 X119.08 Y136.91 E.03089
G3 X129.815 Y143.046 I-20.338 J48.04 E.47317
G3 X142.648 Y156.415 I-31.729 J43.299 E.71108
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y101.378 E2.09441
G1 X143.615 Y100.978 E.01527
; LINE_WIDTH: 0.667626
G1 F12435.062
G1 X143.591 Y100.665 E.01296
; LINE_WIDTH: 0.715256
G1 F11565.236
G1 X143.567 Y100.352 E.01394
; LINE_WIDTH: 0.762886
G1 F10809.143
G3 X143.543 Y99.616 I4.774 J-.521 E.03501
; LINE_WIDTH: 0.762156
G1 F10819.984
G3 X142.863 Y99.702 I-2.06 J-13.573 E.03256
; LINE_WIDTH: 0.762886
G1 F10809.143
G1 X142.82 Y99.976 E.01317
G1 X142.704 Y100.128 E.0091
; LINE_WIDTH: 0.715256
G1 F11434.977
G1 X142.589 Y100.281 E.0085
; LINE_WIDTH: 0.667626
G1 F12078.374
G1 X142.473 Y100.434 E.00791
; LINE_WIDTH: 0.619996
G1 F13373.632
G1 X142.102 Y100.442 E.01415
G1 F13446.369
G1 X142.175 Y100.836 E.01527
G1 X142.524 Y102.717 E.07306
G3 X142.486 Y103.125 I-.767 J.135 E.01581
G1 X142.269 Y103.438 E.01454
G1 X141.997 Y103.592 E.01195
G1 X141.757 Y103.63 E.00928
G1 X132.834 Y103.63 E.34067
G1 X132.503 Y103.556 E.01296
G1 X132.348 Y103.448 E.0072
G1 X132.043 Y103.443 E.01165
G1 X131.835 Y103.357 E.00859
G1 X131.621 Y103.168 E.01089
G1 X131.455 Y102.674 E.01991
G1 X131.503 Y102.165 E.01949
G1 X131.54 Y101.767 E.01527
G1 F12789.594
G1 X131.578 Y101.369 E.01527
; LINE_WIDTH: 0.669004
G1 F11426.395
G1 X131.568 Y101.216 E.00636
; LINE_WIDTH: 0.718011
G1 F10922.782
G1 X131.558 Y101.062 E.00685
; LINE_WIDTH: 0.767019
G1 F10430.488
G1 X131.548 Y100.909 E.00734
; LINE_WIDTH: 0.816026
G1 F9949.576
G1 X131.538 Y100.755 E.00784
; LINE_WIDTH: 0.865034
G1 F9479.986
G1 X131.528 Y100.602 E.00833
; LINE_WIDTH: 0.914041
G1 F8951.865
G1 X131.518 Y100.448 E.00882
; LINE_WIDTH: 0.963049
G1 F8479.481
G1 X131.508 Y100.295 E.00931
; LINE_WIDTH: 1.01206
G1 F8054.454
G1 X131.498 Y100.141 E.0098
; LINE_WIDTH: 1.03838
G1 F7843.313
G1 X131.49 Y100.103 E.00259
G1 X131.466 Y100.135 E.0026
; LINE_WIDTH: 1.01206
G1 F8054.454
G1 X131.396 Y100.272 E.0098
; LINE_WIDTH: 0.963049
G1 F8479.481
G1 X131.327 Y100.409 E.00931
; LINE_WIDTH: 0.914041
G1 F8951.865
G1 X131.258 Y100.547 E.00882
; LINE_WIDTH: 0.865034
G1 F9479.986
G1 X131.189 Y100.684 E.00833
; LINE_WIDTH: 0.816026
G1 F10074.327
G1 X131.119 Y100.821 E.00784
; LINE_WIDTH: 0.767019
G1 F10558.225
G1 X131.05 Y100.958 E.00734
; LINE_WIDTH: 0.718011
G1 F11053.473
G1 X130.981 Y101.096 E.00685
; LINE_WIDTH: 0.669004
G1 F11560.043
G1 X130.912 Y101.233 E.00636
; LINE_WIDTH: 0.619996
G1 F12930.968
G1 X130.782 Y101.611 E.01527
G1 F13446.369
G1 X130.651 Y101.99 E.01527
G1 X130.483 Y102.474 E.01959
G3 X122.498 Y117.086 I-50.631 J-18.182 E.6383
G1 X123.947 Y118.301 E.07222
G1 X124.145 Y118.553 E.01225
G1 X124.225 Y118.868 E.01242
G1 X124.181 Y119.158 E.0112
G1 X124.138 Y119.234 E.00331
M204 S250
G1 X123.62 Y119.042 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.776 Y121.243 E.0909
G1 X121.628 Y121.32 E.00528
G1 X121.533 Y121.291 E.00315
G3 X121.413 Y121.23 I.012 J-.171 E.00436
G1 X121.213 Y121.062 E.00828
G1 X121.013 Y120.894 E.00828
G1 X120.812 Y120.726 E.00828
G1 F3450
G1 X120.612 Y120.558 E.00828
G1 F3300
G1 X120.412 Y120.391 E.00828
G1 F3150
G1 X120.211 Y120.223 E.00828
G1 F3600
G1 X120.07 Y120.104 E.00583
; LINE_WIDTH: 0.520326
G1 X120.043 Y120.082 E.00113
; LINE_WIDTH: 0.520776
G1 X120.004 Y120.05 E.0016
; LINE_WIDTH: 0.521226
G1 X119.964 Y120.018 E.0016
; LINE_WIDTH: 0.521686
G1 X119.925 Y119.985 E.00161
; LINE_WIDTH: 0.522136
G1 X119.886 Y119.953 E.00161
; LINE_WIDTH: 0.522596
G1 X119.847 Y119.921 E.00161
; LINE_WIDTH: 0.523046
G1 X119.808 Y119.889 E.00161
; LINE_WIDTH: 0.523196
G1 X119.795 Y119.879 E.00054
; LINE_WIDTH: 0.531646
G1 X119.767 Y119.758 E.00402
; LINE_WIDTH: 0.544336
G1 X119.724 Y119.576 E.00619
G1 X119.181 Y120.183 E.02706
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.856 J-35.523 E.30731
G1 X116.989 Y130.398 E.19629
G1 X117.374 Y130.721 E.01592
G1 X117.892 Y131.276 E.02403
G1 X118.327 Y131.906 E.02424
G3 X118.984 Y133.777 I-4.951 J2.79 E.06308
G1 X119.044 Y134.594 E.02595
G1 X118.98 Y135.355 E.02418
G1 X118.824 Y136.024 E.02175
G1 X118.547 Y136.754 E.02472
G1 X118.31 Y137.19 E.01571
G3 X129.492 Y143.495 I-19.563 J47.764 E.40749
G3 X142.39 Y157.025 I-31.404 J42.848 E.59492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.385 E1.81445
G1 X144.167 Y98.946 E.0139
G3 X141.692 Y99.076 I-2.245 J-19.067 E.07853
G1 X142.16 Y99.465 E.01926
G1 X142.233 Y99.707 E.00801
G3 X142.125 Y100.004 I-.446 J.006 E.01022
G1 X141.849 Y100.011 E.00873
G1 X141.399 Y99.68 E.01769
G1 X141.979 Y102.813 E.10088
G1 X141.968 Y102.931 E.00376
G1 X141.826 Y103.067 E.00621
G1 X141.757 Y103.078 E.00223
G1 X132.834 Y103.078 E.2825
G1 X132.665 Y103.001 E.00588
G1 X132.601 Y102.802 E.00662
G1 X132.196 Y102.91 E.01325
G1 X132.056 Y102.827 E.00516
G1 X132.009 Y102.666 E.00531
G1 X132.244 Y100.213 E.07803
; LINE_WIDTH: 0.55492
G1 X132.272 Y100.14 E.00266
; LINE_WIDTH: 0.589843
G1 X132.301 Y100.067 E.00284
; LINE_WIDTH: 0.624766
G1 X132.329 Y99.994 E.00301
; LINE_WIDTH: 0.662753
G1 X132.364 Y99.963 E.00191
; LINE_WIDTH: 0.70074
G1 X132.399 Y99.932 E.00202
; LINE_WIDTH: 0.738726
G1 X132.433 Y99.901 E.00214
; LINE_WIDTH: 0.743276
G1 X133.02 Y99.993 E.02745
G1 X133.169 Y99.189 E.03781
G1 X132.622 Y99.178 E.02532
; LINE_WIDTH: 0.735716
G1 X132.454 Y99.148 E.00779
; LINE_WIDTH: 0.698733
G1 X132.286 Y99.117 E.00738
; LINE_WIDTH: 0.66175
G1 X132.118 Y99.087 E.00697
; LINE_WIDTH: 0.624766
G1 X131.924 Y99.039 E.0077
; LINE_WIDTH: 0.589843
G1 X131.73 Y98.991 E.00725
; LINE_WIDTH: 0.55492
G1 X131.536 Y98.943 E.00679
; LINE_WIDTH: 0.519996
G1 X131.038 Y98.936 E.01576
G3 X122.131 Y116.662 I-50.994 J-14.522 E.6318
; LINE_WIDTH: 0.544336
G1 X121.593 Y117.343 E.02884
G1 X121.789 Y117.336 E.00652
; LINE_WIDTH: 0.531156
G1 X121.931 Y117.33 E.00461
; LINE_WIDTH: 0.521596
G1 X121.968 Y117.361 E.00153
; LINE_WIDTH: 0.521526
G1 X122.103 Y117.474 E.00559
; LINE_WIDTH: 0.521296
G1 X122.238 Y117.588 E.00559
; LINE_WIDTH: 0.521066
G1 X122.373 Y117.701 E.00559
; LINE_WIDTH: 0.520836
G1 X122.507 Y117.814 E.00558
; LINE_WIDTH: 0.520616
G1 X122.642 Y117.928 E.00558
; LINE_WIDTH: 0.520386
G1 X122.777 Y118.041 E.00558
; LINE_WIDTH: 0.520156
G1 X122.872 Y118.121 E.00393
; LINE_WIDTH: 0.519996
G1 X122.943 Y118.18 E.00293
G1 X123.044 Y118.265 E.00416
G1 X123.144 Y118.349 E.00416
G1 X123.245 Y118.433 E.00416
G1 X123.345 Y118.518 E.00416
G1 X123.446 Y118.602 E.00416
G1 X123.547 Y118.686 E.00416
G3 X123.622 Y118.782 I-.067 J.13 E.00398
G1 X123.673 Y118.889 E.00375
G1 X123.649 Y118.957 E.00228
; WIPE_START
M204 S10000
G1 X123.015 Y119.731 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X128.887 Y114.855 Z7.48 F36000
G1 X143.065 Y103.081 Z7.48
G1 Z7.08
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.550096
G1 F15268.731
G1 X143.064 Y102.61 E.01585
; LINE_WIDTH: 0.600036
G1 F13920.807
G1 X143.039 Y102.338 E.01007
; LINE_WIDTH: 0.649976
G1 F12791.568
G1 X143.014 Y102.066 E.01096
; LINE_WIDTH: 0.699916
G1 F11831.787
G1 X142.989 Y101.794 E.01185
; LINE_WIDTH: 0.749856
G1 F11005.983
G1 X142.964 Y101.522 E.01274
; LINE_WIDTH: 0.799796
G1 F10287.933
G1 X142.939 Y101.25 E.01363
; LINE_WIDTH: 0.849736
G1 F9657.838
G1 X142.914 Y100.978 E.01452
; WIPE_START
G1 X142.939 Y101.25 E-.10379
G1 X142.964 Y101.522 E-.10379
G1 X142.989 Y101.794 E-.10379
G1 X143.005 Y101.974 E-.06864
; WIPE_END
G1 E-.02 F1800
G1 X135.472 Y100.75 Z7.48 F36000
G1 X131.49 Y100.103 Z7.48
G1 Z7.08
G1 E.4 F1800
; LINE_WIDTH: 1.05486
G1 F7716.654
G1 X131.607 Y99.734 E.02572
; WIPE_START
G1 X131.49 Y100.103 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X127.69 Y106.722 Z7.48 F36000
G1 X121.593 Y117.343 Z7.48
G1 Z7.08
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X119.724 Y119.576 E.09682
; WIPE_START
M204 S10000
G1 X120.366 Y118.81 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.341 Y123.165 Z7.48 F36000
G1 Z7.08
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X119.792 Y122.77 I1.03 J-2.014 E.02593
G3 X118.619 Y123.951 I-24.565 J-23.23 E.06354
G2 X118.887 Y126.58 I3.722 J.949 E.103
G2 X120.95 Y128.937 I9.601 J-6.327 E.11996
G3 X122.144 Y130.822 I-3.636 J3.622 E.08587
G3 X121.224 Y135.2 I-3.808 J1.486 E.18075
G3 X121.116 Y135.986 I-4.347 J-.197 E.03033
G3 X126.067 Y138.57 I-32.345 J68.009 E.21329
G3 X126.707 Y136.477 I3.516 J-.071 E.08497
G1 X127.533 Y135.535 E.04785
G2 X129.232 Y133.65 I-6.521 J-7.588 E.09717
G2 X129.298 Y128.937 I-3.743 J-2.409 E.18964
G1 X128.472 Y127.994 E.04785
G3 X126.772 Y126.109 I6.521 J-7.588 E.09717
G3 X126.707 Y121.396 I3.743 J-2.409 E.18964
G1 X127.533 Y120.453 E.04785
G2 X129.232 Y118.568 I-6.521 J-7.588 E.09717
G2 X129.577 Y114.327 I-3.999 J-2.46 E.16862
G2 X127.773 Y112.226 I-8.857 J5.78 E.10604
G3 X126.549 Y114.223 I-27.828 J-15.677 E.08945
G1 X131.415 Y105.152 F36000
; FEATURE: Floating vertical shell
; LINE_WIDTH: 0.775576
G1 F10624.091
G1 X131.706 Y105.255 E.01491
; LINE_WIDTH: 0.76573
G1 F10767.121
G1 X131.939 Y105.295 E.01123
; LINE_WIDTH: 0.726756
G1 F11373.155
G1 X132.171 Y105.334 E.01063
; LINE_WIDTH: 0.687783
G1 F12000
G1 X132.403 Y105.374 E.01003
; LINE_WIDTH: 0.636706
G1 X132.834 Y105.393 E.01693
G2 X135.201 Y105.391 I1.082 J-133.333 E.09296
; LINE_WIDTH: 0.607336
G1 X135.723 Y105.312 E.01972
; LINE_WIDTH: 0.552912
G1 X136.245 Y105.232 E.01785
G1 X136.441 Y105.232 E.00663
G1 X137 Y105.339 E.01923
; LINE_WIDTH: 0.565816
G1 X137.252 Y105.366 E.00877
; LINE_WIDTH: 0.608776
G1 X137.503 Y105.392 E.00948
; LINE_WIDTH: 0.635977
G2 X141.755 Y105.395 I2.322 J-330.352 E.16676
G1 X141.945 Y143.65 F36000
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X141.945 Y141.307 E.08944
G1 X141.401 Y140.248 E.04549
G3 X141.945 Y136.298 I3.815 J-1.487 E.159
G1 X141.945 Y126.226 E.38452
G1 X141.401 Y125.166 E.04549
G3 X141.945 Y121.216 I3.815 J-1.487 E.159
G1 X141.945 Y111.145 E.38452
G1 X141.853 Y111.028 E.00569
G3 X141.788 Y106.315 I3.743 J-2.409 E.18964
G2 X141.945 Y105.988 I-.108 J-.253 E.01512
G1 X138.078 Y105.991 E.14767
G3 X136.343 Y105.656 I-.264 J-3.299 E.06828
G3 X135.418 Y105.957 I-1.566 J-3.248 E.03724
G2 X134.313 Y107.257 I4.533 J4.973 E.06535
G2 X134.247 Y111.97 I3.743 J2.409 E.18964
G1 X135.073 Y112.913 E.04786
G3 X136.773 Y114.798 I-6.521 J7.588 E.09717
G3 X136.838 Y119.511 I-3.743 J2.409 E.18964
G1 X136.012 Y120.453 E.04785
G2 X134.313 Y122.339 I6.521 J7.587 E.09717
G2 X134.247 Y127.052 I3.743 J2.409 E.18964
G1 X135.073 Y127.994 E.04786
G3 X136.773 Y129.879 I-6.521 J7.588 E.09717
G3 X136.838 Y134.592 I-3.743 J2.409 E.18964
G1 X136.012 Y135.535 E.04785
G2 X134.313 Y137.42 I6.521 J7.587 E.09717
G2 X133.968 Y141.662 I3.999 J2.46 E.16862
G2 X136.032 Y144.018 I9.601 J-6.326 E.11996
G3 X137.473 Y147.453 I-3.425 J3.457 E.14593
G3 X141.516 Y152.017 I-39.289 J38.882 E.23293
G3 X141.945 Y151.379 I1.92 J.828 E.02952
G1 X141.945 Y149.036 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 7.24
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y150.036 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L45
M991 S0 P44 ;notify layer change


G17
G3 Z7.48 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X124.985 Y120.038
G1 Z7.24
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X124.839 Y120.273 E.01056
G3 X124.486 Y120.693 I-3.16 J-2.291 E.02096
G1 X123.202 Y122.226 E.07636
G1 X122.836 Y122.564 E.01901
G1 X122.363 Y122.81 E.02036
G1 X121.871 Y122.916 E.01919
G1 X121.303 Y122.881 E.02175
G1 X121.021 Y122.799 E.0112
M73 P73 R5
G1 X120.556 Y122.547 E.02019
G1 X119.868 Y121.976 E.03415
G3 X114.848 Y126.662 I-41.897 J-39.843 E.26235
G1 X118.015 Y129.012 E.15055
G3 X118.933 Y129.85 I-4.607 J5.969 E.0475
G1 X119.566 Y130.648 E.03892
G1 X120.075 Y131.529 E.03883
G1 X120.449 Y132.477 E.03892
G1 X120.681 Y133.47 E.03893
G1 X120.765 Y134.489 E.03904
G1 X120.698 Y135.506 E.03893
G1 X120.538 Y136.268 E.0297
G3 X130.504 Y142.099 I-22.196 J49.366 E.44166
G3 X142.443 Y154.069 I-32.642 J44.497 E.64803
G1 X142.443 Y104.638 E1.88721
G1 X141.885 Y104.8 E.02219
G1 X141.683 Y104.804 E.00769
G1 X141.585 Y104.918 E.00577
G1 X141.142 Y105.193 E.0199
G1 X140.718 Y105.327 E.01698
G1 X140.345 Y105.363 E.01431
G1 X139.446 Y105.363 E.03432
G1 X138.922 Y105.292 E.02018
G1 X138.487 Y105.111 E.01799
G1 X138.084 Y104.804 E.01935
G1 X133.731 Y104.804 E.1662
G1 X133.257 Y105.145 E.0223
G1 X132.82 Y105.308 E.0178
G1 X132.33 Y105.363 E.01884
G1 X131.737 Y105.316 E.0227
G1 X131.034 Y105.003 E.02935
G1 X130.846 Y104.816 E.01015
G3 X124.027 Y117.012 I-50.852 J-20.427 E.53493
G1 X124.032 Y117.052 E.00155
G1 X124.597 Y117.526 E.02815
G1 X124.968 Y117.94 E.02124
G1 X125.125 Y118.224 E.01238
G1 X125.283 Y118.808 E.02311
G1 X125.265 Y119.357 E.02095
G1 X125.127 Y119.81 E.01808
G1 X125.033 Y119.962 E.00682
G1 X124.514 Y119.697 F36000
G1 F13446.369
G1 X124.447 Y119.823 E.00548
G3 X124.038 Y120.317 I-7.45 J-5.765 E.02449
G1 X122.753 Y121.85 E.07636
G1 X122.413 Y122.141 E.01707
G1 X122.065 Y122.29 E.01448
G1 X121.718 Y122.337 E.01336
G1 X121.251 Y122.259 E.01808
G1 X120.831 Y122.019 E.01847
G1 X119.818 Y121.17 E.05046
G3 X113.893 Y126.683 I-40.272 J-37.34 E.30928
G1 X117.677 Y129.49 E.17987
G1 X118.524 Y130.269 E.04392
G1 X119.1 Y131.002 E.03563
G1 X119.562 Y131.811 E.03555
G1 X119.901 Y132.681 E.03564
G1 X120.108 Y133.591 E.03564
G1 X120.18 Y134.525 E.03575
G1 X120.114 Y135.456 E.03564
G1 X119.931 Y136.289 E.03258
G1 X119.836 Y136.599 E.01238
G3 X130.932 Y143.157 I-21.096 J48.365 E.49335
G3 X142.92 Y155.769 I-32.373 J42.775 E.66728
G1 X143.029 Y155.749 E.00421
G1 X143.029 Y116.42 E1.50154
G3 X143.039 Y103.516 I1866.004 J-5 E.49265
; LINE_WIDTH: 0.599926
G1 F13923.516
G1 X143.053 Y103.3 E.00799
; LINE_WIDTH: 0.572611
G1 F14630.069
G1 X143.066 Y103.083 E.00761
G1 X143.009 Y103.289 E.00748
; LINE_WIDTH: 0.599926
G1 F13923.516
G1 X142.953 Y103.494 E.00786
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.411 Y104.052 E.02969
G1 X141.847 Y104.215 E.02241
G1 X141.305 Y104.218 E.02071
G1 X141.278 Y104.408 E.00734
G1 X140.903 Y104.658 E.01723
G1 X140.606 Y104.752 E.01188
G1 X139.446 Y104.778 E.04429
G1 X139.08 Y104.727 E.01412
G1 X138.775 Y104.601 E.01258
G1 X138.478 Y104.374 E.01427
G1 X138.414 Y104.218 E.00646
G1 X133.51 Y104.218 E.18722
G1 X132.987 Y104.625 E.02528
G3 X131.84 Y104.738 I-.756 J-1.779 E.04468
G1 X131.373 Y104.526 E.01959
G1 X131.028 Y104.17 E.01892
G1 X130.818 Y103.644 E.02161
G1 X130.805 Y103.301 E.01312
G3 X123.208 Y117.126 I-51.115 J-19.091 E.60439
G1 X124.221 Y117.975 E.05047
G1 X124.48 Y118.265 E.01486
G1 X124.67 Y118.696 E.018
G1 X124.707 Y119.079 E.01468
G1 X124.629 Y119.48 E.0156
G1 X124.556 Y119.617 E.00592
G1 X123.988 Y119.445 F36000
G1 F13446.369
G1 X123.941 Y119.52 E.00338
G1 X123.589 Y119.941 E.02095
G1 X122.304 Y121.473 E.07636
G1 X122.082 Y121.656 E.01096
G1 X121.773 Y121.749 E.01233
G1 X121.433 Y121.702 E.01308
G1 X121.207 Y121.57 E.01
G1 X119.758 Y120.355 E.07222
G3 X112.93 Y126.698 I-39.355 J-35.517 E.35627
G1 X117.323 Y129.957 E.20885
G1 X118.114 Y130.688 E.04111
G1 X118.634 Y131.357 E.03234
G1 X119.048 Y132.093 E.03227
G1 X119.352 Y132.885 E.03236
G1 X119.535 Y133.712 E.03236
G1 X119.595 Y134.56 E.03246
G1 X119.531 Y135.405 E.03235
G1 X119.361 Y136.154 E.0293
G1 X119.08 Y136.91 E.03081
G3 X130.579 Y143.624 I-20.648 J48.569 E.50972
G3 X142.648 Y156.415 I-31.963 J42.246 E.67454
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y101.443 E2.09194
G1 X143.615 Y101.043 E.01527
; LINE_WIDTH: 0.657734
G1 F12632.391
G1 X143.596 Y100.815 E.00927
; LINE_WIDTH: 0.695471
G1 F11911.336
G1 X143.577 Y100.588 E.00984
; LINE_WIDTH: 0.733209
G1 F11268.15
G1 X143.558 Y100.36 E.0104
; LINE_WIDTH: 0.770946
G1 F10690.869
G3 X143.543 Y99.617 I4.064 J-.452 E.03576
; LINE_WIDTH: 0.762346
G1 F10817.16
G3 X142.867 Y99.703 I-1.812 J-11.509 E.03238
; LINE_WIDTH: 0.770946
G1 F10690.869
G1 X142.808 Y100.069 E.01781
G1 X142.724 Y100.18 E.0067
; LINE_WIDTH: 0.733209
G1 F11268.15
G1 X142.64 Y100.292 E.00636
; LINE_WIDTH: 0.695471
G1 F11731.751
G1 X142.556 Y100.403 E.00601
; LINE_WIDTH: 0.657734
G1 F12204.698
G1 X142.471 Y100.514 E.00567
; LINE_WIDTH: 0.619996
G1 F13432.092
G1 X142.122 Y100.538 E.01336
G1 F13446.369
G1 X142.195 Y100.932 E.01527
G1 X142.526 Y102.719 E.06942
G3 X142.445 Y103.222 I-.767 J.135 E.0198
G1 X142.131 Y103.538 E.01701
G1 X141.759 Y103.632 E.01465
G1 X140.926 Y103.632 E.03182
G1 X140.857 Y104 E.01426
G1 X140.663 Y104.124 E.00881
G1 X140.345 Y104.192 E.01242
G1 X139.446 Y104.192 E.03432
G1 X139.158 Y104.136 E.01121
G1 X138.894 Y103.962 E.01209
G1 X138.758 Y103.632 E.0136
G1 X133.289 Y103.632 E.20881
G1 X132.874 Y103.998 E.02109
G1 X132.553 Y104.168 E.01388
G1 X132.16 Y104.191 E.01501
G1 X131.979 Y104.169 E.00699
G1 X131.699 Y104.038 E.01181
G1 X131.515 Y103.845 E.01017
G1 X131.393 Y103.531 E.01287
G3 X131.446 Y102.729 I8.682 J.174 E.0307
G1 X131.523 Y101.932 E.03054
G1 X131.561 Y101.534 E.01527
G1 F13260.888
G1 X131.6 Y101.136 E.01527
G1 F11872.098
G1 X131.638 Y100.738 E.01527
; LINE_WIDTH: 0.665112
G1 F10560.107
G1 X131.622 Y100.671 E.00281
; LINE_WIDTH: 0.710227
G1 F10343.582
G1 X131.605 Y100.605 E.00301
; LINE_WIDTH: 0.755343
G1 F10129.276
G1 X131.589 Y100.539 E.00321
; LINE_WIDTH: 0.800458
G1 F9917.207
G1 X131.573 Y100.472 E.00341
; LINE_WIDTH: 0.845574
G1 F9707.389
G1 X131.557 Y100.406 E.00362
; LINE_WIDTH: 0.89069
G1 F9195.97
G1 X131.54 Y100.339 E.00382
; LINE_WIDTH: 0.935805
G1 F8735.743
G1 X131.524 Y100.273 E.00402
; LINE_WIDTH: 0.980921
G1 F8319.385
G1 X131.508 Y100.207 E.00422
; LINE_WIDTH: 1.02604
G1 F7940.91
G1 X131.491 Y100.14 E.00442
G1 X131.445 Y100.199 E.00484
; LINE_WIDTH: 0.97679
G1 F8355.848
G1 X131.399 Y100.258 E.0046
; LINE_WIDTH: 0.927544
G1 F8816.541
G1 X131.352 Y100.316 E.00436
; LINE_WIDTH: 0.878298
G1 F9330.998
G1 X131.306 Y100.375 E.00412
; LINE_WIDTH: 0.829051
G1 F9909.213
G1 X131.259 Y100.434 E.00388
; LINE_WIDTH: 0.779805
G1 F10141.408
G1 X131.213 Y100.492 E.00364
; LINE_WIDTH: 0.730559
G1 F10376.312
G1 X131.166 Y100.551 E.0034
; LINE_WIDTH: 0.681313
G1 F10613.861
G1 X131.12 Y100.61 E.00316
; LINE_WIDTH: 0.632066
G1 F10678.249
G1 X131.108 Y100.626 E.00078
; LINE_WIDTH: 0.619996
G1 F11997.344
G1 X130.983 Y101.006 E.01527
G1 F13393.239
G1 X130.857 Y101.386 E.01527
G1 F13446.369
G1 X130.702 Y101.856 E.01889
G3 X122.395 Y117.208 I-50.307 J-17.297 E.66949
G1 X123.845 Y118.423 E.07222
G1 X124.055 Y118.702 E.01334
G1 X124.123 Y119.02 E.0124
G1 X124.056 Y119.336 E.01231
G1 X124.036 Y119.369 E.00149
M204 S250
G1 X123.517 Y119.165 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.877 Y121.122 E.08086
G1 X121.726 Y121.198 E.00534
G1 X121.634 Y121.169 E.00308
G3 X121.52 Y121.111 I.014 J-.167 E.00414
G1 X121.337 Y120.958 E.00755
G1 X121.155 Y120.805 E.00755
G1 X120.972 Y120.652 E.00755
G1 F3450
G1 X120.789 Y120.499 E.00755
G1 F3300
G1 X120.607 Y120.345 E.00755
G1 F3150
G1 X120.424 Y120.192 E.00755
G1 F3600
G1 X120.295 Y120.084 E.00532
; LINE_WIDTH: 0.520326
G1 X120.256 Y120.052 E.00164
; LINE_WIDTH: 0.520776
G1 X120.199 Y120.005 E.00232
; LINE_WIDTH: 0.521226
G1 X120.143 Y119.958 E.00233
; LINE_WIDTH: 0.521686
G1 X120.086 Y119.912 E.00233
; LINE_WIDTH: 0.522136
G1 X120.03 Y119.865 E.00233
; LINE_WIDTH: 0.522596
G1 X119.973 Y119.818 E.00233
; LINE_WIDTH: 0.523046
G1 X119.917 Y119.771 E.00233
; LINE_WIDTH: 0.523196
G1 X119.898 Y119.756 E.00078
; LINE_WIDTH: 0.531646
G1 X119.869 Y119.635 E.00402
; LINE_WIDTH: 0.544336
G1 X119.827 Y119.454 E.00619
G1 X119.181 Y120.183 E.03237
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.857 J-35.524 E.30731
G1 X116.989 Y130.398 E.1963
G1 X117.728 Y131.083 E.0319
G1 X118.194 Y131.691 E.02425
G1 X118.564 Y132.36 E.0242
G1 X118.833 Y133.078 E.02427
G1 X118.994 Y133.827 E.02426
G1 X119.044 Y134.594 E.02434
G1 X118.98 Y135.358 E.02425
G1 X118.824 Y136.026 E.02173
G1 X118.548 Y136.753 E.02461
G1 X118.31 Y137.19 E.01576
G3 X130.249 Y144.067 I-19.515 J47.679 E.43753
G3 X142.39 Y157.025 I-31.646 J41.818 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.383 E1.8145
G1 X144.167 Y98.946 E.01385
G3 X141.695 Y99.078 I-2.243 J-18.751 E.07846
G1 X142.16 Y99.465 E.01916
G1 X142.232 Y99.71 E.00808
G1 X142.189 Y99.961 E.00806
G1 X142.113 Y100.093 E.00485
G1 X141.845 Y100.112 E.0085
G1 X141.432 Y99.848 E.01552
G1 X141.982 Y102.815 E.09553
G3 X141.759 Y103.08 I-.245 J.02 E.01214
G1 X140.478 Y103.08 E.04056
G2 X140.538 Y103.297 I.284 J.039 E.00734
G1 X140.489 Y103.588 E.00934
G1 X140.345 Y103.639 E.00483
G1 X139.446 Y103.639 E.02846
G1 X139.286 Y103.572 E.00549
G1 X139.22 Y103.413 E.00544
G1 X139.22 Y103.08 E.01057
G1 X133.08 Y103.08 E.19441
G1 X132.509 Y103.583 E.0241
G3 X132.109 Y103.632 I-.269 J-.534 E.01299
G1 X131.975 Y103.538 E.00519
G1 X131.938 Y103.391 E.00479
G1 X132.244 Y100.207 E.10128
; LINE_WIDTH: 0.554576
G1 X132.273 Y100.136 E.00259
; LINE_WIDTH: 0.589156
G1 X132.301 Y100.065 E.00276
; LINE_WIDTH: 0.623736
G1 X132.329 Y99.994 E.00293
; LINE_WIDTH: 0.661753
G1 X132.364 Y99.963 E.00191
; LINE_WIDTH: 0.69977
G1 X132.399 Y99.932 E.00203
; LINE_WIDTH: 0.737786
G1 X132.434 Y99.901 E.00214
; LINE_WIDTH: 0.742146
G1 X133.019 Y99.994 E.02732
G1 X133.167 Y99.191 E.03769
G1 X132.622 Y99.179 E.02518
; LINE_WIDTH: 0.734736
G1 X132.454 Y99.149 E.00779
; LINE_WIDTH: 0.697736
G1 X132.286 Y99.118 E.00738
; LINE_WIDTH: 0.660736
G1 X132.118 Y99.088 E.00697
; LINE_WIDTH: 0.623736
G1 X131.924 Y99.04 E.0077
; LINE_WIDTH: 0.589156
G1 X131.729 Y98.992 E.00725
; LINE_WIDTH: 0.554576
G1 X131.535 Y98.944 E.00679
; LINE_WIDTH: 0.519996
G1 X131.039 Y98.936 E.0157
G3 X122.131 Y116.662 I-50.906 J-14.48 E.63183
; LINE_WIDTH: 0.521596
G1 X121.795 Y117.067 E.0167
; LINE_WIDTH: 0.544336
G1 X121.49 Y117.466 E.01669
G1 X121.686 Y117.458 E.00652
; LINE_WIDTH: 0.531166
G1 X121.829 Y117.453 E.00461
; LINE_WIDTH: 0.521596
G1 X121.844 Y117.466 E.00064
; LINE_WIDTH: 0.521526
G1 X121.9 Y117.513 E.00234
; LINE_WIDTH: 0.521296
G1 X121.957 Y117.561 E.00234
; LINE_WIDTH: 0.521066
G1 X122.013 Y117.608 E.00234
; LINE_WIDTH: 0.520836
G1 X122.069 Y117.656 E.00234
; LINE_WIDTH: 0.520616
G1 X122.126 Y117.703 E.00234
; LINE_WIDTH: 0.520386
G1 X122.182 Y117.751 E.00234
; LINE_WIDTH: 0.520156
G1 X122.222 Y117.784 E.00164
; LINE_WIDTH: 0.519996
G1 X122.351 Y117.892 E.00532
G1 F3150
G1 X122.533 Y118.046 E.00755
G1 F3300
G1 X122.716 Y118.199 E.00755
G1 F3450
G1 X122.899 Y118.352 E.00755
G1 F3600
G1 X123.082 Y118.505 E.00755
G1 X123.265 Y118.659 E.00755
G1 X123.447 Y118.812 E.00755
G3 X123.524 Y118.912 I-.086 J.145 E.00408
G1 X123.569 Y118.996 E.00301
G1 X123.544 Y119.079 E.00276
; WIPE_START
M204 S10000
G1 X122.911 Y119.854 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X128.472 Y114.626 Z7.64 F36000
G1 X142.921 Y101.043 Z7.64
G1 Z7.24
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.835776
G1 F9826.064
G1 X142.945 Y101.304 E.01372
; LINE_WIDTH: 0.78774
G1 F10452.569
G1 X142.969 Y101.566 E.0129
; LINE_WIDTH: 0.739703
G1 F11164.408
G1 X142.993 Y101.827 E.01208
; LINE_WIDTH: 0.691666
G1 F11980.285
G1 X143.017 Y102.089 E.01126
; LINE_WIDTH: 0.64363
G1 F12924.811
G1 X143.041 Y102.351 E.01043
; LINE_WIDTH: 0.595593
G1 F14031.015
G1 X143.065 Y102.612 E.00961
; LINE_WIDTH: 0.547556
G1 F15344.298
G1 X143.066 Y103.083 E.01576
; WIPE_START
G1 X143.065 Y102.612 E-.179
G1 X143.041 Y102.351 E-.09983
G1 X143.017 Y102.089 E-.09983
G1 X143.017 Y102.085 E-.00135
; WIPE_END
G1 E-.02 F1800
G1 X135.493 Y100.8 Z7.64 F36000
G1 X131.485 Y100.115 Z7.64
G1 Z7.24
G1 E.4 F1800
; LINE_WIDTH: 1.05512
G1 F7714.689
G1 X131.606 Y99.735 E.0265
; WIPE_START
G1 X131.485 Y100.115 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X127.676 Y106.728 Z7.64 F36000
G1 X121.49 Y117.466 Z7.64
G1 Z7.24
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X119.827 Y119.454 E.08619
; WIPE_START
M204 S10000
G1 X120.468 Y118.687 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.307 Y122.975 Z7.64 F36000
G1 Z7.24
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X119.904 Y122.654 I.615 J-1.184 E.0198
G3 X118.619 Y123.952 I-25.783 J-24.253 E.06974
G2 X118.801 Y126.109 I4.84 J.677 E.08335
G2 X120.158 Y127.994 I4.738 J-1.98 E.08946
G3 X121.774 Y129.879 I-5.658 J6.482 E.09511
G3 X121.606 Y134.592 I-3.958 J2.219 E.18933
G1 X121.241 Y135.001 E.02092
G3 X121.111 Y135.983 I-4.713 J-.127 E.0379
G3 X126.115 Y138.599 I-32.029 J67.367 E.21565
G1 X126.118 Y138.362 E.00904
G1 X126.342 Y137.42 E.03699
G3 X127.699 Y135.535 I4.738 J1.98 E.08946
G2 X129.314 Y133.65 I-5.658 J-6.482 E.09511
G2 X129.147 Y128.937 I-3.958 J-2.219 E.18933
G2 X127.38 Y127.052 I-20.734 J17.665 E.09869
G3 X126.342 Y122.339 I3.538 J-3.25 E.19303
G3 X127.699 Y120.453 I4.738 J1.98 E.08946
G2 X129.314 Y118.568 I-5.658 J-6.482 E.09511
G2 X129.147 Y113.855 I-3.958 J-2.219 E.18933
G2 X127.721 Y112.317 I-16.907 J14.249 E.08011
G3 X126.493 Y114.313 I-28.021 J-15.859 E.08945
G1 X141.945 Y143.771 F36000
G1 F13446.283
G1 X141.945 Y141.428 E.08944
G3 X141.939 Y136.477 I3.578 J-2.48 E.20105
G1 X141.945 Y126.347 E.38678
G3 X141.939 Y121.396 I3.578 J-2.48 E.20105
G1 X141.945 Y111.266 E.38678
G3 X141.939 Y106.315 I3.578 J-2.48 E.20105
G1 X141.945 Y105.293 E.039
G1 X141.35 Y105.647 E.02647
G3 X139.439 Y105.861 I-1.42 J-4.049 E.07404
G3 X137.921 Y105.302 I-.001 J-2.336 E.06307
G1 X135.909 Y105.302 E.0768
G2 X134.231 Y107.257 I10.431 J10.648 E.09851
G2 X134.398 Y111.97 I3.958 J2.219 E.18933
G2 X136.165 Y113.855 I20.729 J-17.661 E.09869
G3 X137.203 Y118.568 I-3.538 J3.25 E.19303
G3 X135.846 Y120.453 I-4.738 J-1.98 E.08946
G2 X134.231 Y122.339 I5.658 J6.482 E.09511
G2 X134.398 Y127.052 I3.958 J2.219 E.18933
G2 X136.165 Y128.937 I20.729 J-17.661 E.09869
G3 X137.203 Y133.65 I-3.538 J3.25 E.19303
G3 X135.846 Y135.535 I-4.738 J-1.98 E.08946
G2 X134.231 Y137.42 I5.658 J6.482 E.09511
G2 X134.398 Y142.133 I3.958 J2.219 E.18933
G2 X136.165 Y144.018 I20.729 J-17.661 E.09869
G3 X137.431 Y147.407 I-3.373 J3.191 E.14187
G3 X141.617 Y152.147 I-45.124 J44.067 E.24153
G3 X141.945 Y151.551 I1.446 J.409 E.02619
G1 X141.945 Y149.209 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 7.4
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y150.209 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L46
M991 S0 P45 ;notify layer change


G17
G3 Z7.64 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X124.881 Y120.163
G1 Z7.4
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X124.736 Y120.395 E.01048
G1 X123.304 Y122.103 E.08509
G1 X122.917 Y122.456 E.02
G1 X122.452 Y122.692 E.0199
G1 X121.977 Y122.793 E.01853
G1 X121.432 Y122.763 E.02087
G1 X121.199 Y122.702 E.00917
G1 X120.721 Y122.468 E.02035
G3 X119.978 Y121.859 I23.914 J-29.94 E.03667
G3 X114.848 Y126.662 I-40.386 J-37.99 E.26849
G1 X118.015 Y129.011 E.15054
G1 X118.485 Y129.402 E.02334
G1 X119.156 Y130.104 E.03707
G1 X119.749 Y130.931 E.03886
G1 X120.215 Y131.838 E.03891
G1 X120.496 Y132.63 E.03212
G1 X120.684 Y133.486 E.03344
G3 X120.698 Y135.504 I-7.966 J1.063 E.07726
G1 X120.538 Y136.268 E.0298
G3 X130.505 Y142.099 I-22.299 J49.545 E.44169
G3 X142.443 Y154.069 I-32.7 J44.554 E.64798
G1 X142.443 Y105.075 E1.87054
G1 X142.175 Y105.249 E.01221
G1 X141.512 Y105.365 E.02572
G1 X138.872 Y105.365 E.10077
G1 X138.39 Y105.305 E.01854
G1 X137.907 Y105.11 E.0199
G1 X137.512 Y104.806 E.01904
G1 X134.386 Y104.806 E.11933
G1 X134.132 Y105.065 E.01386
G1 X133.571 Y105.306 E.02332
G1 X133.088 Y105.365 E.01859
G1 X132.157 Y105.363 E.03552
G1 X131.702 Y105.308 E.01751
G1 X131.034 Y105.003 E.028
G1 X130.846 Y104.816 E.01015
G3 X123.914 Y117.162 I-50.834 J-20.421 E.5421
G3 X124.713 Y117.863 I-4.048 J5.418 E.04063
G1 X125.033 Y118.373 E.02297
G1 X125.181 Y118.94 E.0224
G1 X125.162 Y119.48 E.0206
G1 X125.023 Y119.936 E.0182
G1 X124.929 Y120.086 E.00678
G1 X124.383 Y119.865 F36000
G1 F13446.369
G1 X124.287 Y120.019 E.00692
G3 X124.14 Y120.194 I-1.744 J-1.312 E.00873
G1 X122.856 Y121.727 E.07636
G1 X122.541 Y122.002 E.01595
G1 X122.125 Y122.178 E.01725
G1 X121.683 Y122.209 E.01692
G1 X121.383 Y122.146 E.01171
G1 X120.934 Y121.896 E.01962
G1 X119.927 Y121.052 E.05015
G3 X113.893 Y126.683 I-40.159 J-36.988 E.3154
G1 X117.678 Y129.491 E.17995
G1 X118.107 Y129.85 E.02136
G1 X118.727 Y130.502 E.03432
G1 X119.266 Y131.263 E.03559
G1 X119.688 Y132.095 E.03563
G3 X120.115 Y135.454 I-6.163 J2.489 E.13072
G1 X119.932 Y136.286 E.03254
G1 X119.836 Y136.599 E.01249
G3 X130.932 Y143.157 I-21.112 J48.391 E.49334
G3 X142.92 Y155.769 I-32.175 J42.587 E.66731
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.717 E1.98651
G1 X143.048 Y103.401 E.01209
; LINE_WIDTH: 0.581376
G1 F14395.659
G1 X143.068 Y103.085 E.01129
G1 X142.965 Y103.374 E.01093
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.862 Y103.663 E.0117
G1 X142.494 Y104.362 E.03016
G1 X141.976 Y104.698 E.0236
G1 X141.512 Y104.78 E.018
G1 X138.872 Y104.78 E.10077
G1 X138.403 Y104.697 E.01817
G1 X138.197 Y104.601 E.00869
G1 X137.9 Y104.373 E.0143
G1 X137.837 Y104.22 E.00629
G1 X134.162 Y104.22 E.14032
G1 X133.82 Y104.57 E.01867
G1 X133.427 Y104.738 E.01631
G1 X133.089 Y104.78 E.01301
G1 X132.159 Y104.777 E.03552
G1 X131.84 Y104.738 E.01225
G1 X131.373 Y104.526 E.01959
G1 X131.028 Y104.17 E.01892
G1 X130.818 Y103.645 E.02161
G1 X130.805 Y103.298 E.01323
G3 X123.111 Y117.253 I-50.906 J-18.968 E.61061
G1 X124.118 Y118.097 E.05015
G1 X124.41 Y118.438 E.01713
G1 X124.575 Y118.855 E.01712
G1 X124.596 Y119.31 E.01737
G1 X124.488 Y119.697 E.01537
G1 X124.431 Y119.789 E.00412
G1 X123.892 Y119.542 F36000
G1 F13446.369
G1 X123.838 Y119.643 E.00437
G3 X123.691 Y119.818 I-1.076 J-.753 E.00874
G1 X122.407 Y121.351 E.07636
G1 X122.171 Y121.54 E.01153
G1 X121.877 Y121.626 E.01171
G1 X121.566 Y121.59 E.01194
G1 X121.31 Y121.447 E.0112
G1 X119.86 Y120.232 E.07222
G3 X112.93 Y126.698 I-38.513 J-34.336 E.36238
G1 X117.324 Y129.958 E.2089
G1 X117.73 Y130.298 E.02024
G1 X118.297 Y130.901 E.03158
G1 X118.783 Y131.594 E.03231
G1 X119.162 Y132.351 E.03235
G3 X119.531 Y135.403 I-5.668 J2.234 E.11865
G1 X119.362 Y136.151 E.02927
G1 X119.08 Y136.91 E.03092
G3 X130.579 Y143.624 I-20.65 J48.573 E.50972
G3 X142.647 Y156.415 I-31.78 J42.073 E.67457
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y101.496 E2.08988
G1 X143.615 Y101.096 E.01527
; LINE_WIDTH: 0.659466
G1 F12597.38
G1 X143.595 Y100.876 E.00904
; LINE_WIDTH: 0.698936
G1 F11849.234
G1 X143.575 Y100.655 E.00961
; LINE_WIDTH: 0.738406
G1 F11184.968
G1 X143.555 Y100.434 E.01018
; LINE_WIDTH: 0.777876
G1 F10591.226
G3 X143.543 Y99.618 I3.983 J-.467 E.03964
; LINE_WIDTH: 0.762536
G1 F10814.338
G3 X142.865 Y99.704 I-1.333 J-7.78 E.03245
; LINE_WIDTH: 0.777876
G1 F10591.226
G1 X142.798 Y100.149 E.02182
G1 X142.716 Y100.257 E.00659
; LINE_WIDTH: 0.738406
G1 F11184.968
G1 X142.634 Y100.366 E.00624
; LINE_WIDTH: 0.698936
G1 F11634.596
G1 X142.552 Y100.474 E.00589
; LINE_WIDTH: 0.659466
G1 F12093.036
G1 X142.47 Y100.583 E.00554
; LINE_WIDTH: 0.619996
G1 F13252.693
G1 X142.14 Y100.624 E.0127
G1 F13446.369
G1 X142.213 Y101.018 E.01527
G1 X142.527 Y102.714 E.06588
G1 X142.529 Y102.991 E.01055
G3 X142.289 Y103.464 I-1.306 J-.365 E.02041
G1 X142.273 Y103.577 E.00434
G1 X142.072 Y103.956 E.01636
G1 X141.776 Y104.148 E.01347
G1 X141.512 Y104.194 E.01027
G1 X138.872 Y104.194 E.10077
G1 X138.581 Y104.138 E.0113
G1 X138.317 Y103.962 E.01212
G1 X138.183 Y103.634 E.0135
G1 X133.877 Y103.634 E.1644
G3 X133.508 Y104.074 I-1.247 J-.672 E.02207
G1 X133.189 Y104.188 E.01293
G3 X131.979 Y104.169 I-.437 J-10.857 E.04625
G1 X131.699 Y104.038 E.01181
G1 X131.515 Y103.845 E.01017
G1 X131.393 Y103.529 E.01296
G3 X131.448 Y102.712 I38.083 J2.164 E.03126
G1 X131.501 Y102.165 E.02096
G1 X131.539 Y101.767 E.01527
G1 F12788.249
G1 X131.577 Y101.369 E.01527
; LINE_WIDTH: 0.669035
G1 F11425.123
G1 X131.567 Y101.216 E.00636
; LINE_WIDTH: 0.718074
G1 F10921.355
G1 X131.557 Y101.062 E.00686
; LINE_WIDTH: 0.767112
G1 F10428.946
G1 X131.548 Y100.908 E.00735
; LINE_WIDTH: 0.816151
G1 F9947.897
G1 X131.538 Y100.755 E.00784
; LINE_WIDTH: 0.86519
G1 F9478.204
G1 X131.528 Y100.601 E.00833
; LINE_WIDTH: 0.914229
G1 F8949.958
G1 X131.518 Y100.448 E.00882
; LINE_WIDTH: 0.963268
G1 F8477.485
G1 X131.508 Y100.294 E.00932
; LINE_WIDTH: 1.01231
G1 F8052.394
G1 X131.498 Y100.141 E.00981
; LINE_WIDTH: 1.03834
G1 F7843.626
G1 X131.49 Y100.103 E.00256
G1 X131.466 Y100.134 E.00258
; LINE_WIDTH: 1.01231
G1 F8052.394
G1 X131.397 Y100.272 E.00981
; LINE_WIDTH: 0.963268
G1 F8477.485
G1 X131.327 Y100.409 E.00932
; LINE_WIDTH: 0.914229
G1 F8949.958
G1 X131.258 Y100.546 E.00882
; LINE_WIDTH: 0.86519
G1 F9478.204
G1 X131.189 Y100.684 E.00833
; LINE_WIDTH: 0.816151
G1 F10072.716
G1 X131.12 Y100.821 E.00784
; LINE_WIDTH: 0.767112
G1 F10556.733
G1 X131.05 Y100.958 E.00735
; LINE_WIDTH: 0.718074
G1 F11052.107
G1 X130.981 Y101.096 E.00686
; LINE_WIDTH: 0.669035
G1 F11558.824
G1 X130.912 Y101.233 E.00636
; LINE_WIDTH: 0.619996
G1 F12929.679
G1 X130.779 Y101.61 E.01527
G1 F13446.369
G1 X130.646 Y101.988 E.01527
G1 X130.26 Y103.081 E.04426
G3 X124.955 Y113.625 I-50.791 J-18.946 E.45156
G3 X122.292 Y117.331 I-34.66 J-22.1 E.17431
G1 X123.742 Y118.546 E.07222
G1 X123.908 Y118.741 E.00978
G1 X124.016 Y119.062 E.01294
G1 X123.992 Y119.352 E.01111
M73 P74 R5
G1 X123.934 Y119.462 E.00474
M204 S250
G1 X123.415 Y119.288 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.983 Y120.996 E.07056
G1 X121.829 Y121.076 E.00548
G1 X121.737 Y121.046 E.00308
G3 X121.627 Y120.992 I.011 J-.16 E.00397
G1 X121.462 Y120.854 E.00682
G1 X121.297 Y120.715 E.00682
G1 X121.132 Y120.577 E.00682
G1 F3450
G1 X120.967 Y120.439 E.00682
G1 F3300
G1 X120.802 Y120.3 E.00682
G1 F3150
G1 X120.637 Y120.162 E.00682
G1 F3600
G1 X120.521 Y120.064 E.0048
; LINE_WIDTH: 0.520326
G1 X120.469 Y120.021 E.00214
; LINE_WIDTH: 0.520776
G1 X120.395 Y119.96 E.00304
; LINE_WIDTH: 0.521226
G1 X120.321 Y119.899 E.00305
; LINE_WIDTH: 0.521686
G1 X120.247 Y119.837 E.00305
; LINE_WIDTH: 0.522136
G1 X120.173 Y119.776 E.00305
; LINE_WIDTH: 0.522596
G1 X120.099 Y119.715 E.00305
; LINE_WIDTH: 0.523046
G1 X120.025 Y119.654 E.00306
; LINE_WIDTH: 0.523196
G1 X120.001 Y119.633 E.00102
; LINE_WIDTH: 0.531656
G1 X119.972 Y119.512 E.00402
; LINE_WIDTH: 0.544336
G1 X119.929 Y119.331 E.00619
G1 X119.181 Y120.182 E.03768
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-40.855 J-37.72 E.30727
G1 X116.99 Y130.398 E.1963
G1 X117.374 Y130.721 E.01591
G1 X117.892 Y131.277 E.02405
G1 X118.327 Y131.906 E.02423
G1 X118.665 Y132.594 E.02426
G1 X118.859 Y133.173 E.01932
G3 X118.98 Y135.356 I-5.652 J1.409 E.06963
G1 X118.824 Y136.023 E.02171
G1 X118.548 Y136.753 E.02473
G1 X118.31 Y137.19 E.01573
G3 X130.249 Y144.067 I-19.805 J48.181 E.4375
G3 X142.39 Y157.025 I-31.645 J41.818 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.381 E1.81456
G1 X144.167 Y98.946 E.01378
G3 X141.697 Y99.08 I-2.242 J-18.441 E.07838
G1 X142.16 Y99.465 E.01906
G1 X142.232 Y99.712 E.00815
G1 X142.175 Y100.04 E.01052
G1 X142.103 Y100.169 E.0047
G1 X141.85 Y100.201 E.00807
G1 X141.462 Y99.998 E.01387
G1 X141.984 Y102.815 E.09071
G1 X141.944 Y102.989 E.00567
G1 X141.762 Y103.082 E.00646
G1 X141.574 Y103.082 E.00596
G1 X141.705 Y103.299 E.00804
G1 X141.732 Y103.462 E.00524
G1 X141.674 Y103.572 E.00393
G1 X141.512 Y103.641 E.00559
G1 X138.872 Y103.641 E.08357
G1 X138.711 Y103.574 E.00552
G1 X138.646 Y103.416 E.00542
G1 X138.646 Y103.082 E.01057
G1 X133.626 Y103.082 E.15895
G1 X133.287 Y103.532 E.01784
G1 X133.148 Y103.634 E.00546
G3 X132.109 Y103.632 I-.482 J-17.822 E.0329
G1 X131.975 Y103.538 E.00519
G1 X131.938 Y103.391 E.0048
G1 X132.244 Y100.213 E.1011
; LINE_WIDTH: 0.55423
G1 X132.272 Y100.14 E.00264
; LINE_WIDTH: 0.588463
G1 X132.3 Y100.068 E.00281
; LINE_WIDTH: 0.622696
G1 X132.329 Y99.995 E.00299
; LINE_WIDTH: 0.660736
G1 X132.364 Y99.964 E.00192
; LINE_WIDTH: 0.698776
G1 X132.399 Y99.933 E.00203
; LINE_WIDTH: 0.736816
G1 X132.434 Y99.902 E.00215
; LINE_WIDTH: 0.741026
G1 X133.017 Y99.994 E.02718
G1 X133.166 Y99.192 E.03757
G1 X132.622 Y99.181 E.02505
; LINE_WIDTH: 0.733736
G1 X132.454 Y99.15 E.00779
; LINE_WIDTH: 0.696723
G1 X132.286 Y99.12 E.00737
; LINE_WIDTH: 0.65971
G1 X132.118 Y99.089 E.00696
; LINE_WIDTH: 0.622696
G1 X131.923 Y99.041 E.00768
; LINE_WIDTH: 0.588463
G1 X131.729 Y98.992 E.00724
; LINE_WIDTH: 0.55423
G1 X131.535 Y98.944 E.00679
; LINE_WIDTH: 0.519996
G1 X131.038 Y98.936 E.01573
G3 X122.131 Y116.662 I-50.886 J-14.467 E.63182
; LINE_WIDTH: 0.521596
G1 X121.692 Y117.189 E.02179
; LINE_WIDTH: 0.544336
G1 X121.388 Y117.589 E.01669
G1 X121.583 Y117.581 E.00652
; LINE_WIDTH: 0.531156
G1 X121.726 Y117.575 E.00461
; LINE_WIDTH: 0.521596
G1 X121.746 Y117.592 E.00084
; LINE_WIDTH: 0.521526
G1 X121.82 Y117.655 E.00307
; LINE_WIDTH: 0.521296
G1 X121.894 Y117.717 E.00307
; LINE_WIDTH: 0.521066
G1 X121.968 Y117.779 E.00307
; LINE_WIDTH: 0.520836
G1 X122.042 Y117.841 E.00306
; LINE_WIDTH: 0.520616
G1 X122.116 Y117.904 E.00306
; LINE_WIDTH: 0.520386
G1 X122.189 Y117.966 E.00306
; LINE_WIDTH: 0.520156
G1 X122.242 Y118.01 E.00216
; LINE_WIDTH: 0.519996
G1 X122.358 Y118.107 E.00481
G1 F3150
G1 X122.523 Y118.246 E.00682
G1 F3300
G1 X122.688 Y118.384 E.00682
G1 F3450
G1 X122.853 Y118.523 E.00682
G1 F3600
G1 X123.018 Y118.661 E.00682
G1 X123.184 Y118.8 E.00682
G1 X123.349 Y118.938 E.00682
G3 X123.421 Y119.035 I-.083 J.138 E.00393
G1 X123.466 Y119.119 E.00303
G1 X123.441 Y119.202 E.00273
; WIPE_START
M204 S10000
G1 X122.81 Y119.978 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X128.672 Y115.09 Z7.8 F36000
G1 X143.068 Y103.085 Z7.8
G1 Z7.4
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.545016
G1 F15420.618
G1 X143.066 Y102.614 E.01567
; LINE_WIDTH: 0.591476
G1 F14134.69
G1 X143.043 Y102.361 E.00923
; LINE_WIDTH: 0.637936
G1 F13046.721
G1 X143.02 Y102.108 E.01
; LINE_WIDTH: 0.684396
G1 F12114.267
G1 X142.997 Y101.855 E.01077
; LINE_WIDTH: 0.730856
G1 F11306.209
G1 X142.973 Y101.602 E.01154
; LINE_WIDTH: 0.777316
G1 F10599.209
G1 X142.95 Y101.349 E.01231
; LINE_WIDTH: 0.823776
G1 F9975.426
G1 X142.927 Y101.096 E.01308
; WIPE_START
G1 X142.95 Y101.349 E-.09654
G1 X142.973 Y101.602 E-.09654
G1 X142.997 Y101.855 E-.09654
G1 X143.018 Y102.092 E-.09037
; WIPE_END
G1 E-.02 F1800
G1 X135.497 Y100.794 Z7.8 F36000
G1 X131.49 Y100.103 Z7.8
G1 Z7.4
G1 E.4 F1800
; LINE_WIDTH: 1.05458
G1 F7718.772
G1 X131.607 Y99.736 E.02564
; WIPE_START
G1 X131.49 Y100.103 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X127.671 Y106.712 Z7.8 F36000
G1 X121.388 Y117.589 Z7.8
G1 Z7.4
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X119.929 Y119.331 E.07555
; WIPE_START
M204 S10000
G1 X120.571 Y118.564 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.286 Y122.751 Z7.8 F36000
G1 Z7.4
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X120.017 Y122.536 E.01312
G3 X118.612 Y123.958 I-27.312 J-25.586 E.07632
G2 X119.461 Y127.052 I4.054 J.551 E.12586
G2 X121.219 Y128.937 I43.731 J-39.004 E.0984
G3 X121.462 Y134.592 I-3.366 J2.978 E.23379
G1 X121.257 Y134.816 E.01159
G3 X121.116 Y135.987 I-6.085 J-.141 E.04508
G3 X126.155 Y138.622 I-30.363 J64.207 E.21716
M73 P74 R4
G3 X127.002 Y136.477 I4.558 J.561 E.08897
G3 X128.759 Y134.592 I43.753 J39.024 E.0984
G2 X129.002 Y128.937 I-3.366 J-2.978 E.23379
G2 X127.245 Y127.052 I-43.731 J39.004 E.0984
G3 X127.002 Y121.396 I3.366 J-2.978 E.23379
G3 X128.759 Y119.511 I43.753 J39.024 E.0984
G2 X129.002 Y113.855 I-3.366 J-2.978 E.23379
G2 X127.661 Y112.407 I-33.591 J29.76 E.07537
G3 X126.438 Y114.404 I-32.959 J-18.816 E.08944
G1 X141.945 Y143.915 F36000
G1 F13446.283
G1 X141.945 Y141.573 E.08944
G3 X141.945 Y136.709 I3.506 J-2.432 E.19757
G1 X141.945 Y126.491 E.39011
G3 X141.945 Y121.628 I3.506 J-2.432 E.19757
G1 X141.945 Y111.41 E.39011
G3 X141.945 Y106.547 I3.506 J-2.432 E.19757
G1 X141.945 Y105.79 E.02888
G1 X141.529 Y105.863 E.01612
G1 X138.865 Y105.863 E.10172
G3 X137.347 Y105.304 I.009 J-2.363 E.06306
G1 X135.746 Y105.304 E.06111
G3 X134.786 Y106.315 I-23.517 J-21.378 E.05325
G2 X134.543 Y111.97 I3.366 J2.978 E.23379
G2 X136.3 Y113.855 I43.742 J-39.013 E.0984
G3 X136.543 Y119.511 I-3.366 J2.978 E.23379
G3 X134.786 Y121.396 I-43.731 J-39.004 E.0984
G2 X134.543 Y127.052 I3.366 J2.978 E.23379
G2 X136.3 Y128.937 I43.742 J-39.013 E.0984
G3 X136.543 Y134.592 I-3.366 J2.978 E.23379
G3 X134.786 Y136.477 I-43.731 J-39.004 E.0984
G2 X134.543 Y142.133 I3.366 J2.978 E.23379
G2 X136.3 Y144.018 I43.742 J-39.013 E.0984
G3 X137.399 Y147.373 I-3.316 J2.944 E.13863
G3 X141.683 Y152.232 I-39.466 J39.116 E.24746
G1 X141.945 Y151.791 E.0196
G1 X141.945 Y149.448 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 7.56
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y150.448 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L47
M991 S0 P46 ;notify layer change


G17
G3 Z7.8 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X124.778 Y120.286
G1 Z7.56
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X124.633 Y120.518 E.01045
G1 X123.407 Y121.981 E.07287
G1 X123.008 Y122.341 E.02054
G1 X122.532 Y122.577 E.02028
G1 X122.043 Y122.674 E.01904
G1 X121.559 Y122.646 E.0185
G1 X120.943 Y122.42 E.02504
G3 X120.087 Y121.742 I8.869 J-12.082 E.04172
G3 X114.848 Y126.662 I-40.956 J-38.353 E.2746
G1 X118.015 Y129.012 E.15057
G3 X118.932 Y129.849 I-4.611 J5.972 E.04747
G1 X119.566 Y130.648 E.03892
G1 X120.076 Y131.531 E.03891
G1 X120.45 Y132.477 E.03887
G3 X120.751 Y134.26 I-7.068 J2.11 E.06921
G3 X120.538 Y136.268 I-7.041 J.27 E.07734
G3 X129.808 Y141.592 I-22.023 J49.078 E.4088
G3 X142.443 Y154.07 I-32.034 J45.073 E.68094
G1 X142.443 Y105.366 E1.85948
G1 X138.451 Y105.368 E.15243
G1 X137.904 Y105.29 E.02107
G1 X137.559 Y105.152 E.01421
G1 X137.131 Y104.853 E.01991
G1 X137.111 Y104.808 E.00189
G1 X135.153 Y104.808 E.07475
G1 X135.013 Y104.965 E.00804
G1 X134.507 Y105.245 E.02208
G1 X133.998 Y105.36 E.01993
G3 X132.157 Y105.363 I-.968 J-31.802 E.07028
G1 X131.708 Y105.309 E.01728
G1 X131.251 Y105.137 E.01863
G1 X130.858 Y104.786 E.02011
G3 X123.817 Y117.29 I-50.798 J-20.369 E.54947
G1 X124.391 Y117.771 E.0286
G1 X124.675 Y118.064 E.01557
G1 X124.955 Y118.557 E.02163
G1 X125.079 Y119.072 E.02025
G1 X125.059 Y119.602 E.02024
G1 X124.919 Y120.061 E.01833
G1 X124.826 Y120.21 E.00668
G1 X124.28 Y119.988 F36000
G1 F13446.369
G1 X124.184 Y120.142 E.0069
G1 X122.958 Y121.604 E.07287
G1 X122.639 Y121.883 E.01618
G1 X122.244 Y122.051 E.01639
G1 X121.9 Y122.092 E.01325
G1 X121.536 Y122.039 E.01402
G1 X121.213 Y121.899 E.01346
G3 X120.035 Y120.934 I19.37 J-24.846 E.05812
G3 X113.893 Y126.683 I-40.654 J-37.281 E.32151
G1 X117.666 Y129.482 E.17938
G3 X118.523 Y130.268 I-4.175 J5.413 E.04445
G1 X119.1 Y131.002 E.03564
G1 X119.562 Y131.813 E.03563
G1 X119.901 Y132.681 E.03559
G3 X120.136 Y133.793 I-10.15 J2.725 E.04339
G1 X120.18 Y134.525 E.02801
G1 X120.114 Y135.456 E.03564
G1 X119.931 Y136.289 E.03257
G1 X119.836 Y136.599 E.01237
G3 X130.16 Y142.573 I-21.55 J49.153 E.45633
G3 X142.92 Y155.769 I-31.893 J43.608 E.70426
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.64 E1.95129
G1 X142.437 Y104.781 E.02325
G1 X138.451 Y104.782 E.15218
G1 X138.026 Y104.714 E.0164
G1 X137.827 Y104.631 E.00826
G1 X137.527 Y104.422 E.01393
G1 X137.44 Y104.222 E.00833
G1 X134.905 Y104.222 E.09678
G1 X134.657 Y104.5 E.01422
G1 X134.251 Y104.714 E.01752
G1 X133.823 Y104.782 E.01654
G1 X132.159 Y104.777 E.06353
G3 X131.525 Y104.619 I.004 J-1.364 E.02519
G1 X131.036 Y104.182 E.02504
G1 X130.817 Y103.636 E.02244
G1 X130.805 Y103.298 E.01291
G3 X123.014 Y117.381 I-51.086 J-19.067 E.6167
G1 X124.015 Y118.22 E.04988
G1 X124.335 Y118.61 E.01928
G1 X124.479 Y119.012 E.0163
G1 X124.492 Y119.437 E.01622
G1 X124.384 Y119.822 E.01528
G1 X124.328 Y119.912 E.00405
G1 X123.789 Y119.665 F36000
G1 F13446.369
G1 X123.736 Y119.765 E.00434
G1 X122.509 Y121.228 E.07287
G1 X122.273 Y121.418 E.01157
G1 X121.965 Y121.505 E.01224
G1 X121.698 Y121.476 E.01025
G1 X121.413 Y121.325 E.01233
G1 X119.963 Y120.11 E.07222
G1 X119.604 Y120.538 E.02132
G3 X112.93 Y126.698 I-39.316 J-35.903 E.3472
G1 X117.325 Y129.959 E.20896
G1 X118.114 Y130.687 E.041
G1 X118.634 Y131.357 E.03235
G1 X119.049 Y132.095 E.03235
G1 X119.352 Y132.885 E.03231
G3 X119.551 Y133.828 I-12.667 J3.17 E.03681
G1 X119.596 Y134.561 E.02801
G1 X119.531 Y135.405 E.03235
G1 X119.361 Y136.154 E.02929
G1 X119.08 Y136.91 E.03081
G3 X129.815 Y143.046 I-20.502 J48.328 E.47315
G3 X142.648 Y156.415 I-31.73 J43.3 E.71109
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y105.063 E1.95373
G1 X143.615 Y104.663 E.01527
; LINE_WIDTH: 0.659729
G1 F12355.125
G1 X143.595 Y104.485 E.00729
; LINE_WIDTH: 0.699461
G1 F11746.824
G1 X143.575 Y104.307 E.00776
; LINE_WIDTH: 0.739194
G1 F11153.882
G1 X143.555 Y104.129 E.00822
; LINE_WIDTH: 0.778926
G1 F10576.291
G1 X143.535 Y103.952 E.00868
; LINE_WIDTH: 0.821508
G1 F10004.173
G1 X143.514 Y103.891 E.0033
; LINE_WIDTH: 0.86409
G1 F9490.773
G1 X143.493 Y103.83 E.00348
; LINE_WIDTH: 0.906671
G1 F9027.496
G1 X143.471 Y103.77 E.00366
; LINE_WIDTH: 0.949253
G1 F8607.342
G1 X143.45 Y103.709 E.00384
; LINE_WIDTH: 0.991835
G1 F8224.558
G1 X143.429 Y103.648 E.00401
; LINE_WIDTH: 1.03442
G1 F7874.37
G1 X143.407 Y103.588 E.00419
G1 X143.369 Y103.632 E.00383
; LINE_WIDTH: 0.991835
G1 F8224.558
G1 X143.33 Y103.676 E.00367
; LINE_WIDTH: 0.949253
G1 F8607.342
G1 X143.291 Y103.721 E.00351
; LINE_WIDTH: 0.906671
G1 F9027.496
G1 X143.253 Y103.765 E.00334
; LINE_WIDTH: 0.86409
G1 F9490.773
G1 X143.214 Y103.809 E.00318
; LINE_WIDTH: 0.821508
G1 F10004.173
G1 X143.175 Y103.853 E.00302
; LINE_WIDTH: 0.778926
G1 F10576.291
G1 X143.073 Y103.921 E.00594
; LINE_WIDTH: 0.739194
G1 F11172.472
G1 X142.971 Y103.988 E.00563
; LINE_WIDTH: 0.699461
G1 F11577.157
G1 X142.868 Y104.055 E.00531
; LINE_WIDTH: 0.659729
G1 F11989.023
G1 X142.766 Y104.123 E.00499
; LINE_WIDTH: 0.619996
G1 F13161.459
G1 X142.437 Y104.196 E.01289
G1 F13446.369
G1 X142.037 Y104.196 E.01527
G1 X138.45 Y104.196 E.13691
G1 X138.142 Y104.132 E.01204
G1 X137.924 Y103.991 E.00993
G1 X137.768 Y103.637 E.01477
G1 X134.603 Y103.637 E.12083
G1 X134.3 Y104.035 E.01912
G1 X134.045 Y104.165 E.01094
G1 X133.724 Y104.196 E.01229
G1 X132.16 Y104.191 E.0597
G1 X131.799 Y104.101 E.01423
G1 X131.513 Y103.842 E.01473
G1 X131.393 Y103.529 E.01281
G3 X131.448 Y102.712 I38.095 J2.165 E.03126
G1 X131.501 Y102.165 E.02096
G1 X131.539 Y101.767 E.01527
G1 F12788.082
G1 X131.577 Y101.369 E.01527
; LINE_WIDTH: 0.669038
G1 F11424.965
G1 X131.567 Y101.216 E.00636
; LINE_WIDTH: 0.718079
G1 F10921.201
G1 X131.557 Y101.062 E.00686
; LINE_WIDTH: 0.76712
G1 F10428.796
G1 X131.548 Y100.908 E.00735
; LINE_WIDTH: 0.816161
G1 F9947.75
G1 X131.538 Y100.755 E.00784
; LINE_WIDTH: 0.865202
G1 F9478.06
G1 X131.528 Y100.601 E.00833
; LINE_WIDTH: 0.914244
G1 F8949.805
G1 X131.518 Y100.448 E.00882
; LINE_WIDTH: 0.963285
G1 F8477.326
G1 X131.508 Y100.294 E.00932
; LINE_WIDTH: 1.01233
G1 F8052.23
G1 X131.498 Y100.141 E.00981
; LINE_WIDTH: 1.03836
G1 F7843.469
G1 X131.49 Y100.103 E.00256
G1 X131.466 Y100.134 E.00258
; LINE_WIDTH: 1.01233
G1 F8052.23
G1 X131.397 Y100.272 E.00981
; LINE_WIDTH: 0.963285
G1 F8477.326
G1 X131.327 Y100.409 E.00932
; LINE_WIDTH: 0.914244
G1 F8949.805
G1 X131.258 Y100.546 E.00882
; LINE_WIDTH: 0.865202
G1 F9478.06
G1 X131.189 Y100.684 E.00833
; LINE_WIDTH: 0.816161
G1 F10072.587
G1 X131.12 Y100.821 E.00784
; LINE_WIDTH: 0.76712
G1 F10556.6
G1 X131.05 Y100.958 E.00735
; LINE_WIDTH: 0.718079
G1 F11051.971
G1 X130.981 Y101.096 E.00686
; LINE_WIDTH: 0.669038
G1 F11558.7
G1 X130.912 Y101.233 E.00636
; LINE_WIDTH: 0.619996
G1 F12929.548
G1 X130.779 Y101.61 E.01527
G1 F13446.369
G1 X130.646 Y101.988 E.01527
G1 X130.263 Y103.082 E.04426
G3 X124.03 Y114.993 I-50.05 J-18.602 E.51463
G3 X122.189 Y117.454 I-29.018 J-19.792 E.11735
G1 X123.639 Y118.669 E.07222
G1 X123.822 Y118.891 E.011
G1 X123.914 Y119.188 E.01186
G1 X123.888 Y119.477 E.01105
G1 X123.831 Y119.585 E.0047
; WIPE_START
G1 X123.736 Y119.765 E-.07742
G1 X123.224 Y120.376 E-.30258
; WIPE_END
G1 E-.02 F1800
G1 X129.042 Y115.436 Z7.96 F36000
G1 X143.407 Y103.241 Z7.96
G1 Z7.56
G1 E.4 F1800
; LINE_WIDTH: 1.03002
G1 F7909.168
G1 X143.43 Y103.177 E.00441
; LINE_WIDTH: 0.985723
G1 F8277.392
G1 X143.453 Y103.113 E.00421
; LINE_WIDTH: 0.94143
G1 F8681.577
G1 X143.476 Y103.049 E.00402
; LINE_WIDTH: 0.897136
G1 F9127.261
G1 X143.499 Y102.918 E.00746
; LINE_WIDTH: 0.850946
G1 F9643.528
G1 X143.522 Y102.788 E.00706
; LINE_WIDTH: 0.804756
G1 F10221.698
G1 X143.545 Y102.657 E.00666
; LINE_WIDTH: 0.758566
G1 F10641.569
G1 X143.568 Y102.526 E.00626
; LINE_WIDTH: 0.712376
G1 F11069.917
G1 X143.591 Y102.396 E.00587
; LINE_WIDTH: 0.666186
G1 F11506.723
G1 X143.615 Y102.265 E.00547
; LINE_WIDTH: 0.619996
G1 F12874.571
G1 X143.615 Y101.865 E.01527
G1 F13423.248
G1 X143.615 Y101.465 E.01527
G1 X143.615 Y101.156 E.01179
; LINE_WIDTH: 0.661349
G1 F12514.692
G1 X143.594 Y100.943 E.00876
; LINE_WIDTH: 0.702701
G1 F11782.485
G1 X143.573 Y100.729 E.00934
; LINE_WIDTH: 0.744054
G1 F11095.966
G1 X143.553 Y100.516 E.00992
; LINE_WIDTH: 0.785406
G1 F10485.042
G1 X143.532 Y100.302 E.0105
G1 X143.543 Y99.618 E.0335
; LINE_WIDTH: 0.762636
G1 F10812.853
G3 X142.864 Y99.704 I-1.096 J-5.936 E.03253
; LINE_WIDTH: 0.785056
G1 F10489.93
G1 X142.787 Y100.235 E.02622
G1 X142.707 Y100.34 E.00648
; LINE_WIDTH: 0.743791
G1 F10914.105
G1 X142.628 Y100.446 E.00612
; LINE_WIDTH: 0.702526
G1 F11346.713
G1 X142.548 Y100.552 E.00577
; LINE_WIDTH: 0.661261
G1 F11787.702
G1 X142.469 Y100.658 E.00541
; LINE_WIDTH: 0.619996
G1 F12871.242
G1 X142.159 Y100.715 E.01203
G1 F13446.369
G1 X142.232 Y101.108 E.01527
G1 X142.375 Y101.879 E.02993
G1 X142.448 Y102.272 E.01527
G1 F12806.562
G1 X142.885 Y102.618 E.02129
; LINE_WIDTH: 0.666186
G1 F10925.807
G1 X142.955 Y102.693 E.0042
; LINE_WIDTH: 0.712376
G1 F10597.735
G1 X143.025 Y102.767 E.00451
; LINE_WIDTH: 0.758566
G1 F10274.665
G1 X143.095 Y102.841 E.00482
; LINE_WIDTH: 0.804756
G1 F9956.597
G1 X143.165 Y102.916 E.00513
; LINE_WIDTH: 0.850946
G1 F9643.528
G1 X143.234 Y102.99 E.00543
; LINE_WIDTH: 0.897136
G1 F9127.261
G1 X143.304 Y103.065 E.00574
; LINE_WIDTH: 0.94143
G1 F8681.577
G1 X143.338 Y103.123 E.00402
; LINE_WIDTH: 0.985723
G1 F8277.392
G1 X143.362 Y103.163 E.00284
; WIPE_START
G1 X143.43 Y103.177 E-.0265
G1 X143.453 Y103.113 E-.02581
G1 X143.476 Y103.049 E-.02581
G1 X143.499 Y102.918 E-.05042
G1 X143.522 Y102.788 E-.05042
G1 X143.545 Y102.657 E-.05042
G1 X143.568 Y102.526 E-.05042
G1 X143.591 Y102.396 E-.05042
G1 X143.614 Y102.267 E-.04978
; WIPE_END
G1 E-.02 F1800
G1 X137.783 Y107.191 Z7.96 F36000
G1 X123.312 Y119.41 Z7.96
G1 Z7.56
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X122.086 Y120.873 E.06043
G1 X121.928 Y120.953 E.00561
G1 X121.837 Y120.924 E.00301
G3 X121.734 Y120.873 I.012 J-.154 E.00375
G1 X121.587 Y120.749 E.00609
G1 X121.439 Y120.626 E.00609
G1 X121.292 Y120.502 E.00609
G1 X121.145 Y120.379 E.00609
G1 X120.997 Y120.255 E.00609
G1 X120.85 Y120.132 E.00609
G1 X120.746 Y120.045 E.00429
; LINE_WIDTH: 0.520326
G1 X120.682 Y119.991 E.00265
; LINE_WIDTH: 0.520776
G1 X120.591 Y119.915 E.00377
; LINE_WIDTH: 0.521226
G1 X120.499 Y119.84 E.00377
; LINE_WIDTH: 0.521686
G1 X120.408 Y119.764 E.00377
; LINE_WIDTH: 0.522136
G1 X120.317 Y119.688 E.00378
; LINE_WIDTH: 0.522596
G1 X120.225 Y119.612 E.00378
; LINE_WIDTH: 0.523046
G1 X120.134 Y119.536 E.00378
; LINE_WIDTH: 0.523196
G1 X120.104 Y119.511 E.00126
; LINE_WIDTH: 0.531656
G1 X120.075 Y119.39 E.00402
; LINE_WIDTH: 0.544336
G1 X120.032 Y119.209 E.00619
G1 X119.181 Y120.183 E.04301
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.856 J-35.522 E.30731
G1 X116.99 Y130.398 E.19633
G1 X117.728 Y131.083 E.03186
G1 X118.193 Y131.691 E.02425
G1 X118.565 Y132.361 E.02425
G1 X118.833 Y133.078 E.02423
G3 X118.999 Y133.862 I-29.583 J6.662 E.02538
G1 X119.044 Y134.594 E.02322
G1 X118.98 Y135.358 E.02426
G1 X118.824 Y136.026 E.02173
G1 X118.548 Y136.753 E.02463
G1 X118.31 Y137.19 E.01573
G3 X129.492 Y143.495 I-19.701 J48.01 E.40747
G3 X142.39 Y157.025 I-31.404 J42.849 E.59492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.379 E1.81461
G1 X144.167 Y98.946 E.01373
G2 X142.892 Y99.077 I.1 J7.236 E.04064
G3 X141.7 Y99.082 I-1.361 J-179.484 E.03776
G1 X142.16 Y99.465 E.01895
G1 X142.238 Y99.677 E.00714
G1 X142.16 Y100.128 E.01449
G1 X142.092 Y100.254 E.00454
G1 X141.852 Y100.298 E.00771
G1 X141.492 Y100.15 E.01232
G1 X141.942 Y102.577 E.07815
G1 X142.542 Y103.052 E.02423
G1 X142.624 Y103.188 E.00504
G1 X142.659 Y103.456 E.00855
G1 X142.532 Y103.622 E.0066
G1 X142.436 Y103.643 E.0031
G1 X138.45 Y103.643 E.1262
G1 X138.298 Y103.584 E.00519
G1 X138.225 Y103.418 E.00575
G1 X138.225 Y103.084 E.01057
G1 X134.228 Y103.084 E.12655
G1 X134.149 Y103.318 E.00783
G1 X133.964 Y103.597 E.01058
G1 X133.826 Y103.643 E.00461
G1 X132.162 Y103.639 E.05268
G1 X132.057 Y103.612 E.00342
G1 X131.94 Y103.446 E.00644
G3 X132.052 Y102.203 I63.12 J5.101 E.03952
G1 X132.244 Y100.213 E.06332
; LINE_WIDTH: 0.553886
G1 X132.272 Y100.14 E.00263
; LINE_WIDTH: 0.587776
G1 X132.3 Y100.068 E.0028
; LINE_WIDTH: 0.621666
G1 X132.329 Y99.996 E.00297
; LINE_WIDTH: 0.659726
G1 X132.364 Y99.965 E.00192
; LINE_WIDTH: 0.697786
G1 X132.399 Y99.933 E.00204
; LINE_WIDTH: 0.735846
G1 X132.435 Y99.902 E.00215
; LINE_WIDTH: 0.739896
G1 X133.015 Y99.994 E.02704
G1 X133.164 Y99.194 E.03745
G1 X132.622 Y99.182 E.02492
; LINE_WIDTH: 0.732736
G1 X132.454 Y99.151 E.00778
; LINE_WIDTH: 0.695713
G1 X132.286 Y99.121 E.00737
; LINE_WIDTH: 0.65869
G1 X132.117 Y99.09 E.00696
; LINE_WIDTH: 0.621666
G1 X131.923 Y99.041 E.00768
; LINE_WIDTH: 0.587776
M73 P75 R4
G1 X131.728 Y98.993 E.00723
; LINE_WIDTH: 0.553886
G1 X131.534 Y98.944 E.00679
; LINE_WIDTH: 0.519996
G1 X131.038 Y98.936 E.01571
G3 X122.131 Y116.662 I-50.886 J-14.467 E.63182
; LINE_WIDTH: 0.521596
G1 X121.589 Y117.312 E.02687
; LINE_WIDTH: 0.544336
G1 X121.285 Y117.711 E.01669
G1 X121.481 Y117.704 E.00652
; LINE_WIDTH: 0.531156
G1 X121.623 Y117.698 E.00461
; LINE_WIDTH: 0.521596
G1 X121.648 Y117.719 E.00104
; LINE_WIDTH: 0.521526
G1 X121.739 Y117.796 E.0038
; LINE_WIDTH: 0.521296
G1 X121.831 Y117.873 E.0038
; LINE_WIDTH: 0.521066
G1 X121.922 Y117.95 E.00379
; LINE_WIDTH: 0.520836
G1 X122.014 Y118.027 E.00379
; LINE_WIDTH: 0.520616
G1 X122.105 Y118.104 E.00379
; LINE_WIDTH: 0.520386
G1 X122.197 Y118.181 E.00379
; LINE_WIDTH: 0.520156
G1 X122.261 Y118.235 E.00267
; LINE_WIDTH: 0.519996
G1 X122.365 Y118.322 E.00429
G1 X122.513 Y118.446 E.00609
G1 X122.66 Y118.57 E.00609
G1 X122.808 Y118.693 E.00609
G1 X122.955 Y118.817 E.00609
G1 X123.103 Y118.94 E.00609
G1 X123.25 Y119.064 E.00609
G3 X123.319 Y119.158 I-.081 J.131 E.00378
G1 X123.364 Y119.243 E.00305
G1 X123.338 Y119.324 E.00269
; WIPE_START
M204 S10000
G1 X122.71 Y120.102 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X128.274 Y114.878 Z7.96 F36000
G1 X142.982 Y101.072 Z7.96
G1 Z7.56
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.810716
G1 F10143.229
G1 X142.885 Y101.072 E.00491
G1 X142.836 Y101.156 E.00491
G1 X142.885 Y101.24 E.00491
G1 X142.982 Y101.24 E.00491
G1 X143.031 Y101.156 E.00491
; WIPE_START
G1 X142.982 Y101.24 E-.076
G1 X142.885 Y101.24 E-.076
G1 X142.836 Y101.156 E-.076
G1 X142.885 Y101.072 E-.076
G1 X142.982 Y101.072 E-.076
; WIPE_END
G1 E-.02 F1800
G1 X143.407 Y103.247 Z7.96 F36000
G1 Z7.56
G1 E.4 F1800
; LINE_WIDTH: 1.03442
G1 F7874.37
G1 X143.407 Y103.588 E.02221
; WIPE_START
G1 X143.407 Y103.247 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X136.027 Y101.3 Z7.96 F36000
G1 X131.49 Y100.103 Z7.96
G1 Z7.56
G1 E.4 F1800
; LINE_WIDTH: 1.05446
G1 F7719.68
G1 X131.606 Y99.736 E.0256
; WIPE_START
G1 X131.49 Y100.103 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X127.663 Y106.706 Z7.96 F36000
G1 X121.285 Y117.711 Z7.96
G1 Z7.56
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X120.032 Y119.209 E.06491
; WIPE_START
M204 S10000
G1 X120.674 Y118.442 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.257 Y122.534 Z7.96 F36000
G1 Z7.56
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X120.125 Y122.424 E.00654
G3 X118.6 Y123.969 I-28.865 J-26.95 E.08291
G2 X119.601 Y127.052 I4.265 J.319 E.12689
G3 X121.356 Y128.937 I-310.797 J291.031 E.09833
G3 X121.322 Y134.592 I-3.458 J2.807 E.23394
G1 X121.26 Y134.659 E.0035
G3 X121.111 Y135.984 I-8.604 J-.297 E.05094
G3 X126.188 Y138.643 I-28.703 J60.982 E.21887
G3 X127.142 Y136.477 I4.73 J.791 E.09128
G2 X128.896 Y134.592 I-311.382 J-291.574 E.09833
G2 X128.863 Y128.937 I-3.458 J-2.807 E.23394
G3 X127.108 Y127.052 I311.382 J-291.574 E.09833
G3 X127.142 Y121.396 I3.458 J-2.807 E.23394
G2 X128.896 Y119.511 I-311.382 J-291.574 E.09833
G2 X128.863 Y113.855 I-3.458 J-2.807 E.23394
G3 X127.606 Y112.507 I223.029 J-209.092 E.07038
G3 X126.376 Y114.5 I-36.243 J-20.976 E.08944
G1 X141.945 Y144.089 F36000
G1 F13446.283
G1 X141.945 Y141.746 E.08944
G3 X141.945 Y136.909 I3.478 J-2.418 E.19654
G1 X141.945 Y126.665 E.39113
G3 X141.945 Y121.828 I3.478 J-2.418 E.19654
G1 X141.945 Y111.583 E.39113
G3 X141.945 Y106.747 I3.478 J-2.418 E.19654
G1 X141.945 Y105.865 E.03366
G1 X138.443 Y105.865 E.13373
G3 X136.914 Y105.306 I.012 J-2.4 E.06341
G1 X135.585 Y105.306 E.05072
G2 X134.649 Y106.315 I168.814 J157.649 E.05256
G2 X134.682 Y111.97 I3.458 J2.807 E.23394
G3 X136.437 Y113.855 I-311.382 J291.574 E.09833
G3 X136.403 Y119.511 I-3.458 J2.807 E.23394
G2 X134.649 Y121.396 I311.973 J292.126 E.09833
G2 X134.682 Y127.052 I3.458 J2.807 E.23394
G3 X136.437 Y128.937 I-311.382 J291.574 E.09833
G3 X136.403 Y134.592 I-3.458 J2.807 E.23394
G2 X134.649 Y136.477 I311.973 J292.126 E.09833
G2 X134.682 Y142.133 I3.458 J2.807 E.23394
G3 X136.437 Y144.018 I-311.382 J291.574 E.09833
G3 X137.375 Y147.348 I-3.26 J2.715 E.13603
G3 X138.99 Y149.044 I-34.757 J34.726 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 7.72
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X138.3 Y148.32 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L48
M991 S0 P47 ;notify layer change


G17
G3 Z7.96 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X124.675 Y120.41
G1 Z7.72
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X124.53 Y120.641 E.01037
G1 X123.51 Y121.858 E.06065
G1 X123.146 Y122.194 E.01889
G1 X122.743 Y122.414 E.01754
G1 X122.267 Y122.539 E.01881
G1 X121.745 Y122.536 E.01993
G1 X121.133 Y122.344 E.02446
G3 X120.196 Y121.624 I4.977 J-7.451 E.04518
G3 X114.848 Y126.662 I-40.035 J-37.137 E.28072
G1 X118.03 Y129.023 E.15128
G3 X118.932 Y129.849 I-4.663 J5.999 E.04675
G1 X119.565 Y130.646 E.03884
G1 X120.075 Y131.53 E.03899
G1 X120.45 Y132.479 E.03896
G1 X120.681 Y133.47 E.03884
G1 X120.765 Y134.489 E.03902
G1 X120.698 Y135.506 E.03893
G1 X120.538 Y136.268 E.0297
G3 X130.509 Y142.102 I-22.128 J49.251 E.4419
G3 X142.443 Y154.069 I-32.643 J44.489 E.64781
G1 X142.443 Y105.365 E1.85949
G2 X138.077 Y105.37 I-.644 J1403.396 E.16671
G1 X137.566 Y105.303 E.01968
G1 X137.117 Y105.121 E.0185
G1 X136.933 Y105.005 E.0083
G1 X136.833 Y104.81 E.00835
G1 X135.822 Y104.81 E.0386
G1 X135.722 Y105.005 E.00835
G1 X135.294 Y105.237 E.0186
G1 X134.863 Y105.35 E.01703
G3 X132.157 Y105.363 I-1.656 J-64.336 E.1033
G1 X131.708 Y105.309 E.01728
G1 X131.251 Y105.137 E.01863
G1 X130.858 Y104.786 E.02011
G3 X123.721 Y117.418 I-50.791 J-20.365 E.55558
G1 X124.289 Y117.894 E.02828
G1 X124.629 Y118.264 E.01921
G1 X124.875 Y118.738 E.02038
G1 X124.977 Y119.204 E.01822
G1 X124.957 Y119.725 E.01989
G1 X124.815 Y120.187 E.01845
G1 X124.722 Y120.334 E.00664
G1 X124.231 Y120.023 F36000
G1 F13446.369
G1 X124.19 Y120.116 E.00388
G1 X124.082 Y120.264 E.00703
G1 X123.061 Y121.482 E.06065
G1 X122.73 Y121.768 E.0167
G1 X122.334 Y121.932 E.01638
G3 X121.376 Y121.81 I-.3 J-1.472 E.03752
G3 X120.139 Y120.812 I9.513 J-13.058 E.06073
G3 X113.893 Y126.683 I-40.6 J-36.932 E.3276
G1 X117.677 Y129.49 E.17987
G1 X118.523 Y130.268 E.0439
G1 X119.099 Y131 E.03555
G1 X119.562 Y131.813 E.0357
G1 X119.901 Y132.683 E.03567
G1 X120.108 Y133.591 E.03556
G1 X120.18 Y134.525 E.03573
G1 X120.114 Y135.456 E.03564
G1 X119.931 Y136.29 E.03259
G1 X119.836 Y136.599 E.01236
G3 X130.933 Y143.157 I-21.111 J48.39 E.49336
G3 X142.92 Y155.769 I-32.176 J42.587 E.6673
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.639 E1.95133
G1 X142.434 Y104.781 E.02336
G2 X138.075 Y104.784 I-1.833 J597.182 E.16644
G1 X137.632 Y104.712 E.0171
G1 X137.274 Y104.529 E.01536
G1 X137.118 Y104.224 E.01306
G1 X135.537 Y104.224 E.06035
G1 X135.381 Y104.529 E.01306
G1 X134.977 Y104.727 E.0172
G3 X134.159 Y104.783 I-.598 J-2.744 E.03141
G1 X132.159 Y104.777 E.07636
G3 X131.525 Y104.619 I.004 J-1.364 E.02519
G1 X131.036 Y104.182 E.02504
G1 X130.807 Y103.291 E.03511
G3 X122.912 Y117.504 I-50.485 J-18.746 E.62311
G1 X123.912 Y118.343 E.04986
G1 X124.151 Y118.602 E.01344
G1 X124.332 Y118.961 E.01537
G1 X124.4 Y119.388 E.0165
G1 X124.348 Y119.761 E.0144
G1 X124.268 Y119.941 E.0075
G1 X123.679 Y119.814 F36000
G1 F13446.369
G1 X123.633 Y119.888 E.00332
G1 X122.612 Y121.106 E.06065
G1 X122.369 Y121.299 E.01187
G1 X122.061 Y121.383 E.01219
G1 X121.828 Y121.361 E.00892
G1 X121.516 Y121.202 E.01339
G1 X120.066 Y119.987 E.07222
G1 X119.604 Y120.538 E.02743
G3 X112.93 Y126.698 I-40.064 J-36.714 E.34718
G1 X117.323 Y129.957 E.20886
G1 X118.114 Y130.687 E.0411
G1 X118.632 Y131.355 E.03227
G1 X119.049 Y132.095 E.03242
G1 X119.352 Y132.887 E.03238
G1 X119.535 Y133.713 E.03228
G1 X119.595 Y134.56 E.03245
G1 X119.531 Y135.405 E.03235
G1 X119.361 Y136.154 E.02931
G1 X119.08 Y136.91 E.0308
G3 X130.579 Y143.624 I-20.56 J48.419 E.50974
G3 X142.647 Y156.415 I-31.78 J42.073 E.67456
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y105.065 E1.95363
G1 X143.615 Y104.665 E.01527
; LINE_WIDTH: 0.659506
G1 F12371.864
G1 X143.595 Y104.487 E.0073
; LINE_WIDTH: 0.699016
G1 F11762.152
G1 X143.575 Y104.309 E.00776
; LINE_WIDTH: 0.738526
G1 F11167.849
G1 X143.555 Y104.131 E.00822
; LINE_WIDTH: 0.778036
G1 F10588.948
G1 X143.536 Y103.953 E.00869
; LINE_WIDTH: 0.820766
G1 F10013.607
G1 X143.514 Y103.892 E.00331
; LINE_WIDTH: 0.863496
G1 F9497.565
G1 X143.493 Y103.831 E.00349
; LINE_WIDTH: 0.906226
G1 F9032.103
G1 X143.471 Y103.77 E.00367
; LINE_WIDTH: 0.948956
G1 F8610.134
G1 X143.45 Y103.709 E.00385
; LINE_WIDTH: 0.991686
G1 F8225.833
G1 X143.429 Y103.648 E.00403
; LINE_WIDTH: 1.03442
G1 F7874.37
G1 X143.407 Y103.588 E.00421
G1 X143.369 Y103.632 E.00385
; LINE_WIDTH: 0.991686
G1 F8225.833
G1 X143.33 Y103.676 E.00368
; LINE_WIDTH: 0.948956
G1 F8610.134
G1 X143.291 Y103.721 E.00352
; LINE_WIDTH: 0.906226
G1 F9032.103
G1 X143.252 Y103.765 E.00335
; LINE_WIDTH: 0.863496
G1 F9497.565
G1 X143.213 Y103.81 E.00319
; LINE_WIDTH: 0.820766
G1 F10013.607
G1 X143.174 Y103.854 E.00302
; LINE_WIDTH: 0.778036
G1 F10588.948
G1 X143.072 Y103.921 E.00594
; LINE_WIDTH: 0.738526
G1 F11183.063
G1 X142.97 Y103.989 E.00563
; LINE_WIDTH: 0.699016
G1 F11588.18
G1 X142.867 Y104.056 E.00531
; LINE_WIDTH: 0.659506
G1 F12000.477
G1 X142.765 Y104.123 E.00499
; LINE_WIDTH: 0.619996
G1 F13173.805
G1 X142.435 Y104.196 E.01289
G1 F13446.369
G1 X142.035 Y104.196 E.01527
G1 X138.072 Y104.198 E.1513
G1 X137.744 Y104.127 E.01281
G1 X137.616 Y104.053 E.00568
G1 X137.403 Y103.639 E.01776
G1 X135.253 Y103.639 E.0821
G1 X135.04 Y104.053 E.01776
G1 X134.755 Y104.18 E.01193
G3 X132.16 Y104.191 I-1.559 J-59.281 E.09905
G1 X131.799 Y104.101 E.01423
G1 X131.51 Y103.838 E.01491
G1 X131.388 Y103.338 E.01963
G1 X131.501 Y102.165 E.045
G1 X131.539 Y101.767 E.01527
G1 F12787.859
G1 X131.577 Y101.369 E.01527
; LINE_WIDTH: 0.669039
G1 F11424.755
G1 X131.567 Y101.215 E.00636
; LINE_WIDTH: 0.718081
G1 F10921.028
G1 X131.557 Y101.062 E.00686
; LINE_WIDTH: 0.767124
G1 F10428.658
G1 X131.548 Y100.908 E.00735
; LINE_WIDTH: 0.816166
G1 F9947.644
G1 X131.538 Y100.755 E.00784
; LINE_WIDTH: 0.865209
G1 F9477.989
G1 X131.528 Y100.601 E.00833
; LINE_WIDTH: 0.914251
G1 F8949.728
G1 X131.518 Y100.448 E.00882
; LINE_WIDTH: 0.963294
G1 F8477.246
G1 X131.508 Y100.294 E.00931
; LINE_WIDTH: 1.01234
G1 F8052.148
G1 X131.498 Y100.141 E.00981
; LINE_WIDTH: 1.03838
G1 F7843.313
G1 X131.49 Y100.103 E.00256
G1 X131.466 Y100.134 E.00258
; LINE_WIDTH: 1.01234
G1 F8052.148
G1 X131.397 Y100.272 E.00981
; LINE_WIDTH: 0.963294
G1 F8477.246
G1 X131.327 Y100.409 E.00931
; LINE_WIDTH: 0.914251
G1 F8949.728
G1 X131.258 Y100.546 E.00882
; LINE_WIDTH: 0.865209
G1 F9477.989
G1 X131.189 Y100.684 E.00833
; LINE_WIDTH: 0.816166
G1 F10072.523
G1 X131.12 Y100.821 E.00784
; LINE_WIDTH: 0.767124
G1 F10556.521
G1 X131.05 Y100.958 E.00735
; LINE_WIDTH: 0.718081
G1 F11051.861
G1 X130.981 Y101.096 E.00686
; LINE_WIDTH: 0.669039
G1 F11558.543
G1 X130.912 Y101.233 E.00636
; LINE_WIDTH: 0.619996
G1 F12929.381
G1 X130.78 Y101.611 E.01527
G1 F13446.369
G1 X130.649 Y101.989 E.01527
G1 X130.371 Y102.788 E.03231
G3 X124.03 Y114.993 I-50.321 J-18.39 E.52657
G3 X122.087 Y117.576 I-28.659 J-19.542 E.12346
G1 X123.536 Y118.791 E.07222
G1 X123.733 Y119.041 E.01215
G1 X123.815 Y119.373 E.01302
G1 X123.746 Y119.707 E.01303
G1 X123.727 Y119.738 E.0014
; WIPE_START
G1 X123.633 Y119.888 E-.06729
G1 X123.104 Y120.519 E-.31272
; WIPE_END
G1 E-.02 F1800
G1 X128.917 Y115.572 Z8.12 F36000
G1 X143.407 Y103.241 Z8.12
G1 Z7.72
G1 E.4 F1800
; LINE_WIDTH: 1.03003
G1 F7909.088
G1 X143.43 Y103.143 E.00652
; LINE_WIDTH: 0.984467
G1 F8288.331
G1 X143.453 Y103.045 E.00622
; LINE_WIDTH: 0.938908
G1 F8705.774
G1 X143.476 Y102.947 E.00592
; LINE_WIDTH: 0.89335
G1 F9167.495
G1 X143.499 Y102.85 E.00562
; LINE_WIDTH: 0.847791
G1 F9680.936
G1 X143.522 Y102.752 E.00533
; LINE_WIDTH: 0.802232
G1 F10255.302
G1 X143.545 Y102.654 E.00503
; LINE_WIDTH: 0.756673
G1 F10572.803
G1 X143.568 Y102.557 E.00473
; LINE_WIDTH: 0.711114
G1 F10895.121
G1 X143.591 Y102.459 E.00443
; LINE_WIDTH: 0.665555
G1 F11222.31
G1 X143.615 Y102.361 E.00413
; LINE_WIDTH: 0.619996
G1 F12573.626
G1 X143.615 Y101.961 E.01527
G1 F13185.489
G1 X143.615 Y101.561 E.01527
G1 X143.615 Y101.213 E.01329
; LINE_WIDTH: 0.663204
G1 F12397.083
G1 X143.593 Y101.007 E.00849
; LINE_WIDTH: 0.706411
G1 F11693.085
G1 X143.571 Y100.801 E.00907
; LINE_WIDTH: 0.749619
G1 F11009.637
G1 X143.55 Y100.595 E.00966
; LINE_WIDTH: 0.792826
G1 F10382.472
G1 X143.528 Y100.389 E.01024
G1 X143.543 Y99.619 E.03808
; LINE_WIDTH: 0.762786
G1 F10810.626
G3 X142.863 Y99.705 I-.979 J-5.011 E.03259
; LINE_WIDTH: 0.792466
G1 F10387.402
G1 X142.776 Y100.321 E.03072
G1 X142.699 Y100.423 E.00633
; LINE_WIDTH: 0.749349
G1 F10796.013
G1 X142.622 Y100.526 E.00597
; LINE_WIDTH: 0.706231
G1 F11212.533
G1 X142.545 Y100.628 E.00561
; LINE_WIDTH: 0.663114
G1 F11636.89
G1 X142.468 Y100.731 E.00525
; LINE_WIDTH: 0.619996
G1 F12658.486
G1 X142.178 Y100.805 E.01142
G1 F13446.369
G1 X142.251 Y101.198 E.01527
G1 X142.348 Y101.725 E.02045
G1 X142.637 Y101.861 E.01216
G1 F13282.268
G1 X142.807 Y102.091 E.01094
G1 F12279.061
G1 X143.045 Y102.413 E.01527
; LINE_WIDTH: 0.665555
G1 F10944.12
G1 X143.085 Y102.505 E.00413
; LINE_WIDTH: 0.711114
G1 F10621.068
G1 X143.125 Y102.597 E.00443
; LINE_WIDTH: 0.756673
G1 F10302.855
G1 X143.165 Y102.689 E.00473
; LINE_WIDTH: 0.802232
G1 F9989.469
G1 X143.206 Y102.781 E.00503
; LINE_WIDTH: 0.847791
G1 F9680.936
G1 X143.246 Y102.873 E.00532
; LINE_WIDTH: 0.89335
G1 F9167.495
G1 X143.286 Y102.965 E.00562
; LINE_WIDTH: 0.938908
G1 F8705.774
G1 X143.326 Y103.057 E.00592
; LINE_WIDTH: 0.984467
G1 F8288.331
G1 X143.367 Y103.149 E.00622
; LINE_WIDTH: 1.03003
G1 F7909.088
G1 X143.371 Y103.158 E.00068
; WIPE_START
G1 X143.43 Y103.143 E-.02321
G1 X143.453 Y103.045 E-.03816
G1 X143.476 Y102.947 E-.03816
G1 X143.499 Y102.85 E-.03816
G1 X143.522 Y102.752 E-.03816
G1 X143.545 Y102.654 E-.03816
G1 X143.568 Y102.557 E-.03816
G1 X143.591 Y102.459 E-.03816
G1 X143.615 Y102.361 E-.03816
G1 X143.615 Y102.226 E-.05153
; WIPE_END
G1 E-.02 F1800
G1 X137.794 Y107.163 Z8.12 F36000
G1 X123.209 Y119.533 Z8.12
G1 Z7.72
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X122.189 Y120.75 E.0503
G1 X122.029 Y120.831 E.00566
G1 X121.939 Y120.801 E.00298
G3 X121.841 Y120.753 I.011 J-.148 E.00355
G1 X121.711 Y120.645 E.00535
G1 X121.582 Y120.536 E.00535
G1 X121.452 Y120.427 E.00535
G1 X121.322 Y120.319 E.00535
G1 X121.193 Y120.21 E.00535
G1 X121.063 Y120.102 E.00535
G1 X120.972 Y120.025 E.00377
; LINE_WIDTH: 0.520326
G1 X120.895 Y119.961 E.00316
; LINE_WIDTH: 0.520776
G1 X120.786 Y119.871 E.00449
; LINE_WIDTH: 0.521226
G1 X120.678 Y119.78 E.00449
; LINE_WIDTH: 0.521686
G1 X120.569 Y119.69 E.00449
; LINE_WIDTH: 0.522136
G1 X120.46 Y119.599 E.0045
; LINE_WIDTH: 0.522596
G1 X120.351 Y119.509 E.0045
; LINE_WIDTH: 0.523046
G1 X120.243 Y119.418 E.00451
; LINE_WIDTH: 0.523196
G1 X120.206 Y119.388 E.00151
; LINE_WIDTH: 0.531646
G1 X120.178 Y119.267 E.00402
; LINE_WIDTH: 0.544336
G1 X120.135 Y119.086 E.00619
G1 X119.181 Y120.183 E.04833
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.855 J-35.522 E.30731
G1 X116.99 Y130.398 E.1963
G1 X117.728 Y131.083 E.03189
G1 X118.192 Y131.689 E.02419
G1 X118.565 Y132.361 E.02432
G1 X118.834 Y133.079 E.02428
G1 X118.994 Y133.827 E.02421
G1 X119.044 Y134.594 E.02434
G1 X118.98 Y135.358 E.02426
G1 X118.824 Y136.026 E.02174
G1 X118.547 Y136.753 E.02462
G1 X118.31 Y137.19 E.01574
G3 X130.249 Y144.067 I-19.583 J47.796 E.43753
G3 X142.39 Y157.025 I-31.645 J41.817 E.56491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.377 E1.81468
G1 X144.167 Y98.946 E.01367
G2 X142.891 Y99.079 I.1 J7.136 E.04068
G3 X141.702 Y99.084 I-1.346 J-173.398 E.03765
G1 X142.16 Y99.465 E.01885
G1 X142.238 Y99.677 E.00715
G1 X142.146 Y100.213 E.01724
G1 X142.081 Y100.336 E.00438
G1 X141.863 Y100.392 E.00715
G1 X141.522 Y100.296 E.01121
G1 X141.848 Y102.061 E.05681
G1 X142.255 Y102.302 E.01499
G1 X142.383 Y102.352 E.00432
G1 X142.501 Y102.512 E.0063
G1 X142.659 Y103.378 E.02789
G3 X142.436 Y103.643 I-.247 J.018 E.01212
G2 X138.07 Y103.646 I-1.835 J598.688 E.13823
G1 X137.938 Y103.603 E.0044
G1 X137.844 Y103.42 E.00653
G1 X137.844 Y103.086 E.01057
G1 X134.812 Y103.086 E.09597
G1 X134.812 Y103.42 E.01057
G1 X134.681 Y103.625 E.00771
G3 X132.162 Y103.639 I-1.517 J-47.418 E.07975
G3 X131.974 Y103.536 I.001 J-.226 E.00708
G1 X131.938 Y103.391 E.00472
G1 X132.244 Y100.213 E.1011
; LINE_WIDTH: 0.56781
G1 X132.281 Y100.128 E.00322
; LINE_WIDTH: 0.615623
G1 X132.319 Y100.043 E.00351
; LINE_WIDTH: 0.663436
G1 X132.356 Y99.959 E.0038
; LINE_WIDTH: 0.695976
G1 X132.422 Y99.932 E.00308
; LINE_WIDTH: 0.728516
G1 X132.489 Y99.905 E.00323
; LINE_WIDTH: 0.738786
G1 X133.014 Y99.994 E.02447
G1 X133.162 Y99.195 E.03733
G1 X132.618 Y99.184 E.025
; LINE_WIDTH: 0.731126
G1 X132.419 Y99.155 E.00914
; LINE_WIDTH: 0.697281
G1 X132.22 Y99.126 E.00869
; LINE_WIDTH: 0.663436
G1 X132.056 Y99.076 E.007
; LINE_WIDTH: 0.615623
G1 X131.893 Y99.026 E.00647
; LINE_WIDTH: 0.56781
G1 X131.73 Y98.976 E.00594
; LINE_WIDTH: 0.519996
G2 X131.038 Y98.936 I-.49 J2.519 E.02201
G3 X122.131 Y116.662 I-50.886 J-14.468 E.63181
; LINE_WIDTH: 0.521596
G1 X121.486 Y117.435 E.03196
; LINE_WIDTH: 0.544336
G1 X121.182 Y117.834 E.01669
G1 X121.378 Y117.826 E.00652
; LINE_WIDTH: 0.531156
G1 X121.52 Y117.821 E.00461
; LINE_WIDTH: 0.521596
G1 X121.55 Y117.846 E.00124
; LINE_WIDTH: 0.521526
G1 X121.659 Y117.937 E.00453
; LINE_WIDTH: 0.521296
G1 X121.768 Y118.029 E.00452
; LINE_WIDTH: 0.521066
G1 X121.877 Y118.121 E.00452
; LINE_WIDTH: 0.520836
G1 X121.986 Y118.213 E.00452
; LINE_WIDTH: 0.520616
G1 X122.095 Y118.304 E.00452
; LINE_WIDTH: 0.520386
G1 X122.204 Y118.396 E.00452
; LINE_WIDTH: 0.520156
G1 X122.281 Y118.461 E.00318
; LINE_WIDTH: 0.519996
G1 X122.373 Y118.537 E.00378
G1 X122.503 Y118.646 E.00536
G1 X122.632 Y118.755 E.00536
G1 X122.762 Y118.864 E.00536
G1 X122.892 Y118.972 E.00536
G1 X123.022 Y119.081 E.00536
G1 X123.151 Y119.19 E.00536
G3 X123.216 Y119.281 I-.079 J.124 E.00362
G1 X123.261 Y119.367 E.00307
G1 X123.236 Y119.447 E.00266
; WIPE_START
M204 S10000
G1 X122.61 Y120.227 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X128.179 Y115.008 Z8.12 F36000
G1 X142.988 Y101.13 Z8.12
G1 Z7.72
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.798156
G1 F10310.021
G1 X142.892 Y101.13 E.00475
G1 X142.844 Y101.213 E.00475
G1 X142.892 Y101.296 E.00475
G1 X142.988 Y101.296 E.00475
G1 X143.035 Y101.213 E.00475
; WIPE_START
G1 X142.988 Y101.296 E-.076
G1 X142.892 Y101.296 E-.076
G1 X142.844 Y101.213 E-.076
G1 X142.892 Y101.13 E-.076
G1 X142.988 Y101.13 E-.07599
; WIPE_END
G1 E-.02 F1800
G1 X143.407 Y103.247 Z8.12 F36000
G1 Z7.72
G1 E.4 F1800
; LINE_WIDTH: 1.03442
G1 F7874.37
G1 X143.407 Y103.588 E.02221
; WIPE_START
G1 X143.407 Y103.247 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X136.027 Y101.3 Z8.12 F36000
G1 X131.49 Y100.103 Z8.12
G1 Z7.72
G1 E.4 F1800
; LINE_WIDTH: 1.05454
G1 F7719.075
G1 X131.607 Y99.736 E.02563
; WIPE_START
G1 X131.49 Y100.103 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X127.654 Y106.701 Z8.12 F36000
G1 X121.182 Y117.834 Z8.12
G1 Z7.72
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X120.135 Y119.086 E.05427
; WIPE_START
M204 S10000
G1 X120.776 Y118.319 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.233 Y122.312 Z8.12 F36000
G1 Z7.72
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X118.585 Y123.977 I-29.063 J-27.121 E.08944
G2 X119.737 Y127.052 I4.481 J.074 E.12836
G3 X121.496 Y128.937 I-32.67 J32.247 E.09845
G3 X121.262 Y134.483 I-3.589 J2.626 E.22886
G3 X121.111 Y135.983 I-8.33 J-.081 E.05766
G3 X126.214 Y138.655 I-36.11 J75.191 E.21997
G3 X127.278 Y136.477 I4.919 J1.055 E.09344
G2 X129.037 Y134.592 I-32.67 J-32.247 E.09845
G2 X128.727 Y128.937 I-3.667 J-2.635 E.23374
G3 X126.968 Y127.052 I32.67 J-32.247 E.09845
G3 X127.278 Y121.396 I3.667 J-2.635 E.23374
G2 X129.037 Y119.511 I-32.67 J-32.247 E.09845
G2 X128.727 Y113.855 I-3.667 J-2.635 E.23374
G3 X127.546 Y112.608 I21.606 J-21.633 E.06559
G3 X126.312 Y114.599 I-56.621 J-33.702 E.08944
G1 X141.945 Y144.298 F36000
G1 F13446.283
G1 X141.945 Y141.955 E.08944
G3 X141.707 Y137.42 I3.685 J-2.468 E.18204
G1 X141.945 Y137.075 E.01602
G1 X141.945 Y126.874 E.38946
G3 X141.707 Y122.339 I3.685 J-2.468 E.18204
G1 X141.945 Y121.994 E.01602
G1 X141.945 Y111.793 E.38946
G3 X141.707 Y107.257 I3.685 J-2.468 E.18204
G1 X141.945 Y106.912 E.01602
G1 X141.945 Y105.865 E.04
G1 X138.073 Y105.867 E.14786
G3 X136.566 Y105.308 I.065 J-2.484 E.06249
G1 X136.09 Y105.308 E.01816
G3 X134.937 Y105.842 I-1.418 J-1.547 E.04931
G2 X134.818 Y111.97 I3.182 J3.126 E.25796
G3 X136.577 Y113.855 I-32.677 J32.254 E.09845
G3 X136.267 Y119.511 I-3.667 J2.635 E.23374
G2 X134.508 Y121.396 I32.664 J32.241 E.09845
G2 X134.818 Y127.052 I3.667 J2.635 E.23374
G3 X136.577 Y128.937 I-32.677 J32.254 E.09845
G3 X136.267 Y134.592 I-3.667 J2.635 E.23374
G2 X134.508 Y136.477 I32.664 J32.241 E.09845
G2 X134.818 Y142.133 I3.667 J2.635 E.23374
G3 X136.577 Y144.018 I-32.677 J32.254 E.09845
G3 X137.356 Y147.331 I-3.21 J2.503 E.13406
G3 X138.973 Y149.026 I-40.758 J40.483 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 7.88
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X138.283 Y148.302 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L49
M991 S0 P48 ;notify layer change


G17
G3 Z8.12 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X124.571 Y120.535
G1 Z7.88
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X124.428 Y120.763 E.01028
M73 P76 R4
G1 X123.613 Y121.735 E.04844
G1 X123.23 Y122.084 E.01976
G1 X122.85 Y122.29 E.01651
G1 X122.384 Y122.415 E.01844
G1 X121.913 Y122.422 E.01797
G1 X121.321 Y122.262 E.02342
G1 X120.866 Y121.977 E.02049
G1 X120.305 Y121.507 E.02796
G3 X114.848 Y126.662 I-41.023 J-37.951 E.28683
G1 X118.015 Y129.011 E.15054
G1 X118.483 Y129.401 E.02324
G1 X119.156 Y130.105 E.0372
G1 X119.749 Y130.932 E.03886
G1 X120.215 Y131.838 E.03888
G1 X120.493 Y132.619 E.03165
G1 X120.684 Y133.482 E.03374
G1 X120.765 Y134.489 E.03859
G1 X120.698 Y135.504 E.03883
G1 X120.538 Y136.268 E.0298
G3 X130.505 Y142.099 I-21.898 J48.857 E.44171
G3 X142.443 Y154.069 I-32.641 J44.494 E.64801
G1 X142.443 Y105.365 E1.85949
G3 X139.683 Y105.363 I-.655 J-1391.592 E.1054
G1 X137.683 Y105.372 E.07636
G1 X137.135 Y105.296 E.02112
G1 X136.642 Y105.076 E.02059
G1 X136.343 Y104.836 E.01465
G1 X136.044 Y105.076 E.01465
G1 X135.551 Y105.296 E.02059
G1 X135.045 Y105.371 E.01956
G3 X134.157 Y105.369 I-.325 J-44.351 E.03391
G1 X132.157 Y105.363 E.07636
G1 X131.702 Y105.308 E.01749
G1 X131.034 Y105.003 E.028
G1 X130.846 Y104.816 E.01015
G3 X123.624 Y117.546 I-51.177 J-20.618 E.56042
G1 X124.186 Y118.016 E.02797
G1 X124.554 Y118.427 E.02107
G1 X124.791 Y118.918 E.0208
G1 X124.875 Y119.336 E.0163
G1 X124.854 Y119.847 E.01953
G1 X124.711 Y120.312 E.01857
G1 X124.619 Y120.459 E.0066
G1 X124.021 Y120.318 F36000
G1 F13446.369
G1 X123.164 Y121.359 E.05149
G1 X122.857 Y121.63 E.01561
G1 X122.53 Y121.783 E.0138
G1 X122.2 Y121.845 E.01283
G1 X121.836 Y121.818 E.01391
G1 X121.538 Y121.717 E.01204
G1 X121.126 Y121.431 E.01915
G1 X120.241 Y120.689 E.04406
G3 X113.893 Y126.683 I-39.293 J-35.259 E.33371
G1 X117.679 Y129.491 E.17996
G1 X118.106 Y129.849 E.02126
G1 X118.727 Y130.503 E.03446
G1 X119.266 Y131.263 E.03557
G1 X119.688 Y132.095 E.0356
G1 X119.937 Y132.804 E.02868
G1 X120.11 Y133.599 E.03109
G1 X120.18 Y134.525 E.03543
G1 X120.115 Y135.453 E.03555
G1 X119.931 Y136.289 E.03266
G1 X119.836 Y136.599 E.01238
G3 X130.932 Y143.157 I-21.222 J48.578 E.49334
G3 X142.92 Y155.769 I-32.176 J42.587 E.66731
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.639 E1.95133
G1 X142.434 Y104.781 E.02336
G2 X139.68 Y104.778 I-3.303 J1429.994 E.10514
G1 X137.68 Y104.786 E.07636
G1 X137.297 Y104.733 E.01478
G1 X136.952 Y104.579 E.0144
G1 X136.7 Y104.376 E.01236
G1 X136.639 Y104.226 E.00618
G1 X136.047 Y104.226 E.02257
G1 X135.986 Y104.376 E.00618
G1 X135.571 Y104.666 E.01934
G1 X135.138 Y104.78 E.01709
G3 X134.158 Y104.783 I-.523 J-10.203 E.03741
G1 X132.158 Y104.777 E.07636
G1 X131.84 Y104.738 E.01224
G1 X131.373 Y104.526 E.01959
G1 X131.028 Y104.17 E.01892
G1 X130.818 Y103.645 E.02161
G1 X130.805 Y103.298 E.01323
G3 X122.809 Y117.626 I-50.151 J-18.594 E.62894
G1 X123.81 Y118.465 E.04986
G1 X124.083 Y118.777 E.01584
G1 X124.242 Y119.124 E.01455
G1 X124.298 Y119.511 E.01492
G1 X124.244 Y119.887 E.01453
G1 X124.087 Y120.239 E.01472
G1 X124.079 Y120.249 E.00048
G1 X123.576 Y119.938 F36000
G1 F13446.369
G1 X123.53 Y120.011 E.0033
G1 X122.715 Y120.983 E.04844
G1 X122.46 Y121.183 E.01238
G1 X122.165 Y121.26 E.01163
G1 X121.958 Y121.245 E.00794
G1 X121.618 Y121.08 E.0144
G1 X120.169 Y119.864 E.07222
G1 X119.604 Y120.538 E.03354
G3 X112.93 Y126.698 I-39.862 J-36.494 E.34719
G1 X117.324 Y129.958 E.2089
G1 X117.728 Y130.297 E.02015
G1 X118.298 Y130.902 E.03172
G1 X118.783 Y131.594 E.03229
G3 X119.536 Y133.717 I-5.5 J3.146 E.08645
G1 X119.595 Y134.56 E.03228
G1 X119.531 Y135.403 E.03226
G1 X119.361 Y136.154 E.02938
G1 X119.08 Y136.91 E.03082
G3 X130.579 Y143.624 I-20.708 J48.672 E.50972
G3 X142.647 Y156.415 I-31.78 J42.073 E.67457
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y105.065 E1.95363
G1 X143.615 Y104.665 E.01527
; LINE_WIDTH: 0.659506
G1 F12371.864
G1 X143.595 Y104.487 E.0073
; LINE_WIDTH: 0.699016
G1 F11762.152
G1 X143.575 Y104.309 E.00776
; LINE_WIDTH: 0.738526
G1 F11167.849
G1 X143.555 Y104.131 E.00822
; LINE_WIDTH: 0.778036
G1 F10588.948
G1 X143.536 Y103.953 E.00869
; LINE_WIDTH: 0.82077
G1 F10013.564
G1 X143.514 Y103.892 E.00331
; LINE_WIDTH: 0.863503
G1 F9497.489
G1 X143.493 Y103.831 E.00349
; LINE_WIDTH: 0.906236
G1 F9032
G1 X143.471 Y103.77 E.00367
; LINE_WIDTH: 0.94897
G1 F8610.008
G1 X143.45 Y103.709 E.00385
; LINE_WIDTH: 0.991703
G1 F8225.689
G1 X143.429 Y103.649 E.00403
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X143.407 Y103.588 E.00421
G1 X143.369 Y103.632 E.00385
; LINE_WIDTH: 0.991703
G1 F8225.689
G1 X143.33 Y103.676 E.00368
; LINE_WIDTH: 0.94897
G1 F8610.008
G1 X143.291 Y103.721 E.00352
; LINE_WIDTH: 0.906236
G1 F9032
G1 X143.252 Y103.765 E.00335
; LINE_WIDTH: 0.863503
G1 F9497.489
G1 X143.213 Y103.81 E.00319
; LINE_WIDTH: 0.82077
G1 F10013.564
G1 X143.174 Y103.854 E.00302
; LINE_WIDTH: 0.778036
G1 F10588.948
G1 X143.072 Y103.921 E.00594
; LINE_WIDTH: 0.738526
G1 F11183.063
G1 X142.97 Y103.989 E.00563
; LINE_WIDTH: 0.699016
G1 F11588.152
G1 X142.867 Y104.056 E.00531
; LINE_WIDTH: 0.659506
G1 F12000.448
G1 X142.765 Y104.123 E.00499
; LINE_WIDTH: 0.619996
G1 F13173.74
G1 X142.435 Y104.196 E.01289
G1 F13446.369
G1 X142.035 Y104.195 E.01527
G1 X139.678 Y104.192 E.09
G1 X137.678 Y104.2 E.07636
G1 X137.355 Y104.132 E.01258
G1 X137.118 Y103.967 E.01104
G1 X136.985 Y103.641 E.01344
G1 X135.701 Y103.641 E.04904
G1 X135.568 Y103.967 E.01344
G1 X135.269 Y104.156 E.0135
G1 X135.009 Y104.2 E.01008
G1 X132.16 Y104.191 E.10877
G1 X131.979 Y104.169 E.00698
G1 X131.699 Y104.038 E.01181
G1 X131.515 Y103.845 E.01017
G1 X131.393 Y103.529 E.01296
G3 X131.448 Y102.712 I38.095 J2.165 E.03126
G1 X131.501 Y102.165 E.02096
G1 X131.539 Y101.767 E.01527
G1 F12788.082
G1 X131.577 Y101.369 E.01527
; LINE_WIDTH: 0.669038
G1 F11424.965
G1 X131.567 Y101.216 E.00636
; LINE_WIDTH: 0.718079
G1 F10921.201
G1 X131.557 Y101.062 E.00686
; LINE_WIDTH: 0.76712
G1 F10428.796
G1 X131.548 Y100.908 E.00735
; LINE_WIDTH: 0.816161
G1 F9947.75
G1 X131.538 Y100.755 E.00784
; LINE_WIDTH: 0.865202
G1 F9478.06
G1 X131.528 Y100.601 E.00833
; LINE_WIDTH: 0.914244
G1 F8949.805
G1 X131.518 Y100.448 E.00882
; LINE_WIDTH: 0.963285
G1 F8477.326
G1 X131.508 Y100.294 E.00932
; LINE_WIDTH: 1.01233
G1 F8052.23
G1 X131.498 Y100.141 E.00981
; LINE_WIDTH: 1.03836
G1 F7843.469
G1 X131.49 Y100.103 E.00256
G1 X131.466 Y100.134 E.00258
; LINE_WIDTH: 1.01233
G1 F8052.23
G1 X131.397 Y100.272 E.00981
; LINE_WIDTH: 0.963285
G1 F8477.326
G1 X131.327 Y100.409 E.00932
; LINE_WIDTH: 0.914244
G1 F8949.805
G1 X131.258 Y100.546 E.00882
; LINE_WIDTH: 0.865202
G1 F9478.06
G1 X131.189 Y100.684 E.00833
; LINE_WIDTH: 0.816161
G1 F10072.587
G1 X131.12 Y100.821 E.00784
; LINE_WIDTH: 0.76712
G1 F10556.6
G1 X131.05 Y100.958 E.00735
; LINE_WIDTH: 0.718079
G1 F11051.971
G1 X130.981 Y101.096 E.00686
; LINE_WIDTH: 0.669038
G1 F11558.7
G1 X130.912 Y101.233 E.00636
; LINE_WIDTH: 0.619996
G1 F12929.548
G1 X130.779 Y101.61 E.01527
G1 F13446.369
G1 X130.646 Y101.988 E.01527
G1 X130.262 Y103.081 E.04426
G3 X124.03 Y114.993 I-50.328 J-18.742 E.51461
G1 X123.661 Y115.524 E.02468
G3 X121.984 Y117.699 I-24.758 J-17.358 E.10489
G1 X123.433 Y118.914 E.07222
G1 X123.643 Y119.19 E.01324
G1 X123.712 Y119.511 E.0125
G1 X123.643 Y119.831 E.0125
G1 X123.624 Y119.861 E.00138
; WIPE_START
G1 X123.53 Y120.011 E-.06702
G1 X123.001 Y120.642 E-.31298
; WIPE_END
G1 E-.02 F1800
G1 X128.808 Y115.689 Z8.28 F36000
G1 X143.407 Y103.241 Z8.28
G1 Z7.88
G1 E.4 F1800
; LINE_WIDTH: 1.03006
G1 F7908.851
G1 X143.43 Y103.067 E.0114
; LINE_WIDTH: 0.984494
G1 F8288.098
G1 X143.453 Y102.892 E.01088
; LINE_WIDTH: 0.938932
G1 F8705.549
G1 X143.476 Y102.718 E.01036
; LINE_WIDTH: 0.89337
G1 F9167.281
G1 X143.499 Y102.544 E.00984
; LINE_WIDTH: 0.847807
G1 F9680.738
G1 X143.522 Y102.37 E.00932
; LINE_WIDTH: 0.802245
G1 F10255.123
G1 X143.545 Y102.196 E.00879
; LINE_WIDTH: 0.756683
G1 F10813.763
G1 X143.568 Y102.022 E.00827
; LINE_WIDTH: 0.711121
G1 F11387.223
G1 X143.591 Y101.847 E.00775
; LINE_WIDTH: 0.665558
G1 F11975.497
G1 X143.615 Y101.673 E.00723
; LINE_WIDTH: 0.619996
G1 F13304.781
G1 X143.615 Y101.273 E.01527
G1 F12595.115
G1 X143.615 Y100.91 E.01388
; LINE_WIDTH: 0.664994
G1 F11362.976
G1 X143.592 Y100.801 E.00458
; LINE_WIDTH: 0.709991
G1 F10997.814
G1 X143.57 Y100.691 E.00491
; LINE_WIDTH: 0.754989
G1 F10638.615
G1 X143.547 Y100.582 E.00524
; LINE_WIDTH: 0.799986
G1 F10285.379
G1 X143.525 Y100.473 E.00556
G1 X143.543 Y99.62 E.04258
; LINE_WIDTH: 0.762936
G1 F10808.401
G3 X142.863 Y99.707 I-.982 J-4.963 E.03261
; LINE_WIDTH: 0.799986
G1 F10285.379
G1 X142.764 Y100.408 E.03532
G1 X142.662 Y100.531 E.00795
; LINE_WIDTH: 0.754989
G1 F10791.882
G1 X142.561 Y100.653 E.00748
; LINE_WIDTH: 0.709991
G1 F11310.559
G1 X142.459 Y100.775 E.00701
; LINE_WIDTH: 0.664994
G1 F11841.409
G1 X142.357 Y100.898 E.00655
; LINE_WIDTH: 0.619996
G1 F12390.842
G1 X142.209 Y100.963 E.00615
G1 F12812.044
G1 X142.231 Y101.082 E.00462
G1 F13446.369
G1 X142.579 Y101.222 E.0143
G1 X142.69 Y101.309 E.00538
G1 X142.922 Y101.736 E.01855
; LINE_WIDTH: 0.665558
G1 F11941.56
G1 X142.976 Y101.903 E.00723
; LINE_WIDTH: 0.711121
G1 F11354.134
G1 X143.029 Y102.07 E.00775
; LINE_WIDTH: 0.756683
G1 F10781.524
G1 X143.083 Y102.237 E.00827
; LINE_WIDTH: 0.802245
G1 F10223.718
G1 X143.137 Y102.405 E.00879
; LINE_WIDTH: 0.847807
G1 F9680.738
G1 X143.191 Y102.572 E.00932
; LINE_WIDTH: 0.89337
G1 F9167.281
G1 X143.245 Y102.739 E.00984
; LINE_WIDTH: 0.938932
G1 F8705.549
G1 X143.299 Y102.906 E.01036
; LINE_WIDTH: 0.984494
G1 F8288.098
G1 X143.353 Y103.073 E.01088
; LINE_WIDTH: 1.03006
G1 F7908.851
G1 X143.379 Y103.155 E.00556
; WIPE_START
G1 X143.43 Y103.067 E-.03876
G1 X143.453 Y102.892 E-.06676
G1 X143.476 Y102.718 E-.06676
G1 X143.499 Y102.544 E-.06676
G1 X143.522 Y102.37 E-.06676
G1 X143.545 Y102.196 E-.06676
G1 X143.548 Y102.176 E-.00744
; WIPE_END
G1 E-.02 F1800
G1 X137.747 Y107.137 Z8.28 F36000
G1 X123.106 Y119.656 Z8.28
G1 Z7.88
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X122.291 Y120.628 E.04017
G1 X122.132 Y120.708 E.00565
G1 X122.042 Y120.679 E.00298
G3 X121.948 Y120.634 I.008 J-.141 E.00339
G1 X121.836 Y120.54 E.00462
G1 X121.724 Y120.447 E.00462
G1 X121.612 Y120.353 E.00462
G1 X121.5 Y120.259 E.00462
G1 X121.388 Y120.165 E.00462
G1 X121.276 Y120.071 E.00462
G1 X121.197 Y120.005 E.00326
; LINE_WIDTH: 0.520326
G1 X121.108 Y119.931 E.00367
; LINE_WIDTH: 0.520776
G1 X120.982 Y119.826 E.00521
; LINE_WIDTH: 0.521226
G1 X120.856 Y119.721 E.00521
; LINE_WIDTH: 0.521686
G1 X120.73 Y119.616 E.00522
; LINE_WIDTH: 0.522136
G1 X120.604 Y119.511 E.00522
; LINE_WIDTH: 0.522596
G1 X120.477 Y119.406 E.00523
; LINE_WIDTH: 0.523046
G1 X120.351 Y119.3 E.00523
; LINE_WIDTH: 0.523196
G1 X120.309 Y119.265 E.00174
; LINE_WIDTH: 0.531656
G1 X120.281 Y119.145 E.00403
; LINE_WIDTH: 0.544336
G1 X120.238 Y118.963 E.00619
G1 X119.926 Y119.299 E.01524
; LINE_WIDTH: 0.523196
G1 X119.181 Y120.183 E.03683
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-40.856 J-37.722 E.30726
G1 X116.99 Y130.398 E.1963
G1 X117.373 Y130.72 E.01584
G1 X117.893 Y131.278 E.02416
G1 X118.327 Y131.907 E.02421
G3 X118.995 Y133.828 I-4.991 J2.81 E.06474
G1 X119.044 Y134.594 E.0243
G1 X118.98 Y135.355 E.02419
G1 X118.824 Y136.026 E.0218
G1 X118.548 Y136.752 E.0246
G1 X118.31 Y137.19 E.01576
G3 X130.249 Y144.067 I-19.584 J47.797 E.43752
G3 X142.39 Y157.025 I-31.645 J41.818 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.376 E1.81473
G1 X144.167 Y98.946 E.01361
G2 X142.89 Y99.081 I.098 J7.035 E.04072
G3 X141.705 Y99.086 I-1.338 J-169.306 E.03753
G1 X142.16 Y99.465 E.01875
G1 X142.238 Y99.677 E.00715
G1 X142.131 Y100.3 E.02002
G1 X142.038 Y100.446 E.00546
G1 X141.878 Y100.485 E.00523
G1 X141.55 Y100.44 E.01047
G1 X141.748 Y101.508 E.03439
G1 X142.224 Y101.66 E.01583
G3 X142.378 Y101.835 I-.069 J.215 E.00773
G1 X142.659 Y103.378 E.04967
G3 X142.436 Y103.643 I-.247 J.018 E.01212
G2 X139.675 Y103.639 I-3.3 J1432.482 E.0874
G1 X137.675 Y103.648 E.06332
G1 X137.513 Y103.58 E.00557
G1 X137.449 Y103.422 E.0054
G1 X137.449 Y103.088 E.01057
G1 X135.237 Y103.088 E.07001
G1 X135.237 Y103.422 E.01057
G1 X135.131 Y103.613 E.00693
G1 X135.011 Y103.648 E.00396
G1 X132.162 Y103.639 E.0902
G1 X132.028 Y103.594 E.00446
G1 X131.94 Y103.446 E.00545
G3 X132.052 Y102.203 I63.12 J5.101 E.03952
G1 X132.244 Y100.213 E.06332
; LINE_WIDTH: 0.567446
G1 X132.281 Y100.128 E.00321
; LINE_WIDTH: 0.614896
G1 X132.319 Y100.044 E.00349
; LINE_WIDTH: 0.662346
G1 X132.356 Y99.959 E.00378
; LINE_WIDTH: 0.694951
G1 X132.423 Y99.932 E.00308
; LINE_WIDTH: 0.727556
G1 X132.489 Y99.905 E.00323
; LINE_WIDTH: 0.737666
G1 X133.012 Y99.995 E.02435
G1 X133.16 Y99.197 E.03721
G1 X132.618 Y99.186 E.02487
; LINE_WIDTH: 0.730116
G1 X132.418 Y99.156 E.00913
; LINE_WIDTH: 0.696231
G1 X132.219 Y99.127 E.00869
; LINE_WIDTH: 0.662346
G1 X132.057 Y99.077 E.00697
; LINE_WIDTH: 0.614896
G1 X131.894 Y99.027 E.00645
; LINE_WIDTH: 0.567446
G1 X131.731 Y98.976 E.00592
; LINE_WIDTH: 0.519996
G2 X131.038 Y98.936 I-.492 J2.491 E.02206
G3 X122.124 Y116.672 I-51.194 J-14.623 E.63213
; LINE_WIDTH: 0.521596
G1 X121.383 Y117.557 E.03667
; LINE_WIDTH: 0.544336
G1 X121.079 Y117.956 E.01669
G1 X121.275 Y117.949 E.00652
; LINE_WIDTH: 0.531156
G1 X121.417 Y117.943 E.00461
; LINE_WIDTH: 0.521596
G1 X121.452 Y117.972 E.00144
; LINE_WIDTH: 0.521526
G1 X121.579 Y118.079 E.00525
; LINE_WIDTH: 0.521296
G1 X121.705 Y118.185 E.00525
; LINE_WIDTH: 0.521066
G1 X121.832 Y118.292 E.00525
; LINE_WIDTH: 0.520836
G1 X121.959 Y118.398 E.00525
; LINE_WIDTH: 0.520616
G1 X122.085 Y118.505 E.00524
; LINE_WIDTH: 0.520386
G1 X122.212 Y118.611 E.00524
; LINE_WIDTH: 0.520156
G1 X122.301 Y118.686 E.00369
; LINE_WIDTH: 0.519996
G1 X122.38 Y118.752 E.00327
G1 X122.492 Y118.846 E.00463
G1 X122.604 Y118.94 E.00463
G1 X122.716 Y119.034 E.00463
G1 X122.828 Y119.128 E.00463
G1 X122.941 Y119.222 E.00463
G1 X123.053 Y119.316 E.00463
G3 X123.113 Y119.404 I-.076 J.117 E.00347
G1 X123.158 Y119.49 E.00308
G1 X123.133 Y119.57 E.00263
; WIPE_START
M204 S10000
G1 X122.511 Y120.352 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X128.052 Y115.104 Z8.28 F36000
G1 X143.098 Y100.854 Z8.28
G1 Z7.88
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.545596
G1 F15403.123
G2 X143.13 Y100.91 I-.028 J.053 E.01031
; WIPE_START
G1 X143.098 Y100.965 E-.076
G1 X143.034 Y100.965 E-.076
G1 X143.002 Y100.91 E-.076
G1 X143.034 Y100.854 E-.076
G1 X143.098 Y100.854 E-.076
; WIPE_END
G1 E-.02 F1800
G1 X143.407 Y103.247 Z8.28 F36000
G1 Z7.88
G1 E.4 F1800
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X143.407 Y103.588 E.02221
; WIPE_START
G1 X143.407 Y103.247 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X136.027 Y101.3 Z8.28 F36000
G1 X131.49 Y100.103 Z8.28
G1 Z7.88
G1 E.4 F1800
; LINE_WIDTH: 1.05442
G1 F7719.983
G1 X131.606 Y99.736 E.02558
; WIPE_START
G1 X131.49 Y100.103 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X127.645 Y106.696 Z8.28 F36000
G1 X121.079 Y117.956 Z8.28
G1 Z7.88
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X120.238 Y118.963 E.04363
; WIPE_START
M204 S10000
G1 X120.879 Y118.196 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.211 Y122.332 Z8.28 F36000
G1 Z7.88
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X118.564 Y123.997 I-27.47 J-25.523 E.08945
G2 X119.871 Y127.052 I4.702 J-.205 E.12964
G3 X121.642 Y128.937 I-16.636 J17.4 E.09879
G3 X121.251 Y134.324 I-3.706 J2.439 E.22172
G3 X121.111 Y135.983 I-9.774 J.006 E.06364
G3 X126.234 Y138.667 I-35.835 J74.647 E.22086
G3 X127.412 Y136.477 I5.108 J1.337 E.09582
G2 X129.478 Y134.121 I-9.193 J-10.143 E.11991
G2 X128.592 Y128.937 I-4.259 J-1.94 E.21322
G3 X126.527 Y126.58 I9.193 J-10.143 E.11991
G3 X127.412 Y121.396 I4.259 J-1.94 E.21322
G2 X129.183 Y119.511 I-16.636 J-17.4 E.09879
G2 X129.294 Y114.798 I-3.712 J-2.445 E.18971
G2 X127.483 Y112.714 I-14.894 J11.116 E.10552
G3 X126.243 Y114.701 I-47.329 J-28.155 E.08943
G1 X141.942 Y144.52 F36000
G1 F13446.283
G1 X141.942 Y142.177 E.08944
G3 X141.792 Y137.42 I3.636 J-2.496 E.19193
G1 X141.943 Y137.217 E.00966
G1 X141.944 Y127.097 E.38637
G3 X141.792 Y122.339 I3.633 J-2.498 E.19201
G1 X141.944 Y122.134 E.00974
G1 X141.945 Y112.017 E.38624
G3 X141.792 Y107.257 I3.631 J-2.499 E.19209
G1 X141.945 Y107.051 E.00983
G1 X141.945 Y105.864 E.0453
G1 X137.469 Y105.861 E.17091
G3 X136.374 Y105.496 I.408 J-3.05 E.04435
G3 X134.766 Y105.86 I-1.475 J-2.785 E.06365
G2 X134.067 Y106.786 I2.396 J2.536 E.04451
G2 X134.953 Y111.97 I4.259 J1.94 E.21322
G3 X137.018 Y114.327 I-9.193 J10.143 E.11991
G3 X136.133 Y119.511 I-4.259 J1.94 E.21322
G2 X134.067 Y121.867 I9.193 J10.143 E.11991
G2 X134.953 Y127.052 I4.259 J1.94 E.21322
G3 X137.018 Y129.408 I-9.193 J10.143 E.11991
G3 X136.133 Y134.592 I-4.259 J1.94 E.21322
G2 X134.067 Y136.949 I9.193 J10.143 E.11991
G2 X134.953 Y142.133 I4.259 J1.94 E.21322
G3 X136.723 Y144.018 I-16.636 J17.4 E.09879
G3 X137.344 Y147.319 I-3.563 J2.379 E.13178
G3 X138.961 Y149.014 I-40.081 J39.863 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 8.04
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X138.271 Y148.29 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L50
M991 S0 P49 ;notify layer change


G17
G3 Z8.28 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X124.445 Y120.71
G1 Z8.04
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X124.325 Y120.886 E.00813
G1 X123.715 Y121.613 E.03622
G1 X123.317 Y121.973 E.02051
G1 X122.936 Y122.174 E.01646
G1 X122.469 Y122.294 E.01839
G1 X121.941 Y122.29 E.02018
G1 X121.472 Y122.16 E.01857
G1 X120.979 Y121.864 E.02194
G1 X120.413 Y121.389 E.0282
G3 X114.848 Y126.662 I-41.011 J-37.711 E.29295
G1 X118.015 Y129.012 E.15055
G1 X118.485 Y129.402 E.02333
G1 X119.155 Y130.103 E.03702
G1 X119.749 Y130.931 E.0389
G1 X120.215 Y131.838 E.03892
G3 X120.715 Y133.691 I-7.775 J3.093 E.07343
G1 X120.765 Y134.489 E.03055
G1 X120.698 Y135.504 E.03882
G1 X120.538 Y136.268 E.02979
G3 X130.508 Y142.102 I-22.419 J49.749 E.44184
G3 X142.443 Y154.069 I-32.646 J44.494 E.64784
G1 X142.443 Y105.365 E1.85948
G2 X139.211 Y105.365 I-1.562 J317.687 E.12342
G1 X137.211 Y105.374 E.07636
G1 X136.729 Y105.315 E.01852
G1 X136.345 Y105.168 E.01572
G1 X135.878 Y105.334 E.01892
G3 X134.156 Y105.369 I-1.129 J-13.103 E.06579
G1 X132.156 Y105.363 E.07636
G1 X131.702 Y105.307 E.01747
G1 X131.034 Y105.003 E.028
G1 X130.846 Y104.816 E.01015
G3 X123.528 Y117.674 I-50.96 J-20.492 E.56656
G1 X124.083 Y118.139 E.02766
G1 X124.498 Y118.622 E.0243
G1 X124.705 Y119.095 E.01971
G1 X124.78 Y119.612 E.01994
G1 X124.738 Y120.039 E.0164
G1 X124.546 Y120.56 E.0212
G1 X124.495 Y120.635 E.00346
G1 X123.995 Y120.321 F36000
G1 F13446.369
G1 X123.932 Y120.439 E.00509
G1 X123.267 Y121.237 E.03965
G1 X122.949 Y121.514 E.01612
G3 X121.697 Y121.62 I-.727 J-1.154 E.04984
G1 X121.326 Y121.39 E.01666
G1 X120.344 Y120.567 E.04891
G1 X120.008 Y120.965 E.0199
G3 X113.893 Y126.683 I-40.194 J-36.856 E.31995
G1 X117.666 Y129.482 E.17936
G1 X118.108 Y129.85 E.02195
G1 X118.726 Y130.502 E.03428
G1 X119.266 Y131.262 E.03562
G1 X119.688 Y132.095 E.03564
G3 X120.136 Y133.793 I-6.992 J2.749 E.06722
G1 X120.18 Y134.525 E.02798
G1 X120.115 Y135.454 E.03554
G1 X119.931 Y136.289 E.03265
G1 X119.836 Y136.599 E.01239
G3 X130.932 Y143.157 I-21.596 J49.211 E.4933
G3 X142.92 Y155.769 I-32.376 J42.778 E.66728
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.639 E1.95133
G1 X142.434 Y104.781 E.02336
G2 X139.208 Y104.78 I-1.974 J676.862 E.12316
G1 X137.208 Y104.788 E.07636
G1 X136.785 Y104.723 E.01637
G1 X136.382 Y104.514 E.0173
G1 X136.345 Y104.467 E.00232
G1 X136.047 Y104.668 E.0137
G1 X135.632 Y104.78 E.01643
G3 X134.158 Y104.784 I-.827 J-39.715 E.05628
G1 X132.158 Y104.777 E.07636
G1 X131.84 Y104.738 E.01223
G1 X131.373 Y104.525 E.01959
G1 X131.028 Y104.17 E.01892
G1 X130.818 Y103.644 E.02161
G1 X130.805 Y103.301 E.01312
G3 X124.897 Y114.773 I-51.13 J-19.078 E.49382
G3 X122.706 Y117.749 I-32.229 J-21.427 E.14114
G1 X123.707 Y118.588 E.04986
G1 X124.012 Y118.951 E.01811
G1 X124.15 Y119.285 E.01379
G1 X124.192 Y119.719 E.01665
G1 X124.11 Y120.107 E.01513
G1 X124.037 Y120.242 E.00586
G1 X123.471 Y120.064 F36000
G1 F13446.369
G1 X123.427 Y120.133 E.00312
G1 X122.818 Y120.86 E.03622
G1 X122.556 Y121.063 E.01265
G1 X122.261 Y121.138 E.01161
G1 X121.922 Y121.079 E.01313
G1 X121.702 Y120.941 E.00991
G1 X120.271 Y119.742 E.07127
G1 X119.604 Y120.538 E.03964
G3 X112.93 Y126.698 I-39.742 J-36.365 E.34719
G1 X117.317 Y129.952 E.20857
G1 X117.73 Y130.298 E.02057
G1 X118.297 Y130.9 E.03155
G1 X118.783 Y131.594 E.03234
G3 X119.513 Y133.605 I-5.361 J3.084 E.08209
G1 X119.596 Y134.561 E.03663
G1 X119.531 Y135.403 E.03226
G1 X119.361 Y136.154 E.02937
G1 X119.08 Y136.91 E.03082
G3 X130.579 Y143.624 I-20.387 J48.122 E.50974
G3 X142.648 Y156.415 I-31.964 J42.247 E.67454
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y105.065 E1.95363
G1 X143.615 Y104.665 E.01527
; LINE_WIDTH: 0.659506
G1 F12371.864
G1 X143.595 Y104.487 E.0073
; LINE_WIDTH: 0.699016
G1 F11762.152
G1 X143.575 Y104.309 E.00776
; LINE_WIDTH: 0.738526
M73 P77 R4
G1 F11167.849
G1 X143.555 Y104.131 E.00822
; LINE_WIDTH: 0.778036
G1 F10588.948
G1 X143.536 Y103.953 E.00869
; LINE_WIDTH: 0.820766
G1 F10013.607
G1 X143.514 Y103.892 E.00331
; LINE_WIDTH: 0.863496
G1 F9497.565
G1 X143.493 Y103.831 E.00349
; LINE_WIDTH: 0.906226
G1 F9032.103
G1 X143.471 Y103.77 E.00367
; LINE_WIDTH: 0.948956
G1 F8610.134
G1 X143.45 Y103.709 E.00385
; LINE_WIDTH: 0.991686
G1 F8225.833
G1 X143.429 Y103.649 E.00403
; LINE_WIDTH: 1.03442
G1 F7874.37
G1 X143.407 Y103.588 E.00421
G1 X143.369 Y103.632 E.00385
; LINE_WIDTH: 0.991686
G1 F8225.833
G1 X143.33 Y103.676 E.00368
; LINE_WIDTH: 0.948956
G1 F8610.134
G1 X143.291 Y103.721 E.00352
; LINE_WIDTH: 0.906226
G1 F9032.103
G1 X143.252 Y103.765 E.00335
; LINE_WIDTH: 0.863496
G1 F9497.565
G1 X143.213 Y103.81 E.00319
; LINE_WIDTH: 0.820766
G1 F10013.607
G1 X143.174 Y103.854 E.00302
; LINE_WIDTH: 0.778036
G1 F10588.948
G1 X143.072 Y103.921 E.00594
; LINE_WIDTH: 0.738526
G1 F11183.063
G1 X142.97 Y103.989 E.00563
; LINE_WIDTH: 0.699016
G1 F11588.18
G1 X142.867 Y104.056 E.00531
; LINE_WIDTH: 0.659506
G1 F12000.477
G1 X142.765 Y104.123 E.00499
; LINE_WIDTH: 0.619996
G1 F13173.805
G1 X142.435 Y104.196 E.01289
G1 F13446.369
G1 X142.035 Y104.196 E.01527
G1 X139.206 Y104.194 E.10802
G1 X137.206 Y104.202 E.07636
G1 X136.883 Y104.134 E.01262
G1 X136.734 Y104.046 E.00657
G1 X136.49 Y103.738 E.01502
G1 X136.47 Y103.643 E.0037
G1 X136.219 Y103.643 E.00958
G1 X136.199 Y103.738 E.0037
G1 X135.955 Y104.046 E.01502
G1 X135.643 Y104.187 E.01306
G3 X134.16 Y104.198 I-.9 J-20.596 E.05663
G1 X132.16 Y104.191 E.07636
G1 X131.979 Y104.169 E.00698
G1 X131.699 Y104.038 E.01181
G1 X131.515 Y103.845 E.01017
G1 X131.393 Y103.531 E.01287
G3 X131.446 Y102.729 I8.679 J.173 E.0307
G1 X131.523 Y101.932 E.03054
G1 X131.561 Y101.534 E.01527
G1 F13260.923
G1 X131.6 Y101.136 E.01527
G1 F11872.13
G1 X131.638 Y100.738 E.01527
; LINE_WIDTH: 0.665112
G1 F10560.138
G1 X131.622 Y100.671 E.00281
; LINE_WIDTH: 0.710227
G1 F10343.582
G1 X131.605 Y100.605 E.00301
; LINE_WIDTH: 0.755343
G1 F10129.276
G1 X131.589 Y100.539 E.00321
; LINE_WIDTH: 0.800458
G1 F9917.207
G1 X131.573 Y100.472 E.00341
; LINE_WIDTH: 0.845574
G1 F9707.389
G1 X131.557 Y100.406 E.00362
; LINE_WIDTH: 0.89069
G1 F9195.97
G1 X131.54 Y100.339 E.00382
; LINE_WIDTH: 0.935805
G1 F8735.743
G1 X131.524 Y100.273 E.00402
; LINE_WIDTH: 0.980921
G1 F8319.385
G1 X131.508 Y100.207 E.00422
; LINE_WIDTH: 1.02604
G1 F7940.91
G1 X131.491 Y100.14 E.00442
G1 X131.445 Y100.199 E.00484
; LINE_WIDTH: 0.97679
G1 F8355.848
G1 X131.399 Y100.258 E.0046
; LINE_WIDTH: 0.927544
G1 F8816.541
G1 X131.352 Y100.316 E.00436
; LINE_WIDTH: 0.878298
G1 F9330.998
G1 X131.306 Y100.375 E.00412
; LINE_WIDTH: 0.829051
G1 F9909.213
G1 X131.259 Y100.434 E.00388
; LINE_WIDTH: 0.779805
G1 F10141.408
G1 X131.213 Y100.492 E.00364
; LINE_WIDTH: 0.730559
G1 F10376.312
G1 X131.166 Y100.551 E.0034
; LINE_WIDTH: 0.681313
G1 F10613.886
G1 X131.12 Y100.61 E.00316
; LINE_WIDTH: 0.632066
G1 F10678.274
G1 X131.108 Y100.626 E.00078
; LINE_WIDTH: 0.619996
G1 F11997.37
G1 X130.979 Y101.005 E.01527
G1 F13393.267
G1 X130.85 Y101.383 E.01527
G1 F13446.369
G1 X130.721 Y101.762 E.01527
G1 X130.372 Y102.785 E.04128
G3 X124.03 Y114.993 I-50.108 J-18.281 E.5267
G3 X121.881 Y117.821 I-26.017 J-17.539 E.13569
G1 X123.331 Y119.037 E.07222
G1 X123.551 Y119.339 E.01427
G1 X123.609 Y119.633 E.01147
G1 X123.54 Y119.955 E.01255
G1 X123.519 Y119.988 E.00151
; WIPE_START
G1 X123.427 Y120.133 E-.06525
G1 X122.895 Y120.768 E-.31475
; WIPE_END
G1 E-.02 F1800
G1 X128.698 Y115.81 Z8.44 F36000
G1 X143.407 Y103.241 Z8.44
G1 Z8.04
G1 E.4 F1800
; LINE_WIDTH: 1.03203
G1 F7893.233
G1 X143.42 Y102.983 E.01679
; LINE_WIDTH: 1.00657
G1 F8099.935
G1 X143.445 Y102.483 E.03169
; LINE_WIDTH: 0.957269
G1 F8532.585
G1 X143.47 Y101.984 E.03008
; LINE_WIDTH: 0.907971
G1 F9014.062
G1 X143.495 Y101.485 E.02848
; LINE_WIDTH: 0.858674
G1 F9553.127
G1 X143.52 Y100.985 E.02687
; LINE_WIDTH: 0.809376
G1 F10160.767
G1 X143.53 Y100.585 E.02021
G1 X143.543 Y100.041 E.02753
; LINE_WIDTH: 0.763086
G1 F10806.176
G1 X143.543 Y99.621 E.01993
G1 X143.259 Y99.658 E.0136
G1 X142.862 Y99.71 E.019
; LINE_WIDTH: 0.807546
G1 F10184.815
G1 X142.808 Y100.1 E.01988
G1 X142.753 Y100.496 E.02016
; LINE_WIDTH: 0.809376
G1 F10160.767
G1 X142.711 Y100.651 E.00812
G1 X142.794 Y100.772 E.00741
G1 X142.892 Y101.042 E.01452
; LINE_WIDTH: 0.834826
G1 F9837.725
G1 X142.95 Y101.293 E.01347
; LINE_WIDTH: 0.884126
G1 F9266.996
G1 X143.065 Y101.78 E.0277
; LINE_WIDTH: 0.933426
G1 F8758.857
G1 X143.179 Y102.267 E.02931
; LINE_WIDTH: 0.982726
G1 F8303.547
G1 X143.293 Y102.754 E.03091
; LINE_WIDTH: 1.03203
G1 F7893.233
G1 X143.386 Y103.153 E.02667
; WIPE_START
G1 X143.42 Y102.983 E-.06593
G1 X143.445 Y102.483 E-.19
G1 X143.461 Y102.157 E-.12407
; WIPE_END
G1 E-.02 F1800
G1 X137.678 Y107.138 Z8.44 F36000
G1 X123.004 Y119.778 Z8.44
G1 Z8.04
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X122.394 Y120.505 E.03004
G1 X122.25 Y120.584 E.0052
G1 X122.152 Y120.555 E.00324
G3 X122.054 Y120.515 I-.003 J-.131 E.00343
G1 X121.96 Y120.436 E.00389
G1 X121.866 Y120.357 E.00389
G1 X121.772 Y120.278 E.00389
G1 X121.677 Y120.199 E.00389
G1 X121.583 Y120.12 E.00389
G1 X121.489 Y120.041 E.00389
G1 X121.422 Y119.985 E.00274
; LINE_WIDTH: 0.520326
G1 X121.321 Y119.901 E.00417
; LINE_WIDTH: 0.520776
G1 X121.178 Y119.781 E.00593
; LINE_WIDTH: 0.521226
G1 X121.034 Y119.661 E.00593
; LINE_WIDTH: 0.521686
G1 X120.891 Y119.542 E.00594
; LINE_WIDTH: 0.522136
G1 X120.747 Y119.422 E.00594
; LINE_WIDTH: 0.522596
G1 X120.603 Y119.302 E.00595
; LINE_WIDTH: 0.523046
G1 X120.46 Y119.183 E.00595
; LINE_WIDTH: 0.523196
G1 X120.412 Y119.143 E.00199
; LINE_WIDTH: 0.531646
G1 X120.383 Y119.022 E.00402
; LINE_WIDTH: 0.544336
G1 X120.34 Y118.841 E.00619
G1 X120.029 Y119.177 E.01524
; LINE_WIDTH: 0.523196
G1 X119.181 Y120.182 E.04192
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-40.285 J-37.094 E.30728
G1 X116.988 Y130.396 E.19623
G1 X117.374 Y130.721 E.01598
G1 X117.892 Y131.276 E.02402
G1 X118.327 Y131.907 E.02426
G3 X118.971 Y133.716 I-4.83 J2.74 E.06112
G1 X119.044 Y134.594 E.0279
G1 X118.98 Y135.356 E.02419
G1 X118.824 Y136.026 E.02179
G1 X118.548 Y136.753 E.02464
G1 X118.31 Y137.19 E.01573
G3 X130.249 Y144.067 I-20.137 J48.759 E.43747
G3 X142.39 Y157.025 I-31.645 J41.818 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.374 E1.81479
G1 X144.167 Y98.946 E.01355
G1 X143.7 Y98.968 E.0148
G1 X142.889 Y99.083 E.02594
G3 X141.707 Y99.088 I-1.329 J-164.64 E.03742
G1 X142.16 Y99.465 E.01865
G1 X142.238 Y99.677 E.00715
G1 X142.115 Y100.39 E.02292
; LINE_WIDTH: 0.562301
G1 X142.11 Y100.442 E.00179
; LINE_WIDTH: 0.604606
G1 X142.105 Y100.494 E.00193
; LINE_WIDTH: 0.646911
G1 X142.1 Y100.545 E.00207
; LINE_WIDTH: 0.689216
G1 X142.094 Y100.597 E.00221
; LINE_WIDTH: 0.737903
G1 X142.039 Y100.647 E.00343
; LINE_WIDTH: 0.78659
G1 X141.983 Y100.697 E.00367
; LINE_WIDTH: 0.835276
G1 X141.928 Y100.747 E.00391
G1 X142 Y100.798 E.00459
; LINE_WIDTH: 0.78659
G1 X142.072 Y100.848 E.00431
; LINE_WIDTH: 0.737903
G1 X142.144 Y100.899 E.00403
; LINE_WIDTH: 0.689216
G1 X142.172 Y100.964 E.00301
; LINE_WIDTH: 0.646911
G1 X142.199 Y101.028 E.00281
; LINE_WIDTH: 0.604606
G1 X142.227 Y101.093 E.00262
; LINE_WIDTH: 0.562301
G1 X142.255 Y101.158 E.00242
; LINE_WIDTH: 0.519996
G1 X142.659 Y103.378 E.07145
G3 X142.436 Y103.643 I-.247 J.018 E.01212
G2 X139.203 Y103.641 I-1.975 J678.785 E.10234
G1 X137.203 Y103.65 E.06332
G1 X137.067 Y103.604 E.00456
G1 X136.977 Y103.424 E.00638
G1 X136.977 Y103.09 E.01057
G1 X135.712 Y103.09 E.04003
G1 X135.693 Y103.515 E.01347
G1 X135.542 Y103.643 E.00629
G3 X134.162 Y103.645 I-.749 J-34.887 E.04368
G1 X132.162 Y103.639 E.06332
G1 X132.028 Y103.594 E.00446
G1 X131.94 Y103.447 E.00543
G3 X132.244 Y100.207 I313.448 J27.854 E.10304
; LINE_WIDTH: 0.569276
G1 X132.286 Y100.102 E.00393
; LINE_WIDTH: 0.618556
G1 X132.328 Y99.998 E.00429
; LINE_WIDTH: 0.6567
G1 X132.364 Y99.966 E.00193
; LINE_WIDTH: 0.694843
G1 X132.4 Y99.935 E.00205
; LINE_WIDTH: 0.732986
G1 X132.436 Y99.904 E.00217
; LINE_WIDTH: 0.736546
G1 X133.011 Y99.995 E.02664
G1 X133.158 Y99.198 E.0371
G1 X132.622 Y99.187 E.02453
; LINE_WIDTH: 0.729746
G1 X132.454 Y99.155 E.00778
; LINE_WIDTH: 0.692683
G1 X132.285 Y99.124 E.00736
; LINE_WIDTH: 0.65562
G1 X132.116 Y99.093 E.00695
; LINE_WIDTH: 0.618556
G1 X131.824 Y99.019 E.01148
; LINE_WIDTH: 0.569276
G1 X131.532 Y98.944 E.01051
; LINE_WIDTH: 0.519996
G1 X131.039 Y98.936 E.01562
G3 X122.124 Y116.671 I-50.9 J-14.478 E.6322
; LINE_WIDTH: 0.521596
G1 X121.281 Y117.68 E.04175
; LINE_WIDTH: 0.544336
G1 X120.976 Y118.079 E.01669
G1 X121.172 Y118.071 E.00652
; LINE_WIDTH: 0.531156
G1 X121.315 Y118.066 E.00461
; LINE_WIDTH: 0.521596
G1 X121.354 Y118.099 E.00164
; LINE_WIDTH: 0.521526
G1 X121.498 Y118.22 E.00598
; LINE_WIDTH: 0.521296
G1 X121.642 Y118.341 E.00598
; LINE_WIDTH: 0.521066
G1 X121.787 Y118.463 E.00598
; LINE_WIDTH: 0.520836
G1 X121.931 Y118.584 E.00597
; LINE_WIDTH: 0.520616
G1 X122.075 Y118.705 E.00597
; LINE_WIDTH: 0.520386
G1 X122.219 Y118.826 E.00597
; LINE_WIDTH: 0.520156
G1 X122.321 Y118.911 E.0042
; LINE_WIDTH: 0.519996
G1 X122.385 Y118.965 E.00265
G1 X122.476 Y119.042 E.00376
G1 X122.567 Y119.118 E.00376
G1 X122.658 Y119.194 E.00376
G1 X122.749 Y119.271 E.00376
G1 X122.84 Y119.347 E.00376
G1 X122.931 Y119.423 E.00376
G3 X123.004 Y119.517 I-.062 J.123 E.00389
G1 X123.056 Y119.631 E.00396
G1 X123.034 Y119.693 E.00211
; WIPE_START
M204 S10000
G1 X122.415 Y120.479 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X128.314 Y115.636 Z8.44 F36000
G1 X143.407 Y103.247 Z8.44
G1 Z8.04
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03442
G1 F7874.37
G1 X143.407 Y103.588 E.02221
; WIPE_START
G1 X143.407 Y103.247 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X136.025 Y101.307 Z8.44 F36000
G1 X131.485 Y100.115 Z8.44
G1 Z8.04
G1 E.4 F1800
; LINE_WIDTH: 1.05446
G1 F7719.68
G1 X131.605 Y99.738 E.02629
; WIPE_START
G1 X131.485 Y100.115 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X127.631 Y106.703 Z8.44 F36000
G1 X120.976 Y118.079 Z8.44
G1 Z8.04
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X120.34 Y118.841 E.03299
; WIPE_START
M204 S10000
G1 X120.976 Y118.079 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.184 Y122.358 Z8.44 F36000
G1 Z8.04
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X118.541 Y124.027 I-36.939 J-34.74 E.08944
G2 X120.005 Y127.052 I4.922 J-.516 E.1309
G3 X121.795 Y128.937 I-10.777 J12.024 E.09935
G1 X122.069 Y129.408 E.02081
G3 X121.245 Y134.184 I-4.188 J1.736 E.19522
G3 X121.111 Y135.983 I-9.105 J.226 E.06899
G3 X126.248 Y138.679 I-30.958 J65.241 E.22155
G3 X127.546 Y136.477 I5.292 J1.637 E.09845
G2 X129.335 Y134.592 I-10.778 J-12.025 E.09935
G2 X129.795 Y133.65 I-2.865 J-1.981 E.0402
G2 X128.459 Y128.937 I-4.822 J-1.178 E.1956
G3 X126.669 Y127.052 I10.777 J-12.024 E.09935
G3 X126.209 Y126.109 I2.865 J-1.981 E.0402
G3 X127.546 Y121.396 I4.822 J-1.178 E.1956
G2 X129.335 Y119.511 I-10.778 J-12.025 E.09935
G1 X129.609 Y119.04 E.02081
G2 X129.213 Y114.798 I-4.377 J-1.73 E.16878
G2 X127.415 Y112.825 I-15.575 J12.389 E.10201
G3 X126.169 Y114.808 I-40.555 J-24.078 E.08943
G1 X141.945 Y144.699 F36000
G1 F13446.283
G1 X141.945 Y142.356 E.08944
G3 X141.476 Y141.662 I1.563 J-1.562 E.03221
G3 X141.872 Y137.42 I4.377 J-1.73 E.16878
G1 X141.945 Y137.329 E.00446
G1 X141.945 Y127.275 E.38384
G3 X141.476 Y126.58 I1.563 J-1.562 E.03221
G3 X141.872 Y122.339 I4.377 J-1.73 E.16878
G1 X141.945 Y122.247 E.00446
G1 X141.945 Y112.194 E.38384
G3 X141.476 Y111.499 I1.563 J-1.562 E.03221
G3 X141.872 Y107.257 I4.377 J-1.73 E.16878
G1 X141.945 Y107.166 E.00446
G1 X141.945 Y105.865 E.04969
G1 X137.202 Y105.872 E.1811
G3 X136.345 Y105.708 I-.035 J-2.143 E.03356
G3 X134.599 Y105.869 I-1.259 J-4.109 E.06741
G2 X133.75 Y107.257 I2.553 J2.515 E.06269
G2 X135.086 Y111.97 I4.822 J1.179 E.1956
G3 X136.876 Y113.855 I-10.779 J12.026 E.09935
G3 X137.336 Y114.798 I-2.865 J1.982 E.04019
G3 X135.999 Y119.511 I-4.822 J1.178 E.1956
G2 X134.21 Y121.396 I10.777 J12.024 E.09935
G2 X133.75 Y122.339 I2.865 J1.981 E.04019
G2 X135.086 Y127.052 I4.822 J1.179 E.1956
G3 X136.876 Y128.937 I-10.779 J12.026 E.09935
G3 X137.336 Y129.879 I-2.865 J1.982 E.04019
G3 X135.999 Y134.592 I-4.822 J1.178 E.1956
G2 X134.21 Y136.477 I10.777 J12.024 E.09935
G2 X133.75 Y137.42 I2.865 J1.981 E.04019
G2 X135.086 Y142.133 I4.822 J1.179 E.1956
G3 X136.876 Y144.018 I-10.779 J12.026 E.09935
G3 X137.337 Y147.314 I-3.57 J2.179 E.13068
G3 X138.956 Y149.007 I-31.069 J31.331 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 8.2
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X138.265 Y148.284 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L51
M991 S0 P50 ;notify layer change


G17
G3 Z8.44 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X124.657 Y119.579
G1 Z8.2
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X124.678 Y119.756 E.00681
G1 X124.634 Y120.168 E.01581
G1 X124.442 Y120.685 E.02106
G3 X123.455 Y121.826 I-4.739 J-3.103 E.05778
G1 X123.022 Y122.058 E.01874
G1 X122.556 Y122.174 E.01832
G1 X122.045 Y122.167 E.01952
G1 X121.653 Y122.068 E.01545
G1 X121.176 Y121.813 E.02065
G3 X120.519 Y121.269 I43.579 J-53.217 E.03255
G3 X114.848 Y126.662 I-40.016 J-36.404 E.29907
G1 X118.03 Y129.023 E.15129
G1 X118.485 Y129.402 E.02259
G1 X119.155 Y130.103 E.03703
G1 X119.749 Y130.931 E.0389
G1 X120.215 Y131.838 E.03893
G3 X120.715 Y133.691 I-7.795 J3.097 E.07342
G1 X120.765 Y134.489 E.03055
G1 X120.698 Y135.507 E.03893
G1 X120.538 Y136.268 E.02969
G3 X130.505 Y142.1 I-22.399 J49.716 E.4417
G3 X142.443 Y154.069 I-32.642 J44.496 E.64798
G1 X142.443 Y105.365 E1.85948
G2 X136.157 Y105.375 I-.892 J1413.929 E.24001
G1 X132.157 Y105.363 E.15272
G1 X131.702 Y105.308 E.0175
G1 X131.034 Y105.003 E.02801
G1 X130.846 Y104.816 E.01015
G3 X123.507 Y117.701 I-51.219 J-20.639 E.56786
G1 X123.448 Y117.816 E.00491
G3 X124.126 Y118.396 I-5.367 J6.944 E.03408
G1 X124.436 Y118.815 E.01988
G1 X124.626 Y119.311 E.02029
G1 X124.647 Y119.489 E.00685
G1 X124.072 Y119.644 F36000
G1 F13446.369
G1 X124.089 Y119.847 E.00776
G1 X124.006 Y120.232 E.01504
G1 X123.828 Y120.562 E.0143
G1 X123.369 Y121.114 E.02742
G1 X123.04 Y121.399 E.01662
G3 X121.854 Y121.518 I-.716 J-1.161 E.04709
G1 X121.448 Y121.283 E.01794
G1 X120.447 Y120.444 E.04986
G1 X120.053 Y120.914 E.02339
G3 X113.893 Y126.683 I-40.871 J-37.467 E.32255
G1 X117.677 Y129.49 E.17988
G1 X118.108 Y129.85 E.02143
G1 X118.726 Y130.502 E.03429
G1 X119.266 Y131.263 E.03562
G1 X119.689 Y132.095 E.03564
G3 X120.135 Y133.793 I-7.013 J2.754 E.06718
G1 X120.18 Y134.527 E.02808
G1 X120.114 Y135.456 E.03557
G1 X119.932 Y136.287 E.03249
G1 X119.836 Y136.599 E.01244
G3 X130.932 Y143.157 I-21.594 J49.208 E.49331
G3 X142.92 Y155.769 I-32.176 J42.588 E.6673
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.639 E1.95133
G1 X142.434 Y104.781 E.02336
G2 X136.159 Y104.789 I-2.1 J840.14 E.23959
G1 X132.159 Y104.777 E.15272
G1 X131.84 Y104.738 E.01224
G1 X131.373 Y104.525 E.01959
G1 X131.028 Y104.17 E.01892
G1 X130.818 Y103.644 E.02161
G1 X130.805 Y103.301 E.01312
G3 X124.897 Y114.773 I-50.791 J-18.903 E.49384
G3 X122.603 Y117.872 I-31.369 J-20.818 E.14724
G1 X123.604 Y118.71 E.04986
G1 X123.923 Y119.098 E.01914
G1 X124.056 Y119.445 E.01419
G1 X124.065 Y119.555 E.00422
G1 X123.494 Y119.659 F36000
G1 F13446.369
G1 X123.504 Y119.694 E.00137
G1 X123.457 Y120.028 E.01287
G1 X123.324 Y120.256 E.01009
G1 X122.921 Y120.738 E.024
G1 X122.652 Y120.944 E.01291
G1 X122.357 Y121.015 E.01158
G1 X122.056 Y120.969 E.01165
G1 X121.824 Y120.834 E.01024
G1 X120.374 Y119.619 E.07222
G1 X119.604 Y120.538 E.04575
G3 X112.93 Y126.698 I-40.062 J-36.711 E.34719
G1 X117.323 Y129.957 E.20887
G1 X117.73 Y130.298 E.02027
G1 X118.297 Y130.9 E.03155
G1 X118.783 Y131.594 E.03234
G3 X119.514 Y133.608 I-5.356 J3.083 E.08222
G1 X119.596 Y134.561 E.03649
G1 X119.531 Y135.406 E.03236
G1 X119.362 Y136.152 E.02921
G1 X119.08 Y136.91 E.03088
G3 X130.579 Y143.624 I-20.386 J48.121 E.50976
G3 X142.647 Y156.415 I-31.78 J42.073 E.67456
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y105.065 E1.95363
G1 X143.615 Y104.665 E.01527
; LINE_WIDTH: 0.659514
G1 F12371.44
G1 X143.595 Y104.487 E.0073
; LINE_WIDTH: 0.699031
G1 F11761.738
G1 X143.575 Y104.309 E.00776
; LINE_WIDTH: 0.738549
G1 F11167.41
G1 X143.555 Y104.131 E.00822
; LINE_WIDTH: 0.778066
G1 F10588.521
G1 X143.536 Y103.953 E.00869
; LINE_WIDTH: 0.820788
G1 F10013.33
G1 X143.514 Y103.892 E.00331
; LINE_WIDTH: 0.86351
G1 F9497.412
G1 X143.493 Y103.831 E.00349
; LINE_WIDTH: 0.906231
G1 F9032.052
G1 X143.471 Y103.77 E.00367
; LINE_WIDTH: 0.948953
G1 F8610.165
G1 X143.45 Y103.709 E.00385
; LINE_WIDTH: 0.991675
G1 F8225.932
G1 X143.429 Y103.648 E.00403
; LINE_WIDTH: 1.0344
G1 F7874.528
G1 X143.407 Y103.588 E.00421
G1 X143.369 Y103.632 E.00385
; LINE_WIDTH: 0.991675
G1 F8225.932
G1 X143.33 Y103.676 E.00368
; LINE_WIDTH: 0.948953
G1 F8610.165
G1 X143.291 Y103.721 E.00352
; LINE_WIDTH: 0.906231
G1 F9032.052
G1 X143.252 Y103.765 E.00335
; LINE_WIDTH: 0.86351
G1 F9497.412
G1 X143.213 Y103.81 E.00319
; LINE_WIDTH: 0.820788
G1 F10013.33
G1 X143.174 Y103.854 E.00302
; LINE_WIDTH: 0.778066
G1 F10588.521
G1 X143.072 Y103.921 E.00594
; LINE_WIDTH: 0.738549
G1 F11182.705
G1 X142.97 Y103.989 E.00563
; LINE_WIDTH: 0.699031
G1 F11587.834
G1 X142.867 Y104.056 E.00531
; LINE_WIDTH: 0.659514
G1 F12000.124
G1 X142.765 Y104.123 E.00499
; LINE_WIDTH: 0.619996
G1 F13173.478
G1 X142.435 Y104.196 E.01289
G1 F13446.369
G1 X142.035 Y104.196 E.01527
G1 X136.16 Y104.204 E.22429
G1 X132.16 Y104.191 E.15272
G1 X131.979 Y104.169 E.00699
G1 X131.699 Y104.038 E.01181
G1 X131.515 Y103.845 E.01017
G1 X131.393 Y103.531 E.01287
G3 X131.446 Y102.729 I8.684 J.174 E.0307
G1 X131.523 Y101.932 E.03054
G1 X131.561 Y101.534 E.01527
G1 F13260.888
G1 X131.6 Y101.136 E.01527
G1 F11872.098
G1 X131.638 Y100.738 E.01527
; LINE_WIDTH: 0.665112
G1 F10560.107
G1 X131.622 Y100.671 E.00281
; LINE_WIDTH: 0.710227
G1 F10343.582
G1 X131.605 Y100.605 E.00301
; LINE_WIDTH: 0.755343
G1 F10129.276
G1 X131.589 Y100.539 E.00321
; LINE_WIDTH: 0.800458
G1 F9917.207
G1 X131.573 Y100.472 E.00341
; LINE_WIDTH: 0.845574
G1 F9707.389
G1 X131.557 Y100.406 E.00362
; LINE_WIDTH: 0.89069
G1 F9195.97
G1 X131.54 Y100.339 E.00382
; LINE_WIDTH: 0.935805
G1 F8735.743
G1 X131.524 Y100.273 E.00402
; LINE_WIDTH: 0.980921
G1 F8319.385
G1 X131.508 Y100.207 E.00422
; LINE_WIDTH: 1.02604
G1 F7940.91
G1 X131.491 Y100.14 E.00442
G1 X131.445 Y100.199 E.00484
; LINE_WIDTH: 0.97679
G1 F8355.848
G1 X131.399 Y100.258 E.0046
; LINE_WIDTH: 0.927544
G1 F8816.541
G1 X131.352 Y100.316 E.00436
; LINE_WIDTH: 0.878298
G1 F9330.998
G1 X131.306 Y100.375 E.00412
; LINE_WIDTH: 0.829051
G1 F9909.213
G1 X131.259 Y100.434 E.00388
; LINE_WIDTH: 0.779805
G1 F10141.408
G1 X131.213 Y100.492 E.00364
; LINE_WIDTH: 0.730559
G1 F10376.312
G1 X131.166 Y100.551 E.0034
; LINE_WIDTH: 0.681313
G1 F10613.886
G1 X131.12 Y100.61 E.00316
; LINE_WIDTH: 0.632066
G1 F10678.248
G1 X131.108 Y100.626 E.00078
; LINE_WIDTH: 0.619996
G1 F11997.343
G1 X130.979 Y101.005 E.01527
G1 F13393.237
G1 X130.85 Y101.383 E.01527
G1 F13446.369
G1 X130.721 Y101.762 E.01527
G1 X130.371 Y102.785 E.04127
G3 X124.03 Y114.993 I-50.418 J-18.434 E.52667
G3 X121.778 Y117.944 I-26.499 J-17.888 E.1418
G1 X123.228 Y119.159 E.07222
G1 X123.418 Y119.395 E.01155
G1 X123.469 Y119.573 E.00707
; WIPE_START
G1 X123.504 Y119.694 E-.04785
G1 X123.457 Y120.028 E-.1281
G1 X123.324 Y120.256 E-.1004
G1 X123.149 Y120.465 E-.10365
; WIPE_END
G1 E-.02 F1800
G1 X128.964 Y115.521 Z8.6 F36000
G1 X143.407 Y103.241 Z8.6
G1 Z8.2
G1 E.4 F1800
; LINE_WIDTH: 1.03181
G1 F7894.975
G1 X143.425 Y102.813 E.02783
; LINE_WIDTH: 0.995961
G1 F8189.264
G1 X143.443 Y102.385 E.02683
; LINE_WIDTH: 0.960116
G1 F8506.341
G1 X143.464 Y101.886 E.03018
; LINE_WIDTH: 0.918249
G1 F8909.253
G1 X143.486 Y101.386 E.02881
; LINE_WIDTH: 0.876381
G1 F9352.233
G1 X143.507 Y100.887 E.02745
; LINE_WIDTH: 0.834514
G1 F9841.567
G1 X143.528 Y100.387 E.02608
; LINE_WIDTH: 0.792646
G1 F10384.936
G1 X143.543 Y99.622 E.03783
; LINE_WIDTH: 0.763236
G1 F10803.953
G3 X142.862 Y99.712 I-.989 J-4.888 E.03264
; LINE_WIDTH: 0.792646
G1 F10384.936
G1 X142.775 Y100.323 E.03053
G1 X142.776 Y100.455 E.00653
; LINE_WIDTH: 0.828486
G1 F9916.263
G1 X142.871 Y100.873 E.02216
; LINE_WIDTH: 0.864326
G1 F9488.067
G1 X142.965 Y101.29 E.02316
; LINE_WIDTH: 0.906196
G1 F9032.414
G1 X143.076 Y101.778 E.02842
; LINE_WIDTH: 0.948066
G1 F8618.521
G1 X143.186 Y102.265 E.02978
; LINE_WIDTH: 0.989936
G1 F8240.896
G1 X143.296 Y102.753 E.03115
; LINE_WIDTH: 1.03181
G1 F7894.975
G1 X143.387 Y103.153 E.02666
; WIPE_START
G1 X143.425 Y102.813 E-.12994
G1 X143.443 Y102.385 E-.16264
G1 X143.453 Y102.156 E-.08742
; WIPE_END
G1 E-.02 F1800
G1 X137.659 Y107.124 Z8.6 F36000
G1 X122.953 Y119.738 Z8.6
G1 Z8.2
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X122.901 Y119.901 E.00542
G1 X122.497 Y120.383 E.0199
G1 X122.351 Y120.462 E.00526
G1 X122.254 Y120.433 E.00321
G3 X122.161 Y120.396 I-.003 J-.125 E.00324
G1 X122.085 Y120.332 E.00316
G1 X122.008 Y120.267 E.00316
G1 X121.932 Y120.203 E.00316
G1 X121.855 Y120.139 E.00316
G1 X121.778 Y120.075 E.00316
G1 X121.702 Y120.011 E.00316
G1 X121.648 Y119.965 E.00223
; LINE_WIDTH: 0.520326
G1 X121.534 Y119.871 E.00468
; LINE_WIDTH: 0.520776
G1 F3150
G1 X121.373 Y119.736 E.00665
; LINE_WIDTH: 0.521226
G1 F3300
G1 X121.212 Y119.602 E.00665
; LINE_WIDTH: 0.521686
G1 F3450
G1 X121.051 Y119.468 E.00666
; LINE_WIDTH: 0.522136
G1 F3600
G1 X120.89 Y119.334 E.00667
; LINE_WIDTH: 0.522596
G1 X120.729 Y119.199 E.00667
; LINE_WIDTH: 0.523046
G1 X120.568 Y119.065 E.00668
; LINE_WIDTH: 0.523196
G1 X120.515 Y119.02 E.00223
; LINE_WIDTH: 0.531646
G1 X120.486 Y118.899 E.00402
; LINE_WIDTH: 0.544336
G1 X120.443 Y118.718 E.00619
G1 X120.131 Y119.054 E.01524
; LINE_WIDTH: 0.523196
G1 X119.181 Y120.182 E.04702
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.855 J-35.521 E.30731
G1 X116.99 Y130.398 E.19631
G1 X117.374 Y130.721 E.0159
G1 X117.892 Y131.276 E.02402
G1 X118.327 Y131.907 E.02425
G3 X118.972 Y133.72 I-4.826 J2.739 E.06123
G1 X119.044 Y134.594 E.02778
G1 X118.98 Y135.358 E.02427
G1 X118.824 Y136.024 E.02166
G1 X118.547 Y136.754 E.0247
G1 X118.31 Y137.19 E.01573
G3 X130.249 Y144.067 I-20.138 J48.76 E.43748
G3 X142.39 Y157.025 I-31.645 J41.817 E.56491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.372 E1.81484
G1 X144.167 Y98.946 E.0135
G1 X143.702 Y98.969 E.01476
G1 X142.888 Y99.085 E.02603
G3 X141.71 Y99.09 I-1.311 J-158.587 E.03731
G1 X142.16 Y99.465 E.01854
G1 X142.238 Y99.677 E.00714
G1 X142.115 Y100.392 E.02299
G1 X142.659 Y103.378 E.09609
G3 X142.436 Y103.643 I-.247 J.018 E.01212
M73 P78 R4
G2 X136.162 Y103.651 I-2.101 J839.111 E.19864
G1 X132.162 Y103.639 E.12664
G1 X132.028 Y103.594 E.00446
G1 X131.94 Y103.447 E.00543
G3 X132.244 Y100.207 I313.919 J27.899 E.10304
; LINE_WIDTH: 0.568771
G1 X132.286 Y100.103 E.00392
; LINE_WIDTH: 0.617546
G1 X132.328 Y99.998 E.00427
; LINE_WIDTH: 0.655706
G1 X132.364 Y99.967 E.00194
; LINE_WIDTH: 0.693866
G1 X132.4 Y99.936 E.00205
; LINE_WIDTH: 0.732026
G1 X132.436 Y99.904 E.00217
; LINE_WIDTH: 0.735426
G1 X133.009 Y99.995 E.02651
G1 X133.156 Y99.2 E.03698
G1 X132.623 Y99.188 E.0244
; LINE_WIDTH: 0.728756
G1 X132.454 Y99.157 E.00778
; LINE_WIDTH: 0.691686
G1 X132.285 Y99.125 E.00736
; LINE_WIDTH: 0.654616
G1 X132.116 Y99.094 E.00694
; LINE_WIDTH: 0.617546
G1 X131.824 Y99.019 E.01147
; LINE_WIDTH: 0.568771
G1 X131.532 Y98.944 E.01051
; LINE_WIDTH: 0.519996
G1 X131.039 Y98.936 E.0156
G3 X122.124 Y116.672 I-51.014 J-14.535 E.63218
; LINE_WIDTH: 0.521596
G1 X121.178 Y117.802 E.04683
; LINE_WIDTH: 0.544336
G1 X120.874 Y118.202 E.01669
G1 X121.07 Y118.194 E.00652
; LINE_WIDTH: 0.531156
G1 X121.212 Y118.188 E.00461
; LINE_WIDTH: 0.521596
G1 X121.256 Y118.226 E.00184
; LINE_WIDTH: 0.521526
G1 X121.418 Y118.362 E.00671
; LINE_WIDTH: 0.521296
G1 X121.58 Y118.498 E.00671
; LINE_WIDTH: 0.521066
G1 X121.741 Y118.633 E.0067
; LINE_WIDTH: 0.520836
G1 F3450
G1 X121.903 Y118.769 E.0067
; LINE_WIDTH: 0.520616
G1 F3300
G1 X122.065 Y118.905 E.0067
; LINE_WIDTH: 0.520386
G1 F3150
G1 X122.227 Y119.041 E.00669
; LINE_WIDTH: 0.520156
G1 F3600
G1 X122.341 Y119.137 E.00472
; LINE_WIDTH: 0.519996
G1 X122.395 Y119.182 E.00223
G1 X122.472 Y119.247 E.00317
G1 X122.548 Y119.311 E.00317
G1 X122.625 Y119.375 E.00317
G1 X122.702 Y119.439 E.00317
G1 X122.778 Y119.504 E.00317
G1 X122.855 Y119.568 E.00317
G3 X122.908 Y119.65 I-.072 J.104 E.00317
G1 X122.911 Y119.658 E.00027
; WIPE_START
M204 S10000
G1 X122.901 Y119.901 E-.09242
G1 X122.497 Y120.383 E-.2389
G1 X122.384 Y120.444 E-.04868
; WIPE_END
G1 E-.02 F1800
G1 X128.292 Y115.611 Z8.6 F36000
G1 X143.407 Y103.247 Z8.6
G1 Z8.2
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.0344
G1 F7874.528
G1 X143.407 Y103.588 E.02221
; WIPE_START
G1 X143.407 Y103.247 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X136.025 Y101.307 Z8.6 F36000
G1 X131.485 Y100.115 Z8.6
G1 Z8.2
G1 E.4 F1800
; LINE_WIDTH: 1.0543
G1 F7720.892
G1 X131.605 Y99.738 E.02624
; WIPE_START
G1 X131.485 Y100.115 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X127.623 Y106.698 Z8.6 F36000
G1 X120.874 Y118.202 Z8.6
G1 Z8.2
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X120.443 Y118.718 E.02235
; WIPE_START
M204 S10000
G1 X120.874 Y118.202 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.16 Y122.382 Z8.6 F36000
G1 Z8.2
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X118.513 Y124.048 I-25.728 J-23.788 E.08945
G2 X119.328 Y126.109 I5.625 J-1.032 E.08515
G2 X121.147 Y127.994 I9.633 J-7.477 E.10022
G3 X122.209 Y129.408 I-3.315 J3.594 E.06787
G3 X121.237 Y134.065 I-4.355 J1.522 E.19082
G3 X121.116 Y135.986 I-7.725 J.476 E.07369
G3 X126.257 Y138.684 I-30.026 J63.465 E.22173
G1 X126.353 Y138.362 E.01282
G1 X126.869 Y137.42 E.04102
G3 X128.688 Y135.535 I9.633 J7.477 E.10022
G2 X129.749 Y134.121 I-3.315 J-3.594 E.06787
G2 X129.136 Y129.879 I-4.534 J-1.51 E.16968
G2 X127.316 Y127.994 I-9.633 J7.477 E.10022
G3 X126.255 Y126.58 I3.315 J-3.594 E.06787
G3 X126.869 Y122.339 I4.534 J-1.51 E.16968
G3 X128.688 Y120.453 I9.633 J7.477 E.10022
G2 X129.749 Y119.04 I-3.315 J-3.594 E.06787
G2 X129.136 Y114.798 I-4.534 J-1.51 E.16968
G2 X127.347 Y112.941 I-9.486 J7.35 E.09863
G3 X126.098 Y114.923 I-47.291 J-28.431 E.08944
G1 X141.945 Y144.947 F36000
G1 F13446.283
G1 X141.945 Y142.605 E.08944
G3 X141.336 Y141.662 I4.354 J-3.48 E.04293
G3 X141.945 Y137.428 I4.529 J-1.509 E.16932
G1 X141.945 Y127.523 E.37815
G3 X141.336 Y126.58 I4.353 J-3.479 E.04293
G3 X141.945 Y122.347 I4.529 J-1.509 E.16932
G1 X141.945 Y112.442 E.37815
G3 X141.336 Y111.499 I4.353 J-3.479 E.04293
G3 X141.945 Y107.265 I4.529 J-1.509 E.16932
G1 X141.945 Y105.864 E.05349
G1 X134.391 Y105.861 E.28843
G2 X133.796 Y106.786 I2.638 J2.351 E.04215
G2 X134.409 Y111.028 I4.533 J1.51 E.16968
G2 X136.229 Y112.913 I9.634 J-7.477 E.10022
G3 X137.29 Y114.327 I-3.315 J3.594 E.06787
G3 X136.676 Y118.568 I-4.534 J1.51 E.16968
G3 X134.857 Y120.453 I-9.633 J-7.477 E.10022
G2 X133.796 Y121.867 I3.315 J3.594 E.06787
G2 X134.409 Y126.109 I4.533 J1.51 E.16968
G2 X136.229 Y127.994 I9.634 J-7.477 E.10022
G3 X137.29 Y129.408 I-3.315 J3.594 E.06787
G3 X136.676 Y133.65 I-4.534 J1.51 E.16968
G3 X134.857 Y135.535 I-9.633 J-7.477 E.10022
G2 X133.796 Y136.949 I3.315 J3.594 E.06787
G2 X134.409 Y141.19 I4.533 J1.51 E.16968
G2 X136.229 Y143.075 I9.634 J-7.477 E.10022
G3 X137.335 Y147.308 I-2.418 J2.893 E.17821
G3 X138.952 Y149.003 I-34.134 J34.181 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 8.36
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X138.262 Y148.28 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L52
M991 S0 P51 ;notify layer change


G17
G3 Z8.6 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X124.299 Y120.861
G1 Z8.36
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X124.271 Y120.924 E.00263
G3 X123.562 Y121.7 I-2.826 J-1.869 E.04031
G1 X123.109 Y121.942 E.0196
G1 X122.644 Y122.053 E.01825
G1 X122.192 Y122.051 E.01724
G1 X121.831 Y121.971 E.01414
G1 X121.371 Y121.754 E.01942
G3 X120.622 Y121.147 I12.892 J-16.649 E.0368
G3 X114.848 Y126.662 I-39.373 J-35.439 E.30516
G1 X118.015 Y129.012 E.15055
G1 X118.485 Y129.403 E.02334
G1 X119.155 Y130.103 E.03702
G1 X119.749 Y130.931 E.0389
G1 X120.215 Y131.838 E.03892
G1 X120.527 Y132.755 E.03699
G3 X120.765 Y134.499 I-7.363 J1.892 E.06736
G1 X120.698 Y135.504 E.03845
G1 X120.538 Y136.268 E.0298
G3 X130.504 Y142.098 I-21.897 J48.857 E.44166
G3 X142.443 Y154.069 I-32.641 J44.497 E.64805
G1 X142.443 Y105.365 E1.85948
G2 X136.157 Y105.375 I-.887 J1415.494 E.24001
G1 X132.157 Y105.363 E.15272
G1 X131.717 Y105.311 E.01692
G1 X131.251 Y105.137 E.01897
G1 X130.858 Y104.786 E.02011
G3 X123.326 Y117.922 I-50.8 J-20.403 E.57999
G3 X124.096 Y118.599 I-3.914 J5.231 E.03919
G1 X124.387 Y119.043 E.02029
G1 X124.535 Y119.484 E.01776
G1 X124.57 Y120.016 E.02035
G1 X124.495 Y120.432 E.01612
G1 X124.337 Y120.779 E.01459
G1 X123.749 Y120.643 F36000
G1 F13446.369
G1 X123.422 Y121.047 E.01984
G1 X123.088 Y121.308 E.01618
G1 X122.674 Y121.457 E.01681
G1 X122.263 Y121.469 E.0157
G1 X122.01 Y121.414 E.00989
G1 X121.55 Y121.16 E.02002
G1 X120.55 Y120.322 E.04986
G1 X120.053 Y120.914 E.02951
G3 X113.893 Y126.683 I-40.872 J-37.468 E.32254
G1 X117.677 Y129.49 E.17987
G1 X118.108 Y129.851 E.02146
G1 X118.726 Y130.502 E.03428
G1 X119.266 Y131.263 E.03562
G1 X119.689 Y132.095 E.03563
G1 X119.969 Y132.932 E.0337
G1 X120.087 Y133.494 E.02194
G1 X120.18 Y134.518 E.03925
G1 X120.115 Y135.454 E.03581
G1 X119.932 Y136.287 E.03256
G1 X119.835 Y136.599 E.01247
G3 X130.932 Y143.157 I-21.317 J48.739 E.49334
G3 X142.92 Y155.769 I-32.372 J42.774 E.66727
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.639 E1.95133
G1 X142.434 Y104.781 E.02336
G2 X136.159 Y104.789 I-2.1 J840.133 E.23959
G1 X132.159 Y104.777 E.15272
G3 X131.525 Y104.619 I.004 J-1.364 E.02518
G1 X131.036 Y104.182 E.02504
G1 X130.817 Y103.636 E.02244
G1 X130.805 Y103.298 E.01292
G3 X124.897 Y114.773 I-50.633 J-18.814 E.49394
G3 X122.501 Y117.994 I-30.988 J-20.548 E.15335
G1 X123.501 Y118.833 E.04986
G1 X123.654 Y118.983 E.00817
G1 X123.919 Y119.446 E.02038
G1 X123.989 Y119.878 E.01671
G1 X123.933 Y120.265 E.01493
G1 X123.796 Y120.567 E.01266
G1 X123.251 Y120.312 F36000
G1 F13446.369
G1 X123.023 Y120.615 E.01447
G1 X122.804 Y120.796 E.01085
G1 X122.513 Y120.889 E.01165
G1 X122.189 Y120.856 E.01246
G1 X121.927 Y120.712 E.01143
G1 X120.477 Y119.497 E.07222
G1 X119.604 Y120.538 E.05187
G3 X112.93 Y126.698 I-40.062 J-36.712 E.34718
G1 X117.323 Y129.957 E.20885
G1 X117.731 Y130.299 E.0203
G1 X118.297 Y130.9 E.03154
G1 X118.783 Y131.594 E.03235
G1 X119.162 Y132.352 E.03235
G1 X119.411 Y133.109 E.03042
G1 X119.533 Y133.721 E.02384
G1 X119.595 Y134.551 E.03176
G1 X119.531 Y135.403 E.03264
G1 X119.362 Y136.151 E.02929
G1 X119.08 Y136.91 E.03091
G3 X130.579 Y143.624 I-20.41 J48.161 E.50975
G3 X142.647 Y156.415 I-31.962 J42.245 E.67453
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y105.065 E1.95363
G1 X143.615 Y104.665 E.01527
; LINE_WIDTH: 0.659514
G1 F12371.474
G1 X143.595 Y104.487 E.0073
; LINE_WIDTH: 0.699031
G1 F11761.771
G1 X143.575 Y104.309 E.00776
; LINE_WIDTH: 0.738549
G1 F11167.442
G1 X143.555 Y104.131 E.00822
; LINE_WIDTH: 0.778066
G1 F10588.521
G1 X143.536 Y103.953 E.00869
; LINE_WIDTH: 0.820795
G1 F10013.246
G1 X143.514 Y103.892 E.00331
; LINE_WIDTH: 0.863523
G1 F9497.259
G1 X143.493 Y103.831 E.00349
; LINE_WIDTH: 0.906251
G1 F9031.845
G1 X143.471 Y103.77 E.00367
; LINE_WIDTH: 0.94898
G1 F8609.914
G1 X143.45 Y103.709 E.00385
; LINE_WIDTH: 0.991708
G1 F8225.646
G1 X143.429 Y103.648 E.00403
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X143.407 Y103.588 E.00421
G1 X143.369 Y103.632 E.00385
; LINE_WIDTH: 0.991708
G1 F8225.646
G1 X143.33 Y103.676 E.00368
; LINE_WIDTH: 0.94898
G1 F8609.914
G1 X143.291 Y103.721 E.00352
; LINE_WIDTH: 0.906251
G1 F9031.845
G1 X143.252 Y103.765 E.00335
; LINE_WIDTH: 0.863523
G1 F9497.259
G1 X143.213 Y103.81 E.00319
; LINE_WIDTH: 0.820795
G1 F10013.246
G1 X143.174 Y103.854 E.00302
; LINE_WIDTH: 0.778066
G1 F10588.521
G1 X143.072 Y103.921 E.00594
; LINE_WIDTH: 0.738549
G1 F11182.705
G1 X142.97 Y103.989 E.00563
; LINE_WIDTH: 0.699031
G1 F11587.834
G1 X142.867 Y104.056 E.00531
; LINE_WIDTH: 0.659514
G1 F12000.153
G1 X142.765 Y104.123 E.00499
; LINE_WIDTH: 0.619996
G1 F13173.473
G1 X142.435 Y104.196 E.01289
G1 F13446.369
G1 X142.035 Y104.196 E.01527
G1 X136.16 Y104.204 E.22429
G1 X132.16 Y104.191 E.15272
G1 X131.814 Y104.109 E.01361
G1 X131.513 Y103.842 E.01535
G1 X131.393 Y103.529 E.01281
G3 X131.448 Y102.712 I38.108 J2.166 E.03126
G1 X131.501 Y102.165 E.02096
G1 X131.539 Y101.767 E.01527
G1 F12788.082
G1 X131.577 Y101.369 E.01527
; LINE_WIDTH: 0.669038
G1 F11424.965
G1 X131.567 Y101.216 E.00636
; LINE_WIDTH: 0.718079
G1 F10921.201
G1 X131.557 Y101.062 E.00686
; LINE_WIDTH: 0.76712
G1 F10428.796
G1 X131.548 Y100.908 E.00735
; LINE_WIDTH: 0.816161
G1 F9947.75
G1 X131.538 Y100.755 E.00784
; LINE_WIDTH: 0.865202
G1 F9478.06
G1 X131.528 Y100.601 E.00833
; LINE_WIDTH: 0.914244
G1 F8949.805
G1 X131.518 Y100.448 E.00882
; LINE_WIDTH: 0.963285
G1 F8477.326
G1 X131.508 Y100.294 E.00932
; LINE_WIDTH: 1.01233
G1 F8052.23
G1 X131.498 Y100.141 E.00981
; LINE_WIDTH: 1.03836
G1 F7843.469
G1 X131.49 Y100.103 E.00256
G1 X131.466 Y100.134 E.00258
; LINE_WIDTH: 1.01233
G1 F8052.23
G1 X131.397 Y100.272 E.00981
; LINE_WIDTH: 0.963285
G1 F8477.326
G1 X131.327 Y100.409 E.00932
; LINE_WIDTH: 0.914244
G1 F8949.805
G1 X131.258 Y100.546 E.00882
; LINE_WIDTH: 0.865202
G1 F9478.06
G1 X131.189 Y100.684 E.00833
; LINE_WIDTH: 0.816161
G1 F10072.587
G1 X131.12 Y100.821 E.00784
; LINE_WIDTH: 0.76712
G1 F10556.6
G1 X131.05 Y100.958 E.00735
; LINE_WIDTH: 0.718079
G1 F11051.971
G1 X130.981 Y101.096 E.00686
; LINE_WIDTH: 0.669038
G1 F11558.685
G1 X130.912 Y101.233 E.00636
; LINE_WIDTH: 0.619996
G1 F12929.532
G1 X130.779 Y101.61 E.01527
G1 F13446.369
G1 X130.646 Y101.988 E.01527
G1 X130.263 Y103.082 E.04426
G3 X124.404 Y114.456 I-50.151 J-18.638 E.48964
G3 X121.676 Y118.067 I-29.675 J-19.587 E.17292
G1 X123.125 Y119.282 E.07222
G1 X123.328 Y119.545 E.0127
G1 X123.404 Y119.878 E.01305
G1 X123.333 Y120.202 E.01265
G1 X123.305 Y120.24 E.0018
; WIPE_START
G1 X123.023 Y120.615 E-.17821
G1 X122.804 Y120.796 E-.10795
G1 X122.569 Y120.871 E-.09383
; WIPE_END
G1 E-.02 F1800
G1 X128.396 Y115.941 Z8.76 F36000
G1 X143.407 Y103.241 Z8.76
G1 Z8.36
G1 E.4 F1800
; LINE_WIDTH: 1.03185
G1 F7894.658
G1 X143.425 Y102.813 E.02783
; LINE_WIDTH: 0.996001
G1 F8188.923
G1 X143.443 Y102.385 E.02683
; LINE_WIDTH: 0.960156
G1 F8505.974
G1 X143.464 Y101.886 E.03018
; LINE_WIDTH: 0.918281
G1 F8908.925
G1 X143.486 Y101.386 E.02881
; LINE_WIDTH: 0.876406
G1 F9351.955
G1 X143.507 Y100.887 E.02745
; LINE_WIDTH: 0.834531
G1 F9841.352
G1 X143.528 Y100.387 E.02608
; LINE_WIDTH: 0.792656
G1 F10384.799
G1 X143.543 Y99.623 E.03778
; LINE_WIDTH: 0.763386
G1 F10801.729
G3 X142.862 Y99.714 I-.991 J-4.843 E.03266
; LINE_WIDTH: 0.792656
G1 F10384.799
G1 X142.775 Y100.323 E.03043
G1 X142.776 Y100.455 E.00653
; LINE_WIDTH: 0.828496
G1 F9916.138
G1 X142.871 Y100.873 E.02216
; LINE_WIDTH: 0.864336
G1 F9487.952
G1 X142.965 Y101.29 E.02316
; LINE_WIDTH: 0.906214
G1 F9032.233
G1 X143.075 Y101.778 E.02842
; LINE_WIDTH: 0.948091
G1 F8618.284
G1 X143.186 Y102.265 E.02978
; LINE_WIDTH: 0.989969
G1 F8240.616
G1 X143.296 Y102.753 E.03115
; LINE_WIDTH: 1.03185
G1 F7894.658
G1 X143.387 Y103.153 E.02666
; WIPE_START
G1 X143.425 Y102.813 E-.12994
G1 X143.443 Y102.385 E-.16264
G1 X143.453 Y102.156 E-.08742
; WIPE_END
G1 E-.02 F1800
G1 X137.681 Y107.149 Z8.76 F36000
G1 X122.798 Y120.023 Z8.76
G1 Z8.36
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X122.506 Y120.326 E.01333
G1 X122.358 Y120.33 E.00469
G3 X121.873 Y119.946 I5.617 J-7.577 E.01958
; LINE_WIDTH: 0.523196
G1 X120.617 Y118.898 E.05213
; LINE_WIDTH: 0.544336
G1 X120.546 Y118.596 E.01032
G1 X120.234 Y118.931 E.01524
; LINE_WIDTH: 0.523196
G1 X119.181 Y120.183 E.05213
; LINE_WIDTH: 0.519996
G3 X112.01 Y126.704 I-38.856 J-35.522 E.30731
G1 X116.989 Y130.398 E.1963
G1 X117.374 Y130.722 E.01593
G1 X117.892 Y131.276 E.02401
G1 X118.327 Y131.907 E.02426
G1 X118.665 Y132.594 E.02426
G1 X118.884 Y133.276 E.02266
G1 X118.988 Y133.817 E.01746
G1 X119.043 Y134.581 E.02426
G1 X118.98 Y135.356 E.02459
G1 X118.824 Y136.024 E.02172
G1 X118.548 Y136.754 E.02471
G1 X118.31 Y137.19 E.01573
G3 X130.249 Y144.067 I-19.956 J48.443 E.43749
G3 X142.39 Y157.025 I-31.645 J41.817 E.56491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.37 E1.8149
G1 X144.167 Y98.946 E.01343
G1 X143.703 Y98.969 E.01471
G1 X142.887 Y99.087 E.02611
G3 X141.712 Y99.093 I-1.307 J-155.223 E.0372
G1 X142.16 Y99.465 E.01844
G1 X142.238 Y99.677 E.00714
G1 X142.115 Y100.392 E.02299
G1 X142.659 Y103.378 E.09609
G3 X142.436 Y103.643 I-.247 J.018 E.01212
G2 X136.157 Y103.651 I-2.102 J842.591 E.1988
G1 X132.157 Y103.639 E.12664
G1 X132.057 Y103.612 E.00326
G1 X131.94 Y103.446 E.00644
G3 X132.052 Y102.203 I63.146 J5.103 E.03952
G1 X132.244 Y100.213 E.06332
; LINE_WIDTH: 0.568296
G1 X132.286 Y100.106 E.004
; LINE_WIDTH: 0.616596
G1 X132.328 Y99.999 E.00436
; LINE_WIDTH: 0.654753
G1 X132.364 Y99.968 E.00194
; LINE_WIDTH: 0.69291
G1 X132.4 Y99.936 E.00206
; LINE_WIDTH: 0.731066
G1 X132.437 Y99.905 E.00218
; LINE_WIDTH: 0.734316
G1 X133.007 Y99.996 E.02638
G1 X133.155 Y99.201 E.03686
G1 X132.623 Y99.19 E.02427
; LINE_WIDTH: 0.727756
G1 X132.454 Y99.158 E.00777
; LINE_WIDTH: 0.690703
G1 X132.285 Y99.126 E.00735
; LINE_WIDTH: 0.65365
G1 X132.116 Y99.095 E.00694
; LINE_WIDTH: 0.616596
G1 X131.824 Y99.02 E.01146
; LINE_WIDTH: 0.568296
G1 X131.531 Y98.945 E.01051
; LINE_WIDTH: 0.519996
G1 X131.038 Y98.936 E.01563
G3 X122.124 Y116.672 I-50.988 J-14.519 E.63217
; LINE_WIDTH: 0.521596
G1 X121.075 Y117.925 E.05191
; LINE_WIDTH: 0.544336
G1 X120.771 Y118.324 E.01669
G1 X121.109 Y118.311 E.01125
; LINE_WIDTH: 0.521596
G1 X122.361 Y119.362 E.05191
; LINE_WIDTH: 0.519996
G1 X122.77 Y119.706 E.01692
G1 X122.85 Y119.862 E.00555
G1 X122.826 Y119.938 E.00254
; WIPE_START
M204 S10000
G1 X122.506 Y120.326 E-.19127
G1 X122.358 Y120.33 E-.05624
G1 X122.085 Y120.113 E-.13248
; WIPE_END
G1 E-.02 F1800
G1 X128.071 Y115.378 Z8.76 F36000
G1 X143.407 Y103.247 Z8.76
G1 Z8.36
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X143.407 Y103.588 E.02221
; WIPE_START
G1 X143.407 Y103.247 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X136.027 Y101.3 Z8.76 F36000
G1 X131.49 Y100.103 Z8.76
G1 Z8.36
G1 E.4 F1800
; LINE_WIDTH: 1.05372
G1 F7725.284
G1 X131.605 Y99.739 E.02538
; WIPE_START
G1 X131.49 Y100.103 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X127.62 Y106.681 Z8.76 F36000
G1 X120.771 Y118.324 Z8.76
G1 Z8.36
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X120.546 Y118.596 E.01171
; WIPE_START
M204 S10000
G1 X120.771 Y118.324 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.131 Y122.412 Z8.76 F36000
G1 Z8.36
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X118.484 Y124.077 I-24.964 J-23.044 E.08945
G2 X119.402 Y126.109 I5.854 J-1.423 E.08561
G2 X121.333 Y127.994 I10.997 J-9.335 E.10319
G3 X122.358 Y129.408 I-2.813 J3.116 E.06715
G3 X121.521 Y133.65 I-4.725 J1.271 E.17097
G1 X121.226 Y133.968 E.01658
G3 X121.116 Y135.986 I-7.457 J.607 E.07741
G3 X126.262 Y138.684 I-34.353 J71.797 E.22188
G1 X126.374 Y138.362 E.01299
G1 X126.943 Y137.42 E.04203
G3 X128.874 Y135.535 I10.997 J9.335 E.10319
G2 X129.898 Y134.121 I-2.813 J-3.117 E.06715
G2 X129.062 Y129.879 I-4.725 J-1.271 E.17097
G2 X127.13 Y127.994 I-10.997 J9.335 E.10319
G3 X126.106 Y126.58 I2.813 J-3.116 E.06715
G3 X126.943 Y122.339 I4.725 J-1.271 E.17097
G3 X128.874 Y120.453 I10.997 J9.335 E.10319
G2 X129.898 Y119.04 I-2.813 J-3.117 E.06715
G2 X129.062 Y114.798 I-4.725 J-1.271 E.17097
G2 X127.285 Y113.051 I-10.187 J8.583 E.09529
G3 X126.028 Y115.027 I-54.074 J-32.987 E.08944
G1 X141.945 Y145.142 F36000
G1 F13446.283
G1 X141.945 Y142.799 E.08944
G3 X141.187 Y141.662 I2.744 J-2.65 E.05248
G3 X141.945 Y137.55 I4.658 J-1.267 E.16509
G1 X141.945 Y127.718 E.37539
G3 X141.187 Y126.58 I2.744 J-2.65 E.05248
G3 X141.945 Y122.469 I4.658 J-1.267 E.16509
G1 X141.945 Y112.636 E.37539
G3 X141.187 Y111.499 I2.744 J-2.65 E.05248
G3 X141.945 Y107.387 I4.658 J-1.267 E.16509
G1 X141.945 Y105.864 E.05815
G1 X134.204 Y105.861 E.29555
G2 X133.647 Y106.786 I2.325 J2.033 E.04146
G2 X134.483 Y111.028 I4.725 J1.271 E.17097
G2 X136.415 Y112.913 I10.998 J-9.336 E.10319
G3 X137.439 Y114.327 I-2.813 J3.116 E.06715
G3 X136.602 Y118.568 I-4.725 J1.271 E.17097
G3 X134.671 Y120.453 I-10.997 J-9.335 E.10319
G2 X133.647 Y121.867 I2.813 J3.116 E.06715
G2 X134.483 Y126.109 I4.725 J1.271 E.17097
G2 X136.415 Y127.994 I10.998 J-9.336 E.10319
G3 X137.439 Y129.408 I-2.813 J3.116 E.06715
G3 X136.602 Y133.65 I-4.725 J1.271 E.17097
G3 X134.671 Y135.535 I-10.997 J-9.335 E.10319
G2 X133.512 Y137.42 I2.423 J2.789 E.08581
G2 X135.359 Y142.133 I5.258 J.658 E.20148
G3 X137.21 Y144.018 I-5.295 J7.053 E.10127
G3 X137.337 Y147.311 I-3.679 J1.791 E.12948
G3 X138.954 Y149.006 I-33.144 J33.227 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 8.52
; LAYER_HEIGHT: 0.160001
; WIPE_START
G1 F13446.283
G1 X138.264 Y148.282 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L53
M991 S0 P52 ;notify layer change


G17
G3 Z8.76 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X130.888 Y104.707
G1 Z8.52
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.131 I-50.713 J-20.213 E.45136
G3 X123.072 Y118.225 I-29.873 J-19.716 E.14699
M73 P79 R4
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-42.75 J-39.331 E.298
G1 X118.015 Y129.012 E.15055
G3 X118.932 Y129.849 I-4.609 J5.971 E.04748
G3 X120.074 Y131.528 I-5.862 J5.215 E.07776
G1 X120.45 Y132.477 E.03894
G3 X120.718 Y133.726 I-10.184 J2.842 E.04882
G1 X120.765 Y134.489 E.02919
G1 X120.698 Y135.504 E.03883
G1 X120.538 Y136.268 E.0298
G3 X130.504 Y142.099 I-22.495 J49.881 E.44163
G3 X142.443 Y154.069 I-32.642 J44.497 E.64803
G1 X142.443 Y105.365 E1.85948
G1 X140.524 Y105.364 E.07329
G1 X139.908 Y105.297 E.02364
G1 X139.349 Y105.364 E.0215
G3 X134.484 Y105.37 I-3.098 J-528.955 E.18573
G1 X132.484 Y105.364 E.07636
G1 X131.929 Y105.281 E.02141
G1 X131.679 Y105.206 E.00996
G1 X131.158 Y104.961 E.02199
G1 X130.954 Y104.769 E.01071
; WIPE_START
G1 X130.508 Y105.664 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X130.899 Y103.037 Z8.92 F36000
G1 Z8.52
G1 E.4 F1800
G1 F13446.369
G3 X124.88 Y114.797 I-50.658 J-18.509 E.50564
G3 X122.623 Y117.848 I-29.319 J-19.329 E.14498
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-40.196 J-36.747 E.32255
G1 X117.666 Y129.482 E.17936
G3 X118.523 Y130.268 I-5.569 J6.934 E.04444
G3 X119.208 Y131.168 I-7.755 J6.608 E.0432
G1 X119.561 Y131.811 E.02801
G1 X119.901 Y132.681 E.03566
G3 X120.136 Y133.793 I-10.154 J2.727 E.04341
G1 X120.18 Y134.525 E.028
G1 X120.115 Y135.454 E.03555
G1 X119.933 Y136.283 E.03241
G1 X119.836 Y136.599 E.01262
G3 X130.932 Y143.157 I-21.116 J48.4 E.49334
G3 X142.92 Y155.769 I-32.376 J42.778 E.66729
G1 X143.029 Y155.749 E.00421
G1 X143.029 Y104.639 E1.95133
G1 X142.434 Y104.781 E.02336
G1 X140.525 Y104.778 E.0729
G1 X139.901 Y104.705 E.02399
G1 X139.346 Y104.778 E.02134
G3 X134.486 Y104.784 I-3.095 J-527.546 E.18557
G1 X132.486 Y104.778 E.07636
G1 X132.049 Y104.706 E.01692
G1 X131.483 Y104.474 E.02334
G1 X131.138 Y104.143 E.01823
G1 X130.911 Y103.647 E.02084
G1 X130.901 Y103.127 E.01985
; WIPE_START
G1 X130.525 Y104.054 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X131.665 Y100.977 Z8.92 F36000
G1 Z8.52
G1 E.4 F1800
; LINE_WIDTH: 0.706925
G1 F9797.349
G1 X131.636 Y100.922 E.00271
; LINE_WIDTH: 0.756244
G1 F9608.533
G1 X131.607 Y100.868 E.00291
; LINE_WIDTH: 0.805563
G1 F9421.568
G1 X131.577 Y100.813 E.00311
; LINE_WIDTH: 0.854882
G1 F9236.441
G1 X131.548 Y100.759 E.00331
; LINE_WIDTH: 0.904201
G1 F9053.136
G1 X131.518 Y100.705 E.00351
; LINE_WIDTH: 0.95352
G1 F8567.388
G1 X131.489 Y100.65 E.00371
; LINE_WIDTH: 1.00284
G1 F8131.112
G1 X131.46 Y100.596 E.00391
; LINE_WIDTH: 1.05216
G1 F7737.116
G1 X131.43 Y100.541 E.0041
; LINE_WIDTH: 1.10148
G1 F7379.538
G1 X131.401 Y100.487 E.0043
G1 X131.359 Y100.537 E.00453
; LINE_WIDTH: 1.05216
G1 F7737.116
G1 X131.317 Y100.587 E.00432
; LINE_WIDTH: 1.00284
G1 F8131.112
G1 X131.276 Y100.637 E.00411
; LINE_WIDTH: 0.95352
G1 F8567.388
G1 X131.234 Y100.687 E.0039
; LINE_WIDTH: 0.904201
G1 F9053.136
G1 X131.192 Y100.737 E.00369
; LINE_WIDTH: 0.854882
G1 F9597.274
G1 X131.151 Y100.787 E.00348
; LINE_WIDTH: 0.805563
G1 F9795.816
G1 X131.109 Y100.836 E.00327
; LINE_WIDTH: 0.756244
G1 F9996.39
G1 X131.067 Y100.886 E.00306
; LINE_WIDTH: 0.706925
G1 F10198.976
G1 X131.026 Y100.936 E.00285
; LINE_WIDTH: 0.657606
G1 F11489.001
G1 X130.893 Y101.314 E.01625
G1 F12634.975
G1 X130.761 Y101.691 E.01625
G1 X130.594 Y102.167 E.02048
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X130.453 Y102.542 E.01527
G1 X129.903 Y104.019 E.06021
G3 X124.399 Y114.463 I-50.01 J-19.685 E.45161
G3 X122.174 Y117.472 I-28.762 J-18.94 E.14298
G1 X119.604 Y120.538 E.15272
G3 X112.93 Y126.698 I-39.858 J-36.491 E.34719
G1 X117.324 Y129.958 E.20891
G3 X118.151 Y130.725 I-6.61 J7.945 E.04308
G3 X118.695 Y131.451 I-12.211 J9.726 E.03461
G1 X119.048 Y132.093 E.02801
G1 X119.352 Y132.885 E.03237
G3 X119.551 Y133.828 I-12.674 J3.172 E.03682
G1 X119.596 Y134.561 E.028
G1 X119.531 Y135.403 E.03227
G1 X119.363 Y136.148 E.02913
G1 X119.08 Y136.91 E.03105
G3 X130.579 Y143.624 I-20.844 J48.905 E.5097
G3 X142.648 Y156.415 I-31.964 J42.247 E.67455
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y105.065 E1.95363
G1 X143.615 Y104.665 E.01527
; LINE_WIDTH: 0.659516
G1 F12371.321
G1 X143.595 Y104.487 E.0073
; LINE_WIDTH: 0.699036
G1 F11761.622
G1 X143.575 Y104.309 E.00776
; LINE_WIDTH: 0.738556
G1 F11167.297
G1 X143.555 Y104.131 E.00823
; LINE_WIDTH: 0.778076
G1 F10588.379
G1 X143.536 Y103.953 E.00869
; LINE_WIDTH: 0.820803
G1 F10013.141
G1 X143.514 Y103.892 E.00331
; LINE_WIDTH: 0.86353
G1 F9497.183
G1 X143.493 Y103.831 E.00349
; LINE_WIDTH: 0.906256
G1 F9031.793
G1 X143.471 Y103.77 E.00367
; LINE_WIDTH: 0.948983
G1 F8609.883
G1 X143.45 Y103.709 E.00385
; LINE_WIDTH: 0.99171
G1 F8225.632
G1 X143.429 Y103.648 E.00403
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X143.407 Y103.588 E.00421
G1 X143.369 Y103.632 E.00385
; LINE_WIDTH: 0.99171
G1 F8225.632
G1 X143.33 Y103.676 E.00368
; LINE_WIDTH: 0.948983
G1 F8609.883
G1 X143.291 Y103.721 E.00352
; LINE_WIDTH: 0.906256
G1 F9031.793
G1 X143.252 Y103.765 E.00335
; LINE_WIDTH: 0.86353
G1 F9497.183
G1 X143.213 Y103.81 E.00319
; LINE_WIDTH: 0.820803
G1 F10013.141
G1 X143.174 Y103.854 E.00302
; LINE_WIDTH: 0.778076
G1 F10588.379
G1 X143.072 Y103.921 E.00594
; LINE_WIDTH: 0.738556
G1 F11182.586
G1 X142.97 Y103.989 E.00563
; LINE_WIDTH: 0.699036
G1 F11587.713
G1 X142.867 Y104.056 E.00531
; LINE_WIDTH: 0.659516
G1 F12000.001
G1 X142.765 Y104.123 E.00499
; LINE_WIDTH: 0.619996
G1 F13173.322
G1 X142.435 Y104.196 E.01289
G1 F13446.369
G1 X142.035 Y104.195 E.01527
G1 X140.526 Y104.192 E.05762
G1 X139.893 Y104.113 E.02434
G1 X139.344 Y104.192 E.02119
G3 X134.488 Y104.198 I-3.092 J-527.701 E.18541
G1 X132.488 Y104.192 E.07636
G1 X132.016 Y104.085 E.01846
G3 X131.611 Y103.798 I.224 J-.746 E.01927
G1 X131.478 Y103.496 E.01261
G1 X131.465 Y103.26 E.00902
G1 X131.615 Y101.785 E.05661
G1 X131.656 Y101.387 E.01527
; LINE_WIDTH: 0.657606
G1 F12634.975
G1 X131.663 Y101.067 E.01302
; WIPE_START
G1 X131.636 Y100.922 E-.05585
G1 X131.607 Y100.868 E-.02351
G1 X131.577 Y100.813 E-.02351
G1 X131.548 Y100.759 E-.02351
G1 X131.518 Y100.705 E-.02351
G1 X131.489 Y100.65 E-.02351
G1 X131.46 Y100.596 E-.02351
G1 X131.43 Y100.541 E-.02351
G1 X131.401 Y100.487 E-.02351
G1 X131.359 Y100.537 E-.02473
G1 X131.317 Y100.587 E-.02473
G1 X131.276 Y100.637 E-.02473
G1 X131.234 Y100.687 E-.02473
G1 X131.192 Y100.737 E-.02473
G1 X131.171 Y100.762 E-.01244
; WIPE_END
G1 E-.02 F1800
G1 X138.652 Y102.277 Z8.92 F36000
G1 X143.407 Y103.241 Z8.92
G1 Z8.52
G1 E.4 F1800
; LINE_WIDTH: 1.03185
G1 F7894.658
G1 X143.425 Y102.813 E.02783
; LINE_WIDTH: 0.996001
G1 F8188.923
G1 X143.443 Y102.385 E.02683
; LINE_WIDTH: 0.960156
G1 F8505.974
G1 X143.464 Y101.886 E.03018
; LINE_WIDTH: 0.918281
G1 F8908.925
G1 X143.486 Y101.386 E.02881
; LINE_WIDTH: 0.876406
G1 F9351.955
G1 X143.507 Y100.887 E.02745
; LINE_WIDTH: 0.834531
G1 F9841.352
G1 X143.528 Y100.387 E.02608
; LINE_WIDTH: 0.792656
G1 F10384.799
G1 X143.543 Y99.624 E.03774
; LINE_WIDTH: 0.763536
G1 F10799.507
G3 X142.862 Y99.716 I-.995 J-4.812 E.03268
; LINE_WIDTH: 0.792656
G1 F10384.799
G1 X142.775 Y100.323 E.03032
G1 X142.776 Y100.455 E.00653
; LINE_WIDTH: 0.828496
G1 F9916.138
G1 X142.871 Y100.873 E.02216
; LINE_WIDTH: 0.864336
G1 F9487.952
G1 X142.965 Y101.29 E.02316
; LINE_WIDTH: 0.906214
G1 F9032.233
G1 X143.075 Y101.778 E.02842
; LINE_WIDTH: 0.948091
G1 F8618.284
G1 X143.186 Y102.265 E.02978
; LINE_WIDTH: 0.989969
G1 F8240.616
G1 X143.296 Y102.753 E.03115
; LINE_WIDTH: 1.03185
G1 F7894.658
G1 X143.387 Y103.153 E.02666
; WIPE_START
G1 X143.425 Y102.813 E-.12994
G1 X143.443 Y102.385 E-.16264
G1 X143.453 Y102.156 E-.08742
; WIPE_END
G1 E-.02 F1800
G1 X135.944 Y100.79 Z8.92 F36000
G1 X132.273 Y100.122 Z8.92
G1 Z8.52
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.567806
G1 F3600
M204 S5000
G1 X132.3 Y100.061 E.00234
; LINE_WIDTH: 0.615616
G1 X132.328 Y100 E.00255
; LINE_WIDTH: 0.65378
G1 X132.364 Y99.968 E.00194
; LINE_WIDTH: 0.691943
G1 X132.401 Y99.937 E.00206
; LINE_WIDTH: 0.730106
G1 X132.437 Y99.905 E.00218
; LINE_WIDTH: 0.733196
G1 X133.006 Y99.996 E.02624
G1 X133.153 Y99.203 E.03675
G1 X132.623 Y99.191 E.02414
; LINE_WIDTH: 0.726766
G1 X132.454 Y99.159 E.00777
; LINE_WIDTH: 0.689716
G1 X132.285 Y99.127 E.00735
; LINE_WIDTH: 0.652666
G1 X132.116 Y99.096 E.00694
; LINE_WIDTH: 0.615616
G1 X131.823 Y99.02 E.01144
; LINE_WIDTH: 0.567806
G1 X131.531 Y98.945 E.0105
; LINE_WIDTH: 0.519996
G1 X131.038 Y98.936 E.01561
G3 X123.946 Y114.147 I-50.937 J-14.491 E.5336
G3 X121.747 Y117.121 I-28.178 J-18.531 E.11716
G1 X119.177 Y120.187 E.12664
G3 X112.01 Y126.704 I-40.868 J-37.745 E.30709
G1 X116.99 Y130.398 E.19631
G3 X117.738 Y131.094 I-6.448 J7.689 E.03237
G3 X118.21 Y131.717 I-28.654 J22.179 E.02475
G1 X118.564 Y132.36 E.02322
G1 X118.833 Y133.078 E.02428
G3 X118.999 Y133.862 I-29.674 J6.682 E.02538
G1 X119.044 Y134.599 E.02337
G1 X118.98 Y135.356 E.02405
G1 X118.825 Y136.02 E.02159
G1 X118.547 Y136.753 E.02483
G1 X118.31 Y137.19 E.01573
G3 X130.249 Y144.067 I-19.589 J47.806 E.43752
G3 X142.39 Y157.025 I-31.646 J41.818 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.369 E1.81495
G1 X144.167 Y98.946 E.01338
G1 X143.705 Y98.969 E.01467
G1 X142.886 Y99.089 E.0262
G3 X141.715 Y99.095 I-1.301 J-151.898 E.03709
G1 X142.16 Y99.465 E.01834
G1 X142.238 Y99.677 E.00714
G1 X142.115 Y100.392 E.02299
G1 X142.659 Y103.378 E.09609
M73 P79 R3
G3 X142.436 Y103.643 I-.247 J.018 E.01212
G1 X140.527 Y103.64 E.06045
G1 X140.399 Y103.623 E.00407
G1 X140.272 Y103.605 E.00406
G1 X140.145 Y103.588 E.00406
G1 X140.018 Y103.571 E.00406
G1 X139.89 Y103.554 E.00406
G1 X139.883 Y103.554 E.00024
G1 X139.78 Y103.571 E.00328
G1 X139.678 Y103.588 E.00328
G1 X139.576 Y103.604 E.00328
G1 X139.473 Y103.621 E.00328
G1 X139.371 Y103.638 E.00328
G3 X134.489 Y103.646 I-3.22 J-469.02 E.15455
G1 X132.489 Y103.64 E.06332
G1 X132.111 Y103.524 E.01253
G1 X132.019 Y103.384 E.00529
G1 X132.038 Y103.174 E.00668
G1 X132.057 Y102.964 E.00668
G1 X132.076 Y102.754 E.00668
G1 X132.095 Y102.544 E.00668
G3 X132.251 Y101 I68.558 J6.141 E.04915
G1 X132.241 Y100.937 E.002
G1 X132.228 Y100.85 E.00279
G1 X132.214 Y100.763 E.00279
G1 X132.201 Y100.676 E.00279
G1 X132.188 Y100.589 E.00279
G1 X132.257 Y100.211 E.01219
; WIPE_START
M204 S10000
G1 X132.3 Y100.061 E-.05929
G1 X132.328 Y100 E-.02553
G1 X132.364 Y99.968 E-.01827
G1 X132.401 Y99.937 E-.01828
G1 X132.437 Y99.905 E-.01828
G1 X133.006 Y99.996 E-.2189
G1 X133.016 Y99.94 E-.02145
; WIPE_END
G1 E-.02 F1800
G1 X140.289 Y102.255 Z8.92 F36000
G1 X143.407 Y103.247 Z8.92
G1 Z8.52
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X143.407 Y103.588 E.02221
; WIPE_START
G1 X143.407 Y103.247 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X136.091 Y101.073 Z8.92 F36000
G1 X131.605 Y99.739 Z8.92
G1 Z8.52
G1 E.4 F1800
; LINE_WIDTH: 1.0992
G1 F7395.338
G1 X131.404 Y100.474 E.05283
; LINE_WIDTH: 1.10148
G1 F7379.538
G1 X131.401 Y100.487 E.00095
; WIPE_START
G1 X131.404 Y100.474 E-.00669
G1 X131.605 Y99.739 E-.37331
; WIPE_END
G1 E-.02 F1800
G1 X128.975 Y106.904 Z8.92 F36000
G1 X125.956 Y115.13 Z8.92
G1 Z8.52
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X127.218 Y113.157 I-26.466 J-18.325 E.08945
G3 X128.991 Y114.798 I-9.163 J11.675 E.09232
G3 X130.059 Y119.04 I-3.893 J3.236 E.17269
G3 X129.075 Y120.453 I-3.351 J-1.282 E.06641
G2 X127.014 Y122.339 I10.514 J13.567 E.10677
G2 X125.946 Y126.58 I3.893 J3.236 E.17269
G2 X126.929 Y127.994 I3.351 J-1.282 E.06641
G3 X128.991 Y129.879 I-10.514 J13.567 E.10677
G3 X130.059 Y134.121 I-3.893 J3.236 E.17269
G3 X129.075 Y135.535 I-3.351 J-1.282 E.06641
G2 X127.014 Y137.42 I10.514 J13.567 E.10677
G1 X126.39 Y138.362 E.04316
G1 X126.261 Y138.686 E.01331
G2 X121.113 Y135.985 I-35.469 J61.344 E.22202
G2 X121.225 Y133.873 I-8.215 J-1.497 E.08096
G1 X121.45 Y133.65 E.01211
G2 X122.518 Y129.408 I-3.893 J-3.236 E.17269
G2 X121.534 Y127.994 I-3.351 J1.282 E.06641
G3 X119.473 Y126.109 I10.514 J-13.567 E.10677
G3 X118.451 Y124.109 I5.093 J-3.864 E.08619
G2 X120.099 Y122.445 I-22.902 J-24.333 E.08945
G1 X125.147 Y119.848 F36000
; FEATURE: Bridge
; LINE_WIDTH: 0.737126
G1 F1800
G1 X123.261 Y122.1 E.13459
G1 X122.99 Y122.265 E.01453
G1 X122.713 Y122.376 E.01365
G1 X122.528 Y122.425 E.00878
G1 X122.239 Y122.463 E.01336
G1 X122.044 Y122.459 E.00896
G1 X124.868 Y119.086 E.20154
G2 X124.585 Y118.413 I-2.05 J.464 E.03361
G1 X124.549 Y118.373 E.00251
G1 X121.281 Y122.275 E.23321
G1 X120.83 Y121.993 E.02434
G1 X120.695 Y121.88 E.00809
G1 X124.228 Y117.662 E.25208
G1 X141.945 Y145.351 F36000
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X141.945 Y143.008 E.08944
G3 X141.027 Y141.662 I2.33 J-2.576 E.06282
G3 X141.945 Y137.646 I4.846 J-1.004 E.16221
G1 X141.945 Y127.927 E.37106
G3 X141.027 Y126.58 I2.33 J-2.576 E.06282
G3 X141.945 Y122.565 I4.846 J-1.004 E.16221
G1 X141.945 Y112.846 E.37106
G3 X141.027 Y111.499 I2.33 J-2.576 E.06282
G3 X141.945 Y107.483 I4.846 J-1.004 E.16221
G1 X141.945 Y105.864 E.06182
G1 X140.512 Y105.861 E.05471
G2 X139.357 Y105.861 I-.578 J3.763 E.04429
G1 X134.001 Y105.862 E.20447
G2 X133.486 Y106.786 I2.042 J1.743 E.04067
G2 X134.554 Y111.028 I4.961 J1.006 E.17268
G2 X136.616 Y112.913 I12.575 J-11.681 E.10677
G3 X137.599 Y114.327 I-2.368 J2.696 E.06641
G3 X136.531 Y118.568 I-4.961 J1.006 E.17269
G3 X134.47 Y120.453 I-12.575 J-11.681 E.10677
G2 X133.486 Y121.867 I2.367 J2.696 E.06641
G2 X134.554 Y126.109 I4.961 J1.006 E.17269
G2 X136.616 Y127.994 I12.575 J-11.681 E.10677
G3 X137.599 Y129.408 I-2.367 J2.696 E.06641
G3 X136.531 Y133.65 I-4.961 J1.006 E.17269
G3 X134.47 Y135.535 I-12.575 J-11.681 E.10677
G2 X133.486 Y136.949 I2.367 J2.696 E.06641
G2 X134.554 Y141.19 I4.961 J1.006 E.17268
G2 X136.616 Y143.075 I12.575 J-11.681 E.10677
G3 X137.397 Y144.018 I-2.439 J2.816 E.04696
G3 X137.342 Y147.319 I-3.663 J1.59 E.12997
G3 X138.961 Y149.012 I-30.89 J31.16 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 8.68
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X138.27 Y148.29 E-.38001
; WIPE_END
G1 E-.01999 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L54
M991 S0 P53 ;notify layer change


G17
G3 Z8.92 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X130.98 Y104.473
G1 Z8.68
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.131 I-51.531 J-20.359 E.46092
G3 X123.072 Y118.225 I-29.869 J-19.713 E.14699
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-42.734 J-39.315 E.298
G1 X118.015 Y129.012 E.15055
G3 X118.933 Y129.85 I-4.607 J5.968 E.04752
G3 X120.076 Y131.532 I-5.877 J5.223 E.07787
G1 X120.45 Y132.48 E.03891
G3 X120.718 Y133.726 I-10.207 J2.843 E.0487
G1 X120.765 Y134.489 E.02919
G1 X120.698 Y135.506 E.03892
G1 X120.538 Y136.268 E.0297
G3 X131.286 Y142.69 I-21.87 J48.804 E.4791
G3 X142.443 Y154.07 I-32.931 J43.447 E.61066
G1 X142.443 Y105.365 E1.85951
G1 X141.705 Y105.366 E.02817
G3 X139.878 Y105.139 I3.101 J-32.41 E.07033
G3 X138.816 Y105.366 I-1.355 J-3.736 E.04156
G3 X135.227 Y105.372 I-2.613 J-473.776 E.13703
G1 X133.227 Y105.366 E.07636
G1 X132.68 Y105.286 E.02113
G1 X131.879 Y105.048 E.03189
G1 X131.402 Y104.834 E.01994
G1 X131.049 Y104.532 E.01777
; WIPE_START
G1 X130.613 Y105.432 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X131.137 Y102.38 Z9.08 F36000
G1 Z8.68
G1 E.4 F1800
G1 F13446.369
G3 X124.88 Y114.797 I-51.342 J-18.087 E.53233
G3 X122.623 Y117.848 I-29.318 J-19.329 E.14498
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-40.191 J-36.741 E.32255
G1 X117.666 Y129.482 E.17936
G3 X118.524 Y130.269 I-5.554 J6.917 E.04447
G3 X119.21 Y131.171 I-7.801 J6.64 E.04329
G1 X119.563 Y131.814 E.02801
G1 X119.901 Y132.684 E.03563
G3 X120.135 Y133.793 I-10.201 J2.733 E.04329
G1 X120.18 Y134.525 E.02801
G1 X120.114 Y135.456 E.03563
G1 X119.931 Y136.289 E.03255
G1 X119.836 Y136.599 E.0124
G3 X130.932 Y143.157 I-21.099 J48.37 E.49334
G3 X142.92 Y155.769 I-32.361 J42.764 E.66728
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.639 E1.95133
G1 X142.434 Y104.781 E.02336
G1 X141.707 Y104.78 E.02777
G3 X139.828 Y104.546 I5.227 J-49.686 E.07227
G3 X138.814 Y104.78 I-1.699 J-5.056 E.0398
G3 X135.229 Y104.786 I-2.609 J-472.655 E.13688
G1 X133.229 Y104.78 E.07636
G1 X132.846 Y104.724 E.01478
G1 X132.045 Y104.487 E.03189
G1 X131.712 Y104.337 E.01395
G1 X131.4 Y104.07 E.01568
G1 X131.148 Y103.638 E.01911
G1 X131.074 Y103.051 E.02258
G1 X131.129 Y102.469 E.0223
; WIPE_START
G1 X130.77 Y103.403 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X131.681 Y100.481 Z9.08 F36000
G1 Z8.68
G1 E.4 F1800
; LINE_WIDTH: 0.76817
G1 F9752.204
G1 X131.672 Y100.465 E.00087
; LINE_WIDTH: 0.815841
G1 F9696.708
G1 X131.642 Y100.407 E.00335
; LINE_WIDTH: 0.863512
G1 F9497.386
G1 X131.612 Y100.349 E.00355
; LINE_WIDTH: 0.911183
G1 F8981.047
G1 X131.581 Y100.291 E.00375
; LINE_WIDTH: 0.958854
G1 F8517.954
G1 X131.551 Y100.232 E.00396
; LINE_WIDTH: 1.00653
G1 F8100.278
G1 X131.521 Y100.174 E.00416
; LINE_WIDTH: 1.0542
G1 F7721.648
G1 X131.49 Y100.116 E.00437
G1 X131.449 Y100.171 E.00457
; LINE_WIDTH: 1.00653
G1 F8100.278
G1 X131.407 Y100.225 E.00436
; LINE_WIDTH: 0.958854
G1 F8517.954
G1 X131.366 Y100.28 E.00414
; LINE_WIDTH: 0.911183
G1 F8981.047
G1 X131.324 Y100.335 E.00393
; LINE_WIDTH: 0.863512
G1 F9497.386
G1 X131.283 Y100.39 E.00372
; LINE_WIDTH: 0.815841
G1 F10076.718
G1 X131.241 Y100.444 E.0035
; LINE_WIDTH: 0.76817
G1 F10291.679
G1 X131.2 Y100.499 E.00329
; LINE_WIDTH: 0.720498
G1 F10508.883
G1 X131.158 Y100.554 E.00307
; LINE_WIDTH: 0.672827
G1 F10728.336
G1 X131.117 Y100.609 E.00286
; LINE_WIDTH: 0.625156
G1 F12050.431
G1 X130.99 Y100.988 E.01541
G1 F13328.934
G1 X130.864 Y101.368 E.01541
G1 X130.702 Y101.856 E.01979
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X124.399 Y114.463 I-50.997 J-17.615 E.53966
G3 X122.174 Y117.472 I-28.762 J-18.94 E.14298
G1 X119.604 Y120.538 E.15272
G3 X112.93 Y126.698 I-39.853 J-36.485 E.34719
G1 X117.325 Y129.958 E.20892
G3 X118.151 Y130.726 I-6.594 J7.927 E.0431
G3 X118.696 Y131.453 I-12.341 J9.819 E.0347
G1 X119.05 Y132.096 E.02801
G1 X119.352 Y132.887 E.03234
G3 X119.551 Y133.828 I-12.769 J3.188 E.03672
G1 X119.596 Y134.56 E.02801
G1 X119.531 Y135.405 E.03235
G1 X119.361 Y136.153 E.02927
G1 X119.08 Y136.91 E.03084
G3 X130.579 Y143.624 I-20.862 J48.935 E.5097
G3 X142.648 Y156.415 I-31.956 J42.239 E.67454
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y105.065 E1.95363
G1 X143.615 Y104.665 E.01527
; LINE_WIDTH: 0.659516
G1 F12371.321
G1 X143.595 Y104.487 E.0073
; LINE_WIDTH: 0.699036
G1 F11761.622
G1 X143.575 Y104.309 E.00776
; LINE_WIDTH: 0.738556
G1 F11167.297
G1 X143.555 Y104.131 E.00823
; LINE_WIDTH: 0.778076
G1 F10588.379
G1 X143.536 Y103.953 E.00869
; LINE_WIDTH: 0.820803
G1 F10013.141
G1 X143.514 Y103.892 E.00331
; LINE_WIDTH: 0.86353
G1 F9497.183
G1 X143.493 Y103.831 E.00349
; LINE_WIDTH: 0.906256
G1 F9031.793
G1 X143.471 Y103.77 E.00367
; LINE_WIDTH: 0.948983
G1 F8609.883
G1 X143.45 Y103.709 E.00385
; LINE_WIDTH: 0.99171
G1 F8225.632
G1 X143.429 Y103.648 E.00403
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X143.407 Y103.588 E.00421
G1 X143.369 Y103.632 E.00385
; LINE_WIDTH: 0.99171
G1 F8225.632
G1 X143.33 Y103.676 E.00368
; LINE_WIDTH: 0.948983
G1 F8609.883
G1 X143.291 Y103.721 E.00352
; LINE_WIDTH: 0.906256
G1 F9031.793
G1 X143.252 Y103.765 E.00335
; LINE_WIDTH: 0.86353
G1 F9497.183
G1 X143.213 Y103.81 E.00319
; LINE_WIDTH: 0.820803
G1 F10013.141
G1 X143.174 Y103.854 E.00302
; LINE_WIDTH: 0.778076
G1 F10588.379
G1 X143.072 Y103.921 E.00594
; LINE_WIDTH: 0.738556
G1 F11182.586
G1 X142.97 Y103.989 E.00563
; LINE_WIDTH: 0.699036
G1 F11587.713
G1 X142.867 Y104.056 E.00531
; LINE_WIDTH: 0.659516
G1 F12000.001
G1 X142.765 Y104.123 E.00499
; LINE_WIDTH: 0.619996
G1 F13173.322
G1 X142.435 Y104.196 E.01289
G1 F13446.369
G1 X142.035 Y104.195 E.01527
G1 X141.708 Y104.194 E.0125
G3 X139.869 Y103.952 I4.537 J-41.407 E.0708
G3 X138.812 Y104.194 I-6.033 J-23.84 E.04143
G3 X135.231 Y104.201 I-2.607 J-472.107 E.13672
G1 X133.231 Y104.194 E.07636
G3 X132.035 Y103.849 I1.354 J-6.919 E.04758
G1 X131.843 Y103.687 E.00958
G1 X131.695 Y103.427 E.01144
G1 X131.661 Y103.07 E.0137
G1 X131.85 Y101.061 E.07704
; LINE_WIDTH: 0.625156
G1 F13179.691
G1 X131.763 Y100.64 E.01655
; LINE_WIDTH: 0.672827
G1 F11695.857
G1 X131.733 Y100.582 E.00273
; LINE_WIDTH: 0.720498
G1 F11476.879
G1 X131.722 Y100.561 E.00104
; WIPE_START
G1 X131.672 Y100.465 E-.0411
G1 X131.642 Y100.407 E-.02495
G1 X131.612 Y100.349 E-.02495
G1 X131.581 Y100.291 E-.02495
G1 X131.551 Y100.232 E-.02495
G1 X131.521 Y100.174 E-.02495
G1 X131.49 Y100.116 E-.02495
G1 X131.449 Y100.171 E-.02612
G1 X131.407 Y100.225 E-.02612
G1 X131.366 Y100.28 E-.02612
G1 X131.324 Y100.335 E-.02612
G1 X131.283 Y100.39 E-.02612
G1 X131.241 Y100.444 E-.02612
G1 X131.2 Y100.499 E-.02612
G1 X131.19 Y100.513 E-.00633
; WIPE_END
G1 E-.02 F1800
M73 P80 R3
G1 X138.821 Y100.388 Z9.08 F36000
G1 X142.775 Y100.323 Z9.08
G1 Z8.68
G1 E.4 F1800
; LINE_WIDTH: 0.792656
G1 F10384.799
G1 X142.776 Y100.455 E.00653
; LINE_WIDTH: 0.828496
G1 F9916.138
G1 X142.871 Y100.873 E.02216
; LINE_WIDTH: 0.864336
G1 F9487.952
G1 X142.965 Y101.29 E.02316
; LINE_WIDTH: 0.906214
G1 F9032.233
G1 X143.075 Y101.778 E.02842
; LINE_WIDTH: 0.948091
G1 F8618.284
G1 X143.186 Y102.265 E.02978
; LINE_WIDTH: 0.989969
G1 F8240.616
G1 X143.296 Y102.753 E.03115
; LINE_WIDTH: 1.03185
G1 F7894.658
G1 X143.407 Y103.241 E.03251
G1 X143.425 Y102.813 E.02783
; LINE_WIDTH: 0.996001
G1 F8188.923
G1 X143.443 Y102.385 E.02683
; LINE_WIDTH: 0.960156
G1 F8505.974
G1 X143.464 Y101.886 E.03018
; LINE_WIDTH: 0.918281
G1 F8908.925
G1 X143.486 Y101.386 E.02881
; LINE_WIDTH: 0.876406
G1 F9351.955
G1 X143.507 Y100.887 E.02745
; LINE_WIDTH: 0.834531
G1 F9841.352
G1 X143.528 Y100.387 E.02608
; LINE_WIDTH: 0.792656
G1 F10384.799
G1 X143.543 Y99.625 E.03769
; LINE_WIDTH: 0.763686
G1 F10797.287
G3 X142.858 Y99.718 I-.986 J-4.694 E.03286
; LINE_WIDTH: 0.792656
G1 F10384.799
G1 X142.788 Y100.234 E.02574
; WIPE_START
G1 X142.776 Y100.455 E-.08421
G1 X142.871 Y100.873 E-.16264
G1 X142.948 Y101.214 E-.13315
; WIPE_END
G1 E-.02 F1800
G1 X135.344 Y100.55 Z9.08 F36000
G1 X132.257 Y100.281 Z9.08
G1 Z8.68
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G3 X132.282 Y100.125 I.346 J-.024 E.00502
; LINE_WIDTH: 0.565576
G1 X132.309 Y100.065 E.00228
; LINE_WIDTH: 0.611156
G1 X132.336 Y100.005 E.00247
; LINE_WIDTH: 0.65067
G1 X132.373 Y99.973 E.00199
; LINE_WIDTH: 0.690183
G1 X132.41 Y99.94 E.00212
; LINE_WIDTH: 0.729696
G1 X132.447 Y99.907 E.00225
; LINE_WIDTH: 0.732096
G1 X133.004 Y99.996 E.02564
G1 X133.151 Y99.204 E.03663
G1 X132.633 Y99.194 E.02354
; LINE_WIDTH: 0.726326
G1 X132.461 Y99.161 E.00793
; LINE_WIDTH: 0.687936
G1 X132.288 Y99.128 E.00749
; LINE_WIDTH: 0.649546
G1 X132.116 Y99.095 E.00705
; LINE_WIDTH: 0.611156
G1 X131.823 Y99.02 E.01136
; LINE_WIDTH: 0.565576
G1 X131.53 Y98.945 E.01047
; LINE_WIDTH: 0.519996
G1 X131.039 Y98.936 E.01555
G3 X123.946 Y114.147 I-51.354 J-14.689 E.53358
G3 X121.747 Y117.121 I-28.177 J-18.531 E.11716
G1 X119.177 Y120.187 E.12664
G3 X112.01 Y126.704 I-40.869 J-37.746 E.30709
G1 X116.99 Y130.398 E.19631
G3 X117.739 Y131.094 I-6.427 J7.665 E.03239
G3 X118.212 Y131.719 I-29.046 J22.463 E.02481
G1 X118.565 Y132.362 E.02323
G1 X118.834 Y133.08 E.02425
G1 X118.981 Y133.764 E.02215
G1 X119.044 Y134.594 E.02637
G1 X118.98 Y135.358 E.02426
G1 X118.824 Y136.025 E.0217
G1 X118.548 Y136.753 E.02465
G1 X118.31 Y137.19 E.01574
G3 X130.249 Y144.067 I-20.137 J48.758 E.43747
G3 X142.39 Y157.025 I-31.645 J41.817 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.367 E1.81501
G1 X144.167 Y98.946 E.01331
G1 X143.706 Y98.969 E.01462
G1 X142.88 Y99.092 E.02644
G3 X141.717 Y99.097 I-1.511 J-219.333 E.03683
G1 X142.118 Y99.431 E.01653
G3 X142.228 Y99.733 I-.242 J.259 E.01057
G1 X142.115 Y100.392 E.02118
G1 X142.659 Y103.378 E.09609
G1 X142.631 Y103.532 E.00495
G1 X142.532 Y103.622 E.00423
G1 X141.709 Y103.642 E.02607
G3 X141.533 Y103.621 I.075 J-1.405 E.00559
G1 X141.386 Y103.602 E.0047
G1 X141.239 Y103.584 E.0047
G1 X141.091 Y103.565 E.0047
G1 X140.944 Y103.546 E.0047
G1 X140.797 Y103.528 E.0047
G1 X140.649 Y103.509 E.0047
G1 X140.502 Y103.49 E.0047
G1 F3000
G3 X139.879 Y103.393 I1.3 J-10.412 E.01996
G2 X139.454 Y103.466 I.101 J1.866 E.01369
G1 F3600
G1 X139.381 Y103.487 E.0024
G1 X139.308 Y103.508 E.0024
G1 X139.235 Y103.529 E.0024
G1 X139.162 Y103.55 E.0024
G1 X139.089 Y103.571 E.0024
G1 X139.016 Y103.591 E.0024
G1 X138.944 Y103.612 E.0024
G1 X138.871 Y103.633 E.0024
G3 X135.228 Y103.648 I-2.557 J-180.772 E.11532
G1 X133.228 Y103.642 E.06332
G1 X133.164 Y103.624 E.0021
G1 X133.1 Y103.606 E.0021
G1 X133.036 Y103.588 E.0021
G1 X132.972 Y103.57 E.0021
G1 X132.909 Y103.552 E.0021
G1 X132.845 Y103.534 E.0021
G1 X132.781 Y103.516 E.0021
G1 X132.717 Y103.498 E.0021
G1 F3000
G3 X132.313 Y103.37 I1.655 J-5.934 E.0134
G1 X132.218 Y103.251 E.00483
G3 X132.408 Y101.032 I62.578 J4.229 E.07051
G1 X132.396 Y100.973 E.0019
; LINE_WIDTH: 0.519986
G1 F3600
G1 X132.379 Y100.886 E.00279
G1 X132.362 Y100.8 E.0028
G1 X132.344 Y100.713 E.00279
G1 X132.327 Y100.627 E.00279
G1 X132.309 Y100.54 E.0028
G1 X132.292 Y100.454 E.00279
G1 X132.275 Y100.369 E.00274
; WIPE_START
M204 S10000
G1 X132.282 Y100.125 E-.09252
G1 X132.309 Y100.065 E-.02498
G1 X132.336 Y100.005 E-.02498
G1 X132.373 Y99.973 E-.01884
G1 X132.41 Y99.94 E-.01884
G1 X132.447 Y99.907 E-.01884
G1 X132.918 Y99.982 E-.18101
; WIPE_END
G1 E-.02 F1800
G1 X140.205 Y102.25 Z9.08 F36000
G1 X143.407 Y103.247 Z9.08
G1 Z8.68
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X143.407 Y103.588 E.02221
; WIPE_START
G1 X143.407 Y103.247 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X136.025 Y101.307 Z9.08 F36000
G1 X131.49 Y100.116 Z9.08
G1 Z8.68
G1 E.4 F1800
; LINE_WIDTH: 1.06292
G1 F7656.187
G1 X131.607 Y99.745 E.02605
; WIPE_START
G1 X131.49 Y100.116 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X128.839 Y107.273 Z9.08 F36000
G1 X125.888 Y115.238 Z9.08
G1 Z8.68
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X127.153 Y113.267 I-59.911 J-39.865 E.08944
G3 X128.922 Y114.798 I-9.012 J12.201 E.08939
G3 X130.232 Y119.04 I-3.946 J3.542 E.17488
G3 X129.298 Y120.453 I-2.902 J-.903 E.06558
G2 X127.082 Y122.339 I10.685 J14.8 E.11118
G2 X125.772 Y126.58 I3.946 J3.542 E.17488
G2 X126.707 Y127.994 I2.902 J-.903 E.06558
G3 X128.922 Y129.879 I-10.684 J14.8 E.11118
G3 X130.232 Y134.121 I-3.946 J3.542 E.17488
G3 X129.298 Y135.535 I-2.903 J-.903 E.06558
G2 X127.082 Y137.42 I10.683 J14.798 E.11118
G1 X126.4 Y138.362 E.04443
G1 X126.256 Y138.683 E.0134
G2 X121.111 Y135.983 I-37.209 J64.666 E.2219
G2 X121.222 Y133.81 I-8.299 J-1.515 E.08332
G2 X122.692 Y129.408 I-3.662 J-3.669 E.18369
G2 X121.757 Y127.994 I-2.902 J.903 E.06558
G3 X119.542 Y126.109 I10.684 J-14.8 E.11118
G3 X118.417 Y124.147 I5.309 J-4.346 E.08673
G2 X120.066 Y122.484 I-29.244 J-30.647 E.08945
G1 X124.76 Y119.226 F36000
; FEATURE: Floating vertical shell
; LINE_WIDTH: 0.619996
G1 F12000
G1 X124.565 Y118.696 E.02156
G1 X124.179 Y118.184 E.02448
G1 X123.995 Y118.023 E.00936
G3 X120.837 Y121.79 I-300.256 J-248.466 E.18769
G1 X121.419 Y122.159 E.02631
G1 X121.902 Y122.317 E.01939
G1 X122.396 Y122.331 E.01889
G1 X122.702 Y122.275 E.01187
G1 X123.164 Y122.094 E.01892
G2 X123.787 Y121.581 I-1.127 J-2.004 E.03098
G1 X124.421 Y120.819 E.03786
G1 X124.665 Y120.374 E.01938
G2 X124.812 Y119.598 I-2.62 J-.899 E.03027
G1 X124.772 Y119.315 E.01087
G1 X124.101 Y119.319 F36000
; LINE_WIDTH: 0.78707
G1 F10461.863
G1 X123.999 Y119.066 E.0134
G3 X121.869 Y121.612 I-1024.11 J-854.473 E.1629
G1 X122.295 Y121.67 E.02108
G1 X122.783 Y121.543 E.02477
G1 X123.049 Y121.371 E.01552
G1 X123.847 Y120.453 E.05974
G1 X124.071 Y120.064 E.022
G1 X124.136 Y119.566 E.02463
G1 X124.114 Y119.408 E.00784
G1 X141.945 Y145.543 F36000
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X141.945 Y143.2 E.08944
G3 X140.853 Y141.662 I1.81 J-2.441 E.07325
G3 X141.945 Y137.721 I5.105 J-.707 E.16053
G1 X141.945 Y128.119 E.3666
G3 X140.853 Y126.58 I1.81 J-2.441 E.07325
G3 X141.945 Y122.64 I5.105 J-.707 E.16053
G1 X141.945 Y113.038 E.3666
G3 X140.853 Y111.499 I1.81 J-2.441 E.07325
G3 X141.945 Y107.559 I5.105 J-.707 E.16053
G1 X141.945 Y105.859 E.06488
G3 X139.911 Y105.647 I-.019 J-9.659 E.07824
G3 X138.825 Y105.864 I-1.31 J-3.734 E.04244
G1 X133.777 Y105.862 E.19271
G2 X133.313 Y106.786 I1.793 J1.48 E.03981
G2 X134.623 Y111.028 I5.256 J.699 E.17488
G2 X136.838 Y112.913 I12.898 J-12.913 E.11118
G3 X137.773 Y114.327 I-1.968 J2.317 E.06558
G3 X136.463 Y118.568 I-5.256 J.699 E.17488
G3 X134.247 Y120.453 I-12.899 J-12.915 E.11118
G2 X133.313 Y121.867 I1.968 J2.317 E.06558
G2 X134.623 Y126.109 I5.256 J.699 E.17488
G2 X136.838 Y127.994 I12.898 J-12.913 E.11118
G3 X137.773 Y129.408 I-1.968 J2.317 E.06558
G3 X136.463 Y133.65 I-5.256 J.699 E.17488
G3 X134.247 Y135.535 I-12.901 J-12.916 E.11118
G2 X133.313 Y136.949 I1.968 J2.317 E.06558
G2 X134.623 Y141.19 I5.256 J.699 E.17488
G2 X136.838 Y143.075 I12.9 J-12.915 E.11118
G3 X137.601 Y144.018 I-1.973 J2.377 E.0466
G3 X137.351 Y147.329 I-3.911 J1.37 E.13044
G3 X138.97 Y149.022 I-30.874 J31.135 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 8.84
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X138.279 Y148.299 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L55
M991 S0 P54 ;notify layer change


G17
G3 Z9.08 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X142.443 Y105.267
G1 Z8.84
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.09 Y105.287 E.01351
G3 X140.035 Y105.003 I4.59 J-40.814 E.0792
G1 X139.771 Y105.006 E.01007
G3 X138.721 Y105.342 I-4.897 J-13.521 E.04212
G1 X138.409 Y105.368 E.01196
G3 X135.97 Y105.374 I-2.8 J-606.912 E.09309
G1 X133.97 Y105.368 E.07636
G3 X131.967 Y104.854 I1.104 J-8.473 E.07915
G1 X131.257 Y104.409 E.03199
G1 X131.094 Y104.185 E.01057
G3 X125.361 Y115.131 I-51.711 J-20.11 E.47276
G3 X123.072 Y118.225 I-29.869 J-19.713 E.14699
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-40.697 J-37.17 E.29803
G1 X118.015 Y129.012 E.15054
G3 X118.932 Y129.849 I-4.61 J5.972 E.04748
G3 X120.076 Y131.531 I-5.878 J5.226 E.07789
G1 X120.45 Y132.479 E.03891
G3 X120.765 Y134.489 I-7.446 J2.194 E.07788
G1 X120.698 Y135.504 E.03883
G1 X120.538 Y136.268 E.0298
G3 X131.286 Y142.69 I-21.792 J48.675 E.4791
G3 X142.443 Y154.069 I-32.801 J43.322 E.61065
G1 X142.443 Y105.357 E1.85978
; WIPE_START
G1 X142.09 Y105.287 E-.13689
G1 X141.456 Y105.2 E-.24311
; WIPE_END
G1 E-.02 F1800
G1 X143.029 Y104.479 Z9.24 F36000
G1 Z8.84
G1 E.4 F1800
G1 F13446.369
G1 X142.671 Y104.678 E.0156
G1 X142.166 Y104.706 E.01935
G3 X139.897 Y104.389 I6.238 J-52.804 E.08746
G2 X138.87 Y104.699 I.673 J4.082 E.04109
G1 X138.406 Y104.782 E.01798
G3 X135.957 Y104.789 I-2.738 J-590.242 E.09353
G1 X133.957 Y104.782 E.07636
G3 X132.165 Y104.303 I1.326 J-8.553 E.07096
G1 X131.668 Y103.991 E.02238
G1 X131.386 Y103.589 E.01876
G1 X131.261 Y103.066 E.02052
G3 X131.367 Y101.712 I45.109 J2.848 E.05186
G3 X124.88 Y114.797 I-51.574 J-17.419 E.55928
G3 X122.623 Y117.848 I-29.318 J-19.328 E.14498
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-39.645 J-36.158 E.32257
G1 X117.677 Y129.49 E.17987
G3 X118.523 Y130.268 I-5.659 J7.007 E.04392
G3 X119.21 Y131.171 I-7.804 J6.644 E.04333
G1 X119.563 Y131.814 E.02799
G1 X119.901 Y132.683 E.03563
G3 X120.135 Y133.79 I-9.585 J2.604 E.04323
G1 X120.18 Y134.525 E.02809
G1 X120.115 Y135.453 E.03554
G1 X119.931 Y136.289 E.03267
G1 X119.836 Y136.599 E.01238
G3 X130.932 Y143.157 I-21.039 J48.268 E.49335
G3 X142.92 Y155.769 I-32.373 J42.775 E.66728
G1 X143.029 Y155.749 E.00421
G1 X143.029 Y104.569 E1.95397
G1 X142.83 Y103.943 F36000
G1 F13446.369
G1 X142.53 Y104.109 E.01309
G1 X142.241 Y104.126 E.01104
G3 X140.217 Y103.846 I4.383 J-39.227 E.07802
G1 X139.891 Y103.795 E.0126
G1 X139.397 Y103.883 E.01918
G1 X138.669 Y104.149 E.02959
G1 X138.404 Y104.197 E.01026
G3 X135.965 Y104.203 I-2.763 J-596.576 E.09312
G1 X133.965 Y104.197 E.07636
G1 X132.999 Y103.976 E.03785
G1 X132.362 Y103.752 E.02577
G1 X132.067 Y103.562 E.0134
G1 X131.918 Y103.344 E.01008
G1 X131.849 Y102.952 E.01519
G1 X132.008 Y101.094 E.07123
; LINE_WIDTH: 0.620576
G1 F13433.066
G1 X131.872 Y100.418 E.02632
G1 X131.891 Y100.048 E.01418
; LINE_WIDTH: 0.637966
G1 F13046.072
G1 X132.072 Y99.644 E.01739
G1 X131.998 Y99.592 E.00355
; LINE_WIDTH: 0.619876
G1 F13449.125
G1 X131.451 Y99.493 E.02123
G1 X131.307 Y99.983 E.0195
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X124.399 Y114.463 I-51.181 J-15.527 E.61483
G3 X122.174 Y117.472 I-28.764 J-18.942 E.14297
G1 X119.604 Y120.538 E.15272
G3 X112.93 Y126.698 I-40.081 J-36.732 E.34719
G1 X117.323 Y129.957 E.20885
G3 X118.151 Y130.725 I-6.59 J7.927 E.04314
G3 X118.696 Y131.453 I-12.352 J9.829 E.03474
G1 X119.049 Y132.096 E.02799
G1 X119.352 Y132.887 E.03235
G3 X119.551 Y133.826 I-11.698 J2.963 E.03666
G1 X119.595 Y134.56 E.02809
G1 X119.531 Y135.403 E.03226
G1 X119.361 Y136.154 E.02939
G1 X119.08 Y136.91 E.03081
G3 X130.579 Y143.624 I-20.72 J48.692 E.50971
G3 X142.648 Y156.415 I-31.963 J42.246 E.67454
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y105.424 E1.93993
G1 X143.615 Y105.024 E.01527
G1 F13158.166
G1 X143.615 Y104.624 E.01527
G1 F11774.914
G1 X143.615 Y104.224 E.01527
; LINE_WIDTH: 0.666318
G1 F10468.462
G1 X143.591 Y104.161 E.00276
; LINE_WIDTH: 0.71264
G1 F10257.041
G1 X143.568 Y104.098 E.00296
; LINE_WIDTH: 0.758962
G1 F10047.788
G1 X143.545 Y104.035 E.00317
; LINE_WIDTH: 0.805283
G1 F9840.691
G1 X143.522 Y103.973 E.00337
; LINE_WIDTH: 0.851605
G1 F9635.75
G1 X143.499 Y103.91 E.00357
; LINE_WIDTH: 0.897927
G1 F9118.902
G1 X143.476 Y103.847 E.00377
; LINE_WIDTH: 0.944249
G1 F8654.677
G1 X143.452 Y103.784 E.00398
; LINE_WIDTH: 0.990571
G1 F8235.428
G1 X143.429 Y103.721 E.00418
; LINE_WIDTH: 1.03689
G1 F7854.92
G1 X143.406 Y103.658 E.00438
; LINE_WIDTH: 1.08321
G1 F7508.021
G1 X143.383 Y103.595 E.00458
; LINE_WIDTH: 1.12954
G1 F7190.468
G1 X143.36 Y103.532 E.00479
G1 X143.316 Y103.567 E.00396
; LINE_WIDTH: 1.08321
G1 F7508.021
G1 X143.273 Y103.602 E.0038
; LINE_WIDTH: 1.03689
G1 F7854.92
G1 X143.23 Y103.636 E.00363
; LINE_WIDTH: 0.990571
G1 F8235.428
G1 X143.186 Y103.671 E.00346
; LINE_WIDTH: 0.944249
G1 F8654.677
G1 X143.143 Y103.706 E.00329
; LINE_WIDTH: 0.897927
G1 F9118.902
G1 X143.1 Y103.74 E.00313
; LINE_WIDTH: 0.851605
G1 F9635.75
G1 X143.056 Y103.775 E.00296
; LINE_WIDTH: 0.805283
G1 F10214.707
G1 X143.013 Y103.81 E.00279
; LINE_WIDTH: 0.758962
G1 F10389.271
G1 X142.97 Y103.844 E.00262
; LINE_WIDTH: 0.71264
G1 F10565.314
G1 X142.926 Y103.879 E.00245
; LINE_WIDTH: 0.666318
G1 F10648.519
G1 X142.906 Y103.895 E.00107
; WIPE_START
G1 X142.53 Y104.109 E-.16434
G1 X142.241 Y104.126 E-.10989
G1 X141.966 Y104.088 E-.10577
; WIPE_END
G1 E-.02 F1800
G1 X143.377 Y102.91 Z9.24 F36000
G1 Z8.84
G1 E.4 F1800
; LINE_WIDTH: 1.0952
G1 F7423.223
G1 X143.393 Y102.647 E.01825
; LINE_WIDTH: 1.06362
G1 F7650.98
G1 X143.409 Y102.384 E.0177
; LINE_WIDTH: 1.03204
G1 F7893.154
G1 X143.432 Y101.984 E.02602
; LINE_WIDTH: 0.984158
G1 F8291.027
G1 X143.456 Y101.585 E.02477
; LINE_WIDTH: 0.93628
G1 F8731.142
G1 X143.48 Y101.186 E.02352
; LINE_WIDTH: 0.888402
G1 F9220.599
G1 X143.504 Y100.786 E.02227
; LINE_WIDTH: 0.840524
G1 F9768.194
G1 X143.528 Y100.387 E.02102
; LINE_WIDTH: 0.792646
G1 F10384.936
G1 X143.536 Y99.987 E.01977
G1 X143.543 Y99.626 E.01787
; LINE_WIDTH: 0.763846
G1 F10794.918
G2 X142.829 Y99.746 I.343 J4.21 E.03444
; LINE_WIDTH: 0.792646
G1 F10384.936
G2 X142.776 Y100.455 I3.56 J.625 E.0352
; LINE_WIDTH: 0.824221
G1 F9969.806
G1 X142.839 Y100.711 E.01359
; LINE_WIDTH: 0.855796
G1 F9586.59
G1 X142.901 Y100.968 E.01413
; LINE_WIDTH: 0.903676
G1 F9058.596
G1 X142.997 Y101.356 E.02267
; LINE_WIDTH: 0.951556
G1 F8585.727
G1 X143.092 Y101.745 E.02392
; LINE_WIDTH: 0.999436
G1 F8159.776
G1 X143.187 Y102.133 E.02517
; LINE_WIDTH: 1.04732
G1 F7774.092
G1 X143.282 Y102.522 E.02641
; LINE_WIDTH: 1.0952
G1 F7423.223
G1 X143.356 Y102.823 E.02144
; WIPE_START
G1 X143.393 Y102.647 E-.06833
G1 X143.409 Y102.384 E-.10026
G1 X143.432 Y101.984 E-.152
G1 X143.442 Y101.828 E-.05941
; WIPE_END
G1 E-.02 F1800
G1 X142.499 Y103.516 Z9.24 F36000
G1 Z8.84
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G3 X142.137 Y103.555 I-.221 J-.356 E.0119
G1 X141.962 Y103.532 E.0056
G1 X141.787 Y103.509 E.0056
G1 X141.611 Y103.486 E.0056
G1 X141.436 Y103.463 E.0056
G1 X141.261 Y103.44 E.0056
G1 X141.085 Y103.417 E.0056
G1 X140.91 Y103.394 E.0056
G1 F3000
G3 X139.886 Y103.234 I2.512 J-19.447 E.03282
G1 X139.203 Y103.365 E.022
G1 F3600
G1 X139.112 Y103.398 E.00305
G1 X139.022 Y103.431 E.00305
G1 X138.931 Y103.464 E.00305
G1 X138.841 Y103.498 E.00305
G1 X138.75 Y103.531 E.00305
G1 X138.66 Y103.564 E.00305
G1 X138.569 Y103.597 E.00305
G3 X138.402 Y103.644 I-.217 J-.454 E.00552
G3 X135.973 Y103.65 I-2.777 J-598.435 E.0769
G1 X133.973 Y103.644 E.06332
G1 X133.883 Y103.622 E.00294
G1 X133.793 Y103.6 E.00294
G1 X133.703 Y103.578 E.00294
G1 X133.612 Y103.556 E.00294
G1 X133.522 Y103.534 E.00294
G1 X133.432 Y103.512 E.00294
G1 X133.342 Y103.49 E.00294
G1 X133.252 Y103.468 E.00294
G1 F3000
G1 X133.161 Y103.446 E.00294
G1 F2475
G1 X132.549 Y103.231 E.02056
G1 X132.464 Y103.154 E.00364
G1 F3000
G1 X132.42 Y103.113 E.0019
G1 X132.4 Y103 E.00365
G1 X132.566 Y101.062 E.06157
G1 X132.554 Y101.002 E.00193
; LINE_WIDTH: 0.519986
G1 F3600
G1 X132.536 Y100.916 E.0028
G1 X132.519 Y100.829 E.0028
G1 X132.502 Y100.743 E.0028
G1 X132.484 Y100.656 E.0028
G1 X132.467 Y100.569 E.0028
G1 X132.449 Y100.483 E.0028
G1 X132.432 Y100.396 E.0028
; LINE_WIDTH: 0.519996
G3 X132.43 Y100.173 I.348 J-.115 E.00718
; LINE_WIDTH: 0.563396
G1 X132.465 Y100.125 E.00205
; LINE_WIDTH: 0.606796
G1 X132.5 Y100.077 E.00222
; LINE_WIDTH: 0.650196
G1 X132.535 Y100.029 E.00239
; LINE_WIDTH: 0.693596
G1 X132.57 Y99.981 E.00256
; LINE_WIDTH: 0.736996
G1 X132.605 Y99.933 E.00272
G1 X133.002 Y99.996 E.01843
; LINE_WIDTH: 0.731986
G1 X133.148 Y99.206 E.03651
; LINE_WIDTH: 0.736996
G1 X132.6 Y99.198 E.02513
G1 X132.499 Y99.168 E.00481
; LINE_WIDTH: 0.693596
G1 X132.399 Y99.138 E.00451
; LINE_WIDTH: 0.650196
G1 X132.298 Y99.108 E.00421
; LINE_WIDTH: 0.606796
G1 X132.198 Y99.078 E.00392
; LINE_WIDTH: 0.563396
G1 X132.097 Y99.048 E.00362
; LINE_WIDTH: 0.519996
G1 X131.483 Y98.936 E.01977
G1 X131.038 Y98.936 E.01409
G3 X123.946 Y114.147 I-51.151 J-14.591 E.53358
G3 X121.747 Y117.121 I-28.178 J-18.531 E.11716
G1 X119.177 Y120.187 E.12664
G3 X112.01 Y126.704 I-38.874 J-35.552 E.30714
G1 X116.989 Y130.398 E.19629
G3 X117.738 Y131.094 I-6.439 J7.681 E.03239
G3 X118.212 Y131.719 I-29.099 J22.508 E.02485
G1 X118.565 Y132.362 E.02321
G1 X118.834 Y133.079 E.02425
G1 X118.953 Y133.631 E.01787
G1 X119.044 Y134.594 E.03063
G1 X118.98 Y135.355 E.02418
G1 X118.824 Y136.026 E.02181
G1 X118.548 Y136.753 E.02463
G1 X118.31 Y137.19 E.01574
G3 X130.249 Y144.067 I-19.584 J47.797 E.43752
G3 X142.39 Y157.025 I-31.645 J41.817 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.365 E1.81507
G1 X144.167 Y98.946 E.01326
G1 X143.708 Y98.969 E.01457
G1 X142.884 Y99.093 E.02637
G3 X141.72 Y99.099 I-1.473 J-183.185 E.03687
G1 X141.982 Y99.339 E.01126
G1 X142.208 Y99.806 E.01642
G1 X142.115 Y100.392 E.01879
G1 X142.599 Y103.052 E.08559
G1 X142.564 Y103.393 E.01085
G1 X142.541 Y103.437 E.00157
M204 S10000
G1 X143.377 Y102.91 F36000
; FEATURE: Inner wall
; LINE_WIDTH: 1.12954
G1 F7190.468
G1 X143.36 Y103.532 E.04442
; WIPE_START
G1 X143.377 Y102.91 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X137.148 Y107.321 Z9.24 F36000
G1 X125.816 Y115.347 Z9.24
G1 Z8.84
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X127.086 Y113.379 I-26.157 J-18.279 E.08946
M73 P81 R3
G3 X128.856 Y114.798 I-9.132 J13.2 E.08669
G3 X130.422 Y119.04 I-4.066 J3.91 E.17761
G3 X129.55 Y120.453 I-2.479 J-.552 E.06459
G2 X127.148 Y122.339 I11.575 J17.222 E.11669
G2 X125.583 Y126.58 I4.066 J3.911 E.17761
G2 X126.454 Y127.994 I2.479 J-.552 E.06459
G3 X128.856 Y129.879 I-11.571 J17.218 E.11669
G3 X130.422 Y134.121 I-4.066 J3.91 E.17761
G3 X129.55 Y135.535 I-2.479 J-.552 E.06459
G2 X127.148 Y137.42 I11.573 J17.22 E.11669
G1 X126.404 Y138.362 E.04585
G1 X126.248 Y138.675 E.01332
G2 X121.111 Y135.983 I-41.482 J72.928 E.22147
G2 X121.218 Y133.74 I-8.35 J-1.523 E.08601
G1 X121.315 Y133.65 E.00507
G2 X122.881 Y129.408 I-4.065 J-3.91 E.17761
G2 X122.01 Y127.994 I-2.479 J.552 E.06459
G3 X119.608 Y126.109 I11.571 J-17.218 E.11669
G3 X118.379 Y124.18 I5.523 J-4.875 E.08766
G2 X120.03 Y122.519 I-22.848 J-24.351 E.08945
G1 X123.304 Y121.961 F36000
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.781 Y121.582 E.02325
G1 X124.318 Y120.942 E.03191
G1 X124.56 Y120.489 E.01961
G2 X124.699 Y119.837 I-1.904 J-.747 E.02555
G1 X124.657 Y119.298 E.02064
G1 X124.47 Y118.812 E.01987
G1 X124.155 Y118.381 E.02039
G1 X123.893 Y118.152 E.01329
G1 X120.948 Y121.669 E.17515
G1 X121.182 Y121.853 E.01136
G1 X121.663 Y122.104 E.02073
G1 X122.194 Y122.212 E.0207
G1 X122.498 Y122.204 E.0116
G1 X123.003 Y122.09 E.01974
G1 X123.221 Y121.996 E.00908
G1 X122.99 Y121.347 F36000
; LINE_WIDTH: 0.792764
G1 F10383.326
G1 X123.332 Y121.066 E.02189
G1 X123.764 Y120.55 E.03326
G2 X123.899 Y119.184 I-1.055 J-.793 E.07136
G2 X121.983 Y121.485 I272.029 J228.422 E.14801
G1 X122.384 Y121.538 E.02001
G1 X122.736 Y121.47 E.0177
G1 X122.909 Y121.386 E.00953
G1 X141.945 Y145.72 F36000
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X141.945 Y143.378 E.08944
G3 X140.664 Y141.662 I1.485 J-2.445 E.08381
G3 X141.945 Y137.78 I5.459 J-.35 E.15995
G1 X141.945 Y128.296 E.36207
G3 X140.664 Y126.58 I1.485 J-2.445 E.08381
G3 X141.945 Y122.699 I5.459 J-.35 E.15995
G1 X141.945 Y113.215 E.36207
G3 X140.664 Y111.499 I1.485 J-2.445 E.08381
G3 X141.945 Y107.617 I5.459 J-.35 E.15995
G1 X141.945 Y105.77 E.07051
G3 X139.908 Y105.487 I4.588 J-40.494 E.07854
G3 X138.418 Y105.866 I-1.594 J-3.154 E.05916
G1 X133.874 Y105.864 E.17349
G1 X133.559 Y105.819 E.01216
G2 X133.123 Y106.786 I2.304 J1.62 E.04075
G2 X134.689 Y111.028 I5.631 J.331 E.17761
G2 X137.091 Y112.913 I13.972 J-15.331 E.11669
G3 X137.962 Y114.327 I-1.608 J1.966 E.06459
G3 X136.397 Y118.568 I-5.631 J.331 E.17761
G3 X133.994 Y120.453 I-13.973 J-15.333 E.11669
G2 X133.123 Y121.867 I1.608 J1.966 E.06459
G2 X134.689 Y126.109 I5.631 J.331 E.17761
G2 X137.091 Y127.994 I13.972 J-15.331 E.11669
G3 X137.962 Y129.408 I-1.608 J1.966 E.06459
G3 X136.397 Y133.65 I-5.631 J.331 E.17761
G3 X133.994 Y135.535 I-13.975 J-15.335 E.11669
G2 X133.123 Y136.949 I1.607 J1.966 E.06459
G2 X134.689 Y141.19 I5.631 J.331 E.17761
G2 X137.091 Y143.075 I13.974 J-15.334 E.11669
G3 X137.962 Y144.489 I-1.608 J1.966 E.06459
G3 X137.366 Y147.338 I-4.649 J.514 E.11301
G3 X138.981 Y149.035 I-52.452 J51.551 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 9
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X138.292 Y148.311 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L56
M991 S0 P55 ;notify layer change


G17
G3 Z9.24 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.194 Y103.929
G1 Z9
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.131 I-51.877 J-19.894 E.48325
G3 X123.072 Y118.225 I-29.875 J-19.717 E.147
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-42.064 J-38.609 E.29801
G1 X118.015 Y129.012 E.15055
G1 X118.485 Y129.402 E.02332
G1 X119.092 Y130.035 E.03349
G3 X119.749 Y130.931 I-4.552 J4.025 E.04247
G1 X120.215 Y131.838 E.03892
G1 X120.485 Y132.59 E.03052
G1 X120.683 Y133.478 E.03474
G3 X120.698 Y135.504 I-7.96 J1.073 E.07756
G1 X120.538 Y136.268 E.02981
G3 X130.503 Y142.098 I-22.401 J49.722 E.44162
G3 X142.443 Y154.069 I-32.641 J44.497 E.64805
G1 X142.443 Y105.096 E1.86973
G1 X141.893 Y105.127 E.02102
G3 X139.891 Y104.827 I4.8 J-38.872 E.07732
G1 X139.693 Y104.86 E.00765
G2 X138.498 Y105.309 I8.383 J24.157 E.04874
G1 X138.02 Y105.37 E.01839
G3 X134.669 Y105.37 I-1.676 J-255.131 E.12794
G3 X133.561 Y105.142 I.538 J-5.437 E.04328
G3 X131.932 Y104.599 I2.898 J-11.398 E.06563
G1 X131.518 Y104.32 E.01905
G1 X131.252 Y103.998 E.01594
; WIPE_START
G1 X130.824 Y104.902 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X131.265 Y102.611 Z9.4 F36000
G1 Z9
G1 E.4 F1800
; LINE_WIDTH: 1.01466
G1 F8033.091
G1 X131.194 Y102.735 E.00915
; LINE_WIDTH: 0.965324
G1 F8458.761
G1 X131.122 Y102.859 E.00869
; LINE_WIDTH: 0.915991
G1 F8932.066
G1 X131.051 Y102.983 E.00823
; LINE_WIDTH: 0.866659
G1 F9461.478
G1 X130.979 Y103.107 E.00777
; LINE_WIDTH: 0.817326
G1 F10057.601
G1 X130.908 Y103.231 E.00731
; LINE_WIDTH: 0.767994
G1 F10507.415
G1 X130.836 Y103.355 E.00685
; LINE_WIDTH: 0.718661
G1 F10967.069
G1 X130.764 Y103.479 E.00639
; LINE_WIDTH: 0.669329
G1 F11436.562
G1 X130.693 Y103.603 E.00593
; LINE_WIDTH: 0.619996
G1 F12800.351
G1 X130.547 Y103.976 E.01527
G1 F13446.369
G1 X130.4 Y104.348 E.01527
G1 X130.208 Y104.835 E.01998
G3 X124.88 Y114.797 I-50.389 J-20.543 E.4321
G3 X122.623 Y117.848 I-29.321 J-19.33 E.14499
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-40.18 J-36.729 E.32256
G1 X117.666 Y129.482 E.17936
G1 X118.107 Y129.85 E.02194
G1 X118.663 Y130.434 E.03075
G1 X119.253 Y131.244 E.0383
G1 X119.688 Y132.095 E.03646
G3 X120.115 Y135.454 I-6.166 J2.489 E.13071
G1 X119.933 Y136.283 E.03243
G1 X119.836 Y136.599 E.01261
G3 X130.932 Y143.157 I-21.104 J48.378 E.49336
G3 X142.92 Y155.769 I-32.176 J42.588 E.6673
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y104.213 E1.9676
G1 X142.478 Y104.518 E.02406
G1 X141.97 Y104.546 E.0194
G3 X139.939 Y104.241 I4.892 J-39.484 E.07843
G1 X139.566 Y104.284 E.01434
G2 X138.353 Y104.742 I6.73 J19.677 E.04952
G1 X138.018 Y104.784 E.01287
G3 X134.671 Y104.784 I-1.674 J-254.532 E.1278
G3 X132.345 Y104.14 I2.201 J-12.47 E.09229
G1 X131.906 Y103.88 E.01949
G1 F12791.56
G1 X131.62 Y103.528 E.01731
; LINE_WIDTH: 0.668221
G1 F11252.329
G1 X131.567 Y103.402 E.00566
; LINE_WIDTH: 0.716446
G1 F10806.974
G1 X131.514 Y103.276 E.00609
; LINE_WIDTH: 0.764671
G1 F10370.611
G1 X131.461 Y103.15 E.00652
; LINE_WIDTH: 0.812896
G1 F9943.225
G1 X131.407 Y103.024 E.00695
; LINE_WIDTH: 0.861121
G1 F9524.847
G1 X131.354 Y102.898 E.00738
; LINE_WIDTH: 0.909346
G1 F8999.898
G1 X131.301 Y102.771 E.00781
; LINE_WIDTH: 0.94445
G1 F8652.769
G1 X131.289 Y102.718 E.00325
; LINE_WIDTH: 0.979553
G1 F8331.425
G1 X131.285 Y102.699 E.0012
; WIPE_START
G1 X131.194 Y102.735 E-.03725
G1 X131.122 Y102.859 E-.05441
G1 X131.051 Y102.983 E-.05441
G1 X130.979 Y103.107 E-.05441
G1 X130.908 Y103.231 E-.05441
G1 X130.836 Y103.355 E-.05441
G1 X130.764 Y103.479 E-.05441
G1 X130.743 Y103.516 E-.0163
; WIPE_END
G1 E-.02 F1800
G1 X132.225 Y99.752 Z9.4 F36000
G1 Z9
G1 E.4 F1800
; LINE_WIDTH: 0.695276
G1 F11914.85
G1 X132.371 Y99.665 E.00733
G3 X131.478 Y99.536 I.687 J-7.899 E.03892
; LINE_WIDTH: 0.701506
G1 F11803.589
G3 X131.162 Y100.511 I-7.34 J-1.837 E.04462
; LINE_WIDTH: 0.660751
G1 F12571.54
G1 X130.991 Y100.983 E.02047
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X130.863 Y101.361 E.01527
G1 X130.481 Y102.474 E.04489
G3 X124.4 Y114.462 I-50.633 J-18.148 E.51459
G3 X122.174 Y117.472 I-28.764 J-18.942 E.14298
G1 X119.604 Y120.538 E.15272
G3 X112.93 Y126.698 I-39.844 J-36.475 E.34719
G1 X117.325 Y129.958 E.20892
G1 X117.73 Y130.298 E.02021
G1 X118.234 Y130.832 E.02801
G1 X118.771 Y131.576 E.03502
G1 X119.162 Y132.352 E.03318
G3 X119.531 Y135.403 I-5.672 J2.234 E.11864
G1 X119.363 Y136.148 E.02915
G1 X119.08 Y136.91 E.03104
G3 X130.579 Y143.624 I-20.394 J48.134 E.50975
G3 X142.648 Y156.415 I-31.78 J42.073 E.67456
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y104.544 E1.97351
G1 X143.615 Y104.144 E.01527
; LINE_WIDTH: 0.649416
G1 F12803.213
G1 X143.6 Y103.767 E.01516
; LINE_WIDTH: 0.678836
G1 F12218.776
G3 X143.599 Y102.967 I5.561 J-.406 E.03361
; LINE_WIDTH: 0.651161
G1 F12766.993
G1 X143.613 Y102.546 E.01696
; LINE_WIDTH: 0.627696
G1 F13271.876
G1 X143.611 Y102.213 E.01286
; LINE_WIDTH: 0.668934
G1 F12409.442
G1 X143.59 Y101.757 E.01891
; LINE_WIDTH: 0.710171
G1 F11652.254
G1 X143.569 Y101.3 E.02013
; LINE_WIDTH: 0.751409
G1 F10982.154
G1 X143.549 Y100.844 E.02136
; LINE_WIDTH: 0.792646
G1 F10384.936
G3 X143.541 Y99.629 I9.883 J-.675 E.06008
; LINE_WIDTH: 0.768046
G1 F10733.124
G3 X142.762 Y99.725 I-.844 J-3.656 E.03759
G1 X142.808 Y100.034 E.01493
; LINE_WIDTH: 0.792646
G1 F10384.936
G1 X142.776 Y100.455 E.0209
G1 X142.838 Y100.908 E.02259
; LINE_WIDTH: 0.751409
G1 F10982.154
G1 X142.899 Y101.361 E.02136
; LINE_WIDTH: 0.710171
G1 F11652.254
G1 X142.961 Y101.814 E.02013
; LINE_WIDTH: 0.668934
G1 F12409.442
G1 X143.022 Y102.266 E.01891
; LINE_WIDTH: 0.675086
G1 F12290.286
G1 X142.949 Y103.298 E.0432
G1 X142.82 Y103.526 E.01092
; LINE_WIDTH: 0.647541
G1 F12842.363
G1 X142.691 Y103.753 E.01045
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.337 Y103.949 E.01546
G1 X142.047 Y103.966 E.01107
G3 X139.988 Y103.655 I4.912 J-39.611 E.07954
G1 X139.518 Y103.695 E.01798
G2 X138.314 Y104.138 I2.713 J9.231 E.04901
G1 X138.016 Y104.199 E.01161
G3 X134.673 Y104.199 I-1.671 J-254.167 E.12765
G3 X133.406 Y103.896 I6.853 J-31.481 E.04975
G1 X132.544 Y103.589 E.03494
G1 X132.293 Y103.441 E.01113
G1 X132.13 Y103.24 E.00988
G1 X132.039 Y102.734 E.01961
G1 X132.135 Y101.525 E.0463
G1 X132.166 Y101.127 E.01527
; LINE_WIDTH: 0.647576
G1 F12841.629
G1 X132.103 Y100.841 E.01171
G1 X132.016 Y100.45 E.01599
; LINE_WIDTH: 0.697476
G1 F11875.321
G1 X131.99 Y100.147 E.01317
G1 X132.097 Y99.869 E.01286
; LINE_WIDTH: 0.695276
G1 F11914.85
G1 X132.143 Y99.801 E.00354
G1 X132.148 Y99.799 E.00024
M204 S250
G1 X132.613 Y100.158 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X132.76 Y100.071 E.00538
G1 X133.085 Y100.119 E.01041
G1 X133.273 Y99.102 E.03273
G3 X131.529 Y98.945 I-.196 J-7.57 E.05557
G1 X131.038 Y98.936 E.01556
G3 X123.946 Y114.147 I-50.893 J-14.471 E.53359
G3 X121.747 Y117.121 I-28.179 J-18.532 E.11716
G1 X119.177 Y120.186 E.12664
G3 X112.01 Y126.704 I-40.867 J-37.744 E.3071
G1 X116.99 Y130.398 E.19631
G3 X117.828 Y131.208 I-2.919 J3.861 E.037
G1 X118.315 Y131.888 E.02647
G3 X118.995 Y133.828 I-4.876 J2.798 E.06544
G1 X119.035 Y134.451 E.01975
G1 X118.98 Y135.356 E.0287
G1 X118.825 Y136.02 E.02161
G1 X118.547 Y136.754 E.02483
G1 X118.31 Y137.19 E.01573
G3 X130.249 Y144.067 I-19.805 J48.181 E.43751
G3 X142.39 Y157.025 I-31.646 J41.817 E.56491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.363 E1.81512
G1 X144.167 Y98.946 E.0132
G1 X143.709 Y98.97 E.01453
G1 X142.883 Y99.095 E.02645
G1 X141.734 Y99.111 E.03638
G1 X142 Y99.593 E.01745
G1 X142.156 Y99.876 E.01023
G1 X142.165 Y99.929 E.0017
G1 X142.179 Y100.02 E.0029
G1 X142.115 Y100.392 E.01197
G1 X142.474 Y102.366 E.06352
G1 X142.46 Y102.485 E.00379
G1 X142.446 Y102.604 E.00379
G1 X142.432 Y102.723 E.00379
G1 X142.418 Y102.841 E.00379
G1 X142.404 Y102.96 E.00379
G1 F3450
G1 X142.39 Y103.079 E.00379
G1 F3300
G1 X142.376 Y103.198 E.00379
G1 F3150
G3 X142.368 Y103.235 I-.056 J.006 E.00124
G1 F3300
G1 X142.34 Y103.289 E.00194
G1 F3450
G1 X142.313 Y103.344 E.00194
G1 F3600
G3 X142.276 Y103.366 I-.029 J-.007 E.00149
G1 F3450
G1 X142.137 Y103.412 E.00464
G1 F3300
G1 X141.636 Y103.353 E.01598
G1 F3150
G3 X140.915 Y103.249 I1.221 J-10.989 E.02307
G1 F3000
G2 X139.649 Y103.103 I-1.314 J5.806 E.04042
G2 X138.556 Y103.438 I1.866 J8.033 E.03623
G1 F3600
G1 X138.499 Y103.462 E.00195
G1 X138.442 Y103.485 E.00195
G1 X138.386 Y103.509 E.00195
G1 X138.329 Y103.533 E.00195
G1 X138.272 Y103.557 E.00195
G1 X138.215 Y103.581 E.00195
G1 X138.158 Y103.605 E.00195
G3 X138.015 Y103.646 I-.175 J-.34 E.00475
G3 X134.675 Y103.646 I-1.669 J-253.601 E.10574
G1 X134.604 Y103.628 E.00232
G1 X134.533 Y103.61 E.00232
G1 X134.462 Y103.592 E.00232
G1 X134.391 Y103.574 E.00232
G1 X134.32 Y103.556 E.00232
G1 X134.249 Y103.538 E.00232
G1 X134.178 Y103.52 E.00232
G1 X134.107 Y103.502 E.00232
G1 F3000
G1 X133.679 Y103.395 E.01396
G1 F2475
G1 X133.54 Y103.36 E.00454
G1 X132.731 Y103.069 E.02723
G1 X132.673 Y103.02 E.0024
G1 F3000
G1 X132.611 Y102.968 E.00257
G1 X132.606 Y102.739 E.00725
G1 F2475
G1 X132.647 Y102.104 E.02013
G1 F3000
G1 X132.723 Y101.092 E.03213
G1 X132.711 Y101.033 E.00192
; LINE_WIDTH: 0.519986
G1 F3600
G1 X132.694 Y100.946 E.0028
G1 X132.676 Y100.859 E.0028
G1 X132.659 Y100.773 E.0028
G1 X132.642 Y100.686 E.0028
G1 X132.624 Y100.599 E.0028
G1 X132.607 Y100.512 E.0028
G1 X132.589 Y100.426 E.0028
; LINE_WIDTH: 0.519996
G1 X132.572 Y100.339 E.0028
G1 X132.593 Y100.246 E.00302
; WIPE_START
M204 S10000
G1 X132.76 Y100.071 E-.09161
G1 X133.085 Y100.119 E-.12499
G1 X133.163 Y99.696 E-.16341
; WIPE_END
G1 E-.02 F1800
G1 X129.911 Y106.601 Z9.4 F36000
G1 X125.74 Y115.457 Z9.4
G1 Z9
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X127.015 Y113.492 I-28.524 J-19.902 E.08946
G3 X129.602 Y115.741 I-4.175 J7.414 E.13177
G3 X130.63 Y119.04 I-4.96 J3.355 E.13383
G3 X129.85 Y120.453 I-2.068 J-.219 E.06332
G2 X127.212 Y122.339 I11.913 J19.454 E.12388
G2 X125.374 Y126.58 I3.988 J4.247 E.18148
G2 X126.155 Y127.994 I2.068 J-.219 E.06332
G3 X128.792 Y129.879 I-11.912 J19.452 E.12388
G3 X130.63 Y134.121 I-3.988 J4.247 E.18148
G3 X129.85 Y135.535 I-2.068 J-.219 E.06332
G2 X127.212 Y137.42 I11.911 J19.452 E.12388
G2 X126.233 Y138.669 I3.478 J3.736 E.06085
G2 X121.113 Y135.984 I-27.244 J45.73 E.22083
G2 X121.203 Y133.691 I-8.544 J-1.483 E.08789
G2 X123.089 Y129.408 I-4.255 J-4.431 E.1833
G2 X122.309 Y127.994 I-2.068 J.219 E.06332
G3 X119.672 Y126.109 I11.912 J-19.452 E.12388
G3 X118.339 Y124.219 I3.84 J-4.122 E.08892
G2 X119.995 Y122.562 I-28.442 J-30.069 E.08945
G1 X123.621 Y121.686 F36000
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.619996
G1 F13446.369
G2 X124.456 Y120.615 I-1.724 J-2.206 E.05235
G2 X124.575 Y119.573 I-2.221 J-.782 E.04038
G1 X124.417 Y119.056 E.02064
G1 X124.135 Y118.604 E.02036
G1 X123.79 Y118.279 E.01809
G1 X121.053 Y121.544 E.16268
G1 X121.367 Y121.785 E.0151
G1 X121.882 Y122.014 E.0215
G1 X122.286 Y122.084 E.01565
G1 X122.761 Y122.058 E.01819
G1 X123.235 Y121.914 E.01889
G1 X123.543 Y121.732 E.01369
G1 X123.191 Y121.178 F36000
; LINE_WIDTH: 0.78384
G1 F10506.959
G1 X123.642 Y120.697 E.0322
G1 X123.825 Y120.4 E.01707
G2 X123.799 Y119.306 I-1.195 J-.518 E.05514
G3 X122.064 Y121.373 I-1047.317 J-877.252 E.13183
G2 X122.93 Y121.323 I.353 J-1.425 E.04303
G1 X123.113 Y121.222 E.0102
G1 X141.945 Y145.895 F36000
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X141.945 Y143.553 E.08944
G1 X141.236 Y143.075 E.03265
G3 X140.456 Y141.662 I1.288 J-1.633 E.06332
G3 X141.945 Y137.784 I5.774 J-.007 E.16222
G1 X141.945 Y128.471 E.35556
G1 X141.236 Y127.994 E.03265
G3 X140.456 Y126.58 I1.288 J-1.633 E.06332
G3 X141.945 Y122.703 I5.774 J-.007 E.16222
G1 X141.945 Y113.39 E.35556
G1 X141.236 Y112.913 E.03265
G3 X140.456 Y111.499 I1.288 J-1.633 E.06332
G3 X141.945 Y107.622 I5.774 J-.007 E.16222
G1 X141.945 Y105.617 E.07652
G3 X139.88 Y105.331 I1.717 J-19.941 E.07963
G1 X138.626 Y105.791 E.05103
G3 X134.418 Y105.855 I-2.659 J-36.277 E.16077
G3 X133.45 Y105.626 I1.549 J-8.681 E.03798
G2 X132.915 Y106.786 I1.35 J1.326 E.04976
G2 X134.753 Y111.028 I5.826 J-.005 E.18148
G2 X137.39 Y112.913 I14.548 J-17.566 E.12388
G3 X138.171 Y114.327 I-1.288 J1.633 E.06332
G3 X136.333 Y118.568 I-5.825 J-.005 E.18148
G3 X133.695 Y120.453 I-14.549 J-17.567 E.12388
G2 X132.915 Y121.867 I1.288 J1.633 E.06332
G2 X134.753 Y126.109 I5.826 J-.005 E.18148
G2 X137.39 Y127.994 I14.548 J-17.566 E.12388
G3 X138.171 Y129.408 I-1.288 J1.633 E.06332
G3 X136.333 Y133.65 I-5.825 J-.005 E.18148
G3 X133.695 Y135.535 I-14.551 J-17.57 E.12388
G2 X132.915 Y136.949 I1.288 J1.633 E.06332
G2 X134.753 Y141.19 I5.826 J-.005 E.18148
G2 X137.39 Y143.075 I14.55 J-17.569 E.12388
G3 X138.083 Y144.018 I-1.163 J1.581 E.04535
G3 X137.382 Y147.356 I-4.821 J.729 E.13303
G3 X138.997 Y149.052 I-33.422 J33.442 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 9.16
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X138.307 Y148.328 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L57
M991 S0 P56 ;notify layer change


G17
G3 Z9.4 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.192 Y103.937
G1 Z9.16
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.131 I-51.324 J-19.616 E.48295
G3 X123.072 Y118.225 I-29.871 J-19.714 E.147
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-40.543 J-37.009 E.29803
G1 X118.014 Y129.011 E.15053
G1 X118.485 Y129.402 E.02336
G1 X119.091 Y130.034 E.03344
G3 X119.749 Y130.932 I-4.534 J4.014 E.04253
G1 X120.215 Y131.838 E.0389
G3 X120.715 Y133.691 I-7.92 J3.132 E.07342
G1 X120.765 Y134.489 E.03056
G1 X120.698 Y135.504 E.03881
G1 X120.538 Y136.268 E.02981
G3 X130.504 Y142.099 I-22.563 J49.995 E.44164
G3 X142.443 Y154.069 I-32.641 J44.496 E.64803
G1 X142.443 Y104.911 E1.87679
G1 X141.701 Y104.967 E.02841
G2 X140.618 Y104.829 I-5.479 J38.683 E.04167
G1 X139.33 Y104.829 E.0492
G1 X138.329 Y105.231 E.04118
G1 X137.824 Y105.359 E.01991
G3 X137.088 Y105.372 I-.479 J-6.432 E.02808
G1 X135.088 Y105.372 E.07636
G1 X134.567 Y105.301 E.02011
G1 X134.026 Y105.099 E.02204
G3 X133.069 Y104.829 I1.322 J-6.506 E.038
G1 X132.525 Y104.809 E.02079
G1 X132.135 Y104.711 E.01533
G1 X131.702 Y104.489 E.01859
G1 X131.253 Y104.003 E.02526
G1 X131.574 Y103.453 F36000
; LINE_WIDTH: 0.663671
G1 F10975.006
G1 X131.57 Y103.445 E.00036
; LINE_WIDTH: 0.707345
G1 F10946.463
G1 X131.523 Y103.332 E.00534
; LINE_WIDTH: 0.751019
G1 F10555.318
G1 X131.476 Y103.22 E.00569
; LINE_WIDTH: 0.794693
G1 F10171.307
G1 X131.429 Y103.108 E.00604
; LINE_WIDTH: 0.838368
G1 F9794.394
G1 X131.382 Y102.995 E.00638
; LINE_WIDTH: 0.882042
G1 F9289.781
G1 X131.335 Y102.883 E.00673
; LINE_WIDTH: 0.925716
G1 F8834.617
G1 X131.288 Y102.771 E.00708
; LINE_WIDTH: 0.970866
G1 F8408.7
G1 X131.273 Y102.702 E.0043
; LINE_WIDTH: 1.01602
G1 F8021.963
G1 X131.259 Y102.633 E.00451
G1 X131.188 Y102.754 E.00898
; LINE_WIDTH: 0.966514
G1 F8447.962
G1 X131.117 Y102.876 E.00853
; LINE_WIDTH: 0.917011
G1 F8921.744
G1 X131.046 Y102.997 E.00807
; LINE_WIDTH: 0.867509
G1 F9451.825
G1 X130.976 Y103.118 E.00762
; LINE_WIDTH: 0.818006
G1 F10048.873
G1 X130.905 Y103.239 E.00717
; LINE_WIDTH: 0.768504
G1 F10489.433
G1 X130.834 Y103.36 E.00672
; LINE_WIDTH: 0.719001
G1 F10939.46
G1 X130.764 Y103.482 E.00626
; LINE_WIDTH: 0.669499
G1 F11398.923
G1 X130.693 Y103.603 E.00581
; LINE_WIDTH: 0.619996
G1 F12760.529
G1 X130.548 Y103.976 E.01527
G1 F13446.369
G1 X130.331 Y104.531 E.02278
G3 X124.88 Y114.797 I-50.511 J-20.239 E.44459
G3 X122.623 Y117.848 I-29.32 J-19.33 E.14499
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-39.644 J-36.157 E.32257
G1 X117.676 Y129.49 E.17985
G1 X118.108 Y129.851 E.02146
G1 X118.662 Y130.433 E.0307
G1 X119.253 Y131.243 E.03828
G1 X119.688 Y132.095 E.03652
G3 X120.136 Y133.793 I-7.132 J2.786 E.0672
G1 X120.18 Y134.525 E.02799
G1 X120.115 Y135.453 E.03553
G1 X119.932 Y136.288 E.03262
G1 X119.836 Y136.599 E.01243
G3 X130.932 Y143.157 I-21.116 J48.399 E.49335
G3 X142.92 Y155.769 I-32.362 J42.764 E.66728
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.768 E1.98459
G1 X142.876 Y104.043 E.01203
G1 X142.27 Y104.362 E.02611
G1 X141.778 Y104.386 E.01882
G2 X140.618 Y104.243 I-4.336 J30.383 E.04461
G1 X139.216 Y104.243 E.05353
G1 X138.11 Y104.688 E.04551
G1 X137.757 Y104.778 E.01393
G3 X137.088 Y104.786 I-.412 J-5.851 E.02553
G1 X135.088 Y104.786 E.07636
G1 X134.723 Y104.737 E.01407
G1 X134.205 Y104.54 E.02116
G3 X133.171 Y104.243 I1.245 J-6.277 E.04113
G1 X132.608 Y104.229 E.02151
G3 X132.032 Y104.005 I.194 J-1.35 E.02379
G1 X131.889 Y103.851 E.00805
G1 X131.618 Y103.557 E.01527
; LINE_WIDTH: 0.663671
G1 F12513.223
G1 X131.609 Y103.536 E.00094
G1 X132.102 Y103.154 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.033 Y102.756 E.0154
G1 X132.247 Y101.596 E.04506
G1 X132.32 Y101.203 E.01527
; LINE_WIDTH: 0.667226
G1 F12442.922
G1 X132.249 Y100.872 E.01395
G1 X132.164 Y100.481 E.0165
; LINE_WIDTH: 0.704053
G1 F11268.313
G1 X132.145 Y100.379 E.00455
; LINE_WIDTH: 0.74088
G1 F10928.378
G1 X132.126 Y100.277 E.0048
; LINE_WIDTH: 0.777706
G1 F10593.649
G1 X132.106 Y100.174 E.00505
G1 X132.196 Y99.919 E.01309
; LINE_WIDTH: 0.774216
G1 F10643.619
G1 X132.417 Y99.71 E.01465
G3 X131.505 Y99.582 I.483 J-6.712 E.04444
; LINE_WIDTH: 0.781476
G1 F10540.194
G3 X131.278 Y100.277 I-3.267 J-.682 E.03572
; LINE_WIDTH: 0.741106
M73 P82 R3
G1 F11142.24
G1 X131.184 Y100.508 E.01149
; LINE_WIDTH: 0.700736
G1 F11817.229
G1 X131.089 Y100.739 E.01084
; LINE_WIDTH: 0.660366
G1 F12579.271
G1 X130.995 Y100.97 E.01018
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X130.867 Y101.349 E.01527
G1 X130.481 Y102.474 E.04538
G3 X124.4 Y114.462 I-50.628 J-18.145 E.51459
G3 X122.174 Y117.472 I-28.764 J-18.941 E.14298
G1 X119.604 Y120.538 E.15272
G3 X112.93 Y126.698 I-40.08 J-36.731 E.34719
G1 X117.323 Y129.957 E.20883
G1 X117.73 Y130.299 E.02031
G1 X118.233 Y130.831 E.02795
G1 X118.77 Y131.574 E.035
G3 X119.516 Y133.621 I-5.252 J3.075 E.08362
G1 X119.596 Y134.561 E.036
G1 X119.531 Y135.403 E.03225
G1 X119.362 Y136.153 E.02934
G1 X119.08 Y136.91 E.03086
G3 X130.579 Y143.624 I-20.881 J48.969 E.50969
G3 X142.648 Y156.415 I-31.956 J42.24 E.67454
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y103.659 E2.00732
G1 X143.615 Y103.259 E.01527
; LINE_WIDTH: 0.668186
G1 F12424.076
G1 X143.59 Y102.949 E.01285
; LINE_WIDTH: 0.716376
G1 F11546.245
G1 X143.566 Y102.639 E.01382
G1 X143.583 Y101.812 E.03679
; LINE_WIDTH: 0.690636
G1 F11999.086
G1 X143.579 Y101.516 E.01265
; LINE_WIDTH: 0.72464
G1 F11408.029
G1 X143.562 Y101.14 E.01695
; LINE_WIDTH: 0.758643
G1 F10872.467
G1 X143.545 Y100.764 E.01779
; LINE_WIDTH: 0.792646
G1 F10384.936
G3 X143.534 Y99.638 I10.648 J-.67 E.05568
; LINE_WIDTH: 0.781186
G1 F10544.287
G3 X142.685 Y99.734 I-.833 J-3.569 E.04169
G1 X142.778 Y99.999 E.0137
; LINE_WIDTH: 0.792646
G1 F10384.936
G2 X142.827 Y100.828 I3.002 J.238 E.04119
; LINE_WIDTH: 0.758643
G1 F10872.467
G1 X142.878 Y101.202 E.01779
; LINE_WIDTH: 0.72464
G1 F11408.029
G1 X142.928 Y101.575 E.01695
; LINE_WIDTH: 0.711536
G1 F11628.767
G1 X142.893 Y102.49 E.04044
G1 X142.808 Y102.833 E.0156
; LINE_WIDTH: 0.665766
G1 F12471.692
G1 X142.723 Y103.177 E.01455
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.481 Y103.61 E.01893
G1 X142.136 Y103.792 E.0149
G1 X141.855 Y103.806 E.01074
G3 X140.866 Y103.657 I1.805 J-15.433 E.03819
G1 X139.103 Y103.657 E.06731
G1 X137.892 Y104.145 E.04984
G1 X137.601 Y104.201 E.0113
G1 X135.088 Y104.201 E.09594
G1 X134.803 Y104.147 E.0111
G2 X133.273 Y103.657 I-9.549 J27.228 E.06132
G1 X132.77 Y103.657 E.0192
G3 X132.362 Y103.522 I.031 J-.778 E.01663
G1 X132.119 Y103.253 E.01384
G1 X132.117 Y103.243 E.00041
M204 S250
G1 X132.603 Y102.987 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G3 X132.613 Y102.659 I.466 J-.152 E.01063
G1 X132.646 Y102.479 E.00577
G1 X132.679 Y102.3 E.00577
G1 X132.712 Y102.12 E.00577
G1 X132.745 Y101.941 E.00577
G1 X132.779 Y101.762 E.00577
G1 X132.812 Y101.582 E.00577
G1 X132.845 Y101.403 E.00577
G2 X132.839 Y100.917 I-1.248 J-.229 E.0155
; LINE_WIDTH: 0.519986
G1 X132.826 Y100.848 E.00222
G1 X132.812 Y100.78 E.00222
G1 X132.798 Y100.711 E.00222
G1 X132.784 Y100.642 E.00222
G1 X132.771 Y100.574 E.00222
G1 X132.757 Y100.505 E.00222
G1 X132.743 Y100.436 E.00222
; LINE_WIDTH: 0.519996
G3 X132.752 Y100.217 I.31 J-.098 E.0071
G1 X132.855 Y100.119 E.0045
G1 X133.083 Y100.119 E.00722
G1 X133.271 Y99.105 E.03265
G3 X131.528 Y98.945 I-.194 J-7.483 E.05552
G1 X131.038 Y98.936 E.01554
G3 X123.946 Y114.147 I-50.893 J-14.471 E.53359
G3 X121.747 Y117.121 I-28.184 J-18.535 E.11716
G1 X119.177 Y120.186 E.12664
G3 X112.01 Y126.704 I-38.873 J-35.55 E.30715
G1 X116.989 Y130.397 E.19628
G3 X117.827 Y131.207 I-2.912 J3.854 E.037
G1 X118.314 Y131.887 E.02646
G3 X118.975 Y133.733 I-4.727 J2.734 E.06242
G1 X119.044 Y134.594 E.02735
G1 X118.98 Y135.355 E.02418
G1 X118.824 Y136.025 E.02177
G1 X118.548 Y136.753 E.02467
G1 X118.31 Y137.19 E.01574
G3 X130.249 Y144.067 I-20.138 J48.759 E.43747
G3 X142.39 Y157.025 I-31.645 J41.818 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.362 E1.81518
G1 X144.167 Y98.947 E.01314
G1 X143.711 Y98.97 E.01448
G1 X142.882 Y99.097 E.02655
G3 X141.909 Y99.102 I-1.397 J-167.516 E.03082
G1 X141.788 Y99.103 E.00383
G1 X141.666 Y99.103 E.00383
G1 X141.636 Y99.103 E.00096
G1 X141.726 Y99.272 E.00604
G1 X142.084 Y99.947 E.0242
G3 X142.135 Y100.114 I-.183 J.147 E.00567
G3 X142.115 Y100.392 I-.501 J.104 E.00894
G1 X142.35 Y101.68 E.04145
G1 X142.344 Y101.762 E.00259
G1 X142.338 Y101.844 E.00259
G1 X142.332 Y101.925 E.00259
G1 X142.327 Y102.007 E.00259
G1 X142.321 Y102.088 E.00259
G1 X142.315 Y102.17 E.00259
G1 X142.309 Y102.252 E.00259
G1 X142.304 Y102.333 E.00259
G1 F3000
G1 X142.298 Y102.415 E.00259
G1 F2475
G1 X142.201 Y102.957 E.01744
G1 F3000
G3 X142.15 Y103.128 I-.279 J.01 E.00577
G1 F3150
G1 X142.117 Y103.186 E.00209
G1 F3300
G1 X141.931 Y103.256 E.0063
G1 F3150
G1 X141.268 Y103.162 E.0212
G1 F3600
G1 X141.177 Y103.151 E.0029
G1 F3150
G1 X140.949 Y103.132 E.00725
G1 F3300
G1 X140.721 Y103.113 E.00725
G1 F3450
G2 X140.369 Y103.105 I-.227 J2.128 E.01117
G1 F3300
G1 X139.814 Y103.105 E.01756
G1 F3150
G1 X139.26 Y103.105 E.01756
G1 F3000
G1 X139.013 Y103.105 E.00782
G2 X138.195 Y103.427 I21.259 J55.176 E.02782
G1 F3600
G1 X138.132 Y103.452 E.00218
G1 X138.068 Y103.478 E.00217
G1 X138.004 Y103.504 E.00217
G1 X137.941 Y103.529 E.00218
G1 X137.877 Y103.555 E.00217
G1 X137.813 Y103.58 E.00218
G1 X137.749 Y103.606 E.00217
G3 X137.601 Y103.648 I-.182 J-.361 E.0049
G1 X135.088 Y103.648 E.07956
G1 X135.026 Y103.625 E.00211
G1 X134.963 Y103.603 E.00211
G1 X134.901 Y103.58 E.00211
G1 X134.838 Y103.557 E.00211
G1 X134.776 Y103.535 E.00211
G1 X134.713 Y103.512 E.00211
G1 X134.651 Y103.489 E.00211
G1 X134.588 Y103.467 E.00211
G1 F3000
G1 X133.859 Y103.28 E.02384
G1 X133.619 Y103.194 E.00807
G1 F2475
G1 X133.341 Y103.105 E.00922
G1 F3000
G1 X133.281 Y103.105 E.00192
G1 F3600
G1 X133.22 Y103.105 E.00192
G1 X133.16 Y103.105 E.00192
G1 X133.099 Y103.105 E.00192
G1 X133.038 Y103.105 E.00192
G1 X132.978 Y103.105 E.00192
G1 X132.917 Y103.105 E.00192
G1 X132.857 Y103.105 E.00192
G3 X132.668 Y103.049 I-.013 J-.304 E.00636
; WIPE_START
M204 S10000
G1 X132.579 Y102.838 E-.08692
G1 X132.613 Y102.659 E-.06931
G1 X132.646 Y102.479 E-.06931
G1 X132.679 Y102.3 E-.06931
G1 X132.712 Y102.12 E-.06931
G1 X132.72 Y102.08 E-.01584
; WIPE_END
G1 E-.02 F1800
G1 X131.639 Y101.174 Z9.56 F36000
G1 Z9.16
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.838116
G1 F9797.458
G1 X131.563 Y101.477 E.01637
; LINE_WIDTH: 0.881001
G1 F9301.201
G1 X131.489 Y101.757 E.01595
; LINE_WIDTH: 0.923886
G1 F8852.791
G1 X131.416 Y102.036 E.01675
; LINE_WIDTH: 0.966771
G1 F8445.629
G1 X131.342 Y102.316 E.01756
; LINE_WIDTH: 1.00966
G1 F8074.273
G1 X131.268 Y102.595 E.01837
; LINE_WIDTH: 1.01602
G1 F8021.963
G1 X131.259 Y102.633 E.00252
; WIPE_START
G1 X131.268 Y102.595 E-.01494
G1 X131.342 Y102.316 E-.10979
G1 X131.416 Y102.036 E-.1098
G1 X131.489 Y101.757 E-.10979
G1 X131.513 Y101.666 E-.03567
; WIPE_END
G1 E-.02 F1800
G1 X128.55 Y108.7 Z9.56 F36000
G1 X125.652 Y115.581 Z9.56
G1 Z9.16
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X126.926 Y113.615 I-33.41 J-23.047 E.08944
G3 X129.611 Y115.741 I-3.506 J7.187 E.13174
G3 X130.837 Y119.511 I-5.15 J3.76 E.15393
G3 X130.227 Y120.453 I-1.418 J-.249 E.04404
G3 X127.893 Y121.867 I-90.038 J-146.048 E.10421
G2 X125.167 Y127.052 I3.424 J5.109 E.23307
G2 X125.777 Y127.994 I1.418 J-.249 E.04404
G2 X128.112 Y129.408 I90.179 J-146.279 E.10421
G3 X130.837 Y134.592 I-3.424 J5.109 E.23307
G3 X130.227 Y135.535 I-1.418 J-.249 E.04404
G3 X127.893 Y136.949 I-90.179 J-146.279 E.10421
G2 X126.215 Y138.657 I4.328 J5.926 E.09182
G2 X121.116 Y135.986 I-36.558 J63.588 E.21983
G2 X121.207 Y133.633 I-9.032 J-1.526 E.09017
G2 X123.297 Y128.937 I-4.063 J-4.621 E.20245
G2 X122.687 Y127.994 I-1.418 J.249 E.04404
G2 X120.352 Y126.58 I-90.179 J146.279 E.10421
G3 X118.303 Y124.258 I4.148 J-5.724 E.11922
G3 X116.573 Y125.836 I-34.807 J-36.402 E.08943
G1 X139.016 Y149.073 F36000
G1 F13446.283
G2 X137.4 Y147.378 I-32.127 J29.025 E.08944
G2 X138.378 Y144.018 I-4.647 J-3.176 E.13586
G2 X137.768 Y143.075 I-1.418 J.249 E.04404
G2 X135.433 Y141.662 I-89.954 J145.908 E.10421
G3 X132.707 Y136.477 I3.424 J-5.109 E.23307
G3 X133.318 Y135.535 I1.418 J.249 E.04404
G3 X135.652 Y134.121 I89.996 J145.979 E.10421
G2 X138.378 Y128.937 I-3.424 J-5.109 E.23307
G2 X137.768 Y127.994 I-1.418 J.249 E.04404
G2 X135.433 Y126.58 I-90.094 J146.139 E.10421
G3 X132.707 Y121.396 I3.424 J-5.109 E.23307
G3 X133.318 Y120.453 I1.418 J.249 E.04404
G3 X135.652 Y119.04 I90.137 J146.211 E.10421
G2 X138.378 Y113.855 I-3.424 J-5.109 E.23307
G2 X137.768 Y112.913 I-1.418 J.249 E.04404
G2 X135.433 Y111.499 I-90.094 J146.139 E.10421
G3 X132.707 Y106.315 I3.424 J-5.109 E.23307
G3 X133.272 Y105.422 I1.336 J.219 E.04143
G2 X134.909 Y105.864 I6.182 J-19.619 E.06476
G2 X137.881 Y105.854 I1.404 J-25.067 E.11355
G2 X139.426 Y105.327 I-1.005 J-5.47 E.06257
G3 X141.653 Y105.463 I.652 J7.612 E.08547
G1 X141.945 Y105.448 E.01119
G1 X141.945 Y107.651 E.08412
G2 X140.248 Y111.97 I4.456 J4.244 E.18166
G2 X140.858 Y112.913 I1.418 J-.249 E.04404
G2 X141.945 Y113.581 I6.267 J-8.983 E.04874
G1 X141.945 Y122.733 E.34942
G2 X140.248 Y127.052 I4.456 J4.244 E.18166
G2 X140.858 Y127.994 I1.418 J-.249 E.04404
G2 X141.945 Y128.662 I6.267 J-8.983 E.04874
G1 X141.945 Y137.814 E.34942
G2 X140.248 Y142.133 I4.456 J4.244 E.18166
G2 X140.858 Y143.075 I1.418 J-.249 E.04404
G2 X141.945 Y143.743 I6.267 J-8.983 E.04874
G1 X141.945 Y146.086 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 9.32
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.945 Y145.086 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L58
M991 S0 P57 ;notify layer change


G17
G3 Z9.56 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.186 Y103.949
G1 Z9.32
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.36 Y115.132 I-51.354 J-19.645 E.48247
G3 X123.072 Y118.225 I-29.87 J-19.714 E.14696
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-40.696 J-37.17 E.29802
G1 X118.03 Y129.023 E.15128
G3 X118.933 Y129.85 I-6.02 J7.481 E.04675
G3 X120.076 Y131.531 I-5.864 J5.215 E.07784
G1 X120.45 Y132.48 E.03893
G1 X120.681 Y133.47 E.03883
G1 X120.765 Y134.489 E.03903
G1 X120.698 Y135.506 E.03892
G1 X120.538 Y136.268 E.02972
G3 X130.508 Y142.102 I-22.334 J49.603 E.44185
G3 X142.443 Y154.069 I-32.72 J44.568 E.64783
G1 X142.443 Y107.508 E1.77766
G2 X142.442 Y104.712 I-666.267 J-1 E.10675
G1 X141.766 Y104.831 E.02618
G1 X138.929 Y104.831 E.10831
G3 X137.708 Y105.274 I-4.122 J-9.447 E.04965
G1 X137.282 Y105.365 E.01663
G1 X137.092 Y105.374 E.00725
G1 X135.609 Y105.374 E.05664
G1 X135.092 Y105.305 E.0199
G2 X133.673 Y104.831 I-104.822 J311.709 E.05713
G1 X132.799 Y104.831 E.03336
G1 X132.486 Y104.806 E.01198
G1 X131.944 Y104.633 E.02172
G1 X131.493 Y104.329 E.02079
G1 X131.243 Y104.019 E.01518
G1 X131.554 Y103.406 F36000
; LINE_WIDTH: 0.717476
G1 F10607.148
G1 X131.516 Y103.301 E.00499
; LINE_WIDTH: 0.766216
G1 F10252.399
G1 X131.474 Y103.185 E.00587
; LINE_WIDTH: 0.814956
G1 F9870.207
G1 X131.432 Y103.07 E.00626
; LINE_WIDTH: 0.863696
G1 F9495.274
G1 X131.39 Y102.954 E.00665
; LINE_WIDTH: 0.912436
G1 F8968.228
G1 X131.348 Y102.839 E.00704
; LINE_WIDTH: 0.961176
G1 F8496.613
G1 X131.306 Y102.723 E.00743
; LINE_WIDTH: 1.00992
G1 F8072.122
G1 X131.264 Y102.607 E.00782
G1 X131.193 Y102.731 E.00909
; LINE_WIDTH: 0.961176
G1 F8496.613
G1 X131.122 Y102.855 E.00864
; LINE_WIDTH: 0.912436
G1 F8968.228
G1 X131.05 Y102.979 E.00818
; LINE_WIDTH: 0.863696
G1 F9495.274
G1 X130.979 Y103.103 E.00773
; LINE_WIDTH: 0.814956
G1 F10088.136
G1 X130.908 Y103.227 E.00727
; LINE_WIDTH: 0.766216
G1 F10537.84
G1 X130.837 Y103.351 E.00682
; LINE_WIDTH: 0.717476
G1 F10997.378
G1 X130.766 Y103.475 E.00637
; LINE_WIDTH: 0.668736
G1 F11466.679
G1 X130.694 Y103.599 E.00591
; LINE_WIDTH: 0.619996
G1 F12832.212
G1 X130.548 Y103.971 E.01527
G1 F13446.369
G1 X130.402 Y104.344 E.01527
G1 X130.208 Y104.834 E.02015
G3 X124.88 Y114.798 I-50.428 J-20.563 E.43215
G3 X122.623 Y117.848 I-29.315 J-19.327 E.14495
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-39.645 J-36.158 E.32256
G1 X117.677 Y129.49 E.17987
G3 X118.524 Y130.269 I-5.661 J7.009 E.04395
G3 X119.208 Y131.169 I-7.749 J6.602 E.04319
G1 X119.563 Y131.813 E.02809
G1 X119.901 Y132.683 E.03564
G1 X120.108 Y133.591 E.03555
G1 X120.18 Y134.525 E.03574
G1 X120.114 Y135.455 E.03563
G1 X119.931 Y136.289 E.03259
G1 X119.836 Y136.599 E.01238
G3 X130.932 Y143.157 I-21.119 J48.404 E.49335
G3 X142.92 Y155.769 I-32.375 J42.776 E.66727
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.284 E2.00303
G1 X142.764 Y103.812 E.02253
G1 X142.238 Y104.161 E.02409
G1 X141.766 Y104.245 E.01831
G1 X138.813 Y104.245 E.11276
G3 X137.523 Y104.719 I-4.124 J-9.238 E.0525
G1 X137.092 Y104.789 E.01666
G1 X135.609 Y104.789 E.05664
G1 X135.247 Y104.74 E.01392
G1 X134.679 Y104.56 E.02277
G2 X133.758 Y104.245 I-2.519 J5.861 E.0372
G1 X132.799 Y104.245 E.03661
G1 X132.58 Y104.228 E.00838
G1 X132.175 Y104.094 E.0163
G1 X131.885 Y103.894 E.01344
G1 X131.6 Y103.532 E.01758
; LINE_WIDTH: 0.668736
G1 F12413.304
G1 X131.585 Y103.491 E.00181
G1 X132.096 Y103.135 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.032 Y102.744 E.01511
G1 X132.4 Y100.761 E.07701
G1 X132.336 Y100.249 E.0197
G1 X132.377 Y100.084 E.00649
G1 X132.587 Y99.75 E.01505
G1 X132.605 Y99.651 E.00386
G3 X131.451 Y99.493 I.235 J-6.046 E.04455
G3 X124.399 Y114.463 I-51.99 J-15.346 E.63426
G3 X122.174 Y117.472 I-28.76 J-18.94 E.14295
G1 X119.604 Y120.538 E.15272
G3 X112.93 Y126.698 I-40.081 J-36.732 E.34718
G1 X117.323 Y129.957 E.20885
G3 X118.151 Y130.726 I-6.59 J7.926 E.04316
G3 X118.695 Y131.451 I-12.208 J9.722 E.0346
G1 X119.049 Y132.096 E.02809
G1 X119.352 Y132.887 E.03235
G1 X119.535 Y133.712 E.03227
G1 X119.595 Y134.56 E.03246
G1 X119.531 Y135.405 E.03234
G1 X119.361 Y136.154 E.02931
G1 X119.08 Y136.91 E.03082
G3 X130.579 Y143.624 I-20.648 J48.57 E.50973
G3 X142.648 Y156.415 I-31.964 J42.247 E.67453
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y103.52 E2.01262
G1 X143.615 Y103.12 E.01527
; LINE_WIDTH: 0.662541
G1 F12484.667
G1 X143.593 Y102.903 E.00894
; LINE_WIDTH: 0.705086
G1 F11740.59
G1 X143.572 Y102.686 E.00954
; LINE_WIDTH: 0.747631
G1 F11040.314
G1 X143.551 Y102.469 E.01015
; LINE_WIDTH: 0.790176
G1 F10418.872
G1 X143.529 Y102.251 E.01075
G1 X143.544 Y101.531 E.0355
G1 X143.552 Y101.131 E.01971
; LINE_WIDTH: 0.794696
G1 F10356.937
G1 X143.527 Y100.364 E.03803
; LINE_WIDTH: 0.801316
G1 F10267.544
G1 X143.524 Y99.649 E.03578
G3 X142.58 Y99.746 I-.848 J-3.596 E.04755
G1 X142.748 Y100.087 E.019
G1 X142.834 Y100.883 E.04005
; LINE_WIDTH: 0.785666
G1 F10481.414
G1 X142.782 Y102.096 E.05942
G1 X142.719 Y102.327 E.01174
; LINE_WIDTH: 0.744249
G1 F11092.917
G1 X142.657 Y102.559 E.0111
; LINE_WIDTH: 0.702831
G1 F11780.193
G1 X142.594 Y102.79 E.01045
; LINE_WIDTH: 0.661414
G1 F12558.258
G1 X142.532 Y103.022 E.0098
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X142.335 Y103.412 E.01669
G1 X142.036 Y103.611 E.01375
G1 X141.766 Y103.66 E.01045
G1 X138.696 Y103.66 E.11722
G3 X137.338 Y104.163 I-4.289 J-9.483 E.05534
G1 X137.092 Y104.203 E.00951
G1 X135.609 Y104.203 E.05664
G1 X135.117 Y104.097 E.01919
G2 X133.843 Y103.66 I-4.412 J10.784 E.05148
G1 X132.799 Y103.66 E.03985
G1 X132.443 Y103.573 E.014
G3 X132.115 Y103.253 I.356 J-.692 E.01777
G1 X132.11 Y103.224 E.00111
M204 S250
G1 X132.611 Y102.984 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G3 X132.599 Y102.719 I.338 J-.148 E.00858
G1 X132.644 Y102.477 E.0078
G1 X132.689 Y102.235 E.0078
G1 X132.734 Y101.993 E.0078
G1 X132.779 Y101.751 E.0078
G1 X132.823 Y101.509 E.0078
G1 F3450
G1 X132.868 Y101.266 E.0078
G1 F3300
G1 X132.913 Y101.024 E.0078
G1 F3150
G1 X132.958 Y100.782 E.0078
G1 F3600
G1 X132.962 Y100.746 E.00114
G1 X132.953 Y100.69 E.0018
G1 X132.944 Y100.634 E.0018
G1 X132.935 Y100.578 E.0018
G1 X132.926 Y100.522 E.0018
G1 X132.917 Y100.465 E.0018
G1 X132.908 Y100.409 E.0018
G1 X132.898 Y100.353 E.0018
G1 X132.889 Y100.297 E.0018
G1 X132.949 Y100.192 E.00383
G1 X133.074 Y100.154 E.00414
G1 X133.268 Y99.107 E.03371
G3 X131.528 Y98.945 I-.192 J-7.396 E.05546
G1 X131.038 Y98.936 E.01552
G3 X123.945 Y114.148 I-51.152 J-14.591 E.53361
G3 X121.747 Y117.121 I-28.175 J-18.53 E.11714
G1 X119.177 Y120.187 E.12664
G3 X112.01 Y126.704 I-38.874 J-35.552 E.30713
G1 X116.989 Y130.398 E.1963
G3 X117.739 Y131.094 I-6.442 J7.683 E.03241
G3 X118.21 Y131.717 I-28.641 J22.166 E.02474
G1 X118.565 Y132.362 E.0233
G1 X118.834 Y133.079 E.02426
G1 X118.994 Y133.827 E.02419
G1 X119.044 Y134.594 E.02435
G1 X118.98 Y135.357 E.02425
G1 X118.824 Y136.026 E.02174
G1 X118.548 Y136.753 E.02461
G1 X118.31 Y137.19 E.01576
G3 X130.249 Y144.067 I-19.515 J47.679 E.43754
G3 X142.39 Y157.025 I-31.645 J41.817 E.56491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.207 E1.82006
G1 X144.167 Y98.938 E.00853
G1 X143.859 Y98.944 E.00977
G1 X142.882 Y99.099 E.03133
G3 X141.869 Y99.105 I-1.508 J-181.447 E.03206
G1 X141.804 Y99.105 E.00207
G1 X141.739 Y99.105 E.00207
G1 X141.673 Y99.105 E.00207
G1 X141.608 Y99.105 E.00207
G1 F3450
G1 X141.543 Y99.105 E.00207
G1 F3300
G1 X141.516 Y99.106 E.00086
G1 X141.931 Y99.941 E.02955
G1 F3450
G1 X141.964 Y99.994 E.00196
G1 F3600
G1 X141.996 Y100.047 E.00196
G1 X142.029 Y100.099 E.00196
G1 X142.061 Y100.152 E.00196
G1 X142.094 Y100.205 E.00196
G1 X142.126 Y100.258 E.00196
G2 X142.228 Y101.053 I3.788 J-.08 E.02544
G1 X142.219 Y101.159 E.00337
G1 X142.211 Y101.265 E.00337
G1 X142.202 Y101.372 E.00337
G1 X142.194 Y101.478 E.00337
G1 X142.185 Y101.584 E.00337
G1 X142.177 Y101.69 E.00337
G1 X142.168 Y101.796 E.00337
G1 X142.159 Y101.902 E.00337
G1 F3000
G1 X142.151 Y102.008 E.00337
G1 F2475
G1 X142.056 Y102.551 E.01747
G1 F3000
G1 X141.968 Y102.961 E.01326
G1 F3600
G1 X141.942 Y103.014 E.00186
G1 X141.931 Y103.035 E.00075
G1 X141.766 Y103.107 E.0057
G1 X141.716 Y103.107 E.00158
G1 X141.528 Y103.107 E.00597
G1 X141.34 Y103.107 E.00597
G1 X141.151 Y103.107 E.00597
G1 X140.963 Y103.107 E.00597
G1 X140.774 Y103.107 E.00597
G1 X140.345 Y103.107 E.01359
G1 X140.104 Y103.107 E.00762
G1 X139.864 Y103.107 E.00762
G1 X139.623 Y103.107 E.00762
G1 X139.382 Y103.107 E.00762
G1 F3450
G1 X139.142 Y103.107 E.00762
G1 F3300
G1 X138.901 Y103.107 E.00762
G1 F3150
G1 X138.66 Y103.107 E.00762
G1 F3000
G1 X138.586 Y103.107 E.00236
G1 X137.857 Y103.407 E.02495
G1 F3600
G1 X137.77 Y103.436 E.00289
G1 X137.684 Y103.465 E.00289
G1 X137.597 Y103.494 E.00289
G1 X137.51 Y103.523 E.00289
G1 X137.424 Y103.552 E.00289
G1 X137.337 Y103.581 E.00289
G1 X137.25 Y103.61 E.00289
G3 X137.092 Y103.65 I-.203 J-.465 E.00519
G1 X135.609 Y103.65 E.04697
G1 X135.545 Y103.632 E.00209
G1 X135.482 Y103.613 E.00209
G1 X135.419 Y103.594 E.00209
G1 X135.355 Y103.576 E.00209
G1 X135.292 Y103.557 E.00209
G1 X135.229 Y103.539 E.00209
G1 X135.165 Y103.52 E.00209
G1 X135.102 Y103.502 E.00209
G1 F3000
G2 X134.25 Y103.204 I-4.654 J11.944 E.0286
G1 F2475
G1 X133.923 Y103.107 E.01079
G1 X133.867 Y103.107 E.00177
G1 F3000
G1 X133.748 Y103.107 E.00376
G1 F3600
G1 X133.629 Y103.107 E.00376
G1 X133.511 Y103.107 E.00376
G1 X133.392 Y103.107 E.00376
G1 X133.273 Y103.107 E.00376
G1 X133.155 Y103.107 E.00376
G1 X133.036 Y103.107 E.00376
G1 X132.917 Y103.107 E.00376
G3 X132.661 Y103.057 I-.059 J-.38 E.00844
; WIPE_START
M204 S10000
G1 X132.577 Y102.84 E-.0884
G1 X132.599 Y102.719 E-.04655
G1 X132.644 Y102.477 E-.0936
G1 X132.689 Y102.235 E-.0936
G1 X132.717 Y102.085 E-.05784
; WIPE_END
G1 E-.02 F1800
G1 X131.264 Y102.607 Z9.72 F36000
G1 Z9.32
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.00992
G1 F8072.122
G1 X131.317 Y102.414 E.01278
; LINE_WIDTH: 0.976676
G1 F8356.857
G1 X131.401 Y102.09 E.02056
; LINE_WIDTH: 0.92815
G1 F8810.565
G1 X131.486 Y101.766 E.0195
; LINE_WIDTH: 0.879623
G1 F9316.368
G1 X131.571 Y101.442 E.01845
; LINE_WIDTH: 0.831096
G1 F9883.78
G1 X131.629 Y101.177 E.01409
; LINE_WIDTH: 0.78123
G1 F10543.674
G1 X131.686 Y100.912 E.01321
; LINE_WIDTH: 0.731363
G1 F11297.991
G1 X131.744 Y100.647 E.01233
; LINE_WIDTH: 0.681496
G1 F12168.553
G1 X131.764 Y100.511 E.0058
; LINE_WIDTH: 0.637896
G1 F13047.586
G1 X131.784 Y100.374 E.00541
; LINE_WIDTH: 0.594296
G1 F14063.505
G1 X131.804 Y100.238 E.00502
; LINE_WIDTH: 0.550696
G1 F15250.99
G1 X131.824 Y100.102 E.00463
; WIPE_START
G1 X131.804 Y100.238 E-.05228
G1 X131.784 Y100.374 E-.05227
G1 X131.764 Y100.511 E-.05228
G1 X131.744 Y100.647 E-.05227
G1 X131.686 Y100.912 E-.10312
G1 X131.648 Y101.086 E-.06778
; WIPE_END
G1 E-.02 F1800
G1 X138.705 Y103.994 Z9.72 F36000
G1 X141.945 Y105.329 Z9.72
G1 Z9.32
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X141.945 Y107.672 E.08944
M73 P83 R3
G2 X139.896 Y111.97 I5.743 J5.376 E.18469
G1 X139.939 Y112.442 E.01807
G1 X140.307 Y112.913 E.02282
G2 X141.945 Y113.74 I7.796 J-13.41 E.07012
G1 X141.945 Y122.753 E.34413
G2 X139.896 Y127.052 I5.743 J5.376 E.18469
G1 X139.939 Y127.523 E.01807
G1 X140.307 Y127.994 E.02282
G2 X141.945 Y128.821 I7.796 J-13.41 E.07012
G1 X141.945 Y137.835 E.34413
G2 X139.896 Y142.133 I5.743 J5.376 E.18469
G1 X139.939 Y142.604 E.01807
G1 X140.307 Y143.075 E.02282
G2 X141.945 Y143.902 I7.796 J-13.41 E.07012
G1 X141.945 Y152.574 E.33106
G2 X137.422 Y147.401 I-47.764 J37.204 E.26249
G2 X138.73 Y144.018 I-6.358 J-4.402 E.13978
G1 X138.687 Y143.547 E.01807
G1 X138.319 Y143.075 E.02282
G3 X135.562 Y141.662 I704.543 J-1377.401 E.11831
G3 X132.541 Y137.42 I3.743 J-5.862 E.20378
G1 X132.356 Y136.477 E.03668
G1 X132.399 Y136.006 E.01807
G1 X132.766 Y135.535 E.02282
G2 X135.524 Y134.121 I-706.621 J-1381.448 E.11831
G2 X138.544 Y129.879 I-3.743 J-5.862 E.20378
G1 X138.73 Y128.937 E.03668
G1 X138.687 Y128.465 E.01807
G1 X138.319 Y127.994 E.02282
G3 X135.562 Y126.58 I696.538 J-1361.789 E.11831
G3 X132.541 Y122.339 I3.743 J-5.862 E.20378
G1 X132.356 Y121.396 E.03668
G1 X132.399 Y120.925 E.01807
G1 X132.766 Y120.453 E.02282
G2 X135.524 Y119.04 I-698.569 J-1365.745 E.11831
G2 X138.544 Y114.798 I-3.743 J-5.862 E.20378
G1 X138.73 Y113.855 E.03668
G1 X138.687 Y113.384 E.01807
G1 X138.319 Y112.913 E.02282
G3 X135.562 Y111.499 I696.538 J-1361.789 E.11831
G3 X132.541 Y107.257 I3.743 J-5.862 E.20378
G1 X132.356 Y106.315 E.03668
G1 X132.399 Y105.843 E.01807
G1 X132.847 Y105.329 E.02606
G3 X131.367 Y104.841 I-.028 J-2.404 E.06058
G3 X126.857 Y113.737 I-50.403 J-19.963 E.38132
G3 X131.189 Y119.511 I-2.715 J6.55 E.28917
G1 X131.146 Y119.982 E.01807
G1 X130.779 Y120.453 E.02282
G2 X128.021 Y121.867 I704.543 J1377.401 E.11831
G2 X125.001 Y126.109 I3.743 J5.862 E.20378
G1 X124.815 Y127.052 E.03668
G1 X124.858 Y127.523 E.01807
G1 X125.226 Y127.994 E.02282
G3 X127.983 Y129.408 I-696.538 J1361.789 E.11831
G3 X131.004 Y133.65 I-3.743 J5.862 E.20378
G1 X131.189 Y134.592 E.03668
G1 X131.146 Y135.063 E.01807
G1 X130.779 Y135.535 E.02282
G2 X128.021 Y136.949 I696.538 J1361.789 E.11831
G2 X126.192 Y138.645 I4.083 J6.238 E.09572
G2 X121.111 Y135.983 I-36.204 J62.921 E.21906
G2 X121.196 Y133.591 I-8.318 J-1.493 E.0917
G2 X123.463 Y129.879 I-4.87 J-5.524 E.16858
G1 X123.649 Y128.937 E.03668
G1 X123.606 Y128.465 E.01807
G1 X123.238 Y127.994 E.02282
G3 X120.481 Y126.58 I698.569 J-1365.745 E.11831
G3 X118.264 Y124.296 I3.864 J-5.967 E.12258
G2 X119.917 Y122.636 I-28.755 J-30.3 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 9.48
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X119.211 Y123.344 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L59
M991 S0 P58 ;notify layer change


G17
G3 Z9.72 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.19 Y103.942
G1 Z9.48
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.131 I-51.333 J-19.626 E.48275
G3 X123.072 Y118.224 I-29.873 J-19.716 E.14699
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-40.901 J-37.385 E.29803
G1 X118.015 Y129.012 E.15056
G3 X118.933 Y129.85 I-4.606 J5.967 E.04749
G3 X120.076 Y131.532 I-5.877 J5.224 E.07788
G1 X120.45 Y132.479 E.03889
G1 X120.682 Y133.473 E.03895
G1 X120.765 Y134.488 E.03889
G1 X120.698 Y135.506 E.03896
G1 X120.538 Y136.268 E.02972
G3 X130.508 Y142.102 I-22.571 J50.012 E.44184
G3 X142.443 Y154.069 I-32.645 J44.492 E.64783
G1 X142.443 Y104.711 E1.88444
G1 X141.795 Y104.833 E.0252
G1 X138.52 Y104.833 E.12501
G1 X137.908 Y105.054 E.02485
G3 X134.901 Y105.09 I-1.563 J-5.032 E.11642
G1 X134.154 Y104.833 E.03018
G1 X132.796 Y104.833 E.05183
G3 X132.146 Y104.721 I0 J-1.949 E.02529
G1 X131.733 Y104.518 E.01758
G1 X131.251 Y104.007 E.02681
G1 X131.566 Y103.437 F36000
; LINE_WIDTH: 0.714909
G1 F10739.764
G1 X131.524 Y103.322 E.00541
; LINE_WIDTH: 0.762365
G1 F10351.766
G1 X131.481 Y103.203 E.00601
; LINE_WIDTH: 0.809821
G1 F9956.717
G1 X131.437 Y103.085 E.0064
; LINE_WIDTH: 0.857278
G1 F9569.335
G1 X131.393 Y102.966 E.00679
; LINE_WIDTH: 0.904734
G1 F9047.591
G1 X131.35 Y102.847 E.00718
; LINE_WIDTH: 0.95219
G1 F8579.799
G1 X131.306 Y102.728 E.00757
; LINE_WIDTH: 0.999646
G1 F8158.002
G1 X131.262 Y102.61 E.00796
G1 X131.191 Y102.733 E.00898
; LINE_WIDTH: 0.95219
G1 F8579.799
G1 X131.12 Y102.857 E.00854
; LINE_WIDTH: 0.904734
G1 F9047.591
G1 X131.049 Y102.981 E.0081
; LINE_WIDTH: 0.857278
G1 F9569.335
G1 X130.978 Y103.105 E.00765
; LINE_WIDTH: 0.809821
G1 F10154.936
G1 X130.907 Y103.228 E.00721
; LINE_WIDTH: 0.762365
G1 F10605.271
G1 X130.836 Y103.352 E.00677
; LINE_WIDTH: 0.714909
G1 F11065.36
G1 X130.765 Y103.476 E.00633
; LINE_WIDTH: 0.667453
G1 F11535.218
G1 X130.694 Y103.6 E.00589
; LINE_WIDTH: 0.619996
G1 F12904.712
G1 X130.549 Y103.972 E.01527
G1 F13446.369
G1 X130.403 Y104.345 E.01527
G1 X130.306 Y104.586 E.00994
G3 X124.88 Y114.797 I-51.767 J-20.959 E.44224
G3 X122.623 Y117.848 I-29.325 J-19.333 E.14498
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-40.186 J-36.735 E.32257
G1 X117.666 Y129.482 E.17937
G3 X118.524 Y130.269 I-6.088 J7.499 E.04445
G3 X119.209 Y131.171 I-7.797 J6.637 E.04329
G1 X119.563 Y131.814 E.02802
G1 X119.901 Y132.683 E.0356
G1 X120.109 Y133.594 E.03567
G1 X120.18 Y134.524 E.0356
G1 X120.114 Y135.456 E.03568
G1 X119.933 Y136.283 E.03235
G1 X119.836 Y136.599 E.01261
G3 X130.932 Y143.157 I-21.125 J48.414 E.49335
G3 X142.92 Y155.769 I-32.375 J42.777 E.66727
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.424 E1.99771
G1 X142.692 Y103.91 E.02258
G1 X142.215 Y104.181 E.02094
G1 X141.795 Y104.247 E.01626
G1 X138.41 Y104.247 E.12924
G1 X137.943 Y104.43 E.01914
G1 X137.063 Y104.658 E.03471
G1 X136.757 Y104.697 E.01175
G1 X136.049 Y104.707 E.02705
G3 X134.258 Y104.247 I.803 J-6.848 E.07079
G1 X132.796 Y104.247 E.05582
G1 X132.342 Y104.169 E.01761
G1 X132.053 Y104.027 E.0123
G1 X131.886 Y103.851 E.00925
G1 X131.612 Y103.56 E.01527
; LINE_WIDTH: 0.667453
G1 F12438.475
G1 X131.598 Y103.521 E.0017
G1 X132.096 Y103.158 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.03 Y102.747 E.01593
G1 X132.603 Y99.652 E.12014
G3 X131.451 Y99.494 I.237 J-5.979 E.04447
G3 X124.4 Y114.462 I-51.784 J-15.249 E.63422
G3 X122.174 Y117.472 I-28.762 J-18.94 E.14297
G1 X119.605 Y120.537 E.15272
G3 X112.93 Y126.698 I-40.841 J-37.555 E.34718
G1 X117.317 Y129.953 E.20857
G3 X118.114 Y130.688 I-5.806 J7.097 E.04142
G3 X118.696 Y131.453 I-9.731 J7.999 E.03672
G1 X119.05 Y132.096 E.02802
G1 X119.352 Y132.887 E.03232
G1 X119.536 Y133.715 E.03238
G1 X119.596 Y134.559 E.03232
G1 X119.531 Y135.405 E.03239
G1 X119.363 Y136.148 E.02907
G1 X119.08 Y136.91 E.03104
G3 X130.579 Y143.624 I-20.824 J48.87 E.50971
G3 X142.647 Y156.415 I-31.963 J42.246 E.67453
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y104.463 E1.97662
G1 X143.615 Y104.063 E.01527
; LINE_WIDTH: 0.666673
G1 F12259.581
G1 X143.591 Y103.827 E.00979
; LINE_WIDTH: 0.71335
G1 F11458.082
G1 X143.568 Y103.59 E.01052
; LINE_WIDTH: 0.760026
G1 F10683.676
G1 X143.545 Y103.354 E.01124
; LINE_WIDTH: 0.808901
G1 F9936.366
G1 X143.52 Y103.232 E.00629
; LINE_WIDTH: 0.857776
G1 F9555.114
G1 X143.496 Y103.109 E.00669
; LINE_WIDTH: 0.892056
G1 F9181.319
G3 X143.497 Y101.566 I16.466 J-.755 E.08632
; LINE_WIDTH: 0.854636
G1 F9600.146
G1 X143.516 Y100.849 E.03837
; LINE_WIDTH: 0.839746
G1 F9777.63
G1 X143.505 Y100.248 E.03155
; LINE_WIDTH: 0.846836
G1 F9692.308
G1 X143.501 Y99.676 E.03029
G3 X142.471 Y99.771 I-.864 J-3.74 E.05494
G1 X142.648 Y100.064 E.01812
G1 X142.732 Y100.656 E.03166
; LINE_WIDTH: 0.852876
G1 F9620.789
G1 X142.681 Y101.769 E.05945
; LINE_WIDTH: 0.892056
G1 F9181.319
G1 X142.614 Y102.285 E.02912
G1 X142.687 Y102.883 E.03368
; LINE_WIDTH: 0.857776
G1 F9563.539
G1 X142.678 Y103.038 E.00834
G1 X142.634 Y103.109 E.00452
; LINE_WIDTH: 0.808901
G1 F9820.049
G1 X142.589 Y103.181 E.00425
; LINE_WIDTH: 0.760026
G1 F10237.511
G1 X142.495 Y103.277 E.00637
; LINE_WIDTH: 0.71335
G1 F10663.664
G1 X142.401 Y103.373 E.00596
; LINE_WIDTH: 0.666673
G1 F11098.483
G1 X142.307 Y103.469 E.00555
; LINE_WIDTH: 0.619996
G1 F12143.646
G1 X142.035 Y103.624 E.01195
G1 F12987.753
G1 X141.795 Y103.662 E.00928
G1 F13446.369
G1 X141.395 Y103.662 E.01527
G1 X138.299 Y103.662 E.11819
G1 X137.729 Y103.885 E.02337
G1 X136.924 Y104.089 E.03172
G1 X136.041 Y104.121 E.03372
G1 X135.155 Y103.946 E.03448
G1 X134.363 Y103.662 E.03214
G1 X132.796 Y103.662 E.05981
G3 X132.372 Y103.536 I0 J-.779 E.01714
G1 X132.112 Y103.254 E.01464
G1 X132.111 Y103.247 E.00026
M204 S250
G1 X132.598 Y102.991 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X132.574 Y102.842 E.00476
G1 X133.266 Y99.109 E.12021
G3 X131.527 Y98.946 I-.19 J-7.316 E.05541
G1 X131.038 Y98.936 E.0155
G3 X123.946 Y114.147 I-51.152 J-14.591 E.53357
G3 X121.747 Y117.121 I-28.18 J-18.532 E.11716
G1 X119.177 Y120.186 E.12664
G3 X112.01 Y126.704 I-40.297 J-37.116 E.30711
G1 X116.988 Y130.397 E.19624
G1 X117.728 Y131.083 E.03196
G3 X118.212 Y131.719 I-22.696 J17.75 E.0253
G1 X118.565 Y132.362 E.02323
G1 X118.834 Y133.079 E.02423
G1 X118.995 Y133.829 E.02428
G1 X119.044 Y134.593 E.02423
G1 X118.98 Y135.358 E.02429
G1 X118.825 Y136.02 E.02154
G1 X118.548 Y136.753 E.02482
G1 X118.31 Y137.19 E.01574
G3 X130.249 Y144.067 I-19.957 J48.445 E.43749
G3 X142.39 Y157.025 I-31.645 J41.817 E.56491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.207 E1.82007
G1 X144.167 Y98.938 E.00851
G1 X143.859 Y98.944 E.00977
G1 X142.881 Y99.101 E.03136
G3 X142.191 Y99.106 I-.82 J-70.086 E.02183
G1 X142.054 Y99.106 E.00434
G1 X141.918 Y99.106 E.00434
G1 X141.781 Y99.107 E.00434
G1 X141.644 Y99.107 E.00434
G1 F3450
G1 X141.507 Y99.108 E.00434
G1 F3300
G1 X141.39 Y99.108 E.00369
G1 X141.541 Y99.423 E.01107
G1 F3150
G1 X141.822 Y100.003 E.0204
G1 F3300
G1 X141.896 Y100.125 E.00451
G1 F3450
G1 X141.97 Y100.247 E.00451
G1 F3600
G1 X142.043 Y100.367 E.00445
G1 X142.067 Y100.595 E.00726
G1 F3450
G3 X142.079 Y100.848 I-.7 J.16 E.00806
G1 F3300
G1 X142.055 Y101.168 E.01017
G1 F3150
G1 X142.031 Y101.489 E.01017
G1 F3000
G3 X141.982 Y101.898 I-2.107 J-.043 E.01309
G1 F2475
G1 X141.92 Y102.324 E.0136
G1 F3000
G1 X141.931 Y102.381 E.00186
G1 F3600
G1 X141.942 Y102.439 E.00186
G1 X141.953 Y102.497 E.00186
G1 X141.963 Y102.555 E.00186
G1 X141.974 Y102.613 E.00186
G1 X141.985 Y102.671 E.00186
G1 X141.995 Y102.728 E.00186
G1 X142.006 Y102.786 E.00186
G3 X142.006 Y102.962 I-.312 J.088 E.00564
G1 X141.864 Y103.098 E.00621
G1 X141.795 Y103.109 E.00223
G1 X139.954 Y103.109 E.05828
G1 X139.713 Y103.109 E.00761
G1 X139.473 Y103.109 E.00761
G1 X139.232 Y103.109 E.00761
G1 X138.992 Y103.109 E.00761
G1 F3450
G1 X138.751 Y103.109 E.00761
G1 F3300
G1 X138.511 Y103.109 E.00761
G1 F3150
G1 X138.27 Y103.109 E.00761
G1 F3000
G1 X138.194 Y103.109 E.00241
G1 X137.527 Y103.37 E.02269
G1 X137.42 Y103.397 E.0035
G1 F3150
G1 X137.136 Y103.467 E.00926
G1 F3300
G1 X136.852 Y103.538 E.00926
G1 F3450
G1 X136.792 Y103.552 E.00194
G1 X136.112 Y103.567 E.02155
G1 F3600
G3 X136.014 Y103.564 I-.044 J-.208 E.00312
G1 F3450
G1 X135.8 Y103.519 E.00692
G1 F3300
G1 X135.586 Y103.473 E.00692
G1 F3150
G1 X135.372 Y103.427 E.00692
G1 F3000
G1 X135.275 Y103.406 E.00316
G1 X134.462 Y103.109 E.02742
G1 X134.342 Y103.109 E.00377
G1 F3600
G1 X134.149 Y103.109 E.00612
G1 X133.956 Y103.109 E.00612
G1 X133.763 Y103.109 E.00612
G1 X133.569 Y103.109 E.00612
G1 X133.376 Y103.109 E.00612
G1 X133.183 Y103.109 E.00612
G1 X132.99 Y103.109 E.00612
G1 X132.796 Y103.109 E.00612
G3 X132.656 Y103.06 I0 J-.226 E.00478
; WIPE_START
M204 S10000
G1 X132.574 Y102.842 E-.08853
G1 X132.714 Y102.088 E-.29147
; WIPE_END
G1 E-.02 F1800
G1 X131.265 Y102.6 Z9.88 F36000
G1 Z9.48
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.00642
G1 F8101.185
G1 X131.316 Y102.413 E.01224
; LINE_WIDTH: 0.974596
G1 F8375.344
G1 X131.4 Y102.089 E.02053
; LINE_WIDTH: 0.92603
G1 F8831.512
G1 X131.485 Y101.765 E.01947
; LINE_WIDTH: 0.877463
G1 F9340.235
G1 X131.57 Y101.441 E.01841
; LINE_WIDTH: 0.828896
G1 F9911.147
G1 X131.648 Y101.13 E.0166
; LINE_WIDTH: 0.789186
G1 F10432.537
G1 X131.725 Y100.819 E.01577
; LINE_WIDTH: 0.749476
G1 F11011.831
G1 X131.803 Y100.508 E.01494
; LINE_WIDTH: 0.709766
G1 F11659.24
G1 X131.881 Y100.197 E.01411
; WIPE_START
G1 X131.803 Y100.508 E-.12179
G1 X131.725 Y100.819 E-.12179
G1 X131.648 Y101.13 E-.12179
G1 X131.638 Y101.167 E-.01464
; WIPE_END
G1 E-.02 F1800
G1 X132.546 Y105.311 Z9.88 F36000
G1 Z9.48
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X131.45 Y104.931 I.299 J-2.632 E.04465
G1 X131.367 Y104.843 E.00462
G1 X130.969 Y105.816 E.04012
G1 X131.301 Y105.794 E.01269
G1 X131.772 Y106.088 E.02122
G3 X132.715 Y108.162 I-21.086 J10.835 E.087
G2 X136.485 Y111.865 I6.201 J-2.542 E.20739
G2 X138.842 Y112.491 I3.141 J-7.07 E.09348
G1 X139.313 Y112.197 E.02122
G2 X140.256 Y110.123 I-21.089 J-10.837 E.087
G3 X141.945 Y107.691 I6.766 J2.898 E.11384
G1 X141.945 Y113.889 E.23663
G2 X139.784 Y113.334 I-2.876 J6.719 E.08551
G1 X139.313 Y113.629 E.02122
G2 X138.371 Y115.703 I21.085 J10.835 E.087
G3 X134.6 Y119.406 I-6.201 J-2.542 E.20739
G3 X132.244 Y120.032 I-3.141 J-7.07 E.09348
G1 X131.772 Y119.737 E.02122
G3 X130.83 Y117.664 I21.086 J-10.835 E.087
G2 X126.783 Y113.859 I-6.104 J2.438 E.21895
G3 X123.83 Y118.095 I-73.838 J-48.326 E.19718
G3 X118.22 Y124.334 I-50.161 J-39.458 E.32056
G2 X121.404 Y126.946 I5.351 J-3.276 E.1602
G2 X123.761 Y127.573 I3.141 J-7.071 E.09348
G1 X124.232 Y127.278 E.02122
G2 X125.174 Y125.204 I-21.089 J-10.837 E.087
G3 X128.945 Y121.501 I6.201 J2.543 E.20739
G3 X131.301 Y120.875 I3.141 J7.072 E.09348
G1 X131.772 Y121.17 E.02122
G3 X132.715 Y123.243 I-21.089 J10.837 E.087
G2 X136.485 Y126.946 I6.201 J-2.543 E.20739
G2 X138.842 Y127.573 I3.141 J-7.071 E.09348
G1 X139.313 Y127.278 E.02122
G2 X140.256 Y125.204 I-21.089 J-10.837 E.087
G3 X141.945 Y122.772 I6.766 J2.898 E.11384
G1 X141.945 Y128.97 E.23663
G2 X139.784 Y128.416 I-2.876 J6.718 E.08551
G1 X139.313 Y128.71 E.02122
G2 X138.371 Y130.784 I21.082 J10.834 E.087
G3 X134.6 Y134.487 I-6.201 J-2.542 E.20739
G3 X132.244 Y135.113 I-3.141 J-7.071 E.09348
G1 X131.772 Y134.819 E.02122
G3 X130.83 Y132.745 I21.089 J-10.837 E.087
G2 X127.06 Y129.042 I-6.201 J2.542 E.20739
G2 X124.703 Y128.416 I-3.141 J7.07 E.09348
G1 X124.232 Y128.71 E.02122
G2 X123.289 Y130.784 I21.086 J10.835 E.087
G3 X121.195 Y133.537 I-6.035 J-2.418 E.13368
G3 X121.113 Y135.985 I-7.215 J.984 E.09398
G3 X126.151 Y138.617 I-36.207 J75.464 E.21706
G3 X128.945 Y136.583 I5.263 J4.292 E.13341
G3 X131.301 Y135.956 I3.141 J7.07 E.09348
G1 X131.772 Y136.251 E.02122
G3 X132.715 Y138.324 I-21.089 J10.837 E.087
G2 X136.485 Y142.027 I6.201 J-2.542 E.20739
G2 X138.842 Y142.654 I3.141 J-7.071 E.09348
G1 X139.313 Y142.359 E.02122
G2 X140.256 Y140.286 I-21.086 J-10.835 E.087
G3 X141.945 Y137.854 I6.766 J2.898 E.11384
G1 X141.945 Y144.051 E.23663
G2 X139.784 Y143.497 I-2.876 J6.719 E.08551
G1 X139.313 Y143.791 E.02122
G1 X138.842 Y144.762 E.0412
G3 X137.462 Y147.442 I-12.166 J-4.568 E.11533
G3 X139.077 Y149.139 I-30.545 J30.691 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 9.64
; LAYER_HEIGHT: 0.160001
; WIPE_START
G1 F13446.283
G1 X138.388 Y148.414 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L60
M991 S0 P59 ;notify layer change


G17
G3 Z9.88 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.188 Y103.946
G1 Z9.64
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.131 I-51.346 J-19.639 E.48257
G3 X123.072 Y118.225 I-29.872 J-19.715 E.147
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-42.466 J-39.032 E.29801
G1 X118.015 Y129.012 E.15056
G3 X118.933 Y129.849 I-4.613 J5.975 E.04749
G3 X120.076 Y131.531 I-5.86 J5.213 E.07787
G1 X120.45 Y132.48 E.03893
G1 X120.681 Y133.47 E.03882
G1 X120.765 Y134.487 E.03897
G1 X120.698 Y135.504 E.03889
G1 X120.538 Y136.268 E.02982
G3 X130.508 Y142.102 I-22.568 J50.007 E.44183
G3 X142.443 Y154.069 I-32.645 J44.492 E.64783
G1 X142.443 Y104.715 E1.88431
G1 X141.797 Y104.835 E.0251
G1 X138.069 Y104.835 E.14232
G1 X137.353 Y105.037 E.02841
G3 X134.63 Y104.835 I-.983 J-5.216 E.10545
G1 X132.794 Y104.835 E.07009
G3 X132.125 Y104.717 I0 J-1.95 E.02606
G1 X131.691 Y104.493 E.01864
G1 X131.249 Y104.012 E.02493
G1 X131.565 Y103.457 F36000
; LINE_WIDTH: 0.663821
G1 F10975.861
G1 X131.561 Y103.447 E.00045
; LINE_WIDTH: 0.707645
G1 F10940.335
G1 X131.514 Y103.334 E.00536
; LINE_WIDTH: 0.751469
G1 F10548.158
G1 X131.468 Y103.221 E.00571
; LINE_WIDTH: 0.795293
G1 F10163.139
G1 X131.421 Y103.108 E.00606
; LINE_WIDTH: 0.839118
G1 F9785.266
G1 X131.374 Y102.996 E.00641
; LINE_WIDTH: 0.882942
G1 F9279.929
G1 X131.327 Y102.883 E.00676
; LINE_WIDTH: 0.926766
G1 F8824.223
G1 X131.281 Y102.77 E.0071
; LINE_WIDTH: 0.968821
G1 F8427.103
G1 X131.267 Y102.705 E.00405
; LINE_WIDTH: 1.01088
G1 F8064.186
G1 X131.254 Y102.64 E.00423
G1 X131.184 Y102.76 E.00887
; LINE_WIDTH: 0.962016
G1 F8488.92
G1 X131.113 Y102.881 E.00843
; LINE_WIDTH: 0.913156
G1 F8960.88
G1 X131.043 Y103.001 E.00798
; LINE_WIDTH: 0.864296
G1 F9488.41
G1 X130.973 Y103.122 E.00754
; LINE_WIDTH: 0.815436
G1 F10081.937
G1 X130.903 Y103.242 E.0071
; LINE_WIDTH: 0.766576
G1 F10520.182
G1 X130.833 Y103.363 E.00665
; LINE_WIDTH: 0.717716
G1 F10967.778
G1 X130.763 Y103.483 E.00621
; LINE_WIDTH: 0.668856
G1 F11424.671
G1 X130.693 Y103.603 E.00576
; LINE_WIDTH: 0.619996
G1 F12787.77
G1 X130.548 Y103.976 E.01527
G1 F13446.369
G1 X130.331 Y104.531 E.02274
G3 X124.88 Y114.797 I-50.545 J-20.257 E.4446
G3 X122.623 Y117.848 I-29.326 J-19.334 E.14499
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-40.078 J-36.62 E.32256
G1 X117.666 Y129.482 E.17937
G3 X118.523 Y130.268 I-5.57 J6.934 E.04444
G3 X119.208 Y131.169 I-7.745 J6.599 E.0432
G1 X119.563 Y131.814 E.02811
G1 X119.901 Y132.684 E.03564
G1 X120.108 Y133.591 E.03555
G1 X120.18 Y134.523 E.03569
G1 X120.115 Y135.454 E.0356
G1 X119.933 Y136.284 E.03245
G1 X119.836 Y136.599 E.01259
G3 X130.932 Y143.157 I-21.125 J48.414 E.49334
G3 X142.92 Y155.769 I-32.375 J42.777 E.66728
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.43 E1.99749
G1 X142.695 Y103.912 E.02241
G1 X142.218 Y104.183 E.02094
G1 X141.797 Y104.249 E.01626
G1 X137.975 Y104.249 E.14592
G1 X137.466 Y104.416 E.02046
G1 X136.567 Y104.552 E.03471
G3 X135.529 Y104.479 I-.142 J-5.413 E.03978
G1 X134.721 Y104.249 E.03208
G1 X132.794 Y104.249 E.07358
G1 X132.326 Y104.167 E.01814
G1 X132.022 Y104.01 E.01304
G1 X131.879 Y103.854 E.0081
G1 X131.608 Y103.56 E.01527
; LINE_WIDTH: 0.663821
G1 F12510.242
G1 X131.6 Y103.54 E.00087
G1 X132.094 Y103.163 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.025 Y102.763 E.01549
G3 X132.236 Y101.621 I45.451 J7.817 E.04434
G1 X132.6 Y99.654 E.07636
G3 X131.451 Y99.494 I.238 J-5.913 E.04439
G3 X124.4 Y114.462 I-51.451 J-15.092 E.63425
G3 X122.174 Y117.472 I-28.761 J-18.94 E.14298
G1 X119.604 Y120.538 E.15272
G3 X112.93 Y126.698 I-39.759 J-36.383 E.34719
G1 X117.325 Y129.959 E.20895
G3 X118.151 Y130.726 I-6.626 J7.96 E.04305
G3 X118.695 Y131.451 I-12.172 J9.696 E.03461
G1 X119.05 Y132.096 E.02811
G1 X119.352 Y132.887 E.03235
G1 X119.535 Y133.712 E.03227
G1 X119.596 Y134.559 E.0324
G1 X119.531 Y135.403 E.03232
G1 X119.363 Y136.148 E.02917
G1 X119.08 Y136.91 E.03102
G3 X130.579 Y143.624 I-20.825 J48.872 E.5097
G3 X142.648 Y156.415 I-31.964 J42.247 E.67454
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y104.464 E1.97658
G1 X143.615 Y104.064 E.01527
; LINE_WIDTH: 0.666453
G1 F12282.784
G1 X143.591 Y103.828 E.00978
; LINE_WIDTH: 0.71291
G1 F11481.309
G1 X143.568 Y103.592 E.0105
; LINE_WIDTH: 0.759366
M73 P84 R3
G1 F10706.873
G1 X143.545 Y103.355 E.01122
; LINE_WIDTH: 0.807941
G1 F9959.479
G1 X143.521 Y103.233 E.00628
; LINE_WIDTH: 0.856516
G1 F9578.195
G1 X143.496 Y103.111 E.00667
G1 X143.499 Y102.647 E.02489
; LINE_WIDTH: 0.851606
G1 F9635.739
G1 X143.522 Y102.461 E.00997
; LINE_WIDTH: 0.805284
G1 F10214.698
G1 X143.545 Y102.276 E.0094
; LINE_WIDTH: 0.758962
G1 F10808.931
G1 X143.568 Y102.09 E.00884
; LINE_WIDTH: 0.71264
G1 F11419.962
G1 X143.591 Y101.904 E.00827
; LINE_WIDTH: 0.666318
G1 F12047.827
G1 X143.615 Y101.719 E.00771
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X143.615 Y101.319 E.01527
G1 X143.615 Y99.544 E.06776
G1 X142.968 Y99.649 E.025
G1 X142.132 Y99.661 E.03193
G1 X142.414 Y100.222 E.02397
G1 X142.481 Y100.783 E.02158
G1 X142.43 Y101.251 E.01799
G1 X142.387 Y101.649 E.01527
G1 F12672.572
G1 X142.393 Y101.831 E.00694
; LINE_WIDTH: 0.666318
G1 F12046.124
G1 X142.45 Y102.009 E.00771
; LINE_WIDTH: 0.71264
G1 F11418.331
G1 X142.507 Y102.187 E.00827
; LINE_WIDTH: 0.758962
G1 F10807.337
G1 X142.563 Y102.366 E.00884
; LINE_WIDTH: 0.805284
G1 F10213.134
G1 X142.62 Y102.544 E.0094
; LINE_WIDTH: 0.851606
G1 F9635.739
G1 X142.676 Y102.722 E.00997
; LINE_WIDTH: 0.856516
G1 F9578.195
G1 X142.68 Y103.04 E.01704
G1 X142.636 Y103.111 E.0045
; LINE_WIDTH: 0.807941
G1 F9834.665
G1 X142.591 Y103.183 E.00424
; LINE_WIDTH: 0.759366
G1 F10252.306
G1 X142.497 Y103.279 E.00636
; LINE_WIDTH: 0.71291
G1 F10678.608
G1 X142.403 Y103.375 E.00595
; LINE_WIDTH: 0.666453
G1 F11113.57
G1 X142.309 Y103.471 E.00554
; LINE_WIDTH: 0.619996
G1 F12159.397
G1 X142.037 Y103.626 E.01195
G1 F13004.043
G1 X141.797 Y103.664 E.00928
G1 F13446.369
G1 X141.397 Y103.664 E.01527
G1 X137.881 Y103.664 E.13425
G1 X137.336 Y103.844 E.02189
G1 X136.489 Y103.972 E.03274
G1 X135.592 Y103.897 E.03433
G1 X134.813 Y103.664 E.03108
G1 X132.794 Y103.664 E.07708
G3 X132.353 Y103.527 I0 J-.778 E.01787
G1 X132.11 Y103.258 E.01387
G1 X132.109 Y103.251 E.00024
M204 S250
G1 X132.596 Y102.993 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X132.572 Y102.844 E.00478
G1 X133.263 Y99.111 E.12021
G3 X131.527 Y98.946 I-.188 J-7.236 E.05536
G1 X131.038 Y98.936 E.01549
G3 X123.946 Y114.147 I-51.152 J-14.591 E.53357
G3 X121.747 Y117.121 I-28.179 J-18.532 E.11717
G1 X119.177 Y120.187 E.12664
G3 X112.01 Y126.704 I-38.873 J-35.551 E.30714
G1 X116.99 Y130.398 E.19633
G3 X117.739 Y131.094 I-6.463 J7.703 E.03237
G3 X118.21 Y131.717 I-28.528 J22.083 E.02474
G1 X118.565 Y132.362 E.02331
G1 X118.834 Y133.08 E.02425
G1 X118.994 Y133.827 E.02419
G1 X119.044 Y134.593 E.0243
G1 X118.98 Y135.356 E.02423
G1 X118.825 Y136.021 E.02162
G1 X118.548 Y136.752 E.02477
G1 X118.31 Y137.19 E.01577
G3 X130.249 Y144.067 I-19.958 J48.448 E.43748
G3 X142.39 Y157.025 I-31.646 J41.818 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.207 E1.82008
G1 X144.167 Y98.938 E.0085
G1 X143.859 Y98.944 E.00977
G1 X142.88 Y99.103 E.03141
G3 X141.643 Y99.109 I-1.884 J-248.008 E.03915
G1 X141.511 Y99.11 E.00418
G1 X141.379 Y99.11 E.00418
G1 X141.327 Y99.11 E.00163
G1 X141.337 Y99.159 E.00159
G1 X141.354 Y99.24 E.00261
G1 F3450
G1 X141.37 Y99.321 E.00261
G1 F3300
G1 X141.387 Y99.401 E.00261
G1 F3150
G1 X141.831 Y100.258 E.03056
G1 F3000
G1 X141.859 Y100.312 E.00193
G1 X141.926 Y100.808 E.01584
G1 X141.878 Y101.236 E.01363
G1 F2475
G1 X141.818 Y101.76 E.01669
G1 X141.83 Y101.824 E.00208
G1 F3000
G1 X141.851 Y101.938 E.00366
G1 F3600
G1 X141.872 Y102.051 E.00366
G1 X141.893 Y102.165 E.00366
G1 X141.914 Y102.279 E.00366
G1 X141.935 Y102.392 E.00366
G1 X141.956 Y102.506 E.00366
G1 X141.977 Y102.619 E.00366
G1 X141.998 Y102.733 E.00366
G3 X142.008 Y102.964 I-.405 J.134 E.00744
G1 X141.867 Y103.1 E.00621
G1 X141.797 Y103.111 E.00223
G1 X139.554 Y103.111 E.071
G1 X139.317 Y103.111 E.00752
G1 X139.08 Y103.111 E.00752
G1 X138.842 Y103.111 E.00752
G1 X138.605 Y103.111 E.00752
G1 F3450
G1 X138.367 Y103.111 E.00752
G1 F3300
G1 X138.13 Y103.111 E.00752
G1 F3150
G1 X137.892 Y103.111 E.00752
G1 F3000
G1 X137.792 Y103.111 E.00319
G1 X137.163 Y103.319 E.02097
G1 X136.415 Y103.424 E.02393
G1 X135.652 Y103.347 E.02426
G1 X134.875 Y103.111 E.02573
G1 F3150
G1 X134.819 Y103.111 E.00176
G1 F3300
G1 X134.627 Y103.111 E.00609
G1 F3450
G1 X134.321 Y103.111 E.00967
G1 F3600
G1 X134.016 Y103.111 E.00967
G1 X133.71 Y103.111 E.00967
G1 X133.405 Y103.111 E.00967
G1 X133.099 Y103.111 E.00967
G1 X132.794 Y103.111 E.00967
G3 X132.654 Y103.062 I0 J-.226 E.00477
; WIPE_START
M204 S10000
G1 X132.572 Y102.844 E-.08861
G1 X132.712 Y102.09 E-.29139
; WIPE_END
G1 E-.02 F1800
G1 X140.339 Y101.815 Z10.04 F36000
G1 X142.999 Y101.719 Z10.04
G1 Z9.64
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.680076
G1 F12195.312
G1 X143.021 Y101.327 E.01651
; LINE_WIDTH: 0.635636
G1 F13096.625
G1 X143.043 Y100.936 E.01537
; LINE_WIDTH: 0.631656
G1 F13183.889
G1 X143.023 Y100.268 E.02599
; LINE_WIDTH: 0.635076
G1 F13108.834
G1 X143.021 Y100.24 E.0011
; WIPE_START
G1 X143.023 Y100.268 E-.01068
G1 X143.043 Y100.936 E-.25363
G1 X143.026 Y101.239 E-.11569
; WIPE_END
G1 E-.02 F1800
G1 X135.447 Y102.141 Z10.04 F36000
G1 X131.254 Y102.64 Z10.04
G1 Z9.64
G1 E.4 F1800
; LINE_WIDTH: 1.01088
G1 F8064.186
G1 X131.263 Y102.602 E.0025
; LINE_WIDTH: 1.00454
G1 F8116.883
G1 X131.338 Y102.32 E.01841
; LINE_WIDTH: 0.961326
G1 F8495.238
G1 X131.412 Y102.039 E.01759
; LINE_WIDTH: 0.918116
G1 F8910.589
G1 X131.486 Y101.757 E.01677
; LINE_WIDTH: 0.874906
G1 F9368.643
G1 X131.56 Y101.476 E.01595
; LINE_WIDTH: 0.831696
G1 F9876.343
G1 X131.569 Y101.441 E.00188
; LINE_WIDTH: 0.826796
G1 F9937.411
G1 X131.647 Y101.13 E.01655
; LINE_WIDTH: 0.787106
G1 F10461.363
G1 X131.724 Y100.819 E.01572
; LINE_WIDTH: 0.747416
G1 F11043.643
G1 X131.802 Y100.508 E.01489
; LINE_WIDTH: 0.707726
G1 F11694.561
G1 X131.88 Y100.197 E.01406
; WIPE_START
G1 X131.802 Y100.508 E-.12174
G1 X131.724 Y100.819 E-.12174
G1 X131.647 Y101.13 E-.12174
G1 X131.637 Y101.168 E-.0148
; WIPE_END
G1 E-.02 F1800
G1 X132.158 Y105.237 Z10.04 F36000
G1 Z9.64
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X131.399 Y104.9 I.343 J-1.797 E.03198
G1 X131.361 Y104.859 E.00215
G1 X130.811 Y106.203 E.05543
G1 X131.301 Y106.341 E.01944
G1 X131.772 Y106.767 E.02426
G3 X132.715 Y108.422 I-13.16 J8.593 E.07274
G2 X138.371 Y112.084 I5.859 J-2.851 E.27034
G1 X138.842 Y111.944 E.01877
G2 X139.784 Y110.748 I-1.561 J-2.201 E.05892
G3 X141.945 Y107.696 I8.072 J3.423 E.1439
G1 X141.945 Y114.027 E.24174
G2 X140.256 Y113.742 I-1.568 J4.132 E.06584
G1 X139.784 Y113.882 E.01877
G1 X139.313 Y114.308 E.02426
G2 X138.371 Y115.963 I13.156 J8.591 E.07274
G3 X132.715 Y119.625 I-5.859 J-2.851 E.27034
G1 X132.244 Y119.484 E.01877
G1 X131.772 Y119.058 E.02426
G3 X130.83 Y117.404 I13.16 J-8.593 E.07274
G2 X126.699 Y113.981 I-5.765 J2.754 E.21136
G3 X123.83 Y118.095 I-71.036 J-46.488 E.19153
G3 X118.172 Y124.384 I-50.484 J-39.723 E.32321
G2 X123.289 Y127.165 I5.358 J-3.76 E.23042
G1 X123.761 Y127.025 E.01877
G1 X124.232 Y126.599 E.02426
G2 X125.174 Y124.944 I-13.16 J-8.593 E.07274
G3 X130.83 Y121.282 I5.859 J2.851 E.27034
G1 X131.301 Y121.423 E.01877
G1 X131.772 Y121.849 E.02426
G3 X132.715 Y123.503 I-13.16 J8.593 E.07274
G2 X138.371 Y127.165 I5.859 J-2.851 E.27034
G1 X138.842 Y127.025 E.01877
G2 X139.784 Y125.83 I-1.561 J-2.201 E.05892
G3 X141.945 Y122.777 I8.072 J3.423 E.1439
G1 X141.945 Y129.109 E.24174
G2 X140.256 Y128.823 I-1.568 J4.132 E.06584
G1 X139.784 Y128.963 E.01877
G1 X139.313 Y129.389 E.02426
G2 X138.371 Y131.044 I13.158 J8.592 E.07274
G3 X132.715 Y134.706 I-5.859 J-2.851 E.27034
G1 X132.244 Y134.566 E.01877
G1 X131.772 Y134.139 E.02426
G3 X130.83 Y132.485 I13.159 J-8.592 E.07274
G2 X125.174 Y128.823 I-5.859 J2.851 E.27034
G1 X124.703 Y128.963 E.01877
G1 X124.232 Y129.389 E.02426
G2 X123.289 Y131.044 I13.16 J8.593 E.07274
G3 X121.191 Y133.51 I-5.818 J-2.825 E.12493
G3 X121.113 Y135.985 I-7.27 J1.008 E.09502
G3 X126.117 Y138.597 I-37.558 J78.086 E.21556
G3 X130.83 Y136.364 I4.918 J4.291 E.20481
G1 X131.301 Y136.504 E.01877
G1 X131.772 Y136.93 E.02426
G3 X132.715 Y138.585 I-13.16 J8.593 E.07274
G2 X138.371 Y142.246 I5.859 J-2.851 E.27034
G1 X138.842 Y142.106 E.01877
G2 X139.784 Y140.911 I-1.561 J-2.201 E.05892
G3 X141.945 Y137.858 I8.072 J3.423 E.1439
G1 X141.945 Y144.19 E.24174
G2 X140.256 Y143.904 I-1.568 J4.132 E.06584
G1 X139.784 Y144.045 E.01877
G2 X138.842 Y145.24 I1.561 J2.201 E.05892
G3 X137.49 Y147.47 I-12.757 J-6.209 E.09972
G3 X139.105 Y149.167 I-30.973 J31.093 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 9.8
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X138.415 Y148.443 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L61
M991 S0 P60 ;notify layer change


G17
G3 Z10.04 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.187 Y103.949
G1 Z9.8
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.132 I-51.323 J-19.63 E.48248
G3 X123.072 Y118.225 I-29.872 J-19.716 E.14697
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-40.976 J-37.465 E.29802
G1 X118.015 Y129.012 E.15056
G3 X118.932 Y129.849 I-4.609 J5.97 E.04745
G3 X120.076 Y131.532 I-5.874 J5.223 E.07792
G1 X120.45 Y132.48 E.0389
G3 X120.765 Y134.488 I-7.468 J2.198 E.07783
G1 X120.698 Y135.504 E.03887
G1 X120.538 Y136.268 E.02982
G3 X131.286 Y142.69 I-21.787 J48.669 E.4791
G3 X142.443 Y154.069 I-32.924 J43.442 E.61064
G1 X142.443 Y104.718 E1.88417
G1 X141.8 Y104.837 E.02499
G1 X137.507 Y104.837 E.16389
G1 X136.938 Y104.953 E.02217
G1 X136.057 Y104.972 E.03364
G3 X135.193 Y104.837 I.165 J-3.876 E.03346
G1 X132.791 Y104.837 E.0917
G3 X132.122 Y104.719 I0 J-1.949 E.02609
G1 X131.687 Y104.495 E.01867
G1 X131.248 Y104.015 E.02483
G1 X131.56 Y103.438 F36000
; LINE_WIDTH: 0.716236
G1 F10695.533
G1 X131.518 Y103.324 E.00543
; LINE_WIDTH: 0.764356
G1 F10308.027
G1 X131.475 Y103.206 E.00599
; LINE_WIDTH: 0.812476
G1 F9915.671
G1 X131.432 Y103.087 E.00639
; LINE_WIDTH: 0.860596
G1 F9530.9
G1 X131.389 Y102.969 E.00678
; LINE_WIDTH: 0.908716
G1 F9006.382
G1 X131.345 Y102.851 E.00718
; LINE_WIDTH: 0.956836
G1 F8536.587
G1 X131.302 Y102.732 E.00757
; LINE_WIDTH: 1.00496
G1 F8113.372
G1 X131.259 Y102.614 E.00797
G1 X131.188 Y102.737 E.00898
; LINE_WIDTH: 0.956836
G1 F8536.587
G1 X131.118 Y102.861 E.00854
; LINE_WIDTH: 0.908716
G1 F9006.382
G1 X131.047 Y102.984 E.00809
; LINE_WIDTH: 0.860596
G1 F9530.9
G1 X130.976 Y103.107 E.00765
; LINE_WIDTH: 0.812476
G1 F10120.287
G1 X130.906 Y103.23 E.0072
; LINE_WIDTH: 0.764356
G1 F10567.66
G1 X130.835 Y103.353 E.00676
; LINE_WIDTH: 0.716236
G1 F11024.726
G1 X130.765 Y103.476 E.00631
; LINE_WIDTH: 0.668116
G1 F11491.451
G1 X130.694 Y103.6 E.00587
; LINE_WIDTH: 0.619996
G1 F12858.416
G1 X130.549 Y103.972 E.01527
G1 F13446.369
G1 X130.331 Y104.531 E.0229
G3 X124.88 Y114.797 I-50.514 J-20.24 E.44463
G3 X122.623 Y117.848 I-29.312 J-19.324 E.14496
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-40.113 J-36.657 E.32256
G1 X117.666 Y129.482 E.17937
G3 X118.523 Y130.268 I-5.361 J6.707 E.04441
G3 X119.21 Y131.171 I-7.801 J6.643 E.04334
G1 X119.563 Y131.814 E.02801
G1 X119.901 Y132.683 E.03561
G3 X120.136 Y133.793 I-9.666 J2.621 E.04331
G1 X120.18 Y134.524 E.02796
G1 X120.115 Y135.453 E.03559
G1 X119.932 Y136.284 E.03246
G1 X119.836 Y136.599 E.01258
G3 X130.932 Y143.157 I-21.054 J48.293 E.49335
G3 X142.92 Y155.769 I-32.361 J42.763 E.66728
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.435 E1.99727
G1 X142.697 Y103.914 E.02224
G1 X142.22 Y104.185 E.02094
G3 X141.44 Y104.251 I-.6 J-2.434 E.03
G1 X137.44 Y104.251 E.15272
G1 X136.878 Y104.37 E.02195
G1 X136.049 Y104.387 E.03166
G3 X135.255 Y104.251 I.2 J-3.564 E.0308
G1 X132.791 Y104.251 E.09408
G1 X132.323 Y104.169 E.01816
G1 X132.019 Y104.012 E.01306
G1 X131.875 Y103.855 E.00812
G1 X131.605 Y103.56 E.01527
; LINE_WIDTH: 0.668116
G1 F12425.448
G1 X131.591 Y103.523 E.00165
G1 X132.091 Y103.165 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.025 Y102.751 E.01601
G1 X132.598 Y99.656 E.12016
G3 X131.451 Y99.494 I.239 J-5.848 E.04431
G3 X124.399 Y114.463 I-51.367 J-15.053 E.63427
G3 X122.174 Y117.472 I-28.762 J-18.941 E.14296
G1 X119.604 Y120.538 E.15272
G3 X112.93 Y126.698 I-39.743 J-36.366 E.34719
G1 X117.317 Y129.953 E.20857
G3 X118.151 Y130.725 I-6.5 J7.846 E.04341
G3 X118.696 Y131.453 I-12.327 J9.811 E.03474
G1 X119.05 Y132.096 E.02801
M73 P84 R2
G1 X119.352 Y132.887 E.03233
G3 X119.551 Y133.828 I-11.824 J2.989 E.03674
G1 X119.596 Y134.559 E.02796
G1 X119.531 Y135.403 E.03231
G1 X119.363 Y136.149 E.02918
G1 X119.08 Y136.91 E.03101
G3 X130.579 Y143.624 I-20.75 J48.744 E.50971
G3 X142.648 Y156.415 I-31.956 J42.239 E.67454
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y104.465 E1.97655
G1 X143.615 Y104.065 E.01527
; LINE_WIDTH: 0.666233
G1 F12296.622
G1 X143.591 Y103.829 E.00977
; LINE_WIDTH: 0.71247
G1 F11495.485
G1 X143.568 Y103.593 E.01048
; LINE_WIDTH: 0.758706
G1 F10721.332
G1 X143.545 Y103.357 E.0112
; LINE_WIDTH: 0.806976
G1 F9974.165
G1 X143.521 Y103.235 E.00626
; LINE_WIDTH: 0.855246
G1 F9593.014
G1 X143.497 Y103.113 E.00666
G1 X143.499 Y102.649 E.02483
; LINE_WIDTH: 0.850366
G1 F9650.381
G1 X143.522 Y102.358 E.01552
; LINE_WIDTH: 0.804292
G1 F10227.858
G1 X143.545 Y102.068 E.01464
; LINE_WIDTH: 0.758218
G1 F10878.845
G1 X143.568 Y101.777 E.01376
; LINE_WIDTH: 0.712144
G1 F11618.336
G1 X143.592 Y101.486 E.01289
; LINE_WIDTH: 0.66607
G1 F12465.691
G1 X143.615 Y101.195 E.01201
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X143.615 Y100.795 E.01527
G1 X143.615 Y99.545 E.04775
G1 X142.968 Y99.65 E.02501
G3 X141.994 Y99.663 I-.841 J-27.262 E.03719
G1 X142.243 Y100.281 E.02542
G1 X142.323 Y100.812 E.02051
G1 X142.3 Y101.316 E.01926
; LINE_WIDTH: 0.66607
G1 F12465.691
G1 X142.376 Y101.598 E.01201
; LINE_WIDTH: 0.712144
G1 F11618.336
G1 X142.451 Y101.879 E.01289
; LINE_WIDTH: 0.758218
G1 F10878.845
G1 X142.527 Y102.161 E.01376
; LINE_WIDTH: 0.804292
G1 F10227.858
G1 X142.603 Y102.443 E.01464
; LINE_WIDTH: 0.850366
G1 F9650.381
G1 X142.678 Y102.725 E.01552
; LINE_WIDTH: 0.855246
G1 F9593.014
G1 X142.682 Y103.042 E.017
G1 X142.638 Y103.113 E.00449
; LINE_WIDTH: 0.806976
G1 F10192.328
G1 X142.594 Y103.185 E.00423
; LINE_WIDTH: 0.758706
G1 F10617.285
G1 X142.5 Y103.281 E.00635
; LINE_WIDTH: 0.71247
G1 F11050.875
G1 X142.406 Y103.377 E.00594
; LINE_WIDTH: 0.666233
G1 F11493.142
G1 X142.312 Y103.473 E.00554
; LINE_WIDTH: 0.619996
G1 F12556.281
G1 X142.04 Y103.628 E.01195
G1 F13414.371
G1 X141.8 Y103.666 E.00928
G1 F13446.369
G1 X141.374 Y103.666 E.01625
G1 X137.374 Y103.666 E.15272
G1 X136.818 Y103.788 E.02172
G1 X136.041 Y103.801 E.02968
G1 X135.318 Y103.666 E.02809
G1 X132.791 Y103.666 E.09646
G3 X132.351 Y103.529 I0 J-.778 E.01789
G1 X132.106 Y103.257 E.01397
G1 X132.106 Y103.254 E.00013
M204 S250
G1 X132.593 Y102.994 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X132.569 Y102.846 E.00475
G1 X133.261 Y99.113 E.12021
G3 X131.526 Y98.946 I-.186 J-7.159 E.05531
G1 X131.038 Y98.936 E.01546
G3 X123.945 Y114.147 I-51.152 J-14.591 E.53359
G3 X121.747 Y117.121 I-28.175 J-18.529 E.11715
G1 X119.177 Y120.187 E.12664
G3 X112.01 Y126.704 I-40.295 J-37.115 E.30711
G1 X116.988 Y130.397 E.19624
G3 X117.738 Y131.094 I-6.406 J7.649 E.03244
G3 X118.212 Y131.719 I-29.076 J22.491 E.02485
G1 X118.565 Y132.362 E.02323
G1 X118.834 Y133.079 E.02424
G3 X118.999 Y133.862 I-26.721 J6.051 E.02533
G1 X119.044 Y134.598 E.02333
G1 X118.98 Y135.355 E.02408
G1 X118.825 Y136.021 E.02163
G1 X118.548 Y136.753 E.0248
G1 X118.31 Y137.19 E.01573
G3 X130.249 Y144.067 I-19.829 J48.224 E.4375
G3 X142.39 Y157.025 I-31.645 J41.817 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.206 E1.82009
G1 X144.167 Y98.938 E.00849
G1 X143.859 Y98.944 E.00976
G1 X142.879 Y99.105 E.03145
G3 X141.33 Y99.112 I-1.803 J-219.084 E.04904
G1 X141.349 Y99.216 E.00333
G1 X141.368 Y99.319 E.00333
G1 X141.387 Y99.422 E.00333
G1 X141.407 Y99.526 E.00333
G1 X141.426 Y99.629 E.00333
G1 X141.445 Y99.732 E.00333
G1 X141.464 Y99.836 E.00333
G2 X141.515 Y99.997 I.319 J-.013 E.0054
G1 F3000
G1 X141.698 Y100.376 E.01333
G1 X141.769 Y100.837 E.01479
G1 X141.727 Y101.208 E.01181
G1 F2475
G1 X141.725 Y101.245 E.00119
G1 F3000
G1 X141.758 Y101.423 E.00573
G1 F3600
G1 X141.791 Y101.602 E.00573
G1 X141.824 Y101.78 E.00573
G1 X141.857 Y101.958 E.00573
G1 X141.89 Y102.136 E.00573
G1 X141.923 Y102.314 E.00573
G1 X141.956 Y102.492 E.00573
G1 X141.989 Y102.67 E.00573
G3 X142.011 Y102.966 I-.513 J.187 E.00953
G1 X141.869 Y103.102 E.00621
G1 X141.8 Y103.113 E.00223
G1 X139.077 Y103.113 E.08618
G1 X138.844 Y103.113 E.0074
G1 X138.61 Y103.113 E.0074
G1 X138.376 Y103.113 E.0074
G1 X138.142 Y103.113 E.0074
G1 F3450
G1 X137.909 Y103.113 E.0074
G1 F3300
G1 X137.675 Y103.113 E.0074
G1 F3150
G1 X137.441 Y103.113 E.0074
G1 F3000
G1 X137.311 Y103.113 E.00411
G1 X136.783 Y103.234 E.01716
G1 X136.033 Y103.249 E.02374
G1 X135.338 Y103.113 E.02244
G1 F3600
G1 X135.269 Y103.113 E.00216
G1 X135.201 Y103.113 E.00216
G1 X135.133 Y103.113 E.00216
G1 X135.064 Y103.113 E.00216
G1 X134.996 Y103.113 E.00216
G1 X134.928 Y103.113 E.00216
G1 X134.86 Y103.113 E.00216
G1 X132.791 Y103.113 E.06548
G3 X132.651 Y103.064 I0 J-.226 E.0048
; WIPE_START
M204 S10000
G1 X132.569 Y102.846 E-.08832
G1 X132.709 Y102.091 E-.29168
; WIPE_END
G1 E-.02 F1800
G1 X140.313 Y101.426 Z10.2 F36000
G1 X142.952 Y101.195 Z10.2
G1 Z9.8
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.774076
G1 F10645.633
G1 X142.958 Y100.566 E.03037
; LINE_WIDTH: 0.801256
G1 F10268.347
G1 X142.938 Y100.334 E.01163
; WIPE_START
G1 X142.958 Y100.566 E-.10252
G1 X142.952 Y101.195 E-.27748
; WIPE_END
G1 E-.02 F1800
G1 X135.375 Y102.115 Z10.2 F36000
G1 X131.259 Y102.614 Z10.2
G1 Z9.8
G1 E.4 F1800
; LINE_WIDTH: 1.00496
G1 F8113.372
G1 X131.314 Y102.413 E.0132
; LINE_WIDTH: 0.970456
G1 F8412.384
G1 X131.398 Y102.089 E.02043
; LINE_WIDTH: 0.921916
G1 F8872.44
G1 X131.483 Y101.765 E.01937
; LINE_WIDTH: 0.873376
G1 F9385.727
G1 X131.568 Y101.441 E.01831
; LINE_WIDTH: 0.824836
G1 F9962.049
G1 X131.645 Y101.13 E.01651
; LINE_WIDTH: 0.785131
G1 F10488.882
G1 X131.723 Y100.819 E.01568
; LINE_WIDTH: 0.745426
G1 F11074.549
G1 X131.801 Y100.509 E.01485
; LINE_WIDTH: 0.705721
G1 F11729.485
G1 X131.879 Y100.198 E.01403
; WIPE_START
G1 X131.801 Y100.509 E-.12177
G1 X131.723 Y100.819 E-.12177
G1 X131.645 Y101.13 E-.12177
G1 X131.636 Y101.168 E-.01469
; WIPE_END
G1 E-.02 F1800
G1 X131.864 Y105.143 Z10.2 F36000
G1 Z9.8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X131.359 Y104.862 E.02203
G3 X130.685 Y106.491 I-30.845 J-11.828 E.0673
G3 X131.772 Y107.184 I-.435 J1.882 E.05022
G1 X132.715 Y108.635 E.06604
G2 X137.428 Y111.823 I5.485 J-3.03 E.2255
G2 X139.313 Y111.101 I.317 J-1.995 E.08071
G1 X140.256 Y109.65 E.06604
G3 X141.945 Y107.694 I8.082 J5.271 E.09901
G1 X141.945 Y114.155 E.2467
G1 X141.198 Y114.003 E.02911
G2 X139.313 Y114.725 I-.317 J1.995 E.08071
G1 X138.371 Y116.175 E.06604
G3 X133.658 Y119.364 I-5.485 J-3.03 E.2255
G3 X131.772 Y118.642 I-.317 J-1.995 E.08071
M73 P85 R2
G1 X130.83 Y117.191 E.06604
G2 X126.626 Y114.107 I-5.409 J2.965 E.20552
G3 X123.83 Y118.095 I-80.883 J-53.737 E.186
G3 X118.119 Y124.432 I-49.507 J-38.877 E.32594
G2 X122.347 Y126.904 I4.995 J-3.691 E.19222
G2 X124.232 Y126.182 I.317 J-1.995 E.08071
G1 X125.174 Y124.732 E.06604
G3 X129.887 Y121.543 I5.485 J3.03 E.2255
G3 X131.772 Y122.265 I.317 J1.995 E.08071
G1 X132.715 Y123.716 E.06604
G2 X137.428 Y126.904 I5.485 J-3.03 E.2255
G2 X139.313 Y126.182 I.317 J-1.995 E.08071
G1 X140.256 Y124.732 E.06604
G3 X141.945 Y122.775 I8.082 J5.271 E.09901
G1 X141.945 Y129.236 E.2467
G1 X141.198 Y129.084 E.02911
G2 X139.313 Y129.806 I-.317 J1.995 E.08071
G1 X138.371 Y131.256 E.06604
G3 X133.658 Y134.445 I-5.485 J-3.03 E.2255
G3 X131.772 Y133.723 I-.317 J-1.995 E.08071
G1 X130.83 Y132.272 E.06604
G2 X126.117 Y129.084 I-5.485 J3.03 E.2255
G2 X124.232 Y129.806 I-.317 J1.995 E.08071
G1 X123.289 Y131.256 E.06604
G3 X121.186 Y133.49 I-5.555 J-3.123 E.11829
G3 X121.113 Y135.985 I-6.649 J1.052 E.09585
G3 X126.084 Y138.576 I-38.569 J80.09 E.21407
G3 X129.887 Y136.625 I4.576 J4.236 E.16657
G3 X131.772 Y137.347 I.317 J1.995 E.08071
G1 X132.715 Y138.797 E.06604
G2 X137.428 Y141.986 I5.485 J-3.03 E.2255
G2 X139.313 Y141.263 I.317 J-1.995 E.08071
G1 X140.256 Y139.813 E.06604
G3 X141.945 Y137.856 I8.081 J5.27 E.09901
G1 X141.945 Y144.318 E.2467
G1 X141.198 Y144.165 E.02911
G2 X139.313 Y144.887 I-.317 J1.995 E.08071
G3 X137.521 Y147.502 I-20.467 J-12.112 E.12111
G3 X139.134 Y149.2 I-30.372 J30.474 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 9.96
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X138.445 Y148.475 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L62
M991 S0 P61 ;notify layer change


G17
G3 Z10.2 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.186 Y103.952
G1 Z9.96
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.132 I-51.363 J-19.655 E.48235
G3 X123.072 Y118.224 I-29.864 J-19.71 E.14697
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-41.989 J-38.53 E.29802
G1 X118.015 Y129.012 E.15055
G1 X118.485 Y129.402 E.02333
G1 X119.091 Y130.034 E.03345
G3 X119.749 Y130.931 I-4.531 J4.011 E.04251
G1 X120.215 Y131.838 E.03891
G1 X120.478 Y132.566 E.02957
G1 X120.683 Y133.478 E.03568
G1 X120.765 Y134.489 E.03875
G1 X120.698 Y135.504 E.03882
G1 X120.538 Y136.268 E.02981
G3 X130.504 Y142.099 I-22.373 J49.673 E.44166
G3 X142.443 Y154.069 I-32.715 J44.57 E.64801
G1 X142.443 Y104.722 E1.88404
G1 X141.802 Y104.839 E.02489
G1 X132.789 Y104.839 E.34412
G3 X132.118 Y104.72 I0 J-1.949 E.02612
G1 X131.684 Y104.496 E.01869
G1 X131.246 Y104.018 E.02471
G1 X131.557 Y103.44 F36000
; LINE_WIDTH: 0.715826
G1 F10706.428
G1 X131.515 Y103.325 E.00542
; LINE_WIDTH: 0.763741
G1 F10318.794
G1 X131.472 Y103.207 E.00598
; LINE_WIDTH: 0.811656
G1 F9926.983
G1 X131.429 Y103.089 E.00637
; LINE_WIDTH: 0.859571
G1 F9542.738
G1 X131.386 Y102.971 E.00676
; LINE_WIDTH: 0.907486
G1 F9019.07
G1 X131.343 Y102.853 E.00716
; LINE_WIDTH: 0.955401
G1 F8549.886
G1 X131.3 Y102.734 E.00755
; LINE_WIDTH: 1.00332
G1 F8127.103
G1 X131.257 Y102.616 E.00794
G1 X131.187 Y102.739 E.00895
; LINE_WIDTH: 0.955401
G1 F8549.886
G1 X131.116 Y102.862 E.0085
; LINE_WIDTH: 0.907486
G1 F9019.07
G1 X131.046 Y102.985 E.00806
; LINE_WIDTH: 0.859571
G1 F9542.738
G1 X130.976 Y103.108 E.00762
; LINE_WIDTH: 0.811656
G1 F10130.964
G1 X130.905 Y103.231 E.00718
; LINE_WIDTH: 0.763741
G1 F10577.464
G1 X130.835 Y103.354 E.00673
; LINE_WIDTH: 0.715826
G1 F11033.592
G1 X130.765 Y103.477 E.00629
; LINE_WIDTH: 0.667911
G1 F11499.32
G1 X130.694 Y103.6 E.00585
; LINE_WIDTH: 0.619996
G1 F12866.74
G1 X130.549 Y103.972 E.01527
G1 F13446.369
G1 X130.331 Y104.531 E.02289
G3 X124.88 Y114.797 I-50.554 J-20.261 E.44464
G3 X122.623 Y117.848 I-29.311 J-19.323 E.14496
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-40.143 J-36.689 E.32257
G1 X117.666 Y129.482 E.17936
G1 X118.107 Y129.85 E.02195
G1 X118.662 Y130.433 E.03071
G1 X119.253 Y131.243 E.03829
G1 X119.688 Y132.094 E.03651
G1 X119.923 Y132.754 E.02672
G1 X120.109 Y133.597 E.03298
G1 X120.18 Y134.525 E.03552
G1 X120.115 Y135.454 E.03554
G1 X119.932 Y136.284 E.03246
G1 X119.836 Y136.599 E.01258
G3 X130.932 Y143.157 I-21.04 J48.27 E.49335
G3 X142.92 Y155.769 I-32.175 J42.587 E.66731
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.441 E1.99705
G1 X142.7 Y103.917 E.02208
G1 X142.223 Y104.187 E.02094
G1 X141.802 Y104.254 E.01626
G1 X132.789 Y104.254 E.34412
G1 X132.32 Y104.17 E.01818
G1 X132.016 Y104.013 E.01308
G1 X131.872 Y103.856 E.00813
G1 X131.601 Y103.561 E.01527
; LINE_WIDTH: 0.667911
G1 F12429.469
G1 X131.588 Y103.524 E.00163
G1 X132.088 Y103.166 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.022 Y102.753 E.01599
G1 X132.596 Y99.658 E.12017
G3 X131.451 Y99.495 I.24 J-5.781 E.04423
G3 X124.399 Y114.463 I-52.073 J-15.386 E.63418
G3 X122.174 Y117.472 I-28.757 J-18.937 E.14295
G1 X119.605 Y120.537 E.15272
G3 X112.93 Y126.698 I-39.811 J-36.439 E.3472
G1 X117.325 Y129.958 E.20893
G1 X117.73 Y130.298 E.02021
G1 X118.233 Y130.831 E.02796
G1 X118.77 Y131.575 E.03501
G1 X119.162 Y132.351 E.03323
G1 X119.368 Y132.942 E.02387
G1 X119.536 Y133.717 E.03029
G1 X119.596 Y134.56 E.03228
G1 X119.531 Y135.403 E.03226
G1 X119.363 Y136.149 E.02918
G1 X119.08 Y136.91 E.03101
G3 X130.579 Y143.624 I-20.655 J48.58 E.50972
G3 X142.647 Y156.415 I-31.78 J42.073 E.67457
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y104.466 E1.97651
G1 X143.615 Y104.066 E.01527
; LINE_WIDTH: 0.666016
G1 F12310.41
G1 X143.592 Y103.83 E.00975
; LINE_WIDTH: 0.712036
G1 F11509.612
G1 X143.569 Y103.594 E.01046
; LINE_WIDTH: 0.758056
G1 F10735.744
G1 X143.546 Y103.359 E.01118
; LINE_WIDTH: 0.806021
G1 F9988.778
G1 X143.522 Y103.237 E.00625
; LINE_WIDTH: 0.853986
G1 F9607.76
G1 X143.498 Y103.115 E.00664
G1 X143.5 Y102.652 E.02476
; LINE_WIDTH: 0.849146
G1 F9664.831
G1 X143.513 Y102.395 E.01366
; LINE_WIDTH: 0.823026
G1 F9984.912
G1 X143.533 Y101.995 E.02057
; LINE_WIDTH: 0.78242
G1 F10526.894
G1 X143.554 Y101.596 E.01951
; LINE_WIDTH: 0.741814
G1 F11131.09
G1 X143.574 Y101.196 E.01845
; LINE_WIDTH: 0.701208
G1 F11808.863
G1 X143.594 Y100.797 E.01739
; LINE_WIDTH: 0.660602
G1 F12574.53
G1 X143.615 Y100.397 E.01633
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X143.615 Y99.997 E.01527
G1 X143.615 Y99.545 E.01725
G1 X142.968 Y99.652 E.02502
G1 X141.997 Y99.665 E.03709
G1 X142.157 Y100.531 E.03362
; LINE_WIDTH: 0.646106
G1 F12872.486
G1 X142.217 Y100.781 E.01026
; LINE_WIDTH: 0.686714
G1 F12071.224
G1 X142.309 Y101.17 E.01701
; LINE_WIDTH: 0.727322
G1 F11363.867
G1 X142.402 Y101.559 E.01807
; LINE_WIDTH: 0.76793
G1 F10734.822
G1 X142.495 Y101.949 E.01913
; LINE_WIDTH: 0.808538
G1 F10171.765
G1 X142.587 Y102.338 E.02019
; LINE_WIDTH: 0.849146
G1 F9664.831
G1 X142.68 Y102.727 E.02125
; LINE_WIDTH: 0.853986
G1 F9607.76
G1 X142.684 Y103.044 E.01696
G1 X142.64 Y103.115 E.00448
; LINE_WIDTH: 0.806021
G1 F10204.942
G1 X142.596 Y103.187 E.00422
; LINE_WIDTH: 0.758056
G1 F10629.982
G1 X142.502 Y103.283 E.00634
; LINE_WIDTH: 0.712036
G1 F11063.695
G1 X142.408 Y103.379 E.00594
; LINE_WIDTH: 0.666016
G1 F11506.079
G1 X142.314 Y103.476 E.00553
; LINE_WIDTH: 0.619996
G1 F12569.803
G1 X142.042 Y103.63 E.01195
G1 F13428.341
G1 X141.802 Y103.668 E.00928
G1 F13446.369
G1 X141.402 Y103.668 E.01527
G1 X132.789 Y103.668 E.32885
G3 X132.347 Y103.531 I0 J-.778 E.01791
G1 X132.103 Y103.258 E.01398
G1 X132.103 Y103.255 E.00011
M204 S250
G1 X132.59 Y102.996 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X132.567 Y102.848 E.00474
G1 X133.258 Y99.115 E.12021
G3 X131.526 Y98.946 I-.184 J-7.082 E.05526
G1 X131.038 Y98.936 E.01545
G3 X123.945 Y114.147 I-51.151 J-14.591 E.53359
G3 X121.747 Y117.121 I-28.178 J-18.531 E.11714
G1 X119.177 Y120.186 E.12664
G3 X112.01 Y126.704 I-38.875 J-35.552 E.30715
G1 X116.99 Y130.398 E.19631
G3 X117.827 Y131.207 I-2.912 J3.852 E.03696
G1 X118.314 Y131.887 E.02646
G1 X118.665 Y132.594 E.02499
G1 X118.845 Y133.119 E.01756
G1 X118.995 Y133.83 E.02301
G1 X119.044 Y134.594 E.02424
G1 X118.98 Y135.356 E.02419
G1 X118.825 Y136.021 E.02164
G1 X118.548 Y136.753 E.0248
G1 X118.31 Y137.19 E.01573
G3 X130.249 Y144.067 I-19.806 J48.182 E.4375
G3 X142.39 Y157.025 I-31.645 J41.817 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.206 E1.8201
G1 X144.167 Y98.938 E.00848
G1 X143.859 Y98.945 E.00976
G1 X142.842 Y99.11 E.03264
G1 X141.332 Y99.114 E.04778
G1 X141.386 Y99.401 E.00923
G1 X141.439 Y99.687 E.00923
G1 X141.492 Y99.974 E.00923
G1 X141.545 Y100.26 E.00923
G1 X141.598 Y100.547 E.00923
G1 F3450
G1 X141.651 Y100.834 E.00923
G1 F3300
G1 X141.67 Y100.94 E.00342
G1 F3450
G1 X141.729 Y101.258 E.01025
G1 F3600
G1 X141.788 Y101.577 E.01025
G1 X141.847 Y101.895 E.01025
G1 X141.906 Y102.214 E.01025
G1 X141.965 Y102.532 E.01025
G1 X142.024 Y102.85 E.01025
G1 X142.013 Y102.969 E.00376
G1 X141.872 Y103.104 E.00621
G1 X138.493 Y103.115 E.10697
G1 X138.197 Y103.115 E.00937
G1 X137.901 Y103.115 E.00937
G1 X137.605 Y103.115 E.00937
G1 X137.309 Y103.115 E.00937
G1 F3450
G1 X137.013 Y103.115 E.00937
G1 F3300
G1 X136.564 Y103.115 E.01423
G1 F3450
G1 X136.268 Y103.115 E.00937
G1 F3600
G1 X135.972 Y103.115 E.00937
G1 X135.676 Y103.115 E.00937
G1 X135.381 Y103.115 E.00937
G1 X135.085 Y103.115 E.00937
G1 X132.789 Y103.115 E.07269
G3 X132.648 Y103.066 I0 J-.226 E.00481
; WIPE_START
M204 S10000
G1 X132.567 Y102.848 E-.08823
G1 X132.707 Y102.093 E-.29177
; WIPE_END
G1 E-.02 F1800
G1 X140.225 Y100.776 Z10.36 F36000
G1 X142.935 Y100.301 Z10.36
G1 Z9.96
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.918496
G1 F8906.76
G1 X142.824 Y100.301 E.00637
G1 X142.769 Y100.397 E.00637
G1 X142.824 Y100.493 E.00637
G1 X142.935 Y100.493 E.00637
G1 X142.99 Y100.397 E.00637
; WIPE_START
G1 X142.935 Y100.493 E-.076
G1 X142.824 Y100.493 E-.076
G1 X142.769 Y100.397 E-.076
G1 X142.824 Y100.301 E-.076
G1 X142.935 Y100.301 E-.076
; WIPE_END
G1 E-.02 F1800
G1 X135.448 Y101.786 Z10.36 F36000
G1 X131.257 Y102.616 Z10.36
G1 Z9.96
G1 E.4 F1800
; LINE_WIDTH: 1.00332
G1 F8127.103
G1 X131.313 Y102.412 E.01335
; LINE_WIDTH: 0.968376
G1 F8431.117
G1 X131.397 Y102.089 E.02038
; LINE_WIDTH: 0.919843
G1 F8893.214
G1 X131.482 Y101.765 E.01932
; LINE_WIDTH: 0.87131
G1 F9408.903
G1 X131.567 Y101.441 E.01826
; LINE_WIDTH: 0.822776
G1 F9988.079
G1 X131.644 Y101.13 E.01647
; LINE_WIDTH: 0.783076
G1 F10517.671
G1 X131.722 Y100.819 E.01564
; LINE_WIDTH: 0.743376
G1 F11106.567
G1 X131.8 Y100.509 E.01481
; LINE_WIDTH: 0.703676
G1 F11765.323
G1 X131.877 Y100.198 E.01398
; WIPE_START
G1 X131.8 Y100.509 E-.12174
G1 X131.722 Y100.819 E-.12173
G1 X131.644 Y101.13 E-.12174
G1 X131.635 Y101.168 E-.0148
; WIPE_END
G1 E-.02 F1800
G1 X131.647 Y105.034 Z10.36 F36000
G1 Z9.96
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X131.358 Y104.866 I.084 J-.476 E.01302
G3 X130.579 Y106.715 I-24.451 J-9.215 E.0766
G3 X131.772 Y107.504 I-.628 J2.247 E.05552
G1 X132.715 Y108.819 E.06177
G2 X137.428 Y111.652 I5.104 J-3.153 E.2181
G2 X139.313 Y110.781 I.116 J-2.223 E.08249
G1 X140.256 Y109.466 E.06177
G3 X141.945 Y107.685 I7.651 J5.566 E.094
G1 X141.945 Y114.273 E.25151
G2 X139.313 Y115.044 I-.749 J2.32 E.11123
G1 X138.371 Y116.359 E.06177
G3 X133.658 Y119.193 I-5.104 J-3.153 E.2181
G3 X131.772 Y118.322 I-.116 J-2.223 E.08249
G1 X130.83 Y117.007 E.06177
G2 X126.544 Y114.23 I-5.056 J3.107 E.20154
G3 X123.83 Y118.095 I-80.956 J-53.977 E.18034
G1 X120.881 Y121.613 E.17528
G3 X118.069 Y124.481 I-88.86 J-84.307 E.15334
G2 X122.347 Y126.734 I4.61 J-3.566 E.19033
G2 X124.232 Y125.863 I.116 J-2.224 E.08249
G1 X125.174 Y124.548 E.06177
G3 X129.887 Y121.714 I5.104 J3.153 E.2181
G3 X131.772 Y122.585 I.116 J2.224 E.08249
G1 X132.715 Y123.9 E.06177
G2 X137.428 Y126.734 I5.104 J-3.153 E.2181
G2 X139.313 Y125.863 I.116 J-2.224 E.08249
G1 X140.256 Y124.548 E.06177
G3 X141.945 Y122.767 I7.65 J5.566 E.094
G1 X141.945 Y129.354 E.25151
G2 X139.313 Y130.126 I-.749 J2.32 E.11123
G1 X138.371 Y131.441 E.06177
G3 X133.658 Y134.274 I-5.104 J-3.153 E.2181
G3 X131.772 Y133.403 I-.116 J-2.223 E.08249
G1 X130.83 Y132.088 E.06177
G2 X126.117 Y129.255 I-5.104 J3.153 E.2181
G2 X124.232 Y130.126 I-.116 J2.223 E.08249
G1 X123.289 Y131.441 E.06177
G3 X121.187 Y133.475 I-5.265 J-3.338 E.11274
G3 X121.113 Y135.985 I-7.282 J1.041 E.09637
G3 X126.045 Y138.553 I-38.609 J80.201 E.21233
G3 X129.887 Y136.795 I4.145 J3.983 E.16518
G3 X131.772 Y137.666 I.116 J2.223 E.08249
G1 X132.715 Y138.981 E.06177
G2 X137.428 Y141.815 I5.104 J-3.153 E.2181
G2 X139.313 Y140.944 I.116 J-2.224 E.08249
G1 X140.256 Y139.629 E.06177
G3 X141.945 Y137.848 I7.651 J5.566 E.094
G1 X141.945 Y144.435 E.25151
G2 X139.313 Y145.207 I-.749 J2.32 E.11123
G3 X137.557 Y147.534 I-18.552 J-12.173 E.1114
G3 X139.167 Y149.236 I-57.297 J55.828 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 10.12
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X138.48 Y148.51 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L63
M991 S0 P62 ;notify layer change


G17
G3 Z10.36 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.185 Y103.955
G1 Z10.12
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.131 I-51.36 J-19.656 E.4822
G3 X123.072 Y118.224 I-29.876 J-19.718 E.14699
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-42.601 J-39.174 E.29801
G1 X118.015 Y129.012 E.15055
G3 X118.933 Y129.85 I-4.61 J5.972 E.0475
G3 X120.076 Y131.532 I-5.876 J5.223 E.07788
G1 X120.45 Y132.48 E.03892
G1 X120.681 Y133.47 E.03883
G1 X120.765 Y134.489 E.03902
G1 X120.698 Y135.507 E.03894
G1 X120.538 Y136.268 E.0297
G3 X130.504 Y142.099 I-22.588 J50.037 E.44164
G3 X142.443 Y154.069 I-32.716 J44.57 E.64801
G1 X142.443 Y104.725 E1.8839
G1 X141.805 Y104.841 E.02479
G1 X132.786 Y104.841 E.34431
G3 X132.115 Y104.722 I0 J-1.949 E.02616
G1 X131.68 Y104.497 E.01871
G1 X131.245 Y104.022 E.0246
G1 X131.557 Y103.462 F36000
; LINE_WIDTH: 0.664019
G1 F10977.948
G1 X131.552 Y103.449 E.00059
; LINE_WIDTH: 0.708042
G1 F10931.596
G1 X131.505 Y103.335 E.00538
; LINE_WIDTH: 0.752065
G1 F10538.264
G1 X131.459 Y103.222 E.00573
; LINE_WIDTH: 0.796088
G1 F10152.139
G1 X131.412 Y103.108 E.00608
; LINE_WIDTH: 0.840111
G1 F9773.209
G1 X131.366 Y102.995 E.00644
; LINE_WIDTH: 0.884133
G1 F9266.919
G1 X131.319 Y102.882 E.00679
; LINE_WIDTH: 0.928156
G1 F8810.499
G1 X131.273 Y102.768 E.00714
; LINE_WIDTH: 0.966966
G1 F8443.863
G1 X131.261 Y102.707 E.00378
; LINE_WIDTH: 1.00578
G1 F8106.523
G1 X131.249 Y102.647 E.00393
G1 X131.179 Y102.766 E.00876
; LINE_WIDTH: 0.957554
G1 F8529.951
G1 X131.11 Y102.886 E.00833
; LINE_WIDTH: 0.909331
G1 F9000.052
G1 X131.04 Y103.005 E.00789
; LINE_WIDTH: 0.861109
G1 F9524.991
G1 X130.971 Y103.125 E.00746
; LINE_WIDTH: 0.812886
G1 F10114.958
G1 X130.901 Y103.245 E.00702
; LINE_WIDTH: 0.764664
G1 F10550.72
G1 X130.832 Y103.364 E.00659
; LINE_WIDTH: 0.716441
G1 F10995.687
G1 X130.762 Y103.484 E.00615
; LINE_WIDTH: 0.668219
G1 F11449.828
G1 X130.693 Y103.604 E.00572
; LINE_WIDTH: 0.619996
G1 F12814.386
G1 X130.548 Y103.976 E.01527
G1 F13446.369
G1 X130.331 Y104.531 E.02272
G3 X124.88 Y114.797 I-50.556 J-20.262 E.44461
G3 X122.623 Y117.848 I-29.324 J-19.333 E.14498
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-40.135 J-36.681 E.32257
G1 X117.68 Y129.492 E.18001
G3 X118.524 Y130.269 I-5.682 J7.025 E.04382
G3 X119.21 Y131.171 I-7.8 J6.64 E.0433
G1 X119.563 Y131.814 E.028
G1 X119.901 Y132.684 E.03563
G1 X120.108 Y133.592 E.03555
G1 X120.18 Y134.525 E.03573
G1 X120.114 Y135.456 E.03565
G1 X119.931 Y136.288 E.03254
G1 X119.835 Y136.599 E.01241
G3 X130.932 Y143.156 I-21.125 J48.414 E.49332
G3 X142.92 Y155.769 I-32.372 J42.775 E.66731
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.447 E1.99683
G1 X142.702 Y103.919 E.02191
G1 X142.225 Y104.189 E.02094
G1 X141.805 Y104.256 E.01626
G1 X132.786 Y104.256 E.34431
G1 X132.317 Y104.172 E.01821
G1 X132.012 Y104.015 E.01309
G1 X131.868 Y103.857 E.00815
G1 X131.598 Y103.562 E.01527
; LINE_WIDTH: 0.664019
G1 F12506.297
G1 X131.591 Y103.545 E.00075
G1 X132.087 Y103.17 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.021 Y102.75 E.01625
G1 X132.593 Y99.66 E.11998
G3 X131.45 Y99.495 I.242 J-5.73 E.04415
G3 X124.4 Y114.462 I-52.076 J-15.388 E.63414
G3 X122.174 Y117.472 I-28.765 J-18.942 E.14298
G1 X119.605 Y120.537 E.15272
G3 X112.93 Y126.698 I-39.807 J-36.434 E.3472
G1 X117.325 Y129.958 E.20893
G3 X118.151 Y130.726 I-6.614 J7.948 E.04308
G3 X118.696 Y131.453 I-12.332 J9.812 E.03471
G1 X119.05 Y132.096 E.028
G1 X119.352 Y132.887 E.03235
G1 X119.535 Y133.713 E.03227
G1 X119.595 Y134.56 E.03245
G1 X119.531 Y135.405 E.03236
G1 X119.362 Y136.153 E.02926
G1 X119.08 Y136.91 E.03085
G3 X130.578 Y143.623 I-20.886 J48.977 E.50967
G3 X142.648 Y156.415 I-31.962 J42.246 E.67457
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y104.467 E1.97647
G1 X143.615 Y104.067 E.01527
; LINE_WIDTH: 0.6658
G1 F12324.173
G1 X143.592 Y103.831 E.00974
; LINE_WIDTH: 0.711603
G1 F11523.717
G1 X143.569 Y103.596 E.01045
; LINE_WIDTH: 0.757406
G1 F10750.136
G1 X143.546 Y103.36 E.01115
; LINE_WIDTH: 0.805066
G1 F10003.4
G1 X143.522 Y103.239 E.00623
; LINE_WIDTH: 0.852726
G1 F9622.552
G1 X143.498 Y103.117 E.00662
G1 X143.501 Y102.654 E.0247
; LINE_WIDTH: 0.847916
G1 F9679.442
G1 X143.514 Y102.396 E.01372
; LINE_WIDTH: 0.821806
G1 F10000.382
G1 X143.534 Y101.996 E.02053
; LINE_WIDTH: 0.781444
G1 F10540.646
G1 X143.554 Y101.597 E.01948
; LINE_WIDTH: 0.741082
G1 F11142.618
G1 X143.574 Y101.197 E.01843
; LINE_WIDTH: 0.70072
G1 F11817.512
G1 X143.594 Y100.798 E.01738
; LINE_WIDTH: 0.660358
G1 F12579.431
G1 X143.615 Y100.398 E.01632
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X143.615 Y99.998 E.01527
G1 X143.615 Y99.546 E.01726
G1 X142.968 Y99.654 E.02503
G1 X141.999 Y99.667 E.03699
G1 X142.159 Y100.532 E.03356
; LINE_WIDTH: 0.646096
G1 F12872.697
G1 X142.219 Y100.783 E.01032
; LINE_WIDTH: 0.68646
G1 F12075.925
G1 X142.312 Y101.172 E.017
; LINE_WIDTH: 0.726824
G1 F11372.039
G1 X142.404 Y101.562 E.01806
; LINE_WIDTH: 0.767188
G1 F10745.689
G1 X142.497 Y101.951 E.01911
; LINE_WIDTH: 0.807552
G1 F10184.736
G1 X142.589 Y102.34 E.02016
; LINE_WIDTH: 0.847916
G1 F9679.442
G1 X142.682 Y102.729 E.02121
; LINE_WIDTH: 0.852726
G1 F9622.552
G1 X142.686 Y103.046 E.01692
G1 X142.642 Y103.117 E.00447
; LINE_WIDTH: 0.805066
G1 F10217.587
G1 X142.598 Y103.189 E.00421
; LINE_WIDTH: 0.757406
G1 F10642.757
G1 X142.504 Y103.285 E.00633
; LINE_WIDTH: 0.711603
G1 F11076.593
G1 X142.411 Y103.381 E.00593
; LINE_WIDTH: 0.6658
G1 F11519.073
G1 X142.317 Y103.478 E.00553
; LINE_WIDTH: 0.619996
G1 F12583.336
G1 X142.045 Y103.632 E.01195
G1 F13442.334
G1 X141.805 Y103.67 E.00928
G1 F13446.369
G1 X141.405 Y103.67 E.01527
G1 X132.786 Y103.67 E.32904
G3 X132.345 Y103.533 I0 J-.778 E.01793
G1 X132.101 Y103.262 E.0139
G1 X132.101 Y103.259 E.0001
M204 S250
G1 X132.588 Y102.999 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X132.564 Y102.85 E.00476
G1 X133.256 Y99.117 E.12021
G3 X131.524 Y98.946 I-.181 J-7.016 E.05522
G1 X131.038 Y98.936 E.01541
G3 X123.946 Y114.147 I-51.15 J-14.591 E.53357
G3 X121.747 Y117.121 I-28.178 J-18.53 E.11716
G1 X119.177 Y120.186 E.12664
G3 X112.01 Y126.704 I-38.874 J-35.552 E.30714
G1 X116.99 Y130.398 E.19632
G3 X117.739 Y131.094 I-6.448 J7.687 E.03239
G3 X118.212 Y131.72 I-29.058 J22.474 E.02482
G1 X118.565 Y132.362 E.02322
G1 X118.834 Y133.08 E.02426
M73 P86 R2
G1 X118.994 Y133.827 E.02419
G1 X119.044 Y134.594 E.02434
G1 X118.98 Y135.358 E.02426
G1 X118.824 Y136.025 E.0217
G1 X118.548 Y136.752 E.02463
G1 X118.31 Y137.19 E.01576
G3 X130.248 Y144.067 I-19.938 J48.412 E.43747
G3 X142.39 Y157.025 I-31.645 J41.818 E.56494
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.206 E1.82011
G1 X144.167 Y98.938 E.00847
G1 X143.859 Y98.945 E.00976
G1 X142.84 Y99.112 E.03269
G1 X141.335 Y99.116 E.04766
G1 X142.027 Y102.852 E.1203
G1 X142.016 Y102.971 E.00376
G1 X141.874 Y103.106 E.00621
G1 X132.786 Y103.117 E.28772
G3 X132.646 Y103.068 I0 J-.226 E.0048
; WIPE_START
M204 S10000
G1 X132.564 Y102.85 E-.08837
G1 X132.704 Y102.096 E-.29163
; WIPE_END
G1 E-.02 F1800
G1 X140.222 Y100.778 Z10.52 F36000
G1 X142.936 Y100.303 Z10.52
G1 Z10.12
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.916236
G1 F8929.585
G1 X142.826 Y100.303 E.00634
G1 X142.771 Y100.398 E.00634
G1 X142.826 Y100.494 E.00634
G1 X142.936 Y100.494 E.00634
G1 X142.991 Y100.398 E.00634
; WIPE_START
G1 X142.936 Y100.494 E-.076
G1 X142.826 Y100.494 E-.076
G1 X142.771 Y100.398 E-.076
G1 X142.826 Y100.303 E-.076
G1 X142.936 Y100.303 E-.07599
; WIPE_END
G1 E-.02 F1800
G1 X135.452 Y101.803 Z10.52 F36000
G1 X131.249 Y102.647 Z10.52
G1 Z10.12
G1 E.4 F1800
; LINE_WIDTH: 1.00578
G1 F8106.523
G1 X131.258 Y102.609 E.00248
; LINE_WIDTH: 0.999436
G1 F8159.776
G1 X131.333 Y102.325 E.01842
; LINE_WIDTH: 0.955976
G1 F8544.552
G1 X131.407 Y102.042 E.0176
; LINE_WIDTH: 0.912516
G1 F8967.411
G1 X131.482 Y101.759 E.01676
; LINE_WIDTH: 0.869056
G1 F9434.302
G1 X131.557 Y101.476 E.01594
; LINE_WIDTH: 0.825596
G1 F9952.482
G1 X131.566 Y101.441 E.00186
; LINE_WIDTH: 0.820696
G1 F10014.499
G1 X131.643 Y101.13 E.01642
; LINE_WIDTH: 0.781006
G1 F10546.829
G1 X131.721 Y100.819 E.01559
; LINE_WIDTH: 0.741316
G1 F11138.93
G1 X131.799 Y100.509 E.01476
; LINE_WIDTH: 0.701626
G1 F11801.467
G1 X131.876 Y100.198 E.01393
; WIPE_START
G1 X131.799 Y100.509 E-.12169
G1 X131.721 Y100.819 E-.12169
G1 X131.643 Y101.13 E-.12169
G1 X131.634 Y101.168 E-.01492
; WIPE_END
G1 E-.02 F1800
G1 X131.47 Y104.946 Z10.52 F36000
G1 Z10.12
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X131.356 Y104.87 I.047 J-.192 E.00531
G3 X130.503 Y106.902 I-38.567 J-14.994 E.08417
G3 X131.301 Y107.311 I-.46 J1.883 E.03453
G3 X132.715 Y108.984 I-5.283 J5.897 E.08392
G2 X136.485 Y111.431 I5.059 J-3.667 E.17554
G2 X138.842 Y110.974 I.668 J-2.86 E.0944
G2 X140.256 Y109.301 I-5.282 J-5.897 E.08392
G3 X141.945 Y107.688 I5.42 J3.986 E.08965
G1 X141.945 Y114.381 E.25556
G2 X139.784 Y114.851 I-.556 J2.649 E.08698
G2 X138.371 Y116.525 I5.283 J5.897 E.08392
G3 X134.6 Y118.971 I-5.059 J-3.667 E.17554
G3 X132.244 Y118.515 I-.668 J-2.86 E.0944
G3 X130.83 Y116.842 I5.283 J-5.897 E.08392
G2 X126.47 Y114.355 I-4.78 J3.315 E.19815
G3 X123.83 Y118.095 I-92.163 J-62.242 E.1748
G1 X120.881 Y121.613 E.17528
G3 X118.015 Y124.53 I-80.693 J-76.404 E.15611
G2 X121.404 Y126.512 I4.635 J-4.037 E.15256
G2 X123.761 Y126.056 I.668 J-2.86 E.0944
G2 X125.174 Y124.382 I-5.282 J-5.897 E.08392
G3 X128.945 Y121.935 I5.059 J3.667 E.17554
G3 X131.301 Y122.392 I.668 J2.86 E.0944
G3 X132.715 Y124.065 I-5.282 J5.897 E.08392
G2 X136.485 Y126.512 I5.059 J-3.667 E.17554
G2 X138.842 Y126.056 I.668 J-2.86 E.0944
G2 X140.256 Y124.382 I-5.282 J-5.896 E.08392
G3 X141.945 Y122.769 I5.42 J3.986 E.08965
G1 X141.945 Y129.463 E.25556
G2 X139.784 Y129.932 I-.556 J2.648 E.08698
G2 X138.371 Y131.606 I5.283 J5.897 E.08392
G3 X134.6 Y134.053 I-5.059 J-3.667 E.17554
G3 X132.244 Y133.596 I-.668 J-2.86 E.0944
G3 X130.83 Y131.923 I5.283 J-5.897 E.08392
G2 X127.06 Y129.476 I-5.059 J3.667 E.17554
G2 X124.703 Y129.932 I-.668 J2.86 E.0944
G2 X123.289 Y131.606 I5.283 J5.897 E.08392
G3 X121.404 Y133.365 I-5.741 J-4.264 E.09897
G1 X121.186 Y133.466 E.00919
G3 X121.111 Y135.983 I-7.217 J1.044 E.09663
G3 X126.001 Y138.526 I-40.898 J84.64 E.21046
G3 X128.945 Y137.017 I3.885 J3.952 E.12826
G3 X131.301 Y137.473 I.668 J2.86 E.0944
G3 X132.715 Y139.147 I-5.283 J5.897 E.08392
G2 X136.485 Y141.593 I5.059 J-3.667 E.17554
G2 X138.842 Y141.137 I.668 J-2.86 E.0944
G2 X140.256 Y139.463 I-5.282 J-5.897 E.08392
G3 X141.945 Y137.85 I5.42 J3.986 E.08965
G1 X141.945 Y144.544 E.25556
G2 X139.313 Y145.473 I-.565 J2.594 E.11221
G3 X137.594 Y147.574 I-16.166 J-11.472 E.10373
G2 X135.913 Y145.943 I-25.167 J24.256 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 10.28
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X136.631 Y146.639 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L64
M991 S0 P63 ;notify layer change


G17
G3 Z10.52 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.179 Y103.969
G1 Z10.28
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.132 I-51.313 J-19.646 E.48164
G3 X123.072 Y118.224 I-29.87 J-19.714 E.14698
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-40.856 J-37.338 E.29803
G1 X118.015 Y129.012 E.15055
G1 X118.485 Y129.402 E.02333
G1 X119.091 Y130.035 E.03345
G3 X120.112 Y131.605 I-6.062 J5.058 E.07168
G1 X120.452 Y132.483 E.03595
G1 X120.682 Y133.473 E.03879
G1 X120.765 Y134.489 E.03893
G1 X120.698 Y135.506 E.03891
G1 X120.538 Y136.268 E.02972
G3 X131.286 Y142.69 I-21.772 J48.644 E.47909
G3 X142.443 Y154.069 I-32.589 J43.114 E.61068
G1 X142.443 Y104.729 E1.88377
G1 X141.807 Y104.843 E.02468
G1 X132.784 Y104.843 E.3445
G1 X132.468 Y104.818 E.01209
G1 X131.924 Y104.643 E.02182
G1 X131.471 Y104.335 E.0209
G1 X131.235 Y104.039 E.01446
G1 X131.52 Y103.367 F36000
; LINE_WIDTH: 0.712876
G1 F10581.989
G1 X131.499 Y103.308 E.00278
; LINE_WIDTH: 0.759316
G1 F10382.683
G1 X131.458 Y103.193 E.00575
; LINE_WIDTH: 0.805756
G1 F10002.198
G1 X131.417 Y103.079 E.00612
; LINE_WIDTH: 0.852196
G1 F9628.787
G1 X131.377 Y102.964 E.00649
; LINE_WIDTH: 0.898636
G1 F9111.42
G1 X131.336 Y102.85 E.00685
; LINE_WIDTH: 0.945076
G1 F8646.816
G1 X131.295 Y102.735 E.00722
; LINE_WIDTH: 0.991516
G1 F8227.293
G1 X131.254 Y102.62 E.00759
G1 X131.184 Y102.743 E.0088
; LINE_WIDTH: 0.945076
G1 F8646.816
G1 X131.114 Y102.865 E.00837
; LINE_WIDTH: 0.898636
G1 F9111.42
G1 X131.044 Y102.988 E.00794
; LINE_WIDTH: 0.852196
G1 F9628.787
G1 X130.974 Y103.11 E.00752
; LINE_WIDTH: 0.805756
G1 F10208.447
G1 X130.904 Y103.232 E.00709
; LINE_WIDTH: 0.759316
G1 F10654.582
G1 X130.834 Y103.355 E.00666
; LINE_WIDTH: 0.712876
G1 F11110.274
G1 X130.764 Y103.477 E.00624
; LINE_WIDTH: 0.666436
G1 F11575.462
G1 X130.694 Y103.6 E.00581
; LINE_WIDTH: 0.619996
G1 F12947.276
G1 X130.548 Y103.972 E.01527
G1 F13446.369
G1 X130.402 Y104.344 E.01527
G1 X130.208 Y104.835 E.02014
G3 X124.88 Y114.797 I-50.409 J-20.554 E.4321
G3 X122.623 Y117.848 I-29.319 J-19.329 E.14497
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-42.055 J-38.731 E.32254
G1 X117.666 Y129.482 E.17936
G1 X118.108 Y129.85 E.02196
G1 X118.662 Y130.433 E.03071
G1 X119.253 Y131.243 E.03829
G1 X119.591 Y131.872 E.02728
G1 X119.902 Y132.686 E.03325
G1 X120.109 Y133.594 E.03554
G1 X120.18 Y134.525 E.03565
G1 X120.114 Y135.456 E.03563
G1 X119.933 Y136.284 E.03237
G1 X119.835 Y136.599 E.01258
G3 X130.932 Y143.157 I-21.04 J48.27 E.49335
G3 X142.92 Y155.769 I-32.175 J42.587 E.66731
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.452 E1.99662
G1 X142.705 Y103.921 E.02174
G1 X142.228 Y104.191 E.02094
G1 X141.807 Y104.258 E.01626
G1 X132.784 Y104.258 E.3445
G1 X132.563 Y104.24 E.00846
G1 X132.156 Y104.105 E.01637
G1 X131.865 Y103.902 E.01352
G1 X131.581 Y103.537 E.01768
; LINE_WIDTH: 0.666436
G1 F12458.473
G1 X131.551 Y103.452 E.00373
G1 X132.079 Y103.144 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.017 Y102.757 E.01498
G1 X132.591 Y99.661 E.12019
G3 X131.45 Y99.495 I.243 J-5.668 E.04407
G3 X124.399 Y114.463 I-52.079 J-15.39 E.63414
G3 X122.174 Y117.472 I-28.762 J-18.94 E.14296
G1 X119.605 Y120.537 E.15272
G3 X112.93 Y126.698 I-39.743 J-36.365 E.3472
G1 X117.317 Y129.952 E.20856
G1 X117.73 Y130.298 E.02058
G1 X118.233 Y130.831 E.02796
G1 X118.77 Y131.575 E.03501
G1 X119.07 Y132.14 E.02445
G1 X119.353 Y132.889 E.03055
G1 X119.536 Y133.715 E.0323
G1 X119.596 Y134.56 E.03237
G1 X119.531 Y135.405 E.03235
G1 X119.363 Y136.148 E.02909
G1 X119.08 Y136.91 E.03102
G3 X130.579 Y143.624 I-20.71 J48.675 E.50971
G3 X142.648 Y156.415 I-31.78 J42.073 E.67457
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y104.468 E1.97644
G1 X143.615 Y104.068 E.01527
; LINE_WIDTH: 0.66558
G1 F12338.154
G1 X143.592 Y103.833 E.00973
; LINE_WIDTH: 0.711163
G1 F11538.035
G1 X143.569 Y103.597 E.01043
; LINE_WIDTH: 0.756746
G1 F10764.703
G1 X143.546 Y103.362 E.01113
; LINE_WIDTH: 0.804101
G1 F10018.195
G1 X143.523 Y103.24 E.00622
; LINE_WIDTH: 0.851456
G1 F9637.509
G1 X143.499 Y103.119 E.0066
G1 X143.501 Y102.656 E.02464
; LINE_WIDTH: 0.846676
G1 F9694.218
G1 X143.514 Y102.396 E.01377
; LINE_WIDTH: 0.820586
G1 F10015.898
G1 X143.534 Y101.997 E.0205
; LINE_WIDTH: 0.780468
G1 F10554.433
G1 X143.554 Y101.597 E.01946
; LINE_WIDTH: 0.74035
G1 F11154.171
G1 X143.574 Y101.198 E.01841
; LINE_WIDTH: 0.700232
G1 F11826.172
G1 X143.594 Y100.798 E.01736
; LINE_WIDTH: 0.660114
G1 F12584.335
G1 X143.615 Y100.399 E.01632
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X143.615 Y99.999 E.01527
G1 X143.615 Y99.547 E.01726
G1 X142.968 Y99.656 E.02505
G1 X142.002 Y99.669 E.03689
G1 X142.162 Y100.532 E.03351
; LINE_WIDTH: 0.646076
G1 F12873.116
G1 X142.222 Y100.785 E.01037
; LINE_WIDTH: 0.686196
G1 F12080.816
G1 X142.314 Y101.175 E.017
; LINE_WIDTH: 0.726316
G1 F11380.387
G1 X142.407 Y101.564 E.01804
; LINE_WIDTH: 0.766436
G1 F10756.728
G1 X142.499 Y101.953 E.01909
; LINE_WIDTH: 0.806556
G1 F10197.872
G1 X142.592 Y102.342 E.02014
; LINE_WIDTH: 0.846676
G1 F9694.218
G1 X142.684 Y102.731 E.02118
; LINE_WIDTH: 0.851456
G1 F9637.509
G1 X142.688 Y103.048 E.01688
G1 X142.644 Y103.119 E.00446
; LINE_WIDTH: 0.804101
G1 F10230.396
G1 X142.6 Y103.191 E.0042
; LINE_WIDTH: 0.756746
G1 F10655.698
G1 X142.507 Y103.287 E.00633
; LINE_WIDTH: 0.711163
G1 F11089.662
G1 X142.413 Y103.383 E.00593
; LINE_WIDTH: 0.66558
G1 F11532.24
G1 X142.319 Y103.48 E.00553
; LINE_WIDTH: 0.619996
G1 F12597.146
G1 X142.047 Y103.634 E.01195
G1 F13446.369
G1 X141.807 Y103.672 E.00928
G1 X132.784 Y103.672 E.3445
G1 X132.425 Y103.585 E.01408
G3 X132.097 Y103.261 I.358 J-.691 E.01787
G1 X132.093 Y103.233 E.00106
M204 S250
G1 X132.585 Y103 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X132.586 Y102.719 E.00888
G1 X133.253 Y99.119 E.11592
G3 X131.524 Y98.946 I-.179 J-6.943 E.05517
G1 X131.038 Y98.936 E.01539
G3 X123.945 Y114.147 I-51.152 J-14.591 E.53359
G3 X121.747 Y117.121 I-28.177 J-18.53 E.11714
G1 X119.177 Y120.186 E.12664
G3 X112.01 Y126.704 I-40.295 J-37.114 E.30712
G1 X116.988 Y130.396 E.19622
G3 X117.828 Y131.207 I-2.909 J3.853 E.03705
G1 X118.314 Y131.887 E.02647
G1 X118.579 Y132.393 E.01806
G1 X118.834 Y133.08 E.02322
G1 X118.995 Y133.829 E.02425
G1 X119.044 Y134.594 E.02427
G1 X118.98 Y135.357 E.02425
G1 X118.825 Y136.021 E.02156
G1 X118.548 Y136.753 E.02478
G1 X118.31 Y137.19 E.01576
G3 X130.249 Y144.067 I-19.805 J48.182 E.4375
G3 X142.39 Y157.025 I-31.645 J41.817 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.205 E1.82013
G1 X144.167 Y98.938 E.00845
G1 X143.859 Y98.945 E.00976
G1 X142.839 Y99.114 E.03275
G1 X141.337 Y99.118 E.04753
G1 X142.029 Y102.855 E.1203
G1 X142.018 Y102.973 E.00376
G1 X141.877 Y103.108 E.00621
G1 X132.784 Y103.119 E.28788
G1 X132.644 Y103.065 E.00473
; WIPE_START
M204 S10000
G1 X132.586 Y102.719 E-.13324
G1 X132.705 Y102.081 E-.24676
; WIPE_END
G1 E-.02 F1800
G1 X140.225 Y100.775 Z10.68 F36000
G1 X142.937 Y100.304 Z10.68
G1 Z10.28
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.913976
G1 F8952.527
G1 X142.827 Y100.304 E.0063
G1 X142.772 Y100.399 E.00631
G1 X142.827 Y100.494 E.0063
G1 X142.937 Y100.494 E.0063
G1 X142.992 Y100.399 E.00631
; WIPE_START
G1 X142.937 Y100.494 E-.076
G1 X142.827 Y100.494 E-.076
G1 X142.772 Y100.399 E-.076
G1 X142.827 Y100.304 E-.076
G1 X142.937 Y100.304 E-.07599
; WIPE_END
G1 E-.02 F1800
G1 X135.449 Y101.783 Z10.68 F36000
G1 X131.257 Y102.611 Z10.68
G1 Z10.28
G1 E.4 F1800
; LINE_WIDTH: 0.997876
G1 F8172.988
G1 X131.301 Y102.448 E.01059
; LINE_WIDTH: 0.970076
G1 F8415.8
G1 X131.386 Y102.123 E.02052
; LINE_WIDTH: 0.92103
G1 F8881.312
G1 X131.471 Y101.797 E.01945
; LINE_WIDTH: 0.871983
G1 F9401.34
G1 X131.557 Y101.472 E.01837
; LINE_WIDTH: 0.822936
G1 F9986.052
G1 X131.565 Y101.441 E.00165
; LINE_WIDTH: 0.818596
G1 F10041.314
G1 X131.642 Y101.13 E.01637
; LINE_WIDTH: 0.778931
G1 F10576.22
G1 X131.72 Y100.819 E.01554
; LINE_WIDTH: 0.739266
G1 F11171.322
G1 X131.798 Y100.509 E.01471
; LINE_WIDTH: 0.699601
G1 F11837.39
G1 X131.875 Y100.198 E.01388
; WIPE_START
G1 X131.798 Y100.509 E-.12164
G1 X131.72 Y100.819 E-.12164
G1 X131.642 Y101.13 E-.12164
G1 X131.633 Y101.168 E-.01509
; WIPE_END
G1 E-.02 F1800
G1 X131.349 Y104.906 Z10.68 F36000
G1 Z10.28
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X130.433 Y107.062 I-50.016 J-19.982 E.08944
G1 X130.83 Y107.225 E.01638
G3 X132.715 Y109.137 I-4.038 J5.866 E.10315
G2 X136.485 Y111.343 I4.576 J-3.497 E.17105
G2 X139.313 Y110.285 I.501 J-2.969 E.12075
G3 X141.198 Y108.176 I14.441 J11.017 E.10808
G1 X141.945 Y107.667 E.03451
G1 X141.945 Y114.481 E.26016
G2 X140.256 Y114.765 I-.361 J3.021 E.06629
G2 X138.371 Y116.678 I4.038 J5.866 E.10315
G3 X134.6 Y118.883 I-4.576 J-3.497 E.17105
G3 X131.772 Y117.825 I-.501 J-2.969 E.12075
G2 X129.887 Y115.717 I-14.441 J11.017 E.10808
G2 X126.378 Y114.478 I-3.353 J3.906 E.1454
G3 X123.83 Y118.094 I-42.694 J-27.365 E.16894
G3 X117.975 Y124.57 I-49.451 J-38.828 E.33359
G2 X120.462 Y126.223 I4.431 J-3.969 E.11523
G2 X123.289 Y126.142 I1.305 J-3.811 E.11036
G2 X125.174 Y124.229 I-4.038 J-5.865 E.10315
G3 X128.002 Y122.225 I4.977 J4.024 E.13401
G3 X130.83 Y122.306 I1.305 J3.811 E.11036
G3 X132.715 Y124.218 I-4.038 J5.865 E.10315
G2 X136.485 Y126.424 I4.576 J-3.497 E.17105
G2 X139.313 Y125.366 I.501 J-2.969 E.12075
G3 X141.198 Y123.258 I14.441 J11.017 E.10808
G1 X141.944 Y122.749 E.03448
G1 X141.944 Y129.563 E.26014
G2 X140.256 Y129.846 I-.361 J3.019 E.06626
G2 X138.371 Y131.759 I4.038 J5.866 E.10315
G3 X135.543 Y133.764 I-4.977 J-4.024 E.13401
G3 X132.715 Y133.682 I-1.305 J-3.811 E.11036
G3 X130.83 Y131.77 I4.038 J-5.866 E.10315
G2 X128.002 Y129.765 I-4.977 J4.024 E.13401
G2 X125.174 Y129.846 I-1.305 J3.811 E.11036
G2 X123.289 Y131.759 I4.038 J5.865 E.10315
G3 X121.404 Y133.373 I-4.878 J-3.79 E.0954
G1 X121.185 Y133.464 E.00903
G3 X121.113 Y135.985 I-7.293 J1.051 E.0968
G3 X125.96 Y138.501 I-41.096 J85.105 E.20853
G3 X128.002 Y137.306 I4.144 J4.739 E.09089
G3 X130.83 Y137.387 I1.305 J3.811 E.11036
G3 X132.715 Y139.3 I-4.038 J5.866 E.10315
G2 X136.485 Y141.505 I4.576 J-3.497 E.17105
G2 X139.313 Y140.447 I.501 J-2.969 E.12075
G3 X141.198 Y138.339 I14.44 J11.016 E.10808
G1 X141.944 Y137.831 E.03444
G1 X141.943 Y144.644 E.26012
G2 X139.784 Y145.247 I-.393 J2.761 E.08807
G2 X137.627 Y147.607 I233.278 J215.433 E.12207
G2 X135.95 Y145.972 I-37.116 J36.406 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 10.44
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X136.666 Y146.67 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L65
M991 S0 P64 ;notify layer change


G17
G3 Z10.68 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.182 Y103.961
G1 Z10.44
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.131 I-51.347 J-19.657 E.48196
G3 X123.072 Y118.225 I-29.872 J-19.715 E.147
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-40.941 J-37.427 E.29802
G1 X118.015 Y129.012 E.15055
G3 X118.932 Y129.849 I-4.608 J5.969 E.04748
G3 X120.074 Y131.528 I-5.862 J5.215 E.07776
G1 X120.449 Y132.477 E.03893
G1 X120.681 Y133.47 E.03893
G3 X120.698 Y135.507 I-7.915 J1.083 E.07799
G1 X120.538 Y136.268 E.02969
M73 P87 R2
G3 X131.285 Y142.689 I-21.719 J48.552 E.47907
G3 X142.443 Y154.07 I-32.798 J43.319 E.61071
G1 X142.443 Y104.732 E1.88366
G1 X141.81 Y104.845 E.02458
G1 X132.781 Y104.845 E.34469
G3 X132.109 Y104.726 I0 J-1.95 E.02621
G1 X131.673 Y104.5 E.01873
G1 X131.243 Y104.028 E.02441
G1 X131.551 Y103.463 F36000
; LINE_WIDTH: 0.664122
G1 F10975.241
G1 X131.545 Y103.45 E.00061
; LINE_WIDTH: 0.708248
G1 F10927.085
G1 X131.499 Y103.336 E.00539
; LINE_WIDTH: 0.752373
G1 F10533.14
G1 X131.453 Y103.222 E.00575
; LINE_WIDTH: 0.796499
G1 F10146.427
G1 X131.407 Y103.109 E.0061
; LINE_WIDTH: 0.840625
G1 F9766.975
G1 X131.361 Y102.995 E.00645
; LINE_WIDTH: 0.884751
G1 F9260.193
G1 X131.314 Y102.881 E.0068
; LINE_WIDTH: 0.928876
G1 F8803.408
G1 X131.268 Y102.767 E.00716
; LINE_WIDTH: 0.965616
G1 F8456.104
G1 X131.257 Y102.709 E.0036
; LINE_WIDTH: 1.00236
G1 F8135.163
G1 X131.245 Y102.651 E.00374
G1 X131.176 Y102.77 E.00869
; LINE_WIDTH: 0.954561
G1 F8557.69
G1 X131.107 Y102.889 E.00826
; LINE_WIDTH: 0.906766
G1 F9026.513
G1 X131.038 Y103.008 E.00783
; LINE_WIDTH: 0.858971
G1 F9549.681
G1 X130.969 Y103.127 E.0074
; LINE_WIDTH: 0.811176
G1 F10137.224
G1 X130.9 Y103.246 E.00697
; LINE_WIDTH: 0.763381
G1 F10571.174
G1 X130.831 Y103.365 E.00654
; LINE_WIDTH: 0.715586
G1 F11014.206
G1 X130.762 Y103.484 E.00611
; LINE_WIDTH: 0.667791
G1 F11466.317
G1 X130.693 Y103.603 E.00568
; LINE_WIDTH: 0.619996
G1 F12831.829
G1 X130.548 Y103.976 E.01527
G1 F13446.369
G1 X130.331 Y104.531 E.02276
G3 X124.88 Y114.797 I-50.548 J-20.259 E.44459
G3 X122.623 Y117.848 I-29.314 J-19.325 E.14499
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-40.2 J-36.75 E.32256
G1 X117.666 Y129.482 E.17936
G3 X118.523 Y130.268 I-4.999 J6.312 E.04444
G3 X119.208 Y131.168 I-7.758 J6.61 E.04319
G1 X119.561 Y131.811 E.02801
G1 X119.9 Y132.681 E.03564
G1 X120.108 Y133.591 E.03564
G3 X120.114 Y135.456 I-7.942 J.958 E.07137
G1 X119.931 Y136.289 E.03257
G1 X119.836 Y136.599 E.01238
G3 X130.932 Y143.156 I-20.97 J48.152 E.49332
G3 X142.92 Y155.769 I-32.175 J42.588 E.66735
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.458 E1.9964
G1 X142.707 Y103.923 E.02157
G1 X142.23 Y104.193 E.02094
G1 X141.81 Y104.26 E.01626
G1 X132.781 Y104.26 E.34469
G1 X132.311 Y104.176 E.01824
G1 X132.006 Y104.018 E.0131
G1 X131.861 Y103.859 E.00822
G1 X131.592 Y103.563 E.01527
; LINE_WIDTH: 0.664122
G1 F12504.254
G1 X131.585 Y103.547 E.00073
G1 X132.081 Y103.173 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.016 Y102.754 E.0162
G1 X132.588 Y99.663 E.12001
G3 X131.45 Y99.496 I.244 J-5.605 E.04399
G3 X124.4 Y114.462 I-52.081 J-15.391 E.63411
G3 X122.174 Y117.472 I-28.76 J-18.939 E.14298
G1 X119.604 Y120.538 E.15272
G3 X112.93 Y126.698 I-40.852 J-37.567 E.34717
G1 X117.317 Y129.952 E.20856
G3 X118.114 Y130.687 I-6.173 J7.496 E.04141
G3 X118.695 Y131.451 I-9.662 J7.95 E.03663
G1 X119.048 Y132.093 E.02801
G1 X119.351 Y132.885 E.03236
G3 X119.531 Y135.406 I-6.108 J1.701 E.09713
G1 X119.361 Y136.154 E.02929
G1 X119.08 Y136.91 E.03081
G3 X130.578 Y143.623 I-20.411 J48.163 E.5097
G3 X142.648 Y156.415 I-31.779 J42.073 E.67461
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y104.469 E1.9764
G1 X143.615 Y104.069 E.01527
; LINE_WIDTH: 0.665363
G1 F12352.019
G1 X143.592 Y103.834 E.00971
; LINE_WIDTH: 0.71073
G1 F11552.237
G1 X143.569 Y103.599 E.01041
; LINE_WIDTH: 0.756096
G1 F10779.192
G1 X143.547 Y103.364 E.01111
; LINE_WIDTH: 0.803146
G1 F10032.915
G1 X143.523 Y103.242 E.0062
; LINE_WIDTH: 0.850196
G1 F9652.392
G1 X143.499 Y103.121 E.00658
G1 X143.502 Y102.658 E.02458
; LINE_WIDTH: 0.845446
G1 F9708.919
G1 X143.515 Y102.397 E.01383
; LINE_WIDTH: 0.819366
G1 F10031.464
G1 X143.54 Y101.898 E.02559
; LINE_WIDTH: 0.769524
G1 F10711.554
G1 X143.565 Y101.399 E.02396
; LINE_WIDTH: 0.719681
G1 F11490.564
G1 X143.59 Y100.899 E.02234
; LINE_WIDTH: 0.669839
G1 F12391.77
G1 X143.615 Y100.4 E.02071
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X143.615 Y100 E.01527
G1 X143.615 Y99.548 E.01727
G1 X142.968 Y99.658 E.02505
G1 X142.004 Y99.671 E.03679
G1 X142.164 Y100.533 E.03345
; LINE_WIDTH: 0.646056
G1 F12803.994
G1 X142.224 Y100.787 E.01043
; LINE_WIDTH: 0.695904
G1 F11903.548
G1 X142.34 Y101.274 E.02156
; LINE_WIDTH: 0.745751
G1 F11069.489
G1 X142.455 Y101.76 E.02319
; LINE_WIDTH: 0.795599
G1 F10344.659
G1 X142.57 Y102.247 E.02481
; LINE_WIDTH: 0.845446
G1 F9708.919
G1 X142.686 Y102.733 E.02644
; LINE_WIDTH: 0.850196
G1 F9652.392
G1 X142.69 Y103.05 E.01684
G1 X142.646 Y103.121 E.00445
; LINE_WIDTH: 0.803146
G1 F10243.104
G1 X142.602 Y103.193 E.00419
; LINE_WIDTH: 0.756096
G1 F10668.536
G1 X142.509 Y103.289 E.00632
; LINE_WIDTH: 0.71073
G1 F11102.602
G1 X142.415 Y103.385 E.00592
; LINE_WIDTH: 0.665363
G1 F11545.299
G1 X142.322 Y103.482 E.00552
; LINE_WIDTH: 0.619996
G1 F12610.747
G1 X142.05 Y103.636 E.01195
G1 F13446.369
G1 X141.81 Y103.674 E.00928
G1 X132.781 Y103.674 E.34469
G1 X132.513 Y103.626 E.01041
G1 X132.339 Y103.536 E.00748
G1 X132.095 Y103.264 E.01394
G1 X132.095 Y103.262 E.00008
M204 S250
G1 X132.582 Y103.002 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X132.559 Y102.855 E.00474
G1 X133.251 Y99.121 E.12021
G3 X131.523 Y98.946 I-.177 J-6.869 E.05512
G1 X131.038 Y98.936 E.01537
G3 X123.946 Y114.147 I-51.152 J-14.591 E.53357
G3 X121.747 Y117.121 I-28.181 J-18.533 E.11717
G1 X119.177 Y120.187 E.12664
G3 X112.01 Y126.704 I-40.297 J-37.117 E.3071
G1 X116.988 Y130.396 E.19623
G1 X117.728 Y131.083 E.03196
G3 X118.21 Y131.717 I-22.467 J17.585 E.02523
G1 X118.564 Y132.36 E.02323
G1 X118.833 Y133.077 E.02426
G3 X118.98 Y135.358 I-5.617 J1.506 E.07282
G1 X118.824 Y136.026 E.02172
G1 X118.548 Y136.753 E.02461
G1 X118.31 Y137.19 E.01575
G3 X130.248 Y144.067 I-19.584 J47.798 E.43749
G3 X142.39 Y157.025 I-31.645 J41.818 E.56495
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.205 E1.82014
G1 X144.167 Y98.938 E.00844
G1 X143.859 Y98.945 E.00976
G1 X142.838 Y99.116 E.0328
G1 X141.34 Y99.121 E.04741
G1 X142.032 Y102.857 E.1203
G1 X142.021 Y102.975 E.00376
G1 X141.879 Y103.11 E.00621
G1 X132.781 Y103.121 E.28804
G3 X132.64 Y103.072 I0 J-.226 E.00481
; WIPE_START
M204 S10000
G1 X132.559 Y102.855 E-.0882
G1 X132.699 Y102.1 E-.2918
; WIPE_END
G1 E-.02 F1800
G1 X140.217 Y100.782 Z10.84 F36000
G1 X142.938 Y100.305 Z10.84
G1 Z10.44
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.911716
G1 F8975.587
G1 X142.828 Y100.305 E.00627
G1 X142.773 Y100.4 E.00627
G1 X142.828 Y100.495 E.00627
G1 X142.938 Y100.495 E.00627
G1 X142.993 Y100.4 E.00627
; WIPE_START
G1 X142.938 Y100.495 E-.076
G1 X142.828 Y100.495 E-.076
G1 X142.773 Y100.4 E-.076
G1 X142.828 Y100.305 E-.076
G1 X142.938 Y100.305 E-.07599
; WIPE_END
G1 E-.02 F1800
G1 X135.455 Y101.806 Z10.84 F36000
G1 X131.245 Y102.651 Z10.84
G1 Z10.44
G1 E.4 F1800
; LINE_WIDTH: 1.00236
G1 F8135.163
G1 X131.255 Y102.613 E.00247
; LINE_WIDTH: 0.996036
G1 F8188.625
G1 X131.33 Y102.328 E.01845
; LINE_WIDTH: 0.952366
G1 F8578.152
G1 X131.405 Y102.044 E.01761
; LINE_WIDTH: 0.908696
G1 F9006.589
G1 X131.48 Y101.759 E.01678
; LINE_WIDTH: 0.865026
G1 F9480.072
G1 X131.555 Y101.475 E.01594
; LINE_WIDTH: 0.821356
G1 F10006.1
G1 X131.564 Y101.44 E.00185
; LINE_WIDTH: 0.816436
G1 F10069.046
G1 X131.641 Y101.129 E.01631
; LINE_WIDTH: 0.776806
G1 F10606.49
G1 X131.719 Y100.819 E.01548
; LINE_WIDTH: 0.737176
G1 F11204.542
G1 X131.796 Y100.509 E.01465
; LINE_WIDTH: 0.697546
G1 F11874.068
G1 X131.874 Y100.199 E.01383
; WIPE_START
G1 X131.796 Y100.509 E-.12154
G1 X131.719 Y100.819 E-.12154
G1 X131.641 Y101.129 E-.12154
G1 X131.631 Y101.169 E-.01539
; WIPE_END
G1 E-.02 F1800
G1 X131.285 Y105.045 Z10.84 F36000
G1 Z10.44
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X130.37 Y107.201 I-51.913 J-20.747 E.08944
G1 X130.83 Y107.416 E.01937
G3 X132.715 Y109.281 I-4.784 J6.72 E.10169
G2 X135.543 Y111.108 I4.629 J-4.064 E.13019
G2 X138.371 Y110.869 I1.102 J-3.796 E.11082
G2 X140.256 Y109.003 I-4.784 J-6.72 E.10169
G3 X141.945 Y107.642 I4.128 J3.392 E.08344
G1 X141.945 Y114.573 E.26466
G2 X140.256 Y114.957 I-.177 J3.135 E.06702
G2 X138.371 Y116.822 I4.783 J6.72 E.10169
G3 X135.543 Y118.649 I-4.629 J-4.064 E.13019
G3 X132.715 Y118.409 I-1.102 J-3.796 E.11082
G3 X130.83 Y116.544 I4.783 J-6.72 E.10169
G2 X126.311 Y114.601 I-4.142 J3.407 E.19508
G3 X123.83 Y118.095 I-62.189 J-41.541 E.16364
G3 X117.914 Y124.629 I-49.666 J-39.022 E.33681
G2 X120.462 Y126.19 I4.199 J-3.996 E.11537
G2 X123.289 Y125.95 I1.102 J-3.796 E.11082
G2 X125.174 Y124.085 I-4.783 J-6.72 E.10169
G3 X128.002 Y122.258 I4.629 J4.064 E.13019
G3 X130.83 Y122.498 I1.102 J3.796 E.11082
G3 X132.715 Y124.363 I-4.784 J6.72 E.10169
G2 X135.543 Y126.19 I4.629 J-4.064 E.13019
G2 X138.371 Y125.95 I1.102 J-3.796 E.11082
G2 X140.256 Y124.085 I-4.783 J-6.72 E.10169
G3 X141.945 Y122.723 I4.128 J3.392 E.08344
G1 X141.945 Y129.655 E.26465
G2 X140.256 Y130.038 I-.177 J3.135 E.06702
G2 X138.371 Y131.903 I4.784 J6.72 E.10169
G3 X135.543 Y133.73 I-4.629 J-4.064 E.13019
G3 X132.715 Y133.491 I-1.102 J-3.796 E.11082
G3 X130.83 Y131.625 I4.783 J-6.72 E.10169
G2 X128.002 Y129.799 I-4.629 J4.064 E.13019
G2 X125.174 Y130.038 I-1.102 J3.796 E.11082
G2 X123.289 Y131.903 I4.784 J6.72 E.10169
G3 X121.404 Y133.387 I-4.518 J-3.803 E.09223
G1 X121.186 Y133.466 E.00884
G3 X121.111 Y135.984 I-8.591 J1.001 E.0965
G3 X125.907 Y138.469 I-38.679 J80.512 E.20627
G3 X128.002 Y137.339 I3.786 J4.514 E.09152
G3 X130.83 Y137.579 I1.102 J3.796 E.11082
G3 X132.715 Y139.444 I-4.784 J6.72 E.10169
G2 X135.543 Y141.271 I4.629 J-4.064 E.13019
G2 X138.371 Y141.031 I1.102 J-3.796 E.11082
G2 X140.256 Y139.166 I-4.784 J-6.72 E.10169
G3 X141.945 Y137.804 I4.128 J3.392 E.08344
G1 X141.945 Y144.736 E.26466
G2 X139.784 Y145.456 I-.222 J2.936 E.0893
G3 X137.671 Y147.651 I-93.953 J-88.374 E.11634
G2 X135.995 Y146.014 I-36.648 J35.835 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 10.6
; LAYER_HEIGHT: 0.160001
; WIPE_START
G1 F13446.283
G1 X136.71 Y146.713 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L66
M991 S0 P65 ;notify layer change


G17
G3 Z10.84 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.181 Y103.963
G1 Z10.6
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.132 I-51.318 J-19.644 E.48188
G3 X123.072 Y118.225 I-29.869 J-19.713 E.14698
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-40.832 J-37.312 E.29802
G1 X118.015 Y129.012 E.15056
G1 X118.485 Y129.403 E.02332
G1 X119.09 Y130.033 E.03336
G3 X119.749 Y130.932 I-4.531 J4.014 E.04261
G1 X120.215 Y131.838 E.03891
G1 X120.473 Y132.55 E.02891
G1 X120.682 Y133.476 E.03623
G1 X120.765 Y134.489 E.03882
G1 X120.698 Y135.504 E.03883
G1 X120.538 Y136.268 E.02979
G3 X131.308 Y142.707 I-21.876 J48.814 E.48016
G3 X142.443 Y154.069 I-32.826 J43.309 E.60959
G1 X142.443 Y104.736 E1.8835
G1 X141.812 Y104.848 E.02447
G1 X132.779 Y104.848 E.34488
G3 X132.114 Y104.731 I0 J-1.949 E.0259
G1 X131.688 Y104.514 E.01826
G1 X131.242 Y104.03 E.02512
G1 X131.549 Y103.466 F36000
; LINE_WIDTH: 0.664158
G1 F10981.477
G1 X131.543 Y103.451 E.00069
; LINE_WIDTH: 0.708319
G1 F10926.733
G1 X131.496 Y103.337 E.0054
; LINE_WIDTH: 0.752481
G1 F10532.171
G1 X131.45 Y103.223 E.00576
; LINE_WIDTH: 0.796642
G1 F10144.864
G1 X131.404 Y103.109 E.00611
; LINE_WIDTH: 0.840803
G1 F9764.813
G1 X131.358 Y102.995 E.00646
; LINE_WIDTH: 0.884965
G1 F9257.86
G1 X131.312 Y102.881 E.00682
; LINE_WIDTH: 0.929126
G1 F8800.948
G1 X131.266 Y102.767 E.00717
; LINE_WIDTH: 0.964891
G1 F8462.692
G1 X131.255 Y102.71 E.00352
; LINE_WIDTH: 1.00066
G1 F8149.475
G1 X131.244 Y102.653 E.00365
G1 X131.175 Y102.772 E.00865
; LINE_WIDTH: 0.953074
G1 F8571.546
G1 X131.106 Y102.891 E.00823
; LINE_WIDTH: 0.905491
G1 F9039.724
G1 X131.037 Y103.01 E.0078
; LINE_WIDTH: 0.857909
G1 F9562.001
G1 X130.968 Y103.129 E.00737
; LINE_WIDTH: 0.810326
G1 F10148.327
G1 X130.899 Y103.247 E.00695
; LINE_WIDTH: 0.762744
G1 F10581.546
G1 X130.83 Y103.366 E.00652
; LINE_WIDTH: 0.715161
G1 F11023.837
G1 X130.761 Y103.485 E.0061
; LINE_WIDTH: 0.667579
G1 F11475.166
G1 X130.693 Y103.604 E.00567
; LINE_WIDTH: 0.619996
G1 F12841.19
G1 X130.544 Y103.975 E.01527
G1 F13446.369
G1 X130.396 Y104.347 E.01527
G1 X130.07 Y105.17 E.03382
G3 X124.88 Y114.797 I-50.263 J-20.885 E.41825
G3 X122.623 Y117.848 I-29.313 J-19.325 E.14497
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-42.036 J-38.711 E.32253
G1 X117.666 Y129.482 E.17937
G1 X118.108 Y129.851 E.02195
G1 X118.661 Y130.431 E.03062
G1 X119.253 Y131.243 E.03837
G1 X119.689 Y132.095 E.03653
G1 X119.919 Y132.739 E.0261
G1 X120.109 Y133.596 E.03352
G1 X120.18 Y134.525 E.03557
G1 X120.115 Y135.454 E.03555
G1 X119.931 Y136.289 E.03264
G1 X119.836 Y136.599 E.01239
G3 X130.954 Y143.174 I-21.116 J48.399 E.4944
G3 X142.92 Y155.769 I-32.385 J42.75 E.66622
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.464 E1.99618
G1 X142.71 Y103.925 E.0214
G1 X142.233 Y104.195 E.02094
G1 X141.812 Y104.262 E.01626
G1 X132.779 Y104.262 E.34488
G1 X132.314 Y104.18 E.01803
G1 X132.016 Y104.028 E.01277
G1 X131.86 Y103.859 E.00879
G1 X131.589 Y103.565 E.01527
; LINE_WIDTH: 0.664158
G1 F12503.545
G1 X131.583 Y103.55 E.00066
G1 X132.079 Y103.177 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.013 Y102.756 E.01625
G1 X132.586 Y99.665 E.12002
G3 X131.45 Y99.496 I.246 J-5.555 E.04391
G3 X124.399 Y114.463 I-51.525 J-15.129 E.63418
G3 X122.174 Y117.472 I-28.759 J-18.938 E.14296
G1 X119.604 Y120.538 E.15272
G3 X112.93 Y126.698 I-39.743 J-36.365 E.34719
G1 X117.317 Y129.953 E.20858
G1 X117.73 Y130.299 E.02057
G1 X118.231 Y130.83 E.02788
G1 X118.77 Y131.575 E.03508
G1 X119.162 Y132.352 E.03325
G1 X119.365 Y132.927 E.02329
G1 X119.536 Y133.716 E.03081
G1 X119.596 Y134.56 E.03232
G1 X119.531 Y135.403 E.03227
G1 X119.362 Y136.153 E.02936
G1 X119.08 Y136.91 E.03083
G3 X130.601 Y143.641 I-20.65 J48.572 E.51077
G3 X142.647 Y156.415 I-31.975 J42.221 E.67349
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y104.47 E1.97636
G1 X143.615 Y104.07 E.01527
; LINE_WIDTH: 0.665143
G1 F12452.527
G1 X143.592 Y103.835 E.0097
; LINE_WIDTH: 0.71029
G1 F11650.214
G1 X143.569 Y103.6 E.0104
; LINE_WIDTH: 0.755436
G1 F10920.817
G1 X143.547 Y103.365 E.01109
; LINE_WIDTH: 0.797296
G1 F10321.643
G1 X143.526 Y103.132 E.01167
; LINE_WIDTH: 0.839156
G1 F9784.797
G1 X143.505 Y102.898 E.01231
; LINE_WIDTH: 0.844206
G1 F9723.783
G3 X143.515 Y102.398 I4.152 J-.163 E.0264
; LINE_WIDTH: 0.818146
G1 F10047.079
G1 X143.54 Y101.899 E.02555
; LINE_WIDTH: 0.768609
G1 F10724.902
G1 X143.565 Y101.399 E.02393
; LINE_WIDTH: 0.719071
G1 F11500.8
G1 X143.59 Y100.9 E.02232
; LINE_WIDTH: 0.669534
G1 F12397.72
G1 X143.615 Y100.401 E.0207
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X143.615 Y100.001 E.01527
G1 X143.615 Y99.548 E.01727
G1 X142.968 Y99.66 E.02506
G1 X142.007 Y99.673 E.03669
G1 X142.166 Y100.534 E.0334
; LINE_WIDTH: 0.646046
G1 F12814.943
G1 X142.227 Y100.789 E.01049
; LINE_WIDTH: 0.695586
G1 F11909.263
G1 X142.342 Y101.276 E.02155
; LINE_WIDTH: 0.745126
G1 F11079.222
G1 X142.457 Y101.763 E.02317
; LINE_WIDTH: 0.794666
G1 F10357.346
G1 X142.572 Y102.249 E.02478
; LINE_WIDTH: 0.844206
G1 F9723.783
G1 X142.688 Y102.736 E.0264
; LINE_WIDTH: 0.848926
G1 F9667.441
G1 X142.692 Y103.052 E.01681
G1 X142.648 Y103.123 E.00444
; LINE_WIDTH: 0.802181
G1 F10255.977
G1 X142.605 Y103.195 E.00418
; LINE_WIDTH: 0.755436
G1 F10681.522
G1 X142.511 Y103.291 E.00631
; LINE_WIDTH: 0.71029
G1 F11115.716
G1 X142.418 Y103.387 E.00592
; LINE_WIDTH: 0.665143
G1 F11558.536
G1 X142.324 Y103.484 E.00552
; LINE_WIDTH: 0.619996
G1 F12624.611
G1 X142.052 Y103.638 E.01195
G1 F13446.369
G1 X141.812 Y103.676 E.00928
G1 X132.779 Y103.676 E.34488
G3 X132.343 Y103.543 I0 J-.778 E.01765
G1 X132.093 Y103.266 E.01427
G1 X132.093 Y103.265 E0
M204 S250
G1 X132.58 Y103.004 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X132.557 Y102.857 E.00473
G1 X133.248 Y99.123 E.12021
G3 X131.523 Y98.947 I-.175 J-6.804 E.05507
G1 X131.038 Y98.936 E.01535
G3 X123.945 Y114.147 I-51.152 J-14.591 E.53359
G3 X121.747 Y117.121 I-28.18 J-18.533 E.11714
G1 X119.177 Y120.186 E.12664
G3 X112.01 Y126.704 I-40.298 J-37.118 E.30711
G1 X116.988 Y130.397 E.19624
G3 X117.826 Y131.206 I-2.907 J3.85 E.03697
G1 X118.314 Y131.887 E.02653
G1 X118.665 Y132.594 E.025
G1 X118.841 Y133.106 E.01712
G1 X118.995 Y133.83 E.02343
G1 X119.044 Y134.594 E.02426
G1 X118.98 Y135.356 E.0242
G1 X118.824 Y136.026 E.02178
G1 X118.548 Y136.753 E.02464
G1 X118.31 Y137.19 E.01574
G3 X130.27 Y144.084 I-19.513 J47.675 E.43839
G3 X142.39 Y157.025 I-31.658 J41.793 E.56405
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.205 E1.82015
G1 X144.167 Y98.938 E.00843
G1 X143.859 Y98.945 E.00976
G1 X142.836 Y99.118 E.03285
G1 X141.342 Y99.123 E.04729
G1 X142.034 Y102.859 E.1203
M73 P88 R2
G1 X142.023 Y102.977 E.00376
G1 X141.882 Y103.112 E.00621
G1 X132.779 Y103.123 E.2882
G3 X132.638 Y103.074 I0 J-.226 E.00482
; WIPE_START
M204 S10000
G1 X132.557 Y102.857 E-.08816
G1 X132.697 Y102.102 E-.29184
; WIPE_END
G1 E-.02 F1800
G1 X140.214 Y100.784 Z11 F36000
G1 X142.939 Y100.306 Z11
G1 Z10.6
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.909456
G1 F8998.766
G1 X142.829 Y100.306 E.00624
G1 X142.775 Y100.401 E.00624
G1 X142.829 Y100.495 E.00624
G1 X142.939 Y100.495 E.00624
G1 X142.994 Y100.401 E.00624
; WIPE_START
G1 X142.939 Y100.495 E-.076
G1 X142.829 Y100.495 E-.076
G1 X142.775 Y100.401 E-.076
G1 X142.829 Y100.306 E-.076
G1 X142.939 Y100.306 E-.076
; WIPE_END
G1 E-.02 F1800
G1 X135.307 Y100.232 Z11 F36000
G1 X131.873 Y100.199 Z11
G1 Z10.6
G1 E.4 F1800
; LINE_WIDTH: 0.695466
G1 F11911.425
G1 X131.795 Y100.509 E.01377
; LINE_WIDTH: 0.735056
G1 F11238.442
G1 X131.718 Y100.819 E.0146
; LINE_WIDTH: 0.774646
G1 F10637.437
G1 X131.64 Y101.129 E.01542
; LINE_WIDTH: 0.814236
G1 F10097.45
G1 X131.563 Y101.439 E.01625
; LINE_WIDTH: 0.819136
G1 F10034.405
G1 X131.554 Y101.474 E.00184
; LINE_WIDTH: 0.862926
G1 F9504.098
G1 X131.479 Y101.759 E.01594
; LINE_WIDTH: 0.906716
G1 F9027.03
G1 X131.404 Y102.044 E.01678
; LINE_WIDTH: 0.950506
G1 F8595.567
G1 X131.328 Y102.33 E.01763
; LINE_WIDTH: 0.994296
G1 F8203.468
G1 X131.253 Y102.615 E.01847
; LINE_WIDTH: 1.00066
G1 F8149.475
G1 X131.244 Y102.653 E.00247
; WIPE_START
G1 X131.253 Y102.615 E-.01487
G1 X131.328 Y102.33 E-.11215
G1 X131.404 Y102.044 E-.11216
G1 X131.479 Y101.759 E-.11215
G1 X131.498 Y101.686 E-.02867
; WIPE_END
G1 E-.02 F1800
G1 X131.917 Y105.798 Z11 F36000
G1 Z10.6
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X131.647 Y105.706 I.076 J-.667 E.01097
G3 X130.824 Y107.59 I-24.921 J-9.762 E.0785
G3 X132.715 Y109.42 I-9.834 J12.052 E.10058
G2 X135.543 Y111.081 I4.288 J-4.063 E.12687
G2 X138.842 Y110.34 I.898 J-3.714 E.13373
G3 X141.198 Y108.033 I32.264 J30.601 E.12593
G1 X141.36 Y107.942 E.00708
G1 X141.36 Y114.72 E.25877
G2 X140.256 Y115.134 I.493 J2.991 E.04531
G2 X138.371 Y116.961 I6.115 J8.195 E.10051
G3 X136.485 Y118.325 I-4.16 J-3.765 E.08949
G3 X132.715 Y118.232 I-1.791 J-3.896 E.14909
G3 X130.83 Y116.405 I6.115 J-8.195 E.10051
G2 X127.06 Y114.638 I-3.957 J3.534 E.16338
G1 X126.976 Y114.647 E.00323
G3 X124.288 Y118.46 I-45.107 J-28.933 E.17819
G1 X121.32 Y122 E.17636
G3 X118.285 Y125.077 I-36.883 J-33.356 E.16504
G1 X118.576 Y125.333 E.01481
G1 X119.519 Y125.866 E.04133
G2 X123.289 Y125.773 I1.791 J-3.896 E.14909
G2 X125.174 Y123.946 I-6.114 J-8.195 E.10051
G3 X127.06 Y122.582 I4.16 J3.765 E.08949
G3 X130.83 Y122.675 I1.791 J3.896 E.14909
G3 X132.715 Y124.502 I-6.116 J8.196 E.10051
G2 X135.543 Y126.162 I4.288 J-4.063 E.12687
G2 X138.842 Y125.421 I.898 J-3.714 E.13373
G3 X141.198 Y123.114 I32.26 J30.597 E.12593
G1 X141.36 Y123.023 E.00708
G1 X141.36 Y129.801 E.25877
G2 X140.256 Y130.215 I.492 J2.99 E.04531
G2 X138.371 Y132.042 I6.115 J8.195 E.10051
G3 X136.485 Y133.406 I-4.16 J-3.765 E.08949
G3 X132.715 Y133.314 I-1.791 J-3.896 E.14909
G3 X130.83 Y131.487 I6.114 J-8.195 E.10051
G2 X128.002 Y129.826 I-4.288 J4.063 E.12687
G2 X124.703 Y130.567 I-.898 J3.714 E.13373
G3 X122.347 Y132.874 I-32.26 J-30.596 E.12593
G1 X121.735 Y133.219 E.02683
G3 X121.792 Y135.431 I-12.611 J1.433 E.08457
G1 X121.774 Y135.64 E.008
G3 X126.357 Y138.06 I-25.352 J53.557 E.19795
G1 X127.06 Y137.663 E.03079
G3 X130.83 Y137.756 I1.791 J3.896 E.14909
G3 X132.715 Y139.583 I-6.115 J8.196 E.10051
G2 X135.543 Y141.244 I4.288 J-4.063 E.12687
G2 X138.842 Y140.502 I.898 J-3.714 E.13373
G3 X141.198 Y138.196 I32.256 J30.593 E.12593
G1 X141.36 Y138.105 E.00708
G1 X141.36 Y144.882 E.25877
G2 X139.784 Y145.648 I.533 J3.099 E.06777
G3 X138.157 Y147.312 I-1291.466 J-1261.683 E.08883
G2 X136.845 Y146.009 I-45.795 J44.854 E.07059
G1 X136.485 Y145.671 E.01885
G1 X126.399 Y114.547 F36000
; FEATURE: Floating vertical shell
; LINE_WIDTH: 0.544336
G1 F12000
G3 X124.464 Y117.332 I-55.821 J-36.72 E.11276
G1 X123.864 Y118.123 E.033
G1 X120.914 Y121.642 E.1527
G3 X115.961 Y126.434 I-41.148 J-37.579 E.22925
; LINE_WIDTH: 0.561636
G1 X115.912 Y126.5 E.00283
; LINE_WIDTH: 0.596236
G1 X115.864 Y126.567 E.00302
; LINE_WIDTH: 0.630836
G1 X115.816 Y126.633 E.0032
G1 X115.868 Y126.697 E.0032
; LINE_WIDTH: 0.596236
G1 X115.921 Y126.76 E.00302
; LINE_WIDTH: 0.544887
G1 X115.974 Y126.823 E.00274
G1 X118.345 Y128.582 E.09825
G1 X118.909 Y129.055 E.02451
G1 X119.555 Y129.749 E.03157
G1 X120.254 Y130.716 E.03969
G1 X120.694 Y131.603 E.03297
G1 X121.047 Y132.586 E.03475
G1 X121.224 Y133.423 E.02849
G1 X121.294 Y134.458 E.03451
G3 X121.193 Y135.941 I-12.459 J-.105 E.04952
G1 X121.335 Y136.041 E.00576
G3 X131.633 Y142.273 I-22.972 J49.58 E.40143
G3 X141.441 Y151.854 I-33.732 J44.344 E.45751
G1 X141.547 Y151.953 E.00482
G1 X141.771 Y151.903 E.00766
G1 X141.881 Y151.836 E.00426
G1 X141.901 Y151.692 E.00486
G1 X141.901 Y105.638 E1.53279
; LINE_WIDTH: 0.563196
G1 X141.88 Y105.534 E.00368
; LINE_WIDTH: 0.588426
G1 X141.858 Y105.43 E.00385
G1 X141.652 Y105.389 E.00755
; LINE_WIDTH: 0.544336
G1 X132.768 Y105.389 E.29537
G3 X131.609 Y105.079 I0 J-2.321 E.04035
; LINE_WIDTH: 0.565256
G1 X131.501 Y105.051 E.00387
; LINE_WIDTH: 0.59315
G1 X131.393 Y105.023 E.00407
G1 X131.266 Y105.207 E.00814
; LINE_WIDTH: 0.544336
G1 X130.846 Y106.234 E.03688
G1 X129.885 Y108.347 E.07717
G3 X126.447 Y114.471 I-50.38 J-24.262 E.23366
; CHANGE_LAYER
; Z_HEIGHT: 10.76
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F12000
G1 X126.974 Y113.621 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L67
M991 S0 P66 ;notify layer change


G17
G3 Z11 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.18 Y103.967
G1 Z10.76
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.348 Y115.15 I-51.312 J-19.645 E.48262
G3 X123.072 Y118.224 I-29.794 J-19.682 E.14611
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-41.983 J-38.524 E.29802
G1 X118.015 Y129.012 E.15054
G1 X118.501 Y129.417 E.02417
G3 X119.749 Y130.931 I-5.151 J5.519 E.07513
G1 X120.215 Y131.838 E.03891
G1 X120.472 Y132.546 E.02875
G1 X120.682 Y133.475 E.03638
G1 X120.765 Y134.489 E.03885
G1 X120.698 Y135.507 E.03892
G1 X120.538 Y136.268 E.02969
G3 X130.504 Y142.098 I-22.367 J49.66 E.44164
G3 X142.443 Y154.069 I-32.641 J44.496 E.64805
G1 X142.443 Y104.739 E1.88337
G1 X141.815 Y104.85 E.02437
G1 X132.776 Y104.85 E.34508
G3 X132.109 Y104.732 I0 J-1.949 E.02598
G1 X131.681 Y104.513 E.01835
G1 X131.241 Y104.033 E.02488
G1 X131.546 Y103.467 F36000
; LINE_WIDTH: 0.664183
G1 F10980.241
G1 X131.539 Y103.451 E.00069
; LINE_WIDTH: 0.708371
G1 F10925.843
G1 X131.493 Y103.337 E.00541
; LINE_WIDTH: 0.752558
G1 F10531.04
G1 X131.447 Y103.223 E.00576
; LINE_WIDTH: 0.796745
G1 F10143.53
G1 X131.401 Y103.109 E.00611
; LINE_WIDTH: 0.840932
G1 F9763.256
G1 X131.355 Y102.995 E.00647
; LINE_WIDTH: 0.885119
G1 F9256.181
G1 X131.309 Y102.881 E.00682
; LINE_WIDTH: 0.929306
G1 F8799.178
G1 X131.263 Y102.767 E.00718
; LINE_WIDTH: 0.964111
G1 F8469.791
G1 X131.253 Y102.711 E.00344
; LINE_WIDTH: 0.998916
G1 F8164.176
G1 X131.242 Y102.655 E.00357
G1 X131.173 Y102.774 E.00862
; LINE_WIDTH: 0.951551
G1 F8585.774
G1 X131.105 Y102.893 E.00819
; LINE_WIDTH: 0.904186
G1 F9053.285
G1 X131.036 Y103.011 E.00777
; LINE_WIDTH: 0.856821
G1 F9574.643
G1 X130.967 Y103.13 E.00735
; LINE_WIDTH: 0.809456
G1 F10159.718
G1 X130.899 Y103.248 E.00692
; LINE_WIDTH: 0.762091
G1 F10592.086
G1 X130.83 Y103.367 E.0065
; LINE_WIDTH: 0.714726
G1 F11033.491
G1 X130.761 Y103.485 E.00608
; LINE_WIDTH: 0.667361
G1 F11483.878
G1 X130.693 Y103.604 E.00565
; LINE_WIDTH: 0.619996
G1 F12850.406
G1 X130.544 Y103.975 E.01527
G1 F13446.369
G1 X130.396 Y104.347 E.01527
G1 X130.071 Y105.168 E.03373
G3 X124.867 Y114.816 I-50.259 J-20.881 E.4192
G3 X122.623 Y117.848 I-29.242 J-19.296 E.1441
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-40.113 J-36.658 E.32257
G1 X117.666 Y129.482 E.17935
G1 X118.122 Y129.865 E.02274
G1 X118.658 Y130.428 E.02967
G1 X119.253 Y131.243 E.03854
G1 X119.688 Y132.095 E.03652
G1 X119.918 Y132.735 E.02595
G1 X120.109 Y133.595 E.03367
G1 X120.18 Y134.525 E.03559
G1 X120.114 Y135.456 E.03564
G1 X119.931 Y136.289 E.03255
G1 X119.836 Y136.599 E.01239
G3 X130.932 Y143.157 I-21.039 J48.268 E.49336
G3 X142.92 Y155.769 I-32.175 J42.587 E.6673
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.47 E1.99596
G1 X142.712 Y103.927 E.02124
G1 X142.235 Y104.197 E.02094
G1 X141.815 Y104.264 E.01626
G1 X132.776 Y104.264 E.34508
G1 X132.31 Y104.182 E.01809
G1 X132.01 Y104.029 E.01284
G1 X131.856 Y103.86 E.00872
G1 X131.585 Y103.565 E.01527
; LINE_WIDTH: 0.664183
G1 F12503.035
G1 X131.579 Y103.55 E.00067
G1 X132.076 Y103.178 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.011 Y102.758 E.01622
G1 X132.583 Y99.667 E.12003
G3 X131.45 Y99.496 I.246 J-5.493 E.04383
G3 X124.387 Y114.481 I-51.521 J-15.128 E.63502
G3 X122.174 Y117.472 I-28.686 J-18.908 E.1421
G1 X119.605 Y120.537 E.15272
G3 X112.93 Y126.698 I-39.788 J-36.414 E.3472
G1 X117.325 Y129.958 E.20893
G3 X118.228 Y130.826 I-3.293 J4.332 E.04791
G1 X118.77 Y131.575 E.03529
G1 X119.162 Y132.352 E.03324
G1 X119.364 Y132.924 E.02315
G1 X119.536 Y133.716 E.03095
G1 X119.596 Y134.561 E.03233
G1 X119.531 Y135.405 E.03235
G1 X119.361 Y136.153 E.02927
G1 X119.08 Y136.91 E.03084
G3 X130.579 Y143.624 I-20.65 J48.572 E.50973
G3 X142.647 Y156.415 I-31.779 J42.072 E.67456
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y104.471 E1.97633
G1 X143.615 Y104.071 E.01527
; LINE_WIDTH: 0.664923
G1 F12459.544
G1 X143.592 Y103.836 E.00969
; LINE_WIDTH: 0.70985
G1 F11657.802
G1 X143.57 Y103.601 E.01038
; LINE_WIDTH: 0.754776
G1 F10930.821
G1 X143.547 Y103.367 E.01107
; LINE_WIDTH: 0.796341
G1 F10334.579
G1 X143.526 Y103.133 E.01165
; LINE_WIDTH: 0.837906
G1 F9800.019
G1 X143.506 Y102.9 E.01228
; LINE_WIDTH: 0.842976
G1 F9738.574
G3 X143.516 Y102.399 I4.18 J-.163 E.02642
; LINE_WIDTH: 0.816936
G1 F10062.612
G1 X143.541 Y101.9 E.02551
; LINE_WIDTH: 0.767701
G1 F10738.173
G1 X143.565 Y101.4 E.0239
; LINE_WIDTH: 0.718466
G1 F11510.972
G1 X143.59 Y100.901 E.0223
; LINE_WIDTH: 0.669231
G1 F12403.627
G1 X143.615 Y100.401 E.02069
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X143.615 Y100.001 E.01527
G1 X143.615 Y99.563 E.01672
G1 X142.925 Y99.668 E.02664
G1 X142.009 Y99.676 E.03495
G1 X142.168 Y100.534 E.03334
; LINE_WIDTH: 0.646026
G1 F12826.09
G1 X142.229 Y100.792 E.01054
; LINE_WIDTH: 0.695264
G1 F11915.075
G1 X142.344 Y101.278 E.02154
; LINE_WIDTH: 0.744501
G1 F11088.973
G1 X142.459 Y101.765 E.02315
; LINE_WIDTH: 0.793739
G1 F10369.996
G1 X142.575 Y102.251 E.02475
; LINE_WIDTH: 0.842976
G1 F9738.574
G1 X142.69 Y102.738 E.02636
; LINE_WIDTH: 0.847666
G1 F9682.417
G1 X142.694 Y103.054 E.01677
G1 X142.65 Y103.125 E.00442
; LINE_WIDTH: 0.801221
G1 F10268.815
G1 X142.607 Y103.197 E.00417
; LINE_WIDTH: 0.754776
G1 F10694.492
G1 X142.513 Y103.293 E.0063
; LINE_WIDTH: 0.70985
G1 F11128.768
G1 X142.42 Y103.39 E.00591
; LINE_WIDTH: 0.664923
G1 F11571.686
G1 X142.327 Y103.486 E.00552
; LINE_WIDTH: 0.619996
G1 F12638.384
G1 X142.055 Y103.64 E.01195
G1 F13446.369
G1 X141.815 Y103.678 E.00928
G1 X132.776 Y103.678 E.34508
G3 X132.339 Y103.544 I0 J-.778 E.01772
G1 X132.09 Y103.267 E.01423
G1 X132.09 Y103.267 E.00001
M204 S250
G1 X132.577 Y103.006 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X132.554 Y102.859 E.00473
G1 X133.246 Y99.125 E.12021
G3 X131.522 Y98.947 I-.173 J-6.734 E.05502
G1 X131.038 Y98.936 E.01534
G3 X123.933 Y114.166 I-51.149 J-14.59 E.5343
G3 X121.747 Y117.121 I-28.099 J-18.496 E.11643
G1 X119.177 Y120.186 E.12664
G3 X112.01 Y126.704 I-38.916 J-35.597 E.30715
G1 X116.99 Y130.398 E.19631
G3 X117.822 Y131.201 I-2.928 J3.866 E.03671
G1 X118.314 Y131.887 E.02672
G1 X118.665 Y132.594 E.025
G1 X118.84 Y133.102 E.01701
G1 X118.995 Y133.829 E.02354
G1 X119.044 Y134.594 E.02426
G1 X118.98 Y135.358 E.02426
G1 X118.824 Y136.025 E.0217
G1 X118.548 Y136.753 E.02466
G1 X118.31 Y137.19 E.01573
G3 X130.249 Y144.067 I-19.583 J47.797 E.43753
G3 X142.39 Y157.025 I-31.645 J41.817 E.56491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.196 E1.82041
G1 X144.167 Y98.938 E.00817
G1 X143.86 Y98.947 E.00975
G3 X142.835 Y99.12 I-1.45 J-5.451 E.03296
G1 X141.345 Y99.125 E.04717
G1 X142.037 Y102.861 E.1203
G1 X142.026 Y102.979 E.00376
G1 X141.884 Y103.115 E.00621
G1 X132.776 Y103.125 E.28836
G3 X132.635 Y103.076 I0 J-.226 E.00483
; WIPE_START
M204 S10000
G1 X132.554 Y102.859 E-.08808
G1 X132.694 Y102.103 E-.29192
; WIPE_END
G1 E-.02 F1800
G1 X140.212 Y100.785 Z11.16 F36000
G1 X142.94 Y100.307 Z11.16
G1 Z10.76
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.907196
G1 F9022.066
G1 X142.831 Y100.307 E.00621
G1 X142.776 Y100.401 E.00621
G1 X142.831 Y100.496 E.00621
G1 X142.94 Y100.496 E.00621
G1 X142.994 Y100.401 E.00621
; WIPE_START
G1 X142.94 Y100.496 E-.076
G1 X142.831 Y100.496 E-.076
G1 X142.776 Y100.401 E-.076
G1 X142.831 Y100.307 E-.076
G1 X142.94 Y100.307 E-.076
; WIPE_END
G1 E-.02 F1800
G1 X135.457 Y101.809 Z11.16 F36000
G1 X131.242 Y102.655 Z11.16
G1 Z10.76
G1 E.4 F1800
; LINE_WIDTH: 0.998916
G1 F8164.176
G1 X131.252 Y102.617 E.00246
; LINE_WIDTH: 0.992616
G1 F8217.85
G1 X131.327 Y102.331 E.01848
; LINE_WIDTH: 0.948721
G1 F8612.347
G1 X131.402 Y102.045 E.01763
; LINE_WIDTH: 0.904826
G1 F9046.63
G1 X131.478 Y101.759 E.01679
; LINE_WIDTH: 0.860931
G1 F9527.036
G1 X131.553 Y101.473 E.01594
; LINE_WIDTH: 0.817036
G1 F10061.326
G1 X131.562 Y101.438 E.00183
; LINE_WIDTH: 0.812136
G1 F10124.712
G1 X131.639 Y101.129 E.01619
; LINE_WIDTH: 0.772571
G1 F10667.336
G1 X131.717 Y100.819 E.01537
; LINE_WIDTH: 0.733006
G1 F11271.416
G1 X131.794 Y100.509 E.01455
; LINE_WIDTH: 0.693441
G1 F11948.021
G1 X131.872 Y100.199 E.01372
; WIPE_START
G1 X131.794 Y100.509 E-.12137
G1 X131.717 Y100.819 E-.12137
G1 X131.639 Y101.129 E-.12137
G1 X131.629 Y101.169 E-.0159
; WIPE_END
G1 E-.02 F1800
G1 X138.678 Y104.095 Z11.16 F36000
G1 X141.068 Y105.087 Z11.16
G1 Z10.76
G1 E.4 F1800
; FEATURE: Bridge
; LINE_WIDTH: 0.60413
; LAYER_HEIGHT: 0.6

G1 X141.913 Y106.311 E.17372
G1 X141.913 Y107.463 E.1345
G1 X140.475 Y105.38 E.29563
G1 X139.681 Y105.38 E.09282
G1 X141.913 Y108.615 E.45906
G1 X141.913 Y109.766 E.1345
G1 X138.886 Y105.38 E.62248
G1 X138.091 Y105.38 E.09282
G1 X141.913 Y110.918 E.78591
G1 X141.913 Y112.07 E.1345
G1 X137.296 Y105.38 E.94933
G1 X136.502 Y105.38 E.09282
G1 X141.913 Y113.221 E1.11275
G1 X141.913 Y114.373 E.1345
G1 X135.707 Y105.38 E1.27618
G1 X134.912 Y105.38 E.09282
G1 X141.913 Y115.525 E1.4396
G1 X141.913 Y116.676 E.13451
G1 X134.117 Y105.38 E1.60303
G1 X133.323 Y105.38 E.09282
G1 X141.913 Y117.828 E1.76645
G1 X141.913 Y118.98 E.1345
M73 P89 R2
G1 X132.513 Y105.359 E1.93287
G3 X132.032 Y105.258 I.043 J-1.415 E.05774
G1 X131.924 Y105.23 E.01299
G1 X131.469 Y104.998 E.05965
G1 X141.913 Y120.131 E2.14752
G1 X141.913 Y121.283 E.1345
G1 X131.092 Y105.602 E2.22515
G3 X130.793 Y106.321 I-9.532 J-3.534 E.09096
G1 X141.913 Y122.435 E2.28652
G1 X141.913 Y123.586 E.1345
G1 X130.478 Y107.016 E2.35143
G1 X130.162 Y107.71 E.08907
G1 X141.913 Y124.738 E2.41633
G1 X141.913 Y125.89 E.1345
G1 X129.845 Y108.402 E2.48155
G1 X129.512 Y109.072 E.08736
G1 X141.913 Y127.041 E2.54988
G1 X141.913 Y128.193 E.13451
G1 X129.177 Y109.737 E2.61892
G1 X128.828 Y110.384 E.0858
G1 X141.913 Y129.344 E2.69057
G1 X141.913 Y130.496 E.1345
G1 X128.48 Y111.031 E2.76222
G3 X128.13 Y111.676 I-8.313 J-4.084 E.08573
G1 X141.913 Y131.648 E2.83407
G1 X141.913 Y132.799 E.1345
G1 X127.764 Y112.296 E2.90947
G1 X127.397 Y112.917 E.08416
M73 P89 R1
G1 X141.913 Y133.951 E2.98487
G1 X141.913 Y135.103 E.1345
G1 X127.028 Y113.533 E3.06081
G1 X126.647 Y114.133 E.08301
G1 X141.913 Y136.254 E3.13906
G1 X141.913 Y137.406 E.1345
G1 X126.264 Y114.73 E3.2178
G1 X125.874 Y115.316 E.08225
M73 P90 R1
G1 X141.913 Y138.558 E3.29802
G1 X141.913 Y139.709 E.1345
G1 X125.474 Y115.888 E3.38026
G1 X125.071 Y116.456 E.08131
G1 X141.913 Y140.861 E3.46312
G1 X141.913 Y142.013 E.1345
G1 X124.668 Y117.024 E3.54597
G3 X124.259 Y117.583 I-8.918 J-6.102 E.0809
G1 X141.913 Y143.164 E3.63011
G1 X141.913 Y144.316 E.1345
G1 X123.842 Y118.13 E3.71583
G1 X123.406 Y118.65 E.07925
G1 X141.913 Y145.468 E3.80546
G1 X141.913 Y146.619 E.1345
G1 X122.97 Y119.17 E3.89509
G1 X122.534 Y119.69 E.07925
G1 X141.913 Y147.771 E3.98472
G1 X141.913 Y148.923 E.13451
G1 X122.099 Y120.21 E4.07435
G1 X121.663 Y120.731 E.07925
G1 X141.913 Y150.074 E4.16398
G1 X141.913 Y151.226 E.1345
M73 P91 R1
G1 X121.227 Y121.251 E4.25361
G3 X120.786 Y121.763 I-5.639 J-4.4 E.07902
G1 X141.574 Y151.886 E4.27442
G1 X141.479 Y151.919 E.01172
G2 X138.199 Y148.147 I-45.259 J36.047 E.584
G1 X120.331 Y122.256 E3.67393
G3 X119.867 Y122.734 I-8.632 J-7.929 E.07789
G1 X135.729 Y145.719 E3.26164
G1 X134.049 Y144.231 E.26205
G1 X133.718 Y143.957 E.05029
G1 X119.401 Y123.211 E2.94385
G1 X118.935 Y123.688 E.07785
G1 X131.936 Y142.527 E2.67319
G2 X130.286 Y141.288 I-23.212 J29.201 E.24088
G1 X118.46 Y124.151 E2.43188
G3 X117.979 Y124.606 I-7.682 J-7.615 E.07734
G1 X128.751 Y140.215 E2.21486
G2 X128.343 Y139.948 I-4.507 J6.432 E.0569
G1 X127.312 Y139.281 E.14347
G1 X117.324 Y124.808 E2.05373
G1 X115.85 Y126.128 F36000
G1 F1800
G1 X116.724 Y127.394 E.1797
G1 X118.353 Y128.603 E.23683
G1 X116.516 Y125.941 E.37775
G1 X117.003 Y125.496 E.0771
G1 X125.948 Y138.456 E1.83913
G2 X124.65 Y137.727 I-11.505 J18.966 E.17386
G1 X120.976 Y132.404 E.7554
M73 P92 R1
G1 X121.037 Y132.592 E.02309
G1 X121.216 Y133.442 E.10145
G1 X121.256 Y133.961 E.06073
G1 X123.394 Y137.059 E.43966
G2 X122.182 Y136.455 I-15.699 J29.98 E.15811
G1 X120.998 Y134.739 E.24348
; WIPE_START
G1 X121.566 Y135.562 E-.38
; WIPE_END
G1 E-.02
G1 X127.455 Y140.418 Z11.16 F36000
G1 X141.92 Y152.343 Z11.16
G1 Z10.76
G1 E.4 F1800
; FEATURE: Floating vertical shell
; LINE_WIDTH: 0.562116
; LAYER_HEIGHT: 0.16
G1 F12000
G2 X141.895 Y152.457 I-.033 J.053 E.00823
; CHANGE_LAYER
; Z_HEIGHT: 10.92
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F12000
G1 X141.854 Y152.457 E-.06645
G1 X141.821 Y152.4 E-.10452
G1 X141.854 Y152.343 E-.10453
G1 X141.92 Y152.343 E-.10451
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L68
M991 S0 P67 ;notify layer change


G17
G3 Z11.16 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.179 Y103.97
G1 Z10.92
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.131 I-51.329 J-19.657 E.48161
G3 X123.072 Y118.225 I-29.874 J-19.717 E.147
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-40.981 J-37.47 E.29802
G1 X118.016 Y129.012 E.15058
G3 X118.932 Y129.849 I-4.611 J5.972 E.04744
G3 X120.075 Y131.529 I-5.864 J5.216 E.07777
G1 X120.45 Y132.48 E.03904
G3 X120.765 Y134.489 I-7.443 J2.194 E.07787
G1 X120.698 Y135.504 E.03884
G1 X120.538 Y136.268 E.02979
G3 X130.504 Y142.099 I-22.609 J50.072 E.44163
G3 X142.443 Y154.069 I-32.714 J44.569 E.64803
G1 X142.443 Y104.743 E1.88323
G1 X141.817 Y104.852 E.02427
G1 X132.774 Y104.852 E.34527
G3 X132.105 Y104.733 I0 J-1.949 E.02607
G1 X131.675 Y104.513 E.01845
G1 X131.24 Y104.036 E.02464
G1 X131.54 Y103.449 F36000
; LINE_WIDTH: 0.713291
G1 F10783.36
G1 X131.498 Y103.332 E.0055
; LINE_WIDTH: 0.759939
G1 F10387.016
G1 X131.457 Y103.215 E.00588
; LINE_WIDTH: 0.806586
G1 F9998.092
G1 X131.415 Y103.098 E.00626
; LINE_WIDTH: 0.853234
G1 F9616.589
G1 X131.373 Y102.981 E.00664
; LINE_WIDTH: 0.899881
G1 F9098.314
G1 X131.331 Y102.864 E.00702
; LINE_WIDTH: 0.946529
G1 F8633.047
G1 X131.289 Y102.747 E.00739
; LINE_WIDTH: 0.993176
G1 F8213.05
G1 X131.247 Y102.63 E.00777
G1 X131.178 Y102.751 E.00873
; LINE_WIDTH: 0.946529
G1 F8633.047
G1 X131.109 Y102.872 E.0083
; LINE_WIDTH: 0.899881
G1 F9098.314
G1 X131.04 Y102.994 E.00788
; LINE_WIDTH: 0.853234
G1 F9616.589
G1 X130.971 Y103.115 E.00745
; LINE_WIDTH: 0.806586
G1 F10197.475
G1 X130.901 Y103.236 E.00703
; LINE_WIDTH: 0.759939
G1 F10639.087
G1 X130.832 Y103.358 E.00661
; LINE_WIDTH: 0.713291
G1 F11090.032
G1 X130.763 Y103.479 E.00618
; LINE_WIDTH: 0.666644
G1 F11550.336
G1 X130.694 Y103.6 E.00576
; LINE_WIDTH: 0.619996
G1 F12920.701
G1 X130.546 Y103.972 E.01527
G1 F13446.369
G1 X130.397 Y104.343 E.01527
G1 X130.072 Y105.166 E.0338
G3 X124.88 Y114.797 I-50.277 J-20.887 E.41841
G3 X122.623 Y117.848 I-29.323 J-19.332 E.14499
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-40.114 J-36.659 E.32256
G1 X117.667 Y129.482 E.17939
G3 X118.523 Y130.268 I-5.362 J6.707 E.04441
G3 X119.208 Y131.168 I-7.761 J6.612 E.04321
G1 X119.561 Y131.811 E.028
G1 X119.901 Y132.683 E.03575
G3 X120.135 Y133.79 I-9.569 J2.601 E.04321
G1 X120.18 Y134.525 E.02809
G1 X120.115 Y135.454 E.03555
G1 X119.931 Y136.289 E.03265
G1 X119.836 Y136.599 E.01239
G3 X130.931 Y143.156 I-21.124 J48.413 E.4933
G3 X142.92 Y155.769 I-32.174 J42.587 E.66736
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.475 E1.99574
G1 X142.715 Y103.929 E.02107
G1 X142.238 Y104.2 E.02094
G1 X141.817 Y104.266 E.01626
G1 X132.774 Y104.266 E.34527
G1 X132.306 Y104.183 E.01815
G1 X132.005 Y104.029 E.01291
G1 X131.852 Y103.861 E.00865
G1 X131.582 Y103.566 E.01527
; LINE_WIDTH: 0.666644
G1 F12454.384
G1 X131.571 Y103.534 E.00142
G1 X132.072 Y103.179 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.007 Y102.765 E.01598
G1 X132.581 Y99.668 E.12024
G3 X131.45 Y99.497 I.248 J-5.439 E.04375
G3 X124.399 Y114.462 I-51.523 J-15.129 E.63414
G3 X122.174 Y117.472 I-28.759 J-18.938 E.14298
G1 X119.604 Y120.538 E.15272
G3 X112.93 Y126.698 I-39.747 J-36.37 E.34719
G1 X117.318 Y129.953 E.20859
G3 X118.151 Y130.725 I-6.504 J7.848 E.0434
G3 X118.695 Y131.451 I-12.206 J9.722 E.03463
G1 X119.048 Y132.093 E.028
G1 X119.352 Y132.887 E.03246
G3 X119.551 Y133.826 I-11.666 J2.956 E.03664
G1 X119.595 Y134.56 E.02809
G1 X119.531 Y135.403 E.03227
G1 X119.361 Y136.154 E.02937
G1 X119.08 Y136.91 E.03082
G3 X130.578 Y143.623 I-20.899 J48.999 E.50965
G3 X142.648 Y156.415 I-31.779 J42.074 E.67461
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y104.472 E1.97629
G1 X143.615 Y104.072 E.01527
; LINE_WIDTH: 0.664703
G1 F12466.574
G1 X143.592 Y103.837 E.00967
; LINE_WIDTH: 0.70941
G1 F11665.4
G1 X143.57 Y103.603 E.01036
; LINE_WIDTH: 0.754116
G1 F10940.844
G1 X143.547 Y103.369 E.01105
; LINE_WIDTH: 0.795381
G1 F10347.615
G1 X143.527 Y103.135 E.01162
; LINE_WIDTH: 0.836646
G1 F9815.409
G1 X143.506 Y102.902 E.01225
; LINE_WIDTH: 0.841736
G1 F9753.53
G3 X143.517 Y102.4 I4.205 J-.164 E.02645
; LINE_WIDTH: 0.815716
G1 F10078.324
G1 X143.541 Y101.9 E.02547
; LINE_WIDTH: 0.766786
G1 F10751.588
G1 X143.566 Y101.401 E.02387
; LINE_WIDTH: 0.717856
G1 F11521.244
G1 X143.59 Y100.902 E.02228
; LINE_WIDTH: 0.668926
G1 F12409.589
G1 X143.615 Y100.402 E.02068
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X143.615 Y100.002 E.01527
G1 X143.615 Y99.564 E.01672
G1 X142.924 Y99.67 E.02666
G1 X142.012 Y99.678 E.03484
G1 X142.17 Y100.535 E.03329
; LINE_WIDTH: 0.646006
G1 F12837.301
G1 X142.232 Y100.794 E.0106
; LINE_WIDTH: 0.694939
G1 F11920.936
G1 X142.347 Y101.28 E.02153
; LINE_WIDTH: 0.743871
G1 F11098.819
G1 X142.462 Y101.767 E.02313
; LINE_WIDTH: 0.792804
G1 F10382.779
G1 X142.577 Y102.253 E.02472
; LINE_WIDTH: 0.841736
G1 F9753.53
G1 X142.692 Y102.74 E.02632
; LINE_WIDTH: 0.846396
G1 F9697.56
G1 X142.695 Y103.056 E.01673
G1 X142.652 Y103.127 E.00441
; LINE_WIDTH: 0.800256
G1 F10281.753
G1 X142.609 Y103.199 E.00416
; LINE_WIDTH: 0.754116
G1 F10707.564
G1 X142.516 Y103.295 E.0063
; LINE_WIDTH: 0.70941
G1 F11141.992
G1 X142.423 Y103.392 E.0059
; LINE_WIDTH: 0.664703
G1 F11585.058
G1 X142.329 Y103.488 E.00551
; LINE_WIDTH: 0.619996
G1 F12652.298
G1 X142.057 Y103.642 E.01195
G1 F13446.369
G1 X141.817 Y103.68 E.00928
G1 X132.774 Y103.68 E.34527
G3 X132.335 Y103.545 I0 J-.779 E.0178
G1 X132.087 Y103.267 E.01423
M204 S250
G1 X132.574 Y103.008 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X132.552 Y102.861 E.0047
G1 X133.243 Y99.127 E.12021
G3 X131.521 Y98.947 I-.17 J-6.671 E.05497
G1 X131.038 Y98.936 E.01531
G3 X123.946 Y114.147 I-51.151 J-14.591 E.53358
G3 X121.747 Y117.121 I-28.177 J-18.53 E.11717
G1 X119.177 Y120.187 E.12664
G3 X112.01 Y126.704 I-40.297 J-37.117 E.3071
G1 X116.988 Y130.397 E.19625
G3 X117.738 Y131.094 I-6.413 J7.655 E.03243
G3 X118.21 Y131.717 I-28.727 J22.235 E.02476
G1 X118.564 Y132.36 E.02322
G3 X118.98 Y135.355 I-5.039 J2.227 E.097
G1 X118.824 Y136.026 E.02179
G1 X118.548 Y136.753 E.02463
G1 X118.31 Y137.19 E.01574
G3 X130.248 Y144.067 I-19.802 J48.176 E.43746
G3 X142.39 Y157.025 I-31.644 J41.818 E.56495
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.196 E1.82042
G1 X144.167 Y98.938 E.00815
G1 X143.86 Y98.947 E.00974
G3 X142.833 Y99.122 I-1.454 J-5.417 E.03301
G1 X141.347 Y99.127 E.04704
G1 X142.039 Y102.863 E.1203
G1 X142.028 Y102.981 E.00376
G1 X141.887 Y103.117 E.00621
G1 X132.774 Y103.128 E.28852
G3 X132.632 Y103.077 I0 J-.226 E.00485
; WIPE_START
M204 S10000
G1 X132.552 Y102.861 E-.08778
G1 X132.692 Y102.105 E-.29222
; WIPE_END
G1 E-.02 F1800
G1 X140.21 Y100.787 Z11.32 F36000
G1 X142.941 Y100.308 Z11.32
G1 Z10.92
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.904936
G1 F9045.486
G1 X142.832 Y100.308 E.00618
G1 X142.778 Y100.402 E.00618
G1 X142.832 Y100.497 E.00618
G1 X142.941 Y100.497 E.00618
G1 X142.995 Y100.402 E.00618
; WIPE_START
G1 X142.941 Y100.497 E-.076
G1 X142.832 Y100.497 E-.076
G1 X142.778 Y100.402 E-.076
G1 X142.832 Y100.308 E-.076
G1 X142.941 Y100.308 E-.076
; WIPE_END
G1 E-.02 F1800
G1 X135.454 Y101.794 Z11.32 F36000
G1 X131.247 Y102.63 Z11.32
G1 Z10.92
G1 E.4 F1800
; LINE_WIDTH: 0.993176
G1 F8213.05
G1 X131.302 Y102.429 E.013
; LINE_WIDTH: 0.958756
G1 F8518.855
G1 X131.388 Y102.099 E.02056
; LINE_WIDTH: 0.909216
G1 F9001.235
G1 X131.474 Y101.769 E.01946
; LINE_WIDTH: 0.859676
G1 F9541.523
G1 X131.561 Y101.439 E.01836
; LINE_WIDTH: 0.810136
G1 F10150.813
G1 X131.638 Y101.129 E.01615
; LINE_WIDTH: 0.770556
G1 F10696.532
G1 X131.716 Y100.819 E.01533
; LINE_WIDTH: 0.730976
G1 F11304.262
G1 X131.793 Y100.509 E.01451
; LINE_WIDTH: 0.691396
G1 F11985.208
G1 X131.871 Y100.199 E.01368
; WIPE_START
G1 X131.793 Y100.509 E-.12138
G1 X131.716 Y100.819 E-.12138
G1 X131.638 Y101.129 E-.12138
G1 X131.628 Y101.169 E-.01586
; WIPE_END
G1 E-.02 F1800
G1 X131.945 Y104.912 Z11.32 F36000
G1 Z10.92
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.626776
G1 F13292.485
G1 X130.92 Y105.937 E.05599
G3 X130.245 Y107.45 I-20.092 J-8.059 E.06401
G1 X132.378 Y105.317 E.1165
G2 X133.183 Y105.35 I.596 J-4.815 E.03117
G1 X129.492 Y109.04 E.20159
G3 X128.574 Y110.797 I-29.052 J-14.074 E.07656
G1 X134.021 Y105.35 E.29751
G1 X134.859 Y105.35 E.03236
G1 X127.472 Y112.737 E.40348
G1 X126.365 Y114.52 E.08107
G1 X126.045 Y115.001 E.02231
G1 X135.697 Y105.35 E.52717
G1 X136.535 Y105.35 E.03236
G1 X123.958 Y117.926 E.6869
G1 X120.88 Y121.614 E.18551
G3 X115.651 Y126.639 I-41.297 J-37.733 E.2803
G1 X115.898 Y126.824 E.01189
G1 X137.372 Y105.35 E1.17288
G1 X138.21 Y105.35 E.03236
G1 X116.378 Y127.182 E1.19243
G1 X116.858 Y127.54 E.02312
G1 X139.048 Y105.35 E1.21199
G1 X139.886 Y105.35 E.03236
G1 X117.338 Y127.898 E1.23154
G1 X117.818 Y128.256 E.02312
G1 X140.724 Y105.35 E1.25109
G1 X141.562 Y105.35 E.03236
G1 X118.298 Y128.613 E1.27064
G3 X118.755 Y128.994 I-1.223 J1.934 E.02304
G1 X141.945 Y105.804 E1.26664
G1 X141.945 Y106.641 E.03236
G1 X119.179 Y129.407 E1.24344
G3 X119.578 Y129.846 I-1.308 J1.589 E.02299
G1 X141.945 Y107.479 E1.22165
G1 X141.945 Y108.317 E.03236
G1 X119.942 Y130.32 E1.20178
G3 X120.262 Y130.838 I-3.518 J2.531 E.02353
G1 X141.945 Y109.155 E1.1843
G1 X141.945 Y109.993 E.03236
G1 X120.555 Y131.383 E1.16833
G3 X120.795 Y131.981 I-3.785 J1.87 E.0249
G1 X141.945 Y110.831 E1.1552
G1 X141.945 Y111.668 E.03236
G1 X120.999 Y132.614 E1.14404
G1 X121.153 Y133.299 E.02708
G1 X141.945 Y112.506 E1.13565
G1 X141.945 Y113.344 E.03236
G1 X121.237 Y134.053 E1.13108
G3 X121.252 Y134.875 I-5.697 J.518 E.0318
G1 X141.945 Y114.182 E1.13024
G1 X141.945 Y115.02 E.03236
G1 X121.142 Y135.824 E1.13628
G1 X121.11 Y135.984 E.00629
G1 X121.59 Y136.213 E.02053
G1 X141.945 Y115.858 E1.11177
G1 X141.945 Y116.695 E.03236
G1 X122.157 Y136.483 E1.08079
G1 X122.724 Y136.754 E.02427
G1 X141.945 Y117.533 E1.04982
G1 X141.945 Y118.371 E.03236
G1 X123.281 Y137.035 E1.0194
G1 X123.832 Y137.323 E.02398
M73 P93 R1
G1 X141.945 Y119.209 E.98935
G1 X141.945 Y120.047 E.03236
G1 X124.382 Y137.61 E.95929
G3 X124.919 Y137.911 I-4.629 J8.889 E.02378
G1 X141.945 Y120.885 E.92997
G1 X141.945 Y121.722 E.03236
G1 X125.455 Y138.213 E.90068
G3 X125.988 Y138.518 I-4.689 J8.798 E.02371
G1 X141.945 Y122.56 E.87159
G1 X141.945 Y123.398 E.03236
G1 X126.51 Y138.833 E.84305
G1 X127.033 Y139.149 E.02357
G1 X141.945 Y124.236 E.81451
G1 X141.945 Y125.074 E.03236
G1 X127.547 Y139.472 E.78643
G1 X128.054 Y139.803 E.02339
G1 X141.945 Y125.912 E.75871
G1 X141.945 Y126.749 E.03236
G1 X128.562 Y140.133 E.73098
G3 X129.066 Y140.467 I-5.142 J8.291 E.02335
G1 X141.945 Y127.587 E.70348
G1 X141.945 Y128.425 E.03236
G1 X129.56 Y140.811 E.67649
G3 X130.053 Y141.155 I-5.354 J8.201 E.02324
G1 X141.945 Y129.263 E.64954
G1 X141.945 Y130.101 E.03236
G1 X130.534 Y141.512 E.62327
G1 X131.015 Y141.869 E.02313
G1 X141.945 Y130.939 E.597
G1 X141.945 Y131.776 E.03236
G1 X131.496 Y142.226 E.57073
G3 X131.967 Y142.592 I-6.398 J8.715 E.02306
G1 X141.945 Y132.614 E.54499
G1 X141.945 Y133.452 E.03236
G1 X132.437 Y142.961 E.51936
G1 X132.906 Y143.33 E.02304
G1 X141.945 Y134.29 E.49374
G1 X141.945 Y135.128 E.03236
G1 X133.367 Y143.707 E.46856
G1 X133.826 Y144.085 E.02299
G1 X141.945 Y135.966 E.44347
G1 X141.945 Y136.803 E.03236
G1 X134.276 Y144.473 E.41887
G1 X134.727 Y144.86 E.02294
G1 X141.945 Y137.641 E.39428
G1 X141.945 Y138.479 E.03236
G1 X135.169 Y145.256 E.37012
G1 X135.61 Y145.652 E.02291
G1 X141.945 Y139.317 E.34603
G1 X141.945 Y140.155 E.03236
G1 X136.042 Y146.058 E.32245
G1 X136.474 Y146.465 E.02289
G1 X141.945 Y140.993 E.29887
G1 X141.945 Y141.83 E.03236
G1 X136.898 Y146.878 E.2757
G1 X137.319 Y147.294 E.02288
G1 X141.945 Y142.668 E.25267
G1 X141.945 Y143.506 E.03236
G1 X137.736 Y147.716 E.22994
G1 X138.148 Y148.141 E.02288
G1 X141.945 Y144.344 E.20739
G1 X141.945 Y145.182 E.03236
G1 X138.553 Y148.574 E.18527
G1 X138.957 Y149.008 E.0229
G1 X141.945 Y146.02 E.16323
G1 X141.945 Y146.857 E.03236
G1 X139.354 Y149.449 E.14153
G1 X139.747 Y149.893 E.02292
G1 X141.945 Y147.695 E.12006
G1 X141.945 Y148.533 E.03236
G1 X140.138 Y150.341 E.09875
G1 X140.52 Y150.796 E.02297
G1 X141.945 Y149.371 E.07784
G1 X141.945 Y150.209 E.03236
G1 X140.902 Y151.253 E.05701
G1 X141.275 Y151.717 E.02301
G1 X141.945 Y151.047 E.0366
G1 X141.945 Y151.885 E.03236
G1 X141.458 Y152.372 E.02663
; CHANGE_LAYER
; Z_HEIGHT: 11.08
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13292.485
G1 X141.945 Y151.885 E-.26199
G1 X141.945 Y151.574 E-.11801
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L69
M991 S0 P68 ;notify layer change


G17
G3 Z11.32 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.178 Y103.973
G1 Z11.08
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.132 I-51.311 J-19.652 E.4815
G3 X123.072 Y118.225 I-29.871 J-19.715 E.14698
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-40.853 J-37.335 E.29802
G1 X118.015 Y129.012 E.15055
G1 X118.495 Y129.412 E.02385
G1 X119.088 Y130.031 E.03274
G3 X120.104 Y131.586 I-6.081 J5.082 E.07108
G1 X120.451 Y132.481 E.03667
G1 X120.681 Y133.47 E.03877
G1 X120.765 Y134.489 E.03901
G1 X120.698 Y135.504 E.03882
G1 X120.538 Y136.268 E.02981
G3 X130.508 Y142.102 I-22.591 J50.042 E.44183
G3 X142.443 Y154.069 I-32.72 J44.568 E.64783
G1 X142.443 Y104.746 E1.8831
G1 X141.82 Y104.854 E.02416
G1 X132.771 Y104.854 E.34546
G3 X132.1 Y104.735 I0 J-1.95 E.02615
G1 X131.668 Y104.512 E.01855
G1 X131.238 Y104.039 E.02439
G1 X131.537 Y103.45 F36000
; LINE_WIDTH: 0.712931
G1 F10791.968
G1 X131.495 Y103.333 E.00549
; LINE_WIDTH: 0.759399
G1 F10396.31
G1 X131.454 Y103.216 E.00586
; LINE_WIDTH: 0.805866
G1 F10008.04
G1 X131.412 Y103.099 E.00624
; LINE_WIDTH: 0.852334
G1 F9627.17
G1 X131.37 Y102.983 E.00662
; LINE_WIDTH: 0.898801
G1 F9109.681
G1 X131.329 Y102.866 E.00699
; LINE_WIDTH: 0.945269
G1 F8644.988
G1 X131.287 Y102.749 E.00737
; LINE_WIDTH: 0.991736
G1 F8225.403
G1 X131.245 Y102.632 E.00774
G1 X131.176 Y102.753 E.00869
; LINE_WIDTH: 0.945269
G1 F8644.988
G1 X131.107 Y102.874 E.00827
; LINE_WIDTH: 0.898801
G1 F9109.681
G1 X131.039 Y102.995 E.00785
; LINE_WIDTH: 0.852334
G1 F9627.17
G1 X130.97 Y103.116 E.00742
; LINE_WIDTH: 0.805866
G1 F10206.992
G1 X130.901 Y103.237 E.007
; LINE_WIDTH: 0.759399
G1 F10647.512
G1 X130.832 Y103.358 E.00658
; LINE_WIDTH: 0.712931
G1 F11097.338
G1 X130.763 Y103.479 E.00616
; LINE_WIDTH: 0.666464
G1 F11556.441
G1 X130.694 Y103.6 E.00574
; LINE_WIDTH: 0.619996
G1 F12927.158
G1 X130.546 Y103.971 E.01527
G1 F13446.369
G1 X130.397 Y104.343 E.01527
G1 X130.072 Y105.164 E.03373
G3 X124.88 Y114.797 I-50.263 J-20.877 E.41852
G3 X122.623 Y117.848 I-29.315 J-19.327 E.14497
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-42.076 J-38.754 E.32252
G1 X117.666 Y129.482 E.17936
G1 X118.117 Y129.859 E.02244
G1 X118.658 Y130.429 E.03001
G1 X119.252 Y131.243 E.03847
G1 X119.584 Y131.855 E.02656
G1 X119.902 Y132.685 E.03394
G1 X120.108 Y133.591 E.03551
G1 X120.18 Y134.525 E.03573
G1 X120.115 Y135.453 E.03554
G1 X119.932 Y136.288 E.03263
G1 X119.836 Y136.599 E.01243
G3 X130.932 Y143.157 I-21.125 J48.415 E.49335
G3 X142.92 Y155.769 I-32.375 J42.776 E.66728
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.481 E1.99553
G1 X142.717 Y103.931 E.0209
G1 X142.24 Y104.202 E.02094
G1 X141.82 Y104.268 E.01626
G1 X132.771 Y104.268 E.34546
G1 X132.302 Y104.185 E.0182
G1 X132 Y104.029 E.01298
G1 X131.848 Y103.863 E.00859
G1 X131.579 Y103.567 E.01527
; LINE_WIDTH: 0.666464
G1 F12457.931
G1 X131.567 Y103.535 E.0014
G1 X132.069 Y103.18 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.005 Y102.767 E.01594
G1 X132.578 Y99.67 E.12026
G3 X131.45 Y99.497 I.249 J-5.385 E.04367
G3 X124.399 Y114.463 I-51.524 J-15.13 E.63415
G3 X122.174 Y117.472 I-28.755 J-18.935 E.14296
G1 X119.604 Y120.538 E.15272
G3 X112.93 Y126.698 I-39.747 J-36.37 E.34719
G1 X117.317 Y129.952 E.20856
G3 X118.229 Y130.827 I-3.27 J4.322 E.04834
G1 X118.769 Y131.574 E.03521
G1 X119.063 Y132.123 E.02379
G1 X119.352 Y132.888 E.03122
G1 X119.535 Y133.713 E.03224
G1 X119.595 Y134.56 E.03245
G1 X119.531 Y135.403 E.03226
G1 X119.362 Y136.153 E.02935
G1 X119.08 Y136.91 E.03086
G3 X130.579 Y143.624 I-20.82 J48.864 E.5097
G3 X142.648 Y156.415 I-31.963 J42.246 E.67454
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y104.473 E1.97626
G1 X143.615 Y104.073 E.01527
; LINE_WIDTH: 0.664483
G1 F12473.611
G1 X143.592 Y103.838 E.00966
; LINE_WIDTH: 0.70897
G1 F11673.009
G1 X143.57 Y103.604 E.01034
; LINE_WIDTH: 0.753456
G1 F10950.885
G1 X143.548 Y103.37 E.01103
; LINE_WIDTH: 0.794426
G1 F10360.616
G1 X143.527 Y103.137 E.01159
; LINE_WIDTH: 0.835396
G1 F9830.726
G1 X143.507 Y102.904 E.01222
; LINE_WIDTH: 0.840506
G1 F9768.412
G3 X143.517 Y102.401 I4.236 J-.164 E.02647
; LINE_WIDTH: 0.814506
G1 F10093.955
G1 X143.542 Y101.901 E.02543
; LINE_WIDTH: 0.765879
G1 F10764.926
G1 X143.566 Y101.402 E.02384
; LINE_WIDTH: 0.717251
G1 F11531.45
G1 X143.59 Y100.903 E.02226
; LINE_WIDTH: 0.668624
G1 F12415.507
G1 X143.615 Y100.403 E.02067
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X143.615 Y100.003 E.01527
G1 X143.615 Y99.565 E.01672
G1 X142.924 Y99.672 E.02669
G1 X142.014 Y99.68 E.03473
G1 X142.173 Y100.535 E.03323
; LINE_WIDTH: 0.645986
G1 F12848.398
G1 X142.234 Y100.796 E.01066
; LINE_WIDTH: 0.694616
G1 F11926.759
G1 X142.349 Y101.282 E.02152
; LINE_WIDTH: 0.743246
G1 F11108.604
G1 X142.464 Y101.769 E.02311
; LINE_WIDTH: 0.791876
G1 F10395.492
G1 X142.579 Y102.256 E.02469
; LINE_WIDTH: 0.840506
G1 F9768.412
G1 X142.693 Y102.742 E.02628
; LINE_WIDTH: 0.845136
G1 F9712.63
G1 X142.697 Y103.058 E.01669
G1 X142.654 Y103.129 E.0044
; LINE_WIDTH: 0.799296
G1 F10294.657
G1 X142.611 Y103.2 E.00415
; LINE_WIDTH: 0.753456
G1 F10720.579
G1 X142.518 Y103.297 E.00629
; LINE_WIDTH: 0.70897
G1 F11155.136
G1 X142.425 Y103.394 E.0059
; LINE_WIDTH: 0.664483
G1 F11598.278
G1 X142.332 Y103.49 E.00551
; LINE_WIDTH: 0.619996
G1 F12666.13
G1 X142.06 Y103.645 E.01195
G1 F13446.369
G1 X141.82 Y103.682 E.00928
G1 X132.771 Y103.682 E.34546
G3 X132.331 Y103.546 I0 J-.778 E.01787
G1 X132.084 Y103.268 E.01419
M204 S250
G1 X132.572 Y103.009 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X132.549 Y102.863 E.00469
G1 X133.241 Y99.13 E.12021
G3 X131.521 Y98.947 I-.175 J-6.543 E.05492
G1 X131.038 Y98.936 E.01529
G3 X123.945 Y114.147 I-51.152 J-14.591 E.5336
G3 X121.747 Y117.121 I-28.177 J-18.53 E.11715
G1 X119.177 Y120.187 E.12664
G3 X112.01 Y126.704 I-40.297 J-37.117 E.30711
G1 X116.988 Y130.396 E.19623
G3 X117.823 Y131.203 I-2.915 J3.857 E.03685
G1 X118.313 Y131.887 E.02664
G1 X118.572 Y132.377 E.01755
G1 X118.834 Y133.08 E.02375
G1 X118.994 Y133.827 E.02419
G1 X119.044 Y134.594 E.02434
G1 X118.98 Y135.355 E.02418
G1 X118.824 Y136.025 E.02177
G1 X118.548 Y136.753 E.02467
G1 X118.31 Y137.19 E.01573
G3 X130.249 Y144.067 I-19.584 J47.798 E.43752
G3 X142.39 Y157.025 I-31.645 J41.817 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.196 E1.82043
G1 X144.167 Y98.939 E.00814
G1 X143.86 Y98.947 E.00974
G3 X142.832 Y99.124 I-1.457 J-5.377 E.03307
G1 X141.35 Y99.129 E.04692
G1 X142.042 Y102.865 E.1203
G1 X142.031 Y102.983 E.00376
G1 X141.889 Y103.119 E.00621
G1 X132.771 Y103.13 E.28868
G3 X132.629 Y103.079 I0 J-.226 E.00486
; WIPE_START
M204 S10000
G1 X132.549 Y102.863 E-.08768
G1 X132.689 Y102.107 E-.29232
; WIPE_END
G1 E-.02 F1800
G1 X140.207 Y100.789 Z11.48 F36000
G1 X142.942 Y100.309 Z11.48
G1 Z11.08
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.902676
G1 F9069.029
G1 X142.833 Y100.309 E.00614
G1 X142.779 Y100.403 E.00614
G1 X142.833 Y100.497 E.00614
G1 X142.942 Y100.497 E.00614
G1 X142.996 Y100.403 E.00614
; WIPE_START
G1 X142.942 Y100.497 E-.076
G1 X142.833 Y100.497 E-.076
G1 X142.779 Y100.403 E-.076
G1 X142.833 Y100.309 E-.076
G1 X142.942 Y100.309 E-.076
; WIPE_END
G1 E-.02 F1800
G1 X135.456 Y101.796 Z11.48 F36000
G1 X131.245 Y102.632 Z11.48
G1 Z11.08
G1 E.4 F1800
; LINE_WIDTH: 0.991736
G1 F8225.403
G1 X131.306 Y102.41 E.01437
; LINE_WIDTH: 0.953616
G1 F8566.488
G1 X131.39 Y102.086 E.02005
; LINE_WIDTH: 0.90509
G1 F9043.894
G1 X131.475 Y101.762 E.019
; LINE_WIDTH: 0.856563
G1 F9577.652
G1 X131.56 Y101.439 E.01794
; LINE_WIDTH: 0.808036
G1 F10178.364
G1 X131.637 Y101.129 E.0161
; LINE_WIDTH: 0.768476
G1 F10726.838
G1 X131.715 Y100.819 E.01528
; LINE_WIDTH: 0.728916
G1 F11337.788
G1 X131.792 Y100.509 E.01446
; LINE_WIDTH: 0.689356
G1 F12022.534
G1 X131.87 Y100.199 E.01363
; WIPE_START
G1 X131.792 Y100.509 E-.12134
G1 X131.715 Y100.819 E-.12134
G1 X131.637 Y101.129 E-.12134
G1 X131.627 Y101.17 E-.01599
; WIPE_END
G1 E-.02 F1800
G1 X138.516 Y104.454 Z11.48 F36000
G1 X142.209 Y106.215 Z11.48
G1 Z11.08
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.624906
G1 F13334.576
G1 X141.345 Y105.352 E.04701
G1 X140.51 Y105.352 E.03215
G1 X141.945 Y106.787 E.07814
G1 X141.945 Y107.622 E.03215
G1 X139.675 Y105.352 E.12361
G1 X138.84 Y105.352 E.03215
G1 X141.945 Y108.457 E.16908
G1 X141.945 Y109.292 E.03215
G1 X138.005 Y105.352 E.21456
G1 X137.17 Y105.352 E.03215
G1 X141.945 Y110.128 E.26003
G1 X141.945 Y110.963 E.03215
G1 X136.334 Y105.352 E.3055
G1 X135.499 Y105.352 E.03215
G1 X141.945 Y111.798 E.35098
G1 X141.945 Y112.633 E.03215
G1 X134.664 Y105.352 E.39645
G1 X133.829 Y105.352 E.03215
G1 X141.945 Y113.468 E.44192
G1 X141.945 Y114.303 E.03215
G1 X132.994 Y105.352 E.48739
G3 X132.039 Y105.232 I-.095 J-3.116 E.03719
G1 X141.945 Y115.139 E.53937
G1 X141.945 Y115.974 E.03215
G1 X131.207 Y105.236 E.58466
G1 X130.964 Y105.828 E.02465
G1 X141.945 Y116.809 E.59787
G1 X141.945 Y117.644 E.03215
G1 X130.717 Y106.416 E.61132
G1 X130.462 Y106.996 E.02439
G1 X141.945 Y118.479 E.62522
G1 X141.945 Y119.315 E.03215
G1 X130.199 Y107.568 E.63954
G1 X129.933 Y108.138 E.02419
G1 X141.945 Y120.15 E.654
G1 X141.945 Y120.985 E.03215
G1 X129.661 Y108.701 E.66882
G1 X129.384 Y109.259 E.02399
G1 X141.945 Y121.82 E.6839
G1 X141.945 Y122.655 E.03215
M73 P94 R1
G1 X129.103 Y109.813 E.69924
G1 X128.816 Y110.361 E.02383
G1 X141.945 Y123.491 E.71486
G1 X141.945 Y124.326 E.03215
G1 X128.518 Y110.899 E.73106
G1 X128.217 Y111.433 E.02361
G1 X141.945 Y125.161 E.74744
G1 X141.945 Y125.996 E.03215
G1 X127.917 Y111.967 E.76381
G1 X127.616 Y112.502 E.02361
G1 X141.945 Y126.831 E.78019
G1 X141.945 Y127.667 E.03215
G1 X127.3 Y113.022 E.79737
G1 X126.985 Y113.541 E.0234
G1 X141.945 Y128.502 E.81455
G1 X141.945 Y129.337 E.03215
G1 X126.662 Y114.053 E.83213
G1 X126.337 Y114.563 E.02329
G1 X141.945 Y130.172 E.84984
G1 X141.945 Y131.007 E.03215
G1 X126.003 Y115.065 E.868
G3 X125.665 Y115.563 I-8.412 J-5.341 E.02315
G1 X141.945 Y131.843 E.88638
G1 X141.945 Y132.678 E.03215
G1 X125.319 Y116.051 E.90525
G1 X124.972 Y116.54 E.02306
G1 X141.945 Y133.513 E.92412
G1 X141.945 Y134.348 E.03215
G1 X124.626 Y117.028 E.94299
G3 X124.273 Y117.511 I-7.676 J-5.237 E.02301
G1 X141.945 Y135.183 E.96219
G1 X141.945 Y136.018 E.03215
G1 X123.913 Y117.986 E.98181
G3 X123.537 Y118.445 I-4.925 J-3.652 E.02286
G1 X141.945 Y136.854 E1.00229
G1 X141.945 Y137.689 E.03215
G1 X123.156 Y118.899 E1.02302
G1 X122.775 Y119.354 E.02282
G1 X141.945 Y138.524 E1.04376
G1 X141.945 Y139.359 E.03215
G1 X122.394 Y119.808 E1.0645
G1 X122.013 Y120.262 E.02282
G1 X141.945 Y140.194 E1.08523
G1 X141.945 Y141.03 E.03215
G1 X121.632 Y120.717 E1.10597
G1 X121.252 Y121.171 E.02282
G1 X141.945 Y141.865 E1.1267
G1 X141.945 Y142.7 E.03215
G1 X120.87 Y121.625 E1.14747
G1 X120.469 Y122.059 E.02276
G1 X141.945 Y143.535 E1.16928
G1 X141.945 Y144.37 E.03215
G1 X120.063 Y122.488 E1.19144
G1 X119.65 Y122.91 E.02274
G1 X141.945 Y145.206 E1.2139
G1 X141.945 Y146.041 E.03215
G1 X119.237 Y123.333 E1.23637
G3 X118.823 Y123.754 I-7.077 J-6.551 E.02274
G1 X141.945 Y146.876 E1.25893
G1 X141.945 Y147.711 E.03215
G1 X118.398 Y124.163 E1.28209
G1 X117.972 Y124.573 E.02274
G1 X141.945 Y148.546 E1.30525
G1 X141.945 Y149.382 E.03215
G1 X117.537 Y124.973 E1.32897
G1 X117.098 Y125.37 E.02276
G1 X141.945 Y150.217 E1.35283
G1 X141.945 Y151.052 E.03215
G1 X116.66 Y125.767 E1.37669
G3 X116.216 Y126.158 I-6.639 J-7.075 E.02278
G1 X141.945 Y151.887 E1.40084
G1 X141.945 Y152.574 E.02643
G2 X131.309 Y142.086 I-43.362 J33.34 E.57687
G1 X120.497 Y131.274 E.58869
G1 X120.616 Y131.504 E.00997
G3 X121.01 Y132.622 I-7.559 J3.291 E.04569
G1 X128.444 Y140.056 E.40472
G2 X126.197 Y138.645 I-22.034 J32.582 E.10214
G1 X121.2 Y133.647 E.27209
G1 X121.261 Y134.544 E.03459
G1 X124.27 Y137.552 E.16382
G1 X122.545 Y136.663 E.07471
G1 X120.969 Y135.087 E.0858
; WIPE_START
G1 X121.676 Y135.794 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X118.044 Y129.081 Z11.48 F36000
G1 X117.737 Y128.514 Z11.48
G1 Z11.08
G1 E.4 F1800
G1 F13334.576
G1 X115.578 Y126.355 E.11751
; CHANGE_LAYER
; Z_HEIGHT: 11.24
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13334.576
G1 X116.286 Y127.062 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L70
M991 S0 P69 ;notify layer change


G17
G3 Z11.48 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.176 Y103.976
G1 Z11.24
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.132 I-51.318 J-19.659 E.48138
G3 X123.072 Y118.225 I-29.864 J-19.71 E.14698
G1 X120.502 Y121.29 E.15272
G3 X114.848 Y126.662 I-40.978 J-37.467 E.29801
G1 X118.015 Y129.012 E.15054
G3 X118.933 Y129.85 I-4.607 J5.969 E.04751
G3 X120.074 Y131.528 I-5.893 J5.235 E.07773
G1 X120.453 Y132.488 E.03939
G1 X120.682 Y133.474 E.03863
G1 X120.765 Y134.489 E.0389
G1 X120.698 Y135.507 E.03892
G1 X120.538 Y136.268 E.02971
G3 X130.508 Y142.102 I-22.585 J50.036 E.44183
G3 X142.443 Y154.069 I-32.721 J44.568 E.64783
G1 X142.443 Y104.75 E1.88297
G1 X141.822 Y104.856 E.02406
G1 X132.769 Y104.856 E.34565
G3 X132.095 Y104.736 I0 J-1.95 E.02626
G1 X131.662 Y104.511 E.01863
G1 X131.237 Y104.043 E.02415
G1 X131.534 Y103.451 F36000
; LINE_WIDTH: 0.710456
G1 F10867.233
G1 X131.493 Y103.334 E.00546
; LINE_WIDTH: 0.755686
G1 F10470.965
G1 X131.451 Y103.218 E.00582
; LINE_WIDTH: 0.800916
G1 F10082.096
G1 X131.41 Y103.101 E.00619
; LINE_WIDTH: 0.846146
G1 F9700.546
G1 X131.368 Y102.984 E.00655
; LINE_WIDTH: 0.891376
G1 F9188.604
G1 X131.327 Y102.868 E.00692
; LINE_WIDTH: 0.936606
G1 F8727.987
G1 X131.285 Y102.751 E.00728
; LINE_WIDTH: 0.981836
G1 F8311.346
G1 X131.244 Y102.634 E.00765
G1 X131.175 Y102.755 E.00858
; LINE_WIDTH: 0.936606
G1 F8727.987
G1 X131.106 Y102.876 E.00817
; LINE_WIDTH: 0.891376
G1 F9188.604
G1 X131.037 Y102.996 E.00776
; LINE_WIDTH: 0.846146
G1 F9700.546
G1 X130.969 Y103.117 E.00735
; LINE_WIDTH: 0.800916
G1 F10272.901
G1 X130.9 Y103.238 E.00694
; LINE_WIDTH: 0.755686
G1 F10713.712
G1 X130.831 Y103.359 E.00653
; LINE_WIDTH: 0.710456
G1 F11163.783
G1 X130.763 Y103.479 E.00612
; LINE_WIDTH: 0.665226
G1 F11623.085
G1 X130.694 Y103.6 E.00571
; LINE_WIDTH: 0.619996
G1 F12997.638
G1 X130.549 Y103.973 E.01527
G1 F13446.369
G1 X130.404 Y104.345 E.01527
G1 X130.327 Y104.54 E.00799
G3 X124.88 Y114.797 I-50.525 J-20.257 E.44426
G3 X122.623 Y117.849 I-29.307 J-19.321 E.14497
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-40.117 J-36.663 E.32255
G1 X117.666 Y129.482 E.17935
G3 X118.524 Y130.269 I-5.346 J6.69 E.04447
G3 X119.208 Y131.168 I-7.833 J6.666 E.04319
G1 X119.561 Y131.811 E.02799
G1 X119.904 Y132.692 E.03607
G1 X120.109 Y133.594 E.03535
G1 X120.18 Y134.525 E.03563
G1 X120.114 Y135.456 E.03564
G1 X119.932 Y136.284 E.03236
G1 X119.835 Y136.599 E.01258
G3 X130.932 Y143.157 I-21.126 J48.415 E.49336
G3 X142.92 Y155.769 I-32.375 J42.777 E.66727
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.487 E1.99531
G1 X142.72 Y103.933 E.02073
G1 X142.243 Y104.204 E.02094
G1 X141.822 Y104.27 E.01626
G1 X132.769 Y104.27 E.34565
G1 X132.297 Y104.186 E.01828
G1 X131.994 Y104.029 E.01304
G1 X131.844 Y103.864 E.00852
G1 X131.576 Y103.567 E.01527
; LINE_WIDTH: 0.665226
G1 F12482.367
G1 X131.564 Y103.536 E.00139
G1 X132.066 Y103.181 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132.002 Y102.769 E.0159
G1 X132.576 Y99.672 E.12027
G3 X131.45 Y99.497 I.25 J-5.334 E.04359
G3 X124.399 Y114.463 I-51.402 J-15.073 E.63415
G3 X122.174 Y117.472 I-28.75 J-18.932 E.14296
G1 X119.604 Y120.538 E.15272
G3 X112.93 Y126.698 I-39.744 J-36.368 E.34718
G1 X117.317 Y129.952 E.20856
G3 X118.151 Y130.726 I-6.486 J7.831 E.04345
G3 X118.695 Y131.451 I-12.503 J9.944 E.03461
G1 X119.048 Y132.093 E.02799
G1 X119.354 Y132.895 E.03276
G1 X119.536 Y133.715 E.03207
G1 X119.596 Y134.561 E.03235
G1 X119.531 Y135.406 E.03236
G1 X119.363 Y136.148 E.02908
G1 X119.08 Y136.91 E.03102
G3 X130.579 Y143.624 I-20.82 J48.864 E.50971
G3 X142.648 Y156.415 I-31.963 J42.246 E.67453
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y104.474 E1.97622
G1 X143.615 Y104.074 E.01527
; LINE_WIDTH: 0.664263
G1 F12480.654
G1 X143.592 Y103.84 E.00965
; LINE_WIDTH: 0.70853
G1 F11680.626
G1 X143.57 Y103.606 E.01033
; LINE_WIDTH: 0.752796
G1 F10960.946
G1 X143.548 Y103.372 E.011
; LINE_WIDTH: 0.793466
G1 F10373.718
G1 X143.528 Y103.139 E.01157
; LINE_WIDTH: 0.834136
G1 F9846.213
G1 X143.507 Y102.906 E.01219
; LINE_WIDTH: 0.839266
G1 F9783.461
G3 X143.518 Y102.402 I4.264 J-.164 E.02649
; LINE_WIDTH: 0.813286
G1 F10109.764
G1 X143.542 Y101.902 E.02539
; LINE_WIDTH: 0.764964
G1 F10778.407
G1 X143.566 Y101.403 E.02381
; LINE_WIDTH: 0.716641
G1 F11541.76
G1 X143.59 Y100.903 E.02224
; LINE_WIDTH: 0.668319
G1 F12421.48
G1 X143.615 Y100.404 E.02066
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X143.615 Y100.004 E.01527
G1 X143.615 Y99.566 E.01672
G1 X142.923 Y99.674 E.02671
G1 X142.017 Y99.682 E.03461
G1 X142.175 Y100.536 E.03317
; LINE_WIDTH: 0.645966
G1 F12859.628
G1 X142.237 Y100.798 E.01072
; LINE_WIDTH: 0.694291
G1 F11932.633
G1 X142.351 Y101.284 E.02151
; LINE_WIDTH: 0.742616
G1 F11118.485
G1 X142.466 Y101.771 E.02309
; LINE_WIDTH: 0.790941
G1 F10408.338
G1 X142.581 Y102.258 E.02466
; LINE_WIDTH: 0.839266
G1 F9783.461
G1 X142.695 Y102.744 E.02624
; LINE_WIDTH: 0.843866
G1 F9727.868
G1 X142.699 Y103.06 E.01666
G1 X142.656 Y103.131 E.00439
; LINE_WIDTH: 0.798331
G1 F10307.66
G1 X142.613 Y103.202 E.00414
; LINE_WIDTH: 0.752796
G1 F10733.696
G1 X142.52 Y103.299 E.00628
; LINE_WIDTH: 0.70853
G1 F11168.36
G1 X142.427 Y103.396 E.00589
; LINE_WIDTH: 0.664263
G1 F11611.651
G1 X142.334 Y103.492 E.00551
; LINE_WIDTH: 0.619996
G1 F12680.136
G1 X142.062 Y103.647 E.01195
G1 F13446.369
G1 X141.822 Y103.685 E.00928
G1 X132.769 Y103.685 E.34565
G1 X132.5 Y103.637 E.01043
G1 X132.327 Y103.547 E.00744
G1 X132.081 Y103.27 E.01415
M204 S250
G1 X132.569 Y103.011 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X132.547 Y102.865 E.00468
G1 X133.238 Y99.132 E.12021
G3 X131.52 Y98.947 I-.173 J-6.481 E.05487
G1 X131.038 Y98.936 E.01528
G3 X123.945 Y114.147 I-51.028 J-14.533 E.53361
G3 X121.747 Y117.121 I-28.167 J-18.524 E.11715
G1 X119.177 Y120.187 E.12664
G3 X112.01 Y126.704 I-40.298 J-37.118 E.3071
G1 X116.988 Y130.396 E.19622
G3 X117.739 Y131.094 I-6.391 J7.633 E.03248
G3 X118.21 Y131.717 I-29.624 J22.912 E.02475
G1 X118.564 Y132.36 E.02321
G1 X118.836 Y133.087 E.02458
G1 X118.995 Y133.829 E.02403
G1 X119.044 Y134.594 E.02427
G1 X118.98 Y135.358 E.02426
G1 X118.825 Y136.021 E.02155
G1 X118.548 Y136.753 E.0248
G1 X118.31 Y137.19 E.01574
G3 X130.249 Y144.067 I-19.959 J48.449 E.43749
G3 X142.39 Y157.025 I-31.645 J41.817 E.56491
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.195 E1.82045
G1 X144.167 Y98.939 E.00813
G1 X143.86 Y98.947 E.00974
G3 X142.831 Y99.126 I-1.459 J-5.337 E.03312
M73 P94 R0
G1 X141.353 Y99.131 E.0468
G1 X142.044 Y102.867 E.1203
M73 P95 R0
G1 X142.033 Y102.985 E.00376
G1 X141.892 Y103.121 E.00621
G1 X132.769 Y103.132 E.28884
G3 X132.626 Y103.081 I0 J-.226 E.00487
; WIPE_START
M204 S10000
G1 X132.547 Y102.865 E-.0876
G1 X132.687 Y102.108 E-.2924
; WIPE_END
G1 E-.02 F1800
G1 X140.205 Y100.79 Z11.64 F36000
G1 X142.943 Y100.31 Z11.64
G1 Z11.24
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.900416
G1 F9092.694
G1 X142.835 Y100.31 E.00611
G1 X142.78 Y100.404 E.00611
G1 X142.835 Y100.498 E.00611
G1 X142.943 Y100.498 E.00611
G1 X142.997 Y100.404 E.00611
; WIPE_START
G1 X142.943 Y100.498 E-.076
G1 X142.835 Y100.498 E-.076
G1 X142.78 Y100.404 E-.076
G1 X142.835 Y100.31 E-.076
G1 X142.943 Y100.31 E-.076
; WIPE_END
G1 E-.02 F1800
G1 X135.455 Y101.791 Z11.64 F36000
G1 X131.246 Y102.624 Z11.64
G1 Z11.24
G1 E.4 F1800
; LINE_WIDTH: 0.988256
G1 F8255.41
G1 X131.305 Y102.41 E.01382
; LINE_WIDTH: 0.951556
G1 F8585.727
G1 X131.389 Y102.086 E.02001
; LINE_WIDTH: 0.903023
G1 F9065.41
G1 X131.474 Y101.762 E.01896
; LINE_WIDTH: 0.85449
G1 F9601.863
G1 X131.559 Y101.438 E.0179
; LINE_WIDTH: 0.805956
G1 F10205.802
G1 X131.636 Y101.129 E.01605
; LINE_WIDTH: 0.766416
G1 F10757.022
G1 X131.714 Y100.819 E.01523
; LINE_WIDTH: 0.726876
G1 F11371.185
G1 X131.791 Y100.509 E.01441
; LINE_WIDTH: 0.687336
G1 F12059.725
G1 X131.868 Y100.2 E.01359
; WIPE_START
G1 X131.791 Y100.509 E-.12128
G1 X131.714 Y100.819 E-.12128
G1 X131.636 Y101.129 E-.12128
G1 X131.626 Y101.17 E-.01617
; WIPE_END
G1 E-.02 F1800
G1 X131.944 Y104.919 Z11.64 F36000
G1 Z11.24
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.626716
G1 F13293.832
G1 X130.916 Y105.947 E.05614
G3 X130.263 Y107.437 I-74.876 J-31.894 E.06282
G1 X132.379 Y105.322 E.11552
G2 X133.184 Y105.354 I.594 J-4.788 E.03118
G1 X129.472 Y109.066 E.20272
G3 X128.578 Y110.797 I-26.449 J-12.558 E.07528
G1 X134.022 Y105.354 E.2973
G1 X134.86 Y105.354 E.03235
G1 X127.47 Y112.743 E.40356
G3 X126.036 Y115.016 I-53.431 J-32.144 E.10379
G1 X135.698 Y105.354 E.52766
G1 X136.535 Y105.354 E.03235
G1 X123.943 Y117.947 E.68773
G1 X120.88 Y121.614 E.18449
G3 X115.651 Y126.639 I-41.29 J-37.726 E.28027
G1 X115.901 Y126.826 E.01203
G1 X137.373 Y105.354 E1.17265
G1 X138.211 Y105.354 E.03235
G1 X116.381 Y127.184 E1.1922
G1 X116.861 Y127.542 E.02312
G1 X139.049 Y105.354 E1.21175
G1 X139.886 Y105.354 E.03235
G1 X117.341 Y127.9 E1.23129
G1 X117.82 Y128.257 E.02312
G1 X140.724 Y105.354 E1.25084
G1 X141.562 Y105.354 E.03235
G1 X118.3 Y128.615 E1.27039
G3 X118.757 Y128.996 I-1.225 J1.932 E.02303
G1 X141.945 Y105.808 E1.26639
G1 X141.945 Y106.646 E.03235
G1 X119.182 Y129.409 E1.2432
G3 X119.58 Y129.849 I-1.309 J1.589 E.02298
G1 X141.945 Y107.483 E1.22143
G1 X141.945 Y108.321 E.03235
G1 X119.944 Y130.322 E1.20156
G3 X120.264 Y130.841 I-3.545 J2.544 E.02353
G1 X141.945 Y109.159 E1.18411
G1 X141.945 Y109.997 E.03235
G1 X120.556 Y131.386 E1.16814
G3 X120.796 Y131.984 I-3.771 J1.861 E.0249
G1 X141.945 Y110.834 E1.15504
G1 X141.945 Y111.672 E.03235
G1 X121 Y132.618 E1.1439
G1 X121.152 Y133.303 E.02712
G1 X141.945 Y112.51 E1.13559
G1 X141.945 Y113.348 E.03235
G1 X121.231 Y134.062 E1.13126
G3 X121.252 Y134.879 I-4.873 J.53 E.03161
G1 X141.945 Y114.185 E1.13015
G1 X141.945 Y115.023 E.03235
G1 X121.147 Y135.822 E1.13586
G1 X121.113 Y135.985 E.00646
G1 X121.594 Y136.213 E.02055
G1 X141.945 Y115.861 E1.11147
G1 X141.945 Y116.699 E.03235
G1 X122.162 Y136.482 E1.0804
G3 X122.729 Y136.753 I-3.721 J8.498 E.02426
G1 X141.945 Y117.536 E1.04947
G1 X141.945 Y118.374 E.03235
G1 X123.282 Y137.038 E1.01926
G1 X123.835 Y137.322 E.02402
G1 X141.945 Y119.212 E.98906
G1 X141.945 Y120.05 E.03235
G1 X124.381 Y137.614 E.95925
G1 X124.918 Y137.914 E.02378
G1 X141.945 Y120.887 E.9299
G1 X141.945 Y121.725 E.03235
G1 X125.456 Y138.215 E.90054
G3 X125.989 Y138.519 I-4.253 J8.074 E.02372
G1 X141.945 Y122.563 E.87141
G1 X141.945 Y123.401 E.03235
G1 X126.512 Y138.834 E.84288
G1 X127.034 Y139.15 E.02356
G1 X141.945 Y124.238 E.81434
G1 X141.945 Y125.076 E.03235
G1 X127.548 Y139.473 E.78627
G1 X128.056 Y139.804 E.02338
G1 X141.945 Y125.914 E.75855
G1 X141.945 Y126.752 E.03235
G1 X128.563 Y140.134 E.73083
G3 X129.066 Y140.468 I-4.723 J7.644 E.02334
G1 X141.945 Y127.589 E.70336
G1 X141.945 Y128.427 E.03235
G1 X129.559 Y140.813 E.67644
G1 X130.052 Y141.158 E.02323
G1 X141.945 Y129.265 E.64952
G1 X141.945 Y130.103 E.03235
G1 X130.538 Y141.51 E.62301
G1 X131.017 Y141.868 E.02312
G1 X141.945 Y130.94 E.59681
G1 X141.945 Y131.778 E.03235
G1 X131.497 Y142.226 E.57061
G3 X131.968 Y142.593 I-7.137 J9.657 E.02306
G1 X141.945 Y132.616 E.54488
G1 X141.945 Y133.454 E.03235
G1 X132.437 Y142.962 E.51926
G1 X132.907 Y143.33 E.02304
G1 X141.945 Y134.291 E.49364
G1 X141.945 Y135.129 E.03235
G1 X133.367 Y143.707 E.46848
G1 X133.827 Y144.086 E.02298
G1 X141.945 Y135.967 E.44339
G1 X141.945 Y136.805 E.03235
G1 X134.277 Y144.473 E.41879
G1 X134.727 Y144.861 E.02294
G1 X141.945 Y137.642 E.39421
G1 X141.945 Y138.48 E.03235
G1 X135.17 Y145.256 E.37005
G1 X135.611 Y145.653 E.02291
G1 X141.945 Y139.318 E.34597
G1 X141.945 Y140.156 E.03235
G1 X136.039 Y146.062 E.32254
G1 X136.468 Y146.471 E.02288
G1 X141.945 Y140.993 E.29912
G1 X141.945 Y141.831 E.03235
G1 X136.897 Y146.88 E.27571
G3 X137.319 Y147.296 I-5.979 J6.484 E.02288
G1 X141.945 Y142.669 E.25267
G1 X141.945 Y143.507 E.03235
G1 X137.734 Y147.719 E.23002
G1 X138.148 Y148.142 E.02288
G1 X141.945 Y144.344 E.20738
G1 X141.945 Y145.182 E.03235
G1 X138.553 Y148.574 E.18524
G1 X138.957 Y149.008 E.02289
G1 X141.945 Y146.02 E.1632
G1 X141.945 Y146.858 E.03235
G1 X139.354 Y149.449 E.14151
G1 X139.747 Y149.894 E.02292
G1 X141.945 Y147.695 E.12004
G1 X141.945 Y148.533 E.03235
G1 X140.137 Y150.341 E.09874
G1 X140.52 Y150.796 E.02296
G1 X141.945 Y149.371 E.07784
G1 X141.945 Y150.209 E.03235
G1 X140.902 Y151.252 E.05698
G1 X141.275 Y151.717 E.02301
G1 X141.945 Y151.046 E.0366
G1 X141.945 Y151.884 E.03235
G1 X141.458 Y152.372 E.02661
; CHANGE_LAYER
; Z_HEIGHT: 11.4
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13293.832
G1 X141.945 Y151.884 E-.26187
G1 X141.945 Y151.573 E-.11813
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L71
M991 S0 P70 ;notify layer change


G17
G3 Z11.64 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X131.17 Y103.992
G1 Z11.4
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X125.361 Y115.131 I-51.153 J-19.59 E.48068
G3 X123.071 Y118.225 I-29.866 J-19.711 E.14703
G1 X120.502 Y121.291 E.15272
G3 X114.848 Y126.662 I-42.479 J-39.048 E.29798
G1 X118.015 Y129.012 E.15055
G3 X118.932 Y129.849 I-4.612 J5.975 E.04747
G3 X120.075 Y131.529 I-5.863 J5.215 E.07778
G1 X120.451 Y132.484 E.03921
G3 X120.765 Y134.489 I-7.442 J2.189 E.0777
G1 X120.698 Y135.504 E.03885
G1 X120.538 Y136.268 E.0298
G3 X130.503 Y142.098 I-22.331 J49.601 E.44162
G3 X142.443 Y154.069 I-32.714 J44.57 E.64804
G1 X142.443 Y104.753 E1.88283
G1 X141.825 Y104.858 E.02396
G1 X132.766 Y104.858 E.34584
G1 X132.447 Y104.832 E.01222
G1 X131.899 Y104.655 E.02197
G1 X131.446 Y104.343 E.02103
G1 X131.225 Y104.063 E.01359
G1 X131.517 Y103.423 F36000
; LINE_WIDTH: 0.712101
G1 F10756.996
G1 X131.479 Y103.316 E.00501
; LINE_WIDTH: 0.758154
G1 F10395.455
G1 X131.44 Y103.202 E.00566
; LINE_WIDTH: 0.804206
G1 F10020.103
G1 X131.4 Y103.089 E.00602
; LINE_WIDTH: 0.850259
G1 F9651.652
G1 X131.361 Y102.976 E.00638
; LINE_WIDTH: 0.896311
G1 F9135.997
G1 X131.321 Y102.863 E.00674
; LINE_WIDTH: 0.942364
G1 F8672.646
G1 X131.281 Y102.75 E.0071
; LINE_WIDTH: 0.988416
G1 F8254.026
G1 X131.242 Y102.636 E.00746
G1 X131.173 Y102.757 E.00862
; LINE_WIDTH: 0.942364
G1 F8672.646
G1 X131.105 Y102.877 E.0082
; LINE_WIDTH: 0.896311
G1 F9135.997
G1 X131.036 Y102.998 E.00778
; LINE_WIDTH: 0.850259
G1 F9651.652
G1 X130.968 Y103.118 E.00737
; LINE_WIDTH: 0.804206
G1 F10229.001
G1 X130.9 Y103.239 E.00695
; LINE_WIDTH: 0.758154
G1 F10667.64
G1 X130.831 Y103.359 E.00654
; LINE_WIDTH: 0.712101
G1 F11115.517
G1 X130.763 Y103.479 E.00612
; LINE_WIDTH: 0.666049
G1 F11572.557
G1 X130.694 Y103.6 E.0057
; LINE_WIDTH: 0.619996
G1 F12944.203
G1 X130.548 Y103.972 E.01527
G1 F13446.369
G1 X130.402 Y104.344 E.01527
G1 X130.208 Y104.834 E.02012
G3 X124.88 Y114.797 I-50.607 J-20.656 E.43209
G3 X122.622 Y117.849 I-29.319 J-19.329 E.14502
G1 X120.053 Y120.914 E.15272
G3 X113.893 Y126.683 I-40.087 J-36.631 E.32253
G1 X117.666 Y129.482 E.17936
G3 X118.523 Y130.268 I-5.575 J6.94 E.04443
G3 X119.208 Y131.169 I-7.762 J6.614 E.04322
G1 X119.561 Y131.811 E.028
G1 X119.902 Y132.687 E.03591
G3 X120.135 Y133.79 I-9.557 J2.594 E.04304
G1 X120.18 Y134.525 E.0281
G1 X120.115 Y135.454 E.03556
G1 X119.933 Y136.283 E.0324
G1 X119.836 Y136.599 E.01263
G3 X130.936 Y143.16 I-21.039 J48.269 E.49355
G3 X142.92 Y155.769 I-32.177 J42.582 E.66711
G1 X143.029 Y155.749 E.00422
G1 X143.029 Y103.492 E1.99509
G1 X142.722 Y103.935 E.02056
G1 X142.245 Y104.206 E.02094
G1 X141.825 Y104.272 E.01626
G1 X132.766 Y104.272 E.34584
G1 X132.543 Y104.254 E.00855
G1 X132.134 Y104.117 E.01648
G1 X131.842 Y103.912 E.01361
G1 X131.558 Y103.542 E.0178
; LINE_WIDTH: 0.666049
G1 F12466.115
G1 X131.546 Y103.508 E.0015
G1 X132.06 Y103.162 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X132 Y102.772 E.0151
G1 X132.573 Y99.674 E.12028
G3 X131.45 Y99.498 I.251 J-5.28 E.04351
G3 X124.4 Y114.462 I-51.524 J-15.131 E.63409
G3 X122.174 Y117.473 I-28.755 J-18.935 E.14301
G1 X119.604 Y120.538 E.15272
G3 X112.93 Y126.698 I-39.766 J-36.391 E.34716
G1 X117.325 Y129.958 E.20894
G3 X118.151 Y130.725 I-6.631 J7.965 E.04304
G3 X118.695 Y131.451 I-12.23 J9.741 E.03463
G1 X119.048 Y132.093 E.028
G1 X119.353 Y132.891 E.03261
G3 X119.551 Y133.826 I-11.657 J2.95 E.03648
G1 X119.595 Y134.56 E.0281
G1 X119.531 Y135.403 E.03228
G1 X119.363 Y136.147 E.02912
G1 X119.08 Y136.91 E.03106
G3 X130.583 Y143.627 I-20.723 J48.697 E.50991
G3 X142.647 Y156.415 I-31.782 J42.068 E.67437
G1 X143.615 Y156.235 E.03755
G1 X143.615 Y104.474 E1.97618
G1 X143.615 Y104.074 E.01527
; LINE_WIDTH: 0.664046
G1 F12487.628
G1 X143.593 Y103.841 E.00964
; LINE_WIDTH: 0.708096
G1 F11688.139
G1 X143.571 Y103.607 E.01031
; LINE_WIDTH: 0.752146
G1 F10970.871
G1 X143.548 Y103.373 E.01098
; LINE_WIDTH: 0.792521
G1 F10386.648
G1 X143.528 Y103.141 E.01154
; LINE_WIDTH: 0.832896
G1 F9861.502
G1 X143.508 Y102.908 E.01216
; LINE_WIDTH: 0.838056
G1 F9798.189
G3 X143.518 Y102.402 I4.289 J-.165 E.02652
; LINE_WIDTH: 0.812096
G1 F10125.233
G1 X143.543 Y101.903 E.02535
; LINE_WIDTH: 0.764071
G1 F10791.59
G1 X143.567 Y101.404 E.02379
; LINE_WIDTH: 0.716046
G1 F11551.834
G1 X143.591 Y100.904 E.02222
; LINE_WIDTH: 0.668021
G1 F12427.311
G1 X143.615 Y100.405 E.02066
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X143.615 Y100.005 E.01527
G1 X143.615 Y99.567 E.01672
G1 X142.923 Y99.676 E.02673
G1 X142.019 Y99.684 E.0345
G1 X142.177 Y100.537 E.03312
; LINE_WIDTH: 0.645946
G1 F12870.615
G1 X142.239 Y100.8 E.01077
; LINE_WIDTH: 0.693974
G1 F11938.376
G1 X142.354 Y101.286 E.0215
; LINE_WIDTH: 0.742001
G1 F11128.148
M73 P96 R0
G1 X142.468 Y101.773 E.02307
; LINE_WIDTH: 0.790029
G1 F10420.906
G1 X142.583 Y102.26 E.02463
; LINE_WIDTH: 0.838056
G1 F9798.189
G1 X142.697 Y102.747 E.0262
; LINE_WIDTH: 0.842626
G1 F9742.791
G1 X142.701 Y103.062 E.01662
G1 X142.658 Y103.133 E.00438
; LINE_WIDTH: 0.797386
G1 F10320.426
G1 X142.616 Y103.204 E.00413
; LINE_WIDTH: 0.752146
G1 F10746.594
G1 X142.523 Y103.301 E.00627
; LINE_WIDTH: 0.708096
G1 F11181.383
G1 X142.43 Y103.398 E.00589
; LINE_WIDTH: 0.664046
G1 F11624.771
G1 X142.337 Y103.494 E.0055
; LINE_WIDTH: 0.619996
G1 F12693.798
G1 X142.065 Y103.649 E.01195
G1 F13446.369
G1 X141.825 Y103.687 E.00928
G1 X132.766 Y103.687 E.34584
G1 X132.405 Y103.598 E.01419
G3 X132.077 Y103.27 I.361 J-.69 E.01799
G1 X132.074 Y103.251 E.00072
M204 S250
G1 X132.566 Y103.013 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X132.544 Y102.867 E.00467
G1 X133.236 Y99.134 E.12021
G3 X131.519 Y98.947 I-.171 J-6.418 E.05482
G1 X131.038 Y98.936 E.01525
G3 X123.946 Y114.147 I-51.152 J-14.591 E.53358
G3 X121.746 Y117.122 I-28.176 J-18.529 E.11719
G1 X119.177 Y120.187 E.12664
G3 X112.01 Y126.704 I-38.871 J-35.55 E.30711
G1 X116.99 Y130.398 E.19632
G3 X117.738 Y131.094 I-6.465 J7.707 E.03236
G3 X118.21 Y131.717 I-28.803 J22.294 E.02477
G1 X118.564 Y132.36 E.02321
G1 X118.835 Y133.083 E.02446
G1 X118.953 Y133.629 E.01769
G1 X119.044 Y134.594 E.03068
G1 X118.98 Y135.356 E.0242
G1 X118.825 Y136.02 E.02158
G1 X118.548 Y136.752 E.02481
G1 X118.31 Y137.19 E.01576
G3 X130.253 Y144.07 I-19.805 J48.181 E.43766
G3 X142.39 Y157.025 I-31.647 J41.813 E.56476
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.195 E1.82046
G1 X144.167 Y98.939 E.00812
G1 X143.86 Y98.947 E.00974
G3 X142.829 Y99.128 I-1.464 J-5.31 E.03318
G1 X141.355 Y99.133 E.04667
G1 X142.047 Y102.869 E.1203
G1 X142.036 Y102.987 E.00376
G1 X141.894 Y103.123 E.00621
G1 X132.766 Y103.134 E.28899
G1 X132.625 Y103.079 E.00479
; WIPE_START
M204 S10000
G1 X132.544 Y102.867 E-.08618
G1 X132.685 Y102.107 E-.29382
; WIPE_END
G1 E-.02 F1800
G1 X140.203 Y100.791 Z11.8 F36000
G1 X142.944 Y100.311 Z11.8
G1 Z11.4
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.898176
G1 F9116.272
G1 X142.836 Y100.311 E.00608
G1 X142.782 Y100.405 E.00608
G1 X142.836 Y100.498 E.00608
G1 X142.944 Y100.498 E.00608
G1 X142.998 Y100.405 E.00608
; WIPE_START
G1 X142.944 Y100.498 E-.076
G1 X142.836 Y100.498 E-.076
G1 X142.782 Y100.405 E-.076
G1 X142.836 Y100.311 E-.076
G1 X142.944 Y100.311 E-.076
; WIPE_END
G1 E-.02 F1800
G1 X135.458 Y101.799 Z11.8 F36000
G1 X131.242 Y102.636 Z11.8
G1 Z11.4
G1 E.4 F1800
; LINE_WIDTH: 0.988416
G1 F8254.026
G1 X131.304 Y102.41 E.01463
; LINE_WIDTH: 0.949496
G1 F8605.053
G1 X131.388 Y102.086 E.01997
; LINE_WIDTH: 0.900956
G1 F9087.028
G1 X131.473 Y101.762 E.01891
; LINE_WIDTH: 0.852416
G1 F9626.198
G1 X131.558 Y101.438 E.01785
; LINE_WIDTH: 0.803876
G1 F10233.387
G1 X131.635 Y101.128 E.016
; LINE_WIDTH: 0.764341
G1 F10787.599
G1 X131.712 Y100.819 E.01518
; LINE_WIDTH: 0.724806
G1 F11405.276
G1 X131.79 Y100.509 E.01436
; LINE_WIDTH: 0.685271
G1 F12097.983
G1 X131.867 Y100.2 E.01354
; WIPE_START
G1 X131.79 Y100.509 E-.12123
G1 X131.712 Y100.819 E-.12122
G1 X131.635 Y101.128 E-.12123
G1 X131.625 Y101.17 E-.01633
; WIPE_END
G1 E-.02 F1800
G1 X138.513 Y104.457 Z11.8 F36000
G1 X142.209 Y106.22 Z11.8
G1 Z11.4
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.624866
G1 F13335.48
G1 X141.345 Y105.356 E.04705
G1 X140.51 Y105.356 E.03215
G1 X141.945 Y106.792 E.07817
G1 X141.945 Y107.627 E.03215
G1 X139.674 Y105.356 E.12363
G1 X138.839 Y105.356 E.03215
G1 X141.945 Y108.462 E.1691
G1 X141.945 Y109.297 E.03215
G1 X138.004 Y105.356 E.21457
G1 X137.169 Y105.356 E.03215
G1 X141.945 Y110.132 E.26004
G1 X141.945 Y110.967 E.03215
G1 X136.334 Y105.356 E.3055
G1 X135.499 Y105.356 E.03215
G1 X141.945 Y111.802 E.35097
G1 X141.945 Y112.638 E.03215
G1 X134.664 Y105.356 E.39644
G1 X133.829 Y105.356 E.03215
G1 X141.945 Y113.473 E.4419
G1 X141.945 Y114.308 E.03215
G1 X132.993 Y105.356 E.48737
G3 X132.358 Y105.323 I-.118 J-3.879 E.02453
G1 X132.008 Y105.206 E.01418
G1 X141.945 Y115.143 E.541
G1 X141.945 Y115.978 E.03215
G1 X131.205 Y105.238 E.58471
G1 X130.963 Y105.831 E.02466
G1 X141.945 Y116.813 E.5979
G1 X141.945 Y117.648 E.03215
G1 X130.714 Y106.417 E.61145
G1 X130.453 Y106.991 E.02428
G1 X141.945 Y118.483 E.62566
G1 X141.945 Y119.319 E.03215
G1 X130.192 Y107.565 E.63987
G1 X129.931 Y108.14 E.02428
G1 X141.945 Y120.154 E.65408
G1 X141.945 Y120.989 E.03215
G1 X129.66 Y108.704 E.66882
G1 X129.385 Y109.264 E.02401
G1 X141.945 Y121.824 E.68382
G1 X141.945 Y122.659 E.03215
G1 X129.096 Y109.81 E.69955
G1 X128.806 Y110.355 E.02378
G1 X141.945 Y123.494 E.71532
G1 X141.945 Y124.329 E.03215
G1 X128.517 Y110.901 E.73109
G3 X128.224 Y111.443 I-8.178 J-4.062 E.02373
G1 X141.945 Y125.165 E.74703
G1 X141.945 Y126 E.03215
G1 X127.919 Y111.973 E.76363
G1 X127.614 Y112.504 E.02355
G1 X141.945 Y126.835 E.78023
G1 X141.945 Y127.67 E.03215
G1 X127.296 Y113.02 E.79757
G1 X126.976 Y113.535 E.02335
G1 X141.945 Y128.505 E.81498
G1 X141.945 Y129.34 E.03215
G1 X126.656 Y114.051 E.8324
G3 X126.335 Y114.565 I-8.529 J-4.966 E.02334
G1 X141.945 Y130.175 E.84987
G1 X141.945 Y131.011 E.03215
G1 X126.002 Y115.067 E.86802
G3 X125.665 Y115.566 I-11.821 J-7.604 E.02316
G1 X141.945 Y131.846 E.88633
G1 X141.945 Y132.681 E.03215
G1 X125.323 Y116.058 E.90498
G3 X124.979 Y116.549 I-11.724 J-7.843 E.02309
G1 X141.945 Y133.516 E.9237
G1 X141.945 Y134.351 E.03215
G1 X124.627 Y117.033 E.94285
G3 X124.272 Y117.513 I-11.8 J-8.352 E.02299
G1 X141.945 Y135.186 E.96218
G1 X141.945 Y136.021 E.03215
G1 X123.912 Y117.987 E.98181
G3 X123.535 Y118.446 I-4.967 J-3.689 E.02285
G1 X141.945 Y136.856 E1.00229
G1 X141.945 Y137.692 E.03215
G1 X123.154 Y118.901 E1.02302
G1 X122.774 Y119.355 E.02282
G1 X141.945 Y138.527 E1.04376
G1 X141.945 Y139.362 E.03215
G1 X122.393 Y119.809 E1.06449
G1 X122.012 Y120.264 E.02282
G1 X141.945 Y140.197 E1.08522
G1 X141.945 Y141.032 E.03215
G1 X121.631 Y120.718 E1.10596
G1 X121.25 Y121.172 E.02282
G1 X141.945 Y141.867 E1.12669
G1 X141.945 Y142.702 E.03215
G1 X120.869 Y121.626 E1.14745
G1 X120.468 Y122.06 E.02275
G1 X141.945 Y143.538 E1.16926
G1 X141.945 Y144.373 E.03215
G1 X120.061 Y122.489 E1.19141
G1 X119.649 Y122.911 E.02274
G1 X141.945 Y145.208 E1.21388
G1 X141.945 Y146.043 E.03215
G1 X119.236 Y123.334 E1.23634
G3 X118.822 Y123.755 I-7.802 J-7.268 E.02274
G1 X141.945 Y146.878 E1.2589
G1 X141.945 Y147.713 E.03215
G1 X118.398 Y124.166 E1.28198
G3 X117.969 Y124.572 I-6.833 J-6.779 E.02275
G1 X141.945 Y148.548 E1.30532
G1 X141.945 Y149.384 E.03215
G1 X117.532 Y124.97 E1.32911
G1 X117.095 Y125.369 E.02276
G1 X141.945 Y150.219 E1.3529
G1 X141.945 Y151.054 E.03215
G1 X116.658 Y125.767 E1.37668
G3 X116.215 Y126.159 I-6.087 J-6.432 E.02278
G1 X141.945 Y151.889 E1.4008
G1 X141.945 Y152.573 E.02634
G2 X131.307 Y142.086 I-43.403 J33.388 E.57686
G1 X120.512 Y131.29 E.58775
G3 X120.955 Y132.415 I-8.292 J3.916 E.04656
G1 X120.999 Y132.613 E.00782
G1 X128.427 Y140.04 E.40437
G2 X126.192 Y138.641 I-23.909 J35.692 E.10152
G1 X121.211 Y133.66 E.27119
G1 X121.265 Y134.549 E.03431
G1 X124.266 Y137.55 E.16339
G1 X122.543 Y136.662 E.07462
G1 X120.963 Y135.082 E.08606
; WIPE_START
G1 X121.67 Y135.789 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X118.05 Y129.069 Z11.8 F36000
G1 X117.77 Y128.549 Z11.8
G1 Z11.4
G1 E.4 F1800
G1 F13335.48
G1 X115.578 Y126.356 E.11938
; CHANGE_LAYER
; Z_HEIGHT: 11.56
; LAYER_HEIGHT: 0.160001
; WIPE_START
G1 F13335.48
G1 X116.285 Y127.063 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L72
M991 S0 P71 ;notify layer change


G17
G3 Z11.8 I1.217 J0 P1  F36000
;======== P2S timelapes gcode ==========
;===== 2025/06/16 ====
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firware, default turned on

M1002 judge_flag timelapse_record_flag
M622 J1
 ; timelapse without wipe tower
  M971 S11 C10 O0
  M1004 S5 P1  ; external shutter

M623
; SKIPPABLE_END

; OBJECT_ID: 15
G1 X132.564 Y103.015
G1 Z11.56
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X132.542 Y102.869 E.00466
G1 X133.233 Y99.136 E.12021
M73 P97 R0
G3 X131.519 Y98.947 I-.169 J-6.354 E.05476
G1 X131.038 Y98.936 E.01525
G3 X123.946 Y114.147 I-51.151 J-14.591 E.53359
G3 X121.747 Y117.121 I-28.175 J-18.528 E.11717
G1 X119.177 Y120.187 E.12664
G3 X112.01 Y126.704 I-38.875 J-35.553 E.30713
G1 X116.989 Y130.398 E.19629
G3 X117.739 Y131.094 I-6.435 J7.676 E.0324
G3 X118.211 Y131.717 I-28.944 J22.397 E.02476
G1 X118.565 Y132.362 E.02329
G1 X118.834 Y133.08 E.02426
G3 X118.999 Y133.861 I-26.672 J6.042 E.0253
G1 X119.044 Y134.599 E.02338
G1 X118.98 Y135.355 E.02403
G1 X118.824 Y136.026 E.02182
G1 X118.547 Y136.754 E.02465
G1 X118.31 Y137.19 E.01572
G3 X130.249 Y144.068 I-19.583 J47.797 E.43754
G3 X142.39 Y157.025 I-31.528 J41.707 E.56492
G1 X144.167 Y156.695 E.05724
G1 X144.167 Y99.195 E1.82047
G1 X144.167 Y98.939 E.0081
G1 X143.86 Y98.948 E.00974
G3 X142.828 Y99.131 I-1.468 J-5.277 E.03324
G1 X141.358 Y99.135 E.04655
G1 X142.049 Y102.871 E.1203
G1 X142.038 Y102.989 E.00376
G1 X141.897 Y103.125 E.00621
G1 X132.764 Y103.136 E.28915
G1 X132.623 Y103.081 E.0048
; WIPE_START
M204 S10000
G1 X132.542 Y102.869 E-.08606
G1 X132.683 Y102.109 E-.29394
; WIPE_END
G1 E-.02 F1800
G1 X134.075 Y109.613 Z11.96 F36000
G1 X142.809 Y156.671 Z11.96
G1 Z11.56
G1 E.4 F1800
; FEATURE: Top surface
; LINE_WIDTH: 0.62
G1 F9000
M204 S2000
G1 X143.895 Y155.584 E.05868
G1 X144.093 Y155.387
G1 X144.093 Y154.558
G1 X143.895 Y154.756
G1 X142.276 Y156.375 E.08745
G1 X142.078 Y156.573
G1 X141.741 Y156.082
G1 X141.938 Y155.885
G1 X143.895 Y153.927 E.10567
G1 X144.093 Y153.73
G1 X144.093 Y152.902
G1 X143.895 Y153.099
G1 X141.601 Y155.394 E.1239
G1 X141.403 Y155.591
G1 X141.061 Y155.105
G1 X141.258 Y154.908
G1 X143.895 Y152.271 E.14239
G1 X144.093 Y152.074
G1 X144.093 Y151.245
G1 X143.895 Y151.443
G1 X140.907 Y154.43 E.16132
G1 X140.71 Y154.628
G1 X140.354 Y154.156
G1 X140.551 Y153.959
G1 X143.895 Y150.614 E.18057
G1 X144.093 Y150.417
G1 X144.093 Y149.589
G1 X143.895 Y149.786
G1 X140.191 Y153.49 E.2
G1 X139.994 Y153.688
G1 X139.629 Y153.224
G1 X139.826 Y153.027
G1 X143.895 Y148.958 E.2197
G1 X144.093 Y148.761
G1 X144.093 Y147.932
G1 X143.895 Y148.13
G1 X139.456 Y152.569 E.23969
G1 X139.259 Y152.766
G1 X138.885 Y152.311
G1 X139.083 Y152.114
G1 X143.895 Y147.301 E.25985
G1 X144.093 Y147.104
G1 X144.093 Y146.276
G1 X143.895 Y146.473
G1 X138.701 Y151.667 E.28045
G1 X138.504 Y151.865
G1 X138.122 Y151.418
G1 X138.32 Y151.221
G1 X143.895 Y145.645 E.30105
G1 X144.093 Y145.448
G1 X144.093 Y144.619
G1 X143.895 Y144.817
G1 X137.928 Y150.784 E.32218
G1 X137.731 Y150.981
G1 X137.335 Y150.548
G1 X137.533 Y150.351
G1 X143.895 Y143.988 E.34354
G1 X144.093 Y143.791
G1 X144.093 Y142.963
G1 X143.895 Y143.16
G1 X137.137 Y149.919 E.36491
G1 X136.94 Y150.116
G1 X136.541 Y149.686
G1 X136.739 Y149.488
G1 X143.895 Y142.332 E.38641
G1 X144.093 Y142.135
G1 X144.093 Y141.306
G1 X143.895 Y141.504
G1 X136.331 Y149.068 E.40843
G1 X136.134 Y149.265
G1 X135.722 Y148.849
G1 X135.919 Y148.652
G1 X143.895 Y140.675 E.43067
G1 X144.093 Y140.478
G1 X144.093 Y139.65
G1 X143.895 Y139.847
G1 X135.502 Y148.241 E.45318
G1 X135.305 Y148.438
G1 X134.885 Y148.029
G1 X135.083 Y147.832
G1 X143.895 Y139.019 E.47583
G1 X144.093 Y138.822
G1 X144.093 Y137.993
G1 X143.895 Y138.191
G1 X134.657 Y147.429 E.49884
G1 X134.459 Y147.627
G1 X134.028 Y147.229
G1 X134.226 Y147.032
G1 X143.895 Y137.362 E.5221
G1 X144.093 Y137.165
G1 X144.093 Y136.337
G1 X143.895 Y136.534
G1 X133.791 Y146.639 E.54559
G1 X133.593 Y146.836
G1 X133.154 Y146.447
G1 X133.352 Y146.25
G1 X143.895 Y135.706 E.56929
G1 X144.093 Y135.509
G1 X144.093 Y134.68
G1 X143.895 Y134.878
G1 X132.907 Y145.866 E.59332
G1 X132.709 Y146.064
G1 X132.262 Y145.682
G1 X132.46 Y145.485
G1 X143.895 Y134.049 E.61745
G1 X144.093 Y133.852
G1 X144.093 Y133.024
G1 X143.895 Y133.221
G1 X132.005 Y145.111 E.64197
G1 X131.808 Y145.308
G1 X131.349 Y144.939
G1 X131.547 Y144.742
G1 X143.895 Y132.393 E.66675
G1 X144.093 Y132.196
G1 X144.093 Y131.367
G1 X143.895 Y131.565
G1 X131.083 Y144.377 E.69178
G1 X130.886 Y144.574
G1 X130.422 Y144.21
G1 X130.619 Y144.013
G1 X143.895 Y130.736 E.71682
G1 X144.093 Y130.539
G1 X144.093 Y129.711
G1 X143.895 Y129.908
G1 X130.149 Y143.654 E.74221
G1 X129.952 Y143.852
G1 X129.476 Y143.499
G1 X129.674 Y143.302
G1 X143.895 Y129.08 E.76788
G1 X144.093 Y128.883
G1 X144.093 Y128.054
G1 X143.895 Y128.252
G1 X129.198 Y142.949 E.79355
G1 X129.001 Y143.146
G1 X128.521 Y142.798
G1 X128.718 Y142.601
G1 X143.895 Y127.424 E.81948
G1 X144.093 Y127.226
G1 X144.093 Y126.398
G1 X143.895 Y126.595
G1 X128.228 Y142.262 E.84592
G1 X128.031 Y142.46
G1 X127.541 Y142.121
G1 X127.738 Y141.924
G1 X143.895 Y125.767 E.87237
G1 X144.093 Y125.57
G1 X144.093 Y124.741
G1 X143.895 Y124.939
G1 X127.24 Y141.594 E.89929
G1 X127.043 Y141.792
G1 X126.54 Y141.466
G1 X126.737 Y141.269
G1 X143.895 Y124.111 E.92645
G1 X144.093 Y123.913
G1 X144.093 Y123.085
G1 X143.895 Y123.282
G1 X126.234 Y140.944 E.95361
G1 X126.036 Y141.141
G1 X125.52 Y140.829
G1 X125.717 Y140.632
G1 X143.895 Y122.454 E.9815
G1 X144.093 Y122.257
G1 X144.093 Y121.428
G1 X143.895 Y121.626
G1 X125.201 Y140.32 E1.00938
G1 X125.004 Y140.517
G1 X124.482 Y140.211
G1 X124.679 Y140.013
G1 X143.895 Y120.798 E1.03754
G1 X144.093 Y120.6
G1 X144.093 Y119.772
G1 X143.895 Y119.969
G1 X124.149 Y139.715 E1.06615
G1 X123.952 Y139.912
G1 X123.422 Y139.614
G1 X123.619 Y139.417
G1 X143.895 Y119.141 E1.09477
G1 X144.093 Y118.944
G1 X144.093 Y118.115
G1 X143.895 Y118.313
G1 X123.078 Y139.13 E1.12401
G1 X122.881 Y139.327
G1 X122.335 Y139.044
G1 X122.533 Y138.847
G1 X143.895 Y117.485 E1.15345
G1 X144.093 Y117.287
G1 X144.093 Y116.459
G1 X143.895 Y116.656
G1 X121.987 Y138.564 E1.18288
G1 X121.79 Y138.761
G1 X121.235 Y138.489
G1 X121.432 Y138.291
G1 X143.895 Y115.828 E1.21288
G1 X144.093 Y115.631
G1 X144.093 Y114.803
G1 X143.895 Y115
G1 X120.871 Y138.024 E1.24316
G1 X120.674 Y138.221
M73 P98 R0
G1 X120.111 Y137.955
G1 X120.309 Y137.758
G1 X143.895 Y114.172 E1.27352
G1 X144.093 Y113.974
G1 X144.093 Y113.146
G1 X143.895 Y113.343
G1 X119.732 Y137.507 E1.30467
G1 X119.535 Y137.704
G1 X118.958 Y137.452
G1 X119.155 Y137.255
G1 X143.895 Y112.515 E1.33582
G1 X144.093 Y112.318
G1 X144.093 Y111.49
G1 X143.895 Y111.687
G1 X118.846 Y136.736 E1.35248
G1 X118.649 Y136.933
G1 X119.017 Y135.736
G1 X119.215 Y135.539
G1 X143.895 Y110.859 E1.3326
G1 X144.093 Y110.661
G1 X144.093 Y109.833
G1 X143.895 Y110.03
G1 X119.314 Y134.611 E1.32723
G1 X119.117 Y134.809
G1 X119.072 Y134.025
G1 X119.269 Y133.828
G1 X143.895 Y109.202 E1.32965
G1 X144.093 Y109.005
G1 X144.093 Y108.177
G1 X143.895 Y108.374
G1 X119.125 Y133.144 E1.33742
G1 X118.928 Y133.341
G1 X118.72 Y132.721
G1 X118.917 Y132.524
G1 X143.895 Y107.546 E1.34866
G1 X144.093 Y107.348
G1 X144.093 Y106.52
G1 X143.895 Y106.717
G1 X118.653 Y131.96 E1.36294
G1 X118.455 Y132.157
G1 X118.146 Y131.639
G1 X118.343 Y131.441
G1 X143.895 Y105.889 E1.37967
G1 X144.093 Y105.692
G1 X144.093 Y104.864
G1 X143.895 Y105.061
G1 X117.986 Y130.971 E1.39896
G1 X117.788 Y131.168
G1 X117.376 Y130.752
G1 X117.574 Y130.554
G1 X143.895 Y104.233 E1.42121
G1 X144.093 Y104.035
G1 X144.093 Y103.207
G1 X143.895 Y103.404
G1 X117.127 Y130.172 E1.4453
G1 X116.93 Y130.37
G1 X116.455 Y130.016
G1 X116.652 Y129.819
G1 X143.895 Y102.576 E1.47094
G1 X144.093 Y102.379
G1 X144.093 Y101.551
G1 X143.895 Y101.748
G1 X116.178 Y129.465 E1.49658
G1 X115.98 Y129.663
G1 X115.505 Y129.309
G1 X115.703 Y129.112
G1 X141.407 Y103.408 E1.38786
G1 X141.604 Y103.211
G1 X140.776 Y103.211
G1 X140.579 Y103.408
G1 X115.228 Y128.759 E1.36878
G1 X115.031 Y128.956
G1 X114.556 Y128.603
G1 X114.753 Y128.405
G1 X139.75 Y103.408 E1.3497
G1 X139.948 Y103.211
G1 X139.119 Y103.211
G1 X138.922 Y103.408
G1 X114.278 Y128.052 E1.33062
G1 X114.081 Y128.249
G1 X113.606 Y127.896
G1 X113.803 Y127.699
G1 X138.094 Y103.408 E1.31154
G1 X138.291 Y103.211
G1 X137.463 Y103.211
G1 X137.266 Y103.408
G1 X113.328 Y127.345 E1.29246
G1 X113.131 Y127.542
G1 X112.656 Y127.189
G1 X112.854 Y126.992
G1 X136.437 Y103.408 E1.27338
G1 X136.635 Y103.211
G1 X135.806 Y103.211
G1 X135.609 Y103.408
G1 X122.794 Y116.223 E.69191
; WIPE_START
M204 S10000
G1 X123.502 Y115.516 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X124.726 Y113.463 Z11.96 F36000
G1 Z11.56
G1 E.4 F1800
G1 F9000
M204 S2000
G1 X134.781 Y103.408 E.54288
G1 X134.978 Y103.211
G1 X134.15 Y103.211
G1 X133.953 Y103.408
G1 X126.047 Y111.313 E.42683
; WIPE_START
M204 S10000
G1 X126.755 Y110.606 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X127.103 Y109.43 Z11.96 F36000
G1 Z11.56
G1 E.4 F1800
G1 F9000
M204 S2000
G1 X133.124 Y103.408 E.32512
G1 X133.322 Y103.211
G1 X132.624 Y103.08
G1 X132.427 Y103.277
G1 X127.959 Y107.745 E.24126
G1 X127.761 Y107.943
G1 X128.49 Y106.386
M73 P99 R0
G1 X128.688 Y106.188
G1 X132.324 Y102.552 E.19633
G1 X132.521 Y102.355
G1 X132.709 Y101.338
G1 X132.512 Y101.536
G1 X129.295 Y104.753 E.1737
G1 X129.098 Y104.95
G1 X129.659 Y103.561
G1 X129.856 Y103.363
G1 X132.7 Y100.519 E.15357
G1 X132.898 Y100.322
G1 X133.086 Y99.305
G1 X132.888 Y99.503
G1 X130.334 Y102.057 E.1379
G1 X130.137 Y102.254
G1 X130.549 Y101.014
G1 X130.746 Y100.817
G1 X132.205 Y99.358 E.07878
; WIPE_START
M204 S10000
G1 X131.498 Y100.065 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X138.934 Y101.785 Z11.96 F36000
G1 X142.26 Y102.555 Z11.96
G1 Z11.56
G1 E.4 F1800
G1 F9000
M204 S2000
G1 X143.895 Y100.92 E.08828
G1 X144.093 Y100.722
G1 X144.093 Y99.894
G1 X143.895 Y100.091
G1 X142.132 Y101.854 E.09519
G1 X141.935 Y102.052
G1 X141.807 Y101.351
G1 X142.004 Y101.154
G1 X143.895 Y99.263 E.10211
G1 X144.093 Y99.066
G1 X143.135 Y99.195
G1 X142.938 Y99.393
G1 X141.876 Y100.454 E.05731
; WIPE_START
M204 S10000
G1 X142.583 Y99.747 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X136.907 Y104.849 Z11.96 F36000
G1 X113.463 Y125.921 Z11.96
G1 Z11.56
G1 E.4 F1800
; FEATURE: Gap infill
; LINE_WIDTH: 0.127538
G1 F3000
G1 X113.144 Y126.204 E.00259
; LINE_WIDTH: 0.17337
G1 X112.825 Y126.487 E.00387
; LINE_WIDTH: 0.219201
G1 X112.506 Y126.769 E.00514
; WIPE_START
G1 X112.825 Y126.487 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X118.193 Y121.061 Z11.96 F36000
G1 X122.888 Y116.316 Z11.96
G1 Z11.56
G1 E.4 F1800
; LINE_WIDTH: 0.310398
G1 F3000
G1 X122.641 Y116.603 E.00681
; LINE_WIDTH: 0.26521
G1 X122.39 Y116.894 E.00579
; LINE_WIDTH: 0.21947
G1 X122.028 Y117.289 E.00646
; LINE_WIDTH: 0.173526
G1 X121.666 Y117.684 E.00486
; LINE_WIDTH: 0.127581
G1 X121.305 Y118.079 E.00325
; WIPE_START
G1 X121.666 Y117.684 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X124.82 Y113.557 Z11.96 F36000
G1 Z11.56
G1 E.4 F1800
; LINE_WIDTH: 0.303514
G1 F3000
G1 X124.664 Y113.747 E.00432
; LINE_WIDTH: 0.261789
G1 X124.508 Y113.938 E.00365
; LINE_WIDTH: 0.220063
G1 X124.352 Y114.129 E.00298
; LINE_WIDTH: 0.178224
G1 X124.195 Y114.32 E.00232
; LINE_WIDTH: 0.144069
G1 X124.089 Y114.448 E.00119
; LINE_WIDTH: 0.117773
G1 X123.983 Y114.575 E.0009
; WIPE_START
G1 X124.089 Y114.448 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X126.142 Y111.408 Z11.96 F36000
G1 Z11.56
G1 E.4 F1800
; LINE_WIDTH: 0.30235
G1 F3000
G1 X126.008 Y111.585 E.00389
; LINE_WIDTH: 0.267644
G1 X125.883 Y111.744 E.00307
; LINE_WIDTH: 0.221063
G1 X125.759 Y111.903 E.00245
; LINE_WIDTH: 0.174483
G1 X125.635 Y112.062 E.00184
; LINE_WIDTH: 0.127902
G1 X125.51 Y112.22 E.00123
; WIPE_START
G1 X125.635 Y112.062 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X127.083 Y109.41 Z11.96 F36000
G1 Z11.56
G1 E.4 F1800
; LINE_WIDTH: 0.250528
G1 F3000
G1 X127.109 Y109.59 E.00257
; LINE_WIDTH: 0.2671
G1 X127.112 Y109.613 E.00035
; LINE_WIDTH: 0.29528
G1 X127.115 Y109.637 E.0004
; LINE_WIDTH: 0.288885
G1 X127.02 Y109.762 E.00262
; LINE_WIDTH: 0.247935
G1 X126.925 Y109.888 E.00219
; LINE_WIDTH: 0.206985
G1 X126.83 Y110.013 E.00177
; LINE_WIDTH: 0.166035
G1 X126.734 Y110.139 E.00135
; LINE_WIDTH: 0.125085
G1 X126.639 Y110.264 E.00093
; WIPE_START
G1 X126.734 Y110.139 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X127.737 Y108.283 Z11.96 F36000
G1 Z11.56
G1 E.4 F1800
; LINE_WIDTH: 0.129237
G1 F3000
G1 X127.593 Y108.484 E.00153
; close powerlost recovery
M1003 S0
; WIPE_START
G1 F3000
G1 X127.737 Y108.283 E-.38
; WIPE_END
G1 E-.02 F1800
G17
G3 Z11.96 I1.217 J0 P1  F36000
M106 S0
M106 P2 S0
M981 S0 P20000 ; close spaghetti detector
; FEATURE: Custom
; MACHINE_END_GCODE_START
; filament end gcode 

;======== P2S end gcode ==========
;===== 2026/05/18 =====
M400 ; wait for buffer to clear
G92 E0 ; zero the extruder
M211 Z1

G90
G1 Z11.96 F900 ; lower z a little
M1002 judge_flag timelapse_record_flag
M622 J1
    G150.3
    M400 ; wait all motion done
    M991 S0 P-1 ;end smooth timelapse at safe pos
    M400 S5 ;wait for last picture to be taken
M623  ;end of "timelapse_record_flag

G90
G1 Z21.56 F900 ; lower z a little

M140 S0 ; turn off bed
M106 S0 ; turn off fan
M106 P2 S0 ; turn off remote part cooling fan
M106 P3 S0 ; turn off chamber cooling fan
M106 P10 S0 ; turn off left aux fan

; pull back filament to AMS
M620 S65535
T65535
G150.1 F8000
M621 S65535

G150.3
M104 S0 ; turn off hotend
M400 ; wait all motion done
M17 S
M17 Z0.4 ; lower z motor current to reduce impact if there is something in the bottom

    
        G1 Z85.78 F600
        G1 Z83.78
    

M400 P100
M17 R ; restore z current


M220 S100  ; Reset feedrate magnitude
M201.2 K1.0 ; Reset acc magnitude
M73.2 R1.0 ;Reset left time magnitude
M1002 set_gcode_claim_speed_level : 0

M1015.3 S0 ;disable clog detect
M1015.4 S0 K0 ;disable air printing detect

;=====printer finish air purification=========
M622.1 S0
M1002 judge_flag print_finish_air_filt_flag

M622 J1
M1002 gcode_claim_action : 66
M145 P1
M106 P2 S255
M400 S180
M106 P2 S0
M623

M622 J2
M1002 gcode_claim_action : 66
M145 P0
M106 P3 S255
M400 S180
M106 P3 S0
M623
;=====printer finish air purification=========

;=====printer finish  sound=========
M17
M400 S1
M1006 S1
M1006 A53 B10 L50 C53 D10 M50 E53 F10 N50 
M1006 A57 B10 L50 C57 D10 M50 E57 F10 N50 
M1006 A0 B15 L0 C0 D15 M0 E0 F15 N0 
M1006 A53 B10 L50 C53 D10 M50 E53 F10 N50 
M1006 A57 B10 L50 C57 D10 M50 E57 F10 N50 
M1006 A0 B15 L0 C0 D15 M0 E0 F15 N0 
M1006 A48 B10 L50 C48 D10 M50 E48 F10 N50 
M1006 A0 B15 L0 C0 D15 M0 E0 F15 N0 
M1006 A60 B10 L50 C60 D10 M50 E60 F10 N50 
M1006 W
;=====printer finish  sound=========
M400
M18
M73 P100 R0
; EXECUTABLE_BLOCK_END

