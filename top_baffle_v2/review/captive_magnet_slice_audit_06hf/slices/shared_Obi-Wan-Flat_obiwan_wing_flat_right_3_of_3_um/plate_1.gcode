; HEADER_BLOCK_START
; BambuStudio 02.07.01.62
; model printing time: 12m 15s; total estimated time: 19m 17s
; total layer number: 72
; total filament length [mm] : 2593.87
; total filament volume [cm^3] : 6238.99
; total filament weight [g] : 7.86
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
M73 P2 R18
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
    
      G29 A1 X98.6759 Y111.577 I58.6363 J32.8504
    
    M400
  M623

  M622 J2
    M1002 gcode_claim_action : 1
    
      G29 A2 X98.6759 Y111.577 I58.6363 J32.8504
    
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
M73 P36 R12
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
G1 X104.743 Y131.892
G1 Z.2
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.61999
G1 F2100
M204 S500
G1 X104.764 Y131.927 E.00192
G1 X104.905 Y132.374 E.02201
G1 X104.94 Y132.97 E.0281
G1 X104.94 Y141.84 E.41708
G1 X104.857 Y142.215 E.01807
G1 X153.375 Y142.215 E2.28151
G3 X143.626 Y132.845 I32.11 J-43.169 E.63752
G3 X136.141 Y120.811 I41.971 J-34.447 E.6683
G1 X135.612 Y120.916 E.02539
G3 X129.986 Y119.353 I-1.05 J-7.127 E.28271
G1 X129.366 Y118.781 E.03968
G1 X128.864 Y118.195 E.03627
G3 X126.651 Y115.217 I465.847 J-348.476 E.17446
G3 X121.653 Y120.479 I-43.688 J-36.493 E.34153
G3 X117.923 Y123.623 I-133.44 J-154.543 E.22939
G3 X104.266 Y131.309 I-33.591 J-43.707 E.73943
G1 X104.485 Y131.474 E.01287
G1 X104.696 Y131.816 E.0189
M204 S6000
G1 X104.222 Y132.138 F36000
G1 F2100
M204 S500
G1 X104.237 Y132.163 E.00137
G1 X104.338 Y132.482 E.01572
G1 X104.363 Y133.062 E.02733
G1 X104.363 Y141.84 E.41274
G1 X104.227 Y142.451 E.02946
G1 X103.917 Y142.792 E.02166
G1 X155.166 Y142.792 E2.40994
G3 X144.07 Y132.477 I30.598 J-44.043 E.71478
G3 X136.467 Y120.127 I41.902 J-34.312 E.68405
G3 X129.769 Y118.368 I-1.911 J-6.355 E.34261
G1 X129.308 Y117.826 E.03344
G3 X126.677 Y114.282 I748.203 J-558.344 E.20756
G3 X121.258 Y120.058 I-42.539 J-34.477 E.37276
G3 X117.552 Y123.18 I-138.287 J-160.376 E.22786
M73 P37 R12
G3 X103.4 Y131.023 I-33.259 J-43.33 E.76366
; LINE_WIDTH: 0.65956
G1 X102.654 Y131.313 E.04021
G1 X103.379 Y131.384 E.03659
; LINE_WIDTH: 0.61999
G1 X104.018 Y131.815 E.03625
G1 X104.174 Y132.062 E.01373
M204 S6000
G1 X103.688 Y132.406 F36000
G1 F2100
M204 S500
G1 X103.767 Y132.571 E.00865
G1 X103.786 Y134.035 E.06882
G1 X103.786 Y141.84 E.36701
G1 X103.704 Y142.207 E.01768
; LINE_WIDTH: 0.63305
G1 X103.362 Y142.591 E.02476
; LINE_WIDTH: 0.67236
G1 X103.075 Y142.718 E.01612
G1 X102.758 Y142.716 E.01623
; LINE_WIDTH: 0.67135
G1 X101.601 Y142.489 E.06036
; LINE_WIDTH: 0.64567
G1 X100.445 Y142.261 E.0579
; LINE_WIDTH: 0.61999
G1 X99.785 Y142.139 E.03155
G3 X99.767 Y143.369 I-19.753 J.331 E.05784
G1 X100.343 Y143.369 E.02707
; LINE_WIDTH: 0.64567
G1 X101.522 Y143.356 E.0579
; LINE_WIDTH: 0.67135
G3 X103.572 Y143.362 I.929 J31.166 E.10501
; LINE_WIDTH: 0.63305
G1 X104.444 Y143.369 E.04192
; LINE_WIDTH: 0.61999
G1 X156.031 Y143.369 E2.42584
G1 X156.094 Y143.03 E.01618
G1 X156.144 Y142.762 E.01284
G3 X144.515 Y132.109 I29.389 J-43.758 E.7444
G3 X136.828 Y119.509 I41.349 J-33.869 E.6963
G1 X136.783 Y119.405 E.0053
G3 X129.752 Y117.457 I-2.23 J-5.614 E.36837
G3 X126.692 Y113.335 I1404.503 J-1045.619 E.2414
G3 X120.863 Y119.638 I-43.855 J-34.715 E.40413
G3 X117.182 Y122.738 I-143.914 J-167.14 E.22632
G3 X102.019 Y130.903 I-32.684 J-42.535 E.81332
; LINE_WIDTH: 0.665946
G1 X101.913 Y130.963 E.00619
; LINE_WIDTH: 0.711902
G1 X101.807 Y131.023 E.00664
; LINE_WIDTH: 0.757858
G1 X101.701 Y131.083 E.0071
; LINE_WIDTH: 0.803814
G1 X101.595 Y131.143 E.00756
; LINE_WIDTH: 0.84977
G1 X101.489 Y131.203 E.00801
G1 X100.79 Y131.403 E.04779
; LINE_WIDTH: 0.802975
G1 X100.091 Y131.603 E.04502
; LINE_WIDTH: 0.75618
G1 X99.812 Y131.684 E.01686
; LINE_WIDTH: 0.75522
G1 X99.845 Y132.372 E.03995
; LINE_WIDTH: 0.78509
G1 X100.529 Y132.23 E.04222
; LINE_WIDTH: 0.81496
G1 X101.212 Y132.088 E.04392
; LINE_WIDTH: 0.84977
G1 X101.691 Y131.982 E.03225
G1 X101.813 Y131.983 E.00801
; LINE_WIDTH: 0.803814
G1 X101.935 Y131.984 E.00756
; LINE_WIDTH: 0.757858
G1 X102.057 Y131.984 E.0071
; LINE_WIDTH: 0.711902
G1 X102.179 Y131.985 E.00664
; LINE_WIDTH: 0.665946
G1 X102.3 Y131.986 E.00619
; LINE_WIDTH: 0.61999
G1 X102.763 Y131.9 E.02211
G1 X103.211 Y131.936 E.02112
G1 X103.591 Y132.204 E.02187
G1 X103.649 Y132.324 E.00629
M204 S6000
G1 X103.164 Y132.642 F36000
; FEATURE: Outer wall
G1 F2100
M204 S500
G1 X103.187 Y134.035 E.06548
G1 X103.187 Y141.84 E.36701
G3 X103.056 Y142.07 I-.267 J0 E.01305
G1 X102.872 Y142.102 E.00878
G1 X99.187 Y141.42 E.17622
G3 X99.146 Y143.967 I-37.019 J.679 E.11985
G1 X156.529 Y143.967 E2.69835
G1 X156.682 Y143.14 E.03958
G1 X156.804 Y142.483 E.0314
G3 X144.972 Y131.723 I28.826 J-43.584 E.75508
G3 X137.093 Y118.6 I40.508 J-33.247 E.72236
G3 X130.971 Y117.896 I-2.555 J-4.757 E.30839
G3 X129.697 Y116.38 I6.238 J-6.535 E.0933
G1 X126.702 Y112.343 E.23636
G3 X120.325 Y119.321 I-42.426 J-32.369 E.44509
G1 X116.797 Y122.279 E.21651
G3 X99.136 Y131.188 I-32.376 J-42.223 E.93562
M73 P37 R11
G3 X99.187 Y133.171 I-51.922 J2.321 E.09328
G1 X102.872 Y132.489 E.17623
G1 X103.01 Y132.5 E.00652
G1 X103.098 Y132.581 E.00565
; WIPE_START
G1 X103.16 Y133.579 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S6000
G1 X102.198 Y131.434 Z.6 F36000
G1 Z.2
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.62369
G1 F2100
M204 S500
G1 X102.426 Y131.374 E.01116
; LINE_WIDTH: 0.65956
G1 X102.654 Y131.313 E.01185
; WIPE_START
G1 X102.426 Y131.374 E-.19
G1 X102.198 Y131.434 E-.19
; WIPE_END
G1 E-.02 F1800
M204 S6000
G1 X100.986 Y138.97 Z.6 F36000
M73 P38 R11
G1 X100.377 Y142.752 Z.6
G1 Z.2
G1 E.4 F1800
; LINE_WIDTH: 0.58272
G1 F2100
M204 S500
G2 X100.411 Y142.81 I-.03 J.056 E.01436
; WIPE_START
G1 X100.377 Y142.869 E-.076
G1 X100.309 Y142.869 E-.076
G1 X100.276 Y142.81 E-.076
G1 X100.309 Y142.752 E-.076
G1 X100.377 Y142.752 E-.07599
; WIPE_END
G1 E-.02 F1800
M204 S6000
G1 X107.131 Y139.197 Z.6 F36000
G1 X137.142 Y123.397 Z.6
G1 Z.2
G1 E.4 F1800
; FEATURE: Bottom surface
; LINE_WIDTH: 0.62488
G1 F3300
M204 S500
G1 X135.195 Y121.449 E.13063
G3 X134.399 Y121.477 I-.536 J-3.994 E.03782
G1 X137.428 Y124.51 E.20328
G2 X138.522 Y126.422 I31.617 J-16.818 E.10449
G1 X133.511 Y121.411 E.33606
G1 X132.744 Y121.273 E.03695
G1 X132.465 Y121.189 E.01381
G1 X139.926 Y128.65 E.50038
G2 X141.95 Y131.497 I46.496 J-30.913 E.16569
G1 X131.119 Y120.666 E.72639
G3 X129.564 Y119.643 I3.336 J-6.769 E.08851
G1 X128.945 Y119.053 E.04054
G1 X128.405 Y118.404 E.04006
G1 X127.106 Y116.653 E.1034
G1 X126.55 Y116.097 E.03728
G1 X126.169 Y116.539 E.02768
G1 X151.354 Y141.724 E1.68905
G1 X150.531 Y141.724 E.03903
G1 X125.783 Y116.976 E1.65973
G1 X125.39 Y117.405 E.02762
G1 X149.708 Y141.724 E1.63095
G1 X148.885 Y141.724 E.03903
G1 X124.996 Y117.835 E1.60216
G1 X124.602 Y118.264 E.02762
G1 X148.062 Y141.724 E1.57338
G1 X147.239 Y141.724 E.03903
G1 X124.198 Y118.683 E1.54529
G3 X123.792 Y119.099 I-10.057 J-9.401 E.0276
G1 X146.416 Y141.724 E1.51734
G1 X145.593 Y141.724 E.03903
G1 X123.378 Y119.508 E1.48991
G1 X122.964 Y119.918 E.0276
G1 X144.77 Y141.724 E1.46247
G1 X143.947 Y141.724 E.03903
G1 X122.54 Y120.317 E1.43566
G1 X122.116 Y120.716 E.02761
G1 X143.124 Y141.724 E1.40894
G1 X142.301 Y141.724 E.03903
G1 X121.685 Y121.108 E1.38262
G1 X121.238 Y121.484 E.0277
G1 X141.478 Y141.724 E1.35744
G1 X140.655 Y141.724 E.03903
G1 X120.79 Y121.859 E1.33227
G1 X120.342 Y122.234 E.0277
G1 X139.832 Y141.724 E1.3071
G1 X139.009 Y141.724 E.03903
G1 X119.895 Y122.61 E1.28193
G1 X119.447 Y122.985 E.0277
M73 P39 R11
G1 X138.186 Y141.724 E1.25675
G1 X137.363 Y141.724 E.03903
G1 X118.999 Y123.36 E1.23158
G1 X118.552 Y123.736 E.0277
G1 X136.54 Y141.724 E1.20641
G1 X135.717 Y141.724 E.03903
G1 X118.097 Y124.104 E1.18169
G1 X117.626 Y124.456 E.02788
G1 X134.894 Y141.724 E1.15807
G1 X134.071 Y141.724 E.03903
G1 X117.155 Y124.809 E1.13445
G3 X116.679 Y125.156 I-6.29 J-8.128 E.02794
G1 X133.248 Y141.724 E1.11118
G1 X132.425 Y141.724 E.03903
G1 X116.197 Y125.496 E1.08837
G3 X115.712 Y125.834 I-5.721 J-7.68 E.02804
G1 X131.602 Y141.724 E1.06568
G1 X130.779 Y141.724 E.03903
G1 X115.216 Y126.161 E1.04376
G1 X114.72 Y126.488 E.02817
G1 X129.956 Y141.724 E1.02183
G1 X129.133 Y141.724 E.03903
G1 X114.224 Y126.815 E.9999
G3 X113.716 Y127.13 I-4.251 J-6.28 E.02835
G1 X128.31 Y141.724 E.97876
G1 X127.487 Y141.724 E.03903
G1 X113.203 Y127.44 E.95798
G1 X112.69 Y127.75 E.02843
G1 X126.664 Y141.724 E.9372
G1 X125.841 Y141.724 E.03903
G1 X112.176 Y128.06 E.91641
G3 X111.652 Y128.358 I-4.747 J-7.749 E.02863
G1 X125.018 Y141.724 E.89642
G1 X124.195 Y141.724 E.03903
G1 X111.124 Y128.653 E.87662
G3 X110.588 Y128.941 I-4.966 J-8.608 E.02883
G1 X123.372 Y141.724 E.85734
G1 X122.549 Y141.724 E.03903
G1 X110.047 Y129.222 E.83846
G1 X109.505 Y129.504 E.02894
G1 X121.726 Y141.724 E.81958
G1 X120.903 Y141.724 E.03903
M73 P40 R11
G1 X108.961 Y129.782 E.80091
G1 X108.405 Y130.049 E.02924
G1 X120.08 Y141.724 E.78299
G1 X119.257 Y141.724 E.03903
G1 X107.849 Y130.316 E.76507
G3 X107.28 Y130.57 I-4.442 J-9.197 E.02956
G1 X118.434 Y141.724 E.74805
G1 X117.611 Y141.724 E.03903
G1 X106.71 Y130.823 E.73109
G1 X106.14 Y131.076 E.02958
G1 X116.788 Y141.724 E.71412
G1 X115.965 Y141.724 E.03903
G1 X105.559 Y131.318 E.69788
G1 X105.082 Y131.513 E.02442
G3 X105.246 Y131.829 I-.57 J.497 E.01706
G1 X115.142 Y141.724 E.66363
G1 X114.319 Y141.724 E.03903
G1 X105.431 Y132.836 E.59608
G1 X105.431 Y133.659 E.03903
G1 X113.496 Y141.724 E.54088
G1 X112.673 Y141.724 E.03903
G1 X105.431 Y134.482 E.48569
G1 X105.431 Y135.305 E.03903
G1 X111.85 Y141.724 E.43049
G1 X111.027 Y141.724 E.03903
G1 X105.431 Y136.128 E.3753
G1 X105.431 Y136.951 E.03903
G1 X110.204 Y141.724 E.3201
G1 X109.381 Y141.724 E.03903
G1 X105.431 Y137.774 E.26491
G1 X105.431 Y138.597 E.03903
G1 X108.558 Y141.724 E.20971
G1 X107.735 Y141.724 E.03903
G1 X105.431 Y139.42 E.15451
G1 X105.431 Y140.243 E.03903
G1 X106.912 Y141.724 E.09932
G1 X106.089 Y141.724 E.03903
G1 X105.171 Y140.807 E.06154
; CHANGE_LAYER
; Z_HEIGHT: 0.36
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F3300
G1 X105.878 Y141.514 E-.38
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
G1 X104.554 Y132.184
G1 Z.36
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.667 Y132.513 E.01331
G3 X104.714 Y133.652 I-6.582 J.842 E.04354
G1 X104.714 Y141.652 E.30543
G1 X104.619 Y142.253 E.02324
G1 X104.511 Y142.443 E.00836
G1 X154.069 Y142.443 E1.89208
G3 X143.803 Y132.701 I32.295 J-44.313 E.54184
G3 X136.271 Y120.545 I41.773 J-34.295 E.5476
G1 X135.453 Y120.705 E.0318
G3 X133.274 Y120.651 I-.889 J-8.192 E.08347
G3 X130.59 Y119.528 I1.611 J-7.613 E.11175
G3 X129.692 Y118.788 I8.051 J-10.703 E.04443
G1 X129.119 Y118.147 E.03283
G3 X127.853 Y116.453 I124.22 J-94.131 E.08074
G1 X126.662 Y114.847 E.07636
G3 X121.362 Y120.438 I-45.174 J-37.506 E.29433
G3 X119.312 Y122.16 I-440.063 J-521.956 E.10223
G1 X117.779 Y123.445 E.07636
G3 X103.708 Y131.278 I-33.38 J-43.411 E.61709
G1 X104.256 Y131.684 E.02605
G1 X104.522 Y132.1 E.01885
G1 X104.016 Y132.407 F36000
G1 F13446.369
G1 X104.095 Y132.641 E.00944
G3 X104.128 Y133.652 I-6.401 J.714 E.03863
G1 X104.128 Y141.652 E.30543
G1 X104.062 Y142.072 E.01626
G1 X103.791 Y142.549 E.02094
; LINE_WIDTH: 0.639256
G1 F13018.251
G1 X103.246 Y142.938 E.02641
; LINE_WIDTH: 0.654016
G1 F12708.173
G1 X103.003 Y143.012 E.01027
G1 X104.009 Y143.029 E.04066
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.409 Y143.029 E.01527
G1 X155.749 Y143.029 E1.96009
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00187
G3 X144.254 Y132.328 I29.8 J-43.95 E.59951
G3 X136.596 Y119.828 I41.805 J-34.21 E.56142
G1 X136.064 Y119.995 E.02127
G1 X135.346 Y120.129 E.02789
G3 X133.382 Y120.076 I-.778 J-7.519 E.07522
G1 X132.578 Y119.868 E.03172
G3 X130.901 Y119.031 I2.913 J-7.935 E.0717
G3 X130.104 Y118.373 I9.095 J-11.816 E.03947
G1 X129.557 Y117.759 E.03138
G3 X127.875 Y115.499 I226.2 J-170.189 E.10757
G1 X126.683 Y113.893 E.07636
G3 X120.964 Y120.009 I-42.541 J-34.048 E.31999
G3 X118.936 Y121.711 I-602.419 J-715.695 E.10112
G1 X117.403 Y122.996 E.07636
G3 X104.25 Y130.442 I-33.041 J-43.022 E.57893
G1 X104.064 Y130.514 E.00764
G1 X103.69 Y130.657 E.01527
G1 F12126.701
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.668064
G1 F10800.307
G1 X103.232 Y130.857 E.00419
; LINE_WIDTH: 0.716132
G1 F10476.324
G1 X103.148 Y130.913 E.00451
; LINE_WIDTH: 0.7642
G1 F10157.248
G1 X103.063 Y130.969 E.00482
; LINE_WIDTH: 0.812268
G1 F9843.106
G1 X102.979 Y131.025 E.00514
; LINE_WIDTH: 0.860336
G1 F9533.899
G1 X102.894 Y131.081 E.00546
; LINE_WIDTH: 0.908404
G1 F9009.597
G1 X102.81 Y131.137 E.00578
; LINE_WIDTH: 0.956472
G1 F8539.955
G1 X102.725 Y131.193 E.0061
; LINE_WIDTH: 1.00454
G1 F8116.851
G1 X102.641 Y131.249 E.00641
; LINE_WIDTH: 1.05261
G1 F7733.69
G1 X102.556 Y131.306 E.00673
; LINE_WIDTH: 1.10068
G1 F7385.075
G1 X102.472 Y131.362 E.00705
G1 X102.552 Y131.393 E.006
; LINE_WIDTH: 1.05261
G1 F7733.69
G1 X102.633 Y131.424 E.00573
; LINE_WIDTH: 1.00454
G1 F8116.851
G1 X102.713 Y131.456 E.00546
; LINE_WIDTH: 0.956472
G1 F8539.955
G1 X102.794 Y131.487 E.00519
; LINE_WIDTH: 0.908404
G1 F9009.597
G1 X102.874 Y131.518 E.00492
; LINE_WIDTH: 0.860336
G1 F9533.899
G1 X102.955 Y131.55 E.00465
; LINE_WIDTH: 0.812268
G1 F10122.994
G1 X103.035 Y131.581 E.00438
; LINE_WIDTH: 0.7642
G1 F10393.864
G1 X103.115 Y131.612 E.00411
; LINE_WIDTH: 0.716132
G1 F10668.34
G1 X103.196 Y131.644 E.00384
; LINE_WIDTH: 0.668064
G1 F10946.374
G1 X103.276 Y131.675 E.00357
; LINE_WIDTH: 0.619996
G1 F12281.448
G1 X103.6 Y131.91 E.01527
G1 F13179.467
G1 X103.808 Y132.061 E.00981
G1 F13446.369
G1 X103.982 Y132.324 E.01205
G1 X103.456 Y132.628 F36000
G1 F13446.369
G1 X103.524 Y132.769 E.006
G3 X103.543 Y135.652 I-96.676 J2.071 E.11005
G1 X103.543 Y141.652 E.22907
G1 X103.505 Y141.892 E.00928
G1 X103.35 Y142.164 E.01195
G1 X103.037 Y142.381 E.01454
G1 X102.764 Y142.43 E.0106
G1 X102.622 Y142.417 E.00544
G1 X99.542 Y141.847 E.11961
G3 X99.507 Y143.615 I-20.481 J.474 E.06754
G1 X156.235 Y143.615 E2.16585
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.705 Y131.955 I29.325 J-43.871 E.60772
G3 X136.912 Y119.083 I40.814 J-33.509 E.57645
G1 X136.093 Y119.382 E.03328
G3 X134.392 Y119.594 I-1.769 J-7.268 E.06557
G3 X133.436 Y119.488 I1.018 J-13.603 E.03673
G1 X132.716 Y119.299 E.02841
G3 X131.168 Y118.505 I2.785 J-7.341 E.06654
G1 X130.564 Y118 E.03008
G1 X129.996 Y117.371 E.03236
G3 X129.081 Y116.142 I204.85 J-153.523 E.05848
G1 X126.698 Y112.93 E.15272
G3 X120.566 Y119.579 I-44.004 J-34.425 E.3457
G3 X118.559 Y121.263 I-1019.602 J-1213.423 E.10002
G1 X117.027 Y122.547 E.07636
G3 X99.509 Y131.447 I-32.587 J-42.457 E.75446
G2 X99.541 Y132.744 I114.295 J-2.228 E.04957
G1 X102.622 Y132.174 E.11964
G1 X103.07 Y132.223 E.01718
G1 X103.37 Y132.451 E.0144
G1 X103.417 Y132.547 E.00406
M204 S250
G1 X102.969 Y132.844 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G3 X102.99 Y135.652 I-78.321 J1.997 E.0889
G1 X102.99 Y141.652 E.18996
G1 X102.934 Y141.8 E.00502
G1 X102.764 Y141.877 E.00591
G1 X102.723 Y141.873 E.00131
G1 X98.99 Y141.182 E.12021
G1 X98.988 Y142.831 E.05219
G2 X98.937 Y144.167 I15.307 J1.26 E.04237
G1 X156.695 Y144.167 E1.82863
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02532
G3 X145.128 Y131.598 I28.615 J-43.5 E.5106
G3 X137.192 Y118.315 I40.31 J-33.096 E.49169
G1 X136.612 Y118.616 E.02069
G1 X135.913 Y118.859 E.02343
G1 X135.137 Y119.01 E.02501
G1 X134.371 Y119.041 E.02427
M73 P41 R11
G1 X133.609 Y118.959 E.02427
G1 X132.868 Y118.767 E.02426
G1 X132.341 Y118.544 E.01811
G1 X131.506 Y118.067 E.03044
G1 X130.92 Y117.577 E.02419
G1 X130.41 Y117.004 E.02427
G3 X129.087 Y115.222 I970.717 J-722.049 E.07027
G1 X126.704 Y112.01 E.12664
G3 X121.199 Y118.226 I-44.205 J-33.598 E.26313
G3 X119.328 Y119.897 I-15.787 J-15.799 E.07946
G1 X116.672 Y122.124 E.10975
G3 X98.936 Y131.038 I-32.254 J-42.072 E.63217
G3 X98.989 Y133.409 I-64.189 J2.633 E.07509
G1 X102.723 Y132.717 E.12022
G1 X102.853 Y132.732 E.00413
G1 X102.904 Y132.781 E.00225
; WIPE_START
M204 S10000
G1 X102.973 Y133.779 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X100.84 Y141.107 Z.76 F36000
G1 X100.344 Y142.812 Z.76
G1 Z.36
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.05452
G1 F7719.226
G1 X100.521 Y142.828 E.0118
; LINE_WIDTH: 1.02208
G1 F7972.746
G1 X100.77 Y142.851 E.0161
; LINE_WIDTH: 0.976359
G1 F8359.673
G1 X101.019 Y142.874 E.01535
; LINE_WIDTH: 0.930641
G1 F8786.072
G1 X101.268 Y142.896 E.01461
; LINE_WIDTH: 0.884924
G1 F9258.308
G1 X101.517 Y142.919 E.01386
; LINE_WIDTH: 0.839206
G1 F9784.19
G1 X101.766 Y142.942 E.01312
; LINE_WIDTH: 0.793489
G1 F10373.411
G1 X102.015 Y142.965 E.01237
; LINE_WIDTH: 0.747771
G1 F11038.147
G1 X102.264 Y142.988 E.01163
; LINE_WIDTH: 0.702054
G1 F11793.912
G1 X102.512 Y143.011 E.01088
; LINE_WIDTH: 0.656336
G1 F12660.773
G1 X103.003 Y143.012 E.01988
; WIPE_START
G1 X102.512 Y143.011 E-.18633
G1 X102.264 Y142.988 E-.095
G1 X102.015 Y142.965 E-.095
G1 X102.005 Y142.964 E-.00367
; WIPE_END
G1 E-.02 F1800
G1 X100.759 Y135.434 Z.76 F36000
G1 X100.181 Y131.946 Z.76
G1 Z.36
G1 E.4 F1800
; LINE_WIDTH: 0.82509
G1 F9958.858
G1 X100.501 Y131.867 E.01699
; LINE_WIDTH: 0.864883
G1 F9481.708
G1 X100.821 Y131.787 E.01785
; LINE_WIDTH: 0.904676
G1 F9048.189
G1 X101.141 Y131.708 E.0187
; LINE_WIDTH: 0.908976
G1 F9003.705
G1 X101.172 Y131.7 E.00185
; LINE_WIDTH: 0.954826
G1 F8555.227
G1 X101.485 Y131.619 E.01935
; LINE_WIDTH: 1.00068
G1 F8149.306
G1 X101.797 Y131.538 E.02031
; LINE_WIDTH: 1.04653
G1 F7780.16
G1 X102.109 Y131.456 E.02128
; LINE_WIDTH: 1.09238
G1 F7443.008
G1 X102.421 Y131.375 E.02224
; LINE_WIDTH: 1.10068
G1 F7385.075
G1 X102.472 Y131.362 E.00369
; WIPE_START
G1 X102.421 Y131.375 E-.02015
G1 X102.109 Y131.456 E-.12253
G1 X101.797 Y131.538 E-.12253
G1 X101.504 Y131.614 E-.11479
; WIPE_END
G1 E-.02 F1800
G1 X106.339 Y130.463 Z.76 F36000
G1 Z.36
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.627336
G1 F13279.934
G1 X104.951 Y131.851 E.07588
G1 X104.982 Y131.9 E.00227
G1 X105.154 Y132.405 E.02059
G1 X105.162 Y132.478 E.00286
G1 X107.344 Y130.297 E.11926
G1 X108.925 Y129.554 E.06752
G1 X105.212 Y133.267 E.20298
G1 X105.212 Y134.106 E.03242
G1 X110.69 Y128.627 E.29952
G2 X112.607 Y127.549 I-15.207 J-29.28 E.08503
G1 X105.212 Y134.944 E.4043
G1 X105.212 Y135.783 E.03242
G1 X114.872 Y126.122 E.52815
G2 X117.718 Y124.115 I-27.5 J-42.011 E.13465
G1 X105.212 Y136.622 E.68374
G1 X105.212 Y137.46 E.03242
G1 X126.801 Y115.871 E1.1803
G1 X127.158 Y116.352 E.02317
G1 X105.212 Y138.299 E1.19983
G1 X105.212 Y139.137 E.03242
G1 X127.516 Y116.834 E1.21935
G1 X127.873 Y117.315 E.02317
G1 X105.212 Y139.976 E1.23888
G1 X105.212 Y140.815 E.03242
G1 X128.23 Y117.796 E1.25841
G1 X128.587 Y118.278 E.02317
G1 X105.212 Y141.653 E1.27794
G3 X105.168 Y141.945 I-.946 J.007 E.01146
G1 X105.758 Y141.945 E.02283
G1 X128.975 Y118.729 E1.26924
G1 X129.376 Y119.166 E.02295
G1 X106.597 Y141.945 E1.24533
G1 X107.436 Y141.945 E.03242
G1 X129.829 Y119.552 E1.22426
G1 X130.19 Y119.858 E.01828
G1 X130.296 Y119.924 E.00483
G1 X108.274 Y141.945 E1.20393
G1 X109.113 Y141.945 E.03242
G1 X130.815 Y120.243 E1.18647
G2 X131.363 Y120.534 I1.212 J-1.617 E.02406
G1 X109.951 Y141.945 E1.17055
G1 X110.79 Y141.945 E.03242
G1 X131.953 Y120.782 E1.15699
G2 X132.585 Y120.989 I1.058 J-2.164 E.02578
G1 X111.629 Y141.945 E1.14568
G1 X112.467 Y141.945 E.03242
G1 X133.263 Y121.15 E1.13691
G2 X134.012 Y121.239 I.983 J-5.031 E.02918
G1 X113.306 Y141.945 E1.132
G1 X114.145 Y141.945 E.03242
G1 X134.843 Y121.247 E1.13158
G2 X135.778 Y121.151 I-.05 J-5.084 E.03638
G1 X114.983 Y141.945 E1.13684
G1 X115.822 Y141.945 E.03242
G1 X136.201 Y121.566 E1.11413
G1 X136.472 Y122.134 E.02432
G1 X116.661 Y141.945 E1.0831
G1 X117.499 Y141.945 E.03242
G1 X136.743 Y122.701 E1.05207
G2 X137.024 Y123.26 I7.862 J-3.603 E.02415
G1 X118.338 Y141.945 E1.02156
G1 X119.176 Y141.945 E.03242
G1 X137.312 Y123.81 E.99145
G1 X137.599 Y124.361 E.02402
G1 X120.015 Y141.945 E.96134
G1 X120.854 Y141.945 E.03242
G1 X137.899 Y124.9 E.93187
G1 X138.2 Y125.438 E.02382
G1 X121.692 Y141.945 E.90245
G1 X122.531 Y141.945 E.03242
G1 X138.51 Y125.966 E.87357
G1 X138.824 Y126.491 E.02363
G1 X123.37 Y141.945 E.84491
G1 X124.208 Y141.945 E.03242
G1 X139.139 Y127.015 E.81626
G2 X139.462 Y127.53 I7.252 J-4.185 E.02353
G1 X125.047 Y141.945 E.78807
G1 X125.885 Y141.945 E.03242
G1 X139.792 Y128.039 E.76029
G1 X140.123 Y128.547 E.02343
G1 X126.724 Y141.945 E.73251
G1 X127.563 Y141.945 E.03242
G1 X140.458 Y129.051 E.70496
G1 X140.803 Y129.544 E.02328
G1 X128.401 Y141.945 E.67798
G1 X129.24 Y141.945 E.03242
G1 X141.148 Y130.038 E.651
G2 X141.5 Y130.524 I8.084 J-5.485 E.02322
G1 X130.079 Y141.945 E.62441
G1 X130.917 Y141.945 E.03242
G1 X141.858 Y131.004 E.59816
G1 X142.217 Y131.484 E.02316
G1 X131.756 Y141.945 E.5719
G1 X132.594 Y141.945 E.03242
G1 X142.584 Y131.956 E.54611
G1 X142.953 Y132.426 E.02309
G1 X133.433 Y141.945 E.52043
G1 X134.272 Y141.945 E.03242
G1 X143.322 Y132.896 E.49476
G2 X143.7 Y133.356 I9.305 J-7.249 E.02304
G1 X135.11 Y141.945 E.46957
G1 X135.949 Y141.945 E.03242
G1 X144.079 Y133.815 E.44449
G2 X144.466 Y134.267 I7.354 J-5.906 E.023
G1 X136.788 Y141.945 E.41979
G1 X137.626 Y141.945 E.03242
G1 X144.859 Y134.712 E.39543
G1 X145.252 Y135.158 E.02297
G1 X138.465 Y141.945 E.37107
G1 X139.304 Y141.945 E.03242
G1 X145.646 Y135.603 E.34675
G1 X146.053 Y136.035 E.02293
G1 X140.142 Y141.945 E.32312
G1 X140.981 Y141.945 E.03242
G1 X146.459 Y136.467 E.29949
G2 X146.874 Y136.891 I8.099 J-7.509 E.02293
G1 X141.819 Y141.945 E.27631
G1 X142.658 Y141.945 E.03242
G1 X147.291 Y137.312 E.2533
G2 X147.71 Y137.732 I6.429 J-5.999 E.02293
G1 X143.497 Y141.945 E.23036
G1 X144.335 Y141.945 E.03242
G1 X148.142 Y138.139 E.20811
G1 X148.574 Y138.546 E.02293
G1 X145.174 Y141.945 E.18587
G1 X146.013 Y141.945 E.03242
G1 X149.006 Y138.952 E.16363
G2 X149.445 Y139.351 I6.499 J-6.724 E.02296
G1 X146.851 Y141.945 E.14182
G1 X147.69 Y141.945 E.03242
G1 X149.891 Y139.745 E.12031
G2 X150.339 Y140.135 I8.062 J-8.794 E.02298
G1 X148.528 Y141.945 E.09896
G1 X149.367 Y141.945 E.03242
G1 X150.794 Y140.518 E.07802
G2 X151.25 Y140.901 I8.864 J-10.108 E.02301
G1 X150.206 Y141.945 E.05711
G1 X151.044 Y141.945 E.03242
G1 X151.715 Y141.274 E.03669
G2 X152.183 Y141.646 I8.807 J-10.611 E.02308
G1 X151.619 Y142.209 E.0308
; CHANGE_LAYER
; Z_HEIGHT: 0.52
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13279.934
G1 X152.183 Y141.646 E-.3028
G1 X152.024 Y141.519 E-.0772
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
G1 X103.711 Y131.277
G1 Z.52
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.257 Y131.68 E.02592
G1 X104.531 Y132.108 E.01941
G1 X104.668 Y132.508 E.01613
G3 X104.716 Y133.654 I-6.559 J.847 E.04385
G1 X104.716 Y141.654 E.30543
G1 X104.621 Y142.255 E.02324
G1 X104.514 Y142.443 E.00825
G1 X154.07 Y142.443 E1.89197
G3 X143.803 Y132.701 I31.769 J-43.761 E.54192
G3 X136.271 Y120.545 I41.788 J-34.304 E.54759
G1 X135.451 Y120.705 E.03191
G3 X133.273 Y120.651 I-.888 J-8.101 E.08342
G3 X130.592 Y119.529 I1.615 J-7.62 E.11162
G3 X129.621 Y118.715 I6.592 J-8.848 E.04838
G1 X129.119 Y118.147 E.02896
G3 X127.853 Y116.453 I124.925 J-94.655 E.08073
G1 X126.662 Y114.847 E.07636
G3 X121.362 Y120.439 I-44.619 J-36.98 E.29437
G3 X119.312 Y122.16 I-445.433 J-528.37 E.1022
G1 X117.779 Y123.445 E.07636
G3 X103.795 Y131.245 I-33.381 J-43.413 E.61352
G1 X103.39 Y131.754 F36000
G1 F13446.369
G1 X103.809 Y132.058 E.01976
G3 X104.097 Y132.637 I-1.043 J.879 E.02493
G3 X104.13 Y133.654 I-6.376 J.719 E.0389
G1 X104.13 Y141.654 E.30543
G1 X104.064 Y142.075 E.01626
G1 X103.793 Y142.552 E.02094
; LINE_WIDTH: 0.637836
G1 F13048.882
G1 X103.248 Y142.94 E.02634
; LINE_WIDTH: 0.651496
G1 F12760.064
G1 X103.005 Y143.013 E.01022
G1 X104.01 Y143.029 E.04045
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.41 Y143.029 E.01527
G1 X155.749 Y143.029 E1.96006
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.043 J-44.213 E.59949
G3 X136.596 Y119.828 I41.244 J-33.866 E.56145
G1 X136.064 Y119.995 E.02128
G1 X135.343 Y120.13 E.02799
G1 X134.411 Y120.179 E.03563
G3 X133.284 Y120.054 I.63 J-10.831 E.04331
G1 X132.575 Y119.867 E.028
G3 X130.827 Y118.981 I2.891 J-7.874 E.075
G1 X130.186 Y118.447 E.03187
G1 X129.557 Y117.759 E.03557
G3 X127.875 Y115.499 I227.482 J-171.138 E.10756
G1 X126.683 Y113.893 E.07636
G3 X120.964 Y120.009 I-42.542 J-34.049 E.32001
G3 X118.936 Y121.712 I-615.113 J-730.839 E.10111
G1 X117.403 Y122.996 E.07636
G3 X104.25 Y130.443 I-33.042 J-43.026 E.57893
G1 X103.69 Y130.657 E.0229
G1 F12134.13
G1 X103.317 Y130.801 E.01527
; LINE_WIDTH: 0.667892
G1 F10807.319
G1 X103.232 Y130.856 E.00418
; LINE_WIDTH: 0.715788
G1 F10484.109
G1 X103.148 Y130.912 E.00449
; LINE_WIDTH: 0.763684
G1 F10165.762
G1 X103.064 Y130.968 E.00481
; LINE_WIDTH: 0.81158
G1 F9852.349
G1 X102.98 Y131.024 E.00512
; LINE_WIDTH: 0.859476
G1 F9543.835
G1 X102.895 Y131.08 E.00544
; LINE_WIDTH: 0.907372
G1 F9020.247
G1 X102.811 Y131.136 E.00576
; LINE_WIDTH: 0.955268
G1 F8551.12
G1 X102.727 Y131.192 E.00607
; LINE_WIDTH: 1.00316
G1 F8128.378
G1 X102.643 Y131.248 E.00639
; LINE_WIDTH: 1.05106
G1 F7745.465
G1 X102.558 Y131.304 E.0067
; LINE_WIDTH: 1.09896
G1 F7397.005
G1 X102.474 Y131.36 E.00702
G1 X102.555 Y131.391 E.00598
; LINE_WIDTH: 1.05106
G1 F7745.465
G1 X102.635 Y131.422 E.00571
; LINE_WIDTH: 1.00316
G1 F8128.378
G1 X102.715 Y131.454 E.00544
; LINE_WIDTH: 0.955268
G1 F8551.12
G1 X102.795 Y131.485 E.00517
; LINE_WIDTH: 0.907372
G1 F9020.247
G1 X102.876 Y131.516 E.0049
; LINE_WIDTH: 0.859476
G1 F9543.835
G1 X102.956 Y131.547 E.00463
; LINE_WIDTH: 0.81158
G1 F10131.955
G1 X103.036 Y131.578 E.00436
; LINE_WIDTH: 0.763684
G1 F10402.341
G1 X103.117 Y131.61 E.00409
; LINE_WIDTH: 0.715788
G1 F10676.289
G1 X103.197 Y131.641 E.00383
; LINE_WIDTH: 0.667892
G1 F10953.797
G1 X103.277 Y131.672 E.00356
; LINE_WIDTH: 0.619996
G1 F11115.728
G1 X103.317 Y131.701 E.0019
G1 X103.119 Y132.257 F36000
G1 F13446.369
G1 X103.372 Y132.448 E.01209
G1 X103.526 Y132.766 E.01347
G3 X103.545 Y135.654 I-95.515 J2.075 E.11029
G1 X103.545 Y141.654 E.22907
M73 P42 R11
G1 X103.507 Y141.894 E.00928
G1 X103.352 Y142.166 E.01195
G1 X103.039 Y142.383 E.01454
G1 X102.766 Y142.433 E.0106
G1 X102.625 Y142.419 E.00544
G1 X99.544 Y141.849 E.11961
G3 X99.508 Y143.615 I-19.82 J.473 E.06744
G1 X156.235 Y143.615 E2.16582
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.316 J-43.861 E.60773
G3 X136.912 Y119.083 I41.189 J-33.736 E.57641
G1 X136.093 Y119.382 E.03328
G3 X132.724 Y119.301 I-1.548 J-5.706 E.13044
G3 X131.17 Y118.507 I2.752 J-7.304 E.06678
G1 X130.563 Y117.998 E.03024
G1 X129.996 Y117.371 E.03228
G3 X129.081 Y116.142 I207.206 J-155.269 E.05848
G1 X126.698 Y112.93 E.15272
G3 X120.566 Y119.579 I-44.006 J-34.427 E.34571
G3 X118.559 Y121.263 I-1039.681 J-1237.376 E.10002
G1 X117.027 Y122.547 E.07636
G3 X99.508 Y131.447 I-32.589 J-42.461 E.75448
G2 X99.543 Y132.742 I166.945 J-3.94 E.04947
G1 X102.625 Y132.171 E.11964
G1 X103.042 Y132.217 E.01603
M204 S250
G1 X102.855 Y132.729 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X102.971 Y132.841 E.0051
G3 X102.992 Y135.654 I-77.977 J2 E.08907
G1 X102.992 Y141.654 E.18996
G1 X102.936 Y141.803 E.00502
G1 X102.766 Y141.88 E.00591
G1 X102.725 Y141.876 E.00131
G1 X98.992 Y141.185 E.12021
G1 X98.991 Y142.822 E.05184
G2 X98.937 Y144.167 I15.148 J1.278 E.04264
G1 X156.695 Y144.167 E1.82862
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.739 J-43.636 E.51059
G3 X137.192 Y118.315 I40.807 J-33.392 E.49164
G1 X136.612 Y118.616 E.02069
G1 X135.913 Y118.859 E.02343
G1 X135.135 Y119.01 E.02509
G1 X134.369 Y119.041 E.02426
G1 X133.607 Y118.959 E.02427
G1 X132.865 Y118.766 E.02426
G1 X132.344 Y118.545 E.01794
G1 X131.508 Y118.068 E.03046
G1 X130.918 Y117.575 E.02433
G1 X130.41 Y117.004 E.0242
G3 X129.087 Y115.222 I987.256 J-734.319 E.07027
G1 X126.704 Y112.01 E.12664
G3 X121.199 Y118.226 I-44.208 J-33.6 E.26313
G3 X119.328 Y119.897 I-15.785 J-15.797 E.07944
G1 X116.671 Y122.124 E.10977
G3 X98.936 Y131.038 I-32.256 J-42.076 E.63217
G3 X98.991 Y133.406 I-61.729 J2.632 E.07501
G1 X102.725 Y132.715 E.12022
G1 X102.765 Y132.719 E.00127
; WIPE_START
M204 S10000
G1 X102.971 Y132.841 E-.09071
G1 X102.984 Y133.602 E-.28929
; WIPE_END
G1 E-.02 F1800
G1 X100.882 Y140.939 Z.92 F36000
G1 X100.345 Y142.813 Z.92
G1 Z.52
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.05224
G1 F7736.516
G1 X100.523 Y142.829 E.01186
; LINE_WIDTH: 1.01955
G1 F7993.22
G1 X100.772 Y142.852 E.01606
; LINE_WIDTH: 0.973827
G1 F8382.197
G1 X101.021 Y142.875 E.01531
; LINE_WIDTH: 0.928109
G1 F8810.968
G1 X101.27 Y142.898 E.01457
; LINE_WIDTH: 0.88239
G1 F9285.969
G1 X101.519 Y142.921 E.01382
; LINE_WIDTH: 0.836671
G1 F9815.103
G1 X101.768 Y142.943 E.01308
; LINE_WIDTH: 0.790953
G1 F10408.183
G1 X102.017 Y142.966 E.01233
; LINE_WIDTH: 0.745234
G1 F11077.547
G1 X102.266 Y142.989 E.01159
; LINE_WIDTH: 0.699515
G1 F11838.925
G1 X102.515 Y143.012 E.01084
; LINE_WIDTH: 0.653796
G1 F12712.686
G1 X103.005 Y143.013 E.01978
; WIPE_START
G1 X102.515 Y143.012 E-.18616
G1 X102.266 Y142.989 E-.095
G1 X102.017 Y142.966 E-.095
G1 X102.007 Y142.965 E-.00384
; WIPE_END
G1 E-.02 F1800
G1 X100.759 Y135.435 Z.92 F36000
G1 X100.181 Y131.945 Z.92
G1 Z.52
G1 E.4 F1800
; LINE_WIDTH: 0.82303
G1 F9984.87
G1 X100.501 Y131.866 E.01694
; LINE_WIDTH: 0.862823
G1 F9505.284
G1 X100.821 Y131.786 E.0178
; LINE_WIDTH: 0.902616
G1 F9069.655
G1 X101.141 Y131.707 E.01865
; LINE_WIDTH: 0.906876
G1 F9025.375
G1 X101.172 Y131.699 E.00184
; LINE_WIDTH: 0.953021
G1 F8572.036
G1 X101.486 Y131.617 E.01942
; LINE_WIDTH: 0.999166
G1 F8162.06
G1 X101.8 Y131.536 E.0204
; LINE_WIDTH: 1.04531
G1 F7789.51
G1 X102.114 Y131.454 E.02138
; LINE_WIDTH: 1.09146
G1 F7449.486
G1 X102.428 Y131.372 E.02235
; LINE_WIDTH: 1.09896
G1 F7397.005
G1 X102.474 Y131.36 E.00333
; WIPE_START
G1 X102.428 Y131.372 E-.01826
G1 X102.114 Y131.454 E-.12325
G1 X101.8 Y131.536 E-.12325
G1 X101.507 Y131.612 E-.11524
; WIPE_END
G1 E-.02 F1800
G1 X104.11 Y138.787 Z.92 F36000
G1 X104.95 Y141.104 Z.92
G1 Z.52
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.626726
G1 F13293.608
G1 X105.792 Y141.945 E.04594
G1 X106.629 Y141.945 E.03235
G1 X105.214 Y140.53 E.0773
G1 X105.214 Y139.692 E.03235
G1 X107.467 Y141.945 E.12305
G1 X108.305 Y141.945 E.03235
G1 X105.214 Y138.855 E.16881
G1 X105.214 Y138.017 E.03235
G1 X109.143 Y141.945 E.21456
G1 X109.98 Y141.945 E.03235
G1 X105.214 Y137.179 E.26031
G1 X105.214 Y136.341 E.03235
G1 X110.818 Y141.945 E.30607
G1 X111.656 Y141.945 E.03235
G1 X105.214 Y135.503 E.35182
G1 X105.214 Y134.666 E.03235
G1 X112.494 Y141.945 E.39757
G1 X113.331 Y141.945 E.03235
G1 X105.214 Y133.828 E.44333
G1 X105.214 Y132.99 E.03235
G1 X114.169 Y141.945 E.48908
G1 X115.007 Y141.945 E.03235
G1 X104.996 Y131.935 E.54672
G1 X104.691 Y131.428 E.02283
G1 X105.145 Y131.246 E.0189
G1 X115.845 Y141.945 E.58436
G1 X116.682 Y141.945 E.03235
G1 X105.742 Y131.005 E.59751
G2 X106.329 Y130.754 I-4.74 J-11.911 E.02465
G1 X117.52 Y141.945 E.6112
G1 X118.358 Y141.945 E.03235
G1 X106.912 Y130.499 E.62512
G1 X107.482 Y130.232 E.02432
G1 X119.196 Y141.945 E.63973
G1 X120.034 Y141.945 E.03235
G1 X108.052 Y129.964 E.65435
G1 X108.622 Y129.696 E.02432
G1 X120.871 Y141.945 E.66896
G1 X121.709 Y141.945 E.03235
G1 X109.186 Y129.422 E.68395
G1 X109.741 Y129.14 E.02407
G1 X122.547 Y141.945 E.69935
G1 X123.385 Y141.945 E.03235
G1 X110.287 Y128.848 E.71531
G1 X110.829 Y128.552 E.02384
G1 X124.222 Y141.945 E.73149
G1 X125.06 Y141.945 E.03235
G1 X111.37 Y128.255 E.74766
G1 X111.912 Y127.959 E.02384
G1 X125.898 Y141.945 E.76384
G1 X126.736 Y141.945 E.03235
G1 X112.439 Y127.649 E.78081
G1 X112.965 Y127.337 E.02362
G1 X127.573 Y141.945 E.7978
G1 X128.411 Y141.945 E.03235
G1 X113.482 Y127.016 E.81537
G1 X113.995 Y126.691 E.02345
G1 X129.249 Y141.945 E.8331
G1 X130.087 Y141.945 E.03235
G1 X114.508 Y126.366 E.85084
G2 X115.016 Y126.036 I-5.586 J-9.16 E.02339
G1 X130.924 Y141.945 E.86885
G1 X131.762 Y141.945 E.03235
G1 X115.516 Y125.699 E.88726
G2 X116.011 Y125.356 I-7.694 J-11.619 E.02325
G1 X132.6 Y141.945 E.906
G1 X133.438 Y141.945 E.03235
G1 X116.502 Y125.01 E.92492
G2 X116.989 Y124.659 I-6.482 J-9.496 E.02318
G1 X134.276 Y141.945 E.9441
G1 X135.113 Y141.945 E.03235
G1 X117.469 Y124.301 E.96363
G1 X117.949 Y123.943 E.02312
G1 X135.951 Y141.945 E.98317
G1 X136.789 Y141.945 E.03235
G1 X118.408 Y123.564 E1.00387
G1 X118.863 Y123.182 E.02297
G1 X137.627 Y141.945 E1.02474
G1 X138.464 Y141.945 E.03235
G1 X119.319 Y122.8 E1.0456
G1 X119.775 Y122.418 E.02297
G1 X139.302 Y141.945 E1.06646
G1 X140.14 Y141.945 E.03235
G1 X120.231 Y122.036 E1.08733
G1 X120.686 Y121.654 E.02297
G1 X140.978 Y141.945 E1.10819
G1 X141.815 Y141.945 E.03235
G1 X121.142 Y121.272 E1.12905
G1 X121.598 Y120.89 E.02297
G1 X142.653 Y141.945 E1.14991
G1 X143.491 Y141.945 E.03235
G1 X122.035 Y120.489 E1.1718
G1 X122.467 Y120.083 E.02289
G1 X144.329 Y141.945 E1.19397
G1 X145.167 Y141.945 E.03235
G1 X122.895 Y119.674 E1.21632
G1 X123.317 Y119.258 E.02288
G1 X146.004 Y141.945 E1.23906
G1 X146.842 Y141.945 E.03235
G1 X123.737 Y118.84 E1.26185
G1 X124.149 Y118.415 E.02288
G1 X147.68 Y141.945 E1.28508
G1 X148.518 Y141.945 E.03235
G1 X124.557 Y117.985 E1.30856
G1 X124.958 Y117.548 E.0229
G1 X149.355 Y141.945 E1.33242
G1 X150.193 Y141.945 E.03235
G1 X125.359 Y117.112 E1.35628
G1 X125.76 Y116.675 E.0229
G1 X151.031 Y141.945 E1.38013
G1 X151.869 Y141.945 E.03235
G1 X126.147 Y116.223 E1.40477
G1 X126.533 Y115.772 E.02295
G1 X127.289 Y116.528 E.04129
G1 X128.635 Y118.342 E.08724
G1 X129.256 Y119.054 E.03648
G1 X129.484 Y119.272 E.01217
M73 P43 R11
G1 X130.192 Y119.86 E.03556
G1 X131.13 Y120.437 E.04253
G1 X131.247 Y120.486 E.00487
G1 X142.133 Y131.372 E.59455
G3 X140.081 Y128.482 I43.712 J-33.215 E.13691
G1 X132.589 Y120.99 E.40915
G2 X133.634 Y121.198 I1.557 J-5.112 E.04121
G1 X138.661 Y126.224 E.27453
G3 X137.624 Y124.408 I29.031 J-17.778 E.0808
G1 X137.561 Y124.287 E.00526
G1 X134.533 Y121.258 E.16539
G2 X135.331 Y121.219 I.132 J-5.41 E.03089
G1 X137.275 Y123.163 E.10616
; CHANGE_LAYER
; Z_HEIGHT: 0.68
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13293.608
G1 X136.568 Y122.456 E-.38
; WIPE_END
G1 E-.02 F1800
;======== P2S layer_change gcode ==========
;===== 2026/05/15 ====





; update layer progress
M73 L4
M991 S0 P3 ;notify layer change


G17
M73 P43 R10
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
G1 X103.714 Y131.275
G1 Z.68
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.258 Y131.676 E.02581
G1 X104.535 Y132.109 E.01959
G1 X104.672 Y132.511 E.01624
G3 X104.718 Y133.657 I-6.656 J.843 E.04382
G1 X104.718 Y141.657 E.30543
G1 X104.623 Y142.258 E.02324
G1 X104.518 Y142.443 E.00814
G1 X154.069 Y142.443 E1.89181
G3 X143.802 Y132.701 I31.812 J-43.803 E.54191
G3 X136.271 Y120.545 I41.663 J-34.225 E.54757
G1 X135.453 Y120.705 E.0318
G3 X133.276 Y120.652 I-.89 J-8.174 E.08338
G3 X130.59 Y119.527 I1.608 J-7.611 E.11184
G3 X129.622 Y118.716 I6.584 J-8.836 E.04827
G1 X129.117 Y118.145 E.02909
G3 X127.854 Y116.454 I124.9 J-94.63 E.08058
G1 X126.662 Y114.848 E.07636
G3 X121.362 Y120.438 I-42.527 J-35.009 E.29435
G3 X119.312 Y122.16 I-438.424 J-520.002 E.10223
G1 X117.779 Y123.445 E.07636
G3 X103.798 Y131.244 I-33.375 J-43.401 E.6134
G1 X103.027 Y131.571 F36000
; LINE_WIDTH: 0.810988
G1 F10139.676
G1 X103.037 Y131.575 E.00057
; LINE_WIDTH: 0.76324
G1 F10409.405
G1 X103.117 Y131.607 E.00408
; LINE_WIDTH: 0.715492
G1 F10682.673
G1 X103.197 Y131.638 E.00381
; LINE_WIDTH: 0.667744
G1 F10959.495
G1 X103.277 Y131.669 E.00355
; LINE_WIDTH: 0.619996
G1 F12295.345
G1 X103.601 Y131.903 E.01527
G1 F13198.729
G1 X103.811 Y132.054 E.00986
G1 F13446.369
G1 X104.004 Y132.357 E.0137
G3 X104.1 Y132.638 I-1.236 J.578 E.01138
G3 X104.133 Y133.657 I-6.48 J.716 E.03893
G1 X104.133 Y141.657 E.30543
G1 X104.066 Y142.077 E.01626
G1 X103.795 Y142.554 E.02094
; LINE_WIDTH: 0.636416
G1 F13079.659
G1 X103.25 Y142.942 E.02627
; LINE_WIDTH: 0.648956
G1 F12812.796
G1 X103.007 Y143.014 E.01017
G1 X104.011 Y143.029 E.04025
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.411 Y143.029 E.01527
G1 X155.749 Y143.029 E1.96002
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.327 I29.803 J-43.952 E.59955
G3 X136.596 Y119.828 I41.127 J-33.793 E.56144
G1 X136.064 Y119.995 E.02128
G1 X135.346 Y120.129 E.02789
G1 X134.412 Y120.179 E.03572
G3 X133.284 Y120.054 I.629 J-10.828 E.04332
G1 X132.575 Y119.867 E.02801
G3 X130.901 Y119.031 I2.924 J-7.949 E.07159
G3 X130.055 Y118.321 I7.292 J-9.55 E.04218
G1 X129.556 Y117.757 E.02876
G3 X127.875 Y115.499 I227.737 J-171.322 E.10747
G1 X126.683 Y113.893 E.07636
G3 X120.964 Y120.009 I-42.54 J-34.047 E.31999
G3 X118.936 Y121.711 I-605.329 J-719.161 E.10112
G1 X117.403 Y122.996 E.07636
G3 X104.25 Y130.442 I-33.037 J-43.016 E.57893
G1 X104.063 Y130.515 E.00764
G1 X103.69 Y130.658 E.01527
G1 F12139.633
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.667744
G1 F10812.512
G1 X103.232 Y130.856 E.00416
; LINE_WIDTH: 0.715492
G1 F10490.175
G1 X103.148 Y130.912 E.00448
; LINE_WIDTH: 0.76324
G1 F10172.715
G1 X103.064 Y130.968 E.00479
; LINE_WIDTH: 0.810988
G1 F9860.107
G1 X102.98 Y131.024 E.0051
; LINE_WIDTH: 0.858736
G1 F9552.402
G1 X102.896 Y131.079 E.00542
; LINE_WIDTH: 0.906484
G1 F9029.433
G1 X102.812 Y131.135 E.00573
; LINE_WIDTH: 0.954232
G1 F8560.75
G1 X102.728 Y131.191 E.00605
; LINE_WIDTH: 1.00198
G1 F8138.324
G1 X102.644 Y131.247 E.00636
; LINE_WIDTH: 1.04973
G1 F7755.625
G1 X102.56 Y131.302 E.00667
; LINE_WIDTH: 1.09748
G1 F7407.303
G1 X102.476 Y131.358 E.00699
G1 X102.557 Y131.389 E.00595
; LINE_WIDTH: 1.04973
G1 F7755.625
G1 X102.637 Y131.42 E.00569
; LINE_WIDTH: 1.00198
G1 F8138.324
G1 X102.717 Y131.451 E.00542
; LINE_WIDTH: 0.954232
G1 F8560.75
G1 X102.797 Y131.482 E.00515
; LINE_WIDTH: 0.906484
G1 F9029.433
G1 X102.877 Y131.513 E.00488
; LINE_WIDTH: 0.858736
G1 F9552.402
G1 X102.943 Y131.539 E.00379
G1 X102.763 Y132.184 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X103.072 Y132.218 E.01188
G1 X103.374 Y132.445 E.01443
G1 X103.528 Y132.765 E.01358
G3 X103.547 Y135.657 I-98.076 J2.075 E.11039
G1 X103.547 Y141.657 E.22907
G1 X103.509 Y141.897 E.00928
G1 X103.355 Y142.169 E.01195
G1 X103.041 Y142.386 E.01454
G1 X102.768 Y142.435 E.0106
G1 X102.627 Y142.422 E.00544
G1 X99.546 Y141.852 E.11961
G3 X99.508 Y143.615 I-19.188 J.471 E.06735
G1 X156.235 Y143.615 E2.16578
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.705 Y131.954 I29.325 J-43.872 E.60776
G3 X136.912 Y119.083 I41.137 J-33.704 E.57637
G1 X136.093 Y119.382 E.03328
G3 X132.724 Y119.301 I-1.548 J-5.699 E.13045
G3 X131.168 Y118.505 I2.744 J-7.288 E.06687
G1 X130.562 Y117.998 E.03018
G1 X129.994 Y117.369 E.03234
G3 X129.081 Y116.142 I207.647 J-155.595 E.05839
G1 X126.698 Y112.93 E.15272
G3 X120.566 Y119.579 I-44.005 J-34.426 E.3457
G3 X118.559 Y121.263 I-1012.385 J-1204.812 E.10002
G1 X117.027 Y122.547 E.07636
G3 X99.507 Y131.447 I-32.528 J-42.34 E.75454
G2 X99.545 Y132.739 I280.114 J-7.658 E.04937
G1 X102.627 Y132.169 E.11964
G1 X102.673 Y132.174 E.00179
M204 S250
G1 X102.727 Y132.712 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X102.856 Y132.727 E.00411
G1 X102.973 Y132.839 E.00512
G3 X102.994 Y135.657 I-78.606 J2.001 E.08922
G1 X102.994 Y141.657 E.18996
G1 X102.938 Y141.805 E.00502
G1 X102.768 Y141.882 E.00591
G1 X102.727 Y141.878 E.00131
G1 X98.994 Y141.187 E.12021
G1 X98.993 Y142.821 E.05173
G2 X98.937 Y144.167 I14.814 J1.287 E.04268
G1 X156.695 Y144.167 E1.82862
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.613 J-43.498 E.51063
G3 X137.192 Y118.315 I40.596 J-33.265 E.49163
G1 X136.614 Y118.615 E.02062
G1 X135.913 Y118.859 E.02351
G1 X135.137 Y119.01 E.02501
G1 X134.369 Y119.041 E.02433
G1 X133.607 Y118.959 E.02427
G1 X132.865 Y118.766 E.02427
G1 X132.343 Y118.545 E.01794
G1 X131.506 Y118.067 E.03054
G1 X130.918 Y117.575 E.02427
G1 X130.409 Y117.003 E.02425
G3 X129.087 Y115.222 I989.264 J-735.801 E.0702
G1 X126.704 Y112.01 E.12664
G3 X121.199 Y118.227 I-42.097 J-31.731 E.26317
G3 X119.329 Y119.896 I-15.78 J-15.792 E.0794
G1 X116.672 Y122.124 E.10979
G3 X98.936 Y131.038 I-32.298 J-42.159 E.63216
G3 X98.994 Y133.404 I-59.512 J2.632 E.07494
G1 X102.639 Y132.729 E.11737
; WIPE_START
M204 S10000
G1 X102.856 Y132.727 E-.08272
G1 X102.973 Y132.839 E-.06143
G1 X102.984 Y133.459 E-.23585
; WIPE_END
G1 E-.02 F1800
G1 X103.002 Y141.092 Z1.08 F36000
G1 X103.007 Y143.014 Z1.08
G1 Z.68
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.651276
G1 F12764.614
G1 X102.517 Y143.013 E.01968
; LINE_WIDTH: 0.696993
G1 F11883.99
G1 X102.268 Y142.99 E.0108
; LINE_WIDTH: 0.742709
G1 F11117.034
G1 X102.019 Y142.968 E.01154
; LINE_WIDTH: 0.788425
G1 F10443.068
G1 X101.77 Y142.945 E.01229
; LINE_WIDTH: 0.834141
G1 F9846.151
G1 X101.521 Y142.922 E.01304
; LINE_WIDTH: 0.879857
G1 F9313.783
G1 X101.272 Y142.899 E.01378
; LINE_WIDTH: 0.925574
G1 F8836.029
G1 X101.023 Y142.876 E.01452
; LINE_WIDTH: 0.97129
G1 F8404.897
G1 X100.774 Y142.853 E.01527
; LINE_WIDTH: 1.01701
G1 F8013.881
G1 X100.525 Y142.83 E.01602
; LINE_WIDTH: 1.04996
G1 F7753.884
G1 X100.346 Y142.814 E.01193
; WIPE_START
G1 X100.525 Y142.83 E-.06845
G1 X100.774 Y142.853 E-.095
G1 X101.023 Y142.876 E-.095
G1 X101.272 Y142.899 E-.095
G1 X101.342 Y142.905 E-.02655
; WIPE_END
G1 E-.02 F1800
G1 X102.088 Y135.309 Z1.08 F36000
G1 X102.476 Y131.358 Z1.08
G1 Z.68
G1 E.4 F1800
; LINE_WIDTH: 1.09748
G1 F7407.303
G1 X102.411 Y131.376 E.00468
; LINE_WIDTH: 1.08682
G1 F7482.327
G1 X102.101 Y131.456 E.02197
; LINE_WIDTH: 1.0413
G1 F7820.569
G1 X101.791 Y131.537 E.02102
; LINE_WIDTH: 0.995776
G1 F8190.839
G1 X101.481 Y131.617 E.02007
; LINE_WIDTH: 0.950256
G1 F8597.913
G1 X101.172 Y131.698 E.01912
; LINE_WIDTH: 0.904736
G1 F9047.565
G1 X101.14 Y131.706 E.00183
; LINE_WIDTH: 0.900516
G1 F9091.645
G1 X100.821 Y131.785 E.0186
; LINE_WIDTH: 0.860736
G1 F9529.284
G1 X100.501 Y131.865 E.01775
; LINE_WIDTH: 0.820956
G1 F10011.188
G1 X100.181 Y131.944 E.01689
; WIPE_START
G1 X100.501 Y131.865 E-.12518
G1 X100.821 Y131.785 E-.12518
G1 X101.14 Y131.706 E-.12518
G1 X101.152 Y131.703 E-.00445
; WIPE_END
G1 E-.02 F1800
G1 X106.354 Y130.45 Z1.08 F36000
G1 Z.68
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.627306
G1 F13280.605
G1 X104.956 Y131.848 E.07642
G1 X105.159 Y132.403 E.02283
G1 X105.167 Y132.476 E.00285
G1 X107.344 Y130.299 E.11902
G2 X108.956 Y129.526 I-8.985 J-20.793 E.06913
G1 X105.216 Y133.266 E.20446
G1 X105.216 Y134.104 E.03242
G1 X110.677 Y128.644 E.29851
G2 X112.613 Y127.546 I-16.913 J-32.091 E.08605
G1 X105.216 Y134.943 E.40436
G1 X105.216 Y135.782 E.03242
G1 X114.879 Y126.118 E.52827
G2 X116.129 Y125.273 I-13.244 J-20.922 E.05833
G1 X116.327 Y125.509 E.01192
G1 X105.216 Y136.62 E.60741
G1 X105.216 Y137.459 E.03242
G1 X116.751 Y125.923 E.63061
G2 X117.198 Y126.247 I2.282 J-2.675 E.02133
G1 X117.241 Y126.272 E.00194
G1 X105.216 Y138.297 E.65737
G1 X105.216 Y139.136 E.03242
G1 X117.791 Y126.561 E.68743
G2 X118.405 Y126.785 I1.513 J-3.192 E.02532
G1 X105.216 Y139.974 E.72102
G1 X105.216 Y140.813 E.03242
G1 X119.128 Y126.901 E.76053
G2 X120.052 Y126.796 I-.03 J-4.38 E.03603
G1 X120.079 Y126.789 E.00105
G1 X105.216 Y141.652 E.8125
G3 X105.173 Y141.945 I-.951 J.011 E.01153
G1 X105.761 Y141.945 E.02273
G1 X128.975 Y118.731 E1.26905
G2 X129.376 Y119.169 I2.701 J-2.075 E.02297
G1 X106.599 Y141.945 E1.24514
G1 X107.438 Y141.945 E.03242
G1 X129.827 Y119.557 E1.22393
G1 X130.19 Y119.858 E.01823
G1 X130.297 Y119.925 E.00489
G1 X108.277 Y141.945 E1.2038
G1 X109.115 Y141.945 E.03242
G1 X130.813 Y120.248 E1.18615
G2 X131.359 Y120.54 I2.424 J-3.869 E.02397
G1 X109.954 Y141.945 E1.17016
G1 X110.792 Y141.945 E.03242
G1 X131.948 Y120.79 E1.15652
G1 X132.087 Y120.849 E.00583
G1 X132.59 Y120.987 E.02016
G1 X111.631 Y141.945 E1.14576
G1 X112.469 Y141.945 E.03242
G1 X133.265 Y121.15 E1.13684
G2 X134.014 Y121.24 I.983 J-5.034 E.02918
G1 X113.308 Y141.945 E1.13193
G1 X114.147 Y141.945 E.03242
G1 X134.845 Y121.247 E1.13154
G2 X135.78 Y121.15 I-.051 J-5.081 E.03639
G1 X114.985 Y141.945 E1.13681
G1 X115.824 Y141.945 E.03242
G1 X136.202 Y121.568 E1.114
G1 X136.473 Y122.135 E.02431
G1 X116.662 Y141.945 E1.08297
G1 X117.501 Y141.945 E.03242
G1 X136.743 Y122.703 E1.05194
G2 X137.024 Y123.261 I7.896 J-3.628 E.02414
G1 X118.34 Y141.945 E1.02145
G1 X119.178 Y141.945 E.03242
G1 X137.312 Y123.811 E.99134
G1 X137.6 Y124.362 E.02402
G1 X120.017 Y141.945 E.96122
G1 X120.855 Y141.945 E.03242
G1 X137.901 Y124.9 E.93184
G1 X138.203 Y125.437 E.0238
G1 X121.694 Y141.945 E.9025
G1 X122.532 Y141.945 E.03242
G1 X138.507 Y125.971 E.87329
G1 X138.823 Y126.494 E.02361
G1 X123.371 Y141.945 E.84471
G1 X124.21 Y141.945 E.03242
G1 X139.139 Y127.016 E.81613
G2 X139.462 Y127.532 I8.639 J-5.053 E.02352
G1 X125.048 Y141.945 E.78794
G1 X125.887 Y141.945 E.03242
G1 X139.791 Y128.041 E.76009
G1 X140.12 Y128.551 E.02344
M73 P44 R10
G1 X126.725 Y141.945 E.73225
G1 X127.564 Y141.945 E.03242
G1 X140.46 Y129.049 E.705
G1 X140.803 Y129.545 E.0233
G1 X128.403 Y141.945 E.67789
G1 X129.241 Y141.945 E.03242
G1 X141.146 Y130.041 E.6508
G1 X141.503 Y130.522 E.02317
G1 X130.08 Y141.945 E.62448
G1 X130.918 Y141.945 E.03242
G1 X141.86 Y131.003 E.59817
G1 X142.218 Y131.485 E.02317
G1 X131.757 Y141.945 E.57185
G1 X132.596 Y141.945 E.03242
G1 X142.584 Y131.957 E.54605
G1 X142.953 Y132.426 E.02309
G1 X133.434 Y141.945 E.52037
G1 X134.273 Y141.945 E.03242
G1 X143.322 Y132.896 E.4947
G2 X143.7 Y133.357 I9.414 J-7.334 E.02304
G1 X135.111 Y141.945 E.46951
G1 X135.95 Y141.945 E.03242
G1 X144.079 Y133.816 E.44442
G2 X144.465 Y134.269 I10.584 J-8.603 E.023
G1 X136.788 Y141.945 E.41964
G1 X137.627 Y141.945 E.03242
G1 X144.853 Y134.719 E.39504
G2 X145.251 Y135.16 I9.265 J-7.941 E.02296
G1 X138.466 Y141.945 E.37092
G1 X139.304 Y141.945 E.03242
G1 X145.65 Y135.6 E.34689
G2 X146.051 Y136.037 I6.672 J-5.729 E.02295
G1 X140.143 Y141.945 E.323
G1 X140.981 Y141.945 E.03242
G1 X146.464 Y136.462 E.29974
G1 X146.878 Y136.888 E.02292
G1 X141.82 Y141.945 E.27649
G1 X142.659 Y141.945 E.03242
G1 X147.291 Y137.313 E.25323
G2 X147.712 Y137.731 I7.339 J-6.969 E.02293
G1 X143.497 Y141.945 E.23039
G1 X144.336 Y141.945 E.03242
G1 X148.137 Y138.144 E.2078
G2 X148.57 Y138.55 I9.43 J-9.634 E.02294
G1 X145.174 Y141.945 E.18563
G1 X146.013 Y141.945 E.03242
G1 X149.005 Y138.954 E.16355
G2 X149.446 Y139.351 I8.231 J-8.683 E.02295
G1 X146.852 Y141.945 E.14181
G1 X147.69 Y141.945 E.03242
G1 X149.891 Y139.745 E.1203
G2 X150.339 Y140.135 I8.059 J-8.791 E.02298
G1 X148.529 Y141.945 E.09895
G1 X149.367 Y141.945 E.03242
G1 X150.794 Y140.518 E.07801
G2 X151.25 Y140.901 I8.876 J-10.124 E.02301
G1 X150.206 Y141.945 E.05711
G1 X151.044 Y141.945 E.03242
G1 X151.715 Y141.274 E.03668
G2 X152.183 Y141.646 I8.812 J-10.617 E.02307
G1 X151.619 Y142.209 E.0308
G1 X123.236 Y119.439 F36000
G1 F13280.605
G1 X126.802 Y115.872 E.19499
G1 X127.16 Y116.354 E.02317
G1 X123.79 Y119.724 E.18422
G3 X124.027 Y120.143 I-3.943 J2.513 E.01863
G1 X124.085 Y120.266 E.00529
G1 X127.517 Y116.835 E.18759
G1 X127.874 Y117.317 E.02317
G1 X124.302 Y120.888 E.19527
G3 X124.429 Y121.381 I-3.164 J1.077 E.01968
G1 X124.455 Y121.574 E.00754
G1 X128.231 Y117.798 E.20645
G1 X128.588 Y118.279 E.02317
G1 X124.112 Y122.756 E.24473
G1 X119.993 Y126.479 F36000
; FEATURE: Top surface
; LINE_WIDTH: 0.62
G1 F9000
M204 S2000
G1 X124.147 Y122.324 E.22432
G1 X124.345 Y122.127
G1 X124.321 Y121.322
G1 X124.124 Y121.519
G1 X119.066 Y126.577 E.27311
G1 X118.869 Y126.774
G1 X118.177 Y126.638
G1 X118.374 Y126.441
G1 X123.955 Y120.86 E.30134
G1 X124.153 Y120.662
G1 X123.923 Y120.063
G1 X123.726 Y120.261
G1 X117.789 Y126.197 E.32054
G1 X117.592 Y126.395
G1 X117.059 Y126.099
G1 X117.256 Y125.902
G1 X123.412 Y119.746 E.33239
G1 X123.61 Y119.549
G1 X123.233 Y119.097
G1 X123.036 Y119.295
G1 X116.795 Y125.535 E.33697
M204 S10000
G1 X124.013 Y122.923 F36000
; FEATURE: Gap infill
; LINE_WIDTH: 0.219169
G1 F3000
G1 X123.804 Y123.225 E.00442
; LINE_WIDTH: 0.258849
G1 X123.72 Y123.34 E.00208
; LINE_WIDTH: 0.302235
G1 X123.636 Y123.454 E.00248
; LINE_WIDTH: 0.345622
G1 X123.552 Y123.569 E.00288
; LINE_WIDTH: 0.389008
G1 X123.468 Y123.684 E.00329
; LINE_WIDTH: 0.423067
G1 X123.411 Y123.758 E.00238
; LINE_WIDTH: 0.452451
G1 X123.31 Y123.883 E.00439
; LINE_WIDTH: 0.479352
G1 X123.228 Y123.979 E.00365
; LINE_WIDTH: 0.515268
G3 X122.79 Y124.44 I-5.189 J-4.493 E.01995
; LINE_WIDTH: 0.501451
G1 X122.453 Y124.749 E.0139
; LINE_WIDTH: 0.46138
G1 X122.116 Y125.057 E.01271
; LINE_WIDTH: 0.42131
G1 X121.78 Y125.365 E.01152
; LINE_WIDTH: 0.381239
G1 X121.443 Y125.674 E.01033
; LINE_WIDTH: 0.341816
G3 X121.245 Y125.842 I-3.29 J-3.67 E.0052
; LINE_WIDTH: 0.308314
G1 X121.169 Y125.902 E.00173
; LINE_WIDTH: 0.276087
G1 X121.048 Y125.993 E.00239
; LINE_WIDTH: 0.233516
G1 X120.927 Y126.085 E.00197
; LINE_WIDTH: 0.189962
G1 X120.817 Y126.163 E.00136
; LINE_WIDTH: 0.151742
G1 X120.735 Y126.218 E.00076
; LINE_WIDTH: 0.119896
G1 X120.652 Y126.274 E.00056
; WIPE_START
G1 X120.735 Y126.218 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X119.135 Y122.733 Z1.08 F36000
G1 Z.68
G1 E.4 F1800
; LINE_WIDTH: 0.125855
G1 F3000
G1 X118.767 Y123.071 E.00298
; LINE_WIDTH: 0.168348
G1 X118.399 Y123.408 E.00436
; LINE_WIDTH: 0.211079
G1 X118.027 Y123.749 E.00582
; LINE_WIDTH: 0.256624
G1 X117.774 Y123.967 E.00484
; LINE_WIDTH: 0.304406
G1 X117.522 Y124.186 E.00587
; LINE_WIDTH: 0.352187
G1 X117.27 Y124.404 E.00691
; LINE_WIDTH: 0.399969
G1 X117.018 Y124.623 E.00795
; LINE_WIDTH: 0.4482
G1 X116.761 Y124.845 E.00917
; LINE_WIDTH: 0.492145
G1 X116.404 Y125.145 E.01391
; CHANGE_LAYER
; Z_HEIGHT: 0.84
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F3000
G1 X116.761 Y124.845 E-.38
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
G1 X122.357 Y120.94
G1 Z.84
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X122.542 Y121.259 E.01408
G1 X122.65 Y121.733 E.01855
G1 X122.635 Y122.239 E.01932
G1 X122.504 Y122.692 E.01803
G1 X122.244 Y123.132 E.01949
G1 X121.963 Y123.422 E.01543
G1 X120.536 Y124.618 E.07109
G1 X120.08 Y124.904 E.02054
G1 X119.62 Y125.044 E.01835
G1 X119.092 Y125.064 E.02017
G3 X118.219 Y124.758 I.298 J-2.245 E.03559
G1 X117.789 Y124.376 E.02194
G1 X117.309 Y123.803 E.02855
G3 X103.718 Y131.274 I-32.974 J-43.889 E.59411
G1 X104.259 Y131.673 E.02567
G1 X104.611 Y132.289 E.02709
G1 X104.7 Y132.654 E.01435
G3 X104.72 Y135.659 I-102.876 J2.185 E.11475
G1 X104.72 Y141.659 E.22907
G1 X104.625 Y142.26 E.02324
G1 X104.521 Y142.443 E.00803
G1 X154.069 Y142.443 E1.89167
G3 X143.803 Y132.701 I31.506 J-43.481 E.54191
G3 X136.316 Y120.644 I41.732 J-34.265 E.54342
G1 X136.271 Y120.545 E.00416
G3 X133.277 Y120.652 I-1.756 J-7.189 E.11518
G1 X132.426 Y120.433 E.03357
G1 X131.481 Y120.051 E.03892
G1 X130.602 Y119.534 E.03893
G1 X129.808 Y118.894 E.03894
G1 X129.291 Y118.341 E.0289
G3 X127.854 Y116.454 I41.582 J-33.151 E.09057
G1 X126.662 Y114.848 E.07636
G3 X121.722 Y120.1 I-41.947 J-34.503 E.27549
G1 X122.205 Y120.675 E.02865
G1 X122.312 Y120.862 E.00823
G1 X121.859 Y121.234 F36000
G1 F13446.369
G1 X122.001 Y121.488 E.01111
G1 X122.074 Y121.895 E.01578
G1 X122.01 Y122.341 E.0172
G1 X121.816 Y122.727 E.01652
G1 X121.587 Y122.973 E.01285
G1 X120.16 Y124.169 E.07109
G1 X119.841 Y124.369 E.01437
G1 X119.456 Y124.477 E.01526
G1 X119.15 Y124.482 E.01169
G1 X118.711 Y124.362 E.01734
G1 X118.324 Y124.094 E.01799
G3 X117.399 Y122.999 I45.859 J-39.704 E.05472
G3 X105.172 Y130.069 I-33.103 J-43.145 E.54074
G1 X104.061 Y130.508 E.04562
G1 X103.689 Y130.654 E.01527
G1 F12148.309
G1 X103.317 Y130.801 E.01527
; LINE_WIDTH: 0.667558
G1 F10820.701
G1 X103.233 Y130.856 E.00415
; LINE_WIDTH: 0.71512
G1 F10499.076
G1 X103.149 Y130.912 E.00446
; LINE_WIDTH: 0.762682
G1 F10182.261
G1 X103.065 Y130.967 E.00477
; LINE_WIDTH: 0.810244
G1 F9870.299
G1 X102.981 Y131.023 E.00509
; LINE_WIDTH: 0.857806
G1 F9563.191
G1 X102.898 Y131.079 E.0054
; LINE_WIDTH: 0.905368
G1 F9041
G1 X102.814 Y131.134 E.00571
; LINE_WIDTH: 0.95293
G1 F8572.885
G1 X102.73 Y131.19 E.00602
; LINE_WIDTH: 1.00049
G1 F8150.858
G1 X102.646 Y131.245 E.00633
; LINE_WIDTH: 1.04805
G1 F7768.433
G1 X102.563 Y131.301 E.00665
; LINE_WIDTH: 1.09562
G1 F7420.285
G1 X102.479 Y131.357 E.00696
G1 X102.559 Y131.388 E.00593
; LINE_WIDTH: 1.04805
G1 F7768.433
G1 X102.639 Y131.418 E.00566
; LINE_WIDTH: 1.00049
G1 F8150.858
G1 X102.719 Y131.449 E.0054
; LINE_WIDTH: 0.95293
G1 F8572.885
G1 X102.799 Y131.48 E.00513
; LINE_WIDTH: 0.905368
G1 F9041
G1 X102.878 Y131.511 E.00487
; LINE_WIDTH: 0.857806
G1 F9563.191
G1 X102.958 Y131.542 E.0046
; LINE_WIDTH: 0.810244
G1 F10149.401
G1 X103.038 Y131.573 E.00434
; LINE_WIDTH: 0.762682
G1 F10418.714
G1 X103.118 Y131.604 E.00407
; LINE_WIDTH: 0.71512
G1 F10691.566
G1 X103.198 Y131.635 E.0038
; LINE_WIDTH: 0.667558
G1 F10967.944
G1 X103.278 Y131.666 E.00354
; LINE_WIDTH: 0.619996
G1 F12304.295
G1 X103.603 Y131.9 E.01527
G1 F13207.475
G1 X103.812 Y132.051 E.00986
G1 F13446.369
G1 X104.068 Y132.509 E.02005
G1 X104.135 Y132.932 E.01633
G1 X104.135 Y141.659 E.3332
G1 X104.068 Y142.08 E.01626
G1 X103.798 Y142.557 E.02094
; LINE_WIDTH: 0.634996
G1 F13110.58
G1 X103.252 Y142.944 E.0262
; LINE_WIDTH: 0.646436
G1 F12865.545
G1 X103.008 Y143.016 E.01012
G1 X104.012 Y143.029 E.04005
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X155.749 Y143.029 E1.97526
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00187
G3 X144.254 Y132.328 I30.059 J-44.23 E.59949
G3 X136.596 Y119.828 I41.489 J-34.015 E.56143
G1 X136.067 Y119.995 E.02116
G1 X135.347 Y120.129 E.02798
G1 X134.414 Y120.179 E.03567
G3 X133.284 Y120.054 I.628 J-10.839 E.0434
G1 X132.562 Y119.863 E.02852
G1 X131.711 Y119.513 E.03512
G1 X130.909 Y119.036 E.03564
G1 X130.185 Y118.446 E.03565
G1 X129.73 Y117.953 E.02561
G3 X127.875 Y115.499 I56.187 J-44.393 E.11746
G1 X126.683 Y113.893 E.07636
G3 X120.917 Y120.051 I-42.62 J-34.133 E.32241
G1 X121.756 Y121.052 E.04986
G1 X121.814 Y121.156 E.00457
G1 X121.357 Y121.516 F36000
G1 F13446.369
G1 X121.447 Y121.677 E.00702
G1 X121.488 Y121.965 E.0111
G1 X121.4 Y122.289 E.01282
G1 X121.21 Y122.524 E.01155
G1 X119.783 Y123.721 E.07109
G1 X119.495 Y123.873 E.01246
G1 X119.207 Y123.899 E.01103
G1 X118.957 Y123.831 E.0099
G1 X118.687 Y123.624 E.01299
G1 X117.472 Y122.174 E.07222
G1 X117.027 Y122.547 E.02217
G3 X99.506 Y131.447 I-32.59 J-42.462 E.75454
G2 X99.547 Y132.737 I733.498 J-22.649 E.04927
G1 X102.629 Y132.166 E.11965
G1 X103.073 Y132.215 E.01708
G1 X103.374 Y132.44 E.01434
G1 X103.511 Y132.691 E.0109
G3 X103.549 Y133.659 I-4.233 J.652 E.03708
G1 X103.549 Y141.659 E.30543
G1 X103.511 Y141.899 E.00928
G1 X103.357 Y142.171 E.01195
G1 X103.044 Y142.388 E.01454
G1 X102.77 Y142.438 E.0106
G1 X102.629 Y142.425 E.00544
G1 X99.548 Y141.854 E.11961
G3 X99.509 Y143.615 I-18.612 J.47 E.06726
G1 X156.235 Y143.615 E2.16575
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.705 Y131.954 I29.103 J-43.629 E.60776
G3 X136.912 Y119.083 I40.828 J-33.517 E.57644
G1 X136.092 Y119.382 E.03332
G1 X135.319 Y119.538 E.03009
G1 X134.392 Y119.594 E.03548
G3 X132.717 Y119.299 I.439 J-7.399 E.06508
G1 X131.942 Y118.975 E.03206
G1 X131.216 Y118.537 E.03236
G1 X130.562 Y117.998 E.03237
G1 X130.168 Y117.565 E.02233
G3 X129.081 Y116.142 I34.003 J-27.106 E.06839
G1 X126.698 Y112.93 E.15272
G3 X121.59 Y118.617 I-42.786 J-33.287 E.2921
G3 X120.092 Y119.978 I-25.293 J-26.344 E.07729
G1 X121.307 Y121.428 E.07222
G1 X121.313 Y121.438 E.00045
M204 S250
G1 X120.883 Y121.783 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X120.936 Y121.938 E.0052
G1 X120.864 Y122.093 E.00541
G3 X119.428 Y123.297 I-24.652 J-27.944 E.05931
G1 X119.261 Y123.349 E.00554
G1 X119.11 Y123.269 E.0054
G1 X118.268 Y122.264 E.0415
; LINE_WIDTH: 0.523196
G1 X117.714 Y121.609 E.02734
; LINE_WIDTH: 0.544336
G1 X117.697 Y121.299 E.01032
G1 X116.671 Y122.124 E.04375
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-32.257 J-42.079 E.63217
G3 X98.996 Y133.401 I-57.339 J2.629 E.07486
G1 X102.729 Y132.71 E.12022
G1 X102.858 Y132.724 E.00411
G1 X102.985 Y132.862 E.00593
G3 X102.996 Y135.659 I-145.7 J1.977 E.08856
G1 X102.996 Y141.659 E.18996
G1 X102.94 Y141.808 E.00502
G1 X102.77 Y141.885 E.00591
G1 X102.729 Y141.881 E.00131
G1 X98.996 Y141.19 E.12021
G1 X98.995 Y142.82 E.05161
G2 X98.937 Y144.167 I14.502 J1.295 E.04272
G1 X156.695 Y144.167 E1.82862
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.613 J-43.498 E.51061
G3 X137.192 Y118.315 I40.272 J-33.072 E.49168
G1 X136.612 Y118.616 E.02068
G1 X135.912 Y118.86 E.02347
G1 X135.218 Y118.995 E.02238
G1 X134.371 Y119.041 E.02685
G1 X133.607 Y118.959 E.02433
G1 X132.865 Y118.766 E.02427
G1 X132.16 Y118.466 E.02426
G1 X131.506 Y118.067 E.02426
G1 X130.918 Y117.575 E.02427
G1 X130.582 Y117.199 E.01594
G3 X129.087 Y115.222 I49.399 J-38.925 E.07848
G1 X126.704 Y112.01 E.12664
G3 X120.192 Y119.172 I-42.396 J-32.003 E.3069
; LINE_WIDTH: 0.521596
G1 X119.525 Y119.734 E.02769
; LINE_WIDTH: 0.544336
G1 X119.156 Y120.074 E.01669
G1 X119.491 Y120.12 E.01125
; LINE_WIDTH: 0.521596
G1 X120.042 Y120.779 E.02729
; LINE_WIDTH: 0.519996
G1 X120.825 Y121.714 E.03861
; WIPE_START
M204 S10000
G1 X120.936 Y121.938 E-.09507
G1 X120.864 Y122.093 E-.06489
G1 X120.42 Y122.465 E-.22004
; WIPE_END
G1 E-.02 F1800
G1 X119.156 Y120.074 Z1.24 F36000
G1 Z.84
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X117.697 Y121.299 E.06335
; WIPE_START
M204 S10000
G1 X118.463 Y120.656 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X111.968 Y124.666 Z1.24 F36000
G1 X100.181 Y131.943 Z1.24
G1 Z.84
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.81887
G1 F10037.815
G1 X100.501 Y131.864 E.01684
; LINE_WIDTH: 0.858643
G1 F9553.485
G1 X100.82 Y131.784 E.0177
; LINE_WIDTH: 0.898416
G1 F9113.741
G1 X101.14 Y131.705 E.01855
; LINE_WIDTH: 0.902656
G1 F9069.238
G1 X101.171 Y131.697 E.00183
; LINE_WIDTH: 0.948801
G1 F8611.594
G1 X101.485 Y131.615 E.01934
; LINE_WIDTH: 0.994946
G1 F8197.916
G1 X101.799 Y131.534 E.02031
; LINE_WIDTH: 1.04109
G1 F7822.161
G1 X102.113 Y131.452 E.02129
; LINE_WIDTH: 1.08724
G1 F7479.343
G1 X102.427 Y131.37 E.02226
; LINE_WIDTH: 1.09562
G1 F7420.285
G1 X102.479 Y131.357 E.0037
; WIPE_START
G1 X102.427 Y131.37 E-.02031
G1 X102.113 Y131.452 E-.12326
G1 X101.799 Y131.534 E-.12326
G1 X101.511 Y131.609 E-.11318
; WIPE_END
G1 E-.02 F1800
G1 X100.722 Y139.2 Z1.24 F36000
G1 X100.347 Y142.815 Z1.24
G1 Z.84
G1 E.4 F1800
; LINE_WIDTH: 1.04768
G1 F7771.331
G1 X100.528 Y142.832 E.01199
; LINE_WIDTH: 1.01448
G1 F8034.567
G1 X100.777 Y142.855 E.01597
; LINE_WIDTH: 0.968759
G1 F8427.665
G1 X101.026 Y142.877 E.01523
; LINE_WIDTH: 0.923041
G1 F8861.209
G1 X101.275 Y142.9 E.01448
; LINE_WIDTH: 0.877324
G1 F9341.776
G1 X101.524 Y142.923 E.01374
; LINE_WIDTH: 0.831606
G1 F9877.457
G1 X101.773 Y142.946 E.01299
; LINE_WIDTH: 0.785889
G1 F10478.311
G1 X102.022 Y142.969 E.01225
; LINE_WIDTH: 0.740171
G1 F11156.999
G1 X102.27 Y142.992 E.0115
; LINE_WIDTH: 0.694454
G1 F11929.695
G1 X102.519 Y143.015 E.01076
; LINE_WIDTH: 0.648736
G1 F12817.384
G1 X103.008 Y143.016 E.01959
; WIPE_START
G1 X102.519 Y143.015 E-.18582
G1 X102.27 Y142.992 E-.095
G1 X102.022 Y142.969 E-.095
G1 X102.011 Y142.968 E-.00418
; WIPE_END
G1 E-.02 F1800
G1 X104.955 Y141.103 Z1.24 F36000
G1 Z.84
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.626676
G1 F13294.73
G1 X105.797 Y141.945 E.04601
G1 X106.635 Y141.945 E.03235
G1 X105.218 Y140.529 E.07737
G1 X105.218 Y139.691 E.03235
G1 X107.473 Y141.945 E.12311
G1 X108.31 Y141.945 E.03235
G1 X105.218 Y138.853 E.16886
G1 X105.218 Y138.016 E.03235
G1 X109.148 Y141.945 E.21461
G1 X109.986 Y141.945 E.03235
G1 X105.218 Y137.178 E.26035
G1 X105.218 Y136.34 E.03235
G1 X110.823 Y141.945 E.3061
G1 X111.661 Y141.945 E.03235
G1 X105.218 Y135.502 E.35184
G1 X105.218 Y134.665 E.03235
G1 X112.499 Y141.945 E.39759
G1 X113.336 Y141.945 E.03235
G1 X105.218 Y133.827 E.44334
G1 X105.218 Y132.989 E.03235
G1 X114.174 Y141.945 E.48908
G1 X115.012 Y141.945 E.03235
M73 P45 R10
G1 X104.95 Y131.884 E.54947
G1 X104.695 Y131.428 E.02017
G1 X105.15 Y131.246 E.01892
G1 X115.849 Y141.945 E.58429
G1 X116.687 Y141.945 E.03235
G1 X105.744 Y131.002 E.59759
G1 X106.326 Y130.746 E.02454
G1 X117.525 Y141.945 E.61157
G1 X118.363 Y141.945 E.03235
G1 X106.908 Y130.49 E.62555
G1 X107.489 Y130.235 E.02454
G1 X119.2 Y141.945 E.63952
G1 X120.038 Y141.945 E.03235
G1 X108.061 Y129.969 E.65403
G1 X108.627 Y129.697 E.02424
G1 X120.876 Y141.945 E.66888
G1 X121.713 Y141.945 E.03235
G1 X109.187 Y129.419 E.68405
G1 X109.738 Y129.133 E.02399
G1 X122.551 Y141.945 E.69969
G1 X123.389 Y141.945 E.03235
G1 X110.29 Y128.846 E.71533
G2 X110.84 Y128.559 I-4.405 J-9.087 E.02397
G1 X124.226 Y141.945 E.73105
G1 X125.064 Y141.945 E.03235
G1 X111.377 Y128.258 E.74745
G1 X111.914 Y127.958 E.02377
G1 X125.902 Y141.945 E.76386
G1 X126.739 Y141.945 E.03235
G1 X112.438 Y127.644 E.78097
G1 X112.962 Y127.33 E.02358
G1 X127.577 Y141.945 E.79813
G1 X128.415 Y141.945 E.03235
G1 X113.485 Y127.016 E.81528
G2 X113.999 Y126.692 I-3.823 J-6.628 E.02346
G1 X129.253 Y141.945 E.83298
G1 X130.09 Y141.945 E.03235
G1 X114.505 Y126.36 E.85109
G1 X115.011 Y126.029 E.02336
G1 X130.928 Y141.945 E.86921
G1 X131.766 Y141.945 E.03235
G1 X115.517 Y125.697 E.88732
G2 X116.01 Y125.352 I-3.995 J-6.244 E.02324
G1 X132.603 Y141.945 E.90613
G1 X133.441 Y141.945 E.03235
G1 X116.499 Y125.003 E.9252
G1 X116.987 Y124.654 E.02319
G1 X134.279 Y141.945 E.94428
G1 X135.116 Y141.945 E.03235
G1 X118.656 Y125.485 E.89888
G1 X118.724 Y125.508 E.00278
G1 X119.22 Y125.571 E.01928
G2 X119.559 Y125.55 I.075 J-1.515 E.01314
G1 X135.954 Y141.945 E.89534
G1 X136.792 Y141.945 E.03235
G1 X120.226 Y125.379 E.90466
G1 X120.759 Y125.075 E.0237
G1 X137.63 Y141.945 E.92131
G1 X138.467 Y141.945 E.03235
G1 X121.215 Y124.694 E.94211
G1 X121.672 Y124.313 E.02297
G1 X139.305 Y141.945 E.96291
G1 X140.143 Y141.945 E.03235
G1 X122.129 Y123.932 E.98371
G2 X122.561 Y123.526 I-2.149 J-2.719 E.02291
G1 X140.98 Y141.945 E1.00588
G1 X141.818 Y141.945 E.03235
G1 X122.894 Y123.021 E1.03345
G2 X123.108 Y122.398 I-1.931 J-1.013 E.02555
G1 X142.656 Y141.945 E1.06748
G1 X143.493 Y141.945 E.03235
G1 X123.128 Y121.58 E1.11213
G2 X123.026 Y121.131 I-1.518 J.11 E.01787
G1 X122.705 Y120.508 E.02703
G1 X122.403 Y120.14 E.01839
G1 X122.466 Y120.08 E.00335
G1 X144.331 Y141.945 E1.19405
G1 X145.169 Y141.945 E.03235
G1 X122.892 Y119.669 E1.21652
G1 X123.318 Y119.257 E.02288
G1 X146.006 Y141.945 E1.23899
G1 X146.844 Y141.945 E.03235
G1 X123.736 Y118.837 E1.26193
G1 X124.149 Y118.413 E.02287
G1 X147.682 Y141.945 E1.28509
G1 X148.52 Y141.945 E.03235
G1 X124.559 Y117.985 E1.30848
G1 X124.96 Y117.548 E.02289
G1 X149.357 Y141.945 E1.33234
G1 X150.195 Y141.945 E.03235
G1 X125.36 Y117.111 E1.3562
G1 X125.761 Y116.674 E.02289
G1 X151.033 Y141.945 E1.38007
G1 X151.87 Y141.945 E.03235
G1 X126.148 Y116.223 E1.40467
G1 X126.535 Y115.772 E.02294
G1 X127.284 Y116.521 E.04092
G1 X128.634 Y118.341 E.08749
G1 X129.234 Y119.029 E.03526
G1 X129.482 Y119.271 E.01337
G1 X130.197 Y119.863 E.03586
G1 X130.971 Y120.347 E.03522
G1 X131.258 Y120.496 E.01251
G1 X142.132 Y131.369 E.59378
G3 X140.077 Y128.477 I40.483 J-30.934 E.13703
G1 X132.588 Y120.988 E.40895
G2 X133.636 Y121.198 I1.535 J-4.944 E.04132
G1 X138.659 Y126.221 E.2743
G3 X137.558 Y124.282 I66.155 J-38.849 E.08608
G1 X134.534 Y121.259 E.16512
G2 X135.332 Y121.219 I.128 J-5.432 E.03087
G1 X137.264 Y123.151 E.10551
; CHANGE_LAYER
; Z_HEIGHT: 1
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13294.73
G1 X136.557 Y122.444 E-.38
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
G1 X122.617 Y120.675
G1 Z1
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X122.756 Y120.896 E.00994
G1 X122.888 Y121.296 E.01608
G1 X122.926 Y121.828 E.02039
G1 X122.826 Y122.33 E.01952
G1 X122.619 Y122.759 E.01818
G3 X121.798 Y123.56 I-3.06 J-2.312 E.04395
G1 X120.266 Y124.845 E.07636
G1 X119.803 Y125.133 E.02082
G1 X119.35 Y125.271 E.01807
G1 X118.801 Y125.289 E.02098
G3 X117.795 Y124.873 I.322 J-2.204 E.04197
G3 X117.027 Y124.016 I5.116 J-5.36 E.04399
G3 X103.721 Y131.273 I-32.545 J-43.846 E.58052
G1 X104.261 Y131.669 E.02557
G1 X104.538 Y132.102 E.01962
G1 X104.676 Y132.505 E.01626
G3 X104.722 Y133.662 I-6.687 J.849 E.04425
G1 X104.722 Y141.662 E.30543
G1 X104.627 Y142.263 E.02324
G1 X104.525 Y142.443 E.00792
G1 X154.07 Y142.443 E1.89157
G3 X143.806 Y132.705 I31.787 J-43.78 E.54171
G3 X136.316 Y120.644 I41.734 J-34.272 E.54362
G1 X136.271 Y120.545 E.00415
G1 X135.539 Y120.689 E.02851
G1 X134.435 Y120.764 E.04223
G3 X133.274 Y120.651 I.627 J-12.507 E.04455
G1 X132.429 Y120.434 E.03334
G1 X131.481 Y120.051 E.03903
G1 X130.601 Y119.534 E.03894
G1 X129.808 Y118.894 E.03893
G1 X129.123 Y118.152 E.03854
G3 X127.853 Y116.453 I116.315 J-88.276 E.08099
G1 X126.662 Y114.847 E.07636
G3 X121.98 Y119.858 I-42.816 J-35.306 E.262
G1 X122.475 Y120.449 E.02941
G1 X122.569 Y120.599 E.00678
G1 X122.12 Y120.975 F36000
G1 F13446.369
G1 X122.223 Y121.138 E.00736
G1 X122.34 Y121.59 E.01784
G1 X122.299 Y122.05 E.01763
G1 X122.127 Y122.441 E.01632
G1 X121.857 Y122.747 E.01557
G1 X119.889 Y124.396 E.09801
G1 X119.566 Y124.598 E.01457
G1 X119.175 Y124.705 E.01547
G1 X118.781 Y124.695 E.01504
G1 X118.338 Y124.536 E.01796
G1 X117.968 Y124.227 E.01843
G1 X117.118 Y123.213 E.05049
G3 X104.25 Y130.442 I-32.753 J-43.233 E.56526
G1 X103.69 Y130.657 E.0229
G1 F12135.381
G1 X103.317 Y130.8 E.01527
; LINE_WIDTH: 0.667686
G1 F10808.499
G1 X103.233 Y130.856 E.00414
; LINE_WIDTH: 0.715376
G1 F10488.104
G1 X103.149 Y130.911 E.00445
; LINE_WIDTH: 0.763066
G1 F10172.511
G1 X103.066 Y130.966 E.00476
; LINE_WIDTH: 0.810756
G1 F9861.714
G1 X102.982 Y131.022 E.00507
; LINE_WIDTH: 0.858446
G1 F9555.764
G1 X102.899 Y131.077 E.00538
; LINE_WIDTH: 0.906136
G1 F9033.035
G1 X102.815 Y131.132 E.0057
; LINE_WIDTH: 0.953826
G1 F8564.531
G1 X102.731 Y131.188 E.00601
; LINE_WIDTH: 1.00152
G1 F8142.229
G1 X102.648 Y131.243 E.00632
; LINE_WIDTH: 1.04921
G1 F7759.615
G1 X102.564 Y131.298 E.00663
; LINE_WIDTH: 1.0969
G1 F7411.346
G1 X102.481 Y131.353 E.00694
G1 X102.56 Y131.384 E.0059
; LINE_WIDTH: 1.04921
G1 F7759.615
G1 X102.64 Y131.415 E.00564
; LINE_WIDTH: 1.00152
G1 F8142.229
G1 X102.719 Y131.446 E.00537
; LINE_WIDTH: 0.953826
G1 F8564.531
G1 X102.798 Y131.476 E.00511
; LINE_WIDTH: 0.906136
G1 F9033.035
G1 X102.878 Y131.507 E.00484
; LINE_WIDTH: 0.858446
G1 F9555.764
G1 X102.957 Y131.538 E.00458
; LINE_WIDTH: 0.810756
G1 F10142.707
G1 X103.037 Y131.569 E.00431
; LINE_WIDTH: 0.763066
G1 F10410.209
G1 X103.116 Y131.599 E.00405
; LINE_WIDTH: 0.715376
G1 F10681.212
G1 X103.196 Y131.63 E.00378
; LINE_WIDTH: 0.667686
G1 F10955.679
G1 X103.275 Y131.661 E.00352
; LINE_WIDTH: 0.619996
G1 F12291.304
G1 X103.6 Y131.894 E.01527
G1 F13211.841
G1 X103.814 Y132.048 E.01005
G1 F13446.369
G1 X104.008 Y132.35 E.01372
G3 X104.104 Y132.632 I-1.235 J.579 E.0114
G3 X104.137 Y133.662 I-6.512 J.722 E.03935
G1 X104.137 Y141.662 E.30543
G1 X104.07 Y142.082 E.01626
G1 X103.8 Y142.559 E.02094
; LINE_WIDTH: 0.633566
G1 F13141.867
G1 X103.253 Y142.945 E.02613
; LINE_WIDTH: 0.643896
G1 F12919.155
G1 X103.01 Y143.017 E.01007
G1 X104.013 Y143.029 E.03984
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X155.749 Y143.029 E1.97522
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.256 Y132.331 I29.801 J-43.95 E.59938
G3 X136.596 Y119.828 I41.507 J-34.031 E.56158
G1 X136.067 Y119.995 E.02116
G1 X135.346 Y120.129 E.02799
G1 X134.413 Y120.179 E.03567
G3 X133.287 Y120.054 I.623 J-10.804 E.04328
G1 X132.565 Y119.865 E.0285
G1 X131.711 Y119.513 E.03526
G1 X130.909 Y119.036 E.03565
G1 X130.185 Y118.446 E.03564
G1 X129.562 Y117.764 E.03527
G3 X127.875 Y115.499 I204.425 J-154.044 E.10783
G1 X126.683 Y113.893 E.07636
G3 X121.176 Y119.81 I-42.99 J-34.491 E.30892
G1 X122.026 Y120.825 E.05054
G1 X122.072 Y120.898 E.00331
G1 X121.631 Y121.312 F36000
G1 F13446.369
G1 X121.742 Y121.539 E.00966
G1 X121.746 Y121.841 E.01154
G1 X121.635 Y122.123 E.01158
G1 X121.481 Y122.298 E.00889
G1 X119.513 Y123.947 E.09801
G1 X119.22 Y124.101 E.01263
G1 X118.881 Y124.118 E.01299
G1 X118.628 Y124.027 E.01025
G1 X118.348 Y123.768 E.01456
G1 X117.201 Y122.401 E.06813
G3 X99.506 Y131.447 I-32.609 J-41.956 E.76327
G3 X99.549 Y132.734 I-834.695 J28.985 E.04917
G1 X102.631 Y132.164 E.11965
G1 X103.075 Y132.212 E.01705
G1 X103.377 Y132.439 E.01445
G1 X103.532 Y132.76 E.0136
G3 X103.551 Y135.662 I-97.718 J2.079 E.11079
G1 X103.551 Y141.662 E.22907
G1 X103.513 Y141.902 E.00928
G1 X103.359 Y142.174 E.01195
G1 X103.046 Y142.391 E.01454
G1 X102.773 Y142.44 E.0106
G1 X102.631 Y142.427 E.00544
G1 X99.548 Y141.856 E.11969
G3 X99.509 Y143.615 I-15.312 J.536 E.06719
G1 X156.235 Y143.615 E2.16577
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.707 Y131.956 I29.325 J-43.871 E.60764
G3 X136.912 Y119.083 I41.371 J-33.849 E.57647
G1 X136.092 Y119.382 E.03329
G1 X135.324 Y119.538 E.02994
G1 X134.392 Y119.594 E.03565
G3 X132.72 Y119.3 I.45 J-7.458 E.06495
G1 X131.942 Y118.974 E.03218
G1 X131.216 Y118.537 E.03236
G1 X130.562 Y117.998 E.03236
G1 X130.001 Y117.376 E.03199
G3 X129.081 Y116.142 I172.234 J-129.336 E.05875
G1 X126.698 Y112.93 E.15272
G3 X120.571 Y119.575 I-42.602 J-33.133 E.34551
G1 X120.362 Y119.752 E.01043
G1 X121.577 Y121.201 E.07222
G1 X121.592 Y121.231 E.00127
M204 S250
G1 X121.153 Y121.556 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.206 Y121.716 E.00532
G1 X121.125 Y121.874 E.00562
G1 X119.158 Y123.523 E.08128
G1 X118.988 Y123.575 E.00561
G1 X118.84 Y123.495 E.00533
G1 X117.771 Y122.22 E.05267
; LINE_WIDTH: 0.523196
G1 X117.444 Y121.836 E.0161
; LINE_WIDTH: 0.544336
G1 X117.427 Y121.526 E.01032
G1 X116.672 Y122.124 E.03202
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-32.254 J-42.072 E.63218
G3 X98.998 Y133.399 I-55.274 J2.626 E.07478
G1 X102.731 Y132.707 E.12022
G1 X102.86 Y132.721 E.0041
G1 X102.977 Y132.833 E.00513
G3 X102.998 Y135.662 I-78.43 J2.006 E.08954
G1 X102.998 Y141.662 E.18996
G1 X102.942 Y141.81 E.00502
G1 X102.773 Y141.887 E.00591
G1 X102.731 Y141.883 E.00131
G1 X98.997 Y141.192 E.12023
G1 X98.992 Y143.019 E.05784
G2 X98.937 Y144.167 I10.386 J1.07 E.03643
G1 X156.695 Y144.167 E1.82861
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.61 J-43.495 E.51059
G3 X137.192 Y118.315 I40.813 J-33.397 E.49165
G1 X136.614 Y118.615 E.02061
M73 P46 R10
G1 X135.912 Y118.859 E.02353
G1 X135.222 Y118.994 E.02226
G1 X134.371 Y119.041 E.02699
G1 X133.609 Y118.959 E.02425
G1 X132.867 Y118.767 E.02427
G1 X132.16 Y118.466 E.02434
G1 X131.506 Y118.066 E.02427
G1 X130.918 Y117.575 E.02426
G1 X130.415 Y117.01 E.02396
G3 X129.087 Y115.222 I571.263 J-425.739 E.07049
G1 X126.704 Y112.01 E.12664
G3 X120.192 Y119.172 I-42.397 J-32.005 E.3069
; LINE_WIDTH: 0.521596
G1 X119.795 Y119.507 E.01649
; LINE_WIDTH: 0.544336
G1 X119.426 Y119.847 E.01669
G1 X119.762 Y119.893 E.01125
; LINE_WIDTH: 0.521596
G1 X120.086 Y120.282 E.01609
; LINE_WIDTH: 0.519996
G1 X121.096 Y121.487 E.04978
; WIPE_START
M204 S10000
G1 X121.206 Y121.716 E-.09644
G1 X121.125 Y121.874 E-.06741
G1 X120.69 Y122.24 E-.21615
; WIPE_END
G1 E-.02 F1800
G1 X119.426 Y119.847 Z1.4 F36000
G1 Z1
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X117.427 Y121.526 E.0868
; WIPE_START
M204 S10000
G1 X118.193 Y120.883 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X111.688 Y124.876 Z1.4 F36000
G1 X100.181 Y131.942 Z1.4
G1 Z1
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.81679
G1 F10064.498
G1 X100.501 Y131.863 E.0168
; LINE_WIDTH: 0.856563
G1 F9577.652
G1 X100.82 Y131.783 E.01765
; LINE_WIDTH: 0.896336
G1 F9135.731
G1 X101.14 Y131.704 E.01851
; LINE_WIDTH: 0.942676
G1 F8669.662
G1 X101.463 Y131.62 E.01975
; LINE_WIDTH: 0.989016
G1 F8248.838
G1 X101.786 Y131.537 E.02075
; LINE_WIDTH: 1.03536
G1 F7866.976
G1 X102.109 Y131.454 E.02176
; LINE_WIDTH: 1.06613
G1 F7632.367
G1 X102.295 Y131.404 E.01296
; LINE_WIDTH: 1.0969
G1 F7411.346
G1 X102.481 Y131.353 E.01335
; WIPE_START
G1 X102.295 Y131.404 E-.07321
G1 X102.109 Y131.454 E-.07321
G1 X101.786 Y131.537 E-.12672
G1 X101.513 Y131.607 E-.10686
; WIPE_END
G1 E-.02 F1800
G1 X100.721 Y139.199 Z1.4 F36000
G1 X100.344 Y142.816 Z1.4
G1 Z1
G1 E.4 F1800
; LINE_WIDTH: 1.04614
G1 F7783.159
G1 X100.53 Y142.833 E.01233
; LINE_WIDTH: 1.01196
G1 F8055.278
G1 X100.779 Y142.856 E.01593
; LINE_WIDTH: 0.966239
G1 F8450.456
G1 X101.028 Y142.879 E.01519
; LINE_WIDTH: 0.920521
G1 F8886.407
G1 X101.277 Y142.901 E.01444
; LINE_WIDTH: 0.874804
G1 F9369.786
G1 X101.526 Y142.924 E.0137
; LINE_WIDTH: 0.829086
G1 F9908.778
G1 X101.775 Y142.947 E.01295
; LINE_WIDTH: 0.783369
G1 F10513.563
G1 X102.024 Y142.97 E.01221
; LINE_WIDTH: 0.737651
G1 F11196.976
G1 X102.273 Y142.993 E.01146
; LINE_WIDTH: 0.691934
G1 F11975.411
G1 X102.522 Y143.016 E.01072
; LINE_WIDTH: 0.646216
G1 F12870.172
G1 X103.01 Y143.017 E.01949
; WIPE_START
G1 X102.522 Y143.016 E-.18565
G1 X102.273 Y142.993 E-.095
G1 X102.024 Y142.97 E-.095
G1 X102.012 Y142.969 E-.00435
; WIPE_END
G1 E-.02 F1800
G1 X104.582 Y135.782 Z1.4 F36000
G1 X105.22 Y133.996 Z1.4
G1 Z1
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.22 Y136.338 E.08944
G3 X106.983 Y138.371 I-4.942 J6.068 E.10322
G3 X107.45 Y141.67 I-4.311 J2.293 E.12982
G1 X107.334 Y141.945 E.01143
G1 X113.134 Y141.945 E.22144
G2 X114.991 Y136.957 I-3.163 J-4.017 E.21375
G2 X114.013 Y135.543 I-3.291 J1.231 E.06631
G3 X111.933 Y133.658 I10.713 J-13.909 E.10728
G3 X110.835 Y129.416 I3.897 J-3.273 E.17294
G3 X111.971 Y127.926 I2.644 J.837 E.07298
G2 X116.943 Y124.691 I-37.264 J-62.724 E.22654
G2 X120.452 Y125.332 I2.082 J-1.473 E.1508
G1 X122.272 Y123.803 E.09076
G3 X122.333 Y127.06 I-3.573 J1.696 E.12819
G3 X121.554 Y128.002 I-3.155 J-1.815 E.04691
G2 X119.474 Y129.887 I10.712 J13.908 E.10728
G2 X118.375 Y134.129 I3.897 J3.273 E.17294
G2 X119.353 Y135.543 I3.292 J-1.231 E.06631
G3 X121.433 Y137.428 I-10.714 J13.91 E.10728
G3 X122.532 Y141.67 I-3.897 J3.273 E.17294
G1 X122.415 Y141.945 E.01143
G1 X128.216 Y141.945 E.22144
G2 X130.072 Y136.957 I-3.163 J-4.017 E.21375
G2 X129.094 Y135.543 I-3.291 J1.231 E.06631
G3 X127.015 Y133.658 I10.714 J-13.91 E.10728
G3 X125.916 Y129.416 I3.897 J-3.273 E.17294
G3 X126.894 Y128.002 I3.291 J1.231 E.06631
G2 X128.974 Y126.117 I-10.712 J-13.908 E.10728
G2 X130.072 Y121.875 I-3.897 J-3.273 E.17294
G2 X129.094 Y120.462 I-3.292 J1.231 E.06631
G3 X127.015 Y118.576 I10.714 J-13.91 E.10728
G3 X125.953 Y116.45 I4.662 J-3.657 E.09135
G1 X126.637 Y115.65 E.04022
G1 X128.635 Y118.342 E.12799
G2 X135.336 Y121.218 I5.944 J-4.604 E.29069
G3 X136.732 Y122.691 I-4.811 J5.96 E.0777
G3 X137.527 Y124.232 I-7.751 J4.973 E.06631
G3 X137.414 Y127.06 I-3.803 J1.264 E.11043
G3 X136.635 Y128.002 I-3.154 J-1.814 E.04691
G2 X134.555 Y129.887 I10.713 J13.909 E.10728
G2 X133.457 Y134.129 I3.897 J3.273 E.17294
G2 X134.434 Y135.543 I3.291 J-1.231 E.06631
G3 X136.514 Y137.428 I-10.713 J13.909 E.10728
G3 X137.613 Y141.67 I-3.897 J3.273 E.17294
G1 X137.497 Y141.945 E.01143
G1 X143.297 Y141.945 E.22145
G2 X145.154 Y136.957 I-3.163 J-4.017 E.21375
G2 X144.176 Y135.543 I-3.291 J1.231 E.06631
G3 X142.096 Y133.658 I10.713 J-13.909 E.10728
G3 X140.929 Y129.73 I4.024 J-3.333 E.16075
G2 X142.317 Y131.617 I99.985 J-72.071 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 1.16
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.725 Y130.811 E-.38
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
G1 X123.102 Y121.722
G1 Z1.16
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.053 Y122.066 E.01323
G1 X122.866 Y122.51 E.0184
G1 X122.566 Y122.905 E.01893
G3 X121.602 Y123.724 I-9.024 J-9.635 E.04834
G1 X120.069 Y125.009 E.07636
G1 X119.602 Y125.3 E.02103
G3 X117.739 Y125.139 I-.78 J-1.844 E.07428
G1 X117.323 Y124.767 E.02131
G1 X116.82 Y124.168 E.02988
G3 X103.724 Y131.272 I-32.414 J-44.131 E.57058
G1 X104.262 Y131.666 E.02545
G1 X104.535 Y132.089 E.01925
G1 X104.674 Y132.486 E.01604
G3 X104.724 Y133.664 I-6.476 J.868 E.04509
G1 X104.724 Y141.664 E.30543
G1 X104.629 Y142.265 E.02324
G1 X104.529 Y142.443 E.00781
G1 X154.069 Y142.443 E1.89141
G3 X143.8 Y132.698 I31.964 J-43.963 E.54204
G3 X136.271 Y120.545 I41.652 J-34.212 E.54742
G3 X133.272 Y120.651 I-1.756 J-7.184 E.11537
G1 X132.428 Y120.434 E.03328
G1 X131.481 Y120.051 E.03902
G1 X130.602 Y119.535 E.03892
G1 X129.808 Y118.894 E.03894
G1 X129.118 Y118.146 E.03884
G3 X127.853 Y116.453 I125.011 J-94.717 E.0807
G1 X126.662 Y114.847 E.07636
G3 X122.167 Y119.683 I-42.422 J-34.922 E.25223
G1 X122.671 Y120.284 E.02996
G1 X122.883 Y120.594 E.01433
G1 X123.074 Y121.089 E.02028
G1 X123.126 Y121.553 E.01779
G1 X123.115 Y121.633 E.00312
G1 X122.527 Y121.687 F36000
G1 F13446.369
G1 X122.461 Y121.998 E.01213
G1 X122.248 Y122.382 E.01678
G1 X122.053 Y122.582 E.01067
G1 X119.693 Y124.56 E.11755
G1 X119.366 Y124.764 E.01471
G1 X118.971 Y124.87 E.01563
G1 X118.533 Y124.849 E.01674
G1 X118.087 Y124.667 E.01837
G1 X117.771 Y124.391 E.01601
G1 X116.914 Y123.368 E.05098
G3 X104.836 Y130.208 I-32.511 J-43.32 E.53137
G1 X104.061 Y130.508 E.03174
G1 X103.689 Y130.654 E.01527
G1 F12162.432
G1 X103.317 Y130.801 E.01527
; LINE_WIDTH: 0.667226
G1 F10834.029
G1 X103.233 Y130.856 E.00413
; LINE_WIDTH: 0.714456
G1 F10513.95
G1 X103.15 Y130.911 E.00443
; LINE_WIDTH: 0.761686
G1 F10198.67
G1 X103.067 Y130.966 E.00474
; LINE_WIDTH: 0.808916
G1 F9888.189
G1 X102.983 Y131.022 E.00505
; LINE_WIDTH: 0.856146
G1 F9582.508
G1 X102.9 Y131.077 E.00536
; LINE_WIDTH: 0.903376
G1 F9061.724
G1 X102.817 Y131.132 E.00566
; LINE_WIDTH: 0.950606
G1 F8594.629
G1 X102.733 Y131.187 E.00597
; LINE_WIDTH: 0.997836
G1 F8173.327
G1 X102.65 Y131.243 E.00628
; LINE_WIDTH: 1.04507
G1 F7791.398
G1 X102.567 Y131.298 E.00659
; LINE_WIDTH: 1.0923
G1 F7443.571
G1 X102.483 Y131.353 E.0069
G1 X102.563 Y131.384 E.00588
; LINE_WIDTH: 1.04507
G1 F7791.398
G1 X102.642 Y131.414 E.00562
; LINE_WIDTH: 0.997836
G1 F8173.327
G1 X102.722 Y131.445 E.00536
; LINE_WIDTH: 0.950606
G1 F8594.629
G1 X102.802 Y131.476 E.00509
; LINE_WIDTH: 0.903376
G1 F9061.724
G1 X102.881 Y131.506 E.00483
; LINE_WIDTH: 0.856146
G1 F9582.508
G1 X102.961 Y131.537 E.00457
; LINE_WIDTH: 0.808916
G1 F10166.801
G1 X103.041 Y131.568 E.00431
; LINE_WIDTH: 0.761686
G1 F10435.05
G1 X103.12 Y131.598 E.00404
; LINE_WIDTH: 0.714456
G1 F10706.792
G1 X103.2 Y131.629 E.00378
; LINE_WIDTH: 0.667226
G1 F10982.009
G1 X103.279 Y131.659 E.00352
; LINE_WIDTH: 0.619996
G1 F12319.191
G1 X103.604 Y131.893 E.01527
G1 F13228.664
G1 X103.815 Y132.045 E.00992
G1 F13446.369
G1 X104.007 Y132.341 E.01347
G3 X104.103 Y132.618 I-1.232 J.586 E.01124
G3 X104.139 Y133.664 I-6.271 J.735 E.04
G1 X104.139 Y141.664 E.30543
G1 X104.072 Y142.085 E.01626
G1 X103.802 Y142.562 E.02094
; LINE_WIDTH: 0.632136
G1 F13173.303
G1 X103.255 Y142.947 E.02606
; LINE_WIDTH: 0.641376
G1 F12972.787
G1 X103.012 Y143.018 E.01002
G1 X104.014 Y143.029 E.03964
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X155.749 Y143.029 E1.97518
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00187
G3 X144.252 Y132.325 I29.896 J-44.054 E.59963
G3 X136.596 Y119.828 I41.14 J-33.798 E.56134
G1 X136.065 Y119.995 E.02126
G1 X135.346 Y120.129 E.02791
G3 X133.38 Y120.075 I-.778 J-7.513 E.07528
G1 X132.577 Y119.868 E.03167
G1 X131.711 Y119.513 E.03573
G1 X130.909 Y119.036 E.03564
G1 X130.185 Y118.446 E.03565
G1 X129.557 Y117.758 E.03556
G3 X127.875 Y115.499 I227.262 J-170.972 E.10754
G1 X126.683 Y113.893 E.07636
G3 X121.362 Y119.635 I-41.965 J-33.55 E.29915
G1 X122.222 Y120.661 E.05109
G1 X122.37 Y120.877 E.01002
G1 X122.514 Y121.266 E.01583
G1 X122.534 Y121.598 E.01269
G1 X121.938 Y121.586 F36000
G1 F13446.369
G1 X121.94 Y121.689 E.00396
G1 X121.823 Y121.97 E.01161
G1 X121.677 Y122.134 E.00838
G1 X119.317 Y124.111 E.11755
G1 X119.021 Y124.266 E.01276
G1 X118.726 Y124.288 E.01129
G1 X118.4 Y124.173 E.01318
G1 X118.004 Y123.757 E.02191
G1 X117.005 Y122.565 E.05941
G3 X99.505 Y131.448 I-32.579 J-42.503 E.75354
G3 X99.551 Y132.732 I-252.642 J9.81 E.04907
G1 X102.633 Y132.161 E.11965
G1 X103.076 Y132.209 E.01701
G1 X103.379 Y132.436 E.01446
G1 X103.533 Y132.751 E.01338
G3 X103.553 Y135.664 I-91.245 J2.088 E.11124
G1 X103.553 Y141.664 E.22907
G1 X103.515 Y141.904 E.00928
G1 X103.361 Y142.176 E.01195
G1 X103.048 Y142.393 E.01454
G1 X102.775 Y142.443 E.0106
G1 X102.633 Y142.43 E.00544
G1 X99.55 Y141.859 E.11969
G3 X99.51 Y143.615 I-14.854 J.535 E.0671
G1 X156.235 Y143.615 E2.16574
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.704 Y131.953 I29.69 J-44.272 E.60775
G3 X136.912 Y119.083 I40.735 J-33.459 E.57638
G1 X136.093 Y119.382 E.03328
G1 X135.329 Y119.537 E.02975
G1 X134.392 Y119.594 E.03585
G3 X132.72 Y119.3 I.446 J-7.436 E.06494
G1 X131.942 Y118.975 E.0322
G1 X131.216 Y118.537 E.03235
G1 X130.562 Y117.998 E.03237
G1 X129.996 Y117.37 E.03228
G3 X129.081 Y116.142 I206.778 J-154.949 E.05846
G1 X126.698 Y112.93 E.15272
G3 X120.57 Y119.575 I-42.601 J-33.132 E.34552
G1 X120.569 Y119.6 E.00096
G1 X121.773 Y121.037 E.07157
G1 X121.934 Y121.358 E.01373
G1 X121.937 Y121.496 E.00525
M204 S250
G1 X121.402 Y121.555 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.322 Y121.71 E.00553
G1 X118.962 Y123.688 E.09748
G1 X118.791 Y123.739 E.00566
G1 X118.644 Y123.66 E.00527
G1 X117.411 Y122.189 E.06077
; LINE_WIDTH: 0.523196
G1 X117.248 Y122 E.00795
; LINE_WIDTH: 0.544336
G1 X117.231 Y121.69 E.01032
G1 X116.672 Y122.124 E.02352
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-32.253 J-42.071 E.63217
G3 X99 Y133.396 I-53.463 J2.625 E.0747
G1 X102.734 Y132.705 E.12022
G1 X102.862 Y132.719 E.00409
G1 X102.978 Y132.83 E.0051
G3 X103 Y135.664 I-76.697 J2.009 E.08974
G1 X103 Y141.664 E.18996
G1 X102.945 Y141.813 E.00502
G1 X102.775 Y141.89 E.00591
G1 X102.734 Y141.886 E.00131
G1 X98.999 Y141.194 E.12023
G1 X98.994 Y143.017 E.05772
G2 X98.937 Y144.167 I10.173 J1.076 E.03647
G1 X156.695 Y144.167 E1.82861
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I29.092 J-44.025 E.51055
G3 X137.192 Y118.315 I40.263 J-33.067 E.49169
G1 X136.612 Y118.616 E.02069
G1 X135.913 Y118.859 E.02344
G1 X135.228 Y118.993 E.0221
G1 X134.371 Y119.041 E.02716
G1 X133.607 Y118.959 E.02433
G1 X132.867 Y118.767 E.0242
G1 X132.16 Y118.466 E.02433
G1 X131.506 Y118.067 E.02425
G1 X130.918 Y117.575 E.02427
G1 X130.41 Y117.004 E.02421
G3 X129.087 Y115.222 I987.356 J-734.385 E.07025
G1 X126.704 Y112.01 E.12664
G3 X120.192 Y119.172 I-42.396 J-32.004 E.3069
; LINE_WIDTH: 0.544336
G1 X119.622 Y119.683 E.02543
G1 X119.958 Y119.729 E.01125
; LINE_WIDTH: 0.521596
G1 X120.145 Y119.955 E.00935
; LINE_WIDTH: 0.519996
G1 X121.349 Y121.392 E.05935
G1 X121.374 Y121.469 E.00256
; WIPE_START
M204 S10000
G1 X121.322 Y121.71 E-.09367
G1 X120.744 Y122.194 E-.28633
; WIPE_END
G1 E-.02 F1800
G1 X119.622 Y119.683 Z1.56 F36000
G1 Z1.16
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X117.231 Y121.69 E.10381
; WIPE_START
M204 S10000
G1 X117.997 Y121.047 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X111.485 Y125.029 Z1.56 F36000
G1 X100.181 Y131.941 Z1.56
G1 Z1.16
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.814696
G1 F10091.497
G1 X100.5 Y131.862 E.01674
; LINE_WIDTH: 0.854416
G1 F9602.723
G1 X100.82 Y131.782 E.01759
; LINE_WIDTH: 0.894136
G1 F9159.107
G1 X101.139 Y131.703 E.01844
; LINE_WIDTH: 0.898416
G1 F9113.741
G1 X101.17 Y131.695 E.00182
; LINE_WIDTH: 0.944576
G1 F8651.565
G1 X101.484 Y131.613 E.01926
; LINE_WIDTH: 0.990736
G1 F8234.003
G1 X101.798 Y131.532 E.02023
; LINE_WIDTH: 1.0369
G1 F7854.892
G1 X102.112 Y131.45 E.02121
; LINE_WIDTH: 1.08306
G1 F7509.154
G1 X102.427 Y131.368 E.02218
; LINE_WIDTH: 1.0923
G1 F7443.571
G1 X102.483 Y131.353 E.00405
; WIPE_START
G1 X102.427 Y131.368 E-.02231
G1 X102.112 Y131.45 E-.12331
G1 X101.798 Y131.532 E-.12331
G1 X101.515 Y131.605 E-.11107
; WIPE_END
G1 E-.02 F1800
G1 X100.723 Y139.196 Z1.56 F36000
G1 X100.345 Y142.817 Z1.56
G1 Z1.16
G1 E.4 F1800
; LINE_WIDTH: 1.04388
G1 F7800.583
G1 X100.532 Y142.834 E.0124
; LINE_WIDTH: 1.00943
G1 F8076.178
G1 X100.781 Y142.857 E.01589
; LINE_WIDTH: 0.96371
G1 F8473.448
G1 X101.03 Y142.88 E.01515
; LINE_WIDTH: 0.917994
G1 F8911.825
G1 X101.279 Y142.903 E.0144
; LINE_WIDTH: 0.872277
G1 F9398.034
G1 X101.528 Y142.926 E.01366
; LINE_WIDTH: 0.826561
G1 F9940.358
G1 X101.777 Y142.948 E.01291
; LINE_WIDTH: 0.780845
G1 F10549.107
G1 X102.026 Y142.971 E.01217
; LINE_WIDTH: 0.735129
G1 F11237.279
G1 X102.275 Y142.994 E.01142
; LINE_WIDTH: 0.689412
G1 F12021.502
G1 X102.524 Y143.017 E.01068
; LINE_WIDTH: 0.643696
G1 F12923.396
G1 X103.012 Y143.018 E.01939
; WIPE_START
G1 X102.524 Y143.017 E-.18548
G1 X102.275 Y142.994 E-.095
G1 X102.026 Y142.971 E-.095
G1 X102.014 Y142.97 E-.00452
; WIPE_END
G1 E-.02 F1800
G1 X104.617 Y135.795 Z1.56 F36000
G1 X105.222 Y134.125 Z1.56
G1 Z1.16
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.222 Y136.468 E.08944
G3 X107.696 Y141.198 I-3.339 J4.758 E.21174
G3 X107.528 Y141.945 I-1.984 J-.053 E.02942
G1 X113.019 Y141.945 E.20964
G2 X115.167 Y136.957 I-3.19 J-4.33 E.21738
G2 X114.239 Y135.543 I-2.847 J.856 E.06547
G3 X112.002 Y133.658 I10.815 J-15.106 E.11182
G3 X110.659 Y129.416 I3.957 J-3.586 E.1752
G3 X111.97 Y127.926 I2.014 J.45 E.07905
G2 X116.736 Y124.842 I-38.648 J-64.954 E.21677
G2 X120.254 Y125.497 I2.081 J-1.405 E.15208
G1 X122.286 Y123.797 E.10115
G3 X122.54 Y127.06 I-3.649 J1.924 E.12846
G3 X121.78 Y128.002 I-2.676 J-1.38 E.04655
G2 X119.542 Y129.887 I10.814 J15.105 E.11182
G2 X118.2 Y134.129 I3.957 J3.586 E.1752
G2 X119.127 Y135.543 I2.847 J-.857 E.06547
G3 X121.365 Y137.428 I-10.813 J15.103 E.11182
G3 X122.707 Y141.67 I-3.957 J3.586 E.1752
G1 X122.609 Y141.945 E.01118
G1 X128.1 Y141.945 E.20964
G2 X130.248 Y136.957 I-3.19 J-4.33 E.21738
G2 X129.32 Y135.543 I-2.847 J.856 E.06547
G3 X127.083 Y133.658 I10.816 J-15.107 E.11182
G3 X125.74 Y129.416 I3.957 J-3.586 E.1752
G3 X126.668 Y128.002 I2.847 J.856 E.06547
G2 X128.905 Y126.117 I-10.814 J-15.105 E.11182
G2 X130.248 Y121.875 I-3.957 J-3.586 E.1752
G2 X129.32 Y120.462 I-2.847 J.857 E.06547
G3 X127.083 Y118.576 I10.814 J-15.105 E.11182
G3 X125.914 Y116.497 I4.777 J-4.054 E.09164
G1 X126.637 Y115.65 E.04253
G1 X128.635 Y118.342 E.12798
G2 X135.191 Y121.234 I5.933 J-4.573 E.28526
G3 X136.861 Y122.948 I-3.447 J5.03 E.09196
G1 X137.597 Y124.357 E.06073
G3 X137.621 Y127.06 I-3.726 J1.384 E.10527
G3 X136.861 Y128.002 I-2.676 J-1.38 E.04655
G2 X134.623 Y129.887 I10.814 J15.105 E.11182
G2 X133.281 Y134.129 I3.957 J3.586 E.1752
G2 X134.209 Y135.543 I2.847 J-.856 E.06547
G3 X136.446 Y137.428 I-10.813 J15.104 E.11182
G3 X137.788 Y141.67 I-3.957 J3.586 E.1752
G1 X137.69 Y141.945 E.01118
G1 X143.181 Y141.945 E.20964
G2 X145.329 Y136.957 I-3.19 J-4.33 E.21737
G2 X144.401 Y135.543 I-2.847 J.856 E.06547
G3 X142.164 Y133.658 I10.815 J-15.106 E.11182
G3 X140.803 Y129.542 I3.988 J-3.602 E.17034
G2 X142.181 Y131.436 I26.664 J-17.96 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 1.32
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
M73 P47 R10
G1 X141.593 Y130.628 E-.38
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
G1 X123.27 Y121.52
G1 Z1.32
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.273 Y121.585 E.00248
G1 X123.158 Y122.089 E.01977
G1 X122.935 Y122.516 E.01837
G1 X122.599 Y122.889 E.01917
G3 X121.447 Y123.854 I-803.038 J-956.703 E.05739
G1 X119.914 Y125.139 E.07636
G1 X119.442 Y125.432 E.0212
G1 X118.999 Y125.565 E.01769
G1 X118.571 Y125.593 E.01637
G1 X118.114 Y125.516 E.01768
G1 X117.886 Y125.434 E.00927
G1 X117.49 Y125.204 E.01746
G1 X117.168 Y124.897 E.01699
G1 X116.656 Y124.288 E.03038
G3 X103.747 Y131.262 I-32.342 J-44.428 E.56186
G1 X104.071 Y131.466 E.01462
G1 X104.416 Y131.869 E.02024
G1 X104.671 Y132.464 E.02472
G3 X104.727 Y133.667 I-6.263 J.889 E.04603
G1 X104.727 Y141.667 E.30543
G1 X104.631 Y142.268 E.02324
G1 X104.532 Y142.443 E.0077
G1 X154.069 Y142.443 E1.89127
G3 X143.8 Y132.698 I31.812 J-43.804 E.54205
G3 X136.271 Y120.545 I41.677 J-34.23 E.54744
G1 X135.454 Y120.705 E.03179
G3 X133.349 Y120.664 I-.893 J-8.171 E.0806
G1 X132.426 Y120.433 E.03632
G1 X131.483 Y120.052 E.03881
G1 X130.602 Y119.535 E.03903
G1 X129.81 Y118.896 E.03884
G1 X129.119 Y118.147 E.03892
G3 X127.854 Y116.454 I124.271 J-94.167 E.08069
G1 X126.662 Y114.848 E.07636
G3 X122.314 Y119.544 I-42.574 J-35.052 E.24448
G1 X122.826 Y120.154 E.0304
G1 X123.128 Y120.648 E.02208
G1 X123.246 Y121.038 E.01557
G1 X123.266 Y121.43 E.01498
G1 X122.67 Y121.535 F36000
G1 F13446.369
G1 X122.64 Y121.794 E.00996
G1 X122.454 Y122.183 E.01645
G1 X122.208 Y122.452 E.01393
G1 X119.538 Y124.69 E.13301
G1 X119.208 Y124.895 E.01483
G1 X118.809 Y125.001 E.01576
G3 X117.842 Y124.735 I-.116 J-1.468 E.03905
G3 X116.752 Y123.49 I14.625 J-13.904 E.06322
G3 X104.555 Y130.322 I-32.295 J-43.35 E.53523
G1 X104.067 Y130.511 E.01998
G1 X103.694 Y130.655 E.01527
G1 F12175.59
G1 X103.321 Y130.799 E.01527
; LINE_WIDTH: 0.667062
G1 F10846.448
G1 X103.237 Y130.854 E.00413
; LINE_WIDTH: 0.714128
G1 F10525.645
G1 X103.154 Y130.909 E.00444
; LINE_WIDTH: 0.761194
G1 F10209.64
G1 X103.07 Y130.965 E.00475
; LINE_WIDTH: 0.80826
G1 F9898.451
G1 X102.987 Y131.02 E.00505
; LINE_WIDTH: 0.855326
G1 F9592.079
G1 X102.903 Y131.075 E.00536
; LINE_WIDTH: 0.902392
G1 F9071.996
G1 X102.82 Y131.13 E.00567
; LINE_WIDTH: 0.949458
G1 F8605.41
G1 X102.736 Y131.186 E.00598
; LINE_WIDTH: 0.996524
G1 F8184.472
G1 X102.653 Y131.241 E.00628
; LINE_WIDTH: 1.04359
G1 F7802.793
G1 X102.569 Y131.296 E.00659
; LINE_WIDTH: 1.09066
G1 F7455.128
G1 X102.485 Y131.352 E.0069
G1 X102.565 Y131.382 E.00586
; LINE_WIDTH: 1.04359
G1 F7802.793
G1 X102.644 Y131.413 E.0056
; LINE_WIDTH: 0.996524
G1 F8184.472
G1 X102.724 Y131.443 E.00534
; LINE_WIDTH: 0.949458
G1 F8605.41
G1 X102.803 Y131.473 E.00508
; LINE_WIDTH: 0.902392
G1 F9071.996
G1 X102.883 Y131.504 E.00482
; LINE_WIDTH: 0.855326
G1 F9592.079
G1 X102.962 Y131.534 E.00456
; LINE_WIDTH: 0.80826
G1 F10175.419
G1 X103.042 Y131.565 E.00429
; LINE_WIDTH: 0.761194
G1 F10443.219
G1 X103.121 Y131.595 E.00403
; LINE_WIDTH: 0.714128
G1 F10714.499
G1 X103.201 Y131.626 E.00377
; LINE_WIDTH: 0.667062
G1 F10989.268
G1 X103.28 Y131.656 E.00351
; LINE_WIDTH: 0.619996
G1 F12577.23
G1 X103.682 Y131.904 E.01804
G1 F13446.369
G1 X103.939 Y132.211 E.01527
G1 X104.108 Y132.627 E.01714
G3 X104.141 Y133.667 I-6.532 J.727 E.03976
G1 X104.141 Y141.667 E.30543
G1 X104.074 Y142.087 E.01626
G1 X103.804 Y142.564 E.02094
; LINE_WIDTH: 0.630716
G1 F13204.669
G1 X103.257 Y142.949 E.02599
; LINE_WIDTH: 0.638856
G1 F13026.865
G1 X103.014 Y143.019 E.00997
G1 X104.015 Y143.029 E.03944
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X155.749 Y143.029 E1.97515
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.252 Y132.326 I29.804 J-43.953 E.59963
G3 X136.596 Y119.828 I41.141 J-33.799 E.56135
G1 X136.064 Y119.995 E.02126
G1 X135.346 Y120.129 E.02789
G1 X134.411 Y120.179 E.03575
G3 X133.284 Y120.054 I.632 J-10.848 E.0433
G1 X132.565 Y119.864 E.0284
G1 X131.714 Y119.514 E.03515
G1 X130.909 Y119.036 E.03574
G1 X130.187 Y118.448 E.03555
G1 X129.557 Y117.759 E.03564
G3 X127.875 Y115.499 I225.715 J-169.825 E.10756
G1 X126.683 Y113.893 E.07636
G3 X121.51 Y119.496 I-44.054 J-35.482 E.29138
G1 X122.377 Y120.531 E.05153
G1 X122.588 Y120.876 E.01545
G1 X122.694 Y121.331 E.01783
G1 X122.68 Y121.446 E.00441
G1 X122.097 Y121.509 F36000
G1 F13446.369
G1 X122.001 Y121.805 E.0119
G1 X121.832 Y122.004 E.00994
G1 X119.162 Y124.241 E.13301
G1 X118.863 Y124.397 E.01286
G1 X118.523 Y124.411 E.01299
G1 X118.194 Y124.267 E.0137
G3 X116.843 Y122.687 I27.874 J-25.199 E.07938
G3 X99.504 Y131.448 I-32.4 J-42.587 E.74583
G3 X99.553 Y132.729 I-153.764 J6.561 E.04897
G1 X102.635 Y132.159 E.11965
G1 X103.08 Y132.207 E.01707
G3 X103.536 Y132.754 I-.367 J.771 E.02808
G3 X103.555 Y135.667 I-97.47 J2.084 E.11119
G1 X103.555 Y141.667 E.22907
G1 X103.517 Y141.907 E.00928
G1 X103.363 Y142.179 E.01195
G1 X103.05 Y142.396 E.01454
G1 X102.777 Y142.445 E.0106
G1 X102.635 Y142.432 E.00544
G1 X99.552 Y141.861 E.1197
G3 X99.511 Y143.615 I-14.526 J.533 E.067
G1 X156.235 Y143.615 E2.16569
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.704 Y131.953 I29.326 J-43.872 E.60778
G3 X136.984 Y119.25 I40.691 J-33.428 E.56945
G1 X136.912 Y119.083 E.00694
G1 X136.093 Y119.382 E.03328
G3 X134.379 Y119.594 I-1.77 J-7.274 E.06605
G1 X133.546 Y119.508 E.03198
G1 X132.724 Y119.301 E.03237
G1 X131.944 Y118.976 E.03226
G1 X131.216 Y118.537 E.03246
G1 X130.564 Y118 E.03227
G1 X129.996 Y117.371 E.03236
G3 X129.081 Y116.142 I205.68 J-154.138 E.05848
G1 X126.698 Y112.93 E.15272
G3 X120.706 Y119.448 I-43.891 J-34.333 E.3384
G1 X121.928 Y120.907 E.07266
G1 X122.086 Y121.215 E.01321
G1 X122.102 Y121.42 E.00788
M204 S250
G1 X121.556 Y121.427 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.477 Y121.58 E.00545
G1 X118.807 Y123.818 E.1103
G1 X118.651 Y123.87 E.0052
G1 X118.489 Y123.79 E.00574
G1 X117.144 Y122.186 E.06627
; LINE_WIDTH: 0.523196
G1 X117.093 Y122.13 E.00241
; LINE_WIDTH: 0.544336
G1 X117.075 Y121.82 E.01032
G1 X116.672 Y122.124 E.0168
; LINE_WIDTH: 0.520516
G1 X116 Y122.631 E.02666
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.538 J-42.493 E.60556
G3 X99.002 Y133.394 I-51.709 J2.624 E.07463
G1 X102.736 Y132.702 E.12022
G1 X102.927 Y132.756 E.00628
G1 X103.002 Y132.924 E.00586
G1 X103.002 Y141.667 E.27678
G1 X102.947 Y141.815 E.00502
G1 X102.777 Y141.892 E.00591
G1 X102.736 Y141.888 E.00131
G1 X99.001 Y141.197 E.12023
G1 X98.996 Y143.017 E.05761
G2 X98.938 Y144.167 I10.177 J1.091 E.0365
G1 X156.695 Y144.167 E1.8286
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.613 J-43.498 E.51059
G3 X137.192 Y118.315 I40.597 J-33.267 E.49167
G1 X136.612 Y118.616 E.02069
G1 X135.913 Y118.859 E.02343
G1 X135.138 Y119.01 E.025
G1 X134.369 Y119.041 E.02436
G1 X133.607 Y118.959 E.02426
G1 X132.865 Y118.766 E.02427
G1 X132.162 Y118.468 E.02418
G1 X131.506 Y118.066 E.02435
G1 X130.92 Y117.577 E.02419
G1 X130.41 Y117.004 E.02427
G3 X129.087 Y115.222 I986.421 J-733.702 E.07026
G1 X126.704 Y112.01 E.12664
G3 X120.192 Y119.172 I-42.397 J-32.004 E.3069
; LINE_WIDTH: 0.544336
G1 X119.778 Y119.553 E.01871
G1 X120.113 Y119.599 E.01125
; LINE_WIDTH: 0.521596
G1 X120.163 Y119.661 E.00254
; LINE_WIDTH: 0.519996
G1 X121.505 Y121.262 E.06612
G1 X121.53 Y121.342 E.00264
; WIPE_START
M204 S10000
G1 X121.477 Y121.58 E-.09275
G1 X120.897 Y122.065 E-.28725
; WIPE_END
G1 E-.02 F1800
G1 X119.778 Y119.553 Z1.72 F36000
G1 Z1.32
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X117.075 Y121.82 E.11727
; WIPE_START
M204 S10000
G1 X117.841 Y121.177 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X111.324 Y125.149 Z1.72 F36000
G1 X100.181 Y131.94 Z1.72
G1 Z1.32
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.81261
G1 F10118.553
G1 X100.5 Y131.861 E.01669
; LINE_WIDTH: 0.852323
G1 F9627.297
G1 X100.819 Y131.781 E.01754
; LINE_WIDTH: 0.892036
G1 F9181.533
G1 X101.139 Y131.702 E.01839
; LINE_WIDTH: 0.896296
G1 F9136.156
G1 X101.17 Y131.694 E.00182
; LINE_WIDTH: 0.942426
G1 F8672.048
G1 X101.484 Y131.612 E.0192
; LINE_WIDTH: 0.988556
G1 F8252.814
G1 X101.798 Y131.531 E.02017
; LINE_WIDTH: 1.03469
G1 F7872.245
G1 X102.111 Y131.449 E.02115
; LINE_WIDTH: 1.08082
G1 F7525.228
G1 X102.425 Y131.368 E.02212
; LINE_WIDTH: 1.09066
G1 F7455.128
G1 X102.485 Y131.352 E.0043
; WIPE_START
G1 X102.425 Y131.368 E-.02372
G1 X102.111 Y131.449 E-.12322
G1 X101.798 Y131.531 E-.12322
G1 X101.518 Y131.604 E-.10985
; WIPE_END
G1 E-.02 F1800
G1 X100.724 Y139.195 Z1.72 F36000
G1 X100.346 Y142.818 Z1.72
G1 Z1.32
G1 E.4 F1800
; LINE_WIDTH: 1.0416
G1 F7818.24
G1 X100.535 Y142.835 E.01246
; LINE_WIDTH: 1.00689
G1 F8097.271
G1 X100.784 Y142.858 E.01585
; LINE_WIDTH: 0.96117
G1 F8496.67
G1 X101.033 Y142.881 E.0151
; LINE_WIDTH: 0.915454
G1 F8937.515
G1 X101.282 Y142.904 E.01436
; LINE_WIDTH: 0.869738
G1 F9426.609
G1 X101.531 Y142.927 E.01361
; LINE_WIDTH: 0.824021
G1 F9972.332
G1 X101.78 Y142.95 E.01287
; LINE_WIDTH: 0.778305
G1 F10585.123
G1 X102.028 Y142.973 E.01212
; LINE_WIDTH: 0.732589
G1 F11278.156
G1 X102.277 Y142.995 E.01138
; LINE_WIDTH: 0.686872
G1 F12068.296
G1 X102.526 Y143.018 E.01063
; LINE_WIDTH: 0.641156
G1 F12977.49
G1 X103.014 Y143.019 E.01929
; WIPE_START
G1 X102.526 Y143.018 E-.18531
G1 X102.277 Y142.995 E-.095
G1 X102.028 Y142.973 E-.095
G1 X102.016 Y142.971 E-.00469
; WIPE_END
G1 E-.02 F1800
G1 X104.652 Y135.808 Z1.72 F36000
G1 X105.224 Y134.252 Z1.72
G1 Z1.32
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.224 Y136.594 E.08944
G3 X107.817 Y141.67 I-3.225 J4.848 E.22738
G1 X107.742 Y141.945 E.01092
G1 X112.878 Y141.945 E.19611
G2 X115.358 Y136.957 I-3.307 J-4.754 E.22189
G2 X114.497 Y135.543 I-2.425 J.508 E.06445
G3 X112.067 Y133.658 I11.659 J-17.532 E.11751
G3 X110.468 Y129.416 I4.087 J-3.964 E.17801
G3 X110.816 Y128.567 I1.788 J.237 E.03542
G2 X116.57 Y124.96 I-27.885 J-50.88 E.25946
G2 X120.098 Y125.628 I2.09 J-1.388 E.15275
G1 X122.303 Y123.785 E.10972
G3 X122.769 Y127.06 I-3.814 J2.214 E.12945
G3 X122.037 Y128.002 I-2.231 J-.978 E.04604
G2 X119.608 Y129.887 I11.656 J17.529 E.11751
G2 X118.008 Y134.129 I4.087 J3.964 E.17801
G2 X118.87 Y135.543 I2.425 J-.508 E.06445
G3 X121.299 Y137.428 I-11.656 J17.529 E.11751
G3 X122.899 Y141.67 I-4.087 J3.964 E.17801
G1 X122.823 Y141.945 E.01092
G1 X127.96 Y141.945 E.19611
G2 X130.439 Y136.957 I-3.307 J-4.754 E.22189
G2 X129.578 Y135.543 I-2.425 J.508 E.06445
G3 X127.149 Y133.658 I11.659 J-17.533 E.11751
G3 X125.549 Y129.416 I4.087 J-3.964 E.17801
G3 X126.41 Y128.002 I2.425 J.508 E.06445
G2 X128.84 Y126.117 I-11.656 J-17.529 E.11751
G2 X130.439 Y121.875 I-4.087 J-3.964 E.17801
G2 X129.578 Y120.462 I-2.425 J.508 E.06445
G3 X127.149 Y118.576 I11.657 J-17.531 E.11751
G3 X125.873 Y116.544 I4.917 J-4.503 E.09209
G1 X126.638 Y115.651 E.04491
G1 X128.645 Y118.355 E.12854
G2 X135.019 Y121.24 I5.926 J-4.607 E.27793
G3 X137.472 Y124.125 I-3.257 J5.253 E.14693
G3 X137.851 Y127.06 I-3.449 J1.937 E.11577
G3 X137.119 Y128.002 I-2.231 J-.978 E.04604
G2 X134.689 Y129.887 I11.655 J17.528 E.11751
G2 X133.089 Y134.129 I4.087 J3.964 E.17801
G2 X133.951 Y135.543 I2.425 J-.508 E.06445
G3 X136.38 Y137.428 I-11.657 J17.53 E.11751
G3 X137.98 Y141.67 I-4.087 J3.964 E.17801
G1 X137.904 Y141.945 E.01092
G1 X143.041 Y141.945 E.19611
G2 X145.521 Y136.957 I-3.307 J-4.754 E.22189
G2 X144.659 Y135.543 I-2.425 J.508 E.06445
G3 X142.23 Y133.658 I11.657 J-17.53 E.11751
G3 X140.654 Y129.329 I3.868 J-3.859 E.18154
G2 X142.027 Y131.227 I28.887 J-19.447 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 1.48
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.441 Y130.417 E-.38
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
G1 X123.397 Y121.42
G1 Z1.48
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.401 Y121.494 E.00282
G1 X123.279 Y122.002 E.01993
G1 X123.051 Y122.427 E.01844
G1 X122.713 Y122.793 E.01901
G1 X119.786 Y125.247 E.14584
G1 X119.44 Y125.479 E.01587
G1 X119.031 Y125.638 E.01675
G1 X118.533 Y125.703 E.01919
G1 X118.095 Y125.653 E.01685
G1 X117.704 Y125.518 E.01579
G1 X117.283 Y125.249 E.01907
G3 X116.52 Y124.387 I14.911 J-13.955 E.04393
G3 X103.751 Y131.261 I-32.152 J-44.433 E.55529
G1 X104.072 Y131.463 E.01449
G1 X104.418 Y131.865 E.02026
G1 X104.673 Y132.461 E.02474
G3 X104.729 Y133.669 I-6.269 J.892 E.04624
G1 X104.729 Y141.669 E.30543
G1 X104.634 Y142.27 E.02324
G1 X104.536 Y142.443 E.00759
G1 X154.069 Y142.443 E1.89114
G3 X143.803 Y132.702 I31.792 J-43.783 E.54187
G3 X136.271 Y120.545 I41.635 J-34.207 E.54761
G3 X133.274 Y120.651 I-1.756 J-7.199 E.11528
G1 X132.426 Y120.433 E.03344
G1 X131.483 Y120.052 E.03883
G1 X130.602 Y119.534 E.03903
G1 X129.808 Y118.894 E.03895
G1 X129.118 Y118.146 E.03884
G3 X127.853 Y116.453 I125.203 J-94.858 E.08068
G1 X126.662 Y114.847 E.07636
G3 X122.436 Y119.428 I-42.517 J-34.976 E.23806
G1 X122.955 Y120.046 E.03081
G1 X123.222 Y120.463 E.01888
G1 X123.378 Y120.944 E.01933
G1 X123.394 Y121.331 E.01476
G1 X122.816 Y121.387 F36000
G1 F13446.369
G1 X122.818 Y121.436 E.00186
G1 X122.733 Y121.791 E.01394
G1 X122.508 Y122.173 E.01692
G3 X120.942 Y123.514 I-18.46 J-19.975 E.07874
G1 X119.409 Y124.798 E.07636
G1 X119.077 Y125.004 E.01493
G1 X118.675 Y125.11 E.01587
G1 X118.226 Y125.082 E.01715
G3 X117.658 Y124.8 I.307 J-1.329 E.02445
G3 X116.617 Y123.59 I27.202 J-24.479 E.06093
G3 X104.555 Y130.321 I-32.282 J-43.678 E.52879
G1 X104.067 Y130.511 E.01998
G1 X103.694 Y130.655 E.01527
G1 F12182.807
G1 X103.321 Y130.799 E.01527
; LINE_WIDTH: 0.666894
G1 F10853.26
G1 X103.238 Y130.854 E.00412
; LINE_WIDTH: 0.713792
G1 F10533.264
G1 X103.154 Y130.909 E.00442
; LINE_WIDTH: 0.76069
G1 F10218.012
G1 X103.071 Y130.964 E.00473
; LINE_WIDTH: 0.807588
G1 F9907.55
G1 X102.988 Y131.019 E.00503
; LINE_WIDTH: 0.854486
G1 F9601.903
G1 X102.904 Y131.074 E.00534
; LINE_WIDTH: 0.901384
G1 F9082.543
G1 X102.821 Y131.129 E.00565
; LINE_WIDTH: 0.948282
G1 F8616.483
G1 X102.738 Y131.185 E.00595
; LINE_WIDTH: 0.99518
G1 F8195.92
G1 X102.654 Y131.24 E.00626
; LINE_WIDTH: 1.04208
G1 F7814.5
G1 X102.571 Y131.295 E.00656
; LINE_WIDTH: 1.08898
G1 F7467.003
G1 X102.488 Y131.35 E.00687
G1 X102.567 Y131.38 E.00584
; LINE_WIDTH: 1.04208
G1 F7814.5
G1 X102.646 Y131.411 E.00558
; LINE_WIDTH: 0.99518
G1 F8195.92
G1 X102.726 Y131.441 E.00532
; LINE_WIDTH: 0.948282
G1 F8616.483
G1 X102.805 Y131.471 E.00506
; LINE_WIDTH: 0.901384
G1 F9082.543
G1 X102.884 Y131.502 E.0048
; LINE_WIDTH: 0.854486
G1 F9601.903
G1 X102.964 Y131.532 E.00454
; LINE_WIDTH: 0.807588
G1 F10184.261
G1 X103.043 Y131.562 E.00428
; LINE_WIDTH: 0.76069
G1 F10451.545
G1 X103.122 Y131.593 E.00402
; LINE_WIDTH: 0.713792
G1 F10722.262
G1 X103.202 Y131.623 E.00376
; LINE_WIDTH: 0.666894
G1 F10996.44
G1 X103.281 Y131.653 E.0035
; LINE_WIDTH: 0.619996
G1 F12586.517
G1 X103.684 Y131.901 E.01805
G1 F13446.369
G1 X103.941 Y132.208 E.01528
G1 X104.11 Y132.624 E.01715
G3 X104.143 Y133.669 I-6.535 J.729 E.03997
G1 X104.143 Y141.669 E.30543
G1 X104.076 Y142.09 E.01626
G1 X103.806 Y142.567 E.02094
; LINE_WIDTH: 0.629276
G1 F13236.629
G1 X103.259 Y142.951 E.02592
; LINE_WIDTH: 0.636316
G1 F13081.831
G1 X103.016 Y143.021 E.00992
G1 X104.016 Y143.029 E.03924
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X155.749 Y143.029 E1.97511
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.045 J-44.215 E.59947
G3 X136.596 Y119.828 I41.125 J-33.793 E.56148
G1 X136.067 Y119.995 E.02117
G1 X135.346 Y120.129 E.02801
G3 X133.383 Y120.076 I-.778 J-7.52 E.07519
G1 X132.575 Y119.867 E.03183
G1 X131.714 Y119.514 E.03555
G1 X130.909 Y119.036 E.03574
G1 X130.185 Y118.446 E.03566
G1 X129.557 Y117.758 E.03556
G3 X127.875 Y115.499 I227.883 J-171.433 E.10752
G1 X126.683 Y113.893 E.07636
G3 X121.633 Y119.381 I-41.961 J-33.546 E.28498
G1 X122.506 Y120.423 E.05189
G3 X122.802 Y121.051 I-1.045 J.876 E.02681
G1 X122.812 Y121.297 E.0094
G1 X122.225 Y121.408 F36000
G1 F13446.369
G1 X122.125 Y121.705 E.012
G1 X121.898 Y121.948 E.01268
G1 X119.033 Y124.349 E.14272
G1 X118.732 Y124.505 E.01295
G1 X118.432 Y124.525 E.01146
G1 X118.202 Y124.457 E.00917
G1 X117.936 Y124.253 E.01279
G1 X116.709 Y122.789 E.07295
G3 X99.495 Y131.451 I-32.528 J-43.208 E.73969
G3 X99.555 Y132.727 I-39.783 J2.537 E.04879
G1 X102.637 Y132.156 E.11965
G1 X103.081 Y132.204 E.01704
G3 X103.538 Y132.752 I-.366 J.771 E.02811
G3 X103.557 Y135.669 I-97.408 J2.086 E.11139
G1 X103.557 Y141.669 E.22907
G1 X103.519 Y141.909 E.00928
G1 X103.365 Y142.181 E.01195
G1 X103.052 Y142.398 E.01454
G1 X102.779 Y142.448 E.0106
G1 X102.637 Y142.435 E.00544
G1 X99.554 Y141.864 E.1197
G3 X99.512 Y143.615 I-14.119 J.532 E.06691
G1 X156.235 Y143.615 E2.16566
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.706 Y131.955 I29.318 J-43.864 E.60771
G3 X136.912 Y119.083 I40.723 J-33.454 E.57646
G1 X136.092 Y119.382 E.03329
G1 X135.337 Y119.535 E.02944
G1 X134.392 Y119.594 E.03614
G3 X132.719 Y119.299 I.444 J-7.424 E.065
G1 X131.944 Y118.976 E.03205
G1 X131.216 Y118.537 E.03246
G1 X130.562 Y117.998 E.03237
G1 X129.995 Y117.37 E.03228
G3 X129.081 Y116.142 I207.391 J-155.408 E.05844
G1 X126.698 Y112.93 E.15272
G3 X120.828 Y119.333 I-42.202 J-32.792 E.332
G1 X122.057 Y120.799 E.07303
G1 X122.212 Y121.095 E.01276
G1 X122.23 Y121.319 E.00859
M204 S250
G1 X121.685 Y121.322 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.605 Y121.472 E.00539
G1 X118.678 Y123.926 E.12094
G1 X118.521 Y123.978 E.00523
G1 X118.36 Y123.898 E.00571
G1 X117.435 Y122.794 E.0456
; LINE_WIDTH: 0.522786
G1 X116.965 Y122.238 E.02319
; LINE_WIDTH: 0.529506
G1 X116.925 Y122.188 E.00206
; LINE_WIDTH: 0.544336
G1 X116.947 Y121.928 E.00867
G1 X116 Y122.631 E.03919
; LINE_WIDTH: 0.519996
G3 X98.938 Y131.037 I-31.543 J-42.502 E.6055
G1 X98.943 Y131.62 E.01844
G3 X99.004 Y133.391 I-14.295 J1.375 E.05616
G1 X102.738 Y132.7 E.12022
G1 X102.928 Y132.753 E.00627
M73 P48 R10
G1 X103.004 Y132.922 E.00586
G1 X103.004 Y141.669 E.27694
G1 X102.949 Y141.818 E.00502
G1 X102.779 Y141.895 E.00591
G1 X102.738 Y141.891 E.00131
G1 X99.004 Y141.199 E.12023
G1 X98.998 Y143.015 E.05748
G2 X98.938 Y144.167 I9.996 J1.098 E.03655
G1 X156.695 Y144.167 E1.8286
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.741 J-43.639 E.51058
G3 X137.192 Y118.315 I40.595 J-33.266 E.49167
G1 X136.614 Y118.615 E.0206
G1 X135.912 Y118.859 E.02354
G1 X135.235 Y118.992 E.02184
G1 X134.371 Y119.041 E.02739
G1 X133.609 Y118.959 E.02426
G1 X132.865 Y118.766 E.02434
G1 X132.162 Y118.468 E.02419
G1 X131.506 Y118.066 E.02435
G1 X130.918 Y117.575 E.02427
G1 X130.409 Y117.004 E.02421
G3 X129.087 Y115.222 I989.507 J-735.985 E.07024
G1 X126.704 Y112.01 E.12664
G3 X121.199 Y118.226 I-44.207 J-33.599 E.26313
; LINE_WIDTH: 0.521096
G1 X120.264 Y119.106 E.04074
; LINE_WIDTH: 0.544336
G1 X119.906 Y119.445 E.01638
G1 X120.21 Y119.453 E.01009
; LINE_WIDTH: 0.526586
G1 X120.241 Y119.491 E.00158
; LINE_WIDTH: 0.521096
G1 X121.065 Y120.476 E.04074
; LINE_WIDTH: 0.519996
M73 P48 R9
G1 X121.633 Y121.154 E.02802
G1 X121.659 Y121.236 E.0027
; WIPE_START
M204 S10000
G1 X121.605 Y121.472 E-.09206
G1 X121.025 Y121.959 E-.28794
; WIPE_END
G1 E-.02 F1800
G1 X119.906 Y119.445 Z1.88 F36000
G1 Z1.48
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X116.947 Y121.928 E.12845
; WIPE_START
M204 S10000
G1 X117.713 Y121.285 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X111.346 Y125.494 Z1.88 F36000
G1 X102.488 Y131.35 Z1.88
G1 Z1.48
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.08898
G1 F7467.003
G1 X102.426 Y131.366 E.00439
; LINE_WIDTH: 1.0789
G1 F7539.06
G1 X102.112 Y131.448 E.02209
; LINE_WIDTH: 1.03274
G1 F7887.62
G1 X101.798 Y131.53 E.02112
; LINE_WIDTH: 0.986576
G1 F8269.974
G1 X101.484 Y131.611 E.02014
; LINE_WIDTH: 0.940416
G1 F8691.286
G1 X101.17 Y131.693 E.01916
; LINE_WIDTH: 0.894256
G1 F9157.829
G1 X101.139 Y131.701 E.00181
; LINE_WIDTH: 0.890016
G1 F9203.208
G1 X100.819 Y131.78 E.01836
; LINE_WIDTH: 0.850263
G1 F9651.604
G1 X100.5 Y131.86 E.01751
; LINE_WIDTH: 0.81051
G1 F10145.93
G1 X100.18 Y131.939 E.01666
; WIPE_START
G1 X100.5 Y131.86 E-.1251
G1 X100.819 Y131.78 E-.12509
G1 X101.139 Y131.701 E-.12509
G1 X101.151 Y131.698 E-.00472
; WIPE_END
G1 E-.02 F1800
G1 X100.6 Y139.31 Z1.88 F36000
G1 X100.347 Y142.819 Z1.88
G1 Z1.48
G1 E.4 F1800
; LINE_WIDTH: 1.03934
G1 F7835.821
G1 X100.537 Y142.837 E.01253
; LINE_WIDTH: 1.00436
G1 F8118.39
G1 X100.786 Y142.86 E.01581
; LINE_WIDTH: 0.958641
G1 F8519.916
G1 X101.035 Y142.882 E.01506
; LINE_WIDTH: 0.912926
G1 F8963.225
G1 X101.284 Y142.905 E.01432
; LINE_WIDTH: 0.867211
G1 F9455.202
G1 X101.533 Y142.928 E.01357
; LINE_WIDTH: 0.821496
G1 F10004.321
G1 X101.782 Y142.951 E.01283
; LINE_WIDTH: 0.775781
G1 F10621.153
G1 X102.031 Y142.974 E.01208
; LINE_WIDTH: 0.730066
G1 F11319.047
G1 X102.28 Y142.997 E.01134
; LINE_WIDTH: 0.684351
G1 F12115.106
G1 X102.529 Y143.02 E.01059
; LINE_WIDTH: 0.638636
G1 F13031.609
G1 X103.016 Y143.021 E.01919
; WIPE_START
G1 X102.529 Y143.02 E-.18514
G1 X102.28 Y142.997 E-.095
G1 X102.031 Y142.974 E-.095
G1 X102.018 Y142.973 E-.00486
; WIPE_END
G1 E-.02 F1800
G1 X104.684 Y135.821 Z1.88 F36000
G1 X105.226 Y134.366 Z1.88
G1 Z1.48
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.226 Y136.709 E.08944
G3 X108.028 Y141.67 I-3.19 J5.074 E.22667
G1 X107.981 Y141.945 E.01068
G1 X112.726 Y141.945 E.18115
G2 X115.569 Y136.957 I-3.171 J-5.111 E.22854
G2 X114.804 Y135.543 I-2.015 J.177 E.06312
G3 X112.131 Y133.658 I11.891 J-19.695 E.12497
G3 X110.257 Y129.416 I4.014 J-4.309 E.18198
G3 X110.42 Y128.775 I1.172 J-.043 E.02562
G2 X116.432 Y125.057 I-26.245 J-49.163 E.27006
G2 X119.968 Y125.737 I2.094 J-1.355 E.15358
G1 X122.322 Y123.768 E.11716
G3 X123.029 Y127.06 I-4.119 J2.608 E.13126
G3 X122.344 Y128.002 I-1.801 J-.588 E.04522
G2 X119.672 Y129.887 I11.887 J19.69 E.12497
G2 X117.797 Y134.129 I4.013 J4.309 E.18198
G2 X118.563 Y135.543 I2.015 J-.177 E.06312
G3 X121.235 Y137.428 I-11.887 J19.69 E.12497
G3 X123.11 Y141.67 I-4.014 J4.309 E.18198
G1 X123.063 Y141.945 E.01068
G1 X127.807 Y141.945 E.18115
G2 X130.65 Y136.957 I-3.171 J-5.111 E.22854
G2 X129.885 Y135.543 I-2.015 J.177 E.06312
G3 X127.212 Y133.658 I11.889 J-19.693 E.12497
G3 X125.338 Y129.416 I4.014 J-4.309 E.18198
G3 X126.103 Y128.002 I2.015 J.177 E.06312
G2 X128.776 Y126.117 I-11.887 J-19.69 E.12497
G2 X130.65 Y121.875 I-4.013 J-4.309 E.18198
G2 X129.885 Y120.462 I-2.015 J.177 E.06312
G3 X127.212 Y118.576 I11.887 J-19.69 E.12497
G3 X125.831 Y116.594 I3.766 J-4.096 E.093
G1 X126.637 Y115.65 E.04739
G1 X128.623 Y118.326 E.12724
G2 X134.828 Y121.248 I5.919 J-4.522 E.27213
G3 X136.861 Y122.948 I-5.429 J8.555 E.10145
G3 X138.002 Y125.174 I-19.492 J11.397 E.09558
G3 X138.111 Y127.06 I-3.474 J1.146 E.07292
G3 X137.426 Y128.002 I-1.801 J-.588 E.04522
G2 X134.753 Y129.887 I11.887 J19.69 E.12497
G2 X132.879 Y134.129 I4.014 J4.309 E.18198
G2 X133.644 Y135.543 I2.015 J-.177 E.06312
G3 X136.316 Y137.428 I-11.889 J19.692 E.12497
G3 X138.191 Y141.67 I-4.014 J4.309 E.18198
G1 X138.144 Y141.945 E.01068
G1 X142.889 Y141.945 E.18115
G2 X145.732 Y136.957 I-3.171 J-5.111 E.22854
G2 X144.966 Y135.543 I-2.015 J.177 E.06312
G3 X142.294 Y133.658 I11.889 J-19.693 E.12497
G3 X140.419 Y129.416 I4.013 J-4.309 E.18198
G1 X140.477 Y129.074 E.01324
G2 X141.842 Y130.978 I28.961 J-19.309 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 1.64
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.259 Y130.165 E-.38
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
G1 X123.505 Y121.343
G1 Z1.64
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.507 Y121.42 E.00295
G1 X123.38 Y121.93 E.02005
G1 X123.147 Y122.353 E.01844
G1 X122.743 Y122.768 E.02214
G1 X119.677 Y125.338 E.15272
G1 X119.33 Y125.571 E.01596
G1 X118.918 Y125.73 E.01685
G1 X118.425 Y125.794 E.019
G1 X117.947 Y125.734 E.01835
G1 X117.407 Y125.507 E.0224
G1 X117.107 Y125.281 E.01433
G1 X116.405 Y124.47 E.04096
G3 X103.734 Y131.268 I-32.549 J-45.458 E.55052
G1 X104.265 Y131.655 E.0251
G1 X104.545 Y132.088 E.01969
G1 X104.683 Y132.493 E.01632
G3 X104.731 Y133.672 I-6.722 J.86 E.04511
G1 X104.731 Y141.672 E.30543
G1 X104.636 Y142.273 E.02324
G1 X104.539 Y142.443 E.00748
G1 X154.069 Y142.443 E1.891
G3 X143.803 Y132.702 I31.961 J-43.96 E.54186
G3 X136.271 Y120.545 I42.388 J-34.674 E.54755
G1 X135.514 Y120.694 E.02948
G1 X134.433 Y120.764 E.04135
G3 X133.274 Y120.651 I.624 J-12.442 E.04446
G1 X132.426 Y120.433 E.03342
G1 X131.483 Y120.052 E.03883
G1 X130.605 Y119.536 E.03891
G1 X129.808 Y118.895 E.03905
G1 X129.117 Y118.145 E.03893
G3 X127.853 Y116.453 I83.542 J-63.731 E.08062
G1 X126.662 Y114.847 E.07636
G3 X122.538 Y119.328 I-45.399 J-37.638 E.23263
G1 X123.075 Y119.969 E.03192
G1 X123.361 Y120.44 E.02104
G1 X123.488 Y120.866 E.01697
G1 X123.502 Y121.253 E.0148
G1 X122.907 Y121.378 F36000
G1 F13446.369
G1 X122.836 Y121.713 E.01308
G1 X122.607 Y122.093 E.01693
G3 X120.834 Y123.604 I-28.446 J-31.581 E.08898
G1 X119.301 Y124.889 E.07636
G1 X118.966 Y125.096 E.01502
G1 X118.562 Y125.201 E.01596
G1 X118.115 Y125.172 E.01708
G1 X117.712 Y125.007 E.01662
G1 X117.379 Y124.72 E.0168
G1 X116.502 Y123.674 E.0521
G3 X104.849 Y130.198 I-32.632 J-44.618 E.51115
G1 X104.061 Y130.508 E.03232
G1 X103.689 Y130.654 E.01527
G1 F12183.76
G1 X103.317 Y130.801 E.01527
; LINE_WIDTH: 0.666728
G1 F10854.159
G1 X103.234 Y130.855 E.00409
; LINE_WIDTH: 0.71346
G1 F10536.453
G1 X103.151 Y130.91 E.00439
; LINE_WIDTH: 0.760192
G1 F10223.465
G1 X103.069 Y130.965 E.00469
; LINE_WIDTH: 0.806924
G1 F9915.196
G1 X102.986 Y131.02 E.00499
; LINE_WIDTH: 0.853656
G1 F9611.63
G1 X102.903 Y131.074 E.0053
; LINE_WIDTH: 0.900388
G1 F9092.988
G1 X102.821 Y131.129 E.0056
; LINE_WIDTH: 0.94712
G1 F8627.452
G1 X102.738 Y131.184 E.0059
; LINE_WIDTH: 0.993852
G1 F8207.263
G1 X102.655 Y131.239 E.0062
; LINE_WIDTH: 1.04058
G1 F7826.102
G1 X102.573 Y131.293 E.0065
; LINE_WIDTH: 1.08732
G1 F7478.775
G1 X102.49 Y131.348 E.00681
G1 X102.569 Y131.378 E.00581
; LINE_WIDTH: 1.04058
G1 F7826.102
G1 X102.648 Y131.409 E.00555
; LINE_WIDTH: 0.993852
G1 F8207.263
G1 X102.727 Y131.439 E.0053
; LINE_WIDTH: 0.94712
G1 F8627.452
G1 X102.806 Y131.469 E.00504
; LINE_WIDTH: 0.900388
G1 F9092.988
G1 X102.885 Y131.499 E.00478
; LINE_WIDTH: 0.853656
G1 F9611.63
G1 X102.965 Y131.529 E.00452
; LINE_WIDTH: 0.806924
G1 F10193.015
G1 X103.044 Y131.559 E.00426
; LINE_WIDTH: 0.760192
G1 F10459.622
G1 X103.123 Y131.59 E.00401
; LINE_WIDTH: 0.71346
G1 F10729.66
G1 X103.202 Y131.62 E.00375
; LINE_WIDTH: 0.666728
G1 F11003.15
G1 X103.281 Y131.65 E.00349
; LINE_WIDTH: 0.619996
G1 F12341.583
G1 X103.607 Y131.883 E.01527
G1 F13258.804
G1 X103.819 Y132.035 E.00999
G1 F13446.369
G1 X104.015 Y132.338 E.01377
G3 X104.112 Y132.621 I-1.234 J.581 E.01144
G3 X104.145 Y133.672 I-6.542 J.732 E.04018
G1 X104.145 Y141.672 E.30543
G1 X104.079 Y142.092 E.01626
G1 X103.808 Y142.569 E.02094
; LINE_WIDTH: 0.627836
G1 F13268.746
G1 X103.261 Y142.953 E.02585
; LINE_WIDTH: 0.633776
G1 F13137.262
G1 X103.018 Y143.022 E.00987
G1 X104.017 Y143.029 E.03904
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X155.749 Y143.029 E1.97508
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I29.896 J-44.054 E.59949
G3 X136.596 Y119.828 I41.819 J-34.218 E.56143
G1 X136.065 Y119.995 E.02125
G1 X135.345 Y120.129 E.02796
G1 X134.411 Y120.179 E.0357
G3 X133.287 Y120.054 I.623 J-10.776 E.04321
G1 X132.566 Y119.864 E.02847
G1 X131.714 Y119.514 E.03517
G1 X130.911 Y119.038 E.03563
G1 X130.185 Y118.446 E.03576
G3 X129.066 Y117.106 I8.765 J-8.453 E.06673
G1 X126.683 Y113.893 E.15272
G3 X121.736 Y119.284 I-42.673 J-34.193 E.27956
G1 X122.622 Y120.341 E.05268
G1 X122.881 Y120.833 E.02121
G1 X122.932 Y121.26 E.0164
G1 X122.926 Y121.29 E.00118
G1 X122.327 Y121.273 F36000
G1 F13446.369
G1 X122.327 Y121.387 E.00435
G1 X122.199 Y121.665 E.01169
G3 X120.457 Y123.156 I-21.838 J-23.757 E.08754
G1 X118.925 Y124.44 E.07636
G1 X118.622 Y124.597 E.01302
G1 X118.32 Y124.615 E.01152
G1 X118.018 Y124.508 E.01225
G1 X117.828 Y124.344 E.00959
G1 X116.596 Y122.874 E.07323
G3 X99.495 Y131.451 I-32.155 J-42.776 E.73437
G3 X99.557 Y132.724 I-38.332 J2.529 E.0487
G1 X102.639 Y132.154 E.11966
G1 X103.08 Y132.2 E.01691
G1 X103.384 Y132.427 E.0145
G1 X103.541 Y132.749 E.01365
G3 X103.559 Y135.672 I-97.238 J2.089 E.11159
G1 X103.559 Y141.672 E.22907
G1 X103.521 Y141.912 E.00928
G1 X103.367 Y142.184 E.01195
G1 X103.054 Y142.401 E.01454
G1 X102.781 Y142.45 E.0106
G1 X102.639 Y142.437 E.00544
G1 X99.556 Y141.866 E.1197
G1 X99.553 Y143.015 E.04388
G1 X99.512 Y143.615 E.02292
G1 X156.235 Y143.615 E2.16563
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.706 Y131.955 I29.69 J-44.271 E.60767
G3 X136.912 Y119.083 I41.355 J-33.837 E.57641
G1 X136.092 Y119.382 E.03328
G1 X135.299 Y119.542 E.03091
G1 X134.389 Y119.593 E.03478
G3 X133.436 Y119.488 I1.016 J-13.542 E.03663
G1 X132.719 Y119.299 E.0283
G1 X131.944 Y118.976 E.03205
G1 X131.218 Y118.539 E.03235
G1 X130.562 Y117.998 E.03247
G1 X129.994 Y117.369 E.03236
G3 X129.081 Y116.142 I131.074 J-98.568 E.05839
G1 X126.698 Y112.93 E.15272
G3 X120.932 Y119.236 I-42.586 J-33.149 E.32658
G1 X122.17 Y120.713 E.0736
G1 X122.329 Y121.039 E.01383
G1 X122.328 Y121.183 E.0055
M204 S250
G1 X121.793 Y121.233 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G3 X121.635 Y121.447 I-.346 J-.09 E.00865
G1 X118.569 Y124.017 E.12664
G1 X118.412 Y124.069 E.00526
G1 X118.252 Y123.989 E.00568
G1 X117.231 Y122.771 E.05031
; LINE_WIDTH: 0.522416
G1 X116.856 Y122.328 E.01844
; LINE_WIDTH: 0.536116
G1 X116.788 Y122.185 E.00519
; LINE_WIDTH: 0.544336
G1 X116.838 Y122.019 E.00578
G1 X116 Y122.631 E.03449
; LINE_WIDTH: 0.519996
G3 X98.938 Y131.037 I-31.539 J-42.496 E.6055
G1 X98.943 Y131.618 E.01839
G3 X99.006 Y133.389 I-13.839 J1.375 E.05613
G1 X102.74 Y132.697 E.12022
G1 X102.867 Y132.711 E.00407
G1 X102.985 Y132.823 E.00515
G3 X103.007 Y135.672 I-78.18 J2.015 E.09019
G1 X103.007 Y141.672 E.18996
G1 X102.951 Y141.82 E.00502
G1 X102.781 Y141.897 E.00591
G1 X102.74 Y141.893 E.00131
G1 X99.006 Y141.202 E.12023
G1 X99 Y143.014 E.05736
G2 X98.938 Y144.167 I9.81 J1.104 E.0366
G1 X156.695 Y144.167 E1.82859
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I29.093 J-44.027 E.51054
G3 X137.192 Y118.315 I40.856 J-33.422 E.49165
G1 X136.612 Y118.616 E.02069
G1 X135.913 Y118.859 E.02344
G1 X135.198 Y118.999 E.02307
G1 X134.369 Y119.041 E.02627
G1 X133.609 Y118.959 E.02419
G1 X132.865 Y118.766 E.02434
G1 X132.162 Y118.468 E.02419
G1 X131.508 Y118.068 E.02426
G1 X130.918 Y117.575 E.02435
G1 X130.409 Y117.003 E.02426
G3 X129.087 Y115.222 I607.38 J-452.293 E.07019
G1 X126.704 Y112.01 E.12664
G3 X121.199 Y118.227 I-42.08 J-31.716 E.26317
; LINE_WIDTH: 0.520636
G1 X120.373 Y119.003 E.03594
; LINE_WIDTH: 0.544336
G1 X120.015 Y119.354 E.01667
G1 X120.235 Y119.314 E.00745
; LINE_WIDTH: 0.533526
G1 X120.349 Y119.401 E.00466
; LINE_WIDTH: 0.520636
G1 X121.077 Y120.27 E.03594
; LINE_WIDTH: 0.519996
G1 X121.743 Y121.065 E.03283
G1 X121.768 Y121.146 E.0027
; WIPE_START
M204 S10000
G1 X121.714 Y121.381 E-.09146
G1 X121.635 Y121.447 E-.03913
G1 X121.132 Y121.869 E-.24941
; WIPE_END
G1 E-.02 F1800
G1 X120.015 Y119.354 Z2.04 F36000
G1 Z1.64
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X116.838 Y122.019 E.13786
; WIPE_START
M204 S10000
G1 X117.604 Y121.376 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X111.077 Y125.333 Z2.04 F36000
G1 X100.18 Y131.938 Z2.04
G1 Z1.64
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.808423
G1 F10173.281
G1 X100.5 Y131.859 E.01661
; LINE_WIDTH: 0.84817
G1 F9676.428
G1 X100.819 Y131.779 E.01746
; LINE_WIDTH: 0.887916
G1 F9225.849
G1 X101.138 Y131.7 E.01831
; LINE_WIDTH: 0.892176
G1 F9180.034
G1 X101.17 Y131.692 E.0018
; LINE_WIDTH: 0.938341
G1 F8711.236
G1 X101.484 Y131.61 E.01912
; LINE_WIDTH: 0.984506
G1 F8287.991
G1 X101.798 Y131.529 E.0201
; LINE_WIDTH: 1.03067
G1 F7903.968
G1 X102.112 Y131.447 E.02108
; LINE_WIDTH: 1.07684
G1 F7553.957
G1 X102.426 Y131.365 E.02205
; LINE_WIDTH: 1.08732
G1 F7478.775
G1 X102.49 Y131.348 E.00455
; WIPE_START
G1 X102.426 Y131.365 E-.02518
G1 X102.112 Y131.447 E-.12331
G1 X101.798 Y131.529 E-.12332
G1 X101.522 Y131.6 E-.10818
; WIPE_END
G1 E-.02 F1800
G1 X100.727 Y139.191 Z2.04 F36000
G1 X100.347 Y142.82 Z2.04
G1 Z1.64
G1 E.4 F1800
; LINE_WIDTH: 1.03708
G1 F7853.481
G1 X100.539 Y142.838 E.0126
; LINE_WIDTH: 1.00183
G1 F8139.62
G1 X100.788 Y142.861 E.01577
; LINE_WIDTH: 0.95611
G1 F8543.311
G1 X101.037 Y142.884 E.01502
; LINE_WIDTH: 0.910394
G1 F8989.137
G1 X101.286 Y142.907 E.01428
; LINE_WIDTH: 0.864677
G1 F9484.053
G1 X101.535 Y142.929 E.01353
; LINE_WIDTH: 0.818961
G1 F10036.643
G1 X101.784 Y142.952 E.01279
; LINE_WIDTH: 0.773245
G1 F10657.609
G1 X102.033 Y142.975 E.01204
; LINE_WIDTH: 0.727529
G1 F11360.482
G1 X102.282 Y142.998 E.0113
; LINE_WIDTH: 0.681813
G1 F12162.609
G1 X102.531 Y143.021 E.01055
; LINE_WIDTH: 0.636096
G1 F13086.614
G1 X103.018 Y143.022 E.01909
; WIPE_START
G1 X102.531 Y143.021 E-.18497
G1 X102.282 Y142.998 E-.095
G1 X102.033 Y142.975 E-.095
G1 X102.02 Y142.974 E-.00503
; WIPE_END
G1 E-.02 F1800
G1 X104.714 Y135.833 Z2.04 F36000
G1 X105.229 Y134.469 Z2.04
G1 Z1.64
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.229 Y136.811 E.08944
M73 P49 R9
G3 X108.239 Y141.945 I-3.225 J5.341 E.23687
G1 X112.556 Y141.945 E.1648
G2 X115.79 Y136.485 I-3.053 J-5.496 E.25397
G2 X115.196 Y135.543 I-1.359 J.198 E.0438
G2 X112.819 Y134.129 I-91.073 J150.376 E.10558
G3 X110.038 Y128.978 I3.641 J-5.293 E.23206
G2 X116.317 Y125.14 I-29.746 J-55.727 E.28115
G2 X119.862 Y125.826 I2.107 J-1.389 E.15358
G1 X122.345 Y123.749 E.12359
G3 X123.33 Y127.06 I-4.677 J3.194 E.13402
G3 X122.736 Y128.002 I-1.359 J-.198 E.0438
G3 X120.36 Y129.416 I-91.259 J-150.688 E.10558
G2 X117.576 Y134.6 I3.644 J5.296 E.23334
G2 X118.171 Y135.543 I1.359 J-.198 E.0438
G2 X120.547 Y136.957 I91.301 J-150.758 E.10558
G3 X123.321 Y141.945 I-3.449 J5.183 E.22638
G1 X127.637 Y141.945 E.1648
G2 X130.871 Y136.485 I-3.053 J-5.496 E.25397
G2 X130.277 Y135.543 I-1.359 J.198 E.0438
G2 X127.901 Y134.129 I-91.158 J150.518 E.10558
G3 X125.117 Y128.945 I3.644 J-5.296 E.23334
G3 X125.711 Y128.002 I1.359 J.198 E.0438
G3 X128.088 Y126.588 I91.259 J150.688 E.10558
G2 X130.871 Y121.404 I-3.644 J-5.296 E.23334
G2 X130.277 Y120.462 I-1.359 J.198 E.0438
G2 X127.901 Y119.048 I-91.301 J150.758 E.10558
G3 X125.787 Y116.644 I3.333 J-5.061 E.12366
G3 X124.199 Y118.366 I-40.461 J-35.724 E.08944
G1 X132.284 Y120.906 F36000
G1 F13446.283
G2 X134.589 Y121.256 I2.31 J-7.437 E.08938
G3 X136.876 Y122.968 I-3.605 J7.2 E.10965
G1 X137.624 Y124.409 E.06197
G3 X138.412 Y127.06 I-3.406 J2.454 E.10759
G3 X137.818 Y128.002 I-1.359 J-.198 E.0438
G3 X136.211 Y128.945 I-8.739 J-13.055 E.07115
G2 X132.658 Y134.6 I2.858 J5.74 E.26822
G2 X133.252 Y135.543 I1.359 J-.198 E.0438
G2 X134.858 Y136.485 I8.739 J-13.055 E.07115
G3 X138.402 Y141.945 I-2.852 J5.731 E.2607
G1 X142.718 Y141.945 E.1648
G2 X145.952 Y136.485 I-3.053 J-5.496 E.25397
G2 X145.358 Y135.543 I-1.359 J.198 E.0438
G2 X143.752 Y134.6 I-8.74 J13.056 E.07115
G3 X140.198 Y128.945 I2.858 J-5.74 E.26822
G1 X140.261 Y128.761 E.00742
G2 X141.615 Y130.672 I29.051 J-19.14 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 1.8
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X141.037 Y129.856 E-.38
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
G1 X123.595 Y121.274
G1 Z1.8
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.597 Y121.358 E.00321
G1 X123.467 Y121.867 E.02004
G1 X123.229 Y122.29 E.01853
G3 X122.651 Y122.846 I-2.627 J-2.155 E.03069
G1 X119.585 Y125.415 E.15272
G1 X119.236 Y125.649 E.01604
G1 X118.822 Y125.808 E.01693
G1 X118.333 Y125.871 E.01884
G1 X117.927 Y125.828 E.01558
G1 X117.413 Y125.64 E.02089
G1 X116.987 Y125.332 E.02004
G1 X116.306 Y124.539 E.03991
G3 X103.737 Y131.267 I-31.981 J-44.645 E.54584
G1 X104.266 Y131.651 E.02497
G1 X104.546 Y132.085 E.01971
G1 X104.685 Y132.49 E.01634
G3 X104.733 Y133.674 I-6.747 J.863 E.04532
G1 X104.733 Y141.674 E.30543
G1 X104.638 Y142.275 E.02324
G1 X104.543 Y142.443 E.00737
G1 X154.071 Y142.443 E1.89095
G3 X143.803 Y132.701 I31.455 J-43.436 E.54198
G3 X136.271 Y120.545 I41.662 J-34.225 E.54759
G1 X135.454 Y120.705 E.03178
G3 X133.275 Y120.652 I-.89 J-8.161 E.08347
G1 X132.426 Y120.433 E.03345
G1 X131.483 Y120.052 E.03884
G1 X130.602 Y119.535 E.03902
G1 X129.81 Y118.896 E.03883
G1 X129.117 Y118.145 E.03901
G3 X127.853 Y116.453 I83.2 J-63.479 E.08064
G1 X126.662 Y114.847 E.07636
G3 X122.624 Y119.244 I-42.847 J-35.294 E.22804
G1 X123.155 Y119.878 E.0316
G1 X123.426 Y120.302 E.01921
G1 X123.583 Y120.801 E.01993
G1 X123.593 Y121.184 E.01466
G1 X123.014 Y121.235 F36000
G1 F13446.369
G1 X123.016 Y121.29 E.00212
G1 X122.924 Y121.646 E.01402
G1 X122.691 Y122.025 E.01701
G3 X120.742 Y123.682 I-28.144 J-31.15 E.09767
G1 X119.209 Y124.966 E.07636
G1 X118.872 Y125.174 E.01509
G1 X118.466 Y125.279 E.01604
G1 X118.049 Y125.255 E.01595
G1 X117.666 Y125.111 E.0156
G1 X117.287 Y124.797 E.0188
G1 X116.405 Y123.745 E.0524
G3 X104.25 Y130.443 I-31.948 J-43.601 E.53134
G1 X103.69 Y130.657 E.02289
G1 F12190.881
G1 X103.317 Y130.801 E.01527
; LINE_WIDTH: 0.666562
G1 F10860.881
G1 X103.234 Y130.855 E.00407
; LINE_WIDTH: 0.713128
G1 F10543.984
G1 X103.152 Y130.91 E.00438
; LINE_WIDTH: 0.759694
G1 F10231.753
G1 X103.069 Y130.964 E.00468
; LINE_WIDTH: 0.80626
G1 F9924.214
G1 X102.987 Y131.019 E.00498
; LINE_WIDTH: 0.852826
G1 F9621.376
G1 X102.904 Y131.073 E.00528
; LINE_WIDTH: 0.899392
G1 F9103.458
G1 X102.822 Y131.128 E.00558
; LINE_WIDTH: 0.945958
G1 F8638.449
G1 X102.739 Y131.183 E.00588
; LINE_WIDTH: 0.992524
G1 F8218.639
G1 X102.657 Y131.237 E.00618
; LINE_WIDTH: 1.03909
G1 F7837.74
G1 X102.575 Y131.292 E.00648
; LINE_WIDTH: 1.08566
G1 F7490.584
G1 X102.492 Y131.346 E.00678
G1 X102.571 Y131.376 E.00579
; LINE_WIDTH: 1.03909
G1 F7837.74
G1 X102.65 Y131.407 E.00553
; LINE_WIDTH: 0.992524
G1 F8218.639
G1 X102.729 Y131.437 E.00528
; LINE_WIDTH: 0.945958
G1 F8638.449
G1 X102.808 Y131.467 E.00502
; LINE_WIDTH: 0.899392
G1 F9103.458
G1 X102.887 Y131.497 E.00476
; LINE_WIDTH: 0.852826
G1 F9621.376
G1 X102.966 Y131.527 E.00451
; LINE_WIDTH: 0.80626
G1 F10201.783
G1 X103.045 Y131.557 E.00425
; LINE_WIDTH: 0.759694
G1 F10467.831
G1 X103.124 Y131.587 E.00399
; LINE_WIDTH: 0.713128
G1 F10737.304
G1 X103.203 Y131.617 E.00374
; LINE_WIDTH: 0.666562
G1 F11010.213
G1 X103.282 Y131.647 E.00348
; LINE_WIDTH: 0.619996
G1 F12349.062
G1 X103.607 Y131.879 E.01527
G1 F13267.901
G1 X103.821 Y132.031 E.01001
G1 F13446.369
G1 X104.017 Y132.335 E.01379
G1 X104.114 Y132.618 E.01143
G3 X104.147 Y133.674 I-6.568 J.735 E.04039
G1 X104.147 Y141.674 E.30543
G1 X104.081 Y142.095 E.01626
G1 X103.81 Y142.572 E.02094
; LINE_WIDTH: 0.626406
G1 F13300.794
G1 X103.263 Y142.955 E.02578
; LINE_WIDTH: 0.631256
G1 F13192.723
G1 X103.02 Y143.023 E.00983
G1 X104.018 Y143.029 E.03884
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X155.749 Y143.029 E1.97504
G1 X155.769 Y142.919 E.00427
G3 X144.254 Y132.328 I30.018 J-44.191 E.59948
G3 X136.596 Y119.828 I41.126 J-33.793 E.56145
G1 X136.067 Y119.995 E.02115
G1 X135.346 Y120.129 E.028
G1 X134.412 Y120.179 E.03574
G3 X133.287 Y120.054 I.622 J-10.772 E.04321
G1 X132.566 Y119.864 E.02848
G1 X131.714 Y119.514 E.03518
G1 X130.909 Y119.036 E.03573
G1 X130.187 Y118.448 E.03554
G3 X129.066 Y117.106 I8.764 J-8.455 E.06683
G1 X126.683 Y113.893 E.15272
G3 X121.824 Y119.202 I-42.841 J-34.337 E.27497
G1 X122.706 Y120.255 E.05246
G1 X122.906 Y120.574 E.01439
G1 X123.005 Y120.9 E.01299
G1 X123.012 Y121.145 E.00935
G1 X122.432 Y121.188 F36000
G1 F13446.369
G1 X122.434 Y121.222 E.00129
G1 X122.348 Y121.497 E.01101
G3 X121.898 Y121.948 I-1.386 J-.932 E.02447
G1 X118.833 Y124.517 E.15272
G1 X118.528 Y124.674 E.01308
G1 X118.222 Y124.691 E.01171
G1 X117.952 Y124.6 E.01087
G1 X117.736 Y124.421 E.01073
G1 X116.5 Y122.946 E.07346
G3 X99.498 Y131.45 I-31.994 J-42.721 E.72966
G3 X99.559 Y132.722 I-41.317 J2.627 E.04864
G1 X102.641 Y132.151 E.11966
G1 X103.081 Y132.197 E.01688
G1 X103.386 Y132.424 E.01451
G1 X103.543 Y132.746 E.01367
G3 X103.561 Y135.674 I-97.041 J2.091 E.11179
G1 X103.561 Y141.674 E.22907
G1 X103.524 Y141.914 E.00928
G1 X103.369 Y142.186 E.01195
G1 X103.056 Y142.403 E.01454
G1 X102.783 Y142.453 E.0106
G1 X102.641 Y142.44 E.00544
G1 X99.561 Y141.869 E.11961
G3 X99.514 Y143.615 I-15.924 J.443 E.0667
G1 X156.235 Y143.615 E2.16559
G1 X156.415 Y142.65 E.03745
G3 X144.705 Y131.954 I29.429 J-43.975 E.60778
G3 X136.912 Y119.083 I41.137 J-33.704 E.57639
G1 X136.092 Y119.382 E.03332
G3 X132.719 Y119.299 I-1.546 J-5.73 E.1306
G1 X131.944 Y118.976 E.03206
G1 X131.216 Y118.537 E.03245
G1 X130.564 Y118 E.03226
G1 X129.995 Y117.369 E.03244
G3 X129.081 Y116.142 I130.382 J-98.056 E.0584
G1 X126.698 Y112.93 E.15272
G3 X121.019 Y119.154 I-42.558 J-33.126 E.32199
G1 X122.258 Y120.631 E.07359
G1 X122.414 Y120.933 E.01301
G1 X122.426 Y121.098 E.00631
M204 S250
G1 X121.885 Y121.157 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X121.806 Y121.304 E.00528
G1 X118.477 Y124.094 E.1375
G1 X118.319 Y124.146 E.00528
G1 X118.16 Y124.066 E.00565
G1 X117.058 Y122.751 E.0543
; LINE_WIDTH: 0.522276
G1 X116.764 Y122.405 E.01443
; LINE_WIDTH: 0.541246
G1 X116.711 Y122.16 E.00831
; LINE_WIDTH: 0.544336
G1 X116.746 Y122.096 E.00242
G1 X116 Y122.631 E.03052
; LINE_WIDTH: 0.519996
G3 X98.939 Y131.037 I-31.542 J-42.501 E.60545
G1 X98.948 Y131.644 E.01923
G3 X99.008 Y133.386 I-13.936 J1.349 E.05522
G1 X102.742 Y132.695 E.12022
G1 X102.869 Y132.708 E.00406
G1 X102.987 Y132.82 E.00515
G3 X103.009 Y135.674 I-78.043 J2.017 E.09035
G1 X103.009 Y141.674 E.18996
G1 X102.953 Y141.823 E.00502
G1 X102.783 Y141.9 E.00591
G1 X102.742 Y141.896 E.00131
G1 X99.008 Y141.205 E.12021
G1 X99.007 Y142.812 E.05091
G2 X98.936 Y144.167 I39.38 J2.752 E.04296
G1 X156.695 Y144.167 E1.82865
G1 X156.959 Y142.745 E.04579
G1 X157.025 Y142.391 E.01142
G3 X145.127 Y131.597 I28.628 J-43.511 E.51063
G3 X137.192 Y118.315 I40.596 J-33.266 E.49165
G1 X136.614 Y118.615 E.0206
G1 X135.911 Y118.86 E.02356
G1 X135.137 Y119.01 E.02497
G1 X134.369 Y119.041 E.02434
G1 X133.61 Y118.959 E.02419
G1 X132.865 Y118.766 E.02436
G1 X132.162 Y118.467 E.02418
G1 X131.506 Y118.066 E.02434
G1 X130.92 Y117.577 E.02418
G1 X130.409 Y117.003 E.02433
G3 X129.087 Y115.222 I600 J-446.819 E.0702
G1 X126.704 Y112.01 E.12664
G3 X121.199 Y118.227 I-42.084 J-31.719 E.26317
; LINE_WIDTH: 0.520436
G1 X120.465 Y118.916 E.0319
; LINE_WIDTH: 0.544336
G1 X120.107 Y119.277 E.01691
G1 X120.226 Y119.228 E.00428
; LINE_WIDTH: 0.538806
G1 X120.441 Y119.324 E.00775
; LINE_WIDTH: 0.520436
G1 X121.087 Y120.095 E.0319
; LINE_WIDTH: 0.519996
G1 X121.834 Y120.986 E.03679
G1 X121.859 Y121.071 E.00281
; WIPE_START
M204 S10000
G1 X121.806 Y121.304 E-.09078
G1 X121.223 Y121.793 E-.28922
; WIPE_END
G1 E-.02 F1800
G1 X120.107 Y119.277 Z2.2 F36000
G1 Z1.8
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X116.746 Y122.096 E.14585
; WIPE_START
M204 S10000
G1 X117.512 Y121.454 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X110.982 Y125.404 Z2.2 F36000
G1 X100.182 Y131.936 Z2.2
G1 Z1.8
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.80645
G1 F10199.281
G1 X100.5 Y131.857 E.01653
; LINE_WIDTH: 0.846103
G1 F9701.065
G1 X100.819 Y131.778 E.01737
; LINE_WIDTH: 0.885756
G1 F9249.255
G1 X101.138 Y131.699 E.01822
; LINE_WIDTH: 0.890016
G1 F9203.208
G1 X101.169 Y131.691 E.0018
; LINE_WIDTH: 0.936196
G1 F8731.954
G1 X101.483 Y131.609 E.01908
; LINE_WIDTH: 0.982376
G1 F8306.612
G1 X101.797 Y131.528 E.02006
; LINE_WIDTH: 1.02856
G1 F7920.783
G1 X102.111 Y131.446 E.02104
; LINE_WIDTH: 1.07474
G1 F7569.204
G1 X102.426 Y131.364 E.02202
; LINE_WIDTH: 1.08566
G1 F7490.584
G1 X102.492 Y131.346 E.00472
; WIPE_START
G1 X102.426 Y131.364 E-.0262
G1 X102.111 Y131.446 E-.12335
G1 X101.797 Y131.528 E-.12335
G1 X101.524 Y131.599 E-.1071
; WIPE_END
G1 E-.02 F1800
G1 X100.732 Y139.19 Z2.2 F36000
G1 X100.353 Y142.822 Z2.2
G1 Z1.8
G1 E.4 F1800
; LINE_WIDTH: 1.034
G1 F7877.679
G1 X100.542 Y142.839 E.01237
; LINE_WIDTH: 0.999286
G1 F8161.045
G1 X100.791 Y142.862 E.01573
; LINE_WIDTH: 0.953573
G1 F8566.895
G1 X101.04 Y142.885 E.01498
; LINE_WIDTH: 0.907859
G1 F9015.223
G1 X101.289 Y142.908 E.01424
; LINE_WIDTH: 0.862145
G1 F9513.068
G1 X101.538 Y142.931 E.01349
; LINE_WIDTH: 0.816431
G1 F10069.111
G1 X101.786 Y142.954 E.01275
; LINE_WIDTH: 0.770718
G1 F10694.189
G1 X102.035 Y142.976 E.012
; LINE_WIDTH: 0.725004
G1 F11402.014
G1 X102.284 Y142.999 E.01126
; LINE_WIDTH: 0.67929
G1 F12210.179
G1 X102.533 Y143.022 E.01051
; LINE_WIDTH: 0.633576
G1 F13141.647
G1 X103.02 Y143.023 E.019
; WIPE_START
G1 X102.533 Y143.022 E-.1848
G1 X102.284 Y142.999 E-.095
G1 X102.035 Y142.976 E-.095
G1 X102.022 Y142.975 E-.0052
; WIPE_END
G1 E-.02 F1800
G1 X104.741 Y135.843 Z2.2 F36000
G1 X105.231 Y134.558 Z2.2
G1 Z1.8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.231 Y136.901 E.08944
G3 X108.401 Y141.198 I-3.609 J5.98 E.2092
G1 X108.567 Y141.945 E.02922
G1 X112.356 Y141.945 E.14467
G2 X115.611 Y138.371 I-2.711 J-5.737 E.18937
G2 X116.151 Y136.485 I-7.685 J-3.222 E.07504
G1 X116.134 Y136.014 E.018
G1 X115.789 Y135.543 E.0223
G3 X112.95 Y134.129 I143.988 J-292.744 E.12111
G3 X109.884 Y129.887 I3.754 J-5.941 E.20473
G1 X109.72 Y129.151 E.02881
G2 X116.22 Y125.206 I-31.53 J-59.282 E.29044
G2 X119.769 Y125.903 I2.108 J-1.349 E.15432
G1 X122.372 Y123.727 E.12952
G3 X123.692 Y127.06 I-6.941 J4.677 E.13793
G1 X123.675 Y127.531 E.018
G1 X123.33 Y128.002 E.0223
G2 X120.49 Y129.416 I143.567 J291.897 E.12111
G2 X117.425 Y133.658 I3.754 J5.941 E.20473
G1 X117.215 Y134.6 E.03687
G1 X117.232 Y135.072 E.018
G1 X117.577 Y135.543 E.0223
G3 X120.417 Y136.957 I-143.317 J291.397 E.12111
G3 X123.482 Y141.198 I-3.754 J5.941 E.20473
G1 X123.648 Y141.945 E.02922
G1 X127.437 Y141.945 E.14467
G2 X130.692 Y138.371 I-2.711 J-5.737 E.18937
G2 X131.232 Y136.485 I-7.685 J-3.222 E.07504
G1 X131.216 Y136.014 E.018
G1 X130.87 Y135.543 E.0223
G3 X128.031 Y134.129 I143.653 J-292.071 E.12111
G3 X124.966 Y129.887 I3.754 J-5.941 E.20473
G1 X124.756 Y128.945 E.03687
G1 X124.773 Y128.473 E.018
G1 X125.118 Y128.002 E.0223
G2 X127.957 Y126.588 I-143.4 J-291.562 E.12111
G2 X131.022 Y122.347 I-3.754 J-5.941 E.20473
G1 X131.232 Y121.404 E.03687
G1 X131.216 Y120.933 E.018
G1 X130.87 Y120.462 E.0223
G3 X128.031 Y119.048 I143.317 J-291.397 E.12111
G3 X125.742 Y116.694 I3.046 J-5.252 E.12692
G3 X124.152 Y118.415 I-54.408 J-48.669 E.08944
G1 X131.992 Y120.806 F36000
G1 F13446.283
G2 X134.281 Y121.256 I2.543 J-6.886 E.08943
G3 X136.874 Y122.971 I-3.002 J7.357 E.11948
G2 X138.518 Y125.989 I61.374 J-31.484 E.13123
G3 X138.773 Y127.06 I-4.389 J1.61 E.04211
G1 X138.756 Y127.531 E.018
G1 X138.411 Y128.002 E.0223
G1 X137.505 Y128.473 E.03898
G2 X134.334 Y130.359 I4.127 J10.553 E.14149
G2 X132.297 Y134.6 I6.301 J5.637 E.18206
G1 X132.313 Y135.072 E.018
G1 X132.658 Y135.543 E.0223
G1 X133.564 Y136.014 E.03898
G3 X136.735 Y137.899 I-4.127 J10.553 E.14149
G3 X138.729 Y141.945 I-6.206 J5.573 E.17439
G1 X142.519 Y141.945 E.14467
G2 X144.276 Y140.727 I-3.465 J-6.874 E.0819
G2 X146.314 Y136.485 I-6.301 J-5.637 E.18206
G1 X146.307 Y136.302 E.007
G3 X145.281 Y135.194 I17.123 J-16.882 E.05768
G3 X141.875 Y133.186 I4.398 J-11.356 E.15162
G3 X139.837 Y128.945 I6.3 J-5.637 E.18206
G1 X139.854 Y128.473 E.018
G1 X139.969 Y128.316 E.00744
G2 X141.295 Y130.247 I29.249 J-18.658 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 1.96
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X140.729 Y129.423 E-.38
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
G1 X123.139 Y122.398
G1 Z1.96
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X122.923 Y122.617 E.01178
G3 X122.572 Y122.911 I-3.52 J-3.843 E.01747
G1 X119.507 Y125.481 E.15272
G1 X119.156 Y125.715 E.01611
G1 X118.74 Y125.875 E.017
G1 X118.254 Y125.936 E.01871
G3 X117.135 Y125.584 I.073 J-2.183 E.04534
G1 X116.76 Y125.239 E.01946
G1 X116.223 Y124.598 E.03193
G3 X103.74 Y131.265 I-31.67 J-44.273 E.54181
G1 X104.267 Y131.648 E.02485
G1 X104.54 Y132.065 E.01905
G1 X104.681 Y132.458 E.01592
G3 X104.735 Y133.677 I-6.378 J.894 E.04664
G1 X104.735 Y141.677 E.30543
G1 X104.64 Y142.278 E.02324
G1 X104.546 Y142.443 E.00726
G1 X154.071 Y142.443 E1.89082
G3 X143.8 Y132.698 I31.92 J-43.926 E.54211
G3 X136.271 Y120.545 I42.491 J-34.734 E.54737
G1 X135.454 Y120.705 E.03179
G1 X134.433 Y120.764 E.03901
G3 X132.415 Y120.431 I.304 J-8.115 E.07832
G1 X131.48 Y120.051 E.03852
G1 X130.601 Y119.534 E.03892
G1 X129.81 Y118.897 E.03879
G1 X129.117 Y118.145 E.03903
G3 X127.853 Y116.453 I83.23 J-63.502 E.08064
G1 X126.662 Y114.847 E.07636
G3 X122.697 Y119.172 I-42.814 J-35.267 E.22413
G1 X123.234 Y119.813 E.03191
G1 X123.495 Y120.217 E.01837
G1 X123.663 Y120.745 E.02116
G1 X123.674 Y121.305 E.02141
G1 X123.54 Y121.814 E.02008
G1 X123.298 Y122.236 E.01857
G1 X123.202 Y122.334 E.00523
G1 X122.665 Y122.05 F36000
G1 F13446.369
G1 X120.663 Y123.747 E.1002
G1 X119.131 Y125.032 E.07636
G1 X118.793 Y125.24 E.01515
G1 X118.384 Y125.345 E.0161
G1 X118.045 Y125.335 E.01296
G1 X117.641 Y125.205 E.01619
G1 X117.258 Y124.918 E.01827
G3 X116.323 Y123.806 I20.042 J-17.796 E.05549
G3 X105.426 Y129.963 I-31.993 J-43.9 E.47892
G1 X104.06 Y130.506 E.0561
G1 X103.688 Y130.653 E.01527
G1 F12198.022
G1 X103.317 Y130.801 E.01527
; LINE_WIDTH: 0.666396
G1 F10867.621
G1 X103.234 Y130.855 E.00406
; LINE_WIDTH: 0.712796
G1 F10551.507
G1 X103.152 Y130.909 E.00436
; LINE_WIDTH: 0.759196
G1 F10240.059
G1 X103.07 Y130.964 E.00466
; LINE_WIDTH: 0.805596
G1 F9933.26
G1 X102.988 Y131.018 E.00496
; LINE_WIDTH: 0.851996
G1 F9631.143
G1 X102.905 Y131.073 E.00526
; LINE_WIDTH: 0.898396
G1 F9113.951
G1 X102.823 Y131.127 E.00555
; LINE_WIDTH: 0.944796
G1 F8649.475
G1 X102.741 Y131.181 E.00585
; LINE_WIDTH: 0.991196
G1 F8230.045
G1 X102.659 Y131.236 E.00615
; LINE_WIDTH: 1.0376
G1 F7849.41
G1 X102.577 Y131.29 E.00645
; LINE_WIDTH: 1.084
G1 F7502.43
G1 X102.494 Y131.345 E.00675
G1 X102.573 Y131.375 E.00577
; LINE_WIDTH: 1.0376
G1 F7849.41
G1 X102.652 Y131.405 E.00551
; LINE_WIDTH: 0.991196
G1 F8230.045
G1 X102.731 Y131.434 E.00526
; LINE_WIDTH: 0.944796
G1 F8649.475
G1 X102.81 Y131.464 E.005
; LINE_WIDTH: 0.898396
G1 F9113.951
G1 X102.888 Y131.494 E.00475
; LINE_WIDTH: 0.851996
G1 F9631.143
G1 X102.967 Y131.524 E.00449
; LINE_WIDTH: 0.805596
G1 F10210.565
G1 X103.046 Y131.554 E.00424
; LINE_WIDTH: 0.759196
G1 F10476.066
G1 X103.125 Y131.584 E.00398
; LINE_WIDTH: 0.712796
G1 F10744.974
G1 X103.203 Y131.614 E.00373
; LINE_WIDTH: 0.666396
G1 F11017.301
G1 X103.282 Y131.644 E.00347
; LINE_WIDTH: 0.619996
G1 F12356.569
G1 X103.608 Y131.876 E.01527
G1 F13277.912
G1 X103.822 Y132.028 E.01003
G1 F13446.369
G1 X104.013 Y132.32 E.01333
G3 X104.111 Y132.595 I-1.228 J.594 E.01116
G3 X104.149 Y133.677 I-6.149 J.757 E.04137
G1 X104.149 Y141.677 E.30543
G1 X104.083 Y142.097 E.01626
G1 X103.812 Y142.574 E.02094
; LINE_WIDTH: 0.624976
G1 F13332.995
G1 X103.264 Y142.956 E.02571
; LINE_WIDTH: 0.628736
G1 F13248.655
G1 X103.021 Y143.025 E.00978
G1 X104.019 Y143.029 E.03864
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X155.749 Y143.029 E1.975
G1 X155.769 Y142.919 E.00427
G3 X144.252 Y132.325 I29.997 J-44.168 E.59959
G3 X136.596 Y119.828 I41.884 J-34.254 E.56129
G1 X136.065 Y119.995 E.02125
G1 X135.346 Y120.129 E.0279
G1 X134.412 Y120.179 E.03573
G1 X133.484 Y120.091 E.03557
G1 X132.578 Y119.868 E.03562
G1 X131.711 Y119.513 E.03579
G1 X130.909 Y119.036 E.03563
G1 X130.187 Y118.448 E.03551
M73 P50 R9
G3 X129.066 Y117.106 I8.76 J-8.452 E.06684
G1 X126.683 Y113.893 E.15272
G3 X121.898 Y119.132 I-43.029 J-34.497 E.27106
G1 X122.785 Y120.189 E.05268
G1 X122.967 Y120.472 E.01285
G1 X123.085 Y120.841 E.0148
G1 X123.093 Y121.233 E.01498
G1 X122.999 Y121.589 E.01405
G1 X122.762 Y121.968 E.01708
G1 X122.734 Y121.992 E.0014
G1 X122.305 Y121.582 F36000
G1 F13446.369
G1 X122.239 Y121.662 E.00394
G1 X118.754 Y124.583 E.17362
G1 X118.506 Y124.723 E.01089
G1 X118.205 Y124.764 E.01158
G1 X117.904 Y124.682 E.01192
G1 X117.658 Y124.487 E.01201
G1 X116.418 Y123.008 E.07366
G3 X99.499 Y131.449 I-32.223 J-43.408 E.72564
G3 X99.561 Y132.719 I-40.58 J2.647 E.04855
G1 X102.643 Y132.149 E.11966
G1 X103.082 Y132.195 E.01684
G1 X103.387 Y132.421 E.01452
G1 X103.542 Y132.732 E.01326
G3 X103.564 Y135.677 I-86.141 J2.105 E.11243
G1 X103.564 Y141.677 E.22907
G1 X103.526 Y141.917 E.00928
G1 X103.371 Y142.189 E.01195
G1 X103.058 Y142.406 E.01454
G1 X102.785 Y142.455 E.0106
G1 X102.643 Y142.442 E.00544
G1 X99.564 Y141.872 E.11958
G3 X99.514 Y143.615 I-18.81 J.342 E.06659
G1 X156.235 Y143.615 E2.16556
G1 X156.415 Y142.65 E.03745
G3 X144.704 Y131.953 I29.417 J-43.962 E.60784
G3 X136.912 Y119.083 I40.817 J-33.508 E.57636
G1 X136.093 Y119.382 E.03328
G3 X134.39 Y119.593 I-1.767 J-7.264 E.06565
G1 X133.549 Y119.509 E.03228
G1 X132.727 Y119.302 E.03235
G1 X131.942 Y118.974 E.0325
G1 X131.216 Y118.537 E.03235
G1 X130.564 Y118 E.03224
G1 X129.994 Y117.369 E.03246
G3 X129.081 Y116.142 I130.531 J-98.166 E.05839
G1 X126.698 Y112.93 E.15272
G3 X121.094 Y119.084 I-42.552 J-33.121 E.31808
G1 X122.336 Y120.565 E.07381
G1 X122.493 Y120.872 E.01316
G1 X122.512 Y121.161 E.01107
G1 X122.424 Y121.436 E.01103
G1 X122.362 Y121.512 E.00375
M204 S250
G1 X121.884 Y121.238 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.399 Y124.159 E.14397
G1 X118.24 Y124.212 E.0053
G1 X118.081 Y124.132 E.00563
G1 X116.91 Y122.734 E.05773
; LINE_WIDTH: 0.521736
G1 X116.687 Y122.471 E.01097
; LINE_WIDTH: 0.544266
G1 X116.677 Y122.153 E.01057
G1 X116 Y122.631 E.02753
; LINE_WIDTH: 0.519996
G3 X98.939 Y131.037 I-31.773 J-42.969 E.60541
G1 X98.949 Y131.645 E.01927
G3 X99.01 Y133.384 I-13.528 J1.347 E.05511
G1 X102.744 Y132.692 E.12023
G1 X102.871 Y132.706 E.00405
G1 X102.988 Y132.816 E.00509
G3 X103.011 Y135.677 I-75.012 J2.021 E.09058
G1 X103.011 Y141.677 E.18996
G1 X102.955 Y141.825 E.00502
G1 X102.785 Y141.902 E.00591
G1 X102.744 Y141.898 E.00131
G1 X99.011 Y141.207 E.1202
G3 X98.986 Y143.32 I-29.569 J.708 E.0669
G2 X98.936 Y144.167 I4.86 J.711 E.02692
G1 X156.695 Y144.167 E1.82865
G1 X156.959 Y142.745 E.0458
G1 X157.025 Y142.391 E.01141
G3 X145.128 Y131.598 I28.63 J-43.514 E.51062
G3 X137.192 Y118.315 I40.596 J-33.266 E.49166
G1 X136.614 Y118.615 E.0206
G1 X135.913 Y118.859 E.02353
G1 X135.137 Y119.01 E.025
G1 X134.369 Y119.041 E.02434
G1 X133.609 Y118.959 E.0242
G1 X132.868 Y118.767 E.02426
G1 X132.159 Y118.466 E.02437
G1 X131.506 Y118.066 E.02425
G1 X130.92 Y117.577 E.02417
G1 X130.408 Y117.003 E.02435
G3 X129.087 Y115.222 I606.009 J-451.273 E.07019
G1 X126.704 Y112.01 E.12664
G3 X121.199 Y118.226 I-42.087 J-31.722 E.26316
; LINE_WIDTH: 0.521736
G1 X120.545 Y118.843 E.02856
; LINE_WIDTH: 0.544336
G1 X120.21 Y119.191 E.01608
G1 X120.521 Y119.257 E.01056
; LINE_WIDTH: 0.521736
G1 X121.097 Y119.947 E.02856
; LINE_WIDTH: 0.519996
G1 X121.912 Y120.92 E.0402
G1 X121.963 Y121.093 E.0057
G1 X121.927 Y121.159 E.00238
; WIPE_START
M204 S10000
G1 X121.166 Y121.807 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.21 Y119.191 Z2.36 F36000
G1 Z1.96
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G2 X118.233 Y120.848 I317.589 J381.128 E.08575
G2 X116.678 Y122.152 I27.378 J34.218 E.06747
; WIPE_START
M204 S10000
G1 X117.444 Y121.509 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X110.911 Y125.455 Z2.36 F36000
G1 X100.182 Y131.935 Z2.36
G1 Z1.96
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.80439
G1 F10226.565
G1 X100.5 Y131.856 E.01648
; LINE_WIDTH: 0.844023
G1 F9725.986
G1 X100.819 Y131.777 E.01732
; LINE_WIDTH: 0.883656
G1 F9272.124
G1 X101.137 Y131.698 E.01817
; LINE_WIDTH: 0.887936
G1 F9225.634
G1 X101.169 Y131.69 E.00179
; LINE_WIDTH: 0.934101
G1 F8752.286
G1 X101.483 Y131.608 E.01904
; LINE_WIDTH: 0.980266
G1 F8325.14
G1 X101.797 Y131.527 E.02001
; LINE_WIDTH: 1.02643
G1 F7937.748
G1 X102.111 Y131.445 E.02099
; LINE_WIDTH: 1.0726
G1 F7584.805
G1 X102.425 Y131.363 E.02197
; LINE_WIDTH: 1.084
G1 F7502.43
G1 X102.494 Y131.345 E.00492
; WIPE_START
G1 X102.425 Y131.363 E-.0273
G1 X102.111 Y131.445 E-.12333
G1 X101.797 Y131.527 E-.12333
G1 X101.527 Y131.597 E-.10604
; WIPE_END
G1 E-.02 F1800
G1 X100.731 Y139.188 Z2.36 F36000
G1 X100.349 Y142.823 Z2.36
G1 Z1.96
G1 E.4 F1800
; LINE_WIDTH: 1.03252
G1 F7889.359
G1 X100.544 Y142.841 E.01272
; LINE_WIDTH: 0.996756
G1 F8182.499
G1 X100.793 Y142.863 E.01569
; LINE_WIDTH: 0.951041
G1 F8590.551
G1 X101.042 Y142.886 E.01494
; LINE_WIDTH: 0.905326
G1 F9041.436
G1 X101.291 Y142.909 E.01419
; LINE_WIDTH: 0.859611
G1 F9542.275
G1 X101.54 Y142.932 E.01345
; LINE_WIDTH: 0.813896
G1 F10101.853
G1 X101.789 Y142.955 E.01271
; LINE_WIDTH: 0.768181
G1 F10731.15
G1 X102.038 Y142.978 E.01196
; LINE_WIDTH: 0.722466
G1 F11444.059
G1 X102.287 Y143.001 E.01121
; LINE_WIDTH: 0.676751
G1 F12258.432
G1 X102.536 Y143.023 E.01047
; LINE_WIDTH: 0.631036
G1 F13197.588
G1 X103.021 Y143.025 E.0189
; WIPE_START
G1 X102.536 Y143.023 E-.18463
G1 X102.287 Y143.001 E-.095
G1 X102.038 Y142.978 E-.095
G1 X102.024 Y142.976 E-.00537
; WIPE_END
G1 E-.02 F1800
G1 X104.767 Y135.854 Z2.36 F36000
G1 X105.233 Y134.646 Z2.36
G1 Z1.96
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.233 Y136.988 E.08944
G3 X107.729 Y139.359 I-2.77 J5.416 E.13324
G3 X108.99 Y141.945 I-103.172 J51.917 E.10988
G1 X112.126 Y141.945 E.11972
G2 X115.269 Y139.268 I-2.253 J-5.829 E.16068
G2 X116.683 Y136.382 I-195.487 J-97.575 E.12268
G1 X117.154 Y136.051 E.02199
G3 X118.568 Y136.271 I.249 J3.049 E.05514
G3 X122.81 Y139.359 I-1.739 J6.847 E.20521
G3 X124.071 Y141.945 I-103.199 J51.931 E.10988
G1 X127.207 Y141.945 E.11972
G2 X130.351 Y139.268 I-2.253 J-5.829 E.16068
G2 X131.764 Y136.382 I-195.786 J-97.721 E.12268
G1 X132.236 Y136.051 E.02199
G3 X133.65 Y136.271 I.249 J3.049 E.05514
G3 X137.891 Y139.359 I-1.739 J6.847 E.20521
G3 X139.153 Y141.945 I-103.246 J51.953 E.10988
G1 X142.288 Y141.945 E.11972
G2 X145.432 Y139.268 I-2.253 J-5.829 E.16068
G2 X146.684 Y136.7 I-99.171 J-49.938 E.10905
G3 X144.925 Y134.802 I50.237 J-48.293 E.09879
G3 X140.719 Y131.727 I1.752 J-6.811 E.20379
G3 X139.305 Y128.841 I195.412 J-97.538 E.12268
G1 X138.834 Y128.51 E.02199
G2 X137.42 Y128.731 I-.249 J3.049 E.05514
G2 X133.178 Y131.818 I1.739 J6.847 E.20522
G2 X131.764 Y134.704 I195.412 J97.538 E.12268
G1 X131.293 Y135.035 E.02199
G3 X129.879 Y134.814 I-.249 J-3.049 E.05514
G3 X125.638 Y131.727 I1.739 J-6.847 E.20522
G3 X124.224 Y128.841 I195.711 J-97.684 E.12268
G1 X123.752 Y128.51 E.02199
G2 X122.339 Y128.731 I-.249 J3.049 E.05514
G2 X118.097 Y131.818 I1.739 J6.847 E.20522
G2 X116.683 Y134.704 I195.711 J97.684 E.12268
G1 X116.212 Y135.035 E.02199
G3 X114.798 Y134.814 I-.249 J-3.049 E.05514
G3 X110.556 Y131.727 I1.739 J-6.847 E.20522
G3 X109.386 Y129.32 I70.047 J-35.534 E.10218
G2 X111.453 Y128.218 I-28.022 J-55.025 E.08944
; WIPE_START
G1 X110.571 Y128.688 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X117.748 Y126.092 Z2.36 F36000
G1 X132.168 Y120.875 Z2.36
G1 Z1.96
G1 E.4 F1800
G1 F13446.283
G1 X132.138 Y120.867 E.00118
G3 X130.104 Y119.786 I3.32 J-8.702 E.08816
G3 X125.701 Y116.74 I1.506 J-6.881 E.2097
G3 X123.372 Y119.203 I-41.263 J-36.676 E.12944
G3 X124.001 Y122.003 I-1.841 J1.884 E.11587
G3 X122.415 Y123.69 I-4.144 J-2.305 E.08929
G1 X122.81 Y124.277 E.02701
G3 X124.224 Y127.163 I-195.561 J97.612 E.12268
G1 X124.695 Y127.494 E.02199
G2 X126.109 Y127.274 I.249 J-3.049 E.05514
G2 X130.351 Y124.186 I-1.739 J-6.847 E.20522
G2 X131.764 Y121.301 I-195.711 J-97.684 E.12268
G3 X133.017 Y121.104 I.742 J.638 E.05266
G3 X136.867 Y122.958 I-1.03 J7.064 E.16567
G2 X137.97 Y125.024 I36.799 J-18.32 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 2.12
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.499 Y124.142 E-.38
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
G1 X123.221 Y122.333
G1 Z2.12
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.058 Y122.504 E.00899
G1 X119.44 Y125.537 E.18026
G1 X119.088 Y125.772 E.01617
G1 X118.67 Y125.931 E.01707
G1 X118.188 Y125.992 E.01858
G1 X117.724 Y125.936 E.01782
G1 X117.185 Y125.714 E.02228
G1 X116.75 Y125.359 E.02143
G1 X116.151 Y124.648 E.03548
G3 X103.744 Y131.264 I-31.809 J-44.713 E.53832
G1 X104.269 Y131.644 E.02473
G1 X104.55 Y132.078 E.01976
G1 X104.689 Y132.484 E.01637
G3 X104.737 Y133.679 I-6.768 J.868 E.04573
G1 X104.737 Y141.679 E.30543
G1 X104.642 Y142.28 E.02324
G1 X104.55 Y142.443 E.00715
G1 X154.069 Y142.443 E1.8906
G3 X143.803 Y132.701 I31.963 J-43.962 E.5419
G3 X136.318 Y120.647 I41.935 J-34.39 E.54327
G1 X136.271 Y120.545 E.00428
G3 X134.434 Y120.764 I-1.885 J-7.996 E.07081
G1 X133.468 Y120.681 E.03699
G3 X132.358 Y120.414 I3.194 J-15.747 E.04361
G1 X131.483 Y120.052 E.03614
G1 X130.602 Y119.535 E.039
G1 X129.81 Y118.896 E.03884
G1 X129.117 Y118.145 E.03902
G3 X127.853 Y116.453 I124.719 J-94.498 E.08064
G1 X126.662 Y114.847 E.07636
G3 X122.758 Y119.11 I-44.348 J-36.687 E.22078
G1 X123.3 Y119.757 E.03222
G1 X123.596 Y120.236 E.02149
G1 X123.731 Y120.698 E.01837
G1 X123.74 Y121.26 E.02148
G1 X123.601 Y121.77 E.02017
G1 X123.357 Y122.191 E.0186
G1 X123.283 Y122.268 E.00406
G1 X122.802 Y121.92 F36000
G1 F13446.369
G1 X122.682 Y122.055 E.00688
G1 X119.064 Y125.088 E.18026
G1 X118.724 Y125.296 E.01521
G1 X118.314 Y125.401 E.01617
G1 X117.863 Y125.367 E.01725
G1 X117.464 Y125.199 E.01657
G1 X117.142 Y124.919 E.01627
G1 X116.252 Y123.857 E.05287
G3 X104.25 Y130.443 I-31.795 J-43.716 E.52411
G1 X103.69 Y130.657 E.02288
G1 F12200.925
G1 X103.317 Y130.8 E.01527
; LINE_WIDTH: 0.666294
G1 F10870.361
G1 X103.235 Y130.855 E.00405
; LINE_WIDTH: 0.712592
G1 F10555.116
G1 X103.153 Y130.909 E.00435
; LINE_WIDTH: 0.75889
G1 F10244.492
G1 X103.071 Y130.963 E.00464
; LINE_WIDTH: 0.805188
G1 F9938.508
G1 X102.989 Y131.017 E.00494
; LINE_WIDTH: 0.851486
G1 F9637.154
G1 X102.907 Y131.072 E.00524
; LINE_WIDTH: 0.897784
G1 F9120.412
G1 X102.825 Y131.126 E.00553
; LINE_WIDTH: 0.944082
G1 F8656.263
G1 X102.743 Y131.18 E.00583
; LINE_WIDTH: 0.99038
G1 F8237.07
G1 X102.661 Y131.234 E.00613
; LINE_WIDTH: 1.03668
G1 F7856.6
G1 X102.579 Y131.289 E.00642
; LINE_WIDTH: 1.08298
G1 F7509.726
G1 X102.497 Y131.343 E.00672
G1 X102.575 Y131.373 E.00574
; LINE_WIDTH: 1.03668
G1 F7856.6
G1 X102.654 Y131.402 E.00549
; LINE_WIDTH: 0.99038
G1 F8237.07
G1 X102.732 Y131.432 E.00523
; LINE_WIDTH: 0.944082
G1 F8656.263
G1 X102.811 Y131.462 E.00498
; LINE_WIDTH: 0.897784
G1 F9120.412
G1 X102.889 Y131.492 E.00473
; LINE_WIDTH: 0.851486
G1 F9637.154
G1 X102.968 Y131.521 E.00447
; LINE_WIDTH: 0.805188
G1 F10215.969
G1 X103.046 Y131.551 E.00422
; LINE_WIDTH: 0.75889
G1 F10480.67
G1 X103.125 Y131.581 E.00397
; LINE_WIDTH: 0.712592
G1 F10748.756
G1 X103.203 Y131.611 E.00371
; LINE_WIDTH: 0.666294
G1 F11020.228
G1 X103.282 Y131.64 E.00346
; LINE_WIDTH: 0.619996
G1 F12359.669
G1 X103.608 Y131.872 E.01527
G1 F13286.252
G1 X103.824 Y132.025 E.01008
G1 F13446.369
G1 X104.02 Y132.329 E.01383
G1 X104.118 Y132.612 E.01145
G3 X104.151 Y133.679 I-6.586 J.739 E.04079
G1 X104.151 Y141.679 E.30543
G1 X104.085 Y142.1 E.01626
G1 X103.814 Y142.577 E.02094
; LINE_WIDTH: 0.623526
G1 F13365.808
G1 X103.266 Y142.958 E.02564
; LINE_WIDTH: 0.626196
G1 F13305.512
G1 X103.023 Y143.026 E.00973
G1 X104.02 Y143.029 E.03843
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X155.749 Y143.029 E1.97497
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00187
G3 X144.254 Y132.327 I29.895 J-44.053 E.59953
G3 X136.596 Y119.828 I41.407 J-33.964 E.56142
G1 X136.064 Y119.995 E.02126
G1 X135.346 Y120.129 E.0279
G1 X134.412 Y120.179 E.03573
G1 X133.531 Y120.098 E.03377
G3 X132.485 Y119.84 I3.372 J-15.891 E.04114
G1 X131.714 Y119.514 E.03196
G1 X130.909 Y119.036 E.03572
G1 X130.187 Y118.448 E.03555
G1 X129.556 Y117.757 E.03574
G3 X127.875 Y115.499 I227.085 J-170.839 E.10748
G1 X126.683 Y113.893 E.07636
G3 X122.004 Y119.032 I-44.563 J-35.877 E.26552
G1 X121.977 Y119.09 E.00244
G1 X122.851 Y120.133 E.05196
G1 X123.106 Y120.596 E.02015
G1 X123.17 Y120.985 E.01505
G1 X123.126 Y121.352 E.01411
G1 X122.944 Y121.761 E.01712
G1 X122.862 Y121.853 E.0047
G1 X122.371 Y121.528 F36000
G1 F13446.369
G1 X122.306 Y121.606 E.00388
G1 X118.688 Y124.639 E.18026
G1 X118.438 Y124.779 E.01093
G1 X118.137 Y124.819 E.01162
G1 X117.923 Y124.775 E.00831
G1 X117.613 Y124.568 E.01422
G3 X116.349 Y123.06 I33.926 J-29.744 E.07514
G3 X99.499 Y131.449 I-31.921 J-42.999 E.72238
G3 X99.563 Y132.717 I-39.876 J2.668 E.04846
G1 X102.645 Y132.146 E.11966
G1 X103.083 Y132.192 E.01681
G1 X103.389 Y132.418 E.01454
G1 X103.547 Y132.741 E.0137
G3 X103.566 Y135.679 I-96.999 J2.095 E.11219
G1 X103.566 Y141.679 E.22907
G1 X103.528 Y141.919 E.00928
G1 X103.373 Y142.191 E.01195
G1 X103.06 Y142.408 E.01454
G1 X102.787 Y142.458 E.0106
G1 X102.645 Y142.445 E.00544
G1 X99.566 Y141.874 E.11958
G3 X99.513 Y143.615 I-15.248 J.409 E.06651
G1 X156.235 Y143.615 E2.16561
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.705 Y131.954 I29.69 J-44.271 E.60771
G3 X136.984 Y119.25 I40.947 J-33.585 E.56947
G1 X136.912 Y119.083 E.00693
G1 X136.093 Y119.382 E.03328
G1 X135.299 Y119.542 E.03091
G1 X134.39 Y119.593 E.03476
G1 X133.593 Y119.516 E.03055
G1 X132.728 Y119.302 E.03403
G1 X131.944 Y118.976 E.03242
G1 X131.216 Y118.537 E.03244
G1 X130.564 Y118 E.03227
G1 X129.995 Y117.369 E.03245
G3 X129.081 Y116.142 I206.738 J-154.919 E.0584
G1 X126.698 Y112.93 E.15272
G3 X121.157 Y119.024 I-43.33 J-33.827 E.31475
G1 X122.403 Y120.509 E.074
G1 X122.561 Y120.819 E.01329
G1 X122.578 Y121.11 E.0111
G1 X122.488 Y121.385 E.01108
G1 X122.428 Y121.458 E.00359
M204 S250
G1 X121.951 Y121.182 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.333 Y124.215 E.14948
G1 X118.173 Y124.268 E.00532
G1 X118.015 Y124.187 E.00561
G1 X116.785 Y122.72 E.06062
; LINE_WIDTH: 0.521606
G1 X116.62 Y122.527 E.00806
; LINE_WIDTH: 0.544336
G1 X116.609 Y122.207 E.01063
G1 X116 Y122.631 E.02464
; LINE_WIDTH: 0.519996
G3 X98.939 Y131.037 I-31.542 J-42.502 E.60547
G1 X98.949 Y131.646 E.0193
G3 X99.012 Y133.381 I-13.15 J1.345 E.055
G1 X102.746 Y132.69 E.12023
G1 X102.873 Y132.703 E.00404
G1 X102.991 Y132.815 E.00516
G3 X103.013 Y135.679 I-77.952 J2.021 E.09068
G1 X103.013 Y141.679 E.18996
G1 X102.957 Y141.828 E.00502
G1 X102.787 Y141.905 E.00591
G1 X102.746 Y141.901 E.00131
G1 X99.013 Y141.21 E.1202
G3 X98.937 Y143.857 I-21.511 J.709 E.0839
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I29.092 J-44.026 E.51057
G3 X137.192 Y118.315 I40.428 J-33.165 E.49166
G1 X136.612 Y118.616 E.02068
M73 P51 R9
G1 X135.913 Y118.859 E.02344
G1 X135.198 Y118.999 E.02306
G1 X134.369 Y119.041 E.02626
G1 X133.653 Y118.966 E.02282
G1 X132.866 Y118.766 E.02569
G1 X132.162 Y118.467 E.02422
G1 X131.506 Y118.067 E.02433
G1 X130.92 Y117.577 E.02419
G1 X130.409 Y117.003 E.02434
G3 X129.087 Y115.222 I992.892 J-738.485 E.0702
G1 X126.704 Y112.01 E.12664
G3 X121.2 Y118.226 I-44.208 J-33.6 E.26311
; LINE_WIDTH: 0.520346
G1 X120.611 Y118.779 E.02559
; LINE_WIDTH: 0.544336
G1 X120.251 Y119.154 E.0173
G1 X120.586 Y119.202 E.01127
; LINE_WIDTH: 0.520346
G1 X121.127 Y119.848 E.0267
; LINE_WIDTH: 0.519996
G1 X121.979 Y120.864 E.04197
G1 X122.03 Y121.038 E.00574
G1 X121.994 Y121.103 E.00235
; WIPE_START
M204 S10000
G1 X121.232 Y121.751 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.609 Y122.207 Z2.52 F36000
G1 Z2.12
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G2 X118.666 Y120.485 I-225.179 J-271.035 E.08921
G1 X120.251 Y119.154 E.0688
; WIPE_START
M204 S10000
G1 X119.485 Y119.797 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X113.023 Y123.86 Z2.52 F36000
G1 X100.182 Y131.934 Z2.52
G1 Z2.12
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.80233
G1 F10253.996
G1 X100.5 Y131.855 E.01643
; LINE_WIDTH: 0.841963
G1 F9750.794
G1 X100.819 Y131.776 E.01727
; LINE_WIDTH: 0.881596
G1 F9294.669
G1 X101.137 Y131.697 E.01812
; LINE_WIDTH: 0.885716
G1 F9249.689
G1 X101.168 Y131.689 E.00175
; LINE_WIDTH: 0.929706
G1 F8795.247
G1 X101.468 Y131.611 E.0181
; LINE_WIDTH: 0.973696
G1 F8383.368
G1 X101.768 Y131.533 E.01899
; LINE_WIDTH: 1.01769
G1 F8008.339
G1 X102.068 Y131.455 E.01988
; LINE_WIDTH: 1.06168
G1 F7665.427
G1 X102.368 Y131.377 E.02077
; LINE_WIDTH: 1.08298
G1 F7509.726
G1 X102.497 Y131.343 E.00909
; WIPE_START
G1 X102.368 Y131.377 E-.05052
G1 X102.068 Y131.455 E-.11784
G1 X101.768 Y131.533 E-.11784
G1 X101.529 Y131.595 E-.0938
; WIPE_END
G1 E-.02 F1800
G1 X100.733 Y139.186 Z2.52 F36000
G1 X100.352 Y142.824 Z2.52
G1 Z2.12
G1 E.4 F1800
; LINE_WIDTH: 1.02996
G1 F7909.645
G1 X100.546 Y142.842 E.01268
; LINE_WIDTH: 0.994226
G1 F8204.066
G1 X100.795 Y142.865 E.01564
; LINE_WIDTH: 0.94851
G1 F8614.337
G1 X101.044 Y142.887 E.0149
; LINE_WIDTH: 0.902794
G1 F9067.802
G1 X101.293 Y142.91 E.01415
; LINE_WIDTH: 0.857077
G1 F9571.662
G1 X101.542 Y142.933 E.01341
; LINE_WIDTH: 0.811361
G1 F10134.81
G1 X101.791 Y142.956 E.01266
; LINE_WIDTH: 0.765645
G1 F10768.366
G1 X102.04 Y142.979 E.01192
; LINE_WIDTH: 0.719929
G1 F11486.416
G1 X102.289 Y143.002 E.01117
; LINE_WIDTH: 0.674213
G1 F12307.068
G1 X102.538 Y143.025 E.01043
; LINE_WIDTH: 0.628496
G1 F13254.008
G1 X103.023 Y143.026 E.0188
; WIPE_START
G1 X102.538 Y143.025 E-.18446
G1 X102.289 Y143.002 E-.095
G1 X102.04 Y142.979 E-.095
G1 X102.026 Y142.978 E-.00554
; WIPE_END
G1 E-.02 F1800
G1 X104.791 Y135.864 Z2.52 F36000
G1 X105.235 Y134.722 Z2.52
G1 Z2.12
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.235 Y137.064 E.08944
G3 X106.315 Y137.702 I-1.497 J3.768 E.04808
G3 X108.671 Y140.871 I-5.577 J6.608 E.1521
G2 X109.491 Y141.945 I2.343 J-.937 E.05225
G1 X111.798 Y141.945 E.08809
G2 X115.269 Y139.399 I-1.728 J-5.994 E.16781
G3 X116.683 Y137 I155.799 J90.23 E.1063
G3 X117.626 Y136.415 I1.115 J.744 E.04368
G3 X121.396 Y137.702 I-.147 J6.597 E.15452
G3 X123.752 Y140.871 I-5.577 J6.608 E.1521
G2 X124.572 Y141.945 I2.343 J-.937 E.05225
G1 X126.88 Y141.945 E.08809
G2 X130.351 Y139.399 I-1.728 J-5.994 E.16781
G3 X131.764 Y137 I155.551 J90.084 E.1063
G3 X132.707 Y136.415 I1.115 J.744 E.04368
G3 X136.477 Y137.702 I-.147 J6.597 E.15452
G3 X138.834 Y140.871 I-5.577 J6.608 E.1521
G2 X139.654 Y141.945 I2.343 J-.937 E.05225
G1 X141.961 Y141.945 E.08809
G2 X145.432 Y139.399 I-1.728 J-5.994 E.16781
G3 X146.846 Y137 I155.551 J90.084 E.1063
G1 X146.916 Y136.936 E.00365
G3 X144.702 Y134.544 I66.123 J-63.428 E.12445
G3 X140.719 Y131.858 I1.307 J-6.234 E.18807
G2 X139.305 Y129.46 I-155.727 J90.188 E.1063
G2 X138.362 Y128.874 I-1.115 J.744 E.04368
G2 X133.178 Y131.687 I.156 J6.471 E.23381
G3 X131.764 Y134.085 I-155.945 J-90.317 E.1063
G3 X130.822 Y134.671 I-1.115 J-.744 E.04368
G3 X125.638 Y131.858 I.156 J-6.471 E.23381
G2 X124.224 Y129.46 I-155.479 J90.042 E.1063
G2 X123.281 Y128.874 I-1.115 J.744 E.04368
G2 X118.097 Y131.687 I.156 J6.471 E.23381
G3 X116.683 Y134.085 I-155.697 J-90.17 E.1063
G3 X115.741 Y134.671 I-1.115 J-.744 E.04368
G3 X110.556 Y131.858 I.156 J-6.471 E.23381
G2 X109.142 Y129.46 I-155.479 J90.042 E.1063
G1 X109.13 Y129.449 E.00063
G2 X111.203 Y128.356 I-12.807 J-26.806 E.08946
; WIPE_START
G1 X110.318 Y128.822 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X117.456 Y126.119 Z2.52 F36000
G1 X131.752 Y120.705 Z2.52
G1 Z2.12
G1 E.4 F1800
G1 F13446.283
G3 X129.758 Y119.5 I3.395 J-7.869 E.08924
G3 X125.651 Y116.795 I1.226 J-6.332 E.19262
G3 X123.433 Y119.14 I-39.696 J-35.34 E.12327
G3 X124.061 Y121.962 I-1.85 J1.893 E.1168
G3 X122.448 Y123.663 I-4.284 J-2.448 E.09033
G3 X124.224 Y126.545 I-48.511 J31.872 E.12925
G2 X125.166 Y127.13 I1.115 J-.744 E.04368
G2 X130.351 Y124.317 I-.156 J-6.471 E.23381
G3 X131.764 Y121.919 I155.697 J90.17 E.1063
G3 X132.707 Y121.334 I1.115 J.744 E.04368
G3 X136.857 Y122.94 I-.11 J6.449 E.17346
G2 X137.963 Y125.005 I55.181 J-28.229 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 2.28
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.491 Y124.123 E-.38
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
G1 X123.266 Y122.296
G1 Z2.28
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.084 Y122.482 E.00995
G1 X119.384 Y125.584 E.18433
G1 X119.03 Y125.82 E.01622
G1 X118.611 Y125.979 E.01713
G1 X118.131 Y126.039 E.01846
G3 X117.064 Y125.721 I.041 J-2.089 E.04304
G1 X116.658 Y125.367 E.02059
G1 X116.091 Y124.691 E.03367
G3 X103.747 Y131.263 I-31.725 J-44.709 E.53537
G1 X104.27 Y131.64 E.02461
G1 X104.552 Y132.075 E.01978
G1 X104.691 Y132.481 E.01638
G3 X104.739 Y133.682 I-6.785 J.871 E.04594
G1 X104.739 Y141.682 E.30543
G1 X104.644 Y142.283 E.02324
G1 X104.553 Y142.443 E.00704
G1 X154.07 Y142.443 E1.89049
G3 X143.803 Y132.701 I31.489 J-43.465 E.54193
G3 X136.269 Y120.54 I42.194 J-34.556 E.54776
G3 X133.275 Y120.652 I-1.761 J-7.045 E.11521
G1 X132.429 Y120.434 E.03332
G1 X131.48 Y120.051 E.03908
G1 X130.604 Y119.536 E.0388
G1 X129.811 Y118.897 E.03892
G1 X129.117 Y118.145 E.03903
G3 X127.853 Y116.453 I60.487 J-46.509 E.08064
G1 X126.662 Y114.847 E.07636
G3 X122.811 Y119.06 I-44.592 J-36.889 E.21798
G1 X123.356 Y119.71 E.0324
G1 X123.634 Y120.148 E.01979
G1 X123.788 Y120.658 E.02034
G1 X123.794 Y121.223 E.02157
G1 X123.653 Y121.733 E.02023
G1 X123.406 Y122.153 E.01857
G1 X123.329 Y122.232 E.00421
G1 X122.848 Y121.881 F36000
G1 F13446.369
G1 X122.708 Y122.034 E.00793
G1 X119.008 Y125.135 E.18433
G1 X118.667 Y125.344 E.01526
G1 X118.255 Y125.448 E.01622
G1 X117.835 Y125.421 E.01605
G1 X117.385 Y125.231 E.01867
G1 X117.077 Y124.955 E.01579
G1 X116.193 Y123.901 E.05251
G3 X104.25 Y130.443 I-31.753 J-43.793 E.5213
G1 X103.69 Y130.658 E.02289
G1 F12212.15
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.666064
G1 F10880.957
G1 X103.235 Y130.855 E.00404
; LINE_WIDTH: 0.712132
G1 F10566.511
G1 X103.153 Y130.909 E.00433
; LINE_WIDTH: 0.7582
G1 F10256.632
G1 X103.071 Y130.963 E.00462
; LINE_WIDTH: 0.804268
G1 F9951.391
G1 X102.989 Y131.017 E.00492
; LINE_WIDTH: 0.850336
G1 F9650.736
G1 X102.908 Y131.071 E.00521
; LINE_WIDTH: 0.896404
G1 F9135.011
G1 X102.826 Y131.125 E.00551
; LINE_WIDTH: 0.942472
G1 F8671.609
G1 X102.744 Y131.179 E.0058
; LINE_WIDTH: 0.98854
G1 F8252.953
G1 X102.662 Y131.233 E.0061
; LINE_WIDTH: 1.03461
G1 F7872.858
G1 X102.581 Y131.287 E.00639
; LINE_WIDTH: 1.08068
G1 F7526.234
G1 X102.499 Y131.341 E.00669
G1 X102.577 Y131.371 E.00572
; LINE_WIDTH: 1.03461
G1 F7872.858
G1 X102.656 Y131.401 E.00547
; LINE_WIDTH: 0.98854
G1 F8252.953
G1 X102.734 Y131.43 E.00521
; LINE_WIDTH: 0.942472
G1 F8671.609
G1 X102.813 Y131.46 E.00496
; LINE_WIDTH: 0.896404
G1 F9135.011
G1 X102.891 Y131.489 E.00471
; LINE_WIDTH: 0.850336
G1 F9650.736
G1 X102.969 Y131.519 E.00446
; LINE_WIDTH: 0.804268
G1 F10228.176
G1 X103.048 Y131.549 E.00421
; LINE_WIDTH: 0.7582
G1 F10492.539
G1 X103.126 Y131.578 E.00396
; LINE_WIDTH: 0.712132
G1 F10760.274
G1 X103.205 Y131.608 E.0037
; LINE_WIDTH: 0.666064
G1 F11031.395
G1 X103.283 Y131.638 E.00345
; LINE_WIDTH: 0.619996
G1 F12371.495
G1 X103.609 Y131.869 E.01527
G1 F13298.394
G1 X103.825 Y132.021 E.01008
G1 F13446.369
G1 X104.022 Y132.326 E.01384
G1 X104.12 Y132.609 E.01146
G3 X104.153 Y133.682 I-6.602 J.742 E.041
G1 X104.153 Y141.682 E.30543
G1 X104.087 Y142.102 E.01626
G1 X103.816 Y142.579 E.02094
; LINE_WIDTH: 0.622096
G1 F13398.328
G1 X103.268 Y142.96 E.02557
; LINE_WIDTH: 0.623676
G1 F13362.407
G1 X103.025 Y143.027 E.00968
G1 X104.02 Y143.029 E.03824
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X155.749 Y143.029 E1.97493
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.058 J-44.229 E.59949
G3 X136.598 Y119.834 I41.615 J-34.093 E.56116
G1 X135.957 Y120.021 E.0255
G1 X135.033 Y120.161 E.03567
G3 X133.383 Y120.076 I-.406 J-8.155 E.06321
G1 X132.578 Y119.868 E.03172
G1 X131.711 Y119.513 E.03578
G1 X130.911 Y119.038 E.03552
G1 X130.187 Y118.448 E.03564
G1 X129.556 Y117.757 E.03574
G3 X127.875 Y115.499 I227.287 J-170.989 E.10748
G1 X126.683 Y113.893 E.07636
G3 X122.015 Y119.021 I-42.67 J-34.155 E.26494
G1 X122.908 Y120.086 E.05306
G1 X123.159 Y120.539 E.01977
G1 X123.226 Y120.944 E.01567
G1 X123.181 Y121.312 E.01418
G1 X122.996 Y121.72 E.01709
G1 X122.909 Y121.815 E.0049
G1 X122.418 Y121.485 F36000
G1 F13446.369
G1 X122.331 Y121.585 E.00504
G1 X118.631 Y124.686 E.18433
G1 X118.381 Y124.827 E.01097
G1 X118.078 Y124.866 E.01166
G1 X117.849 Y124.815 E.00895
G1 X117.543 Y124.599 E.0143
G3 X116.29 Y123.104 I189.324 J-159.961 E.07449
G3 X99.499 Y131.449 I-31.865 J-43.048 E.71956
G3 X99.565 Y132.714 I-39.112 J2.683 E.04837
G1 X102.647 Y132.144 E.11967
G1 X103.084 Y132.189 E.01677
G1 X103.391 Y132.415 E.01456
G1 X103.549 Y132.738 E.01371
G3 X103.568 Y135.682 I-96.886 J2.098 E.11239
G1 X103.568 Y141.682 E.22907
G1 X103.53 Y141.922 E.00928
G1 X103.375 Y142.194 E.01195
G1 X103.062 Y142.411 E.01454
G1 X102.789 Y142.46 E.0106
G1 X102.647 Y142.447 E.00544
G1 X99.568 Y141.877 E.11958
G3 X99.514 Y143.615 I-14.825 J.409 E.06642
G1 X156.235 Y143.615 E2.16558
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.705 Y131.954 I29.103 J-43.629 E.60776
G3 X136.908 Y119.075 I41.169 J-33.724 E.57673
G1 X136.501 Y119.253 E.01698
G1 X135.794 Y119.458 E.0281
G1 X134.955 Y119.581 E.03238
G3 X133.437 Y119.488 I-.274 J-7.983 E.05815
G1 X132.719 Y119.299 E.02833
G1 X131.942 Y118.974 E.03217
G1 X131.218 Y118.539 E.03223
G1 X130.564 Y118 E.03236
G1 X129.995 Y117.369 E.03246
G3 X129.081 Y116.142 I206.117 J-154.458 E.0584
G1 X126.698 Y112.93 E.15272
G3 X121.211 Y118.974 I-42.17 J-32.77 E.31196
G1 X122.459 Y120.462 E.07416
G1 X122.618 Y120.775 E.0134
G1 X122.634 Y121.066 E.01114
G1 X122.542 Y121.342 E.0111
G1 X122.477 Y121.417 E.00378
M204 S250
G1 X122.007 Y121.135 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.276 Y124.263 E.15413
G1 X118.116 Y124.315 E.00534
G1 X117.958 Y124.235 E.0056
G1 X116.679 Y122.708 E.06307
; LINE_WIDTH: 0.521546
G1 X116.564 Y122.574 E.00561
; LINE_WIDTH: 0.546236
G1 X116.551 Y122.253 E.01071
G1 X116 Y122.631 E.02228
; LINE_WIDTH: 0.519996
G3 X98.939 Y131.037 I-31.538 J-42.494 E.60546
G1 X98.95 Y131.647 E.01933
G3 X99.014 Y133.379 I-12.788 J1.342 E.05489
G1 X102.748 Y132.687 E.12023
G1 X102.875 Y132.7 E.00403
G1 X102.993 Y132.813 E.00517
G3 X103.015 Y135.682 I-77.92 J2.024 E.09084
G1 X103.015 Y141.682 E.18996
G1 X102.959 Y141.83 E.00502
G1 X102.789 Y141.907 E.00591
G1 X102.748 Y141.904 E.00131
G1 X99.015 Y141.212 E.1202
G1 X99.015 Y142.631 E.04491
G1 X98.937 Y143.857 E.0389
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.612 J-43.496 E.51061
G3 X137.192 Y118.316 I40.308 J-33.094 E.49164
G3 X136.346 Y118.722 I-5.2 J-9.749 E.02972
G1 X135.64 Y118.927 E.0233
G1 X134.88 Y119.033 E.02428
G1 X134.372 Y119.038 E.0161
G1 X133.609 Y118.959 E.02427
G1 X132.868 Y118.767 E.02424
G1 X132.159 Y118.466 E.02437
G1 X131.508 Y118.068 E.02416
G1 X130.92 Y117.577 E.02427
G1 X130.409 Y117.003 E.02434
G3 X129.087 Y115.222 I986.829 J-733.996 E.0702
G1 X126.704 Y112.01 E.12664
G3 X121.199 Y118.227 I-44.203 J-33.596 E.26314
; LINE_WIDTH: 0.520146
G1 X120.668 Y118.726 E.02308
; LINE_WIDTH: 0.544336
G1 X120.306 Y119.105 E.01744
G1 X120.642 Y119.155 E.01131
; LINE_WIDTH: 0.520146
G1 X121.129 Y119.737 E.02402
; LINE_WIDTH: 0.519996
G1 X122.035 Y120.817 E.04465
G1 X122.086 Y120.992 E.00577
G1 X122.05 Y121.056 E.00231
; WIPE_START
M204 S10000
G1 X121.288 Y121.704 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.551 Y122.253 Z2.68 F36000
G1 Z2.28
G1 E.4 F1800
; LINE_WIDTH: 0.546236
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.00585
; LINE_WIDTH: 0.544336
G1 X120.306 Y119.105 E.15707
; WIPE_START
M204 S10000
G1 X119.54 Y119.748 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X113.08 Y123.814 Z2.68 F36000
G1 X100.182 Y131.933 Z2.68
G1 Z2.28
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.800256
G1 F10281.753
G1 X100.5 Y131.854 E.01638
; LINE_WIDTH: 0.839876
G1 F9776.052
G1 X100.818 Y131.775 E.01723
; LINE_WIDTH: 0.879496
G1 F9317.763
G1 X101.137 Y131.696 E.01807
; LINE_WIDTH: 0.883716
G1 F9271.47
G1 X101.168 Y131.688 E.00178
; LINE_WIDTH: 0.929866
G1 F8793.675
G1 X101.482 Y131.606 E.01894
; LINE_WIDTH: 0.976016
G1 F8362.714
G1 X101.796 Y131.525 E.01991
; LINE_WIDTH: 1.02217
G1 F7972.02
G1 X102.11 Y131.443 E.02089
; LINE_WIDTH: 1.06832
G1 F7616.201
G1 X102.424 Y131.362 E.02186
; LINE_WIDTH: 1.08068
G1 F7526.234
G1 X102.499 Y131.341 E.00531
; WIPE_START
G1 X102.424 Y131.362 E-.0296
G1 X102.11 Y131.443 E-.12325
G1 X101.796 Y131.525 E-.12325
G1 X101.531 Y131.594 E-.1039
; WIPE_END
G1 E-.02 F1800
G1 X102.52 Y139.162 Z2.68 F36000
G1 X103.025 Y143.027 Z2.68
G1 Z2.28
G1 E.4 F1800
; LINE_WIDTH: 0.625976
G1 F13310.46
G1 X102.54 Y143.026 E.0187
; LINE_WIDTH: 0.67169
G1 F12355.777
G1 X102.291 Y143.003 E.01039
; LINE_WIDTH: 0.717404
G1 F11528.876
G1 X102.042 Y142.98 E.01113
; LINE_WIDTH: 0.763117
G1 F10805.712
G1 X101.793 Y142.957 E.01188
; LINE_WIDTH: 0.808831
G1 F10167.916
G1 X101.544 Y142.934 E.01262
; LINE_WIDTH: 0.854545
G1 F9601.215
G1 X101.296 Y142.912 E.01337
; LINE_WIDTH: 0.900259
G1 F9094.348
G1 X101.047 Y142.889 E.01411
; LINE_WIDTH: 0.945973
G1 F8638.314
G1 X100.798 Y142.866 E.01486
; LINE_WIDTH: 0.991686
G1 F8225.833
G1 X100.549 Y142.843 E.0156
; LINE_WIDTH: 1.02768
G1 F7927.8
G1 X100.353 Y142.825 E.01274
; WIPE_START
G1 X100.549 Y142.843 E-.07479
G1 X100.798 Y142.866 E-.095
G1 X101.047 Y142.889 E-.095
G1 X101.296 Y142.912 E-.095
G1 X101.348 Y142.916 E-.02022
; WIPE_END
G1 E-.02 F1800
G1 X104.643 Y136.032 Z2.68 F36000
G1 X105.237 Y134.79 Z2.68
G1 Z2.28
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.237 Y137.133 E.08944
G3 X107.257 Y138.535 I-2.076 J5.147 E.0947
G3 X109.142 Y141.226 I-17.918 J14.56 E.12554
G2 X110.085 Y141.907 I1.506 J-1.092 E.04515
G2 X114.798 Y140.091 I.529 J-5.65 E.19985
G2 X116.683 Y137.4 I-17.918 J-14.56 E.12554
G3 X118.097 Y136.643 I1.57 J1.231 E.06302
G3 X122.339 Y138.535 I-.103 J5.929 E.18221
G3 X124.224 Y141.226 I-17.919 J14.56 E.12554
G2 X125.166 Y141.907 I1.506 J-1.092 E.04515
G2 X129.879 Y140.091 I.529 J-5.65 E.19985
G2 X131.764 Y137.4 I-17.921 J-14.562 E.12554
G3 X133.178 Y136.643 I1.57 J1.231 E.06302
G3 X137.42 Y138.535 I-.103 J5.929 E.18221
G3 X139.305 Y141.226 I-17.921 J14.562 E.12554
G2 X140.248 Y141.907 I1.506 J-1.092 E.04515
G2 X144.961 Y140.091 I.529 J-5.65 E.19985
G2 X146.846 Y137.4 I-17.921 J-14.562 E.12554
G1 X147.117 Y137.139 E.01436
G3 X144.531 Y134.347 I55.944 J-54.382 E.1453
G3 X141.19 Y132.55 I1.192 J-6.222 E.14709
G3 X139.305 Y129.859 I17.918 J-14.56 E.12554
G2 X137.891 Y129.102 I-1.57 J1.231 E.06302
G2 X133.65 Y130.995 I.103 J5.929 E.18221
G2 X131.764 Y133.686 I17.918 J14.56 E.12554
G3 X130.351 Y134.443 I-1.57 J-1.231 E.06302
G3 X126.109 Y132.55 I.103 J-5.929 E.18221
G3 X124.224 Y129.859 I17.921 J-14.562 E.12554
G2 X122.81 Y129.102 I-1.57 J1.231 E.06302
G2 X118.568 Y130.995 I.103 J5.929 E.18221
G2 X116.683 Y133.686 I17.921 J14.562 E.12554
G3 X115.269 Y134.443 I-1.57 J-1.231 E.06302
G3 X111.028 Y132.55 I.103 J-5.929 E.18221
G3 X109.142 Y129.859 I17.921 J-14.562 E.12554
G1 X108.858 Y129.586 E.01507
G2 X110.937 Y128.505 I-15.217 J-31.803 E.08945
; WIPE_START
G1 X110.049 Y128.966 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.476 Y124.849 Z2.68 F36000
G1 X127.989 Y117.472 Z2.68
G1 Z2.28
G1 E.4 F1800
G1 F13446.283
G2 X129.483 Y119.272 I10.922 J-7.548 E.08941
G3 X125.604 Y116.848 I.971 J-5.871 E.17926
G3 X123.486 Y119.09 I-41.514 J-37.1 E.11775
G3 X124.111 Y121.928 I-1.859 J1.898 E.11743
G3 X122.485 Y123.632 I-4.362 J-2.536 E.09075
G3 X124.224 Y126.145 I-18.996 J15 E.11674
G2 X125.638 Y126.902 I1.57 J-1.231 E.06302
G2 X129.879 Y125.01 I-.103 J-5.929 E.18221
G2 X131.764 Y122.319 I-17.921 J-14.562 E.12554
G3 X133.178 Y121.561 I1.57 J1.231 E.06302
G3 X136.853 Y122.922 I-.259 J6.341 E.15208
G2 X137.95 Y124.991 I33.958 J-16.676 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 2.44
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.482 Y124.108 E-.38
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
G1 X123.41 Y122.142
G1 Z2.44
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.294 Y122.295 E.00732
M73 P52 R9
G3 X122.403 Y123.053 I-26.659 J-30.439 E.04468
G1 X119.338 Y125.623 E.15272
G1 X118.983 Y125.859 E.01626
G1 X118.563 Y126.019 E.01717
G1 X118.085 Y126.078 E.01839
G1 X117.571 Y126.009 E.01982
G1 X117.021 Y125.762 E.02302
G1 X116.591 Y125.381 E.02193
G1 X116.042 Y124.726 E.03264
G3 X103.75 Y131.262 I-31.691 J-44.771 E.53292
G1 X104.271 Y131.637 E.02449
G1 X104.543 Y132.051 E.01893
G1 X104.685 Y132.441 E.01585
G3 X104.741 Y133.684 I-6.324 J.91 E.04758
G1 X104.741 Y141.684 E.30543
G1 X104.646 Y142.285 E.02324
G1 X104.557 Y142.443 E.00693
G1 X154.07 Y142.443 E1.89037
G3 X143.803 Y132.701 I31.689 J-43.677 E.54192
G3 X136.271 Y120.545 I42.246 J-34.585 E.54755
G1 X135.514 Y120.694 E.02948
G1 X134.435 Y120.764 E.04125
G3 X133.273 Y120.651 I.635 J-12.581 E.0446
G1 X132.429 Y120.434 E.03328
G1 X131.483 Y120.053 E.03892
G1 X130.604 Y119.536 E.03892
G1 X129.808 Y118.894 E.03905
G1 X129.117 Y118.145 E.03893
G3 X127.853 Y116.453 I65.899 J-50.551 E.08062
G1 X126.662 Y114.847 E.07636
G3 X122.855 Y119.017 I-42.118 J-34.627 E.21568
G1 X123.403 Y119.671 E.03257
G1 X123.604 Y119.961 E.0135
G1 X123.803 Y120.461 E.0205
G1 X123.858 Y120.903 E.01704
G1 X123.791 Y121.432 E.02034
G1 X123.605 Y121.884 E.01865
G1 X123.464 Y122.07 E.00893
G1 X122.978 Y121.747 F36000
G1 F13446.369
G1 X122.914 Y121.845 E.00448
G3 X122.027 Y122.604 I-19.165 J-21.503 E.04459
G1 X118.961 Y125.174 E.15272
G1 X118.62 Y125.383 E.0153
G1 X118.207 Y125.487 E.01626
G3 X117.319 Y125.257 I-.091 J-1.476 E.03558
G1 X116.988 Y124.943 E.01742
G1 X116.144 Y123.937 E.05016
G3 X105.413 Y129.968 I-31.906 J-44.209 E.47099
G1 X104.06 Y130.506 E.05558
G1 X103.688 Y130.653 E.01527
G1 F12219.233
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.6659
G1 F10887.643
G1 X103.235 Y130.854 E.00402
; LINE_WIDTH: 0.711804
G1 F10573.966
G1 X103.153 Y130.908 E.00432
; LINE_WIDTH: 0.757708
G1 F10264.874
G1 X103.072 Y130.962 E.00461
; LINE_WIDTH: 0.803612
G1 F9960.366
G1 X102.99 Y131.016 E.0049
; LINE_WIDTH: 0.849516
G1 F9660.444
G1 X102.909 Y131.07 E.00519
; LINE_WIDTH: 0.89542
G1 F9145.45
G1 X102.827 Y131.124 E.00549
; LINE_WIDTH: 0.941324
G1 F8682.585
G1 X102.746 Y131.178 E.00578
; LINE_WIDTH: 0.987228
G1 F8264.315
G1 X102.664 Y131.232 E.00607
; LINE_WIDTH: 1.03313
G1 F7884.493
G1 X102.583 Y131.286 E.00636
; LINE_WIDTH: 1.07904
G1 F7538.049
G1 X102.501 Y131.34 E.00666
G1 X102.579 Y131.369 E.00569
; LINE_WIDTH: 1.03313
G1 F7884.493
G1 X102.658 Y131.399 E.00544
; LINE_WIDTH: 0.987228
G1 F8264.315
G1 X102.736 Y131.428 E.00519
; LINE_WIDTH: 0.941324
G1 F8682.585
G1 X102.814 Y131.458 E.00494
; LINE_WIDTH: 0.89542
G1 F9145.45
G1 X102.892 Y131.487 E.00469
; LINE_WIDTH: 0.849516
G1 F9660.444
G1 X102.971 Y131.517 E.00444
; LINE_WIDTH: 0.803612
G1 F10236.898
G1 X103.049 Y131.546 E.00419
; LINE_WIDTH: 0.757708
G1 F10500.711
G1 X103.127 Y131.576 E.00394
; LINE_WIDTH: 0.711804
G1 F10767.88
G1 X103.205 Y131.605 E.00369
; LINE_WIDTH: 0.6659
G1 F11038.417
G1 X103.284 Y131.634 E.00344
; LINE_WIDTH: 0.619996
G1 F12378.931
G1 X103.61 Y131.865 E.01527
G1 F13308.091
G1 X103.826 Y132.018 E.0101
G1 F13446.369
G1 X104.017 Y132.308 E.01324
G3 X104.116 Y132.581 I-1.226 J.599 E.01111
G3 X104.155 Y133.684 I-6.084 J.77 E.0422
G1 X104.155 Y141.684 E.30543
G1 X104.089 Y142.105 E.01626
G1 X103.818 Y142.582 E.02094
; LINE_WIDTH: 0.620646
G1 F13431.463
G1 X103.27 Y142.962 E.0255
; LINE_WIDTH: 0.621136
G1 F13420.247
G1 X103.027 Y143.028 E.00963
G1 X104.021 Y143.029 E.03804
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X155.749 Y143.029 E1.97489
G1 X155.769 Y142.918 E.00431
G3 X144.254 Y132.328 I29.963 J-44.135 E.59945
G3 X136.596 Y119.828 I41.151 J-33.809 E.56146
G1 X136.064 Y119.995 E.02127
G1 X135.343 Y120.13 E.02802
G1 X134.414 Y120.179 E.03553
G3 X133.285 Y120.054 I.629 J-10.853 E.04337
G1 X132.568 Y119.865 E.0283
G1 X131.714 Y119.514 E.03526
G1 X130.911 Y119.038 E.03564
G1 X130.185 Y118.446 E.03576
G3 X129.066 Y117.105 I8.093 J-7.891 E.06673
G1 X126.683 Y113.893 E.15272
G3 X122.058 Y118.979 I-42.743 J-34.226 E.26262
G1 X122.954 Y120.047 E.05324
G1 X123.095 Y120.25 E.00945
G1 X123.256 Y120.714 E.01876
G1 X123.26 Y121.11 E.0151
G1 X123.159 Y121.469 E.01423
G1 X123.027 Y121.671 E.00921
G1 X122.475 Y121.441 F36000
G1 F13446.369
G1 X122.409 Y121.52 E.00394
G1 X118.585 Y124.725 E.19047
G1 X118.334 Y124.866 E.01099
G1 X118.031 Y124.905 E.01169
G1 X117.794 Y124.85 E.00927
G1 X117.488 Y124.628 E.01442
G1 X116.241 Y123.14 E.07415
G3 X99.49 Y131.452 I-31.855 J-43.162 E.71759
G3 X99.568 Y132.712 I-24.589 J2.139 E.04821
G1 X102.65 Y132.141 E.11967
G1 X103.086 Y132.186 E.01673
G1 X103.393 Y132.412 E.01456
G1 X103.547 Y132.721 E.01319
G3 X103.57 Y135.684 I-83.361 J2.115 E.11314
G1 X103.57 Y141.684 E.22907
G1 X103.532 Y141.924 E.00928
G1 X103.377 Y142.196 E.01195
G1 X103.064 Y142.413 E.01454
G1 X102.791 Y142.463 E.0106
G1 X102.65 Y142.45 E.00544
G1 X99.57 Y141.879 E.11958
G3 X99.514 Y143.615 I-14.43 J.408 E.06633
G1 X156.235 Y143.615 E2.16556
G1 X156.415 Y142.65 E.03745
G3 X144.705 Y131.954 I29.416 J-43.961 E.60778
G3 X136.984 Y119.25 I41.208 J-33.744 E.56947
G1 X136.912 Y119.083 E.00693
G1 X136.093 Y119.382 E.03328
G1 X135.299 Y119.542 E.03091
G1 X134.392 Y119.594 E.03469
G3 X132.715 Y119.298 I.447 J-7.445 E.06514
G1 X131.944 Y118.976 E.0319
G1 X131.218 Y118.539 E.03235
G1 X130.562 Y117.998 E.03247
G1 X129.994 Y117.369 E.03235
G3 X129.081 Y116.142 I189.97 J-142.427 E.05839
G1 X126.698 Y112.93 E.15272
G3 X121.255 Y118.932 I-42.298 J-32.885 E.30965
G1 X122.505 Y120.423 E.07429
G1 X122.665 Y120.739 E.01349
G1 X122.68 Y121.03 E.01114
G1 X122.586 Y121.307 E.01116
G1 X122.532 Y121.372 E.00321
M204 S250
G1 X122.053 Y121.096 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.23 Y124.301 E.15795
G1 X118.069 Y124.353 E.00535
G1 X117.912 Y124.273 E.00559
G1 X116.592 Y122.698 E.06508
; LINE_WIDTH: 0.521566
G1 X116.518 Y122.612 E.00359
; LINE_WIDTH: 0.549316
G1 X116.504 Y122.291 E.01081
G1 X116 Y122.631 E.02039
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.538 J-42.493 E.60555
G1 X98.939 Y131.545 E.01605
G3 X99.016 Y133.376 I-12.559 J1.445 E.05809
G1 X102.75 Y132.685 E.12023
G1 X102.877 Y132.698 E.00402
G1 X102.994 Y132.808 E.00509
G3 X103.017 Y135.684 I-74.089 J2.028 E.09108
G1 X103.017 Y141.684 E.18996
G1 X102.961 Y141.833 E.00502
G1 X102.791 Y141.91 E.00591
G1 X102.75 Y141.906 E.00131
G1 X99.017 Y141.215 E.1202
G1 X99.017 Y142.631 E.04484
G1 X98.937 Y143.857 E.0389
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.959 Y142.745 E.04581
G1 X157.025 Y142.391 E.0114
G3 X145.128 Y131.598 I28.628 J-43.511 E.51062
G3 X137.192 Y118.315 I40.246 J-33.057 E.49168
G1 X136.612 Y118.616 E.02068
G1 X135.913 Y118.859 E.02344
G1 X135.198 Y118.999 E.02306
G1 X134.371 Y119.041 E.0262
G1 X133.607 Y118.959 E.02432
G1 X132.868 Y118.767 E.0242
G1 X132.162 Y118.468 E.02427
G1 X131.508 Y118.068 E.02426
G1 X130.918 Y117.575 E.02435
G1 X130.408 Y117.003 E.02425
G3 X129.087 Y115.222 I902.157 J-671.132 E.07019
G1 X126.704 Y112.01 E.12664
G3 X120.714 Y118.682 I-42.039 J-31.715 E.28421
; LINE_WIDTH: 0.546756
G1 X120.351 Y119.065 E.01764
G1 X120.688 Y119.116 E.01141
; LINE_WIDTH: 0.519996
G1 X122.081 Y120.778 E.06866
G1 X122.134 Y120.938 E.00532
G1 X122.094 Y121.016 E.00276
; WIPE_START
M204 S10000
G1 X121.332 Y121.664 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.504 Y122.291 Z2.84 F36000
G1 Z2.44
G1 E.4 F1800
; LINE_WIDTH: 0.549316
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.00792
; LINE_WIDTH: 0.544336
G1 X120.198 Y119.2 E.15231
; LINE_WIDTH: 0.546756
G1 X120.351 Y119.065 E.00679
; WIPE_START
M204 S10000
G1 X120.198 Y119.2 E-.07729
G1 X119.588 Y119.712 E-.30271
; WIPE_END
G1 E-.02 F1800
G1 X113.129 Y123.779 Z2.84 F36000
G1 X100.18 Y131.933 Z2.84
G1 Z2.44
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.798036
G1 F10311.641
G1 X100.499 Y131.853 E.01636
; LINE_WIDTH: 0.837716
G1 F9802.336
G1 X100.818 Y131.774 E.01721
; LINE_WIDTH: 0.877396
G1 F9340.973
G1 X101.136 Y131.695 E.01806
; LINE_WIDTH: 0.881636
G1 F9294.229
G1 X101.168 Y131.687 E.00178
; LINE_WIDTH: 0.927781
G1 F8814.197
G1 X101.482 Y131.605 E.01889
; LINE_WIDTH: 0.973926
G1 F8381.316
G1 X101.795 Y131.524 E.01987
; LINE_WIDTH: 1.02007
G1 F7988.963
G1 X102.109 Y131.442 E.02084
; LINE_WIDTH: 1.06622
G1 F7631.701
G1 X102.423 Y131.361 E.02182
; LINE_WIDTH: 1.07904
G1 F7538.049
G1 X102.501 Y131.34 E.00549
; WIPE_START
G1 X102.423 Y131.361 E-.03062
G1 X102.109 Y131.442 E-.12325
G1 X101.795 Y131.524 E-.12325
G1 X101.533 Y131.592 E-.10288
; WIPE_END
G1 E-.02 F1800
G1 X100.736 Y139.183 Z2.84 F36000
G1 X100.354 Y142.826 Z2.84
G1 Z2.44
G1 E.4 F1800
; LINE_WIDTH: 1.02542
G1 F7945.878
G1 X100.551 Y142.844 E.01281
; LINE_WIDTH: 0.989156
G1 F8247.628
G1 X100.8 Y142.867 E.01556
; LINE_WIDTH: 0.943441
G1 F8662.366
G1 X101.049 Y142.89 E.01482
; LINE_WIDTH: 0.897726
G1 F9121.023
G1 X101.298 Y142.913 E.01407
; LINE_WIDTH: 0.852011
G1 F9630.966
G1 X101.547 Y142.936 E.01333
; LINE_WIDTH: 0.806296
G1 F10201.307
G1 X101.796 Y142.959 E.01258
; LINE_WIDTH: 0.760581
G1 F10843.449
G1 X102.045 Y142.981 E.01184
; LINE_WIDTH: 0.714866
G1 F11571.863
G1 X102.294 Y143.004 E.01109
; LINE_WIDTH: 0.669151
G1 F12405.19
G1 X102.543 Y143.027 E.01035
; LINE_WIDTH: 0.623436
G1 F13367.85
G1 X103.027 Y143.028 E.01861
; WIPE_START
G1 X102.543 Y143.027 E-.18412
G1 X102.294 Y143.004 E-.095
G1 X102.045 Y142.981 E-.095
G1 X102.029 Y142.98 E-.00588
; WIPE_END
G1 E-.02 F1800
G1 X104.832 Y135.881 Z2.84 F36000
G1 X105.239 Y134.85 Z2.84
G1 Z2.44
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.239 Y137.193 E.08944
G3 X107.257 Y138.472 I-2.544 J6.248 E.09171
G3 X109.142 Y140.915 I-15.798 J14.138 E.11793
G2 X110.556 Y141.771 I1.9 J-1.542 E.06437
G2 X114.798 Y140.154 I.25 J-5.715 E.17821
G2 X116.683 Y137.711 I-15.798 J-14.138 E.11793
G3 X118.097 Y136.855 I1.9 J1.542 E.06437
G3 X122.339 Y138.472 I.25 J5.715 E.17821
G3 X124.224 Y140.915 I-15.799 J14.139 E.11793
G2 X125.638 Y141.771 I1.9 J-1.542 E.06437
G2 X129.879 Y140.154 I.25 J-5.715 E.17821
G2 X131.764 Y137.711 I-15.8 J-14.14 E.11793
G3 X133.178 Y136.855 I1.9 J1.542 E.06437
G3 X137.42 Y138.472 I.25 J5.715 E.17821
G3 X139.305 Y140.915 I-15.8 J14.14 E.11793
G2 X140.719 Y141.771 I1.9 J-1.542 E.06437
G2 X144.961 Y140.154 I.25 J-5.715 E.17821
G2 X146.846 Y137.711 I-15.8 J-14.14 E.11793
G1 X147.273 Y137.295 E.02277
G3 X144.397 Y134.191 I63.125 J-61.376 E.16158
G3 X141.19 Y132.614 I.709 J-5.489 E.13892
G3 X139.305 Y130.171 I15.798 J-14.138 E.11793
G2 X137.891 Y129.314 I-1.9 J1.542 E.06437
G2 X133.65 Y130.931 I-.25 J5.715 E.17821
G2 X131.764 Y133.374 I15.796 J14.137 E.11793
G3 X130.351 Y134.231 I-1.9 J-1.542 E.06437
G3 X126.109 Y132.614 I-.25 J-5.715 E.17821
G3 X124.224 Y130.171 I15.8 J-14.14 E.11793
G2 X122.81 Y129.314 I-1.9 J1.542 E.06437
G2 X118.568 Y130.931 I-.25 J5.715 E.17821
G2 X116.683 Y133.374 I15.799 J14.139 E.11793
G3 X115.269 Y134.231 I-1.9 J-1.542 E.06437
G3 X111.028 Y132.614 I-.25 J-5.715 E.17821
G3 X109.142 Y130.171 I15.8 J-14.14 E.11793
G2 X108.641 Y129.694 I-1.18 J.74 E.02669
G2 X110.722 Y128.619 I-20.618 J-42.474 E.08944
; WIPE_START
G1 X109.834 Y129.078 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.914 Y126.228 Z2.84 F36000
G1 X131.212 Y120.472 Z2.84
G1 Z2.44
G1 E.4 F1800
G1 F13446.283
G3 X129.317 Y119.11 I3.17 J-6.41 E.08948
G3 X125.555 Y116.902 I.783 J-5.644 E.17092
G3 X123.53 Y119.047 I-44.473 J-39.957 E.11265
G3 X124.153 Y121.9 I-1.868 J1.903 E.11798
G3 X122.526 Y123.598 I-4.388 J-2.577 E.0906
G3 X124.224 Y125.834 I-17.093 J14.741 E.10726
G2 X125.638 Y126.69 I1.9 J-1.542 E.06437
G2 X129.879 Y125.073 I.25 J-5.715 E.17821
G2 X131.764 Y122.63 I-15.799 J-14.139 E.11793
G3 X133.178 Y121.774 I1.9 J1.542 E.06437
G3 X136.864 Y122.944 I.274 J5.526 E.15082
G2 X137.964 Y125.012 I31.751 J-15.556 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 2.6
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.494 Y124.129 E-.38
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
G1 X123.342 Y122.23
G1 Z2.6
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.084 Y122.482 E.01378
G3 X122.366 Y123.084 I-6.671 J-7.229 E.0358
G1 X119.3 Y125.654 E.15272
G1 X118.945 Y125.891 E.01629
G1 X118.524 Y126.051 E.0172
G1 X118.048 Y126.11 E.01832
G1 X117.517 Y126.036 E.02045
G1 X116.961 Y125.779 E.0234
G1 X116.553 Y125.412 E.02092
G1 X116.001 Y124.754 E.0328
G3 X103.777 Y131.251 I-31.595 J-44.702 E.52997
G1 X104.078 Y131.437 E.01351
G1 X104.428 Y131.841 E.02041
G1 X104.687 Y132.439 E.0249
G3 X104.743 Y133.687 I-6.354 J.911 E.04774
G1 X104.743 Y141.687 E.30543
G1 X104.648 Y142.288 E.02324
G1 X104.56 Y142.443 E.00682
G1 X154.07 Y142.443 E1.89023
G3 X143.803 Y132.701 I31.798 J-43.791 E.5419
G3 X136.271 Y120.545 I41.777 J-34.297 E.54759
G1 X135.45 Y120.705 E.03191
G3 X133.29 Y120.654 I-.889 J-8.143 E.08273
G1 X132.429 Y120.434 E.03394
G1 X131.481 Y120.051 E.03904
G1 X130.602 Y119.534 E.03894
G1 X129.808 Y118.894 E.03893
G1 X129.117 Y118.145 E.03892
G3 X127.853 Y116.453 I69.757 J-53.433 E.08062
G1 X126.662 Y114.847 E.07636
G3 X122.889 Y118.981 I-41.221 J-33.829 E.2138
G1 X123.44 Y119.639 E.03279
G1 X123.709 Y120.06 E.01907
G1 X123.873 Y120.598 E.02148
G1 X123.877 Y121.165 E.02162
G1 X123.731 Y121.677 E.02032
G1 X123.479 Y122.096 E.01868
G1 X123.407 Y122.167 E.00389
G1 X122.945 Y121.801 F36000
G1 F13446.369
G1 X122.822 Y121.938 E.007
G1 X118.924 Y125.205 E.19421
G1 X118.582 Y125.415 E.01533
G1 X118.168 Y125.519 E.01629
G3 X117.266 Y125.278 I-.089 J-1.475 E.03623
G1 X116.917 Y124.934 E.01873
G1 X116.104 Y123.965 E.04828
G3 X104.555 Y130.321 I-31.846 J-44.197 E.50456
G1 X104.067 Y130.511 E.01999
G1 X103.693 Y130.655 E.01527
G1 F12232.404
G1 X103.32 Y130.799 E.01527
; LINE_WIDTH: 0.665732
G1 F10900.075
G1 X103.239 Y130.853 E.00403
; LINE_WIDTH: 0.711468
G1 F10585.774
G1 X103.157 Y130.907 E.00432
; LINE_WIDTH: 0.757204
G1 F10276.054
G1 X103.075 Y130.961 E.00461
; LINE_WIDTH: 0.80294
G1 F9970.932
G1 X102.994 Y131.015 E.0049
; LINE_WIDTH: 0.848676
G1 F9670.408
G1 X102.912 Y131.068 E.0052
; LINE_WIDTH: 0.894412
G1 F9156.168
G1 X102.83 Y131.122 E.00549
; LINE_WIDTH: 0.940148
G1 F8693.857
G1 X102.748 Y131.176 E.00578
; LINE_WIDTH: 0.985884
G1 F8275.989
G1 X102.667 Y131.23 E.00607
; LINE_WIDTH: 1.03162
G1 F7896.447
G1 X102.585 Y131.284 E.00636
; LINE_WIDTH: 1.07736
G1 F7550.191
G1 X102.503 Y131.338 E.00666
G1 X102.581 Y131.367 E.00567
; LINE_WIDTH: 1.03162
G1 F7896.447
G1 X102.66 Y131.397 E.00542
; LINE_WIDTH: 0.985884
G1 F8275.989
G1 X102.738 Y131.426 E.00518
; LINE_WIDTH: 0.940148
G1 F8693.857
G1 X102.816 Y131.455 E.00493
; LINE_WIDTH: 0.894412
G1 F9156.168
G1 X102.894 Y131.485 E.00468
; LINE_WIDTH: 0.848676
G1 F9670.408
G1 X102.972 Y131.514 E.00443
; LINE_WIDTH: 0.80294
G1 F10245.849
G1 X103.05 Y131.543 E.00418
; LINE_WIDTH: 0.757204
G1 F10509.144
G1 X103.128 Y131.573 E.00393
; LINE_WIDTH: 0.711468
G1 F10775.81
G1 X103.206 Y131.602 E.00368
; LINE_WIDTH: 0.665732
G1 F11045.797
G1 X103.284 Y131.631 E.00343
; LINE_WIDTH: 0.619996
G1 F12650.628
G1 X103.692 Y131.878 E.01818
G1 F13446.369
G1 X103.953 Y132.185 E.01539
G1 X104.124 Y132.603 E.01726
G3 X104.158 Y133.687 I-6.618 J.747 E.04142
G1 X104.158 Y141.687 E.30543
G1 X104.091 Y142.107 E.01626
G1 X103.821 Y142.584 E.02094
G1 X103.272 Y142.964 E.02547
; LINE_WIDTH: 0.619206
G1 F13464.532
G1 X103.029 Y143.03 E.00959
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.868 J1974.114 E.10383
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I29.803 J-43.952 E.59952
G3 X136.596 Y119.828 I41.877 J-34.253 E.56141
G1 X136.064 Y119.995 E.02128
G1 X135.343 Y120.13 E.02799
G1 X134.414 Y120.179 E.03554
G3 X133.287 Y120.054 I.626 J-10.833 E.04328
G1 X132.568 Y119.865 E.0284
G1 X131.711 Y119.513 E.03536
G1 X130.909 Y119.036 E.03565
G1 X130.185 Y118.446 E.03564
G3 X129.066 Y117.106 I8.24 J-8.013 E.06673
G1 X126.683 Y113.893 E.15272
G3 X122.093 Y118.944 I-45.274 J-36.533 E.26073
G1 X122.991 Y120.016 E.0534
G1 X123.238 Y120.454 E.01921
G1 X123.31 Y120.882 E.01658
G1 X123.262 Y121.252 E.01422
G1 X123.073 Y121.66 E.01717
G1 X123.006 Y121.735 E.00384
G1 X122.512 Y121.41 F36000
G1 F13446.369
G1 X122.446 Y121.489 E.00391
G1 X118.548 Y124.756 E.19421
G1 X118.296 Y124.897 E.01101
G1 X117.992 Y124.936 E.01171
G1 X117.749 Y124.879 E.00952
G1 X117.451 Y124.66 E.01413
G1 X116.201 Y123.169 E.07428
G3 X99.49 Y131.452 I-31.769 J-43.096 E.71573
G3 X99.57 Y132.709 I-23.926 J2.139 E.04812
G1 X102.652 Y132.139 E.11967
G1 X103.089 Y132.184 E.01679
G3 X103.553 Y132.733 I-.359 J.774 E.0283
G3 X103.572 Y135.687 I-96.485 J2.103 E.11279
G1 X103.572 Y141.687 E.22907
G1 X103.534 Y141.927 E.00928
G1 X103.38 Y142.199 E.01195
G1 X103.066 Y142.416 E.01454
G1 X102.793 Y142.465 E.0106
G1 X102.652 Y142.452 E.00544
G1 X99.572 Y141.882 E.11958
G3 X99.515 Y143.615 I-14.061 J.407 E.06623
G1 X156.235 Y143.615 E2.16553
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.705 Y131.954 I29.325 J-43.872 E.60773
G3 X136.912 Y119.083 I41.409 J-33.869 E.57638
G1 X136.092 Y119.382 E.03329
G3 X134.382 Y119.594 I-1.771 J-7.279 E.06596
G3 X133.436 Y119.488 I1.145 J-14.508 E.03632
G1 X132.721 Y119.3 E.02823
G1 X131.942 Y118.975 E.03224
G1 X131.216 Y118.537 E.03236
G1 X130.562 Y117.998 E.03236
G1 X129.994 Y117.369 E.03235
G3 X129.081 Y116.142 I207.416 J-155.421 E.05839
G1 X126.698 Y112.93 E.15272
G3 X121.291 Y118.899 I-44.055 J-34.474 E.30776
G1 X122.543 Y120.392 E.07439
G1 X122.703 Y120.709 E.01357
G1 X122.717 Y121.001 E.01115
G1 X122.622 Y121.278 E.01117
G1 X122.569 Y121.341 E.00315
M204 S250
G1 X122.091 Y121.065 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.193 Y124.333 E.16105
G1 X118.031 Y124.385 E.00536
G3 X117.806 Y124.223 I.079 J-.347 E.00904
G1 X116.521 Y122.69 E.06332
; LINE_WIDTH: 0.521546
G1 X116.48 Y122.644 E.00196
; LINE_WIDTH: 0.551836
G1 X116.465 Y122.321 E.01089
G1 X116 Y122.631 E.01884
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.538 J-42.492 E.60555
G1 X98.939 Y131.544 E.01604
G3 X99.018 Y133.374 I-12.229 J1.444 E.05803
G1 X102.752 Y132.682 E.12023
G1 X102.942 Y132.734 E.00623
G1 X103.019 Y132.904 E.0059
M73 P53 R9
G1 X103.019 Y141.687 E.27805
G1 X102.963 Y141.835 E.00502
G1 X102.793 Y141.912 E.00591
G1 X102.752 Y141.909 E.00131
G1 X99.019 Y141.217 E.1202
G1 X99.019 Y142.631 E.04476
G1 X98.937 Y143.857 E.0389
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.615 J-43.5 E.51061
G3 X137.192 Y118.315 I40.31 J-33.095 E.49168
G1 X136.614 Y118.615 E.0206
G1 X135.912 Y118.859 E.02353
G1 X135.135 Y119.01 E.02508
G1 X134.371 Y119.041 E.02419
G1 X133.61 Y118.959 E.02426
G1 X132.868 Y118.767 E.02427
G1 X132.16 Y118.466 E.02434
G1 X131.506 Y118.066 E.02426
G1 X130.918 Y117.575 E.02426
G1 X130.408 Y117.003 E.02426
G3 X129.087 Y115.222 I986.892 J-734.033 E.07019
G1 X126.704 Y112.01 E.12664
G3 X120.752 Y118.646 I-41.962 J-31.646 E.28258
; LINE_WIDTH: 0.549476
G1 X120.387 Y119.033 E.01783
G1 X120.726 Y119.085 E.0115
; LINE_WIDTH: 0.519996
G1 X122.119 Y120.747 E.06866
G1 X122.171 Y120.908 E.00535
G1 X122.132 Y120.985 E.00274
; WIPE_START
M204 S10000
G1 X121.37 Y121.632 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.465 Y122.321 Z3 F36000
G1 Z2.6
G1 E.4 F1800
; LINE_WIDTH: 0.551836
G1 F3600
M204 S5000
G1 X116.687 Y122.143 E.00961
; LINE_WIDTH: 0.544336
G1 X120.199 Y119.2 E.15232
; LINE_WIDTH: 0.549476
G1 X120.387 Y119.033 E.00847
; WIPE_START
M204 S10000
G1 X120.199 Y119.2 E-.09579
G1 X119.625 Y119.68 E-.28421
; WIPE_END
G1 E-.02 F1800
G1 X113.168 Y123.749 Z3 F36000
G1 X100.18 Y131.932 Z3
G1 Z2.6
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.795963
G1 F10339.713
G1 X100.499 Y131.852 E.01631
; LINE_WIDTH: 0.83563
G1 F9827.862
G1 X100.817 Y131.773 E.01716
; LINE_WIDTH: 0.875296
G1 F9364.299
G1 X101.136 Y131.694 E.01801
; LINE_WIDTH: 0.879536
G1 F9317.322
G1 X101.167 Y131.686 E.00177
; LINE_WIDTH: 0.925686
G1 F8834.914
G1 X101.481 Y131.604 E.01885
; LINE_WIDTH: 0.971836
G1 F8400.001
G1 X101.795 Y131.523 E.01983
; LINE_WIDTH: 1.01799
G1 F8005.897
G1 X102.109 Y131.441 E.0208
; LINE_WIDTH: 1.06414
G1 F7647.116
G1 X102.423 Y131.36 E.02178
; LINE_WIDTH: 1.07736
G1 F7550.191
G1 X102.503 Y131.338 E.00565
; WIPE_START
G1 X102.423 Y131.36 E-.03155
G1 X102.109 Y131.441 E-.12327
G1 X101.795 Y131.523 E-.12327
G1 X101.536 Y131.59 E-.1019
; WIPE_END
G1 E-.02 F1800
G1 X100.738 Y139.181 Z3 F36000
G1 X100.354 Y142.827 Z3
G1 Z2.6
G1 E.4 F1800
; LINE_WIDTH: 1.02318
G1 F7963.877
G1 X100.553 Y142.846 E.01288
; LINE_WIDTH: 0.986646
G1 F8269.367
G1 X100.802 Y142.868 E.01552
; LINE_WIDTH: 0.94093
G1 F8686.361
G1 X101.051 Y142.891 E.01478
; LINE_WIDTH: 0.895214
G1 F9147.644
G1 X101.3 Y142.914 E.01403
; LINE_WIDTH: 0.849497
G1 F9660.666
G1 X101.549 Y142.937 E.01328
; LINE_WIDTH: 0.803781
G1 F10234.65
G1 X101.798 Y142.96 E.01254
; LINE_WIDTH: 0.758065
G1 F10881.15
G1 X102.047 Y142.983 E.01179
; LINE_WIDTH: 0.712349
G1 F11614.831
G1 X102.296 Y143.006 E.01105
; LINE_WIDTH: 0.666632
G1 F12454.606
G1 X102.545 Y143.028 E.0103
; LINE_WIDTH: 0.620916
G1 F13425.28
G1 X103.029 Y143.03 E.01851
; WIPE_START
G1 X102.545 Y143.028 E-.18395
G1 X102.296 Y143.006 E-.095
G1 X102.047 Y142.983 E-.095
G1 X102.031 Y142.981 E-.00605
; WIPE_END
G1 E-.02 F1800
G1 X104.85 Y135.888 Z3 F36000
G1 X105.241 Y134.903 Z3
G1 Z2.6
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.241 Y137.246 E.08944
G3 X107.257 Y138.406 I-2.205 J6.164 E.08929
G3 X109.142 Y140.655 I-13.373 J13.126 E.11214
G2 X110.556 Y141.579 I2.247 J-1.895 E.06541
G2 X114.798 Y140.22 I.634 J-5.322 E.17536
G2 X116.683 Y137.972 I-13.373 J-13.126 E.11214
G3 X118.097 Y137.048 I2.247 J1.895 E.06541
G3 X122.339 Y138.406 I.634 J5.322 E.17536
G3 X124.224 Y140.655 I-13.373 J13.127 E.11214
G2 X125.638 Y141.579 I2.247 J-1.895 E.06541
G2 X129.879 Y140.22 I.634 J-5.322 E.17536
G2 X131.764 Y137.972 I-13.375 J-13.128 E.11214
G3 X133.178 Y137.048 I2.247 J1.895 E.06541
G3 X137.42 Y138.406 I.634 J5.322 E.17536
G3 X139.305 Y140.655 I-13.375 J13.128 E.11214
G2 X140.719 Y141.579 I2.247 J-1.895 E.06541
G2 X144.961 Y140.22 I.634 J-5.322 E.17536
G2 X146.846 Y137.972 I-13.375 J-13.128 E.11214
G3 X147.422 Y137.446 I1.552 J1.123 E.02998
G3 X144.288 Y134.065 I84.047 J-81.036 E.176
G3 X141.19 Y132.679 I.513 J-5.303 E.13188
G3 X139.305 Y130.431 I13.375 J-13.127 E.11214
G2 X137.891 Y129.507 I-2.247 J1.895 E.06541
G2 X133.65 Y130.865 I-.634 J5.322 E.17536
G2 X131.764 Y133.114 I13.372 J13.125 E.11214
G3 X130.351 Y134.038 I-2.247 J-1.895 E.06541
G3 X126.109 Y132.679 I-.634 J-5.322 E.17536
G3 X124.224 Y130.431 I13.377 J-13.129 E.11214
G2 X122.81 Y129.507 I-2.247 J1.895 E.06541
G2 X118.568 Y130.865 I-.634 J5.322 E.17536
G2 X116.683 Y133.114 I13.374 J13.127 E.11214
G3 X115.269 Y134.038 I-2.247 J-1.895 E.06541
G3 X111.028 Y132.679 I-.634 J-5.322 E.17536
G3 X109.142 Y130.431 I13.377 J-13.129 E.11214
G2 X108.406 Y129.803 I-1.878 J1.458 E.03721
G2 X110.492 Y128.738 I-15.031 J-31.994 E.08945
; WIPE_START
G1 X109.601 Y129.193 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.663 Y126.295 Z3 F36000
G1 X131.055 Y120.39 Z3
G1 Z2.6
G1 E.4 F1800
G1 F13446.283
G3 X129.234 Y119.029 I3.461 J-6.527 E.08716
G3 X125.504 Y116.959 I.89 J-6.001 E.16643
G3 X123.563 Y119.011 I-39.804 J-35.713 E.10784
G3 X124.188 Y121.875 I-1.873 J1.909 E.11846
G3 X122.571 Y123.56 I-4.571 J-2.767 E.08989
G3 X124.224 Y125.573 I-14.827 J13.859 E.09952
G2 X125.638 Y126.497 I2.247 J-1.895 E.06541
G2 X129.879 Y125.139 I.634 J-5.322 E.17536
G2 X131.764 Y122.89 I-13.375 J-13.128 E.11214
G3 X133.178 Y121.966 I2.247 J1.895 E.06541
G3 X136.84 Y122.898 I.645 J5.126 E.14769
G2 X137.937 Y124.968 I34.163 J-16.782 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 2.76
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.469 Y124.084 E-.38
; WIPE_END
M73 P53 R8
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
G1 X123.473 Y122.091
G1 Z2.76
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.355 Y122.245 E.00739
G3 X122.336 Y123.109 I-41.695 J-48.133 E.05101
G1 X119.271 Y125.679 E.15272
G1 X118.915 Y125.916 E.01632
G1 X118.493 Y126.075 E.01723
G1 X118.018 Y126.134 E.01826
G1 X117.475 Y126.057 E.02095
G1 X116.914 Y125.791 E.02371
G1 X116.524 Y125.437 E.02011
G1 X115.97 Y124.776 E.03292
G3 X103.757 Y131.259 I-31.93 J-45.407 E.52928
G1 X104.273 Y131.629 E.02425
G1 X104.545 Y132.041 E.01884
G1 X104.687 Y132.43 E.0158
G3 X104.745 Y133.689 I-6.29 J.92 E.0482
G1 X104.745 Y141.689 E.30543
G1 X104.65 Y142.29 E.02324
G1 X104.564 Y142.443 E.00671
G1 X154.071 Y142.443 E1.89014
G3 X143.803 Y132.701 I31.919 J-43.924 E.54193
G3 X136.318 Y120.647 I41.743 J-34.272 E.54331
G1 X136.271 Y120.545 E.00427
G3 X135.168 Y120.737 I-2.138 J-9.027 E.04278
G3 X133.291 Y120.655 I-.568 J-8.493 E.07187
G1 X132.427 Y120.433 E.03408
G1 X131.48 Y120.051 E.03898
G1 X130.602 Y119.534 E.0389
G1 X129.808 Y118.894 E.03895
G1 X129.117 Y118.145 E.03889
G3 X127.853 Y116.453 I73.155 J-55.972 E.08064
G1 X126.662 Y114.847 E.07636
G3 X122.917 Y118.956 I-42.234 J-34.729 E.21234
G1 X123.47 Y119.615 E.03284
G1 X123.643 Y119.857 E.01139
G1 X123.872 Y120.414 E.02295
G1 X123.925 Y120.859 E.01712
G1 X123.855 Y121.387 E.02035
G1 X123.666 Y121.838 E.01867
G1 X123.528 Y122.02 E.00871
G1 X122.974 Y121.778 F36000
G1 F13446.369
G1 X122.852 Y121.913 E.00695
G1 X118.895 Y125.23 E.19713
G1 X118.552 Y125.44 E.01535
G1 X118.137 Y125.543 E.01632
G3 X117.225 Y125.294 I-.087 J-1.474 E.03674
G1 X116.861 Y124.927 E.01975
G1 X116.073 Y123.987 E.04682
G3 X105.403 Y129.969 I-32.108 J-44.765 E.46798
G1 X104.06 Y130.506 E.05523
G1 X103.688 Y130.653 E.01527
G1 F12233.846
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.665566
G1 F10901.437
G1 X103.235 Y130.854 E.004
; LINE_WIDTH: 0.711136
G1 F10589.354
G1 X103.154 Y130.908 E.00429
; LINE_WIDTH: 0.756706
G1 F10281.804
G1 X103.073 Y130.961 E.00458
; LINE_WIDTH: 0.802276
G1 F9978.76
G1 X102.992 Y131.015 E.00486
; LINE_WIDTH: 0.847846
G1 F9680.274
G1 X102.911 Y131.068 E.00515
; LINE_WIDTH: 0.893416
G1 F9166.784
G1 X102.83 Y131.122 E.00544
; LINE_WIDTH: 0.938986
G1 F8705.024
G1 X102.749 Y131.176 E.00573
; LINE_WIDTH: 0.984556
G1 F8287.555
G1 X102.668 Y131.229 E.00602
; LINE_WIDTH: 1.03013
G1 F7908.294
G1 X102.587 Y131.283 E.00631
; LINE_WIDTH: 1.0757
G1 F7562.226
G1 X102.506 Y131.336 E.0066
G1 X102.584 Y131.365 E.00565
; LINE_WIDTH: 1.03013
G1 F7908.294
G1 X102.661 Y131.395 E.0054
; LINE_WIDTH: 0.984556
G1 F8287.555
G1 X102.739 Y131.424 E.00515
; LINE_WIDTH: 0.938986
G1 F8705.024
G1 X102.817 Y131.453 E.00491
; LINE_WIDTH: 0.893416
G1 F9166.784
G1 X102.895 Y131.482 E.00466
; LINE_WIDTH: 0.847846
G1 F9680.274
G1 X102.973 Y131.511 E.00441
; LINE_WIDTH: 0.802276
G1 F10254.708
G1 X103.051 Y131.541 E.00417
; LINE_WIDTH: 0.756706
G1 F10517.424
G1 X103.129 Y131.57 E.00392
; LINE_WIDTH: 0.711136
G1 F10783.464
G1 X103.207 Y131.599 E.00367
; LINE_WIDTH: 0.665566
G1 F11052.826
G1 X103.285 Y131.628 E.00342
; LINE_WIDTH: 0.619996
G1 F12394.19
G1 X103.612 Y131.859 E.01527
G1 F13328.487
G1 X103.829 Y132.012 E.01015
G1 F13446.369
G1 X104.02 Y132.3 E.01318
G3 X104.119 Y132.572 I-1.224 J.602 E.01108
G3 X104.16 Y133.689 I-6.043 J.779 E.04275
G1 X104.16 Y141.689 E.30543
G1 X104.093 Y142.11 E.01626
G1 X103.823 Y142.587 E.02094
G1 X103.274 Y142.966 E.02546
; LINE_WIDTH: 0.617756
G1 F13497.996
G1 X103.031 Y143.031 E.00956
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.859 J690.085 E.10376
G1 X155.749 Y143.029 E1.90895
G1 X155.769 Y142.919 E.00427
G3 X144.254 Y132.328 I29.995 J-44.166 E.59947
G3 X136.596 Y119.828 I41.525 J-34.038 E.56143
G1 X136.065 Y119.995 E.02126
G1 X135.345 Y120.129 E.02793
G1 X134.414 Y120.179 E.03562
G3 X133.287 Y120.054 I.624 J-10.815 E.0433
G1 X132.566 Y119.865 E.02845
G1 X131.711 Y119.513 E.03532
G1 X130.909 Y119.036 E.03562
G1 X130.185 Y118.446 E.03566
G3 X129.066 Y117.106 I8.362 J-8.115 E.06671
G1 X126.683 Y113.893 E.15272
G3 X122.121 Y118.917 I-42.677 J-34.172 E.25928
G1 X123.021 Y119.991 E.0535
G1 X123.265 Y120.424 E.019
G1 X123.339 Y120.861 E.01692
G1 X123.29 Y121.231 E.01424
G1 X123.1 Y121.639 E.01717
G1 X123.035 Y121.711 E.00375
G1 X122.541 Y121.386 F36000
G1 F13446.369
G1 X122.475 Y121.464 E.00389
G1 X118.518 Y124.781 E.19713
G1 X118.266 Y124.922 E.01103
G1 X117.962 Y124.961 E.01173
G1 X117.714 Y124.901 E.00972
G1 X117.422 Y124.684 E.01389
G1 X116.17 Y123.192 E.07437
G3 X99.49 Y131.452 I-31.782 J-43.206 E.71426
G3 X99.572 Y132.707 I-15.489 J1.633 E.04803
G1 X102.654 Y132.136 E.11968
G1 X103.088 Y132.18 E.01667
G1 X103.396 Y132.406 E.01458
G1 X103.551 Y132.713 E.01314
G3 X103.574 Y135.689 I-81.566 J2.122 E.11362
G1 X103.574 Y141.689 E.22907
G1 X103.536 Y141.929 E.00928
G1 X103.382 Y142.201 E.01195
G1 X103.069 Y142.418 E.01454
G1 X102.795 Y142.468 E.0106
G1 X102.654 Y142.455 E.00544
G1 X99.574 Y141.884 E.11958
G3 X99.516 Y143.615 I-13.957 J.398 E.06614
G1 X156.235 Y143.615 E2.1655
G1 X156.415 Y142.65 E.03745
G3 X144.705 Y131.954 I29.415 J-43.96 E.60778
G3 X136.912 Y119.083 I41.087 J-33.674 E.57641
G1 X136.092 Y119.382 E.0333
G1 X135.372 Y119.529 E.02805
G1 X134.392 Y119.594 E.0375
G3 X132.719 Y119.299 I.435 J-7.373 E.06498
G1 X131.941 Y118.974 E.03219
G1 X131.216 Y118.537 E.03233
G1 X130.562 Y117.998 E.03238
G3 X129.081 Y116.142 I11.896 J-11.013 E.09071
G1 X126.698 Y112.93 E.15272
G3 X121.319 Y118.872 I-42.376 J-32.951 E.30631
G1 X122.572 Y120.367 E.07448
G1 X122.732 Y120.686 E.01363
G1 X122.746 Y120.978 E.01116
G1 X122.65 Y121.255 E.01118
G1 X122.599 Y121.317 E.00308
M204 S250
G1 X122.12 Y121.04 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.163 Y124.357 E.16347
G1 X118.002 Y124.409 E.00537
G3 X117.751 Y124.216 I.103 J-.393 E.0103
G1 X116.466 Y122.684 E.06332
; LINE_WIDTH: 0.553776
G1 X116.435 Y122.345 E.01151
G1 X116 Y122.631 E.01761
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.54 J-42.496 E.60555
G1 X98.941 Y131.548 E.01615
G3 X99.02 Y133.371 I-11.968 J1.438 E.05784
G1 X102.754 Y132.68 E.12023
G1 X102.88 Y132.693 E.00401
G1 X102.998 Y132.802 E.00509
G3 X103.021 Y135.689 I-73.465 J2.033 E.09141
G1 X103.021 Y141.689 E.18996
G1 X102.965 Y141.838 E.00502
G1 X102.795 Y141.915 E.00591
G1 X102.754 Y141.911 E.00131
G1 X99.021 Y141.22 E.1202
G1 X99.021 Y142.631 E.04469
G1 X98.937 Y143.857 E.0389
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.959 Y142.746 E.04578
G1 X157.025 Y142.391 E.01143
G3 X145.128 Y131.597 I28.628 J-43.512 E.51062
G3 X137.192 Y118.315 I40.815 J-33.397 E.49164
G1 X136.612 Y118.616 E.02068
G1 X135.912 Y118.859 E.02346
G1 X135.271 Y118.985 E.02069
G1 X134.371 Y119.041 E.02853
G1 X133.61 Y118.959 E.02426
G1 X132.865 Y118.766 E.02434
G1 X132.159 Y118.466 E.02429
G1 X131.506 Y118.066 E.02424
G1 X130.918 Y117.575 E.02428
G1 X130.409 Y117.003 E.02424
G3 X129.087 Y115.222 I885.331 J-658.636 E.0702
G1 X126.704 Y112.01 E.12664
G3 X120.781 Y118.619 I-42.058 J-31.731 E.28129
; LINE_WIDTH: 0.551636
G1 X120.416 Y119.007 E.01799
G1 X120.755 Y119.06 E.01158
; LINE_WIDTH: 0.519996
G1 X122.148 Y120.723 E.06866
G1 X122.2 Y120.884 E.00536
G1 X122.161 Y120.96 E.00273
; WIPE_START
M204 S10000
G1 X121.399 Y121.608 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.435 Y122.345 Z3.16 F36000
G1 Z2.76
G1 E.4 F1800
; LINE_WIDTH: 0.553776
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.01095
; LINE_WIDTH: 0.544336
G1 X120.199 Y119.2 E.15232
; LINE_WIDTH: 0.551636
G1 X120.416 Y119.007 E.00979
; WIPE_START
M204 S10000
G1 X120.199 Y119.2 E-.11037
G1 X119.655 Y119.656 E-.26963
; WIPE_END
G1 E-.02 F1800
G1 X113.198 Y123.726 Z3.16 F36000
G1 X100.188 Y131.929 Z3.16
G1 Z2.76
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.794516
G1 F10359.389
G1 X100.504 Y131.85 E.01614
; LINE_WIDTH: 0.833856
G1 F9849.661
G1 X100.82 Y131.771 E.01697
; LINE_WIDTH: 0.873196
G1 F9387.741
G1 X101.136 Y131.693 E.01781
; LINE_WIDTH: 0.877436
G1 F9340.53
G1 X101.167 Y131.685 E.00176
; LINE_WIDTH: 0.923601
G1 F8855.628
G1 X101.481 Y131.603 E.01881
; LINE_WIDTH: 0.969766
G1 F8418.588
G1 X101.795 Y131.522 E.01979
; LINE_WIDTH: 1.01593
G1 F8022.658
G1 X102.109 Y131.44 E.02076
; LINE_WIDTH: 1.0621
G1 F7662.295
G1 X102.423 Y131.359 E.02174
; LINE_WIDTH: 1.0757
G1 F7562.226
G1 X102.506 Y131.336 E.0058
; WIPE_START
G1 X102.423 Y131.359 E-.03247
G1 X102.109 Y131.44 E-.12329
G1 X101.795 Y131.522 E-.1233
G1 X101.538 Y131.589 E-.10094
; WIPE_END
G1 E-.02 F1800
G1 X100.739 Y139.179 Z3.16 F36000
G1 X100.355 Y142.828 Z3.16
G1 Z2.76
G1 E.4 F1800
; LINE_WIDTH: 1.0209
G1 F7982.282
G1 X100.556 Y142.847 E.01294
; LINE_WIDTH: 0.984096
G1 F8291.569
G1 X100.805 Y142.87 E.01548
; LINE_WIDTH: 0.938384
G1 F8710.826
G1 X101.054 Y142.893 E.01473
; LINE_WIDTH: 0.892671
G1 F9174.74
G1 X101.302 Y142.915 E.01399
; LINE_WIDTH: 0.846959
G1 F9690.847
G1 X101.551 Y142.938 E.01324
; LINE_WIDTH: 0.801246
G1 F10268.481
G1 X101.8 Y142.961 E.0125
; LINE_WIDTH: 0.755534
G1 F10919.34
G1 X102.049 Y142.984 E.01175
; LINE_WIDTH: 0.709821
G1 F11658.292
G1 X102.298 Y143.007 E.01101
; LINE_WIDTH: 0.664109
G1 F12504.517
G1 X102.547 Y143.03 E.01026
; LINE_WIDTH: 0.618396
G1 F13483.205
G1 X103.031 Y143.031 E.01841
; WIPE_START
G1 X102.547 Y143.03 E-.18378
G1 X102.298 Y143.007 E-.095
G1 X102.049 Y142.984 E-.095
G1 X102.033 Y142.982 E-.00622
; WIPE_END
G1 E-.02 F1800
G1 X105.243 Y139.634 Z3.16 F36000
G1 Z2.76
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.243 Y137.292 E.08944
G3 X107.257 Y138.338 I-1.873 J6.067 E.08713
M73 P54 R8
G3 X109.142 Y140.427 I-12.2 J12.904 E.10755
G2 X110.556 Y141.402 I2.62 J-2.287 E.06626
G2 X114.798 Y140.288 I.95 J-5.013 E.17307
G2 X116.683 Y138.199 I-12.2 J-12.904 E.10755
G3 X118.097 Y137.224 I2.62 J2.287 E.06626
G3 X122.339 Y138.338 I.95 J5.013 E.17307
G3 X124.224 Y140.427 I-12.201 J12.905 E.10755
G2 X125.638 Y141.402 I2.62 J-2.287 E.06626
G2 X129.879 Y140.288 I.95 J-5.013 E.17307
G2 X131.764 Y138.199 I-12.2 J-12.904 E.10755
G3 X133.178 Y137.224 I2.62 J2.287 E.06626
G3 X137.42 Y138.338 I.95 J5.013 E.17307
G3 X139.305 Y140.427 I-12.2 J12.904 E.10755
G2 X142.133 Y141.503 I2.354 J-1.933 E.12087
G2 X145.903 Y139.327 I-.764 J-5.679 E.17048
G3 X147.556 Y137.58 I6.045 J4.062 E.09223
G3 X144.201 Y133.962 I48.992 J-48.795 E.18841
G3 X141.19 Y132.747 I.329 J-5.154 E.12608
G3 X139.305 Y130.658 I12.199 J-12.903 E.10755
G2 X137.891 Y129.683 I-2.62 J2.287 E.06626
G2 X133.65 Y130.797 I-.95 J5.013 E.17307
G2 X131.764 Y132.887 I12.199 J12.903 E.10755
G3 X130.351 Y133.862 I-2.62 J-2.287 E.06626
G3 X126.109 Y132.747 I-.95 J-5.013 E.17307
G3 X124.224 Y130.658 I12.199 J-12.903 E.10755
G2 X122.81 Y129.683 I-2.62 J2.287 E.06626
G2 X118.568 Y130.797 I-.95 J5.013 E.17307
G2 X116.683 Y132.887 I12.199 J12.903 E.10755
G3 X115.269 Y133.862 I-2.62 J-2.287 E.06626
G3 X111.028 Y132.747 I-.95 J-5.013 E.17307
G3 X109.142 Y130.658 I12.199 J-12.903 E.10755
G2 X108.219 Y129.893 I-2.68 J2.295 E.04602
G2 X110.31 Y128.837 I-16.161 J-34.583 E.08945
; WIPE_START
G1 X109.417 Y129.288 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.462 Y126.352 Z3.16 F36000
G1 X130.939 Y120.319 Z3.16
G1 Z2.76
G1 E.4 F1800
G1 F13446.283
G3 X129.105 Y118.881 I4.696 J-7.875 E.08924
G3 X125.461 Y117.006 I.489 J-5.43 E.1604
G3 X123.592 Y118.986 I-45.21 J-40.817 E.10398
G3 X124.214 Y121.857 I-1.88 J1.911 E.11872
G3 X122.607 Y123.53 I-4.559 J-2.772 E.08931
G3 X124.224 Y125.346 I-10.614 J11.081 E.09293
G2 X125.638 Y126.321 I2.62 J-2.287 E.06626
G2 X129.879 Y125.207 I.95 J-5.013 E.17307
G2 X131.764 Y123.118 I-12.199 J-12.903 E.10755
G3 X133.178 Y122.143 I2.62 J2.287 E.06626
G3 X136.814 Y122.849 I.95 J4.823 E.14499
G2 X137.912 Y124.918 I31.728 J-15.501 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 2.92
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.443 Y124.035 E-.38
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
G1 X123.492 Y122.076
G1 Z2.92
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.374 Y122.23 E.00742
G3 X122.316 Y123.126 I-41.424 J-47.815 E.05295
G1 X119.25 Y125.696 E.15272
G1 X118.894 Y125.933 E.01634
G1 X118.471 Y126.093 E.01726
G1 X117.997 Y126.152 E.01821
G3 X116.851 Y125.779 I.042 J-2.08 E.04669
G3 X115.948 Y124.792 I4.806 J-5.304 E.05116
G3 X103.787 Y131.247 I-31.326 J-44.333 E.52705
G1 X104.05 Y131.405 E.01173
G1 X104.396 Y131.783 E.01955
G1 X104.691 Y132.432 E.02721
G3 X104.747 Y133.692 I-6.359 J.918 E.04823
G1 X104.747 Y141.692 E.30543
G1 X104.652 Y142.293 E.02324
G1 X104.567 Y142.443 E.0066
G1 X154.069 Y142.443 E1.88993
G3 X143.802 Y132.7 I31.508 J-43.483 E.54196
G3 X136.271 Y120.545 I41.695 J-34.245 E.54755
G1 X135.451 Y120.705 E.03189
G3 X133.274 Y120.651 I-.887 J-8.187 E.08338
G1 X132.429 Y120.434 E.03332
G1 X131.48 Y120.051 E.03906
G1 X130.604 Y119.536 E.03881
G1 X129.81 Y118.897 E.03891
G1 X129.118 Y118.145 E.03901
G3 X127.853 Y116.453 I75.008 J-57.358 E.08065
G1 X126.662 Y114.847 E.07636
G3 X122.936 Y118.937 I-42.281 J-34.77 E.2113
G3 X123.655 Y119.825 I-5.797 J5.422 E.04365
G1 X123.893 Y120.399 E.02372
G1 X123.946 Y120.846 E.0172
G1 X123.874 Y121.375 E.02039
G1 X123.685 Y121.825 E.01863
G1 X123.547 Y122.004 E.00863
G1 X122.994 Y121.762 F36000
G1 F13446.369
G1 X122.872 Y121.895 E.0069
G1 X118.874 Y125.247 E.1992
G1 X118.53 Y125.458 E.01538
G1 X118.115 Y125.561 E.01635
G1 X117.664 Y125.524 E.01728
G1 X117.216 Y125.32 E.0188
G1 X116.858 Y124.966 E.01921
G1 X116.051 Y124.003 E.04798
G3 X104.479 Y130.352 I-31.621 J-43.909 E.50517
G1 X104.066 Y130.511 E.01691
G1 X103.693 Y130.655 E.01527
G1 F12245.736
G1 X103.32 Y130.799 E.01527
; LINE_WIDTH: 0.665402
G1 F10912.661
G1 X103.238 Y130.853 E.004
; LINE_WIDTH: 0.710808
G1 F10600.211
G1 X103.157 Y130.906 E.00429
; LINE_WIDTH: 0.756214
G1 F10292.283
G1 X103.076 Y130.96 E.00458
; LINE_WIDTH: 0.80162
G1 F9988.893
G1 X102.995 Y131.013 E.00486
; LINE_WIDTH: 0.847026
G1 F9690.042
G1 X102.914 Y131.067 E.00515
; LINE_WIDTH: 0.892432
G1 F9177.295
G1 X102.833 Y131.12 E.00544
; LINE_WIDTH: 0.937838
G1 F8716.086
G1 X102.751 Y131.174 E.00573
; LINE_WIDTH: 0.983244
G1 F8299.014
G1 X102.67 Y131.228 E.00602
; LINE_WIDTH: 1.02865
G1 F7920.034
G1 X102.589 Y131.281 E.0063
; LINE_WIDTH: 1.07406
G1 F7574.155
G1 X102.508 Y131.335 E.00659
G1 X102.586 Y131.364 E.00563
; LINE_WIDTH: 1.02865
G1 F7920.034
G1 X102.663 Y131.393 E.00538
; LINE_WIDTH: 0.983244
G1 F8299.014
G1 X102.741 Y131.422 E.00514
; LINE_WIDTH: 0.937838
G1 F8716.086
G1 X102.819 Y131.451 E.00489
; LINE_WIDTH: 0.892432
G1 F9177.295
G1 X102.897 Y131.48 E.00464
; LINE_WIDTH: 0.847026
G1 F9690.042
G1 X102.974 Y131.509 E.0044
; LINE_WIDTH: 0.80162
G1 F10263.476
G1 X103.052 Y131.538 E.00415
; LINE_WIDTH: 0.756214
G1 F10525.701
G1 X103.13 Y131.567 E.00391
; LINE_WIDTH: 0.710808
G1 F10791.204
G1 X103.208 Y131.596 E.00366
; LINE_WIDTH: 0.665402
G1 F11060.026
G1 X103.285 Y131.625 E.00341
; LINE_WIDTH: 0.619996
G1 F12578.507
G1 X103.674 Y131.854 E.01722
G1 F13446.369
G1 X103.932 Y132.142 E.01477
G1 X104.128 Y132.598 E.01894
G3 X104.162 Y133.692 I-6.642 J.752 E.04183
G1 X104.162 Y141.692 E.30543
G1 X104.095 Y142.112 E.01626
G1 X103.825 Y142.589 E.02094
G1 X103.275 Y142.967 E.02546
; LINE_WIDTH: 0.616306
G1 F13531.627
G1 X103.033 Y143.032 E.00953
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.856 J418.892 E.10369
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.253 Y132.327 I30.06 J-44.232 E.59952
G3 X136.596 Y119.828 I41.156 J-33.81 E.56143
G1 X136.066 Y119.995 E.0212
G1 X135.344 Y120.13 E.02805
G3 X133.382 Y120.076 I-.776 J-7.516 E.07512
G1 X132.578 Y119.868 E.03172
G1 X131.711 Y119.513 E.03577
G1 X130.911 Y119.037 E.03552
G1 X130.187 Y118.448 E.03564
G3 X129.066 Y117.105 I8.434 J-8.18 E.06684
G1 X126.683 Y113.893 E.15272
G3 X122.14 Y118.898 I-42.658 J-34.156 E.25824
G1 X123.041 Y119.974 E.05358
G1 X123.285 Y120.403 E.01885
G1 X123.36 Y120.847 E.0172
G1 X123.31 Y121.218 E.01426
G1 X123.119 Y121.624 E.01714
G1 X123.055 Y121.695 E.00367
G1 X122.561 Y121.369 F36000
G1 F13446.369
G1 X122.496 Y121.447 E.00387
G1 X118.498 Y124.798 E.1992
G1 X118.245 Y124.94 E.01105
G1 X117.94 Y124.978 E.01175
G1 X117.689 Y124.917 E.00986
G1 X117.401 Y124.702 E.01372
G1 X116.148 Y123.208 E.07444
G3 X99.49 Y131.452 I-31.759 J-43.22 E.71322
G3 X99.574 Y132.704 I-15.083 J1.632 E.04794
G1 X102.656 Y132.134 E.11968
G1 X103.091 Y132.178 E.0167
G3 X103.445 Y132.467 I-.293 J.721 E.01771
G1 X103.557 Y132.727 E.01081
G3 X103.576 Y135.692 I-96.287 J2.108 E.11319
G1 X103.576 Y141.692 E.22907
G1 X103.538 Y141.932 E.00928
G1 X103.384 Y142.204 E.01195
G1 X103.071 Y142.421 E.01454
G1 X102.798 Y142.47 E.0106
G1 X102.656 Y142.457 E.00544
G1 X99.576 Y141.887 E.11958
G3 X99.517 Y143.615 I-13.62 J.397 E.06605
G1 X156.235 Y143.615 E2.16547
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.105 J-43.631 E.60778
G3 X136.912 Y119.083 I41.137 J-33.703 E.57638
G1 X136.092 Y119.382 E.03329
G3 X132.721 Y119.3 I-1.547 J-5.731 E.13054
G1 X131.942 Y118.974 E.03226
G1 X131.218 Y118.539 E.03224
G1 X130.564 Y118 E.03236
G3 X129.081 Y116.142 I11.98 J-11.085 E.09083
G1 X126.698 Y112.93 E.15272
G3 X121.339 Y118.854 I-42.406 J-32.977 E.30528
G1 X122.593 Y120.35 E.07453
G1 X122.753 Y120.67 E.01367
G1 X122.766 Y120.963 E.0112
G1 X122.67 Y121.239 E.01117
G1 X122.619 Y121.3 E.00303
M204 S250
G1 X122.141 Y121.023 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.142 Y124.375 E.16519
G1 X117.981 Y124.427 E.00538
G1 X117.825 Y124.347 E.00556
G1 X116.847 Y123.18 E.04819
; LINE_WIDTH: 0.521466
G1 X116.43 Y122.686 E.02054
; LINE_WIDTH: 0.555176
G1 X116.414 Y122.362 E.011
G1 X115.477 Y123.015 E.03877
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.117 J-43.088 E.58496
G1 X98.941 Y131.547 E.01613
G3 X99.022 Y133.369 I-11.67 J1.437 E.05778
G1 X102.756 Y132.677 E.12023
G1 X102.943 Y132.726 E.00609
G1 X103.018 Y132.849 E.00456
G3 X103.023 Y135.692 I-287.41 J1.985 E.08999
G1 X103.023 Y141.692 E.18996
G1 X102.967 Y141.84 E.00502
G1 X102.798 Y141.917 E.00591
G1 X102.756 Y141.914 E.00131
G1 X99.023 Y141.222 E.1202
G1 X99.023 Y142.631 E.04461
G1 X98.937 Y143.857 E.0389
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.639 J-43.527 E.51061
G3 X137.192 Y118.315 I40.594 J-33.265 E.49165
G1 X136.614 Y118.615 E.02061
G1 X135.912 Y118.859 E.02352
G1 X135.135 Y119.01 E.02507
G1 X134.371 Y119.041 E.02421
G1 X133.609 Y118.959 E.02425
G1 X132.867 Y118.767 E.02427
G1 X132.159 Y118.466 E.02436
G1 X131.508 Y118.068 E.02417
G1 X130.92 Y117.577 E.02427
G1 X130.409 Y117.003 E.02432
G3 X129.087 Y115.222 I881.044 J-655.46 E.07021
G1 X126.704 Y112.01 E.12664
G3 X120.802 Y118.599 I-42.07 J-31.741 E.28038
; LINE_WIDTH: 0.553196
G1 X120.436 Y118.989 E.0181
G1 X120.776 Y119.043 E.01163
; LINE_WIDTH: 0.519996
G1 X122.169 Y120.705 E.06866
G1 X122.221 Y120.867 E.00538
G1 X122.182 Y120.943 E.00271
; WIPE_START
M204 S10000
G1 X121.42 Y121.59 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.414 Y122.362 Z3.32 F36000
G1 Z2.92
G1 E.4 F1800
; LINE_WIDTH: 0.555176
G1 F3600
M204 S5000
G1 X116.687 Y122.143 E.0119
; LINE_WIDTH: 0.544336
G1 X120.199 Y119.2 E.15232
; LINE_WIDTH: 0.553196
G1 X120.436 Y118.989 E.01074
; WIPE_START
M204 S10000
G1 X120.199 Y119.2 E-.12066
G1 X119.676 Y119.638 E-.25934
; WIPE_END
G1 E-.02 F1800
G1 X113.22 Y123.709 Z3.32 F36000
G1 X100.188 Y131.927 Z3.32
G1 Z2.92
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.792476
G1 F10387.265
G1 X100.504 Y131.849 E.01609
; LINE_WIDTH: 0.831816
G1 F9874.856
G1 X100.82 Y131.77 E.01693
; LINE_WIDTH: 0.871156
G1 F9410.626
G1 X101.136 Y131.692 E.01776
; LINE_WIDTH: 0.875396
G1 F9363.186
G1 X101.167 Y131.684 E.00176
; LINE_WIDTH: 0.921536
G1 F8876.24
G1 X101.481 Y131.602 E.01876
; LINE_WIDTH: 0.967676
G1 F8437.441
G1 X101.795 Y131.521 E.01973
; LINE_WIDTH: 1.01382
G1 F8039.981
G1 X102.109 Y131.439 E.02071
; LINE_WIDTH: 1.05996
G1 F7678.282
G1 X102.423 Y131.358 E.02168
; LINE_WIDTH: 1.07406
G1 F7574.155
G1 X102.508 Y131.335 E.00599
; WIPE_START
G1 X102.423 Y131.358 E-.03358
G1 X102.109 Y131.439 E-.12324
G1 X101.795 Y131.521 E-.12324
G1 X101.54 Y131.587 E-.09994
; WIPE_END
G1 E-.02 F1800
G1 X100.741 Y139.177 Z3.32 F36000
G1 X100.356 Y142.83 Z3.32
G1 Z2.92
G1 E.4 F1800
; LINE_WIDTH: 1.01864
G1 F8000.61
G1 X100.558 Y142.848 E.01301
; LINE_WIDTH: 0.981566
G1 F8313.715
G1 X100.807 Y142.871 E.01544
; LINE_WIDTH: 0.935852
G1 F8735.284
G1 X101.056 Y142.894 E.01469
; LINE_WIDTH: 0.890139
G1 F9201.89
G1 X101.305 Y142.917 E.01395
; LINE_WIDTH: 0.844425
G1 F9721.158
G1 X101.554 Y142.94 E.0132
; LINE_WIDTH: 0.798711
G1 F10302.535
G1 X101.803 Y142.962 E.01246
; LINE_WIDTH: 0.752998
G1 F10957.875
G1 X102.052 Y142.985 E.01171
; LINE_WIDTH: 0.707284
G1 F11702.251
G1 X102.301 Y143.008 E.01097
; LINE_WIDTH: 0.66157
G1 F12555.129
G1 X102.55 Y143.031 E.01022
; LINE_WIDTH: 0.615856
G1 F13542.099
G1 X103.033 Y143.032 E.01832
; WIPE_START
G1 X102.55 Y143.031 E-.18361
G1 X102.301 Y143.008 E-.095
G1 X102.052 Y142.985 E-.095
G1 X102.035 Y142.984 E-.0064
; WIPE_END
G1 E-.02 F1800
G1 X105.245 Y139.674 Z3.32 F36000
G1 Z2.92
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.245 Y137.331 E.08944
G3 X107.257 Y138.268 I-1.55 J5.962 E.0852
G3 X109.142 Y140.223 I-9.748 J11.286 E.10383
G2 X110.556 Y141.239 I3.031 J-2.723 E.06701
G2 X114.798 Y140.359 I1.222 J-4.767 E.17126
G2 X116.683 Y138.404 I-9.748 J-11.286 E.10383
G3 X118.097 Y137.387 I3.031 J2.723 E.06701
G3 X122.339 Y138.268 I1.222 J4.767 E.17126
G3 X124.224 Y140.223 I-9.748 J11.287 E.10383
G2 X125.638 Y141.239 I3.031 J-2.723 E.06701
G2 X129.879 Y140.359 I1.222 J-4.767 E.17126
G2 X131.764 Y138.404 I-9.748 J-11.286 E.10383
G3 X133.178 Y137.387 I3.031 J2.723 E.06701
G3 X137.42 Y138.268 I1.222 J4.767 E.17126
G3 X139.305 Y140.223 I-9.748 J11.286 E.10383
G2 X141.19 Y141.369 I2.719 J-2.349 E.08562
G2 X145.903 Y139.47 I.594 J-5.324 E.20211
G3 X147.674 Y137.693 I6.869 J5.073 E.09616
G3 X144.129 Y133.876 I38.594 J-39.406 E.19897
G3 X141.19 Y132.818 I.151 J-5.03 E.12124
G3 X139.305 Y130.863 I9.749 J-11.287 E.10383
G2 X137.891 Y129.846 I-3.031 J2.723 E.06701
G2 X133.65 Y130.727 I-1.222 J4.767 E.17126
G2 X131.764 Y132.682 I9.748 J11.286 E.10383
G3 X130.351 Y133.699 I-3.031 J-2.723 E.06701
G3 X126.109 Y132.818 I-1.222 J-4.767 E.17126
G3 X124.224 Y130.863 I9.749 J-11.287 E.10383
G2 X122.81 Y129.846 I-3.031 J2.723 E.06701
G2 X118.568 Y130.727 I-1.222 J4.767 E.17126
G2 X116.683 Y132.682 I9.748 J11.286 E.10383
G3 X114.798 Y133.828 I-2.719 J-2.349 E.08562
G3 X110.085 Y131.93 I-.594 J-5.324 E.20211
G2 X108.014 Y129.982 I-6.273 J4.594 E.10916
G2 X110.113 Y128.943 I-16.325 J-35.619 E.08944
; WIPE_START
G1 X109.217 Y129.386 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.25 Y126.42 Z3.32 F36000
G1 X130.84 Y120.265 Z3.32
G1 Z2.92
G1 E.4 F1800
G1 F13446.283
G3 X129.031 Y118.795 I4.683 J-7.61 E.08924
G3 X125.403 Y117.071 I.314 J-5.341 E.15725
G3 X123.615 Y118.966 I-34.385 J-30.647 E.09947
G3 X124.232 Y121.846 I-1.898 J1.913 E.11896
G3 X122.658 Y123.487 I-4.507 J-2.747 E.08753
G3 X124.224 Y125.141 I-8.262 J9.39 E.08707
G2 X125.638 Y126.158 I3.031 J-2.723 E.06701
G2 X129.879 Y125.277 I1.222 J-4.767 E.17126
G2 X131.764 Y123.322 I-9.748 J-11.286 E.10383
G3 X133.178 Y122.305 I3.031 J2.723 E.06701
G3 X136.79 Y122.799 I1.212 J4.589 E.14283
G2 X137.884 Y124.87 I58.432 J-29.536 E.08943
; CHANGE_LAYER
; Z_HEIGHT: 3.08
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.417 Y123.986 E-.38
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
G1 X123.504 Y122.066
G1 Z3.08
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.385 Y122.22 E.00744
G3 X122.302 Y123.138 I-49.776 J-57.677 E.0542
G1 X119.236 Y125.707 E.15272
G1 X118.88 Y125.945 E.01635
G1 X118.457 Y126.105 E.01727
G3 X117.205 Y126 I-.455 J-2.114 E.04868
G1 X116.829 Y125.784 E.01653
G3 X115.933 Y124.802 I5.101 J-5.553 E.05084
G3 X103.791 Y131.245 I-31.511 J-44.72 E.52618
G1 X104.05 Y131.4 E.0115
G1 X104.395 Y131.776 E.01951
G1 X104.692 Y132.428 E.02736
G3 X104.749 Y133.694 I-6.368 J.921 E.04845
G1 X104.749 Y141.694 E.30543
G1 X104.654 Y142.295 E.02324
G1 X104.571 Y142.443 E.00649
G1 X154.07 Y142.443 E1.88982
G3 X143.803 Y132.702 I31.776 J-43.768 E.54188
G3 X136.269 Y120.54 I42.215 J-34.569 E.54778
G3 X135.116 Y120.741 I-1.615 J-5.858 E.04476
G1 X134.176 Y120.751 E.03589
G1 X133.156 Y120.624 E.03922
G1 X132.502 Y120.455 E.02581
G1 X131.484 Y120.053 E.04179
G1 X130.602 Y119.535 E.03904
G1 X129.81 Y118.896 E.03884
G1 X129.117 Y118.145 E.03901
G3 X127.853 Y116.453 I76.474 J-58.454 E.08064
G1 X126.662 Y114.847 E.07636
G3 X122.949 Y118.924 I-42.323 J-34.807 E.21063
G3 X123.662 Y119.804 I-6.003 J5.594 E.04327
G1 X123.907 Y120.389 E.02422
G1 X123.959 Y120.837 E.01722
G1 X123.887 Y121.365 E.02035
G1 X123.696 Y121.817 E.01872
G1 X123.559 Y121.995 E.00856
G1 X122.985 Y121.764 F36000
G1 F13446.369
G1 X122.788 Y121.966 E.01077
G1 X118.86 Y125.259 E.19568
G1 X118.517 Y125.469 E.01538
G1 X118.101 Y125.572 E.01635
G3 X117.176 Y125.312 I-.085 J-1.474 E.03735
G1 X116.795 Y124.918 E.02094
G1 X116.036 Y124.013 E.04509
G3 X104.475 Y130.353 I-31.748 J-44.183 E.50465
G1 X104.066 Y130.511 E.01674
G1 X103.693 Y130.655 E.01527
G1 F12253.06
G1 X103.32 Y130.799 E.01527
; LINE_WIDTH: 0.665234
G1 F10919.575
G1 X103.239 Y130.853 E.00399
; LINE_WIDTH: 0.710472
G1 F10607.964
G1 X103.158 Y130.906 E.00427
; LINE_WIDTH: 0.75571
G1 F10300.82
G1 X103.077 Y130.959 E.00456
; LINE_WIDTH: 0.800948
G1 F9998.188
G1 X102.996 Y131.013 E.00485
; LINE_WIDTH: 0.846186
G1 F9700.069
G1 X102.915 Y131.066 E.00513
; LINE_WIDTH: 0.891424
G1 F9188.088
G1 X102.834 Y131.119 E.00542
; LINE_WIDTH: 0.936662
G1 F8727.445
G1 X102.753 Y131.173 E.0057
; LINE_WIDTH: 0.9819
G1 F8310.785
G1 X102.672 Y131.226 E.00599
; LINE_WIDTH: 1.02714
G1 F7932.095
G1 X102.591 Y131.28 E.00627
; LINE_WIDTH: 1.07238
G1 F7586.413
G1 X102.51 Y131.333 E.00656
G1 X102.588 Y131.362 E.0056
; LINE_WIDTH: 1.02714
G1 F7932.095
G1 X102.665 Y131.391 E.00536
; LINE_WIDTH: 0.9819
G1 F8310.785
G1 X102.743 Y131.42 E.00511
; LINE_WIDTH: 0.936662
G1 F8727.445
G1 X102.82 Y131.449 E.00487
; LINE_WIDTH: 0.891424
G1 F9188.088
G1 X102.898 Y131.478 E.00463
; LINE_WIDTH: 0.846186
G1 F9700.069
G1 X102.976 Y131.506 E.00438
; LINE_WIDTH: 0.800948
G1 F10272.472
G1 X103.053 Y131.535 E.00414
; LINE_WIDTH: 0.75571
G1 F10534.12
G1 X103.131 Y131.564 E.00389
; LINE_WIDTH: 0.710472
G1 F10799.058
G1 X103.208 Y131.593 E.00365
; LINE_WIDTH: 0.665234
G1 F11067.287
G1 X103.286 Y131.622 E.00341
; LINE_WIDTH: 0.619996
G1 F12582.603
G1 X103.674 Y131.85 E.01718
G1 F13446.369
G1 X103.932 Y132.137 E.01474
G1 X104.13 Y132.595 E.01905
G3 X104.164 Y133.694 I-6.652 J.755 E.04204
G1 X104.164 Y141.694 E.30543
G1 X104.097 Y142.115 E.01626
G1 X103.827 Y142.592 E.02094
G1 X103.277 Y142.969 E.02545
; LINE_WIDTH: 0.614856
G1 F13565.425
G1 X103.035 Y143.033 E.0095
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.854 J299.769 E.10362
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00187
G3 X144.255 Y132.328 I30.045 J-44.215 E.59946
G3 X136.599 Y119.835 I41.128 J-33.795 E.56122
G1 X135.959 Y120.02 E.02541
G1 X135.036 Y120.161 E.03564
G1 X134.183 Y120.166 E.0326
G1 X133.246 Y120.046 E.03603
G1 X132.66 Y119.891 E.02316
G1 X131.714 Y119.514 E.03887
G1 X130.909 Y119.036 E.03575
G1 X130.187 Y118.448 E.03556
G3 X129.066 Y117.106 I8.492 J-8.228 E.06683
G1 X126.683 Y113.893 E.15272
G3 X122.152 Y118.886 I-42.655 J-34.154 E.25757
G1 X123.055 Y119.963 E.05364
G1 X123.298 Y120.39 E.01876
G1 X123.373 Y120.838 E.01735
G1 X123.323 Y121.207 E.01424
G1 X123.131 Y121.615 E.01721
G1 X123.048 Y121.7 E.00453
G1 X122.574 Y121.358 F36000
G1 F13446.369
G1 X122.509 Y121.435 E.00385
G1 X118.484 Y124.81 E.20054
G1 X118.231 Y124.951 E.01105
G1 X117.926 Y124.99 E.01175
G1 X117.673 Y124.927 E.00996
G1 X117.387 Y124.713 E.01361
G1 X116.134 Y123.218 E.07448
G3 X99.49 Y131.452 I-31.629 J-42.997 E.71257
G3 X99.576 Y132.702 I-14.701 J1.632 E.04785
G1 X102.658 Y132.131 E.11968
G1 X103.092 Y132.175 E.01667
G3 X103.446 Y132.463 I-.293 J.721 E.01767
G1 X103.559 Y132.724 E.01087
M73 P55 R8
G3 X103.578 Y135.694 I-96.221 J2.11 E.11339
G1 X103.578 Y141.694 E.22907
G1 X103.54 Y141.934 E.00928
G1 X103.386 Y142.206 E.01195
G1 X103.073 Y142.423 E.01454
G1 X102.8 Y142.473 E.0106
G1 X102.658 Y142.46 E.00544
G1 X99.578 Y141.889 E.11958
G3 X99.517 Y143.615 I-13.299 J.395 E.06596
G1 X156.235 Y143.615 E2.16544
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.706 Y131.955 I29.319 J-43.865 E.6077
G3 X136.908 Y119.075 I40.725 J-33.456 E.5768
G1 X136.498 Y119.254 E.01708
G1 X135.796 Y119.458 E.0279
G1 X134.957 Y119.581 E.03237
G1 X134.19 Y119.58 E.02931
G1 X133.337 Y119.467 E.03285
G1 X132.818 Y119.327 E.02051
G1 X131.945 Y118.976 E.03594
G1 X131.216 Y118.537 E.03246
G1 X130.564 Y118 E.03227
G3 X129.081 Y116.142 I12.054 J-11.144 E.09082
G1 X126.698 Y112.93 E.15272
G3 X121.351 Y118.842 I-42.433 J-33 E.30461
G1 X122.606 Y120.339 E.07457
G1 X122.767 Y120.659 E.01369
G1 X122.78 Y120.953 E.0112
G1 X122.683 Y121.229 E.0112
G1 X122.632 Y121.289 E.00299
M204 S250
G1 X122.154 Y121.012 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.129 Y124.386 E.1663
G1 X117.967 Y124.438 E.00538
G1 X117.811 Y124.358 E.00556
G1 X116.821 Y123.177 E.04879
; LINE_WIDTH: 0.521366
G1 X116.417 Y122.697 E.01993
; LINE_WIDTH: 0.556076
G1 X116.4 Y122.373 E.01103
G1 X115.477 Y123.015 E.03824
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.117 J-43.088 E.58496
G1 X98.941 Y131.547 E.01613
G3 X99.025 Y133.366 I-11.384 J1.436 E.05772
G1 X102.759 Y132.675 E.12023
G1 X102.944 Y132.723 E.00608
G1 X103.02 Y132.847 E.00458
G3 X103.025 Y135.694 I-287.241 J1.988 E.09015
G1 X103.025 Y141.694 E.18996
G1 X102.97 Y141.843 E.00502
G1 X102.8 Y141.92 E.00591
G1 X102.759 Y141.916 E.00131
G1 X99.025 Y141.225 E.1202
G1 X99.025 Y142.631 E.04453
G1 X98.937 Y143.857 E.0389
G1 X98.936 Y144.167 E.00984
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.615 J-43.5 E.51059
G3 X137.192 Y118.316 I40.595 J-33.266 E.49164
G3 X136.344 Y118.723 I-5.229 J-9.813 E.0298
G1 X135.642 Y118.927 E.02314
G1 X134.883 Y119.033 E.02427
G1 X134.196 Y119.027 E.02173
G1 X133.422 Y118.921 E.02475
G1 X132.864 Y118.766 E.01833
G1 X132.162 Y118.468 E.02414
G1 X131.506 Y118.066 E.02435
G1 X130.92 Y117.577 E.02419
G1 X130.409 Y117.003 E.02433
G3 X129.087 Y115.222 I543.378 J-404.777 E.0702
G1 X126.704 Y112.01 E.12664
G3 X120.816 Y118.586 I-42.082 J-31.752 E.27979
; LINE_WIDTH: 0.554176
G1 X120.449 Y118.978 E.01817
G1 X120.789 Y119.032 E.01167
; LINE_WIDTH: 0.519996
G1 X122.182 Y120.694 E.06866
G1 X122.234 Y120.856 E.00539
G1 X122.195 Y120.932 E.0027
; WIPE_START
M204 S10000
G1 X121.433 Y121.579 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.4 Y122.373 Z3.48 F36000
G1 Z3.08
G1 E.4 F1800
; LINE_WIDTH: 0.556076
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.01253
; LINE_WIDTH: 0.544336
G1 X120.199 Y119.2 E.15232
; LINE_WIDTH: 0.554176
G1 X120.449 Y118.978 E.01136
; WIPE_START
M204 S10000
G1 X120.199 Y119.2 E-.12735
G1 X119.689 Y119.627 E-.25265
; WIPE_END
G1 E-.02 F1800
G1 X113.382 Y123.925 Z3.48 F36000
G1 X102.51 Y131.333 Z3.48
G1 Z3.08
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.07238
G1 F7586.413
G1 X102.422 Y131.357 E.00616
; LINE_WIDTH: 1.05784
G1 F7694.187
G1 X102.108 Y131.438 E.02164
; LINE_WIDTH: 1.0117
G1 F8057.421
G1 X101.794 Y131.52 E.02066
; LINE_WIDTH: 0.965556
G1 F8456.649
G1 X101.48 Y131.601 E.01969
; LINE_WIDTH: 0.919416
G1 F8897.502
G1 X101.167 Y131.683 E.01871
; LINE_WIDTH: 0.873276
G1 F9386.846
G1 X101.136 Y131.691 E.00175
; LINE_WIDTH: 0.869056
G1 F9434.302
G1 X100.82 Y131.769 E.01771
; LINE_WIDTH: 0.829736
G1 F9900.679
G1 X100.504 Y131.848 E.01688
; LINE_WIDTH: 0.790416
G1 F10415.565
G1 X100.188 Y131.926 E.01604
; WIPE_START
G1 X100.504 Y131.848 E-.12367
G1 X100.82 Y131.769 E-.12367
G1 X101.136 Y131.691 E-.12367
G1 X101.158 Y131.685 E-.009
; WIPE_END
G1 E-.02 F1800
G1 X100.611 Y139.298 Z3.48 F36000
G1 X100.357 Y142.831 Z3.48
G1 Z3.08
G1 E.4 F1800
; LINE_WIDTH: 1.0164
G1 F8018.858
G1 X100.56 Y142.849 E.01307
; LINE_WIDTH: 0.979056
G1 F8335.804
G1 X100.809 Y142.872 E.0154
; LINE_WIDTH: 0.933341
G1 F8759.685
G1 X101.058 Y142.895 E.01465
; LINE_WIDTH: 0.887626
G1 F9228.985
G1 X101.307 Y142.918 E.01391
; LINE_WIDTH: 0.841911
G1 F9751.417
G1 X101.556 Y142.941 E.01316
; LINE_WIDTH: 0.796196
G1 F10336.546
G1 X101.805 Y142.964 E.01242
; LINE_WIDTH: 0.750481
G1 F10996.378
G1 X102.054 Y142.987 E.01167
; LINE_WIDTH: 0.704766
G1 F11746.193
G1 X102.303 Y143.009 E.01093
; LINE_WIDTH: 0.659051
G1 F12605.75
G1 X102.552 Y143.032 E.01018
; LINE_WIDTH: 0.613336
G1 F13601.038
G1 X103.035 Y143.033 E.01822
; WIPE_START
G1 X102.552 Y143.032 E-.18343
G1 X102.303 Y143.009 E-.095
G1 X102.054 Y142.987 E-.095
G1 X102.037 Y142.985 E-.00657
; WIPE_END
G1 E-.02 F1800
G1 X105.247 Y139.707 Z3.48 F36000
G1 Z3.08
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.247 Y137.364 E.08944
G3 X107.257 Y138.194 I-1.24 J5.853 E.08349
G3 X109.142 Y140.034 I-7.803 J9.881 E.10076
G2 X110.556 Y141.088 I3.496 J-3.213 E.06773
G2 X114.798 Y140.432 I1.465 J-4.568 E.16989
G2 X116.683 Y138.592 I-7.803 J-9.881 E.10076
G3 X118.097 Y137.538 I3.496 J3.213 E.06773
G3 X122.339 Y138.194 I1.465 J4.567 E.16989
G3 X124.224 Y140.034 I-7.803 J9.881 E.10076
G2 X125.638 Y141.088 I3.496 J-3.213 E.06773
G2 X129.879 Y140.432 I1.465 J-4.568 E.16989
G2 X131.764 Y138.592 I-7.803 J-9.881 E.10076
G3 X133.178 Y137.538 I3.496 J3.213 E.06773
G3 X137.42 Y138.194 I1.465 J4.567 E.16989
G3 X139.305 Y140.034 I-7.803 J9.881 E.10076
G2 X141.19 Y141.244 I3.098 J-2.753 E.08661
G2 X145.903 Y139.609 I.879 J-5.076 E.19882
G3 X147.777 Y137.794 I8.207 J6.602 E.09982
G3 X144.069 Y133.803 I44.819 J-45.354 E.20806
G3 X141.19 Y132.892 I-.025 J-4.922 E.11717
G3 X139.305 Y131.052 I7.804 J-9.882 E.10076
G2 X137.891 Y129.997 I-3.496 J3.213 E.06773
G2 X133.65 Y130.653 I-1.465 J4.568 E.16989
G2 X131.764 Y132.493 I7.803 J9.881 E.10076
G3 X130.351 Y133.548 I-3.496 J-3.213 E.06773
G3 X126.109 Y132.892 I-1.465 J-4.568 E.16989
G3 X124.224 Y131.052 I7.804 J-9.882 E.10076
G2 X122.81 Y129.997 I-3.496 J3.213 E.06773
G2 X118.568 Y130.653 I-1.465 J4.568 E.16989
G2 X116.683 Y132.493 I7.803 J9.881 E.10076
G3 X114.798 Y133.704 I-3.098 J-2.752 E.08661
G3 X110.085 Y132.068 I-.879 J-5.076 E.19882
G2 X107.848 Y130.06 I-6.991 J5.539 E.11533
G2 X109.952 Y129.031 I-20.672 J-44.967 E.08944
; WIPE_START
G1 X109.054 Y129.471 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.075 Y126.477 Z3.48 F36000
G1 X130.76 Y120.215 Z3.48
G1 Z3.08
G1 E.4 F1800
G1 F13446.283
G3 X128.969 Y118.724 I4.611 J-7.359 E.08923
G3 X125.341 Y117.139 I.14 J-5.266 E.15497
G3 X123.622 Y118.952 I-30.694 J-27.382 E.09541
G3 X124.245 Y121.837 I-1.886 J1.917 E.11925
G3 X122.714 Y123.441 I-4.429 J-2.695 E.08531
G3 X124.224 Y124.953 I-6.432 J7.934 E.08173
G2 X125.638 Y126.007 I3.496 J-3.213 E.06773
G2 X129.879 Y125.351 I1.465 J-4.568 E.16989
G2 X131.764 Y123.511 I-7.803 J-9.881 E.10076
G3 X133.178 Y122.457 I3.496 J3.213 E.06773
G3 X136.765 Y122.748 I1.447 J4.41 E.14109
G2 X137.856 Y124.821 I55.337 J-27.797 E.08943
; CHANGE_LAYER
; Z_HEIGHT: 3.24
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.39 Y123.936 E-.38
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
G1 X123.51 Y122.061
G1 Z3.24
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.39 Y122.216 E.00746
G3 X122.297 Y123.142 I-51.078 J-59.215 E.05473
G1 X119.231 Y125.712 E.15272
G1 X118.875 Y125.949 E.01635
G1 X118.451 Y126.109 E.01727
G3 X117.196 Y126.004 I-.454 J-2.116 E.04878
G1 X116.821 Y125.787 E.01657
G3 X115.927 Y124.806 I5.222 J-5.655 E.0507
G3 X103.767 Y131.255 I-31.338 J-44.401 E.52694
G1 X104.277 Y131.619 E.0239
G1 X104.548 Y132.027 E.01872
G1 X104.691 Y132.413 E.01573
G3 X104.752 Y133.697 I-6.237 J.936 E.04914
G1 X104.752 Y141.697 E.30543
G1 X104.657 Y142.298 E.02324
G1 X104.574 Y142.443 E.00638
G1 X154.069 Y142.443 E1.88966
G3 X143.803 Y132.701 I31.971 J-43.97 E.54186
G3 X136.271 Y120.546 I42.208 J-34.565 E.54753
G1 X135.444 Y120.707 E.03218
G3 X133.351 Y120.664 I-.88 J-8.212 E.08015
G1 X132.429 Y120.434 E.03626
G1 X131.481 Y120.051 E.03902
G1 X130.602 Y119.534 E.03896
G1 X129.808 Y118.894 E.03892
G1 X129.117 Y118.145 E.03892
G3 X127.853 Y116.453 I77.363 J-59.116 E.08062
G1 X126.662 Y114.847 E.07636
G3 X122.954 Y118.919 I-40.769 J-33.391 E.21036
G3 X123.666 Y119.796 I-6.065 J5.648 E.04315
G1 X123.912 Y120.385 E.02438
G1 X123.965 Y120.832 E.01718
G1 X123.892 Y121.361 E.02038
G1 X123.702 Y121.812 E.0187
G1 X123.565 Y121.99 E.00856
G1 X123.013 Y121.747 F36000
G1 F13446.369
G1 X122.891 Y121.88 E.00687
G1 X118.855 Y125.263 E.20109
G1 X118.511 Y125.474 E.01539
G1 X118.095 Y125.577 E.01636
G3 X117.168 Y125.315 I-.085 J-1.473 E.03745
G1 X116.785 Y124.917 E.02113
G1 X116.03 Y124.017 E.04482
G3 X105.392 Y129.978 I-31.589 J-43.9 E.46656
G1 X104.06 Y130.506 E.05471
G1 X103.688 Y130.653 E.01527
G1 F12255.764
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.665068
G1 F10922.127
G1 X103.236 Y130.854 E.00396
; LINE_WIDTH: 0.71014
G1 F10612.428
G1 X103.156 Y130.907 E.00424
; LINE_WIDTH: 0.755212
G1 F10307.165
G1 X103.075 Y130.96 E.00453
; LINE_WIDTH: 0.800284
G1 F10006.375
G1 X102.995 Y131.013 E.00481
; LINE_WIDTH: 0.845356
G1 F9709.996
G1 X102.914 Y131.066 E.00509
; LINE_WIDTH: 0.890428
G1 F9198.778
G1 X102.834 Y131.119 E.00538
; LINE_WIDTH: 0.9355
G1 F8738.699
G1 X102.754 Y131.172 E.00566
; LINE_WIDTH: 0.980572
G1 F8322.449
G1 X102.673 Y131.225 E.00594
; LINE_WIDTH: 1.02564
G1 F7944.051
G1 X102.593 Y131.278 E.00623
; LINE_WIDTH: 1.07072
G1 F7598.565
G1 X102.512 Y131.331 E.00651
G1 X102.59 Y131.36 E.00558
; LINE_WIDTH: 1.02564
G1 F7944.051
G1 X102.667 Y131.389 E.00534
; LINE_WIDTH: 0.980572
G1 F8322.449
G1 X102.745 Y131.418 E.00509
; LINE_WIDTH: 0.9355
G1 F8738.699
G1 X102.822 Y131.446 E.00485
; LINE_WIDTH: 0.890428
G1 F9198.778
G1 X102.899 Y131.475 E.00461
; LINE_WIDTH: 0.845356
G1 F9709.996
G1 X102.977 Y131.504 E.00437
; LINE_WIDTH: 0.800284
G1 F10281.378
G1 X103.054 Y131.533 E.00412
; LINE_WIDTH: 0.755212
G1 F10542.435
G1 X103.131 Y131.561 E.00388
; LINE_WIDTH: 0.71014
G1 F10806.765
G1 X103.209 Y131.59 E.00364
; LINE_WIDTH: 0.665068
G1 F11074.349
G1 X103.286 Y131.619 E.00339
; LINE_WIDTH: 0.619996
G1 F12416.981
G1 X103.614 Y131.848 E.01527
G1 F13359.097
G1 X103.834 Y132.002 E.01023
G1 F13446.369
G1 X104.023 Y132.287 E.01309
G3 X104.124 Y132.558 I-1.222 J.607 E.01103
G3 X104.166 Y133.697 I-5.976 J.792 E.04358
G1 X104.166 Y141.697 E.30543
G1 X104.099 Y142.117 E.01626
G1 X103.829 Y142.594 E.02094
G1 X103.279 Y142.971 E.02544
; LINE_WIDTH: 0.613396
G1 F13599.629
G1 X103.036 Y143.035 E.00947
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.852 J233.635 E.10355
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.039 J-44.209 E.59948
G3 X136.599 Y119.836 I41.635 J-34.105 E.56111
G3 X135.339 Y120.13 I-2.049 J-5.919 E.04948
G1 X134.412 Y120.179 E.03546
G3 X133.287 Y120.054 I.628 J-10.831 E.04321
G1 X132.568 Y119.865 E.02838
G1 X131.712 Y119.513 E.03535
G1 X130.909 Y119.036 E.03567
G1 X130.185 Y118.446 E.03564
G3 X129.066 Y117.106 I8.53 J-8.255 E.06672
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.881 I-42.453 J-33.971 E.2573
G1 X123.06 Y119.958 E.05365
G1 X123.303 Y120.384 E.01872
G1 X123.379 Y120.833 E.01739
G1 X123.328 Y121.203 E.01426
G1 X123.137 Y121.611 E.0172
G1 X123.073 Y121.68 E.00359
G1 X122.58 Y121.353 F36000
G1 F13446.369
G1 X122.515 Y121.431 E.00385
G1 X118.479 Y124.814 E.20109
G1 X118.226 Y124.956 E.01105
G1 X117.92 Y124.994 E.01176
G1 X117.666 Y124.931 E.01
G1 X117.382 Y124.718 E.01357
G1 X116.128 Y123.222 E.0745
G3 X99.49 Y131.452 I-31.739 J-43.234 E.71227
G3 X99.578 Y132.699 I-14.337 J1.631 E.04776
G1 X102.66 Y132.129 E.11968
G1 X103.092 Y132.172 E.01656
G1 X103.401 Y132.397 E.01462
G1 X103.556 Y132.702 E.01306
G3 X103.58 Y135.697 I-78.985 J2.132 E.11434
G1 X103.58 Y141.697 E.22907
G1 X103.542 Y141.937 E.00928
G1 X103.388 Y142.209 E.01195
G1 X103.075 Y142.426 E.01454
G1 X102.802 Y142.475 E.0106
G1 X102.66 Y142.462 E.00544
G1 X99.58 Y141.892 E.11958
G3 X99.521 Y143.615 I-15.369 J.336 E.06585
G1 X156.235 Y143.615 E2.1653
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.705 Y131.955 I29.338 J-43.886 E.60772
G3 X136.908 Y119.075 I40.723 J-33.454 E.57679
G3 X135.978 Y119.414 I-1.914 J-3.81 E.03788
G1 X135.235 Y119.554 E.02888
G1 X134.39 Y119.593 E.03229
G1 X133.549 Y119.509 E.03228
G1 X132.727 Y119.302 E.03235
G1 X131.942 Y118.975 E.03245
G1 X131.216 Y118.537 E.03239
G1 X130.562 Y117.998 E.03235
G3 X129.081 Y116.142 I12.112 J-11.186 E.09073
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.837 I-42.305 J-32.883 E.30433
G1 X122.612 Y120.334 E.07458
G1 X122.773 Y120.655 E.01371
G1 X122.785 Y120.948 E.01118
G1 X122.688 Y121.225 E.01121
G1 X122.638 Y121.285 E.00299
M204 S250
G1 X122.16 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16676
G1 X117.962 Y124.443 E.00538
G1 X117.806 Y124.363 E.00556
G1 X116.811 Y123.176 E.04901
; LINE_WIDTH: 0.521326
G1 X116.411 Y122.701 E.01971
; LINE_WIDTH: 0.556456
G1 X116.394 Y122.378 E.01104
G1 X115.476 Y123.015 E.03804
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.116 J-43.089 E.58495
G1 X98.941 Y131.547 E.01611
G3 X99.027 Y133.364 I-11.117 J1.435 E.05766
G1 X102.761 Y132.672 E.12023
G1 X102.886 Y132.685 E.00398
G1 X103.004 Y132.794 E.00509
G3 X103.027 Y135.697 I-72.574 J2.04 E.09192
G1 X103.027 Y141.697 E.18996
G1 X102.972 Y141.845 E.00502
G1 X102.802 Y141.922 E.00591
G1 X102.761 Y141.919 E.00131
G1 X99.027 Y141.227 E.1202
G3 X98.998 Y143.312 I-24.705 J.698 E.06602
G2 X98.936 Y144.167 I4.022 J.723 E.02722
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.61 J-43.495 E.5106
G3 X137.192 Y118.316 I40.595 J-33.266 E.49162
G1 X136.732 Y118.548 E.0163
G3 X135.136 Y119.01 I-2.233 J-4.722 E.05284
G1 X134.369 Y119.041 E.02429
G1 X133.609 Y118.959 E.0242
G1 X132.868 Y118.767 E.02426
G1 X132.16 Y118.467 E.02433
G1 X131.506 Y118.066 E.02428
M73 P56 R8
G1 X130.918 Y117.575 E.02426
G1 X130.408 Y117.003 E.02426
G3 X129.087 Y115.222 I551.803 J-411.029 E.07019
G1 X126.704 Y112.01 E.12664
G3 X120.821 Y118.581 I-41.978 J-31.659 E.27956
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.973 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.188 Y120.689 E.06866
G1 X122.24 Y120.851 E.00539
G1 X122.201 Y120.927 E.0027
; WIPE_START
M204 S10000
G1 X121.439 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z3.64 F36000
G1 Z3.24
G1 E.4 F1800
; LINE_WIDTH: 0.556456
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.01278
; LINE_WIDTH: 0.544336
G1 X120.198 Y119.2 E.15231
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.973 E.01162
; WIPE_START
M204 S10000
G1 X120.198 Y119.2 E-.13014
G1 X119.695 Y119.622 E-.24986
; WIPE_END
G1 E-.02 F1800
G1 X113.239 Y123.694 Z3.64 F36000
G1 X100.188 Y131.925 Z3.64
G1 Z3.24
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.788336
G1 F10444.298
G1 X100.504 Y131.847 E.01598
; LINE_WIDTH: 0.827636
G1 F9926.889
G1 X100.819 Y131.768 E.01682
; LINE_WIDTH: 0.866936
G1 F9458.324
G1 X101.135 Y131.69 E.01765
; LINE_WIDTH: 0.871156
G1 F9410.626
G1 X101.166 Y131.682 E.00175
; LINE_WIDTH: 0.917311
G1 F8918.713
G1 X101.48 Y131.6 E.01868
; LINE_WIDTH: 0.963466
G1 F8475.672
G1 X101.794 Y131.519 E.01965
; LINE_WIDTH: 1.00962
G1 F8074.564
G1 X102.108 Y131.437 E.02063
; LINE_WIDTH: 1.05578
G1 F7709.704
G1 X102.422 Y131.356 E.0216
; LINE_WIDTH: 1.07072
G1 F7598.565
G1 X102.512 Y131.331 E.00631
; WIPE_START
G1 X102.422 Y131.356 E-.03551
G1 X102.108 Y131.437 E-.12329
G1 X101.794 Y131.519 E-.12329
G1 X101.545 Y131.584 E-.0979
; WIPE_END
G1 E-.02 F1800
G1 X100.742 Y139.174 Z3.64 F36000
G1 X100.356 Y142.832 Z3.64
G1 Z3.24
G1 E.4 F1800
; LINE_WIDTH: 1.0145
G1 F8034.402
G1 X100.563 Y142.851 E.01327
; LINE_WIDTH: 0.976506
G1 F8358.365
G1 X100.812 Y142.873 E.01536
; LINE_WIDTH: 0.930793
G1 F8784.59
G1 X101.06 Y142.896 E.01461
; LINE_WIDTH: 0.885079
G1 F9256.621
G1 X101.309 Y142.919 E.01386
; LINE_WIDTH: 0.839365
G1 F9782.26
G1 X101.558 Y142.942 E.01312
; LINE_WIDTH: 0.793651
G1 F10371.191
G1 X101.807 Y142.965 E.01238
; LINE_WIDTH: 0.747938
G1 F11035.576
G1 X102.056 Y142.988 E.01163
; LINE_WIDTH: 0.702224
G1 F11790.909
G1 X102.305 Y143.011 E.01088
; LINE_WIDTH: 0.65651
G1 F12657.237
G1 X102.554 Y143.033 E.01014
; LINE_WIDTH: 0.610796
G1 F13660.965
G1 X103.036 Y143.035 E.01812
; WIPE_START
G1 X102.554 Y143.033 E-.18326
G1 X102.305 Y143.011 E-.095
G1 X102.056 Y142.988 E-.095
G1 X102.039 Y142.986 E-.00674
; WIPE_END
G1 E-.02 F1800
G1 X105.249 Y139.734 Z3.64 F36000
G1 Z3.24
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.249 Y137.391 E.08944
G3 X107.257 Y138.117 I-.942 J5.744 E.08199
G3 X109.142 Y139.857 I-6.267 J8.683 E.09817
G2 X110.556 Y140.947 I4.035 J-3.772 E.06846
G2 X114.798 Y140.509 I1.689 J-4.405 E.16893
G2 X116.683 Y138.769 I-6.267 J-8.683 E.09817
G3 X118.097 Y137.679 I4.035 J3.772 E.06846
G3 X122.339 Y138.117 I1.689 J4.405 E.16893
G3 X124.224 Y139.857 I-6.268 J8.684 E.09817
G2 X125.638 Y140.947 I4.035 J-3.772 E.06846
G2 X129.879 Y140.509 I1.689 J-4.405 E.16893
G2 X131.764 Y138.769 I-6.267 J-8.683 E.09817
G3 X133.178 Y137.679 I4.035 J3.772 E.06846
G3 X137.42 Y138.117 I1.689 J4.405 E.16893
G3 X139.305 Y139.857 I-6.267 J8.683 E.09817
G2 X141.19 Y141.128 I3.536 J-3.211 E.08765
G2 X145.903 Y139.744 I1.132 J-4.862 E.19607
G3 X147.884 Y137.895 I11.458 J10.289 E.10358
G3 X144.019 Y133.742 I37.787 J-39.044 E.21669
G3 X141.19 Y132.968 I-.2 J-4.826 E.11376
G3 X139.305 Y131.229 I6.268 J-8.684 E.09817
G2 X137.891 Y130.139 I-4.035 J3.772 E.06846
G2 X133.65 Y130.577 I-1.689 J4.405 E.16893
G2 X131.764 Y132.316 I6.268 J8.684 E.09817
G3 X130.351 Y133.406 I-4.035 J-3.772 E.06846
G3 X126.109 Y132.968 I-1.689 J-4.405 E.16893
G3 X124.224 Y131.229 I6.268 J-8.684 E.09817
G2 X122.81 Y130.139 I-4.035 J3.772 E.06846
G2 X118.568 Y130.577 I-1.689 J4.405 E.16893
G2 X116.683 Y132.316 I6.268 J8.684 E.09817
G3 X115.269 Y133.406 I-4.035 J-3.772 E.06846
G3 X111.028 Y132.968 I-1.689 J-4.405 E.16893
G3 X109.142 Y131.229 I6.268 J-8.684 E.09817
G2 X107.702 Y130.128 I-6.092 J6.484 E.06933
G2 X109.809 Y129.106 I-15.556 J-34.737 E.08945
; WIPE_START
G1 X108.909 Y129.542 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X115.92 Y126.525 Z3.64 F36000
G1 X130.688 Y120.17 Z3.64
G1 Z3.24
G1 E.4 F1800
G1 F13446.283
G3 X128.913 Y118.659 I4.653 J-7.263 E.08923
G3 X125.277 Y117.211 I-.166 J-4.872 E.15377
G3 X123.628 Y118.949 I-51.23 J-46.962 E.09144
G3 X124.25 Y121.833 I-1.887 J1.916 E.11922
G3 X122.774 Y123.39 I-4.337 J-2.632 E.08254
G3 X124.224 Y124.776 I-5.02 J6.701 E.07674
G2 X125.638 Y125.866 I4.034 J-3.771 E.06846
G2 X129.879 Y125.428 I1.689 J-4.405 E.16893
G2 X131.764 Y123.688 I-6.268 J-8.684 E.09817
G3 X136.477 Y122.562 I3.137 J2.705 E.19761
G1 X136.733 Y122.69 E.01091
G2 X137.827 Y124.762 I31.279 J-15.202 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 3.4
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.36 Y123.877 E-.38
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
G1 X123.51 Y122.061
G1 Z3.4
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.391 Y122.216 E.00744
G3 X122.296 Y123.143 I-40.756 J-47.025 E.05478
G1 X119.23 Y125.712 E.15272
G1 X118.874 Y125.95 E.01636
G1 X118.451 Y126.11 E.01727
G1 X117.978 Y126.168 E.01818
G1 X117.576 Y126.126 E.01541
G1 X117.11 Y125.964 E.01885
G1 X116.858 Y125.814 E.0112
G1 X116.484 Y125.471 E.0194
G1 X115.927 Y124.807 E.03309
G3 X103.799 Y131.242 I-31.86 J-45.395 E.52552
G1 X104.048 Y131.39 E.01104
G1 X104.394 Y131.763 E.01944
G1 X104.696 Y132.422 E.02767
G3 X104.754 Y133.699 I-6.385 J.927 E.04889
G1 X104.754 Y141.699 E.30543
G1 X104.659 Y142.301 E.02324
G1 X104.578 Y142.443 E.00627
G1 X154.069 Y142.443 E1.88953
G3 X143.802 Y132.7 I31.972 J-43.972 E.54191
G3 X136.269 Y120.54 I42.196 J-34.555 E.54772
G1 X135.832 Y120.644 E.01715
G3 X133.899 Y120.738 I-1.383 J-8.484 E.07402
G3 X132.492 Y120.452 I1.118 J-9.107 E.05488
G1 X131.484 Y120.053 E.04141
G1 X130.604 Y119.536 E.03893
G1 X129.81 Y118.897 E.03892
G1 X129.117 Y118.145 E.03903
G3 X127.853 Y116.453 I76.926 J-58.791 E.08064
G1 X126.662 Y114.847 E.07636
G3 X122.955 Y118.919 I-42.7 J-35.149 E.21032
G1 X123.51 Y119.581 E.033
G1 X123.732 Y119.909 E.01513
G1 X123.913 Y120.385 E.01944
G1 X123.965 Y120.833 E.01723
G1 X123.892 Y121.363 E.0204
G1 X123.701 Y121.813 E.01869
G1 X123.565 Y121.99 E.00853
G1 X122.989 Y121.761 F36000
G1 F13446.369
G1 X122.78 Y121.973 E.01135
G1 X118.854 Y125.264 E.19559
G1 X118.51 Y125.474 E.01539
G1 X118.095 Y125.577 E.01636
G3 X117.194 Y125.335 I-.06 J-1.572 E.03614
G1 X116.822 Y124.963 E.02009
G1 X116.03 Y124.018 E.04709
G3 X104.465 Y130.355 I-32.054 J-44.774 E.50467
G1 X104.066 Y130.512 E.01639
G1 X103.693 Y130.655 E.01527
G1 F12086.579
G1 X103.319 Y130.799 E.01527
; LINE_WIDTH: 0.669892
G1 F10762.445
G1 X103.23 Y130.858 E.00444
; LINE_WIDTH: 0.719787
G1 F10420.994
G1 X103.141 Y130.917 E.00479
; LINE_WIDTH: 0.769683
G1 F10085.004
G1 X103.051 Y130.976 E.00513
; LINE_WIDTH: 0.819578
G1 F9754.52
G1 X102.962 Y131.035 E.00548
; LINE_WIDTH: 0.869474
G1 F9429.582
G1 X102.872 Y131.094 E.00583
; LINE_WIDTH: 0.91937
G1 F8897.971
G1 X102.783 Y131.153 E.00618
; LINE_WIDTH: 0.969265
G1 F8423.101
G1 X102.693 Y131.212 E.00653
; LINE_WIDTH: 1.01916
G1 F7996.349
G1 X102.604 Y131.271 E.00688
; LINE_WIDTH: 1.06906
G1 F7610.755
G1 X102.515 Y131.33 E.00722
G1 X102.6 Y131.361 E.00617
; LINE_WIDTH: 1.01916
G1 F7996.349
G1 X102.686 Y131.393 E.00588
; LINE_WIDTH: 0.969265
G1 F8423.101
G1 X102.772 Y131.425 E.00558
; LINE_WIDTH: 0.91937
G1 F8897.971
G1 X102.858 Y131.457 E.00528
; LINE_WIDTH: 0.869474
G1 F9429.582
G1 X102.944 Y131.489 E.00498
; LINE_WIDTH: 0.819578
G1 F10028.754
G1 X103.029 Y131.52 E.00469
; LINE_WIDTH: 0.769683
G1 F10314.765
G1 X103.115 Y131.552 E.00439
; LINE_WIDTH: 0.719787
G1 F10604.797
G1 X103.201 Y131.584 E.00409
; LINE_WIDTH: 0.669892
G1 F10898.85
G1 X103.287 Y131.616 E.00379
; LINE_WIDTH: 0.619996
G1 F12395.571
G1 X103.674 Y131.841 E.0171
G1 F13446.369
G1 X103.933 Y132.126 E.01469
G1 X104.134 Y132.589 E.01927
G3 X104.168 Y133.699 I-6.667 J.76 E.04245
G1 X104.168 Y141.699 E.30543
G1 X104.102 Y142.12 E.01626
G1 X103.831 Y142.597 E.02094
G1 X103.281 Y142.973 E.02544
; LINE_WIDTH: 0.611956
G1 F13633.532
G1 X103.038 Y143.036 E.00944
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.851 J191.02 E.10347
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00187
G3 X144.254 Y132.327 I30.04 J-44.21 E.59952
G3 X136.599 Y119.835 I41.226 J-33.854 E.56115
G1 X135.957 Y120.021 E.02551
G1 X135.035 Y120.161 E.03559
G3 X132.65 Y119.888 I-.463 J-6.516 E.0922
G1 X131.714 Y119.514 E.03846
G1 X130.911 Y119.037 E.03565
G1 X130.187 Y118.448 E.03564
G3 X129.066 Y117.106 I8.514 J-8.247 E.06684
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-45.789 J-36.998 E.25724
G1 X123.061 Y119.957 E.05366
G1 X123.303 Y120.384 E.01872
G1 X123.38 Y120.833 E.01742
G1 X123.329 Y121.204 E.01427
G1 X123.137 Y121.611 E.01719
G1 X123.052 Y121.697 E.0046
G1 X122.581 Y121.353 F36000
G1 F13446.369
G1 X122.516 Y121.43 E.00385
G1 X118.478 Y124.815 E.20115
G1 X118.225 Y124.956 E.01106
G1 X117.92 Y124.995 E.01176
G1 X117.631 Y124.915 E.01142
G1 X117.381 Y124.718 E.01216
G1 X116.128 Y123.223 E.0745
G3 X99.49 Y131.452 I-31.738 J-43.234 E.71224
G3 X99.58 Y132.697 I-13.994 J1.631 E.04768
G1 X102.662 Y132.126 E.11968
G1 X103.095 Y132.169 E.01659
G3 X103.448 Y132.455 I-.291 J.722 E.0176
G1 X103.563 Y132.719 E.011
G3 X103.582 Y135.699 I-95.936 J2.115 E.11379
G1 X103.582 Y141.699 E.22907
G1 X103.544 Y141.939 E.00928
G1 X103.39 Y142.212 E.01195
G1 X103.077 Y142.428 E.01454
G1 X102.804 Y142.478 E.0106
G1 X102.662 Y142.465 E.00544
G1 X99.582 Y141.894 E.11958
G3 X99.522 Y143.615 I-15.033 J.334 E.06575
G1 X156.235 Y143.615 E2.16526
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.705 Y131.954 I29.338 J-43.886 E.60775
G3 X136.908 Y119.075 I40.817 J-33.51 E.57675
G1 X136.498 Y119.254 E.01709
G1 X135.794 Y119.458 E.02799
G1 X134.956 Y119.581 E.03231
G1 X134.438 Y119.583 E.01978
G3 X132.807 Y119.324 I.108 J-5.95 E.06326
G1 X131.945 Y118.976 E.03552
G1 X131.218 Y118.539 E.03237
G1 X130.564 Y118 E.03235
G3 X129.081 Y116.142 I12.081 J-11.166 E.09083
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-44.496 J-34.865 E.30427
G1 X122.612 Y120.334 E.07459
G1 X122.773 Y120.655 E.01371
G1 X122.786 Y120.948 E.01121
G1 X122.689 Y121.225 E.0112
G1 X122.639 Y121.284 E.00297
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00555
G1 X116.81 Y123.176 E.04905
; LINE_WIDTH: 0.521326
G1 X116.411 Y122.702 E.01966
; LINE_WIDTH: 0.556496
G1 X116.394 Y122.378 E.01104
G1 X115.477 Y123.015 E.038
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.117 J-43.088 E.58496
G1 X98.941 Y131.546 E.0161
G3 X99.029 Y133.361 I-10.857 J1.434 E.0576
G1 X102.763 Y132.67 E.12023
G1 X102.948 Y132.718 E.00605
G1 X103.024 Y132.842 E.0046
G3 X103.03 Y135.699 I-285.882 J1.992 E.09047
G1 X103.03 Y141.699 E.18996
G1 X102.974 Y141.848 E.00502
G1 X102.804 Y141.925 E.00591
G1 X102.763 Y141.921 E.00131
G1 X99.029 Y141.23 E.1202
G3 X99 Y143.31 I-24.171 J.697 E.0659
G2 X98.936 Y144.167 I3.943 J.725 E.02726
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.597 I28.641 J-43.529 E.51061
G3 X137.192 Y118.316 I40.595 J-33.265 E.49162
G3 X136.344 Y118.723 I-5.234 J-9.822 E.0298
G3 X132.956 Y118.792 I-1.795 J-4.973 E.10921
G1 X132.162 Y118.468 E.02715
G1 X131.508 Y118.068 E.02427
G1 X130.92 Y117.577 E.02426
G1 X130.408 Y117.003 E.02434
G3 X129.087 Y115.222 I547.278 J-407.676 E.07019
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-41.979 J-31.66 E.27953
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.851 E.00539
G1 X122.202 Y120.927 E.0027
; WIPE_START
M204 S10000
G1 X121.44 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z3.8 F36000
G1 Z3.4
G1 E.4 F1800
; LINE_WIDTH: 0.556496
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.01281
; LINE_WIDTH: 0.544336
G1 X120.198 Y119.2 E.15231
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.198 Y119.2 E-.13045
G1 X119.695 Y119.622 E-.24955
; WIPE_END
G1 E-.02 F1800
G1 X113.239 Y123.693 Z3.8 F36000
G1 X100.188 Y131.924 Z3.8
G1 Z3.4
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.78627
G1 F10473.003
G1 X100.504 Y131.846 E.01593
; LINE_WIDTH: 0.825543
G1 F9953.153
G1 X100.819 Y131.767 E.01677
; LINE_WIDTH: 0.864816
G1 F9482.469
G1 X101.135 Y131.689 E.0176
; LINE_WIDTH: 0.869036
G1 F9434.528
G1 X101.166 Y131.681 E.00174
; LINE_WIDTH: 0.915186
G1 F8940.229
G1 X101.48 Y131.599 E.01863
; LINE_WIDTH: 0.961336
G1 F8495.146
G1 X101.794 Y131.518 E.0196
; LINE_WIDTH: 1.00749
G1 F8092.278
G1 X102.108 Y131.436 E.02058
; LINE_WIDTH: 1.05364
G1 F7725.891
G1 X102.421 Y131.355 E.02156
; LINE_WIDTH: 1.06906
G1 F7610.755
G1 X102.515 Y131.33 E.0065
; WIPE_START
G1 X102.421 Y131.355 E-.03662
G1 X102.108 Y131.436 E-.12327
G1 X101.794 Y131.518 E-.12327
G1 X101.547 Y131.582 E-.09684
; WIPE_END
G1 E-.02 F1800
G1 X102.532 Y139.15 Z3.8 F36000
G1 X103.038 Y143.036 Z3.8
G1 Z3.4
G1 E.4 F1800
; LINE_WIDTH: 0.608256
G1 F13721.425
G1 X102.556 Y143.035 E.01803
; LINE_WIDTH: 0.653973
G1 F12709.07
G1 X102.308 Y143.012 E.0101
; LINE_WIDTH: 0.699689
G1 F11835.832
G1 X102.059 Y142.989 E.01084
; LINE_WIDTH: 0.745405
G1 F11074.879
G1 X101.81 Y142.966 E.01159
; LINE_WIDTH: 0.791121
G1 F10405.862
G1 X101.561 Y142.943 E.01233
; LINE_WIDTH: 0.836837
G1 F9813.069
G1 X101.312 Y142.92 E.01308
; LINE_WIDTH: 0.882554
G1 F9284.176
G1 X101.063 Y142.898 E.01382
; LINE_WIDTH: 0.92827
G1 F8809.378
G1 X100.814 Y142.875 E.01457
; LINE_WIDTH: 0.973986
G1 F8380.78
G1 X100.565 Y142.852 E.01531
; LINE_WIDTH: 1.01226
G1 F8052.806
G1 X100.357 Y142.833 E.01334
; WIPE_START
G1 X100.565 Y142.852 E-.07951
G1 X100.814 Y142.875 E-.095
G1 X101.063 Y142.898 E-.095
G1 X101.312 Y142.92 E-.095
G1 X101.352 Y142.924 E-.01549
; WIPE_END
G1 E-.02 F1800
G1 X105.251 Y139.755 Z3.8 F36000
G1 Z3.4
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.251 Y137.412 E.08944
G3 X108.2 Y138.749 I-.387 J4.774 E.12608
G2 X110.556 Y140.814 I9.728 J-8.725 E.11991
G2 X115.741 Y139.877 I1.899 J-4.308 E.21345
G3 X118.097 Y137.813 I9.728 J8.724 E.11991
G3 X123.281 Y138.749 I1.899 J4.308 E.21345
G2 X125.638 Y140.814 I9.729 J-8.725 E.11991
G2 X130.822 Y139.877 I1.899 J-4.308 E.21345
G3 X133.178 Y137.813 I9.728 J8.724 E.11991
G3 X138.362 Y138.749 I1.899 J4.308 E.21345
G2 X140.719 Y140.814 I9.728 J-8.725 E.11991
G2 X145.903 Y139.877 I1.899 J-4.308 E.21345
G3 X147.788 Y138.104 I16.004 J15.121 E.09888
G1 X147.979 Y137.986 E.00855
G3 X143.968 Y133.681 I37.6 J-39.056 E.22477
G3 X140.248 Y132.337 I-.366 J-4.807 E.15561
G2 X137.891 Y130.272 I-9.728 J8.724 E.11991
G2 X132.707 Y131.208 I-1.899 J4.308 E.21345
G3 X130.351 Y133.273 I-9.728 J-8.725 E.11991
G3 X125.166 Y132.337 I-1.899 J-4.308 E.21345
G2 X122.81 Y130.272 I-9.727 J8.724 E.11991
G2 X117.626 Y131.208 I-1.899 J4.308 E.21345
G3 X115.741 Y132.982 I-16.003 J-15.119 E.09888
G3 X111.028 Y133.048 I-2.41 J-3.735 E.18967
G3 X109.142 Y131.398 I5.049 J-7.667 E.09597
G2 X107.566 Y130.201 I-3.798 J3.367 E.07604
G2 X109.669 Y129.169 I-22.233 J-47.97 E.08943
; WIPE_START
G1 X108.771 Y129.61 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X115.773 Y126.571 Z3.8 F36000
G1 X130.62 Y120.128 Z3.8
G1 Z3.4
G1 E.4 F1800
G1 F13446.283
G3 X128.861 Y118.599 I4.746 J-7.239 E.08923
G3 X125.209 Y117.287 I-.361 J-4.736 E.15264
G3 X123.629 Y118.948 I-49.551 J-45.565 E.08751
G3 X124.25 Y121.834 I-1.887 J1.916 E.11932
G3 X122.84 Y123.335 I-4.23 J-2.561 E.07918
G1 X123.281 Y123.668 E.02109
G2 X125.638 Y125.732 I9.728 J-8.725 E.11991
G2 X130.822 Y124.796 I1.899 J-4.308 E.21345
G3 X132.707 Y123.022 I16.003 J15.119 E.09888
G3 X136.477 Y122.532 I2.399 J3.702 E.15008
G1 X136.705 Y122.634 E.00952
G2 X137.794 Y124.707 I70.628 J-35.798 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 3.56
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.329 Y123.822 E-.38
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
G1 X123.508 Y122.064
G1 Z3.56
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.385 Y122.221 E.0076
G3 X122.296 Y123.143 I-44.047 J-50.963 E.05448
G1 X119.23 Y125.712 E.15272
G1 X118.874 Y125.95 E.01636
G1 X118.45 Y126.11 E.01728
G1 X117.978 Y126.168 E.01818
G1 X117.493 Y126.107 E.01866
G1 X117.196 Y126.004 E.01201
G1 X116.82 Y125.787 E.01657
G3 X115.927 Y124.807 I5.234 J-5.665 E.05069
G3 X103.759 Y131.258 I-31.536 J-44.781 E.52722
G1 X104.393 Y131.757 E.03083
G1 X104.698 Y132.419 E.02782
G3 X104.756 Y133.702 I-6.395 J.93 E.0491
G1 X104.756 Y141.702 E.30543
G1 X104.661 Y142.303 E.02324
G1 X104.581 Y142.443 E.00616
G1 X154.069 Y142.443 E1.88942
G3 X143.803 Y132.702 I31.776 J-43.768 E.54189
G3 X136.268 Y120.538 I41.709 J-34.256 E.54789
G3 X133.124 Y120.613 I-1.735 J-6.757 E.12112
G1 X132.097 Y120.319 E.04077
G1 X131.179 Y119.893 E.03862
G1 X130.487 Y119.454 E.03132
G1 X129.802 Y118.89 E.03387
G1 X129.119 Y118.148 E.03851
G3 X127.853 Y116.453 I77.023 J-58.869 E.08075
G1 X126.662 Y114.847 E.07636
G3 X122.955 Y118.919 I-42.385 J-34.862 E.21032
G3 X123.665 Y119.794 I-6.104 J5.681 E.04308
G1 X123.913 Y120.386 E.0245
G1 X123.965 Y120.834 E.01722
G1 X123.902 Y121.329 E.01904
G1 X123.698 Y121.819 E.02029
G1 X123.563 Y121.993 E.0084
G1 X123.001 Y121.751 F36000
G1 F13446.369
G1 X122.831 Y121.93 E.00941
M73 P57 R8
G1 X118.854 Y125.264 E.19814
G1 X118.51 Y125.474 E.01539
G1 X118.094 Y125.577 E.01636
G1 X117.639 Y125.539 E.01746
G1 X117.188 Y125.33 E.01896
G1 X116.82 Y124.961 E.01991
G1 X116.03 Y124.018 E.04698
G3 X104.462 Y130.359 I-31.605 J-43.931 E.5049
G1 X104.066 Y130.512 E.01622
G1 X103.693 Y130.656 E.01527
G1 F12094.053
G1 X103.319 Y130.799 E.01527
; LINE_WIDTH: 0.669707
G1 F10769.498
G1 X103.23 Y130.858 E.00442
; LINE_WIDTH: 0.719418
G1 F10428.925
G1 X103.141 Y130.917 E.00477
; LINE_WIDTH: 0.76913
G1 F10093.808
G1 X103.052 Y130.976 E.00511
; LINE_WIDTH: 0.818841
G1 F9764.163
G1 X102.963 Y131.034 E.00546
; LINE_WIDTH: 0.868552
G1 F9440.007
G1 X102.874 Y131.093 E.00581
; LINE_WIDTH: 0.918263
G1 F8909.111
G1 X102.784 Y131.152 E.00615
; LINE_WIDTH: 0.967974
G1 F8434.749
G1 X102.695 Y131.21 E.0065
; LINE_WIDTH: 1.01769
G1 F8008.348
G1 X102.606 Y131.269 E.00684
; LINE_WIDTH: 1.0674
G1 F7622.984
G1 X102.517 Y131.328 E.00719
G1 X102.602 Y131.359 E.00615
; LINE_WIDTH: 1.01769
G1 F8008.348
G1 X102.688 Y131.391 E.00585
; LINE_WIDTH: 0.967974
G1 F8434.749
G1 X102.774 Y131.423 E.00556
; LINE_WIDTH: 0.918263
G1 F8909.111
G1 X102.859 Y131.454 E.00526
; LINE_WIDTH: 0.868552
G1 F9440.007
G1 X102.945 Y131.486 E.00497
; LINE_WIDTH: 0.818841
G1 F10038.184
G1 X103.031 Y131.518 E.00467
; LINE_WIDTH: 0.76913
G1 F10323.593
G1 X103.116 Y131.549 E.00437
; LINE_WIDTH: 0.719418
G1 F10613.002
G1 X103.202 Y131.581 E.00408
; LINE_WIDTH: 0.669707
G1 F10906.393
G1 X103.287 Y131.613 E.00378
; LINE_WIDTH: 0.619996
G1 F12239.096
G1 X103.604 Y131.857 E.01527
G1 F13446.369
G1 X103.917 Y132.097 E.01504
G1 X104.13 Y132.56 E.01947
G3 X104.17 Y133.702 I-6.149 J.788 E.04367
G1 X104.17 Y141.702 E.30543
G1 X104.104 Y142.122 E.01626
G1 X103.833 Y142.599 E.02094
G1 X103.283 Y142.975 E.02543
; LINE_WIDTH: 0.610496
G1 F13668.079
G1 X103.04 Y143.037 E.00941
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.849 J161.669 E.1034
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.046 J-44.216 E.59948
G3 X136.599 Y119.836 I41.615 J-34.093 E.56113
G1 X135.676 Y120.077 E.03642
G3 X133.236 Y120.039 I-1.117 J-6.611 E.09369
G1 X132.277 Y119.762 E.03813
G1 X131.436 Y119.367 E.03546
G1 X130.811 Y118.966 E.02836
G1 X130.182 Y118.444 E.03123
G3 X129.066 Y117.105 I8.599 J-8.301 E.06657
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-42.687 J-34.183 E.25726
G1 X123.061 Y119.957 E.05366
G1 X123.303 Y120.384 E.01872
G1 X123.38 Y120.834 E.01744
G1 X123.311 Y121.261 E.0165
G1 X123.148 Y121.595 E.0142
G1 X123.062 Y121.685 E.00475
G1 X122.567 Y121.359 F36000
G1 F13446.369
G1 X122.455 Y121.481 E.0063
G1 X118.478 Y124.815 E.19814
G1 X118.225 Y124.956 E.01106
G1 X117.92 Y124.994 E.01176
G1 X117.665 Y124.931 E.01
G1 X117.381 Y124.718 E.01356
G1 X116.128 Y123.223 E.0745
G3 X99.517 Y131.444 I-31.739 J-43.235 E.71116
G1 X99.49 Y131.484 E.00185
G1 X99.579 Y132.486 E.03838
G1 X99.582 Y132.694 E.00798
G1 X102.664 Y132.124 E.11969
G1 X103.096 Y132.167 E.01656
G1 X103.449 Y132.451 E.01731
G1 X103.565 Y132.716 E.01106
G3 X103.584 Y135.702 I-95.84 J2.117 E.11399
G1 X103.584 Y141.702 E.22907
G1 X103.546 Y141.942 E.00928
G1 X103.392 Y142.214 E.01195
G1 X103.079 Y142.431 E.01454
G1 X102.806 Y142.48 E.0106
G1 X102.664 Y142.467 E.00544
G1 X99.584 Y141.897 E.11958
G3 X99.523 Y143.615 I-14.703 J.333 E.06566
G1 X156.235 Y143.615 E2.16523
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.955 I29.319 J-43.865 E.60772
G3 X136.91 Y119.08 I41.169 J-33.724 E.57655
G1 X136.339 Y119.309 E.02349
G1 X135.541 Y119.507 E.03141
G3 X134.56 Y119.595 I-1.464 J-10.765 E.0376
G3 X132.457 Y119.204 I.09 J-6.337 E.08208
G1 X131.693 Y118.841 E.0323
G1 X131.135 Y118.478 E.0254
G1 X130.561 Y117.997 E.02859
G3 X129.081 Y116.142 I12.164 J-11.223 E.09068
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-42.464 J-33.027 E.3043
G1 X122.612 Y120.333 E.07459
G1 X122.773 Y120.655 E.01373
G1 X122.786 Y120.944 E.01103
G1 X122.687 Y121.227 E.01147
G1 X122.627 Y121.293 E.00339
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00555
G1 X116.81 Y123.176 E.04905
; LINE_WIDTH: 0.521316
G1 X116.411 Y122.702 E.01966
; LINE_WIDTH: 0.556516
G1 X116.394 Y122.378 E.01104
G1 X115.477 Y123.015 E.038
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.117 J-43.088 E.58496
G1 X98.941 Y131.546 E.01608
G3 X99.031 Y133.359 I-10.61 J1.434 E.05754
G1 X102.765 Y132.667 E.12023
G1 X102.89 Y132.68 E.00398
G1 X103.025 Y132.835 E.00651
G3 X103.032 Y135.702 I-243.662 J1.999 E.09077
G1 X103.032 Y141.702 E.18996
G1 X102.976 Y141.85 E.00502
G1 X102.806 Y141.927 E.00591
G1 X102.765 Y141.924 E.00131
G1 X99.032 Y141.232 E.1202
G3 X99.001 Y143.309 I-23.699 J.695 E.06578
G2 X98.936 Y144.167 I3.864 J.727 E.0273
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.741 J-43.639 E.51058
G3 X137.19 Y118.31 I40.309 J-33.095 E.49187
G3 X136.134 Y118.795 I-2.583 J-4.226 E.03689
G1 X135.413 Y118.969 E.02347
G1 X134.813 Y119.027 E.01907
G1 X133.859 Y118.999 E.03021
G3 X132.626 Y118.678 I.719 J-5.299 E.04043
G3 X130.919 Y117.576 I2.197 J-5.277 E.06468
G1 X130.41 Y117.005 E.02423
G3 X129.087 Y115.222 I544.95 J-405.958 E.07027
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-42.101 J-31.769 E.27953
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.85 E.00537
G1 X122.202 Y120.926 E.00271
; WIPE_START
M204 S10000
G1 X121.439 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z3.96 F36000
G1 Z3.56
G1 E.4 F1800
; LINE_WIDTH: 0.556516
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.01281
; LINE_WIDTH: 0.544336
G1 X120.199 Y119.2 E.15232
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.199 Y119.2 E-.13038
G1 X119.695 Y119.622 E-.24962
; WIPE_END
G1 E-.02 F1800
G1 X113.239 Y123.693 Z3.96 F36000
G1 X100.188 Y131.923 Z3.96
G1 Z3.56
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.78421
G1 F10501.774
G1 X100.504 Y131.845 E.01588
; LINE_WIDTH: 0.823463
G1 F9979.388
G1 X100.819 Y131.766 E.01671
; LINE_WIDTH: 0.862716
G1 F9506.507
G1 X101.134 Y131.688 E.01755
; LINE_WIDTH: 0.866956
G1 F9458.097
G1 X101.165 Y131.68 E.00174
; LINE_WIDTH: 0.913106
G1 F8961.39
G1 X101.479 Y131.598 E.01858
; LINE_WIDTH: 0.959256
G1 F8514.251
G1 X101.793 Y131.517 E.01956
; LINE_WIDTH: 1.00541
G1 F8109.612
G1 X102.107 Y131.435 E.02054
; LINE_WIDTH: 1.05156
G1 F7741.688
G1 X102.421 Y131.354 E.02151
; LINE_WIDTH: 1.0674
G1 F7622.984
G1 X102.517 Y131.328 E.00667
; WIPE_START
G1 X102.421 Y131.354 E-.03763
G1 X102.107 Y131.435 E-.12327
G1 X101.793 Y131.517 E-.12327
G1 X101.549 Y131.58 E-.09582
; WIPE_END
G1 E-.02 F1800
G1 X100.745 Y139.17 Z3.96 F36000
G1 X100.357 Y142.834 Z3.96
G1 Z3.56
G1 E.4 F1800
; LINE_WIDTH: 1.01002
G1 F8071.294
G1 X100.567 Y142.853 E.01341
; LINE_WIDTH: 0.971466
G1 F8403.317
G1 X100.816 Y142.876 E.01527
; LINE_WIDTH: 0.92575
G1 F8834.282
G1 X101.065 Y142.899 E.01453
; LINE_WIDTH: 0.880034
G1 F9311.841
G1 X101.314 Y142.922 E.01378
; LINE_WIDTH: 0.834318
G1 F9843.981
G1 X101.563 Y142.945 E.01304
; LINE_WIDTH: 0.788601
G1 F10440.628
G1 X101.812 Y142.967 E.01229
; LINE_WIDTH: 0.742885
G1 F11114.268
G1 X102.061 Y142.99 E.01155
; LINE_WIDTH: 0.697169
G1 F11880.83
G1 X102.31 Y143.013 E.0108
; LINE_WIDTH: 0.651452
G1 F12760.968
G1 X102.559 Y143.036 E.01006
; LINE_WIDTH: 0.605736
G1 F13781.941
G1 X103.04 Y143.037 E.01793
; WIPE_START
G1 X102.559 Y143.036 E-.18293
G1 X102.31 Y143.013 E-.095
G1 X102.061 Y142.99 E-.095
G1 X102.042 Y142.989 E-.00708
; WIPE_END
G1 E-.02 F1800
G1 X105.254 Y139.771 Z3.96 F36000
G1 Z3.56
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.254 Y137.428 E.08944
G3 X108.2 Y138.615 I-.124 J4.559 E.12387
G2 X110.085 Y140.376 I27.62 J-27.682 E.0985
G2 X115.741 Y140.011 I2.6 J-3.714 E.23373
G3 X117.626 Y138.251 I27.62 J27.682 E.0985
G3 X123.281 Y138.615 I2.6 J3.714 E.23373
G2 X125.166 Y140.376 I27.624 J-27.687 E.0985
G2 X130.822 Y140.011 I2.6 J-3.714 E.23373
G3 X132.707 Y138.251 I27.62 J27.682 E.0985
G3 X138.362 Y138.615 I2.6 J3.714 E.23373
G2 X140.248 Y140.376 I27.62 J-27.682 E.0985
G2 X145.903 Y140.011 I2.6 J-3.714 E.23373
G3 X147.788 Y138.251 I27.62 J27.682 E.0985
G1 X148.077 Y138.085 E.0127
G3 X143.926 Y133.63 I38.118 J-39.671 E.23259
G3 X140.248 Y132.471 I-.583 J-4.567 E.15194
G2 X138.362 Y130.71 I-27.62 J27.682 E.0985
G2 X132.707 Y131.074 I-2.6 J3.714 E.23373
G3 X130.822 Y132.835 I-27.615 J-27.678 E.0985
G3 X125.166 Y132.471 I-2.6 J-3.714 E.23373
G2 X123.281 Y130.71 I-27.615 J27.678 E.0985
G2 X117.626 Y131.074 I-2.6 J3.714 E.23373
G3 X115.741 Y132.835 I-27.615 J-27.678 E.0985
G3 X111.97 Y133.508 I-2.588 J-3.599 E.1512
G3 X109.142 Y131.561 I2.18 J-6.194 E.13261
G2 X107.418 Y130.262 I-3.859 J3.326 E.08307
G2 X109.53 Y129.247 I-15.442 J-34.859 E.08945
; WIPE_START
G1 X108.628 Y129.68 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X115.622 Y126.624 Z3.96 F36000
G1 X130.562 Y120.096 Z3.96
G1 Z3.56
G1 E.4 F1800
G1 F13446.283
G3 X128.818 Y118.55 I4.688 J-7.046 E.08928
G3 X125.166 Y117.39 I-.561 J-4.56 E.15089
G1 X125.139 Y117.363 E.00145
G3 X123.628 Y118.947 I-26.92 J-24.183 E.08358
G3 X124.212 Y121.916 I-1.758 J1.888 E.12346
G3 X122.912 Y123.275 I-3.926 J-2.454 E.07229
G3 X125.166 Y125.294 I-23.147 J28.101 E.11558
G2 X130.822 Y124.93 I2.6 J-3.714 E.23373
G3 X132.707 Y123.169 I27.62 J27.682 E.0985
G3 X136.477 Y122.496 I2.588 J3.599 E.1512
G1 X136.677 Y122.576 E.00821
G2 X137.761 Y124.653 I104.734 J-53.321 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 3.72
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.298 Y123.766 E-.38
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
G1 X123.51 Y122.061
G1 Z3.72
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.391 Y122.215 E.00743
G3 X122.296 Y123.143 I-51.089 J-59.226 E.0548
G1 X119.23 Y125.712 E.15272
G1 X118.874 Y125.95 E.01636
G1 X118.451 Y126.11 E.01727
G3 X117.196 Y126.004 I-.454 J-2.116 E.04879
G1 X116.82 Y125.787 E.01657
G3 X115.927 Y124.807 I5.236 J-5.666 E.05069
G3 X103.762 Y131.257 I-31.446 J-44.611 E.52709
G1 X104.393 Y131.751 E.03059
G1 X104.7 Y132.416 E.02798
G3 X104.758 Y133.704 I-6.404 J.932 E.04932
G1 X104.758 Y141.704 E.30543
G1 X104.663 Y142.306 E.02324
G1 X104.585 Y142.443 E.00605
G1 X154.069 Y142.443 E1.88926
G3 X143.803 Y132.701 I31.812 J-43.803 E.5419
G3 X136.269 Y120.54 I41.827 J-34.326 E.54777
G3 X133.351 Y120.664 I-1.761 J-7.048 E.11228
G1 X132.426 Y120.433 E.03639
G1 X131.483 Y120.052 E.03882
G1 X130.602 Y119.535 E.03902
G1 X129.81 Y118.896 E.03884
G1 X129.119 Y118.147 E.0389
G3 X127.854 Y116.454 I77.108 J-58.931 E.08069
G1 X126.662 Y114.848 E.07636
G3 X122.955 Y118.919 I-42.942 J-35.387 E.21031
G3 X123.665 Y119.794 I-6.105 J5.681 E.04308
G1 X123.913 Y120.386 E.0245
G1 X123.965 Y120.834 E.01721
G1 X123.893 Y121.361 E.02034
G1 X123.702 Y121.813 E.01872
G1 X123.565 Y121.99 E.00853
G1 X123.013 Y121.746 F36000
G1 F13446.369
G1 X122.892 Y121.879 E.00686
G1 X118.854 Y125.264 E.20115
G1 X118.51 Y125.474 E.01539
G1 X118.095 Y125.577 E.01636
G3 X117.168 Y125.316 I-.085 J-1.473 E.03746
G1 X116.783 Y124.917 E.02115
G1 X116.03 Y124.018 E.04479
G3 X104.457 Y130.358 I-32.054 J-44.774 E.50502
G1 X104.066 Y130.512 E.01604
G1 X103.693 Y130.656 E.01527
G1 F12101.626
G1 X103.319 Y130.799 E.01527
; LINE_WIDTH: 0.669523
G1 F10776.644
G1 X103.23 Y130.858 E.00441
; LINE_WIDTH: 0.71905
G1 F10436.949
G1 X103.142 Y130.916 E.00475
; LINE_WIDTH: 0.768576
G1 F10102.677
G1 X103.053 Y130.975 E.0051
; LINE_WIDTH: 0.818103
G1 F9773.846
G1 X102.964 Y131.034 E.00544
; LINE_WIDTH: 0.86763
G1 F9450.455
G1 X102.875 Y131.092 E.00578
; LINE_WIDTH: 0.917156
G1 F8920.279
G1 X102.786 Y131.151 E.00613
; LINE_WIDTH: 0.966683
G1 F8446.43
G1 X102.697 Y131.209 E.00647
; LINE_WIDTH: 1.01621
G1 F8020.383
G1 X102.608 Y131.268 E.00681
; LINE_WIDTH: 1.06574
G1 F7635.253
G1 X102.519 Y131.326 E.00716
G1 X102.604 Y131.358 E.00612
; LINE_WIDTH: 1.01621
G1 F8020.383
G1 X102.69 Y131.389 E.00583
; LINE_WIDTH: 0.966683
G1 F8446.43
G1 X102.775 Y131.421 E.00553
; LINE_WIDTH: 0.917156
G1 F8920.279
G1 X102.861 Y131.452 E.00524
; LINE_WIDTH: 0.86763
G1 F9450.455
G1 X102.946 Y131.484 E.00495
; LINE_WIDTH: 0.818103
G1 F10047.634
G1 X103.032 Y131.515 E.00465
; LINE_WIDTH: 0.768576
G1 F10332.44
G1 X103.117 Y131.547 E.00436
; LINE_WIDTH: 0.71905
G1 F10621.237
G1 X103.203 Y131.578 E.00406
; LINE_WIDTH: 0.669523
G1 F10913.984
G1 X103.288 Y131.61 E.00377
; LINE_WIDTH: 0.619996
G1 F12247.138
G1 X103.605 Y131.853 E.01527
G1 F13446.369
G1 X103.917 Y132.092 E.01498
G1 X104.132 Y132.557 E.01957
G3 X104.172 Y133.704 I-6.156 J.791 E.04388
G1 X104.172 Y141.704 E.30543
G1 X104.106 Y142.125 E.01626
G1 X103.835 Y142.602 E.02094
G1 X103.285 Y142.977 E.02542
; LINE_WIDTH: 0.609036
G1 F13702.802
G1 X103.042 Y143.038 E.00938
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.848 J140.075 E.10333
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I29.803 J-43.952 E.59953
G3 X136.599 Y119.835 I41.64 J-34.108 E.56114
G3 X133.436 Y120.085 I-2.088 J-6.275 E.12232
G1 X132.575 Y119.867 E.0339
G1 X131.714 Y119.514 E.03553
G1 X130.909 Y119.036 E.03574
G1 X130.187 Y118.448 E.03556
G3 X129.066 Y117.105 I8.519 J-8.25 E.06682
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-42.712 J-34.206 E.25727
G1 X123.061 Y119.957 E.05366
G1 X123.303 Y120.384 E.01872
G1 X123.38 Y120.834 E.01743
G1 X123.329 Y121.203 E.01423
G1 X123.137 Y121.611 E.01721
G1 X123.074 Y121.68 E.00358
G1 X122.581 Y121.353 F36000
G1 F13446.369
G1 X122.516 Y121.43 E.00385
G1 X118.478 Y124.815 E.20115
G1 X118.225 Y124.956 E.01105
G1 X117.92 Y124.995 E.01176
G1 X117.666 Y124.931 E.01
G1 X117.381 Y124.718 E.01356
G1 X116.128 Y123.223 E.0745
G3 X99.516 Y131.444 I-31.739 J-43.236 E.71121
G1 X99.49 Y131.483 E.00177
G1 X99.581 Y132.484 E.03838
G1 X99.584 Y132.692 E.00794
G1 X102.666 Y132.121 E.11969
G1 X103.097 Y132.164 E.01652
G1 X103.45 Y132.446 E.01727
G1 X103.567 Y132.713 E.01113
G3 X103.586 Y135.704 I-95.703 J2.119 E.11419
G1 X103.586 Y141.704 E.22907
G1 X103.549 Y141.944 E.00928
G1 X103.394 Y142.217 E.01195
G1 X103.081 Y142.433 E.01454
G1 X102.808 Y142.483 E.0106
G1 X102.666 Y142.47 E.00544
G1 X99.586 Y141.899 E.11958
G3 X99.524 Y143.615 I-14.391 J.332 E.06557
G1 X156.235 Y143.615 E2.1652
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.705 Y131.954 I29.326 J-43.872 E.60774
G3 X137.039 Y119.379 I41.152 J-33.709 E.56409
G1 X136.908 Y119.075 E.01263
G1 X136.5 Y119.253 E.01699
G1 X135.796 Y119.458 E.028
G1 X134.957 Y119.581 E.03237
G1 X134.386 Y119.589 E.0218
G1 X133.544 Y119.508 E.0323
G1 X132.724 Y119.301 E.0323
G1 X131.944 Y118.976 E.03225
G1 X131.216 Y118.537 E.03246
G1 X130.564 Y118 E.03227
G1 X129.996 Y117.371 E.03234
G3 X129.081 Y116.142 I119.347 J-89.839 E.05849
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-42.476 J-33.038 E.3043
G1 X122.612 Y120.333 E.07459
G1 X122.773 Y120.655 E.01372
G1 X122.786 Y120.948 E.01119
G1 X122.689 Y121.225 E.0112
G1 X122.639 Y121.284 E.00298
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00555
G1 X116.81 Y123.176 E.04903
; LINE_WIDTH: 0.521316
G1 X116.411 Y122.702 E.01968
; LINE_WIDTH: 0.556496
G1 X116.394 Y122.378 E.01104
G1 X115.476 Y123.015 E.03802
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.116 J-43.088 E.58495
G1 X98.941 Y131.546 E.01608
G3 X99.033 Y133.356 I-10.38 J1.433 E.05747
G1 X102.767 Y132.665 E.12023
G1 X102.892 Y132.677 E.00397
G1 X103.027 Y132.832 E.00651
G3 X103.034 Y135.704 I-243.162 J2.001 E.09093
G1 X103.034 Y141.704 E.18996
G1 X102.978 Y141.853 E.00502
G1 X102.808 Y141.93 E.00591
G1 X102.767 Y141.926 E.00131
G1 X99.034 Y141.235 E.1202
G3 X99.003 Y143.308 I-23.2 J.694 E.06567
G2 X98.936 Y144.167 I3.793 J.729 E.02735
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.617 J-43.502 E.51062
G3 X137.192 Y118.316 I40.346 J-33.116 E.49163
G3 X136.346 Y118.722 I-5.223 J-9.796 E.02973
M73 P58 R8
G1 X135.642 Y118.927 E.02322
G3 X133.608 Y118.959 I-1.106 J-5.614 E.06474
G1 X132.865 Y118.766 E.02431
G1 X132.162 Y118.468 E.02417
G1 X131.506 Y118.066 E.02435
G1 X130.92 Y117.577 E.02419
G1 X130.41 Y117.004 E.02426
G3 X129.087 Y115.222 I547.826 J-408.087 E.07027
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-42.109 J-31.776 E.27953
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.851 E.00539
G1 X122.202 Y120.927 E.0027
; WIPE_START
M204 S10000
G1 X121.44 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z4.12 F36000
G1 Z3.72
G1 E.4 F1800
; LINE_WIDTH: 0.556496
G1 F3600
M204 S5000
G1 X116.687 Y122.143 E.0128
; LINE_WIDTH: 0.544336
G1 X120.199 Y119.2 E.15232
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.199 Y119.2 E-.13038
G1 X119.695 Y119.622 E-.24962
; WIPE_END
G1 E-.02 F1800
G1 X113.239 Y123.693 Z4.12 F36000
G1 X100.188 Y131.922 Z4.12
G1 Z3.72
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.782156
G1 F10530.61
G1 X100.504 Y131.844 E.01583
; LINE_WIDTH: 0.821396
G1 F10005.591
G1 X100.819 Y131.765 E.01666
; LINE_WIDTH: 0.860636
G1 F9530.438
G1 X101.134 Y131.687 E.01749
; LINE_WIDTH: 0.864836
G1 F9482.241
G1 X101.165 Y131.679 E.00173
; LINE_WIDTH: 0.910991
G1 F8983.01
G1 X101.479 Y131.597 E.01854
; LINE_WIDTH: 0.957146
G1 F8533.718
G1 X101.793 Y131.516 E.01952
; LINE_WIDTH: 1.0033
G1 F8127.229
G1 X102.107 Y131.434 E.02049
; LINE_WIDTH: 1.04946
G1 F7757.704
G1 X102.421 Y131.353 E.02147
; LINE_WIDTH: 1.06574
G1 F7635.253
G1 X102.519 Y131.326 E.00684
; WIPE_START
G1 X102.421 Y131.353 E-.03864
G1 X102.107 Y131.434 E-.12327
G1 X101.793 Y131.516 E-.12327
G1 X101.551 Y131.579 E-.09481
; WIPE_END
G1 E-.02 F1800
G1 X100.747 Y139.169 Z4.12 F36000
G1 X100.358 Y142.835 Z4.12
G1 Z3.72
G1 E.4 F1800
; LINE_WIDTH: 1.00776
G1 F8090.034
G1 X100.57 Y142.854 E.01347
; LINE_WIDTH: 0.968926
G1 F8426.155
G1 X100.818 Y142.877 E.01523
; LINE_WIDTH: 0.92321
G1 F8859.526
G1 X101.067 Y142.9 E.01449
; LINE_WIDTH: 0.877494
G1 F9339.893
G1 X101.316 Y142.923 E.01374
; LINE_WIDTH: 0.831778
G1 F9875.336
G1 X101.565 Y142.946 E.013
; LINE_WIDTH: 0.786061
G1 F10475.906
G1 X101.814 Y142.969 E.01225
; LINE_WIDTH: 0.740345
G1 F11154.254
G1 X102.063 Y142.992 E.01151
; LINE_WIDTH: 0.694629
G1 F11926.533
G1 X102.312 Y143.014 E.01076
; LINE_WIDTH: 0.648912
G1 F12813.708
G1 X102.561 Y143.037 E.01002
; LINE_WIDTH: 0.603196
G1 F13843.477
G1 X103.042 Y143.038 E.01783
; WIPE_START
G1 X102.561 Y143.037 E-.18275
G1 X102.312 Y143.014 E-.095
G1 X102.063 Y142.992 E-.095
G1 X102.044 Y142.99 E-.00725
; WIPE_END
G1 E-.02 F1800
G1 X105.256 Y139.78 Z4.12 F36000
G1 Z3.72
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.256 Y137.438 E.08944
G3 X108.2 Y138.479 I.12 J4.344 E.12196
G2 X110.085 Y140.234 I112.726 J-119.204 E.09833
G2 X115.741 Y140.147 I2.775 J-3.493 E.2339
G3 X117.626 Y138.392 I112.726 J119.204 E.09833
G3 X123.281 Y138.479 I2.775 J3.493 E.2339
G2 X125.166 Y140.234 I112.801 J-119.285 E.09833
G2 X130.822 Y140.147 I2.775 J-3.493 E.2339
G3 X132.707 Y138.392 I112.726 J119.204 E.09833
G3 X138.362 Y138.479 I2.775 J3.493 E.2339
G2 X140.248 Y140.234 I112.726 J-119.204 E.09833
G2 X145.903 Y140.147 I2.775 J-3.493 E.2339
G3 X147.788 Y138.392 I112.726 J119.204 E.09833
G1 X148.157 Y138.163 E.01656
G3 X143.892 Y133.588 I37.794 J-39.506 E.23894
G3 X140.248 Y132.607 I-.784 J-4.346 E.14887
G2 X138.362 Y130.852 I-112.726 J119.204 E.09833
G2 X132.707 Y130.938 I-2.775 J3.493 E.2339
G3 X130.822 Y132.693 I-112.808 J-119.291 E.09833
G3 X125.166 Y132.607 I-2.775 J-3.493 E.2339
G2 X123.281 Y130.852 I-112.65 J119.123 E.09833
G2 X117.626 Y130.938 I-2.775 J3.493 E.2339
G3 X115.741 Y132.693 I-112.808 J-119.291 E.09833
G3 X111.97 Y133.549 I-2.785 J-3.536 E.15257
G3 X109.142 Y131.722 I1.647 J-5.651 E.13035
G2 X107.325 Y130.308 I-4.591 J4.026 E.08844
G2 X109.435 Y129.291 I-19.508 J-43.16 E.08943
; WIPE_START
G1 X108.534 Y129.725 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X115.522 Y126.655 Z4.12 F36000
G1 X130.519 Y120.064 Z4.12
G1 Z3.72
G1 E.4 F1800
G1 F13446.283
G3 X128.783 Y118.509 I5.004 J-7.333 E.08924
G3 X125.166 Y117.525 I-.764 J-4.328 E.14782
G1 X125.072 Y117.437 E.00492
G3 X123.628 Y118.947 I-25.584 J-23.04 E.07979
G3 X124.464 Y120.83 I-2.001 J2.015 E.08051
G3 X122.991 Y123.209 I-3.008 J-.217 E.11107
G3 X125.166 Y125.153 I-16.232 J20.352 E.11144
G2 X130.822 Y125.066 I2.775 J-3.493 E.2339
G3 X132.707 Y123.311 I112.808 J119.291 E.09833
G3 X136.477 Y122.455 I2.785 J3.536 E.15257
G1 X136.648 Y122.515 E.00691
G2 X137.732 Y124.591 I57.541 J-28.711 E.08943
; CHANGE_LAYER
; Z_HEIGHT: 3.88
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.269 Y123.705 E-.38
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
G1 X123.51 Y122.061
G1 Z3.88
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.391 Y122.215 E.00743
G3 X122.296 Y123.143 I-50.898 J-58.997 E.05481
G1 X119.23 Y125.712 E.15272
G1 X118.874 Y125.95 E.01636
G1 X118.451 Y126.11 E.01727
G3 X117.195 Y126.004 I-.454 J-2.116 E.0488
G1 X116.82 Y125.787 E.01657
G3 X115.927 Y124.807 I5.234 J-5.665 E.05069
G3 X103.765 Y131.255 I-31.446 J-44.611 E.52695
G1 X104.392 Y131.744 E.03034
G1 X104.702 Y132.413 E.02813
G3 X104.76 Y133.707 I-6.412 J.935 E.04954
G1 X104.76 Y141.707 E.30543
G1 X104.665 Y142.308 E.02324
G1 X104.588 Y142.443 E.00594
G1 X154.069 Y142.443 E1.88913
G3 X143.802 Y132.7 I31.509 J-43.484 E.54197
G3 X136.316 Y120.644 I41.946 J-34.396 E.54337
G1 X136.271 Y120.545 E.00413
G3 X135.168 Y120.737 I-2.219 J-9.499 E.04277
G3 X133.273 Y120.651 I-.569 J-8.442 E.0726
G1 X132.426 Y120.433 E.03338
G1 X131.481 Y120.051 E.03893
G1 X130.604 Y119.536 E.03882
G1 X129.81 Y118.896 E.03892
G1 X129.118 Y118.147 E.03895
G3 X127.853 Y116.453 I77.237 J-59.026 E.08071
G1 X126.662 Y114.847 E.07636
G3 X122.955 Y118.919 I-40.771 J-33.394 E.21033
G3 X123.666 Y119.795 I-6.078 J5.659 E.04314
G1 X123.913 Y120.386 E.02443
G1 X123.965 Y120.833 E.01721
G1 X123.892 Y121.363 E.02039
G1 X123.702 Y121.812 E.01865
G1 X123.565 Y121.99 E.00855
G1 X123.013 Y121.746 F36000
G1 F13446.369
G1 X122.892 Y121.879 E.00687
G1 X118.854 Y125.264 E.20115
G1 X118.51 Y125.474 E.01539
G1 X118.095 Y125.577 E.01636
G3 X117.168 Y125.316 I-.084 J-1.474 E.03746
G1 X116.783 Y124.917 E.02115
G1 X116.03 Y124.018 E.04479
G3 X104.453 Y130.36 I-32.054 J-44.773 E.50519
G1 X104.066 Y130.512 E.01587
G1 X103.693 Y130.656 E.01527
G1 F12109.096
G1 X103.319 Y130.799 E.01527
; LINE_WIDTH: 0.669338
G1 F10783.694
G1 X103.231 Y130.858 E.00439
; LINE_WIDTH: 0.718681
G1 F10444.906
G1 X103.142 Y130.916 E.00474
; LINE_WIDTH: 0.768023
G1 F10111.507
G1 X103.053 Y130.974 E.00508
; LINE_WIDTH: 0.817365
G1 F9783.508
G1 X102.965 Y131.033 E.00542
; LINE_WIDTH: 0.866707
G1 F9460.925
G1 X102.876 Y131.091 E.00576
; LINE_WIDTH: 0.91605
G1 F8931.475
G1 X102.787 Y131.149 E.0061
; LINE_WIDTH: 0.965392
G1 F8458.143
G1 X102.699 Y131.208 E.00644
; LINE_WIDTH: 1.01473
G1 F8032.455
G1 X102.61 Y131.266 E.00678
; LINE_WIDTH: 1.06408
G1 F7647.562
G1 X102.521 Y131.324 E.00713
G1 X102.607 Y131.356 E.0061
; LINE_WIDTH: 1.01473
G1 F8032.455
G1 X102.692 Y131.387 E.00581
; LINE_WIDTH: 0.965392
G1 F8458.143
G1 X102.777 Y131.418 E.00551
; LINE_WIDTH: 0.91605
G1 F8931.475
G1 X102.862 Y131.45 E.00522
; LINE_WIDTH: 0.866707
G1 F9460.925
G1 X102.947 Y131.481 E.00493
; LINE_WIDTH: 0.817365
G1 F10057.101
G1 X103.033 Y131.513 E.00464
; LINE_WIDTH: 0.768023
G1 F10341.315
G1 X103.118 Y131.544 E.00434
; LINE_WIDTH: 0.718681
G1 F10629.489
G1 X103.203 Y131.575 E.00405
; LINE_WIDTH: 0.669338
G1 F10921.604
G1 X103.288 Y131.607 E.00376
; LINE_WIDTH: 0.619996
G1 F12255.21
G1 X103.606 Y131.849 E.01527
G1 F13446.369
G1 X103.917 Y132.087 E.01491
G1 X104.134 Y132.554 E.01968
G3 X104.174 Y133.707 I-6.162 J.793 E.04409
G1 X104.174 Y141.707 E.30543
G1 X104.108 Y142.127 E.01626
G1 X103.837 Y142.604 E.02094
G1 X103.286 Y142.978 E.02542
; LINE_WIDTH: 0.607576
G1 F13737.703
G1 X103.044 Y143.04 E.00935
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.846 J123.409 E.10326
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.253 Y132.327 I30.061 J-44.232 E.59954
G3 X136.596 Y119.828 I41.406 J-33.963 E.56139
G1 X136.068 Y119.994 E.02114
G1 X135.346 Y120.129 E.02801
G1 X134.411 Y120.179 E.03575
G3 X133.284 Y120.054 I.627 J-10.808 E.04331
G1 X132.566 Y119.865 E.02835
G1 X131.711 Y119.513 E.03529
G1 X130.911 Y119.037 E.03554
G1 X130.187 Y118.448 E.03564
G3 X129.066 Y117.105 I8.516 J-8.248 E.06684
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-42.448 J-33.966 E.25727
G1 X123.061 Y119.957 E.05366
G1 X123.303 Y120.383 E.01871
G1 X123.38 Y120.834 E.01743
G1 X123.329 Y121.204 E.01427
G1 X123.137 Y121.61 E.01715
G1 X123.074 Y121.68 E.00359
G1 X122.581 Y121.353 F36000
G1 F13446.369
G1 X122.516 Y121.43 E.00385
G1 X118.478 Y124.815 E.20115
G1 X118.225 Y124.956 E.01105
G1 X117.92 Y124.995 E.01176
G1 X117.665 Y124.931 E.01001
G1 X117.381 Y124.718 E.01356
G1 X116.128 Y123.223 E.0745
G3 X99.515 Y131.445 I-31.738 J-43.235 E.71127
G1 X99.49 Y131.481 E.00167
G1 X99.583 Y132.483 E.0384
G1 X99.586 Y132.69 E.00789
G1 X102.668 Y132.119 E.11969
G1 X103.098 Y132.161 E.01649
G1 X103.451 Y132.442 E.01724
G1 X103.569 Y132.711 E.01119
G3 X103.589 Y135.707 I-95.51 J2.122 E.11439
G1 X103.589 Y141.707 E.22907
G1 X103.551 Y141.947 E.00928
G1 X103.396 Y142.219 E.01195
G1 X103.083 Y142.436 E.01454
G1 X102.81 Y142.485 E.0106
G1 X102.668 Y142.472 E.00544
G1 X99.589 Y141.902 E.11958
G3 X99.525 Y143.615 I-14.095 J.331 E.06548
G1 X156.235 Y143.615 E2.16517
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.105 J-43.631 E.60779
G3 X136.912 Y119.083 I41.646 J-34.011 E.57634
G1 X136.092 Y119.382 E.03329
G3 X132.719 Y119.299 I-1.545 J-5.807 E.13058
G1 X131.942 Y118.975 E.03216
G1 X131.218 Y118.539 E.03226
G1 X130.564 Y118 E.03236
G1 X129.996 Y117.37 E.03238
G3 X129.081 Y116.142 I119.365 J-89.852 E.05846
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-42.305 J-32.884 E.3043
G1 X122.612 Y120.333 E.07459
G1 X122.773 Y120.655 E.01372
G1 X122.786 Y120.948 E.0112
G1 X122.689 Y121.224 E.01119
G1 X122.639 Y121.284 E.00298
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00555
G1 X116.81 Y123.176 E.04903
; LINE_WIDTH: 0.521316
G1 X116.411 Y122.702 E.01968
; LINE_WIDTH: 0.556516
G1 X116.394 Y122.378 E.01104
G1 X115.476 Y123.015 E.03802
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.116 J-43.088 E.58495
G1 X98.941 Y131.545 E.01607
G3 X99.035 Y133.354 I-10.154 J1.433 E.05741
G1 X102.769 Y132.662 E.12023
G1 X102.894 Y132.674 E.00396
G1 X103.029 Y132.83 E.00652
M73 P58 R7
G3 X103.036 Y135.707 I-242.415 J2.003 E.09109
G1 X103.036 Y141.707 E.18996
G1 X102.98 Y141.855 E.00502
G1 X102.81 Y141.932 E.00591
G1 X102.769 Y141.929 E.00131
G1 X99.036 Y141.237 E.12021
G3 X99.005 Y143.307 I-22.745 J.693 E.06555
G2 X98.936 Y144.167 I3.724 J.73 E.02739
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.647 J-43.535 E.51062
G3 X137.192 Y118.315 I40.427 J-33.165 E.49166
G1 X136.612 Y118.616 E.02069
G1 X135.912 Y118.859 E.02345
G1 X135.3 Y118.98 E.01975
G1 X134.369 Y119.041 E.02955
G1 X133.607 Y118.959 E.02426
G1 X132.865 Y118.766 E.02426
G1 X132.16 Y118.466 E.02426
G1 X131.508 Y118.068 E.02419
G1 X130.92 Y117.577 E.02427
G1 X130.41 Y117.004 E.02428
G3 X129.087 Y115.222 I548.144 J-408.325 E.07025
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-41.979 J-31.66 E.27953
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.851 E.00539
G1 X122.202 Y120.927 E.0027
; WIPE_START
M204 S10000
G1 X121.44 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z4.28 F36000
G1 Z3.88
G1 E.4 F1800
; LINE_WIDTH: 0.556516
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.01281
; LINE_WIDTH: 0.544336
G1 X120.199 Y119.2 E.15232
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.199 Y119.2 E-.13038
G1 X119.695 Y119.622 E-.24962
; WIPE_END
G1 E-.02 F1800
G1 X113.239 Y123.693 Z4.28 F36000
G1 X100.189 Y131.921 Z4.28
G1 Z3.88
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.780096
G1 F10559.699
G1 X100.504 Y131.842 E.01578
; LINE_WIDTH: 0.819316
G1 F10032.103
G1 X100.819 Y131.764 E.01661
; LINE_WIDTH: 0.858536
G1 F9554.721
G1 X101.134 Y131.686 E.01744
; LINE_WIDTH: 0.862776
G1 F9505.819
G1 X101.165 Y131.678 E.00173
; LINE_WIDTH: 0.908926
G1 F9004.22
G1 X101.479 Y131.596 E.0185
; LINE_WIDTH: 0.955076
G1 F8552.904
G1 X101.793 Y131.515 E.01947
; LINE_WIDTH: 1.00123
G1 F8144.67
G1 X102.107 Y131.433 E.02045
; LINE_WIDTH: 1.04738
G1 F7773.632
G1 X102.42 Y131.352 E.02142
; LINE_WIDTH: 1.06408
G1 F7647.562
G1 X102.521 Y131.324 E.00701
; WIPE_START
G1 X102.42 Y131.352 E-.03966
G1 X102.107 Y131.433 E-.12327
G1 X101.793 Y131.515 E-.12327
G1 X101.554 Y131.577 E-.0938
; WIPE_END
G1 E-.02 F1800
G1 X100.748 Y139.167 Z4.28 F36000
G1 X100.359 Y142.836 Z4.28
G1 Z3.88
G1 E.4 F1800
; LINE_WIDTH: 1.00552
G1 F8108.692
G1 X100.572 Y142.856 E.01354
; LINE_WIDTH: 0.966406
G1 F8448.937
G1 X100.821 Y142.879 E.01519
; LINE_WIDTH: 0.92069
G1 F8884.715
G1 X101.07 Y142.901 E.01445
; LINE_WIDTH: 0.874974
G1 F9367.891
G1 X101.319 Y142.924 E.0137
; LINE_WIDTH: 0.829257
G1 F9906.643
G1 X101.568 Y142.947 E.01296
; LINE_WIDTH: 0.783541
G1 F10511.142
G1 X101.817 Y142.97 E.01221
; LINE_WIDTH: 0.737825
G1 F11194.21
G1 X102.066 Y142.993 E.01146
; LINE_WIDTH: 0.692109
G1 F11972.225
G1 X102.315 Y143.016 E.01072
; LINE_WIDTH: 0.646393
G1 F12866.465
G1 X102.563 Y143.039 E.00997
; LINE_WIDTH: 0.600676
G1 F13905.077
G1 X103.044 Y143.04 E.01774
; WIPE_START
G1 X102.563 Y143.039 E-.18258
G1 X102.315 Y143.016 E-.095
G1 X102.066 Y142.993 E-.095
G1 X102.046 Y142.991 E-.00742
; WIPE_END
G1 E-.02 F1800
G1 X105.258 Y139.785 Z4.28 F36000
G1 Z3.88
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.258 Y137.442 E.08944
G3 X108.2 Y138.34 I.346 J4.133 E.12032
G3 X110.085 Y140.097 I-50.311 J55.895 E.09837
G2 X115.741 Y140.286 I2.944 J-3.373 E.23386
G2 X117.626 Y138.53 I-50.311 J-55.895 E.09837
G3 X123.281 Y138.34 I2.944 J3.373 E.23386
G3 X125.166 Y140.097 I-50.295 J55.878 E.09837
G2 X130.822 Y140.286 I2.944 J-3.373 E.23386
G2 X132.707 Y138.53 I-50.311 J-55.895 E.09837
G3 X138.362 Y138.34 I2.944 J3.373 E.23386
G3 X140.248 Y140.097 I-50.311 J55.895 E.09837
G2 X144.018 Y141.135 I2.989 J-3.487 E.15424
G2 X146.846 Y139.423 I-1.187 J-5.152 E.12837
G3 X148.23 Y138.234 I4.197 J3.485 E.07
G3 X143.862 Y133.554 I52.118 J-53.022 E.24448
G3 X140.248 Y132.745 I-.96 J-4.193 E.14619
G3 X138.362 Y130.989 I50.294 J-55.876 E.09837
G2 X132.707 Y130.8 I-2.944 J3.373 E.23386
G2 X130.822 Y132.556 I50.311 J55.895 E.09837
G3 X125.166 Y132.745 I-2.944 J-3.373 E.23386
G3 X123.281 Y130.989 I50.31 J-55.894 E.09837
G2 X117.626 Y130.8 I-2.944 J3.373 E.23386
G2 X115.741 Y132.556 I50.311 J55.895 E.09837
G3 X111.97 Y133.595 I-2.989 J-3.487 E.15424
G3 X109.142 Y131.882 I1.188 J-5.152 E.12837
G2 X107.257 Y130.357 I-5.437 J4.794 E.09301
G1 X107.234 Y130.349 E.00096
G2 X109.346 Y129.337 I-19.439 J-43.302 E.08943
; WIPE_START
G1 X108.444 Y129.769 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X115.427 Y126.687 Z4.28 F36000
G1 X130.482 Y120.041 Z4.28
G1 Z3.88
G1 E.4 F1800
G1 F13446.283
G3 X128.754 Y118.476 I5.168 J-7.44 E.08925
G3 X125.166 Y117.664 I-.943 J-4.166 E.14519
G1 X125.001 Y117.513 E.00856
G3 X123.629 Y118.948 I-34.261 J-31.371 E.07582
G3 X124.327 Y120.024 I-2.489 J2.378 E.04923
G3 X123.078 Y123.136 I-2.614 J.758 E.13795
G3 X125.166 Y125.015 I-13.367 J16.958 E.10734
G2 X130.822 Y125.205 I2.944 J-3.373 E.23386
G2 X132.707 Y123.448 I-50.328 J-55.913 E.09837
G3 X136.477 Y122.41 I2.989 J3.487 E.15424
G1 X136.619 Y122.452 E.00565
G2 X137.694 Y124.533 I86.139 J-43.178 E.08943
; CHANGE_LAYER
; Z_HEIGHT: 4.04
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.235 Y123.644 E-.38
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
G1 X123.506 Y122.065
G1 Z4.04
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.379 Y122.227 E.00785
G3 X122.296 Y123.143 I-61.027 J-71.108 E.05415
G1 X119.23 Y125.712 E.15272
M73 P59 R7
G1 X118.874 Y125.95 E.01636
G1 X118.45 Y126.11 E.01728
G1 X117.978 Y126.168 E.01818
G1 X117.493 Y126.107 E.01866
G1 X117.195 Y126.004 E.01201
G1 X116.82 Y125.787 E.01657
G3 X115.927 Y124.807 I5.236 J-5.667 E.05069
G3 X103.769 Y131.254 I-31.535 J-44.779 E.5268
G1 X104.391 Y131.738 E.0301
G1 X104.704 Y132.409 E.02829
G3 X104.762 Y133.709 I-6.422 J.938 E.04976
G1 X104.762 Y141.709 E.30543
G1 X104.667 Y142.311 E.02324
G1 X104.592 Y142.443 E.00582
G1 X154.069 Y142.443 E1.88899
G3 X143.806 Y132.705 I31.791 J-43.781 E.54171
G3 X136.269 Y120.54 I41.975 J-34.425 E.54796
G1 X135.832 Y120.644 E.01714
G3 X133.899 Y120.738 I-1.383 J-8.488 E.07403
G3 X132.106 Y120.322 I1.154 J-9.043 E.07038
G1 X131.179 Y119.893 E.039
G1 X130.477 Y119.447 E.03177
G1 X129.8 Y118.888 E.03349
G1 X129.118 Y118.145 E.03852
G3 X127.853 Y116.453 I77.093 J-58.916 E.08065
G1 X126.662 Y114.847 E.07636
G3 X122.955 Y118.919 I-42.427 J-34.901 E.21032
G3 X123.665 Y119.794 I-6.108 J5.684 E.04307
G1 X123.913 Y120.385 E.02448
G1 X123.965 Y120.833 E.01722
G1 X123.903 Y121.324 E.01888
G1 X123.698 Y121.818 E.02043
G1 X123.561 Y121.994 E.00852
G1 X123.016 Y121.741 F36000
G1 F13446.369
G1 X122.892 Y121.879 E.00707
G1 X118.854 Y125.264 E.20115
G1 X118.51 Y125.474 E.01539
G1 X118.094 Y125.577 E.01636
G1 X117.639 Y125.539 E.01746
G1 X117.188 Y125.33 E.01896
G1 X116.82 Y124.961 E.01991
G1 X116.03 Y124.018 E.04698
G3 X104.45 Y130.364 I-31.606 J-43.934 E.50542
G1 X104.066 Y130.512 E.0157
G1 X103.693 Y130.656 E.01527
G1 F12116.665
G1 X103.319 Y130.799 E.01527
; LINE_WIDTH: 0.669154
G1 F10790.836
G1 X103.231 Y130.858 E.00438
; LINE_WIDTH: 0.718312
G1 F10452.927
G1 X103.142 Y130.916 E.00472
; LINE_WIDTH: 0.76747
G1 F10120.376
G1 X103.054 Y130.974 E.00506
; LINE_WIDTH: 0.816627
G1 F9793.201
G1 X102.966 Y131.032 E.0054
; LINE_WIDTH: 0.865785
G1 F9471.419
G1 X102.877 Y131.09 E.00574
; LINE_WIDTH: 0.914943
G1 F8942.7
G1 X102.789 Y131.148 E.00608
; LINE_WIDTH: 0.964101
G1 F8469.888
G1 X102.7 Y131.206 E.00641
; LINE_WIDTH: 1.01326
G1 F8044.562
G1 X102.612 Y131.265 E.00675
; LINE_WIDTH: 1.06242
G1 F7659.91
G1 X102.523 Y131.323 E.00709
G1 X102.609 Y131.354 E.00607
; LINE_WIDTH: 1.01326
G1 F8044.562
G1 X102.694 Y131.385 E.00578
; LINE_WIDTH: 0.964101
G1 F8469.888
G1 X102.779 Y131.416 E.00549
; LINE_WIDTH: 0.914943
G1 F8942.7
G1 X102.864 Y131.448 E.0052
; LINE_WIDTH: 0.865785
G1 F9471.419
G1 X102.949 Y131.479 E.00491
; LINE_WIDTH: 0.816627
G1 F10066.586
G1 X103.034 Y131.51 E.00462
; LINE_WIDTH: 0.76747
G1 F10350.208
G1 X103.119 Y131.541 E.00433
; LINE_WIDTH: 0.718312
G1 F10637.729
G1 X103.204 Y131.572 E.00404
; LINE_WIDTH: 0.669154
G1 F10929.2
G1 X103.289 Y131.604 E.00375
; LINE_WIDTH: 0.619996
G1 F12263.256
G1 X103.607 Y131.846 E.01527
G1 F13446.369
G1 X103.917 Y132.081 E.01485
G1 X104.136 Y132.551 E.01979
G3 X104.176 Y133.709 I-6.172 J.796 E.0443
G1 X104.176 Y141.709 E.30543
G1 X104.11 Y142.13 E.01626
G1 X103.839 Y142.607 E.02094
G1 X103.288 Y142.98 E.02541
; LINE_WIDTH: 0.606116
G1 F13772.782
G1 X103.046 Y143.041 E.00932
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.845 J110.34 E.10319
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.256 Y132.33 I30.043 J-44.213 E.59937
G3 X136.598 Y119.835 I41.403 J-33.967 E.56129
G1 X135.957 Y120.021 E.02551
G3 X133.964 Y120.156 I-1.504 J-7.401 E.07647
G3 X132.283 Y119.764 I1.025 J-8.189 E.06604
G1 X131.436 Y119.367 E.03571
G1 X130.802 Y118.959 E.02879
G1 X130.18 Y118.442 E.03086
G3 X129.066 Y117.105 I8.616 J-8.313 E.06649
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-42.713 J-34.207 E.25726
G1 X123.061 Y119.957 E.05366
G1 X123.303 Y120.384 E.01872
G1 X123.38 Y120.833 E.01742
G1 X123.312 Y121.257 E.01638
G1 X123.148 Y121.595 E.01433
G1 X123.076 Y121.675 E.00411
G1 X122.581 Y121.354 F36000
G1 F13446.369
G1 X122.516 Y121.43 E.00383
G1 X118.478 Y124.815 E.20115
G1 X118.225 Y124.956 E.01106
G1 X117.92 Y124.994 E.01176
G1 X117.665 Y124.931 E.01
G1 X117.381 Y124.718 E.01356
G1 X116.128 Y123.223 E.0745
G3 X99.513 Y131.445 I-31.739 J-43.236 E.71133
G1 X99.49 Y131.48 E.00157
G1 X99.585 Y132.482 E.03842
G1 X99.588 Y132.687 E.00785
G1 X102.67 Y132.116 E.11969
G1 X103.099 Y132.158 E.01645
G1 X103.452 Y132.438 E.01721
G1 X103.571 Y132.708 E.01125
G3 X103.591 Y135.709 I-95.426 J2.124 E.11459
G1 X103.591 Y141.709 E.22907
G1 X103.553 Y141.949 E.00928
G1 X103.398 Y142.222 E.01195
G1 X103.085 Y142.438 E.01454
G1 X102.812 Y142.488 E.0106
G1 X102.67 Y142.475 E.00544
G1 X99.591 Y141.904 E.11958
G3 X99.525 Y143.615 I-13.804 J.33 E.06538
G1 X156.235 Y143.615 E2.16513
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.706 Y131.956 I29.317 J-43.863 E.60766
G3 X136.908 Y119.075 I40.982 J-33.613 E.57682
G1 X136.5 Y119.253 E.01699
G1 X135.794 Y119.458 E.02809
G1 X134.957 Y119.581 E.03229
G3 X133.993 Y119.569 I-.317 J-13.704 E.03681
G1 X133.269 Y119.453 E.028
G1 X132.46 Y119.205 E.03231
G3 X130.56 Y117.996 I2.402 J-5.871 E.08643
G3 X129.081 Y116.142 I12.192 J-11.243 E.09063
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-42.485 J-33.046 E.3043
G1 X122.612 Y120.334 E.07459
G1 X122.773 Y120.655 E.01371
G1 X122.786 Y120.943 E.01102
G1 X122.688 Y121.227 E.01148
G1 X122.639 Y121.285 E.0029
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00555
G1 X116.81 Y123.176 E.04905
; LINE_WIDTH: 0.521326
G1 X116.411 Y122.702 E.01966
; LINE_WIDTH: 0.556496
G1 X116.394 Y122.378 E.01104
G1 X115.477 Y123.015 E.038
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.117 J-43.088 E.58496
G1 X98.941 Y131.545 E.01605
G1 X99.034 Y132.524 E.03114
G1 X99.037 Y133.351 E.02619
G1 X102.771 Y132.66 E.12023
G1 X102.895 Y132.672 E.00395
G1 X103.031 Y132.827 E.00653
G3 X103.038 Y135.709 I-242.29 J2.005 E.09125
G1 X103.038 Y141.709 E.18996
G1 X102.982 Y141.858 E.00502
G1 X102.812 Y141.935 E.00591
G1 X102.771 Y141.931 E.00131
G1 X99.038 Y141.24 E.1202
G3 X99.006 Y143.306 I-22.301 J.692 E.06544
G2 X98.936 Y144.167 I3.657 J.732 E.02743
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.61 J-43.494 E.51061
G3 X137.192 Y118.316 I40.27 J-33.072 E.49164
G3 X136.346 Y118.722 I-5.226 J-9.803 E.02973
G1 X135.639 Y118.927 E.02329
G1 X134.882 Y119.033 E.0242
G1 X134.114 Y119.026 E.02433
G1 X133.466 Y118.925 E.02077
G1 X132.626 Y118.678 E.02769
G3 X130.918 Y117.576 I2.207 J-5.293 E.06471
G1 X130.409 Y117.003 E.02427
G3 X129.087 Y115.222 I549.767 J-409.525 E.07021
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-42.118 J-31.784 E.27953
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.85 E.00537
G1 X122.202 Y120.926 E.00272
; WIPE_START
M204 S10000
G1 X121.439 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z4.44 F36000
G1 Z4.04
G1 E.4 F1800
; LINE_WIDTH: 0.556496
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.01281
; LINE_WIDTH: 0.544336
G1 X120.199 Y119.2 E.15232
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.199 Y119.2 E-.13038
G1 X119.695 Y119.622 E-.24962
; WIPE_END
G1 E-.02 F1800
G1 X113.239 Y123.692 Z4.44 F36000
G1 X100.189 Y131.92 Z4.44
G1 Z4.04
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.778043
G1 F10588.853
G1 X100.504 Y131.841 E.01574
; LINE_WIDTH: 0.81727
G1 F10058.328
G1 X100.819 Y131.763 E.01657
; LINE_WIDTH: 0.856496
G1 F9578.428
G1 X101.134 Y131.685 E.0174
; LINE_WIDTH: 0.860716
G1 F9529.515
G1 X101.165 Y131.677 E.00172
; LINE_WIDTH: 0.906851
G1 F9025.633
G1 X101.478 Y131.595 E.01845
; LINE_WIDTH: 0.952986
G1 F8572.363
G1 X101.792 Y131.514 E.01942
; LINE_WIDTH: 0.999121
G1 F8162.441
G1 X102.106 Y131.432 E.0204
; LINE_WIDTH: 1.04526
G1 F7789.935
G1 X102.42 Y131.351 E.02137
; LINE_WIDTH: 1.06242
G1 F7659.91
G1 X102.523 Y131.323 E.00717
; WIPE_START
G1 X102.42 Y131.351 E-.04067
G1 X102.106 Y131.432 E-.12324
G1 X101.792 Y131.514 E-.12324
G1 X101.556 Y131.575 E-.09284
; WIPE_END
G1 E-.02 F1800
G1 X100.75 Y139.165 Z4.44 F36000
G1 X100.36 Y142.837 Z4.44
G1 Z4.04
G1 E.4 F1800
; LINE_WIDTH: 1.00328
G1 F8127.439
G1 X100.574 Y142.857 E.0136
; LINE_WIDTH: 0.963886
G1 F8471.841
G1 X100.823 Y142.88 E.01515
; LINE_WIDTH: 0.91817
G1 F8910.047
G1 X101.072 Y142.903 E.0144
; LINE_WIDTH: 0.872454
G1 F9396.058
G1 X101.321 Y142.926 E.01366
; LINE_WIDTH: 0.826738
G1 F9938.148
G1 X101.57 Y142.948 E.01291
; LINE_WIDTH: 0.781021
G1 F10546.617
G1 X101.819 Y142.971 E.01217
; LINE_WIDTH: 0.735305
G1 F11234.453
G1 X102.068 Y142.994 E.01142
; LINE_WIDTH: 0.689589
G1 F12018.269
G1 X102.317 Y143.017 E.01068
; LINE_WIDTH: 0.643872
G1 F12919.659
G1 X102.566 Y143.04 E.00993
; LINE_WIDTH: 0.598156
G1 F13967.224
G1 X103.046 Y143.041 E.01764
; WIPE_START
G1 X102.566 Y143.04 E-.18241
G1 X102.317 Y143.017 E-.095
G1 X102.068 Y142.994 E-.095
G1 X102.048 Y142.992 E-.00759
; WIPE_END
G1 E-.02 F1800
G1 X105.26 Y139.783 Z4.44 F36000
G1 Z4.04
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.26 Y137.441 E.08944
G3 X108.2 Y138.197 I.556 J3.932 E.11893
G3 X110.085 Y139.962 I-19.824 J23.067 E.09862
G2 X114.798 Y140.954 I3.208 J-3.547 E.1927
G2 X116.683 Y139.585 I-2.076 J-4.84 E.0897
G3 X118.568 Y137.985 I6.294 J5.506 E.09474
G3 X123.281 Y138.197 I2.182 J3.985 E.18939
G3 X125.166 Y139.962 I-19.821 J23.064 E.09862
G2 X129.879 Y140.954 I3.208 J-3.547 E.1927
G2 X131.764 Y139.585 I-2.076 J-4.84 E.0897
G3 X133.65 Y137.985 I6.294 J5.506 E.09474
G3 X138.362 Y138.197 I2.182 J3.985 E.18939
G3 X140.248 Y139.962 I-19.824 J23.067 E.09862
G2 X144.961 Y140.954 I3.208 J-3.547 E.1927
G2 X146.846 Y139.585 I-2.076 J-4.84 E.0897
G3 X148.298 Y138.297 I5.033 J4.215 E.07437
G3 X143.842 Y133.526 I39.751 J-41.59 E.24936
G3 X141.19 Y133.413 I-1.125 J-4.772 E.10263
G3 X139.305 Y132.044 I2.076 J-4.84 E.0897
G2 X137.42 Y130.444 I-6.294 J5.507 E.09474
G2 X132.707 Y130.656 I-2.182 J3.985 E.18939
G2 X130.822 Y132.421 I19.83 J23.072 E.09862
G3 X126.109 Y133.413 I-3.208 J-3.547 E.1927
G3 X124.224 Y132.044 I2.076 J-4.84 E.0897
G2 X122.339 Y130.444 I-6.295 J5.507 E.09474
G2 X117.626 Y130.656 I-2.182 J3.985 E.18939
G2 X115.741 Y132.421 I19.83 J23.072 E.09862
G3 X111.028 Y133.413 I-3.208 J-3.547 E.1927
G3 X109.142 Y132.044 I2.076 J-4.84 E.0897
G2 X107.257 Y130.444 I-6.294 J5.507 E.09474
G1 X107.137 Y130.394 E.00499
G2 X109.252 Y129.388 I-15.295 J-34.912 E.08945
; WIPE_START
G1 X108.349 Y129.818 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X115.327 Y126.725 Z4.44 F36000
G1 X130.453 Y120.023 Z4.44
G1 Z4.04
G1 E.4 F1800
G1 F13446.283
G3 X128.732 Y118.45 I4.875 J-7.063 E.08929
G3 X125.166 Y117.807 I-1.113 J-4.037 E.14304
G1 X124.926 Y117.592 E.01233
G1 X123.628 Y118.947 E.07164
G3 X123.174 Y123.056 I-1.746 J1.887 E.18337
G1 X124.224 Y123.96 E.05291
G2 X126.109 Y125.56 I6.294 J-5.507 E.09474
G2 X130.822 Y125.348 I2.182 J-3.985 E.18939
G2 X132.707 Y123.583 I-19.824 J-23.067 E.09862
G3 X136.477 Y122.358 I3.206 J3.451 E.15622
G1 X136.587 Y122.385 E.0043
G2 X137.659 Y124.468 I74.462 J-37.04 E.08943
; CHANGE_LAYER
; Z_HEIGHT: 4.2
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.201 Y123.579 E-.38
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
G1 X123.51 Y122.061
G1 Z4.2
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.391 Y122.215 E.00744
G3 X122.296 Y123.143 I-40.57 J-46.801 E.05481
G1 X119.23 Y125.712 E.15272
G1 X118.874 Y125.95 E.01636
G1 X118.451 Y126.11 E.01727
G3 X117.196 Y126.004 I-.454 J-2.116 E.04879
G1 X116.82 Y125.787 E.01657
G3 X115.927 Y124.807 I5.235 J-5.665 E.05069
G3 X103.772 Y131.253 I-31.514 J-44.74 E.52667
G1 X104.39 Y131.731 E.02985
G1 X104.706 Y132.406 E.02845
G3 X104.764 Y133.712 I-6.431 J.941 E.04998
G1 X104.764 Y141.712 E.30543
G1 X104.669 Y142.313 E.02324
G1 X104.595 Y142.443 E.00572
G1 X154.069 Y142.443 E1.88888
G3 X143.802 Y132.7 I31.949 J-43.95 E.54194
G3 X136.271 Y120.545 I41.778 J-34.296 E.54754
G1 X135.453 Y120.705 E.0318
G3 X133.274 Y120.651 I-.891 J-8.103 E.08348
G1 X132.426 Y120.433 E.03341
G1 X131.481 Y120.051 E.03895
G1 X130.601 Y119.534 E.03893
G1 X129.81 Y118.896 E.03883
G1 X129.119 Y118.147 E.03891
G3 X127.853 Y116.453 I77.119 J-58.939 E.08072
G1 X126.662 Y114.847 E.07636
G3 X122.955 Y118.919 I-42.782 J-35.224 E.21032
G1 X123.51 Y119.581 E.033
G1 X123.726 Y119.898 E.01466
G1 X123.913 Y120.386 E.01994
G1 X123.965 Y120.833 E.01721
G1 X123.893 Y121.361 E.02034
G1 X123.702 Y121.812 E.01868
G1 X123.565 Y121.99 E.00856
G1 X123.013 Y121.746 F36000
G1 F13446.369
G1 X122.892 Y121.879 E.00687
G1 X118.854 Y125.264 E.20115
G1 X118.51 Y125.474 E.01539
G1 X118.095 Y125.577 E.01636
G3 X117.168 Y125.316 I-.085 J-1.473 E.03746
G1 X116.783 Y124.917 E.02115
G1 X116.03 Y124.018 E.04479
G3 X104.445 Y130.365 I-31.73 J-44.166 E.50558
G1 X104.066 Y130.512 E.01553
G1 X103.692 Y130.656 E.01527
G1 F12124.228
G1 X103.319 Y130.799 E.01527
; LINE_WIDTH: 0.66897
G1 F10797.974
G1 X103.231 Y130.857 E.00436
; LINE_WIDTH: 0.717943
G1 F10460.972
G1 X103.143 Y130.915 E.0047
; LINE_WIDTH: 0.766916
G1 F10129.269
G1 X103.055 Y130.973 E.00504
; LINE_WIDTH: 0.81589
G1 F9802.935
G1 X102.967 Y131.031 E.00538
; LINE_WIDTH: 0.864863
G1 F9481.936
G1 X102.878 Y131.089 E.00571
; LINE_WIDTH: 0.913836
G1 F8953.951
G1 X102.79 Y131.147 E.00605
; LINE_WIDTH: 0.96281
G1 F8481.666
G1 X102.702 Y131.205 E.00639
; LINE_WIDTH: 1.01178
G1 F8056.706
G1 X102.614 Y131.263 E.00672
; LINE_WIDTH: 1.06076
G1 F7672.298
G1 X102.526 Y131.321 E.00706
G1 X102.611 Y131.352 E.00605
; LINE_WIDTH: 1.01178
G1 F8056.706
G1 X102.695 Y131.383 E.00576
; LINE_WIDTH: 0.96281
G1 F8481.666
G1 X102.78 Y131.414 E.00547
; LINE_WIDTH: 0.913836
G1 F8953.951
G1 X102.865 Y131.445 E.00518
; LINE_WIDTH: 0.864863
G1 F9481.936
G1 X102.95 Y131.476 E.00489
; LINE_WIDTH: 0.81589
G1 F10076.088
G1 X103.035 Y131.507 E.0046
; LINE_WIDTH: 0.766916
G1 F10359.066
G1 X103.12 Y131.538 E.00432
; LINE_WIDTH: 0.717943
G1 F10645.974
G1 X103.205 Y131.569 E.00403
; LINE_WIDTH: 0.66897
G1 F10936.801
G1 X103.289 Y131.6 E.00374
; LINE_WIDTH: 0.619996
G1 F12271.307
G1 X103.608 Y131.842 E.01527
G1 F13446.369
G1 X103.917 Y132.076 E.01479
G1 X104.138 Y132.548 E.0199
G3 X104.178 Y133.712 I-6.179 J.799 E.04451
G1 X104.178 Y141.712 E.30543
G1 X104.112 Y142.132 E.01626
G1 X103.841 Y142.609 E.02094
G1 X103.29 Y142.982 E.0254
; LINE_WIDTH: 0.604656
G1 F13808.038
G1 X103.048 Y143.042 E.00929
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.843 J99.667 E.10312
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.253 Y132.327 I29.897 J-44.055 E.59955
G3 X136.596 Y119.828 I41.873 J-34.25 E.56137
G1 X136.065 Y119.995 E.02125
G1 X135.346 Y120.129 E.02791
G1 X134.412 Y120.179 E.03572
G3 X133.286 Y120.054 I.624 J-10.785 E.04324
G1 X132.566 Y119.865 E.02844
G1 X131.711 Y119.513 E.0353
G1 X130.909 Y119.036 E.03565
G1 X130.187 Y118.448 E.03555
G3 X129.066 Y117.105 I8.516 J-8.247 E.06681
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-45.891 J-37.09 E.25724
G1 X123.061 Y119.957 E.05366
G1 X123.303 Y120.384 E.01872
G1 X123.38 Y120.833 E.01742
G1 X123.329 Y121.203 E.01423
G1 X123.137 Y121.61 E.01718
G1 X123.074 Y121.68 E.0036
G1 X122.581 Y121.353 F36000
G1 F13446.369
G1 X122.516 Y121.43 E.00385
G1 X118.478 Y124.815 E.20115
G1 X118.225 Y124.956 E.01105
G1 X117.92 Y124.994 E.01176
G1 X117.666 Y124.931 E.01
G1 X117.381 Y124.718 E.01356
G1 X116.128 Y123.223 E.0745
G3 X99.512 Y131.446 I-31.623 J-43.001 E.71142
G3 X99.59 Y132.685 I-29.864 J2.504 E.0474
G1 X102.673 Y132.114 E.1197
G1 X103.1 Y132.155 E.01641
G1 X103.453 Y132.434 E.01717
G1 X103.573 Y132.705 E.01132
G3 X103.593 Y135.712 I-95.33 J2.127 E.11479
G1 X103.593 Y141.712 E.22907
G1 X103.555 Y141.952 E.00928
G1 X103.4 Y142.224 E.01195
G1 X103.087 Y142.441 E.01454
G1 X102.814 Y142.49 E.0106
G1 X102.673 Y142.477 E.00544
G1 X99.593 Y141.907 E.11958
G3 X99.526 Y143.615 I-13.529 J.329 E.06529
G1 X156.235 Y143.615 E2.1651
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.691 J-44.272 E.60771
G3 X136.912 Y119.083 I41.407 J-33.867 E.57636
G1 X136.092 Y119.382 E.03329
G3 X134.38 Y119.594 I-1.77 J-7.28 E.06604
G3 X133.436 Y119.488 I1.14 J-14.428 E.03628
G1 X132.719 Y119.299 E.02828
G1 X131.942 Y118.974 E.03217
G1 X131.216 Y118.537 E.03236
G1 X130.564 Y117.999 E.03227
G1 X129.996 Y117.371 E.03235
G3 X129.081 Y116.142 I118.989 J-89.573 E.05847
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-44.569 J-34.93 E.30427
G1 X122.612 Y120.334 E.07459
G1 X122.751 Y120.577 E.01068
G1 X122.792 Y120.893 E.01216
G1 X122.689 Y121.224 E.01325
G1 X122.639 Y121.284 E.00299
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00556
G1 X116.81 Y123.176 E.04903
; LINE_WIDTH: 0.521326
G1 X116.411 Y122.702 E.01968
; LINE_WIDTH: 0.556496
G1 X116.394 Y122.378 E.01104
G1 X115.476 Y123.015 E.03802
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.116 J-43.088 E.58495
G1 X98.941 Y131.544 E.01604
G1 X99.036 Y132.524 E.03115
G1 X99.039 Y133.349 E.02612
G1 X102.773 Y132.657 E.12023
G1 X102.897 Y132.669 E.00395
G1 X103.033 Y132.824 E.00653
G3 X103.04 Y135.712 I-241.909 J2.008 E.09141
G1 X103.04 Y141.712 E.18996
G1 X102.984 Y141.86 E.00502
M73 P60 R7
G1 X102.814 Y141.937 E.00591
G1 X102.773 Y141.934 E.00131
G1 X99.04 Y141.242 E.1202
G3 X99.008 Y143.305 I-21.859 J.691 E.06532
G2 X98.936 Y144.167 I3.594 J.734 E.02748
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I29.093 J-44.027 E.51056
G3 X137.192 Y118.315 I40.309 J-33.095 E.49168
G1 X136.614 Y118.615 E.0206
G1 X135.912 Y118.859 E.02353
G1 X135.137 Y119.01 E.025
G1 X134.369 Y119.041 E.02433
G1 X133.609 Y118.959 E.02421
G1 X132.865 Y118.766 E.02432
G1 X132.16 Y118.466 E.02428
G1 X131.506 Y118.066 E.02427
G1 X130.919 Y117.576 E.02419
G1 X130.41 Y117.004 E.02426
G3 X129.087 Y115.222 I548.523 J-408.606 E.07026
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-41.978 J-31.659 E.27953
; LINE_WIDTH: 0.554636
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.851 E.00539
G1 X122.202 Y120.927 E.0027
; WIPE_START
M204 S10000
G1 X121.44 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z4.6 F36000
G1 Z4.2
G1 E.4 F1800
; LINE_WIDTH: 0.556496
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.01281
; LINE_WIDTH: 0.544336
G1 X120.198 Y119.2 E.15231
; LINE_WIDTH: 0.554636
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.198 Y119.2 E-.13045
G1 X119.695 Y119.622 E-.24955
; WIPE_END
G1 E-.02 F1800
G1 X113.239 Y123.692 Z4.6 F36000
G1 X100.189 Y131.919 Z4.6
G1 Z4.2
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.775983
G1 F10618.264
G1 X100.503 Y131.84 E.01568
; LINE_WIDTH: 0.81517
G1 F10085.379
G1 X100.818 Y131.762 E.01651
; LINE_WIDTH: 0.854356
G1 F9603.425
G1 X101.133 Y131.684 E.01734
; LINE_WIDTH: 0.858576
G1 F9554.257
G1 X101.164 Y131.676 E.00171
; LINE_WIDTH: 0.904721
G1 F9047.721
G1 X101.478 Y131.594 E.01841
; LINE_WIDTH: 0.950866
G1 F8592.191
G1 X101.792 Y131.513 E.01938
; LINE_WIDTH: 0.997011
G1 F8180.332
G1 X102.106 Y131.431 E.02036
; LINE_WIDTH: 1.04316
G1 F7806.15
G1 X102.42 Y131.35 E.02133
; LINE_WIDTH: 1.06076
G1 F7672.298
G1 X102.526 Y131.321 E.00734
; WIPE_START
G1 X102.42 Y131.35 E-.04169
G1 X102.106 Y131.431 E-.12327
G1 X101.792 Y131.513 E-.12327
G1 X101.558 Y131.574 E-.09177
; WIPE_END
G1 E-.02 F1800
G1 X100.751 Y139.163 Z4.6 F36000
G1 X100.36 Y142.838 Z4.6
G1 Z4.2
G1 E.4 F1800
; LINE_WIDTH: 1.00102
G1 F8146.44
G1 X100.576 Y142.858 E.01367
; LINE_WIDTH: 0.961346
G1 F8495.054
G1 X100.825 Y142.881 E.01511
; LINE_WIDTH: 0.91563
G1 F8935.727
G1 X101.074 Y142.904 E.01436
; LINE_WIDTH: 0.869914
G1 F9424.621
G1 X101.323 Y142.927 E.01362
; LINE_WIDTH: 0.824197
G1 F9970.106
G1 X101.572 Y142.95 E.01287
; LINE_WIDTH: 0.778481
G1 F10582.616
G1 X101.821 Y142.972 E.01213
; LINE_WIDTH: 0.732765
G1 F11275.31
G1 X102.07 Y142.995 E.01138
; LINE_WIDTH: 0.687049
G1 F12065.037
G1 X102.319 Y143.018 E.01064
; LINE_WIDTH: 0.641333
G1 F12973.722
G1 X102.568 Y143.041 E.00989
; LINE_WIDTH: 0.595616
G1 F14030.431
G1 X103.048 Y143.042 E.01755
; WIPE_START
G1 X102.568 Y143.041 E-.18224
G1 X102.319 Y143.018 E-.095
G1 X102.07 Y142.995 E-.095
G1 X102.05 Y142.993 E-.00776
; WIPE_END
G1 E-.02 F1800
G1 X105.262 Y139.777 Z4.6 F36000
G1 Z4.2
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.262 Y137.434 E.08944
G3 X107.729 Y137.764 I.748 J3.791 E.09674
G3 X110.085 Y139.828 I-6.659 J9.981 E.11994
G2 X115.269 Y140.862 I3.367 J-3.369 E.21396
G2 X117.626 Y138.798 I-6.66 J-9.981 E.11994
G3 X122.81 Y137.764 I3.367 J3.369 E.21396
G3 X125.166 Y139.828 I-6.659 J9.98 E.11994
G2 X130.351 Y140.862 I3.367 J-3.369 E.21396
G2 X132.707 Y138.798 I-6.66 J-9.981 E.11994
G3 X137.891 Y137.764 I3.367 J3.369 E.21396
G3 X140.248 Y139.828 I-6.659 J9.981 E.11994
G2 X144.018 Y141.244 I3.462 J-3.491 E.15843
G2 X146.846 Y139.75 I-.6 J-4.558 E.12471
G3 X148.362 Y138.354 I6.079 J5.079 E.07891
G3 X143.823 Y133.506 I37.524 J-39.682 E.25372
G3 X140.719 Y133.322 I-1.318 J-4.032 E.12153
G3 X138.362 Y131.258 I6.66 J-9.981 E.11994
G2 X133.178 Y130.223 I-3.367 J3.369 E.21396
G2 X130.822 Y132.287 I6.66 J9.981 E.11994
G3 X125.638 Y133.322 I-3.367 J-3.369 E.21396
G3 X123.281 Y131.258 I6.66 J-9.982 E.11994
G2 X118.097 Y130.223 I-3.367 J3.369 E.21396
G2 X115.741 Y132.287 I6.66 J9.981 E.11994
G3 X111.97 Y133.703 I-3.462 J-3.491 E.15843
G3 X109.142 Y132.209 I.6 J-4.558 E.12471
G2 X107.257 Y130.527 I-7.354 J6.343 E.09674
G1 X107.055 Y130.432 E.00852
G2 X109.172 Y129.429 I-15.271 J-34.967 E.08945
; WIPE_START
G1 X108.269 Y129.857 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X115.243 Y126.757 Z4.6 F36000
G1 X130.431 Y120.007 Z4.6
G1 Z4.2
G1 E.4 F1800
G1 F13446.283
G3 X128.714 Y118.431 I5.289 J-7.482 E.08925
G3 X124.848 Y117.677 I-1.275 J-3.755 E.15739
G3 X123.629 Y118.948 I-30.797 J-28.317 E.06725
G3 X123.281 Y122.966 I-1.752 J1.873 E.17721
G3 X125.166 Y124.747 I-18.063 J21.014 E.09904
G2 X130.351 Y125.781 I3.367 J-3.369 E.21396
G2 X132.707 Y123.717 I-6.66 J-9.981 E.11994
G3 X136.477 Y122.301 I3.462 J3.491 E.15843
G1 X136.554 Y122.316 E.00299
G2 X137.621 Y124.402 I29.292 J-13.664 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 4.36
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.165 Y123.511 E-.38
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
G1 X123.51 Y122.061
G1 Z4.36
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.391 Y122.216 E.00745
G3 X122.296 Y123.143 I-51.235 J-59.401 E.05479
G1 X119.23 Y125.712 E.15272
G1 X118.874 Y125.95 E.01636
G1 X118.451 Y126.11 E.01727
G3 X117.196 Y126.004 I-.454 J-2.116 E.04879
G1 X116.82 Y125.787 E.01657
G3 X115.927 Y124.807 I5.233 J-5.663 E.05069
G3 X103.776 Y131.251 I-31.533 J-44.776 E.52652
G1 X104.39 Y131.725 E.0296
G1 X104.708 Y132.403 E.0286
G3 X104.766 Y133.714 I-6.439 J.944 E.05019
G1 X104.766 Y141.714 E.30543
G1 X104.671 Y142.316 E.02324
G1 X104.599 Y142.443 E.0056
G1 X154.071 Y142.443 E1.88879
G3 X143.8 Y132.698 I31.786 J-43.783 E.5421
G3 X136.271 Y120.545 I42.405 J-34.681 E.54738
G1 X135.451 Y120.705 E.03188
G3 X133.274 Y120.651 I-.889 J-8.1 E.08339
G1 X132.429 Y120.434 E.03331
G1 X131.481 Y120.051 E.03903
G1 X130.604 Y119.536 E.03885
G1 X129.808 Y118.894 E.03902
G1 X129.119 Y118.147 E.03883
G3 X127.853 Y116.453 I77.11 J-58.931 E.08072
G1 X126.662 Y114.847 E.07636
G3 X122.955 Y118.919 I-43.095 J-35.51 E.21032
G3 X123.663 Y119.791 I-6.177 J5.741 E.04292
G1 X123.913 Y120.386 E.02465
G1 X123.965 Y120.832 E.01716
G1 X123.893 Y121.362 E.02041
G1 X123.702 Y121.813 E.01868
G1 X123.565 Y121.99 E.00855
G1 X123.013 Y121.746 F36000
G1 F13446.369
G1 X122.892 Y121.879 E.00687
G1 X118.854 Y125.264 E.20115
G1 X118.51 Y125.474 E.01539
G1 X118.095 Y125.577 E.01636
G3 X117.168 Y125.316 I-.085 J-1.473 E.03746
G1 X116.783 Y124.917 E.02115
G1 X116.03 Y124.018 E.04479
G3 X104.441 Y130.367 I-31.606 J-43.933 E.50577
G1 X104.066 Y130.512 E.01536
G1 X103.692 Y130.656 E.01527
G1 F12131.784
G1 X103.319 Y130.799 E.01527
; LINE_WIDTH: 0.668785
G1 F10805.105
G1 X103.231 Y130.857 E.00435
; LINE_WIDTH: 0.717574
G1 F10469.01
G1 X103.143 Y130.915 E.00469
; LINE_WIDTH: 0.766363
G1 F10138.182
G1 X103.055 Y130.973 E.00502
; LINE_WIDTH: 0.815152
G1 F9812.682
G1 X102.968 Y131.031 E.00535
; LINE_WIDTH: 0.863941
G1 F9492.477
G1 X102.88 Y131.088 E.00569
; LINE_WIDTH: 0.91273
G1 F8965.233
G1 X102.792 Y131.146 E.00602
; LINE_WIDTH: 0.961518
G1 F8493.477
G1 X102.704 Y131.204 E.00636
; LINE_WIDTH: 1.01031
G1 F8068.887
G1 X102.616 Y131.262 E.00669
; LINE_WIDTH: 1.0591
G1 F7684.726
G1 X102.528 Y131.319 E.00703
G1 X102.613 Y131.35 E.00602
; LINE_WIDTH: 1.01031
G1 F8068.887
G1 X102.697 Y131.381 E.00573
; LINE_WIDTH: 0.961518
G1 F8493.477
G1 X102.782 Y131.412 E.00545
; LINE_WIDTH: 0.91273
G1 F8965.233
G1 X102.867 Y131.443 E.00516
; LINE_WIDTH: 0.863941
G1 F9492.477
G1 X102.951 Y131.474 E.00487
; LINE_WIDTH: 0.815152
G1 F10085.609
G1 X103.036 Y131.505 E.00459
; LINE_WIDTH: 0.766363
G1 F10367.995
G1 X103.121 Y131.536 E.0043
; LINE_WIDTH: 0.717574
G1 F10654.279
G1 X103.205 Y131.566 E.00401
; LINE_WIDTH: 0.668785
G1 F10944.443
G1 X103.29 Y131.597 E.00373
; LINE_WIDTH: 0.619996
G1 F12279.402
G1 X103.609 Y131.838 E.01527
G1 F13446.369
G1 X103.917 Y132.071 E.01473
G1 X104.14 Y132.545 E.02001
G3 X104.181 Y133.714 I-6.186 J.801 E.04472
G1 X104.181 Y141.714 E.30543
G1 X104.114 Y142.135 E.01626
G1 X103.843 Y142.612 E.02094
G1 X103.292 Y142.984 E.0254
; LINE_WIDTH: 0.603176
G1 F13843.965
G1 X103.05 Y143.043 E.00926
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.842 J90.912 E.10305
G1 X155.749 Y143.029 E1.90895
G1 X155.769 Y142.919 E.00428
G3 X144.252 Y132.325 I29.981 J-44.153 E.59958
G3 X136.596 Y119.828 I41.407 J-33.962 E.56132
G1 X136.067 Y119.995 E.02117
G1 X135.344 Y120.129 E.02808
G1 X134.411 Y120.179 E.03566
G3 X133.287 Y120.054 I.626 J-10.801 E.04321
G1 X132.568 Y119.865 E.02838
G1 X131.712 Y119.513 E.03536
G1 X130.911 Y119.037 E.03556
G1 X130.185 Y118.446 E.03574
G3 X129.066 Y117.105 I8.509 J-8.238 E.06673
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-43.155 J-34.608 E.25726
G1 X123.061 Y119.957 E.05366
G1 X123.304 Y120.384 E.01875
G1 X123.38 Y120.833 E.01737
G1 X123.329 Y121.203 E.01428
G1 X123.137 Y121.61 E.01718
G1 X123.074 Y121.68 E.00359
G1 X122.581 Y121.353 F36000
G1 F13446.369
G1 X122.516 Y121.43 E.00385
G1 X118.478 Y124.815 E.20115
G1 X118.225 Y124.956 E.01105
G1 X117.92 Y124.995 E.01176
G1 X117.666 Y124.931 E.01
G1 X117.381 Y124.718 E.01356
G1 X116.128 Y123.223 E.0745
G3 X99.51 Y131.446 I-31.739 J-43.235 E.71145
G3 X99.592 Y132.682 I-26.276 J2.355 E.0473
G1 X102.675 Y132.111 E.1197
G1 X103.102 Y132.152 E.01638
G1 X103.454 Y132.43 E.01714
G1 X103.575 Y132.702 E.01138
G3 X103.595 Y135.714 I-95.149 J2.129 E.11499
G1 X103.595 Y141.714 E.22907
G1 X103.557 Y141.954 E.00928
G1 X103.403 Y142.227 E.01195
G1 X103.089 Y142.443 E.01454
G1 X102.816 Y142.493 E.0106
G1 X102.675 Y142.48 E.00544
G1 X99.595 Y141.909 E.11958
G3 X99.527 Y143.615 I-13.266 J.327 E.0652
G1 X156.235 Y143.615 E2.16507
G1 X156.415 Y142.65 E.03745
G3 X144.704 Y131.953 I29.416 J-43.961 E.60784
G3 X136.912 Y119.083 I40.991 J-33.614 E.57636
G1 X136.092 Y119.382 E.0333
G3 X132.721 Y119.3 I-1.547 J-5.732 E.13053
G1 X131.942 Y118.975 E.03223
G1 X131.218 Y118.539 E.03228
G1 X130.562 Y117.998 E.03245
G3 X129.081 Y116.142 I12.089 J-11.168 E.09073
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-42.773 J-33.307 E.3043
G1 X122.612 Y120.333 E.07459
G1 X122.773 Y120.655 E.01372
G1 X122.786 Y120.947 E.01117
G1 X122.689 Y121.224 E.01121
G1 X122.639 Y121.284 E.00298
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00555
G1 X116.81 Y123.176 E.04903
; LINE_WIDTH: 0.521316
G1 X116.411 Y122.702 E.01968
; LINE_WIDTH: 0.556496
G1 X116.394 Y122.378 E.01104
G1 X115.476 Y123.015 E.03802
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.116 J-43.088 E.58495
G1 X98.942 Y131.544 E.01602
G1 X99.038 Y132.523 E.03116
G1 X99.041 Y133.346 E.02605
G1 X102.775 Y132.655 E.12023
G1 X102.899 Y132.667 E.00394
G1 X103.035 Y132.822 E.00654
G3 X103.042 Y135.714 I-241.423 J2.01 E.09158
G1 X103.042 Y141.714 E.18996
G1 X102.986 Y141.863 E.00502
G1 X102.816 Y141.94 E.00591
G1 X102.775 Y141.936 E.00131
G1 X99.042 Y141.245 E.1202
G3 X99.009 Y143.303 I-21.454 J.689 E.06521
G2 X98.936 Y144.167 I3.532 J.735 E.02752
G1 X156.695 Y144.167 E1.82866
G1 X156.959 Y142.745 E.04581
G1 X157.025 Y142.391 E.0114
G3 X145.128 Y131.598 I28.627 J-43.51 E.51061
G3 X137.192 Y118.315 I40.271 J-33.072 E.4917
G1 X136.612 Y118.616 E.02069
G1 X135.912 Y118.859 E.02346
G1 X135.135 Y119.01 E.02506
G1 X134.369 Y119.041 E.02427
G1 X133.609 Y118.959 E.0242
G1 X132.868 Y118.767 E.02426
G1 X132.16 Y118.467 E.02434
G1 X131.508 Y118.068 E.0242
G1 X130.918 Y117.575 E.02434
G1 X130.41 Y117.004 E.02419
G3 X129.087 Y115.222 I543.136 J-404.608 E.07027
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-42.345 J-31.987 E.27952
; LINE_WIDTH: 0.554636
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.851 E.00539
G1 X122.202 Y120.926 E.0027
; WIPE_START
M204 S10000
G1 X121.44 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z4.76 F36000
G1 Z4.36
G1 E.4 F1800
; LINE_WIDTH: 0.556496
G1 F3600
M204 S5000
G1 X116.687 Y122.143 E.0128
; LINE_WIDTH: 0.544336
G1 X120.198 Y119.2 E.15232
; LINE_WIDTH: 0.554636
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.198 Y119.2 E-.13045
G1 X119.695 Y119.622 E-.24955
; WIPE_END
G1 E-.02 F1800
G1 X113.238 Y123.692 Z4.76 F36000
G1 X100.189 Y131.917 Z4.76
G1 Z4.36
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.773923
G1 F10647.84
G1 X100.503 Y131.839 E.01563
; LINE_WIDTH: 0.81309
G1 F10112.317
G1 X100.818 Y131.761 E.01646
; LINE_WIDTH: 0.852256
G1 F9628.082
G1 X101.133 Y131.683 E.01729
; LINE_WIDTH: 0.856456
G1 F9578.895
G1 X101.164 Y131.675 E.00171
; LINE_WIDTH: 0.902606
G1 F9069.761
G1 X101.478 Y131.593 E.01836
; LINE_WIDTH: 0.948756
G1 F8612.017
G1 X101.792 Y131.512 E.01934
; LINE_WIDTH: 0.994906
G1 F8198.258
G1 X102.106 Y131.43 E.02031
; LINE_WIDTH: 1.04106
G1 F7822.434
G1 X102.419 Y131.349 E.02129
; LINE_WIDTH: 1.0591
G1 F7684.726
G1 X102.528 Y131.319 E.00751
; WIPE_START
G1 X102.419 Y131.349 E-.0427
G1 X102.106 Y131.43 E-.12327
G1 X101.792 Y131.512 E-.12327
G1 X101.56 Y131.572 E-.09076
; WIPE_END
G1 E-.02 F1800
G1 X100.753 Y139.162 Z4.76 F36000
G1 X100.361 Y142.84 Z4.76
G1 Z4.36
G1 E.4 F1800
; LINE_WIDTH: 0.998776
G1 F8165.361
G1 X100.579 Y142.859 E.01373
; LINE_WIDTH: 0.958826
G1 F8518.211
G1 X100.828 Y142.882 E.01507
; LINE_WIDTH: 0.91311
G1 F8961.351
G1 X101.077 Y142.905 E.01432
; LINE_WIDTH: 0.867394
G1 F9453.13
G1 X101.326 Y142.928 E.01358
; LINE_WIDTH: 0.821678
G1 F10002.017
G1 X101.575 Y142.951 E.01283
; LINE_WIDTH: 0.775961
G1 F10618.574
G1 X101.824 Y142.974 E.01209
; LINE_WIDTH: 0.730245
G1 F11316.14
G1 X102.073 Y142.997 E.01134
; LINE_WIDTH: 0.684529
G1 F12111.798
G1 X102.321 Y143.019 E.0106
; LINE_WIDTH: 0.638812
G1 F13027.809
G1 X102.57 Y143.042 E.00985
; LINE_WIDTH: 0.593096
G1 F14093.708
G1 X103.05 Y143.043 E.01745
; WIPE_START
G1 X102.57 Y143.042 E-.18207
G1 X102.321 Y143.019 E-.095
G1 X102.073 Y142.997 E-.095
G1 X102.052 Y142.995 E-.00793
; WIPE_END
G1 E-.02 F1800
G1 X105.264 Y139.764 Z4.76 F36000
G1 Z4.36
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.264 Y137.421 E.08944
G1 X105.372 Y137.391 E.00429
G3 X109.142 Y138.705 I.769 J3.862 E.15986
G2 X111.028 Y140.48 I8.681 J-7.33 E.09908
G2 X115.269 Y140.998 I2.634 J-3.943 E.16924
G2 X116.683 Y139.921 I-2.411 J-4.632 E.06819
G3 X118.568 Y138.146 I8.681 J7.33 E.09908
G3 X122.81 Y137.628 I2.634 J3.943 E.16924
G3 X124.224 Y138.705 I-2.411 J4.632 E.06819
G2 X126.109 Y140.48 I8.681 J-7.33 E.09908
G2 X130.351 Y140.998 I2.634 J-3.943 E.16924
G2 X131.764 Y139.921 I-2.411 J-4.632 E.06819
G3 X133.65 Y138.146 I8.681 J7.33 E.09908
G3 X137.891 Y137.628 I2.634 J3.943 E.16924
G3 X139.305 Y138.705 I-2.411 J4.632 E.06819
G2 X141.19 Y140.48 I8.681 J-7.33 E.09908
G2 X145.432 Y140.998 I2.634 J-3.943 E.16924
G2 X146.846 Y139.921 I-2.411 J-4.632 E.06819
G3 X148.417 Y138.408 I7.384 J6.094 E.08349
G3 X143.809 Y133.49 I42.041 J-44.022 E.25744
G3 X139.305 Y132.381 I-1.472 J-3.719 E.18891
G2 X137.42 Y130.606 I-8.681 J7.33 E.09908
G2 X133.178 Y130.087 I-2.634 J3.943 E.16924
G2 X131.764 Y131.164 I2.411 J4.632 E.06819
M73 P61 R7
G3 X129.879 Y132.939 I-8.681 J-7.33 E.09908
G3 X125.638 Y133.458 I-2.634 J-3.943 E.16924
G3 X124.224 Y132.381 I2.411 J-4.632 E.06819
G2 X122.339 Y130.606 I-8.681 J7.33 E.09908
G2 X118.097 Y130.087 I-2.634 J3.943 E.16924
G2 X116.683 Y131.164 I2.411 J4.632 E.06819
G3 X114.798 Y132.939 I-8.681 J-7.33 E.09908
G3 X110.556 Y133.458 I-2.634 J-3.943 E.16924
G3 X109.142 Y132.381 I2.411 J-4.632 E.06819
G2 X106.987 Y130.464 I-7.437 J6.194 E.11054
G2 X115.269 Y125.917 I-32.544 J-69.094 E.36097
G2 X115.891 Y125.539 I-.952 J-2.269 E.02789
G2 X117.859 Y126.654 I2.155 J-1.508 E.08927
G1 X130.414 Y119.999 F36000
G1 F13446.283
G3 X128.702 Y118.416 I5.528 J-7.698 E.08925
G3 X124.742 Y117.788 I-1.446 J-3.68 E.16052
G1 X123.628 Y118.947 E.06137
G3 X123.379 Y122.884 I-1.733 J1.867 E.17227
G3 X125.166 Y124.613 I-7.682 J9.73 E.09508
G2 X127.052 Y125.892 I3.895 J-3.711 E.08764
G2 X130.822 Y125.655 I1.642 J-3.987 E.14929
G2 X132.707 Y123.851 I-7.497 J-9.722 E.09981
G3 X136.477 Y122.238 I3.737 J3.522 E.16106
G3 X137.584 Y124.331 I-26.874 J15.549 E.09042
; CHANGE_LAYER
; Z_HEIGHT: 4.52
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.117 Y123.447 E-.38
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
G1 X123.51 Y122.061
G1 Z4.52
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.391 Y122.216 E.00745
G3 X122.296 Y123.143 I-51.291 J-59.467 E.05479
G1 X119.23 Y125.712 E.15272
G1 X118.874 Y125.95 E.01636
G1 X118.45 Y126.11 E.01728
G1 X117.978 Y126.168 E.01818
G1 X117.493 Y126.107 E.01866
G1 X117.195 Y126.004 E.01201
G1 X116.82 Y125.787 E.01657
G3 X115.927 Y124.807 I5.233 J-5.664 E.05069
G3 X103.794 Y131.245 I-31.887 J-45.439 E.52575
G1 X104.286 Y131.59 E.02293
G1 X104.69 Y132.328 E.03215
G1 X104.768 Y132.874 E.02105
G1 X104.768 Y141.717 E.3376
G1 X104.673 Y142.318 E.02324
G1 X104.602 Y142.443 E.0055
G1 X154.069 Y142.443 E1.88859
G3 X143.8 Y132.698 I31.813 J-43.805 E.54205
G3 X136.271 Y120.545 I42.038 J-34.453 E.5474
G1 X135.452 Y120.705 E.03186
G3 X133.301 Y120.656 I-.89 J-8.198 E.08239
G1 X132.428 Y120.434 E.03437
G1 X131.481 Y120.052 E.03899
G1 X130.603 Y119.536 E.03888
G1 X129.808 Y118.894 E.03902
G1 X129.118 Y118.146 E.03884
G3 X127.853 Y116.453 I77.358 J-59.115 E.0807
G1 X126.662 Y114.847 E.07636
G3 X122.955 Y118.919 I-42.476 J-34.946 E.21032
G3 X123.665 Y119.794 I-6.114 J5.689 E.04306
G1 X123.913 Y120.386 E.02451
G1 X123.965 Y120.832 E.01716
G1 X123.893 Y121.362 E.02041
G1 X123.702 Y121.813 E.01869
G1 X123.565 Y121.99 E.00855
G1 X123.013 Y121.746 F36000
G1 F13446.369
G1 X122.892 Y121.879 E.00687
G1 X118.854 Y125.264 E.20115
G1 X118.51 Y125.474 E.01539
G1 X118.094 Y125.577 E.01636
G1 X117.639 Y125.539 E.01746
G1 X117.188 Y125.33 E.01896
G1 X116.82 Y124.961 E.01991
G1 X116.03 Y124.018 E.04698
G3 X105.33 Y129.999 I-32.056 J-44.776 E.46897
G1 X104.06 Y130.506 E.05222
G1 X103.688 Y130.653 E.01527
G1 F12135.011
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.668601
G1 F10808.15
G1 X103.229 Y130.858 E.00432
; LINE_WIDTH: 0.717205
G1 F10474.031
G1 X103.142 Y130.915 E.00465
; LINE_WIDTH: 0.76581
G1 F10145.115
G1 X103.054 Y130.973 E.00499
; LINE_WIDTH: 0.814414
G1 F9821.463
G1 X102.967 Y131.03 E.00532
; LINE_WIDTH: 0.863018
G1 F9503.04
G1 X102.88 Y131.088 E.00565
; LINE_WIDTH: 0.911623
G1 F8976.542
G1 X102.792 Y131.145 E.00598
; LINE_WIDTH: 0.960227
G1 F8505.321
G1 X102.705 Y131.203 E.00631
; LINE_WIDTH: 1.00883
G1 F8081.104
G1 X102.618 Y131.26 E.00664
; LINE_WIDTH: 1.05744
G1 F7697.195
G1 X102.53 Y131.318 E.00697
G1 X102.615 Y131.348 E.00599
; LINE_WIDTH: 1.00883
G1 F8081.104
G1 X102.699 Y131.379 E.00571
; LINE_WIDTH: 0.960227
G1 F8505.321
G1 X102.784 Y131.41 E.00542
; LINE_WIDTH: 0.911623
G1 F8976.542
G1 X102.868 Y131.441 E.00514
; LINE_WIDTH: 0.863018
G1 F9503.04
G1 X102.953 Y131.471 E.00486
; LINE_WIDTH: 0.814414
G1 F10095.147
G1 X103.037 Y131.502 E.00457
; LINE_WIDTH: 0.76581
G1 F10376.869
G1 X103.121 Y131.533 E.00429
; LINE_WIDTH: 0.717205
G1 F10662.469
G1 X103.206 Y131.563 E.004
; LINE_WIDTH: 0.668601
G1 F10951.945
G1 X103.29 Y131.594 E.00372
; LINE_WIDTH: 0.619996
G1 F12287.349
G1 X103.62 Y131.821 E.01527
G1 F13243.182
G1 X103.845 Y131.976 E.01043
G1 F13446.369
G1 X104.128 Y132.492 E.02249
G1 X104.183 Y132.976 E.01858
G1 X104.183 Y141.717 E.33372
G1 X104.116 Y142.137 E.01626
G1 X103.846 Y142.614 E.02094
G1 X103.294 Y142.986 E.02539
; LINE_WIDTH: 0.601716
G1 F13879.588
G1 X103.051 Y143.045 E.00923
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.84 J83.497 E.10298
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.252 Y132.326 I29.804 J-43.953 E.59964
G3 X136.596 Y119.828 I41.465 J-33.997 E.56132
G1 X136.067 Y119.995 E.02117
G1 X135.345 Y120.129 E.02805
G1 X134.414 Y120.179 E.03559
G3 X133.285 Y120.054 I.632 J-10.88 E.04339
G1 X132.568 Y119.865 E.02829
G1 X131.712 Y119.513 E.03534
G1 X130.911 Y119.037 E.03559
G1 X130.185 Y118.446 E.03573
G3 X129.066 Y117.105 I8.523 J-8.25 E.06672
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-42.739 J-34.23 E.25726
G1 X123.061 Y119.957 E.05366
G1 X123.303 Y120.384 E.01872
G1 X123.38 Y120.833 E.01739
G1 X123.329 Y121.203 E.01428
G1 X123.137 Y121.61 E.01718
G1 X123.074 Y121.68 E.00358
G1 X122.581 Y121.353 F36000
G1 F13446.369
G1 X122.516 Y121.43 E.00385
G1 X118.478 Y124.815 E.20115
G1 X118.225 Y124.956 E.01106
G1 X117.92 Y124.994 E.01176
G1 X117.665 Y124.931 E.01
G1 X117.381 Y124.718 E.01356
G1 X116.128 Y123.223 E.0745
G3 X99.509 Y131.446 I-31.739 J-43.235 E.7115
G3 X99.594 Y132.68 I-23.475 J2.239 E.04719
G1 X102.677 Y132.109 E.1197
G1 X103.101 Y132.149 E.01628
G1 X103.415 Y132.373 E.01473
G1 X103.57 Y132.672 E.01285
G3 X103.597 Y135.717 I-72.631 J2.159 E.11626
G1 X103.597 Y141.717 E.22907
G1 X103.559 Y141.957 E.00928
G1 X103.405 Y142.229 E.01195
G1 X103.092 Y142.446 E.01454
G1 X102.818 Y142.495 E.0106
G1 X102.677 Y142.482 E.00544
G1 X99.597 Y141.912 E.11958
G3 X99.528 Y143.615 I-13.01 J.326 E.06511
G1 X156.235 Y143.615 E2.16504
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.704 Y131.953 I29.326 J-43.873 E.60779
G3 X136.912 Y119.083 I41.031 J-33.638 E.57636
G1 X136.092 Y119.382 E.03331
G3 X132.721 Y119.3 I-1.548 J-5.688 E.13055
G1 X131.943 Y118.975 E.03221
G1 X131.218 Y118.538 E.03231
G1 X130.562 Y117.998 E.03244
G1 X129.996 Y117.37 E.03227
G3 X129.081 Y116.142 I119.83 J-90.197 E.05846
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-42.501 J-33.061 E.3043
G1 X122.612 Y120.333 E.07459
G1 X122.773 Y120.655 E.01372
G1 X122.786 Y120.947 E.01117
G1 X122.689 Y121.224 E.01121
G1 X122.639 Y121.284 E.00298
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00555
G1 X116.81 Y123.176 E.04905
; LINE_WIDTH: 0.521326
G1 X116.411 Y122.702 E.01966
; LINE_WIDTH: 0.556496
G1 X116.394 Y122.378 E.01104
G1 X115.477 Y123.015 E.038
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.117 J-43.089 E.58496
G1 X98.942 Y131.543 E.01601
G1 X99.04 Y132.523 E.03117
G1 X99.043 Y133.344 E.02598
G1 X102.777 Y132.652 E.12023
G1 X102.9 Y132.664 E.00391
G1 X103.035 Y132.811 E.00631
G3 X103.044 Y135.717 I-182.761 J2.02 E.092
G1 X103.044 Y141.717 E.18996
G1 X102.988 Y141.865 E.00502
G1 X102.818 Y141.942 E.00591
G1 X102.777 Y141.939 E.00131
G1 X99.044 Y141.247 E.1202
G3 X99.011 Y143.302 I-21.041 J.688 E.06509
G2 X98.936 Y144.167 I3.474 J.737 E.02757
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.617 J-43.502 E.5106
G3 X137.192 Y118.315 I40.733 J-33.348 E.49165
G1 X136.612 Y118.616 E.02069
G1 X135.912 Y118.859 E.02346
G1 X135.136 Y119.01 E.02504
G1 X134.371 Y119.041 E.02422
G1 X133.607 Y118.959 E.02434
G1 X132.867 Y118.767 E.0242
G1 X132.16 Y118.467 E.02432
G1 X131.507 Y118.068 E.02422
G1 X130.918 Y117.575 E.02433
G1 X130.41 Y117.004 E.02419
G3 X129.087 Y115.222 I548.101 J-408.29 E.07025
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-42.132 J-31.797 E.27953
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.851 E.00539
G1 X122.202 Y120.927 E.0027
; WIPE_START
M204 S10000
G1 X121.44 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z4.92 F36000
G1 Z4.52
G1 E.4 F1800
; LINE_WIDTH: 0.556496
G1 F3600
M204 S5000
G1 X116.687 Y122.143 E.0128
; LINE_WIDTH: 0.544336
G1 X120.199 Y119.2 E.15232
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.199 Y119.2 E-.13038
G1 X119.695 Y119.622 E-.24962
; WIPE_END
G1 E-.02 F1800
G1 X113.238 Y123.692 Z4.92 F36000
G1 X100.189 Y131.916 Z4.92
G1 Z4.52
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.77185
G1 F10677.773
G1 X100.503 Y131.838 E.01558
; LINE_WIDTH: 0.811003
G1 F10139.486
G1 X100.818 Y131.76 E.01641
; LINE_WIDTH: 0.850156
G1 F9652.865
G1 X101.132 Y131.682 E.01723
; LINE_WIDTH: 0.854356
G1 F9603.425
G1 X101.163 Y131.674 E.0017
; LINE_WIDTH: 0.900516
G1 F9091.645
G1 X101.477 Y131.592 E.01832
; LINE_WIDTH: 0.946676
G1 F8631.651
G1 X101.791 Y131.511 E.0193
; LINE_WIDTH: 0.992836
G1 F8215.963
G1 X102.105 Y131.429 E.02027
; LINE_WIDTH: 1.039
G1 F7838.472
G1 X102.419 Y131.348 E.02125
; LINE_WIDTH: 1.05744
G1 F7697.195
G1 X102.53 Y131.318 E.00766
; WIPE_START
G1 X102.419 Y131.348 E-.04362
G1 X102.105 Y131.429 E-.12329
G1 X101.791 Y131.511 E-.1233
G1 X101.563 Y131.57 E-.08979
; WIPE_END
G1 E-.02 F1800
G1 X100.754 Y139.16 Z4.92 F36000
G1 X100.362 Y142.841 Z4.92
G1 Z4.52
G1 E.4 F1800
; LINE_WIDTH: 0.996516
G1 F8184.54
G1 X100.581 Y142.861 E.0138
; LINE_WIDTH: 0.956286
G1 F8541.679
G1 X100.83 Y142.884 E.01503
; LINE_WIDTH: 0.91057
G1 F8987.328
G1 X101.079 Y142.906 E.01428
; LINE_WIDTH: 0.864854
G1 F9482.041
G1 X101.328 Y142.929 E.01354
; LINE_WIDTH: 0.819138
G1 F10034.388
G1 X101.577 Y142.952 E.01279
; LINE_WIDTH: 0.773421
G1 F10655.068
G1 X101.826 Y142.975 E.01205
; LINE_WIDTH: 0.727705
G1 F11357.594
G1 X102.075 Y142.998 E.0113
; LINE_WIDTH: 0.681989
G1 F12159.299
G1 X102.324 Y143.021 E.01055
; LINE_WIDTH: 0.636272
G1 F13082.782
G1 X102.573 Y143.044 E.00981
; LINE_WIDTH: 0.590556
G1 F14158.068
G1 X103.051 Y143.045 E.01736
; WIPE_START
G1 X102.573 Y143.044 E-.1819
G1 X102.324 Y143.021 E-.095
G1 X102.075 Y142.998 E-.095
G1 X102.054 Y142.996 E-.0081
; WIPE_END
G1 E-.02 F1800
G1 X105.266 Y139.746 Z4.92 F36000
G1 Z4.52
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.266 Y137.403 E.08944
G1 X105.372 Y137.368 E.00426
G3 X109.142 Y138.524 I.956 J3.61 E.15861
G2 X111.028 Y140.405 I10.357 J-8.497 E.10183
G2 X115.269 Y141.143 I2.864 J-3.898 E.17035
G2 X116.683 Y140.102 I-1.902 J-4.064 E.06747
G3 X118.568 Y138.222 I10.357 J8.497 E.10183
G3 X122.81 Y137.483 I2.864 J3.898 E.17035
G3 X124.224 Y138.524 I-1.902 J4.065 E.06747
G2 X126.109 Y140.405 I10.357 J-8.497 E.10183
G2 X130.351 Y141.143 I2.864 J-3.898 E.17035
G2 X131.764 Y140.102 I-1.902 J-4.065 E.06747
G3 X133.65 Y138.222 I10.357 J8.497 E.10183
G3 X137.891 Y137.483 I2.864 J3.898 E.17035
G3 X139.305 Y138.524 I-1.902 J4.065 E.06747
G2 X141.19 Y140.405 I10.357 J-8.497 E.10183
G2 X145.432 Y141.143 I2.864 J-3.898 E.17035
G2 X146.846 Y140.102 I-1.902 J-4.065 E.06747
G3 X148.47 Y138.456 I9.052 J7.306 E.08842
G3 X143.8 Y133.48 I44.751 J-46.671 E.26065
G3 X140.248 Y133.363 I-1.652 J-3.84 E.14017
G3 X138.362 Y131.528 I5.751 J-7.793 E.10077
G2 X133.65 Y129.796 I-3.934 J3.427 E.19999
G2 X131.764 Y130.983 I1.066 J3.784 E.08624
G3 X129.879 Y132.864 I-10.357 J-8.496 E.10183
G3 X125.638 Y133.603 I-2.864 J-3.898 E.17035
G3 X124.224 Y132.562 I1.902 J-4.065 E.06746
G2 X122.339 Y130.681 I-10.357 J8.497 E.10183
G2 X118.097 Y129.942 I-2.864 J3.898 E.17035
G2 X116.683 Y130.983 I1.902 J4.065 E.06746
G3 X114.798 Y132.864 I-10.357 J-8.496 E.10183
G3 X110.556 Y133.603 I-2.864 J-3.898 E.17035
G3 X109.142 Y132.562 I1.902 J-4.065 E.06746
G2 X106.921 Y130.486 I-8.098 J6.443 E.1165
G2 X114.798 Y126.208 I-24.071 J-53.711 E.34257
G2 X115.98 Y125.645 I-.808 J-3.221 E.05032
G2 X117.997 Y126.665 I2.037 J-1.525 E.08938
G1 X130.404 Y119.993 F36000
G1 F13446.283
G3 X128.694 Y118.407 I5.591 J-7.744 E.08926
G3 X127.052 Y118.754 I-1.717 J-4.068 E.06448
G3 X124.645 Y117.885 I-.141 J-3.378 E.10018
G1 X123.628 Y118.947 E.05612
G3 X123.487 Y122.793 I-1.722 J1.863 E.16684
G3 X125.166 Y124.476 I-6.252 J7.916 E.09097
G2 X127.052 Y125.868 I4.246 J-3.778 E.09011
G2 X130.822 Y125.823 I1.839 J-3.872 E.14906
G2 X132.707 Y123.987 I-5.752 J-7.793 E.10077
G3 X136.477 Y122.169 I4.029 J3.536 E.16418
G2 X137.545 Y124.26 I252.201 J-127.453 E.08964
; CHANGE_LAYER
; Z_HEIGHT: 4.68
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.09 Y123.37 E-.38
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
G1 X123.51 Y122.061
G1 Z4.68
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.391 Y122.215 E.00744
G3 X122.296 Y123.143 I-51.039 J-59.164 E.05481
G1 X119.23 Y125.712 E.15272
G1 X118.874 Y125.95 E.01636
G1 X118.45 Y126.11 E.01728
G1 X117.978 Y126.168 E.01818
G1 X117.493 Y126.107 E.01866
G1 X117.195 Y126.004 E.01201
G1 X116.82 Y125.787 E.01657
G3 X115.927 Y124.807 I5.234 J-5.665 E.05069
G3 X103.797 Y131.244 I-31.887 J-45.438 E.52561
G1 X104.287 Y131.586 E.02281
G1 X104.691 Y132.323 E.03208
G1 X104.77 Y132.872 E.02117
G1 X104.77 Y141.719 E.3378
G1 X104.675 Y142.321 E.02324
G1 X104.606 Y142.443 E.00538
G1 X154.069 Y142.443 E1.88846
G3 X143.803 Y132.701 I31.794 J-43.784 E.54191
G3 X136.271 Y120.545 I42.033 J-34.455 E.54755
G1 X135.451 Y120.705 E.03188
G3 X133.304 Y120.657 I-.889 J-8.197 E.08224
G1 X132.426 Y120.433 E.03457
G1 X131.482 Y120.052 E.0389
G1 X130.602 Y119.535 E.03895
G1 X129.808 Y118.894 E.03895
G1 X129.117 Y118.145 E.0389
G3 X127.853 Y116.453 I77.387 J-59.133 E.08063
G1 X126.662 Y114.847 E.07636
G3 X122.955 Y118.919 I-42.488 J-34.956 E.21032
G3 X123.665 Y119.794 I-6.11 J5.685 E.04306
G1 X123.913 Y120.385 E.02449
G1 X123.965 Y120.832 E.01717
G1 X123.893 Y121.361 E.02038
G1 X123.702 Y121.812 E.01872
G1 X123.565 Y121.99 E.00855
G1 X123.013 Y121.746 F36000
G1 F13446.369
G1 X122.892 Y121.879 E.00687
G1 X118.854 Y125.264 E.20115
G1 X118.51 Y125.474 E.01539
G1 X118.094 Y125.577 E.01636
G1 X117.639 Y125.539 E.01746
G1 X117.188 Y125.33 E.01896
G1 X116.82 Y124.961 E.01991
G1 X116.03 Y124.018 E.04698
G3 X105.326 Y130.001 I-32.055 J-44.775 E.46914
G1 X104.06 Y130.506 E.05204
G1 X103.688 Y130.653 E.01527
G1 F12142.858
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.668414
G1 F10815.556
G1 X103.229 Y130.858 E.00431
; LINE_WIDTH: 0.716832
G1 F10482.316
G1 X103.142 Y130.915 E.00464
; LINE_WIDTH: 0.76525
G1 F10154.246
G1 X103.055 Y130.972 E.00497
; LINE_WIDTH: 0.813667
G1 F9831.393
G1 X102.968 Y131.03 E.0053
; LINE_WIDTH: 0.862085
G1 F9513.755
G1 X102.881 Y131.087 E.00562
; LINE_WIDTH: 0.910503
G1 F8988.017
G1 X102.794 Y131.144 E.00595
; LINE_WIDTH: 0.958921
G1 F8517.341
G1 X102.707 Y131.201 E.00628
; LINE_WIDTH: 1.00734
G1 F8093.508
G1 X102.62 Y131.259 E.00661
; LINE_WIDTH: 1.05576
G1 F7709.855
G1 X102.532 Y131.316 E.00694
G1 X102.617 Y131.347 E.00597
; LINE_WIDTH: 1.00734
G1 F8093.508
G1 X102.701 Y131.377 E.00569
; LINE_WIDTH: 0.958921
G1 F8517.341
G1 X102.785 Y131.408 E.0054
; LINE_WIDTH: 0.910503
G1 F8988.017
G1 X102.87 Y131.438 E.00512
; LINE_WIDTH: 0.862085
G1 F9513.755
G1 X102.954 Y131.469 E.00484
; LINE_WIDTH: 0.813667
G1 F10104.82
G1 X103.038 Y131.499 E.00455
; LINE_WIDTH: 0.76525
G1 F10385.951
G1 X103.122 Y131.53 E.00427
; LINE_WIDTH: 0.716832
G1 F10670.928
G1 X103.207 Y131.56 E.00399
; LINE_WIDTH: 0.668414
G1 F10959.742
G1 X103.291 Y131.591 E.00371
; LINE_WIDTH: 0.619996
G1 F12295.608
G1 X103.621 Y131.817 E.01527
G1 F13254.067
G1 X103.846 Y131.972 E.01045
G1 F13446.369
G1 X104.13 Y132.488 E.02245
G1 X104.185 Y132.972 E.01863
G1 X104.185 Y141.719 E.33395
G1 X104.118 Y142.14 E.01626
G1 X103.848 Y142.617 E.02094
G1 X103.296 Y142.987 E.02538
; LINE_WIDTH: 0.600236
G1 F13915.888
G1 X103.053 Y143.046 E.0092
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.839 J77.226 E.10291
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.327 I30.046 J-44.216 E.59951
G3 X136.596 Y119.828 I41.461 J-33.998 E.56142
G1 X136.064 Y119.995 E.02126
G1 X135.344 Y120.129 E.02798
G1 X134.413 Y120.179 E.03558
G3 X133.287 Y120.054 I.626 J-10.824 E.04329
G1 X132.566 Y119.865 E.02846
G1 X131.712 Y119.513 E.03525
G1 X130.909 Y119.036 E.03567
G1 X130.185 Y118.446 E.03566
G3 X129.066 Y117.105 I8.526 J-8.252 E.06672
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-42.751 J-34.242 E.25726
G1 X123.061 Y119.957 E.05366
G1 X123.303 Y120.384 E.01873
G1 X123.38 Y120.832 E.01738
G1 X123.329 Y121.202 E.01426
G1 X123.137 Y121.61 E.0172
G1 X123.074 Y121.68 E.00359
G1 X122.581 Y121.353 F36000
G1 F13446.369
G1 X122.516 Y121.43 E.00385
G1 X118.478 Y124.815 E.20115
G1 X118.225 Y124.956 E.01106
G1 X117.92 Y124.994 E.01176
G1 X117.665 Y124.931 E.01
G1 X117.381 Y124.718 E.01356
G1 X116.128 Y123.223 E.0745
G3 X99.507 Y131.447 I-31.739 J-43.236 E.71156
G3 X99.596 Y132.677 I-21.235 J2.146 E.04709
G1 X102.679 Y132.106 E.11971
G1 X103.102 Y132.146 E.01624
G1 X103.416 Y132.37 E.01474
G1 X103.572 Y132.668 E.01283
G3 X103.599 Y135.719 I-71.881 J2.162 E.1165
G1 X103.599 Y141.719 E.22907
G1 X103.561 Y141.959 E.00928
G1 X103.407 Y142.232 E.01195
G1 X103.094 Y142.448 E.01454
G1 X102.821 Y142.498 E.0106
G1 X102.679 Y142.485 E.00543
G1 X99.599 Y141.914 E.11958
G3 X99.529 Y143.615 I-12.765 J.325 E.06501
G1 X156.235 Y143.615 E2.165
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.318 J-43.864 E.60775
G3 X136.912 Y119.083 I40.794 J-33.496 E.57641
G1 X136.093 Y119.382 E.03328
G3 X132.719 Y119.299 I-1.548 J-5.69 E.13066
G1 X131.943 Y118.975 E.03212
M73 P62 R7
G1 X131.216 Y118.537 E.03239
G1 X130.562 Y117.998 E.03237
G3 X129.081 Y116.142 I12.11 J-11.184 E.09072
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-42.508 J-33.067 E.3043
G1 X122.612 Y120.333 E.07459
G1 X122.773 Y120.655 E.01371
G1 X122.786 Y120.947 E.01118
G1 X122.689 Y121.224 E.01121
G1 X122.639 Y121.284 E.00298
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00555
G1 X116.81 Y123.176 E.04905
; LINE_WIDTH: 0.521326
G1 X116.411 Y122.702 E.01966
; LINE_WIDTH: 0.556496
G1 X116.394 Y122.378 E.01104
G1 X115.477 Y123.015 E.038
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.117 J-43.088 E.58496
G1 X98.942 Y131.543 E.01599
G1 X99.042 Y132.523 E.03118
G1 X99.045 Y133.341 E.02592
G1 X102.779 Y132.65 E.12023
G1 X102.902 Y132.661 E.0039
G1 X103.037 Y132.808 E.00631
G3 X103.046 Y135.719 I-180.993 J2.023 E.09217
G1 X103.046 Y141.719 E.18996
G1 X102.99 Y141.868 E.00502
G1 X102.821 Y141.945 E.00591
G1 X102.779 Y141.941 E.00131
G1 X99.046 Y141.25 E.1202
G3 X99.012 Y143.301 I-20.64 J.687 E.06498
G2 X98.936 Y144.167 I3.42 J.739 E.02761
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.615 J-43.499 E.51063
G3 X137.192 Y118.315 I40.732 J-33.347 E.49163
G1 X136.613 Y118.615 E.02065
G1 X135.913 Y118.859 E.02348
G1 X135.135 Y119.01 E.02507
G1 X134.371 Y119.041 E.02422
G1 X133.609 Y118.959 E.02425
G1 X132.865 Y118.766 E.02435
G1 X132.161 Y118.467 E.02423
G1 X131.506 Y118.067 E.02429
G1 X130.918 Y117.575 E.02427
G1 X130.409 Y117.003 E.02425
G3 X129.087 Y115.222 I554.454 J-413 E.0702
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-42.136 J-31.801 E.27953
; LINE_WIDTH: 0.554636
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.851 E.00539
G1 X122.202 Y120.926 E.0027
; WIPE_START
M204 S10000
G1 X121.44 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z5.08 F36000
G1 Z4.68
G1 E.4 F1800
; LINE_WIDTH: 0.556496
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.01281
; LINE_WIDTH: 0.544336
G1 X120.199 Y119.2 E.15232
; LINE_WIDTH: 0.554636
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.199 Y119.2 E-.13038
G1 X119.695 Y119.622 E-.24962
; WIPE_END
G1 E-.02 F1800
G1 X113.238 Y123.691 Z5.08 F36000
G1 X100.189 Y131.915 Z5.08
G1 Z4.68
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.769783
G1 F10707.78
G1 X100.503 Y131.837 E.01553
; LINE_WIDTH: 0.80893
G1 F10166.625
G1 X100.818 Y131.759 E.01636
; LINE_WIDTH: 0.848076
G1 F9677.539
G1 X101.132 Y131.681 E.01718
; LINE_WIDTH: 0.852276
G1 F9627.847
G1 X101.163 Y131.673 E.0017
; LINE_WIDTH: 0.898431
G1 F9113.582
G1 X101.477 Y131.591 E.01828
; LINE_WIDTH: 0.944586
G1 F8651.47
G1 X101.791 Y131.51 E.01925
; LINE_WIDTH: 0.990741
G1 F8233.96
G1 X102.105 Y131.428 E.02023
; LINE_WIDTH: 1.0369
G1 F7854.892
G1 X102.419 Y131.347 E.02121
; LINE_WIDTH: 1.05576
G1 F7709.855
G1 X102.532 Y131.316 E.00782
; WIPE_START
G1 X102.419 Y131.347 E-.04463
G1 X102.105 Y131.428 E-.12329
G1 X101.791 Y131.51 E-.12329
G1 X101.565 Y131.569 E-.08878
; WIPE_END
G1 E-.02 F1800
G1 X100.756 Y139.158 Z5.08 F36000
G1 X100.363 Y142.842 Z5.08
G1 Z4.68
G1 E.4 F1800
; LINE_WIDTH: 0.994276
G1 F8203.639
G1 X100.583 Y142.862 E.01386
; LINE_WIDTH: 0.953756
G1 F8565.182
G1 X100.832 Y142.885 E.01498
; LINE_WIDTH: 0.908039
G1 F9013.367
G1 X101.081 Y142.908 E.01424
; LINE_WIDTH: 0.862321
G1 F9511.042
G1 X101.33 Y142.931 E.01349
; LINE_WIDTH: 0.816604
G1 F10066.89
G1 X101.579 Y142.953 E.01275
; LINE_WIDTH: 0.770886
G1 F10691.74
G1 X101.828 Y142.976 E.012
; LINE_WIDTH: 0.725169
G1 F11399.291
G1 X102.077 Y142.999 E.01126
; LINE_WIDTH: 0.679451
G1 F12207.128
G1 X102.326 Y143.022 E.01051
; LINE_WIDTH: 0.633734
G1 F13138.194
G1 X102.575 Y143.045 E.00977
; LINE_WIDTH: 0.588016
G1 F14223.017
G1 X103.053 Y143.046 E.01726
; WIPE_START
G1 X102.575 Y143.045 E-.18173
G1 X102.326 Y143.022 E-.095
G1 X102.077 Y142.999 E-.095
G1 X102.055 Y142.997 E-.00827
; WIPE_END
G1 E-.02 F1800
G1 X105.268 Y139.722 Z5.08 F36000
G1 Z4.68
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.268 Y137.379 E.08944
G1 X105.372 Y137.34 E.00424
G3 X108.2 Y137.542 I1.15 J3.794 E.11069
G3 X110.085 Y139.418 I-4.517 J6.425 E.10202
G2 X114.798 Y141.417 I4.23 J-3.418 E.20351
G2 X116.683 Y140.296 I-.705 J-3.331 E.08525
G3 X118.568 Y138.294 I12.479 J9.861 E.10513
G3 X122.81 Y137.328 I3.116 J3.886 E.17188
G3 X124.224 Y138.33 I-1.459 J3.557 E.06674
G2 X126.109 Y140.332 I12.479 J-9.861 E.10513
G2 X130.351 Y141.298 I3.116 J-3.886 E.17188
G2 X131.764 Y140.296 I-1.459 J-3.558 E.06674
G3 X133.65 Y138.294 I12.479 J9.861 E.10513
G3 X137.891 Y137.328 I3.116 J3.886 E.17188
G3 X139.305 Y138.33 I-1.459 J3.558 E.06674
G2 X141.19 Y140.332 I12.479 J-9.861 E.10513
G2 X145.432 Y141.298 I3.116 J-3.886 E.17188
G2 X146.846 Y140.296 I-1.459 J-3.558 E.06674
G3 X148.522 Y138.497 I11.207 J8.759 E.09401
G3 X143.797 Y133.474 I39.522 J-41.915 E.26343
G3 X140.248 Y133.544 I-1.847 J-3.649 E.1402
G3 X138.362 Y131.668 I4.518 J-6.425 E.10202
G2 X133.65 Y129.668 I-4.23 J3.418 E.20351
G2 X131.764 Y130.789 I.705 J3.331 E.08525
G3 X129.879 Y132.792 I-12.478 J-9.86 E.10513
G3 X125.638 Y133.758 I-3.116 J-3.886 E.17188
G3 X124.224 Y132.756 I1.46 J-3.558 E.06674
G2 X122.339 Y130.753 I-12.48 J9.862 E.10513
G2 X118.097 Y129.787 I-3.116 J3.886 E.17188
G2 X116.683 Y130.789 I1.459 J3.558 E.06674
G3 X114.798 Y132.792 I-12.478 J-9.86 E.10513
G3 X110.556 Y133.758 I-3.116 J-3.886 E.17188
G3 X109.142 Y132.756 I1.46 J-3.558 E.06674
G2 X107.257 Y130.753 I-12.48 J9.861 E.10513
G1 X106.872 Y130.508 E.01745
G2 X114.541 Y126.345 I-25.397 J-55.938 E.33345
G2 X116.079 Y125.764 I-.061 J-2.487 E.06399
G2 X118.15 Y126.652 I1.903 J-1.58 E.08931
G1 X130.399 Y119.99 F36000
G1 F13446.283
G3 X128.69 Y118.403 I5.608 J-7.752 E.08926
G3 X127.052 Y118.83 I-1.949 J-4.122 E.06503
G3 X124.547 Y117.987 I-.361 J-3.068 E.10423
G1 X123.628 Y118.947 E.05075
G3 X123.607 Y122.691 I-1.662 J1.863 E.16165
G3 X125.166 Y124.337 I-5.705 J6.967 E.08677
G2 X127.994 Y126.205 I4.713 J-4.058 E.13106
G2 X130.822 Y126.003 I1.15 J-3.794 E.11069
G2 X132.707 Y124.127 I-4.517 J-6.425 E.10202
G3 X136.453 Y122.098 I4.33 J3.52 E.16691
G2 X137.508 Y124.189 I30.514 J-14.077 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 4.84
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.058 Y123.296 E-.38
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
G1 X123.51 Y122.061
G1 Z4.84
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.391 Y122.216 E.00745
G3 X122.296 Y123.143 I-40.793 J-47.068 E.05479
G1 X119.23 Y125.712 E.15272
G1 X118.874 Y125.95 E.01636
G1 X118.45 Y126.11 E.01728
G1 X117.978 Y126.168 E.01818
G1 X117.493 Y126.107 E.01866
G1 X117.196 Y126.004 E.01201
G1 X116.82 Y125.787 E.01657
G3 X115.927 Y124.807 I5.233 J-5.664 E.05069
G3 X103.786 Y131.247 I-31.533 J-44.777 E.5261
G1 X104.387 Y131.706 E.02886
G1 X104.714 Y132.393 E.02907
G3 X104.772 Y133.722 I-6.465 J.952 E.05085
G1 X104.772 Y141.722 E.30543
G1 X104.677 Y142.323 E.02324
G1 X104.609 Y142.443 E.00527
G1 X154.069 Y142.443 E1.88834
G3 X143.803 Y132.701 I31.69 J-43.677 E.54191
G3 X136.318 Y120.647 I42.321 J-34.63 E.54327
G1 X136.271 Y120.545 E.00427
G3 X133.274 Y120.651 I-1.756 J-7.188 E.11529
G1 X132.429 Y120.434 E.03333
G1 X131.481 Y120.051 E.03903
G1 X130.604 Y119.536 E.03885
G1 X129.808 Y118.894 E.03904
G1 X129.118 Y118.146 E.03884
G3 X127.853 Y116.453 I77.277 J-59.053 E.08069
G1 X126.662 Y114.847 E.07636
G3 X122.955 Y118.919 I-42.507 J-34.974 E.21032
G3 X123.665 Y119.794 I-6.117 J5.691 E.04306
G1 X123.913 Y120.386 E.02451
G1 X123.965 Y120.832 E.01716
G1 X123.893 Y121.362 E.02042
G1 X123.702 Y121.812 E.01868
G1 X123.565 Y121.99 E.00855
G1 X123.013 Y121.746 F36000
G1 F13446.369
G1 X122.892 Y121.879 E.00687
G1 X118.854 Y125.264 E.20115
G1 X118.51 Y125.474 E.01539
G1 X118.094 Y125.577 E.01636
G1 X117.639 Y125.539 E.01746
G1 X117.188 Y125.33 E.01897
G1 X116.82 Y124.961 E.01991
G1 X116.03 Y124.018 E.04698
G3 X104.429 Y130.372 I-31.612 J-43.944 E.50628
G1 X104.066 Y130.512 E.01485
G1 X103.692 Y130.656 E.01527
G1 F12154.739
G1 X103.319 Y130.8 E.01527
; LINE_WIDTH: 0.668232
G1 F10826.768
G1 X103.232 Y130.857 E.00431
; LINE_WIDTH: 0.716467
G1 F10493.317
G1 X103.145 Y130.914 E.00464
; LINE_WIDTH: 0.764703
G1 F10165.08
G1 X103.058 Y130.971 E.00496
; LINE_WIDTH: 0.812938
G1 F9842.043
G1 X102.97 Y131.028 E.00529
; LINE_WIDTH: 0.861174
G1 F9524.238
G1 X102.883 Y131.086 E.00562
; LINE_WIDTH: 0.90941
G1 F8999.246
G1 X102.796 Y131.143 E.00595
; LINE_WIDTH: 0.957645
G1 F8529.107
G1 X102.709 Y131.2 E.00627
; LINE_WIDTH: 1.00588
G1 F8105.651
G1 X102.622 Y131.257 E.0066
; LINE_WIDTH: 1.05412
G1 F7722.254
G1 X102.535 Y131.314 E.00693
G1 X102.619 Y131.345 E.00594
; LINE_WIDTH: 1.00588
G1 F8105.651
G1 X102.703 Y131.375 E.00566
; LINE_WIDTH: 0.957645
G1 F8529.107
G1 X102.787 Y131.406 E.00538
; LINE_WIDTH: 0.90941
G1 F8999.246
G1 X102.871 Y131.436 E.0051
; LINE_WIDTH: 0.861174
G1 F9524.238
G1 X102.955 Y131.466 E.00482
; LINE_WIDTH: 0.812938
G1 F10114.28
G1 X103.039 Y131.497 E.00454
; LINE_WIDTH: 0.764703
G1 F10394.834
G1 X103.123 Y131.527 E.00426
; LINE_WIDTH: 0.716467
G1 F10679.227
G1 X103.207 Y131.558 E.00398
; LINE_WIDTH: 0.668232
G1 F10967.438
G1 X103.291 Y131.588 E.00369
; LINE_WIDTH: 0.619996
G1 F12303.759
G1 X103.612 Y131.827 E.01527
G1 F13446.369
G1 X103.917 Y132.055 E.01454
G1 X104.146 Y132.536 E.02034
G3 X104.187 Y133.722 I-6.209 J.809 E.04535
G1 X104.187 Y141.722 E.30543
G1 X104.12 Y142.142 E.01626
G1 X103.85 Y142.62 E.02094
G1 X103.297 Y142.989 E.02538
; LINE_WIDTH: 0.598766
G1 F13952.13
G1 X103.055 Y143.047 E.00917
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.837 J71.814 E.10284
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I29.896 J-44.054 E.59951
G3 X136.596 Y119.828 I41.769 J-34.187 E.56141
G1 X136.065 Y119.995 E.02126
G1 X135.344 Y120.129 E.02797
G1 X134.411 Y120.179 E.03568
G3 X133.287 Y120.054 I.626 J-10.798 E.0432
G1 X132.568 Y119.865 E.02839
G1 X131.712 Y119.513 E.03535
G1 X130.911 Y119.037 E.03556
G1 X130.185 Y118.446 E.03575
G3 X129.066 Y117.106 I8.519 J-8.246 E.06671
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-42.761 J-34.25 E.25726
G1 X123.061 Y119.957 E.05366
G1 X123.303 Y120.384 E.01873
G1 X123.38 Y120.833 E.01738
G1 X123.329 Y121.203 E.01428
G1 X123.137 Y121.61 E.01718
G1 X123.074 Y121.68 E.00359
G1 X122.581 Y121.353 F36000
G1 F13446.369
G1 X122.516 Y121.43 E.00385
G1 X118.478 Y124.815 E.20115
G1 X118.225 Y124.956 E.01106
G1 X117.92 Y124.994 E.01176
G1 X117.665 Y124.931 E.01
G1 X117.381 Y124.718 E.01356
G1 X116.128 Y123.223 E.0745
G3 X99.506 Y131.447 I-31.738 J-43.235 E.71162
G3 X99.598 Y132.675 I-19.389 J2.07 E.04699
G1 X102.681 Y132.104 E.11971
G1 X103.105 Y132.144 E.01627
G1 X103.457 Y132.418 E.01703
G1 X103.581 Y132.694 E.01157
G3 X103.601 Y135.722 I-94.784 J2.136 E.1156
G1 X103.601 Y141.722 E.22907
G1 X103.563 Y141.962 E.00928
G1 X103.409 Y142.234 E.01195
G1 X103.096 Y142.451 E.01454
G1 X102.823 Y142.5 E.0106
G1 X102.681 Y142.487 E.00544
G1 X99.601 Y141.917 E.11958
G3 X99.53 Y143.615 I-12.532 J.323 E.06492
G1 X156.235 Y143.615 E2.16497
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.705 Y131.954 I29.69 J-44.271 E.60769
G3 X136.912 Y119.083 I41.311 J-33.81 E.57639
G1 X136.092 Y119.382 E.03332
G1 X135.3 Y119.542 E.03085
G1 X134.389 Y119.593 E.03481
G3 X133.436 Y119.488 I1.019 J-13.572 E.03663
G1 X132.721 Y119.3 E.02822
G1 X131.942 Y118.975 E.03223
G1 X131.218 Y118.539 E.03228
G1 X130.562 Y117.998 E.03246
G3 X129.081 Y116.142 I12.1 J-11.176 E.09071
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-42.515 J-33.073 E.3043
G1 X122.612 Y120.333 E.07459
G1 X122.773 Y120.655 E.01372
G1 X122.786 Y120.947 E.01118
G1 X122.689 Y121.224 E.01121
G1 X122.639 Y121.284 E.00298
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00555
G1 X116.81 Y123.176 E.04905
; LINE_WIDTH: 0.521326
G1 X116.411 Y122.702 E.01966
; LINE_WIDTH: 0.556496
G1 X116.394 Y122.378 E.01104
G1 X115.477 Y123.015 E.038
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.117 J-43.088 E.58496
G1 X98.942 Y131.542 E.01598
G1 X99.044 Y132.522 E.03118
G1 X99.047 Y133.339 E.02586
G1 X102.782 Y132.647 E.12023
G1 X102.904 Y132.659 E.00391
G1 X103.041 Y132.814 E.00656
G3 X103.048 Y135.722 I-240.199 J2.016 E.09206
G1 X103.048 Y141.722 E.18996
G1 X102.993 Y141.87 E.00502
G1 X102.823 Y141.947 E.00591
G1 X102.782 Y141.944 E.00131
G1 X99.048 Y141.252 E.1202
G3 X99.014 Y143.3 I-20.269 J.685 E.06486
G2 X98.936 Y144.167 I3.365 J.74 E.02766
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I29.092 J-44.025 E.51056
G3 X137.192 Y118.315 I40.309 J-33.095 E.49168
G1 X136.614 Y118.615 E.02061
G1 X135.912 Y118.859 E.02354
G1 X135.198 Y118.999 E.02301
G1 X134.369 Y119.041 E.02629
G1 X133.609 Y118.959 E.0242
G1 X132.867 Y118.767 E.02426
G1 X132.16 Y118.466 E.02433
G1 X131.508 Y118.068 E.0242
G1 X130.918 Y117.575 E.02435
G1 X130.409 Y117.004 E.0242
G3 X129.087 Y115.222 I546.157 J-406.843 E.07024
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-42.144 J-31.807 E.27953
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.851 E.00539
G1 X122.202 Y120.926 E.0027
; WIPE_START
M204 S10000
G1 X121.44 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z5.24 F36000
G1 Z4.84
G1 E.4 F1800
; LINE_WIDTH: 0.556496
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.01281
; LINE_WIDTH: 0.544336
G1 X120.199 Y119.2 E.15232
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.199 Y119.2 E-.13038
G1 X119.695 Y119.622 E-.24962
; WIPE_END
G1 E-.02 F1800
M73 P63 R7
G1 X113.238 Y123.691 Z5.24 F36000
G1 X100.189 Y131.914 Z5.24
G1 Z4.84
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.767723
G1 F10737.857
G1 X100.503 Y131.836 E.01548
; LINE_WIDTH: 0.80685
G1 F10193.999
G1 X100.818 Y131.758 E.01631
; LINE_WIDTH: 0.845976
G1 F9702.579
G1 X101.132 Y131.68 E.01713
; LINE_WIDTH: 0.850196
G1 F9652.392
G1 X101.163 Y131.672 E.00169
; LINE_WIDTH: 0.896336
G1 F9135.731
G1 X101.477 Y131.59 E.01823
; LINE_WIDTH: 0.942476
G1 F8671.57
G1 X101.791 Y131.509 E.0192
; LINE_WIDTH: 0.988616
G1 F8252.295
G1 X102.105 Y131.427 E.02018
; LINE_WIDTH: 1.03476
G1 F7871.694
G1 X102.418 Y131.346 E.02116
; LINE_WIDTH: 1.05412
G1 F7722.254
G1 X102.535 Y131.314 E.008
; WIPE_START
G1 X102.418 Y131.346 E-.04574
G1 X102.105 Y131.427 E-.12327
G1 X101.791 Y131.509 E-.12327
G1 X101.567 Y131.567 E-.08772
; WIPE_END
G1 E-.02 F1800
G1 X100.757 Y139.156 Z5.24 F36000
G1 X100.364 Y142.843 Z5.24
G1 Z4.84
G1 E.4 F1800
; LINE_WIDTH: 0.992016
G1 F8222.998
G1 X100.586 Y142.863 E.01393
; LINE_WIDTH: 0.951216
G1 F8588.911
G1 X100.835 Y142.886 E.01494
; LINE_WIDTH: 0.905501
G1 F9039.62
G1 X101.084 Y142.909 E.0142
; LINE_WIDTH: 0.859786
G1 F9540.252
G1 X101.333 Y142.932 E.01345
; LINE_WIDTH: 0.814071
G1 F10099.586
G1 X101.582 Y142.955 E.01271
; LINE_WIDTH: 0.768356
G1 F10728.592
G1 X101.831 Y142.978 E.01196
; LINE_WIDTH: 0.722641
G1 F11441.15
G1 X102.079 Y143 E.01122
; LINE_WIDTH: 0.676926
G1 F12255.094
G1 X102.328 Y143.023 E.01047
; LINE_WIDTH: 0.631211
G1 F13193.718
G1 X102.577 Y143.046 E.00973
; LINE_WIDTH: 0.585496
G1 F14288.048
G1 X103.055 Y143.047 E.01717
; WIPE_START
G1 X102.577 Y143.046 E-.18156
G1 X102.328 Y143.023 E-.095
G1 X102.079 Y143 E-.095
G1 X102.057 Y142.998 E-.00845
; WIPE_END
G1 E-.02 F1800
G1 X105.27 Y139.691 Z5.24 F36000
G1 Z4.84
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.27 Y137.348 E.08944
G3 X107.257 Y137.072 I1.818 J5.784 E.07695
G3 X109.142 Y138.118 I-.382 J2.91 E.08428
G2 X111.028 Y140.263 I13.795 J-10.223 E.10915
G2 X115.269 Y141.466 I3.403 J-3.916 E.17386
G2 X116.683 Y140.509 I-1.064 J-3.094 E.06596
G3 X118.568 Y138.364 I13.793 J10.222 E.10915
G3 X122.81 Y137.16 I3.402 J3.916 E.17386
G3 X124.224 Y138.118 I-1.064 J3.094 E.06596
G2 X126.109 Y140.263 I13.795 J-10.223 E.10915
G2 X130.351 Y141.466 I3.403 J-3.916 E.17386
G2 X131.764 Y140.509 I-1.064 J-3.094 E.06596
G3 X133.65 Y138.364 I13.791 J10.22 E.10915
G3 X137.891 Y137.16 I3.402 J3.916 E.17386
G3 X139.305 Y138.118 I-1.064 J3.094 E.06596
G2 X141.19 Y140.263 I13.793 J-10.222 E.10915
G2 X145.432 Y141.466 I3.403 J-3.916 E.17386
G2 X146.846 Y140.509 I-1.064 J-3.094 E.06596
G3 X148.556 Y138.532 I14.242 J10.593 E.0999
G3 X143.796 Y133.473 I42.783 J-45.028 E.26533
G3 X140.248 Y133.74 I-2.059 J-3.666 E.14028
G3 X139.305 Y132.968 I1.62 J-2.941 E.04677
G2 X137.42 Y130.823 I-13.794 J10.223 E.10915
G2 X133.178 Y129.62 I-3.402 J3.916 E.17386
G2 X131.764 Y130.577 I1.064 J3.094 E.06596
G3 X129.879 Y132.722 I-13.796 J-10.224 E.10915
G3 X125.638 Y133.925 I-3.402 J-3.916 E.17386
G3 X124.224 Y132.968 I1.064 J-3.094 E.06596
G2 X122.339 Y130.823 I-13.793 J10.222 E.10915
G2 X118.097 Y129.62 I-3.402 J3.916 E.17386
G2 X116.683 Y130.577 I1.064 J3.094 E.06596
G3 X114.798 Y132.722 I-13.794 J-10.223 E.10915
G3 X110.556 Y133.925 I-3.402 J-3.916 E.17386
G3 X109.142 Y132.968 I1.064 J-3.094 E.06596
G2 X107.257 Y130.823 I-13.793 J10.221 E.10915
G1 X106.841 Y130.533 E.01939
G2 X114.354 Y126.464 I-23.84 J-52.99 E.32652
G2 X116.205 Y125.889 I.245 J-2.479 E.07599
G2 X118.327 Y126.636 I1.819 J-1.78 E.08898
G1 X130.399 Y119.99 F36000
G1 F13446.283
G3 X128.69 Y118.403 I5.63 J-7.775 E.08926
G3 X127.052 Y118.912 I-2.208 J-4.21 E.06587
G3 X124.441 Y118.098 I-.534 J-2.879 E.10862
G1 X123.628 Y118.947 E.04489
G3 X123.731 Y122.565 I-1.648 J1.857 E.15483
G3 X125.166 Y124.19 I-6.784 J7.437 E.08297
G2 X128.937 Y126.453 I4.801 J-3.728 E.17177
G2 X131.293 Y125.885 I.536 J-2.95 E.09527
G2 X132.707 Y124.273 I-6.244 J-6.905 E.08203
G3 X136.409 Y122.026 I4.786 J3.712 E.16907
G2 X137.47 Y124.114 I53.83 J-26.061 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 5
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X137.017 Y123.223 E-.38
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
G1 X123.51 Y122.061
G1 Z5
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.391 Y122.215 E.00744
G3 X122.296 Y123.143 I-50.963 J-59.075 E.05481
G1 X119.23 Y125.712 E.15272
G1 X118.874 Y125.95 E.01636
G1 X118.45 Y126.11 E.01728
G1 X117.978 Y126.168 E.01818
G1 X117.493 Y126.107 E.01866
G1 X117.195 Y126.004 E.01201
G1 X116.82 Y125.787 E.01657
G3 X115.927 Y124.807 I5.233 J-5.664 E.05069
G3 X103.79 Y131.246 I-31.533 J-44.776 E.52595
G1 X104.386 Y131.699 E.02861
G1 X104.715 Y132.39 E.02922
G3 X104.775 Y133.724 I-6.472 J.955 E.05107
G1 X104.775 Y141.724 E.30543
G1 X104.679 Y142.326 E.02324
G1 X104.613 Y142.443 E.00516
G1 X154.069 Y142.443 E1.88819
G3 X143.802 Y132.7 I31.829 J-43.821 E.54195
G3 X136.271 Y120.545 I42.109 J-34.5 E.5475
G1 X135.451 Y120.705 E.03188
G3 X133.305 Y120.657 I-.89 J-8.149 E.08218
G1 X132.426 Y120.433 E.03464
G1 X131.481 Y120.051 E.03892
G1 X130.604 Y119.536 E.03883
G1 X129.808 Y118.895 E.03903
G1 X129.119 Y118.147 E.03883
G3 X127.853 Y116.453 I77.09 J-58.916 E.08072
G1 X126.662 Y114.847 E.07636
G3 X122.955 Y118.919 I-43.095 J-35.51 E.21032
G3 X123.663 Y119.791 I-6.181 J5.745 E.04292
G1 X123.913 Y120.386 E.02465
G1 X123.965 Y120.832 E.01716
G1 X123.892 Y121.362 E.02042
G1 X123.702 Y121.813 E.01867
G1 X123.565 Y121.99 E.00854
G1 X123.013 Y121.746 F36000
G1 F13446.369
G1 X122.892 Y121.879 E.00687
G1 X118.854 Y125.264 E.20115
G1 X118.51 Y125.474 E.01539
G1 X118.094 Y125.577 E.01636
G1 X117.639 Y125.539 E.01746
G1 X117.188 Y125.33 E.01896
G1 X116.82 Y124.961 E.01991
G1 X116.03 Y124.018 E.04698
G3 X104.424 Y130.374 I-31.612 J-43.944 E.50645
G1 X104.066 Y130.512 E.01468
G1 X103.692 Y130.656 E.01527
G1 F12162.414
G1 X103.319 Y130.8 E.01527
; LINE_WIDTH: 0.668047
G1 F10834.012
G1 X103.232 Y130.857 E.00429
; LINE_WIDTH: 0.716098
G1 F10501.469
G1 X103.145 Y130.914 E.00462
; LINE_WIDTH: 0.76415
G1 F10174.084
G1 X103.058 Y130.971 E.00494
; LINE_WIDTH: 0.812201
G1 F9851.891
G1 X102.971 Y131.028 E.00527
; LINE_WIDTH: 0.860252
G1 F9534.874
G1 X102.885 Y131.085 E.0056
; LINE_WIDTH: 0.908303
G1 F9010.642
G1 X102.798 Y131.142 E.00592
; LINE_WIDTH: 0.956354
G1 F8541.05
G1 X102.711 Y131.199 E.00625
; LINE_WIDTH: 1.00441
G1 F8117.98
G1 X102.624 Y131.256 E.00657
; LINE_WIDTH: 1.05246
G1 F7734.845
G1 X102.537 Y131.313 E.0069
G1 X102.621 Y131.343 E.00592
; LINE_WIDTH: 1.00441
G1 F8117.98
G1 X102.705 Y131.373 E.00564
; LINE_WIDTH: 0.956354
G1 F8541.05
G1 X102.789 Y131.403 E.00536
; LINE_WIDTH: 0.908303
G1 F9010.642
G1 X102.872 Y131.434 E.00508
; LINE_WIDTH: 0.860252
G1 F9534.874
G1 X102.956 Y131.464 E.0048
; LINE_WIDTH: 0.812201
G1 F10123.872
G1 X103.04 Y131.494 E.00452
; LINE_WIDTH: 0.76415
G1 F10403.833
G1 X103.124 Y131.524 E.00424
; LINE_WIDTH: 0.716098
G1 F10687.582
G1 X103.208 Y131.555 E.00396
; LINE_WIDTH: 0.668047
G1 F10975.149
G1 X103.292 Y131.585 E.00368
; LINE_WIDTH: 0.619996
G1 F12311.926
G1 X103.613 Y131.824 E.01527
G1 F13446.369
G1 X103.917 Y132.05 E.01448
G1 X104.148 Y132.533 E.02045
G3 X104.189 Y133.724 I-6.214 J.812 E.04556
G1 X104.189 Y141.724 E.30543
G1 X104.122 Y142.145 E.01626
G1 X103.852 Y142.622 E.02094
G1 X103.299 Y142.991 E.02537
; LINE_WIDTH: 0.597286
G1 F13988.809
G1 X103.057 Y143.049 E.00914
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.836 J67.097 E.10277
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.253 Y132.327 I30.04 J-44.21 E.59954
G3 X136.596 Y119.828 I41.248 J-33.866 E.5614
G1 X136.064 Y119.995 E.02126
G1 X135.344 Y120.129 E.02798
G1 X134.411 Y120.179 E.03567
G3 X133.286 Y120.054 I.627 J-10.809 E.04322
G1 X132.566 Y119.865 E.02844
G1 X131.712 Y119.513 E.03527
G1 X130.911 Y119.037 E.03555
G1 X130.185 Y118.446 E.03575
G3 X129.066 Y117.105 I8.507 J-8.237 E.06674
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-43.155 J-34.608 E.25726
G1 X123.061 Y119.957 E.05366
G1 X123.304 Y120.384 E.01875
G1 X123.38 Y120.833 E.01737
G1 X123.329 Y121.203 E.01429
G1 X123.137 Y121.61 E.01717
G1 X123.074 Y121.68 E.00359
G1 X122.581 Y121.353 F36000
G1 F13446.369
G1 X122.516 Y121.43 E.00385
G1 X118.478 Y124.815 E.20115
G1 X118.225 Y124.956 E.01106
G1 X117.92 Y124.994 E.01176
G1 X117.665 Y124.931 E.01
G1 X117.381 Y124.718 E.01356
G1 X116.128 Y123.223 E.0745
G3 X99.505 Y131.448 I-31.739 J-43.235 E.71166
G3 X99.6 Y132.672 I-17.998 J2.015 E.0469
G1 X102.683 Y132.101 E.11971
G1 X103.106 Y132.141 E.01623
G1 X103.458 Y132.414 E.017
G1 X103.583 Y132.691 E.01164
G3 X103.603 Y135.724 I-94.605 J2.138 E.1158
G1 X103.603 Y141.724 E.22907
G1 X103.565 Y141.964 E.00928
G1 X103.411 Y142.237 E.01195
G1 X103.098 Y142.453 E.01454
G1 X102.825 Y142.503 E.0106
G1 X102.683 Y142.49 E.00544
G1 X99.603 Y141.919 E.11958
G3 X99.531 Y143.615 I-12.294 J.323 E.06483
G1 X156.235 Y143.615 E2.16494
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.336 J-43.884 E.60776
G3 X136.912 Y119.083 I40.832 J-33.519 E.5764
G1 X136.093 Y119.382 E.03328
G3 X134.379 Y119.594 I-1.769 J-7.269 E.06605
G1 X133.548 Y119.509 E.03191
G1 X132.724 Y119.301 E.03244
G1 X131.942 Y118.975 E.03235
G1 X131.218 Y118.539 E.03227
G1 X130.562 Y117.998 E.03246
G3 X129.081 Y116.142 I12.086 J-11.166 E.09073
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-42.771 J-33.305 E.3043
G1 X122.612 Y120.334 E.07459
G1 X122.773 Y120.655 E.01372
G1 X122.786 Y120.947 E.01118
G1 X122.689 Y121.224 E.01121
G1 X122.639 Y121.284 E.00298
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00555
G1 X116.81 Y123.176 E.04905
; LINE_WIDTH: 0.521326
G1 X116.411 Y122.702 E.01966
; LINE_WIDTH: 0.556496
G1 X116.394 Y122.378 E.01104
G1 X115.477 Y123.015 E.038
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.117 J-43.088 E.58496
G1 X98.942 Y131.542 E.01597
G1 X99.046 Y132.522 E.03118
G1 X99.049 Y133.336 E.02579
G1 X102.784 Y132.645 E.12024
G1 X102.906 Y132.656 E.0039
G1 X103.044 Y132.811 E.00656
G3 X103.05 Y135.724 I-239.373 J2.019 E.09222
G1 X103.05 Y141.724 E.18996
G1 X102.995 Y141.873 E.00502
G1 X102.825 Y141.95 E.00591
G1 X102.784 Y141.946 E.00131
G1 X99.05 Y141.255 E.12021
G3 X99.016 Y143.298 I-19.951 J.684 E.06474
G2 X98.936 Y144.167 I3.311 J.742 E.02771
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I29.092 J-44.026 E.51056
G3 X137.192 Y118.315 I40.815 J-33.397 E.49163
G1 X136.614 Y118.615 E.0206
G1 X135.913 Y118.859 E.02352
G1 X135.135 Y119.01 E.02507
G1 X134.369 Y119.041 E.02429
G1 X133.609 Y118.959 E.0242
G1 X132.865 Y118.766 E.02433
G1 X132.16 Y118.467 E.02425
G1 X131.508 Y118.068 E.02419
G1 X130.918 Y117.575 E.02435
G1 X130.41 Y117.004 E.02419
G3 X129.087 Y115.222 I547.009 J-407.481 E.07027
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-42.345 J-31.987 E.27952
; LINE_WIDTH: 0.554636
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.851 E.00539
G1 X122.202 Y120.926 E.0027
; WIPE_START
M204 S10000
G1 X121.44 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z5.4 F36000
G1 Z5
G1 E.4 F1800
; LINE_WIDTH: 0.556496
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.01281
; LINE_WIDTH: 0.544336
G1 X120.198 Y119.2 E.15231
; LINE_WIDTH: 0.554636
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.198 Y119.2 E-.13045
G1 X119.695 Y119.622 E-.24955
; WIPE_END
G1 E-.02 F1800
G1 X113.238 Y123.691 Z5.4 F36000
G1 X100.189 Y131.913 Z5.4
G1 Z5
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.765696
G1 F10767.612
G1 X100.503 Y131.835 E.01543
; LINE_WIDTH: 0.804816
G1 F10220.902
G1 X100.818 Y131.757 E.01626
; LINE_WIDTH: 0.843936
G1 F9727.027
G1 X101.132 Y131.679 E.01708
; LINE_WIDTH: 0.848116
G1 F9677.064
G1 X101.163 Y131.671 E.00169
; LINE_WIDTH: 0.894256
G1 F9157.829
G1 X101.476 Y131.589 E.01818
; LINE_WIDTH: 0.940396
G1 F8691.478
G1 X101.79 Y131.508 E.01916
; LINE_WIDTH: 0.986536
G1 F8270.322
G1 X102.104 Y131.426 E.02013
; LINE_WIDTH: 1.03268
G1 F7888.095
G1 X102.418 Y131.345 E.02111
; LINE_WIDTH: 1.05246
G1 F7734.845
G1 X102.537 Y131.313 E.00817
; WIPE_START
G1 X102.418 Y131.345 E-.04675
G1 X102.104 Y131.426 E-.12324
G1 X101.79 Y131.508 E-.12324
G1 X101.569 Y131.565 E-.08676
; WIPE_END
G1 E-.02 F1800
G1 X100.759 Y139.155 Z5.4 F36000
G1 X100.364 Y142.844 Z5.4
G1 Z5
G1 E.4 F1800
; LINE_WIDTH: 0.989756
G1 F8242.449
G1 X100.588 Y142.865 E.01399
; LINE_WIDTH: 0.948676
G1 F8612.771
G1 X100.837 Y142.887 E.0149
; LINE_WIDTH: 0.902961
G1 F9066.054
G1 X101.086 Y142.91 E.01416
; LINE_WIDTH: 0.857246
G1 F9569.699
G1 X101.335 Y142.933 E.01341
; LINE_WIDTH: 0.811531
G1 F10132.593
G1 X101.584 Y142.956 E.01267
; LINE_WIDTH: 0.765816
G1 F10765.845
G1 X101.833 Y142.979 E.01192
; LINE_WIDTH: 0.720101
G1 F11483.527
G1 X102.082 Y143.002 E.01118
; LINE_WIDTH: 0.674386
G1 F12303.727
G1 X102.331 Y143.025 E.01043
; LINE_WIDTH: 0.628671
G1 F13250.105
G1 X102.58 Y143.047 E.00969
; LINE_WIDTH: 0.582956
G1 F14354.199
G1 X103.057 Y143.049 E.01707
; WIPE_START
G1 X102.58 Y143.047 E-.18139
G1 X102.331 Y143.025 E-.095
G1 X102.082 Y143.002 E-.095
G1 X102.059 Y143 E-.00861
; WIPE_END
G1 E-.02 F1800
G1 X105.272 Y139.654 Z5.4 F36000
G1 Z5
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.272 Y137.311 E.08944
G3 X107.257 Y136.924 I2.228 J6.131 E.07753
G3 X109.142 Y137.88 I-.087 J2.509 E.0833
G2 X111.028 Y140.195 I16.16 J-11.228 E.11412
G2 X115.269 Y141.648 I3.741 J-4.004 E.17635
G2 X116.683 Y140.747 I-.702 J-2.661 E.06505
G3 X118.568 Y138.431 I16.158 J11.227 E.11412
G3 X122.81 Y136.978 I3.741 J4.004 E.17635
G3 X124.224 Y137.88 I-.702 J2.661 E.06505
G2 X126.109 Y140.195 I16.16 J-11.228 E.11412
G2 X130.351 Y141.648 I3.741 J-4.004 E.17635
G2 X131.764 Y140.747 I-.702 J-2.661 E.06505
M73 P63 R6
G3 X133.65 Y138.431 I16.155 J11.225 E.11412
G3 X137.891 Y136.978 I3.741 J4.004 E.17635
G3 X139.305 Y137.88 I-.702 J2.661 E.06505
G2 X141.19 Y140.195 I16.157 J-11.226 E.11412
G2 X145.432 Y141.648 I3.741 J-4.004 E.17635
G2 X146.846 Y140.747 I-.702 J-2.661 E.06505
G3 X148.593 Y138.575 I16.61 J11.575 E.10649
G3 X143.798 Y133.477 I37.919 J-40.459 E.26738
G3 X140.719 Y134.107 I-2.475 J-4.252 E.12216
G3 X139.305 Y133.206 I.702 J-2.661 E.06505
G2 X137.42 Y130.89 I-16.159 J11.228 E.11412
G2 X133.178 Y129.438 I-3.741 J4.004 E.17635
G2 X131.764 Y130.339 I.702 J2.661 E.06505
G3 X129.879 Y132.655 I-16.159 J-11.228 E.11412
G3 X125.638 Y134.107 I-3.741 J-4.004 E.17635
G3 X124.224 Y133.206 I.702 J-2.661 E.06505
G2 X122.339 Y130.89 I-16.158 J11.227 E.11412
G2 X118.097 Y129.438 I-3.741 J4.004 E.17635
G2 X116.683 Y130.339 I.702 J2.661 E.06505
G3 X114.798 Y132.655 I-16.157 J-11.226 E.11412
G3 X110.556 Y134.107 I-3.741 J-4.004 E.17635
G3 X109.142 Y133.206 I.702 J-2.661 E.06505
G2 X107.257 Y130.89 I-16.157 J11.226 E.11412
G1 X106.807 Y130.548 E.02161
G2 X114.186 Y126.57 I-26.569 J-58.126 E.32027
G2 X116.331 Y126.008 I.518 J-2.401 E.08786
G2 X118.498 Y126.605 I1.64 J-1.724 E.08934
G1 X130.403 Y119.992 F36000
G1 F13446.283
G3 X128.694 Y118.407 I5.602 J-7.755 E.08926
G3 X126.109 Y119.081 I-2.726 J-5.159 E.10289
G3 X124.324 Y118.223 I.034 J-2.356 E.07802
G1 X123.628 Y118.947 E.03836
G3 X123.854 Y122.438 I-1.655 J1.86 E.14811
G1 X124.224 Y122.798 E.0197
G1 X125.166 Y124.035 E.05936
G2 X128.937 Y126.543 I5.142 J-3.642 E.17684
G2 X131.764 Y125.665 I.703 J-2.731 E.119
G1 X132.707 Y124.429 E.05936
G3 X136.38 Y121.948 I5.119 J3.62 E.17295
G2 X137.431 Y124.041 I30.331 J-13.931 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 5.16
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X136.982 Y123.147 E-.38
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
G1 X123.51 Y122.061
G1 Z5.16
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.391 Y122.215 E.00744
G3 X122.296 Y123.143 I-51.218 J-59.379 E.05479
G1 X119.23 Y125.712 E.15272
G1 X118.874 Y125.95 E.01636
G1 X118.45 Y126.11 E.01728
G1 X117.978 Y126.168 E.01818
G1 X117.493 Y126.107 E.01866
G1 X117.196 Y126.004 E.01201
G1 X116.82 Y125.787 E.01657
G3 X115.927 Y124.807 I5.237 J-5.667 E.05069
G3 X103.793 Y131.245 I-31.533 J-44.775 E.52581
G1 X104.386 Y131.693 E.02836
G1 X104.717 Y132.387 E.02938
G3 X104.777 Y133.727 I-6.482 J.958 E.05129
G1 X104.777 Y141.727 E.30543
G1 X104.682 Y142.328 E.02324
G1 X104.616 Y142.443 E.00505
G1 X154.069 Y142.443 E1.88807
G3 X143.803 Y132.702 I31.48 J-43.455 E.5419
G3 X136.271 Y120.545 I41.786 J-34.303 E.54761
G1 X135.451 Y120.705 E.03189
G1 X134.435 Y120.764 E.03884
G3 X133.305 Y120.657 I.708 J-13.487 E.04337
G1 X132.428 Y120.434 E.03456
G1 X131.481 Y120.051 E.03897
G1 X130.604 Y119.536 E.03883
G1 X129.81 Y118.896 E.03896
G1 X129.117 Y118.145 E.03899
G3 X127.853 Y116.453 I77.211 J-59.003 E.08064
G1 X126.662 Y114.847 E.07636
G3 X122.955 Y118.919 I-42.53 J-34.995 E.21032
G3 X123.665 Y119.794 I-6.122 J5.695 E.04305
G1 X123.913 Y120.385 E.02449
G1 X123.965 Y120.833 E.01721
G1 X123.892 Y121.362 E.02042
G1 X123.702 Y121.812 E.01865
G1 X123.565 Y121.99 E.00855
G1 X123.013 Y121.746 F36000
G1 F13446.369
G1 X122.892 Y121.879 E.00687
G1 X118.854 Y125.264 E.20115
G1 X118.51 Y125.474 E.01539
G1 X118.094 Y125.577 E.01636
G1 X117.639 Y125.539 E.01746
G1 X117.188 Y125.33 E.01896
G1 X116.82 Y124.961 E.01991
G1 X116.03 Y124.018 E.04698
G3 X104.42 Y130.376 I-31.612 J-43.944 E.50663
G1 X104.065 Y130.512 E.01451
G1 X103.692 Y130.656 E.01527
G1 F12170.13
G1 X103.319 Y130.8 E.01527
; LINE_WIDTH: 0.667863
G1 F10841.295
G1 X103.232 Y130.856 E.00428
; LINE_WIDTH: 0.71573
G1 F10509.634
G1 X103.146 Y130.913 E.0046
; LINE_WIDTH: 0.763596
G1 F10183.108
G1 X103.059 Y130.97 E.00493
; LINE_WIDTH: 0.811463
G1 F9861.752
G1 X102.972 Y131.027 E.00525
; LINE_WIDTH: 0.85933
G1 F9545.532
G1 X102.886 Y131.084 E.00557
; LINE_WIDTH: 0.907196
G1 F9022.066
G1 X102.799 Y131.14 E.0059
; LINE_WIDTH: 0.955063
G1 F8553.028
G1 X102.712 Y131.197 E.00622
; LINE_WIDTH: 1.00293
G1 F8130.347
G1 X102.626 Y131.254 E.00654
; LINE_WIDTH: 1.0508
G1 F7747.477
G1 X102.539 Y131.311 E.00687
G1 X102.623 Y131.341 E.00589
; LINE_WIDTH: 1.00293
G1 F8130.347
G1 X102.707 Y131.371 E.00562
; LINE_WIDTH: 0.955063
G1 F8553.028
G1 X102.79 Y131.401 E.00534
; LINE_WIDTH: 0.907196
G1 F9022.066
G1 X102.874 Y131.431 E.00506
; LINE_WIDTH: 0.85933
G1 F9545.532
G1 X102.958 Y131.461 E.00478
; LINE_WIDTH: 0.811463
G1 F10133.484
G1 X103.041 Y131.492 E.00451
; LINE_WIDTH: 0.763596
G1 F10412.811
G1 X103.125 Y131.522 E.00423
; LINE_WIDTH: 0.71573
G1 F10695.935
G1 X103.209 Y131.552 E.00395
; LINE_WIDTH: 0.667863
G1 F10982.868
G1 X103.292 Y131.582 E.00367
; LINE_WIDTH: 0.619996
G1 F12320.102
G1 X103.614 Y131.82 E.01527
G1 F13446.369
G1 X103.917 Y132.045 E.01441
G1 X104.15 Y132.53 E.02056
G3 X104.191 Y133.727 I-6.223 J.814 E.04577
G1 X104.191 Y141.727 E.30543
G1 X104.124 Y142.147 E.01626
G1 X103.854 Y142.625 E.02094
G1 X103.301 Y142.993 E.02536
; LINE_WIDTH: 0.595816
G1 F14025.434
G1 X103.059 Y143.05 E.00911
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.835 J62.919 E.1027
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.427 J-44.63 E.59943
G3 X136.596 Y119.828 I41.245 J-33.866 E.56148
G1 X136.065 Y119.995 E.02126
M73 P64 R6
G1 X135.344 Y120.13 E.02799
G3 X133.403 Y120.08 I-.777 J-7.549 E.07432
G1 X132.577 Y119.867 E.03258
G1 X131.712 Y119.513 E.03569
G1 X130.911 Y119.037 E.03555
G1 X130.186 Y118.448 E.03568
G3 X129.066 Y117.106 I8.516 J-8.248 E.0668
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-42.781 J-34.268 E.25726
G1 X123.061 Y119.957 E.05366
G1 X123.303 Y120.384 E.01873
G1 X123.38 Y120.833 E.0174
G1 X123.329 Y121.204 E.01428
G1 X123.137 Y121.61 E.01716
G1 X123.074 Y121.68 E.00359
G1 X122.581 Y121.353 F36000
G1 F13446.369
G1 X122.516 Y121.43 E.00385
G1 X118.478 Y124.815 E.20115
G1 X118.225 Y124.956 E.01106
G1 X117.92 Y124.994 E.01176
G1 X117.666 Y124.931 E.01
G1 X117.381 Y124.718 E.01356
G1 X116.128 Y123.223 E.0745
G3 X99.503 Y131.448 I-31.738 J-43.234 E.71172
G3 X99.602 Y132.67 I-16.646 J1.959 E.0468
G1 X102.685 Y132.099 E.11971
G1 X103.107 Y132.138 E.01619
G1 X103.459 Y132.41 E.01696
G1 X103.585 Y132.689 E.0117
G3 X103.605 Y135.727 I-94.466 J2.141 E.116
G1 X103.605 Y141.727 E.22907
G1 X103.567 Y141.967 E.00928
G1 X103.413 Y142.239 E.01195
G1 X103.1 Y142.456 E.01454
G1 X102.827 Y142.505 E.0106
G1 X102.685 Y142.492 E.00544
G1 X99.605 Y141.922 E.11958
G3 X99.53 Y143.615 I-10.867 J.364 E.06475
G1 X156.235 Y143.615 E2.16497
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.706 Y131.955 I29.319 J-43.865 E.60771
G3 X136.912 Y119.083 I41.088 J-33.675 E.57643
G1 X136.092 Y119.382 E.03331
G3 X132.72 Y119.3 I-1.548 J-5.681 E.13058
G1 X131.942 Y118.975 E.0322
G1 X131.218 Y118.539 E.03227
G1 X130.563 Y117.999 E.0324
G3 X129.081 Y116.142 I12.09 J-11.172 E.09079
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-42.527 J-33.084 E.3043
G1 X122.612 Y120.333 E.07459
G1 X122.773 Y120.655 E.01371
G1 X122.786 Y120.948 E.0112
G1 X122.689 Y121.224 E.01119
G1 X122.639 Y121.284 E.00298
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00555
G1 X116.81 Y123.176 E.04903
; LINE_WIDTH: 0.521316
G1 X116.411 Y122.702 E.01968
; LINE_WIDTH: 0.556516
G1 X116.394 Y122.378 E.01104
G1 X115.476 Y123.015 E.03802
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.116 J-43.088 E.58495
G1 X98.942 Y131.542 E.01595
G1 X99.048 Y132.521 E.0312
G1 X99.051 Y133.334 E.02572
G1 X102.786 Y132.642 E.12024
G1 X102.908 Y132.654 E.00389
G1 X103.046 Y132.809 E.00657
G3 X103.052 Y135.727 I-239.002 J2.021 E.09238
G1 X103.052 Y141.727 E.18996
G1 X102.997 Y141.875 E.00502
G1 X102.827 Y141.952 E.00591
G1 X102.786 Y141.949 E.00131
G1 X99.052 Y141.257 E.1202
G1 X99.052 Y142.625 E.04331
G3 X98.938 Y143.862 I-6.883 J-.01 E.03938
G1 X98.936 Y144.167 E.00966
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I29.092 J-44.026 E.51054
G3 X137.192 Y118.315 I40.272 J-33.073 E.4917
G1 X136.613 Y118.615 E.02064
G1 X135.912 Y118.859 E.02351
G1 X135.135 Y119.01 E.02505
G1 X134.371 Y119.041 E.02421
G1 X133.607 Y118.959 E.02433
G1 X132.867 Y118.767 E.02423
G1 X132.16 Y118.467 E.0243
G1 X131.508 Y118.068 E.02419
G1 X130.919 Y117.576 E.0243
G1 X130.409 Y117.003 E.0243
G3 X129.087 Y115.222 I553.297 J-412.141 E.0702
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-42.153 J-31.815 E.27953
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.851 E.00539
G1 X122.202 Y120.927 E.0027
; WIPE_START
M204 S10000
G1 X121.44 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z5.56 F36000
G1 Z5.16
G1 E.4 F1800
; LINE_WIDTH: 0.556516
G1 F3600
M204 S5000
G1 X116.687 Y122.143 E.0128
; LINE_WIDTH: 0.544336
G1 X120.198 Y119.2 E.15232
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.198 Y119.2 E-.13045
G1 X119.695 Y119.622 E-.24955
; WIPE_END
G1 E-.02 F1800
G1 X113.238 Y123.691 Z5.56 F36000
G1 X100.189 Y131.912 Z5.56
G1 Z5.16
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.763596
G1 F10798.619
G1 X100.503 Y131.834 E.01538
; LINE_WIDTH: 0.802696
G1 F10249.103
G1 X100.817 Y131.756 E.01621
; LINE_WIDTH: 0.841796
G1 F9752.806
G1 X101.131 Y131.678 E.01703
; LINE_WIDTH: 0.845996
G1 F9702.339
G1 X101.162 Y131.67 E.00168
; LINE_WIDTH: 0.892141
G1 F9180.409
G1 X101.476 Y131.588 E.01814
; LINE_WIDTH: 0.938286
G1 F8711.766
G1 X101.79 Y131.507 E.01912
; LINE_WIDTH: 0.984431
G1 F8288.645
G1 X102.104 Y131.425 E.02009
; LINE_WIDTH: 1.03058
G1 F7904.722
G1 X102.418 Y131.344 E.02107
; LINE_WIDTH: 1.0508
G1 F7747.477
G1 X102.539 Y131.311 E.00833
; WIPE_START
G1 X102.418 Y131.344 E-.04776
G1 X102.104 Y131.425 E-.12327
G1 X101.79 Y131.507 E-.12327
G1 X101.572 Y131.564 E-.08569
; WIPE_END
G1 E-.02 F1800
G1 X100.761 Y139.153 Z5.56 F36000
G1 X100.366 Y142.845 Z5.56
G1 Z5.16
G1 E.4 F1800
; LINE_WIDTH: 0.987296
G1 F8263.727
G1 X100.59 Y142.866 E.01397
; LINE_WIDTH: 0.946166
G1 F8636.479
G1 X100.839 Y142.889 E.01486
; LINE_WIDTH: 0.90045
G1 F9092.34
G1 X101.088 Y142.912 E.01412
; LINE_WIDTH: 0.854734
G1 F9599.005
G1 X101.337 Y142.934 E.01337
; LINE_WIDTH: 0.809017
G1 F10165.473
G1 X101.586 Y142.957 E.01263
; LINE_WIDTH: 0.763301
G1 F10802.989
G1 X101.835 Y142.98 E.01188
; LINE_WIDTH: 0.717585
G1 F11525.817
G1 X102.084 Y143.003 E.01114
; LINE_WIDTH: 0.671869
G1 F12352.313
G1 X102.333 Y143.026 E.01039
; LINE_WIDTH: 0.626153
G1 F13306.496
G1 X102.582 Y143.049 E.00965
; LINE_WIDTH: 0.580436
G1 F14420.437
G1 X103.059 Y143.05 E.01698
; WIPE_START
G1 X102.582 Y143.049 E-.18122
G1 X102.333 Y143.026 E-.095
G1 X102.084 Y143.003 E-.095
G1 X102.061 Y143.001 E-.00878
; WIPE_END
G1 E-.02 F1800
G1 X105.274 Y139.61 Z5.56 F36000
G1 Z5.16
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.274 Y137.267 E.08944
G3 X107.729 Y136.779 I2.072 J3.998 E.09681
G3 X109.142 Y137.603 I-.362 J2.246 E.06392
G2 X111.499 Y140.565 I14.332 J-8.985 E.1448
G2 X115.269 Y141.848 I3.698 J-4.687 E.15502
G2 X116.683 Y141.023 I-.362 J-2.246 E.06392
G3 X119.04 Y138.061 I14.331 J8.984 E.1448
G3 X122.81 Y136.779 I3.698 J4.687 E.15502
G3 X124.224 Y137.603 I-.362 J2.246 E.06392
G2 X126.58 Y140.565 I14.332 J-8.985 E.1448
G2 X130.351 Y141.848 I3.698 J-4.687 E.15502
G2 X131.764 Y141.023 I-.362 J-2.246 E.06392
G3 X134.121 Y138.061 I14.331 J8.984 E.1448
G3 X137.891 Y136.779 I3.698 J4.687 E.15502
G3 X139.305 Y137.603 I-.362 J2.246 E.06392
G2 X141.662 Y140.565 I14.332 J-8.985 E.1448
G2 X145.432 Y141.848 I3.698 J-4.687 E.15502
G2 X146.846 Y141.023 I-.362 J-2.246 E.06392
G3 X148.635 Y138.605 I18.918 J12.129 E.11492
G3 X143.805 Y133.484 I36.958 J-39.687 E.26897
G3 X140.719 Y134.307 I-2.857 J-4.513 E.12385
G3 X139.305 Y133.483 I.362 J-2.246 E.06392
G2 X136.949 Y130.521 I-14.331 J8.984 E.1448
G2 X133.178 Y129.238 I-3.698 J4.687 E.15502
G2 X131.764 Y130.062 I.362 J2.246 E.06392
G3 X129.408 Y133.024 I-14.332 J-8.985 E.14481
G3 X125.638 Y134.307 I-3.698 J-4.687 E.15502
G3 X124.224 Y133.483 I.362 J-2.246 E.06392
G2 X121.867 Y130.521 I-14.332 J8.985 E.14481
G2 X118.097 Y129.238 I-3.698 J4.687 E.15502
G2 X116.683 Y130.062 I.362 J2.246 E.06392
G3 X114.327 Y133.024 I-14.332 J-8.985 E.14481
G3 X110.556 Y134.307 I-3.698 J-4.687 E.15502
G3 X109.142 Y133.483 I.362 J-2.246 E.06392
G2 X106.812 Y130.545 I-14.474 J9.088 E.14344
G2 X114.033 Y126.667 I-26.706 J-58.384 E.31314
G2 X116.212 Y126.4 I.784 J-2.623 E.08623
G1 X116.476 Y126.144 E.01404
G2 X118.689 Y126.556 I1.5 J-1.904 E.08941
G1 X130.412 Y119.998 F36000
G1 F13446.283
G3 X128.701 Y118.415 I5.536 J-7.702 E.08926
G3 X126.109 Y119.241 I-3.182 J-5.503 E.10467
G3 X124.193 Y118.357 I-.14 J-2.215 E.08396
G1 X123.628 Y118.947 E.03118
G3 X123.981 Y122.285 I-1.723 J1.87 E.14001
G1 X124.224 Y122.522 E.01294
G1 X125.166 Y123.865 E.06266
G2 X129.879 Y126.782 I5.194 J-3.127 E.21976
G2 X131.764 Y125.942 I.165 J-2.165 E.08211
G1 X132.707 Y124.598 E.06266
G3 X136.345 Y121.867 I5.454 J3.476 E.17747
G2 X137.39 Y123.963 I30.514 J-13.907 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 5.32
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X136.944 Y123.068 E-.38
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
G1 X123.51 Y122.061
G1 Z5.32
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.391 Y122.216 E.00745
G3 X122.296 Y123.143 I-51.317 J-59.497 E.05479
G1 X119.23 Y125.712 E.15272
G1 X118.874 Y125.95 E.01636
G1 X118.45 Y126.11 E.01728
G1 X117.978 Y126.168 E.01818
G1 X117.493 Y126.107 E.01866
G1 X117.196 Y126.004 E.01201
G1 X116.82 Y125.787 E.01657
G3 X115.927 Y124.807 I5.234 J-5.664 E.05069
G3 X103.797 Y131.243 I-31.532 J-44.774 E.52567
G1 X104.385 Y131.686 E.02811
G1 X104.719 Y132.384 E.02954
G3 X104.779 Y133.729 I-6.49 J.96 E.0515
G1 X104.779 Y141.729 E.30543
G1 X104.684 Y142.331 E.02324
G1 X104.62 Y142.443 E.00494
G1 X154.069 Y142.443 E1.88792
G3 X143.803 Y132.702 I31.881 J-43.876 E.54187
G3 X136.317 Y120.645 I42.085 J-34.485 E.54338
G1 X136.271 Y120.545 E.00419
G3 X133.307 Y120.657 I-1.757 J-7.197 E.11404
G1 X132.429 Y120.434 E.03459
G1 X131.481 Y120.051 E.03902
G1 X130.602 Y119.535 E.03892
G1 X129.81 Y118.896 E.03884
G1 X129.117 Y118.145 E.039
G3 X127.853 Y116.453 I76.968 J-58.823 E.08065
G1 X126.662 Y114.847 E.07636
G3 X122.955 Y118.919 I-42.545 J-35.008 E.21032
G3 X123.665 Y119.793 I-6.12 J5.693 E.04305
G1 X123.913 Y120.385 E.0245
G1 X123.965 Y120.833 E.01723
G1 X123.893 Y121.362 E.02035
G1 X123.702 Y121.812 E.01868
G1 X123.565 Y121.99 E.00856
G1 X123.013 Y121.746 F36000
G1 F13446.369
G1 X122.892 Y121.879 E.00687
G1 X118.854 Y125.264 E.20115
G1 X118.51 Y125.474 E.01539
G1 X118.094 Y125.577 E.01636
G1 X117.639 Y125.539 E.01746
G1 X117.188 Y125.33 E.01896
G1 X116.82 Y124.961 E.01991
G1 X116.03 Y124.018 E.04698
G3 X104.416 Y130.377 I-31.614 J-43.947 E.5068
G1 X104.065 Y130.512 E.01433
G1 X103.692 Y130.656 E.01527
G1 F12177.86
G1 X103.319 Y130.8 E.01527
; LINE_WIDTH: 0.667678
G1 F10848.591
G1 X103.233 Y130.856 E.00426
; LINE_WIDTH: 0.715361
G1 F10517.84
G1 X103.146 Y130.913 E.00459
; LINE_WIDTH: 0.763043
G1 F10192.183
G1 X103.06 Y130.969 E.00491
; LINE_WIDTH: 0.810725
G1 F9871.63
G1 X102.973 Y131.026 E.00523
; LINE_WIDTH: 0.858407
G1 F9556.215
G1 X102.887 Y131.083 E.00555
; LINE_WIDTH: 0.90609
G1 F9033.519
G1 X102.801 Y131.139 E.00587
; LINE_WIDTH: 0.953772
G1 F8565.038
G1 X102.714 Y131.196 E.00619
; LINE_WIDTH: 1.00145
G1 F8142.752
G1 X102.628 Y131.253 E.00651
; LINE_WIDTH: 1.04914
G1 F7760.15
G1 X102.541 Y131.309 E.00683
G1 X102.625 Y131.339 E.00587
; LINE_WIDTH: 1.00145
G1 F8142.752
G1 X102.708 Y131.369 E.00559
; LINE_WIDTH: 0.953772
G1 F8565.038
G1 X102.792 Y131.399 E.00532
; LINE_WIDTH: 0.90609
G1 F9033.519
G1 X102.875 Y131.429 E.00504
; LINE_WIDTH: 0.858407
G1 F9556.215
G1 X102.959 Y131.459 E.00477
; LINE_WIDTH: 0.810725
G1 F10143.114
G1 X103.042 Y131.489 E.00449
; LINE_WIDTH: 0.763043
G1 F10421.846
G1 X103.126 Y131.519 E.00421
; LINE_WIDTH: 0.715361
G1 F10704.326
G1 X103.209 Y131.549 E.00394
; LINE_WIDTH: 0.667678
G1 F10990.584
G1 X103.293 Y131.579 E.00366
; LINE_WIDTH: 0.619996
G1 F12328.273
G1 X103.615 Y131.816 E.01527
G1 F13446.369
G1 X103.917 Y132.039 E.01435
G1 X104.151 Y132.527 E.02067
G3 X104.193 Y133.729 I-6.229 J.817 E.04598
G1 X104.193 Y141.729 E.30543
G1 X104.127 Y142.15 E.01626
G1 X103.856 Y142.627 E.02094
G1 X103.303 Y142.995 E.02536
; LINE_WIDTH: 0.594326
G1 F13896.678
G1 X103.061 Y143.051 E.00908
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.833 J59.245 E.10263
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.104 J-44.28 E.59947
G3 X136.596 Y119.828 I41.197 J-33.837 E.56147
G1 X136.066 Y119.995 E.02118
G1 X135.344 Y120.129 E.02805
G1 X134.411 Y120.179 E.03568
G3 X133.286 Y120.054 I.63 J-10.835 E.04324
G1 X132.568 Y119.865 E.02834
G1 X131.712 Y119.513 E.03536
G1 X130.909 Y119.036 E.03564
G1 X130.187 Y118.448 E.03556
G3 X129.066 Y117.106 I8.514 J-8.247 E.06683
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-42.795 J-34.281 E.25726
G1 X123.061 Y119.957 E.05366
G1 X123.303 Y120.384 E.01873
G1 X123.38 Y120.833 E.01742
G1 X123.329 Y121.203 E.01424
G1 X123.137 Y121.61 E.01718
G1 X123.074 Y121.68 E.00359
G1 X122.581 Y121.353 F36000
G1 F13446.369
G1 X122.516 Y121.43 E.00385
G1 X118.478 Y124.815 E.20115
G1 X118.225 Y124.956 E.01106
G1 X117.92 Y124.994 E.01176
G1 X117.665 Y124.931 E.01
G1 X117.381 Y124.718 E.01356
G1 X116.128 Y123.223 E.0745
G3 X99.502 Y131.448 I-31.739 J-43.235 E.71176
G3 X99.604 Y132.667 I-14.692 J1.839 E.0467
G1 X102.687 Y132.096 E.11972
G1 X103.108 Y132.135 E.01616
G1 X103.46 Y132.405 E.01693
G1 X103.587 Y132.686 E.01177
G3 X103.607 Y135.729 I-94.289 J2.143 E.1162
G1 X103.607 Y141.729 E.22907
G1 X103.569 Y141.969 E.00928
G1 X103.415 Y142.242 E.01195
G1 X103.102 Y142.458 E.01454
G1 X102.829 Y142.508 E.0106
G1 X102.687 Y142.495 E.00544
G1 X99.607 Y141.924 E.11958
G3 X99.531 Y143.615 I-10.689 J.362 E.06466
G1 X156.235 Y143.615 E2.16493
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.955 I29.377 J-43.928 E.60771
G3 X136.983 Y119.247 I41.159 J-33.715 E.56959
G1 X136.912 Y119.083 E.00682
G1 X136.092 Y119.382 E.03331
G1 X135.299 Y119.542 E.0309
G1 X134.389 Y119.593 E.03477
G3 X132.721 Y119.3 I.435 J-7.361 E.06481
G1 X131.942 Y118.975 E.03223
G1 X131.216 Y118.537 E.03235
G1 X130.564 Y118 E.03228
G3 X129.081 Y116.142 I12.078 J-11.164 E.09083
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-42.535 J-33.092 E.3043
G1 X122.612 Y120.333 E.07459
G1 X122.773 Y120.655 E.01371
G1 X122.786 Y120.948 E.0112
G1 X122.689 Y121.224 E.01119
G1 X122.639 Y121.284 E.00299
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00555
G1 X116.81 Y123.176 E.04905
; LINE_WIDTH: 0.521326
G1 X116.411 Y122.702 E.01966
; LINE_WIDTH: 0.556496
G1 X116.394 Y122.378 E.01104
G1 X115.477 Y123.015 E.038
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.117 J-43.088 E.58496
G1 X98.942 Y131.541 E.01594
G1 X99.05 Y132.522 E.03124
G1 X99.054 Y133.331 E.02562
G1 X102.788 Y132.64 E.12024
G1 X102.91 Y132.651 E.00388
G1 X103.048 Y132.806 E.00657
G3 X103.055 Y135.729 I-238.531 J2.023 E.09254
G1 X103.055 Y141.729 E.18996
G1 X102.999 Y141.878 E.00502
G1 X102.829 Y141.955 E.00591
G1 X102.788 Y141.951 E.00131
G1 X99.055 Y141.26 E.1202
G1 X99.054 Y142.625 E.04323
G3 X98.938 Y143.863 I-6.786 J-.011 E.03943
G1 X98.936 Y144.167 E.00963
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.795 J-43.698 E.51058
G3 X137.192 Y118.315 I40.72 J-33.34 E.49166
G1 X136.614 Y118.615 E.02061
M73 P65 R6
G1 X135.912 Y118.859 E.02354
G1 X135.197 Y118.999 E.02305
G1 X134.369 Y119.041 E.02626
G1 X133.608 Y118.959 E.02423
G1 X132.867 Y118.767 E.02423
G1 X132.16 Y118.466 E.02434
G1 X131.506 Y118.067 E.02425
G1 X130.92 Y117.577 E.0242
G1 X130.409 Y117.003 E.02433
G3 X129.087 Y115.222 I546.352 J-406.986 E.0702
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-42.158 J-31.82 E.27953
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.851 E.00539
G1 X122.202 Y120.927 E.0027
; WIPE_START
M204 S10000
G1 X121.44 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z5.72 F36000
G1 Z5.32
G1 E.4 F1800
; LINE_WIDTH: 0.556496
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.01281
; LINE_WIDTH: 0.544336
G1 X120.199 Y119.2 E.15232
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.199 Y119.2 E-.13038
G1 X119.695 Y119.622 E-.24962
; WIPE_END
G1 E-.02 F1800
G1 X113.238 Y123.69 Z5.72 F36000
G1 X100.191 Y131.91 Z5.72
G1 Z5.32
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.76173
G1 F10826.33
G1 X100.505 Y131.833 E.01531
; LINE_WIDTH: 0.800743
G1 F10275.226
G1 X100.818 Y131.755 E.01613
; LINE_WIDTH: 0.839756
G1 F9777.508
G1 X101.131 Y131.677 E.01695
; LINE_WIDTH: 0.843936
G1 F9727.027
G1 X101.162 Y131.669 E.00168
; LINE_WIDTH: 0.890071
G1 F9202.616
G1 X101.476 Y131.587 E.01809
; LINE_WIDTH: 0.936206
G1 F8731.858
G1 X101.79 Y131.506 E.01907
; LINE_WIDTH: 0.982341
G1 F8306.919
G1 X102.104 Y131.424 E.02004
; LINE_WIDTH: 1.02848
G1 F7921.42
G1 X102.417 Y131.343 E.02102
; LINE_WIDTH: 1.04914
G1 F7760.15
G1 X102.541 Y131.309 E.00849
; WIPE_START
G1 X102.417 Y131.343 E-.04878
G1 X102.104 Y131.424 E-.12324
G1 X101.79 Y131.506 E-.12324
G1 X101.574 Y131.562 E-.08474
; WIPE_END
G1 E-.02 F1800
G1 X100.762 Y139.151 Z5.72 F36000
G1 X100.367 Y142.846 Z5.72
G1 Z5.32
G1 E.4 F1800
; LINE_WIDTH: 0.985016
G1 F8283.545
G1 X100.593 Y142.867 E.01403
; LINE_WIDTH: 0.943606
G1 F8660.794
G1 X100.842 Y142.89 E.01482
; LINE_WIDTH: 0.897892
G1 F9119.268
G1 X101.091 Y142.913 E.01407
; LINE_WIDTH: 0.852179
G1 F9628.994
G1 X101.34 Y142.936 E.01333
; LINE_WIDTH: 0.806465
G1 F10199.077
G1 X101.588 Y142.959 E.01258
; LINE_WIDTH: 0.760751
G1 F10840.911
G1 X101.837 Y142.981 E.01184
; LINE_WIDTH: 0.715038
G1 F11568.953
G1 X102.086 Y143.004 E.01109
; LINE_WIDTH: 0.669324
G1 F12401.821
G1 X102.335 Y143.027 E.01035
; LINE_WIDTH: 0.62361
G1 F13363.908
G1 X102.584 Y143.05 E.0096
; LINE_WIDTH: 0.577896
G1 F14487.822
G1 X103.061 Y143.051 E.01688
; WIPE_START
G1 X102.584 Y143.05 E-.18104
G1 X102.335 Y143.027 E-.095
G1 X102.086 Y143.004 E-.095
G1 X102.063 Y143.002 E-.00896
; WIPE_END
G1 E-.02 F1800
G1 X105.277 Y139.559 Z5.72 F36000
G1 Z5.32
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.277 Y137.216 E.08944
G3 X108.2 Y136.612 I2.498 J4.711 E.11552
G3 X109.614 Y137.967 I-1.023 J2.483 E.07652
G2 X111.97 Y140.917 I7.739 J-3.764 E.14535
G2 X115.899 Y141.945 I3.298 J-4.583 E.15861
G2 X117.154 Y140.66 I-1.205 J-2.432 E.06992
G3 X119.511 Y137.709 I7.739 J3.764 E.14535
G3 X122.81 Y136.558 I3.675 J5.23 E.13512
G3 X124.224 Y137.265 I-.035 J1.837 E.0624
G2 X126.109 Y140.068 I39.813 J-24.748 E.12897
G2 X130.98 Y141.945 I4.249 J-3.765 E.20712
G2 X132.236 Y140.66 I-1.205 J-2.432 E.06992
G3 X134.592 Y137.709 I7.739 J3.764 E.14535
G3 X137.891 Y136.558 I3.675 J5.23 E.13512
G3 X139.305 Y137.265 I-.035 J1.837 E.0624
G2 X141.19 Y140.068 I39.813 J-24.748 E.12897
G2 X146.062 Y141.945 I4.249 J-3.765 E.20712
G2 X147.317 Y140.66 I-1.205 J-2.432 E.06992
G3 X148.665 Y138.641 I12.599 J6.955 E.09277
G3 X143.814 Y133.495 I37.57 J-40.268 E.27019
G3 X140.248 Y134.474 I-3.16 J-4.528 E.14399
G3 X138.834 Y133.119 I1.023 J-2.483 E.07652
G2 X136.477 Y130.169 I-7.739 J3.764 E.14535
G2 X133.178 Y129.017 I-3.675 J5.23 E.13512
G2 X131.764 Y129.725 I.035 J1.837 E.0624
G3 X129.879 Y132.527 I-39.813 J-24.748 E.12897
G3 X125.638 Y134.528 I-4.567 J-4.186 E.18363
G3 X124.224 Y133.82 I.035 J-1.837 E.0624
G2 X122.339 Y131.018 I-39.818 J24.752 E.12897
G2 X118.097 Y129.017 I-4.567 J4.186 E.18363
G2 X116.683 Y129.725 I.035 J1.837 E.0624
G3 X114.798 Y132.527 I-39.813 J-24.748 E.12897
G3 X110.556 Y134.528 I-4.567 J-4.186 E.18363
G3 X109.142 Y133.82 I.035 J-1.837 E.0624
G2 X107.257 Y131.018 I-39.813 J24.748 E.12897
G1 X106.794 Y130.553 E.02506
G2 X113.894 Y126.755 I-26.808 J-58.649 E.30763
G2 X115.741 Y126.933 I1.284 J-3.651 E.07152
G2 X116.682 Y126.28 I-.419 J-1.61 E.04466
G2 X118.924 Y126.475 I1.309 J-2.066 E.08929
G1 X130.427 Y120.005 F36000
G1 F13446.283
G3 X128.711 Y118.427 I5.32 J-7.507 E.08925
G3 X126.109 Y119.415 I-3.778 J-6.024 E.10696
G3 X124.06 Y118.496 I-.399 J-1.853 E.09177
G1 X123.628 Y118.947 E.02386
G3 X124.132 Y122.064 I-1.709 J1.876 E.13013
G3 X125.166 Y123.676 I-72.237 J47.504 E.07311
G2 X129.879 Y126.956 I5.581 J-2.993 E.22753
G2 X131.293 Y126.728 I.366 J-2.231 E.05563
G2 X132.707 Y124.788 I-4.968 J-5.106 E.0921
G3 X136.297 Y121.786 I5.563 J3.006 E.18302
G2 X137.347 Y123.881 I43.46 J-20.474 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 5.48
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X136.899 Y122.987 E-.38
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
G1 X123.51 Y122.061
G1 Z5.48
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.391 Y122.215 E.00744
G3 X122.296 Y123.143 I-40.554 J-46.782 E.05481
G1 X119.23 Y125.712 E.15272
G1 X118.874 Y125.95 E.01636
G1 X118.451 Y126.11 E.01727
G3 X117.245 Y126.025 I-.456 J-2.137 E.04676
G1 X116.82 Y125.787 E.01859
G3 X115.927 Y124.807 I5.233 J-5.664 E.0507
G3 X103.814 Y131.237 I-31.458 J-44.631 E.52496
G1 X104.292 Y131.568 E.0222
G1 X104.697 Y132.294 E.03173
G1 X104.781 Y132.858 E.02175
G1 X104.781 Y141.732 E.33881
G1 X104.686 Y142.333 E.02324
G1 X104.623 Y142.443 E.00483
G1 X154.069 Y142.443 E1.88779
G3 X143.806 Y132.705 I31.489 J-43.462 E.54172
G3 X136.271 Y120.545 I42.339 J-34.651 E.54774
G1 X135.453 Y120.705 E.0318
G3 X133.307 Y120.657 I-.892 J-8.182 E.0822
G1 X132.426 Y120.433 E.03471
G1 X131.481 Y120.051 E.03893
G1 X130.602 Y119.534 E.03893
G1 X129.809 Y118.895 E.03887
G1 X129.117 Y118.145 E.03897
G3 X127.853 Y116.453 I77.218 J-59.008 E.08063
G1 X126.662 Y114.847 E.07636
G3 X122.955 Y118.919 I-42.927 J-35.356 E.21032
G1 X123.51 Y119.581 E.033
G1 X123.717 Y119.881 E.01389
G1 X123.913 Y120.385 E.02067
G1 X123.965 Y120.833 E.01721
G1 X123.893 Y121.361 E.02037
G1 X123.702 Y121.812 E.01869
G1 X123.565 Y121.99 E.00856
G1 X123.013 Y121.746 F36000
G1 F13446.369
G1 X122.892 Y121.879 E.00687
G1 X118.854 Y125.264 E.20115
G1 X118.51 Y125.474 E.01539
G1 X118.095 Y125.577 E.01636
G3 X117.168 Y125.316 I-.093 J-1.443 E.03749
G1 X116.783 Y124.917 E.02115
G1 X116.03 Y124.018 E.04479
G3 X105.306 Y130.013 I-31.677 J-44.066 E.47006
G1 X104.06 Y130.507 E.05117
G1 X103.688 Y130.654 E.01527
G1 F12181.756
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.667492
G1 F10852.268
G1 X103.23 Y130.857 E.00424
; LINE_WIDTH: 0.714987
G1 F10523.392
G1 X103.145 Y130.913 E.00456
; LINE_WIDTH: 0.762483
G1 F10199.558
G1 X103.059 Y130.97 E.00487
; LINE_WIDTH: 0.809978
G1 F9880.76
G1 X102.973 Y131.026 E.00519
; LINE_WIDTH: 0.857474
G1 F9567.049
G1 X102.887 Y131.082 E.00551
; LINE_WIDTH: 0.90497
G1 F9045.14
G1 X102.801 Y131.139 E.00583
; LINE_WIDTH: 0.952465
G1 F8577.227
G1 X102.715 Y131.195 E.00615
; LINE_WIDTH: 0.999961
G1 F8155.345
G1 X102.63 Y131.251 E.00646
; LINE_WIDTH: 1.04746
G1 F7773.018
G1 X102.544 Y131.308 E.00678
G1 X102.627 Y131.337 E.00584
; LINE_WIDTH: 0.999961
G1 F8155.345
G1 X102.71 Y131.367 E.00557
; LINE_WIDTH: 0.952465
G1 F8577.227
G1 X102.794 Y131.397 E.00529
; LINE_WIDTH: 0.90497
G1 F9045.14
G1 X102.877 Y131.427 E.00502
; LINE_WIDTH: 0.857474
G1 F9567.049
G1 X102.96 Y131.456 E.00475
; LINE_WIDTH: 0.809978
G1 F10152.878
G1 X103.043 Y131.486 E.00447
; LINE_WIDTH: 0.762483
G1 F10430.937
G1 X103.127 Y131.516 E.0042
; LINE_WIDTH: 0.714987
G1 F10712.732
G1 X103.21 Y131.546 E.00392
; LINE_WIDTH: 0.667492
G1 F10998.284
G1 X103.293 Y131.576 E.00365
; LINE_WIDTH: 0.619996
G1 F12336.428
G1 X103.624 Y131.8 E.01527
G1 F13308.228
G1 X103.853 Y131.956 E.01058
G1 F13446.369
G1 X104.137 Y132.464 E.0222
G1 X104.195 Y132.956 E.01893
G1 X104.195 Y141.732 E.33504
G1 X104.129 Y142.152 E.01626
G1 X103.858 Y142.63 E.02094
G1 X103.305 Y142.997 E.02535
; LINE_WIDTH: 0.592856
G1 F13896.38
G1 X103.063 Y143.052 E.00905
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.832 J55.967 E.10256
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.256 Y132.331 I29.96 J-44.123 E.59936
G3 X136.596 Y119.828 I41.749 J-34.179 E.56155
G1 X136.067 Y119.995 E.02116
G1 X135.346 Y120.129 E.02801
G1 X134.412 Y120.179 E.03572
G3 X133.284 Y120.054 I.63 J-10.835 E.04332
G1 X132.566 Y119.865 E.02836
G1 X131.711 Y119.513 E.03529
G1 X130.909 Y119.036 E.03564
G1 X130.186 Y118.447 E.03559
G3 X129.066 Y117.106 I8.525 J-8.253 E.06678
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-46.075 J-37.257 E.25724
G1 X123.061 Y119.957 E.05366
G1 X123.303 Y120.384 E.01872
G1 X123.38 Y120.833 E.0174
G1 X123.329 Y121.203 E.01425
G1 X123.137 Y121.61 E.01718
G1 X123.074 Y121.68 E.0036
G1 X122.581 Y121.353 F36000
G1 F13446.369
G1 X122.516 Y121.43 E.00385
G1 X118.478 Y124.815 E.20115
G1 X118.225 Y124.956 E.01105
G1 X117.92 Y124.994 E.01176
G1 X117.685 Y124.94 E.0092
G1 X117.381 Y124.718 E.01435
G1 X116.128 Y123.223 E.0745
G3 X99.501 Y131.449 I-31.623 J-43.001 E.71186
G3 X99.606 Y132.665 I-13.67 J1.795 E.0466
G1 X102.689 Y132.094 E.11972
G1 X103.108 Y132.132 E.01606
G1 X103.425 Y132.356 E.0148
G1 X103.581 Y132.649 E.01269
G3 X103.609 Y135.732 I-68.343 J2.18 E.1177
G1 X103.609 Y141.732 E.22907
G1 X103.572 Y141.972 E.00928
G1 X103.417 Y142.244 E.01195
G1 X103.104 Y142.461 E.01454
G1 X102.831 Y142.51 E.0106
G1 X102.689 Y142.497 E.00544
G1 X99.609 Y141.927 E.11958
G3 X99.532 Y143.615 I-10.505 J.361 E.06457
G1 X156.235 Y143.615 E2.1649
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.707 Y131.956 I29.093 J-43.618 E.60768
G3 X136.912 Y119.083 I40.768 J-33.483 E.57653
G1 X136.092 Y119.382 E.03329
G3 X132.719 Y119.299 I-1.548 J-5.682 E.13065
G1 X131.942 Y118.974 E.03216
G1 X131.216 Y118.537 E.03236
G1 X130.563 Y117.999 E.03231
G3 X129.081 Y116.142 I12.101 J-11.179 E.09078
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-44.683 J-35.033 E.30427
G1 X122.612 Y120.334 E.07459
G1 X122.773 Y120.654 E.01371
G1 X122.786 Y120.947 E.01119
G1 X122.689 Y121.224 E.0112
G1 X122.639 Y121.284 E.00299
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00556
G1 X116.81 Y123.176 E.04905
; LINE_WIDTH: 0.521326
G1 X116.411 Y122.702 E.01966
; LINE_WIDTH: 0.556496
G1 X116.394 Y122.378 E.01104
G1 X115.477 Y123.015 E.038
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.117 J-43.088 E.58496
G1 X98.942 Y131.54 E.01592
G1 X99.052 Y132.522 E.03125
G1 X99.056 Y133.329 E.02555
G1 X102.79 Y132.637 E.12024
G1 X102.911 Y132.648 E.00386
G1 X103.047 Y132.794 E.0063
G3 X103.057 Y135.732 I-172.039 J2.035 E.09302
G1 X103.057 Y141.732 E.18996
G1 X103.001 Y141.88 E.00502
G1 X102.831 Y141.957 E.00591
G1 X102.79 Y141.954 E.00131
G1 X99.057 Y141.262 E.12021
G1 X99.057 Y142.625 E.04315
G3 X98.939 Y143.863 I-6.681 J-.012 E.03943
G1 X98.936 Y144.167 E.00963
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.612 J-43.497 E.5106
G3 X137.192 Y118.315 I40.344 J-33.116 E.49168
G1 X136.612 Y118.616 E.02068
G1 X135.912 Y118.859 E.02345
G1 X135.137 Y119.01 E.02501
G1 X134.369 Y119.041 E.02432
G1 X133.607 Y118.959 E.02427
G1 X132.865 Y118.766 E.02427
G1 X132.16 Y118.466 E.02427
G1 X131.506 Y118.066 E.02427
G1 X130.919 Y117.576 E.02422
G1 X130.408 Y117.003 E.0243
G3 X129.087 Y115.222 I546.359 J-406.994 E.07019
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-41.98 J-31.661 E.27953
; LINE_WIDTH: 0.554636
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.851 E.00539
G1 X122.202 Y120.927 E.0027
; WIPE_START
M204 S10000
G1 X121.44 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z5.88 F36000
G1 Z5.48
G1 E.4 F1800
; LINE_WIDTH: 0.556496
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.01281
; LINE_WIDTH: 0.544336
G1 X120.199 Y119.2 E.15232
; LINE_WIDTH: 0.554636
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.199 Y119.2 E-.13039
G1 X119.695 Y119.622 E-.24961
; WIPE_END
G1 E-.02 F1800
G1 X113.237 Y123.69 Z5.88 F36000
G1 X100.191 Y131.909 Z5.88
G1 Z5.48
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.759656
G1 F10857.278
G1 X100.504 Y131.832 E.01525
; LINE_WIDTH: 0.798636
G1 F10303.547
G1 X100.818 Y131.754 E.01607
; LINE_WIDTH: 0.837616
G1 F9803.556
G1 X101.131 Y131.676 E.01689
; LINE_WIDTH: 0.841776
G1 F9753.048
G1 X101.161 Y131.668 E.00167
; LINE_WIDTH: 0.887936
G1 F9225.634
G1 X101.475 Y131.586 E.01806
; LINE_WIDTH: 0.934096
G1 F8752.335
G1 X101.789 Y131.505 E.01903
; LINE_WIDTH: 0.980256
G1 F8325.229
G1 X102.103 Y131.423 E.02001
; LINE_WIDTH: 1.02642
G1 F7937.868
G1 X102.417 Y131.342 E.02098
; LINE_WIDTH: 1.04746
G1 F7773.018
G1 X102.544 Y131.308 E.00864
; WIPE_START
G1 X102.417 Y131.342 E-.0497
G1 X102.103 Y131.423 E-.12329
G1 X101.789 Y131.505 E-.12329
G1 X101.576 Y131.56 E-.08371
; WIPE_END
G1 E-.02 F1800
G1 X100.764 Y139.149 Z5.88 F36000
G1 X100.368 Y142.847 Z5.88
G1 Z5.48
G1 E.4 F1800
; LINE_WIDTH: 0.982796
G1 F8302.934
G1 X100.595 Y142.868 E.01409
; LINE_WIDTH: 0.941106
G1 F8684.673
G1 X100.844 Y142.891 E.01478
; LINE_WIDTH: 0.89539
G1 F9145.772
G1 X101.093 Y142.914 E.01403
; LINE_WIDTH: 0.849674
G1 F9658.578
G1 X101.342 Y142.937 E.01329
; LINE_WIDTH: 0.803958
G1 F10232.306
G1 X101.591 Y142.96 E.01254
; LINE_WIDTH: 0.758241
G1 F10878.5
G1 X101.84 Y142.983 E.0118
; LINE_WIDTH: 0.712525
G1 F11611.813
G1 X102.089 Y143.005 E.01105
; LINE_WIDTH: 0.666809
G1 F12451.136
G1 X102.338 Y143.028 E.01031
; LINE_WIDTH: 0.621092
G1 F13421.248
G1 X102.587 Y143.051 E.00956
; LINE_WIDTH: 0.575376
G1 F14555.302
G1 X103.063 Y143.052 E.01679
; WIPE_START
G1 X102.587 Y143.051 E-.18088
G1 X102.338 Y143.028 E-.095
G1 X102.089 Y143.005 E-.095
G1 X102.065 Y143.003 E-.00912
; WIPE_END
G1 E-.02 F1800
G1 X105.279 Y139.5 Z5.88 F36000
G1 Z5.48
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.279 Y137.157 E.08944
G3 X108.2 Y136.288 I3.178 J5.336 E.11757
G1 X108.671 Y136.399 E.01849
G1 X109.142 Y136.812 E.0239
G3 X110.556 Y139.351 I-216.904 J122.408 E.11098
G2 X113.865 Y141.945 I5.403 J-3.485 E.16353
G1 X116.534 Y141.945 E.10187
G1 X116.683 Y141.815 E.00758
G2 X118.097 Y139.275 I-217.154 J-122.547 E.11098
G3 X123.281 Y136.288 I5.614 J3.752 E.23665
G1 X123.752 Y136.399 E.01849
G1 X124.224 Y136.812 E.0239
G3 X125.638 Y139.351 I-216.904 J122.408 E.11098
G2 X128.947 Y141.945 I5.403 J-3.484 E.16353
G1 X131.615 Y141.945 E.10187
G1 X131.764 Y141.815 E.00758
G2 X133.178 Y139.275 I-217.605 J-122.798 E.11098
G3 X138.362 Y136.288 I5.614 J3.752 E.23665
G1 X138.834 Y136.399 E.01849
G1 X139.305 Y136.812 E.0239
G3 X140.719 Y139.351 I-217.354 J122.658 E.11098
G2 X144.028 Y141.945 I5.403 J-3.484 E.16353
G1 X146.696 Y141.945 E.10187
G1 X146.846 Y141.815 E.00758
G2 X148.26 Y139.275 I-217.605 J-122.798 E.11098
G1 X148.695 Y138.67 E.02847
G3 X143.828 Y133.509 I37.851 J-40.572 E.27099
G3 X140.248 Y134.798 I-3.974 J-5.424 E.14731
G1 X139.776 Y134.686 E.01849
G1 X139.305 Y134.274 E.0239
G3 X137.891 Y131.734 I217.027 J-122.477 E.11098
G2 X132.707 Y128.747 I-5.614 J3.752 E.23665
G1 X132.236 Y128.859 E.01849
G1 X131.764 Y129.271 E.0239
G2 X130.351 Y131.811 I216.904 J122.408 E.11098
G3 X125.166 Y134.798 I-5.614 J-3.752 E.23665
G1 X124.695 Y134.686 E.01849
G1 X124.224 Y134.274 E.0239
G3 X122.81 Y131.734 I217.251 J-122.602 E.11098
G2 X117.626 Y128.747 I-5.614 J3.752 E.23665
G1 X117.154 Y128.859 E.01849
G1 X116.683 Y129.271 E.0239
G2 X115.269 Y131.811 I217.354 J122.658 E.11098
G3 X110.085 Y134.798 I-5.614 J-3.752 E.23665
G1 X109.614 Y134.686 E.01849
G1 X109.142 Y134.274 E.0239
G3 X107.729 Y131.734 I217.477 J-122.727 E.11098
G2 X106.772 Y130.552 I-7.182 J4.832 E.05814
G2 X113.788 Y126.831 I-22.746 J-51.37 E.30344
G2 X115.741 Y127.257 I2.59 J-7.188 E.07652
G1 X116.212 Y127.145 E.01849
G2 X116.885 Y126.391 I-.804 J-1.395 E.03926
G2 X119.412 Y126.203 I1.094 J-2.363 E.10103
G1 X123.596 Y122.703 E.20826
G2 X124.263 Y121.797 I-2.497 J-2.54 E.04313
G3 X125.638 Y124.27 I-269.757 J151.541 E.10801
G2 X130.822 Y127.257 I5.614 J-3.752 E.23665
G1 X131.293 Y127.145 E.01849
G1 X131.764 Y126.733 E.0239
G2 X133.178 Y124.194 I-217.477 J-122.727 E.11098
G3 X136.262 Y121.697 I5.189 J3.256 E.15427
G1 X135.984 Y121.112 E.02472
G3 X128.725 Y118.442 I-1.405 J-7.383 E.3102
G3 X125.166 Y119.717 I-3.938 J-5.393 E.1463
G1 X124.695 Y119.605 E.01849
G3 X123.91 Y118.66 I1.036 J-1.66 E.04771
G2 X125.511 Y116.951 I-40.844 J-39.876 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 5.64
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X124.828 Y117.681 E-.38
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
G1 X123.51 Y122.061
G1 Z5.64
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.391 Y122.215 E.00743
M73 P66 R6
G3 X122.296 Y123.143 I-50.984 J-59.1 E.05481
G1 X119.23 Y125.712 E.15272
G1 X118.874 Y125.95 E.01636
G1 X118.45 Y126.11 E.01728
G1 X117.978 Y126.168 E.01818
G1 X117.493 Y126.107 E.01866
G1 X117.196 Y126.004 E.01201
G1 X116.82 Y125.787 E.01657
G3 X115.927 Y124.807 I5.234 J-5.664 E.05069
G3 X103.818 Y131.236 I-31.328 J-44.387 E.52484
G1 X104.293 Y131.565 E.02208
G1 X104.698 Y132.288 E.03166
G1 X104.783 Y132.855 E.02186
G1 X104.783 Y141.734 E.33901
G1 X104.688 Y142.336 E.02324
G1 X104.627 Y142.443 E.00472
G1 X154.069 Y142.443 E1.88767
G3 X143.803 Y132.701 I31.49 J-43.466 E.54193
G3 X136.271 Y120.545 I42.018 J-34.446 E.54757
G1 X135.451 Y120.705 E.03189
G3 X133.273 Y120.651 I-.886 J-8.189 E.08344
G1 X132.426 Y120.433 E.03339
G1 X131.481 Y120.051 E.03891
G1 X130.602 Y119.535 E.03892
G1 X129.81 Y118.896 E.03883
G1 X129.117 Y118.145 E.039
G3 X127.853 Y116.453 I76.747 J-58.658 E.08065
G1 X126.662 Y114.847 E.07636
G3 X122.955 Y118.919 I-43.097 J-35.511 E.21032
G3 X123.663 Y119.791 I-6.182 J5.746 E.04292
G1 X123.913 Y120.385 E.02462
G1 X123.965 Y120.833 E.01723
G1 X123.893 Y121.361 E.02035
G1 X123.702 Y121.812 E.01868
G1 X123.565 Y121.99 E.00856
G1 X123.013 Y121.746 F36000
G1 F13446.369
G1 X122.892 Y121.879 E.00687
G1 X118.854 Y125.264 E.20115
G1 X118.51 Y125.474 E.01539
G1 X118.094 Y125.577 E.01636
G1 X117.639 Y125.539 E.01746
G1 X117.188 Y125.33 E.01896
G1 X116.82 Y124.961 E.01991
G1 X116.03 Y124.018 E.04698
G3 X105.303 Y130.016 I-31.599 J-43.92 E.47024
G1 X104.06 Y130.507 E.05099
G1 X103.688 Y130.654 E.01527
G1 F12189.637
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.667307
G1 F10859.706
G1 X103.231 Y130.857 E.00422
; LINE_WIDTH: 0.714618
G1 F10531.712
G1 X103.145 Y130.913 E.00454
; LINE_WIDTH: 0.76193
G1 F10208.705
G1 X103.06 Y130.969 E.00486
; LINE_WIDTH: 0.809241
G1 F9890.728
G1 X102.974 Y131.025 E.00517
; LINE_WIDTH: 0.856552
G1 F9577.782
G1 X102.888 Y131.081 E.00549
; LINE_WIDTH: 0.903863
G1 F9056.652
G1 X102.803 Y131.137 E.0058
; LINE_WIDTH: 0.951174
G1 F8589.305
G1 X102.717 Y131.194 E.00612
; LINE_WIDTH: 0.998485
G1 F8167.826
G1 X102.631 Y131.25 E.00643
; LINE_WIDTH: 1.0458
G1 F7785.775
G1 X102.546 Y131.306 E.00675
G1 X102.629 Y131.335 E.00582
; LINE_WIDTH: 0.998485
G1 F8167.826
G1 X102.712 Y131.365 E.00555
; LINE_WIDTH: 0.951174
G1 F8589.305
G1 X102.795 Y131.395 E.00527
; LINE_WIDTH: 0.903863
G1 F9056.652
G1 X102.878 Y131.424 E.005
; LINE_WIDTH: 0.856552
G1 F9577.782
G1 X102.961 Y131.454 E.00473
; LINE_WIDTH: 0.809241
G1 F10162.544
G1 X103.044 Y131.484 E.00446
; LINE_WIDTH: 0.76193
G1 F10439.978
G1 X103.128 Y131.513 E.00418
; LINE_WIDTH: 0.714618
G1 F10721.148
G1 X103.211 Y131.543 E.00391
; LINE_WIDTH: 0.667307
G1 F11006.035
G1 X103.294 Y131.573 E.00364
; LINE_WIDTH: 0.619996
G1 F12344.637
G1 X103.625 Y131.797 E.01527
G1 F13319.127
G1 X103.855 Y131.953 E.0106
G1 F13446.369
G1 X104.138 Y132.459 E.02215
G1 X104.197 Y132.953 E.01899
G1 X104.197 Y141.734 E.33526
G1 X104.131 Y142.155 E.01626
G1 X103.86 Y142.632 E.02094
G1 X103.307 Y142.998 E.02535
; LINE_WIDTH: 0.591376
G1 F13896.063
G1 X103.064 Y143.054 E.00902
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.83 J53.023 E.10249
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.106 J-44.281 E.59949
G3 X136.596 Y119.828 I41.447 J-33.99 E.56143
G1 X136.064 Y119.995 E.02127
G1 X135.344 Y120.13 E.02798
G3 X133.38 Y120.075 I-.776 J-7.514 E.0752
G1 X132.575 Y119.867 E.03177
G1 X131.711 Y119.513 E.03563
G1 X130.909 Y119.036 E.03563
G1 X130.187 Y118.448 E.03555
G3 X129.066 Y117.106 I8.506 J-8.24 E.06683
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-43.157 J-34.61 E.25726
G1 X123.061 Y119.957 E.05366
G1 X123.304 Y120.384 E.01875
G1 X123.38 Y120.833 E.01739
G1 X123.329 Y121.203 E.01424
G1 X123.137 Y121.61 E.01718
G1 X123.074 Y121.68 E.0036
G1 X122.581 Y121.353 F36000
G1 F13446.369
G1 X122.516 Y121.43 E.00385
G1 X118.478 Y124.815 E.20115
G1 X118.225 Y124.956 E.01106
G1 X117.92 Y124.994 E.01176
G1 X117.665 Y124.931 E.01
G1 X117.381 Y124.718 E.01356
G1 X116.128 Y123.223 E.0745
G3 X99.499 Y131.449 I-31.738 J-43.235 E.71189
G3 X99.608 Y132.662 I-13.616 J1.833 E.0465
G1 X102.691 Y132.091 E.11972
G1 X103.109 Y132.129 E.01603
G1 X103.426 Y132.353 E.01482
G1 X103.582 Y132.645 E.01267
G3 X103.612 Y135.734 I-67.606 J2.183 E.11794
G1 X103.612 Y141.734 E.22907
G1 X103.574 Y141.974 E.00928
G1 X103.419 Y142.247 E.01195
G1 X103.106 Y142.463 E.01454
G1 X102.833 Y142.513 E.0106
G1 X102.691 Y142.5 E.00544
G1 X99.611 Y141.929 E.11958
G3 X99.532 Y143.615 I-10.336 J.359 E.06448
G1 X156.235 Y143.615 E2.16487
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.1 J-43.626 E.60776
G3 X136.912 Y119.083 I41.265 J-33.782 E.5764
G1 X136.093 Y119.382 E.03327
G3 X132.719 Y119.299 I-1.548 J-5.712 E.13066
G1 X131.942 Y118.975 E.03215
G1 X131.216 Y118.537 E.03235
G1 X130.564 Y118 E.03227
G3 X129.081 Y116.142 I12.07 J-11.158 E.09083
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-42.774 J-33.308 E.3043
G1 X122.612 Y120.333 E.07459
G1 X122.773 Y120.655 E.01371
G1 X122.786 Y120.948 E.0112
G1 X122.689 Y121.224 E.01119
G1 X122.639 Y121.284 E.00299
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00555
G1 X116.81 Y123.176 E.04903
; LINE_WIDTH: 0.521326
G1 X116.411 Y122.702 E.01968
; LINE_WIDTH: 0.556496
G1 X116.394 Y122.378 E.01104
G1 X115.476 Y123.015 E.03802
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.116 J-43.088 E.58495
G1 X98.942 Y131.54 E.0159
G1 X99.054 Y132.52 E.03123
G1 X99.058 Y133.326 E.02552
G1 X102.792 Y132.635 E.12024
G1 X102.913 Y132.646 E.00385
G1 X103.049 Y132.791 E.00629
G3 X103.059 Y135.734 I-169.974 J2.038 E.09319
G1 X103.059 Y141.734 E.18996
G1 X103.003 Y141.883 E.00502
G1 X102.833 Y141.96 E.00591
G1 X102.792 Y141.956 E.00131
G1 X99.059 Y141.265 E.1202
G1 X99.059 Y142.625 E.04307
G3 X98.939 Y143.864 I-6.597 J-.014 E.03947
G1 X98.936 Y144.167 E.0096
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.615 J-43.5 E.51061
G3 X137.192 Y118.315 I40.714 J-33.336 E.49165
G1 X136.612 Y118.616 E.02069
G1 X135.913 Y118.859 E.02343
G1 X135.135 Y119.01 E.02509
G1 X134.371 Y119.041 E.02421
G1 X133.607 Y118.959 E.02433
G1 X132.865 Y118.766 E.02427
G1 X132.16 Y118.466 E.02425
G1 X131.506 Y118.067 E.02426
G1 X130.92 Y117.577 E.0242
G1 X130.409 Y117.003 E.02433
G3 X129.087 Y115.222 I548.94 J-408.905 E.0702
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-42.348 J-31.99 E.27952
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.851 E.00539
G1 X122.202 Y120.927 E.0027
; WIPE_START
M204 S10000
G1 X121.44 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z6.04 F36000
G1 Z5.64
G1 E.4 F1800
; LINE_WIDTH: 0.556496
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.01281
; LINE_WIDTH: 0.544336
G1 X120.199 Y119.2 E.15232
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.199 Y119.2 E-.13039
G1 X119.695 Y119.622 E-.24961
; WIPE_END
G1 E-.02 F1800
G1 X113.237 Y123.69 Z6.04 F36000
G1 X100.189 Y131.909 Z6.04
G1 Z5.64
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.757423
G1 F10890.812
G1 X100.503 Y131.831 E.01523
; LINE_WIDTH: 0.79647
G1 F10332.838
G1 X100.817 Y131.753 E.01606
; LINE_WIDTH: 0.835516
G1 F9829.253
G1 X101.13 Y131.675 E.01688
; LINE_WIDTH: 0.839696
G1 F9778.237
G1 X101.161 Y131.667 E.00167
; LINE_WIDTH: 0.885851
G1 F9248.223
G1 X101.475 Y131.585 E.01801
; LINE_WIDTH: 0.932006
G1 F8772.713
G1 X101.789 Y131.504 E.01899
; LINE_WIDTH: 0.978161
G1 F8343.708
G1 X102.103 Y131.422 E.01996
; LINE_WIDTH: 1.02432
G1 F7954.706
G1 X102.417 Y131.341 E.02094
; LINE_WIDTH: 1.0458
G1 F7785.775
G1 X102.546 Y131.306 E.0088
; WIPE_START
G1 X102.417 Y131.341 E-.05071
G1 X102.103 Y131.422 E-.12329
G1 X101.789 Y131.504 E-.12329
G1 X101.578 Y131.559 E-.0827
; WIPE_END
G1 E-.02 F1800
G1 X100.765 Y139.148 Z6.04 F36000
G1 X100.369 Y142.849 Z6.04
G1 Z5.64
G1 E.4 F1800
; LINE_WIDTH: 0.980556
G1 F8322.59
G1 X100.597 Y142.87 E.01415
; LINE_WIDTH: 0.938586
G1 F8708.875
G1 X100.846 Y142.892 E.01474
; LINE_WIDTH: 0.89287
G1 F9172.616
G1 X101.095 Y142.915 E.01399
; LINE_WIDTH: 0.847154
G1 F9688.522
G1 X101.344 Y142.938 E.01325
; LINE_WIDTH: 0.801437
G1 F10265.92
G1 X101.593 Y142.961 E.0125
; LINE_WIDTH: 0.755721
G1 F10916.502
G1 X101.842 Y142.984 E.01176
; LINE_WIDTH: 0.710005
G1 F11655.121
G1 X102.091 Y143.007 E.01101
; LINE_WIDTH: 0.664289
G1 F12500.943
G1 X102.34 Y143.03 E.01027
; LINE_WIDTH: 0.618573
G1 F13479.137
G1 X102.589 Y143.052 E.00952
; LINE_WIDTH: 0.572856
G1 F14623.414
G1 X103.064 Y143.054 E.01669
; WIPE_START
G1 X102.589 Y143.052 E-.18071
G1 X102.34 Y143.03 E-.095
G1 X102.091 Y143.007 E-.095
G1 X102.067 Y143.005 E-.0093
; WIPE_END
G1 E-.02 F1800
G1 X105.281 Y139.433 Z6.04 F36000
G1 Z5.64
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.281 Y137.09 E.08944
G3 X108.671 Y135.779 I6.473 J11.697 E.13922
G1 X109.142 Y135.928 E.01886
G2 X110.556 Y139.215 I23.043 J-7.962 E.13676
G2 X113.612 Y141.945 I5.454 J-3.029 E.15939
G1 X116.988 Y141.945 E.1289
G3 X118.568 Y138.679 I13.597 J4.564 E.13891
G3 X122.339 Y136.192 I5.957 J4.93 E.17499
G3 X123.752 Y135.779 I3.312 J8.708 E.0563
G1 X124.224 Y135.928 E.01887
G2 X125.638 Y139.215 I23.043 J-7.962 E.13676
G2 X128.693 Y141.945 I5.454 J-3.029 E.15939
G1 X132.069 Y141.945 E.1289
G3 X133.65 Y138.679 I13.596 J4.563 E.13891
G3 X137.42 Y136.192 I5.957 J4.93 E.17499
G3 X138.834 Y135.779 I3.312 J8.708 E.0563
G1 X139.305 Y135.928 E.01886
G2 X140.719 Y139.215 I23.043 J-7.962 E.13676
G2 X143.774 Y141.945 I5.454 J-3.028 E.15939
G1 X147.15 Y141.945 E.1289
G3 X148.721 Y138.694 I13.589 J4.561 E.13824
G3 X143.843 Y133.529 I45.573 J-47.932 E.27136
G3 X141.19 Y134.893 I-5.853 J-8.123 E.11431
G3 X139.776 Y135.307 I-3.313 J-8.711 E.0563
G1 X139.305 Y135.158 E.01886
G2 X137.891 Y131.87 I-23.043 J7.962 E.13676
G2 X134.592 Y129.019 I-5.591 J3.135 E.16986
G2 X132.236 Y128.238 I-6.017 J14.215 E.09488
G1 X131.764 Y128.387 E.01886
G3 X130.351 Y131.675 I-23.043 J-7.962 E.13676
G3 X127.052 Y134.526 I-5.591 J-3.136 E.16986
G3 X124.695 Y135.307 I-6.018 J-14.216 E.09488
G1 X124.224 Y135.158 E.01886
G2 X122.81 Y131.87 I-23.044 J7.963 E.13676
G2 X119.511 Y129.019 I-5.591 J3.135 E.16986
G2 X117.154 Y128.238 I-6.017 J14.215 E.09488
G1 X116.683 Y128.387 E.01886
G3 X115.269 Y131.675 I-23.043 J-7.962 E.13676
G3 X111.97 Y134.526 I-5.591 J-3.135 E.16986
G3 X109.614 Y135.307 I-6.018 J-14.216 E.09488
G1 X109.142 Y135.158 E.01886
G2 X107.729 Y131.87 I-23.043 J7.962 E.13676
G2 X106.771 Y130.563 I-7.341 J4.371 E.06197
G2 X113.673 Y126.895 I-27.134 J-59.387 E.2986
G2 X116.212 Y127.766 I9.035 J-22.197 E.10252
G1 X116.683 Y127.617 E.01886
G1 X117.132 Y126.507 E.04573
G2 X119.366 Y126.231 I.819 J-2.544 E.08872
; WIPE_START
G1 X118.933 Y126.472 E-.18833
G1 X118.45 Y126.615 E-.19167
; WIPE_END
G1 E-.02 F1800
G1 X123.512 Y120.903 Z6.04 F36000
G1 X127.335 Y116.59 Z6.04
G1 Z5.64
G1 E.4 F1800
G1 F13446.283
G2 X128.742 Y118.462 I12.799 J-8.155 E.08951
G1 X127.994 Y118.977 E.03466
G3 X125.166 Y120.114 I-7.049 J-13.442 E.11656
G3 X124.349 Y120.116 I-.41 J-.686 E.03273
G3 X124.407 Y121.301 I-2.572 J.719 E.04564
G1 X125.166 Y123.177 E.07729
G2 X126.58 Y125.428 I7.669 J-3.248 E.10191
G2 X130.822 Y127.655 I6.556 J-7.334 E.18477
G2 X131.764 Y127.617 I.44 J-.812 E.0378
G1 X132.707 Y125.286 E.096
G3 X136.22 Y121.606 I6.079 J2.285 E.1996
G2 X137.257 Y123.706 I29.587 J-13.303 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 5.8
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X136.814 Y122.81 E-.38
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
G1 X123.51 Y122.061
G1 Z5.8
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.391 Y122.215 E.00743
G3 X122.296 Y123.143 I-51.038 J-59.164 E.0548
G1 X119.23 Y125.712 E.15272
G1 X118.874 Y125.95 E.01636
G1 X118.45 Y126.11 E.01728
G1 X117.978 Y126.168 E.01818
G1 X117.493 Y126.107 E.01866
G1 X117.196 Y126.004 E.01201
G1 X116.82 Y125.787 E.01657
G3 X115.927 Y124.807 I5.233 J-5.663 E.05069
G3 X103.821 Y131.235 I-31.885 J-45.434 E.52464
G1 X104.295 Y131.561 E.02196
G1 X104.699 Y132.283 E.0316
G1 X104.785 Y132.852 E.02197
G1 X104.785 Y141.737 E.33922
G1 X104.69 Y142.338 E.02324
G1 X104.63 Y142.443 E.00461
G1 X154.069 Y142.443 E1.88752
G3 X143.803 Y132.701 I32.294 J-44.311 E.54185
G3 X136.271 Y120.545 I41.663 J-34.226 E.54759
G1 X135.454 Y120.705 E.03178
G3 X133.273 Y120.651 I-.892 J-8.102 E.08352
G1 X132.426 Y120.433 E.03339
G1 X131.481 Y120.051 E.03893
G1 X130.604 Y119.536 E.03883
G1 X129.81 Y118.896 E.03892
G1 X129.117 Y118.145 E.03902
G3 X127.854 Y116.454 I76.929 J-58.793 E.08059
G1 X126.662 Y114.848 E.07636
G3 X122.955 Y118.919 I-43.14 J-35.567 E.21031
G3 X123.665 Y119.793 I-6.128 J5.7 E.04304
G1 X123.913 Y120.385 E.0245
G1 X123.965 Y120.833 E.01722
G1 X123.892 Y121.363 E.02041
G1 X123.702 Y121.813 E.01865
G1 X123.565 Y121.99 E.00855
G1 X123.013 Y121.746 F36000
G1 F13446.369
G1 X122.892 Y121.879 E.00687
G1 X118.854 Y125.264 E.20115
G1 X118.51 Y125.474 E.01539
G1 X118.094 Y125.577 E.01636
G1 X117.639 Y125.539 E.01746
G1 X117.188 Y125.33 E.01896
G1 X116.82 Y124.961 E.01991
G1 X116.03 Y124.018 E.04698
G3 X105.297 Y130.013 I-32.055 J-44.775 E.47036
G1 X104.06 Y130.507 E.05083
G1 X103.688 Y130.654 E.01527
G1 F12197.352
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.667125
G1 F10866.989
G1 X103.231 Y130.857 E.00421
; LINE_WIDTH: 0.714254
G1 F10539.863
G1 X103.146 Y130.912 E.00452
; LINE_WIDTH: 0.761383
G1 F10217.711
G1 X103.06 Y130.968 E.00484
; LINE_WIDTH: 0.808512
G1 F9900.558
G1 X102.975 Y131.024 E.00515
; LINE_WIDTH: 0.855641
G1 F9588.406
G1 X102.89 Y131.08 E.00546
; LINE_WIDTH: 0.90277
G1 F9068.053
G1 X102.804 Y131.136 E.00578
; LINE_WIDTH: 0.949898
G1 F8601.274
G1 X102.719 Y131.192 E.00609
; LINE_WIDTH: 0.997027
G1 F8180.195
G1 X102.633 Y131.248 E.00641
; LINE_WIDTH: 1.04416
G1 F7798.419
G1 X102.548 Y131.304 E.00672
G1 X102.631 Y131.334 E.00579
; LINE_WIDTH: 0.997027
G1 F8180.195
G1 X102.714 Y131.363 E.00552
; LINE_WIDTH: 0.949898
G1 F8601.274
G1 X102.797 Y131.393 E.00525
; LINE_WIDTH: 0.90277
G1 F9068.053
G1 X102.88 Y131.422 E.00498
; LINE_WIDTH: 0.855641
G1 F9588.406
G1 X102.963 Y131.452 E.00471
; LINE_WIDTH: 0.808512
G1 F10172.112
G1 X103.046 Y131.481 E.00444
; LINE_WIDTH: 0.761383
G1 F10448.92
G1 X103.128 Y131.51 E.00417
; LINE_WIDTH: 0.714254
G1 F10729.443
G1 X103.211 Y131.54 E.0039
; LINE_WIDTH: 0.667125
G1 F11013.683
G1 X103.294 Y131.569 E.00363
; LINE_WIDTH: 0.619996
G1 F12352.737
G1 X103.626 Y131.793 E.01527
G1 F13329.874
G1 X103.856 Y131.949 E.01063
G1 F13446.369
G1 X104.139 Y132.454 E.02211
G1 X104.199 Y132.95 E.01905
G1 X104.199 Y141.737 E.33548
G1 X104.133 Y142.158 E.01626
G1 X103.862 Y142.635 E.02094
G1 X103.308 Y143 E.02534
; LINE_WIDTH: 0.589876
G1 F14175.399
G1 X103.066 Y143.055 E.00899
; LINE_WIDTH: 0.594026
G1 F14070.29
G1 X103.554 Y143.042 E.01779
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.341 J41.206 E.0838
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I29.803 J-43.952 E.59953
G3 X136.596 Y119.828 I41.127 J-33.793 E.56145
G1 X136.065 Y119.995 E.02126
G1 X135.347 Y120.129 E.02789
G1 X134.411 Y120.179 E.03576
G3 X133.286 Y120.054 I.625 J-10.795 E.04326
G1 X132.566 Y119.865 E.02839
G1 X131.712 Y119.513 E.03529
G1 X130.911 Y119.037 E.03555
G1 X130.187 Y118.448 E.03564
G3 X129.066 Y117.106 I8.513 J-8.246 E.06683
G1 X126.683 Y113.893 E.15272
G3 X122.158 Y118.88 I-42.837 J-34.319 E.25726
G1 X123.061 Y119.957 E.05366
G1 X123.303 Y120.384 E.01873
G1 X123.38 Y120.833 E.01741
G1 X123.329 Y121.204 E.01428
G1 X123.137 Y121.61 E.01715
G1 X123.074 Y121.68 E.00359
G1 X122.581 Y121.353 F36000
G1 F13446.369
G1 X122.516 Y121.43 E.00385
G1 X118.478 Y124.815 E.20115
G1 X118.225 Y124.956 E.01106
G1 X117.92 Y124.994 E.01176
G1 X117.665 Y124.931 E.01
G1 X117.381 Y124.718 E.01356
G1 X116.128 Y123.223 E.0745
G3 X99.498 Y131.45 I-31.738 J-43.235 E.71194
G3 X99.61 Y132.66 I-12.066 J1.728 E.04641
G1 X102.693 Y132.089 E.11972
G1 X103.111 Y132.126 E.01599
G1 X103.428 Y132.35 E.01483
G1 X103.584 Y132.642 E.01264
G3 X103.614 Y135.737 I-67.004 J2.187 E.11819
G1 X103.614 Y141.737 E.22907
G1 X103.576 Y141.977 E.00928
G1 X103.421 Y142.249 E.01195
G1 X103.108 Y142.466 E.01454
G1 X102.835 Y142.515 E.0106
G1 X102.693 Y142.502 E.00544
G1 X99.614 Y141.932 E.11958
G3 X99.533 Y143.615 I-10.171 J.358 E.06439
G1 X156.235 Y143.615 E2.16483
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.325 J-43.872 E.60774
G3 X136.912 Y119.083 I41.193 J-33.738 E.5764
G1 X136.092 Y119.382 E.03329
G3 X132.72 Y119.299 I-1.548 J-5.719 E.13062
G1 X131.942 Y118.975 E.03215
G1 X131.218 Y118.539 E.03227
G1 X130.564 Y118 E.03237
G3 X129.081 Y116.142 I12.083 J-11.168 E.09082
G1 X126.698 Y112.93 E.15272
G3 X121.357 Y118.836 I-42.557 J-33.111 E.3043
G1 X122.612 Y120.333 E.07459
G1 X122.773 Y120.655 E.01371
G1 X122.786 Y120.948 E.01121
G1 X122.689 Y121.224 E.01119
G1 X122.639 Y121.284 E.00298
M204 S250
G1 X122.161 Y121.007 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.123 Y124.391 E.16681
M73 P67 R6
G1 X117.961 Y124.443 E.00538
G1 X117.805 Y124.363 E.00555
G1 X116.81 Y123.176 E.04903
; LINE_WIDTH: 0.521316
G1 X116.411 Y122.702 E.01968
; LINE_WIDTH: 0.556496
G1 X116.394 Y122.378 E.01104
G1 X115.476 Y123.015 E.03802
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.116 J-43.088 E.58495
G1 X98.943 Y131.539 E.01589
G1 X99.056 Y132.521 E.03128
G1 X99.06 Y133.324 E.02542
G1 X102.794 Y132.632 E.12024
G1 X102.915 Y132.643 E.00384
G1 X103.051 Y132.788 E.00629
G3 X103.061 Y135.737 I-168.717 J2.04 E.09336
G1 X103.061 Y141.737 E.18996
G1 X103.005 Y141.885 E.00502
G1 X102.835 Y141.962 E.00591
G1 X102.794 Y141.959 E.00131
G1 X99.061 Y141.267 E.1202
G1 X99.061 Y142.625 E.04299
G3 X98.939 Y143.865 I-6.502 J-.014 E.0395
G1 X98.936 Y144.167 E.00958
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.615 J-43.5 E.51062
G3 X137.192 Y118.315 I40.271 J-33.072 E.49168
G1 X136.612 Y118.616 E.02068
G1 X135.913 Y118.859 E.02345
G1 X135.138 Y119.01 E.02499
G1 X134.369 Y119.041 E.02436
G1 X133.608 Y118.959 E.02423
G1 X132.866 Y118.766 E.02429
G1 X132.16 Y118.467 E.02426
G1 X131.508 Y118.068 E.0242
G1 X130.92 Y117.577 E.02427
G1 X130.409 Y117.003 E.02433
G3 X129.087 Y115.222 I547.292 J-407.683 E.07019
G1 X126.704 Y112.01 E.12664
G3 X120.822 Y118.581 I-42.172 J-31.832 E.27953
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.0182
G1 X120.795 Y119.027 E.01168
; LINE_WIDTH: 0.519996
G1 X122.189 Y120.689 E.06866
G1 X122.241 Y120.851 E.00539
G1 X122.202 Y120.927 E.0027
; WIPE_START
M204 S10000
G1 X121.44 Y121.574 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.394 Y122.378 Z6.2 F36000
G1 Z5.8
G1 E.4 F1800
; LINE_WIDTH: 0.556496
G1 F3600
M204 S5000
G1 X116.688 Y122.143 E.01281
; LINE_WIDTH: 0.544336
G1 X120.198 Y119.2 E.15231
; LINE_WIDTH: 0.554616
G1 X120.455 Y118.972 E.01164
; WIPE_START
M204 S10000
G1 X120.198 Y119.2 E-.13045
G1 X119.695 Y119.622 E-.24955
; WIPE_END
G1 E-.02 F1800
G1 X113.237 Y123.69 Z6.2 F36000
G1 X100.192 Y131.907 Z6.2
G1 Z5.8
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.75555
G1 F10919.1
G1 X100.504 Y131.829 E.01515
; LINE_WIDTH: 0.794483
G1 F10359.844
G1 X100.817 Y131.752 E.01597
; LINE_WIDTH: 0.833416
G1 F9855.085
G1 X101.13 Y131.674 E.01679
; LINE_WIDTH: 0.837596
G1 F9803.8
G1 X101.161 Y131.666 E.00166
; LINE_WIDTH: 0.883756
G1 F9271.033
G1 X101.475 Y131.584 E.01797
; LINE_WIDTH: 0.929916
G1 F8793.185
G1 X101.789 Y131.503 E.01894
; LINE_WIDTH: 0.976076
G1 F8362.181
G1 X102.103 Y131.421 E.01992
; LINE_WIDTH: 1.02224
G1 F7971.455
G1 X102.417 Y131.34 E.0209
; LINE_WIDTH: 1.04416
G1 F7798.419
G1 X102.548 Y131.304 E.00896
; WIPE_START
G1 X102.417 Y131.34 E-.05173
G1 X102.103 Y131.421 E-.12329
G1 X101.789 Y131.503 E-.12329
G1 X101.581 Y131.557 E-.08168
; WIPE_END
G1 E-.02 F1800
G1 X100.767 Y139.146 Z6.2 F36000
G1 X100.37 Y142.85 Z6.2
G1 Z5.8
G1 E.4 F1800
; LINE_WIDTH: 0.978296
G1 F8342.516
G1 X100.6 Y142.871 E.01421
; LINE_WIDTH: 0.936056
G1 F8733.31
G1 X100.849 Y142.894 E.0147
; LINE_WIDTH: 0.890339
G1 F9199.74
G1 X101.098 Y142.917 E.01395
; LINE_WIDTH: 0.844621
G1 F9718.804
G1 X101.346 Y142.939 E.01321
; LINE_WIDTH: 0.798904
G1 F10299.941
G1 X101.595 Y142.962 E.01246
; LINE_WIDTH: 0.753186
G1 F10954.999
G1 X101.844 Y142.985 E.01172
; LINE_WIDTH: 0.707469
G1 F11699.035
G1 X102.093 Y143.008 E.01097
; LINE_WIDTH: 0.661751
G1 F12551.501
G1 X102.342 Y143.031 E.01023
; LINE_WIDTH: 0.616034
G1 F13537.966
G1 X102.591 Y143.054 E.00948
; LINE_WIDTH: 0.570316
G1 F14692.714
G1 X103.066 Y143.055 E.0166
; WIPE_START
G1 X102.591 Y143.054 E-.18054
G1 X102.342 Y143.031 E-.095
G1 X102.093 Y143.008 E-.095
G1 X102.069 Y143.006 E-.00946
; WIPE_END
G1 E-.02 F1800
G1 X105.283 Y139.353 Z6.2 F36000
G1 Z5.8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.283 Y137.01 E.08944
G2 X107.987 Y135.543 I-121.776 J-227.576 E.11746
G1 X108.383 Y135.072 E.02351
G1 X108.468 Y134.6 E.01828
G2 X106.754 Y130.56 I-7.63 J.852 E.16994
G2 X113.567 Y126.963 I-26.152 J-57.782 E.2943
G3 X115.527 Y128.002 I-20.373 J40.801 E.08474
G1 X115.924 Y128.473 E.02351
G1 X116.008 Y128.945 E.01828
G3 X112.087 Y134.6 I-6.827 J-.547 E.27529
G2 X110.298 Y135.543 I6.771 J15.015 E.07723
G1 X109.902 Y136.014 E.02351
G1 X109.817 Y136.485 E.01828
G2 X113.391 Y141.945 I6.764 J-.527 E.25991
G1 X117.388 Y141.945 E.15259
G3 X121.28 Y136.485 I6.76 J.701 E.26772
G2 X123.068 Y135.543 I-6.77 J-15.012 E.07723
G1 X123.464 Y135.072 E.02351
G1 X123.549 Y134.6 E.01828
G2 X119.627 Y128.945 I-6.827 J.547 E.27529
G3 X117.839 Y128.002 I6.771 J-15.014 E.07723
G1 X117.443 Y127.531 E.02351
G3 X117.427 Y126.602 I1.357 J-.487 E.03611
G2 X119.412 Y126.203 I.538 J-2.454 E.07953
G1 X123.593 Y122.705 E.2081
G2 X123.628 Y118.947 I-1.629 J-1.895 E.16245
G2 X126.639 Y115.651 I-38.692 J-38.376 E.17047
G1 X128.769 Y118.493 E.13559
G3 X126.207 Y119.99 I-7.83 J-10.455 E.11352
G1 X125.38 Y120.462 E.03637
G1 X124.983 Y120.933 E.02351
G1 X124.899 Y121.404 E.01828
G2 X128.82 Y127.06 I6.827 J-.547 E.27529
G3 X130.609 Y128.002 I-6.771 J15.014 E.07723
G1 X131.005 Y128.473 E.02351
G1 X131.089 Y128.945 E.01828
G3 X127.168 Y134.6 I-6.827 J-.547 E.27529
G2 X125.38 Y135.543 I6.77 J15.012 E.07723
G1 X124.983 Y136.014 E.02351
G1 X124.899 Y136.485 E.01828
G2 X128.472 Y141.945 I6.764 J-.527 E.25991
G1 X132.469 Y141.945 E.15259
G3 X136.361 Y136.485 I6.76 J.701 E.26772
G2 X138.149 Y135.543 I-6.771 J-15.015 E.07723
G1 X138.546 Y135.072 E.02351
G1 X138.63 Y134.6 E.01828
G2 X134.709 Y128.945 I-6.827 J.547 E.27529
G3 X132.92 Y128.002 I6.771 J-15.015 E.07723
G1 X132.524 Y127.531 E.02351
G1 X132.439 Y127.06 E.01828
G3 X136.172 Y121.51 I6.785 J.534 E.26693
G2 X143.867 Y133.559 I49.558 J-23.169 E.54739
G3 X141.289 Y135.072 I-7.794 J-10.331 E.11439
G1 X140.461 Y135.543 E.03637
G1 X140.064 Y136.014 E.02351
G1 X139.98 Y136.485 E.01828
G2 X143.554 Y141.945 I6.763 J-.527 E.25991
G1 X147.55 Y141.945 E.15259
G3 X148.766 Y138.727 I6.533 J.629 E.13291
G2 X150.515 Y140.285 I34.868 J-37.39 E.08943
; CHANGE_LAYER
; Z_HEIGHT: 5.96
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X149.769 Y139.62 E-.38
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
G1 X123.451 Y122.109
G1 Z5.96
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.329 Y122.268 E.00767
G3 X122.357 Y123.091 I-41.55 J-48.035 E.04862
G1 X119.292 Y125.661 E.15272
G1 X118.936 Y125.898 E.0163
G1 X118.515 Y126.058 E.01721
G3 X117.39 Y126.005 I-.464 J-2.144 E.04349
G1 X116.984 Y125.806 E.01726
G1 X116.545 Y125.419 E.02234
G1 X115.992 Y124.76 E.03283
G3 X103.811 Y131.238 I-31.999 J-45.483 E.5281
G1 X104.381 Y131.66 E.02709
G1 X104.727 Y132.371 E.03019
G3 X104.787 Y133.739 I-6.522 J.972 E.05238
G1 X104.787 Y141.739 E.30543
G1 X104.692 Y142.341 E.02324
G1 X104.634 Y142.443 E.0045
G1 X154.069 Y142.443 E1.88739
G3 X143.803 Y132.702 I31.818 J-43.809 E.54187
G3 X136.333 Y120.681 I42.1 J-34.493 E.54186
G1 X136.269 Y120.54 E.00592
G3 X134.093 Y120.75 I-1.842 J-7.697 E.08373
G1 X133.083 Y120.609 E.03894
G3 X131.465 Y120.045 I1.836 J-7.862 E.06554
G1 X130.604 Y119.536 E.03817
G1 X129.809 Y118.895 E.03898
G1 X129.117 Y118.145 E.03897
G3 X127.853 Y116.453 I70.462 J-53.961 E.08064
G1 X126.662 Y114.847 E.07636
G3 X122.897 Y118.975 I-42.502 J-34.974 E.21338
G1 X123.449 Y119.632 E.03275
G1 X123.63 Y119.889 E.012
G1 X123.85 Y120.428 E.02224
G1 X123.904 Y120.873 E.01713
G1 X123.834 Y121.402 E.02035
G1 X123.647 Y121.854 E.01867
G1 X123.506 Y122.037 E.00883
G1 X122.953 Y121.794 F36000
G1 F13446.369
G1 X122.831 Y121.93 E.007
G1 X118.916 Y125.212 E.19504
G1 X118.573 Y125.422 E.01534
G1 X118.159 Y125.526 E.0163
G1 X117.775 Y125.505 E.01468
G1 X117.325 Y125.329 E.01843
G1 X116.942 Y124.981 E.01977
G1 X116.095 Y123.971 E.05031
G3 X104.399 Y130.384 I-31.76 J-44.058 E.51056
G1 X104.065 Y130.512 E.01365
G1 X103.692 Y130.656 E.01527
G1 F12209.086
G1 X103.319 Y130.8 E.01527
; LINE_WIDTH: 0.666938
G1 F10878.064
G1 X103.233 Y130.856 E.00421
; LINE_WIDTH: 0.713881
G1 F10550.901
G1 X103.148 Y130.911 E.00452
; LINE_WIDTH: 0.760823
G1 F10228.707
G1 X103.063 Y130.967 E.00483
; LINE_WIDTH: 0.807765
G1 F9911.491
G1 X102.977 Y131.023 E.00514
; LINE_WIDTH: 0.854707
G1 F9599.315
G1 X102.892 Y131.079 E.00546
; LINE_WIDTH: 0.90165
G1 F9079.765
G1 X102.806 Y131.135 E.00577
; LINE_WIDTH: 0.948592
G1 F8613.566
G1 X102.721 Y131.191 E.00608
; LINE_WIDTH: 0.995534
G1 F8192.903
G1 X102.636 Y131.247 E.00639
; LINE_WIDTH: 1.04248
G1 F7811.415
G1 X102.55 Y131.302 E.00671
G1 X102.633 Y131.332 E.00577
; LINE_WIDTH: 0.995534
G1 F8192.903
G1 X102.716 Y131.361 E.0055
; LINE_WIDTH: 0.948592
G1 F8613.566
G1 X102.799 Y131.39 E.00523
; LINE_WIDTH: 0.90165
G1 F9079.765
G1 X102.881 Y131.42 E.00496
; LINE_WIDTH: 0.854707
G1 F9599.315
G1 X102.964 Y131.449 E.00469
; LINE_WIDTH: 0.807765
G1 F10181.933
G1 X103.047 Y131.478 E.00443
; LINE_WIDTH: 0.760823
G1 F10458.166
G1 X103.129 Y131.508 E.00416
; LINE_WIDTH: 0.713881
G1 F10738.107
G1 X103.212 Y131.537 E.00389
; LINE_WIDTH: 0.666938
G1 F11021.745
G1 X103.295 Y131.566 E.00362
; LINE_WIDTH: 0.619996
G1 F12361.276
G1 X103.619 Y131.801 E.01527
G1 F13446.369
G1 X103.917 Y132.018 E.01409
G1 X104.159 Y132.515 E.02112
G1 X104.201 Y132.947 E.01654
G1 X104.201 Y141.739 E.3357
G1 X104.135 Y142.16 E.01626
G1 X103.864 Y142.637 E.02094
G1 X103.31 Y143.002 E.02533
; LINE_WIDTH: 0.588396
G1 F14213.263
G1 X103.068 Y143.056 E.00896
; LINE_WIDTH: 0.592766
G1 F14102.038
G1 X103.555 Y143.043 E.01774
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.34 J39.215 E.08375
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I29.803 J-43.953 E.59951
G3 X136.598 Y119.835 I41.246 J-33.867 E.5612
G3 X134.836 Y120.173 I-2.282 J-7.118 E.0687
G1 X134.1 Y120.165 E.02808
G1 X133.176 Y120.031 E.03565
G1 X132.407 Y119.809 E.03054
G1 X131.701 Y119.509 E.02931
G1 X130.911 Y119.037 E.03511
G1 X130.186 Y118.447 E.0357
G3 X129.066 Y117.106 I8.267 J-8.038 E.06679
G1 X126.683 Y113.893 E.15272
G3 X122.101 Y118.936 I-42.905 J-34.377 E.26033
G1 X123 Y120.009 E.05342
G1 X123.246 Y120.446 E.01915
G1 X123.318 Y120.877 E.0167
G1 X123.27 Y121.247 E.01424
G1 X123.085 Y121.648 E.01686
G1 X123.014 Y121.727 E.00408
G1 X122.52 Y121.403 F36000
G1 F13446.369
G1 X122.454 Y121.482 E.00389
G1 X118.539 Y124.763 E.19504
G1 X118.288 Y124.905 E.01102
G3 X117.618 Y124.821 I-.239 J-.817 E.0265
G1 X117.349 Y124.556 E.01442
G1 X116.192 Y123.175 E.06876
G3 X99.497 Y131.45 I-31.69 J-42.959 E.71508
G3 X99.612 Y132.657 I-11.423 J1.701 E.04631
G1 X102.695 Y132.086 E.11973
G1 X103.113 Y132.124 E.01601
G1 X103.463 Y132.389 E.01678
G1 X103.595 Y132.675 E.01203
G3 X103.616 Y135.739 I-93.761 J2.153 E.117
G1 X103.616 Y141.739 E.22907
G1 X103.578 Y141.979 E.00928
G1 X103.423 Y142.252 E.01195
G1 X103.11 Y142.468 E.01454
G1 X102.837 Y142.518 E.0106
G1 X102.695 Y142.505 E.00544
G1 X99.616 Y141.934 E.11958
G3 X99.536 Y143.615 I-11.201 J.312 E.06428
G1 X156.235 Y143.615 E2.16472
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.705 Y131.955 I29.326 J-43.872 E.60772
G3 X136.908 Y119.075 I41.091 J-33.677 E.57675
G1 X136.5 Y119.253 E.01699
G1 X135.794 Y119.458 E.02809
G1 X134.957 Y119.581 E.03229
G1 X134.387 Y119.582 E.02176
G3 X131.937 Y118.973 I.108 J-5.662 E.09721
G1 X131.218 Y118.539 E.03206
G1 X130.563 Y117.999 E.03241
G3 X129.081 Y116.142 I11.763 J-10.91 E.09078
G1 X126.698 Y112.93 E.15272
G3 X121.299 Y118.891 I-42.486 J-33.052 E.30736
G1 X122.551 Y120.385 E.07441
G1 X122.711 Y120.703 E.01359
G1 X122.725 Y120.995 E.01117
G1 X122.63 Y121.272 E.01118
G1 X122.578 Y121.334 E.00312
M204 S250
G1 X122.099 Y121.058 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.184 Y124.34 E.16174
G1 X118.023 Y124.392 E.00536
G3 X117.81 Y124.245 I.067 J-.324 E.0084
G1 X117.445 Y123.809 E.01803
G1 X116.505 Y122.688 E.0463
; LINE_WIDTH: 0.521546
G1 X116.472 Y122.651 E.00159
; LINE_WIDTH: 0.552376
G1 X116.457 Y122.328 E.01091
G1 X116 Y122.631 E.01848
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.549 J-42.515 E.60555
G1 X98.943 Y131.539 E.01588
G1 X99.058 Y132.521 E.03128
G1 X99.062 Y133.321 E.02535
G1 X102.796 Y132.63 E.12024
G3 X103.019 Y132.717 I.041 J.222 E.00799
G1 X103.063 Y132.947 E.00739
G1 X103.063 Y141.739 E.27838
G1 X103.007 Y141.888 E.00502
G1 X102.837 Y141.965 E.00591
G1 X102.796 Y141.961 E.00131
G1 X99.063 Y141.27 E.1202
G1 X99.063 Y142.625 E.04291
G3 X98.939 Y143.865 I-6.742 J-.047 E.03952
G1 X98.936 Y144.167 E.00956
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.614 J-43.499 E.5106
G3 X137.192 Y118.316 I40.814 J-33.397 E.49161
G3 X136.346 Y118.722 I-5.225 J-9.8 E.02972
G1 X135.64 Y118.927 E.02329
G1 X134.882 Y119.033 E.0242
G1 X134.283 Y119.028 E.01897
G1 X133.357 Y118.907 E.02959
G3 X132.16 Y118.467 I1.249 J-5.243 E.04047
G1 X131.508 Y118.068 E.02419
G1 X130.919 Y117.576 E.0243
G1 X130.409 Y117.003 E.02429
G3 X129.087 Y115.222 I786.316 J-585.136 E.0702
G1 X126.704 Y112.01 E.12664
G3 X120.76 Y118.638 I-42.157 J-31.82 E.28221
; LINE_WIDTH: 0.550116
G1 X120.396 Y119.025 E.01788
G1 X120.734 Y119.078 E.01152
; LINE_WIDTH: 0.519996
G1 X120.772 Y119.123 E.00185
G1 X120.809 Y119.168 E.00185
G1 X121.236 Y119.677 E.02105
G1 X121.682 Y120.209 E.02196
G1 X122.127 Y120.74 E.02196
G1 X122.179 Y120.901 E.00535
G1 X122.14 Y120.978 E.00273
; WIPE_START
M204 S10000
G1 X121.378 Y121.625 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.457 Y122.328 Z6.36 F36000
G1 Z5.96
G1 E.4 F1800
; LINE_WIDTH: 0.552376
G1 F3600
M204 S5000
G1 X116.687 Y122.143 E.00999
; LINE_WIDTH: 0.544336
G1 X120.198 Y119.2 E.15232
; LINE_WIDTH: 0.550116
G1 X120.396 Y119.025 E.00885
; WIPE_START
M204 S10000
G1 X120.198 Y119.2 E-.10004
G1 X119.634 Y119.673 E-.27996
; WIPE_END
G1 E-.02 F1800
M73 P68 R6
G1 X113.174 Y123.738 Z6.36 F36000
G1 X100.192 Y131.906 Z6.36
G1 Z5.96
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.75351
G1 F10950.074
G1 X100.505 Y131.828 E.01511
; LINE_WIDTH: 0.792443
G1 F10387.721
G1 X100.817 Y131.751 E.01592
; LINE_WIDTH: 0.831376
G1 F9880.308
G1 X101.13 Y131.673 E.01674
; LINE_WIDTH: 0.835516
G1 F9829.253
G1 X101.161 Y131.665 E.00165
; LINE_WIDTH: 0.881661
G1 F9293.955
G1 X101.474 Y131.583 E.01792
; LINE_WIDTH: 0.927806
G1 F8813.951
G1 X101.788 Y131.502 E.01889
; LINE_WIDTH: 0.973951
G1 F8381.093
G1 X102.102 Y131.42 E.01987
; LINE_WIDTH: 1.0201
G1 F7988.76
G1 X102.416 Y131.339 E.02084
; LINE_WIDTH: 1.04248
G1 F7811.415
G1 X102.55 Y131.302 E.00914
; WIPE_START
G1 X102.416 Y131.339 E-.05283
G1 X102.102 Y131.42 E-.12324
G1 X101.788 Y131.502 E-.12324
G1 X101.583 Y131.555 E-.08068
; WIPE_END
G1 E-.02 F1800
G1 X100.767 Y139.144 Z6.36 F36000
G1 X100.369 Y142.851 Z6.36
G1 Z5.96
G1 E.4 F1800
; LINE_WIDTH: 0.976276
G1 F8360.406
G1 X100.602 Y142.872 E.01436
; LINE_WIDTH: 0.933506
G1 F8758.077
G1 X100.851 Y142.895 E.01465
; LINE_WIDTH: 0.887792
G1 F9227.187
G1 X101.1 Y142.918 E.01391
; LINE_WIDTH: 0.842079
G1 F9749.395
G1 X101.349 Y142.941 E.01316
; LINE_WIDTH: 0.796365
G1 F10334.256
G1 X101.598 Y142.964 E.01242
; LINE_WIDTH: 0.750651
G1 F10993.768
G1 X101.847 Y142.986 E.01167
; LINE_WIDTH: 0.704938
G1 F11743.194
G1 X102.096 Y143.009 E.01093
; LINE_WIDTH: 0.659224
G1 F12602.27
G1 X102.345 Y143.032 E.01018
; LINE_WIDTH: 0.61351
G1 F13596.957
G1 X102.594 Y143.055 E.00944
; LINE_WIDTH: 0.567796
G1 F14762.121
G1 X103.068 Y143.056 E.01651
; WIPE_START
G1 X102.594 Y143.055 E-.18037
G1 X102.345 Y143.032 E-.095
G1 X102.096 Y143.009 E-.095
G1 X102.07 Y143.007 E-.00964
; WIPE_END
G1 E-.02 F1800
G1 X105.285 Y139.266 Z6.36 F36000
G1 Z5.96
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.285 Y136.923 E.08944
G2 X107.496 Y135.543 I-80.721 J-131.722 E.09952
G2 X108.132 Y134.6 I-.89 J-1.286 E.04443
G2 X107.536 Y131.772 I-5.82 J-.251 E.11149
G2 X106.759 Y130.56 I-4.479 J2.017 E.05521
G2 X113.463 Y127.03 I-22.722 J-51.295 E.2895
G3 X115.036 Y128.002 I-14.568 J25.34 E.07062
G3 X115.672 Y128.945 I-.89 J1.286 E.04443
G3 X115.077 Y131.772 I-5.82 J.251 E.11149
G3 X112.317 Y134.6 I-5.82 J-2.92 E.15329
G2 X110.344 Y136.014 I2.94 J6.185 E.09315
G1 X110.153 Y136.485 E.01941
G2 X110.749 Y139.313 I5.82 J.251 E.11149
G2 X113.204 Y141.945 I5.601 J-2.763 E.13941
G1 X117.694 Y141.945 E.17142
G3 X118.289 Y139.313 I5.722 J-.089 E.10402
G3 X121.05 Y136.485 I5.82 J2.92 E.15329
G2 X123.022 Y135.072 I-2.94 J-6.185 E.09315
G1 X123.213 Y134.6 E.01941
G2 X122.618 Y131.772 I-5.82 J-.251 E.11149
G2 X119.857 Y128.945 I-5.82 J2.92 E.15329
G3 X117.885 Y127.531 I2.941 J-6.185 E.09315
G3 X117.694 Y126.581 I1.077 J-.71 E.03791
G2 X119.474 Y126.151 I.263 J-2.807 E.07119
G1 X123.531 Y122.757 E.20195
G2 X123.571 Y119.004 I-1.686 J-1.895 E.16134
G2 X126.637 Y115.65 I-37.61 J-37.464 E.17355
G1 X128.793 Y118.521 E.13709
G3 X126.586 Y119.99 I-13.163 J-17.38 E.10128
G2 X125.425 Y120.933 I1.416 J2.93 E.05761
G1 X125.235 Y121.404 E.01941
G2 X125.83 Y124.232 I5.82 J.251 E.11149
G2 X128.59 Y127.06 I5.82 J-2.92 E.15329
G3 X130.563 Y128.473 I-2.941 J6.185 E.09315
G1 X130.754 Y128.945 E.01941
G3 X130.158 Y131.772 I-5.82 J.251 E.11149
G3 X127.398 Y134.6 I-5.82 J-2.92 E.15329
G2 X125.425 Y136.014 I2.94 J6.185 E.09315
G1 X125.235 Y136.485 E.01941
G2 X125.83 Y139.313 I5.82 J.251 E.11149
G2 X128.285 Y141.945 I5.601 J-2.763 E.13941
G1 X132.775 Y141.945 E.17142
G3 X133.371 Y139.313 I5.721 J-.089 E.10402
G3 X136.131 Y136.485 I5.82 J2.92 E.15329
G2 X138.103 Y135.072 I-2.94 J-6.185 E.09315
G1 X138.294 Y134.6 E.01941
G2 X137.699 Y131.772 I-5.82 J-.251 E.11149
G2 X134.939 Y128.945 I-5.82 J2.92 E.15329
G3 X132.966 Y127.531 I2.94 J-6.185 E.09315
G1 X132.775 Y127.06 E.01941
G3 X133.371 Y124.232 I5.82 J-.251 E.11149
G3 X136.126 Y121.407 I5.815 J2.917 E.15305
G2 X143.89 Y133.587 I50.099 J-23.374 E.55304
G3 X141.668 Y135.072 I-12.628 J-16.499 E.10211
G2 X140.507 Y136.014 I1.416 J2.93 E.05761
G1 X140.316 Y136.485 E.01941
G2 X140.911 Y139.313 I5.82 J.251 E.11149
G2 X143.367 Y141.945 I5.601 J-2.763 E.13941
G1 X147.856 Y141.945 E.17142
G3 X148.785 Y138.75 I5.413 J-.159 E.12914
G2 X150.539 Y140.303 I38.959 J-42.201 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 6.12
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X149.79 Y139.64 E-.38
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
G1 X123.277 Y122.273
G1 Z6.12
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X123.128 Y122.445 E.00869
G3 X122.48 Y122.989 I-218.78 J-259.812 E.03229
G1 X119.414 Y125.558 E.15272
G1 X119.062 Y125.794 E.01619
G1 X118.643 Y125.953 E.01709
G1 X118.162 Y126.014 E.01853
G1 X117.68 Y125.953 E.01854
G1 X117.143 Y125.726 E.02226
G1 X116.708 Y125.363 E.02164
G1 X116.124 Y124.668 E.03467
G3 X103.815 Y131.236 I-31.739 J-44.659 E.53413
G1 X104.38 Y131.654 E.02683
G1 X104.729 Y132.368 E.03035
G1 X104.789 Y132.943 E.02209
G1 X104.789 Y141.742 E.33592
G1 X104.694 Y142.343 E.02324
G1 X104.637 Y142.443 E.00439
G1 X154.071 Y142.443 E1.88733
G3 X143.802 Y132.701 I31.442 J-43.422 E.542
G3 X136.269 Y120.54 I41.785 J-34.301 E.54776
G3 X132.413 Y120.43 I-1.732 J-6.895 E.14913
G1 X131.483 Y120.052 E.03832
G1 X130.602 Y119.535 E.03902
G1 X129.808 Y118.894 E.03894
G1 X129.117 Y118.145 E.03889
G3 X127.853 Y116.453 I123.602 J-93.662 E.08065
G1 X126.662 Y114.847 E.07636
G3 X122.783 Y119.087 I-42.265 J-34.763 E.21951
G1 X123.326 Y119.735 E.03227
G1 X123.558 Y120.081 E.01588
G1 X123.742 Y120.598 E.02096
G1 X123.778 Y121.103 E.01936
G1 X123.674 Y121.626 E.02034
G1 X123.456 Y122.067 E.01878
G1 X123.336 Y122.205 E.00699
G1 X122.758 Y121.976 F36000
G1 F13446.369
G1 X122.104 Y122.54 E.03297
G1 X119.038 Y125.109 E.15272
G1 X118.698 Y125.318 E.01523
G1 X118.287 Y125.422 E.01619
G3 X117.428 Y125.214 I-.097 J-1.477 E.0343
G1 X117.116 Y124.94 E.01581
G1 X116.225 Y123.877 E.05296
G3 X105.45 Y129.952 I-31.972 J-44.119 E.4733
G1 X104.062 Y130.505 E.05701
G1 X103.69 Y130.652 E.01527
G1 F12216.977
G1 X103.319 Y130.8 E.01527
; LINE_WIDTH: 0.666754
G1 F10885.513
G1 X103.234 Y130.855 E.00419
; LINE_WIDTH: 0.713512
G1 F10559.234
G1 X103.148 Y130.911 E.0045
; LINE_WIDTH: 0.76027
G1 F10237.894
G1 X103.063 Y130.967 E.00481
; LINE_WIDTH: 0.807027
G1 F9921.528
G1 X102.978 Y131.022 E.00512
; LINE_WIDTH: 0.853785
G1 F9610.118
G1 X102.893 Y131.078 E.00543
; LINE_WIDTH: 0.900543
G1 F9091.364
G1 X102.808 Y131.134 E.00574
; LINE_WIDTH: 0.947301
G1 F8625.748
G1 X102.723 Y131.189 E.00605
; LINE_WIDTH: 0.994058
G1 F8205.5
G1 X102.638 Y131.245 E.00636
; LINE_WIDTH: 1.04082
G1 F7824.298
G1 X102.553 Y131.301 E.00667
G1 X102.635 Y131.33 E.00574
; LINE_WIDTH: 0.994058
G1 F8205.5
G1 X102.718 Y131.359 E.00548
; LINE_WIDTH: 0.947301
G1 F8625.748
G1 X102.8 Y131.388 E.00521
; LINE_WIDTH: 0.900543
G1 F9091.364
G1 X102.883 Y131.417 E.00494
; LINE_WIDTH: 0.853785
G1 F9610.118
G1 X102.965 Y131.447 E.00468
; LINE_WIDTH: 0.807027
G1 F10191.654
G1 X103.048 Y131.476 E.00441
; LINE_WIDTH: 0.76027
G1 F10467.292
G1 X103.13 Y131.505 E.00414
; LINE_WIDTH: 0.713512
G1 F10746.577
G1 X103.213 Y131.534 E.00388
; LINE_WIDTH: 0.666754
G1 F11029.55
G1 X103.295 Y131.563 E.00361
; LINE_WIDTH: 0.619996
G1 F12369.541
G1 X103.619 Y131.798 E.01527
G1 F13446.369
G1 X103.917 Y132.013 E.01402
G1 X104.161 Y132.512 E.02123
G1 X104.203 Y132.943 E.01653
G1 X104.203 Y141.742 E.33592
G1 X104.137 Y142.163 E.01626
G1 X103.866 Y142.64 E.02094
G1 X103.312 Y143.004 E.02533
; LINE_WIDTH: 0.586906
G1 F14251.59
G1 X103.07 Y143.057 E.00893
; LINE_WIDTH: 0.591496
G1 F14134.183
G1 X103.557 Y143.043 E.01768
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.339 J37.421 E.0837
G1 X155.749 Y143.029 E1.90895
G1 X155.769 Y142.919 E.00427
G3 X144.254 Y132.327 I30.008 J-44.18 E.5995
G3 X136.598 Y119.835 I41.529 J-34.039 E.56113
G3 X133.36 Y120.072 I-2.09 J-6.312 E.12522
G1 X132.575 Y119.867 E.03097
G1 X131.714 Y119.514 E.03555
G1 X130.909 Y119.036 E.03574
G1 X130.185 Y118.446 E.03565
G1 X129.556 Y117.757 E.03561
G3 X127.875 Y115.499 I225.46 J-169.628 E.10749
G1 X126.683 Y113.893 E.07636
G3 X122 Y119.035 I-44.822 J-36.113 E.2657
G3 X123.025 Y120.327 I-34.044 J28.059 E.06297
G1 X123.179 Y120.772 E.01796
G1 X123.184 Y121.165 E.01502
G1 X123.086 Y121.523 E.01416
G1 X122.845 Y121.901 E.0171
G1 X122.826 Y121.917 E.00098
G1 X122.395 Y121.508 F36000
G1 F13446.369
G1 X122.331 Y121.585 E.00384
G1 X118.662 Y124.661 E.18277
G1 X118.412 Y124.801 E.01094
G1 X118.11 Y124.841 E.01164
G1 X117.886 Y124.792 E.00876
G1 X117.581 Y124.583 E.0141
G3 X116.322 Y123.081 I48.339 J-41.814 E.07484
G3 X99.495 Y131.45 I-31.827 J-42.887 E.72127
G3 X99.614 Y132.655 I-11.381 J1.733 E.04622
G1 X102.698 Y132.084 E.11973
G1 X103.114 Y132.121 E.01597
G1 X103.464 Y132.385 E.01675
G1 X103.597 Y132.672 E.01209
G3 X103.618 Y135.742 I-93.72 J2.155 E.1172
G1 X103.618 Y141.742 E.22907
G1 X103.58 Y141.982 E.00928
G1 X103.425 Y142.254 E.01195
G1 X103.112 Y142.471 E.01454
G1 X102.839 Y142.52 E.0106
G1 X102.698 Y142.507 E.00543
G1 X99.618 Y141.937 E.11958
G3 X99.537 Y143.615 I-11.027 J.31 E.06419
G1 X156.235 Y143.615 E2.16469
G1 X156.415 Y142.65 E.03745
G3 X144.705 Y131.954 I29.423 J-43.968 E.6078
G3 X136.908 Y119.075 I41.19 J-33.736 E.5767
G1 X136.5 Y119.253 E.01698
G1 X135.794 Y119.458 E.02809
G1 X134.957 Y119.581 E.03229
G1 X134.404 Y119.587 E.0211
G1 X133.539 Y119.508 E.0332
G1 X132.725 Y119.301 E.03207
G1 X131.944 Y118.976 E.03227
G1 X131.216 Y118.537 E.03245
G1 X130.562 Y117.998 E.03236
G1 X129.995 Y117.369 E.03233
G3 X129.081 Y116.142 I204.31 J-153.108 E.05841
G1 X126.698 Y112.93 E.15272
G3 X121.182 Y119.001 I-43.519 J-33.997 E.31346
G1 X122.428 Y120.488 E.07407
G1 X122.587 Y120.799 E.01334
G1 X122.604 Y121.089 E.01109
G1 X122.512 Y121.366 E.01112
G1 X122.452 Y121.439 E.00361
M204 S250
G1 X121.976 Y121.162 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.307 Y124.237 E.15156
G1 X118.147 Y124.289 E.00533
G1 X118.058 Y124.244 E.00317
G3 X117.944 Y124.155 I.047 J-.178 E.00469
G1 X117.748 Y123.921 E.00966
G1 X117.552 Y123.687 E.00966
G1 X117.356 Y123.454 E.00966
G1 F3450
G1 X117.16 Y123.22 E.00966
G1 F3300
G1 X116.964 Y122.986 E.00966
G1 F3150
G1 X116.768 Y122.752 E.00966
; LINE_WIDTH: 0.520036
G1 F3600
G1 X116.732 Y122.71 E.00174
; LINE_WIDTH: 0.520276
G1 X116.71 Y122.684 E.00107
; LINE_WIDTH: 0.520516
G1 X116.688 Y122.659 E.00107
; LINE_WIDTH: 0.520746
G1 X116.667 Y122.633 E.00107
; LINE_WIDTH: 0.520986
G1 X116.645 Y122.607 E.00107
; LINE_WIDTH: 0.521226
G1 X116.623 Y122.581 E.00107
; LINE_WIDTH: 0.521466
G1 X116.601 Y122.556 E.00107
; LINE_WIDTH: 0.521526
G1 X116.594 Y122.548 E.00031
; LINE_WIDTH: 0.531006
G1 X116.589 Y122.415 E.00431
; LINE_WIDTH: 0.544336
G1 X116.582 Y122.228 E.00623
G1 X116 Y122.631 E.02352
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.538 J-42.493 E.60555
G1 X98.943 Y131.539 E.01587
G1 X99.06 Y132.519 E.03126
G1 X99.064 Y133.319 E.02531
G1 X102.798 Y132.627 E.12024
G3 X103.02 Y132.714 I.041 J.222 E.00797
G1 X103.065 Y132.943 E.00738
G1 X103.065 Y141.742 E.27856
G1 X103.009 Y141.89 E.00502
G1 X102.839 Y141.967 E.00591
G1 X102.798 Y141.964 E.00131
G1 X99.065 Y141.272 E.1202
G1 X99.065 Y142.625 E.04283
G3 X98.939 Y143.866 I-6.659 J-.049 E.03955
G1 X98.936 Y144.167 E.00954
G1 X156.695 Y144.167 E1.82865
G1 X156.959 Y142.745 E.04581
G1 X157.025 Y142.391 E.0114
G3 X145.127 Y131.597 I28.624 J-43.507 E.51064
G3 X137.192 Y118.316 I40.271 J-33.071 E.49162
G3 X136.346 Y118.722 I-5.229 J-9.809 E.02972
G1 X135.64 Y118.927 E.02329
G3 X133.605 Y118.959 I-1.106 J-5.725 E.06475
G1 X132.865 Y118.766 E.0242
G1 X132.162 Y118.467 E.0242
G1 X131.506 Y118.066 E.02434
G1 X130.918 Y117.575 E.02426
G1 X130.409 Y117.003 E.02424
G3 X129.087 Y115.222 I977.277 J-726.899 E.07021
G1 X126.704 Y112.01 E.12664
G3 X121.199 Y118.227 I-44.202 J-33.595 E.26314
; LINE_WIDTH: 0.520256
G1 X120.637 Y118.755 E.02442
; LINE_WIDTH: 0.544336
G1 X120.276 Y119.132 E.01736
G1 X120.467 Y119.159 E.00641
; LINE_WIDTH: 0.530646
G1 X120.612 Y119.181 E.00474
; LINE_WIDTH: 0.520256
G1 X120.63 Y119.202 E.00088
; LINE_WIDTH: 0.520246
G1 X120.704 Y119.29 E.00366
; LINE_WIDTH: 0.520206
G1 X120.778 Y119.379 E.00366
; LINE_WIDTH: 0.520166
G1 X120.852 Y119.467 E.00366
; LINE_WIDTH: 0.520126
G1 X120.926 Y119.556 E.00366
; LINE_WIDTH: 0.520096
G1 X121 Y119.644 E.00365
; LINE_WIDTH: 0.520056
G1 X121.074 Y119.733 E.00366
; LINE_WIDTH: 0.519996
G1 X121.216 Y119.902 E.00697
G1 X121.342 Y120.053 E.00624
G1 X121.469 Y120.204 E.00624
G1 X121.596 Y120.355 E.00624
G1 X121.722 Y120.506 E.00624
G1 X121.849 Y120.657 E.00624
G1 X121.976 Y120.808 E.00624
G3 X122.027 Y120.919 I-.1 J.114 E.00397
G1 X122.055 Y121.017 E.00325
G1 X122.019 Y121.083 E.00238
; WIPE_START
M204 S10000
G1 X121.257 Y121.73 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X116.582 Y122.228 Z6.52 F36000
G1 Z6.12
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G2 X120.276 Y119.132 I-685.732 J-821.7 E.16025
; WIPE_START
M204 S10000
G1 X119.509 Y119.774 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X113.046 Y123.833 Z6.52 F36000
G1 X100.19 Y131.905 Z6.52
G1 Z6.12
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.75125
G1 F10984.592
G1 X100.503 Y131.828 E.01509
; LINE_WIDTH: 0.790263
G1 F10417.679
G1 X100.816 Y131.75 E.01591
; LINE_WIDTH: 0.829276
G1 F9906.408
G1 X101.13 Y131.672 E.01673
; LINE_WIDTH: 0.833456
G1 F9854.591
G1 X101.16 Y131.664 E.00165
; LINE_WIDTH: 0.879591
G1 F9316.716
G1 X101.474 Y131.582 E.01787
; LINE_WIDTH: 0.925726
G1 F8834.518
G1 X101.788 Y131.501 E.01885
; LINE_WIDTH: 0.971861
G1 F8399.777
G1 X102.102 Y131.419 E.01982
; LINE_WIDTH: 1.018
G1 F8005.815
G1 X102.416 Y131.338 E.0208
; LINE_WIDTH: 1.04082
G1 F7824.298
G1 X102.553 Y131.301 E.0093
; WIPE_START
G1 X102.416 Y131.338 E-.05384
G1 X102.102 Y131.419 E-.12324
G1 X101.788 Y131.501 E-.12324
G1 X101.585 Y131.554 E-.07967
; WIPE_END
G1 E-.02 F1800
G1 X100.769 Y139.142 Z6.52 F36000
G1 X100.37 Y142.852 Z6.52
G1 Z6.12
G1 E.4 F1800
; LINE_WIDTH: 0.974016
G1 F8380.513
G1 X100.604 Y142.873 E.01442
; LINE_WIDTH: 0.930966
G1 F8782.887
G1 X100.853 Y142.896 E.01461
; LINE_WIDTH: 0.885252
G1 F9254.731
G1 X101.102 Y142.919 E.01387
; LINE_WIDTH: 0.839539
G1 F9780.15
G1 X101.351 Y142.942 E.01312
; LINE_WIDTH: 0.793825
G1 F10368.818
G1 X101.6 Y142.965 E.01238
; LINE_WIDTH: 0.748111
G1 F11032.89
G1 X101.849 Y142.988 E.01163
; LINE_WIDTH: 0.702398
G1 F11787.842
G1 X102.098 Y143.011 E.01089
; LINE_WIDTH: 0.656684
G1 F12653.704
G1 X102.347 Y143.033 E.01014
; LINE_WIDTH: 0.61097
G1 F13656.849
G1 X102.596 Y143.056 E.0094
; LINE_WIDTH: 0.565256
G1 F14832.745
G1 X103.07 Y143.057 E.01641
; WIPE_START
G1 X102.596 Y143.056 E-.18019
G1 X102.347 Y143.033 E-.095
G1 X102.098 Y143.011 E-.095
G1 X102.072 Y143.008 E-.00981
; WIPE_END
G1 E-.02 F1800
G1 X105.287 Y139.177 Z6.52 F36000
G1 Z6.12
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.287 Y136.834 E.08944
G3 X107.142 Y135.543 I35.765 J49.401 E.08629
G2 X107.955 Y133.658 I-1.306 J-1.681 E.08179
G2 X106.766 Y130.556 I-5.894 J.48 E.12854
G2 X113.362 Y127.093 I-23.488 J-52.74 E.28462
G1 X114.682 Y128.002 E.06122
G3 X115.495 Y129.887 I-1.306 J1.681 E.08179
G3 X112.512 Y134.6 I-6.089 J-.554 E.22114
G1 X111.143 Y135.543 E.06344
G2 X110.33 Y137.428 I1.306 J1.681 E.08179
G2 X113.041 Y141.945 I5.963 J-.506 E.20828
G1 X117.937 Y141.945 E.1869
G3 X118.855 Y138.371 I5.239 J-.56 E.14397
G3 X120.855 Y136.485 I7.11 J5.54 E.10535
G1 X122.223 Y135.543 E.06344
G2 X123.036 Y133.658 I-1.306 J-1.681 E.08179
G2 X120.052 Y128.945 I-6.089 J.554 E.22114
G1 X118.684 Y128.002 E.06344
G3 X117.877 Y126.488 I1.177 J-1.6 E.06776
G2 X119.597 Y126.047 I.284 J-2.472 E.06934
G1 X123.408 Y122.86 E.18966
G2 X123.458 Y119.118 I-1.682 J-1.894 E.16078
G2 X126.637 Y115.65 I-47.716 J-46.93 E.17967
G1 X128.821 Y118.553 E.1387
G3 X126.224 Y120.462 I-21.475 J-26.498 E.12307
G2 X125.412 Y122.347 I1.306 J1.681 E.08179
G2 X128.395 Y127.06 I6.089 J-.554 E.22114
G1 X129.764 Y128.002 E.06344
G3 X130.577 Y129.887 I-1.306 J1.681 E.08179
G3 X127.593 Y134.6 I-6.089 J-.554 E.22114
G1 X126.224 Y135.543 E.06344
G2 X125.412 Y137.428 I1.306 J1.681 E.08179
G2 X128.123 Y141.945 I5.963 J-.506 E.20828
G1 X133.018 Y141.945 E.1869
G3 X133.936 Y138.371 I5.24 J-.56 E.14397
G3 X135.936 Y136.485 I7.109 J5.539 E.10535
G1 X137.304 Y135.543 E.06344
G2 X138.117 Y133.658 I-1.306 J-1.681 E.08179
G2 X135.133 Y128.945 I-6.089 J.554 E.22114
G1 X133.765 Y128.002 E.06344
G3 X132.952 Y126.117 I1.306 J-1.681 E.08179
G3 X136.076 Y121.308 I6.26 J.647 E.22734
G2 X143.917 Y133.619 I50.255 J-23.354 E.55889
G3 X141.306 Y135.543 I-20.09 J-24.531 E.12389
G2 X140.493 Y137.428 I1.306 J1.681 E.08179
G2 X143.204 Y141.945 I5.963 J-.506 E.20828
G1 X148.099 Y141.945 E.1869
G3 X148.803 Y138.768 I4.988 J-.562 E.12653
G2 X150.555 Y140.322 I33.31 J-35.813 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 6.28
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X149.807 Y139.659 E-.38
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
G1 X123.115 Y122.419
G1 Z6.28
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X122.923 Y122.617 E.01053
G3 X122.602 Y122.886 I-2.989 J-3.24 E.01598
G1 X119.537 Y125.456 E.15272
G1 X119.187 Y125.69 E.01609
G1 X118.771 Y125.849 E.01698
G1 X118.284 Y125.911 E.01874
G3 X117.305 Y125.647 I.041 J-2.101 E.03911
G1 X116.879 Y125.313 E.02067
G1 X116.255 Y124.575 E.0369
G3 X103.831 Y131.231 I-31.859 J-44.548 E.53959
G1 X104.298 Y131.55 E.02159
G1 X104.703 Y132.266 E.03139
G1 X104.791 Y132.843 E.02231
G1 X104.791 Y141.744 E.33983
G1 X104.696 Y142.346 E.02324
G1 X104.641 Y142.443 E.00428
G1 X154.069 Y142.443 E1.88712
G3 X143.803 Y132.701 I31.703 J-43.688 E.54188
G3 X136.317 Y120.647 I42.455 J-34.716 E.54326
G1 X136.271 Y120.545 E.00429
G3 X135.166 Y120.737 I-2.371 J-10.368 E.04284
G3 X133.273 Y120.651 I-.565 J-8.462 E.07252
G1 X132.426 Y120.433 E.03339
M73 P68 R5
G1 X131.482 Y120.052 E.03884
G1 X130.604 Y119.536 E.03887
G1 X129.81 Y118.896 E.03894
G1 X129.117 Y118.145 E.03902
G3 X127.853 Y116.453 I123.78 J-93.797 E.08064
G1 X126.662 Y114.847 E.07636
G3 X122.668 Y119.2 I-42.293 J-34.788 E.22564
G1 X123.203 Y119.838 E.03179
G1 X123.491 Y120.298 E.02071
G1 X123.632 Y120.766 E.01866
G1 X123.645 Y121.326 E.02138
G1 X123.511 Y121.836 E.02011
G1 X123.271 Y122.257 E.0185
G1 X123.177 Y122.354 E.00517
G1 X122.631 Y122.078 F36000
G1 F13446.369
G1 X120.693 Y123.722 E.09702
G1 X119.161 Y125.007 E.07636
G1 X118.823 Y125.214 E.01514
G1 X118.415 Y125.319 E.01609
G1 X118.029 Y125.301 E.01474
G1 X117.599 Y125.141 E.01753
G1 X117.239 Y124.838 E.01798
G1 X116.355 Y123.783 E.05256
G3 X105.312 Y130.011 I-32.053 J-43.928 E.48513
G1 X104.06 Y130.507 E.0514
G1 X103.688 Y130.654 E.01527
G1 F12221.09
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.666572
G1 F10889.395
G1 X103.232 Y130.856 E.00417
; LINE_WIDTH: 0.713147
G1 F10564.896
G1 X103.147 Y130.911 E.00448
; LINE_WIDTH: 0.759723
G1 F10245.288
G1 X103.063 Y130.967 E.00478
; LINE_WIDTH: 0.806298
G1 F9930.589
G1 X102.978 Y131.022 E.00509
; LINE_WIDTH: 0.852874
G1 F9620.815
G1 X102.893 Y131.078 E.0054
; LINE_WIDTH: 0.89945
G1 F9102.854
G1 X102.809 Y131.133 E.0057
; LINE_WIDTH: 0.946025
G1 F8637.816
G1 X102.724 Y131.188 E.00601
; LINE_WIDTH: 0.992601
G1 F8217.983
G1 X102.639 Y131.244 E.00632
; LINE_WIDTH: 1.03918
G1 F7837.068
G1 X102.555 Y131.299 E.00662
G1 X102.637 Y131.328 E.00572
; LINE_WIDTH: 0.992601
G1 F8217.983
G1 X102.719 Y131.357 E.00545
; LINE_WIDTH: 0.946025
G1 F8637.816
G1 X102.802 Y131.386 E.00519
; LINE_WIDTH: 0.89945
G1 F9102.854
G1 X102.884 Y131.415 E.00492
; LINE_WIDTH: 0.852874
G1 F9620.815
G1 X102.966 Y131.444 E.00466
; LINE_WIDTH: 0.806298
G1 F10201.278
G1 X103.049 Y131.473 E.00439
; LINE_WIDTH: 0.759723
G1 F10476.218
G1 X103.131 Y131.502 E.00413
; LINE_WIDTH: 0.713147
G1 F10754.814
G1 X103.213 Y131.531 E.00386
; LINE_WIDTH: 0.666572
G1 F11037.077
G1 X103.296 Y131.56 E.0036
; LINE_WIDTH: 0.619996
G1 F12377.512
G1 X103.628 Y131.783 E.01527
G1 F13362.736
G1 X103.86 Y131.94 E.0107
G1 F13446.369
G1 X104.144 Y132.44 E.02196
G1 X104.206 Y132.94 E.01923
G1 X104.206 Y141.744 E.33614
G1 X104.139 Y142.165 E.01626
G1 X103.869 Y142.642 E.02094
G1 X103.314 Y143.006 E.02532
; LINE_WIDTH: 0.585416
M73 P69 R5
G1 F14290.122
G1 X103.072 Y143.059 E.0089
; LINE_WIDTH: 0.590236
G1 F14166.219
G1 X103.558 Y143.044 E.01762
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.339 J35.779 E.08365
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I29.897 J-44.054 E.5995
G3 X136.596 Y119.828 I41.872 J-34.25 E.56142
G1 X136.064 Y119.995 E.02127
G1 X135.343 Y120.13 E.02799
G1 X134.413 Y120.179 E.03557
G3 X133.284 Y120.054 I.629 J-10.845 E.04338
G1 X132.566 Y119.864 E.02837
G1 X131.713 Y119.514 E.03521
G1 X130.911 Y119.038 E.03559
G1 X130.187 Y118.448 E.03566
G1 X129.556 Y117.757 E.03573
G3 X127.875 Y115.499 I225.085 J-169.351 E.10748
G1 X126.683 Y113.893 E.07636
G3 X121.87 Y119.159 I-44.533 J-35.877 E.27256
G1 X122.755 Y120.214 E.05259
G1 X123.014 Y120.693 E.02077
G1 X123.072 Y121.158 E.01793
G1 X122.97 Y121.612 E.01774
G1 X122.735 Y121.99 E.017
G1 X122.7 Y122.02 E.00175
G1 X122.264 Y121.611 F36000
G1 F13446.369
G1 X122.171 Y121.719 E.00547
G3 X121.85 Y121.988 I-1.667 J-1.663 E.01599
G1 X118.784 Y124.558 E.15272
G1 X118.479 Y124.715 E.01312
G1 X118.139 Y124.726 E.01299
G3 X117.688 Y124.461 I.145 J-.765 E.02036
G1 X116.45 Y122.984 E.07359
G3 X99.493 Y131.451 I-32.013 J-42.899 E.72742
G3 X99.616 Y132.652 I-10.735 J1.702 E.04612
G1 X102.7 Y132.081 E.11973
G1 X103.114 Y132.117 E.01588
G1 X103.433 Y132.341 E.01487
G1 X103.589 Y132.63 E.01256
G1 X103.62 Y132.94 E.01188
G1 X103.62 Y141.744 E.33614
G1 X103.582 Y141.984 E.00928
G1 X103.428 Y142.257 E.01195
G1 X103.114 Y142.473 E.01454
G1 X102.841 Y142.523 E.0106
G1 X102.7 Y142.51 E.00544
G1 X99.62 Y141.939 E.11958
G3 X99.538 Y143.615 I-10.863 J.309 E.06409
G1 X156.235 Y143.615 E2.16465
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.955 I29.69 J-44.271 E.60767
G3 X136.912 Y119.083 I40.816 J-33.51 E.57645
G1 X136.092 Y119.382 E.0333
G3 X135.123 Y119.566 I-2.621 J-11.137 E.03769
G3 X133.434 Y119.488 I-.516 J-7.069 E.06471
G1 X132.719 Y119.299 E.02822
G1 X131.943 Y118.975 E.03208
G1 X131.218 Y118.539 E.03231
G1 X130.564 Y118 E.03238
G1 X129.995 Y117.369 E.03244
G3 X129.081 Y116.142 I204.253 J-153.068 E.0584
G1 X126.698 Y112.93 E.15272
G3 X121.065 Y119.111 I-42.144 J-32.749 E.3196
G1 X122.306 Y120.59 E.07372
G1 X122.471 Y120.928 E.01436
G1 X122.482 Y121.185 E.00979
G1 X122.394 Y121.46 E.01103
G1 X122.323 Y121.543 E.00418
M204 S250
G1 X121.854 Y121.264 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.429 Y124.134 E.14148
G1 X118.27 Y124.186 E.0053
G1 X118.181 Y124.141 E.00319
G3 X118.073 Y124.061 I.04 J-.165 E.00436
G1 X117.908 Y123.864 E.00813
G1 X117.743 Y123.667 E.00813
G1 X117.578 Y123.47 E.00813
G1 F3450
G1 X117.413 Y123.273 E.00813
G1 F3300
G1 X117.248 Y123.076 E.00813
G1 F3150
G1 X117.083 Y122.879 E.00813
G1 F3600
G1 X116.967 Y122.741 E.00573
; LINE_WIDTH: 0.520236
G1 X116.942 Y122.711 E.00123
; LINE_WIDTH: 0.520556
G1 X116.906 Y122.669 E.00175
; LINE_WIDTH: 0.520886
G1 X116.87 Y122.627 E.00175
; LINE_WIDTH: 0.521216
G1 X116.834 Y122.585 E.00175
; LINE_WIDTH: 0.521546
G1 X116.799 Y122.543 E.00175
; LINE_WIDTH: 0.521876
G1 X116.763 Y122.501 E.00175
; LINE_WIDTH: 0.522206
G1 X116.727 Y122.459 E.00175
; LINE_WIDTH: 0.522296
G1 X116.716 Y122.446 E.00053
; LINE_WIDTH: 0.530916
G1 X116.702 Y122.326 E.00389
; LINE_WIDTH: 0.543306
G1 X116.68 Y122.155 E.00574
G1 X116 Y122.631 E.02754
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.54 J-42.497 E.60555
G1 X98.943 Y131.538 E.01585
G1 X99.062 Y132.519 E.03127
G1 X99.066 Y133.316 E.02524
G1 X102.8 Y132.625 E.12024
G1 X102.92 Y132.635 E.00382
G1 X103.058 Y132.784 E.00642
G3 X103.067 Y135.744 I-187.669 J2.043 E.09373
G1 X103.067 Y141.744 E.18996
G1 X103.011 Y141.893 E.00502
G1 X102.841 Y141.97 E.00591
G1 X102.8 Y141.966 E.00131
G1 X99.067 Y141.275 E.1202
G1 X99.067 Y142.625 E.04275
G3 X98.939 Y143.867 I-6.578 J-.05 E.03958
G1 X98.936 Y144.167 E.00951
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I29.092 J-44.026 E.51055
G3 X137.192 Y118.315 I40.856 J-33.422 E.49164
G1 X136.612 Y118.616 E.02069
G1 X135.912 Y118.859 E.02345
G3 X135.102 Y119.014 I-4.226 J-19.926 E.02611
G1 X134.365 Y119.041 E.02337
G1 X133.607 Y118.959 E.02412
G1 X132.865 Y118.766 E.02428
G1 X132.161 Y118.467 E.0242
G1 X131.508 Y118.068 E.02423
G1 X130.92 Y117.577 E.02428
G1 X130.409 Y117.003 E.02433
G3 X129.087 Y115.222 I985.16 J-732.75 E.0702
G1 X126.704 Y112.01 E.12664
G3 X121.199 Y118.227 I-44.206 J-33.599 E.26314
; LINE_WIDTH: 0.522746
G1 X120.516 Y118.873 E.02994
; LINE_WIDTH: 0.543746
G1 X120.213 Y119.194 E.01467
G1 X120.38 Y119.246 E.0058
; LINE_WIDTH: 0.531176
G1 X120.491 Y119.282 E.0038
; LINE_WIDTH: 0.522746
G1 X120.519 Y119.315 E.0014
; LINE_WIDTH: 0.522616
G1 X120.605 Y119.418 E.00425
; LINE_WIDTH: 0.522226
G1 X120.691 Y119.521 E.00425
; LINE_WIDTH: 0.521826
G1 X120.776 Y119.624 E.00425
; LINE_WIDTH: 0.521436
G1 X120.862 Y119.726 E.00424
; LINE_WIDTH: 0.521046
G1 X120.947 Y119.829 E.00424
; LINE_WIDTH: 0.520656
G1 X121.033 Y119.932 E.00424
; LINE_WIDTH: 0.520266
G1 X121.093 Y120.004 E.00299
; LINE_WIDTH: 0.519996
G1 X121.173 Y120.1 E.00395
G1 X121.287 Y120.236 E.00561
G1 X121.401 Y120.371 E.00561
G1 X121.514 Y120.507 E.00561
G1 X121.628 Y120.643 E.00561
G1 X121.742 Y120.779 E.00561
G1 X121.856 Y120.914 E.00561
G3 X121.904 Y121.021 I-.098 J.109 E.0038
G1 X121.933 Y121.118 E.00321
G1 X121.897 Y121.184 E.0024
; WIPE_START
M204 S10000
G1 X121.135 Y121.832 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.201 Y119.202 Z6.68 F36000
G1 Z6.28
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G2 X116.698 Y122.137 I535.16 J642.384 E.15192
; WIPE_START
M204 S10000
G1 X117.465 Y121.494 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X110.927 Y125.434 Z6.68 F36000
G1 X100.19 Y131.904 Z6.68
G1 Z6.28
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.749216
G1 F11015.836
G1 X100.503 Y131.826 E.01503
; LINE_WIDTH: 0.788196
G1 F10446.237
G1 X100.816 Y131.749 E.01585
; LINE_WIDTH: 0.827176
G1 F9932.648
G1 X101.129 Y131.671 E.01667
; LINE_WIDTH: 0.831336
G1 F9880.803
G1 X101.16 Y131.663 E.00164
; LINE_WIDTH: 0.877481
G1 F9340.031
G1 X101.474 Y131.581 E.01783
; LINE_WIDTH: 0.923626
G1 F8855.38
G1 X101.788 Y131.5 E.01881
; LINE_WIDTH: 0.969771
G1 F8418.544
G1 X102.102 Y131.418 E.01978
; LINE_WIDTH: 1.01592
G1 F8022.78
G1 X102.416 Y131.337 E.02076
; LINE_WIDTH: 1.03918
G1 F7837.068
G1 X102.555 Y131.299 E.00944
; WIPE_START
G1 X102.416 Y131.337 E-.05477
G1 X102.102 Y131.418 E-.12326
G1 X101.788 Y131.5 E-.12327
G1 X101.587 Y131.552 E-.0787
; WIPE_END
G1 E-.02 F1800
G1 X100.77 Y139.141 Z6.68 F36000
G1 X100.371 Y142.853 Z6.68
G1 Z6.28
G1 E.4 F1800
; LINE_WIDTH: 0.971796
G1 F8400.359
G1 X100.607 Y142.875 E.01448
; LINE_WIDTH: 0.928456
G1 F8807.543
G1 X100.856 Y142.898 E.01457
; LINE_WIDTH: 0.882739
G1 F9282.151
G1 X101.104 Y142.92 E.01383
; LINE_WIDTH: 0.837021
G1 F9810.823
G1 X101.353 Y142.943 E.01308
; LINE_WIDTH: 0.791304
G1 F10403.354
G1 X101.602 Y142.966 E.01234
; LINE_WIDTH: 0.745586
G1 F11072.057
G1 X101.851 Y142.989 E.01159
; LINE_WIDTH: 0.699869
G1 F11832.632
G1 X102.1 Y143.012 E.01085
; LINE_WIDTH: 0.654151
G1 F12705.405
G1 X102.349 Y143.035 E.0101
; LINE_WIDTH: 0.608434
G1 F13717.183
G1 X102.598 Y143.058 E.00936
; LINE_WIDTH: 0.562716
G1 F14904.048
G1 X103.072 Y143.059 E.01632
; WIPE_START
G1 X102.598 Y143.058 E-.18002
G1 X102.349 Y143.035 E-.095
G1 X102.1 Y143.012 E-.095
G1 X102.074 Y143.009 E-.00998
; WIPE_END
G1 E-.02 F1800
G1 X105.289 Y139.074 Z6.68 F36000
G1 Z6.28
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.289 Y136.732 E.08944
G3 X106.856 Y135.543 I25.17 J31.545 E.0751
G2 X107.792 Y133.658 I-1.497 J-1.918 E.0831
G2 X106.76 Y130.557 I-5.262 J.03 E.12686
G2 X113.267 Y127.154 I-27.553 J-60.613 E.28051
G3 X114.856 Y128.473 I-3.452 J5.772 E.07915
G3 X115.242 Y130.83 I-2.439 J1.61 E.09392
G3 X112.684 Y134.6 I-6.177 J-1.438 E.17792
G1 X111.429 Y135.543 E.05993
G2 X110.493 Y137.428 I1.497 J1.918 E.0831
G2 X112.894 Y141.945 I5.758 J-.163 E.20238
G1 X118.166 Y141.945 E.20128
G3 X118.395 Y139.313 I4.305 J-.951 E.10245
G3 X120.682 Y136.485 I6.06 J2.562 E.14068
G1 X121.937 Y135.543 E.05993
G2 X122.873 Y133.658 I-1.497 J-1.918 E.0831
G2 X120.225 Y128.945 I-5.946 J.241 E.21425
G1 X118.97 Y128.002 E.05993
G3 X118.061 Y126.387 I1.466 J-1.888 E.07267
G2 X119.721 Y125.944 I.134 J-2.828 E.06663
G1 X123.308 Y122.944 E.17856
G2 X123.344 Y119.23 I-1.705 J-1.873 E.15913
G2 X126.637 Y115.65 I-44.612 J-44.34 E.18579
G1 X128.851 Y118.588 E.14046
G3 X126.51 Y120.462 I-13.743 J-14.772 E.11458
G2 X125.575 Y122.347 I1.497 J1.918 E.0831
G2 X128.223 Y127.06 I5.946 J-.241 E.21425
G1 X129.478 Y128.002 E.05993
G3 X130.414 Y129.887 I-1.497 J1.918 E.0831
G3 X127.765 Y134.6 I-5.946 J-.241 E.21425
G1 X126.51 Y135.543 E.05993
G2 X125.575 Y137.428 I1.497 J1.918 E.0831
G2 X127.975 Y141.945 I5.758 J-.163 E.20238
G1 X133.247 Y141.945 E.20128
G3 X133.476 Y139.313 I4.305 J-.951 E.10245
G3 X135.763 Y136.485 I6.06 J2.562 E.14068
G1 X137.018 Y135.543 E.05993
G2 X137.954 Y133.658 I-1.497 J-1.918 E.0831
G2 X135.306 Y128.945 I-5.946 J.241 E.21425
G1 X134.051 Y128.002 E.05993
G3 X133.115 Y126.117 I1.497 J-1.918 E.0831
G3 X135.763 Y121.404 I5.946 J.241 E.21425
G1 X136.028 Y121.205 E.01266
G2 X143.945 Y133.658 I52.384 J-24.559 E.56491
G3 X141.592 Y135.543 I-13.746 J-14.747 E.11523
G2 X140.656 Y137.428 I1.497 J1.918 E.0831
G2 X143.057 Y141.945 I5.758 J-.163 E.20238
G1 X148.328 Y141.945 E.20128
G3 X148.819 Y138.778 I4.467 J-.93 E.12504
G2 X150.571 Y140.332 I36.671 J-39.579 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 6.44
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X149.823 Y139.669 E-.38
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
G1 X122.725 Y122.783
G1 Z6.44
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X119.66 Y125.353 E.15272
G1 X119.312 Y125.586 E.01598
G1 X118.9 Y125.745 E.01686
G1 X118.407 Y125.808 E.01897
G1 X117.863 Y125.731 E.02097
G1 X117.496 Y125.582 E.01513
G1 X117.079 Y125.286 E.01955
G1 X116.386 Y124.483 E.04048
G3 X103.835 Y131.229 I-31.985 J-44.456 E.54558
G1 X104.299 Y131.547 E.02147
G1 X104.675 Y132.176 E.02799
G1 X104.788 Y132.7 E.02045
G3 X104.793 Y133.747 I-14.127 J.596 E.03999
G1 X104.793 Y141.747 E.30543
G1 X104.698 Y142.348 E.02324
G1 X104.644 Y142.443 E.00417
G1 X154.069 Y142.443 E1.887
G3 X143.803 Y132.701 I31.495 J-43.471 E.54192
G3 X136.271 Y120.545 I41.683 J-34.237 E.54759
G1 X135.514 Y120.694 E.02946
G1 X134.433 Y120.764 E.04136
G3 X133.275 Y120.652 I.625 J-12.457 E.04445
G1 X132.429 Y120.434 E.03333
G1 X131.481 Y120.051 E.03904
G1 X130.603 Y119.535 E.0389
G1 X129.809 Y118.895 E.03893
G1 X129.117 Y118.145 E.03895
G3 X127.853 Y116.453 I101.25 J-76.964 E.08065
G1 X126.662 Y114.847 E.07636
G3 X122.554 Y119.312 I-45.57 J-37.796 E.23175
G1 X123.081 Y119.942 E.03135
G1 X123.374 Y120.414 E.02119
G1 X123.506 Y120.854 E.01755
G1 X123.524 Y121.409 E.0212
G1 X123.397 Y121.917 E.02001
G1 X123.163 Y122.341 E.01847
G1 X122.804 Y122.717 E.01985
G1 X122.794 Y122.725 E.00051
G1 X122.365 Y122.308 F36000
G1 F13446.369
G1 X122.086 Y122.554 E.01421
G3 X120.816 Y123.619 I-29.952 J-34.443 E.06328
G1 X119.283 Y124.904 E.07636
G1 X118.948 Y125.111 E.01503
G1 X118.543 Y125.216 E.01598
G1 X118.184 Y125.204 E.01374
G1 X117.77 Y125.064 E.01668
G1 X117.362 Y124.735 E.02003
G1 X116.484 Y123.688 E.05216
G3 X105.169 Y130.071 I-32.138 J-43.751 E.49717
G1 X104.061 Y130.508 E.04549
G1 X103.689 Y130.654 E.01527
G1 F12229.251
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.666385
G1 F10897.099
G1 X103.232 Y130.856 E.00415
; LINE_WIDTH: 0.712774
G1 F10573.482
G1 X103.148 Y130.911 E.00446
; LINE_WIDTH: 0.759163
G1 F10254.699
G1 X103.063 Y130.966 E.00476
; LINE_WIDTH: 0.805552
G1 F9940.796
G1 X102.979 Y131.021 E.00507
; LINE_WIDTH: 0.851941
G1 F9631.798
G1 X102.895 Y131.077 E.00537
; LINE_WIDTH: 0.89833
G1 F9114.654
G1 X102.81 Y131.132 E.00568
; LINE_WIDTH: 0.944718
G1 F8650.214
G1 X102.726 Y131.187 E.00598
; LINE_WIDTH: 0.991107
G1 F8230.809
G1 X102.641 Y131.242 E.00629
; LINE_WIDTH: 1.0375
G1 F7850.193
G1 X102.557 Y131.297 E.00659
G1 X102.639 Y131.326 E.00569
; LINE_WIDTH: 0.991107
G1 F8230.809
G1 X102.721 Y131.355 E.00543
; LINE_WIDTH: 0.944718
G1 F8650.214
G1 X102.803 Y131.384 E.00517
; LINE_WIDTH: 0.89833
G1 F9114.654
G1 X102.886 Y131.413 E.0049
; LINE_WIDTH: 0.851941
G1 F9631.798
G1 X102.968 Y131.442 E.00464
; LINE_WIDTH: 0.805552
G1 F10211.154
G1 X103.05 Y131.471 E.00438
; LINE_WIDTH: 0.759163
G1 F10485.5
G1 X103.132 Y131.499 E.00411
; LINE_WIDTH: 0.712774
G1 F10763.452
G1 X103.214 Y131.528 E.00385
; LINE_WIDTH: 0.666385
G1 F11045.04
G1 X103.296 Y131.557 E.00359
; LINE_WIDTH: 0.619996
G1 F12385.945
G1 X103.628 Y131.78 E.01527
G1 F13374.168
G1 X103.862 Y131.936 E.01073
G1 F13446.369
G1 X104.125 Y132.377 E.01958
G1 X104.208 Y132.844 E.01812
G1 X104.208 Y141.747 E.3399
G1 X104.141 Y142.168 E.01626
G1 X103.871 Y142.645 E.02094
G1 X103.316 Y143.007 E.02531
; LINE_WIDTH: 0.583926
G1 F14328.865
G1 X103.074 Y143.06 E.00887
; LINE_WIDTH: 0.588976
G1 F14198.401
G1 X103.559 Y143.044 E.01757
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.338 J34.27 E.08359
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.059 J-44.23 E.59948
G3 X136.596 Y119.828 I41.532 J-34.042 E.56144
G1 X136.064 Y119.995 E.02127
G1 X135.345 Y120.129 E.02794
G1 X134.411 Y120.179 E.0357
G3 X133.287 Y120.054 I.624 J-10.789 E.04319
G1 X132.568 Y119.865 E.0284
G1 X131.712 Y119.513 E.03535
G1 X130.91 Y119.036 E.03561
G1 X130.186 Y118.447 E.03564
G1 X129.556 Y117.757 E.03566
G3 X127.875 Y115.499 I182.476 J-137.624 E.10748
G1 X126.683 Y113.893 E.07636
G3 X121.753 Y119.268 I-44.387 J-35.764 E.27867
G1 X122.632 Y120.318 E.05228
G1 X122.847 Y120.673 E.01586
G1 X122.93 Y120.956 E.01124
G1 X122.942 Y121.344 E.01483
G1 X122.853 Y121.7 E.014
G1 X122.623 Y122.08 E.01698
G1 X122.433 Y122.248 E.0097
G1 X121.979 Y121.864 F36000
G1 F13446.369
G1 X121.71 Y122.106 E.01379
G3 X120.44 Y123.17 I-22.729 J-25.826 E.06328
G1 X118.907 Y124.455 E.07636
G1 X118.604 Y124.612 E.01303
G1 X118.28 Y124.626 E.01239
G1 X118.043 Y124.547 E.00952
G1 X117.81 Y124.359 E.01143
G1 X116.578 Y122.888 E.07327
G3 X99.493 Y131.454 I-32.119 J-42.736 E.73363
G3 X99.618 Y132.65 I-10.214 J1.674 E.04591
G1 X102.702 Y132.078 E.11974
G1 X103.115 Y132.114 E.01585
G1 X103.434 Y132.337 E.01484
G1 X103.58 Y132.591 E.01118
G1 X103.622 Y132.844 E.0098
G1 X103.622 Y141.747 E.3399
G1 X103.584 Y141.987 E.00928
G1 X103.43 Y142.259 E.01195
G1 X103.117 Y142.476 E.01454
G1 X102.843 Y142.525 E.0106
G1 X102.702 Y142.512 E.00544
G1 X99.622 Y141.942 E.11958
G3 X99.539 Y143.615 I-10.707 J.307 E.064
G1 X156.235 Y143.615 E2.16462
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.955 I29.104 J-43.63 E.60775
G3 X136.912 Y119.083 I41.093 J-33.678 E.57642
G1 X136.092 Y119.382 E.03329
G1 X135.299 Y119.542 E.03089
G1 X134.389 Y119.593 E.03479
G3 X133.436 Y119.488 I1.015 J-13.538 E.03662
G1 X132.721 Y119.3 E.02823
G1 X131.942 Y118.975 E.03223
G1 X131.217 Y118.538 E.03233
G1 X130.563 Y117.999 E.03236
G1 X129.995 Y117.369 E.03238
G3 X129.081 Y116.142 I163.317 J-122.581 E.05841
G1 X126.698 Y112.93 E.15272
G3 X120.948 Y119.22 I-42.396 J-32.976 E.32571
G1 X122.183 Y120.694 E.0734
G1 X122.347 Y121.025 E.01411
G1 X122.344 Y121.373 E.01329
G1 X122.216 Y121.651 E.01169
G1 X122.046 Y121.804 E.00873
M204 S250
G1 X121.6 Y121.477 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.552 Y124.031 E.12591
G1 X118.394 Y124.084 E.00526
G1 X118.304 Y124.038 E.0032
G3 X118.2 Y123.962 I.036 J-.158 E.00419
G1 X118.05 Y123.784 E.00737
G1 X117.901 Y123.606 E.00737
G1 X117.751 Y123.428 E.00737
G1 F3450
G1 X117.602 Y123.249 E.00737
G1 F3300
G1 X117.452 Y123.071 E.00737
G1 F3150
G1 X117.303 Y122.893 E.00737
G1 F3600
G1 X117.198 Y122.767 E.00519
; LINE_WIDTH: 0.520236
G1 X117.162 Y122.724 E.00177
; LINE_WIDTH: 0.520576
G1 X117.11 Y122.664 E.00251
; LINE_WIDTH: 0.520916
G1 X117.059 Y122.603 E.00251
; LINE_WIDTH: 0.521256
G1 X117.008 Y122.543 E.00251
; LINE_WIDTH: 0.521596
G1 X116.957 Y122.482 E.00251
; LINE_WIDTH: 0.521936
G1 X116.906 Y122.422 E.00252
; LINE_WIDTH: 0.522276
G1 X116.855 Y122.362 E.00252
; LINE_WIDTH: 0.522376
G1 X116.839 Y122.343 E.00077
; LINE_WIDTH: 0.528406
G1 X116.811 Y122.277 E.00231
; LINE_WIDTH: 0.537106
G1 X116.77 Y122.182 E.0034
; LINE_WIDTH: 0.544336
G1 X116.821 Y122.034 E.0052
G1 X116 Y122.631 E.03372
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.541 J-42.498 E.60555
G1 X98.943 Y131.538 E.01583
G1 X99.064 Y132.519 E.03129
G1 X99.068 Y133.314 E.02517
G1 X102.802 Y132.622 E.12024
G1 X102.922 Y132.632 E.00381
G1 X103.057 Y132.771 E.0061
G1 X103.069 Y132.844 E.00236
G1 X103.069 Y141.747 E.28187
G1 X103.013 Y141.895 E.00502
G1 X102.843 Y141.973 E.00591
G1 X102.802 Y141.969 E.00131
G1 X99.069 Y141.277 E.1202
G1 X99.069 Y142.625 E.04267
G3 X98.939 Y143.868 I-6.507 J-.052 E.03962
G1 X98.936 Y144.167 E.00948
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.614 J-43.499 E.5106
G3 X137.192 Y118.315 I40.566 J-33.248 E.49167
G1 X136.612 Y118.616 E.02068
M73 P70 R5
G1 X135.913 Y118.859 E.02344
G1 X135.198 Y118.999 E.02305
G1 X134.369 Y119.041 E.02628
G1 X133.61 Y118.959 E.02418
G1 X132.868 Y118.767 E.02427
G1 X132.16 Y118.466 E.02434
G1 X131.507 Y118.067 E.02424
G1 X130.919 Y117.576 E.02427
G1 X130.409 Y117.003 E.02428
G3 X129.087 Y115.222 I766.185 J-570.192 E.07021
G1 X126.704 Y112.01 E.12664
G3 X121.199 Y118.227 I-44.713 J-34.048 E.26314
; LINE_WIDTH: 0.520586
G1 X120.391 Y118.987 E.03517
; LINE_WIDTH: 0.544336
G1 X120.032 Y119.339 E.01671
G1 X120.235 Y119.294 E.00692
; LINE_WIDTH: 0.534566
G1 X120.31 Y119.347 E.00298
; LINE_WIDTH: 0.526596
G1 X120.367 Y119.386 E.00221
; LINE_WIDTH: 0.520586
G1 X120.392 Y119.416 E.00124
; LINE_WIDTH: 0.520556
G1 X120.494 Y119.538 E.00506
; LINE_WIDTH: 0.520476
G1 X120.597 Y119.661 E.00506
; LINE_WIDTH: 0.520386
G1 X120.699 Y119.783 E.00506
; LINE_WIDTH: 0.520306
G1 X120.802 Y119.906 E.00506
; LINE_WIDTH: 0.520216
G1 X120.904 Y120.028 E.00506
; LINE_WIDTH: 0.520136
G1 X121.007 Y120.15 E.00506
; LINE_WIDTH: 0.519996
G1 X121.148 Y120.319 E.00697
G1 X121.246 Y120.436 E.00483
G1 X121.344 Y120.553 E.00483
G1 X121.442 Y120.67 E.00483
G1 X121.54 Y120.787 E.00483
G1 X121.638 Y120.904 E.00483
G1 X121.736 Y121.021 E.00483
G3 X121.782 Y121.122 I-.096 J.104 E.00361
G1 X121.811 Y121.218 E.00319
G3 X121.673 Y121.424 I-.4 J-.119 E.00796
; WIPE_START
M204 S10000
G1 X120.905 Y122.065 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.032 Y119.339 Z6.84 F36000
G1 Z6.44
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X116.821 Y122.034 E.13938
; WIPE_START
M204 S10000
G1 X117.587 Y121.391 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X111.054 Y125.338 Z6.84 F36000
G1 X100.19 Y131.903 Z6.84
G1 Z6.44
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.747156
G1 F11047.67
G1 X100.503 Y131.825 E.01498
; LINE_WIDTH: 0.786096
G1 F10475.418
G1 X100.816 Y131.748 E.01579
; LINE_WIDTH: 0.825036
G1 F9959.53
G1 X101.129 Y131.67 E.01661
; LINE_WIDTH: 0.829196
G1 F9907.405
G1 X101.159 Y131.662 E.00164
; LINE_WIDTH: 0.875346
G1 F9363.742
G1 X101.473 Y131.581 E.01779
; LINE_WIDTH: 0.921496
G1 F8876.64
G1 X101.787 Y131.499 E.01876
; LINE_WIDTH: 0.967646
G1 F8437.712
G1 X102.101 Y131.417 E.01974
; LINE_WIDTH: 1.0138
G1 F8040.144
G1 X102.415 Y131.336 E.02071
; LINE_WIDTH: 1.0375
G1 F7850.193
G1 X102.557 Y131.297 E.00961
; WIPE_START
G1 X102.415 Y131.336 E-.05583
G1 X102.101 Y131.417 E-.12328
G1 X101.787 Y131.499 E-.12328
G1 X101.59 Y131.55 E-.0776
; WIPE_END
G1 E-.02 F1800
G1 X100.772 Y139.139 Z6.84 F36000
G1 X100.371 Y142.854 Z6.84
G1 Z6.44
G1 E.4 F1800
; LINE_WIDTH: 0.969536
G1 F8420.659
G1 X100.609 Y142.876 E.01454
; LINE_WIDTH: 0.925916
G1 F8832.635
G1 X100.858 Y142.899 E.01453
; LINE_WIDTH: 0.880201
G1 F9309.998
G1 X101.107 Y142.922 E.01379
; LINE_WIDTH: 0.834486
G1 F9841.905
G1 X101.356 Y142.944 E.01304
; LINE_WIDTH: 0.788771
G1 F10438.276
G1 X101.605 Y142.967 E.0123
; LINE_WIDTH: 0.743056
G1 F11111.582
G1 X101.854 Y142.99 E.01155
; LINE_WIDTH: 0.697341
G1 F11877.739
G1 X102.103 Y143.013 E.01081
; LINE_WIDTH: 0.651626
G1 F12757.377
G1 X102.352 Y143.036 E.01006
; LINE_WIDTH: 0.605911
G1 F13777.72
G1 X102.601 Y143.059 E.00932
; LINE_WIDTH: 0.560196
G1 F14975.469
G1 X103.074 Y143.06 E.01623
; WIPE_START
G1 X102.601 Y143.059 E-.17985
G1 X102.352 Y143.036 E-.095
G1 X102.103 Y143.013 E-.095
G1 X102.076 Y143.011 E-.01015
; WIPE_END
G1 E-.02 F1800
G1 X105.29 Y138.96 Z6.84 F36000
G1 Z6.44
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.289 Y136.618 E.08944
G2 X107.069 Y135.072 I-14.697 J-18.715 E.09005
G2 X107.61 Y132.715 I-2.383 J-1.787 E.09505
G2 X106.782 Y130.556 I-5.774 J.975 E.08888
G2 X113.173 Y127.212 I-29.195 J-63.593 E.2755
G3 X114.92 Y128.945 I-3.084 J4.856 E.09466
G3 X114.929 Y131.772 I-3.834 J1.426 E.11024
G3 X112.842 Y134.6 I-5.95 J-2.207 E.13595
G2 X111.216 Y136.014 I4.999 J7.392 E.08248
G2 X110.675 Y138.371 I2.384 J1.787 E.09505
G2 X112.757 Y141.945 I5.919 J-1.053 E.16124
G1 X118.372 Y141.945 E.21437
G3 X118.438 Y139.313 I3.96 J-1.218 E.1023
G3 X120.524 Y136.485 I5.95 J2.207 E.13595
G2 X122.151 Y135.072 I-4.999 J-7.392 E.08248
G2 X122.691 Y132.715 I-2.383 J-1.787 E.09505
G2 X120.383 Y128.945 I-6.023 J1.096 E.17269
G3 X118.756 Y127.531 I4.998 J-7.391 E.08248
G3 X118.216 Y126.297 I2.107 J-1.659 E.05201
G2 X119.844 Y125.84 I.191 J-2.454 E.06591
G1 X123.208 Y123.027 E.16742
G2 X123.23 Y119.343 I-1.772 J-1.852 E.15691
G2 X126.637 Y115.65 I-45.4 J-45.304 E.19192
G1 X128.883 Y118.625 E.14232
G3 X126.755 Y120.462 I-12.591 J-12.437 E.10745
G2 X125.756 Y123.289 I1.895 J2.259 E.12011
G2 X128.065 Y127.06 I6.023 J-1.096 E.17269
G3 X129.691 Y128.473 I-4.998 J7.391 E.08248
G3 X130.232 Y130.83 I-2.383 J1.787 E.09505
G3 X127.923 Y134.6 I-6.023 J-1.096 E.17269
G2 X126.297 Y136.014 I4.999 J7.392 E.08248
G2 X125.756 Y138.371 I2.383 J1.787 E.09505
G2 X127.838 Y141.945 I5.919 J-1.053 E.16124
G1 X133.453 Y141.945 E.21437
G3 X133.519 Y139.313 I3.96 J-1.218 E.1023
G3 X135.606 Y136.485 I5.95 J2.207 E.13595
G2 X137.232 Y135.072 I-4.999 J-7.392 E.08248
G2 X137.772 Y132.715 I-2.384 J-1.787 E.09505
G2 X135.464 Y128.945 I-6.023 J1.096 E.17269
G3 X133.838 Y127.531 I4.999 J-7.392 E.08248
G3 X133.297 Y125.174 I2.384 J-1.787 E.09505
G3 X135.606 Y121.404 I6.023 J1.096 E.17269
G1 X135.985 Y121.112 E.01828
G2 X143.978 Y133.692 I50.108 J-23.007 E.57079
G3 X141.836 Y135.543 I-12.593 J-12.409 E.10817
G2 X140.838 Y138.371 I1.895 J2.259 E.12011
G2 X142.919 Y141.945 I5.918 J-1.053 E.16124
G1 X148.534 Y141.945 E.21437
G3 X148.829 Y138.79 I4.142 J-1.205 E.12389
G2 X150.584 Y140.341 I39.164 J-42.535 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 6.6
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X149.834 Y139.679 E-.38
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
G1 X122.997 Y122.477
G1 Z6.6
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X122.909 Y122.605 E.00595
G3 X121.315 Y123.965 I-23.023 J-25.37 E.08
G1 X119.782 Y125.25 E.07636
G1 X119.437 Y125.482 E.01587
G1 X119.028 Y125.641 E.01675
G1 X118.53 Y125.705 E.01919
G1 X118.005 Y125.634 E.02021
G1 X117.685 Y125.513 E.01308
G1 X117.277 Y125.25 E.0185
G1 X116.558 Y124.438 E.04143
G1 X116.496 Y124.405 E.00268
G3 X103.825 Y131.232 I-32.026 J-44.265 E.55109
G1 X104.377 Y131.634 E.02605
G1 X104.735 Y132.358 E.03084
G1 X104.795 Y132.934 E.02208
G1 X104.795 Y141.749 E.33658
G1 X104.7 Y142.351 E.02324
G1 X104.648 Y142.443 E.00406
G1 X154.069 Y142.443 E1.88685
G3 X143.803 Y132.701 I31.506 J-43.481 E.54191
G3 X136.271 Y120.545 I41.764 J-34.289 E.54759
G1 X135.451 Y120.705 E.03188
G3 X133.274 Y120.651 I-.889 J-8.103 E.08339
G1 X132.426 Y120.433 E.03342
G1 X131.481 Y120.051 E.03893
G1 X130.603 Y119.535 E.03889
G1 X129.808 Y118.894 E.03898
G1 X129.117 Y118.145 E.03892
G3 X127.853 Y116.453 I124.228 J-94.129 E.08063
G1 X126.662 Y114.847 E.07636
G3 X122.44 Y119.425 I-42.913 J-35.341 E.23789
G1 X122.958 Y120.044 E.03082
G1 X123.305 Y120.654 E.0268
G1 X123.412 Y121.205 E.02144
G1 X123.366 Y121.725 E.01993
G1 X123.202 Y122.179 E.0184
G1 X123.048 Y122.403 E.01038
G1 X122.405 Y122.266 F36000
G1 F13446.369
G1 X122.086 Y122.554 E.01642
G3 X120.939 Y123.516 I-18.234 J-20.588 E.05718
G1 X119.406 Y124.801 E.07636
G1 X119.073 Y125.007 E.01493
G1 X118.671 Y125.112 E.01587
G1 X118.337 Y125.106 E.01279
G1 X117.938 Y124.985 E.01589
G1 X117.654 Y124.801 E.01294
G3 X116.613 Y123.593 I27.71 J-24.909 E.06088
G3 X104.382 Y130.391 I-32.209 J-43.551 E.53575
G1 X104.065 Y130.513 E.01296
G1 X103.692 Y130.656 E.01527
G1 F12240.454
G1 X103.318 Y130.8 E.01527
; LINE_WIDTH: 0.666203
G1 F10907.674
G1 X103.234 Y130.855 E.00415
; LINE_WIDTH: 0.71241
G1 F10584.082
G1 X103.15 Y130.91 E.00445
; LINE_WIDTH: 0.758616
G1 F10265.363
G1 X103.065 Y130.965 E.00476
; LINE_WIDTH: 0.804823
G1 F9951.516
G1 X102.981 Y131.02 E.00506
; LINE_WIDTH: 0.85103
G1 F9642.542
G1 X102.897 Y131.075 E.00536
; LINE_WIDTH: 0.897236
G1 F9126.204
G1 X102.812 Y131.13 E.00567
; LINE_WIDTH: 0.943443
G1 F8662.351
G1 X102.728 Y131.185 E.00597
; LINE_WIDTH: 0.98965
G1 F8243.369
G1 X102.644 Y131.241 E.00627
; LINE_WIDTH: 1.03586
G1 F7863.049
G1 X102.559 Y131.296 E.00658
G1 X102.641 Y131.324 E.00567
; LINE_WIDTH: 0.98965
G1 F8243.369
G1 X102.723 Y131.353 E.00541
; LINE_WIDTH: 0.943443
G1 F8662.351
G1 X102.805 Y131.382 E.00515
; LINE_WIDTH: 0.897236
G1 F9126.204
G1 X102.887 Y131.411 E.00488
; LINE_WIDTH: 0.85103
G1 F9642.542
G1 X102.969 Y131.439 E.00462
; LINE_WIDTH: 0.804823
G1 F10220.814
G1 X103.051 Y131.468 E.00436
; LINE_WIDTH: 0.758616
G1 F10494.562
G1 X103.133 Y131.497 E.0041
; LINE_WIDTH: 0.71241
G1 F10771.928
G1 X103.215 Y131.525 E.00384
; LINE_WIDTH: 0.666203
G1 F11052.911
G1 X103.297 Y131.554 E.00358
; LINE_WIDTH: 0.619996
G1 F12394.279
G1 X103.622 Y131.787 E.01527
G1 F13446.369
G1 X103.917 Y131.997 E.01382
G1 X104.167 Y132.504 E.02158
G1 X104.21 Y132.934 E.0165
G1 X104.21 Y141.749 E.33658
G1 X104.143 Y142.17 E.01626
G1 X103.873 Y142.647 E.02094
G1 X103.317 Y143.009 E.02531
; LINE_WIDTH: 0.582436
G1 F14349.578
G1 X103.076 Y143.061 E.00884
; LINE_WIDTH: 0.587716
G1 F14230.729
G1 X103.561 Y143.045 E.01751
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.337 J32.859 E.08354
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00187
G3 X144.254 Y132.328 I30.059 J-44.231 E.59949
G3 X136.596 Y119.828 I41.75 J-34.176 E.56141
G1 X136.066 Y119.995 E.0212
G1 X135.344 Y120.129 E.02805
G1 X134.411 Y120.179 E.03567
G3 X133.286 Y120.054 I.625 J-10.789 E.04321
G1 X132.566 Y119.864 E.02846
G1 X131.712 Y119.513 E.03525
G1 X130.91 Y119.037 E.0356
G1 X130.185 Y118.446 E.03569
G1 X129.556 Y117.757 E.03564
G3 X127.875 Y115.499 I225.899 J-169.953 E.10747
G1 X126.683 Y113.893 E.07636
G3 X121.636 Y119.378 I-42.235 J-33.798 E.28481
G1 X122.509 Y120.42 E.0519
G1 X122.761 Y120.874 E.01983
G1 X122.827 Y121.336 E.01783
G1 X122.736 Y121.789 E.01761
G1 X122.511 Y122.17 E.0169
G1 X122.472 Y122.205 E.00202
G1 X121.991 Y121.846 F36000
G1 F13446.369
G1 X121.71 Y122.105 E.0146
G1 X119.03 Y124.352 E.13353
G1 X118.729 Y124.508 E.01295
G1 X118.419 Y124.526 E.01183
G1 X118.192 Y124.457 E.00907
G1 X117.933 Y124.256 E.01253
G1 X116.705 Y122.791 E.07296
G3 X99.491 Y131.453 I-32.213 J-42.583 E.7398
G3 X99.62 Y132.647 I-9.629 J1.646 E.04589
G1 X102.704 Y132.076 E.11974
G1 X103.118 Y132.112 E.01586
G1 X103.467 Y132.372 E.01664
G1 X103.604 Y132.664 E.01229
G3 X103.624 Y135.749 I-93.289 J2.162 E.1178
G1 X103.624 Y141.749 E.22907
G1 X103.586 Y141.989 E.00928
G1 X103.432 Y142.262 E.01195
G1 X103.119 Y142.478 E.01454
G1 X102.846 Y142.528 E.0106
G1 X102.704 Y142.515 E.00544
G1 X99.624 Y141.944 E.11958
G3 X99.54 Y143.615 I-10.548 J.305 E.06391
G1 X156.235 Y143.615 E2.16459
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.705 Y131.954 I29.104 J-43.63 E.60776
G3 X136.912 Y119.083 I40.808 J-33.505 E.57644
G1 X136.093 Y119.382 E.03328
G3 X134.379 Y119.594 I-1.77 J-7.279 E.06606
G3 X133.436 Y119.488 I1.139 J-14.414 E.03626
G1 X132.719 Y119.299 E.02829
G1 X131.942 Y118.975 E.03213
G1 X131.217 Y118.538 E.03232
G1 X130.562 Y117.998 E.03241
G1 X129.994 Y117.369 E.03235
G3 X129.081 Y116.142 I205.943 J-154.325 E.05839
G1 X126.698 Y112.93 E.15272
G3 X120.832 Y119.33 I-42.395 J-32.969 E.33183
G1 X122.06 Y120.796 E.07303
G1 X122.223 Y121.123 E.01395
G1 X122.223 Y121.468 E.01314
G1 X122.099 Y121.747 E.01166
G1 X122.057 Y121.785 E.00216
M204 S250
G1 X121.609 Y121.469 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.675 Y123.929 E.12122
G1 X118.518 Y123.981 E.00523
G1 X118.427 Y123.936 E.00322
G3 X118.326 Y123.864 I.032 J-.152 E.00403
G1 X118.192 Y123.704 E.0066
G1 X118.058 Y123.544 E.0066
G1 X117.924 Y123.385 E.0066
G1 F3450
G1 X117.79 Y123.225 E.0066
G1 F3300
G1 X117.657 Y123.065 E.0066
G1 F3150
G1 X117.523 Y122.906 E.0066
G1 F3600
G1 X117.428 Y122.793 E.00465
; LINE_WIDTH: 0.520276
G1 X117.381 Y122.738 E.0023
; LINE_WIDTH: 0.520676
G1 X117.315 Y122.659 E.00326
; LINE_WIDTH: 0.521066
G1 X117.248 Y122.58 E.00327
; LINE_WIDTH: 0.521466
G1 X117.182 Y122.502 E.00327
; LINE_WIDTH: 0.521856
G1 X117.115 Y122.423 E.00327
; LINE_WIDTH: 0.522246
G1 X117.049 Y122.344 E.00328
; LINE_WIDTH: 0.522646
G1 X116.983 Y122.266 E.00328
; LINE_WIDTH: 0.522766
G1 X116.961 Y122.241 E.00105
; LINE_WIDTH: 0.525586
G1 X116.945 Y122.22 E.00086
; LINE_WIDTH: 0.529716
G1 X116.92 Y122.189 E.00128
; LINE_WIDTH: 0.544336
G1 X116.943 Y121.931 E.0086
G1 X116 Y122.631 E.03903
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-31.539 J-42.494 E.60556
G1 X98.943 Y131.537 E.01582
G1 X99.066 Y132.519 E.0313
G1 X99.07 Y133.311 E.0251
G1 X102.804 Y132.62 E.12024
G3 X103.071 Y132.841 I.023 J.243 E.01222
G1 X103.071 Y141.749 E.28202
G1 X103.015 Y141.898 E.00502
G1 X102.846 Y141.975 E.00591
G1 X102.804 Y141.971 E.00131
G1 X99.071 Y141.28 E.1202
G1 X99.071 Y142.625 E.04259
G3 X98.939 Y143.868 I-6.435 J-.054 E.03964
G1 X98.936 Y144.167 E.00947
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.597 I28.613 J-43.497 E.51061
G3 X137.192 Y118.315 I40.31 J-33.095 E.49168
G1 X136.613 Y118.616 E.02066
G1 X135.913 Y118.859 E.02347
G1 X135.135 Y119.01 E.02507
G1 X134.369 Y119.041 E.02428
G1 X133.609 Y118.959 E.0242
G1 X132.865 Y118.766 E.02432
G1 X132.16 Y118.467 E.02426
G1 X131.507 Y118.067 E.02424
G1 X130.918 Y117.575 E.0243
G1 X130.409 Y117.003 E.02426
G3 X129.087 Y115.222 I975.438 J-725.537 E.0702
G1 X126.704 Y112.01 E.12664
G3 X121.199 Y118.227 I-44.207 J-33.599 E.26314
; LINE_WIDTH: 0.521086
G1 X120.267 Y119.103 E.04058
; LINE_WIDTH: 0.544336
G1 X119.91 Y119.442 E.01639
G1 X120.211 Y119.449 E.01003
; LINE_WIDTH: 0.526816
G1 X120.23 Y119.472 E.00096
; LINE_WIDTH: 0.523516
G1 X120.245 Y119.488 E.0007
; LINE_WIDTH: 0.521086
G1 X120.275 Y119.525 E.00152
; LINE_WIDTH: 0.521036
G1 X120.393 Y119.666 E.00582
; LINE_WIDTH: 0.520886
G1 X120.511 Y119.807 E.00582
; LINE_WIDTH: 0.520726
G1 X120.629 Y119.948 E.00582
; LINE_WIDTH: 0.520566
G1 X120.747 Y120.088 E.00582
; LINE_WIDTH: 0.520416
G1 X120.864 Y120.229 E.00582
; LINE_WIDTH: 0.520256
G1 X120.982 Y120.37 E.00582
; LINE_WIDTH: 0.520106
G1 X121.065 Y120.469 E.0041
; LINE_WIDTH: 0.519996
G1 X121.123 Y120.539 E.00286
G1 X121.206 Y120.637 E.00406
G1 X121.288 Y120.735 E.00406
G1 X121.371 Y120.834 E.00406
G1 X121.453 Y120.932 E.00406
G1 X121.535 Y121.03 E.00406
G1 X121.618 Y121.129 E.00406
G3 X121.659 Y121.224 I-.093 J.097 E.00339
G1 X121.688 Y121.319 E.00314
G1 X121.651 Y121.39 E.00253
; WIPE_START
M204 S10000
G1 X120.89 Y122.039 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X119.91 Y119.442 Z7 F36000
G1 Z6.6
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X116.943 Y121.931 E.12874
; WIPE_START
M204 S10000
G1 X117.709 Y121.288 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X111.181 Y125.243 Z7 F36000
G1 X100.191 Y131.902 Z7
G1 Z6.6
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.745116
G1 F11079.379
G1 X100.503 Y131.824 E.01492
; LINE_WIDTH: 0.784036
G1 F10504.202
G1 X100.816 Y131.747 E.01574
; LINE_WIDTH: 0.822956
G1 F9985.799
G1 X101.128 Y131.669 E.01656
; LINE_WIDTH: 0.827116
G1 F9933.399
G1 X101.159 Y131.661 E.00163
; LINE_WIDTH: 0.873261
G1 F9387.014
G1 X101.473 Y131.58 E.01774
; LINE_WIDTH: 0.919406
G1 F8897.602
G1 X101.787 Y131.498 E.01872
; LINE_WIDTH: 0.965551
G1 F8456.694
G1 X102.101 Y131.416 E.01969
; LINE_WIDTH: 1.0117
G1 F8057.421
G1 X102.415 Y131.335 E.02067
; LINE_WIDTH: 1.03586
G1 F7863.049
G1 X102.559 Y131.296 E.00977
; WIPE_START
G1 X102.415 Y131.335 E-.05688
G1 X102.101 Y131.416 E-.12327
G1 X101.787 Y131.498 E-.12327
G1 X101.592 Y131.549 E-.07657
; WIPE_END
G1 E-.02 F1800
G1 X100.773 Y139.137 Z7 F36000
G1 X100.372 Y142.855 Z7
G1 Z6.6
G1 E.4 F1800
; LINE_WIDTH: 0.967296
G1 F8440.877
G1 X100.611 Y142.877 E.0146
; LINE_WIDTH: 0.923386
G1 F8857.77
G1 X100.86 Y142.9 E.01449
; LINE_WIDTH: 0.87767
G1 F9337.941
G1 X101.109 Y142.923 E.01374
; LINE_WIDTH: 0.831954
G1 F9873.154
G1 X101.358 Y142.946 E.013
; LINE_WIDTH: 0.786237
G1 F10473.451
G1 X101.607 Y142.969 E.01225
; LINE_WIDTH: 0.740521
G1 F11151.47
G1 X101.856 Y142.991 E.01151
; LINE_WIDTH: 0.694805
G1 F11923.35
G1 X102.105 Y143.014 E.01076
; LINE_WIDTH: 0.649089
G1 F12810.035
G1 X102.354 Y143.037 E.01002
; LINE_WIDTH: 0.603373
G1 F13839.189
G1 X102.603 Y143.06 E.00927
; LINE_WIDTH: 0.557656
G1 F15048.156
G1 X103.076 Y143.061 E.01613
; WIPE_START
G1 X102.603 Y143.06 E-.17968
G1 X102.354 Y143.037 E-.095
G1 X102.105 Y143.014 E-.095
G1 X102.078 Y143.012 E-.01032
; WIPE_END
G1 E-.02 F1800
G1 X105.293 Y138.83 Z7 F36000
G1 Z6.6
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.293 Y136.487 E.08944
G2 X106.85 Y135.072 I-15.236 J-18.32 E.08038
G2 X107.526 Y132.715 I-2.4 J-1.963 E.09622
G2 X106.801 Y130.541 I-5.584 J.655 E.08813
G2 X113.08 Y127.269 I-23.266 J-52.321 E.27051
G3 X114.72 Y128.945 I-3.961 J5.518 E.08997
G3 X114.892 Y131.772 I-3.622 J1.639 E.11059
G3 X112.99 Y134.6 I-5.955 J-1.953 E.13179
G2 X111.105 Y136.485 I4.318 J6.202 E.10231
G2 X110.933 Y139.313 I3.622 J1.639 E.11059
G2 X112.643 Y141.945 I5.704 J-1.834 E.12126
G1 X118.559 Y141.945 E.22586
G3 X118.474 Y139.313 I3.738 J-1.439 E.10246
G3 X120.377 Y136.485 I5.955 J1.953 E.13179
G2 X122.261 Y134.6 I-4.319 J-6.202 E.10231
G2 X122.433 Y131.772 I-3.622 J-1.639 E.11059
G2 X120.53 Y128.945 I-5.954 J1.953 E.13179
G3 X118.646 Y127.06 I4.318 J-6.202 E.10231
G3 X118.342 Y126.192 I2.207 J-1.261 E.03531
G2 X119.965 Y125.739 I.12 J-2.708 E.06542
G1 X123.107 Y123.11 E.15644
G2 X123.114 Y119.457 I-1.793 J-1.83 E.15529
G2 X126.637 Y115.65 I-43.459 J-43.753 E.1981
G1 X128.915 Y118.662 E.14419
G3 X126.516 Y120.933 I-141.518 J-147.118 E.12612
G2 X125.84 Y123.289 I2.4 J1.963 E.09622
G2 X127.917 Y127.06 I5.597 J-.626 E.16863
G3 X129.802 Y128.945 I-4.318 J6.202 E.10231
G3 X129.974 Y131.772 I-3.623 J1.639 E.11059
G3 X128.071 Y134.6 I-5.955 J-1.953 E.13179
G2 X126.186 Y136.485 I4.318 J6.202 E.10231
G2 X126.014 Y139.313 I3.622 J1.639 E.11059
G2 X127.725 Y141.945 I5.704 J-1.834 E.12126
G1 X133.64 Y141.945 E.22586
G3 X133.555 Y139.313 I3.738 J-1.439 E.10246
G3 X135.458 Y136.485 I5.955 J1.953 E.13179
G2 X137.342 Y134.6 I-4.319 J-6.202 E.10231
G2 X137.514 Y131.772 I-3.622 J-1.639 E.11059
G2 X135.612 Y128.945 I-5.955 J1.953 E.13179
G3 X133.727 Y127.06 I4.318 J-6.202 E.10231
G3 X133.555 Y124.232 I3.622 J-1.639 E.11059
G3 X135.458 Y121.404 I5.955 J1.953 E.13179
G3 X135.984 Y121.112 I.515 J.309 E.02405
G2 X144.009 Y133.73 I50.232 J-23.087 E.57269
G3 X141.597 Y136.014 I-137.426 J-142.727 E.12682
G2 X140.922 Y138.371 I2.4 J1.963 E.09622
G2 X142.806 Y141.945 I5.265 J-.491 E.15831
G1 X148.722 Y141.945 E.22586
G3 X148.836 Y138.796 I3.937 J-1.434 E.12334
G2 X150.591 Y140.347 I37.176 J-40.295 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 6.76
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X149.842 Y139.685 E-.38
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
G1 X122.883 Y122.57
M73 P71 R5
G1 Z6.76
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X122.802 Y122.691 E.00555
G3 X121.438 Y123.862 I-13.183 J-13.969 E.06867
G1 X119.905 Y125.147 E.07636
G1 X119.433 Y125.44 E.02121
G1 X118.989 Y125.573 E.01768
G3 X117.871 Y125.439 I-.307 J-2.174 E.04348
G1 X117.475 Y125.207 E.01752
G1 X117.158 Y124.905 E.01673
G1 X116.646 Y124.295 E.03041
G3 X103.842 Y131.227 I-32.334 J-44.435 E.55754
G1 X104.301 Y131.539 E.02122
G1 X104.706 Y132.248 E.03117
G1 X104.797 Y132.835 E.02266
G1 X104.797 Y141.752 E.34045
G1 X104.702 Y142.353 E.02324
G1 X104.651 Y142.443 E.00395
G1 X154.07 Y142.443 E1.88675
G3 X143.803 Y132.701 I31.485 J-43.463 E.54196
G3 X136.269 Y120.54 I41.717 J-34.259 E.54777
G3 X133.274 Y120.651 I-1.757 J-6.937 E.11527
G1 X132.427 Y120.434 E.03339
G1 X131.483 Y120.053 E.03885
G1 X130.603 Y119.535 E.03898
G1 X129.81 Y118.896 E.03889
G1 X129.117 Y118.145 E.039
G3 X127.854 Y116.454 I83.216 J-63.49 E.0806
G1 X126.662 Y114.848 E.07636
G3 X122.324 Y119.536 I-43.245 J-35.674 E.244
G3 X122.973 Y120.331 I-6.414 J5.897 E.03921
G1 X123.199 Y120.807 E.02013
G1 X123.288 Y121.291 E.01878
G1 X123.248 Y121.808 E.0198
G1 X123.089 Y122.262 E.01836
G1 X122.933 Y122.495 E.01072
G1 X122.336 Y122.315 F36000
G1 F13446.369
G1 X122.218 Y122.444 E.00669
G1 X119.529 Y124.698 E.13396
G1 X119.198 Y124.903 E.01484
G1 X118.799 Y125.009 E.01576
G1 X118.346 Y124.982 E.01734
G3 X117.829 Y124.74 I.306 J-1.329 E.02196
G3 X116.742 Y123.497 I15.229 J-14.413 E.06306
G3 X105.299 Y130.015 I-32.484 J-43.723 E.50401
G1 X104.06 Y130.507 E.05088
G1 X103.688 Y130.654 E.01527
G1 F12245.234
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.666016
G1 F10912.186
G1 X103.232 Y130.855 E.00413
; LINE_WIDTH: 0.712036
G1 F10590.323
G1 X103.149 Y130.91 E.00443
; LINE_WIDTH: 0.758056
G1 F10273.252
G1 X103.065 Y130.965 E.00473
; LINE_WIDTH: 0.804076
G1 F9961.009
G1 X102.981 Y131.02 E.00503
; LINE_WIDTH: 0.850096
G1 F9653.576
G1 X102.897 Y131.075 E.00533
; LINE_WIDTH: 0.896116
G1 F9138.063
G1 X102.813 Y131.13 E.00563
; LINE_WIDTH: 0.942136
G1 F8674.819
G1 X102.729 Y131.184 E.00593
; LINE_WIDTH: 0.988156
G1 F8256.275
G1 X102.645 Y131.239 E.00623
; LINE_WIDTH: 1.03418
G1 F7876.261
G1 X102.562 Y131.294 E.00653
G1 X102.643 Y131.323 E.00564
; LINE_WIDTH: 0.988156
G1 F8256.275
G1 X102.725 Y131.351 E.00538
; LINE_WIDTH: 0.942136
G1 F8674.819
G1 X102.807 Y131.38 E.00512
; LINE_WIDTH: 0.896116
G1 F9138.063
G1 X102.888 Y131.408 E.00486
; LINE_WIDTH: 0.850096
G1 F9653.576
G1 X102.97 Y131.437 E.0046
; LINE_WIDTH: 0.804076
G1 F10230.728
G1 X103.052 Y131.465 E.00434
; LINE_WIDTH: 0.758056
G1 F10503.811
G1 X103.134 Y131.494 E.00408
; LINE_WIDTH: 0.712036
G1 F10780.49
G1 X103.215 Y131.522 E.00382
; LINE_WIDTH: 0.666016
G1 F11060.735
G1 X103.297 Y131.551 E.00356
; LINE_WIDTH: 0.619996
G1 F12402.565
G1 X103.63 Y131.773 E.01527
G1 F13395.96
G1 X103.865 Y131.93 E.01078
G1 F13446.369
G1 X104.148 Y132.426 E.02181
G1 X104.212 Y132.93 E.01942
G1 X104.212 Y141.752 E.3368
G1 X104.145 Y142.173 E.01626
G1 X103.875 Y142.65 E.02094
G1 X103.319 Y143.011 E.0253
; LINE_WIDTH: 0.580946
G1 F14349.05
G1 X103.078 Y143.062 E.00881
; LINE_WIDTH: 0.586456
G1 F14263.206
G1 X103.562 Y143.046 E.01746
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.336 J31.574 E.08349
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.327 I30.059 J-44.23 E.59951
G3 X136.599 Y119.835 I41.178 J-33.825 E.56116
G1 X135.957 Y120.021 E.0255
G1 X135.036 Y120.161 E.03559
G1 X134.345 Y120.172 E.02635
G1 X133.467 Y120.089 E.03368
G1 X132.576 Y119.867 E.03506
G1 X131.714 Y119.514 E.03556
G1 X130.91 Y119.037 E.0357
G1 X130.187 Y118.448 E.03561
G3 X129.066 Y117.106 I8.764 J-8.455 E.06681
G1 X126.683 Y113.893 E.15272
G3 X121.519 Y119.488 I-42.486 J-34.034 E.29092
G1 X122.387 Y120.523 E.05156
G1 X122.633 Y120.961 E.0192
G1 X122.703 Y121.323 E.0141
G1 X122.649 Y121.788 E.01784
G1 X122.463 Y122.176 E.01644
G1 X122.396 Y122.248 E.00375
G1 X121.904 Y121.921 F36000
G1 F13446.369
G1 X121.841 Y121.996 E.00372
G1 X119.152 Y124.249 E.13396
G1 X118.853 Y124.405 E.01286
G1 X118.556 Y124.425 E.01139
G1 X118.34 Y124.366 E.00853
G1 X118.056 Y124.153 E.01358
G1 X116.833 Y122.695 E.07265
G3 X99.49 Y131.452 I-32.391 J-42.596 E.74592
G3 X99.622 Y132.645 I-9.278 J1.633 E.04584
G1 X102.706 Y132.073 E.11974
G1 X103.118 Y132.109 E.01578
G1 X103.438 Y132.332 E.01491
G1 X103.594 Y132.619 E.01248
G1 X103.626 Y132.93 E.01195
G1 X103.626 Y141.752 E.3368
G1 X103.588 Y141.992 E.00928
G1 X103.434 Y142.264 E.01195
G1 X103.121 Y142.481 E.01454
G1 X102.848 Y142.53 E.0106
G1 X102.706 Y142.517 E.00544
G1 X99.626 Y141.947 E.11958
G3 X99.539 Y143.615 I-9.329 J.348 E.06384
G1 X156.235 Y143.615 E2.16462
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.705 Y131.954 I29.104 J-43.629 E.60778
G3 X136.908 Y119.075 I40.882 J-33.549 E.57674
G1 X136.498 Y119.254 E.01709
G1 X135.794 Y119.458 E.02798
G1 X134.957 Y119.581 E.03231
G1 X134.348 Y119.586 E.02325
G1 X133.538 Y119.508 E.03104
G1 X132.725 Y119.301 E.03204
G1 X131.945 Y118.976 E.03228
G1 X131.217 Y118.538 E.03242
G1 X130.564 Y118 E.03232
G1 X129.995 Y117.369 E.03243
G3 X129.081 Y116.142 I130.578 J-98.2 E.0584
G1 X126.698 Y112.93 E.15272
G3 X120.715 Y119.44 I-42.752 J-33.287 E.33795
G1 X121.938 Y120.899 E.07269
G1 X122.095 Y121.206 E.01317
G1 X122.116 Y121.47 E.01014
G1 X122.01 Y121.798 E.01315
G1 X121.963 Y121.853 E.00276
M204 S250
G1 X121.486 Y121.572 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.797 Y123.826 E.11109
G1 X118.642 Y123.878 E.0052
G1 X118.55 Y123.833 E.00324
G3 X118.475 Y123.793 I.008 J-.105 E.00275
G1 X118.459 Y123.773 E.00082
G1 X118.442 Y123.753 E.00082
G1 X118.425 Y123.733 E.00082
G1 F3450
G1 X117.546 Y122.685 E.04332
G1 F3600
G1 X117.126 Y122.184 E.02069
; LINE_WIDTH: 0.520646
G1 X117.118 Y122.175 E.0004
; LINE_WIDTH: 0.521736
G1 X117.103 Y122.159 E.00068
; LINE_WIDTH: 0.522836
G1 X117.088 Y122.143 E.00068
; LINE_WIDTH: 0.523196
G1 X117.083 Y122.138 E.00023
; LINE_WIDTH: 0.531656
G1 X117.076 Y122.014 E.00402
; LINE_WIDTH: 0.544336
G1 X117.066 Y121.828 E.00619
G1 X116.662 Y122.131 E.01677
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-32.247 J-42.085 E.6318
G1 X98.943 Y131.537 E.01581
G1 X99.068 Y132.518 E.03131
G1 X99.072 Y133.309 E.02503
G1 X102.807 Y132.617 E.12024
G1 X102.926 Y132.627 E.00379
G1 X103.064 Y132.775 E.00641
G3 X103.073 Y135.752 I-181.8 J2.051 E.09424
G1 X103.073 Y141.752 E.18996
G1 X103.018 Y141.9 E.00502
G1 X102.848 Y141.978 E.00591
G1 X102.807 Y141.974 E.00131
G1 X99.073 Y141.282 E.1202
G1 X99.073 Y142.625 E.04251
G3 X98.941 Y143.858 I-6.035 J-.022 E.03931
G1 X98.937 Y144.167 E.00981
G1 X156.695 Y144.167 E1.82862
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.614 J-43.499 E.51063
G3 X137.192 Y118.316 I40.279 J-33.076 E.49162
G3 X136.344 Y118.723 I-5.233 J-9.822 E.0298
G1 X135.64 Y118.927 E.02321
G3 X133.606 Y118.959 I-1.106 J-5.7 E.06474
G1 X132.866 Y118.766 E.02421
G1 X132.162 Y118.468 E.0242
G1 X131.507 Y118.067 E.02431
G1 X130.919 Y117.576 E.02423
G1 X130.409 Y117.003 E.02432
G3 X129.087 Y115.222 I605.948 J-451.23 E.0702
G1 X126.704 Y112.01 E.12664
G3 X120.192 Y119.172 I-42.717 J-32.295 E.30689
; LINE_WIDTH: 0.544336
G1 X119.787 Y119.545 E.01829
G1 X119.887 Y119.566 E.00338
; LINE_WIDTH: 0.537536
G1 X119.986 Y119.586 E.00334
; LINE_WIDTH: 0.530746
G1 X120.086 Y119.607 E.00329
; LINE_WIDTH: 0.523956
G1 X120.144 Y119.619 E.00189
; LINE_WIDTH: 0.519996
G1 X120.153 Y119.63 E.00047
G1 X120.176 Y119.657 E.00111
G1 F3450
G1 X120.199 Y119.684 E.00111
G1 F3300
G1 X120.221 Y119.711 E.00111
G1 F3150
G1 X120.313 Y119.821 E.00454
G1 F3300
G1 X120.543 Y120.095 E.01132
G1 F3450
G1 X120.772 Y120.369 E.01132
G1 F3600
G1 X121.002 Y120.643 E.01132
G1 X121.232 Y120.917 E.01132
G1 X121.461 Y121.191 E.01132
G3 X121.537 Y121.326 I-.125 J.158 E.00503
G1 X121.566 Y121.42 E.00311
G1 X121.528 Y121.492 E.00259
; WIPE_START
M204 S10000
G1 X120.768 Y122.142 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X119.787 Y119.545 Z7.16 F36000
G1 Z6.76
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X117.066 Y121.828 E.1181
; WIPE_START
M204 S10000
G1 X117.832 Y121.185 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X111.309 Y125.148 Z7.16 F36000
G1 X100.191 Y131.901 Z7.16
G1 Z6.76
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.743056
G1 F11111.582
G1 X100.503 Y131.823 E.01487
; LINE_WIDTH: 0.781956
G1 F10533.427
G1 X100.816 Y131.746 E.01569
; LINE_WIDTH: 0.820856
G1 F10012.46
G1 X101.128 Y131.668 E.0165
; LINE_WIDTH: 0.825016
G1 F9959.782
G1 X101.159 Y131.66 E.00163
; LINE_WIDTH: 0.871176
G1 F9410.402
G1 X101.473 Y131.579 E.0177
; LINE_WIDTH: 0.917336
G1 F8918.461
G1 X101.787 Y131.497 E.01868
; LINE_WIDTH: 0.963496
G1 F8475.398
G1 X102.101 Y131.415 E.01965
; LINE_WIDTH: 1.00966
G1 F8074.273
G1 X102.415 Y131.334 E.02063
; LINE_WIDTH: 1.03418
G1 F7876.261
G1 X102.562 Y131.294 E.00992
; WIPE_START
G1 X102.415 Y131.334 E-.05781
G1 X102.101 Y131.415 E-.12329
G1 X101.787 Y131.497 E-.12329
G1 X101.594 Y131.547 E-.0756
; WIPE_END
G1 E-.02 F1800
G1 X100.776 Y139.135 Z7.16 F36000
G1 X100.374 Y142.857 Z7.16
G1 Z6.76
G1 E.4 F1800
; LINE_WIDTH: 0.964796
G1 F8463.557
G1 X100.614 Y142.878 E.01457
; LINE_WIDTH: 0.920876
G1 F8882.849
G1 X100.862 Y142.901 E.01445
; LINE_WIDTH: 0.875159
G1 F9365.83
G1 X101.111 Y142.924 E.0137
; LINE_WIDTH: 0.829441
G1 F9904.353
G1 X101.36 Y142.947 E.01296
; LINE_WIDTH: 0.783724
G1 F10508.583
G1 X101.609 Y142.97 E.01221
; LINE_WIDTH: 0.738006
G1 F11191.326
G1 X101.858 Y142.993 E.01147
; LINE_WIDTH: 0.692289
G1 F11968.95
G1 X102.107 Y143.016 E.01072
; LINE_WIDTH: 0.646571
G1 F12862.708
G1 X102.356 Y143.038 E.00998
; LINE_WIDTH: 0.600854
G1 F13900.72
G1 X102.605 Y143.061 E.00923
; LINE_WIDTH: 0.555136
G1 F15120.969
G1 X103.078 Y143.062 E.01604
; WIPE_START
G1 X102.605 Y143.061 E-.17951
G1 X102.356 Y143.038 E-.095
G1 X102.107 Y143.016 E-.095
G1 X102.08 Y143.013 E-.01049
; WIPE_END
G1 E-.02 F1800
G1 X105.295 Y138.701 Z7.16 F36000
G1 Z6.76
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.295 Y136.358 E.08944
G2 X106.997 Y134.6 I-5.35 J-6.881 E.09371
G2 X107.322 Y131.772 I-3.489 J-1.834 E.11118
G2 X106.826 Y130.527 I-3.199 J.551 E.05155
G2 X112.979 Y127.317 I-22.494 J-50.614 E.26515
G3 X114.537 Y128.945 I-5.053 J6.395 E.08632
G3 X114.863 Y131.772 I-3.489 J1.834 E.11118
G3 X113.131 Y134.6 I-5.799 J-1.608 E.12826
G2 X111.288 Y136.485 I5.652 J7.367 E.10098
G2 X110.963 Y139.313 I3.489 J1.834 E.11119
G2 X112.517 Y141.945 I5.559 J-1.507 E.11811
G1 X118.732 Y141.945 E.23728
G3 X118.821 Y138.371 I3.996 J-1.689 E.14072
G3 X120.236 Y136.485 I5.201 J2.43 E.09063
G2 X122.078 Y134.6 I-5.651 J-7.367 E.10098
G2 X122.403 Y131.772 I-3.489 J-1.834 E.11118
G2 X120.671 Y128.945 I-5.8 J1.608 E.12826
G3 X118.829 Y127.06 I5.652 J-7.367 E.10098
G3 X118.451 Y126.088 I3.298 J-1.842 E.03992
G2 X120.088 Y125.636 I.209 J-2.434 E.06624
G1 X122.958 Y123.236 E.14283
G2 X122.998 Y119.571 I-1.754 J-1.852 E.15614
G2 X126.639 Y115.651 I-45.332 J-45.749 E.2043
G1 X128.952 Y118.707 E.14633
G3 X126.715 Y120.933 I-44.88 J-42.874 E.12051
G2 X125.917 Y123.289 I2.463 J2.147 E.09744
G2 X127.776 Y127.06 I5.395 J-.317 E.16484
G3 X129.619 Y128.945 I-5.652 J7.367 E.10098
G3 X129.944 Y131.772 I-3.489 J1.834 E.11118
G3 X128.212 Y134.6 I-5.799 J-1.608 E.12826
G2 X126.369 Y136.485 I5.651 J7.367 E.10098
G2 X126.044 Y139.313 I3.489 J1.834 E.11118
G2 X127.598 Y141.945 I5.56 J-1.507 E.11811
G1 X133.813 Y141.945 E.23728
G3 X133.902 Y138.371 I3.996 J-1.689 E.14072
G3 X135.317 Y136.485 I5.201 J2.43 E.09063
G2 X137.159 Y134.6 I-5.653 J-7.368 E.10098
G2 X137.484 Y131.772 I-3.489 J-1.834 E.11118
G2 X135.753 Y128.945 I-5.799 J1.608 E.12826
G3 X133.91 Y127.06 I5.651 J-7.367 E.10098
G3 X133.585 Y124.232 I3.489 J-1.834 E.11118
G3 X135.317 Y121.404 I5.799 J1.608 E.12826
G3 X135.988 Y121.121 I.547 J.359 E.0295
G2 X144.046 Y133.775 I50.281 J-23.126 E.57452
G3 X141.796 Y136.014 I-44.505 J-42.46 E.1212
G2 X140.999 Y138.371 I2.463 J2.147 E.09744
G2 X142.679 Y141.945 I5.078 J-.205 E.1549
G1 X148.894 Y141.945 E.23728
G3 X148.839 Y138.799 I3.808 J-1.64 E.12322
G2 X150.594 Y140.35 I37.395 J-40.544 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 6.92
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X149.845 Y139.688 E-.38
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
G1 X122.716 Y122.727
G1 Z6.92
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X122.603 Y122.875 E.00712
G3 X121.56 Y123.76 I-10.801 J-11.673 E.05222
G1 X120.027 Y125.044 E.07636
G1 X119.558 Y125.336 E.02108
G1 X119.112 Y125.471 E.01782
G3 X118.055 Y125.362 I-.314 J-2.146 E.04096
G1 X117.672 Y125.158 E.0166
G1 X117.281 Y124.803 E.02018
G1 X116.776 Y124.2 E.03001
G3 X103.833 Y131.23 I-32.454 J-44.326 E.56401
G1 X104.375 Y131.621 E.02553
G1 X104.738 Y132.352 E.03117
G1 X104.8 Y132.927 E.02208
G1 X104.8 Y141.754 E.33702
G1 X104.705 Y142.356 E.02324
G1 X104.655 Y142.443 E.00384
G1 X154.069 Y142.443 E1.8866
G3 X143.806 Y132.705 I31.802 J-43.794 E.54171
G3 X136.271 Y120.545 I42.204 J-34.567 E.54775
G1 X135.451 Y120.705 E.03188
G1 X134.432 Y120.764 E.03896
G3 X132.413 Y120.43 I.293 J-8.046 E.07836
G1 X131.481 Y120.051 E.03841
G1 X130.603 Y119.535 E.0389
G1 X129.808 Y118.894 E.03896
G1 X129.117 Y118.145 E.03892
G3 X127.853 Y116.453 I123.531 J-93.608 E.08063
G1 X126.662 Y114.847 E.07636
G3 X122.207 Y119.645 I-41.719 J-34.264 E.25013
G3 X122.905 Y120.523 I-4.744 J4.49 E.04289
G1 X123.1 Y120.988 E.01924
G1 X123.168 Y121.523 E.02059
G1 X123.093 Y122.037 E.01983
G1 X122.905 Y122.481 E.01842
G1 X122.771 Y122.655 E.00839
G1 X122.198 Y122.441 F36000
G1 F13446.369
G1 X122.086 Y122.555 E.00611
G1 X119.651 Y124.596 E.12129
G1 X119.323 Y124.799 E.01475
G1 X118.927 Y124.906 E.01567
G1 X118.479 Y124.882 E.01713
G1 X118.027 Y124.691 E.01871
G1 X117.729 Y124.426 E.01522
G1 X116.87 Y123.401 E.05109
G3 X102.416 Y131.127 I-32.449 J-43.325 E.62815
G1 X102.84 Y131.473 E.02093
G1 X103.323 Y131.557 E.01871
G1 X103.917 Y131.986 E.02797
G1 X104.171 Y132.498 E.02181
G1 X104.214 Y132.927 E.01648
G1 X104.214 Y141.754 E.33702
G1 X104.147 Y142.175 E.01626
G1 X103.877 Y142.652 E.02094
G1 X103.321 Y143.013 E.0253
; LINE_WIDTH: 0.579436
G1 F14348.41
G1 X103.079 Y143.064 E.00878
; LINE_WIDTH: 0.585186
G1 F14296.09
G1 X103.564 Y143.046 E.0174
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.335 J30.382 E.08344
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.256 Y132.33 I29.801 J-43.95 E.59939
G3 X136.596 Y119.828 I41.63 J-34.106 E.56156
G1 X136.064 Y119.995 E.02127
G1 X135.344 Y120.129 E.02797
G1 X134.411 Y120.179 E.03567
G1 X133.483 Y120.091 E.03559
G1 X132.578 Y119.868 E.03558
G1 X131.712 Y119.513 E.03574
G1 X130.91 Y119.036 E.03562
G1 X130.185 Y118.446 E.03567
G1 X129.556 Y117.757 E.03563
G3 X127.875 Y115.499 I225.103 J-169.362 E.10747
G1 X126.683 Y113.893 E.07636
G3 X121.402 Y119.597 I-44.343 J-35.753 E.29702
G1 X122.264 Y120.625 E.05121
G1 X122.399 Y120.817 E.00895
G1 X122.556 Y121.234 E.017
G1 X122.578 Y121.614 E.01454
G1 X122.501 Y121.967 E.0138
G1 X122.286 Y122.351 E.0168
G1 X122.261 Y122.376 E.00137
G1 X121.775 Y122.035 F36000
G1 F13446.369
G1 X121.71 Y122.106 E.00369
G1 X119.275 Y124.147 E.12129
G1 X118.978 Y124.302 E.01278
G1 X118.682 Y124.323 E.01132
G1 X118.348 Y124.201 E.01358
G1 X117.924 Y123.747 E.02374
G1 X116.961 Y122.598 E.05723
G3 X102.18 Y130.587 I-32.565 J-42.581 E.64414
G1 X101.922 Y130.675 E.01041
G1 X101.543 Y130.802 E.01527
G1 F13348.619
G1 X101.164 Y130.93 E.01527
G1 F11955.115
G1 X100.784 Y131.057 E.01527
; LINE_WIDTH: 0.667352
G1 F10638.412
G1 X100.713 Y131.104 E.00352
; LINE_WIDTH: 0.714707
G1 F10367.378
G1 X100.642 Y131.151 E.00379
; LINE_WIDTH: 0.762063
G1 F10099.816
G1 X100.571 Y131.198 E.00405
; LINE_WIDTH: 0.809418
G1 F9835.76
G1 X100.5 Y131.245 E.00431
; LINE_WIDTH: 0.856774
G1 F9575.193
G1 X100.429 Y131.292 E.00458
; LINE_WIDTH: 0.90413
G1 F9053.875
G1 X100.357 Y131.34 E.00484
; LINE_WIDTH: 0.951485
G1 F8586.392
G1 X100.286 Y131.387 E.0051
; LINE_WIDTH: 0.998841
G1 F8164.815
G1 X100.215 Y131.434 E.00537
; LINE_WIDTH: 1.0462
G1 F7782.697
G1 X100.144 Y131.481 E.00563
G1 X100.228 Y131.497 E.00563
; LINE_WIDTH: 0.998841
G1 F8164.815
G1 X100.312 Y131.513 E.00537
; LINE_WIDTH: 0.951485
G1 F8586.392
G1 X100.395 Y131.529 E.0051
; LINE_WIDTH: 0.90413
G1 F9053.875
G1 X100.479 Y131.545 E.00484
; LINE_WIDTH: 0.856774
G1 F9575.193
G1 X100.563 Y131.562 E.00458
; LINE_WIDTH: 0.809418
G1 F10160.214
G1 X100.647 Y131.578 E.00431
; LINE_WIDTH: 0.762063
G1 F10428.557
G1 X100.731 Y131.594 E.00405
; LINE_WIDTH: 0.714707
G1 F10700.405
G1 X100.815 Y131.61 E.00379
; LINE_WIDTH: 0.667352
G1 F10975.75
G1 X100.898 Y131.626 E.00352
; LINE_WIDTH: 0.619996
G1 F12312.562
G1 X101.296 Y131.587 E.01527
G1 F13446.369
G1 X101.698 Y131.548 E.01541
G1 X102.121 Y131.626 E.01641
G1 X102.392 Y131.849 E.01341
G3 X102.473 Y132.114 I-.212 J.21 E.01101
G1 X102.905 Y132.06 E.01662
G1 X103.12 Y132.106 E.00839
G1 X103.469 Y132.364 E.01656
G1 X103.608 Y132.659 E.01243
G3 X103.628 Y135.754 I-93.053 J2.167 E.11821
G1 X103.628 Y141.754 E.22907
G1 X103.59 Y141.994 E.00928
G1 X103.436 Y142.267 E.01195
G1 X103.123 Y142.483 E.01454
G1 X102.85 Y142.533 E.0106
G1 X102.708 Y142.52 E.00544
G1 X99.628 Y141.949 E.11958
G3 X99.54 Y143.615 I-9.196 J.346 E.06375
G1 X156.235 Y143.615 E2.16459
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.707 Y131.956 I29.324 J-43.871 E.60765
G3 X136.912 Y119.083 I40.72 J-33.454 E.57652
G1 X136.093 Y119.382 E.03328
G3 X134.389 Y119.593 I-1.766 J-7.25 E.06568
G1 X133.547 Y119.509 E.03231
G1 X132.727 Y119.301 E.0323
G1 X131.942 Y118.975 E.03245
G1 X131.217 Y118.538 E.03233
G1 X130.562 Y117.998 E.03239
G1 X129.995 Y117.369 E.03235
G3 X129.081 Y116.142 I204.604 J-153.328 E.0584
G1 X126.698 Y112.93 E.15272
G3 X120.598 Y119.549 I-44.266 J-34.67 E.34404
G1 X121.815 Y121.002 E.07234
G1 X121.975 Y121.319 E.01358
G1 X121.981 Y121.657 E.01289
G1 X121.864 Y121.938 E.01162
G1 X121.836 Y121.968 E.00158
M204 S250
G1 X121.364 Y121.675 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X118.92 Y123.723 E.10095
G1 X118.748 Y123.774 E.00567
G1 X118.666 Y123.729 E.00297
G3 X118.554 Y123.638 I.06 J-.186 E.00467
G1 X118.346 Y123.39 E.01025
G1 X118.138 Y123.142 E.01025
G1 X117.93 Y122.894 E.01025
G1 F3450
G1 X117.722 Y122.646 E.01025
G1 F3300
G1 X117.514 Y122.398 E.01025
G1 F3150
G1 X117.334 Y122.182 E.00891
; LINE_WIDTH: 0.520446
G1 F3600
G1 X117.316 Y122.161 E.00086
; LINE_WIDTH: 0.520966
G1 X117.295 Y122.138 E.00099
; LINE_WIDTH: 0.521486
G1 X117.275 Y122.114 E.001
; LINE_WIDTH: 0.521996
G1 X117.254 Y122.09 E.001
; LINE_WIDTH: 0.522516
G1 X117.234 Y122.067 E.001
; LINE_WIDTH: 0.523026
G1 X117.213 Y122.043 E.001
; LINE_WIDTH: 0.523196
G1 X117.206 Y122.035 E.00033
; LINE_WIDTH: 0.531656
G1 X117.199 Y121.911 E.00402
; LINE_WIDTH: 0.544336
G1 X117.188 Y121.725 E.00619
G1 X116.672 Y122.124 E.0217
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-32.254 J-42.072 E.63217
G1 X98.974 Y131.728 E.02189
; LINE_WIDTH: 0.562156
G1 X99.014 Y131.884 E.00553
; LINE_WIDTH: 0.604316
G1 X99.054 Y132.039 E.00598
; LINE_WIDTH: 0.646476
M73 P72 R5
G1 X99.094 Y132.195 E.00642
; LINE_WIDTH: 0.688636
G1 X99.134 Y132.351 E.00686
; LINE_WIDTH: 0.730796
G1 X99.174 Y132.507 E.0073
; LINE_WIDTH: 0.745976
G1 X99.187 Y133.17 E.03079
G1 X99.992 Y133.021 E.03801
G1 X99.907 Y132.527 E.02325
; LINE_WIDTH: 0.734336
G1 X99.903 Y132.469 E.00269
; LINE_WIDTH: 0.730796
G1 X99.957 Y132.425 E.00313
; LINE_WIDTH: 0.688636
G1 X100.01 Y132.382 E.00294
; LINE_WIDTH: 0.646476
G1 X100.064 Y132.339 E.00275
; LINE_WIDTH: 0.604316
G1 X100.117 Y132.295 E.00256
; LINE_WIDTH: 0.562156
G1 X100.171 Y132.252 E.00237
; LINE_WIDTH: 0.519996
G2 X101.752 Y132.098 I-119.192 J-1233.852 E.05029
G1 X101.939 Y132.169 E.00633
G1 X101.989 Y132.392 E.00725
G2 X101.918 Y132.78 I.61 J.312 E.01264
G1 X102.809 Y132.615 E.02868
G3 X103.075 Y132.836 I.024 J.243 E.01222
G1 X103.075 Y141.754 E.28234
G1 X103.02 Y141.903 E.00502
G1 X102.85 Y141.98 E.00591
G1 X102.809 Y141.976 E.00131
G1 X99.075 Y141.285 E.1202
G1 X99.075 Y142.625 E.04243
G3 X98.942 Y143.858 I-5.969 J-.024 E.03932
G1 X98.937 Y144.167 E.00981
G1 X156.695 Y144.167 E1.82862
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.611 J-43.496 E.5106
G3 X137.192 Y118.315 I40.429 J-33.166 E.49167
G1 X136.612 Y118.616 E.02069
G1 X135.913 Y118.859 E.02344
G1 X135.135 Y119.01 E.02507
G1 X134.369 Y119.041 E.02429
G1 X133.608 Y118.959 E.02422
G1 X132.867 Y118.767 E.02422
G1 X132.16 Y118.466 E.02434
G1 X131.507 Y118.067 E.02424
G1 X130.918 Y117.575 E.02429
G1 X130.409 Y117.003 E.02426
G3 X129.087 Y115.222 I976.73 J-726.493 E.0702
G1 X126.704 Y112.01 E.12664
G3 X120.192 Y119.172 I-42.396 J-32.004 E.3069
; LINE_WIDTH: 0.544336
G1 X119.664 Y119.648 E.02361
G1 X119.859 Y119.674 E.00651
; LINE_WIDTH: 0.531166
G1 X120 Y119.694 E.00461
; LINE_WIDTH: 0.521596
G1 X120.006 Y119.701 E.00031
; LINE_WIDTH: 0.521526
G1 X120.029 Y119.729 E.00114
; LINE_WIDTH: 0.521286
G1 X120.052 Y119.756 E.00114
; LINE_WIDTH: 0.521036
G1 X120.074 Y119.784 E.00114
; LINE_WIDTH: 0.520796
G1 X120.097 Y119.812 E.00114
; LINE_WIDTH: 0.520556
G1 X120.12 Y119.839 E.00114
; LINE_WIDTH: 0.520306
G1 X120.143 Y119.867 E.00114
; LINE_WIDTH: 0.519996
G1 X120.21 Y119.947 E.00329
G1 F3150
G1 X120.399 Y120.173 E.00935
G1 F3300
G1 X120.589 Y120.399 E.00935
G1 F3450
G1 X120.779 Y120.626 E.00935
G1 F3600
G1 X120.968 Y120.852 E.00935
G1 X121.158 Y121.078 E.00935
G1 X121.348 Y121.305 E.00935
G3 X121.414 Y121.428 I-.117 J.142 E.00454
G1 X121.444 Y121.52 E.00307
G1 X121.405 Y121.595 E.00265
; WIPE_START
M204 S10000
G1 X120.645 Y122.245 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X119.664 Y119.648 Z7.32 F36000
G1 Z6.92
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X117.188 Y121.725 E.10746
; WIPE_START
M204 S10000
G1 X117.954 Y121.083 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X111.363 Y124.931 Z7.32 F36000
G1 X100.144 Y131.481 Z7.32
G1 Z6.92
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.0548
G1 F7717.108
G1 X99.735 Y131.607 E.02848
; WIPE_START
G1 X100.144 Y131.481 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X100.299 Y139.112 Z7.32 F36000
G1 X100.375 Y142.858 Z7.32
G1 Z6.92
G1 E.4 F1800
; LINE_WIDTH: 0.962536
G1 F8484.163
G1 X100.616 Y142.88 E.01463
; LINE_WIDTH: 0.918336
G1 F8908.372
G1 X100.865 Y142.903 E.01441
; LINE_WIDTH: 0.872619
G1 F9394.209
G1 X101.114 Y142.925 E.01366
; LINE_WIDTH: 0.826901
G1 F9936.095
G1 X101.363 Y142.948 E.01292
; LINE_WIDTH: 0.781184
G1 F10544.322
G1 X101.612 Y142.971 E.01217
; LINE_WIDTH: 0.735466
G1 F11231.869
G1 X101.861 Y142.994 E.01143
; LINE_WIDTH: 0.689749
G1 F12015.335
G1 X102.11 Y143.017 E.01068
; LINE_WIDTH: 0.644031
G1 F12916.296
G1 X102.359 Y143.04 E.00994
; LINE_WIDTH: 0.598314
G1 F13963.325
G1 X102.607 Y143.063 E.00919
; LINE_WIDTH: 0.552596
G1 F15195.077
G1 X103.079 Y143.064 E.01595
; WIPE_START
G1 X102.607 Y143.063 E-.17934
G1 X102.359 Y143.04 E-.095
G1 X102.11 Y143.017 E-.095
G1 X102.082 Y143.014 E-.01066
; WIPE_END
G1 E-.02 F1800
G1 X105.297 Y138.564 Z7.32 F36000
G1 Z6.92
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.297 Y136.221 E.08944
G2 X107.085 Y134.129 I-5.362 J-6.391 E.10553
G2 X106.859 Y130.515 I-4.337 J-1.543 E.14209
G2 X112.896 Y127.379 I-24.11 J-53.784 E.25987
G3 X114.625 Y129.416 I-5.228 J6.191 E.1025
G3 X114.065 Y133.658 I-4.493 J1.564 E.16943
G3 X111.818 Y136.014 I-25.673 J-22.233 E.12436
G2 X110.987 Y139.313 I2.788 J2.456 E.13491
G2 X112.393 Y141.945 I5.393 J-1.189 E.11537
G1 X118.892 Y141.945 E.24812
G3 X118.799 Y138.371 I3.903 J-1.89 E.14073
G3 X120.099 Y136.485 I5.027 J2.076 E.0881
G2 X121.908 Y134.6 I-7.479 J-8.989 E.09997
G2 X122.108 Y130.83 I-3.762 J-2.09 E.14923
G2 X120.808 Y128.945 I-5.027 J2.076 E.0881
G3 X118.999 Y127.06 I7.477 J-8.987 E.09997
G3 X118.556 Y125.985 I2.72 J-1.748 E.04461
G2 X120.212 Y125.533 I.222 J-2.442 E.06694
G1 X122.857 Y123.32 E.13166
G2 X122.886 Y119.681 I-1.691 J-1.833 E.15574
G2 X126.637 Y115.65 I-44.133 J-44.823 E.2103
G1 X128.996 Y118.754 E.14885
G3 X126.899 Y120.933 I-26.684 J-23.582 E.11547
G2 X126.068 Y124.232 I2.788 J2.456 E.13491
G2 X127.64 Y127.06 I5.621 J-1.273 E.12519
G3 X129.449 Y128.945 I-7.477 J8.987 E.09997
G3 X129.649 Y132.715 I-3.762 J2.09 E.14923
G3 X128.349 Y134.6 I-5.027 J-2.076 E.0881
G2 X126.539 Y136.485 I7.478 J8.988 E.09997
G2 X126.339 Y140.256 I3.762 J2.09 E.14923
G2 X127.474 Y141.945 I4.49 J-1.79 E.0783
G1 X133.973 Y141.945 E.24812
G3 X133.88 Y138.371 I3.903 J-1.89 E.14073
G3 X135.18 Y136.485 I5.027 J2.077 E.0881
G2 X136.99 Y134.6 I-7.478 J-8.988 E.09997
G2 X137.19 Y130.83 I-3.762 J-2.09 E.14923
G2 X135.889 Y128.945 I-5.027 J2.077 E.0881
G3 X134.08 Y127.06 I7.477 J-8.987 E.09997
G3 X133.88 Y123.289 I3.762 J-2.09 E.14923
G3 X135.382 Y121.213 I6.833 J3.363 E.09832
G2 X135.984 Y121.112 I-.382 J-4.132 E.02332
G2 X144.088 Y133.822 I50.188 J-23.063 E.5773
G3 X141.98 Y136.014 I-26.567 J-23.439 E.11615
G2 X141.15 Y139.313 I2.788 J2.456 E.13491
G2 X142.555 Y141.945 I5.392 J-1.188 E.11537
G1 X149.054 Y141.945 E.24812
G3 X148.838 Y138.799 I3.733 J-1.837 E.1235
G2 X150.593 Y140.35 I39.711 J-43.161 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 7.08
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X149.844 Y139.688 E-.38
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
G1 X122.541 Y122.886
G1 Z7.08
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X122.366 Y123.084 E.01011
G3 X121.683 Y123.657 I-49.13 J-57.901 E.03402
G1 X120.15 Y124.942 E.07636
G1 X119.684 Y125.231 E.02094
G1 X119.234 Y125.368 E.01795
G1 X118.824 Y125.396 E.01569
G1 X118.272 Y125.294 E.02146
G1 X117.867 Y125.102 E.0171
G1 X117.403 Y124.7 E.02343
G1 X116.905 Y124.105 E.02962
G3 X104.208 Y131.087 I-32.59 J-44.236 E.55482
G1 X104.375 Y131.257 E.00911
G3 X104.72 Y132.276 I-9.154 J3.666 E.04111
G1 X104.802 Y132.829 E.02134
G1 X104.802 Y141.757 E.34085
G1 X104.707 Y142.358 E.02324
G1 X104.658 Y142.443 E.00373
G1 X154.069 Y142.443 E1.88645
G3 X143.801 Y132.699 I31.502 J-43.477 E.54202
G3 X136.271 Y120.545 I41.778 J-34.294 E.54748
G1 X135.454 Y120.705 E.03178
G3 X133.274 Y120.651 I-.892 J-8.104 E.08351
G1 X132.429 Y120.434 E.03332
G1 X131.483 Y120.052 E.03896
G1 X130.602 Y119.535 E.039
G1 X129.809 Y118.895 E.03891
G1 X129.117 Y118.145 E.03894
G3 X127.853 Y116.453 I99.185 J-75.419 E.08063
G1 X126.662 Y114.847 E.07636
G3 X122.09 Y119.755 I-44.772 J-37.119 E.25623
G1 X122.59 Y120.352 E.02974
G1 X122.83 Y120.713 E.01656
G1 X123.005 Y121.208 E.02003
G1 X123.041 Y121.748 E.02066
G1 X122.936 Y122.251 E.01962
G1 X122.723 Y122.679 E.01825
G1 X122.6 Y122.818 E.00709
G1 X122.081 Y122.537 F36000
G1 F13446.369
G1 X121.972 Y122.65 E.00597
G1 X119.774 Y124.493 E.10952
G1 X119.448 Y124.695 E.01465
G1 X119.054 Y124.802 E.01557
G1 X118.635 Y124.786 E.01603
G1 X118.176 Y124.605 E.01883
G1 X117.71 Y124.154 E.02479
G1 X116.998 Y123.304 E.04231
G3 X102.926 Y130.942 I-32.482 J-43.063 E.61355
G1 X103.305 Y131.017 E.01476
G1 X103.854 Y131.527 E.02861
G3 X104.216 Y132.831 I-5.181 J2.139 E.05178
G1 X104.216 Y141.757 E.3408
G1 X104.15 Y142.178 E.01626
G1 X103.879 Y142.655 E.02094
G1 X103.323 Y143.015 E.02529
; LINE_WIDTH: 0.577936
G1 F14347.862
G1 X103.081 Y143.065 E.00875
; LINE_WIDTH: 0.583926
G1 F14328.865
G1 X103.565 Y143.047 E.01735
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.049 Y143.029 E.01848
G1 X155.749 Y143.029 E1.97385
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.253 Y132.326 I30.053 J-44.224 E.59957
G3 X136.596 Y119.828 I41.878 J-34.251 E.56132
G1 X136.067 Y119.995 E.02115
G1 X135.346 Y120.129 E.02799
G1 X134.411 Y120.179 E.03575
G3 X133.286 Y120.054 I.625 J-10.793 E.04323
G1 X132.566 Y119.865 E.02845
G1 X131.713 Y119.514 E.03521
G1 X130.909 Y119.036 E.03571
G1 X130.186 Y118.447 E.03562
G3 X129.074 Y117.116 I6.387 J-6.465 E.06628
G1 X126.683 Y113.893 E.15323
G3 X121.286 Y119.707 I-42.38 J-33.931 E.30316
G1 X122.141 Y120.728 E.05087
G1 X122.309 Y120.981 E.01159
G1 X122.432 Y121.327 E.01401
G1 X122.456 Y121.705 E.01445
G1 X122.383 Y122.057 E.01372
G1 X122.173 Y122.442 E.01674
G1 X122.143 Y122.472 E.00164
G1 X121.656 Y122.134 F36000
G1 F13446.369
G1 X121.596 Y122.201 E.00343
G1 X119.398 Y124.044 E.10952
G1 X119.103 Y124.198 E.0127
G1 X118.762 Y124.214 E.01303
G1 X118.486 Y124.108 E.01128
G1 X118.129 Y123.742 E.01954
G1 X117.086 Y122.498 E.06197
G3 X102.165 Y130.593 I-32.657 J-42.395 E.65086
G1 X101.922 Y130.676 E.00978
G1 X101.543 Y130.803 E.01527
G1 F13348.486
G1 X101.164 Y130.93 E.01527
G1 F11954.99
G1 X100.784 Y131.057 E.01527
; LINE_WIDTH: 0.667354
G1 F10638.294
G1 X100.713 Y131.104 E.00352
; LINE_WIDTH: 0.714712
G1 F10367.261
G1 X100.642 Y131.151 E.00379
; LINE_WIDTH: 0.76207
G1 F10099.683
G1 X100.571 Y131.198 E.00405
; LINE_WIDTH: 0.809427
G1 F9835.646
G1 X100.5 Y131.245 E.00431
; LINE_WIDTH: 0.856785
G1 F9575.064
G1 X100.429 Y131.292 E.00458
; LINE_WIDTH: 0.904143
G1 F9053.737
G1 X100.357 Y131.34 E.00484
; LINE_WIDTH: 0.951501
G1 F8586.247
G1 X100.286 Y131.387 E.0051
; LINE_WIDTH: 0.998858
G1 F8164.665
G1 X100.215 Y131.434 E.00537
; LINE_WIDTH: 1.04622
G1 F7782.544
G1 X100.144 Y131.481 E.00563
G1 X100.228 Y131.497 E.00563
; LINE_WIDTH: 0.998858
G1 F8164.665
G1 X100.312 Y131.513 E.00537
; LINE_WIDTH: 0.951501
G1 F8586.247
G1 X100.395 Y131.529 E.0051
; LINE_WIDTH: 0.904143
G1 F9053.737
G1 X100.479 Y131.545 E.00484
; LINE_WIDTH: 0.856785
G1 F9575.064
G1 X100.563 Y131.562 E.00458
; LINE_WIDTH: 0.809427
G1 F10160.097
G1 X100.647 Y131.578 E.00431
; LINE_WIDTH: 0.76207
G1 F10428.445
G1 X100.731 Y131.594 E.00405
; LINE_WIDTH: 0.714712
G1 F10700.285
G1 X100.815 Y131.61 E.00379
; LINE_WIDTH: 0.667354
G1 F10975.629
G1 X100.898 Y131.626 E.00352
; LINE_WIDTH: 0.619996
G1 F12312.434
G1 X101.296 Y131.587 E.01527
G1 F13446.369
G1 X101.695 Y131.548 E.01527
G1 X102.612 Y131.459 E.03519
G1 X103.04 Y131.539 E.01663
G1 X103.361 Y131.843 E.0169
G1 X103.466 Y132.235 E.01547
G1 X103.448 Y132.348 E.00438
G1 X103.598 Y132.611 E.01155
G1 X103.63 Y132.924 E.012
G1 X103.63 Y141.757 E.33724
G1 X103.592 Y141.997 E.00928
G1 X103.438 Y142.269 E.01195
G1 X103.125 Y142.486 E.01454
G3 X102.41 Y142.467 I-.325 J-1.207 E.02768
G1 X100.837 Y142.175 E.06109
G1 X100.444 Y142.103 E.01527
G1 F13396.694
G1 X100.434 Y142.473 E.01414
; LINE_WIDTH: 0.66713
G1 F12101.026
G1 X100.281 Y142.588 E.0079
; LINE_WIDTH: 0.714263
G1 F11457.365
G1 X100.128 Y142.704 E.00848
; LINE_WIDTH: 0.761396
G1 F10831.294
G1 X99.976 Y142.819 E.00907
; LINE_WIDTH: 0.762386
G1 F10816.565
G1 X99.689 Y142.852 E.0137
G3 X99.627 Y143.543 I-3.737 J.013 E.03299
G1 X100.077 Y143.567 E.02141
; LINE_WIDTH: 0.714923
G1 F11570.901
G1 X100.528 Y143.591 E.02001
; LINE_WIDTH: 0.66746
G1 F12438.337
G1 X100.978 Y143.615 E.01862
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X101.378 Y143.615 E.01527
G1 X156.235 Y143.615 E2.09441
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.704 Y131.953 I29.094 J-43.619 E.60781
G3 X136.912 Y119.083 I40.816 J-33.508 E.57639
G1 X136.093 Y119.382 E.03327
G3 X134.379 Y119.594 I-1.772 J-7.297 E.06607
G3 X133.435 Y119.488 I1.141 J-14.435 E.03627
G1 X132.72 Y119.3 E.02825
G1 X131.944 Y118.975 E.03212
G1 X131.216 Y118.537 E.03242
G1 X130.563 Y117.998 E.03234
G1 X129.994 Y117.369 E.03237
G3 X129.081 Y116.142 I159.366 J-119.639 E.05839
G1 X126.698 Y112.93 E.15272
G3 X120.571 Y119.575 I-42.82 J-33.334 E.34551
G1 X120.478 Y119.655 E.00467
G1 X121.693 Y121.104 E.07222
G1 X121.855 Y121.433 E.014
G1 X121.86 Y121.752 E.01217
G1 X121.746 Y122.033 E.01161
G1 X121.716 Y122.067 E.00171
M204 S250
G1 X121.241 Y121.777 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X119.042 Y123.62 E.09082
G1 X118.872 Y123.672 E.00564
G1 X118.789 Y123.627 E.00299
G3 X118.686 Y123.546 I.05 J-.169 E.00424
G1 X118.518 Y123.346 E.00828
G1 X118.35 Y123.145 E.00828
G1 X118.182 Y122.945 E.00828
G1 F3450
G1 X118.014 Y122.744 E.00828
G1 F3300
G1 X117.846 Y122.544 E.00828
G1 F3150
G1 X117.678 Y122.343 E.00828
G1 F3600
G1 X117.559 Y122.202 E.00584
; LINE_WIDTH: 0.520326
G1 X117.536 Y122.175 E.00112
; LINE_WIDTH: 0.520776
G1 X117.503 Y122.137 E.0016
; LINE_WIDTH: 0.521226
G1 X117.471 Y122.098 E.0016
; LINE_WIDTH: 0.521686
G1 X117.438 Y122.06 E.0016
; LINE_WIDTH: 0.522136
G1 X117.405 Y122.022 E.0016
; LINE_WIDTH: 0.522596
G1 X117.372 Y121.984 E.0016
; LINE_WIDTH: 0.523046
G1 X117.34 Y121.945 E.0016
; LINE_WIDTH: 0.523196
G1 X117.329 Y121.933 E.00054
; LINE_WIDTH: 0.531656
G1 X117.322 Y121.809 E.00402
; LINE_WIDTH: 0.544336
G1 X117.311 Y121.623 E.00619
G1 X116.671 Y122.124 E.02702
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-32.253 J-42.072 E.63216
G1 X98.974 Y131.728 E.0219
; LINE_WIDTH: 0.561944
G1 X99.015 Y131.884 E.00553
; LINE_WIDTH: 0.603892
G1 X99.055 Y132.04 E.00597
; LINE_WIDTH: 0.64584
G1 X99.095 Y132.196 E.00641
; LINE_WIDTH: 0.687788
G1 X99.136 Y132.352 E.00685
; LINE_WIDTH: 0.729736
G1 X99.176 Y132.507 E.00729
; LINE_WIDTH: 0.744876
G1 X99.188 Y133.169 E.03063
G1 X99.992 Y133.02 E.03789
G1 X99.905 Y132.501 E.02433
; LINE_WIDTH: 0.729656
G1 X99.941 Y132.455 E.00264
; LINE_WIDTH: 0.684419
G1 X99.977 Y132.409 E.00247
; LINE_WIDTH: 0.639181
G1 X100.012 Y132.363 E.0023
; LINE_WIDTH: 0.593944
G1 X100.048 Y132.317 E.00213
; LINE_WIDTH: 0.548706
G1 X100.214 Y132.244 E.00608
; LINE_WIDTH: 0.519996
G2 X102.666 Y132.009 I-18.345 J-204.457 E.07799
G1 X102.79 Y132.032 E.004
G1 X102.91 Y132.196 E.00644
G1 X102.811 Y132.612 E.01354
G1 X103.001 Y132.665 E.00626
G1 X103.078 Y132.834 E.00588
G1 X103.078 Y141.757 E.2825
G1 X103.022 Y141.905 E.00502
G1 X102.852 Y141.983 E.00591
G1 X102.811 Y141.979 E.00131
G1 X99.686 Y141.4 E.10062
G1 X100.012 Y141.851 E.01761
G1 X100.004 Y142.125 E.00869
G1 X99.866 Y142.206 E.00507
G1 X99.582 Y142.234 E.00902
G1 X99.465 Y142.16 E.00439
G1 X99.077 Y141.694 E.01919
G3 X98.942 Y143.858 I-9.885 J.468 E.06878
G1 X98.937 Y144.167 E.0098
G1 X156.695 Y144.167 E1.82862
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.65 J-43.538 E.51061
G3 X137.192 Y118.315 I40.309 J-33.095 E.49167
G1 X136.613 Y118.616 E.02066
G1 X135.913 Y118.859 E.02345
G1 X135.137 Y119.01 E.02501
G1 X134.369 Y119.041 E.02435
G1 X133.609 Y118.959 E.02421
G1 X132.867 Y118.767 E.02425
G1 X132.161 Y118.467 E.02428
G1 X131.506 Y118.066 E.02432
G1 X130.918 Y117.575 E.02424
G1 X130.409 Y117.003 E.02428
G3 X129.087 Y115.222 I744.128 J-553.819 E.0702
G1 X126.704 Y112.01 E.12664
G3 X120.192 Y119.172 I-42.602 J-32.191 E.30689
; LINE_WIDTH: 0.544336
G1 X119.542 Y119.75 E.02893
G1 X119.736 Y119.777 E.00652
; LINE_WIDTH: 0.531156
G1 X119.877 Y119.796 E.00461
; LINE_WIDTH: 0.521596
G1 X119.887 Y119.809 E.00051
; LINE_WIDTH: 0.521526
G1 X119.924 Y119.853 E.00185
; LINE_WIDTH: 0.521296
G1 X119.962 Y119.898 E.00184
; LINE_WIDTH: 0.521066
G1 X119.999 Y119.943 E.00184
; LINE_WIDTH: 0.520836
G1 X120.036 Y119.987 E.00184
; LINE_WIDTH: 0.520616
G1 X120.073 Y120.032 E.00184
; LINE_WIDTH: 0.520386
G1 X120.11 Y120.077 E.00184
; LINE_WIDTH: 0.520156
G1 X120.136 Y120.108 E.0013
; LINE_WIDTH: 0.519996
G1 X120.251 Y120.246 E.00567
G1 F3150
G1 X120.415 Y120.44 E.00805
G1 F3300
G1 X120.578 Y120.635 E.00805
G1 F3450
G1 X120.741 Y120.83 E.00805
G1 F3600
G1 X120.905 Y121.025 E.00805
G1 X121.068 Y121.22 E.00805
G1 X121.231 Y121.415 E.00805
G3 X121.292 Y121.53 I-.112 J.132 E.00422
G1 X121.321 Y121.621 E.00304
G1 X121.282 Y121.697 E.00271
; WIPE_START
M204 S10000
G1 X120.523 Y122.349 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X119.542 Y119.75 Z7.48 F36000
G1 Z7.08
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X117.311 Y121.623 E.09682
; WIPE_START
M204 S10000
G1 X118.077 Y120.98 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X111.491 Y124.837 Z7.48 F36000
G1 X100.144 Y131.481 Z7.48
G1 Z7.08
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.05474
G1 F7717.562
G1 X99.735 Y131.607 E.02846
; WIPE_START
G1 X100.144 Y131.481 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X100.699 Y139.093 Z7.48 F36000
G1 X100.978 Y142.914 Z7.48
G1 Z7.08
G1 E.4 F1800
; LINE_WIDTH: 0.849736
G1 F9657.838
G1 X101.25 Y142.939 E.01452
; LINE_WIDTH: 0.799793
M73 P73 R5
G1 F10287.977
G1 X101.522 Y142.964 E.01363
; LINE_WIDTH: 0.74985
G1 F11006.085
G1 X101.794 Y142.989 E.01274
; LINE_WIDTH: 0.699906
G1 F11831.965
G1 X102.066 Y143.014 E.01185
; LINE_WIDTH: 0.649963
G1 F12791.844
G1 X102.338 Y143.039 E.01096
; LINE_WIDTH: 0.60002
G1 F13921.217
G1 X102.61 Y143.064 E.01007
; LINE_WIDTH: 0.550076
G1 F15269.323
G1 X103.081 Y143.065 E.01585
; WIPE_START
G1 X102.61 Y143.064 E-.17917
G1 X102.338 Y143.039 E-.10378
G1 X102.084 Y143.016 E-.09704
; WIPE_END
G1 E-.02 F1800
G1 X105.985 Y137.65 Z7.48 F36000
G1 Z7.08
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X105.656 Y136.343 I-2.808 J.011 E.05195
G2 X105.958 Y135.41 I-3.229 J-1.559 E.03754
G2 X106.899 Y130.506 I-2.822 J-3.084 E.20434
G2 X112.804 Y127.433 I-24.893 J-55.044 E.25426
G3 X114.488 Y129.416 I-6.866 J7.537 E.0996
G3 X114.143 Y133.658 I-4.344 J1.782 E.16862
G3 X112.443 Y135.543 I-8.221 J-5.703 E.09717
G1 X111.617 Y136.485 E.04785
G2 X111.683 Y141.198 I3.808 J2.304 E.18964
G1 X112.27 Y141.945 E.03629
G1 X119.042 Y141.945 E.25855
G3 X119.223 Y137.428 I3.936 J-2.105 E.181
G3 X120.923 Y135.543 I8.22 J5.702 E.09717
G1 X121.749 Y134.6 E.04785
G2 X121.684 Y129.887 I-3.808 J-2.304 E.18964
G2 X119.984 Y128.002 I-8.221 J5.703 E.09717
G1 X119.158 Y127.06 E.04786
G3 X118.641 Y125.871 I2.651 J-1.859 E.04982
G2 X120.335 Y125.429 I.286 J-2.374 E.06844
G1 X122.777 Y123.384 E.12162
G2 X122.771 Y119.793 I-1.828 J-1.792 E.15194
G2 X126.637 Y115.65 I-55.283 J-55.461 E.2164
G1 X128.616 Y118.318 E.12683
G1 X129.042 Y118.808 E.02479
G3 X127.525 Y120.462 I-7.225 J-5.107 E.08592
G1 X126.699 Y121.404 E.04785
G2 X126.764 Y126.117 I3.808 J2.304 E.18964
G2 X128.463 Y128.002 I8.221 J-5.703 E.09717
G1 X129.29 Y128.945 E.04786
G3 X129.224 Y133.658 I-3.808 J2.304 E.18964
G3 X127.525 Y135.543 I-8.22 J-5.702 E.09717
G1 X126.699 Y136.485 E.04785
G2 X126.764 Y141.198 I3.808 J2.304 E.18964
G1 X127.352 Y141.945 E.03629
G1 X134.123 Y141.945 E.25855
G3 X134.305 Y137.428 I3.936 J-2.105 E.181
G3 X136.004 Y135.543 I8.221 J5.703 E.09717
G1 X136.83 Y134.6 E.04785
G2 X136.765 Y129.887 I-3.808 J-2.304 E.18964
G2 X135.065 Y128.002 I-8.221 J5.703 E.09717
G1 X134.239 Y127.06 E.04785
G3 X134.305 Y122.347 I3.808 J-2.304 E.18964
G3 X135.222 Y121.231 I4.822 J3.029 E.05531
G2 X135.984 Y121.112 I-.421 J-5.216 E.02946
G2 X144.131 Y133.88 I50.701 J-23.369 E.58003
G3 X142.606 Y135.543 I-7.265 J-5.13 E.08638
G1 X141.78 Y136.485 E.04785
G2 X141.845 Y141.198 I3.808 J2.304 E.18964
G1 X142.433 Y141.945 E.03629
G1 X149.205 Y141.945 E.25855
G3 X148.832 Y138.795 I3.696 J-2.034 E.12416
G2 X150.586 Y140.348 I52.788 J-57.857 E.08944
G1 X105.156 Y131.461 F36000
; FEATURE: Floating vertical shell
; LINE_WIDTH: 0.784911
G1 F10491.956
G1 X105.244 Y131.77 E.01571
; LINE_WIDTH: 0.752821
G1 F10960.563
G1 X105.333 Y132.079 E.01504
; LINE_WIDTH: 0.71934
G1 F11496.295
G1 X105.352 Y132.314 E.01055
; LINE_WIDTH: 0.684466
G1 F12000
G1 X105.371 Y132.549 E.01001
; LINE_WIDTH: 0.634534
G3 X105.392 Y132.828 I-3.309 J.386 E.01095
G3 X105.392 Y135.199 I-127.355 J1.173 E.09277
; LINE_WIDTH: 0.607591
G1 X105.312 Y135.722 E.01976
; LINE_WIDTH: 0.552955
G1 X105.232 Y136.245 E.01788
G1 X105.232 Y136.441 E.00663
G1 X105.339 Y137 E.01923
; LINE_WIDTH: 0.565821
G1 X105.366 Y137.251 E.00878
; LINE_WIDTH: 0.608791
G1 X105.392 Y137.503 E.00948
; LINE_WIDTH: 0.636129
G3 X105.395 Y141.755 I-322.618 J2.319 E.1668
; CHANGE_LAYER
; Z_HEIGHT: 7.24
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F12000
G1 X105.394 Y140.755 E-.38
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
G1 X122.428 Y122.978
G1 Z7.24
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X122.26 Y123.17 E.00976
G3 X121.805 Y123.554 I-2.571 J-2.583 E.02274
G1 X120.273 Y124.839 E.07636
G1 X119.81 Y125.127 E.02081
G1 X119.357 Y125.265 E.01808
G1 X118.808 Y125.283 E.02096
G3 X117.804 Y124.869 I.321 J-2.203 E.04188
G3 X117.034 Y124.011 I4.972 J-5.236 E.04408
G3 X104.788 Y130.857 I-32.53 J-43.809 E.53716
G1 X105.24 Y131.481 E.02942
G1 X105.361 Y132.076 E.02315
G1 X105.347 Y132.611 E.02045
G1 X105.241 Y133.039 E.01684
G1 X105.121 Y133.301 E.01099
G1 X104.804 Y133.696 E.01936
G1 X104.804 Y138.084 E.16753
G1 X105.111 Y138.487 E.01935
G1 X105.292 Y138.922 E.01799
G1 X105.363 Y139.446 E.02018
G1 X105.363 Y140.345 E.03432
G1 X105.327 Y140.718 E.01431
G1 X105.193 Y141.142 E.01698
G1 X104.918 Y141.585 E.0199
G1 X104.804 Y141.683 E.00577
G1 X104.8 Y141.885 E.00769
G1 X104.638 Y142.443 E.02219
G1 X154.07 Y142.443 E1.88724
G3 X143.8 Y132.698 I31.861 J-43.858 E.54206
G3 X136.269 Y120.54 I42.479 J-34.727 E.54759
G3 X133.394 Y120.67 I-1.761 J-7.103 E.11059
G1 X132.428 Y120.434 E.03794
G1 X131.481 Y120.051 E.03903
G1 X130.604 Y119.536 E.03882
G1 X129.808 Y118.895 E.03902
G1 X129.131 Y118.16 E.03814
G3 X127.853 Y116.453 I82.008 J-62.701 E.08142
G1 X126.662 Y114.847 E.07636
G3 X121.973 Y119.865 I-44.537 J-36.913 E.26235
G1 X122.468 Y120.455 E.02939
G1 X122.749 Y120.901 E.02012
G1 X122.881 Y121.305 E.01623
G1 X122.919 Y121.834 E.02026
G1 X122.819 Y122.336 E.01954
G1 X122.612 Y122.763 E.01812
G1 X122.487 Y122.909 E.00735
G1 X121.976 Y122.61 F36000
G1 F13446.369
G1 X121.85 Y122.753 E.00727
G3 X121.429 Y123.105 I-2.25 J-2.257 E.02097
G1 X119.896 Y124.39 E.07636
G1 X119.573 Y124.592 E.01456
G1 X119.182 Y124.699 E.01547
G1 X118.79 Y124.689 E.01498
G1 X118.348 Y124.532 E.01791
G1 X117.975 Y124.221 E.01854
G1 X117.126 Y123.208 E.05047
G3 X103.318 Y130.801 I-32.851 J-43.386 E.60372
G1 X104.242 Y131.079 E.03682
G1 X104.691 Y131.686 E.02883
G1 X104.777 Y132.159 E.01834
G1 X104.735 Y132.697 E.0206
G1 X104.608 Y133.018 E.0132
G3 X104.218 Y133.493 I-10.832 J-8.509 E.02345
G1 X104.218 Y138.414 E.18788
G1 X104.374 Y138.478 E.00646
G1 X104.601 Y138.775 E.01427
G1 X104.727 Y139.08 E.01258
G1 X104.778 Y139.446 E.01412
G1 X104.778 Y140.345 E.03432
G3 X104.658 Y140.903 I-1.364 J0 E.02193
G1 X104.408 Y141.278 E.01723
G1 X104.218 Y141.305 E.00734
G1 X104.217 Y141.807 E.01919
G1 X104.052 Y142.411 E.0239
G1 X103.494 Y142.953 E.02969
; LINE_WIDTH: 0.599936
G1 F13923.269
G1 X103.289 Y143.009 E.00786
; LINE_WIDTH: 0.572626
G1 F14629.663
G1 X103.083 Y143.066 E.00748
; LINE_WIDTH: 0.582656
G1 F14362.053
G1 X103.352 Y143.059 E.0096
G1 X103.751 Y143.048 E.0143
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X105.749 Y143.029 I1.333 J35.741 E.07626
G1 X155.749 Y143.029 E1.90895
G1 X155.769 Y142.918 E.00431
G3 X144.252 Y132.326 I29.932 J-44.101 E.59956
G3 X136.598 Y119.835 I41.232 J-33.855 E.56108
G3 X133.414 Y120.081 I-2.093 J-6.333 E.12314
G1 X132.577 Y119.868 E.03297
G1 X131.711 Y119.513 E.03574
G1 X130.911 Y119.037 E.03554
G1 X130.185 Y118.446 E.03573
G1 X129.57 Y117.772 E.03485
G3 X129.066 Y117.105 I42.205 J-32.367 E.0319
G1 X126.683 Y113.893 E.15272
G3 X121.169 Y119.817 I-44.214 J-35.629 E.30926
G1 X122.019 Y120.831 E.05052
G1 X122.215 Y121.143 E.01408
G1 X122.333 Y121.595 E.01783
G1 X122.292 Y122.056 E.01766
G1 X122.12 Y122.446 E.01628
G1 X122.035 Y122.543 E.00491
G1 X121.535 Y122.235 F36000
G1 F13446.369
G1 X121.473 Y122.304 E.00353
G1 X119.52 Y123.941 E.09731
G1 X119.227 Y124.095 E.01263
G1 X118.889 Y124.112 E.01295
G1 X118.637 Y124.022 E.01022
G1 X118.361 Y123.77 E.01426
G1 X117.208 Y122.395 E.0685
G3 X102.165 Y130.593 I-32.691 J-42.086 E.65695
G1 X101.922 Y130.676 E.00978
G1 X101.543 Y130.803 E.01527
G1 F13348.516
G1 X101.164 Y130.93 E.01527
G1 F11955.018
G1 X100.784 Y131.057 E.01527
; LINE_WIDTH: 0.667354
G1 F10638.32
G1 X100.713 Y131.104 E.00352
; LINE_WIDTH: 0.714712
G1 F10367.288
G1 X100.642 Y131.151 E.00379
; LINE_WIDTH: 0.76207
G1 F10099.709
G1 X100.571 Y131.198 E.00405
; LINE_WIDTH: 0.809427
G1 F9835.646
G1 X100.5 Y131.245 E.00431
; LINE_WIDTH: 0.856785
G1 F9575.064
G1 X100.429 Y131.292 E.00458
; LINE_WIDTH: 0.904143
G1 F9053.737
G1 X100.357 Y131.34 E.00484
; LINE_WIDTH: 0.951501
G1 F8586.247
G1 X100.286 Y131.387 E.0051
; LINE_WIDTH: 0.998858
G1 F8164.665
G1 X100.215 Y131.434 E.00537
; LINE_WIDTH: 1.04622
G1 F7782.544
G1 X100.144 Y131.481 E.00563
G1 X100.228 Y131.497 E.00563
; LINE_WIDTH: 0.998858
G1 F8164.665
G1 X100.312 Y131.513 E.00537
; LINE_WIDTH: 0.951501
G1 F8586.247
G1 X100.395 Y131.529 E.0051
; LINE_WIDTH: 0.904143
G1 F9053.737
G1 X100.479 Y131.545 E.00484
; LINE_WIDTH: 0.856785
G1 F9575.064
G1 X100.563 Y131.562 E.00458
; LINE_WIDTH: 0.809427
G1 F10160.097
G1 X100.647 Y131.578 E.00431
; LINE_WIDTH: 0.76207
G1 F10428.476
G1 X100.731 Y131.594 E.00405
; LINE_WIDTH: 0.714712
G1 F10700.317
G1 X100.815 Y131.61 E.00379
; LINE_WIDTH: 0.667354
G1 F10975.661
G1 X100.898 Y131.626 E.00352
; LINE_WIDTH: 0.619996
G1 F12312.468
G1 X101.296 Y131.587 E.01527
G1 F13446.369
G1 X101.695 Y131.548 E.01527
G1 X103.344 Y131.387 E.06326
G1 X103.886 Y131.544 E.02155
G1 X104.148 Y131.906 E.01704
G1 X104.192 Y132.357 E.01732
G1 X104.143 Y132.631 E.01061
G3 X103.632 Y133.277 I-2.845 J-1.726 E.03151
G1 X103.632 Y138.758 E.20927
G1 X103.962 Y138.894 E.0136
G1 X104.136 Y139.158 E.01209
G1 X104.192 Y139.446 E.01121
G1 X104.192 Y140.345 E.03432
G1 X104.124 Y140.663 E.01242
G1 X103.981 Y140.878 E.00983
G1 X103.632 Y140.926 E.01344
G1 X103.632 Y141.759 E.03182
G1 X103.538 Y142.131 E.01465
G1 X103.222 Y142.445 E.01701
G3 X102.504 Y142.486 I-.417 J-1.002 E.028
G1 X100.93 Y142.195 E.06109
G1 X100.537 Y142.122 E.01527
G1 F13151.346
G1 X100.514 Y142.471 E.01337
; LINE_WIDTH: 0.669853
G1 F11936.606
G1 X100.366 Y142.583 E.0077
; LINE_WIDTH: 0.71971
G1 F11315.496
G1 X100.217 Y142.695 E.00831
; LINE_WIDTH: 0.769566
G1 F10710.935
G1 X100.069 Y142.807 E.00891
G1 X99.691 Y142.852 E.01826
; LINE_WIDTH: 0.762566
G1 F10813.892
G3 X99.628 Y143.543 I-3.694 J.012 E.03298
; LINE_WIDTH: 0.770596
G1 F10695.951
G3 X100.357 Y143.558 I.283 J4.038 E.03506
; LINE_WIDTH: 0.732946
G1 F11272.384
G1 X100.586 Y143.577 E.01044
; LINE_WIDTH: 0.695296
G1 F11914.49
G1 X100.814 Y143.596 E.00988
; LINE_WIDTH: 0.657646
G1 F12634.164
G1 X101.042 Y143.615 E.00931
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X101.442 Y143.615 E.01527
G1 X156.235 Y143.615 E2.09194
G1 X156.415 Y142.65 E.03745
G3 X144.704 Y131.953 I29.387 J-43.929 E.60783
G3 X136.908 Y119.075 I40.819 J-33.511 E.57672
G1 X136.5 Y119.253 E.01699
G1 X135.794 Y119.458 E.02808
G1 X134.957 Y119.581 E.03231
G1 X134.412 Y119.586 E.02077
G1 X133.538 Y119.508 E.03354
G1 X132.726 Y119.301 E.03195
G1 X131.942 Y118.974 E.03245
G1 X131.218 Y118.539 E.03225
G1 X130.562 Y117.998 E.03245
G1 X130.008 Y117.384 E.03156
G3 X129.081 Y116.142 I102.569 J-77.531 E.05918
G1 X126.698 Y112.93 E.15272
G3 X120.571 Y119.575 I-42.824 J-33.338 E.34551
G1 X120.355 Y119.758 E.01078
G1 X121.57 Y121.207 E.07222
G1 X121.735 Y121.546 E.01441
G1 X121.739 Y121.847 E.01149
G1 X121.628 Y122.129 E.01156
G1 X121.594 Y122.167 E.00194
M204 S250
G1 X121.118 Y121.88 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X119.165 Y123.517 E.08069
G1 X118.996 Y123.569 E.00561
G1 X118.912 Y123.524 E.00301
G3 X118.812 Y123.447 I.045 J-.163 E.00408
G1 X118.659 Y123.265 E.00755
G1 X118.505 Y123.082 E.00755
G1 X118.352 Y122.899 E.00755
G1 F3450
G1 X118.199 Y122.716 E.00755
G1 F3300
G1 X118.046 Y122.533 E.00755
G1 F3150
G1 X117.892 Y122.351 E.00755
G1 F3600
G1 X117.784 Y122.222 E.00532
; LINE_WIDTH: 0.520326
G1 X117.751 Y122.182 E.00163
; LINE_WIDTH: 0.520776
G1 X117.704 Y122.127 E.00232
; LINE_WIDTH: 0.521226
G1 X117.656 Y122.071 E.00232
; LINE_WIDTH: 0.521686
G1 X117.609 Y122.015 E.00232
; LINE_WIDTH: 0.522136
G1 X117.562 Y121.96 E.00232
; LINE_WIDTH: 0.522596
G1 X117.514 Y121.904 E.00233
; LINE_WIDTH: 0.523046
G1 X117.467 Y121.848 E.00233
; LINE_WIDTH: 0.523196
G1 X117.451 Y121.83 E.00078
; LINE_WIDTH: 0.531656
G1 X117.444 Y121.706 E.00402
; LINE_WIDTH: 0.544336
G1 X117.434 Y121.52 E.00619
G1 X116.671 Y122.124 E.03233
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-32.253 J-42.071 E.63217
G1 X98.944 Y131.535 E.01575
; LINE_WIDTH: 0.548436
G1 X99.035 Y132.014 E.01635
; LINE_WIDTH: 0.593524
G1 X99.071 Y132.137 E.00467
; LINE_WIDTH: 0.638611
G1 X99.106 Y132.261 E.00505
; LINE_WIDTH: 0.683699
G1 X99.142 Y132.384 E.00543
; LINE_WIDTH: 0.728786
G1 X99.177 Y132.507 E.0058
; LINE_WIDTH: 0.743756
G1 X99.19 Y133.167 E.03052
G1 X99.993 Y133.018 E.03777
G1 X99.906 Y132.501 E.02422
; LINE_WIDTH: 0.728646
G1 X99.942 Y132.455 E.00264
; LINE_WIDTH: 0.683594
G1 X99.977 Y132.409 E.00247
; LINE_WIDTH: 0.638541
G1 X100.013 Y132.363 E.0023
; LINE_WIDTH: 0.593489
G1 X100.048 Y132.317 E.00212
; LINE_WIDTH: 0.548436
G1 X100.214 Y132.244 E.00606
; LINE_WIDTH: 0.519996
G2 X101.407 Y132.132 I-8.788 J-100.068 E.03794
G1 X103.397 Y131.938 E.06332
G1 X103.554 Y131.986 E.00519
G1 X103.639 Y132.162 E.00618
G1 X103.611 Y132.468 E.00974
G1 X103.08 Y133.073 E.02548
G1 X103.08 Y139.22 E.19464
G1 X103.413 Y139.22 E.01057
G1 X103.61 Y139.335 E.0072
G1 X103.639 Y139.446 E.00363
G1 X103.639 Y140.345 E.02846
G1 X103.578 Y140.499 E.00526
G1 X103.297 Y140.538 E.00898
G2 X103.08 Y140.478 I-.183 J.238 E.00732
G1 X103.08 Y141.759 E.04056
G1 X103.052 Y141.867 E.00352
G1 X102.893 Y141.982 E.00621
G1 X102.813 Y141.981 E.00254
G1 X99.843 Y141.431 E.09562
G1 X100.111 Y141.844 E.01558
G1 X100.093 Y142.113 E.00854
G3 X99.583 Y142.234 I-.462 J-.812 E.01684
G1 X99.465 Y142.16 E.00441
G1 X99.08 Y141.696 E.01908
G3 X98.942 Y143.858 I-9.744 J.466 E.06871
G1 X98.937 Y144.167 E.0098
G1 X156.695 Y144.167 E1.82862
G1 X156.959 Y142.745 E.0458
G1 X157.025 Y142.391 E.01141
G3 X145.128 Y131.598 I28.628 J-43.511 E.5106
G3 X137.192 Y118.316 I40.595 J-33.266 E.49164
G3 X136.346 Y118.722 I-5.216 J-9.781 E.02973
G1 X135.64 Y118.927 E.02328
G3 X133.605 Y118.959 I-1.107 J-5.775 E.06474
G1 X132.867 Y118.767 E.02415
G1 X132.16 Y118.466 E.02434
G1 X131.508 Y118.068 E.02418
G1 X130.918 Y117.575 E.02434
G1 X130.422 Y117.018 E.0236
G3 X129.087 Y115.222 I249.124 J-186.644 E.07085
G1 X126.704 Y112.01 E.12664
G3 X120.192 Y119.172 I-42.607 J-32.195 E.30689
; LINE_WIDTH: 0.521596
G1 X119.788 Y119.513 E.01678
; LINE_WIDTH: 0.544336
G1 X119.419 Y119.853 E.01669
G1 X119.613 Y119.88 E.00652
; LINE_WIDTH: 0.531156
G1 X119.754 Y119.899 E.00461
; LINE_WIDTH: 0.521596
G1 X119.767 Y119.915 E.00064
; LINE_WIDTH: 0.521526
G1 X119.815 Y119.972 E.00235
; LINE_WIDTH: 0.521296
G1 X119.862 Y120.028 E.00235
; LINE_WIDTH: 0.521066
G1 X119.909 Y120.085 E.00234
; LINE_WIDTH: 0.520836
G1 X119.957 Y120.142 E.00234
; LINE_WIDTH: 0.520616
G1 X120.004 Y120.199 E.00234
; LINE_WIDTH: 0.520386
G1 X120.051 Y120.255 E.00234
; LINE_WIDTH: 0.520156
G1 X120.084 Y120.295 E.00165
; LINE_WIDTH: 0.519996
G1 X120.192 Y120.424 E.00532
G1 F3150
G1 X120.345 Y120.607 E.00755
G1 F3300
G1 X120.499 Y120.79 E.00755
G1 F3450
G1 X120.652 Y120.972 E.00755
G1 F3600
G1 X120.805 Y121.155 E.00755
G1 X120.958 Y121.337 E.00755
G1 X121.111 Y121.52 E.00755
G3 X121.169 Y121.632 I-.11 J.128 E.00408
G1 X121.199 Y121.722 E.00301
G1 X121.159 Y121.8 E.00277
; WIPE_START
M204 S10000
G1 X120.401 Y122.453 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X119.419 Y119.853 Z7.64 F36000
G1 Z7.24
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X117.434 Y121.52 E.08619
; WIPE_START
M204 S10000
G1 X118.2 Y120.877 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X111.618 Y124.742 Z7.64 F36000
G1 X100.144 Y131.481 Z7.64
G1 Z7.24
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.05462
G1 F7718.469
G1 X99.735 Y131.607 E.02842
; WIPE_START
G1 X100.144 Y131.481 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X100.742 Y139.09 Z7.64 F36000
G1 X101.042 Y142.921 Z7.64
G1 Z7.24
G1 E.4 F1800
; LINE_WIDTH: 0.835776
G1 F9826.064
G1 X101.304 Y142.945 E.01373
; LINE_WIDTH: 0.78774
G1 F10452.569
G1 X101.566 Y142.969 E.0129
; LINE_WIDTH: 0.739703
G1 F11164.408
G1 X101.827 Y142.993 E.01208
; LINE_WIDTH: 0.691666
G1 F11980.285
G1 X102.089 Y143.017 E.01126
; LINE_WIDTH: 0.64363
G1 F12924.811
G1 X102.351 Y143.041 E.01043
; LINE_WIDTH: 0.595593
G1 F14031.015
G1 X102.612 Y143.065 E.00961
; LINE_WIDTH: 0.547556
G1 F15344.298
G1 X103.083 Y143.066 E.01576
; WIPE_START
G1 X102.612 Y143.065 E-.179
G1 X102.351 Y143.041 E-.09983
G1 X102.089 Y143.017 E-.09983
G1 X102.085 Y143.017 E-.00135
; WIPE_END
G1 E-.02 F1800
G1 X105.511 Y138.195 Z7.64 F36000
G1 Z7.24
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.302 Y137.921 E.01319
G1 X105.302 Y135.923 E.07625
G2 X106.517 Y134.6 I-13.321 J-13.455 E.06862
G2 X106.94 Y130.486 I-3.673 J-2.457 E.16432
G2 X112.71 Y127.487 I-30.826 J-66.37 E.24836
G1 X113.216 Y128.002 E.02754
G3 X114.573 Y129.887 I-3.381 J3.865 E.08946
G3 X113.535 Y134.6 I-4.576 J1.463 E.19303
G2 X111.768 Y136.485 I18.967 J19.551 E.09869
G2 X112.147 Y141.945 I3.692 J2.487 E.22505
G1 X119.202 Y141.945 E.26934
G1 X118.793 Y141.198 E.03252
G3 X119.831 Y136.485 I4.576 J-1.463 E.19303
G2 X121.598 Y134.6 I-18.967 J-19.551 E.09869
G2 X122.168 Y130.83 I-3.642 J-2.479 E.15056
G2 X120.15 Y128.002 I-6.521 J2.52 E.13404
G3 X118.793 Y126.117 I3.381 J-3.866 E.08946
G1 X118.71 Y125.767 E.01375
G2 X120.838 Y125.007 I.271 J-2.602 E.08922
G1 X124.886 Y141.945 F36000
G1 F13446.283
G1 X127.229 Y141.945 E.08944
G3 X126.334 Y137.428 I3.596 J-3.06 E.18367
G3 X127.691 Y135.543 I4.738 J1.98 E.08946
G2 X129.306 Y133.658 I-5.658 J-6.482 E.09511
G2 X129.139 Y128.945 I-3.958 J-2.219 E.18933
G2 X127.372 Y127.06 I-20.729 J17.661 E.09869
G3 X126.334 Y122.347 I3.538 J-3.25 E.19303
G3 X127.691 Y120.462 I4.738 J1.98 E.08946
G2 X129.094 Y118.867 I-4.768 J-5.607 E.08136
G2 X135.071 Y121.242 I5.484 J-5.092 E.25384
G2 X133.874 Y126.117 I3.293 J3.393 E.20201
G2 X135.232 Y128.002 I4.738 J-1.98 E.08946
G3 X136.847 Y129.887 I-5.658 J6.482 E.09511
G3 X136.679 Y134.6 I-3.958 J2.219 E.18933
G3 X134.912 Y136.485 I-20.734 J-17.665 E.09869
G2 X133.82 Y138.371 I3.524 J3.301 E.08391
G2 X134.283 Y141.945 I4.151 J1.279 E.14192
G1 X142.31 Y141.945 E.30645
G3 X141.415 Y137.428 I3.596 J-3.06 E.18367
G3 X142.772 Y135.543 I4.738 J1.98 E.08946
G2 X144.182 Y133.939 I-4.797 J-5.636 E.08181
G2 X148.822 Y138.787 I50.175 J-43.377 E.25631
G2 X149.364 Y141.945 I3.735 J.984 E.12618
G1 X151.707 Y141.945 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 7.4
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X150.707 Y141.945 E-.38
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
G1 X122.264 Y123.14
G1 Z7.4
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X122.103 Y123.304 E.00878
G3 X121.928 Y123.451 I-1.655 J-1.797 E.00873
G1 X120.395 Y124.736 E.07636
G1 X119.936 Y125.023 E.02068
G1 X119.48 Y125.162 E.01821
G1 X118.941 Y125.181 E.02058
G3 X117.993 Y124.817 I.308 J-2.216 E.03911
G1 X117.648 Y124.494 E.01801
G1 X117.162 Y123.914 E.02891
G3 X104.787 Y130.858 I-32.831 J-44.013 E.54329
G1 X105.24 Y131.481 E.02943
G1 X105.361 Y132.076 E.02316
G1 X105.365 Y133.029 E.03638
G1 X105.306 Y133.571 E.02084
G1 X105.086 Y134.1 E.02186
G1 X104.806 Y134.413 E.01604
G1 X104.806 Y137.512 E.11831
G1 X105.11 Y137.907 E.01904
G1 X105.305 Y138.39 E.0199
G3 X105.365 Y139.512 I-4.436 J.802 E.04299
G1 X105.365 Y141.512 E.07636
G1 X105.249 Y142.175 E.02572
G1 X105.075 Y142.443 E.01221
G1 X154.069 Y142.443 E1.87054
G3 X143.801 Y132.699 I31.509 J-43.484 E.54204
G3 X136.271 Y120.545 I42.386 J-34.67 E.54742
G1 X135.454 Y120.705 E.03179
G3 X133.322 Y120.66 I-.893 J-8.169 E.08165
G1 X132.429 Y120.434 E.03515
G1 X131.483 Y120.052 E.03895
G1 X130.602 Y119.535 E.03901
G1 X129.81 Y118.896 E.03885
G1 X129.206 Y118.245 E.03391
G3 X127.853 Y116.453 I40.066 J-31.648 E.08573
G1 X126.662 Y114.847 E.07636
G3 X121.856 Y119.974 I-41.499 J-34.075 E.26849
G1 X122.353 Y120.567 E.02953
G1 X122.647 Y121.052 E.02164
G1 X122.793 Y121.634 E.02293
G1 X122.772 Y122.142 E.0194
G1 X122.635 Y122.596 E.01811
G3 X122.361 Y123.027 I-2.187 J-1.089 E.01954
G1 X122.323 Y123.072 E.00223
G1 X121.855 Y122.709 F36000
G1 F13446.369
G1 X121.727 Y122.856 E.00744
G3 X121.552 Y123.002 I-1.433 J-1.532 E.00873
G1 X120.019 Y124.287 E.07636
G1 X119.698 Y124.488 E.01447
G1 X119.31 Y124.596 E.01537
G1 X118.943 Y124.591 E.01399
G1 X118.518 Y124.455 E.01705
G1 X118.121 Y124.145 E.01924
G3 X117.253 Y123.111 I652.343 J-548.07 E.05152
G3 X103.318 Y130.801 I-32.968 J-43.274 E.60984
G1 X104.239 Y131.077 E.03669
G1 X104.691 Y131.686 E.02896
G1 X104.777 Y132.159 E.01835
G1 X104.78 Y133.104 E.0361
G1 X104.697 Y133.562 E.01775
M73 P74 R5
G1 X104.584 Y133.797 E.00997
G1 X104.247 Y134.175 E.01932
G1 X104.22 Y134.229 E.00232
G1 X104.22 Y137.837 E.13776
G1 X104.373 Y137.9 E.00629
G1 X104.601 Y138.197 E.0143
G1 X104.737 Y138.535 E.01392
G1 X104.78 Y138.872 E.01297
G1 X104.78 Y141.512 E.10077
G1 X104.698 Y141.976 E.01799
G1 X104.362 Y142.494 E.0236
G1 X103.698 Y142.846 E.02867
G1 X103.392 Y142.957 E.01244
; LINE_WIDTH: 0.581396
G1 F14395.13
G1 X103.085 Y143.068 E.01162
G1 X103.401 Y143.048 E.0113
; LINE_WIDTH: 0.619996
G1 F13446.369
G3 X103.749 Y143.029 I.332 J2.834 E.01329
G1 X155.749 Y143.029 E1.98531
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00187
G3 X144.252 Y132.326 I30.059 J-44.231 E.59958
G3 X136.596 Y119.828 I41.792 J-34.198 E.56132
G1 X136.065 Y119.995 E.02126
G1 X135.346 Y120.129 E.0279
G1 X134.411 Y120.179 E.03575
G3 X133.287 Y120.054 I.625 J-10.802 E.04319
G1 X132.565 Y119.864 E.02853
G1 X131.714 Y119.514 E.03514
G1 X130.909 Y119.036 E.03573
G1 X130.187 Y118.448 E.03557
G3 X129.066 Y117.106 I10.336 J-9.767 E.0668
G1 X126.683 Y113.893 E.15272
G3 X121.052 Y119.927 I-42.274 J-33.81 E.31541
G1 X121.902 Y120.941 E.05052
G1 X122.117 Y121.303 E.01608
G1 X122.209 Y121.687 E.01509
G1 X122.173 Y122.146 E.01755
G1 X122.005 Y122.538 E.01629
G1 X121.914 Y122.641 E.00524
G1 X121.383 Y122.365 F36000
G1 F13446.369
G1 X121.176 Y122.554 E.01071
G1 X119.643 Y123.838 E.07636
G1 X119.352 Y123.992 E.01255
G1 X119.029 Y124.012 E.01236
G1 X118.786 Y123.934 E.00973
G1 X118.546 Y123.742 E.01174
G1 X117.331 Y122.292 E.07222
G1 X116.334 Y123.072 E.04834
G3 X102.165 Y130.593 I-31.939 J-43.064 E.61475
G1 X101.922 Y130.676 E.00978
G1 X101.543 Y130.803 E.01527
G1 F13348.486
G1 X101.164 Y130.93 E.01527
G1 F11954.99
G1 X100.784 Y131.057 E.01527
; LINE_WIDTH: 0.667354
G1 F10638.294
G1 X100.713 Y131.104 E.00352
; LINE_WIDTH: 0.714712
G1 F10367.261
G1 X100.642 Y131.151 E.00379
; LINE_WIDTH: 0.76207
G1 F10099.683
G1 X100.571 Y131.198 E.00405
; LINE_WIDTH: 0.809427
G1 F9835.62
G1 X100.5 Y131.245 E.00431
; LINE_WIDTH: 0.856785
G1 F9575.064
G1 X100.429 Y131.292 E.00458
; LINE_WIDTH: 0.904143
G1 F9053.737
G1 X100.357 Y131.34 E.00484
; LINE_WIDTH: 0.951501
G1 F8586.247
G1 X100.286 Y131.387 E.0051
; LINE_WIDTH: 0.998858
G1 F8164.665
G1 X100.215 Y131.434 E.00537
; LINE_WIDTH: 1.04622
G1 F7782.544
G1 X100.144 Y131.481 E.00563
G1 X100.228 Y131.497 E.00563
; LINE_WIDTH: 0.998858
G1 F8164.665
G1 X100.312 Y131.513 E.00537
; LINE_WIDTH: 0.951501
G1 F8586.247
G1 X100.395 Y131.529 E.0051
; LINE_WIDTH: 0.904143
G1 F9053.737
G1 X100.479 Y131.545 E.00484
; LINE_WIDTH: 0.856785
G1 F9575.064
G1 X100.563 Y131.562 E.00458
; LINE_WIDTH: 0.809427
G1 F10160.097
G1 X100.647 Y131.578 E.00431
; LINE_WIDTH: 0.76207
G1 F10428.445
G1 X100.731 Y131.594 E.00405
; LINE_WIDTH: 0.714712
G1 F10700.285
G1 X100.815 Y131.61 E.00379
; LINE_WIDTH: 0.667354
G1 F10975.629
G1 X100.898 Y131.626 E.00352
; LINE_WIDTH: 0.619996
G1 F12312.434
G1 X101.296 Y131.587 E.01527
G1 F13446.369
G1 X101.695 Y131.548 E.01527
G1 X103.344 Y131.387 E.06326
G1 X103.884 Y131.543 E.02148
G1 X104.148 Y131.906 E.01711
G1 X104.191 Y132.16 E.00987
G1 X104.194 Y133.1 E.03585
G1 X104.138 Y133.384 E.01106
G1 X103.89 Y133.71 E.01565
G1 X103.634 Y133.907 E.01231
G1 X103.634 Y138.183 E.16327
G1 X103.962 Y138.317 E.0135
G1 X104.147 Y138.605 E.01305
G1 X104.194 Y138.872 E.01037
G1 X104.194 Y141.512 E.10077
G1 X104.148 Y141.776 E.01027
G1 X103.956 Y142.072 E.01347
G1 X103.577 Y142.273 E.01636
G1 X103.403 Y142.316 E.00684
G1 X103.316 Y142.39 E.00434
G1 X102.991 Y142.529 E.01352
G3 X102.592 Y142.505 I-.133 J-1.117 E.01533
G1 X101.019 Y142.213 E.06109
G1 X100.625 Y142.141 E.01527
G1 F13083.972
G1 X100.583 Y142.47 E.0127
; LINE_WIDTH: 0.659449
G1 F11931.898
G1 X100.474 Y142.552 E.00554
; LINE_WIDTH: 0.698901
G1 F11476.479
G1 X100.366 Y142.634 E.00589
; LINE_WIDTH: 0.738354
G1 F11029.921
G1 X100.257 Y142.716 E.00624
; LINE_WIDTH: 0.777806
G1 F10592.224
G1 X100.149 Y142.798 E.00659
G1 X99.693 Y142.853 E.02227
; LINE_WIDTH: 0.762716
G1 F10811.665
G3 X99.629 Y143.543 I-3.661 J.011 E.03298
; LINE_WIDTH: 0.777456
G1 F10597.213
G3 X100.431 Y143.556 I.34 J3.957 E.03891
; LINE_WIDTH: 0.738091
G1 F11189.975
G1 X100.653 Y143.575 E.01022
; LINE_WIDTH: 0.698726
G1 F11852.979
G1 X100.875 Y143.595 E.00965
; LINE_WIDTH: 0.659361
G1 F12599.498
G1 X101.097 Y143.615 E.00908
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X101.497 Y143.615 E.01527
G1 X156.235 Y143.615 E2.08988
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.704 Y131.953 I29.104 J-43.63 E.60782
G3 X136.912 Y119.083 I41.332 J-33.82 E.57633
G1 X136.092 Y119.382 E.03329
M73 P74 R4
G3 X134.379 Y119.594 I-1.77 J-7.277 E.06605
G1 X133.549 Y119.509 E.03188
G1 X132.727 Y119.302 E.03235
G1 X131.944 Y118.976 E.03237
G1 X131.216 Y118.537 E.03244
G1 X130.564 Y118 E.03228
G1 X130.083 Y117.469 E.02734
G3 X129.081 Y116.142 I34.147 J-26.833 E.06348
G1 X126.698 Y112.93 E.15272
G3 X120.57 Y119.575 I-44.099 J-34.513 E.34549
G1 X120.232 Y119.86 E.01689
G1 X121.451 Y121.314 E.07241
G1 X121.615 Y121.659 E.0146
G1 X121.605 Y122.001 E.01306
G3 X121.451 Y122.294 I-1.026 J-.355 E.01268
G1 X121.445 Y122.3 E.0003
M204 S250
G1 X120.996 Y121.983 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X119.288 Y123.415 E.07056
G1 X119.119 Y123.466 E.00557
G1 X119.035 Y123.421 E.00303
G3 X118.938 Y123.349 I.041 J-.156 E.00393
G1 X118.8 Y123.184 E.00682
G1 X118.661 Y123.018 E.00682
G1 X118.523 Y122.853 E.00682
G1 F3450
G1 X118.384 Y122.688 E.00682
G1 F3300
G1 X118.246 Y122.523 E.00682
G1 F3150
G1 X118.107 Y122.358 E.00682
G1 F3600
G1 X118.01 Y122.242 E.00481
; LINE_WIDTH: 0.520326
G1 X117.966 Y122.19 E.00214
; LINE_WIDTH: 0.520776
G1 X117.904 Y122.117 E.00304
; LINE_WIDTH: 0.521226
G1 X117.842 Y122.044 E.00304
; LINE_WIDTH: 0.521686
G1 X117.78 Y121.971 E.00304
; LINE_WIDTH: 0.522136
G1 X117.718 Y121.898 E.00305
; LINE_WIDTH: 0.522596
G1 X117.657 Y121.824 E.00305
; LINE_WIDTH: 0.523046
G1 X117.595 Y121.751 E.00305
; LINE_WIDTH: 0.523196
G1 X117.574 Y121.727 E.00102
; LINE_WIDTH: 0.531656
G1 X117.567 Y121.603 E.00402
; LINE_WIDTH: 0.544336
G1 X117.556 Y121.417 E.00619
G1 X116.672 Y122.124 E.03765
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-32.254 J-42.072 E.63217
G1 X98.944 Y131.535 E.01574
; LINE_WIDTH: 0.548176
G1 X99.036 Y132.014 E.01635
; LINE_WIDTH: 0.593086
G1 X99.072 Y132.137 E.00467
; LINE_WIDTH: 0.637996
G1 X99.108 Y132.26 E.00504
; LINE_WIDTH: 0.682906
G1 X99.143 Y132.383 E.00542
; LINE_WIDTH: 0.727816
G1 X99.179 Y132.506 E.00579
; LINE_WIDTH: 0.742636
G1 X99.191 Y133.165 E.03041
G1 X99.993 Y133.016 E.03766
G1 X99.906 Y132.501 E.02411
; LINE_WIDTH: 0.727636
G1 X99.942 Y132.455 E.00264
; LINE_WIDTH: 0.682771
G1 X99.978 Y132.409 E.00247
; LINE_WIDTH: 0.637906
G1 X100.013 Y132.363 E.0023
; LINE_WIDTH: 0.593041
G1 X100.049 Y132.317 E.00212
; LINE_WIDTH: 0.548176
G1 X100.214 Y132.244 E.00604
; LINE_WIDTH: 0.519996
G2 X101.407 Y132.132 I-8.79 J-100.088 E.03794
G1 X103.397 Y131.938 E.06332
G1 X103.553 Y131.986 E.00517
G1 X103.639 Y132.162 E.00619
G1 X103.641 Y133.095 E.02954
G1 X103.609 Y133.21 E.00377
G1 X103.082 Y133.634 E.02144
G1 X103.082 Y138.646 E.15868
G1 X103.416 Y138.646 E.01057
G1 X103.612 Y138.76 E.00718
G1 X103.641 Y138.872 E.00366
G1 X103.641 Y141.512 E.08357
G1 X103.572 Y141.674 E.00559
G1 X103.43 Y141.737 E.00493
G3 X103.082 Y141.574 I.221 J-.925 E.01225
G1 X103.078 Y141.804 E.00729
G1 X102.989 Y141.944 E.00524
G1 X102.815 Y141.984 E.00567
G1 X100.006 Y141.464 E.09045
G1 X100.202 Y141.851 E.01376
G1 X100.169 Y142.103 E.00804
G1 X100.039 Y142.176 E.00472
G1 X99.583 Y142.234 E.01455
G1 X99.465 Y142.16 E.00443
G1 X99.082 Y141.699 E.01898
G3 X98.942 Y143.858 I-9.613 J.464 E.06864
G1 X98.937 Y144.167 E.0098
G1 X156.695 Y144.167 E1.82861
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.648 J-43.536 E.51062
G3 X137.192 Y118.315 I40.594 J-33.264 E.49164
G1 X136.612 Y118.616 E.02069
G1 X135.912 Y118.859 E.02344
G1 X135.138 Y119.01 E.02499
G1 X134.369 Y119.041 E.02435
G1 X133.609 Y118.959 E.02418
G1 X132.868 Y118.767 E.02427
G1 X132.162 Y118.467 E.02428
G1 X131.506 Y118.067 E.02433
G1 X130.92 Y117.577 E.0242
G1 X130.497 Y117.103 E.0201
G3 X129.087 Y115.222 I52.904 J-41.145 E.07441
G1 X126.704 Y112.01 E.12664
G3 X120.192 Y119.172 I-42.717 J-32.295 E.30689
; LINE_WIDTH: 0.521596
G1 X119.666 Y119.616 E.02186
; LINE_WIDTH: 0.544336
G1 X119.297 Y119.956 E.01669
G1 X119.491 Y119.983 E.00652
; LINE_WIDTH: 0.531156
G1 X119.632 Y120.002 E.00461
; LINE_WIDTH: 0.521596
G1 X119.649 Y120.022 E.00084
; LINE_WIDTH: 0.521526
G1 X119.711 Y120.097 E.00307
; LINE_WIDTH: 0.521296
G1 X119.773 Y120.171 E.00307
; LINE_WIDTH: 0.521066
G1 X119.835 Y120.245 E.00307
; LINE_WIDTH: 0.520836
G1 X119.897 Y120.32 E.00307
; LINE_WIDTH: 0.520616
G1 X119.959 Y120.394 E.00307
; LINE_WIDTH: 0.520386
G1 X120.021 Y120.468 E.00307
; LINE_WIDTH: 0.520156
G1 X120.064 Y120.521 E.00216
; LINE_WIDTH: 0.519996
G1 X120.161 Y120.636 E.00476
G1 F3150
G1 X120.298 Y120.799 E.00675
G1 F3300
G1 X120.435 Y120.963 E.00675
G1 F3450
G1 X120.572 Y121.126 E.00675
G1 F3600
G1 X120.709 Y121.289 E.00675
G1 X120.846 Y121.453 E.00675
G1 X120.982 Y121.616 E.00675
G3 X121.046 Y121.73 I-.111 J.136 E.00424
G1 X121.076 Y121.823 E.00309
G1 X121.036 Y121.903 E.00282
; WIPE_START
M204 S10000
G1 X120.28 Y122.557 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X119.297 Y119.956 Z7.8 F36000
G1 Z7.4
G1 E.4 F1800
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X117.556 Y121.417 E.07555
; WIPE_START
M204 S10000
G1 X118.322 Y120.774 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X111.746 Y124.648 Z7.8 F36000
G1 X100.144 Y131.481 Z7.8
G1 Z7.4
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.05446
G1 F7719.68
G1 X99.736 Y131.606 E.02838
; WIPE_START
G1 X100.144 Y131.481 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X100.777 Y139.087 Z7.8 F36000
G1 X101.097 Y142.927 Z7.8
G1 Z7.4
G1 E.4 F1800
; LINE_WIDTH: 0.823736
G1 F9975.932
G1 X101.35 Y142.95 E.01307
; LINE_WIDTH: 0.777283
G1 F10599.686
G1 X101.603 Y142.973 E.0123
; LINE_WIDTH: 0.73083
G1 F11306.642
G1 X101.856 Y142.997 E.01153
; LINE_WIDTH: 0.684376
G1 F12114.641
G1 X102.108 Y143.02 E.01076
; LINE_WIDTH: 0.637923
G1 F13047.009
G1 X102.361 Y143.043 E.01
; LINE_WIDTH: 0.59147
G1 F14134.858
G1 X102.614 Y143.066 E.00923
; LINE_WIDTH: 0.545016
G1 F15420.618
G1 X103.085 Y143.068 E.01567
; WIPE_START
G1 X102.614 Y143.066 E-.17883
G1 X102.361 Y143.043 E-.09653
G1 X102.108 Y143.02 E-.09654
G1 X102.087 Y143.018 E-.0081
; WIPE_END
G1 E-.02 F1800
G1 X105.699 Y137.989 Z7.8 F36000
G1 Z7.4
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X105.304 Y137.347 I-1.678 J.589 E.02901
G1 X105.304 Y135.763 E.06048
G2 X106.372 Y134.6 I-25.883 J-24.87 E.06029
G2 X106.988 Y130.466 I-3.606 J-2.65 E.16605
G2 X112.616 Y127.543 I-24.859 J-54.754 E.24225
G3 X113.913 Y128.945 I-31.212 J30.174 E.0729
G3 X113.67 Y134.6 I-3.609 J2.678 E.23379
G2 X111.913 Y136.485 I41.996 J40.909 E.0984
G2 X112.023 Y141.945 I3.507 J2.66 E.22491
G1 X119.337 Y141.945 E.27926
G3 X119.696 Y136.485 I3.702 J-2.498 E.22485
G2 X121.454 Y134.6 I-41.974 J-40.889 E.0984
G2 X122.206 Y130.83 I-3.57 J-2.673 E.15176
G2 X120.313 Y128.002 I-5.952 J1.938 E.13159
G3 X118.893 Y126.117 I3.873 J-4.394 E.09072
G1 X118.761 Y125.658 E.01822
G2 X120.906 Y124.95 I.376 J-2.464 E.08949
G1 X124.761 Y141.945 F36000
G1 F13446.283
G1 X127.104 Y141.945 E.08944
G3 X126.994 Y136.485 I3.397 J-2.8 E.22491
G3 X128.751 Y134.6 I43.731 J39.004 E.0984
G2 X128.994 Y128.945 I-3.366 J-2.978 E.23379
G2 X127.237 Y127.06 I-43.742 J39.013 E.0984
G3 X126.162 Y123.289 I3.479 J-3.03 E.1546
G3 X127.853 Y120.462 I5.06 J1.106 E.12802
G2 X129.15 Y118.932 I-3.97 J-4.682 E.07691
G2 X134.931 Y121.244 I5.463 J-5.279 E.24489
G2 X134.535 Y127.06 I3.3 J3.146 E.24146
G2 X136.292 Y128.945 I43.731 J-39.004 E.0984
G3 X136.535 Y134.6 I-3.366 J2.978 E.23379
G3 X134.778 Y136.485 I-43.753 J-39.024 E.0984
G2 X134.418 Y141.945 I3.343 J2.962 E.22485
G1 X142.185 Y141.945 E.29654
G3 X141.244 Y138.371 I3.523 J-2.84 E.1454
G3 X142.935 Y135.543 I5.06 J1.106 E.12802
G2 X144.238 Y134.003 I-3.997 J-4.707 E.07737
G2 X148.806 Y138.769 I41.132 J-34.851 E.25221
G2 X149.5 Y141.945 I3.84 J.825 E.12796
G1 X151.842 Y141.945 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 7.56
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X150.842 Y141.945 E-.38
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
G1 X122.163 Y123.219
G1 Z7.56
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X121.981 Y123.407 E.01002
G1 X120.518 Y124.633 E.07287
G1 X120.061 Y124.919 E.02056
G1 X119.602 Y125.059 E.01834
G1 X119.073 Y125.079 E.02022
G3 X118.19 Y124.766 I.299 J-2.24 E.03603
G1 X117.771 Y124.391 E.02145
G1 X117.29 Y123.817 E.0286
G3 X104.776 Y130.862 I-32.964 J-43.922 E.54983
G1 X105.177 Y131.333 E.02363
G1 X105.361 Y132.071 E.02905
G3 X105.367 Y133.821 I-440.008 J2.602 E.0668
G1 X105.289 Y134.373 E.02127
G3 X104.931 Y135.056 I-1.872 J-.546 E.02967
G1 X104.808 Y135.148 E.00586
G1 X104.808 Y137.111 E.07493
G1 X105.152 Y137.559 E.02155
G1 X105.29 Y137.904 E.01421
G1 X105.368 Y138.451 E.02107
G1 X105.374 Y142.443 E.15243
G1 X154.07 Y142.443 E1.85917
G3 X143.8 Y132.698 I31.802 J-43.797 E.54209
G3 X136.269 Y120.54 I41.791 J-34.3 E.54761
G3 X135.113 Y120.742 I-1.605 J-5.794 E.04487
G1 X134.175 Y120.751 E.0358
G1 X133.266 Y120.646 E.03493
G1 X132.414 Y120.43 E.03355
G1 X131.483 Y120.052 E.03835
G1 X130.602 Y119.535 E.03901
G1 X129.81 Y118.896 E.03886
G1 X129.281 Y118.33 E.0296
G3 X127.853 Y116.453 I42.454 J-33.774 E.09002
G1 X126.662 Y114.847 E.07636
G3 X121.74 Y120.084 I-43.251 J-35.716 E.27459
G1 X122.222 Y120.66 E.0287
G1 X122.556 Y121.233 E.02529
G1 X122.669 Y121.721 E.01915
G1 X122.652 Y122.227 E.01931
G1 X122.52 Y122.681 E.01806
G1 X122.26 Y123.119 E.01947
G1 X122.226 Y123.154 E.00184
G1 X121.735 Y122.819 F36000
G1 F13446.369
G1 X121.604 Y122.958 E.00729
G1 X120.142 Y124.184 E.07287
G1 X119.822 Y124.384 E.01438
G1 X119.437 Y124.492 E.01527
G1 X119.047 Y124.485 E.0149
G1 X118.687 Y124.374 E.01439
G1 X118.298 Y124.101 E.01816
G3 X117.381 Y123.014 I56.872 J-48.937 E.0543
G3 X103.317 Y130.801 I-33.106 J-43.197 E.61602
G1 X104.161 Y131.022 E.03333
G1 X104.647 Y131.582 E.02832
G1 X104.776 Y132.099 E.02032
G3 X104.782 Y133.823 I-600.859 J3.033 E.06582
G1 X104.714 Y134.251 E.01654
G1 X104.476 Y134.687 E.01897
G1 X104.222 Y134.877 E.01211
G1 X104.222 Y137.439 E.09783
G1 X104.422 Y137.527 E.00833
G1 X104.696 Y137.974 E.02003
G3 X104.782 Y138.451 I-1.278 J.476 E.01857
G1 X104.781 Y142.437 E.15218
G1 X104.653 Y143.014 E.02258
G2 X105.749 Y143.029 I.907 J-26.425 E.04183
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.252 Y132.325 I30.037 J-44.207 E.59962
G3 X136.599 Y119.835 I41.505 J-34.022 E.56104
G1 X135.956 Y120.021 E.02552
G1 X135.034 Y120.161 E.03563
G1 X134.182 Y120.166 E.03252
G1 X133.351 Y120.067 E.03195
G1 X132.567 Y119.865 E.03091
G1 X131.714 Y119.514 E.03522
G1 X130.909 Y119.036 E.03573
G1 X130.187 Y118.448 E.03557
G1 X129.719 Y117.942 E.02632
G3 X127.875 Y115.499 I57.569 J-45.392 E.11686
G1 X126.683 Y113.893 E.07636
G3 X120.934 Y120.035 I-42.735 J-34.237 E.32152
G1 X121.774 Y121.036 E.04989
G1 X122.016 Y121.462 E.01871
G1 X122.092 Y121.882 E.01629
G1 X122.028 Y122.328 E.01719
G1 X121.832 Y122.714 E.01652
G1 X121.796 Y122.753 E.00203
G1 X121.305 Y122.414 F36000
G1 F13446.369
G1 X121.228 Y122.509 E.00467
G1 X119.765 Y123.736 E.07287
G1 X119.477 Y123.888 E.01247
G1 X119.141 Y123.907 E.01285
G1 X118.836 Y123.788 E.01249
G1 X118.41 Y123.331 E.02386
G1 X117.454 Y122.189 E.05686
G1 X117.026 Y122.548 E.02128
G3 X102.165 Y130.593 I-32.603 J-42.476 E.64793
G1 X101.922 Y130.676 E.00978
G1 X101.543 Y130.803 E.01527
G1 F13348.486
G1 X101.164 Y130.93 E.01527
G1 F11954.99
G1 X100.784 Y131.057 E.01527
; LINE_WIDTH: 0.667354
G1 F10638.294
G1 X100.713 Y131.104 E.00352
; LINE_WIDTH: 0.714712
G1 F10367.261
G1 X100.642 Y131.151 E.00379
; LINE_WIDTH: 0.76207
G1 F10099.683
G1 X100.571 Y131.198 E.00405
; LINE_WIDTH: 0.809427
G1 F9835.62
G1 X100.5 Y131.245 E.00431
; LINE_WIDTH: 0.856785
G1 F9575.064
G1 X100.429 Y131.292 E.00458
; LINE_WIDTH: 0.904143
G1 F9053.737
G1 X100.357 Y131.34 E.00484
; LINE_WIDTH: 0.951501
G1 F8586.247
G1 X100.286 Y131.387 E.0051
; LINE_WIDTH: 0.998858
G1 F8164.665
G1 X100.215 Y131.434 E.00537
; LINE_WIDTH: 1.04622
G1 F7782.544
G1 X100.144 Y131.481 E.00563
G1 X100.228 Y131.497 E.00563
; LINE_WIDTH: 0.998858
G1 F8164.665
G1 X100.312 Y131.513 E.00537
; LINE_WIDTH: 0.951501
G1 F8586.247
G1 X100.395 Y131.529 E.0051
; LINE_WIDTH: 0.904143
G1 F9053.737
G1 X100.479 Y131.545 E.00484
; LINE_WIDTH: 0.856785
G1 F9575.064
G1 X100.563 Y131.562 E.00458
; LINE_WIDTH: 0.809427
G1 F10160.097
G1 X100.647 Y131.578 E.00431
; LINE_WIDTH: 0.76207
G1 F10428.445
G1 X100.731 Y131.594 E.00405
; LINE_WIDTH: 0.714712
G1 F10700.285
G1 X100.815 Y131.61 E.00379
; LINE_WIDTH: 0.667354
G1 F10975.629
G1 X100.898 Y131.626 E.00352
; LINE_WIDTH: 0.619996
G1 F12312.434
G1 X101.296 Y131.587 E.01527
G1 F13446.369
G1 X101.695 Y131.548 E.01527
G1 X103.344 Y131.387 E.06326
G1 X103.84 Y131.512 E.01953
G1 X104.124 Y131.846 E.01677
G1 X104.191 Y132.16 E.01226
G1 X104.196 Y133.825 E.06353
G1 X104.132 Y134.135 E.01212
G1 X104.022 Y134.318 E.00814
G1 X103.637 Y134.579 E.01777
G1 X103.637 Y137.768 E.12175
G1 X103.991 Y137.924 E.01477
G1 X104.165 Y138.232 E.01354
G1 X104.196 Y138.45 E.00841
G1 X104.196 Y142.037 E.13691
G1 X104.196 Y142.437 E.01527
G1 F12941.74
G1 X104.123 Y142.766 E.01289
; LINE_WIDTH: 0.659729
G1 F11779.354
G1 X104.055 Y142.868 E.00499
; LINE_WIDTH: 0.699461
G1 F11371.165
G1 X103.988 Y142.971 E.00531
; LINE_WIDTH: 0.739194
G1 F10970.128
G1 X103.921 Y143.073 E.00563
; LINE_WIDTH: 0.778926
G1 F10576.291
G1 X103.853 Y143.175 E.00594
; LINE_WIDTH: 0.821511
G1 F10004.129
G1 X103.809 Y143.214 E.00302
; LINE_WIDTH: 0.864096
G1 F9490.697
G1 X103.765 Y143.253 E.00318
; LINE_WIDTH: 0.906681
G1 F9027.393
G1 X103.721 Y143.291 E.00334
; LINE_WIDTH: 0.949266
G1 F8607.217
G1 X103.676 Y143.33 E.00351
; LINE_WIDTH: 0.991851
G1 F8224.414
G1 X103.632 Y143.369 E.00367
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.588 Y143.407 E.00383
G1 X103.648 Y143.429 E.00419
; LINE_WIDTH: 0.991851
G1 F8224.414
G1 X103.709 Y143.45 E.00401
; LINE_WIDTH: 0.949266
G1 F8607.217
G1 X103.77 Y143.471 E.00384
; LINE_WIDTH: 0.906681
G1 F9027.393
G1 X103.83 Y143.493 E.00366
; LINE_WIDTH: 0.864096
G1 F9490.697
G1 X103.891 Y143.514 E.00348
; LINE_WIDTH: 0.821511
G1 F10004.129
G1 X103.952 Y143.535 E.0033
; LINE_WIDTH: 0.778926
G1 F10576.291
G1 X104.129 Y143.555 E.00868
; LINE_WIDTH: 0.739194
G1 F11172.472
G1 X104.307 Y143.575 E.00822
; LINE_WIDTH: 0.699461
G1 F11765.902
G1 X104.485 Y143.595 E.00776
; LINE_WIDTH: 0.659729
G1 F12374.69
G1 X104.663 Y143.615 E.00729
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X105.063 Y143.615 E.01527
G1 X156.235 Y143.615 E1.95373
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.704 Y131.953 I29.332 J-43.88 E.6078
G3 X136.908 Y119.075 I40.834 J-33.519 E.5767
G1 X136.498 Y119.254 E.01709
G1 X135.793 Y119.458 E.028
G1 X134.955 Y119.581 E.03235
G3 X133.436 Y119.487 I-.377 J-6.269 E.05825
G1 X132.72 Y119.3 E.02826
G1 X131.944 Y118.976 E.03208
G1 X131.216 Y118.537 E.03245
G1 X130.564 Y118 E.03229
G1 X130.158 Y117.553 E.02303
G3 X129.081 Y116.142 I34.875 J-27.727 E.06777
G1 X126.698 Y112.93 E.15272
G3 X122.293 Y117.926 I-45.884 J-36.007 E.25445
G3 X120.11 Y119.963 I-27.297 J-27.076 E.11403
G1 X121.325 Y121.413 E.07222
G1 X121.463 Y121.656 E.01067
G1 X121.506 Y121.951 E.01139
G1 X121.417 Y122.275 E.01283
G1 X121.362 Y122.344 E.00338
; WIPE_START
G1 X121.228 Y122.509 E-.08073
G1 X120.625 Y123.015 E-.29927
; WIPE_END
G1 E-.02 F1800
G1 X119.174 Y120.059 Z7.96 F36000
G1 Z7.56
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X117.679 Y121.314 E.06491
; WIPE_START
M204 S10000
G1 X118.445 Y120.671 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X114.202 Y127.016 Z7.96 F36000
G1 X103.241 Y143.407 Z7.96
G1 Z7.56
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03554
G1 F7865.561
G1 X103.182 Y143.373 E.00441
; LINE_WIDTH: 0.99112
G1 F8230.703
G1 X103.123 Y143.34 E.00422
; LINE_WIDTH: 0.946703
G1 F8631.399
G1 X103.064 Y143.307 E.00402
; LINE_WIDTH: 0.902286
G1 F9073.104
G1 X102.98 Y143.222 E.00677
; LINE_WIDTH: 0.855238
G1 F9593.111
G1 X102.895 Y143.137 E.0064
; LINE_WIDTH: 0.80819
G1 F10176.347
G1 X102.811 Y143.052 E.00603
; LINE_WIDTH: 0.761141
G1 F10553.692
G1 X102.727 Y142.967 E.00567
; LINE_WIDTH: 0.714093
G1 F10937.884
G1 X102.643 Y142.883 E.0053
; LINE_WIDTH: 0.667045
G1 F11328.968
G1 X102.558 Y142.798 E.00493
; LINE_WIDTH: 0.619996
G1 F12840.52
G1 X102.283 Y142.45 E.01695
G1 F13446.369
G1 X101.889 Y142.377 E.01527
G1 X101.107 Y142.232 E.03036
G1 X100.714 Y142.159 E.01527
G1 F12881.748
G1 X100.658 Y142.469 E.01203
; LINE_WIDTH: 0.660969
G1 F11797.744
G1 X100.553 Y142.548 E.00538
; LINE_WIDTH: 0.701941
G1 F11358.949
G1 X100.448 Y142.627 E.00573
; LINE_WIDTH: 0.742914
G1 F10928.469
G1 X100.343 Y142.707 E.00608
; LINE_WIDTH: 0.783886
G1 F10506.305
G1 X100.237 Y142.786 E.00643
G1 X99.695 Y142.853 E.02672
; LINE_WIDTH: 0.762866
G1 F10809.439
G3 X99.63 Y143.543 I-3.625 J.01 E.03298
; LINE_WIDTH: 0.785056
G1 F10489.93
G1 X100.298 Y143.532 E.0327
G1 X100.513 Y143.553 E.01055
; LINE_WIDTH: 0.743791
G1 F11100.071
G1 X100.727 Y143.573 E.00997
; LINE_WIDTH: 0.702526
G1 F11785.571
G1 X100.942 Y143.594 E.00939
; LINE_WIDTH: 0.661261
G1 F12521.479
G1 X101.156 Y143.615 E.00881
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X101.556 Y143.615 E.01527
G1 X102.092 Y143.615 E.02045
; LINE_WIDTH: 0.667045
G1 F11650.803
G1 X102.251 Y143.591 E.00665
; LINE_WIDTH: 0.714093
G1 F11117.648
G1 X102.411 Y143.568 E.00715
; LINE_WIDTH: 0.761141
G1 F10596.985
G1 X102.57 Y143.544 E.00764
; LINE_WIDTH: 0.80819
G1 F10088.807
G1 X102.73 Y143.52 E.00814
; LINE_WIDTH: 0.855238
G1 F9593.111
G1 X102.889 Y143.497 E.00863
; LINE_WIDTH: 0.902286
G1 F9073.104
G1 X103.049 Y143.473 E.00913
; LINE_WIDTH: 0.946703
G1 F8631.399
G1 X103.113 Y143.451 E.00402
; LINE_WIDTH: 0.99112
G1 F8230.703
G1 X103.156 Y143.436 E.00282
; WIPE_START
G1 X103.182 Y143.373 E-.02587
G1 X103.123 Y143.34 E-.0257
G1 X103.064 Y143.307 E-.0257
G1 X102.98 Y143.222 E-.04546
G1 X102.895 Y143.137 E-.04546
G1 X102.811 Y143.052 E-.04546
G1 X102.727 Y142.967 E-.04546
G1 X102.643 Y142.883 E-.04546
G1 X102.558 Y142.798 E-.04546
G1 X102.509 Y142.736 E-.02999
; WIPE_END
G1 E-.02 F1800
G1 X101.205 Y142.849 Z7.96 F36000
G1 Z7.56
G1 E.4 F1800
; LINE_WIDTH: 0.810696
G1 F10143.491
G1 X101.108 Y142.849 E.00491
G1 X101.059 Y142.934 E.00491
G1 X101.108 Y143.018 E.00491
G1 X101.205 Y143.018 E.00491
G1 X101.253 Y142.934 E.00491
; WIPE_START
G1 X101.205 Y143.018 E-.076
G1 X101.108 Y143.018 E-.076
G1 X101.059 Y142.934 E-.076
G1 X101.108 Y142.849 E-.076
G1 X101.205 Y142.849 E-.076
; WIPE_END
G1 E-.02 F1800
G1 X106.454 Y137.309 Z7.96 F36000
G1 X120.881 Y122.079 Z7.96
G1 Z7.56
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G3 X119.41 Y123.312 I-25.634 J-29.079 E.06078
G1 X119.243 Y123.364 E.00554
G1 X119.158 Y123.319 E.00305
G3 X119.064 Y123.25 I.037 J-.15 E.00378
G1 X118.94 Y123.103 E.00609
G1 X118.817 Y122.955 E.00609
G1 X118.693 Y122.808 E.00609
G1 X118.57 Y122.66 E.00609
G1 X118.446 Y122.513 E.00609
G1 X118.322 Y122.365 E.00609
G1 X118.235 Y122.261 E.00429
; LINE_WIDTH: 0.520326
G1 X118.181 Y122.198 E.00265
; LINE_WIDTH: 0.520776
G1 X118.105 Y122.107 E.00376
; LINE_WIDTH: 0.521226
G1 X118.028 Y122.017 E.00376
; LINE_WIDTH: 0.521686
G1 X117.952 Y121.926 E.00377
; LINE_WIDTH: 0.522136
G1 X117.875 Y121.836 E.00377
; LINE_WIDTH: 0.522596
G1 X117.799 Y121.745 E.00377
; LINE_WIDTH: 0.523046
G1 X117.722 Y121.654 E.00378
; LINE_WIDTH: 0.523196
G1 X117.697 Y121.624 E.00126
; LINE_WIDTH: 0.531656
G1 X117.69 Y121.5 E.00402
; LINE_WIDTH: 0.544336
G1 X117.679 Y121.314 E.00619
G1 X116.671 Y122.124 E.04297
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-32.253 J-42.071 E.63216
G1 X98.944 Y131.534 E.01572
; LINE_WIDTH: 0.547976
G1 X99.038 Y132.014 E.01635
; LINE_WIDTH: 0.592699
G1 X99.073 Y132.137 E.00467
; LINE_WIDTH: 0.637421
G1 X99.109 Y132.26 E.00504
; LINE_WIDTH: 0.682144
G1 X99.145 Y132.383 E.00541
; LINE_WIDTH: 0.726866
M73 P75 R4
G1 X99.18 Y132.506 E.00579
; LINE_WIDTH: 0.741526
G1 X99.193 Y133.163 E.0303
G1 X99.993 Y133.015 E.03754
G1 X99.907 Y132.501 E.024
; LINE_WIDTH: 0.726616
G1 X99.942 Y132.455 E.00263
; LINE_WIDTH: 0.681956
G1 X99.978 Y132.409 E.00246
; LINE_WIDTH: 0.637296
G1 X100.014 Y132.363 E.00229
; LINE_WIDTH: 0.592636
G1 X100.049 Y132.316 E.00212
; LINE_WIDTH: 0.547976
G1 X100.214 Y132.244 E.00602
; LINE_WIDTH: 0.519996
G2 X101.406 Y132.132 I-8.763 J-99.797 E.03793
G1 X103.397 Y131.938 E.06332
G1 X103.54 Y131.976 E.0047
G1 X103.638 Y132.152 E.00637
G1 X103.643 Y133.826 E.053
G1 X103.593 Y133.969 E.0048
G1 X103.084 Y134.211 E.01784
G1 X103.084 Y138.225 E.12709
G1 X103.418 Y138.225 E.01057
G1 X103.618 Y138.347 E.00744
G1 X103.643 Y138.45 E.00336
G1 X103.643 Y142.436 E.1262
G1 X103.622 Y142.532 E.0031
G1 X103.456 Y142.659 E.0066
G1 X103.188 Y142.624 E.00855
G1 X103.008 Y142.465 E.00761
G1 X102.583 Y141.943 E.02132
G1 X100.148 Y141.492 E.07841
G1 X100.297 Y141.852 E.01233
G1 X100.254 Y142.092 E.00773
G1 X100.127 Y142.161 E.00456
G1 X99.598 Y142.238 E.01691
G1 X99.465 Y142.16 E.00489
G1 X99.084 Y141.701 E.01888
G3 X98.943 Y143.858 I-9.485 J.463 E.06857
G1 X98.937 Y144.167 E.0098
G1 X156.695 Y144.167 E1.82861
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.611 J-43.496 E.51061
G3 X137.192 Y118.316 I40.272 J-33.073 E.49165
G3 X136.344 Y118.723 I-5.221 J-9.796 E.0298
G1 X135.639 Y118.927 E.02322
G1 X134.881 Y119.033 E.02426
G3 X132.864 Y118.766 I-.223 J-6.061 E.0647
G1 X132.162 Y118.468 E.02415
G1 X131.506 Y118.067 E.02434
G1 X130.92 Y117.577 E.0242
G1 X130.572 Y117.187 E.01653
G3 X129.087 Y115.222 I50.991 J-40.08 E.07797
G1 X126.704 Y112.01 E.12664
G3 X120.192 Y119.172 I-42.398 J-32.006 E.3069
; LINE_WIDTH: 0.521596
G1 X119.543 Y119.719 E.02694
; LINE_WIDTH: 0.544336
G1 X119.174 Y120.059 E.01669
G1 X119.368 Y120.085 E.00652
; LINE_WIDTH: 0.531156
G1 X119.509 Y120.105 E.00461
; LINE_WIDTH: 0.521596
G1 X119.53 Y120.13 E.00104
; LINE_WIDTH: 0.521526
G1 X119.607 Y120.222 E.0038
; LINE_WIDTH: 0.521296
G1 X119.684 Y120.314 E.0038
; LINE_WIDTH: 0.521066
G1 X119.76 Y120.406 E.0038
; LINE_WIDTH: 0.520836
G1 X119.837 Y120.498 E.0038
; LINE_WIDTH: 0.520616
G1 X119.914 Y120.589 E.00379
; LINE_WIDTH: 0.520386
G1 X119.991 Y120.681 E.00379
; LINE_WIDTH: 0.520156
G1 X120.045 Y120.746 E.00267
; LINE_WIDTH: 0.519996
G1 X120.132 Y120.85 E.00429
G1 X120.255 Y120.997 E.00609
G1 X120.379 Y121.145 E.00609
G1 X120.502 Y121.292 E.00609
G1 X120.626 Y121.439 E.00609
G1 X120.749 Y121.587 E.00609
G1 X120.873 Y121.734 E.00609
G3 X120.924 Y121.836 I-.104 J.116 E.00369
G1 X120.954 Y121.924 E.00294
G1 X120.919 Y121.997 E.00256
; WIPE_START
M204 S10000
G1 X120.165 Y122.654 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X115.402 Y128.617 Z7.96 F36000
G1 X103.588 Y143.407 Z7.96
G1 Z7.56
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03554
G1 F7865.561
G1 X103.241 Y143.407 E.02265
; WIPE_START
G1 X103.588 Y143.407 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X101.47 Y136.074 Z7.96 F36000
G1 X100.144 Y131.481 Z7.96
G1 Z7.56
G1 E.4 F1800
; LINE_WIDTH: 1.05432
G1 F7720.74
G1 X99.737 Y131.606 E.02833
; WIPE_START
G1 X100.144 Y131.481 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X105.218 Y137.182 Z7.96 F36000
G1 X105.778 Y137.811 Z7.96
G1 Z7.56
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X105.306 Y136.914 I-2.691 J.845 E.03891
G1 X105.306 Y135.594 E.05042
G3 X106.233 Y134.6 I164.265 J152.373 E.05188
G2 X107.04 Y130.439 I-3.552 J-2.848 E.16834
G2 X112.517 Y127.597 I-30.4 J-65.284 E.23566
G2 X113.773 Y128.945 I223.514 J-207.025 E.07034
G3 X113.807 Y134.6 I-3.424 J2.848 E.23394
G3 X112.052 Y136.485 I-313.137 J-289.689 E.09833
G2 X111.896 Y141.945 I3.386 J2.829 E.22487
G1 X109.553 Y141.945 E.08944
G1 X120.956 Y124.91 F36000
G1 F13446.283
G3 X118.79 Y125.533 I-1.677 J-1.757 E.08947
G1 X118.987 Y126.117 E.02355
G1 X119.593 Y127.06 E.04277
G3 X121.347 Y128.945 I-311.382 J291.574 E.09833
G3 X121.314 Y134.6 I-3.458 J2.807 E.23394
G2 X119.559 Y136.485 I311.973 J292.126 E.09833
G2 X119.467 Y141.945 I3.408 J2.788 E.22489
G1 X126.977 Y141.945 E.28671
G3 X127.134 Y136.485 I3.543 J-2.631 E.22487
G2 X128.888 Y134.6 I-311.973 J-292.126 E.09833
G2 X128.855 Y128.945 I-3.458 J-2.807 E.23394
G3 X127.1 Y127.06 I311.382 J-291.574 E.09833
G3 X127.134 Y121.404 I3.458 J-2.807 E.23394
G2 X128.888 Y119.519 I-311.973 J-292.126 E.09833
G1 X129.213 Y119.004 E.02324
G2 X134.782 Y121.252 I5.438 J-5.448 E.23554
G2 X134.674 Y127.06 I3.409 J2.968 E.24083
G3 X136.429 Y128.945 I-311.382 J291.574 E.09833
G3 X136.395 Y134.6 I-3.458 J2.807 E.23394
G2 X134.641 Y136.485 I311.382 J291.574 E.09833
G2 X134.549 Y141.945 I3.408 J2.788 E.22489
G1 X142.058 Y141.945 E.28671
G3 X142.215 Y136.485 I3.543 J-2.631 E.22487
G2 X143.969 Y134.6 I-311.382 J-291.574 E.09833
G1 X144.3 Y134.076 E.02366
G2 X148.784 Y138.753 I43.106 J-36.839 E.24751
G2 X149.63 Y141.945 I3.973 J.656 E.12992
G1 X151.973 Y141.945 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 7.72
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X150.973 Y141.945 E-.38
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
G1 X122.145 Y123.183
G1 Z7.72
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X122.027 Y123.35 E.00776
G1 X121.858 Y123.51 E.0089
G1 X120.641 Y124.53 E.06065
G1 X120.187 Y124.815 E.02043
G1 X119.725 Y124.957 E.01847
G1 X119.206 Y124.977 E.01984
G1 X118.663 Y124.846 E.02129
G3 X118.084 Y124.485 I.725 J-1.81 E.02621
G3 X117.418 Y123.721 I9.702 J-9.122 E.03872
G3 X104.777 Y130.862 I-32.99 J-43.644 E.55594
G1 X105.176 Y131.33 E.02349
G1 X105.361 Y132.071 E.02918
G3 X105.37 Y134.59 I-631.09 J3.525 E.09615
G1 X105.304 Y135.09 E.01927
G1 X105.121 Y135.539 E.0185
G1 X105.005 Y135.722 E.0083
G1 X104.81 Y135.824 E.00838
G1 X104.81 Y136.833 E.03853
G1 X105.005 Y136.933 E.00835
G1 X105.237 Y137.362 E.0186
G1 X105.357 Y137.848 E.01913
G3 X105.366 Y141.894 I-255.015 J2.596 E.15446
G1 X105.377 Y142.443 E.02099
G1 X154.069 Y142.443 E1.859
G3 X143.803 Y132.701 I31.508 J-43.483 E.54191
G3 X136.271 Y120.545 I42.477 J-34.73 E.54754
G1 X135.454 Y120.705 E.03178
G3 X133.324 Y120.66 I-.894 J-8.184 E.08157
G1 X132.427 Y120.434 E.03532
G1 X131.483 Y120.052 E.03886
G1 X130.605 Y119.536 E.0389
G1 X129.809 Y118.895 E.03903
G3 X129.012 Y118.015 I9.197 J-9.131 E.04535
G1 X126.662 Y114.847 E.15059
G3 X121.623 Y120.194 I-41.956 J-34.49 E.28072
G3 X122.244 Y120.958 I-5.827 J5.369 E.03762
G1 X122.459 Y121.411 E.01914
G1 X122.554 Y121.956 E.02112
G1 X122.504 Y122.461 E.01938
G1 X122.34 Y122.906 E.0181
G1 X122.196 Y123.11 E.00954
G1 X121.617 Y122.914 F36000
G1 F13446.369
G1 X121.482 Y123.061 E.00764
G1 X120.264 Y124.082 E.06065
G1 X119.947 Y124.28 E.01429
G1 X119.565 Y124.389 E.01517
G1 X119.102 Y124.37 E.01768
G1 X118.687 Y124.206 E.01705
G1 X118.343 Y123.912 E.01728
G1 X117.504 Y122.912 E.04986
G3 X103.317 Y130.801 I-32.956 J-42.561 E.62214
G1 X104.161 Y131.022 E.03332
G1 X104.646 Y131.58 E.02823
G1 X104.776 Y132.099 E.02041
G3 X104.784 Y134.589 I-866.453 J4.168 E.09506
G1 X104.712 Y135.023 E.01682
G1 X104.529 Y135.381 E.01536
G1 X104.224 Y135.54 E.0131
G1 X104.224 Y137.118 E.06025
G1 X104.529 Y137.274 E.01306
G1 X104.727 Y137.679 E.0172
G1 X104.784 Y138.075 E.01526
G2 X104.778 Y140.434 I207.395 J1.717 E.09008
G1 X104.781 Y142.434 E.07636
G1 X104.654 Y143.012 E.02259
G2 X105.749 Y143.029 I.909 J-23.091 E.04179
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00187
G3 X144.254 Y132.328 I30.059 J-44.231 E.59949
G3 X136.596 Y119.828 I41.874 J-34.251 E.5614
G1 X136.067 Y119.995 E.02115
G1 X135.346 Y120.129 E.028
G1 X134.414 Y120.179 E.03565
G3 X133.287 Y120.054 I.626 J-10.835 E.0433
G1 X132.561 Y119.863 E.02865
G1 X131.714 Y119.514 E.035
G1 X130.911 Y119.038 E.03562
G1 X130.186 Y118.447 E.03574
G3 X129.066 Y117.106 I10.964 J-10.288 E.06673
G1 X126.683 Y113.893 E.15272
G3 X120.812 Y120.139 I-42.51 J-34.079 E.32761
G1 X121.651 Y121.139 E.04986
G1 X121.902 Y121.592 E.01979
G1 X121.969 Y121.974 E.01477
G1 X121.909 Y122.419 E.01717
G1 X121.718 Y122.805 E.01643
G1 X121.679 Y122.848 E.00222
G1 X121.181 Y122.524 F36000
G1 F13446.369
G1 X121.106 Y122.612 E.00445
G1 X119.888 Y123.633 E.06065
G1 X119.601 Y123.785 E.01239
G1 X119.307 Y123.81 E.01127
G1 X118.988 Y123.704 E.01286
G1 X118.791 Y123.536 E.00986
G1 X117.576 Y122.087 E.07222
G1 X117.027 Y122.547 E.02738
G3 X102.164 Y130.593 I-32.622 J-42.512 E.64794
G1 X101.922 Y130.676 E.00976
G1 X101.543 Y130.803 E.01527
G1 F13348.486
G1 X101.164 Y130.93 E.01527
G1 F11954.99
G1 X100.784 Y131.057 E.01527
; LINE_WIDTH: 0.667354
G1 F10638.294
G1 X100.713 Y131.104 E.00352
; LINE_WIDTH: 0.714712
G1 F10367.261
G1 X100.642 Y131.151 E.00379
; LINE_WIDTH: 0.76207
G1 F10099.683
G1 X100.571 Y131.198 E.00405
; LINE_WIDTH: 0.809427
G1 F9835.62
G1 X100.5 Y131.245 E.00431
; LINE_WIDTH: 0.856785
G1 F9575.064
G1 X100.429 Y131.292 E.00458
; LINE_WIDTH: 0.904143
G1 F9053.737
G1 X100.357 Y131.34 E.00484
; LINE_WIDTH: 0.951501
G1 F8586.247
G1 X100.286 Y131.387 E.0051
; LINE_WIDTH: 0.998858
G1 F8164.665
G1 X100.215 Y131.434 E.00537
; LINE_WIDTH: 1.04622
G1 F7782.544
G1 X100.144 Y131.481 E.00563
G1 X100.228 Y131.497 E.00563
; LINE_WIDTH: 0.998858
G1 F8164.665
G1 X100.312 Y131.513 E.00537
; LINE_WIDTH: 0.951501
G1 F8586.247
G1 X100.395 Y131.529 E.0051
; LINE_WIDTH: 0.904143
G1 F9053.737
G1 X100.479 Y131.545 E.00484
; LINE_WIDTH: 0.856785
G1 F9575.064
G1 X100.563 Y131.562 E.00458
; LINE_WIDTH: 0.809427
G1 F10160.097
G1 X100.647 Y131.578 E.00431
; LINE_WIDTH: 0.76207
G1 F10428.445
G1 X100.731 Y131.594 E.00405
; LINE_WIDTH: 0.714712
G1 F10700.285
G1 X100.815 Y131.61 E.00379
; LINE_WIDTH: 0.667354
G1 F10975.629
G1 X100.898 Y131.626 E.00352
; LINE_WIDTH: 0.619996
G1 F12312.434
G1 X101.296 Y131.587 E.01527
G1 F13446.369
G1 X101.695 Y131.548 E.01527
G1 X103.344 Y131.387 E.06327
G1 X103.84 Y131.512 E.01953
G1 X104.124 Y131.845 E.01671
G1 X104.191 Y132.16 E.01232
G1 X104.198 Y134.588 E.09267
G1 X104.127 Y134.911 E.01265
G1 X104.053 Y135.04 E.00568
G1 X103.639 Y135.256 E.01783
G1 X103.639 Y137.403 E.08197
G1 X104.053 Y137.616 E.01776
G1 X104.18 Y137.901 E.01194
G1 X104.198 Y138.072 E.00657
G2 X104.192 Y140.435 I207.949 J1.719 E.09021
G1 X104.195 Y142.035 E.06109
G1 X104.196 Y142.435 E.01527
G1 F12956.62
G1 X104.123 Y142.765 E.01289
; LINE_WIDTH: 0.659509
G1 F11793.255
G1 X104.056 Y142.867 E.00499
; LINE_WIDTH: 0.699021
G1 F11384.565
G1 X103.989 Y142.97 E.00531
; LINE_WIDTH: 0.738534
G1 F10983.082
G1 X103.921 Y143.072 E.00563
; LINE_WIDTH: 0.778046
G1 F10588.806
G1 X103.854 Y143.174 E.00594
; LINE_WIDTH: 0.820778
G1 F10013.459
G1 X103.81 Y143.213 E.00302
; LINE_WIDTH: 0.86351
G1 F9497.412
G1 X103.765 Y143.252 E.00319
; LINE_WIDTH: 0.906241
G1 F9031.948
G1 X103.721 Y143.291 E.00335
; LINE_WIDTH: 0.948973
G1 F8609.978
G1 X103.676 Y143.33 E.00352
; LINE_WIDTH: 0.991705
G1 F8225.674
G1 X103.632 Y143.369 E.00368
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.588 Y143.407 E.00385
G1 X103.649 Y143.429 E.00421
; LINE_WIDTH: 0.991705
G1 F8225.674
G1 X103.709 Y143.45 E.00403
; LINE_WIDTH: 0.948973
G1 F8609.978
G1 X103.77 Y143.471 E.00385
; LINE_WIDTH: 0.906241
G1 F9031.948
G1 X103.831 Y143.493 E.00367
; LINE_WIDTH: 0.86351
G1 F9497.412
G1 X103.892 Y143.514 E.00349
; LINE_WIDTH: 0.820778
G1 F10013.459
G1 X103.953 Y143.536 E.00331
; LINE_WIDTH: 0.778046
G1 F10588.806
G1 X104.131 Y143.555 E.00869
; LINE_WIDTH: 0.738534
G1 F11182.943
G1 X104.309 Y143.575 E.00822
; LINE_WIDTH: 0.699021
G1 F11777.642
G1 X104.487 Y143.595 E.00776
; LINE_WIDTH: 0.659509
G1 F12387.751
G1 X104.665 Y143.615 E.0073
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X105.065 Y143.615 E.01527
G1 X156.235 Y143.615 E1.95363
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.705 Y131.954 I29.104 J-43.63 E.60776
G3 X136.912 Y119.083 I41.407 J-33.868 E.57639
G1 X136.093 Y119.382 E.03327
G3 X134.381 Y119.594 I-1.774 J-7.304 E.06599
G1 X133.549 Y119.509 E.03195
G1 X132.725 Y119.301 E.03244
G1 X131.944 Y118.976 E.03229
G1 X131.218 Y118.539 E.03234
G1 X130.562 Y117.998 E.03245
G1 X130.037 Y117.417 E.02992
G3 X129.081 Y116.142 I60.136 J-46.096 E.06083
G1 X126.698 Y112.93 E.15272
G3 X121.59 Y118.617 I-43.156 J-33.619 E.29211
G3 X119.987 Y120.066 I-23.255 J-24.124 E.0825
G1 X121.202 Y121.516 E.07222
G1 X121.35 Y121.788 E.01183
G1 X121.375 Y122.134 E.01326
G1 X121.272 Y122.417 E.01149
G1 X121.24 Y122.455 E.0019
; WIPE_START
G1 X121.106 Y122.612 E-.07853
G1 X120.498 Y123.122 E-.30147
; WIPE_END
G1 E-.02 F1800
G1 X119.051 Y120.161 Z8.12 F36000
G1 Z7.72
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X117.802 Y121.212 E.05427
; WIPE_START
M204 S10000
G1 X118.567 Y120.568 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X114.314 Y126.906 Z8.12 F36000
G1 X103.241 Y143.407 Z8.12
G1 Z7.72
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03554
G1 F7865.561
G1 X103.149 Y143.367 E.00655
; LINE_WIDTH: 0.989365
G1 F8245.824
G1 X103.057 Y143.326 E.00625
; LINE_WIDTH: 0.943194
G1 F8664.723
G1 X102.965 Y143.286 E.00595
; LINE_WIDTH: 0.897023
G1 F9128.46
G1 X102.873 Y143.246 E.00565
; LINE_WIDTH: 0.850852
G1 F9644.643
G1 X102.781 Y143.206 E.00534
; LINE_WIDTH: 0.804681
G1 F10222.7
G1 X102.689 Y143.165 E.00504
; LINE_WIDTH: 0.75851
G1 F10539.682
G1 X102.597 Y143.125 E.00474
; LINE_WIDTH: 0.712338
G1 F10861.517
G1 X102.505 Y143.085 E.00444
; LINE_WIDTH: 0.666167
G1 F11188.209
G1 X102.413 Y143.045 E.00414
; LINE_WIDTH: 0.619996
G1 F12537.528
G1 X102.091 Y142.807 E.01527
G1 F13446.369
G1 X101.861 Y142.637 E.01094
G1 X101.725 Y142.348 E.01216
G1 X101.198 Y142.251 E.02045
G1 X100.805 Y142.178 E.01527
G1 F12669.07
G1 X100.731 Y142.468 E.01142
; LINE_WIDTH: 0.662811
G1 F11647.103
G1 X100.629 Y142.545 E.00522
; LINE_WIDTH: 0.705626
G1 F11224.953
G1 X100.527 Y142.621 E.00558
; LINE_WIDTH: 0.748441
G1 F10810.571
G1 X100.426 Y142.698 E.00593
; LINE_WIDTH: 0.791256
G1 F10404.006
G1 X100.324 Y142.775 E.00629
G1 X99.783 Y142.853 E.02699
; LINE_WIDTH: 0.763016
G1 F10807.214
G1 X99.697 Y142.853 E.00408
G3 X99.631 Y143.543 I-3.59 J.008 E.03297
; LINE_WIDTH: 0.792476
G1 F10387.265
G1 X100.385 Y143.528 E.03726
G1 X100.592 Y143.55 E.01028
; LINE_WIDTH: 0.749356
G1 F11013.678
G1 X100.799 Y143.571 E.0097
; LINE_WIDTH: 0.706236
G1 F11700.737
G1 X101.006 Y143.593 E.00911
; LINE_WIDTH: 0.663116
G1 F12408.547
G1 X101.213 Y143.615 E.00853
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X101.613 Y143.615 E.01527
G1 X101.961 Y143.615 E.01329
G1 F12238.237
G1 X102.361 Y143.615 E.01527
; LINE_WIDTH: 0.666167
G1 F10905.582
G1 X102.459 Y143.591 E.00414
; LINE_WIDTH: 0.712338
G1 F10583.077
G1 X102.557 Y143.568 E.00444
; LINE_WIDTH: 0.75851
G1 F10265.413
G1 X102.654 Y143.545 E.00474
; LINE_WIDTH: 0.804681
G1 F9952.627
G1 X102.752 Y143.522 E.00504
; LINE_WIDTH: 0.850852
G1 F9644.643
G1 X102.85 Y143.499 E.00534
; LINE_WIDTH: 0.897023
G1 F9128.46
G1 X102.947 Y143.476 E.00565
; LINE_WIDTH: 0.943194
G1 F8664.723
G1 X103.045 Y143.453 E.00595
; LINE_WIDTH: 0.989365
G1 F8245.824
G1 X103.143 Y143.43 E.00625
; LINE_WIDTH: 1.03554
G1 F7865.561
G1 X103.153 Y143.427 E.00068
; WIPE_START
G1 X103.149 Y143.367 E-.0232
G1 X103.057 Y143.326 E-.03816
G1 X102.965 Y143.286 E-.03816
G1 X102.873 Y143.246 E-.03816
G1 X102.781 Y143.206 E-.03816
G1 X102.689 Y143.165 E-.03816
G1 X102.597 Y143.125 E-.03816
G1 X102.505 Y143.085 E-.03816
G1 X102.413 Y143.045 E-.03816
G1 X102.304 Y142.964 E-.05153
; WIPE_END
G1 E-.02 F1800
G1 X101.261 Y142.857 Z8.12 F36000
G1 Z7.72
G1 E.4 F1800
; LINE_WIDTH: 0.798136
G1 F10310.292
G1 X101.165 Y142.857 E.00475
G1 X101.118 Y142.94 E.00475
G1 X101.165 Y143.022 E.00475
G1 X101.261 Y143.022 E.00475
G1 X101.309 Y142.94 E.00475
; WIPE_START
G1 X101.261 Y143.022 E-.076
G1 X101.165 Y143.022 E-.076
G1 X101.118 Y142.94 E-.076
G1 X101.165 Y142.857 E-.076
G1 X101.261 Y142.857 E-.07599
; WIPE_END
G1 E-.02 F1800
G1 X106.497 Y137.304 Z8.12 F36000
G1 X120.761 Y122.18 Z8.12
G1 Z7.72
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G3 X119.533 Y123.209 I-17.657 J-19.808 E.05074
G1 X119.365 Y123.261 E.00557
G1 X119.28 Y123.216 E.00303
G3 X119.19 Y123.151 I.035 J-.144 E.0036
G1 X119.081 Y123.021 E.00536
G1 X118.972 Y122.892 E.00536
G1 X118.864 Y122.762 E.00536
G1 X118.755 Y122.632 E.00536
G1 X118.646 Y122.502 E.00536
G1 X118.537 Y122.373 E.00536
G1 X118.461 Y122.281 E.00378
; LINE_WIDTH: 0.520326
G1 X118.396 Y122.205 E.00316
; LINE_WIDTH: 0.520776
G1 X118.305 Y122.097 E.00448
; LINE_WIDTH: 0.521226
G1 X118.214 Y121.989 E.00448
; LINE_WIDTH: 0.521686
G1 X118.123 Y121.881 E.00449
; LINE_WIDTH: 0.522136
G1 X118.032 Y121.773 E.00449
; LINE_WIDTH: 0.522596
G1 X117.941 Y121.665 E.0045
; LINE_WIDTH: 0.523046
G1 X117.85 Y121.557 E.0045
; LINE_WIDTH: 0.523196
G1 X117.819 Y121.521 E.0015
; LINE_WIDTH: 0.531656
G1 X117.812 Y121.398 E.00402
; LINE_WIDTH: 0.544336
G1 X117.802 Y121.212 E.00619
G1 X116.672 Y122.124 E.04828
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-32.254 J-42.072 E.63217
G1 X98.944 Y131.534 E.01571
; LINE_WIDTH: 0.534786
G1 X99.027 Y131.979 E.01477
; LINE_WIDTH: 0.582584
G1 X99.066 Y132.11 E.0049
; LINE_WIDTH: 0.630381
G1 X99.105 Y132.242 E.00533
; LINE_WIDTH: 0.678179
G1 X99.143 Y132.373 E.00575
; LINE_WIDTH: 0.725976
G1 X99.182 Y132.505 E.00618
; LINE_WIDTH: 0.740416
G1 X99.194 Y133.161 E.03021
G1 X99.994 Y133.013 E.03742
G1 X99.907 Y132.501 E.02389
; LINE_WIDTH: 0.725616
G1 X99.946 Y132.454 E.00275
; LINE_WIDTH: 0.677909
G1 X99.984 Y132.407 E.00256
; LINE_WIDTH: 0.630201
G1 X100.023 Y132.36 E.00237
; LINE_WIDTH: 0.582494
G1 X100.061 Y132.312 E.00218
; LINE_WIDTH: 0.534786
G1 X100.214 Y132.244 E.00545
; LINE_WIDTH: 0.519996
G2 X101.406 Y132.132 I-8.773 J-99.906 E.03793
G1 X103.397 Y131.938 E.06332
G1 X103.54 Y131.976 E.0047
G1 X103.638 Y132.152 E.00637
G1 X103.646 Y134.587 E.07708
G1 X103.603 Y134.718 E.00436
G1 X103.423 Y134.812 E.00644
G1 X103.086 Y134.817 E.01067
G1 X103.086 Y137.844 E.09583
G1 X103.42 Y137.844 E.01057
G1 X103.625 Y137.975 E.00771
G1 X103.646 Y138.07 E.00308
G2 X103.639 Y140.436 I208.473 J1.721 E.07491
G1 X103.643 Y142.436 E.06332
G1 X103.622 Y142.532 E.0031
G1 X103.456 Y142.659 E.00661
G1 X102.512 Y142.501 E.03032
G1 X102.34 Y142.358 E.00707
G2 X102.061 Y141.848 I-1.804 J.66 E.01847
G1 X100.296 Y141.522 E.05681
G1 X100.392 Y141.863 E.01121
G1 X100.336 Y142.081 E.00715
G1 X100.213 Y142.146 E.0044
G1 X99.677 Y142.238 E.01722
G3 X99.465 Y142.16 I-.038 J-.222 E.00748
G1 X99.086 Y141.704 E.01877
G3 X98.943 Y143.858 I-9.357 J.461 E.0685
G1 X98.937 Y144.167 E.00979
G1 X156.695 Y144.167 E1.82861
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.597 I28.615 J-43.5 E.51061
G3 X137.192 Y118.315 I40.597 J-33.267 E.49166
G1 X136.612 Y118.616 E.02069
G1 X135.913 Y118.859 E.02343
G1 X135.137 Y119.01 E.02501
G1 X134.371 Y119.041 E.02427
G1 X133.61 Y118.959 E.02426
G1 X132.866 Y118.766 E.02433
G1 X132.162 Y118.467 E.02421
G1 X131.508 Y118.068 E.02425
G1 X130.918 Y117.575 E.02434
G1 X130.451 Y117.051 E.02224
G3 X129.087 Y115.222 I107.552 J-81.669 E.07222
G1 X126.704 Y112.01 E.12664
G3 X120.192 Y119.172 I-42.62 J-32.207 E.30689
; LINE_WIDTH: 0.521596
G1 X119.421 Y119.822 E.03203
; LINE_WIDTH: 0.544336
G1 X119.051 Y120.161 E.01669
G1 X119.246 Y120.188 E.00652
; LINE_WIDTH: 0.531156
G1 X119.387 Y120.208 E.00461
; LINE_WIDTH: 0.521596
G1 X119.412 Y120.238 E.00124
; LINE_WIDTH: 0.521526
M73 P76 R4
G1 X119.503 Y120.347 E.00453
; LINE_WIDTH: 0.521296
G1 X119.595 Y120.457 E.00453
; LINE_WIDTH: 0.521066
G1 X119.686 Y120.566 E.00453
; LINE_WIDTH: 0.520836
G1 X119.778 Y120.676 E.00452
; LINE_WIDTH: 0.520616
G1 X119.869 Y120.785 E.00452
; LINE_WIDTH: 0.520386
G1 X119.96 Y120.894 E.00452
; LINE_WIDTH: 0.520156
G1 X120.025 Y120.972 E.00319
; LINE_WIDTH: 0.519996
G1 X120.102 Y121.063 E.00378
G1 X120.21 Y121.193 E.00535
G1 X120.319 Y121.322 E.00535
G1 X120.428 Y121.452 E.00535
G1 X120.536 Y121.582 E.00536
G1 X120.645 Y121.711 E.00535
G1 X120.753 Y121.841 E.00535
G3 X120.801 Y121.938 I-.101 J.11 E.0035
G1 X120.831 Y122.025 E.00291
G1 X120.798 Y122.098 E.00253
; WIPE_START
M204 S10000
G1 X120.047 Y122.758 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X115.289 Y128.726 Z8.12 F36000
G1 X103.588 Y143.407 Z8.12
G1 Z7.72
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03554
G1 F7865.561
G1 X103.241 Y143.407 E.02264
; WIPE_START
G1 X103.588 Y143.407 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X101.47 Y136.074 Z8.12 F36000
G1 X100.144 Y131.481 Z8.12
G1 Z7.72
G1 E.4 F1800
; LINE_WIDTH: 1.05406
G1 F7722.708
G1 X99.738 Y131.606 E.02825
; WIPE_START
G1 X100.144 Y131.481 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X105.529 Y136.89 Z8.12 F36000
G1 X105.588 Y136.949 Z8.12
G1 Z7.72
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.39 Y136.608 E.01507
G1 X105.308 Y136.566 E.00351
G1 X105.308 Y136.09 E.01815
G1 X105.39 Y136.047 E.00354
G2 X105.849 Y134.859 I-3.064 J-1.865 E.0489
G2 X107.101 Y130.411 I-3.175 J-3.294 E.18494
G2 X112.421 Y127.659 I-32.476 J-69.306 E.22877
G2 X113.637 Y128.945 I23.496 J-20.994 E.06758
G3 X113.947 Y134.6 I-3.357 J3.02 E.23374
G3 X112.188 Y136.485 I-34.429 J-30.362 E.09845
G2 X111.764 Y141.945 I3.335 J3.006 E.22487
G1 X109.422 Y141.945 E.08944
G1 X120.991 Y124.881 F36000
G1 F13446.283
G3 X118.801 Y125.408 I-1.619 J-1.915 E.08919
G1 X119.077 Y126.117 E.02906
G1 X119.729 Y127.06 E.04377
G3 X121.488 Y128.945 I-32.677 J32.254 E.09845
G3 X121.178 Y134.6 I-3.667 J2.635 E.23374
G2 X119.419 Y136.485 I32.664 J32.241 E.09845
G2 X119.594 Y141.945 I3.551 J2.619 E.2249
G1 X126.846 Y141.945 E.27687
G3 X127.27 Y136.485 I3.76 J-2.454 E.22487
G2 X129.029 Y134.6 I-32.664 J-32.241 E.09845
G2 X128.719 Y128.945 I-3.667 J-2.635 E.23374
G3 X126.959 Y127.06 I32.677 J-32.254 E.09845
G3 X127.27 Y121.404 I3.667 J-2.635 E.23374
G2 X129.029 Y119.519 I-32.664 J-32.241 E.09845
G1 X129.285 Y119.079 E.01945
G2 X134.636 Y121.255 I5.289 J-5.339 E.22634
G2 X134.81 Y127.06 I3.481 J2.8 E.24137
G3 X136.569 Y128.945 I-32.67 J32.247 E.09845
G3 X136.259 Y134.6 I-3.667 J2.635 E.23374
G2 X134.5 Y136.485 I32.67 J32.247 E.09845
G2 X134.675 Y141.945 I3.551 J2.619 E.2249
G1 X141.927 Y141.945 E.27687
G3 X142.351 Y136.485 I3.76 J-2.454 E.22487
G2 X144.11 Y134.6 I-32.67 J-32.247 E.09845
G1 X144.369 Y134.156 E.01962
G2 X148.755 Y138.722 I41.016 J-35.021 E.24186
G2 X149.756 Y141.945 I4.184 J.468 E.13261
G1 X147.414 Y141.945 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 7.88
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X148.414 Y141.945 E-.38
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
G1 X122.032 Y123.273
G1 Z7.88
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X121.919 Y123.437 E.0076
G1 X121.735 Y123.613 E.0097
G1 X120.763 Y124.428 E.04844
G1 X120.313 Y124.711 E.02031
G1 X119.848 Y124.854 E.01858
G1 X119.337 Y124.875 E.01952
G1 X118.809 Y124.752 E.0207
G1 X118.312 Y124.471 E.0218
G3 X117.546 Y123.624 I4.576 J-4.911 E.04365
G3 X104.777 Y130.861 I-33.193 J-43.683 E.56204
G1 X105.174 Y131.327 E.02336
G1 X105.361 Y132.071 E.02931
G3 X105.372 Y135.005 I-753.622 J4.318 E.11203
G1 X105.316 Y135.474 E.01802
G3 X104.836 Y136.343 I-1.905 J-.486 E.03831
G1 X105.076 Y136.642 E.01465
G1 X105.296 Y137.135 E.02059
G1 X105.372 Y137.683 E.02112
G2 X105.364 Y140.433 I664.331 J3.358 E.10501
G1 X105.377 Y142.443 E.07675
G1 X154.069 Y142.443 E1.859
G3 X143.803 Y132.702 I31.816 J-43.807 E.54186
G3 X136.271 Y120.545 I41.927 J-34.389 E.54759
G3 X135.169 Y120.737 I-2.448 J-10.813 E.04271
G3 X133.272 Y120.651 I-.566 J-8.534 E.07265
G1 X132.426 Y120.433 E.03338
G1 X131.483 Y120.052 E.03882
G1 X130.602 Y119.535 E.03899
G3 X129.011 Y118.015 I4.7 J-6.511 E.08428
G1 X126.662 Y114.847 E.15059
G3 X121.506 Y120.304 I-43.516 J-35.951 E.28683
G3 X122.176 Y121.151 I-4.403 J4.169 E.04129
G1 X122.368 Y121.62 E.01937
G1 X122.431 Y122.043 E.01632
G1 X122.385 Y122.546 E.01927
G1 X122.226 Y122.991 E.01807
G1 X122.083 Y123.199 E.00963
G1 X121.547 Y122.945 F36000
G1 F13446.369
G1 X121.448 Y123.082 E.00645
G1 X121.359 Y123.164 E.00462
G1 X120.387 Y123.979 E.04844
G1 X120.072 Y124.177 E.01421
G1 X119.692 Y124.285 E.01509
G1 X119.272 Y124.277 E.01601
G1 X118.859 Y124.132 E.01673
G1 X118.465 Y123.81 E.01942
G1 X117.626 Y122.809 E.04986
G3 X103.317 Y130.801 I-33.109 J-42.473 E.62821
G1 X104.161 Y131.022 E.03333
G1 X104.645 Y131.578 E.02814
G1 X104.776 Y132.099 E.0205
G3 X104.786 Y135.007 I-1034.886 J5.172 E.11105
G1 X104.733 Y135.389 E.01472
G1 X104.579 Y135.734 E.0144
G1 X104.376 Y135.986 E.01236
G1 X104.226 Y136.047 E.00618
G1 X104.226 Y136.639 E.02257
G1 X104.376 Y136.7 E.00618
G1 X104.666 Y137.115 E.01934
G1 X104.781 Y137.559 E.01749
G3 X104.778 Y140.434 I-586.561 J.762 E.10978
G1 X104.781 Y142.434 E.07636
G1 X104.654 Y143.012 E.02259
G2 X105.749 Y143.029 I.908 J-23.065 E.04179
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I29.803 J-43.952 E.5995
G3 X136.596 Y119.828 I41.376 J-33.947 E.56146
G1 X136.067 Y119.995 E.02117
G1 X135.347 Y120.129 E.02796
G1 X134.414 Y120.179 E.03567
G3 X133.284 Y120.054 I.628 J-10.84 E.04343
G1 X132.56 Y119.863 E.02858
G1 X131.714 Y119.514 E.03495
G1 X130.91 Y119.036 E.03571
G1 X130.185 Y118.446 E.03567
G1 X129.598 Y117.805 E.03318
G3 X129.066 Y117.106 I28.086 J-21.915 E.03356
G1 X126.683 Y113.893 E.15272
G3 X120.971 Y120.002 I-44.419 J-35.805 E.31959
G1 X120.689 Y120.241 E.01412
G1 X121.528 Y121.242 E.04986
G1 X121.667 Y121.442 E.00928
G1 X121.83 Y121.906 E.01879
G1 X121.833 Y122.314 E.01557
G1 X121.703 Y122.729 E.01663
G1 X121.6 Y122.872 E.00672
G1 X121.062 Y122.622 F36000
G1 F13446.369
G1 X120.983 Y122.715 E.00466
G1 X120.011 Y123.53 E.04844
G1 X119.726 Y123.682 E.01232
G1 X119.391 Y123.703 E.01281
G1 X119.139 Y123.617 E.01016
G1 X118.851 Y123.358 E.01478
G1 X117.699 Y121.984 E.06847
G1 X117.026 Y122.548 E.0335
G3 X102.164 Y130.591 I-32.793 J-42.839 E.64788
G1 X101.922 Y130.676 E.00978
G1 X101.543 Y130.803 E.01527
G1 F13348.486
G1 X101.164 Y130.93 E.01527
G1 F11954.99
G1 X100.784 Y131.057 E.01527
; LINE_WIDTH: 0.667354
G1 F10638.294
G1 X100.713 Y131.104 E.00352
; LINE_WIDTH: 0.714712
G1 F10367.261
G1 X100.642 Y131.151 E.00379
; LINE_WIDTH: 0.76207
G1 F10099.683
G1 X100.571 Y131.198 E.00405
; LINE_WIDTH: 0.809427
G1 F9835.646
G1 X100.5 Y131.245 E.00431
; LINE_WIDTH: 0.856785
G1 F9575.064
G1 X100.429 Y131.292 E.00458
; LINE_WIDTH: 0.904143
G1 F9053.737
G1 X100.357 Y131.34 E.00484
; LINE_WIDTH: 0.951501
G1 F8586.247
G1 X100.286 Y131.387 E.0051
; LINE_WIDTH: 0.998858
G1 F8164.665
G1 X100.215 Y131.434 E.00537
; LINE_WIDTH: 1.04622
G1 F7782.544
G1 X100.144 Y131.481 E.00563
G1 X100.228 Y131.497 E.00563
; LINE_WIDTH: 0.998858
G1 F8164.665
G1 X100.312 Y131.513 E.00537
; LINE_WIDTH: 0.951501
G1 F8586.247
G1 X100.395 Y131.529 E.0051
; LINE_WIDTH: 0.904143
G1 F9053.737
G1 X100.479 Y131.545 E.00484
; LINE_WIDTH: 0.856785
G1 F9575.064
G1 X100.563 Y131.562 E.00458
; LINE_WIDTH: 0.809427
G1 F10160.097
G1 X100.647 Y131.578 E.00431
; LINE_WIDTH: 0.76207
G1 F10428.445
G1 X100.731 Y131.594 E.00405
; LINE_WIDTH: 0.714712
G1 F10700.285
G1 X100.815 Y131.61 E.00379
; LINE_WIDTH: 0.667354
G1 F10975.629
G1 X100.898 Y131.626 E.00352
; LINE_WIDTH: 0.619996
G1 F12312.434
G1 X101.296 Y131.587 E.01527
G1 F13446.369
G1 X101.695 Y131.548 E.01527
G1 X103.344 Y131.387 E.06326
G1 X103.84 Y131.512 E.01953
G1 X104.123 Y131.844 E.01666
G1 X104.191 Y132.16 E.01236
G1 X104.2 Y135.009 E.10877
G1 X104.132 Y135.331 E.01255
G1 X103.967 Y135.568 E.01104
G1 X103.641 Y135.701 E.01344
G1 X103.641 Y136.985 E.04904
G1 X103.967 Y137.118 E.01344
G1 X104.156 Y137.416 E.0135
G1 X104.2 Y137.678 E.01011
G2 X104.192 Y140.435 I666.292 J3.365 E.10528
G1 X104.195 Y142.035 E.06109
G1 X104.196 Y142.435 E.01527
G1 F12956.62
G1 X104.123 Y142.765 E.01289
; LINE_WIDTH: 0.659509
G1 F11793.255
G1 X104.056 Y142.867 E.00499
; LINE_WIDTH: 0.699021
G1 F11384.565
G1 X103.989 Y142.97 E.00531
; LINE_WIDTH: 0.738534
G1 F10983.082
G1 X103.921 Y143.072 E.00563
; LINE_WIDTH: 0.778046
G1 F10588.806
G1 X103.854 Y143.174 E.00594
; LINE_WIDTH: 0.820778
G1 F10013.459
G1 X103.81 Y143.213 E.00302
; LINE_WIDTH: 0.86351
G1 F9497.412
G1 X103.765 Y143.252 E.00319
; LINE_WIDTH: 0.906241
G1 F9031.948
G1 X103.721 Y143.291 E.00335
; LINE_WIDTH: 0.948973
G1 F8609.978
G1 X103.676 Y143.33 E.00352
; LINE_WIDTH: 0.991705
G1 F8225.674
G1 X103.632 Y143.369 E.00368
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.588 Y143.407 E.00385
G1 X103.649 Y143.429 E.00421
; LINE_WIDTH: 0.991705
G1 F8225.674
G1 X103.709 Y143.45 E.00403
; LINE_WIDTH: 0.948973
G1 F8609.978
G1 X103.77 Y143.471 E.00385
; LINE_WIDTH: 0.906241
G1 F9031.948
G1 X103.831 Y143.493 E.00367
; LINE_WIDTH: 0.86351
G1 F9497.412
G1 X103.892 Y143.514 E.00349
; LINE_WIDTH: 0.820778
G1 F10013.459
G1 X103.953 Y143.536 E.00331
; LINE_WIDTH: 0.778046
G1 F10588.806
G1 X104.131 Y143.555 E.00869
; LINE_WIDTH: 0.738534
G1 F11182.943
G1 X104.309 Y143.575 E.00822
; LINE_WIDTH: 0.699021
G1 F11777.642
G1 X104.487 Y143.595 E.00776
; LINE_WIDTH: 0.659509
G1 F12387.751
G1 X104.665 Y143.615 E.0073
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X105.065 Y143.615 E.01527
G1 X156.235 Y143.615 E1.95363
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.706 Y131.955 I29.325 J-43.872 E.60771
G3 X136.912 Y119.083 I40.76 J-33.477 E.57646
G1 X136.092 Y119.382 E.0333
G3 X132.716 Y119.298 I-1.547 J-5.719 E.13077
G1 X131.944 Y118.976 E.03193
G1 X131.217 Y118.538 E.03243
G1 X130.562 Y117.998 E.03238
G1 X130.037 Y117.417 E.0299
G3 X129.081 Y116.142 I59.558 J-45.665 E.06084
G1 X126.698 Y112.93 E.15272
G3 X121.59 Y118.617 I-43.164 J-33.626 E.29211
G3 X119.864 Y120.169 I-14.891 J-14.829 E.08863
G1 X121.08 Y121.618 E.07222
G1 X121.236 Y121.92 E.01295
G1 X121.253 Y122.23 E.01186
G1 X121.154 Y122.514 E.01149
G1 X121.12 Y122.553 E.00198
; WIPE_START
G1 X120.983 Y122.715 E-.08063
G1 X120.379 Y123.221 E-.29937
; WIPE_END
G1 E-.02 F1800
G1 X118.929 Y120.264 Z8.28 F36000
G1 Z7.88
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X117.924 Y121.109 E.04363
; WIPE_START
M204 S10000
G1 X118.69 Y120.465 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X114.426 Y126.796 Z8.28 F36000
G1 X103.241 Y143.407 Z8.28
G1 Z7.88
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03554
G1 F7865.561
G1 X103.073 Y143.353 E.01147
; LINE_WIDTH: 0.989365
G1 F8245.824
G1 X102.906 Y143.299 E.01094
; LINE_WIDTH: 0.943194
G1 F8664.723
G1 X102.739 Y143.245 E.01041
; LINE_WIDTH: 0.897023
G1 F9128.46
G1 X102.572 Y143.191 E.00988
; LINE_WIDTH: 0.850852
G1 F9644.643
G1 X102.405 Y143.137 E.00935
; LINE_WIDTH: 0.804681
G1 F10222.7
G1 X102.237 Y143.083 E.00882
; LINE_WIDTH: 0.75851
G1 F10780.468
G1 X102.07 Y143.029 E.00829
; LINE_WIDTH: 0.712338
G1 F11353.061
G1 X101.903 Y142.976 E.00777
; LINE_WIDTH: 0.666167
G1 F11940.492
G1 X101.736 Y142.922 E.00724
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X101.309 Y142.69 E.01855
G1 X101.133 Y142.39 E.01326
G1 F13390.511
G1 X101.083 Y142.232 E.00636
G1 F12800.084
G1 X100.963 Y142.209 E.00465
G1 F12376.734
G1 X100.899 Y142.355 E.00605
; LINE_WIDTH: 0.665086
G1 F11836.131
G1 X100.777 Y142.457 E.00655
; LINE_WIDTH: 0.710176
G1 F11305.406
G1 X100.655 Y142.559 E.00702
; LINE_WIDTH: 0.755266
G1 F10786.809
G1 X100.533 Y142.662 E.00748
; LINE_WIDTH: 0.800356
G1 F10280.411
G1 X100.411 Y142.764 E.00795
G1 X99.783 Y142.853 E.03171
; LINE_WIDTH: 0.763176
G1 F10804.841
G1 X99.699 Y142.853 E.00398
G3 X99.633 Y143.543 I-3.559 J.007 E.03297
; LINE_WIDTH: 0.799986
G1 F10285.379
G1 X100.473 Y143.525 E.04196
G1 X100.582 Y143.547 E.00556
; LINE_WIDTH: 0.754989
G1 F10638.615
G1 X100.691 Y143.57 E.00524
; LINE_WIDTH: 0.709991
G1 F10997.839
G1 X100.801 Y143.592 E.00491
; LINE_WIDTH: 0.664994
G1 F11363.002
G1 X100.91 Y143.615 E.00458
; LINE_WIDTH: 0.619996
G1 F12658.717
G1 X101.31 Y143.615 E.01527
G1 X101.673 Y143.615 E.01388
; LINE_WIDTH: 0.666167
G1 F11901.504
G1 X101.847 Y143.591 E.00724
; LINE_WIDTH: 0.712338
G1 F11315.04
G1 X102.022 Y143.568 E.00777
; LINE_WIDTH: 0.75851
G1 F10743.424
G1 X102.196 Y143.545 E.00829
; LINE_WIDTH: 0.804681
G1 F10186.628
G1 X102.37 Y143.522 E.00882
; LINE_WIDTH: 0.850852
G1 F9644.643
G1 X102.544 Y143.499 E.00935
; LINE_WIDTH: 0.897023
G1 F9128.46
G1 X102.718 Y143.476 E.00988
; LINE_WIDTH: 0.943194
G1 F8664.723
G1 X102.892 Y143.453 E.01041
; LINE_WIDTH: 0.989365
G1 F8245.824
G1 X103.067 Y143.43 E.01094
; LINE_WIDTH: 1.03554
G1 F7865.561
G1 X103.151 Y143.419 E.00559
; WIPE_START
G1 X103.073 Y143.353 E-.03876
G1 X102.906 Y143.299 E-.06676
G1 X102.739 Y143.245 E-.06676
G1 X102.572 Y143.191 E-.06676
G1 X102.405 Y143.137 E-.06676
G1 X102.237 Y143.083 E-.06676
G1 X102.219 Y143.077 E-.00744
; WIPE_END
G1 E-.02 F1800
G1 X100.942 Y143.011 Z8.28 F36000
G1 Z7.88
G1 E.4 F1800
; LINE_WIDTH: 0.545596
G1 F15403.123
G2 X100.974 Y143.066 I-.028 J.053 E.01031
; WIPE_START
G1 X100.942 Y143.121 E-.076
G1 X100.878 Y143.121 E-.076
G1 X100.846 Y143.066 E-.076
G1 X100.878 Y143.011 E-.076
G1 X100.942 Y143.011 E-.076
; WIPE_END
G1 E-.02 F1800
G1 X106.199 Y137.478 Z8.28 F36000
G1 X120.64 Y122.28 Z8.28
G1 Z7.88
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G3 X119.656 Y123.106 I-11.85 J-13.124 E.04069
G1 X119.49 Y123.158 E.00548
G1 X119.404 Y123.113 E.00308
G3 X119.316 Y123.053 I.029 J-.137 E.00347
G1 X119.222 Y122.941 E.00463
G1 X119.128 Y122.828 E.00463
G1 X119.034 Y122.716 E.00463
G1 X118.94 Y122.604 E.00463
G1 X118.846 Y122.492 E.00463
G1 X118.752 Y122.38 E.00463
G1 X118.686 Y122.301 E.00327
; LINE_WIDTH: 0.520326
G1 X118.612 Y122.213 E.00367
; LINE_WIDTH: 0.520776
G1 X118.506 Y122.087 E.0052
; LINE_WIDTH: 0.521226
G1 X118.4 Y121.962 E.00521
; LINE_WIDTH: 0.521686
G1 X118.294 Y121.837 E.00521
; LINE_WIDTH: 0.522136
G1 X118.189 Y121.711 E.00522
; LINE_WIDTH: 0.522596
G1 X118.083 Y121.586 E.00522
; LINE_WIDTH: 0.523046
G1 X117.977 Y121.46 E.00523
; LINE_WIDTH: 0.523196
G1 X117.942 Y121.419 E.00175
; LINE_WIDTH: 0.531656
G1 X117.935 Y121.295 E.00402
; LINE_WIDTH: 0.544336
G1 X117.924 Y121.109 E.00619
G1 X117.558 Y121.385 E.01524
; LINE_WIDTH: 0.523196
G1 X116.671 Y122.124 E.03679
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-32.253 J-42.072 E.63216
G1 X98.944 Y131.533 E.01569
; LINE_WIDTH: 0.535156
G1 X99.029 Y131.98 E.01484
; LINE_WIDTH: 0.582626
G1 X99.068 Y132.111 E.00489
; LINE_WIDTH: 0.630096
G1 X99.106 Y132.242 E.00531
; LINE_WIDTH: 0.677566
G1 X99.145 Y132.373 E.00573
; LINE_WIDTH: 0.725036
G1 X99.183 Y132.504 E.00615
; LINE_WIDTH: 0.739306
G1 X99.196 Y133.159 E.03011
G1 X99.994 Y133.012 E.03731
G1 X99.908 Y132.501 E.02378
; LINE_WIDTH: 0.724606
G1 X99.946 Y132.454 E.00274
; LINE_WIDTH: 0.677244
G1 X99.985 Y132.407 E.00255
; LINE_WIDTH: 0.629881
G1 X100.023 Y132.359 E.00236
; LINE_WIDTH: 0.582519
G1 X100.061 Y132.312 E.00217
; LINE_WIDTH: 0.535156
G1 X100.214 Y132.244 E.00546
; LINE_WIDTH: 0.519996
G2 X101.406 Y132.132 I-8.765 J-99.816 E.03793
G1 X103.397 Y131.938 E.06332
G1 X103.54 Y131.976 E.0047
G1 X103.638 Y132.152 E.00637
G1 X103.648 Y135.017 E.0907
G1 X103.58 Y135.173 E.00538
G1 X103.422 Y135.237 E.0054
G1 X103.088 Y135.237 E.01057
G1 X103.088 Y137.449 E.07001
G1 X103.422 Y137.449 E.01057
G1 X103.613 Y137.555 E.00693
G1 X103.648 Y137.675 E.00396
G2 X103.639 Y140.436 I665.945 J3.363 E.0874
G1 X103.643 Y142.436 E.06332
G1 X103.622 Y142.532 E.0031
G1 X103.456 Y142.659 E.00661
G1 X103.377 Y142.658 E.00251
G1 X101.835 Y142.378 E.04963
G1 X101.686 Y142.278 E.00568
G1 X101.51 Y141.748 E.01767
G1 X100.44 Y141.55 E.03445
G1 X100.485 Y141.878 E.01047
G1 X100.446 Y142.038 E.00521
G1 X100.299 Y142.131 E.0055
G1 X99.677 Y142.238 E.02
G3 X99.465 Y142.16 I-.038 J-.222 E.00748
G1 X99.088 Y141.706 E.01867
G3 X98.943 Y143.858 I-9.235 J.459 E.06843
G1 X98.937 Y144.167 E.00979
G1 X156.695 Y144.167 E1.82861
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.614 J-43.499 E.51059
G3 X137.192 Y118.315 I40.27 J-33.072 E.4917
G1 X136.614 Y118.615 E.02061
G3 X134.365 Y119.041 I-2.18 J-5.353 E.07296
G1 X133.607 Y118.959 E.02414
G1 X132.865 Y118.766 E.02426
G1 X132.162 Y118.467 E.02419
G1 X131.506 Y118.067 E.02432
G1 X130.918 Y117.575 E.02428
G1 X130.451 Y117.051 E.02222
G3 X129.087 Y115.222 I106.187 J-80.653 E.07223
G1 X126.704 Y112.01 E.12664
G3 X120.183 Y119.181 I-42.042 J-31.683 E.30731
; LINE_WIDTH: 0.521596
G1 X119.298 Y119.924 E.03671
; LINE_WIDTH: 0.544336
G1 X118.929 Y120.264 E.01669
G1 X119.123 Y120.291 E.00651
; LINE_WIDTH: 0.531166
G1 X119.264 Y120.31 E.00461
; LINE_WIDTH: 0.521596
G1 X119.293 Y120.345 E.00144
; LINE_WIDTH: 0.521526
G1 X119.399 Y120.472 E.00526
; LINE_WIDTH: 0.521296
G1 X119.505 Y120.599 E.00526
; LINE_WIDTH: 0.521066
G1 X119.612 Y120.726 E.00526
; LINE_WIDTH: 0.520836
G1 X119.718 Y120.853 E.00525
; LINE_WIDTH: 0.520616
G1 X119.824 Y120.98 E.00525
; LINE_WIDTH: 0.520386
G1 X119.93 Y121.108 E.00525
; LINE_WIDTH: 0.520156
G1 X120.005 Y121.197 E.0037
; LINE_WIDTH: 0.519996
G1 X120.071 Y121.276 E.00326
G1 X120.165 Y121.388 E.00462
G1 X120.259 Y121.5 E.00462
G1 X120.353 Y121.612 E.00462
G1 X120.447 Y121.724 E.00462
G1 X120.54 Y121.836 E.00462
G1 X120.634 Y121.948 E.00462
G3 X120.679 Y122.04 I-.098 J.104 E.00331
G1 X120.708 Y122.126 E.00288
G1 X120.676 Y122.198 E.0025
; WIPE_START
M204 S10000
G1 X119.929 Y122.863 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X115.178 Y128.836 Z8.28 F36000
G1 X103.588 Y143.407 Z8.28
G1 Z7.88
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03554
G1 F7865.561
G1 X103.241 Y143.407 E.02264
; WIPE_START
G1 X103.588 Y143.407 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X101.47 Y136.074 Z8.28 F36000
G1 X100.144 Y131.481 Z8.28
G1 Z7.88
G1 E.4 F1800
; LINE_WIDTH: 1.05392
G1 F7723.769
G1 X99.738 Y131.605 E.02821
; WIPE_START
G1 X100.144 Y131.481 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X105.627 Y136.791 Z8.28 F36000
G1 X105.743 Y136.904 Z8.28
G1 Z7.88
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X105.496 Y136.312 I5.44 J-2.618 E.02448
G2 X105.86 Y134.704 I-2.24 J-1.352 E.06406
G2 X107.168 Y130.379 I-3.285 J-3.354 E.17998
G2 X112.317 Y127.717 I-32.302 J-68.791 E.22133
G2 X113.503 Y128.945 I12.002 J-10.411 E.0652
G3 X114.388 Y134.129 I-3.374 J3.244 E.21322
G3 X112.323 Y136.485 I-11.259 J-7.786 E.11991
G2 X111.437 Y141.67 I3.374 J3.244 E.21322
G1 X111.61 Y141.945 E.01242
G1 X109.267 Y141.945 E.08944
G1 X121.002 Y124.871 F36000
G1 F13446.283
G3 X118.785 Y125.261 I-1.474 J-1.876 E.08952
G2 X119.863 Y127.06 I4.461 J-1.452 E.08074
G3 X121.929 Y129.416 I-9.193 J10.143 E.11991
G3 X121.044 Y134.6 I-4.259 J1.94 E.21322
G2 X119.273 Y136.485 I16.636 J17.4 E.09879
G2 X119.718 Y141.945 I3.739 J2.444 E.22515
G1 X126.691 Y141.945 E.26624
G1 X126.518 Y141.67 E.01242
G3 X127.404 Y136.485 I4.259 J-1.94 E.21322
G2 X129.47 Y134.129 I-9.193 J-10.143 E.11991
G2 X128.584 Y128.945 I-4.259 J-1.94 E.21322
G3 X126.518 Y126.588 I9.193 J-10.143 E.11991
G3 X127.404 Y121.404 I4.259 J-1.94 E.21322
G2 X129.175 Y119.519 I-16.636 J-17.4 E.09879
G1 X129.388 Y119.179 E.01533
G2 X134.482 Y121.261 I5.218 J-5.494 E.21502
G2 X134.059 Y121.875 I1.611 J1.561 E.02862
G2 X134.945 Y127.06 I4.259 J1.94 E.21322
G3 X137.01 Y129.416 I-9.193 J10.143 E.11991
G3 X136.125 Y134.6 I-4.259 J1.94 E.21322
G2 X134.354 Y136.485 I16.636 J17.4 E.09879
G2 X134.799 Y141.945 I3.738 J2.444 E.22515
G1 X141.773 Y141.945 E.26624
G1 X141.6 Y141.67 E.01243
G3 X142.485 Y136.485 I4.259 J-1.94 E.21322
G2 X144.256 Y134.6 I-16.636 J-17.4 E.09879
G1 X144.466 Y134.265 E.01509
G2 X148.718 Y138.688 I55.695 J-49.301 E.23431
G2 X149.88 Y141.945 I4.4 J.267 E.1357
G1 X147.538 Y141.945 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 8.04
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X148.538 Y141.945 E-.38
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
G1 X121.776 Y123.542
G1 Z8.04
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X121.613 Y123.716 E.00911
G1 X120.886 Y124.325 E.03622
G1 X120.438 Y124.607 E.0202
G1 X119.97 Y124.751 E.0187
G1 X119.468 Y124.773 E.01918
G1 X118.99 Y124.671 E.01866
G1 X118.511 Y124.425 E.02055
G1 X118.139 Y124.083 E.01931
G1 X117.674 Y123.528 E.02766
G3 X104.783 Y130.859 I-33.657 J-44.181 E.56787
G1 X105.24 Y131.481 E.02946
G1 X105.361 Y132.076 E.02317
G3 X105.367 Y133.48 I-405.577 J2.539 E.05363
G1 X105.374 Y135.48 E.07636
G1 X105.315 Y135.96 E.01846
G1 X105.168 Y136.345 E.01572
G1 X105.334 Y136.811 E.01892
G1 X105.374 Y137.211 E.01532
G2 X105.364 Y140.433 I769.047 J4.062 E.12302
G1 X105.377 Y142.443 E.07675
G1 X154.069 Y142.443 E1.859
G3 X143.803 Y132.701 I31.508 J-43.482 E.5419
G3 X136.271 Y120.545 I41.92 J-34.385 E.54758
G3 X135.168 Y120.737 I-2.456 J-10.865 E.04275
G3 X133.273 Y120.651 I-.566 J-8.517 E.07261
G1 X132.429 Y120.434 E.03327
G1 X131.484 Y120.053 E.03891
G1 X130.604 Y119.536 E.03893
M73 P77 R4
G1 X129.808 Y118.895 E.03904
G1 X129.16 Y118.193 E.03648
G3 X127.853 Y116.453 I76.227 J-58.599 E.08307
G1 X126.662 Y114.847 E.07636
G3 X121.389 Y120.413 I-44.573 J-36.939 E.29294
G1 X121.855 Y120.969 E.02766
G1 X122.1 Y121.341 E.01704
G1 X122.29 Y121.941 E.02401
G1 X122.292 Y122.484 E.02075
G1 X122.174 Y122.937 E.01787
G1 X121.929 Y123.38 E.01931
G1 X121.838 Y123.476 E.00505
G1 X121.395 Y123.093 F36000
G1 F13446.369
G1 X121.322 Y123.189 E.00459
G1 X120.51 Y123.876 E.04063
G1 X120.196 Y124.073 E.01413
G1 X119.819 Y124.182 E.015
G1 X119.39 Y124.173 E.01637
G1 X119.029 Y124.054 E.0145
G1 X118.623 Y123.748 E.0194
G3 X117.749 Y122.706 I397.041 J-334.238 E.05193
G1 X117.348 Y123.04 E.0199
G3 X103.318 Y130.801 I-32.924 J-42.956 E.61442
G1 X104.227 Y131.068 E.03619
G1 X104.691 Y131.686 E.02948
G1 X104.777 Y132.158 E.01833
G1 X104.788 Y135.482 E.12691
G1 X104.723 Y135.905 E.01633
G1 X104.514 Y136.307 E.0173
G1 X104.467 Y136.345 E.00232
G1 X104.668 Y136.642 E.0137
G1 X104.78 Y137.057 E.01643
G3 X104.778 Y140.434 I-432.691 J1.367 E.12893
G1 X104.781 Y142.434 E.07636
G1 X104.654 Y143.012 E.02259
G2 X105.749 Y143.029 I.909 J-23.091 E.04179
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.06 J-44.231 E.59948
G3 X136.596 Y119.828 I41.369 J-33.942 E.56145
G1 X136.067 Y119.995 E.02115
G1 X135.347 Y120.129 E.02799
G1 X134.411 Y120.179 E.03576
G3 X133.284 Y120.054 I.629 J-10.823 E.0433
G1 X132.578 Y119.868 E.0279
G1 X131.714 Y119.514 E.03563
G1 X130.911 Y119.038 E.03564
G1 X130.185 Y118.446 E.03575
G1 X129.598 Y117.805 E.03319
G3 X127.875 Y115.499 I116.37 J-88.782 E.10991
G1 X126.683 Y113.893 E.07636
G3 X124.317 Y116.658 I-41.415 J-33.038 E.13898
G3 X120.567 Y120.344 I-33.292 J-30.124 E.20087
G1 X121.406 Y121.345 E.04986
G1 X121.578 Y121.606 E.01192
G1 X121.714 Y122.055 E.01794
G1 X121.694 Y122.509 E.01735
G1 X121.542 Y122.903 E.01611
G1 X121.45 Y123.022 E.00575
G1 X120.877 Y122.779 F36000
G1 F13446.369
G1 X120.133 Y123.427 E.03765
G1 X119.85 Y123.578 E.01225
G1 X119.51 Y123.599 E.013
G1 X119.185 Y123.467 E.01339
G3 X117.821 Y121.881 I28.383 J-25.783 E.07988
G1 X117.027 Y122.547 E.0396
G3 X102.165 Y130.593 I-32.622 J-42.512 E.64793
G1 X101.922 Y130.676 E.00978
G1 X101.543 Y130.803 E.01527
G1 F13348.486
G1 X101.164 Y130.93 E.01527
G1 F11954.99
G1 X100.784 Y131.057 E.01527
; LINE_WIDTH: 0.667354
G1 F10638.294
G1 X100.713 Y131.104 E.00352
; LINE_WIDTH: 0.714712
G1 F10367.261
G1 X100.642 Y131.151 E.00379
; LINE_WIDTH: 0.76207
G1 F10099.683
G1 X100.571 Y131.198 E.00405
; LINE_WIDTH: 0.809427
G1 F9835.62
G1 X100.5 Y131.245 E.00431
; LINE_WIDTH: 0.856785
G1 F9575.064
G1 X100.429 Y131.292 E.00458
; LINE_WIDTH: 0.904143
G1 F9053.737
G1 X100.357 Y131.34 E.00484
; LINE_WIDTH: 0.951501
G1 F8586.247
G1 X100.286 Y131.387 E.0051
; LINE_WIDTH: 0.998858
G1 F8164.665
G1 X100.215 Y131.434 E.00537
; LINE_WIDTH: 1.04622
G1 F7782.544
G1 X100.144 Y131.481 E.00563
G1 X100.228 Y131.497 E.00563
; LINE_WIDTH: 0.998858
G1 F8164.665
G1 X100.312 Y131.513 E.00537
; LINE_WIDTH: 0.951501
G1 F8586.247
G1 X100.395 Y131.529 E.0051
; LINE_WIDTH: 0.904143
G1 F9053.737
G1 X100.479 Y131.545 E.00484
; LINE_WIDTH: 0.856785
G1 F9575.064
G1 X100.563 Y131.562 E.00458
; LINE_WIDTH: 0.809427
G1 F10160.097
G1 X100.647 Y131.578 E.00431
; LINE_WIDTH: 0.76207
G1 F10428.445
G1 X100.731 Y131.594 E.00405
; LINE_WIDTH: 0.714712
G1 F10700.285
G1 X100.815 Y131.61 E.00379
; LINE_WIDTH: 0.667354
G1 F10975.629
G1 X100.898 Y131.626 E.00352
; LINE_WIDTH: 0.619996
G1 F12312.434
G1 X101.296 Y131.587 E.01527
G1 F13446.369
G1 X101.695 Y131.548 E.01527
G1 X103.344 Y131.387 E.06326
G1 X103.878 Y131.538 E.02119
G1 X104.148 Y131.906 E.01741
G1 X104.191 Y132.16 E.00985
G1 X104.202 Y135.484 E.12691
G1 X104.134 Y135.807 E.01259
G1 X104.046 Y135.955 E.00657
G1 X103.738 Y136.199 E.01502
G1 X103.643 Y136.219 E.0037
G1 X103.643 Y136.47 E.00958
G1 X103.738 Y136.49 E.0037
G1 X104.046 Y136.734 E.01502
G1 X104.187 Y137.046 E.01306
G1 X104.202 Y137.206 E.00612
G2 X104.192 Y140.435 I774.976 J4.082 E.12329
G1 X104.195 Y142.035 E.06109
G1 X104.196 Y142.435 E.01527
G1 F12956.62
G1 X104.123 Y142.765 E.01289
; LINE_WIDTH: 0.659509
G1 F11793.255
G1 X104.056 Y142.867 E.00499
; LINE_WIDTH: 0.699021
G1 F11384.565
G1 X103.989 Y142.97 E.00531
; LINE_WIDTH: 0.738534
G1 F10983.082
G1 X103.921 Y143.072 E.00563
; LINE_WIDTH: 0.778046
G1 F10588.806
G1 X103.854 Y143.174 E.00594
; LINE_WIDTH: 0.820778
G1 F10013.459
G1 X103.81 Y143.213 E.00302
; LINE_WIDTH: 0.86351
G1 F9497.412
G1 X103.765 Y143.252 E.00319
; LINE_WIDTH: 0.906241
G1 F9031.948
G1 X103.721 Y143.291 E.00335
; LINE_WIDTH: 0.948973
G1 F8609.978
G1 X103.676 Y143.33 E.00352
; LINE_WIDTH: 0.991705
G1 F8225.674
G1 X103.632 Y143.369 E.00368
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.588 Y143.407 E.00385
G1 X103.649 Y143.429 E.00421
; LINE_WIDTH: 0.991705
G1 F8225.674
G1 X103.709 Y143.45 E.00403
; LINE_WIDTH: 0.948973
G1 F8609.978
G1 X103.77 Y143.471 E.00385
; LINE_WIDTH: 0.906241
G1 F9031.948
G1 X103.831 Y143.493 E.00367
; LINE_WIDTH: 0.86351
G1 F9497.412
G1 X103.892 Y143.514 E.00349
; LINE_WIDTH: 0.820778
G1 F10013.459
G1 X103.953 Y143.536 E.00331
; LINE_WIDTH: 0.778046
G1 F10588.806
G1 X104.131 Y143.555 E.00869
; LINE_WIDTH: 0.738534
G1 F11182.943
G1 X104.309 Y143.575 E.00822
; LINE_WIDTH: 0.699021
G1 F11777.642
G1 X104.487 Y143.595 E.00776
; LINE_WIDTH: 0.659509
G1 F12387.751
G1 X104.665 Y143.615 E.0073
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X105.065 Y143.615 E.01527
G1 X156.235 Y143.615 E1.95363
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.955 I29.104 J-43.63 E.60775
G3 X136.912 Y119.083 I41.188 J-33.736 E.57642
G1 X136.092 Y119.382 E.03331
G3 X132.717 Y119.299 I-1.547 J-5.727 E.13068
G1 X131.945 Y118.976 E.03198
G1 X131.218 Y118.539 E.03236
G1 X130.562 Y117.998 E.03247
G1 X130.037 Y117.417 E.0299
G3 X129.081 Y116.142 I77.319 J-58.984 E.06084
G1 X126.698 Y112.93 E.15272
G3 X122.6 Y117.614 I-42.576 J-33.114 E.23777
G3 X119.742 Y120.271 I-24.475 J-23.456 E.14907
G1 X120.957 Y121.721 E.07222
G1 X121.12 Y122.05 E.01401
G1 X121.121 Y122.386 E.01281
G1 X120.986 Y122.684 E.01249
G1 X120.945 Y122.72 E.00212
; WIPE_START
G1 X120.191 Y123.377 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X118.806 Y120.367 Z8.44 F36000
G1 Z8.04
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X118.047 Y121.006 E.03299
; WIPE_START
M204 S10000
G1 X118.806 Y120.367 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X114.533 Y126.691 Z8.44 F36000
G1 X103.241 Y143.407 Z8.44
G1 Z8.04
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03554
G1 F7865.561
G1 X102.989 Y143.348 E.01685
; LINE_WIDTH: 1.00968
G1 F8074.109
G1 X102.6 Y143.257 E.02543
; LINE_WIDTH: 0.969618
G1 F8419.921
G1 X102.21 Y143.165 E.02439
; LINE_WIDTH: 0.92956
G1 F8796.682
G1 X101.821 Y143.074 E.02334
; LINE_WIDTH: 0.889502
G1 F9208.738
G1 X101.431 Y142.983 E.0223
; LINE_WIDTH: 0.849444
G1 F9661.298
G1 X101.042 Y142.892 E.02126
; LINE_WIDTH: 0.809386
G1 F10160.635
G1 X100.772 Y142.794 E.01452
G1 X100.651 Y142.711 E.00741
G1 X99.982 Y142.819 E.03426
; LINE_WIDTH: 0.763336
G1 F10802.47
G1 X99.701 Y142.854 E.01346
G1 X99.672 Y143.145 E.01391
G1 X99.634 Y143.543 E.01901
; LINE_WIDTH: 0.807556
G1 F10184.683
G1 X100.162 Y143.53 E.02663
G1 X100.562 Y143.521 E.02016
; LINE_WIDTH: 0.809386
G1 F10160.635
G1 X100.985 Y143.52 E.0214
; LINE_WIDTH: 0.835236
G1 F9832.689
G1 X101.243 Y143.507 E.01348
; LINE_WIDTH: 0.875296
G1 F9364.299
G1 X101.643 Y143.487 E.02193
; LINE_WIDTH: 0.915356
G1 F8938.504
G1 X102.042 Y143.467 E.02297
; LINE_WIDTH: 0.955416
G1 F8549.746
G1 X102.442 Y143.447 E.02402
; LINE_WIDTH: 0.995476
G1 F8193.396
G1 X102.841 Y143.427 E.02506
; LINE_WIDTH: 1.03554
G1 F7865.561
G1 X103.151 Y143.411 E.02023
; WIPE_START
G1 X102.989 Y143.348 E-.06593
G1 X102.6 Y143.257 E-.152
G1 X102.21 Y143.165 E-.152
G1 X102.185 Y143.159 E-.01007
; WIPE_END
G1 E-.02 F1800
G1 X107.234 Y137.436 Z8.44 F36000
G1 X120.505 Y122.394 Z8.44
G1 Z8.04
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X119.778 Y123.004 E.03003
G1 X119.614 Y123.055 E.00545
G1 X119.527 Y123.01 E.0031
G3 X119.442 Y122.954 I.025 J-.131 E.00332
G1 X119.363 Y122.859 E.0039
G1 X119.284 Y122.765 E.0039
G1 X119.205 Y122.671 E.0039
G1 X119.125 Y122.576 E.0039
G1 X119.046 Y122.482 E.0039
G1 X118.967 Y122.387 E.0039
G1 X118.911 Y122.321 E.00275
; LINE_WIDTH: 0.520326
G1 X118.827 Y122.22 E.00417
; LINE_WIDTH: 0.520776
G1 X118.706 Y122.077 E.00592
; LINE_WIDTH: 0.521226
G1 X118.586 Y121.935 E.00593
; LINE_WIDTH: 0.521686
G1 X118.466 Y121.792 E.00593
; LINE_WIDTH: 0.522136
G1 X118.345 Y121.649 E.00594
; LINE_WIDTH: 0.522596
G1 X118.225 Y121.506 E.00594
; LINE_WIDTH: 0.523046
G1 X118.105 Y121.364 E.00595
; LINE_WIDTH: 0.523196
G1 X118.064 Y121.316 E.00199
; LINE_WIDTH: 0.531656
G1 X118.057 Y121.192 E.00402
; LINE_WIDTH: 0.544336
G1 X118.047 Y121.006 E.00619
G1 X117.681 Y121.282 E.01524
; LINE_WIDTH: 0.523196
G1 X116.671 Y122.124 E.04189
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-32.253 J-42.071 E.63217
G1 X98.945 Y131.533 E.01568
; LINE_WIDTH: 0.535526
G1 X99.031 Y131.981 E.01491
; LINE_WIDTH: 0.582664
G1 X99.069 Y132.112 E.00487
; LINE_WIDTH: 0.629801
G1 X99.108 Y132.243 E.00529
; LINE_WIDTH: 0.676939
G1 X99.146 Y132.373 E.00571
; LINE_WIDTH: 0.724076
G1 X99.185 Y132.504 E.00613
; LINE_WIDTH: 0.738196
G1 X99.197 Y133.158 E.02999
G1 X99.994 Y133.01 E.03719
G1 X99.908 Y132.501 E.02367
; LINE_WIDTH: 0.723596
G1 X99.947 Y132.454 E.00273
; LINE_WIDTH: 0.676579
G1 X99.985 Y132.407 E.00254
; LINE_WIDTH: 0.629561
G1 X100.023 Y132.359 E.00236
; LINE_WIDTH: 0.582544
G1 X100.061 Y132.312 E.00217
; LINE_WIDTH: 0.535526
G1 X100.214 Y132.244 E.00546
; LINE_WIDTH: 0.519996
G2 X101.407 Y132.132 I-8.767 J-99.832 E.03794
G1 X103.397 Y131.938 E.06332
G1 X103.551 Y131.985 E.0051
G1 X103.639 Y132.162 E.00626
G1 X103.65 Y135.486 E.10524
G1 X103.604 Y135.622 E.00455
G1 X103.424 Y135.712 E.00638
G1 X103.09 Y135.712 E.01057
G1 X103.09 Y136.977 E.04003
G1 X103.515 Y136.996 E.01347
G1 X103.643 Y137.148 E.00629
G3 X103.639 Y140.436 I-501.748 J1.106 E.10411
G1 X103.643 Y142.436 E.06332
G1 X103.622 Y142.532 E.0031
G1 X103.456 Y142.659 E.00661
G3 X103.126 Y142.613 I-.037 J-.948 E.01062
G1 X101.158 Y142.255 E.06332
; LINE_WIDTH: 0.562301
G1 X101.093 Y142.227 E.00242
; LINE_WIDTH: 0.604606
G1 X101.028 Y142.199 E.00262
; LINE_WIDTH: 0.646911
G1 X100.964 Y142.172 E.00281
; LINE_WIDTH: 0.689216
G1 X100.899 Y142.144 E.00301
; LINE_WIDTH: 0.737903
G1 X100.848 Y142.072 E.00403
; LINE_WIDTH: 0.78659
G1 X100.798 Y142 E.00431
; LINE_WIDTH: 0.835276
G1 X100.747 Y141.928 E.00459
G1 X100.697 Y141.983 E.00391
; LINE_WIDTH: 0.78659
G1 X100.647 Y142.039 E.00367
; LINE_WIDTH: 0.737903
G1 X100.597 Y142.094 E.00343
; LINE_WIDTH: 0.689216
G1 X100.417 Y142.122 E.00778
; LINE_WIDTH: 0.646911
G1 X100.236 Y142.149 E.00728
; LINE_WIDTH: 0.604606
G1 X100.056 Y142.176 E.00678
; LINE_WIDTH: 0.562301
G1 X99.876 Y142.204 E.00627
; LINE_WIDTH: 0.519996
G1 X99.586 Y142.235 E.00925
G1 X99.465 Y142.16 E.0045
G1 X99.09 Y141.709 E.01857
G3 X98.943 Y143.858 I-9.112 J.458 E.06836
G1 X98.937 Y144.167 E.00979
G1 X156.695 Y144.167 E1.82861
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.613 J-43.498 E.5106
G3 X137.192 Y118.315 I40.666 J-33.308 E.49166
G1 X136.612 Y118.616 E.02069
G3 X134.363 Y119.041 I-2.175 J-5.346 E.07295
G1 X133.607 Y118.959 E.02407
G1 X132.868 Y118.767 E.02418
G1 X132.162 Y118.468 E.02426
G1 X131.508 Y118.068 E.02427
G1 X130.918 Y117.575 E.02435
G1 X130.451 Y117.051 E.02222
G3 X129.087 Y115.222 I140.179 J-106.013 E.07222
G1 X126.704 Y112.01 E.12664
G3 X120.183 Y119.181 I-42.042 J-31.684 E.30731
; LINE_WIDTH: 0.521596
G1 X119.175 Y120.027 E.04179
; LINE_WIDTH: 0.544336
G1 X118.806 Y120.367 E.01669
G1 X119 Y120.394 E.00652
; LINE_WIDTH: 0.531156
G1 X119.141 Y120.413 E.00461
; LINE_WIDTH: 0.521596
G1 X119.175 Y120.453 E.00164
; LINE_WIDTH: 0.521526
G1 X119.295 Y120.597 E.00599
; LINE_WIDTH: 0.521296
G1 X119.416 Y120.742 E.00598
; LINE_WIDTH: 0.521066
G1 X119.537 Y120.887 E.00598
; LINE_WIDTH: 0.520836
G1 X119.658 Y121.031 E.00598
; LINE_WIDTH: 0.520616
G1 X119.779 Y121.176 E.00598
; LINE_WIDTH: 0.520386
G1 X119.9 Y121.321 E.00597
; LINE_WIDTH: 0.520156
G1 X119.985 Y121.422 E.00421
; LINE_WIDTH: 0.519996
G1 X120.041 Y121.489 E.00274
G1 X120.12 Y121.583 E.00389
G1 X120.199 Y121.677 E.00389
G1 X120.278 Y121.772 E.00389
G1 X120.357 Y121.866 E.00389
G1 X120.436 Y121.96 E.00389
G1 X120.515 Y122.055 E.00389
G3 X120.555 Y122.153 I-.091 J.094 E.00345
G1 X120.584 Y122.252 E.00327
G1 X120.549 Y122.315 E.0023
; WIPE_START
M204 S10000
G1 X119.803 Y122.981 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X115.057 Y128.959 Z8.44 F36000
G1 X103.588 Y143.407 Z8.44
G1 Z8.04
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03554
G1 F7865.561
G1 X103.241 Y143.407 E.02264
; WIPE_START
G1 X103.588 Y143.407 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X101.47 Y136.074 Z8.44 F36000
G1 X100.144 Y131.481 Z8.44
G1 Z8.04
G1 E.4 F1800
; LINE_WIDTH: 1.05376
G1 F7724.981
G1 X99.739 Y131.605 E.02816
; WIPE_START
G1 X100.144 Y131.481 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X105.694 Y136.72 Z8.44 F36000
G1 X105.837 Y136.855 Z8.44
G1 Z8.04
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G2 X105.708 Y136.345 I-1.292 J.056 E.02026
G2 X105.868 Y134.551 I-4.196 J-1.278 E.06924
G2 X107.245 Y130.343 I-3.448 J-3.457 E.17544
G2 X112.209 Y127.781 I-33.846 J-71.687 E.21334
G3 X114.124 Y129.887 I-18.299 J18.556 E.10871
G3 X114.52 Y134.129 I-3.981 J2.511 E.16878
G1 X114.246 Y134.6 E.02081
G3 X112.456 Y136.485 I-12.568 J-10.14 E.09935
G2 X111.12 Y141.198 I3.485 J3.534 E.1956
G2 X111.466 Y141.945 I2.626 J-.763 E.03156
G1 X109.123 Y141.945 E.08944
G1 X121.001 Y124.853 F36000
G1 F13446.283
G3 X118.763 Y125.115 I-1.365 J-1.965 E.08955
G2 X119.997 Y127.06 I6.165 J-2.548 E.08835
G3 X121.787 Y128.945 I-10.779 J12.026 E.09935
G3 X122.247 Y129.887 I-2.865 J1.982 E.04019
G3 X120.91 Y134.6 I-4.822 J1.178 E.1956
G2 X119.12 Y136.485 I10.777 J12.024 E.09935
G1 X118.846 Y136.957 E.02081
G2 X119.841 Y141.945 I4.289 J1.739 E.20562
G1 X126.547 Y141.945 E.25606
G3 X126.201 Y141.198 I2.28 J-1.511 E.03156
G3 X127.538 Y136.485 I4.822 J-1.179 E.1956
G2 X129.327 Y134.6 I-10.777 J-12.024 E.09935
G2 X129.787 Y133.658 I-2.865 J-1.981 E.04019
G2 X128.451 Y128.945 I-4.822 J-1.178 E.1956
G3 X126.661 Y127.06 I10.779 J-12.026 E.09935
G3 X126.201 Y126.117 I2.865 J-1.982 E.04019
G3 X127.538 Y121.404 I4.822 J-1.178 E.1956
G2 X129.327 Y119.519 I-10.777 J-12.024 E.09935
G1 X129.476 Y119.264 E.01125
G2 X134.328 Y121.259 I5.139 J-5.601 E.20452
G2 X133.742 Y122.347 I2.742 J2.181 E.04743
G2 X135.078 Y127.06 I4.822 J1.178 E.1956
G3 X136.868 Y128.945 I-10.777 J12.024 E.09935
G3 X137.328 Y129.887 I-2.865 J1.981 E.0402
G3 X135.991 Y134.6 I-4.822 J1.178 E.1956
G2 X134.201 Y136.485 I10.778 J12.025 E.09935
G1 X133.927 Y136.957 E.02081
G2 X134.922 Y141.945 I4.289 J1.739 E.20562
G1 X141.628 Y141.945 E.25606
G3 X141.282 Y141.198 I2.28 J-1.51 E.03156
G3 X142.619 Y136.485 I4.822 J-1.179 E.1956
G2 X144.409 Y134.6 I-10.778 J-12.025 E.09935
G1 X144.548 Y134.36 E.01062
G2 X148.672 Y138.645 I39.931 J-34.303 E.22716
G2 X150.003 Y141.945 I4.619 J.056 E.13949
G1 X147.66 Y141.945 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 8.2
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X148.66 Y141.945 E-.38
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
G1 X119.524 Y124.655
G1 Z8.2
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X119.079 Y124.556 E.01739
G3 X118.548 Y124.258 I.677 J-1.829 E.02337
G3 X117.799 Y123.428 I9.75 J-9.55 E.04268
G3 X104.8 Y130.853 I-33.357 J-43.31 E.57333
G1 X105.137 Y131.253 E.01996
G1 X105.345 Y131.899 E.02593
G3 X105.364 Y140.433 I-392.566 J5.126 E.32582
G1 X105.377 Y142.443 E.07675
G1 X154.069 Y142.443 E1.859
G3 X143.8 Y132.698 I31.507 J-43.482 E.54207
G3 X136.271 Y120.545 I41.738 J-34.267 E.54743
G3 X135.168 Y120.737 I-2.449 J-10.815 E.04275
G3 X133.273 Y120.651 I-.564 J-8.542 E.07259
G1 X132.429 Y120.434 E.03328
G1 X131.484 Y120.053 E.03891
G1 X130.602 Y119.534 E.03905
G1 X129.808 Y118.894 E.03896
G1 X129.012 Y118.015 E.04526
G1 X126.663 Y114.849 E.15051
G3 X121.269 Y120.519 I-42.755 J-35.269 E.29904
G3 X121.934 Y121.363 I-4.28 J4.058 E.04108
G1 X122.145 Y121.921 E.02276
G1 X122.187 Y122.359 E.0168
G1 X122.115 Y122.851 E.019
G1 X121.936 Y123.283 E.01784
G1 X121.629 Y123.69 E.01948
G1 X121.008 Y124.222 E.0312
G1 X120.564 Y124.502 E.02007
G1 X120.093 Y124.648 E.01883
G1 X119.612 Y124.67 E.01838
G1 X119.669 Y124.078 F36000
G1 F13446.369
G1 X119.555 Y124.077 E.00434
G1 X119.123 Y123.936 E.01734
G1 X118.911 Y123.798 E.00967
G3 X117.872 Y122.603 I17.831 J-16.551 E.06047
G1 X117.403 Y122.996 E.02334
G3 X103.298 Y130.805 I-32.928 J-42.833 E.61786
G1 X103.638 Y130.817 E.013
G1 X104.228 Y131.069 E.02449
G1 X104.619 Y131.526 E.02298
G1 X104.768 Y132.003 E.01908
G3 X104.778 Y140.434 I-782.335 J5.156 E.32188
G1 X104.781 Y142.434 E.07636
G1 X104.654 Y143.012 E.02259
G2 X105.749 Y143.029 I.909 J-23.092 E.04179
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.252 Y132.326 I30.06 J-44.232 E.5996
G3 X136.596 Y119.828 I41.81 J-34.209 E.56131
G1 X136.065 Y119.995 E.02126
G1 X135.346 Y120.129 E.02791
G1 X134.411 Y120.179 E.03573
G3 X133.285 Y120.054 I.628 J-10.823 E.04329
G1 X132.563 Y119.864 E.02851
G1 X131.714 Y119.514 E.03505
G1 X130.909 Y119.036 E.03576
G1 X130.185 Y118.446 E.03566
G3 X129.065 Y117.104 I6.487 J-6.55 E.06682
G1 X126.682 Y113.891 E.15272
G3 X122.925 Y118.118 I-42.636 J-34.119 E.21601
G3 X120.444 Y120.447 I-26.342 J-25.568 E.12995
G1 X121.283 Y121.448 E.04986
G1 X121.522 Y121.864 E.01834
G1 X121.6 Y122.251 E.01505
G1 X121.551 Y122.693 E.01698
G1 X121.373 Y123.081 E.01629
G1 X121.114 Y123.369 E.01481
G1 X120.632 Y123.773 E.024
G1 X120.321 Y123.969 E.01404
G1 X119.946 Y124.079 E.0149
G1 X119.759 Y124.078 E.00716
G1 X119.716 Y123.493 F36000
G1 F13446.369
G1 X119.641 Y123.498 E.00288
G1 X119.378 Y123.408 E.01062
G1 X119.1 Y123.158 E.01428
G1 X117.944 Y121.778 E.06871
G1 X117.027 Y122.547 E.0457
G3 X102.489 Y130.475 I-32.836 J-42.921 E.6347
G1 X101.918 Y130.673 E.02309
G1 X101.539 Y130.801 E.01527
G1 F13266.743
G1 X101.16 Y130.929 E.01527
G1 F11877.638
G1 X100.781 Y131.058 E.01527
; LINE_WIDTH: 0.651116
G1 F10565.332
G1 X100.615 Y131.128 E.00726
; LINE_WIDTH: 0.700686
G1 F9997.757
G1 X100.556 Y131.172 E.00319
; LINE_WIDTH: 0.750256
G1 F9771.427
G1 X100.497 Y131.216 E.00343
; LINE_WIDTH: 0.799826
G1 F9547.688
G1 X100.438 Y131.26 E.00367
; LINE_WIDTH: 0.849396
G1 F9326.523
G1 X100.379 Y131.304 E.0039
; LINE_WIDTH: 0.898966
G1 F9107.943
G1 X100.32 Y131.348 E.00414
; LINE_WIDTH: 0.948536
G1 F8614.089
G1 X100.262 Y131.392 E.00438
; LINE_WIDTH: 0.998106
G1 F8171.037
G1 X100.203 Y131.436 E.00462
; LINE_WIDTH: 1.04768
G1 F7771.331
G1 X100.144 Y131.48 E.00485
G1 X100.215 Y131.499 E.00485
; LINE_WIDTH: 0.998106
G1 F8171.037
G1 X100.286 Y131.517 E.00462
; LINE_WIDTH: 0.948536
G1 F8614.089
G1 X100.357 Y131.535 E.00438
; LINE_WIDTH: 0.898966
G1 F9107.943
G1 X100.428 Y131.554 E.00414
; LINE_WIDTH: 0.849396
G1 F9661.867
G1 X100.5 Y131.572 E.0039
; LINE_WIDTH: 0.799826
G1 F9886.937
G1 X100.571 Y131.591 E.00367
; LINE_WIDTH: 0.750256
G1 F10114.6
G1 X100.642 Y131.609 E.00343
; LINE_WIDTH: 0.700686
G1 F10344.853
G1 X100.713 Y131.627 E.00319
; LINE_WIDTH: 0.651116
G1 F11643.796
G1 X101.112 Y131.595 E.01608
G1 F12767.925
G1 X101.702 Y131.547 E.02382
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X102.1 Y131.509 E.01527
G1 X103.337 Y131.388 E.04745
G1 X103.541 Y131.395 E.00779
G1 X103.887 Y131.545 E.0144
G1 X104.108 Y131.812 E.0132
G1 X104.191 Y132.16 E.01369
G3 X104.192 Y140.435 I-1296.9 J4.261 E.31592
G1 X104.195 Y142.035 E.06109
G1 X104.196 Y142.435 E.01527
G1 F12956.402
G1 X104.123 Y142.765 E.01289
; LINE_WIDTH: 0.659514
G1 F11793.048
G1 X104.056 Y142.867 E.00499
; LINE_WIDTH: 0.699031
G1 F11384.361
G1 X103.989 Y142.97 E.00531
; LINE_WIDTH: 0.738549
G1 F10982.837
G1 X103.921 Y143.072 E.00563
; LINE_WIDTH: 0.778066
G1 F10588.521
G1 X103.854 Y143.174 E.00594
; LINE_WIDTH: 0.820795
G1 F10013.246
G1 X103.81 Y143.213 E.00302
; LINE_WIDTH: 0.863523
G1 F9497.259
G1 X103.765 Y143.252 E.00319
; LINE_WIDTH: 0.906251
G1 F9031.845
G1 X103.721 Y143.291 E.00335
; LINE_WIDTH: 0.94898
G1 F8609.914
G1 X103.676 Y143.33 E.00352
; LINE_WIDTH: 0.991708
G1 F8225.646
G1 X103.632 Y143.369 E.00368
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.588 Y143.407 E.00385
G1 X103.648 Y143.429 E.00421
; LINE_WIDTH: 0.991708
G1 F8225.646
G1 X103.709 Y143.45 E.00403
; LINE_WIDTH: 0.94898
G1 F8609.914
G1 X103.77 Y143.471 E.00385
; LINE_WIDTH: 0.906251
G1 F9031.845
G1 X103.831 Y143.493 E.00367
; LINE_WIDTH: 0.863523
G1 F9497.259
G1 X103.892 Y143.514 E.00349
; LINE_WIDTH: 0.820795
G1 F10013.246
G1 X103.953 Y143.536 E.00331
; LINE_WIDTH: 0.778066
G1 F10588.521
G1 X104.131 Y143.555 E.00869
; LINE_WIDTH: 0.738549
G1 F11182.705
G1 X104.309 Y143.575 E.00822
; LINE_WIDTH: 0.699031
G1 F11777.435
G1 X104.487 Y143.595 E.00776
; LINE_WIDTH: 0.659514
G1 F12387.538
G1 X104.665 Y143.615 E.0073
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X105.065 Y143.615 E.01527
G1 X156.235 Y143.615 E1.95363
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.704 Y131.953 I29.105 J-43.631 E.60781
G3 X136.912 Y119.083 I40.793 J-33.494 E.57638
G1 X136.093 Y119.382 E.03328
G3 X134.39 Y119.593 I-1.741 J-7.049 E.06567
G3 X133.434 Y119.488 I1.024 J-13.61 E.03672
G1 X132.717 Y119.299 E.02832
G1 X131.945 Y118.976 E.03195
G1 X131.216 Y118.537 E.03247
G3 X129.953 Y117.317 I4.165 J-5.577 E.06723
G1 X126.698 Y112.93 E.20857
G3 X122.505 Y117.709 I-41.488 J-32.166 E.24287
G3 X119.619 Y120.374 I-24.805 J-23.964 E.15007
G1 X120.834 Y121.824 E.07222
G1 X120.971 Y122.062 E.01046
G1 X121.016 Y122.338 E.01069
G1 X120.94 Y122.659 E.01261
G1 X120.738 Y122.921 E.01262
G1 X120.256 Y123.324 E.024
G1 X119.975 Y123.475 E.01217
G1 X119.806 Y123.486 E.00645
; WIPE_START
G1 X119.641 Y123.498 E-.06284
G1 X119.378 Y123.408 E-.10566
G1 X119.1 Y123.158 E-.14217
G1 X118.983 Y123.018 E-.06933
; WIPE_END
G1 E-.02 F1800
G1 X118.683 Y120.47 Z8.6 F36000
G1 Z8.2
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X118.169 Y120.903 E.02235
; WIPE_START
M204 S10000
G1 X118.683 Y120.47 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X114.422 Y126.802 Z8.6 F36000
G1 X103.247 Y143.407 Z8.6
G1 Z8.2
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.241 Y143.407 E.00042
; LINE_WIDTH: 1.03185
G1 F7894.658
G1 X102.823 Y143.312 E.02783
; LINE_WIDTH: 0.995996
G1 F8188.965
G1 X102.406 Y143.218 E.02683
; LINE_WIDTH: 0.960146
G1 F8506.066
G1 X101.918 Y143.107 E.03018
; LINE_WIDTH: 0.918271
G1 F8909.027
G1 X101.43 Y142.997 E.02881
; LINE_WIDTH: 0.876396
G1 F9352.066
G1 X100.943 Y142.886 E.02745
; LINE_WIDTH: 0.834521
G1 F9841.475
G1 X100.455 Y142.776 E.02608
; LINE_WIDTH: 0.792646
G1 F10384.936
G1 X99.783 Y142.853 E.03347
; LINE_WIDTH: 0.763486
G1 F10800.248
G1 X99.703 Y142.854 E.0038
G3 X99.635 Y143.543 I-3.497 J.004 E.03296
; LINE_WIDTH: 0.792646
G1 F10384.936
G1 X100.387 Y143.528 E.0372
; LINE_WIDTH: 0.828491
G1 F9916.201
G1 X100.815 Y143.51 E.02216
; LINE_WIDTH: 0.864336
G1 F9487.952
G1 X101.242 Y143.492 E.02316
; LINE_WIDTH: 0.906214
G1 F9032.233
G1 X101.742 Y143.471 E.02842
; LINE_WIDTH: 0.948091
G1 F8618.284
G1 X102.242 Y143.449 E.02978
; LINE_WIDTH: 0.989969
G1 F8240.616
G1 X102.741 Y143.428 E.03115
; LINE_WIDTH: 1.03185
G1 F7894.658
G1 X103.157 Y143.41 E.02708
; WIPE_START
G1 X103.241 Y143.407 E-.03176
G1 X102.823 Y143.312 E-.16264
G1 X102.406 Y143.218 E-.16264
G1 X102.347 Y143.204 E-.02296
; WIPE_END
G1 E-.02 F1800
G1 X107.319 Y137.414 Z8.6 F36000
G1 X119.737 Y122.953 Z8.6
G1 Z8.2
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X119.65 Y122.908 E.00311
G3 X119.568 Y122.855 I.022 J-.124 E.00316
G1 X119.504 Y122.778 E.00317
G1 X119.439 Y122.702 E.00317
G1 X119.375 Y122.625 E.00317
G1 X119.311 Y122.548 E.00317
G1 X119.246 Y122.471 E.00317
G1 X119.182 Y122.395 E.00317
G1 X119.137 Y122.341 E.00223
; LINE_WIDTH: 0.520326
G1 X119.042 Y122.228 E.00468
; LINE_WIDTH: 0.520776
G1 F3150
G1 X118.907 Y122.068 E.00664
; LINE_WIDTH: 0.521226
G1 F3300
G1 X118.772 Y121.907 E.00665
; LINE_WIDTH: 0.521686
G1 F3450
G1 X118.637 Y121.747 E.00665
; LINE_WIDTH: 0.522136
G1 F3600
G1 X118.502 Y121.587 E.00666
; LINE_WIDTH: 0.522596
G1 X118.367 Y121.427 E.00667
; LINE_WIDTH: 0.523046
G1 X118.232 Y121.267 E.00667
; LINE_WIDTH: 0.523196
G1 X118.187 Y121.213 E.00223
; LINE_WIDTH: 0.531656
G1 X118.18 Y121.089 E.00402
; LINE_WIDTH: 0.544336
G1 X118.169 Y120.903 E.00619
G1 X117.804 Y121.179 E.01524
; LINE_WIDTH: 0.523196
G1 X116.672 Y122.124 E.04698
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-32.238 J-42.038 E.63218
G1 X98.945 Y131.532 E.01564
; LINE_WIDTH: 0.535816
G1 X99.032 Y131.982 E.01498
; LINE_WIDTH: 0.582639
G1 X99.071 Y132.112 E.00486
; LINE_WIDTH: 0.629461
G1 X99.109 Y132.243 E.00528
; LINE_WIDTH: 0.676284
G1 X99.148 Y132.373 E.00569
; LINE_WIDTH: 0.723106
M73 P78 R4
G1 X99.186 Y132.504 E.00611
; LINE_WIDTH: 0.737076
G1 X99.199 Y133.156 E.02988
G1 X99.994 Y133.008 E.03707
G1 X99.909 Y132.501 E.02356
; LINE_WIDTH: 0.722586
G1 X99.947 Y132.454 E.00272
; LINE_WIDTH: 0.675894
G1 X99.985 Y132.407 E.00254
; LINE_WIDTH: 0.629201
G1 X100.023 Y132.359 E.00235
; LINE_WIDTH: 0.582509
G1 X100.061 Y132.312 E.00217
; LINE_WIDTH: 0.535816
G1 X100.214 Y132.244 E.00547
; LINE_WIDTH: 0.519996
G2 X101.4 Y132.132 I-8.646 J-98.527 E.03774
G1 X103.391 Y131.938 E.06332
G1 X103.55 Y131.984 E.00525
G1 X103.637 Y132.136 E.00555
G3 X103.639 Y140.436 I-1164.377 J4.477 E.26277
G1 X103.643 Y142.436 E.06332
G1 X103.622 Y142.532 E.0031
G1 X103.456 Y142.659 E.00661
G1 X103.377 Y142.658 E.00251
G1 X100.392 Y142.115 E.09605
G1 X99.677 Y142.238 E.02299
G3 X99.465 Y142.16 I-.038 J-.222 E.00748
G1 X99.092 Y141.712 E.01846
G3 X98.944 Y143.858 I-8.995 J.456 E.06829
G1 X98.938 Y144.167 E.00979
G1 X156.695 Y144.167 E1.8286
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.617 J-43.502 E.51059
G3 X137.192 Y118.315 I40.289 J-33.083 E.49169
G3 X134.369 Y119.041 I-2.729 J-4.761 E.0934
G1 X133.607 Y118.959 E.02426
G1 X132.867 Y118.767 E.0242
G1 X132.162 Y118.468 E.02426
G1 X131.506 Y118.066 E.02436
G1 X130.918 Y117.575 E.02427
G1 X130.409 Y117.003 E.02423
G3 X130.278 Y116.829 I1.964 J-1.604 E.00689
G1 X126.704 Y112.01 E.18996
G3 X120.182 Y119.181 I-42.73 J-32.309 E.3073
; LINE_WIDTH: 0.521596
G1 X119.053 Y120.13 E.04687
; LINE_WIDTH: 0.544336
G1 X118.683 Y120.47 E.01669
G1 X118.878 Y120.497 E.00652
; LINE_WIDTH: 0.531156
G1 X119.019 Y120.516 E.00461
; LINE_WIDTH: 0.521596
G1 X119.056 Y120.56 E.00184
; LINE_WIDTH: 0.521526
G1 X119.192 Y120.723 E.00671
; LINE_WIDTH: 0.521296
G1 X119.327 Y120.885 E.00671
; LINE_WIDTH: 0.521066
G1 X119.463 Y121.047 E.00671
; LINE_WIDTH: 0.520836
G1 F3450
G1 X119.598 Y121.209 E.0067
; LINE_WIDTH: 0.520616
G1 F3300
G1 X119.734 Y121.371 E.0067
; LINE_WIDTH: 0.520386
G1 F3150
G1 X119.87 Y121.533 E.0067
; LINE_WIDTH: 0.520156
G1 F3600
G1 X119.965 Y121.648 E.00472
; LINE_WIDTH: 0.519996
G1 X120.011 Y121.702 E.00223
G1 X120.075 Y121.778 E.00316
G1 X120.139 Y121.855 E.00316
G1 X120.203 Y121.932 E.00316
G1 X120.267 Y122.008 E.00317
G1 X120.332 Y122.085 E.00316
G1 X120.396 Y122.161 E.00316
G3 X120.433 Y122.254 I-.088 J.089 E.00326
G1 X120.462 Y122.352 E.00323
G1 X120.383 Y122.497 E.00521
G1 X119.901 Y122.901 E.0199
G1 X119.823 Y122.925 E.00258
; WIPE_START
M204 S10000
G1 X119.65 Y122.908 E-.06612
G1 X119.568 Y122.855 E-.03699
G1 X119.504 Y122.778 E-.03805
G1 X119.439 Y122.702 E-.03805
G1 X119.375 Y122.625 E-.03805
G1 X119.311 Y122.548 E-.03805
G1 X119.246 Y122.471 E-.03805
G1 X119.182 Y122.395 E-.03805
G1 X119.137 Y122.341 E-.02681
G1 X119.1 Y122.297 E-.02179
; WIPE_END
G1 E-.02 F1800
G1 X114.517 Y128.4 Z8.6 F36000
G1 X103.247 Y143.407 Z8.6
G1 Z8.2
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.588 Y143.407 E.02221
; WIPE_START
G1 X103.247 Y143.407 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X101.325 Y136.021 Z8.6 F36000
G1 X100.144 Y131.48 Z8.6
G1 Z8.2
G1 E.4 F1800
; LINE_WIDTH: 1.05148
G1 F7742.297
G1 X99.739 Y131.607 E.02815
; WIPE_START
G1 X100.144 Y131.48 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X105.757 Y136.652 Z8.6 F36000
G1 X105.862 Y136.748 Z8.6
G1 Z8.2
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.861 Y134.406 E.08944
G2 X107.331 Y130.302 I-3.459 J-3.553 E.17229
G2 X112.085 Y127.854 I-38.378 J-80.351 E.20419
G3 X114.046 Y129.887 I-16.837 J18.207 E.10794
G3 X114.66 Y134.129 I-3.92 J2.732 E.16968
G3 X113.599 Y135.543 I-4.377 J-2.18 E.06787
G2 X111.779 Y137.428 I7.814 J9.362 E.10022
G2 X111.166 Y141.67 I3.92 J2.732 E.16968
G1 X111.313 Y141.945 E.01194
G1 X108.971 Y141.945 E.08944
G1 X120.983 Y124.832 F36000
G1 F13446.283
G3 X118.737 Y124.948 I-1.234 J-2.09 E.08928
G1 X118.804 Y125.174 E.00901
G1 X119.32 Y126.117 E.04102
G2 X121.139 Y128.002 I9.634 J-7.477 E.10022
G3 X122.201 Y129.416 I-3.315 J3.594 E.06787
G3 X121.587 Y133.658 I-4.533 J1.51 E.16968
G3 X119.768 Y135.543 I-9.633 J-7.477 E.10022
G2 X118.706 Y136.957 I3.315 J3.594 E.06787
G2 X119.963 Y141.945 I4.542 J1.508 E.20737
G1 X126.395 Y141.945 E.24554
G1 X126.247 Y141.67 E.01194
G3 X126.86 Y137.428 I4.534 J-1.51 E.16968
G3 X128.68 Y135.543 I9.633 J7.477 E.10022
G2 X129.741 Y134.129 I-3.315 J-3.594 E.06787
G2 X129.128 Y129.887 I-4.534 J-1.51 E.16968
G2 X127.308 Y128.002 I-9.634 J7.477 E.10022
G3 X126.086 Y126.117 I2.836 J-3.178 E.0868
G3 X127.672 Y121.404 I5.033 J-.929 E.19825
G2 X129.581 Y119.346 I-7.072 J-8.475 E.10746
G2 X134.156 Y121.251 I5.037 J-5.651 E.19273
G2 X133.788 Y121.875 I1.798 J1.48 E.02778
G2 X134.401 Y126.117 I4.534 J1.51 E.16968
G2 X136.22 Y128.002 I9.633 J-7.477 E.10022
G3 X137.282 Y129.416 I-3.315 J3.594 E.06787
G3 X136.668 Y133.658 I-4.534 J1.51 E.16968
G3 X134.849 Y135.543 I-9.633 J-7.477 E.10022
G2 X133.788 Y136.957 I3.315 J3.594 E.06787
G2 X135.045 Y141.945 I4.542 J1.508 E.20737
G1 X141.476 Y141.945 E.24554
G3 X141.167 Y141.198 I2.128 J-1.316 E.031
G3 X142.754 Y136.485 I5.034 J-.929 E.19825
G2 X144.64 Y134.471 I-7.63 J-9.033 E.10561
G2 X148.615 Y138.591 I46.66 J-41.031 E.21867
G2 X150.126 Y141.945 I4.839 J-.163 E.14408
G1 X147.783 Y141.945 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 8.36
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X148.783 Y141.945 E-.38
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
G1 X121.577 Y123.701
G1 Z8.36
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X121.514 Y123.769 E.00352
G3 X120.689 Y124.398 I-2.168 J-1.987 E.03984
G1 X120.215 Y124.546 E.01894
G1 X119.732 Y124.569 E.01849
G1 X119.261 Y124.474 E.01833
G1 X118.737 Y124.205 E.0225
G1 X118.384 Y123.878 E.01838
G1 X117.922 Y123.326 E.0275
G3 X104.8 Y130.853 I-33.312 J-42.868 E.57945
G1 X105.137 Y131.252 E.01997
G1 X105.345 Y131.899 E.02592
G3 X105.364 Y140.433 I-392.057 J5.127 E.32583
G1 X105.377 Y142.443 E.07675
G1 X154.069 Y142.443 E1.859
G3 X143.803 Y132.701 I31.494 J-43.468 E.54192
G3 X136.271 Y120.545 I41.788 J-34.304 E.54759
G1 X135.454 Y120.705 E.03178
G3 X133.327 Y120.661 I-.893 J-8.182 E.08146
G1 X132.426 Y120.433 E.03546
G1 X131.483 Y120.052 E.03883
G1 X130.604 Y119.536 E.03891
G1 X129.81 Y118.897 E.03893
G1 X129.012 Y118.015 E.04542
G1 X126.663 Y114.849 E.1505
G3 X121.147 Y120.622 I-41.735 J-34.357 E.30514
G1 X121.609 Y121.174 E.0275
G1 X121.858 Y121.554 E.01733
G1 X122.035 Y122.087 E.02144
G1 X122.065 Y122.45 E.0139
G1 X121.996 Y122.94 E.01889
G1 X121.822 Y123.37 E.01772
G1 X121.631 Y123.629 E.01229
G1 X121.12 Y123.33 F36000
G1 F13446.369
G1 X121.047 Y123.422 E.00449
G1 X120.61 Y123.777 E.0215
G1 X120.265 Y123.933 E.01445
G1 X119.875 Y123.989 E.01505
G1 X119.446 Y123.919 E.01659
G1 X119.08 Y123.731 E.01574
G1 X118.833 Y123.501 E.01286
G1 X117.994 Y122.501 E.04986
G1 X117.403 Y122.996 E.02946
G3 X103.298 Y130.805 I-32.984 J-42.934 E.61784
G1 X103.638 Y130.817 E.013
G1 X104.225 Y131.067 E.02436
G1 X104.619 Y131.526 E.02311
G1 X104.768 Y132.003 E.01908
G3 X104.778 Y140.434 I-781.339 J5.158 E.32189
G1 X104.781 Y142.434 E.07636
G1 X104.654 Y143.012 E.02259
G2 X105.749 Y143.029 I.908 J-23.052 E.04179
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.425 J-44.629 E.59945
G3 X136.596 Y119.828 I41.246 J-33.867 E.56145
G1 X136.064 Y119.995 E.02126
G1 X135.346 Y120.129 E.02789
G1 X134.412 Y120.179 E.03573
G3 X133.285 Y120.054 I.63 J-10.837 E.04331
G1 X132.575 Y119.867 E.02801
G1 X131.714 Y119.514 E.03555
G1 X130.911 Y119.038 E.03563
G1 X130.187 Y118.448 E.03565
G3 X129.065 Y117.104 I6.48 J-6.549 E.06696
G1 X126.682 Y113.891 E.15272
G3 X123.64 Y117.383 I-40.488 J-32.195 E.17687
G3 X120.322 Y120.55 I-31.75 J-29.955 E.1752
G1 X121.16 Y121.55 E.04986
G1 X121.335 Y121.816 E.01213
G1 X121.469 Y122.262 E.0178
G1 X121.453 Y122.692 E.01643
G1 X121.309 Y123.086 E.01602
G1 X121.175 Y123.259 E.00834
G1 X120.672 Y122.955 F36000
G1 F13446.369
G1 X120.615 Y123.023 E.0034
G1 X120.202 Y123.333 E.01971
G1 X119.877 Y123.404 E.01272
G1 X119.545 Y123.328 E.01298
G1 X119.282 Y123.125 E.01269
G1 X118.067 Y121.676 E.07222
G1 X117.026 Y122.548 E.05182
G3 X102.489 Y130.476 I-32.748 J-42.753 E.63471
G1 X101.918 Y130.673 E.02309
G1 X101.539 Y130.801 E.01527
G1 F13267.011
G1 X101.16 Y130.929 E.01527
G1 F11877.891
G1 X100.781 Y131.058 E.01527
; LINE_WIDTH: 0.651016
G1 F10565.571
G1 X100.615 Y131.128 E.00726
; LINE_WIDTH: 0.700599
G1 F9998.412
G1 X100.556 Y131.172 E.00319
; LINE_WIDTH: 0.750181
G1 F9772.031
G1 X100.497 Y131.216 E.00343
; LINE_WIDTH: 0.799764
G1 F9548.201
G1 X100.438 Y131.26 E.00367
; LINE_WIDTH: 0.849346
G1 F9326.982
G1 X100.379 Y131.304 E.00391
; LINE_WIDTH: 0.898929
G1 F9108.337
G1 X100.32 Y131.348 E.00414
; LINE_WIDTH: 0.948511
G1 F8614.325
G1 X100.262 Y131.392 E.00438
; LINE_WIDTH: 0.998094
G1 F8171.144
G1 X100.203 Y131.436 E.00462
; LINE_WIDTH: 1.04768
G1 F7771.331
G1 X100.144 Y131.48 E.00486
G1 X100.215 Y131.499 E.00486
; LINE_WIDTH: 0.998094
G1 F8171.144
G1 X100.286 Y131.517 E.00462
; LINE_WIDTH: 0.948511
G1 F8614.325
G1 X100.357 Y131.535 E.00438
; LINE_WIDTH: 0.898929
G1 F9108.337
G1 X100.428 Y131.554 E.00414
; LINE_WIDTH: 0.849346
G1 F9662.459
G1 X100.5 Y131.572 E.0039
; LINE_WIDTH: 0.799764
G1 F9887.597
G1 X100.571 Y131.591 E.00367
; LINE_WIDTH: 0.750181
G1 F10115.327
G1 X100.642 Y131.609 E.00343
; LINE_WIDTH: 0.700599
G1 F10345.658
G1 X100.713 Y131.627 E.00319
; LINE_WIDTH: 0.651016
G1 F11644.65
G1 X101.112 Y131.595 E.01608
G1 F12769.995
G1 X101.702 Y131.548 E.0238
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X102.1 Y131.509 E.01527
G1 X103.337 Y131.388 E.04746
G1 X103.541 Y131.395 E.00779
G1 X103.886 Y131.544 E.01433
G1 X104.108 Y131.812 E.01327
G1 X104.191 Y132.16 E.01369
G3 X104.192 Y140.435 I-1296.897 J4.261 E.31592
G1 X104.195 Y142.035 E.06109
G1 X104.196 Y142.435 E.01527
G1 F12956.402
G1 X104.123 Y142.765 E.01289
; LINE_WIDTH: 0.659514
G1 F11793.048
G1 X104.056 Y142.867 E.00499
; LINE_WIDTH: 0.699031
G1 F11384.361
G1 X103.989 Y142.97 E.00531
; LINE_WIDTH: 0.738549
G1 F10982.837
G1 X103.921 Y143.072 E.00563
; LINE_WIDTH: 0.778066
G1 F10588.521
G1 X103.854 Y143.174 E.00594
; LINE_WIDTH: 0.820795
G1 F10013.246
G1 X103.81 Y143.213 E.00302
; LINE_WIDTH: 0.863523
G1 F9497.259
G1 X103.765 Y143.252 E.00319
; LINE_WIDTH: 0.906251
G1 F9031.845
G1 X103.721 Y143.291 E.00335
; LINE_WIDTH: 0.94898
G1 F8609.914
G1 X103.676 Y143.33 E.00352
; LINE_WIDTH: 0.991708
G1 F8225.646
G1 X103.632 Y143.369 E.00368
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.588 Y143.407 E.00385
G1 X103.648 Y143.429 E.00421
; LINE_WIDTH: 0.991708
G1 F8225.646
G1 X103.709 Y143.45 E.00403
; LINE_WIDTH: 0.94898
G1 F8609.914
G1 X103.77 Y143.471 E.00385
; LINE_WIDTH: 0.906251
G1 F9031.845
G1 X103.831 Y143.493 E.00367
; LINE_WIDTH: 0.863523
G1 F9497.259
G1 X103.892 Y143.514 E.00349
; LINE_WIDTH: 0.820795
G1 F10013.246
G1 X103.953 Y143.536 E.00331
; LINE_WIDTH: 0.778066
G1 F10588.521
G1 X104.131 Y143.555 E.00869
; LINE_WIDTH: 0.738549
G1 F11182.705
G1 X104.309 Y143.575 E.00822
; LINE_WIDTH: 0.699031
G1 F11777.435
G1 X104.487 Y143.595 E.00776
; LINE_WIDTH: 0.659514
G1 F12387.538
G1 X104.665 Y143.615 E.0073
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X105.065 Y143.615 E.01527
G1 X156.235 Y143.615 E1.95363
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.318 J-43.864 E.60773
G3 X136.912 Y119.083 I40.833 J-33.52 E.57643
G1 X136.092 Y119.382 E.03329
G3 X132.714 Y119.298 I-1.549 J-5.683 E.13085
G1 X131.944 Y118.976 E.03186
G1 X131.218 Y118.539 E.03235
G3 X129.952 Y117.317 I4.163 J-5.58 E.06735
G1 X126.698 Y112.93 E.20856
G3 X120.537 Y119.605 I-42.061 J-32.639 E.34721
G1 X119.497 Y120.477 E.05186
G1 X120.712 Y121.927 E.07222
G1 X120.858 Y122.194 E.01165
G1 X120.888 Y122.519 E.01245
G1 X120.796 Y122.803 E.0114
G1 X120.729 Y122.885 E.00406
; WIPE_START
G1 X120.615 Y123.023 E-.06799
G1 X120.202 Y123.333 E-.19617
G1 X119.904 Y123.398 E-.11584
; WIPE_END
G1 E-.02 F1800
G1 X118.561 Y120.573 Z8.76 F36000
G1 Z8.36
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.544336
G1 F3600
M204 S5000
G1 X118.292 Y120.8 E.01171
; WIPE_START
M204 S10000
G1 X118.561 Y120.573 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X114.31 Y126.912 Z8.76 F36000
G1 X103.247 Y143.407 Z8.76
G1 Z8.36
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.241 Y143.407 E.00042
; LINE_WIDTH: 1.03185
G1 F7894.658
G1 X102.823 Y143.312 E.02783
; LINE_WIDTH: 0.995996
G1 F8188.965
G1 X102.406 Y143.218 E.02683
; LINE_WIDTH: 0.960146
G1 F8506.066
G1 X101.918 Y143.107 E.03018
; LINE_WIDTH: 0.918271
G1 F8909.027
G1 X101.43 Y142.997 E.02881
; LINE_WIDTH: 0.876396
G1 F9352.066
G1 X100.943 Y142.886 E.02745
; LINE_WIDTH: 0.834521
G1 F9841.475
G1 X100.455 Y142.776 E.02608
; LINE_WIDTH: 0.792646
G1 F10384.936
G1 X99.783 Y142.854 E.03347
; LINE_WIDTH: 0.763646
G1 F10797.878
G1 X99.705 Y142.854 E.0037
G3 X99.636 Y143.543 I-3.468 J.002 E.03296
; LINE_WIDTH: 0.792646
G1 F10384.936
G1 X100.387 Y143.528 E.03714
; LINE_WIDTH: 0.828491
G1 F9916.201
G1 X100.815 Y143.51 E.02216
; LINE_WIDTH: 0.864336
G1 F9487.952
G1 X101.242 Y143.492 E.02316
; LINE_WIDTH: 0.906214
G1 F9032.233
G1 X101.742 Y143.471 E.02842
; LINE_WIDTH: 0.948091
G1 F8618.284
G1 X102.242 Y143.449 E.02978
; LINE_WIDTH: 0.989969
G1 F8240.616
G1 X102.741 Y143.428 E.03115
; LINE_WIDTH: 1.03185
G1 F7894.658
G1 X103.157 Y143.41 E.02708
; WIPE_START
G1 X103.241 Y143.407 E-.03176
G1 X102.823 Y143.312 E-.16264
G1 X102.406 Y143.218 E-.16264
G1 X102.347 Y143.204 E-.02296
; WIPE_END
G1 E-.02 F1800
G1 X107.354 Y137.444 Z8.76 F36000
G1 X120.26 Y122.6 Z8.76
G1 Z8.36
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X119.917 Y122.847 E.01338
G1 X119.746 Y122.808 E.00556
G1 X119.362 Y122.361 E.01866
; LINE_WIDTH: 0.523196
G1 X118.31 Y121.11 E.05209
; LINE_WIDTH: 0.544336
G1 X118.292 Y120.8 E.01032
G1 X117.926 Y121.077 E.01524
; LINE_WIDTH: 0.523196
G1 X116.671 Y122.124 E.05209
; LINE_WIDTH: 0.519996
G3 X98.936 Y131.038 I-32.251 J-42.064 E.63217
G1 X98.945 Y131.531 E.01562
; LINE_WIDTH: 0.536126
G1 X99.034 Y131.983 E.01505
; LINE_WIDTH: 0.582634
G1 X99.072 Y132.113 E.00485
; LINE_WIDTH: 0.629141
G1 X99.111 Y132.243 E.00526
; LINE_WIDTH: 0.675649
G1 X99.149 Y132.373 E.00567
; LINE_WIDTH: 0.722156
G1 X99.188 Y132.503 E.00608
; LINE_WIDTH: 0.735976
G1 X99.2 Y133.154 E.02977
G1 X99.995 Y133.007 E.03696
G1 X99.909 Y132.501 E.02345
; LINE_WIDTH: 0.721576
G1 X99.947 Y132.454 E.00272
; LINE_WIDTH: 0.675214
G1 X99.985 Y132.406 E.00253
; LINE_WIDTH: 0.628851
G1 X100.023 Y132.359 E.00235
; LINE_WIDTH: 0.582489
G1 X100.061 Y132.312 E.00217
; LINE_WIDTH: 0.536126
G1 X100.214 Y132.244 E.00547
; LINE_WIDTH: 0.519996
G2 X101.4 Y132.132 I-8.651 J-98.583 E.03774
G1 X103.391 Y131.938 E.06332
G1 X103.55 Y131.983 E.00523
G1 X103.637 Y132.136 E.00557
G3 X103.639 Y140.436 I-1163.571 J4.478 E.26277
G1 X103.643 Y142.436 E.06332
G1 X103.622 Y142.532 E.0031
G1 X103.456 Y142.659 E.00661
G1 X103.377 Y142.658 E.00251
G1 X100.392 Y142.115 E.09605
G1 X99.677 Y142.238 E.02299
G3 X99.465 Y142.16 I-.038 J-.222 E.00748
G1 X99.094 Y141.714 E.01836
G3 X98.944 Y143.858 I-8.882 J.454 E.06822
G1 X98.938 Y144.167 E.00978
G1 X156.695 Y144.167 E1.8286
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I29.092 J-44.026 E.51056
G3 X137.192 Y118.315 I40.272 J-33.072 E.49168
G1 X136.612 Y118.616 E.02068
G1 X135.912 Y118.859 E.02345
G1 X135.138 Y119.01 E.02499
G1 X134.369 Y119.041 E.02434
G1 X133.608 Y118.959 E.02426
G1 X132.865 Y118.766 E.02428
G1 X132.162 Y118.468 E.02419
G1 X131.508 Y118.068 E.02426
G1 X130.92 Y117.577 E.02427
G1 X130.409 Y117.003 E.02433
G3 X130.278 Y116.829 I1.947 J-1.592 E.00688
G1 X126.704 Y112.01 E.18996
G3 X120.182 Y119.181 I-42.73 J-32.309 E.3073
; LINE_WIDTH: 0.521596
G1 X118.93 Y120.233 E.05195
; LINE_WIDTH: 0.544336
G1 X118.561 Y120.573 E.01669
G1 X118.896 Y120.619 E.01125
; LINE_WIDTH: 0.521596
G1 X119.945 Y121.873 E.05195
; LINE_WIDTH: 0.519996
G1 X120.288 Y122.282 E.01688
G1 X120.339 Y122.454 E.00567
G1 X120.303 Y122.521 E.00241
; WIPE_START
M204 S10000
G1 X119.917 Y122.847 E-.192
G1 X119.746 Y122.808 E-.06674
G1 X119.538 Y122.566 E-.12126
; WIPE_END
G1 E-.02 F1800
G1 X114.838 Y128.579 Z8.76 F36000
G1 X103.247 Y143.407 Z8.76
G1 Z8.36
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.588 Y143.407 E.02221
; WIPE_START
G1 X103.247 Y143.407 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X101.325 Y136.021 Z8.76 F36000
G1 X100.144 Y131.48 Z8.76
G1 Z8.36
G1 E.4 F1800
; LINE_WIDTH: 1.05132
G1 F7743.516
G1 X99.739 Y131.606 E.0281
; WIPE_START
G1 X100.144 Y131.48 E-.38
; WIPE_END
M73 P79 R4
G1 E-.02 F1800
G1 X105.823 Y136.579 Z8.76 F36000
G1 X105.862 Y136.615 Z8.76
G1 Z8.36
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.861 Y134.272 E.08944
G2 X107.429 Y130.267 I-3.49 J-3.675 E.16956
G2 X111.966 Y127.924 I-25.539 J-55.005 E.19501
G3 X113.972 Y129.887 I-18.453 J20.868 E.10724
G3 X114.809 Y134.129 I-3.888 J2.97 E.17097
G3 X113.785 Y135.543 I-3.838 J-1.703 E.06715
G2 X111.853 Y137.428 I9.066 J11.22 E.10319
G2 X111.017 Y141.67 I3.888 J2.97 E.17097
G1 X111.15 Y141.945 E.01171
G1 X108.808 Y141.945 E.08944
G1 X120.926 Y124.835 F36000
G1 F13446.283
G3 X118.676 Y124.747 I-1.038 J-2.271 E.08917
G1 X118.825 Y125.174 E.01728
G1 X119.394 Y126.117 E.04203
G2 X121.325 Y128.002 I10.998 J-9.336 E.10319
G3 X122.35 Y129.416 I-2.813 J3.116 E.06715
G3 X121.513 Y133.658 I-4.725 J1.271 E.17097
G3 X119.582 Y135.543 I-10.997 J-9.335 E.10319
G2 X118.557 Y136.957 I2.813 J3.116 E.06715
G2 X120.088 Y141.945 I4.739 J1.275 E.21006
G1 X126.232 Y141.945 E.23457
G1 X126.098 Y141.67 E.0117
G3 X126.935 Y137.428 I4.725 J-1.271 E.17097
G3 X128.866 Y135.543 I10.997 J9.335 E.10319
G2 X129.89 Y134.129 I-2.813 J-3.117 E.06715
G2 X129.054 Y129.887 I-4.725 J-1.271 E.17097
G2 X127.122 Y128.002 I-10.998 J9.336 E.10319
G3 X125.963 Y126.117 I2.423 J-2.789 E.08581
G3 X127.81 Y121.404 I5.258 J-.658 E.20148
G2 X129.696 Y119.448 I-5.285 J-6.984 E.10416
G2 X133.988 Y121.237 I4.789 J-5.447 E.18073
G2 X133.638 Y121.875 I1.622 J1.304 E.02794
G2 X134.475 Y126.117 I4.725 J1.271 E.17097
G2 X136.407 Y128.002 I10.997 J-9.335 E.10319
G3 X137.431 Y129.416 I-2.813 J3.116 E.06715
G3 X136.594 Y133.658 I-4.725 J1.271 E.17097
G3 X134.663 Y135.543 I-10.997 J-9.335 E.10319
G2 X133.638 Y136.957 I2.813 J3.116 E.06715
G2 X135.169 Y141.945 I4.739 J1.275 E.21006
G1 X141.313 Y141.945 E.23457
G3 X141.045 Y141.198 I1.996 J-1.139 E.03046
G3 X142.891 Y136.485 I5.258 J-.658 E.20148
G2 X144.746 Y134.595 I-5.677 J-7.424 E.10146
G2 X148.543 Y138.516 I39.078 J-34.036 E.20849
G2 X150.25 Y141.945 I5.061 J-.38 E.14998
G1 X147.908 Y141.945 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 8.52
; LAYER_HEIGHT: 0.160001
; WIPE_START
G1 F13446.283
G1 X148.908 Y141.945 E-.38
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
G1 X104.704 Y130.892
G1 Z8.52
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X105.084 Y131.37 E.02334
G1 X105.281 Y131.929 E.02263
G1 X105.364 Y132.484 E.0214
G3 X105.372 Y137.349 I-707.69 J3.593 E.18573
G1 X105.364 Y139.349 E.07636
G1 X105.297 Y139.908 E.0215
G1 X105.364 Y140.524 E.02364
G1 X105.377 Y142.443 E.07329
G1 X154.069 Y142.443 E1.859
G3 X143.806 Y132.705 I31.495 J-43.469 E.5417
G3 X136.271 Y120.545 I42.359 J-34.662 E.54775
G1 X135.514 Y120.694 E.02948
G1 X134.434 Y120.764 E.04133
G3 X133.273 Y120.651 I.636 J-12.568 E.04454
G3 X131.228 Y119.92 I1.665 J-7.883 E.08314
G3 X129.672 Y118.769 I4.348 J-7.501 E.07406
G3 X129.012 Y118.015 I7.84 J-7.538 E.03829
G1 X126.663 Y114.849 E.15049
G3 X121.439 Y120.368 I-43.298 J-35.752 E.29038
G3 X119.312 Y122.158 I-381.953 J-451.737 E.10613
G1 X117.78 Y123.444 E.07636
G3 X104.787 Y130.858 I-33.372 J-43.391 E.57293
; WIPE_START
G1 X105.084 Y131.37 E-.22489
G1 X105.22 Y131.755 E-.15511
; WIPE_END
G1 E-.02 F1800
G1 X103.045 Y130.899 Z8.92 F36000
G1 Z8.52
G1 E.4 F1800
G1 F13446.369
G1 X103.619 Y130.905 E.02192
G1 X104.192 Y131.175 E.02419
G1 X104.56 Y131.632 E.02238
G1 X104.749 Y132.21 E.02323
G1 X104.779 Y132.774 E.02155
G3 X104.786 Y137.346 I-664.946 J3.319 E.17457
G1 X104.778 Y139.346 E.07636
G1 X104.705 Y139.901 E.02134
G1 X104.778 Y140.525 E.02399
G1 X104.781 Y142.434 E.0729
G1 X104.654 Y143.012 E.02259
G2 X105.749 Y143.029 I.908 J-23.052 E.04179
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.257 Y132.331 I30.425 J-44.628 E.59929
G3 X136.596 Y119.828 I41.795 J-34.208 E.56156
G1 X136.067 Y119.995 E.02118
G1 X135.342 Y120.13 E.02812
G1 X134.412 Y120.179 E.03558
G3 X133.285 Y120.054 I.63 J-10.839 E.04332
G1 X132.574 Y119.867 E.02806
G3 X130.746 Y118.918 I2.897 J-7.817 E.07883
G1 X130.187 Y118.448 E.0279
G3 X129.065 Y117.104 I6.486 J-6.554 E.06694
G1 X126.682 Y113.891 E.15272
G3 X121.037 Y119.942 I-41.662 J-33.208 E.31624
G3 X118.935 Y121.71 I-523.042 J-619.835 E.10486
G1 X117.404 Y122.996 E.07636
G3 X103.129 Y130.868 I-32.981 J-42.927 E.62474
; WIPE_START
G1 X103.619 Y130.905 E-.18662
G1 X104.079 Y131.122 E-.19338
; WIPE_END
G1 E-.02 F1800
G1 X100.474 Y131.405 Z8.92 F36000
G1 Z8.52
G1 E.4 F1800
; LINE_WIDTH: 1.09722
G1 F7409.116
G1 X100.525 Y131.428 E.0039
; LINE_WIDTH: 1.05141
G1 F7742.78
G1 X100.576 Y131.452 E.00373
; LINE_WIDTH: 1.00561
G1 F8107.913
G1 X100.627 Y131.475 E.00357
; LINE_WIDTH: 0.959806
G1 F8509.19
G1 X100.678 Y131.499 E.0034
; LINE_WIDTH: 0.914003
G1 F8952.256
G1 X100.73 Y131.522 E.00323
; LINE_WIDTH: 0.8682
G1 F9443.994
G1 X100.781 Y131.546 E.00306
; LINE_WIDTH: 0.822396
G1 F9992.895
G1 X100.841 Y131.579 E.00355
; LINE_WIDTH: 0.778404
G1 F10583.718
G1 X100.902 Y131.613 E.00335
; LINE_WIDTH: 0.734411
G1 F10805.192
G1 X100.962 Y131.646 E.00315
; LINE_WIDTH: 0.690419
G1 F11028.932
G1 X101.023 Y131.679 E.00296
; LINE_WIDTH: 0.646426
G1 F12368.886
G1 X101.421 Y131.647 E.01596
G1 F12865.756
G1 X101.719 Y131.622 E.01192
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X103.26 Y131.465 E.05914
G1 X103.511 Y131.481 E.0096
G1 X103.835 Y131.64 E.01376
G1 X104.042 Y131.905 E.01286
G1 X104.179 Y132.344 E.01756
G3 X104.2 Y137.344 I-318.611 J3.885 E.19089
G1 X104.192 Y139.344 E.07636
G1 X104.113 Y139.893 E.02119
G3 X104.192 Y140.526 I-4.568 J.896 E.02436
G1 X104.195 Y142.035 E.05763
G1 X104.196 Y142.435 E.01527
G1 F12956.402
G1 X104.123 Y142.765 E.01289
; LINE_WIDTH: 0.659514
G1 F11793.048
G1 X104.056 Y142.867 E.00499
; LINE_WIDTH: 0.699031
G1 F11384.361
G1 X103.989 Y142.97 E.00531
; LINE_WIDTH: 0.738549
G1 F10982.837
G1 X103.921 Y143.072 E.00563
; LINE_WIDTH: 0.778066
G1 F10588.521
G1 X103.854 Y143.174 E.00594
; LINE_WIDTH: 0.820795
G1 F10013.246
G1 X103.81 Y143.213 E.00302
; LINE_WIDTH: 0.863523
G1 F9497.259
G1 X103.765 Y143.252 E.00319
; LINE_WIDTH: 0.906251
G1 F9031.845
G1 X103.721 Y143.291 E.00335
; LINE_WIDTH: 0.94898
G1 F8609.914
G1 X103.676 Y143.33 E.00352
; LINE_WIDTH: 0.991708
G1 F8225.646
G1 X103.632 Y143.369 E.00368
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.588 Y143.407 E.00385
G1 X103.648 Y143.429 E.00421
; LINE_WIDTH: 0.991708
G1 F8225.646
G1 X103.709 Y143.45 E.00403
; LINE_WIDTH: 0.94898
G1 F8609.914
G1 X103.77 Y143.471 E.00385
; LINE_WIDTH: 0.906251
G1 F9031.845
G1 X103.831 Y143.493 E.00367
; LINE_WIDTH: 0.863523
G1 F9497.259
G1 X103.892 Y143.514 E.00349
; LINE_WIDTH: 0.820795
G1 F10013.246
G1 X103.953 Y143.536 E.00331
; LINE_WIDTH: 0.778066
G1 F10588.521
G1 X104.131 Y143.555 E.00869
; LINE_WIDTH: 0.738549
G1 F11182.705
G1 X104.309 Y143.575 E.00822
; LINE_WIDTH: 0.699031
G1 F11777.435
G1 X104.487 Y143.595 E.00776
; LINE_WIDTH: 0.659514
G1 F12387.538
G1 X104.665 Y143.615 E.0073
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X105.065 Y143.615 E.01527
G1 X156.235 Y143.615 E1.95363
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.707 Y131.956 I29.318 J-43.864 E.60763
G3 X136.912 Y119.083 I41.337 J-33.829 E.5765
M73 P79 R3
G1 X136.092 Y119.382 E.03329
G1 X135.299 Y119.542 E.0309
G1 X134.39 Y119.594 E.03475
G3 X133.434 Y119.488 I1.026 J-13.629 E.03675
G1 X132.716 Y119.299 E.02832
G3 X131.123 Y118.47 I2.874 J-7.473 E.06871
G3 X129.952 Y117.317 I4.347 J-5.585 E.06288
G1 X126.698 Y112.93 E.20855
G3 X120.636 Y119.515 I-43.031 J-33.525 E.34212
G3 X118.559 Y121.261 I-877.617 J-1042.212 E.1036
G1 X117.027 Y122.547 E.07636
G3 X101.56 Y130.802 I-32.893 J-43.01 E.67232
; LINE_WIDTH: 0.646426
G1 F12718.249
G1 X101.309 Y130.89 E.0106
G1 F11807.306
G1 X100.932 Y131.022 E.01596
; LINE_WIDTH: 0.691505
G1 F10499.005
G1 X100.886 Y131.06 E.00256
; LINE_WIDTH: 0.736584
G1 F10310.329
G1 X100.84 Y131.099 E.00273
; LINE_WIDTH: 0.781663
G1 F10123.343
G1 X100.794 Y131.137 E.00291
; LINE_WIDTH: 0.826742
G1 F9938.088
G1 X100.749 Y131.175 E.00308
; LINE_WIDTH: 0.871821
G1 F9403.154
G1 X100.703 Y131.214 E.00326
; LINE_WIDTH: 0.9169
G1 F8922.866
G1 X100.657 Y131.252 E.00343
; LINE_WIDTH: 0.961979
G1 F8489.258
G1 X100.611 Y131.29 E.00361
; LINE_WIDTH: 1.00706
G1 F8095.839
G1 X100.565 Y131.328 E.00379
; LINE_WIDTH: 1.05214
G1 F7737.268
G1 X100.543 Y131.347 E.00195
; WIPE_START
G1 X100.525 Y131.428 E-.03159
G1 X100.576 Y131.452 E-.0214
G1 X100.627 Y131.475 E-.0214
G1 X100.678 Y131.499 E-.0214
G1 X100.73 Y131.522 E-.0214
G1 X100.781 Y131.546 E-.0214
G1 X100.841 Y131.579 E-.02626
G1 X100.902 Y131.613 E-.02626
G1 X100.962 Y131.646 E-.02627
G1 X101.023 Y131.679 E-.02626
G1 X101.38 Y131.65 E-.13638
; WIPE_END
G1 E-.02 F1800
G1 X100.253 Y139.199 Z8.92 F36000
G1 X99.707 Y142.854 Z8.92
G1 Z8.52
G1 E.4 F1800
; LINE_WIDTH: 0.763786
G1 F10795.806
G3 X99.637 Y143.543 I-3.44 J.001 E.03296
; LINE_WIDTH: 0.792656
G1 F10384.799
G1 X100.387 Y143.528 E.03708
; LINE_WIDTH: 0.828496
G1 F9916.138
G1 X100.815 Y143.51 E.02216
; LINE_WIDTH: 0.864336
G1 F9487.952
G1 X101.242 Y143.492 E.02316
; LINE_WIDTH: 0.906214
G1 F9032.233
G1 X101.742 Y143.471 E.02842
; LINE_WIDTH: 0.948091
G1 F8618.284
G1 X102.242 Y143.449 E.02978
; LINE_WIDTH: 0.989969
G1 F8240.616
G1 X102.741 Y143.428 E.03115
; LINE_WIDTH: 1.03185
G1 F7894.658
G1 X103.241 Y143.407 E.03251
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.247 Y143.407 E.00042
G1 X103.241 Y143.407 E.00042
; LINE_WIDTH: 1.03185
G1 F7894.658
G1 X102.823 Y143.312 E.02783
; LINE_WIDTH: 0.996001
G1 F8188.923
G1 X102.406 Y143.218 E.02683
; LINE_WIDTH: 0.960156
G1 F8505.974
G1 X101.918 Y143.107 E.03018
; LINE_WIDTH: 0.918281
G1 F8908.925
G1 X101.43 Y142.997 E.02881
; LINE_WIDTH: 0.876406
G1 F9351.955
G1 X100.943 Y142.886 E.02745
; LINE_WIDTH: 0.834531
G1 F9841.352
G1 X100.455 Y142.776 E.02608
; LINE_WIDTH: 0.792656
G1 F10384.799
G1 X99.797 Y142.852 E.03277
; WIPE_START
G1 X99.637 Y143.543 E-.26933
G1 X99.928 Y143.537 E-.11067
; WIPE_END
G1 E-.02 F1800
G1 X100.113 Y135.907 Z8.92 F36000
G1 X100.201 Y132.245 Z8.92
G1 Z8.52
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X100.663 Y132.189 E.01473
G1 X100.734 Y132.202 E.0023
G1 X100.806 Y132.215 E.0023
G1 X100.877 Y132.228 E.0023
G1 X100.949 Y132.241 E.0023
G1 X101 Y132.251 E.00165
G1 X101.604 Y132.189 E.01922
G1 X102.032 Y132.146 E.01362
G1 X102.46 Y132.102 E.01362
G1 X102.888 Y132.059 E.01362
G1 X103.316 Y132.015 E.01362
G1 X103.483 Y132.066 E.00551
G1 X103.555 Y132.175 E.00415
G1 X103.64 Y132.489 E.0103
G3 X103.648 Y137.342 I-704.374 J3.586 E.15363
G1 X103.64 Y139.342 E.06332
G1 X103.622 Y139.45 E.00347
G1 X103.605 Y139.558 E.00347
G1 X103.588 Y139.666 E.00347
G1 X103.571 Y139.774 E.00347
G1 X103.554 Y139.883 E.00347
G1 X103.554 Y139.89 E.00024
G1 X103.571 Y140.012 E.00388
G1 X103.588 Y140.133 E.00388
G1 X103.604 Y140.254 E.00388
G1 X103.621 Y140.376 E.00388
G3 X103.64 Y140.527 I-1.087 J.211 E.00482
G1 X103.643 Y142.436 E.06045
G1 X103.622 Y142.532 E.0031
G1 X103.456 Y142.659 E.00661
G1 X103.377 Y142.658 E.00251
G1 X100.392 Y142.115 E.09605
G1 X99.677 Y142.238 E.02299
G3 X99.465 Y142.16 I-.038 J-.222 E.00748
G1 X99.096 Y141.717 E.01826
G3 X98.944 Y143.858 I-8.775 J.452 E.06816
G1 X98.938 Y144.167 E.00978
G1 X156.695 Y144.167 E1.8286
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I29.093 J-44.027 E.51053
G3 X137.192 Y118.315 I40.856 J-33.422 E.49166
G1 X136.612 Y118.616 E.02069
G1 X135.912 Y118.859 E.02344
G1 X135.198 Y118.999 E.02306
G1 X134.37 Y119.041 E.02624
G1 X133.607 Y118.959 E.02428
G3 X130.92 Y117.577 I1.103 J-5.448 E.0969
G1 X130.408 Y117.003 E.02433
G3 X130.278 Y116.829 I1.947 J-1.592 E.00688
G1 X126.704 Y112.01 E.18996
G3 X121.199 Y118.226 I-42.915 J-32.456 E.26315
G3 X119.313 Y119.906 I-15.812 J-15.858 E.08002
G1 X116.671 Y122.124 E.10918
G3 X98.936 Y131.038 I-32.227 J-42.017 E.63218
G1 X98.945 Y131.531 E.01561
; LINE_WIDTH: 0.536426
G1 X99.035 Y131.984 E.01511
; LINE_WIDTH: 0.582621
G1 X99.074 Y132.114 E.00484
; LINE_WIDTH: 0.628816
G1 X99.112 Y132.243 E.00524
; LINE_WIDTH: 0.675011
G1 X99.151 Y132.373 E.00565
; LINE_WIDTH: 0.721206
G1 X99.189 Y132.503 E.00606
; LINE_WIDTH: 0.734866
G1 X99.202 Y133.152 E.02965
G1 X99.995 Y133.005 E.03684
G1 X99.91 Y132.501 E.02334
; LINE_WIDTH: 0.720566
G1 X99.948 Y132.454 E.00271
; LINE_WIDTH: 0.674531
G1 X99.985 Y132.406 E.00253
; LINE_WIDTH: 0.628496
G1 X100.023 Y132.359 E.00235
; LINE_WIDTH: 0.582461
G1 X100.061 Y132.312 E.00216
; LINE_WIDTH: 0.536426
G1 X100.12 Y132.284 E.00214
; WIPE_START
M204 S10000
G1 X100.663 Y132.189 E-.20952
G1 X100.734 Y132.202 E-.02758
G1 X100.806 Y132.215 E-.02757
G1 X100.877 Y132.228 E-.02757
G1 X100.949 Y132.241 E-.02758
G1 X101 Y132.251 E-.01984
G1 X101.106 Y132.24 E-.04034
; WIPE_END
G1 E-.02 F1800
G1 X102.543 Y139.736 Z8.92 F36000
G1 X103.247 Y143.407 Z8.92
G1 Z8.52
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.588 Y143.407 E.02221
; WIPE_START
G1 X103.247 Y143.407 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X101.529 Y135.971 Z8.92 F36000
G1 X100.474 Y131.405 Z8.92
G1 Z8.52
G1 E.4 F1800
; LINE_WIDTH: 1.09722
G1 F7409.116
G1 X99.74 Y131.606 E.05273
; WIPE_START
G1 X100.474 Y131.405 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X105.862 Y136.497 Z8.92 F36000
G1 Z8.52
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.862 Y134.155 E.08944
G2 X107.538 Y130.207 I-3.542 J-3.834 E.16873
G2 X111.84 Y128.002 I-29.569 J-62.972 E.18459
G3 X113.901 Y129.887 I-10.514 J13.567 E.10677
G3 X114.969 Y134.129 I-3.893 J3.236 E.17269
G3 X113.986 Y135.543 I-3.351 J-1.282 E.06641
G2 X111.924 Y137.428 I10.514 J13.567 E.10677
G2 X110.856 Y141.67 I3.893 J3.236 E.17269
G1 X110.975 Y141.945 E.01146
G1 X108.632 Y141.945 E.08944
G1 X122.08 Y120.523 F36000
; FEATURE: Bridge
; LINE_WIDTH: 0.737126
G1 F1800
G1 X117.865 Y124.061 E.25208
G1 X117.981 Y124.199 E.00826
G2 X118.294 Y124.501 I1.764 J-1.516 E.01995
G1 X118.373 Y124.552 E.00432
G1 X122.264 Y121.286 E.23274
G1 X122.394 Y121.587 E.015
G1 X122.443 Y121.809 E.0104
G1 X122.465 Y122.035 E.01043
G1 X119.092 Y124.866 E.20178
G1 X119.507 Y124.917 E.01917
G2 X120.351 Y124.727 I-.013 J-2.028 E.03993
G1 X122.607 Y122.833 E.13495
G1 X121.211 Y124.857 F36000
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X119.032 Y125.458 I-1.701 J-1.919 E.08934
G1 X119.465 Y126.117 E.0301
G2 X121.526 Y128.002 I12.575 J-11.681 E.10677
G3 X122.51 Y129.416 I-2.367 J2.696 E.06641
G3 X121.442 Y133.658 I-4.961 J1.006 E.17269
G3 X119.38 Y135.543 I-12.575 J-11.681 E.10677
G2 X118.397 Y136.957 I2.368 J2.696 E.06641
G2 X120.215 Y141.945 I4.987 J1.008 E.21328
G1 X126.056 Y141.945 E.22299
G1 X125.937 Y141.67 E.01146
G3 X127.006 Y137.428 I4.961 J-1.006 E.17269
G3 X129.067 Y135.543 I12.575 J11.681 E.10677
G2 X130.051 Y134.129 I-2.368 J-2.696 E.06641
G2 X128.983 Y129.887 I-4.961 J-1.006 E.17268
G2 X126.921 Y128.002 I-12.575 J11.681 E.10677
G3 X125.937 Y126.588 I2.367 J-2.696 E.06641
G3 X127.006 Y122.347 I4.961 J-1.006 E.17269
G3 X129.067 Y120.462 I12.575 J11.681 E.10677
G2 X129.825 Y119.553 I-2.348 J-2.728 E.04537
G2 X133.81 Y121.217 I4.795 J-5.879 E.16718
G2 X133.478 Y121.875 I1.472 J1.153 E.02833
G2 X134.546 Y126.117 I4.961 J1.006 E.17269
G2 X136.608 Y128.002 I12.576 J-11.682 E.10677
G3 X137.591 Y129.416 I-2.367 J2.696 E.06641
G3 X136.523 Y133.658 I-4.961 J1.006 E.17269
G3 X134.462 Y135.543 I-12.575 J-11.681 E.10677
G2 X133.478 Y136.957 I2.367 J2.696 E.06641
G2 X135.297 Y141.945 I4.987 J1.008 E.21328
G1 X141.137 Y141.945 E.22299
G1 X141.019 Y141.67 E.01146
G3 X142.087 Y137.428 I4.961 J-1.006 E.17269
G3 X144.148 Y135.543 I12.575 J11.681 E.10677
G2 X144.854 Y134.71 I-2.146 J-2.533 E.04185
G2 X148.457 Y138.445 I50.162 J-44.791 E.1982
G2 X150.378 Y141.945 I5.272 J-.616 E.15626
G1 X148.035 Y141.945 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 8.68
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X149.035 Y141.945 E-.38
; WIPE_END
G1 E-.02 F1800
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
G1 X104.473 Y130.984
G1 Z8.68
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.959 Y131.636 E.03106
G3 X105.355 Y133.033 I-8.664 J3.216 E.05547
G3 X105.374 Y136.816 I-290.825 J3.302 E.14446
G1 X105.366 Y138.816 E.07636
G3 X105.143 Y139.903 I-4.873 J-.435 E.04244
G3 X105.35 Y141.464 I-35.383 J5.504 E.06012
G1 X105.367 Y142.433 E.03701
G2 X108.069 Y142.443 I1.72 J-96.86 E.10316
G1 X154.069 Y142.443 E1.75624
G3 X143.803 Y132.702 I31.617 J-43.598 E.54189
G3 X136.271 Y120.545 I41.751 J-34.281 E.54761
G1 X135.454 Y120.705 E.03178
G3 X133.273 Y120.651 I-.891 J-8.104 E.08355
G3 X131.229 Y119.921 I1.666 J-7.886 E.08312
G3 X129.692 Y118.788 I4.398 J-7.579 E.07303
G3 X129.012 Y118.015 I7.734 J-7.488 E.03934
G1 X126.663 Y114.849 E.1505
G3 X121.438 Y120.369 I-43.298 J-35.752 E.29039
G3 X119.312 Y122.158 I-383.452 J-453.52 E.10611
G1 X117.78 Y123.444 E.07636
G3 X104.556 Y130.951 I-33.396 J-43.426 E.58244
; WIPE_START
G1 X104.959 Y131.636 E-.30198
G1 X105.021 Y131.832 E-.07802
; WIPE_END
G1 E-.02 F1800
G1 X102.38 Y131.137 Z9.08 F36000
G1 Z8.68
G1 E.4 F1800
G1 F13446.369
G1 X103.051 Y131.074 E.02573
G1 X103.61 Y131.138 E.0215
G1 X104.109 Y131.434 E.02213
G1 X104.424 Y131.876 E.02071
G3 X104.775 Y133.118 I-8.237 J2.998 E.04934
G3 X104.788 Y136.814 I-456.132 J3.439 E.14112
G1 X104.78 Y138.814 E.07636
G3 X104.549 Y139.897 I-10.682 J-1.716 E.0423
G3 X104.769 Y141.537 I-34.471 J5.469 E.06319
G1 X104.781 Y142.434 E.03424
G1 X104.654 Y143.012 E.02259
G2 X105.749 Y143.029 I.908 J-23.066 E.04179
G1 X155.749 Y143.029 E1.90895
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.033 J-44.202 E.59948
G3 X136.596 Y119.828 I41.785 J-34.197 E.56143
G1 X136.065 Y119.995 E.02125
G1 X135.346 Y120.129 E.0279
G1 X134.412 Y120.179 E.03572
G3 X133.285 Y120.054 I.628 J-10.83 E.04332
M73 P80 R3
G1 X132.564 Y119.864 E.02845
G3 X130.749 Y118.92 I2.943 J-7.876 E.07832
G1 X130.187 Y118.448 E.028
G3 X129.065 Y117.104 I6.48 J-6.55 E.06696
G1 X126.682 Y113.891 E.15272
G3 X121.037 Y119.942 I-41.657 J-33.203 E.31625
G3 X118.935 Y121.71 I-527.937 J-625.663 E.10485
G1 X117.404 Y122.996 E.07636
G3 X102.465 Y131.108 I-32.982 J-42.928 E.65171
; WIPE_START
G1 X103.051 Y131.074 E-.22305
G1 X103.461 Y131.121 E-.15695
; WIPE_END
G1 E-.02 F1800
G1 X99.947 Y131.535 Z9.08 F36000
G1 Z8.68
G1 E.4 F1800
; LINE_WIDTH: 1.03692
G1 F7854.735
G1 X100.024 Y131.561 E.00533
; LINE_WIDTH: 0.990592
G1 F8235.247
G1 X100.101 Y131.587 E.00508
; LINE_WIDTH: 0.944267
G1 F8654.502
G1 X100.179 Y131.612 E.00484
; LINE_WIDTH: 0.897943
G1 F9118.736
G1 X100.256 Y131.638 E.00459
; LINE_WIDTH: 0.851618
G1 F9635.596
G1 X100.333 Y131.664 E.00434
; LINE_WIDTH: 0.805294
G1 F10214.568
G1 X100.411 Y131.689 E.0041
; LINE_WIDTH: 0.75897
G1 F10471.46
G1 X100.488 Y131.715 E.00385
; LINE_WIDTH: 0.712645
G1 F10731.512
G1 X100.565 Y131.741 E.0036
; LINE_WIDTH: 0.666321
G1 F10994.764
G1 X100.643 Y131.766 E.00336
; LINE_WIDTH: 0.619996
G1 F12635.588
G1 X101.124 Y131.844 E.01861
G1 F13446.369
G1 X101.522 Y131.807 E.01527
G1 X103.106 Y131.657 E.06072
G1 X103.437 Y131.698 E.01275
G1 X103.718 Y131.871 E.01259
G1 X103.925 Y132.211 E.01522
G1 X104.162 Y133.012 E.03189
G1 X104.196 Y133.747 E.02807
G3 X104.203 Y136.812 I-721.868 J3.047 E.11703
G1 X104.194 Y138.812 E.07636
G2 X103.955 Y139.891 I23.737 J5.833 E.04222
G3 X104.188 Y141.611 I-28.838 J4.788 E.06627
G1 X104.192 Y142.035 E.01619
G1 X104.196 Y142.435 E.01527
G1 F12956.309
G1 X104.123 Y142.765 E.01289
; LINE_WIDTH: 0.659516
G1 F11792.926
G1 X104.056 Y142.867 E.00499
; LINE_WIDTH: 0.699036
G1 F11384.214
G1 X103.989 Y142.97 E.00531
; LINE_WIDTH: 0.738556
G1 F10982.692
G1 X103.921 Y143.072 E.00563
; LINE_WIDTH: 0.778076
G1 F10588.379
G1 X103.854 Y143.174 E.00594
; LINE_WIDTH: 0.820803
G1 F10013.141
G1 X103.81 Y143.213 E.00302
; LINE_WIDTH: 0.86353
G1 F9497.183
G1 X103.765 Y143.252 E.00319
; LINE_WIDTH: 0.906256
G1 F9031.793
G1 X103.721 Y143.291 E.00335
; LINE_WIDTH: 0.948983
G1 F8609.883
G1 X103.676 Y143.33 E.00352
; LINE_WIDTH: 0.99171
G1 F8225.632
G1 X103.632 Y143.369 E.00368
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.588 Y143.407 E.00385
G1 X103.648 Y143.429 E.00421
; LINE_WIDTH: 0.99171
G1 F8225.632
G1 X103.709 Y143.45 E.00403
; LINE_WIDTH: 0.948983
G1 F8609.883
G1 X103.77 Y143.471 E.00385
; LINE_WIDTH: 0.906256
G1 F9031.793
G1 X103.831 Y143.493 E.00367
; LINE_WIDTH: 0.86353
G1 F9497.183
G1 X103.892 Y143.514 E.00349
; LINE_WIDTH: 0.820803
G1 F10013.141
G1 X103.953 Y143.536 E.00331
; LINE_WIDTH: 0.778076
G1 F10588.379
G1 X104.131 Y143.555 E.00869
; LINE_WIDTH: 0.738556
G1 F11182.586
G1 X104.309 Y143.575 E.00823
; LINE_WIDTH: 0.699036
G1 F11777.313
G1 X104.487 Y143.595 E.00776
; LINE_WIDTH: 0.659516
G1 F12387.447
G1 X104.665 Y143.615 E.0073
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X105.065 Y143.615 E.01527
G1 X156.235 Y143.615 E1.95363
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.706 Y131.955 I29.325 J-43.872 E.60771
G3 X136.912 Y119.083 I41.325 J-33.818 E.57641
G1 X136.092 Y119.382 E.03329
G3 X130.998 Y118.365 I-1.531 J-5.597 E.20563
G3 X129.952 Y117.317 I4.307 J-5.345 E.05663
G1 X126.698 Y112.93 E.20856
G3 X120.636 Y119.516 I-43.032 J-33.526 E.34213
G3 X118.559 Y121.261 I-891.603 J-1058.866 E.10358
G1 X117.027 Y122.547 E.07636
G3 X102.489 Y130.475 I-32.831 J-42.913 E.63471
G1 X101.706 Y130.745 E.0316
G1 X101.328 Y130.874 E.01527
G1 F13309.093
G1 X100.949 Y131.003 E.01527
; LINE_WIDTH: 0.653526
G1 F11917.711
G1 X100.661 Y131.112 E.01244
; LINE_WIDTH: 0.687056
G1 F10897.937
G1 X100.373 Y131.221 E.01312
; LINE_WIDTH: 0.737036
G1 F9923.737
G1 X100.312 Y131.266 E.00346
; LINE_WIDTH: 0.787016
G1 F9691.769
G1 X100.251 Y131.311 E.00371
; LINE_WIDTH: 0.836996
G1 F9462.526
G1 X100.19 Y131.356 E.00396
; LINE_WIDTH: 0.886976
G1 F9236.02
G1 X100.129 Y131.401 E.0042
; LINE_WIDTH: 0.936956
G1 F8724.602
G1 X100.068 Y131.446 E.00445
; LINE_WIDTH: 0.986936
G1 F8266.849
G1 X100.019 Y131.482 E.0038
; WIPE_START
G1 X100.024 Y131.561 E-.03009
G1 X100.101 Y131.587 E-.03098
G1 X100.179 Y131.612 E-.03098
G1 X100.256 Y131.638 E-.03098
G1 X100.333 Y131.664 E-.03098
G1 X100.411 Y131.689 E-.03098
G1 X100.488 Y131.715 E-.03098
G1 X100.565 Y131.741 E-.03098
G1 X100.643 Y131.766 E-.03098
G1 X100.908 Y131.809 E-.10209
; WIPE_END
G1 E-.02 F1800
G1 X100.084 Y139.397 Z9.08 F36000
G1 X99.709 Y142.847 Z9.08
G1 Z8.68
G1 E.4 F1800
; LINE_WIDTH: 0.763946
G1 F10793.439
G3 X99.638 Y143.543 I-3.442 J.001 E.03332
; LINE_WIDTH: 0.792646
G1 F10384.936
G1 X100.387 Y143.528 E.03703
; LINE_WIDTH: 0.828491
G1 F9916.201
G1 X100.815 Y143.51 E.02216
; LINE_WIDTH: 0.864336
G1 F9487.952
G1 X101.242 Y143.492 E.02316
; LINE_WIDTH: 0.906214
G1 F9032.233
G1 X101.742 Y143.471 E.02842
; LINE_WIDTH: 0.948091
G1 F8618.284
G1 X102.242 Y143.449 E.02978
; LINE_WIDTH: 0.989969
G1 F8240.616
G1 X102.741 Y143.428 E.03115
; LINE_WIDTH: 1.03185
G1 F7894.658
G1 X103.241 Y143.407 E.03251
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.247 Y143.407 E.00042
G1 X103.241 Y143.407 E.00042
; LINE_WIDTH: 1.03185
G1 F7894.658
G1 X102.823 Y143.312 E.02783
; LINE_WIDTH: 0.995996
G1 F8188.965
G1 X102.406 Y143.218 E.02683
; LINE_WIDTH: 0.960146
G1 F8506.066
G1 X101.918 Y143.107 E.03018
; LINE_WIDTH: 0.918271
G1 F8909.027
G1 X101.43 Y142.997 E.02881
; LINE_WIDTH: 0.876396
G1 F9352.066
G1 X100.943 Y142.886 E.02745
; LINE_WIDTH: 0.834521
G1 F9841.475
G1 X100.455 Y142.776 E.02608
; LINE_WIDTH: 0.792646
G1 F10384.936
G1 X99.818 Y142.848 E.03171
; LINE_WIDTH: 0.763946
G1 F10793.439
G1 X99.799 Y142.848 E.00088
; WIPE_START
G1 X99.638 Y143.543 E-.27109
G1 X99.925 Y143.537 E-.10891
; WIPE_END
G1 E-.02 F1800
G1 X100.091 Y135.906 Z9.08 F36000
G1 X100.17 Y132.263 Z9.08
G1 Z8.68
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X100.257 Y132.277 E.0028
G1 X100.344 Y132.291 E.0028
G1 X100.431 Y132.305 E.0028
G1 X100.519 Y132.319 E.0028
G1 X100.606 Y132.334 E.0028
G1 X100.693 Y132.348 E.0028
G1 X100.78 Y132.362 E.0028
G1 X100.867 Y132.376 E.0028
G1 F3000
G2 X101.176 Y132.394 I.24 J-1.41 E.00982
G1 X103.158 Y132.208 E.06302
G3 X103.422 Y132.457 I-.019 J.284 E.01247
G1 F3600
G1 X103.448 Y132.546 E.00294
G1 X103.474 Y132.635 E.00294
G1 X103.501 Y132.724 E.00294
G1 X103.527 Y132.813 E.00294
G1 X103.553 Y132.902 E.00294
G1 X103.58 Y132.991 E.00294
G1 X103.606 Y133.08 E.00294
G1 X103.632 Y133.169 E.00294
G3 X103.65 Y136.81 I-279.153 J3.147 E.11526
G1 X103.642 Y138.81 E.06332
G1 X103.621 Y138.886 E.00252
G1 X103.601 Y138.963 E.00252
G1 X103.58 Y139.04 E.00252
G1 X103.56 Y139.117 E.00252
G1 X103.539 Y139.194 E.00252
G1 X103.519 Y139.271 E.00252
G1 X103.498 Y139.348 E.00252
G1 X103.478 Y139.425 E.00252
G1 F3000
G2 X103.394 Y139.886 I2.694 J.725 E.01484
G3 X103.491 Y140.505 I-10.345 J1.925 E.01986
G1 F3600
G1 X103.509 Y140.652 E.00469
G1 X103.528 Y140.799 E.00469
G1 X103.547 Y140.946 E.00469
G1 X103.565 Y141.093 E.00469
G1 X103.584 Y141.24 E.00469
G1 X103.603 Y141.387 E.00469
G1 X103.621 Y141.534 E.00469
G1 X103.64 Y141.681 E.00469
G1 X103.643 Y142.436 E.02392
G1 X103.622 Y142.532 E.0031
G1 X103.456 Y142.659 E.00661
G1 X103.377 Y142.658 E.00252
G1 X100.392 Y142.115 E.09605
G1 X99.712 Y142.232 E.02186
G1 X99.502 Y142.175 E.00689
G1 X99.098 Y141.719 E.01928
G3 X98.944 Y143.859 I-8.667 J.451 E.06809
G1 X98.938 Y144.167 E.00978
G1 X156.695 Y144.167 E1.8286
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I29.093 J-44.027 E.51054
G3 X137.192 Y118.315 I40.292 J-33.085 E.49169
G1 X136.612 Y118.616 E.0207
G1 X135.912 Y118.859 E.02344
G1 X135.138 Y119.01 E.02499
G1 X134.37 Y119.041 E.02433
G1 X133.607 Y118.959 E.02428
G3 X130.92 Y117.577 I1.108 J-5.457 E.0969
G1 X130.409 Y117.003 E.02433
G3 X130.278 Y116.829 I1.948 J-1.592 E.00688
G1 X126.704 Y112.01 E.18996
G3 X121.199 Y118.226 I-42.914 J-32.454 E.26315
G3 X119.312 Y119.907 I-15.815 J-15.86 E.08006
G1 X116.672 Y122.124 E.10914
G3 X98.936 Y131.038 I-32.433 J-42.426 E.63212
G1 X98.945 Y131.53 E.01559
; LINE_WIDTH: 0.536576
G1 X99.037 Y131.985 E.01518
; LINE_WIDTH: 0.582761
G1 X99.075 Y132.114 E.00482
; LINE_WIDTH: 0.628946
G1 X99.114 Y132.244 E.00523
; LINE_WIDTH: 0.675131
G1 X99.153 Y132.373 E.00564
; LINE_WIDTH: 0.721316
G1 X99.191 Y132.502 E.00604
; LINE_WIDTH: 0.733766
G1 X99.203 Y133.15 E.02955
G1 X99.995 Y133.004 E.03673
G1 X99.912 Y132.506 E.02298
; LINE_WIDTH: 0.720026
G1 X99.95 Y132.46 E.0027
; LINE_WIDTH: 0.674164
G1 X99.988 Y132.413 E.00252
; LINE_WIDTH: 0.628301
G1 X100.026 Y132.366 E.00234
; LINE_WIDTH: 0.582439
G1 X100.065 Y132.32 E.00216
; LINE_WIDTH: 0.536576
G1 X100.091 Y132.306 E.00097
; WIPE_START
M204 S10000
G1 X100.257 Y132.277 E-.06416
G1 X100.344 Y132.291 E-.03355
G1 X100.431 Y132.305 E-.03355
G1 X100.519 Y132.319 E-.03355
G1 X100.606 Y132.334 E-.03355
G1 X100.693 Y132.348 E-.03355
G1 X100.78 Y132.362 E-.03355
G1 X100.867 Y132.376 E-.03355
G1 X101.08 Y132.389 E-.081
; WIPE_END
G1 E-.02 F1800
G1 X102.553 Y139.878 Z9.08 F36000
G1 X103.247 Y143.407 Z9.08
G1 Z8.68
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.03444
G1 F7874.213
G1 X103.588 Y143.407 E.02221
; WIPE_START
G1 X103.247 Y143.407 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X108.441 Y141.945 Z9.08 F36000
G1 Z8.68
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X110.783 Y141.945 E.08944
G1 X110.683 Y141.67 E.01121
G3 X111.993 Y137.428 I5.256 J-.699 E.17488
G3 X114.208 Y135.543 I12.898 J12.913 E.11118
G2 X115.143 Y134.129 I-1.968 J-2.317 E.06558
G2 X113.833 Y129.887 I-5.256 J-.699 E.17488
G2 X111.706 Y128.073 I-12.468 J12.457 E.10685
G3 X107.664 Y130.155 I-26.576 J-46.621 E.17366
G3 X105.861 Y134.09 I-5.585 J-.178 E.16968
G3 X105.873 Y136.433 I-126.257 J1.815 E.08944
G1 X122.333 Y122.236 F36000
; FEATURE: Floating vertical shell
; LINE_WIDTH: 0.619996
G1 F12000
G1 X122.282 Y121.738 E.01911
G1 X122.064 Y121.227 E.02118
G1 X121.779 Y120.849 E.01808
G1 X118.023 Y123.998 E.18713
G1 X118.448 Y124.414 E.02274
G1 X118.826 Y124.632 E.01665
G1 X119.371 Y124.79 E.02167
G2 X120.868 Y124.377 I.243 J-2.042 E.06078
G1 X121.62 Y123.755 E.03724
G1 X121.96 Y123.375 E.0195
G1 X122.184 Y122.97 E.01765
G1 X122.309 Y122.515 E.01801
G1 X122.325 Y122.325 E.00728
G1 X121.664 Y122.18 F36000
; LINE_WIDTH: 0.796904
G1 F10326.954
G1 X121.611 Y121.867 E.01579
G2 X119.073 Y124.002 I577.663 J689.211 E.16489
G1 X119.436 Y124.117 E.01892
G1 X119.893 Y124.107 E.02273
G2 X120.438 Y123.857 I-.328 J-1.431 E.03001
G1 X121.145 Y123.269 E.0457
G2 X121.656 Y122.27 I-.819 J-1.049 E.05762
G1 X121.168 Y124.892 F36000
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X118.942 Y125.3 I-1.525 J-2.046 E.0895
G1 X119.533 Y126.117 E.03852
G2 X121.749 Y128.002 I12.898 J-12.913 E.11118
G3 X122.684 Y129.416 I-1.968 J2.317 E.06558
G3 X121.373 Y133.658 I-5.256 J.699 E.17488
G3 X119.158 Y135.543 I-12.9 J-12.915 E.11118
G2 X118.223 Y136.957 I1.968 J2.317 E.06558
G2 X120.33 Y141.945 I5.302 J.7 E.21679
G1 X125.864 Y141.945 E.2113
G1 X125.764 Y141.67 E.01121
G3 X127.074 Y137.428 I5.256 J-.699 E.17488
G3 X129.289 Y135.543 I12.899 J12.915 E.11118
G2 X130.224 Y134.129 I-1.968 J-2.317 E.06558
G2 X128.914 Y129.887 I-5.256 J-.699 E.17488
G2 X126.699 Y128.002 I-12.898 J12.913 E.11118
G3 X125.764 Y126.588 I1.968 J-2.317 E.06558
G3 X127.074 Y122.347 I5.256 J-.699 E.17488
G3 X129.289 Y120.462 I12.901 J12.916 E.11118
G2 X129.956 Y119.668 I-1.649 J-2.063 E.03985
G2 X133.611 Y121.195 I4.706 J-6.125 E.15295
G2 X133.305 Y121.875 I1.338 J1.013 E.02874
G2 X134.615 Y126.117 I5.256 J.699 E.17488
G2 X136.83 Y128.002 I12.898 J-12.913 E.11118
G3 X137.765 Y129.416 I-1.968 J2.317 E.06558
G3 X136.455 Y133.658 I-5.256 J.699 E.17488
G3 X134.239 Y135.543 I-12.9 J-12.915 E.11118
G2 X133.305 Y136.957 I1.967 J2.317 E.06558
G2 X135.411 Y141.945 I5.302 J.7 E.21679
G1 X140.946 Y141.945 E.2113
G1 X140.845 Y141.67 E.01121
G3 X142.155 Y137.428 I5.256 J-.699 E.17488
G3 X144.371 Y135.543 I12.897 J12.912 E.11118
G2 X144.973 Y134.849 I-1.432 J-1.851 E.03531
G2 X148.356 Y138.348 I62.107 J-56.658 E.18584
G2 X150.493 Y141.945 I5.7 J-.952 E.16346
G1 X148.15 Y141.945 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 8.84
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X149.15 Y141.945 E-.38
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
G1 X105.267 Y142.443
G1 Z8.84
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X154.069 Y142.443 E1.86321
G3 X143.803 Y132.702 I31.794 J-43.784 E.54187
G3 X136.271 Y120.545 I41.697 J-34.248 E.54762
G1 X135.454 Y120.705 E.03179
G3 X133.274 Y120.651 I-.891 J-8.109 E.08348
G3 X131.226 Y119.919 I1.672 J-7.906 E.08329
G3 X129.674 Y118.771 I4.37 J-7.528 E.07388
G3 X129.011 Y118.015 I7.831 J-7.534 E.0384
G1 X126.663 Y114.849 E.15048
G3 X121.438 Y120.369 I-43.3 J-35.754 E.29039
G3 X119.312 Y122.158 I-380.518 J-450.034 E.10611
G1 X117.78 Y123.444 E.07636
G3 X104.153 Y131.107 I-33.69 J-43.961 E.59891
G1 X104.579 Y131.455 E.021
G1 X104.854 Y131.967 E.02219
G3 X105.346 Y133.683 I-10.795 J4.019 E.06822
G3 X105.375 Y136.409 I-103.699 J2.493 E.10406
G1 X105.368 Y138.409 E.07636
G1 X105.342 Y138.721 E.01196
G3 X105.006 Y139.771 I-13.855 J-3.846 E.04212
G1 X105.003 Y140.035 E.01007
G3 X105.287 Y142.09 I-40.005 J6.572 E.0792
G1 X105.272 Y142.353 E.01008
; WIPE_START
G1 X106.272 Y142.355 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X104.479 Y143.029 Z9.24 F36000
G1 Z8.84
G1 E.4 F1800
G1 F13446.369
G1 X155.749 Y143.029 E1.95741
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.047 J-44.217 E.59947
G3 X136.596 Y119.828 I41.158 J-33.813 E.56148
G1 X136.064 Y119.995 E.02127
G1 X135.346 Y120.129 E.02788
G1 X134.413 Y120.179 E.03569
G3 X133.287 Y120.054 I.622 J-10.789 E.04327
G3 X131.534 Y119.421 I1.734 J-7.539 E.07135
G3 X130.092 Y118.36 I3.941 J-6.866 E.06847
G3 X129.065 Y117.104 I7.128 J-6.875 E.06202
G1 X126.682 Y113.891 E.15272
G3 X121.037 Y119.942 I-41.661 J-33.207 E.31625
G3 X118.935 Y121.709 I-525.011 J-622.187 E.10484
G1 X117.404 Y122.995 E.07636
G3 X101.754 Y131.36 I-33.317 J-43.516 E.68047
G2 X102.903 Y131.266 I-10.245 J-132.137 E.044
G1 X103.579 Y131.381 E.02619
G1 X104.11 Y131.807 E.026
G3 X104.741 Y133.645 I-6.058 J3.106 E.07447
G1 X104.783 Y134.245 E.02296
G3 X104.79 Y136.406 I-164.352 J1.588 E.08252
G1 X104.782 Y138.406 E.07636
G1 X104.699 Y138.87 E.01798
G1 X104.451 Y139.55 E.02763
G1 X104.389 Y139.897 E.01348
G3 X104.706 Y142.166 I-43.581 J7.259 E.08746
G1 X104.678 Y142.671 E.01935
G1 X104.523 Y142.95 E.01216
G1 X103.94 Y142.835 F36000
G1 F10494.412
G1 X103.914 Y142.883 E.00207
; LINE_WIDTH: 0.66632
G1 F10322.796
G1 X103.879 Y142.926 E.00229
; LINE_WIDTH: 0.712644
G1 F10148.794
G1 X103.844 Y142.97 E.00245
; LINE_WIDTH: 0.758967
G1 F9976.271
G1 X103.81 Y143.013 E.00262
; LINE_WIDTH: 0.805291
G1 F9805.227
G1 X103.775 Y143.056 E.00279
; LINE_WIDTH: 0.851614
G1 F9635.643
G1 X103.74 Y143.1 E.00296
; LINE_WIDTH: 0.897938
G1 F9118.787
G1 X103.706 Y143.143 E.00313
; LINE_WIDTH: 0.944262
G1 F8654.555
G1 X103.671 Y143.186 E.00329
; LINE_WIDTH: 0.990585
G1 F8235.302
G1 X103.636 Y143.23 E.00346
; LINE_WIDTH: 1.03691
G1 F7854.792
G1 X103.602 Y143.273 E.00363
; LINE_WIDTH: 1.08323
G1 F7507.891
G1 X103.567 Y143.316 E.0038
; LINE_WIDTH: 1.12956
G1 F7190.336
G1 X103.532 Y143.36 E.00396
G1 X103.595 Y143.383 E.00479
; LINE_WIDTH: 1.08323
G1 F7507.891
G1 X103.658 Y143.406 E.00458
; LINE_WIDTH: 1.03691
G1 F7854.792
G1 X103.721 Y143.429 E.00438
; LINE_WIDTH: 0.990585
G1 F8235.302
G1 X103.784 Y143.452 E.00418
; LINE_WIDTH: 0.944262
G1 F8654.555
G1 X103.847 Y143.476 E.00398
; LINE_WIDTH: 0.897938
G1 F9118.787
G1 X103.91 Y143.499 E.00377
; LINE_WIDTH: 0.851614
G1 F9635.643
G1 X103.973 Y143.522 E.00357
; LINE_WIDTH: 0.805291
G1 F10214.611
G1 X104.035 Y143.545 E.00337
; LINE_WIDTH: 0.758967
G1 F10425.586
G1 X104.098 Y143.568 E.00317
; LINE_WIDTH: 0.712644
G1 F10638.717
G1 X104.161 Y143.591 E.00296
; LINE_WIDTH: 0.66632
G1 F10854.016
G1 X104.224 Y143.615 E.00276
; LINE_WIDTH: 0.619996
G1 F12183.608
G1 X104.624 Y143.615 E.01527
G1 F13446.369
G1 X105.024 Y143.615 E.01527
G1 X156.235 Y143.615 E1.9552
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.706 Y131.955 I29.319 J-43.865 E.60771
G3 X136.912 Y119.083 I40.752 J-33.472 E.57645
G1 X136.091 Y119.382 E.03333
G3 X132.725 Y119.301 I-1.546 J-5.715 E.13038
G3 X131.125 Y118.472 I2.846 J-7.447 E.06892
G3 X129.952 Y117.317 I4.339 J-5.58 E.063
G1 X126.698 Y112.93 E.20855
G3 X120.636 Y119.516 I-43.033 J-33.527 E.34213
G3 X118.559 Y121.261 I-886.122 J-1052.35 E.10357
G1 X117.027 Y122.547 E.07636
G3 X100.384 Y131.183 I-32.728 J-42.72 E.71955
; LINE_WIDTH: 0.619536
G1 F13456.938
G1 X99.867 Y131.339 E.02059
; LINE_WIDTH: 0.619076
G1 F13467.525
G1 X99.493 Y131.453 E.01491
G1 X99.621 Y132.292 E.03236
G1 X99.908 Y131.945 E.01717
G1 X100.037 Y131.894 E.00529
; LINE_WIDTH: 0.619536
G1 F13456.938
G1 X100.418 Y131.873 E.01455
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X101.021 Y131.994 E.02349
G2 X102.952 Y131.849 I-1.706 J-35.753 E.07394
G1 X103.351 Y131.921 E.01548
G1 X103.65 Y132.169 E.01484
G1 X103.953 Y132.925 E.03109
G1 X104.173 Y133.787 E.03397
G3 X104.204 Y136.404 I-113.736 J2.649 E.09992
G1 X104.197 Y138.404 E.07636
G1 X104.149 Y138.669 E.01026
G1 X103.883 Y139.397 E.02959
G1 X103.795 Y139.891 E.01918
G3 X104.126 Y142.241 I-44.785 J7.5 E.09062
G1 X104.109 Y142.53 E.01104
G1 X103.984 Y142.757 E.00989
; WIPE_START
G1 X103.914 Y142.883 E-.05483
G1 X103.879 Y142.926 E-.02109
G1 X103.844 Y142.97 E-.02109
G1 X103.81 Y143.013 E-.02109
G1 X103.775 Y143.056 E-.0211
G1 X103.74 Y143.1 E-.02109
G1 X103.706 Y143.143 E-.02109
G1 X103.671 Y143.186 E-.02109
G1 X103.636 Y143.23 E-.0211
G1 X103.602 Y143.273 E-.02109
G1 X103.567 Y143.316 E-.02109
G1 X103.532 Y143.36 E-.0211
G1 X103.595 Y143.383 E-.02547
G1 X103.658 Y143.406 E-.02547
G1 X103.721 Y143.429 E-.02547
G1 X103.762 Y143.444 E-.01671
; WIPE_END
G1 E-.02 F1800
G1 X102.91 Y143.377 Z9.24 F36000
G1 Z8.84
G1 E.4 F1800
; LINE_WIDTH: 1.0952
G1 F7423.223
G1 X102.654 Y143.314 E.01825
; LINE_WIDTH: 1.06362
G1 F7650.98
G1 X102.398 Y143.252 E.0177
; LINE_WIDTH: 1.03204
G1 F7893.154
G1 X102.009 Y143.156 E.02602
; LINE_WIDTH: 0.98416
G1 F8291.01
G1 X101.621 Y143.061 E.02477
; LINE_WIDTH: 0.936284
G1 F8731.103
G1 X101.232 Y142.966 E.02352
; LINE_WIDTH: 0.888408
G1 F9220.534
G1 X100.844 Y142.871 E.02227
; LINE_WIDTH: 0.840532
G1 F9768.097
G1 X100.455 Y142.776 E.02102
; LINE_WIDTH: 0.792656
G1 F10384.799
G1 X100.055 Y142.791 E.01977
G1 X99.714 Y142.803 E.01687
; LINE_WIDTH: 0.764106
G1 F10791.071
G3 X99.64 Y143.542 I-4.197 J-.052 E.0354
; LINE_WIDTH: 0.792656
G1 F10384.799
G1 X100.387 Y143.528 E.03697
; LINE_WIDTH: 0.824231
G1 F9969.679
G1 X100.651 Y143.512 E.01359
; LINE_WIDTH: 0.855806
G1 F9586.474
G1 X100.914 Y143.497 E.01413
; LINE_WIDTH: 0.903684
G1 F9058.514
G1 X101.313 Y143.473 E.02267
; LINE_WIDTH: 0.951562
G1 F8585.67
G1 X101.712 Y143.449 E.02392
; LINE_WIDTH: 0.99944
G1 F8159.743
G1 X102.112 Y143.425 E.02517
; LINE_WIDTH: 1.04732
G1 F7774.077
G1 X102.511 Y143.401 E.02641
; LINE_WIDTH: 1.0952
G1 F7423.223
G1 X102.82 Y143.382 E.02144
; WIPE_START
G1 X102.654 Y143.314 E-.06833
G1 X102.398 Y143.252 E-.10025
G1 X102.009 Y143.156 E-.152
G1 X101.857 Y143.119 E-.05941
; WIPE_END
G1 E-.02 F1800
G1 X103.516 Y142.499 Z9.24 F36000
G1 Z8.84
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.384 Y142.566 E.00469
G1 X103.052 Y142.599 E.01056
G1 X100.392 Y142.115 E.08559
G1 X99.866 Y142.205 E.01691
G1 X99.723 Y142.183 E.00458
G1 X99.271 Y141.926 E.01647
G1 X99.1 Y141.722 E.00842
G3 X98.944 Y143.859 I-8.563 J.449 E.06802
G1 X98.938 Y144.167 E.00978
G1 X156.695 Y144.167 E1.8286
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.617 J-43.502 E.51059
G3 X137.192 Y118.315 I40.652 J-33.3 E.49167
G1 X136.614 Y118.615 E.0206
G1 X135.911 Y118.859 E.02356
G1 X135.138 Y119.01 E.02495
G1 X134.37 Y119.041 E.02431
G1 X133.609 Y118.959 E.02423
G3 X130.92 Y117.577 I1.105 J-5.457 E.09697
G1 X130.408 Y117.002 E.02434
G3 X130.278 Y116.829 I1.943 J-1.589 E.00687
G1 X126.704 Y112.01 E.18996
G3 X121.199 Y118.226 I-42.916 J-32.456 E.26315
G3 X119.316 Y119.903 I-15.78 J-15.823 E.07987
G1 X116.672 Y122.124 E.10933
G3 X98.936 Y131.038 I-32.26 J-42.081 E.63218
G1 X98.945 Y131.53 E.01557
; LINE_WIDTH: 0.565954
M73 P81 R3
G1 X99.005 Y131.772 E.00865
; LINE_WIDTH: 0.611911
G1 X99.065 Y132.015 E.0094
; LINE_WIDTH: 0.657869
G1 X99.124 Y132.257 E.01015
; LINE_WIDTH: 0.703826
G1 X99.184 Y132.5 E.0109
; LINE_WIDTH: 0.733676
G1 X99.205 Y133.148 E.02956
G1 X99.995 Y133.002 E.03661
; LINE_WIDTH: 0.735836
G1 X99.935 Y132.589 E.01907
G1 X99.988 Y132.555 E.00287
; LINE_WIDTH: 0.692668
G1 X100.041 Y132.521 E.00269
; LINE_WIDTH: 0.6495
G1 X100.093 Y132.488 E.00251
; LINE_WIDTH: 0.606332
G1 X100.146 Y132.454 E.00234
; LINE_WIDTH: 0.563164
G1 X100.199 Y132.42 E.00216
; LINE_WIDTH: 0.519996
G1 X100.285 Y132.434 E.00277
G1 X100.372 Y132.448 E.00277
G1 X100.458 Y132.462 E.00277
G1 X100.545 Y132.476 E.00277
G1 X100.631 Y132.49 E.00277
G1 X100.718 Y132.504 E.00277
G1 X100.804 Y132.518 E.00277
G1 X100.891 Y132.532 E.00277
G1 F3000
G2 X101.214 Y132.553 I.26 J-1.539 E.01028
G1 X103 Y132.4 E.05674
G1 X103.115 Y132.421 E.00372
G1 X103.155 Y132.464 E.00187
G1 F2475
G1 X103.231 Y132.549 E.0036
G1 X103.441 Y133.142 E.01992
G1 F3000
G1 X103.463 Y133.229 E.00283
G1 F3600
G1 X103.484 Y133.315 E.00283
G1 X103.506 Y133.402 E.00283
G1 X103.528 Y133.489 E.00283
G1 X103.55 Y133.575 E.00283
G1 X103.572 Y133.662 E.00283
G1 X103.593 Y133.748 E.00283
G1 X103.615 Y133.835 E.00283
G1 X103.637 Y133.922 E.00283
G3 X103.651 Y136.402 I-333.783 J3.163 E.07853
G1 X103.644 Y138.402 E.06332
G1 X103.617 Y138.481 E.00262
G1 X103.59 Y138.559 E.00262
G1 X103.563 Y138.637 E.00262
G1 X103.536 Y138.716 E.00262
G1 X103.509 Y138.794 E.00262
G1 X103.482 Y138.872 E.00262
G1 X103.455 Y138.951 E.00262
G1 X103.428 Y139.029 E.00262
G1 F3000
G2 X103.234 Y139.886 I6.789 J1.987 E.02782
G3 X103.395 Y140.915 I-19.488 J3.569 E.03299
G1 F3600
G1 X103.418 Y141.09 E.00558
G1 X103.441 Y141.264 E.00558
G1 X103.464 Y141.439 E.00558
G1 X103.486 Y141.614 E.00558
G1 X103.509 Y141.789 E.00558
G1 X103.532 Y141.963 E.00558
G1 X103.555 Y142.138 E.00558
G3 X103.555 Y142.417 I-.394 J.14 E.00901
; WIPE_START
M204 S10000
G1 X103.384 Y142.566 E-.08614
G1 X103.052 Y142.599 E-.1268
G1 X102.619 Y142.521 E-.16707
; WIPE_END
G1 E-.02 F1800
G1 X102.91 Y143.377 Z9.24 F36000
G1 Z8.84
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 1.12324
G1 F7232.069
G1 X103.491 Y143.363 E.04126
; LINE_WIDTH: 1.12956
G1 F7190.336
G1 X103.532 Y143.36 E.00292
; WIPE_START
G1 X103.491 Y143.363 E-.02501
G1 X102.91 Y143.377 E-.35499
; WIPE_END
G1 E-.02 F1800
G1 X108.229 Y141.945 Z9.24 F36000
G1 Z8.84
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X110.572 Y141.945 E.08944
G1 X110.493 Y141.67 E.01095
G3 X112.059 Y137.428 I5.631 J-.331 E.17761
G3 X114.461 Y135.543 I13.975 J15.335 E.11669
G2 X115.332 Y134.129 I-1.608 J-1.966 E.06459
G2 X113.767 Y129.887 I-5.631 J-.331 E.17761
G2 X111.565 Y128.15 I-12.974 J14.176 E.10715
G3 X107.807 Y130.091 I-29.953 J-53.38 E.16153
G3 X105.866 Y133.989 I-5.814 J-.464 E.17037
G1 X105.866 Y136.332 E.08944
G1 X121.081 Y124.917 F36000
G1 F13446.283
G3 X118.845 Y125.131 I-1.342 J-2.233 E.08864
G1 X119.6 Y126.117 E.04739
G2 X122.002 Y128.002 I13.974 J-15.333 E.11669
G3 X122.873 Y129.416 I-1.607 J1.966 E.06459
G3 X121.307 Y133.658 I-5.631 J.331 E.17761
G3 X118.905 Y135.543 I-13.975 J-15.335 E.11669
G2 X118.034 Y136.957 I1.608 J1.966 E.06459
G2 X120.469 Y141.945 I5.744 J.285 E.22118
G1 X125.653 Y141.945 E.19793
G1 X125.575 Y141.67 E.01095
G3 X127.14 Y137.428 I5.631 J-.331 E.17761
G3 X129.542 Y135.543 I13.974 J15.333 E.11669
G2 X130.414 Y134.129 I-1.608 J-1.966 E.06459
G2 X128.848 Y129.887 I-5.631 J-.331 E.17761
G2 X126.446 Y128.002 I-13.974 J15.333 E.11669
G3 X125.575 Y126.588 I1.608 J-1.966 E.06459
G3 X127.14 Y122.347 I5.631 J-.331 E.17761
G3 X129.542 Y120.462 I13.975 J15.335 E.11669
G2 X130.115 Y119.797 I-1.067 J-1.499 E.03383
G2 X133.387 Y121.17 I4.381 J-5.852 E.13685
G2 X133.115 Y121.875 I1.226 J.878 E.02916
G2 X134.681 Y126.117 I5.631 J.331 E.17761
G2 X137.083 Y128.002 I13.974 J-15.333 E.11669
G3 X137.954 Y129.416 I-1.608 J1.966 E.06459
G3 X136.389 Y133.658 I-5.631 J.331 E.17761
G3 X133.986 Y135.543 I-13.975 J-15.335 E.11669
G2 X133.115 Y136.957 I1.608 J1.966 E.06459
G2 X135.55 Y141.945 I5.744 J.285 E.22118
G1 X140.735 Y141.945 E.19793
G1 X140.656 Y141.67 E.01095
G3 X142.222 Y137.428 I5.631 J-.331 E.17761
G3 X144.624 Y135.543 I13.974 J15.333 E.11669
G2 X145.122 Y135.005 I-.845 J-1.283 E.02825
G2 X148.213 Y138.219 I39.644 J-35.05 E.17029
G2 X150.632 Y141.945 I6.167 J-1.354 E.17333
G1 X148.289 Y141.945 E.08944
G1 X121.906 Y123.373 F36000
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X122.122 Y122.905 E.01968
G1 X122.205 Y122.453 E.01751
G1 X122.205 Y122.169 E.01087
G1 X122.108 Y121.67 E.01941
G1 X121.908 Y121.266 E.01722
G1 X121.661 Y120.948 E.01535
G1 X118.158 Y123.887 E.17456
G1 X118.523 Y124.279 E.02043
G1 X118.911 Y124.513 E.0173
G1 X119.401 Y124.67 E.01966
G1 X119.832 Y124.7 E.01652
G2 X120.991 Y124.274 I-.186 J-2.296 E.04773
G1 X121.571 Y123.793 E.02877
G1 X121.85 Y123.443 E.01705
G1 X121.316 Y123.057 F36000
; LINE_WIDTH: 0.794523
G1 F10359.297
G2 X121.491 Y121.964 I-1.074 J-.732 E.05664
G2 X119.197 Y123.898 I278.734 J332.983 E.14869
G2 X120.56 Y123.754 I.559 J-1.234 E.07123
G1 X121.077 Y123.324 E.03332
G1 X121.256 Y123.124 E.01332
; CHANGE_LAYER
; Z_HEIGHT: 9
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F10359.297
G1 X121.077 Y123.324 E-.10213
G1 X120.56 Y123.754 E-.25548
G1 X120.509 Y123.784 E-.02239
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
G1 X105.366 Y135.666
G1 Z9
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X105.37 Y138.02 E.0899
G1 X105.312 Y138.487 E.01794
G2 X104.845 Y139.781 I24.762 J9.669 E.05255
G1 X104.836 Y139.945 E.00627
G3 X105.127 Y141.893 I-30.266 J5.517 E.07523
G1 X105.096 Y142.443 E.02102
G1 X154.069 Y142.443 E1.86973
G3 X143.803 Y132.701 I31.812 J-43.804 E.54189
G3 X136.271 Y120.545 I41.753 J-34.282 E.54759
G1 X135.454 Y120.705 E.03179
G3 X133.274 Y120.651 I-.889 J-8.189 E.08347
G1 X132.429 Y120.434 E.03332
G3 X130.601 Y119.534 I2.481 J-7.342 E.078
G1 X129.81 Y118.896 E.03881
G1 X129.117 Y118.145 E.039
G1 X126.663 Y114.849 E.15691
G3 X121.434 Y120.373 I-43.304 J-35.756 E.29061
G3 X119.312 Y122.158 I-422.811 J-500.412 E.10589
G1 X117.78 Y123.444 E.07636
G3 X103.921 Y131.198 I-33.709 J-43.99 E.6084
G1 X104.478 Y131.724 E.02923
G1 X104.691 Y132.146 E.01808
G1 X104.975 Y132.934 E.03197
G3 X105.364 Y134.528 I-18.738 J5.421 E.06266
G1 X105.366 Y135.576 E.03999
G1 X104.781 Y135.258 F36000
G1 F13446.369
G1 X104.784 Y138.018 E.10539
G1 X104.744 Y138.345 E.01255
G2 X104.281 Y139.623 I19.795 J7.891 E.05192
G1 X104.243 Y139.953 E.01267
G3 X104.546 Y141.97 I-31.054 J5.693 E.0779
G1 X104.518 Y142.478 E.0194
G1 X104.213 Y143.029 E.02406
G1 X155.749 Y143.029 E1.9676
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I29.804 J-43.953 E.59953
G3 X136.596 Y119.828 I41.787 J-34.198 E.56141
G1 X136.064 Y119.995 E.02126
G1 X135.346 Y120.129 E.02789
G3 X133.382 Y120.076 I-.778 J-7.516 E.07522
G3 X131.533 Y119.42 I1.641 J-7.566 E.07511
G1 X130.909 Y119.036 E.028
G1 X130.187 Y118.448 E.03553
G3 X129.065 Y117.104 I6.482 J-6.551 E.06695
G1 X126.682 Y113.891 E.15272
G3 X121.034 Y119.945 I-41.664 J-33.21 E.31641
G3 X118.935 Y121.709 I-582.542 J-690.711 E.10469
G1 X117.404 Y122.995 E.07636
G3 X101.16 Y131.55 I-32.976 J-42.917 E.70433
G1 X101.111 Y131.566 E.00194
G2 X102.78 Y131.446 I-1.314 J-29.874 E.06387
G1 X103.528 Y131.62 E.02935
G1 X103.991 Y132.049 E.0241
G3 X104.424 Y133.133 I-8.677 J4.098 E.04458
G3 X104.78 Y134.573 I-24.119 J6.733 E.05663
G1 X104.781 Y135.168 E.02274
G1 X104.199 Y134.858 F36000
G1 F13446.369
G1 X104.199 Y138.016 E.12057
G1 X104.141 Y138.307 E.01131
G1 X103.82 Y139.098 E.0326
G1 X103.664 Y139.655 E.02208
G1 X103.68 Y140.137 E.01843
G3 X103.966 Y142.047 I-29.118 J5.325 E.07375
G1 X103.949 Y142.337 E.01107
G1 X103.753 Y142.691 E.01546
; LINE_WIDTH: 0.647541
G1 F12842.363
G1 X103.526 Y142.82 E.01045
; LINE_WIDTH: 0.675086
G1 F12290.286
G1 X103.298 Y142.949 E.01092
G1 X102.903 Y142.988 E.01659
; LINE_WIDTH: 0.649291
G1 F12805.816
G1 X102.508 Y143.026 E.01592
; LINE_WIDTH: 0.627696
G1 F13256.016
G1 X102.266 Y143.022 E.00933
; LINE_WIDTH: 0.668936
G1 F12409.393
G1 X101.814 Y142.961 E.01891
; LINE_WIDTH: 0.710176
G1 F11652.167
G1 X101.361 Y142.899 E.02013
; LINE_WIDTH: 0.751416
G1 F10982.039
G1 X100.908 Y142.838 E.02136
; LINE_WIDTH: 0.792656
G1 F10384.799
G1 X100.455 Y142.776 E.02259
G1 X100.035 Y142.807 E.02081
; LINE_WIDTH: 0.768116
G1 F10732.101
G1 X99.722 Y142.753 E.01521
G3 X99.643 Y143.54 I-4.454 J-.05 E.03791
; LINE_WIDTH: 0.775776
G1 F10621.224
G1 X100.189 Y143.537 E.02641
; LINE_WIDTH: 0.792656
G1 F10384.799
G3 X100.844 Y143.549 I.258 J3.723 E.03239
; LINE_WIDTH: 0.751416
G1 F10982.039
G1 X101.3 Y143.569 E.02136
; LINE_WIDTH: 0.710176
G1 F11652.167
G1 X101.757 Y143.59 E.02013
; LINE_WIDTH: 0.668936
G1 F12409.393
G1 X102.213 Y143.611 E.01891
; LINE_WIDTH: 0.675086
G1 F12290.286
G3 X103.742 Y143.601 I.854 J13.82 E.06389
; LINE_WIDTH: 0.647541
G1 F12842.363
G1 X104.144 Y143.615 E.0161
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.544 Y143.615 E.01527
G1 X156.235 Y143.615 E1.97351
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.326 J-43.873 E.60774
G3 X136.912 Y119.083 I40.798 J-33.498 E.57642
G1 X136.092 Y119.382 E.03329
G3 X134.392 Y119.594 I-1.769 J-7.276 E.06555
G3 X133.436 Y119.488 I1.017 J-13.594 E.03673
G1 X132.72 Y119.3 E.02828
G3 X131.124 Y118.471 I2.872 J-7.479 E.06878
G3 X129.952 Y117.317 I4.344 J-5.584 E.06293
G1 X126.698 Y112.93 E.20856
G3 X120.634 Y119.517 I-43.034 J-33.528 E.34222
G3 X118.559 Y121.261 I-988.547 J-1174.338 E.10349
G1 X117.027 Y122.547 E.07636
G3 X100.977 Y130.993 I-32.605 J-42.484 E.69579
; LINE_WIDTH: 0.621906
G1 F13402.659
G1 X100.786 Y131.055 E.00771
G1 X100.405 Y131.177 E.01532
; LINE_WIDTH: 0.658231
G1 F12622.317
G1 X100.133 Y131.28 E.01182
; LINE_WIDTH: 0.694556
G1 F11927.844
G1 X99.861 Y131.382 E.0125
; LINE_WIDTH: 0.694966
G1 F11920.441
G1 X99.537 Y131.479 E.01459
G1 X99.666 Y132.329 E.03701
G1 X100.055 Y132.013 E.02156
; LINE_WIDTH: 0.694556
G1 F11927.844
G1 X100.357 Y132.043 E.01304
; LINE_WIDTH: 0.657276
G1 F12641.668
G1 X100.658 Y132.072 E.0123
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X101.209 Y132.16 E.02129
G1 X102.813 Y132.03 E.06144
G1 X103.252 Y132.137 E.01727
G1 X103.512 Y132.387 E.01375
G3 X103.873 Y133.331 I-11.971 J5.12 E.03862
G3 X104.199 Y134.673 I-18.846 J5.283 E.05272
G1 X104.199 Y134.768 E.00364
; WIPE_START
G1 X104.199 Y135.768 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X103.079 Y132.759 Z9.4 F36000
G1 Z9
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F2475
M204 S5000
G1 X103.367 Y133.574 E.02737
G1 F3000
G1 X103.416 Y133.768 E.00634
G1 F3600
G1 X103.465 Y133.962 E.00634
G1 X103.514 Y134.157 E.00634
G1 X103.561 Y134.341 E.00603
G1 X103.578 Y134.408 E.00218
G1 X103.595 Y134.475 E.00218
G1 X103.612 Y134.541 E.00218
G1 X103.629 Y134.608 E.00218
G1 X103.646 Y134.675 E.00218
G3 X103.646 Y138.015 I-252.899 J1.67 E.10574
G1 X103.624 Y138.074 E.002
G1 X103.602 Y138.133 E.002
G1 X103.58 Y138.193 E.002
G1 X103.558 Y138.252 E.002
G1 X103.536 Y138.311 E.002
G1 X103.514 Y138.371 E.002
G1 X103.492 Y138.43 E.002
G1 X103.47 Y138.49 E.002
G1 F3000
G2 X103.112 Y139.573 I10.688 J4.127 E.03614
G1 X103.103 Y140.036 E.01464
G1 X103.246 Y140.893 E.02752
G1 F3150
G3 X103.353 Y141.635 I-11.207 J2.004 E.02373
G1 F3300
G1 X103.412 Y142.137 E.01603
G1 F3450
G1 X103.366 Y142.277 E.00464
G1 F3600
G3 X103.344 Y142.313 I-.029 J.007 E.00148
G1 F3450
G1 X103.289 Y142.34 E.00194
G1 F3300
G1 X103.235 Y142.368 E.00194
G1 F3150
G3 X103.2 Y142.377 I-.03 J-.046 E.00117
G1 F3300
G1 X103.09 Y142.391 E.00348
G1 F3450
G1 X102.981 Y142.405 E.00348
G1 F3600
G1 X102.872 Y142.419 E.00348
G1 X102.763 Y142.434 E.00348
G1 X102.654 Y142.448 E.00348
G1 X102.545 Y142.462 E.00348
G3 X102.366 Y142.474 I-.129 J-.566 E.0057
G1 X100.392 Y142.115 E.06352
G1 X99.996 Y142.182 E.01273
G1 X99.933 Y142.166 E.00204
G1 X99.871 Y142.151 E.00204
G1 X99.829 Y142.131 E.00148
G1 X99.483 Y141.949 E.01237
G3 X99.107 Y141.737 I.575 J-1.458 E.0137
G1 X99.102 Y142.421 E.02164
G3 X98.945 Y143.859 I-6.624 J.001 E.0459
G1 X98.938 Y144.167 E.00978
G1 X156.695 Y144.167 E1.8286
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.615 J-43.5 E.51062
G3 X137.192 Y118.315 I40.294 J-33.086 E.49167
G1 X136.614 Y118.615 E.0206
G1 X135.912 Y118.859 E.02354
G1 X135.137 Y119.01 E.02499
G1 X134.371 Y119.041 E.02427
G1 X133.609 Y118.959 E.02427
G1 X132.867 Y118.767 E.02427
G3 X131.479 Y118.047 I2.852 J-7.2 E.0496
G1 X130.92 Y117.577 E.02313
G1 X130.409 Y117.003 E.02433
G3 X130.278 Y116.829 I1.944 J-1.589 E.00688
G1 X126.704 Y112.01 E.18996
G3 X121.894 Y117.543 I-42.012 J-31.657 E.2323
G3 X119.317 Y119.903 I-23.19 J-22.748 E.11069
G1 X116.672 Y122.124 E.10934
G3 X98.936 Y131.038 I-32.26 J-42.081 E.63218
G1 X98.945 Y131.529 E.01555
G1 X99.095 Y132.511 E.03144
G1 X99.101 Y133.274 E.02414
G1 X100.119 Y133.085 E.03277
G1 X100.083 Y132.714 E.01179
G1 X100.228 Y132.578 E.00632
G1 X100.314 Y132.591 E.00276
G1 X100.4 Y132.605 E.00276
G1 X100.486 Y132.619 E.00275
G1 X100.572 Y132.633 E.00276
G1 X100.658 Y132.647 E.00275
G1 X100.744 Y132.661 E.00276
G1 X100.83 Y132.675 E.00275
G1 X100.915 Y132.689 E.00276
G1 F3000
G2 X101.25 Y132.711 I.283 J-1.685 E.01064
G1 X102.102 Y132.647 E.02706
G1 F2475
G1 X102.804 Y132.602 E.02225
G1 F3000
G1 X102.972 Y132.613 E.00533
G1 X103.022 Y132.674 E.00249
G1 F2475
G1 X103.031 Y132.684 E.00044
; WIPE_START
M204 S10000
G1 X103.367 Y133.574 E-.36136
G1 X103.379 Y133.621 E-.01864
; WIPE_END
G1 E-.02 F1800
G1 X105.873 Y136.321 Z9.4 F36000
G1 Z9
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.861 Y134.49 E.0699
G2 X105.769 Y133.987 I-2.463 J.191 E.01955
G2 X107.975 Y130.004 I-3.674 J-4.636 E.17843
G2 X111.416 Y128.238 I-36.422 J-75.189 E.14769
G3 X114.153 Y130.359 I-8.394 J13.662 E.13246
G3 X115.541 Y134.129 I-4.844 J3.923 E.15617
G3 X114.76 Y135.543 I-2.068 J-.219 E.06332
G2 X112.123 Y137.428 I11.911 J19.452 E.12388
G2 X110.285 Y141.67 I3.988 J4.247 E.18148
G1 X110.336 Y141.945 E.01071
G1 X107.994 Y141.945 E.08944
G1 X121.816 Y123.426 F36000
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X122.002 Y123.001 E.01769
G1 X122.085 Y122.545 E.0177
G1 X122.084 Y122.277 E.01025
G1 X121.987 Y121.791 E.01889
G1 X121.829 Y121.448 E.01445
G1 X121.534 Y121.055 E.01875
G1 X118.269 Y123.793 E.16269
G1 X118.671 Y124.192 E.02163
G1 X119.104 Y124.443 E.01912
G1 X119.541 Y124.571 E.01739
G2 X121.116 Y124.167 I.328 J-1.992 E.06389
G1 X121.537 Y123.808 E.0211
G1 X121.763 Y123.498 E.01466
G1 X121.073 Y123.308 F36000
; LINE_WIDTH: 0.79874
G1 F10302.149
G1 X121.337 Y122.881 E.02499
G2 X121.369 Y122.073 I-1.421 J-.462 E.04081
G2 X120.03 Y123.198 I396.046 J473.114 E.08717
G1 X119.313 Y123.799 E.0466
G2 X120.678 Y123.653 I.557 J-1.252 E.07164
G1 X121.006 Y123.367 E.02166
G1 X120.968 Y124.952 F36000
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X118.706 Y124.907 I-1.084 J-2.445 E.0892
G2 X119.664 Y126.117 I4.453 J-2.542 E.05913
G2 X122.301 Y128.002 I14.55 J-17.568 E.12388
G3 X123.081 Y129.416 I-1.288 J1.633 E.06332
G3 X121.243 Y133.658 I-5.825 J-.005 E.18148
G3 X118.606 Y135.543 I-14.549 J-17.568 E.12388
G2 X117.826 Y136.957 I1.288 J1.633 E.06332
G2 X120.62 Y141.945 I5.978 J-.071 E.22762
G1 X125.417 Y141.945 E.18318
G1 X125.366 Y141.67 E.01071
G3 X127.204 Y137.428 I5.825 J.005 E.18148
G3 X129.842 Y135.543 I14.547 J17.565 E.12388
G2 X130.622 Y134.129 I-1.288 J-1.633 E.06332
G2 X128.784 Y129.887 I-5.826 J.005 E.18148
G2 X126.147 Y128.002 I-14.548 J17.566 E.12388
G3 X125.366 Y126.588 I1.288 J-1.633 E.06332
G3 X127.204 Y122.347 I5.825 J.005 E.18148
G3 X129.842 Y120.462 I14.549 J17.568 E.12388
G1 X130.32 Y119.945 E.02687
G2 X133.134 Y121.126 I4.121 J-5.88 E.11742
G2 X132.907 Y121.875 I1.177 J.766 E.03032
G2 X134.745 Y126.117 I5.826 J-.005 E.18148
G2 X137.382 Y128.002 I14.549 J-17.567 E.12388
G3 X138.163 Y129.416 I-1.288 J1.633 E.06332
G3 X136.325 Y133.658 I-5.826 J-.005 E.18148
G3 X133.687 Y135.543 I-14.55 J-17.569 E.12388
G2 X132.907 Y136.957 I1.288 J1.633 E.06332
G2 X135.701 Y141.945 I5.978 J-.071 E.22762
G1 X140.499 Y141.945 E.18318
G1 X140.448 Y141.67 E.01071
G3 X142.286 Y137.428 I5.826 J.005 E.18148
G3 X144.923 Y135.543 I14.548 J17.566 E.12388
G1 X145.275 Y135.178 E.01935
G2 X148.118 Y138.126 I48.574 J-44.003 E.15638
G2 X150.782 Y141.945 I5.831 J-1.229 E.18265
G1 X148.44 Y141.945 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 9.16
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X149.44 Y141.945 E-.38
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
G1 X103.959 Y131.211
G1 Z9.16
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.489 Y131.702 E.02757
G1 X104.766 Y132.312 E.02559
G3 X104.829 Y133.069 I-3.371 J.659 E.02905
G1 X105.046 Y133.815 E.0297
G3 X105.361 Y134.883 I-5.925 J2.327 E.04256
G3 X105.372 Y135.601 I-11.507 J.533 E.02742
G1 X105.372 Y137.601 E.07636
G1 X105.323 Y138.036 E.01669
G3 X104.872 Y139.213 I-8.9 J-2.736 E.04818
G1 X104.829 Y139.44 E.00883
G1 X104.829 Y140.784 E.0513
G1 X104.962 Y141.667 E.03408
G1 X104.939 Y142.373 E.02696
G1 X104.903 Y142.443 E.00303
G1 X154.069 Y142.443 E1.87711
G3 X143.8 Y132.698 I31.508 J-43.483 E.54206
G3 X136.271 Y120.545 I41.766 J-34.285 E.54744
G3 X135.167 Y120.737 I-2.491 J-11.053 E.04278
G3 X133.274 Y120.651 I-.563 J-8.548 E.0725
G3 X130.408 Y119.398 I1.536 J-7.416 E.12032
G3 X129.011 Y118.014 I5.437 J-6.884 E.0752
G1 X126.663 Y114.849 E.15047
G3 X121.362 Y120.439 I-43.549 J-35.993 E.29436
G3 X119.312 Y122.16 I-446.528 J-529.676 E.1022
G1 X117.779 Y123.445 E.07636
G3 X103.994 Y131.17 I-33.38 J-43.405 E.60543
G1 X103.625 Y131.687 F36000
G1 F13446.369
G1 X104.005 Y132.032 E.01962
G1 X104.206 Y132.483 E.01885
G1 X104.243 Y133.171 E.02629
G1 X104.405 Y133.666 E.0199
G2 X104.731 Y134.703 I4.492 J-.842 E.0416
G3 X104.786 Y135.601 I-3.913 J.692 E.03442
G1 X104.786 Y137.601 E.07636
G1 X104.752 Y137.905 E.01168
G3 X104.243 Y139.223 I-28.296 J-10.173 E.05394
G1 X104.243 Y140.828 E.06127
G1 X104.383 Y141.754 E.03576
G1 X104.367 Y142.248 E.01887
G1 X104.053 Y142.864 E.02638
G1 X103.761 Y143.029 E.01281
G1 X155.749 Y143.029 E1.98484
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.253 Y132.326 I30.06 J-44.232 E.59958
G3 X136.596 Y119.828 I41.803 J-34.205 E.56132
G1 X136.064 Y119.995 E.02127
M73 P82 R3
G1 X135.345 Y120.129 E.02795
G1 X134.414 Y120.179 E.0356
G3 X130.774 Y118.941 I.203 J-6.567 E.14897
G1 X130.187 Y118.448 E.02924
G3 X129.065 Y117.104 I7.74 J-7.602 E.06693
G1 X126.682 Y113.891 E.15272
G3 X120.964 Y120.009 I-44.273 J-35.649 E.32
G3 X118.936 Y121.711 I-612.933 J-728.239 E.1011
G1 X117.403 Y122.996 E.07636
G3 X105.475 Y129.943 I-33.033 J-43.006 E.52844
G1 X104.37 Y130.386 E.04547
G1 X103.998 Y130.535 E.01527
G1 F13199.313
G1 X103.627 Y130.684 E.01527
; LINE_WIDTH: 0.665823
G1 F11813.84
G1 X103.479 Y130.763 E.0069
; LINE_WIDTH: 0.71165
G1 F11256.436
G1 X103.331 Y130.842 E.0074
; LINE_WIDTH: 0.757476
G1 F10712.502
G1 X103.184 Y130.921 E.0079
; LINE_WIDTH: 0.803303
G1 F10182.022
G1 X103.036 Y131 E.0084
; LINE_WIDTH: 0.84913
G1 F9665.027
G1 X102.888 Y131.078 E.0089
; LINE_WIDTH: 0.894956
G1 F9150.381
G1 X102.74 Y131.157 E.0094
; LINE_WIDTH: 0.933036
G1 F8762.658
G1 X102.692 Y131.194 E.00357
; LINE_WIDTH: 0.971116
G1 F8406.457
G1 X102.644 Y131.232 E.00373
; LINE_WIDTH: 1.0092
G1 F8078.083
G1 X102.595 Y131.269 E.00388
G1 X102.654 Y131.28 E.00382
; LINE_WIDTH: 0.971116
G1 F8406.457
G1 X102.713 Y131.292 E.00367
; LINE_WIDTH: 0.933036
G1 F8762.658
G1 X102.772 Y131.304 E.00352
; LINE_WIDTH: 0.894956
G1 F9150.381
G1 X102.899 Y131.354 E.00768
; LINE_WIDTH: 0.84913
G1 F9665.027
G1 X103.026 Y131.404 E.00727
; LINE_WIDTH: 0.803303
G1 F10241.018
G1 X103.154 Y131.454 E.00686
; LINE_WIDTH: 0.757476
G1 F10674.579
G1 X103.281 Y131.505 E.00645
; LINE_WIDTH: 0.71165
G1 F11117.13
G1 X103.408 Y131.555 E.00604
; LINE_WIDTH: 0.665823
G1 F11568.681
G1 X103.535 Y131.605 E.00563
; LINE_WIDTH: 0.619996
G1 F11671.775
G1 X103.558 Y131.626 E.00118
G1 X103.301 Y132.163 F36000
G1 F13446.369
G1 X103.53 Y132.374 E.0119
G1 X103.636 Y132.62 E.01022
G1 X103.657 Y133.273 E.02496
G1 X103.837 Y133.809 E.02157
G2 X104.169 Y134.869 I4.436 J-.808 E.04251
G3 X104.201 Y135.601 I-4.168 J.547 E.02804
G1 X104.201 Y137.601 E.07636
G1 X104.152 Y137.872 E.01049
G2 X103.657 Y139.118 I634.06 J252.603 E.05117
G1 X103.657 Y140.872 E.06698
G1 X103.804 Y141.841 E.03743
G1 X103.794 Y142.123 E.01077
G1 X103.616 Y142.475 E.01506
G1 X103.177 Y142.723 E.01925
; LINE_WIDTH: 0.668191
G1 F12423.978
G1 X102.876 Y142.803 E.01285
; LINE_WIDTH: 0.716386
G1 F11546.075
G1 X102.576 Y142.884 E.01382
G1 X101.786 Y142.935 E.0352
; LINE_WIDTH: 0.690626
G1 F11999.27
G1 X101.575 Y142.928 E.00902
; LINE_WIDTH: 0.724633
G1 F11408.139
G1 X101.202 Y142.878 E.01695
; LINE_WIDTH: 0.75864
G1 F10872.517
G1 X100.828 Y142.827 E.01779
; LINE_WIDTH: 0.792646
G1 F10384.936
G2 X100.094 Y142.788 I-.483 J2.158 E.03655
; LINE_WIDTH: 0.781146
G1 F10544.852
G1 X99.737 Y142.691 E.018
G3 X99.652 Y143.534 I-4.902 J-.067 E.04131
; LINE_WIDTH: 0.789296
G1 F10431.017
G1 X100.348 Y143.53 E.03425
; LINE_WIDTH: 0.792646
G1 F10384.936
G3 X100.764 Y143.545 I.12 J2.376 E.02058
; LINE_WIDTH: 0.75864
G1 F10872.517
G1 X101.14 Y143.562 E.01779
; LINE_WIDTH: 0.724633
G1 F11408.139
G1 X101.516 Y143.579 E.01695
; LINE_WIDTH: 0.711536
G1 F11628.767
G1 X102.518 Y143.569 E.04421
G1 X102.888 Y143.592 E.0164
; LINE_WIDTH: 0.665766
G1 F12471.692
G1 X103.259 Y143.615 E.01529
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X103.659 Y143.615 E.01527
G1 X156.235 Y143.615 E2.00732
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.105 J-43.631 E.6078
G3 X136.912 Y119.083 I41.341 J-33.826 E.57635
G1 X136.093 Y119.382 E.03328
G3 X134.381 Y119.594 I-1.758 J-7.175 E.06599
G3 X133.436 Y119.488 I1.14 J-14.455 E.03632
G1 X132.714 Y119.298 E.0285
G3 X131.14 Y118.483 I2.719 J-7.186 E.06783
G1 X130.564 Y118 E.02871
G1 X130.204 Y117.606 E.02036
G3 X129.081 Y116.142 I13.425 J-11.466 E.07048
G1 X126.698 Y112.93 E.15272
G3 X120.566 Y119.579 I-42.951 J-33.455 E.34573
G3 X118.559 Y121.263 I-1038.917 J-1236.476 E.1
G1 X117.027 Y122.547 E.07636
G3 X99.494 Y131.453 I-32.615 J-42.5 E.75507
G2 X99.662 Y132.539 I29.514 J-4.023 E.04199
G1 X99.72 Y132.459 E.00379
G1 X100.122 Y132.199 E.01828
G1 X100.477 Y132.187 E.01354
G1 X101.156 Y132.324 E.02645
G2 X102.737 Y132.036 I-20.72 J-118.375 E.06136
G1 X103.228 Y132.115 E.01899
M204 S250
G1 X102.991 Y132.606 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.098 Y132.749 E.00566
G1 X103.099 Y132.814 E.00208
G1 X103.1 Y132.88 E.00208
G1 X103.1 Y132.945 E.00208
G1 X103.101 Y133.011 E.00208
G1 X103.102 Y133.077 E.00208
G1 X103.102 Y133.142 E.00208
G1 X103.103 Y133.208 E.00208
G1 X103.104 Y133.273 E.00208
G1 F3000
G1 X103.104 Y133.339 E.00208
G1 F2475
G1 X103.193 Y133.616 E.00921
G1 F3000
G1 X103.275 Y133.842 E.0076
G1 X103.466 Y134.584 E.02426
G1 F3600
G1 X103.488 Y134.639 E.00187
G1 X103.509 Y134.694 E.00187
G1 X103.531 Y134.749 E.00187
G1 X103.552 Y134.804 E.00187
G1 X103.574 Y134.859 E.00187
G1 X103.596 Y134.915 E.00187
G1 X103.617 Y134.97 E.00187
G3 X103.648 Y135.601 I-2.009 J.415 E.02011
G1 X103.648 Y137.601 E.06332
G1 X103.623 Y137.67 E.00232
G1 X103.597 Y137.739 E.00231
G1 X103.572 Y137.807 E.00232
G1 X103.547 Y137.876 E.00231
G1 X103.522 Y137.944 E.00231
G1 X103.496 Y138.013 E.00232
G1 X103.471 Y138.082 E.00231
G1 X103.446 Y138.15 E.00231
G1 F3000
G3 X103.105 Y139.018 I-19.19 J-7.049 E.02952
G1 X103.105 Y140.913 E.06001
G1 X103.122 Y141.031 E.00376
G1 F3150
G1 X103.257 Y141.924 E.02859
G1 X103.247 Y141.956 E.00107
G1 F3300
G1 X103.186 Y142.116 E.00543
G1 F3150
G1 X103.128 Y142.149 E.0021
G1 F3600
G3 X102.971 Y142.199 I-.151 J-.206 E.00534
G1 X102.434 Y142.296 E.01725
G1 X102.357 Y142.302 E.00245
G1 X102.28 Y142.308 E.00245
G1 X102.203 Y142.315 E.00245
G1 X102.125 Y142.321 E.00245
G1 X102.048 Y142.327 E.00245
G1 X101.971 Y142.334 E.00245
G1 X101.894 Y142.34 E.00245
G1 X101.816 Y142.346 E.00245
G3 X101.68 Y142.35 I-.081 J-.512 E.00431
G1 X100.392 Y142.115 E.04145
G1 X100.126 Y142.155 E.00852
G1 X99.805 Y141.993 E.01138
G1 X99.485 Y141.831 E.01138
G1 X99.164 Y141.669 E.01138
G1 X99.105 Y141.639 E.0021
G1 X99.105 Y141.696 E.00181
G1 X99.105 Y142.006 E.00981
G1 X99.105 Y142.316 E.00981
G3 X98.945 Y143.859 I-7.086 J.046 E.0492
G1 X98.938 Y144.167 E.00977
G1 X156.695 Y144.167 E1.82859
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.621 J-43.506 E.51058
G3 X137.192 Y118.315 I40.31 J-33.096 E.49171
G1 X136.612 Y118.616 E.02069
G3 X134.365 Y119.041 I-2.175 J-5.342 E.07288
G1 X133.61 Y118.959 E.02405
G1 X132.865 Y118.766 E.02435
G1 X132.343 Y118.545 E.01795
G1 X131.508 Y118.068 E.03045
G1 X130.92 Y117.577 E.02427
G1 X130.408 Y117.002 E.02435
G3 X129.087 Y115.222 I339.224 J-253.217 E.07018
G1 X126.704 Y112.01 E.12664
G3 X121.894 Y117.543 I-42.586 J-32.156 E.2323
G3 X119.328 Y119.897 I-23.864 J-23.446 E.11028
G1 X116.672 Y122.124 E.10976
G3 X98.936 Y131.038 I-32.26 J-42.081 E.63218
G1 X98.946 Y131.529 E.01553
G1 X99.097 Y132.511 E.03146
G1 X99.103 Y133.271 E.02407
G1 X100.119 Y133.083 E.0327
G1 X100.119 Y132.855 E.00722
G1 X100.265 Y132.733 E.00603
G1 X100.346 Y132.746 E.0026
G1 X100.427 Y132.759 E.0026
G1 X100.508 Y132.773 E.0026
G1 X100.589 Y132.786 E.0026
G1 X100.671 Y132.799 E.0026
G1 X100.752 Y132.813 E.0026
G1 X100.833 Y132.826 E.0026
G1 X100.914 Y132.84 E.0026
G2 X101.274 Y132.869 I.351 J-2.077 E.01146
G1 X101.403 Y132.845 E.00416
G1 X101.582 Y132.812 E.00577
G1 X101.762 Y132.779 E.00577
G1 X101.941 Y132.745 E.00577
G1 X102.121 Y132.712 E.00577
G1 X102.3 Y132.679 E.00577
G1 X102.479 Y132.646 E.00577
G1 X102.659 Y132.613 E.00577
G3 X102.903 Y132.584 I.176 J.45 E.00789
; WIPE_START
M204 S10000
G1 X103.098 Y132.749 E-.09694
G1 X103.099 Y132.814 E-.02493
G1 X103.1 Y132.88 E-.02493
G1 X103.1 Y132.945 E-.02493
G1 X103.101 Y133.011 E-.02492
G1 X103.102 Y133.077 E-.02493
G1 X103.102 Y133.142 E-.02493
G1 X103.103 Y133.208 E-.02492
G1 X103.104 Y133.273 E-.02493
G1 X103.104 Y133.339 E-.02493
G1 X103.152 Y133.486 E-.05874
; WIPE_END
G1 E-.02 F1800
G1 X101.174 Y131.639 Z9.56 F36000
G1 Z9.16
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.839226
G1 F9783.946
G1 X101.472 Y131.563 E.01612
; LINE_WIDTH: 0.880116
G1 F9310.933
G1 X101.77 Y131.487 E.01693
; LINE_WIDTH: 0.923143
G1 F8860.195
G1 X102.045 Y131.414 E.01649
; LINE_WIDTH: 0.96617
G1 F8451.082
G1 X102.32 Y131.341 E.01729
; LINE_WIDTH: 1.0092
G1 F8078.083
G1 X102.595 Y131.269 E.01809
; WIPE_START
G1 X102.32 Y131.341 E-.10817
G1 X102.045 Y131.414 E-.10817
G1 X101.77 Y131.487 E-.10817
G1 X101.628 Y131.523 E-.05548
; WIPE_END
G1 E-.02 F1800
G1 X105.864 Y136.325 Z9.56 F36000
G1 Z9.16
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X105.857 Y134.832 E.05702
G2 X105.628 Y134.017 I-2.581 J.287 E.03245
G2 X108.17 Y129.913 I-3.765 J-5.171 E.18898
G2 X111.224 Y128.345 I-20.761 J-44.192 E.1311
G3 X114.131 Y130.359 I-5.785 J11.46 E.13544
G3 X115.748 Y134.6 I-4.681 J4.213 E.1773
G3 X115.138 Y135.543 I-1.418 J-.249 E.04404
G3 X112.803 Y136.957 I-90.094 J-146.139 E.10421
G2 X110.084 Y141.945 I3.389 J5.083 E.22562
G1 X107.741 Y141.945 E.08944
G1 X119.901 Y122.312 F36000
G1 F13446.283
G1 X118.105 Y123.817 E.08944
G2 X120.344 Y126.588 I5.515 J-2.165 E.13815
G3 X122.679 Y128.002 I-87.759 J147.553 E.10421
G3 X123.289 Y128.945 I-.808 J1.191 E.04404
G3 X120.563 Y134.129 I-6.15 J.075 E.23307
G2 X118.228 Y135.543 I87.662 J147.393 E.10421
G2 X117.618 Y136.485 I.808 J1.191 E.04404
G2 X120.787 Y141.945 I6.201 J.051 E.25288
G1 X125.165 Y141.945 E.16715
G3 X127.885 Y136.957 I6.109 J.094 E.22562
G2 X130.219 Y135.543 I-87.802 J-147.625 E.10421
G2 X130.829 Y134.6 I-.808 J-1.191 E.04404
G2 X128.104 Y129.416 I-6.15 J-.075 E.23307
G3 X125.769 Y128.002 I87.759 J-147.553 E.10421
G3 X125.159 Y127.06 I.808 J-1.191 E.04404
G3 X126.386 Y123.289 I6.376 J-.01 E.15393
G3 X129.483 Y120.933 I6.647 J5.523 E.14987
G2 X130.563 Y120.093 I-1.091 J-2.517 E.05279
G2 X132.827 Y121.053 I4.292 J-6.97 E.09427
G1 X132.699 Y121.404 E.01425
G2 X135.425 Y126.588 I6.15 J.075 E.23307
G3 X137.76 Y128.002 I-87.844 J147.693 E.10421
G3 X138.37 Y128.945 I-.808 J1.191 E.04404
G3 X135.644 Y134.129 I-6.15 J.075 E.23307
G2 X133.31 Y135.543 I87.619 J147.322 E.10421
G2 X132.699 Y136.485 I.808 J1.191 E.04404
G2 X135.869 Y141.945 I6.201 J.051 E.25288
G1 X140.246 Y141.945 E.16715
G3 X142.966 Y136.957 I6.109 J.094 E.22562
G2 X145.301 Y135.543 I-87.759 J-147.553 E.10421
G1 X145.45 Y135.383 E.00836
G2 X147.948 Y137.96 I59.94 J-55.58 E.13706
G2 X150.95 Y141.945 I6.006 J-1.401 E.1961
G1 X148.607 Y141.945 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 9.32
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X149.607 Y141.945 E-.38
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
G1 X103.954 Y131.203
G1 Z9.32
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.49 Y131.698 E.02786
G1 X104.768 Y132.309 E.02564
G1 X104.831 Y132.878 E.02187
G1 X104.831 Y133.673 E.03033
G3 X105.349 Y135.297 I-18.641 J6.844 E.06511
G1 X105.374 Y135.609 E.01194
G1 X105.374 Y137.092 E.05664
G1 X105.334 Y137.488 E.01518
G3 X104.831 Y138.929 I-9.613 J-2.544 E.05836
G1 X104.831 Y141.766 E.1083
G1 X104.712 Y142.442 E.02618
G2 X106.069 Y142.443 I1.077 J-323.407 E.05181
G1 X154.069 Y142.443 E1.8326
G3 X143.803 Y132.702 I31.807 J-43.799 E.54187
G3 X136.269 Y120.54 I42.392 J-34.679 E.54777
G3 X135.115 Y120.741 I-1.602 J-5.779 E.04478
G3 X133.311 Y120.655 I-.506 J-8.34 E.06908
G1 X132.419 Y120.431 E.03512
G1 X131.664 Y120.13 E.03104
G3 X129.117 Y118.145 I2.952 J-6.415 E.12437
G3 X127.855 Y116.455 I53.894 J-41.582 E.08054
G1 X126.663 Y114.849 E.07636
G3 X121.362 Y120.438 I-43.359 J-35.812 E.29433
G3 X119.312 Y122.16 I-437.527 J-518.939 E.10224
G1 X117.779 Y123.445 E.07636
G3 X104.007 Y131.165 I-33.379 J-43.405 E.60489
G1 X103.622 Y131.68 F36000
G1 F13446.369
G1 X104.007 Y132.028 E.0198
G1 X104.208 Y132.48 E.01888
G3 X104.245 Y133.758 I-10.385 J.946 E.04882
G2 X104.669 Y135.02 I24.424 J-7.486 E.05086
G1 X104.787 Y135.548 E.02063
G3 X104.789 Y137.092 I-34.677 J.803 E.05897
G1 X104.719 Y137.523 E.01666
G3 X104.245 Y138.813 I-9.91 J-2.907 E.0525
G1 X104.245 Y141.766 E.11276
G1 X104.161 Y142.238 E.01831
G1 X103.812 Y142.764 E.02409
G1 X103.285 Y143.029 E.02252
G1 X155.749 Y143.029 E2.00303
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I29.804 J-43.953 E.5995
G3 X136.598 Y119.835 I41.229 J-33.857 E.5612
G1 X135.956 Y120.021 E.02553
G3 X133.966 Y120.156 I-1.503 J-7.401 E.07638
G3 X129.99 Y118.247 I.653 J-6.458 E.17184
G3 X129.065 Y117.104 I8.446 J-7.775 E.05618
G1 X126.682 Y113.891 E.15272
G3 X120.964 Y120.009 I-44.519 J-35.878 E.31997
G3 X118.935 Y121.712 I-603.999 J-717.583 E.10114
G1 X117.403 Y122.996 E.07636
G3 X105.475 Y129.943 I-33.033 J-43.007 E.52842
G1 X104.369 Y130.387 E.0455
G1 X103.998 Y130.535 E.01527
G1 F13191.696
G1 X103.626 Y130.684 E.01527
; LINE_WIDTH: 0.665915
G1 F11806.634
G1 X103.479 Y130.763 E.00689
; LINE_WIDTH: 0.711833
G1 F11249.706
G1 X103.331 Y130.842 E.00739
; LINE_WIDTH: 0.757751
G1 F10706.204
G1 X103.183 Y130.921 E.0079
; LINE_WIDTH: 0.80367
G1 F10176.186
G1 X103.036 Y131 E.0084
; LINE_WIDTH: 0.849588
G1 F9659.595
G1 X102.888 Y131.079 E.0089
; LINE_WIDTH: 0.895506
G1 F9144.536
G1 X102.741 Y131.158 E.0094
; LINE_WIDTH: 0.932836
G1 F8764.609
G1 X102.693 Y131.194 E.00352
; LINE_WIDTH: 0.970166
G1 F8414.99
G1 X102.645 Y131.231 E.00366
; LINE_WIDTH: 1.0075
G1 F8092.195
G1 X102.597 Y131.267 E.00381
G1 X102.655 Y131.278 E.00375
; LINE_WIDTH: 0.970166
G1 F8414.99
G1 X102.713 Y131.29 E.00361
; LINE_WIDTH: 0.932836
G1 F8764.609
G1 X102.771 Y131.301 E.00346
; LINE_WIDTH: 0.895506
G1 F9144.536
G1 X102.899 Y131.351 E.00769
; LINE_WIDTH: 0.849588
G1 F9659.595
G1 X103.026 Y131.401 E.00728
; LINE_WIDTH: 0.80367
G1 F10236.136
G1 X103.154 Y131.451 E.00687
; LINE_WIDTH: 0.757751
G1 F10669.847
G1 X103.281 Y131.502 E.00646
; LINE_WIDTH: 0.711833
G1 F11112.528
G1 X103.408 Y131.552 E.00605
; LINE_WIDTH: 0.665915
G1 F11564.217
G1 X103.536 Y131.602 E.00564
; LINE_WIDTH: 0.619996
G1 F11653.748
G1 X103.556 Y131.62 E.00102
G1 X103.301 Y132.158 F36000
G1 F13446.369
G1 X103.532 Y132.371 E.012
G1 X103.638 Y132.617 E.01023
G3 X103.66 Y133.843 I-17.606 J.922 E.04681
G2 X104.111 Y135.199 I12.914 J-3.543 E.0546
G1 X104.203 Y135.609 E.01604
G1 X104.203 Y137.092 E.05664
G1 X104.163 Y137.338 E.00951
G3 X103.66 Y138.696 I-10.031 J-2.947 E.05534
G1 X103.66 Y141.766 E.11722
G1 X103.611 Y142.036 E.01045
G1 X103.412 Y142.335 E.01375
G1 X103.022 Y142.532 E.01669
; LINE_WIDTH: 0.662546
G1 F12484.478
G1 X102.812 Y142.592 E.00894
; LINE_WIDTH: 0.705096
G1 F11740.415
G1 X102.602 Y142.652 E.00954
; LINE_WIDTH: 0.747646
G1 F11040.081
G1 X102.392 Y142.712 E.01015
; LINE_WIDTH: 0.790196
G1 F10418.597
G1 X102.182 Y142.772 E.01075
G1 X101.502 Y142.816 E.03361
G1 X101.103 Y142.841 E.01971
; LINE_WIDTH: 0.794696
G1 F10356.937
G1 X100.432 Y142.773 E.03339
; LINE_WIDTH: 0.801306
G1 F10267.678
G1 X100.157 Y142.762 E.01376
G1 X99.75 Y142.628 E.02144
G1 X99.7 Y143.317 E.03455
G1 X99.664 Y143.524 E.0105
G1 X100.293 Y143.528 E.03142
; LINE_WIDTH: 0.793676
G1 F10370.849
G1 X100.819 Y143.548 E.02606
; LINE_WIDTH: 0.785656
G1 F10481.553
G1 X102.126 Y143.532 E.06401
G1 X102.374 Y143.552 E.01222
; LINE_WIDTH: 0.744241
G1 F11093.035
G1 X102.623 Y143.573 E.01154
; LINE_WIDTH: 0.702826
G1 F11780.282
G1 X102.872 Y143.594 E.01087
; LINE_WIDTH: 0.661411
G1 F12558.308
G1 X103.12 Y143.615 E.0102
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X103.52 Y143.615 E.01527
G1 X156.235 Y143.615 E2.01262
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.706 Y131.955 I29.326 J-43.872 E.60771
G3 X136.908 Y119.075 I41.342 J-33.829 E.57674
G1 X136.5 Y119.253 E.017
G1 X135.793 Y119.458 E.0281
G1 X134.957 Y119.581 E.03226
G3 X131.635 Y118.796 I-.391 J-5.764 E.13228
G3 X130.677 Y118.095 I2.066 J-3.832 E.04545
G1 X129.995 Y117.369 E.03806
G3 X129.081 Y116.142 I77.258 J-58.491 E.0584
G1 X126.698 Y112.93 E.15272
G3 X120.567 Y119.579 I-42.955 J-33.458 E.34572
G3 X118.559 Y121.263 I-1022.191 J-1216.528 E.10003
G1 X117.026 Y122.548 E.07636
G3 X99.494 Y131.453 I-32.616 J-42.502 E.75504
G1 X99.647 Y132.436 E.038
G1 X99.654 Y132.605 E.00644
G1 X99.797 Y132.563 E.00568
G1 X100.201 Y132.344 E.01755
G3 X100.773 Y132.398 I-.101 J4.181 E.02192
G1 X102.739 Y132.033 E.07636
G1 X103.227 Y132.111 E.01887
M204 S250
G1 X102.993 Y132.603 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.1 Y132.746 E.00567
G3 X103.107 Y133.475 I-40.255 J.709 E.02307
G1 X103.107 Y133.528 E.00167
G1 X103.107 Y133.581 E.00167
G1 X103.107 Y133.633 E.00167
G1 X103.107 Y133.686 E.00167
G1 X103.107 Y133.739 E.00167
G1 X103.107 Y133.792 E.00167
G1 X103.107 Y133.845 E.00167
G1 F3000
G1 X103.107 Y133.898 E.00167
G1 F2475
G1 X103.204 Y134.25 E.01155
G1 F3000
G2 X103.328 Y134.648 I3.713 J-.938 E.01322
G1 F3600
G1 X103.39 Y134.823 E.0059
G1 X103.453 Y134.999 E.0059
G1 X103.515 Y135.174 E.0059
G1 X103.578 Y135.35 E.0059
G1 X103.599 Y135.423 E.00242
G1 X103.616 Y135.485 E.00203
G1 X103.633 Y135.547 E.00203
G1 X103.65 Y135.609 E.00203
G1 X103.65 Y137.092 E.04697
G1 X103.621 Y137.185 E.0031
G1 X103.591 Y137.279 E.0031
G1 X103.561 Y137.372 E.0031
G1 X103.532 Y137.465 E.0031
G1 X103.502 Y137.558 E.0031
G1 X103.473 Y137.652 E.0031
G1 X103.443 Y137.745 E.0031
G1 X103.414 Y137.838 E.0031
G1 F3000
G1 X103.107 Y138.586 E.02559
G1 X103.107 Y138.63 E.00139
G1 F3600
G1 X103.107 Y138.772 E.0045
G1 X103.107 Y138.914 E.0045
G1 X103.107 Y139.056 E.0045
G1 X103.107 Y139.198 E.0045
G1 X103.107 Y139.34 E.0045
G1 X103.107 Y139.482 E.0045
G1 X103.107 Y139.624 E.0045
G1 X103.107 Y140.086 E.01461
G1 X103.107 Y140.405 E.01011
G1 X103.107 Y140.724 E.01011
G1 X103.107 Y141.043 E.01011
G1 X103.107 Y141.363 E.01011
G1 F3450
G1 X103.107 Y141.682 E.01011
G1 F3300
G1 X103.107 Y141.766 E.00267
G1 X103.035 Y141.931 E.0057
G1 X103.014 Y141.942 E.00076
G1 F3150
G1 X102.961 Y141.968 E.00186
G1 F3000
G1 X102.579 Y142.051 E.01238
G1 F2475
G1 X102.025 Y142.149 E.01781
G1 F3000
G1 X101.917 Y142.157 E.00343
G1 F3600
G1 X101.809 Y142.166 E.00343
G1 X101.701 Y142.175 E.00343
G1 X101.593 Y142.184 E.00343
G1 X101.485 Y142.193 E.00343
G1 X101.377 Y142.201 E.00343
G1 X101.269 Y142.21 E.00343
G1 X101.161 Y142.219 E.00343
G1 X101.053 Y142.228 E.00343
G2 X100.397 Y142.121 I-3.467 J19.197 E.02104
G1 X100.247 Y142.098 E.0048
G3 X100.094 Y142.046 I-.013 J-.212 E.00525
G1 X99.937 Y141.953 E.00579
G1 X99.78 Y141.859 E.00579
G1 F3450
G1 X99.623 Y141.765 E.00579
G1 F3300
G2 X99.107 Y141.516 I-1.331 J2.099 E.01819
G1 X99.107 Y141.589 E.00231
G1 F3450
G1 X99.107 Y141.762 E.00547
G1 F3600
G1 X99.107 Y141.935 E.00547
G1 X99.107 Y142.108 E.00547
G1 X99.107 Y142.281 E.00547
G1 X99.107 Y142.453 E.00547
G3 X98.941 Y143.857 I-6.754 J-.085 E.04482
G1 X98.936 Y144.167 E.00984
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.615 J-43.5 E.51059
G3 X137.192 Y118.316 I40.596 J-33.267 E.49163
G3 X136.346 Y118.722 I-5.228 J-9.807 E.02973
G1 X135.639 Y118.928 E.0233
G1 X134.882 Y119.033 E.02418
G1 X134.117 Y119.026 E.02425
G1 X133.554 Y118.948 E.01799
G1 X132.864 Y118.766 E.02259
G1 X132.343 Y118.545 E.01791
G1 X131.506 Y118.067 E.03053
G1 X131.033 Y117.672 E.01949
G1 X130.409 Y117.003 E.02899
G3 X129.087 Y115.222 I335.842 J-250.707 E.0702
G1 X126.704 Y112.01 E.12664
G3 X121.433 Y117.999 I-42.504 J-32.089 E.25283
G3 X119.329 Y119.896 I-22.079 J-22.378 E.08973
G1 X116.671 Y122.124 E.10979
G3 X98.936 Y131.038 I-32.259 J-42.081 E.63217
G1 X98.946 Y131.528 E.01551
G1 X99.099 Y132.511 E.03148
G1 X99.105 Y133.269 E.024
G1 X100.154 Y133.074 E.03376
G1 X100.207 Y132.936 E.0047
M73 P83 R3
G1 X100.4 Y132.887 E.00631
G1 X100.444 Y132.897 E.00141
G1 X100.487 Y132.906 E.00141
G1 X100.531 Y132.915 E.00141
G1 X100.574 Y132.925 E.00141
G1 X100.617 Y132.934 E.00141
G1 X100.661 Y132.944 E.00141
G1 X100.704 Y132.953 E.00141
G1 X100.748 Y132.962 E.00141
G1 X100.76 Y132.962 E.00039
G1 F3150
G1 X100.819 Y132.951 E.00191
G1 F3300
G1 X100.902 Y132.936 E.00267
G1 F3450
G1 X101.225 Y132.876 E.0104
G1 F3600
G1 X101.548 Y132.816 E.0104
G1 X101.871 Y132.756 E.0104
G1 X102.194 Y132.697 E.0104
G1 X102.517 Y132.637 E.0104
G1 X102.84 Y132.577 E.0104
G1 X102.904 Y132.588 E.00207
; WIPE_START
M204 S10000
G1 X103.1 Y132.746 E-.09581
G1 X103.107 Y133.475 E-.27691
G1 X103.107 Y133.494 E-.00728
; WIPE_END
G1 E-.02 F1800
G1 X100.771 Y131.74 Z9.72 F36000
G1 Z9.32
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.78711
G1 F10461.317
G1 X101.104 Y131.656 E.01685
; LINE_WIDTH: 0.832563
G1 F9865.621
G1 X101.436 Y131.571 E.01787
; LINE_WIDTH: 0.878016
G1 F9334.109
G1 X101.769 Y131.486 E.01889
; LINE_WIDTH: 0.921176
G1 F8879.844
G1 X102.045 Y131.413 E.01651
; LINE_WIDTH: 0.964336
G1 F8467.743
G1 X102.321 Y131.34 E.01731
; LINE_WIDTH: 1.0075
G1 F8092.195
G1 X102.597 Y131.267 E.01812
; WIPE_START
G1 X102.321 Y131.34 E-.10851
G1 X102.045 Y131.413 E-.10851
G1 X101.769 Y131.486 E-.10851
G1 X101.63 Y131.522 E-.05447
; WIPE_END
G1 E-.02 F1800
G1 X105.069 Y131.768 Z9.72 F36000
G1 Z9.32
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X105.329 Y133.597 I-3.214 J1.39 E.07136
G1 X105.479 Y134.042 E.01796
G2 X108.374 Y129.887 I-3.849 J-5.768 E.19792
G1 X108.389 Y129.812 E.00295
G3 X111.01 Y128.473 I527.581 J1030.257 E.11237
G3 X114.537 Y130.83 I-2.859 J8.096 E.16364
G3 X116.1 Y134.6 I-6.336 J4.836 E.15758
G1 X116.057 Y135.072 E.01807
G1 X115.689 Y135.543 E.02282
G2 X112.932 Y136.957 I696.538 J1361.789 E.11831
G2 X109.911 Y141.198 I3.743 J5.862 E.20378
G1 X109.764 Y141.945 E.02907
G1 X105.901 Y141.945 E.1475
G3 X105.329 Y141.627 I1.224 J-2.869 E.02506
G1 X105.329 Y139.284 E.08944
G1 X119.837 Y122.366 F36000
G1 F13446.283
G1 X118.041 Y123.871 E.08944
G2 X120.473 Y126.588 I5.506 J-2.48 E.14143
G2 X123.23 Y128.002 I703.377 J-1368.336 E.11831
G1 X123.597 Y128.473 E.02282
G1 X123.641 Y128.945 E.01807
G1 X123.455 Y129.887 E.03668
G3 X120.434 Y134.129 I-6.763 J-1.62 E.20378
G3 X117.677 Y135.543 I-709.378 J-1380.034 E.11831
G1 X117.309 Y136.014 E.02282
G1 X117.266 Y136.485 E.01807
G1 X117.452 Y137.428 E.03668
G2 X120.982 Y141.945 I6.692 J-1.592 E.22578
G1 X124.845 Y141.945 E.1475
G1 X124.992 Y141.198 E.02907
G3 X128.013 Y136.957 I6.763 J1.62 E.20378
G3 X130.771 Y135.543 I701.326 J1364.331 E.11831
G1 X131.138 Y135.072 E.02282
G1 X131.181 Y134.6 E.01807
G1 X130.996 Y133.658 E.03668
G2 X127.975 Y129.416 I-6.763 J1.62 E.20378
G2 X125.218 Y128.002 I-699.295 J1360.375 E.11831
G1 X124.85 Y127.531 E.02282
G1 X124.807 Y127.06 E.01807
G1 X124.992 Y126.117 E.03668
G3 X128.013 Y121.875 I6.763 J1.62 E.20378
G3 X130.771 Y120.462 I709.378 J1380.034 E.11831
G1 X130.899 Y120.297 E.00798
G2 X132.39 Y120.937 I4.942 J-9.452 E.06202
G1 X132.348 Y121.404 E.01791
G1 X132.533 Y122.347 E.03668
G2 X135.554 Y126.588 I6.763 J-1.62 E.20378
G2 X138.311 Y128.002 I699.295 J-1360.375 E.11831
G1 X138.679 Y128.473 E.02282
G1 X138.722 Y128.945 E.01807
G1 X138.536 Y129.887 E.03668
G3 X135.516 Y134.129 I-6.764 J-1.62 E.20378
G3 X132.758 Y135.543 I-707.3 J-1375.987 E.11831
G1 X132.391 Y136.014 E.02282
G1 X132.348 Y136.485 E.01807
G1 X132.533 Y137.428 E.03668
G2 X136.063 Y141.945 I6.692 J-1.592 E.22578
G1 X139.927 Y141.945 E.1475
G1 X140.074 Y141.198 E.02907
G3 X143.095 Y136.957 I6.764 J1.62 E.20378
G3 X145.679 Y135.636 I162.117 J314.073 E.1108
G2 X147.717 Y137.729 I46.331 J-43.066 E.11155
G2 X151.145 Y141.945 I6.258 J-1.586 E.21413
G1 X148.802 Y141.945 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 9.48
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X149.802 Y141.945 E-.38
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
G1 X103.949 Y131.194
G1 Z9.48
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.492 Y131.695 E.02818
G1 X104.77 Y132.306 E.02564
G1 X104.833 Y132.875 E.02187
G1 X104.833 Y134.17 E.04944
G1 X105.036 Y134.708 E.02194
G1 X105.246 Y135.608 E.0353
G1 X105.292 Y136.057 E.01723
G3 X104.833 Y138.52 I-5.764 J.199 E.09643
G1 X104.833 Y141.795 E.12501
G1 X104.738 Y142.396 E.02324
G1 X104.711 Y142.443 E.00208
G1 X154.069 Y142.443 E1.88445
G3 X143.803 Y132.702 I31.501 J-43.476 E.5419
G3 X136.271 Y120.545 I42.215 J-34.569 E.54758
G1 X135.454 Y120.705 E.03177
G1 X134.486 Y120.764 E.03703
G3 X133.334 Y120.662 I.845 J-15.994 E.04417
G1 X132.426 Y120.433 E.03576
G1 X131.664 Y120.13 E.03129
G3 X129.118 Y118.146 I2.952 J-6.415 E.12435
G3 X127.855 Y116.455 I53.93 J-41.611 E.08056
G1 X126.663 Y114.849 E.07636
G3 X121.363 Y120.438 I-43 J-35.473 E.29432
G3 X119.312 Y122.16 I-434.014 J-514.745 E.10225
G1 X117.779 Y123.445 E.07636
G3 X104.02 Y131.16 I-33.379 J-43.404 E.60435
G1 X103.62 Y131.674 F36000
G1 F13446.369
G1 X104.009 Y132.025 E.02
G1 X104.21 Y132.477 E.01888
G1 X104.247 Y132.875 E.01525
G1 X104.247 Y134.278 E.05356
G1 X104.461 Y134.837 E.02286
G1 X104.675 Y135.735 E.03522
G1 X104.707 Y136.049 E.01206
G1 X104.697 Y136.757 E.02705
G3 X104.247 Y138.41 I-6.665 J-.927 E.06555
G1 X104.247 Y141.795 E.12924
G1 X104.181 Y142.215 E.01626
G1 X103.91 Y142.692 E.02094
G1 X103.424 Y143.029 E.02258
G1 X155.749 Y143.029 E1.99771
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.059 J-44.231 E.59947
G3 X136.596 Y119.828 I41.637 J-34.107 E.56144
G1 X136.067 Y119.995 E.02116
G1 X135.347 Y120.129 E.02799
G1 X134.462 Y120.179 E.03381
G3 X133.284 Y120.054 I.636 J-11.582 E.04526
G3 X129.99 Y118.247 I1.351 J-6.371 E.14552
G3 X129.065 Y117.104 I8.446 J-7.775 E.05618
G1 X126.682 Y113.891 E.15272
G3 X120.965 Y120.008 I-42.664 J-34.145 E.31999
G3 X118.936 Y121.711 I-599.254 J-711.918 E.10114
G1 X117.403 Y122.996 E.07636
G3 X105.475 Y129.943 I-33.033 J-43.007 E.52842
G1 X104.37 Y130.386 E.04548
G1 X103.998 Y130.535 E.01527
G1 F13186.698
G1 X103.627 Y130.684 E.01527
; LINE_WIDTH: 0.666008
G1 F11801.906
G1 X103.479 Y130.763 E.0069
; LINE_WIDTH: 0.71202
G1 F11244.75
G1 X103.332 Y130.842 E.0074
; LINE_WIDTH: 0.758031
G1 F10701.05
G1 X103.184 Y130.921 E.0079
; LINE_WIDTH: 0.804043
G1 F10170.823
G1 X103.036 Y131 E.00841
; LINE_WIDTH: 0.850055
G1 F9654.068
G1 X102.888 Y131.079 E.00891
; LINE_WIDTH: 0.896066
G1 F9138.594
G1 X102.741 Y131.158 E.00941
; LINE_WIDTH: 0.93265
G1 F8766.429
G1 X102.694 Y131.194 E.00346
; LINE_WIDTH: 0.969233
G1 F8423.392
G1 X102.647 Y131.229 E.0036
; LINE_WIDTH: 1.00582
G1 F8106.189
G1 X102.6 Y131.265 E.00374
G1 X102.657 Y131.276 E.00369
; LINE_WIDTH: 0.969233
G1 F8423.392
G1 X102.714 Y131.287 E.00355
; LINE_WIDTH: 0.93265
G1 F8766.429
G1 X102.771 Y131.298 E.00341
; LINE_WIDTH: 0.896066
G1 F9138.594
G1 X102.899 Y131.348 E.0077
; LINE_WIDTH: 0.850055
G1 F9654.068
G1 X103.026 Y131.399 E.00729
; LINE_WIDTH: 0.804043
G1 F10231.171
G1 X103.154 Y131.449 E.00688
; LINE_WIDTH: 0.758031
G1 F10665.442
G1 X103.282 Y131.499 E.00647
; LINE_WIDTH: 0.71202
G1 F11108.738
G1 X103.409 Y131.549 E.00606
; LINE_WIDTH: 0.666008
G1 F11561.029
G1 X103.537 Y131.599 E.00565
; LINE_WIDTH: 0.619996
G1 F11634.455
G1 X103.553 Y131.614 E.00084
G1 X103.3 Y132.154 F36000
G1 F13446.369
G1 X103.533 Y132.368 E.0121
G1 X103.64 Y132.614 E.01024
G3 X103.662 Y134.386 I-34.506 J1.306 E.06765
G1 X103.914 Y135.046 E.02698
G1 X104.103 Y135.862 E.03196
G1 X104.112 Y136.749 E.03389
G1 X103.917 Y137.63 E.03443
G1 X103.662 Y138.299 E.02733
G1 X103.662 Y141.395 E.11819
G1 X103.662 Y141.795 E.01527
G1 F12987.61
G1 X103.624 Y142.035 E.00928
G1 F12143.508
G1 X103.469 Y142.307 E.01195
; LINE_WIDTH: 0.666676
G1 F11098.335
G1 X103.373 Y142.401 E.00555
; LINE_WIDTH: 0.713356
G1 F10663.542
G1 X103.277 Y142.495 E.00596
; LINE_WIDTH: 0.760036
G1 F10237.392
G1 X103.181 Y142.589 E.00637
; LINE_WIDTH: 0.808911
G1 F9819.931
G1 X103.109 Y142.634 E.00425
; LINE_WIDTH: 0.857786
G1 F9563.423
G1 X103.038 Y142.678 E.00451
; LINE_WIDTH: 0.858896
G1 F9550.549
G1 X102.72 Y142.677 E.01713
; LINE_WIDTH: 0.892056
G1 F9181.319
G1 X102.209 Y142.627 E.0287
G1 X101.514 Y142.681 E.03896
; LINE_WIDTH: 0.854641
G1 F9600.089
G1 X100.819 Y142.734 E.03726
; LINE_WIDTH: 0.839756
G1 F9777.508
G1 X100.293 Y142.702 E.0277
; LINE_WIDTH: 0.847196
G1 F9688.016
G1 X99.956 Y142.597 E.01872
G1 X99.775 Y142.474 E.01156
G3 X99.694 Y143.501 I-8.468 J-.156 E.0546
G1 X100.248 Y143.505 E.02938
; LINE_WIDTH: 0.839756
G1 F9777.508
G1 X100.754 Y143.517 E.02656
; LINE_WIDTH: 0.852876
G1 F9620.789
G1 X101.799 Y143.498 E.0558
; LINE_WIDTH: 0.858896
G1 F9550.549
G3 X103.109 Y143.496 I.912 J137.419 E.07042
; LINE_WIDTH: 0.857786
G1 F9563.423
G1 X103.3 Y143.519 E.01032
; LINE_WIDTH: 0.810228
G1 F10149.609
G1 X103.491 Y143.543 E.00972
; LINE_WIDTH: 0.76267
G1 F10758.508
G1 X103.682 Y143.567 E.00913
; LINE_WIDTH: 0.715112
G1 F11385.144
G1 X103.872 Y143.591 E.00853
; LINE_WIDTH: 0.667554
G1 F12029.516
G1 X104.063 Y143.615 E.00793
; LINE_WIDTH: 0.619996
G1 F13427.23
G1 X104.463 Y143.615 E.01527
G1 F13446.369
G1 X104.863 Y143.615 E.01527
G1 X156.235 Y143.615 E1.96135
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.706 Y131.955 I29.104 J-43.63 E.60774
G3 X136.912 Y119.083 I41.19 J-33.737 E.57642
G1 X136.092 Y119.382 E.03332
G1 X135.574 Y119.491 E.0202
G3 X133.853 Y119.541 I-1.02 J-5.477 E.06598
G3 X131.636 Y118.796 I.648 J-5.602 E.08995
G3 X130.678 Y118.095 I2.068 J-3.835 E.04547
G1 X129.995 Y117.369 E.03804
G3 X129.081 Y116.142 I77.481 J-58.657 E.05842
G1 X126.698 Y112.93 E.15272
G3 X120.567 Y119.579 I-42.974 J-33.476 E.34571
G3 X118.559 Y121.263 I-1014.768 J-1207.665 E.10003
G1 X117.027 Y122.547 E.07636
G3 X99.494 Y131.453 I-32.799 J-42.862 E.75497
G3 X99.656 Y132.536 I-11.089 J2.205 E.04184
G1 X99.656 Y132.602 E.00252
G1 X102.741 Y132.031 E.11979
G1 X103.227 Y132.108 E.01876
M204 S250
G1 X102.995 Y132.6 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.103 Y132.744 E.00567
G3 X103.109 Y133.571 I-47.501 J.773 E.0262
G1 X103.109 Y133.688 E.0037
G1 X103.109 Y133.805 E.0037
G1 X103.109 Y133.922 E.0037
G1 X103.109 Y134.039 E.0037
G1 X103.109 Y134.156 E.0037
G1 X103.109 Y134.273 E.0037
G1 X103.109 Y134.39 E.0037
G1 F3000
G1 X103.109 Y134.469 E.00251
G1 X103.418 Y135.336 E.02912
G1 F3150
G1 X103.47 Y135.566 E.00748
G1 F3300
G1 X103.522 Y135.796 E.00748
G1 F3450
G1 X103.563 Y135.981 E.006
G1 X103.559 Y136.742 E.02407
G1 X103.525 Y136.883 E.00461
G1 F3300
G1 X103.465 Y137.137 E.00825
G1 F3150
G1 X103.405 Y137.391 E.00825
G1 F3000
G1 X103.38 Y137.499 E.00351
G1 X103.109 Y138.194 E.02363
G1 X103.109 Y138.255 E.00193
G1 F3600
G1 X103.109 Y138.447 E.00609
G1 X103.109 Y138.64 E.00609
G1 X103.109 Y138.832 E.00609
G1 X103.109 Y139.025 E.00609
G1 X103.109 Y139.217 E.00609
G1 X103.109 Y139.41 E.00609
G1 X103.109 Y139.602 E.00609
G1 X103.109 Y141.795 E.06941
G1 X103.053 Y141.943 E.00502
G1 X102.883 Y142.02 E.00591
G1 X102.821 Y142.009 E.00199
G1 X102.759 Y141.998 E.00199
G1 X102.697 Y141.987 E.00199
G1 X102.636 Y141.976 E.00199
G1 X102.574 Y141.965 E.00199
G1 X102.512 Y141.954 E.00199
G1 X102.45 Y141.943 E.00199
G1 X102.388 Y141.932 E.00199
G1 F3000
G1 X102.326 Y141.921 E.00199
G1 F2475
G1 X102.288 Y141.914 E.00122
G1 X101.9 Y141.983 E.01249
G1 F3000
G3 X101.498 Y142.029 I-.436 J-2.006 E.01284
G1 F3150
G1 X101.173 Y142.054 E.0103
G1 F3300
G1 X100.849 Y142.079 E.0103
G1 F3450
G1 X100.771 Y142.085 E.00248
G1 X100.267 Y142.007 E.01615
G1 X100.153 Y141.928 E.00437
G1 F3300
G1 X100.009 Y141.829 E.00554
G1 F3150
G2 X99.62 Y141.622 I-.998 J1.407 E.01398
G1 F3300
G2 X99.109 Y141.395 I-1.813 J3.402 E.01773
G1 X99.109 Y141.509 E.00359
G1 F3450
G1 X99.109 Y141.695 E.00589
G1 F3600
G1 X99.109 Y141.881 E.00589
G1 X99.109 Y142.067 E.00589
G1 X99.109 Y142.253 E.00589
G1 X99.109 Y142.44 E.00589
G3 X98.941 Y143.857 I-6.665 J-.07 E.04526
G1 X98.936 Y144.167 E.00984
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.616 J-43.5 E.51059
G3 X137.192 Y118.315 I40.637 J-33.291 E.49167
G1 X136.612 Y118.616 E.02069
G1 X135.912 Y118.86 E.02347
G1 X135.138 Y119.01 E.02496
G1 X134.416 Y119.042 E.02286
G1 X133.911 Y118.991 E.01608
G3 X132.343 Y118.545 I.604 J-5.104 E.05181
G1 X131.506 Y118.066 E.03055
G1 X131.033 Y117.672 E.01948
G1 X130.409 Y117.003 E.02898
G3 X129.087 Y115.222 I339.429 J-253.368 E.07021
G1 X126.704 Y112.01 E.12664
G3 X121.433 Y117.999 I-42.597 J-32.171 E.25283
G3 X119.329 Y119.896 I-22.079 J-22.377 E.08971
G1 X116.671 Y122.124 E.10981
G3 X98.936 Y131.038 I-32.26 J-42.081 E.63217
G1 X98.946 Y131.528 E.01549
G3 X99.103 Y132.539 I-11.827 J2.355 E.03243
G1 X99.107 Y133.266 E.02301
G1 X102.842 Y132.574 E.12025
G1 X102.906 Y132.585 E.00206
; WIPE_START
M204 S10000
G1 X103.103 Y132.744 E-.09585
G1 X103.108 Y133.491 E-.28415
; WIPE_END
G1 E-.02 F1800
G1 X100.19 Y131.882 Z9.88 F36000
G1 Z9.48
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.707056
G1 F11706.209
G1 X100.495 Y131.807 E.01378
; LINE_WIDTH: 0.743096
G1 F11110.956
G1 X100.8 Y131.732 E.01452
; LINE_WIDTH: 0.787356
G1 F10457.89
G1 X101.123 Y131.65 E.01635
; LINE_WIDTH: 0.831616
G1 F9877.334
G1 X101.446 Y131.568 E.01731
; LINE_WIDTH: 0.875876
G1 F9357.845
G1 X101.769 Y131.485 E.01828
; LINE_WIDTH: 0.91919
G1 F8899.78
G1 X102.046 Y131.412 E.01653
; LINE_WIDTH: 0.962503
G1 F8484.468
G1 X102.323 Y131.339 E.01734
; LINE_WIDTH: 1.00582
G1 F8106.189
G1 X102.6 Y131.265 E.01815
; WIPE_START
G1 X102.323 Y131.339 E-.10889
G1 X102.046 Y131.412 E-.10889
G1 X101.769 Y131.485 E-.10889
G1 X101.633 Y131.52 E-.05333
; WIPE_END
G1 E-.02 F1800
G1 X108.455 Y128.098 Z9.88 F36000
G1 X119.769 Y122.423 Z9.88
G1 Z9.48
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G1 X117.999 Y123.906 E.08815
G2 X121.396 Y126.954 I5.424 J-2.626 E.17871
G2 X123.752 Y127.581 I3.141 J-7.07 E.09348
G1 X124.224 Y127.286 E.02122
G2 X125.166 Y125.212 I-21.089 J-10.837 E.087
G3 X128.937 Y121.509 I6.201 J2.542 E.20739
G3 X131.293 Y120.883 I3.141 J7.071 E.09348
G1 X131.764 Y121.178 E.02122
G3 X132.707 Y123.251 I-21.086 J10.835 E.087
G2 X136.477 Y126.954 I6.201 J-2.542 E.20739
G2 X138.834 Y127.581 I3.141 J-7.07 E.09348
G1 X139.304 Y127.287 E.02115
G1 X140.055 Y128.442 E.0526
G1 X139.776 Y128.424 E.01065
G1 X139.305 Y128.718 E.02122
G2 X138.362 Y130.792 I21.089 J10.837 E.087
G3 X134.592 Y134.495 I-6.201 J-2.542 E.20739
G3 X132.236 Y135.121 I-3.141 J-7.071 E.09348
G1 X131.764 Y134.827 E.02122
G3 X130.822 Y132.753 I21.086 J-10.835 E.087
G2 X127.052 Y129.05 I-6.201 J2.542 E.20739
G2 X124.695 Y128.424 I-3.141 J7.07 E.09348
G1 X124.224 Y128.718 E.02122
G2 X123.281 Y130.792 I21.085 J10.835 E.087
G3 X119.511 Y134.495 I-6.201 J-2.542 E.20739
G3 X117.154 Y135.121 I-3.141 J-7.071 E.09348
G1 X116.683 Y134.827 E.02122
G1 X116.212 Y133.856 E.0412
G2 X114.327 Y130.574 I-10.403 J3.794 E.14521
G2 X110.711 Y128.62 I-5.809 J6.426 E.15843
G3 X108.678 Y129.676 I-20.115 J-36.251 E.08748
G2 X108.2 Y130.792 I11.43 J5.553 E.04637
G3 X105.331 Y134.06 I-6.038 J-2.407 E.16931
G3 X105.331 Y138.611 I-5.477 J2.275 E.17831
G1 X105.331 Y141.553 E.11232
G2 X106.128 Y141.945 I1.742 J-2.534 E.03404
G1 X109.347 Y141.945 E.12292
G2 X110.085 Y140.294 I-16.805 J-8.497 E.06909
G3 X113.855 Y136.591 I6.201 J2.543 E.20739
G3 X116.212 Y135.964 I3.141 J7.071 E.09348
G1 X116.683 Y136.259 E.02122
G3 X117.626 Y138.333 I-21.089 J10.837 E.087
G2 X121.209 Y141.945 I6.042 J-2.409 E.19961
G1 X124.429 Y141.945 E.12292
G2 X125.166 Y140.294 I-16.805 J-8.497 E.06909
G3 X128.937 Y136.591 I6.201 J2.543 E.20739
G3 X131.293 Y135.964 I3.141 J7.071 E.09348
G1 X131.764 Y136.259 E.02122
G3 X132.707 Y138.333 I-21.089 J10.837 E.087
G2 X136.29 Y141.945 I6.042 J-2.409 E.19961
G1 X139.51 Y141.945 E.12292
G1 X139.776 Y141.397 E.0233
G3 X141.662 Y138.115 I10.403 J3.794 E.14521
G3 X145.903 Y135.995 I6 J6.703 E.1832
G1 X146.006 Y135.989 E.00394
G2 X147.402 Y137.428 I33.628 J-31.202 E.07657
G1 X147.788 Y138.333 E.03754
G2 X151.372 Y141.945 I6.042 J-2.409 E.19961
G1 X152.573 Y141.945 E.04588
G3 X151.675 Y141.242 I15.143 J-20.258 E.04356
; CHANGE_LAYER
; Z_HEIGHT: 9.64
; LAYER_HEIGHT: 0.160001
; WIPE_START
G1 F13446.283
G1 X152.463 Y141.859 E-.38
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
G1 X104.073 Y131.264
G1 Z9.64
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.125 Y131.289 E.00219
G1 X104.621 Y131.905 E.03019
G1 X104.81 Y132.48 E.0231
G3 X104.835 Y134.63 I-27.545 J1.4 E.08209
G1 X105.003 Y135.159 E.02122
G3 X105.132 Y136.645 I-6.572 J1.32 E.05706
G1 X105.037 Y137.353 E.02727
G1 X104.835 Y138.069 E.02841
G1 X104.835 Y141.797 E.14232
G1 X104.74 Y142.398 E.02324
G1 X104.715 Y142.443 E.00197
G1 X154.069 Y142.443 E1.88431
G3 X143.803 Y132.702 I31.502 J-43.477 E.5419
G3 X136.271 Y120.545 I41.678 J-34.236 E.54761
G3 X135.169 Y120.737 I-2.499 J-11.102 E.04273
G3 X133.274 Y120.651 I-.566 J-8.52 E.07256
G3 X130.408 Y119.398 I1.536 J-7.416 E.12032
G3 X129.012 Y118.015 I5.44 J-6.888 E.07518
G1 X126.663 Y114.849 E.1505
G3 X121.362 Y120.439 I-43.548 J-35.992 E.29436
G3 X119.312 Y122.16 I-444.229 J-526.935 E.1022
G1 X117.779 Y123.445 E.07636
G3 X103.928 Y131.194 I-33.365 J-43.383 E.60812
G1 X103.992 Y131.225 E.00273
G1 X103.771 Y131.771 F36000
G1 F13446.369
G1 X104.1 Y132.172 E.01978
G1 X104.235 Y132.599 E.01711
G3 X104.249 Y134.721 I-40.43 J1.327 E.08103
G1 X104.422 Y135.26 E.02162
G3 X104.564 Y136.392 I-9.212 J1.731 E.04358
G1 X104.456 Y137.275 E.03395
G1 X104.249 Y137.975 E.02788
G1 X104.249 Y141.797 E.14592
G1 X104.183 Y142.218 E.01626
G1 X103.912 Y142.695 E.02094
G1 X103.43 Y143.029 E.02241
G1 X155.749 Y143.029 E1.99749
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.059 J-44.231 E.59947
G3 X136.596 Y119.828 I41.148 J-33.807 E.56148
G1 X136.064 Y119.995 E.02126
G1 X135.346 Y120.129 E.02789
G1 X134.412 Y120.179 E.03573
G3 X130.774 Y118.941 I.205 J-6.568 E.1489
G1 X130.187 Y118.448 E.02923
G3 X129.065 Y117.104 I7.734 J-7.597 E.06694
G1 X126.682 Y113.891 E.15272
G3 X120.964 Y120.009 I-44.275 J-35.651 E.32
G3 X118.936 Y121.711 I-612.166 J-727.325 E.1011
G1 X117.403 Y122.996 E.07636
G3 X105.781 Y129.813 I-33.036 J-43.01 E.51574
G1 X104.308 Y130.414 E.06074
G1 X103.937 Y130.565 E.01527
; LINE_WIDTH: 0.649791
G1 F12795.413
G1 X103.476 Y130.754 E.02
; LINE_WIDTH: 0.679586
G1 F11776.624
G1 X103.015 Y130.943 E.02097
; LINE_WIDTH: 0.726105
G1 F10160.005
G1 X102.956 Y130.989 E.00337
; LINE_WIDTH: 0.772623
G1 F9928.29
G1 X102.897 Y131.035 E.00359
; LINE_WIDTH: 0.819142
G1 F9699.249
G1 X102.838 Y131.08 E.00382
; LINE_WIDTH: 0.865661
G1 F9472.838
G1 X102.779 Y131.126 E.00404
; LINE_WIDTH: 0.912179
G1 F8970.855
G1 X102.72 Y131.172 E.00427
; LINE_WIDTH: 0.958698
G1 F8519.395
G1 X102.661 Y131.217 E.0045
; LINE_WIDTH: 1.00522
G1 F8111.199
G1 X102.602 Y131.263 E.00472
G1 X102.668 Y131.284 E.00443
; LINE_WIDTH: 0.958698
G1 F8519.395
G1 X102.735 Y131.306 E.00422
; LINE_WIDTH: 0.912179
G1 F8970.855
G1 X102.802 Y131.327 E.004
; LINE_WIDTH: 0.865661
G1 F9472.838
G1 X102.868 Y131.348 E.00379
; LINE_WIDTH: 0.819142
G1 F10034.332
G1 X102.935 Y131.369 E.00358
; LINE_WIDTH: 0.772623
G1 F10252.736
G1 X103.002 Y131.391 E.00337
; LINE_WIDTH: 0.726105
G1 F10473.462
G1 X103.068 Y131.412 E.00316
; LINE_WIDTH: 0.679586
G1 F11670.903
G1 X103.401 Y131.569 E.01545
; LINE_WIDTH: 0.649791
G1 F12795.413
G1 X103.706 Y131.712 E.01353
G1 X103.359 Y132.184 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X103.38 Y132.193 E.00089
G1 X103.585 Y132.452 E.01259
G1 X103.664 Y132.794 E.0134
G1 X103.664 Y134.813 E.07708
G1 X103.864 Y135.439 E.02511
G3 X103.972 Y136.489 I-7.121 J1.257 E.04031
G1 X103.876 Y137.196 E.02727
G1 X103.664 Y137.881 E.02736
G1 X103.664 Y141.397 E.13425
G1 X103.664 Y141.797 E.01527
G1 F13004.032
G1 X103.626 Y142.037 E.00928
G1 F12159.387
G1 X103.471 Y142.309 E.01195
; LINE_WIDTH: 0.666453
G1 F11113.56
G1 X103.375 Y142.403 E.00554
; LINE_WIDTH: 0.71291
G1 F10678.598
G1 X103.279 Y142.497 E.00595
; LINE_WIDTH: 0.759366
G1 F10252.296
G1 X103.183 Y142.591 E.00636
; LINE_WIDTH: 0.807941
G1 F9834.656
G1 X103.111 Y142.636 E.00424
; LINE_WIDTH: 0.856516
G1 F9578.195
G1 X103.04 Y142.68 E.0045
G1 X102.722 Y142.676 E.01704
; LINE_WIDTH: 0.851606
G1 F9635.739
G1 X102.553 Y142.621 E.00951
; LINE_WIDTH: 0.805284
G1 F10214.698
G1 X102.383 Y142.566 E.00897
; LINE_WIDTH: 0.758962
G1 F10781.204
G1 X102.213 Y142.511 E.00843
; LINE_WIDTH: 0.71264
G1 F11363.03
G1 X102.043 Y142.456 E.00789
; LINE_WIDTH: 0.666318
G1 F11960.146
G1 X101.873 Y142.401 E.00735
; LINE_WIDTH: 0.619996
G1 F12709.791
G1 X101.656 Y142.4 E.00832
G1 F13446.369
G1 X101.257 Y142.439 E.01527
G1 X100.841 Y142.479 E.01596
G1 X100.334 Y142.435 E.01942
G3 X99.664 Y142.125 I.561 J-2.093 E.02835
G3 X99.612 Y143.309 I-7.9 J.25 E.04529
G1 X99.555 Y143.615 E.01187
G1 X100.962 Y143.615 E.0537
G1 X101.362 Y143.615 E.01527
G1 F13321.363
G1 X101.762 Y143.615 E.01527
; LINE_WIDTH: 0.666318
G1 F11929.322
G1 X101.939 Y143.591 E.00735
; LINE_WIDTH: 0.71264
G1 F11332.976
G1 X102.116 Y143.568 E.00789
; LINE_WIDTH: 0.758962
G1 F10751.952
G1 X102.293 Y143.545 E.00843
; LINE_WIDTH: 0.805284
G1 F10186.185
G1 X102.47 Y143.522 E.00897
; LINE_WIDTH: 0.851606
G1 F9635.739
G1 X102.647 Y143.499 E.00951
; LINE_WIDTH: 0.856516
G1 F9578.195
G3 X103.302 Y143.52 I.246 J2.54 E.03522
; LINE_WIDTH: 0.809212
G1 F10162.917
G1 X103.492 Y143.544 E.0097
; LINE_WIDTH: 0.761908
G1 F10771.584
G1 X103.683 Y143.567 E.00911
; LINE_WIDTH: 0.714604
G1 F11397.951
G1 X103.873 Y143.591 E.00852
; LINE_WIDTH: 0.6673
G1 F12042.023
G1 X104.064 Y143.615 E.00792
; LINE_WIDTH: 0.619996
G1 F13440.443
G1 X104.464 Y143.615 E.01527
G1 F13446.369
G1 X104.864 Y143.615 E.01527
G1 X156.235 Y143.615 E1.96131
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.706 Y131.955 I29.104 J-43.63 E.60774
G3 X136.912 Y119.083 I41.188 J-33.736 E.57643
G1 X136.093 Y119.382 E.03328
M73 P84 R3
G3 X132.724 Y119.301 I-1.547 J-5.715 E.13043
G3 X131.14 Y118.483 I2.673 J-7.128 E.06823
G1 X130.564 Y118 E.0287
G1 X130.204 Y117.606 E.02039
G3 X129.081 Y116.142 I13.434 J-11.472 E.07046
G1 X126.698 Y112.93 E.15272
G3 X120.566 Y119.579 I-42.951 J-33.455 E.34573
G3 X118.559 Y121.263 I-1029.786 J-1225.586 E.1
G1 X117.027 Y122.547 E.07636
G3 X99.494 Y131.451 I-32.733 J-42.744 E.75497
G3 X99.658 Y132.536 I-10.883 J2.194 E.04193
G1 X99.658 Y132.6 E.00243
G1 X102.743 Y132.028 E.11979
G1 X102.988 Y132.022 E.00932
G1 X103.276 Y132.148 E.01202
M204 S250
G1 X103.029 Y132.62 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.111 Y132.794 E.0061
G1 X103.111 Y133.136 E.01084
G1 X103.111 Y133.373 E.00751
G1 X103.111 Y133.61 E.00751
G1 X103.111 Y133.847 E.00751
G1 X103.111 Y134.085 E.00751
G1 F3450
G1 X103.111 Y134.322 E.00751
G1 F3300
G1 X103.111 Y134.559 E.00751
G1 F3150
G1 X103.111 Y134.796 E.00751
G1 F3000
G1 X103.111 Y134.899 E.00326
G1 X103.338 Y135.608 E.02356
G1 X103.415 Y136.275 E.02126
G1 X103.328 Y137.122 E.02697
G1 X103.111 Y137.792 E.02228
G1 X103.111 Y137.884 E.00292
G1 F3150
G1 X103.111 Y138.123 E.00757
G1 F3300
G1 X103.111 Y138.362 E.00757
G1 F3450
G1 X103.111 Y138.601 E.00757
G1 F3600
G1 X103.111 Y138.841 E.00757
G1 X103.111 Y139.08 E.00757
G1 X103.111 Y139.319 E.00757
G1 X103.111 Y139.558 E.00757
G1 X103.111 Y141.797 E.07089
G1 X103.055 Y141.946 E.00502
G1 X102.885 Y142.023 E.00591
G1 X102.743 Y141.997 E.00457
G1 X102.601 Y141.971 E.00457
G1 X102.46 Y141.946 E.00457
G1 X102.318 Y141.92 E.00457
G1 X102.176 Y141.894 E.00457
G1 X102.034 Y141.869 E.00457
G1 X101.935 Y141.851 E.00318
G1 X101.867 Y141.838 E.00218
G1 F3000
G1 X101.8 Y141.825 E.00218
G1 F2475
G1 X101.275 Y141.874 E.01668
G1 F3000
G1 X100.8 Y141.927 E.01514
G1 X100.298 Y141.851 E.01608
G1 X100.258 Y141.83 E.00141
G1 F3150
G1 X99.409 Y141.39 E.03027
G1 F3300
G1 X99.326 Y141.373 E.00268
G1 F3450
G1 X99.243 Y141.355 E.00268
G1 F3600
G1 X99.16 Y141.338 E.00268
G1 X99.111 Y141.328 E.0016
G1 X99.111 Y141.482 E.00488
G1 X99.111 Y141.863 E.01207
G1 X99.111 Y142.244 E.01207
G3 X98.941 Y143.857 I-6.657 J.115 E.05145
G1 X98.936 Y144.167 E.00984
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.617 J-43.502 E.51059
G3 X137.192 Y118.315 I40.633 J-33.289 E.49167
G3 X134.363 Y119.041 I-2.731 J-4.764 E.09359
G1 X133.609 Y118.959 E.024
G1 X132.865 Y118.766 E.02434
G1 X132.344 Y118.545 E.01794
G1 X131.508 Y118.068 E.03046
G1 X130.92 Y117.577 E.02426
G1 X130.409 Y117.003 E.02434
G3 X129.087 Y115.222 I342.362 J-255.546 E.0702
G1 X126.704 Y112.01 E.12664
G3 X121.894 Y117.543 I-42.587 J-32.156 E.2323
G3 X119.329 Y119.896 I-23.866 J-23.449 E.11027
G1 X116.672 Y122.124 E.10977
G3 X98.936 Y131.038 I-32.253 J-42.07 E.63217
G1 X98.946 Y131.527 E.0155
G3 X99.105 Y132.539 I-11.572 J2.336 E.03245
G1 X99.109 Y133.264 E.02293
G1 X102.844 Y132.572 E.12026
G1 X102.942 Y132.597 E.00319
; WIPE_START
M204 S10000
G1 X103.111 Y132.794 E-.09862
G1 X103.111 Y133.136 E-.13009
G1 X103.111 Y133.373 E-.0901
G1 X103.111 Y133.534 E-.06119
; WIPE_END
G1 E-.02 F1800
G1 X100.893 Y140.838 Z10.04 F36000
G1 X100.231 Y143.021 Z10.04
G1 Z9.64
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.636396
G1 F13080.094
G1 X100.507 Y143.034 E.01085
; LINE_WIDTH: 0.609466
G1 F13692.557
G1 X100.783 Y143.048 E.01036
; LINE_WIDTH: 0.632196
G1 F13171.982
G1 X101.495 Y143.023 E.02778
; LINE_WIDTH: 0.672096
G1 F12347.906
G1 X101.762 Y143.003 E.01112
; WIPE_START
G1 X101.495 Y143.023 E-.10164
G1 X100.783 Y143.048 E-.2709
G1 X100.763 Y143.047 E-.00746
; WIPE_END
G1 E-.02 F1800
G1 X100.371 Y135.424 Z10.04 F36000
G1 X100.189 Y131.882 Z10.04
G1 Z9.64
G1 E.4 F1800
; LINE_WIDTH: 0.70567
G1 F11730.389
G1 X100.5 Y131.805 E.01403
; LINE_WIDTH: 0.744423
G1 F11090.196
G1 X100.811 Y131.727 E.01484
; LINE_WIDTH: 0.783176
G1 F10516.266
G1 X101.123 Y131.65 E.01565
; LINE_WIDTH: 0.82951
G1 F9903.502
G1 X101.445 Y131.567 E.01728
; LINE_WIDTH: 0.875843
G1 F9358.216
G1 X101.768 Y131.483 E.01829
; LINE_WIDTH: 0.922176
G1 F8869.842
G1 X102.091 Y131.4 E.0193
; LINE_WIDTH: 0.963696
G1 F8473.574
G1 X102.346 Y131.331 E.01602
; LINE_WIDTH: 1.00522
G1 F8111.199
G1 X102.602 Y131.263 E.01673
; WIPE_START
G1 X102.346 Y131.331 E-.10045
G1 X102.091 Y131.4 E-.10045
G1 X101.768 Y131.483 E-.12671
G1 X101.635 Y131.518 E-.05239
; WIPE_END
G1 E-.02 F1800
G1 X108.461 Y128.103 Z10.04 F36000
G1 X119.703 Y122.478 Z10.04
G1 Z9.64
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X117.903 Y123.976 I-12.722 J-13.449 E.08949
G2 X123.281 Y127.173 I5.548 J-3.212 E.24953
G1 X123.752 Y127.033 E.01877
G1 X124.224 Y126.607 E.02426
G2 X125.166 Y124.952 I-13.16 J-8.593 E.07274
G3 X130.822 Y121.29 I5.859 J2.851 E.27034
G1 X131.293 Y121.431 E.01877
G1 X131.764 Y121.857 E.02426
G3 X132.707 Y123.511 I-13.16 J8.593 E.07274
G2 X138.362 Y127.173 I5.859 J-2.851 E.27034
G2 X139.037 Y126.849 I-.002 J-.868 E.02956
G2 X140.311 Y128.836 I30.38 J-18.068 E.09014
G1 X139.776 Y128.971 E.02106
G1 X139.305 Y129.397 E.02426
G2 X138.362 Y131.052 I13.159 J8.592 E.07274
G3 X132.707 Y134.714 I-5.859 J-2.851 E.27034
G1 X132.236 Y134.574 E.01877
G1 X131.764 Y134.148 E.02426
G3 X130.822 Y132.493 I13.16 J-8.593 E.07274
G2 X125.166 Y128.831 I-5.859 J2.851 E.27034
G1 X124.695 Y128.971 E.01877
G1 X124.224 Y129.397 E.02426
G2 X123.281 Y131.052 I13.156 J8.591 E.07274
G3 X117.626 Y134.714 I-5.859 J-2.851 E.27034
G1 X117.154 Y134.574 E.01877
G1 X116.683 Y134.148 E.02426
G3 X115.741 Y132.493 I13.16 J-8.593 E.07274
G2 X110.29 Y128.847 I-5.832 J2.823 E.26248
G3 X109.101 Y129.465 I-11.776 J-21.212 E.05118
G2 X108.2 Y131.052 I12.621 J8.216 E.0697
G3 X105.372 Y133.972 I-5.892 J-2.876 E.15779
G1 X105.333 Y134.554 E.02226
G3 X105.333 Y138.149 I-5.09 J1.798 E.13994
G1 X105.333 Y141.488 E.12748
G2 X106.427 Y141.945 I6.826 J-14.79 E.04528
G1 X108.858 Y141.945 E.09282
G1 X109.142 Y141.688 E.01465
G2 X110.085 Y140.034 I-13.16 J-8.593 E.07274
G3 X115.741 Y136.372 I5.859 J2.851 E.27034
G1 X116.212 Y136.512 E.01877
G1 X116.683 Y136.938 E.02426
G3 X117.626 Y138.593 I-13.16 J8.593 E.07274
G2 X121.508 Y141.945 I5.771 J-2.758 E.20152
G1 X123.939 Y141.945 E.09282
G1 X124.224 Y141.688 E.01465
G2 X125.166 Y140.034 I-13.16 J-8.593 E.07274
G3 X130.822 Y136.372 I5.859 J2.851 E.27034
G1 X131.293 Y136.512 E.01877
G1 X131.764 Y136.938 E.02426
G3 X132.707 Y138.593 I-13.16 J8.593 E.07274
G2 X136.589 Y141.945 I5.771 J-2.758 E.20152
G1 X139.021 Y141.945 E.09282
G1 X139.305 Y141.688 E.01464
G2 X140.248 Y140.034 I-13.16 J-8.593 E.07274
G3 X145.903 Y136.372 I5.859 J2.851 E.27034
G1 X146.374 Y136.512 E.01877
G1 X146.846 Y136.938 E.02426
G3 X147.788 Y138.593 I-13.16 J8.593 E.07274
G2 X151.671 Y141.945 I5.771 J-2.758 E.20152
G1 X152.573 Y141.945 E.03447
G3 X151.442 Y141.054 I19.168 J-25.492 E.05498
; CHANGE_LAYER
; Z_HEIGHT: 9.8
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X152.228 Y141.673 E-.38
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
G1 X104.079 Y131.263
G1 Z9.8
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.126 Y131.285 E.00197
G1 X104.622 Y131.902 E.03022
G1 X104.812 Y132.477 E.02312
G3 X104.837 Y133.193 I-9.131 J.684 E.02737
G1 X104.837 Y135.193 E.07636
G1 X104.963 Y135.834 E.02493
G1 X104.963 Y136.765 E.03557
G1 X104.913 Y137.176 E.01577
G2 X104.837 Y137.8 I1.319 J.478 E.02421
G1 X104.837 Y141.8 E.15272
G1 X104.742 Y142.401 E.02324
G1 X104.718 Y142.443 E.00186
G1 X154.069 Y142.443 E1.88417
G3 X143.803 Y132.701 I31.812 J-43.803 E.54189
M73 P84 R2
G3 X136.269 Y120.54 I42.21 J-34.565 E.54775
G3 X135.158 Y120.737 I-1.722 J-6.478 E.04311
G3 X133.273 Y120.651 I-.566 J-8.282 E.0722
G3 X130.408 Y119.398 I1.538 J-7.417 E.12028
G3 X129.118 Y118.146 I5.483 J-6.939 E.06875
G3 X127.855 Y116.455 I53.995 J-41.659 E.08056
G1 X126.663 Y114.849 E.07636
G3 X121.362 Y120.439 I-43.285 J-35.743 E.29436
G3 X119.312 Y122.16 I-444.376 J-527.106 E.10219
G1 X117.779 Y123.445 E.07636
G3 X103.932 Y131.193 I-33.691 J-43.966 E.60788
G1 X103.998 Y131.224 E.00279
G1 X103.771 Y131.766 F36000
G1 F13446.369
G1 X104.101 Y132.169 E.01989
G1 X104.237 Y132.596 E.01712
G3 X104.251 Y133.255 I-12.493 J.596 E.02518
G1 X104.251 Y135.255 E.07636
G1 X104.363 Y135.775 E.0203
G1 X104.377 Y136.757 E.03749
G1 X104.251 Y137.44 E.02652
G1 X104.251 Y141.8 E.16642
G1 X104.185 Y142.22 E.01626
G1 X103.914 Y142.697 E.02094
G1 X103.435 Y143.029 E.02225
G1 X155.749 Y143.029 E1.99727
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I29.803 J-43.952 E.59953
G3 X136.598 Y119.835 I41.148 J-33.806 E.56118
G1 X135.96 Y120.02 E.02539
G1 X135.077 Y120.157 E.0341
G3 X130.774 Y118.941 I-.488 J-6.495 E.1743
G3 X129.065 Y117.104 I5.224 J-6.572 E.09614
G1 X126.682 Y113.891 E.15272
G3 X120.964 Y120.009 I-44.493 J-35.854 E.31999
G3 X118.936 Y121.711 I-610.847 J-725.747 E.1011
G1 X117.403 Y122.996 E.07636
G3 X105.781 Y129.813 I-33.036 J-43.011 E.51575
G1 X104.308 Y130.414 E.06073
G1 X103.937 Y130.565 E.01527
; LINE_WIDTH: 0.650541
G1 F12779.839
G1 X103.476 Y130.755 E.02003
; LINE_WIDTH: 0.681086
G1 F11776.19
G1 X103.015 Y130.944 E.02102
; LINE_WIDTH: 0.727153
G1 F10159.59
G1 X102.957 Y130.989 E.00335
; LINE_WIDTH: 0.773221
G1 F9929.207
G1 X102.898 Y131.035 E.00357
; LINE_WIDTH: 0.819288
G1 F9701.465
G1 X102.839 Y131.08 E.0038
; LINE_WIDTH: 0.865355
G1 F9476.324
G1 X102.78 Y131.125 E.00402
; LINE_WIDTH: 0.911422
G1 F8978.599
G1 X102.721 Y131.171 E.00424
; LINE_WIDTH: 0.957489
G1 F8530.549
G1 X102.663 Y131.216 E.00446
; LINE_WIDTH: 1.00356
G1 F8125.091
G1 X102.604 Y131.261 E.00469
G1 X102.67 Y131.282 E.0044
; LINE_WIDTH: 0.957489
G1 F8530.549
G1 X102.737 Y131.303 E.00419
; LINE_WIDTH: 0.911422
G1 F8978.599
G1 X102.803 Y131.324 E.00398
; LINE_WIDTH: 0.865355
G1 F9476.324
G1 X102.869 Y131.345 E.00377
; LINE_WIDTH: 0.819288
G1 F10032.469
G1 X102.936 Y131.366 E.00356
; LINE_WIDTH: 0.773221
G1 F10249.709
G1 X103.002 Y131.387 E.00335
; LINE_WIDTH: 0.727153
G1 F10469.246
G1 X103.069 Y131.408 E.00314
; LINE_WIDTH: 0.681086
G1 F11668.002
G1 X103.401 Y131.565 E.01551
; LINE_WIDTH: 0.650541
G1 F12779.839
G1 X103.705 Y131.708 E.01348
G1 X103.362 Y132.182 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X103.382 Y132.19 E.00082
G1 X103.587 Y132.449 E.0126
G1 X103.666 Y132.791 E.01342
G1 X103.666 Y135.318 E.09646
G1 X103.784 Y135.866 E.02142
G1 X103.792 Y136.749 E.03371
G1 X103.666 Y137.374 E.02433
G1 X103.666 Y141.4 E.15369
G1 X103.666 Y141.8 E.01527
G1 F13020.293
G1 X103.628 Y142.04 E.00928
G1 F12175.112
G1 X103.473 Y142.312 E.01195
; LINE_WIDTH: 0.66624
G1 F11128.594
G1 X103.377 Y142.406 E.00554
; LINE_WIDTH: 0.712483
G1 F10693.466
G1 X103.281 Y142.5 E.00594
; LINE_WIDTH: 0.758726
G1 F10267.016
G1 X103.185 Y142.594 E.00635
; LINE_WIDTH: 0.806996
G1 F9849.2
G1 X103.113 Y142.638 E.00423
; LINE_WIDTH: 0.855266
G1 F9592.78
G1 X103.042 Y142.682 E.00449
G1 X102.725 Y142.678 E.017
; LINE_WIDTH: 0.850386
G1 F9650.144
G1 X102.443 Y142.603 E.01552
; LINE_WIDTH: 0.804308
G1 F10227.645
G1 X102.161 Y142.527 E.01464
; LINE_WIDTH: 0.75823
G1 F10878.666
G1 X101.879 Y142.451 E.01376
; LINE_WIDTH: 0.712152
G1 F11618.199
G1 X101.598 Y142.376 E.01289
; LINE_WIDTH: 0.666074
G1 F12465.611
G1 X101.316 Y142.3 E.01201
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X100.812 Y142.323 E.01926
G3 X100.26 Y142.259 I.102 J-3.283 E.02123
G1 X99.747 Y142.012 E.02176
G1 X99.666 Y141.995 E.00317
G3 X99.614 Y143.308 I-8.66 J.315 E.05024
G1 X99.556 Y143.615 E.01189
G1 X100.795 Y143.615 E.0473
G1 X101.195 Y143.615 E.01527
; LINE_WIDTH: 0.666074
G1 F12465.611
G1 X101.486 Y143.592 E.01201
; LINE_WIDTH: 0.712152
G1 F11618.199
G1 X101.777 Y143.568 E.01289
; LINE_WIDTH: 0.75823
G1 F10878.666
G1 X102.068 Y143.545 E.01377
; LINE_WIDTH: 0.804308
G1 F10227.645
G1 X102.358 Y143.522 E.01464
; LINE_WIDTH: 0.850386
G1 F9650.144
G1 X102.649 Y143.499 E.01552
; LINE_WIDTH: 0.855266
G1 F9592.78
G3 X103.303 Y143.52 I.245 J2.55 E.03513
; LINE_WIDTH: 0.808212
G1 F10176.049
G1 X103.494 Y143.544 E.00968
; LINE_WIDTH: 0.761158
G1 F10784.449
G1 X103.684 Y143.568 E.00909
; LINE_WIDTH: 0.714104
G1 F11410.541
G1 X103.875 Y143.591 E.0085
; LINE_WIDTH: 0.66705
G1 F12054.267
G1 X104.065 Y143.615 E.00791
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.465 Y143.615 E.01527
G1 X156.235 Y143.615 E1.97655
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.326 J-43.873 E.60774
G3 X136.908 Y119.075 I40.744 J-33.466 E.57676
G1 X136.501 Y119.253 E.01698
G1 X135.796 Y119.458 E.02799
G1 X134.997 Y119.577 E.03087
G3 X131.634 Y118.795 I-.433 J-5.757 E.13388
G3 X130.676 Y118.094 I2.073 J-3.839 E.04546
G1 X129.995 Y117.369 E.03795
G3 X129.081 Y116.142 I77.58 J-58.731 E.05842
G1 X126.698 Y112.93 E.15272
G3 X120.566 Y119.579 I-42.953 J-33.456 E.34573
G3 X118.559 Y121.263 I-1030.809 J-1226.801 E.1
G1 X117.027 Y122.547 E.07636
G3 X99.495 Y131.451 I-32.761 J-42.798 E.75495
G3 X99.66 Y132.536 I-10.761 J2.193 E.04194
G1 X99.66 Y132.597 E.00233
G1 X102.746 Y132.026 E.1198
G1 X102.989 Y132.019 E.00928
G1 X103.28 Y132.146 E.01211
M204 S250
G1 X103.031 Y132.617 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.113 Y132.791 E.0061
G1 X103.113 Y133.61 E.02592
G1 X103.113 Y133.843 E.00739
G1 X103.113 Y134.077 E.00739
G1 X103.113 Y134.31 E.00739
G1 X103.113 Y134.543 E.00739
G1 F3450
G1 X103.113 Y134.776 E.00739
G1 F3300
G1 X103.113 Y135.01 E.00739
G1 F3150
G1 X103.113 Y135.243 E.00739
G1 F3000
G1 X103.113 Y135.377 E.00423
G1 X103.245 Y135.988 E.0198
G1 X103.239 Y136.742 E.02386
G1 X103.113 Y137.343 E.01945
G1 F3600
G1 X103.113 Y137.4 E.00181
G1 X103.113 Y137.457 E.00181
G1 X103.113 Y137.514 E.00181
G1 X103.113 Y137.571 E.00181
G1 X103.113 Y137.628 E.00181
G1 X103.113 Y137.685 E.00181
G1 X103.113 Y137.742 E.00181
G1 X103.113 Y141.8 E.12845
G1 X103.057 Y141.948 E.00502
G3 X102.668 Y141.988 I-.23 J-.321 E.01295
G1 X102.49 Y141.956 E.00573
G1 X102.313 Y141.923 E.00573
G1 X102.135 Y141.89 E.00573
G1 X101.957 Y141.857 E.00573
G1 X101.779 Y141.824 E.00573
G1 X101.601 Y141.791 E.00573
G1 X101.423 Y141.758 E.00573
G1 F3000
G1 X101.245 Y141.725 E.00573
G1 F2475
G1 X101.197 Y141.728 E.00152
G1 F3000
G1 X100.787 Y141.769 E.01307
G1 X100.329 Y141.696 E.01467
G1 F3150
G1 X100.022 Y141.552 E.01075
G1 F3300
G2 X99.789 Y141.455 I-.303 J.398 E.00808
G1 F3450
G1 X99.676 Y141.434 E.00363
G1 F3600
G1 X99.563 Y141.413 E.00363
G1 X99.451 Y141.393 E.00363
G1 X99.338 Y141.372 E.00363
G1 X99.226 Y141.351 E.00363
G1 X99.113 Y141.33 E.00363
G1 X99.113 Y142.626 E.04102
G3 X98.941 Y143.857 I-5.044 J-.075 E.03945
G1 X98.936 Y144.167 E.00984
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.616 J-43.501 E.51062
G3 X137.192 Y118.316 I40.272 J-33.072 E.49163
G3 X136.346 Y118.722 I-5.215 J-9.78 E.02972
G1 X135.642 Y118.927 E.02321
G1 X134.921 Y119.03 E.02308
G1 X134.365 Y119.041 E.01759
G1 X133.608 Y118.959 E.02412
G1 X132.867 Y118.767 E.02421
G1 X132.341 Y118.544 E.01809
G1 X131.508 Y118.068 E.03038
G1 X131.031 Y117.671 E.01965
G1 X130.409 Y117.003 E.0289
G3 X129.087 Y115.222 I338.541 J-252.712 E.07021
G1 X126.704 Y112.01 E.12664
G3 X121.433 Y117.999 I-42.503 J-32.088 E.25282
G3 X119.327 Y119.897 I-22.082 J-22.38 E.0898
G1 X116.672 Y122.124 E.10972
G3 X98.936 Y131.038 I-32.253 J-42.07 E.63218
G1 X98.946 Y131.527 E.01548
G3 X99.107 Y132.539 I-11.458 J2.339 E.03248
G1 X99.111 Y133.261 E.02285
G1 X102.846 Y132.569 E.12026
G1 X102.944 Y132.595 E.00318
; WIPE_START
M204 S10000
G1 X103.113 Y132.791 E-.09869
G1 X103.113 Y133.532 E-.28131
; WIPE_END
G1 E-.02 F1800
G1 X100.938 Y140.847 Z10.2 F36000
G1 X100.314 Y142.946 Z10.2
G1 Z9.8
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.786696
G1 F10467.064
G1 X100.973 Y142.964 E.03233
; LINE_WIDTH: 0.774076
G1 F10645.633
G1 X101.195 Y142.952 E.01075
; WIPE_START
G1 X100.973 Y142.964 E-.09601
G1 X100.314 Y142.946 E-.28399
; WIPE_END
G1 E-.02 F1800
G1 X100.228 Y135.314 Z10.2 F36000
G1 X100.189 Y131.881 Z10.2
G1 Z9.8
G1 E.4 F1800
; LINE_WIDTH: 0.703616
G1 F11766.376
G1 X100.5 Y131.804 E.01399
; LINE_WIDTH: 0.742356
G1 F11122.568
G1 X100.811 Y131.726 E.01479
; LINE_WIDTH: 0.781096
G1 F10545.557
G1 X101.122 Y131.649 E.0156
; LINE_WIDTH: 0.82743
G1 F9929.475
G1 X101.445 Y131.566 E.01724
; LINE_WIDTH: 0.873763
G1 F9381.404
G1 X101.768 Y131.482 E.01825
; LINE_WIDTH: 0.920096
G1 F8890.671
G1 X102.091 Y131.399 E.01925
; LINE_WIDTH: 0.961826
G1 F8490.658
G1 X102.347 Y131.33 E.01606
; LINE_WIDTH: 1.00356
G1 F8125.091
G1 X102.604 Y131.261 E.01679
; WIPE_START
G1 X102.347 Y131.33 E-.10095
G1 X102.091 Y131.399 E-.10095
G1 X101.768 Y131.482 E-.1267
G1 X101.637 Y131.516 E-.05139
; WIPE_END
G1 E-.02 F1800
G1 X108.466 Y128.108 Z10.2 F36000
G1 X119.635 Y122.535 Z10.2
G1 Z9.8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X117.833 Y124.031 I-14.637 J-15.805 E.08947
G2 X122.339 Y126.912 I5.223 J-3.204 E.21132
M73 P85 R2
G2 X124.224 Y126.19 I.317 J-1.995 E.08071
G1 X125.166 Y124.74 E.06604
G3 X129.879 Y121.551 I5.485 J3.03 E.2255
G3 X131.764 Y122.273 I.317 J1.995 E.08071
G1 X132.707 Y123.724 E.06604
G2 X137.42 Y126.912 I5.485 J-3.03 E.2255
G2 X138.884 Y126.595 I.319 J-2.06 E.05851
G2 X140.495 Y129.106 I61.712 J-37.822 E.11392
G2 X139.305 Y129.814 I.296 J1.851 E.05416
G1 X138.362 Y131.265 E.06604
G3 X133.65 Y134.453 I-5.485 J-3.03 E.2255
G3 X131.764 Y133.731 I-.317 J-1.995 E.08071
G1 X130.822 Y132.28 E.06604
G2 X126.109 Y129.092 I-5.485 J3.03 E.2255
G2 X124.224 Y129.814 I-.317 J1.995 E.08071
G1 X123.281 Y131.265 E.06604
G3 X118.568 Y134.453 I-5.485 J-3.03 E.2255
G3 X116.683 Y133.731 I-.317 J-1.995 E.08071
G1 X115.741 Y132.28 E.06604
G2 X111.028 Y129.092 I-5.485 J3.03 E.2255
G2 X109.142 Y129.814 I-.317 J1.995 E.08071
G1 X108.2 Y131.265 E.06604
G3 X105.372 Y133.911 I-5.629 J-3.18 E.15018
G1 X105.335 Y135.14 E.04695
G3 X105.335 Y137.56 I-6.973 J1.21 E.09286
G1 X105.335 Y141.431 E.14776
G2 X108.188 Y141.945 I2.27 J-4.414 E.11233
G2 X109.142 Y141.272 I-.851 J-2.218 E.04507
G1 X110.085 Y139.821 E.06604
G3 X114.798 Y136.633 I5.485 J3.03 E.2255
G3 X116.683 Y137.355 I.317 J1.995 E.08071
G2 X118.097 Y139.504 I64.153 J-40.658 E.09823
G2 X123.269 Y141.945 I4.739 J-3.341 E.22836
G2 X124.224 Y141.272 I-.851 J-2.218 E.04507
G1 X125.166 Y139.821 E.06604
G3 X129.879 Y136.633 I5.485 J3.03 E.2255
G3 X131.764 Y137.355 I.317 J1.995 E.08071
G2 X133.178 Y139.504 I64.153 J-40.658 E.09823
G2 X138.351 Y141.945 I4.739 J-3.342 E.22836
G2 X139.305 Y141.272 I-.851 J-2.219 E.04507
G1 X140.248 Y139.821 E.06604
G3 X144.961 Y136.633 I5.485 J3.03 E.2255
G3 X146.846 Y137.355 I.317 J1.995 E.08071
G1 X147.788 Y138.805 E.06604
G2 X152.574 Y141.945 I5.3 J-2.861 E.22772
G3 X150.983 Y140.678 I32.781 J-42.77 E.07765
; CHANGE_LAYER
; Z_HEIGHT: 9.96
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X151.765 Y141.301 E-.38
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
G1 X104.156 Y131.318
G1 Z9.96
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.624 Y131.898 E.02847
G1 X104.814 Y132.474 E.02314
G3 X104.839 Y133.802 I-16.889 J.989 E.05072
G1 X104.839 Y141.802 E.30543
G1 X104.744 Y142.403 E.02324
G1 X104.722 Y142.443 E.00175
G1 X154.069 Y142.443 E1.88404
G3 X143.803 Y132.702 I31.812 J-43.804 E.54185
G3 X136.269 Y120.54 I41.836 J-34.334 E.54782
G3 X135.154 Y120.738 I-1.713 J-6.422 E.04326
G3 X133.339 Y120.662 I-.562 J-8.371 E.0695
G1 X132.429 Y120.434 E.03581
G3 X130.472 Y119.447 I3.014 J-8.409 E.0839
G3 X129.012 Y118.015 I5.817 J-7.393 E.07823
G1 X126.663 Y114.849 E.15052
G3 X121.363 Y120.437 I-42.697 J-35.185 E.29429
G3 X119.312 Y122.16 I-429.229 J-509.031 E.10228
G1 X117.779 Y123.445 E.07636
G3 X103.937 Y131.191 I-33.694 J-43.97 E.60769
G1 X104.088 Y131.263 E.00638
G1 X103.77 Y131.761 F36000
G1 F13446.369
G1 X104.103 Y132.166 E.02001
G1 X104.239 Y132.593 E.01714
G3 X104.254 Y133.802 I-22.829 J.87 E.04615
G1 X104.254 Y141.802 E.30543
G1 X104.187 Y142.223 E.01626
G1 X103.917 Y142.7 E.02094
G1 X103.441 Y143.029 E.02208
G1 X155.749 Y143.029 E1.99705
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.255 Y132.328 I29.803 J-43.952 E.59949
G3 X136.598 Y119.835 I41.803 J-34.209 E.56117
G1 X135.96 Y120.02 E.0254
G1 X135.074 Y120.158 E.03423
G3 X133.287 Y120.054 I-.458 J-7.6 E.06848
G1 X132.566 Y119.865 E.02847
G3 X130.819 Y118.975 I2.911 J-7.88 E.07503
G3 X129.065 Y117.104 I5.01 J-6.453 E.09833
G1 X126.682 Y113.891 E.15272
G3 X120.965 Y120.008 I-42.661 J-34.142 E.31996
G3 X118.936 Y121.711 I-593.018 J-704.473 E.10116
G1 X117.403 Y122.996 E.07636
G3 X105.781 Y129.813 I-33.038 J-43.014 E.51574
G1 X104.308 Y130.414 E.06075
G1 X103.937 Y130.565 E.01527
; LINE_WIDTH: 0.651291
G1 F12764.303
G1 X103.476 Y130.755 E.02005
; LINE_WIDTH: 0.682586
G1 F11775.813
G1 X103.016 Y130.945 E.02107
; LINE_WIDTH: 0.728199
G1 F10159.27
G1 X102.957 Y130.99 E.00334
; LINE_WIDTH: 0.773812
G1 F9930.215
G1 X102.899 Y131.035 E.00356
; LINE_WIDTH: 0.819425
G1 F9703.772
G1 X102.84 Y131.08 E.00378
; LINE_WIDTH: 0.865038
G1 F9479.941
G1 X102.782 Y131.125 E.00399
; LINE_WIDTH: 0.910651
G1 F8986.503
G1 X102.723 Y131.17 E.00421
; LINE_WIDTH: 0.956263
G1 F8541.891
G1 X102.665 Y131.215 E.00443
; LINE_WIDTH: 1.00188
G1 F8139.199
G1 X102.606 Y131.26 E.00465
G1 X102.672 Y131.28 E.00437
; LINE_WIDTH: 0.956263
G1 F8541.891
G1 X102.738 Y131.301 E.00416
; LINE_WIDTH: 0.910651
G1 F8986.503
G1 X102.804 Y131.322 E.00396
; LINE_WIDTH: 0.865038
G1 F9479.941
G1 X102.871 Y131.343 E.00375
; LINE_WIDTH: 0.819425
G1 F10030.716
G1 X102.937 Y131.363 E.00354
; LINE_WIDTH: 0.773812
G1 F10246.765
G1 X103.003 Y131.384 E.00334
; LINE_WIDTH: 0.728199
G1 F10465.116
G1 X103.069 Y131.405 E.00313
; LINE_WIDTH: 0.682586
G1 F11665.193
G1 X103.402 Y131.562 E.01557
; LINE_WIDTH: 0.651291
G1 F12764.303
G1 X103.704 Y131.704 E.01341
G1 X103.366 Y132.179 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X103.383 Y132.187 E.00073
G1 X103.588 Y132.446 E.01262
G1 X103.668 Y132.789 E.01343
G1 X103.668 Y141.402 E.32885
G1 X103.668 Y141.802 E.01527
G1 F13036.68
G1 X103.63 Y142.042 E.00928
G1 F12190.958
G1 X103.476 Y142.314 E.01195
; LINE_WIDTH: 0.66602
G1 F11143.772
G1 X103.379 Y142.408 E.00553
; LINE_WIDTH: 0.712043
G1 F10708.478
G1 X103.283 Y142.502 E.00594
; LINE_WIDTH: 0.758066
G1 F10281.855
G1 X103.187 Y142.596 E.00634
; LINE_WIDTH: 0.806036
G1 F9863.882
G1 X103.115 Y142.64 E.00422
; LINE_WIDTH: 0.854006
G1 F9607.526
G1 X103.044 Y142.684 E.00448
G1 X102.727 Y142.68 E.01696
; LINE_WIDTH: 0.849156
G1 F9664.711
G1 X102.468 Y142.618 E.01415
; LINE_WIDTH: 0.822216
G1 F9995.177
G1 X102.079 Y142.526 E.02054
; LINE_WIDTH: 0.781772
G1 F10536.02
G1 X101.689 Y142.433 E.01949
; LINE_WIDTH: 0.741328
G1 F11138.74
G1 X101.3 Y142.341 E.01844
; LINE_WIDTH: 0.700884
G1 F11814.605
G1 X100.911 Y142.248 E.01738
; LINE_WIDTH: 0.66044
G1 F12577.784
G1 X100.522 Y142.155 E.01633
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X100.129 Y142.083 E.01527
G1 X99.668 Y141.997 E.01789
G3 X99.615 Y143.308 I-8.546 J.314 E.05013
G1 X99.557 Y143.615 E.01192
G1 X100.388 Y143.615 E.03172
; LINE_WIDTH: 0.646926
G1 F12855.256
G1 X100.654 Y143.601 E.01064
; LINE_WIDTH: 0.687372
G1 F12059.061
G1 X101.054 Y143.581 E.01703
; LINE_WIDTH: 0.727818
G1 F11355.738
G1 X101.453 Y143.561 E.01808
; LINE_WIDTH: 0.768264
G1 F10729.937
G1 X101.853 Y143.54 E.01914
; LINE_WIDTH: 0.80871
G1 F10169.506
G1 X102.252 Y143.52 E.02019
; LINE_WIDTH: 0.849156
G1 F9664.711
G1 X102.652 Y143.5 E.02125
; LINE_WIDTH: 0.854006
G1 F9607.526
G3 X103.305 Y143.521 I.245 J2.559 E.03504
; LINE_WIDTH: 0.807204
G1 F10189.321
G1 X103.495 Y143.544 E.00966
; LINE_WIDTH: 0.760402
G1 F10797.486
G1 X103.686 Y143.568 E.00907
; LINE_WIDTH: 0.7136
G1 F11423.31
G1 X103.876 Y143.591 E.00849
; LINE_WIDTH: 0.666798
G1 F12066.73
G1 X104.066 Y143.615 E.0079
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.466 Y143.615 E.01527
G1 X156.235 Y143.615 E1.97651
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.706 Y131.955 I29.325 J-43.872 E.6077
G3 X136.908 Y119.075 I41.344 J-33.83 E.57675
G1 X136.5 Y119.253 E.01699
G1 X135.796 Y119.458 E.02799
G1 X134.993 Y119.577 E.031
G1 X134.375 Y119.594 E.02361
G1 X133.549 Y119.509 E.03172
G1 X132.727 Y119.302 E.03235
G3 X131.166 Y118.503 I2.723 J-7.253 E.06711
G3 X129.953 Y117.318 I4.86 J-6.184 E.06488
G1 X126.698 Y112.93 E.20859
G3 X120.567 Y119.579 I-42.971 J-33.472 E.3457
G3 X118.559 Y121.263 I-998.464 J-1188.209 E.10004
G1 X117.027 Y122.547 E.07636
G3 X99.495 Y131.45 I-32.844 J-42.962 E.75491
G3 X99.662 Y132.536 I-10.638 J2.191 E.04195
G1 X99.662 Y132.595 E.00224
G1 X102.748 Y132.023 E.1198
G1 X102.989 Y132.017 E.00924
G1 X103.283 Y132.144 E.01222
M204 S250
G1 X103.033 Y132.614 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.115 Y132.789 E.00611
G1 X103.115 Y134.153 E.04319
G1 X103.115 Y134.504 E.01111
G1 X103.115 Y134.855 E.01111
G1 X103.115 Y135.205 E.01111
G1 X103.115 Y135.556 E.01111
G1 F3450
G1 X103.115 Y136.048 E.01557
G1 F3600
G1 X103.115 Y136.399 E.01111
G1 X103.115 Y136.75 E.01111
G1 X103.115 Y137.1 E.01111
G1 X103.115 Y137.451 E.01111
; LINE_WIDTH: 0.519986
G1 X103.115 Y141.802 E.13774
G1 X103.059 Y141.951 E.00502
G1 X102.889 Y142.028 E.00591
G1 X102.266 Y141.916 E.02006
G1 X101.683 Y141.808 E.01876
G1 X101.101 Y141.7 E.01876
; LINE_WIDTH: 0.519996
G1 X101.06 Y141.693 E.0013
G1 X100.412 Y141.573 E.02088
G1 X99.763 Y141.453 E.02088
G1 X99.115 Y141.333 E.02088
G1 X99.115 Y142.626 E.04094
G3 X98.941 Y143.857 I-5.002 J-.077 E.03946
G1 X98.936 Y144.167 E.00984
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.617 J-43.502 E.51058
G3 X137.192 Y118.316 I40.596 J-33.267 E.49164
G3 X136.346 Y118.722 I-5.225 J-9.801 E.02973
G1 X135.642 Y118.927 E.02321
G1 X134.917 Y119.03 E.02318
G1 X134.365 Y119.041 E.01748
G1 X133.609 Y118.959 E.02407
G1 X132.868 Y118.767 E.02426
G1 X132.337 Y118.543 E.01824
G1 X131.508 Y118.068 E.03025
G3 X130.278 Y116.829 I3.698 J-4.899 E.05546
G1 X126.704 Y112.01 E.18996
G3 X121.466 Y117.967 I-42.495 J-32.082 E.25139
G3 X119.323 Y119.901 I-20.332 J-20.374 E.09141
G1 X116.672 Y122.124 E.10954
G3 X98.936 Y131.038 I-32.254 J-42.072 E.63217
G1 X98.946 Y131.526 E.01546
G3 X99.109 Y132.539 I-11.323 J2.338 E.03251
G1 X99.113 Y133.259 E.02277
G1 X102.848 Y132.567 E.12026
G1 X102.945 Y132.592 E.00318
; WIPE_START
M204 S10000
G1 X103.115 Y132.789 E-.09876
G1 X103.115 Y133.529 E-.28124
; WIPE_END
G1 E-.02 F1800
G1 X100.998 Y140.862 Z10.36 F36000
G1 X100.443 Y142.783 Z10.36
G1 Z9.96
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.920176
G1 F8889.868
G1 X100.333 Y142.783 E.00639
G1 X100.277 Y142.879 E.00639
G1 X100.333 Y142.975 E.00639
G1 X100.443 Y142.975 E.00639
G1 X100.499 Y142.879 E.00639
; WIPE_START
G1 X100.443 Y142.975 E-.07601
G1 X100.333 Y142.975 E-.076
G1 X100.277 Y142.879 E-.076
G1 X100.333 Y142.783 E-.07601
G1 X100.443 Y142.783 E-.07599
; WIPE_END
G1 E-.02 F1800
G1 X100.265 Y135.153 Z10.36 F36000
G1 X100.189 Y131.88 Z10.36
G1 Z9.96
G1 E.4 F1800
; LINE_WIDTH: 0.70157
G1 F11802.469
G1 X100.5 Y131.802 E.01394
; LINE_WIDTH: 0.740303
G1 F11154.918
G1 X100.811 Y131.725 E.01475
; LINE_WIDTH: 0.779036
G1 F10574.729
G1 X101.122 Y131.648 E.01556
; LINE_WIDTH: 0.825356
G1 F9955.501
G1 X101.445 Y131.565 E.01719
; LINE_WIDTH: 0.871676
G1 F9404.783
G1 X101.768 Y131.481 E.0182
; LINE_WIDTH: 0.917996
G1 F8911.8
G1 X102.09 Y131.398 E.0192
; LINE_WIDTH: 0.959936
G1 F8507.995
G1 X102.348 Y131.329 E.01611
; LINE_WIDTH: 1.00188
G1 F8139.199
G1 X102.606 Y131.26 E.01684
; WIPE_START
G1 X102.348 Y131.329 E-.10146
G1 X102.09 Y131.398 E-.10146
G1 X101.768 Y131.481 E-.12667
G1 X101.639 Y131.514 E-.05041
; WIPE_END
G1 E-.02 F1800
G1 X108.472 Y128.114 Z10.36 F36000
G1 X119.565 Y122.594 Z10.36
G1 Z9.96
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X117.759 Y124.086 I-14.597 J-15.829 E.08947
G2 X122.339 Y126.742 I4.939 J-3.24 E.20959
G2 X124.224 Y125.871 I.116 J-2.223 E.08249
G1 X125.166 Y124.556 E.06177
G3 X129.879 Y121.722 I5.104 J3.153 E.2181
G3 X131.764 Y122.593 I.116 J2.223 E.08249
G1 X132.707 Y123.908 E.06177
G2 X136.477 Y126.616 I5.402 J-3.543 E.18128
G2 X138.755 Y126.374 I.854 J-2.804 E.08983
G2 X140.641 Y129.308 I43.219 J-25.718 E.1332
G2 X139.305 Y130.134 I.495 J2.294 E.06115
G1 X138.362 Y131.449 E.06177
G3 X133.65 Y134.282 I-5.104 J-3.153 E.2181
G3 X131.764 Y133.411 I-.116 J-2.223 E.08249
G1 X130.822 Y132.096 E.06177
G2 X126.109 Y129.263 I-5.104 J3.153 E.2181
G2 X124.224 Y130.134 I-.116 J2.223 E.08249
G1 X123.281 Y131.449 E.06177
G3 X118.568 Y134.282 I-5.104 J-3.153 E.2181
G3 X116.683 Y133.411 I-.116 J-2.223 E.08249
G1 X115.741 Y132.096 E.06177
G2 X111.028 Y129.263 I-5.104 J3.153 E.2181
G2 X109.142 Y130.134 I-.116 J2.223 E.08249
G1 X108.2 Y131.449 E.06177
G3 X105.372 Y133.858 I-5.336 J-3.399 E.14395
G1 X105.337 Y141.38 E.28721
G2 X107.257 Y141.823 I2.689 J-7.277 E.07544
G2 X109.142 Y140.952 I.116 J-2.224 E.08249
G1 X110.085 Y139.637 E.06177
G3 X114.798 Y136.803 I5.104 J3.153 E.2181
G3 X116.683 Y137.674 I.116 J2.224 E.08249
G1 X117.626 Y138.989 E.06177
G2 X122.339 Y141.823 I5.104 J-3.153 E.2181
G2 X124.224 Y140.952 I.116 J-2.224 E.08249
G1 X125.166 Y139.637 E.06177
G3 X129.879 Y136.803 I5.104 J3.153 E.2181
G3 X131.764 Y137.674 I.116 J2.224 E.08249
G1 X132.707 Y138.989 E.06177
G2 X137.42 Y141.823 I5.104 J-3.153 E.2181
G2 X139.305 Y140.952 I.116 J-2.224 E.08249
G1 X140.248 Y139.637 E.06177
G3 X144.961 Y136.803 I5.104 J3.153 E.2181
G3 X146.846 Y137.674 I.116 J2.224 E.08249
G1 X147.788 Y138.989 E.06177
G2 X152.396 Y141.809 I5.085 J-3.135 E.21402
G3 X150.576 Y140.335 I31.253 J-40.463 E.08944
; CHANGE_LAYER
; Z_HEIGHT: 10.12
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X151.353 Y140.964 E-.38
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
G1 X104.022 Y131.228
G1 Z10.12
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.127 Y131.278 E.00445
G1 X104.626 Y131.895 E.03029
G1 X104.816 Y132.471 E.02316
G3 X104.841 Y133.805 I-16.9 J.992 E.05093
G1 X104.841 Y141.805 E.30543
G1 X104.746 Y142.406 E.02324
G1 X104.725 Y142.443 E.00164
G1 X154.069 Y142.443 E1.88391
G3 X143.803 Y132.702 I31.808 J-43.8 E.54187
G3 X136.269 Y120.541 I41.758 J-34.286 E.54776
G1 X135.413 Y120.712 E.03334
G3 X133.274 Y120.651 I-.835 J-8.32 E.08191
G3 X130.472 Y119.447 I1.604 J-7.595 E.1172
G3 X129.012 Y118.015 I5.827 J-7.402 E.07827
G1 X126.663 Y114.849 E.15049
G3 X121.364 Y120.437 I-42.697 J-35.184 E.29428
G3 X119.312 Y122.16 I-426.217 J-505.442 E.1023
G1 X117.779 Y123.445 E.07636
G3 X103.942 Y131.189 I-33.694 J-43.972 E.60745
G1 X103.687 Y131.692 F36000
; LINE_WIDTH: 0.651996
G1 F12749.734
G1 X103.736 Y131.715 E.00219
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.105 Y132.163 E.02213
G1 X104.242 Y132.591 E.01715
G3 X104.256 Y133.805 I-22.834 J.872 E.04636
G1 X104.256 Y141.805 E.30543
G1 X104.189 Y142.225 E.01626
G1 X103.919 Y142.702 E.02094
G1 X103.447 Y143.029 E.02191
G1 X155.749 Y143.029 E1.99683
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I29.803 J-43.953 E.5995
G3 X136.599 Y119.835 I41.215 J-33.848 E.5612
G3 X135.317 Y120.134 I-2.045 J-5.858 E.05034
G1 X134.411 Y120.179 E.03463
G3 X133.287 Y120.054 I.625 J-10.798 E.04319
G1 X132.569 Y119.865 E.02836
G3 X130.819 Y118.975 I2.907 J-7.88 E.07513
G3 X129.065 Y117.104 I5.013 J-6.455 E.09833
G1 X126.682 Y113.891 E.15272
G3 X120.965 Y120.008 I-42.664 J-34.144 E.31996
G3 X118.935 Y121.712 I-588.602 J-699.21 E.10118
G1 X117.403 Y122.996 E.07636
G3 X105.781 Y129.813 I-33.038 J-43.015 E.51573
G1 X104.308 Y130.414 E.06075
G1 X103.937 Y130.565 E.01527
; LINE_WIDTH: 0.651996
G1 F12749.734
G1 X103.477 Y130.755 E.02007
; LINE_WIDTH: 0.683996
G1 F11775.97
G1 X103.016 Y130.945 E.02111
; LINE_WIDTH: 0.729168
G1 F10159.578
G1 X102.958 Y130.99 E.00332
; LINE_WIDTH: 0.774339
G1 F9931.826
G1 X102.899 Y131.035 E.00354
; LINE_WIDTH: 0.819511
G1 F9706.613
G1 X102.841 Y131.079 E.00375
; LINE_WIDTH: 0.864682
G1 F9484.002
G1 X102.783 Y131.124 E.00397
; LINE_WIDTH: 0.909853
G1 F8994.686
G1 X102.725 Y131.169 E.00419
; LINE_WIDTH: 0.955025
G1 F8553.381
G1 X102.667 Y131.213 E.0044
; LINE_WIDTH: 1.0002
G1 F8153.356
G1 X102.608 Y131.258 E.00462
G1 X102.674 Y131.278 E.00434
; LINE_WIDTH: 0.955025
G1 F8553.381
G1 X102.74 Y131.299 E.00414
; LINE_WIDTH: 0.909853
G1 F8994.686
G1 X102.806 Y131.319 E.00393
; LINE_WIDTH: 0.864682
G1 F9484.002
G1 X102.872 Y131.34 E.00373
; LINE_WIDTH: 0.819511
G1 F10029.621
G1 X102.937 Y131.36 E.00353
; LINE_WIDTH: 0.774339
G1 F10244.547
G1 X103.003 Y131.381 E.00332
; LINE_WIDTH: 0.729168
G1 F10461.761
G1 X103.069 Y131.401 E.00312
; LINE_WIDTH: 0.683996
G1 F11663.052
G1 X103.403 Y131.558 E.01562
; LINE_WIDTH: 0.651996
G1 F12425.626
G1 X103.606 Y131.654 E.00903
G1 X103.345 Y132.167 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X103.385 Y132.184 E.00164
G1 X103.59 Y132.443 E.01263
G1 X103.67 Y132.786 E.01345
G1 X103.67 Y141.405 E.32904
G1 X103.67 Y141.805 E.01527
G1 F13053.259
G1 X103.632 Y142.045 E.00928
G1 F12206.99
G1 X103.478 Y142.317 E.01195
; LINE_WIDTH: 0.6658
G1 F11159.117
G1 X103.381 Y142.411 E.00553
; LINE_WIDTH: 0.711603
G1 F10723.651
G1 X103.285 Y142.504 E.00593
; LINE_WIDTH: 0.757406
G1 F10296.852
G1 X103.189 Y142.598 E.00633
; LINE_WIDTH: 0.805071
G1 F9878.72
G1 X103.117 Y142.642 E.00421
; LINE_WIDTH: 0.852736
G1 F9622.435
G1 X103.046 Y142.686 E.00447
G1 X102.729 Y142.682 E.01692
; LINE_WIDTH: 0.847926
G1 F9679.323
G1 X102.468 Y142.62 E.01421
; LINE_WIDTH: 0.820996
G1 F10010.679
G1 X102.079 Y142.528 E.02051
; LINE_WIDTH: 0.780796
G1 F10549.796
G1 X101.69 Y142.435 E.01946
; LINE_WIDTH: 0.740596
G1 F11150.286
G1 X101.301 Y142.343 E.01842
; LINE_WIDTH: 0.700396
G1 F11823.26
G1 X100.912 Y142.25 E.01737
; LINE_WIDTH: 0.660196
G1 F12582.687
G1 X100.523 Y142.158 E.01632
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X100.129 Y142.085 E.01527
G1 X99.67 Y142 E.01783
G3 X99.617 Y143.307 I-8.423 J.312 E.05002
G1 X99.558 Y143.615 E.01194
G1 X100.389 Y143.615 E.03172
; LINE_WIDTH: 0.646916
G1 F12855.464
G1 X100.656 Y143.601 E.0107
; LINE_WIDTH: 0.687118
G1 F12063.753
G1 X101.056 Y143.581 E.01702
; LINE_WIDTH: 0.72732
G1 F11363.9
G1 X101.455 Y143.561 E.01807
; LINE_WIDTH: 0.767522
G1 F10740.795
G1 X101.855 Y143.541 E.01912
; LINE_WIDTH: 0.807724
G1 F10182.471
G1 X102.254 Y143.521 E.02017
; LINE_WIDTH: 0.847926
G1 F9679.323
G1 X102.654 Y143.501 E.02121
; LINE_WIDTH: 0.852736
G1 F9622.435
G3 X103.307 Y143.521 I.245 J2.566 E.03495
; LINE_WIDTH: 0.806188
G1 F10202.733
G1 X103.497 Y143.545 E.00963
; LINE_WIDTH: 0.75964
G1 F10810.669
G1 X103.687 Y143.568 E.00905
; LINE_WIDTH: 0.713092
G1 F11436.193
G1 X103.877 Y143.591 E.00847
; LINE_WIDTH: 0.666544
G1 F12079.312
G1 X104.067 Y143.615 E.00789
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.467 Y143.615 E.01527
G1 X156.235 Y143.615 E1.97647
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.706 Y131.955 I29.326 J-43.873 E.60771
G3 X136.908 Y119.075 I40.807 J-33.505 E.57678
G3 X135.221 Y119.556 I-2.329 J-4.97 E.06726
G3 X133.436 Y119.488 I-.626 J-6.978 E.06839
G1 X132.722 Y119.3 E.02821
G3 X131.166 Y118.503 I2.758 J-7.304 E.06688
G3 X129.952 Y117.317 I4.87 J-6.194 E.06491
G1 X126.698 Y112.93 E.20856
G3 X120.567 Y119.578 I-42.974 J-33.475 E.34569
G3 X118.559 Y121.263 I-995.709 J-1184.931 E.10005
G1 X117.026 Y122.548 E.07636
G3 X99.495 Y131.45 I-32.656 J-42.592 E.75494
G3 X99.664 Y132.536 I-10.551 J2.194 E.04196
G1 X99.664 Y132.592 E.00215
G1 X102.75 Y132.021 E.1198
G1 X102.99 Y132.014 E.00919
G1 X103.263 Y132.131 E.01132
M204 S250
G1 X103.034 Y132.612 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.115 Y132.754 E.00517
G3 X103.117 Y135.805 I-938.237 J2.245 E.09658
G1 X103.117 Y141.805 E.18996
G1 X103.061 Y141.953 E.00502
G1 X102.891 Y142.03 E.00591
G1 X102.85 Y142.026 E.00131
G1 X99.117 Y141.335 E.12021
G1 X99.117 Y142.626 E.04086
G3 X98.942 Y143.857 I-4.964 J-.08 E.03947
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.615 J-43.5 E.51059
G3 X137.192 Y118.316 I40.301 J-33.09 E.49165
G1 X136.731 Y118.548 E.01635
M73 P86 R2
G3 X135.131 Y119.011 I-2.148 J-4.429 E.05299
G1 X134.369 Y119.041 E.02414
G1 X133.609 Y118.959 E.02419
G1 X132.867 Y118.767 E.02428
G1 X132.342 Y118.544 E.01807
G1 X131.508 Y118.068 E.0304
G3 X130.278 Y116.829 I3.743 J-4.943 E.05545
G1 X126.704 Y112.01 E.18996
G3 X121.462 Y117.971 I-42.494 J-32.081 E.25153
G3 X119.328 Y119.897 I-20.292 J-20.337 E.09108
G1 X116.671 Y122.124 E.10974
G3 X98.936 Y131.038 I-32.254 J-42.072 E.63217
G1 X98.946 Y131.525 E.01542
G3 X99.111 Y132.539 I-11.261 J2.348 E.03256
G1 X99.115 Y133.256 E.02269
G1 X102.85 Y132.564 E.12026
G1 X102.947 Y132.589 E.00317
; WIPE_START
M204 S10000
G1 X103.115 Y132.754 E-.08924
G1 X103.115 Y133.519 E-.29076
; WIPE_END
G1 E-.02 F1800
G1 X101.001 Y140.853 Z10.52 F36000
G1 X100.444 Y142.784 Z10.52
G1 Z10.12
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.917936
G1 F8912.405
G1 X100.334 Y142.784 E.00636
G1 X100.278 Y142.88 E.00636
G1 X100.334 Y142.976 E.00636
G1 X100.444 Y142.976 E.00636
G1 X100.499 Y142.88 E.00636
; WIPE_START
G1 X100.444 Y142.976 E-.076
G1 X100.334 Y142.976 E-.076
G1 X100.278 Y142.88 E-.076
G1 X100.334 Y142.784 E-.076
G1 X100.444 Y142.784 E-.076
; WIPE_END
G1 E-.02 F1800
G1 X100.266 Y135.154 Z10.52 F36000
G1 X100.189 Y131.879 Z10.52
G1 Z10.12
G1 E.4 F1800
; LINE_WIDTH: 0.69949
G1 F11839.376
G1 X100.5 Y131.801 E.01389
; LINE_WIDTH: 0.738203
G1 F11188.2
G1 X100.811 Y131.724 E.0147
; LINE_WIDTH: 0.776916
G1 F10604.919
G1 X101.122 Y131.647 E.01551
; LINE_WIDTH: 0.823243
G1 F9982.17
G1 X101.445 Y131.564 E.01714
; LINE_WIDTH: 0.86957
G1 F9428.504
G1 X101.767 Y131.48 E.01815
; LINE_WIDTH: 0.915896
G1 F8933.028
G1 X102.09 Y131.397 E.01916
; LINE_WIDTH: 0.958046
G1 F8525.404
G1 X102.349 Y131.327 E.01616
; LINE_WIDTH: 1.0002
G1 F8153.356
G1 X102.608 Y131.258 E.0169
; WIPE_START
G1 X102.349 Y131.327 E-.10196
G1 X102.09 Y131.397 E-.10197
G1 X101.767 Y131.48 E-.12666
G1 X101.641 Y131.513 E-.04941
; WIPE_END
G1 E-.02 F1800
G1 X108.478 Y128.12 Z10.52 F36000
G1 X119.492 Y122.655 Z10.52
G1 Z10.12
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X117.683 Y124.143 I-14.56 J-15.858 E.08947
G2 X121.396 Y126.52 I4.984 J-3.696 E.17209
G2 X123.752 Y126.064 I.668 J-2.86 E.0944
G2 X125.166 Y124.39 I-5.282 J-5.897 E.08392
G3 X128.937 Y121.943 I5.059 J3.667 E.17554
G3 X131.293 Y122.4 I.668 J2.86 E.0944
G3 X132.707 Y124.073 I-5.283 J5.897 E.08392
G2 X136.477 Y126.52 I5.059 J-3.667 E.17554
G2 X138.639 Y126.186 I.667 J-2.844 E.08561
G2 X140.756 Y129.478 I49.966 J-29.803 E.14945
G2 X139.776 Y129.941 I.732 J2.821 E.04161
G2 X138.362 Y131.614 I5.283 J5.897 E.08392
G3 X134.592 Y134.061 I-5.059 J-3.667 E.17554
G3 X132.236 Y133.604 I-.668 J-2.86 E.0944
G3 X130.822 Y131.931 I5.283 J-5.897 E.08392
G2 X127.052 Y129.484 I-5.059 J3.667 E.17554
G2 X124.695 Y129.941 I-.668 J2.86 E.0944
G2 X123.281 Y131.614 I5.283 J5.897 E.08392
G3 X119.511 Y134.061 I-5.059 J-3.667 E.17554
G3 X117.154 Y133.604 I-.668 J-2.86 E.0944
G3 X115.741 Y131.931 I5.283 J-5.897 E.08392
G2 X111.97 Y129.484 I-5.059 J3.667 E.17554
G2 X109.614 Y129.941 I-.668 J2.86 E.0944
G2 X108.2 Y131.614 I5.283 J5.897 E.08392
G3 X105.372 Y133.811 I-5.237 J-3.821 E.13853
G1 X105.339 Y141.337 E.28731
G1 X106.315 Y141.601 E.0386
G2 X108.671 Y141.145 I.668 J-2.86 E.0944
G2 X110.085 Y139.471 I-5.282 J-5.897 E.08392
G3 X113.855 Y137.025 I5.059 J3.667 E.17554
G3 X116.212 Y137.481 I.668 J2.86 E.0944
G3 X117.626 Y139.155 I-5.282 J5.897 E.08392
G2 X121.396 Y141.601 I5.059 J-3.667 E.17554
G2 X123.752 Y141.145 I.668 J-2.86 E.0944
G2 X125.166 Y139.471 I-5.282 J-5.896 E.08392
G3 X128.937 Y137.025 I5.059 J3.667 E.17554
G3 X131.293 Y137.481 I.668 J2.86 E.0944
G3 X132.707 Y139.155 I-5.282 J5.897 E.08392
G2 X136.477 Y141.601 I5.059 J-3.667 E.17554
G2 X138.834 Y141.145 I.668 J-2.86 E.0944
G2 X140.248 Y139.471 I-5.282 J-5.897 E.08392
G3 X144.018 Y137.025 I5.059 J3.667 E.17554
G3 X146.374 Y137.481 I.668 J2.86 E.0944
G3 X147.788 Y139.155 I-5.282 J5.897 E.08392
G2 X152.197 Y141.645 I4.78 J-3.315 E.19999
G3 X150.378 Y140.169 I22.207 J-29.211 E.08945
; CHANGE_LAYER
; Z_HEIGHT: 10.28
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X151.154 Y140.799 E-.38
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
G1 X104.057 Y131.24
G1 Z10.28
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.128 Y131.274 E.00301
G1 X104.627 Y131.891 E.03031
G1 X104.784 Y132.305 E.01689
G1 X104.843 Y132.784 E.01842
G1 X104.843 Y141.807 E.3445
G1 X104.748 Y142.408 E.02324
G1 X104.729 Y142.443 E.00153
G1 X154.071 Y142.443 E1.88385
G3 X143.803 Y132.702 I31.434 J-43.414 E.54196
G3 X136.271 Y120.545 I42.247 J-34.589 E.54758
G1 X135.451 Y120.705 E.0319
G1 X134.435 Y120.764 E.03883
G3 X132.361 Y120.415 I.3 J-8.123 E.08056
G3 X130.472 Y119.447 I3.291 J-8.749 E.08118
G3 X129.012 Y118.015 I5.861 J-7.437 E.07827
G1 X126.663 Y114.849 E.15048
G3 X121.363 Y120.438 I-45.319 J-37.673 E.29426
G3 X119.312 Y122.16 I-447.339 J-530.591 E.10228
G1 X117.779 Y123.445 E.07636
G3 X103.946 Y131.188 I-33.374 J-43.398 E.60736
G1 X103.975 Y131.201 E.00124
G1 X103.769 Y131.75 F36000
G1 F13446.369
G1 X104.106 Y132.159 E.02026
G3 X104.258 Y132.784 I-1.213 J.624 E.02476
G1 X104.258 Y141.807 E.3445
G1 X104.191 Y142.228 E.01626
G1 X103.921 Y142.705 E.02094
G1 X103.452 Y143.029 E.02174
G1 X155.749 Y143.029 E1.99662
G1 X155.769 Y142.919 E.00427
G3 X144.254 Y132.328 I30.001 J-44.173 E.59945
G3 X136.596 Y119.828 I41.668 J-34.126 E.56144
G1 X136.064 Y119.995 E.02127
G1 X135.344 Y120.13 E.02798
G1 X134.414 Y120.179 E.03555
G1 X133.485 Y120.091 E.03562
G1 X132.578 Y119.868 E.03565
G3 X130.819 Y118.975 I2.886 J-7.868 E.0755
G3 X129.065 Y117.104 I5.013 J-6.455 E.09833
G1 X126.682 Y113.891 E.15272
G3 X120.965 Y120.008 I-42.659 J-34.14 E.31997
G3 X118.936 Y121.711 I-616.653 J-732.626 E.10116
G1 X117.403 Y122.996 E.07636
G3 X105.462 Y129.945 I-33.388 J-43.637 E.52885
G1 X104.308 Y130.414 E.04758
G1 X103.937 Y130.565 E.01527
; LINE_WIDTH: 0.652736
G1 F12734.477
G1 X103.477 Y130.756 E.0201
; LINE_WIDTH: 0.685476
G1 F11775.705
G1 X103.016 Y130.946 E.02116
; LINE_WIDTH: 0.730199
G1 F10159.231
G1 X102.958 Y130.99 E.00331
; LINE_WIDTH: 0.774922
G1 F9932.808
G1 X102.9 Y131.035 E.00352
; LINE_WIDTH: 0.819645
G1 F9708.913
G1 X102.842 Y131.079 E.00373
; LINE_WIDTH: 0.864368
G1 F9487.593
G1 X102.784 Y131.123 E.00395
; LINE_WIDTH: 0.909091
G1 F9002.529
G1 X102.727 Y131.168 E.00416
; LINE_WIDTH: 0.953813
G1 F8564.651
G1 X102.669 Y131.212 E.00437
; LINE_WIDTH: 0.998536
G1 F8167.393
G1 X102.611 Y131.256 E.00458
G1 X102.676 Y131.277 E.00431
; LINE_WIDTH: 0.953813
G1 F8564.651
G1 X102.742 Y131.297 E.00411
; LINE_WIDTH: 0.909091
G1 F9002.529
G1 X102.807 Y131.317 E.00391
; LINE_WIDTH: 0.864368
G1 F9487.593
G1 X102.873 Y131.337 E.00371
; LINE_WIDTH: 0.819645
G1 F10027.906
G1 X102.938 Y131.357 E.00351
; LINE_WIDTH: 0.774922
G1 F10241.674
G1 X103.004 Y131.378 E.00331
; LINE_WIDTH: 0.730199
G1 F10457.675
G1 X103.069 Y131.398 E.00311
; LINE_WIDTH: 0.685476
G1 F11660.289
G1 X103.403 Y131.555 E.01567
; LINE_WIDTH: 0.652736
G1 F12734.477
G1 X103.701 Y131.695 E.01325
G1 X103.358 Y132.169 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X103.386 Y132.181 E.00118
G1 X103.593 Y132.442 E.01272
G1 X103.672 Y132.784 E.01338
G1 X103.672 Y141.407 E.32923
G1 X103.672 Y141.807 E.01527
G1 F13069.803
G1 X103.634 Y142.047 E.00928
G1 F12222.989
G1 X103.48 Y142.319 E.01195
; LINE_WIDTH: 0.665583
G1 F11174.398
G1 X103.383 Y142.413 E.00553
; LINE_WIDTH: 0.71117
G1 F10738.786
G1 X103.287 Y142.507 E.00593
; LINE_WIDTH: 0.756756
G1 F10311.834
G1 X103.191 Y142.6 E.00633
; LINE_WIDTH: 0.804116
G1 F9893.521
G1 X103.119 Y142.644 E.0042
; LINE_WIDTH: 0.851476
G1 F9637.272
G1 X103.048 Y142.688 E.00446
G1 X102.731 Y142.684 E.01688
; LINE_WIDTH: 0.846696
G1 F9693.978
G1 X102.469 Y142.622 E.01427
; LINE_WIDTH: 0.819776
G1 F10026.228
G1 X101.982 Y142.506 E.0256
; LINE_WIDTH: 0.769831
G1 F10707.076
G1 X101.496 Y142.391 E.02397
; LINE_WIDTH: 0.719886
G1 F11487.128
G1 X101.01 Y142.275 E.02235
; LINE_WIDTH: 0.669941
G1 F12389.772
G1 X100.523 Y142.16 E.02072
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X100.13 Y142.087 E.01527
G1 X99.672 Y142.002 E.01777
G3 X99.618 Y143.307 I-8.307 J.311 E.0499
G1 X99.559 Y143.615 E.01197
G1 X100.389 Y143.615 E.03171
; LINE_WIDTH: 0.646906
G1 F12814.239
G1 X100.659 Y143.601 E.01076
; LINE_WIDTH: 0.696854
G1 F11886.479
G1 X101.158 Y143.576 E.02159
; LINE_WIDTH: 0.746801
G1 F11053.176
G1 X101.657 Y143.551 E.02322
; LINE_WIDTH: 0.796749
G1 F10329.055
G1 X102.157 Y143.526 E.02485
; LINE_WIDTH: 0.846696
G1 F9693.978
G1 X102.656 Y143.501 E.02648
; LINE_WIDTH: 0.851476
G1 F9637.272
G3 X103.308 Y143.522 I.245 J2.576 E.03486
; LINE_WIDTH: 0.80518
G1 F10216.075
G1 X103.498 Y143.545 E.00961
; LINE_WIDTH: 0.758884
G1 F10823.772
G1 X103.688 Y143.568 E.00903
; LINE_WIDTH: 0.712588
G1 F11449.029
G1 X103.878 Y143.591 E.00846
; LINE_WIDTH: 0.666292
G1 F12091.808
G1 X104.068 Y143.615 E.00788
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.468 Y143.615 E.01527
G1 X156.235 Y143.615 E1.97644
G1 X156.415 Y142.65 E.03745
G3 X144.706 Y131.955 I29.42 J-43.965 E.60776
G3 X136.912 Y119.083 I40.835 J-33.522 E.57645
G1 X136.091 Y119.382 E.03333
G3 X134.392 Y119.594 I-1.767 J-7.27 E.06553
G1 X133.549 Y119.509 E.03234
G1 X132.727 Y119.302 E.03236
G3 X131.166 Y118.503 I2.751 J-7.307 E.0671
G3 X129.952 Y117.317 I4.904 J-6.229 E.06491
G1 X126.698 Y112.93 E.20856
G3 X120.567 Y119.579 I-42.974 J-33.475 E.3457
G3 X118.559 Y121.263 I-1045.807 J-1244.635 E.10003
G1 X117.027 Y122.547 E.07636
G3 X99.496 Y131.45 I-32.601 J-42.483 E.75496
G3 X99.666 Y132.536 I-10.451 J2.195 E.04197
G1 X99.666 Y132.59 E.00206
G1 X102.752 Y132.018 E.11981
G1 X102.991 Y132.011 E.00915
G1 X103.275 Y132.133 E.0118
M204 S250
G1 X103.036 Y132.609 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.119 Y132.77 E.00573
G3 X103.119 Y133.807 I-887.279 J.869 E.03283
G1 X103.119 Y141.807 E.25328
G1 X103.063 Y141.956 E.00502
G1 X102.894 Y142.033 E.00591
G1 X102.852 Y142.029 E.00131
G1 X99.119 Y141.338 E.12021
G1 X99.119 Y142.626 E.04079
G3 X98.942 Y143.857 I-4.927 J-.082 E.03948
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.959 Y142.746 E.04578
G1 X157.025 Y142.391 E.01143
G3 X145.128 Y131.598 I28.63 J-43.513 E.51061
G3 X137.192 Y118.315 I40.814 J-33.397 E.49165
G1 X136.612 Y118.616 E.02068
G1 X135.912 Y118.859 E.02348
G1 X135.135 Y119.01 E.02504
G1 X134.371 Y119.041 E.0242
G1 X133.61 Y118.96 E.02425
G1 X132.868 Y118.767 E.02427
G1 X132.313 Y118.533 E.01906
G1 X131.509 Y118.068 E.0294
G3 X130.278 Y116.829 I3.725 J-4.927 E.05548
G1 X126.704 Y112.01 E.18996
G3 X121.459 Y117.974 I-42.957 J-32.489 E.25167
G3 X119.279 Y119.938 I-20.515 J-20.577 E.09297
G1 X116.671 Y122.124 E.10771
G3 X98.936 Y131.038 I-32.501 J-42.563 E.63209
G1 X98.947 Y131.524 E.0154
G3 X99.113 Y132.539 I-11.174 J2.355 E.03258
G1 X99.118 Y133.254 E.02262
G1 X102.852 Y132.562 E.12026
G1 X102.949 Y132.587 E.00316
; WIPE_START
M204 S10000
G1 X103.119 Y132.77 E-.09494
G1 X103.119 Y133.52 E-.28506
; WIPE_END
G1 E-.02 F1800
G1 X101.002 Y140.853 Z10.68 F36000
G1 X100.445 Y142.786 Z10.68
G1 Z10.28
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.915696
G1 F8935.056
G1 X100.334 Y142.786 E.00633
G1 X100.279 Y142.881 E.00633
G1 X100.334 Y142.976 E.00633
G1 X100.445 Y142.976 E.00633
G1 X100.5 Y142.881 E.00633
; WIPE_START
G1 X100.445 Y142.976 E-.076
G1 X100.334 Y142.976 E-.076
G1 X100.279 Y142.881 E-.076
G1 X100.334 Y142.786 E-.076
G1 X100.445 Y142.786 E-.07599
; WIPE_END
G1 E-.02 F1800
G1 X100.266 Y135.155 Z10.68 F36000
G1 X100.189 Y131.878 Z10.68
G1 Z10.28
G1 E.4 F1800
; LINE_WIDTH: 0.697416
G1 F11876.396
G1 X100.5 Y131.8 E.01384
; LINE_WIDTH: 0.736096
G1 F11221.786
G1 X100.811 Y131.723 E.01464
; LINE_WIDTH: 0.774776
G1 F10635.569
G1 X101.121 Y131.646 E.01545
; LINE_WIDTH: 0.821116
G1 F10009.152
G1 X101.444 Y131.563 E.0171
; LINE_WIDTH: 0.867456
G1 F9452.421
G1 X101.767 Y131.479 E.01811
; LINE_WIDTH: 0.913796
G1 F8954.359
G1 X102.09 Y131.396 E.01912
; LINE_WIDTH: 0.956166
G1 F8542.79
G1 X102.35 Y131.326 E.0162
; LINE_WIDTH: 0.998536
G1 F8167.393
G1 X102.611 Y131.256 E.01695
; WIPE_START
G1 X102.35 Y131.326 E-.10247
G1 X102.09 Y131.396 E-.10247
G1 X101.767 Y131.479 E-.1267
G1 X101.644 Y131.511 E-.04836
; WIPE_END
G1 E-.02 F1800
G1 X108.485 Y128.127 Z10.68 F36000
G1 X119.417 Y122.718 Z10.68
G1 Z10.28
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X117.603 Y124.199 I-13.187 J-14.301 E.08947
G1 X117.626 Y124.226 E.00136
G2 X120.453 Y126.231 I4.977 J-4.024 E.13401
G2 X123.281 Y126.15 I1.305 J-3.811 E.11036
G2 X125.166 Y124.237 I-4.038 J-5.865 E.10315
G3 X127.994 Y122.233 I4.977 J4.024 E.13401
G3 X130.822 Y122.314 I1.305 J3.811 E.11036
G3 X132.707 Y124.226 I-4.038 J5.866 E.10315
G2 X136.477 Y126.432 I4.576 J-3.497 E.17105
G2 X138.542 Y126.028 I.493 J-2.959 E.08208
G2 X140.862 Y129.632 I52.451 J-31.204 E.16367
G2 X140.248 Y129.854 I.278 J1.729 E.02508
G2 X138.362 Y131.767 I4.038 J5.866 E.10315
G3 X135.535 Y133.772 I-4.977 J-4.024 E.13401
G3 X132.707 Y133.69 I-1.305 J-3.811 E.11036
G3 X130.822 Y131.778 I4.038 J-5.865 E.10315
G2 X127.994 Y129.773 I-4.977 J4.024 E.13401
G2 X125.166 Y129.854 I-1.305 J3.811 E.11036
G2 X123.281 Y131.767 I4.038 J5.866 E.10315
G3 X120.453 Y133.772 I-4.977 J-4.024 E.13401
G3 X117.626 Y133.69 I-1.305 J-3.811 E.11036
G3 X115.741 Y131.778 I4.038 J-5.865 E.10315
G2 X112.913 Y129.773 I-4.977 J4.024 E.13401
G2 X110.085 Y129.854 I-1.305 J3.811 E.11036
G2 X108.2 Y131.767 I4.038 J5.866 E.10315
G3 X105.372 Y133.772 I-4.977 J-4.024 E.13401
G1 X105.372 Y141.312 E.2879
G2 X108.2 Y141.231 I1.305 J-3.811 E.11036
G2 X110.085 Y139.319 I-4.038 J-5.865 E.10315
G3 X112.913 Y137.314 I4.977 J4.024 E.13401
G3 X115.741 Y137.395 I1.305 J3.811 E.11036
G3 X117.626 Y139.308 I-4.038 J5.865 E.10315
G2 X120.453 Y141.312 I4.977 J-4.024 E.13401
G2 X123.281 Y141.231 I1.305 J-3.811 E.11036
G2 X125.166 Y139.319 I-4.038 J-5.865 E.10315
G3 X127.994 Y137.314 I4.977 J4.024 E.13401
G3 X130.822 Y137.395 I1.305 J3.811 E.11036
G3 X132.707 Y139.308 I-4.038 J5.865 E.10315
G2 X135.535 Y141.312 I4.977 J-4.024 E.13401
G2 X138.362 Y141.231 I1.305 J-3.811 E.11036
G2 X140.248 Y139.319 I-4.038 J-5.865 E.10315
G3 X144.018 Y137.113 I4.576 J3.497 E.17105
G3 X146.846 Y138.171 I.501 J2.969 E.12075
G2 X148.731 Y140.279 I14.441 J-11.017 E.10808
G2 X152.028 Y141.517 I3.34 J-3.888 E.13724
G3 X150.217 Y140.03 I32.376 J-41.264 E.08943
; CHANGE_LAYER
; Z_HEIGHT: 10.44
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13446.283
G1 X150.99 Y140.665 E-.38
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
G1 X104.058 Y131.237
G1 Z10.44
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.128 Y131.271 E.00298
G1 X104.629 Y131.888 E.03034
G1 X104.786 Y132.302 E.0169
G1 X104.845 Y132.781 E.01844
G1 X104.845 Y141.81 E.34469
G1 X104.75 Y142.411 E.02324
G1 X104.732 Y142.443 E.00142
G1 X154.069 Y142.443 E1.88364
G3 X143.803 Y132.702 I31.809 J-43.801 E.54187
G3 X136.269 Y120.54 I42.212 J-34.566 E.54777
G3 X135.143 Y120.739 I-1.684 J-6.258 E.04371
G3 X133.275 Y120.651 I-.539 J-8.514 E.07155
G3 X130.472 Y119.447 I1.607 J-7.602 E.11721
G3 X129.012 Y118.015 I6.06 J-7.641 E.07826
G1 X126.663 Y114.849 E.15048
M73 P87 R2
G3 X121.363 Y120.437 I-45.317 J-37.671 E.29426
G3 X119.312 Y122.16 I-426.681 J-505.99 E.10228
G1 X117.779 Y123.445 E.07636
G3 X103.951 Y131.186 I-33.269 J-43.21 E.60719
G1 X103.977 Y131.198 E.0011
G1 X103.77 Y131.747 F36000
G1 F13446.369
G1 X104.108 Y132.156 E.02026
G3 X104.26 Y132.781 I-1.213 J.625 E.02479
G1 X104.26 Y141.81 E.34469
G1 X104.193 Y142.23 E.01626
G1 X103.923 Y142.707 E.02094
G1 X103.458 Y143.029 E.02157
G1 X155.749 Y143.029 E1.9964
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I29.803 J-43.952 E.5995
G3 X136.598 Y119.835 I41.249 J-33.869 E.5612
G1 X135.96 Y120.02 E.0254
G1 X135.063 Y120.159 E.03465
G3 X133.287 Y120.054 I-.426 J-7.915 E.06804
G1 X132.578 Y119.868 E.02802
G3 X130.819 Y118.975 I2.864 J-7.823 E.07547
G3 X129.065 Y117.104 I5.012 J-6.455 E.09833
G1 X126.682 Y113.891 E.15272
G3 X120.965 Y120.008 I-42.659 J-34.14 E.31996
G3 X118.936 Y121.711 I-586.235 J-696.386 E.10116
G1 X117.403 Y122.996 E.07636
G3 X105.463 Y129.947 I-33.116 J-43.153 E.52889
G1 X104.308 Y130.414 E.04758
G1 X103.937 Y130.565 E.01527
; LINE_WIDTH: 0.653456
G1 F12719.667
G1 X103.477 Y130.756 E.02012
; LINE_WIDTH: 0.686916
G1 F12067.488
G1 X103.016 Y130.947 E.02121
; LINE_WIDTH: 0.731191
G1 F10481.521
G1 X102.959 Y130.991 E.00329
; LINE_WIDTH: 0.775465
G1 F10252.86
G1 X102.901 Y131.035 E.0035
; LINE_WIDTH: 0.819739
G1 F10026.701
G1 X102.843 Y131.079 E.00371
; LINE_WIDTH: 0.864013
G1 F9491.644
G1 X102.786 Y131.123 E.00392
; LINE_WIDTH: 0.908288
G1 F9010.799
G1 X102.728 Y131.167 E.00413
; LINE_WIDTH: 0.952562
G1 F8576.323
G1 X102.671 Y131.211 E.00434
; LINE_WIDTH: 0.996836
G1 F8181.819
G1 X102.613 Y131.255 E.00455
G1 X102.678 Y131.275 E.00428
; LINE_WIDTH: 0.952562
G1 F8576.323
G1 X102.743 Y131.295 E.00408
; LINE_WIDTH: 0.908288
G1 F9010.799
G1 X102.808 Y131.315 E.00388
; LINE_WIDTH: 0.864013
G1 F9491.644
G1 X102.874 Y131.335 E.00369
; LINE_WIDTH: 0.819739
G1 F10026.701
G1 X102.939 Y131.354 E.00349
; LINE_WIDTH: 0.775465
G1 F10625.687
G1 X103.004 Y131.374 E.00329
; LINE_WIDTH: 0.731191
G1 F10844.508
G1 X103.069 Y131.394 E.0031
; LINE_WIDTH: 0.686916
G1 F12067.488
G1 X103.404 Y131.552 E.01573
; LINE_WIDTH: 0.653456
G1 F12719.667
G1 X103.702 Y131.692 E.0133
G1 X103.36 Y132.166 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X103.388 Y132.178 E.00117
G1 X103.595 Y132.439 E.01274
G1 X103.674 Y132.781 E.01339
G1 X103.674 Y141.41 E.32942
G1 X103.674 Y141.81 E.01527
G1 F13086.459
G1 X103.636 Y142.05 E.00928
G1 F12239.097
G1 X103.482 Y142.322 E.01195
; LINE_WIDTH: 0.665363
G1 F11189.799
G1 X103.385 Y142.415 E.00552
; LINE_WIDTH: 0.71073
G1 F10754.038
G1 X103.289 Y142.509 E.00592
; LINE_WIDTH: 0.756096
G1 F10326.909
G1 X103.193 Y142.602 E.00632
; LINE_WIDTH: 0.803151
G1 F9908.435
G1 X103.121 Y142.646 E.00419
; LINE_WIDTH: 0.850206
G1 F9652.273
G1 X103.05 Y142.69 E.00445
G1 X102.733 Y142.686 E.01684
; LINE_WIDTH: 0.845456
G1 F9708.798
G1 X102.47 Y142.623 E.01433
; LINE_WIDTH: 0.818546
G1 F10041.954
G1 X101.983 Y142.508 E.02556
; LINE_WIDTH: 0.768909
G1 F10720.522
G1 X101.497 Y142.393 E.02394
; LINE_WIDTH: 0.719271
G1 F11497.442
G1 X101.01 Y142.277 E.02233
; LINE_WIDTH: 0.669634
G1 F12395.768
G1 X100.524 Y142.162 E.02071
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X100.13 Y142.089 E.01527
G1 X99.674 Y142.005 E.01771
G3 X99.62 Y143.306 I-8.193 J.31 E.04978
G1 X99.56 Y143.615 E.012
G1 X100.39 Y143.615 E.03171
; LINE_WIDTH: 0.646896
G1 F12825.589
G1 X100.661 Y143.601 E.01082
; LINE_WIDTH: 0.696536
G1 F11892.179
G1 X101.16 Y143.576 E.02158
; LINE_WIDTH: 0.746176
G1 F11062.88
G1 X101.66 Y143.551 E.0232
; LINE_WIDTH: 0.795816
G1 F10341.704
G1 X102.159 Y143.527 E.02482
; LINE_WIDTH: 0.845456
G1 F9708.798
G1 X102.658 Y143.502 E.02644
; LINE_WIDTH: 0.850206
G1 F9652.273
G3 X103.31 Y143.522 I.244 J2.585 E.03477
; LINE_WIDTH: 0.804164
G1 F10229.559
G1 X103.5 Y143.545 E.00959
; LINE_WIDTH: 0.758122
G1 F10836.995
G1 X103.689 Y143.569 E.00901
; LINE_WIDTH: 0.71208
G1 F11461.951
G1 X103.879 Y143.592 E.00844
; LINE_WIDTH: 0.666038
G1 F12104.425
G1 X104.069 Y143.615 E.00787
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.469 Y143.615 E.01527
G1 X156.235 Y143.615 E1.9764
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.706 Y131.955 I29.326 J-43.873 E.60771
G3 X136.908 Y119.075 I40.837 J-33.523 E.57678
G3 X132.727 Y119.301 I-2.382 J-5.272 E.1636
G3 X131.166 Y118.503 I2.728 J-7.261 E.06708
G3 X129.952 Y117.317 I5.112 J-6.442 E.0649
G1 X126.698 Y112.93 E.20856
G3 X120.567 Y119.579 I-42.976 J-33.477 E.3457
G3 X118.559 Y121.263 I-986.29 J-1173.693 E.10004
G1 X117.027 Y122.547 E.07636
G3 X99.496 Y131.45 I-32.587 J-42.456 E.75495
G3 X99.668 Y132.536 I-10.331 J2.193 E.04198
G1 X99.668 Y132.587 E.00197
G1 X102.754 Y132.016 E.11981
G1 X102.992 Y132.009 E.00911
G1 X103.277 Y132.131 E.01182
M204 S250
G1 X103.038 Y132.606 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.121 Y132.768 E.00573
G3 X103.121 Y133.81 I-890.643 J.871 E.03299
G1 X103.121 Y141.81 E.25328
G1 X103.066 Y141.958 E.00502
G1 X102.896 Y142.035 E.00591
G1 X102.855 Y142.031 E.00131
G1 X99.121 Y141.34 E.12021
G1 X99.121 Y142.626 E.04071
G3 X98.942 Y143.857 I-4.891 J-.085 E.03949
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.616 J-43.501 E.51059
G3 X137.192 Y118.316 I40.815 J-33.397 E.49161
G3 X136.346 Y118.722 I-5.225 J-9.801 E.02972
G1 X135.642 Y118.927 E.02322
G1 X134.907 Y119.031 E.0235
G1 X134.403 Y119.041 E.01595
G1 X133.61 Y118.959 E.02526
G1 X132.867 Y118.767 E.02428
G1 X132.339 Y118.543 E.01816
G1 X131.508 Y118.068 E.03031
G3 X130.278 Y116.829 I3.699 J-4.9 E.05546
G1 X126.704 Y112.01 E.18996
G3 X121.432 Y118 I-42.964 J-32.494 E.25286
G3 X119.325 Y119.9 I-20.193 J-20.285 E.08987
G1 X116.671 Y122.124 E.10961
G3 X98.936 Y131.038 I-32.425 J-42.413 E.63212
G1 X98.947 Y131.523 E.01538
G3 X99.115 Y132.539 I-11.063 J2.357 E.03261
G1 X99.12 Y133.251 E.02254
G1 X102.855 Y132.559 E.12026
G1 X102.951 Y132.584 E.00316
; WIPE_START
M204 S10000
G1 X103.121 Y132.768 E-.09499
G1 X103.121 Y133.518 E-.28501
; WIPE_END
G1 E-.02 F1800
G1 X101.004 Y140.851 Z10.84 F36000
G1 X100.445 Y142.787 Z10.84
G1 Z10.44
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.913476
G1 F8957.618
G1 X100.335 Y142.787 E.0063
G1 X100.28 Y142.882 E.0063
G1 X100.335 Y142.977 E.0063
G1 X100.445 Y142.977 E.0063
G1 X100.5 Y142.882 E.0063
; WIPE_START
G1 X100.445 Y142.977 E-.076
G1 X100.335 Y142.977 E-.076
G1 X100.28 Y142.882 E-.076
G1 X100.335 Y142.787 E-.076
G1 X100.445 Y142.787 E-.07599
; WIPE_END
G1 E-.02 F1800
G1 X100.266 Y135.157 Z10.84 F36000
G1 X100.189 Y131.876 Z10.84
G1 Z10.44
G1 E.4 F1800
; LINE_WIDTH: 0.695376
G1 F11913.047
G1 X100.5 Y131.799 E.01379
; LINE_WIDTH: 0.734056
G1 F11254.503
G1 X100.811 Y131.722 E.0146
; LINE_WIDTH: 0.772736
G1 F10664.952
G1 X101.121 Y131.645 E.01541
; LINE_WIDTH: 0.819063
G1 F10035.343
G1 X101.444 Y131.562 E.01705
; LINE_WIDTH: 0.86539
G1 F9475.926
G1 X101.767 Y131.478 E.01806
; LINE_WIDTH: 0.911716
G1 F8975.587
G1 X102.089 Y131.395 E.01906
; LINE_WIDTH: 0.954276
G1 F8560.341
G1 X102.351 Y131.325 E.01625
; LINE_WIDTH: 0.996836
G1 F8181.819
G1 X102.613 Y131.255 E.017
; WIPE_START
G1 X102.351 Y131.325 E-.10298
G1 X102.089 Y131.395 E-.10298
G1 X101.767 Y131.478 E-.12666
G1 X101.646 Y131.509 E-.04739
; WIPE_END
G1 E-.02 F1800
G1 X108.491 Y128.133 Z10.84 F36000
G1 X119.343 Y122.78 Z10.84
G1 Z10.44
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X117.525 Y124.257 I-13.135 J-14.313 E.08946
G2 X120.453 Y126.198 I4.881 J-4.183 E.13586
G2 X123.281 Y125.958 I1.102 J-3.796 E.11082
G2 X125.166 Y124.093 I-4.784 J-6.72 E.10169
G3 X127.994 Y122.266 I4.629 J4.064 E.13019
G3 X130.822 Y122.506 I1.102 J3.796 E.11082
G3 X132.707 Y124.371 I-4.784 J6.72 E.10169
G2 X135.535 Y126.198 I4.629 J-4.064 E.13019
G2 X138.459 Y125.889 I1.087 J-3.705 E.11516
G2 X140.957 Y129.762 I48.476 J-28.53 E.17602
G2 X140.248 Y130.046 I.438 J2.123 E.02933
G2 X138.362 Y131.911 I4.783 J6.72 E.10169
G3 X135.535 Y133.738 I-4.629 J-4.064 E.13019
G3 X132.707 Y133.499 I-1.102 J-3.796 E.11082
G3 X130.822 Y131.633 I4.783 J-6.72 E.10169
G2 X127.994 Y129.807 I-4.629 J4.063 E.13019
G2 X125.166 Y130.046 I-1.102 J3.796 E.11082
G2 X123.281 Y131.911 I4.784 J6.72 E.10169
G3 X120.453 Y133.738 I-4.629 J-4.064 E.13019
G3 X117.626 Y133.499 I-1.102 J-3.796 E.11082
G3 X115.741 Y131.633 I4.784 J-6.72 E.10169
G2 X112.913 Y129.807 I-4.629 J4.063 E.13019
G2 X110.085 Y130.046 I-1.102 J3.796 E.11082
G2 X108.2 Y131.911 I4.783 J6.72 E.10169
G3 X105.372 Y133.738 I-4.629 J-4.064 E.13019
G1 X105.372 Y141.279 E.2879
G2 X108.2 Y141.039 I1.102 J-3.796 E.11082
G2 X110.085 Y139.174 I-4.783 J-6.72 E.10169
G3 X112.913 Y137.347 I4.629 J4.064 E.13019
G3 X115.741 Y137.587 I1.102 J3.796 E.11082
G3 X117.626 Y139.452 I-4.783 J6.719 E.10169
G2 X120.453 Y141.279 I4.629 J-4.064 E.13019
G2 X123.281 Y141.039 I1.102 J-3.796 E.11082
G2 X125.166 Y139.174 I-4.784 J-6.72 E.10169
G3 X127.994 Y137.347 I4.629 J4.064 E.13019
G3 X130.822 Y137.587 I1.102 J3.796 E.11082
G3 X132.707 Y139.452 I-4.783 J6.72 E.10169
G2 X135.535 Y141.279 I4.629 J-4.064 E.13019
G2 X138.362 Y141.039 I1.102 J-3.796 E.11082
G2 X140.248 Y139.174 I-4.784 J-6.72 E.10169
G3 X143.075 Y137.347 I4.629 J4.064 E.13019
G3 X145.903 Y137.587 I1.102 J3.796 E.11082
G3 X147.788 Y139.452 I-4.783 J6.72 E.10169
G2 X151.906 Y141.415 I4.16 J-3.426 E.17984
G1 X152.579 Y141.944 E.03267
G1 X151.092 Y141.944 E.05677
; CHANGE_LAYER
; Z_HEIGHT: 10.6
; LAYER_HEIGHT: 0.160001
; WIPE_START
G1 F13446.283
G1 X152.092 Y141.944 E-.38
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
G1 X104.06 Y131.234
G1 Z10.6
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.129 Y131.267 E.00296
G1 X104.64 Y131.903 E.03112
G1 X104.79 Y132.307 E.01646
G1 X104.848 Y132.779 E.01815
G1 X104.848 Y141.812 E.34488
G1 X104.736 Y142.443 E.02447
G1 X154.069 Y142.443 E1.8835
G3 X143.803 Y132.702 I31.793 J-43.784 E.54187
G3 X136.271 Y120.545 I41.671 J-34.231 E.54762
G3 X135.166 Y120.737 I-2.492 J-11.062 E.04283
G3 X133.275 Y120.652 I-.561 J-8.558 E.07244
G3 X130.592 Y119.529 I1.613 J-7.619 E.11169
G3 X129.741 Y118.835 I8.971 J-11.878 E.04192
G3 X129.012 Y118.015 I12.451 J-11.804 E.04191
G1 X126.663 Y114.849 E.15048
G3 X121.362 Y120.439 I-42.788 J-35.273 E.29437
G3 X119.312 Y122.16 I-446.063 J-529.118 E.1022
G1 X117.779 Y123.445 E.07636
G3 X103.956 Y131.184 I-33.705 J-43.989 E.60692
G1 X103.978 Y131.195 E.00096
G1 X103.772 Y131.745 F36000
G1 F13446.369
G1 X104.116 Y132.166 E.02076
G3 X104.262 Y132.779 I-1.219 J.613 E.02427
G1 X104.262 Y141.812 E.34488
G1 X104.195 Y142.233 E.01626
G1 X103.925 Y142.71 E.02094
G1 X103.464 Y143.029 E.0214
G1 X155.749 Y143.029 E1.99618
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.046 J-44.216 E.59947
G3 X136.596 Y119.828 I41.138 J-33.801 E.56148
G1 X136.064 Y119.995 E.02129
G1 X135.344 Y120.13 E.02797
G1 X134.414 Y120.179 E.03555
G3 X133.287 Y120.054 I.624 J-10.818 E.04328
G1 X132.578 Y119.868 E.02802
G3 X130.827 Y118.981 I2.882 J-7.862 E.0751
G1 X130.185 Y118.446 E.03189
G3 X129.065 Y117.104 I7.783 J-7.633 E.06682
G1 X126.682 Y113.891 E.15272
G3 X120.964 Y120.009 I-44.292 J-35.666 E.31999
G3 X118.936 Y121.711 I-616.559 J-732.56 E.10111
G1 X117.403 Y122.996 E.07636
G3 X105.477 Y129.942 I-33.041 J-43.021 E.52835
G1 X104.308 Y130.414 E.04813
G1 X103.938 Y130.565 E.01527
; LINE_WIDTH: 0.654171
G1 F12704.995
G1 X103.477 Y130.756 E.02015
; LINE_WIDTH: 0.688346
G1 F12041.101
G1 X103.017 Y130.948 E.02126
; LINE_WIDTH: 0.732179
G1 F10477.415
G1 X102.959 Y130.991 E.00328
; LINE_WIDTH: 0.776012
G1 F10250.15
G1 X102.902 Y131.035 E.00348
; LINE_WIDTH: 0.819845
G1 F10025.352
G1 X102.845 Y131.078 E.00369
; LINE_WIDTH: 0.863678
G1 F9495.487
G1 X102.787 Y131.122 E.0039
; LINE_WIDTH: 0.907511
G1 F9018.819
G1 X102.73 Y131.166 E.0041
; LINE_WIDTH: 0.951343
G1 F8587.72
G1 X102.673 Y131.209 E.00431
; LINE_WIDTH: 0.995176
G1 F8195.954
G1 X102.615 Y131.253 E.00451
G1 X102.68 Y131.273 E.00425
; LINE_WIDTH: 0.951343
G1 F8587.72
G1 X102.745 Y131.292 E.00405
; LINE_WIDTH: 0.907511
G1 F9018.819
G1 X102.81 Y131.312 E.00386
; LINE_WIDTH: 0.863678
G1 F9495.487
G1 X102.875 Y131.332 E.00367
; LINE_WIDTH: 0.819845
G1 F10025.352
G1 X102.94 Y131.352 E.00347
; LINE_WIDTH: 0.776012
G1 F10617.849
G1 X103.004 Y131.371 E.00328
; LINE_WIDTH: 0.732179
G1 F10835.388
G1 X103.069 Y131.391 E.00308
; LINE_WIDTH: 0.688346
G1 F12041.101
G1 X103.404 Y131.548 E.01578
; LINE_WIDTH: 0.654171
G1 F12704.995
G1 X103.705 Y131.689 E.0134
G1 X103.361 Y132.163 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X103.39 Y132.175 E.00117
G1 X103.601 Y132.444 E.01304
G1 X103.676 Y132.779 E.01311
G1 X103.676 Y141.412 E.32961
G1 X103.676 Y141.812 E.01527
G1 F13103.086
G1 X103.638 Y142.052 E.00928
G1 F12255.176
G1 X103.484 Y142.324 E.01195
; LINE_WIDTH: 0.665146
G1 F11205.175
G1 X103.387 Y142.418 E.00552
; LINE_WIDTH: 0.710296
G1 F10769.243
G1 X103.291 Y142.511 E.00592
; LINE_WIDTH: 0.755446
G1 F10341.959
G1 X103.195 Y142.605 E.00631
; LINE_WIDTH: 0.802196
G1 F9923.304
G1 X103.123 Y142.648 E.00418
; LINE_WIDTH: 0.848946
G1 F9667.203
G1 X103.052 Y142.692 E.00444
G1 X102.736 Y142.688 E.01681
; LINE_WIDTH: 0.844226
G1 F9723.543
G1 X102.47 Y142.625 E.01439
; LINE_WIDTH: 0.817326
G1 F10057.601
G1 X101.984 Y142.51 E.02552
; LINE_WIDTH: 0.767994
G1 F10733.892
G1 X101.497 Y142.395 E.02391
; LINE_WIDTH: 0.718661
G1 F11507.691
G1 X101.011 Y142.279 E.02231
; LINE_WIDTH: 0.669329
G1 F12401.722
G1 X100.524 Y142.164 E.0207
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X100.131 Y142.091 E.01527
G1 X99.676 Y142.007 E.01765
G3 X99.621 Y143.306 I-8.083 J.309 E.04967
G1 X99.561 Y143.615 E.01202
G1 X100.391 Y143.615 E.0317
; LINE_WIDTH: 0.646886
G1 F12836.933
G1 X100.663 Y143.601 E.01089
; LINE_WIDTH: 0.696221
G1 F11897.838
G1 X101.163 Y143.576 E.02157
; LINE_WIDTH: 0.745556
G1 F11072.524
G1 X101.662 Y143.552 E.02318
; LINE_WIDTH: 0.794891
G1 F10354.282
G1 X102.161 Y143.527 E.02479
; LINE_WIDTH: 0.844226
G1 F9723.543
G1 X102.661 Y143.502 E.0264
; LINE_WIDTH: 0.848946
G1 F9667.203
G3 X103.312 Y143.523 I.244 J2.594 E.03468
; LINE_WIDTH: 0.803156
G1 F10242.97
G1 X103.501 Y143.546 E.00957
; LINE_WIDTH: 0.757366
G1 F10850.176
G1 X103.691 Y143.569 E.009
; LINE_WIDTH: 0.711576
G1 F11474.861
G1 X103.88 Y143.592 E.00843
; LINE_WIDTH: 0.665786
G1 F12116.999
G1 X104.07 Y143.615 E.00786
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.47 Y143.615 E.01527
G1 X156.235 Y143.615 E1.97636
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.706 Y131.955 I29.318 J-43.864 E.60771
G3 X136.932 Y119.13 I40.719 J-33.45 E.57452
G1 X136.912 Y119.083 E.00194
G1 X136.092 Y119.382 E.0333
G3 X132.727 Y119.301 I-1.547 J-5.724 E.13032
G3 X131.17 Y118.507 I2.742 J-7.29 E.06687
G1 X130.562 Y117.998 E.03026
G3 X129.081 Y116.142 I11.925 J-11.037 E.09073
G1 X126.698 Y112.93 E.15272
G3 X121.017 Y119.156 I-42.498 J-33.071 E.32212
G1 X120.566 Y119.579 E.0236
G3 X118.559 Y121.263 I-1040.297 J-1238.114 E.10001
G1 X117.027 Y122.547 E.07636
G3 X99.51 Y131.446 I-32.629 J-42.539 E.75437
G2 X99.67 Y132.536 I40.596 J-5.395 E.04204
G1 X99.67 Y132.585 E.00188
G1 X102.756 Y132.013 E.11981
G1 X102.993 Y132.006 E.00906
G1 X103.279 Y132.128 E.01185
M204 S250
G1 X103.04 Y132.604 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.123 Y132.765 E.00574
G3 X103.123 Y133.812 I-894.189 J.874 E.03315
G1 X103.123 Y141.812 E.25328
G1 X103.068 Y141.961 E.00502
G1 X102.898 Y142.038 E.00591
G1 X102.857 Y142.034 E.00131
G1 X99.123 Y141.343 E.12021
G1 X99.123 Y142.626 E.04063
G3 X98.942 Y143.857 I-4.854 J-.087 E.0395
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.614 J-43.499 E.51059
G3 X137.192 Y118.315 I40.27 J-33.072 E.4917
G1 X136.612 Y118.616 E.02069
G1 X135.912 Y118.859 E.02345
G1 X135.135 Y119.01 E.02506
G1 X134.371 Y119.041 E.0242
G1 X133.61 Y118.959 E.02426
G1 X132.867 Y118.767 E.02428
G1 X132.342 Y118.544 E.01807
G1 X131.508 Y118.068 E.03039
G1 X130.918 Y117.575 E.02435
G1 X130.409 Y117.003 E.02425
G3 X129.087 Y115.222 I369.812 J-275.926 E.0702
G1 X126.704 Y112.01 E.12664
G3 X121.199 Y118.227 I-42.075 J-31.712 E.26317
G3 X119.328 Y119.897 I-15.791 J-15.804 E.07947
G1 X116.672 Y122.124 E.10973
G3 X98.942 Y131.036 I-32.502 J-42.567 E.63189
G1 X98.963 Y131.622 E.01855
G1 X99.114 Y132.503 E.02829
G1 X99.122 Y133.249 E.02362
G1 X102.857 Y132.557 E.12026
G1 X102.953 Y132.581 E.00315
; WIPE_START
M204 S10000
M73 P88 R2
G1 X103.123 Y132.765 E-.09504
G1 X103.123 Y133.515 E-.28496
; WIPE_END
G1 E-.02 F1800
G1 X101.006 Y140.848 Z11 F36000
G1 X100.446 Y142.788 Z11
G1 Z10.6
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.911236
G1 F8980.5
G1 X100.336 Y142.788 E.00627
G1 X100.281 Y142.883 E.00627
G1 X100.336 Y142.978 E.00627
G1 X100.446 Y142.978 E.00627
G1 X100.501 Y142.883 E.00627
; WIPE_START
G1 X100.446 Y142.978 E-.076
G1 X100.336 Y142.978 E-.076
G1 X100.281 Y142.883 E-.076
G1 X100.336 Y142.788 E-.076
G1 X100.446 Y142.788 E-.07599
; WIPE_END
G1 E-.02 F1800
G1 X100.259 Y135.158 Z11 F36000
G1 X100.179 Y131.878 Z11
G1 Z10.6
G1 E.4 F1800
; LINE_WIDTH: 0.692463
G1 F11965.783
G1 X100.493 Y131.8 E.01388
; LINE_WIDTH: 0.73155
G1 F11294.965
G1 X100.807 Y131.722 E.0147
; LINE_WIDTH: 0.770636
G1 F10695.369
G1 X101.121 Y131.644 E.01553
; LINE_WIDTH: 0.816956
G1 F10062.356
G1 X101.444 Y131.561 E.01701
; LINE_WIDTH: 0.863276
G1 F9500.085
G1 X101.766 Y131.477 E.01801
; LINE_WIDTH: 0.909596
G1 F8997.327
G1 X102.089 Y131.394 E.01902
; LINE_WIDTH: 0.952386
G1 F8577.965
G1 X102.352 Y131.323 E.0163
; LINE_WIDTH: 0.995176
G1 F8195.954
G1 X102.615 Y131.253 E.01706
; WIPE_START
G1 X102.352 Y131.323 E-.10348
G1 X102.089 Y131.394 E-.10348
G1 X101.766 Y131.477 E-.12666
G1 X101.648 Y131.508 E-.04638
; WIPE_END
G1 E-.02 F1800
G1 X108.589 Y128.332 Z11 F36000
G1 X119.691 Y123.252 Z11
G1 Z10.6
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.62
G1 F13446.283
G3 X117.873 Y124.728 I-13.127 J-14.318 E.08946
G2 X119.511 Y125.874 I3.521 J-3.291 E.07687
G2 X123.281 Y125.781 I1.791 J-3.896 E.14909
G2 X125.166 Y123.954 I-6.115 J-8.195 E.10051
G3 X127.994 Y122.293 I4.288 J4.063 E.12687
G3 X131.293 Y123.034 I.898 J3.714 E.13373
G2 X133.65 Y125.341 I32.252 J-30.589 E.12593
G2 X137.867 Y126.034 I2.807 J-3.908 E.16907
G2 X140.477 Y130.104 I100.05 J-61.279 E.18462
G2 X138.362 Y132.05 I5.417 J8.006 E.11011
G3 X136.477 Y133.414 I-4.16 J-3.765 E.08949
G3 X132.707 Y133.322 I-1.791 J-3.896 E.14909
G3 X130.822 Y131.495 I6.115 J-8.195 E.10051
G2 X128.937 Y130.131 I-4.16 J3.765 E.08949
G2 X125.166 Y130.223 I-1.791 J3.896 E.14909
G2 X123.281 Y132.05 I6.115 J8.196 E.10051
G3 X121.396 Y133.414 I-4.16 J-3.765 E.08949
G3 X117.626 Y133.322 I-1.791 J-3.896 E.14909
G3 X115.741 Y131.495 I6.115 J-8.196 E.10051
G2 X113.855 Y130.131 I-4.16 J3.765 E.08949
G2 X110.085 Y130.223 I-1.791 J3.896 E.14909
G2 X108.2 Y132.05 I6.115 J8.196 E.10051
G3 X106.315 Y133.414 I-4.16 J-3.765 E.08949
G1 X105.927 Y133.536 E.0155
G1 X105.931 Y141.315 E.29699
G2 X108.2 Y140.862 I.54 J-3.205 E.09032
G2 X110.085 Y139.035 I-6.115 J-8.196 E.10051
G3 X111.97 Y137.671 I4.16 J3.765 E.08949
G3 X115.741 Y137.764 I1.791 J3.896 E.14909
G3 X117.626 Y139.591 I-6.114 J8.195 E.10051
G2 X119.511 Y140.955 I4.16 J-3.765 E.08949
G2 X123.281 Y140.862 I1.791 J-3.896 E.14909
G2 X125.166 Y139.035 I-6.115 J-8.196 E.10051
G3 X127.052 Y137.671 I4.16 J3.765 E.08949
G3 X130.822 Y137.764 I1.791 J3.896 E.14909
G3 X132.707 Y139.591 I-6.115 J8.196 E.10051
G2 X134.592 Y140.955 I4.16 J-3.765 E.08949
G2 X138.362 Y140.862 I1.791 J-3.896 E.14909
G2 X140.248 Y139.035 I-6.116 J-8.196 E.10051
G3 X142.133 Y137.671 I4.16 J3.765 E.08949
G3 X145.903 Y137.764 I1.791 J3.896 E.14909
G3 X147.788 Y139.591 I-6.115 J8.196 E.10051
G2 X149.673 Y140.955 I4.16 J-3.765 E.08949
G3 X150.887 Y141.36 I-.777 J4.353 E.04903
G1 X150.124 Y141.36 E.02914
G1 X140.326 Y128.939 F36000
; FEATURE: Floating vertical shell
; LINE_WIDTH: 0.545609
G1 F12000
G3 X136.041 Y121.335 I45.818 J-30.826 E.2912
G1 X135.939 Y121.191 E.00589
G3 X132.178 Y120.924 I-1.314 J-8.129 E.12681
G3 X131.094 Y120.466 I4.898 J-13.096 E.03921
G1 X130.233 Y119.936 E.03372
G1 X129.354 Y119.199 E.03821
G3 X128.428 Y118.138 I8.271 J-8.151 E.04698
G1 X126.82 Y115.97 E.08994
; LINE_WIDTH: 0.561493
G1 X126.758 Y115.918 E.00282
; LINE_WIDTH: 0.595806
G1 X126.695 Y115.865 E.003
; LINE_WIDTH: 0.63012
G1 X126.632 Y115.812 E.00319
G1 X126.565 Y115.86 E.00318
; LINE_WIDTH: 0.595806
G1 X126.499 Y115.908 E.00299
; LINE_WIDTH: 0.544389
G1 X126.432 Y115.956 E.00272
G3 X121.726 Y120.84 I-40.919 J-34.719 E.22569
G1 X118.026 Y123.941 E.16052
G3 X105.21 Y131.263 I-33.555 J-43.853 E.49227
; LINE_WIDTH: 0.56277
G1 X105.141 Y131.314 E.00292
; LINE_WIDTH: 0.599636
G1 X105.073 Y131.364 E.00313
; LINE_WIDTH: 0.636503
G1 X105.005 Y131.414 E.00333
G1 X105.09 Y131.616 E.00858
; LINE_WIDTH: 0.599636
G1 X105.175 Y131.817 E.00806
; LINE_WIDTH: 0.544745
G1 X105.26 Y132.019 E.00728
G3 X105.385 Y132.642 I-3.373 J1.003 E.0212
G1 X105.389 Y141.653 E.2998
; LINE_WIDTH: 0.563191
G1 X105.409 Y141.755 E.0036
; LINE_WIDTH: 0.588247
G1 X105.43 Y141.858 E.00377
G1 X105.638 Y141.901 E.0077
; LINE_WIDTH: 0.544818
G1 X151.691 Y141.901 E1.53256
G1 X151.836 Y141.88 E.00487
G2 X151.947 Y141.548 I-.387 J-.314 E.01191
G3 X146.738 Y136.818 I41.136 J-50.532 E.23426
G1 X145.769 Y135.802 E.04671
G1 X144.229 Y134.065 E.07726
G3 X140.377 Y129.013 I47.455 J-40.18 E.21151
; CHANGE_LAYER
; Z_HEIGHT: 10.76
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F12000
G1 X140.948 Y129.833 E-.38
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
G1 X104.065 Y131.233
G1 Z10.76
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.134 Y131.267 E.00294
G1 X104.512 Y131.679 E.02133
G1 X104.736 Y132.122 E.01895
G1 X104.844 Y132.622 E.01953
G3 X104.85 Y133.815 I-81.901 J1.017 E.04554
G1 X104.85 Y141.815 E.30543
G1 X104.739 Y142.443 E.02437
G1 X154.069 Y142.443 E1.88337
G3 X143.803 Y132.701 I31.96 J-43.96 E.54189
G3 X136.271 Y120.545 I42.235 J-34.58 E.54755
G1 X135.451 Y120.705 E.0319
G3 X133.343 Y120.663 I-.89 J-8.166 E.08072
G3 X130.408 Y119.398 I1.45 J-7.4 E.12296
G1 X129.811 Y118.897 E.02976
G1 X129.117 Y118.145 E.03904
G1 X126.663 Y114.849 E.1569
G3 X121.362 Y120.439 I-42.966 J-35.44 E.29437
G3 X119.312 Y122.16 I-443.327 J-525.858 E.1022
G1 X117.779 Y123.445 E.07636
G3 X103.96 Y131.182 I-33.373 J-43.396 E.60679
G1 X103.984 Y131.194 E.00102
G1 X103.77 Y131.733 F36000
G1 F13446.369
G1 X104.028 Y132.009 E.01442
G1 X104.185 Y132.318 E.01326
G1 X104.264 Y132.776 E.01774
G1 X104.264 Y141.815 E.34508
G1 X104.197 Y142.235 E.01626
G1 X103.927 Y142.712 E.02094
G1 X103.47 Y143.029 E.02124
G1 X155.749 Y143.029 E1.99596
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I29.897 J-44.054 E.59952
G3 X136.596 Y119.828 I41.659 J-34.119 E.56141
G1 X136.064 Y119.995 E.02128
G1 X135.344 Y120.13 E.02798
G1 X134.411 Y120.179 E.03566
G3 X130.774 Y118.941 I.206 J-6.567 E.14886
G1 X130.187 Y118.448 E.02923
G3 X129.065 Y117.104 I6.483 J-6.552 E.06697
G1 X126.682 Y113.891 E.15272
G3 X120.964 Y120.009 I-42.742 J-34.218 E.32002
G3 X118.936 Y121.711 I-612.632 J-727.882 E.1011
G1 X117.403 Y122.996 E.07636
G3 X105.169 Y130.069 I-33.372 J-43.609 E.54103
G1 X104.313 Y130.415 E.03525
G1 X103.941 Y130.563 E.01527
; LINE_WIDTH: 0.654911
G1 F12689.845
G1 X103.479 Y130.756 E.02025
; LINE_WIDTH: 0.689826
G1 F12013.915
G1 X103.017 Y130.948 E.02139
; LINE_WIDTH: 0.733205
G1 F10473.074
G1 X102.96 Y130.992 E.00326
; LINE_WIDTH: 0.776583
G1 F10247.245
G1 X102.903 Y131.035 E.00347
; LINE_WIDTH: 0.819962
G1 F10023.858
G1 X102.846 Y131.078 E.00367
; LINE_WIDTH: 0.863341
G1 F9499.349
G1 X102.789 Y131.121 E.00387
; LINE_WIDTH: 0.906719
G1 F9027.001
G1 X102.732 Y131.165 E.00407
; LINE_WIDTH: 0.950098
G1 F8599.402
G1 X102.674 Y131.208 E.00428
; LINE_WIDTH: 0.993476
G1 F8210.48
G1 X102.617 Y131.251 E.00448
G1 X102.682 Y131.271 E.00422
; LINE_WIDTH: 0.950098
G1 F8599.402
G1 X102.747 Y131.29 E.00402
; LINE_WIDTH: 0.906719
G1 F9027.001
G1 X102.811 Y131.31 E.00383
; LINE_WIDTH: 0.863341
G1 F9499.349
G1 X102.876 Y131.329 E.00364
; LINE_WIDTH: 0.819962
G1 F10023.858
G1 X102.94 Y131.349 E.00345
; LINE_WIDTH: 0.776583
G1 F10609.675
G1 X103.005 Y131.368 E.00326
; LINE_WIDTH: 0.733205
G1 F10825.9
G1 X103.069 Y131.387 E.00307
; LINE_WIDTH: 0.689826
G1 F12013.915
G1 X103.405 Y131.545 E.01584
; LINE_WIDTH: 0.654911
G1 F12689.845
G1 X103.697 Y131.682 E.01307
G1 X103.364 Y132.161 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X103.393 Y132.174 E.00119
G3 X103.678 Y132.776 I-.558 J.633 E.02617
G1 X103.678 Y141.415 E.3298
G1 X103.678 Y141.815 E.01527
G1 F13119.889
G1 X103.64 Y142.055 E.00928
G1 F12271.461
G1 X103.486 Y142.327 E.01195
; LINE_WIDTH: 0.664926
G1 F11220.717
G1 X103.389 Y142.42 E.00552
; LINE_WIDTH: 0.709856
G1 F10784.611
G1 X103.293 Y142.513 E.00591
; LINE_WIDTH: 0.754786
G1 F10357.149
G1 X103.197 Y142.607 E.0063
; LINE_WIDTH: 0.801231
G1 F9938.33
G1 X103.125 Y142.65 E.00417
; LINE_WIDTH: 0.847676
G1 F9682.299
G1 X103.054 Y142.694 E.00442
G1 X102.738 Y142.69 E.01677
; LINE_WIDTH: 0.842986
G1 F9738.454
G1 X102.471 Y142.627 E.01445
; LINE_WIDTH: 0.816096
G1 F10073.425
G1 X101.984 Y142.512 E.02548
; LINE_WIDTH: 0.767071
G1 F10747.406
G1 X101.498 Y142.397 E.02388
; LINE_WIDTH: 0.718046
G1 F11518.042
G1 X101.011 Y142.281 E.02229
; LINE_WIDTH: 0.669021
G1 F12407.731
G1 X100.525 Y142.166 E.02069
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X100.131 Y142.094 E.01527
G1 X99.678 Y142.01 E.01759
G3 X99.623 Y143.305 I-7.979 J.307 E.04955
G1 X99.561 Y143.615 E.01205
G1 X100.392 Y143.615 E.0317
; LINE_WIDTH: 0.646876
G1 F12848.299
G1 X100.665 Y143.601 E.01095
; LINE_WIDTH: 0.695904
G1 F11903.548
G1 X101.165 Y143.577 E.02156
; LINE_WIDTH: 0.744931
G1 F11082.262
G1 X101.664 Y143.552 E.02316
; LINE_WIDTH: 0.793959
G1 F10366.993
G1 X102.164 Y143.528 E.02476
; LINE_WIDTH: 0.842986
G1 F9738.454
G1 X102.663 Y143.503 E.02636
; LINE_WIDTH: 0.847676
G1 F9682.299
G3 X103.314 Y143.523 I.244 J2.604 E.0346
; LINE_WIDTH: 0.80214
G1 F10256.525
G1 X103.503 Y143.546 E.00954
; LINE_WIDTH: 0.756604
G1 F10863.499
G1 X103.692 Y143.569 E.00898
; LINE_WIDTH: 0.711068
G1 F11487.888
G1 X103.881 Y143.592 E.00841
; LINE_WIDTH: 0.665532
G1 F12129.722
G1 X104.071 Y143.615 E.00784
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.471 Y143.615 E.01527
G1 X156.235 Y143.615 E1.97633
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.69 J-44.272 E.6077
G3 X136.912 Y119.083 I40.834 J-33.521 E.57642
G1 X136.093 Y119.382 E.03327
G3 X134.379 Y119.594 I-1.768 J-7.259 E.06607
G1 X133.546 Y119.508 E.03199
G1 X132.726 Y119.301 E.03227
G3 X131.14 Y118.483 I2.666 J-7.118 E.06831
G3 X129.952 Y117.317 I4.34 J-5.605 E.0637
G1 X126.698 Y112.93 E.20855
G3 X120.566 Y119.579 I-42.13 J-32.697 E.34575
G3 X118.559 Y121.263 I-1031.86 J-1228.06 E.1
G1 X117.027 Y122.547 E.07636
G3 X99.51 Y131.446 I-32.603 J-42.488 E.75438
G2 X99.672 Y132.536 I42.665 J-5.776 E.04205
G1 X99.672 Y132.582 E.00178
G1 X102.758 Y132.011 E.11982
G1 X102.994 Y132.003 E.00902
G1 X103.281 Y132.126 E.01192
M204 S250
G1 X103.043 Y132.602 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.125 Y132.762 E.00572
G3 X103.125 Y133.815 I-897.651 J.876 E.03331
G1 X103.125 Y141.815 E.25328
G1 X103.07 Y141.963 E.00502
G1 X102.9 Y142.04 E.00591
G1 X102.859 Y142.036 E.00131
G1 X99.125 Y141.345 E.1202
G1 X99.125 Y142.626 E.04054
G3 X98.942 Y143.857 I-4.818 J-.089 E.03952
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I29.093 J-44.027 E.51056
G3 X137.192 Y118.315 I40.272 J-33.072 E.49167
G1 X136.612 Y118.616 E.02069
G1 X135.913 Y118.859 E.02343
G1 X135.135 Y119.01 E.02509
G1 X134.369 Y119.041 E.02428
G1 X133.607 Y118.959 E.02427
G1 X132.867 Y118.767 E.0242
G1 X132.342 Y118.544 E.01806
G1 X131.508 Y118.068 E.0304
G1 X130.92 Y117.577 E.02426
G1 X130.409 Y117.003 E.02434
G3 X130.278 Y116.829 I1.945 J-1.59 E.00688
G1 X126.704 Y112.01 E.18996
G3 X121.894 Y117.543 I-42.016 J-31.66 E.2323
G3 X119.328 Y119.897 I-23.879 J-23.462 E.11031
G1 X116.671 Y122.124 E.10973
G3 X98.942 Y131.036 I-32.199 J-41.962 E.63199
G1 X98.963 Y131.619 E.01848
G1 X99.116 Y132.502 E.02837
G1 X99.124 Y133.246 E.02355
G1 X102.859 Y132.554 E.12026
G1 X102.956 Y132.579 E.00316
; WIPE_START
M204 S10000
G1 X103.125 Y132.762 E-.09489
G1 X103.125 Y133.513 E-.28511
; WIPE_END
G1 E-.02 F1800
G1 X101.008 Y140.846 Z11.16 F36000
G1 X100.446 Y142.79 Z11.16
G1 Z10.76
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.908976
G1 F9003.705
G1 X100.337 Y142.79 E.00623
G1 X100.282 Y142.884 E.00623
G1 X100.337 Y142.979 E.00623
G1 X100.446 Y142.979 E.00623
G1 X100.501 Y142.884 E.00623
; WIPE_START
G1 X100.446 Y142.979 E-.076
G1 X100.337 Y142.979 E-.076
G1 X100.282 Y142.884 E-.076
G1 X100.337 Y142.79 E-.076
G1 X100.446 Y142.79 E-.076
; WIPE_END
G1 E-.02 F1800
G1 X100.259 Y135.16 Z11.16 F36000
G1 X100.179 Y131.877 Z11.16
G1 Z10.76
G1 E.4 F1800
; LINE_WIDTH: 0.69037
G1 F12003.964
G1 X100.493 Y131.799 E.01383
; LINE_WIDTH: 0.729443
G1 F11329.198
G1 X100.807 Y131.721 E.01465
; LINE_WIDTH: 0.768516
G1 F10726.253
G1 X101.12 Y131.643 E.01547
; LINE_WIDTH: 0.81485
G1 F10089.514
G1 X101.443 Y131.56 E.01696
; LINE_WIDTH: 0.861183
G1 F9524.137
G1 X101.766 Y131.476 E.01797
; LINE_WIDTH: 0.907516
G1 F9018.76
G1 X102.089 Y131.393 E.01898
; LINE_WIDTH: 0.950496
G1 F8595.661
G1 X102.353 Y131.322 E.01634
; LINE_WIDTH: 0.993476
G1 F8210.48
G1 X102.617 Y131.251 E.01711
; WIPE_START
G1 X102.353 Y131.322 E-.10398
G1 X102.089 Y131.393 E-.10399
G1 X101.766 Y131.476 E-.12668
G1 X101.65 Y131.506 E-.04534
; WIPE_END
G1 E-.02 F1800
G1 X104.406 Y138.624 Z11.16 F36000
G1 X105.086 Y140.383 Z11.16
G1 Z10.76
G1 E.4 F1800
; FEATURE: Bridge
; LINE_WIDTH: 0.60126
; LAYER_HEIGHT: 0.6

G1 X105.998 Y141.913 E.20606
G1 X106.757 Y141.913 E.08771
G1 X105.379 Y139.601 E.31135
G1 X105.378 Y138.328 E.14727
G1 X107.515 Y141.913 E.4828
G1 X108.273 Y141.913 E.08771
G1 X105.377 Y137.055 E.65424
G1 X105.377 Y135.782 E.14727
G1 X109.031 Y141.913 E.82568
G1 X109.789 Y141.913 E.08771
G1 X105.376 Y134.509 E.99712
G1 X105.376 Y133.236 E.14727
G1 X110.547 Y141.913 E1.16856
G1 X111.306 Y141.913 E.08771
G1 X105.004 Y131.34 E1.42397
G2 X105.615 Y131.094 I-3.74 J-10.201 E.07627
G1 X112.064 Y141.913 E1.45704
G1 X112.822 Y141.913 E.08771
G1 X106.217 Y130.832 E1.4924
G1 X106.819 Y130.569 E.07594
G1 X113.58 Y141.913 E1.52775
G1 X114.338 Y141.913 E.08771
G1 X107.42 Y130.307 E1.56311
G2 X108.013 Y130.029 I-4.685 J-10.752 E.07572
M73 P89 R2
G1 X115.096 Y141.913 E1.60055
G1 X115.854 Y141.913 E.08771
G1 X108.603 Y129.746 E1.63862
G1 X109.188 Y129.455 E.07556
G1 X116.613 Y141.913 E1.67774
G1 X117.371 Y141.913 E.08771
G1 X109.77 Y129.16 E1.71754
G1 X110.345 Y128.852 E.07542
G1 X118.129 Y141.913 E1.75895
G1 X118.887 Y141.913 E.08771
G1 X110.919 Y128.545 E1.80036
G1 X110.955 Y128.526 E.00469
G2 X111.492 Y128.234 I-5.023 J-9.871 E.07072
G1 X119.645 Y141.913 E1.84228
G1 X120.403 Y141.913 E.08771
G1 X112.058 Y127.911 E1.88578
G2 X112.618 Y127.578 I-5.741 J-10.306 E.07535
G1 X121.162 Y141.913 E1.93056
G1 X121.92 Y141.913 E.08771
G1 X113.174 Y127.24 E1.97613
G1 X113.731 Y126.901 E.07534
G1 X122.678 Y141.913 E2.02171
G1 X123.436 Y141.913 E.08771
G1 X114.281 Y126.553 E2.06859
G1 X114.828 Y126.198 E.0754
G1 X124.194 Y141.913 E2.11648
G1 X124.952 Y141.913 E.08771
G1 X115.37 Y125.835 E2.16525
G1 X115.907 Y125.464 E.07551
G1 X125.711 Y141.913 E2.21525
G1 X126.469 Y141.913 E.08771
M73 P89 R1
G1 X116.444 Y125.093 E2.26526
G2 X116.972 Y124.707 I-5.742 J-8.404 E.07569
G1 X127.227 Y141.913 E2.31727
G1 X127.985 Y141.913 E.08771
G1 X117.498 Y124.317 E2.36972
G1 X118.024 Y123.928 E.07572
G1 X128.743 Y141.913 E2.42216
G1 X129.501 Y141.913 E.08771
G1 X118.529 Y123.504 E2.47923
G1 X119.035 Y123.08 E.07632
M73 P90 R1
G1 X130.26 Y141.913 E2.5363
G1 X131.018 Y141.913 E.08771
G1 X119.54 Y122.656 E2.59337
G1 X120.046 Y122.233 E.07632
G1 X131.776 Y141.913 E2.65044
G1 X132.534 Y141.913 E.08771
G1 X120.552 Y121.809 E2.70751
G1 X121.057 Y121.385 E.07632
G1 X133.292 Y141.913 E2.76458
G1 X134.05 Y141.913 E.08771
G1 X121.563 Y120.961 E2.82164
G2 X122.055 Y120.516 I-3.892 J-4.799 E.07687
G1 X134.808 Y141.913 E2.88163
G1 X135.567 Y141.913 E.08771
G1 X122.542 Y120.061 E2.94291
G1 X123.022 Y119.593 E.07749
G1 X136.325 Y141.913 E3.00592
G1 X137.083 Y141.913 E.08771
G1 X123.5 Y119.124 E3.06906
G1 X123.971 Y118.642 E.07796
G1 X137.841 Y141.913 E3.13401
G1 X138.599 Y141.913 E.08771
G1 X124.441 Y118.159 E3.19906
G1 X124.903 Y117.662 E.07851
G1 X139.357 Y141.913 E3.26603
G1 X140.116 Y141.913 E.08771
G1 X125.365 Y117.164 E3.33301
G2 X125.818 Y116.652 I-10.128 J-9.417 E.07909
M73 P91 R1
G1 X140.874 Y141.913 E3.40196
G1 X141.632 Y141.913 E.08771
G1 X126.27 Y116.139 E3.47111
G1 X126.637 Y115.703 E.06589
G1 X127.307 Y116.606 E.13003
G1 X142.39 Y141.913 E3.40821
G1 X143.148 Y141.913 E.08771
G1 X129.869 Y119.634 E3.00046
G2 X130.312 Y119.979 I2.043 J-2.164 E.06506
G1 X131.126 Y120.47 E.10993
G1 X143.906 Y141.913 E2.88784
G1 X144.665 Y141.913 E.08771
G1 X132.137 Y120.894 E2.83066
G1 X132.596 Y121.026 E.05522
G2 X133.042 Y121.141 I.99 J-2.901 E.05338
G1 X145.423 Y141.913 E2.79743
G1 X146.181 Y141.913 E.08771
G1 X133.869 Y121.256 E2.78198
G2 X134.645 Y121.286 I.563 J-4.457 E.08998
G1 X146.939 Y141.913 E2.77789
G1 X147.697 Y141.913 E.08771
G1 X135.379 Y121.246 E2.78336
G2 X135.965 Y121.149 I-.922 J-7.381 E.06877
G1 X136.272 Y121.795 E.0828
G2 X142.452 Y131.841 I49.246 J-23.373 E1.36719
G1 X148.455 Y141.913 E1.3564
G1 X149.214 Y141.913 E.08771
G1 X145.157 Y135.107 E.91665
G1 X145.434 Y135.414 E.04786
G2 X147.209 Y137.278 I38.481 J-34.859 E.29777
M73 P92 R1
G1 X149.972 Y141.913 E.62428
G1 X150.73 Y141.913 E.08771
G1 X148.968 Y138.958 E.39802
G2 X150.568 Y140.37 I23.028 J-24.48 E.24686
G1 X151.662 Y142.206 E.24728
G1 X152.433 Y141.829 F36000
; FEATURE: Floating vertical shell
; LINE_WIDTH: 0.562116
; LAYER_HEIGHT: 0.16
G1 F12000
G2 X152.409 Y141.944 I-.033 J.053 E.00823
; CHANGE_LAYER
; Z_HEIGHT: 10.92
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F12000
G1 X152.367 Y141.944 E-.06645
G1 X152.334 Y141.887 E-.10452
G1 X152.367 Y141.829 E-.10453
G1 X152.433 Y141.829 E-.10451
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
G1 X104.062 Y131.227
G1 Z10.92
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.131 Y131.26 E.00289
G1 X104.64 Y131.89 E.03095
G1 X104.793 Y132.298 E.01663
G1 X104.852 Y132.774 E.0183
G1 X104.852 Y141.817 E.34527
G1 X104.743 Y142.443 E.02427
G1 X154.069 Y142.443 E1.88323
G3 X143.803 Y132.701 I31.507 J-43.482 E.54192
G3 X136.271 Y120.545 I42.214 J-34.567 E.54755
G3 X135.168 Y120.737 I-2.509 J-11.164 E.04274
G3 X133.344 Y120.663 I-.566 J-8.624 E.06984
G3 X130.406 Y119.397 I1.484 J-7.482 E.12305
G1 X129.808 Y118.894 E.02983
G1 X129.117 Y118.145 E.0389
G1 X126.663 Y114.849 E.15688
G3 X121.36 Y120.441 I-42.976 J-35.449 E.29448
G3 X119.312 Y122.16 I-484.473 J-574.923 E.10209
G1 X117.779 Y123.445 E.07636
G3 X103.965 Y131.18 I-33.374 J-43.397 E.60659
G1 X103.981 Y131.188 E.00068
G1 X103.775 Y131.739 F36000
G1 F13446.369
G1 X104.118 Y132.156 E.02062
G3 X104.266 Y132.774 I-1.216 J.618 E.02449
G1 X104.266 Y141.817 E.34527
G1 X104.2 Y142.238 E.01626
G1 X103.929 Y142.715 E.02094
G1 X103.475 Y143.029 E.02107
G1 X155.749 Y143.029 E1.99574
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.06 J-44.232 E.5995
G3 X136.596 Y119.828 I41.64 J-34.108 E.56141
G1 X136.067 Y119.995 E.02115
G1 X135.346 Y120.129 E.028
G1 X134.411 Y120.179 E.03576
G3 X133.284 Y120.054 I.634 J-10.866 E.0433
G1 X132.577 Y119.868 E.02791
G3 X130.772 Y118.939 I2.775 J-7.617 E.07771
G1 X130.185 Y118.446 E.02929
G3 X129.065 Y117.104 I6.488 J-6.551 E.06683
G1 X126.682 Y113.891 E.15272
G3 X120.962 Y120.01 I-42.743 J-34.219 E.3201
G3 X118.936 Y121.711 I-670.633 J-797.053 E.10103
G1 X117.403 Y122.996 E.07636
G3 X105.472 Y129.941 I-33.374 J-43.612 E.52847
G1 X104.308 Y130.414 E.04796
G1 X103.937 Y130.565 E.01527
; LINE_WIDTH: 0.655606
G1 F12675.649
G1 X103.477 Y130.757 E.02019
; LINE_WIDTH: 0.691216
G1 F11988.492
G1 X103.017 Y130.949 E.02135
; LINE_WIDTH: 0.734159
G1 F10469.27
G1 X102.96 Y130.992 E.00325
; LINE_WIDTH: 0.777102
G1 F10244.832
G1 X102.904 Y131.035 E.00345
; LINE_WIDTH: 0.820045
G1 F10022.801
G1 X102.847 Y131.078 E.00365
; LINE_WIDTH: 0.862988
G1 F9503.394
G1 X102.79 Y131.121 E.00385
; LINE_WIDTH: 0.905931
G1 F9035.167
G1 X102.733 Y131.164 E.00404
; LINE_WIDTH: 0.948873
G1 F8610.914
G1 X102.676 Y131.207 E.00424
; LINE_WIDTH: 0.991816
G1 F8224.716
G1 X102.62 Y131.25 E.00444
G1 X102.684 Y131.269 E.00419
; LINE_WIDTH: 0.948873
G1 F8610.914
G1 X102.748 Y131.288 E.004
; LINE_WIDTH: 0.905931
G1 F9035.167
G1 X102.812 Y131.307 E.00381
; LINE_WIDTH: 0.862988
G1 F9503.394
G1 X102.877 Y131.326 E.00362
; LINE_WIDTH: 0.820045
G1 F10022.801
G1 X102.941 Y131.346 E.00343
; LINE_WIDTH: 0.777102
G1 F10602.268
G1 X103.005 Y131.365 E.00325
; LINE_WIDTH: 0.734159
G1 F10817.261
G1 X103.069 Y131.384 E.00306
; LINE_WIDTH: 0.691216
G1 F11988.492
G1 X103.405 Y131.541 E.01589
; LINE_WIDTH: 0.655606
G1 F12675.649
G1 X103.707 Y131.682 E.01349
G1 X103.365 Y132.157 F36000
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X103.392 Y132.169 E.00115
G1 X103.603 Y132.436 E.01298
G1 X103.68 Y132.774 E.01323
G1 X103.68 Y141.417 E.33
G1 X103.68 Y141.817 E.01527
G1 F13136.659
G1 X103.642 Y142.057 E.00928
G1 F12287.645
G1 X103.488 Y142.329 E.01195
; LINE_WIDTH: 0.664706
G1 F11236.239
G1 X103.392 Y142.423 E.00551
; LINE_WIDTH: 0.709416
G1 F10799.959
G1 X103.295 Y142.516 E.0059
; LINE_WIDTH: 0.754126
G1 F10372.318
G1 X103.199 Y142.609 E.0063
; LINE_WIDTH: 0.800271
G1 F9953.293
G1 X103.127 Y142.652 E.00416
; LINE_WIDTH: 0.846416
G1 F9697.321
G1 X103.056 Y142.695 E.00441
G1 X102.74 Y142.692 E.01673
; LINE_WIDTH: 0.841756
G1 F9753.29
G1 X102.472 Y142.628 E.01451
; LINE_WIDTH: 0.814886
G1 F10089.04
G1 X101.985 Y142.513 E.02544
; LINE_WIDTH: 0.766164
G1 F10760.733
G1 X101.498 Y142.398 E.02385
; LINE_WIDTH: 0.717441
G1 F11528.243
G1 X101.012 Y142.284 E.02227
; LINE_WIDTH: 0.668719
G1 F12413.648
G1 X100.525 Y142.169 E.02068
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X100.132 Y142.096 E.01527
G1 X99.68 Y142.012 E.01753
G3 X99.624 Y143.305 I-7.869 J.306 E.04944
G1 X99.562 Y143.615 E.01207
G1 X100.392 Y143.615 E.03169
; LINE_WIDTH: 0.646856
G1 F12856.724
G1 X100.668 Y143.601 E.01101
; LINE_WIDTH: 0.695581
G1 F11909.354
G1 X101.167 Y143.577 E.02155
; LINE_WIDTH: 0.744306
G1 F11092.018
G1 X101.667 Y143.552 E.02314
; LINE_WIDTH: 0.793031
G1 F10379.666
G1 X102.166 Y143.528 E.02473
; LINE_WIDTH: 0.841756
G1 F9753.29
G1 X102.665 Y143.504 E.02632
; LINE_WIDTH: 0.846416
G1 F9697.321
G3 X103.315 Y143.524 I.244 J2.613 E.03451
; LINE_WIDTH: 0.801132
G1 F10270.007
G1 X103.504 Y143.547 E.00952
; LINE_WIDTH: 0.755848
G1 F10876.715
G1 X103.693 Y143.569 E.00896
; LINE_WIDTH: 0.710564
G1 F11500.833
G1 X103.883 Y143.592 E.0084
; LINE_WIDTH: 0.66528
G1 F12142.361
G1 X104.072 Y143.615 E.00783
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.472 Y143.615 E.01527
G1 X156.235 Y143.615 E1.97629
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.106 J-43.631 E.60777
G3 X136.912 Y119.083 I40.731 J-33.458 E.57643
G1 X136.093 Y119.382 E.03326
G3 X134.389 Y119.593 I-1.779 J-7.36 E.06569
G1 X133.546 Y119.508 E.03237
G1 X132.726 Y119.301 E.03227
G3 X131.138 Y118.482 I2.669 J-7.126 E.06839
G3 X129.952 Y117.317 I4.338 J-5.6 E.06361
G1 X126.698 Y112.93 E.20856
G3 X120.565 Y119.58 I-42.13 J-32.697 E.34579
G3 X118.559 Y121.263 I-1132.417 J-1347.999 E.09996
G1 X117.027 Y122.547 E.07636
G3 X99.51 Y131.446 I-32.602 J-42.485 E.75437
G2 X99.674 Y132.535 I44.494 J-6.124 E.04206
G1 X99.674 Y132.58 E.00169
G1 X102.76 Y132.008 E.11982
G1 X102.995 Y132.001 E.00897
G1 X103.282 Y132.122 E.01189
M204 S250
G1 X103.044 Y132.598 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.127 Y132.76 E.00575
G3 X103.128 Y133.817 I-859.348 J.878 E.03347
G1 X103.128 Y141.817 E.25328
G1 X103.072 Y141.966 E.00502
G1 X102.902 Y142.043 E.00591
G1 X102.861 Y142.039 E.00131
G1 X99.128 Y141.348 E.1202
G1 X99.127 Y142.626 E.04046
G3 X98.942 Y143.857 I-4.786 J-.091 E.03953
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.616 J-43.501 E.51062
G3 X137.192 Y118.315 I40.61 J-33.274 E.49164
G1 X136.612 Y118.616 E.02069
G1 X135.913 Y118.859 E.02342
G1 X135.137 Y119.01 E.02502
G1 X134.369 Y119.041 E.02435
G1 X133.607 Y118.959 E.02427
G1 X132.867 Y118.767 E.02419
G1 X132.326 Y118.538 E.01861
G1 X131.506 Y118.067 E.02992
G1 X130.918 Y117.575 E.02429
G1 X130.409 Y117.003 E.02425
G3 X130.278 Y116.829 I1.959 J-1.601 E.00688
G1 X126.704 Y112.01 E.18996
G3 X121.895 Y117.543 I-42.012 J-31.657 E.23229
G3 X119.302 Y119.919 I-23.943 J-23.528 E.11139
G1 X116.671 Y122.124 E.10867
G3 X98.942 Y131.036 I-32.199 J-41.964 E.63198
G1 X98.963 Y131.618 E.01844
G1 X99.118 Y132.502 E.0284
G1 X99.126 Y133.244 E.02349
G1 X102.861 Y132.552 E.12026
G1 X102.957 Y132.576 E.00314
; WIPE_START
M204 S10000
G1 X103.127 Y132.76 E-.09517
G1 X103.127 Y133.509 E-.28483
; WIPE_END
G1 E-.02 F1800
G1 X101.01 Y140.842 Z11.32 F36000
G1 X100.447 Y142.791 Z11.32
G1 Z10.92
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.906736
G1 F9026.823
G1 X100.338 Y142.791 E.0062
G1 X100.283 Y142.886 E.0062
G1 X100.338 Y142.98 E.0062
G1 X100.447 Y142.98 E.0062
G1 X100.501 Y142.886 E.0062
; WIPE_START
G1 X100.447 Y142.98 E-.076
G1 X100.338 Y142.98 E-.076
G1 X100.283 Y142.886 E-.076
G1 X100.338 Y142.791 E-.076
G1 X100.447 Y142.791 E-.07599
; WIPE_END
G1 E-.02 F1800
G1 X100.26 Y135.161 Z11.32 F36000
G1 X100.179 Y131.876 Z11.32
G1 Z10.92
G1 E.4 F1800
; LINE_WIDTH: 0.688296
G1 F12042.022
G1 X100.493 Y131.798 E.01378
; LINE_WIDTH: 0.727356
G1 F11363.309
G1 X100.806 Y131.72 E.0146
; LINE_WIDTH: 0.766416
G1 F10757.022
G1 X101.12 Y131.642 E.01542
; LINE_WIDTH: 0.812743
G1 F10116.821
G1 X101.443 Y131.559 E.01692
; LINE_WIDTH: 0.85907
G1 F9548.542
G1 X101.766 Y131.475 E.01792
; LINE_WIDTH: 0.905396
G1 F9040.71
G1 X102.088 Y131.392 E.01893
; LINE_WIDTH: 0.948606
G1 F8613.43
G1 X102.354 Y131.321 E.01639
; LINE_WIDTH: 0.991816
G1 F8224.716
G1 X102.62 Y131.25 E.01716
; WIPE_START
G1 X102.354 Y131.321 E-.10449
G1 X102.088 Y131.392 E-.10449
G1 X101.766 Y131.475 E-.12668
G1 X101.653 Y131.504 E-.04434
; WIPE_END
G1 E-.02 F1800
G1 X106.467 Y130.409 Z11.32 F36000
G1 Z10.92
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.626586
G1 F13296.751
G1 X105.123 Y131.753 E.07338
G1 X105.276 Y132.176 E.01737
G1 X105.311 Y132.402 E.00884
G1 X107.461 Y130.252 E.11737
G2 X109.076 Y129.475 I-17.133 J-37.665 E.06919
G1 X105.346 Y133.205 E.20367
G1 X105.346 Y134.042 E.03232
G1 X110.835 Y128.554 E.29969
G2 X112.791 Y127.435 I-18.037 J-33.81 E.08702
G1 X105.346 Y134.88 E.40649
G1 X105.347 Y135.717 E.03232
G1 X115.047 Y126.016 E.52967
G2 X117.984 Y123.917 I-40.867 J-60.278 E.13938
G1 X105.347 Y136.554 E.68999
G1 X105.348 Y137.391 E.03232
G1 X126.83 Y115.909 E1.17294
G1 X127.186 Y116.39 E.02312
G1 X105.348 Y138.228 E1.1924
G1 X105.348 Y139.065 E.03232
G1 X127.543 Y116.871 E1.21186
G1 X127.9 Y117.352 E.02312
G1 X105.349 Y139.903 E1.23131
G1 X105.349 Y140.74 E.03232
G1 X128.257 Y117.832 E1.25077
G1 X128.613 Y118.313 E.02312
G1 X105.349 Y141.577 E1.27023
G3 X105.332 Y141.945 I-1.171 J.129 E.0143
G1 X105.819 Y141.945 E.0188
G1 X129.003 Y118.761 E1.26588
G1 X129.306 Y119.107 E.01775
G1 X129.408 Y119.194 E.00517
G1 X106.656 Y141.945 E1.24224
G1 X107.494 Y141.945 E.03234
G1 X129.858 Y119.581 E1.2211
G2 X130.327 Y119.95 I2.167 J-2.278 E.02307
G1 X108.331 Y141.945 E1.20098
G1 X109.169 Y141.945 E.03234
G1 X130.849 Y120.265 E1.18376
G2 X131.401 Y120.551 I1.245 J-1.724 E.02407
G1 X110.006 Y141.945 E1.16814
G1 X110.844 Y141.945 E.03234
G1 X131.991 Y120.798 E1.15464
G2 X132.627 Y121 I1.05 J-2.203 E.02583
G1 X111.682 Y141.945 E1.14362
G1 X112.519 Y141.945 E.03234
G1 X133.307 Y121.158 E1.13503
G1 X134.058 Y121.245 E.02918
G1 X113.357 Y141.945 E1.13029
G1 X114.194 Y141.945 E.03234
G1 X134.895 Y121.245 E1.13026
G2 X135.838 Y121.14 I-.088 J-5.062 E.03669
G1 X115.032 Y141.945 E1.13602
G1 X115.869 Y141.945 E.03234
G1 X136.215 Y121.6 E1.11089
G1 X136.484 Y122.168 E.02428
G1 X116.707 Y141.945 E1.07986
G1 X117.545 Y141.945 E.03234
G1 X136.755 Y122.735 E1.04892
G1 X137.04 Y123.287 E.02401
G1 X118.382 Y141.945 E1.01874
G1 X119.22 Y141.945 E.03234
M73 P93 R1
G1 X137.325 Y123.84 E.98857
G2 X137.617 Y124.386 I8.292 J-4.083 E.0239
G1 X120.057 Y141.945 E.95877
G1 X120.895 Y141.945 E.03234
G1 X137.917 Y124.923 E.92943
G1 X138.217 Y125.461 E.02377
G1 X121.732 Y141.945 E.90009
G1 X122.57 Y141.945 E.03234
G1 X138.522 Y125.994 E.87098
G1 X138.837 Y126.516 E.02355
G1 X123.408 Y141.945 E.84246
G1 X124.245 Y141.945 E.03234
G1 X139.152 Y127.038 E.81393
G2 X139.476 Y127.552 I7.774 J-4.542 E.02345
G1 X125.083 Y141.945 E.78588
G1 X125.92 Y141.945 E.03234
G1 X139.806 Y128.06 E.75818
G1 X140.136 Y128.567 E.02337
G1 X126.758 Y141.945 E.73047
G1 X127.595 Y141.945 E.03234
G1 X140.471 Y129.07 E.70301
G1 X140.816 Y129.563 E.02322
G1 X128.433 Y141.945 E.6761
G1 X129.271 Y141.945 E.03234
G1 X141.16 Y130.056 E.6492
G2 X141.513 Y130.541 I8.085 J-5.5 E.02316
G1 X130.108 Y141.945 E.6227
G1 X130.946 Y141.945 E.03234
G1 X141.871 Y131.02 E.59651
G1 X142.229 Y131.5 E.02311
G1 X131.783 Y141.945 E.57032
G1 X132.621 Y141.945 E.03234
G1 X142.595 Y131.971 E.54461
G1 X142.964 Y132.44 E.02303
G1 X133.458 Y141.945 E.519
G1 X134.296 Y141.945 E.03234
G1 X143.332 Y132.909 E.49339
G2 X143.71 Y133.369 I9.252 J-7.217 E.02298
G1 X135.134 Y141.945 E.46829
G1 X135.971 Y141.945 E.03234
G1 X144.089 Y133.827 E.44326
G2 X144.474 Y134.28 I10.302 J-8.356 E.02294
G1 X136.809 Y141.945 E.41853
G1 X137.646 Y141.945 E.03234
G1 X144.863 Y134.729 E.39402
G2 X145.258 Y135.171 I10.588 J-9.082 E.0229
G1 X138.484 Y141.945 E.36989
G1 X139.321 Y141.945 E.03234
G1 X145.656 Y135.611 E.34587
G2 X146.059 Y136.045 I10.137 J-9.02 E.02288
G1 X140.159 Y141.945 E.32217
G1 X140.997 Y141.945 E.03234
G1 X146.466 Y136.476 E.29865
G2 X146.88 Y136.899 I9.881 J-9.237 E.02287
G1 X141.834 Y141.945 E.27551
G1 X142.672 Y141.945 E.03234
G1 X147.296 Y137.321 E.25249
G2 X147.718 Y137.737 I9.959 J-9.672 E.02287
G1 X143.509 Y141.945 E.22978
G1 X144.347 Y141.945 E.03234
G1 X148.143 Y138.15 E.20725
G2 X148.575 Y138.554 I9.174 J-9.374 E.02288
G1 X145.184 Y141.945 E.18515
G1 X146.022 Y141.945 E.03234
G1 X149.01 Y138.958 E.16314
G2 X149.449 Y139.355 I8.213 J-8.636 E.0229
G1 X146.859 Y141.945 E.14141
G1 X147.697 Y141.945 E.03234
G1 X149.894 Y139.748 E.11997
G2 X150.342 Y140.138 I8.383 J-9.172 E.02292
G1 X148.535 Y141.945 E.09868
G1 X149.372 Y141.945 E.03234
G1 X150.797 Y140.521 E.07778
G2 X151.253 Y140.902 I9.056 J-10.378 E.02296
G1 X150.21 Y141.945 E.05697
G1 X151.047 Y141.945 E.03234
G1 X151.717 Y141.276 E.03657
G2 X152.184 Y141.646 I8.493 J-10.236 E.02302
G1 X151.621 Y142.209 E.03074
; CHANGE_LAYER
; Z_HEIGHT: 11.08
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13296.751
G1 X152.184 Y141.646 E-.30252
G1 X152.024 Y141.519 E-.07748
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
G1 X103.938 Y131.19
G1 Z11.08
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.354 Y131.468 E.01912
G1 X104.731 Y132.09 E.02775
G1 X104.831 Y132.472 E.01509
G3 X104.854 Y133.82 I-39.122 J1.344 E.05146
G1 X104.854 Y141.82 E.30543
G1 X104.759 Y142.421 E.02324
G2 X106.069 Y142.443 I1.014 J-21.069 E.05003
G1 X154.069 Y142.443 E1.8326
G3 X143.803 Y132.701 I31.506 J-43.48 E.54192
G3 X136.271 Y120.545 I41.793 J-34.306 E.54758
G1 X135.454 Y120.705 E.03178
G3 X133.273 Y120.651 I-.89 J-8.17 E.08356
G3 X130.408 Y119.398 I1.567 J-7.483 E.12023
G3 X129.012 Y118.015 I5.44 J-6.888 E.07519
G1 X126.663 Y114.849 E.15049
G3 X121.359 Y120.441 I-42.731 J-35.219 E.29449
G3 X119.312 Y122.16 I-484.8 J-575.314 E.10208
G1 X117.779 Y123.445 E.07636
G3 X104.022 Y131.158 I-33.347 J-43.357 E.60427
G1 X103.462 Y131.565 F36000
G1 F13446.369
G1 X103.902 Y131.841 E.0198
G1 X104.182 Y132.294 E.02037
G1 X104.268 Y132.771 E.0185
G1 X104.268 Y141.82 E.34546
G1 X104.202 Y142.24 E.01626
G1 X103.931 Y142.717 E.02094
G1 X103.481 Y143.029 E.0209
G1 X155.749 Y143.029 E1.99553
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.059 J-44.231 E.5995
G3 X136.596 Y119.828 I41.249 J-33.868 E.56144
G1 X136.067 Y119.995 E.02115
G1 X135.346 Y120.129 E.028
G1 X134.411 Y120.179 E.03576
G3 X133.285 Y120.054 I.629 J-10.83 E.04329
G1 X132.577 Y119.868 E.02791
G3 X130.774 Y118.941 I2.778 J-7.622 E.07763
G1 X130.187 Y118.448 E.02923
G3 X129.065 Y117.104 I7.731 J-7.594 E.06694
G1 X126.682 Y113.891 E.15272
G3 X120.962 Y120.011 I-44.275 J-35.651 E.32008
G3 X118.936 Y121.711 I-669.967 J-796.257 E.10102
G1 X117.403 Y122.996 E.07636
G3 X105.186 Y130.064 I-33.031 J-43.005 E.54039
G1 X104.061 Y130.508 E.04618
G1 X103.688 Y130.654 E.01527
G1 F12277.59
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.666164
G1 F10942.732
G1 X103.23 Y130.857 E.00425
; LINE_WIDTH: 0.712331
G1 F10610.589
G1 X103.143 Y130.913 E.00456
; LINE_WIDTH: 0.758499
G1 F10283.564
G1 X103.056 Y130.968 E.00488
; LINE_WIDTH: 0.804666
G1 F9961.659
G1 X102.969 Y131.024 E.00519
; LINE_WIDTH: 0.850834
G1 F9644.855
G1 X102.882 Y131.08 E.0055
; LINE_WIDTH: 0.897001
G1 F9128.689
G1 X102.796 Y131.136 E.00581
; LINE_WIDTH: 0.943169
G1 F8664.964
G1 X102.709 Y131.192 E.00612
; LINE_WIDTH: 0.989336
G1 F8246.074
G1 X102.622 Y131.248 E.00643
G1 X102.708 Y131.276 E.00561
; LINE_WIDTH: 0.943169
G1 F8664.964
G1 X102.793 Y131.303 E.00534
; LINE_WIDTH: 0.897001
G1 F9128.689
G1 X102.879 Y131.331 E.00506
; LINE_WIDTH: 0.850834
G1 F9644.855
G1 X102.965 Y131.358 E.00479
; LINE_WIDTH: 0.804666
G1 F10222.893
G1 X103.051 Y131.386 E.00452
; LINE_WIDTH: 0.758499
G1 F10506.967
G1 X103.137 Y131.413 E.00425
; LINE_WIDTH: 0.712331
G1 F10794.944
G1 X103.222 Y131.441 E.00398
; LINE_WIDTH: 0.666164
G1 F11086.783
G1 X103.308 Y131.468 E.00371
; LINE_WIDTH: 0.619996
G1 F11389.137
G1 X103.386 Y131.517 E.00351
G1 X103.214 Y132.075 F36000
G1 F13446.369
G1 X103.483 Y132.251 E.01225
G1 X103.639 Y132.515 E.01171
G1 X103.682 Y132.843 E.01264
G1 X103.682 Y141.42 E.32745
G1 X103.682 Y141.82 E.01527
G1 F13153.551
G1 X103.645 Y142.06 E.00928
G1 F12303.983
G1 X103.49 Y142.332 E.01195
; LINE_WIDTH: 0.664486
G1 F11251.846
G1 X103.394 Y142.425 E.00551
; LINE_WIDTH: 0.708976
G1 F10815.437
G1 X103.297 Y142.518 E.0059
; LINE_WIDTH: 0.753466
G1 F10387.615
G1 X103.2 Y142.611 E.00629
; LINE_WIDTH: 0.799306
G1 F9968.426
G1 X103.129 Y142.654 E.00415
; LINE_WIDTH: 0.845146
G1 F9712.511
G1 X103.058 Y142.697 E.0044
G1 X102.742 Y142.693 E.01669
; LINE_WIDTH: 0.840516
G1 F9768.291
G1 X102.473 Y142.63 E.01456
; LINE_WIDTH: 0.813686
G1 F10104.575
G1 X101.986 Y142.515 E.0254
; LINE_WIDTH: 0.765264
G1 F10773.983
G1 X101.499 Y142.4 E.02382
; LINE_WIDTH: 0.716841
G1 F11538.377
G1 X101.013 Y142.286 E.02225
; LINE_WIDTH: 0.668419
G1 F12419.52
G1 X100.526 Y142.171 E.02067
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X100.133 Y142.098 E.01527
G1 X99.682 Y142.015 E.01748
G3 X99.626 Y143.304 I-7.723 J.307 E.04934
G1 X99.563 Y143.615 E.01208
G1 X100.393 Y143.615 E.0317
; LINE_WIDTH: 0.646816
G1 F12857.564
G1 X100.67 Y143.601 E.01106
; LINE_WIDTH: 0.695241
G1 F11915.48
G1 X101.169 Y143.577 E.02154
; LINE_WIDTH: 0.743666
G1 F11102.027
G1 X101.669 Y143.553 E.02312
; LINE_WIDTH: 0.792091
G1 F10392.542
G1 X102.168 Y143.529 E.0247
; LINE_WIDTH: 0.840516
G1 F9768.291
G1 X102.668 Y143.504 E.02628
; LINE_WIDTH: 0.845146
G1 F9712.511
G3 X103.317 Y143.524 I.243 J2.623 E.03442
; LINE_WIDTH: 0.800116
G1 F10283.633
G1 X103.506 Y143.547 E.0095
; LINE_WIDTH: 0.755086
G1 F10890.081
G1 X103.695 Y143.57 E.00894
; LINE_WIDTH: 0.710056
G1 F11513.932
G1 X103.884 Y143.592 E.00838
; LINE_WIDTH: 0.665026
G1 F12155.127
G1 X104.073 Y143.615 E.00782
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.473 Y143.615 E.01527
G1 X156.235 Y143.615 E1.97626
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.105 J-43.631 E.60777
G3 X136.912 Y119.083 I41.375 J-33.848 E.57638
G1 X136.093 Y119.382 E.03328
G3 X132.727 Y119.301 I-1.547 J-5.713 E.13036
G3 X131.14 Y118.484 I2.673 J-7.134 E.06831
G1 X130.564 Y118 E.0287
G1 X130.204 Y117.606 E.0204
G3 X129.081 Y116.142 I13.444 J-11.479 E.07046
G1 X126.698 Y112.93 E.15272
G3 X120.565 Y119.58 I-42.95 J-33.454 E.34578
G3 X118.559 Y121.263 I-1143.244 J-1360.91 E.09996
G1 X117.027 Y122.547 E.07636
G3 X99.511 Y131.446 I-32.589 J-42.46 E.75437
G2 X99.676 Y132.535 I46.196 J-6.459 E.04207
G1 X99.676 Y132.577 E.0016
G1 X102.762 Y132.006 E.11982
G1 X103.137 Y132.031 E.01435
M204 S250
G1 X102.975 Y132.557 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.117 Y132.697 E.00632
G1 X103.13 Y141.82 E.28883
G1 X103.074 Y141.968 E.00502
G1 X102.904 Y142.045 E.00591
G1 X102.863 Y142.041 E.00131
G1 X99.13 Y141.35 E.12021
G1 X99.13 Y142.63 E.04051
G3 X98.942 Y143.857 I-4.756 J-.098 E.03942
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.615 J-43.5 E.51062
G3 X137.192 Y118.315 I40.815 J-33.396 E.49163
G1 X136.612 Y118.616 E.02069
G1 X135.913 Y118.859 E.02343
G1 X135.137 Y119.01 E.02501
G1 X134.369 Y119.041 E.02435
G1 X133.607 Y118.959 E.02426
G1 X132.867 Y118.767 E.02419
G1 X132.329 Y118.539 E.01851
G1 X131.509 Y118.068 E.02995
G1 X130.92 Y117.577 E.02428
G1 X130.409 Y117.003 E.02434
G3 X129.087 Y115.222 I338.314 J-252.541 E.0702
G1 X126.704 Y112.01 E.12664
G3 X121.894 Y117.544 I-43.13 J-32.629 E.2323
G3 X119.306 Y119.915 I-23.931 J-23.517 E.11118
G1 X116.671 Y122.124 E.10885
G3 X98.942 Y131.036 I-32.427 J-42.417 E.63191
G1 X98.964 Y131.617 E.01839
G1 X99.12 Y132.501 E.02844
G1 X99.128 Y133.241 E.02342
G1 X102.863 Y132.549 E.12026
G1 X102.885 Y132.551 E.0007
; WIPE_START
M204 S10000
G1 X103.117 Y132.697 E-.1042
G1 X103.118 Y133.423 E-.2758
; WIPE_END
G1 E-.02 F1800
G1 X101.026 Y140.763 Z11.48 F36000
G1 X100.448 Y142.792 Z11.48
G1 Z11.08
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.904476
G1 F9050.269
G1 X100.339 Y142.792 E.00617
G1 X100.285 Y142.887 E.00617
G1 X100.339 Y142.981 E.00617
G1 X100.448 Y142.981 E.00617
G1 X100.502 Y142.887 E.00617
; WIPE_START
G1 X100.448 Y142.981 E-.076
G1 X100.339 Y142.981 E-.076
G1 X100.285 Y142.887 E-.076
G1 X100.339 Y142.792 E-.076
G1 X100.448 Y142.792 E-.076
; WIPE_END
G1 E-.02 F1800
G1 X100.26 Y135.162 Z11.48 F36000
G1 X100.179 Y131.875 Z11.48
G1 Z11.08
G1 E.4 F1800
; LINE_WIDTH: 0.686223
G1 F12080.322
G1 X100.493 Y131.797 E.01373
; LINE_WIDTH: 0.72527
G1 F11397.627
G1 X100.806 Y131.719 E.01455
; LINE_WIDTH: 0.764316
G1 F10787.968
G1 X101.12 Y131.641 E.01537
; LINE_WIDTH: 0.768396
G1 F10728.007
G1 X101.15 Y131.633 E.00148
; LINE_WIDTH: 0.814546
G1 F10093.438
G1 X101.464 Y131.552 E.0165
; LINE_WIDTH: 0.860696
G1 F9529.746
G1 X101.778 Y131.47 E.01748
; LINE_WIDTH: 0.906846
G1 F9025.685
G1 X102.092 Y131.388 E.01845
; LINE_WIDTH: 0.952996
G1 F8572.269
G1 X102.406 Y131.307 E.01943
; LINE_WIDTH: 0.989336
G1 F8246.074
G1 X102.622 Y131.248 E.01395
; WIPE_START
G1 X102.406 Y131.307 E-.08517
G1 X102.092 Y131.388 E-.12328
G1 X101.778 Y131.47 E-.12329
G1 X101.655 Y131.502 E-.04826
; WIPE_END
G1 E-.02 F1800
G1 X104.23 Y138.687 Z11.48 F36000
G1 X105.088 Y141.081 Z11.48
G1 Z11.08
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.624876
G1 F13335.253
G1 X105.952 Y141.945 E.04705
G1 X106.787 Y141.945 E.03215
G1 X105.352 Y140.51 E.07817
G1 X105.352 Y139.675 E.03215
G1 X107.623 Y141.945 E.12364
G1 X108.458 Y141.945 E.03215
G1 X105.352 Y138.839 E.1691
G1 X105.352 Y138.004 E.03215
G1 X109.293 Y141.945 E.21457
G1 X110.128 Y141.945 E.03215
G1 X105.352 Y137.169 E.26004
G1 X105.352 Y136.334 E.03215
G1 X110.963 Y141.945 E.30551
G1 X111.798 Y141.945 E.03215
G1 X105.352 Y135.499 E.35098
G1 X105.352 Y134.664 E.03215
G1 X112.633 Y141.945 E.39645
G1 X113.469 Y141.945 E.03215
G1 X105.352 Y133.828 E.44192
G1 X105.352 Y132.993 E.03215
G1 X114.304 Y141.945 E.48738
G1 X115.139 Y141.945 E.03215
G1 X105.23 Y132.036 E.53948
G1 X105.196 Y131.906 E.00518
G1 X104.865 Y131.363 E.0245
G1 X105.24 Y131.211 E.01558
G1 X115.974 Y141.945 E.5844
G1 X116.809 Y141.945 E.03215
G1 X105.829 Y130.965 E.5978
G1 X106.409 Y130.71 E.02439
G1 X117.644 Y141.945 E.61169
G1 X118.479 Y141.945 E.03215
G1 X106.989 Y130.455 E.62559
G1 X107.569 Y130.2 E.02439
G1 X119.315 Y141.945 E.63948
G1 X120.15 Y141.945 E.03215
G1 X108.137 Y129.933 E.65402
G1 X108.701 Y129.662 E.02409
G1 X120.985 Y141.945 E.66878
G1 X121.82 Y141.945 E.03215
G1 X109.258 Y129.383 E.68394
G1 X109.807 Y129.098 E.02385
M73 P94 R1
G1 X122.655 Y141.945 E.69948
G1 X123.49 Y141.945 E.03215
G1 X110.357 Y128.812 E.71502
G2 X110.903 Y128.523 I-4.952 J-10.014 E.02379
G1 X124.326 Y141.945 E.73076
G1 X125.161 Y141.945 E.03215
G1 X111.44 Y128.225 E.74698
G2 X111.97 Y127.92 I-5.226 J-9.692 E.02355
G1 X125.996 Y141.945 E.76359
G1 X126.831 Y141.945 E.03215
G1 X112.495 Y127.61 E.78049
G1 X113.02 Y127.299 E.02347
G1 X127.666 Y141.945 E.7974
G1 X128.501 Y141.945 E.03215
G1 X113.541 Y126.985 E.81451
G1 X114.052 Y126.661 E.0233
G1 X129.336 Y141.945 E.83215
G1 X130.172 Y141.945 E.03215
G1 X114.563 Y126.337 E.84979
G1 X115.062 Y126.001 E.02316
G1 X131.007 Y141.945 E.86808
G1 X131.842 Y141.945 E.03215
G1 X115.561 Y125.665 E.88638
G2 X116.057 Y125.326 I-5.331 J-8.337 E.02314
G1 X132.677 Y141.945 E.90483
G1 X133.512 Y141.945 E.03215
G1 X116.543 Y124.977 E.92384
G1 X117.029 Y124.627 E.02304
G1 X134.347 Y141.945 E.94285
G1 X135.182 Y141.945 E.03215
G1 X117.511 Y124.274 E.96212
G1 X117.985 Y123.913 E.02295
G1 X136.018 Y141.945 E.98174
G1 X136.853 Y141.945 E.03215
G1 X118.444 Y123.537 E1.00222
G1 X118.898 Y123.156 E.02282
G1 X137.688 Y141.945 E1.02297
G1 X138.523 Y141.945 E.03215
G1 X119.352 Y122.775 E1.04372
G1 X119.806 Y122.394 E.02282
G1 X139.358 Y141.945 E1.06446
G1 X140.193 Y141.945 E.03215
G1 X120.261 Y122.013 E1.08521
G1 X120.715 Y121.632 E.02282
G1 X141.029 Y141.945 E1.10595
G1 X141.864 Y141.945 E.03215
G1 X121.169 Y121.251 E1.1267
G1 X121.623 Y120.869 E.02282
G1 X142.699 Y141.945 E1.14744
G1 X143.534 Y141.945 E.03215
G1 X122.058 Y120.47 E1.16922
G1 X122.49 Y120.066 E.02275
G1 X144.369 Y141.945 E1.19118
G1 X145.204 Y141.945 E.03215
G1 X122.913 Y119.654 E1.2136
G1 X123.336 Y119.242 E.02274
G1 X146.039 Y141.945 E1.23607
G1 X146.875 Y141.945 E.03215
G1 X123.75 Y118.821 E1.25898
G1 X124.161 Y118.397 E.02274
G1 X147.71 Y141.945 E1.28206
G1 X148.545 Y141.945 E.03215
G1 X124.572 Y117.973 E1.30514
G2 X124.974 Y117.539 I-7.487 J-7.322 E.02275
G1 X149.38 Y141.945 E1.32877
G1 X150.215 Y141.945 E.03215
G1 X125.373 Y117.103 E1.35249
G2 X125.766 Y116.661 I-6.992 J-6.618 E.02278
G1 X151.05 Y141.945 E1.37655
G1 X151.885 Y141.945 E.03215
G1 X126.153 Y116.213 E1.40097
G1 X126.539 Y115.764 E.0228
G1 X127.248 Y116.472 E.03856
G1 X128.617 Y118.318 E.08847
G1 X129.317 Y119.118 E.04093
G1 X130.099 Y119.788 E.03963
G2 X131.284 Y120.508 I6.42 J-9.231 E.05341
G1 X142.091 Y131.316 E.5884
G3 X140.059 Y128.449 I43.211 J-32.778 E.13531
G1 X132.6 Y120.99 E.4061
G2 X133.644 Y121.199 I1.506 J-4.811 E.04107
G1 X138.647 Y126.202 E.27238
G3 X137.553 Y124.272 I63.425 J-37.245 E.08539
G1 X134.539 Y121.258 E.1641
G2 X135.334 Y121.218 I.128 J-5.399 E.03068
G1 X137.262 Y123.146 E.10494
; CHANGE_LAYER
; Z_HEIGHT: 11.24
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13335.253
G1 X136.555 Y122.439 E-.38
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
G1 X103.94 Y131.189
G1 Z11.24
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.365 Y131.475 E.01958
G1 X104.646 Y131.888 E.01906
G1 X104.8 Y132.306 E.01702
G1 X104.856 Y132.84 E.02047
G1 X104.856 Y141.822 E.34294
G1 X104.761 Y142.423 E.02324
G2 X106.069 Y142.443 I1.012 J-23.643 E.04995
G1 X154.069 Y142.443 E1.8326
G3 X143.803 Y132.701 I31.793 J-43.783 E.5419
G3 X136.271 Y120.545 I41.667 J-34.228 E.54758
G1 X135.451 Y120.705 E.03188
G3 X133.267 Y120.65 I-.888 J-8.08 E.08367
G3 X130.472 Y119.447 I1.618 J-7.608 E.11692
G3 X129.012 Y118.015 I6.007 J-7.586 E.07826
G1 X126.663 Y114.849 E.15049
G3 X121.364 Y120.437 I-42.7 J-35.187 E.29428
G3 X119.312 Y122.16 I-431.637 J-511.895 E.10229
G1 X117.779 Y123.445 E.07636
G3 X104.024 Y131.157 I-33.346 J-43.354 E.60418
G1 X103.458 Y131.559 F36000
G1 F13446.369
G1 X103.907 Y131.842 E.02028
G1 X104.123 Y132.153 E.01444
G1 X104.231 Y132.445 E.01191
G1 X104.27 Y132.84 E.01513
G1 X104.27 Y141.822 E.34294
G1 X104.204 Y142.243 E.01626
G1 X103.933 Y142.72 E.02094
G1 X103.487 Y143.029 E.02073
G1 X155.749 Y143.029 E1.99531
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.046 J-44.216 E.5995
G3 X136.596 Y119.828 I41.131 J-33.796 E.56145
G1 X136.067 Y119.995 E.02116
G1 X135.344 Y120.129 E.02809
G1 X134.411 Y120.179 E.03567
G3 X133.277 Y120.052 I.648 J-10.955 E.04359
G1 X132.574 Y119.867 E.02773
G3 X130.819 Y118.975 I2.879 J-7.843 E.07534
G3 X129.065 Y117.104 I5.013 J-6.455 E.09833
G1 X126.682 Y113.891 E.15272
G3 X120.965 Y120.008 I-42.661 J-34.141 E.31995
G3 X118.936 Y121.711 I-590.927 J-701.983 E.10117
G1 X117.403 Y122.996 E.07636
G3 X104.855 Y130.2 I-33.032 J-43.005 E.55404
G1 X104.061 Y130.508 E.03253
G1 X103.689 Y130.654 E.01527
G1 F12286.908
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.665956
G1 F10951.528
G1 X103.23 Y130.856 E.00424
; LINE_WIDTH: 0.711916
G1 F10620.312
G1 X103.143 Y130.912 E.00455
; LINE_WIDTH: 0.757876
G1 F10294.164
G1 X103.057 Y130.968 E.00486
; LINE_WIDTH: 0.803836
G1 F9973.102
G1 X102.97 Y131.024 E.00516
; LINE_WIDTH: 0.849796
G1 F9657.127
G1 X102.884 Y131.079 E.00547
; LINE_WIDTH: 0.895756
G1 F9141.883
G1 X102.797 Y131.135 E.00578
; LINE_WIDTH: 0.941716
G1 F8678.834
G1 X102.711 Y131.191 E.00609
; LINE_WIDTH: 0.987676
G1 F8260.432
G1 X102.624 Y131.247 E.0064
G1 X102.71 Y131.274 E.00558
; LINE_WIDTH: 0.941716
G1 F8678.834
G1 X102.795 Y131.301 E.00531
; LINE_WIDTH: 0.895756
G1 F9141.883
G1 X102.881 Y131.329 E.00504
; LINE_WIDTH: 0.849796
G1 F9657.127
G1 X102.966 Y131.356 E.00477
; LINE_WIDTH: 0.803836
G1 F10233.919
G1 X103.052 Y131.383 E.0045
; LINE_WIDTH: 0.757876
G1 F10517.294
G1 X103.137 Y131.411 E.00424
; LINE_WIDTH: 0.711916
G1 F10804.509
G1 X103.223 Y131.438 E.00397
; LINE_WIDTH: 0.665956
G1 F11095.593
G1 X103.308 Y131.465 E.0037
; LINE_WIDTH: 0.619996
G1 F11379.697
G1 X103.382 Y131.511 E.0033
G1 X103.214 Y132.072 F36000
G1 F13446.369
G1 X103.489 Y132.252 E.01254
G1 X103.662 Y132.584 E.0143
G3 X103.685 Y133.822 I-34.168 J1.232 E.04728
G1 X103.685 Y141.422 E.29016
G1 X103.685 Y141.822 E.01527
G1 F13170.361
G1 X103.647 Y142.062 E.00928
G1 F12320.241
G1 X103.492 Y142.334 E.01195
; LINE_WIDTH: 0.664266
G1 F11267.393
G1 X103.396 Y142.427 E.00551
; LINE_WIDTH: 0.708536
G1 F10830.81
G1 X103.299 Y142.52 E.00589
; LINE_WIDTH: 0.752806
G1 F10402.831
G1 X103.202 Y142.613 E.00628
; LINE_WIDTH: 0.798346
G1 F9983.458
G1 X103.131 Y142.656 E.00414
; LINE_WIDTH: 0.843886
G1 F9727.628
G1 X103.06 Y142.699 E.00439
G1 X102.744 Y142.695 E.01666
; LINE_WIDTH: 0.839286
G1 F9783.217
G1 X102.473 Y142.632 E.01463
; LINE_WIDTH: 0.812456
G1 F10120.548
G1 X101.986 Y142.517 E.02536
; LINE_WIDTH: 0.764341
G1 F10787.599
G1 X101.5 Y142.402 E.02379
; LINE_WIDTH: 0.716226
G1 F11548.784
G1 X101.013 Y142.288 E.02223
; LINE_WIDTH: 0.668111
G1 F12425.546
G1 X100.526 Y142.173 E.02066
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X100.133 Y142.1 E.01527
G1 X99.685 Y142.017 E.01741
G3 X99.627 Y143.303 I-7.668 J.303 E.04921
G1 X99.564 Y143.615 E.01212
G1 X100.394 Y143.615 E.03168
; LINE_WIDTH: 0.646816
G1 F12857.564
G1 X100.672 Y143.601 E.01113
; LINE_WIDTH: 0.694934
G1 F11921.026
G1 X101.172 Y143.577 E.02153
; LINE_WIDTH: 0.743051
G1 F11111.661
G1 X101.671 Y143.553 E.0231
; LINE_WIDTH: 0.791169
G1 F10405.209
G1 X102.171 Y143.529 E.02467
; LINE_WIDTH: 0.839286
G1 F9783.217
G1 X102.67 Y143.505 E.02624
; LINE_WIDTH: 0.843886
G1 F9727.628
G3 X103.319 Y143.525 I.243 J2.633 E.03433
; LINE_WIDTH: 0.799108
G1 F10297.187
G1 X103.507 Y143.547 E.00948
; LINE_WIDTH: 0.75433
G1 F10903.397
G1 X103.696 Y143.57 E.00892
; LINE_WIDTH: 0.709552
G1 F11526.949
G1 X103.885 Y143.592 E.00837
; LINE_WIDTH: 0.664774
G1 F12167.838
G1 X104.074 Y143.615 E.00781
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.474 Y143.615 E.01527
G1 X156.235 Y143.615 E1.97622
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.648 E.01662
G3 X144.705 Y131.954 I29.318 J-43.864 E.60774
G3 X136.912 Y119.083 I40.727 J-33.456 E.57643
G1 X136.093 Y119.382 E.03328
G3 X132.724 Y119.3 I-1.548 J-5.683 E.13049
G3 X131.165 Y118.503 I2.743 J-7.283 E.06697
G3 X129.952 Y117.317 I5.057 J-6.385 E.0649
G1 X126.698 Y112.93 E.20855
G3 X120.567 Y119.578 I-42.973 J-33.474 E.34569
G3 X118.559 Y121.263 I-999.561 J-1189.526 E.10005
G1 X117.027 Y122.547 E.07636
G3 X99.511 Y131.446 I-32.589 J-42.46 E.75437
G2 X99.678 Y132.535 I49.193 J-7.004 E.04208
G1 X99.678 Y132.575 E.00151
G1 X102.764 Y132.003 E.11983
G1 X103.137 Y132.028 E.01427
M204 S250
G1 X102.977 Y132.554 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.107 Y132.667 E.00546
G3 X103.132 Y133.822 I-27.138 J1.149 E.03659
G1 X103.132 Y141.822 E.25328
G1 X103.076 Y141.971 E.00502
G1 X102.906 Y142.048 E.00591
G1 X102.865 Y142.044 E.00131
G1 X99.132 Y141.353 E.12021
G1 X99.132 Y142.626 E.0403
G3 X98.942 Y143.857 I-4.719 J-.096 E.03956
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.127 Y131.597 I28.616 J-43.501 E.51062
G3 X137.192 Y118.315 I40.603 J-33.27 E.49164
G1 X136.612 Y118.616 E.02068
M73 P94 R0
G1 X135.913 Y118.859 E.02344
G1 X135.135 Y119.01 E.02508
G1 X134.369 Y119.041 E.02428
G1 X133.6 Y118.958 E.02449
G1 X132.865 Y118.766 E.02404
G1 X132.341 Y118.544 E.01803
G1 X131.508 Y118.068 E.03037
G3 X130.278 Y116.829 I3.683 J-4.884 E.05546
G1 X126.704 Y112.01 E.18996
G3 X121.44 Y117.993 I-42.502 J-32.087 E.25254
G3 X119.324 Y119.9 I-20.216 J-20.297 E.09022
G1 X116.671 Y122.124 E.10959
G3 X98.942 Y131.036 I-32.253 J-42.07 E.63197
G1 X98.964 Y131.614 E.01832
G1 X99.122 Y132.501 E.02851
G1 X99.13 Y133.238 E.02335
G1 X102.865 Y132.547 E.12026
G1 X102.887 Y132.548 E.00069
; WIPE_START
M204 S10000
M73 P95 R0
G1 X103.107 Y132.667 E-.09516
G1 X103.123 Y133.416 E-.28484
; WIPE_END
G1 E-.02 F1800
G1 X101.029 Y140.756 Z11.64 F36000
G1 X100.448 Y142.794 Z11.64
G1 Z11.24
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.902256
G1 F9073.418
G1 X100.34 Y142.794 E.00614
G1 X100.285 Y142.888 E.00614
G1 X100.34 Y142.982 E.00614
G1 X100.448 Y142.982 E.00614
G1 X100.502 Y142.888 E.00614
; WIPE_START
G1 X100.448 Y142.982 E-.076
G1 X100.34 Y142.982 E-.076
G1 X100.285 Y142.888 E-.076
G1 X100.34 Y142.794 E-.076
G1 X100.448 Y142.794 E-.07599
; WIPE_END
G1 E-.02 F1800
G1 X100.26 Y135.164 Z11.64 F36000
G1 X100.179 Y131.874 Z11.64
G1 Z11.24
G1 E.4 F1800
; LINE_WIDTH: 0.684156
G1 F12118.742
G1 X100.492 Y131.796 E.01368
; LINE_WIDTH: 0.723176
G1 F11432.264
G1 X100.806 Y131.718 E.0145
; LINE_WIDTH: 0.762196
G1 F10819.389
G1 X101.119 Y131.64 E.01532
; LINE_WIDTH: 0.766276
G1 F10759.08
G1 X101.149 Y131.632 E.00147
; LINE_WIDTH: 0.812456
G1 F10120.548
G1 X101.463 Y131.551 E.01646
; LINE_WIDTH: 0.858636
G1 F9553.562
G1 X101.778 Y131.469 E.01744
; LINE_WIDTH: 0.904816
G1 F9046.734
G1 X102.092 Y131.387 E.01842
; LINE_WIDTH: 0.950996
G1 F8590.972
G1 X102.406 Y131.306 E.0194
; LINE_WIDTH: 0.987676
G1 F8260.432
G1 X102.624 Y131.247 E.01406
; WIPE_START
G1 X102.406 Y131.306 E-.08597
G1 X102.092 Y131.387 E-.12335
G1 X101.778 Y131.469 E-.12335
G1 X101.657 Y131.5 E-.04734
; WIPE_END
G1 E-.02 F1800
G1 X106.545 Y130.366 Z11.64 F36000
G1 Z11.24
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.626226
G1 F13304.838
G1 X105.134 Y131.778 E.07703
G1 X105.285 Y132.188 E.0169
G1 X105.315 Y132.433 E.0095
G1 X107.532 Y130.217 E.12094
G2 X109.139 Y129.447 I-31.908 J-68.654 E.06876
G1 X105.354 Y133.231 E.20654
G1 X105.354 Y134.069 E.0323
G1 X110.913 Y128.51 E.30334
G2 X112.858 Y127.401 I-21.659 J-40.277 E.08641
G1 X105.354 Y134.906 E.4095
G1 X105.354 Y135.743 E.0323
G1 X115.16 Y125.937 E.53509
G2 X118.004 Y123.902 I-28.929 J-43.451 E.13498
G1 X118.172 Y123.762 E.00842
G1 X105.354 Y136.58 E.69944
G1 X105.354 Y137.417 E.0323
G1 X126.843 Y115.927 E1.17263
G1 X127.2 Y116.408 E.02309
G1 X105.354 Y138.254 E1.19209
G1 X105.354 Y139.091 E.0323
G1 X127.556 Y116.888 E1.21154
G1 X127.913 Y117.369 E.02309
G1 X105.354 Y139.928 E1.23099
G1 X105.354 Y140.765 E.0323
G1 X128.269 Y117.85 E1.25044
G2 X128.626 Y118.329 I3.31 J-2.093 E.0231
G1 X105.354 Y141.602 E1.26994
G3 X105.337 Y141.945 I-1.095 J.118 E.01332
G1 X105.847 Y141.945 E.0197
G1 X129.016 Y118.777 E1.26424
G2 X129.42 Y119.21 I2.812 J-2.222 E.02288
G1 X106.684 Y141.945 E1.24062
G1 X107.522 Y141.945 E.0323
G1 X129.872 Y119.595 E1.21962
G2 X130.349 Y119.955 I1.503 J-1.495 E.02314
G1 X108.359 Y141.945 E1.19997
G1 X109.196 Y141.945 E.0323
G1 X130.867 Y120.274 E1.18255
G2 X131.42 Y120.559 I1.186 J-1.627 E.02408
G1 X110.033 Y141.945 E1.16704
G1 X110.87 Y141.945 E.0323
G1 X132.009 Y120.806 E1.15353
G2 X132.647 Y121.006 I1.041 J-2.205 E.02585
G1 X111.707 Y141.945 E1.14263
G1 X112.544 Y141.945 E.0323
G1 X133.328 Y121.162 E1.13413
G1 X134.079 Y121.247 E.02919
G1 X113.381 Y141.945 E1.12947
G1 X114.218 Y141.945 E.0323
G1 X134.919 Y121.244 E1.12962
G2 X135.866 Y121.134 I-.109 J-5.072 E.03684
G1 X115.055 Y141.945 E1.13562
G1 X115.892 Y141.945 E.0323
G1 X136.222 Y121.615 E1.10936
G1 X136.491 Y122.183 E.02425
G1 X116.729 Y141.945 E1.07837
G1 X117.566 Y141.945 E.0323
G1 X136.763 Y122.749 E1.04752
G1 X137.049 Y123.3 E.02396
G1 X118.403 Y141.945 E1.01745
G1 X119.24 Y141.945 E.0323
G1 X137.335 Y123.851 E.98738
G1 X137.621 Y124.402 E.02396
G1 X120.077 Y141.945 E.9573
G1 X120.914 Y141.945 E.0323
G1 X137.922 Y124.938 E.92807
G1 X138.223 Y125.473 E.02372
G1 X121.752 Y141.945 E.89884
G1 X122.589 Y141.945 E.0323
G1 X138.528 Y126.006 E.8698
G1 X138.844 Y126.528 E.02352
G1 X123.426 Y141.945 E.84132
G1 X124.263 Y141.945 E.0323
G1 X139.159 Y127.049 E.81284
G2 X139.482 Y127.563 I8.494 J-4.988 E.02343
G1 X125.1 Y141.945 E.78481
G1 X125.937 Y141.945 E.0323
G1 X139.811 Y128.071 E.75707
G1 X140.14 Y128.58 E.02336
G1 X126.774 Y141.945 E.72933
G1 X127.611 Y141.945 E.0323
G1 X140.48 Y129.076 E.70223
G1 X140.823 Y129.57 E.0232
G1 X128.448 Y141.945 E.6753
G1 X129.285 Y141.945 E.0323
G1 X141.167 Y130.064 E.64836
G2 X141.519 Y130.549 I7.463 J-5.042 E.02313
G1 X130.122 Y141.945 E.62188
G1 X130.959 Y141.945 E.0323
G1 X141.876 Y131.028 E.59573
G1 X142.234 Y131.508 E.02308
G1 X131.796 Y141.945 E.56957
G1 X132.633 Y141.945 E.0323
G1 X142.601 Y131.978 E.5439
G1 X142.969 Y132.447 E.023
G1 X133.47 Y141.945 E.51832
G1 X134.307 Y141.945 E.0323
G1 X143.337 Y132.915 E.49274
G2 X143.715 Y133.375 I9.444 J-7.375 E.02295
G1 X135.144 Y141.945 E.46767
G1 X135.982 Y141.945 E.0323
G1 X144.094 Y133.833 E.44267
G2 X144.479 Y134.285 I10.27 J-8.359 E.02291
G1 X136.819 Y141.945 E.418
G1 X137.656 Y141.945 E.0323
G1 X144.867 Y134.734 E.39351
G2 X145.266 Y135.172 I7.148 J-6.096 E.02287
G1 X138.493 Y141.945 E.36958
G1 X139.33 Y141.945 E.0323
G1 X145.667 Y135.608 E.34582
G1 X146.069 Y136.044 E.02286
G1 X140.167 Y141.945 E.32205
G1 X141.004 Y141.945 E.0323
G1 X146.47 Y136.479 E.29828
G2 X146.885 Y136.902 I6.025 J-5.494 E.02284
G1 X141.841 Y141.945 E.27522
G1 X142.678 Y141.945 E.0323
G1 X147.301 Y137.322 E.25229
G2 X147.72 Y137.741 I6.34 J-5.917 E.02284
G1 X143.515 Y141.945 E.22944
G1 X144.352 Y141.945 E.0323
G1 X148.151 Y138.147 E.20728
G1 X148.582 Y138.553 E.02285
G1 X145.189 Y141.945 E.18512
G1 X146.026 Y141.945 E.0323
G1 X149.013 Y138.959 E.16296
G2 X149.452 Y139.357 I6.468 J-6.697 E.02287
G1 X146.863 Y141.945 E.14125
G1 X147.7 Y141.945 E.0323
G1 X149.896 Y139.75 E.11982
G2 X150.343 Y140.139 I8.203 J-8.961 E.02289
G1 X148.537 Y141.945 E.09855
G1 X149.374 Y141.945 E.0323
G1 X150.798 Y140.522 E.07768
G2 X151.254 Y140.903 I9.038 J-10.354 E.02293
G1 X150.212 Y141.945 E.05689
G1 X151.049 Y141.945 E.0323
G1 X151.718 Y141.276 E.03651
G2 X152.185 Y141.646 I8.515 J-10.265 E.02299
G1 X151.622 Y142.209 E.0307
; CHANGE_LAYER
; Z_HEIGHT: 11.4
; LAYER_HEIGHT: 0.16
; WIPE_START
G1 F13304.838
G1 X152.185 Y141.646 E-.30234
G1 X152.025 Y141.519 E-.07766
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
G1 X103.945 Y131.187
G1 Z11.4
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.356 Y131.46 E.01886
G1 X104.732 Y132.078 E.0276
G1 X104.834 Y132.464 E.01524
G3 X104.858 Y133.825 I-38.661 J1.352 E.05197
G1 X104.858 Y141.825 E.30543
G1 X104.763 Y142.426 E.02324
G2 X106.069 Y142.443 I1.009 J-26.909 E.04987
G1 X154.069 Y142.443 E1.8326
G3 X143.803 Y132.701 I31.793 J-43.783 E.54187
G3 X136.271 Y120.545 I41.664 J-34.227 E.54761
G1 X135.451 Y120.705 E.0319
G3 X133.27 Y120.651 I-.886 J-8.165 E.08354
G3 X130.408 Y119.398 I1.54 J-7.414 E.12014
G3 X129.012 Y118.015 I5.369 J-6.817 E.07517
G1 X126.663 Y114.849 E.15052
G3 X121.362 Y120.439 I-43.647 J-36.085 E.29437
G3 X119.312 Y122.16 I-445.167 J-528.045 E.10219
G1 X117.779 Y123.445 E.07636
G3 X104.029 Y131.155 I-33.241 J-43.167 E.60399
G1 X103.469 Y131.563 F36000
G1 F13446.369
G1 X103.904 Y131.834 E.01959
G1 X104.184 Y132.284 E.02025
G1 X104.272 Y132.766 E.0187
G1 X104.272 Y141.825 E.34584
G1 X104.206 Y142.245 E.01626
G1 X103.935 Y142.722 E.02094
G1 X103.492 Y143.029 E.02056
G1 X155.749 Y143.029 E1.99509
G1 X155.76 Y142.968 E.00235
G1 X155.769 Y142.92 E.00186
G3 X144.254 Y132.328 I30.046 J-44.217 E.59948
G3 X136.596 Y119.828 I41.129 J-33.795 E.56147
G1 X136.064 Y119.995 E.02129
G1 X135.344 Y120.13 E.02798
G1 X134.411 Y120.179 E.03566
G3 X130.774 Y118.941 I.207 J-6.568 E.14886
G1 X130.185 Y118.446 E.02935
G3 X129.065 Y117.104 I7.774 J-7.625 E.06682
G1 X126.682 Y113.891 E.15272
G3 X120.964 Y120.009 I-43.619 J-35.038 E.32002
G3 X118.936 Y121.711 I-614.584 J-730.199 E.10109
G1 X117.403 Y122.996 E.07636
G3 X105.178 Y130.068 I-32.928 J-42.819 E.54075
G1 X104.061 Y130.508 E.04584
G1 X103.689 Y130.654 E.01527
G1 F12275.421
G1 X103.316 Y130.801 E.01527
; LINE_WIDTH: 0.666041
G1 F10940.684
G1 X103.23 Y130.856 E.00422
; LINE_WIDTH: 0.712086
G1 F10610.979
G1 X103.144 Y130.911 E.00453
; LINE_WIDTH: 0.758131
G1 F10286.301
G1 X103.058 Y130.967 E.00484
; LINE_WIDTH: 0.804176
G1 F9966.685
G1 X102.971 Y131.022 E.00514
; LINE_WIDTH: 0.850221
G1 F9652.096
G1 X102.885 Y131.078 E.00545
; LINE_WIDTH: 0.896266
G1 F9136.474
G1 X102.799 Y131.133 E.00576
; LINE_WIDTH: 0.942311
G1 F8673.147
G1 X102.712 Y131.188 E.00607
; LINE_WIDTH: 0.988356
G1 F8254.545
G1 X102.626 Y131.244 E.00638
G1 X102.711 Y131.271 E.00554
; LINE_WIDTH: 0.942311
G1 F8673.147
G1 X102.796 Y131.298 E.00527
; LINE_WIDTH: 0.896266
G1 F9136.474
G1 X102.881 Y131.325 E.005
; LINE_WIDTH: 0.850221
G1 F9652.096
G1 X102.966 Y131.352 E.00474
; LINE_WIDTH: 0.804176
G1 F10229.399
G1 X103.05 Y131.38 E.00447
; LINE_WIDTH: 0.758131
G1 F10510.399
G1 X103.135 Y131.407 E.0042
; LINE_WIDTH: 0.712086
G1 F10795.228
G1 X103.22 Y131.434 E.00394
; LINE_WIDTH: 0.666041
G1 F11083.844
G1 X103.305 Y131.461 E.00367
; LINE_WIDTH: 0.619996
G1 F11421.795
G1 X103.392 Y131.515 E.00393
G1 X103.22 Y132.072 F36000
G1 F13446.369
G1 X103.486 Y132.245 E.01213
G1 X103.642 Y132.507 E.01165
G1 X103.687 Y132.836 E.01268
G1 X103.687 Y141.425 E.3279
G1 X103.687 Y141.825 E.01527
G1 F13187.228
G1 X103.649 Y142.065 E.00928
G1 F12336.555
G1 X103.494 Y142.337 E.01195
; LINE_WIDTH: 0.664046
G1 F11282.994
G1 X103.398 Y142.43 E.0055
; LINE_WIDTH: 0.708096
G1 F10846.237
G1 X103.301 Y142.523 E.00589
; LINE_WIDTH: 0.752146
G1 F10418.101
G1 X103.204 Y142.616 E.00627
; LINE_WIDTH: 0.797386
G1 F9998.564
G1 X103.133 Y142.658 E.00413
; LINE_WIDTH: 0.842626
G1 F9742.791
G1 X103.062 Y142.701 E.00438
G1 X102.747 Y142.697 E.01662
; LINE_WIDTH: 0.838056
G1 F9798.189
G1 X102.474 Y142.633 E.01468
; LINE_WIDTH: 0.811246
G1 F10136.31
G1 X101.987 Y142.519 E.02532
; LINE_WIDTH: 0.763434
G1 F10801.026
G1 X101.5 Y142.404 E.02377
; LINE_WIDTH: 0.715621
G1 F11559.041
G1 X101.014 Y142.29 E.02221
; LINE_WIDTH: 0.667809
G1 F12431.479
G1 X100.527 Y142.175 E.02065
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X100.134 Y142.103 E.01527
G1 X99.687 Y142.02 E.01735
G3 X99.629 Y143.303 I-7.549 J.303 E.04911
G1 X99.565 Y143.615 E.01214
G1 X100.395 Y143.615 E.03168
; LINE_WIDTH: 0.646796
G1 F12857.984
G1 X100.675 Y143.601 E.01119
; LINE_WIDTH: 0.694611
G1 F11926.85
G1 X101.174 Y143.577 E.02152
; LINE_WIDTH: 0.742426
G1 F11121.468
G1 X101.674 Y143.553 E.02308
; LINE_WIDTH: 0.790241
G1 F10417.977
G1 X102.173 Y143.529 E.02464
; LINE_WIDTH: 0.838056
G1 F9798.189
G1 X102.672 Y143.506 E.0262
; LINE_WIDTH: 0.842626
G1 F9742.791
G3 X103.32 Y143.526 I.243 J2.642 E.03424
; LINE_WIDTH: 0.7981
G1 F10310.778
G1 X103.509 Y143.548 E.00945
; LINE_WIDTH: 0.753574
G1 F10916.726
G1 X103.697 Y143.57 E.0089
; LINE_WIDTH: 0.709048
G1 F11540.007
G1 X103.886 Y143.592 E.00835
; LINE_WIDTH: 0.664522
G1 F12180.561
G1 X104.074 Y143.615 E.0078
; LINE_WIDTH: 0.619996
G1 F13446.369
G1 X104.474 Y143.615 E.01527
G1 X156.235 Y143.615 E1.97618
G1 X156.336 Y143.075 E.02094
G1 X156.415 Y142.647 E.01662
G3 X144.705 Y131.955 I29.319 J-43.865 E.60772
G3 X136.912 Y119.083 I41.142 J-33.707 E.57642
G1 X136.093 Y119.382 E.03328
G3 X132.724 Y119.3 I-1.548 J-5.692 E.13047
G3 X131.14 Y118.484 I2.673 J-7.126 E.0682
G1 X130.562 Y117.998 E.02881
G1 X130.204 Y117.606 E.02024
G3 X129.081 Y116.142 I13.5 J-11.522 E.07049
G1 X126.698 Y112.93 E.15272
G3 X120.566 Y119.579 I-42.608 J-33.138 E.34575
G3 X118.559 Y121.263 I-1032.707 J-1229.051 E.1
G1 X117.027 Y122.547 E.07636
G3 X99.511 Y131.446 I-32.816 J-42.907 E.75429
G2 X99.68 Y132.535 I52.187 J-7.553 E.04209
G1 X99.68 Y132.572 E.00142
G1 X102.766 Y132.001 E.11983
G1 X103.143 Y132.026 E.01443
M204 S250
G1 X102.978 Y132.552 F36000
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.121 Y132.691 E.00631
G3 X103.134 Y133.825 I-49.078 J1.124 E.03589
G1 X103.134 Y141.825 E.25328
G1 X103.078 Y141.973 E.00502
M73 P96 R0
G1 X102.908 Y142.05 E.00591
G1 X102.867 Y142.047 E.00131
G1 X99.134 Y141.355 E.12021
G1 X99.134 Y142.627 E.04026
G3 X98.942 Y143.857 I-4.692 J-.1 E.03953
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.614 J-43.499 E.5106
G3 X137.192 Y118.315 I40.272 J-33.073 E.49169
G1 X136.612 Y118.616 E.02068
G1 X135.913 Y118.859 E.02344
G1 X135.135 Y119.01 E.02509
G1 X134.369 Y119.041 E.02428
G1 X133.603 Y118.958 E.02438
G1 X132.865 Y118.766 E.02415
G1 X132.344 Y118.545 E.01792
G1 X131.508 Y118.068 E.03047
G1 X130.918 Y117.575 E.02435
G1 X130.409 Y117.003 E.02423
G3 X129.087 Y115.222 I1357.011 J-1008.81 E.07022
G1 X126.704 Y112.01 E.12664
G3 X121.665 Y117.772 I-43.077 J-32.588 E.24254
G3 X119.329 Y119.896 I-22.99 J-22.932 E.10001
G1 X116.672 Y122.124 E.10978
G3 X98.942 Y131.036 I-32.296 J-42.157 E.63195
G1 X98.964 Y131.613 E.01827
G1 X99.124 Y132.5 E.02856
G1 X99.132 Y133.236 E.02329
G1 X102.867 Y132.544 E.12026
G1 X102.889 Y132.546 E.00068
; WIPE_START
M204 S10000
G1 X103.121 Y132.691 E-.10415
G1 X103.129 Y133.417 E-.27585
; WIPE_END
G1 E-.02 F1800
G1 X101.032 Y140.756 Z11.8 F36000
G1 X100.449 Y142.795 Z11.8
G1 Z11.4
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.900016
G1 F9096.895
G1 X100.341 Y142.795 E.00611
G1 X100.286 Y142.889 E.00611
G1 X100.341 Y142.983 E.00611
G1 X100.449 Y142.983 E.00611
G1 X100.503 Y142.889 E.00611
; WIPE_START
G1 X100.449 Y142.983 E-.076
G1 X100.341 Y142.983 E-.076
G1 X100.286 Y142.889 E-.076
G1 X100.341 Y142.795 E-.076
G1 X100.449 Y142.795 E-.07599
; WIPE_END
G1 E-.02 F1800
G1 X100.26 Y135.165 Z11.8 F36000
G1 X100.179 Y131.873 Z11.8
G1 Z11.4
G1 E.4 F1800
; LINE_WIDTH: 0.682083
G1 F12157.531
G1 X100.492 Y131.795 E.01364
; LINE_WIDTH: 0.72111
G1 F11466.665
G1 X100.806 Y131.717 E.01446
; LINE_WIDTH: 0.760136
G1 F10850.097
G1 X101.119 Y131.639 E.01528
; LINE_WIDTH: 0.806703
G1 F10195.936
G1 X101.444 Y131.555 E.01687
; LINE_WIDTH: 0.85327
G1 F9616.167
G1 X101.768 Y131.471 E.01788
; LINE_WIDTH: 0.899836
G1 F9098.787
G1 X102.092 Y131.388 E.0189
; LINE_WIDTH: 0.944096
G1 F8656.13
G1 X102.359 Y131.316 E.0164
; LINE_WIDTH: 0.988356
G1 F8254.545
G1 X102.626 Y131.244 E.0172
; WIPE_START
G1 X102.359 Y131.316 E-.10508
G1 X102.092 Y131.388 E-.10508
G1 X101.768 Y131.471 E-.12729
G1 X101.659 Y131.499 E-.04254
; WIPE_END
G1 E-.02 F1800
G1 X104.234 Y138.685 Z11.8 F36000
G1 X105.092 Y141.081 Z11.8
G1 Z11.4
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.624826
G1 F13336.383
G1 X105.957 Y141.945 E.04705
G1 X106.792 Y141.945 E.03215
G1 X105.356 Y140.51 E.07816
G1 X105.356 Y139.675 E.03215
G1 X107.627 Y141.945 E.12363
G1 X108.462 Y141.945 E.03215
G1 X105.356 Y138.839 E.16909
G1 X105.356 Y138.004 E.03215
G1 X109.297 Y141.945 E.21455
G1 X110.132 Y141.945 E.03215
G1 X105.356 Y137.169 E.26001
G1 X105.356 Y136.334 E.03215
G1 X110.967 Y141.945 E.30547
G1 X111.802 Y141.945 E.03215
G1 X105.356 Y135.499 E.35093
G1 X105.356 Y134.664 E.03215
G1 X112.637 Y141.945 E.39639
G1 X113.472 Y141.945 E.03215
G1 X105.356 Y133.829 E.44185
G1 X105.356 Y132.994 E.03215
G1 X114.307 Y141.945 E.48731
G1 X115.142 Y141.945 E.03215
G1 X105.235 Y132.038 E.53933
G1 X105.197 Y131.893 E.00577
G1 X104.874 Y131.359 E.02402
G1 X105.239 Y131.206 E.01521
G1 X115.977 Y141.945 E.58462
G1 X116.813 Y141.945 E.03215
G1 X105.827 Y130.96 E.59806
G1 X106.415 Y130.713 E.02456
G1 X117.648 Y141.945 E.6115
G1 X118.483 Y141.945 E.03215
G1 X106.998 Y130.461 E.62521
G1 X107.572 Y130.2 E.02428
G1 X119.318 Y141.945 E.63942
G1 X120.153 Y141.945 E.03215
G1 X108.137 Y129.929 E.65415
G1 X108.696 Y129.654 E.02401
G1 X120.988 Y141.945 E.66915
G1 X121.823 Y141.945 E.03215
G1 X109.256 Y129.378 E.68414
G1 X109.815 Y129.103 E.02401
G1 X122.658 Y141.945 E.69914
G1 X123.493 Y141.945 E.03215
G1 X110.362 Y128.814 E.71486
G2 X110.905 Y128.522 I-6.5 J-12.724 E.02373
G1 X124.328 Y141.945 E.73077
G1 X125.163 Y141.945 E.03215
G1 X111.44 Y128.223 E.74705
G2 X111.975 Y127.922 I-6.662 J-12.497 E.02361
G1 X125.998 Y141.945 E.7634
G1 X126.834 Y141.945 E.03215
G1 X112.502 Y127.613 E.78021
G2 X113.023 Y127.3 I-5.373 J-9.539 E.02343
G1 X127.669 Y141.945 E.79727
G1 X128.504 Y141.945 E.03215
G1 X113.537 Y126.979 E.81476
G1 X114.051 Y126.658 E.02333
G1 X129.339 Y141.945 E.83225
G1 X130.174 Y141.945 E.03215
G1 X114.564 Y126.336 E.84978
G1 X115.065 Y126.001 E.02318
G1 X131.009 Y141.945 E.86799
G1 X131.844 Y141.945 E.03215
G1 X115.565 Y125.667 E.8862
G2 X116.057 Y125.323 I-6.737 J-10.17 E.02309
G1 X132.679 Y141.945 E.90489
G1 X133.514 Y141.945 E.03215
G1 X116.547 Y124.978 E.92368
G2 X117.031 Y124.627 I-6.342 J-9.26 E.02302
G1 X134.349 Y141.945 E.94278
G1 X135.184 Y141.945 E.03215
G1 X117.51 Y124.271 E.96219
G1 X117.988 Y123.914 E.02297
G1 X136.019 Y141.945 E.9816
G1 X136.854 Y141.945 E.03215
G1 X118.443 Y123.534 E1.00228
G1 X118.898 Y123.153 E.02282
G1 X137.69 Y141.945 E1.02301
G1 X138.525 Y141.945 E.03215
G1 X119.352 Y122.773 E1.04374
G1 X119.806 Y122.392 E.02282
G1 X139.36 Y141.945 E1.06447
G1 X140.195 Y141.945 E.03215
G1 X120.26 Y122.011 E1.0852
G1 X120.715 Y121.63 E.02282
G1 X141.03 Y141.945 E1.10593
G1 X141.865 Y141.945 E.03215
G1 X121.169 Y121.25 E1.12666
G1 X121.623 Y120.869 E.02282
G1 X142.7 Y141.945 E1.14739
G1 X143.535 Y141.945 E.03215
G1 X122.056 Y120.466 E1.16931
G1 X122.484 Y120.059 E.02274
G1 X144.37 Y141.945 E1.19147
G1 X145.205 Y141.945 E.03215
G1 X122.912 Y119.652 E1.21363
G2 X123.335 Y119.24 I-5.283 J-5.859 E.02274
G1 X146.04 Y141.945 E1.23603
G1 X146.875 Y141.945 E.03215
G1 X123.748 Y118.818 E1.25904
G1 X124.16 Y118.395 E.02273
G1 X147.71 Y141.945 E1.28205
G1 X148.546 Y141.945 E.03215
G1 X124.573 Y117.972 E1.30506
G2 X124.975 Y117.54 I-7.477 J-7.356 E.02275
G1 X149.381 Y141.945 E1.32861
G1 X150.216 Y141.945 E.03215
G1 X125.375 Y117.105 E1.35229
G1 X125.767 Y116.662 E.02277
G1 X151.051 Y141.945 E1.37642
G1 X151.886 Y141.945 E.03215
G1 X126.159 Y116.218 E1.40055
G2 X126.542 Y115.766 I-9.278 J-8.255 E.02281
G1 X127.247 Y116.471 E.03837
G1 X128.617 Y118.318 E.08854
G1 X129.233 Y119.028 E.03616
G1 X129.434 Y119.229 E.01094
G1 X130.099 Y119.789 E.03348
G1 X130.33 Y119.952 E.01088
G1 X131.131 Y120.437 E.03603
G1 X131.272 Y120.496 E.00589
G1 X142.094 Y131.319 E.58917
G3 X140.059 Y128.448 I39.465 J-30.142 E.1355
G1 X132.605 Y120.995 E.40576
G2 X133.644 Y121.198 I1.536 J-5.079 E.04082
G1 X138.646 Y126.2 E.27227
G3 X137.553 Y124.272 I66.717 J-39.091 E.0853
G1 X134.539 Y121.258 E.16407
G2 X135.334 Y121.218 I.127 J-5.407 E.03068
G1 X137.262 Y123.146 E.10493
; CHANGE_LAYER
; Z_HEIGHT: 11.56
; LAYER_HEIGHT: 0.160001
; WIPE_START
G1 F13336.383
G1 X136.555 Y122.439 E-.38
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
G1 X103.05 Y132.586
G1 Z11.56
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.519996
G1 F3600
M204 S5000
G1 X103.129 Y132.71 E.00465
G3 X103.136 Y133.827 I-94.336 J1.105 E.03537
G1 X103.136 Y141.827 E.25328
G1 X103.08 Y141.976 E.00502
G1 X102.91 Y142.053 E.00591
G1 X102.869 Y142.049 E.00131
G1 X99.136 Y141.358 E.12021
G1 X99.136 Y142.626 E.04014
G3 X98.942 Y143.857 I-4.66 J-.101 E.03959
G1 X98.936 Y144.167 E.00983
G1 X156.695 Y144.167 E1.82865
G1 X156.879 Y143.176 E.03191
M73 P97 R0
G1 X157.025 Y142.39 E.02533
G3 X145.128 Y131.598 I28.611 J-43.495 E.51061
G3 X137.192 Y118.315 I40.271 J-33.072 E.49169
G1 X136.612 Y118.616 E.02069
G1 X135.913 Y118.859 E.02344
G1 X135.137 Y119.01 E.025
G1 X134.369 Y119.041 E.02435
G1 X133.607 Y118.959 E.02427
G1 X132.865 Y118.766 E.02427
G1 X132.341 Y118.544 E.01801
G1 X131.508 Y118.068 E.03037
G1 X130.918 Y117.575 E.02436
G1 X130.408 Y117.003 E.02426
G3 X129.087 Y115.222 I993.901 J-739.237 E.07019
G1 X126.704 Y112.01 E.12664
G3 X121.199 Y118.227 I-44.205 J-33.597 E.26315
G3 X119.325 Y119.9 I-15.808 J-15.821 E.07957
G1 X116.671 Y122.124 E.10962
G3 X98.936 Y131.038 I-32.252 J-42.069 E.63217
G1 X98.947 Y131.519 E.01524
G3 X99.136 Y133.233 I-6.222 J1.551 E.05476
G1 X102.869 Y132.542 E.12021
G1 X102.963 Y132.565 E.00305
; WIPE_START
M204 S10000
G1 X103.129 Y132.71 E-.08402
G1 X103.134 Y133.489 E-.29598
; WIPE_END
G1 E-.02 F1800
G1 X110.653 Y134.798 Z11.96 F36000
G1 X156.671 Y142.809 Z11.96
G1 Z11.56
G1 E.4 F1800
; FEATURE: Top surface
; LINE_WIDTH: 0.62
G1 F9000
M204 S2000
G1 X155.584 Y143.895 E.05867
G1 X155.387 Y144.093
G1 X154.558 Y144.093
G1 X154.756 Y143.895
G1 X156.375 Y142.276 E.08744
G1 X156.573 Y142.078
G1 X156.082 Y141.741
G1 X155.885 Y141.938
G1 X153.927 Y143.895 E.10567
G1 X153.73 Y144.093
G1 X152.902 Y144.093
G1 X153.099 Y143.895
G1 X155.394 Y141.601 E.1239
G1 X155.591 Y141.403
G1 X155.105 Y141.061
G1 X154.908 Y141.258
G1 X152.271 Y143.895 E.14239
G1 X152.074 Y144.093
G1 X151.245 Y144.093
G1 X151.443 Y143.895
G1 X154.43 Y140.908 E.16132
G1 X154.628 Y140.71
G1 X154.156 Y140.354
G1 X153.959 Y140.551
G1 X150.614 Y143.895 E.18057
G1 X150.417 Y144.093
G1 X149.589 Y144.093
G1 X149.786 Y143.895
G1 X153.49 Y140.191 E.2
G1 X153.688 Y139.994
G1 X153.224 Y139.63
G1 X153.026 Y139.827
G1 X148.958 Y143.895 E.21967
G1 X148.761 Y144.093
G1 X147.932 Y144.093
G1 X148.13 Y143.895
G1 X152.567 Y139.458 E.2396
G1 X152.765 Y139.26
G1 X152.312 Y138.884
G1 X152.115 Y139.082
G1 X147.301 Y143.895 E.2599
G1 X147.104 Y144.093
G1 X146.276 Y144.093
G1 X146.473 Y143.895
G1 X151.665 Y138.703 E.28034
G1 X151.863 Y138.506
G1 X151.418 Y138.122
G1 X151.221 Y138.319
G1 X145.645 Y143.895 E.30107
G1 X145.448 Y144.093
G1 X144.619 Y144.093
G1 X144.817 Y143.895
G1 X150.781 Y137.931 E.32206
G1 X150.979 Y137.733
G1 X150.543 Y137.341
G1 X150.346 Y137.538
G1 X143.988 Y143.895 E.34324
G1 X143.791 Y144.093
G1 X142.963 Y144.093
G1 X143.16 Y143.895
G1 X149.916 Y137.139 E.36477
G1 X150.113 Y136.942
G1 X149.686 Y136.541
G1 X149.489 Y136.738
G1 X142.332 Y143.895 E.38644
G1 X142.135 Y144.093
G1 X141.306 Y144.093
G1 X141.504 Y143.895
G1 X149.072 Y136.327 E.40863
G1 X149.269 Y136.13
G1 X148.852 Y135.719
G1 X148.654 Y135.916
G1 X140.675 Y143.895 E.43081
G1 X140.478 Y144.093
G1 X139.65 Y144.093
G1 X139.847 Y143.895
G1 X148.239 Y135.503 E.45312
G1 X148.437 Y135.306
G1 X148.035 Y134.879
G1 X147.838 Y135.076
G1 X139.019 Y143.895 E.47618
G1 X138.822 Y144.093
G1 X137.993 Y144.093
G1 X138.191 Y143.895
G1 X147.437 Y134.649 E.49923
G1 X147.634 Y134.452
G1 X147.233 Y134.025
G1 X147.036 Y134.222
G1 X137.362 Y143.895 E.52229
G1 X137.165 Y144.093
G1 X136.337 Y144.093
G1 X136.534 Y143.895
G1 X146.638 Y133.792 E.54553
G1 X146.835 Y133.594
G1 X146.448 Y133.153
G1 X146.251 Y133.351
G1 X135.706 Y143.895 E.56934
G1 X135.509 Y144.093
G1 X134.68 Y144.093
G1 X134.878 Y143.895
G1 X145.864 Y132.909 E.5932
G1 X146.061 Y132.712
G1 X145.685 Y132.259
G1 X145.488 Y132.457
G1 X134.049 Y143.895 E.61761
G1 X133.852 Y144.093
G1 X133.024 Y144.093
G1 X133.221 Y143.895
G1 X145.112 Y132.004 E.64203
G1 X145.309 Y131.807
G1 X144.939 Y131.349
G1 X144.742 Y131.547
G1 X132.393 Y143.895 E.66675
G1 X132.196 Y144.093
G1 X131.367 Y144.093
G1 X131.565 Y143.895
G1 X144.377 Y131.083 E.69179
G1 X144.574 Y130.886
G1 X144.21 Y130.422
G1 X144.013 Y130.619
G1 X130.736 Y143.895 E.71682
G1 X130.539 Y144.093
G1 X129.711 Y144.093
G1 X129.908 Y143.895
G1 X143.654 Y130.149 E.7422
G1 X143.852 Y129.952
G1 X143.499 Y129.476
G1 X143.302 Y129.674
G1 X129.08 Y143.895 E.76788
G1 X128.883 Y144.093
G1 X128.054 Y144.093
G1 X128.252 Y143.895
G1 X142.949 Y129.198 E.79355
G1 X143.146 Y129.001
G1 X142.798 Y128.521
G1 X142.6 Y128.718
G1 X127.424 Y143.895 E.81946
G1 X127.226 Y144.093
G1 X126.398 Y144.093
G1 X126.595 Y143.895
G1 X142.261 Y128.23 E.84583
G1 X142.458 Y128.033
G1 X142.121 Y127.542
G1 X141.923 Y127.739
G1 X125.767 Y143.895 E.87234
G1 X125.57 Y144.093
G1 X124.741 Y144.093
G1 X124.939 Y143.895
G1 X141.597 Y127.237 E.89943
G1 X141.794 Y127.04
G1 X141.467 Y126.538
G1 X141.27 Y126.736
G1 X124.111 Y143.895 E.92651
G1 X123.913 Y144.093
G1 X123.085 Y144.093
G1 X123.282 Y143.895
G1 X140.944 Y126.234 E.95362
G1 X141.141 Y126.036
G1 X140.829 Y125.52
G1 X140.632 Y125.717
G1 X122.454 Y143.895 E.9815
G1 X122.257 Y144.093
G1 X121.428 Y144.093
G1 X121.626 Y143.895
G1 X140.32 Y125.201 E1.00938
G1 X140.517 Y125.004
M73 P98 R0
G1 X140.211 Y124.482
G1 X140.013 Y124.679
G1 X120.798 Y143.895 E1.03754
G1 X120.6 Y144.093
G1 X119.772 Y144.093
G1 X119.969 Y143.895
G1 X139.715 Y124.149 E1.06615
G1 X139.912 Y123.952
G1 X139.614 Y123.422
G1 X139.417 Y123.619
G1 X119.141 Y143.895 E1.09477
G1 X118.944 Y144.093
G1 X118.115 Y144.093
G1 X118.313 Y143.895
G1 X139.13 Y123.078 E1.12401
G1 X139.328 Y122.881
G1 X139.044 Y122.335
G1 X138.847 Y122.533
G1 X117.485 Y143.895 E1.15345
G1 X117.287 Y144.093
G1 X116.459 Y144.093
G1 X116.656 Y143.895
G1 X138.564 Y121.987 E1.18289
G1 X138.761 Y121.79
G1 X138.489 Y121.235
G1 X138.291 Y121.432
G1 X115.828 Y143.895 E1.21288
G1 X115.631 Y144.093
G1 X114.802 Y144.093
G1 X115 Y143.895
G1 X138.024 Y120.871 E1.24316
G1 X138.221 Y120.674
G1 X137.955 Y120.111
G1 X137.758 Y120.309
G1 X114.172 Y143.895 E1.27353
G1 X113.974 Y144.093
G1 X113.146 Y144.093
G1 X113.343 Y143.895
G1 X137.507 Y119.732 E1.30467
G1 X137.704 Y119.535
G1 X137.453 Y118.958
G1 X137.255 Y119.155
G1 X112.515 Y143.895 E1.33582
G1 X112.318 Y144.093
G1 X111.49 Y144.093
G1 X111.687 Y143.895
G1 X136.722 Y118.861 E1.35172
G1 X136.919 Y118.663
G1 X135.743 Y119.011
G1 X135.546 Y119.208
G1 X110.859 Y143.895 E1.33294
G1 X110.661 Y144.093
G1 X109.833 Y144.093
G1 X110.03 Y143.895
G1 X134.622 Y119.304 E1.32779
G1 X134.819 Y119.107
G1 X134.038 Y119.059
G1 X133.841 Y119.257
G1 X109.202 Y143.895 E1.33033
G1 X109.005 Y144.093
G1 X108.177 Y144.093
G1 X108.374 Y143.895
G1 X133.148 Y119.121 E1.33764
G1 X133.345 Y118.924
G1 X132.724 Y118.717
G1 X132.527 Y118.914
G1 X107.546 Y143.895 E1.34882
G1 X107.348 Y144.093
G1 X106.52 Y144.093
G1 X106.717 Y143.895
G1 X131.952 Y118.661 E1.36251
G1 X132.149 Y118.463
G1 X131.637 Y118.147
G1 X131.44 Y118.344
G1 X105.889 Y143.895 E1.37959
G1 X105.692 Y144.093
G1 X104.864 Y144.093
G1 X105.061 Y143.895
G1 X130.977 Y117.979 E1.3993
G1 X131.174 Y117.782
G1 X130.752 Y117.376
G1 X130.554 Y117.573
G1 X104.233 Y143.895 E1.42122
G1 X104.035 Y144.093
G1 X103.207 Y144.093
G1 X103.404 Y143.895
G1 X130.167 Y117.132 E1.44502
G1 X130.364 Y116.935
G1 X130.012 Y116.46
G1 X129.814 Y116.657
G1 X102.576 Y143.895 E1.47069
G1 X102.379 Y144.093
G1 X101.551 Y144.093
G1 X101.748 Y143.895
G1 X129.461 Y116.182 E1.49636
G1 X129.659 Y115.984
G1 X129.306 Y115.509
G1 X129.109 Y115.706
G1 X103.408 Y141.407 E1.38766
G1 X103.211 Y141.604
G1 X103.211 Y140.776
G1 X103.408 Y140.579
G1 X128.756 Y115.231 E1.36861
G1 X128.953 Y115.034
G1 X128.6 Y114.558
G1 X128.403 Y114.756
G1 X103.408 Y139.75 E1.34956
G1 X103.211 Y139.948
G1 X103.211 Y139.119
G1 X103.408 Y138.922
G1 X128.05 Y114.28 E1.33051
G1 X128.247 Y114.083
G1 X127.894 Y113.608
G1 X127.697 Y113.805
G1 X103.408 Y138.094 E1.31145
G1 X103.211 Y138.291
M73 P99 R0
G1 X103.211 Y137.463
G1 X103.408 Y137.266
G1 X127.344 Y113.329 E1.2924
G1 X127.541 Y113.132
G1 X127.189 Y112.657
G1 X126.991 Y112.854
G1 X103.408 Y136.437 E1.27335
G1 X103.211 Y136.635
G1 X103.211 Y135.806
G1 X103.408 Y135.609
G1 X116.209 Y122.808 E.69119
; WIPE_START
M204 S10000
G1 X115.502 Y123.515 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X113.482 Y124.707 Z11.96 F36000
G1 Z11.56
G1 E.4 F1800
G1 F9000
M204 S2000
G1 X103.408 Y134.781 E.54395
G1 X103.211 Y134.978
G1 X103.211 Y134.15
G1 X103.408 Y133.953
G1 X111.311 Y126.05 E.4267
; WIPE_START
M204 S10000
G1 X110.604 Y126.757 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X109.455 Y127.078 Z11.96 F36000
G1 Z11.56
G1 E.4 F1800
G1 F9000
M204 S2000
G1 X103.408 Y133.124 E.32648
G1 X103.211 Y133.322
G1 X103.076 Y132.628
G1 X103.273 Y132.431
G1 X107.76 Y127.945 E.24225
G1 X107.957 Y127.747
G1 X106.4 Y128.476
G1 X106.203 Y128.673
G1 X102.552 Y132.324 E.19712
G1 X102.355 Y132.521
G1 X101.338 Y132.709
G1 X101.536 Y132.512
G1 X104.746 Y129.301 E.17336
G1 X104.944 Y129.104
G1 X103.571 Y129.649
G1 X103.373 Y129.846
G1 X100.519 Y132.7 E.15411
G1 X100.322 Y132.898
G1 X99.305 Y133.086
G1 X99.503 Y132.889
G1 X102.057 Y130.334 E.1379
G1 X102.254 Y130.137
G1 X101.006 Y130.557
G1 X100.808 Y130.754
G1 X99.358 Y132.205 E.07833
; WIPE_START
M204 S10000
G1 X100.065 Y131.498 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X101.785 Y138.934 Z11.96 F36000
G1 X102.555 Y142.26 Z11.96
G1 Z11.56
G1 E.4 F1800
G1 F9000
M204 S2000
G1 X100.92 Y143.895 E.08828
G1 X100.722 Y144.093
G1 X99.894 Y144.093
G1 X100.091 Y143.895
G1 X101.854 Y142.132 E.09519
G1 X102.052 Y141.935
G1 X101.351 Y141.807
G1 X101.154 Y142.004
G1 X99.263 Y143.895 E.10211
G1 X99.066 Y144.093
G1 X99.183 Y143.147
G1 X99.381 Y142.95
G1 X100.454 Y141.876 E.05796
; WIPE_START
M204 S10000
G1 X99.747 Y142.583 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X104.848 Y136.906 Z11.96 F36000
G1 X126.769 Y112.506 Z11.96
G1 Z11.56
G1 E.4 F1800
; FEATURE: Gap infill
; LINE_WIDTH: 0.219562
G1 F3000
G1 X126.478 Y112.834 E.0053
; LINE_WIDTH: 0.173582
G1 X126.186 Y113.162 E.00398
; LINE_WIDTH: 0.127602
G1 X125.895 Y113.49 E.00267
; WIPE_START
G1 X126.186 Y113.162 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X120.801 Y118.571 Z11.96 F36000
G1 X118.083 Y121.3 Z11.96
G1 Z11.56
G1 E.4 F1800
; LINE_WIDTH: 0.128043
G1 F3000
G1 X117.681 Y121.669 E.00333
; LINE_WIDTH: 0.174884
G1 X117.279 Y122.037 E.005
; LINE_WIDTH: 0.221986
G1 X116.872 Y122.409 E.00675
; LINE_WIDTH: 0.267691
G1 X116.587 Y122.655 E.00572
; LINE_WIDTH: 0.311371
G1 X116.303 Y122.901 E.00679
; WIPE_START
G1 X116.587 Y122.655 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X114.61 Y123.947 Z11.96 F36000
G1 Z11.56
G1 E.4 F1800
; LINE_WIDTH: 0.127259
G1 F3000
G1 X114.411 Y124.113 E.00157
; LINE_WIDTH: 0.172533
G1 X114.212 Y124.28 E.00234
; LINE_WIDTH: 0.217806
G1 X114.012 Y124.446 E.00311
; LINE_WIDTH: 0.263555
G1 X113.808 Y124.616 E.00396
; LINE_WIDTH: 0.303839
G1 X113.577 Y124.801 E.00521
; WIPE_START
G1 X113.808 Y124.616 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X112.239 Y125.492 Z11.96 F36000
G1 Z11.56
G1 E.4 F1800
; LINE_WIDTH: 0.121613
G1 F3000
G1 X112.111 Y125.594 E.00093
; LINE_WIDTH: 0.155622
G1 X111.982 Y125.696 E.0013
; LINE_WIDTH: 0.189937
G1 X111.852 Y125.801 E.00169
; LINE_WIDTH: 0.225375
G1 X111.703 Y125.915 E.00234
; LINE_WIDTH: 0.261436
G1 X111.554 Y126.03 E.00278
; LINE_WIDTH: 0.297496
G1 X111.406 Y126.144 E.00322
; WIPE_START
G1 X111.554 Y126.03 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X110.258 Y126.645 Z11.96 F36000
G1 Z11.56
G1 E.4 F1800
; LINE_WIDTH: 0.125183
G1 F3000
G1 X110.117 Y126.751 E.00105
; LINE_WIDTH: 0.166305
G1 X109.975 Y126.856 E.00152
; LINE_WIDTH: 0.207427
G1 X109.833 Y126.962 E.00199
; LINE_WIDTH: 0.248549
G1 X109.692 Y127.067 E.00247
; LINE_WIDTH: 0.289671
G1 X109.55 Y127.173 E.00294
; WIPE_START
G1 X109.692 Y127.067 E-.38
; WIPE_END
G1 E-.02 F1800
G1 X108.488 Y127.589 Z11.96 F36000
G1 Z11.56
G1 E.4 F1800
; LINE_WIDTH: 0.127553
G1 F3000
G1 X108.298 Y127.724 E.00141
; close powerlost recovery
M1003 S0
; WIPE_START
G1 F3000
G1 X108.488 Y127.589 E-.38
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

