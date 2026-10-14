"""map the released H2C print STLs back into the design frame (inverse of each print.json source_to_stl_matrix)"""
import sys, json, numpy as np, trimesh
d, R = sys.argv[1], sys.argv[2]
for nm, out in (("h2c_obiwan_core_lm_carrier_no_floor_stand", "released_lm_no_floor_design.stl"), ("h2c_obiwan_core_lm_carrier_floor_stand", "released_lm_floor_design.stl"), ("floor_obiwan_optional_lm_keyed_2_of_2_top", "released_lm_keyed_top_floor_design.stl"), ("floor_obiwan_optional_lm_keyed_1_of_2_bottom", "released_lm_keyed_bottom_floor_design.stl"),
                ("h2c_obiwan_wing_flat_left", "released_h2c_obiwan_wing_flat_left_design.stl"), ("h2c_obiwan_wing_flat_right", "released_h2c_obiwan_wing_flat_right_design.stl")):
    P = json.load(open(f"{R}/{nm}.print.json")); m = trimesh.load(f"{R}/{nm}.stl"); m.apply_transform(np.linalg.inv(np.array(P["source_to_stl_matrix"]))); m.export(f"{d}/{out}")
    print(out, np.round(m.bounds, 2).tolist(), "watertight", m.is_watertight)
