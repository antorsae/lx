import json, math, numpy as np, ast
C = json.load(open("c7.json")); P = json.load(open("tweeter_placement.json"))
lanes = C["lanes"] if isinstance(C["lanes"], dict) else ast.literal_eval(C["lanes"])
for key in ("T", "t"):
    c = np.array(C[key]); path = np.array(lanes[key], float); end = path[-1]
    T4 = np.array(P[key]); W = lambda p: T4[:3, :3] @ np.asarray(p, float) + T4[:3, 3]
    am = math.degrees(math.atan2(end[1] - c[1], end[0] - c[0]))
    print(f"{key}: cable lane ends at ({end[0]:.1f}, {end[1]:.1f}, z {end[2]:.1f}), {am:.1f} deg around the tweeter (GAP {C['GAP'][key]})")
    for name, az, x0, x1, rt in (("A narrow", 89.7, 22.0, 31.43, 19.83), ("B wide", 177.6, 22.0, 29.86, 19.34)):
        p = W([x0 + 0.55 * (x1 - x0), (rt + 0.5) * math.cos(math.radians(az)), (rt + 0.5) * math.sin(math.radians(az))])
        ap = math.degrees(math.atan2(p[1] - c[1], p[0] - c[0]))
        print(f"   tab {name}: solder point z {p[2]:.1f} at {ap:.1f} deg; straight run {np.linalg.norm(p - end):.1f} mm")
