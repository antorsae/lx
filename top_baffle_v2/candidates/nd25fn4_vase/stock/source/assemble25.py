"""vase25: place the rim-centred vendor ND25FN-4 on the MEASURED land of each cavity (vase24 assumed a land 3.59 mm too far back),
tab midpoint clocked to the cable lane mouth (equal runs to both tabs); export placement transforms. Interference vs the (unkeyed) body."""
import sys, json, math, ast, numpy as np, trimesh
d = sys.argv[1]; C = json.load(open(f"{d}/c7.json")); LAND = json.load(open(f"{d}/land_measured.json"))
drv = trimesh.load(f"{d}/nd25fn4_vendor_rimcentred.stl"); v = drv.vertices; r = np.hypot(v[:, 1], v[:, 2])
fr = (r > 15.9) & (r < 21.5) & (v[:, 0] < 12); XF = float(np.percentile(v[fr, 0], 0.5))          # forward flange ring outside the throat: bears on the land
TERM_MID = 133.65                                                                                 # midpoint of the two tabs (89.7 / 177.6 deg) in the rim-centred model
def place(c, front, gap, zl):
    ex = np.array([0, 0, -1.0 if front else 1.0]); best = None
    for phi in np.radians(np.arange(0, 360, 0.25)):
        ey = np.array([math.cos(phi), math.sin(phi), 0]); ez = np.cross(ex, ey); M = np.c_[ex, ey, ez]
        b = M @ np.array([0, math.cos(math.radians(TERM_MID)), math.sin(math.radians(TERM_MID))]); a = math.degrees(math.atan2(b[1], b[0])) % 360
        e = abs((a - gap + 180) % 360 - 180)
        if best is None or e < best[0]: best = (e, M)
    M = best[1]; T4 = np.eye(4); T4[:3, :3] = M; T4[:3, 3] = np.array([c[0], c[1], zl]) - M @ np.array([XF, 0, 0])
    m = drv.copy(); m.apply_transform(T4); return m, T4
LANES = C["lanes"] if isinstance(C["lanes"], dict) else ast.literal_eval(C["lanes"])
MOUTH = {k: math.degrees(math.atan2(LANES[k][-1][1] - C[k][1], LANES[k][-1][0] - C[k][0])) for k in ("T", "t")}   # owner: equal cable runs -> tab midpoint faces the cable lane mouth
print("cable lane mouths (deg around each tweeter):", {k: round(v, 1) for k, v in MOUTH.items()})
T_, T4T = place(C["T"], True, MOUTH["T"], LAND["T"]); t_, T4t = place(C["t"], False, MOUTH["t"], LAND["t"])
print(f"XF {XF:.2f}; front T z {T_.bounds[0,2]:.1f}..{T_.bounds[1,2]:.1f} (land {LAND['T']}), rear t z {t_.bounds[0,2]:.1f}..{t_.bounds[1,2]:.1f} (land {LAND['t']})")
print(f"dome apex: T z {T_.bounds[1,2]:.2f} = {C['TH'] + 5.4 - T_.bounds[1,2]:.2f} mm behind the MU10 flange-top plane (vase24 intent 7.9 with the tilted model)")
trimesh.util.concatenate([T_, t_]).export(f"{d}/drivers_nd25fn4_placed.stl")
json.dump({"T": T4T.tolist(), "t": T4t.tolist(), "XF": XF, "TERM_AZ": [89.7, 177.6], "land": LAND}, open(f"{d}/tweeter_placement.json", "w"))
body = trimesh.load(f"{d}/_body_with_lanes_front.stl"); body = max(body.split(only_watertight=False), key=lambda p: abs(p.volume))
for nm, m in (("front T", T_), ("rear t", t_)):
    ins = np.concatenate([body.contains(m.vertices[i:i + 2000]) for i in range(0, len(m.vertices), 2000)])
    pen = trimesh.proximity.ProximityQuery(body).signed_distance(m.vertices[ins]) if ins.any() else np.zeros(1)
    print(f"  {nm}: {ins.sum()} / {len(ins)} vertices inside the printed body, max penetration {pen.max():.2f} mm")
