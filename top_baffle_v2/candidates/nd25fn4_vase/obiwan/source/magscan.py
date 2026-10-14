"""where can the LM wing magnet sit on the ring without meeting the released duct lumens? rotate the released 64 deg cavity (+ its
116 deg mirror) along the ring and measure: wall to the nearest lumen, wall to any released surface, distance to the L22 pilots"""
import json, math, numpy as np, trimesh
rel = trimesh.load("released_lm_keyed_top_floor_design.stl"); parts = rel.split(only_watertight=False)
main = max(parts, key=lambda p: abs(p.volume)); LM = np.array([0, 200.981])
cav = {round(math.degrees(math.atan2(p.centroid[1] - LM[1], p.centroid[0] - LM[0]))): p for p in parts if p.volume < 0}
print("released cavities at", sorted(cav))
rc = json.load(open("contract.json"))["duct_collision_contract"]["regions"]
lum = [(np.array(r["points_xyz_mm"]), r["radius_mm"], r["name"]) for r in rc if "route" in r["name"] or "lane" in r["name"]]
pq = trimesh.proximity.ProximityQuery(main)
for a in list(range(0, 46, 3)):
    row = []
    for a0, a1 in ((64, a), (116, 180 - a)):
        q = cav[a0].copy(); q.apply_transform(trimesh.transformations.rotation_matrix(math.radians(a1 - a0), [0, 0, 1], [LM[0], LM[1], 0]))
        V = q.vertices[::2]
        wl = min(float(np.min(np.linalg.norm(V[:, None, :] - P[None], axis=2)) - r) for P, r, n in lum)
        ws = float(np.abs(pq.signed_distance(V)).min()); inside = main.contains(V).mean()
        row.append(f"{a1:3.0f} deg: lumen wall {wl:5.2f}  surface wall {ws:4.2f}  inside {inside:.2f}")
    print(" | ".join(row))
