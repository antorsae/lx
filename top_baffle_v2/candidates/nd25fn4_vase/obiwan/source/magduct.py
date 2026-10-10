"""wall between each relocated LM magnet cavity (40/140 deg) and the nearest released duct lumen, plus lumen clearance at the stations"""
import sys, json, math, numpy as np, trimesh
m = trimesh.load(sys.argv[1]); rc = json.load(open(sys.argv[2]))["duct_collision_contract"]["regions"]; mir = len(sys.argv) > 3 and sys.argv[3] == "mirror"
parts = m.split(only_watertight=False); main = max(parts, key=lambda p: abs(p.volume)); cav = [p for p in parts if p.volume < 0]
LM = np.array([0, 200.981])
for c in cav:
    a = math.degrees(math.atan2(c.centroid[1] - LM[1], c.centroid[0] - LM[0]))
    best = None
    for r in rc:
        if r["name"] not in ("um_route_lumen", "t_route_lumen"): continue
        P = np.array(r["points_xyz_mm"], float)
        if mir: P[:, 0] *= -1
        d = np.min(np.linalg.norm(c.vertices[:, None, :] - P[None, :, :], axis=2), axis=1).min() - r["radius_mm"]
        if best is None or d < best[0]: best = (d, r["name"])
    print(f"magnet cavity at {a:.0f} deg: nearest lumen {best[1]}, wall {best[0]:.2f} mm")
for r in rc:
    if r["name"] not in ("um_route_lumen", "t_route_lumen"): continue
    P = np.array(r["points_xyz_mm"], float)
    if mir: P[:, 0] *= -1
    k = (P[:, 1] > 255) & (P[:, 1] < 285); Q = P[k]
    ring = []
    for i in range(1, len(Q) - 1):
        ax = Q[i + 1] - Q[i - 1]; ax /= np.linalg.norm(ax); u = np.cross(ax, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(ax, u)
        ring += [Q[i] + 0.9 * r["radius_mm"] * (math.cos(t) * u + math.sin(t) * v) for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    solid = main.contains(np.array(ring)).sum() if ring else 0
    print(f"{r['name']}: points at 0.9 r inside the lumen between y 255 and 285 that are solid: {solid} / {len(ring)}")
