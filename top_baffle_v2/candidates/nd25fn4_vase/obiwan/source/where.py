import sys, json, math, numpy as np, trimesh
for f in sys.argv[1:]:
    m = max(trimesh.load(f).split(only_watertight=False), key=lambda p: abs(p.volume))
    rc = json.load(open("/home/antor/temp/claude-obiwan-concept/pkg/rel/obiwan_optional_lm_keyed_2_of_2_top.support_blocker.json"))["duct_collision_contract"]["regions"]
    for r in rc:
        if r["name"] not in ("um_route_lumen", "t_route_lumen"): continue
        P = np.array(r["points_xyz_mm"]); Q = P[(P[:, 1] > 180) & (P[:, 1] < 277.5)]; pts = []
        for i in range(1, len(Q) - 1):
            ax = Q[i + 1] - Q[i - 1]; ax /= np.linalg.norm(ax); u = np.cross(ax, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(ax, u)
            pts += [Q[i] + 0.9 * r["radius_mm"] * (math.cos(t) * u + math.sin(t) * v) for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
        pts = np.array(pts); s = m.contains(pts)
        print(f.split("/")[-1][:40], r["name"], "solid at y:", np.round(np.unique(np.round(pts[s][:, 1])), 0).tolist()[:20], "z:", np.round(pts[s][:, 2].min(), 1) if s.any() else "-", np.round(pts[s][:, 2].max(), 1) if s.any() else "-")
