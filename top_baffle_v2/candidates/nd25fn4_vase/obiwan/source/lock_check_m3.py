"""twist-lock check by exact solid intersection (manifold3d): overlap volume of each cap with the body at 0..26 deg of twist.
Replaces the point-in-mesh version (lock_check_v5.py), which needed 30-58 GiB per run without embree."""
import sys, json, math, numpy as np, trimesh, manifold3d as m3
d = sys.argv[1]; V = sys.argv[2]; C = json.load(open(f"{d}/{V}_build.json"))
to_m = lambda q: m3.Manifold(m3.Mesh(vert_properties=np.asarray(q.vertices, np.float32), tri_verts=np.asarray(q.faces, np.uint32)))
body = trimesh.load(f"{d}/{V}.stl"); body = max(body.split(only_watertight=False), key=lambda p: abs(p.volume)); B = to_m(body)
for nm, c in ((f"{V}_cap_frontT", C["T"]), (f"{V}_cap_reart", C["t"])):
    cap = trimesh.load(f"{d}/{nm}.stl")
    for ang in (0, 8, 16, 22, 26):
        q = cap.copy(); q.apply_transform(trimesh.transformations.rotation_matrix(math.radians(ang), [0, 0, 1], [c[0], c[1], 0]))
        print(f"{nm}: twist {ang:+3d} deg -> overlap with body {(B ^ to_m(q)).volume():.3f} mm3", flush=True)
