"""V11 caps vs the placed vendor ND25FN-4 tweeters over the bayonet twist: overlap, pad gap, tab-tip clearance."""
import sys, json, math, numpy as np, trimesh, manifold3d as m3
d, V, drv_glb = sys.argv[1], sys.argv[2], sys.argv[3]; C = json.load(open(f"{d}/{V}_build.json"))
to_m = lambda q: m3.Manifold(m3.Mesh(vert_properties=np.asarray(q.vertices, np.float32), tri_verts=np.asarray(q.faces, np.uint32)))
sc = trimesh.load(drv_glb)
for capn, twn, c in ((f"{V}_cap_frontT", "tweeter_front_T", C["T"]), (f"{V}_cap_reart", "tweeter_rear_t", C["t"])):
    cap0 = trimesh.load(f"{d}/{capn}.stl"); tw = sc.geometry[twn]; TW = to_m(tw)
    for ang in (0, 8, 16):
        cap = cap0.copy(); cap.apply_transform(trimesh.transformations.rotation_matrix(math.radians(ang), [0, 0, 1], [c[0], c[1], 0]))
        ov = (TW ^ to_m(cap)).volume()
        pq = trimesh.proximity.ProximityQuery(cap)
        v = tw.vertices; r = np.hypot(v[:, 0] - c[0], v[:, 1] - c[1])
        tips = v[r > 18.5]; body = v[r < 11]
        dt = np.abs(pq.signed_distance(tips[::3])); db = np.abs(pq.signed_distance(body[::5]))
        print(f"{capn} twist {ang:2d}: overlap {ov:.3f} mm3 | nearest tab/rim point to cap {dt.min():.2f} mm | tweeter rear face to boss {db.min():.2f} mm")
