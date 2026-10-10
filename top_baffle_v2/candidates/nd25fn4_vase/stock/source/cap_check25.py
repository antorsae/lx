"""vase25 caps vs placed tweeters: overlap locked / mid / entry; tab-tip clearance; foam-ring and felt-pad contact."""
import sys, json, math, numpy as np, trimesh, manifold3d as m3
d = sys.argv[1]; C = json.load(open(f"{d}/c7.json"))
to_m = lambda q: m3.Manifold(m3.Mesh(vert_properties=np.asarray(q.vertices, np.float32), tri_verts=np.asarray(q.faces, np.uint32)))
drv = trimesh.load(f"{d}/drivers_nd25fn4_placed.stl"); parts = drv.split(only_watertight=False)
for key, capn in (("T", "cap_front_tweeter_REAR_FACE"), ("t", "cap_rear_tweeter_FRONT_FACE")):
    c = np.array(C[key]); tw = min(parts, key=lambda p: np.linalg.norm(p.centroid[:2] - c)); TW = to_m(tw)
    cap0 = trimesh.load(f"{d}/{capn}.stl"); foam = trimesh.load(f"{d}/foam_ring_{'frontT' if key == 'T' else 'reart'}.stl")
    for lab, ang in (("locked", 0.0), ("mid-turn", -12.5), ("entry", -25.0)):
        q = cap0.copy(); q.apply_transform(trimesh.transformations.rotation_matrix(math.radians(ang), [0, 0, 1], [c[0], c[1], 0]))
        pq = trimesh.proximity.ProximityQuery(q); v = tw.vertices; r = np.hypot(v[:, 0] - c[0], v[:, 1] - c[1])
        tabs = v[r > 18.8]; dist = np.abs(pq.signed_distance(tabs[::2]))
        print(f"{key} cap {lab:8s}: overlap with tweeter {(TW ^ to_m(q)).volume():.3f} mm3 | nearest tab/insulator point to the cap {dist.min():.2f} mm")
    ov = (TW ^ to_m(foam)).volume(); pqf = trimesh.proximity.ProximityQuery(foam)
    print(f"{key} foam ring + felt pad: overlap with tweeter {ov:.2f} mm3 (intended compression contact); nearest tweeter point {np.abs(pqf.signed_distance(tw.vertices[::7])).min():.2f} mm")
