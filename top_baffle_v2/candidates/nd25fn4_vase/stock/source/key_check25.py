"""Keyed V11 top vs the rim-centred vendor tweeters: seated overlap, axial insertion path, rotation lock, allowed clockings."""
import sys, json, math, numpy as np, trimesh, manifold3d as m3
d = sys.argv[1]; P = json.load(open(f"{d}/tweeter_placement.json"))
to_m = lambda q: m3.Manifold(m3.Mesh(vert_properties=np.asarray(q.vertices, np.float32), tri_verts=np.asarray(q.faces, np.uint32)))
body = trimesh.load(f"{d}/_body_keyed.stl"); B = to_m(body)
tw0 = trimesh.load(f"{d}/nd25fn4_vendor_rimcentred.stl")
for key in ("T", "t"):
    T4 = np.array(P[key]); axis = T4[:3, :3] @ np.array([1., 0, 0]); ctr = T4[:3, 3]      # model +x = rearward (insertion comes from there)
    tw = tw0.copy(); tw.apply_transform(T4)
    def ov(m): return (B ^ to_m(m)).volume()
    print(f"== tweeter {key}: seated overlap {ov(tw):.3f} mm3")
    worst = max(ov(tw.copy().apply_translation(axis * s)) for s in (0.3, 1, 2, 3, 4, 5, 6, 8, 10, 14, 20, 30))
    print(f"   insertion path (0.3..30 mm back along the axis): worst overlap {worst:.3f} mm3")
    for deg in (2, 5, 45, 90, 180, 270, -2, -5):
        m = tw.copy(); m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(deg), axis, ctr))
        # rotation test at the seated position AND slid 2 mm back (to see whether it can even enter in that clocking)
        m2 = m.copy().apply_translation(axis * 8.0)
        print(f"   rotated {deg:+4d} deg: seated overlap {ov(m):8.2f} mm3 | 8 mm before seating {ov(m2):8.2f} mm3")
