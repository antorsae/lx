"""V11 split checks: new top vs released keyed top (seam region identical), vs released keyed bottom (no overlap, seam contact), H2C bed fit, ducts."""
import sys, json, math, numpy as np, trimesh, manifold3d as m3
from shapely.geometry import MultiPoint
d, V = sys.argv[1], sys.argv[2]; C = json.load(open(f"{d}/{V}_build.json"))
to_m = lambda q: m3.Manifold(m3.Mesh(vert_properties=np.asarray(q.vertices, np.float32), tri_verts=np.asarray(q.faces, np.uint32)))
def load(n):
    q = trimesh.load(f"{d}/{n}"); q.merge_vertices(merge_tex=True, merge_norm=True); return q
top, rtop, rbot = load(f"{V}.stl"), load("released_lm_keyed_top_floor_design.stl"), load("released_lm_keyed_bottom_floor_design.stl")
T, Rt, Rb = to_m(top), to_m(rtop), to_m(rbot)
box = lambda y0, y1: m3.Manifold.cube((400, y1 - y0, 400)).translate((-200, y0, -200))
seam = C.get("SEAM_Y", 172.481)
for y0, y1, nm in ((seam - 1, seam + 30, "seam band y 172.5..202.5"), (seam - 1, 262.0, "released region y 172.5..262")):
    a, b = T ^ box(y0, y1), Rt ^ box(y0, y1)
    print(f"new top vs released keyed top, {nm}: new-minus-released {(a - b).volume():.2f} mm3 | released-minus-new {(b - a).volume():.2f} mm3")
print(f"new top vs released keyed BOTTOM: overlap {(T ^ Rb).volume():.3f} mm3 (must be ~0) | top y-min {top.bounds[0, 1]:.2f}, bottom y-max {rbot.bounds[1, 1]:.2f}")
print(f"top: watertight {top.is_watertight}, volume {top.volume / 1000:.1f} cm3, bounds {np.round(top.bounds, 1).tolist()}")
# H2C bed: printable 330 x 320, single-nozzle 325 x 320; front-face-down => footprint = plan outline
hull = MultiPoint(top.vertices[:, :2]).convex_hull; best = None
for a in np.arange(0, 180, 1.0):
    P = np.array(hull.exterior.coords); c, s_ = math.cos(math.radians(a)), math.sin(math.radians(a)); Q = P @ np.array([[c, s_], [-s_, c]])
    w, h = np.ptp(Q[:, 0]), np.ptp(Q[:, 1]); m_ = min(325 - w, 320 - h)
    if best is None or m_ > best[0]: best = (m_, a, w, h)
print(f"H2C fit (325 x 320 single-nozzle area): best in-plane rotation {best[1]:.0f} deg -> footprint {best[2]:.1f} x {best[3]:.1f} mm, min margin {best[0]:.1f} mm | print height {np.ptp(top.vertices[:, 2]):.1f} mm")
