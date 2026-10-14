"""Rebuild the body's skins as ONE outline-conforming mesh (no grid booleans against the outline).
Front and rear skins share a 2D mesh whose nodes lie exactly on the outline and are graded towards it;
the razor-nose profile is evaluated analytically at every node. CAD features are applied as cutters."""
import sys, json, numpy as np, trimesh, manifold3d as m3, shapely, gmsh
from shapely.geometry import Polygon, LineString, Point, box
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

d = sys.argv[1]
C = json.load(open(f"{d}/c7.json"))
TH, ZR, WG, SEAM = C["TH"], C["ZR"], C["WG_DEPTH"], C["SEAM_Y"]
UM, T, t = np.array(C["UM"][:2]), np.array(C["T"][:2]), np.array(C["t"][:2])
EDGE_T, E_W, THROAT_R, MOUTH_R, R_POCKET = 0.6, 24.0, 15.5, 50.0, 49.25
plan = Polygon(C["plan"]).buffer(0)
S5 = lambda u: (lambda v: v * v * v * (v * (v * 6 - 15) + 10))(np.clip(u, 0, 1))


def to_m(q):
    return m3.Manifold(m3.Mesh(vert_properties=np.asarray(q.vertices, np.float32), tri_verts=np.asarray(q.faces, np.uint32)))


def to_t(M):
    mm = M.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)


def prism(poly, z0, z1):
    polys = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
    out = None
    for g in polys:
        rings = [np.asarray(g.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in g.interiors]
        cs = m3.CrossSection(rings, m3.FillRule.EvenOdd)
        solid = m3.Manifold.extrude(cs, z1 - z0).translate((0, 0, z0))
        out = solid if out is None else out + solid
    return out


def cyl(c, r, z0, z1):
    q = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=256)
    q.apply_translation((c[0], c[1], (z0 + z1) / 2))
    return to_m(q)


# ---------------------------------------------------------------- fields
RS = np.load(f"{d}/rear_surface.npz")
gi = lambda name: RegularGridInterpolator((RS["ys"], RS["xs"]), RS[name], bounds_error=False, fill_value=None)
f_zrear, f_zfront, f_tedge, f_lam, f_depth, f_ramp = (gi(k) for k in ("zrear", "zfront", "t_edge", "lam", "depth", "ramp"))
FS = np.load(f"{d}/front_surface.npz")
f_F = RegularGridInterpolator((FS["ys"], FS["xs"]), FS["F"], bounds_error=False, fill_value=np.nan)
free = LineString(RS["free"])

# rear tweeter's riding flare on a fine local grid (same formula as the caps use)
hh = 0.3
lx = np.arange(t[0] - 62, t[0] + 62 + hh, hh)
ly = np.arange(t[1] - 62, t[1] + 62 + hh, hh)
LX, LY = np.meshgrid(lx, ly)
rho = np.hypot(LX - t[0], LY - t[1])
g = np.where(rho < MOUTH_R, WG - WG * (np.clip((rho - THROAT_R) / (MOUTH_R - THROAT_R), 0, 1)) ** (1 / 2.1), 0.0)
zr0 = f_zrear(np.c_[LY.ravel(), LX.ravel()]).reshape(LX.shape)
de = shapely.distance(shapely.points(LX.ravel(), LY.ravel()), plan.exterior).reshape(LX.shape)
kT = S5((np.hypot(LX - T[0], LY - T[1]) - 31.0) / 8.0)
shape = g / WG
old_s = np.maximum(zr0, ZR + g)
znew = zr0 + np.maximum(ZR + WG - zr0, 0) * shape
rw2 = S5((rho - 30.0) / 10.0) * (1 - S5((rho - 58.0) / 10.0)) * S5((de - 1.5) / 4.0) * kT
dl = znew - zr0
rear_local = np.maximum(znew + rw2 * (gaussian_filter(dl, sigma=5.0 / hh, mode="nearest") - dl), old_s)
f_rear_local = RegularGridInterpolator((ly, lx), rear_local, bounds_error=False, fill_value=np.nan)

# ---------------------------------------------------------------- conforming 2D mesh
ring = np.asarray(plan.exterior.coords)[:-1]
keep = [0]
for i in range(1, len(ring)):
    if np.hypot(*(ring[i] - ring[keep[-1]])) > 1e-4:
        keep.append(i)
ring = ring[keep]
if np.hypot(*(ring[0] - ring[-1])) < 1e-4:
    ring = ring[:-1]
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("skin")
pts = [gmsh.model.geo.addPoint(float(x), float(y), 0) for x, y in ring]
lines = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
surf = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(lines)])
gmsh.model.geo.synchronize()
fd = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(fd, "CurvesList", lines)
gmsh.model.mesh.field.setNumber(fd, "Sampling", 40)
ft = gmsh.model.mesh.field.add("Threshold")
for k, v in (("InField", fd), ("SizeMin", 0.2), ("SizeMax", 1.0), ("DistMin", 0.4), ("DistMax", 9.0)):
    gmsh.model.mesh.field.setNumber(ft, k, v)
gmsh.model.mesh.field.setAsBackgroundMesh(ft)
for k in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
    gmsh.option.setNumber(k, 0)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.model.mesh.generate(2)
tags, xyz, _ = gmsh.model.mesh.getNodes()
xyz = xyz.reshape(-1, 3)
_, _, conn = gmsh.model.mesh.getElements(2, surf)
gmsh.finalize()
index = {int(tg): i for i, tg in enumerate(tags)}
tri = np.array([index[int(v)] for v in conn[0]]).reshape(-1, 3)
used = np.unique(tri)
remap = -np.ones(len(xyz), int)
remap[used] = np.arange(len(used))
XY = xyz[used, :2]
tri = remap[tri]
a, b, c = XY[tri[:, 0]], XY[tri[:, 1]], XY[tri[:, 2]]
flip = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) < 0
tri[flip] = tri[flip][:, ::-1]
print("conforming mesh:", len(XY), "nodes,", len(tri), "triangles", flush=True)

# ---------------------------------------------------------------- skins at the nodes
q = np.c_[XY[:, 1], XY[:, 0]]
e = shapely.distance(shapely.points(XY[:, 0], XY[:, 1]), free)
t_edge, lam, Rb, ramp = f_tedge(q), np.clip(f_lam(q), 0, 1), -f_depth(q), np.clip(f_ramp(q), 0, 1)
EW = 5.0 + (E_W - 5.0) * ramp
u = np.clip(e / EW, 0, 1)
gn = np.sqrt(np.maximum(1 - (1 - u) ** 3, 0))
z_lo = np.minimum((1 - lam) * (TH - t_edge) + lam * Rb, TH - t_edge)
z_hi = z_lo + t_edge
F_raw = z_hi + (TH - z_hi) * gn
R_raw = z_lo + (Rb - z_lo) * gn
F_grid = f_F(q)
F_grid = np.where(np.isnan(F_grid), f_zfront(q), F_grid)
R_loc = f_rear_local(q)
R_grid = np.where(np.isnan(R_loc), f_zrear(q), R_loc)
k = 1 - S5(e / 3.0)
F = k * F_raw + (1 - k) * F_grid
R = k * R_raw + (1 - k) * R_grid
R = np.minimum(R, F - EDGE_T)
print("skin thickness: min %.3f mm; on the razor run (e = 0, ramp > 0.99): %.3f .. %.3f mm" % ((F - R).min(), (F - R)[(e < 1e-6) & (ramp > .99)].min(), (F - R)[(e < 1e-6) & (ramp > .99)].max()), flush=True)

n = len(XY)
V = np.vstack([np.c_[XY, F], np.c_[XY, R]])
edges = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
directed = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
_, inv, cnt = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
bnd = directed[cnt[inv.ravel()] == 1]
faces = [tri, tri[:, ::-1] + n, np.c_[bnd[:, 0], bnd[:, 0] + n, bnd[:, 1] + n], np.c_[bnd[:, 0], bnd[:, 1] + n, bnd[:, 1]]]
skin = trimesh.Trimesh(V, np.vstack(faces), process=False)
if skin.volume < 0:
    skin.invert()
print("skin solid: watertight", skin.is_watertight, "volume cm3", round(skin.volume / 1000, 1), flush=True)
Sm = to_m(skin)
print("   manifold status:", Sm.status(), flush=True)

# ---------------------------------------------------------------- CAD features as cutters
block = trimesh.load(f"{d}/_body_block.stl")
block.merge_vertices(digits_vertex=4)
big = prism(plan.buffer(.3, 32), ZR - .02, TH + .02)
shell = prism(plan.buffer(.3, 32), ZR - 1, TH + 1) - prism(plan.buffer(-.12, 32), ZR - 1.5, TH + 1.5)
neg = (big - to_m(block)) - shell
# the cutter must not leave a slab above the CAD block's top (it sliced a 0.02 mm slit under the raised front). Keep it strictly below the front plane,
# and continue every feature that is OPEN on the front face straight up through the raised front instead.
top_cs = neg.slice(TH - .05)
neg = (neg ^ prism(plan.buffer(1.0, 16), ZR - 1.0, TH - .04)) + m3.Manifold.extrude(top_cs, 12.0).translate((0, 0, TH - .045))
fill_front = prism(Point(*T).buffer(52.6, 128), TH - WG - .02, TH + 12.5)
fill_rear = prism(Point(*t).buffer(52.6, 128), ZR - .05, ZR + WG + .02)
SEAT_R = C["SEAT_D"] / 2
fill_front = fill_front - cyl(t, SEAT_R + .05, ZR - 2, TH + 12)
fill_rear = fill_rear - cyl(T, SEAT_R + .05, ZR - 2, TH + 12)
neg = neg - fill_front - fill_rear
ins = trimesh.creation.cylinder(radius=2.3, height=5.3, sections=64)
ins.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
ins.apply_translation((0.0, SEAM - 1.0 + 5.3 / 2, 12.55))
body = Sm - neg - to_m(ins) - cyl(UM, R_POCKET, TH, TH + 12) - cyl(T, THROAT_R, ZR - 2, TH + 12) - cyl(t, THROAT_R, ZR - 2, TH + 12)

# H2C seam B (LX_H2C_UPPER = released H2C Stock upper, design frame): fill the P2S dovetail pockets with its material (plate + D3 pins)
# BEFORE the lanes and the vestibule are cut: in the mirrored frame the vestibule overlaps one pocket box.
import os
if os.environ.get("LX_H2C_UPPER"):
    U = trimesh.load(os.environ["LX_H2C_UPPER"]); U.merge_vertices(digits_vertex=4); Um = to_m(U)
    for x0, x1 in ((-26.2, -11.8), (21.8, 36.2)):
        bx = to_m(trimesh.creation.box(bounds=[[x0, SEAM - 4.5, 0.0], [x1, SEAM + 6.2, 18.3]])); body = body + (Um ^ bx)
    print("H2C seam B: dovetail pockets filled, D3 pins added", flush=True)
    # the real MU10 is not mirrored: move its four M3 pilots (D4.6 x 4.0 below the seat) to 180-a so they read 58/148/238/328 deg after mirroring back
    if os.environ.get("LX_HAND", "std") == "mirror":
        UMc = np.array(C["UM"], float); seat = TH; P0 = (58.0, 148.0, 238.0, 328.0)
        def pil(a_, r_, z0, z1): return cyl(UMc + 44.75 * np.array([np.cos(np.radians(a_)), np.sin(np.radians(a_))]), r_, z0, z1)
        for a_ in P0: body = body + pil(a_, 2.3 + 0.02, seat - 4.0 - 0.02, seat - 0.001)
        for a_ in P0: body = body - pil((180.0 - a_) % 360, 2.3, seat - 4.0, seat + 1.0)
        print("MU10 pilots moved to", [round((180.0 - a_) % 360) for a_ in P0], "deg (mirrored frame)", flush=True)
# cable tubes and the vestibule on the seam face
for nm in ("lane_rear_t.stl", "lane_front_T.stl"):
    lt = trimesh.load(f"{d}/{nm}")
    lt.merge_vertices(digits_vertex=4)
    body = body - to_m(lt)
sl = C["ports"]["slot"]
y0, y1 = SEAM - .5, SEAM + sl["len"] + .5
hull = lambda P: to_m(trimesh.convex.convex_hull(np.array(P, float)))
body = body - hull([(x, y, z) for x in (sl["x_out"], sl["x_in"]) for (y, z) in ((y0, sl["z_floor_seam"]), (y1, sl["z_floor_end"]), (y1, sl["z_top"]), (y0, sl["z_top"]))])
body = body - hull([(x, y0, z) for x in (sl["x_out"] - .2, sl["x_in"] + .9) for z in (sl["z_floor_seam"] - .9, sl["z_top"] + .7)] + [(x, SEAM + 2.0, z) for x in (sl["x_out"], sl["x_in"]) for z in (sl["z_floor_seam"] - .4, sl["z_top"])])

out = to_t(body)
out.merge_vertices(digits_vertex=5)
parts = out.split(only_watertight=False)
out = max(parts, key=lambda p: len(p.faces))
print("   fragments dropped (mm3):", sorted([round(abs(p.volume), 2) for p in parts if p is not out], reverse=True)[:6], flush=True)
print("conforming body: watertight", out.is_watertight, "| pieces", len(parts), "| volume cm3", round(out.volume / 1000, 1), flush=True)
out.export(f"{d}/_body_with_lanes_front.stl")
