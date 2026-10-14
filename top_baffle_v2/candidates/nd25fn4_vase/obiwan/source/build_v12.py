"""Obiwan option C / V3d as ONE printable piece: released no-floor LM carrier (kept below the crown cut) + new fused
upper body (MU10 flush seat, front ND25FN-4 waveguide T, rear ND25FN-4 waveguide t, razor edge, buried cable tubes).
Mesh level: outline-conforming skin (gmsh) + manifold3d cutters. Units mm, design frame (front face z=18.3, core rear z=6.8)."""
import os, sys, json, math, numpy as np, trimesh, manifold3d as m3, shapely, gmsh
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from shapely.ops import linemerge

d = sys.argv[1]; COLLAR = False; RIMS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0; OUT = sys.argv[3] if len(sys.argv) > 3 else "obiwan_v12_top"
TH, ZR = 18.3, 6.8                                   # released Obiwan plate: front face / core rear
LM = np.array([0.0, 200.981]); R_LM, R_LMrec, R_LMbore, SEAT_LM, MEMB = 113.94, 110.6, 95.0, 12.3, 0.85   # visible ring = core 113.0 + 0.94 fairing (released bbox ±113.94)
R_UM, R_UMrec, R_UMbore, SEAT_UM = 52.64, 49.3, 41.0, 14.3
UM_PILOT_R, UM_PILOT_ANG, UM_PILOT_D, UM_PILOT_DEPTH = 89.5 / 2, (58.0, 148.0, 238.0, 328.0), 4.6, 4.0
if os.environ.get("LX_HAND", "std") == "mirror": UM_PILOT_ANG = tuple((180.0 - a_) % 360 for a_ in UM_PILOT_ANG)   # the real MU10 is not mirrored: its pilots read 180-a here
HAND = os.environ.get("LX_HAND", "std")               # "mirror": built in the x-mirrored frame (inputs mirrored by the caller, outputs mirrored back)
Y_CUT = 306.5                                        # old crown cut (kept as a reference line only)
Y2 = 278.0                                           # V6: the released LM is kept only below y 278 (both ducts still buried in its core there); above it the rear fades over the ring top
PILOT_XY = [(52.375, 291.697), (-52.375, 291.697)]   # released L22 M5 insert pilots at 60/120 deg (flush.py LM_PILOT_XY)
WALL, G_LM, G = 3.5, 1.0, 4.0
THROAT, WG, BOSS, R_SKIRT, R_KEY, KEY_D, LUG_N = 15.9, 11.9, 1.0, 23.2, 23.4, 4.6, 3
R_CAPB, CH_D, TS_R = 29.0, 10.0, 3.0   # V11: tweeter duct D6 like the released lane in the reused bottom half (lead 4x0.5 mm2, OD <= 5.2)
def ring_clamp(x_, y_, r_max=108.6):
    r_ = math.hypot(x_, y_ - 200.981)
    return (x_ * r_max / r_, 200.981 + (y_ - 200.981) * r_max / r_) if r_ > r_max else (x_, y_)                # V10: R29 x 10 wiring chamber at each tweeter's terminal end (cap + bayonet on it); tweeter duct D8.2 (4x0.75 mm2 lead from the NL8)
ZV, FADE_L, DISC = -22.8, 14.0, 2.0                  # vase rear plane under the tweeter cluster (T cap flush there), S-fade length from the crown, cap disc thickness
MU_FLARE_W, MU_FLARE_TOP = 30.0, 7.0                 # rear flare around the MU10 opening (vase24 style), rising to z 7 at the opening edge (pilots bottom at 10.3)   # Stock vase24 datum: throat floor 12.6 behind the MU10 flange plane, faceplate boss (r 15.6) flush with it, flange plane 1.0 below
BUMP_PLAT, BUMP_BLEND, RIM_MIN = R_SKIRT + 1.2 + 0.8, 6.0, 32.0   # (V4: no bumps; RIM_MIN keeps room for the flush caps + razor)   # cap-side bump: plateau covers bore + bayonet pockets + 0.8 wall; blends over 6 mm; rims must reach the foot
TUBE_R, SKIN = 3.0, 0.75                             # released TS duct D6; released tunnel skins 0.75..0.85
EDGE_T, EW, EW_CAP = 0.6, 24.0, 1.5                # vase24 nose width where the lobes allow it; limited near the caps / MU10 seat (see EWn)                   # razor edge thickness; nose width on the lobes; short nose where the rear tweeter's cap bore needs the full front
S5 = lambda u: (lambda v: v * v * v * (v * (v * 6 - 15) + 10))(np.clip(u, 0, 1))


# ------------------------------------------------------------------ layout (V3d, t re-solved for the real cap footprint)
def rim_r(th, R):
    c, s = np.cos(th), np.sin(th)
    return R["r"] * c * c * (c > 0) + R["l"] * c * c * (c < 0) + R["u"] * s * s * (s > 0) + R["d"] * s * s * (s < 0)


def rim_poly(c, R, inset=0.0, n=720):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False); r = rim_r(th, R) - inset
    return Polygon(np.c_[c[0] + r * np.cos(th), c[1] + r * np.sin(th)])


R0 = 38.0
RM = max(RIM_MIN, RIMS)
T_rim = dict(l=max(R0, RM), r=max(R0, RM), u=max(R0, RM), d=max(.8 * R0, RM)); t_rim = dict(l=max(.7 * R0, RM), r=max(.7 * R0, RM), u=max(.7 * R0, RM), d=max(.8 * R0, RM))
LMc = Point(LM).buffer(R_LM, 256)
D = R_UM + R0 + G; xM, xT = -D / 2, D / 2
yM = LM[1] + math.sqrt((R_LM + R_UM + G_LM) ** 2 - xM ** 2)
yT = brentq(lambda y: rim_poly((xT, y), T_rim).distance(LMc) - G_LM, LM[1] + 100, LM[1] + 300)
MU, T = np.array([xM, yM]), np.array([xT, yT])
T_rim["l"] = float(np.linalg.norm(T - MU) - R_UM)
R_CAP = R_CAPB + 0.8                                  # t's cavity wall (3 mm) must clear T's front mouth
Tmouth = rim_poly(T, T_rim, WALL); Tbody = Point(T).buffer(R_CAP); MUrim = Point(MU).buffer(R_UM + 1.0); MUfl = Point(MU).buffer(R_UMrec + 1.0)
sd = lambda a, b: -1.0 if a.intersects(b) else a.distance(b)
ok = lambda c: min(sd(Point(c).buffer(R_CAP), Tmouth), sd(rim_poly(c, t_rim, WALL), Tbody), sd(Point(c).buffer(R_CAP), MUrim), sd(rim_poly(c, t_rim, WALL), MUfl)) >= 0
best = None
for x in np.arange(-10, 50.1, 0.5):
    last = None
    for y in np.arange(470, 340, -0.5):
        if not ok((x, y)):
            break
        last = (x, y)
    if last and (best is None or last[1] < best[1]):
        best = last
t = np.array(best, float)
print("layout: MU10", MU.round(2), "T", T.round(2), "t", t.round(2), "T rim", {k: round(v, 1) for k, v in T_rim.items()}, "t rim", t_rim, flush=True)

lobes = unary_union([Point(MU).buffer(R_UM, 256), rim_poly(T, T_rim), rim_poly(t, t_rim), LMc])
body2d = unary_union([lobes.buffer(14, 64).buffer(-14, 64), lobes.buffer(30, 64).buffer(-30, 64).intersection(box(-400, 0, 400, 350))])   # R14 between the lobes, R30 tangential cusps at the LM ring
plan = body2d.intersection(box(-400, Y2 - 0.3, 400, 700)).difference(Point(LM).buffer(R_LMbore, 256))    # V6: whole ring top above y 278 is new (0.3 overlap into the released ring), L22 bore excluded
plan = plan.buffer(0)
plan = max(plan.geoms, key=lambda g: g.area) if plan.geom_type != "Polygon" else plan
print("plan area cm2", round(plan.area / 100, 1), "holes", len(plan.interiors), flush=True)


# ------------------------------------------------------------------ helpers
def to_m(q):
    return m3.Manifold(m3.Mesh(vert_properties=np.asarray(q.vertices, np.float32), tri_verts=np.asarray(q.faces, np.uint32)))


def to_t(M):
    mm = M.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)


def prism(poly, z0, z1):
    out = None
    for g in ([poly] if poly.geom_type == "Polygon" else list(poly.geoms)):
        rings = [np.asarray(g.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in g.interiors]
        s = m3.Manifold.extrude(m3.CrossSection(rings, m3.FillRule.EvenOdd), z1 - z0).translate((0, 0, z0))
        out = s if out is None else out + s
    return out


def cyl(c, r, z0, z1, n=192):
    q = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=n); q.apply_translation((c[0], c[1], (z0 + z1) / 2)); return to_m(q)


def tube(path, r, step=0.8, n=32):
    """capsule chain along a polyline (already densely sampled)"""
    P = np.asarray(path, float); out = None
    for a, b in zip(P[:-1], P[1:]):
        v = b - a; L = np.linalg.norm(v)
        if L < 1e-6:
            continue
        q = trimesh.creation.cylinder(radius=r, height=L + 0.3, sections=n)
        q.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], v / L)); q.apply_translation((a + b) / 2)
        out = to_m(q) if out is None else out + to_m(q)
    for p in P[::4]:
        s = trimesh.creation.icosphere(subdivisions=2, radius=r); s.apply_translation(p); out = out + to_m(s)
    return out


def spline_path(pts, step=0.8):
    pts = np.asarray(pts, float); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    cs = PchipInterpolator(s, pts); u = np.arange(0, s[-1], step)
    return cs(np.r_[u, s[-1]])


# ------------------------------------------------------------------ waveguide fields
def flare(XY, c, R, depth):
    """circular-arc flare: depth at the throat, tangent to the face at the 4-radius mouth"""
    dx, dy = XY[:, 0] - c[0], XY[:, 1] - c[1]; rho = np.hypot(dx, dy); th = np.arctan2(dy, dx)
    M = rim_r(th, R) - WALL; u = np.clip((rho - THROAT) / (M - THROAT), 0, 1)
    return np.where(rho < M, depth * (1 - np.sqrt(np.maximum(1 - (1 - u) ** 2, 0))), 0.0)


# ------------------------------------------------------------------ conforming 2D mesh (outline + L22 bore loop)
def clean_ring(coords):
    ring = np.asarray(coords)[:-1]; keep = [0]
    for i in range(1, len(ring)):
        if np.hypot(*(ring[i] - ring[keep[-1]])) > 1e-4:
            keep.append(i)
    ring = ring[keep]
    return ring[:-1] if np.hypot(*(ring[0] - ring[-1])) < 1e-4 else ring


gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0); gmsh.model.add("skin")
loops, all_lines = [], []
for coords in [plan.exterior.coords] + [r.coords for r in plan.interiors]:
    ring = clean_ring(coords)
    pts = [gmsh.model.geo.addPoint(float(x), float(y), 0) for x, y in ring]
    lines = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    loops.append(gmsh.model.geo.addCurveLoop(lines)); all_lines += lines
surf = gmsh.model.geo.addPlaneSurface(loops); gmsh.model.geo.synchronize()
fd = gmsh.model.mesh.field.add("Distance"); gmsh.model.mesh.field.setNumbers(fd, "CurvesList", all_lines); gmsh.model.mesh.field.setNumber(fd, "Sampling", 40)
ft = gmsh.model.mesh.field.add("Threshold")
for k, v in (("InField", fd), ("SizeMin", 0.25), ("SizeMax", 0.8), ("DistMin", 0.4), ("DistMax", 26.0)):
    gmsh.model.mesh.field.setNumber(ft, k, v)
# refine the waveguides too (throat rings are steep)
fc = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(fc, "F", "0.35 + 0.9*min(1, max(0, (sqrt((x-(%f))^2+(y-(%f))^2)-14)/8)) * min(1, max(0, (sqrt((x-(%f))^2+(y-(%f))^2)-14)/8))" % (T[0], T[1], t[0], t[1]))
fk = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(fk, "F", "0.45 + 0.55*min(1, max(0, (min(sqrt((x-(%f))^2+(y-(%f))^2), sqrt((x-(%f))^2+(y-(%f))^2)) - 50)/8))" % (T[0], T[1], t[0], t[1]))
fm = gmsh.model.mesh.field.add("Min"); gmsh.model.mesh.field.setNumbers(fm, "FieldsList", [ft, fc, fk]); gmsh.model.mesh.field.setAsBackgroundMesh(fm)
for k in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
    gmsh.option.setNumber(k, 0)
gmsh.option.setNumber("Mesh.Algorithm", 6); gmsh.model.mesh.generate(2)
tags, xyz, _ = gmsh.model.mesh.getNodes(); xyz = xyz.reshape(-1, 3)
_, _, conn = gmsh.model.mesh.getElements(2, surf); gmsh.finalize()
index = {int(tg): i for i, tg in enumerate(tags)}
tri = np.array([index[int(v)] for v in conn[0]]).reshape(-1, 3)
used = np.unique(tri); remap = -np.ones(len(xyz), int); remap[used] = np.arange(len(used)); XY = xyz[used, :2]; tri = remap[tri]
a, b, c = XY[tri[:, 0]], XY[tri[:, 1]], XY[tri[:, 2]]
flip = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) < 0; tri[flip] = tri[flip][:, ::-1]
print("conforming mesh:", len(XY), "nodes", len(tri), "triangles", flush=True)

# ------------------------------------------------------------------ vase24-style sculpt on a grid (then sampled at the conforming nodes)
E_W, FADE_W, MU_W, DM_W, RUN, SIG = 24.0, 40.0, 44.0, 36.0, 60.0, 4.0     # vase24: nose 24, zone fade 40, MU rear flare 44, mating fade 36; run ends 60; low-pass sigma 4 mm
LAND_T = TH - WG - BOSS; LAND_t = ZV + WG + BOSS; ZB_t = TH; THROAT_t = ZV + WG
G = ZR - ZV                                                                                   # 29.6 mm of vase depth below the released core rear
mating = unary_union([box(-400, -100, 400, Y2 + 0.05), Point(LM).buffer(R_LM + 0.3, 256).difference(Point(LM).buffer(R_LM - 0.3, 256)), Point(LM).buffer(R_LMbore + 0.3, 256)])
ring_ = clean_ring(plan.exterior.coords); k0 = int(np.argmin(ring_[:, 0])); ring_ = np.vstack([ring_[k0:], ring_[:k0], ring_[k0:k0 + 1]])
free = LineString(ring_).difference(mating); free = linemerge(free) if free.geom_type != "LineString" else free
free = max(getattr(free, "geoms", [free]), key=lambda g_: g_.length); Lf = free.length
mate_line = plan.exterior.intersection(mating)
print("razor run %.0f mm of %.0f | from %s to %s" % (Lf, plan.exterior.length, np.round(free.coords[0], 1), np.round(free.coords[-1], 1)), flush=True)
zone = unary_union([rim_poly(T, T_rim), rim_poly(t, t_rim)])                                  # the tweeter cluster is at full vase depth; the rest fades back to the plate
h = 0.6; xs = np.arange(plan.bounds[0] - 3, plan.bounds[2] + 3 + h, h); ys = np.arange(plan.bounds[1] - 3, plan.bounds[3] + 3 + h, h); X, Y = np.meshgrid(xs, ys); P = shapely.points(X.ravel(), Y.ravel())
e_g = shapely.distance(P, free).reshape(X.shape); dz = shapely.distance(P, zone).reshape(X.shape); dm = shapely.distance(P, mate_line).reshape(X.shape)
rM = np.hypot(X - MU[0], Y - MU[1]); dT_g = np.hypot(X - T[0], Y - T[1]); dt_g = np.hypot(X - t[0], Y - t[1])
_q = np.linspace(0, 1, 2001); _sl = 1 - S5((_q - .40) / .60); _f = np.concatenate([[0], np.cumsum((_sl[1:] + _sl[:-1]) / 2 * (_q[1] - _q[0]))]); _f /= _f[-1]
FLARE = lambda q: np.where(q < 0, q * (_f[1] / _q[1]), np.interp(np.clip(q, 0, 1), _q, _f))   # vase24 MU rear horn: straight from the bore, C2 landing
w_tcav = 1 - S5((dt_g - (R_CAPB + 4.0)) / 10.0)                                             # t's 28 mm cavity needs the full vase depth round it: overrides the MU10 rear horn locally (t is only 59 mm from the MU10 axis here, 112 in vase24)
depth = G * (1 - S5(dz / FADE_W)) * np.maximum(FLARE(np.clip((rM - R_UMbore) / MU_W, -.6, 1)), w_tcav) * S5(dm / DM_W)
# V8 (owner: "bury the T duct fully"): fairing along both released-style duct paths. Under each pipe the rear is lowered to >= 1 mm below the
# pipe's outer wall, fading to nothing 18 mm to either side (pipe-narrow at the y 278 joint so it meets the released pipe below the cut).
TS_REL = [(-79.3, 276.0, 8.3), (-78.4, 278, 8.2), (-76.3, 280, 7.9), (-71.9, 284, 6.4), (-67.1, 288, 3.9), (-61.6, 292, 1.6), (-58.4, 294, 1.0), (-54.8, 296, 0.8), (-52, 297.6, 0.8),
          (-48, 299.1, 1.5), (-44, 300.1, 2.7), (-40, 300.4, 4.4), (-36, 300.2, 6.1), (-32, 299.8, 7.3), (-28, 299.4, 8.1), (-24, 299.2, 8.3), (-16, 300.0, 8.3), (-8, 302.5, 8.3), (0, 306.5, 8.3)]
dir0 = np.array([0.68, 0.73, 0.0]); dir0 /= np.linalg.norm(dir0); p0 = np.array([0.0, 306.5, 8.3])   # V11: D6 on the released centreline
a_kT = math.atan2(Y_CUT - T[1], -0.16 - T[0]); KT = T + (R_SKIRT + 2.5) * np.array([math.cos(a_kT), math.sin(a_kT)])   # tube mouth inside T's cavity wall
dKT = np.array([math.cos(a_kT), math.sin(a_kT)]); KT = T + (R_CAPB + 1.0) * dKT
pT = spline_path(TS_REL + [p0 + dir0 * 8, [p0[0] + dir0[0] * 16, p0[1] + dir0[1] * 16, 8.0], [KT[0] + 8 * dKT[0], KT[1] + 8 * dKT[1], -2.0], [KT[0], KT[1], -15.5], [*(T + (R_CAPB - 2.0) * dKT), -15.5]])   # V10: into T's wiring chamber
# V13 (owner): the V10-V12 T lead dropped 13.5 mm over its last 8 mm (59 deg) into T's wiring chamber. Descend over a longer
# approach instead (<= ~30 deg): right hand spirals down round T's west side (entry 140 deg), the mirrored frame comes
# straight from the ring side (entry -100 deg). Entries stay >= 30 deg from the T->t link (110 deg) so the cap's bayonet
# pockets (entry + 60/180/300) clear it.
def _u(a): return np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
if HAND == "mirror":
    a_kT = math.radians(-100.0); dKT = _u(-100.0); KT = T + (R_CAPB + 1.0) * dKT
    TS_REL = [(79.3, 276.0, 8.3), (78.4, 278, 8.2), (76.3, 280, 7.9), (71.9, 284, 6.4), (67.1, 288, 3.9), (61.6, 292, 1.6)]   # the released route's own shape (mirror of the right hand's)
    _a, _b = np.array([61.6, 292.0]), KT; _mid = []
    for _f in (0.25, 0.5, 0.75):
        _q = _a + (_b - _a) * _f; _mid.append([_q[0], _q[1], 1.6 - 16.6 * (3 * _f ** 2 - 2 * _f ** 3)])
    pT = spline_path(TS_REL + _mid + [[KT[0], KT[1], -15.0], [*(T + (R_CAPB - 2.0) * dKT), -15.5]])
else:
    a_kT = math.radians(140.0); dKT = _u(140.0); KT = T + (R_CAPB + 1.0) * dKT
    pT = spline_path(TS_REL + [p0 + dir0 * 8, [p0[0] + dir0[0] * 16, p0[1] + dir0[1] * 16, 8.0], [8.0, 326.0, 7.5], [5.2, 336.0, 3.5],
                               [*(T + 40.0 * _u(180.0)), -3.5], [*(T + 35.0 * _u(160.0)), -10.5], [KT[0], KT[1], -15.0], [*(T + (R_CAPB - 2.0) * dKT), -15.5]])
_sl = np.degrees(np.arctan2(-np.diff(pT[:, 2]), np.linalg.norm(np.diff(pT[:, :2], axis=0), axis=1)))
print("V13 T lead: steepest descent %.0f deg, entry %.0f deg" % (_sl.max(), math.degrees(a_kT)), flush=True)
UM_R = 4.1
UM_REL = [(77.9, 276.0, 7.2), (76.8, 278, 7.0), (74.7, 280, 6.6), (72.5, 282, 5.9), (70.2, 284, 4.9), (67.8, 286, 3.7), (65.2, 288, 2.4), (62.4, 290, 1.2), (59.1, 292, 0.3), (55.0, 294, -0.2),
          (50.8, 296, -0.3), (46.7, 298, -0.3), (42.5, 300, -0.1), (38.5, 302, 0.2), (35.1, 304, 0.5), (32.6, 306, 0.9), (30.8, 308, 1.1), (26.5, 311, 1.4)]
def duct_fairing(path, r_out, W_full=18.0, margin=1.0):
    Pp = np.asarray(path, float); ls = LineString(Pp[:, :2]); sP = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pp[:, :2], axis=0), axis=1))]
    dd = shapely.distance(P, ls).reshape(X.shape); ss = shapely.line_locate_point(ls, P).reshape(X.shape); zc = np.interp(ss, sP, Pp[:, 2])
    need = np.maximum(ZR - (zc - r_out - margin), 0.0)
    W = r_out + (W_full - r_out) * S5((Y - Y2) / 12.0)
    return need * S5((W - dd) / np.maximum(W - r_out, 0.5)) * (Y > Y2 - 1.0)
UM_TERM_DEG = 283.0                                                                          # MU10 terminal strip clock (owner photo); exit dives out near the MU10 rim there
UM_NEW = [(20.0, 315.0, -1.0), (12.0, 318.5, -1.8), (4.0, 319.0, 0.2), (-4.0, 317.5, 4.5), (-12.0, 315.5, 8.6), (-20.0, 314.0, 10.3), (-26.0, 313.0, 10.6)]   # V12: under the T duct (>= 7.9 mm apart), then up into the plate
if HAND == "mirror":                                                                          # the real driver is not mirrored: its 283 deg strip reads 180-283 = 257 deg in this frame
    UM_TERM_DEG = 257.0
    UM_REL = [(-77.9, 276.0, 7.2), (-76.8, 278, 7.0), (-74.7, 280, 6.6), (-72.5, 282, 5.9), (-70.2, 284, 4.9), (-67.8, 286, 3.7), (-65.2, 288, 2.4), (-62.4, 290, 1.2), (-59.1, 292, 0.3)]
    UM_NEW = [(-58.2, 297.0, 0.4), (-58.4, 302.0, 0.8), (-58.8, 306.0, 1.2)]
depth = (np.maximum(depth, 0) ** 4 + duct_fairing(pT, TS_R + 0.8, margin=2.0) ** 4 + duct_fairing(spline_path(UM_REL + UM_NEW), 4.1 + 0.8) ** 4) ** 0.25   # p-norm, not max: a hard max creased where the fairing met the body depth
Rb = ZR - depth
sl = shapely.line_locate_point(free, P).reshape(X.shape); ramp = S5(np.minimum(sl / RUN, (Lf - sl) / RUN))
pe = shapely.get_coordinates(shapely.line_interpolate_point(free, sl.ravel())).reshape(X.shape + (2,)); lam_g = S5((pe[..., 1] - T[1]) / (t[1] - T[1]))
lam_g = gaussian_filter(lam_g, sigma=5.0 / h, mode='nearest'); ramp = gaussian_filter(ramp, sigma=5.0 / h, mode='nearest')   # both come from the nearest outline point, which jumps across concave notches (T/t valley kink line)
t_edge = EDGE_T + ((TH - ZR) - EDGE_T) * (1 - ramp)                                          # edge thickness: full plate (11.5) at the crown ends of the run -> 0.6 razor
EWg = 5.0 + (E_W - 5.0) * ramp
smin_e = lambda *A, k=3.0: -k * np.log(sum(np.exp(-np.asarray(a_, float) / k) for a_ in A))   # smooth minimum (never above the hard min)
EWg = np.maximum(smin_e(EWg, e_g + dT_g - (R_CAPB + 2.0), e_g + dt_g - (R_CAPB + 2.0)), 1.5)   # nose may use the whole edge-to-cavity-rim gap (minus 1 mm), not half of it   # V7: hard min of the two cavity distances creased the front along the T/t bisector   # the sweep must not reach the cap cavities (their rims stay at the cap datum)
u_g = np.clip(e_g / EWg, 0, 1); gn_g = np.sqrt(np.maximum(1 - (1 - u_g) ** 3, 0))
z_lo_g = np.minimum((1 - lam_g) * (TH - t_edge) + lam_g * Rb, TH - t_edge); z_hi_g = z_lo_g + t_edge
zfront = z_hi_g + (TH - z_hi_g) * gn_g; zrear = z_lo_g + (Rb - z_lo_g) * gn_g
zs = gaussian_filter(zrear, sigma=SIG / h, mode="nearest"); keep = np.clip(1 - e_g / 6.0, 0, 1); zrear = keep * zrear + (1 - keep) * zs      # NO SHARP EDGES on the rear; the razor itself stays crisp
zrear = np.minimum(zrear, zfront - EDGE_T)
zneg = zrear - ZR; zrear = zrear - np.maximum(zneg, 0) * (1 - S5((dm - 1.0) / 10.0))          # exactly flush with the released core rear along the crown
w_seat = 1 - S5((dT_g - (R_CAPB + 2.3)) / 12.0); zrear = w_seat * ZV + (1 - w_seat) * zrear     # level cap datum over T's cavity, applied last (the tweeter's rear end and the cap need it)
zrear = np.minimum(zrear, zfront - EDGE_T)
np.savez_compressed(f"{d}/{OUT}_fields.npz", xs=xs, ys=ys, zrear=zrear, zfront=zfront, t_edge=t_edge, lam=lam_g, Rb=Rb, EW=EWg, ramp=ramp)
gi = lambda A: RegularGridInterpolator((ys, xs), A, bounds_error=False, fill_value=None)
f_zr, f_zf, f_te, f_lam, f_Rb, f_EW = (gi(A) for A in (zrear, zfront, t_edge, lam_g, Rb, EWg))
# ------------------------------------------------------------------ skins at the nodes: grid fields inside, analytic razor within 3 mm of the free edge (no serration)
q = np.c_[XY[:, 1], XY[:, 0]]
e = shapely.distance(shapely.points(XY[:, 0], XY[:, 1]), free)
te_n, lam_n, Rb_n, EW_n = f_te(q), np.clip(f_lam(q), 0, 1), f_Rb(q), np.maximum(f_EW(q), 1.5)
u = np.clip(e / EW_n, 0, 1); gn = np.sqrt(np.maximum(1 - (1 - u) ** 3, 0))
z_lo = np.minimum((1 - lam_n) * (TH - te_n) + lam_n * Rb_n, TH - te_n); z_hi = z_lo + te_n
F_raw = z_hi + (TH - z_hi) * gn; R_raw = z_lo + (Rb_n - z_lo) * gn
k = 1 - S5(e / 3.0)
F = k * F_raw + (1 - k) * f_zf(q); R = k * R_raw + (1 - k) * f_zr(q)
rho_T = np.hypot(XY[:, 0] - T[0], XY[:, 1] - T[1]); rho_t = np.hypot(XY[:, 0] - t[0], XY[:, 1] - t[1])
def smin(a, b, k):
    h_ = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1); return b * (1 - h_) + a * h_ - k * h_ * (1 - h_)
_d1, _d2 = np.maximum(TH - F, 0), flare(XY, T, T_rim, WG)
F = TH - (_d1 ** 3 + _d2 ** 3) ** (1 / 3)                                                   # p-norm of the two depths: exact where either is zero (flat face stays at 18.3), smooth where the T mouth meets the razor nose                                               # soft-min: the T mouth meets the razor nose without a kink (V6 had a hard min -> 29 deg crease)                                             # T front waveguide in the flat Obiwan face
shape_t = flare(XY, t, t_rim, WG) / WG; R = np.maximum(R, R + (THROAT_t - R) * shape_t)     # t rear waveguide riding on the local rear surface (throat lands at the Stock depth)
R = np.where(shape_t > 0, R + (THROAT_t - R) * shape_t, R)
ZB_T = float(R[np.abs(rho_T - R_CAPB) < 1.5].min()) - 0.05
print("T cavity rim rear height: %.2f .. %.2f (cap datum %.2f)" % (R[np.abs(rho_T - R_CAPB) < 1.5].min(), R[np.abs(rho_T - R_CAPB) < 1.5].max(), ZB_T), flush=True)
R = np.minimum(R, F - EDGE_T)
print("skin thickness min %.2f; nodes with thickness < 2 mm: %d (razor band expected)" % ((F - R).min(), ((F - R) < 2).sum()), flush=True)
n = len(XY); V = np.vstack([np.c_[XY, F], np.c_[XY, R]])
edges = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1); directed = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
_, inv, cnt = np.unique(edges, axis=0, return_inverse=True, return_counts=True); bnd = directed[cnt[inv.ravel()] == 1]
faces = [tri, tri[:, ::-1] + n, np.c_[bnd[:, 0], bnd[:, 0] + n, bnd[:, 1] + n], np.c_[bnd[:, 0], bnd[:, 1] + n, bnd[:, 1]]]
skin = trimesh.Trimesh(V, np.vstack(faces), process=False)
if skin.volume < 0:
    skin.invert()
Sm = to_m(skin); print("skin solid watertight", skin.is_watertight, "status", Sm.status(), "vol cm3", round(skin.volume / 1000, 1), flush=True)

# ------------------------------------------------------------------ cutters: seats, bores, keyways, bayonets
cut = None
def add(c):
    global cut
    cut = c if cut is None else cut + c

above = box(-400, Y2 - 1.0, 400, 700)   # cutters also clear the 0.3 mm overlap strip into the released crown
add(cyl(LM, R_LMrec, SEAT_LM, TH + 5) ^ prism(above, SEAT_LM - 1, TH + 6))                 # L22 flush recess (above the cut)
for pxy in PILOT_XY:                                                                          # released stepped M5 insert bores: D6.5 x 2.0 then D6.4, 6.2 deep from the seat
    add(cyl(pxy, 3.25, SEAT_LM - 2.0, SEAT_LM + 1)); add(cyl(pxy, 3.2, SEAT_LM - 6.2, SEAT_LM - 1.9))
add(cyl(MU, R_UMrec, SEAT_UM, TH + 5)); add(cyl(MU, R_UMbore, ZV - 5, TH + 5))               # MU10 flush recess + opening
for ang in UM_PILOT_ANG:
    add(cyl(MU + UM_PILOT_R * np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))]), UM_PILOT_D / 2, SEAT_UM - UM_PILOT_DEPTH, SEAT_UM + 1))
add(cyl(T, THROAT, ZV - 5, TH + 5)); add(cyl(t, THROAT, ZV - 5, TH + 5))
add(cyl(T, R_SKIRT, ZB_T - 15, LAND_T)); add(cyl(t, R_SKIRT, LAND_t, TH + 15))
add(cyl(T, R_CAPB, ZB_T - 15, ZB_T + CH_D)); add(cyl(t, R_CAPB, TH - CH_D, TH + 15))   # V10: wiring chambers (T rear, t front)                  # tweeter cavities: T from the vase rear up to its flange plane, t from the front face down to its flange plane
def key_arc(c, a0, z0, z1):
    """cable keyway as an arc slot (width KEY_D at radius R_KEY) from a0-2 to a0+18 degrees: the cap blister sweeps it while twisting to lock"""
    out = None
    for a in np.radians(np.arange(-2, 18.5, 1.0)) + a0:
        k = cyl(c + R_KEY * np.array([math.cos(a), math.sin(a)]), KEY_D / 2, z0, z1, n=48); out = k if out is None else out + k
    return out
# (V4: no keyways - the leads live inside the cavities, tubes enter the cavity walls)                 # cable keyways (t's continues down to its tube)


def bayonet(c, a0, face_z, into):
    """3 L-pockets in the skirt bore wall: axial entry slot from the face (6.8 wide, 3.2 deep), then a 25-degree circumferential slot
    at depth 1.8..3.0 under a 1.8 mm ledge. Lugs on the cap skirt (1.0 radial) ride under the ledge after a twist."""
    out = None
    hull = lambda q: to_m(trimesh.convex.convex_hull(np.array(q, float)))
    for k in range(LUG_N):
        a = a0 + math.radians(60 + 120 * k)
        u_ = np.array([math.cos(a), math.sin(a)]); v_ = np.array([-u_[1], u_[0]])
        q = [[*(c + u_ * rr + v_ * w), face_z + into * dz] for rr in (R_CAPB - 0.5, R_CAPB + 1.2) for w in (-3.4, 3.4) for dz in (-1.0, DISC + 3.2)]
        h = hull(q); out = h if out is None else out + h
        arc = np.radians(np.linspace(-4, 25, 14)) + a
        for a1, a2 in zip(arc[:-1], arc[1:]):
            q = []
            for aa in (a1, a2):
                uu = np.array([math.cos(aa), math.sin(aa)])
                q += [[*(c + uu * rr), face_z + into * dz] for rr in (R_CAPB - 0.5, R_CAPB + 1.2) for dz in (DISC + 1.8, DISC + 3.0)]
            out = out + hull(q)
    return out



# ------------------------------------------------------------------ cable tubes
# T: continue the released D6 duct from its section at the crown cut (-0.16, 306.5, z 8.41, heading (0.68, 0.73)) to T's keyway
TS_REL = [(-79.3, 276.0, 8.3), (-78.4, 278, 8.2), (-76.3, 280, 7.9), (-71.9, 284, 6.4), (-67.1, 288, 3.9), (-61.6, 292, 1.6), (-58.4, 294, 1.0), (-54.8, 296, 0.8), (-52, 297.6, 0.8),
          (-48, 299.1, 1.5), (-44, 300.1, 2.7), (-40, 300.4, 4.4), (-36, 300.2, 6.1), (-32, 299.8, 7.3), (-28, 299.4, 8.1), (-24, 299.2, 8.3), (-16, 300.0, 8.3), (-8, 302.5, 8.3), (0, 306.5, 8.3)]
dir0 = np.array([0.68, 0.73, 0.0]); dir0 /= np.linalg.norm(dir0); p0 = np.array([0.0, 306.5, 8.3])   # V11: D6 on the released centreline
a_kT = math.atan2(Y_CUT - T[1], -0.16 - T[0]); KT = T + (R_SKIRT + 2.5) * np.array([math.cos(a_kT), math.sin(a_kT)])   # tube mouth inside T's cavity wall
dKT = np.array([math.cos(a_kT), math.sin(a_kT)]); KT = T + (R_CAPB + 1.0) * dKT
pT = spline_path(TS_REL + [p0 + dir0 * 8, [p0[0] + dir0[0] * 16, p0[1] + dir0[1] * 16, 8.0], [KT[0] + 8 * dKT[0], KT[1] + 8 * dKT[1], -2.0], [KT[0], KT[1], -15.5], [*(T + (R_CAPB - 2.0) * dKT), -15.5]])   # V10: into T's wiring chamber
# V13 (owner): the V10-V12 T lead dropped 13.5 mm over its last 8 mm (59 deg) into T's wiring chamber. Descend over a longer
# approach instead (<= ~30 deg): right hand spirals down round T's west side (entry 140 deg), the mirrored frame comes
# straight from the ring side (entry -100 deg). Entries stay >= 30 deg from the T->t link (110 deg) so the cap's bayonet
# pockets (entry + 60/180/300) clear it.
def _u(a): return np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
if HAND == "mirror":
    a_kT = math.radians(-100.0); dKT = _u(-100.0); KT = T + (R_CAPB + 1.0) * dKT
    TS_REL = [(79.3, 276.0, 8.3), (78.4, 278, 8.2), (76.3, 280, 7.9), (71.9, 284, 6.4), (67.1, 288, 3.9), (61.6, 292, 1.6)]   # the released route's own shape (mirror of the right hand's)
    _a, _b = np.array([61.6, 292.0]), KT; _mid = []
    for _f in (0.25, 0.5, 0.75):
        _q = _a + (_b - _a) * _f; _mid.append([_q[0], _q[1], 1.6 - 16.6 * (3 * _f ** 2 - 2 * _f ** 3)])
    pT = spline_path(TS_REL + _mid + [[KT[0], KT[1], -15.0], [*(T + (R_CAPB - 2.0) * dKT), -15.5]])
else:
    a_kT = math.radians(140.0); dKT = _u(140.0); KT = T + (R_CAPB + 1.0) * dKT
    pT = spline_path(TS_REL + [p0 + dir0 * 8, [p0[0] + dir0[0] * 16, p0[1] + dir0[1] * 16, 8.0], [8.0, 326.0, 7.5], [5.2, 336.0, 3.5],
                               [*(T + 40.0 * _u(180.0)), -3.5], [*(T + 35.0 * _u(160.0)), -10.5], [KT[0], KT[1], -15.0], [*(T + (R_CAPB - 2.0) * dKT), -15.5]])
_sl = np.degrees(np.arctan2(-np.diff(pT[:, 2]), np.linalg.norm(np.diff(pT[:, :2], axis=0), axis=1)))
print("V13 T lead: steepest descent %.0f deg, entry %.0f deg" % (_sl.max(), math.degrees(a_kT)), flush=True)
UM_R = 4.1
UM_REL = [(77.9, 276.0, 7.2), (76.8, 278, 7.0), (74.7, 280, 6.6), (72.5, 282, 5.9), (70.2, 284, 4.9), (67.8, 286, 3.7), (65.2, 288, 2.4), (62.4, 290, 1.2), (59.1, 292, 0.3), (55.0, 294, -0.2),
          (50.8, 296, -0.3), (46.7, 298, -0.3), (42.5, 300, -0.1), (38.5, 302, 0.2), (35.1, 304, 0.5), (32.6, 306, 0.9), (30.8, 308, 1.1), (26.5, 311, 1.4)]
if HAND == "mirror": UM_REL = [(-77.9, 276.0, 7.2), (-76.8, 278, 7.0), (-74.7, 280, 6.6), (-72.5, 282, 5.9), (-70.2, 284, 4.9), (-67.8, 286, 3.7), (-65.2, 288, 2.4), (-62.4, 290, 1.2), (-59.1, 292, 0.3)]
def exit_rear(path, slope_deg=35.0, r=3.0, step=0.8, smax=90.0):
    """continue a tube from its last point in its plan direction, diving at slope_deg, until it is fully out of the vase rear"""
    P = np.asarray(path, float); dxy = P[-1, :2] - P[-4, :2]; dxy /= np.linalg.norm(dxy); ext = []
    for s_ in np.arange(step, smax, step):
        q = np.r_[P[-1, :2] + dxy * s_, P[-1, 2] - s_ * math.tan(math.radians(slope_deg))]; ext.append(q)
        if q[2] < f_zr([[q[1], q[0]]])[0] - (r + 1.0):
            break
    return np.vstack([P, ext]) if ext else P
pUM_rel = spline_path(UM_REL + UM_NEW); pUM = exit_rear(spline_path(UM_REL + UM_NEW), 58.0, UM_R)   # V12 (owner): exit next to the MU10 terminals instead of behind the T cluster
_term = MU + 37.5 * np.array([math.cos(math.radians(UM_TERM_DEG)), math.sin(math.radians(UM_TERM_DEG))])
_dT = cKDTree(pT).query(pUM[:, :3])[0].min(); _pil = min(np.hypot(*(pUM[:, :2] - (MU + UM_PILOT_R * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))]))).T).min() for a in UM_PILOT_ANG)
print("UM lead leaves the rear at", np.round(pUM[-1], 1), "| %.1f mm from the MU10 terminal strip (283 deg, r 37.5) | min axis distance to the T duct %.1f (need 7.9) | to MU10 pilots %.1f" % (np.linalg.norm(pUM[-1, :2] - _term), _dT, _pil), flush=True)
assert _dT >= 7.9, "UM lead too close to the T duct"
tubeT = tube(pT, TS_R, n=48)
webT = tube(pT, TS_R + 0.8) + tube(pUM_rel, UM_R + 0.8)                                  # released-style duct shells (pipes where the rear is still at 6.8, buried in the bulge elsewhere)
for pxy in PILOT_XY: webT = webT + cyl(pxy, 4.8, 5.3, SEAT_LM - 0.5)                       # released D9.6 insert pads (the bores go below the core rear)   # burial web only matters near the crown where the rear is still at 6.8  # burial web below the core rear; cleared under the cap footprint (the open trough there sits under the cap blister)
# t: from t's keyway (z 10.55 inside the core) to a rear mouth at the crown, exiting with a 30-degree dive
# V9 (owner): no exit pocket, tweeter cables hidden. Both T and t leads come up the LM duct into T's cavity (pT, unchanged);
# a straight D6 tube joins T's cavity wall to t's cavity wall at the depth where the two cavities overlap, so the t lead is
# fished on from T's cavity (T cap off) into t's cavity (t cap off). The t lead has no route of its own any more.
ZTT = 0.5 * (max(ZB_T, LAND_t) + min(LAND_T, TH))
u_tt = (t - T) / np.linalg.norm(t - T)
a_kt = math.atan2(T[1] - t[1], T[0] - t[0])                                   # t's bayonet clocking: the link mouth sits between two pockets (pockets at +-60 and 180 deg from it)
pt = np.array([[*(T + (R_CAPB - 1.5) * u_tt), ZB_T + 5.0], [*(t - (R_CAPB - 1.5) * u_tt), TH - 5.0]])   # V10: straight link from T's rear chamber up to t's front chamber
def surface_cross(path):
    Pp = np.asarray(path); zr_ = f_zr(np.c_[Pp[:, 1], Pp[:, 0]]); k = int(np.argmax(Pp[:, 2] < zr_)); return Pp[k], float(zr_[k])
Su, zu_ = surface_cross(pUM)
mouth = Su[:2]
def poly_dist(A, B):
    from scipy.spatial import cKDTree
    return cKDTree(np.asarray(B)).query(np.asarray(A))[0]
print("V10 T-t link: z %.1f (shared cavity band %.1f..%.1f), length %.1f mm | clearance to T duct %.1f, to MU10 duct %.1f | MU10 lead exit %s (rear z %.1f)"
      % (ZTT, max(ZB_T, LAND_t), min(LAND_T, TH), np.linalg.norm(pt[1] - pt[0]), poly_dist(np.linspace(pt[0], pt[1], 60), pT).min(), poly_dist(np.linspace(pt[0], pt[1], 60), pUM).min(), np.round(Su, 1), zu_), flush=True)
tubet = to_m(trimesh.creation.cylinder(radius=TUBE_R, segment=[pt[0], pt[-1]], sections=48))   # straight link: one cylinder (a chain of collinear capsules leaves coincident faces)
add(tubeT); add(tubet); add(tube(pUM, UM_R))
add(bayonet(T, a_kT, ZB_T, +1)); add(bayonet(t, a_kt, ZB_t, -1))

LANE_PTS = [(-12.0, 7.5, -78.5), (-12.0, 7.5, -60.0), (-12.0, 7.5, -30.0), (-12.0, 9.0, -15.0), (-12.1, 12.0, -10.4), (-12.4, 18.1, -5.0), (-12.8, 24.0, -1.6),
            (-14.0, 36.0, 2.0), (-15.8, 48.0, 3.9), (-18.3, 60.0, 4.9), (-21.9, 72.0, 5.2), (-26.4, 84.0, 4.0), (-36.6, 96.0, 4.8), (-58.5, 108.0, 1.1),
            (-74.3, 120.0, 7.4), (-85.6, 132.0, 8.3), (-94.0, 144.0, 8.3), (-100.3, 156.0, 8.3), (-104.9, 168.0, 8.3), (-107.9, 180.0, 6.6), (-109.6, 192.0, 1.8),
            (-109.9, 204.0, 0.8), (-108.9, 216.0, 4.1), (-106.5, 228.0, 8.0), (-102.8, 240.0, 8.3), (-97.4, 252.0, 8.3), (-90.0, 264.0, 8.3), (-80.3, 276.0, 8.3), (-78.4, 278.0, 8.2), (-76.3, 280.0, 7.9)]
PILOTS_REL = [(-104.75, 200.981), (-52.375, 110.265)]                                            # released M5 insert pads the lane passes behind (180 and 240 deg)
def ring_clamp(x_, y_, r_max=108.6):
    """keep the D8.2 duct and its 0.8 wall inside the released ring's outer face (R113.94, wing contact)"""
    r_ = math.hypot(x_, y_ - LM[1])
    return (x_ * r_max / r_, LM[1] + (y_ - LM[1]) * r_max / r_) if r_ > r_max else (x_, y_)
def lane_pt(x_, y_, z_):
    if y_ > 105.0: x_, y_ = ring_clamp(x_, y_)
    z_ = min(z_, 7.3) if y_ > 115.0 else z_                                                      # ring run: stay 0.8 below the L22 seat
    if 50.0 < y_ < 100.0: z_ = max(z_, 5.3)                                                      # stem: the released lane sits 0.45 above the stem rear face; D8.2 needs >= 0.8
    if min(math.hypot(x_ - px, y_ - py) for px, py in PILOTS_REL) < 9.7: z_ = min(z_, 0.4)       # behind a D9.6 insert pad (bottom z 5.3)
    return (x_, y_, z_)
LANE = spline_path([lane_pt(*p_) for p_ in LANE_PTS])
json.dump(dict(MU=MU.tolist(), T=T.tolist(), t=t.tolist(), T_rim=T_rim, t_rim=t_rim, a_kT=a_kT, a_kt=a_kt, KT=KT.tolist(), Kt=pt[-1, :2].tolist(), LAND_T=LAND_T, LAND_t=LAND_t,
               R_SKIRT=R_SKIRT, R_CAPB=R_CAPB, CH_D=CH_D, TS_R=TS_R, tubeLane=LANE.tolist(), R_KEY=R_KEY, ZB_T=ZB_T, ZB_t=ZB_t, BOSS=BOSS, ZV=ZV, DISC=DISC, KEY_D=KEY_D, TH=TH, ZR=ZR, THROAT=THROAT, WG=WG, Y_CUT=Y_CUT, Y2=Y2, tubeUM=pUM.tolist(), plan=np.asarray(plan.exterior.coords).tolist(),
               tubeT=pT.tolist(), tubet=pt.tolist(), mouth_t=mouth.tolist()), open(f"{d}/{OUT}_build.json", "w"), indent=1)

# ------------------------------------------------------------------ assemble with the released LM carrier
rel = trimesh.load(f"{d}/released_lm_keyed_top_floor_design.stl"); rel.merge_vertices(digits_vertex=4); RM = to_m(rel)   # V11: the released keyed TOP half (seam y 172.5, pins) is the base; the released bottom half is reused unchanged
keep = prism(box(-300, -50, 300, Y2), -200, 40)                                               # V6: released LM only below y 278
rel_keep = RM ^ keep
body = (Sm + webT) - cut
body = body + rel_keep
ANG_NEW = {64.0: 9.0, 116.0: 171.0}   # V13: 40/140 sat on the released T and MU10 lumens (roof -0.65/-1.10 mm into them); 9/171 is the highest ring position with >= 1.5 mm to both, where they dive behind the L22 insert pads
# V13 (owner: "the tweeters duct is strangled"): the vase ducts start from hand-copied approximations of the released
# lumen centrelines, and their burial shells (webT, unioned before the cut) reach a few mm below y 278, so they filled
# part of the released lumens at the junction (y ~267-279, since V11). Re-cut the released lumens along their own
# contract polylines through the junction band.
_rc = json.load(open(os.environ["LX_REL_CONTRACT"]))["duct_collision_contract"]["regions"]
for _r in _rc:
    if _r["name"] not in ("um_route_lumen", "t_route_lumen"): continue
    _P = np.array(_r["points_xyz_mm"], float)
    if HAND == "mirror": _P[:, 0] *= -1
    _k = np.flatnonzero((_P[:, 1] > 262.0) & (_P[:, 1] < 282.0))
    if len(_k) >= 2:
        body = body - tube(_P[max(_k[0] - 1, 0): _k[-1] + 2], _r["radius_mm"])
        print(f"V13: released {_r['name']} (D{2 * _r['radius_mm']:.1f}) re-cut through the y 262-282 junction", flush=True)
# V13: no backing blocks. At 9/171 deg the stations sit in the solid collar (0.57 mm skins, as released); the V9 blocks had
# been unioned after the ducts were cut and partly filled the released lumens.
out = to_t(body); out.merge_vertices(merge_tex=True, merge_norm=True)
parts = out.split(only_watertight=False); main = max(parts, key=lambda p: abs(p.volume))
voids = [p for p in parts if p is not main and abs(p.volume) > 10 and main.contains(p.vertices[:1]).all()]   # released captive magnet cavities (internal shells)
print("fragment locations (>0.1 mm3):", [np.round(p.vertices.mean(0), 1).tolist() for p in parts if p is not main and p not in voids and abs(p.volume) > 0.1], flush=True)
print("fragments dropped (mm3):", sorted([round(abs(p.volume), 2) for p in parts if p is not main and p not in voids], reverse=True)[:8], "| internal voids kept:", len(voids), flush=True)
ANG_NEW = {64.0: 9.0, 116.0: 171.0}   # V13: 40/140 sat on the released T and MU10 lumens (roof -0.65/-1.10 mm into them); 9/171 is the highest ring position with >= 1.5 mm to both, where they dive behind the L22 insert pads
src = trimesh.load(f"{d}/released_lm_keyed_top_floor_design.stl").split(only_watertight=False)
for p in src:
    c = p.vertices.mean(0); a = math.degrees(math.atan2(c[1] - LM[1], c[0] - LM[0]))
    for a0, a1 in ANG_NEW.items():
        if abs(p.volume) < 100 and abs(a - a0) < 1.0:
            q = p.copy(); q.apply_transform(trimesh.transformations.rotation_matrix(math.radians(a1 - a0), [0, 0, 1], [LM[0], LM[1], 0]))
            ok_in = main.contains(q.vertices).all(); dmin = abs(trimesh.proximity.signed_distance(main, q.vertices)).min()
            print(f"LM wing magnet station {a0:.0f} -> {a1:.0f} deg: cavity fully inside material {ok_in}, min wall to any surface {dmin:.2f} mm", flush=True)
            voids.append(q)
final = trimesh.util.concatenate([main] + voids)
print("BODY watertight", main.is_watertight, "| pieces", len(parts), "| volume cm3", round(main.volume / 1000, 1), "| bounds", main.bounds.round(1).tolist(), flush=True)
final.export(f"{d}/{OUT}.stl")
np.savez(f"{d}/{OUT}_skin.npz", XY=XY, F=F, R=R, tri=tri)
