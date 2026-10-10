"""V11 caps = V10 caps + a central pad boss (1.5 mm foam compressed to 1.0 mm on the tweeter rear face) and 1 mm tip pockets over the soldered ND25FN-4 tabs (swept over the 16 deg twist).
usage: caps_v11.py DIR VARIANT PLACEMENT_JSON

V4 flush caps: a 2 mm disc flush with the body surface + skirt into the cavity, 3 bayonet lugs, two spanner holes. Same part for T (rear) and t (front)."""
import sys, json, math, numpy as np, trimesh, manifold3d as m3
d = sys.argv[1]; V = sys.argv[2] if len(sys.argv) > 2 else "obiwan_v5_single_piece"; C = json.load(open(f"{d}/{V}_build.json")); P = json.load(open(sys.argv[3])); R_SKIRT, DISC = C.get("R_CAPB", C["R_SKIRT"]), C["DISC"]
CLR, WALL_S, INTO = 0.2, 1.1, 3.4; R_OUT, R_IN = R_SKIRT - CLR, R_SKIRT - CLR - WALL_S   # thin skirt: the ND25FN-4 terminal tabs reach r 21.6
def to_m(q): return m3.Manifold(m3.Mesh(vert_properties=np.asarray(q.vertices, np.float32), tri_verts=np.asarray(q.faces, np.uint32)))
def to_t(M): mm = M.to_mesh(); return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)
def cyl(r, z0, z1, n=192): q = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=n); q.apply_translation((0, 0, (z0 + z1) / 2)); return to_m(q)
hull = lambda q: to_m(trimesh.convex.convex_hull(np.array(q, float)))
# local frame: cap datum at z=0, +z into the body. The disc's outer skin follows the body surface (field sampled around the cavity), thickness DISC.
from scipy.interpolate import LinearNDInterpolator
Z = np.load(f"{d}/{V}_skin.npz"); XY = Z["XY"]
def field_solid(c, key, sign, offset, half=30.0, h=0.5, zfar=80.0):
    """solid ABOVE (sign=+1, +z world into the body for T) the surface (field + offset) over a square patch: used as a cutter"""
    f = LinearNDInterpolator(XY, Z[key]); xs = np.arange(c[0] - half, c[0] + half + h, h); ys = np.arange(c[1] - half, c[1] + half + h, h); GX, GY = np.meshgrid(xs, ys)
    z = f(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape); z = np.where(np.isnan(z), np.nanmean(z), z) + offset
    nx, ny = len(xs), len(ys); idx = np.arange(nx * ny).reshape(ny, nx)
    tri = np.concatenate([np.c_[idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel()], np.c_[idx[:-1, :-1].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()]])
    V1 = np.c_[GX.ravel(), GY.ravel(), z.ravel()]; V2 = np.c_[GX.ravel(), GY.ravel(), np.full(z.size, c[2] + sign * zfar)]
    n = len(V1); e = np.concatenate([idx[0, :], idx[:, -1], idx[-1, ::-1], idx[::-1, 0]]); e = np.array([e[i] for i in range(len(e)) if i == 0 or e[i] != e[i - 1]]); ring = np.append(e, e[0])
    side = np.concatenate([np.c_[ring[:-1], ring[1:] + n, ring[:-1] + n], np.c_[ring[:-1], ring[1:], ring[1:] + n]])
    faces = np.concatenate([tri[:, ::-1] if sign > 0 else tri, (tri + n) if sign > 0 else (tri + n)[:, ::-1], side if sign > 0 else side[:, ::-1]])
    m = trimesh.Trimesh(np.vstack([V1, V2]), faces, process=False)
    if m.volume < 0: m.invert()
    return to_m(m)
def make(c, key, sign, datum):
    """cap in WORLD coordinates around cavity centre c: skin between surface and surface+DISC*sign, skirt + lugs measured from the datum plane"""
    cz = np.array([c[0], c[1], datum]); above = field_solid(cz, key, sign, 0.0); above2 = field_solid(cz, key, sign, sign * DISC)
    lo, hi = (datum - 3, datum + DISC + INTO + 12) if sign > 0 else (datum - DISC - INTO - 12, datum + 3)
    disc = (cyl(R_OUT, lo, hi).translate((c[0], c[1], 0)) ^ above) - above2
    zs0, zs1 = (datum + DISC - 0.01, datum + DISC + INTO) if sign > 0 else (datum - DISC - INTO, datum - DISC + 0.01)
    skirt = (cyl(R_OUT, zs0, zs1) - cyl(R_IN, zs0 - 1, zs1 + 1)).translate((c[0], c[1], 0))
    cap = (disc + skirt) - above                                                        # nothing outside the body surface
    for k in range(3):
        a = math.radians(60 + 120 * k); u = np.array([math.cos(a), math.sin(a)]); v = np.array([-u[1], u[0]])
        zl = (datum + DISC + 1.9, datum + DISC + 2.9) if sign > 0 else (datum - DISC - 2.9, datum - DISC - 1.9)
        cap = cap + hull([[*(c[:2] + u * rr + v * w), z] for rr in (R_OUT - 0.3, R_OUT + 1.0) for w in (-3.0, 3.0) for z in zl])
    for sx in (-7, 7):
        cap = cap - cyl(1.25, lo - 1, hi + 1).translate((c[0] + sx, c[1], 0))
    return cap

# --- V11 additions (vendor ND25FN-4 model, aligned: x = depth behind the dome apex) ---
REAR_FACE_X = 29.16           # flat rear face of the motor/heatsink, r <= ~11 (vendor model)
PAD_COMPRESSED, BOSS_R = 1.0, 10.0
TABS = ((89.7, 31.43, 3.0, 19.83), (177.6, 29.86, 5.4, 19.34))   # (azimuth, tip x, width, mean radius) of tabs A and B, rim-centred vendor model
import os
if os.environ.get("LX_HAND", "std") == "mirror": TABS = tuple((-az, x, w, r) for az, x, w, r in TABS)   # x-mirrored frame: vendor model mirrored in its local z
TAB_R0, TAB_R1 = 18.8, 22.4   # radial band of the tip pockets (tabs sit at r 19.6..21.0)
TWIST, TW_MARGIN, POCKET_D = 16.0, 3.0, 1.0
def world(T4, p): return (np.array(T4)[:3, :3] @ np.asarray(p, float)) + np.array(T4)[:3, 3]
def sector_prism(c, r0, r1, a0, a1, z0, z1, n=48):
    """annular sector between angles a0..a1 (rad, world) as a manifold prism z0..z1"""
    aa = np.linspace(a0, a1, n); outer = [(c[0] + r1 * math.cos(a), c[1] + r1 * math.sin(a)) for a in aa]; inner = [(c[0] + r0 * math.cos(a), c[1] + r0 * math.sin(a)) for a in aa[::-1]]
    poly = np.array(outer + inner); return m3.Manifold.extrude(m3.CrossSection([poly.tolist()]), z1 - z0).translate((0, 0, z0))
def extras(cap, c, key, sign, datum, T4):
    rear_z = world(T4, [REAR_FACE_X, 0, 0])[2]
    # boss from 0.5 mm inside the disc to PAD_COMPRESSED short of the tweeter rear face
    if sign > 0: zb0, zb1 = datum + DISC - 0.5, rear_z - PAD_COMPRESSED
    else:        zb0, zb1 = rear_z + PAD_COMPRESSED, datum - DISC + 0.5
    boss = cyl(BOSS_R, zb0, zb1).translate((c[0], c[1], 0)); cap = cap + boss
    print(f"   pad boss R{BOSS_R}: z {min(zb0, zb1):.2f}..{max(zb0, zb1):.2f} (tweeter rear face z {rear_z:.2f}; {PAD_COMPRESSED} mm left for the pad)")
    # tip pockets: cap material within POCKET_D of the disc inner face, over the tab tips, swept over the twist
    shell = field_solid(np.array([c[0], c[1], datum]), key, sign, sign * (DISC - POCKET_D))
    for az, tip_x, w, rt in TABS:
        tip = world(T4, [tip_x, rt * math.cos(math.radians(az)), rt * math.sin(math.radians(az))])
        a = math.atan2(tip[1] - c[1], tip[0] - c[0]); half = (w / 2 + 1.0) / rt
        a0, a1 = a - math.radians(TWIST + TW_MARGIN) - half, a + math.radians(TW_MARGIN) + half   # cap turns +TWIST: the tip moves -TWIST in the cap frame
        zlo, zhi = (datum - 5, datum + DISC + 5) if sign > 0 else (datum - DISC - 5, datum + 5)
        cap = cap - (sector_prism(c, TAB_R0, TAB_R1, a0, a1, zlo, zhi) ^ shell)
        print(f"   tip pocket for the {w:.1f} mm tab at {math.degrees(a):.1f} deg (world), arc {math.degrees(a0):.1f}..{math.degrees(a1):.1f}, depth {POCKET_D} mm; tab tip z {tip[2]:.2f}")
    return cap
cap = None
# lug angles must match the body's pockets: pockets sit at a_key + 60 + 120k (world); the caps are built in world coordinates, so rotate the lug frame accordingly
import types
def build(name, c, key, sign, datum, a_key, T4):
    global make
    src = make
    def make_rot(c, key, sign, datum):
        cap = None
        cz = np.array([c[0], c[1], datum]); above = field_solid(cz, key, sign, 0.0); above2 = field_solid(cz, key, sign, sign * DISC)
        lo, hi = (datum - 3, datum + DISC + INTO + 12) if sign > 0 else (datum - DISC - INTO - 12, datum + 3)
        disc = (cyl(R_OUT, lo, hi).translate((c[0], c[1], 0)) ^ above) - above2
        zs0, zs1 = (datum + DISC - 0.01, datum + DISC + INTO) if sign > 0 else (datum - DISC - INTO, datum - DISC + 0.01)
        skirt = (cyl(R_OUT, zs0, zs1) - cyl(R_IN, zs0 - 1, zs1 + 1)).translate((c[0], c[1], 0))
        cap = (disc + skirt) ^ above                                                        # keep only what lies inside the body surface
        for k in range(3):
            a = a_key + math.radians(60 + 120 * k); u = np.array([math.cos(a), math.sin(a)]); v = np.array([-u[1], u[0]])
            zl = (datum + DISC + 1.9, datum + DISC + 2.9) if sign > 0 else (datum - DISC - 2.9, datum - DISC - 1.9)
            cap = cap + hull([[*(np.array(c[:2]) + u * rr + v * w), z] for rr in (R_OUT - 0.3, R_OUT + 1.0) for w in (-3.0, 3.0) for z in zl])
        # no spanner holes: the outer skin stays continuous; twist with a rubber grip pad, lift with a mini suction cup
        # (the 1.5 mm foam pad in the 1.0 mm gap pushes the unlocked cap ~0.5 mm proud)
        return extras(cap, c, key, sign, datum, T4)
    cm = to_t(make_rot(c, key, sign, datum)); cm.merge_vertices(merge_tex=True, merge_norm=True); cm = max(cm.split(only_watertight=False), key=lambda p: abs(p.volume))
    print(f"{name}: watertight {cm.is_watertight} volume {cm.volume/1000:.2f} cm3 | outer skin z {cm.bounds[0,2]:.1f}..{cm.bounds[1,2]:.1f}"); cm.export(f"{d}/{name}.stl"); return cm
build(f"{V}_cap_frontT", C["T"], "R", +1, C["ZB_T"], C["a_kT"], P["T"])
build(f"{V}_cap_reart", C["t"], "F", -1, C["ZB_t"], C["a_kt"], P["t"])
