"""vase25 caps: vase24 caps re-derived from the rim-centred vendor ND25FN-4 seated on the MEASURED land (land, flange stack, C-gap, felt pad),
plus 1 mm tab-tip pockets swept over the twist. Seamless twist-lock caps over the back of each ND25FN-4. Cap skins are cut from the SAME surface fields as the body, so they continue the face exactly. Mesh level (manifold)."""
import sys, json, numpy as np, trimesh, manifold3d as m3, shapely
from shapely.geometry import Polygon
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from trimesh.transformations import rotation_matrix, translation_matrix
import os
AZA, AZB = (-89.7, -177.6) if os.environ.get('LX_HAND', 'std') == 'mirror' else (89.7, 177.6)   # tab azimuths in the (possibly local-z mirrored) vendor model
d = sys.argv[1]; C = json.load(open(f"{d}/c7.json")); PL = json.load(open(f"{d}/tweeter_placement.json")); LANDM = PL["land"]; TH, ZR, WG, LAND = C["TH"], C["ZR"], C["WG_DEPTH"], C["LAND_T"]; T, t = np.array(C["T"]), np.array(C["t"]); GAP = C["GAP"]; SKIN = C["CAP_SKIN"]
R_BORE, R_CAP, PART, R_SL_O, R_SL_I = C["SEAT_D"] / 2, 26.0, .15, C["SEAT_D"] / 2 - .25, 17.4
import trimesh as _tm
_V = _tm.load(f"{d}/nd25fn4_vendor_rimcentred.stl"); _n = _V.face_normals; _c = _V.triangles_center; _r = np.hypot(_c[:, 1], _c[:, 2])
_k = (_n[:, 0] > 0.99) & (_c[:, 0] > 10) & (_c[:, 0] < 14) & (_r > 16.5) & (_r < 20)
RIM_BACK_X = float(np.average(_c[_k, 0], weights=_V.area_faces[_k])); FL_STACK_REAL = RIM_BACK_X - PL["XF"]
print(f"vendor flange stack: bearing ring x {PL['XF']:.2f} -> rim back x {RIM_BACK_X:.2f} = {FL_STACK_REAL:.2f} mm")
SKIRT_H, FOAM, FL_STACK, LUG_W, LUG_H, LUG_R, TRAVEL, LAND_TH, PRELOAD = 6.0, 1.5, 6.0, 6.0, 2.5, 23.75, 25.0, 2.2, .15
to_m = lambda q: m3.Manifold(m3.Mesh(vert_properties=np.asarray(q.vertices, np.float32), tri_verts=np.asarray(q.faces, np.uint32)))
def to_t(M): mm = M.to_mesh(); q = trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False); return q
S5 = lambda u: (lambda v: v * v * v * (v * (v * 6 - 15) + 10))(np.clip(u, 0, 1))
def slab(X, Y, ztop, zbot):
    ny, nx = X.shape; idx = np.arange(nx * ny).reshape(ny, nx); N = nx * ny; V = np.vstack([np.c_[X.ravel(), Y.ravel(), np.broadcast_to(ztop, X.shape).ravel()], np.c_[X.ravel(), Y.ravel(), np.broadcast_to(zbot, X.shape).ravel()]])
    a, b, c, e = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel(); F = [np.c_[a, b, c], np.c_[a, c, e], np.c_[a + N, c + N, b + N], np.c_[a + N, e + N, c + N]]
    for loop in (idx[0, :], idx[:, -1], idx[-1, ::-1], idx[::-1, 0]): l0, l1 = loop[:-1], loop[1:]; F += [np.c_[l0, l0 + N, l1 + N], np.c_[l0, l1 + N, l1]]
    q = trimesh.Trimesh(V, np.vstack(F), process=False); q.fix_normals(); return q
def field(c, fn, half=32.0, h=.4):
    xs = np.arange(c[0] - half, c[0] + half + h, h); ys = np.arange(c[1] - half, c[1] + half + h, h); X, Y = np.meshgrid(xs, ys); return X, Y, fn(X, Y)
FS = np.load(f"{d}/front_surface.npz"); f_front = RegularGridInterpolator((FS["ys"], FS["xs"]), FS["F"], bounds_error=False, fill_value=TH)
RS = np.load(f"{d}/rear_surface.npz"); f_rear0 = RegularGridInterpolator((RS["ys"], RS["xs"]), RS["zrear"], bounds_error=False, fill_value=0.0); plan = Polygon(C["plan"]).buffer(0)
def rear_total(X, Y):                                               # body's real rear surface: sculpted rear, or the rear tweeter's (rim-rounded) flare where that is deeper. Same formulas as the body build.
    h = X[0, 1] - X[0, 0]; rho = np.hypot(X - t[0], Y - t[1]); g = np.where(rho < 50.0, WG - WG * (np.clip((rho - 15.5) / 34.5, 0, 1)) ** (1 / 2.1), 0.0); gs = gaussian_filter(g, sigma=5.0 / h, mode="nearest")
    de = shapely.distance(shapely.points(X.ravel(), Y.ravel()), plan.exterior).reshape(X.shape); kT = S5((np.hypot(X - T[0], Y - T[1]) - 31.0) / 8.0); w = S5((rho - 36.0) / 8.0) * (1 - S5((rho - 56.0) / 8.0)) * S5((de - 1.5) / 4.0) * kT
    zr0 = f_rear0(np.c_[Y.ravel(), X.ravel()]).reshape(X.shape); shape = g / WG; old_s = np.maximum(zr0, ZR + g); znew = zr0 + np.maximum(ZR + WG - zr0, 0) * shape      # same riding-flare formula as the body build
    rw2 = S5((rho - 30.0) / 10.0) * (1 - S5((rho - 58.0) / 10.0)) * S5((de - 1.5) / 4.0) * kT; dl_ = znew - zr0; return np.maximum(znew + rw2 * (gaussian_filter(dl_, sigma=5.0 / h, mode="nearest") - dl_), old_s)
def cyl(c, r, z0, z1, sec=192): q = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=sec); q.apply_translation((c[0], c[1], (z0 + z1) / 2)); return q
def ann(c, r0, r1, z0, z1): q = trimesh.creation.annulus(r_min=r0, r_max=r1, height=z1 - z0, sections=192); q.apply_translation((c[0], c[1], (z0 + z1) / 2)); return q
def wedge(c, a_mid, width, z0, z1, R=40.0):
    a = np.radians(np.linspace(a_mid - width / 2, a_mid + width / 2, 24)); P = [(0, 0)] + [(R * np.cos(x), R * np.sin(x)) for x in a]; q = trimesh.convex.convex_hull(np.array([(x, y, z) for x, y in P for z in (z0, z1)], float)); q.apply_translation((c[0], c[1], 0)); return q
def rbox(c, ang, r0, r1, wid, z0, z1):
    q = trimesh.creation.box(extents=(r1 - r0, wid, z1 - z0)); q.apply_translation(((r0 + r1) / 2, 0, (z0 + z1) / 2)); q.apply_transform(rotation_matrix(np.radians(ang), [0, 0, 1])); q.apply_translation((c[0], c[1], 0)); return q
def make(c, key, on_front):
    s = 1.0 if on_front else -1.0                                   # s: direction from the cap's ledge plane toward the outside
    face = TH if on_front else ZR; zl = face - s * SKIN; zland = LANDM[key]; fl_back = zland + s * FL_STACK_REAL; sl_end = fl_back + s * FOAM; DV = DRV.vertices[np.hypot(DRV.vertices[:, 0] - c[0], DRV.vertices[:, 1] - c[1]) < 30]; rr = np.hypot(DV[:, 0] - c[0], DV[:, 1] - c[1]); back = (DV[:, 2] > zland + FL_STACK + .5) if on_front else (DV[:, 2] < zland - FL_STACK - .5)
    rel = (np.degrees(np.arctan2(DV[:, 1] - c[1], DV[:, 0] - c[0])) - GAP[key] + 180) % 360 - 180; rel = rel[back & (rr > 18.6)]; tmin, tmax = float(rel.min()), float(rel.max())
    g0, g1 = tmin - 8.0, tmax + 8.0 + TRAVEL; GC, GW = GAP[key] + (g0 + g1) / 2, (g1 - g0); solid = 360.0 - GW; lugs = [GAP[key] + g1 + solid * f for f in (.09, .5, .91)]      # measured from the placed driver: the gap clears the terminal block in the locked AND the entry pose; lugs spread over the solid arc
    print(f"   {key}: terminal block spans {tmin:.0f}..{tmax:.0f} deg from the gap direction -> cap gap {g0:.0f}..{g1:.0f} ({GW:.0f} deg), lugs at +{[round(g1 + solid * f) for f in (.09, .5, .91)]}")      # the driver's terminal block reaches R21.5 right up to its back end, so BOTH the skirt and the sleeve are C-shaped; the gap is wide enough to clear the terminals and the cable in the locked AND in the entry pose (25 deg apart)
    zs = lambda a, b: (min(a, b), max(a, b))
    T4 = np.array(PL[key]); W = lambda p: T4[:3, :3] @ np.asarray(p, float) + T4[:3, 3]
    motor_back = W([29.16, 0, 0])[2]; pad_gap = abs(motor_back - zl); print(f"   {key}: land z {zland:.2f}, flange back z {fl_back:.2f}, motor rear face z {motor_back:.2f}, gap to the cap skin {pad_gap:.2f} mm -> R10 boss {pad_gap - 1.0:.2f} mm + 1.5 mm foam pad compressed to 1.0 mm (as Obiwan)")
    tips = [(W([31.43, 19.83 * np.cos(np.radians(AZA)), 19.83 * np.sin(np.radians(AZA))]), 3.0, 19.83), (W([29.86, 19.34 * np.cos(np.radians(AZB)), 19.34 * np.sin(np.radians(AZB))]), 5.4, 19.34)]
    X, Y, Z = field(c, (lambda X, Y: f_front(np.c_[Y.ravel(), X.ravel()]).reshape(X.shape)) if on_front else rear_total)
    air = slab(X, Y, TH + 9.0, Z) if on_front else slab(X, Y, Z, ZR - 9.0)                        # everything outside the body's own surface
    skin = to_m(cyl(c, R_CAP, *zs(zl, face + s * 3.0))) - to_m(air)
    skirt = to_m(ann(c, R_SL_I, R_SL_O, *zs(zl + s * .01, sl_end))) - to_m(wedge(c, GC, GW, *zs(zl + s * .05, sl_end - s * 1.0))); sleeve = m3.Manifold()      # gap cut now clears the skirt start: vase24 left a 0.01 mm film across the gap (the caps' second, open shell)
    cap = skin + skirt + sleeve
    cap = cap + to_m(cyl(c, 10.0, *zs(zl + s * 0.5, zl - s * (pad_gap - 1.0))))      # Obiwan-style pad boss, 0.5 mm into the skin
    for a in lugs: cap = cap + to_m(rbox(c, a, R_SL_O - .6, LUG_R, LUG_W, *zs(zl - s * (LAND_TH + .2), zl - s * (LAND_TH + .2 + LUG_H))))
    pocket = to_m(cyl(c, R_CAP + PART, *zs(zl, face + s * 6.0))); slots = None
    for a in lugs:
        cut = to_m(rbox(c, a - TRAVEL, R_SL_O - .8, LUG_R + .35, LUG_W + 1.2, *zs(zl + s * .01, zl - s * (LAND_TH + .2 + LUG_H + .3))))                      # axial entry notch
        for th in np.arange(a - TRAVEL, a + 1.01, 2.0):
            k = np.clip((th - (a - TRAVEL)) / TRAVEL, 0, 1); cut = cut + to_m(rbox(c, th, R_SL_O - .8, LUG_R + .35, LUG_W + 1.2, *zs(zl - s * (LAND_TH + (.2 + PRELOAD) * k), zl - s * (LAND_TH + .2 + LUG_H + .3))))   # ramped channel: 0.2 mm free at entry -> 0.15 mm interference at lock
        slots = cut if slots is None else slots + cut
    foam = (to_m(ann(c, R_SL_I, R_SL_O, *zs(sl_end, fl_back))) - to_m(wedge(c, GC, GW, *zs(sl_end - s * .5, fl_back + s * .5)))) + to_m(cyl(c, 10.0, *zs(zl - s * (pad_gap - 1.0), zl - s * pad_gap)))      # foam ring under the sleeve + a felt pad between the cap skin and the motor's back (centred preload, and it damps the skin)
    for tip, w, rt in tips:                                         # 1 mm tip pockets in the skin's inner face, swept over the twist (built locked; the entry pose is -TRAVEL)
        at = np.degrees(np.arctan2(tip[1] - c[1], tip[0] - c[0])); half = np.degrees((w / 2 + 1.0) / rt)
        a0, a1 = at - 3 - half, at + TRAVEL + 3 + half; aa = np.radians(np.linspace(a0, a1, 40))
        poly = np.array([(c[0] + (rt + 2.0) * np.cos(q), c[1] + (rt + 2.0) * np.sin(q)) for q in aa] + [(c[0] + (rt - 2.0) * np.cos(q), c[1] + (rt - 2.0) * np.sin(q)) for q in aa[::-1]])
        z0, z1 = zs(zl, zl + s * 1.0); zlo, zhi = (z0 - 0.01, z1 + 0.1) if s < 0 else (z0 - 0.1, z1 + 0.01); pk = m3.Manifold.extrude(m3.CrossSection([poly.tolist()]), zhi - zlo).translate((0, 0, zlo)); cap = cap - pk      # overshoot both faces so the pocket opens on the inner face for either cap orientation
        print(f"   {key}: tip pocket for the {w} mm tab at {at:.1f} deg, arc {a0:.1f}..{a1:.1f}, tip z {tip[2]:.2f} vs skin inner plane {zl:.2f}")
    return cap, pocket + slots, foam, dict(face=face, zl=zl, sl_end=sl_end, fl_back=fl_back, lugs=lugs, sleeve_len=abs(sl_end - zl))
DRV = trimesh.load(f"{d}/drivers_nd25fn4_placed.stl"); body = trimesh.load(f"{d}/_body_keyed.stl"); B = to_m(body); info = {}
for key, c, on_front, nm in (("T", T, False, "cap_front_tweeter_REAR_FACE"), ("t", t, True, "cap_rear_tweeter_FRONT_FACE")):
    cap, cut, foam, inf = make(c, key, on_front); B = B - cut; _ct = to_t(cap); _ct.merge_vertices(); _ct.update_faces(_ct.nondegenerate_faces(height=1e-6)); _ct.remove_unreferenced_vertices(); _ct = max(_ct.split(only_watertight=False), key=lambda q: abs(q.volume)); _ct.export(f"{d}/{nm}.stl"); print(f"   {nm}: exported solid watertight {_ct.is_watertight}");      # weld, drop the zero-volume flap left by the skin boolean
    to_t(foam).export(f"{d}/foam_ring_{'frontT' if key == 'T' else 'reart'}.stl"); info[key] = dict(inf, name=nm, cap_cm3=round(cap.volume() / 1000, 2)); info[key]["_cap"] = cap
    # (foam files are never named _T/_t: they collide on the case-insensitive Mac disk)
out = to_t(B); out.merge_vertices(digits_vertex=5); out = max(out.split(only_watertight=False), key=lambda q: len(q.faces)); out.export(f"{d}/um_vase_nd25fn4_c25_body.stl"); print("body with cap pockets and twist-lock slots: watertight", out.is_watertight, "volume cm3", round(out.volume / 1000, 1))
Bf = to_m(out); drv = trimesh.load(f"{d}/drivers_nd25fn4_placed.stl") if False else None
for key, c in (("T", T), ("t", t)):
    cap = info[key].pop("_cap"); lock = (cap ^ Bf).volume(); q = to_t(cap); q.apply_transform(translation_matrix([-c[0], -c[1], 0])); q.apply_transform(rotation_matrix(np.radians(-25.0), [0, 0, 1])); q.apply_transform(translation_matrix([c[0], c[1], 0])); entry = (to_m(q) ^ Bf).volume()
    info[key].update(interference_locked_mm3=round(lock, 1), interference_entry_pose_mm3=round(entry, 2)); print(key, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in info[key].items() if k != "lugs"})
json.dump(info, open(f"{d}/caps.json", "w"), indent=1)
