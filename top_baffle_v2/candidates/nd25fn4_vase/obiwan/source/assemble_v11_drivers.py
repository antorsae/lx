"""V11 top with the three drivers in place (no tweeter caps) plus best-guess MU10 terminals/Fastons.

usage: assemble_v11_drivers.py WORKDIR
WORKDIR holds obiwan_v11_top.stl, obiwan_v11_top_build.json, nd25fn4_axis.stl, H1658-04_MU10RB-SL_driver.stl.
Frame: Obi-Wan design frame, X lateral / Y up / Z forward (front face z 18.3).
"""
import sys, json, math, numpy as np, trimesh
d = sys.argv[1]; C = json.load(open(f"{d}/obiwan_v11_top_build.json"))
body = trimesh.load(f"{d}/obiwan_v11_top.stl"); body = max(body.split(only_watertight=False), key=lambda p: abs(p.volume))

# ND25FN-4: same placement as assemble_v10 (flange bearing plane on the land, terminal block clocked to the keyway)
# nd25fn4_vendor_aligned.stl: Dayton's own model (vendor/DAYTON/ND25FN-4/ND25FN-4_3D_File.stl), posed on its rear pot
# face normal (180 mm2, 0.025 mm flat), apex at x=0; axis re-centred on the rim/lug circle (the pot-face centroid was ~1 mm off). The flange front face, which the cap presses on the
# land, is at x=XF (area-weighted fit of the front-facing flats at r 16..21.5).
drv = trimesh.load(f"{d}/nd25fn4_vendor_rimcentred.stl"); v = drv.vertices; r = np.hypot(v[:, 1], v[:, 2])
# The flange front face is slightly bevelled on the scan (x 6.1 near r 16 .. 7.8 at the rim): on the flat land it bears
# at its most forward ring outside the throat radius, so XF = the 0.5th percentile x of the flange at r 15.9..21.5.
fr = (r > 15.9) & (r < 21.5) & (v[:, 0] < 12)
XF = float(np.percentile(v[fr, 0], 0.5))
# Terminal clocking: the two axial Faston tabs sit at 91 and 175 deg on the vendor model; clock their midpoint (133 deg)
# toward each tweeter's cable keyway, as intended since V10.
TERM_AZ = (89.7, 177.6)
blk = math.radians(133.65)
print(f"tweeter model: flange bearing plane XF = {XF:.2f} mm behind the dome apex; terminals at {TERM_AZ} deg, midpoint clocked to the keyway")
def place_tweeter(c, a_key, land_z, front):
    ex = np.array([0, 0, -1.0 if front else 1.0]); best = None
    for phi in np.radians(np.arange(0, 360, 0.5)):
        ey = np.array([math.cos(phi), math.sin(phi), 0]); ez = np.cross(ex, ey); M = np.c_[ex, ey, ez]
        b = M @ np.array([0, math.cos(blk), math.sin(blk)]); err = abs((math.atan2(b[1], b[0]) - a_key + math.pi) % (2 * math.pi) - math.pi)
        if best is None or err < best[0]: best = (err, M)
    M = best[1]; T4 = np.eye(4); T4[:3, :3] = M; T4[:3, 3] = np.array([c[0], c[1], land_z]) - M @ np.array([XF, 0, 0])
    m = drv.copy(); m.apply_transform(T4); m.metadata['T4'] = T4.tolist(); return m
tw_front = place_tweeter(C["T"], C["a_kT"], C["LAND_T"], True)
tw_rear = place_tweeter(C["t"], C["a_kt"], C["LAND_t"], False)

# MU10: released seat transform Pos(x, y, 14.3) * Rot(Z=58) * Rot(X=90), moved to the V11 MU10 centre (same pilot clocking)
MU = np.array(C["MU"]); SEAT = 14.3
mu10 = trimesh.load(f"{d}/H1658-04_MU10RB-SL_driver.stl")
mu10.apply_transform(trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0]))
mu10.apply_transform(trimesh.transformations.rotation_matrix(math.radians(58), [0, 0, 1]))
mu10.apply_translation([MU[0], MU[1], SEAT])

# MU10 terminals, owner measurements 2026-09-24 (flange front face = SEAT + 5.4 = z 19.7):
#   underside of the small terminal PCB 11.46 behind the flange front; PCB + wider tab base plate 3.28 thick;
#   two 2.8 mm Faston tabs (2.78 / 2.76 measured) pointing axially rearward; insulated 90° receptacle ends 16.26 behind the PCB underside.
# Still estimated: tab pitch (19), tab-plane radius (TAB_R, env LX_TAB_R), tab length (7), receptacle body 5.5 x 4.5.
import os
CLOCK = math.radians(283.0); FL = SEAT + 5.4
Z_PCB_FRONT = FL - 11.46; Z_PCB_REAR = Z_PCB_FRONT - 3.28; Z_BOOT_END = Z_PCB_FRONT - 16.26
TAB_R = float(os.environ.get("LX_TAB_R", "37.5")); PITCH = 19.0
u = np.array([math.cos(CLOCK), math.sin(CLOCK), 0.]); w = np.array([-u[1], u[0], 0.]); ez = np.array([0, 0, 1.])
def local_box(r0, r1, t0, t1, z0, z1):
    """Box spanning radius r0..r1, tangential t0..t1, z0..z1 in the MU10 terminal frame."""
    b = trimesh.creation.box([r1 - r0, t1 - t0, z1 - z0])
    R = np.eye(4); R[:3, :3] = np.c_[u, w, ez]; R[:3, 3] = np.r_[MU, 0.] + u * (r0 + r1) / 2 + w * (t0 + t1) / 2 + ez * (z0 + z1) / 2
    b.apply_transform(R); return b
board = local_box(TAB_R - 5.5, TAB_R + 1.5, -12.0, 12.0, Z_PCB_REAR, Z_PCB_FRONT)
tabs = [local_box(TAB_R - 0.25, TAB_R + 0.25, s * PITCH / 2 - 1.4, s * PITCH / 2 + 1.4, Z_PCB_REAR - 7.0, Z_PCB_REAR) for s in (-1, 1)]
rcpt = [local_box(TAB_R - 2.25, TAB_R + 2.25, s * PITCH / 2 - 2.75, s * PITCH / 2 + 2.75, Z_BOOT_END, Z_PCB_REAR) for s in (-1, 1)]
ZT = Z_BOOT_END + 1.5
print(f"MU10 terminals: PCB z {Z_PCB_REAR:.2f}..{Z_PCB_FRONT:.2f}, tabs to z {Z_PCB_REAR-7:.2f}, insulated receptacles z {Z_BOOT_END:.2f}..{Z_PCB_REAR:.2f}, tab plane r {TAB_R}")
# Leads: from each receptacle out to the MU10 lead mouth on the rear (V11 exit near (9.3, 321)), hanging behind the rear
mouth = np.array(C["mouth_t"], float)
hit = body.ray.intersects_location([[mouth[0], mouth[1], -60.]], [[0, 0, 1.]], multiple_hits=False)[0]
z_mouth = float(hit[0][2]) if len(hit) else -5.0
def rear_floor(xy):
    """Lowest body surface z over a 3 mm neighbourhood of each xy (inf where the column is empty)."""
    ring = np.array([[0, 0]] + [[3 * math.cos(a), 3 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)])
    q = (xy[:, None, :] + ring[None]).reshape(-1, 2)
    loc, ir, _ = body.ray.intersects_location(np.c_[q, np.full(len(q), -80.)], np.tile([0, 0, 1.], (len(q), 1)), multiple_hits=False)
    z = np.full(len(q), np.inf); z[ir] = loc[:, 2]
    return z.reshape(len(xy), -1).min(1)
leads = []; lead_paths = []
for s_ in (-1, 1):
    a = np.r_[MU, Z_BOOT_END - 0.5] + u * TAB_R + w * s_ * PITCH / 2
    end = np.array([mouth[0] + s_ * 1.2, mouth[1]])
    n = int(np.linalg.norm(end - a[:2]) / 0.5) + 2
    xy = np.linspace(a[:2], end, n)
    zf = rear_floor(xy); zf[~np.isfinite(zf)] = a[2] + 2.5
    zp = np.minimum(zf - 2.5, a[2])                                      # hang 2.5 mm behind the lowest nearby surface, never above the Faston
    zp = np.convolve(np.r_[[zp[0]] * 4, zp, [zp[-1]] * 4], np.ones(9) / 9, mode="valid")
    zp = np.minimum(zp, zf - 1.5)
    path = [a] + [np.r_[p_, z_] for p_, z_ in zip(xy[1:], zp[1:])] + [np.r_[end, z_mouth - 1.0]]
    lead_paths.append(path)
    pts = np.array(path)
    for p0, p1 in zip(pts[:-1:2], pts[2::2]): leads.append(trimesh.creation.cylinder(radius=1.1, segment=[p0, p1], sections=16))
    for p_ in pts[::2]: leads.append(trimesh.creation.icosphere(subdivisions=1, radius=1.1).apply_translation(p_))
free_len = [sum(np.linalg.norm(np.subtract(q, p)) for p, q in zip(pp[:-1], pp[1:])) for pp in lead_paths]
print(f"MU10 lead mouth at ({mouth[0]:.1f}, {mouth[1]:.1f}), rear surface z {z_mouth:.1f}; free lead per terminal {free_len[0]:.0f} / {free_len[1]:.0f} mm")
terminals = trimesh.util.concatenate([board] + tabs); fastons = trimesh.util.concatenate(rcpt); wires = trimesh.util.concatenate(leads)

# Interference / clearance report
pq = trimesh.proximity.ProximityQuery(body)
def inside_frac(m, step=5):
    pts = m.vertices[::step]
    ins = np.concatenate([body.contains(pts[i:i + 400]) for i in range(0, len(pts), 400)])
    return ins.mean(), ins.sum(), len(ins)
print("placement:")
print(f"  MU10 at ({MU[0]:.2f}, {MU[1]:.2f}), seat z {SEAT}; body z {mu10.bounds[0,2]:.1f}..{mu10.bounds[1,2]:.1f}")
print(f"  front tweeter T at ({C['T'][0]:.2f}, {C['T'][1]:.2f}), z {tw_front.bounds[0,2]:.1f}..{tw_front.bounds[1,2]:.1f}")
print(f"  rear tweeter t at ({C['t'][0]:.2f}, {C['t'][1]:.2f}), z {tw_rear.bounds[0,2]:.1f}..{tw_rear.bounds[1,2]:.1f}")
print("interference with the printed top (sampled vertices inside the body):")
for nm, m in (("MU10", mu10), ("front tweeter T", tw_front), ("rear tweeter t", tw_rear), ("MU10 terminal board+tabs", terminals), ("2.8 Faston receptacles", fastons), ("leads to the MU10 lead mouth", wires)):
    f, n, tot = inside_frac(m, 1 if len(m.vertices) < 5000 else 5); print(f"  {nm:26s} {n:6d} / {tot:6d}  ({f*100:.2f} %)")
for nm, m in (("terminal board+tabs", terminals), ("Faston receptacles", fastons)):
    pts = trimesh.sample.sample_surface_even(m, 4000)[0]; sd = pq.signed_distance(pts)
    print(f"  {nm}: closest approach to the body {(-sd).min():.2f} mm" if sd.max() <= 0 else f"  {nm}: PENETRATES the body by up to {sd.max():.2f} mm")

# Exports: colored GLB (body translucent), multi-object 3MF, combined STL
COL = {"top": [205, 170, 120, 110], "mu10": [60, 60, 66, 255], "tweeter_front_T": [120, 120, 128, 255], "tweeter_rear_t": [150, 150, 158, 255],
       "mu10_terminals": [200, 120, 50, 255], "mu10_fastons_2p8": [40, 90, 200, 255], "mu10_leads": [200, 30, 30, 255]}
parts = {"top": body, "mu10": mu10, "tweeter_front_T": tw_front, "tweeter_rear_t": tw_rear, "mu10_terminals": terminals, "mu10_fastons_2p8": fastons, "mu10_leads": wires}
scene = trimesh.Scene()
for nm, m in parts.items():
    m = m.copy(); m.visual = trimesh.visual.ColorVisuals(m, face_colors=COL[nm]); scene.add_geometry(m, node_name=nm, geom_name=nm)
scene.export(f"{d}/obiwan_v11_top_with_drivers.glb")
scene.export(f"{d}/obiwan_v11_top_with_drivers.3mf")
trimesh.util.concatenate([m for nm, m in parts.items() if nm != "top"]).export(f"{d}/obiwan_v11_drivers_placed.stl")
json.dump({"T": tw_front.metadata["T4"], "t": tw_rear.metadata["T4"], "XF": XF, "TERM_AZ": TERM_AZ}, open(f"{d}/tweeter_placement.json", "w"))
print("wrote obiwan_v11_top_with_drivers.glb/.3mf and obiwan_v11_drivers_placed.stl")
