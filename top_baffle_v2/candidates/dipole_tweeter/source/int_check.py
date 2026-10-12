"""Strict check of a built integrated LM + T + UM part (lm_um_t)."""
import os, sys, math
os.environ['LX_DIPOLE_T_INTEGRATED'] = '1'; os.environ.setdefault('LX_ROUTING_PROFILE', 'obiwan')
sys.path[:0] = ['scripts', 'src', 'tools', 'vendor/GRS/PT2522-4']
import numpy as np, trimesh, manifold3d as m3
from shapely import contains_xy, affinity
from shapely.geometry import MultiPoint
import lx521_baffle.obiwan.lm_um_t as L
d = L.d
sfx = '_landscape' if d.LANDSCAPE else ''
stl = os.environ.get('STL', f'build/{d.DRIVER.name}/obiwan_{d.DRIVER.name}_lm_um_t{sfx}.stl')
m = trimesh.load(stl)
tag = f'{d.DRIVER_KEY} {d.ORIENTATION} lm_um_t'
MM = lambda t: m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))
mc = np.array(d.MU10_CENTER)
OPEN = affinity.translate(d._O, 0, d.PLATE_CENTER_Y)
if d._CH is not None: OPEN = OPEN.union(affinity.translate(d._CH, 0, d.PLATE_CENTER_Y))


def section_samples(P, w, h, off):
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    sd = np.cross(T, [0, 0, 1.0]); sd /= np.linalg.norm(sd, axis=1, keepdims=True); up = np.cross(sd, T)
    hh = (h / 2 + off)[:, None]; o = ((w - h) / 2)[:, None]; pts = []
    for k in np.linspace(-1, 1, 7): pts += [P + k * o * sd + hh * up, P + k * o * sd - hh * up]
    for sg in (-1, 1):
        for t in np.linspace(-np.pi / 2, np.pi / 2, 9): pts.append(P + sg * (o + hh * np.cos(t)) * sd + hh * np.sin(t) * up)
    return np.vstack(pts), np.tile(np.arange(len(P)), len(pts))


fail = []
for which in ('um', 't'):
    P, cover, w, h = L.duct(which)
    S, own = section_samples(P, w, h, 0.3)
    Lc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    outside = ~m.contains(S)
    if which == 'um': allowed = np.hypot(S[:, 0] - mc[0], S[:, 1] - mc[1]) < d.RING_CUTOUT_R
    else: allowed = contains_xy(OPEN.buffer(0.3), S[:, 0], S[:, 1]) & (Lc[own] > Lc[-1] - 5.0)
    out = outside & ~allowed & (Lc[own] > 3.0)
    rel = L.released_route(which); j = L.branch_index(which)
    r_join = d.min_bend_radius(L.resample(np.vstack([rel[j - 8:j], P])))
    print(f'{tag} {which} duct {w[0]:.1f}..{w[-1]:.1f} x {h[0]:.1f}..{h[-1]:.1f}: branch y {P[0,1]:.1f}, min bend R{r_join:.1f}, length {Lc[-1]:.0f} mm, '
          f'cover {cover:.0f} mm, exposed {int(out.sum())} of {len(S)}', np.round(S[out][:3], 1).tolist())
    if out.any(): fail.append(f'{which} exposed')
    # upstream: the LM's own duct from the keyed seam to the branch stays enclosed
    up_ = rel[(rel[:, 1] >= L.SEAM_Y + 2.0)][: max(j - int(np.argmax(rel[:, 1] >= L.SEAM_Y + 2.0)), 0)]
    if len(up_) > 2:
        r = (L.UM_DUCT_R if which == 'um' else L.TS_DUCT_R)
        Su, _ = section_samples(up_, np.full(len(up_), 2 * r), np.full(len(up_), 2 * r), 0.3)
        nu = int((~m.contains(Su)).sum()); print(f'{tag} {which} LM duct seam->branch: exposed {nu} of {len(Su)}')
        if nu: fail.append(f'{which} LM duct exposed')
    # the old tail past the branch: its old void is gone (material, or open air behind the rebuilt rear)
    tail = rel[j + 6:]; tail = tail[tail[:, 1] <= L.OLD_TAIL_END_Y]
    print(f'{tag} {which} old tail: {len(tail)} points, {int((~m.contains(tail)).sum())} in open air or new ducts')
# insert bores
for q in d._PIL:
    px, py = q.x, q.y + d.PLATE_CENTER_Y; floor = d.THICKNESS_MM - d.PLATE_T
    depth = min(d.FRAME_PILOT_DEPTH_MM, floor - d._pilot_rear_z(px, py) - d.FRAME_SKIN_MM)
    bot = np.array([[px, py, floor - depth - 0.05 - k * 0.05] for k in range(40)]); flo = int(m.contains(bot).sum()) * 0.05
    best = 99
    for z in np.linspace(floor - depth + 0.3, floor - 0.5, 6):
        for ang in np.linspace(0, 2 * np.pi, 72, endpoint=False):
            rr = np.arange(d.PILOT_D_MM / 2 + 0.05, 7, 0.05); Q = np.c_[px + rr * np.cos(ang), py + rr * np.sin(ang), np.full(len(rr), z)]
            o = ~m.contains(Q)
            if o.any(): best = min(best, rr[o.argmax()] - d.PILOT_D_MM / 2)
    print(f'{tag} M3 insert bore D{d.PILOT_D_MM} at ({px:.1f},{py:.1f}) depth {depth:.2f} wall {best:.2f} floor {flo:.2f}')
    if best < 1.15 or depth < 3.95: fail.append('insert bore')
# front, bodies, seam, bed
proud = int((m.vertices[:, 2] > d.THICKNESS_MM + 0.01).sum())
bodies = m.split(only_watertight=False)
cav = [b for b in bodies if b.volume < -0.5]
print(f'{tag} front: max z {m.bounds[1][2]:.2f}, vertices proud {proud}; shells {len(bodies)} (sealed cavities {len(cav)}: '
      f'{[np.round(b.bounds.mean(0), 1).tolist() for b in cav]}), watertight {m.is_watertight}, vol {m.volume / 1e3:.1f} cm3')
if proud: fail.append('proud of the front')
if cav: fail.append(f'{len(cav)} sealed cavities')
if not m.is_watertight:
    fail.append('not watertight')
    bf = trimesh.repair.broken_faces(m)
    c = m.triangles_center[bf] if len(bf) else np.zeros((0, 3))
    print(f'{tag} open mesh: {len(bf)} broken faces', np.round(c[:6], 1).tolist(), 'bbox', np.round([c.min(0), c.max(0)], 1).tolist() if len(c) else None)
top = trimesh.load(os.environ.get('LM_TOP', 'dtw/lm_keyed_top_installed.stl')) if os.path.exists(os.environ.get('LM_TOP', 'dtw/lm_keyed_top_installed.stl')) else None
if top is not None:
    bx = m3.Manifold.cube([300, 255 - 170, 40]).translate([-150, 170, -20])
    A, B = MM(m) ^ bx, MM(top) ^ bx
    print(f'{tag} keyed seam zone (y<255) vs released keyed top: extra {(A - B).volume():.2f} mm3, missing {(B - A).volume():.2f} mm3')
hull = MultiPoint(m.vertices[:, :2]).convex_hull; best = None
for a in np.arange(0, 180, 0.5):
    g = affinity.rotate(hull, a, origin='centroid'); x0, y0, x1, y1 = g.bounds; wd, ht = x1 - x0, y1 - y0
    sl = min(330 - max(wd, ht), 320 - min(wd, ht)); best = max(best or (sl, a, wd, ht), (sl, a, wd, ht))
print(f'{tag} H2C bed 330x320: best {best[2]:.1f} x {best[3]:.1f} at {best[1]:.1f} deg, slack {best[0]:.1f} mm')
if best[0] < 0: fail.append('bed')
print(f'{tag} RESULT', 'PASS' if not fail else 'FAIL ' + ', '.join(fail))
