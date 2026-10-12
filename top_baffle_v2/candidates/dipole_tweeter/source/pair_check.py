"""Strict check of the built pair (lm_um_t pair): keyed LM top variant + T + UM piece on the released joint."""
import os, sys
os.environ['LX_DIPOLE_T_INTEGRATED'] = '1'; os.environ['LX_DIPOLE_T_PAIR'] = '1'; os.environ.setdefault('LX_ROUTING_PROFILE', 'obiwan')
sys.path[:0] = ['scripts', 'src', 'tools', 'vendor/GRS/PT2522-4']
import numpy as np, trimesh, manifold3d as m3
from shapely import contains_xy, affinity
from shapely.geometry import MultiPoint
import lx521_baffle.obiwan.lm_um_t as L
d = L.d
B = f'build/{d.DRIVER.name}/obiwan_{d.DRIVER.name}_pair_'
lm, pc = trimesh.load(B + 'lm_top.stl'), trimesh.load(B + 'um_t.stl')
tag = f'{d.DRIVER_KEY} {d.ORIENTATION} pair'
def MM(t):
    m = m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))
    if m.status() != m3.Error.NoError:      # float32 rounding can pinch a vertex pair: merge and retry
        mesh = m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)); mesh.merge()
        m = m3.Manifold(mesh)
    if m.status() != m3.Error.NoError or m.is_empty(): raise SystemExit(f'not a valid solid for booleans: {m.status()}')
    return m
mc = np.array(d.MU10_CENTER)
OPEN = affinity.translate(d._O, 0, d.PLATE_CENTER_Y)
from trimesh.proximity import signed_distance


def enclosed(S):
    """inside either print, or in the fit gap between them (within 0.3 of both)"""
    a, b = lm.contains(S), pc.contains(S)
    out = ~(a | b)
    if out.any():
        i = np.flatnonzero(out)
        da = np.abs(signed_distance(lm, S[i])); db = np.abs(signed_distance(pc, S[i]))
        out[i] = ~((da < 0.3) & (db < 0.3))
    return ~out


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
    if which == 'um': allowed = np.hypot(S[:, 0] - mc[0], S[:, 1] - mc[1]) < d.RING_CUTOUT_R
    else: allowed = contains_xy(OPEN.buffer(0.3), S[:, 0], S[:, 1]) & (Lc[own] > Lc[-1] - 5.0)
    out = ~enclosed(S) & ~allowed & (Lc[own] > 3.0)
    rel = L.released_route(which); j = L.branch_index(which)
    r_join = d.min_bend_radius(L.resample(np.vstack([rel[j - 8:j], P])))
    cross = P[np.argmin(np.abs(P[:, 1] - 315.5))]
    print(f'{tag} {which} duct {w[0]:.1f}..{w[-1]:.1f} x {h[0]:.1f}..{h[-1]:.1f}: branch ({P[0,0]:.1f},{P[0,1]:.1f}), crosses the joint near '
          f'({cross[0]:.1f},{cross[2]:.1f}) [x,z], min bend R{r_join:.1f}, length {Lc[-1]:.0f} mm, cover {cover:.0f} mm, exposed {int(out.sum())} of {len(S)}',
          np.round(S[out][:3], 1).tolist())
    if out.any(): fail.append(f'{which} exposed')
    # the M2 LM--UM tie screw (axis along y at TIE x, z; pilot in the LM top, counterbore in the piece) stays clear
    tie = np.c_[np.full(60, d.TIE['x']), np.linspace(304.0, d.tie_window_y(), 60), np.full(60, d.TIE['z'])]
    from scipy.spatial import cKDTree
    dt = cKDTree(P).query(tie)[0].min(); need = max(w.max(), h.max()) / 2 * 0 + min(w.min(), h.min()) / 2 + 1.6 + 0.8
    print(f'{tag} {which} duct centre to the M2 tie screw path: {dt:.2f} mm (need >= {need:.2f})')
    if dt < need: fail.append(f'{which} hits the M2 tie')
    up_ = rel[(rel[:, 1] >= L.SEAM_Y + 2.0)][: max(j - int(np.argmax(rel[:, 1] >= L.SEAM_Y + 2.0)), 0)]
    if len(up_) > 2:
        r = (L.UM_DUCT_R if which == 'um' else L.TS_DUCT_R)
        Su, _ = section_samples(up_, np.full(len(up_), 2 * r), np.full(len(up_), 2 * r), 0.3)
        nu = int((~lm.contains(Su)).sum()); print(f'{tag} {which} LM duct seam->branch: exposed {nu} of {len(Su)}')
        if nu: fail.append(f'{which} LM duct exposed')
for q in d._PIL:
    m = pc
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
for nm, m in (('lm_top', lm), ('um_t', pc)):
    proud = int((m.vertices[:, 2] > d.THICKNESS_MM + 0.01).sum())
    bodies = m.split(only_watertight=False)
    cav = [b for b in bodies if b.volume < -0.5]
    print(f'{tag} {nm}: bounds {np.round(m.bounds, 1).tolist()}, max z {m.bounds[1][2]:.2f}, proud {proud}; shells {len(bodies)} '
          f'(cavities {len(cav)}), watertight {m.is_watertight}, vol {m.volume / 1e3:.1f} cm3')
    if proud: fail.append(f'{nm} proud')
    if cav: fail.append(f'{nm} cavities')
    if not m.is_watertight: fail.append(f'{nm} not watertight')
gap = MM(lm) ^ MM(pc)
print(f'{tag} overlap between the prints {gap.volume():.4f} mm3')
if gap.volume() > 0.01: fail.append('prints overlap')
band = pc.vertices[(pc.vertices[:, 1] < 340)]
gd = np.abs(signed_distance(lm, band))
um_rel = trimesh.load('dtw/um_carrier_installed.stl') if os.path.exists('dtw/um_carrier_installed.stl') else None
new = np.ones(len(band), bool)
if um_rel is not None:    # faces inherited unchanged from the released UM keep the released joint's own clearances
    new = np.abs(signed_distance(um_rel, band)) > 1e-3
i = np.argmin(np.where(new, gd, 9e9)); j = np.argmin(gd)
print(f'{tag} least gap piece->LM top: {gd[j]:.3f} mm at {np.round(band[j], 2).tolist()} (inherited {not new[j]}); '
      f'new surfaces {gd[i]:.3f} mm at {np.round(band[i], 2).tolist()}')
rel_min = gd[~new].min() if (~new).any() else 0.15   # the released joint's own least clearance
print(f'{tag} released joint least clearance {rel_min:.3f} mm; new surfaces must keep >= min(0.15, that)')
if gd[i] < min(0.15, rel_min) - 0.005: fail.append('fit gap')
top = trimesh.load('dtw/lm_keyed_top_installed.stl') if os.path.exists('dtw/lm_keyed_top_installed.stl') else None
if top is not None:
    bx = m3.Manifold.cube([300, 255 - 170, 40]).translate([-150, 170, -20])
    A, Bm = MM(lm) ^ bx, MM(top) ^ bx
    print(f'{tag} keyed seam zone (y<255) vs released keyed top: extra {(A - Bm).volume():.2f} mm3, missing {(Bm - A).volume():.2f} mm3')
    jb = m3.Manifold.cube([300, 340 - 305, 40]).translate([-150, 305, -20])
    print(f'{tag} joint zone (y 305..340) LM top variant vs released: extra {((MM(lm) ^ jb) - (MM(top) ^ jb)).volume():.1f} mm3, '
          f'missing {((MM(top) ^ jb) - (MM(lm) ^ jb)).volume():.1f} mm3')
print(f'{tag} RESULT', 'PASS' if not fail else 'FAIL ' + ', '.join(fail))
