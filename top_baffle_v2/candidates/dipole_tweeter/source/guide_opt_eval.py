"""Search the hidden UM cable guide of a UM + T below part against its solid with the T duct already cut (stage
'pre_guide', SC/gsolid_<key>.stl).

The guide starts ON the released UM cable just past the LM's UM duct exit (same point and tangent), may open only as
a mouth on the rear face early on (material in front, nothing behind), is otherwise fully walled (0.8 mm skin in
material, pilots keep their 1.2 mm walls), and ends radially in the raised MU10's open back near its terminals.
Maximises the smallest bend radius (>= R14 required) of one clamped cubic B-spline."""
import sys, os, json, time
sys.path[:0] = ['scripts', 'src', 'tools']
import numpy as np, trimesh, manifold3d as m3
from scipy.interpolate import BSpline
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import Polygon
import lx521_baffle.obiwan.dipole_tweeter as d
from lx521_baffle.obiwan.route import route_cable_points

SC = os.environ['SC']; SEED = int(os.environ.get('SEED', 1))
key = f'{d.DRIVER_KEY}_{d.ORIENTATION}' + os.environ.get('RAISE_TAG', '')
sol = trimesh.load(SC + os.environ.get('SOLID', 'gsolid') + f'_{key}.stl')    # 'solid' = no T duct yet (guide first)
man = m3.Manifold(m3.Mesh(vert_properties=np.asarray(sol.vertices, np.float32), tri_verts=np.asarray(sol.faces, np.uint32)))
mc = np.array(d.MU10_CENTER)
PITCH = 0.3
lo = np.array([-70, 300, -2.0]); hi = np.array([70, mc[1] + 5, 18.9])
xs = np.arange(lo[0], hi[0], PITCH); ys = np.arange(lo[1], hi[1], PITCH); zs = np.arange(lo[2], hi[2], PITCH)
X, Y = np.meshgrid(xs, ys, indexing='ij')
occ = np.zeros((len(xs), len(ys), len(zs)), bool)
for k, z in enumerate(zs):
    acc = np.zeros(X.shape, bool)
    for loop in man.slice(float(z)).to_polygons():
        pg = Polygon(np.asarray(loop))
        if pg.area > 1e-6: acc ^= contains_xy(pg, X, Y)
    occ[:, :, k] = acc
if os.environ.get('LUMEN_KEEP'):                                   # guide first: keep the released T lumen stretch the
    from lx521_baffle.obiwan.route import ts_cable_points             # T duct branches from free (lumen + its 0.8 wall)
    from scipy.spatial import cKDTree as _K
    _L = np.asarray(ts_cable_points(0.5)); _L = _L[(_L[:, 1] >= d.UM_T_FILL_FROM_Y - 1) & (_L[:, 1] <= d.SPLIT_Y + 3)]
    _ix = np.argwhere(occ); _pts = lo + _ix * PITCH
    _near = _K(_L).query(_pts, distance_upper_bound=4.0)[0] < 3.0
    occ[tuple(_ix[_near].T)] = False
below = np.logical_or.accumulate(occ, axis=2)                      # solid at or behind this depth in the column
above = np.logical_or.accumulate(occ[:, :, ::-1], axis=2)[:, :, ::-1]   # solid at or in front of it
fl = d.THICKNESS_MM - d.PLATE_T
for q in d._PIL:                                                   # pilots: 0.4 bigger, so 0.8 skin -> 1.2 wall
    pad = float(os.environ.get('PILOT_PAD', 0.6))                 # 0.4 + voxel rounding
    m_ = np.hypot(X - q.x, Y - (q.y + d.PLATE_CENTER_Y)) < d.PILOT_D_MM / 2 + pad
    occ[:, :, (zs >= fl - d.FRAME_PILOT_DEPTH_MM - pad) & (zs <= fl + 0.5)] &= ~m_[:, :, None]
# the LM (with its 0.2 mm fit gap) is a hard obstacle for the guide and the free cable: its half-lap tongues fill
# the space behind the front layer just above the split
import build_dipole_tweeter as bdt
lmm = bdt.released_installed('h2c_obiwan_core_lm_carrier_no_floor_stand')
lmM = m3.Manifold(m3.Mesh(vert_properties=np.asarray(lmm.vertices, np.float32), tri_verts=np.asarray(lmm.faces, np.uint32)))
lmM = (lmM ^ m3.Manifold.cube([300, 40, 80]).translate([-150, 290, -40])).minkowski_sum(m3.Manifold.cube([0.4] * 3, center=True))
lmocc = np.zeros_like(occ)
for k, z in enumerate(zs):
    acc = np.zeros(X.shape, bool)
    for loop in lmM.slice(float(z)).to_polygons():
        pg = Polygon(np.asarray(loop))
        if pg.area > 1e-6: acc ^= contains_xy(pg, X, Y)
    lmocc[:, :, k] = acc
print('occupancy', key, int(occ.sum()), 'LM', int(lmocc.sum()), flush=True)


def look(P, grid):
    idx = np.floor((P - lo) / PITCH + 0.5).astype(int)
    ok = (idx >= 0).all(1) & (idx[:, 0] < len(xs)) & (idx[:, 1] < len(ys)) & (idx[:, 2] < len(zs))
    out = np.zeros(len(P), bool); i = idx[ok]; out[ok] = grid[i[:, 0], i[:, 1], i[:, 2]]; return out


def bspline(ctrl, n=320):
    ctrl = np.asarray(ctrl, float); k = 3; m = len(ctrl)
    t = np.r_[[0] * k, np.linspace(0, 1, m - k + 1), [1] * k]
    return BSpline(t, ctrl, k)(np.linspace(0, 1, n))


def minR(P):
    a, b, c = P[:-2], P[1:-1], P[2:]
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    R = np.linalg.norm(b - a, axis=1) * np.linalg.norm(c - b, axis=1) * np.linalg.norm(a - c, axis=1) / np.maximum(2 * area2, 1e-12)
    return float(R.min())


W, H = d.UM_GUIDE_W, d.UM_GUIDE_H


def skin(P, s, Wp=None, Hp=None):
    """Wall samples of the stadium section (Wp wide in the part's plane, Hp deep, per point), offset s outward."""
    Wp = np.full(len(P), W) if Wp is None else Wp; Hp = np.full(len(P), H) if Hp is None else Hp
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    sd = np.cross(T, [0, 0, 1.0]); n = np.linalg.norm(sd, axis=1, keepdims=True); sd = np.where(n > 1e-6, sd / np.maximum(n, 1e-12), [1.0, 0, 0])
    up = np.cross(sd, T); h = (Hp / 2 + s)[:, None]; off = ((Wp - Hp) / 2)[:, None]; pts = []
    for k in np.linspace(-1, 1, 5):
        pts += [P + k * off * sd + h * up, P + k * off * sd - h * up]
    for sg in (-1, 1):
        for t in np.linspace(-np.pi / 2, np.pi / 2, 9):
            pts.append(P + sg * (off + h * np.cos(t)) * sd + h * np.sin(t) * up)
    return np.vstack(pts), np.tile(np.arange(len(P)), len(pts))


REL = np.asarray(route_cable_points(0.5))
i_exit = int(np.argmax(REL[:, 1] >= 313.0))                          # the cable is out of the LM's duct cover here
cand = np.flatnonzero((REL[:, 1] >= float(os.environ.get('CAND_Y0', 313.0))) & (REL[:, 1] <= float(os.environ.get('CAND_Y1', d.SPLIT_Y + 4.0))) & (REL[:, 0] > 0))
cand = cand[(cand >= 4) & (cand < len(REL) - 4)]
print('start candidates y', REL[cand[0], 1].round(1), '..', REL[cand[-1], 1].round(1), flush=True)
R_IN = d.RING_CUTOUT_R - 1.0
Z_END = d.UM_GUIDE_RING_Z
MOUTH_ARC = float(os.environ.get('MOUTH_ARC', 50.0))
TURN_MAX = float(os.environ.get('TURN_MAX', 220.0))
CABLE_R = 3.6


def ctrl_of(v):
    b = int(cand[int(np.clip(round(v[0]), 0, len(cand) - 1))])
    t0 = REL[b + 2] - REL[b - 2]; t0 /= np.linalg.norm(t0)
    ang = np.radians(v[11]); E = np.array([mc[0] + R_IN * np.cos(ang), mc[1] + R_IN * np.sin(ang), Z_END])
    te = -np.array([np.cos(ang), np.sin(ang), 0.0])
    return b, [REL[b], REL[b] + v[1] * t0, v[2:5], v[5:8], v[8:11], E - v[12] * te, E]


def curve(v):
    b, c = ctrl_of(v)
    return b, bspline(c)


def section(v, P):
    if len(v) < 15: return None, None
    return d.um_guide_section(P, v[13], v[13] + v[14])


def score(v, ret=False):
    b, P = curve(v)
    Wp, Hp = section(v, P)
    S, own = skin(P, 0.8, Wp, Hp)
    Lc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    solid = look(S, occ)
    in_mu = np.hypot(S[:, 0] - mc[0], S[:, 1] - mc[1]) < d.RING_CUTOUT_R
    rear_open = ~look(S, below) & (look(S, above) | (S[:, 1] < d.SPLIT_Y + 0.5)) & (Lc[own] < MOUTH_ARC)
    bad = int((~(solid | in_mu | rear_open)).sum()) + int(look(S, lmocc).sum())
    if os.environ.get('GUIDE_LANE'):                                # keep a corridor free for the T duct beside the window
        gx, gy0, gy1 = map(float, os.environ['GUIDE_LANE'].split(','))
        bad += int(((S[:, 0] < gx) & (S[:, 1] > gy0) & (S[:, 1] < gy1)).sum())
    # the released free cable from the LM exit up to the start stays clear of the part
    fc = REL[i_exit:b + 1]
    if len(fc) > 1:
        a = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        T = np.gradient(fc, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
        U = np.cross(T, [0, 0, 1.0]); U /= np.linalg.norm(U, axis=1, keepdims=True); V = np.cross(T, U)
        ring = (fc[:, None, :] + CABLE_R * (np.cos(a)[None, :, None] * U[:, None, :] + np.sin(a)[None, :, None] * V[:, None, :])).reshape(-1, 3)
        bad += int(look(ring, occ).sum()) + int(look(ring, lmocc).sum())
    R = minR(np.vstack([REL[b - 8:b], P]))
    Tg = np.diff(P, axis=0); Tg /= np.linalg.norm(Tg, axis=1, keepdims=True)
    turn = float(np.degrees(np.arccos(np.clip(np.sum(Tg[1:] * Tg[:-1], axis=1), -1, 1)).sum()))
    if turn > TURN_MAX: bad += int(turn - TURN_MAX) + 1
    nopen = int((rear_open & ~solid).sum())
    groove = int((rear_open & ~solid & look(S, above) & (S[:, 1] > d.SPLIT_Y + 0.5)).sum())   # open on the part's rear face
    # those rear samples get a cover continuing the LM's duct cover: its extent along the guide, and the free gap
    # between the LM's exit and the cover's start, both kept short
    cov = rear_open & ~solid
    cover_arc = float(Lc[own[cov]].max()) if cov.any() else 0.0
    gap = float(np.sum(np.linalg.norm(np.diff(REL[i_exit:b + 1], axis=0), axis=1))) if b > i_exit else 0.0
    if ret: return dict(bad=bad, R=R, len=float(Lc[-1]), turn=turn, cover_samples=nopen, cover_arc=cover_arc, gap=gap, start_y=float(REL[b, 1]))
    return (1000 + bad) if bad else -min(R, 40) + 0.02 * Lc[-1] + 0.08 * cover_arc + float(os.environ.get('GAP_W', 0.3)) * gap


ylo, yhi = d.SPLIT_Y - 2, mc[1] - d.RING_OUTER_R + 10
lim = [(0, len(cand) - 1), (float(os.environ.get('HANDLE0', 2)), 25),
       (0, 65), (d.SPLIT_Y - 6, yhi), (-1, 18.5),
       (0, 65), (d.SPLIT_Y, yhi + 20), (0, 18.5),
       (0, 65), (d.SPLIT_Y + 10, mc[1] - 20), (0, 18.5),
       (-115, -25), (3, 35), (0, 35), (10, 25)]
if os.environ.get('EVAL'):
    v = np.array(json.load(open(os.environ['EVAL']))['x']); b_, P = curve(v)
    S, own = skin(P, 0.8, *section(v, P)); Lc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    solid = look(S, occ); in_mu = np.hypot(S[:, 0] - mc[0], S[:, 1] - mc[1]) < d.RING_CUTOUT_R
    rear_open = ~look(S, below) & (look(S, above) | (S[:, 1] < d.SPLIT_Y + 0.5)) & (Lc[own] < MOUTH_ARC)
    badm = ~(solid | in_mu | rear_open)
    print(score(v, True))
    Q = np.vstack([REL[b_ - 8:b_], P]); A, B, C = Q[:-2], Q[1:-1], Q[2:]
    ar = np.linalg.norm(np.cross(B - A, C - A), axis=1)
    Rr = np.linalg.norm(B - A, axis=1) * np.linalg.norm(C - B, axis=1) * np.linalg.norm(A - C, axis=1) / np.maximum(2 * ar, 1e-12)
    for j in np.argsort(Rr)[:1]:
        k = j + 1 - 8
        print('  tightest R %.1f at %s (curve index %d of %d, arc %.1f)' % (Rr[j], np.round(Q[j + 1], 1), k, len(P), Lc[max(k, 0)]))
    for j in range(0, len(P), 20): print('    P', j, np.round(P[j], 1), 'R here %.1f' % Rr[min(j + 7, len(Rr) - 1)])
    lmh = look(S, lmocc); beh = look(S, below); fr = look(S, above)
    from shapely import affinity as _af
    win = _af.translate(d._O, 0, d.PLATE_CENTER_Y); inwin = contains_xy(win.buffer(0.8), S[:, 0], S[:, 1])
    cls = np.where(lmh, 'LM', np.where(inwin, 'window', np.where(beh & fr, 'internal-void/front', np.where(~fr & ~beh, 'outside-outline', np.where(beh & ~fr, 'front-breakout', 'late-rear')))))
    import collections
    for i in np.unique(own[badm | lmh]):
        q = badm | lmh
        sel = q & (own == i)
        print('  arc %.1f centre %s %s' % (Lc[i], np.round(P[i], 1), dict(collections.Counter(cls[sel]))))
    sys.exit()
    for i in np.unique(own[badm])[::3]:
        q = S[badm & (own == i)]
        print('  arc %.1f centre %s bad %d  e.g. %s' % (Lc[i], np.round(P[i], 1), len(q), np.round(q[0], 1)))
    sys.exit()
t0 = time.time()
lx, lz = d.UM_GUIDE_LANE_X, d.FRAME_REAR_Z + d.UM_GUIDE_WALL_MM + d.UM_GUIDE_H / 2 + d.UM_GUIDE_SLACK_MM / 2
x0 = np.array([len(cand) / 2, 10, 30.0, d.SPLIT_Y + 10, 2.5, min(lx, 64), d.SPLIT_Y + 30, lz,
               min(lx - 4, 64), yhi - 2, (lz + Z_END) / 2, -60, 15])
if os.environ.get('X0_FROM'): x0 = np.array(json.load(open(os.environ['X0_FROM']))['x'])
if len(x0) < len(lim): x0 = np.r_[x0, [15.0, 15.0][:len(lim) - len(x0)]]
x0 = np.clip(x0, [l for l, _ in lim], [h for _, h in lim])
print('x0', score(x0, True), flush=True)
res = differential_evolution(score, lim, popsize=int(os.environ.get('POP', 60)), maxiter=int(os.environ.get('MAXITER', 900)), tol=1e-8, seed=SEED,
                             polish=False, updating='deferred', workers=1, x0=x0)
info = score(res.x, True)
print('GUIDE', key, SEED, info, round(time.time() - t0), 's', flush=True)
b, P = curve(res.x)
json.dump(dict(x=res.x.tolist(), points=np.round(P, 4).tolist(), w=W, h=H, round_to=float(res.x[13]), flat_from=float(res.x[13] + res.x[14]), **info),
          open(SC + f'guide_{key}_s{SEED}.json', 'w'), indent=1)
