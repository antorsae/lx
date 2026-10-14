"""Duct search for the integrated LM + T + UM part (lm_um_t). DUCT=um | t, against SC/isolid_<key>_<DUCT><TAG>.stl
(the part with the old ducts filled from the first branch candidate; for t also with the UM duct cut).

The duct starts ON the released cable inside the LM's own duct (same point and tangent) between BRANCH_Y_RANGE, may
run behind the part's rear only early on (it gets a cover continuing the LM's duct cover), is otherwise walled 0.8 mm
in material (pilots keep 1.2 mm), and ends: UM radially in the raised MU10's open back; T at the driver's terminals /
lead channel, where it may open into the window / channel over its last 5 mm. One clamped cubic B-spline; maximises
the smallest bend radius (the released lead-in included); keeps the rear-cover stretch short."""
import sys, os, json, time
sys.path[:0] = ['scripts', 'src', 'tools', 'vendor/GRS/PT2522-4']
os.environ['LX_DIPOLE_T_INTEGRATED'] = '1'
import numpy as np, trimesh, manifold3d as m3
from scipy.interpolate import BSpline
from scipy.optimize import differential_evolution
from shapely import contains_xy, affinity
from shapely.geometry import Polygon
import lx521_baffle.obiwan.lm_um_t as L
d = L.d

SC = os.environ['SC']; SEED = int(os.environ.get('SEED', 1)); DUCT = os.environ.get('DUCT', 'um'); TAG = os.environ.get('RAISE_TAG', '')
key = f'{d.DRIVER_KEY}_{d.ORIENTATION}'
sol = trimesh.load(SC + f'isolid_{key}_{DUCT}{TAG}.stl')
man = m3.Manifold(m3.Mesh(vert_properties=np.asarray(sol.vertices, np.float32), tri_verts=np.asarray(sol.faces, np.uint32)))
mc = np.array(d.MU10_CENTER)
PITCH = 0.3
lo = np.array([-120, 245, -7.0]); hi = np.array([120, mc[1] + 5, 18.9])
xs = np.arange(lo[0], hi[0], PITCH); ys = np.arange(lo[1], hi[1], PITCH); zs = np.arange(lo[2], hi[2], PITCH)
X, Y = np.meshgrid(xs, ys, indexing='ij')
occ = np.zeros((len(xs), len(ys), len(zs)), bool)
for k, z in enumerate(zs):
    acc = np.zeros(X.shape, bool)
    for loop in man.slice(float(z)).to_polygons():
        pg = Polygon(np.asarray(loop))
        if pg.area > 1e-6: acc ^= contains_xy(pg, X, Y)
    occ[:, :, k] = acc
below = np.logical_or.accumulate(occ, axis=2)
above = np.logical_or.accumulate(occ[:, :, ::-1], axis=2)[:, :, ::-1]
fl = d.THICKNESS_MM - d.PLATE_T
pad = float(os.environ.get('PILOT_PAD', 0.6))
for q in d._PIL:                       # insert bores keep a 1.2 mm wall (0.8 skin + pad - voxel rounding)
    m_ = np.hypot(X - q.x, Y - (q.y + d.PLATE_CENTER_Y)) < d.PILOT_D_MM / 2 + pad
    occ[:, :, (zs >= fl - d.FRAME_PILOT_DEPTH_MM - pad) & (zs <= fl + 0.5)] &= ~m_[:, :, None]
print('occupancy', key, DUCT, int(occ.sum()), flush=True)


def look(P, grid):
    idx = np.floor((P - lo) / PITCH + 0.5).astype(int)
    ok = (idx >= 0).all(1) & (idx[:, 0] < len(xs)) & (idx[:, 1] < len(ys)) & (idx[:, 2] < len(zs))
    out = np.zeros(len(P), bool); i = idx[ok]; out[ok] = grid[i[:, 0], i[:, 1], i[:, 2]]; return out


def bspline(ctrl, n=360):
    ctrl = np.asarray(ctrl, float); k = 3; m = len(ctrl)
    t = np.r_[[0] * k, np.linspace(0, 1, m - k + 1), [1] * k]
    return BSpline(t, ctrl, k)(np.linspace(0, 1, n))


def minR(P):
    a, b, c = P[:-2], P[1:-1], P[2:]
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    R = np.linalg.norm(b - a, axis=1) * np.linalg.norm(c - b, axis=1) * np.linalg.norm(a - c, axis=1) / np.maximum(2 * area2, 1e-12)
    return float(R.min())


def skin(P, s, Wp, Hp):
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    sd = np.cross(T, [0, 0, 1.0]); n = np.linalg.norm(sd, axis=1, keepdims=True); sd = np.where(n > 1e-6, sd / np.maximum(n, 1e-12), [1.0, 0, 0])
    up = np.cross(sd, T); h = (Hp / 2 + s)[:, None]; off = ((Wp - Hp) / 2)[:, None]; pts = []
    for k in np.linspace(-1, 1, 5):
        pts += [P + k * off * sd + h * up, P + k * off * sd - h * up]
    for sg in (-1, 1):
        for t in np.linspace(-np.pi / 2, np.pi / 2, 9):
            pts.append(P + sg * (off + h * np.cos(t)) * sd + h * np.sin(t) * up)
    return np.vstack(pts), np.tile(np.arange(len(P)), len(pts))


REL = L.released_route(DUCT)
side = 1.0 if DUCT == 'um' else -1.0
cand = np.flatnonzero((REL[:, 1] >= L.BRANCH_Y_RANGE[0]) & (REL[:, 1] <= L.BRANCH_Y_RANGE[1]) & (np.sign(REL[:, 0]) == side))
cand = cand[(cand >= 8) & (cand < len(REL) - 4)]
print('start candidates y', REL[cand[0], 1].round(1), '..', REL[cand[-1], 1].round(1), flush=True)
MOUTH_ARC = float(os.environ.get('MOUTH_ARC', 50.0))
TURN_MAX = float(os.environ.get('TURN_MAX', 240.0))
D_UM = d.UM_GUIDE_D; WF, HF = d.UM_GUIDE_W, d.UM_GUIDE_H
D_T = d.T_BRANCH_D
# UM end: radially into the MU10 opening, centred in the ring band under the seat
R_IN = d.RING_CUTOUT_R - 1.0; Z_END = d.UM_GUIDE_RING_Z
# T end: at the driver's terminals / lead channel (installed), front-ish depth inside the window / channel
LEAD = np.array(d._rot(*d.DRIVER.lead_point())) + [0.0, d.PLATE_CENTER_Y]
ZT = d.THICKNESS_MM - d.PLATE_T - d.FRAME_SKIN_MM - D_T / 2
OPEN = affinity.translate(d._O, 0, d.PLATE_CENTER_Y)
if d._CH is not None: OPEN = OPEN.union(affinity.translate(d._CH, 0, d.PLATE_CENTER_Y))
ROUND_LANE = d._round_lane
# PT2522 (solder lugs inside the window, no lead channel): like the released T-below routes, the duct ends anywhere in
# the window's bottom 20 mm (the cable runs on inside the window to the lugs)
WIN_END = DUCT == 't' and d._CH is None
WB = OPEN.bounds


def ctrl_of(v):
    b = int(cand[int(np.clip(round(v[0]), 0, len(cand) - 1))])
    t0 = REL[b + 2] - REL[b - 2]; t0 /= np.linalg.norm(t0)
    if DUCT == 'um':
        ang = np.radians(v[11]); E = np.array([mc[0] + R_IN * np.cos(ang), mc[1] + R_IN * np.sin(ang), Z_END])
        te = -np.array([np.cos(ang), np.sin(ang), 0.0])
    else:
        E = (np.array([v[13], v[14], ZT]) if WIN_END else np.array([LEAD[0] + v[13], LEAD[1] + v[14], ZT])); ang = np.radians(v[11]); te = np.array([np.cos(ang), np.sin(ang), 0.0])
    return b, [REL[b], REL[b] + v[1] * t0, v[2:5], v[5:8], v[8:11], E - v[12] * te, E]


def curve(v):
    b, c = ctrl_of(v)
    return b, bspline(c)


def knots(v, P):
    L_ = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    if DUCT == 't': return [[0.0, D_T, D_T], [L_, D_T, D_T]]
    if ROUND_LANE:      # round beside the window; flat only into the MU10's seat band at the end
        a, t = v[13], v[14]; a1 = max(L_ - a, 1.0); a0 = max(a1 - t, 0.5)
        return [[0.0, D_UM, D_UM], [a0, D_UM, D_UM], [a1, WF, HF], [L_, WF, HF]]
    r, t = v[13], v[14]   # round inside its rear cover; flat once in the 7 mm frame
    return [[0.0, D_UM, D_UM], [r, D_UM, D_UM], [r + t, WF, HF], [L_, WF, HF]]


def score(v, ret=False):
    b, P = curve(v)
    Wp, Hp = d.section_from_knots(P, knots(v, P))
    S, own = skin(P, 0.8, Wp, Hp)
    Lc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    solid = look(S, occ)
    rear_open = ~look(S, below) & look(S, above) & (Lc[own] < MOUTH_ARC)
    if DUCT == 'um':
        end_ok = np.hypot(S[:, 0] - mc[0], S[:, 1] - mc[1]) < d.RING_CUTOUT_R
    else:
        end_ok = contains_xy(OPEN.buffer(0.3), S[:, 0], S[:, 1]) & (Lc[own] > Lc[-1] - 5.0)
        if not contains_xy(OPEN.buffer(-D_T / 2 - 0.3), P[-1, 0], P[-1, 1]): return 9000 if not ret else dict(bad=9999, R=0)
    bad = int((~(solid | end_ok | rear_open)).sum())
    R = minR(np.vstack([REL[b - 8:b], P]))
    Tg = np.diff(P, axis=0); Tg /= np.linalg.norm(Tg, axis=1, keepdims=True)
    turn = float(np.degrees(np.arccos(np.clip(np.sum(Tg[1:] * Tg[:-1], axis=1), -1, 1)).sum()))
    if turn > TURN_MAX: bad += int(turn - TURN_MAX) + 1
    cov = rear_open & ~solid
    cover_arc = float(Lc[own[cov]].max()) if cov.any() else 0.0
    if ret: return dict(bad=bad, R=R, len=float(Lc[-1]), turn=turn, cover_arc=cover_arc, start_y=float(REL[b, 1]))
    return (1000 + bad) if bad else -min(R, 40) + 0.02 * Lc[-1] + 0.08 * cover_arc


mc_low = mc[1] - d.RING_OUTER_R
lim = [(0, len(cand) - 1), (3, 30),
       (-115, 115), (L.BRANCH_Y_RANGE[0], mc_low + 10), (-6, 18.5),
       (-115, 115), (L.BRANCH_Y_RANGE[0] + 10, mc_low + 30), (-6, 18.5),
       (-115, 115), (d.SPLIT_Y, mc[1] - 10), (-6, 18.5)]
if DUCT == 'um': lim += [(-115, -25), (3, 35), (0, 35), (10, 25)]
elif WIN_END: lim += [(0, 360), (3, 30), (WB[0], WB[2]), (WB[1], WB[1] + 20.0)]
else: lim += [(0, 360), (3, 30), (-12, 12), (-12, 12)]
if os.environ.get('EVAL'):
    v = np.array(json.load(open(os.environ['EVAL']))['x']); b_, P = curve(v); print(score(v, True))
    Wp, Hp = d.section_from_knots(P, knots(v, P)); S_, own = skin(P, 0.8, Wp, Hp)
    Lc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    sol_ = look(S_, occ); ro = ~look(S_, below) & look(S_, above) & (Lc[own] < MOUTH_ARC)
    eo = np.hypot(S_[:, 0] - mc[0], S_[:, 1] - mc[1]) < d.RING_CUTOUT_R
    bad_ = ~(sol_ | eo | ro)
    print('body half width', d.BODY_HALF_W, 'flat to', d.BODY_HALF_W - d.RAZOR_FLARE_MM)
    for i in np.unique(own[bad_]): print('  arc %.1f centre %s bad %s' % (Lc[i], np.round(P[i], 1), np.round(S_[bad_ & (own == i)][:2], 1).tolist()))
    sys.exit()
t0 = time.time()
x0 = np.clip(np.array(json.load(open(os.environ['X0_FROM']))['x']), [l for l, _ in lim], [h for _, h in lim]) if os.environ.get('X0_FROM') else None
res = differential_evolution(score, lim, popsize=int(os.environ.get('POP', 60)), maxiter=int(os.environ.get('MAXITER', 900)),
                             tol=1e-8, seed=SEED, polish=False, updating='deferred', workers=1, x0=x0)
info = score(res.x, True)
print('DUCT', key, DUCT, SEED, info, round(time.time() - t0), 's', flush=True)
b, P = curve(res.x)
json.dump(dict(x=res.x.tolist(), points=np.round(P, 4).tolist(), section_knots=knots(res.x, P),
               d=(D_T if DUCT == 't' else D_UM), w=(D_T if DUCT == 't' else WF), h=(D_T if DUCT == 't' else HF), **info),
          open(SC + f'int_{key}_{DUCT}{TAG}_s{SEED}.json', 'w'), indent=1)
