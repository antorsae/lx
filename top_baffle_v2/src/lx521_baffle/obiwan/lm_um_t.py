"""Candidate Obi-Wan part: the keyed LM top half, a dipole tweeter and the raised MU10 in ONE piece (T below the UM).

Replaces the separate "UM + T below" part for the drivers whose integrated piece fits the H2C bed (PT2522-4 rotated,
AMT17D2.1 upright and rotated). The released keyed LM top (h2c_obiwan_lm_keyed_top, mating the released keyed bottom
at the seam y=172.48) is fused with the dipole_tweeter T-below body: the LM's top bar and the UM's LM-joint piece
become solid (no half-laps, no M2 tie), so the cables never leave the part:

  * UM (MU10) cable: stays in the LM's own UM duct up the +x side of the woofer ring, branches off it near the top of
    the ring and climbs -- inside a continuation of the LM's own rear duct cover until the frame buries it -- into a
    lane beside the window, ending radially in the raised MU10's open back.
  * T cable: likewise on the -x side, from the LM's T duct to the driver's terminals / lead channel (rotated drivers
    turn the other way so their terminals / leads face -x; the AMT upright's lead exit is reached under the window).

Past each branch the LM's old duct is filled and its rear cover removed. Each side carries one duct, so the frame is
only as wide as one lane. Driver fixing: the project's standard M3 heat-set inserts (dipole_tweeter, INTEGRATED).

Routes: dipole_tweeter_lm_routes.json, keys DRIVER/ORIENTATION with 'um' and 't' (searched centrelines, section knots
and rear-cover lengths). Mesh pipeline (manifold3d); `tessellate(shape) -> trimesh` turns the build123d cutters."""
import os
os.environ.setdefault('LX_DIPOLE_T_INTEGRATED', '1')
os.environ.setdefault('LX_ROUTING_PROFILE', 'obiwan')
import json
import math

from . import dipole_tweeter as d
from .route import DUCT_R as UM_DUCT_R, TS_DUCT_R, route_cable_points, ts_cable_points
from ..base import L22_CUTOUT, L22_PILOT_ANGLES_DEG, L22_PILOT_PCD_MM, THICKNESS_MM
from ..flush import LM_RECESS_R, LM_SEAT_Z

if not d.INTEGRATED:
    raise RuntimeError('lm_um_t needs LX_DIPOLE_T_INTEGRATED=1 before dipole_tweeter is first imported')

SUPPORTED = {('pt2522_4', 'landscape'), ('amt17d2_1', 'portrait'), ('amt17d2_1', 'landscape')}
# Pair variant (dipole_tweeter.PAIR): the PT2522 upright's one piece would exceed the H2C's 325 mm nozzle reach, so the
# same routing is printed as two parts meeting at the released LM--UM joint (see lm_um_t_pair_mesh).
PAIR_SUPPORTED = {('pt2522_4', 'portrait')}
if d.PAIR: SUPPORTED = PAIR_SUPPORTED
KEY = (d.DRIVER_KEY, d.ORIENTATION)
SEAM_Y = 172.481                       # lm_split.LM_SPLIT_SEAM_Y (the released keyed seam stays untouched)
JOINT_HALF_W = 36.9                    # the LM top bar / LM-joint piece's outer half-width at the split
JOINT_Y0 = 300.0                       # the joint zone filled solid: from here to just above the split ...
JOINT_Z0 = d.CORE_REAR_Z               # ... from the core rear plane to the front, outside the woofer's flange recess
BRANCH_Y_RANGE = (262.0, 306.0)        # where each duct may leave the LM's own duct (the top of the woofer ring)
OLD_TAIL_END_Y = d.SPLIT_Y + 14.0      # the old T lumen runs on into the former UM joint piece: filled up to here
TOP_ZONE_HALF_W = 125.0                # the top zone rebuilt above the branches (the old duct tails run inside it)
JOINT_FOOTPRINT_Z = 15.0               # the part's footprint in the joint zone, taken at this depth
LM_WEB_REAR_Z = LM_SEAT_Z - 0.85       # the LM's web behind the woofer flange (route.LM_SEAT_MEMBRANE_BOTTOM_Z, 11.45)
WOOFER_BOSS_R, WOOFER_BOSS_REAR_Z = 5.2, 3.85   # the woofer's M5 insert bosses (D6.4 bores): kept down to their own rear
WOOFER_CUTOUT_R = L22_CUTOUT[2] / 2.0
COVER_LEAD_IN_MM = 8.0                 # a new duct's cover starts this far back along the LM's own duct
UPSTREAM_KEEP_MM = 6.0                 # the rear rebuild stays this far (plan) off the live LM ducts upstream of a branch
ROUTES_FILE = d.UM_T_ROUTES_FILE


def routes():
    if KEY not in SUPPORTED:
        raise RuntimeError(f'integrated LM + T + UM is built for {sorted(SUPPORTED)}, not {KEY}')
    r = json.loads(ROUTES_FILE.read_text()) if ROUTES_FILE.exists() else {}
    return r.get(f'{d.DRIVER_KEY}/{d.ORIENTATION}', {})


def released_route(which):
    import numpy as np
    return np.asarray((route_cable_points if which == 'um' else ts_cable_points)(0.5))


def branch_index(which, y=None):
    """Index on the released route where the new duct leaves it: the stored route's start, or (search stages) the
    first candidate of BRANCH_Y_RANGE on that cable's side."""
    import numpy as np
    rel = released_route(which)
    if y is None:
        rt = routes().get(which)
        if rt is None: raise RuntimeError(f'no {which} route for {KEY} in {ROUTES_FILE.name}')
        j = int(np.argmin(np.linalg.norm(rel - np.asarray(rt['points'][0]), axis=1)))
        if np.linalg.norm(rel[j] - rt['points'][0]) > 0.01: raise RuntimeError(f'{which} route does not start on the released cable')
        return j
    side = 1.0 if which == 'um' else -1.0
    return int(np.flatnonzero((rel[:, 1] >= y) & (np.sign(rel[:, 0]) == side))[0])


def duct(which):
    """(points, cover_arc_mm, w, h) of a stored duct."""
    import numpy as np
    rt = routes()[which]
    pts = np.asarray(rt['points'], float)
    w, h = d.section_from_knots(pts, rt['section_knots'])
    r_min = d.min_bend_radius(resample(np.vstack([released_route(which)[branch_index(which) - 8:branch_index(which)], pts])))
    floor = {'um': d.UM_GUIDE_MIN_BEND_R, 't': T_MIN_BEND_R}[which]
    if r_min < min(rt['min_bend_r_mm'], floor) - 0.05:
        raise RuntimeError(f'{KEY} {which}: bends at R{r_min:.1f} (recorded R{rt["min_bend_r_mm"]}, rule R{floor})')
    return pts, float(rt['cover_arc_mm']), w, h


T_MIN_BEND_R = 14.0


def resample(P, step=0.5):
    """Polyline at even arc spacing: bend radii from three-point circles are only meaningful on evenly spaced points
    (a stored spline crowds its samples near the ends, where 0.1 um rounding alone reads as a tight bend)."""
    import numpy as np
    P = np.asarray(P, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.arange(0.0, s[-1], step)
    return np.c_[[np.interp(t, s, P[:, k]) for k in range(3)]].T


def _top_rear_cutter(y_um, y_t, upstream=()):
    """Everything behind the rebuilt rear of the top zone above each branch (+x above the UM branch, -x above the T
    branch, up to OLD_TAIL_END_Y): inside the woofer's flange-recess circle the LM's own web behind the flange
    (LM_WEB_REAR_Z), outside it (top bar, joint zone) the core rear plane -- the two levels the LM already has, met
    along its recess-edge ledge. Removes the old duct covers, the old UM exit mouth and the old voids below them; the
    woofer's M5 insert bosses keep their own rear (WOOFER_BOSS_REAR_Z)."""
    import manifold3d as m3
    import numpy as np
    from shapely.geometry import Point, box as sbox
    from shapely.geometry.polygon import orient
    from shapely.ops import unary_union
    zone = unary_union([sbox(0.0, y_um, TOP_ZONE_HALF_W, OLD_TAIL_END_Y), sbox(-TOP_ZONE_HALF_W, y_t, 0.0, OLD_TAIL_END_Y)])
    # never across the live LM ducts upstream of the branches (the T duct runs along y~300 across the ring top)
    from shapely.geometry import LineString
    for up in upstream:
        if len(up) > 1: zone = zone.difference(LineString(up[:, :2]).buffer(UPSTREAM_KEEP_MM, resolution=24))
    circle = Point(L22_CUTOUT[0], L22_CUTOUT[1]).buffer(LM_RECESS_R, resolution=256)
    bosses = unary_union([Point(L22_CUTOUT[0] + L22_PILOT_PCD_MM / 2 * math.cos(math.radians(a)),
                                L22_CUTOUT[1] + L22_PILOT_PCD_MM / 2 * math.sin(math.radians(a))).buffer(WOOFER_BOSS_R, resolution=48)
                          for a in L22_PILOT_ANGLES_DEG])
    def prism(g, z1):
        out = []
        for q in getattr(g, 'geoms', [g]):
            if q.area < 1e-3: continue
            q = orient(q, sign=1.0)
            out.append(m3.CrossSection([np.asarray(q.exterior.coords)[:-1]] + [np.asarray(h.coords)[:-1] for h in q.interiors]).extrude(z1 + 30.0).translate([0, 0, -30.0]))
        return out
    parts = (prism(zone.intersection(circle).difference(bosses), LM_WEB_REAR_Z + 0.02) + prism(zone.difference(circle).difference(bosses), d.CORE_REAR_Z + 0.02)
             + prism(zone.intersection(bosses), WOOFER_BOSS_REAR_Z))
    return m3.Manifold.batch_boolean(parts, m3.OpType.Add)


def lm_um_t_mesh(lm_top_installed, um_installed, tessellate, lm_installed, stage='final'):
    """stage: 'pre_ducts' (old ducts filled from the first branch candidates, no new ducts: the UM duct is searched
    against it), 'pre_t' (+ the UM duct and its cover: the T duct is searched against it), 'final'."""
    if d.PAIR: return lm_um_t_pair_mesh(lm_top_installed, um_installed, tessellate, stage)
    import manifold3d as m3
    import numpy as np
    import trimesh

    def M(t): return m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))
    def box(x0, y0, z0, x1, y1, z1): return m3.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
    def cyl(c, r, z0=-50.0, z1=80.0): return m3.Manifold.cylinder(z1 - z0, r, r, 256).translate([c[0], c[1], z0])
    def chain(points, r):
        balls = [m3.Manifold.sphere(r, 24).translate(list(map(float, q))) for q in points]
        return m3.Manifold.batch_boolean([m3.Manifold.batch_hull([a, b]) for a, b in zip(balls, balls[1:])], m3.OpType.Add)

    # 1. the UM + T body (lower piece, sculpted body, raised ring) fused onto the keyed LM top
    part = (M(d.um_t_mesh(um_installed, tessellate, lm_installed, stage='lm_base')) - cyl(L22_CUTOUT, LM_RECESS_R)) + M(lm_top_installed)
    #    The keyed LM top's upper captive-magnet pockets (magnets.side_magnet_sites lm_upper_left/right, D5 x 2 buried at a
    #    print pause) only align the side wings, which do not fit the T-below layouts: filled solid.
    for shell in lm_top_installed.split(only_watertight=False):
        if shell.volume < -1.0: part = part + M(shell.convex_hull)
    # 2. the LM-joint zone solid: the LM top bar and the UM's joint piece (half-laps, fit gaps, tie bores) become one,
    #    outside the woofer's flange recess and cutout
    woofer = cyl(L22_CUTOUT, WOOFER_CUTOUT_R) + cyl(L22_CUTOUT, LM_RECESS_R, LM_SEAT_Z, 80.0)
    #    ... and full depth under the part's own footprint there (the body's bottom flare was a front layer only, with
    #    the LM's half-lap tongue behind it)
    from shapely.geometry import box as sbox
    from shapely.geometry.polygon import orient as _orient
    sec = part.slice(JOINT_FOOTPRINT_Z)
    foot = m3.CrossSection(sec.to_polygons()) ^ m3.CrossSection([np.asarray(sbox(-TOP_ZONE_HALF_W, JOINT_Y0, TOP_ZONE_HALF_W, d.SPLIT_Y + 1.0).exterior.coords)[:-1]])
    part = part + (foot.extrude(THICKNESS_MM - JOINT_Z0).translate([0, 0, JOINT_Z0]) - cyl(L22_CUTOUT, LM_RECESS_R))
    part = part + (box(-JOINT_HALF_W, JOINT_Y0, JOINT_Z0, JOINT_HALF_W, d.SPLIT_Y + 1.0, THICKNESS_MM) - cyl(L22_CUTOUT, LM_RECESS_R))
    # 3. the LM's old ducts past each branch: filled, and their rear covers removed down to the rear the part would
    #    have without them (interpolated from the surrounding rear surface: no ledges; the old UM exit mouth goes too);
    #    the new duct's own cover takes over at the branch
    stored = stage == 'final'
    fills, tails, ups = [], [], []
    for which, r in (('um', UM_DUCT_R), ('t', TS_DUCT_R)):
        rel = released_route(which)
        j = branch_index(which) if (stored or (stage == 'pre_t' and which == 'um')) else branch_index(which, BRANCH_Y_RANGE[0])
        tail = rel[j:]
        tail = tail[tail[:, 1] <= OLD_TAIL_END_Y]
        fills.append(chain(tail[::2], r * 1.08)); tails.append(float(rel[j, 1])); ups.append(rel[max(j - 400, 0):j + 1])
    part = part + (m3.Manifold.batch_boolean(fills, m3.OpType.Add) - woofer)
    part = part - _top_rear_cutter(*tails, upstream=ups)
    # 4. covers of the new ducts (where they climb behind the rear), before the driver cutters
    ducts = []
    if stage in ('pre_t', 'final'): ducts.append('um')
    if stage == 'final': ducts.append('t')
    for which in ducts:
        pts, cover, w, h = duct(which)
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        k = np.flatnonzero(arc <= cover + 2.0)[::2]
        if len(k) > 1:
            # starting COVER_LEAD_IN_MM back along the LM's own duct, so it overlaps the LM's cover across the branch
            rel = released_route(which); j = branch_index(which)
            lead = rel[max(j - int(2 * COVER_LEAD_IN_MM), 0):j:2]
            cp = np.r_[lead, pts[k]]; cw = np.r_[np.full(len(lead), w[0]), w[k]]; ch = np.r_[np.full(len(lead), h[0]), h[k]]
            cov = [d.stadium_tube(cp, cw + 2 * d.UM_GUIDE_WALL_MM, ch + 2 * d.UM_GUIDE_WALL_MM)]
            kb = k[pts[k, 2] + h[k] / 2 + d.UM_GUIDE_WALL_MM < d.FRAME_REAR_Z + 0.5]
            if len(kb) > 1:     # fused with the rear where it runs behind it (no air sliver)
                cov += [d.stadium_tube(pts[kb] + [0.0, 0.0, dz], w[kb] + 2 * d.UM_GUIDE_WALL_MM, h[kb] + 2 * d.UM_GUIDE_WALL_MM) for dz in (0.5, 1.0)]
            part = part + (m3.Manifold.batch_boolean(cov, m3.OpType.Add) - woofer)
    # 5. driver: recess, window, pockets, lead channel, M3 heat-set insert bores
    for c in d.um_t_cutters(): part = part - M(tessellate(c))
    # 6. the new ducts, each continuing the LM's own duct 3 mm back from its branch
    for which in ducts:
        pts, cover, w, h = duct(which)
        rel = released_route(which); j = branch_index(which)
        pre = rel[max(j - 6, 0):j]; k = np.r_[np.arange(0, len(pts) - 1, 2), len(pts) - 1]
        part = part - d.stadium_tube(np.r_[pre, pts[k]], np.r_[np.full(len(pre), w[0]), w[k]], np.r_[np.full(len(pre), h[0]), h[k]])
    if part.status() != m3.Error.NoError or part.is_empty(): raise RuntimeError(f'LM + T + UM boolean failed: {part.status()}')
    out = part.to_mesh()
    mesh = trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=True)
    if stage != 'final': return mesh
    # one solid: the main shell plus its sealed cavities (the keyed LM top's own closed chambers); stray slivers go
    bodies = mesh.split(only_watertight=False)
    main = max(bodies, key=lambda b: abs(b.volume))
    stray = [b.volume for b in bodies if b is not main and b.volume > 0]
    if any(v > 1000.0 for v in stray): raise RuntimeError(f'LM + T + UM: disconnected material {stray} mm3')
    cav = [b for b in bodies if b is not main and b.volume < -0.5]
    if cav: raise RuntimeError(f'LM + T + UM: sealed cavities left {[np.round(c.bounds.mean(0), 1).tolist() for c in cav]}')
    return main                                  # zero-volume boolean slivers dropped


def facts():
    return dict(part=('keyed LM top variant + T + UM piece (pair, released LM--UM joint)' if d.PAIR else 'LM keyed top + T + UM (integrated)'), driver=d.DRIVER.label, orientation=d.ORIENTATION,
                seam_y_mm=SEAM_Y, mu10_shift_mm=round(d.MU10_SHIFT, 3), plate_margin_above_split_mm=d.PLATE_MARGIN_BELOW,
                fixing=d.DRIVER.screw, insert_bore_mm=[d.M3_INSERT_BORE_D_MM, d.M3_INSERT_BORE_DEPTH_MM])


SUPPORT_BLOCKER_GROW_MM = 0.25          # into the duct walls, as the released carrier blockers (obiwan_support_blocker)
RELEASED_LM_TOP_BLOCKER = 'build/no_floor_stand/support_blockers/obiwan_optional_lm_keyed_2_of_2_top.support_blocker.json'


def support_blocker(root):
    """Slicer-only support blocker for the integrated part, in installed coordinates: the released keyed LM top's own
    duct blocker (its LM ducts up to the branches; past them it sits in now-solid material, where it is harmless) plus
    the new UM and T ducts at their real per-point section and their 3 mm lead-in, grown SUPPORT_BLOCKER_GROW_MM into
    the walls. Returns (trimesh, duct collision contract: polylines + radii a support bead must stay out of)."""
    import numpy as np
    import trimesh
    import manifold3d as m3
    from pathlib import Path
    cfg = json.loads((Path(root) / RELEASED_LM_TOP_BLOCKER).read_text())
    lm = trimesh.load(Path(root) / Path(RELEASED_LM_TOP_BLOCKER).parent / cfg['support_blocker'])
    lm.apply_transform(np.linalg.inv(np.asarray(cfg['source_to_stl_matrix'])))
    parts = [m3.Manifold(m3.Mesh(vert_properties=np.asarray(lm.vertices, np.float32), tri_verts=np.asarray(lm.faces, np.uint32)))]
    regions = []
    for which in ('um', 't'):
        pts, cover, w, h = duct(which)
        rel = released_route(which); j = branch_index(which)
        pre = rel[max(j - 6, 0):j]; k = np.r_[np.arange(0, len(pts) - 1, 2), len(pts) - 1]
        P = np.r_[pre, pts[k]]; W = np.r_[np.full(len(pre), w[0]), w[k]]; H = np.r_[np.full(len(pre), h[0]), h[k]]
        parts.append(d.stadium_tube(P, W + 2 * SUPPORT_BLOCKER_GROW_MM, H + 2 * SUPPORT_BLOCKER_GROW_MM))
        regions.append(dict(kind='polyline_tube', name=f'{which}_duct', points_xyz_mm=np.round(P, 4).tolist(),
                            radius_mm=round(float(min(W.min(), H.min())) / 2, 3)))
        up = rel[(rel[:, 1] >= SEAM_Y + 1.0) & (np.arange(len(rel)) < j)]
        regions.append(dict(kind='polyline_tube', name=f'{which}_lm_duct_upstream', points_xyz_mm=np.round(up, 4).tolist(),
                            radius_mm=float(UM_DUCT_R if which == 'um' else TS_DUCT_R)))
    if d._CH is not None:   # the AMT's lead channel: the T duct ends in it; open at the rear (up when printed), no support
        from shapely import affinity
        from shapely.geometry.polygon import orient
        ch = orient(affinity.translate(d._CH, 0.0, d.PLATE_CENTER_Y).buffer(SUPPORT_BLOCKER_GROW_MM), sign=1.0)
        parts.append(m3.CrossSection([np.asarray(ch.exterior.coords)[:-1]]).extrude(THICKNESS_MM + 5.0).translate([0, 0, -5.0]))
        regions.append(dict(kind='plan_prism', name='lead_channel', outline_xy_mm=np.round(np.asarray(ch.exterior.coords), 3).tolist()))
    out = m3.Manifold.batch_boolean(parts, m3.OpType.Add).to_mesh()
    mesh = trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=True)
    return mesh, dict(coordinate_space='installed_mm', modifier_clearance_mm=SUPPORT_BLOCKER_GROW_MM, regions=regions)


PAIR_COVER_SPLIT_Y = 315.0              # a new duct's rear cover belongs to the LM top below this, to the T + UM piece above
PAIR_JOINT_BAND_Y = (280.0, 345.0)      # where the two prints meet (fit-gap shells are built only here)
PAIR_SIDE_TRIM_X = 37.0                 # beyond the joint's ears (outer x 36.9) the piece stops at the LM top's outline


def _pair_parts(lm_top_installed, um_installed, tessellate, stage):
    """The keyed LM top variant and the T + UM piece before the new ducts are cut, each with the new ducts' covers on
    its own side of the joint, plus the shared tools: (lm, piece, cutters, fit-gap shells, ducts)."""
    import manifold3d as m3
    import numpy as np

    def M(t): return m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))
    def box(x0, y0, z0, x1, y1, z1): return m3.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
    def cyl(c, r, z0=-50.0, z1=80.0): return m3.Manifold.cylinder(z1 - z0, r, r, 256).translate([c[0], c[1], z0])
    def chain(points, r):
        balls = [m3.Manifold.sphere(r, 24).translate(list(map(float, q))) for q in points]
        return m3.Manifold.batch_boolean([m3.Manifold.batch_hull([a, b]) for a, b in zip(balls, balls[1:])], m3.OpType.Add)
    grow = lambda m: m.minkowski_sum(m3.Manifold.cube([2 * d.LM_FIT_GAP_MM] * 3, center=True))
    band = box(-200, PAIR_JOINT_BAND_Y[0], -50, 200, PAIR_JOINT_BAND_Y[1], 80)
    woofer = cyl(L22_CUTOUT, WOOFER_CUTOUT_R) + cyl(L22_CUTOUT, LM_RECESS_R, LM_SEAT_Z, 80.0)

    lm = M(lm_top_installed)
    # the keyed LM top's upper captive-magnet pockets: no wings on the T-below layouts, filled (as the one piece); the
    # piece's body (cut by the released LM's fit gap, pockets open) is kept out of them too
    pockets = [M(shell.convex_hull) for shell in lm_top_installed.split(only_watertight=False) if shell.volume < -1.0]
    for q in pockets: lm = lm + q
    lm_fit = grow(lm ^ band)
    # the T + UM piece on the released joint (its LM fit gap kept), outside the woofer's flange recess
    piece = M(d.um_t_mesh(um_installed, tessellate, lm_top_installed, stage='pair_base')) - cyl(L22_CUTOUT, LM_RECESS_R)
    for q in pockets: piece = piece - grow(q)
    # outside the joint the frame's straight sides stop at the LM top's outline (its plan silhouette plus the fit gap),
    # so no tip of the piece reaches into a recess of the LM's outer edge; between the ears the released half-lap stays
    sil = (lm ^ band).project().offset(d.LM_FIT_GAP_MM, m3.JoinType.Round)
    sides = m3.CrossSection.square([200.0, 400.0]).translate([PAIR_SIDE_TRIM_X, 0.0])
    sides = sides + m3.CrossSection.square([200.0, 400.0]).translate([-200.0 - PAIR_SIDE_TRIM_X, 0.0])
    piece = piece - (sil ^ sides).extrude(100.0).translate([0.0, 0.0, -50.0])
    piece_fit = grow(piece ^ band)
    # the LM's old ducts past each branch: filled up to the joint (the piece fills the released UM's own T lumen),
    # their rear covers removed (as the one piece)
    stored = stage == 'final'
    fills, tails, ups = [], [], []
    for which, r in (('um', UM_DUCT_R), ('t', TS_DUCT_R)):
        rel = released_route(which)
        j = branch_index(which) if (stored or (stage == 'pre_t' and which == 'um')) else branch_index(which, BRANCH_Y_RANGE[0])
        tail = rel[j:]
        tail = tail[tail[:, 1] <= OLD_TAIL_END_Y]
        fills.append(chain(tail[::2], r * 1.08)); tails.append(float(rel[j, 1])); ups.append(rel[max(j - 400, 0):j + 1])
    lm = lm + ((m3.Manifold.batch_boolean(fills, m3.OpType.Add) ^ box(-200, 0, -50, 200, PAIR_COVER_SPLIT_Y, 80)) - woofer - piece_fit)
    rear = _top_rear_cutter(*tails, upstream=ups)
    lm, piece = lm - rear, piece - rear
    # the new ducts' covers where they climb behind the rear: split at PAIR_COVER_SPLIT_Y between the two prints
    ducts = []
    if stage in ('pre_t', 'final'): ducts.append('um')
    if stage == 'final': ducts.append('t')
    for which in ducts:
        pts, cover, w, h = duct(which)
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        k = np.flatnonzero(arc <= cover + 2.0)[::2]
        if len(k) < 2: continue
        rel = released_route(which); j = branch_index(which)
        lead = rel[max(j - int(2 * COVER_LEAD_IN_MM), 0):j:2]
        cp = np.r_[lead, pts[k]]; cw = np.r_[np.full(len(lead), w[0]), w[k]]; ch = np.r_[np.full(len(lead), h[0]), h[k]]
        cov = [d.stadium_tube(cp, cw + 2 * d.UM_GUIDE_WALL_MM, ch + 2 * d.UM_GUIDE_WALL_MM)]
        kb = k[pts[k, 2] + h[k] / 2 + d.UM_GUIDE_WALL_MM < d.FRAME_REAR_Z + 0.5]
        if len(kb) > 1:
            cov += [d.stadium_tube(pts[kb] + [0.0, 0.0, dz], w[kb] + 2 * d.UM_GUIDE_WALL_MM, h[kb] + 2 * d.UM_GUIDE_WALL_MM) for dz in (0.5, 1.0)]
        cov = m3.Manifold.batch_boolean(cov, m3.OpType.Add) - woofer
        lm = lm + ((cov ^ box(-200, 0, -50, 200, PAIR_COVER_SPLIT_Y, 80)) - piece_fit)
        piece = piece + ((cov ^ box(-200, PAIR_COVER_SPLIT_Y + d.LM_FIT_GAP_MM, -50, 200, 900, 80)) - lm_fit)
    cutters = [M(tessellate(c)) for c in d.um_t_cutters()]
    yw = d.tie_window_y()
    tie_bore = m3.Manifold.cylinder(yw + 1.0 - d.TIE['y0'], d.TIE['cbore_r'], d.TIE['cbore_r'], 48).rotate([-90, 0, 0]).translate(
        [d.TIE['x'], d.TIE['y0'], d.TIE['z']])
    return lm, piece, cutters + [tie_bore], (lm_fit, piece_fit), ducts


def _pair_duct_tube(which):
    import numpy as np
    pts, cover, w, h = duct(which)
    rel = released_route(which); j = branch_index(which)
    pre = rel[max(j - 6, 0):j]; k = np.r_[np.arange(0, len(pts) - 1, 2), len(pts) - 1]
    return d.stadium_tube(np.r_[pre, pts[k]], np.r_[np.full(len(pre), w[0]), w[k]], np.r_[np.full(len(pre), h[0]), h[k]])


def lm_um_t_pair_mesh(lm_top_installed, um_installed, tessellate, stage='final'):
    """Pair variant. Search stages ('pre_ducts', 'pre_t'): ONE solid -- both prints with the fit gap between them closed
    (each duct is searched as one centreline through the joint; the tie and ear bores stay open). 'final': the two
    prints (keyed LM top variant, T + UM piece), each with the same duct tubes cut, as (lm_top, piece) trimeshes."""
    import manifold3d as m3
    import numpy as np
    import trimesh
    lm, piece, cutters, (lm_fit, piece_fit), ducts = _pair_parts(lm_top_installed, um_installed, tessellate, stage)
    tool = m3.Manifold.batch_boolean(cutters, m3.OpType.Add)
    tubes = [_pair_duct_tube(w) for w in ducts]
    def tm(m):
        if m.status() != m3.Error.NoError or m.is_empty(): raise RuntimeError(f'pair boolean failed: {m.status()}')
        # sub-5 um boolean slivers (where the joint's fit gap meets the old lumen fill and the tie bore) would turn into
        # open edges once the vertices are merged: removed while the solid is still manifold
        o = m.simplify(0.005).to_mesh(); return trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=True)
    if stage != 'final':
        whole = lm + piece + (lm_fit ^ piece_fit) - tool
        for t in tubes: whole = whole - t
        return tm(whole)
    def M(t): return m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))
    def cut(m):
        m = m - tool
        for t in tubes: m = m - t
        return m
    # The piece's straight frame sides run down past the LM's outer arc; their tips there (with the UM duct's wall
    # through them) are cut off from the piece by the fit gap. They sit in the LM top's outline: the LM top takes them,
    # fused across the gap.
    pm = tm(cut(piece)).split(only_watertight=False)
    pmain = max(pm, key=lambda b: abs(b.volume))
    tips = [b for b in pm if b is not pmain and b.volume > 1.0 and b.bounds[1][1] < PAIR_COVER_SPLIT_Y]
    if tips:
        print('pair: frame tips cut off by the fit gap, given to the LM top', [(round(t_.volume, 1), np.round(t_.bounds, 1).tolist()) for t_ in tips], flush=True)
        grow = lambda m: m.minkowski_sum(m3.Manifold.cube([2 * d.LM_FIT_GAP_MM + 0.02] * 3, center=True))
        T_ = m3.Manifold.batch_boolean([M(b) for b in tips], m3.OpType.Add)
        lm = lm + T_ + ((grow(T_) ^ lm_fit) - piece_fit)      # grown only into the gap to the LM top
    out = []
    for name, m in (('keyed LM top', lm), ('T + UM piece', piece)):
        mesh = tm(cut(m))
        # small air pockets sealed where a new cover arches past the filled magnet slot / the LM's outer edge: filled
        cav = [b for b in mesh.split(only_watertight=False) if b.volume < -0.5]
        if cav and all(-b.volume < 50.0 for b in cav):
            print(f'pair {name}: filled sealed air pockets', [(round(-b.volume, 2), np.round(b.bounds.mean(0), 1).tolist()) for b in cav], flush=True)
            mesh = tm(cut(m + m3.Manifold.batch_boolean([M(b.convex_hull) for b in cav], m3.OpType.Add)))
        bodies = mesh.split(only_watertight=False)
        main = max(bodies, key=lambda b: abs(b.volume))
        stray = [b for b in bodies if b is not main and b.volume > 5.0 and not (name == 'T + UM piece' and any(b is t_ or np.allclose(b.bounds, t_.bounds, atol=0.05) for t_ in tips))]
        if stray: raise RuntimeError(f'pair {name}: disconnected material ' + str([(round(b.volume, 1), np.round(b.bounds, 1).tolist()) for b in stray]))
        cav = [b for b in bodies if b is not main and b.volume < -0.5]
        if cav: raise RuntimeError(f'pair {name}: sealed cavities left {[np.round(c.bounds.mean(0), 1).tolist() for c in cav]}')
        out.append(main)
    return tuple(out)


def pair_support_blockers(root, prints):
    """Pair variant: {'lm_top': (mesh, contract), 'um_t': (mesh, contract)} in installed coordinates. `prints`: the built
    {'lm_top': trimesh, 'um_t': trimesh}. Each print gets the stretches of the new ducts it holds (centreline inside it or
    within 1.5 mm of it, one point of overlap each way); the LM top also keeps the released keyed top's own duct blocker
    (its LM ducts up to the branches)."""
    import numpy as np
    import trimesh
    import manifold3d as m3
    from pathlib import Path
    from trimesh.proximity import signed_distance
    cfg = json.loads((Path(root) / RELEASED_LM_TOP_BLOCKER).read_text())
    rel_blk = trimesh.load(Path(root) / Path(RELEASED_LM_TOP_BLOCKER).parent / cfg['support_blocker'])
    rel_blk.apply_transform(np.linalg.inv(np.asarray(cfg['source_to_stl_matrix'])))
    ducts = []
    for which in ('um', 't'):
        pts, cover, w, h = duct(which)
        rel = released_route(which); j = branch_index(which)
        pre = rel[max(j - 6, 0):j]; k = np.r_[np.arange(0, len(pts) - 1, 2), len(pts) - 1]
        P = np.r_[pre, pts[k]]; W = np.r_[np.full(len(pre), w[0]), w[k]]; H = np.r_[np.full(len(pre), h[0]), h[k]]
        up = rel[(rel[:, 1] >= SEAM_Y + 1.0) & (np.arange(len(rel)) < j)]
        ducts.append((which, P, W, H, up))
    out = {}
    for side, part in prints.items():
        parts = [m3.Manifold(m3.Mesh(vert_properties=np.asarray(rel_blk.vertices, np.float32), tri_verts=np.asarray(rel_blk.faces, np.uint32)))] if side == 'lm_top' else []
        regs = []
        for which, P, W, H, up in ducts:
            # owned where a ring 0.4 mm into the duct's wall lies in this print
            T_ = np.gradient(P, axis=0); T_ /= np.linalg.norm(T_, axis=1, keepdims=True)
            sd = np.cross(T_, [0.0, 0.0, 1.0]); sd /= np.maximum(np.linalg.norm(sd, axis=1, keepdims=True), 1e-9); upv = np.cross(sd, T_)
            ring = [P + np.cos(a_) * (W[:, None] / 2 + 0.4) * sd + np.sin(a_) * (H[:, None] / 2 + 0.4) * upv
                    for a_ in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
            near = np.any([part.contains(q) for q in ring], axis=0)
            keep = near | np.r_[near[1:], False] | np.r_[False, near[:-1]]
            idx = np.flatnonzero(keep)
            for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                if len(run) < 2: continue
                parts.append(d.stadium_tube(P[run], W[run] + 2 * SUPPORT_BLOCKER_GROW_MM, H[run] + 2 * SUPPORT_BLOCKER_GROW_MM))
                regs.append(dict(kind='polyline_tube', name=f'{which}_duct', points_xyz_mm=np.round(P[run], 4).tolist(),
                                 radius_mm=round(float(min(W[run].min(), H[run].min())) / 2, 3)))
            if side == 'lm_top' and len(up) > 1:
                regs.append(dict(kind='polyline_tube', name=f'{which}_lm_duct_upstream', points_xyz_mm=np.round(up, 4).tolist(),
                                 radius_mm=float(UM_DUCT_R if which == 'um' else TS_DUCT_R)))
        o = m3.Manifold.batch_boolean(parts, m3.OpType.Add).to_mesh()
        mesh = trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=True)
        out[side] = (mesh, dict(coordinate_space='installed_mm', modifier_clearance_mm=SUPPORT_BLOCKER_GROW_MM, regions=regs))
    return out
