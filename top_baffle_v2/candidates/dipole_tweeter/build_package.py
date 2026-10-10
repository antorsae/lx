"""Approved Obiwan dipole-tweeter parts for the H2C shelf: installed-frame STLs, print authorities, support blockers.

The geometry is built by scripts/build_dipole_tweeter.py (mesh booleans on the audited Obiwan parts, run on Osado) into
build/planar_pt2522/ and build/amt17d2/; this script only packages the approved meshes:
  * STL/<part>.stl + <part>.print.json (identity print authority, installed frame);
  * support_blockers/<part>.support_blocker.{stl,json} (release schema), see blocker(): live ducts only (the released
    lumens up to each branch, the new ducts where the print holds them), every M3 insert bore (MU10 and driver), the
    released bore blockers that still exist, the AMT lead channel. Contracts list the polyline ducts (the shelf audit
    reads capsules); the AMT lead channel stays in the blocker geometry.
usage: build_package.py            (from the repo root or anywhere)
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
PT, AMT = 'grs_pt2522_4', 'mundorf_amt17d2_1'
BUILD = {PT: ROOT / 'build/planar_pt2522', AMT: ROOT / 'build/amt17d2'}
STEM = {PT: 'obiwan_planar_pt2522', AMT: 'obiwan_amt17d2'}
DRIVER_KEY = {PT: 'pt2522_4', AMT: 'amt17d2_1'}
# (family, layout, orientation, part or None) -> built mesh suffix
PARTS = {
    (PT, 't_above', 'upright', None): 'um_t_above', (PT, 't_above', 'rotated', None): 'um_t_above_landscape',
    (PT, 't_below', 'rotated', None): 'lm_um_t_landscape',
    (PT, 't_below', 'upright', 'lm_top'): 'pair_lm_top', (PT, 't_below', 'upright', 'um_t'): 'pair_um_t',
    (AMT, 't_above', 'upright', None): 'um_t_above', (AMT, 't_above', 'rotated', None): 'um_t_above_landscape',
    (AMT, 't_below', 'upright', None): 'lm_um_t', (AMT, 't_below', 'rotated', None): 'lm_um_t_landscape',
}
RELEASED_UM_BLOCKER = ROOT / 'build/no_floor_stand/support_blockers/obiwan_core_2_of_2_um_carrier.support_blocker.json'
GROW = 0.25


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def name(family, layout, orient, part):
    return f'{family}_obiwan_{layout}_{orient}' + (f'_{part}' if part else '')


# (family, layout, orientation, part) -> the dipole_tweeter configuration it was built with
def variant_env(family, layout, orient, part):
    env = dict(LX_ROUTING_PROFILE='obiwan', LX_DIPOLE_T_DRIVER=DRIVER_KEY[family], LX_DIPOLE_T_INSERTS='1',
               LX_DIPOLE_T_ORIENTATION='landscape' if orient == 'rotated' else 'portrait')
    if layout == 't_below':
        env['LX_DIPOLE_T_INTEGRATED'] = '1'
        if part: env['LX_DIPOLE_T_PAIR'] = '1'
    return env


INFO = r"""
import json, os, numpy as np
from shapely import affinity
import lx521_baffle.obiwan.dipole_tweeter as d
above = os.environ.get('LX_DIPOLE_T_INTEGRATED') != '1'
if above:
    _, (pts, r) = d.um_t_above_tools()
    cx, cy = d.ABOVE_CENTER; floor = d.THICKNESS_MM; shift = 0.0
    lum = np.asarray(d._lumen()[0])
    k = int(np.argmin(np.linalg.norm(lum - pts[0], axis=1)))
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.asarray(pts), axis=0), axis=1))]
    dd = np.maximum(2 * r, d.t_taper_d(arc))                # the AMT's D4 duct opens to the lumen's D6 at the branch
    ducts = [dict(name='t', points=np.asarray(pts).tolist(), w=dd.tolist(), h=dd.tolist(),
                  stale=lum[k + 6:].tolist(), stale_r=3.0, upstream=None)]
else:
    import lx521_baffle.obiwan.lm_um_t as L
    cx, cy = 0.0, d.PLATE_CENTER_Y; floor = d.THICKNESS_MM - d.PLATE_T; shift = d.MU10_SHIFT
    ducts = []
    for which in ('um', 't'):
        pts, cover, w, h = L.duct(which)
        rel = L.released_route(which); j = L.branch_index(which)
        pre = rel[max(j - 6, 0):j]
        up = rel[(rel[:, 1] >= L.SEAM_Y + 1.0) & (np.arange(len(rel)) < j)]
        ducts.append(dict(name=which, points=np.r_[pre, pts].tolist(), w=np.r_[np.full(len(pre), w[0]), w].tolist(),
                          h=np.r_[np.full(len(pre), h[0]), h].tolist(), stale=rel[j + 6:].tolist(),
                          stale_r=float(L.UM_DUCT_R if which == 'um' else L.TS_DUCT_R), upstream=up.tolist(),
                          upstream_r=float(L.UM_DUCT_R if which == 'um' else L.TS_DUCT_R)))
ch = None if d._CH is None else np.asarray(affinity.translate(d._CH, cx, cy).exterior.coords).tolist()
print('JSON' + json.dumps(dict(above=above, ducts=ducts, channel=ch, shift=float(shift), z_front=float(d.THICKNESS_MM),
      bores=[[cx + q.x, cy + q.y, floor - d.PILOT_DEPTH_MM, floor] for q in d._PIL], bore_d=float(d.PILOT_D_MM))))
"""


def variant_info(family, layout, orient, part):
    env = dict(os.environ, **variant_env(family, layout, orient, part),
               PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in ('src', 'scripts', 'tools', 'vendor/GRS/PT2522-4')))
    for k in ('LX_DIPOLE_T_INTEGRATED', 'LX_DIPOLE_T_PAIR', 'LX_DIPOLE_T_ROUTES'):
        if k not in variant_env(family, layout, orient, part): env.pop(k, None)
    out = subprocess.run([sys.executable, '-c', INFO], env=env, cwd=ROOT, text=True, capture_output=True, check=True).stdout
    return json.loads(next(l for l in out.splitlines() if l.startswith('JSON'))[4:])


RELEASED_LM_TOP_BLOCKER = ROOT / 'build/no_floor_stand/support_blockers/obiwan_optional_lm_keyed_2_of_2_top.support_blocker.json'
EARS = [(-32.0, 315.770102), (32.0, 315.770102)]        # LM--UM joint ear receivers (filled in the one-piece T below)
BORE_BLOCK_EXTRA = 0.3                                 # radial and axial, as the released insert-bore blockers


def released_blocker(path):
    cfg = json.loads(path.read_text())
    m = trimesh.load(path.with_name(cfg['support_blocker']))
    m.apply_transform(np.linalg.inv(np.asarray(cfg['source_to_stl_matrix'])))
    return m, cfg


def blocker(family, layout, orient, part, mesh):
    """Support blocker + duct contract of one packaged part (installed frame):
      * the released part's own blocker (UM carrier for T above, keyed LM top for T below / the pair's LM top): its duct
        lumens without the old tails past each branch (filled, or cut away behind the rebuilt rear), its small bore
        blockers kept -- except the LM--UM joint ears where the one-piece T below fills the joint;
      * the MU10's four M3 insert-bore blockers at the MU10's actual (raised) position (the released UM's, shifted);
      * the driver's four M3 insert-bore blockers;
      * the new ducts (grown GROW) where this print holds them, and the AMT lead channel;
    the contract lists the new ducts and the live released lumens up to the branches (polylines only)."""
    import manifold3d as m3
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient as _orient
    M = lambda t: m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))
    info = variant_info(family, layout, orient, part)
    sys.path[:0] = [str(ROOT / p) for p in ('src', 'scripts', 'tools', 'vendor/GRS/PT2522-4') if str(ROOT / p) not in sys.path]
    def chain(points, r):
        balls = [m3.Manifold.sphere(r, 20).translate(list(map(float, q))) for q in points]
        return m3.Manifold.batch_boolean([m3.Manifold.batch_hull([a, b]) for a, b in zip(balls, balls[1:])], m3.OpType.Add)
    def cyl(x, y, r, z0, z1):
        return m3.Manifold.cylinder(z1 - z0, r, r, 48).translate([x, y, z0])
    parts, regions = [], []
    holds_lm = layout == 't_above' or part in (None, 'lm_top')         # the print that carries the released lumens
    um_rel, um_cfg = released_blocker(RELEASED_UM_BLOCKER)
    if holds_lm:
        rel, cfg = released_blocker(RELEASED_UM_BLOCKER if layout == 't_above' else RELEASED_LM_TOP_BLOCKER)
        stale = m3.Manifold.batch_boolean([chain(np.asarray(g['stale'])[::2], g['stale_r'] + 1.0) for g in info['ducts']
                                           if len(g['stale']) > 2], m3.OpType.Add)
        for comp in rel.split(only_watertight=False):
            if comp.volume > 1000.0:                                      # duct lumens: drop the old tails
                parts.append(M(comp) - stale)
            elif not (layout == 't_below' and part is None and
                      any(np.hypot(*(comp.bounds.mean(0)[:2] - e)) < 3.0 for e in EARS)):
                if layout == 't_above' and comp.volume < 150.0: continue  # MU10 bores: re-added below, at the MU10
                parts.append(M(comp))
        # live released lumens up to the branches
        if layout == 't_above':
            (lumen,) = [g for g in cfg['duct_collision_contract']['regions'] if g['name'] == 'um_carrier_t_route_lumen']
            L_ = np.asarray(lumen['points_xyz_mm'], float); P0 = np.asarray(info['ducts'][0]['points'][0])
            k = int(np.argmin(np.linalg.norm(L_ - P0, axis=1)))
            regions.append(dict(kind='polyline_tube', name='um_carrier_t_route_lumen', points_xyz_mm=np.round(L_[:k + 1], 4).tolist(),
                                radius_mm=lumen['radius_mm']))
        else:
            for g in info['ducts']:
                regions.append(dict(kind='polyline_tube', name=f"{g['name']}_lm_duct_upstream",
                                    points_xyz_mm=np.round(g['upstream'], 4).tolist(), radius_mm=g['upstream_r']))
    holds_mu10 = layout == 't_above' or part in (None, 'um_t')
    if holds_mu10:
        for comp in um_rel.split(only_watertight=False):
            if comp.volume < 150.0: parts.append(M(comp).translate([0.0, info['shift'], 0.0]))
        rb = info['bore_d'] / 2 + BORE_BLOCK_EXTRA
        for x, y, z0, z1 in info['bores']:
            parts.append(cyl(x, y, rb, z0 - BORE_BLOCK_EXTRA, z1 + BORE_BLOCK_EXTRA))
        if info['channel']:
            ch = _orient(Polygon(info['channel']).buffer(GROW), sign=1.0)
            parts.append(m3.CrossSection([np.asarray(ch.exterior.coords)[:-1]]).extrude(info['z_front'] + 5.0).translate([0, 0, -5.0]))
    from lx521_baffle.obiwan.dipole_tweeter import stadium_tube
    for g in info['ducts']:
        P = np.asarray(g['points'], float); W = np.asarray(g['w'], float); H = np.asarray(g['h'], float)
        # the stretches this print holds: a ring 0.4 mm into the duct wall lies in it (the pair splits each duct)
        T_ = np.gradient(P, axis=0); T_ /= np.linalg.norm(T_, axis=1, keepdims=True)
        sd = np.cross(T_, [0.0, 0.0, 1.0]); sd /= np.maximum(np.linalg.norm(sd, axis=1, keepdims=True), 1e-9); up = np.cross(sd, T_)
        ring = [P + np.cos(a) * (W[:, None] / 2 + 0.4) * sd + np.sin(a) * (H[:, None] / 2 + 0.4) * up
                for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
        near = np.any([mesh.contains(q) for q in ring], axis=0)
        keep = near | np.r_[near[1:], False] | np.r_[False, near[:-1]]
        idx = np.flatnonzero(keep)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(run) < 2: continue
            run = run[::2] if len(run) > 4 else run
            parts.append(stadium_tube(P[run], W[run] + 2 * GROW, H[run] + 2 * GROW))
            regions.append(dict(kind='polyline_tube', name=f"{g['name']}_duct", points_xyz_mm=np.round(P[run], 4).tolist(),
                                radius_mm=round(float(min(W[run].min(), H[run].min())) / 2, 3)))
    o = m3.Manifold.batch_boolean(parts, m3.OpType.Add).to_mesh()
    out = trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=True)
    return out, dict(coordinate_space='installed_mm', modifier_clearance_mm=GROW, regions=regions)


def main():
    (PKG / 'STL').mkdir(exist_ok=True); (PKG / 'support_blockers').mkdir(exist_ok=True)
    rows = []
    for (family, layout, orient, part), suffix in PARTS.items():
        n = name(family, layout, orient, part)
        src = BUILD[family] / f'{STEM[family]}_{suffix}.stl'
        mesh = trimesh.load(src, process=True)
        if not mesh.is_watertight or len(mesh.split(only_watertight=False)) != 1:
            raise SystemExit(f'{src}: not one watertight solid')
        dst = PKG / 'STL' / f'{n}.stl'
        mesh.export(dst)
        json.dump(dict(schema_version=1, part=n, stl=dst.name, stl_sha256=sha(dst), source_stl=str(src.relative_to(ROOT)),
                       source_frame='Obi-Wan installed design frame: X lateral, Y up, Z forward (towards the listener), mm',
                       source_to_stl_matrix=np.eye(4).tolist(), qualification='digital checks passed; not yet printed'),
                  open(dst.with_suffix('.print.json'), 'w'), indent=1)
        bmesh, contract = blocker(family, layout, orient, part, mesh)
        bstl = PKG / 'support_blockers' / f'{n}.support_blocker.stl'
        bmesh.export(bstl)
        json.dump(dict(schema_version=1, kind='bambu_support_blocker', part=n, purpose='forbid_support_inside_functional_ducts',
                       main_stl=f'../STL/{dst.name}', main_stl_sha256=sha(dst), support_blocker=bstl.name,
                       support_blocker_sha256=sha(bstl), source_to_stl_matrix=np.eye(4).tolist(), modifier_clearance_mm=GROW,
                       duct_collision_contract=contract, watertight=bool(bmesh.is_watertight)),
                  open(bstl.with_suffix('.json'), 'w'), indent=1)
        rows.append((n, round(mesh.volume / 1e3, 1), np.round(mesh.extents, 1).tolist(), [g['name'] for g in contract['regions']]))
        print(*rows[-1], flush=True)


if __name__ == '__main__':
    main()
