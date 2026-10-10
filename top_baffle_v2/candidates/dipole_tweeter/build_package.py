"""Approved Obiwan dipole-tweeter parts for the H2C shelf: installed-frame STLs, print authorities, support blockers.

The geometry is built by scripts/build_dipole_tweeter.py (mesh booleans on the audited Obiwan parts, run on Osado) into
build/planar_pt2522/ and build/amt17d2/; this script only packages the approved meshes:
  * STL/<part>.stl + <part>.print.json (identity print authority, installed frame);
  * support_blockers/<part>.support_blocker.{stl,json} (release schema). T below (one piece and pair): the blockers
    written next to the built meshes by lm_um_t.support_blocker / pair_support_blockers. T above: the released UM
    carrier's own lumen blocker up to the T branch, plus the new T duct (grown 0.25 mm) and, for the AMT, its lead
    channel. Contracts keep the polyline duct regions only (the shelf audit reads capsules); the AMT lead channel stays
    in the blocker geometry.
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


def above_info(family, orient):
    """The T-above T duct, the driver centre and (AMT) the lead channel outline, from dipole_tweeter in a subprocess
    (its configuration is read from the environment at import)."""
    env = dict(os.environ, LX_ROUTING_PROFILE='obiwan', LX_DIPOLE_T_DRIVER=DRIVER_KEY[family],
               LX_DIPOLE_T_ORIENTATION='landscape' if orient == 'rotated' else 'portrait', LX_DIPOLE_T_INSERTS='1',
               PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in ('src', 'scripts', 'tools', 'vendor/GRS/PT2522-4')))
    env.pop('LX_DIPOLE_T_INTEGRATED', None); env.pop('LX_DIPOLE_T_PAIR', None)
    code = ('import json, numpy as np\nfrom shapely import affinity\nimport lx521_baffle.obiwan.dipole_tweeter as d\n'
            '_, (pts, r) = d.um_t_above_tools()\ncx, cy = d.ABOVE_CENTER\n'
            'ch = None if d._CH is None else np.asarray(affinity.translate(d._CH, cx, cy).exterior.coords).tolist()\n'
            'print("JSON" + json.dumps(dict(points=np.asarray(pts).tolist(), r=float(r), channel=ch, z_front=float(d.THICKNESS_MM + 5.0))))\n')
    out = subprocess.run([sys.executable, '-c', code], env=env, cwd=ROOT, text=True, capture_output=True, check=True).stdout
    return json.loads(next(l for l in out.splitlines() if l.startswith('JSON'))[4:])


def above_blocker(family, orient):
    import manifold3d as m3
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient as _orient
    M = lambda t: m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))
    info = above_info(family, orient)
    P = np.asarray(info['points'], float); r = info['r']
    cfg = json.loads(RELEASED_UM_BLOCKER.read_text())
    rel = trimesh.load(RELEASED_UM_BLOCKER.with_name(cfg['support_blocker']))
    rel.apply_transform(np.linalg.inv(np.asarray(cfg['source_to_stl_matrix'])))
    # the released lumen past the T branch is filled in the T-above parts: keep its region only up to the branch
    (lumen,) = [g for g in cfg['duct_collision_contract']['regions'] if g['name'] == 'um_carrier_t_route_lumen']
    L = np.asarray(lumen['points_xyz_mm'], float)
    k = int(np.argmin(np.linalg.norm(L - P[0], axis=1)))
    regions = [dict(kind='polyline_tube', name='um_carrier_t_route_lumen', points_xyz_mm=np.round(L[:k + 1], 4).tolist(),
                    radius_mm=lumen['radius_mm']),
               dict(kind='polyline_tube', name='t_duct', points_xyz_mm=np.round(P[::2], 4).tolist(), radius_mm=round(r, 3))]
    balls = [m3.Manifold.sphere(r + GROW, 24).translate(list(map(float, q))) for q in P[::2]]
    parts = [M(rel), m3.Manifold.batch_boolean([m3.Manifold.batch_hull([a, b]) for a, b in zip(balls, balls[1:])], m3.OpType.Add)]
    if info['channel']:
        ch = _orient(Polygon(info['channel']).buffer(GROW), sign=1.0)
        parts.append(m3.CrossSection([np.asarray(ch.exterior.coords)[:-1]]).extrude(info['z_front'] + 5.0).translate([0, 0, -5.0]))
    o = m3.Manifold.batch_boolean(parts, m3.OpType.Add).to_mesh()
    mesh = trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=True)
    return mesh, dict(coordinate_space='installed_mm', modifier_clearance_mm=GROW, regions=regions)


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
        if layout == 't_above':
            bmesh, contract = above_blocker(family, orient)
        else:
            built = BUILD[family] / f'{STEM[family]}_{suffix}.support_blocker.json'
            cfg = json.loads(built.read_text())
            bmesh = trimesh.load(built.with_name(cfg['support_blocker']))
            contract = cfg['duct_collision_contract']
            contract = dict(contract, regions=[g for g in contract['regions'] if g['kind'] == 'polyline_tube'])
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
