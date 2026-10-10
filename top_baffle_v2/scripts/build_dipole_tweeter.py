#!/usr/bin/env python3
"""Build the candidate one-dipole-tweeter Obi-Wan parts (crescent, UM + T) and check the driver fit.

Run guarded:  scripts/run_memory_guarded.py -- python scripts/build_dipole_tweeter.py [crescent] [um_t]
                  [--driver pt2522_4|amt17d2_1] [--landscape]
um_t_above: one-piece UM + T above (whole UM, fused frame, hidden T duct; used for the AMT).
--landscape builds the driver turned 90 deg (terminals/leads to +x); outputs get a _landscape suffix.
Outputs: build/<driver name>/ (build/planar_pt2522, build/amt17d2).
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('LX_ROUTING_PROFILE', 'obiwan')
if 'lm_um_t' in sys.argv:                # the integrated LM + T + UM part: set before dipole_tweeter is imported
    os.environ['LX_DIPOLE_T_INTEGRATED'] = '1'
if '--pair' in sys.argv:                 # ... as two prints on the released LM--UM joint (PT2522 upright)
    sys.argv.remove('--pair'); os.environ['LX_DIPOLE_T_PAIR'] = '1'
if '--landscape' in sys.argv:            # read by dipole_tweeter at import
    sys.argv.remove('--landscape'); os.environ['LX_DIPOLE_T_ORIENTATION'] = 'landscape'
if '--driver' in sys.argv:
    i = sys.argv.index('--driver'); os.environ['LX_DIPOLE_T_DRIVER'] = sys.argv[i + 1]; del sys.argv[i:i + 2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tools'), str(ROOT / 'vendor/GRS/PT2522-4')]
import numpy as np, trimesh
from build123d import export_step, export_stl, Pos, Rot
from lx521_baffle.base import THICKNESS_MM
from lx521_baffle.obiwan import dipole_tweeter as pl

OUT = ROOT / 'build' / pl.DRIVER.name


def driver_at(cx, cy, surface=False):
    """The vendor envelope, front face out, in the installed orientation: plate/flange flush in the recess, or
    (surface=True) its rear face seated on the part front."""
    return Pos(cx, cy, THICKNESS_MM + (pl.PLATE_T if surface else 0.0)) * Rot(0, 0, pl.ROT_DEG) * pl.DRIVER.build()


def tessellate(shape):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.stl') as f:
        export_stl(shape, f.name, tolerance=0.01, angular_tolerance=0.08)
        return trimesh.load(f.name, process=True)


def released_installed(name):
    """An audited shelf part restored to installed coordinates."""
    job = next(j for j in json.loads((ROOT / 'to_print/catalog.json').read_text())['jobs'] if j['name'] == name)
    m = trimesh.load(ROOT / job['stl'], process=True); m.apply_transform(np.linalg.inv(np.asarray(job['source_to_stl_matrix'])))
    return m


def released_um_installed():
    return released_installed('h2c_obiwan_core_2_of_2_um_carrier')


def mesh_clash(a, b):
    import manifold3d as m3
    M = lambda t: m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))
    return float((M(a) ^ M(b)).volume())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    facts_path = OUT / 'facts.json'
    report = json.loads(facts_path.read_text()) if facts_path.exists() else dict(parts={})
    report['facts' + ('_landscape' if pl.LANDSCAPE else '')] = pl.facts()
    wanted = set(sys.argv[1:]) or {'crescent', 'um_t'}
    for key, center in (('crescent', pl.CRESCENT_CENTER), ('um_t', pl.um_t_plate_center()), ('um_t_above', pl.ABOVE_CENTER),
                        ('lm_um_t', pl.um_t_plate_center())):
        if key not in wanted: continue
        name = f'obiwan_{pl.DRIVER.name}_{key}' + ('_landscape' if pl.LANDSCAPE else '')
        if key == 'lm_um_t' and pl.PAIR:
            from lx521_baffle.obiwan import lm_um_t
            lm_top, piece = lm_um_t.lm_um_t_mesh(released_installed('h2c_obiwan_lm_keyed_top'), released_um_installed(),
                                                 tessellate, None)
            drv = tessellate(driver_at(*center))
            for suffix, mesh in (('pair_lm_top', lm_top), ('pair_um_t', piece)):
                pname = f'obiwan_{pl.DRIVER.name}_{suffix}' + ('_landscape' if pl.LANDSCAPE else '')
                mesh.export(OUT / f'{pname}.stl')
                clash = mesh_clash(mesh, drv)
                report['parts'][pname] = dict(volume_cm3=round(mesh.volume / 1000, 1), bounds_mm=np.round(mesh.bounds, 2).tolist(),
                                              watertight=bool(mesh.is_watertight), driver_clash_mm3=round(clash, 3), step=None,
                                              method='mesh booleans: keyed LM top variant + T + UM piece on the released joint (lm_um_t pair)',
                                              **lm_um_t.facts())
                print(pname, report['parts'][pname], flush=True)
                if clash > 0.01: raise SystemExit(f'{pname}: driver envelope intersects the part ({clash:.3f} mm3)')
            between = mesh_clash(lm_top, piece)
            print('pair overlap mm3', round(between, 4), flush=True)
            if between > 0.01: raise SystemExit(f'pair: the two prints overlap ({between:.3f} mm3)')
            drv.export(OUT / f'obiwan_{pl.DRIVER.name}_pair.driver.stl')
            continue
        if key == 'lm_um_t':
            from lx521_baffle.obiwan import lm_um_t
            mesh = lm_um_t.lm_um_t_mesh(released_installed('h2c_obiwan_lm_keyed_top'), released_um_installed(), tessellate,
                                        released_installed('h2c_obiwan_core_lm_carrier_no_floor_stand'))
            mesh.export(OUT / f'{name}.stl')
            drv = tessellate(driver_at(*center)); drv.export(OUT / f'{name}.driver.stl')
            clash = mesh_clash(mesh, drv)
            report['parts'][name] = dict(volume_cm3=round(mesh.volume / 1000, 1), bounds_mm=np.round(mesh.bounds, 2).tolist(),
                                         watertight=bool(mesh.is_watertight), driver_clash_mm3=round(clash, 3), step=None,
                                         method='mesh booleans: keyed LM top + UM + T body (lm_um_t)', **lm_um_t.facts())
            print(name, report['parts'][name], flush=True)
            if clash > 0.01: raise SystemExit(f'{name}: driver envelope intersects the part ({clash:.3f} mm3)')
            continue
        if key == 'um_t_above':
            mesh = pl.um_t_above_mesh(released_um_installed(), tessellate)
            mesh.export(OUT / f'{name}.stl')
            drv = tessellate(driver_at(*center, surface=True)); drv.export(OUT / f'{name}.driver.stl')
            clash = mesh_clash(mesh, drv)
            report['parts'][name] = dict(volume_cm3=round(mesh.volume / 1000, 1), bounds_mm=np.round(mesh.bounds, 2).tolist(),
                                         watertight=bool(mesh.is_watertight), driver_clash_mm3=round(clash, 3), step=None,
                                         method='mesh booleans on the audited UM + fused frame')
            print(name, report['parts'][name], flush=True)
            if clash > 0.01: raise SystemExit(f'{name}: driver envelope intersects the part ({clash:.3f} mm3)')
            continue
        if key == 'um_t':
            mesh = pl.um_t_mesh(released_um_installed(), tessellate, released_installed('h2c_obiwan_core_lm_carrier_no_floor_stand'))
            mesh.export(OUT / f'{name}.stl')
            drv = tessellate(driver_at(*center)); drv.export(OUT / f'{name}.driver.stl')
            clash = mesh_clash(mesh, drv)
            report['parts'][name] = dict(volume_cm3=round(mesh.volume / 1000, 1), bounds_mm=np.round(mesh.bounds, 2).tolist(),
                                         watertight=bool(mesh.is_watertight), driver_clash_mm3=round(clash, 3), step=None,
                                         method='mesh booleans on the audited UM (BREP booleans exceed 40 GB)')
            print(name, report['parts'][name], flush=True)
            if clash > 0.01: raise SystemExit(f'{name}: driver envelope intersects the part ({clash:.3f} mm3)')
            continue
        part = pl.crescent()
        export_step(part, str(OUT / f'{name}.step'))
        tessellate(part).export(OUT / f'{name}.stl')
        drv = driver_at(*center, surface=True); export_stl(drv, str(OUT / f'{name}.driver.stl'))
        clash = (part & drv).volume
        m = trimesh.load(OUT / f'{name}.stl')
        report['parts'][name] = dict(volume_cm3=round(part.volume / 1000, 1), bounds_mm=np.round(m.bounds, 2).tolist(),
                                     watertight=bool(m.is_watertight), driver_clash_mm3=round(clash, 3))
        print(name, report['parts'][name], flush=True)
        if clash > 0.01: raise SystemExit(f'{name}: driver envelope intersects the part ({clash:.3f} mm3)')
    facts_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
