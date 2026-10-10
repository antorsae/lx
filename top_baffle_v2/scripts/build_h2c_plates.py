#!/usr/bin/env python3
"""Combine audited H2C shelf parts into one multi-plate project per product and stand state.

Each plate holds several parts, nested on their actual bed footprints (not bounding boxes) inside the
two-nozzle region with a prime-tower reserve. Every part keeps its own audited geometry, support blockers,
seam paint and per-object process settings. Magnet pauses are recomputed per plate from the combined
slice, and each plate is audited with the single-part checks (deposition, pauses, magnet closure,
support vs duct contracts, static G-code validation). No printer calls.
"""
from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'candidates/nd25fn4_crescent')]
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve
from lx521_baffle.io import sha256_file
from lx521_baffle.h2c import shelf
from lx521_baffle.h2c.printing import job_settings, job_process, write_json, policy
from prepare_print import CORE, BASE, metadata
from build_h2c_release import seam_mask, slice_command, WORK
from audit_h2c_print import deposition, regular_specs, pause_audit, support_audit, static_validation
from print_magnets import discover_specs

LANE = 'petg_gf_pla'
PLATE_GAP = 1 / 5                      # Bambu's logical gap between plates in the scene
BED = (330.0, 320.0)
REGION = (27.0, 2.0, 323.0, 318.0)    # two-nozzle region (x 25..325) minus 2 mm edge margins
CLEAR = 7                              # 5 mm brim + 2 mm gap, mm
PER_OBJECT = {'enable_support', 'sparse_infill_density', 'sparse_infill_pattern', 'support_threshold_angle', 'support_style',
              'wall_generator', 'detect_thin_wall', 'wall_sequence', 'precise_outer_wall', 'brim_width'}
IGNORED = {'print_settings_id', 'wipe_tower_x', 'wipe_tower_y'}

UM = 'h2c_obiwan_core_2_of_2_um_carrier'
LID = 'h2c_obiwan_addon_nl8_service_lid'
ND = 'h2c_dayton_nd25fn4_obiwan_tweeters_{hand}_{part}'
NDS = 'h2c_dayton_nd25fn4_stock_tweeters_{hand}_{part}'


DIPOLES = (('grs_pt2522_4', 'PT2522-4'), ('mundorf_amt17d2_1', 'AMT17D2.1'))


def obiwan_plan(state):
    floor = state == 'floor_stand'
    return dict(product='obiwan', title=f"Obiwan, {state.replace('_', ' ')}", plates=[
        *([('NL8 service lid (every option)', [LID])] if floor else []),
        ('A-C, E · LM carrier (shared by A, B, C and the dipole T above)', [f'h2c_obiwan_core_lm_carrier_{state}']),
        ('A-C · UM carrier', [UM]),
        ('A · ND25FW-4 · crescent', ['h2c_obiwan_addon_tweeter_crescent']),
        ('B · BMR coaxial · crescent', ['h2c_obiwan_bmr_crescent_TEBM35C10-4']),
        ('C · BMR opposed · crescent', ['h2c_obiwan_bmr_crescent_opposed_TEBM35C10-4']),
        *[(f'Wings A-C · {slug} · {side}', [f'h2c_obiwan_wing_{slug}_{side}']) for slug in ('flat', 'graded') for side in ('left', 'right')],
        ('D, F · keyed LM bottom (also A-C, E with the keyed LM top instead of the LM carrier)', [f'h2c_obiwan_lm_keyed_bottom_{state}']),
        ('A-C, E · keyed LM top (with the keyed LM bottom, instead of the LM carrier)', ['h2c_obiwan_lm_keyed_top']),
        *[(f'D · ND25FN-4 tweeters {hand} · vase top + caps',
           [ND.format(hand=hand, part='top'), ND.format(hand=hand, part='cap_front_tweeter'),
            ND.format(hand=hand, part='cap_rear_tweeter')]) for hand in ('right', 'left')],
        *[(f'D · ND25FN-4 tweeters {hand} · {slug} wings left + right',
           [ND.format(hand=hand, part=f'wing_{slug}_left'), ND.format(hand=hand, part=f'wing_{slug}_right')])
          for hand in ('right', 'left') for slug in ('flat', 'graded')],
        *[(f'E · {label} · T above · {o}', [f'h2c_{fam}_obiwan_t_above_{o}'])
          for fam, label in DIPOLES for o in ('upright', 'rotated')],
        ('F · PT2522-4 · T below · upright · keyed LM top', ['h2c_grs_pt2522_4_obiwan_t_below_upright_lm_top']),
        ('F · PT2522-4 · T below · upright · T + UM', ['h2c_grs_pt2522_4_obiwan_t_below_upright_um_t']),
        ('F · PT2522-4 · T below · rotated', ['h2c_grs_pt2522_4_obiwan_t_below_rotated']),
        *[(f'F · AMT17D2.1 · T below · {o}', [f'h2c_mundorf_amt17d2_1_obiwan_t_below_{o}']) for o in ('upright', 'rotated')],
    ])


def proud_plan(family, state):
    lm = ([(f'LM (shared by every option)', [f'h2c_{family}_lm_no_floor_stand'])] if state == 'no_floor_stand' else
          [('LM lower with floor stand (shared by every option)', [f'h2c_{family}_lm_lower_floor_stand']),
           ('LM upper (shared by every option)', [f'h2c_{family}_lm_upper_floor_stand'])])
    plates = [*lm, ('A · ND25FW-4 · upper module', [f'h2c_{family}_upper']),
              ('B · BMR · upper module', [f'h2c_{family}_upper_bmr'])]
    if family == 'stock':
        plates += [(f'C · ND25FN-4 tweeters {hand} · vase + caps',
                    [NDS.format(hand=hand, part='upper'), NDS.format(hand=hand, part='cap_front_tweeter'),
                     NDS.format(hand=hand, part='cap_rear_tweeter')]) for hand in ('right', 'left')]
    plates += [('Wings for A · A shoulders (4 pieces)', [f'h2c_{family}_0{n}_A_shoulder_{p}' for n, p in
                ((5, 'bottom_left_1_of_4'), (6, 'top_left_2_of_4'), (7, 'bottom_right_3_of_4'), (8, 'top_right_4_of_4'))]),
               ('Wings for A · B1 wings left + right', [f'h2c_{family}_09_B1_wing_left_1_of_2', f'h2c_{family}_10_B1_wing_right_2_of_2'])]
    return dict(product=family, title=f"{family.title()}, {state.replace('_', ' ')}", plates=plates)


FILES = {f'{p}_{s}': (obiwan_plan(s) if p == 'obiwan' else proud_plan(p, s))
         for p in ('obiwan', 'stock', 'slim') for s in ('floor_stand', 'no_floor_stand')}
for _key, _spec in FILES.items():
    # Owner rule: every part appears on exactly one plate of its file (options share plates, parts are never duplicated).
    _parts = [n for _, names in _spec['plates'] for n in names]
    _dupes = sorted({n for n in _parts if _parts.count(n) > 1})
    if _dupes: raise ValueError(('Part on more than one plate', _key, _dupes))
MANIFEST = shelf.SHELF / 'plates.json'


def rel(p): return str(Path(p).relative_to(ROOT))


def bundle():
    d = WORK / 'profiles' / LANE
    return [json.loads((d / 'machine.json').read_text()), json.loads((d / 'process.json').read_text()),
            [json.loads((d / 'model.json').read_text()), json.loads((d / 'interface.json').read_text())]]


def rotation(deg):
    r = np.eye(4); c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    r[:2, :2] = [[c, -s], [s, c]]; return r


def footprint(triangles_xy):
    """Bed-contact raster (1 mm) of posed triangles, dilated by brim + gap; returns (mask, origin_xy)."""
    lo = triangles_xy.reshape(-1, 2).min(0) - CLEAR - 1
    size = np.ceil(triangles_xy.reshape(-1, 2).max(0) - lo + CLEAR + 2).astype(int)
    img = Image.new('1', tuple(size), 0); draw = ImageDraw.Draw(img)
    for t in triangles_xy - lo:
        draw.polygon([tuple(p) for p in t], fill=1)
    return binary_dilation(np.array(img, bool).T, iterations=CLEAR), lo


def first_fit(occ, mask):
    if mask.shape[0] > occ.shape[0] or mask.shape[1] > occ.shape[1]: return None
    hit = fftconvolve(occ.astype(float), mask[::-1, ::-1].astype(float), mode='valid') > 0.5
    free = np.argwhere(~hit)
    if not len(free): return None
    return tuple(free[np.lexsort((free[:, 0], free[:, 1]))[0]])


def nest(meshes):
    """Largest first, 15-degree steps, lowest then leftmost. Returns per-part 4x4 bed placements and the tower corner."""
    W, H = int(REGION[2] - REGION[0]), int(REGION[3] - REGION[1])
    occ = np.zeros((W, H), bool); placements = {}
    tri = {k: m.triangles[:, :, :2] for k, m in meshes.items()}
    for key in sorted(meshes, key=lambda k: -np.prod(meshes[k].extents[:2])):
        best = None
        for deg in range(0, 360, 15):
            r = rotation(deg)[:2, :2]
            mask, lo = footprint(tri[key] @ r.T)
            pos = first_fit(occ, mask)
            if pos is None: continue
            reach = pos[1] + mask.shape[1]
            if best is None or reach < best[0]: best = (reach, deg, pos, mask, lo)
        if best is None: raise ValueError(('Part does not fit the plate', key))
        _, deg, (px, py), mask, lo = best
        occ[px:px + mask.shape[0], py:py + mask.shape[1]] |= mask
        place = rotation(deg); place[:2, 3] = np.array([REGION[0] + px, REGION[1] + py]) - lo
        placements[key] = place
    cfg = policy()['prime_tower_reserve_mm']
    t = first_fit(occ, np.ones((cfg['side'], cfg['side']), bool))
    if t is None: raise ValueError('No prime tower space')
    return placements, (REGION[0] + t[0] + cfg['origin_inset'], REGION[1] + t[1] + cfg['origin_inset'])


def plate_origin(index, count):
    cols = math.ceil(math.sqrt(count) - 1e-9)
    return np.array([(index % cols) * BED[0] * (1 + PLATE_GAP), -(index // cols) * BED[1] * (1 + PLATE_GAP), 0.0])


def transform_attr(m):
    """3MF row-vector 3x4 transform of a column-vector 4x4 matrix."""
    return ' '.join(f'{v:.9g}' for v in [*m[:3, :3].T.ravel(), *m[:3, 3]])


def write_plates(path, plates, process, pauses):
    """plates: [(name, [(object_name, groups, bed_placement, object_settings)], tower)]; pauses: {plate: [(z, names)]}."""
    root = ET.Element(f'{{{CORE}}}model', unit='millimeter')
    ET.SubElement(root, f'{{{CORE}}}metadata', name='Application').text = 'BambuStudio-02.08.04.61'
    ET.SubElement(root, f'{{{CORE}}}metadata', name='BambuStudio:3mfVersion').text = '1'
    ET.SubElement(root, f'{{{CORE}}}metadata', name='Title').text = path.stem
    resources = ET.SubElement(root, f'{{{CORE}}}resources'); build = ET.SubElement(root, f'{{{CORE}}}build')
    config = ET.Element('config'); meshes = []; next_id = 1
    for index, (plate_name, objects, _) in enumerate(plates):
        origin = plate_origin(index, len(plates))
        plate = ET.Element('plate')
        for key, value in dict(plater_id=index + 1, plater_name=plate_name, locked='true', filament_map_mode='Manual',
                               filament_maps='1 2', filament_volume_maps='1 1', gcode_file='').items():
            metadata(plate, key, value)
        for object_name, group, place, settings in objects:
            children = []
            for source, subtype, overrides, _ in group:
                overrides = dict(overrides); seam = overrides.pop('__paint_seam__', None)
                mesh = trimesh.load_mesh(source, process=True)
                if seam is not None and len(seam) != len(mesh.faces): raise ValueError(('Seam mask size', source))
                oid = next_id; next_id += 1; meshes.append((oid, mesh, seam)); children.append((oid, source, subtype, overrides, len(mesh.faces)))
            parent = next_id; next_id += 1
            obj = ET.SubElement(resources, f'{{{CORE}}}object', id=str(parent), type='model')
            comps = ET.SubElement(obj, f'{{{CORE}}}components')
            for oid, *_ in children:
                ET.SubElement(comps, f'{{{CORE}}}component', objectid=str(oid), transform='1 0 0 0 1 0 0 0 1 0 0 0')
            world = place.copy(); world[:3, 3] += origin
            ET.SubElement(build, f'{{{CORE}}}item', objectid=str(parent), printable='1', transform=transform_attr(world))
            oc = ET.SubElement(config, 'object', id=str(parent))
            metadata(oc, 'name', object_name)
            for k, v in settings.items(): metadata(oc, k, v)
            for oid, source, subtype, overrides, faces in children:
                part = ET.SubElement(oc, 'part', id=str(oid), subtype=subtype)
                metadata(part, 'name', Path(source).stem); metadata(part, 'source_file', Path(source).name)
                metadata(part, 'matrix', '1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1')
                if subtype == 'normal_part': metadata(part, 'extruder', '1')
                for k, v in overrides.items(): metadata(part, k, v)
                ET.SubElement(part, 'mesh_stat', face_count=str(faces), edges_fixed='0', degenerate_facets='0',
                              facets_removed='0', facets_reversed='0', backwards_edges='0')
            mi = ET.SubElement(plate, 'model_instance')
            metadata(mi, 'object_id', parent); metadata(mi, 'instance_id', 0); metadata(mi, 'identify_id', parent * 10)
        config.append(plate)
    ET.SubElement(config, 'assemble')
    shell = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    prefix, suffix = shell.split(b'</resources>')
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        with archive.open('3D/3dmodel.model', 'w') as out:
            out.write(prefix)
            for oid, mesh, seam in meshes:
                out.write(f'<object id="{oid}" type="model"><mesh><vertices>'.encode())
                for s in range(0, len(mesh.vertices), 10000):
                    out.write(''.join(f'<vertex x="{x:.9g}" y="{y:.9g}" z="{z:.9g}"/>' for x, y, z in mesh.vertices[s:s + 10000]).encode())
                out.write(b'</vertices><triangles>')
                for s in range(0, len(mesh.faces), 10000):
                    faces = mesh.faces[s:s + 10000]
                    paint = seam[s:s + 10000] if seam is not None else np.zeros(len(faces), bool)
                    out.write(''.join(f'<triangle v1="{a}" v2="{b}" v3="{c}"' + (' paint_seam="4"/>' if e else '/>')
                                      for (a, b, c), e in zip(faces, paint)).encode())
                out.write(b'</triangles></mesh></object>')
            out.write(b'</resources>' + suffix)
        archive.writestr('Metadata/model_settings.config', ET.tostring(config, encoding='utf-8', xml_declaration=True))
        archive.writestr('Metadata/project_settings.config', json.dumps(process, indent=2) + '\n')
        with zipfile.ZipFile(BASE) as baseline:
            for name in ['[Content_Types].xml', '_rels/.rels']: archive.writestr(name, baseline.read(name))
        if pauses:
            proot = ET.Element('custom_gcodes_per_layer')
            for index, rows in sorted(pauses.items()):
                if not rows: continue
                p = ET.SubElement(proot, 'plate'); ET.SubElement(p, 'plate_info', id=str(index + 1))
                for z, names in rows:
                    program = (f'; LX521_H2C_MAGNET_INSERTION\n; Insert {len(names)}: ' + ', '.join(names) +
                               f'\nG90\nM400\nG1 Z300 F1200\nM400\nM400 U1\nG1 Z{z:.2f} F1200\nM400\n; LX521_H2C_MAGNET_RESUME')
                    ET.SubElement(p, 'layer', top_z=f'{z:.2f}', type='4', extruder='1', color='', extra=program, gcode=program)
                ET.SubElement(p, 'mode', value='MultiExtruder')
            archive.writestr('Metadata/custom_gcode_per_layer.xml', ET.tostring(proot, encoding='utf-8'))


def placed_job(job, place, work):
    """A job view whose STL/authority sit where the plate puts the part (bed frame, zero offset), for the audits."""
    work.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load_mesh(ROOT / job['stl'], process=True); mesh.apply_transform(place)
    stl = work / (Path(job['stl']).stem + '.placed.stl'); mesh.export(stl)
    matrix = place @ np.asarray(job['source_to_stl_matrix'])
    write_json(stl.with_suffix('.print.json'), dict(schema_version=1, part=stl.stem, stl=stl.name, stl_sha256=sha256_file(stl),
                                                    source_to_stl_matrix=matrix.tolist()))
    return dict(job, stl=rel(stl), authority=rel(stl.with_suffix('.print.json')), offset=[0.0, 0.0, 0.0],
                source_to_stl_matrix=matrix.tolist(), work=rel(work))


def plate_inputs(spec, jobs):
    return dict(parts={n: jobs[n]['project_sha256'] for _, names in spec['plates'] for n in names},
                plan=[[name, names] for name, names in spec['plates']],
                code={rel(p): sha256_file(p) for p in (Path(__file__), ROOT / 'print_policy_h2c.json', ROOT / 'src/lx521_baffle/h2c/shelf.py',
                                                       ROOT / 'scripts/build_h2c_release.py', ROOT / 'scripts/audit_h2c_print.py')})


def manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else dict(schema_version=1, files={})


def is_current(entry, inputs):
    return bool(entry) and entry.get('inputs') == inputs and all(
        (ROOT / entry[k]).exists() and sha256_file(ROOT / entry[k]) == entry[k + '_sha256'] for k in ('project', 'sliced_project'))


def build(key, slice_only=False):
    spec = FILES[key]; jobs = {j['name']: j for j in json.loads(shelf.CATALOG.read_text())['jobs'] if j['lane'] == LANE}
    inputs = plate_inputs(spec, jobs)
    if is_current(manifest()['files'].get(key), inputs):
        print('CURRENT', key, flush=True); return manifest()['files'][key]
    for _, names in spec['plates']:
        for n in names:
            if jobs[n]['status'] != 'sliced_audited': raise ValueError(('Part not audited on the shelf', n))
    b = bundle(); work = WORK / 'plates' / key; work.mkdir(parents=True, exist_ok=True)
    base_name = spec['plates'][0][1][0]; base_job = jobs[base_name]
    base = job_settings(b, LANE, base_job['role'], base_name, brim=base_job['brim_mm'], tower=base_job['tower'])
    plates = []; audit_jobs = []
    for index, (plate_name, names) in enumerate(spec['plates']):
        if len(names) == 1:
            # A part alone keeps its audited single-part placement and tower (large LMs need the full-left region).
            job = jobs[names[0]]; place = np.eye(4); place[:3, 3] = job['offset']
            placements, tower = {f'0:{names[0]}': place}, tuple(job['tower'])
        else:
            placements, tower = nest({f'{i}:{n}': trimesh.load_mesh(ROOT / jobs[n]['stl']) for i, n in enumerate(names)})
        objects = []
        for i, n in enumerate(names):
            job = jobs[n]; place = placements[f'{i}:{n}']
            mine = job_settings(b, LANE, job['role'], n, brim=job['brim_mm'], tower=job['tower'])
            diff = {k: v for k, v in mine.items() if base.get(k) != v and k not in IGNORED}
            if set(diff) - PER_OBJECT: raise ValueError(('Plate-level setting differs between parts', n, sorted(set(diff) - PER_OBJECT)))
            group = []
            for s in job['preparation']['sources']:
                overrides = dict(s['overrides'])
                if s['subtype'] == 'normal_part':
                    mask, _ = seam_mask(n, ROOT / s['path'], np.asarray(job['source_to_stl_matrix']))
                    if mask is not None: overrides['__paint_seam__'] = mask
                group.append((ROOT / s['path'], s['subtype'], overrides, [0, 0, 0]))
            label = shelf.location(n)[2]
            objects.append((label, group, place, diff))
            audit_jobs.append((index, placed_job(job, place, work / f'plate_{index + 1}' / f'{i}_{label}')))
        plates.append((plate_name, objects, tower))
        print(f'PLATE {index + 1}: {plate_name}: ' + ', '.join(o[0] for o in objects), flush=True)
    settings = dict(base, wipe_tower_x=[f'{t[0]:.1f}' for *_, t in plates], wipe_tower_y=[f'{t[1]:.1f}' for *_, t in plates])
    process = job_process(b, settings); write_json(work / 'process.json', process)
    project = shelf.SHELF / spec['product'] / f'{key}.{shelf.LANE_TAGS[LANE]}.3mf'
    command = slice_command(project, LANE, work); command[command.index('--slice') + 1] = '0'

    def run(output, pauses):
        write_plates(project, plates, settings, pauses)
        cmd = command.copy(); cmd[cmd.index('--export-3mf') + 1] = output
        with (work / (output + '.log')).open('w') as log: subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        out = work / output; gcodes = {}
        with zipfile.ZipFile(out) as z:
            for index in range(len(plates)):
                g = work / f'plate_{index + 1}.gcode'; g.write_bytes(z.read(f'Metadata/plate_{index + 1}.gcode')); gcodes[index] = g
        return out, gcodes

    print('SLICE discovery', flush=True)
    _, gcodes = run('discovery.gcode.3mf', {})
    pauses = defaultdict(lambda: defaultdict(list)); specs = {}
    for index, pj in audit_jobs:
        if not pj['magnet_count']: continue
        specs[id(pj)] = regular_specs(pj)
        for r in discover_specs(specs[id(pj)], gcodes[index]):
            if not r['magnet_below_resume_plane'] or not r['opening_clear_through_prior_layers']: raise ValueError((pj['name'], r))
            pauses[index][r['pause_before_z_mm']].append(f"{shelf.location(pj['name'])[2]}:{r['name']}")
    plan = {i: sorted((z, names) for z, names in rows.items()) for i, rows in pauses.items()}
    print('SLICE ready', {plates[i][0]: [z for z, _ in rows] for i, rows in plan.items()}, flush=True)
    sliced, gcodes = run('ready.gcode.3mf', plan)
    report = dict(status='pass', file=rel(project), plates=[])
    for index, (plate_name, objects, tower) in enumerate(plates):
        g = gcodes[index]; entry = dict(plate=index + 1, name=plate_name, parts=[o[0] for o in objects], tower_xy_mm=list(tower))
        entry['deposition'] = deposition(g)
        expected = [z for z, _ in plan.get(index, [])]
        entry['magnet_pauses_mm'] = expected
        entry['pause_program'] = pause_audit(g, expected) if expected else 'no magnets'
        entry['parts_audit'] = []
        for i, pj in audit_jobs:
            if i != index: continue
            row = dict(part=shelf.location(pj['name'])[2], object_settings=next(o[3] for o in objects if o[0] == shelf.location(pj['name'])[2]))
            if pj['magnet_count']:
                again = discover_specs(regular_specs(pj), g)
                row['magnet_pauses_mm'] = sorted({r['pause_before_z_mm'] for r in again})
                if not set(row['magnet_pauses_mm']) <= set(expected): raise ValueError(('Magnet closure changed', pj['name']))
            row['supports'] = support_audit(pj, g)
            entry['parts_audit'].append(row)
        pj0 = dict(audit_jobs[[i for i, _ in audit_jobs].index(index)][1], work=rel(work))
        entry['static_validation'] = bool(static_validation(g, pj0).get('ok'))
        report['plates'].append(entry)
        print(f'PASS plate {index + 1}: {plate_name}', flush=True)
    delivered = project.with_name(project.stem + '.gcode.3mf')
    delivered.write_bytes(sliced.read_bytes())
    report.update(project_sha256=sha256_file(project), sliced_project=rel(delivered), sliced_project_sha256=sha256_file(delivered))
    write_json(work / 'audit.json', report)
    entry = dict(title=spec['title'], product=spec['product'], project=rel(project), project_sha256=report['project_sha256'],
                 sliced_project=rel(delivered), sliced_project_sha256=report['sliced_project_sha256'], audit=rel(work / 'audit.json'),
                 status='sliced_audited', inputs=inputs,
                 plates=[dict(plate=e['plate'], name=e['name'], parts=e['parts'], magnet_pauses_mm=e['magnet_pauses_mm']) for e in report['plates']])
    data = manifest(); data['files'][key] = entry; write_json(MANIFEST, data)
    return entry


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('files', nargs='*', default=list(FILES))
    for key in ap.parse_args().files:
        r = build(key); print(json.dumps(dict(file=r['project'], plates=len(r['plates'])), indent=1))


if __name__ == '__main__':
    main()
