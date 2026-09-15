"""Slice a cropped body/wing D6 pair without changing their local pocket geometry."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import trimesh
import manifold3d as md

from prepare_print import HERE, ROOT, WORK, settings, write_project, slice_command, sha
from print_magnets import magnet_geometry, discover
from mesh_ops import solid, to_trimesh, preserve_void_winding
from finalize_print import add_pauses, SKILL
sys.path.insert(0, str(ROOT/'scripts'))
from artifact_emit import _encode_ready_project_custom_gcode_newlines
from audit_print import geometry
from audit_gui_slice import material_extrusions
from captive_wall_audit import audit_captive_walls
from lx521_baffle.print_policy import policy_sha256, role_settings


def main():
    output=HERE/'print/qualification';output.mkdir(exist_ok=True)
    directory=WORK/'D6_coupon';directory.mkdir(exist_ok=True)
    groups=[];cropped=[]
    sources=[('01_D6_body_coupon.stl', HERE/'print/geometry/01_UM_Crescent_V4_PRINT.stl', 'crescent_body', [60,90,0]),
             ('D6_wing_coupon.stl', HERE/'print/geometry/V4_flat_left_UPPER_PRINT.stl', 'crescent_wing', [120,90,0])]
    for name,source,role,position in sources:
        authority=json.loads(source.with_suffix('.print.json').read_text())
        specs=magnet_geometry(source,authority,[0,0,0])
        site=next(s for s in specs if '168deg' in s['name'])
        center=np.asarray(site['center_bed_mm'])
        original=trimesh.load_mesh(source,process=True)
        box=md.Manifold.cube((26,26,60)).translate((center[0]-13,center[1]-13,-.001))
        crop=to_trimesh(solid(original)^box)
        assert crop.is_watertight
        voids=[s for s in crop.split(only_watertight=False) if s.volume<0]
        assert len(voids)==1
        source_void=min([s for s in original.split(only_watertight=False) if s.volume<0],
                        key=lambda s:np.linalg.norm(s.center_mass-center))
        assert abs(voids[0].volume-source_void.volume)<.001
        shift=-crop.bounds[0]
        crop.apply_translation(shift)
        target=output/name;preserve_void_winding(crop).export(target)
        matrix=np.asarray(authority['source_to_stl_matrix']);matrix[:3,3]+=shift
        authority.update(stl=name,stl_sha256=sha(target),source_to_stl_matrix=matrix.tolist(),
                         coupon_source=str(source.relative_to(ROOT)),coupon_source_sha256=sha(source),
                         coupon_note='Exact local body/wing geometry and pocket angle, cropped and translated to the bed. Validate this coupon slice independently; physical result pending.')
        target.with_suffix('.print.json').write_text(json.dumps(authority,indent=2)+'\n')
        cavity=voids[0].copy();cavity.invert();cavity.apply_translation(shift)
        blocker=directory/(target.stem+'_blocker.stl');cavity.export(blocker)
        overrides=role_settings(role)
        groups.append([(target,'normal_part',overrides,position),(blocker,'support_blocker',{},position)])
        cropped.append((target,authority,position))
    process=settings();process.update(support_critical_regions_only='0',support_remove_small_overhang='0')
    prepared=directory/'D6_coupon.3mf'
    prep=write_project(prepared,groups,process)
    command=slice_command(prepared)
    with (directory/'discovery.log').open('w') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    rows=[]
    for target,authority,position in cropped:
        rows.extend(discover(target,authority,position,directory/'plate_1.gcode'))
    assert len(rows)==2
    (directory/'magnet_discovery.json').write_text(json.dumps(rows,indent=2)+'\n')
    add_pauses(prepared,rows)
    command[slice_command(prepared).index('--export-3mf')+1]='ready.gcode.3mf'
    with (directory/'final.log').open('w') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    native=directory/'ready.gcode.3mf';_encode_ready_project_custom_gcode_newlines(native)
    with zipfile.ZipFile(native) as archive:
        gcode=archive.read('Metadata/plate_1.gcode')
    (directory/'plate_1.gcode').write_bytes(gcode)
    toolpath=directory/'plate_1.gcode'
    supported='; FEATURE: Support' in gcode.decode()
    report=dict(status='pass',physical_qualification='pending',policy_sha256=policy_sha256(),
                geometry=geometry(native,prep),materials=material_extrusions(toolpath,supported),
                pauses=rows, walls=[])
    for target,authority,position in cropped:
        specs=[{k:v for k,v in s.items() if not k.startswith('_')} for s in magnet_geometry(target,authority,position)]
        report['walls'].append(audit_captive_walls(target,position,specs,toolpath))
    with (directory/'static.json').open('w') as stream:
        subprocess.run([sys.executable,str(SKILL),'validate','--gcode',str(toolpath),
                        '--profile',str(WORK/'profiles/validation_wrapper.json'),'--json'],stdout=stream,check=True)
    static=json.loads((directory/'static.json').read_text());assert static['ok'] and not static['errors']
    # Verify every requested pause occurs before its layer's first extrusion.
    lines=gcode.decode().splitlines()
    expected=sorted(set(s['pause_before_z_mm'] for s in rows));actual=[]
    for i,line in enumerate(lines):
        if line!='; ND25FN_MAGNET_INSERTION':continue
        block=lines[i:i+11];assert 'G1 Z250 F1200' in block and 'M400 U1' in block
        actual.append(float(next(s for s in reversed(block) if s.startswith('G1 Z')).split()[1][1:]))
    assert actual==expected
    target=output/'D6_body_wing_fit_06HF_PETG_GF_PLA.gcode.3mf';target.write_bytes(native.read_bytes())
    report.update(project=str(target.relative_to(ROOT)),project_sha256=sha(target),
                  gcode_sha256=sha(toolpath),source_script_sha256=sha(__file__),static_audit=str((directory/'static.json').relative_to(ROOT)))
    (output/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'README.md').write_text('''# D6 body / wing test pair

Print `D6_body_wing_fit_06HF_PETG_GF_PLA.gcode.3mf` with the same 0.6 mm High Flow nozzle and materials as the body. This contains one cropped body station (100% infill) and its corresponding wing station (10%). Both pockets, side skins and inclined loading directions come from the actual print STLs. Cropping and bed translation make this a separate print; its toolpaths and pauses are checked independently.

Use two Ø6×3 mm N45 discs. At each embedded pause, check that the retaining wall is continuous, seat the indicated disc fully, and check the pole orientation against its mate before continuing. After cooling, confirm the roof is closed, the wall and resumed layers are bonded, the discs cannot escape, and the two original mating faces fit without rocking. Record filament lot, drying, printer/nozzle and observations in `physical_result.json` alongside this file. A split skin, loose disc, delamination or blocked insertion is a failure. Do not treat the digital pass as a measured holding force.

No printer operation has been initiated. Physical qualification remains pending your print.
''')
    print('D6 coupon sliced and audited:',target,flush=True)


if __name__=='__main__':main()
