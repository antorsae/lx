"""One actual V4 body magnet station, for a short surface/retention print."""
import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
import translucent_print as lane
from prepare_print import write_project
from print_magnets import discover
from finalize_print import add_pauses, SKILL
from audit_print import geometry, settings_check
from audit_gui_slice import material_extrusions
from captive_wall_audit import audit_captive_walls
from artifact_emit import _encode_ready_project_custom_gcode_newlines

HERE=lane.HERE;ROOT=lane.ROOT
WORK=lane.WORK/'surface_coupon'
OUT=lane.OUT/'qualification'


def prepare():
    source=HERE/'print/qualification/01_D6_body_coupon.stl'
    sidecar=source.with_suffix('.print.json')
    authority=json.loads(sidecar.read_text())
    assert authority['stl_sha256']==lane.sha(source)
    assert authority['coupon_source_sha256']==lane.sha(ROOT/authority['coupon_source'])
    body_prep=json.loads((lane.WORK/'preparation.json').read_text())['body']
    process=lane.read_settings(ROOT/body_prep['path'])
    process.update(print_settings_id='V4 magnet surface test - PETG Translucent 0.6HF',
                   support_critical_regions_only='0',support_remove_small_overhang='0')
    blocker=lane.BASE/'D6_coupon/01_D6_body_coupon_blocker.stl'
    parts=[[(source,'normal_part',{},[0,0,0]),(blocker,'support_blocker',{},[0,0,0])]]
    row=write_project(WORK/'surface_coupon.3mf',parts,process,offset=[90,90,0])
    row.update(authority=lane.rel(sidecar),authority_sha256=lane.sha(sidecar),
        parent_body_sha256=authority['coupon_source_sha256'],material_policy_sha256=lane.sha(lane.POLICY))
    lane.write_json(WORK/'preparation.json',row)
    lane.write_json(WORK/'dry_run.json',lane.slice_command(ROOT/row['path'],'discovery.gcode.3mf'))
    print(json.dumps(lane.slice_command(ROOT/row['path'],'discovery.gcode.3mf'),indent=2),flush=True)


def run(reslice=True):
    prep=json.loads((WORK/'preparation.json').read_text())
    prepared=ROOT/prep['path'];source=ROOT/prep['sources'][0]['path']
    authority=json.loads((ROOT/prep['authority']).read_text())
    if reslice:
        lane.run_slice(prepared,'discovery.gcode.3mf')
        rows=discover(source,authority,prep['offset'],WORK/'plate_1.gcode')
        assert len(rows)==1 and rows[0]['diameter_mm']==6
        lane.write_json(WORK/'magnet_discovery.json',rows)
        add_pauses(prepared,rows)
        prep['sha256']=lane.sha(prepared);lane.write_json(WORK/'preparation.json',prep)
        lane.run_slice(prepared,'ready.gcode.3mf')
    else:
        assert lane.sha(prepared)==prep['sha256']
        rows=json.loads((WORK/'magnet_discovery.json').read_text())
    native=WORK/'ready.gcode.3mf'
    if reslice:_encode_ready_project_custom_gcode_newlines(native)
    lane.extract(native,WORK)
    gcode=WORK/'plate_1.gcode';text=gcode.read_text()
    process=settings_check(native,prepared)
    lane.check_recipe(process)
    cfg=json.loads(lane.POLICY.read_text())
    for k,v in cfg['body_surface_finish'].items():assert process[k]==v
    supported='; FEATURE: Support' in text
    report=dict(status='pass',physical_qualification='pending user print',
        materials=material_extrusions(gcode,supported),geometry=geometry(native,prep),
        walls=audit_captive_walls(source,prep['offset'],rows,gcode),
        pauses_mm=lane.check_pauses(gcode,rows),magnet_specs=rows,
        supports_generated=supported,material_policy_sha256=lane.sha(lane.POLICY),
        source_script_sha256=lane.sha(__file__),preparation_sha256=lane.sha(WORK/'preparation.json'))
    bed=sorted(set(float(v) for v in re.findall(r'^[ \t]*M(?:140|190) S([\d.]+)',text,re.M)))
    assert 70 in bed and set(bed)<={0.,70.}
    report['actual_bed_commands_c']=bed
    result=json.loads((WORK/'result.json').read_text())
    assert result['return_code']==0 and not result['sliced_plates'][0]['warning_message']
    report['slicer_result']=result['sliced_plates'][0]
    with (WORK/'gcode_static.json').open('w') as stream:
        subprocess.run([sys.executable,str(SKILL),'validate','--gcode',str(gcode),
            '--profile',str(lane.WORK/'profiles/validation_wrapper.json'),'--json'],stdout=stream,check=True)
    static=json.loads((WORK/'gcode_static.json').read_text());assert static['ok'] and not static['errors']
    OUT.mkdir(exist_ok=True)
    target=OUT/'00_Magnet_Surface_Test_06HF_PETG_TRANSLUCENT_PLA_TRANSLUCENT.gcode.3mf'
    shutil.copy2(native,target)
    report.update(project=lane.rel(target),project_sha256=lane.sha(target),gcode_sha256=lane.sha(gcode),
        static_audit=lane.rel(WORK/'gcode_static.json'),static_audit_sha256=lane.sha(WORK/'gcode_static.json'))
    lane.write_json(OUT/'qualification.json',report)
    minutes=round(report['slicer_result']['total_predication']/60)
    pause=rows[0]['pause_before_z_mm']
    (OUT/'README.md').write_text(f'''# V4 body magnet surface test

[Prepared test 3MF]({target.name}) — approximately **{minutes} minutes**, excluding the insertion pause.

This is a 26 × 26 mm crop of one actual body magnet station, with its original curved outside surface, cover thickness, inclined loading pocket and print orientation. It uses the revised body's **Classic / outer-first / 0.52 mm outer-wall** settings and 100% body infill. No design geometry is modified.

Use the P2S **0.6 mm High Flow nozzle**, **PETG Translucent in AMS slot 4** and **PLA Translucent in slot 2**. Both material bed settings are 70 °C. The native job {'includes PETG supports with PLA interfaces' if supported else 'does not need generated supports'}.

Have **one Ø6×3 mm N45 magnet** ready. At the embedded pause before **Z{pause:.2f} mm**, orient and fully seat it below the printing plane, then resume. After cooling, inspect the curved outside face in raking light and by touch: look for the pocket outline, a texture/gloss change or a ridge. Also check that the cover and resumed layers are bonded and the magnet is retained. The cut sides of this test are crop boundaries, not production surfaces.

The revised toolpaths remove the measured exterior bead-width change. This test lets you check the physical result; a crop does not reproduce every thermal effect of the full body. Record filament lot, printer/nozzle and observations in `physical_result.json` alongside this file. Physical surface finish and retention are not yet verified.

[Qualification measurements](qualification.json) bind this file to its geometry, material-role, D6 wall, pause and static G-code checks.
''')
    print('Surface coupon passed:',target,'estimated minutes:',minutes,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','audit']);a=p.parse_args()
    prepare() if a.action=='prepare' else run(reslice=a.action=='run')
