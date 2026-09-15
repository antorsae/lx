"""Bind the September 13 GF changeover calibration to before/after evidence."""
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'candidates/nd25fn4_crescent'
WORK=ROOT/'review/purge_update_20260913'
sys.path[:0]=[str(ROOT/'src'),str(HERE),str(ROOT/'scripts')]
from lx521_baffle.print_policy import policy, policy_sha256, validate_material_mapping
from audit_changeover import audit_changeover_text
from prepare_print import sha


def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def payload(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read('Metadata/project_settings.config')),archive.read('Metadata/plate_1.gcode').decode()


def main():
    frozen=json.loads((WORK/'before.json').read_text())
    geometry=[];images=[]
    for rel,digest in frozen.items():
        if not rel.endswith(('.stl','.png')):continue
        assert sha(ROOT/rel)==digest,('Unexpected geometry/image change',rel)
        (geometry if rel.endswith('.stl') else images).append(rel)
    manifest=json.loads((HERE/'print/manifest.json').read_text())
    coupon=json.loads((HERE/'print/qualification/qualification.json').read_text())
    assert coupon['source_script_sha256']==sha(HERE/'build_magnet_coupon.py')
    assert coupon['status']=='pass' and coupon['policy_sha256']==policy_sha256()
    jobs=[dict(path=r['path'],sha256=r['sha256'],audit=r['audit']) for r in manifest['projects']]
    jobs.append(dict(path=coupon['project'],sha256=coupon['project_sha256'],
                     audit=str((HERE/'print/qualification/qualification.json').relative_to(ROOT))))
    allowed={'filament_flush_volumetric_speed','flush_volumes_matrix','flush_volumes_vector','flush_multiplier'}
    rows=[]
    for job in jobs:
        path=ROOT/job['path'];assert sha(path)==job['sha256']
        old,previous=payload(WORK/'before'/job['path']);new,gcode=payload(path)
        changed={k:dict(before=old.get(k),after=v) for k,v in new.items() if old.get(k)!=v}
        assert set(changed)<=allowed and old.keys()==new.keys(),job['path']
        validate_material_mapping(new)
        changeover=audit_changeover_text(gcode)
        assert audit_changeover_text(previous,purge_mm3=280,pla_flow_mm3_s=40)['change_count']==changeover['change_count']
        # No temperature or insertion instruction was changed by the calibration.
        temperatures=lambda text:re.findall(r'^M620\.10 A[01].*? T([\d.]+) P([\d.]+)',text,re.M)
        assert temperatures(previous)==temperatures(gcode),job['path']
        pauses=lambda text:re.findall(r'; ND25FN_MAGNET_INSERTION\n(?:[^\n]*\n){8}',text)
        assert pauses(previous)==pauses(gcode),job['path']
        report=json.loads((ROOT/job['audit']).read_text());assert report['status']=='pass'
        assert report['project_sha256']==sha(path) and report['policy_sha256']==policy_sha256()
        assert report['changeover']==changeover
        rows.append(dict(**job,before_sha256=frozen[job['path']],changed_settings=changed,
            changeover_count=changeover['change_count'],seconds=report['slicer_result']['total_predication'],
            temperature_handoff_requests_unchanged=True,magnet_pause_programs_unchanged=True))
    gui=[]
    for path in sorted((ROOT/'to_print/obiwan/3mf_06hf_petg-gf_pla').glob('*.3mf')):
        rel=str(path.relative_to(ROOT))
        with zipfile.ZipFile(WORK/'before'/rel) as z:a={n:z.read(n) for n in z.namelist()}
        with zipfile.ZipFile(path) as z:b={n:z.read(n) for n in z.namelist()}
        assert a.keys()==b.keys()
        assert all(a[n]==b[n] for n in a if n!='Metadata/project_settings.config')
        settings=json.loads(b['Metadata/project_settings.config']);validate_material_mapping(settings)
        gui.append(dict(path=rel,sha256=sha(path),status='prepared_GUI_requires_slicing'))
    trans=json.loads((WORK/'unchanged_translucent_jobs.json').read_text())
    assert all(sha(ROOT/p)==digest for p,digest in trans.items())
    report=dict(status='pass',authorization='User: do both — 560 mm3 each direction and 12 mm3/s PLA flush for the GF/PLA setup.',
        policy_sha256=policy_sha256(),source_script_sha256=sha(__file__),
        geometry_files_byte_identical=geometry,preview_images_byte_identical=images,
        sliced_GF_jobs=rows,regular_GUI_projects=gui,unchanged_translucent_projects=trans,
        physical_qualification='Revised changeover calibration pending a physical test; no printer job sent or started.')
    result=WORK/'verification.json';write(result,report)
    # The translucent lane recorded these GF files before its creation. Record
    # this later authorized GF revision explicitly instead of erasing that history.
    baseline=json.loads((ROOT/'review/nd25fn4_translucent/preserved_GF_files.json').read_text())
    revision=dict(report=str(result.relative_to(ROOT)),report_sha256=sha(result),files={
        p:dict(before_sha256=v,sha256=sha(ROOT/p)) for p,v in baseline.items() if sha(ROOT/p)!=v})
    write(ROOT/'review/nd25fn4_translucent/authorized_GF_revisions.json',[revision])
    print('PASS:',len(rows),'sliced GF jobs;',sum(r['changeover_count'] for r in rows),'checked swaps;',
          len(gui),'prepared regular GUI projects;',len(trans),'translucent native files byte-identical')


if __name__=='__main__':main()
