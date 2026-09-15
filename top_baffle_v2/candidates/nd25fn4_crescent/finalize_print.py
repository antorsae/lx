"""Add measured pauses, reslice, audit and publish the native print projects."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from prepare_print import ROOT,HERE,WORK,sha

SKILL=Path('/Users/antor/.codex/plugins/cache/text-to-cad/cad/0.5.1/skills/gcode/scripts/gcode_tool.py')
KEYS=['body','accessories','V4_flat_left_UPPER','V4_flat_right_UPPER','V4_graded_left_UPPER','V4_graded_right_UPPER']


def refresh_index():
    """Bind the published jobs to current passing evidence and preparation code."""
    path=HERE/'print/manifest.json';manifest=json.loads(path.read_text())
    for row in manifest['projects']:
        audit_path=ROOT/row['audit'];audit=json.loads(audit_path.read_text())
        static_path=ROOT/row['static_audit'];static=json.loads(static_path.read_text())
        assert audit['status']=='pass' and static['ok'] and not static['errors']
        assert row['sha256']==audit['project_sha256']==sha(ROOT/row['path'])
        row.update(audit_sha256=sha(audit_path),static_audit_sha256=sha(static_path))
        for part in audit['geometry']:
            source=ROOT/part['source']
            assert sha(source)==part['sha256']
            sidecar=source.with_suffix('.print.json')
            if source.parent==HERE/'print/geometry' and sidecar.exists():
                data=json.loads(sidecar.read_text())
                data.update(qualification='sliced static checks passed; physical print not performed',
                            sliced_project=row['path'],sliced_project_sha256=row['sha256'])
                sidecar.write_text(json.dumps(data,indent=2)+'\n')
    scripts=[HERE/n for n in ['prepare_print.py','prepare_magnet_loading.py','print_magnets.py',
             'slice_print.py','audit_print.py','finalize_print.py','print_views.py','verify_print.py','cap_support_check.py']]
    scripts += [ROOT/'scripts/audit_gui_slice.py',ROOT/'scripts/gcode_analysis.py',
                ROOT/'scripts/captive_wall_audit.py',ROOT/'scripts/audit_changeover.py',
                ROOT/'src/lx521_baffle/print_policy.py',ROOT/'print_policy.json']
    manifest.update(slicer='Bambu Studio 02.07.01.62',
                    preparation_sha256=sha(ROOT/manifest['preparation']),
                    source_scripts={str(p.relative_to(ROOT)):sha(p) for p in scripts})
    path.write_text(json.dumps(manifest,indent=2)+'\n')


def add_pauses(project,records):
    groups={}
    for r in records:
        assert r['magnet_below_resume_plane'] and r['opening_clear_through_prior_layers'],r
        groups.setdefault(r['pause_before_z_mm'],[]).append(f"{r['name']} D{r['diameter_mm']:g}x{r['depth_mm']:g}")
    root=ET.Element('custom_gcodes_per_layer');plate=ET.SubElement(root,'plate');ET.SubElement(plate,'plate_info',id='1')
    for z,names in sorted(groups.items()):
        program=(f'; ND25FN_MAGNET_INSERTION\n; Insert {len(names)} magnet(s): '+', '.join(names)+
                 f'\nG90\nM400\nG1 Z250 F1200\nM400\nM400 U1\nG1 Z{z:.2f} F1200\nM400')
        ET.SubElement(plate,'layer',top_z=f'{z:.2f}',type='4',extruder='1',color='',extra=program,gcode=program)
    ET.SubElement(plate,'mode',value='SingleExtruder')
    with zipfile.ZipFile(project) as z:data={n:z.read(n) for n in z.namelist()}
    data['Metadata/custom_gcode_per_layer.xml']=ET.tostring(root,encoding='utf-8',xml_declaration=True)
    temp=project.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for name,payload in data.items():z.writestr(name,payload)
    temp.replace(project)


def main():
    prep=json.loads((WORK/'preparation.json').read_text());records=[]
    keys=sys.argv[1:] or KEYS
    for key in keys:
        row=prep[key];prepared=ROOT/row['path'];directory=prepared.parent
        discovery=directory/'magnet_discovery.json'
        if discovery.exists():add_pauses(prepared,json.loads(discovery.read_text()))
        row['sha256']=sha(prepared)
        (WORK/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
        command=json.loads((directory/'command.json').read_text())
        command[command.index('--export-3mf')+1]='ready.gcode.3mf'
        (directory/'final_command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (directory/'final_slice.log').open('w') as f:
            subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
        # Native export writes literal newlines inside XML attributes. Encode
        # them so a later GUI reload retains the complete insertion program.
        sys.path.insert(0, str(ROOT/'scripts'))
        from artifact_emit import _encode_ready_project_custom_gcode_newlines
        if discovery.exists():
            _encode_ready_project_custom_gcode_newlines(directory/'ready.gcode.3mf')
        print(key,'final slice completed',flush=True)
        with (directory/'audit.log').open('w') as f:
            subprocess.run([sys.executable,str(HERE/'audit_print.py'),key],stdout=f,stderr=subprocess.STDOUT,check=True)
        with (directory/'gcode_static.json').open('w') as f:
            subprocess.run([sys.executable,str(SKILL),'validate','--gcode',str(directory/'plate_1.gcode'),
                            '--profile',str(WORK/'profiles/validation_wrapper.json'),'--json'],stdout=f,check=True)
        print(key,'geometry/material/pauses/static checks passed',flush=True)
        ready=directory/'ready.gcode.3mf';audit=json.loads((directory/'audit.json').read_text())
        assert audit['project_sha256']==sha(ready)
        name={'body':'01_UM_Crescent_V4','accessories':'02_Caps_TWO_Retainers_TWO'}.get(key,key)
        target=HERE/'print'/f'{name}_06HF_PETG_GF_PLA.gcode.3mf'
        target.write_bytes(ready.read_bytes())
        records.append(dict(key=key,path=str(target.relative_to(ROOT)),sha256=sha(target),
                            audit=str((directory/'audit.json').relative_to(ROOT)),audit_sha256=sha(directory/'audit.json'),
                            static_audit=str((directory/'gcode_static.json').relative_to(ROOT)),
                            static_audit_sha256=sha(directory/'gcode_static.json'),
                            seconds=audit['slicer_result']['total_predication'],
                            filaments=audit['slicer_result']['filaments'],pauses_mm=audit.get('magnet_pauses_mm',[])))
        manifest=HERE/'print/manifest.json'
        old=json.loads(manifest.read_text()) if manifest.exists() else {'projects':[]}
        old['projects']=[r for r in old['projects'] if r['key']!=key]+[records[-1]]
        old.update(status='sliced_static_checks_passed',physical_print_test='not performed',
                   printer='Bambu Lab P2S',nozzle='0.6 mm High Flow hardened steel',
                   preparation=str((WORK/'preparation.json').relative_to(ROOT)),
                   source_scripts={str(p.relative_to(ROOT)):sha(p) for p in [HERE/n for n in ['prepare_print.py','prepare_magnet_loading.py','print_magnets.py','audit_print.py','finalize_print.py']]})
        manifest.write_text(json.dumps(old,indent=2)+'\n')
    refresh_index()
    print('Published',len(records),'native sliced projects',flush=True)


if __name__=='__main__':main()
