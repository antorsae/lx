"""Prepare existing GF/PLA projects for the shared changeover calibration.

Only changeover metadata is changed. Native slicing and geometric/toolpath
qualification run through the existing candidate pipeline after this dry run.
"""
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'candidates/nd25fn4_crescent'
WORK=ROOT/'review/purge_update_20260913'
sys.path[:0]=[str(ROOT/'src'),str(HERE)]
from lx521_baffle.print_policy import normalize_material_mapping, policy, policy_sha256, interface_changeover_overrides
from prepare_print import WORK as PRINT_WORK, BASE, sha, slice_command


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')


def patch_project(path):
    with zipfile.ZipFile(path) as archive:
        members={n:archive.read(n) for n in archive.namelist()}
    assert not any(n.endswith('.gcode') for n in members), 'Do not patch already sliced files'
    old=json.loads(members['Metadata/project_settings.config'])
    new=normalize_material_mapping(json.loads(json.dumps(old)))
    changes={k:dict(before=old.get(k),after=v) for k,v in new.items() if v!=old.get(k)}
    allowed={'flush_volumes_matrix','flush_volumes_vector','flush_multiplier','filament_flush_volumetric_speed'}
    assert set(changes)<=allowed,(path,changes)
    before=sha(path)
    if changes:
        members['Metadata/project_settings.config']=(json.dumps(new,indent=4)+'\n').encode()
        temporary=path.with_suffix('.changeover.tmp')
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
            for name,payload in members.items():archive.writestr(name,payload)
        with zipfile.ZipFile(temporary) as archive:
            assert all(archive.read(n)==b for n,b in members.items())
        temporary.replace(path)
    return dict(path=str(path.relative_to(ROOT)),before_sha256=before,sha256=sha(path),
                changed=changes,geometry_placement_blockers_and_pauses_byte_preserved=True)


def main():
    WORK.mkdir(exist_ok=True)
    gui=[patch_project(p) for p in sorted((ROOT/'to_print/obiwan/3mf_06hf_petg-gf_pla').glob('*.3mf'))]
    profile_path=PRINT_WORK/'profiles/resolved_support_interface_filament.json'
    profile=json.loads(profile_path.read_text());profile.update(interface_changeover_overrides());write(profile_path,profile)
    preparation=PRINT_WORK/'preparation.json'
    prep=json.loads(preparation.read_text());rows=[];commands={}
    for key,row in prep.items():
        if key=='authority':continue
        project=ROOT/row['path'];result=patch_project(project);row['sha256']=result['sha256'];rows.append(result)
        commands[key]=slice_command(project,'ready.gcode.3mf')
        write(project.parent/'command.json',slice_command(project))
    prep['authority'].update(project=str(BASE.relative_to(ROOT)),sha256=sha(BASE),
        script_sha256=sha(HERE/'prepare_print.py'),policy_sha256=policy_sha256(),
        profile_files={str(p.relative_to(ROOT)):sha(p) for p in (PRINT_WORK/'profiles').glob('*.json')},
        purge_mm3_each_direction=policy()['materials']['purge_each_direction_mm3'],
        interface_changeover_overrides=interface_changeover_overrides())
    write(preparation,prep)
    write(WORK/'preparation.json',dict(status='prepared_not_sliced',policy_sha256=policy_sha256(),
        regular_GUI_projects=gui,candidate_projects=rows,commands=commands))
    print(json.dumps(dict(regular_GUI_projects=len(gui),candidate_projects=len(rows),
                         settings=policy()['materials'],commands=commands),indent=2))


if __name__=='__main__':main()
