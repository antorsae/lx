"""Build two alternate translucent jobs through the existing audited slicer pipeline.

Geometry, materials and support implementation are shared with translucent_print.
This module supplies an isolated workspace, output directory and policy overlay.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import zipfile

import translucent_print as lane
from audit_changeover import audit_changeover_text

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
POLICY=HERE/'translucent_changeover_policy.json'
WORK=ROOT/'review/nd25fn4_translucent_changeover'
OUT=HERE/'print_translucent_changeover'
BASE_CHECK_RECIPE=lane.check_recipe


def check_recipe(process):
    """Check explicit local exceptions, then reuse the complete native recipe gate."""
    config=json.loads(lane.POLICY.read_text())
    baseline=deepcopy(process)
    for key,wanted in config.get('process_overrides',{}).items():
        assert process[key]==wanted,(key,process[key],wanted)
    for key,wanted in config.get('support_overrides',{}).items():
        assert process[key]==wanted,(key,process[key],wanted)
        # The common gate checks shared support defaults. Only these explicitly
        # checked local settings differ; all filament fields remain untouched.
        baseline[key]=lane.policy()['support'][key]
    return BASE_CHECK_RECIPE(baseline)


def configure(prepare=False):
    overlay=json.loads(POLICY.read_text())
    base=HERE/overlay['base_material_policy']
    config=deepcopy(json.loads(base.read_text()))
    config['name']=overlay['name']
    config['flush_volumes_matrix_mm3']=overlay['flush_volumes_matrix_mm3']
    config['minimum_purge_mm3']=min(v for v in overlay['flush_volumes_matrix_mm3'] if v)
    config['flush_source']=overlay['authorization']
    config['interface']['profile_overrides']['filament_flush_volumetric_speed']=[str(overlay['PLA_flush_mm3_s'])]*2
    temperature=str(overlay['plate']['bed_temperature_c'])
    for kind in ['model','interface']:
        config[kind].setdefault('profile_overrides',{}).update(
            eng_plate_temp=[temperature],eng_plate_temp_initial_layer=[temperature])
    for key in ['plate','process_overrides','support_overrides','guide_review']:
        config[key]=deepcopy(overlay[key])
    config['exceptions']=[s for s in config['exceptions'] if not s.startswith('Retain the selected translucent colour purge matrix')]
    config['exceptions'].append('Alternate changeover calibration: 560 mm3 each direction, multiplier 1, explicit 12 mm3/s PLA flush. Printing temperature and printing flow limits retain the translucent recipes.')
    config['exceptions'].append('Engineering Plate with glue: both filament engineering-plate temperatures are 70 C. No raft; 5 mm outer brim. Bottom support gap is zero; the mutual-support guide excludes these translucent filaments, so this remains a custom qualification.')
    lane.WORK=WORK;lane.OUT=OUT;lane.POLICY=WORK/'material_policy.json';lane.KEYS=overlay['projects']
    lane.check_recipe=check_recipe
    if prepare:
        WORK.mkdir(parents=True,exist_ok=True)
        lane.write_json(lane.POLICY,config)
        preservation={lane.rel(p):lane.sha(p) for d in ['print','print_translucent']
                      for p in (HERE/d).rglob('*') if p.is_file()}
        frozen=WORK/'preserved_existing_files.json'
        if frozen.exists():assert json.loads(frozen.read_text())==preservation
        else:lane.write_json(frozen,preservation)
    else:
        assert json.loads(lane.POLICY.read_text())==config,'Resolved policy differs from the current overlay'
    return overlay,base


def preserve_check():
    frozen=json.loads((WORK/'preserved_existing_files.json').read_text())
    assert all((ROOT/p).is_file() and lane.sha(ROOT/p)==digest for p,digest in frozen.items()),'Existing release changed'
    return len(frozen)


def prepare():
    overlay,base=configure(prepare=True)
    lane.prepare()
    prep=json.loads((WORK/'preparation.json').read_text())
    for key in lane.KEYS:
        path=ROOT/prep[key]['path']
        with zipfile.ZipFile(path) as archive:data={n:archive.read(n) for n in archive.namelist()}
        settings=json.loads(data['Metadata/project_settings.config'])
        settings['flush_volumes_vector']=[str(v) for v in overlay['flush_volumes_vector_mm3']]
        settings['flush_multiplier']=[str(overlay['flush_multiplier'])]
        settings.update(overlay['process_overrides'])
        settings.update(overlay['support_overrides'])
        data['Metadata/project_settings.config']=(json.dumps(settings,indent=2)+'\n').encode()
        temporary=path.with_suffix('.tmp')
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
            for name,value in data.items():archive.writestr(name,value)
        temporary.replace(path);prep[key]['sha256']=lane.sha(path)
    prep['authority']['alternate_policy']=dict(path=lane.rel(POLICY),sha256=lane.sha(POLICY),
        base=lane.rel(base),base_sha256=lane.sha(base),script=lane.rel(Path(__file__)),script_sha256=lane.sha(__file__))
    lane.write_json(WORK/'preparation.json',prep)
    print('Dry run prepared:',WORK/'dry_run.json',flush=True)


def changeover_checks():
    overlay,_=configure()
    prep=json.loads((WORK/'preparation.json').read_text());reports={}
    for key in lane.KEYS:
        directory=(ROOT/prep[key]['path']).parent
        report=json.loads((directory/'audit.json').read_text())
        assert report['status']=='pass' and report['project_sha256']==lane.sha(directory/'ready.gcode.3mf')
        process=lane.read_settings(directory/'ready.gcode.3mf')
        check_recipe(process)
        assert process['flush_multiplier']==['1']
        assert process['flush_volumes_vector']==[str(v) for v in overlay['flush_volumes_vector_mm3']]
        checked=audit_changeover_text((directory/'plate_1.gcode').read_text(),
            purge_mm3=overlay['flush_volumes_matrix_mm3'][1],pla_flow_mm3_s=overlay['PLA_flush_mm3_s'])
        checked.update(project_sha256=report['project_sha256'],gcode_sha256=lane.sha(directory/'plate_1.gcode'))
        reports[key]=checked
    return reports


def publish():
    overlay,base=configure();changes=changeover_checks();lane.publish()
    manifest=json.loads((OUT/'manifest.json').read_text())
    for row in manifest['projects']:
        original=ROOT/row['path'];target=original.with_name(original.name.replace('_06HF_','_PURGE560_06HF_'))
        if original!=target:original.replace(target)
        row.update(path=lane.rel(target),changeover=changes[row['key']])
    manifest['source_files'].update({lane.rel(p):lane.sha(p) for p in [Path(__file__),POLICY,base]})
    manifest['alternate_policy']=dict(path=lane.rel(POLICY),sha256=lane.sha(POLICY))
    manifest['plate']=overlay['plate']
    manifest['guide_review']=overlay['guide_review']
    manifest['physical_print_test']='The user printed a preceding translucent version. This Engineering Plate/changeover configuration has not been physically qualified.'
    references=ROOT/overlay['guide_review']['source_record']
    manifest['source_files'][lane.rel(references)]=lane.sha(references)
    manifest['preserved_existing_file_count']=preserve_check()
    lane.write_json(OUT/'manifest.json',manifest)
    docs(manifest)
    verify()


def docs(manifest):
    rows={r['key']:r for r in manifest['projects']}
    table=[]
    for key,title in [('body','V4 fused UM + crescent body'),('accessories','Two caps + two M3 retainers')]:
        r=rows[key];minutes=round(r['seconds']/60)
        table.append(f"| [{title}]({Path(r['path']).name}) | {minutes//60} h {minutes%60:02} min |")
    body=json.loads((ROOT/rows['body']['audit']).read_text())
    caps=json.loads((ROOT/rows['accessories']['audit']).read_text())['cap_ceilings']
    coverage=min(l['inner_ceiling_coverage'] for c in caps['caps'] for l in c['interface_layers'])
    pause=', '.join(f'{z:.2f}' for z in rows['body']['pauses_mm'])
    text='''# V4 — PETG Translucent + PLA support, alternate changeover calibration

Sliced native Bambu Studio files for the **P2S, 0.6 mm High Flow nozzle**, using the **Engineering Plate with glue, 70 °C, a 5 mm outer brim and no raft**. The plate selection is stored in each project; re-open the updated files before printing.
Use **Bambu PETG Translucent in AMS slot 4** and **Bambu PLA Translucent in AMS slot 2**, as previously confirmed. Open as a project and map the two filaments in the Send dialog. Native nozzle-map numbers are not AMS slots.

| File | Estimated time |
|---|---:|
'''+ '\n'.join(table)+f'''

This alternate pair uses **560 mm³ purge in both directions**, multiplier **1**, and an explicit **12 mm³/s PLA flush**. The preceding translucent files used 298 mm³ PETG → PLA and 575 mm³ PLA → PETG; their generated PLA flush already ran at 12 mm³/s. The new calibration increases the PETG → PLA purge and standardizes the return purge. It has not been physically shown to resolve a blockage.

PETG printing: **250 °C first layer / 245 °C later**, flow 0.95, printing limit 16 mm³/s. PLA printing: **220 °C**, flow 0.98, printing limit 12 mm³/s. Both materials keep the Engineering Plate at **70 °C**, including during material changes. The native changeover temperature requests are preserved from the existing translucent recipe.

The [Bambu mutual-support guide](https://wiki.bambulab.com/en/filament-acc/filament/h2d-pla-and-petg-mutual-support) **explicitly excludes PETG Translucent and only covers PLA Basic paired with PETG Basic/HF**. It lists PEI/High Temp plates. This user-selected material/plate combination is a custom setup, not a guide-qualified preset. We retain its material-specific temperatures and cooling instead of importing the guide's 60 °C bed and 230 °C PLA Basic values. Open the P2S door and/or top cover while PLA is in use, and dry each filament according to its own supplier instructions. The [policy](../translucent_changeover_policy.json) records every retained exception.

The body retains Classic outer-first walls, 0.52 mm exterior paths, six walls, **100% UM infill** and **15% gyroid in the tweeter region**. Driver seats, the flush LM interface, cable routing, insert/magnet support blockers and source geometry are retained. Body magnet insertion pauses before **Z{pause} mm**; have four Ø6×3 mm N45 discs ready and fully seat them below the printing plane before resuming.

Support bodies use PETG; **PLA is used for the support interfaces**. Top and bottom Z gaps are zero. Both cap ceilings have **three dense PLA contact layers** at Z8.68, Z8.84 and Z9.00 mm before the PETG ceiling at Z9.16 mm. The measured inner ceiling coverage is at least **{coverage*100:.2f}%**. The two M3 retainers require no generated support. Cap/retainer infill remains 15% gyroid.

The cap ceiling has an approximately **44.6 mm unsupported span** without support. Bridging that distance might work in a test, but sagging could spoil the internal clearance and surface. These production files retain removable supports; no unsupported bridge has been physically qualified. A raft is not needed in this setup: the existing outer brim supplies extra adhesion without raising the whole part on a sacrificial base.

Both jobs pass exact source-mesh/placement comparison, full material-recipe checks, actual model/support/interface material checks, the changeover-volume/flow gate, native-warning checks and supplemental static G-code validation. Body D6 wall continuity and insertion timing pass; all 12 insert bores and both cable routes remain clear of support. No physical qualification or printer operation is claimed.

[Manifest and checks](manifest.json) · [Policy overlay](../translucent_changeover_policy.json) · [Existing translucent wing alternatives](../print_translucent/README.md)

Rebuild from the project root:

```bash
../.venv/bin/python candidates/nd25fn4_crescent/translucent_changeover_print.py prepare
# Inspect review/nd25fn4_translucent_changeover/dry_run.json.
../.venv/bin/python candidates/nd25fn4_crescent/translucent_changeover_print.py slice
../.venv/bin/python candidates/nd25fn4_crescent/translucent_changeover_print.py publish
```

Verify without reslicing:

```bash
../.venv/bin/python candidates/nd25fn4_crescent/translucent_changeover_print.py verify
```
'''
    (OUT/'README.md').write_text(text)


def verify():
    configure();lane.verify();changes=changeover_checks()
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert {r['key'] for r in manifest['projects']}=={'body','accessories'}
    for row in manifest['projects']:assert changes[row['key']]==row['changeover']
    print('PASS: two alternate translucent files;',sum(r['change_count'] for r in changes.values()),
          'native changes checked;',preserve_check(),'existing files preserved',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','slice','audit','publish','verify'])
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='publish':publish()
    elif args.action=='verify':verify()
    else:
        configure()
        prep=json.loads((WORK/'preparation.json').read_text())
        for key in lane.KEYS:
            if args.action=='slice':lane.slice_job(key)
            else:
                path=(ROOT/prep[key]['path']).parent/'magnet_discovery.json'
                lane.audit(key,prep[key],json.loads(path.read_text()) if path.exists() else [])
        changeover_checks()
